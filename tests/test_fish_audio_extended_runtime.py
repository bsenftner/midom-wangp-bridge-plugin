"""Additive v3 negotiation, exact historical limits, and real long-file delivery."""

import hashlib
import io
import json
import shutil
import struct
import subprocess
import types
import wave
from pathlib import Path

import pytest

from test_fish_audio_drama3 import Runtime, Session, bridge, candidate_for, fish_job, wav_bytes
from test_fish_audio_reviewable_runtime import ReviewablePipeline, v2_job
from test_fish_audio_v2_contract_fixtures import CAPABILITIES, JOBS


PROPOSAL_PATH = Path(__file__).parent / "fixtures/fish_audio_v3_capabilities.json"
PROPOSAL_SHA256 = "451e6870c43a3b751eb15c40429aa73c452737a663c9cbbb5dbe4330fbb093c3"
PROPOSAL = json.loads(PROPOSAL_PATH.read_text())
V3, SAFETY = "fish_drama3_tts_v3", "fish_audio_240s_50mib_v1"


def v3_job(refs=0, *, speakers=None, model="fish_drama3", fmt="wav"):
    job = v2_job(refs, speakers=speakers, model=model, fmt=fmt)
    job["generation"].update(speech_contract_version=V3, artifact_safety_profile_id=SAFETY, duration_seconds=180)
    job["limits"] = {"max_artifact_bytes": 52428800}
    return job


def test_complete_capability_proposal_and_exact_historical_records(bridge):
    module, plugin = bridge
    assert hashlib.sha256(PROPOSAL_PATH.read_bytes()).hexdigest() == PROPOSAL_SHA256
    assert PROPOSAL["fixture_set_id"] == "fish_audio_extended_v3_proposal_v1"
    assert plugin._fish_audio_capabilities() == PROPOSAL["models"]
    for current, old in zip(plugin._fish_audio_capabilities(), CAPABILITIES["models"]):
        current["limits"]["fish_audio_contracts"].pop(V3)
        assert current == old
    assert module.FISH_AUDIO_MAX_DURATION == 120


@pytest.mark.parametrize("model", JOBS["checkpoint_matrix"])
@pytest.mark.parametrize("fmt", ["wav", "mp3"])
@pytest.mark.parametrize("route", JOBS["route_matrix"])
def test_serialized_v3_candidate_and_claim_agree(bridge, model, fmt, route):
    _, plugin = bridge
    job = v3_job(route["reference_count"], speakers=route["speaker_count"], model=model, fmt=fmt)
    candidate = json.loads(json.dumps(candidate_for(plugin, job)))
    claim = json.loads(json.dumps(job))
    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    settings = plugin._validate_job(claim, None)
    metadata = settings["_midom_fish_audio"]
    for key in ("speech_contract_version", "speech_profile_id", "acceptance_policy_id", "artifact_safety_profile_id"):
        assert metadata[key] == claim["generation"][key] == candidate["summary"][key]
    assert settings["duration_seconds"] == metadata["maximum_duration_seconds"] == 180
    assert settings["audio_prompt_type"] == ("", "A", "AB", "ABD")[route["reference_count"]]
    assert settings["_api"] == {"return_side_files": False, "return_audio": True}
    assert metadata["recipe_version"] == "fish_drama3_standard_recipe_v1"
    assert "max_duration_seconds" not in claim["output"]


@pytest.mark.parametrize("version,maximum", [("v1", 120), ("v2", 120), ("v3", 180)])
@pytest.mark.parametrize("duration", [1, 60, 120, 121, 180, 181, 0, None, True, 180.0])
def test_generation_bounds_are_per_contract_for_actual_payloads(bridge, version, maximum, duration):
    _, plugin = bridge
    job = {"v1": fish_job, "v2": v2_job, "v3": v3_job}[version]()
    job["generation"]["duration_seconds"] = duration
    if version == "v1":
        job["output"]["max_duration_seconds"] = duration
    candidate = json.loads(json.dumps(candidate_for(plugin, job)))
    accepted = type(duration) is int and 1 <= duration <= maximum
    assert (plugin._candidate_incompatibility_reason(candidate, None) is None) is accepted
    if accepted:
        assert plugin._validate_audio_job(json.loads(json.dumps(job)), 123)["duration_seconds"] == duration
    else:
        with pytest.raises(ValueError):
            plugin._validate_audio_job(json.loads(json.dumps(job)), 123)


@pytest.mark.parametrize("field,value", [
    ("speech_contract_version", "fish_drama3_tts_v2"),
    ("artifact_safety_profile_id", "fish_audio_180s_50mib_v1"),
    ("artifact_safety_profile_id", None), ("acceptance_policy_id", None),
    ("acceptance_policy_id", "other_policy"), ("speech_profile_id", "other_profile"),
    ("duration_seconds", 181),
])
def test_selection_mismatch_never_falls_back(bridge, field, value):
    _, plugin = bridge
    job = v3_job()
    job["generation"][field] = value
    assert plugin._candidate_incompatibility_reason(candidate_for(plugin, job), None)
    with pytest.raises(ValueError):
        plugin._validate_audio_job(job, 123)


@pytest.mark.parametrize("field,value", [("max_duration_seconds", 180), ("count", 2), ("format", "flac")])
def test_v3_output_declarations_stay_strict(bridge, field, value):
    _, plugin = bridge
    job = v3_job()
    job["output"][field] = value
    with pytest.raises(ValueError):
        plugin._validate_audio_job(job, 123)


@pytest.mark.parametrize("field,value", [("max_artifact_bytes", 52428801), ("max_artifact_duration_seconds", 300), ("max_decoded_mono_samples", 20000000)])
def test_v3_does_not_accept_raw_safety_overrides(bridge, field, value):
    _, plugin = bridge
    job = v3_job()
    job["limits"][field] = value
    with pytest.raises(ValueError):
        plugin._validate_audio_job(job, 123)


def test_v3_file_bytes_remain_limited_to_50_mib(bridge, monkeypatch, tmp_path):
    _, plugin = bridge
    settings = plugin._validate_audio_job(v3_job(), 123)
    path = tmp_path / "oversized.wav"
    with path.open("wb") as target:
        target.truncate(52428801)
    monkeypatch.setattr(plugin, "_fish_audio_bounded_process", lambda *args, **kwargs: pytest.fail("Oversized bytes must fail before probing or decoding"))
    with pytest.raises(ValueError, match="artifact_byte_safety_exceeded"):
        plugin._validate_fish_audio_file(path, settings, native=False)


def test_each_checkpoint_requires_its_own_complete_v3_capability(bridge, monkeypatch):
    _, plugin = bridge
    records = plugin._fish_audio_capabilities()
    records[0]["limits"]["fish_audio_contracts"].pop(V3)
    monkeypatch.setattr(plugin, "_fish_audio_capabilities", lambda: records)
    for model, accepted in ((records[0]["model_id"], False), (records[1]["model_id"], True)):
        job = v3_job(model=model)
        assert (plugin._candidate_incompatibility_reason(candidate_for(plugin, job), None) is None) is accepted
        if accepted:
            plugin._validate_audio_job(job, 123)
        else:
            with pytest.raises(ValueError):
                plugin._validate_audio_job(job, 123)


def test_wav_only_discovery_is_consistent_across_all_versions(bridge):
    _, plugin = bridge
    plugin._audio_output_mime_types = lambda: ["audio/wav"]
    for model in plugin._fish_audio_capabilities():
        assert model["limits"]["output_mime_types"] == ["audio/wav"]
        assert all(record["output_mime_types"] == ["audio/wav"] for record in model["limits"]["fish_audio_contracts"].values())
    job = v3_job(fmt="mp3")
    assert plugin._candidate_incompatibility_reason(candidate_for(plugin, job), None)
    with pytest.raises(ValueError):
        plugin._validate_audio_job(job, 123)


@pytest.mark.parametrize("version,probe,samples,accepted,over_budget", [
    ("v2", "180", 7938000, True, True), ("v2", "182", 8026200, False, True),
    ("v2", "180", 7938001, False, True),
    ("v3", "182", 8026200, True, True), ("v3", "180.25", 7949025, True, False),
    ("v3", "180.250001", 7949025, True, True), ("v3", "180.25", 7949026, True, True),
    ("v3", "240", 10584000, True, True), ("v3", "240.000001", 10584000, False, True),
    ("v3", "240", 10584001, False, True), ("v3", "180", 10584000, True, True),
])
def test_artifact_duration_and_sample_boundaries(bridge, monkeypatch, tmp_path, version, probe, samples, accepted, over_budget):
    _, plugin = bridge
    job = v3_job() if version == "v3" else v2_job()
    settings = plugin._validate_audio_job(job, 123)
    settings["_midom_fish_audio"].update(finish_reason="eos")
    path = tmp_path / "recording.wav"
    path.write_bytes(wav_bytes(0.1))
    result = {"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "44100", "channels": 1,
                           "duration": probe}], "format": {"format_name": "wav", "duration": probe}}
    monkeypatch.setattr(plugin, "_fish_audio_bounded_process", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, json.dumps(result).encode(), b""))
    seen = []

    def decode(path, *, maximum_samples):
        seen.append(maximum_samples)
        return {"decoded_mono_sample_count": samples, "decoded_duration_seconds": samples / 44100, "peak_amplitude": 0.5}

    monkeypatch.setattr(plugin, "_fish_audio_reviewable_decoded_measure", decode)
    if accepted:
        plugin._validate_fish_audio_file(path, settings, native=False)
        assert bool(settings["_midom_fish_audio"]["warnings"]) is over_budget
    else:
        with pytest.raises(ValueError, match="safety_exceeded"):
            plugin._validate_fish_audio_file(path, settings, native=False)
    if seen:
        assert seen == [10584000 if version == "v3" else 7938000]


@pytest.mark.parametrize("maximum", [7938000, 10584000])
def test_decode_receives_an_explicit_profile_bound(bridge, monkeypatch, tmp_path, maximum):
    _, plugin = bridge

    def run(command, **bounds):
        assert bounds == {"max_stdout_bytes": maximum * 4, "timeout_seconds": 90}
        assert not {"-ar", "-ac", "-ss", "-t"}.intersection(command)
        return subprocess.CompletedProcess(command, 0, struct.pack("<f", 0.5),
            b"Output #0, f32le, to pipe:1:\n Audio: pcm_f32le, 44100 Hz, mono, flt, 1411 kb/s\n")

    monkeypatch.setattr(plugin, "_fish_audio_bounded_process", run)
    assert plugin._fish_audio_reviewable_decoded_measure(tmp_path / "x.wav", maximum_samples=maximum)["decoded_mono_sample_count"] == 1


@pytest.mark.parametrize("maximum", [7938000, 10584000])
def test_real_decoder_rejects_first_sample_above_profile_bound(bridge, tmp_path, maximum):
    _, plugin = bridge
    path = tmp_path / "oversized.wav"
    path.write_bytes(wav_bytes((maximum + 1.25) / 44100))
    with wave.open(str(path)) as source:
        assert source.getnframes() == maximum + 1
    with pytest.raises(ValueError, match="artifact_decode_failed"):
        plugin._fish_audio_reviewable_decoded_measure(path, maximum_samples=maximum)


@pytest.mark.parametrize("model", JOBS["checkpoint_matrix"])
@pytest.mark.parametrize("fmt", ["wav", "mp3"])
@pytest.mark.parametrize("seconds,reason", [(182, "eos"), (0.3, "silence"), (0.3, "length"), (0.3, "unknown"), (0.3, "early_stop")])
def test_v3_real_file_collection_upload_completion(bridge, monkeypatch, tmp_path, model, fmt, seconds, reason):
    module, plugin = bridge
    job = v3_job(model=model, fmt=fmt)
    path = tmp_path / "native" / "speech_seed123.wav"
    path.parent.mkdir()
    recording = wav_bytes(seconds)
    session = Session(Runtime(ReviewablePipeline(reason), lambda: path.write_bytes(recording)),
                      ["native/speech_seed123.wav"], artifact_files=[str(path)], root=tmp_path)
    plugin._get_generation_session = lambda: session
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    uploaded = []

    def upload(url, **kwargs):
        payload = kwargs["files"]["file"][1].read()
        assert kwargs["data"]["sha256"] == hashlib.sha256(payload).hexdigest()
        uploaded.append(payload)
        return types.SimpleNamespace(status_code=200, json=lambda: {"artifact_id": 12, "file_id": 34, "artifact_index": 0})

    monkeypatch.setattr(module.requests, "post", upload)
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7")
    plugin._run_job(connection, json.loads(json.dumps(job)))
    assert len(session.submitted) == 1
    assert session.submitted[0]["duration_seconds"] == 180
    assert not any(key.startswith("_midom_") for key in session.submitted[0])
    if reason == "early_stop":
        assert plugin._fish_updates[-1][0] == "fail"
        assert uploaded == []
        return
    assert plugin._fish_updates[-1][0] == "complete", plugin._fish_updates[-1]
    assert len(uploaded) == 1
    fish = plugin._fish_updates[-1][1]["generation_metadata"]["fish_audio"]
    assert fish["speech_contract_version"] == V3 and fish["artifact_safety_profile_id"] == SAFETY
    assert fish["script_coverage"] == "not_verified"
    expected = {"eos": [], "silence": ["speech_stopped_on_silence"], "length": ["generation_duration_limit_reached"],
                "unknown": ["termination_unverified"]}[reason]
    if seconds == 182:
        expected.append("generated_duration_over_budget")
    assert [warning["code"] for warning in fish["warnings"]] == expected
    assert fish["artifacts"][0]["sha256"] == hashlib.sha256(uploaded[0]).hexdigest()
    assert path.read_bytes() == recording
    if fmt == "wav":
        assert uploaded[0] == recording
        with wave.open(io.BytesIO(uploaded[0])) as audio:
            count = audio.getnframes()
    else:
        delivered = tmp_path / "delivered.mp3"
        delivered.write_bytes(uploaded[0])
        decoded = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(delivered), "-f", "f32le", "pipe:1"],
                                 capture_output=True, check=True, timeout=90)
        count = len(decoded.stdout) // 4
    assert count == fish["artifacts"][0]["decoded_mono_sample_count"]
    if seconds == 182:
        assert fish["artifacts"][0]["duration_seconds"] > 180
