"""Exercise implemented v2 paths against the immutable, mirrored contract fixtures."""

import copy
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

from test_fish_audio_drama3 import Pipeline, Response, Runtime, Session, bridge, candidate_for, fish_job, wav_bytes
from test_fish_audio_v2_contract_fixtures import CAPABILITIES, JOBS, MEASUREMENTS, REJECTIONS, depth, json_bytes, mutate


def v2_job(refs=0, *, speakers=None, model="fish_drama3", fmt="wav"):
    job = fish_job(refs, speakers=speakers, model=model, fmt=fmt)
    job["generation"].update(speech_contract_version="fish_drama3_tts_v2", acceptance_policy_id="reviewable_audio_v1",
                             artifact_safety_profile_id="fish_audio_180s_50mib_v1")
    job["output"].pop("max_duration_seconds")
    return job


def settings_for(plugin, **kwargs):
    return plugin._validate_audio_job(v2_job(**kwargs), 123)


class ReviewablePipeline(Pipeline):
    def generate(self, *, input_prompt, **kwargs):
        if self.reason == "unknown":
            self.last_finish_reason = None
        else:
            super().generate(input_prompt=input_prompt, **kwargs)
        return {"x": types.SimpleNamespace(numel=lambda: 100), "audio_sampling_rate": 44100}


def observe(plugin, settings, path, reason="eos", *, rewrite=True):
    runtime = Runtime(ReviewablePipeline(reason), (lambda: path.write_bytes(path.read_bytes())) if rewrite else None)
    session = Session(runtime, [str(path)])
    with plugin._fish_audio_execution(session, settings):
        result = plugin._submit_wangp_job(session, settings, 1, None).result()
    return result


@pytest.mark.parametrize("model", JOBS["checkpoint_matrix"])
@pytest.mark.parametrize("fmt", [item["format"] for item in JOBS["format_matrix"]])
@pytest.mark.parametrize("route", JOBS["route_matrix"])
def test_real_candidate_and_claim_contract_selection(bridge, model, fmt, route):
    _, plugin = bridge
    job = v2_job(route["reference_count"], speakers=route["speaker_count"], model=model, fmt=fmt)
    assert plugin._candidate_incompatibility_reason(candidate_for(plugin, job), None) is None
    settings = plugin._validate_audio_job(job, 123)
    assert settings["_midom_fish_audio"]["speech_contract_version"] == "fish_drama3_tts_v2"
    assert settings["duration_seconds"] == 60
    assert settings["custom_settings"] == {"dialogue_cut_lines": 0, "dialogue_history_seconds": 20}
    assert settings["_midom_fish_audio"]["script_coverage"] == "not_verified"
    assert settings["_api"] == {"return_side_files": False, "return_audio": True}


@pytest.mark.parametrize("case", REJECTIONS["selection_and_claim_cases"], ids=lambda case: case["id"])
def test_runtime_rejects_frozen_bad_selections(bridge, monkeypatch, case):
    _, plugin = bridge
    target = copy.deepcopy({"candidate": JOBS["candidate"], "claim": JOBS["claim"], "capability": CAPABILITIES["models"][0]}[case["target"]])
    mutate(target, case)
    if case["target"] == "capability":
        monkeypatch.setattr(plugin, "_fish_audio_capabilities", lambda: [target])
        with pytest.raises(ValueError):
            plugin._validate_audio_job(copy.deepcopy(JOBS["claim"]), 2401)
    elif case["target"] == "candidate":
        assert plugin._fish_audio_candidate_incompatibility_reason(target["summary"], target["model_id"])
    else:
        with pytest.raises(ValueError):
            plugin._validate_audio_job(target, 2401)


@pytest.mark.parametrize("model", JOBS["checkpoint_matrix"])
@pytest.mark.parametrize("fmt", ["wav", "mp3"])
@pytest.mark.parametrize("reason", ["eos", "silence", "length", "unknown"])
@pytest.mark.parametrize("refs", [0, 3])
def test_complete_claimed_job_preserves_reviewable_audio(bridge, monkeypatch, tmp_path, model, fmt, reason, refs):
    module, plugin = bridge
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    job = v2_job(refs, model=model, fmt=fmt)
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response(wav_bytes()))
    path = tmp_path / "generated_seed123.wav"
    recording = wav_bytes(0.5)
    runtime = Runtime(ReviewablePipeline(reason), lambda: path.write_bytes(recording))
    session = Session(runtime, [str(path)])
    plugin._get_generation_session = lambda: session
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()
    uploaded = []

    def upload(url, **kwargs):
        data = kwargs["files"]["file"][1].read()
        assert hashlib.sha256(data).hexdigest() == kwargs["data"]["sha256"]
        uploaded.append(data)
        return types.SimpleNamespace(status_code=200, json=lambda: {"artifact_id": 12, "file_id": 34, "artifact_index": 0})

    monkeypatch.setattr(module.requests, "post", upload)
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7")
    plugin._run_job(connection, job)
    assert plugin._fish_updates[-1][0] == "complete", plugin._fish_updates[-1]
    metadata = plugin._fish_updates[-1][1]["generation_metadata"]
    fish = metadata["fish_audio"]
    assert fish["finish_reason"] == reason
    expected = {"eos": [], "silence": ["speech_stopped_on_silence"], "length": ["generation_duration_limit_reached"],
                "unknown": ["termination_unverified"]}[reason]
    assert [item["code"] for item in fish["warnings"]] == expected
    assert fish["script_coverage"] == "not_verified"
    assert len(fish["reference_inputs"]) == refs
    assert session.submitted[0]["audio_prompt_type"] == ("ABD" if refs else "")
    assert fish["execution_evidence"] == {
        "job_id": 123, "client_id": session.submitted[0]["client_id"], "session_succeeded": True,
        "cancelled": False, "native_call_returned_output": True, "observed_calls": 1,
    }
    assert fish["artifacts"][0]["sha256"] == hashlib.sha256(uploaded[0]).hexdigest()
    assert fish["artifacts"][0]["duration_seconds"] > 0
    assert depth(metadata) <= 8 and json_bytes(metadata) <= 32768
    if fmt == "wav":
        assert uploaded == [recording]
    else:
        assert uploaded[0] != recording


@pytest.mark.parametrize("reason", ["eos", "silence", "length", "unknown"])
@pytest.mark.parametrize("failure", ["cancel", "stop", "session_failed", "session_errors", "not_returned", "stale", "no_native_audio",
                                     "missing_artifact", "extra_artifact", "artifact_path", "artifact_media_type", "wrong_job", "wrong_client_evidence"])
def test_reviewable_never_masks_execution_or_association_failure(bridge, tmp_path, reason, failure):
    _, plugin = bridge
    settings = settings_for(plugin)
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes(0.1))
    result = observe(plugin, settings, path, reason, rewrite=failure != "stale")
    if failure == "cancel":
        result.cancelled = True
    elif failure == "stop":
        plugin._stop_event.set()
    elif failure == "session_failed":
        result.success = False
    elif failure == "session_errors":
        result.errors = ["decoding failed"]
    elif failure == "not_returned":
        result.generated_files = []
    elif failure == "no_native_audio":
        settings["_midom_fish_audio"]["execution_evidence"]["native_call_returned_output"] = False
    elif failure == "stale":
        result.artifacts[0].client_id = "previous_job_client_id"
    elif failure == "missing_artifact":
        result.artifacts = ()
    elif failure == "extra_artifact":
        result.artifacts += result.artifacts
    elif failure == "artifact_path":
        result.artifacts[0].path = "different.wav"
    elif failure == "artifact_media_type":
        result.artifacts[0].media_type = "video"
    elif failure == "wrong_job":
        settings["_midom_fish_audio"]["execution_evidence"]["job_id"] = 124
    elif failure == "wrong_client_evidence":
        settings["_midom_fish_audio"]["execution_evidence"]["client_id"] = "another_job"
    with pytest.raises(ValueError):
        plugin._validate_fish_audio_completion(settings, [str(path)], result)


@pytest.mark.parametrize("reason", ["early_stop", "invalid", None])
def test_missing_or_malformed_native_diagnostics_are_not_unknown(bridge, tmp_path, reason):
    _, plugin = bridge
    settings = settings_for(plugin)
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes(0.1))
    result = observe(plugin, settings, path, reason)
    with pytest.raises(ValueError):
        plugin._validate_fish_audio_completion(settings, [str(path)], result)


def test_explicit_unknown_requires_empty_parts(bridge, tmp_path):
    _, plugin = bridge
    settings = settings_for(plugin)
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes(0.1))
    result = observe(plugin, settings, path, "unknown")
    settings["_midom_fish_audio"]["parts"] = [{"part_index": 0, "finish_reason": "unknown", "generated_frames": 100}]
    with pytest.raises(ValueError, match="termination_diagnostics_invalid"):
        plugin._validate_fish_audio_completion(settings, [str(path)], result)


@pytest.mark.parametrize("seconds,warnings", [(0.1, []), (1.25, []), (1.2501, ["generated_duration_over_budget"]), (122, ["generated_duration_over_budget"])])
def test_real_decode_duration_uses_safety_not_generation_ceiling(bridge, tmp_path, seconds, warnings):
    _, plugin = bridge
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    settings = settings_for(plugin)
    settings["duration_seconds"] = 120 if seconds == 122 else 1
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes(seconds))
    result = observe(plugin, settings, path)
    plugin._validate_fish_audio_completion(settings, [str(path)], result)
    metadata = plugin._validate_fish_audio_file(path, settings, native=True)
    final = plugin._validate_fish_audio_file(path, settings, native=False)
    assert final == metadata
    assert [item["code"] for item in settings["_midom_fish_audio"]["warnings"]] == warnings


@pytest.mark.parametrize("rate,channels", [(48000, 1), (44100, 2)])
def test_v2_rejects_instead_of_repairing_geometry(bridge, tmp_path, rate, channels):
    _, plugin = bridge
    settings = settings_for(plugin)
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes(0.1, rate=rate, channels=channels))
    result = observe(plugin, settings, path)
    plugin._validate_fish_audio_completion(settings, [str(path)], result)
    with pytest.raises(ValueError, match="geometry_invalid"):
        plugin._validate_fish_audio_file(path, settings, native=True)


def test_decode_is_bounded_without_trim_downmix_or_resample(bridge, monkeypatch, tmp_path):
    _, plugin = bridge
    commands = []

    def run(command, **bounds):
        commands.append((command, bounds))
        return subprocess.CompletedProcess(command, 0, struct.pack("<f", 0.5),
            b"Output #0, f32le, to pipe:1:\n Stream #0:0: Audio: pcm_f32le, 44100 Hz, mono, flt, 1411 kb/s\n")

    monkeypatch.setattr(plugin, "_fish_audio_bounded_process", run)
    assert plugin._fish_audio_reviewable_decoded_measure(tmp_path / "a.wav", maximum_samples=7938000)["decoded_mono_sample_count"] == 1
    command, bounds = commands[0]
    assert not {"-t", "-ss", "-ar", "-ac"}.intersection(command)
    assert bounds == {"max_stdout_bytes": 7938000 * 4, "timeout_seconds": 90}


@pytest.mark.parametrize("case", [item for item in MEASUREMENTS["cases"] if "probe" in item["id"] or "durations" in item["id"]], ids=lambda item: item["id"])
def test_exact_probe_duration_boundaries(bridge, case):
    _, plugin = bridge
    values = {**MEASUREMENTS["base"], **case["set"]}
    stream, container = {"duration": values["probe_stream_duration"]}, {"duration": values["probe_format_duration"]}
    if case.get("reason") == "artifact_probe_invalid":
        with pytest.raises(ValueError, match="artifact_probe_invalid"):
            plugin._fish_audio_probe_duration(stream, container)
    else:
        value = plugin._fish_audio_probe_duration(stream, container)
        assert value > 0


@pytest.mark.parametrize("kind", ["stdout", "stderr", "timeout", "cancel"])
def test_real_bounded_subprocess_guards(bridge, kind):
    _, plugin = bridge
    code = {"stdout": "import sys; sys.stdout.buffer.write(b'x'*4096)",
            "stderr": "import sys; sys.stderr.buffer.write(b'x'*70000)", "timeout": "import time; time.sleep(10)",
            "cancel": "import time; time.sleep(10)"}[kind]
    if kind == "cancel":
        plugin._cancel_requested_by_midom = True
    with pytest.raises(ValueError):
        plugin._fish_audio_bounded_process([sys.executable, "-c", code], max_stdout_bytes=10, timeout_seconds=0 if kind == "timeout" else 2)


@pytest.mark.parametrize("case", REJECTIONS["metadata_cases"], ids=lambda item: item["id"])
def test_runtime_whole_metadata_bounds(bridge, case):
    _, plugin = bridge
    metadata = copy.deepcopy(JOBS["completion"]["generation_metadata"])
    construction = case["construction"]
    if construction == "non_fish_depth":
        node = "x"
        for _ in range(case["value"] - 2):
            node = {"child": node}
        metadata["fixture_depth"] = node
    elif construction == "old_conditioning_shape":
        metadata["fish_audio"]["native_conditioning"]["prompt_assemblies"][0]["reference_code_fingerprints"] = [{"shape": [10, 100]}]
    elif construction == "whole_utf8_bytes":
        padding = metadata["padding"] = []
        while json_bytes(metadata) < case["value"]:
            padding.append("")
            padding[-1] = "x" * min(7000, case["value"] - json_bytes(metadata))
    elif construction == "non_fish_string_chars":
        metadata["text"] = "x" * case["value"]
    else:
        metadata["texts"] = ["\u00e9" * case["value"]] * 3
    if case["accept"]:
        plugin._validate_fish_audio_metadata_envelope(metadata)
    else:
        with pytest.raises(ValueError):
            plugin._validate_fish_audio_metadata_envelope(metadata)


@pytest.mark.parametrize("case", MEASUREMENTS["cases"], ids=lambda item: item["id"])
def test_runtime_frozen_measurement_matrix(bridge, monkeypatch, tmp_path, case):
    _, plugin = bridge
    value = {**MEASUREMENTS["base"], **case["set"]}
    job = v2_job(fmt=value["requested_format"])
    job["generation"]["duration_seconds"] = value["generation_ceiling_seconds"]
    job["output"]["mime_type"] = value["mime_type"]
    path = tmp_path / "render.wav"
    path.write_bytes(wav_bytes(0.1) if value["genuine_container"] else b"not a WAV container")

    class MeasuredPath:
        def __str__(self):
            return str(path)

        def __getattr__(self, key):
            return getattr(path, key)

        def stat(self):
            return types.SimpleNamespace(st_size=value["bytes"])

    def run(command, **bounds):
        if "-show_streams" in command:
            streams = [{"codec_type": "audio", "sample_rate": value["sample_rate_hz"], "channels": value["channels"],
                        "codec_name": value["codec_name"], "duration": value["probe_stream_duration"]}] * value["probe_audio_streams"]
            streams += [{"codec_type": "video"}] * value["probe_video_streams"]
            probe = {"streams": streams, "format": {"duration": value["probe_format_duration"],
                    "format_name": "mp3" if value["codec_name"] == "mp3" else "wav"}}
            return subprocess.CompletedProcess(command, 0, json.dumps(probe).encode(), b"")
        if value["decode_timed_out"]:
            raise ValueError("artifact_decode_failed: timeout")
        peak = float(value["peak_amplitude"]) if value["decoded_all_samples_finite"] else float("nan")
        raw = struct.pack("<f", peak) * value["decoded_mono_samples"]
        if len(raw) > bounds["max_stdout_bytes"]:
            raise ValueError("artifact_sample_safety_exceeded")
        layout = "mono" if value["decoded_channels"] == 1 else "stereo"
        stderr = f"Output #0, f32le, to pipe:1:\n Audio: pcm_f32le, {value['decoded_sample_rate_hz']} Hz, {layout}, flt, 1411 kb/s\n".encode()
        return subprocess.CompletedProcess(command, 0 if value["decode_complete"] else 1, raw, stderr)

    monkeypatch.setattr(plugin, "_fish_audio_bounded_process", run)
    if type(value["decoded_mono_samples"]) is not int:
        monkeypatch.setattr(plugin, "_fish_audio_reviewable_decoded_measure", lambda path, **kwargs: {
            "decoded_mono_sample_count": value["decoded_mono_samples"], "decoded_duration_seconds": 60, "peak_amplitude": 0.5})

    def validate():
        settings = plugin._validate_audio_job(job, 123)
        result = observe(plugin, settings, path, value["finish_reason"])
        result.success, result.cancelled = value["session_succeeded"], value["cancelled"]
        metadata = settings["_midom_fish_audio"]
        metadata["observed_calls"] = value["observed_calls"]
        metadata["execution_evidence"].update(native_call_returned_output=value["native_call_returned_output"],
                                               observed_calls=value["observed_calls"])
        if not value["job_association_valid"]:
            result.artifacts[0].client_id = "previous_job_client_id"
        if value["source_echo"]:
            metadata["reference_inputs"] = [{"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
        plugin._validate_fish_audio_completion(settings, [str(path)], result)
        plugin._validate_fish_audio_file(MeasuredPath(), settings, native=False)
        return [item["code"] for item in metadata["warnings"]]

    if case["accept"]:
        assert validate() == case["warnings"]
    else:
        with pytest.raises(ValueError):
            validate()


@pytest.mark.parametrize("values,returncode", [([float("nan")], 0), ([float("inf")], 0), ([0.0], 0), ([0.5], 1)])
def test_decode_rejects_nonfinite_silent_and_failed_partial_output(bridge, monkeypatch, tmp_path, values, returncode):
    _, plugin = bridge
    monkeypatch.setattr(plugin, "_fish_audio_bounded_process", lambda command, **kwargs: subprocess.CompletedProcess(
        command, returncode, struct.pack("<" + "f" * len(values), *values),
        b"Output #0, f32le, to pipe:1:\n Audio: pcm_f32le, 44100 Hz, mono, flt, 1411 kb/s\n"))
    with pytest.raises(ValueError):
        plugin._fish_audio_reviewable_decoded_measure(tmp_path / "audio.wav", maximum_samples=7938000)
