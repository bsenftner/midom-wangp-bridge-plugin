import copy
import ast
import hashlib
import importlib.util
import io
import math
import os
import shutil
import struct
import sys
import threading
import types
import wave
from pathlib import Path

import pytest


def load_plugin():
    sys.modules.setdefault("gradio", types.SimpleNamespace(Error=Exception))
    for name in ("shared", "shared.api", "shared.utils", "shared.utils.plugins"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["shared.api"].init = lambda **kwargs: None
    sys.modules["shared.utils.plugins"].WAN2GPPlugin = object
    spec = importlib.util.spec_from_file_location("midom_fish_test", Path(__file__).resolve().parents[1] / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bridge(monkeypatch):
    module = load_plugin()
    plugin = object.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._audio_output_mime_types = lambda: ["audio/wav", "audio/mpeg"]
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._validate_job_scope = lambda *args: None
    plugin._headers = lambda connection: {"Authorization": "Bearer test"}
    plugin._active_job_status = {}
    plugin._stop_event = threading.Event()
    plugin._cancel_requested_by_midom = False
    plugin._reset_idle_backoff_for_active_use = lambda *args, **kwargs: None
    updates = []
    plugin._post_job_update = lambda connection, job_id, status, payload: updates.append((status, payload))
    plugin._fish_updates = updates
    audio_video = types.ModuleType("shared.utils.audio_video")
    audio_video._ffmpeg_binary = lambda: shutil.which("ffmpeg") or "ffmpeg"
    monkeypatch.setitem(sys.modules, "shared.utils.audio_video", audio_video)
    return module, plugin


def fish_job(refs=0, *, speakers=None, model="fish_drama3", fmt="wav"):
    speakers = max(refs, 1) if speakers is None else speakers
    mode = "multi_speaker_dialogue" if speakers > 1 else "single_reference" if refs else "new_voice"
    generation = {
        "audio_task": "speech_tts", "speech_contract_version": "fish_drama3_tts_v1",
        "speech_profile_id": "standard_v1", "voice_mode": mode,
        "duration_mode": "maximum_seconds", "duration_seconds": 60, "seed": 123,
        "prompt_enhancement_by_worker": False, "options": {"multi_prompts_gen_type": "FG"},
    }
    if speakers > 1:
        generation["dialogue_mode"] = "speaker_lines"
    return {
        "job_id": 123, "family": "media_generation", "media_type": "audio", "model_id": model,
        "prompt": "\n".join(f"Speaker {speaker}: [warm] Welcome to the neighborhood." for speaker in range(1, speakers + 1)),
        "generation": generation,
        "inputs": [{
            "kind": f"speaker{speaker}_reference_audio", "role": "speaker_reference_audio",
            "speaker_id": speaker, "sequence": speaker - 1, "input_id": 100 + speaker,
            "mime_type": "audio/wav", "transcript": f"Reference recording number {speaker}.",
        } for speaker in range(1, refs + 1)],
        "output": {"count": 1, "format": fmt, "max_duration_seconds": 60},
    }


def summary(job):
    speakers = list(range(1, len(job["prompt"].splitlines()) + 1))
    refs = [item["speaker_id"] for item in job["inputs"]]
    return {
        **{key: value for key, value in job["generation"].items() if key not in {"seed", "options", "prompt_enhancement_by_worker"}},
        "speaker_ids": speakers, "referenced_speaker_ids": refs,
        "speaker_count": len(speakers), "speaker_reference_audio_count": len(refs),
        "reference_transcript_count": len(refs), "input_audio_count": len(refs),
        "reference_audio_count": 0, "source_audio_count": 0, "emotion_reference_audio_count": 0,
        "output_count": 1, "output_format": job["output"]["format"],
    }


def candidate_for(plugin, job):
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 2, "project_id": 3, "paired_user_id": 4}[key]
    return {"job_id": 123, "worker_id": 7, "org_id": 2, "project_id": 3, "requested_by_user_id": 4,
            "media_type": "audio", "model_id": job["model_id"], "summary": summary(job)}


@pytest.mark.parametrize("model", ["fish_drama3", "fish_drama3_int4"])
@pytest.mark.parametrize("refs,speakers", [(0, 1), (1, 1), (0, 3), (1, 3), (2, 2), (3, 3), (3, 16)])
def test_candidate_and_claim_all_routes(bridge, model, refs, speakers):
    module, plugin = bridge
    job = fish_job(refs, speakers=speakers, model=model)
    assert plugin._candidate_incompatibility_reason(candidate_for(plugin, job), None) is None
    settings = plugin._validate_job(job, None)
    assert settings["model_type"] == model
    assert settings["audio_prompt_type"] == ("", "A", "AB", "ABD")[refs]
    assert settings["_midom_fish_audio"]["speaker_ids"] == list(range(1, speakers + 1))
    assert settings["custom_settings"]["dialogue_cut_lines"] == 0
    assert settings["prompt"] == job["prompt"]


def test_capabilities_are_available_without_qualification_flag(bridge):
    module, plugin = bridge
    records = {item["model_id"]: item for item in plugin._fish_audio_capabilities()}
    assert set(records) == module.FISH_AUDIO_MODEL_IDS
    assert set(records).issubset(module.ALLOWED_AUDIO_MODEL_TYPES)
    for record in records.values():
        limits = record["limits"]
        assert limits["max_reference_audio_files"] == 3
        assert limits["required_reference_audio_files"] == 0
        assert limits["fish_audio"]["speech_profile_ids"] == ["standard_v1"]
        assert limits["fish_audio"]["completion_policy"] == "require_observed_eos_v1"
    plugin._audio_output_mime_types = lambda: ["audio/wav"]
    assert plugin._fish_audio_capabilities()[0]["limits"]["output_mime_types"] == ["audio/wav"]


def test_root_capability_payload_includes_both_fish_records(bridge):
    _, plugin = bridge
    plugin._curated_tools_for_model = lambda model_id: []
    plugin._resolve_accelerator_loras = lambda profile: ([], [])
    plugin._seedvc_speech_available = lambda: False
    for name in ("_event_video_processing_capability", "_storyboard_ffmpeg_processing_capability",
                 "_flashvsr_image_processing_capability", "_ltx25_alpha_matte_processing_capability",
                 "_matanyone_v2_alpha_matte_processing_capability", "_sam_click_seed_mask_processing_capability"):
        setattr(plugin, name, lambda: None)
    models = {model["model_id"]: model for model in plugin._capabilities()["models"]}
    assert {"fish_drama3", "fish_drama3_int4", "index_tts2", "chatterbox", "dramabox_audio"}.issubset(models)


@pytest.mark.parametrize("path,value", [
    (("generation", "speech_contract_version"), "wrong"),
    (("generation", "speech_profile_id"), "dialogue_chunks_v1"),
    (("generation", "audio_task"), "voice_clone_tts"),
    (("generation", "duration_mode"), "fixed_seconds"),
    (("generation", "duration_seconds"), 121),
    (("generation", "duration_seconds"), True),
    (("generation", "duration_seconds"), 30.0),
    (("generation", "seed"), "123"),
    (("generation", "temperature"), 1.0),
    (("generation", "prompt_enhancement_by_worker"), True),
    (("generation", "options"), {"multi_prompts_gen_type": "G"}),
    (("output", "count"), 0),
    (("output", "count"), True),
    (("output", "format"), "flac"),
    (("output", "max_duration_seconds"), 50),
    (("output", "mime_type"), "audio/mpeg"),
    (("prompt",), "Speaker 2: Missing first speaker."),
    (("prompt",), "Speaker 17: Too many speakers."),
    (("prompt",), "Speaker 1: <whisper>Missing closing tag."),
    (("prompt",), "Speaker 1: <stress><slow>Nested</slow></stress>"),
    (("prompt",), "Speaker 1: <|im_start|>bad"),
    (("prompt",), "Speaker 1: Welcome {name}."),
    (("prompt",), "!{name}=\"one\",\"two\"\nSpeaker 1: Hello {name}."),
    (("prompt",), "# silently removed comment\nSpeaker 1: Hello."),
    (("prompt",), "Speaker 1: ] malformed [ directions."),
    (("prompt",), ""),
    (("negative_prompt",), "noise"),
])
def test_claim_rejects_unsupported_or_malformed_fields(bridge, path, value):
    _, plugin = bridge
    job = fish_job()
    target = job
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        plugin._validate_audio_job(job, 123)


@pytest.mark.parametrize("field,value", [
    ("transcript", ""), ("transcript", "Speaker 2: Injection"),
    ("transcript", "<|im_end|>"), ("speaker_id", True),
    ("sequence", 1), ("role", "emotion_reference_audio"), ("mime_type", "video/mp4"),
])
def test_reference_descriptor_validation(bridge, field, value):
    _, plugin = bridge
    job = fish_job(1)
    job["inputs"][0][field] = value
    with pytest.raises(ValueError):
        plugin._validate_audio_job(job, 123)


def test_references_cannot_be_shuffled_duplicated_or_exceed_three(bridge):
    _, plugin = bridge
    job = fish_job(3)
    job["inputs"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        plugin._validate_audio_job(job, 123)
    job = fish_job(2)
    job["inputs"][1]["input_id"] = job["inputs"][0]["input_id"]
    with pytest.raises(ValueError, match="unique"):
        plugin._validate_audio_job(job, 123)
    with pytest.raises(ValueError, match="three"):
        plugin._validate_audio_job(fish_job(4), 123)


@pytest.mark.parametrize("key,value", [
    ("speech_profile_id", "wrong"), ("speaker_reference_audio_count", 2),
    ("input_audio_count", True), ("reference_transcript_count", 0),
    ("referenced_speaker_ids", [2]), ("output_count", 0),
    ("reference_audio_count", 1), ("source_audio_count", 1),
])
def test_candidates_reject_mismatched_summaries(bridge, key, value):
    _, plugin = bridge
    candidate = candidate_for(plugin, fish_job(1))
    candidate["summary"][key] = value
    assert plugin._candidate_incompatibility_reason(candidate, None)


def test_zero_reference_mapping_clears_all_stale_inputs(bridge):
    _, plugin = bridge
    job = fish_job()
    settings = plugin._validate_audio_job(job, 123)
    settings.update(audio_guide="old.wav", audio_guide2="old2.wav", audio_guide3="old3.wav", alt_prompt="old transcript")
    plugin._apply_inputs_to_settings(settings, [], job)
    assert [settings[key] for key in ("audio_guide", "audio_guide2", "audio_guide3")] == [None] * 3
    assert settings["alt_prompt"] == ""


def test_line_boundary_normalization_keeps_original_script_hash(bridge):
    _, plugin = bridge
    job = fish_job()
    job["prompt"] = "  Speaker 1: [warm] Hello.  \r\n\r\n"
    settings = plugin._validate_audio_job(job, 123)
    assert settings["prompt"] == "Speaker 1: [warm] Hello."
    assert settings["_midom_fish_audio"]["script_sha256"] == hashlib.sha256(job["prompt"].encode()).hexdigest()


def test_settings_match_real_wangp_generation_signature(bridge):
    _, plugin = bridge
    root = Path(os.environ.get("WAN2GP_ROOT") or Path(__file__).resolve().parents[3])
    if not (root / "wgp.py").is_file():
        pytest.skip("Native source signature check requires the WanGP checkout")
    tree = ast.parse((root / "wgp.py").read_text())
    native = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_media")
    accepted = {arg.arg for arg in native.args.args + native.args.kwonlyargs}
    settings = plugin._validate_audio_job(fish_job(3), 123)
    assert {key for key in settings if not key.startswith("_midom_") and key != "_api"}.issubset(accepted)
    assert "audio_guide3" in accepted


class Pipeline:
    engine = "vllm"

    def __init__(self, reason="eos"):
        self.reason = reason
        self.last_finish_reason = "stale"

    def _generate_codes(self):
        return types.SimpleNamespace(shape=(10, 20)), self.reason

    def generate(self, *, input_prompt, **kwargs):
        _, reason = self._generate_codes()
        self.last_finish_reason = reason
        return {"audio_sampling_rate": 44100}


class Runtime:
    def __init__(self, pipeline, on_generated=None):
        self.pipeline = pipeline
        self.server_config = {"audio_stand_alone_output_codec": "mp3"}
        self.profile = 3
        self.transformer_quantization = "int8"
        self.on_generated = on_generated

        def native_call(model_def, label, fn, *args, **kwargs):
            return fn(*args, **kwargs)

        def generate_media(task, send_cmd=None, *, client_id, model_type, prompt, alt_prompt, **kwargs):
            self.active_codec = self.server_config["audio_stand_alone_output_codec"]
            self.call_with_default_device({}, model_type, self.pipeline.generate,
                                          input_prompt=prompt, alt_prompt=alt_prompt, model_type="fish_drama3",
                                          **{key: kwargs.get(key) for key in ("audio_prompt_type", "audio_guide", "audio_guide2", "audio_guide3")})
            if self.on_generated:
                self.on_generated()
            return True

        self.call_with_default_device = native_call
        self.generate_media = generate_media


class Session:
    def __init__(self, runtime, files=None, *, artifact_files=None, root=None):
        self.runtime = types.SimpleNamespace(module=runtime)
        self.files = files or []
        self.artifact_files = artifact_files
        self._root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self.submitted = []

    def _ensure_runtime(self):
        return self.runtime

    def submit_task(self, settings, callbacks=None):
        self.submitted.append(copy.deepcopy(settings))
        self.runtime.module.generate_media(None, **settings)
        return types.SimpleNamespace(result=lambda: types.SimpleNamespace(
            success=True, cancelled=False, generated_files=self.files, errors=[],
            artifacts=tuple(types.SimpleNamespace(path=path, media_type="audio", client_id=settings["client_id"])
                            for path in (self.files if self.artifact_files is None else self.artifact_files))
            if settings.get("_api", {}).get("return_audio") else ()))


@pytest.mark.parametrize("reason", ["eos", "length", "silence", "early_stop"])
def test_native_observer_captures_reason_and_restores_all_hooks(bridge, reason):
    _, plugin = bridge
    settings = plugin._validate_audio_job(fish_job(), 123)
    pipeline = Pipeline(reason)
    runtime = Runtime(pipeline)
    session = Session(runtime)
    original_generate, original_call = runtime.generate_media, runtime.call_with_default_device
    with plugin._fish_audio_execution(session, settings):
        plugin._submit_wangp_job(session, settings, 1, None)
        assert runtime.active_codec == "wav"
        assert runtime.server_config["audio_stand_alone_output_codec"] == "mp3"
    assert runtime.generate_media is original_generate
    assert runtime.call_with_default_device is original_call
    assert "_generate_codes" not in vars(pipeline)
    assert settings["_midom_fish_audio"]["finish_reason"] == reason
    assert settings["_midom_fish_audio"]["parts"][0]["finish_reason"] == reason
    if reason == "eos":
        plugin._validate_fish_audio_completion(settings, ["generated.wav"])
    else:
        with pytest.raises(ValueError):
            plugin._validate_fish_audio_completion(settings, ["generated.wav"])


def test_unrelated_native_job_is_not_observed_or_given_wav_override(bridge):
    _, plugin = bridge
    settings = plugin._validate_audio_job(fish_job(), 123)
    runtime = Runtime(Pipeline())
    with plugin._fish_audio_execution(Session(runtime), settings):
        runtime.generate_media(None, client_id="another-job", model_type="fish_drama3", prompt="Other", alt_prompt="")
    assert runtime.active_codec == "mp3"
    assert settings["_midom_fish_audio"]["observed_calls"] == 0
    with pytest.raises(ValueError, match="diagnostics_unavailable"):
        plugin._validate_fish_audio_completion(settings, ["generated.wav"])


def test_observer_restores_absent_codec_and_hooks_after_exception(bridge):
    _, plugin = bridge
    settings = plugin._validate_audio_job(fish_job(), 123)

    def fail():
        raise RuntimeError("decode failed")

    runtime = Runtime(Pipeline(), fail)
    runtime.server_config.clear()
    original = runtime.generate_media
    with pytest.raises(RuntimeError, match="decode failed"):
        with plugin._fish_audio_execution(Session(runtime), settings):
            plugin._submit_wangp_job(Session(runtime), settings, 1, None)
    assert runtime.server_config == {}
    assert runtime.generate_media is original
    assert "_generate_codes" not in vars(runtime.pipeline)


def test_private_metadata_filtered_but_api_preserved(bridge):
    _, plugin = bridge
    settings = plugin._validate_audio_job(fish_job(), 123)
    settings["_midom_future_bookkeeping"] = "private"
    settings["_api"] = {"return_side_files": False}
    session = Session(Runtime(Pipeline()))
    with plugin._fish_audio_execution(session, settings):
        plugin._submit_wangp_job(session, settings, 1, None)
    assert not any(key.startswith("_midom_") for key in session.submitted[0])
    assert session.submitted[0]["_api"] == {"return_side_files": False}


def wav_bytes(seconds=3.1, *, rate=44100, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        period = struct.pack("<" + "h" * 100, *[int(2000 * math.sin(index * math.pi / 50)) for index in range(100)])
        writer.writeframes((period * (int(seconds * rate * channels) // 100 + 1))[:int(seconds * rate * channels) * 2])
    return buffer.getvalue()


class Response:
    def __init__(self, data, *, mime="audio/wav", sha=None):
        self.data = data
        self.headers = {"Content-Type": mime, "Content-Length": str(len(data))}
        if sha:
            self.headers["X-Midom-SHA256"] = sha

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data

    def close(self):
        pass


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
@pytest.mark.parametrize("refs", [0, 1, 2, 3])
@pytest.mark.parametrize("model", ["fish_drama3", "fish_drama3_int4"])
def test_real_audio_download_mapping_and_complete_claimed_job(bridge, monkeypatch, tmp_path, refs, model):
    module, plugin = bridge
    job = fish_job(refs, model=model)
    reference = wav_bytes()
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response(reference))
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    output = tmp_path / "generated_seed123.wav"
    output.write_bytes(wav_bytes(0.5))
    session = Session(Runtime(Pipeline()), [str(output)])
    plugin._get_generation_session = lambda: session
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()
    uploaded = []

    def upload(url, **kwargs):
        data = kwargs["files"]["file"][1].read()
        assert kwargs["data"]["sha256"] == hashlib.sha256(data).hexdigest()
        uploaded.append(data)
        return types.SimpleNamespace(status_code=200, json=lambda: {"artifact_id": 12, "file_id": 34, "artifact_index": 0})

    monkeypatch.setattr(module.requests, "post", upload)
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7")
    plugin._run_job(connection, job)
    assert plugin._fish_updates[-1][0] == "complete", plugin._fish_updates
    assert uploaded == [output.read_bytes()]
    completion = plugin._fish_updates[-1][1]
    assert completion["artifacts"] == [{"artifact_id": 12, "file_id": 34, "artifact_index": 0}]
    metadata = completion["generation_metadata"]["fish_audio"]
    assert metadata["finish_reason"] == "eos"
    assert len(metadata["reference_inputs"]) == refs
    assert metadata["artifacts"][0]["sha256"] == hashlib.sha256(uploaded[0]).hexdigest()
    submitted = session.submitted[0]
    assert submitted["audio_prompt_type"] == ("", "A", "AB", "ABD")[refs]
    assert len(submitted["alt_prompt"].splitlines()) == refs
    for index in range(refs):
        item = metadata["reference_inputs"][index]
        assert item["sha256"] == hashlib.sha256(reference).hexdigest()
        assert item["speaker_id"] == index + 1
        assert submitted[("audio_guide", "audio_guide2", "audio_guide3")[index]].endswith(f"fish-speaker-{index + 1}.wav")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
@pytest.mark.parametrize("rate,channels", [(22050, 1), (44100, 2)])
def test_wrong_native_output_rate_or_channels_rejected(bridge, tmp_path, rate, channels):
    _, plugin = bridge
    settings = plugin._validate_audio_job(fish_job(), 123)
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    path = tmp_path / "wrong.wav"
    path.write_bytes(wav_bytes(0.5, rate=rate, channels=channels))
    with pytest.raises(ValueError, match="44100 Hz mono"):
        plugin._validate_fish_audio_file(path, settings, native=True)


@pytest.mark.parametrize("reason", ["silence", "length"])
def test_partial_speech_failure_does_not_upload_or_complete(bridge, reason):
    _, plugin = bridge
    session = Session(Runtime(Pipeline(reason)), ["partial.wav"])
    plugin._get_generation_session = lambda: session
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()
    plugin._upload_audio_artifact = lambda *args: pytest.fail("Partial speech must never upload")
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7")
    plugin._run_job(connection, fish_job())
    assert plugin._fish_updates[-1][0] == "fail"
    assert plugin._fish_updates[-1][1]["reason"] == {"silence": "speech_stopped_prematurely", "length": "speech_duration_limit_reached"}[reason]


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [0.0, 0.0]])
def test_nonfinite_or_silent_decoded_audio_is_rejected(bridge, tmp_path, values):
    _, plugin = bridge
    plugin._fish_audio_process = lambda command: types.SimpleNamespace(returncode=0, stdout=struct.pack("<" + "f" * len(values), *values))
    with pytest.raises(ValueError, match="nonfinite|silent"):
        plugin._fish_audio_decoded_measure(tmp_path / "audio.wav", 60)


@pytest.mark.parametrize("case", ["duplicate_outputs", "no_outputs", "stale", "hidden_failed_part"])
def test_completion_rejects_extra_files_or_unobserved_work(bridge, case):
    _, plugin = bridge
    settings = plugin._validate_audio_job(fish_job(), 123)
    evidence = settings["_midom_fish_audio"]
    evidence.update(observed_calls=1, finish_reason="eos", parts=[{"finish_reason": "eos"}])
    files = ["generated.wav"]
    if case == "duplicate_outputs":
        files.append("other.wav")
    elif case == "no_outputs":
        files.clear()
    elif case == "stale":
        evidence["observed_calls"] = 0
    else:
        evidence["parts"].insert(0, {"finish_reason": "silence"})
    with pytest.raises(ValueError):
        plugin._validate_fish_audio_completion(settings, files)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
@pytest.mark.parametrize("problem", ["hash", "mime", "duration", "sample_rate", "decode"])
def test_download_rejects_bad_reference_media(bridge, monkeypatch, tmp_path, problem):
    module, plugin = bridge
    job = fish_job(1)
    data = wav_bytes(1 if problem == "duration" else 3.1, rate=8000 if problem == "sample_rate" else 44100)
    if problem == "decode":
        data = b"not audio"
    if problem == "hash":
        job["inputs"][0]["sha256"] = "0" * 64
    response = Response(data, mime="audio/mpeg" if problem == "mime" else "audio/wav")
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7)
    with pytest.raises(ValueError):
        plugin._download_job_inputs(connection, job, str(tmp_path))


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_mp3_is_explicitly_encoded_and_final_hash_matches_upload(bridge, monkeypatch, tmp_path):
    module, plugin = bridge
    settings = plugin._validate_audio_job(fish_job(fmt="mp3"), 123)
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    source = tmp_path / "generated.wav"
    source.write_bytes(wav_bytes(0.5))
    uploaded = []

    def upload(url, **kwargs):
        uploaded.append(kwargs["files"]["file"][1].read())
        return types.SimpleNamespace(status_code=200, json=lambda: {"artifact_id": 1, "file_id": 2})

    monkeypatch.setattr(module.requests, "post", upload)
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7)
    plugin._upload_audio_artifact(connection, 123, str(source), 0, settings, str(tmp_path))
    metadata = settings["_midom_fish_audio"]
    assert metadata["artifacts"][0]["codec_name"] == "mp3"
    assert metadata["artifacts"][0]["sha256"] == hashlib.sha256(uploaded[0]).hexdigest()
    assert metadata["native_audio"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert uploaded[0] != source.read_bytes()


def test_cancellation_kills_active_audio_preparation(bridge, monkeypatch):
    module, plugin = bridge
    state = {"killed": False}

    class Process:
        def poll(self):
            return -9 if state["killed"] else None

        def kill(self):
            state["killed"] = True

        def communicate(self, **kwargs):
            return b"", b""

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Process())
    plugin._cancel_requested_by_midom = True
    with pytest.raises(ValueError, match="cancel_requested"):
        plugin._fish_audio_process(["ffmpeg"])
    assert state["killed"]


def test_upload_failure_is_reported_as_failed_not_completed(bridge):
    _, plugin = bridge
    runtime = Runtime(Pipeline())
    session = Session(runtime, ["generated.wav"])
    plugin._get_generation_session = lambda: session
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()

    def reject(*args):
        raise ValueError("Midom rejected artifact upload")

    plugin._upload_audio_artifact = reject
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7")
    plugin._run_job(connection, fish_job())
    assert plugin._fish_updates[-1][0] == "fail"
    assert not any(status == "complete" for status, _ in plugin._fish_updates)
    assert runtime.server_config["audio_stand_alone_output_codec"] == "mp3"
