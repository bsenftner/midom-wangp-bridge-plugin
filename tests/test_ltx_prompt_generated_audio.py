import hashlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


PLUGIN_PATH = Path(__file__).resolve().parents[1] / "plugin.py"


def load_plugin_module():
    sys.modules.setdefault("gradio", types.SimpleNamespace(Error=Exception))

    shared_module = types.ModuleType("shared")
    shared_api_module = types.ModuleType("shared.api")
    shared_api_module.init = lambda *args, **kwargs: None
    shared_utils_module = types.ModuleType("shared.utils")
    shared_plugins_module = types.ModuleType("shared.utils.plugins")
    shared_plugins_module.WAN2GPPlugin = type("WAN2GPPlugin", (), {})

    sys.modules.setdefault("shared", shared_module)
    sys.modules.setdefault("shared.api", shared_api_module)
    sys.modules.setdefault("shared.utils", shared_utils_module)
    sys.modules.setdefault("shared.utils.plugins", shared_plugins_module)

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_ltx_prompt_audio_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._find_lora_relative_path = lambda *args, **kwargs: None
    plugin._lora_hash_cache = {}
    plugin._active_job_id = None
    return plugin


def prompt_audio_job(module, model_id=None):
    return {
        "job_id": 101,
        "media_type": "video",
        "model_id": model_id or module.LTX25_DISTILLED_VIDEO_MODEL_ID,
        "prompt": "A small robot crosses a metal workshop as its footsteps ring on the floor.",
        "generation": {
            "video_task": "audio_conditioned_video",
            "audio_video_mode": module.LTX_PROMPT_GENERATED_AUDIO_MODE,
            "duration_mode": module.LTX_FIXED_DURATION_MODE,
            "duration_seconds": 4,
            "speed_profile_id": "standard",
            "prompt_mode": "plain",
            "seed": 12345,
        },
        "output": {
            "count": 1,
            "format": "mp4",
            "width": 1280,
            "height": 720,
            "max_duration_seconds": 4,
        },
        "inputs": [{"kind": "start_image", "role": "start_image", "input_id": 1}],
    }


def test_capability_advertises_prompt_generated_audio_for_both_models():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    for model_id in module.LTX_VIDEO_MODEL_IDS:
        capability = plugin._ltx_video_capability(model_id, "LTX")
        assert capability["capabilities"]["prompt_generated_audio"] is True
        assert module.LTX_FIXED_DURATION_MODE in capability["limits"]["duration_modes"]
        assert capability["limits"]["prompt_generated_audio_required_input_kinds"] == ["start_image"]
        assert capability["limits"]["prompt_generated_audio_optional_input_kinds"] == ["end_image"]
        assert capability["limits"]["prompt_generated_audio_duration_modes"] == ["fixed_seconds"]


@pytest.mark.parametrize("model_id_attr", ["LTX23_VIDEO_MODEL_ID", "LTX25_DISTILLED_VIDEO_MODEL_ID"])
def test_claim_validation_accepts_fixed_duration_prompt_generated_audio(model_id_attr):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = prompt_audio_job(module, getattr(module, model_id_attr))

    settings = plugin._validate_ltx_video_job(job, 101, job["model_id"])

    assert settings["audio_prompt_type"] == ""
    assert settings["duration_seconds"] == 4
    assert settings["video_length"] == 97
    assert settings["image_prompt_type"] == "S"
    assert settings["_midom_requires_generated_audio"] is True
    assert settings["_midom_audio_video_mode"] == "prompt_generated_audio"
    assert settings["_midom_duration_mode"] == "fixed_seconds"


def test_claim_validation_rejects_mixed_driving_audio_and_malformed_duration():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = prompt_audio_job(module)
    job["inputs"].append({"kind": "driving_audio", "input_id": 2})

    with pytest.raises(ValueError, match="must not include driving_audio"):
        plugin._validate_ltx_video_job(job, 101, job["model_id"])

    job = prompt_audio_job(module)
    job["generation"]["duration_seconds"] = 4.5
    with pytest.raises(ValueError, match="integer from 1 through 20"):
        plugin._validate_ltx_video_job(job, 101, job["model_id"])


def test_candidate_compatibility_accepts_zero_audio_for_prompt_generated_mode():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._worker_id = lambda config: 20
    plugin._connection_scope_value = lambda config, key: {
        "org_id": 30,
        "project_id": 40,
        "paired_user_id": 50,
    }.get(key)
    candidate = {
        "job_id": 101,
        "worker_id": 20,
        "org_id": 30,
        "project_id": 40,
        "requested_by_user_id": 50,
        "media_type": "video",
        "model_id": module.LTX25_DISTILLED_VIDEO_MODEL_ID,
        "summary": {
            "output_format": "mp4",
            "video_task": "audio_conditioned_video",
            "audio_video_mode": "prompt_generated_audio",
            "duration_mode": "fixed_seconds",
            "duration_seconds": 4,
            "start_image_count": 1,
            "end_image_count": 0,
            "control_video_count": 0,
            "input_audio_count": 0,
            "video_sync_profile_id": "standard",
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    candidate["summary"]["input_audio_count"] = 1
    assert plugin._candidate_incompatibility_reason(candidate, None) == "unsupported input_audio_count: 1"


def test_apply_inputs_clears_guides_and_selects_native_soundtrack_generation(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = prompt_audio_job(module)
    settings = plugin._validate_ltx_video_job(job, 101, job["model_id"])
    settings.update({"audio_guide": "stale.wav", "audio_guide2": "stale2.wav", "video_guide": "stale.mp4"})

    start_path = tmp_path / "start.png"
    overscan_path = tmp_path / "start-overscan.png"
    start_path.write_bytes(b"start")
    overscan_path.write_bytes(b"overscan")
    plugin._validate_image_input_exact_size = lambda *args, **kwargs: None
    plugin._prepare_ltx_overscan_image = lambda *args, **kwargs: str(overscan_path)

    plugin._apply_inputs_to_settings(
        settings,
        [{"kind": "start_image", "path": str(start_path), "input_id": 1}],
        job,
    )

    assert settings["image_start"] == [str(overscan_path)]
    assert settings["image_end"] is None
    assert settings["image_prompt_type"] == "S"
    assert settings["audio_prompt_type"] == ""
    assert settings["audio_guide"] is None
    assert settings["audio_guide2"] is None
    assert settings["video_guide"] is None
    plugin._assert_ltx_prompt_generated_audio_submission_integrity(settings)


def test_download_accepts_image_anchors_without_driving_audio(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._headers = lambda connection: {}
    image_bytes = b"qualification-start-image"
    digest = hashlib.sha256(image_bytes).hexdigest()

    class Response:
        headers = {"Content-Type": "image/png", "Content-Length": str(len(image_bytes))}

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def iter_content(chunk_size):
            yield image_bytes

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())
    job = prompt_audio_job(module)
    job["inputs"][0].update({"filename": "start.png", "mime_type": "image/png", "sha256": digest})
    connection = types.SimpleNamespace(api_base_url="https://midom.invalid", worker_id=7)

    downloaded = plugin._download_video_job_inputs(connection, job, str(tmp_path), job["inputs"])

    assert len(downloaded) == 1
    assert downloaded[0]["kind"] == "start_image"
    assert downloaded[0]["sha256"] == digest
    assert Path(downloaded[0]["path"]).read_bytes() == image_bytes


def test_wangp_submission_receives_no_audio_or_video_guide(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = prompt_audio_job(module)
    settings = plugin._validate_ltx_video_job(job, 101, job["model_id"])
    start_path = tmp_path / "start.png"
    overscan_path = tmp_path / "start-overscan.png"
    start_path.write_bytes(b"start")
    overscan_path.write_bytes(b"overscan")
    plugin._validate_image_input_exact_size = lambda *args, **kwargs: None
    plugin._prepare_ltx_overscan_image = lambda *args, **kwargs: str(overscan_path)
    plugin._apply_inputs_to_settings(
        settings,
        [{"kind": "start_image", "path": str(start_path), "input_id": 1}],
        job,
    )
    submitted = {}

    class Session:
        @staticmethod
        def submit_task(public_settings, callbacks=None):
            submitted.update(public_settings)
            return "handle"

    result = plugin._submit_wangp_job(Session(), settings, 1, callbacks=None)

    assert result == "handle"
    assert submitted["model_type"] == module.LTX25_DISTILLED_VIDEO_MODEL_ID
    assert submitted["audio_prompt_type"] == ""
    assert submitted["audio_guide"] is None
    assert submitted["audio_guide2"] is None
    assert submitted["video_guide"] is None
    assert submitted["image_start"] == [str(overscan_path)]
    assert all(not key.startswith("_midom_") for key in submitted)


def test_output_validation_requires_real_non_silent_audio(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    output = tmp_path / "generated.mp4"
    output.write_bytes(b"video")
    plugin._probe_event_video_metadata = lambda path: {
        "has_audio": True,
        "duration_seconds": 4.04,
        "audio_duration_seconds": 4.02,
        "audio_codec": "aac",
    }
    plugin._ffmpeg_binary = lambda: "ffmpeg"
    run_kwargs = []

    def tone_probe(*args, **kwargs):
        run_kwargs.append(kwargs)
        return subprocess.CompletedProcess(args[0], 0, "", "max_volume: -3.2 dB\n")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        tone_probe,
    )

    validation = plugin._validate_ltx_prompt_generated_audio_output(output, 4, 101, 0)
    assert validation["has_audio_stream"] is True
    assert validation["audio_codec"] == "aac"
    assert validation["max_volume_db"] == -3.2
    assert validation["verification_result"] == "passed"
    assert run_kwargs[0]["encoding"] == "utf-8"
    assert run_kwargs[0]["errors"] == "replace"

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", "max_volume: -inf dB\n"),
    )
    with pytest.raises(ValueError, match="digitally silent"):
        plugin._validate_ltx_prompt_generated_audio_output(output, 4, 101, 0)


def test_generated_video_probe_uses_utf8_on_windows_style_subprocess(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    output = tmp_path / "generated.mp4"
    output.write_bytes(b"video")
    plugin._ffprobe_binary = lambda: "ffprobe"
    seen = {}
    payload = (
        '{"streams":['
        '{"codec_type":"video","codec_name":"h264","width":1280,"height":720,'
        '"duration":"4.04","avg_frame_rate":"24/1"},'
        '{"codec_type":"audio","codec_name":"aac","duration":"4.04"}'
        '],"format":{"duration":"4.04","tags":{"title":"Cinematic café"}}}'
    )

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, payload, "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    metadata = plugin._probe_event_video_metadata(output)

    assert metadata["has_audio"] is True
    assert metadata["audio_codec"] == "aac"
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"


def test_completion_metadata_records_prompt_generated_audio_validation():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = {
        "model_type": module.LTX25_DISTILLED_VIDEO_MODEL_ID,
        "_midom_requires_generated_audio": True,
        "_midom_audio_video_mode": module.LTX_PROMPT_GENERATED_AUDIO_MODE,
        "_midom_duration_mode": module.LTX_FIXED_DURATION_MODE,
        "duration_seconds": 4,
        "audio_prompt_type": "",
        "audio_guide": None,
        "video_guide": None,
        "_midom_ltx_prompt_generated_audio_output_validation": [
            {"artifact_index": 0, "has_audio_stream": True, "max_volume_db": -3.2, "verification_result": "passed"}
        ],
    }
    result = types.SimpleNamespace()

    metadata = plugin._build_generation_metadata(settings, result, [])

    assert metadata["prompt_generated_audio"]["mode"] == "prompt_generated_audio"
    assert metadata["prompt_generated_audio"]["duration_mode"] == "fixed_seconds"
    assert metadata["prompt_generated_audio"]["audio_guide_supplied"] is False
    assert metadata["prompt_generated_audio"]["output_validation"][0]["verification_result"] == "passed"
