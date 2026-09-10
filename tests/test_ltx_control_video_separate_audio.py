import importlib.util
import sys
import types
from pathlib import Path


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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_ltx_control_audio_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._find_lora_relative_path = lambda *args, **kwargs: None
    return plugin


def ltx_control_job(module):
    return {
        "job_id": 1,
        "media_type": "video",
        "model_id": module.LTX23_VIDEO_MODEL_ID,
        "prompt": "A presenter speaks clearly to camera.",
        "generation": {
            "video_task": "control_video_guided_video",
            "audio_video_mode": "control_video_audio_guided",
            "duration_mode": module.LTX_CONTROL_VIDEO_DURATION_MODE,
            "control_video_mode": "human_motion",
            "speed_profile_id": "standard",
            "prompt_mode": "plain",
        },
        "output": {"count": 1, "format": "mp4", "width": 1280, "height": 720, "max_duration_seconds": 8},
        "inputs": [
            {"kind": "start_image", "input_id": 1},
            {"kind": "control_video", "input_id": 2},
            {"kind": "driving_audio", "input_id": 3},
        ],
    }


def test_ltx_capability_reports_optional_separate_control_video_driving_audio():
    module = load_plugin_module()
    capability = plugin_instance(module)._ltx_video_capability(module.LTX23_VIDEO_MODEL_ID, "LTX")

    assert capability["capabilities"]["separate_driving_audio_with_control_video"] is True
    assert "driving_audio" in capability["limits"]["control_video_optional_input_kinds"]


def test_ltx_control_video_validation_accepts_optional_driving_audio():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = ltx_control_job(module)

    settings = plugin._validate_ltx_control_video_job(
        job,
        1,
        module.LTX23_VIDEO_MODEL_ID,
        job["generation"],
    )

    assert settings["_midom_video_task"] == "control_video_guided_video"
    assert settings["audio_prompt_type"] == "K"


def test_ltx_control_video_candidate_allows_optional_driving_audio():
    module = load_plugin_module()
    candidate = {
        "job_id": 10,
        "worker_id": 20,
        "org_id": 30,
        "project_id": 40,
        "requested_by_user_id": 50,
        "media_type": "video",
        "model_id": module.LTX23_VIDEO_MODEL_ID,
        "summary": {
            "output_format": "mp4",
            "video_task": "control_video_guided_video",
            "audio_video_mode": "control_video_audio_guided",
            "duration_mode": module.LTX_CONTROL_VIDEO_DURATION_MODE,
            "control_video_mode": "human_motion",
            "start_image_count": 1,
            "control_video_count": 1,
            "input_audio_count": 1,
            "video_sync_profile_id": "standard",
        },
    }
    plugin = plugin_instance(module)
    plugin._worker_id = lambda config: 20
    plugin._connection_scope_value = lambda config, key: {
        "org_id": 30,
        "project_id": 40,
        "paired_user_id": 50,
    }.get(key)

    assert plugin._candidate_incompatibility_reason(candidate, None) is None


def test_ltx_control_video_apply_inputs_uses_separate_driving_audio(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = ltx_control_job(module)
    settings = plugin._validate_ltx_control_video_job(
        job,
        1,
        module.LTX23_VIDEO_MODEL_ID,
        job["generation"],
    )

    start_path = tmp_path / "start.png"
    control_path = tmp_path / "control.mp4"
    audio_path = tmp_path / "voice.wav"
    overscan_start_path = tmp_path / "start-overscan.png"
    overscan_control_path = tmp_path / "control-overscan.mp4"
    for path in (start_path, control_path, audio_path, overscan_start_path, overscan_control_path):
        path.write_bytes(b"x")

    seen_require_audio = []
    plugin._probe_control_video_metadata = lambda path, require_audio=True: seen_require_audio.append(require_audio) or {
        "width": 1280,
        "height": 720,
        "video_duration_seconds": 8.0,
        "audio_duration_seconds": 0.0,
        "fps": module.LTX_VIDEO_FPS,
    }
    plugin._probe_audio_metadata = lambda path: {"duration_seconds": 8.0, "sample_rate_hz": 48000}
    plugin._validate_image_input_exact_size = lambda *args, **kwargs: None
    plugin._prepare_ltx_overscan_image = lambda *args, **kwargs: str(overscan_start_path)
    plugin._prepare_ltx_overscan_control_video = lambda *args, **kwargs: str(overscan_control_path)

    plugin._apply_inputs_to_settings(
        settings,
        [
            {"kind": "start_image", "path": str(start_path)},
            {"kind": "control_video", "path": str(control_path), "sha256": "abc"},
            {"kind": "driving_audio", "path": str(audio_path)},
        ],
        job,
    )

    assert seen_require_audio == [False]
    assert settings["video_guide"] == str(overscan_control_path)
    assert settings["audio_guide"] == str(audio_path)
    assert settings["audio_prompt_type"] == "A"
    assert settings["video_prompt_type"] == module.LTX_CONTROL_VIDEO_MODES["human_motion"]
