import importlib.util
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
    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_ltx25_alpha_matte_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._find_lora_relative_path = lambda directory, filename: filename if (
        directory == "ltx2" and filename == module.LTX25_ALPHA_GEN_LORA_FILENAME
    ) else ""
    plugin._active_job_id = None
    return plugin


def alpha_matte_job(module):
    return {
        "job_id": 981,
        "family": "media_processing",
        "media_type": "video",
        "processing_task": module.VIDEO_ALPHA_MATTING_PROCESSING_TASK,
        "processor_id": module.LTX25_ALPHA_MATTE_PROCESSOR_ID,
        "operation_type": module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION,
        "processing": {
            "operation_type": module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION,
            "contract_version": module.LTX25_ALPHA_MATTE_CONTRACT_VERSION,
            "profile_id": module.LTX25_ALPHA_MATTE_PROFILE_ID,
            "target_filename": "dancer-alpha-matte.mp4",
        },
        "inputs": [{
            "kind": "source_video",
            "role": "matte_source",
            "sequence": 0,
            "input_id": 61,
            "mime_type": "video/mp4",
        }],
        "output": {
            "count": 1,
            "format": "mp4",
            "width": 1280,
            "height": 720,
            "color_mode": "grayscale",
            "artifacts": [{
                "artifact_index": 0,
                "role": "alpha_matte_video",
                "mime_type": "video/mp4",
                "required": True,
            }],
        },
    }


def test_capability_advertises_the_experimental_typed_contract():
    module = load_plugin_module()
    capability = plugin_instance(module)._ltx25_alpha_matte_processing_capability()

    assert capability["experimental"] is True
    assert capability["processor_id"] == module.LTX25_ALPHA_MATTE_PROCESSOR_ID
    assert capability["operation_types"] == [module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION]
    assert capability["contracts"][module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION]["required_output_roles"] == [
        "alpha_matte_video"
    ]
    limits = capability["limits"]["video_alpha_matting"]
    assert limits["window_frames"] == 145
    assert limits["window_overlap_frames"] == 25
    assert limits["output_color_mode"] == "grayscale"


def test_validation_accepts_only_the_narrow_alpha_matte_contract():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_ltx25_alpha_matte_processing_job(alpha_matte_job(module), 981)

    assert settings["model_type"] == module.LTX25_DISTILLED_VIDEO_MODEL_ID
    assert settings["_midom_processor_id"] == module.LTX25_ALPHA_MATTE_PROCESSOR_ID
    assert settings["_midom_processing"]["window_frames"] == 145
    assert settings["_midom_processing"]["window_overlap_frames"] == 25
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "alpha_matte_video", "mime_type": "video/mp4"}
    ]

    bad = alpha_matte_job(module)
    bad["output"]["color_mode"] = "rgb"
    with pytest.raises(ValueError, match="color_mode='grayscale'"):
        plugin._validate_ltx25_alpha_matte_processing_job(bad, 981)

    bad = alpha_matte_job(module)
    bad["inputs"][0]["role"] = "source"
    with pytest.raises(ValueError, match="role must be matte_source"):
        plugin._validate_ltx25_alpha_matte_processing_job(bad, 981)

    bad = alpha_matte_job(module)
    bad["generation"] = {"prompt": "segment the dancer"}
    with pytest.raises(ValueError, match="does not accept prompts or generation settings"):
        plugin._validate_ltx25_alpha_matte_processing_job(bad, 981)


def test_validation_requires_the_locally_installed_alpha_gen_lora():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._find_lora_relative_path = lambda *args, **kwargs: ""

    with pytest.raises(ValueError, match="install"):
        plugin._validate_ltx25_alpha_matte_processing_job(alpha_matte_job(module), 981)


def test_candidate_compatibility_requires_single_grayscale_matte_output():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._worker_id = lambda config: 8
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)
    candidate = {
        "job_id": 981,
        "worker_id": 8,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "family": "media_processing",
        "media_type": "video",
        "model_id": module.LTX25_ALPHA_MATTE_PROCESSOR_ID,
        "processing_task": module.VIDEO_ALPHA_MATTING_PROCESSING_TASK,
        "summary": {
            "operation_type": module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION,
            "processor_id": module.LTX25_ALPHA_MATTE_PROCESSOR_ID,
            "profile_id": module.LTX25_ALPHA_MATTE_PROFILE_ID,
            "source_video_count": 1,
            "output_count": 1,
            "output_format": "mp4",
            "output_color_mode": "grayscale",
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    candidate["summary"]["output_color_mode"] = "rgb"
    assert "output_color_mode=grayscale" in plugin._candidate_incompatibility_reason(candidate, None)


def test_native_submission_uses_fixed_alpha_gen_recipe(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._set_active_job_status = lambda *args, **kwargs: None
    plugin._post_job_update = lambda *args, **kwargs: None
    plugin._active_job_status = {}
    plugin._cancel_requested_by_midom = False
    plugin._callbacks_for_job = lambda *args, **kwargs: object()
    plugin._wait_for_wangp_result = lambda *args, **kwargs: types.SimpleNamespace(success=True, cancelled=False)
    result_settings = {}

    def result_video_files(_result, settings):
        result_settings.update(settings)
        return [str(tmp_path / "raw.mp4")]

    plugin._result_video_files = result_video_files
    finalize_arguments = {}

    def finalize_alpha_matte(generated_path, temp_dir, requested_size, internal_size, source_duration, processing):
        finalize_arguments.update({
            "generated_path": generated_path,
            "temp_dir": temp_dir,
            "requested_size": requested_size,
            "internal_size": internal_size,
            "source_duration": source_duration,
        })
        return tmp_path / "matte.mp4", {"final_sha256": "a" * 64}

    plugin._finalize_ltx25_alpha_matte_output = finalize_alpha_matte
    (tmp_path / "source.mp4").write_bytes(b"source")
    (tmp_path / "source-ltx-overscan.mp4").write_bytes(b"overscan")
    (tmp_path / "raw.mp4").write_bytes(b"raw")
    (tmp_path / "matte.mp4").write_bytes(b"matte")
    plugin._prepare_ltx_overscan_control_video = lambda *args, **kwargs: str(tmp_path / "source-ltx-overscan.mp4")
    submitted = {}

    class Session:
        def submit_task(self, settings, callbacks=None):
            submitted.update(settings)
            return object()

    plugin._get_generation_session = lambda: Session()
    processing = plugin._validate_ltx25_alpha_matte_processing_job(alpha_matte_job(module), 981)
    source = {
        "input_id": 61,
        "kind": "source_video",
        "path": str(tmp_path / "source.mp4"),
        "sha256": "b" * 64,
        "metadata": {"duration_seconds": 18.0, "fps": 24.0, "display_width": 1280, "display_height": 720},
    }
    process_handle = types.SimpleNamespace(cancelled=False)
    artifacts, metadata = plugin._run_ltx25_alpha_matte_processing_job(
        types.SimpleNamespace(worker_id=8), 981, processing, [source], str(tmp_path), process_handle
    )

    assert submitted["prompt"] == "."
    assert submitted["image_prompt_type"] == ""
    assert submitted["video_prompt_type"] == "VG"
    assert submitted["audio_prompt_type"] == ""
    assert submitted["resolution"] == "1280x768"
    assert submitted["video_guide"] == str(tmp_path / "source-ltx-overscan.mp4")
    assert "video_source" not in submitted
    assert not any(str(key).startswith("_midom_") for key in submitted)
    assert submitted["_api"] == {"return_media": True, "suppress_source_audio": True}
    assert submitted["activated_loras"] == [module.LTX25_ALPHA_GEN_LORA_FILENAME]
    assert submitted["sliding_window_size"] == 145
    assert submitted["sliding_window_overlap"] == 25
    assert submitted["sliding_window_discard_last_frames"] == 0
    assert submitted["sliding_window_trim_first_frames"] == 0
    assert submitted["video_length"] == 433
    assert result_settings["_midom_input_video_paths"] == [
        str((tmp_path / "source.mp4").resolve()),
        str((tmp_path / "source-ltx-overscan.mp4").resolve()),
    ]
    assert result_settings["_midom_input_video_sha256s"][0] == "b" * 64
    assert len(result_settings["_midom_input_video_sha256s"][1]) == 64
    assert finalize_arguments["requested_size"] == (1280, 720)
    assert finalize_arguments["internal_size"] == (1280, 768)
    assert artifacts[0]["role"] == "alpha_matte_video"
    assert metadata["neutral_prompt"] == "."


@pytest.mark.parametrize(
    ("requested_size", "internal_size", "expected_crop"),
    [
        ((1280, 720), (1280, 768), "crop=1280:720:0:24,format=gray,setsar=1"),
        ((720, 1280), (768, 1280), "crop=720:1280:24:0,format=gray,setsar=1"),
    ],
)
def test_alpha_matte_finalization_center_crops_internal_overscan(
    tmp_path, monkeypatch, requested_size, internal_size, expected_crop
):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    generated_path = tmp_path / "generated.mp4"
    generated_path.write_bytes(b"raw")
    commands = []

    plugin._ffmpeg_binary = lambda: "ffmpeg"
    plugin._probe_video_display_info = lambda path: {"display_size": internal_size}
    plugin._probe_event_video_metadata = lambda path: {
        "display_width": requested_size[0],
        "display_height": requested_size[1],
        "has_audio": False,
        "fps": 24.0,
        "duration_seconds": 18.0,
    }
    plugin._sha256_file = lambda path: "c" * 64

    def run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"final")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    _, metadata = plugin._finalize_ltx25_alpha_matte_output(
        generated_path,
        tmp_path,
        requested_size,
        internal_size,
        18.0,
        {"target_filename": "alpha.mp4"},
    )

    assert commands[0][commands[0].index("-vf") + 1] == expected_crop
    assert metadata["requested_delivery_dimensions"] == f"{requested_size[0]}x{requested_size[1]}"
    assert metadata["internal_overscan_dimensions"] == f"{internal_size[0]}x{internal_size[1]}"
    assert (metadata["crop_offset_x"], metadata["crop_offset_y"]) == (
        (internal_size[0] - requested_size[0]) // 2,
        (internal_size[1] - requested_size[1]) // 2,
    )
