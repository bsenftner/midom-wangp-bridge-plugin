import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


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
    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_matanyone_v2_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._active_job_id = None
    plugin._matanyone_v2_alpha_matte_runtime = lambda: {
        "matanyone_version": "v2",
        "weights_path": "/models/matanyone2.safetensors",
        "config_path": "/models/config.json",
    }
    plugin._ffmpeg_processing_probe = lambda: {"ffmpeg_available": True, "ffprobe_available": True}
    return plugin


def alpha_matte_job(module):
    return {
        "job_id": 982,
        "family": "media_processing",
        "media_type": "video",
        "processing_task": module.VIDEO_ALPHA_MATTING_PROCESSING_TASK,
        "processor_id": module.MATANYONE_V2_PROCESSOR_ID,
        "operation_type": module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION,
        "processing": {
            "operation_type": module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION,
            "contract_version": module.MATANYONE_V2_ALPHA_MATTE_CONTRACT_VERSION,
            "profile_id": module.MATANYONE_V2_ALPHA_MATTE_PROFILE_ID,
            "seed_frame_index": 0,
            "target_filename": "dancer-matanyone-alpha.mp4",
        },
        "inputs": [
            {
                "kind": "source_video",
                "role": "matte_source",
                "sequence": 0,
                "input_id": 71,
                "mime_type": "video/mp4",
            },
            {
                "kind": "mask_image",
                "role": "target_seed_mask",
                "sequence": 1,
                "input_id": 72,
                "mime_type": "image/png",
            },
        ],
        "output": {
            "count": 1,
            "format": "mp4",
            "width": 1280,
            "height": 720,
            "color_mode": "grayscale",
            "artifacts": [
                {
                    "artifact_index": 0,
                    "role": "alpha_matte_video",
                    "mime_type": "video/mp4",
                    "required": True,
                }
            ],
        },
    }


def test_capability_advertises_explicit_v2_seeded_alpha_contract():
    module = load_plugin_module()
    capability = plugin_instance(module)._matanyone_v2_alpha_matte_processing_capability()

    assert capability["processor_id"] == module.MATANYONE_V2_PROCESSOR_ID
    assert capability["supported_operations"] == [module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION]
    contract = capability["contracts"][module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION]
    assert contract["contract_version"] == module.MATANYONE_V2_ALPHA_MATTE_CONTRACT_VERSION
    assert contract["input_roles"] == ["matte_source", "target_seed_mask"]
    limits = capability["limits"]["video_alpha_matting"]
    assert limits["matanyone_version"] == "v2"
    assert limits["seed_frame_indices"] == [0]
    assert limits["source_rotation"] == "reject_nonzero"
    assert limits["output_audio"] == "none"


def test_validation_rejects_non_v2_seeded_contract_variants():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_matanyone_v2_alpha_matte_processing_job(alpha_matte_job(module), 982)

    assert settings["_midom_processor_id"] == module.MATANYONE_V2_PROCESSOR_ID
    assert settings["_midom_processing"]["seed_frame_index"] == 0
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "alpha_matte_video", "mime_type": "video/mp4"}
    ]

    bad = alpha_matte_job(module)
    bad["processing"]["seed_frame_index"] = 1
    with pytest.raises(ValueError, match="seed_frame_index=0"):
        plugin._validate_matanyone_v2_alpha_matte_processing_job(bad, 982)

    bad = alpha_matte_job(module)
    bad["inputs"][1]["sequence"] = 0
    with pytest.raises(ValueError, match="target_seed_mask sequence=1"):
        plugin._validate_matanyone_v2_alpha_matte_processing_job(bad, 982)

    bad = alpha_matte_job(module)
    bad["generation"] = {"prompt": "segment the dancer"}
    with pytest.raises(ValueError, match="does not accept prompts or generation settings"):
        plugin._validate_matanyone_v2_alpha_matte_processing_job(bad, 982)


def test_candidate_compatibility_requires_one_source_one_mask_and_forward_seed():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._worker_id = lambda config: 8
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)
    candidate = {
        "job_id": 982,
        "worker_id": 8,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "family": "media_processing",
        "media_type": "video",
        "model_id": module.MATANYONE_V2_PROCESSOR_ID,
        "processing_task": module.VIDEO_ALPHA_MATTING_PROCESSING_TASK,
        "summary": {
            "operation_type": module.EXTRACT_VIDEO_ALPHA_MATTE_OPERATION,
            "processor_id": module.MATANYONE_V2_PROCESSOR_ID,
            "profile_id": module.MATANYONE_V2_ALPHA_MATTE_PROFILE_ID,
            "source_video_count": 1,
            "mask_image_count": 1,
            "seed_frame_index": 0,
            "output_count": 1,
            "output_format": "mp4",
            "output_color_mode": "grayscale",
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    candidate["summary"]["mask_image_count"] = 2
    assert "mask_image_count=1" in plugin._candidate_incompatibility_reason(candidate, None)


def test_seed_mask_requires_binary_foreground_and_background(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    valid = tmp_path / "seed.png"
    image = Image.new("L", (4, 3), 0)
    image.putpixel((1, 1), 255)
    image.save(valid)
    result = plugin._load_matanyone_v2_seed_mask(valid, (4, 3))
    assert result.shape == (3, 4)

    soft = tmp_path / "soft.png"
    Image.new("L", (4, 3), 127).save(soft)
    with pytest.raises(ValueError, match="binary black"):
        plugin._load_matanyone_v2_seed_mask(soft, (4, 3))


def test_runner_uses_seeded_forward_propagation_and_typed_grayscale_output(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._set_active_job_status = lambda *args, **kwargs: None
    plugin._post_job_update = lambda *args, **kwargs: None
    plugin._active_job_status = {}
    plugin._cancel_requested_by_midom = False
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    mask_path = tmp_path / "seed.png"
    seed = Image.new("L", (1280, 720), 0)
    seed.putpixel((10, 10), 255)
    seed.save(mask_path)
    plugin._decode_matanyone_v2_source_frames = lambda *args: ["frame0", "frame1"]
    plugin._load_matanyone_v2_alpha_matte_model = lambda: ("processor", "model", {"matanyone_version": "v2"})
    plugin._release_matanyone_v2_alpha_matte_model = lambda model: None
    captured = {}

    def propagate(processor, frames, seed_mask):
        captured.update({"processor": processor, "frames": frames, "seed_shape": seed_mask.shape})
        return ["alpha0", "alpha1"]

    plugin._propagate_matanyone_v2_alpha = propagate
    output_path = tmp_path / "matte.mp4"
    output_path.write_bytes(b"matte")
    plugin._write_matanyone_v2_alpha_matte_video = lambda *args: (
        output_path,
        {
            "output_dimensions": "1280x720",
            "output_fps": 24.0,
            "output_duration_seconds": 2.0 / 24.0,
            "output_frame_count": 2,
            "final_sha256": "c" * 64,
        },
    )
    settings = plugin._validate_matanyone_v2_alpha_matte_processing_job(alpha_matte_job(module), 982)
    inputs = [
        {
            "input_id": 71,
            "kind": "source_video",
            "path": str(source_path),
            "sha256": "a" * 64,
            "metadata": {"duration_seconds": 2.0 / 24.0, "fps": 24.0, "display_width": 1280, "display_height": 720},
        },
        {"input_id": 72, "kind": "mask_image", "path": str(mask_path), "sha256": "b" * 64},
    ]
    artifacts, provenance = plugin._run_matanyone_v2_alpha_matte_processing_job(
        types.SimpleNamespace(worker_id=8), 982, settings, inputs, str(tmp_path), types.SimpleNamespace(cancelled=False)
    )

    assert captured == {"processor": "processor", "frames": ["frame0", "frame1"], "seed_shape": (720, 1280)}
    assert artifacts == [{"path": str(output_path), "artifact_index": 0, "role": "alpha_matte_video", "mime_type": "video/mp4"}]
    assert provenance["matanyone_version"] == "v2"
    assert provenance["source_frame_count"] == 2
    assert provenance["seed_mask_sha256"] == "b" * 64
