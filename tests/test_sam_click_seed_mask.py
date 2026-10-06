import hashlib
import importlib.util
import io
import sys
import types
from pathlib import Path

import numpy as np
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
    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_sam_click_seed_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._headers = lambda connection: {"Authorization": "Bearer test"}
    plugin._set_active_job_status = lambda *args, **kwargs: None
    plugin._post_job_update = lambda *args, **kwargs: None
    plugin._active_job_status = {}
    plugin._cancel_requested_by_midom = False
    plugin._sam_click_seed_mask_runtime = lambda: {
        "model": "sam_vit_h",
        "checkpoint_path": "/models/sam_vit_h_4b8939_fp16.safetensors",
        "checkpoint_filename": "sam_vit_h_4b8939_fp16.safetensors",
    }
    return plugin


def seed_job(module, *, width=1280, height=720):
    return {
        "job_id": 1201,
        "family": "media_processing",
        "media_type": "image",
        "processing_task": module.VIDEO_ALPHA_MATTE_SEED_PROCESSING_TASK,
        "processor_id": module.SAM_SEED_MASK_PROCESSOR_ID,
        "operation_type": module.GENERATE_VIDEO_MATTE_SEED_MASK_OPERATION,
        "processing": {
            "operation_type": module.GENERATE_VIDEO_MATTE_SEED_MASK_OPERATION,
            "contract_version": module.SAM_CLICK_SEED_CONTRACT_VERSION,
            "profile_id": module.SAM_VIT_H_CLICK_SEED_PROFILE_ID,
            "seed_frame_index": 0,
            "coordinate_space": "delivery_pixels_v1",
            "positive_points": [{"x": 300, "y": 250}],
            "negative_points": [{"x": 10, "y": 10}],
            "target_filename": "dancer-seed-mask.png",
        },
        "inputs": [{
            "kind": "source_image",
            "role": "seed_frame",
            "sequence": 0,
            "input_id": 88,
            "mime_type": "image/png",
        }],
        "output": {
            "count": 1,
            "format": "png",
            "width": width,
            "height": height,
            "color_mode": "grayscale",
            "artifacts": [{
                "artifact_index": 0,
                "role": "target_seed_mask",
                "mime_type": "image/png",
                "required": True,
            }],
        },
    }


def rgb_png_bytes(width=1280, height=720):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (25, 50, 75)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_capability_advertises_typed_binary_seed_mask_contract():
    module = load_plugin_module()
    capability = plugin_instance(module)._sam_click_seed_mask_processing_capability()

    assert capability["processor_id"] == module.SAM_SEED_MASK_PROCESSOR_ID
    assert capability["supported_operations"] == [module.GENERATE_VIDEO_MATTE_SEED_MASK_OPERATION]
    contract = capability["contracts"][module.GENERATE_VIDEO_MATTE_SEED_MASK_OPERATION]
    assert contract["contract_version"] == module.SAM_CLICK_SEED_CONTRACT_VERSION
    assert contract["input_roles"] == ["seed_frame"]
    assert contract["required_output_roles"] == ["target_seed_mask"]
    limits = capability["limits"]["video_alpha_matte_seed"]
    assert limits["coordinate_space"] == "delivery_pixels_v1"
    assert limits["mask_semantics"] == "binary_black_background_white_target_v1"
    assert limits["positive_points"] == {"min": 1, "max": 32}


def test_validation_requires_exact_stateless_click_seed_contract():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_sam_click_seed_mask_processing_job(seed_job(module), 1201)

    assert settings["_midom_processor_id"] == module.SAM_SEED_MASK_PROCESSOR_ID
    assert settings["_midom_processing"]["positive_points"] == [{"x": 300, "y": 250}]
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "target_seed_mask", "mime_type": "image/png"}
    ]

    parent_operation = seed_job(module)
    parent_operation["operation_type"] = "video_matte_extraction_parent"
    assert plugin._validate_sam_click_seed_mask_processing_job(parent_operation, 1201)[
        "_midom_operation_type"
    ] == module.GENERATE_VIDEO_MATTE_SEED_MASK_OPERATION

    omitted_operation = seed_job(module)
    omitted_operation.pop("operation_type")
    omitted_operation["processing"].pop("operation_type")
    assert plugin._validate_sam_click_seed_mask_processing_job(omitted_operation, 1201)[
        "_midom_operation_type"
    ] == module.GENERATE_VIDEO_MATTE_SEED_MASK_OPERATION

    bad = seed_job(module)
    bad["processing"]["positive_points"] = []
    with pytest.raises(ValueError, match="positive_points count"):
        plugin._validate_sam_click_seed_mask_processing_job(bad, 1201)

    bad = seed_job(module)
    bad["processing"]["negative_points"] = [{"x": 1280, "y": 1}]
    with pytest.raises(ValueError, match="outside frame bounds"):
        plugin._validate_sam_click_seed_mask_processing_job(bad, 1201)

    bad = seed_job(module)
    bad["processing"]["coordinate_space"] = "normalized"
    with pytest.raises(ValueError, match="coordinate_space"):
        plugin._validate_sam_click_seed_mask_processing_job(bad, 1201)

    bad = seed_job(module)
    bad["output"]["artifacts"][0]["role"] = "mask"
    with pytest.raises(ValueError, match="target_seed_mask"):
        plugin._validate_sam_click_seed_mask_processing_job(bad, 1201)


def test_candidate_compatibility_requires_the_narrow_click_seed_summary():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)
    candidate = {
        "job_id": 1201,
        "worker_id": 7,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "family": "media_processing",
        "media_type": "image",
        "model_id": module.SAM_SEED_MASK_PROCESSOR_ID,
        "processing_task": module.VIDEO_ALPHA_MATTE_SEED_PROCESSING_TASK,
        "summary": {
            "operation_type": module.GENERATE_VIDEO_MATTE_SEED_MASK_OPERATION,
            "processor_id": module.SAM_SEED_MASK_PROCESSOR_ID,
            "profile_id": module.SAM_VIT_H_CLICK_SEED_PROFILE_ID,
            "source_image_count": 1,
            "seed_frame_index": 0,
            "coordinate_space": "delivery_pixels_v1",
            "positive_point_count": 1,
            "negative_point_count": 1,
            "output_count": 1,
            "output_format": "png",
            "output_color_mode": "grayscale",
            "output_width": 1280,
            "output_height": 720,
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    candidate["summary"]["positive_point_count"] = 0
    assert "positive_point_count" in plugin._candidate_incompatibility_reason(candidate, None)


def test_candidate_compatibility_reads_operation_from_nested_processing_summary():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)
    candidate = {
        "job_id": 1202,
        "worker_id": 7,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "family": "media_processing",
        "media_type": "image",
        "model_id": module.SAM_SEED_MASK_PROCESSOR_ID,
        "processing_task": module.VIDEO_ALPHA_MATTE_SEED_PROCESSING_TASK,
        "summary": {
            "processing": {"operation_type": module.GENERATE_VIDEO_MATTE_SEED_MASK_OPERATION},
            "processor_id": module.SAM_SEED_MASK_PROCESSOR_ID,
            "profile_id": module.SAM_VIT_H_CLICK_SEED_PROFILE_ID,
            "source_image_count": 1,
            "seed_frame_index": 0,
            "coordinate_space": "delivery_pixels_v1",
            "positive_point_count": 1,
            "negative_point_count": 0,
            "output_count": 1,
            "output_format": "png",
            "output_color_mode": "grayscale",
            "output_width": 1280,
            "output_height": 720,
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, None) is None

    candidate["operation_type"] = "video_matte_extraction_parent"
    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    candidate.pop("operation_type")

    candidate["summary"].pop("processing")
    assert plugin._candidate_incompatibility_reason(candidate, None) is None

    candidate["summary"]["operation_type"] = "some_other_operation"
    assert "operation_type" in plugin._candidate_incompatibility_reason(candidate, None)

    candidate["summary"].pop("operation_type")
    for field in (
        "profile_id", "source_image_count", "seed_frame_index", "coordinate_space",
        "positive_point_count", "negative_point_count", "output_count", "output_format",
        "output_color_mode", "output_width", "output_height",
    ):
        candidate["summary"].pop(field)
    assert plugin._candidate_incompatibility_reason(candidate, None) is None

    candidate["summary"]["profile_id"] = "some_other_profile"
    assert "profile_id" in plugin._candidate_incompatibility_reason(candidate, None)


def test_download_requires_exact_rgb_png_seed_frame(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = seed_job(module)
    payload = rgb_png_bytes()

    class Response:
        headers = {"Content-Type": "image/png"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())
    plugin._read_limited_response_content = lambda response, input_id, maximum: payload
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7)
    downloaded = plugin._download_sam_click_seed_mask_processing_inputs(
        connection, job, str(tmp_path), job["inputs"]
    )

    assert downloaded[0]["role"] == "seed_frame"
    assert downloaded[0]["width"] == 1280
    assert downloaded[0]["sha256"] == hashlib.sha256(payload).hexdigest()

    rgba_buffer = io.BytesIO()
    Image.new("RGBA", (1280, 720), (0, 0, 0, 0)).save(rgba_buffer, format="PNG")
    plugin._read_limited_response_content = lambda response, input_id, maximum: rgba_buffer.getvalue()
    with pytest.raises(ValueError, match="opaque RGB PNG"):
        plugin._download_sam_click_seed_mask_processing_inputs(connection, job, str(tmp_path), job["inputs"])


def test_inference_uses_positive_and_negative_delivery_pixel_clicks(tmp_path):
    module = load_plugin_module()
    source_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), (50, 60, 70)).save(source_path)
    calls = []

    class Segmenter:
        def set_image(self, frame):
            assert frame.shape == (3, 4, 3)

        def predict(self, prompts, mode, multimask):
            calls.append((dict(prompts), mode, multimask))
            masks = np.zeros((2, 3, 4), dtype=bool)
            masks[1, 1, 2] = True
            scores = np.asarray([0.1, 0.9])
            logits = np.zeros((2, 2, 2), dtype=np.float32)
            return masks, scores, logits

    pixels, score = module.AwsWorkerBridgePlugin._run_sam_vit_h_click_seed_inference(
        Segmenter(), source_path, (4, 3), [{"x": 2, "y": 1}], [{"x": 0, "y": 0}]
    )

    assert [mode for _prompts, mode, _multimask in calls] == ["point", "both"]
    assert calls[0][0]["point_labels"].tolist() == [1, 0]
    assert pixels[1, 2] == 255
    assert set(np.unique(pixels)) == {0, 255}
    assert score == pytest.approx(0.9)


def test_runner_writes_binary_typed_seed_artifact_and_provenance(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    source_path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 720), (20, 30, 40)).save(source_path)
    plugin._load_sam_vit_h_click_seed_model = lambda: ("segmenter", {"model": "sam_vit_h"})
    plugin._release_sam_click_seed_model = lambda segmenter: None
    pixels = np.zeros((720, 1280), dtype=np.uint8)
    pixels[100:200, 300:400] = 255
    plugin._run_sam_vit_h_click_seed_inference = lambda *args: (pixels, 0.97)
    settings = plugin._validate_sam_click_seed_mask_processing_job(seed_job(module), 1201)
    artifacts, provenance = plugin._run_sam_click_seed_mask_processing_job(
        types.SimpleNamespace(worker_id=7),
        1201,
        settings,
        [{
            "input_id": 88,
            "kind": "source_image",
            "role": "seed_frame",
            "path": str(source_path),
            "sha256": "a" * 64,
            "width": 1280,
            "height": 720,
        }],
        str(tmp_path),
        types.SimpleNamespace(cancelled=False),
    )

    output_path = Path(artifacts[0]["path"])
    with Image.open(output_path) as result:
        assert result.mode == "L"
        assert result.size == (1280, 720)
        assert set(result.getdata()) == {0, 255}
    assert artifacts[0]["role"] == "target_seed_mask"
    assert provenance["selected_score"] == pytest.approx(0.97)
    assert provenance["foreground_pixels"] == 10000
    assert provenance["artifact_mime_type"] == "image/png"
