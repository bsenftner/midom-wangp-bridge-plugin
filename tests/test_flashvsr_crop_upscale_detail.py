import hashlib
import importlib.util
import io
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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_flashvsr_crop_detail_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Headers(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._headers = lambda connection: {"Authorization": "Bearer test"}
    plugin._post_job_update = lambda *args, **kwargs: None
    plugin._set_active_job_status = lambda *args, **kwargs: None
    plugin._active_job_status = {}
    plugin._active_job = None
    plugin._cancel_requested_by_midom = False
    plugin._flashvsr_image_processing_runtime = lambda: {
        "method": "flashvsr",
        "spatial_upsampling": module.FLASHVSR_X2_SPATIAL_UPSAMPLING,
        "model_variant": "Full",
        "sparse_backend": "cuda",
        "topk_ratio": 0.2,
    }
    return plugin


def rgb_png_bytes(width=1280, height=720):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 80, 150)).save(buffer, format="PNG")
    return buffer.getvalue()


def restoration_job(module, *, width=1280, height=720):
    return {
        "job_id": 9001,
        "family": "media_processing",
        "media_type": "image",
        "processing_task": module.IMAGE_DETAIL_RESTORATION_PROCESSING_TASK,
        "processor_id": module.FLASHVSR_IMAGE_PROCESSOR_ID,
        "operation_type": module.RESTORE_CROP_UPSCALE_DETAIL_OPERATION,
        "processing": {
            "operation_type": module.RESTORE_CROP_UPSCALE_DETAIL_OPERATION,
            "contract_version": module.RESTORE_CROP_UPSCALE_DETAIL_CONTRACT_VERSION,
            "profile_id": module.FLASHVSR_X2_STANDARD_PROFILE_ID,
            "method": "flashvsr",
            "intermediate_scale": 2,
            "post_resize": {"method": "lanczos", "target_width": width, "target_height": height},
            "target_filename": "restored-crop.png",
        },
        "inputs": [
            {
                "kind": "source_image",
                "role": "crop_upscale_derivative",
                "sequence": 0,
                "input_id": 101,
                "mime_type": "image/png",
                "filename": "crop.png",
            }
        ],
        "output": {
            "count": 1,
            "format": "png",
            "width": width,
            "height": height,
            "color_mode": "rgb",
            "mime_type": "image/png",
            "artifacts": [
                {
                    "artifact_index": 0,
                    "role": "restored_crop_image",
                    "mime_type": "image/png",
                    "required": True,
                }
            ],
        },
    }


def test_capability_advertises_flashvsr_crop_detail_contract():
    module = load_plugin_module()
    capability = plugin_instance(module)._flashvsr_image_processing_capability()

    assert capability["processor_id"] == module.FLASHVSR_IMAGE_PROCESSOR_ID
    assert capability["operation_types"] == [module.RESTORE_CROP_UPSCALE_DETAIL_OPERATION]
    assert capability["contracts"][module.RESTORE_CROP_UPSCALE_DETAIL_OPERATION]["contract_version"] == (
        module.RESTORE_CROP_UPSCALE_DETAIL_CONTRACT_VERSION
    )
    assert capability["contracts"][module.RESTORE_CROP_UPSCALE_DETAIL_OPERATION]["required_output_roles"] == [
        "restored_crop_image"
    ]
    assert capability["limits"]["crop_upscale_detail_restoration"]["delivery_resolutions"] == [
        "768x768", "1024x1024", "1280x720", "720x1280"
    ]


def test_candidate_compatibility_requires_the_narrow_flashvsr_summary():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)
    candidate = {
        "job_id": 9001,
        "worker_id": 7,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "family": "media_processing",
        "media_type": "image",
        "model_id": module.FLASHVSR_IMAGE_PROCESSOR_ID,
        "processing_task": module.IMAGE_DETAIL_RESTORATION_PROCESSING_TASK,
        "summary": {
            "operation_type": module.RESTORE_CROP_UPSCALE_DETAIL_OPERATION,
            "processor_id": module.FLASHVSR_IMAGE_PROCESSOR_ID,
            "profile_id": module.FLASHVSR_X2_STANDARD_PROFILE_ID,
            "source_image_count": 1,
            "input_color_mode": "rgb",
            "output_count": 1,
            "output_format": "png",
            "output_color_mode": "rgb",
            "output_width": 1280,
            "output_height": 720,
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    candidate["summary"]["input_color_mode"] = "rgba"
    assert "input_color_mode='rgb'" in plugin._candidate_incompatibility_reason(candidate, None)


def test_claim_validation_accepts_exact_contract_and_rejects_alternatives():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_flashvsr_image_processing_job(restoration_job(module), 9001)

    assert settings["_midom_processor_id"] == module.FLASHVSR_IMAGE_PROCESSOR_ID
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "restored_crop_image", "mime_type": "image/png"}
    ]
    assert settings["_midom_processing"]["post_resize"]["method"] == "lanczos"

    bad_count = restoration_job(module)
    bad_count["output"]["count"] = 2
    with pytest.raises(ValueError, match="output.count=1"):
        plugin._validate_flashvsr_image_processing_job(bad_count, 9001)

    bad_profile = restoration_job(module)
    bad_profile["processing"]["profile_id"] = "other"
    with pytest.raises(ValueError, match="profile_id"):
        plugin._validate_flashvsr_image_processing_job(bad_profile, 9001)

    bad_input = restoration_job(module)
    bad_input["inputs"].append(dict(bad_input["inputs"][0], input_id=102))
    with pytest.raises(ValueError, match="exactly one source_image"):
        plugin._validate_flashvsr_image_processing_job(bad_input, 9001)


def test_download_requires_rgb_png_exactly_matching_the_crop_canvas(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    body = rgb_png_bytes()
    digest = hashlib.sha256(body).hexdigest()
    job = restoration_job(module)
    job["inputs"][0].update({"bytes": len(body), "sha256": digest})

    class Response:
        headers = Headers({"Content-Type": "image/png", "Content-Length": str(len(body))})

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def iter_content(chunk_size):
            yield body

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())
    downloaded = plugin._download_flashvsr_image_processing_inputs(
        types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
        job,
        str(tmp_path),
        job["inputs"],
    )

    assert downloaded[0]["sha256"] == digest
    assert downloaded[0]["decoded_color_mode"] == "RGB"
    assert (downloaded[0]["width"], downloaded[0]["height"]) == (1280, 720)


def test_flashvsr_result_is_lanczos_resized_to_rgb_png_and_typed(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_flashvsr_image_processing_job(restoration_job(module), 9001)
    source = tmp_path / "crop.png"
    source.write_bytes(rgb_png_bytes())
    native = tmp_path / "flashvsr-native.png"
    Image.new("RGB", (2560, 1440), (200, 100, 50)).save(native, format="PNG")
    submitted = {}

    class Session:
        @staticmethod
        def submit_task(public_settings, callbacks=None):
            submitted.update(public_settings)
            return types.SimpleNamespace(done=True)

    plugin._get_generation_session = lambda: Session()
    plugin._callbacks_for_job = lambda *args, **kwargs: types.SimpleNamespace()
    plugin._wait_for_wangp_result = lambda *args, **kwargs: types.SimpleNamespace(
        success=True, cancelled=False, errors=[], generated_files=[str(native)], artifacts=[]
    )
    downloaded = [{
        "kind": "source_image", "role": "crop_upscale_derivative", "sequence": 0,
        "input_id": 101, "path": str(source), "sha256": "input-sha",
        "width": 1280, "height": 720,
    }]
    process_handle = types.SimpleNamespace(cancelled=False)

    artifacts, metadata = plugin._run_flashvsr_image_processing_job(
        types.SimpleNamespace(worker_id=7), 9001, settings, downloaded, str(tmp_path), process_handle
    )

    assert submitted["mode"] == "edit_postprocessing"
    assert submitted["spatial_upsampling"] == module.FLASHVSR_X2_SPATIAL_UPSAMPLING
    assert submitted["video_source"] == str(source)
    output = Path(artifacts[0]["path"])
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (1280, 720)
    assert artifacts[0]["artifact_index"] == 0
    assert artifacts[0]["role"] == "restored_crop_image"
    assert artifacts[0]["mime_type"] == "image/png"
    assert metadata["native_intermediate_dimensions"] == "2560x1440"
    assert metadata["post_resize"]["method"] == "lanczos"
    assert metadata["final_dimensions"] == "1280x720"


def test_final_output_rejects_rgba(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "invalid.png"
    Image.new("RGBA", (1280, 720), (1, 2, 3, 255)).save(path, format="PNG")

    with pytest.raises(ValueError, match="RGB PNG"):
        plugin._validate_flashvsr_crop_upscale_output(path, (1280, 720), (2560, 1440))


def test_typed_upload_uses_restored_crop_image_role(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "restored-crop.png"
    path.write_bytes(rgb_png_bytes())
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"artifact_id": 40, "file_id": 50, "artifact_index": 0, "role": "restored_crop_image"}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured["data"] = data
        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)
    result = plugin._upload_media_processing_artifact(
        types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
        9001,
        {"path": str(path), "artifact_index": 0, "role": "restored_crop_image", "mime_type": "image/png"},
        {},
    )

    assert result == {"artifact_id": 40, "file_id": 50, "artifact_index": 0, "role": "restored_crop_image"}
    assert captured["data"]["artifact_index"] == "0"
    assert captured["data"]["role"] == "restored_crop_image"
    assert captured["data"]["mime_type"] == "image/png"
