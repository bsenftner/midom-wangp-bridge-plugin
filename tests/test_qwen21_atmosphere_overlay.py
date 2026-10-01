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
    spec = importlib.util.spec_from_file_location("midom_bridge_atmosphere_overlay_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._headers = lambda _connection: {}
    return plugin


def overlay_job():
    return {
        "job_id": 2901,
        "family": "media_generation",
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "Warm theatrical haze with soft amber bloom from stage left.",
        "negative_prompt": "",
        "generation": {
            "image_task": "atmosphere_overlay",
            "atmosphere_overlay_contract_version": "qwen21_atmosphere_overlay_v1",
            "atmosphere_overlay_mode": "atmosphere_tint",
            "reference_mode": "primary_image_edit",
            "accelerator_profile_id": "standard",
            "prompt_enhancement_by_worker": False,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": [{
            "kind": "reference_image", "role": "scene_image", "sequence": 0,
            "input_id": 901, "mime_type": "image/png",
        }],
        "output": {"count": 1, "format": "png", "color_mode": "rgba", "width": 1280, "height": 720},
    }


def connection(module):
    return module.ConnectionContext(
        connection_id="worker-1", api_base_url="https://midom.test", worker_id=1,
        worker_token="token", org_id=2, project_id=3, paired_user_id=4,
        machine_name="GPU", capabilities_revision=1, token_expires_at="",
        allow_insecure_local_dev=False, allow_insecure_lan_dev=False,
    )


def test_atmosphere_overlay_capability_and_recipe():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    capability = plugin._qwen21_atmosphere_overlay_capability()
    settings = plugin._validate_image_job(overlay_job(), 2901)

    assert capability["experimental"] is True
    assert capability["modes"] == ["atmosphere_tint"]
    assert capability["output_color_modes"] == ["rgba"]
    assert capability["alpha_validation"]["min_partial_alpha_fraction"] == 0.01
    assert settings["resolution"] == "1280x736"
    assert settings["num_inference_steps"] == 50
    assert settings["guidance_scale"] == 8.0
    assert settings["custom_settings"] == {"qwen21_kv_cache": "Disabled", "rgba": "Enabled"}
    assert "mostly semi-transparent atmospheric effects" in settings["prompt"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda job: job["generation"].update(atmosphere_overlay_mode="fog"), "atmosphere_overlay_mode"),
        (lambda job: job["generation"].update(accelerator_profile_id="qwen21_pruna_v01_8"), "only accelerator_profile_id='standard'"),
        (lambda job: job["output"].update(color_mode="rgb"), "output.color_mode"),
        (lambda job: job["inputs"][0].update(role="source_image"), "role='scene_image'"),
        (lambda job: job["inputs"].append({"kind": "mask_image", "role": "integration_matte", "input_id": 902}), "exactly one scene_image"),
    ],
)
def test_atmosphere_overlay_rejects_contract_drift(mutator, message):
    module = load_plugin_module()
    job = overlay_job()
    mutator(job)
    with pytest.raises(ValueError, match=message):
        plugin_instance(module)._validate_image_job(job, 2901)


def test_atmosphere_overlay_candidate_requires_rgba_and_one_scene_reference():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 2902, "worker_id": 1, "org_id": 2, "project_id": 3,
        "requested_by_user_id": 4, "media_type": "image", "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "atmosphere_overlay",
            "atmosphere_overlay_contract_version": "qwen21_atmosphere_overlay_v1",
            "atmosphere_overlay_mode": "atmosphere_tint",
            "reference_mode": "primary_image_edit",
            "reference_image_count": 1,
            "mask_image_count": 0,
            "output_count": 1,
            "output_format": "png",
            "output_color_mode": "rgba",
            "accelerator_profile_id": "standard",
        },
    }
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["output_color_mode"] = "rgb"
    assert "requires output_color_mode=rgba" in plugin._candidate_incompatibility_reason(candidate, connection(module))


def test_atmosphere_alpha_validation_rejects_opaque_and_accepts_partial(tmp_path, monkeypatch):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    partial_path = tmp_path / "partial.png"
    opaque_path = tmp_path / "opaque.png"
    partial = Image.new("RGBA", (32, 32), (230, 180, 100, 80))
    partial.save(partial_path)
    Image.new("RGBA", (32, 32), (230, 180, 100, 255)).save(opaque_path)

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"artifact_id": 1, "file_id": 2, "artifact_index": 0}

    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: Response())
    validation = []
    plugin._upload_artifact(
        connection(module), 2903, str(partial_path), 0, "32x32", "resize", "image/png", "RGBA",
        False, validation, None, True,
    )
    assert validation[0]["native_alpha_validation"] == "atmosphere_overlay_passed"
    assert validation[0]["partial_alpha_fraction"] == 1.0
    with pytest.raises(ValueError, match="did not produce enough semi-transparent pixels"):
        plugin._upload_artifact(
            connection(module), 2903, str(opaque_path), 0, "32x32", "resize", "image/png", "RGBA",
            False, [], None, True,
        )
