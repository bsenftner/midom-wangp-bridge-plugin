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
    spec = importlib.util.spec_from_file_location("midom_bridge_scene_integration_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._ensure_job_flow_enabled = lambda: None
    return plugin


def draft_job():
    return {
        "job_id": 2801,
        "family": "media_generation",
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "Unify the dusk lighting around the actors.",
        "negative_prompt": "",
        "generation": {
            "image_task": "scene_integration",
            "scene_integration_contract_version": "qwen21_scene_integration_v1",
            "integration_profile_id": "draft_v1",
            "integration_mode": "full_scene",
            "reference_mode": "scene_integration",
            "accelerator_profile_id": "standard",
            "prompt_enhancement_by_worker": False,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": [{
            "kind": "reference_image", "role": "scene_composite", "sequence": 0,
            "input_id": 801, "mime_type": "image/png",
        }],
        "output": {"count": 2, "format": "png", "color_mode": "rgb", "width": 1280, "height": 720},
    }


def production_job():
    return {
        "job_id": 2802,
        "family": "media_generation",
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "Create natural contact shadows beneath the performer.",
        "negative_prompt": "",
        "generation": {
            "image_task": "scene_integration",
            "scene_integration_contract_version": "qwen21_scene_integration_v1",
            "integration_profile_id": "production_v1",
            "integration_mode": "edge_matte",
            "reference_mode": "scene_integration",
            "mask_semantics": "luminance_white_edit_v1",
            "accelerator_profile_id": "standard",
            "prompt_enhancement_by_worker": False,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": [
            {
                "kind": "reference_image", "role": "location_plate", "sequence": 0,
                "input_id": 802, "mime_type": "image/png",
            },
            {
                "kind": "reference_image", "role": "placed_asset", "sequence": 1,
                "input_id": 803, "mime_type": "image/png",
                "placement": {"x": 400, "y": 200, "width": 240, "height": 400},
            },
            {
                "kind": "mask_image", "role": "integration_matte", "input_id": 804,
                "mime_type": "image/png",
            },
        ],
        "output": {"count": 1, "format": "png", "color_mode": "rgb", "width": 1280, "height": 720},
    }


def flattened_production_job():
    return {
        "job_id": 2804,
        "family": "media_generation",
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "Create a natural contact shadow below the performers.",
        "negative_prompt": "",
        "generation": {
            "image_task": "scene_integration",
            "scene_integration_contract_version": "qwen21_scene_integration_v2",
            "integration_profile_id": "production_flattened_matte_v1",
            "integration_mode": "visible_edge_matte",
            "reference_mode": "scene_integration",
            "mask_semantics": "luminance_white_edit_v1",
            "accelerator_profile_id": "standard",
            "prompt_enhancement_by_worker": False,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": [
            {
                "kind": "reference_image", "role": "scene_composite", "sequence": 0,
                "input_id": 805, "mime_type": "image/png",
            },
            {
                "kind": "mask_image", "role": "integration_matte", "input_id": 806,
                "mime_type": "image/png",
            },
        ],
        "output": {"count": 1, "format": "png", "color_mode": "rgb", "width": 1280, "height": 720},
    }


def connection(module):
    return module.ConnectionContext(
        connection_id="worker-1", api_base_url="https://midom.test", worker_id=1,
        worker_token="token", org_id=2, project_id=3, paired_user_id=4,
        machine_name="GPU", capabilities_revision=1, token_expires_at="",
        allow_insecure_local_dev=False, allow_insecure_lan_dev=False,
    )


def test_scene_integration_capability_exposes_draft_and_production_profiles():
    module = load_plugin_module()
    capability = plugin_instance(module)._qwen21_scene_integration_capability()

    assert capability["contract_version"] == "qwen21_scene_integration_v1"
    assert capability["output_color_modes"] == ["rgb"]
    assert [profile["profile_id"] for profile in capability["profiles"]] == ["draft_v1", "production_v1"]
    assert capability["profiles"][0]["mask_required"] is False
    assert capability["profiles"][1]["mask_required"] is True
    assert capability["profiles"][1]["protected_pixel_guarantee"] == "exact_post_generation_restore_v1"
    assert capability["profiles"][1]["internal_render_resolutions"]["1280x720"] == "1920x1088"
    flattened = plugin_instance(module)._qwen21_scene_integration_v2_capability()
    assert flattened["contract_version"] == "qwen21_scene_integration_v2"
    assert flattened["profiles"][0]["profile_id"] == "production_flattened_matte_v1"
    assert flattened["profiles"][0]["input_roles"] == ["scene_composite", "integration_matte"]
    assert flattened["profiles"][0]["max_editable_area_fraction"] == 0.60


def test_draft_and_production_use_distinct_recipes():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    draft = plugin._validate_image_job(draft_job(), 2801)
    production = plugin._validate_image_job(production_job(), 2802)

    assert draft["resolution"] == "1280x736"
    assert draft["image_mode"] == 1
    assert draft["video_prompt_type"] == "KI"
    assert draft["num_inference_steps"] == 40
    assert draft["guidance_scale"] == 4.0
    assert production["resolution"] == "1920x1088"
    assert production["image_mode"] == 2
    assert production["video_prompt_type"] == "VAG"
    assert production["num_inference_steps"] == 50
    assert production["guidance_scale"] == 8.0
    assert production["_midom_scene_integration_production"] is True


def test_flattened_production_uses_the_production_recipe_without_v1_asset_inputs():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(flattened_production_job(), 2804)

    assert settings["resolution"] == "1920x1088"
    assert settings["image_mode"] == 2
    assert settings["video_prompt_type"] == "VAG"
    assert settings["num_inference_steps"] == 50
    assert settings["guidance_scale"] == 8.0
    assert settings["_midom_scene_integration_production"] is True
    assert settings["_midom_scene_integration_flattened_production"] is True


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda job: job["generation"].update(integration_mode="full_scene"), "integration_mode='edge_matte'"),
        (lambda job: job["generation"].update(accelerator_profile_id="qwen21_pruna_v01_8"), "only accelerator_profile_id='standard'"),
        (lambda job: job["output"].update(count=2), "output.count=1"),
        (lambda job: job["inputs"].pop(), "requires one location_plate"),
        (lambda job: job["inputs"][1].update(placement={"x": -1, "y": 0, "width": 10, "height": 10}), "non-negative integers"),
        (lambda job: job["inputs"][1].update(placement={"x": 1200, "y": 0, "width": 100, "height": 10}), "within the delivery canvas"),
    ],
)
def test_production_rejects_contract_drift(mutator, message):
    module = load_plugin_module()
    job = production_job()
    mutator(job)
    with pytest.raises(ValueError, match=message):
        plugin_instance(module)._validate_image_job(job, 2802)


def test_production_composite_and_restoration_preserve_protected_pixels(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    location_path = tmp_path / "location.png"
    asset_path = tmp_path / "asset.png"
    matte_path = tmp_path / "matte.png"
    generated_path = tmp_path / "generated.png"
    Image.new("RGB", (1280, 720), (20, 40, 60)).save(location_path)
    asset = Image.new("RGBA", (20, 40), (200, 30, 20, 0))
    for x in range(5, 15):
        for y in range(5, 35):
            asset.putpixel((x, y), (200, 30, 20, 255))
    asset.save(asset_path)
    matte = Image.new("L", (1280, 720), 0)
    matte.paste(255, (390, 190, 651, 611))
    matte.save(matte_path)
    prepared = plugin._prepare_qwen21_scene_integration_production_inputs(
        {"input_id": 802, "path": str(location_path), "sha256": "a" * 64},
        {
            "input_id": 803, "path": str(asset_path), "sha256": "b" * 64,
            "placement": {"x": 400, "y": 200, "width": 240, "height": 400},
        },
        {"input_id": 804, "path": str(matte_path), "sha256": "c" * 64},
        (1280, 720),
        (1920, 1088),
    )
    assert Image.open(prepared["internal_composite_path"]).size == (1920, 1088)
    assert Image.open(prepared["internal_mask_path"]).size == (1920, 1088)
    Image.new("RGB", (1920, 1088), (255, 0, 0)).save(generated_path)
    settings = {
        "_midom_requested_resolution": "1280x720",
        "_midom_scene_integration_base_composite_path": prepared["delivery_composite_path"],
        "_midom_scene_integration_delivery_mask_path": prepared["delivery_mask_path"],
        "_midom_scene_integration_output_validation": [],
    }
    final_path = plugin._finalize_qwen21_scene_integration_production_output(
        str(generated_path), settings, 0, str(tmp_path)
    )
    final = Image.open(final_path).convert("RGB")
    base = Image.open(prepared["delivery_composite_path"]).convert("RGB")
    assert final.size == (1280, 720)
    assert final.getpixel((10, 10)) == base.getpixel((10, 10))
    assert final.getpixel((500, 400)) == (255, 0, 0)
    validation = settings["_midom_scene_integration_output_validation"][0]
    assert validation["pixel_mismatch_count"] == 0
    assert validation["maximum_channel_difference"] == 0
    assert validation["verification_result"] == "passed"


def test_flattened_production_preparation_and_restoration_preserve_protected_pixels(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    composite_path = tmp_path / "composite.png"
    matte_path = tmp_path / "matte.png"
    generated_path = tmp_path / "generated.png"
    composite = Image.new("RGB", (1280, 720), (20, 40, 60))
    composite.paste((180, 30, 20), (400, 200, 640, 600))
    composite.save(composite_path)
    matte = Image.new("L", (1280, 720), 0)
    matte.paste(255, (390, 590, 651, 661))
    matte.save(matte_path)

    prepared = plugin._prepare_qwen21_flattened_scene_integration_inputs(
        {"input_id": 805, "path": str(composite_path), "sha256": "d" * 64},
        {"input_id": 806, "path": str(matte_path), "sha256": "e" * 64},
        (1280, 720),
        (1920, 1088),
    )
    assert Image.open(prepared["internal_composite_path"]).size == (1920, 1088)
    assert Image.open(prepared["internal_mask_path"]).size == (1920, 1088)
    assert prepared["provenance"]["integration_matte"]["editable_area_fraction"] < 0.60
    Image.new("RGB", (1920, 1088), (255, 0, 0)).save(generated_path)
    settings = {
        "_midom_requested_resolution": "1280x720",
        "_midom_scene_integration_base_composite_path": prepared["delivery_composite_path"],
        "_midom_scene_integration_delivery_mask_path": prepared["delivery_mask_path"],
        "_midom_scene_integration_output_validation": [],
    }
    final_path = plugin._finalize_qwen21_scene_integration_production_output(
        str(generated_path), settings, 0, str(tmp_path)
    )
    final = Image.open(final_path).convert("RGB")
    assert final.getpixel((10, 10)) == (20, 40, 60)
    assert final.getpixel((500, 620)) == (255, 0, 0)
    validation = settings["_midom_scene_integration_output_validation"][0]
    assert validation["pixel_mismatch_count"] == 0
    assert validation["maximum_channel_difference"] == 0


@pytest.mark.parametrize(
    ("mode", "fill", "message"),
    [
        ("RGBA", (255, 255, 255, 255), "without alpha"),
        ("L", 128, "binary black/white"),
        ("L", 255, "protected black pixels"),
    ],
)
def test_flattened_production_rejects_noncanonical_or_unprotected_mattes(tmp_path, mode, fill, message):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    composite_path = tmp_path / "composite.png"
    matte_path = tmp_path / "matte.png"
    Image.new("RGB", (1280, 720), (20, 40, 60)).save(composite_path)
    Image.new(mode, (1280, 720), fill).save(matte_path)
    with pytest.raises(ValueError, match=message):
        plugin._prepare_qwen21_flattened_scene_integration_inputs(
            {"input_id": 805, "path": str(composite_path), "sha256": "d" * 64},
            {"input_id": 806, "path": str(matte_path), "sha256": "e" * 64},
            (1280, 720),
            (1920, 1088),
        )


def test_flattened_production_rejects_near_full_frame_editable_matte(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    composite_path = tmp_path / "composite.png"
    matte_path = tmp_path / "matte.png"
    Image.new("RGB", (1280, 720), (20, 40, 60)).save(composite_path)
    matte = Image.new("L", (1280, 720), 255)
    matte.paste(0, (0, 0, 10, 10))
    matte.save(matte_path)
    with pytest.raises(ValueError, match="editable matte area exceeds"):
        plugin._prepare_qwen21_flattened_scene_integration_inputs(
            {"input_id": 805, "path": str(composite_path), "sha256": "d" * 64},
            {"input_id": 806, "path": str(matte_path), "sha256": "e" * 64},
            (1280, 720),
            (1920, 1088),
        )


def test_candidate_requires_production_matte_and_standard_profile():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 2803, "worker_id": 1, "org_id": 2, "project_id": 3,
        "requested_by_user_id": 4, "media_type": "image", "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "scene_integration",
            "scene_integration_contract_version": "qwen21_scene_integration_v1",
            "integration_profile_id": "production_v1",
            "integration_mode": "edge_matte",
            "reference_mode": "scene_integration",
            "reference_image_count": 2,
            "mask_image_count": 1,
            "output_count": 1,
            "output_format": "png",
            "output_color_mode": "rgb",
            "accelerator_profile_id": "standard",
        },
    }
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["mask_image_count"] = 0
    assert "requires mask_image_count=1" in plugin._candidate_incompatibility_reason(candidate, connection(module))


def test_candidate_accepts_flattened_production_v2_contract():
    module = load_plugin_module()
    candidate = {
        "job_id": 2805, "worker_id": 1, "org_id": 2, "project_id": 3,
        "requested_by_user_id": 4, "media_type": "image", "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "scene_integration",
            "scene_integration_contract_version": "qwen21_scene_integration_v2",
            "integration_profile_id": "production_flattened_matte_v1",
            "integration_mode": "visible_edge_matte",
            "reference_mode": "scene_integration",
            "reference_image_count": 1,
            "mask_image_count": 1,
            "output_count": 1,
            "output_format": "png",
            "output_color_mode": "rgb",
            "accelerator_profile_id": "standard",
        },
    }
    plugin = plugin_instance(module)
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["reference_image_count"] = 2
    assert "requires reference_image_count=1" in plugin._candidate_incompatibility_reason(candidate, connection(module))
