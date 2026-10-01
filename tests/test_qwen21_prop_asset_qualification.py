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
    spec = importlib.util.spec_from_file_location("midom_bridge_prop_asset_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._ensure_job_flow_enabled = lambda: None
    return plugin


def prop_job():
    return {
        "job_id": 2501,
        "family": "media_generation",
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "Keep the original handle and the brushed blue finish.",
        "negative_prompt": "",
        "generation": {
            "image_task": "curated_variation",
            "variation_contract_version": "qwen21_visual_variation_v2",
            "variation_mode": "prop_asset",
            "reference_mode": "primary_image_edit",
            "variation": {
                "camera_view": "front_right",
                "camera_height": "eye_level",
                "shot_distance": "medium",
                "object_state": "assembled",
                "use_or_configuration": "standing upright",
                "material_and_finish": "brushed metal",
                "colorway_or_surface_treatment": "matte blue",
                "accessories_or_attached_parts": "original handle",
                "lighting": "soft studio key light",
            },
            "accelerator_profile_id": "standard",
            "prompt_enhancement_by_worker": False,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": [{
            "kind": "reference_image",
            "role": "source_image",
            "sequence": 0,
            "input_id": 601,
            "mime_type": "image/png",
        }],
        "output": {
            "count": 1,
            "format": "png",
            "color_mode": "rgba",
            "width": 1280,
            "height": 720,
        },
    }


def connection(module):
    return module.ConnectionContext(
        connection_id="worker-1", api_base_url="https://midom.test", worker_id=1,
        worker_token="token", org_id=2, project_id=3, paired_user_id=4,
        machine_name="GPU", capabilities_revision=1, token_expires_at="",
        allow_insecure_local_dev=False, allow_insecure_lan_dev=False,
    )


def test_prop_asset_contract_is_advertised_alongside_v1():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    contract = plugin._qwen21_prop_asset_capability()

    assert contract["contract_version"] == "qwen21_visual_variation_v2"
    assert contract["recipe_version"] == "qwen21_prop_asset_candidate_v1"
    assert contract["max_source_images"] == 1
    assert contract["max_supporting_reference_images"] == 2
    assert contract["max_reference_images"] == 3
    assert contract["reference_roles"] == ["source_image", "supporting_reference_image"]
    assert contract["reference_background_policies"] == ["keep_all"]
    assert contract["max_outputs"] == 1
    assert contract["modes"] == [{
        "mode_id": "prop_asset",
        "output_color_mode": "rgba",
        "reference_background_policy": "keep_all",
        "alpha_validation": "transparent_and_opaque_pixels_v1",
    }]

    plugin._curated_tools_for_model = lambda _model_id: []
    plugin._audio_output_mime_types = lambda: ["audio/wav"]
    plugin._seedvc_speech_available = lambda: False
    plugin._event_video_processing_capability = lambda: None
    plugin._storyboard_ffmpeg_processing_capability = lambda: None
    plugin._ltx_video_capability = lambda model_id, display_name: {
        "model_id": model_id, "media_type": "video", "display_name": display_name,
    }
    plugin._resolve_accelerator_loras = lambda profile: ([], [])
    capability = next(item for item in plugin._capabilities()["models"] if item["model_id"] == module.QWEN21_MODEL_ID)

    assert capability["limits"]["curated_variation"]["contract_version"] == "qwen21_visual_variation_v1"
    assert "prop_asset" not in [mode["mode_id"] for mode in capability["limits"]["curated_variation"]["modes"]]
    contracts = capability["limits"]["curated_variation_contracts"]
    assert [item["contract_version"] for item in contracts] == [
        "qwen21_visual_variation_v1",
        "qwen21_visual_variation_v2",
        "qwen21_visual_variation_v3",
    ]
    assert contracts[0] == capability["limits"]["curated_variation"]
    assert contracts[1] == contract


def test_prop_asset_uses_dedicated_standard_rgba_recipe_and_prompt():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(prop_job(), 2501)

    assert settings["_midom_qwen21_visual_variation_contract_version"] == "qwen21_visual_variation_v2"
    assert settings["_midom_qwen21_visual_variation_recipe_version"] == "qwen21_prop_asset_candidate_v1"
    assert settings["_midom_qwen21_visual_variation_mode"] == "prop_asset"
    assert settings["num_inference_steps"] == 50
    assert settings["guidance_scale"] == 8.0
    assert settings["custom_settings"] == {"qwen21_kv_cache": "Disabled", "rgba": "Enabled"}
    assert settings["_midom_output_color_mode"] == "RGBA"
    assert settings["_midom_require_meaningful_alpha"] is True
    assert settings["remove_background_images_ref"] == 0
    assert "isolated prop asset" in settings["prompt"]
    assert "cast shadow" in settings["prompt"]
    assert "transparent background stored in the alpha channel" in settings["prompt"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda job: job["generation"].update(variation_contract_version="qwen21_visual_variation_v1"), "Unsupported"),
        (lambda job: job["generation"].update(variation_mode="dressed_set"), "Unsupported"),
        (lambda job: job["generation"].update(accelerator_profile_id="qwen21_pruna_v01_8"), "only accelerator_profile_id='standard'"),
        (lambda job: job["generation"].update(reference_background_policy="remove_supporting_backgrounds"), "background removal"),
        (lambda job: job["output"].update(color_mode="rgb"), "output.color_mode"),
        (lambda job: job["output"].update(count=2), "output.count=1"),
        (lambda job: job["inputs"].extend([
            {"kind": "reference_image", "role": "supporting_reference_image", "sequence": 1, "input_id": 602, "reference_purpose": "style"},
            {"kind": "reference_image", "role": "supporting_reference_image", "sequence": 2, "input_id": 603, "reference_purpose": "style"},
            {"kind": "reference_image", "role": "supporting_reference_image", "sequence": 3, "input_id": 604, "reference_purpose": "object"},
        ]), "at most two supporting"),
        (lambda job: job["inputs"][0].update(kind="mask_image"), "accepts only reference_image"),
    ],
)
def test_prop_asset_rejects_contract_drift(mutator, message):
    module = load_plugin_module()
    job = prop_job()
    mutator(job)
    with pytest.raises(ValueError, match=message):
        plugin_instance(module)._validate_image_job(job, 2502)


def test_prop_asset_preserves_ordered_references_and_records_sha256_provenance():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(prop_job(), 2503)
    job = prop_job()
    job["inputs"].extend([
        {
            "kind": "reference_image", "role": "supporting_reference_image", "sequence": 1,
            "input_id": 602, "reference_purpose": "style", "mime_type": "image/png",
        },
        {
            "kind": "reference_image", "role": "supporting_reference_image", "sequence": 2,
            "input_id": 603, "reference_purpose": "accessory", "mime_type": "image/png",
        },
    ])
    settings = plugin._validate_image_job(job, 2503)
    plugin._apply_inputs_to_settings(settings, [
        {
            "kind": "reference_image", "role": "supporting_reference_image", "sequence": 2,
            "reference_purpose": "accessory", "input_id": 603,
            "path": "/tmp/accessory.png", "sha256": "c" * 64,
        },
        {
            "kind": "reference_image", "role": "source_image", "sequence": 0,
            "input_id": 601, "path": "/tmp/prop.png", "sha256": "a" * 64,
        },
        {
            "kind": "reference_image", "role": "supporting_reference_image", "sequence": 1,
            "reference_purpose": "style", "input_id": 602,
            "path": "/tmp/material.png", "sha256": "b" * 64,
        },
    ], job)

    assert settings["image_refs"] == ["/tmp/prop.png", "/tmp/material.png", "/tmp/accessory.png"]
    assert settings["video_prompt_type"] == "KI"
    assert settings["_midom_qwen21_reference_inputs"] == [
        {
            "input_id": 601, "role": "source_image", "sequence": 0,
            "reference_purpose": "primary_source", "sha256": "a" * 64,
            "background_removal_applied": False,
        },
        {
            "input_id": 602, "role": "supporting_reference_image", "sequence": 1,
            "reference_purpose": "style", "sha256": "b" * 64,
            "background_removal_applied": False,
        },
        {
            "input_id": 603, "role": "supporting_reference_image", "sequence": 2,
            "reference_purpose": "accessory", "sha256": "c" * 64,
            "background_removal_applied": False,
        },
    ]

    settings["_midom_qwen21_visual_variation_output_validation"] = [{
        "artifact_index": 0, "color_mode": "RGBA", "native_alpha_validation": "passed",
        "transparent_pixel_count": 100, "opaque_pixel_count": 200,
    }]
    metadata = plugin._build_generation_metadata(settings, types.SimpleNamespace(), ["prop.png"])
    variation = metadata["visual_variation"]
    assert variation["variation_mode"] == "prop_asset"
    assert [reference["sha256"] for reference in variation["references"]] == [
        "a" * 64, "b" * 64, "c" * 64,
    ]
    assert variation["output_color_mode"] == "RGBA"
    assert variation["alpha_required"] is True
    assert variation["output_validation"][0]["native_alpha_validation"] == "passed"


def test_prop_asset_candidate_supports_ordered_references_and_rejects_excess():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 2504,
        "worker_id": 1,
        "org_id": 2,
        "project_id": 3,
        "requested_by_user_id": 4,
        "media_type": "image", "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "curated_variation",
            "variation_contract_version": "qwen21_visual_variation_v2",
            "variation_mode": "prop_asset",
            "reference_mode": "primary_image_edit",
            "reference_image_count": 1,
            "source_image_count": 1,
            "supporting_reference_image_count": 0,
            "control_image_count": 0,
            "accelerator_profile_id": "standard",
            "output_count": 1,
            "output_format": "png",
            "output_color_mode": "rgba",
        },
    }
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["supporting_reference_image_count"] = 2
    candidate["summary"]["reference_image_count"] = 3
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["supporting_reference_image_count"] = 3
    candidate["summary"]["reference_image_count"] = 4
    assert "at most two supporting references" in plugin._candidate_incompatibility_reason(candidate, connection(module))
