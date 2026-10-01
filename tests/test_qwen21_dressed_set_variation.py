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
    spec = importlib.util.spec_from_file_location("midom_bridge_dressed_set_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._ensure_job_flow_enabled = lambda: None
    return plugin


def dressed_set_job():
    return {
        "job_id": 2601,
        "family": "media_generation",
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "Keep the theatrical staging coherent.",
        "negative_prompt": "",
        "generation": {
            "image_task": "curated_variation",
            "variation_contract_version": "qwen21_visual_variation_v3",
            "variation_mode": "dressed_set",
            "reference_mode": "primary_image_edit",
            "accelerator_profile_id": "standard",
            "prompt_enhancement_by_worker": False,
            "options": {"multi_prompts_gen_type": "FG"},
            "variation": {
                "camera_view": "front_right",
                "camera_height": "eye_level",
                "shot_distance": "wide",
                "character_blocking": "two performers stand at center stage",
                "character_behavior_or_action": "speaking to each other",
                "interaction_and_prop_use": "one performer holds a lantern",
                "wardrobe_or_character_state": "period costumes remain consistent",
                "set_arrangement": "the full stage remains visible",
                "set_dressing": "theater scenery and furniture remain coherent",
                "active_props_and_prop_placement": "lantern remains in the foreground",
                "lighting": "warm theatrical key light",
                "time_of_day": "evening",
                "weather": "none",
                "atmosphere": "clean indoor stage air",
            },
        },
        "inputs": [{
            "kind": "reference_image", "role": "source_image", "sequence": 0,
            "input_id": 701, "mime_type": "image/png",
        }],
        "output": {
            "count": 1, "format": "png", "color_mode": "rgb",
            "width": 1280, "height": 720,
        },
    }


def connection(module):
    return module.ConnectionContext(
        connection_id="worker-1", api_base_url="https://midom.test", worker_id=1,
        worker_token="token", org_id=2, project_id=3, paired_user_id=4,
        machine_name="GPU", capabilities_revision=1, token_expires_at="",
        allow_insecure_local_dev=False, allow_insecure_lan_dev=False,
    )


def test_dressed_set_is_immediately_advertised_as_its_own_v3_contract():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    contract = plugin._qwen21_dressed_set_capability()

    assert contract == {
        "contract_version": "qwen21_visual_variation_v3",
        "recipe_version": "qwen21_dressed_set_candidate_v1",
        "reference_mode": "primary_image_edit",
        "max_source_images": 1,
        "max_supporting_reference_images": 2,
        "max_reference_images": 3,
        "reference_roles": ["source_image", "supporting_reference_image"],
        "supporting_reference_purposes": [
            "accessory", "atmosphere", "composition", "identity", "lighting", "location",
            "object", "prop", "style", "wardrobe",
        ],
        "reference_background_policies": ["keep_all"],
        "accelerator_profile_ids": ["standard"],
        "output_mime_types": ["image/png"],
        "delivery_resolutions": ["768x768", "1024x1024", "1280x720", "720x1280"],
        "max_outputs": 1,
        "modes": [{
            "mode_id": "dressed_set", "output_color_mode": "rgb",
            "reference_background_policy": "keep_all",
        }],
    }
    contracts = plugin._qwen21_visual_variation_contracts()
    assert [item["contract_version"] for item in contracts] == [
        "qwen21_visual_variation_v1",
        "qwen21_visual_variation_v2",
        "qwen21_visual_variation_v3",
    ]
    assert contracts[2] == contract


def test_dressed_set_uses_its_own_rgb_scene_recipe():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(dressed_set_job(), 2601)

    assert settings["_midom_qwen21_visual_variation_contract_version"] == "qwen21_visual_variation_v3"
    assert settings["_midom_qwen21_visual_variation_recipe_version"] == "qwen21_dressed_set_candidate_v1"
    assert settings["_midom_qwen21_visual_variation_mode"] == "dressed_set"
    assert settings["num_inference_steps"] == 50
    assert settings["guidance_scale"] == 8.0
    assert settings["custom_settings"] == {"qwen21_kv_cache": "Disabled", "rgba": "Disabled"}
    assert settings["_midom_output_color_mode"] == "RGB"
    assert settings["_midom_require_meaningful_alpha"] is False
    assert settings["remove_background_images_ref"] == 0
    assert "one coherent generative scene" in settings["prompt"]
    assert "not an isolated asset" in settings["prompt"]
    assert "duplicate people" in settings["prompt"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda job: job["generation"].update(variation_contract_version="qwen21_visual_variation_v2"), "Unsupported"),
        (lambda job: job["generation"].update(variation_mode="location_plate"), "Unsupported"),
        (lambda job: job["generation"].update(accelerator_profile_id="qwen21_pruna_v01_8"), "only accelerator_profile_id='standard'"),
        (lambda job: job["generation"].update(reference_background_policy="remove_supporting_backgrounds"), "background removal"),
        (lambda job: job["output"].update(color_mode="rgba"), "output.color_mode"),
        (lambda job: job["output"].update(count=2), "output.count=1"),
        (lambda job: job["inputs"].extend([
            {"kind": "reference_image", "role": "supporting_reference_image", "sequence": 1, "input_id": 702, "reference_purpose": "location"},
            {"kind": "reference_image", "role": "supporting_reference_image", "sequence": 2, "input_id": 703, "reference_purpose": "prop"},
            {"kind": "reference_image", "role": "supporting_reference_image", "sequence": 3, "input_id": 704, "reference_purpose": "lighting"},
        ]), "at most two supporting"),
        (lambda job: job["inputs"][0].update(kind="control_image"), "accepts only reference_image"),
        (lambda job: job["generation"]["variation"].update(raw_ffmpeg="no"), "unsupported variation fields"),
    ],
)
def test_dressed_set_rejects_contract_drift(mutator, message):
    module = load_plugin_module()
    job = dressed_set_job()
    mutator(job)
    with pytest.raises(ValueError, match=message):
        plugin_instance(module)._validate_image_job(job, 2602)


def test_dressed_set_preserves_ordered_reference_hashes_and_rgb_provenance():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = dressed_set_job()
    job["inputs"].extend([
        {
            "kind": "reference_image", "role": "supporting_reference_image", "sequence": 1,
            "input_id": 702, "reference_purpose": "prop", "mime_type": "image/png",
        },
        {
            "kind": "reference_image", "role": "supporting_reference_image", "sequence": 2,
            "input_id": 703, "reference_purpose": "lighting", "mime_type": "image/png",
        },
    ])
    settings = plugin._validate_image_job(job, 2603)
    plugin._apply_inputs_to_settings(settings, [
        {
            "kind": "reference_image", "role": "supporting_reference_image", "sequence": 2,
            "reference_purpose": "lighting", "input_id": 703,
            "path": "/tmp/lighting.png", "sha256": "d" * 64,
        },
        {
            "kind": "reference_image", "role": "source_image", "sequence": 0,
            "input_id": 701, "path": "/tmp/staged-scene.png", "sha256": "b" * 64,
        },
        {
            "kind": "reference_image", "role": "supporting_reference_image", "sequence": 1,
            "reference_purpose": "prop", "input_id": 702,
            "path": "/tmp/lantern.png", "sha256": "c" * 64,
        },
    ], job)

    assert settings["image_refs"] == ["/tmp/staged-scene.png", "/tmp/lantern.png", "/tmp/lighting.png"]
    assert settings["video_prompt_type"] == "KI"
    assert [reference["sha256"] for reference in settings["_midom_qwen21_reference_inputs"]] == [
        "b" * 64, "c" * 64, "d" * 64,
    ]
    assert all(
        reference["background_removal_applied"] is False
        for reference in settings["_midom_qwen21_reference_inputs"]
    )
    settings["_midom_qwen21_visual_variation_output_validation"] = [{
        "artifact_index": 0, "color_mode": "RGB", "final_sha256": "c" * 64,
        "native_alpha_validation": "not_required",
    }]
    metadata = plugin._build_generation_metadata(settings, types.SimpleNamespace(), ["scene.png"])
    variation = metadata["visual_variation"]
    assert variation["variation_mode"] == "dressed_set"
    assert variation["recipe_version"] == "qwen21_dressed_set_candidate_v1"
    assert [reference["sha256"] for reference in variation["references"]] == [
        "b" * 64, "c" * 64, "d" * 64,
    ]
    assert variation["output_color_mode"] == "RGB"
    assert variation["alpha_required"] is False
    assert variation["output_validation"][0]["final_sha256"] == "c" * 64


def test_dressed_set_candidate_supports_ordered_references_and_rejects_excess():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 2604, "worker_id": 1, "org_id": 2, "project_id": 3,
        "requested_by_user_id": 4, "media_type": "image", "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "curated_variation",
            "variation_contract_version": "qwen21_visual_variation_v3",
            "variation_mode": "dressed_set",
            "reference_mode": "primary_image_edit",
            "reference_image_count": 1,
            "source_image_count": 1,
            "supporting_reference_image_count": 0,
            "control_image_count": 0,
            "accelerator_profile_id": "standard",
            "output_count": 1,
            "output_format": "png",
            "output_color_mode": "rgb",
        },
    }
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["supporting_reference_image_count"] = 2
    candidate["summary"]["reference_image_count"] = 3
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["supporting_reference_image_count"] = 3
    candidate["summary"]["reference_image_count"] = 4
    assert "at most two supporting references" in plugin._candidate_incompatibility_reason(candidate, connection(module))
    candidate["summary"]["supporting_reference_image_count"] = 0
    candidate["summary"]["reference_image_count"] = 1
    candidate["summary"]["output_color_mode"] = "rgba"
    assert "requires output color mode rgb" in plugin._candidate_incompatibility_reason(candidate, connection(module))
