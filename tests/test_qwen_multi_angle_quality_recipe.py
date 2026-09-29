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

    spec = importlib.util.spec_from_file_location("midom_bridge_multi_angle_quality_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._resolve_job_resolution = lambda job, model_id=None: "1280x720"
    plugin._qwen_multi_angle_tool_availability = lambda: {
        "available": True,
        "unavailable_reason": None,
        "lora_relative_path": module.QWEN_MULTI_ANGLE_LORA_FILENAME,
        "lora_sha256": module.QWEN_MULTI_ANGLE_LORA_SHA256,
    }
    return plugin


def multi_angle_job(strength="strong", profile_id="standard"):
    return {
        "job_id": 2511,
        "media_type": "image",
        "model_id": "qwen_image_edit_plus2_20B",
        "tool_id": "qwen_image_edit_2511_multiple_angles",
        "prompt": "Keep the same character and clothing.",
        "generation": {
            "view_azimuth": "front_left",
            "view_elevation": "eye_level",
            "shot_distance": "medium",
            "view_change_strength": strength,
            "accelerator_profile_id": profile_id,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": [
            {"kind": "reference_image", "role": "source", "sequence": 0, "input_id": 101},
        ],
        "output": {"count": 1, "format": "png", "width": 1280, "height": 720},
    }


def test_standard_multi_angle_uses_deterministic_full_quality_recipe():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    settings = plugin._validate_image_job(multi_angle_job(), 2511)

    assert settings["num_inference_steps"] == 30
    assert settings["guidance_scale"] == 4.0
    assert settings["sample_solver"] == "default"
    assert settings["model_mode"] == 0
    assert settings["denoising_strength"] == 1.0
    assert settings["override_attention"] == "sdpa"
    assert settings["prompt_enhancer"] == ""
    assert settings["spatial_upsampling"] == ""
    assert settings["_midom_output_mime_type"] == "image/png"
    assert settings["_midom_output_color_mode"] == "RGB"
    assert settings["_midom_curated_tool_recipe_version"] == "3"
    assert settings["_midom_curated_tool_lora_multiplier"] == "0.95"
    assert settings["_midom_curated_tool_recipe"]["image_output_codec"] == "png"


def test_multi_angle_strengths_stay_within_upstream_recommended_range():
    module = load_plugin_module()

    assert module.QWEN_MULTI_ANGLE_VIEW_CHANGE_STRENGTHS == {
        "subtle": "0.8",
        "standard": "0.9",
        "strong": "0.95",
        "maximum": "1.0",
    }


def test_accelerated_multi_angle_keeps_profile_steps_but_pins_safe_attention():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    def apply_fast_profile(settings, model_id, generation):
        settings["num_inference_steps"] = 8
        settings["guidance_scale"] = 1.0
        settings["sample_solver"] = "default"
        settings["_midom_accelerator_profile_id"] = "qwen_edit_2511_lightning_8"
        return "qwen_edit_2511_lightning_8"

    plugin._apply_accelerator_profile = apply_fast_profile
    settings = plugin._validate_image_job(
        multi_angle_job(profile_id="qwen_edit_2511_lightning_8"),
        2512,
    )

    assert settings["num_inference_steps"] == 8
    assert settings["guidance_scale"] == 1.0
    assert settings["override_attention"] == "sdpa"


def test_multi_angle_completion_metadata_records_recipe_and_runtime():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(multi_angle_job("standard"), 2513)
    settings["_midom_curated_tool_runtime_diagnostics"] = {
        "phase": "post_generation",
        "model_checkpoint": "qwen_image_edit_plus2_20B_quanto_bf16_int8.safetensors",
        "vae_checkpoint": "qwen_vae.safetensors",
        "attention_mode": "sdpa",
    }

    metadata = plugin._build_generation_metadata(
        settings,
        types.SimpleNamespace(seed=1234, metadata={}),
        [],
    )

    curated = metadata["curated_tool"]
    assert curated["recipe_version"] == "3"
    assert curated["expanded_recipe"]["num_inference_steps"] == 30
    assert curated["expanded_recipe"]["attention_mode"] == "sdpa"
    assert curated["runtime_diagnostics"]["vae_checkpoint"] == "qwen_vae.safetensors"


def test_multi_angle_temporarily_forces_lossless_wangp_output_and_restores_it():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(multi_angle_job("standard"), 2514)
    server_config = {"image_output_codec": "jpeg_95"}
    runtime = types.SimpleNamespace(module=types.SimpleNamespace(server_config=server_config))
    api_session = types.SimpleNamespace(_ensure_runtime=lambda: runtime)

    override_state = plugin._begin_qwen_multi_angle_lossless_output(api_session, settings)

    assert server_config["image_output_codec"] == "png"
    plugin._restore_qwen_multi_angle_output_codec(override_state)
    assert server_config["image_output_codec"] == "jpeg_95"


def test_multi_angle_rejects_non_png_output_contract():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = multi_angle_job()
    job["output"]["format"] = "jpeg"

    try:
        plugin._validate_image_job(job, 2515)
    except ValueError as exc:
        assert "output.format must be 'png'" in str(exc)
    else:
        raise AssertionError("Expected curated multi-angle JPEG output to be rejected")
