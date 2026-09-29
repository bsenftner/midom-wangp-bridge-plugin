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
    spec = importlib.util.spec_from_file_location("midom_bridge_qwen21_rgba_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._ensure_job_flow_enabled = lambda: None
    return plugin


def connection(module):
    return module.ConnectionContext(
        connection_id="worker-1",
        api_base_url="https://midom.test",
        worker_id=1,
        worker_token="token",
        org_id=2,
        project_id=3,
        paired_user_id=4,
        machine_name="GPU",
        capabilities_revision=1,
        token_expires_at="",
        allow_insecure_local_dev=False,
        allow_insecure_lan_dev=False,
    )


def rgba_job(
    task="generate",
    reference_mode="none",
    reference_count=0,
    resolution=(1024, 1024),
    profile_id="standard",
):
    inputs = []
    for sequence in range(reference_count):
        inputs.append({
            "kind": "reference_image",
            "role": "source_image" if reference_mode == "primary_image_edit" else "ordered_reference_image",
            "sequence": sequence,
            "input_id": 700 + sequence,
            "mime_type": "image/png",
        })
    return {
        "job_id": 2701,
        "family": "media_generation",
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "A detailed clockwork robot with a clean silhouette.",
        "negative_prompt": "blurry",
        "generation": {
            "image_task": task,
            "reference_mode": reference_mode,
            "transparent_output_contract_version": "qwen21_native_rgba_v1",
            "transparent_output_mode": "isolated_asset",
            "accelerator_profile_id": profile_id,
            "prompt_enhancement_by_worker": False,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": inputs,
        "output": {
            "count": 2,
            "format": "png",
            "color_mode": "rgba",
            "width": resolution[0],
            "height": resolution[1],
        },
    }


def prepare_capability_plugin(module):
    plugin = plugin_instance(module)
    plugin._curated_tools_for_model = lambda model_id: []
    plugin._audio_output_mime_types = lambda: ["audio/wav"]
    plugin._seedvc_speech_available = lambda: False
    plugin._event_video_processing_capability = lambda: None
    plugin._storyboard_ffmpeg_processing_capability = lambda: None
    plugin._ltx_video_capability = lambda model_id, display_name: {
        "model_id": model_id,
        "media_type": "video",
        "display_name": display_name,
    }
    plugin._resolve_accelerator_loras = lambda profile: ([], [])
    return plugin


def test_native_rgba_capability_is_advertised_for_local_gpu_qualification():
    module = load_plugin_module()
    plugin = prepare_capability_plugin(module)
    contract = plugin._qwen21_native_rgba_capability()

    assert contract["contract_version"] == "qwen21_native_rgba_v1"
    assert contract["modes"] == ["isolated_asset"]
    assert contract["accelerator_profile_ids"] == ["standard"]
    assert contract["max_outputs"] == 6
    assert contract["routes"][2]["max_reference_images"] == 3

    capability = next(
        item for item in plugin._capabilities()["models"]
        if item.get("model_id") == module.QWEN21_MODEL_ID
    )
    assert capability["capabilities"]["rgba"] is True
    assert capability["limits"]["rgba_output"] == contract


@pytest.mark.parametrize("resolution", [(768, 768), (1024, 1024), (1280, 720), (720, 1280)])
def test_native_rgba_text_to_image_uses_job_scoped_native_alpha(resolution):
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(
        rgba_job(resolution=resolution),
        2701,
    )

    assert settings["custom_settings"] == {"qwen21_kv_cache": "Disabled", "rgba": "Enabled"}
    assert settings["num_inference_steps"] == 40
    assert settings["guidance_scale"] == 4.0
    assert settings["_midom_output_mime_type"] == "image/png"
    assert settings["_midom_output_color_mode"] == "RGBA"
    assert settings["_midom_require_meaningful_alpha"] is True
    assert settings["_midom_qwen21_native_rgba_user_prompt"].startswith("A detailed")
    assert settings["prompt"].startswith("This is an RGBA image with transparency.")
    assert settings["prompt"].endswith("The image has an alpha channel and the background is transparent.")


def test_native_rgba_primary_edit_requires_source_role_and_preserves_hash_provenance():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = rgba_job("edit", "primary_image_edit", 1)
    settings = plugin._validate_image_job(job, 2702)
    plugin._apply_inputs_to_settings(settings, [{
        "kind": "reference_image",
        "role": "source_image",
        "sequence": 0,
        "input_id": 700,
        "path": "/tmp/source.png",
        "sha256": "a" * 64,
    }], job)

    assert settings["video_prompt_type"] == "KI"
    assert settings["image_refs"] == ["/tmp/source.png"]
    assert settings["_midom_qwen21_reference_inputs"][0]["sha256"] == "a" * 64

    invalid = rgba_job("edit", "primary_image_edit", 1)
    invalid["inputs"][0]["role"] = "ordered_reference_image"
    with pytest.raises(ValueError, match="role='source_image'"):
        plugin._validate_image_job(invalid, 2703)


def test_native_rgba_ordered_edit_accepts_three_ordered_references_and_rejects_four():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = rgba_job("edit", "ordered_reference_images", 3)
    settings = plugin._validate_image_job(job, 2704)
    downloaded = [{
        **descriptor,
        "path": f"/tmp/ref-{descriptor['sequence']}.png",
        "sha256": str(descriptor["sequence"]) * 64,
    } for descriptor in job["inputs"]]
    plugin._apply_inputs_to_settings(settings, downloaded, job)

    assert settings["video_prompt_type"] == "I"
    assert [item["input_id"] for item in settings["_midom_qwen21_reference_inputs"]] == [700, 701, 702]

    with pytest.raises(ValueError, match="supports one to 3 references"):
        plugin._validate_image_job(rgba_job("edit", "ordered_reference_images", 4), 2705)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda job: job["generation"].pop("transparent_output_contract_version"), "contract_version"),
        (lambda job: job["generation"].update(transparent_output_mode="unknown"), "isolated_asset"),
        (lambda job: job["generation"].update(accelerator_profile_id="qwen21_pruna_v01_8"), "only accelerator_profile_id='standard'"),
        (lambda job: job["output"].update(format="jpeg"), "output.format must be 'png'"),
        (lambda job: job["output"].update(color_mode="rgb"), "output.color_mode='rgba'"),
        (lambda job: job["generation"].update(image_task="masked_edit"), "not supported for masked_edit"),
        (lambda job: job["generation"].update(image_task="outpaint"), "not supported for outpaint"),
    ],
)
def test_native_rgba_rejects_unsupported_contract_combinations(mutator, message):
    module = load_plugin_module()
    job = rgba_job()
    mutator(job)
    with pytest.raises(ValueError, match=message):
        plugin_instance(module)._validate_image_job(job, 2706)


def test_rgba_color_mode_alone_does_not_activate_native_alpha():
    module = load_plugin_module()
    job = rgba_job()
    job["generation"].pop("transparent_output_contract_version")
    job["generation"].pop("transparent_output_mode")
    with pytest.raises(ValueError, match="contract_version"):
        plugin_instance(module)._validate_image_job(job, 2707)


def test_native_rgba_uses_existing_six_candidate_limit():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = rgba_job()
    job["output"]["count"] = 6
    assert plugin._validate_image_job(job, 27071)["_midom_output_count"] == 6
    job["output"]["count"] = 7
    with pytest.raises(ValueError, match="Unsupported output count"):
        plugin._validate_image_job(job, 27072)


def test_generic_rgb_qwen_job_remains_unchanged():
    module = load_plugin_module()
    job = rgba_job()
    job["generation"].pop("transparent_output_contract_version")
    job["generation"].pop("transparent_output_mode")
    job["output"]["color_mode"] = "rgb"
    settings = plugin_instance(module)._validate_image_job(job, 2708)

    assert settings["custom_settings"]["rgba"] == "Disabled"
    assert settings["prompt"] == job["prompt"]
    assert not settings.get("_midom_require_meaningful_alpha")


def test_native_rgba_candidate_compatibility_requires_complete_contract():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 2709,
        "worker_id": 1,
        "org_id": 2,
        "project_id": 3,
        "requested_by_user_id": 4,
        "media_type": "image",
        "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "edit",
            "reference_mode": "ordered_reference_images",
            "reference_image_count": 3,
            "control_image_count": 0,
            "transparent_output_contract_version": "qwen21_native_rgba_v1",
            "transparent_output_mode": "isolated_asset",
            "accelerator_profile_id": "standard",
            "output_count": 2,
            "output_format": "png",
            "output_color_mode": "rgba",
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["reference_image_count"] = 4
    assert "one to 3 references" in plugin._candidate_incompatibility_reason(candidate, connection(module))
    candidate["summary"]["reference_image_count"] = 3
    candidate["summary"].pop("transparent_output_contract_version")
    assert "contract version" in plugin._candidate_incompatibility_reason(candidate, connection(module))


def test_native_rgba_upload_crops_internal_canvas_and_reports_alpha_statistics(tmp_path, monkeypatch):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    source_path = tmp_path / "native-rgba.png"
    image = Image.new("RGBA", (1280, 736), (0, 0, 0, 0))
    image.paste((200, 120, 40, 255), (240, 80, 1040, 680))
    image.putpixel((240, 80), (200, 120, 40, 128))
    image.save(source_path, format="PNG")
    captured = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"artifact_id": 11, "file_id": 12, "artifact_index": 0}

    def fake_post(url, headers, data, files, timeout):
        filename, reader, mime_type = files["file"]
        payload = reader.read()
        with Image.open(io.BytesIO(payload)) as uploaded:
            captured.update(
                filename=filename,
                mime_type=mime_type,
                payload=payload,
                format=uploaded.format,
                mode=uploaded.mode,
                size=uploaded.size,
            )
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)
    validation = []
    plugin._upload_artifact(
        connection(module),
        2710,
        str(source_path),
        0,
        "1280x720",
        "center_crop_downscale",
        "image/png",
        "RGBA",
        True,
        validation,
        "1280x736",
    )

    assert captured["format"] == "PNG"
    assert captured["mode"] == "RGBA"
    assert captured["size"] == (1280, 720)
    assert validation[0]["zero_alpha_pixel_count"] > 0
    assert validation[0]["partial_alpha_pixel_count"] == 1
    assert validation[0]["opaque_pixel_count"] > 0
    assert validation[0]["total_pixel_count"] == 1280 * 720
    assert validation[0]["normalization_applied"] is True
    assert validation[0]["final_sha256"] == hashlib.sha256(captured["payload"]).hexdigest()


def test_native_rgba_upload_rejects_opaque_rgb_and_unexpected_dimensions(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    opaque_path = tmp_path / "opaque.png"
    Image.new("RGBA", (64, 64), (20, 30, 40, 255)).save(opaque_path)
    with pytest.raises(ValueError, match="meaningful native alpha"):
        plugin._upload_artifact(
            connection(module), 2711, str(opaque_path), 0, "64x64", "resize",
            "image/png", "RGBA", True, [], "64x64",
        )

    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(rgb_path)
    with pytest.raises(ValueError, match="meaningful native alpha"):
        plugin._upload_artifact(
            connection(module), 2712, str(rgb_path), 0, "64x64", "resize",
            "image/png", "RGBA", True, [], "64x64",
        )

    wrong_path = tmp_path / "wrong.png"
    wrong = Image.new("RGBA", (80, 64), (0, 0, 0, 0))
    wrong.paste((20, 30, 40, 255), (10, 10, 60, 50))
    wrong.save(wrong_path)
    with pytest.raises(ValueError, match="dimensions do not match"):
        plugin._upload_artifact(
            connection(module), 2713, str(wrong_path), 0, "64x64", "resize",
            "image/png", "RGBA", True, [], "64x64",
        )

    malformed_path = tmp_path / "malformed.png"
    malformed_path.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="could not be decoded"):
        plugin._upload_artifact(
            connection(module), 2714, str(malformed_path), 0, "64x64", "resize",
            "image/png", "RGBA", True, [], "64x64",
        )


def test_native_rgba_completion_metadata_contains_recipe_runtime_crop_and_alpha():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = rgba_job("edit", "primary_image_edit", 1, resolution=(1280, 720))
    settings = plugin._validate_image_job(job, 2715)
    settings["_midom_qwen21_reference_inputs"] = [{
        "input_id": 700,
        "role": "source_image",
        "sequence": 0,
        "sha256": "d" * 64,
    }]
    settings["_midom_qwen21_native_rgba_runtime_diagnostics"] = {
        "phase": "post_generation",
        "model_checkpoint": "qwen_image_21_7B_int8.safetensors",
        "transformer_quantization": "int8",
        "loaded_vae_dtype": "torch.bfloat16",
    }
    settings["_midom_qwen21_native_rgba_output_validation"] = [{
        "artifact_index": 0,
        "zero_alpha_pixel_count": 100,
        "partial_alpha_pixel_count": 20,
        "opaque_pixel_count": 800,
        "total_pixel_count": 920,
        "final_sha256": "e" * 64,
    }]

    metadata = plugin._build_generation_metadata(
        settings,
        types.SimpleNamespace(seed=42),
        ["result-seed42.png"],
    )

    rgba = metadata["native_rgba"]
    assert metadata["image_task"] == "edit"
    assert rgba["contract_version"] == "qwen21_native_rgba_v1"
    assert rgba["references"][0]["sha256"] == "d" * 64
    assert rgba["expanded_bridge_recipe"]["rgba"] == "Enabled"
    assert rgba["delivery_crop"]["crop_y"] == 8
    assert rgba["runtime_diagnostics"]["transformer_quantization"] == "int8"
    assert rgba["output_validation"][0]["final_sha256"] == "e" * 64
