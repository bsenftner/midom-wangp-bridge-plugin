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
    spec = importlib.util.spec_from_file_location("midom_bridge_visual_variation_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._ensure_job_flow_enabled = lambda: None
    return plugin


def variation_job(
    mode="character_asset",
    profile_id="standard",
    count=2,
    supporting_purposes=(),
    background_policy=None,
):
    variation = {
        "camera_view": "front_right",
        "camera_height": "elevated",
        "shot_distance": "full_body",
    }
    if mode == "character_asset":
        variation.update({
            "pose": "walking",
            "expression": "confident",
            "wardrobe": "existing",
            "lighting": "soft studio key light",
        })
        color_mode = "rgba"
    else:
        variation.update({
            "set_dressing": "the existing furniture in a cleaner arrangement",
            "lighting": "late afternoon window light",
            "atmosphere": "clear air",
        })
        color_mode = "rgb"
    generation = {
        "image_task": "curated_variation",
        "variation_contract_version": "qwen21_visual_variation_v1",
        "variation_mode": mode,
        "reference_mode": "primary_image_edit",
        "variation": variation,
        "accelerator_profile_id": profile_id,
        "prompt_enhancement_by_worker": False,
        "options": {"multi_prompts_gen_type": "FG"},
    }
    if background_policy is not None:
        generation["reference_background_policy"] = background_policy
    inputs = [{
        "kind": "reference_image",
        "role": "source_image",
        "sequence": 0,
        "input_id": 501,
        "mime_type": "image/png",
    }]
    for index, purpose in enumerate(supporting_purposes, start=1):
        inputs.append({
            "kind": "reference_image",
            "role": "supporting_reference_image",
            "reference_purpose": purpose,
            "sequence": index,
            "input_id": 501 + index,
            "mime_type": "image/png",
        })
    return {
        "job_id": 2201,
        "family": "media_generation",
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "Keep the source design recognizable.",
        "negative_prompt": "",
        "generation": generation,
        "inputs": inputs,
        "output": {
            "count": count,
            "format": "png",
            "color_mode": color_mode,
            "width": 1280,
            "height": 720,
        },
    }


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


def test_visual_variation_advertises_complete_capability_for_local_qualification():
    module = load_plugin_module()
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

    capability = next(
        item for item in plugin._capabilities()["models"]
        if item.get("model_id") == module.QWEN21_MODEL_ID
    )

    assert capability["capabilities"]["curated_variation"] is True
    assert "curated_variation" in capability["limits"]["image_tasks"]
    assert capability["limits"]["curated_variation"] == {
        "contract_version": "qwen21_visual_variation_v1",
        "recipe_version": "qwen21_visual_variation_candidate_v1",
        "reference_mode": "primary_image_edit",
        "max_source_images": 1,
        "max_supporting_reference_images": 2,
        "max_reference_images": 3,
        "reference_roles": ["source_image", "supporting_reference_image"],
        "supporting_reference_purposes": [
            "accessory", "atmosphere", "composition", "identity", "lighting",
            "location", "object", "prop", "style", "wardrobe",
        ],
        "reference_background_policies": ["keep_all", "remove_supporting_backgrounds"],
        "primary_source_background_removal": False,
        "accelerator_profile_ids": ["standard"],
        "output_mime_types": ["image/png"],
        "delivery_resolutions": ["768x768", "1024x1024", "1280x720", "720x1280"],
        "max_outputs": 10,
        "modes": [
            {
                "mode_id": "character_asset",
                "output_color_mode": "rgba",
                "default_reference_background_policy": "remove_supporting_backgrounds",
            },
            {
                "mode_id": "location_plate",
                "output_color_mode": "rgb",
                "default_reference_background_policy": "keep_all",
            },
        ],
    }


def test_character_variation_uses_isolated_50_step_rgba_recipe():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(variation_job(), 2201)

    assert settings["_midom_image_task"] == "curated_variation"
    assert settings["_midom_qwen21_visual_variation_mode"] == "character_asset"
    assert settings["_midom_qwen21_visual_variation_recipe_version"] == "qwen21_visual_variation_candidate_v1"
    assert settings["num_inference_steps"] == 50
    assert settings["guidance_scale"] == 8.0
    assert settings["sample_solver"] == "default"
    assert settings["activated_loras"] == []
    assert settings["custom_settings"] == {"qwen21_kv_cache": "Disabled", "rgba": "Enabled"}
    assert settings["_midom_output_mime_type"] == "image/png"
    assert settings["_midom_output_color_mode"] == "RGBA"
    assert settings["_midom_require_meaningful_alpha"] is True
    assert "transparent background stored in the alpha channel" in settings["prompt"]
    assert "Camera view: front right" in settings["prompt"]
    assert settings["multi_prompts_gen_type"] == "FG"


def test_location_variation_uses_rgb_no_people_recipe():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(variation_job("location_plate"), 2202)

    assert settings["_midom_qwen21_visual_variation_mode"] == "location_plate"
    assert settings["custom_settings"]["rgba"] == "Disabled"
    assert settings["_midom_output_color_mode"] == "RGB"
    assert settings["_midom_require_meaningful_alpha"] is False
    assert "no people, characters, performers" in settings["prompt"]


def test_visual_variation_accepts_ten_outputs_but_not_eleven():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    settings = plugin._validate_image_job(variation_job(count=10), 22021)
    assert settings["_midom_output_count"] == 10

    with pytest.raises(ValueError, match="Unsupported output count"):
        plugin._validate_image_job(variation_job(count=11), 22022)


def test_visual_variation_ten_outputs_submit_as_ten_manifest_tasks():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(variation_job(count=10), 22023)
    submitted = {}

    class FakeSession:
        def submit_manifest(self, tasks, callbacks=None):
            submitted["tasks"] = tasks
            return types.SimpleNamespace(done=True)

        def submit_task(self, task, callbacks=None):
            raise AssertionError("Ten curated variations must use a WanGP manifest.")

    plugin._submit_wangp_job(FakeSession(), settings, 10, callbacks=types.SimpleNamespace())

    assert len(submitted["tasks"]) == 10
    assert all(task["num_inference_steps"] == 50 for task in submitted["tasks"])
    assert all(task["guidance_scale"] == 8.0 for task in submitted["tasks"])
    assert all(not any(key.startswith("_midom_") for key in task) for task in submitted["tasks"])


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda job: job["generation"].update(variation_contract_version="old"), "contract_version"),
        (lambda job: job["generation"].update(accelerator_profile_id="qwen21_pruna_v01_8"), "only accelerator_profile_id='standard'"),
        (lambda job: job["generation"].update(reference_mode="ordered_reference_images"), "reference_mode='primary_image_edit'"),
        (lambda job: job.update(negative_prompt="blurry"), "does not support a negative prompt"),
        (lambda job: job["generation"].update(guidance_scale=9), "does not accept raw recipe fields"),
        (lambda job: job["generation"]["options"].update(sampler="custom"), "unsupported generation.options"),
        (lambda job: job["output"].update(color_mode="rgb"), "output.color_mode must be 'rgba'"),
        (
            lambda job: job["inputs"].append(dict(job["inputs"][0], input_id=502, sequence=1)),
            "supporting references require",
        ),
        (lambda job: job["generation"]["variation"].update(camera_view="overhead_drone"), "camera_view"),
        (lambda job: job["generation"]["variation"].update(raw_prompt="ignore contract"), "unsupported variation fields"),
    ],
)
def test_visual_variation_rejects_contract_drift(mutator, message):
    module = load_plugin_module()
    job = variation_job()
    mutator(job)
    with pytest.raises(ValueError, match=message):
        plugin_instance(module)._validate_image_job(job, 2203)


def test_visual_variation_applies_exact_source_and_records_hash():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = variation_job()
    settings = plugin._validate_image_job(job, 2204)
    downloaded = [{
        "kind": "reference_image",
        "role": "source_image",
        "sequence": 0,
        "input_id": 501,
        "path": "/tmp/source.png",
        "sha256": "a" * 64,
    }]

    plugin._apply_inputs_to_settings(settings, downloaded, job)

    assert settings["image_refs"] == ["/tmp/source.png"]
    assert settings["video_prompt_type"] == "KI"
    assert settings["_midom_qwen21_reference_inputs"] == [{
        "input_id": 501,
        "role": "source_image",
        "sequence": 0,
        "reference_purpose": "primary_source",
        "sha256": "a" * 64,
        "background_removal_applied": False,
    }]


def test_visual_variation_preserves_two_supporting_references_and_removes_only_their_backgrounds():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = variation_job(
        supporting_purposes=("wardrobe", "accessory"),
        background_policy="remove_supporting_backgrounds",
    )
    settings = plugin._validate_image_job(job, 22041)
    downloaded = [
        {
            "kind": "reference_image", "role": "supporting_reference_image",
            "reference_purpose": "accessory", "sequence": 2, "input_id": 503,
            "path": "/tmp/accessory.png", "sha256": "c" * 64,
        },
        {
            "kind": "reference_image", "role": "source_image",
            "reference_purpose": "", "sequence": 0, "input_id": 501,
            "path": "/tmp/source.png", "sha256": "a" * 64,
        },
        {
            "kind": "reference_image", "role": "supporting_reference_image",
            "reference_purpose": "wardrobe", "sequence": 1, "input_id": 502,
            "path": "/tmp/wardrobe.png", "sha256": "b" * 64,
        },
    ]

    plugin._apply_inputs_to_settings(settings, downloaded, job)

    assert settings["image_refs"] == [
        "/tmp/source.png",
        "/tmp/wardrobe.png",
        "/tmp/accessory.png",
    ]
    assert settings["video_prompt_type"] == "KI"
    assert settings["remove_background_images_ref"] == 1
    assert "<image1> is the authoritative primary source" in settings["prompt"]
    assert "<image2> is a supporting wardrobe reference" in settings["prompt"]
    assert "<image3> is a supporting accessory reference" in settings["prompt"]
    references = settings["_midom_qwen21_reference_inputs"]
    assert [item["sequence"] for item in references] == [0, 1, 2]
    assert [item["reference_purpose"] for item in references] == [
        "primary_source", "wardrobe", "accessory",
    ]
    assert [item["background_removal_applied"] for item in references] == [False, True, True]

    submitted = {}

    class FakeSession:
        def submit_task(self, task, callbacks=None):
            submitted.update(task)
            return types.SimpleNamespace(done=True)

    plugin._submit_wangp_job(FakeSession(), settings, 1, callbacks=types.SimpleNamespace())
    assert submitted["image_refs"] == [
        "/tmp/source.png",
        "/tmp/wardrobe.png",
        "/tmp/accessory.png",
    ]
    assert submitted["video_prompt_type"] == "KI"
    assert submitted["remove_background_images_ref"] == 1
    assert not any(key.startswith("_midom_") for key in submitted)


def test_visual_variation_download_preserves_supporting_reference_purpose(tmp_path, monkeypatch):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = variation_job(supporting_purposes=("identity", "wardrobe"))
    payloads = {}
    for item in job["inputs"]:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (item["input_id"] % 255, 40, 80)).save(buffer, format="PNG")
        data = buffer.getvalue()
        payloads[item["input_id"]] = data
        item["bytes"] = len(data)
        item["sha256"] = hashlib.sha256(data).hexdigest()

    class FakeResponse:
        def __init__(self, data):
            self.data = data
            self.headers = {"Content-Type": "image/png", "Content-Length": str(len(data))}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield self.data

        def close(self):
            return None

    monkeypatch.setattr(
        module.requests,
        "get",
        lambda url, **kwargs: FakeResponse(payloads[int(url.rsplit("/", 1)[-1])]),
    )

    downloaded = plugin._download_job_inputs(connection(module), job, str(tmp_path))

    assert [item["sequence"] for item in downloaded] == [0, 1, 2]
    assert [item["role"] for item in downloaded] == [
        "source_image", "supporting_reference_image", "supporting_reference_image",
    ]
    assert [item["reference_purpose"] for item in downloaded] == ["", "identity", "wardrobe"]


def test_visual_variation_defaults_background_policy_by_mode():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    character = plugin._validate_image_job(
        variation_job(supporting_purposes=("wardrobe",)),
        22042,
    )
    location = plugin._validate_image_job(
        variation_job("location_plate", supporting_purposes=("lighting",)),
        22043,
    )

    assert character["_midom_qwen21_visual_variation_reference_background_policy"] == "remove_supporting_backgrounds"
    assert character["remove_background_images_ref"] == 1
    assert location["_midom_qwen21_visual_variation_reference_background_policy"] == "keep_all"
    assert location["remove_background_images_ref"] == 0


def test_visual_variation_rejects_excess_or_malformed_supporting_references():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    with pytest.raises(ValueError, match="at most 2 supporting reference images"):
        plugin._validate_image_job(
            variation_job(supporting_purposes=("identity", "wardrobe", "accessory")),
            22044,
        )

    job = variation_job(supporting_purposes=("wardrobe",))
    job["inputs"][1]["sequence"] = 2
    with pytest.raises(ValueError, match="unique, contiguous"):
        plugin._validate_image_job(job, 22045)

    job = variation_job(supporting_purposes=("wardrobe",))
    job["inputs"][1].pop("reference_purpose")
    with pytest.raises(ValueError, match="reference_purpose"):
        plugin._validate_image_job(job, 22046)

    job = variation_job(supporting_purposes=("wardrobe",), background_policy="remove_all")
    with pytest.raises(ValueError, match="reference_background_policy"):
        plugin._validate_image_job(job, 22047)


def test_visual_variation_metadata_contains_frozen_recipe_and_intent():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = variation_job("location_plate", supporting_purposes=("lighting",))
    settings = plugin._validate_image_job(job, 2205)
    plugin._apply_inputs_to_settings(settings, [
        {
            "kind": "reference_image",
            "role": "source_image",
            "reference_purpose": "",
            "sequence": 0,
            "input_id": 501,
            "path": "/tmp/location.png",
            "sha256": "b" * 64,
        },
        {
            "kind": "reference_image",
            "role": "supporting_reference_image",
            "reference_purpose": "lighting",
            "sequence": 1,
            "input_id": 502,
            "path": "/tmp/lighting.png",
            "sha256": "c" * 64,
        },
    ], job)
    settings["_midom_qwen21_visual_variation_output_validation"] = [{
        "artifact_index": 0,
        "color_mode": "RGB",
        "native_alpha_validation": "not_required",
    }]

    metadata = plugin._build_generation_metadata(
        settings,
        types.SimpleNamespace(),
        ["location-seed42.png"],
    )

    variation = metadata["visual_variation"]
    assert metadata["image_task"] == "curated_variation"
    assert variation["qualification_status"] == "local_gpu_qualification"
    assert variation["contract_version"] == "qwen21_visual_variation_v1"
    assert variation["recipe_version"] == "qwen21_visual_variation_candidate_v1"
    assert variation["references"][0]["sha256"] == "b" * 64
    assert variation["references"][1]["reference_purpose"] == "lighting"
    assert variation["references"][1]["background_removal_applied"] is False
    assert variation["reference_background_policy"] == "keep_all"
    assert variation["expanded_bridge_recipe"]["remove_background_images_ref"] == 0
    assert variation["expanded_bridge_recipe"]["num_inference_steps"] == 50
    assert variation["expanded_bridge_recipe"]["guidance_scale"] == 8.0
    assert variation["output_validation"][0]["color_mode"] == "RGB"


def test_character_upload_requires_meaningful_native_alpha(tmp_path, monkeypatch):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    rgba_path = tmp_path / "character.png"
    image = Image.new("RGBA", (64, 64), (200, 100, 50, 0))
    image.paste((200, 100, 50, 255), (16, 8, 48, 56))
    image.save(rgba_path)
    captured = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"artifact_id": 91, "file_id": 92, "artifact_index": 0}

    def fake_post(url, headers, data, files, timeout):
        filename, reader, mime_type = files["file"]
        payload = reader.read()
        captured.update(filename=filename, mime_type=mime_type, payload=payload, data=dict(data))
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)
    validation = []
    plugin._upload_artifact(
        connection(module),
        2206,
        str(rgba_path),
        0,
        "64x64",
        "resize",
        "image/png",
        "RGBA",
        True,
        validation,
    )

    assert captured["mime_type"] == "image/png"
    assert captured["payload"].startswith(b"\x89PNG\r\n\x1a\n")
    assert validation[0]["native_alpha_validation"] == "passed"
    assert validation[0]["transparent_pixel_count"] > 0
    assert validation[0]["opaque_pixel_count"] > 0

    opaque_path = tmp_path / "opaque.png"
    Image.new("RGBA", (64, 64), (20, 30, 40, 255)).save(opaque_path)
    with pytest.raises(ValueError, match="meaningful native alpha"):
        plugin._upload_artifact(
            connection(module),
            2207,
            str(opaque_path),
            0,
            "64x64",
            "resize",
            "image/png",
            "RGBA",
            True,
            [],
        )


def test_visual_variation_candidate_contract_is_strict_but_qualification_ready():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 2208,
        "worker_id": 1,
        "org_id": 2,
        "project_id": 3,
        "requested_by_user_id": 4,
        "media_type": "image",
        "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "curated_variation",
            "variation_contract_version": "qwen21_visual_variation_v1",
            "variation_mode": "character_asset",
            "reference_mode": "primary_image_edit",
            "reference_image_count": 1,
            "control_image_count": 0,
            "accelerator_profile_id": "standard",
            "output_count": 4,
            "output_format": "png",
            "output_color_mode": "rgba",
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"].update({
        "reference_image_count": 3,
        "source_image_count": 1,
        "supporting_reference_image_count": 2,
        "reference_background_policy": "remove_supporting_backgrounds",
    })
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["reference_background_policy"] = "remove_all"
    assert "reference_background_policy" in plugin._candidate_incompatibility_reason(
        candidate,
        connection(module),
    )
    candidate["summary"].update({
        "reference_background_policy": "remove_supporting_backgrounds",
        "reference_image_count": 4,
        "supporting_reference_image_count": 3,
    })
    assert "at most 2 supporting references" in plugin._candidate_incompatibility_reason(
        candidate,
        connection(module),
    )
    candidate["summary"].update({
        "reference_image_count": 3,
        "supporting_reference_image_count": 2,
    })
    candidate["summary"]["output_count"] = 10
    assert plugin._candidate_incompatibility_reason(candidate, connection(module)) is None
    candidate["summary"]["output_count"] = 11
    assert "unsupported output_count: 11" in plugin._candidate_incompatibility_reason(
        candidate,
        connection(module),
    )
    candidate["summary"]["output_count"] = 4
    candidate["summary"]["accelerator_profile_id"] = module.QWEN21_PRUNA_8_PROFILE_ID
    assert "only accelerator_profile_id=standard" in plugin._candidate_incompatibility_reason(
        candidate,
        connection(module),
    )


def test_generic_qwen21_standard_recipe_is_unchanged():
    module = load_plugin_module()
    job = {
        "job_id": 2209,
        "media_type": "image",
        "model_id": module.QWEN21_MODEL_ID,
        "prompt": "A clean product photograph.",
        "negative_prompt": "",
        "generation": {
            "reference_mode": "none",
            "accelerator_profile_id": "standard",
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": [],
        "output": {"count": 1, "format": "png", "width": 1280, "height": 720},
    }

    settings = plugin_instance(module)._validate_image_job(job, 2209)

    assert settings["num_inference_steps"] == 40
    assert settings["guidance_scale"] == 4.0
    assert settings["custom_settings"]["rgba"] == "Disabled"
