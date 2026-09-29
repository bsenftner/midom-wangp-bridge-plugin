import importlib.util
import hashlib
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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_qwen21_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    return plugin


def qwen21_job(reference_mode="none", reference_count=0, profile_id="standard", resolution=(1280, 720)):
    return {
        "job_id": 2101,
        "media_type": "image",
        "model_id": "qwen_image_21_7B",
        "prompt": "Create one cohesive editorial image using the ordered references.",
        "negative_prompt": "",
        "generation": {
            "reference_mode": reference_mode,
            "accelerator_profile_id": profile_id,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "output": {
            "count": 1,
            "format": "png",
            "width": resolution[0],
            "height": resolution[1],
        },
        "inputs": [
            {
                "kind": "reference_image",
                "input_id": 1000 + index,
                "sequence": index,
                "mime_type": "image/png",
            }
            for index in range(reference_count)
        ],
    }


def qwen21_masked_edit_job(
    method="masked_denoising",
    resolution=(1280, 720),
    allow_full_image_edit=False,
    supporting_reference_count=0,
):
    job = qwen21_job("primary_image_edit", 1, "standard", resolution)
    job["prompt"] = "Replace the sign with a blue sign reading OPEN."
    job["generation"].update({
        "image_task": "masked_edit",
        "masked_edit_contract_version": "qwen21_masked_edit_v1",
        "masked_edit_method": method,
        "edit_strength": "balanced",
        "allow_full_image_edit": allow_full_image_edit,
        "mask_semantics": "luminance_white_edit_v1",
    })
    job["inputs"][0]["role"] = "source_image"
    for index in range(supporting_reference_count):
        job["inputs"].append({
            "kind": "reference_image",
            "role": "supporting_reference_image",
            "sequence": index + 1,
            "input_id": 1100 + index,
            "mime_type": "image/png",
        })
    job["inputs"].append({
        "kind": "mask_image",
        "role": "edit_mask",
        "input_id": 2000,
        "mime_type": "image/png",
    })
    return job


def qwen21_outpaint_job(
    source_size=(1024, 1024),
    resolution=(1280, 720),
    placement=None,
    output_count=1,
):
    source_width, source_height = source_size
    output_width, output_height = resolution
    if source_width <= output_width and source_height <= output_height:
        fitted_width, fitted_height = source_width, source_height
    elif output_width * source_height <= output_height * source_width:
        fitted_width = output_width
        fitted_height = max(1, int(round(source_height * output_width / source_width)))
    else:
        fitted_height = output_height
        fitted_width = max(1, int(round(source_width * output_height / source_height)))
    if placement is None:
        placement = {
            "x": (output_width - fitted_width) // 2,
            "y": (output_height - fitted_height) // 2,
            "width": fitted_width,
            "height": fitted_height,
        }
    job = qwen21_job("primary_image_edit", 1, "standard", resolution)
    job["prompt"] = "Continue the room naturally beyond the existing photograph."
    job["output"]["count"] = output_count
    job["generation"].update({
        "image_task": "outpaint",
        "outpaint_contract_version": "qwen21_outpaint_v1",
        "placement_mode": "explicit_rectangle",
        "source_scale_mode": "fit_without_crop_no_upscale",
        "source_width": source_width,
        "source_height": source_height,
        "source_placement": dict(placement),
        "prompt_enhancement_by_worker": False,
    })
    job["inputs"][0].update({
        "role": "source_image",
        "source_width": source_width,
        "source_height": source_height,
    })
    return job


def enable_viggle(plugin, module):
    plugin._resolve_accelerator_loras = lambda profile: (
        [module.QWEN21_VIGGLE_LORA_FILENAME],
        [],
    )


def qwen21_accelerator_filename(module, profile_id):
    return {
        module.QWEN21_VIGGLE_PROFILE_ID: module.QWEN21_VIGGLE_LORA_FILENAME,
        module.QWEN21_PRUNA_8_PROFILE_ID: module.QWEN21_PRUNA_8_LORA_FILENAME,
        module.QWEN21_PRUNA_5_PROFILE_ID: module.QWEN21_PRUNA_5_LORA_FILENAME,
    }[profile_id]


def enable_qwen21_accelerators(plugin, module):
    plugin._resolve_accelerator_loras = lambda profile: (
        [qwen21_accelerator_filename(module, profile["profile_id"])]
        if profile.get("profile_id") in module.QWEN21_ACCELERATOR_PROFILE_IDS
        else [],
        [],
    )


def test_qwen21_capability_reports_first_pass_contract():
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
    enable_qwen21_accelerators(plugin, module)

    capability = next(
        item for item in plugin._capabilities()["models"]
        if item.get("model_id") == module.QWEN21_MODEL_ID
    )

    assert capability["display_name"] == "Qwen Image 2.1 7B"
    assert capability["capabilities"]["image_edit"] is True
    assert capability["capabilities"]["ordered_reference_images"] is True
    assert capability["capabilities"]["control"] is False
    assert capability["capabilities"]["inpaint"] is True
    assert capability["capabilities"]["masked_edit"] is True
    assert capability["capabilities"]["outpaint"] is True
    assert capability["capabilities"]["curated_variation"] is True
    assert capability["capabilities"]["rgba"] is True
    assert capability["limits"]["rgba_output"] == plugin._qwen21_native_rgba_capability()
    assert capability["limits"]["max_reference_images"] == 10
    assert capability["limits"]["output_mime_types"] == ["image/png"]
    assert capability["limits"]["internal_render_resolutions"]["1280x720"] == "1280x736"
    masked_edit = capability["limits"]["masked_edit"]
    assert masked_edit["contract_version"] == "qwen21_masked_edit_v1"
    assert "max_reference_images" not in masked_edit
    assert masked_edit["max_supporting_reference_images"] == 1
    assert masked_edit["methods"] == ["lanpaint_5", "masked_denoising"]
    assert masked_edit["mask_semantics"] == ["luminance_white_edit_v1"]
    assert masked_edit["accelerator_profile_ids"] == ["standard"]
    assert masked_edit["output_color_mode"] == "RGB"
    assert capability["limits"]["image_tasks"] == [
        "generate",
        "edit",
        "masked_edit",
        "outpaint",
        "curated_variation",
    ]
    outpaint = capability["limits"]["outpaint"]
    assert outpaint == {
        "contract_version": "qwen21_outpaint_v1",
        "reference_mode": "primary_image_edit",
        "max_source_images": 1,
        "placement_modes": ["explicit_rectangle"],
        "source_scale_modes": ["fit_without_crop_no_upscale"],
        "accelerator_profile_ids": ["standard"],
        "output_modes": ["rgb"],
        "output_mime_types": ["image/png"],
        "delivery_resolutions": ["768x768", "1024x1024", "1280x720", "720x1280"],
    }
    assert "modes" not in outpaint
    assert "preserve_original" not in outpaint
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
    profiles = {item["profile_id"]: item for item in capability["accelerator_profiles"]}
    assert profiles["standard"]["steps"] == 40
    assert profiles["standard"]["max_reference_images"] == 10
    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["steps"] == 6
    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["max_reference_images"] == 3
    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["negative_prompt"] is False
    assert profiles[module.QWEN21_PRUNA_8_PROFILE_ID]["steps"] == 8
    assert profiles[module.QWEN21_PRUNA_8_PROFILE_ID]["max_reference_images"] == 3
    assert profiles[module.QWEN21_PRUNA_8_PROFILE_ID]["negative_prompt"] is False
    assert profiles[module.QWEN21_PRUNA_8_PROFILE_ID]["available"] is True
    assert profiles[module.QWEN21_PRUNA_5_PROFILE_ID]["steps"] == 5
    assert profiles[module.QWEN21_PRUNA_5_PROFILE_ID]["max_reference_images"] == 3
    assert profiles[module.QWEN21_PRUNA_5_PROFILE_ID]["negative_prompt"] is False
    assert profiles[module.QWEN21_PRUNA_5_PROFILE_ID]["available"] is True


def test_qwen21_standard_text_generation_settings_are_deterministic():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(qwen21_job(), 2101)

    assert settings["model_type"] == module.QWEN21_MODEL_ID
    assert settings["resolution"] == "1280x736"
    assert settings["_midom_requested_resolution"] == "1280x720"
    assert settings["_midom_image_delivery_adapter"] == "qwen21_32px_center_crop"
    assert settings["num_inference_steps"] == 40
    assert settings["guidance_scale"] == 4.0
    assert settings["sample_solver"] == "default"
    assert settings["activated_loras"] == []
    assert settings["multi_prompts_gen_type"] == "FG"
    assert settings["video_prompt_type"] == ""
    assert settings["prompt_enhancer"] == ""
    assert settings["custom_settings"] == {"qwen21_kv_cache": "Disabled", "rgba": "Disabled"}


@pytest.mark.parametrize(
    ("method", "model_mode"),
    [("masked_denoising", 0), ("lanpaint_5", 3)],
)
def test_qwen21_masked_edit_maps_settled_methods(method, model_mode):
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(qwen21_masked_edit_job(method), 21011)

    assert settings["image_mode"] == 2
    assert settings["video_prompt_type"] == "VAG"
    assert settings["model_mode"] == model_mode
    assert settings["denoising_strength"] == 1.0
    assert settings["masking_strength"] == 1.0
    assert settings["num_inference_steps"] == 40
    assert settings["_midom_accelerator_profile_id"] == "standard"
    assert settings["_midom_image_task"] == "masked_edit"
    assert settings["_midom_qwen21_masked_edit_contract_version"] == "qwen21_masked_edit_v1"
    assert settings["_midom_output_color_mode"] == "RGB"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("masked_edit_contract_version", "old", "contract_version"),
        ("reference_mode", "ordered_reference_images", "reference_mode"),
        ("masked_edit_method", "lanpaint_10", "masked_edit_method"),
        ("edit_strength", "strong", "edit_strength='balanced'"),
        ("mask_semantics", "alpha_white_edit", "mask_semantics"),
        ("accelerator_profile_id", "qwen21_pruna_v01_8", "only accelerator_profile_id='standard'"),
    ],
)
def test_qwen21_masked_edit_rejects_contract_drift(field, value, message):
    module = load_plugin_module()
    job = qwen21_masked_edit_job()
    job["generation"][field] = value
    with pytest.raises(ValueError, match=message):
        plugin_instance(module)._validate_image_job(job, 21012)


def test_qwen21_masked_edit_rejects_wrong_input_roles_and_mask_mime():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_masked_edit_job()
    job["inputs"][0]["role"] = "reference"
    with pytest.raises(ValueError, match="accepts only"):
        plugin._validate_image_job(job, 21013)

    job = qwen21_masked_edit_job()
    job["inputs"][1]["mime_type"] = "image/jpeg"
    with pytest.raises(ValueError, match="must declare mime_type='image/png'"):
        plugin._validate_image_job(job, 21014)


def test_qwen21_masked_edit_validates_supporting_reference_limit_and_order():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(
        qwen21_masked_edit_job(supporting_reference_count=1),
        210141,
    )
    assert settings["_midom_qwen21_reference_image_count"] == 2
    assert settings["video_prompt_type"] == "VAGI"
    assert [item["role"] for item in settings["_midom_qwen21_reference_descriptors"]] == [
        "source_image",
        "supporting_reference_image",
    ]
    assert [item["sequence"] for item in settings["_midom_qwen21_reference_descriptors"]] == [0, 1]

    with pytest.raises(ValueError, match="at most 1 supporting reference image"):
        plugin._validate_image_job(
            qwen21_masked_edit_job(supporting_reference_count=2),
            210142,
        )

    job = qwen21_masked_edit_job(supporting_reference_count=1)
    job["inputs"][1]["sequence"] = 2
    with pytest.raises(ValueError, match="unique, contiguous"):
        plugin._validate_image_job(job, 210143)


def test_qwen21_masked_edit_prepares_paired_landscape_canvas(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_masked_edit_job()
    settings = plugin._validate_image_job(job, 21017)
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    source = Image.new("RGB", (1280, 720), (20, 40, 60))
    source.putpixel((0, 0), (1, 2, 3))
    source.putpixel((1279, 719), (7, 8, 9))
    source.save(source_path)
    mask = Image.new("L", (1280, 720), 0)
    for x in range(500, 780):
        for y in range(250, 470):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    downloaded = [
        {
            "kind": "reference_image", "role": "source_image", "input_id": 1000,
            "sequence": 0, "path": str(source_path), "sha256": "a" * 64,
        },
        {
            "kind": "mask_image", "role": "edit_mask", "input_id": 2000,
            "path": str(mask_path), "sha256": "b" * 64,
        },
    ]

    plugin._apply_inputs_to_settings(settings, downloaded, job)

    assert settings["image_mode"] == 2
    assert settings["video_prompt_type"] == "VAG"
    assert settings["image_refs"] == []
    with Image.open(settings["image_guide"]) as prepared_source:
        assert prepared_source.mode == "RGB"
        assert prepared_source.size == (1280, 736)
        assert prepared_source.getpixel((0, 0)) == (1, 2, 3)
        assert prepared_source.getpixel((1279, 735)) == (7, 8, 9)
    with Image.open(settings["image_mask"]) as prepared_mask:
        assert prepared_mask.mode == "L"
        assert prepared_mask.size == (1280, 736)
        assert prepared_mask.getpixel((640, 0)) == 0
        assert prepared_mask.getpixel((640, 368)) == 255
    adapter = settings["_midom_qwen21_canvas_adapter"]
    assert adapter["pad_top"] == 8
    assert adapter["pad_bottom"] == 8
    assert adapter["mask_padding"] == "black_protected"


def test_qwen21_masked_edit_downloads_typed_source_and_png_mask(tmp_path, monkeypatch):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_masked_edit_job(resolution=(768, 768), supporting_reference_count=1)
    payloads = {}
    for input_id, mode, color, size in (
        (1000, "RGB", (20, 30, 40), (768, 768)),
        (1100, "RGB", (80, 90, 100), (320, 640)),
        (2000, "L", 0, (768, 768)),
    ):
        image = Image.new(mode, size, color)
        if input_id == 2000:
            image.paste(255, (100, 100, 300, 300))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payloads[input_id] = buffer.getvalue()
    for item in job["inputs"]:
        data = payloads[item["input_id"]]
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

    def fake_get(url, **kwargs):
        return FakeResponse(payloads[int(url.rsplit("/", 1)[-1])])

    monkeypatch.setattr(module.requests, "get", fake_get)
    connection = module.ConnectionContext(
        connection_id="worker-1", api_base_url="https://midom.test", worker_id=1,
        worker_token="token", org_id=2, project_id=3, paired_user_id=4,
        machine_name="GPU", capabilities_revision=1, token_expires_at="",
        allow_insecure_local_dev=False, allow_insecure_lan_dev=False,
    )

    downloaded = plugin._download_job_inputs(connection, job, str(tmp_path))

    assert [(item["kind"], item["role"]) for item in downloaded] == [
        ("reference_image", "source_image"),
        ("reference_image", "supporting_reference_image"),
        ("mask_image", "edit_mask"),
    ]
    assert downloaded[1]["sequence"] == 1
    assert downloaded[2]["mime_type"] == "image/png"
    assert downloaded[2]["sha256"] == job["inputs"][2]["sha256"]


def test_qwen21_masked_edit_prepares_paired_portrait_canvas(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_masked_edit_job(resolution=(720, 1280))
    settings = plugin._validate_image_job(job, 21018)
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (720, 1280), (30, 50, 70)).save(source_path)
    mask = Image.new("L", (720, 1280), 0)
    mask.paste(255, (100, 100, 200, 200))
    mask.save(mask_path)
    downloaded = [
        {"kind": "reference_image", "role": "source_image", "input_id": 1000, "sequence": 0, "path": str(source_path), "sha256": "a" * 64},
        {"kind": "mask_image", "role": "edit_mask", "input_id": 2000, "path": str(mask_path), "sha256": "b" * 64},
    ]

    plugin._apply_inputs_to_settings(settings, downloaded, job)

    with Image.open(settings["image_mask"]) as prepared_mask:
        assert prepared_mask.size == (736, 1280)
        assert prepared_mask.getpixel((0, 150)) == 0
        assert prepared_mask.getpixel((108, 150)) == 255
    assert settings["_midom_qwen21_canvas_adapter"]["pad_left"] == 8
    assert settings["_midom_qwen21_canvas_adapter"]["pad_right"] == 8


def test_qwen21_masked_edit_submission_uses_wangp_native_mask_path(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_masked_edit_job(resolution=(768, 768), supporting_reference_count=1)
    settings = plugin._validate_image_job(job, 21019)
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    first_reference_path = tmp_path / "first-reference.png"
    Image.new("RGB", (768, 768), "navy").save(source_path)
    Image.new("RGB", (320, 640), "green").save(first_reference_path)
    mask = Image.new("L", (768, 768), 0)
    mask.paste(255, (100, 100, 300, 300))
    mask.save(mask_path)
    plugin._apply_inputs_to_settings(settings, [
        {"kind": "reference_image", "role": "source_image", "input_id": 1000, "sequence": 0, "path": str(source_path), "sha256": "a" * 64},
        {"kind": "mask_image", "role": "edit_mask", "input_id": 2000, "path": str(mask_path), "sha256": "b" * 64},
        {"kind": "reference_image", "role": "supporting_reference_image", "input_id": 1100, "sequence": 1, "path": str(first_reference_path), "sha256": "c" * 64},
    ], job)
    submitted = {}

    class FakeSession:
        def submit_task(self, task, callbacks=None):
            submitted.update(task)
            return types.SimpleNamespace(done=True)

    plugin._submit_wangp_job(FakeSession(), settings, 1, callbacks=types.SimpleNamespace())

    assert submitted["image_mode"] == 2
    assert submitted["video_prompt_type"] == "VAGI"
    assert submitted["model_mode"] == 0
    assert Path(submitted["image_guide"]).is_file()
    assert Path(submitted["image_mask"]).is_file()
    assert submitted["image_refs"] == [str(first_reference_path)]
    with Image.open(submitted["image_mask"]) as submitted_mask:
        assert submitted_mask.mode == "L"
        assert submitted_mask.getpixel((50, 50)) == 0
        assert submitted_mask.getpixel((150, 150)) == 255
    assert not any(key.startswith("_midom_") for key in submitted)


def test_qwen21_masked_edit_rejects_alpha_gray_and_unconfirmed_full_masks(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    source_path = tmp_path / "source.png"
    Image.new("RGB", (768, 768), "navy").save(source_path)
    source_input = {"path": str(source_path), "input_id": 1}

    alpha_path = tmp_path / "alpha.png"
    alpha_mask = Image.new("RGBA", (768, 768), (255, 255, 255, 0))
    alpha_mask.save(alpha_path)
    with pytest.raises(ValueError, match="alpha must be fully opaque"):
        plugin._prepare_qwen21_masked_edit_inputs(
            source_input, {"path": str(alpha_path), "input_id": 2}, (768, 768), (768, 768), False
        )

    gray_path = tmp_path / "gray.png"
    Image.new("L", (768, 768), 128).save(gray_path)
    with pytest.raises(ValueError, match="binary black/white"):
        plugin._prepare_qwen21_masked_edit_inputs(
            source_input, {"path": str(gray_path), "input_id": 2}, (768, 768), (768, 768), False
        )

    full_path = tmp_path / "full.png"
    Image.new("L", (768, 768), 255).save(full_path)
    with pytest.raises(ValueError, match="allow_full_image_edit=true"):
        plugin._prepare_qwen21_masked_edit_inputs(
            source_input, {"path": str(full_path), "input_id": 2}, (768, 768), (768, 768), False
        )
    prepared = plugin._prepare_qwen21_masked_edit_inputs(
        source_input, {"path": str(full_path), "input_id": 2}, (768, 768), (768, 768), True
    )
    assert Path(prepared["mask_path"]).is_file()


def test_qwen21_accepts_ten_ordered_references_in_standard_mode():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(
        qwen21_job("ordered_reference_images", 10),
        21015,
    )

    assert settings["_midom_qwen21_reference_image_count"] == 10
    assert [item["sequence"] for item in settings["_midom_qwen21_reference_descriptors"]] == list(range(10))


def test_qwen21_multiline_prompt_remains_one_fg_task():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_job()
    job["prompt"] = "Subject identity\nCamera composition\nLighting direction\nDetails to preserve"
    settings = plugin._validate_image_job(job, 21016)
    submitted = {}

    class FakeSession:
        def submit_task(self, task, callbacks=None):
            submitted.update(task)
            return types.SimpleNamespace(done=True)

        def submit_manifest(self, tasks, callbacks=None):
            raise AssertionError("A single Qwen Image 2.1 output must not split a multiline prompt.")

    plugin._submit_wangp_job(FakeSession(), settings, 1, callbacks=types.SimpleNamespace())

    assert submitted["prompt"] == job["prompt"]
    assert submitted["multi_prompts_gen_type"] == "FG"


def test_qwen21_viggle_applies_complete_profile():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    enable_viggle(plugin, module)
    settings = plugin._validate_image_job(
        qwen21_job("ordered_reference_images", 3, module.QWEN21_VIGGLE_PROFILE_ID),
        2102,
    )

    assert settings["num_inference_steps"] == 6
    assert settings["guidance_scale"] == 1.0
    assert settings["sample_solver"] == "viggle_v02"
    assert settings["activated_loras"] == [module.QWEN21_VIGGLE_LORA_FILENAME]
    assert settings["loras_multipliers"] == "1"
    assert settings["negative_prompt"] == ""


@pytest.mark.parametrize(
    ("profile_id", "steps", "lora_filename"),
    [
        ("qwen21_pruna_v01_8", 8, "p_qwen_image_2.1_8step_v0.1.safetensors"),
        ("qwen21_pruna_v01_5", 5, "p_qwen_image_2.1_5step_v0.1.safetensors"),
    ],
)
def test_qwen21_pruna_applies_complete_profile(profile_id, steps, lora_filename):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    enable_qwen21_accelerators(plugin, module)
    settings = plugin._validate_image_job(
        qwen21_job("ordered_reference_images", 3, profile_id),
        21021,
    )

    assert settings["num_inference_steps"] == steps
    assert settings["guidance_scale"] == 1.0
    assert settings["sample_solver"] == "pruna"
    assert settings["activated_loras"] == [lora_filename]
    assert settings["loras_multipliers"] == "1"
    assert settings["negative_prompt"] == ""


def test_qwen21_viggle_is_reported_unavailable_without_lora():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._resolve_accelerator_loras = lambda profile: ([], [module.QWEN21_VIGGLE_LORA_FILENAME])

    profiles = {item["profile_id"]: item for item in plugin._accelerator_profiles_for_model(module.QWEN21_MODEL_ID)}

    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["available"] is False
    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["unavailable_reason"] == "required_accelerator_files_not_installed"


@pytest.mark.parametrize(
    ("profile_id", "lora_filename"),
    [
        ("qwen21_pruna_v01_8", "p_qwen_image_2.1_8step_v0.1.safetensors"),
        ("qwen21_pruna_v01_5", "p_qwen_image_2.1_5step_v0.1.safetensors"),
    ],
)
def test_qwen21_pruna_is_reported_unavailable_without_lora(profile_id, lora_filename):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._resolve_accelerator_loras = lambda profile: (
        ([], [lora_filename]) if profile.get("profile_id") == profile_id else ([], [])
    )

    profiles = {item["profile_id"]: item for item in plugin._accelerator_profiles_for_model(module.QWEN21_MODEL_ID)}

    assert profiles[profile_id]["available"] is False
    assert profiles[profile_id]["unavailable_reason"] == "required_accelerator_files_not_installed"


@pytest.mark.parametrize(
    ("delivery", "internal", "adapter"),
    [
        ((1280, 720), "1280x736", "qwen21_32px_center_crop"),
        ((720, 1280), "736x1280", "qwen21_32px_center_crop"),
        ((1024, 1024), "1024x1024", ""),
        ((768, 768), "768x768", ""),
    ],
)
def test_qwen21_delivery_resolution_mapping(delivery, internal, adapter):
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(qwen21_job(resolution=delivery), 2103)

    assert settings["resolution"] == internal
    assert settings["_midom_image_delivery_adapter"] == adapter


def test_qwen21_rejects_invalid_reference_contracts():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    with pytest.raises(ValueError, match="reference_mode=none"):
        plugin._validate_image_job(qwen21_job("none", 1), 2104)

    with pytest.raises(ValueError, match="requires at least one reference image"):
        plugin._validate_image_job(qwen21_job("primary_image_edit", 0), 2105)

    job = qwen21_job("ordered_reference_images", 2)
    job["inputs"][1]["sequence"] = 0
    with pytest.raises(ValueError, match="unique, contiguous"):
        plugin._validate_image_job(job, 2106)

    job = qwen21_job("ordered_reference_images", 1)
    job["inputs"].append({"kind": "control_image", "input_id": 99, "sequence": 1})
    with pytest.raises(ValueError, match="does not support input kind"):
        plugin._validate_image_job(job, 2107)


def test_qwen21_profile_specific_reference_and_negative_prompt_limits():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    enable_viggle(plugin, module)

    with pytest.raises(ValueError, match="at most 3 reference images"):
        plugin._validate_image_job(
            qwen21_job("ordered_reference_images", 4, module.QWEN21_VIGGLE_PROFILE_ID),
            2108,
        )

    job = qwen21_job("ordered_reference_images", 1, module.QWEN21_VIGGLE_PROFILE_ID)
    job["negative_prompt"] = "unwanted text"
    with pytest.raises(ValueError, match="does not support a negative prompt"):
        plugin._validate_image_job(job, 2109)

    with pytest.raises(ValueError, match="at most 10 reference images"):
        plugin._validate_image_job(qwen21_job("ordered_reference_images", 11), 2110)


@pytest.mark.parametrize("profile_id", ["qwen21_pruna_v01_8", "qwen21_pruna_v01_5"])
def test_qwen21_pruna_reference_negative_prompt_and_step_limits(profile_id):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    enable_qwen21_accelerators(plugin, module)

    with pytest.raises(ValueError, match="at most 3 reference images"):
        plugin._validate_image_job(qwen21_job("ordered_reference_images", 4, profile_id), 21081)

    job = qwen21_job("ordered_reference_images", 1, profile_id)
    job["negative_prompt"] = "unwanted text"
    with pytest.raises(ValueError, match="does not support a negative prompt"):
        plugin._validate_image_job(job, 21082)

    job = qwen21_job("ordered_reference_images", 1, profile_id)
    job["generation"]["steps"] = 40
    with pytest.raises(ValueError, match="requires (5|8) steps"):
        plugin._validate_image_job(job, 21083)


def test_qwen21_rejects_non_png_and_worker_prompt_enhancement():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_job()
    job["output"]["format"] = "jpeg"
    with pytest.raises(ValueError, match="output.format must be 'png'"):
        plugin._validate_image_job(job, 2111)

    job = qwen21_job()
    job["generation"]["prompt_enhancement_by_worker"] = True
    with pytest.raises(ValueError, match="worker prompt enhancement is not supported"):
        plugin._validate_image_job(job, 2112)


def test_qwen21_applies_downloaded_references_in_sequence_order():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(qwen21_job("ordered_reference_images", 3), 2113)
    downloaded = [
        {"kind": "reference_image", "input_id": 1002, "sequence": 2, "path": "/tmp/third.png", "sha256": "c" * 64},
        {"kind": "reference_image", "input_id": 1000, "sequence": 0, "path": "/tmp/first.png", "sha256": "a" * 64},
        {"kind": "reference_image", "input_id": 1001, "sequence": 1, "path": "/tmp/second.png", "sha256": "b" * 64},
    ]

    plugin._apply_inputs_to_settings(settings, downloaded, qwen21_job("ordered_reference_images", 3))

    assert settings["image_refs"] == ["/tmp/first.png", "/tmp/second.png", "/tmp/third.png"]
    assert settings["video_prompt_type"] == "I"
    assert [item["input_id"] for item in settings["_midom_qwen21_reference_inputs"]] == [1000, 1001, 1002]


def test_qwen21_primary_image_edit_uses_ki():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_job("primary_image_edit", 1)
    settings = plugin._validate_image_job(job, 2114)
    downloaded = [
        {"kind": "reference_image", "input_id": 1000, "sequence": 0, "path": "/tmp/primary.png", "sha256": "a" * 64},
    ]

    plugin._apply_inputs_to_settings(settings, downloaded, job)

    assert settings["image_refs"] == ["/tmp/primary.png"]
    assert settings["video_prompt_type"] == "KI"


def test_qwen21_artifact_is_center_cropped_and_converted_to_png(tmp_path, monkeypatch):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    source = tmp_path / "qwen-output.jpg"
    Image.new("RGB", (1280, 736), (20, 40, 60)).save(source, format="JPEG", quality=95)
    captured = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"artifact_id": 91, "file_id": 92, "artifact_index": 0}

    def fake_post(url, headers, data, files, timeout):
        file_name, reader, mime_type = files["file"]
        payload = reader.read()
        captured.update(filename=file_name, mime_type=mime_type, payload=payload, data=dict(data))
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)
    connection = module.ConnectionContext(
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

    plugin._upload_artifact(
        connection,
        2115,
        str(source),
        0,
        "1280x720",
        "center_crop_downscale",
        "image/png",
        "RGB",
    )

    assert captured["mime_type"] == "image/png"
    assert captured["payload"].startswith(b"\x89PNG\r\n\x1a\n")
    normalized = tmp_path / "normalized.png"
    normalized.write_bytes(captured["payload"])
    with Image.open(normalized) as image:
        assert image.size == (1280, 720)
        assert image.format == "PNG"
        assert image.mode == "RGB"


def test_qwen21_generation_metadata_records_profile_references_and_adapter():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(qwen21_job("ordered_reference_images", 2), 2116)
    settings["_midom_qwen21_reference_inputs"] = [
        {"input_id": 1000, "sequence": 0, "sha256": "a" * 64},
        {"input_id": 1001, "sequence": 1, "sha256": "b" * 64},
    ]

    metadata = plugin._build_generation_metadata(
        settings,
        types.SimpleNamespace(),
        ["result-seed123.png"],
    )

    assert metadata["reference_mode"] == "ordered_reference_images"
    assert metadata["accelerator_profile_id"] == "standard"
    assert metadata["ordered_reference_inputs"][1]["input_id"] == 1001
    assert metadata["ordered_reference_inputs"][1]["sha256"] == "b" * 64
    assert metadata["requested_resolution"] == "1280x720"
    assert metadata["internal_render_resolution"] == "1280x736"
    assert metadata["image_delivery_adapter"] == "qwen21_32px_center_crop"
    assert metadata["sample_solver"] == "default"


def test_qwen21_masked_edit_generation_metadata_records_inputs_recipe_and_canvas():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(
        qwen21_masked_edit_job("lanpaint_5", supporting_reference_count=1),
        21162,
    )
    settings["_midom_qwen21_reference_inputs"] = [
        {"input_id": 1000, "role": "source_image", "sequence": 0, "sha256": "a" * 64},
        {"input_id": 1100, "role": "supporting_reference_image", "sequence": 1, "sha256": "e" * 64},
    ]
    settings["_midom_qwen21_masked_edit_inputs"] = {
        "source": {"input_id": 1000, "role": "source_image", "sequence": 0, "sha256": "a" * 64, "prepared_sha256": "c" * 64},
        "mask": {"input_id": 2000, "role": "edit_mask", "sequence": None, "sha256": "b" * 64, "prepared_sha256": "d" * 64},
    }
    settings["_midom_qwen21_canvas_adapter"] = {
        "adapter_id": "qwen21_masked_edit_canvas_v1",
        "pad_top": 8,
        "pad_bottom": 8,
    }

    metadata = plugin._build_generation_metadata(settings, types.SimpleNamespace(), ["result-seed123.png"])

    assert metadata["image_task"] == "masked_edit"
    assert metadata["masked_edit"]["contract_version"] == "qwen21_masked_edit_v1"
    assert metadata["masked_edit"]["method"] == "lanpaint_5"
    assert metadata["masked_edit"]["source"]["input_id"] == 1000
    assert metadata["masked_edit"]["source"]["role"] == "source_image"
    assert metadata["masked_edit"]["mask"]["input_id"] == 2000
    assert metadata["masked_edit"]["mask"]["role"] == "edit_mask"
    assert metadata["masked_edit"]["references"][1] == {
        "input_id": 1100,
        "role": "supporting_reference_image",
        "sequence": 1,
        "sha256": "e" * 64,
    }
    assert metadata["masked_edit"]["expanded_bridge_recipe"]["model_mode"] == 3
    assert metadata["masked_edit"]["expanded_bridge_recipe"]["video_prompt_type"] == "VAGI"
    assert metadata["masked_edit"]["canvas_adapter"]["pad_top"] == 8


def test_qwen21_masked_edit_candidate_contract_is_gated():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    connection = module.ConnectionContext(
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
    candidate = {
        "job_id": 21163,
        "worker_id": 1,
        "org_id": 2,
        "project_id": 3,
        "requested_by_user_id": 4,
        "media_type": "image",
        "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "masked_edit",
            "masked_edit_contract_version": "qwen21_masked_edit_v1",
            "masked_edit_method": "masked_denoising",
            "edit_strength": "balanced",
            "mask_semantics": "luminance_white_edit_v1",
            "reference_mode": "primary_image_edit",
            "reference_image_count": 1,
            "mask_image_count": 1,
            "control_image_count": 0,
            "accelerator_profile_id": "standard",
            "output_count": 1,
            "output_format": "png",
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, connection) is None
    candidate["summary"]["reference_image_count"] = 2
    assert plugin._candidate_incompatibility_reason(candidate, connection) is None
    candidate["summary"]["reference_image_count"] = 3
    assert "at most 1 supporting reference image" in plugin._candidate_incompatibility_reason(candidate, connection)
    candidate["summary"]["reference_image_count"] = 1
    candidate["summary"]["supporting_reference_image_count"] = 2
    assert "at most 1 supporting reference image" in plugin._candidate_incompatibility_reason(candidate, connection)
    candidate["summary"]["supporting_reference_image_count"] = 0
    candidate["summary"]["mask_image_count"] = 0
    assert "requires one mask image" in plugin._candidate_incompatibility_reason(candidate, connection)
    candidate["summary"]["mask_image_count"] = 1
    candidate["summary"]["accelerator_profile_id"] = module.QWEN21_PRUNA_8_PROFILE_ID
    assert "only accelerator_profile_id=standard" in plugin._candidate_incompatibility_reason(candidate, connection)


def test_qwen21_generation_metadata_records_pruna_recipe():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    enable_qwen21_accelerators(plugin, module)
    settings = plugin._validate_image_job(
        qwen21_job("primary_image_edit", 1, module.QWEN21_PRUNA_8_PROFILE_ID),
        21161,
    )

    metadata = plugin._build_generation_metadata(
        settings,
        types.SimpleNamespace(),
        ["result-seed123.png"],
    )

    assert metadata["accelerator_profile_id"] == module.QWEN21_PRUNA_8_PROFILE_ID
    assert metadata["num_inference_steps"] == 8
    assert metadata["guidance_scale"] == 1.0
    assert metadata["sample_solver"] == "pruna"


def test_qwen21_candidate_compatibility_enforces_viggle_limit():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    enable_viggle(plugin, module)
    connection = module.ConnectionContext(
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
    candidate = {
        "job_id": 2117,
        "worker_id": 1,
        "org_id": 2,
        "project_id": 3,
        "requested_by_user_id": 4,
        "media_type": "image",
        "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "output_count": 1,
            "output_format": "png",
            "reference_mode": "ordered_reference_images",
            "reference_image_count": 4,
            "control_image_count": 0,
            "accelerator_profile_id": module.QWEN21_VIGGLE_PROFILE_ID,
        },
    }

    reason = plugin._candidate_incompatibility_reason(candidate, connection)

    assert reason is not None
    assert "at most 3 reference images" in reason


@pytest.mark.parametrize("profile_id", ["qwen21_pruna_v01_8", "qwen21_pruna_v01_5"])
def test_qwen21_candidate_compatibility_enforces_pruna_availability_and_limit(profile_id):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    connection = module.ConnectionContext(
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
    candidate = {
        "job_id": 21171,
        "worker_id": 1,
        "org_id": 2,
        "project_id": 3,
        "requested_by_user_id": 4,
        "media_type": "image",
        "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "output_count": 1,
            "output_format": "png",
            "reference_mode": "ordered_reference_images",
            "reference_image_count": 4,
            "control_image_count": 0,
            "accelerator_profile_id": profile_id,
        },
    }
    plugin._resolve_accelerator_loras = lambda profile: (
        [],
        [qwen21_accelerator_filename(module, profile_id)],
    )

    reason = plugin._candidate_incompatibility_reason(candidate, connection)
    assert reason == f"Qwen Image 2.1 accelerator unavailable: {profile_id}"

    enable_qwen21_accelerators(plugin, module)
    reason = plugin._candidate_incompatibility_reason(candidate, connection)
    assert reason is not None
    assert "at most 3 reference images" in reason


@pytest.mark.parametrize(
    ("source_size", "output_size", "expected"),
    [
        ((1024, 1024), (1280, 720), (720, 720)),
        ((720, 1280), (1280, 720), (405, 720)),
        ((1280, 720), (720, 1280), (720, 405)),
        ((320, 200), (1280, 720), (320, 200)),
        ((2000, 1000), (1280, 720), (1280, 640)),
        ((1016, 1280), (1280, 720), (572, 720)),
    ],
)
def test_qwen21_outpaint_geometry_fixtures(source_size, output_size, expected):
    module = load_plugin_module()

    assert module.AwsWorkerBridgePlugin._qwen21_outpaint_fitted_size(source_size, output_size) == expected


def test_qwen21_outpaint_contract_maps_to_red_canvas_recipe():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_image_job(qwen21_outpaint_job(), 21200)

    assert settings["_midom_image_task"] == "outpaint"
    assert settings["_midom_qwen21_outpaint_contract_version"] == "qwen21_outpaint_v1"
    assert settings["_midom_qwen21_outpaint_requested_placement"] == {
        "x": 280,
        "y": 0,
        "width": 720,
        "height": 720,
    }
    assert settings["resolution"] == "1280x736"
    assert settings["image_mode"] == 1
    assert settings["video_prompt_type"] == "KI"
    assert settings["video_guide_outpainting"] == ""
    assert settings["video_guide_outpainting_ratio"] == ""
    assert settings["num_inference_steps"] == 40
    assert settings["guidance_scale"] == 4.0
    assert settings["sample_solver"] == "default"
    assert settings["prompt"].endswith(module.QWEN21_RED_OUTPAINTING_PROMPT)
    assert settings["prompt"].count(module.QWEN21_RED_OUTPAINTING_PROMPT) == 1


def test_qwen21_outpaint_rejects_retired_preserve_original_contract():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_outpaint_job()
    job["generation"]["outpaint_mode"] = "preserve_original"
    job["generation"]["preserve_original_contract_version"] = "qwen21_preserve_original_outpaint_v1"
    job["generation"]["preservation_profile_id"] = "exact_source_v1"
    with pytest.raises(ValueError, match="Preserve Original is no longer supported"):
        plugin._validate_image_job(job, 212001)


def test_qwen21_outpaint_rejects_generation_descriptor_dimension_mismatch():
    module = load_plugin_module()
    job = qwen21_outpaint_job()
    job["generation"]["source_width"] = 1000

    with pytest.raises(ValueError, match="generation source dimensions must match"):
        plugin_instance(module)._validate_image_job(job, 21200)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("outpaint_contract_version", "old", "outpaint_contract_version"),
        ("reference_mode", "ordered_reference_images", "reference_mode"),
        ("placement_mode", "preset", "placement_mode"),
        ("source_scale_mode", "stretch", "source_scale_mode"),
        ("accelerator_profile_id", "qwen21_pruna_v01_8", "only accelerator_profile_id='standard'"),
    ],
)
def test_qwen21_outpaint_rejects_contract_drift(field, value, message):
    module = load_plugin_module()
    job = qwen21_outpaint_job()
    job["generation"][field] = value

    with pytest.raises(ValueError, match=message):
        plugin_instance(module)._validate_image_job(job, 21201)


def test_qwen21_outpaint_rejects_wrong_input_geometry_and_combinations():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    job = qwen21_outpaint_job()
    job["inputs"][0]["role"] = "supporting_reference_image"
    with pytest.raises(ValueError, match="role='source_image'"):
        plugin._validate_image_job(job, 21202)

    job = qwen21_outpaint_job()
    job["inputs"].append({
        "kind": "mask_image",
        "role": "edit_mask",
        "input_id": 2000,
        "mime_type": "image/png",
    })
    with pytest.raises(ValueError, match="exactly one source_image"):
        plugin._validate_image_job(job, 21203)

    job = qwen21_outpaint_job()
    job["generation"]["source_placement"]["width"] = 719
    with pytest.raises(ValueError, match="expected 720x720"):
        plugin._validate_image_job(job, 21204)

    job = qwen21_outpaint_job()
    job["generation"]["source_placement"]["x"] = 600
    with pytest.raises(ValueError, match="exceeds the output width"):
        plugin._validate_image_job(job, 21205)

    job = qwen21_outpaint_job()
    job["generation"]["source_placement"]["x"] = 1.5
    with pytest.raises(ValueError, match="must be an integer"):
        plugin._validate_image_job(job, 21206)

    with pytest.raises(ValueError, match="nonzero outpainting margin"):
        plugin._validate_image_job(
            qwen21_outpaint_job(source_size=(1024, 1024), resolution=(1024, 1024)),
            21207,
        )


def test_qwen21_outpaint_prepares_exact_internal_canvas_and_submission(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_outpaint_job()
    settings = plugin._validate_image_job(job, 21208)
    source_path = tmp_path / "source.png"
    Image.new("RGB", (1024, 1024), (10, 20, 200)).save(source_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    downloaded = [{
        "kind": "reference_image",
        "role": "source_image",
        "input_id": 1000,
        "sequence": 0,
        "path": str(source_path),
        "sha256": source_sha256,
    }]

    plugin._apply_inputs_to_settings(settings, downloaded, job)

    assert settings["video_prompt_type"] == "KI"
    assert settings["video_guide_outpainting"] == ""
    assert settings["video_guide_outpainting_ratio"] == ""
    assert "image_guide" not in settings
    assert len(settings["image_refs"]) == 1
    canvas_path = Path(settings["image_refs"][0])
    assert canvas_path.is_file()
    with Image.open(canvas_path) as canvas:
        assert canvas.mode == "RGB"
        assert canvas.size == (1280, 736)
        assert canvas.getpixel((0, 0)) == (255, 0, 0)
        assert canvas.getpixel((279, 8)) == (255, 0, 0)
        assert canvas.getpixel((280, 8)) == (10, 20, 200)
        assert canvas.getpixel((999, 727)) == (10, 20, 200)
        assert canvas.getpixel((1000, 727)) == (255, 0, 0)
    assert settings["_midom_qwen21_outpaint_resolved_placement"] == {
        "x": 280,
        "y": 0,
        "width": 720,
        "height": 720,
    }
    assert settings["_midom_qwen21_outpaint_internal_placement"] == {
        "x": 280,
        "y": 8,
        "width": 720,
        "height": 720,
    }
    assert settings["_midom_qwen21_outpaint_input"]["sha256"] == source_sha256
    assert settings["_midom_qwen21_outpaint_input"]["prepared_canvas_sha256"] == hashlib.sha256(
        canvas_path.read_bytes()
    ).hexdigest()

    submitted = {}

    class FakeSession:
        def submit_task(self, task, callbacks=None):
            submitted.update(task)
            return types.SimpleNamespace(done=True)

    plugin._submit_wangp_job(FakeSession(), settings, 1, callbacks=types.SimpleNamespace())

    assert submitted["image_refs"] == [str(canvas_path)]
    assert submitted["video_prompt_type"] == "KI"
    assert submitted["video_guide_outpainting"] == ""
    assert submitted["resolution"] == "1280x736"
    assert not any(key.startswith("_midom_") for key in submitted)


def test_qwen21_outpaint_uses_exif_oriented_source_dimensions(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    placement = {"x": 0, "y": 0, "width": 50, "height": 100}
    job = qwen21_outpaint_job(
        source_size=(50, 100),
        resolution=(768, 768),
        placement=placement,
    )
    settings = plugin._validate_image_job(job, 21209)
    source_path = tmp_path / "rotated-source.jpg"
    source = Image.new("RGB", (100, 50), (40, 120, 200))
    exif = source.getexif()
    exif[274] = 6
    source.save(source_path, format="JPEG", exif=exif)
    downloaded = [{
        "kind": "reference_image",
        "role": "source_image",
        "input_id": 1000,
        "sequence": 0,
        "path": str(source_path),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }]

    plugin._apply_inputs_to_settings(settings, downloaded, job)

    assert settings["_midom_qwen21_outpaint_input"]["display_width"] == 50
    assert settings["_midom_qwen21_outpaint_input"]["display_height"] == 100
    with Image.open(settings["image_refs"][0]) as canvas:
        assert canvas.getpixel((0, 0)) != (255, 0, 0)
        assert canvas.getpixel((50, 0)) == (255, 0, 0)


def test_qwen21_outpaint_rejects_downloaded_display_dimension_mismatch(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_outpaint_job(source_size=(640, 480), resolution=(1280, 720))
    settings = plugin._validate_image_job(job, 21210)
    source_path = tmp_path / "wrong-size.png"
    Image.new("RGB", (641, 480), "blue").save(source_path)

    with pytest.raises(ValueError, match="do not match the claim descriptor"):
        plugin._apply_inputs_to_settings(settings, [{
            "kind": "reference_image",
            "role": "source_image",
            "input_id": 1000,
            "sequence": 0,
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }], job)


def test_qwen21_outpaint_generation_metadata_records_geometry_and_recipe(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen21_outpaint_job()
    settings = plugin._validate_image_job(job, 21211)
    source_path = tmp_path / "source.png"
    Image.new("RGB", (1024, 1024), "blue").save(source_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    plugin._apply_inputs_to_settings(settings, [{
        "kind": "reference_image",
        "role": "source_image",
        "input_id": 1000,
        "sequence": 0,
        "path": str(source_path),
        "sha256": source_sha256,
    }], job)

    metadata = plugin._build_generation_metadata(settings, types.SimpleNamespace(), ["result-seed123.png"])

    assert metadata["image_task"] == "outpaint"
    outpaint = metadata["outpaint"]
    assert outpaint["contract_version"] == "qwen21_outpaint_v1"
    assert outpaint["recipe_version"] == "qwen21_red_canvas_v1"
    assert outpaint["source"]["input_id"] == 1000
    assert outpaint["source"]["sha256"] == source_sha256
    assert outpaint["requested_placement"] == outpaint["resolved_placement"]
    assert outpaint["internal_placement"]["y"] == 8
    assert outpaint["canvas_adapter"]["crop_top"] == 8
    assert outpaint["accelerator_profile_id"] == "standard"
    assert outpaint["expanded_bridge_recipe"]["video_guide_outpainting"] == ""
    assert outpaint["expanded_bridge_recipe"]["num_inference_steps"] == 40


def test_qwen21_outpaint_candidate_contract_is_gated():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    connection = module.ConnectionContext(
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
    candidate = {
        "job_id": 21212,
        "worker_id": 1,
        "org_id": 2,
        "project_id": 3,
        "requested_by_user_id": 4,
        "media_type": "image",
        "model_id": module.QWEN21_MODEL_ID,
        "summary": {
            "image_task": "outpaint",
            "outpaint_contract_version": "qwen21_outpaint_v1",
            "reference_mode": "primary_image_edit",
            "reference_image_count": 1,
            "source_image_count": 1,
            "mask_image_count": 0,
            "control_image_count": 0,
            "placement_mode": "explicit_rectangle",
            "source_scale_mode": "fit_without_crop_no_upscale",
            "accelerator_profile_id": "standard",
            "output_count": 1,
            "output_format": "png",
        },
    }

    assert plugin._candidate_incompatibility_reason(candidate, connection) is None
    candidate["summary"]["source_image_count"] = 2
    assert "source_image_count=1" in plugin._candidate_incompatibility_reason(candidate, connection)
    candidate["summary"]["source_image_count"] = 1
    candidate["summary"]["mask_image_count"] = 1
    assert "does not support mask images" in plugin._candidate_incompatibility_reason(candidate, connection)
    candidate["summary"]["mask_image_count"] = 0
    candidate["summary"]["accelerator_profile_id"] = module.QWEN21_PRUNA_8_PROFILE_ID
    assert "only accelerator_profile_id=standard" in plugin._candidate_incompatibility_reason(candidate, connection)


def test_qwen21_outpaint_candidate_rejects_retired_preserve_original_contract():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    connection = module.ConnectionContext(
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
    summary = {
        "image_task": "outpaint",
        "outpaint_contract_version": "qwen21_outpaint_v1",
        "outpaint_mode": "preserve_original",
        "preserve_original_contract_version": "qwen21_preserve_original_outpaint_v1",
        "preservation_profile_id": "exact_source_v1",
        "reference_mode": "primary_image_edit",
        "reference_image_count": 1,
        "source_image_count": 1,
        "mask_image_count": 0,
        "control_image_count": 0,
        "placement_mode": "explicit_rectangle",
        "source_scale_mode": "fit_without_crop_no_upscale",
        "accelerator_profile_id": "standard",
        "output_count": 1,
        "output_format": "png",
    }
    candidate = {
        "job_id": 21215,
        "worker_id": 1,
        "org_id": 2,
        "project_id": 3,
        "requested_by_user_id": 4,
        "media_type": "image",
        "model_id": module.QWEN21_MODEL_ID,
        "summary": summary,
    }

    assert "Preserve Original is no longer supported" in plugin._candidate_incompatibility_reason(
        candidate, connection
    )
