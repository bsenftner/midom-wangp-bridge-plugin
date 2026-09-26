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


def enable_viggle(plugin, module):
    plugin._resolve_accelerator_loras = lambda profile: (
        [module.QWEN21_VIGGLE_LORA_FILENAME],
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
    plugin._resolve_accelerator_loras = lambda profile: (
        [module.QWEN21_VIGGLE_LORA_FILENAME] if profile.get("profile_id") == module.QWEN21_VIGGLE_PROFILE_ID else [],
        [],
    )

    capability = next(
        item for item in plugin._capabilities()["models"]
        if item.get("model_id") == module.QWEN21_MODEL_ID
    )

    assert capability["display_name"] == "Qwen Image 2.1 7B"
    assert capability["capabilities"]["image_edit"] is True
    assert capability["capabilities"]["ordered_reference_images"] is True
    assert capability["capabilities"]["control"] is False
    assert capability["capabilities"]["inpaint"] is False
    assert capability["capabilities"]["rgba"] is False
    assert capability["limits"]["max_reference_images"] == 10
    assert capability["limits"]["output_mime_types"] == ["image/png"]
    assert capability["limits"]["internal_render_resolutions"]["1280x720"] == "1280x736"
    profiles = {item["profile_id"]: item for item in capability["accelerator_profiles"]}
    assert profiles["standard"]["steps"] == 40
    assert profiles["standard"]["max_reference_images"] == 10
    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["steps"] == 6
    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["max_reference_images"] == 3
    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["negative_prompt"] is False


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


def test_qwen21_viggle_is_reported_unavailable_without_lora():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    plugin._resolve_accelerator_loras = lambda profile: ([], [module.QWEN21_VIGGLE_LORA_FILENAME])

    profiles = {item["profile_id"]: item for item in plugin._accelerator_profiles_for_model(module.QWEN21_MODEL_ID)}

    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["available"] is False
    assert profiles[module.QWEN21_VIGGLE_PROFILE_ID]["unavailable_reason"] == "required_accelerator_files_not_installed"


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
    )

    assert captured["mime_type"] == "image/png"
    assert captured["payload"].startswith(b"\x89PNG\r\n\x1a\n")
    normalized = tmp_path / "normalized.png"
    normalized.write_bytes(captured["payload"])
    with Image.open(normalized) as image:
        assert image.size == (1280, 720)
        assert image.format == "PNG"


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
