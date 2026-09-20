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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_image_prompt_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._resolve_job_resolution = lambda job, model_id=None: "1280x720"
    plugin._apply_accelerator_profile = lambda settings, model_id, generation: "standard"
    return plugin


def qwen_multiline_job():
    return {
        "job_id": 901,
        "media_type": "image",
        "model_id": "qwen_image_edit_plus2_20B",
        "prompt": "\n".join([
            "Create a clean product-style character turnaround.",
            "Keep the same outfit and facial identity.",
            "Use a neutral studio background.",
            "Return a polished editorial image.",
        ]),
        "generation": {
            "options": {
                "multi_prompts_gen_type": "FG",
            },
        },
        "output": {"count": 1, "format": "png", "width": 1280, "height": 720},
        "inputs": [
            {"kind": "reference_image", "input_id": 1},
            {"kind": "reference_image", "input_id": 2},
            {"kind": "reference_image", "input_id": 3},
        ],
    }


class FakeApiSession:
    def __init__(self):
        self.submitted_settings = None

    def submit_task(self, settings, callbacks=None):
        self.submitted_settings = dict(settings)
        return types.SimpleNamespace(done=True, result=lambda: types.SimpleNamespace(success=True, generated_files=[], errors=[]))

    def submit_manifest(self, tasks, callbacks=None):
        raise AssertionError("Generic Midom image jobs should submit one task, not a split manifest.")


def test_qwen_edit_plus2_multiline_prompt_uses_fg_for_single_task_submit():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(qwen_multiline_job(), 901)

    assert settings["model_type"] == "qwen_image_edit_plus2_20B"
    assert settings["multi_prompts_gen_type"] == "FG"
    assert settings["_midom_prompt_processing_mode"] == "FG"

    api_session = FakeApiSession()
    plugin._submit_wangp_job(api_session, settings, 1, callbacks=types.SimpleNamespace())

    assert api_session.submitted_settings is not None
    assert api_session.submitted_settings["multi_prompts_gen_type"] == "FG"
    assert "_midom_prompt_processing_mode" not in api_session.submitted_settings


def test_generic_image_rejects_prompt_splitting_modes():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen_multiline_job()
    job["generation"]["options"]["multi_prompts_gen_type"] = "PG"

    with pytest.raises(ValueError, match="multi_prompts_gen_type='FG'"):
        plugin._validate_image_job(job, 902)


def test_generic_image_defaults_to_fg_when_option_is_missing():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = qwen_multiline_job()
    job["generation"]["options"] = {}

    settings = plugin._validate_image_job(job, 903)

    assert settings["multi_prompts_gen_type"] == "FG"


def test_sensenova_preserves_existing_explicit_fg_handling():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_image_job(
        {
            "job_id": 904,
            "media_type": "image",
            "model_id": module.SENSENOVA_MODEL_ID,
            "prompt": "Create a precise infographic with short readable labels.",
            "generation": {
                "reference_mode": "none",
                "native_high_res_render": True,
                "prompt_enhancement_by_worker": False,
                "options": {"multi_prompts_gen_type": "FG"},
            },
            "output": {"count": 1, "format": "png", "width": 1280, "height": 720},
            "inputs": [],
        },
        904,
    )

    assert settings["model_type"] == module.SENSENOVA_MODEL_ID
    assert settings["multi_prompts_gen_type"] == "FG"
