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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_for_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validator(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._resolve_job_resolution = lambda job, model_id=None: "1280x720"
    plugin._apply_accelerator_profile = lambda settings, model_id, generation: "standard"
    return plugin


def test_sensenova_accepts_prompt_over_default_4000_limit():
    module = load_plugin_module()
    settings = validator(module)._validate_image_job(
        {
            "job_id": 1,
            "media_type": "image",
            "model_id": module.SENSENOVA_MODEL_ID,
            "prompt": "x" * 4181,
            "generation": {
                "reference_mode": "none",
                "native_high_res_render": True,
                "prompt_enhancement_by_worker": False,
            },
            "output": {"count": 1, "format": "png", "width": 1280, "height": 720},
            "inputs": [],
        },
        1,
    )

    assert settings["model_type"] == module.SENSENOVA_MODEL_ID
    assert len(settings["prompt"]) == 4181
    assert settings["resolution"] == "5440x3072"


def test_non_sensenova_image_prompt_still_uses_default_4000_limit():
    module = load_plugin_module()
    with pytest.raises(ValueError, match="Job prompt exceeds 4000 characters"):
        validator(module)._validate_image_job(
            {
                "job_id": 2,
                "media_type": "image",
                "model_id": "qwen_image_20B",
                "prompt": "x" * 4001,
                "generation": {},
                "output": {"count": 1, "format": "png", "width": 1280, "height": 720},
                "inputs": [],
            },
            2,
        )
