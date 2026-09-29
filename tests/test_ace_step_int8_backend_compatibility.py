import importlib.util
import sys
import types
from pathlib import Path


def _load_plugin_module():
    sys.modules.setdefault("gradio", types.SimpleNamespace(Error=Exception))

    shared = types.ModuleType("shared")
    api = types.ModuleType("shared.api")
    api.init = lambda: None
    utils = types.ModuleType("shared.utils")
    plugins = types.ModuleType("shared.utils.plugins")
    plugins.WAN2GPPlugin = object
    sys.modules.setdefault("shared", shared)
    sys.modules.setdefault("shared.api", api)
    sys.modules.setdefault("shared.utils", utils)
    sys.modules.setdefault("shared.utils.plugins", plugins)

    plugin_path = Path(__file__).resolve().parents[1] / "plugin.py"
    spec = importlib.util.spec_from_file_location("midom_bridge_ace_int8_test", plugin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _KitchenBackend:
    @staticmethod
    def kitchen_enabled():
        return True


class _PyTorchBackend:
    @staticmethod
    def kitchen_enabled():
        return False


class _RuntimeModule:
    def __init__(self, *, quantization="int8", backend=None, selection="auto"):
        self.transformer_quantization = quantization
        self.int8_backend = backend or _KitchenBackend()
        self.int8_kernels = selection
        self.applied = []

    def apply_int8_kernel_setting(self, selection):
        self.applied.append(selection)


class _Session:
    def __init__(self, module):
        self.runtime = types.SimpleNamespace(module=module)

    def _ensure_runtime(self):
        return self.runtime


def _plugin(module):
    plugin = object.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    return plugin


def test_ace_step_int8_kitchen_uses_scoped_pytorch_backend_and_restores():
    module = _load_plugin_module()
    runtime_module = _RuntimeModule()
    plugin = _plugin(module)

    state = plugin._begin_ace_step15_int8_backend_compatibility(
        _Session(runtime_module),
        {"model_type": module.ACE_STEP15_WANGP_MODEL_TYPE},
    )

    assert state == (runtime_module, "auto")
    assert runtime_module.applied == ["disabled"]

    plugin._restore_ace_step15_int8_backend(state)
    assert runtime_module.applied == ["disabled", "auto"]


def test_ace_step_bf16_does_not_change_backend():
    module = _load_plugin_module()
    runtime_module = _RuntimeModule(quantization="bf16")
    plugin = _plugin(module)

    state = plugin._begin_ace_step15_int8_backend_compatibility(
        _Session(runtime_module),
        {"model_type": module.ACE_STEP15_WANGP_MODEL_TYPE},
    )

    assert state is None
    assert runtime_module.applied == []


def test_ace_step_non_kitchen_int8_does_not_change_backend():
    module = _load_plugin_module()
    runtime_module = _RuntimeModule(backend=_PyTorchBackend())
    plugin = _plugin(module)

    state = plugin._begin_ace_step15_int8_backend_compatibility(
        _Session(runtime_module),
        {"model_type": module.ACE_STEP15_WANGP_MODEL_TYPE},
    )

    assert state is None
    assert runtime_module.applied == []


def test_other_models_do_not_change_backend():
    module = _load_plugin_module()
    runtime_module = _RuntimeModule()
    plugin = _plugin(module)

    state = plugin._begin_ace_step15_int8_backend_compatibility(
        _Session(runtime_module),
        {"model_type": "stable_audio3_small"},
    )

    assert state is None
    assert runtime_module.applied == []
