import __future__
import ast
import copy
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from test_fish_audio_drama3 import load_plugin


@pytest.fixture
def native(monkeypatch):
    module = load_plugin()
    root = Path(os.environ.get("WAN2GP_ROOT") or Path(__file__).resolve().parents[3])
    bridge_path = root / "postprocessing/seedvc_processor/bridge.py"
    if not bridge_path.is_file():
        pytest.skip("Native SeedVC compatibility checks require WAN2GP_ROOT")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, value)
        exec(compile(path.read_text(), str(path), "exec", flags=__future__.annotations.compiler_flag), value.__dict__)
        return value

    package = types.ModuleType("postprocessing")
    package.__path__ = [str(root / "postprocessing")]
    monkeypatch.setitem(sys.modules, "postprocessing", package)
    seed_package = types.ModuleType("postprocessing.seedvc_processor")
    seed_package.__path__ = [str(bridge_path.parent)]
    monkeypatch.setitem(sys.modules, seed_package.__name__, seed_package)
    monkeypatch.setattr(sys.modules["shared.utils"], "offload_registry", types.SimpleNamespace(), raising=False)
    for name in ("memory", "processor_status"):
        load("postprocessing." + name, root / "postprocessing" / (name + ".py"))
    api = load("postprocessing.audio_processors", root / "postprocessing/audio_processors.py")
    package.audio_processors = api

    # Keep the real native config/registry behavior, without importing GPU loaders.
    bridge = types.ModuleType("postprocessing.seedvc_processor.bridge")
    tree = ast.parse(bridge_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SeedVCBridge")
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(bridge_path), "exec",
                 flags=__future__.annotations.compiler_flag), bridge.__dict__)
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    processor_module = load("postprocessing.seedvc_processor.audio_processor", bridge_path.with_name("audio_processor.py"))
    config = {"audio_processors": {"seedvc": {"mode": 1}}}
    handler = processor_module.SeedVCProcessor(config, types.SimpleNamespace())
    api._audio_processor_handlers = [handler]
    plugin = object.__new__(module.AwsWorkerBridgePlugin)
    logs = []
    plugin._log = lambda message, **kwargs: logs.append(message)
    plugin._audio_output_mime_types = lambda: ["audio/wav"]
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._validate_job_scope = lambda *args: None
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 2, "project_id": 3, "paired_user_id": 4}[key]
    return types.SimpleNamespace(module=module, plugin=plugin, api=api, handler=handler,
                                 bridge=bridge, config=config, logs=logs)


def job_and_candidate():
    job = {"job_id": 123, "media_type": "audio", "model_id": "seedvc_voice_replacement",
           "generation": {"audio_task": "voice_conversion", "audio_category": "voice_replacement",
                          "voice_mode": "one_speaker_voice_replacement",
                          "options": {"seedvc_method": "seedvc_one_speaker"}},
           "inputs": [{"kind": "source_audio", "input_id": 10},
                      {"kind": "reference_audio", "input_id": 11}],
           "output": {"count": 1, "format": "wav"}}
    candidate = {"job_id": 123, "worker_id": 7, "org_id": 2, "project_id": 3,
                 "requested_by_user_id": 4, "media_type": "audio", "model_id": job["model_id"],
                 "summary": {"audio_task": "voice_conversion", "audio_category": "voice_replacement",
                             "voice_mode": "one_speaker_voice_replacement", "source_audio_count": 1,
                             "reference_audio_count": 1, "output_count": 1, "output_format": "wav"}}
    return json.loads(json.dumps(job)), json.loads(json.dumps(candidate))


def test_discovery_with_current_native_speech_processor(native):
    before = copy.deepcopy(native.config)
    assert not hasattr(native.bridge.SeedVCBridge, "MODE_OFF")
    assert native.plugin._seedvc_speech_available(), native.logs
    assert native.config == before


@pytest.mark.parametrize("mode", [2, 3, 0, None, "invalid", {}, False])
def test_non_speech_modes_never_advertise_or_accept_claim(native, mode):
    native.config["audio_processors"]["seedvc"]["mode"] = mode
    before = copy.deepcopy(native.config)
    job, candidate = job_and_candidate()
    assert not native.plugin._seedvc_speech_available()
    assert native.plugin._candidate_incompatibility_reason(candidate, None) == "SeedVC speech voice replacement is not enabled"
    with pytest.raises(ValueError, match="SeedVC speech voice replacement is not enabled"):
        native.plugin._validate_job(job, None)
    assert native.config == before


def test_missing_native_mode_uses_singing_default_not_speech(native):
    native.config["audio_processors"]["seedvc"].clear()
    assert not native.plugin._seedvc_speech_available()
    assert native.config == {"audio_processors": {"seedvc": {}}}


@pytest.mark.parametrize("problem", ["unregistered", "wrong_stage", "validation_error", "missing_config", "disabled"])
def test_unavailable_processors_stay_unavailable(native, monkeypatch, problem):
    if problem == "unregistered":
        native.api._audio_processor_handlers = []
    elif problem == "wrong_stage":
        monkeypatch.setattr(native.api, "method_has_type", lambda *args: False)
    elif problem == "validation_error":
        monkeypatch.setattr(native.handler, "validate_method", lambda *args, **kwargs: "invalid native method")
    elif problem == "missing_config":
        native.handler.server_config = None
    else:
        monkeypatch.setattr(native.handler, "method_available", lambda method: False, raising=False)
    assert not native.plugin._seedvc_speech_available(), native.logs


def test_missing_mode_off_constant_is_not_needed_for_zero_mode(native, monkeypatch):
    monkeypatch.setattr(native.api, "read_config_section", lambda *args: {"mode": 0})
    assert not native.plugin._seedvc_speech_available()
    assert any("not speech v1. mode=0" in line for line in native.logs)
    assert not any("capability check failed" in line for line in native.logs)


def test_speech_candidate_claim_and_submission_use_same_native_method(native):
    plugin = native.plugin
    job, candidate = job_and_candidate()
    before = copy.deepcopy(native.config)
    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    settings = plugin._validate_job(job, None)
    settings.update(audio_source="source.wav", replace_voice_sample="voice.wav")
    captured = []
    handle = object()

    class Session:
        def submit_audio_postprocessing(self, source, **kwargs):
            captured.append((source, kwargs))
            return handle

    callbacks = object()
    assert plugin._submit_wangp_job(Session(), settings, 1, callbacks) is handle
    assert captured == [("source.wav", {"callbacks": callbacks, "postprocess_audio": "seedvc_one_speaker",
                                         "replace_voice_sample": "voice.wav", "replace_voice_sample2": None})]
    assert settings["_midom_seedvc_runtime_mode"] == "speech_v1"
    assert native.config == before


def test_capability_payload_retains_existing_speech_only_contract(native):
    plugin = native.plugin
    plugin._curated_tools_for_model = lambda model_id: []
    plugin._resolve_accelerator_loras = lambda profile: ([], [])
    for name in ("_event_video_processing_capability", "_storyboard_ffmpeg_processing_capability",
                 "_flashvsr_image_processing_capability", "_ltx25_alpha_matte_processing_capability",
                 "_matanyone_v2_alpha_matte_processing_capability", "_sam_click_seed_mask_processing_capability"):
        setattr(plugin, name, lambda: None)
    models = {item["model_id"]: item for item in plugin._capabilities()["models"]}
    record = models["seedvc_voice_replacement"]
    assert record["limits"]["seedvc_method"] == "seedvc_one_speaker"
    assert record["limits"]["seedvc_runtime_mode"] == "speech_v1"
    assert record["limits"]["max_outputs"] == 1
    assert not record["capabilities"]["text_to_speech"]
    native.config["audio_processors"]["seedvc"]["mode"] = 2
    assert "seedvc_voice_replacement" not in {item["model_id"] for item in plugin._capabilities()["models"]}
