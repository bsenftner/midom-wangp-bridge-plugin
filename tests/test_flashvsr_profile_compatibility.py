import ast
import copy
import __future__
import io
import os
import sys
import threading
import types
from pathlib import Path

import pytest
from PIL import Image

from test_flashvsr_crop_upscale_detail import load_plugin_module, restoration_job


@pytest.fixture
def native(monkeypatch, tmp_path):
    root = Path(os.environ.get("WAN2GP_ROOT") or Path(__file__).resolve().parents[3])
    path = root / "postprocessing/flashvsr/wgp_bridge.py"
    if not path.is_file():
        pytest.skip("Native FlashVSR checks require WAN2GP_ROOT")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashVSRBridge")
    namespace = {
        "os": os, "SPARSE_BACKEND_AUTO": "auto", "SPARSE_BACKEND_TRITON_SPARSE": "triton_sparse",
        "SPARSE_BACKEND_SPARGE": "sparge", "normalize_sparse_backend": lambda value: value or "auto",
        "format_multiplier_value": lambda prefix, scale: f"{prefix}*{scale:g}",
        "parse_multiplier_suffix": lambda text, prefix, default: float(text.split("*")[1]) if text.startswith(prefix + "*") else None,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    handler_class = namespace["FlashVSRBridge"]
    if not hasattr(handler_class, "MODEL_PLUS_TURBO"):
        pytest.skip("Plus model tests require the modern native FlashVSR handler")
    config = {"spatial_upsamplers": {"flashvsr": {"mode": 1, "quantization": "int8", "backend": "auto", "topk_ratio": 0.2}}}
    locator = types.SimpleNamespace(locate_file=lambda name, **kwargs: str(tmp_path / name))
    registered = handler_class(config, locator)
    upsamplers = types.ModuleType("postprocessing.spatial_upsamplers")

    def read(server_config, handler):
        sections = server_config.get("spatial_upsamplers", {})
        values = sections.get("flashvsr", handler.legacy_config(server_config))
        return handler.normalize_config_section(values)

    def write(server_config, handler, values):
        values = handler.normalize_config_section(values)
        server_config.setdefault("spatial_upsamplers", {})["flashvsr"] = values
        return values

    upsamplers.read_config_section = read
    upsamplers.write_config_section = write
    upsamplers.find_postprocessing_upsampler = lambda value: registered if registered.is_upsampling(value) else None
    runtime = types.ModuleType("postprocessing.flashvsr.runtime")
    state = types.SimpleNamespace(dit=None, model=None, variant=None, transformer_path=None)
    runtime._RUNTIME = state
    runtime.FlashVSRPaths = types.SimpleNamespace
    loads, downloads, outputs = [], [], []

    def load(paths, *, variant, model="v1_1", profile, **kwargs):
        loads.append((model, variant, paths.transformer))
        state.dit = object()
        state.model, state.variant, state.transformer_path = model, variant, paths.transformer
        state.profile = 0 if hasattr(handler_class, "METHOD_MODELS") or model == "plus_turbo" else profile

    runtime.load_models = load
    runtime.upscale_video = lambda sample, scale, **kwargs: outputs.append((state.model, kwargs)) or "upscaled"
    attention = types.ModuleType("postprocessing.flashvsr.attention_backend")
    attention.set_sparse_backend = lambda backend: None
    download = types.ModuleType("shared.utils.download")
    download.process_files_def_if_needed = lambda definition, **kwargs: downloads.append(definition) or True
    postprocessing = types.ModuleType("postprocessing")
    postprocessing.spatial_upsamplers = upsamplers
    flashvsr = types.ModuleType("postprocessing.flashvsr")
    flashvsr.runtime = runtime
    for name, module in (("postprocessing", postprocessing), ("postprocessing.spatial_upsamplers", upsamplers),
                         ("postprocessing.flashvsr", flashvsr), ("postprocessing.flashvsr.runtime", runtime),
                         ("postprocessing.flashvsr.attention_backend", attention), ("shared.utils.download", download)):
        monkeypatch.setitem(sys.modules, name, module)
    handler_class.vae_tile_size = lambda *args, **kwargs: 0
    module = load_plugin_module()
    plugin = object.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._post_job_update = lambda *args, **kwargs: None
    plugin._set_active_job_status = lambda *args, **kwargs: None
    plugin._active_job_status = {}
    plugin._active_job = None
    plugin._cancel_requested_by_midom = False
    captured = []
    wgp = types.SimpleNamespace(torch=types.SimpleNamespace(__version__="test"))

    def generate_media(task=None, send_cmd=None, *, client_id="", spatial_upsampling="", **kwargs):
        handler = upsamplers.find_postprocessing_upsampler(spatial_upsampling)
        captured.append(handler)
        handler.load_upsampler(spatial_upsampling, process_files=lambda *args, **kwargs: None,
                              init_pipe=lambda *args: 4, profile=4)
        size = (1280, 720)
        if kwargs.get("video_source"):
            with Image.open(kwargs["video_source"]) as image:
                size = image.size
        handler.upscale(types.SimpleNamespace(shape=(3, 1, size[1], size[0])), spatial_upsampling, vae_config=0, still_image=True)
        return True

    wgp.generate_media = generate_media
    session = types.SimpleNamespace(_ensure_runtime=lambda: types.SimpleNamespace(module=wgp))
    return types.SimpleNamespace(module=module, plugin=plugin, config=config, handler_class=handler_class,
                                 registered=registered, upsamplers=upsamplers, runtime=runtime, state=state,
                                 loads=loads, downloads=downloads, outputs=outputs, wgp=wgp,
                                 session=session, captured=captured, tmp_path=tmp_path)


def settings(f, profile):
    diagnostics = f.plugin._flashvsr_image_processing_runtime(profile)
    return {"client_id": "selected-job", "spatial_upsampling": diagnostics["spatial_upsampling"]}, diagnostics


def execute(f, job):
    failures = []
    def run():
        try:
            f.wgp.generate_media(**job)
        except BaseException as error:
            failures.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "execution thread deadlocked"
    if failures:
        raise failures[0]


def use_configuration_selected_protocol(f, monkeypatch):
    """Simulate 18.21's single-method, configuration-selected model interface."""
    monkeypatch.delattr(f.handler_class, "METHOD_MODELS", raising=False)
    monkeypatch.setattr(f.handler_class, "scale_for_upsampling", classmethod(
        lambda cls, value: 2.0 if value == "flashvsr*2" else None))
    monkeypatch.setattr(f.handler_class, "is_two_pass_upsampling", classmethod(lambda cls, value: False))
    def load(handler, value, *, process_files, init_pipe, profile, **kwargs):
        handler.download(process_files)
        f.runtime.load_models(handler.paths(handler.settings()[1]), variant=handler.settings()[1],
                              model=handler.model(), profile=profile, init_pipe=init_pipe)
    monkeypatch.setattr(f.handler_class, "load_upsampler", load)


@pytest.mark.parametrize("profile", ["flashvsr_x2_standard_v1", "flashvsr_plus_turbo_x2_standard_v1"])
def test_configuration_selected_runtime_preserves_both_profiles(native, monkeypatch, profile):
    f = native
    use_configuration_selected_protocol(f, monkeypatch)
    before = copy.deepcopy(f.config)
    job, diagnostics = settings(f, profile)
    assert job["spatial_upsampling"] == "flashvsr*2"
    with f.plugin._flashvsr_image_execution(f.session, job, diagnostics):
        execute(f, job)
    assert f.loads[-1][0] == f.module.FLASHVSR_IMAGE_PROFILES[profile]["model_id"]
    assert diagnostics["verification"] == "loaded_model_verified"
    assert f.config == before


@pytest.mark.parametrize("profile", ["flashvsr_x2_standard_v1", "flashvsr_plus_turbo_x2_standard_v1"])
def test_native_method_and_download_manifest_match_profile(native, profile):
    f = native
    job, diagnostics = settings(f, profile)
    modern = hasattr(f.handler_class, "METHOD_MODELS")
    expected = "flashvsrplus*2" if modern and profile == f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID else "flashvsr*2"
    assert job["spatial_upsampling"] == expected
    model = f.module.FLASHVSR_IMAGE_PROFILES[profile]["model_id"]
    if modern:
        assert f.registered.model_for_upsampling(expected) == model
    transformer = f.handler_class.transformer_filename(model, "bf16")
    assert transformer in [item["filename"] for item in diagnostics["checkpoint_components"]]
    with f.plugin._flashvsr_image_execution(f.session, job, diagnostics):
        execute(f, job)
    downloaded = [filename for request in f.downloads for group in request["fileList"] for filename in group]
    assert transformer in downloaded
    assert Path(f.loads[-1][2]).name == transformer
    assert diagnostics["spatial_upsampling"] == expected


@pytest.mark.parametrize("mapping", [{"flashvsr": "v1_1"}, {"flashvsr": "v1_1", "flashvsrplus": "v1_1"}, "unsupported"])
def test_explicit_method_protocol_never_falls_back_to_another_model(native, monkeypatch, mapping):
    f = native
    monkeypatch.setattr(f.handler_class, "METHOD_MODELS", mapping, raising=False)
    assert f.plugin._flashvsr_image_processing_runtime(f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID) is None


def test_wrong_native_method_is_rejected_before_execution_hooks_are_installed(native):
    f = native
    job, diagnostics = settings(f, f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID)
    job["spatial_upsampling"] = "flashvsr*4"
    generate, find = f.wgp.generate_media, f.upsamplers.find_postprocessing_upsampler
    with pytest.raises(ValueError, match="submission method"):
        with f.plugin._flashvsr_image_execution(f.session, job, diagnostics):
            pytest.fail("An incompatible method reached native execution")
    assert f.wgp.generate_media is generate and f.upsamplers.find_postprocessing_upsampler is find
    assert not f.loads


def test_native_default_cannot_switch_existing_profile_and_polling_is_read_only(native):
    f = native
    before = copy.deepcopy(f.config)
    assert f.registered.config()["model"] == "plus_turbo"
    cap = f.plugin._flashvsr_image_processing_capability()
    limits = cap["limits"]["crop_upscale_detail_restoration"]
    assert limits["profile_ids"] == list(f.module.FLASHVSR_IMAGE_PROFILES)
    assert [p["model_id"] for p in limits["profiles"]] == ["v1_1", "plus_turbo"]
    runtime = f.plugin._flashvsr_image_processing_runtime()
    assert runtime["model_id"] == "v1_1" and runtime["transformer_quantization"] == "bf16"
    assert runtime["model_variant"] == "tiny"
    assert f.config == before


def test_checkpoint_diagnostics_read_grouped_manifest_and_available_file_sizes(native):
    f = native
    transformer = f.tmp_path / "FlashVSR/FlashVSR_plus_turbo_transformer_bf16.safetensors"
    transformer.parent.mkdir()
    transformer.write_bytes(b"model")
    codec = f.tmp_path / f.handler_class.VAE_FILENAME
    codec.write_bytes(b"codec-data")
    runtime = f.plugin._flashvsr_image_processing_runtime(f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID)
    components = {item["filename"]: item for item in runtime["checkpoint_components"]}
    assert components[transformer.name]["bytes"] == 5
    assert components[codec.name]["bytes"] == 10
    assert any("lq_proj" in name for name in components)


def test_wait_error_cancels_and_joins_before_restoring_native_routing(native, monkeypatch):
    f = native
    before = copy.deepcopy(f.config)
    find, generate = f.upsamplers.find_postprocessing_upsampler, f.wgp.generate_media
    source = f.tmp_path / "crop.png"
    Image.new("RGB", (1280, 720)).save(source)
    claim = f.plugin._validate_flashvsr_image_processing_job(restoration_job(f.module), 9001)
    entered, stop = threading.Event(), threading.Event()
    original_load = f.runtime.load_models
    errors = []
    threads = []
    def load(*args, **kwargs):
        original_load(*args, **kwargs)
        entered.set()
        assert stop.wait(timeout=5)
    monkeypatch.setattr(f.runtime, "load_models", load)
    def submit(public_settings, **kwargs):
        def worker():
            try:
                f.wgp.generate_media(**public_settings)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)
        def result():
            assert f.wgp.generate_media is not generate
            assert f.upsamplers.find_postprocessing_upsampler is not find
            thread.join(timeout=5)
            assert not thread.is_alive()
        return types.SimpleNamespace(cancel=lambda: stop.set(), result=result)
    f.session.submit_task = submit
    f.plugin._get_generation_session = lambda: f.session
    f.plugin._callbacks_for_job = lambda *args: None
    def wait(*args):
        assert entered.wait(timeout=5)
        raise PermissionError("authorization or lease rejected")
    f.plugin._wait_for_wangp_result = wait
    with pytest.raises(PermissionError, match="authorization"):
        f.plugin._run_flashvsr_image_processing_job(
            types.SimpleNamespace(worker_id=7), 9001, claim,
            [{"kind": "source_image", "path": str(source), "width": 1280, "height": 720}],
            str(f.tmp_path), types.SimpleNamespace(cancelled=False))
    assert not errors and not any(thread.is_alive() for thread in threads)
    assert f.wgp.generate_media is generate and f.upsamplers.find_postprocessing_upsampler is find
    assert f.config == before


@pytest.mark.parametrize("profile", ["flashvsr_x2_standard_v1", "flashvsr_plus_turbo_x2_standard_v1"])
def test_candidate_and_claim_retain_exact_profile(native, profile):
    f = native
    job = restoration_job(f.module)
    job["processing"]["profile_id"] = profile
    claim = f.plugin._validate_flashvsr_image_processing_job(job, 9001)
    assert claim["_midom_processing"]["profile_id"] == profile
    candidate = {key: job[key] for key in ("family", "media_type", "processor_id", "processing_task", "operation_type")}
    candidate["summary"] = {"profile_id": profile, "source_image_count": 1, "input_color_mode": "rgb",
                            "output_format": "png", "output_color_mode": "rgb", "output_count": 1,
                            "output_width": 1280, "output_height": 720}
    assert f.plugin._flashvsr_image_candidate_incompatibility_reason(candidate) is None
    candidate["summary"]["profile_id"] = "not-a-profile"
    assert "profile_id" in f.plugin._flashvsr_image_candidate_incompatibility_reason(candidate)
    job["processing"]["profile_id"] = "not-a-profile"
    with pytest.raises(ValueError, match="profile_id"):
        f.plugin._validate_flashvsr_image_processing_job(job, 9001)


def test_native_handler_switching_and_thread_scoped_execution(native):
    f = native
    f.registered.settings()
    before = copy.deepcopy(f.config)
    find, generate = f.upsamplers.find_postprocessing_upsampler, f.wgp.generate_media
    for profile in [f.module.FLASHVSR_X2_STANDARD_PROFILE_ID, f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID,
                    f.module.FLASHVSR_X2_STANDARD_PROFILE_ID]:
        job, diagnostics = settings(f, profile)
        with f.plugin._flashvsr_image_execution(f.session, job, diagnostics):
            # Same operation queried from polling/another thread still sees the registered handler.
            assert f.upsamplers.find_postprocessing_upsampler("flashvsr*2") is f.registered
            assert f.plugin._flashvsr_image_processing_runtime()["verification"] == "configuration_only"
            execute(f, job)
            # A different client on the SAME thread must not inherit the selected handler.
            f.wgp.generate_media(client_id="another-job", spatial_upsampling="flashvsr*2")
            assert f.captured[-1] is f.registered
        assert f.upsamplers.find_postprocessing_upsampler is find
        assert f.wgp.generate_media is generate
        assert diagnostics["verification"] == "loaded_model_verified"
        assert diagnostics["profile_id"] == profile
        assert f.config == before
    selected = [entry for entry in f.loads[::2]]
    assert [entry[0] for entry in selected] == ["v1_1", "plus_turbo", "v1_1"]
    assert all(entry[2].endswith("_bf16.safetensors") for entry in selected)
    assert selected[1][1] == "full"
    # Only explicitly executing the unrelated native job can normalize its operator config.
    assert f.registered.config()["quantization"] == "int8"


def test_unrelated_thread_and_operator_change_survive_active_execution(native, monkeypatch):
    f = native
    job, diagnostics = settings(f, f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID)
    loaded, release = threading.Event(), threading.Event()
    original = f.runtime.load_models
    def load(*args, **kwargs):
        original(*args, **kwargs)
        loaded.set()
        assert release.wait(timeout=5)
    monkeypatch.setattr(f.runtime, "load_models", load)
    errors = []
    def worker():
        try:
            f.wgp.generate_media(**job)
        except BaseException as error:
            errors.append(error)
    with f.plugin._flashvsr_image_execution(f.session, job, diagnostics):
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            assert loaded.wait(timeout=5)
            assert f.upsamplers.find_postprocessing_upsampler("flashvsr*2") is f.registered
            assert f.plugin._flashvsr_image_processing_runtime()["model_id"] == "v1_1"
            f.config["operator_change"] = "keep"
        finally:
            release.set()
            thread.join(timeout=5)
        assert not thread.is_alive() and not errors
    assert f.config["operator_change"] == "keep"
    assert diagnostics["model_id"] == "plus_turbo"


@pytest.mark.parametrize("failure", ["submit", "inference", "cancel_requested", "wrong_model", "wrong_precision"])
def test_hooks_restore_and_wrong_loaded_model_fails_before_upscale(native, monkeypatch, failure):
    f = native
    before = copy.deepcopy(f.config)
    find, generate = f.upsamplers.find_postprocessing_upsampler, f.wgp.generate_media
    job, diagnostics = settings(f, f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID)
    original = f.runtime.load_models
    def load(*args, **kwargs):
        original(*args, **kwargs)
        if failure == "wrong_model":
            f.state.model = "v1_1"
        elif failure == "wrong_precision":
            f.state.transformer_path = "/wrong/int8.safetensors"
        else:
            raise RuntimeError(failure)
    monkeypatch.setattr(f.runtime, "load_models", load)
    with pytest.raises((RuntimeError, ValueError)):
        with f.plugin._flashvsr_image_execution(f.session, job, diagnostics):
            if failure == "submit":
                raise RuntimeError("submit")
            execute(f, job)
    assert f.wgp.generate_media is generate and f.upsamplers.find_postprocessing_upsampler is find
    assert f.config == before
    assert not f.outputs


def test_legacy_handler_retains_v11_and_rejects_plus(native, monkeypatch):
    f = native
    use_configuration_selected_protocol(f, monkeypatch)
    monkeypatch.delattr(f.handler_class, "_MODEL_FILE_STEMS")
    # Model/quantization fields do not exist on the known pre-Plus handler.
    original = f.handler_class.normalize_config_section.__func__
    def normalize(cls, config):
        cls._MODEL_FILE_STEMS = {"v1_1": "FlashVSR_v1.1"}
        try:
            result = original(cls, config)
        finally:
            del cls._MODEL_FILE_STEMS
        result.pop("model", None)
        result.pop("quantization", None)
        return result
    monkeypatch.setattr(f.handler_class, "normalize_config_section", classmethod(normalize))
    monkeypatch.setattr(f.handler_class, "query_download_def", lambda self: {
        "fileList": [self.TRANSFORMER_FILENAME, self.LQ_PROJ_FILENAME, self.TCDECODER_FILENAME]})
    legacy = f.plugin._flashvsr_profile_handler(f.module.FLASHVSR_X2_STANDARD_PROFILE_ID)
    assert legacy.TRANSFORMER_FILENAME.endswith("v1.1_transformer_bf16.safetensors")
    with pytest.raises(ValueError, match="interface"):
        f.plugin._flashvsr_profile_handler(f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID)
    cap = f.plugin._flashvsr_image_processing_capability()
    assert cap["limits"]["crop_upscale_detail_restoration"]["profile_ids"] == [f.module.FLASHVSR_X2_STANDARD_PROFILE_ID]
    monkeypatch.setattr(f.handler_class, "TRANSFORMER_FILENAME", "unknown_model.safetensors")
    assert f.plugin._flashvsr_image_processing_capability() is None


def test_disabled_extension_and_unsupported_native_model_are_not_advertised(native, monkeypatch):
    f = native
    f.config["spatial_upsamplers"]["flashvsr"]["mode"] = 0
    before = copy.deepcopy(f.config)
    assert f.plugin._flashvsr_image_processing_capability() is None
    assert f.config == before
    f.config["spatial_upsamplers"]["flashvsr"]["mode"] = 1
    monkeypatch.setattr(f.handler_class, "_MODEL_FILE_STEMS", {"v1_1": "FlashVSR_v1.1"})
    cap = f.plugin._flashvsr_image_processing_capability()
    assert cap["limits"]["crop_upscale_detail_restoration"]["profile_ids"] == [f.module.FLASHVSR_X2_STANDARD_PROFILE_ID]


def test_native_cache_reloads_when_model_decoder_or_weight_path_changes(native):
    root = Path(os.environ.get("WAN2GP_ROOT") or Path(__file__).resolve().parents[3])
    path = root / "postprocessing/flashvsr/runtime.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "FlashVSRRuntime")
    load = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load")
    load.decorator_list = []
    namespace = {"require_sparge_attention": lambda: None, "FLASHVSR_MODEL_V11": "v1_1", "FLASHVSR_VARIANT_TINY_LONG": "tiny_long",
                 "FLASHVSR_PLUS_MODELS": ("plus_turbo",)}
    exec(compile(ast.Module(body=[load], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    class ReloadRequired(Exception):
        pass
    def release():
        raise ReloadRequired()
    state = types.SimpleNamespace(dit=object(), variant="tiny", profile=0, model="v1_1",
                                  transformer_path="v11_bf16.safetensors", release=release)
    kwargs = {"variant": "tiny", "profile": 0, "init_pipe": None, "model": "v1_1"}
    namespace["load"](state, types.SimpleNamespace(transformer=state.transformer_path), **kwargs)
    for changes in ({"variant": "full"}, {"model": "plus_turbo"}):
        with pytest.raises(ReloadRequired):
            namespace["load"](state, types.SimpleNamespace(transformer=state.transformer_path), **{**kwargs, **changes})
    with pytest.raises(ReloadRequired):
        namespace["load"](state, types.SimpleNamespace(transformer="v11_int8.safetensors"), **kwargs)
    if hasattr(native.handler_class, "METHOD_MODELS"):
        # Current WanGP normalizes every FlashVSR load to profile 0 before its cache check.
        namespace["load"](state, types.SimpleNamespace(transformer=state.transformer_path), **{**kwargs, "profile": 4})
    else:
        with pytest.raises(ReloadRequired):
            namespace["load"](state, types.SimpleNamespace(transformer=state.transformer_path), **{**kwargs, "profile": 4})


@pytest.mark.parametrize("condition", ["cancelled", "failed", "unobserved"])
def test_canceled_failed_and_unobserved_native_jobs_never_upload_or_complete(native, condition):
    f = native
    before = copy.deepcopy(f.config)
    find, generate = f.upsamplers.find_postprocessing_upsampler, f.wgp.generate_media
    source = f.tmp_path / "crop.png"
    Image.new("RGB", (1280, 720)).save(source)
    plugin = f.plugin
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._validate_job_scope = lambda *args: None
    plugin._reset_idle_backoff_for_active_use = lambda *args, **kwargs: None
    plugin._callbacks_for_job = lambda *args: None
    plugin._get_generation_session = lambda: f.session
    f.session.submit_task = lambda *args, **kwargs: types.SimpleNamespace(done=True)
    plugin._wait_for_wangp_result = lambda *args: types.SimpleNamespace(
        success=condition != "failed", cancelled=condition == "cancelled", errors=["native execution failed"],
        generated_files=[str(source)], artifacts=[])
    plugin._download_job_inputs = lambda *args: [{"kind": "source_image", "path": str(source), "width": 1280, "height": 720}]
    def no_upload(*args):
        pytest.fail("An invalid native result reached artifact upload")
    plugin._upload_media_processing_artifact = no_upload
    updates = []
    plugin._post_job_update = lambda connection, job_id, status, payload: updates.append((status, payload))
    plugin._run_job(types.SimpleNamespace(worker_id=7, connection_id="worker-7"), restoration_job(f.module))
    assert updates[-1][0] == "fail"
    if condition == "cancelled":
        assert updates[-1][1]["reason"] == "cancel_requested"
    elif condition == "unobserved":
        assert "not observed" in updates[-1][1]["message"]
    else:
        assert "native execution failed" in updates[-1][1]["message"]
    assert f.wgp.generate_media is generate and f.upsamplers.find_postprocessing_upsampler is find
    assert f.config == before


@pytest.mark.parametrize("size", [(768, 768), (1024, 1024), (1280, 720), (720, 1280)])
@pytest.mark.parametrize("profile", ["flashvsr_x2_standard_v1", "flashvsr_plus_turbo_x2_standard_v1"])
def test_execution_through_final_png_and_typed_upload(native, monkeypatch, size, profile):
    f = native
    plugin = f.plugin
    job = restoration_job(f.module, width=size[0], height=size[1])
    job["processing"]["profile_id"] = profile
    source = f.tmp_path / "crop.png"
    Image.new("RGB", size, (70, 80, 90)).save(source)
    generated = f.tmp_path / "native.png"
    Image.new("RGB", (size[0] * 2, size[1] * 2), (90, 100, 110)).save(generated)
    def submit(public_settings, **kwargs):
        execute(f, public_settings)
        return types.SimpleNamespace(done=True)
    f.session.submit_task = submit
    plugin._get_generation_session = lambda: f.session
    plugin._callbacks_for_job = lambda *args: None
    plugin._wait_for_wangp_result = lambda *args: types.SimpleNamespace(
        success=True, cancelled=False, generated_files=[str(generated)], artifacts=[])
    connection = types.SimpleNamespace(worker_id=7, api_base_url="https://midom.test", connection_id="worker-7")
    updates, uploads = [], []
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._validate_job_scope = lambda *args: None
    plugin._reset_idle_backoff_for_active_use = lambda *args, **kwargs: None
    plugin._post_job_update = lambda connection, job_id, status, payload: updates.append((status, payload))
    plugin._headers = lambda connection: {}
    class Response:
        headers = {"Content-Type": "image/png"}
        status_code = 200
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size):
            yield source.read_bytes()
        def close(self):
            pass
    monkeypatch.setattr(f.module.requests, "get", lambda *args, **kwargs: Response())
    def post(url, **kwargs):
        uploads.append((kwargs["data"], kwargs["files"]["file"][1].read()))
        return types.SimpleNamespace(status_code=200, json=lambda: {
            "artifact_id": 40, "file_id": 50, "artifact_index": 0, "role": "restored_crop_image"})
    monkeypatch.setattr(f.module.requests, "post", post)
    plugin._run_job(connection, job)
    assert updates[-1][0] == "complete", updates[-1]
    payload = updates[-1][1]
    assert payload["backend"] == "wangp_flashvsr"
    assert payload["artifacts"][0]["role"] == "restored_crop_image"
    metadata = payload["processing_metadata"]
    assert metadata["profile_id"] == profile
    assert metadata["flashvsr_runtime"]["verification"] == "loaded_model_verified"
    expected_method = ("flashvsrplus*2" if hasattr(f.handler_class, "METHOD_MODELS")
                       and profile == f.module.FLASHVSR_PLUS_TURBO_X2_PROFILE_ID else "flashvsr*2")
    assert metadata["flashvsr_runtime"]["spatial_upsampling"] == expected_method
    assert metadata["final_dimensions"] == f"{size[0]}x{size[1]}"
    assert metadata["native_intermediate_dimensions"] == f"{size[0]*2}x{size[1]*2}"
    assert uploads[0][0]["role"] == "restored_crop_image"
    assert metadata["final_sha256"] == f.module.hashlib.sha256(uploads[0][1]).hexdigest()
    with Image.open(io.BytesIO(uploads[0][1])) as image:
        assert image.mode == "RGB" and image.format == "PNG" and image.size == size
