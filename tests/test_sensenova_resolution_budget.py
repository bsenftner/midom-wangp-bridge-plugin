import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from test_sensenova_prompt_limit import load_plugin_module, validator


@pytest.fixture
def resolutions(monkeypatch):
    configured_root = os.environ.get("WAN2GP_ROOT")
    candidates = [Path(configured_root).expanduser()] if configured_root else list(Path(__file__).resolve().parents)
    root = next((path for path in candidates if (path / "shared/resolutions.py").is_file()), None)
    if root is None:
        if configured_root:
            pytest.fail("WAN2GP_ROOT must point to a WanGP installation containing shared/resolutions.py")
        pytest.skip("Set WAN2GP_ROOT to run the WanGP v18 validator integration tests")
    spec = importlib.util.spec_from_file_location("shared.match_archi", root / "shared/match_archi.py")
    architecture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(architecture)
    monkeypatch.setitem(sys.modules, "shared.match_archi", architecture)
    spec = importlib.util.spec_from_file_location("wangp_resolution_validator_for_test", root / "shared/resolutions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "validate_agent_resolution"):
        pytest.skip("WanGP v18 or newer is required for validator integration tests")
    return module


def session(model_def):
    return types.SimpleNamespace(
        _ensure_runtime=lambda: types.SimpleNamespace(
            module=types.SimpleNamespace(get_model_def=lambda model_id: model_def)
        )
    )


def settings(module, plugin, delivery="1280x720"):
    result = {"model_type": module.SENSENOVA_MODEL_ID}
    plugin._apply_sensenova_job_settings(result, {"inputs": []}, {"reference_mode": "none"}, delivery)
    return result


@pytest.mark.parametrize("delivery", ["1280x720", "720x1280", "1024x1024", "2048x2048"])
def test_v18_validator_accepts_established_internal_resolution_only_during_job(resolutions, delivery):
    module = load_plugin_module()
    plugin = validator(module)
    native = settings(module, plugin, delivery)
    model_def = resolutions.migrate_resolution_definition({
        "resolutions_categories": ["<=4096sq"],
        "guidance_max_phases": 1,
        "vae_block_size": 32,
    })
    original = dict(model_def)
    with pytest.raises(ValueError, match="outside the accepted range"):
        resolutions.validate_agent_resolution(model_def, native["resolution"])
    with plugin._sensenova_high_resolution_budget(session(model_def), native):
        resolutions.validate_agent_resolution(model_def, native["resolution"])
        assert model_def["resolution_budget"] == 5440 * 3072
        assert native["_midom_final_output_resolution"] == delivery
        with pytest.raises(ValueError, match="outside the accepted range"):
            resolutions.validate_agent_resolution(model_def, "8192x8192")
    assert model_def == original


@pytest.mark.parametrize("failure", [RuntimeError("submission failed"), RuntimeError("generation failed"), RuntimeError("cancelled")])
@pytest.mark.parametrize("has_null_budget", [False, True])
def test_resolution_budget_restored_on_failure(failure, has_null_budget):
    module = load_plugin_module()
    plugin = validator(module)
    native = settings(module, plugin)
    model_def = {"resolution_budget": None} if has_null_budget else {}
    original = dict(model_def)
    with pytest.raises(RuntimeError, match=str(failure)):
        with plugin._sensenova_high_resolution_budget(session(model_def), native):
            assert model_def["resolution_budget"] == 5440 * 3072
            raise failure
    assert model_def == original


def test_explicit_wangp_budget_not_overridden(resolutions):
    module = load_plugin_module()
    plugin = validator(module)
    native = settings(module, plugin)
    model_def = {"resolution_budget": "1920x1088"}
    with plugin._sensenova_high_resolution_budget(session(model_def), native):
        with pytest.raises(ValueError, match="outside the accepted range"):
            resolutions.validate_agent_resolution(model_def, native["resolution"])
    assert model_def == {"resolution_budget": "1920x1088"}


def test_other_models_do_not_touch_runtime_budget():
    module = load_plugin_module()
    plugin = validator(module)
    with plugin._sensenova_high_resolution_budget(None, {"model_type": module.QWEN21_MODEL_ID}):
        pass


def test_arbitrary_sensenova_render_size_rejected():
    module = load_plugin_module()
    plugin = validator(module)
    native = settings(module, plugin)
    native["resolution"] = "8192x8192"
    with pytest.raises(ValueError, match="declared Midom render adapter"):
        with plugin._sensenova_high_resolution_budget(None, native):
            pytest.fail("Invalid dimensions must not reach submission")


def test_four_fast_candidates_reach_actual_validator_without_changing_recipe(resolutions):
    module = load_plugin_module()
    plugin = validator(module)
    native = settings(module, plugin)
    plugin._resolve_accelerator_loras = lambda profile: (profile["lora_filenames"], [])
    module.AwsWorkerBridgePlugin._apply_accelerator_profile(
        plugin, native, module.SENSENOVA_MODEL_ID,
        {"accelerator_profile_id": module.SENSENOVA_ACCELERATOR_PROFILE_ID},
    )
    native.update(
        seed=42,
        _midom_media_type="image",
    )
    plugin._capture_qwen21_native_rgba_runtime_diagnostics = lambda *args, **kwargs: None
    plugin._capture_qwen_multi_angle_runtime_diagnostics = lambda *args, **kwargs: None
    model_def = {"guidance_max_phases": 1}
    api = session(model_def)
    submitted = []

    def submit_manifest(tasks, **kwargs):
        for task in tasks:
            resolutions.validate_agent_resolution(model_def, task["resolution"])
            assert task["num_inference_steps"] == 12
            assert task["guidance_scale"] == 1.0
            assert task["activated_loras"] == [module.SENSENOVA_ACCELERATOR_LORA_FILENAME]
            assert not any(key.startswith("_midom_") for key in task)
        submitted.extend(tasks)
        return "job handle"

    api.submit_manifest = submit_manifest
    with plugin._sensenova_high_resolution_budget(api, native):
        assert plugin._submit_wangp_job(api, native, 4, None) == "job handle"
    assert len(submitted) == 4
    assert [task["seed"] for task in submitted] == [42, 43, 44, 45]
    assert "resolution_budget" not in model_def


@pytest.mark.parametrize("outcome", ["success", "submission_error", "wait_error", "cancelled", "generation_error", "upload_error"])
def test_claimed_job_keeps_budget_through_wait_and_restores_before_completion(resolutions, outcome):
    module = load_plugin_module()
    plugin = validator(module)
    native = settings(module, plugin)
    native.update(_midom_media_type="image", _midom_output_count=4)
    model_def = {"guidance_max_phases": 1}
    api = session(model_def)
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._validate_job = lambda *args: dict(native)
    plugin._download_job_inputs = lambda *args: []
    plugin._apply_inputs_to_settings = lambda *args: None
    plugin._callbacks_for_job = lambda *args: None
    plugin._get_generation_session = lambda: api
    plugin._capture_qwen_multi_angle_runtime_diagnostics = lambda *args, **kwargs: None
    plugin._capture_qwen21_native_rgba_runtime_diagnostics = lambda *args, **kwargs: None
    plugin._reset_idle_backoff_for_active_use = lambda *args, **kwargs: None
    generated = [f"candidate-{index}.png" for index in range(4)]
    plugin._result_image_files = lambda result: generated
    plugin._build_generation_metadata = lambda *args: {}
    updates = []
    plugin._post_job_update = lambda connection, job_id, status, payload: updates.append((status, payload))
    handle = object()

    def submit(api_session, submitted_settings, count, callbacks):
        resolutions.validate_agent_resolution(model_def, submitted_settings["resolution"])
        assert count == 4
        if outcome == "submission_error":
            raise RuntimeError("submission failed")
        return handle

    def wait(connection, job_id, submitted_handle, callbacks):
        assert submitted_handle is handle
        resolutions.validate_agent_resolution(model_def, native["resolution"])
        if outcome == "wait_error":
            raise RuntimeError("generation failed")
        return types.SimpleNamespace(
            success=outcome != "generation_error", cancelled=outcome == "cancelled",
            generated_files=generated, errors=["generation failed"] if outcome == "generation_error" else [],
        )

    def upload(connection, job_id, file_path, index, expected_resolution, normalization, *args):
        assert "resolution_budget" not in model_def
        assert expected_resolution == "1280x720"
        assert normalization == "center_crop_downscale"
        if outcome == "upload_error":
            raise RuntimeError("upload failed")
        return {"artifact_index": index, "artifact_id": index + 1}

    plugin._submit_wangp_job = submit
    plugin._wait_for_wangp_result = wait
    plugin._upload_artifact = upload
    connection = types.SimpleNamespace(connection_id="test", worker_id=1)
    plugin._run_job(connection, {
        "job_id": 733, "model_id": module.SENSENOVA_MODEL_ID, "media_type": "image", "inputs": [],
    })
    assert "resolution_budget" not in model_def
    assert plugin._active_job is None
    terminal_updates = [update for update in updates if update[0] in {"complete", "fail"}]
    assert len(terminal_updates) == 1
    assert terminal_updates[0][0] == ("complete" if outcome == "success" else "fail")
    if outcome == "success":
        assert len(terminal_updates[0][1]["artifacts"]) == 4
