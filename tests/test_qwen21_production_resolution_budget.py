import types

import pytest

from test_qwen21_scene_integration import (
    draft_job,
    flattened_production_job,
    load_plugin_module,
    plugin_instance,
    production_job,
)
from test_sensenova_resolution_budget import resolutions, session


CONTRACTS = [production_job, flattened_production_job]
DELIVERY_SIZES = ["768x768", "1024x1024", "1280x720", "720x1280"]


def validated_settings(plugin, factory, delivery="1024x1024"):
    job = factory()
    width, height = map(int, delivery.split("x"))
    job["output"].update(width=width, height=height)
    return plugin._validate_image_job(job, job["job_id"])


@pytest.mark.parametrize("factory", CONTRACTS)
@pytest.mark.parametrize("delivery", DELIVERY_SIZES)
def test_both_production_contracts_pass_v18_validator_without_changing_recipe(resolutions, factory, delivery):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = validated_settings(plugin, factory, delivery)
    before = dict(settings)
    model_def = resolutions.migrate_resolution_definition({
        "resolutions_categories": ["<=4096sq"], "guidance_max_phases": 1, "vae_block_size": 32,
    })
    original = dict(model_def)
    if delivery == "1024x1024":
        with pytest.raises(ValueError, match="outside the accepted range"):
            resolutions.validate_agent_resolution(model_def, settings["resolution"], 1, "VAG")
    else:
        resolutions.validate_agent_resolution(model_def, settings["resolution"], 1, "VAG")
    with plugin._qwen21_production_resolution_budget(session(model_def), settings):
        resolutions.validate_agent_resolution(model_def, settings["resolution"], 1, "VAG")
        assert model_def["resolution_budget"] == 1536 * 1536
        with pytest.raises(ValueError, match="outside the accepted range"):
            resolutions.validate_agent_resolution(model_def, "2048x2048", 1, "VAG")
    assert settings == before
    assert settings["num_inference_steps"] == 50
    assert settings["guidance_scale"] == 8.0
    assert settings["_midom_final_output_resolution"] == delivery
    assert model_def == original


@pytest.mark.parametrize("factory", CONTRACTS)
@pytest.mark.parametrize("has_null_budget", [False, True])
def test_budget_restored_after_exception(factory, has_null_budget):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = validated_settings(plugin, factory)
    model_def = {"resolution_budget": None} if has_null_budget else {}
    original = dict(model_def)
    with pytest.raises(RuntimeError, match="inference failed"):
        with plugin._qwen21_production_resolution_budget(session(model_def), settings):
            assert model_def["resolution_budget"] == 1536 * 1536
            raise RuntimeError("inference failed")
    assert model_def == original


@pytest.mark.parametrize("explicit_budget", ["1920x1088", "4096x4096"])
def test_explicit_model_budget_respected(resolutions, explicit_budget):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = validated_settings(plugin, flattened_production_job)
    model_def = {"resolution_budget": explicit_budget}
    with plugin._qwen21_production_resolution_budget(session(model_def), settings):
        assert model_def == {"resolution_budget": explicit_budget}
        if explicit_budget == "1920x1088":
            with pytest.raises(ValueError, match="outside the accepted range"):
                resolutions.validate_agent_resolution(model_def, settings["resolution"])
        else:
            resolutions.validate_agent_resolution(model_def, settings["resolution"])
    assert model_def == {"resolution_budget": explicit_budget}


@pytest.mark.parametrize("mutator", [
    lambda settings: settings.update(resolution="2048x2048"),
    lambda settings: settings.update(_midom_requested_resolution="2048x2048"),
    lambda settings: settings.update(_midom_scene_integration_contract_version="unknown"),
    lambda settings: settings.update(_midom_scene_integration_profile_id="draft_v1"),
])
def test_unsupported_production_contracts_and_dimensions_cannot_expand_budget(mutator):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = validated_settings(plugin, production_job)
    mutator(settings)
    with pytest.raises(ValueError, match="declared contract and render adapter"):
        with plugin._qwen21_production_resolution_budget(None, settings):
            pytest.fail("Invalid production settings must not reach submission")


@pytest.mark.parametrize("image_task", ["generate", "edit", "masked_edit", "outpaint", "curated_variation", "atmosphere_overlay"])
def test_other_qwen_tasks_do_not_touch_runtime_budget(image_task):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    with plugin._qwen21_production_resolution_budget(None, {
        "model_type": module.QWEN21_MODEL_ID, "_midom_image_task": image_task,
    }):
        pass


def test_draft_and_other_models_do_not_touch_runtime_budget():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    draft = plugin._validate_image_job(draft_job(), 2801)
    with plugin._qwen21_production_resolution_budget(None, draft):
        pass
    with plugin._qwen21_production_resolution_budget(None, {"model_type": module.SENSENOVA_MODEL_ID}):
        pass


@pytest.mark.parametrize("factory", CONTRACTS)
@pytest.mark.parametrize("outcome", ["success", "submission_error", "wait_error", "cancelled", "generation_error", "upload_error"])
def test_claimed_job_scopes_budget_through_submit_wait_and_all_terminal_paths(resolutions, factory, outcome):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = validated_settings(plugin, factory)
    model_def = {"guidance_max_phases": 1}
    api = session(model_def)
    plugin._validate_job = lambda *args: dict(settings)
    plugin._download_job_inputs = lambda *args: []
    plugin._apply_inputs_to_settings = lambda *args: None
    plugin._callbacks_for_job = lambda *args: None
    plugin._get_generation_session = lambda: api
    plugin._capture_qwen_multi_angle_runtime_diagnostics = lambda *args, **kwargs: None
    plugin._capture_qwen21_native_rgba_runtime_diagnostics = lambda *args, **kwargs: None
    plugin._reset_idle_backoff_for_active_use = lambda *args, **kwargs: None
    generated = ["generated.png"]
    plugin._result_image_files = lambda result: generated
    plugin._build_generation_metadata = lambda *args: {}
    updates = []
    plugin._post_job_update = lambda connection, job_id, status, payload: updates.append((status, payload))
    handle = object()

    def submit_task(submitted, **kwargs):
        resolutions.validate_agent_resolution(model_def, submitted["resolution"], 1, submitted["video_prompt_type"])
        assert submitted["resolution"] == "1536x1536"
        assert submitted["num_inference_steps"] == 50
        assert submitted["guidance_scale"] == 8.0
        assert not any(key.startswith("_midom_") for key in submitted)
        if outcome == "submission_error":
            raise RuntimeError("submission failed")
        return handle

    def wait(connection, job_id, submitted_handle, callbacks):
        assert submitted_handle is handle
        resolutions.validate_agent_resolution(model_def, settings["resolution"], 1, "VAG")
        if outcome == "wait_error":
            raise RuntimeError("generation failed")
        return types.SimpleNamespace(
            success=outcome != "generation_error", cancelled=outcome == "cancelled",
            generated_files=generated, errors=["generation failed"] if outcome == "generation_error" else [],
        )

    finalized = []

    def finalize(file_path, native, index, temp_dir):
        assert "resolution_budget" not in model_def
        assert native["_midom_scene_integration_production"] is True
        finalized.append(index)
        return "protected-pixels-verified.png"

    def upload(connection, job_id, file_path, index, expected_resolution, normalization, *args):
        assert "resolution_budget" not in model_def
        assert file_path == "protected-pixels-verified.png"
        assert expected_resolution == "1024x1024"
        if outcome == "upload_error":
            raise RuntimeError("upload failed")
        return {"artifact_index": index, "artifact_id": 1}

    api.submit_task = submit_task
    plugin._wait_for_wangp_result = wait
    plugin._finalize_qwen21_scene_integration_production_output = finalize
    plugin._upload_artifact = upload
    connection = types.SimpleNamespace(connection_id="test", worker_id=1)
    job = factory()
    job["output"].update(width=1024, height=1024)
    plugin._run_job(connection, job)
    assert "resolution_budget" not in model_def
    assert plugin._active_job is None
    terminal = [update for update in updates if update[0] in {"complete", "fail"}]
    assert len(terminal) == 1
    assert terminal[0][0] == ("complete" if outcome == "success" else "fail")
    if outcome == "success":
        assert finalized == [0]
        assert len(terminal[0][1]["artifacts"]) == 1
