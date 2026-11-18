"""The first H3 route is soundtrack-conditioned FL2VA, not reference/control video."""

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from test_fish_audio_drama3 import bridge


def native_root(module):
    root = Path(os.environ.get("WAN2GP_ROOT") or Path(module.__file__).parents[2])
    if not (root / "models/minimax_h3/minimax_h3_handler.py").is_file():
        pytest.skip("Native H3 source checks require WAN2GP_ROOT")
    return root


def job_for(module, size=(1280, 720), ending=False):
    return {
        "job_id": 501, "media_type": "video", "model_id": module.H3_VIDEO_MODEL_ID,
        "prompt": "A seated presenter speaks to the camera in a quiet studio.",
        "generation": {
            "video_contract_version": module.H3_VIDEO_CONTRACT,
            "video_task": "audio_conditioned_video", "audio_video_mode": "driving_audio_guided",
            "speed_profile_id": module.H3_SPEED_PROFILE, "duration_mode": module.H3_DURATION_MODE,
            "duration_seconds": 15, "prompt_mode": "plain", "video_sync_profile_id": "standard",
            "prompt_enhancement_by_worker": False, "seed": 123,
            "options": {"multi_prompts_gen_type": "FG"},
        },
        "inputs": [{"kind": kind, "role": kind, "input_id": index + 1}
                   for index, kind in enumerate(["start_image", "driving_audio"] + (["end_image"] if ending else []))],
        "output": {"count": 1, "format": "mp4", "width": size[0], "height": size[1]},
    }


def candidate_for(plugin, job):
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 2, "project_id": 3, "paired_user_id": 4}[key]
    summary = {**job["generation"], "output_count": 1, "output_format": "mp4",
               "output_width": job["output"]["width"], "output_height": job["output"]["height"],
               "start_image_count": 1, "end_image_count": len(job["inputs"]) - 2,
               "input_audio_count": 1, "control_video_count": 0}
    return json.loads(json.dumps({"job_id": 501, "worker_id": 7, "org_id": 2, "project_id": 3,
                                 "requested_by_user_id": 4, "media_type": "video",
                                 "model_id": job["model_id"], "summary": summary}))


def ffmpeg(*args):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/FFprobe are required")
    subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error", "-nostdin", *map(str, args)],
                   check=True, capture_output=True, timeout=60)


def inputs_for(plugin, job, tmp_path, duration=5.0):
    plugin._ffmpeg_binary = lambda: shutil.which("ffmpeg")
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    result = []
    for descriptor in job["inputs"]:
        kind = descriptor["kind"]
        path = tmp_path / (kind + (".wav" if kind == "driving_audio" else ".png"))
        if kind == "driving_audio":
            ffmpeg("-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration}", path)
        else:
            Image.new("RGB", (job["output"]["width"], job["output"]["height"]), (31, 71, 121)).save(path)
        result.append({**descriptor, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def test_discovery_is_narrow_and_does_not_change_ltx(bridge):
    module, plugin = bridge
    cap = plugin._h3_video_capability()
    assert cap["model_id"] == module.H3_VIDEO_MODEL_ID
    assert cap["limits"]["video_contract_version"] == module.H3_VIDEO_CONTRACT
    assert cap["limits"]["required_input_kinds"] == ["start_image", "driving_audio"]
    assert not cap["capabilities"]["control_video"]
    assert not cap["capabilities"]["prompt_generated_audio"]
    plugin._find_lora_relative_path = lambda *args: None
    assert plugin._ltx_video_capability(module.LTX23_VIDEO_MODEL_ID, "LTX")["capabilities"]["prompt_generated_audio"]


@pytest.mark.parametrize("size", [(1280, 720), (720, 1280)])
@pytest.mark.parametrize("ending", [False, True])
def test_serialized_candidate_and_claim_agree(bridge, size, ending):
    module, plugin = bridge
    job = json.loads(json.dumps(job_for(module, size, ending)))
    assert plugin._candidate_incompatibility_reason(candidate_for(plugin, job), None) is None
    settings = plugin._validate_job(job, None)
    assert settings["model_type"] == module.H3_VIDEO_MODEL_ID
    assert settings["audio_prompt_type"] == "AS"
    assert settings["resolution"] == module.H3_RENDER_RESOLUTIONS[f"{size[0]}x{size[1]}"]
    assert settings["seed"] == 123
    assert settings["activated_loras"] == []
    assert settings["image_prompt_type"] == ("SE" if ending else "S")


@pytest.mark.parametrize("field,value", [
    ("video_contract_version", "h3_reference_video_v1"), ("speed_profile_id", "standard"),
    ("audio_video_mode", "prompt_generated_audio"), ("video_task", "control_video_guided_video"),
    ("duration_mode", "fixed_seconds"), ("prompt_mode", "enhanced"),
    ("video_sync_profile_id", "omninft_rl_lora_sync"), ("prompt_enhancement_by_worker", True),
    ("duration_seconds", 4), ("duration_seconds", 16), ("duration_seconds", True),
    ("duration_seconds", 5.5),
])
def test_candidate_and_claim_reject_unsupported_selection(bridge, field, value):
    module, plugin = bridge
    job = job_for(module)
    job["generation"][field] = value
    assert plugin._candidate_incompatibility_reason(candidate_for(plugin, job), None)
    with pytest.raises(ValueError):
        plugin._validate_job(job, None)


@pytest.mark.parametrize("kind", ["control_video", "reference_image", "mask_image", "control_image"])
def test_no_extra_inputs_or_generated_audio_fallback(bridge, kind):
    module, plugin = bridge
    job = job_for(module)
    job["inputs"].append({"kind": kind, "input_id": 99})
    candidate = candidate_for(plugin, job)
    candidate["summary"][kind + "_count"] = 1
    assert plugin._candidate_incompatibility_reason(candidate, None)
    with pytest.raises(ValueError):
        plugin._validate_job(job, None)
    job = job_for(module)
    job["inputs"].pop()
    with pytest.raises(ValueError, match="one driving_audio"):
        plugin._validate_job(job, None)


@pytest.mark.parametrize("change", [
    {"count": 2}, {"count": True}, {"format": "webm"}, {"width": 1920},
    {"mime_type": "image/png"}, {"resolution": "720x1280"}, {"color_mode": "rgba"},
    {"artifacts": [{"artifact_index": 0, "role": "image", "mime_type": "video/mp4"}]},
])
def test_reject_output_changes(bridge, change):
    module, plugin = bridge
    job = job_for(module)
    job["output"].update(change)
    with pytest.raises(ValueError):
        plugin._validate_job(job, None)


@pytest.mark.parametrize("duration", [5, 5.25, 15])
def test_native_scheduler_keeps_end_anchor_on_delivered_timeline(bridge, tmp_path, duration):
    module, plugin = bridge
    job = job_for(module, ending=True)
    settings = plugin._validate_job(job, None)
    plugin._apply_inputs_to_settings(settings, inputs_for(plugin, job, tmp_path, duration), job)
    assert settings["video_length"] == settings["_midom_h3"]["render_frames"]
    assert settings["frames_positions"] == str(math.ceil(duration * 24))
    assert settings["_midom_h3"]["end_anchor_frame_index"] == math.ceil(duration * 24) - 1
    scheduler_path = native_root(module) / "shared/utils/frame_scheduler.py"
    spec = importlib.util.spec_from_file_location("h3_test_scheduler", scheduler_path)
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    plan = scheduler.build_default_window_plan(total_frames=settings["video_length"], window_size=362,
                                              default_overlap=18, discard_last_frames=0, minimum=107,
                                              step=17, frame_offset=5, preserve_exact_output_frames=True)
    assert len(plan) == 1
    assert plan[0]["frame_num"] == settings["_midom_h3"]["render_frames"]
    assert plan[0]["output_frames"] == settings["video_length"]
    assert settings["_midom_h3"]["source_audio_duration_seconds"] == pytest.approx(duration)
    plugin._assert_h3_submission_integrity(settings)
    with Image.open(settings["image_start"][0]) as image:
        assert image.size == (1280, 736)
        assert image.crop((0, 8, 1280, 728)).getpixel((0, 0)) == (31, 71, 121)


def test_submission_is_native_hyperflow_without_private_metadata(bridge, tmp_path):
    module, plugin = bridge
    job = job_for(module)
    settings = plugin._validate_job(job, None)
    plugin._apply_inputs_to_settings(settings, inputs_for(plugin, job, tmp_path), job)
    captured = {}
    session = types.SimpleNamespace(_ensure_runtime=lambda: types.SimpleNamespace(module=types.SimpleNamespace()),
                                    submit_task=lambda values, callbacks: captured.update(values))
    plugin._submit_wangp_job(session, settings, 1, None)
    assert captured["audio_prompt_type"] == "AS"
    assert captured["_api"] == {"return_media": True}
    assert not any(key.startswith("_midom_") for key in captured)
    source = native_root(module) / "wgp.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_media")
    native = {arg.arg for arg in function.args.args}
    # These are settings consumed outside generate_media by WanGP's API/model loader.
    assert not set(captured) - native - {"model_type", "prompt_enhancer", "custom_settings", "_api"}
    settings["video_guide"] = "unexpected.mp4"
    with pytest.raises(ValueError, match="video_guide"):
        plugin._submit_wangp_job(session, settings, 1, None)


@pytest.mark.parametrize("duration", [4.9, 15.1])
def test_actual_audio_bounds_are_not_silently_trimmed(bridge, tmp_path, duration):
    module, plugin = bridge
    job = job_for(module)
    settings = plugin._validate_job(job, None)
    with pytest.raises(ValueError, match="driving audio"):
        plugin._apply_inputs_to_settings(settings, inputs_for(plugin, job, tmp_path, duration), job)


@pytest.mark.parametrize("size,ending", [((1280, 720), False), ((720, 1280), True)])
def test_real_media_through_claim_submission_upload_completion(bridge, monkeypatch, tmp_path, size, ending):
    module, plugin = bridge
    job = job_for(module, size, ending)
    downloaded = inputs_for(plugin, job, tmp_path)
    native = tmp_path / "generated_seed123.mp4"
    width, height = map(int, module.H3_RENDER_RESOLUTIONS[f"{size[0]}x{size[1]}"].split("x"))
    ffmpeg("-f", "lavfi", "-i", f"color=c=green:s={width}x{height}:r=24:d={124 / 24}",
           "-f", "lavfi", "-i", f"sine=frequency=880:duration={124 / 24}", "-c:v", "libx264",
           "-preset", "ultrafast", "-c:a", "aac", "-shortest", native)
    result = types.SimpleNamespace(success=True, cancelled=False, generated_files=[str(native)], artifacts=[], errors=[])
    submitted = []
    def submit(settings, callbacks):
        submitted.append(settings)
        result.artifacts = [types.SimpleNamespace(path=str(native), media_type="video", client_id=settings["client_id"])]
        return types.SimpleNamespace(result=lambda: result)

    session = types.SimpleNamespace(
        _ensure_runtime=lambda: types.SimpleNamespace(module=types.SimpleNamespace(WanGP_version="20.00")),
        submit_task=submit)
    plugin._get_generation_session = lambda: session
    plugin._download_job_inputs = lambda *args: downloaded
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()
    uploaded = []

    def upload(url, **kwargs):
        data = kwargs["files"]["file"][1].read()
        assert kwargs["data"]["sha256"] == hashlib.sha256(data).hexdigest()
        uploaded.append(data)
        return types.SimpleNamespace(status_code=200, json=lambda: {"artifact_id": 12, "file_id": 34, "artifact_index": 0})

    monkeypatch.setattr(module.requests, "post", upload)
    plugin._run_job(types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7"), job)
    assert plugin._fish_updates[-1][0] == "complete", plugin._fish_updates
    assert len(uploaded) == len(submitted) == 1
    metadata = plugin._fish_updates[-1][1]["generation_metadata"]["h3_video"]
    assert metadata["artifact_sha256"] == hashlib.sha256(uploaded[0]).hexdigest()
    assert metadata["final_dimensions"] == list(size)
    assert metadata["native_audio_discarded"] is True
    assert metadata["lip_sync_verification"] == "not_verified"
    final = tmp_path / "uploaded.mp4"
    final.write_bytes(uploaded[0])
    pcm = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(final), "-map", "0:a:0",
                          "-t", "1", "-ar", "8000", "-ac", "1", "-f", "f32le", "pipe:1"],
                         check=True, capture_output=True).stdout
    import array
    samples = array.array("f", pcm)
    crossings = sum(samples[i - 1] <= 0 < samples[i] for i in range(1, len(samples)))
    assert 430 <= crossings <= 450  # original 440 Hz, not the native output's 880 Hz


def test_download_dispatch_handles_h3_anchors_and_audio(bridge, monkeypatch, tmp_path):
    module, plugin = bridge
    job = job_for(module, ending=True)
    responses = []
    for item in job["inputs"]:
        data = b"bounded-download-data"
        mime = "audio/wav" if item["kind"] == "driving_audio" else "image/png"
        responses.append(types.SimpleNamespace(headers={"Content-Type": mime}, raise_for_status=lambda: None,
                                               iter_content=lambda chunk_size: iter([data]), close=lambda: None))
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: responses.pop(0))
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7)
    paths = plugin._download_job_inputs(connection, job, str(tmp_path))
    assert [item["kind"] for item in paths] == ["start_image", "driving_audio", "end_image"]


@pytest.mark.parametrize("case", ["alias", "extra", "wrong_client", "missing_artifact"])
def test_video_collection_never_selects_first_of_multiple_outputs(bridge, tmp_path, case):
    module, plugin = bridge
    settings = plugin._validate_job(job_for(module), None)
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"associated-output")
    other = tmp_path / "different.mp4"
    other.write_bytes(output.read_bytes())
    artifact = types.SimpleNamespace(path="clip.mp4", media_type="video", client_id=settings["client_id"])
    result = types.SimpleNamespace(generated_files=[str(output)], artifacts=[artifact])
    if case == "extra":
        result.generated_files.append(str(other))
    elif case == "wrong_client":
        artifact.client_id = "another-attempt"
    elif case == "missing_artifact":
        result.artifacts = []
    session = types.SimpleNamespace(_root=tmp_path)
    if case == "alias":
        assert plugin._result_h3_video_files(result, settings, session) == [str(output)]
    else:
        with pytest.raises(ValueError):
            plugin._result_h3_video_files(result, settings, session)


def test_cancelled_preparation_cannot_deliver_partial_output(bridge, tmp_path):
    _, plugin = bridge
    plugin._cancel_requested_by_midom = True
    with pytest.raises(ValueError, match="cancel_requested"):
        plugin._h3_process(["sleep", "30"])


@pytest.mark.parametrize("ending", [False, True])
def test_actual_native_handler_accepts_prepared_recipe(bridge, monkeypatch, tmp_path, ending):
    module, plugin = bridge
    job = job_for(module, ending=ending)
    settings = plugin._validate_job(job, None)
    plugin._apply_inputs_to_settings(settings, inputs_for(plugin, job, tmp_path, 5.25), job)
    root = native_root(module)
    handler_path = root / "models/minimax_h3/minimax_h3_handler.py"
    tree = ast.parse(handler_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "family_handler")
    validator = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                                   and node.name == "validate_generative_settings"))
    validator.decorator_list = []
    namespace = {}
    exec((root / "models/minimax_h3/constants.py").read_text(), namespace)
    # Read literal native constants without importing transformer/CUDA dependencies.
    for filename in ["minimax_h3_handler.py", "hyperflow.py", "pika.py", "dmad.py", "vae_upsampler.py"]:
        for node in ast.parse((handler_path.parent / filename).read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    namespace[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    namespace["HYPERFLOW_STEPS"] = tuple(sorted(namespace["SIGMA_GRIDS"]))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CONTROL_ARCHITECTURES"
                                              for target in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "native_control_constants", "exec"), namespace)
    scheduler = types.ModuleType("native_h3_scheduler")
    exec((root / "shared/utils/frame_scheduler.py").read_text(), scheduler.__dict__)
    namespace.update({"normalize_overlap": scheduler.normalize_overlap, "get_override": lambda *args: None,
                      "H3_IMAGE_ARCHITECTURES": set(), "P_H3_IMAGE_ARCHITECTURE": "p_h3_image",
                      "VIGGLE_REMAKE_ARCHITECTURE": "viggle_remake", "VIGGLE_ARCHITECTURE": "viggle"})
    utils = types.ModuleType("shared.utils.utils")
    utils.get_outpainting_dims = lambda *args: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    exec(compile(ast.Module(body=[validator], type_ignores=[]), str(handler_path), "exec"), namespace)
    native_settings = {key: value for key, value in settings.items() if not key.startswith("_midom_")}
    native_settings["overrides"] = ""
    model_def = json.loads((root / "defaults/minimax_h3_fl2va_hyperflow.json").read_text())["model"]
    assert namespace["validate_generative_settings"](model_def["architecture"], model_def, native_settings) is None
    if ending:
        images_tree = ast.parse((root / "shared/prompt_enhancer/images.py").read_text())
        resolver = next(node for node in images_tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "resolve_injected_positions")
        namespace = {}
        exec(compile(ast.Module(body=[resolver], type_ignores=[]), "native_images", "exec"), namespace)
        positions = namespace[resolver.name](settings["frames_positions"], 1,
                                            windows=[{"frame_num": settings["video_length"], "discard_last_frames": 0,
                                                      "trim_last_frames": 0, "overlap_frames": 1}],
                                            source_frames=1, source_overlap=1, window_size=362,
                                            discard_frames=0, reuse_frames=18, alignment_shift=0)
        assert positions == [settings["_midom_h3"]["end_anchor_frame_index"]]
