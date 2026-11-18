import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from PIL import Image
import pytest
import requests

from test_overlay_png_on_video import load_plugin_module, plugin_instance
import test_media_assembly_overlay_contract_fixtures as frozen


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/media_assembly_overlays_v1"
JOBS = json.loads((FIXTURES / "jobs.json").read_text())
GEOMETRY = json.loads((FIXTURES / "geometry.json").read_text())
REJECTIONS = json.loads((FIXTURES / "rejections.json").read_text())


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    monkeypatch.setattr(module.AwsWorkerBridgePlugin, "_config_path", property(lambda self: tmp_path / "worker_config.json"))
    plugin._heartbeat_once = lambda *args: None
    plugin._close_job_progress = lambda *args: None
    plugin._reset_idle_backoff_for_active_use = lambda *args, **kwargs: None
    plugin._stop_event = threading.Event()
    plugin._active_job = None
    plugin._cancel_requested_by_midom = False
    connection = module.ConnectionContext("test", "http://midom.test", 85, "secret", 10, 180, 20, "test", 1, "")
    return module, plugin, connection, module._overlay_module()


def png(size=(64, 36), color=(30, 60, 90, 255), fmt="PNG", orientation=None):
    image = Image.new("RGBA", size, color)
    if fmt == "JPEG":
        image = image.convert("RGB")
    data = io.BytesIO()
    kwargs = {}
    if orientation:
        exif = image.getexif()
        exif[274] = orientation
        kwargs["exif"] = exif.tobytes()
    image.save(data, format=fmt, **kwargs)
    return data.getvalue()


def claim(route="image", size=(64, 36), base=None, overlay=None):
    job = copy.deepcopy(JOBS[route]["claim"])
    sources = [base if base is not None else png(size), overlay if overlay is not None else png((8, 6), (240, 10, 20, 128))]
    job["output"].update(width=size[0], height=size[1])
    job["processing"]["visual"].update(scale_percent=100, offset_x=0, offset_y=0, opacity=1, anchor="bottom_right")
    for item, data in zip(job["inputs"], sources):
        item["sha256"] = hashlib.sha256(data).hexdigest()
        item["bytes"] = len(data)
    return job, sources


def candidate(job, sources, overlay_size=(8, 6), duration=0.5):
    route = "image" if job["media_type"] == "image" else "video"
    result = copy.deepcopy(JOBS[route]["candidate"])
    result["job_id"] = job["job_id"]
    result["summary"]["processing"] = copy.deepcopy(job["processing"])
    result["summary"].update(output_width=job["output"]["width"], output_height=job["output"]["height"])
    for index, (item, original, data) in enumerate(zip(result["summary"]["inputs"], job["inputs"], sources)):
        item["input_id"] = original["input_id"]
        item["encoded_bytes"] = len(data)
        item["mime_type"] = original["mime_type"]
        prefix = "display_" if route == "video" and index == 0 else "oriented_"
        size = (job["output"]["width"], job["output"]["height"]) if index == 0 else overlay_size
        item.update({prefix + "width": size[0], prefix + "height": size[1]})
        if route == "video" and index == 0:
            item["duration_seconds"] = duration
    return result


class Response:
    def __init__(self, data=b"", mime="image/png", payload=None, status=200):
        self.data, self.payload, self.status_code = data, payload, status
        self.headers = {"Content-Type": mime, "Content-Length": str(len(data))}
        self.closed = False
    def iter_content(self, chunk_size):
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i:i + chunk_size]
    def close(self):
        self.closed = True
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("test rejection", response=self)
    def json(self):
        return self.payload


def fake_server(monkeypatch, module, plugin, job, sources, upload_event=None, completion_event=None):
    requests_seen, uploads, updates, responses = [], [], [], []
    def get(url, **kwargs):
        item = next(i for i in job["inputs"] if url.endswith("/" + str(i["input_id"])))
        requests_seen.append(url)
        response = Response(sources[item["sequence"]], item["mime_type"])
        responses.append(response)
        assert kwargs["allow_redirects"] is False
        return response
    def post(url, **kwargs):
        assert url.endswith("/artifacts")
        data = kwargs["files"]["file"][1].read()
        info = kwargs["data"]
        assert hashlib.sha256(data).hexdigest() == info["sha256"]
        uploads.append((data, info))
        if upload_event:
            return upload_event(info)
        return Response(payload={"artifact_id": 501, "file_id": 601, "artifact_index": 0, "role": info["role"], "sha256": info["sha256"]})
    def update(connection, job_id, status, payload):
        updates.append((status, copy.deepcopy(payload)))
        if status == "complete" and completion_event:
            completion_event(payload)
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setattr(module.requests, "post", post)
    plugin._post_job_update = update
    return requests_seen, uploads, updates, responses


def real_video(tmp_path, *, audio=False, rotation=False, fps=24, size=(64, 36), vfr=False, duration=0.5):
    path = tmp_path / "base.mp4"
    command = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=blue:s={size[0]}x{size[1]}:r={fps}:d={duration}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}"]
    if vfr:
        command += ["-vf", "select=not(eq(n\\,4))", "-vsync", "vfr"]
    command += ["-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=15)
    if rotation:
        rotated = tmp_path / "rotated.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-c", "copy", "-metadata:s:v:0", "rotate=90", str(rotated)], check=True, capture_output=True, timeout=15)
        path = rotated
    return path.read_bytes()


@pytest.mark.parametrize("route", ("image", "video"))
def test_frozen_shapes_reach_real_runtime_validators(runtime, route):
    _, plugin, connection, helper = runtime
    job, selected = JOBS[route]["claim"], JOBS[route]["candidate"]
    assert plugin._candidate_incompatibility_reason(selected, connection) is None
    settings = plugin._validate_job(job, connection)
    assert settings["_midom_operation_type"] == job["operation_type"]
    helper.match_candidate_claim(selected, job)
    helper.validate_completion_metadata(JOBS[route]["completion"]["processing_metadata"], job, helper.validate_claim(job))


@pytest.mark.parametrize("case", [c for c in REJECTIONS["cases"] if c["stage"] == "claim"], ids=lambda c: c["id"])
def test_frozen_rejections_reach_real_claim_validation(runtime, case):
    _, plugin, connection, _ = runtime
    docs = frozen.documents(case["route"])
    target = docs[case["target"]]
    for key in case["path"][:-1]:
        target = target[key]
    if "append" in case:
        target[case["path"][-1]].append(case["append"])
    else:
        target[case["path"][-1]] = case["value"]
    with pytest.raises((ValueError, RuntimeError)):
        plugin._validate_job(docs["claim"], connection)


@pytest.mark.parametrize("case", GEOMETRY["placement_cases"], ids=lambda c: c["id"])
def test_real_geometry_matches_frozen(runtime, case):
    helper = runtime[3]
    visual = {"anchor": case["anchor"], "scale_percent": case["scale_percent"], "offset_x": case["offset"][0], "offset_y": case["offset"][1], "opacity": 1}
    actual = helper.placement(case["base"], case["overlay"], visual, helper.limits(helper.IMAGE_OPERATION))
    assert [actual["overlay_width"], actual["overlay_height"]] == case["expected_scaled"]
    assert [actual["left"], actual["top"]] == case["expected_position"]
    assert actual["visible_box"] == case["expected_visible_box"]


@pytest.mark.parametrize("case", GEOMETRY["opacity_cases"])
def test_actual_pixel_alpha_rounding(runtime, case):
    helper = runtime[3]
    layer = Image.new("RGBA", (6, 1), (255, 0, 0, 255))
    layer.putalpha(Image.frombytes("L", (6, 1), bytes(case["input_alpha"])))
    visual = {"anchor": "center", "scale_percent": 100, "offset_x": 0, "offset_y": 0, "opacity": case["opacity"]}
    result, info = helper.composite(Image.new("RGBA", (6, 1)), layer, visual, helper.limits(helper.IMAGE_OPERATION))
    assert list(result.getchannel("A").getdata()) == case["expected_alpha"]
    assert info["effective_opacity"] == case["effective_opacity"]


@pytest.mark.parametrize("route", ("image", "video"))
def test_advertised_contract_is_complete_and_selection_compatible(runtime, route):
    _, plugin, _, _ = runtime
    docs = frozen.documents(route)
    operation = docs["claim"]["operation_type"]
    cap = plugin._bounded_overlay_capability(operation) if route == "image" else plugin._storyboard_ffmpeg_processing_capability()
    assert cap is not None
    docs["capability"] = cap
    assert frozen.selection_rejection(route, docs) is None
    if route == "video":
        assert cap["contracts"]["overlay_png_on_video"] == frozen.CAPS["processors"][1]["contracts"]["overlay_png_on_video"]


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_image_claim_download_render_upload_complete(runtime, monkeypatch, fmt, mime):
    module, plugin, connection, helper = runtime
    job, sources = claim(base=png(fmt=fmt))
    job["inputs"][0].update(filename="misleading.png", mime_type=mime)
    selected = candidate(job, sources)
    assert plugin._candidate_incompatibility_reason(selected, connection) is None
    job["_midom_overlay_candidate"] = selected
    gets, uploads, updates, responses = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert len(gets) == 2 and len(uploads) == 1
    assert all(r.closed for r in responses)
    assert updates[-1][0] == "complete", updates
    completion = updates[-1][1]
    helper.validate_completion_metadata(completion["processing_metadata"], job, helper.validate_claim(job))
    assert completion["backend"] == "pillow"
    with Image.open(io.BytesIO(uploads[0][0])) as result:
        assert result.format == "PNG" and result.mode == "RGBA" and result.size == (64, 36)
        assert result.getpixel((63, 35))[0] > 120
    assert not list(plugin._config_path.parent.glob("overlay_outcomes*.sqlite3*"))


@pytest.mark.parametrize("audio,rotation,fps,vfr", [(False, False, 24, False), (True, False, 30, False), (True, True, 24, False), (False, False, 24, True)])
def test_video_real_ffmpeg_preserves_timeline_geometry_and_audio(runtime, monkeypatch, tmp_path, audio, rotation, fps, vfr):
    module, plugin, connection, helper = runtime
    source = real_video(tmp_path, audio=audio, rotation=rotation, fps=fps, vfr=vfr)
    size = (36, 64) if rotation else (64, 36)
    job, sources = claim("video", size=size, base=source)
    job["_midom_overlay_candidate"] = candidate(job, sources)
    gets, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert updates[-1][0] == "complete", updates
    assert len(gets) == 2 and len(uploads) == 1
    output = updates[-1][1]["processing_metadata"]["outputs"][0]
    assert [output["width"], output["height"]] == list(size)
    assert output["frame_count"] == (11 if vfr else fps // 2)
    assert output["rotation_degrees"] == 0 and output["has_audio"]
    assert output["video_codec"] == "h264" and output["audio_codec"] == "aac"
    assert updates[-1][1]["processing_metadata"]["audio_action"] == ("base_transcoded_to_aac" if audio else "silent_fill")
    target = tmp_path / "delivered.mp4"
    target.write_bytes(uploads[0][0])
    frame = subprocess.run(["ffmpeg", "-v", "error", "-i", str(target), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True, timeout=10).stdout
    result = Image.frombytes("RGB", size, frame)
    assert result.getpixel((size[0] - 2, size[1] - 2))[0] > 60
    assert result.getpixel((2, 2))[2] > 150
    # Container/audio padding is not video timeline drift. Verify actual frame PTS.
    for name, data in (("before", source), ("after", uploads[0][0])):
        file = tmp_path / f"{name}.mp4"
        file.write_bytes(data)
    def timestamps(file):
        raw = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(file)], check=True, capture_output=True, timeout=10)
        return [float(frame["best_effort_timestamp_time"]) for frame in json.loads(raw.stdout)["frames"]]
    assert timestamps(tmp_path / "after.mp4") == pytest.approx(timestamps(tmp_path / "before.mp4"), abs=0.0001)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_authoritative_geometry_through_delivery(runtime, monkeypatch, orientation):
    module, plugin, connection, _ = runtime
    size = (36, 64) if orientation >= 5 else (64, 36)
    job, sources = claim(size=size, base=png(orientation=orientation))
    _, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert updates[-1][0] == "complete", updates
    assert Image.open(io.BytesIO(uploads[0][0])).size == size


@pytest.mark.parametrize("case", GEOMETRY["encoding_cases"], ids=lambda c: c["id"])
def test_frozen_actual_bytes_reach_decoder(runtime, tmp_path, case):
    helper = runtime[3]
    path = tmp_path / case["filename"]
    path.write_bytes(base64.b64decode(GEOMETRY["media_samples"][case["sample"]]))
    item = {"path": str(path), "mime_type": case["declared_mime"], "kind": "overlay_png" if case["role"] == "overlay_image" else "source_image"}
    if case["expected"] == "accepted":
        assert helper.read_image(item, helper.limits(helper.IMAGE_OPERATION)).size == (2, 3)
    else:
        with pytest.raises((ValueError, OSError)):
            helper.read_image(item, helper.limits(helper.IMAGE_OPERATION))


@pytest.mark.parametrize("boundary", ("upload", "completion", "conflict", "bad_ack", "unauthorized"))
def test_uncertain_delivery_does_not_retry_or_persist_outcome(runtime, monkeypatch, boundary):
    module, plugin, connection, helper = runtime
    job, sources = claim()
    def lost(info):
        if boundary == "conflict":
            return Response(status=409)
        if boundary == "unauthorized":
            return Response(status=403)
        if boundary == "bad_ack":
            return Response(payload={"artifact_id": 501, "file_id": 601, "role": "video"})
        raise requests.Timeout("acknowledgement lost")
    def lost_completion(payload):
        raise requests.Timeout("completion acknowledgement lost")
    plugin._handle_authorization_failure = lambda *args: setattr(plugin, "_cancel_requested_by_midom", True)
    gets, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources,
        upload_event=lost if boundary != "completion" else None,
        completion_event=lost_completion if boundary == "completion" else None)
    plugin._run_job(connection, job)
    assert len(uploads) == 1
    assert len(gets) == 2
    assert not any(status == "fail" for status, _ in updates)
    assert sum(status == "complete" for status, _ in updates) == int(boundary == "completion")
    assert not list(plugin._config_path.parent.glob("overlay_outcomes*.sqlite3*"))
    # There is no worker-owned restart fence. Midom alone decides whether to
    # authorize another claim; validation does not consult local outcome history.
    fresh = plugin_instance(module)
    assert fresh._validate_job(job, connection)["_midom_operation_type"] == job["operation_type"]
    assert plugin._active_job is None and plugin._active_job_id is None


@pytest.mark.parametrize("problem", ("hash", "mime", "corrupt", "oversize", "cancel", "orientation", "output_limit"))
def test_definite_preupload_errors_fail_without_partial_delivery(runtime, monkeypatch, problem):
    module, plugin, connection, _ = runtime
    job, sources = claim()
    if problem == "hash":
        job["inputs"][0]["sha256"] = "0" * 64
    elif problem == "mime":
        sources[1] = png((8, 6), fmt="JPEG")
        job["inputs"][1]["sha256"] = hashlib.sha256(sources[1]).hexdigest()
        job["inputs"][1]["bytes"] = len(sources[1])
    elif problem == "corrupt":
        sources[1] = b"not an image"
        job["inputs"][1].update(sha256=hashlib.sha256(sources[1]).hexdigest(), bytes=len(sources[1]))
    elif problem == "oversize":
        job["limits"] = {"max_image_input_bytes": 1}
    elif problem == "cancel":
        plugin._stop_event.set()
    elif problem == "orientation":
        job["output"]["width"] += 1
    elif problem == "output_limit":
        job["limits"] = {"max_artifact_bytes": 1}
    _, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert not uploads
    assert updates[-1][0] == "fail", updates
    if problem == "cancel":
        assert updates[-1][1]["reason"] == "cancel_requested"
    assert not list(plugin._config_path.parent.glob("overlay_outcomes*.sqlite3*"))


def test_no_scaled_allocation_when_budget_exceeded(runtime, monkeypatch):
    helper = runtime[3]
    def forbidden(*args, **kwargs):
        pytest.fail("Allocated a scaled image before checking budget")
    monkeypatch.setattr(Image.Image, "resize", forbidden)
    visual = {"anchor": "center", "scale_percent": 500, "opacity": 1, "offset_x": 0, "offset_y": 0}
    with pytest.raises(ValueError, match="Scaled overlay"):
        helper.composite(Image.new("RGBA", (2, 2)), Image.new("RGBA", (2, 2)), visual, {"max_scaled_overlay_pixels": 99})


def test_candidate_claim_mutation_is_rejected_before_download(runtime, monkeypatch):
    module, plugin, connection, _ = runtime
    job, sources = claim()
    job["_midom_overlay_candidate"] = candidate(job, sources)
    job["processing"]["visual"]["offset_x"] = 10
    gets, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert not gets and not uploads and updates[-1][0] == "fail"


def test_image_advertisement_does_not_depend_on_ffmpeg(runtime, monkeypatch):
    _, plugin, _, helper = runtime
    monkeypatch.setattr(plugin, "_ffmpeg_binary", lambda: None)
    monkeypatch.setattr(plugin, "_ffprobe_binary", lambda: None)
    assert plugin._bounded_overlay_capability(helper.IMAGE_OPERATION) is not None
    assert plugin._bounded_overlay_capability(helper.VIDEO_OPERATION) is None


@pytest.mark.parametrize("reason", ("timeout", "cancel", "scratch"))
def test_supervisor_stops_and_verifies_child_exit(runtime, tmp_path, monkeypatch, reason):
    module, _, _, helper = runtime
    handle = module.LocalProcessJob()
    if reason == "cancel":
        handle.cancel()
    if reason == "scratch":
        monkeypatch.setattr(helper.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 0})())
    with pytest.raises((ValueError, RuntimeError)):
        helper.supervise([sys.executable, "-c", "import time; time.sleep(15)"], tmp_path, 0.2, handle=handle)
    assert handle._process is None


def test_linux_resource_limits_are_installed_in_real_child(runtime):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux resource limits; Windows uses a Job Object")
    helper = runtime[3]
    source = f"import importlib.util,resource; s=importlib.util.spec_from_file_location('h',{str(helper.__file__)!r}); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.apply_resource_limits(); print(resource.getrlimit(resource.RLIMIT_AS)[0])"
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, check=True, timeout=10)
    assert int(result.stdout) == helper.MEMORY_BYTES


def test_discovery_and_validation_do_not_need_local_state_storage(runtime, monkeypatch):
    module, plugin, connection, helper = runtime
    def forbidden_config_path(self):
        pytest.fail("Overlay discovery or validation accessed worker state storage")
    monkeypatch.setattr(module.AwsWorkerBridgePlugin, "_config_path", property(forbidden_config_path))
    for route in ("image", "video"):
        job, selected = JOBS[route]["claim"], JOBS[route]["candidate"]
        assert plugin._bounded_overlay_capability(job["operation_type"]) is not None
        assert plugin._candidate_incompatibility_reason(selected, connection) is None
        assert plugin._validate_job(job, connection)["_midom_operation_type"] == job["operation_type"]
    assert not hasattr(plugin, "_overlay_journal")
    assert not hasattr(helper, "OutcomeJournal")


def test_nonfinite_and_huge_integers_rejected(runtime):
    helper = runtime[3]
    for value in (float("nan"), float("inf"), -float("inf"), 10**1000, True):
        job, _ = claim()
        job["processing"]["visual"]["scale_percent"] = value
        with pytest.raises(ValueError):
            helper.validate_claim(job)


@pytest.mark.parametrize("kind", ("opaque", "transparent", "off_canvas", "oversized", "odd_image"))
def test_valid_non_generative_alpha_and_clipping(runtime, monkeypatch, kind):
    module, plugin, connection, _ = runtime
    size = (63, 35) if kind == "odd_image" else (64, 36)
    job, sources = claim(size=size, overlay=png((8, 6), (240, 10, 20, 0 if kind == "transparent" else 255)))
    if kind == "off_canvas":
        job["processing"]["visual"]["offset_x"] = 16384
    elif kind == "oversized":
        job["processing"]["visual"]["scale_percent"] = 500
    _, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert updates[-1][0] == "complete", updates
    with Image.open(io.BytesIO(uploads[0][0])) as output:
        assert output.size == size
        if kind in ("transparent", "off_canvas"):
            assert output.tobytes() == Image.open(io.BytesIO(sources[0])).convert("RGBA").tobytes()


def test_reject_animated_png(runtime, tmp_path):
    helper = runtime[3]
    path = tmp_path / "animated.png"
    Image.new("RGBA", (8, 8), "red").save(path, save_all=True, append_images=[Image.new("RGBA", (8, 8), "blue")], duration=100)
    with pytest.raises(ValueError, match="Animated"):
        helper.read_image({"path": str(path), "kind": "overlay_png", "mime_type": "image/png"}, helper.limits(helper.IMAGE_OPERATION))


def test_maximum_canvas_image_render_under_installed_memory_guard(runtime, monkeypatch):
    module, plugin, connection, _ = runtime
    job, sources = claim(size=(4096, 4096), base=png((4096, 4096)))
    _, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert updates[-1][0] == "complete", updates
    assert Image.open(io.BytesIO(uploads[0][0])).size == (4096, 4096)


def test_video_720p_real_render_under_installed_memory_guard(runtime, monkeypatch, tmp_path):
    module, plugin, connection, _ = runtime
    job, sources = claim("video", size=(1280, 720), base=real_video(tmp_path, size=(1280, 720), audio=True))
    _, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert len(uploads) == 1 and updates[-1][0] == "complete", updates


def test_video_source_ceiling_is_timeline_not_aac_padding(runtime, tmp_path):
    helper = runtime[3]
    path = tmp_path / "boundary.mp4"
    path.write_bytes(real_video(tmp_path, audio=True, duration=120))
    source = helper.probe_video(path, "ffprobe", tmp_path)
    assert source["duration_seconds"] == 120
    assert source["frame_count"] == 2880


def test_existing_journals_are_not_read_changed_or_replaced(runtime, monkeypatch):
    module, plugin, connection, _ = runtime
    evidence = plugin._config_path.with_name("overlay_outcomes.sqlite3")
    original = b"Historical evidence, deliberately not a valid SQLite database"
    evidence.write_bytes(original)
    job, sources = claim()
    _, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources)
    plugin._run_job(connection, job)
    assert len(uploads) == 1 and updates[-1][0] == "complete", updates
    assert evidence.read_bytes() == original
    assert list(evidence.parent.glob("overlay_outcomes*.sqlite3*")) == [evidence]


@pytest.mark.parametrize("status", (200, 302, 400, 401, 403, 409, 500))
def test_real_complete_http_boundary_does_not_retry_or_follow_redirects(runtime, monkeypatch, status):
    module, plugin, connection, helper = runtime
    calls, rejected = [], []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response(status=status)
    monkeypatch.setattr(module.requests, "post", post)
    plugin._handle_authorization_failure = lambda *args: rejected.append(args)
    plugin._midom_error_message = lambda response: "test"
    payload = JOBS["image"]["completion"]
    if status == 200:
        plugin._send_job_update(connection, 22001, "complete", payload)
    else:
        with pytest.raises((requests.HTTPError, helper.OutcomeUncertain)):
            plugin._send_job_update(connection, 22001, "complete", payload)
    assert len(calls) == 1 and calls[0][1]["allow_redirects"] is False
    assert bool(rejected) == (status in (401, 403))


@pytest.mark.parametrize("route", ("image", "video"))
def test_lower_advertised_worker_limits_are_used_for_both_candidate_and_claim(runtime, monkeypatch, route):
    _, plugin, connection, helper = runtime
    job, selected = copy.deepcopy(JOBS[route]["claim"]), copy.deepcopy(JOBS[route]["candidate"])
    original = plugin._bounded_overlay_capability
    def restricted(operation):
        record = original(operation)
        record["limits"][operation]["scale_percent"]["max"] = 20
        return record
    monkeypatch.setattr(plugin, "_bounded_overlay_capability", restricted)
    assert "scale_percent" in plugin._candidate_incompatibility_reason(selected, connection)
    with pytest.raises(ValueError, match="scale_percent"):
        plugin._validate_job(job, connection)


@pytest.mark.parametrize("geometry", ((1279, 720), (720, 1279)))
def test_odd_video_claim_and_candidate_fail_without_resizing(runtime, geometry):
    _, plugin, connection, _ = runtime
    job, sources = claim("video", size=geometry, base=b"not downloaded")
    assert "even" in plugin._candidate_incompatibility_reason(candidate(job, sources), connection)
    with pytest.raises(ValueError, match="even"):
        plugin._validate_job(job, connection)


@pytest.mark.parametrize("key,value", (("max_duration_seconds", 2), ("color_mode", "rgb"), ("format", "jpeg")))
def test_alternate_image_output_declarations_fail(runtime, key, value):
    _, plugin, connection, _ = runtime
    job, _ = claim()
    job["output"][key] = value
    with pytest.raises(ValueError, match="typed output"):
        plugin._validate_job(job, connection)


def test_matching_typed_artifact_replay_acknowledgement_can_complete(runtime, monkeypatch):
    module, plugin, connection, _ = runtime
    job, sources = claim()
    def accepted_existing(identity):
        return Response(payload={"artifact_id": 501, "file_id": 601, "artifact_index": 0,
            "role": identity["role"], "sha256": identity["sha256"], "mime_type": identity["mime_type"]})
    _, uploads, updates, _ = fake_server(monkeypatch, module, plugin, job, sources, upload_event=accepted_existing)
    plugin._run_job(connection, job)
    assert len(uploads) == 1 and updates[-1][0] == "complete"
    assert updates[-1][1]["artifacts"][0]["artifact_id"] == 501


def test_missing_helper_does_not_disable_existing_full_frame_route(runtime, monkeypatch):
    from test_overlay_png_on_video import overlay_job
    module, plugin, connection, _ = runtime
    def missing():
        raise FileNotFoundError("helper missing")
    monkeypatch.setattr(module, "_overlay_module", missing)
    capability = plugin._storyboard_ffmpeg_processing_capability()
    assert "overlay_png_on_video_positioned" not in capability["supported_operations"]
    assert capability["contracts"]["overlay_png_on_video"] == frozen.CAPS["processors"][1]["contracts"]["overlay_png_on_video"]
    job = overlay_job(module)
    job["target"] = JOBS["video"]["claim"]["target"]
    assert plugin._validate_job(job, connection)["_midom_operation_type"] == "overlay_png_on_video"
