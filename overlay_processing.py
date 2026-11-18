"""Bounded deterministic overlay contracts and isolated processing worker."""

import copy
import hashlib
import io
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from functools import lru_cache
import tempfile

from PIL import Image, ImageOps, features


IMAGE_OPERATION = "overlay_png_on_image"
VIDEO_OPERATION = "overlay_png_on_video_positioned"
POLICY = "midom_overlay_geometry_v1"
ROUTES = {
    IMAGE_OPERATION: ("image", "image_overlay_processor", "deterministic_image_processing", "png_image_overlay_v1", "image", "png", "image/png", "rgba"),
    VIDEO_OPERATION: ("video", "storyboard_ffmpeg_processor", "storyboard_ffmpeg_processing", "positioned_png_video_overlay_v1", "video", "mp4", "video/mp4", "rgb"),
}
ANCHORS = ["center", "top_left", "top_right", "bottom_left", "bottom_right"]
MIMES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
IMAGE_BYTES = 16 * 1024 * 1024
PIXELS = 16_777_216
VIDEO_BYTES = 80 * 1024 * 1024
PNG_BYTES = 50 * 1024 * 1024
MEMORY_BYTES = 1536 * 1024 * 1024
SCRATCH_BYTES = 512 * 1024 * 1024
LOG_BYTES = 128 * 1024
RESULT_BYTES = 64 * 1024
TIMEOUTS = {IMAGE_OPERATION: 120, VIDEO_OPERATION: 1800}
COMMON_LIMITS = {
    "max_outputs": 1, "max_image_input_bytes": IMAGE_BYTES,
    "max_oriented_image_pixels": PIXELS, "max_scaled_overlay_pixels": PIXELS,
    "max_output_pixels": PIXELS,
    "scale_percent": {"min": 1, "max": 500}, "opacity": {"min": 0, "max": 1},
    "offset_x": {"min": -16384, "max": 16384}, "offset_y": {"min": -16384, "max": 16384},
}


def limits(operation):
    result = copy.deepcopy(COMMON_LIMITS)
    result["max_artifact_bytes"] = PNG_BYTES if operation == IMAGE_OPERATION else VIDEO_BYTES
    if operation == VIDEO_OPERATION:
        result.update(max_video_input_bytes=VIDEO_BYTES, max_video_duration_seconds=120)
    return result


def capability(operation, image_mimes):
    media, processor, task, version, role, fmt, mime, color = ROUTES[operation]
    recipe = {
        "visual.anchor": ANCHORS.copy(), "visual.scale_reference": ["oriented_overlay_source"],
        "visual.alpha_mode": ["png_alpha"], "orientation_mode": ["exif_transpose" if media == "image" else "physical_display_orientation"],
        "output_dimensions": ["oriented_base_exact" if media == "image" else "base_display_exact"],
    }
    if media == "image":
        recipe["output_color_modes"] = ["rgba"]
    else:
        recipe.update({"visual.end_behavior": ["repeat"], "duration_mode": ["source_full"], "audio.mode": ["base_or_silence"],
            "output.container": ["mp4"], "output.video_codec": ["h264"], "output.audio_codec": ["aac"], "output.pixel_format": ["yuv420p"],
            "output.faststart": [True], "base_display_dimension_alignment": [2]})
    contract = {"contract_version": version, "input_roles": ["base_image" if media == "image" else "base_video", "overlay_image"],
        "output_roles": [role], "required_output_roles": [role], "optional_output_roles": [],
        "output_mime_types_by_role": {role: [mime]}, "geometry_policy_id": POLICY, "recipe": recipe}
    return {
        "family": "media_processing", "media_type": media, "processing_task": task, "processor_id": processor,
        "supported": True, "detected": True, "dependencies": {"pillow": True},
        "supported_operations": [operation], "operation_types": [operation],
        "input_mime_types": {"source_image" if media == "image" else "source_video": image_mimes if media == "image" else ["video/mp4", "video/webm", "video/quicktime"], "overlay_png": ["image/png"]},
        "contracts": {operation: contract},
        "operation_features": {operation: [version, POLICY, "source_relative_scale", "signed_pixel_offsets", "canvas_clipping", "typed_artifact_roles"] + (["source_display_dimensions", "base_audio_or_silence"] if media == "video" else [])},
        "limits": {operation: limits(operation)},
        "resource_safeguards": {"memory_bytes": MEMORY_BYTES, "max_scratch_bytes": SCRATCH_BYTES,
            "memory_limit_scope": "job_process_tree" if os.name == "nt" else "per_process_address_space",
            "max_execution_seconds": TIMEOUTS[operation], "max_active_jobs": 1},
    }


class ProcessTree:
    def __init__(self, process):
        self.process = process
    def poll(self):
        return self.process.poll()
    def terminate(self):
        self._signal(signal.SIGTERM)
    def kill(self):
        self._signal(signal.SIGKILL if os.name != "nt" else signal.SIGTERM)
    def _signal(self, sig):
        if os.name == "nt":
            if self.process.poll() is None:
                self.process.kill()
        else:
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
    def stop(self):
        if self.poll() is None:
            self.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.kill()
                self.process.wait(timeout=5)
        if os.name != "nt":
            # The supervisor may already have exited while a decoder descendant remains.
            self.kill()
        require(self.poll() is not None, "Overlay process exit could not be verified")


def supervise(command, directory, timeout, handle=None, tick=None, output_limit=VIDEO_BYTES):
    directory = Path(directory)
    require(shutil.disk_usage(directory).free >= SCRATCH_BYTES, "Insufficient overlay scratch space")
    log = directory / "overlay-worker.log"
    with log.open("wb") as stream:
        raw = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=os.name != "nt")
        process = ProcessTree(raw)
        if handle:
            handle.set_process(process)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if tick:
                    tick()
                if handle and handle.cancelled:
                    raise RuntimeError("cancel_requested")
                require(time.monotonic() < deadline, "Overlay execution timed out")
                sizes = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file()}
                require(sum(sizes.values()) <= SCRATCH_BYTES and shutil.disk_usage(directory).free >= 64 * 1024 * 1024, "Overlay scratch limit exceeded")
                require(sizes.get(log.name, 0) <= LOG_BYTES and sizes.get("native-command.log", 0) <= LOG_BYTES, "Overlay diagnostic limit exceeded")
                require(sizes.get("overlay-result.mp4", 0) <= output_limit and sizes.get("overlay-result.png", 0) <= output_limit, "Overlay output byte limit exceeded")
                time.sleep(0.1)
            if tick:
                tick()
            if handle and handle.cancelled:
                raise RuntimeError("cancel_requested")
            require(raw.returncode == 0, "Bounded overlay worker failed validation or execution")
            require(log.stat().st_size <= LOG_BYTES, "Overlay diagnostic limit exceeded")
        finally:
            process.stop()
            if handle:
                handle.clear_process(process)
    return log


@lru_cache(maxsize=8)
def dependency_probe(ffmpeg=None, ffprobe=None):
    with tempfile.TemporaryDirectory(prefix="midom-overlay-probe-") as directory:
        log = supervise([sys.executable, str(Path(__file__).resolve()), "probe"], directory, 15)
        require(log.stat().st_size <= RESULT_BYTES, "Overlay dependency probe exceeded limit")
        result = json.loads(log.read_text(encoding="utf-8"))
        if ffmpeg and ffprobe:
            # Only trusted installed executables, never request-supplied arguments.
            output = supervise([ffmpeg, "-hide_banner", "-encoders"], directory, 15).read_text(encoding="utf-8", errors="replace")
            result["video_available"] = any(line.split()[1:2] == ["libx264"] for line in output.splitlines()) and any(line.split()[1:2] == ["aac"] for line in output.splitlines())
            supervise([ffprobe, "-version"], directory, 15)
        return result


def number(value, low, high, integer=False):
    try:
        return type(value) in ((int,) if integer else (int, float)) and math.isfinite(value) and low <= value <= high
    except OverflowError:
        return False


class OverlayValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise OverlayValidationError(message)


def identify(payload):
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    processing = payload.get("processing") or summary.get("processing") or {}
    values = [payload.get("operation_type"), summary.get("operation_type")]
    if isinstance(processing, dict):
        values += [processing.get("operation_type"), processing.get("operation")]
    if IMAGE_OPERATION in values or payload.get("processor_id") == "image_overlay_processor" or payload.get("model_id") == "image_overlay_processor":
        return IMAGE_OPERATION
    return VIDEO_OPERATION if VIDEO_OPERATION in values else None


def effective_limits(operation, advertised=None, requested=None):
    result = limits(operation)
    for supplied in (advertised, requested):
        if supplied is None:
            continue
        require(isinstance(supplied, dict) and not set(supplied) - set(result), "Unsupported overlay limits")
        for key, value in supplied.items():
            if isinstance(result[key], dict):
                require(isinstance(value, dict) and set(value) == {"min", "max"}, "Invalid overlay range limit")
                require(all(number(value[k], -1e9, 1e9) for k in value), "Invalid overlay range limit")
                result[key] = {"min": max(result[key]["min"], value["min"]), "max": min(result[key]["max"], value["max"])}
                require(result[key]["min"] <= result[key]["max"], "Empty overlay range limit")
            else:
                require(number(value, 1, 2**53, integer=True), "Invalid overlay resource limit")
                result[key] = min(result[key], value)
    return result


def placement(base, overlay, visual, bounds):
    scaled = [max(1, round(d * (visual["scale_percent"] / 100.0))) for d in overlay]
    require(math.prod(scaled) <= bounds["max_scaled_overlay_pixels"], "Scaled overlay pixel limit exceeded")
    anchor = visual["anchor"]
    left = 0 if anchor.endswith("left") else base[0] - scaled[0] if anchor.endswith("right") else round((base[0] - scaled[0]) / 2.0)
    top = 0 if anchor.startswith("top") else base[1] - scaled[1] if anchor.startswith("bottom") else round((base[1] - scaled[1]) / 2.0)
    left, top = left + visual["offset_x"], top + visual["offset_y"]
    box = [max(0, left), max(0, top), min(base[0], left + scaled[0]), min(base[1], top + scaled[1])]
    return {
        "anchor": anchor, "scale_percent": visual["scale_percent"], "offset_x": visual["offset_x"], "offset_y": visual["offset_y"],
        "requested_opacity": visual["opacity"], "effective_opacity": visual["opacity"] if visual["opacity"] < 0.999 else 1.0,
        "overlay_width": scaled[0], "overlay_height": scaled[1], "left": left, "top": top,
        "visible_box": box if box[2] > box[0] and box[3] > box[1] else None,
    }


def validate_claim(job, advertised=None, candidate=False):
    operation = identify(job)
    require(operation in ROUTES, "Unsupported overlay operation")
    media, processor, task, version, role, fmt, mime, color = ROUTES[operation]
    require(not any(k in job for k in ("prompt", "generation", "custom_settings")), "Overlay processing does not accept generation settings or prompts")
    require(all(job.get(k) == v for k, v in {"family": "media_processing", "media_type": media, "model_id": processor,
        "processor_id": processor, "processing_task": task, "operation_type": operation}.items()), "Overlay processor/task/operation mismatch")
    processing = job.get("processing")
    allowed = {"operation_type", "contract_version", "mediaassemblyjobid", "geometry_policy_id", "orientation_mode", "visual"}
    if media == "video":
        allowed |= {"duration_mode", "audio", "output"}
    require(isinstance(processing, dict) and set(processing) == allowed, "Unsupported or missing overlay recipe field")
    require(processing["operation_type"] == operation and processing["contract_version"] == version, "Overlay contract version mismatch")
    require("contract_version" not in job or job["contract_version"] == version, "Overlay top-level contract mismatch")
    require(processing["geometry_policy_id"] == POLICY and number(processing["mediaassemblyjobid"], 1, 2**53, True), "Invalid overlay geometry policy or parent identity")
    bounds = effective_limits(operation, advertised, job.get("limits"))
    visual = processing["visual"]
    keys = {"anchor", "scale_reference", "scale_percent", "offset_x", "offset_y", "opacity", "alpha_mode"} | ({"end_behavior"} if media == "video" else set())
    require(isinstance(visual, dict) and set(visual) == keys, "Unsupported or missing overlay visual field")
    require(visual["anchor"] in ANCHORS and visual["scale_reference"] == "oriented_overlay_source" and visual["alpha_mode"] == "png_alpha", "Unsupported overlay placement")
    for key in ("scale_percent", "opacity", "offset_x", "offset_y"):
        require(number(visual[key], bounds[key]["min"], bounds[key]["max"], key.startswith("offset")), f"Invalid overlay {key}")
    require(processing["orientation_mode"] == ("exif_transpose" if media == "image" else "physical_display_orientation"), "Unsupported overlay orientation")
    if media == "video":
        require(processing["duration_mode"] == "source_full" and visual["end_behavior"] == "repeat"
            and processing["audio"] == {"mode": "base_or_silence"}
            and processing["output"] == {"container": "mp4", "video_codec": "h264", "audio_codec": "aac", "pixel_format": "yuv420p", "faststart": True}
            and processing["output"]["faststart"] is True, "Unsupported positioned video recipe")
    inputs = job.get("inputs")
    require(isinstance(inputs, list) and len(inputs) == 2, "Overlay requires exactly two inputs")
    for index, item in enumerate(inputs):
        kind, input_role = (("source_image", "base_image") if media == "image" else ("source_video", "base_video")) if index == 0 else ("overlay_png", "overlay_image")
        mimes = set(MIMES.values()) if kind == "source_image" else {"video/mp4", "video/quicktime", "video/webm"} if kind == "source_video" else {"image/png"}
        require(isinstance(item, dict) and item.get("kind") == kind and item.get("role") == input_role
            and type(item.get("sequence")) is int and item["sequence"] == index
            and number(item.get("input_id"), 1, 2**53, True) and item.get("mime_type") in mimes, "Invalid overlay input role, sequence, ID or MIME")
        if not candidate:
            require(isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in item["sha256"]), "Overlay input requires SHA-256")
    require(inputs[0]["input_id"] != inputs[1]["input_id"], "Overlay input IDs must be distinct")
    output = job.get("output")
    require(isinstance(output, dict) and not set(output) - {"count", "format", "mime_type", "color_mode", "width", "height", "artifacts", "contract_version"}
        and ("contract_version" not in output or output["contract_version"] == version)
        and type(output.get("count")) is int and output["count"] == 1
        and output.get("format") == fmt and output.get("mime_type") == mime and output.get("color_mode") == color
        and output.get("artifacts") == [{"artifact_index": 0, "role": role, "mime_type": mime, "required": True}]
        and type(output["artifacts"][0]["artifact_index"]) is int and output["artifacts"][0]["required"] is True, "Invalid overlay typed output")
    require(all(number(output.get(k), 1, bounds["max_output_pixels"], True) for k in ("width", "height"))
        and output["width"] * output["height"] <= bounds["max_output_pixels"], "Overlay output pixel limit exceeded")
    if media == "video":
        require(output["width"] % 2 == output["height"] % 2 == 0, "Positioned video requires even display dimensions; no resizing is permitted")
    return {"operation": operation, "processing": copy.deepcopy(processing), "output": copy.deepcopy(output), "limits": bounds}


def validate_candidate(payload, advertised=None):
    summary = payload.get("summary")
    require(isinstance(summary, dict), "Overlay candidate summary is required")
    require(isinstance(summary.get("processing"), dict) and summary.get("contract_version") == summary["processing"].get("contract_version"), "Overlay candidate contract mismatch")
    operation = identify(payload)
    require(operation in ROUTES, "Unsupported overlay candidate operation")
    media, _, _, _, role, fmt, mime, color = ROUTES[operation]
    inputs = summary.get("inputs")
    require(isinstance(inputs, list) and len(inputs) == 2, "Overlay candidate input descriptors are required")
    output = {key: summary.get("output_" + key) for key in ("count", "format", "mime_type", "color_mode", "width", "height")}
    output["artifacts"] = [{"artifact_index": 0, "role": role, "mime_type": mime, "required": True}]
    job = {**payload, "processing": summary["processing"], "inputs": inputs, "output": output}
    recipe = validate_claim(job, advertised, candidate=True)
    expected_counts = {"overlay_png_count": 1, "audio_input_count": 0, "video_input_count": int(media == "video"), "source_image_count" if media == "image" else "source_video_count": 1}
    require(all(type(summary.get(k)) is int and summary[k] == v for k, v in expected_counts.items()), "Overlay candidate input counts mismatch")
    for index, item in enumerate(inputs):
        validate_facts(item, recipe["limits"], video=media == "video" and index == 0)
    base = [inputs[0][("display_" if media == "video" else "oriented_") + k] for k in ("width", "height")]
    require(base == [output["width"], output["height"]], "Overlay must preserve source display dimensions")
    placement(base, [inputs[1]["oriented_width"], inputs[1]["oriented_height"]], recipe["processing"]["visual"], recipe["limits"])
    return recipe


def match_candidate_claim(candidate, claim, advertised=None):
    selected, actual = validate_candidate(candidate, advertised), validate_claim(claim, advertised)
    for recipe in (selected, actual):
        recipe["output"].pop("contract_version", None)
    require(candidate["job_id"] == claim["job_id"] and selected == actual, "Overlay candidate/claim recipe or output mismatch")
    keys = ("input_id", "kind", "role", "sequence", "mime_type")
    require([[item[k] for k in keys] for item in candidate["summary"]["inputs"]]
        == [[item[k] for k in keys] for item in claim["inputs"]], "Overlay candidate/claim input mismatch")
    require(all(candidate[k] == claim["target"][k] for k in ("org_id", "project_id", "requested_by_user_id")), "Overlay candidate/claim scope mismatch")


def validate_completion_metadata(metadata, job, recipe):
    def depth(value):
        if isinstance(value, dict):
            return 1 + max([depth(v) for v in value.values()] or [0])
        if isinstance(value, list):
            return 1 + max([depth(v) for v in value] or [0])
        return 0
    require(isinstance(metadata, dict) and depth(metadata) <= 7
        and len(json.dumps(metadata, allow_nan=False).encode()) <= 30000, "Overlay metadata envelope limit exceeded")
    operation = recipe["operation"]
    require(all(metadata.get(k) == v for k, v in {
        "operation_type": operation, "contract_version": ROUTES[operation][3], "geometry_policy_id": POLICY,
        "mediaassemblyjobid": recipe["processing"]["mediaassemblyjobid"], "worker_job_id": job["job_id"],
        "orientation_mode": recipe["processing"]["orientation_mode"],
    }.items()), "Overlay completion identity mismatch")
    inputs = metadata.get("inputs")
    require(isinstance(inputs, list) and len(inputs) == 2, "Overlay completion input provenance missing")
    for actual, supplied in zip(inputs, job["inputs"]):
        require(all(actual.get(k) == supplied[k] for k in ("input_id", "role", "sequence", "sha256")), "Overlay completion input identity mismatch")
    outputs = metadata.get("outputs")
    require(isinstance(outputs, list) and len(outputs) == 1, "Overlay completion must contain one artifact")
    output = outputs[0]
    require(output.get("artifact_index") == 0 and output.get("role") == ROUTES[operation][4]
        and output.get("mime_type") == ROUTES[operation][6]
        and all(output.get(k) == recipe["output"][k] for k in ("width", "height")), "Overlay completion output declaration mismatch")
    require(number(output.get("encoded_bytes"), 1, recipe["limits"]["max_artifact_bytes"], True)
        and isinstance(output.get("sha256"), str) and len(output["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in output["sha256"]), "Overlay completion output integrity metadata invalid")
    base_prefix = "oriented_" if operation == IMAGE_OPERATION else "display_"
    expected = placement([inputs[0][base_prefix + k] for k in ("width", "height")],
        [inputs[1]["oriented_" + k] for k in ("width", "height")], recipe["processing"]["visual"], recipe["limits"])
    require(metadata.get("placement") == expected, "Overlay completion placement mismatch")
    if operation == IMAGE_OPERATION:
        require(output.get("decoded_format") == "PNG" and output.get("color_mode") == "rgba", "Overlay completion must be RGBA PNG")
    else:
        require(output.get("video_codec") == "h264" and output.get("audio_codec") == "aac" and output.get("container") == "mp4"
            and output.get("pixel_format") == "yuv420p" and output.get("has_audio") is True
            and output.get("rotation_degrees") == 0, "Overlay completion must be physical-orientation H264/AAC MP4")


def validate_facts(item, bounds, video=False):
    require(number(item.get("encoded_bytes"), 1, bounds["max_video_input_bytes"] if video else bounds["max_image_input_bytes"], True), "Overlay input byte limit exceeded")
    prefix = "display_" if video else "oriented_"
    require(all(number(item.get(prefix + k), 1, bounds["max_output_pixels"] if video else bounds["max_oriented_image_pixels"], True) for k in ("width", "height")), "Invalid overlay input dimensions")
    require(item[prefix + "width"] * item[prefix + "height"] <= (bounds["max_output_pixels"] if video else bounds["max_oriented_image_pixels"]), "Overlay input pixel limit exceeded")
    if video:
        require(number(item.get("duration_seconds"), 0.000001, bounds["max_video_duration_seconds"]), "Overlay source duration limit exceeded")
        require(item["display_width"] % 2 == item["display_height"] % 2 == 0, "Positioned video requires even display dimensions")


class OutcomeUncertain(RuntimeError):
    overlay_outcome_uncertain = True


def apply_resource_limits(file_limit=VIDEO_BYTES):
    if sys.platform.startswith("linux"):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        return
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("user", ctypes.c_int64), ("job", ctypes.c_int64), ("flags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t), ("active", w.DWORD), ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.GetCurrentProcess.restype = w.HANDLE
        handle = kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.basic.flags = 0x2000 | 0x200
        info.job_memory = MEMORY_BYTES
        require(handle and kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info))
            and kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()), "Windows job memory/process-tree guard unavailable")
        # The handle stays alive until worker exit; KILL_ON_JOB_CLOSE covers its FFmpeg children.
        return
    raise ValueError("Bounded overlay execution is unsupported on this platform")


class LimitedWriter:
    def __init__(self, path, limit):
        self.file = open(path, "wb")
        self.limit = limit
    def write(self, data):
        require(self.file.tell() + len(data) <= self.limit, "Overlay PNG output byte limit exceeded")
        return self.file.write(data)
    def flush(self):
        self.file.flush()
    def close(self):
        self.file.close()


def save_png(image, path, limit):
    writer = LimitedWriter(path, limit)
    try:
        image.save(writer, format="PNG", optimize=True)
    finally:
        writer.close()


def read_image(item, bounds):
    path = Path(item["path"])
    require(0 < path.stat().st_size <= bounds["max_image_input_bytes"], "Overlay image byte limit exceeded")
    with Image.open(path) as source:
        require(source.format in MIMES and MIMES[source.format] == item["mime_type"], "Overlay image encoding/MIME mismatch")
        require(item["kind"] != "overlay_png" or source.format == "PNG", "Overlay must decode as genuine PNG")
        require(getattr(source, "n_frames", 1) == 1, "Animated overlay inputs are unsupported")
        require(source.width * source.height <= bounds["max_oriented_image_pixels"], "Overlay decoded pixel limit exceeded")
        image = ImageOps.exif_transpose(source).convert("RGBA")
        image.load()
        return image


def composite(base, layer, visual, bounds):
    info = placement(base.size, layer.size, visual, bounds)
    size = (info["overlay_width"], info["overlay_height"])
    if layer.size != size:
        layer = layer.resize(size, Image.Resampling.LANCZOS)
    if visual["opacity"] < 0.999:
        layer.putalpha(layer.getchannel("A").point(lambda alpha: round(alpha * visual["opacity"])))
    result = base.copy()
    box = info["visible_box"]
    if box:
        left, top = info["left"], info["top"]
        visible = layer.crop((box[0] - left, box[1] - top, box[2] - left, box[3] - top))
        result.alpha_composite(visible, dest=(box[0], box[1]))
    return result, info


def bounded_command(command, directory, timeout, watched=None, output_limit=VIDEO_BYTES):
    log = Path(directory) / "native-command.log"
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                require(time.monotonic() < deadline, "Overlay native command timed out")
                require(log.stat().st_size <= LOG_BYTES, "Overlay native diagnostic limit exceeded")
                if watched and Path(watched).exists():
                    require(Path(watched).stat().st_size <= output_limit, "Overlay output byte limit exceeded")
                time.sleep(0.05)
            require(process.returncode == 0, "Overlay native command failed")
            require(log.stat().st_size <= LOG_BYTES, "Overlay native diagnostic limit exceeded")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
    return log


def probe_video(path, ffprobe, directory, source=True):
    log = bounded_command([ffprobe, "-v", "error", "-threads", "2", "-protocol_whitelist", "file,pipe", "-count_frames",
        "-show_entries", "format=format_name,duration:stream=codec_type,codec_name,width,height,sample_aspect_ratio,avg_frame_rate,nb_read_frames,duration,pix_fmt:stream_tags=rotate:stream_side_data_list",
        "-of", "json", str(path)], directory, 120)
    require(log.stat().st_size <= RESULT_BYTES, "Overlay probe result exceeds limit")
    value = json.loads(log.read_text(encoding="utf-8"))
    videos = [s for s in value.get("streams", []) if s.get("codec_type") == "video"]
    audios = [s for s in value.get("streams", []) if s.get("codec_type") == "audio"]
    require(len(videos) == 1 and len(audios) <= 1, "Overlay requires one video stream and at most one base audio stream")
    video = videos[0]
    rotation = float(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    require(number(rotation, -3600, 3600) and rotation % 90 == 0, "Unsupported overlay video rotation")
    require(video.get("sample_aspect_ratio", "1:1") in {"1:1", "N/A"}, "Non-square video pixels are unsupported")
    width, height = int(video["width"]), int(video["height"])
    if int(rotation) % 180:
        width, height = height, width
    durations = [float(s["duration"]) for s in [video, value.get("format", {})] if s.get("duration") not in (None, "N/A")]
    require(durations and all(number(d, 0.000001, 121) for d in durations), "Overlay video duration limit exceeded or unavailable")
    duration = float(video["duration"]) if video.get("duration") not in (None, "N/A") else max(durations)
    if source:
        require(duration <= 120 and max(durations) <= duration + 0.1, "Overlay source video duration limit exceeded or audio extends beyond the video timeline")
    fps = Fraction(video.get("avg_frame_rate", "0/1"))
    frames = int(video.get("nb_read_frames", 0))
    require(fps > 0 and frames > 0, "Overlay video timeline could not be decoded")
    return {"display_width": width, "display_height": height, "rotation_degrees": int(rotation) % 360,
        "duration_seconds": duration, "measured_duration_seconds": max(durations),
        "fps": float(fps), "frame_count": frames, "has_audio": bool(audios),
        "video_codec": video.get("codec_name"), "audio_codec": audios[0].get("codec_name") if audios else None,
        "pixel_format": video.get("pix_fmt"), "container": value.get("format", {}).get("format_name", "")}


def execute(spec, directory):
    operation = spec["operation"]
    recipe, inputs = spec["recipe"], spec["inputs"]
    bounds, visual = recipe["limits"], recipe["processing"]["visual"]
    facts = []
    for item in inputs:
        require(hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"], "Overlay downloaded input hash changed")
    layer = read_image(inputs[1], bounds)
    if operation == IMAGE_OPERATION:
        base = read_image(inputs[0], bounds)
        base_size = list(base.size)
        source = None
    else:
        source = probe_video(inputs[0]["path"], spec["ffprobe"], directory)
        require(any(token in source["container"] for token in ("mov", "mp4", "webm", "matroska")), "Unsupported overlay video container")
        validate_facts({**source, "encoded_bytes": Path(inputs[0]["path"]).stat().st_size}, bounds, video=True)
        base_size = [source["display_width"], source["display_height"]]
        base = Image.new("RGBA", tuple(base_size), (0, 0, 0, 0))
    require(base_size == [recipe["output"]["width"], recipe["output"]["height"]], "Overlay output must match actual oriented source dimensions")
    require(math.prod(base_size) <= bounds["max_output_pixels"], "Overlay output pixel limit exceeded")
    overlay_size = list(layer.size)
    result, geometry = composite(base, layer, visual, bounds)
    for index, item in enumerate(inputs):
        facts.append({k: item[k] for k in ("input_id", "role", "sequence", "sha256")})
        prefix = "display_" if index == 0 and source else "oriented_"
        facts[-1].update({prefix + "width": base_size[0] if index == 0 else overlay_size[0], prefix + "height": base_size[1] if index == 0 else overlay_size[1]})
    output = Path(directory) / ("overlay-result.png" if operation == IMAGE_OPERATION else "overlay-result.mp4")
    if operation == IMAGE_OPERATION:
        save_png(result, output, bounds["max_artifact_bytes"])
        with Image.open(output) as check:
            check.load()
            require(check.format == "PNG" and check.mode == "RGBA" and list(check.size) == base_size, "Overlay PNG validation failed")
        measured = {"decoded_format": "PNG", "color_mode": "rgba"}
    else:
        canvas = Path(directory) / "positioned-overlay.png"
        save_png(result, canvas, PNG_BYTES)
        base = layer = result = None
        duration = source["duration_seconds"]
        command = [spec["ffmpeg"], "-y", "-hide_banner", "-v", "error", "-xerror", "-threads", "2", "-filter_complex_threads", "1", "-filter_threads", "1",
            "-protocol_whitelist", "file,pipe", "-i", inputs[0]["path"], "-loop", "1", "-threads", "1", "-i", str(canvas)]
        if not source["has_audio"]:
            command += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio = "0:a:0" if source["has_audio"] else "2:a:0"
        filters = f"[0:v]setpts=PTS-STARTPTS,setsar=1,format=rgba[base];[base][1:v]overlay=0:0:format=rgb:eof_action=repeat:shortest=1,format=yuv420p[v];[{audio}]asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,apad[a]"
        command += ["-filter_complex", filters, "-map", "[v]", "-map", "[a]", "-t", str(duration), "-vsync", "0",
            "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
            "-metadata:s:v:0", "rotate=0", "-movflags", "+faststart", "-fs", str(bounds["max_artifact_bytes"] + 1), str(output)]
        bounded_command(command, directory, 1500, output, bounds["max_artifact_bytes"])
        measured = probe_video(output, spec["ffprobe"], directory, source=False)
        require(measured["display_width"] == base_size[0] and measured["display_height"] == base_size[1]
            and measured["rotation_degrees"] == 0 and measured["video_codec"] == "h264" and measured["audio_codec"] == "aac"
            and measured["pixel_format"] == "yuv420p" and measured["has_audio"]
            and measured["frame_count"] == source["frame_count"]
            and abs(measured["duration_seconds"] - duration) <= max(0.001, 1 / source["fps"])
            and measured["measured_duration_seconds"] <= duration + max(0.1, 1 / source["fps"]), "Positioned video output/timeline validation failed")
        measured.update(container="mp4")
    size = output.stat().st_size
    require(0 < size <= bounds["max_artifact_bytes"], "Overlay artifact byte limit exceeded")
    measured.update(artifact_index=0, role=ROUTES[operation][4], mime_type=ROUTES[operation][6], width=base_size[0], height=base_size[1], encoded_bytes=size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    return {"operation_type": operation, "contract_version": ROUTES[operation][3], "geometry_policy_id": POLICY,
        "mediaassemblyjobid": recipe["processing"]["mediaassemblyjobid"], "worker_job_id": spec["job_id"],
        "orientation_mode": recipe["processing"]["orientation_mode"], "inputs": facts, "placement": geometry, "outputs": [measured],
        **({"duration_mode": "source_full", "audio_action": "base_transcoded_to_aac" if source["has_audio"] else "silent_fill"} if source else {})}


def main():
    apply_resource_limits()
    if len(sys.argv) == 2 and sys.argv[1] == "probe":
        print(json.dumps({"memory_guard": True, "image_mime_types": [mime for fmt, mime in MIMES.items() if fmt == "PNG" and features.check("zlib") or fmt == "JPEG" and features.check("jpg") or fmt == "WEBP" and features.check("webp")]}))
        return
    path = Path(sys.argv[1])
    require(path.stat().st_size <= RESULT_BYTES, "Overlay job specification too large")
    spec = json.loads(path.read_text(encoding="utf-8"))
    result = execute(spec, path.parent)
    encoded = json.dumps(result, allow_nan=False).encode()
    require(len(encoded) <= 32768, "Overlay completion metadata too large")
    (path.parent / "overlay-metadata.json").write_bytes(encoded)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Native decoder/FFmpeg errors may include paths or attacker-controlled metadata.
        print(str(exc) if isinstance(exc, OverlayValidationError) else "Bounded overlay worker failed decoding or execution", file=sys.stderr)
        sys.exit(1)
