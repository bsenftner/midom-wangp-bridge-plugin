"""Frozen wire examples and independent policy oracles, not overlay runtime support.

These tests validate the agreement and small encoded-image fixtures. They do
not execute a new Bridge renderer, Midom routing, or a recovery endpoint.
"""

import base64
import copy
import hashlib
import io
import json
import math
import re
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from test_overlay_png_on_video import load_plugin_module, overlay_job, plugin_instance


ROOT = Path(__file__).parent / "fixtures" / "media_assembly_overlays_v1"
SET_ID = "media_assembly_agent_overlays_v1_fixtures_v1"
FROZEN_MANIFEST_SHA256 = "f79b7c977b949024ebc53ea38ab68074a7454a376a2fe6749a85f6907bc3fc9e"
CAPS, JOBS, GEOMETRY, REJECTIONS, RECOVERY = [
    json.loads((ROOT / name).read_text(encoding="utf-8"))
    for name in ("capabilities.json", "jobs.json", "geometry.json", "rejections.json", "recovery.json")
]
ROUTES = {
    "image": ("image_overlay_processor", "deterministic_image_processing", "overlay_png_on_image", "png_image_overlay_v1", "image", "png", "image/png", "rgba"),
    "video": ("storyboard_ffmpeg_processor", "storyboard_ffmpeg_processing", "overlay_png_on_video_positioned", "positioned_png_video_overlay_v1", "video", "mp4", "video/mp4", "rgb"),
}
ANCHORS = {"center", "top_left", "top_right", "bottom_left", "bottom_right"}
MIMES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def documents(route):
    return {"capability": copy.deepcopy(CAPS["processors"][route == "video"]),
            "worker": copy.deepcopy(CAPS["worker_context"]), **copy.deepcopy(JOBS[route])}


def numeric(value, minimum, maximum):
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def placement(base, overlay, visual):
    width, height = [max(1, round(dimension * (visual["scale_percent"] / 100.0))) for dimension in overlay]
    anchor = visual["anchor"]
    left = 0 if anchor.endswith("left") else base[0] - width if anchor.endswith("right") else round((base[0] - width) / 2.0)
    top = 0 if anchor.startswith("top") else base[1] - height if anchor.startswith("bottom") else round((base[1] - height) / 2.0)
    left, top = left + visual["offset_x"], top + visual["offset_y"]
    visible = [max(0, left), max(0, top), min(base[0], left + width), min(base[1], top + height)]
    return [width, height], [left, top], visible if visible[2] > visible[0] and visible[3] > visible[1] else None


def effective_limits(route, capability):
    operation = ROUTES[route][2]
    ceiling = CAPS["processors"][route == "video"]["limits"][operation]
    advertised = capability.get("limits", {}).get(operation, {})
    result = {}
    for key, upper in ceiling.items():
        value = advertised.get(key)
        if isinstance(upper, dict):
            if not isinstance(value, dict) or not all(numeric(value.get(k), -1e9, 1e9) for k in ("min", "max")):
                return None
            result[key] = {"min": max(upper["min"], value["min"]), "max": min(upper["max"], value["max"])}
            if result[key]["min"] > result[key]["max"]:
                return None
        else:
            if type(value) is not int or value <= 0:
                return None
            result[key] = min(value, upper)
    return result


def claim_rejection(route, docs):
    claim = docs["claim"]
    processor, task, operation, version, role, fmt, mime, color = ROUTES[route]
    if "prompt" in claim or "generation" in claim:
        return "unsupported_generation"
    if any(claim.get(k) != v for k, v in {"family": "media_processing", "media_type": route,
        "model_id": processor, "processor_id": processor, "processing_task": task, "operation_type": operation}.items()):
        return "selection_invalid"
    processing = claim.get("processing", {})
    allowed = {"operation_type", "contract_version", "mediaassemblyjobid", "geometry_policy_id", "orientation_mode", "visual"}
    if route == "video":
        allowed |= {"duration_mode", "audio", "output"}
    if set(processing) - allowed:
        return "unsupported_recipe_field"
    if processing.get("operation_type") != operation or processing.get("contract_version") != version:
        return "contract_unavailable"
    if processing.get("geometry_policy_id") != "midom_overlay_geometry_v1":
        return "invalid_placement"
    visual = processing.get("visual", {})
    visual_keys = {"anchor", "scale_reference", "scale_percent", "offset_x", "offset_y", "opacity", "alpha_mode"}
    if route == "video":
        visual_keys.add("end_behavior")
    if set(visual) - visual_keys:
        return "unsupported_recipe_field"
    if (visual.get("anchor") not in ANCHORS or visual.get("scale_reference") != "oriented_overlay_source"
        or visual.get("alpha_mode") != "png_alpha" or not numeric(visual.get("scale_percent"), 1, 500)
        or not numeric(visual.get("opacity"), 0, 1)
        or any(type(visual.get(key)) is not int or not -16384 <= visual[key] <= 16384 for key in ("offset_x", "offset_y"))):
        return "invalid_placement"
    limits = effective_limits(route, docs["capability"])
    if limits is None:
        return "limits_unavailable"
    if any(not numeric(visual[key], limits[key]["min"], limits[key]["max"]) for key in ("scale_percent", "opacity", "offset_x", "offset_y")):
        return "resource_limit_exceeded"
    if route == "image":
        if processing.get("orientation_mode") != "exif_transpose":
            return "invalid_orientation"
    elif (processing.get("orientation_mode") != "physical_display_orientation"
          or processing.get("duration_mode") != "source_full" or visual.get("end_behavior") != "repeat"
          or processing.get("audio") != {"mode": "base_or_silence"}
          or processing.get("output") != {"container": "mp4", "video_codec": "h264", "audio_codec": "aac", "pixel_format": "yuv420p", "faststart": True}):
        return "unsupported_recipe"
    inputs = claim.get("inputs", [])
    if len(inputs) != 2:
        return "input_contract_invalid"
    expected = [("source_image", "base_image") if route == "image" else ("source_video", "base_video"), ("overlay_png", "overlay_image")]
    for index, (item, pair) in enumerate(zip(inputs, expected)):
        allowed_mimes = (set(MIMES.values()) if route == "image" else {"video/mp4", "video/webm", "video/quicktime"}) if index == 0 else {"image/png"}
        if ((item.get("kind"), item.get("role")) != pair or type(item.get("sequence")) is not int
            or item["sequence"] != index or item.get("mime_type") not in allowed_mimes):
            return "input_contract_invalid"
    output = claim.get("output", {})
    if (type(output.get("count")) is not int or output["count"] != 1 or output.get("format") != fmt
        or output.get("mime_type") != mime or output.get("color_mode") != color
        or any(type(output.get(k)) is not int or output[k] <= 0 for k in ("width", "height"))
        or output.get("artifacts") != [{"artifact_index": 0, "role": role, "mime_type": mime, "required": True}]):
        return "output_contract_invalid"
    return None


def media_rejection(route, docs):
    claim, inputs = docs["claim"], docs["decoded_inputs"]
    limits = effective_limits(route, docs["capability"])
    if limits is None:
        return "limits_unavailable"
    if len(inputs) != 2:
        return "input_contract_invalid"
    for index, item in enumerate(inputs):
        if item["input_id"] != claim["inputs"][index]["input_id"] or item["sequence"] != index:
            return "input_contract_invalid"
        if route == "video" and index == 0:
            if (not numeric(item["duration_seconds"], 0.000001, limits["max_video_duration_seconds"])
                or not numeric(item["encoded_bytes"], 1, limits["max_video_input_bytes"])):
                return "resource_limit_exceeded"
            if item["display_width"] % 2 or item["display_height"] % 2:
                return "video_geometry_unsupported"
            continue
        if index == 1 and item["decoded_format"] != "PNG":
            return "overlay_not_png"
        if item["decoded_format"] not in MIMES:
            return "unsupported_image_format"
        if MIMES[item["decoded_format"]] != claim["inputs"][index]["mime_type"]:
            return "mime_encoding_mismatch"
        if item["frames"] != 1:
            return "animated_image_unsupported"
        if (not numeric(item["encoded_bytes"], 1, limits["max_image_input_bytes"])
            or not 1 <= item["oriented_width"] * item["oriented_height"] <= limits["max_oriented_image_pixels"]):
            return "resource_limit_exceeded"
    base = [inputs[0][f"{'oriented' if route == 'image' else 'display'}_{key}"] for key in ("width", "height")]
    if base != [claim["output"][key] for key in ("width", "height")]:
        return "output_dimensions_mismatch"
    overlay = [inputs[1][f"oriented_{key}"] for key in ("width", "height")]
    scaled, _, _ = placement(base, overlay, claim["processing"]["visual"])
    if math.prod(scaled) > limits["max_scaled_overlay_pixels"] or math.prod(base) > limits["max_output_pixels"]:
        return "resource_limit_exceeded"
    return None


def selection_rejection(route, docs):
    worker, capability, candidate, claim = (docs[k] for k in ("worker", "capability", "candidate", "claim"))
    for key, code in (("paired", "worker_unpaired"), ("online", "worker_offline"), ("authorized", "worker_unauthorized"), ("accepting", "worker_not_accepting")):
        if worker.get(key) is not True:
            return code
    if worker.get("status") not in {"idle", "busy"}:
        return "worker_unavailable"
    processor, task, operation, version, role, _, mime, _ = ROUTES[route]
    if capability.get("processor_id") != processor or capability.get("processing_task") != task:
        return "processor_unavailable"
    if (capability.get("supported") is not True or capability.get("detected") is not True
        or (route == "image" and capability.get("dependencies", {}).get("pillow") is not True)
        or (route == "video" and (not capability.get("h264_encoder")
            or any(capability.get(k) is not True for k in ("ffmpeg_available", "ffprobe_available", "aac_encoder_available"))))):
        return "dependency_unavailable"
    if any(operation not in capability.get(k, []) for k in ("supported_operations", "operation_types")):
        return "operation_unavailable"
    record = capability.get("contracts", {}).get(operation, {})
    if record.get("contract_version") != version:
        return "contract_unavailable"
    if (record.get("required_output_roles") != [role] or record.get("output_mime_types_by_role", {}).get(role) != [mime]
        or record.get("input_roles") != (["base_image", "overlay_image"] if route == "image" else ["base_video", "overlay_image"])
        or record.get("geometry_policy_id") != "midom_overlay_geometry_v1"):
        return "output_contract_unavailable"
    if effective_limits(route, capability) is None:
        return "limits_unavailable"
    summary = candidate.get("summary", {})
    if (any(candidate.get(k) != claim.get(k) for k in ("job_id", "family", "media_type", "model_id", "processor_id", "processing_task", "operation_type"))
        or summary.get("contract_version") != claim["processing"]["contract_version"]
        or summary.get("processing") != claim["processing"]
        or summary.get("output_count") != claim["output"]["count"]
        or any(summary.get(f"output_{k}") != claim["output"][k] for k in ("format", "mime_type", "width", "height", "color_mode"))
        or len(summary.get("inputs", [])) != len(claim["inputs"])
        or summary.get("overlay_png_count") != 1 or summary.get("audio_input_count") != 0
        or summary.get("source_image_count" if route == "image" else "source_video_count") != 1
        or summary.get("video_input_count") != (1 if route == "video" else 0)):
        return "candidate_claim_mismatch"
    for ref, selected in zip(claim["inputs"], summary.get("inputs", [])):
        if any(ref.get(key) != selected.get(key) for key in ("input_id", "kind", "role", "sequence", "mime_type")):
            return "candidate_claim_mismatch"
        if ref["mime_type"] not in capability.get("input_mime_types", {}).get(ref["kind"], []):
            return "input_format_unavailable"
    for selected, decoded in zip(summary["inputs"], docs["decoded_inputs"]):
        keys = ["encoded_bytes", *(f"{'display' if selected['kind'] == 'source_video' else 'oriented'}_{k}" for k in ("width", "height"))]
        if selected["kind"] == "source_video":
            keys.append("duration_seconds")
        if any(selected.get(k) != decoded.get(k) for k in keys):
            return "candidate_claim_mismatch"
    if claim["processing"]["visual"]["anchor"] not in record.get("recipe", {}).get("visual.anchor", []):
        return "placement_feature_unavailable"
    return claim_rejection(route, docs) or media_rejection(route, docs)


def depth(value):
    children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return 1 + max((depth(child) for child in children), default=0)


def completion_rejection(route, docs):
    claim, complete = docs["claim"], docs["completion"]
    processor, _, operation, version, role, _, mime, color = ROUTES[route]
    metadata = complete["processing_metadata"]
    outputs = metadata.get("outputs", [])
    if (len(complete.get("artifacts", [])) != 1 or complete["artifacts"][0].get("artifact_index") != 0
        or complete["artifacts"][0].get("role") != role or len(outputs) != 1
        or complete.get("model_id") != processor or complete.get("processor_id") != processor
        or metadata.get("operation_type") != operation or metadata.get("contract_version") != version
        or metadata.get("mediaassemblyjobid") != claim["processing"]["mediaassemblyjobid"]
        or metadata.get("worker_job_id") != claim["job_id"] or len(metadata.get("inputs", [])) != 2):
        return "completion_invalid"
    for ref, recorded in zip(claim["inputs"], metadata.get("inputs", [])):
        if any(ref.get(k) != recorded.get(k) for k in ("input_id", "role", "sequence", "sha256")):
            return "completion_invalid"
    out = outputs[0]
    if (out.get("artifact_index") != 0 or out.get("role") != role or out.get("mime_type") != mime
        or not re.fullmatch("[0-9a-f]{64}", out.get("sha256", ""))
        or any(out.get(k) != claim["output"][k] for k in ("width", "height"))
        or not numeric(out.get("encoded_bytes"), 1, effective_limits(route, docs["capability"])["max_artifact_bytes"])):
        return "completion_invalid"
    if route == "image" and (out.get("decoded_format") != "PNG" or out.get("color_mode") != color):
        return "completion_invalid"
    if route == "video" and (any(out.get(k) != v for k, v in {"container": "mp4", "video_codec": "h264", "audio_codec": "aac", "pixel_format": "yuv420p", "rotation_degrees": 0, "has_audio": True}.items())
        or any(out.get(k) != docs["decoded_inputs"][0][k] for k in ("duration_seconds", "fps", "frame_count"))):
        return "completion_invalid"
    visual = claim["processing"]["visual"]
    scaled, position, visible = placement([out["width"], out["height"]], [docs["decoded_inputs"][1][f"oriented_{k}"] for k in ("width", "height")], visual)
    recorded = metadata["placement"]
    if ([recorded["overlay_width"], recorded["overlay_height"]] != scaled or [recorded["left"], recorded["top"]] != position
        or recorded["visible_box"] != visible or recorded["requested_opacity"] != visual["opacity"]
        or recorded["effective_opacity"] != (visual["opacity"] if visual["opacity"] < 0.999 else 1)):
        return "completion_invalid"
    if depth(complete) > 8 or len(json.dumps(complete, allow_nan=False).encode()) > 32768:
        return "completion_invalid"
    return None


def test_frozen_manifest_and_strict_json():
    assert hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest() == FROZEN_MANIFEST_SHA256
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["fixture_set_id"] == SET_ID
    assert manifest["runtime_implementation"] is manifest["runtime_advertisement"] is False
    for name, digest in manifest["files"].items():
        data = (ROOT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        def reject_constant(value):
            raise AssertionError(f"Non-JSON number {value}")
        assert json.loads(data, parse_constant=reject_constant)["fixture_set_id"] == SET_ID
    assert GEOMETRY["semantics"]["processing_order"] == [
        "decode_single_frame", "exif_transpose", "validate_scaled_pixel_budget", "resize_rgba_lanczos",
        "apply_opacity_threshold", "anchor_and_offset", "clip_to_canvas", "source_over",
    ]


@pytest.mark.parametrize("route", ROUTES)
def test_serialized_candidate_claim_and_completion_agree(route):
    docs = json.loads(json.dumps(documents(route), allow_nan=False))
    assert selection_rejection(route, docs) is None
    assert claim_rejection(route, docs) is None
    assert media_rejection(route, docs) is None
    assert completion_rejection(route, docs) is None
    module = load_plugin_module()
    plugin_instance(module)._validate_job_scope(docs["claim"], {"org_id": 10, "project_id": 180, "paired_user_id": 20})


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("field", ["org_id", "project_id", "requested_by_user_id"])
def test_claims_use_existing_scope_authorization_boundary(route, field):
    docs = documents(route)
    docs["claim"]["target"][field] += 1
    module = load_plugin_module()
    with pytest.raises(ValueError, match="does not match paired worker scope"):
        plugin_instance(module)._validate_job_scope(docs["claim"], {"org_id": 10, "project_id": 180, "paired_user_id": 20})


@pytest.mark.parametrize("case", REJECTIONS["cases"], ids=lambda case: case["id"])
def test_frozen_rejections(case):
    docs = documents(case["route"])
    target = docs[case["target"]]
    for key in case["path"][:-1]:
        target = target[key]
    key = case["path"][-1]
    if case.get("delete"):
        del target[key]
    elif "append" in case:
        target[key].append(case["append"])
    else:
        target[key] = case["value"]
    oracle = {"selection": selection_rejection, "claim": claim_rejection, "media": media_rejection, "completion": completion_rejection}[case["stage"]]
    assert oracle(case["route"], docs) == case["expected"]


@pytest.mark.parametrize("case", GEOMETRY["placement_cases"], ids=lambda case: case["id"])
def test_placement_geometry(case):
    visual = {"anchor": case["anchor"], "scale_percent": case["scale_percent"], "offset_x": case["offset"][0], "offset_y": case["offset"][1]}
    assert placement(case["base"], case["overlay"], visual) == (case["expected_scaled"], case["expected_position"], case["expected_visible_box"])


@pytest.mark.parametrize("case", GEOMETRY["opacity_cases"])
def test_alpha_multiplication_and_near_one_threshold(case):
    image = Image.new("RGBA", (len(case["input_alpha"]), 1))
    image.putalpha(Image.frombytes("L", image.size, bytes(case["input_alpha"])))
    if case["opacity"] < 0.999:
        image.putalpha(image.getchannel("A").point(lambda alpha: round(alpha * case["opacity"])))
    assert list(image.getchannel("A").getdata()) == case["expected_alpha"]
    assert (case["opacity"] if case["opacity"] < 0.999 else 1) == case["effective_opacity"]


@pytest.mark.parametrize("case", GEOMETRY["orientation_cases"])
def test_real_exif_orientation(case):
    image = Image.new("RGB", tuple(case["stored"]))
    image.putdata([(i, 0, 0) for i in range(1, 7)])
    exif = image.getexif()
    exif[274] = case["exif_orientation"]
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", exif=exif)
    with Image.open(io.BytesIO(encoded.getvalue())) as source:
        oriented = ImageOps.exif_transpose(source)
        assert list(oriented.size) == case["expected_oriented"]
        assert oriented.getexif().get(274, 1) == 1


@pytest.mark.parametrize("case", GEOMETRY["encoding_cases"], ids=lambda case: case["id"])
def test_real_encoded_bytes_not_extension(case):
    data = base64.b64decode(GEOMETRY["media_samples"][case["sample"]], validate=True)
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            assert decoded.size == (2, 3)
        result = "overlay_not_png" if case["role"] == "overlay_image" and fmt != "PNG" else "mime_encoding_mismatch" if MIMES[fmt] != case["declared_mime"] else "accepted"
    except (OSError, ValueError):
        result = "malformed_image"
    assert result == case["expected"]


@pytest.mark.parametrize("case", REJECTIONS["resource_boundaries"], ids=lambda case: case["id"])
def test_inclusive_resource_boundaries(case):
    limit = effective_limits(case["route"], documents(case["route"])["capability"])[case["measurement"]]
    assert ("accepted" if 0 < case["value"] <= limit else "resource_limit_exceeded") == case["expected"]


@pytest.mark.parametrize("route", ROUTES)
def test_lower_worker_limits_and_higher_advertisement_cannot_expand_contract(route):
    docs = documents(route)
    operation = ROUTES[route][2]
    limits = docs["capability"]["limits"][operation]
    for key, value in list(limits.items()):
        if key.startswith("max_"):
            original = effective_limits(route, docs["capability"])[key]
            limits[key] = original * 2
            assert effective_limits(route, docs["capability"])[key] == original
            limits[key] = max(1, original // 2)
            assert effective_limits(route, docs["capability"])[key] == max(1, original // 2)
            limits[key] = value
    limits["max_image_input_bytes"] = 100
    assert selection_rejection(route, docs) == "resource_limit_exceeded"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["scale_percent", "opacity", "offset_x", "offset_y"])
def test_nonfinite_values_rejected_before_serialization(value, field):
    docs = documents("image")
    docs["claim"]["processing"]["visual"][field] = value
    assert claim_rejection("image", docs) == "invalid_placement"
    with pytest.raises(ValueError):
        json.dumps(docs["claim"], allow_nan=False)


def test_image_is_independent_of_ffmpeg_and_busy_is_queueable():
    docs = documents("image")
    docs["capability"].update(ffmpeg_available=False, ffprobe_available=False)
    docs["worker"]["status"] = "busy"
    assert selection_rejection("image", docs) is None
    assert CAPS["discovery_policy"]["busy_accepting_worker"] == "queueable_not_immediately_claimable"
    assert CAPS["discovery_policy"]["video_hosted_fallback"] == "forbidden_including_dispatch_failure_and_lease_expiry"


def test_existing_full_frame_contract_is_unchanged():
    module = load_plugin_module()
    current = plugin_instance(module)._storyboard_ffmpeg_processing_capability()
    assert current["contracts"]["overlay_png_on_video"] == CAPS["processors"][1]["contracts"]["overlay_png_on_video"]
    # The frozen manifest records the earlier design-only stage. Runtime additions
    # are tested separately; they must never reinterpret this legacy contract.
    assert "overlay_png_on_image" not in current["supported_operations"]


def test_old_claim_still_uses_full_frame_and_rejects_placement_fields():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = overlay_job(module)
    assert plugin._validate_storyboard_ffmpeg_processing_job(job, job["job_id"])["_midom_processing"]["fit_mode"] == "scale_to_base"
    job["processing"]["visual"]["anchor"] = "bottom_right"
    with pytest.raises(ValueError, match="Unsupported.*visual field"):
        plugin._validate_storyboard_ffmpeg_processing_job(job, job["job_id"])


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("field,value", [("scale_percent", 20), ("opacity", 0.5), ("offset_x", -20), ("offset_y", 0)])
def test_worker_numeric_restrictions_are_effective(route, field, value):
    docs = documents(route)
    limits = docs["capability"]["limits"][ROUTES[route][2]]
    limits[field]["max"] = value if value >= 0 else limits[field]["max"]
    if field == "offset_x":
        limits[field]["min"] = value
    elif field == "offset_y":
        limits[field]["min"] = value
    assert claim_rejection(route, docs) == "resource_limit_exceeded"


@pytest.mark.parametrize("route", ROUTES)
def test_incomplete_candidate_and_missing_completion_provenance_fail(route):
    docs = documents(route)
    docs["candidate"]["summary"]["inputs"] = []
    assert selection_rejection(route, docs) == "candidate_claim_mismatch"
    docs = documents(route)
    docs["completion"]["processing_metadata"]["inputs"] = []
    assert completion_rejection(route, docs) == "completion_invalid"


@pytest.mark.parametrize("route", ROUTES)
def test_complete_metadata_envelope_is_bounded(route):
    docs = documents(route)
    docs["completion"]["processing_metadata"]["diagnostic"] = "x" * 32768
    assert completion_rejection(route, docs) == "completion_invalid"
    nested = {}
    for _ in range(9):
        nested = {"child": nested}
    docs["completion"]["processing_metadata"]["diagnostic"] = nested
    assert completion_rejection(route, docs) == "completion_invalid"


def test_opaque_or_fully_transparent_png_overlay_is_not_rejected_as_bad_native_alpha():
    sample = base64.b64decode(GEOMETRY["media_samples"]["png_rgba"])
    for alpha in (0, 255):
        with Image.open(io.BytesIO(sample)) as image:
            image = image.convert("RGBA")
            image.putalpha(alpha)
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
        with Image.open(io.BytesIO(encoded.getvalue())) as decoded:
            decoded.load()
            assert decoded.format == "PNG" and decoded.getchannel("A").getextrema() == (alpha, alpha)


@pytest.mark.parametrize("case", RECOVERY["upload_replays"], ids=lambda case: case["id"])
def test_replay_identity_hash_and_lease(case):
    original = RECOVERY["identity"]
    replay = {**original, **case["changes"]}
    result = "reject_unauthorized" if not case["authorized"] else "reject_expired_lease" if not case["lease_current"] else "artifact_conflict" if replay != original else "return_existing_artifact_identity"
    assert result == case["expected"]


@pytest.mark.parametrize("case", RECOVERY["session_recovery"], ids=lambda case: case["id"])
def test_fresh_session_rechecks_access_without_reviving_grant(case):
    allowed = all(case[k] is True for k in ("fresh_authorization", "project_access", "file_access", "original_identity_matches"))
    assert ("authorized_existing_result_lookup_no_old_grant_revival" if allowed else "recovery_access_denied") == case["expected"]


@pytest.mark.parametrize("case", RECOVERY["cases"], ids=lambda case: case["id"])
def test_lost_response_reconciliation_requirements(case):
    assert case["rerender"] is case["new_parent"] is False
    if case["event"] in {"upload_timeout", "completion_timeout", "crash_after_output_creation"}:
        outcomes = {
            "unavailable": "uncertain", "artifact_saved": "artifact_saved_parent_not_proven_complete",
            "parent_complete_receipt_pending": "reconcile_receipt_without_rerender",
            "artifact_saved_parent_failed": "artifact_saved_explicit_parent_recovery",
            "complete_with_receipt": "return_existing_authorized_receipt",
        }
        assert case["reconcile"] is True and case["expected"] == outcomes[case["lookup"]]
    elif case["event"] in {"upload_attempt", "completion_attempt"}:
        assert case["lease_current"] is False and case["expected"] == "reject_expired_lease"
    else:
        assert case["expected"] == ("canceled_no_successful_partial_delivery" if case["process_exit_verified"] else "stop_requested_execution_state_unverified")
    assert RECOVERY["policy"]["exactly_once_end_to_end_established"] is False
    assert RECOVERY["policy"]["automatic_render_retry"] is False
    assert RECOVERY["policy"]["video_hosted_fallback"] is False
