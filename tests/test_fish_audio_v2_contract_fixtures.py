"""Immutable contract-freeze examples and policy oracles, mirrored by Midom.

The manifest records the original fixture-only stage. Runtime coverage lives in
test_fish_audio_reviewable_runtime.py; frozen JSON files remain byte-identical.
"""

import copy
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from test_fish_audio_drama3 import load_plugin


FIXTURES = Path(__file__).parent / "fixtures" / "fish_audio_v2"
CAPABILITIES, JOBS, MEASUREMENTS, REJECTIONS = [
    json.loads((FIXTURES / name).read_text())
    for name in ("capabilities.json", "jobs.json", "measurements.json", "rejections.json")
]
V1, V2 = "fish_drama3_tts_v1", "fish_drama3_tts_v2"
POLICY, SAFETY = "reviewable_audio_v1", "fish_audio_180s_50mib_v1"


def test_mirrored_fixture_manifest_and_identifiers():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert manifest["runtime_implementation"] is manifest["runtime_advertisement"] is False
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest
        assert json.loads((FIXTURES / name).read_text())["fixture_set_id"] == manifest["fixture_set_id"]


def json_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def depth(value):
    children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return 1 + max((depth(child) for child in children), default=0)


def longest_string(value):
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return max((longest_string(item) for pair in value.items() for item in pair), default=0)
    if isinstance(value, list):
        return max(map(longest_string, value), default=0)
    return 0


def metadata_rejection(metadata):
    if json_bytes(metadata) > 32768:
        return "metadata_too_large"
    if depth(metadata) > 8:
        return "metadata_too_deep"
    if longest_string(metadata) > 8192:
        return "metadata_overlong_string"
    return None


def measurement_expectation(value):
    """Independent test oracle for recorded measurements, not a media decoder."""
    if type(value["generation_ceiling_seconds"]) is not int or not 1 <= value["generation_ceiling_seconds"] <= 120:
        return "generation_ceiling_invalid"
    if value["cancelled"] or value["finish_reason"] == "early_stop":
        return "cancel_requested"
    if not value["job_association_valid"] or value["source_echo"]:
        return "artifact_association_invalid"
    if (value["session_succeeded"] is not True or value["native_call_returned_output"] is not True
            or type(value["observed_calls"]) is not int or value["observed_calls"] != 1):
        return "execution_evidence_invalid"
    if value["finish_reason"] not in {"eos", "silence", "length", "unknown"}:
        return "termination_diagnostics_invalid"
    if type(value["bytes"]) is not int or type(value["decoded_mono_samples"]) is not int:
        return "artifact_measurements_invalid"
    if value["bytes"] > 52428800:
        return "artifact_byte_safety_exceeded"
    if value["decoded_mono_samples"] > 7938000:
        return "artifact_sample_safety_exceeded"
    peak = Decimal(value["peak_amplitude"])
    if (value["bytes"] <= 0 or value["decoded_mono_samples"] <= 0
            or not value["decoded_all_samples_finite"] or not peak.is_finite() or peak <= 0):
        return "artifact_unusable"
    if (any(type(value[key]) is not int for key in ("sample_rate_hz", "channels", "decoded_sample_rate_hz", "decoded_channels"))
            or value["sample_rate_hz"] != 44100 or value["channels"] != 1
            or value["decoded_sample_rate_hz"] != 44100 or value["decoded_channels"] != 1
            or value["probe_audio_streams"] != 1 or value["probe_video_streams"] != 0):
        return "artifact_geometry_invalid"
    fmt = value["requested_format"]
    if (not value["genuine_container"] or value["mime_type"] != {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(fmt)
            or not (fmt == "wav" and value["codec_name"].startswith("pcm_") or fmt == "mp3" and value["codec_name"] == "mp3")):
        return "artifact_format_invalid"
    if not value["decode_complete"] or value["decode_timed_out"]:
        return "artifact_decode_failed"
    try:
        probed = [Decimal(item) for item in (value["probe_stream_duration"], value["probe_format_duration"]) if item not in (None, "N/A")]
    except InvalidOperation:
        return "artifact_probe_invalid"
    if not probed or any(not item.is_finite() or item <= 0 for item in probed):
        return "artifact_probe_invalid"
    measured = max(*probed, Decimal(value["decoded_mono_samples"]) / Decimal(44100))
    if measured > 180:
        return "artifact_duration_safety_exceeded"
    warnings = {"eos": [], "silence": ["speech_stopped_on_silence"],
                "length": ["generation_duration_limit_reached"], "unknown": ["termination_unverified"]}[value["finish_reason"]]
    if measured > Decimal(value["generation_ceiling_seconds"]) + Decimal("0.25"):
        warnings.append("generated_duration_over_budget")
    return warnings


def documents():
    return {"capability": copy.deepcopy(CAPABILITIES["models"][0]),
            **{key: copy.deepcopy(JOBS[key]) for key in ("candidate", "claim", "completion")}}


def mutate(document, case):
    parent = document
    for segment in case["path"][:-1]:
        parent = parent[segment]
    if case.get("delete"):
        del parent[case["path"][-1]]
    else:
        parent[case["path"][-1]] = copy.deepcopy(case["value"])


def selection_rejection(docs):
    cap, candidate, claim = (docs[key] for key in ("capability", "candidate", "claim"))
    summary, generation = candidate["summary"], claim["generation"]
    if candidate["model_id"] != cap["model_id"] or claim["model_id"] != cap["model_id"]:
        return "checkpoint_unavailable"
    version = summary.get("speech_contract_version")
    record = cap["limits"]["fish_audio_contracts"].get(version)
    if not record or record.get("contract_version") != version:
        return "contract_unavailable"
    if generation.get("speech_contract_version") != version:
        return "candidate_claim_mismatch"
    for fields in (summary, generation):
        if fields.get("acceptance_policy_id") != record.get("acceptance_policy_id") or not fields.get("acceptance_policy_id"):
            return "acceptance_policy_unavailable"
        if fields.get("artifact_safety_profile_id") != record.get("artifact_safety_profile_id") or not record.get("artifact_safety"):
            return "safety_profile_unavailable"
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(claim["output"]["format"])
    if mime not in record["output_mime_types"] or mime not in cap["limits"]["output_mime_types"]:
        return "format_unavailable"
    if "max_duration_seconds" in claim["output"]:
        return "ambiguous_v1_output_limit"
    if (set(claim["limits"]) - {"max_artifact_bytes"}
            or claim["limits"]["max_artifact_bytes"] > record["artifact_safety"]["max_bytes"]):
        return "safety_override_forbidden"
    allowed = {"audio_task", "speech_contract_version", "speech_profile_id", "acceptance_policy_id", "artifact_safety_profile_id",
               "voice_mode", "dialogue_mode", "duration_mode", "duration_seconds", "seed", "prompt_enhancement_by_worker", "options"}
    if set(generation) - allowed:
        return "unsupported_generation_field"
    if generation["prompt_enhancement_by_worker"] is not False or generation["options"] != {"multi_prompts_gen_type": "FG"}:
        return "unsupported_prompt_handling"
    refs = claim["inputs"]
    if summary["reference_transcript_count"] != len(refs):
        return "reference_mapping_invalid"
    for index, ref in enumerate(refs):
        if (ref["speaker_id"] != index + 1 or ref["sequence"] != index or not ref.get("transcript")
                or ref["role"] != "speaker_reference_audio" or ref["kind"] != f"speaker{index + 1}_reference_audio"):
            return "reference_mapping_invalid"
    return None


def completion_rejection(docs):
    claim, complete = docs["claim"], docs["completion"]
    fish = complete["generation_metadata"]["fish_audio"]
    if (complete["model_id"] != claim["model_id"] or fish["model_id"] != claim["model_id"]
            or fish["script_sha256"] != hashlib.sha256(claim["prompt"].encode()).hexdigest() or any(
        fish.get(key) != claim["generation"].get(key)
        for key in ("speech_contract_version", "speech_profile_id", "acceptance_policy_id", "artifact_safety_profile_id")
    )):
        return "completion_contract_mismatch"
    refs = complete["artifacts"]
    accepted = JOBS["accepted_artifact"]
    if (len(refs) != 1 or any(refs[0].get(key) != accepted[key] for key in ("artifact_id", "file_id", "artifact_index"))
            or fish["artifacts"][0]["sha256"] != accepted["sha256"]):
        return "artifact_association_invalid"
    evidence = fish.get("execution_evidence", {})
    if evidence.get("cancelled") is True or fish.get("finish_reason") == "early_stop":
        return "cancel_requested"
    if (evidence.get("job_id") != claim["job_id"] or not evidence.get("client_id")
            or evidence.get("session_succeeded") is not True or evidence.get("cancelled") is not False
            or evidence.get("native_call_returned_output") is not True
            or type(evidence.get("observed_calls")) is not int or evidence.get("observed_calls") != 1
            or type(fish.get("observed_calls")) is not int or fish.get("observed_calls") != 1):
        return "execution_evidence_invalid"
    reason, parts = fish.get("finish_reason"), fish.get("parts")
    if (not isinstance(parts, list) or reason not in {"eos", "silence", "length", "unknown"}
            or len(parts) != (0 if reason == "unknown" else 1)
            or any(part["finish_reason"] != reason for part in parts)):
        return "termination_diagnostics_invalid"
    codes = {"eos": [], "silence": ["speech_stopped_on_silence"], "length": ["generation_duration_limit_reached"],
             "unknown": ["termination_unverified"]}[reason] + ["generated_duration_over_budget"]
    warnings = [{"code": code, "severity": "warning", "artifact_index": 0} for code in codes]
    if fish["warnings"] != warnings:
        return "warning_mismatch"
    if fish["script_coverage"] != "not_verified" or "editorial_review_status" in fish:
        return "unsupported_content_claim"
    return metadata_rejection(complete["generation_metadata"])


@pytest.mark.parametrize("model", CAPABILITIES["models"], ids=lambda item: item["model_id"])
def test_additive_records_preserve_exact_v1_capability(model):
    module = load_plugin()
    plugin = module.AwsWorkerBridgePlugin()
    plugin._audio_output_mime_types = lambda: ["audio/wav", "audio/mpeg"]
    current = next(item for item in plugin._fish_audio_capabilities() if item["model_id"] == model["model_id"])
    legacy = copy.deepcopy(model)
    contracts = legacy["limits"].pop("fish_audio_contracts")
    historical = copy.deepcopy(current)
    historical["limits"]["fish_audio_contracts"].pop("fish_drama3_tts_v3")
    assert historical == model
    current_legacy = copy.deepcopy(current)
    current_legacy["limits"].pop("fish_audio_contracts")
    assert legacy == current_legacy
    assert set(contracts) == {V1, V2}
    for key, value in model["limits"]["fish_audio"].items():
        assert contracts[V1][key] == value
    for version, record in contracts.items():
        assert record["contract_version"] == version
        assert record["recipe_version"] == "fish_drama3_standard_recipe_v1"
        assert record["output_mime_types"] == model["limits"]["output_mime_types"]
        assert record["max_reference_audio_files"] == record["max_cloned_speakers"] == 3
        assert record["max_generation_duration_seconds"] == 120
    assert contracts[V2]["artifact_safety"] == {"max_duration_seconds": 180, "max_bytes": 52428800,
        "max_decoded_mono_samples": 7938000, "duration_basis": "probe_and_decoded_duration_v1"}
    assert "acceptance_policy_id" not in contracts[V1]
    assert {version: current["limits"]["fish_audio_contracts"][version] for version in (V1, V2)} == contracts


@pytest.mark.parametrize("model", JOBS["checkpoint_matrix"])
@pytest.mark.parametrize("fmt", JOBS["format_matrix"], ids=lambda item: item["format"])
@pytest.mark.parametrize("route", JOBS["route_matrix"], ids=lambda item: f"{item['voice_mode']}-{item['reference_count']}-{item['speaker_count']}")
def test_candidate_claim_fixture_routes_are_consistent(model, fmt, route):
    docs = documents()
    cap = next(item for item in CAPABILITIES["models"] if item["model_id"] == model)
    docs["capability"] = cap
    candidate, claim = docs["candidate"], docs["claim"]
    candidate["model_id"] = claim["model_id"] = model
    generation, summary = claim["generation"], candidate["summary"]
    generation["voice_mode"] = summary["voice_mode"] = route["voice_mode"]
    claim["inputs"] = claim["inputs"][:route["reference_count"]]
    claim["prompt"] = "\n".join(f"Speaker {index}: Hello." for index in range(1, route["speaker_count"] + 1))
    if route["voice_mode"] != "multi_speaker_dialogue":
        generation.pop("dialogue_mode")
        summary.pop("dialogue_mode")
    summary.update(speaker_ids=list(range(1, route["speaker_count"] + 1)), speaker_count=route["speaker_count"],
                   referenced_speaker_ids=list(range(1, route["reference_count"] + 1)), output_format=fmt["format"])
    for field in ("speaker_reference_audio_count", "reference_transcript_count", "input_audio_count"):
        summary[field] = route["reference_count"]
    claim["output"].update(format=fmt["format"], mime_type=fmt["mime_type"])
    assert selection_rejection(docs) is None
    assert "max_duration_seconds" not in claim["output"]
    for key, value in generation.items():
        if key not in {"seed", "options", "prompt_enhancement_by_worker"}:
            assert summary[key] == value


def test_canonical_completion_binding_and_whole_metadata_limits():
    docs = documents()
    assert selection_rejection(docs) is None
    assert completion_rejection(docs) is None
    metadata = docs["completion"]["generation_metadata"]
    assert depth(metadata) == 7
    assert json_bytes(metadata) <= 32768
    assert longest_string(metadata) <= 8192
    assert JOBS["expected_midom_state"]["job_status"] == "succeeded"
    assert JOBS["expected_midom_state"]["editorial_review_status"] == "pending_review"
    for reference, provenance in zip(JOBS["claim"]["inputs"], metadata["fish_audio"]["reference_inputs"]):
        assert hashlib.sha256(reference["transcript"].encode()).hexdigest() == provenance["transcript_sha256"]
    assert hashlib.sha256(JOBS["claim"]["prompt"].encode()).hexdigest() == metadata["fish_audio"]["script_sha256"]


@pytest.mark.parametrize("termination", JOBS["termination_matrix"], ids=lambda item: item["finish_reason"])
def test_reviewable_completion_examples_for_each_termination(termination):
    docs = documents()
    fish = docs["completion"]["generation_metadata"]["fish_audio"]
    fish.update(finish_reason=termination["finish_reason"], parts=termination["parts"])
    fish["warnings"] = [{"code": code, "severity": "warning", "artifact_index": 0}
                        for code in termination["warnings"] + ["generated_duration_over_budget"]]
    assert completion_rejection(docs) is None


def test_wav_only_worker_has_no_implicit_mp3_route():
    docs = documents()
    limits = docs["capability"]["limits"]
    limits["output_mime_types"] = ["audio/wav"]
    for record in limits["fish_audio_contracts"].values():
        record["output_mime_types"] = ["audio/wav"]
    assert selection_rejection(docs) == "format_unavailable"
    docs["claim"]["output"].update(format="wav", mime_type="audio/wav")
    docs["candidate"]["summary"]["output_format"] = "wav"
    assert selection_rejection(docs) is None


@pytest.mark.parametrize("model", JOBS["checkpoint_matrix"])
@pytest.mark.parametrize("case", MEASUREMENTS["cases"], ids=lambda item: item["id"])
def test_measurement_and_termination_expectations(model, case):
    assert model in {record["model_id"] for record in CAPABILITIES["models"]}
    result = measurement_expectation({**MEASUREMENTS["base"], **case["set"]})
    assert result == (case["warnings"] if case["accept"] else case["reason"])


@pytest.mark.parametrize("case", REJECTIONS["selection_and_claim_cases"], ids=lambda item: item["id"])
def test_selection_and_claim_rejection_examples(case):
    docs = documents()
    mutate(docs[case["target"]], case)
    assert selection_rejection(docs) == case["reason"]


@pytest.mark.parametrize("case", REJECTIONS["completion_cases"], ids=lambda item: item["id"])
def test_completion_rejection_examples(case):
    docs = documents()
    mutate(docs["completion"], case)
    assert completion_rejection(docs) == case["reason"]


@pytest.mark.parametrize("case", REJECTIONS["metadata_cases"], ids=lambda item: item["id"])
def test_entire_metadata_envelope_boundaries(case):
    metadata = copy.deepcopy(JOBS["completion"]["generation_metadata"])
    construction = case["construction"]
    if construction == "non_fish_depth":
        nested = "x"
        for _ in range(case["value"] - 2):
            nested = {"level": nested}
        metadata["fixture_depth"] = nested
        assert depth(metadata) == case["value"]
    elif construction == "old_conditioning_shape":
        assembly = metadata["fish_audio"]["native_conditioning"]["prompt_assemblies"][0]
        assembly["reference_code_fingerprints"] = [{"shape": [10, 100], "sha256_int64": "7" * 64}]
        assert depth(metadata) == 9
    elif construction == "whole_utf8_bytes":
        padding = metadata["fixture_padding"] = []
        while json_bytes(metadata) < case["value"]:
            padding.append("")
            padding[-1] = "x" * min(7000, case["value"] - json_bytes(metadata))
        assert json_bytes(metadata) == case["value"]
        assert longest_string(metadata) <= 8192
    elif construction == "non_fish_string_chars":
        metadata["fixture_string"] = "x" * case["value"]
    elif construction == "non_fish_multibyte_strings":
        metadata["fixture_strings"] = [chr(0xE9) * case["value"]] * 3
        assert longest_string(metadata) == 8192
    else:
        pytest.fail(f"Unknown fixture construction: {construction}")
    assert metadata_rejection(metadata) == (None if case["accept"] else case["reason"])


def test_recovery_remains_separate_from_v2_generation():
    recovery = REJECTIONS["recovery_expectations"]
    assert recovery["uploaded_artifact"]["source_job_unchanged"]
    assert recovery["uploaded_artifact"]["source_contract_unchanged"]
    assert recovery["uploaded_artifact"]["missing_historical_execution_evidence"] == "unknown"
    assert recovery["worker_assisted"]["payload_status"] == "deferred_separate_freeze_required"
    assert all(not item["new_inference"] and not item["new_inference_charge"] for item in recovery.values())
