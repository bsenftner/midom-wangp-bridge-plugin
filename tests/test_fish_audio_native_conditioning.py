import ast
import gc
import hashlib
import math
import os
import re
import sys
import types
from contextlib import ExitStack, nullcontext
from functools import wraps
from pathlib import Path
from typing import Sequence

import pytest

from test_fish_audio_drama3 import Runtime, Session, fish_job, load_plugin, wav_bytes


class Tokenizer:
    def __init__(self):
        self.ids = {"<|im_end|>": 1, **{f"<|speaker:{index}|>": 100 + index for index in range(16)}}

    def token_to_id(self, token):
        match = re.fullmatch(r"<\|semantic:([0-9]+)\|>", token)
        return 10000 + int(match[1]) if match else self.ids.get(token)

    def encode(self, text):
        result = []
        for token in re.findall(r"<\|[^>]*\|>|.", text, re.DOTALL):
            if token not in self.ids:
                self.ids[token] = 1000 + len(self.ids)
            result.append(self.ids[token])
        return types.SimpleNamespace(ids=result)


@pytest.fixture
def native(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    module = load_plugin()
    plugin = module.AwsWorkerBridgePlugin()
    logs = []
    plugin._log = lambda text, **kwargs: logs.append(text)
    plugin._audio_output_mime_types = lambda: ["audio/wav"]
    plugin._ensure_job_flow_enabled = lambda: None
    root = Path(os.environ.get("WAN2GP_ROOT") or Path(__file__).resolve().parents[3])
    source = root / "models/TTS/fish_audio"
    if not source.is_dir():
        pytest.skip("Native conditioning checks require the WanGP checkout")
    namespace = {"re": re, "torch": torch, "Sequence": Sequence, "Tokenizer": object, "FishOmniConfig": object,
                 "SPEAKER_LINE_PATTERN": re.compile(r"^([ \t]*)Speaker[ \t]*(\d+)[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)}
    tree = ast.parse((source / "prompt.py").read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(source / "prompt.py"), "exec"), namespace)
    package = "_midom_fish_native_conditioning_test"
    for name in ("codec", "codec_rvq"):
        stub = types.ModuleType(f"{package}.{name}")
        for cls in ("TransformerBlock", "EncoderBlock", "DecoderBlock", "ConvNeXtBlock"):
            setattr(stub, cls, type(cls, (), {}))
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    namespace.update(__package__=package, math=math, gc=gc, ExitStack=ExitStack,
                     runtime=types.SimpleNamespace(execution_scope=lambda *args: nullcontext()),
                     DIALOGUE_CUT_LINES={"id": "dialogue_cut_lines", "default": 0},
                     DIALOGUE_HISTORY_SECONDS={"id": "dialogue_history_seconds", "default": 20},
                     GenerationCancelled=type("GenerationCancelled", (Exception,), {}))
    tree = ast.parse((source / "pipeline.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FishAudioPipeline")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {"generate", "_reference_transcripts", "_history"}]
    for node in cls.body:
        node.decorator_list = []
    # Execute native routing and prompt construction; only codec/model inference is stubbed.
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source / "pipeline.py"), "exec"), namespace)
    pipeline = namespace["FishAudioPipeline"]()
    pipeline.engine = "vllm"
    pipeline.sample_rate, pipeline.frame_length = 44100, 2048
    pipeline.vae = types.SimpleNamespace(modules=lambda: [])
    pipeline.config = types.SimpleNamespace(audio_decoder=types.SimpleNamespace(num_codebooks=2, vocab_size=256),
                                            semantic_start_token_id=10000, semantic_end_token_id=10255, eos_token_id=1)
    pipeline.tokenizer = Tokenizer()
    pipeline._check_abort = lambda *args: None
    pipeline._release_decoder_caches = lambda: None
    pipeline._decode = lambda *args: torch.ones((1, 2048))
    pipeline._generate_codes = lambda *args: (torch.ones((2, 20), dtype=torch.long), "eos")
    pipeline._reference_codes = lambda path, *args: torch.arange(6).view(2, 3) + int(Path(path).stem.rsplit("-", 1)[-1]) * 10
    fixture = types.SimpleNamespace(module=module, plugin=plugin, pipeline=pipeline, namespace=namespace,
                                    torch=torch, tmp_path=tmp_path, logs=logs)
    return fixture


def prepare(f, count, *, model="fish_drama3", speakers=None, contract_version="fish_drama3_tts_v1"):
    job = fish_job(count, model=model, speakers=speakers)
    if contract_version != "fish_drama3_tts_v1":
        extended = contract_version == "fish_drama3_tts_v3"
        job["generation"].update(speech_contract_version=contract_version,
                                 acceptance_policy_id="reviewable_audio_v1",
                                 artifact_safety_profile_id="fish_audio_240s_50mib_v1" if extended else "fish_audio_180s_50mib_v1",
                                 duration_seconds=180 if extended else 60)
        job["output"].pop("max_duration_seconds")
    settings = f.plugin._validate_fish_audio_job(job, 123)
    downloaded = []
    for index, descriptor in enumerate(job["inputs"]):
        path = f.tmp_path / f"reference-{index + 1}.wav"
        path.write_bytes(wav_bytes(3.1 + index * 0.1))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        downloaded.append({**descriptor, "path": str(path), "sha256": digest, "prepared_sha256": digest,
                           "preparation": "mono_44100hz_float32_wav_v1", "metadata": {}})
    f.plugin._apply_fish_audio_inputs(settings, downloaded)
    runtime = Runtime(f.pipeline)
    return settings, runtime, Session(runtime)


@pytest.mark.parametrize("model", ["fish_drama3", "fish_drama3_int4"])
@pytest.mark.parametrize("refs", [0, 1, 2, 3])
@pytest.mark.parametrize("contract_version", ["fish_drama3_tts_v2", "fish_drama3_tts_v3"])
def test_reviewable_observes_real_native_return_and_flat_conditioning(native, model, refs, contract_version):
    f = native
    settings, runtime, session = prepare(f, refs, model=model, contract_version=contract_version)
    output = f.tmp_path / "generated.wav"
    runtime.on_generated = lambda: output.write_bytes(wav_bytes(0.1))
    session.files = [str(output)]
    with f.plugin._fish_audio_execution(session, settings):
        result = f.plugin._submit_wangp_job(session, settings, 1, None).result()
    f.plugin._validate_fish_audio_completion(settings, [str(output)], result)
    metadata = settings["_midom_fish_audio"]
    assert metadata["execution_evidence"]["native_call_returned_output"] is True
    assert metadata["execution_evidence"]["session_succeeded"] is True
    assert metadata["native_conditioning"]["verification_result"] == "prompt_assembly_verified"
    assert metadata["native_conditioning"]["reference_count"] == refs
    f.plugin._validate_fish_audio_metadata_envelope({"fish_audio": metadata})


@pytest.mark.parametrize("model", ["fish_drama3", "fish_drama3_int4"])
@pytest.mark.parametrize("refs,speakers", [(0, 1), (1, 1), (2, 2), (3, 3), (3, 16)])
def test_real_native_routing_and_prompt_contain_every_reference(native, model, refs, speakers):
    f = native
    settings, runtime, session = prepare(f, refs, model=model, speakers=speakers)
    previous_encode = f.pipeline._reference_codes
    previous_build = f.namespace["build_drama3_prompt"]
    with f.plugin._fish_audio_execution(session, settings):
        f.plugin._submit_wangp_job(session, settings, 1, None)
    trace = settings["_midom_fish_audio"]["native_conditioning"]
    assert trace["verification_result"] == "prompt_assembly_verified"
    assert trace["reference_count"] == refs
    assert trace["audio_prompt_type"] == ("", "A", "AB", "ABD")[refs]
    assert [item["slot"] for item in trace["references"]] == ["audio_guide", "audio_guide2", "audio_guide3"][:refs]
    assembly = trace["prompt_assemblies"][0]
    assert assembly["native_speaker_token_order"] == list(range(refs)) + list(range(speakers))
    assert assembly["reference_audio_rows"] == refs * 3
    assert assembly["distinct_encoded_reference_count"] == refs
    assert len(set(item["prepared_sha256"] for item in trace["references"])) == refs
    assert [item["filename"] for item in trace["native_sources"]] == ["pipeline.py", "prompt.py"]
    assert all(len(item["sha256"]) == 64 for item in trace["native_sources"])
    f.plugin._validate_fish_audio_completion(settings, ["result.wav"])
    assert f.pipeline._reference_codes is previous_encode
    assert f.namespace["build_drama3_prompt"] is previous_build
    assert runtime.server_config["audio_stand_alone_output_codec"] == "mp3"
    assert not any("Welcome to the neighborhood" in text or "Reference recording number" in text for text in f.logs)


@pytest.mark.parametrize("mutation", ["mode", "missing_second", "missing_third", "swapped", "bytes"])
def test_native_boundary_rejects_lost_or_changed_reference(native, mutation):
    f = native
    settings, runtime, session = prepare(f, 3)
    if mutation == "bytes":
        Path(settings["audio_guide2"]).write_bytes(b"changed reference")
    original = runtime.generate_media

    @wraps(original)
    def changed(*args, **kwargs):
        if mutation == "mode":
            kwargs["audio_prompt_type"] = "A"
        elif mutation == "missing_second":
            kwargs["audio_guide2"] = None
        elif mutation == "missing_third":
            kwargs["audio_guide3"] = None
        elif mutation == "swapped":
            kwargs["audio_guide"], kwargs["audio_guide2"] = kwargs["audio_guide2"], kwargs["audio_guide"]
        return original(*args, **kwargs)

    runtime.generate_media = changed
    with f.plugin._fish_audio_execution(session, settings):
        with pytest.raises(ValueError, match="native reference"):
            f.plugin._submit_wangp_job(session, settings, 1, None)
    assert runtime.server_config["audio_stand_alone_output_codec"] == "mp3"
    assert settings["_midom_fish_audio"].get("finish_reason") is None


@pytest.mark.parametrize("fault", ["drop_reference", "reuse_codes", "speaker_token"])
def test_observer_checks_actual_assembled_prompt_not_only_input_arguments(native, fault):
    f = native
    settings, _, session = prepare(f, 2)
    original = f.namespace["build_drama3_prompt"]

    @wraps(original)
    def broken(*args, **kwargs):
        prompt = original(*args, **kwargs)
        rows = (prompt[:, 0] >= 10000).nonzero().flatten()
        if fault == "drop_reference":
            prompt[rows[3:]] = 0
        elif fault == "reuse_codes":
            prompt[rows[3:]] = prompt[rows[:3]].clone()
        else:
            prompt[prompt[:, 0] == 101, 0] = 100
        return prompt

    f.namespace["build_drama3_prompt"] = broken
    previous_encode, previous_generate = f.pipeline._reference_codes, f.pipeline._generate_codes
    with f.plugin._fish_audio_execution(session, settings):
        with pytest.raises(ValueError, match="assembled prompt"):
            f.plugin._submit_wangp_job(session, settings, 1, None)
    assert f.namespace["build_drama3_prompt"] is broken
    assert f.pipeline._reference_codes is previous_encode
    assert f.pipeline._generate_codes is previous_generate


def test_identical_codec_results_are_reported_not_misrepresented_as_distinct_voices(native):
    f = native
    settings, _, session = prepare(f, 2)
    f.pipeline._reference_codes = lambda *args: f.torch.ones((2, 3), dtype=f.torch.long)
    with f.plugin._fish_audio_execution(session, settings):
        f.plugin._submit_wangp_job(session, settings, 1, None)
    trace = settings["_midom_fish_audio"]["native_conditioning"]
    assert len({item["prepared_sha256"] for item in trace["references"]}) == 2
    assert trace["prompt_assemblies"][0]["distinct_encoded_reference_count"] == 1


def test_transcript_parser_cannot_silently_reuse_first_reference(native):
    f = native
    settings, _, session = prepare(f, 2)
    original = f.namespace["reference_transcripts"]
    f.namespace["reference_transcripts"] = lambda text, count: original(text, count)[:1] * count
    with f.plugin._fish_audio_execution(session, settings):
        with pytest.raises(ValueError, match="transcript parsing"):
            f.plugin._submit_wangp_job(session, settings, 1, None)


def test_multiline_transcripts_and_initial_unlabelled_speech_keep_correct_tags(native):
    f = native
    job = fish_job(2)
    job["prompt"] = "Opening speech from the first speaker.\nSpeaker 2: Answer from the second.\nSpeaker 1: Closing reply."
    settings = f.plugin._validate_fish_audio_job(job, 123)
    _, runtime, session = prepare(f, 2)
    downloaded = []
    for index, descriptor in enumerate(job["inputs"]):
        transcript = f" First line for {index + 1}.  \r\n   Continued words.  \r\n"
        path = f.tmp_path / f"reference-{index + 1}.wav"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        downloaded.append({**descriptor, "transcript": transcript, "path": str(path), "sha256": digest,
                           "prepared_sha256": digest, "preparation": "mono_44100hz_float32_wav_v1", "metadata": {}})
    f.plugin._apply_fish_audio_inputs(settings, downloaded)
    with f.plugin._fish_audio_execution(session, settings):
        f.plugin._submit_wangp_job(session, settings, 1, None)
    trace = settings["_midom_fish_audio"]["native_conditioning"]
    assert trace["prompt_assemblies"][0]["native_speaker_token_order"] == [0, 1, 0, 1, 0]
    assert runtime.server_config["audio_stand_alone_output_codec"] == "mp3"
