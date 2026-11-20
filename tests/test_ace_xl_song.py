"""ACE XL's separate song contract: native observation and real MP3 delivery."""

import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from test_fish_audio_drama3 import bridge, candidate_for
from test_job_progress_coalescing import progress, enqueue


ROOT = Path(os.environ.get("WAN2GP_ROOT") or Path(__file__).resolve().parents[3])
PLUGIN = Path(__file__).resolve().parents[1]
FIXTURES = PLUGIN / "tests/fixtures/ace_step_xl_song_v1"


def song_job():
    return json.loads((FIXTURES / "claim.json").read_text())


def song_candidate(plugin, job):
    candidate = candidate_for(plugin, {**job, "prompt": "", "inputs": []})
    candidate["summary"] = {
        **{key: value for key, value in job["generation"].items() if key not in ("seed", "options")},
        "prompt_processing_mode": "FG", "output_count": 1, "output_format": "mp3",
        "input_audio_count": 0, "input_image_count": 0, "input_video_count": 0,
    }
    return candidate


@pytest.fixture
def song_bridge(bridge, monkeypatch):
    module, plugin = bridge
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    plugin._ffmpeg_binary = lambda: shutil.which("ffmpeg")
    # Load WanGP's actual FG parser without importing its UI/runtime dependencies.
    parser_path = ROOT / "shared/utils/prompt_parser.py"
    if not parser_path.is_file() or not (ROOT / "wgp.py").is_file():
        pytest.skip("ACE XL native integration checks require WAN2GP_ROOT or installation under Wan2GP/plugins")
    tree = ast.parse(parser_path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "split_prompt_units")
    namespace = {"ENHANCED_PROMPT_PREFIX": "!enhanced!\n"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(parser_path), "exec"), namespace)
    monkeypatch.setitem(sys.modules, "shared.utils.prompt_parser", types.SimpleNamespace(split_prompt_units=namespace["split_prompt_units"]))
    return module, plugin


def audio(frames=4800):
    samples = (0.3 * np.sin(np.arange(frames) * (2 * np.pi * 440 / 48000))).astype(np.float32)
    return np.stack((samples, samples))


def encode(path, samples=None, *, rate=48000, channels=2, fmt="mp3"):
    samples = audio() if samples is None else samples
    command = [shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(rate),
               "-ac", str(channels), "-i", "pipe:0"]
    if fmt == "mp3":
        command += ["-c:a", "libmp3lame", "-b:a", "320k"]
    elif fmt == "flac":
        command += ["-c:a", "flac"]
    else:
        command += ["-c:a", "pcm_f32le"]
    subprocess.run(command + ["-f", fmt, str(path)], input=samples.T.astype("<f4").tobytes(), check=True, capture_output=True)


class Tokenizer:
    backend_tokenizer = types.SimpleNamespace(to_str=lambda: "test tokenizer, not GPU evidence")

    def __init__(self, counts=(80, 40)):
        self.counts = iter(counts)
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [1] * next(self.counts)}


class Pipeline:
    enable_lm = False
    _early_stop = False
    _interrupt = False
    ace_step_transformer = type("AceStepConditionGenerationModel", (), {"__module__": "models.TTS.ace_step15.models.ace_step15_xl_hf"})()

    def __init__(self, counts=(80, 40)):
        self.pre_text_tokenizer = Tokenizer(counts)
        self.encoded = []

    def _encode_prompt(self, prompt, max_length=256, use_embed_tokens=False):
        self.encoded.append((prompt, max_length, use_embed_tokens))

    def generate(self, input_prompt, model_mode, audio_guide, *, alt_prompt, custom_settings, **kwargs):
        self._encode_prompt("assembled description: " + alt_prompt, max_length=256)
        self._encode_prompt("assembled lyrics: " + input_prompt, max_length=2048, use_embed_tokens=True)
        return {"x": audio()[None], "audio_sampling_rate": 48000,
                "overridden_inputs": {"custom_settings": custom_settings}}


class Session:
    def __init__(self, root, path, *, counts=(80, 40), codec="mp3"):
        self._root, self.path = root, path
        self.pipeline, self.submitted = Pipeline(counts), []
        self.definition = {"ace_step15_transformer_variant": "xl_turbo"}
        self.result = None
        self.config = {"audio_stand_alone_output_codec": codec}
        self.after_generate = lambda: None

        def call(model_def, label, fn, *args, **kwargs):
            return fn(*args, **kwargs)

        def generate_media(task, *, client_id, model_type, prompt, alt_prompt, **kwargs):
            from shared.utils.prompt_parser import split_prompt_units
            kwargs["sampling_steps"] = kwargs.pop("num_inference_steps")
            kwargs["shift"] = kwargs.pop("flow_shift")
            kwargs["guide_scale"] = kwargs.pop("guidance_scale")
            kwargs["alt_guide_scale"] = kwargs.pop("alt_guidance_scale")
            self.module.call_with_default_device(self.definition, model_type, self.pipeline.generate,
                                                 input_prompt=split_prompt_units(prompt, "FG")[0], alt_prompt=split_prompt_units(alt_prompt, "FG")[0], **kwargs)
            self.after_generate()
            return True

        self.module = types.SimpleNamespace(generate_media=generate_media, call_with_default_device=call,
                                            server_config=self.config, transformer_quantization="int8", WanGP_version="20.00")

    def _ensure_runtime(self):
        return types.SimpleNamespace(module=self.module)

    def submit_task(self, settings, callbacks=None):
        self.submitted.append(copy.deepcopy(settings))
        assert all(not key.startswith("_midom_") for key in settings)
        assert settings["_api"] == {"return_audio": True, "return_side_files": False}
        self.module.generate_media(None, **settings)
        self.result = types.SimpleNamespace(success=True, cancelled=False, errors=[], generated_files=[str(self.path.relative_to(self._root))],
                                            artifacts=(types.SimpleNamespace(path=str(self.path), client_id=settings["client_id"],
                                                                             media_type="audio", audio_tensor=audio(), audio_sampling_rate=48000),))
        return types.SimpleNamespace(result=lambda: self.result)


def run_song(plugin, module, monkeypatch, tmp_path, *, fmt="mp3", counts=(80, 40), modify=lambda session: None, job=None):
    path = tmp_path / ("native_seed424202." + fmt)
    encode(path, fmt=fmt)
    session = Session(tmp_path, path, counts=counts, codec=fmt)
    modify(session)
    plugin._get_generation_session = lambda: session
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()
    uploads = []

    def upload(url, **kwargs):
        content = kwargs["files"]["file"][1].read()
        assert kwargs["data"]["filename"] == "ace-song-2501.mp3"
        assert kwargs["data"]["mime_type"] == "audio/mpeg"
        assert hashlib.sha256(content).hexdigest() == kwargs["data"]["sha256"]
        uploads.append(content)
        return types.SimpleNamespace(status_code=200, json=lambda: {"artifact_id": 12, "file_id": 34, "artifact_index": 0})

    monkeypatch.setattr(module.requests, "post", upload)
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7")
    original_generate, original_call = session.module.generate_media, session.module.call_with_default_device
    prior_encode = vars(session.pipeline).get("_encode_prompt")
    plugin._run_job(connection, job or song_job())
    assert session.module.generate_media is original_generate and session.module.call_with_default_device is original_call
    assert vars(session.pipeline).get("_encode_prompt") is prior_encode
    assert session.config == {"audio_stand_alone_output_codec": fmt}
    return session, uploads, plugin._fish_updates[-1]


def test_discovery_matches_design_and_preserves_legacy(song_bridge):
    module, plugin = song_bridge
    assert plugin._ace_xl_song_capability() == json.loads((FIXTURES / "capability.json").read_text())
    assert module.MAX_AUDIO_BYTES == 52_428_800 and module.ACE_STEP15_MAX_DURATION_SECONDS == 120
    assert module.ACE_XL_MODEL_ID in module.ALLOWED_AUDIO_MODEL_TYPES
    plugin._ffprobe_binary = lambda: "/missing/ffprobe"
    assert plugin._ace_xl_song_capability() is None


@pytest.mark.parametrize("mode", ["supplied", "instrumental"])
@pytest.mark.parametrize("seconds", [5, 30, 360])
def test_actual_candidate_and_claim_agree(song_bridge, mode, seconds):
    _, plugin = song_bridge
    job = song_job()
    job["generation"].update(lyrics_mode=mode, duration_seconds=seconds)
    if mode == "instrumental":
        job["lyrics"] = " \n "
        job["generation"]["options"].pop("language")
    assert plugin._candidate_incompatibility_reason(song_candidate(plugin, job), None) is None
    settings = plugin._validate_job(job, None)
    assert settings["model_type"] == "ace_step_v1_5_xl"
    assert settings["duration_seconds"] == seconds and settings["num_inference_steps"] == 8
    assert settings["_midom_music_song"]["lyrics_coverage"] == "not_verified"
    assert settings["prompt"] == ("[Instrumental]" if mode == "instrumental" else job["lyrics"])


@pytest.mark.parametrize("key", ["contract_version", "native_preset", "recipe_version", "artifact_safety_profile_id",
                                  "audio_task", "duration_mode", "lyrics_mode", "prompt_enhancement_by_worker", "duration_seconds"])
def test_missing_selection_never_defaults(song_bridge, key):
    _, plugin = song_bridge
    job = song_job()
    del job["generation"][key]
    assert plugin._candidate_incompatibility_reason(song_candidate(plugin, job), None)
    with pytest.raises(ValueError):
        plugin._validate_job(job, None)


@pytest.mark.parametrize("path,value", [
    (("generation", "contract_version"), "fish_drama3_tts_v3"),
    (("generation", "native_preset"), "ace_step_v1_5_xl_with_lm"),
    (("generation", "recipe_version"), "other"), (("generation", "artifact_safety_profile_id"), "other"),
    (("generation", "duration_seconds"), True), (("generation", "duration_seconds"), 30.0),
    (("generation", "duration_seconds"), 4), (("generation", "duration_seconds"), 361),
    (("generation", "seed"), "42"), (("generation", "seed"), True), (("generation", "seed"), -2),
    (("generation", "options", "bpm"), 29), (("generation", "options", "bpm"), 301),
    (("generation", "options", "bpm"), True), (("generation", "options", "time_signature"), 5),
    (("generation", "options", "key"), "H major"), (("generation", "options", "language"), "EN"),
    (("generation", "options", "planner"), "4B"), (("generation", "options", "multi_prompts_gen_type"), "G"),
    (("generation", "num_inference_steps"), 4), (("generation", "prompt_enhancement_by_worker"), 0),
    (("output", "count"), True), (("output", "count"), 2), (("output", "format"), "wav"),
    (("output", "mime_type"), "audio/wav"), (("output", "max_duration_seconds"), 30),
    (("limits", "max_artifact_bytes"), 52_428_800), (("limits", "max_decoded_frames"), True),
    (("inputs",), [{"kind": "reference_audio"}]), (("inputs",), None),
    (("lyrics",), "[Instrumental]"), (("lyrics",), "[inst]"), (("lyrics",), " "), (("lyrics",), 23),
    (("lyrics",), "# silently lost words"), (("lyrics",), "Hello\n #!PROMPT!:hidden"),
    (("lyrics",), "[/window]bad"), (("lyrics",), "<|im_start|>bad"),
    (("music_description",), ""), (("music_description",), "a" * 4097),
    (("music_description",), "!enhanced!\nnew"), (("prompt",), "legacy prompt"),
])
def test_strict_claim_rejections(song_bridge, path, value):
    _, plugin = song_bridge
    job, target = song_job(), None
    target = job
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        plugin._validate_job(job, None)


@pytest.mark.parametrize("seed", [None, -1, 0, 2147483647])
def test_controls_and_original_text_hashes(song_bridge, seed):
    _, plugin = song_bridge
    job = song_job()
    job["lyrics"] = "  [Verse]\r\nBlake in Denver  \r\n"
    job["generation"]["seed"] = seed
    job["generation"]["options"] = {"multi_prompts_gen_type": "FG", "key": "d#minor", "bpm": None}
    settings = plugin._validate_audio_job(job, 2501)
    assert settings["custom_settings"] == {"bpm": 120, "keyscale": "D# minor", "timesignature": 2, "language": "unknown"}
    assert settings["prompt"] == job["lyrics"]
    assert settings["_midom_music_song"]["original_lyrics_sha256"] == hashlib.sha256(job["lyrics"].encode()).hexdigest()
    assert type(settings["seed"]) is int and 0 <= settings["seed"] <= 2147483647


@pytest.mark.parametrize("fmt", ["mp3", "wav", "flac"])
def test_claim_through_real_mp3_upload_completion(song_bridge, monkeypatch, tmp_path, fmt):
    module, plugin = song_bridge
    session, uploads, update = run_song(plugin, module, monkeypatch, tmp_path, fmt=fmt)
    assert update[0] == "complete", update
    assert len(uploads) == 1
    metadata = update[1]["generation_metadata"]
    song = metadata["music_song"]
    assert song["encoding_source"] == ("native_mp3" if fmt == "mp3" else "native_raw_audio")
    if fmt == "mp3":
        assert uploads[0] == session.path.read_bytes()
    assert song["decoded_frames"] == 4800
    assert song["decoded_duration_seconds"] == 0.1 and song["requested_duration_seconds"] == 30
    assert "warnings" not in song and song["lyrics_coverage"] == "not_verified"
    assert song["session_succeeded"] is True and song["observed_native_calls"] == 1
    assert song["artifact_sha256"] == hashlib.sha256(uploads[0]).hexdigest()
    plugin._validate_ace_xl_song_metadata(metadata)
    assert all(call == {"padding": False, "truncation": False} for call in session.pipeline.pre_text_tokenizer.calls)


@pytest.mark.parametrize("counts,success", [((256, 2048), True), ((257, 40), False), ((80, 2049), False)])
def test_loaded_tokenizer_boundary_interceptor(song_bridge, monkeypatch, tmp_path, counts, success):
    module, plugin = song_bridge
    session, uploads, update = run_song(plugin, module, monkeypatch, tmp_path, counts=counts)
    assert (update[0] == "complete") is success, update
    assert bool(uploads) is success
    if not success:
        assert "tokens" in update[1]["message"]


@pytest.mark.parametrize("failure", ["cancel", "stop", "failed", "errors", "wrong_client", "wrong_path", "two_files", "unrecognized_extra", "two_artifacts", "early_stop", "planner", "variant", "raw_nan", "raw_zero"])
def test_execution_and_association_failures_cannot_upload(song_bridge, monkeypatch, tmp_path, failure):
    module, plugin = song_bridge

    def modify(session):
        if failure == "early_stop":
            generate = session.pipeline.generate
            def interrupted(*args, **kwargs):
                output = generate(*args, **kwargs)
                session.pipeline._early_stop = True
                return output
            session.pipeline.generate = types.MethodType(lambda self, *args, **kwargs: interrupted(*args, **kwargs), session.pipeline)
        elif failure == "planner":
            session.pipeline.enable_lm = True
        elif failure == "variant":
            session.definition["ace_step15_transformer_variant"] = "other"
        else:
            submit = session.submit_task

            def broken_submit(settings, callbacks=None):
                handle = submit(settings, callbacks)
                result = session.result
                if failure == "cancel":
                    plugin._cancel_requested_by_midom = True
                elif failure == "stop":
                    plugin._stop_event.set()
                elif failure == "failed":
                    result.success = False
                elif failure == "errors":
                    result.errors = ["native decode failed"]
                elif failure == "wrong_client":
                    result.artifacts[0].client_id = "stale"
                elif failure == "wrong_path":
                    result.artifacts[0].path = str(tmp_path / "unrelated.wav")
                elif failure in ("two_files", "unrecognized_extra"):
                    other = tmp_path / ("other.ogg" if failure == "unrecognized_extra" else "other.wav")
                    other.write_bytes(session.path.read_bytes())
                    result.generated_files.append(str(other))
                elif failure == "two_artifacts":
                    result.artifacts += result.artifacts
                elif failure == "raw_nan":
                    result.artifacts[0].audio_tensor[:] = np.nan
                elif failure == "raw_zero":
                    result.artifacts[0].audio_tensor[:] = 0
                return handle
            session.submit_task = broken_submit

    _, uploads, update = run_song(plugin, module, monkeypatch, tmp_path, fmt="wav" if failure.startswith("raw_") else "mp3", modify=modify)
    assert update[0] == "fail", update
    assert not uploads


@pytest.mark.parametrize("case", ["mono", "rate", "fake", "zero", "corrupt", "over_frames"])
def test_real_mp3_validation_rejects_bad_media(song_bridge, tmp_path, case):
    _, plugin = song_bridge
    path = tmp_path / "song.mp3"
    encode(path, channels=1 if case == "mono" else 2, rate=44100 if case == "rate" else 48000,
           fmt="wav" if case == "fake" else "mp3", samples=np.zeros((2, 4800), dtype=np.float32) if case == "zero" else None)
    if case == "corrupt":
        path.write_bytes(b"ID3malformed")
    with pytest.raises(ValueError):
        plugin._inspect_ace_xl_song_mp3(path, maximum_frames=4799 if case == "over_frames" else 17376000)


def test_frame_limit_inclusive_and_finite_peak_above_unity(song_bridge, tmp_path):
    _, plugin = song_bridge
    path = tmp_path / "song.mp3"
    encode(path, samples=audio() * 4)
    result = plugin._inspect_ace_xl_song_mp3(path, maximum_frames=4800)
    assert result["decoded_frames"] == 4800
    assert result["peak_amplitude"] > 1
    with pytest.raises(ValueError, match="frame safety"):
        plugin._inspect_ace_xl_song_mp3(path, maximum_frames=4799)


def test_process_timeout_and_cancellation(song_bridge):
    _, plugin = song_bridge
    with pytest.raises(ValueError, match="timed out"):
        plugin._ace_xl_song_process([sys.executable, "-c", "import time; time.sleep(10)"], timeout_seconds=0.02)
    plugin._cancel_requested_by_midom = True
    with pytest.raises(ValueError, match="cancel_requested"):
        plugin._ace_xl_song_process([sys.executable, "-c", "print('should not start')"])


def test_native_signature_and_recipe(song_bridge):
    _, plugin = song_bridge
    tree = ast.parse((ROOT / "wgp.py").read_text())
    native = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_media")
    accepted = {argument.arg for argument in native.args.args + native.args.kwonlyargs}
    settings = plugin._validate_audio_job(song_job(), 2501)
    assert {key for key in settings if not key.startswith("_midom_") and key != "_api"}.issubset(accepted)
    assert settings["flow_shift"] == 1 and "shift" not in settings
    assert settings["model_mode"] == 0
    assert settings["activated_loras"] == [] and settings["prompt_enhancer"] == ""
    call = next(node for node in ast.walk(native) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "call_with_default_device")
    keyword_values = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keyword_values["sampling_steps"] == "num_inference_steps"
    assert keyword_values["input_prompt"] == "prompt" and keyword_values["alt_prompt"] == "current_alt_prompt"
    assert keyword_values["custom_settings"] == "custom_settings_for_model"
    assert keyword_values["shift"] == "flow_shift"
    assert keyword_values["guide_scale"] == "guidance_scale"
    assert keyword_values["alt_guide_scale"] == "alt_guidance_scale"


def test_whole_metadata_bounds(song_bridge):
    _, plugin = song_bridge
    for value in ({"outside_song": "a" * 8193}, {str(index): "x" * 8000 for index in range(5)}, {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": 1}}}}}}}}):
        with pytest.raises(ValueError):
            plugin._validate_ace_xl_song_metadata(value)


REJECTIONS = json.loads((FIXTURES / "rejections.json").read_text())["cases"]


def test_fixture_hashes_and_capability_match_implementation(song_bridge):
    _, plugin = song_bridge
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        assert hashlib.sha256((FIXTURES / filename).read_bytes()).hexdigest() == digest
    assert plugin._ace_xl_song_capability() == json.loads((FIXTURES / "capability.json").read_text())


@pytest.mark.parametrize("case", REJECTIONS, ids=lambda case: case["id"])
def test_serialized_fixture_rejections(song_bridge, case):
    _, plugin = song_bridge
    candidate = json.loads((FIXTURES / "candidate.json").read_text())
    claim = json.loads((FIXTURES / "claim.json").read_text())
    target = candidate if case["target"] == "candidate" else claim
    for key in case["path"][:-1]:
        target = target[key]
    if case.get("delete"):
        del target[case["path"][-1]]
    else:
        target[case["path"][-1]] = case["value"]
    if case["target"] == "candidate":
        assert plugin._ace_xl_song_candidate_incompatibility_reason(candidate["summary"])
    else:
        with pytest.raises(ValueError):
            plugin._validate_job(claim, None)


def test_real_artifact_byte_boundary_and_no_raw_repair(song_bridge, monkeypatch, tmp_path):
    module, plugin = song_bridge
    path = tmp_path / "song.mp3"
    encode(path)
    monkeypatch.setattr(module, "ACE_XL_MAX_BYTES", path.stat().st_size)
    assert plugin._inspect_ace_xl_song_mp3(path)["artifact_bytes"] == path.stat().st_size
    monkeypatch.setattr(module, "ACE_XL_MAX_BYTES", path.stat().st_size - 1)
    with pytest.raises(ValueError, match="byte safety"):
        plugin._inspect_ace_xl_song_mp3(path)
    monkeypatch.setattr(module, "ACE_XL_MAX_BYTES", 83886080)
    encode(path, fmt="wav")
    settings = plugin._validate_audio_job(song_job(), 2501)
    settings["_midom_ace_xl_artifact"] = types.SimpleNamespace(audio_tensor=audio(), audio_sampling_rate=48000)
    with pytest.raises(ValueError, match="purported native MP3"):
        plugin._prepare_ace_xl_song_mp3(str(path), settings, str(tmp_path))


@pytest.mark.parametrize("frames,accepted", [(17376000, True), (17376001, False)])
def test_real_mp3_full_safety_frame_boundary(song_bridge, tmp_path, frames, accepted):
    _, plugin = song_bridge
    path = tmp_path / "safety-boundary.mp3"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                    "-af", f"atrim=end_sample={frames}", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", str(path)],
                   check=True, timeout=90, capture_output=True)
    if accepted:
        measured = plugin._inspect_ace_xl_song_mp3(path)
        assert measured["decoded_frames"] == frames and measured["decoded_duration_seconds"] == 362
        assert measured["container_duration_seconds"] > 362
    else:
        with pytest.raises(ValueError, match="frame safety"):
            plugin._inspect_ace_xl_song_mp3(path)


@pytest.mark.parametrize("case", ["nonfinite", "wrong_rate", "mono", "empty", "too_many_frames", "integer"])
def test_raw_array_strict_safety(song_bridge, monkeypatch, case):
    module, plugin = song_bridge
    artifact = types.SimpleNamespace(audio_tensor=audio(), audio_sampling_rate=48000)
    if case == "nonfinite":
        artifact.audio_tensor[0, 0] = np.inf
    elif case == "wrong_rate":
        artifact.audio_sampling_rate = 44100
    elif case == "mono":
        artifact.audio_tensor = artifact.audio_tensor[:1]
    elif case == "empty":
        artifact.audio_tensor = artifact.audio_tensor[:, :0]
    elif case == "too_many_frames":
        monkeypatch.setattr(module, "ACE_XL_MAX_FRAMES", 4799)
    else:
        artifact.audio_tensor = artifact.audio_tensor.astype(np.int16)
    with pytest.raises(ValueError):
        list(plugin._ace_xl_song_raw_chunks(artifact))


def test_bounded_stderr_and_probe_stdout(song_bridge):
    _, plugin = song_bridge
    output = plugin._ace_xl_song_process([sys.executable, "-c", "import sys; sys.stderr.write('x'*1000000); print('ok')"])
    assert output.strip() == b"ok"
    with pytest.raises(ValueError, match="64 KiB"):
        plugin._ace_xl_song_process([sys.executable, "-c", "print('x'*65537)"])


@pytest.mark.parametrize("original,expected", [("F#m", "F# minor"), ("cmaj", "C major"), ("BB major", None),
                                               ("b\u266d min", "Bb minor"), ("G\u266fM", "G# minor")])
def test_native_key_normalization_examples(song_bridge, original, expected):
    _, plugin = song_bridge
    job = song_job()
    job["generation"]["options"]["key"] = original
    if expected is None:
        with pytest.raises(ValueError, match="key"):
            plugin._validate_audio_job(job, 2501)
    else:
        settings = plugin._validate_audio_job(job, 2501)
        assert settings["custom_settings"]["keyscale"] == expected
        assert settings["_midom_music_song"]["requested_controls"]["key"] == original


def test_serialized_candidate_and_claim_exact_shape(song_bridge):
    _, plugin = song_bridge
    candidate = json.loads((FIXTURES / "candidate.json").read_text())
    claim = json.loads((FIXTURES / "claim.json").read_text())
    plugin._worker_id = lambda config: candidate["worker_id"]
    plugin._connection_scope_value = lambda config, key: {"org_id": candidate["org_id"], "project_id": candidate["project_id"],
                                                        "paired_user_id": candidate["requested_by_user_id"]}[key]
    assert plugin._candidate_incompatibility_reason(candidate, None) is None
    settings = plugin._validate_job(claim, None)
    song = settings["_midom_music_song"]
    for key in ("contract_version", "recipe_version", "native_preset", "artifact_safety_profile_id", "lyrics_mode", "duration_mode"):
        assert song[key] == candidate["summary"][key]


def test_instrumental_language_and_utf8_boundaries(song_bridge):
    _, plugin = song_bridge
    job = song_job()
    job["lyrics"] = ""
    job["generation"]["lyrics_mode"] = "instrumental"
    with pytest.raises(ValueError, match="language"):
        plugin._validate_audio_job(job, 2501)
    job["generation"]["options"]["language"] = "unknown"
    assert plugin._validate_audio_job(job, 2501)["prompt"] == "[Instrumental]"
    job = song_job()
    job["lyrics"] = "\u00e9" * 8192
    assert plugin._validate_audio_job(job, 2501)["prompt"] == job["lyrics"]
    job["lyrics"] += "x"
    with pytest.raises(ValueError, match="UTF-8"):
        plugin._validate_audio_job(job, 2501)


def test_authorization_failure_is_not_a_reviewable_take(song_bridge, monkeypatch, tmp_path):
    module, plugin = song_bridge
    original = plugin._ace_xl_song_check_active
    def failed_lease():
        original()
        raise ValueError("lease expired")
    monkeypatch.setattr(plugin, "_ace_xl_song_check_active", failed_lease)
    _, uploads, update = run_song(plugin, module, monkeypatch, tmp_path)
    assert update[0] == "fail" and "lease expired" in update[1]["message"]
    assert not uploads


@pytest.mark.parametrize("failure", [False, True])
def test_restore_preexisting_instance_encoder_wrapper(song_bridge, monkeypatch, tmp_path, failure):
    module, plugin = song_bridge
    def modify(session):
        original = session.pipeline._encode_prompt
        def existing(*args, **kwargs):
            if failure:
                raise ValueError("prior encoder failed")
            return original(*args, **kwargs)
        session.pipeline._encode_prompt = existing
    _, uploads, update = run_song(plugin, module, monkeypatch, tmp_path, modify=modify)
    assert (update[0] == "complete") is (not failure)
    assert bool(uploads) is (not failure)


def test_description_parser_formatting_keeps_original_provenance(song_bridge, monkeypatch, tmp_path):
    module, plugin = song_bridge
    job = song_job()
    job["music_description"] = " Warm pop. \r\n Piano.  \r\n"
    _, uploads, update = run_song(plugin, module, monkeypatch, tmp_path, job=job)
    assert uploads and update[0] == "complete", update
    metadata = update[1]["generation_metadata"]["music_song"]
    assert metadata["original_music_description_sha256"] == hashlib.sha256(job["music_description"].encode()).hexdigest()
    assert metadata["effective_music_description_sha256"] == hashlib.sha256("Warm pop.\n Piano.".encode()).hexdigest()


def test_song_stages_share_progress_429_budget(progress):
    f = progress
    f.replies.append(f.response(429, retry="9"))
    reporter = enqueue(f, progress=1)
    reporter.flush_due()
    for index in range(100):
        f.plugin._ace_xl_song_progress("validating_output", f"stage {index}", 92)
    f.plugin._ace_xl_song_progress("uploading", "Uploading verified song", 95)
    assert reporter.error is None and len(f.posts) == 1
    f.clock.now += 9
    reporter.flush_due()
    assert len(f.posts) == 2 and f.posts[-1][2]["phase"] == "uploading"
    assert not f.plugin._stop_event.is_set()


def test_song_stage_does_not_swallow_authorization_failure(progress):
    f = progress
    f.replies.append(f.response(403))
    reporter = enqueue(f, progress=1)
    reporter.flush_due()
    with pytest.raises(Exception):
        f.plugin._ace_xl_song_progress("uploading", "Must not upload", 95)
