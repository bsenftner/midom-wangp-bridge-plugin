import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_PATH = Path(__file__).resolve().parents[1] / "plugin.py"


def load_plugin_module():
    sys.modules.setdefault("gradio", types.SimpleNamespace(Error=Exception))

    shared_module = types.ModuleType("shared")
    shared_api_module = types.ModuleType("shared.api")
    shared_api_module.init = lambda *args, **kwargs: None
    shared_utils_module = types.ModuleType("shared.utils")
    shared_plugins_module = types.ModuleType("shared.utils.plugins")
    shared_plugins_module.WAN2GPPlugin = type("WAN2GPPlugin", (), {})

    sys.modules.setdefault("shared", shared_module)
    sys.modules.setdefault("shared.api", shared_api_module)
    sys.modules.setdefault("shared.utils", shared_utils_module)
    sys.modules.setdefault("shared.utils.plugins", shared_plugins_module)

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_prepare_transcription_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Headers(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def plugin_instance(module):
    plugin = module.AwsWorkerBridgePlugin.__new__(module.AwsWorkerBridgePlugin)
    plugin._log = lambda *args, **kwargs: None
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._headers = lambda connection: {"Authorization": "Bearer token"}
    plugin._ffmpeg_binary = lambda: "ffmpeg"
    plugin._ffprobe_binary = lambda: "ffprobe"
    plugin._ffmpeg_version_string = lambda: "ffmpeg test"
    plugin._ffmpeg_processing_probe = lambda: {
        "ffmpeg_available": True,
        "ffprobe_available": True,
        "nvenc_available": False,
        "quicktime_demux_available": True,
    }
    plugin._post_job_update = lambda *args, **kwargs: None
    plugin._set_active_job_status = lambda *args, **kwargs: None
    plugin._active_job_status = {}
    return plugin


def transcription_job(module, *, input_count=1, mime_type="video/mp4"):
    inputs = [
        {
            "kind": "transcription_source",
            "role": "source",
            "input_id": 123 + index,
            "dbfileid": 123 + index,
            "filename": f"source-{index}.webm" if mime_type == "video/webm" else f"source-{index}.mp4",
            "mime_type": mime_type,
            "source_duration_seconds": 245.8,
        }
        for index in range(input_count)
    ]
    return {
        "job_id": 4001,
        "family": "media_processing",
        "media_type": "video",
        "processing_task": module.STORYBOARD_FFMPEG_PROCESSING_TASK,
        "processor_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "operation_type": module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION,
        "output": {
            "count": 1,
            "format": "mp3",
            "contract_version": module.PREPARE_TRANSCRIPTION_AUDIO_CONTRACT_VERSION,
            "artifacts": [
                {
                    "artifact_index": 0,
                    "role": "transcription_audio",
                    "mime_type": "audio/mpeg",
                    "required": True,
                }
            ],
        },
        "inputs": inputs,
        "processing": {
            "operation": module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION,
            "operation_type": module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION,
            "contract_version": module.PREPARE_TRANSCRIPTION_AUDIO_CONTRACT_VERSION,
            "source_kind": "video",
            "recipe": {
                "output_format": "mp3",
                "audio_codec": "libmp3lame",
                "sample_rate_hz": 16000,
                "channels": 1,
                "bitrate_kbps": 96,
                "preserve_full_duration": True,
            },
        },
    }


def mp3_bytes():
    return b"ID3\x04\x00\x00\x00\x00\x00\x10" + b"mp3-bytes"


def test_prepare_transcription_audio_capability_is_advertised():
    module = load_plugin_module()
    capability = plugin_instance(module)._storyboard_ffmpeg_processing_capability()

    assert module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION in capability["operation_types"]
    assert module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION in capability["supported_operations"]
    assert "video/webm" in capability["input_mime_types"]["transcription_source"]
    assert "audio/mpeg" in capability["output_mime_types"]
    contract = capability["contracts"][module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION]
    assert contract["contract_version"] == module.PREPARE_TRANSCRIPTION_AUDIO_CONTRACT_VERSION
    assert contract["required_output_roles"] == ["transcription_audio"]
    assert contract["output_mime_types_by_role"]["transcription_audio"] == ["audio/mpeg"]
    assert "mono_16khz_96kbps" in capability["operation_features"][module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION]


def test_prepare_transcription_audio_candidate_is_compatible():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 4001,
        "worker_id": 7,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "media_type": "video",
        "model_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "summary": {
            "operation_type": module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION,
            "transcription_source_count": 1,
            "audio_input_count": 1,
            "output_count": 1,
            "output_format": "mp3",
        },
    }
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)

    assert plugin._candidate_incompatibility_reason(candidate, None) is None


def test_prepare_transcription_audio_claim_validation_accepts_contract():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_storyboard_ffmpeg_processing_job(transcription_job(module), 4001)

    assert settings["_midom_operation_type"] == module.PREPARE_TRANSCRIPTION_AUDIO_OPERATION
    assert settings["_midom_output_format"] == "mp3"
    assert settings["_midom_output_mime_type"] == "audio/mpeg"
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "transcription_audio", "mime_type": "audio/mpeg"}
    ]
    assert settings["_midom_processing"]["sample_rate_hz"] == 16000
    assert settings["_midom_processing"]["channels"] == 1
    assert settings["_midom_processing"]["bitrate_kbps"] == 96


def test_prepare_transcription_audio_rejects_missing_or_multiple_sources():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    for input_count in (0, 2):
        try:
            plugin._validate_storyboard_ffmpeg_processing_job(transcription_job(module, input_count=input_count), 4001)
        except ValueError as exc:
            assert "exactly one transcription_source" in str(exc)
        else:
            raise AssertionError(f"Expected input_count={input_count} rejection.")


def test_prepare_transcription_audio_rejects_unsupported_recipe_value():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = transcription_job(module)
    job["processing"]["recipe"]["sample_rate_hz"] = 48000

    try:
        plugin._validate_storyboard_ffmpeg_processing_job(job, 4001)
    except ValueError as exc:
        assert "16000" in str(exc)
    else:
        raise AssertionError("Expected unsupported sample-rate rejection.")


def test_prepare_transcription_audio_download_uses_descriptor_duration_fallback(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)

    class Response:
        status_code = 200

        def __init__(self):
            self.headers = Headers({"Content-Type": "video/webm"})

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"webm"

        def close(self):
            return None

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())

    def fake_probe(path, **kwargs):
        assert kwargs["duration_fallback_seconds"] == 245.8
        assert kwargs["duration_fallback_source"] == "input.source_duration_seconds"
        return {
            "duration_seconds": 245.8,
            "duration_source": "input.source_duration_seconds",
            "sample_rate_hz": 48000,
            "channels": 2,
            "codec_name": "opus",
        }

    plugin._probe_audio_metadata = fake_probe
    job = transcription_job(module, mime_type="video/webm")
    downloaded = plugin._download_storyboard_ffmpeg_processing_inputs(
        types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
        job,
        str(tmp_path),
        job["inputs"],
    )

    assert len(downloaded) == 1
    assert downloaded[0]["kind"] == "transcription_source"
    assert downloaded[0]["metadata"]["duration_source"] == "input.source_duration_seconds"


def test_prepare_transcription_audio_download_rejects_missing_audio_stream(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)

    class Response:
        status_code = 200
        headers = Headers({"Content-Type": "video/webm"})

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"webm"

        def close(self):
            return None

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())
    plugin._probe_audio_metadata = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Audio input must contain an audio stream."))

    try:
        plugin._download_storyboard_ffmpeg_processing_inputs(
            types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
            transcription_job(module, mime_type="video/webm"),
            str(tmp_path),
            transcription_job(module, mime_type="video/webm")["inputs"],
        )
    except ValueError as exc:
        assert "audio stream" in str(exc)
    else:
        raise AssertionError("Expected missing-audio-stream rejection.")


def test_prepare_transcription_audio_renderer_extracts_full_duration_mp3(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_storyboard_ffmpeg_processing_job(transcription_job(module), 4001)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mp4")
    downloaded = [
        {
            "kind": "transcription_source",
            "category": "audio",
            "input_id": 123,
            "path": str(source),
            "mime_type": "video/mp4",
            "sha256": "source-sha",
            "metadata": {
                "duration_seconds": 245.8,
                "duration_source": "container",
                "sample_rate_hz": 48000,
                "channels": 2,
                "codec_name": "aac",
            },
        }
    ]
    commands = []

    def fake_run_ffmpeg(connection, job_id, command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(mp3_bytes())

    plugin._run_ffmpeg_with_progress = fake_run_ffmpeg
    plugin._validate_prepare_transcription_audio_output = lambda path, **kwargs: {
        "duration_seconds": 245.8,
        "duration_source": "container",
        "sample_rate_hz": 16000,
        "channels": 1,
        "codec_name": "mp3",
        "mime_type": "audio/mpeg",
        "bytes": Path(path).stat().st_size,
    }

    artifacts, metadata = plugin._run_storyboard_ffmpeg_processing_job(
        types.SimpleNamespace(worker_id=7),
        4001,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
    )

    command = commands[0]
    assert "-vn" in command
    assert command[command.index("-acodec") + 1] == "libmp3lame"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-b:a") + 1] == "96k"
    assert "-t" not in command
    assert artifacts[0]["artifact_index"] == 0
    assert artifacts[0]["role"] == "transcription_audio"
    assert artifacts[0]["mime_type"] == "audio/mpeg"
    assert metadata["audio_sample_rate_hz"] == 16000
    assert metadata["audio_channels"] == 1
    assert metadata["preserve_full_duration"] is True


def test_prepare_transcription_audio_output_validation_rejects_wrong_sample_rate(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "prepared-transcription-audio.mp3"
    path.write_bytes(mp3_bytes())
    plugin._probe_audio_metadata = lambda path: {
        "duration_seconds": 10.0,
        "sample_rate_hz": 48000,
        "channels": 1,
    }

    try:
        plugin._validate_prepare_transcription_audio_output(path, source_duration=10.0, max_bytes=module.MAX_AUDIO_BYTES)
    except ValueError as exc:
        assert "16000" in str(exc)
    else:
        raise AssertionError("Expected wrong sample-rate rejection.")


def test_prepare_transcription_audio_typed_upload(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "prepared-transcription-audio.mp3"
    path.write_bytes(mp3_bytes())
    connection = types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7)

    class Response:
        status_code = 200

        def json(self):
            return {"artifact_id": 50, "file_id": 60, "artifact_index": 0, "role": "transcription_audio"}

    captured = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured["data"] = data
        captured["files"] = files
        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = plugin._upload_media_processing_artifact(
        connection,
        4001,
        {"path": str(path), "artifact_index": 0, "role": "transcription_audio", "mime_type": "audio/mpeg"},
        {},
    )

    assert result == {"artifact_id": 50, "file_id": 60, "artifact_index": 0, "role": "transcription_audio"}
    assert captured["data"]["role"] == "transcription_audio"
    assert captured["data"]["mime_type"] == "audio/mpeg"
    assert captured["data"]["artifact_index"] == "0"
