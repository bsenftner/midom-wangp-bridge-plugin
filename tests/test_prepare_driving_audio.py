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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_prepare_audio_test", PLUGIN_PATH)
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
    plugin._ffmpeg_processing_probe = lambda: {
        "ffmpeg_available": True,
        "ffprobe_available": True,
        "nvenc_available": False,
        "quicktime_demux_available": True,
    }
    plugin._post_job_update = lambda *args, **kwargs: None
    plugin._set_active_job_status = lambda *args, **kwargs: None
    return plugin


def prepare_audio_job(module):
    return {
        "job_id": 1001,
        "family": "media_processing",
        "media_type": "video",
        "processing_task": module.STORYBOARD_FFMPEG_PROCESSING_TASK,
        "processor_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "operation_type": module.PREPARE_DRIVING_AUDIO_OPERATION,
        "output": {
            "contract_version": module.PREPARE_DRIVING_AUDIO_CONTRACT_VERSION,
            "count": 1,
            "format": "mp3",
            "artifacts": [
                {
                    "artifact_index": 0,
                    "role": "driving_audio",
                    "media_type": "audio",
                    "format": "mp3",
                    "mime_type": "audio/mpeg",
                    "mime_types": ["audio/mpeg"],
                    "required": True,
                }
            ],
        },
        "inputs": [
            {
                "kind": "driving_audio_source",
                "input_id": 2752,
                "sequence": 0,
                "source_start_seconds": 18.56,
                "source_end_seconds": 30.023,
                "source_duration_seconds": 30.023,
                "mime_type": "video/webm",
            },
            {
                "kind": "driving_audio_source",
                "input_id": 2753,
                "sequence": 1,
                "source_start_seconds": 0.0,
                "source_end_seconds": 7.217,
                "source_duration_seconds": 30.001,
                "mime_type": "video/webm",
            },
        ],
        "processing": {
            "operation_type": module.PREPARE_DRIVING_AUDIO_OPERATION,
            "contract_version": module.PREPARE_DRIVING_AUDIO_CONTRACT_VERSION,
            "profile_id": "audio_guided_driving_audio_mp3_v1",
            "editorial_duration_seconds": 18.68,
            "audio_codec": "mp3",
            "audio_sample_rate_hz": 48000,
            "audio_channels": 2,
            "audio_bitrate": "192k",
            "head_seconds": 0.0,
            "tail_seconds": 0.0,
            "padding_mode": "silence",
        },
    }


def test_prepare_driving_audio_capability_is_advertised():
    module = load_plugin_module()
    capability = plugin_instance(module)._storyboard_ffmpeg_processing_capability()

    assert module.PREPARE_DRIVING_AUDIO_OPERATION in capability["operation_types"]
    assert module.PREPARE_DRIVING_AUDIO_OPERATION in capability["supported_operations"]
    contract = capability["contracts"][module.PREPARE_DRIVING_AUDIO_OPERATION]
    assert contract["contract_version"] == module.PREPARE_DRIVING_AUDIO_CONTRACT_VERSION
    assert contract["output_mime_types_by_role"]["driving_audio"] == ["audio/mpeg"]
    assert "audio/mpeg" in capability["output_mime_types"]
    assert "mp3_48khz_stereo" in capability["operation_features"][module.PREPARE_DRIVING_AUDIO_OPERATION]


def test_prepare_driving_audio_candidate_is_compatible():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 1001,
        "worker_id": 7,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "media_type": "video",
        "model_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "summary": {
            "operation_type": module.PREPARE_DRIVING_AUDIO_OPERATION,
            "output_format": "mp3",
            "output_count": 1,
            "audio_input_count": 2,
            "driving_audio_source_count": 2,
        },
    }
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)

    assert plugin._candidate_incompatibility_reason(candidate, None) is None


def test_prepare_driving_audio_claim_validation_accepts_two_sources():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_storyboard_ffmpeg_processing_job(prepare_audio_job(module), 1001)

    assert settings["_midom_operation_type"] == module.PREPARE_DRIVING_AUDIO_OPERATION
    assert settings["_midom_output_format"] == "mp3"
    assert settings["_midom_output_mime_type"] == "audio/mpeg"
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "driving_audio", "mime_type": "audio/mpeg"}
    ]


def test_prepare_driving_audio_rejects_unsupported_recipe_field():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = prepare_audio_job(module)
    job["processing"]["ffmpeg_args"] = "-filter_complex unsafe"

    try:
        plugin._validate_storyboard_ffmpeg_processing_job(job, 1001)
    except ValueError as exc:
        assert "unsupported" in str(exc).lower()
    else:
        raise AssertionError("Expected unsupported recipe field rejection.")


def test_prepare_driving_audio_download_uses_descriptor_duration_fallback(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)

    class Response:
        status_code = 200

        def __init__(self, body=b"webm"):
            self.headers = Headers({"Content-Type": "video/webm"})
            self.body = body

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield self.body

        def close(self):
            return None

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())
    plugin._probe_audio_metadata = lambda path, **kwargs: {
        "duration_seconds": kwargs.get("duration_fallback_seconds"),
        "duration_source": kwargs.get("duration_fallback_source"),
        "sample_rate_hz": 48000,
    }

    downloaded = plugin._download_storyboard_ffmpeg_processing_inputs(
        types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
        prepare_audio_job(module),
        str(tmp_path),
        prepare_audio_job(module)["inputs"],
    )

    assert [item["input_id"] for item in downloaded] == [2752, 2753]
    assert downloaded[0]["source_start_seconds"] == 18.56
    assert downloaded[0]["metadata"]["duration_source"] == "input.source_duration_seconds"


def test_prepare_driving_audio_download_rejects_missing_audio_stream(monkeypatch, tmp_path):
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
            prepare_audio_job(module),
            str(tmp_path),
            prepare_audio_job(module)["inputs"],
        )
    except ValueError as exc:
        assert "audio stream" in str(exc)
    else:
        raise AssertionError("Expected no-audio rejection.")


def test_prepare_driving_audio_renderer_crosses_segment_boundary(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_storyboard_ffmpeg_processing_job(prepare_audio_job(module), 1001)
    source_a = tmp_path / "a.webm"
    source_b = tmp_path / "b.webm"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    downloaded = [
        {
            "kind": "driving_audio_source",
            "category": "audio",
            "input_id": 2752,
            "sequence": 0,
            "path": str(source_a),
            "mime_type": "video/webm",
            "sha256": "sha-a",
            "source_start_seconds": 18.56,
            "source_end_seconds": 30.023,
            "source_duration_seconds": 30.023,
            "metadata": {"duration_seconds": 30.023, "duration_source": "input.source_duration_seconds", "sample_rate_hz": 48000},
        },
        {
            "kind": "driving_audio_source",
            "category": "audio",
            "input_id": 2753,
            "sequence": 1,
            "path": str(source_b),
            "mime_type": "video/webm",
            "sha256": "sha-b",
            "source_start_seconds": 0.0,
            "source_end_seconds": 7.217,
            "source_duration_seconds": 30.001,
            "metadata": {"duration_seconds": 30.001, "duration_source": "input.source_duration_seconds", "sample_rate_hz": 48000},
        },
    ]
    commands = []

    def fake_run_ffmpeg(connection, job_id, command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"mp3")

    plugin._run_ffmpeg_with_progress = fake_run_ffmpeg
    plugin._validate_prepare_driving_audio_output = lambda path, **kwargs: {"duration_seconds": 18.68, "sample_rate_hz": 48000}
    plugin._ffmpeg_version_string = lambda: "ffmpeg test"

    artifacts, metadata = plugin._run_prepare_driving_audio_job(
        types.SimpleNamespace(worker_id=7),
        1001,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
        progress_start=5,
        progress_end=95,
    )

    command_text = " ".join(commands[0])
    assert "atrim=start=18.560:duration=11.463" in command_text
    assert "atrim=start=0.000:duration=7.217" in command_text
    assert "libmp3lame" in command_text
    assert "192k" in command_text
    assert artifacts[0]["role"] == "driving_audio"
    assert artifacts[0]["mime_type"] == "audio/mpeg"
    assert metadata["audio_source_ranges"][0]["source_start_seconds"] == 18.56
    assert round(metadata["audio_source_ranges"][0]["selected_duration_seconds"], 3) == 11.463
    assert metadata["audio_source_ranges"][1]["source_end_seconds"] == 7.217


def test_prepare_driving_audio_head_tail_silence_in_command(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = prepare_audio_job(module)
    job["processing"]["head_seconds"] = 1.0
    job["processing"]["tail_seconds"] = 2.0
    settings = plugin._validate_storyboard_ffmpeg_processing_job(job, 1001)
    source = tmp_path / "a.webm"
    source.write_bytes(b"a")
    downloaded = [{
        "kind": "driving_audio_source",
        "category": "audio",
        "input_id": 2752,
        "sequence": 0,
        "path": str(source),
        "source_start_seconds": 0.0,
        "source_end_seconds": 18.68,
        "source_duration_seconds": 30.0,
        "metadata": {"duration_seconds": 30.0, "duration_source": "container"},
    }]
    commands = []
    plugin._run_ffmpeg_with_progress = lambda connection, job_id, command, **kwargs: commands.append(command) or Path(command[-1]).write_bytes(b"mp3")
    plugin._validate_prepare_driving_audio_output = lambda path, **kwargs: {"duration_seconds": 21.68, "sample_rate_hz": 48000}
    plugin._ffmpeg_version_string = lambda: "ffmpeg test"

    plugin._run_prepare_driving_audio_job(
        types.SimpleNamespace(worker_id=7),
        1001,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
        progress_start=5,
        progress_end=95,
    )

    command_text = " ".join(commands[0])
    assert "atrim=duration=1.000" in command_text
    assert "atrim=duration=2.000" in command_text


def test_prepare_driving_audio_typed_upload(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "prepared-driving-audio.mp3"
    path.write_bytes(b"mp3-bytes")
    connection = types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7)

    class Response:
        status_code = 200

        def json(self):
            return {"artifact_id": 50, "file_id": 60, "artifact_index": 0, "role": "driving_audio"}

    captured = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured["data"] = data
        captured["files"] = files
        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = plugin._upload_media_processing_artifact(
        connection,
        1001,
        {"path": str(path), "artifact_index": 0, "role": "driving_audio", "mime_type": "audio/mpeg"},
        {},
    )

    assert result == {"artifact_id": 50, "file_id": 60, "artifact_index": 0, "role": "driving_audio"}
    assert captured["data"]["role"] == "driving_audio"
    assert captured["data"]["mime_type"] == "audio/mpeg"
    assert captured["data"]["artifact_index"] == "0"
