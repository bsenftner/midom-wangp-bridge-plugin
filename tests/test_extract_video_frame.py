import importlib.util
import sys
import types
from pathlib import Path

from PIL import Image


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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_extract_frame_test", PLUGIN_PATH)
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


def extract_frame_job(module, *, width=1920, height=1080, frame_time=12.345, mime_type="video/mp4"):
    return {
        "job_id": 2001,
        "family": "media_processing",
        "media_type": "video",
        "processing_task": module.STORYBOARD_FFMPEG_PROCESSING_TASK,
        "processor_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "operation_type": module.EXTRACT_VIDEO_FRAME_OPERATION,
        "output": {
            "count": 1,
            "format": "png",
            "mime_type": "image/png",
            "max_duration_seconds": 60,
            "artifacts": [
                {
                    "artifact_index": 0,
                    "role": "frame_image",
                    "media_type": "image",
                    "format": "png",
                    "mime_type": "image/png",
                    "required": True,
                }
            ],
        },
        "inputs": [
            {
                "kind": "source_video",
                "input_id": 3101,
                "dbfileid": 3101,
                "role": "source",
                "filename": "source-video.webm" if mime_type == "video/webm" else "source-video.mp4",
                "mime_type": mime_type,
                "source_duration_seconds": 30.0,
                "source_width": width,
                "source_height": height,
            }
        ],
        "processing": {
            "operation_type": module.EXTRACT_VIDEO_FRAME_OPERATION,
            "operation": module.EXTRACT_VIDEO_FRAME_OPERATION,
            "contract_version": module.EXTRACT_VIDEO_FRAME_CONTRACT_VERSION,
            "frame_time_seconds": frame_time,
            "seek_mode": "source_timeline",
            "rotation_mode": "autorotate",
            "target_filename": "saved-frame.png",
            "output": {
                "format": "png",
                "mime_type": "image/png",
                "width": width,
                "height": height,
                "preserve_source_display_dimensions": True,
            },
        },
    }


def write_png(path: Path, width: int, height: int, color=(20, 40, 80, 255)):
    image = Image.new("RGBA", (width, height), color)
    image.save(path, format="PNG")


def test_extract_video_frame_capability_is_advertised():
    module = load_plugin_module()
    capability = plugin_instance(module)._storyboard_ffmpeg_processing_capability()

    assert module.EXTRACT_VIDEO_FRAME_OPERATION in capability["operation_types"]
    assert module.EXTRACT_VIDEO_FRAME_OPERATION in capability["supported_operations"]
    assert "image/png" in capability["output_mime_types"]
    assert "video/webm" in capability["input_mime_types"]["source_video"]
    contract = capability["contracts"][module.EXTRACT_VIDEO_FRAME_OPERATION]
    assert contract["contract_version"] == module.EXTRACT_VIDEO_FRAME_CONTRACT_VERSION
    assert contract["required_output_roles"] == ["frame_image"]
    assert contract["output_mime_types_by_role"]["frame_image"] == ["image/png"]
    assert "decode_to_requested_timestamp" in capability["operation_features"][module.EXTRACT_VIDEO_FRAME_OPERATION]


def test_extract_video_frame_candidate_is_compatible():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 2001,
        "worker_id": 7,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "media_type": "video",
        "model_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "summary": {
            "operation_type": module.EXTRACT_VIDEO_FRAME_OPERATION,
            "output_format": "png",
            "output_count": 1,
            "video_input_count": 1,
            "source_video_count": 1,
            "audio_input_count": 0,
        },
    }
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)

    assert plugin._candidate_incompatibility_reason(candidate, None) is None


def test_extract_video_frame_claim_validation_accepts_source_png_contract():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_storyboard_ffmpeg_processing_job(extract_frame_job(module), 2001)

    assert settings["_midom_operation_type"] == module.EXTRACT_VIDEO_FRAME_OPERATION
    assert settings["_midom_output_format"] == "png"
    assert settings["_midom_output_mime_type"] == "image/png"
    assert settings["_midom_max_artifact_bytes"] == module.MAX_IMAGE_BYTES
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "frame_image", "mime_type": "image/png"}
    ]


def test_extract_video_frame_download_uses_descriptor_duration_fallback(monkeypatch, tmp_path):
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
        assert kwargs["duration_fallback_seconds"] == 30.0
        assert kwargs["duration_fallback_source"] == "input.source_duration_seconds"
        return {
            "duration_seconds": 30.0,
            "duration_source": "input.source_duration_seconds",
            "width": 1280,
            "height": 720,
            "display_width": 1280,
            "display_height": 720,
            "has_audio": True,
        }

    plugin._probe_event_video_metadata = fake_probe

    job = extract_frame_job(module, width=1280, height=720, mime_type="video/webm")
    downloaded = plugin._download_storyboard_ffmpeg_processing_inputs(
        types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
        job,
        str(tmp_path),
        job["inputs"],
    )

    assert len(downloaded) == 1
    assert downloaded[0]["kind"] == "source_video"
    assert downloaded[0]["metadata"]["duration_source"] == "input.source_duration_seconds"


def test_extract_video_frame_renderer_uses_accurate_seek_and_autorotate(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = extract_frame_job(module, width=720, height=1280, frame_time=12.345)
    settings = plugin._validate_storyboard_ffmpeg_processing_job(job, 2001)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mp4")
    downloaded = [
        {
            "kind": "source_video",
            "category": "video",
            "input_id": 3101,
            "path": str(source),
            "mime_type": "video/mp4",
            "sha256": "sha",
            "metadata": {
                "duration_seconds": 30.0,
                "duration_source": "container",
                "width": 1280,
                "height": 720,
                "display_width": 720,
                "display_height": 1280,
            },
        }
    ]
    commands = []

    def fake_run_ffmpeg(connection, job_id, command, **kwargs):
        commands.append(command)
        write_png(Path(command[-1]), 720, 1280)

    plugin._run_ffmpeg_with_progress = fake_run_ffmpeg

    artifacts, metadata = plugin._run_storyboard_ffmpeg_processing_job(
        types.SimpleNamespace(worker_id=7),
        2001,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
    )

    command = commands[0]
    assert "-autorotate" in command
    assert command.index("-ss") > command.index("-i")
    assert command[command.index("-ss") + 1] == "12.345000"
    assert "-frames:v" in command
    assert command[command.index("-frames:v") + 1] == "1"
    assert "-c:v" in command
    assert command[command.index("-c:v") + 1] == "png"
    assert "scale" not in " ".join(command)
    assert artifacts[0]["artifact_index"] == 0
    assert artifacts[0]["role"] == "frame_image"
    assert artifacts[0]["mime_type"] == "image/png"
    assert metadata["output_width"] == 720
    assert metadata["output_height"] == 1280
    assert metadata["rotation_mode"] == "autorotate"


def test_extract_video_frame_png_validation_rejects_wrong_dimensions(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    png_path = tmp_path / "wrong.png"
    write_png(png_path, 640, 480)

    try:
        plugin._validate_storyboard_frame_png_output(
            png_path,
            expected_width=1280,
            expected_height=720,
            max_bytes=module.MAX_IMAGE_BYTES,
        )
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("Expected wrong-dimension PNG rejection.")


def test_extract_video_frame_png_validation_rejects_malformed_png(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    png_path = tmp_path / "bad.png"
    png_path.write_bytes(b"not a png")

    try:
        plugin._validate_storyboard_frame_png_output(
            png_path,
            expected_width=1280,
            expected_height=720,
            max_bytes=module.MAX_IMAGE_BYTES,
        )
    except ValueError as exc:
        assert "PNG" in str(exc)
    else:
        raise AssertionError("Expected malformed PNG rejection.")


def test_extract_video_frame_typed_upload(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "saved-frame.png"
    write_png(path, 1280, 720)
    connection = types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7)

    class Response:
        status_code = 200

        def json(self):
            return {"artifact_id": 50, "file_id": 60, "artifact_index": 0, "role": "frame_image"}

    captured = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured["data"] = data
        captured["files"] = files
        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = plugin._upload_media_processing_artifact(
        connection,
        2001,
        {"path": str(path), "artifact_index": 0, "role": "frame_image", "mime_type": "image/png"},
        {},
    )

    assert result == {"artifact_id": 50, "file_id": 60, "artifact_index": 0, "role": "frame_image"}
    assert captured["data"]["role"] == "frame_image"
    assert captured["data"]["mime_type"] == "image/png"
    assert captured["data"]["artifact_index"] == "0"
