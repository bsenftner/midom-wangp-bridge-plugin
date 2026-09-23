import importlib.util
import io
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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_overlay_png_test", PLUGIN_PATH)
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


def png_bytes(width=64, height=36, alpha=128):
    buffer = io.BytesIO()
    Image.new("RGBA", (width, height), (255, 0, 0, alpha)).save(buffer, format="PNG")
    return buffer.getvalue()


def mp4_bytes():
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32


def overlay_job(module, *, mime_type="video/mp4"):
    return {
        "job_id": 3001,
        "family": "media_processing",
        "media_type": "video",
        "processing_task": module.STORYBOARD_FFMPEG_PROCESSING_TASK,
        "processor_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "operation_type": module.OVERLAY_PNG_ON_VIDEO_OPERATION,
        "output": {
            "count": 1,
            "format": "mp4",
            "contract_version": module.SINGLE_OUTPUT_VIDEO_TRANSFORM_CONTRACT_VERSION,
            "artifacts": [
                {
                    "artifact_index": 0,
                    "role": "video",
                    "mime_type": "video/mp4",
                    "required": True,
                }
            ],
        },
        "inputs": [
            {
                "kind": "source_video",
                "role": "base_video",
                "input_id": 123,
                "dbfileid": 123,
                "filename": "source.webm" if mime_type == "video/webm" else "source.mp4",
                "mime_type": mime_type,
                "source_duration_seconds": 12.5,
            },
            {
                "kind": "overlay_png",
                "role": "overlay_image",
                "input_id": 456,
                "dbfileid": 456,
                "filename": "overlay.png",
                "mime_type": "image/png",
            },
        ],
        "processing": {
            "operation_type": module.OVERLAY_PNG_ON_VIDEO_OPERATION,
            "operation": module.OVERLAY_PNG_ON_VIDEO_OPERATION,
            "contract_version": module.SINGLE_OUTPUT_VIDEO_TRANSFORM_CONTRACT_VERSION,
            "duration_seconds": 12.5,
            "target_filename": "png-overlay-result.mp4",
            "visual": {
                "fit_mode": "scale_to_base",
                "x": 0,
                "y": 0,
                "opacity": 0.5,
                "active_start_seconds": 0.0,
                "active_end_seconds": 12.5,
                "alpha_mode": "png_alpha",
                "end_behavior": "repeat",
            },
            "audio": {
                "mode": "base",
            },
            "output": {
                "container": "mp4",
                "video_codec": "h264",
                "audio_codec": "aac",
                "pixel_format": "yuv420p",
                "faststart": True,
            },
        },
    }


def test_overlay_png_on_video_capability_is_advertised():
    module = load_plugin_module()
    capability = plugin_instance(module)._storyboard_ffmpeg_processing_capability()

    assert module.OVERLAY_PNG_ON_VIDEO_OPERATION in capability["operation_types"]
    assert module.OVERLAY_PNG_ON_VIDEO_OPERATION in capability["supported_operations"]
    assert capability["input_mime_types"]["overlay_png"] == ["image/png"]
    contract = capability["contracts"][module.OVERLAY_PNG_ON_VIDEO_OPERATION]
    assert contract["contract_version"] == module.SINGLE_OUTPUT_VIDEO_TRANSFORM_CONTRACT_VERSION
    assert contract["required_output_roles"] == ["video"]
    assert contract["output_mime_types_by_role"]["video"] == ["video/mp4"]
    assert "png_alpha" in capability["operation_features"][module.OVERLAY_PNG_ON_VIDEO_OPERATION]


def test_overlay_png_on_video_candidate_is_compatible():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 3001,
        "worker_id": 7,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "media_type": "video",
        "model_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "summary": {
            "operation_type": module.OVERLAY_PNG_ON_VIDEO_OPERATION,
            "output_format": "mp4",
            "output_count": 1,
            "video_input_count": 1,
            "source_video_count": 1,
            "overlay_png_count": 1,
            "audio_input_count": 0,
        },
    }
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)

    assert plugin._candidate_incompatibility_reason(candidate, None) is None


def test_overlay_png_on_video_claim_validation_accepts_contract():
    module = load_plugin_module()
    settings = plugin_instance(module)._validate_storyboard_ffmpeg_processing_job(overlay_job(module), 3001)

    assert settings["_midom_operation_type"] == module.OVERLAY_PNG_ON_VIDEO_OPERATION
    assert settings["_midom_output_format"] == "mp4"
    assert settings["_midom_output_mime_type"] == "video/mp4"
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "video", "mime_type": "video/mp4"}
    ]
    assert settings["_midom_processing"]["opacity"] == 0.5
    assert settings["_midom_processing"]["audio_mode"] == "base"


def test_overlay_png_on_video_downloads_base_video_and_png_with_duration_fallback(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)

    class Response:
        status_code = 200

        def __init__(self, content_type, body):
            self.headers = Headers({"Content-Type": content_type})
            self.body = body

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield self.body

        def close(self):
            return None

    def fake_get(url, *args, **kwargs):
        if str(url).endswith("/123"):
            return Response("video/webm", b"webm")
        if str(url).endswith("/456"):
            return Response("image/png", png_bytes())
        raise AssertionError(url)

    monkeypatch.setattr(module.requests, "get", fake_get)

    def fake_probe(path, **kwargs):
        assert kwargs["duration_fallback_seconds"] == 12.5
        assert kwargs["duration_fallback_source"] == "input.source_duration_seconds"
        return {
            "duration_seconds": 12.5,
            "duration_source": "input.source_duration_seconds",
            "width": 1280,
            "height": 720,
            "display_width": 1280,
            "display_height": 720,
            "has_audio": True,
            "audio_codec": "opus",
        }

    plugin._probe_event_video_metadata = fake_probe
    job = overlay_job(module, mime_type="video/webm")
    downloaded = plugin._download_storyboard_ffmpeg_processing_inputs(
        types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
        job,
        str(tmp_path),
        job["inputs"],
    )

    assert [item["kind"] for item in downloaded] == ["source_video", "overlay_png"]
    assert downloaded[0]["metadata"]["duration_source"] == "input.source_duration_seconds"
    assert downloaded[1]["decoded_format"] == "PNG"


def test_overlay_png_on_video_renderer_uses_png_alpha_opacity_and_base_audio(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_storyboard_ffmpeg_processing_job(overlay_job(module), 3001)
    source = tmp_path / "source.mp4"
    overlay = tmp_path / "overlay.png"
    source.write_bytes(mp4_bytes())
    overlay.write_bytes(png_bytes(1280, 720))
    downloaded = [
        {
            "kind": "source_video",
            "category": "video",
            "input_id": 123,
            "path": str(source),
            "mime_type": "video/mp4",
            "sha256": "video-sha",
            "metadata": {
                "duration_seconds": 12.5,
                "duration_source": "container",
                "width": 1280,
                "height": 720,
                "display_width": 1280,
                "display_height": 720,
                "has_audio": True,
                "audio_codec": "aac",
            },
        },
        {
            "kind": "overlay_png",
            "category": "image",
            "input_id": 456,
            "path": str(overlay),
            "mime_type": "image/png",
            "sha256": "png-sha",
            "width": 1280,
            "height": 720,
            "decoded_format": "PNG",
        },
    ]
    commands = []

    def fake_run_ffmpeg(connection, job_id, command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(mp4_bytes())

    plugin._run_ffmpeg_with_progress = fake_run_ffmpeg
    plugin._validate_storyboard_ffmpeg_output = lambda path, **kwargs: {
        "duration_seconds": 12.5,
        "display_width": kwargs["expected_width"],
        "display_height": kwargs["expected_height"],
        "video_codec": "h264",
        "audio_codec": "aac",
        "has_audio": True,
    }

    artifacts, metadata = plugin._run_storyboard_ffmpeg_processing_job(
        types.SimpleNamespace(worker_id=7),
        3001,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
    )

    command_text = " ".join(commands[0])
    assert "scale=1280:720" in command_text
    assert "format=rgba,colorchannelmixer=aa=0.500000" in command_text
    assert "overlay=0:0:format=auto:eof_action=repeat" in command_text
    assert "[0:a:0]aresample=48000" in command_text
    assert "anullsrc" not in command_text
    assert "-movflags" in commands[0]
    assert artifacts[0]["artifact_index"] == 0
    assert artifacts[0]["role"] == "video"
    assert artifacts[0]["mime_type"] == "video/mp4"
    assert metadata["output_width"] == 1280
    assert metadata["output_height"] == 720


def test_overlay_png_on_video_preserves_portrait_display_dimensions(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_storyboard_ffmpeg_processing_job(overlay_job(module), 3001)
    source = tmp_path / "portrait.mp4"
    overlay = tmp_path / "overlay.png"
    source.write_bytes(mp4_bytes())
    overlay.write_bytes(png_bytes(720, 1280))
    downloaded = [
        {
            "kind": "source_video",
            "category": "video",
            "input_id": 123,
                "path": str(source),
                "metadata": {
                    "duration_seconds": 12.5,
                "width": 1280,
                "height": 720,
                "display_width": 720,
                "display_height": 1280,
                "rotation_degrees": 90,
                "has_audio": True,
            },
        },
        {"kind": "overlay_png", "category": "image", "input_id": 456, "path": str(overlay), "decoded_format": "PNG"},
    ]
    commands = []
    plugin._run_ffmpeg_with_progress = lambda connection, job_id, command, **kwargs: commands.append(command) or Path(command[-1]).write_bytes(mp4_bytes())
    plugin._validate_storyboard_ffmpeg_output = lambda path, **kwargs: {
        "duration_seconds": 12.5,
        "display_width": kwargs["expected_width"],
        "display_height": kwargs["expected_height"],
        "video_codec": "h264",
        "audio_codec": "aac",
        "has_audio": True,
    }

    _artifacts, metadata = plugin._run_storyboard_ffmpeg_processing_job(
        types.SimpleNamespace(worker_id=7),
        3001,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
    )

    assert "scale=720:1280" in " ".join(commands[0])
    assert metadata["output_width"] == 720
    assert metadata["output_height"] == 1280


def test_overlay_png_on_video_rejects_wrong_output_role():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = overlay_job(module)
    job["output"]["artifacts"][0]["role"] = "frame_image"

    try:
        plugin._validate_storyboard_ffmpeg_processing_job(job, 3001)
    except ValueError as exc:
        assert "role must be video" in str(exc)
    else:
        raise AssertionError("Expected wrong-role rejection.")


def test_overlay_png_on_video_rejects_unsupported_visual_field():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = overlay_job(module)
    job["processing"]["visual"]["blend_mode"] = "screen"

    try:
        plugin._validate_storyboard_ffmpeg_processing_job(job, 3001)
    except ValueError as exc:
        assert "unsupported" in str(exc).lower()
    else:
        raise AssertionError("Expected unsupported recipe rejection.")


def test_overlay_png_on_video_typed_upload(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "png-overlay-result.mp4"
    path.write_bytes(mp4_bytes())
    connection = types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7)

    class Response:
        status_code = 200

        def json(self):
            return {"artifact_id": 50, "file_id": 60, "artifact_index": 0, "role": "video"}

    captured = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured["data"] = data
        captured["files"] = files
        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = plugin._upload_media_processing_artifact(
        connection,
        3001,
        {"path": str(path), "artifact_index": 0, "role": "video", "mime_type": "video/mp4"},
        {},
    )

    assert result == {"artifact_id": 50, "file_id": 60, "artifact_index": 0, "role": "video"}
    assert captured["data"]["role"] == "video"
    assert captured["data"]["mime_type"] == "video/mp4"
    assert captured["data"]["artifact_index"] == "0"
