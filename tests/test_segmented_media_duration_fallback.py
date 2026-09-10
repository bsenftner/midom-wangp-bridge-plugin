import importlib.util
import json
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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_segment_test", PLUGIN_PATH)
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
    plugin._ffprobe_binary = lambda: "ffprobe"
    return plugin


def test_segmented_media_duration_fallback_prefers_input_source_duration():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    duration, source = plugin._storyboard_segment_duration_fallback(
        {"source_duration_seconds": "30.023", "duration_seconds": "29.0"},
        {"duration_seconds": "28.0", "mediaassembly_input": {"duration_seconds": "27.0"}},
        Headers({"X-Midom-Input-Duration-Seconds": "26.0"}),
    )

    assert duration == 30.023
    assert source == "input.source_duration_seconds"


def test_segmented_media_probe_accepts_missing_container_duration_with_fallback(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "segment.webm"
    path.write_bytes(b"fake-webm")

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "vp9",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "30/1",
            }
        ],
        "format": {},
    }

    completed = types.SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)

    metadata = plugin._probe_event_video_metadata(
        path,
        duration_fallback_seconds=5.05,
        duration_fallback_source="processing.source_duration_seconds",
    )

    assert metadata["duration_seconds"] == 5.05
    assert metadata["duration_source"] == "processing.source_duration_seconds"
    assert metadata["display_width"] == 1280
    assert metadata["display_height"] == 720


def test_storyboard_source_audio_fallback_does_not_use_processing_duration():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    duration, source = plugin._storyboard_source_audio_duration_fallback(
        {"source_duration_seconds": "30.0", "duration_seconds": "29.9"},
        Headers({"X-Midom-Input-Duration-Seconds": "28.0"}),
    )

    assert duration == 30.0
    assert source == "input.source_duration_seconds"


def test_audio_probe_accepts_missing_duration_with_fallback(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "source-audio.webm"
    path.write_bytes(b"fake-webm")

    payload = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "opus",
                "sample_rate": "48000",
            }
        ],
        "format": {},
    }

    completed = types.SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)

    metadata = plugin._probe_audio_metadata(
        path,
        duration_fallback_seconds=30.0,
        duration_fallback_source="input.source_duration_seconds",
    )

    assert metadata["duration_seconds"] == 30.0
    assert metadata["duration_source"] == "input.source_duration_seconds"
    assert metadata["sample_rate_hz"] == 48000


def test_pass_through_source_audio_download_uses_descriptor_duration_fallback(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)

    class Response:
        status_code = 200

        def __init__(self, content_type, body=b"fake-webm", headers=None):
            self.headers = Headers({"Content-Type": content_type, **(headers or {})})
            self.body = body

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield self.body

        def close(self):
            return None

    connection = types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7)
    plugin._headers = lambda connection: {}
    plugin._ffmpeg_processing_probe = lambda: {"quicktime_demux_available": True}
    plugin._probe_event_video_metadata = lambda path, **kwargs: {
        "duration_seconds": 4.88,
        "width": 1280,
        "height": 720,
        "display_width": 1280,
        "display_height": 720,
        "has_audio": True,
    }

    def fake_get(url, *args, **kwargs):
        if str(url).endswith("/99"):
            return Response(
                "video/webm",
                headers={"X-Midom-Input-Duration-Seconds": "28.0"},
            )
        return Response("video/mp4", body=b"fake-mp4")

    monkeypatch.setattr(module.requests, "get", fake_get)

    payload = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "opus",
                "sample_rate": "48000",
            }
        ],
        "format": {},
    }
    completed = types.SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)

    downloaded = plugin._download_storyboard_ffmpeg_processing_inputs(
        connection,
        {
            "job_id": 10,
            "operation_type": "multicam_card_pass_through_take",
            "processing": {
                "duration_seconds": 4.88,
                "soundtrack_start": 18.56,
            },
        },
        str(tmp_path),
        [
            {
                "kind": "source_video",
                "input_id": 98,
                "role": "visual",
                "filename": "visual.mp4",
                "mime_type": "video/mp4",
            },
            {
                "kind": "source_audio",
                "input_id": 99,
                "role": "source",
                "filename": "screen-seg-0001.webm",
                "mime_type": "video/webm",
                "source_duration_seconds": 30.0,
            }
        ],
    )

    audio = next(item for item in downloaded if item["kind"] == "source_audio")
    assert audio["duration_seconds"] == 30.0
    assert audio["metadata"]["duration_source"] == "input.source_duration_seconds"
