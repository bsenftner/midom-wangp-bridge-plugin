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
    plugin._ffmpeg_binary = lambda: "ffmpeg"
    plugin._active_job_status = {}
    plugin._ffmpeg_processing_probe = lambda: {
        "ffmpeg_available": True,
        "ffprobe_available": True,
        "quicktime_demux_available": True,
    }
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

    duration, source = plugin._storyboard_input_source_duration_fallback(
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


def test_pass_through_source_video_download_uses_descriptor_duration_fallback(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)

    class Response:
        status_code = 200

        def __init__(self, content_type, body=b"fake", headers=None):
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

    def fake_get(url, *args, **kwargs):
        if str(url).endswith("/98"):
            return Response(
                "video/webm",
                headers={"X-Midom-Input-Duration-Seconds": "28.0"},
            )
        return Response("audio/wav", body=b"fake-wav")

    monkeypatch.setattr(module.requests, "get", fake_get)

    def fake_probe_video(path, **kwargs):
        assert kwargs["duration_fallback_seconds"] == 30.023
        assert kwargs["duration_fallback_source"] == "input.source_duration_seconds"
        return {
            "duration_seconds": kwargs["duration_fallback_seconds"],
            "duration_source": kwargs["duration_fallback_source"],
            "width": 1280,
            "height": 720,
            "display_width": 1280,
            "display_height": 720,
            "has_audio": True,
        }

    plugin._probe_event_video_metadata = fake_probe_video
    plugin._probe_audio_metadata = lambda path, **kwargs: {
        "duration_seconds": 30.0,
        "duration_source": "container",
        "sample_rate_hz": 48000,
    }

    downloaded = plugin._download_storyboard_ffmpeg_processing_inputs(
        connection,
        {
            "job_id": 10,
            "operation_type": "multicam_card_pass_through_take",
            "processing": {
                "trim_start_seconds": 13.84,
                "trim_duration_seconds": 9.5,
                "trim_end_seconds": 23.34,
                "duration_seconds": 9.5,
            },
        },
        str(tmp_path),
        [
            {
                "kind": "source_video",
                "input_id": 98,
                "role": "visual",
                "filename": "screen-seg-0001.webm",
                "mime_type": "video/webm",
                "source_duration_seconds": 30.023,
            },
            {
                "kind": "source_audio",
                "input_id": 99,
                "role": "source",
                "filename": "screen-seg-0001.wav",
                "mime_type": "audio/wav",
                "source_duration_seconds": 30.023,
            },
        ],
    )

    video = next(item for item in downloaded if item["kind"] == "source_video")
    assert video["metadata"]["duration_seconds"] == 30.023
    assert video["metadata"]["duration_source"] == "input.source_duration_seconds"


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


def segmented_extract_job(module):
    return {
        "job_id": 20,
        "family": "media_processing",
        "media_type": "video",
        "processing_task": module.STORYBOARD_FFMPEG_PROCESSING_TASK,
        "processor_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "operation_type": "segmented_media_extract_range",
        "target_filename": "segmented-preview.mp4",
        "inputs": [
            {"kind": "source_video", "input_id": 123, "dbfileid": 123, "role": "range_segment", "filename": "seg-0001.webm"},
            {"kind": "source_video", "input_id": 124, "dbfileid": 124, "role": "range_segment", "filename": "seg-0002.webm"},
        ],
        "processing": {
            "operation_type": "segmented_media_extract_range",
            "track_key": "screen",
            "start_seconds": 29.0,
            "duration_seconds": 13.0,
            "segments": [
                {
                    "segmentedmediasegmentid": 31,
                    "segment_index": 0,
                    "dbfileid": 123,
                    "segment_start_seconds": 0.0,
                    "segment_end_seconds": 30.0,
                    "trim_start_seconds": 29.0,
                    "trim_duration_seconds": 1.0,
                },
                {
                    "segmentedmediasegmentid": 32,
                    "segment_index": 1,
                    "dbfileid": 124,
                    "segment_start_seconds": 30.0,
                    "segment_end_seconds": 60.0,
                    "trim_start_seconds": 0.0,
                    "trim_duration_seconds": 12.0,
                },
            ],
            "target_width": 1280,
            "target_height": 720,
            "fps": 30,
            "crf": 20,
            "preset": "veryfast",
            "audio_bitrate": "128k",
            "mediaassembly_input": {"filename": "segmented-preview.mp4"},
        },
        "output": {"count": 1, "format": "mp4", "max_duration_seconds": 60},
    }


def test_segmented_extract_range_capability_is_advertised():
    module = load_plugin_module()
    capability = plugin_instance(module)._storyboard_ffmpeg_processing_capability()

    assert "segmented_media_extract_range" in capability["operation_types"]
    assert "segmented_media_extract_range" in capability["supported_operations"]
    assert "cross_segment_extract" in capability["operation_features"]["segmented_media_extract_range"]


def test_segmented_extract_range_candidate_is_compatible():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    candidate = {
        "job_id": 20,
        "worker_id": 7,
        "org_id": 1,
        "project_id": 2,
        "requested_by_user_id": 3,
        "media_type": "video",
        "model_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "summary": {
            "operation_type": "segmented_media_extract_range",
            "output_format": "mp4",
            "output_count": 1,
            "video_input_count": 2,
            "source_video_count": 2,
        },
    }
    plugin._worker_id = lambda config: 7
    plugin._connection_scope_value = lambda config, key: {"org_id": 1, "project_id": 2, "paired_user_id": 3}.get(key)

    assert plugin._candidate_incompatibility_reason(candidate, None) is None


def test_segmented_extract_range_claim_validation_accepts_two_ordered_sources():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    settings = plugin._validate_storyboard_ffmpeg_processing_job(segmented_extract_job(module), 20)

    assert settings["_midom_operation_type"] == "segmented_media_extract_range"
    assert settings["_midom_processing"]["output_width"] == 1280
    assert settings["_midom_processing"]["output_height"] == 720
    assert settings["_midom_processing"]["target_filename"] == "segmented-preview.mp4"


def test_segmented_extract_range_duration_fallback_uses_matching_segment_range():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = segmented_extract_job(module)

    duration, source = plugin._storyboard_extract_range_duration_fallback(
        {"input_id": 124, "dbfileid": 124},
        job["processing"],
        Headers({"X-Midom-Input-Duration-Seconds": "99"}),
    )

    assert duration == 30.0
    assert source == "processing.segments[1].segment_end_seconds-segment_start_seconds"


def test_segmented_extract_range_download_uses_segment_duration_fallback(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)

    class Response:
        status_code = 200

        def __init__(self, content_type, body=b"fake-webm"):
            self.headers = Headers({"Content-Type": content_type})
            self.body = body

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield self.body

        def close(self):
            return None

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response("video/webm"))
    connection = types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7)
    plugin._headers = lambda connection: {}
    seen_fallbacks = []

    def fake_probe_video(path, **kwargs):
        seen_fallbacks.append((kwargs["duration_fallback_seconds"], kwargs["duration_fallback_source"]))
        return {
            "duration_seconds": kwargs["duration_fallback_seconds"],
            "duration_source": kwargs["duration_fallback_source"],
            "width": 1280,
            "height": 720,
            "display_width": 1280,
            "display_height": 720,
            "has_audio": True,
        }

    plugin._probe_event_video_metadata = fake_probe_video

    downloaded = plugin._download_storyboard_ffmpeg_processing_inputs(
        connection,
        segmented_extract_job(module),
        str(tmp_path),
        segmented_extract_job(module)["inputs"],
    )

    assert [item["input_id"] for item in downloaded] == [123, 124]
    assert seen_fallbacks == [
        (30.0, "processing.segments[0].segment_end_seconds-segment_start_seconds"),
        (30.0, "processing.segments[1].segment_end_seconds-segment_start_seconds"),
    ]


def test_segmented_extract_range_ordering_and_renderer(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = segmented_extract_job(module)
    settings = plugin._validate_storyboard_ffmpeg_processing_job(job, 20)
    connection = types.SimpleNamespace(worker_id=7)
    plugin._post_job_update = lambda *args, **kwargs: None
    plugin._validate_storyboard_ffmpeg_output = lambda path, **kwargs: {
        "display_width": 1280,
        "display_height": 720,
        "duration_seconds": 13.0,
    }
    rendered_order = []

    def fake_extract(connection, job_id, video_input, output_path, output_width, output_height, process_handle, **kwargs):
        rendered_order.append((video_input["input_id"], kwargs["processing"]["trim_start_seconds"], kwargs["processing"]["trim_duration_seconds"]))
        output_path.write_bytes(b"mp4")

    def fake_concat(connection, job_id, segment_paths, output_path, total_seconds, process_handle, **kwargs):
        assert [path.name for path in segment_paths] == ["segmented-range-part-0.mp4", "segmented-range-part-1.mp4"]
        assert total_seconds == 13.0
        output_path.write_bytes(b"mp4")

    plugin._run_storyboard_segment_extract_ffmpeg = fake_extract
    plugin._concat_event_video_segments = fake_concat
    downloaded = [
        {
            "kind": "source_video",
            "category": "video",
            "input_id": 124,
            "dbfileid": 124,
            "path": str(tmp_path / "seg-0002.webm"),
            "metadata": {"duration_seconds": 30.0, "has_audio": True, "display_width": 1280, "display_height": 720},
        },
        {
            "kind": "source_video",
            "category": "video",
            "input_id": 123,
            "dbfileid": 123,
            "path": str(tmp_path / "seg-0001.webm"),
            "metadata": {"duration_seconds": 30.0, "has_audio": True, "display_width": 1280, "display_height": 720},
        },
    ]
    for item in downloaded:
        Path(item["path"]).write_bytes(b"webm")

    output_path, metadata = plugin._run_storyboard_ffmpeg_processing_job(
        connection,
        20,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
    )

    assert Path(output_path).name == "segmented-preview.mp4"
    assert rendered_order == [(123, 29.0, 1.0), (124, 0.0, 12.0)]
    assert metadata["operation_type"] == "segmented_media_extract_range"
    assert metadata["segment_count"] == 2


def test_segmented_extract_ffmpeg_preserves_audio_when_present(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    commands = []
    plugin._run_ffmpeg_with_progress = lambda connection, job_id, command, **kwargs: commands.append(command)
    source = tmp_path / "seg.webm"
    source.write_bytes(b"webm")

    plugin._run_storyboard_segment_extract_ffmpeg(
        types.SimpleNamespace(),
        20,
        {"input_id": 123, "path": str(source), "metadata": {"duration_seconds": 30.0, "has_audio": True}},
        tmp_path / "out.mp4",
        1280,
        720,
        types.SimpleNamespace(),
        processing={"trim_start_seconds": 29.0, "trim_duration_seconds": 1.0, "fps": 30},
        progress_start=5,
        progress_end=40,
        status="extract",
    )

    command_text = " ".join(commands[0])
    assert "anullsrc" not in command_text
    assert "[0:a:0]aresample=48000" in command_text


def test_segmented_extract_ffmpeg_synthesizes_silence_only_when_audio_missing(tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    commands = []
    plugin._run_ffmpeg_with_progress = lambda connection, job_id, command, **kwargs: commands.append(command)
    source = tmp_path / "seg.webm"
    source.write_bytes(b"webm")

    plugin._run_storyboard_segment_extract_ffmpeg(
        types.SimpleNamespace(),
        20,
        {"input_id": 123, "path": str(source), "metadata": {"duration_seconds": 30.0, "has_audio": False}},
        tmp_path / "out.mp4",
        1280,
        720,
        types.SimpleNamespace(),
        processing={"trim_start_seconds": 29.0, "trim_duration_seconds": 1.0, "fps": 30},
        progress_start=5,
        progress_end=40,
        status="extract",
    )

    command_text = " ".join(commands[0])
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in command_text
    assert "[1:a:0]aresample=48000" in command_text
