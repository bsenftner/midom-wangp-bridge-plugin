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

    spec = importlib.util.spec_from_file_location("midom_bridge_plugin_control_guided_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def control_guided_job(module, *, outputs=True):
    output_payload = {"count": 2, "format": "mixed"}
    if outputs:
        output_payload["artifacts"] = [
            {"artifact_index": 0, "role": "control_video", "mime_type": "video/mp4"},
            {"artifact_index": 1, "role": "driving_audio", "mime_type": "audio/mpeg"},
        ]
    return {
        "job_id": 901,
        "family": "media_processing",
        "media_type": "video",
        "processing_task": module.STORYBOARD_FFMPEG_PROCESSING_TASK,
        "processor_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
        "operation_type": module.CONTROL_GUIDED_INPUT_PREPARATION_OPERATION,
        "output": output_payload,
        "inputs": [
            {
                "kind": "control_video_source",
                "input_id": 10,
                "sequence": 0,
                "source_start_seconds": 2.0,
                "source_end_seconds": 5.0,
                "source_duration_seconds": 30.0,
                "mime_type": "video/webm",
            },
            {
                "kind": "driving_audio_source",
                "input_id": 11,
                "sequence": 0,
                "source_start_seconds": 2.0,
                "source_end_seconds": 5.0,
                "source_duration_seconds": 30.0,
                "mime_type": "video/webm",
            },
        ],
        "processing": {
            "operation_type": module.CONTROL_GUIDED_INPUT_PREPARATION_OPERATION,
            "contract_version": module.CONTROL_GUIDED_INPUT_PREPARATION_CONTRACT_VERSION,
            "profile_id": "ltx_control_guided_720p",
            "output_width": 1280,
            "output_height": 720,
            "fps": 30,
            "fit_mode": "content_detect_then_cover",
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "embedded_audio": "selected_soundtrack",
            "audio_codec": "aac",
            "audio_sample_rate_hz": 48000,
            "head_seconds": 0,
            "tail_seconds": 1.25,
            "tail_audio_mode": "silence",
            "editorial_duration_seconds": 3.0,
            "prepared_duration_seconds": 4.25,
        },
    }


def test_capability_advertises_control_guided_preparation():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    capability = plugin._storyboard_ffmpeg_processing_capability()

    assert module.CONTROL_GUIDED_INPUT_PREPARATION_OPERATION in capability["operation_types"]
    assert module.CONTROL_GUIDED_INPUT_PREPARATION_OPERATION in capability["operation_features"]
    contract = capability["contracts"][module.CONTROL_GUIDED_INPUT_PREPARATION_OPERATION]
    assert contract["contract_version"] == module.CONTROL_GUIDED_INPUT_PREPARATION_CONTRACT_VERSION
    assert contract["output_mime_types_by_role"]["control_video"] == ["video/mp4"]
    assert "audio/mpeg" in contract["output_mime_types_by_role"]["driving_audio"]


def test_validate_control_guided_claim_accepts_two_typed_outputs():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    settings = plugin._validate_storyboard_ffmpeg_processing_job(control_guided_job(module), 901)

    assert settings["_midom_operation_type"] == module.CONTROL_GUIDED_INPUT_PREPARATION_OPERATION
    assert settings["_midom_output_count"] == 2
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "control_video", "mime_type": "video/mp4"},
        {"artifact_index": 1, "role": "driving_audio", "mime_type": "audio/mpeg"},
    ]
    assert settings["_midom_processing"]["contract_version"] == module.CONTROL_GUIDED_INPUT_PREPARATION_CONTRACT_VERSION


def test_validate_control_guided_claim_accepts_control_video_only_without_audio():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = control_guided_job(module, outputs=False)
    job["output"] = {
        "count": 1,
        "format": "mp4",
        "artifacts": [
            {"artifact_index": 0, "role": "control_video", "mime_type": "video/mp4"},
        ],
    }
    job["inputs"] = [job["inputs"][0]]
    job["processing"]["embedded_audio"] = "none"
    job["processing"]["tail_audio_mode"] = "none"

    settings = plugin._validate_storyboard_ffmpeg_processing_job(job, 901)

    assert settings["_midom_output_count"] == 1
    assert settings["_midom_output_declarations"] == [
        {"artifact_index": 0, "role": "control_video", "mime_type": "video/mp4"},
    ]
    assert settings["_midom_processing"]["embedded_audio"] == "none"


def test_validate_control_guided_rejects_unknown_recipe_field():
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = control_guided_job(module)
    job["processing"]["filter_complex"] = "[0:v]null[vout]"

    try:
        plugin._validate_storyboard_ffmpeg_processing_job(job, 901)
    except ValueError as exc:
        assert "Unsupported control-guided input preparation field" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_candidate_allows_control_guided_two_outputs():
    module = load_plugin_module()
    plugin = plugin_instance(module)

    reason = plugin._storyboard_ffmpeg_candidate_incompatibility_reason(
        {
            "family": "media_processing",
            "media_type": "video",
            "processing_task": module.STORYBOARD_FFMPEG_PROCESSING_TASK,
            "processor_id": module.STORYBOARD_FFMPEG_PROCESSOR_ID,
            "operation_type": module.CONTROL_GUIDED_INPUT_PREPARATION_OPERATION,
            "summary": {
                "output_count": 2,
                "output_format": "mixed",
                "video_input_count": 2,
                "control_video_source_count": 2,
                "driving_audio_source_count": 1,
            },
        }
    )

    assert reason is None


def test_control_guided_renderer_returns_typed_artifacts(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    settings = plugin._validate_storyboard_ffmpeg_processing_job(control_guided_job(module), 901)

    visual_path = tmp_path / "visual.webm"
    audio_path = tmp_path / "audio.webm"
    visual_path.write_bytes(b"visual")
    audio_path.write_bytes(b"audio")
    downloaded = [
        {
            "kind": "control_video_source",
            "input_id": 10,
            "sequence": 0,
            "path": str(visual_path),
            "category": "video",
            "mime_type": "video/webm",
            "source_start_seconds": 2.0,
            "source_end_seconds": 5.0,
            "source_duration_seconds": 30.0,
            "metadata": {"duration_seconds": 30.0, "duration_source": "input.source_duration_seconds", "has_audio": True},
        },
        {
            "kind": "driving_audio_source",
            "input_id": 11,
            "sequence": 0,
            "path": str(audio_path),
            "category": "audio",
            "mime_type": "video/webm",
            "source_start_seconds": 2.0,
            "source_end_seconds": 5.0,
            "source_duration_seconds": 30.0,
            "metadata": {"duration_seconds": 30.0, "duration_source": "input.source_duration_seconds", "sample_rate_hz": 48000},
        },
    ]

    def write_video(*args, **kwargs):
        args[3].write_bytes(b"\x00\x00\x00\x18ftypmp42")

    def write_audio(*args, **kwargs):
        args[3].write_bytes(b"mp3")

    def write_final(*args, **kwargs):
        args[4].write_bytes(b"\x00\x00\x00\x18ftypmp42")

    monkeypatch.setattr(plugin, "_run_control_guided_visual_segment_ffmpeg", write_video)
    monkeypatch.setattr(plugin, "_run_control_guided_audio_assembly_ffmpeg", write_audio)
    monkeypatch.setattr(plugin, "_run_control_guided_final_video_ffmpeg", write_final)
    monkeypatch.setattr(
        plugin,
        "_validate_control_guided_video_output",
        lambda *args, **kwargs: {
            "display_width": 1280,
            "display_height": 720,
            "duration_seconds": 4.25,
            "video_codec": "h264",
            "audio_codec": "aac",
            "has_audio": True,
        },
    )
    monkeypatch.setattr(
        plugin,
        "_validate_control_guided_audio_output",
        lambda *args, **kwargs: {"duration_seconds": 4.25, "sample_rate_hz": 48000},
    )
    monkeypatch.setattr(plugin, "_ffmpeg_version_string", lambda: "ffmpeg test")

    artifacts, metadata = plugin._run_control_guided_input_preparation_job(
        types.SimpleNamespace(worker_id=7),
        901,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
        progress_start=5,
        progress_end=95,
    )

    assert [(item["artifact_index"], item["role"], item["mime_type"]) for item in artifacts] == [
        (0, "control_video", "video/mp4"),
        (1, "driving_audio", "audio/mpeg"),
    ]
    assert metadata["contract_version"] == module.CONTROL_GUIDED_INPUT_PREPARATION_CONTRACT_VERSION
    assert metadata["source_duration_fallback_used"] is True
    assert metadata["visual_source_ranges"][0]["source_duration_seconds"] == 30.0
    assert metadata["visual_source_ranges"][0]["selected_duration_seconds"] == 3.0
    assert metadata["visual_source_ranges"][0]["source_end_seconds"] == 5.0
    assert metadata["artifact_roles"][1]["role"] == "driving_audio"


def test_control_guided_download_preserves_claim_source_range(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    job = control_guided_job(module)
    for descriptor in job["inputs"]:
        descriptor["source_start_seconds"] = 6.280
        descriptor["source_end_seconds"] = 14.960
        descriptor["source_duration_seconds"] = 30.0
        descriptor["duration_seconds"] = 30.0
    job["processing"]["editorial_duration_seconds"] = 8.680
    job["processing"]["prepared_duration_seconds"] = 9.930
    settings = plugin._validate_storyboard_ffmpeg_processing_job(job, 901)

    class Response:
        status_code = 200

        def __init__(self, content_type, body):
            self.headers = {"Content-Type": content_type}
            self.body = body

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield self.body

        def close(self):
            return None

    def fake_get(url, *args, **kwargs):
        if str(url).endswith("/10"):
            return Response("video/webm", b"visual")
        return Response("video/webm", b"audio")

    monkeypatch.setattr(module.requests, "get", fake_get)
    plugin._probe_event_video_metadata = lambda path, **kwargs: {
        "duration_seconds": kwargs.get("duration_fallback_seconds") or 30.0,
        "duration_source": kwargs.get("duration_fallback_source") or "container",
        "width": 1280,
        "height": 720,
        "display_width": 1280,
        "display_height": 720,
        "has_audio": True,
    }
    plugin._probe_audio_metadata = lambda path, **kwargs: {
        "duration_seconds": kwargs.get("duration_fallback_seconds") or 30.0,
        "duration_source": kwargs.get("duration_fallback_source") or "container",
        "sample_rate_hz": 48000,
    }

    downloaded = plugin._download_storyboard_ffmpeg_processing_inputs(
        types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
        job,
        str(tmp_path),
        job["inputs"],
    )

    visual = next(item for item in downloaded if item["kind"] == "control_video_source")
    audio = next(item for item in downloaded if item["kind"] == "driving_audio_source")
    assert visual["source_start_seconds"] == 6.280
    assert visual["source_end_seconds"] == 14.960
    assert visual["source_duration_seconds"] == 30.0
    assert visual["duration_seconds"] == 30.0
    assert audio["source_start_seconds"] == 6.280
    assert audio["source_end_seconds"] == 14.960

    seen_visual_ranges = []
    seen_audio_ranges = []

    def write_video(connection, job_id, video_input, output_path, output_width, output_height, duration_seconds, process_handle, **kwargs):
        start, duration = plugin._control_guided_source_range(video_input)
        seen_visual_ranges.append((start, duration, duration_seconds))
        output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    def write_audio(connection, job_id, audio_inputs, output_path, editorial_duration, head_seconds, tail_seconds, tail_audio_mode, process_handle, **kwargs):
        start, duration = plugin._control_guided_source_range(audio_inputs[0])
        seen_audio_ranges.append((start, duration, editorial_duration))
        output_path.write_bytes(b"mp3")

    def write_final(connection, job_id, visual_path, driving_audio_path, output_path, output_width, output_height, prepared_duration, head_seconds, tail_seconds, process_handle, **kwargs):
        output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    monkeypatch.setattr(plugin, "_run_control_guided_visual_segment_ffmpeg", write_video)
    monkeypatch.setattr(plugin, "_run_control_guided_audio_assembly_ffmpeg", write_audio)
    monkeypatch.setattr(plugin, "_run_control_guided_final_video_ffmpeg", write_final)
    monkeypatch.setattr(
        plugin,
        "_validate_control_guided_video_output",
        lambda *args, **kwargs: {"display_width": 1280, "display_height": 720, "duration_seconds": 9.93, "has_audio": True},
    )
    monkeypatch.setattr(plugin, "_validate_control_guided_audio_output", lambda *args, **kwargs: {"duration_seconds": 9.93})
    monkeypatch.setattr(plugin, "_ffmpeg_version_string", lambda: "ffmpeg test")

    _artifacts, metadata = plugin._run_control_guided_input_preparation_job(
        types.SimpleNamespace(worker_id=7),
        901,
        settings,
        downloaded,
        str(tmp_path),
        types.SimpleNamespace(),
        progress_start=5,
        progress_end=95,
    )

    assert seen_visual_ranges == [(6.280, 8.680, 8.680)]
    assert seen_audio_ranges == [(6.280, 8.680, 8.680)]
    assert metadata["visual_source_ranges"][0]["source_start_seconds"] == 6.280
    assert metadata["visual_source_ranges"][0]["source_end_seconds"] == 14.96
    assert metadata["visual_source_ranges"][0]["source_duration_seconds"] == 30.0
    assert metadata["visual_source_ranges"][0]["selected_duration_seconds"] == 8.68
    assert metadata["audio_source_ranges"][0]["source_start_seconds"] == 6.280
    assert metadata["audio_source_ranges"][0]["source_end_seconds"] == 14.96


def test_typed_media_processing_upload_sends_role_and_exact_mime(monkeypatch, tmp_path):
    module = load_plugin_module()
    plugin = plugin_instance(module)
    path = tmp_path / "prepared.mp3"
    path.write_bytes(b"fake mp3 data")
    captured = {}

    class Response:
        status_code = 200
        content = b"{}"

        def json(self):
            return {"artifact_id": 44, "file_id": 55, "artifact_index": 1, "role": "driving_audio"}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured["data"] = data
        captured["files"] = files
        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)

    artifact = plugin._upload_media_processing_artifact(
        types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7),
        901,
        {"path": str(path), "artifact_index": 1, "role": "driving_audio", "mime_type": "audio/mpeg"},
        {},
    )

    assert captured["data"]["artifact_index"] == "1"
    assert captured["data"]["role"] == "driving_audio"
    assert captured["data"]["mime_type"] == "audio/mpeg"
    assert artifact == {"artifact_id": 44, "file_id": 55, "artifact_index": 1, "role": "driving_audio"}
