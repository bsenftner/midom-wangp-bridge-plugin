import copy
import math
import shutil
import sys
import threading
import types

import pytest

from test_fish_audio_drama3 import Pipeline, Runtime, Session, fish_job, load_plugin, wav_bytes


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def progress(monkeypatch):
    module = load_plugin()
    clock, posts, logs = Clock(), [], []
    original = module.JobProgressCoalescer
    monkeypatch.setattr(original, "start", lambda self: None)
    monkeypatch.setattr(module, "JobProgressCoalescer", lambda *args: original(*args, clock=clock))
    plugin = module.AwsWorkerBridgePlugin()
    plugin._log = lambda message, **kwargs: logs.append(message)
    plugin._headers = lambda connection: {"Authorization": "Bearer test"}
    plugin._ensure_job_flow_enabled = lambda: None
    connection = types.SimpleNamespace(connection_id="worker-85", api_base_url="http://midom", worker_id=85)
    plugin._active_job_id, plugin._active_job_context = 1811, connection

    def response(code=200, *, retry=None, cancel=False):
        result = types.SimpleNamespace(status_code=code, headers={}, content=b"{}", json=lambda: {"cancel_requested": cancel})
        if retry is not None:
            result.headers["Retry-After"] = retry

        def check():
            if code >= 400:
                raise module.requests.HTTPError(f"HTTP {code}", response=result)

        result.raise_for_status = check
        return result

    replies = []

    def post(url, **kwargs):
        posts.append((clock(), url.rsplit("/", 1)[-1], copy.deepcopy(kwargs["json"])))
        return replies.pop(0) if replies else response()

    monkeypatch.setattr(module.requests, "post", post)
    fixture = types.SimpleNamespace(module=module, plugin=plugin, connection=connection, clock=clock,
                                    posts=posts, replies=replies, response=response, logs=logs)
    yield fixture
    if plugin._active_job_id is not None:
        plugin._close_job_progress(plugin._active_job_context, plugin._active_job_id)


def enqueue(fixture, **payload):
    fixture.plugin._post_job_update(fixture.connection, fixture.plugin._active_job_id, "progress", payload or {"progress": 1})
    return fixture.plugin._job_progress[1]


def test_changing_steps_statuses_and_fish_stages_share_one_budget(progress):
    f = progress
    callbacks = f.plugin._callbacks_for_job(f.connection, 1811)
    for index in range(600):
        callbacks.on_progress(types.SimpleNamespace(progress=index % 99, phase="speech", status=f"step {index}",
                                                   current_step=index, total_steps=600))
        callbacks.on_status(f"status {index}")
        if index % 20 == 0:
            f.plugin._fish_audio_progress(f.connection, 1811, "preparing", f"stage {index}")
        f.plugin._job_progress[1].flush_due()
        f.clock.now += 0.001
    assert len(f.posts) == 1
    assert f.plugin._active_job_status["current_step"] == 599
    assert f.plugin._active_job_status["status"] == "status 599"
    f.plugin._fish_audio_progress(f.connection, 1811, "validating_output", "Latest Fish stage")
    f.clock.now = 1002.0
    f.plugin._job_progress[1].flush_due()
    assert len(f.posts) == 2
    assert f.posts[-1][2]["status"] == "Latest Fish stage"
    assert f.posts[-1][2]["current_step"] == 599
    assert f.posts[-1][2]["phase"] == "validating_output"


def test_progress_429_after_callback_burst_does_not_fail_fish_stage(progress):
    f = progress
    f.replies.append(f.response(429, retry="9"))
    reporter = enqueue(f, progress=1)
    reporter.flush_due()
    callbacks = f.plugin._callbacks_for_job(f.connection, 1811)
    for step in range(500):
        callbacks.on_progress(types.SimpleNamespace(progress=20, current_step=step, total_steps=500))
    f.plugin._fish_audio_progress(f.connection, 1811, "uploading", "Uploading verified audio")
    assert reporter.error is None
    assert not f.plugin._stop_event.is_set()
    f.clock.now += 8.999
    reporter.flush_due()
    assert len(f.posts) == 1
    f.clock.now += 0.0011
    reporter.flush_due()
    assert len(f.posts) == 2
    assert f.posts[-1][2]["phase"] == "uploading"
    assert sum("Progress throttled" in message for message in f.logs) == 1
    assert any("cancellation delivery may be delayed" in message for message in f.logs)
    assert any("recovered" in message for message in f.logs)


@pytest.mark.parametrize("header,expected", [(None, 15), ("", 15), ("bad", 15), ("-3", 15),
                                           ("1.2", 15), ("Wed, 21 Oct 2026", 15),
                                           ("0", 2), ("1", 2), ("7", 7), ("999999999", 120)])
def test_retry_after_fallback_and_bounds(progress, header, expected):
    f = progress
    f.replies.append(f.response(429, retry=header))
    reporter = enqueue(f)
    reporter.flush_due()
    assert reporter._budget["retry_at"] == f.clock() + expected
    for _ in range(100):
        enqueue(f, progress=50)
        reporter.flush_due()
    assert len(f.posts) == 1
    f.clock.now += expected
    reporter.flush_due()
    assert len(f.posts) == 2


def test_repeated_429_logs_once_and_retries_without_new_callback(progress):
    f = progress
    f.replies.extend([f.response(429, retry="3"), f.response(429, retry="4"), f.response()])
    reporter = enqueue(f)
    reporter.flush_due()
    f.clock.now += 3
    reporter.flush_due()
    f.clock.now += 4
    reporter.flush_due()
    assert len(f.posts) == 3
    assert sum("Progress throttled" in message for message in f.logs) == 1


def test_backoff_recovery_still_observes_cancellation(progress):
    f = progress
    cancelled = []
    f.plugin._active_job = types.SimpleNamespace(done=False, cancel=lambda: cancelled.append(True))
    f.replies.extend([f.response(429, retry="5"), f.response(cancel=True)])
    reporter = enqueue(f)
    reporter.flush_due()
    assert cancelled == []
    f.clock.now += 5
    reporter.flush_due()
    assert cancelled == [True]
    assert f.plugin._cancel_requested_by_midom
    with pytest.raises(ValueError, match="cancel_requested"):
        f.plugin._fish_audio_progress(f.connection, 1811, "uploading", "Must not complete")
    with pytest.raises(RuntimeError, match="cancel_requested"):
        f.plugin._post_job_update(f.connection, 1811, "complete", {"artifacts": []})
    f.plugin._post_job_update(f.connection, 1811, "fail", {"reason": "cancel_requested"})
    assert [status for _, status, _ in f.posts] == ["progress", "progress", "fail"]


def test_idle_progress_keeps_lease_and_cancellation_check_alive(progress):
    f = progress
    reporter = enqueue(f, phase="loading", status="Loading model", progress=1)
    reporter.flush_due()
    f.clock.now += 59.999
    reporter.flush_due()
    assert len(f.posts) == 1
    f.clock.now += 0.002
    f.replies.append(f.response(cancel=True))
    reporter.flush_due()
    assert len(f.posts) == 2
    assert f.plugin._cancel_requested_by_midom


@pytest.mark.parametrize("code", [401, 403, 409, 410, 422, 500])
def test_only_progress_429_is_nonfatal(progress, code):
    f = progress
    cancelled = []
    f.plugin._active_job = types.SimpleNamespace(done=False, cancel=lambda: cancelled.append(True))
    f.replies.append(f.response(code))
    reporter = enqueue(f)
    reporter.flush_due()
    assert cancelled
    assert f.plugin._stop_event.is_set() == (code in {401, 403})
    with pytest.raises(f.module.requests.HTTPError, match=str(code)):
        f.plugin._fish_audio_progress(f.connection, 1811, "validating_output", "Do not hide failure")
    with pytest.raises(f.module.requests.HTTPError, match=str(code)):
        f.plugin._post_job_update(f.connection, 1811, "complete", {"artifacts": []})
    f.plugin._post_job_update(f.connection, 1811, "fail", {"reason": "actual_failure"})
    assert f.posts[-1][1] == "fail"


def test_progress_transport_failure_is_not_silenced(progress, monkeypatch):
    f = progress
    monkeypatch.setattr(f.module.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(
        f.module.requests.ConnectionError("connection lost")))
    reporter = enqueue(f)
    reporter.flush_due()
    with pytest.raises(f.module.requests.ConnectionError, match="connection lost"):
        f.plugin._raise_job_progress_error(f.connection, 1811)


@pytest.mark.parametrize("status", ["complete", "fail"])
@pytest.mark.parametrize("code", [401, 403, 409, 429, 500])
def test_terminal_http_errors_are_not_coalesced_or_swallowed(progress, status, code):
    f = progress
    reporter = enqueue(f)
    f.replies.append(f.response(code))
    with pytest.raises(f.module.requests.HTTPError, match=str(code)):
        f.plugin._post_job_update(f.connection, 1811, status, {"artifacts": []})
    assert reporter._closed
    assert [item[1] for item in f.posts] == [status]


def test_terminal_closes_pending_progress_without_waiting_for_backoff(progress):
    f = progress
    f.replies.append(f.response(429, retry="120"))
    reporter = enqueue(f)
    reporter.flush_due()
    enqueue(f, phase="completing", progress=99)
    f.plugin._post_job_update(f.connection, 1811, "complete", {"artifacts": [{"artifact_id": 1}]})
    reporter.flush_due()
    enqueue(f, progress=12, status="Late callback")
    assert [status for _, status, _ in f.posts] == ["progress", "complete"]
    assert f.clock() == 1000
    assert f.plugin._active_job_status["phase"] == "completing"


def test_next_job_retains_worker_backoff_but_not_old_payload(progress):
    f = progress
    f.replies.append(f.response(429, retry="10"))
    old = enqueue(f, status="old job")
    old.flush_due()
    f.plugin._post_job_update(f.connection, 1811, "fail", {"reason": "unrelated"})
    old_callbacks = f.plugin._callbacks_for_job(f.connection, 1811)
    f.plugin._active_job_id = 1812
    f.plugin._active_job_status = {}
    new = enqueue(f, status="new job")
    old_callbacks.on_status("Old job must not leak")
    assert f.plugin._active_job_status == {"status": "new job"}
    new.flush_due()
    assert len(f.posts) == 2
    f.clock.now += 10
    new.flush_due()
    assert f.posts[-1][2] == {"status": "new job"}
    assert old._closed


def test_separate_connections_have_separate_budgets(progress):
    f = progress
    old = enqueue(f)
    old.flush_due()
    f.plugin._close_job_progress(f.connection, 1811)
    f.connection = types.SimpleNamespace(connection_id="worker-86", api_base_url="http://midom", worker_id=86)
    f.plugin._active_job_context = f.connection
    enqueue(f).flush_due()
    assert len(f.posts) == 2
    assert len(f.plugin._progress_budgets) == 2


def test_removing_pairing_cleans_worker_budget(progress, monkeypatch, tmp_path):
    f = progress
    enqueue(f)
    monkeypatch.setattr(f.plugin, "_config_with_connections", lambda *args: {})
    monkeypatch.setattr(f.plugin, "_connection_records", lambda config: [])
    monkeypatch.setattr(f.plugin, "_load_config", lambda: {})
    monkeypatch.setattr(type(f.plugin), "_config_path", property(lambda self: tmp_path / "worker_config.json"))
    f.plugin._save_connections([])
    assert f.plugin._progress_budgets == {}


def test_concurrent_callbacks_and_keepalives_coalesce_while_http_is_blocked(progress, monkeypatch):
    f = progress
    entered, release = threading.Event(), threading.Event()

    def blocked(url, **kwargs):
        f.posts.append((f.clock(), "progress", copy.deepcopy(kwargs["json"])))
        entered.set()
        assert release.wait(5)
        return f.response()

    monkeypatch.setattr(f.module.requests, "post", blocked)
    reporter = enqueue(f, progress=1)
    sender = threading.Thread(target=reporter.flush_due)
    sender.start()
    try:
        assert entered.wait(2)
        callbacks = f.plugin._callbacks_for_job(f.connection, 1811)
        writers = [threading.Thread(target=lambda: [callbacks.on_status(f"step {i}") for i in range(150)])
                   for _ in range(4)]
        writers.append(threading.Thread(target=lambda: [enqueue(f, phase="keepalive") for _ in range(150)]))
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join(2)
            assert not writer.is_alive(), "Callback blocked on HTTP"
        f.plugin._fish_audio_progress(f.connection, 1811, "final_stage", "Latest status")
        reporter.flush_due()
        assert len(f.posts) == 1
    finally:
        release.set()
        sender.join(5)
    f.clock.now += 2
    reporter.flush_due()
    assert f.posts[-1][2]["status"] == "Latest status"
    assert f.posts[-1][2]["phase"] == "final_stage"


def test_real_scheduler_retries_without_callback_and_shuts_down():
    module = load_plugin()
    received, recovered = threading.Event(), threading.Event()
    replies = []
    clock = Clock()

    def send(payload):
        replies.append(payload)
        if len(replies) == 1:
            received.set()
            return types.SimpleNamespace(status_code=429, headers={"Retry-After": "2"})
        recovered.set()
        return types.SimpleNamespace(status_code=200)

    budget = {"last_attempt": -math.inf, "retry_at": 0.0, "throttled": False}
    reporter = module.JobProgressCoalescer(send, lambda exc: pytest.fail(str(exc)),
                                         lambda *args, **kwargs: None, budget, clock=clock)
    reporter.start()
    try:
        reporter.submit({"status": "automatic delivery"})
        assert received.wait(2)
        with reporter._condition:
            assert reporter._condition.wait_for(lambda: budget["retry_at"] > 0, timeout=2)
            clock.now += 2
            reporter._condition.notify_all()
        assert recovered.wait(2)
    finally:
        reporter.close()
    assert not reporter._thread.is_alive()
    assert len(replies) == 2


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
@pytest.mark.parametrize("failure", [None, "upload", "complete", "length", "silence"])
@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_fish_job_lifecycle_with_real_progress_boundary(progress, monkeypatch, tmp_path, failure, version):
    reviewable = version != "v1"
    f = progress
    plugin = f.plugin
    plugin._validate_job_scope = lambda *args: None
    plugin._audio_output_mime_types = lambda: ["audio/wav"]
    plugin._reset_idle_backoff_for_active_use = lambda *args, **kwargs: None
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    audio_video = types.ModuleType("shared.utils.audio_video")
    audio_video._ffmpeg_binary = lambda: shutil.which("ffmpeg")
    monkeypatch.setitem(sys.modules, "shared.utils.audio_video", audio_video)
    path = tmp_path / "speech_seed123.wav"
    path.write_bytes(wav_bytes(0.5))

    class AudioPipeline(Pipeline):
        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            result["x"] = types.SimpleNamespace(numel=lambda: 100)
            return result

    pipeline = AudioPipeline(failure if failure in {"length", "silence"} else "eos")
    native = Session(Runtime(pipeline, lambda: path.write_bytes(wav_bytes(0.5))), [str(path)])
    plugin._get_generation_session = lambda: native
    original_submit = native.submit_task
    progress_count = 0

    def post(url, **kwargs):
        nonlocal progress_count
        route = url.rsplit("/", 1)[-1]
        f.posts.append((f.clock(), route, copy.deepcopy(kwargs.get("json", {}))))
        if route == "progress":
            progress_count += 1
            return f.response(429, retry="8") if progress_count == 1 else f.response()
        if route == "artifacts":
            if failure == "upload":
                return f.response(429)
            assert kwargs["files"]["file"][1].read() == path.read_bytes()
            result = f.response()
            result.json = lambda: {"artifact_id": 12, "file_id": 34, "artifact_index": 0}
            return result
        return f.response(500) if route == "complete" and failure == "complete" else f.response()

    monkeypatch.setattr(f.module.requests, "post", post)

    def submit(settings, callbacks):
        reporter = plugin._job_progress[1]
        reporter.flush_due()
        for step in range(500):
            callbacks.on_progress(types.SimpleNamespace(progress=50, current_step=step, total_steps=500))
            callbacks.on_status(f"speech step {step}")
        plugin._fish_audio_progress(f.connection, 123, "decoding", "Fish stage after callback burst")
        reporter.flush_due()
        assert progress_count == 1
        f.clock.now += 8
        reporter.flush_due()
        return original_submit(settings, callbacks)

    native.submit_task = submit
    job = fish_job(speakers=3)
    if reviewable:
        job["generation"].update(speech_contract_version=f"fish_drama3_tts_{version}",
                                 acceptance_policy_id="reviewable_audio_v1",
                                 artifact_safety_profile_id="fish_audio_240s_50mib_v1" if version == "v3" else "fish_audio_180s_50mib_v1",
                                 duration_seconds=180 if version == "v3" else 60)
        job["output"].pop("max_duration_seconds")
    plugin._run_job(f.connection, job)
    routes = [route for _, route, _ in f.posts]
    assert routes[:2] == ["progress", "progress"]
    assert len(native.submitted) == 1
    assert native.runtime.module.server_config["audio_stand_alone_output_codec"] == "mp3"
    assert plugin._job_progress is None
    assert plugin._active_job_id is None
    if failure is None or (reviewable and failure in {"length", "silence"}):
        assert routes == ["progress", "progress", "artifacts", "complete"]
        complete = f.posts[-1][2]
        assert complete["artifacts"] == [{"artifact_id": 12, "file_id": 34, "artifact_index": 0}]
        metadata = complete["generation_metadata"]["fish_audio"]
        assert metadata["finish_reason"] == (failure if failure in {"length", "silence"} else "eos")
        if reviewable:
            assert metadata["execution_evidence"]["session_succeeded"] is True
            assert bool(metadata["warnings"]) == (failure in {"length", "silence"})
    else:
        assert routes[-1] == "fail"
        if failure in {"length", "silence"}:
            assert "artifacts" not in routes
            assert "complete" not in routes
            assert f.posts[-1][2]["reason"] == {
                "length": "speech_duration_limit_reached", "silence": "speech_stopped_prematurely",
            }[failure]
