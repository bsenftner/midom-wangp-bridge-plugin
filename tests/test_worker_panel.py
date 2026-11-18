import json
import types
from pathlib import Path

import pytest

from test_fish_audio_drama3 import load_plugin


def connection_record(worker_id=80):
    return {
        "connection_id": f"worker-{worker_id}", "worker_id": worker_id,
        "project_id": worker_id + 100, "org_id": 10, "paired_user_id": 20,
        "api_base_url": "https://midom.example", "worker_token": "PRIVATE_TOKEN",
        "machine_name": "GPU", "capabilities_revision": 1,
    }


class Component:
    def __init__(self, gr, kind, value=None, **kwargs):
        self.kind, self.value, self.options, self.events = kind, value, kwargs, []
        self.parent = gr.stack[-1] if gr.stack else None
        self.gr = gr
        gr.components.append(self)

    def __enter__(self):
        self.gr.stack.append(self)
        return self

    def __exit__(self, *args):
        self.gr.stack.pop()

    def click(self, **kwargs):
        self.events.append(kwargs)


class PanelGradio:
    Error = type("PanelError", (Exception,), {})

    def __init__(self):
        self.components, self.stack = [], []

    @staticmethod
    def update(**kwargs):
        return {"__type__": "update", **kwargs}

    def __getattr__(self, kind):
        return lambda value=None, **kwargs: Component(self, kind, value, **kwargs)

    def find(self, label):
        return next(c for c in self.components if c.options.get("label") == label or c.options.get("elem_id") == label or c.value == label)


@pytest.fixture
def panel(monkeypatch, tmp_path):
    module = load_plugin()
    gr = PanelGradio()
    monkeypatch.setattr(module, "gr", gr)
    plugin = module.AwsWorkerBridgePlugin()
    config = {"connections": [connection_record()]}
    monkeypatch.setattr(plugin, "_load_config", lambda: config)
    monkeypatch.setattr(type(plugin), "_config_path", property(lambda self: tmp_path / "worker_config.json"))
    monkeypatch.setattr(plugin, "_save_config", lambda value: None)
    monkeypatch.setattr(plugin, "_event_processing_can_keep_awake", lambda config=None: False)
    monkeypatch.setattr(plugin, "_event_processing_keep_awake_text", lambda config=None: "Disabled")
    return types.SimpleNamespace(module=module, plugin=plugin, config=config, gr=gr)


def set_running(panel, running=True):
    panel.plugin._worker_thread = types.SimpleNamespace(is_alive=lambda: running)


def context(panel, worker_id=80):
    return panel.plugin._connection_context(connection_record(worker_id))


def confirmed(panel, worker_id=80):
    panel.plugin._record_heartbeat_result(context(panel, worker_id))


def test_version_matches_metadata_and_loaded_build(panel, tmp_path):
    module, plugin = panel.module, panel.plugin
    metadata = Path(module.__file__).with_name("plugin_info.json")
    assert plugin.version == json.loads(metadata.read_text())["version"] == module.PLUGIN_VERSION
    assert module.USER_AGENT == f"{module.PLUGIN_ID}/{plugin.version}"
    replacement = tmp_path / "plugin_info.json"
    replacement.write_text('{"version":"9.8.7"}')
    assert module._read_plugin_version(replacement) == ("9.8.7", "")
    assert plugin.version == module.PLUGIN_VERSION != "9.8.7"
    plugin.create_ui(None)
    assert panel.gr.find(f"## {module.PLUGIN_NAME} | v{module.PLUGIN_VERSION}")


@pytest.mark.parametrize("content", [None, "{", "[]", "{}", '{"version":3}', '{"version":"<script>"}', '\ufffd'])
def test_bad_version_is_nonfatal_and_does_not_expose_metadata(panel, tmp_path, monkeypatch, content):
    path = tmp_path / "plugin_info.json"
    if content is not None:
        path.write_text(content)
    version, warning = panel.module._read_plugin_version(path)
    assert version == "unavailable" and "Plugin version unavailable" in warning
    assert str(path) not in warning and "<script>" not in warning
    monkeypatch.setattr(panel.module, "PLUGIN_VERSION", version)
    monkeypatch.setattr(panel.module, "PLUGIN_VERSION_WARNING", warning)
    plugin = panel.module.AwsWorkerBridgePlugin()
    assert plugin.version == "unavailable" and warning in plugin._log_text()


@pytest.mark.parametrize("state", ["Unpaired", "Stopped", "Starting", "Ready", "Standby", "Waiting for WanGP", "Working", "Stopping", "Connection Error"])
def test_status_states(panel, state):
    plugin = panel.plugin
    if state == "Unpaired":
        panel.config.clear()
    elif state != "Stopped":
        set_running(panel)
        if state not in {"Starting", "Connection Error"}:
            confirmed(panel)
        if state == "Standby":
            plugin._last_idle_mode = "standby"
        elif state == "Waiting for WanGP":
            plugin.state = types.SimpleNamespace(value={"gen": {"in_progress": True}})
        elif state in {"Working", "Stopping"}:
            plugin._active_job_id, plugin._active_job_context = 42, context(panel)
            plugin._active_job_status = {"phase": "uploading", "progress": 97, "status": "PRIVATE_PROMPT"}
            if state == "Stopping":
                plugin._stop_event.set()
        elif state == "Connection Error":
            plugin._record_heartbeat_result(context(panel), panel.module.requests.Timeout("PRIVATE_TOKEN"))
    assert plugin._worker_status_snapshot()["state"] == state
    summary = plugin._worker_status_summary()
    assert f"State: {state}" in summary and "Snapshot refreshed:" in summary
    assert "PRIVATE_TOKEN" not in summary and "PRIVATE_PROMPT" not in summary
    if state in {"Working", "Stopping"}:
        assert "Active job: 42 | Project: 180 | Worker: 80" in summary
        assert "Stage: Uploading | Progress: 97%" in summary


def test_partial_connectivity_and_error_recovery_are_per_pairing(panel):
    panel.config["connections"].append(connection_record(81))
    set_running(panel)
    confirmed(panel)
    confirmed(panel, 81)
    error = panel.module.requests.HTTPError("SECRET", response=types.SimpleNamespace(status_code=503))
    panel.plugin._record_heartbeat_result(context(panel, 81), error)
    snapshot = panel.plugin._worker_status_snapshot()
    assert snapshot["state"] == "Ready"
    assert [c["confirmed"] for c in snapshot["connections"]] == [True, False]
    assert snapshot["connections"][1]["last_success_at"] is not None
    assert "2 paired | 1 heartbeat confirmed | 1 connection errors" in panel.plugin._worker_status_summary()
    confirmed(panel)
    assert panel.plugin._worker_status_snapshot()["connections"][1]["error"]
    confirmed(panel, 81)
    assert all(c["confirmed"] and not c["error"] for c in panel.plugin._worker_status_snapshot()["connections"])


def test_old_heartbeat_does_not_make_restarted_worker_ready(panel):
    confirmed(panel)
    panel.plugin._worker_started_at = panel.module.time.monotonic() + 1
    set_running(panel)
    assert panel.plugin._worker_status_snapshot()["state"] == "Starting"
    assert "Last successful heartbeat:" in panel.plugin._worker_status_summary()


def test_render_is_read_only_and_does_not_use_native_status_text(panel, monkeypatch):
    plugin = panel.plugin
    set_running(panel)
    confirmed(panel)
    plugin._active_job_id, plugin._active_job_context = 42, context(panel)
    plugin._active_job_status = {"phase": "SECRET_PHASE", "status": "PRIVATE_PROMPT", "progress": 30}
    plugin._idle_since_at = 456
    monkeypatch.setattr(panel.module.requests, "post", lambda *a, **k: pytest.fail("render sent a request"))
    monkeypatch.setattr(plugin, "_idle_mode", lambda *a, **k: pytest.fail("render mutated idle state"))
    monkeypatch.setattr(plugin, "_ffmpeg_processing_probe", lambda *a, **k: pytest.fail("render probed dependencies"))
    summary = plugin._worker_status_summary()
    assert "Stage: Processing | Progress: 30%" in summary
    assert "SECRET" not in summary and "PRIVATE" not in summary
    assert plugin._idle_since_at == 456
    plugin._active_job_status = {"phase": "failed", "status": "job failed"}
    plugin._active_job_id, plugin._active_job_context = None, None
    assert plugin._worker_status_snapshot()["state"] == "Ready"


@pytest.mark.parametrize("paired", [False, True])
def test_setup_defaults_and_shared_callback_output_wiring(panel, paired):
    if not paired:
        panel.config.clear()
    plugin, gr = panel.plugin, panel.gr
    plugin.create_ui(None)
    setup = gr.find("Project Pairing Setup")
    development = gr.find("Development Connection Settings")
    assert setup.options["open"] is (not paired)
    assert development.options["open"] is False and development.parent is setup
    assert gr.find("Pairing Code").options["type"] == "password"
    assert gr.find("Pair Project & Start Worker").parent is setup
    assert gr.find("midom-worker-status").parent is None
    assert gr.find("midom-worker-status").kind == "Markdown"
    assert gr.find("Refresh Status & Log").parent is not setup
    for button in (c for c in gr.components if c.events):
        for event in button.events:
            expected = 15 if button.value == "Forget All Local Pairings" else 14
            assert len(event["outputs"]) == expected
            assert event["outputs"][13] is gr.find("midom-worker-status")
    refresh = gr.find("Refresh Status & Log").events[0]
    result = refresh["fn"]()
    assert len(result) == len(refresh["outputs"]) == 14
    assert "State:" in result[13]
    # Refresh does not overwrite setup inputs or reopen its accordion.
    assert all(c not in refresh["outputs"] for c in (setup, gr.find("API Base URL"), gr.find("Machine Name"), gr.find("Pairing Code")))


@pytest.mark.parametrize("path", ["pair_success", "pair_error", "start", "stop", "toggle_error", "capabilities", "capabilities_error", "wake", "forget", "disconnect", "keep_awake", "keep_awake_error", "disable_keep_awake"])
def test_action_callbacks_return_matching_summary_and_controls(panel, monkeypatch, path):
    plugin, gr = panel.plugin, panel.gr
    plugin.create_ui(None)
    if path.startswith("pair_"):
        monkeypatch.setattr(plugin, "_start_worker", lambda session: "started")
        def pair(*args):
            if path == "pair_error":
                raise gr.Error("bad pairing")
        monkeypatch.setattr(plugin, "_pair_worker", pair)
        result = plugin._pair_update_capabilities_and_start_worker(None, "url", "code", "name", False, False)
    elif path in {"start", "stop", "toggle_error"}:
        if path == "stop":
            set_running(panel)
        def action(*args, **kwargs):
            if path == "toggle_error":
                raise gr.Error("failed action")
            return "state changed"
        monkeypatch.setattr(plugin, "_start_worker", action)
        monkeypatch.setattr(plugin, "_stop_worker", action)
        result = plugin._toggle_worker(None)
    elif path in {"capabilities", "capabilities_error", "wake"}:
        monkeypatch.setattr(plugin, "_worker_is_idle_standby", lambda: path == "wake")
        def capabilities():
            if path == "capabilities_error":
                raise panel.module.requests.ConnectionError("offline")
            return "updated"
        monkeypatch.setattr(plugin, "_update_capabilities", capabilities)
        monkeypatch.setattr(plugin, "_wake_worker_for_active_use", lambda: "awake")
        result = plugin._wake_or_update_capabilities_ui()
    elif path == "forget":
        monkeypatch.setattr(plugin, "_forget_local_pairing", lambda: "forgotten")
        result = plugin._forget_local_pairing_ui()
        assert result[-1] == gr.update(value="")
    elif path == "disconnect":
        monkeypatch.setattr(plugin, "_disconnect_selected_pairing", lambda selected: "disconnected")
        result = plugin._disconnect_selected_pairing_ui("worker-80")
    elif path in {"keep_awake", "keep_awake_error"}:
        def enable(*args):
            if path == "keep_awake_error":
                raise gr.Error("unavailable")
            return "enabled"
        monkeypatch.setattr(plugin, "_set_event_processing_keep_awake", enable)
        result = plugin._event_keep_awake_ui("4h")
    else:
        monkeypatch.setattr(plugin, "_disable_event_processing_keep_awake", lambda: "disabled")
        result = plugin._event_keep_awake_disable_ui()
    assert len(result) == (15 if path == "forget" else 14)
    assert "State:" in result[13]


def test_action_labels_and_pending_stop(panel):
    plugin = panel.plugin
    assert plugin._worker_toggle_label() == "Start Worker"
    assert plugin._worker_toggle_interactive()
    panel.config.clear()
    assert not plugin._worker_toggle_interactive()
    panel.config["connections"] = [connection_record()]
    set_running(panel)
    assert plugin._worker_toggle_label() == "Stop Worker"
    plugin._active_job_id = 42
    assert plugin._worker_toggle_label() == "Stop Worker & Cancel Current Job"
    plugin._stop_event.set()
    updates = plugin._ui_result()
    assert updates[1] == panel.gr.update(value="Stopping...", interactive=False)
    assert updates[2]["interactive"] is False
    plugin._stop_event.clear()
    plugin._active_job_id = None
    plugin._idle_since_at = panel.module.time.monotonic() - panel.module.IDLE_STANDBY_AFTER_SECONDS - 1
    assert plugin._capabilities_button_label() == "Wake Worker"


def test_stop_still_cancels_active_native_job(panel):
    canceled = []
    panel.plugin._active_job = types.SimpleNamespace(done=False, cancel=lambda: canceled.append(True))
    panel.plugin._stop_worker()
    assert canceled == [True] and panel.plugin._stop_event.is_set() and panel.plugin._wake_event.is_set()


def test_finished_inference_still_shows_upload_and_active_job(panel):
    set_running(panel)
    plugin = panel.plugin
    plugin._active_job_id, plugin._active_job_context = 42, context(panel)
    plugin._active_job = types.SimpleNamespace(done=True)
    plugin._active_job_status = {"phase": "uploading"}
    assert plugin._current_active_job_id() is None
    assert plugin._worker_status_snapshot()["state"] == "Working"
    assert "Stage: Uploading" in plugin._worker_status_summary()
    assert plugin._worker_toggle_label() == "Stop Worker & Cancel Current Job"
    assert "active job 42" in plugin._connection_list_text()
    assert "active job None" not in plugin._connection_list_text()


def test_saving_pairings_prunes_only_removed_connection_health(panel):
    confirmed(panel)
    confirmed(panel, 81)
    panel.plugin._save_connections([connection_record()])
    assert list(panel.plugin._connection_health) == ["worker-80"]


@pytest.mark.parametrize("result", ["success", "timeout", "http_503", "bad_json"])
def test_real_heartbeat_path_records_results_without_changing_errors(panel, monkeypatch, result):
    plugin, module = panel.plugin, panel.module
    set_running(panel)
    plugin._headers = lambda connection: {}
    error = None
    if result == "timeout":
        error = module.requests.Timeout("PRIVATE_TOKEN")
    elif result == "http_503":
        error = module.requests.HTTPError("PRIVATE_TOKEN", response=types.SimpleNamespace(status_code=503))
    elif result == "bad_json":
        error = ValueError("PRIVATE_TOKEN")
    def post(*args, **kwargs):
        if result == "timeout":
            raise error
        def check():
            if result == "http_503":
                raise error
        def decode():
            if result == "bad_json":
                raise error
            return {}
        return types.SimpleNamespace(status_code=503 if result == "http_503" else 200, raise_for_status=check, json=decode)
    monkeypatch.setattr(module.requests, "post", post)
    if error is not None:
        with pytest.raises(type(error)) as raised:
            plugin._heartbeat_once(context(panel))
        assert raised.value is error
    else:
        plugin._heartbeat_once(context(panel))
    health = plugin._worker_status_snapshot()["connections"][0]
    assert health["confirmed"] is (result == "success")
    assert bool(health["error"]) is (result != "success")
    assert "PRIVATE_TOKEN" not in plugin._worker_status_summary()
    assert not plugin._stop_event.is_set()


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("code", [401, 403])
def test_heartbeat_authorization_behavior_is_preserved(panel, monkeypatch, active, code):
    plugin, module = panel.plugin, panel.module
    panel.config["connections"].append(connection_record(81))
    confirmed(panel)
    confirmed(panel, 81)
    plugin._headers = lambda connection: {}
    if active:
        plugin._active_job_id, plugin._active_job_context = 42, context(panel)
    def save(value):
        panel.config.clear()
        panel.config.update(value)
    monkeypatch.setattr(plugin, "_save_config", save)
    response = module.requests.Response()
    response.status_code, response._content = code, b'{"message":"revoked"}'
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: response)
    if active:
        with pytest.raises(module.requests.HTTPError):
            plugin._heartbeat_once(context(panel))
        assert len(plugin._connection_records()) == 2
        assert plugin._connection_health["worker-80"]["error"] == f"Heartbeat failed (HTTP {code})"
    else:
        plugin._heartbeat_once(context(panel))
        assert [c["connection_id"] for c in plugin._connection_records()] == ["worker-81"]
        assert "worker-80" not in plugin._connection_health
    assert not plugin._stop_event.is_set()
    assert not plugin._connection_health["worker-81"]["error"]
