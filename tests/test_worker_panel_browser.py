"""Opt-in real Gradio/Chromium panel check; no Midom server or inference is used."""

import importlib.util
import os
import re
import socket
import sys
import types
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("MIDOM_RUN_WORKER_PANEL_BROWSER") != "1",
    reason="Set MIDOM_RUN_WORKER_PANEL_BROWSER=1 in an isolated Gradio 5.29/Playwright environment.",
)


@pytest.mark.parametrize("paired", [False, True])
def test_real_worker_panel(tmp_path, monkeypatch, paired):
    import gradio as gr
    from playwright.sync_api import expect, sync_playwright

    assert gr.__version__ == "5.29.0", "Run this file separately with WanGP's pinned Gradio version."
    for name in ("shared", "shared.api", "shared.utils", "shared.utils.plugins"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["shared.api"].init = lambda **kwargs: None
    sys.modules["shared.utils.plugins"].WAN2GPPlugin = object
    spec = importlib.util.spec_from_file_location("midom_panel_browser", Path(__file__).resolve().parents[1] / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin = module.AwsWorkerBridgePlugin()
    record = {
        "connection_id": "worker-80", "worker_id": 80, "org_id": 10, "project_id": 180,
        "paired_user_id": 20, "api_base_url": "https://midom.example", "worker_token": "PRIVATE_TOKEN",
        "machine_name": "Test GPU", "capabilities_revision": 1,
    }
    config = {"connections": [record]} if paired else {}
    monkeypatch.setattr(plugin, "_load_config", lambda: config)
    monkeypatch.setattr(plugin, "_event_processing_can_keep_awake", lambda config=None: False)
    monkeypatch.setattr(plugin, "_event_processing_keep_awake_text", lambda config=None: "Disabled")
    monkeypatch.setattr(plugin, "_headers", lambda *a: pytest.fail("UI attempted an authorized request"))
    running = {"value": False}
    plugin._worker_thread = types.SimpleNamespace(is_alive=lambda: running["value"], join=lambda **kwargs: None)

    def start(session):
        running["value"] = True
        plugin._stop_event.clear()
        plugin._worker_started_at = module.time.monotonic()
        plugin._record_heartbeat_result(plugin._connection_context(record))
        return "Started test worker"

    def pair(*args):
        config.update(connections=[record])

    monkeypatch.setattr(plugin, "_start_worker", start)
    monkeypatch.setattr(plugin, "_pair_worker", pair)
    monkeypatch.setattr(plugin, "_update_capabilities", lambda: "Updated test capabilities")
    with gr.Blocks(analytics_enabled=False) as app:
        with gr.Tabs():
            with gr.Tab(module.PLUGIN_NAME):
                plugin.create_ui(None)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    errors = []
    try:
        app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True, show_error=True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{port}")
                expect(page.get_by_role("heading", name=f"{module.PLUGIN_NAME} | v{module.PLUGIN_VERSION}")).to_be_visible()
                status = page.locator("#midom-worker-status")
                expect(status).to_contain_text(re.compile("State: Stopped" if paired else "State: Unpaired"))
                code = page.get_by_label("Pairing Code", exact=True)
                if paired:
                    expect(code).not_to_be_visible()
                    page.get_by_text("Project Pairing Setup", exact=True).click()
                expect(code).to_be_visible()
                expect(page.get_by_label("Allow HTTP for localhost development only", exact=True)).not_to_be_visible()
                page.get_by_label("API Base URL", exact=True).fill("https://midom.example")
                page.get_by_label("Machine Name", exact=True).fill("Test GPU")
                code.fill("PRIVATE_PAIRING_CODE")
                page.get_by_role("button", name="Refresh Status & Log", exact=True).click()
                expect(code).to_have_value("PRIVATE_PAIRING_CODE")
                if paired:
                    page.get_by_role("button", name="Start Worker", exact=True).click()
                else:
                    page.get_by_role("button", name="Pair Project & Start Worker", exact=True).click()
                expect(status).to_contain_text(re.compile("State: Ready"))
                page.get_by_role("button", name="Refresh Midom Capabilities", exact=True).click()
                expect(page.get_by_role("button", name="Stop Worker", exact=True)).to_be_enabled()

                plugin._active_job_id, plugin._active_job_context = 42, plugin._connection_context(record)
                plugin._active_job = types.SimpleNamespace(done=True)
                plugin._active_job_status = {"phase": "uploading", "progress": 97, "status": "PRIVATE_PROMPT"}
                page.get_by_role("button", name="Refresh Status & Log", exact=True).click()
                expect(status).to_contain_text(re.compile("State: Working.*Stage: Uploading", re.S))
                assert "PRIVATE" not in status.inner_text()
                page.get_by_text("Project Pairing Setup", exact=True).click()
                screenshot_dir = Path(os.environ.get("MIDOM_PANEL_SCREENSHOTS", str(tmp_path)))
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                for name, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
                    page.set_viewport_size({"width": width, "height": height})
                    expect(page.get_by_role("button", name="Stop Worker & Cancel Current Job", exact=True)).to_be_visible()
                    assert status.evaluate("node => node.scrollHeight <= node.clientHeight + 1")
                    page.screenshot(path=str(screenshot_dir / f"worker-panel-{name}-paired-{paired}.png"), full_page=True)
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                    overflowing = page.get_by_role("button").filter(has_text=re.compile(
                        r"^(Stop Worker|Refresh Midom Capabilities|Refresh Status & Log|Disconnect Selected Pairing|"
                        r"Forget All Local Pairings|Keep Awake|Disable Keep Awake)"
                    )).evaluate_all(
                        "buttons => buttons.filter(b => b.getClientRects().length && b.scrollWidth > b.clientWidth + 1)"
                        ".map(b => ({text: b.textContent, width: b.clientWidth, contentWidth: b.scrollWidth}))"
                    )
                    assert not overflowing, overflowing
                page.get_by_role("button", name="Stop Worker & Cancel Current Job", exact=True).click()
                expect(status).to_contain_text(re.compile("State: Stopping"))
                expect(page.get_by_role("button", name="Stopping...", exact=True)).to_be_disabled()
                assert plugin._stop_event.is_set()
                running["value"] = False
                plugin._active_job_id, plugin._active_job_context, plugin._active_job = None, None, None
                page.get_by_role("button", name="Refresh Status & Log", exact=True).click()
                expect(status).to_contain_text(re.compile("State: Stopped"))
                expect(page.get_by_role("button", name="Start Worker", exact=True)).to_be_enabled()
                assert not errors
            finally:
                browser.close()
    finally:
        app.close()
