"""Audio result paths are filesystem identities, not gallery/artifact strings."""

import hashlib
import ntpath
import os
import shutil
import types
from pathlib import Path, PureWindowsPath

import pytest

from test_fish_audio_drama3 import Runtime, Session, bridge, fish_job, wav_bytes
from test_fish_audio_reviewable_runtime import ReviewablePipeline, v2_job


def result_for(generated, artifacts):
    return types.SimpleNamespace(generated_files=generated, artifacts=tuple(
        types.SimpleNamespace(path=path, media_type="audio") for path in artifacts))


@pytest.mark.parametrize("case", ["identical", "relative_absolute", "absolute_relative", "duplicates_in_generated", "dot_segments"])
def test_audio_aliases_resolve_from_session_root(bridge, monkeypatch, tmp_path, case):
    _, plugin = bridge
    root = tmp_path / "wangp"
    output = root / "outputs" / "speech.wav"
    output.parent.mkdir(parents=True)
    output.write_bytes(wav_bytes(0.1))
    caller = tmp_path / "unrelated_cwd"
    caller.mkdir()
    monkeypatch.chdir(caller)
    plugin._generation_session = types.SimpleNamespace(_root=root)
    relative, absolute = "outputs/speech.wav", str(output)
    paths = {
        "identical": ([relative], [relative]),
        "relative_absolute": ([relative], [absolute]),
        "absolute_relative": ([absolute], [relative]),
        "duplicates_in_generated": ([relative, relative, absolute], [absolute]),
        "dot_segments": (["outputs/../outputs/./speech.wav"], [absolute]),
    }
    logs = []
    plugin._log = lambda message, **kwargs: logs.append(message)
    assert plugin._result_audio_files(result_for(*paths[case])) == [absolute]
    assert any("raw_path=" in message and "canonical_path=" in message for message in logs)
    assert any("Deduplicated WanGP audio output alias" in message for message in logs)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_existing_filesystem_aliases_are_one_output(bridge, tmp_path, kind):
    _, plugin = bridge
    output, alias = tmp_path / "speech.wav", tmp_path / "alias.wav"
    output.write_bytes(wav_bytes(0.1))
    try:
        if kind == "symlink":
            alias.symlink_to(output)
        else:
            os.link(output, alias)
    except OSError as exc:
        pytest.skip(f"Filesystem alias unavailable: {exc}")
    files = plugin._result_audio_files(result_for([str(output)], [str(alias)]), base_path=tmp_path)
    assert files == [str(output.resolve())]


@pytest.mark.parametrize("case", ["same_basename_and_bytes", "posix_case"])
def test_distinct_files_are_not_content_or_basename_deduplicated(bridge, tmp_path, case):
    _, plugin = bridge
    first = tmp_path / "a" / "speech.wav"
    second = tmp_path / "b" / "speech.wav" if case != "posix_case" else first.with_name("Speech.wav")
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(wav_bytes(0.1))
    second.write_bytes(first.read_bytes())
    if first.samefile(second):
        pytest.skip("This filesystem does not distinguish the case-only paths")
    assert plugin._result_audio_files(result_for([str(first)], [str(second)]), base_path=tmp_path) == [str(first), str(second)]


@pytest.mark.parametrize("generated,artifact,count", [
    (r"outputs\Speech.wav", "c:/WANGP/outputs/speech.WAV", 1),
    (r"C:\WanGP\outputs\Speech.wav", r"c:\wangp\OUTPUTS\speech.wav", 1),
    (r"outputs\a\speech.wav", r"C:\WanGP\outputs\b\speech.wav", 2),
    (r"C:\WanGP\outputs\speech.wav", r"D:\WanGP\outputs\speech.wav", 2),
])
def test_windows_collector_case_and_separator_rules_on_any_host(bridge, monkeypatch, generated, artifact, count):
    module, plugin = bridge

    class WindowsPath(PureWindowsPath):
        def resolve(self):
            return type(self)(ntpath.normpath(str(self)))

        def samefile(self, other):
            raise OSError("Synthetic Windows paths have no host filesystem identity")

        def is_file(self):
            return False

    monkeypatch.setattr(module, "Path", WindowsPath)
    monkeypatch.setattr(module, "os", types.SimpleNamespace(path=ntpath))
    files = plugin._result_audio_files(result_for([generated], [artifact]), base_path=r"C:\WanGP")
    assert len(files) == count
    assert files[0] == str((WindowsPath(r"C:\WanGP") / generated).resolve())


@pytest.mark.parametrize("model", ["fish_drama3", "fish_drama3_int4"])
@pytest.mark.parametrize("fmt", ["wav", "mp3"])
@pytest.mark.parametrize("case", ["relative_absolute", "absolute_relative", "duplicates_in_generated", "hardlink", "symlink", "windows_case", "distinct_outputs", "distinct_artifact", "extra_artifact", "wrong_client"])
def test_fish_output_identity_through_upload_and_completion(bridge, monkeypatch, tmp_path, model, fmt, case):
    if case == "windows_case" and os.name != "nt":
        pytest.skip("Native Windows filesystem/FFmpeg integration requires Windows")
    module, plugin = bridge
    root = tmp_path / "session"
    output = root / "outputs" / "speech_seed123.wav"
    output.parent.mkdir(parents=True)
    recording = wav_bytes(0.3)
    other = root / "different" / output.name
    other.parent.mkdir()
    other.write_bytes(recording)
    if case in {"hardlink", "symlink"}:
        output.write_bytes(recording)
        other.unlink()
        try:
            if case == "hardlink":
                os.link(output, other)
            else:
                other.symlink_to(output)
        except OSError as exc:
            pytest.skip(f"Filesystem alias unavailable: {exc}")
    relative = "outputs/speech_seed123.wav"
    generated, artifacts = [relative], [str(output)]
    if case == "absolute_relative":
        generated, artifacts = [str(output)], [relative]
    elif case == "duplicates_in_generated":
        generated = [relative, str(output), relative]
    elif case in {"hardlink", "symlink"}:
        artifacts = [str(other)]
    elif case == "windows_case":
        generated, artifacts = [relative.replace("/", "\\")], [str(output).upper().replace("\\", "/")]
    elif case == "distinct_outputs":
        generated.append(str(other))
    elif case == "distinct_artifact":
        artifacts = [str(other)]
    elif case == "extra_artifact":
        artifacts.append(str(output))
    caller = tmp_path / "caller"
    caller.mkdir()
    # A same-named CWD file must never override the authoritative session output.
    misleading = caller / relative
    misleading.parent.mkdir()
    misleading.write_bytes(wav_bytes(0.1))
    monkeypatch.chdir(caller)
    runtime = Runtime(ReviewablePipeline(), lambda: output.write_bytes(recording))
    session = Session(runtime, generated, artifact_files=artifacts, root=root)
    original_submit = session.submit_task

    def submit(settings, callbacks=None):
        handle = original_submit(settings, callbacks)
        result = handle.result()
        if case == "wrong_client":
            result.artifacts[0].client_id = "another_job"
        return types.SimpleNamespace(result=lambda: result)

    session.submit_task = submit
    plugin._get_generation_session = lambda: session
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    uploaded = []

    def upload(url, **kwargs):
        data = kwargs["files"]["file"][1].read()
        assert kwargs["data"]["sha256"] == hashlib.sha256(data).hexdigest()
        uploaded.append(data)
        return types.SimpleNamespace(status_code=200, json=lambda: {"artifact_id": 12, "file_id": 34, "artifact_index": 0})

    monkeypatch.setattr(module.requests, "post", upload)
    connection = types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7")
    plugin._run_job(connection, v2_job(model=model, fmt=fmt))
    if case in {"distinct_outputs", "distinct_artifact", "extra_artifact", "wrong_client"}:
        assert plugin._fish_updates[-1][0] == "fail", plugin._fish_updates
        assert uploaded == []
        assert not any(status == "complete" for status, _ in plugin._fish_updates)
    else:
        assert plugin._fish_updates[-1][0] == "complete", plugin._fish_updates[-1]
        assert len(uploaded) == 1
        assert sum(status == "complete" for status, _ in plugin._fish_updates) == 1
        fish = plugin._fish_updates[-1][1]["generation_metadata"]["fish_audio"]
        assert fish["artifacts"][0]["sha256"] == hashlib.sha256(uploaded[0]).hexdigest()
        assert fish["execution_evidence"]["client_id"] == session.submitted[0]["client_id"]
        if fmt == "wav":
            assert uploaded[0] == recording
    assert output.read_bytes() == recording
    assert other.read_bytes() == recording


def test_v1_duplicate_generated_paths_still_deliver_one_unchanged_wav(bridge, monkeypatch, tmp_path):
    module, plugin = bridge
    path = tmp_path / "speech.wav"
    recording = wav_bytes(0.1)
    session = Session(Runtime(ReviewablePipeline(), lambda: path.write_bytes(recording)), ["speech.wav", str(path)], root=tmp_path)
    plugin._get_generation_session = lambda: session
    plugin._wait_for_wangp_result = lambda connection, job_id, handle, callbacks: handle.result()
    plugin._ffprobe_binary = lambda: shutil.which("ffprobe")
    uploaded = []

    def upload(url, **kwargs):
        uploaded.append(kwargs["files"]["file"][1].read())
        return types.SimpleNamespace(status_code=200, json=lambda: {"artifact_id": 12, "file_id": 34, "artifact_index": 0})

    monkeypatch.setattr(module.requests, "post", upload)
    plugin._run_job(types.SimpleNamespace(api_base_url="http://midom", worker_id=7, connection_id="worker-7"), fish_job())
    assert plugin._fish_updates[-1][0] == "complete", plugin._fish_updates[-1]
    assert uploaded == [recording]
