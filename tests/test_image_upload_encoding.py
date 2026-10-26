import hashlib
import io
import types
from contextlib import nullcontext

import pytest
from PIL import Image

from test_qwen_image_21_support import load_plugin_module


def encoded_image(format, *, mode="RGB", size=(32, 24), alpha=None):
    image = Image.new(mode, size, (30, 90, 150, 255) if mode == "RGBA" else (30, 90, 150))
    if alpha is not None:
        image.putalpha(alpha)
        image.putpixel((0, 0), (30, 90, 150, 0))
        image.putpixel((1, 0), (30, 90, 150, 255))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


@pytest.fixture
def upload(monkeypatch):
    module = load_plugin_module()
    plugin = object.__new__(module.AwsWorkerBridgePlugin)
    plugin._ensure_job_flow_enabled = lambda: None
    plugin._headers = lambda connection: {"Authorization": "Bearer test-lease"}
    plugin._log = lambda *args, **kwargs: None
    connection = types.SimpleNamespace(api_base_url="https://midom.test", worker_id=7, connection_id="worker-7")
    uploads = []
    def post(url, headers, data, files, timeout):
        filename, reader, mime_type = files["file"]
        uploads.append({"url": url, "headers": headers, "data": data,
                        "filename": filename, "mime_type": mime_type, "bytes": reader.read()})
        return types.SimpleNamespace(status_code=200, json=lambda: {
            "artifact_id": 12, "file_id": 34, "artifact_index": 0})
    monkeypatch.setattr(module.requests, "post", post)
    return types.SimpleNamespace(module=module, plugin=plugin, connection=connection, uploads=uploads)


@pytest.mark.parametrize("format,mime,filename,expected", [
    ("JPEG", "image/jpeg", "infographic.png", "infographic.jpg"),
    ("JPEG", "image/jpeg", "infographic.webp", "infographic.jpg"),
    ("PNG", "image/png", "image.jpg", "image.png"),
    ("PNG", "image/png", "image.webp", "image.png"),
    ("WEBP", "image/webp", "image.png", "image.webp"),
    ("WEBP", "image/webp", "image.jpeg", "image.webp"),
    ("JPEG", "image/jpeg", "image.jpeg", "image.jpeg"),
    ("JPEG", "image/jpeg", "image.JPEG", "image.jpeg"),
    ("PNG", "image/png", "image.PNG", "image.png"),
    ("WEBP", "image/webp", "image.webp", "image.webp"),
])
def test_upload_names_and_mime_follow_bytes_without_reencoding(upload, tmp_path, monkeypatch, format, mime, filename, expected):
    f = upload
    path = tmp_path / filename
    data = encoded_image(format)
    path.write_bytes(data)
    def no_normalization(*args):
        pytest.fail("Filename correction alone must not re-encode the image")
    monkeypatch.setattr(f.plugin, "_normalize_artifact_dimensions", no_normalization)
    validation = []
    result = f.plugin._upload_artifact(f.connection, 123, str(path), 0, "32x24", output_validation=validation)
    posted = f.uploads[0]
    assert result == {"artifact_id": 12, "file_id": 34, "artifact_index": 0}
    assert posted["filename"] == posted["data"]["filename"] == expected
    assert posted["mime_type"] == posted["data"]["mime_type"] == mime
    assert posted["bytes"] == data == path.read_bytes()
    assert posted["data"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert posted["headers"]["Authorization"] == "Bearer test-lease"
    assert validation[0]["mime_type"] == mime and validation[0]["upload_filename"] == expected
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
def test_generic_transparent_image_preserves_original_bytes_and_alpha(upload, tmp_path, format):
    f = upload
    path = tmp_path / "transparent.jpg"
    data = encoded_image(format, mode="RGBA", alpha=128)
    path.write_bytes(data)
    f.plugin._upload_artifact(f.connection, 123, str(path), 0, "32x24")
    assert f.uploads[0]["bytes"] == data
    with Image.open(io.BytesIO(f.uploads[0]["bytes"])) as image:
        assert image.mode == "RGBA" and image.getchannel("A").getextrema() == (0, 255)


@pytest.mark.parametrize("required", [False, True])
def test_rgba_png_with_wrong_extension_is_not_reencoded(upload, tmp_path, monkeypatch, required):
    f = upload
    path = tmp_path / "character.webp"
    data = encoded_image("PNG", mode="RGBA", alpha=128)
    path.write_bytes(data)
    monkeypatch.setattr(f.plugin, "_normalize_artifact_dimensions", lambda *args: pytest.fail("No normalization is needed"))
    validation = []
    f.plugin._upload_artifact(f.connection, 123, str(path), 0, "32x24",
                             required_mime_type="image/png" if required else None,
                             required_color_mode="RGBA", require_meaningful_alpha=True,
                             output_validation=validation)
    assert f.uploads[0]["filename"] == "character.png"
    assert f.uploads[0]["bytes"] == data == path.read_bytes()
    assert validation[0]["normalization_applied"] is False
    assert validation[0]["final_sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_truncated_data_is_rejected_before_upload(upload, tmp_path, format):
    path = tmp_path / "broken.png"
    data = encoded_image(format)
    path.write_bytes(data[:len(data) // 2])
    with pytest.raises(ValueError, match="could not be decoded"):
        upload.plugin._upload_artifact(upload.connection, 123, str(path), 0)
    assert not upload.uploads


def test_readable_jpeg_header_does_not_hide_truncated_pixel_data(upload, tmp_path):
    path = tmp_path / "truncated.png"
    path.write_bytes(encoded_image("JPEG", size=(128, 96))[:-20])
    with Image.open(path) as image:
        assert image.format == "JPEG" and image.size == (128, 96)
    with pytest.raises(ValueError, match="could not be decoded"):
        upload.plugin._upload_artifact(upload.connection, 123, str(path), 0)
    assert not upload.uploads


@pytest.mark.parametrize("format", ["GIF", "TIFF", "BMP"])
def test_unsupported_encoding_cannot_hide_behind_png_extension(upload, tmp_path, format):
    path = tmp_path / "unsupported.png"
    path.write_bytes(encoded_image(format))
    with pytest.raises(ValueError, match="Unsupported generated artifact encoding"):
        upload.plugin._upload_artifact(upload.connection, 123, str(path), 0)
    assert not upload.uploads


@pytest.mark.parametrize("name", ["output.txt", "output"])
def test_existing_extension_allowlist_is_not_relaxed(upload, tmp_path, name):
    path = tmp_path / name
    path.write_bytes(encoded_image("PNG"))
    with pytest.raises(ValueError, match="extension"):
        upload.plugin._upload_artifact(upload.connection, 123, str(path), 0)
    assert not upload.uploads


def test_required_png_uses_existing_conversion_even_when_jpeg_is_misnamed_png(upload, tmp_path):
    f = upload
    path = tmp_path / "wrong.png"
    data = encoded_image("JPEG")
    path.write_bytes(data)
    f.plugin._upload_artifact(f.connection, 123, str(path), 0, "32x24",
                             required_mime_type="image/png", required_color_mode="RGB")
    posted = f.uploads[0]
    assert posted["filename"].endswith(".png") and posted["mime_type"] == "image/png"
    assert posted["bytes"] != data and path.read_bytes() == data
    with Image.open(io.BytesIO(posted["bytes"])) as image:
        assert image.format == "PNG" and image.mode == "RGB" and image.size == (32, 24)


@pytest.mark.parametrize("wrong", ["encoding", "dimensions", "color_mode"])
def test_failed_normalization_cannot_bypass_final_contract_checks(upload, tmp_path, monkeypatch, wrong):
    f = upload
    path = tmp_path / "source.jpg"
    path.write_bytes(encoded_image("JPEG"))
    normalized = tmp_path / "normalized.png"
    image = Image.new("L" if wrong == "color_mode" else "RGB", (16, 16) if wrong == "dimensions" else (32, 24))
    image.save(normalized, format="JPEG" if wrong == "encoding" else "PNG")
    monkeypatch.setattr(f.plugin, "_normalize_artifact_dimensions", lambda *args: normalized)
    with pytest.raises(ValueError, match="MIME|dimensions|color mode"):
        f.plugin._upload_artifact(f.connection, 123, str(path), 0, "32x24",
                                 required_mime_type="image/png", required_color_mode="RGB")
    assert not f.uploads


def test_final_artifact_byte_limit_remains_enforced(upload, tmp_path, monkeypatch):
    f = upload
    path = tmp_path / "too-large.png"
    data = encoded_image("JPEG")
    path.write_bytes(data)
    monkeypatch.setattr(f.module, "MAX_IMAGE_BYTES", len(data) - 1)
    with pytest.raises(ValueError, match="size is outside allowed bounds"):
        f.plugin._upload_artifact(f.connection, 123, str(path), 0)
    assert not f.uploads


@pytest.mark.parametrize("format,mode", [("JPEG", "RGB"), ("PNG", "RGB"), ("PNG", "RGBA")])
def test_alpha_contract_still_rejects_rgb_and_all_opaque_rgba(upload, tmp_path, format, mode):
    f = upload
    path = tmp_path / "opaque.png"
    path.write_bytes(encoded_image(format, mode=mode))
    with pytest.raises(ValueError, match="meaningful native alpha"):
        f.plugin._upload_artifact(f.connection, 123, str(path), 0, "32x24", required_mime_type="image/png",
                                 required_color_mode="RGBA", require_meaningful_alpha=True)
    assert not f.uploads


@pytest.mark.parametrize("format", ["JPEG", "WEBP"])
def test_typed_png_upload_still_rejects_non_png_bytes(upload, tmp_path, format):
    f = upload
    path = tmp_path / "typed.png"
    path.write_bytes(encoded_image(format))
    with pytest.raises(ValueError, match="PNG bytes"):
        f.plugin._upload_media_processing_artifact(f.connection, 123, {
            "path": str(path), "artifact_index": 0, "role": "restored_crop_image", "mime_type": "image/png"}, {})
    assert not f.uploads


def test_upload_uses_validated_snapshot_even_if_file_changes_before_request(upload, tmp_path):
    f = upload
    path = tmp_path / "snapshot.png"
    data = encoded_image("JPEG")
    path.write_bytes(data)
    def headers(connection):
        path.write_bytes(b"changed after validation")
        return {"Authorization": "Bearer test-lease"}
    f.plugin._headers = headers
    f.plugin._upload_artifact(f.connection, 123, str(path), 0)
    assert f.uploads[0]["bytes"] == data
    assert f.uploads[0]["data"]["sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("status", [400, 403, 410])
def test_upload_rejection_is_not_swallowed_or_retried(upload, tmp_path, monkeypatch, status):
    f = upload
    path = tmp_path / "result.png"
    path.write_bytes(encoded_image("JPEG"))
    calls = []
    def reject(*args, **kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(status_code=status, json=lambda: {"detail": "lease or artifact rejected"})
    monkeypatch.setattr(f.module.requests, "post", reject)
    with pytest.raises(ValueError, match=str(status)):
        f.plugin._upload_artifact(f.connection, 123, str(path), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("format,mime,extension", [("PNG", "image/png", ".png"),
                                                ("JPEG", "image/jpeg", ".jpg"),
                                                ("WEBP", "image/webp", ".webp")])
def test_generic_png_claim_delivers_actual_encoding_through_completion(upload, tmp_path, format, mime, extension):
    f = upload
    plugin = f.plugin
    job = {"job_id": 123, "media_type": "image", "model_id": f.module.SENSENOVA_MODEL_ID,
           "prompt": "An infographic", "generation": {"reference_mode": "none"},
           "output": {"format": "png", "count": 1, "width": 1280, "height": 720}, "inputs": []}
    plugin._validate_job_scope = lambda *args: None
    plugin._resolve_job_resolution = lambda *args, **kwargs: "1280x720"
    plugin._apply_accelerator_profile = lambda *args: "standard"
    claim = plugin._validate_image_job(job, 123)
    assert "_midom_output_mime_type" not in claim
    path = tmp_path / "generated.png"
    data = encoded_image(format, size=(1280, 720))
    path.write_bytes(data)
    plugin._callbacks_for_job = lambda *args: None
    plugin._reset_idle_backoff_for_active_use = lambda *args, **kwargs: None
    plugin._sensenova_high_resolution_budget = lambda *args: nullcontext()
    plugin._get_generation_session = lambda: types.SimpleNamespace(submit_task=lambda *args, **kwargs: object())
    plugin._wait_for_wangp_result = lambda *args: types.SimpleNamespace(success=True, cancelled=False,
                                                                      generated_files=[str(path)], artifacts=[], errors=[])
    updates = []
    plugin._post_job_update = lambda connection, job_id, status, payload: updates.append((status, payload))
    plugin._run_job(f.connection, job)
    assert updates[-1][0] == "complete", updates[-1]
    assert len(f.uploads) == 1 and f.uploads[0]["mime_type"] == mime
    assert f.uploads[0]["filename"].endswith(extension) and f.uploads[0]["bytes"] == data
