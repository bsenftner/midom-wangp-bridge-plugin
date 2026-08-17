import hashlib
import ipaddress
import json
import math
import mimetypes
import queue
import re
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import gradio as gr
import requests
from PIL import Image

from shared.api import init as init_wangp_session
from shared.utils.plugins import WAN2GPPlugin


PLUGIN_ID = "Midom-at-AWS-worker-bridge"
PLUGIN_NAME = "Midom Remote Worker"
CONFIG_FILENAME = "worker_config.json"
LOCAL_WANGP_BUSY_MESSAGE = "Local WanGP is busy with another UI or Deepy generation; queued Midom jobs will wait."
MAX_PROMPT_CHARS = 4000
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_AUDIO_BYTES = 52_428_800
MAX_VIDEO_BYTES = 209_715_200
MAX_EVENT_VIDEO_INPUT_BYTES = 1_073_741_824
MAX_EVENT_VIDEO_OUTPUT_BYTES = 536_870_912
MAX_EVENT_VIDEO_DURATION_SECONDS = 600
MAX_EVENT_ASSET_PIXELS = 33_177_600
MIN_EVENT_ASSET_DIMENSION = 16
EVENT_VIDEO_PROCESSOR_ID = "event_video_ffmpeg_processor"
EVENT_VIDEO_PROCESSING_TASK = "event_video_processing"
EVENT_VIDEO_OUTPUT_PROFILE = "mobile_public_720p"
EVENT_VIDEO_BUMPER_SECONDS = 2
EVENT_VIDEO_H264_ENCODER = "libx264"
EVENT_VIDEO_X264_CRF = 21
EVENT_VIDEO_X264_MAXRATE = "8M"
EVENT_VIDEO_X264_BUFSIZE = "16M"
EVENT_VIDEO_TIMEOUT_BASE_SECONDS = 120
EVENT_VIDEO_TIMEOUT_MULTIPLIER = 12
EVENT_VIDEO_TIMEOUT_MAX_SECONDS = 7200
STORYBOARD_FFMPEG_PROCESSOR_ID = "storyboard_ffmpeg_processor"
STORYBOARD_FFMPEG_PROCESSING_TASK = "storyboard_ffmpeg_processing"
MAX_STORYBOARD_VIDEO_INPUT_BYTES = 1_073_741_824
MAX_STORYBOARD_VIDEO_OUTPUT_BYTES = 536_870_912
MAX_STORYBOARD_AUDIO_INPUT_BYTES = 104_857_600
MAX_STORYBOARD_IMAGE_INPUT_BYTES = MAX_IMAGE_BYTES
MAX_STORYBOARD_VIDEO_DURATION_SECONDS = 30 * 60
STORYBOARD_TIMEOUT_BASE_SECONDS = 120
STORYBOARD_TIMEOUT_MULTIPLIER = 12
STORYBOARD_TIMEOUT_MAX_SECONDS = 7200
STORYBOARD_OUTPUT_FPS = 30
MAX_AUDIO_DURATION_SECONDS = 120
DRAMABOX_MAX_DURATION_SECONDS = 120
DRAMABOX_MAX_DIALOGUE_SPEAKERS = 8
DRAMABOX_DEFAULT_DURATION_MULTIPLIER = 1.1
DRAMABOX_MIN_DURATION_MULTIPLIER = 0.5
DRAMABOX_MAX_DURATION_MULTIPLIER = 3.0
CHATTERBOX_MAX_PROMPT_CHARS = 2000
CHATTERBOX_MAX_PROMPT_CHARS_FG = 350
CHATTERBOX_MAX_PROMPT_CHARS_PER_SPLIT = 300
MAX_LONGCAT_VIDEO_DURATION_SECONDS = 20
MAX_LTX_VIDEO_DURATION_SECONDS = 20
LTX_CONTROL_VIDEO_DURATION_MODE = "fit_to_control_video_audio_max_20"
LTX_CONTROL_VIDEO_MODES = {
    "human_motion": "PVG",
    "human_motion_aligned": "OVG",
    "depth": "DVG",
    "canny_edges": "EVG",
}
MIN_SVI_VIDEO_DURATION_SECONDS = 1
DEFAULT_SVI_VIDEO_DURATION_SECONDS = 5
MAX_SVI_VIDEO_DURATION_SECONDS = 30
VIDEO_FIT_TO_AUDIO_DURATION_MODE = "fit_to_driving_audio_max_20"
LONGCAT_DURATION_MODE = VIDEO_FIT_TO_AUDIO_DURATION_MODE
LTX_DURATION_MODE = VIDEO_FIT_TO_AUDIO_DURATION_MODE
SVI_DURATION_MODE = "fixed_seconds"
MIN_REFERENCE_AUDIO_SECONDS = 10
MAX_REFERENCE_AUDIO_SECONDS = 30
MIN_REFERENCE_AUDIO_SAMPLE_RATE_HZ = 16000
SEEDVC_MODEL_ID = "seedvc_voice_replacement"
SEEDVC_METHOD_ONE_SPEAKER = "seedvc_one_speaker"
SEEDVC_RUNTIME_MODE_SPEECH = "speech_v1"
SEEDVC_MIN_REFERENCE_AUDIO_SECONDS = 5
SEEDVC_RECOMMENDED_REFERENCE_AUDIO_SECONDS = 10
SEEDVC_MAX_REFERENCE_AUDIO_SECONDS = 25
SEEDVC_MAX_SOURCE_AUDIO_SECONDS = 120
MAX_STEPS = 40
MAX_CONTROL_IMAGES = 1
DEFAULT_STEPS = 20
POLL_INTERVAL_SECONDS = 5
IDLE_BACKOFF_AFTER_SECONDS = 40 * 60
IDLE_STANDBY_AFTER_SECONDS = 2 * 60 * 60
IDLE_BACKOFF_INTERVAL_SECONDS = 60
IDLE_STANDBY_HEARTBEAT_SECONDS = 10 * 60
EVENT_KEEP_AWAKE_DURATION_SECONDS = {
    "4h": 4 * 60 * 60,
    "8h": 8 * 60 * 60,
    "12h": 12 * 60 * 60,
}
JOB_KEEPALIVE_SECONDS = 60
JOB_KEEPALIVE_RETRY_SECONDS = 15
JOB_WAIT_LOG_SECONDS = 10
SIBLING_BUSY_MESSAGE = "Shared local WanGP engine is busy with another paired project."
REQUEST_TIMEOUT_SECONDS = 30
PAIRING_TIMEOUT_SECONDS = 15
UPLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
USER_AGENT = f"{PLUGIN_ID}/0.1.0"
VERBOSE_LOGGING = True
UI_LOG_LINES = 300
HEARTBEAT_LOG_SECONDS = 60
IDLE_POLL_LOG_SECONDS = 30
MP3_ENCODER_CACHE_SECONDS = 60
MAX_OUTPUTS = 6
QWEN_LAYERED_MAX_OUTPUTS = 9
QWEN_MULTI_ANGLE_TOOL_ID = "qwen_image_edit_2511_multiple_angles"
QWEN_MULTI_ANGLE_TOOL_DISPLAY_NAME = "Generate Alternate View"
QWEN_MULTI_ANGLE_TOOL_VERSION = "1"
QWEN_MULTI_ANGLE_BASE_MODEL_ID = "qwen_image_edit_plus2_20B"
QWEN_MULTI_ANGLE_LORA_DIR = "qwen"
QWEN_MULTI_ANGLE_LORA_FILENAME = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
QWEN_MULTI_ANGLE_LORA_SHA256 = "42426ded4e25fd22879d9e198b857556445ef4ca56e8da3246d0345155bb6765"
QWEN_MULTI_ANGLE_LORA_LICENSE = "Apache 2.0"
QWEN_MULTI_ANGLE_DEFAULT_VIEW_CHANGE_STRENGTH = "standard"
QWEN_MULTI_ANGLE_VIEW_CHANGE_STRENGTHS = {
    "subtle": "0.75",
    "standard": "0.9",
    "strong": "1.05",
    "maximum": "1.15",
}
QWEN_MULTI_ANGLE_ALLOWED_ACCELERATOR_PROFILE_IDS = {
    "standard",
    "qwen_edit_2511_lightning_8",
    "qwen_edit_v1_lightning_8",
}
SHARED_DEV_NETWORK = ipaddress.ip_network("100.64.0.0/10")
CONTROL_MODE_DEFINITIONS = {
    "pose": {"display_name": "Human pose", "description": "Use the control image to guide human pose."},
    "depth": {"display_name": "Depth", "description": "Use the control image to guide scene depth."},
    "edges": {"display_name": "Canny edges", "description": "Use the control image edges as structure."},
    "shapes": {"display_name": "Shapes / sketch", "description": "Use the control image shapes as structure."},
    "recolorize": {"display_name": "Recolorize", "description": "Use the control image for recolorization."},
    "raw": {"display_name": "Raw control image", "description": "Use an already prepared control image directly."},
}


def _image_max_outputs_for_model(model_id: str) -> int:
    if str(model_id or "") == "qwen_image_layered_20B":
        return QWEN_LAYERED_MAX_OUTPUTS
    return MAX_OUTPUTS
CONTROL_MODE_WANGP_CODES = {
    "pose": "PV",
    "depth": "DV",
    "edges": "EV",
    "shapes": "SV",
    "recolorize": "CV",
    "raw": "V",
}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_AUDIO_INPUT_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg"}
ALLOWED_AUDIO_OUTPUT_MIME_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav"}
ALLOWED_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg"}
ALLOWED_AUDIO_OUTPUT_SUFFIXES = {".mp3", ".wav"}
SVI_VIDEO_MODEL_ID = "i2v_2_2_Enhanced_Lightning_v2_svi2pro"
SVI_SPEED_PROFILE_ID = "enhanced_lightning_v2_8_step"
LTX23_VIDEO_MODEL_ID = "ltx2_22B_1_1"
LTX25_DISTILLED_VIDEO_MODEL_ID = "ltx2_25_22B_distilled"
LTX_VIDEO_MODEL_IDS = {LTX23_VIDEO_MODEL_ID, LTX25_DISTILLED_VIDEO_MODEL_ID}
ALLOWED_VIDEO_MODEL_TYPES = {"longcat_avatar_v1_5", *LTX_VIDEO_MODEL_IDS, SVI_VIDEO_MODEL_ID}
ALLOWED_CONTROL_VIDEO_MIME_TYPES = {"video/mp4"}
ALLOWED_VIDEO_INPUT_MIME_TYPES = ALLOWED_IMAGE_MIME_TYPES | ALLOWED_AUDIO_INPUT_MIME_TYPES | ALLOWED_CONTROL_VIDEO_MIME_TYPES
ALLOWED_VIDEO_OUTPUT_MIME_TYPES = {"video/mp4"}
ALLOWED_VIDEO_SUFFIXES = {".mp4"}
ALLOWED_EVENT_SOURCE_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime"}
ALLOWED_STORYBOARD_SOURCE_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime", "video/webm"}
ALLOWED_STORYBOARD_AUDIO_CONTAINER_MIME_TYPES = ALLOWED_AUDIO_INPUT_MIME_TYPES | {"video/mp4", "video/quicktime", "video/webm"}
ALLOWED_STORYBOARD_MATTE_MIME_TYPES = {"image/png", "image/jpeg"}
ALLOWED_EVENT_OVERLAY_MIME_TYPES = {"image/png"}
ALLOWED_EVENT_BUMPER_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
QWEN_MULTI_ANGLE_AZIMUTHS = {
    "front": "front view",
    "front_left": "front-left quarter view",
    "left": "left side view",
    "back_left": "back-left quarter view",
    "back": "back view",
    "back_right": "back-right quarter view",
    "right": "right side view",
    "front_right": "front-right quarter view",
}
QWEN_MULTI_ANGLE_ELEVATIONS = {
    "low": "low-angle shot",
    "eye_level": "eye-level shot",
    "elevated": "elevated shot",
    "high": "high-angle shot",
}
QWEN_MULTI_ANGLE_DISTANCES = {
    "close": "close-up",
    "medium": "medium shot",
    "wide": "wide shot",
    "full_body": "wide shot",
}
ALLOWED_EVENT_INPUT_KINDS = {"source_video", "overlay_png", "bumper_image"}
STORYBOARD_FFMPEG_OPERATION_TYPES = {
    "multicam_card_overlay_take",
    "multicam_card_pass_through_take",
    "multicam_card_trim_take",
    "multicam_final_assembly",
    "multicam_optimize_video",
    "optimize_video",
    "replace_video_soundtrack",
    "multicam_seekable_mp4",
    "multicam_ai_video_take_prepare",
    "mediastoryboard_card_pass_through_take",
    "mediastoryboard_card_local_video_take",
    "mediastoryboard_card_trim_take",
    "mediastoryboard_card_edge_trim_take",
}
STORYBOARD_TRIM_OPERATION_TYPES = {
    "multicam_card_trim_take",
    "mediastoryboard_card_trim_take",
    "mediastoryboard_card_edge_trim_take",
}
STORYBOARD_SINGLE_VIDEO_OPERATION_TYPES = {
    "multicam_card_overlay_take",
    "multicam_card_pass_through_take",
    "multicam_card_trim_take",
    "multicam_optimize_video",
    "optimize_video",
    "replace_video_soundtrack",
    "multicam_seekable_mp4",
    "multicam_ai_video_take_prepare",
    "mediastoryboard_card_pass_through_take",
    "mediastoryboard_card_local_video_take",
    "mediastoryboard_card_trim_take",
    "mediastoryboard_card_edge_trim_take",
}
STORYBOARD_VIDEO_INPUT_KINDS = {
    "source_video",
    "video",
    "input_video",
    "take_video",
    "card_video",
    "segment_video",
    "overlay_video",
    "control_video",
}
STORYBOARD_IMAGE_INPUT_KINDS = {
    "image",
    "source_image",
    "overlay_image",
    "overlay_png",
    "poster_image",
    "start_image",
    "end_image",
}
STORYBOARD_AUDIO_INPUT_KINDS = {
    "audio",
    "source_audio",
    "soundtrack_audio",
    "driving_audio",
    "narration_audio",
}
ALLOWED_STORYBOARD_INPUT_KINDS = STORYBOARD_VIDEO_INPUT_KINDS | STORYBOARD_IMAGE_INPUT_KINDS | STORYBOARD_AUDIO_INPUT_KINDS
PROMPT_PROCESSING_MODE_CHOICES = ("G", "PG", "FG")
PROMPT_PROCESSING_MODES = set(PROMPT_PROCESSING_MODE_CHOICES)
DEFAULT_PROMPT_PROCESSING_MODE = "FG"
CHATTERBOX_MODEL_ID = "chatterbox"
DRAMABOX_MODEL_ID = "dramabox_audio"
STABLE_AUDIO3_MUSIC_MODEL_ID = "stable_audio3_small_music"
STABLE_AUDIO3_SFX_MODEL_ID = "stable_audio3_small_sfx"
STABLE_AUDIO3_MODEL_IDS = {STABLE_AUDIO3_MUSIC_MODEL_ID, STABLE_AUDIO3_SFX_MODEL_ID}
STABLE_AUDIO3_WANGP_MODEL_TYPES = {
    STABLE_AUDIO3_MUSIC_MODEL_ID: "stable_audio3_small",
    STABLE_AUDIO3_SFX_MODEL_ID: "stable_audio3_small_sfx",
}
STABLE_AUDIO3_SAMPLE_SOLVERS = ("pingpong", "euler", "dpmpp", "rk4")
STABLE_AUDIO3_MAX_DURATION_SECONDS = 120
STABLE_AUDIO3_MAX_NEGATIVE_PROMPT_CHARS = 2000
ACE_STEP15_MUSIC_MODEL_ID = "ace_step_v1_5_music"
ACE_STEP15_WANGP_MODEL_TYPE = "ace_step_v1_5"
ACE_STEP15_INSTRUMENTAL_LYRICS = "[Instrumental]"
ACE_STEP15_MIN_DURATION_SECONDS = 5
ACE_STEP15_DEFAULT_DURATION_SECONDS = 30
ACE_STEP15_MAX_DURATION_SECONDS = 120
ACE_STEP15_MAX_MUSIC_CAPTION_CHARS = 2000
ACE_STEP15_MAX_LYRICS_CHARS = 32
ACE_STEP15_BPM_MIN = 30
ACE_STEP15_BPM_MAX = 300
ACE_STEP15_TIME_SIGNATURE_VALUES = (2, 3, 4, 6)
ACE_STEP15_LANGUAGES = (
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
    "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
    "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
    "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
    "unknown",
)
CHATTERBOX_LANGUAGES = (
    "ar",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "sw",
    "tr",
    "zh",
)
CHATTERBOX_CONTROL_LIMITS = {
    "exaggeration": {"type": "float", "default": 0.5, "min": 0.25, "max": 2.0},
    "pace": {"type": "float", "default": 0.5, "min": 0.2, "max": 1.0},
}
MIME_EXTENSION = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SEED_FILENAME_RE = re.compile(r"(?:^|[_-])seed(-?\d+)(?:[_\-.]|$)", re.IGNORECASE)
ALLOWED_RESOLUTIONS = {
    "768x768",
    "1024x1024",
    "1280x720",
    "720x1280",
}
LONGCAT_VIDEO_RESOLUTIONS = {"832x480", "1280x720", "720x1280"}
LONGCAT_VIDEO_FPS = 25
LTX_VIDEO_RESOLUTIONS = {"1280x720", "720x1280"}
LTX_OVERSCAN_RENDER_RESOLUTIONS = {
    LTX23_VIDEO_MODEL_ID: {
        "1280x720": "1280x768",
        "720x1280": "768x1280",
    },
    LTX25_DISTILLED_VIDEO_MODEL_ID: {
        "1280x720": "1280x768",
        "720x1280": "768x1280",
    },
}
LTX_CONTROL_VIDEO_PROFILES = (
    {"profile_id": "landscape_16_9", "width": 1280, "height": 720, "fps": 24},
    {"profile_id": "portrait_9_16", "width": 720, "height": 1280, "fps": 24},
)
LTX_VIDEO_FPS = 24
SVI_VIDEO_RESOLUTIONS = {"1280x720", "720x1280"}
SVI_VIDEO_FPS = 16
LTX_VIDEO_SYNC_OMNINFT_PROFILE_ID = "omninft_rl_lora_sync"
LTX_VIDEO_SYNC_OMNINFT_LORA = "omninft-ltx2.3-22b-rl-lora-r32.safetensors"
JOB_FLOW_ENABLED = True

MODEL_CAPABILITY_OVERRIDES = {
    "flux2_dev": {"family": "flux2", "display_name": "Flux 2 Dev", "image_reference": True, "control": False, "max_reference_images": 3},
    "pi_flux2": {"family": "flux2", "display_name": "pi-FLUX.2", "image_reference": True, "control": False, "max_reference_images": 3},
    "flux2_klein_4b": {"family": "flux2", "display_name": "Flux 2 Klein 4B", "image_reference": True, "control": False, "max_reference_images": 3},
    "flux2_klein_9b": {"family": "flux2", "display_name": "Flux 2 Klein 9B", "image_reference": True, "control": False, "max_reference_images": 3},
    "qwen_image_20B": {"family": "qwen", "display_name": "Qwen Image 20B", "image_reference": False, "control": False, "max_reference_images": 0},
    "qwen_image_edit_20B": {"family": "qwen", "display_name": "Qwen Image Edit 20B", "image_reference": True, "control": False, "max_reference_images": 3},
    "qwen_image_edit_plus_20B": {"family": "qwen", "display_name": "Qwen Image Edit Plus 20B", "image_reference": True, "control": True, "max_reference_images": 3, "control_modes": ["pose", "depth", "shapes", "recolorize", "raw"]},
    "qwen_image_edit_plus2_20B": {"family": "qwen", "display_name": "Qwen Image Edit Plus2 20B", "image_reference": True, "control": True, "max_reference_images": 3, "control_modes": ["pose", "depth", "shapes", "recolorize", "raw"]},
    "qwen_image_layered_20B": {
        "family": "qwen",
        "display_name": "Qwen Image Layered 20B",
        "image_reference": False,
        "control": True,
        "layer_decomposition": True,
        "source_echo_output": True,
        "ordered_layer_outputs": True,
        "max_reference_images": 0,
        "control_modes": ["raw"],
        "output_roles": ["source_echo", "decomposition_layer"],
    },
    "z_image": {"family": "z_image", "display_name": "Z-Image Turbo", "image_reference": False, "control": False, "max_reference_images": 0},
    "z_image_base": {"family": "z_image", "display_name": "Z-Image Base", "image_reference": False, "control": False, "max_reference_images": 0},
    "z_image_control": {"family": "z_image", "display_name": "Z-Image Control", "image_reference": False, "control": True, "max_reference_images": 0, "control_modes": ["pose", "depth", "edges", "raw"]},
    "z_image_control2": {"family": "z_image", "display_name": "Z-Image Control2", "image_reference": False, "control": True, "max_reference_images": 0, "control_modes": ["pose", "depth", "edges", "raw"]},
    "z_image_control2_1": {"family": "z_image", "display_name": "Z-Image Control2.1", "image_reference": False, "control": True, "max_reference_images": 0, "control_modes": ["pose", "depth", "edges", "raw"]},
}
ALLOWED_MODEL_TYPES = set(MODEL_CAPABILITY_OVERRIDES.keys())
ALLOWED_AUDIO_MODEL_TYPES = {
    "index_tts2",
    CHATTERBOX_MODEL_ID,
    DRAMABOX_MODEL_ID,
    SEEDVC_MODEL_ID,
    *STABLE_AUDIO3_MODEL_IDS,
    ACE_STEP15_MUSIC_MODEL_ID,
}

QWEN_EDIT_MODELS = {
    "qwen_image_edit_20B",
    "qwen_image_edit_plus_20B",
    "qwen_image_edit_plus2_20B",
    "qwen_image_layered_20B",
}
QWEN_TEXT_MODELS = {"qwen_image_20B"}
ACCELERATOR_PROFILE_DEFINITIONS = [
    {
        "profile_id": "qwen_2512_lightning_8",
        "display_name": "Fast - 8 steps",
        "description": "Use WanGP's installed Qwen Lightning accelerator for faster text-to-image generation.",
        "quality_tier": "fast",
        "steps": 8,
        "model_ids": sorted(QWEN_TEXT_MODELS),
        "lora_dir": "qwen",
        "lora_filenames": ["Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"],
        "loras_multipliers": "1",
        "guidance_scale": 1,
    },
    {
        "profile_id": "qwen_2512_lightning_4",
        "display_name": "Very fast - 4 steps",
        "description": "Use WanGP's installed Qwen Lightning accelerator for fastest text-to-image generation.",
        "quality_tier": "very_fast",
        "steps": 4,
        "model_ids": sorted(QWEN_TEXT_MODELS),
        "lora_dir": "qwen",
        "lora_filenames": ["Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"],
        "loras_multipliers": "1",
        "guidance_scale": 1,
    },
    {
        "profile_id": "qwen_lightning_v1_1_8",
        "display_name": "Fast - 8 steps",
        "description": "Use WanGP's installed Qwen Lightning accelerator for faster text-to-image generation.",
        "quality_tier": "fast",
        "steps": 8,
        "model_ids": sorted(QWEN_TEXT_MODELS),
        "lora_dir": "qwen",
        "lora_filenames": ["Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors"],
        "loras_multipliers": "1",
        "guidance_scale": 1,
    },
    {
        "profile_id": "qwen_lightning_v1_0_4",
        "display_name": "Very fast - 4 steps",
        "description": "Use WanGP's installed Qwen Lightning accelerator for fastest text-to-image generation.",
        "quality_tier": "very_fast",
        "steps": 4,
        "model_ids": sorted(QWEN_TEXT_MODELS),
        "lora_dir": "qwen",
        "lora_filenames": ["Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"],
        "loras_multipliers": "1",
        "guidance_scale": 1,
    },
    {
        "profile_id": "qwen_edit_2511_lightning_8",
        "display_name": "Fast - 8 steps",
        "description": "Use WanGP's installed Qwen Edit Lightning accelerator for faster image editing.",
        "quality_tier": "fast",
        "steps": 8,
        "model_ids": sorted(QWEN_EDIT_MODELS),
        "lora_dir": "qwen",
        "lora_filenames": ["Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"],
        "loras_multipliers": "1",
        "guidance_scale": 1,
    },
    {
        "profile_id": "qwen_edit_2511_lightning_4",
        "display_name": "Very fast - 4 steps",
        "description": "Use WanGP's installed Qwen Edit Lightning accelerator for fastest image editing.",
        "quality_tier": "very_fast",
        "steps": 4,
        "model_ids": sorted(QWEN_EDIT_MODELS),
        "lora_dir": "qwen",
        "lora_filenames": ["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"],
        "loras_multipliers": "1",
        "guidance_scale": 1,
    },
    {
        "profile_id": "qwen_edit_v1_lightning_8",
        "display_name": "Fast - 8 steps",
        "description": "Use WanGP's installed Qwen Edit Lightning accelerator for faster image editing.",
        "quality_tier": "fast",
        "steps": 8,
        "model_ids": sorted(QWEN_EDIT_MODELS),
        "lora_dir": "qwen",
        "lora_filenames": ["Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"],
        "loras_multipliers": "1",
        "guidance_scale": 1,
    },
    {
        "profile_id": "qwen_edit_v1_lightning_4",
        "display_name": "Very fast - 4 steps",
        "description": "Use WanGP's installed Qwen Edit Lightning accelerator for fastest image editing.",
        "quality_tier": "very_fast",
        "steps": 4,
        "model_ids": sorted(QWEN_EDIT_MODELS),
        "lora_dir": "qwen",
        "lora_filenames": ["Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"],
        "loras_multipliers": "1",
        "guidance_scale": 1,
    },
]
ACCELERATOR_PROFILE_BY_ID = {
    str(profile["profile_id"]): profile for profile in ACCELERATOR_PROFILE_DEFINITIONS
}


@dataclass(frozen=True)
class ConnectionContext:
    connection_id: str
    api_base_url: str
    worker_id: int
    worker_token: str
    org_id: int
    project_id: int
    paired_user_id: int
    machine_name: str
    capabilities_revision: int
    token_expires_at: str
    allow_insecure_local_dev: bool = False
    allow_insecure_lan_dev: bool = False

    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        token = str(self.worker_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


class LocalProcessJob:
    def __init__(self):
        self.done = False
        self.cancelled = False
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def set_process(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._process = process
            if self.cancelled and process.poll() is None:
                process.terminate()

    def clear_process(self, process: subprocess.Popen) -> None:
        with self._lock:
            if self._process is process:
                self._process = None

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def mark_done(self) -> None:
        with self._lock:
            self.done = True


def _add_prompt_type_letter(value: str, letter: str) -> str:
    value = str(value or "")
    letter = str(letter or "")
    return value if not letter or letter in value else f"{value}{letter}"


class AwsWorkerBridgePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PLUGIN_NAME
        self.version = "0.3.0"
        self.description = "Connects this local WanGP workstation to Midom as a scoped project media worker."
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._active_job = None
        self._active_job_id = None
        self._active_job_context = None
        self._cancel_requested_by_midom = False
        self._last_heartbeat_log_at = 0.0
        self._last_idle_poll_log_at = 0.0
        self._last_local_wangp_busy_log_at = 0.0
        self._last_heartbeat_at = 0.0
        self._last_candidate_poll_at = 0.0
        self._idle_since_at = None
        self._last_idle_mode = None
        self._next_connection_poll_index = 0
        self._wake_event = threading.Event()
        self._generation_session = None
        self._active_job_status = {}
        self._log_lock = threading.Lock()
        self._log_lines = deque(maxlen=UI_LOG_LINES)
        self._mp3_encoder_cache = None
        self._mp3_encoder_cache_at = 0.0
        self._ffmpeg_probe_cache = None
        self._ffmpeg_probe_cache_at = 0.0
        self._ffmpeg_probe_log_signature = None
        self._lora_hash_cache = {}

    def _log(self, message: str, *, force: bool = False) -> None:
        if not VERBOSE_LOGGING and not force:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{PLUGIN_NAME} {timestamp}] {message}"
        with self._log_lock:
            self._log_lines.append(line)
        print(line, flush=True)

    def _log_text(self) -> str:
        with self._log_lock:
            lines = list(self._log_lines)
        if lines:
            return "\n".join(lines)
        return self._status_text()

    def _job_label(self, job_id: Any = None) -> str:
        job_id = self._active_job_id if job_id is None else job_id
        return f"job_id={job_id}" if job_id else "job_id=none"

    def setup_ui(self):
        self.request_component("state")
        self.add_tab(
            tab_id="aws_worker_bridge",
            label=PLUGIN_NAME,
            component_constructor=self.create_ui,
        )

    @property
    def _config_path(self) -> Path:
        return Path(__file__).resolve().parent / CONFIG_FILENAME

    def _load_config(self) -> dict[str, Any]:
        path = self._config_path
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as reader:
                data = json.load(reader)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_config(self, config: dict[str, Any]) -> None:
        path = self._config_path
        safe_config = dict(config)
        safe_config["allow_insecure_local_dev"] = bool(safe_config.get("allow_insecure_local_dev", False))
        safe_config["allow_insecure_lan_dev"] = bool(safe_config.get("allow_insecure_lan_dev", False))
        safe_config["event_processing_keep_awake_until"] = self._coerce_int(
            safe_config.get("event_processing_keep_awake_until"),
            0,
            0,
            4_102_444_800,
        )
        with path.open("w", encoding="utf-8") as writer:
            json.dump(safe_config, writer, indent=2)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _normalize_connection_record(self, record: dict[str, Any], config: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        if not isinstance(record, dict):
            return None
        config = config or {}
        worker_id = str(record.get("worker_id") or "").strip()
        worker_token = str(record.get("worker_token") or "").strip()
        api_base_url = str(record.get("api_base_url") or config.get("api_base_url") or "").strip().rstrip("/")
        if not worker_id or not worker_token or not api_base_url:
            return None
        try:
            worker_id_int = int(worker_id)
            org_id_int = int(record.get("org_id"))
            project_id_int = int(record.get("project_id"))
            paired_user_id_int = int(record.get("paired_user_id"))
        except (TypeError, ValueError):
            return None
        if min(worker_id_int, org_id_int, project_id_int, paired_user_id_int) <= 0:
            return None
        normalized = {
            "connection_id": str(record.get("connection_id") or f"worker-{worker_id}").strip(),
            "api_base_url": api_base_url,
            "worker_id": worker_id_int,
            "org_id": org_id_int,
            "project_id": project_id_int,
            "paired_user_id": paired_user_id_int,
            "machine_name": str(record.get("machine_name") or config.get("machine_name") or "").strip(),
            "worker_token": worker_token,
            "token_expires_at": str(record.get("token_expires_at") or "").strip(),
            "capabilities_revision": self._coerce_int(record.get("capabilities_revision"), 1, 1, 1_000_000),
            "allow_insecure_local_dev": bool(record.get("allow_insecure_local_dev", config.get("allow_insecure_local_dev", False))),
            "allow_insecure_lan_dev": bool(record.get("allow_insecure_lan_dev", config.get("allow_insecure_lan_dev", False))),
            "paired_at": self._coerce_int(record.get("paired_at"), int(time.time()), 1, 4_102_444_800),
        }
        if not normalized["connection_id"]:
            normalized["connection_id"] = f"worker-{normalized['worker_id']}"
        return normalized

    def _connection_records(self, config: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        config = self._load_config() if config is None else config
        records = config.get("connections")
        if isinstance(records, list):
            normalized = []
            for record in records:
                normalized_record = self._normalize_connection_record(record, config)
                if normalized_record is not None:
                    normalized.append(normalized_record)
            return normalized
        legacy_record = self._normalize_connection_record(config, config)
        return [legacy_record] if legacy_record is not None else []

    def _config_with_connections(self, base_config: dict[str, Any], connections: list[dict[str, Any]]) -> dict[str, Any]:
        next_config = dict(base_config)
        normalized_connections = []
        for record in connections:
            normalized_record = self._normalize_connection_record(record, next_config)
            if normalized_record is not None:
                normalized_connections.append(normalized_record)
        if normalized_connections:
            primary = dict(normalized_connections[0])
            next_config.update(primary)
            next_config["connections"] = normalized_connections
            next_config["api_base_url"] = primary["api_base_url"]
            next_config["machine_name"] = primary.get("machine_name") or next_config.get("machine_name") or ""
            next_config["allow_insecure_local_dev"] = bool(primary.get("allow_insecure_local_dev", False))
            next_config["allow_insecure_lan_dev"] = bool(primary.get("allow_insecure_lan_dev", False))
        else:
            for key in (
                "connection_id",
                "worker_id",
                "org_id",
                "project_id",
                "paired_user_id",
                "worker_token",
                "token_expires_at",
                "capabilities_revision",
                "paired_at",
                "connections",
                "event_processing_keep_awake_until",
            ):
                next_config.pop(key, None)
        return next_config

    def _save_connections(self, connections: list[dict[str, Any]], base_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        config = self._config_with_connections(base_config or self._load_config(), connections)
        if self._connection_records(config):
            self._save_config(config)
        else:
            try:
                self._config_path.unlink()
            except FileNotFoundError:
                pass
            config = {}
        return config

    def _forget_connection_locally(self, connection_id: str) -> dict[str, Any]:
        config = self._load_config()
        remaining = [
            record
            for record in self._connection_records(config)
            if str(record.get("connection_id") or "") != str(connection_id)
        ]
        return self._save_connections(remaining, config)

    def _is_active_connection(self, connection: ConnectionContext) -> bool:
        active_context = self._active_job_context
        return (
            self._current_active_job_id() is not None
            and active_context is not None
            and str(active_context.connection_id) == str(connection.connection_id)
        )

    def _forget_unauthorized_connection(
        self,
        connection: ConnectionContext,
        operation: str,
        message: str = "",
    ) -> bool:
        if self._is_active_connection(connection):
            return False
        config = self._forget_connection_locally(connection.connection_id)
        remaining_count = len(self._connection_records(config))
        detail = f" message={message!r}" if str(message or "").strip() else ""
        self._log(
            f"{operation} rejected by Midom authorization; removed stale project pairing locally. "
            f"connection_id={connection.connection_id} worker_id={connection.worker_id}{detail} "
            f"remaining_pairings={remaining_count}.",
            force=True,
        )
        self._wake_event.set()
        if remaining_count <= 0:
            self._stop_event.set()
        return True

    def _connection_contexts(self, config: Optional[dict[str, Any]] = None) -> list[ConnectionContext]:
        return [self._connection_context(record) for record in self._connection_records(config)]

    def _connection_context(self, config: dict[str, Any]) -> ConnectionContext:
        worker_id = self._worker_id(config)
        worker_token = str(config.get("worker_token") or "").strip()
        if not worker_token:
            raise gr.Error("Pair this worker before starting it.")
        return ConnectionContext(
            connection_id=str(config.get("connection_id") or f"worker-{worker_id}"),
            api_base_url=self._configured_api_base_url(config),
            worker_id=worker_id,
            worker_token=worker_token,
            org_id=self._coerce_required_int(config, "org_id"),
            project_id=self._coerce_required_int(config, "project_id"),
            paired_user_id=self._coerce_required_int(config, "paired_user_id"),
            machine_name=str(config.get("machine_name") or "").strip(),
            capabilities_revision=self._coerce_int(config.get("capabilities_revision"), 1, 1, 1_000_000),
            token_expires_at=str(config.get("token_expires_at") or "").strip(),
            allow_insecure_local_dev=bool(config.get("allow_insecure_local_dev", False)),
            allow_insecure_lan_dev=bool(config.get("allow_insecure_lan_dev", False)),
        )

    def _headers(self, connection_or_config: Any) -> dict[str, str]:
        if isinstance(connection_or_config, ConnectionContext):
            return connection_or_config.headers()
        config = connection_or_config if isinstance(connection_or_config, dict) else {}
        token = str(config.get("worker_token") or "").strip()
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _is_loopback_host(hostname: Optional[str]) -> bool:
        hostname = str(hostname or "").strip().lower().rstrip(".")
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _is_private_lan_address(address: Any) -> bool:
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            return False
        return bool(address.is_private or address.is_link_local or address in SHARED_DEV_NETWORK)

    def _is_private_lan_dev_host(self, hostname: Optional[str]) -> bool:
        hostname = str(hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        try:
            address = ipaddress.ip_address(hostname)
            return self._is_private_lan_address(address)
        except ValueError:
            pass
        if hostname.endswith(".local") or "." not in hostname:
            return True
        try:
            resolved = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        except OSError:
            return False
        addresses = []
        for info in resolved:
            sockaddr = info[4]
            if not sockaddr:
                continue
            try:
                addresses.append(ipaddress.ip_address(str(sockaddr[0])))
            except ValueError:
                continue
        return bool(addresses) and all(self._is_private_lan_address(address) for address in addresses)

    def _normalize_api_base_url(
        self,
        api_base_url: str,
        allow_insecure_local_dev: bool = False,
        allow_insecure_lan_dev: bool = False,
    ) -> str:
        api_base_url = str(api_base_url or "").strip().rstrip("/")
        if not api_base_url:
            raise gr.Error("API Base URL is required.")
        parsed = urlparse(api_base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise gr.Error("API Base URL must be a complete http(s) URL.")
        if parsed.params or parsed.query or parsed.fragment:
            raise gr.Error("API Base URL must not include query strings, fragments, or URL parameters.")
        is_loopback = self._is_loopback_host(parsed.hostname)
        is_private_lan = self._is_private_lan_dev_host(parsed.hostname)
        insecure_dev_allowed = (allow_insecure_local_dev and is_loopback) or (allow_insecure_lan_dev and is_private_lan)
        if parsed.scheme != "https" and not insecure_dev_allowed:
            hostname = str(parsed.hostname or "").strip()
            raise gr.Error(
                "HTTPS is required for this API Base URL. To use HTTP during development, enable the matching "
                "localhost or private LAN / tailnet checkbox and make sure the host resolves to that kind of address. "
                f"Host={hostname!r}; localhost_http={bool(allow_insecure_local_dev)}; "
                f"private_lan_http={bool(allow_insecure_lan_dev)}; "
                f"is_loopback={is_loopback}; is_private_lan={is_private_lan}."
            )
        if parsed.username or parsed.password:
            raise gr.Error("API Base URL must not include credentials.")
        return api_base_url

    def _coerce_required_int(self, payload: dict[str, Any], key: str) -> int:
        try:
            value = int(payload.get(key))
        except (TypeError, ValueError):
            raise gr.Error(f"Pairing response did not include a valid {key}.")
        if value <= 0:
            raise gr.Error(f"Pairing response did not include a valid {key}.")
        return value

    @staticmethod
    def _midom_error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            for key in ("message", "detail", "error", "reason"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            detail = payload.get("detail")
            if isinstance(detail, list) and detail:
                return str(detail[0])
        text = str(response.text or "").strip()
        return text[:300] if text else response.reason

    def _raise_pairing_error(self, response: requests.Response) -> None:
        server_message = self._midom_error_message(response)
        if response.status_code == 400:
            message = (
                "Pairing failed. Pairing codes are one-time use and expire after 8 hours. "
                "Generate a new pairing code in Midom and try again."
            )
        elif response.status_code in {401, 403}:
            message = "Pairing failed because Midom rejected this request. Generate a new pairing code and try again."
        else:
            message = f"Pairing failed with Midom HTTP {response.status_code}."
        if server_message:
            message = f"{message} Midom said: {server_message}"
        raise gr.Error(message)

    @staticmethod
    def _friendly_exception_message(exc: BaseException) -> str:
        message = getattr(exc, "message", None)
        if not message and getattr(exc, "args", None):
            message = exc.args[0]
        message = str(message or exc).strip()
        return message.strip("'\"") if message else exc.__class__.__name__

    def _pair_worker(
        self,
        api_base_url: str,
        pairing_code: str,
        machine_name: str,
        allow_insecure_local_dev: bool,
        allow_insecure_lan_dev: bool,
    ) -> str:
        api_base_url = self._normalize_api_base_url(api_base_url, allow_insecure_local_dev, allow_insecure_lan_dev)
        pairing_code = pairing_code.strip()
        machine_name = str(machine_name or "").strip()
        if not pairing_code:
            raise gr.Error("Pairing Code is required.")
        if not machine_name:
            raise gr.Error("Machine Name is required.")
        if self._worker_thread is not None and self._worker_thread.is_alive():
            raise gr.Error("Stop the worker before pairing again.")

        self._log(f"Pairing worker with Midom at {api_base_url}; machine_name={machine_name!r}.")
        capabilities_started_at = time.monotonic()
        capabilities = self._capabilities()
        self._log(
            "Prepared capabilities for pairing; "
            f"models={len(capabilities.get('models') or [])} "
            f"media_processing={len(capabilities.get('media_processing') or [])} "
            f"elapsed_seconds={time.monotonic() - capabilities_started_at:.2f}."
        )
        response = requests.post(
            f"{api_base_url}/b1/media-workers/pair",
            json={
                "pairing_code": pairing_code,
                "plugin_id": PLUGIN_ID,
                "machine_name": machine_name,
                "capabilities": capabilities,
            },
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            timeout=PAIRING_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            self._log(f"Pairing rejected by Midom; http_status={response.status_code} message={self._midom_error_message(response)!r}.", force=True)
            self._raise_pairing_error(response)
        payload = response.json()
        if not isinstance(payload, dict):
            raise gr.Error("Pairing response was not a JSON object.")
        token = str(payload.get("worker_token") or "").strip()
        token_expires_at = str(payload.get("token_expires_at") or "").strip()
        if not token:
            raise gr.Error("Pairing response did not include a worker id and token.")
        if not token_expires_at:
            raise gr.Error("Pairing response did not include token_expires_at.")

        connection = {
            "api_base_url": api_base_url,
            "worker_id": self._coerce_required_int(payload, "worker_id"),
            "org_id": self._coerce_required_int(payload, "org_id"),
            "project_id": self._coerce_required_int(payload, "project_id"),
            "paired_user_id": self._coerce_required_int(payload, "paired_user_id"),
            "machine_name": str(payload.get("machine_name") or machine_name),
            "worker_token": token,
            "token_expires_at": token_expires_at,
            "capabilities_revision": self._coerce_int(payload.get("capabilities_revision"), 1, 1, 1_000_000),
            "allow_insecure_local_dev": bool(allow_insecure_local_dev),
            "allow_insecure_lan_dev": bool(allow_insecure_lan_dev),
            "paired_at": int(time.time()),
        }
        connection["connection_id"] = f"worker-{connection['worker_id']}"

        config = self._load_config()
        connections = self._connection_records(config)
        if connections:
            primary = connections[0]
            if str(primary.get("api_base_url") or "").rstrip("/") != api_base_url:
                raise gr.Error("Multiple project pairings must use the same Midom API Base URL in this first pass.")
            if int(primary.get("paired_user_id") or 0) != int(connection["paired_user_id"]):
                raise gr.Error("Multiple project pairings must belong to the same Midom user in this first pass.")
            if any(int(record.get("worker_id") or 0) == int(connection["worker_id"]) for record in connections):
                raise gr.Error("This Midom project pairing is already stored locally.")
        connections.append(connection)
        config = self._config_with_connections(config, connections)
        self._save_config(config)
        self._log(
            "Project pairing added; "
            f"worker_id={connection['worker_id']} org_id={connection['org_id']} "
            f"project_id={connection['project_id']} paired_user_id={connection['paired_user_id']} "
            f"capabilities_revision={connection['capabilities_revision']} token_expires_at={token_expires_at} "
            f"stored_pairings={len(connections)}."
        )
        return self._status_text(config=config)

    def _capabilities(self) -> dict[str, Any]:
        models = []
        for model_id in sorted(ALLOWED_MODEL_TYPES):
            metadata = MODEL_CAPABILITY_OVERRIDES[model_id]
            image_reference = bool(metadata["image_reference"])
            control = bool(metadata["control"])
            control_modes = self._control_modes_for_model(model_id)
            model_capabilities = {
                "text_to_image": True,
                "image_reference": image_reference,
                "inpaint": image_reference,
                "control": control,
                "multi_output": True,
            }
            for capability_key in ("layer_decomposition", "source_echo_output", "ordered_layer_outputs"):
                if metadata.get(capability_key):
                    model_capabilities[capability_key] = True
            model_limits = {
                "max_outputs": _image_max_outputs_for_model(model_id),
                "max_steps": MAX_STEPS,
                "max_reference_images": int(metadata["max_reference_images"]),
                "max_control_images": MAX_CONTROL_IMAGES if control else 0,
                "control_modes": control_modes,
                "max_artifact_bytes": MAX_IMAGE_BYTES,
                "resolutions": sorted(ALLOWED_RESOLUTIONS),
                "input_mime_types": sorted(ALLOWED_IMAGE_MIME_TYPES),
                "output_mime_types": sorted(ALLOWED_IMAGE_MIME_TYPES),
            }
            if metadata.get("output_roles"):
                model_limits["output_roles"] = list(metadata["output_roles"])
            curated_tools = self._curated_tools_for_model(model_id)
            models.append({
                "model_id": model_id,
                "family": metadata["family"],
                "media_type": "image",
                "display_name": metadata["display_name"],
                "supported": True,
                "detected": True,
                "capabilities": model_capabilities,
                "limits": model_limits,
                "accelerator_profiles": self._accelerator_profiles_for_model(model_id),
                **({"curated_tools": curated_tools} if curated_tools else {}),
            })
        audio_output_mime_types = self._audio_output_mime_types()
        models.append({
            "model_id": "index_tts2",
            "family": "tts",
            "media_type": "audio",
            "display_name": "Index TTS 2",
            "supported": True,
            "detected": True,
            "capabilities": {
                "text_to_speech": True,
                "voice_clone": True,
                "voice_design": False,
                "multi_speaker": True,
                "multi_output": False,
                "emotion_reference": True,
                "inline_emotion_tags": True,
                "two_speaker_dialogue": True,
                "speaker_tag_dialogue": True,
                "prompt_processing_modes": list(PROMPT_PROCESSING_MODE_CHOICES),
            },
            "limits": {
                "max_outputs": 1,
                "max_reference_audio_files": 2,
                "required_reference_audio_files": 1,
                "max_emotion_reference_audio_files": 1,
                "max_dialogue_speakers": 2,
                "required_dialogue_speaker_reference_audio_files": 2,
                "min_reference_audio_seconds": MIN_REFERENCE_AUDIO_SECONDS,
                "max_reference_audio_seconds": MAX_REFERENCE_AUDIO_SECONDS,
                "min_reference_audio_sample_rate_hz": MIN_REFERENCE_AUDIO_SAMPLE_RATE_HZ,
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "max_duration_seconds": MAX_AUDIO_DURATION_SECONDS,
                "default_prompt_processing_mode": DEFAULT_PROMPT_PROCESSING_MODE,
                "max_artifact_bytes": MAX_AUDIO_BYTES,
                "input_mime_types": sorted(ALLOWED_AUDIO_INPUT_MIME_TYPES),
                "output_mime_types": audio_output_mime_types,
            },
        })
        models.append({
            "model_id": CHATTERBOX_MODEL_ID,
            "family": "tts",
            "media_type": "audio",
            "display_name": "Chatterbox Multilingual",
            "supported": True,
            "detected": True,
            "capabilities": {
                "text_to_speech": True,
                "voice_clone": True,
                "voice_design": False,
                "multi_speaker": False,
                "multi_output": False,
                "emotion_reference": False,
                "inline_emotion_tags": False,
                "two_speaker_dialogue": False,
                "speaker_tag_dialogue": False,
                "language_selection": True,
                "prompt_processing_modes": list(PROMPT_PROCESSING_MODE_CHOICES),
            },
            "limits": {
                "max_outputs": 1,
                "max_reference_audio_files": 1,
                "required_reference_audio_files": 1,
                "max_emotion_reference_audio_files": 0,
                "min_reference_audio_seconds": MIN_REFERENCE_AUDIO_SECONDS,
                "max_reference_audio_seconds": MAX_REFERENCE_AUDIO_SECONDS,
                "min_reference_audio_sample_rate_hz": MIN_REFERENCE_AUDIO_SAMPLE_RATE_HZ,
                "max_prompt_chars": CHATTERBOX_MAX_PROMPT_CHARS,
                "max_prompt_chars_fg": CHATTERBOX_MAX_PROMPT_CHARS_FG,
                "recommended_max_prompt_chars_per_line": CHATTERBOX_MAX_PROMPT_CHARS_PER_SPLIT,
                "recommended_max_prompt_chars_per_paragraph": CHATTERBOX_MAX_PROMPT_CHARS_PER_SPLIT,
                "default_prompt_processing_mode": DEFAULT_PROMPT_PROCESSING_MODE,
                "languages": list(CHATTERBOX_LANGUAGES),
                "controls": {key: value.copy() for key, value in CHATTERBOX_CONTROL_LIMITS.items()},
                "max_artifact_bytes": MAX_AUDIO_BYTES,
                "input_mime_types": sorted(ALLOWED_AUDIO_INPUT_MIME_TYPES),
                "output_mime_types": audio_output_mime_types,
            },
        })
        models.append({
            "model_id": DRAMABOX_MODEL_ID,
            "family": "tts",
            "media_type": "audio",
            "display_name": "DramaBox Audio",
            "supported": True,
            "detected": True,
            "capabilities": {
                "text_to_speech": True,
                "voice_clone": True,
                "voice_design": False,
                "multi_speaker": True,
                "dialogue_generation": True,
                "n_voice_dialogue": True,
                "two_voice_clone_dialogue": True,
                "two_speaker_dialogue": False,
                "speaker_tag_dialogue": True,
                "multi_output": False,
                "emotion_reference": False,
                "inline_emotion_tags": False,
                "language_selection": False,
                "prompt_processing_modes": [DEFAULT_PROMPT_PROCESSING_MODE],
            },
            "limits": {
                "max_outputs": 1,
                "max_reference_audio_files": 2,
                "required_reference_audio_files": 2,
                "max_emotion_reference_audio_files": 0,
                "max_reference_voice_anchors": 2,
                "max_dialogue_speakers": DRAMABOX_MAX_DIALOGUE_SPEAKERS,
                "scripted_speaker_limit_kind": "practical_ui_cap",
                "min_reference_audio_seconds": MIN_REFERENCE_AUDIO_SECONDS,
                "max_reference_audio_seconds": MAX_REFERENCE_AUDIO_SECONDS,
                "internal_reference_budget_seconds": 10,
                "min_reference_audio_sample_rate_hz": MIN_REFERENCE_AUDIO_SAMPLE_RATE_HZ,
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "max_duration_seconds": DRAMABOX_MAX_DURATION_SECONDS,
                "default_duration_seconds": 0,
                "default_prompt_processing_mode": DEFAULT_PROMPT_PROCESSING_MODE,
                "controls": {
                    "remove_unexpected_words": {"type": "boolean", "default": False},
                    "duration_multiplier": {
                        "type": "float",
                        "default": DRAMABOX_DEFAULT_DURATION_MULTIPLIER,
                        "min": DRAMABOX_MIN_DURATION_MULTIPLIER,
                        "max": DRAMABOX_MAX_DURATION_MULTIPLIER,
                    },
                },
                "max_artifact_bytes": MAX_AUDIO_BYTES,
                "input_mime_types": sorted(ALLOWED_AUDIO_INPUT_MIME_TYPES),
                "output_mime_types": audio_output_mime_types,
            },
        })
        if self._seedvc_speech_available():
            models.append({
                "model_id": SEEDVC_MODEL_ID,
                "family": "voice_conversion",
                "media_type": "audio",
                "display_name": "SeedVC Voice Replacement",
                "audio_task": "voice_conversion",
                "audio_category": "voice_replacement",
                "supported": True,
                "detected": True,
                "capabilities": {
                    "voice_conversion": True,
                    "voice_replacement": True,
                    "source_audio_required": True,
                    "reference_voice_required": True,
                    "text_to_speech": False,
                    "voice_clone": False,
                    "voice_design": False,
                    "dialogue_generation": False,
                    "two_speaker_dialogue": False,
                    "speaker_tag_dialogue": False,
                    "multi_output": False,
                    "prompt_processing_modes": [],
                },
                "limits": {
                    "max_outputs": 1,
                    "required_source_audio_files": 1,
                    "max_source_audio_files": 1,
                    "required_reference_audio_files": 1,
                    "max_reference_audio_files": 1,
                    "max_emotion_reference_audio_files": 0,
                    "min_reference_audio_seconds": SEEDVC_MIN_REFERENCE_AUDIO_SECONDS,
                    "recommended_reference_audio_seconds": SEEDVC_RECOMMENDED_REFERENCE_AUDIO_SECONDS,
                    "max_reference_audio_seconds": SEEDVC_MAX_REFERENCE_AUDIO_SECONDS,
                    "min_reference_audio_sample_rate_hz": MIN_REFERENCE_AUDIO_SAMPLE_RATE_HZ,
                    "max_source_audio_seconds": SEEDVC_MAX_SOURCE_AUDIO_SECONDS,
                    "max_artifact_bytes": MAX_AUDIO_BYTES,
                    "input_mime_types": sorted(ALLOWED_AUDIO_INPUT_MIME_TYPES),
                    "output_mime_types": audio_output_mime_types,
                    "seedvc_method": SEEDVC_METHOD_ONE_SPEAKER,
                    "seedvc_runtime_mode": SEEDVC_RUNTIME_MODE_SPEECH,
                    "controls": {},
                },
            })
        for stable_audio_model_id, stable_audio_subtype, default_duration, display_name in (
            (STABLE_AUDIO3_MUSIC_MODEL_ID, "music", 30, "Stable Audio 3 Small Music"),
            (STABLE_AUDIO3_SFX_MODEL_ID, "sound_effect", 8, "Stable Audio 3 Small SFX"),
        ):
            is_music = stable_audio_subtype == "music"
            models.append({
                "model_id": stable_audio_model_id,
                "family": "music_and_sounds",
                "media_type": "audio",
                "display_name": display_name,
                "audio_task": "generative_audio",
                "audio_category": "music_and_sounds",
                "audio_subtype": stable_audio_subtype,
                "supported": True,
                "detected": True,
                "capabilities": {
                    "music_generation": is_music,
                    "sound_effect_generation": not is_music,
                    "source_audio_edit": False,
                    "negative_prompt": True,
                    "voice_clone": False,
                    "text_to_speech": False,
                    "dialogue_generation": False,
                    "multi_output": False,
                    "prompt_processing_modes": [DEFAULT_PROMPT_PROCESSING_MODE],
                },
                "limits": {
                    "max_outputs": 1,
                    "default_duration_seconds": default_duration,
                    "max_duration_seconds": STABLE_AUDIO3_MAX_DURATION_SECONDS,
                    "max_prompt_chars": MAX_PROMPT_CHARS,
                    "max_negative_prompt_chars": STABLE_AUDIO3_MAX_NEGATIVE_PROMPT_CHARS,
                    "prompt_processing_modes": [DEFAULT_PROMPT_PROCESSING_MODE],
                    "default_prompt_processing_mode": DEFAULT_PROMPT_PROCESSING_MODE,
                    "max_artifact_bytes": MAX_AUDIO_BYTES,
                    "input_mime_types": [],
                    "output_mime_types": audio_output_mime_types,
                    "controls": {
                        "negative_prompt": {"type": "text", "default": ""},
                        "sample_solver": {
                            "type": "select",
                            "default": "pingpong",
                            "values": list(STABLE_AUDIO3_SAMPLE_SOLVERS),
                        },
                        "num_inference_steps": {"type": "int", "default": 8, "min": 1, "max": 50},
                        "guidance_scale": {"type": "float", "default": 1.0, "min": 0.0, "max": 20.0},
                    },
                },
            })
        models.append({
            "model_id": ACE_STEP15_MUSIC_MODEL_ID,
            "family": "music_and_sounds",
            "media_type": "audio",
            "display_name": "ACE-Step 1.5 Music",
            "audio_task": "generative_audio",
            "audio_category": "music_and_sounds",
            "audio_subtype": "music",
            "supported": True,
            "detected": True,
            "capabilities": {
                "music_generation": True,
                "sound_effect_generation": False,
                "source_audio_edit": False,
                "negative_prompt": False,
                "voice_clone": False,
                "text_to_speech": False,
                "dialogue_generation": False,
                "multi_output": False,
                "prompt_processing_modes": [DEFAULT_PROMPT_PROCESSING_MODE],
            },
            "limits": {
                "max_outputs": 1,
                "default_duration_seconds": ACE_STEP15_DEFAULT_DURATION_SECONDS,
                "min_duration_seconds": ACE_STEP15_MIN_DURATION_SECONDS,
                "max_duration_seconds": ACE_STEP15_MAX_DURATION_SECONDS,
                "max_music_caption_chars": ACE_STEP15_MAX_MUSIC_CAPTION_CHARS,
                "max_lyrics_chars": ACE_STEP15_MAX_LYRICS_CHARS,
                "prompt_processing_modes": [DEFAULT_PROMPT_PROCESSING_MODE],
                "default_prompt_processing_mode": DEFAULT_PROMPT_PROCESSING_MODE,
                "max_artifact_bytes": MAX_AUDIO_BYTES,
                "input_mime_types": [],
                "output_mime_types": audio_output_mime_types,
                "controls": {
                    "bpm": {"type": "int", "nullable": True, "default": None, "min": ACE_STEP15_BPM_MIN, "max": ACE_STEP15_BPM_MAX},
                    "keyscale": {"type": "text", "nullable": True, "default": ""},
                    "timesignature": {
                        "type": "select",
                        "nullable": True,
                        "default": None,
                        "values": list(ACE_STEP15_TIME_SIGNATURE_VALUES),
                    },
                    "language": {
                        "type": "select",
                        "nullable": True,
                        "default": "",
                        "values": list(ACE_STEP15_LANGUAGES),
                    },
                },
            },
        })
        models.append({
            "model_id": "longcat_avatar_v1_5",
            "family": "longcat",
            "media_type": "video",
            "display_name": "LongCat Avatar 1.5 Distilled",
            "supported": True,
            "detected": True,
            "capabilities": {
                "talking_avatar": True,
                "audio_driven_video": True,
                "lip_sync": True,
                "image_to_video": False,
                "multi_output": False,
            },
            "limits": {
                "max_outputs": 1,
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "max_duration_seconds": MAX_LONGCAT_VIDEO_DURATION_SECONDS,
                "max_artifact_bytes": MAX_VIDEO_BYTES,
                "resolutions": sorted(LONGCAT_VIDEO_RESOLUTIONS),
                "input_mime_types": sorted(ALLOWED_VIDEO_INPUT_MIME_TYPES),
                "output_mime_types": sorted(ALLOWED_VIDEO_OUTPUT_MIME_TYPES),
                "prompt_modes": ["plain"],
                "duration_modes": [LONGCAT_DURATION_MODE],
                "default_duration_mode": LONGCAT_DURATION_MODE,
                "speed_profiles": [
                    {
                        "profile_id": "distilled_8_step",
                        "display_name": "Fast",
                        "description": "Use LongCat Avatar 1.5 distilled 8-step generation.",
                        "quality_tier": "fast",
                        "steps": 8,
                        "available": True,
                        "default": True,
                    }
                ],
            },
        })
        models.append(self._ltx_video_capability(LTX23_VIDEO_MODEL_ID, "LTX-2 2.3 Dev 1.1 22B"))
        models.append(self._ltx_video_capability(LTX25_DISTILLED_VIDEO_MODEL_ID, "LTX-2 2.5 Distilled 22B"))
        models.append({
            "model_id": SVI_VIDEO_MODEL_ID,
            "family": "wan2_2",
            "media_type": "video",
            "display_name": "Wan2.2 SVI 2 Pro Enhanced Lightning v2",
            "supported": True,
            "detected": True,
            "capabilities": {
                "image_to_video": True,
                "cinematic_i2v": True,
                "start_image": True,
                "timed_cinematic_prompt": True,
                "camera_direction_prompting": True,
                "video_continuation": False,
                "anchor_image": False,
                "end_image": True,
                "ending_image_target": True,
                "driving_audio": False,
                "multi_output": False,
            },
            "limits": {
                "max_outputs": 1,
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "default_duration_seconds": DEFAULT_SVI_VIDEO_DURATION_SECONDS,
                "min_duration_seconds": MIN_SVI_VIDEO_DURATION_SECONDS,
                "max_duration_seconds": MAX_SVI_VIDEO_DURATION_SECONDS,
                "max_artifact_bytes": MAX_VIDEO_BYTES,
                "resolutions": sorted(SVI_VIDEO_RESOLUTIONS),
                "input_mime_types": sorted(ALLOWED_IMAGE_MIME_TYPES),
                "output_mime_types": sorted(ALLOWED_VIDEO_OUTPUT_MIME_TYPES),
                "prompt_modes": ["plain", "timed_cinematic_seconds"],
                "duration_modes": [SVI_DURATION_MODE],
                "default_duration_mode": SVI_DURATION_MODE,
                "required_input_kinds": ["start_image"],
                "optional_input_kinds": ["end_image"],
                "max_end_images": 1,
                "speed_profiles": [
                    {
                        "profile_id": SVI_SPEED_PROFILE_ID,
                        "display_name": "Enhanced Lightning",
                        "description": "Use plugin-curated Wan2.2 SVI Enhanced Lightning v2 defaults.",
                        "quality_tier": "fast",
                        "steps": 8,
                        "available": True,
                        "default": True,
                    }
                ],
            },
        })
        media_processing = []
        event_capability = self._event_video_processing_capability()
        if event_capability:
            media_processing.append(event_capability)
        storyboard_capability = self._storyboard_ffmpeg_processing_capability()
        if storyboard_capability:
            media_processing.append(storyboard_capability)
        curated_tools = [
            dict(tool)
            for model in models
            if isinstance(model, dict)
            for tool in (model.get("curated_tools") or [])
            if isinstance(tool, dict)
        ]
        return {
            "schema_version": 1,
            "media_types": ["audio", "image", "video"],
            "models": models,
            "curated_tools": curated_tools,
            "media_processing": media_processing,
            "unsupported_detected_models": [],
        }

    def _event_video_processing_capability(self) -> Optional[dict[str, Any]]:
        probe = self._ffmpeg_processing_probe()
        if not probe.get("ffmpeg_available") or not probe.get("ffprobe_available"):
            self._log(
                "Event Video Processing capability withheld; "
                f"ffmpeg_available={probe.get('ffmpeg_available')} ffprobe_available={probe.get('ffprobe_available')}.",
                force=True,
            )
            return None
        source_mime_types = ["video/mp4"]
        if probe.get("quicktime_demux_available"):
            source_mime_types.append("video/quicktime")
        return {
            "family": "media_processing",
            "media_type": "video",
            "processing_task": EVENT_VIDEO_PROCESSING_TASK,
            "processor_id": EVENT_VIDEO_PROCESSOR_ID,
            "display_name": "Event Video Processing",
            "supported": True,
            "detected": True,
            "ffmpeg_available": True,
            "ffprobe_available": True,
            "nvenc_available": bool(probe.get("nvenc_available")),
            "h264_encoder": EVENT_VIDEO_H264_ENCODER,
            "supports_overlay_png": True,
            "supports_bumper": True,
            "supports_poster": False,
            "max_input_bytes": MAX_EVENT_VIDEO_INPUT_BYTES,
            "max_duration_seconds": MAX_EVENT_VIDEO_DURATION_SECONDS,
            "supported_output_profiles": [EVENT_VIDEO_OUTPUT_PROFILE],
            "output_mime_types": ["video/mp4"],
            "input_mime_types": {
                "source_video": source_mime_types,
                "overlay_png": ["image/png"],
                "bumper_image": ["image/png", "image/jpeg", "image/webp"],
            },
        }

    def _storyboard_ffmpeg_processing_capability(self) -> Optional[dict[str, Any]]:
        probe = self._ffmpeg_processing_probe()
        if not probe.get("ffmpeg_available") or not probe.get("ffprobe_available"):
            self._log(
                "Storyboard FFmpeg Processing capability withheld; "
                f"ffmpeg_available={probe.get('ffmpeg_available')} ffprobe_available={probe.get('ffprobe_available')}.",
                force=True,
            )
            return None
        source_mime_types = ["video/mp4", "video/webm"]
        if probe.get("quicktime_demux_available"):
            source_mime_types.append("video/quicktime")
        return {
            "family": "media_processing",
            "media_type": "video",
            "processing_task": STORYBOARD_FFMPEG_PROCESSING_TASK,
            "processor_id": STORYBOARD_FFMPEG_PROCESSOR_ID,
            "display_name": "Storyboard FFmpeg Processing",
            "supported": True,
            "detected": True,
            "ffmpeg_available": True,
            "ffprobe_available": True,
            "nvenc_available": bool(probe.get("nvenc_available")),
            "h264_encoder": EVENT_VIDEO_H264_ENCODER,
            "operation_types": sorted(STORYBOARD_FFMPEG_OPERATION_TYPES),
            "supported_operations": sorted(STORYBOARD_FFMPEG_OPERATION_TYPES),
            "max_input_bytes": MAX_STORYBOARD_VIDEO_INPUT_BYTES,
            "max_duration_seconds": MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
            "supports_trim": True,
            "supports_concat": True,
            "supports_overlay": True,
            "supports_faststart": True,
            "operation_features": {
                "multicam_card_overlay_take": [
                    "static_rectangle_overlay",
                    "animated_rectangle_overlay",
                    "animated_position",
                    "animated_opacity",
                    "fixed_canvas_animated_scale",
                    "luminance_matte_png",
                    "luminance_matte_jpeg",
                    "base_or_overlay_audio",
                ],
                "multicam_final_assembly": [
                    "ordered_segments",
                    "segment_trim",
                    "normalize_before_concat",
                    "silent_audio_fill",
                    "h264_aac_mp4_faststart",
                ],
                "multicam_optimize_video": [
                    "h264_aac_reencode",
                    "faststart",
                    "max_dimension_scale",
                    "crf_preset_audio_bitrate",
                ],
                "optimize_video": [
                    "h264_aac_reencode",
                    "faststart",
                    "max_dimension_scale",
                    "crf_preset_audio_bitrate",
                ],
                "replace_video_soundtrack": [
                    "source_video_stream",
                    "soundtrack_audio_replacement",
                    "soundtrack_video_container_audio",
                    "video_and_audio_start_offsets",
                    "duration_trim",
                    "head_tail_silence",
                    "h264_aac_mp4_faststart",
                ],
            },
            "output_mime_types": ["video/mp4"],
            "input_mime_types": {
                "video": source_mime_types,
                "source_video": source_mime_types,
                "input_video": source_mime_types,
                "take_video": source_mime_types,
                "card_video": source_mime_types,
                "segment_video": source_mime_types,
                "overlay_video": source_mime_types,
                "image": sorted(ALLOWED_IMAGE_MIME_TYPES),
                "source_image": sorted(ALLOWED_IMAGE_MIME_TYPES),
                "overlay_image": sorted(ALLOWED_IMAGE_MIME_TYPES),
                "overlay_png": ["image/png"],
                "poster_image": sorted(ALLOWED_IMAGE_MIME_TYPES),
                "audio": sorted(ALLOWED_AUDIO_INPUT_MIME_TYPES),
                "source_audio": sorted(ALLOWED_STORYBOARD_AUDIO_CONTAINER_MIME_TYPES),
                "soundtrack_audio": sorted(ALLOWED_STORYBOARD_AUDIO_CONTAINER_MIME_TYPES),
                "driving_audio": sorted(ALLOWED_AUDIO_INPUT_MIME_TYPES),
                "narration_audio": sorted(ALLOWED_AUDIO_INPUT_MIME_TYPES),
            },
        }

    def _audio_output_mime_types(self) -> list[str]:
        output_mime_types = ["audio/wav"]
        if self._mp3_transcode_available():
            output_mime_types.insert(0, "audio/mpeg")
        return output_mime_types

    def _ffmpeg_binary(self) -> str:
        try:
            from shared.utils.audio_video import _ffmpeg_binary

            return str(_ffmpeg_binary())
        except Exception:
            return "ffmpeg"

    def _ffmpeg_processing_probe(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._ffmpeg_probe_cache is not None and now - self._ffmpeg_probe_cache_at < MP3_ENCODER_CACHE_SECONDS:
            return dict(self._ffmpeg_probe_cache)
        ffmpeg_binary = self._ffmpeg_binary()
        ffprobe_binary = self._ffprobe_binary()
        probe = {
            "ffmpeg_binary": ffmpeg_binary,
            "ffprobe_binary": ffprobe_binary,
            "ffmpeg_available": False,
            "ffprobe_available": False,
            "quicktime_demux_available": False,
            "nvenc_available": False,
        }
        try:
            completed = subprocess.run(
                [ffmpeg_binary, "-hide_banner", "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            probe["ffmpeg_available"] = completed.returncode == 0
        except Exception:
            probe["ffmpeg_available"] = False
        try:
            completed = subprocess.run(
                [ffprobe_binary, "-hide_banner", "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            probe["ffprobe_available"] = completed.returncode == 0
        except Exception:
            probe["ffprobe_available"] = False
        if probe["ffmpeg_available"]:
            # Avoid running `ffmpeg -demuxers` during pairing/UI refresh. On some
            # Windows installs it can be surprisingly slow, and downloaded inputs
            # are still ffprobed before a job is accepted for processing.
            probe["quicktime_demux_available"] = True
            # First-pass Event Video Processing always uses libx264. Do not report NVENC
            # until we add a real encode smoke check, because listing h264_nvenc is not
            # enough to prove the GPU encoder can run in this process.
            probe["nvenc_available"] = False
        self._ffmpeg_probe_cache = dict(probe)
        self._ffmpeg_probe_cache_at = now
        signature = (
            probe["ffmpeg_binary"],
            probe["ffprobe_binary"],
            probe["ffmpeg_available"],
            probe["ffprobe_available"],
            probe["quicktime_demux_available"],
            probe["nvenc_available"],
        )
        if signature != self._ffmpeg_probe_log_signature:
            self._ffmpeg_probe_log_signature = signature
            self._log(
                "FFmpeg processing probe; "
                f"ffmpeg={ffmpeg_binary!r} ffmpeg_available={probe['ffmpeg_available']} "
                f"ffprobe={ffprobe_binary!r} ffprobe_available={probe['ffprobe_available']} "
                f"quicktime_demux_available={probe['quicktime_demux_available']} "
                f"nvenc_available={probe['nvenc_available']}."
            )
        return probe

    def _seedvc_speech_available(self) -> bool:
        try:
            from postprocessing import audio_processors as audio_processor_api
            from postprocessing.seedvc.wgp_bridge import SeedVCBridge

            handler = audio_processor_api.find_processor(SEEDVC_METHOD_ONE_SPEAKER)
            if handler is None:
                self._log("SeedVC capability withheld; SeedVC audio processor is not registered.")
                return False
            if not audio_processor_api.method_has_type(SEEDVC_METHOD_ONE_SPEAKER, audio_processor_api.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT):
                self._log("SeedVC capability withheld; SeedVC is not available as an audio-edit processor.")
                return False
            server_config = getattr(handler, "server_config", None)
            if not isinstance(server_config, dict):
                self._log("SeedVC capability withheld; SeedVC handler has no server_config.")
                return False
            config = audio_processor_api.read_config_section(server_config, handler)
            mode = int(config.get("mode") or SeedVCBridge.MODE_OFF)
            if mode != SeedVCBridge.MODE_V1:
                self._log(f"SeedVC capability withheld; configured SeedVC mode is not speech v1. mode={mode}.")
                return False
            validation_error = audio_processor_api.validate_method(
                SEEDVC_METHOD_ONE_SPEAKER,
                audio_processor_api.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT,
                voice_sample="capability-reference.wav",
            )
            if validation_error:
                self._log(f"SeedVC capability withheld; validation failed: {validation_error}")
                return False
            return True
        except Exception as exc:
            self._log(f"SeedVC capability check failed; capability withheld. error={exc}", force=True)
            return False

    def _mp3_transcode_available(self) -> bool:
        now = time.monotonic()
        if self._mp3_encoder_cache is not None and now - self._mp3_encoder_cache_at < MP3_ENCODER_CACHE_SECONDS:
            return bool(self._mp3_encoder_cache)
        try:
            from shared.utils.audio_video import _ffmpeg_binary

            binary = _ffmpeg_binary()
            completed = subprocess.run(
                [binary, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            encoders = f"{completed.stdout}\n{completed.stderr}".lower()
            available = completed.returncode == 0 and "libmp3lame" in encoders
            self._mp3_encoder_cache = available
            self._mp3_encoder_cache_at = now
            self._log(f"MP3 encoder check; ffmpeg={binary!r} libmp3lame_available={available}.")
            return available
        except Exception as exc:
            self._mp3_encoder_cache = False
            self._mp3_encoder_cache_at = now
            self._log(f"MP3 encoder check failed; audio/mpeg will not be reported. error={exc}", force=True)
            return False

    def _ffprobe_binary(self) -> str:
        try:
            from shared.utils.audio_video import _ffmpeg_binary

            ffmpeg_path = Path(str(_ffmpeg_binary()))
            executable = "ffprobe.exe" if ffmpeg_path.name.lower().endswith(".exe") else "ffprobe"
            candidate = ffmpeg_path.with_name(executable)
            if candidate.is_file():
                return str(candidate)
        except Exception:
            pass
        return "ffprobe"

    def _accelerator_profiles_for_model(self, model_id: str) -> list[dict[str, Any]]:
        profiles = [
            {
                "profile_id": "standard",
                "display_name": "Standard",
                "description": "Use normal WanGP generation settings.",
                "quality_tier": "standard",
                "available": True,
                "default": True,
            }
        ]
        for profile in ACCELERATOR_PROFILE_DEFINITIONS:
            if model_id not in set(profile.get("model_ids") or []):
                continue
            resolved_loras, missing_loras = self._resolve_accelerator_loras(profile)
            available = bool(resolved_loras) and not missing_loras
            item = {
                "profile_id": str(profile["profile_id"]),
                "display_name": str(profile["display_name"]),
                "description": str(profile["description"]),
                "quality_tier": str(profile["quality_tier"]),
                "steps": int(profile["steps"]),
                "available": available,
            }
            if not available:
                item["unavailable_reason"] = "required_accelerator_files_not_installed"
            profiles.append(item)
        return profiles

    def _curated_tools_for_model(self, model_id: str) -> list[dict[str, Any]]:
        if str(model_id or "").strip() != QWEN_MULTI_ANGLE_BASE_MODEL_ID:
            return []
        availability = self._qwen_multi_angle_tool_availability()
        return [
            {
                "tool_id": QWEN_MULTI_ANGLE_TOOL_ID,
                "display_name": QWEN_MULTI_ANGLE_TOOL_DISPLAY_NAME,
                "media_type": "image",
                "base_model_id": QWEN_MULTI_ANGLE_BASE_MODEL_ID,
                "available": bool(availability["available"]),
                "unavailable_reason": availability["unavailable_reason"],
                "input_kinds": ["reference_image"],
                "parameters": {
                    "view_azimuth": list(QWEN_MULTI_ANGLE_AZIMUTHS.keys()),
                    "view_elevation": list(QWEN_MULTI_ANGLE_ELEVATIONS.keys()),
                    "shot_distance": list(QWEN_MULTI_ANGLE_DISTANCES.keys()),
                    "view_change_strength": list(QWEN_MULTI_ANGLE_VIEW_CHANGE_STRENGTHS.keys()),
                },
                "default_parameters": {
                    "view_change_strength": QWEN_MULTI_ANGLE_DEFAULT_VIEW_CHANGE_STRENGTH,
                },
                "output": {
                    "min_count": 1,
                    "max_count": _image_max_outputs_for_model(QWEN_MULTI_ANGLE_BASE_MODEL_ID),
                },
                "accelerator_profiles": self._qwen_multi_angle_accelerator_profiles(),
                "tool_version": QWEN_MULTI_ANGLE_TOOL_VERSION,
                "recipe_version": QWEN_MULTI_ANGLE_TOOL_VERSION,
                "license": QWEN_MULTI_ANGLE_LORA_LICENSE,
            }
        ]

    def _qwen_multi_angle_accelerator_profiles(self) -> list[dict[str, Any]]:
        profiles = []
        for profile in self._accelerator_profiles_for_model(QWEN_MULTI_ANGLE_BASE_MODEL_ID):
            profile_id = str(profile.get("profile_id") or "").strip()
            if profile_id in QWEN_MULTI_ANGLE_ALLOWED_ACCELERATOR_PROFILE_IDS:
                item = dict(profile)
                item["default"] = profile_id == "standard"
                profiles.append(item)
        return profiles

    def _qwen_multi_angle_tool_availability(self) -> dict[str, Any]:
        if QWEN_MULTI_ANGLE_BASE_MODEL_ID not in ALLOWED_MODEL_TYPES:
            return {
                "available": False,
                "unavailable_reason": "base_model_not_supported_by_bridge",
                "lora_relative_path": "",
                "lora_sha256": "",
            }
        relative_path = self._find_lora_relative_path(QWEN_MULTI_ANGLE_LORA_DIR, QWEN_MULTI_ANGLE_LORA_FILENAME)
        if not relative_path:
            return {
                "available": False,
                "unavailable_reason": "required_lora_file_not_installed",
                "lora_relative_path": "",
                "lora_sha256": "",
            }
        absolute_path = self._lora_absolute_path(QWEN_MULTI_ANGLE_LORA_DIR, relative_path)
        try:
            actual_sha256 = self._sha256_file(absolute_path)
        except Exception as exc:
            self._log(f"Qwen multi-angle LoRA hash check failed: {exc}", force=True)
            return {
                "available": False,
                "unavailable_reason": "required_lora_file_hash_unreadable",
                "lora_relative_path": relative_path,
                "lora_sha256": "",
            }
        if actual_sha256.lower() != QWEN_MULTI_ANGLE_LORA_SHA256:
            return {
                "available": False,
                "unavailable_reason": "required_lora_file_hash_mismatch",
                "lora_relative_path": relative_path,
                "lora_sha256": actual_sha256,
            }
        return {
            "available": True,
            "unavailable_reason": None,
            "lora_relative_path": relative_path,
            "lora_sha256": actual_sha256,
        }

    def _ltx_video_capability(self, model_id: str, display_name: str) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "family": "ltx2",
            "media_type": "video",
            "display_name": display_name,
            "supported": True,
            "detected": True,
            "capabilities": {
                "audio_conditioned_video": True,
                "audio_guided_video": True,
                "driving_audio_guided": True,
                "prompt_generated_audio": False,
                "reference_voice_video": False,
                "control_video": True,
                "control_video_audio": True,
                "control_video_audio_guided": True,
                "control_video_portrait": True,
                "separate_driving_audio_with_control_video": False,
                "control_video_modes": list(LTX_CONTROL_VIDEO_MODES.keys()),
                "duration_modes": [LTX_DURATION_MODE, LTX_CONTROL_VIDEO_DURATION_MODE],
                "end_image": True,
                "ending_image_target": True,
                "image_to_video": False,
                "talking_avatar": False,
                "multi_output": False,
            },
            "limits": {
                "max_outputs": 1,
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "max_duration_seconds": MAX_LTX_VIDEO_DURATION_SECONDS,
                "max_artifact_bytes": MAX_VIDEO_BYTES,
                "resolutions": sorted(LTX_VIDEO_RESOLUTIONS),
                "control_video_profiles": [dict(profile) for profile in LTX_CONTROL_VIDEO_PROFILES],
                "input_mime_types": sorted(ALLOWED_VIDEO_INPUT_MIME_TYPES),
                "output_mime_types": sorted(ALLOWED_VIDEO_OUTPUT_MIME_TYPES),
                "prompt_modes": ["plain"],
                "duration_modes": [LTX_DURATION_MODE, LTX_CONTROL_VIDEO_DURATION_MODE],
                "default_duration_mode": LTX_DURATION_MODE,
                "required_input_kinds": ["start_image", "driving_audio"],
                "optional_input_kinds": ["end_image"],
                "control_video_required_input_kinds": ["start_image", "control_video"],
                "control_video_optional_input_kinds": [],
                "max_end_images": 1,
                "speed_profiles": [
                    {
                        "profile_id": "standard",
                        "display_name": "Standard",
                        "description": f"Use plugin-curated {display_name} audio-guided defaults.",
                        "quality_tier": "standard",
                        "steps": 8,
                        "available": True,
                        "default": True,
                    }
                ],
            },
            "video_sync_profiles": self._ltx_video_sync_profiles(),
        }

    def _ltx_video_sync_profiles(self) -> list[dict[str, Any]]:
        omni_lora = self._find_lora_relative_path("ltx2", LTX_VIDEO_SYNC_OMNINFT_LORA)
        omni_available = bool(omni_lora)
        profiles = [
            {
                "profile_id": "standard",
                "display_name": "Standard",
                "description": "Use standard LTX audio/video sync.",
                "quality_tier": "standard",
                "available": True,
                "default": True,
            },
            {
                "profile_id": LTX_VIDEO_SYNC_OMNINFT_PROFILE_ID,
                "display_name": "Better Audio/Video Sync",
                "description": "Use WanGP's installed OmniNFT RL-LoRA for LTX audio/video sync.",
                "quality_tier": "sync",
                "available": omni_available,
            },
        ]
        if not omni_available:
            profiles[1]["unavailable_reason"] = "required_sync_profile_file_not_installed"
        return profiles

    def _apply_ltx_video_sync_profile(self, settings: dict[str, Any], profile_id: str) -> None:
        profile_id = str(profile_id or "standard").strip() or "standard"
        if profile_id == "standard":
            settings["_midom_video_sync_profile_id"] = "standard"
            return
        if profile_id != LTX_VIDEO_SYNC_OMNINFT_PROFILE_ID:
            raise ValueError(f"Unsupported LTX video_sync_profile_id: {profile_id}")
        omni_lora = self._find_lora_relative_path("ltx2", LTX_VIDEO_SYNC_OMNINFT_LORA)
        if not omni_lora:
            raise ValueError(
                "LTX OmniNFT sync profile is not available on this WanGP worker. "
                "Install the required WanGP LoRA, update capabilities, and try again."
            )
        settings["activated_loras"] = [omni_lora]
        settings["loras_multipliers"] = "1"
        settings["_midom_video_sync_profile_id"] = profile_id
        self._log(
            "Applied LTX video sync profile; "
            f"profile_id={profile_id} local_lora={omni_lora!r}."
        )

    def _select_accelerator_profile(self, model_id: str, generation: dict[str, Any]) -> Optional[dict[str, Any]]:
        profile_id = str(generation.get("accelerator_profile_id") or "standard").strip() or "standard"
        if profile_id == "standard":
            return None
        profile = ACCELERATOR_PROFILE_BY_ID.get(profile_id)
        if not profile or model_id not in set(profile.get("model_ids") or []):
            raise ValueError(f"Unsupported accelerator_profile_id for {model_id}: {profile_id}")
        resolved_loras, missing_loras = self._resolve_accelerator_loras(profile)
        if missing_loras or not resolved_loras:
            raise ValueError(
                f"Accelerator profile {profile_id} is not available on this WanGP worker. "
                "Install the required WanGP accelerator files, update capabilities, and try again."
            )
        selected = dict(profile)
        selected["_resolved_loras"] = resolved_loras
        return selected

    def _apply_accelerator_profile(self, settings: dict[str, Any], model_id: str, generation: dict[str, Any]) -> str:
        selected = self._select_accelerator_profile(model_id, generation)
        if selected is None:
            settings["_midom_accelerator_profile_id"] = "standard"
            if generation.get("steps") is not None:
                settings["num_inference_steps"] = self._coerce_int(generation.get("steps"), DEFAULT_STEPS, 1, MAX_STEPS)
                steps_label = str(settings.get("num_inference_steps"))
            else:
                settings.pop("num_inference_steps", None)
                steps_label = "WanGP default"
            self._log(
                f"Using standard WanGP speed profile; model_id={model_id} "
                f"steps={steps_label}."
            )
            return "standard"
        steps = self._coerce_int(selected.get("steps"), DEFAULT_STEPS, 1, MAX_STEPS)
        settings["num_inference_steps"] = steps
        settings["activated_loras"] = list(selected["_resolved_loras"])
        settings["loras_multipliers"] = str(selected.get("loras_multipliers") or "1")
        if selected.get("guidance_scale") is not None:
            settings["guidance_scale"] = selected["guidance_scale"]
        settings["_midom_accelerator_profile_id"] = str(selected["profile_id"])
        self._log(
            "Applied WanGP speed profile; "
            f"model_id={model_id} profile_id={selected['profile_id']} "
            f"display_name={selected['display_name']!r} steps={steps} "
            f"local_loras={len(settings['activated_loras'])}."
        )
        return str(selected["profile_id"])

    def _resolve_accelerator_loras(self, profile: dict[str, Any]) -> tuple[list[str], list[str]]:
        lora_dir_name = str(profile.get("lora_dir") or "").strip()
        filenames = [str(item).strip() for item in (profile.get("lora_filenames") or []) if str(item).strip()]
        resolved = []
        missing = []
        for filename in filenames:
            relative_path = self._find_lora_relative_path(lora_dir_name, filename)
            if relative_path:
                resolved.append(relative_path)
            else:
                missing.append(filename)
        return resolved, missing

    def _find_lora_relative_path(self, lora_dir_name: str, filename: str) -> str:
        if not lora_dir_name or not filename:
            return ""
        wangp_root = Path(__file__).resolve().parents[2]
        lora_dir = wangp_root / "loras" / lora_dir_name
        for relative in (Path(filename), Path("loras_accelerators") / filename):
            if (lora_dir / relative).is_file():
                return relative.as_posix()
        return ""

    def _lora_absolute_path(self, lora_dir_name: str, relative_path: str) -> Path:
        if not lora_dir_name or not relative_path:
            raise ValueError("LoRA path is incomplete.")
        wangp_root = Path(__file__).resolve().parents[2]
        lora_dir = wangp_root / "loras" / lora_dir_name
        absolute_path = (lora_dir / relative_path).resolve()
        try:
            absolute_path.relative_to(lora_dir.resolve())
        except ValueError:
            raise ValueError(f"LoRA path escapes expected directory: {relative_path}")
        if not absolute_path.is_file():
            raise ValueError(f"LoRA file is missing: {relative_path}")
        return absolute_path

    def _sha256_file(self, path: Path) -> str:
        stat = path.stat()
        cache_key = str(path.resolve())
        cache_value = self._lora_hash_cache.get(cache_key) if isinstance(self._lora_hash_cache, dict) else None
        signature = (int(stat.st_size), int(stat.st_mtime_ns))
        if isinstance(cache_value, dict) and cache_value.get("signature") == signature and cache_value.get("sha256"):
            return str(cache_value["sha256"])
        digest = hashlib.sha256()
        with path.open("rb") as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        if isinstance(self._lora_hash_cache, dict):
            self._lora_hash_cache[cache_key] = {"signature": signature, "sha256": value}
        return value

    @staticmethod
    def _control_modes_for_model(model_id: str) -> list[dict[str, str]]:
        metadata = MODEL_CAPABILITY_OVERRIDES.get(model_id) or {}
        modes = metadata.get("control_modes") or []
        return [
            {
                "mode_id": mode_id,
                "display_name": CONTROL_MODE_DEFINITIONS[mode_id]["display_name"],
                "description": CONTROL_MODE_DEFINITIONS[mode_id]["description"],
            }
            for mode_id in modes
            if mode_id in CONTROL_MODE_DEFINITIONS
        ]

    def _update_capabilities(self) -> str:
        config = self._load_config()
        connections = self._connection_records(config)
        if not connections:
            raise gr.Error("Pair this worker before updating capabilities.")
        capabilities = self._capabilities()
        self._log(f"Updating capabilities for {len(connections)} project pairing(s); models={len(capabilities.get('models') or [])}.")
        audio_models = [model for model in capabilities.get("models", []) if isinstance(model, dict) and model.get("media_type") == "audio"]
        for model in audio_models:
            limits = model.get("limits") or {}
            self._log(
                "Audio capability prepared; "
                f"model_id={model.get('model_id')} output_mime_types={limits.get('output_mime_types') or []}."
            )
        video_models = [model for model in capabilities.get("models", []) if isinstance(model, dict) and model.get("media_type") == "video"]
        for model in video_models:
            limits = model.get("limits") or {}
            self._log(
                "Video capability prepared; "
                f"model_id={model.get('model_id')} output_mime_types={limits.get('output_mime_types') or []} "
                f"speed_profiles={limits.get('speed_profiles') or []} "
                f"video_sync_profiles={model.get('video_sync_profiles') or []}."
            )
        processing_capabilities = capabilities.get("media_processing") or []
        for capability in processing_capabilities:
            if not isinstance(capability, dict):
                continue
            self._log(
                "Media processing capability prepared; "
                f"processor_id={capability.get('processor_id')} processing_task={capability.get('processing_task')} "
                f"h264_encoder={capability.get('h264_encoder')} "
                f"input_mime_types={capability.get('input_mime_types') or {}}."
            )
        updated_connections = []
        updated_count = 0
        removed_count = 0
        failed_updates = []
        for record in connections:
            connection = self._connection_context(record)
            response = requests.put(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/capabilities",
                headers={**self._headers(connection), "Content-Type": "application/json"},
                json=capabilities,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                message = self._midom_error_message(response)
                self._log(
                    "Capabilities update rejected by Midom; "
                    f"connection_id={connection.connection_id} worker_id={connection.worker_id} "
                    f"http_status={response.status_code} message={message!r}.",
                    force=True,
                )
                if response.status_code in {401, 403} and self._forget_unauthorized_connection(
                    connection,
                    "Capabilities update",
                    message,
                ):
                    removed_count += 1
                    continue
                failed_updates.append((connection, response.status_code, message))
                updated_connections.append(record)
                continue
            payload = response.json()
            next_record = dict(record)
            if isinstance(payload, dict):
                next_record["capabilities_revision"] = self._coerce_int(
                    payload.get("capabilities_revision"),
                    int(record.get("capabilities_revision") or 1),
                    1,
                    1_000_000,
                )
            updated_connections.append(next_record)
            updated_count += 1
            self._log(
                "Capabilities update accepted; "
                f"connection_id={connection.connection_id} worker_id={connection.worker_id} "
                f"capabilities_revision={next_record['capabilities_revision']}."
            )
        config = self._save_connections(updated_connections, config)
        if failed_updates:
            connection, http_status, message = failed_updates[0]
            raise gr.Error(
                "Capabilities update failed for one or more project pairings. "
                f"First failure: worker_id={connection.worker_id} Midom HTTP {http_status}. {message}"
            )
        if updated_count <= 0 and removed_count:
            self._log(
                "All stored project pairings were rejected by Midom authorization and removed locally. "
                "Generate new Midom pairing codes before reconnecting.",
                force=True,
            )
        elif removed_count:
            self._log(
                f"Capabilities updated for {updated_count} project pairing(s); "
                f"removed {removed_count} stale unauthorized pairing(s).",
                force=True,
            )
        return self._status_text(config=config)

    def _validate_job(self, job: dict[str, Any], config: Any) -> dict[str, Any]:
        if not isinstance(job, dict):
            raise ValueError("Job payload must be a JSON object.")
        job_id = self._coerce_job_id(job)
        self._validate_job_scope(job, config)
        if self._is_event_video_processing_job(job):
            return self._validate_event_video_processing_job(job, job_id)
        if self._is_storyboard_ffmpeg_processing_job(job):
            return self._validate_storyboard_ffmpeg_processing_job(job, job_id)
        media_type = str(job.get("media_type") or job.get("kind") or "image").lower()
        if media_type == "audio":
            return self._validate_audio_job(job, job_id)
        if media_type == "video":
            return self._validate_video_job(job, job_id)
        if media_type != "image":
            raise ValueError(f"Unsupported media type: {media_type}")
        return self._validate_image_job(job, job_id)

    def _validate_image_job(self, job: dict[str, Any], job_id: int) -> dict[str, Any]:
        model_type = str(job.get("model_id") or job.get("model_type") or job.get("model") or "").strip()
        if model_type not in ALLOWED_MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {model_type}")

        generation = job.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError("Job generation must be a JSON object when provided.")
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("Job output must be a JSON object.")
        has_output_count = output.get("count") is not None
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported output count: {output.get('count')}")
        if model_type == "qwen_image_layered_20B":
            image_task = str(generation.get("image_task") or "layer_decomposition").strip().lower()
            if image_task != "layer_decomposition":
                raise ValueError(f"Unsupported Qwen Image Layered image_task: {image_task}")
            layer_mode = str(generation.get("layer_mode") or "control_image_decomposition").strip().lower()
            if layer_mode != "control_image_decomposition":
                raise ValueError(f"Unsupported Qwen Image Layered layer_mode: {layer_mode}")
            layer_output_value = generation.get("layer_output_count")
            if layer_output_value is not None:
                try:
                    layer_output_count = int(layer_output_value)
                except (TypeError, ValueError):
                    raise ValueError(f"Unsupported Qwen Image Layered layer_output_count: {layer_output_value}")
                if has_output_count and layer_output_count != output_count:
                    raise ValueError("Qwen Image Layered output.count and generation.layer_output_count must match.")
                output_count = layer_output_count
        max_outputs = _image_max_outputs_for_model(model_type)
        if output_count < 1 or output_count > max_outputs:
            raise ValueError(f"Unsupported output count: {output.get('count')}")
        tool_id = str(job.get("tool_id") or generation.get("tool_id") or "").strip()
        tool_payload = self._validate_qwen_multi_angle_tool_request(job, generation, output_count, tool_id) if tool_id else None
        settings = {}
        settings["model_type"] = model_type
        prompt = str(job.get("prompt") or "").strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Job prompt exceeds {MAX_PROMPT_CHARS} characters.")
        settings["prompt"] = str(tool_payload["expanded_prompt"] if tool_payload else prompt)
        if not settings["prompt"]:
            raise ValueError("Job prompt is required.")
        negative_prompt = str(job.get("negative_prompt") or "").strip()
        if negative_prompt:
            settings["negative_prompt"] = negative_prompt[:MAX_PROMPT_CHARS]
        resolution = self._resolve_job_resolution(job)
        if resolution:
            settings["resolution"] = resolution
        settings["image_mode"] = 1
        settings["video_length"] = 1
        self._apply_accelerator_profile(settings, model_type, generation)
        if tool_payload:
            self._apply_curated_image_tool_settings(settings, tool_payload["settings"])
        seed = generation.get("seed")
        if seed is not None:
            settings["seed"] = self._coerce_int(seed, 0, 0, 2_147_483_647)
            settings["_midom_requested_seed"] = settings["seed"]
        else:
            settings["_midom_requested_seed"] = None
        settings["_midom_job_id"] = job_id
        settings["_midom_media_type"] = "image"
        settings["_midom_output_count"] = output_count
        if model_type == "qwen_image_layered_20B":
            settings["_midom_image_task"] = "layer_decomposition"
            settings["_midom_layer_mode"] = "control_image_decomposition"
            settings["_midom_layer_output_count"] = output_count
        return settings

    def _apply_curated_image_tool_settings(self, settings: dict[str, Any], tool_settings: dict[str, Any]) -> None:
        existing_loras = [str(item).strip() for item in (settings.get("activated_loras") or []) if str(item).strip()]
        tool_loras = [str(item).strip() for item in (tool_settings.get("activated_loras") or []) if str(item).strip()]
        if tool_loras:
            existing_multipliers = self._lora_multiplier_items(settings.get("loras_multipliers"), len(existing_loras))
            tool_multipliers = self._lora_multiplier_items(tool_settings.get("loras_multipliers"), len(tool_loras))
            settings["activated_loras"] = existing_loras + tool_loras
            settings["loras_multipliers"] = " ".join(existing_multipliers + tool_multipliers).strip()
        for key, value in tool_settings.items():
            if key in {"activated_loras", "loras_multipliers"}:
                continue
            settings[key] = value

    @staticmethod
    def _lora_multiplier_items(value: Any, count: int) -> list[str]:
        if count <= 0:
            return []
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            text = str(value or "").replace("\r", "\n").replace("|", " ").strip()
            items = [item.strip() for item in re.split(r"\s+", text) if item.strip()]
        if len(items) < count:
            items.extend(["1"] * (count - len(items)))
        return items[:count]

    def _validate_qwen_multi_angle_tool_request(
        self,
        job: dict[str, Any],
        generation: dict[str, Any],
        output_count: int,
        tool_id: str,
    ) -> dict[str, Any]:
        if tool_id != QWEN_MULTI_ANGLE_TOOL_ID:
            raise ValueError(f"Unsupported image tool_id: {tool_id}")
        model_type = str(job.get("model_id") or job.get("model_type") or job.get("model") or "").strip()
        if model_type != QWEN_MULTI_ANGLE_BASE_MODEL_ID:
            raise ValueError(
                f"{QWEN_MULTI_ANGLE_TOOL_ID} requires model_id {QWEN_MULTI_ANGLE_BASE_MODEL_ID}; got {model_type}."
            )
        max_outputs = _image_max_outputs_for_model(model_type)
        if output_count < 1 or output_count > max_outputs:
            raise ValueError(f"{QWEN_MULTI_ANGLE_TOOL_ID} supports 1 to {max_outputs} output image(s) per request.")
        accelerator_profile_id = str(generation.get("accelerator_profile_id") or "standard").strip() or "standard"
        if accelerator_profile_id not in QWEN_MULTI_ANGLE_ALLOWED_ACCELERATOR_PROFILE_IDS:
            raise ValueError(
                f"{QWEN_MULTI_ANGLE_TOOL_ID} supports only standard or 8-step Qwen Edit accelerator profiles; "
                f"got accelerator_profile_id={accelerator_profile_id!r}."
            )
        inputs = job.get("inputs") or []
        if not isinstance(inputs, list):
            raise ValueError("Job inputs must be a list.")
        reference_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "reference_image").strip() == "reference_image")
        control_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "").strip() == "control_image")
        if reference_count != 1 or control_count:
            raise ValueError(
                f"{QWEN_MULTI_ANGLE_TOOL_ID} requires exactly one reference_image input and no control_image inputs; "
                f"got reference_image={reference_count}, control_image={control_count}."
            )
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError(f"{QWEN_MULTI_ANGLE_TOOL_ID} requires an output object.")
        requested_resolution = self._resolve_job_resolution(job)
        if not requested_resolution:
            raise ValueError(f"{QWEN_MULTI_ANGLE_TOOL_ID} requires explicit output.width and output.height.")
        requested_size = self._parse_resolution_size(requested_resolution)
        if requested_size is None:
            raise ValueError(f"{QWEN_MULTI_ANGLE_TOOL_ID} could not resolve requested output resolution.")
        view_azimuth = str(generation.get("view_azimuth") or "").strip().lower()
        view_elevation = str(generation.get("view_elevation") or "").strip().lower()
        shot_distance = str(generation.get("shot_distance") or "").strip().lower()
        view_change_strength = str(
            generation.get("view_change_strength") or QWEN_MULTI_ANGLE_DEFAULT_VIEW_CHANGE_STRENGTH
        ).strip().lower()
        if view_azimuth not in QWEN_MULTI_ANGLE_AZIMUTHS:
            raise ValueError(f"Unsupported view_azimuth for {QWEN_MULTI_ANGLE_TOOL_ID}: {view_azimuth}")
        if view_elevation not in QWEN_MULTI_ANGLE_ELEVATIONS:
            raise ValueError(f"Unsupported view_elevation for {QWEN_MULTI_ANGLE_TOOL_ID}: {view_elevation}")
        if shot_distance not in QWEN_MULTI_ANGLE_DISTANCES:
            raise ValueError(f"Unsupported shot_distance for {QWEN_MULTI_ANGLE_TOOL_ID}: {shot_distance}")
        if view_change_strength not in QWEN_MULTI_ANGLE_VIEW_CHANGE_STRENGTHS:
            raise ValueError(f"Unsupported view_change_strength for {QWEN_MULTI_ANGLE_TOOL_ID}: {view_change_strength}")
        availability = self._qwen_multi_angle_tool_availability()
        if not availability["available"]:
            reason = str(availability["unavailable_reason"] or "unavailable")
            raise ValueError(
                f"Curated image tool {QWEN_MULTI_ANGLE_TOOL_ID!r} is not available on this WanGP worker: {reason}."
            )
        prompt_notes = str(generation.get("prompt_notes") or job.get("prompt") or "").strip()
        if len(prompt_notes) > MAX_PROMPT_CHARS:
            raise ValueError(f"{QWEN_MULTI_ANGLE_TOOL_ID} prompt_notes exceeds {MAX_PROMPT_CHARS} characters.")
        expanded_prompt = self._qwen_multi_angle_expanded_prompt(
            view_azimuth=view_azimuth,
            view_elevation=view_elevation,
            shot_distance=shot_distance,
            prompt_notes=prompt_notes,
            requested_size=requested_size,
        )
        lora_multiplier = QWEN_MULTI_ANGLE_VIEW_CHANGE_STRENGTHS[view_change_strength]
        return {
            "expanded_prompt": expanded_prompt,
            "settings": {
                "activated_loras": [str(availability["lora_relative_path"])],
                "loras_multipliers": lora_multiplier,
                "_midom_curated_tool_id": QWEN_MULTI_ANGLE_TOOL_ID,
                "_midom_curated_tool_display_name": QWEN_MULTI_ANGLE_TOOL_DISPLAY_NAME,
                "_midom_curated_tool_version": QWEN_MULTI_ANGLE_TOOL_VERSION,
                "_midom_curated_tool_base_model_id": QWEN_MULTI_ANGLE_BASE_MODEL_ID,
                "_midom_curated_tool_lora_filename": QWEN_MULTI_ANGLE_LORA_FILENAME,
                "_midom_curated_tool_lora_sha256": QWEN_MULTI_ANGLE_LORA_SHA256,
                "_midom_curated_tool_lora_license": QWEN_MULTI_ANGLE_LORA_LICENSE,
                "_midom_curated_tool_lora_multiplier": lora_multiplier,
                "_midom_curated_tool_parameters": {
                    "view_azimuth": view_azimuth,
                    "view_elevation": view_elevation,
                    "shot_distance": shot_distance,
                    "view_change_strength": view_change_strength,
                    "prompt_notes": prompt_notes,
                    "requested_resolution": requested_resolution,
                    "output_count": output_count,
                },
                "_midom_curated_tool_expanded_prompt": expanded_prompt,
            },
        }

    @staticmethod
    def _qwen_multi_angle_expanded_prompt(
        *,
        view_azimuth: str,
        view_elevation: str,
        shot_distance: str,
        prompt_notes: str,
        requested_size: tuple[int, int],
    ) -> str:
        requested_width, requested_height = requested_size
        if requested_width > requested_height:
            composition = "landscape"
        elif requested_height > requested_width:
            composition = "portrait"
        else:
            composition = "square"
        camera_terms = " ".join(
            [
                "<sks>",
                QWEN_MULTI_ANGLE_AZIMUTHS[view_azimuth],
                QWEN_MULTI_ANGLE_ELEVATIONS[view_elevation],
                QWEN_MULTI_ANGLE_DISTANCES[shot_distance],
            ]
        )
        canvas_terms = (
            f"Keep a {composition} {requested_width}x{requested_height} composition. "
            "Fill the full canvas naturally. Do not add borders, panels, letterboxing, pillarboxing, or change the image aspect ratio."
        )
        notes = re.sub(r"\s+", " ", str(prompt_notes or "")).strip()
        if notes:
            return f"{camera_terms}. {canvas_terms} {notes}"
        return (
            f"{camera_terms}. {canvas_terms} "
            "Preserve the same subject identity, outfit, materials, and visual style from the reference image."
        )

    def _is_event_video_processing_job(self, job: dict[str, Any]) -> bool:
        processing_task = str(job.get("processing_task") or "").strip().lower()
        processor_id = str(job.get("processor_id") or job.get("model_id") or "").strip()
        operation_type = self._storyboard_operation_type(job)
        return (
            operation_type not in STORYBOARD_FFMPEG_OPERATION_TYPES
            and (processing_task == EVENT_VIDEO_PROCESSING_TASK or processor_id == EVENT_VIDEO_PROCESSOR_ID)
        )

    def _is_storyboard_ffmpeg_processing_job(self, job: dict[str, Any]) -> bool:
        family = str(job.get("family") or "").strip().lower()
        processing_task = str(job.get("processing_task") or "").strip().lower()
        processor_id = str(job.get("processor_id") or job.get("model_id") or "").strip()
        operation_type = self._storyboard_operation_type(job)
        return (
            operation_type in STORYBOARD_FFMPEG_OPERATION_TYPES
            or processing_task == STORYBOARD_FFMPEG_PROCESSING_TASK
            or processor_id == STORYBOARD_FFMPEG_PROCESSOR_ID
            or (family == "media_processing" and operation_type in STORYBOARD_FFMPEG_OPERATION_TYPES)
        )

    @staticmethod
    def _storyboard_operation_type(payload: dict[str, Any]) -> str:
        processing = payload.get("processing") if isinstance(payload.get("processing"), dict) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        return str(
            payload.get("operation_type")
            or payload.get("operation")
            or payload.get("mediaassembly_operation_type")
            or payload.get("mediaassembly_operation")
            or payload.get("mediaassemblyjob_operation_type")
            or processing.get("operation_type")
            or processing.get("operation")
            or processing.get("mediaassembly_operation_type")
            or processing.get("mediaassembly_operation")
            or summary.get("operation_type")
            or summary.get("operation")
            or summary.get("mediaassembly_operation_type")
            or summary.get("mediaassembly_operation")
            or ""
        ).strip()

    def _validate_event_video_processing_job(self, job: dict[str, Any], job_id: int) -> dict[str, Any]:
        family = str(job.get("family") or "").strip().lower()
        if family != "media_processing":
            raise ValueError(f"Unsupported media processing family: {family}")
        media_type = str(job.get("media_type") or "").strip().lower()
        if media_type != "video":
            raise ValueError(f"Unsupported media processing media_type: {media_type}")
        processing_task = str(job.get("processing_task") or "").strip().lower()
        if processing_task != EVENT_VIDEO_PROCESSING_TASK:
            raise ValueError(f"Unsupported media processing task: {processing_task}")
        processor_id = str(job.get("processor_id") or job.get("model_id") or "").strip()
        if processor_id != EVENT_VIDEO_PROCESSOR_ID:
            raise ValueError(f"Unsupported media processor_id: {processor_id}")
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("Event Video Processing output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported Event Video Processing output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported Event Video Processing output count: {output_count}")
        output_format = str(output.get("format") or "mp4").strip().lower()
        if output_format != "mp4":
            raise ValueError(f"Unsupported Event Video Processing output format: {output_format}")
        output_profile = str(output.get("profile") or EVENT_VIDEO_OUTPUT_PROFILE).strip()
        if output_profile != EVENT_VIDEO_OUTPUT_PROFILE:
            raise ValueError(f"Unsupported Event Video Processing output profile: {output_profile}")
        processing = job.get("processing") or {}
        if not isinstance(processing, dict):
            raise ValueError("Event Video Processing processing payload must be a JSON object.")
        normalize = self._coerce_bool(processing.get("normalize"), True)
        optimize_for_web = self._coerce_bool(processing.get("optimize_for_web"), True)
        preserve_orientation = self._coerce_bool(processing.get("preserve_orientation"), True)
        if not normalize:
            raise ValueError("Event Video Processing requires processing.normalize = true.")
        if not optimize_for_web:
            raise ValueError("Event Video Processing requires processing.optimize_for_web = true.")
        if not preserve_orientation:
            raise ValueError("Event Video Processing requires processing.preserve_orientation = true.")
        apply_overlay = self._coerce_bool(processing.get("apply_overlay"), False)
        add_ending_bumper = self._coerce_bool(processing.get("add_ending_bumper"), False)
        target_max_width = self._coerce_int(processing.get("target_max_width"), 720, 1, 4096)
        target_max_height = self._coerce_int(processing.get("target_max_height"), 1280, 1, 4096)
        inputs = job.get("inputs") or []
        if not isinstance(inputs, list):
            raise ValueError("Event Video Processing inputs must be a list.")
        source_count = 0
        overlay_orientations = set()
        bumper_orientations = set()
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("Event Video Processing input descriptor must be a JSON object.")
            kind = str(item.get("kind") or "").strip()
            if kind not in ALLOWED_EVENT_INPUT_KINDS:
                raise ValueError(f"Unsupported Event Video Processing input kind: {kind}")
            if kind == "source_video":
                source_count += 1
                continue
            orientation = str(item.get("orientation") or "").strip().lower()
            if orientation not in {"portrait", "landscape"}:
                raise ValueError(f"Event Video Processing {kind} inputs require orientation portrait or landscape.")
            if kind == "overlay_png":
                overlay_orientations.add(orientation)
            elif kind == "bumper_image":
                bumper_orientations.add(orientation)
        if source_count != 1:
            raise ValueError(f"Event Video Processing requires exactly one source_video input; got {source_count}.")
        if not apply_overlay and overlay_orientations:
            raise ValueError("Event Video Processing received overlay_png inputs but apply_overlay is false.")
        if not add_ending_bumper and bumper_orientations:
            raise ValueError("Event Video Processing received bumper_image inputs but add_ending_bumper is false.")
        if apply_overlay and not overlay_orientations:
            raise ValueError("Event Video Processing apply_overlay is true but no overlay_png inputs were provided.")
        if add_ending_bumper and not bumper_orientations:
            raise ValueError("Event Video Processing add_ending_bumper is true but no bumper_image inputs were provided.")
        if not self._event_video_processing_capability():
            raise ValueError("Event Video Processing is unavailable because local ffmpeg/ffprobe support is incomplete.")
        self._log(
            "Validated claimed Event Video Processing job; "
            f"job_id={job_id} processor_id={processor_id} output_profile={output_profile!r} "
            f"target_max={target_max_width}x{target_max_height} apply_overlay={apply_overlay} "
            f"add_ending_bumper={add_ending_bumper} overlay_orientations={sorted(overlay_orientations)} "
            f"bumper_orientations={sorted(bumper_orientations)}."
        )
        return {
            "model_type": EVENT_VIDEO_PROCESSOR_ID,
            "_midom_job_id": job_id,
            "_midom_media_type": "media_processing",
            "_midom_processing_family": "media_processing",
            "_midom_processing_task": EVENT_VIDEO_PROCESSING_TASK,
            "_midom_processor_id": EVENT_VIDEO_PROCESSOR_ID,
            "_midom_output_count": 1,
            "_midom_output_format": "mp4",
            "_midom_output_mime_type": "video/mp4",
            "_midom_output_profile": output_profile,
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_EVENT_VIDEO_OUTPUT_BYTES, 1, MAX_EVENT_VIDEO_OUTPUT_BYTES),
            "_midom_processing": {
                "normalize": normalize,
                "optimize_for_web": optimize_for_web,
                "apply_overlay": apply_overlay,
                "add_ending_bumper": add_ending_bumper,
                "target_max_width": target_max_width,
                "target_max_height": target_max_height,
                "preserve_orientation": preserve_orientation,
            },
        }

    def _validate_storyboard_ffmpeg_processing_job(self, job: dict[str, Any], job_id: int) -> dict[str, Any]:
        family = str(job.get("family") or "").strip().lower()
        if family != "media_processing":
            raise ValueError(f"Unsupported storyboard processing family: {family}")
        media_type = str(job.get("media_type") or "").strip().lower()
        if media_type != "video":
            raise ValueError(f"Unsupported storyboard processing media_type: {media_type}")
        operation_type = self._storyboard_operation_type(job)
        if operation_type not in STORYBOARD_FFMPEG_OPERATION_TYPES:
            raise ValueError(f"Unsupported storyboard FFmpeg operation_type: {operation_type}")
        processing_task = str(job.get("processing_task") or STORYBOARD_FFMPEG_PROCESSING_TASK).strip().lower()
        if processing_task not in {"", STORYBOARD_FFMPEG_PROCESSING_TASK, "mediaassemblyjob", "mediaassembly_ffmpeg", "storyboard_ffmpeg"}:
            raise ValueError(f"Unsupported storyboard processing_task: {processing_task}")
        processor_id = str(job.get("processor_id") or job.get("model_id") or STORYBOARD_FFMPEG_PROCESSOR_ID).strip()
        if processor_id not in {"", STORYBOARD_FFMPEG_PROCESSOR_ID}:
            raise ValueError(f"Unsupported storyboard processor_id: {processor_id}")
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("Storyboard FFmpeg output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported storyboard output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported storyboard output count: {output_count}")
        output_format = str(output.get("format") or "mp4").strip().lower()
        if output_format != "mp4":
            raise ValueError(f"Unsupported storyboard output format: {output_format}")
        processing = job.get("processing") or {}
        if not isinstance(processing, dict):
            raise ValueError("Storyboard FFmpeg processing payload must be a JSON object.")
        operation_payload = job.get("operation_payload") or {}
        nested_operation_payload = processing.get("operation_payload") or {}
        if operation_payload and not isinstance(operation_payload, dict):
            raise ValueError("Storyboard FFmpeg operation_payload must be a JSON object when provided.")
        if nested_operation_payload and not isinstance(nested_operation_payload, dict):
            raise ValueError("Storyboard FFmpeg processing.operation_payload must be a JSON object when provided.")
        processing = {
            **(operation_payload if isinstance(operation_payload, dict) else {}),
            **(nested_operation_payload if isinstance(nested_operation_payload, dict) else {}),
            **{key: value for key, value in processing.items() if key != "operation_payload"},
        }
        width = self._coerce_int(output.get("width") or processing.get("width") or processing.get("output_width"), 0, 0, 4096)
        height = self._coerce_int(output.get("height") or processing.get("height") or processing.get("output_height"), 0, 0, 4096)
        if (width == 0) != (height == 0):
            raise ValueError("Storyboard output width and height must be supplied together.")
        trim_start = self._coerce_float(
            processing.get("trim_start_seconds", processing.get("start_seconds", processing.get("start_time_seconds"))),
            0.0,
            0.0,
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        trim_duration = self._optional_positive_float(
            processing.get("duration_seconds", processing.get("trim_duration_seconds")),
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        trim_end = self._optional_positive_float(
            processing.get("trim_end_seconds", processing.get("end_seconds", processing.get("end_time_seconds"))),
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        inputs = job.get("inputs") or []
        if not isinstance(inputs, list):
            raise ValueError("Storyboard FFmpeg inputs must be a list.")
        video_count = 0
        image_count = 0
        audio_count = 0
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("Storyboard FFmpeg input descriptor must be a JSON object.")
            kind = str(item.get("kind") or "").strip()
            if kind not in ALLOWED_STORYBOARD_INPUT_KINDS:
                raise ValueError(f"Unsupported storyboard FFmpeg input kind: {kind}")
            if kind in STORYBOARD_VIDEO_INPUT_KINDS:
                video_count += 1
            elif kind in STORYBOARD_IMAGE_INPUT_KINDS:
                image_count += 1
            elif kind in STORYBOARD_AUDIO_INPUT_KINDS:
                audio_count += 1
        if operation_type == "multicam_final_assembly":
            if video_count < 1:
                raise ValueError("Storyboard final assembly requires at least one video input.")
        elif operation_type == "replace_video_soundtrack":
            if video_count != 1:
                raise ValueError(f"Storyboard replace_video_soundtrack requires exactly one source video input; got {video_count}.")
            if audio_count != 1:
                raise ValueError(f"Storyboard replace_video_soundtrack requires exactly one soundtrack audio input; got {audio_count}.")
        elif operation_type in STORYBOARD_SINGLE_VIDEO_OPERATION_TYPES and video_count < 1:
            raise ValueError(f"Storyboard operation {operation_type} requires at least one video input.")
        if not self._storyboard_ffmpeg_processing_capability():
            raise ValueError("Storyboard FFmpeg Processing is unavailable because local ffmpeg/ffprobe support is incomplete.")
        self._log(
            "Validated claimed Storyboard FFmpeg Processing job; "
            f"job_id={job_id} operation_type={operation_type} output={width}x{height} "
            f"trim_start={trim_start} trim_duration={trim_duration} trim_end={trim_end} "
            f"video_inputs={video_count} image_inputs={image_count} audio_inputs={audio_count}."
        )
        return {
            "model_type": STORYBOARD_FFMPEG_PROCESSOR_ID,
            "_midom_job_id": job_id,
            "_midom_media_type": "media_processing",
            "_midom_processing_family": "media_processing",
            "_midom_processing_task": STORYBOARD_FFMPEG_PROCESSING_TASK,
            "_midom_processor_id": STORYBOARD_FFMPEG_PROCESSOR_ID,
            "_midom_operation_type": operation_type,
            "_midom_output_count": 1,
            "_midom_output_format": "mp4",
            "_midom_output_mime_type": "video/mp4",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_STORYBOARD_VIDEO_OUTPUT_BYTES, 1, MAX_STORYBOARD_VIDEO_OUTPUT_BYTES),
            "_midom_processing": {
                **processing,
                "operation_type": operation_type,
                "output_width": width,
                "output_height": height,
                "trim_start_seconds": trim_start,
                "trim_duration_seconds": trim_duration,
                "trim_end_seconds": trim_end,
            },
        }

    def _validate_audio_job(self, job: dict[str, Any], job_id: int) -> dict[str, Any]:
        model_type = str(job.get("model_id") or job.get("model_type") or job.get("model") or "").strip()
        if model_type not in ALLOWED_AUDIO_MODEL_TYPES:
            raise ValueError(f"Unsupported audio model_type: {model_type}")
        if self._is_stable_audio3_model(model_type):
            return self._validate_stable_audio3_job(job, job_id, model_type)
        if self._is_ace_step15_music_model(model_type):
            return self._validate_ace_step15_music_job(job, job_id)
        if model_type == SEEDVC_MODEL_ID:
            return self._validate_seedvc_voice_replacement_job(job, job_id)
        generation = job.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError("Audio job generation must be a JSON object when provided.")
        options = self._audio_generation_options(generation)
        audio_task = str(generation.get("audio_task") or "").strip().lower()
        if model_type == DRAMABOX_MODEL_ID:
            if audio_task != "dialogue_voice_clone_tts":
                raise ValueError(f"Unsupported DramaBox audio_task: {audio_task}")
        elif audio_task != "voice_clone_tts":
            raise ValueError(f"Unsupported audio_task: {audio_task}")
        voice_mode = str(generation.get("voice_mode") or "").strip().lower()
        if model_type == DRAMABOX_MODEL_ID:
            if voice_mode != "two_voice_clone_n_voice_dialogue":
                raise ValueError(f"Unsupported DramaBox voice_mode: {voice_mode}")
        elif voice_mode not in {"single_reference", "two_speaker_dialogue"}:
            raise ValueError(f"Unsupported voice_mode: {voice_mode}")
        if model_type == CHATTERBOX_MODEL_ID and voice_mode != "single_reference":
            raise ValueError("Chatterbox supports only single_reference voice clone jobs.")
        dialogue_mode = str(generation.get("dialogue_mode") or "").strip().lower()
        if model_type == DRAMABOX_MODEL_ID:
            if dialogue_mode != "dramabox_speaker_blocks":
                raise ValueError(f"Unsupported DramaBox dialogue_mode: {dialogue_mode}")
        elif voice_mode == "two_speaker_dialogue" and dialogue_mode != "speaker_tags":
            raise ValueError(f"Unsupported dialogue_mode: {dialogue_mode}")
        if model_type == CHATTERBOX_MODEL_ID and dialogue_mode:
            raise ValueError(f"Chatterbox does not support dialogue_mode: {dialogue_mode}")
        prompt_processing_mode = self._prompt_processing_mode(options, force_fg=(voice_mode in {"two_speaker_dialogue", "two_voice_clone_n_voice_dialogue"}))
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("Audio job output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported audio output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported audio output count: {output_count}")
        output_format = str(output.get("format") or "mp3").strip().lower()
        if output_format not in {"mp3", "wav"}:
            raise ValueError(f"Unsupported audio output format: {output_format}")
        if output_format == "mp3" and "audio/mpeg" not in self._audio_output_mime_types():
            raise ValueError("MP3 output requested but MP3 transcoding is unavailable on this WanGP worker.")
        prompt = str(job.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Audio prompt is required.")
        if model_type == CHATTERBOX_MODEL_ID:
            self._validate_chatterbox_prompt(prompt, prompt_processing_mode)
        elif model_type == DRAMABOX_MODEL_ID:
            if len(prompt) > MAX_PROMPT_CHARS:
                raise ValueError(f"DramaBox prompt exceeds {MAX_PROMPT_CHARS} characters.")
            self._validate_dramabox_prompt(prompt)
        elif len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Audio prompt exceeds {MAX_PROMPT_CHARS} characters.")
        duration_value = generation.get("duration_seconds")
        if duration_value is None:
            duration_value = output.get("max_duration_seconds")
        if model_type == DRAMABOX_MODEL_ID:
            duration_seconds = self._coerce_int(duration_value, 0, 0, DRAMABOX_MAX_DURATION_SECONDS)
        else:
            duration_seconds = self._coerce_int(duration_value, MAX_AUDIO_DURATION_SECONDS, 1, MAX_AUDIO_DURATION_SECONDS)
        inputs = job.get("inputs") or []
        reference_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "reference_audio")
        emotion_reference_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "emotion_reference_audio")
        speaker1_reference_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "speaker1_reference_audio")
        speaker2_reference_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "speaker2_reference_audio")
        if voice_mode in {"two_speaker_dialogue", "two_voice_clone_n_voice_dialogue"}:
            if voice_mode == "two_speaker_dialogue" and model_type != "index_tts2":
                raise ValueError(f"Model {model_type} does not support two-speaker dialogue jobs.")
            if voice_mode == "two_voice_clone_n_voice_dialogue" and model_type != DRAMABOX_MODEL_ID:
                raise ValueError(f"Model {model_type} does not support two voice clone N-voice dialogue jobs.")
            if reference_audio_count or emotion_reference_audio_count:
                raise ValueError("Dialogue jobs must use speaker1_reference_audio and speaker2_reference_audio inputs.")
            if speaker1_reference_audio_count != 1:
                raise ValueError(f"Dialogue job requires exactly one speaker1_reference_audio input; got {speaker1_reference_audio_count}.")
            if speaker2_reference_audio_count != 1:
                raise ValueError(f"Dialogue job requires exactly one speaker2_reference_audio input; got {speaker2_reference_audio_count}.")
            if voice_mode == "two_speaker_dialogue":
                self._validate_dialogue_prompt(prompt)
        else:
            if speaker1_reference_audio_count or speaker2_reference_audio_count:
                raise ValueError("Single-reference audio jobs must not use speaker dialogue input kinds.")
            if reference_audio_count != 1:
                raise ValueError(f"Audio job requires exactly one reference_audio input; got {reference_audio_count}.")
            if model_type == CHATTERBOX_MODEL_ID and emotion_reference_audio_count:
                raise ValueError("Chatterbox does not support emotion_reference_audio inputs.")
            if emotion_reference_audio_count > 1:
                raise ValueError(f"Audio job supports at most one emotion_reference_audio input; got {emotion_reference_audio_count}.")
        language = str(generation.get("language") or "").strip().lower()
        custom_settings: dict[str, Any] = {}
        if model_type == CHATTERBOX_MODEL_ID:
            if not language:
                language = "en" if "en" in CHATTERBOX_LANGUAGES else CHATTERBOX_LANGUAGES[0]
            if language not in CHATTERBOX_LANGUAGES:
                raise ValueError(f"Unsupported Chatterbox language: {language}")
            for control_name, control_def in CHATTERBOX_CONTROL_LIMITS.items():
                custom_settings[control_name] = self._coerce_float(
                    options.get(control_name),
                    float(control_def["default"]),
                    float(control_def["min"]),
                    float(control_def["max"]),
                )
        elif model_type == DRAMABOX_MODEL_ID:
            custom_settings["duration_multiplier"] = self._coerce_float(
                options.get("duration_multiplier"),
                DRAMABOX_DEFAULT_DURATION_MULTIPLIER,
                DRAMABOX_MIN_DURATION_MULTIPLIER,
                DRAMABOX_MAX_DURATION_MULTIPLIER,
            )
            custom_settings["remove_unexpected_words"] = self._coerce_bool(options.get("remove_unexpected_words"), False)
        for item in inputs:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind not in {"reference_audio", "emotion_reference_audio", "speaker1_reference_audio", "speaker2_reference_audio"}:
                raise ValueError(f"Unsupported audio job input kind: {kind}")
        emotion_mode = str(generation.get("emotion_mode") or "none").strip().lower()
        if model_type == CHATTERBOX_MODEL_ID and emotion_mode not in {"", "none"}:
            raise ValueError(f"Chatterbox does not support emotion_mode: {emotion_mode}")
        if model_type == DRAMABOX_MODEL_ID and emotion_mode not in {"", "none"}:
            raise ValueError(f"DramaBox does not support emotion_mode: {emotion_mode}")
        self._log(
            "Validated claimed audio job; "
            f"job_id={job_id} model_type={model_type} voice_mode={voice_mode!r} "
            f"dialogue_mode={dialogue_mode!r} output_format={output_format} "
            f"duration_seconds={duration_seconds} reference_audio_count={reference_audio_count} "
            f"emotion_reference_audio_count={emotion_reference_audio_count} "
            f"speaker1_reference_audio_count={speaker1_reference_audio_count} "
            f"speaker2_reference_audio_count={speaker2_reference_audio_count} "
            f"emotion_mode={emotion_mode!r} "
            f"prompt_processing_mode={prompt_processing_mode!r} "
            f"prompt_chars={len(prompt)}."
        )
        self._log(
            "Audio prompt/script summary; "
            f"job_id={job_id} {self._audio_prompt_summary(prompt)}."
        )
        if voice_mode == "two_speaker_dialogue":
            audio_prompt_type = "AB2"
        elif model_type == DRAMABOX_MODEL_ID:
            audio_prompt_type = "AB0" if custom_settings.get("remove_unexpected_words") else "AB"
        else:
            audio_prompt_type = "A"
        settings = {
            "model_type": model_type,
            "prompt": prompt,
            "audio_prompt_type": audio_prompt_type,
            "video_length": 0,
            "repeat_generation": 1,
            "multi_prompts_gen_type": prompt_processing_mode,
            "negative_prompt": "",
            "duration_seconds": duration_seconds,
            "pause_seconds": 0.2 if voice_mode == "two_speaker_dialogue" else 0,
            "_midom_job_id": job_id,
            "_midom_media_type": "audio",
            "_midom_requested_seed": self._optional_seed(generation.get("seed")),
            "_midom_voice_mode": voice_mode,
            "_midom_output_count": 1,
            "_midom_output_format": output_format,
            "_midom_output_mime_type": "audio/mpeg" if output_format == "mp3" else "audio/wav",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_AUDIO_BYTES, 1, MAX_AUDIO_BYTES),
            "_midom_prompt_processing_mode": prompt_processing_mode,
        }
        if model_type == CHATTERBOX_MODEL_ID:
            settings.update({
                "model_mode": language,
                "custom_settings": custom_settings,
                "temperature": 0.8,
                "guidance_scale": 1.0,
            })
        elif model_type == DRAMABOX_MODEL_ID:
            settings.update({
                "custom_settings": {"duration_multiplier": custom_settings["duration_multiplier"]},
                "num_inference_steps": 30,
                "guidance_scale": 2.5,
                "audio_guidance_scale": 1.5,
                "alt_scale": 0.0,
            })
        return settings

    def _validate_stable_audio3_job(self, job: dict[str, Any], job_id: int, model_id: str) -> dict[str, Any]:
        generation = job.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError("Stable Audio 3 job generation must be a JSON object when provided.")
        options = self._audio_generation_options(generation)
        audio_task = str(generation.get("audio_task") or job.get("audio_task") or "").strip().lower()
        if audio_task != "generative_audio":
            raise ValueError(f"Unsupported Stable Audio 3 audio_task: {audio_task}")
        audio_category = str(generation.get("audio_category") or job.get("audio_category") or "").strip().lower()
        if audio_category != "music_and_sounds":
            raise ValueError(f"Unsupported Stable Audio 3 audio_category: {audio_category}")
        expected_subtype = self._stable_audio3_subtype(model_id)
        audio_subtype = str(generation.get("audio_subtype") or job.get("audio_subtype") or expected_subtype).strip().lower()
        if audio_subtype != expected_subtype:
            raise ValueError(f"Unsupported Stable Audio 3 audio_subtype for {model_id}: {audio_subtype}")
        for unsupported_key in ("voice_mode", "dialogue_mode", "emotion_mode"):
            value = str(generation.get(unsupported_key) or "").strip().lower()
            if value and value != "none":
                raise ValueError(f"Stable Audio 3 does not support {unsupported_key}: {value}")
        requested_prompt_mode = str(options.get("multi_prompts_gen_type") or DEFAULT_PROMPT_PROCESSING_MODE).strip().upper()
        if requested_prompt_mode != DEFAULT_PROMPT_PROCESSING_MODE:
            raise ValueError(f"Stable Audio 3 supports only {DEFAULT_PROMPT_PROCESSING_MODE} prompt processing.")
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("Stable Audio 3 job output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported Stable Audio 3 output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported Stable Audio 3 output count: {output_count}")
        output_format = str(output.get("format") or "mp3").strip().lower()
        if output_format not in {"mp3", "wav"}:
            raise ValueError(f"Unsupported Stable Audio 3 output format: {output_format}")
        if output_format == "mp3" and "audio/mpeg" not in self._audio_output_mime_types():
            raise ValueError("MP3 output requested but MP3 transcoding is unavailable on this WanGP worker.")
        inputs = job.get("inputs") or []
        if not isinstance(inputs, list):
            raise ValueError("Stable Audio 3 job inputs must be a list.")
        if inputs:
            raise ValueError("Stable Audio 3 first-pass jobs do not support inputs.")
        prompt = str(job.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Stable Audio 3 prompt is required.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Stable Audio 3 prompt exceeds {MAX_PROMPT_CHARS} characters.")
        negative_prompt = str(options.get("negative_prompt") if options.get("negative_prompt") is not None else job.get("negative_prompt") or "").strip()
        if len(negative_prompt) > STABLE_AUDIO3_MAX_NEGATIVE_PROMPT_CHARS:
            raise ValueError(f"Stable Audio 3 negative prompt exceeds {STABLE_AUDIO3_MAX_NEGATIVE_PROMPT_CHARS} characters.")
        duration_value = generation.get("duration_seconds")
        if duration_value is None:
            duration_value = output.get("max_duration_seconds")
        duration_seconds = self._coerce_int(
            duration_value,
            self._stable_audio3_default_duration(model_id),
            1,
            STABLE_AUDIO3_MAX_DURATION_SECONDS,
        )
        sample_solver = str(options.get("sample_solver") or "pingpong").strip().lower()
        if sample_solver not in STABLE_AUDIO3_SAMPLE_SOLVERS:
            raise ValueError(f"Unsupported Stable Audio 3 sample_solver: {sample_solver}")
        num_inference_steps = self._coerce_int(options.get("num_inference_steps"), 8, 1, 50)
        guidance_scale = self._coerce_float(options.get("guidance_scale"), 1.0, 0.0, 20.0)
        requested_seed = self._optional_seed(generation.get("seed"))
        self._log(
            "Validated claimed Stable Audio 3 job; "
            f"job_id={job_id} model_id={model_id} wangp_model_type={self._stable_audio3_wangp_model_type(model_id)} "
            f"audio_subtype={audio_subtype!r} output_format={output_format} duration_seconds={duration_seconds} "
            f"sample_solver={sample_solver!r} steps={num_inference_steps} guidance_scale={guidance_scale} "
            f"prompt_chars={len(prompt)} negative_prompt_chars={len(negative_prompt)}."
        )
        settings = {
            "model_type": self._stable_audio3_wangp_model_type(model_id),
            "prompt": prompt,
            "audio_prompt_type": "",
            "video_length": 0,
            "repeat_generation": 1,
            "multi_prompts_gen_type": DEFAULT_PROMPT_PROCESSING_MODE,
            "negative_prompt": negative_prompt,
            "duration_seconds": duration_seconds,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "sample_solver": sample_solver,
            "audio_scale": 0.9,
            "_midom_job_id": job_id,
            "_midom_media_type": "audio",
            "_midom_model_id": model_id,
            "_midom_audio_task": "generative_audio",
            "_midom_audio_category": "music_and_sounds",
            "_midom_audio_subtype": audio_subtype,
            "_midom_requested_seed": requested_seed,
            "_midom_voice_mode": "",
            "_midom_output_count": 1,
            "_midom_output_format": output_format,
            "_midom_output_mime_type": "audio/mpeg" if output_format == "mp3" else "audio/wav",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_AUDIO_BYTES, 1, MAX_AUDIO_BYTES),
            "_midom_prompt_processing_mode": DEFAULT_PROMPT_PROCESSING_MODE,
        }
        if requested_seed is not None:
            settings["seed"] = requested_seed
        return settings

    def _validate_ace_step15_music_job(self, job: dict[str, Any], job_id: int) -> dict[str, Any]:
        generation = job.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError("ACE-Step 1.5 job generation must be a JSON object when provided.")
        options = self._audio_generation_options(generation)
        audio_task = str(generation.get("audio_task") or job.get("audio_task") or "").strip().lower()
        if audio_task != "generative_audio":
            raise ValueError(f"Unsupported ACE-Step 1.5 audio_task: {audio_task}")
        audio_category = str(generation.get("audio_category") or job.get("audio_category") or "").strip().lower()
        if audio_category != "music_and_sounds":
            raise ValueError(f"Unsupported ACE-Step 1.5 audio_category: {audio_category}")
        audio_subtype = str(generation.get("audio_subtype") or job.get("audio_subtype") or "music").strip().lower()
        if audio_subtype != "music":
            raise ValueError(f"Unsupported ACE-Step 1.5 audio_subtype: {audio_subtype}")
        for unsupported_key in ("voice_mode", "dialogue_mode", "emotion_mode"):
            value = str(generation.get(unsupported_key) or "").strip().lower()
            if value and value != "none":
                raise ValueError(f"ACE-Step 1.5 music does not support {unsupported_key}: {value}")
        requested_prompt_mode = str(options.get("multi_prompts_gen_type") or DEFAULT_PROMPT_PROCESSING_MODE).strip().upper()
        if requested_prompt_mode != DEFAULT_PROMPT_PROCESSING_MODE:
            raise ValueError(f"ACE-Step 1.5 supports only {DEFAULT_PROMPT_PROCESSING_MODE} prompt processing.")
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("ACE-Step 1.5 job output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported ACE-Step 1.5 output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported ACE-Step 1.5 output count: {output_count}")
        output_format = str(output.get("format") or "mp3").strip().lower()
        if output_format not in {"mp3", "wav"}:
            raise ValueError(f"Unsupported ACE-Step 1.5 output format: {output_format}")
        if output_format == "mp3" and "audio/mpeg" not in self._audio_output_mime_types():
            raise ValueError("MP3 output requested but MP3 transcoding is unavailable on this WanGP worker.")
        inputs = job.get("inputs") or []
        if not isinstance(inputs, list):
            raise ValueError("ACE-Step 1.5 job inputs must be a list.")
        if inputs:
            raise ValueError("ACE-Step 1.5 instrumental first-pass jobs do not support inputs.")
        music_caption = str(job.get("prompt") or "").strip()
        if not music_caption:
            raise ValueError("ACE-Step 1.5 music caption prompt is required.")
        if len(music_caption) > ACE_STEP15_MAX_MUSIC_CAPTION_CHARS:
            raise ValueError(f"ACE-Step 1.5 music caption exceeds {ACE_STEP15_MAX_MUSIC_CAPTION_CHARS} characters.")
        lyrics = str(options.get("lyrics") or ACE_STEP15_INSTRUMENTAL_LYRICS).strip()
        if len(lyrics) > ACE_STEP15_MAX_LYRICS_CHARS:
            raise ValueError(f"ACE-Step 1.5 lyrics prompt exceeds {ACE_STEP15_MAX_LYRICS_CHARS} characters.")
        if lyrics.lower() != ACE_STEP15_INSTRUMENTAL_LYRICS.lower():
            raise ValueError("ACE-Step 1.5 first-pass support is instrumental-only; lyrics must be [Instrumental].")
        duration_value = generation.get("duration_seconds")
        if duration_value is None:
            duration_value = output.get("max_duration_seconds")
        duration_seconds = self._coerce_int(
            duration_value,
            ACE_STEP15_DEFAULT_DURATION_SECONDS,
            ACE_STEP15_MIN_DURATION_SECONDS,
            ACE_STEP15_MAX_DURATION_SECONDS,
        )
        custom_settings: dict[str, Any] = {}
        bpm_value = options.get("bpm")
        if str(bpm_value or "").strip():
            custom_settings["bpm"] = self._coerce_int(bpm_value, ACE_STEP15_BPM_MIN, ACE_STEP15_BPM_MIN, ACE_STEP15_BPM_MAX)
        keyscale = str(options.get("keyscale") or "").strip()
        if keyscale:
            if len(keyscale) > 64:
                raise ValueError("ACE-Step 1.5 keyscale exceeds 64 characters.")
            custom_settings["keyscale"] = keyscale
        timesignature_value = options.get("timesignature")
        if str(timesignature_value or "").strip():
            try:
                timesignature = int(timesignature_value)
            except (TypeError, ValueError):
                raise ValueError(f"Unsupported ACE-Step 1.5 timesignature: {timesignature_value}")
            if timesignature not in ACE_STEP15_TIME_SIGNATURE_VALUES:
                raise ValueError(f"Unsupported ACE-Step 1.5 timesignature: {timesignature}")
            custom_settings["timesignature"] = timesignature
        language = str(options.get("language") or "").strip().lower()
        if language:
            if language not in ACE_STEP15_LANGUAGES:
                raise ValueError(f"Unsupported ACE-Step 1.5 language: {language}")
            custom_settings["language"] = language
        requested_seed = self._optional_seed(generation.get("seed"))
        self._log(
            "Validated claimed ACE-Step 1.5 music job; "
            f"job_id={job_id} output_format={output_format} duration_seconds={duration_seconds} "
            f"caption_chars={len(music_caption)} lyrics={lyrics!r} custom_settings={custom_settings!r}."
        )
        settings = {
            "model_type": ACE_STEP15_WANGP_MODEL_TYPE,
            "prompt": ACE_STEP15_INSTRUMENTAL_LYRICS,
            "alt_prompt": music_caption,
            "audio_prompt_type": "",
            "video_length": 0,
            "repeat_generation": 1,
            "multi_prompts_gen_type": DEFAULT_PROMPT_PROCESSING_MODE,
            "negative_prompt": "",
            "duration_seconds": duration_seconds,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "alt_guidance_scale": 2.5,
            "alt_scale": 0.0,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 0,
            "audio_scale": 1.0,
            "custom_settings": custom_settings,
            "_midom_job_id": job_id,
            "_midom_media_type": "audio",
            "_midom_model_id": ACE_STEP15_MUSIC_MODEL_ID,
            "_midom_audio_task": "generative_audio",
            "_midom_audio_category": "music_and_sounds",
            "_midom_audio_subtype": "music",
            "_midom_requested_seed": requested_seed,
            "_midom_voice_mode": "",
            "_midom_output_count": 1,
            "_midom_output_format": output_format,
            "_midom_output_mime_type": "audio/mpeg" if output_format == "mp3" else "audio/wav",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_AUDIO_BYTES, 1, MAX_AUDIO_BYTES),
            "_midom_prompt_processing_mode": DEFAULT_PROMPT_PROCESSING_MODE,
        }
        if requested_seed is not None:
            settings["seed"] = requested_seed
        return settings

    def _validate_seedvc_voice_replacement_job(self, job: dict[str, Any], job_id: int) -> dict[str, Any]:
        if not self._seedvc_speech_available():
            raise ValueError("SeedVC speech voice replacement is not enabled in this WanGP worker.")
        generation = job.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError("SeedVC job generation must be a JSON object when provided.")
        options = self._audio_generation_options(generation)
        audio_task = str(generation.get("audio_task") or job.get("audio_task") or "").strip().lower()
        if audio_task != "voice_conversion":
            raise ValueError(f"Unsupported SeedVC audio_task: {audio_task}")
        audio_category = str(generation.get("audio_category") or job.get("audio_category") or "").strip().lower()
        if audio_category != "voice_replacement":
            raise ValueError(f"Unsupported SeedVC audio_category: {audio_category}")
        voice_mode = str(generation.get("voice_mode") or "").strip().lower()
        if voice_mode != "one_speaker_voice_replacement":
            raise ValueError(f"Unsupported SeedVC voice_mode: {voice_mode}")
        seedvc_method = str(options.get("seedvc_method") or SEEDVC_METHOD_ONE_SPEAKER).strip().lower()
        if seedvc_method != SEEDVC_METHOD_ONE_SPEAKER:
            raise ValueError(f"Unsupported SeedVC method: {seedvc_method}")
        for unsupported_key in ("dialogue_mode", "emotion_mode", "language"):
            value = str(generation.get(unsupported_key) or "").strip().lower()
            if value and value != "none":
                raise ValueError(f"SeedVC does not support {unsupported_key}: {value}")
        prompt_processing_mode = str(options.get("multi_prompts_gen_type") or "").strip().upper()
        if prompt_processing_mode:
            raise ValueError("SeedVC does not support prompt processing modes.")
        prompt = str(job.get("prompt") or "").strip()
        if prompt:
            raise ValueError("SeedVC voice replacement does not use prompt text.")
        if generation.get("duration_seconds") is not None:
            raise ValueError("SeedVC voice replacement does not use generation.duration_seconds.")
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("SeedVC job output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported SeedVC output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported SeedVC output count: {output_count}")
        output_format = str(output.get("format") or "mp3").strip().lower()
        if output_format not in {"mp3", "wav"}:
            raise ValueError(f"Unsupported SeedVC output format: {output_format}")
        if output_format == "mp3" and "audio/mpeg" not in self._audio_output_mime_types():
            raise ValueError("MP3 output requested but MP3 transcoding is unavailable on this WanGP worker.")
        inputs = job.get("inputs") or []
        if not isinstance(inputs, list):
            raise ValueError("SeedVC job inputs must be a list.")
        source_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "source_audio")
        reference_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "reference_audio")
        emotion_reference_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "emotion_reference_audio")
        speaker1_reference_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "speaker1_reference_audio")
        speaker2_reference_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "speaker2_reference_audio")
        if source_audio_count != 1:
            raise ValueError(f"SeedVC requires exactly one source_audio input; got {source_audio_count}.")
        if reference_audio_count != 1:
            raise ValueError(f"SeedVC requires exactly one reference_audio input; got {reference_audio_count}.")
        if emotion_reference_audio_count or speaker1_reference_audio_count or speaker2_reference_audio_count:
            raise ValueError("SeedVC first-pass jobs must not include emotion or speaker reference inputs.")
        for item in inputs:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind not in {"source_audio", "reference_audio"}:
                raise ValueError(f"Unsupported SeedVC job input kind: {kind}")
        self._log(
            "Validated claimed SeedVC voice replacement job; "
            f"job_id={job_id} output_format={output_format} "
            f"source_audio_count={source_audio_count} reference_audio_count={reference_audio_count}."
        )
        return {
            "model_type": SEEDVC_MODEL_ID,
            "prompt": "",
            "postprocess_audio": SEEDVC_METHOD_ONE_SPEAKER,
            "replace_voice_sample": None,
            "replace_voice_sample2": None,
            "audio_source": None,
            "_midom_job_id": job_id,
            "_midom_media_type": "audio",
            "_midom_model_id": SEEDVC_MODEL_ID,
            "_midom_audio_task": "voice_conversion",
            "_midom_audio_category": "voice_replacement",
            "_midom_seedvc_method": SEEDVC_METHOD_ONE_SPEAKER,
            "_midom_seedvc_runtime_mode": SEEDVC_RUNTIME_MODE_SPEECH,
            "_midom_requested_seed": None,
            "_midom_voice_mode": "one_speaker_voice_replacement",
            "_midom_output_count": 1,
            "_midom_output_format": output_format,
            "_midom_output_mime_type": "audio/mpeg" if output_format == "mp3" else "audio/wav",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_AUDIO_BYTES, 1, MAX_AUDIO_BYTES),
            "_midom_prompt_processing_mode": "",
        }

    def _validate_video_job(self, job: dict[str, Any], job_id: int) -> dict[str, Any]:
        model_type = str(job.get("model_id") or job.get("model_type") or job.get("model") or "").strip()
        if model_type not in ALLOWED_VIDEO_MODEL_TYPES:
            raise ValueError(f"Unsupported video model_type: {model_type}")
        if model_type in LTX_VIDEO_MODEL_IDS:
            return self._validate_ltx_video_job(job, job_id, model_type)
        if model_type == SVI_VIDEO_MODEL_ID:
            return self._validate_svi_video_job(job, job_id, model_type)
        generation = job.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError("Video job generation must be a JSON object when provided.")
        video_task = str(generation.get("video_task") or "").strip().lower()
        if video_task != "talking_avatar":
            raise ValueError(f"Unsupported video_task for {model_type}: {video_task}")
        speed_profile_id = str(generation.get("speed_profile_id") or "distilled_8_step").strip() or "distilled_8_step"
        if speed_profile_id != "distilled_8_step":
            raise ValueError(f"Unsupported LongCat speed_profile_id: {speed_profile_id}")
        prompt_mode = str(generation.get("prompt_mode") or "plain").strip().lower()
        if prompt_mode != "plain":
            raise ValueError(f"Unsupported LongCat prompt_mode: {prompt_mode}")
        duration_mode = str(generation.get("duration_mode") or LONGCAT_DURATION_MODE).strip().lower()
        if duration_mode != LONGCAT_DURATION_MODE:
            raise ValueError(f"Unsupported LongCat duration_mode: {duration_mode}")
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("Video job output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported video output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported video output count: {output_count}")
        output_format = str(output.get("format") or "mp4").strip().lower()
        if output_format != "mp4":
            raise ValueError(f"Unsupported video output format: {output_format}")
        prompt = str(job.get("prompt") or "").strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Video prompt exceeds {MAX_PROMPT_CHARS} characters.")
        if not prompt:
            raise ValueError("Video prompt is required.")
        duration_value = generation.get("duration_seconds")
        if duration_value is None:
            duration_value = output.get("max_duration_seconds")
        duration_seconds = self._coerce_int(duration_value, 5, 1, MAX_LONGCAT_VIDEO_DURATION_SECONDS)
        inputs = job.get("inputs") or []
        reference_image_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "reference_image")
        driving_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "driving_audio")
        if reference_image_count != 1:
            raise ValueError(f"LongCat video job requires exactly one reference_image input; got {reference_image_count}.")
        if driving_audio_count != 1:
            raise ValueError(f"LongCat video job requires exactly one driving_audio input; got {driving_audio_count}.")
        for item in inputs:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind not in {"reference_image", "driving_audio"}:
                raise ValueError(f"Unsupported LongCat video input kind: {kind}")
        resolution = self._resolve_video_resolution(output)
        video_length = self._longcat_video_length_for_duration(duration_seconds)
        settings = {
            "model_type": model_type,
            "prompt": prompt,
            "negative_prompt": str(job.get("negative_prompt") or "Close-up, overexposed, static, blurred details, distorted mouth, extra teeth.").strip()[:MAX_PROMPT_CHARS],
            "resolution": resolution,
            "image_mode": 0,
            "video_length": video_length,
            "duration_seconds": duration_seconds,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "sample_solver": "distill",
            "video_prompt_type": "KI",
            "audio_prompt_type": "A",
            "sliding_window_size": 93,
            "sliding_window_overlap": 13,
            "repeat_generation": 1,
            "multi_prompts_gen_type": "FG",
            "multi_images_gen_type": "SI",
            "force_fps": str(LONGCAT_VIDEO_FPS),
            "_api": {"return_media": True},
            "_midom_job_id": job_id,
            "_midom_media_type": "video",
            "_midom_requested_seed": self._optional_seed(generation.get("seed")),
            "_midom_video_task": video_task,
            "_midom_output_count": 1,
            "_midom_output_format": "mp4",
            "_midom_output_mime_type": "video/mp4",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_VIDEO_BYTES, 1, MAX_VIDEO_BYTES),
            "_midom_speed_profile_id": speed_profile_id,
            "_midom_prompt_mode": prompt_mode,
            "_midom_duration_mode": duration_mode,
        }
        self._log(
            "Validated claimed LongCat video job; "
            f"job_id={job_id} model_type={model_type} video_task={video_task!r} "
            f"duration_seconds={duration_seconds} video_length={video_length} resolution={resolution} "
            f"duration_mode={duration_mode!r} speed_profile_id={speed_profile_id!r} prompt_chars={len(prompt)} "
            f"reference_image_count={reference_image_count} driving_audio_count={driving_audio_count}."
        )
        return settings

    def _validate_svi_video_job(self, job: dict[str, Any], job_id: int, model_type: str) -> dict[str, Any]:
        generation = job.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError("SVI video job generation must be a JSON object when provided.")
        video_task = str(generation.get("video_task") or "").strip().lower()
        if video_task != "cinematic_i2v":
            raise ValueError(f"Unsupported SVI video_task: {video_task}")
        speed_profile_id = str(generation.get("speed_profile_id") or SVI_SPEED_PROFILE_ID).strip() or SVI_SPEED_PROFILE_ID
        if speed_profile_id != SVI_SPEED_PROFILE_ID:
            raise ValueError(f"Unsupported SVI speed_profile_id: {speed_profile_id}")
        prompt_mode = str(generation.get("prompt_mode") or "plain").strip().lower()
        if prompt_mode not in {"plain", "timed_cinematic_seconds"}:
            raise ValueError(f"Unsupported SVI prompt_mode: {prompt_mode}")
        duration_mode = str(generation.get("duration_mode") or SVI_DURATION_MODE).strip().lower()
        if duration_mode != SVI_DURATION_MODE:
            raise ValueError(f"Unsupported SVI duration_mode: {duration_mode}")
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("SVI video job output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported SVI output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported SVI output count: {output_count}")
        output_format = str(output.get("format") or "mp4").strip().lower()
        if output_format != "mp4":
            raise ValueError(f"Unsupported SVI output format: {output_format}")
        prompt = str(job.get("prompt") or "").strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"SVI prompt exceeds {MAX_PROMPT_CHARS} characters.")
        if not prompt:
            raise ValueError("SVI prompt is required.")
        duration_value = generation.get("duration_seconds")
        if duration_value is None:
            duration_value = output.get("max_duration_seconds")
        if duration_value is None:
            duration_seconds = DEFAULT_SVI_VIDEO_DURATION_SECONDS
        else:
            try:
                duration_seconds = int(duration_value)
            except (TypeError, ValueError):
                raise ValueError(f"Unsupported SVI duration_seconds: {duration_value}")
        if duration_seconds < MIN_SVI_VIDEO_DURATION_SECONDS or duration_seconds > MAX_SVI_VIDEO_DURATION_SECONDS:
            raise ValueError(
                f"SVI v1 supports {MIN_SVI_VIDEO_DURATION_SECONDS}-{MAX_SVI_VIDEO_DURATION_SECONDS} second jobs; "
                f"got {duration_seconds}."
            )
        resolution = self._resolve_video_resolution(output, model_type)
        video_length = self._svi_video_length_for_duration(duration_seconds)
        inputs = job.get("inputs") or []
        start_image_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "start_image")
        end_image_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "end_image")
        if start_image_count != 1:
            raise ValueError(f"SVI video job requires exactly one start_image input; got {start_image_count}.")
        if end_image_count > 1:
            raise ValueError(f"SVI video job supports at most one end_image input; got {end_image_count}.")
        for item in inputs:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind not in {"start_image", "end_image"}:
                raise ValueError(f"Unsupported SVI video input kind: {kind}")
        settings = {
            "model_type": model_type,
            "prompt": prompt,
            "negative_prompt": str(job.get("negative_prompt") or "").strip()[:MAX_PROMPT_CHARS],
            "resolution": resolution,
            "image_mode": 0,
            "image_prompt_type": "S",
            "video_prompt_type": "",
            "audio_prompt_type": "",
            "video_length": video_length,
            "duration_seconds": duration_seconds,
            "num_inference_steps": 8,
            "guidance_phases": 2,
            "switch_threshold": 900,
            "guidance_scale": 1.0,
            "guidance2_scale": 1.0,
            "flow_shift": 5,
            "sample_solver": "unipc",
            "sliding_window_size": 81,
            "sliding_window_overlap": 4,
            "repeat_generation": 1,
            "multi_prompts_gen_type": "FG",
            "multi_images_gen_type": "SI",
            "force_fps": str(SVI_VIDEO_FPS),
            "input_video_strength": 1.0,
            "apg_switch": 0,
            "cfg_star_switch": 0,
            "perturbation_switch": 0,
            "self_refiner_setting": 0,
            "_api": {"return_media": True},
            "_midom_job_id": job_id,
            "_midom_media_type": "video",
            "_midom_requested_seed": self._optional_seed(generation.get("seed")),
            "_midom_video_task": video_task,
            "_midom_audio_video_mode": "",
            "_midom_output_count": 1,
            "_midom_output_format": "mp4",
            "_midom_output_mime_type": "video/mp4",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_VIDEO_BYTES, 1, MAX_VIDEO_BYTES),
            "_midom_speed_profile_id": speed_profile_id,
            "_midom_prompt_mode": prompt_mode,
            "_midom_duration_mode": duration_mode,
        }
        self._log(
            "Validated claimed SVI cinematic I2V job; "
            f"job_id={job_id} model_type={model_type} video_task={video_task!r} "
            f"duration_seconds={duration_seconds} video_length={video_length} resolution={resolution} "
            f"duration_mode={duration_mode!r} speed_profile_id={speed_profile_id!r} "
            f"prompt_mode={prompt_mode!r} prompt_chars={len(prompt)} "
            f"start_image_count={start_image_count} end_image_count={end_image_count}."
        )
        return settings

    def _validate_ltx_video_job(self, job: dict[str, Any], job_id: int, model_type: str) -> dict[str, Any]:
        generation = job.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError("LTX video job generation must be a JSON object when provided.")
        video_task = str(generation.get("video_task") or "").strip().lower()
        if video_task == "control_video_guided_video":
            return self._validate_ltx_control_video_job(job, job_id, model_type, generation)
        if video_task != "audio_conditioned_video":
            raise ValueError(f"Unsupported LTX video_task: {video_task}")
        audio_video_mode = str(generation.get("audio_video_mode") or "").strip().lower()
        if audio_video_mode != "driving_audio_guided":
            raise ValueError(f"Unsupported LTX audio_video_mode: {audio_video_mode}")
        speed_profile_id = str(generation.get("speed_profile_id") or "standard").strip() or "standard"
        if speed_profile_id != "standard":
            raise ValueError(f"Unsupported LTX speed_profile_id: {speed_profile_id}")
        prompt_mode = str(generation.get("prompt_mode") or "plain").strip().lower()
        if prompt_mode != "plain":
            raise ValueError(f"Unsupported LTX prompt_mode: {prompt_mode}")
        duration_mode = str(generation.get("duration_mode") or LTX_DURATION_MODE).strip().lower()
        if duration_mode != LTX_DURATION_MODE:
            raise ValueError(f"Unsupported LTX duration_mode: {duration_mode}")
        video_sync_profile_id = str(generation.get("video_sync_profile_id") or "standard").strip() or "standard"
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("LTX video job output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported LTX output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported LTX output count: {output_count}")
        output_format = str(output.get("format") or "mp4").strip().lower()
        if output_format != "mp4":
            raise ValueError(f"Unsupported LTX output format: {output_format}")
        prompt = str(job.get("prompt") or "").strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"LTX prompt exceeds {MAX_PROMPT_CHARS} characters.")
        if not prompt:
            raise ValueError("LTX prompt is required.")
        duration_value = generation.get("duration_seconds")
        if duration_value is None:
            duration_value = output.get("max_duration_seconds")
        duration_seconds = self._coerce_int(duration_value, 5, 1, MAX_LTX_VIDEO_DURATION_SECONDS)
        resolution = self._resolve_video_resolution(output, model_type)
        video_length = self._ltx_video_length_for_duration(duration_seconds)
        inputs = job.get("inputs") or []
        start_image_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "start_image")
        end_image_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "end_image")
        driving_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "driving_audio")
        if start_image_count != 1:
            raise ValueError(f"LTX video job requires exactly one start_image input; got {start_image_count}.")
        if end_image_count > 1:
            raise ValueError(f"LTX video job supports at most one end_image input; got {end_image_count}.")
        if driving_audio_count != 1:
            raise ValueError(f"LTX video job requires exactly one driving_audio input; got {driving_audio_count}.")
        for item in inputs:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind not in {"start_image", "end_image", "driving_audio"}:
                raise ValueError(f"Unsupported LTX video input kind: {kind}")
        settings = {
            "model_type": model_type,
            "prompt": prompt,
            "negative_prompt": str(job.get("negative_prompt") or "").strip()[:MAX_PROMPT_CHARS],
            "resolution": resolution,
            "image_mode": 0,
            "image_prompt_type": "SE" if end_image_count else "S",
            "video_prompt_type": "",
            "audio_prompt_type": "A",
            "video_length": video_length,
            "duration_seconds": duration_seconds,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "audio_cfg_scale": 1.0,
            "alt_guidance_scale": 1.0,
            "alt_scale": 0.0,
            "sample_solver": "distilled_8_steps",
            "guidance_phases": 2,
            "sliding_window_size": 481,
            "sliding_window_overlap": 17,
            "repeat_generation": 1,
            "multi_prompts_gen_type": "FG",
            "multi_images_gen_type": "SI",
            "force_fps": str(LTX_VIDEO_FPS),
            "input_video_strength": 1.0,
            "audio_scale": 1.0,
            "apg_switch": 0,
            "cfg_star_switch": 0,
            "perturbation_switch": 0,
            "self_refiner_setting": 0,
            "_api": {"return_media": True},
            "_midom_job_id": job_id,
            "_midom_media_type": "video",
            "_midom_requested_seed": self._optional_seed(generation.get("seed")),
            "_midom_video_task": video_task,
            "_midom_audio_video_mode": audio_video_mode,
            "_midom_output_count": 1,
            "_midom_output_format": "mp4",
            "_midom_output_mime_type": "video/mp4",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_VIDEO_BYTES, 1, MAX_VIDEO_BYTES),
            "_midom_speed_profile_id": speed_profile_id,
            "_midom_prompt_mode": prompt_mode,
            "_midom_duration_mode": duration_mode,
        }
        self._apply_ltx_delivery_adapter(settings)
        self._apply_ltx_video_sync_profile(settings, video_sync_profile_id)
        self._log(
            "Validated claimed LTX video job; "
            f"job_id={job_id} model_type={model_type} video_task={video_task!r} "
            f"audio_video_mode={audio_video_mode!r} duration_seconds={duration_seconds} "
            f"video_length={video_length} requested_resolution={settings.get('_midom_requested_resolution')} "
            f"internal_resolution={settings.get('resolution')} duration_mode={duration_mode!r} "
            f"speed_profile_id={speed_profile_id!r} video_sync_profile_id={video_sync_profile_id!r} "
            f"prompt_chars={len(prompt)} start_image_count={start_image_count} "
            f"end_image_count={end_image_count} driving_audio_count={driving_audio_count}."
        )
        return settings

    def _validate_ltx_control_video_job(
        self,
        job: dict[str, Any],
        job_id: int,
        model_type: str,
        generation: dict[str, Any],
    ) -> dict[str, Any]:
        video_task = "control_video_guided_video"
        audio_video_mode = str(generation.get("audio_video_mode") or "").strip().lower()
        if audio_video_mode != "control_video_audio_guided":
            raise ValueError(f"Unsupported LTX control-video audio_video_mode: {audio_video_mode}")
        speed_profile_id = str(generation.get("speed_profile_id") or "standard").strip() or "standard"
        if speed_profile_id != "standard":
            raise ValueError(f"Unsupported LTX control-video speed_profile_id: {speed_profile_id}")
        prompt_mode = str(generation.get("prompt_mode") or "plain").strip().lower()
        if prompt_mode != "plain":
            raise ValueError(f"Unsupported LTX control-video prompt_mode: {prompt_mode}")
        duration_mode = str(generation.get("duration_mode") or LTX_CONTROL_VIDEO_DURATION_MODE).strip().lower()
        if duration_mode != LTX_CONTROL_VIDEO_DURATION_MODE:
            raise ValueError(f"Unsupported LTX control-video duration_mode: {duration_mode}")
        video_sync_profile_id = str(generation.get("video_sync_profile_id") or "standard").strip() or "standard"
        output = job.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("LTX control-video job output must be a JSON object.")
        try:
            output_count = int(output.get("count") or 1)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported LTX control-video output count: {output.get('count')}")
        if output_count != 1:
            raise ValueError(f"Unsupported LTX control-video output count: {output_count}")
        output_format = str(output.get("format") or "mp4").strip().lower()
        if output_format != "mp4":
            raise ValueError(f"Unsupported LTX control-video output format: {output_format}")
        prompt = str(job.get("prompt") or "").strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"LTX control-video prompt exceeds {MAX_PROMPT_CHARS} characters.")
        if not prompt:
            raise ValueError("LTX control-video prompt is required.")
        options = generation.get("options") or {}
        if options and not isinstance(options, dict):
            raise ValueError("LTX control-video generation.options must be a JSON object when provided.")
        control_video_mode = str(generation.get("control_video_mode") or (options or {}).get("control_video_mode") or "").strip().lower()
        if control_video_mode not in LTX_CONTROL_VIDEO_MODES:
            raise ValueError(f"Unsupported LTX control_video_mode: {control_video_mode}")
        duration_value = generation.get("duration_seconds")
        if duration_value is None:
            duration_value = output.get("max_duration_seconds")
        requested_duration_seconds = self._coerce_int(duration_value, MAX_LTX_VIDEO_DURATION_SECONDS, 1, MAX_LTX_VIDEO_DURATION_SECONDS)
        resolution = self._resolve_video_resolution(output, model_type)
        video_length = self._ltx_video_length_for_duration(requested_duration_seconds)
        inputs = job.get("inputs") or []
        start_image_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "start_image")
        end_image_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "end_image")
        driving_audio_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "driving_audio")
        control_video_count = sum(1 for item in inputs if isinstance(item, dict) and str(item.get("kind") or "") == "control_video")
        if start_image_count != 1:
            raise ValueError(f"LTX control-video job requires exactly one start_image input; got {start_image_count}.")
        if control_video_count != 1:
            raise ValueError(f"LTX control-video job requires exactly one control_video input; got {control_video_count}.")
        if driving_audio_count:
            raise ValueError("LTX control-video jobs must not include a separate driving_audio input.")
        if end_image_count:
            raise ValueError("LTX control-video first pass does not support end_image inputs.")
        for item in inputs:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind not in {"start_image", "control_video"}:
                raise ValueError(f"Unsupported LTX control-video input kind: {kind}")
        settings = {
            "model_type": model_type,
            "prompt": prompt,
            "negative_prompt": str(job.get("negative_prompt") or "").strip()[:MAX_PROMPT_CHARS],
            "resolution": resolution,
            "image_mode": 0,
            "image_prompt_type": "S",
            "video_prompt_type": LTX_CONTROL_VIDEO_MODES[control_video_mode],
            "audio_prompt_type": "K",
            "video_length": video_length,
            "duration_seconds": requested_duration_seconds,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "audio_cfg_scale": 1.0,
            "alt_guidance_scale": 1.0,
            "alt_scale": 0.0,
            "sample_solver": "distilled_8_steps",
            "guidance_phases": 2,
            "sliding_window_size": 481,
            "sliding_window_overlap": 17,
            "repeat_generation": 1,
            "multi_prompts_gen_type": "FG",
            "multi_images_gen_type": "SI",
            "force_fps": str(LTX_VIDEO_FPS),
            "input_video_strength": 1.0,
            "denoising_strength": 1.0,
            "masking_strength": 0,
            "audio_scale": 1.0,
            "apg_switch": 0,
            "cfg_star_switch": 0,
            "perturbation_switch": 0,
            "self_refiner_setting": 0,
            "_api": {"return_media": True},
            "_midom_job_id": job_id,
            "_midom_media_type": "video",
            "_midom_requested_seed": self._optional_seed(generation.get("seed")),
            "_midom_video_task": video_task,
            "_midom_audio_video_mode": audio_video_mode,
            "_midom_control_video_mode": control_video_mode,
            "_midom_output_count": 1,
            "_midom_output_format": "mp4",
            "_midom_output_mime_type": "video/mp4",
            "_midom_max_artifact_bytes": self._coerce_int((job.get("limits") or {}).get("max_artifact_bytes"), MAX_VIDEO_BYTES, 1, MAX_VIDEO_BYTES),
            "_midom_speed_profile_id": speed_profile_id,
            "_midom_prompt_mode": prompt_mode,
            "_midom_duration_mode": duration_mode,
        }
        self._apply_ltx_delivery_adapter(settings)
        self._apply_ltx_video_sync_profile(settings, video_sync_profile_id)
        self._log(
            "Validated claimed LTX control-video job; "
            f"job_id={job_id} model_type={model_type} control_video_mode={control_video_mode!r} "
            f"audio_video_mode={audio_video_mode!r} requested_duration_seconds={requested_duration_seconds} "
            f"video_length={video_length} requested_resolution={settings.get('_midom_requested_resolution')} "
            f"internal_resolution={settings.get('resolution')} duration_mode={duration_mode!r} "
            f"speed_profile_id={speed_profile_id!r} video_sync_profile_id={video_sync_profile_id!r} "
            f"prompt_chars={len(prompt)} start_image_count={start_image_count} control_video_count={control_video_count}."
        )
        return settings

    def _validate_dialogue_prompt(self, prompt: str) -> None:
        lines = [line.strip() for line in str(prompt or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
        if not lines:
            raise ValueError("Dialogue prompt is required.")
        speaker_ids = set()
        for line in lines:
            match = re.match(r"^Speaker\s+([12]):\s*(.+)$", line)
            if not match:
                raise ValueError("Dialogue prompt lines must start with exactly 'Speaker 1:' or 'Speaker 2:'.")
            text = match.group(2).strip()
            if not text:
                raise ValueError("Dialogue prompt contains an empty speaker line.")
            speaker_ids.add(int(match.group(1)))
        if speaker_ids != {1, 2}:
            raise ValueError("Dialogue prompt must include both Speaker 1 and Speaker 2.")

    def _validate_dramabox_prompt(self, prompt: str) -> None:
        lines = [line.strip() for line in str(prompt or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
        if not lines:
            raise ValueError("DramaBox prompt is required.")
        speaker_ids = set()
        current_speaker = None
        for line in lines:
            header_match = re.match(r"^Speaker\s+(\d+)\s*(?:\{[^\n{}]*\})?\s*:\s*(.*)$", line, re.IGNORECASE)
            if header_match:
                speaker_id = int(header_match.group(1))
                if speaker_id < 1 or speaker_id > DRAMABOX_MAX_DIALOGUE_SPEAKERS:
                    raise ValueError(f"DramaBox speaker id must be between 1 and {DRAMABOX_MAX_DIALOGUE_SPEAKERS}.")
                speaker_ids.add(speaker_id)
                current_speaker = speaker_id
                continue
            if current_speaker is None:
                raise ValueError("DramaBox dialogue prompts must start with a Speaker N: block.")
        if not {1, 2}.issubset(speaker_ids):
            raise ValueError("DramaBox prompt must include both Speaker 1: and Speaker 2: blocks.")
        if len(speaker_ids) > DRAMABOX_MAX_DIALOGUE_SPEAKERS:
            raise ValueError(f"DramaBox prompt supports up to {DRAMABOX_MAX_DIALOGUE_SPEAKERS} scripted speakers.")

    @staticmethod
    def _audio_prompt_summary(prompt: Any) -> str:
        text = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
        non_empty_lines = [line.strip() for line in text.split("\n") if line.strip()]
        speaker1_lines = sum(1 for line in non_empty_lines if re.match(r"^Speaker\s+1:", line))
        speaker2_lines = sum(1 for line in non_empty_lines if re.match(r"^Speaker\s+2:", line))
        emotion_tags = re.findall(r"\[[^\]\n]{1,80}\]", text)
        preview = re.sub(r"\s+", " ", text).strip()
        if len(preview) > 140:
            preview = f"{preview[:137]}..."
        return (
            f"chars={len(text)} lines={len(non_empty_lines)} "
            f"speaker1_lines={speaker1_lines} speaker2_lines={speaker2_lines} "
            f"emotion_tags={len(emotion_tags)} preview={preview!r}"
        )

    @staticmethod
    def _audio_generation_options(generation: dict[str, Any]) -> dict[str, Any]:
        options = generation.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("Audio job generation.options must be a JSON object when provided.")
        return options

    @staticmethod
    def _is_stable_audio3_model(model_id: str) -> bool:
        return str(model_id or "").strip() in STABLE_AUDIO3_MODEL_IDS

    @staticmethod
    def _is_ace_step15_music_model(model_id: str) -> bool:
        return str(model_id or "").strip() == ACE_STEP15_MUSIC_MODEL_ID

    @staticmethod
    def _stable_audio3_wangp_model_type(model_id: str) -> str:
        model_id = str(model_id or "").strip()
        if model_id not in STABLE_AUDIO3_WANGP_MODEL_TYPES:
            raise ValueError(f"Unsupported Stable Audio 3 model_id: {model_id}")
        return STABLE_AUDIO3_WANGP_MODEL_TYPES[model_id]

    @staticmethod
    def _stable_audio3_subtype(model_id: str) -> str:
        return "sound_effect" if str(model_id or "").strip() == STABLE_AUDIO3_SFX_MODEL_ID else "music"

    @staticmethod
    def _stable_audio3_default_duration(model_id: str) -> int:
        return 8 if str(model_id or "").strip() == STABLE_AUDIO3_SFX_MODEL_ID else 30

    @staticmethod
    def _prompt_processing_mode(options: dict[str, Any], *, force_fg: bool = False) -> str:
        if force_fg:
            return DEFAULT_PROMPT_PROCESSING_MODE
        mode = str(options.get("multi_prompts_gen_type") or DEFAULT_PROMPT_PROCESSING_MODE).strip().upper()
        if mode not in PROMPT_PROCESSING_MODES:
            raise ValueError(f"Unsupported audio multi_prompts_gen_type: {mode}")
        return mode

    @staticmethod
    def _prompt_chunks_for_mode(prompt: str, prompt_processing_mode: str) -> list[str]:
        text = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
        if prompt_processing_mode == "FG":
            return [text.strip()] if text.strip() else []
        if prompt_processing_mode == "PG":
            chunks: list[str] = []
            current_lines: list[str] = []
            for raw_line in text.split("\n"):
                if not raw_line.strip():
                    if current_lines:
                        chunks.append("\n".join(current_lines).strip())
                        current_lines = []
                    continue
                current_lines.append(raw_line.rstrip())
            if current_lines:
                chunks.append("\n".join(current_lines).strip())
            return [chunk for chunk in chunks if chunk]
        return [line.strip() for line in text.split("\n") if line.strip()]

    def _validate_chatterbox_prompt(self, prompt: str, prompt_processing_mode: str) -> None:
        if len(prompt) > CHATTERBOX_MAX_PROMPT_CHARS:
            raise ValueError(f"Chatterbox prompt exceeds {CHATTERBOX_MAX_PROMPT_CHARS} characters.")
        chunks = self._prompt_chunks_for_mode(prompt, prompt_processing_mode)
        if not chunks:
            raise ValueError("Chatterbox prompt is required.")
        if prompt_processing_mode == "FG":
            if len(chunks[0]) > CHATTERBOX_MAX_PROMPT_CHARS_FG:
                raise ValueError(f"Chatterbox FG prompt exceeds {CHATTERBOX_MAX_PROMPT_CHARS_FG} characters.")
            return
        for index, chunk in enumerate(chunks, start=1):
            if len(chunk) > CHATTERBOX_MAX_PROMPT_CHARS_PER_SPLIT:
                label = "paragraph" if prompt_processing_mode == "PG" else "line"
                raise ValueError(
                    f"Chatterbox prompt {label} {index} exceeds "
                    f"{CHATTERBOX_MAX_PROMPT_CHARS_PER_SPLIT} characters."
                )

    @staticmethod
    def _coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(maximum, number))

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    def _coerce_job_id(self, job: dict[str, Any]) -> int:
        try:
            job_id = int(job.get("job_id"))
        except (TypeError, ValueError):
            raise ValueError("Job id is required.")
        if job_id <= 0:
            raise ValueError("Job id is required.")
        return job_id

    def _worker_id(self, config: Any) -> int:
        if isinstance(config, ConnectionContext):
            return int(config.worker_id)
        try:
            worker_id = int(config.get("worker_id"))
        except (TypeError, ValueError):
            raise gr.Error("Pair this worker before starting it.")
        if worker_id <= 0:
            raise gr.Error("Pair this worker before starting it.")
        return worker_id

    def _configured_api_base_url(self, config: Any) -> str:
        if isinstance(config, ConnectionContext):
            return config.api_base_url
        return self._normalize_api_base_url(
            str(config.get("api_base_url") or "").strip().rstrip("/"),
            bool(config.get("allow_insecure_local_dev", False)),
            bool(config.get("allow_insecure_lan_dev", False)),
        )

    def _connection_scope_value(self, connection_or_config: Any, key: str) -> str:
        if isinstance(connection_or_config, ConnectionContext):
            mapping = {
                "worker_id": connection_or_config.worker_id,
                "org_id": connection_or_config.org_id,
                "project_id": connection_or_config.project_id,
                "paired_user_id": connection_or_config.paired_user_id,
            }
            return str(mapping.get(key) or "").strip()
        if isinstance(connection_or_config, dict):
            return str(connection_or_config.get(key) or "").strip()
        return ""

    def _validate_job_scope(self, job: dict[str, Any], config: Any) -> None:
        target = job.get("target") or {}
        if not isinstance(target, dict):
            raise ValueError("Job target must be a JSON object when provided.")
        for config_key, target_key in (
            ("org_id", "org_id"),
            ("project_id", "project_id"),
            ("paired_user_id", "requested_by_user_id"),
        ):
            expected = self._connection_scope_value(config, config_key)
            actual = str(target.get(target_key) or "").strip()
            if not expected or not actual or expected != actual:
                raise ValueError(f"Job target {target_key} does not match paired worker scope.")

    def _resolve_job_resolution(self, job: dict[str, Any]) -> str:
        output = job.get("output") or {}
        resolution = ""
        if isinstance(output, dict):
            width = output.get("width")
            height = output.get("height")
            if width and height:
                resolution = f"{int(width)}x{int(height)}"
        if not resolution:
            return ""
        if resolution not in ALLOWED_RESOLUTIONS:
            raise ValueError(f"Unsupported resolution: {resolution}")
        return resolution

    def _resolve_video_resolution(self, output: dict[str, Any], model_type: str = "longcat_avatar_v1_5") -> str:
        resolution = ""
        if isinstance(output, dict):
            width = output.get("width")
            height = output.get("height")
            if width and height:
                resolution = f"{int(width)}x{int(height)}"
        if model_type in LTX_VIDEO_MODEL_IDS:
            if not resolution:
                return "1280x720"
            if resolution not in LTX_VIDEO_RESOLUTIONS:
                raise ValueError(f"Unsupported LTX video resolution: {resolution}")
            return resolution
        if model_type == SVI_VIDEO_MODEL_ID:
            if not resolution:
                return "1280x720"
            if resolution not in SVI_VIDEO_RESOLUTIONS:
                raise ValueError(f"Unsupported SVI video resolution: {resolution}")
            return resolution
        if not resolution:
            return "832x480"
        if resolution not in LONGCAT_VIDEO_RESOLUTIONS:
            raise ValueError(f"Unsupported LongCat video resolution: {resolution}")
        return resolution

    @staticmethod
    def _is_ltx_video_model(model_type: Any) -> bool:
        return str(model_type or "").strip() in LTX_VIDEO_MODEL_IDS

    def _apply_ltx_delivery_adapter(self, settings: dict[str, Any]) -> None:
        if not self._is_ltx_video_model(settings.get("model_type")):
            return
        requested_resolution = str(settings.get("resolution") or "").strip()
        model_type = str(settings.get("model_type") or "").strip()
        internal_resolution = LTX_OVERSCAN_RENDER_RESOLUTIONS.get(model_type, {}).get(requested_resolution)
        if not internal_resolution:
            raise ValueError(f"Unsupported LTX delivery resolution: {requested_resolution}")
        requested_size = self._parse_resolution_size(requested_resolution)
        internal_size = self._parse_resolution_size(internal_resolution)
        if requested_size is None or internal_size is None:
            raise ValueError(f"Could not resolve LTX delivery sizes for {requested_resolution}.")
        if internal_size[0] < requested_size[0] or internal_size[1] < requested_size[1]:
            raise ValueError(
                "LTX internal render resolution must be at least the requested Midom delivery resolution; "
                f"requested={requested_resolution} internal={internal_resolution}."
            )
        settings["_midom_requested_resolution"] = requested_resolution
        settings["_midom_ltx_internal_resolution"] = internal_resolution
        settings["_midom_ltx_delivery_adapter"] = "overscan_center_crop"
        settings["resolution"] = internal_resolution

    @staticmethod
    def _longcat_video_length_for_duration(duration_seconds: int) -> int:
        requested_frames = max(1, int(duration_seconds) * LONGCAT_VIDEO_FPS)
        # LongCat/WanGP expects frame counts aligned to 4n + 1.
        return ((requested_frames - 1) // 4) * 4 + 1

    @staticmethod
    def _ltx_video_length_for_duration(duration_seconds: int) -> int:
        requested_frames = max(1, int(duration_seconds) * LTX_VIDEO_FPS)
        # LTX-2.3 expects frame counts aligned to 8n + 1.
        return ((requested_frames + 7) // 8) * 8 + 1

    @staticmethod
    def _svi_video_length_for_duration(duration_seconds: int) -> int:
        requested_frames = max(1, int(duration_seconds) * SVI_VIDEO_FPS)
        # Wan I2V/SVI uses frame counts aligned to 4n + 1.
        return ((requested_frames + 3) // 4) * 4 + 1

    @staticmethod
    def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(maximum, number))

    @staticmethod
    def _coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(float(minimum), min(float(maximum), number))

    @classmethod
    def _optional_positive_float(cls, value: Any, maximum: float) -> Optional[float]:
        if value is None or value == "":
            return None
        number = cls._coerce_float(value, 0.0, 0.0, maximum)
        return number if number > 0 else None

    def _coerce_input_id(self, item: dict[str, Any]) -> int:
        try:
            input_id = int(item.get("input_id"))
        except (TypeError, ValueError):
            raise ValueError("Job input descriptor did not include a valid input_id.")
        if input_id <= 0:
            raise ValueError("Job input descriptor did not include a valid input_id.")
        return input_id

    def _max_input_bytes_for_mime(self, mime_type: str) -> int:
        mime_type = str(mime_type or "").split(";", 1)[0].strip().lower()
        if mime_type in ALLOWED_IMAGE_MIME_TYPES:
            return MAX_IMAGE_BYTES
        if mime_type in ALLOWED_AUDIO_INPUT_MIME_TYPES:
            return MAX_AUDIO_BYTES
        if mime_type in ALLOWED_VIDEO_OUTPUT_MIME_TYPES:
            return MAX_VIDEO_BYTES
        return MAX_IMAGE_BYTES

    def _read_limited_response_content(self, response: requests.Response, input_id: int, max_bytes: int) -> bytes:
        content_length = str(response.headers.get("Content-Length") or "").strip()
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                raise ValueError(f"Input {input_id} returned an invalid Content-Length.")
            if declared_size < 0 or declared_size > max_bytes:
                raise ValueError(f"Input {input_id} exceeds maximum allowed download size.")
        data = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ValueError(f"Input {input_id} exceeds maximum allowed download size.")
        finally:
            response.close()
        return bytes(data)

    def _optional_seed(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            text = str(value).strip()
            if text == "":
                return None
            number = int(text)
        except (TypeError, ValueError):
            return None
        if number < 0:
            return None
        return max(0, min(2_147_483_647, number))

    def _extract_seed_from_text(self, text: Any) -> Optional[int]:
        match = SEED_FILENAME_RE.search(str(text or ""))
        if not match:
            return None
        return self._optional_seed(match.group(1))

    def _extract_seed_from_path(self, file_path: Any) -> Optional[int]:
        if not file_path:
            return None
        path = Path(str(file_path))
        return self._extract_seed_from_text(path.name)

    def _extract_seed_from_result(self, result) -> tuple[Optional[int], str]:
        for attr in ("seed", "resolved_seed", "generation_seed"):
            seed = self._optional_seed(getattr(result, attr, None))
            if seed is not None:
                return seed, "wangp_result"
        for attr in ("metadata", "generation_metadata", "settings"):
            value = getattr(result, attr, None)
            if isinstance(value, dict):
                for key in ("seed", "resolved_seed", "generation_seed"):
                    seed = self._optional_seed(value.get(key))
                    if seed is not None:
                        return seed, "wangp_result"
        return None, "unknown"

    def _build_generation_metadata(self, settings: dict[str, Any], result, generated_files: list[str]) -> dict[str, Any]:
        requested_seed = self._optional_seed(settings.get("_midom_requested_seed"))
        artifact_seeds = []
        for artifact_index, file_path in enumerate(generated_files):
            seed = self._extract_seed_from_path(file_path)
            if seed is not None:
                artifact_seeds.append(
                    {
                        "artifact_index": int(artifact_index),
                        "seed": int(seed),
                        "seed_source": "wangp_filename",
                    }
                )
        resolved_seed, seed_source = self._extract_seed_from_result(result)
        if resolved_seed is None and artifact_seeds:
            resolved_seed = int(artifact_seeds[0]["seed"])
            seed_source = str(artifact_seeds[0]["seed_source"])
        if resolved_seed is None and requested_seed is not None:
            resolved_seed = requested_seed
            seed_source = "midom_request"
        metadata: dict[str, Any] = {
            "requested_seed": requested_seed,
            "resolved_seed": resolved_seed,
            "seed_source": seed_source if resolved_seed is not None else "unknown",
        }
        if artifact_seeds:
            metadata["artifact_seeds"] = artifact_seeds
        if str(settings.get("model_type") or "") == "qwen_image_layered_20B":
            metadata["image_task"] = "layer_decomposition"
            metadata["layer_mode"] = "control_image_decomposition"
            metadata["layer_output_count"] = self._coerce_int(
                settings.get("_midom_layer_output_count"),
                len(generated_files),
                1,
                QWEN_LAYERED_MAX_OUTPUTS,
            )
            metadata["artifact_roles"] = [
                {
                    "artifact_index": int(artifact_index),
                    "role": "source_echo" if artifact_index == 0 else "decomposition_layer",
                    **({} if artifact_index == 0 else {"layer_index": int(artifact_index)}),
                }
                for artifact_index, _file_path in enumerate(generated_files)
            ]
        curated_tool_id = str(settings.get("_midom_curated_tool_id") or "").strip()
        if curated_tool_id:
            metadata["curated_tool"] = {
                "tool_id": curated_tool_id,
                "display_name": str(settings.get("_midom_curated_tool_display_name") or ""),
                "tool_version": str(settings.get("_midom_curated_tool_version") or ""),
                "recipe_version": str(settings.get("_midom_curated_tool_version") or ""),
                "base_model_id": str(settings.get("_midom_curated_tool_base_model_id") or settings.get("model_type") or ""),
                "lora_filename": str(settings.get("_midom_curated_tool_lora_filename") or ""),
                "lora_sha256": str(settings.get("_midom_curated_tool_lora_sha256") or ""),
                "lora_license": str(settings.get("_midom_curated_tool_lora_license") or ""),
                "lora_multiplier": str(settings.get("_midom_curated_tool_lora_multiplier") or ""),
                "accelerator_profile_id": str(settings.get("_midom_accelerator_profile_id") or "standard"),
                "parameters": settings.get("_midom_curated_tool_parameters") if isinstance(settings.get("_midom_curated_tool_parameters"), dict) else {},
                "expanded_prompt": str(settings.get("_midom_curated_tool_expanded_prompt") or ""),
            }
        combined_audio_segment_count = self._coerce_int(settings.get("_midom_combined_audio_segment_count"), 0, 0, 10_000)
        if combined_audio_segment_count > 1:
            metadata["audio_prompt_processing_mode"] = str(settings.get("_midom_prompt_processing_mode") or "")
            metadata["combined_audio_segment_count"] = combined_audio_segment_count
        if str(settings.get("_midom_audio_task") or "") == "generative_audio":
            metadata["audio_task"] = "generative_audio"
            metadata["audio_category"] = str(settings.get("_midom_audio_category") or "music_and_sounds")
            metadata["audio_subtype"] = str(settings.get("_midom_audio_subtype") or "")
        if str(settings.get("_midom_audio_task") or "") == "voice_conversion":
            metadata["audio_task"] = "voice_conversion"
            metadata["audio_category"] = str(settings.get("_midom_audio_category") or "voice_replacement")
            metadata["voice_mode"] = str(settings.get("_midom_voice_mode") or "one_speaker_voice_replacement")
            metadata["seedvc_method"] = str(settings.get("_midom_seedvc_method") or SEEDVC_METHOD_ONE_SPEAKER)
            metadata["seedvc_runtime_mode"] = str(settings.get("_midom_seedvc_runtime_mode") or SEEDVC_RUNTIME_MODE_SPEECH)
        if str(settings.get("_midom_ltx_delivery_adapter") or ""):
            metadata["video_delivery_adapter"] = str(settings.get("_midom_ltx_delivery_adapter") or "")
            metadata["requested_resolution"] = str(settings.get("_midom_requested_resolution") or "")
            metadata["internal_render_resolution"] = str(settings.get("_midom_ltx_internal_resolution") or settings.get("resolution") or "")
        return metadata

    def create_ui(self, api_session):
        config = self._load_config()

        gr.Markdown(f"## {PLUGIN_NAME}")
        gr.Markdown("Pair this local WanGP worker only with Midom projects you control. HTTPS is required except explicit local development modes.")
        gr.Markdown(
            "Local Media Processing is reported separately from AI generation. "
            "When local FFmpeg/FFprobe are available, this worker can also opt in to deterministic Event Video Processing jobs."
        )
        with gr.Row():
            api_base_url = gr.Textbox(
                label="API Base URL",
                value=str(config.get("api_base_url") or ""),
                placeholder="https://your-app.example.com",
            )
            machine_name = gr.Textbox(
                label="Machine Name",
                value=str(config.get("machine_name") or ""),
                placeholder="Studio GPU Workstation",
            )
            pairing_code = gr.Textbox(label="Pairing Code", type="password")
        allow_insecure_local_dev = gr.Checkbox(
            label="Allow HTTP for localhost development only",
            value=bool(config.get("allow_insecure_local_dev", False)),
        )
        allow_insecure_lan_dev = gr.Checkbox(
            label="Allow HTTP for private LAN / tailnet development only",
            value=bool(config.get("allow_insecure_lan_dev", False)),
        )

        with gr.Row():
            pair_btn = gr.Button("Add Project Pairing", interactive=not self._worker_is_running())
            update_capabilities_btn = gr.Button(self._capabilities_button_label())
            refresh_log_btn = gr.Button("Refresh Log")
            worker_toggle_btn = gr.Button(
                self._worker_toggle_label(),
                interactive=self._worker_is_running() or self._is_paired(config),
            )
        connection_list = gr.Textbox(
            label="Paired Project Connections",
            interactive=False,
            value=self._connection_list_text(config),
            lines=6,
        )
        with gr.Row():
            connection_selector = gr.Dropdown(
                label="Project Pairing",
                choices=self._connection_choices(config),
                value=self._default_connection_choice(config),
                interactive=self._is_paired(config),
            )
            disconnect_selected_btn = gr.Button("Disconnect Selected Pairing", interactive=self._is_paired(config))
            disconnect_all_btn = gr.Button("Forget All Local Pairings", interactive=self._is_paired(config))

        event_keep_awake_can_enable = self._event_processing_can_keep_awake(config)
        event_keep_awake_active = self._event_processing_keep_awake_active(config)
        event_keep_awake_status = gr.Textbox(
            label="Event Video Processing Keep Awake",
            interactive=False,
            value=self._event_processing_keep_awake_text(config),
            lines=2,
        )
        with gr.Row():
            event_keep_awake_4h_btn = gr.Button("Keep Awake 4h", interactive=event_keep_awake_can_enable)
            event_keep_awake_8h_btn = gr.Button("Keep Awake 8h", interactive=event_keep_awake_can_enable)
            event_keep_awake_12h_btn = gr.Button("Keep Awake 12h", interactive=event_keep_awake_can_enable)
            event_keep_awake_disable_btn = gr.Button(
                "Disable Keep Awake",
                interactive=event_keep_awake_active,
            )

        status = gr.Textbox(label="Plugin Log", interactive=False, value=self._log_text(), lines=18)
        self._log("Plugin Log visible auto-refresh is disabled for Gradio compatibility; use Refresh Log to update the panel.")
        ui_outputs = [
            status,
            worker_toggle_btn,
            pair_btn,
            disconnect_selected_btn,
            disconnect_all_btn,
            update_capabilities_btn,
            connection_selector,
            connection_list,
            event_keep_awake_status,
            event_keep_awake_4h_btn,
            event_keep_awake_8h_btn,
            event_keep_awake_12h_btn,
            event_keep_awake_disable_btn,
        ]

        pair_btn.click(
            fn=lambda api_base_url_value, pairing_code_value, machine_name_value, allow_insecure_local_value, allow_insecure_lan_value: self._pair_update_capabilities_and_start_worker(
                api_session,
                api_base_url_value,
                pairing_code_value,
                machine_name_value,
                allow_insecure_local_value,
                allow_insecure_lan_value,
            ),
            inputs=[api_base_url, pairing_code, machine_name, allow_insecure_local_dev, allow_insecure_lan_dev],
            outputs=ui_outputs,
        )
        update_capabilities_btn.click(
            fn=self._wake_or_update_capabilities_ui,
            inputs=[],
            outputs=ui_outputs,
        )
        refresh_log_btn.click(
            fn=self._refresh_log_ui,
            inputs=[],
            outputs=ui_outputs,
        )

        worker_toggle_btn.click(
            fn=lambda: self._toggle_worker(api_session),
            inputs=[],
            outputs=ui_outputs,
        )
        disconnect_selected_btn.click(
            fn=self._disconnect_selected_pairing_ui,
            inputs=[connection_selector],
            outputs=ui_outputs,
        )
        disconnect_all_btn.click(
            fn=self._forget_local_pairing_ui,
            inputs=[],
            outputs=[*ui_outputs, pairing_code],
        )
        event_keep_awake_4h_btn.click(
            fn=lambda: self._event_keep_awake_ui("4h"),
            inputs=[],
            outputs=ui_outputs,
        )
        event_keep_awake_8h_btn.click(
            fn=lambda: self._event_keep_awake_ui("8h"),
            inputs=[],
            outputs=ui_outputs,
        )
        event_keep_awake_12h_btn.click(
            fn=lambda: self._event_keep_awake_ui("12h"),
            inputs=[],
            outputs=ui_outputs,
        )
        event_keep_awake_disable_btn.click(
            fn=self._event_keep_awake_disable_ui,
            inputs=[],
            outputs=ui_outputs,
        )

    def _ui_result(self, status_text: Optional[str] = None):
        config = self._load_config()
        choices = self._connection_choices(config)
        return (
            self._log_text(),
            gr.update(
                value=self._worker_toggle_label(),
                interactive=self._worker_is_running() or self._is_paired(config),
            ),
            gr.update(interactive=not self._worker_is_running()),
            gr.update(interactive=self._is_paired()),
            gr.update(interactive=self._is_paired()),
            gr.update(value=self._capabilities_button_label()),
            gr.update(choices=choices, value=self._default_connection_choice(config), interactive=bool(choices)),
            gr.update(value=self._connection_list_text(config)),
            gr.update(value=self._event_processing_keep_awake_text(config)),
            *self._event_processing_keep_awake_button_updates(config),
        )

    def _connection_label(self, record: dict[str, Any]) -> str:
        active_context = self._active_job_context
        connection_id = str(record.get("connection_id") or f"worker-{record.get('worker_id')}").strip()
        active_marker = ""
        if active_context is not None and str(active_context.connection_id) == connection_id:
            active_marker = f" | active job {self._current_active_job_id()}"
        return (
            f"{connection_id} | worker {record.get('worker_id')} | "
            f"org {record.get('org_id')} | project {record.get('project_id')} | "
            f"user {record.get('paired_user_id')}{active_marker}"
        )

    def _connection_choices(self, config: Optional[dict[str, Any]] = None) -> list[str]:
        return [self._connection_label(record) for record in self._connection_records(config)]

    def _default_connection_choice(self, config: Optional[dict[str, Any]] = None) -> Optional[str]:
        choices = self._connection_choices(config)
        return choices[0] if choices else None

    @staticmethod
    def _connection_id_from_choice(choice: Any) -> str:
        return str(choice or "").split(" | ", 1)[0].strip()

    def _connection_list_text(self, config: Optional[dict[str, Any]] = None) -> str:
        records = self._connection_records(config)
        if not records:
            return "No Midom project pairings are stored locally."
        return "\n".join(self._connection_label(record) for record in records)

    def _is_paired(self, config: Optional[dict[str, Any]] = None) -> bool:
        config = self._load_config() if config is None else config
        return bool(self._connection_records(config))

    def _worker_is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def _worker_toggle_label(self) -> str:
        return "Stop Worker" if self._worker_is_running() else "Start Worker"

    def _capabilities_button_label(self) -> str:
        return "Wake for Active Use" if self._worker_is_idle_standby() else "Update Capabilities"

    def _event_processing_keep_awake_until(self, config: Optional[dict[str, Any]] = None) -> int:
        config = self._load_config() if config is None else config
        return self._coerce_int(config.get("event_processing_keep_awake_until"), 0, 0, 4_102_444_800)

    def _event_processing_keep_awake_active(self, config: Optional[dict[str, Any]] = None, now: Optional[float] = None) -> bool:
        until = self._event_processing_keep_awake_until(config)
        now_epoch = int(time.time() if now is None else now)
        return until > now_epoch and self._event_processing_can_keep_awake(config)

    def _event_processing_keep_awake_text(self, config: Optional[dict[str, Any]] = None) -> str:
        until = self._event_processing_keep_awake_until(config)
        now = int(time.time())
        if until > now:
            remaining = until - now
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            expires = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until))
            return (
                "Event Video Processing keep-awake is active. "
                f"Remaining: {hours}h {minutes}m. Expires: {expires}."
            )
        return "Event Video Processing keep-awake is off."

    def _event_processing_can_keep_awake(self, config: Optional[dict[str, Any]] = None) -> bool:
        return bool(self._connection_records(config)) and bool(self._event_video_processing_capability())

    def _event_processing_keep_awake_button_updates(self, config: Optional[dict[str, Any]] = None) -> tuple[Any, Any, Any, Any]:
        can_enable = self._event_processing_can_keep_awake(config)
        active = self._event_processing_keep_awake_active(config)
        return (
            gr.update(interactive=can_enable),
            gr.update(interactive=can_enable),
            gr.update(interactive=can_enable),
            gr.update(interactive=active),
        )

    def _set_event_processing_keep_awake(self, duration_key: str) -> str:
        if duration_key not in EVENT_KEEP_AWAKE_DURATION_SECONDS:
            raise gr.Error(f"Unsupported keep-awake duration: {duration_key}")
        config = self._load_config()
        if not self._connection_records(config):
            raise gr.Error("Pair this worker before enabling Event Video Processing keep-awake.")
        if not self._event_video_processing_capability():
            raise gr.Error("Event Video Processing is not available because local FFmpeg/FFprobe support is incomplete.")
        until = int(time.time()) + int(EVENT_KEEP_AWAKE_DURATION_SECONDS[duration_key])
        config["event_processing_keep_awake_until"] = until
        self._save_config(config)
        self._reset_idle_backoff_for_active_use(f"event video processing keep-awake enabled for {duration_key}", wake=True)
        self._log(
            "Event Video Processing keep-awake enabled; "
            f"duration={duration_key} expires_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(until))}.",
            force=True,
        )
        return self._status_text(config=config)

    def _disable_event_processing_keep_awake(self) -> str:
        config = self._load_config()
        if config.get("event_processing_keep_awake_until"):
            config["event_processing_keep_awake_until"] = 0
            self._save_config(config)
            self._log("Event Video Processing keep-awake disabled.", force=True)
        else:
            self._log("Event Video Processing keep-awake was already off.")
        return self._status_text(config=config)

    def _event_keep_awake_ui(self, duration_key: str):
        try:
            status_text = self._set_event_processing_keep_awake(duration_key)
            return self._ui_result(status_text)
        except gr.Error as exc:
            self._log(f"Event Video Processing keep-awake was not enabled: {self._friendly_exception_message(exc)}", force=True)
            return self._ui_result()

    def _event_keep_awake_disable_ui(self):
        status_text = self._disable_event_processing_keep_awake()
        return self._ui_result(status_text)

    def _current_active_job_id(self) -> Any:
        active_job_id = self._active_job_id
        if self._active_job is not None and getattr(self._active_job, "done", True):
            return None
        return active_job_id

    def _idle_elapsed_seconds(self, now: Optional[float] = None) -> float:
        if self._current_active_job_id() is not None:
            self._idle_since_at = None
            return 0.0
        now = time.monotonic() if now is None else now
        if self._idle_since_at is None:
            self._idle_since_at = now
        return max(0.0, now - self._idle_since_at)

    def _idle_mode(self, now: Optional[float] = None) -> str:
        if self._current_active_job_id() is not None:
            self._idle_since_at = None
            return "busy"
        if self._event_processing_keep_awake_active(now=time.time()):
            if self._idle_since_at is None:
                self._idle_since_at = time.monotonic() if now is None else now
            return "active"
        idle_seconds = self._idle_elapsed_seconds(time.monotonic() if now is None else now)
        if idle_seconds >= IDLE_STANDBY_AFTER_SECONDS:
            return "standby"
        if idle_seconds >= IDLE_BACKOFF_AFTER_SECONDS:
            return "backoff"
        return "active"

    def _worker_is_idle_standby(self) -> bool:
        return self._worker_is_running() and self._idle_mode() == "standby"

    def _reset_idle_backoff_for_active_use(self, reason: str, *, wake: bool = False) -> None:
        self._idle_since_at = time.monotonic()
        self._last_heartbeat_at = 0.0
        self._last_candidate_poll_at = 0.0
        self._last_idle_mode = None
        if wake:
            self._wake_event.set()
        self._log(f"Worker active-use polling reset; reason={reason}.")

    def _wake_worker_for_active_use(self) -> str:
        self._reset_idle_backoff_for_active_use("manual wake for active use", wake=True)
        self._log("Worker woken for active use; normal heartbeat and candidate polling will resume.", force=True)
        return self._status_text()

    def _pair_update_capabilities_and_start_worker(
        self,
        api_session,
        api_base_url: str,
        pairing_code: str,
        machine_name: str,
        allow_insecure_local_dev: bool,
        allow_insecure_lan_dev: bool,
    ):
        try:
            self._pair_worker(api_base_url, pairing_code, machine_name, allow_insecure_local_dev, allow_insecure_lan_dev)
            status_text = self._start_worker(api_session)
            self._log("Pairing flow complete; capabilities were sent during pairing and worker start was requested.")
            return self._ui_result(status_text)
        except gr.Error as exc:
            message = self._friendly_exception_message(exc)
            self._log(f"Pairing was not completed: {message}", force=True)
            return self._ui_result()
        except requests.RequestException as exc:
            message = self._friendly_exception_message(exc)
            self._log(f"Pairing was not completed because the Midom API request failed: {message}", force=True)
            return self._ui_result()
        except Exception as exc:
            message = self._friendly_exception_message(exc)
            self._log(f"Pairing was not completed because the bridge hit a local error: {message}", force=True)
            return self._ui_result()

    def _wake_or_update_capabilities_ui(self):
        try:
            if self._worker_is_idle_standby():
                status_text = self._wake_worker_for_active_use()
                return self._ui_result(status_text)
            status_text = self._update_capabilities()
            return self._ui_result(status_text)
        except gr.Error as exc:
            message = self._friendly_exception_message(exc)
            self._log(f"Capabilities update was not completed: {message}", force=True)
            return self._ui_result()
        except requests.RequestException as exc:
            message = self._friendly_exception_message(exc)
            self._log(f"Capabilities update failed because the Midom API request failed: {message}", force=True)
            return self._ui_result()

    def _refresh_log_ui(self):
        return self._ui_result()

    def _refresh_log_text_ui(self):
        return self._log_text()

    def _toggle_worker(self, api_session):
        try:
            if self._worker_is_running():
                status_text = self._stop_worker(wait=True)
            else:
                status_text = self._start_worker(api_session)
            return self._ui_result(status_text)
        except gr.Error as exc:
            message = self._friendly_exception_message(exc)
            self._log(f"Worker start/stop request was not completed: {message}", force=True)
            return self._ui_result()
        except requests.RequestException as exc:
            message = self._friendly_exception_message(exc)
            self._log(f"Worker start/stop request failed because the Midom API request failed: {message}", force=True)
            return self._ui_result()

    def _forget_local_pairing_ui(self):
        status_text = self._forget_local_pairing()
        return (*self._ui_result(status_text), gr.update(value=""))

    def _disconnect_selected_pairing_ui(self, selected_connection: Any):
        status_text = self._disconnect_selected_pairing(selected_connection)
        return self._ui_result(status_text)

    def _status_text(self, config: Optional[dict[str, Any]] = None) -> str:
        config = self._load_config() if config is None else config
        connections = self._connection_records(config)
        running = self._worker_is_running()
        if connections:
            state = "long-idle standby" if self._worker_is_idle_standby() else ("running heartbeat" if running else "stopped")
            active_context = self._active_job_context
            parts = []
            for record in connections:
                scope = self._scope_text(record)
                expiry = str(record.get("token_expires_at") or "").strip()
                active_marker = ""
                if active_context is not None and str(record.get("connection_id")) == str(active_context.connection_id):
                    active_marker = f" Active job: {self._current_active_job_id()}."
                suffix = f" Token expires: {expiry}." if expiry else ""
                parts.append(f"worker {record.get('worker_id')}: {scope}{suffix}{active_marker}")
            keep_awake_text = ""
            if self._event_processing_keep_awake_active(config):
                keep_awake_text = " " + self._event_processing_keep_awake_text(config)
            return f"{len(connections)} project pairing(s) are {state}.{keep_awake_text} " + " ".join(parts)
        return "Worker is not paired."

    def _scope_text(self, config: dict[str, Any]) -> str:
        parts = []
        for label, key in (("org", "org_id"), ("project", "project_id"), ("user", "paired_user_id")):
            value = str(config.get(key) or "").strip()
            if value:
                parts.append(f"{label}: {value}")
        return "Scope: " + ", ".join(parts) + "." if parts else "Scope was not provided by Midom."

    def _start_worker(self, api_session) -> str:
        config = self._load_config()
        connections = self._connection_records(config)
        if not connections:
            raise gr.Error("Pair this worker before starting it.")
        for record in connections:
            self._connection_context(record)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return self._status_text()
        self._stop_event.clear()
        self._wake_event.clear()
        self._log(
            f"Starting worker loop for {len(connections)} project pairing(s). "
            "WanGP jobs will use a background-safe session, not the live Gradio request session."
        )
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=PLUGIN_ID,
            daemon=True,
        )
        self._worker_thread.start()
        return self._status_text()

    def _request_worker_stop(self, reason: str) -> None:
        self._log(f"Stop requested; reason={reason}.")
        self._stop_event.set()
        self._wake_event.set()
        job = self._active_job
        if job is not None and not getattr(job, "done", True):
            try:
                job.cancel()
                self._log(f"Cancelled active local WanGP job due to stop request; {self._job_label()}.")
            except Exception:
                pass

    def _wait_for_worker_stop(self, timeout_seconds: float = 8.0) -> None:
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_seconds)))
        if thread is not None and not thread.is_alive() and self._worker_thread is thread:
            self._worker_thread = None

    def _stop_worker(self, wait: bool = False) -> str:
        self._request_worker_stop("WanGP plugin UI")
        if wait:
            self._wait_for_worker_stop()
        return self._status_text()

    def _forget_local_pairing(self) -> str:
        return self._disconnect_worker()

    def _disconnect_selected_pairing(self, selected_connection: Any) -> str:
        connection_id = self._connection_id_from_choice(selected_connection)
        if not connection_id:
            self._log("No project pairing was selected for disconnect.", force=True)
            return self._status_text()
        if self._worker_is_running():
            self._request_worker_stop(f"disconnect project pairing {connection_id}")
            self._wait_for_worker_stop()
        if self._worker_is_running():
            raise gr.Error("Worker is still stopping. Wait a few seconds and click Disconnect Selected Pairing again.")
        config = self._load_config()
        connections = self._connection_records(config)
        selected_record = None
        remaining = []
        for record in connections:
            if str(record.get("connection_id") or "") == connection_id:
                selected_record = record
            else:
                remaining.append(record)
        if selected_record is None:
            self._log(f"Selected project pairing was not found locally; connection_id={connection_id}.", force=True)
            return self._status_text(config=config)
        try:
            connection = self._connection_context(selected_record)
            self._log(
                "Disconnecting selected worker remotely before forgetting local pairing; "
                f"connection_id={connection.connection_id} worker_id={connection.worker_id}."
            )
            requests.post(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/disconnect",
                headers={**self._headers(connection), "Content-Type": "application/json"},
                json={"reason": "plugin_user_disconnect", "active_job_policy": "cancel"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            ).raise_for_status()
            self._log(f"Remote disconnect accepted; connection_id={connection.connection_id} worker_id={connection.worker_id}.")
        except Exception as exc:
            self._log(f"Remote worker disconnect failed; forgetting selected local pairing anyway: {exc}", force=True)
        config = self._save_connections(remaining, config)
        self._log(f"Selected local project pairing forgotten; connection_id={connection_id} remaining_pairings={len(remaining)}.")
        return self._status_text(config=config)

    def _disconnect_worker(self) -> str:
        if self._worker_is_running():
            self._request_worker_stop("forget all local pairings")
            self._wait_for_worker_stop()
        if self._worker_is_running():
            raise gr.Error("Worker is still stopping. Wait a few seconds and click Forget All Local Pairings again.")
        config = self._load_config()
        connections = self._connection_records(config)
        for record in connections:
            try:
                connection = self._connection_context(record)
                self._log(
                    "Disconnecting worker remotely before forgetting local pairing; "
                    f"connection_id={connection.connection_id} worker_id={connection.worker_id}."
                )
                requests.post(
                    f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/disconnect",
                    headers={**self._headers(connection), "Content-Type": "application/json"},
                    json={"reason": "plugin_user_disconnect", "active_job_policy": "cancel"},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                ).raise_for_status()
                self._log(f"Remote disconnect accepted; connection_id={connection.connection_id} worker_id={connection.worker_id}.")
            except Exception as exc:
                self._log(f"Remote worker disconnect failed; forgetting local pairing anyway: {exc}", force=True)
        try:
            self._config_path.unlink()
        except FileNotFoundError:
            pass
        self._log("All local pairings forgotten; new Midom pairing codes are required before this worker can reconnect.")
        return self._status_text(config={})

    def _worker_loop(self) -> None:
        self._log("Worker loop entered.")
        self._idle_since_at = time.monotonic()
        self._last_heartbeat_at = 0.0
        self._last_candidate_poll_at = 0.0
        self._last_idle_mode = None
        while not self._stop_event.is_set():
            try:
                self._worker_loop_iteration()
            except Exception as exc:
                self._log(f"Worker loop iteration failed: {exc}", force=True)
            wait_seconds = self._worker_loop_wait_seconds()
            if self._wake_event.wait(wait_seconds):
                self._wake_event.clear()
        self._log("Worker loop exited.")

    def _worker_loop_iteration(self) -> None:
        now = time.monotonic()
        idle_mode = self._idle_mode(now)
        if idle_mode != self._last_idle_mode:
            self._last_idle_mode = idle_mode
            if idle_mode == "backoff":
                self._log(
                    f"Worker has been idle for {int(IDLE_BACKOFF_AFTER_SECONDS / 60)} minutes; "
                    f"heartbeat and candidate polling are backing off to every {IDLE_BACKOFF_INTERVAL_SECONDS} seconds.",
                    force=True,
                )
            elif idle_mode == "standby":
                self._log(
                    "Worker entered long-idle standby; reporting paused/not accepting and pausing candidate polling. "
                    "Use Wake for Active Use to resume normal polling.",
                    force=True,
                )
            elif idle_mode == "active":
                if self._event_processing_keep_awake_active():
                    self._log("Worker is using Event Video Processing keep-awake heartbeat and candidate polling.")
                else:
                    self._log("Worker is using normal idle heartbeat and candidate polling.")
        heartbeat_interval = self._heartbeat_interval_seconds(idle_mode)
        if now - self._last_heartbeat_at >= heartbeat_interval:
            self._heartbeat_once()
            self._last_heartbeat_at = time.monotonic()
        if self._stop_event.is_set():
            return
        if idle_mode != "standby" and self._current_active_job_id() is None:
            local_busy_reason = self._local_wangp_busy_reason()
            if local_busy_reason:
                if now - self._last_local_wangp_busy_log_at >= IDLE_POLL_LOG_SECONDS:
                    self._last_local_wangp_busy_log_at = now
                    self._log(f"Candidate polling deferred; {local_busy_reason}")
                self._last_candidate_poll_at = time.monotonic()
                return
            poll_interval = self._candidate_poll_interval_seconds(idle_mode)
            if now - self._last_candidate_poll_at >= poll_interval:
                ran_job = self._poll_once()
                self._last_candidate_poll_at = 0.0 if ran_job else time.monotonic()

    def _worker_loop_wait_seconds(self) -> float:
        idle_mode = self._idle_mode()
        return float(min(self._heartbeat_interval_seconds(idle_mode), self._candidate_poll_interval_seconds(idle_mode)))

    @staticmethod
    def _heartbeat_interval_seconds(idle_mode: str) -> int:
        if idle_mode == "standby":
            return IDLE_STANDBY_HEARTBEAT_SECONDS
        if idle_mode == "backoff":
            return IDLE_BACKOFF_INTERVAL_SECONDS
        return POLL_INTERVAL_SECONDS

    @staticmethod
    def _candidate_poll_interval_seconds(idle_mode: str) -> int:
        if idle_mode == "standby":
            return IDLE_STANDBY_HEARTBEAT_SECONDS
        if idle_mode == "backoff":
            return IDLE_BACKOFF_INTERVAL_SECONDS
        return POLL_INTERVAL_SECONDS

    def _heartbeat_once(self, connection: Optional[ConnectionContext] = None) -> None:
        if connection is None:
            contexts = self._connection_contexts()
            if not contexts:
                raise gr.Error("Pair this worker before sending heartbeats.")
            for context in contexts:
                try:
                    self._heartbeat_once(context)
                except Exception as exc:
                    self._log(
                        "Heartbeat failed for project pairing; "
                        f"connection_id={context.connection_id} worker_id={context.worker_id} error={exc}",
                        force=True,
                    )
            return
        active_job_id = self._current_active_job_id()
        idle_mode = self._idle_mode()
        active_context = self._active_job_context
        local_busy_reason = self._local_wangp_busy_reason() if active_job_id is None else None
        is_active_connection = (
            active_job_id is not None
            and active_context is not None
            and str(active_context.connection_id) == str(connection.connection_id)
        )
        is_sibling_busy = active_job_id is not None and not is_active_connection
        if local_busy_reason:
            accepting = bool(JOB_FLOW_ENABLED and not self._stop_event.is_set() and idle_mode != "standby")
            status = "busy"
            heartbeat_active_job_id = None
            message = LOCAL_WANGP_BUSY_MESSAGE
        elif is_sibling_busy:
            accepting = bool(JOB_FLOW_ENABLED and not self._stop_event.is_set())
            status = "busy"
            heartbeat_active_job_id = None
            message = SIBLING_BUSY_MESSAGE
        else:
            accepting = bool(JOB_FLOW_ENABLED and not self._stop_event.is_set() and idle_mode != "standby")
            status = "busy" if active_job_id is not None else ("paused" if idle_mode == "standby" else "idle")
            heartbeat_active_job_id = active_job_id
            message = self._heartbeat_message(status, accepting, active_job_id, idle_mode)
        response = requests.post(
            f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/heartbeat",
            headers={**self._headers(connection), "Content-Type": "application/json"},
            json={
                "status": status,
                "accepting": accepting,
                "active_job_id": heartbeat_active_job_id,
                "message": message,
                "capabilities_revision": connection.capabilities_revision,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code in {401, 403} and self._forget_unauthorized_connection(
            connection,
            "Heartbeat",
            self._midom_error_message(response),
        ):
            return
        response.raise_for_status()
        payload = response.json()
        now = time.monotonic()
        if now - self._last_heartbeat_log_at >= HEARTBEAT_LOG_SECONDS or active_job_id is not None:
            self._last_heartbeat_log_at = now
            self._log(
                f"Heartbeat accepted; connection_id={connection.connection_id} worker_id={connection.worker_id} status={status} "
                f"accepting={accepting} active_job_id={heartbeat_active_job_id} message={message!r}."
            )
        if isinstance(payload, dict) and payload.get("revoke_requested"):
            if is_active_connection:
                self._stop_event.set()
                self._log(
                    "Midom requested active worker revocation; stopping worker and cancelling local job.",
                    force=True,
                )
            else:
                self._forget_connection_locally(connection.connection_id)
                self._log(
                    "Midom requested worker revocation; removed only that project pairing locally. "
                    f"connection_id={connection.connection_id} worker_id={connection.worker_id}.",
                    force=True,
                )

    def _heartbeat_message(self, status: str, accepting: bool, active_job_id: Any, idle_mode: str = "active") -> str:
        if idle_mode == "standby" and status != "busy":
            return "Long-idle standby. Wake this worker locally before queueing active use."
        if idle_mode == "backoff" and accepting and status != "busy":
            return "Idle and accepting Midom media jobs with long-idle polling backoff."
        if accepting and status != "busy" and self._event_processing_keep_awake_active():
            return "Idle and accepting Event Video Processing jobs; keep-awake is active for a live event window."
        if accepting and status != "busy":
            return "Idle and accepting Midom media jobs."
        if accepting and status == "busy":
            active_text = self._active_job_message(active_job_id)
            return f"{active_text}. Accepting additional queued Midom media jobs."[:500]
        if status != "busy" or active_job_id is None:
            return "Worker is not accepting new jobs."
        return self._active_job_message(active_job_id)

    def _local_wangp_busy_reason(self) -> str:
        state_component = getattr(self, "state", None)
        state = getattr(state_component, "value", None)
        if not isinstance(state, dict):
            return ""
        gen = state.get("gen")
        if not isinstance(gen, dict):
            return ""
        inline_queue = gen.get("inline_queue")
        if inline_queue is not None:
            return "WanGP has a pending inline UI or Deepy queue request."
        queue_value = gen.get("queue")
        if isinstance(queue_value, list) and queue_value:
            return f"WanGP UI queue has {len(queue_value)} pending or running task(s)."
        if bool(gen.get("in_progress")) or bool(gen.get("main_process_running")):
            return "WanGP main generation is in progress."
        process_status = str(gen.get("process_status") or "").strip()
        if process_status and process_status not in {"process:main"}:
            return f"WanGP GPU process is busy ({process_status})."
        return ""

    def _active_job_message(self, active_job_id: Any) -> str:
        snapshot = dict(self._active_job_status or {})
        phase = str(snapshot.get("phase") or "").strip()
        text = str(snapshot.get("status") or "").strip()
        progress = snapshot.get("progress")
        step = snapshot.get("current_step")
        total_steps = snapshot.get("total_steps")
        parts = [f"Working on Midom job {active_job_id}"]
        if phase:
            parts.append(phase)
        if text and text != phase:
            parts.append(text)
        if progress is not None:
            try:
                parts.append(f"{int(progress)}%")
            except (TypeError, ValueError):
                pass
        if step is not None and total_steps is not None:
            parts.append(f"step {step}/{total_steps}")
        return " - ".join(parts)[:500]

    def _ensure_job_flow_enabled(self) -> None:
        if not JOB_FLOW_ENABLED:
            raise RuntimeError("Job execution flow is disabled in plugin configuration.")

    def _get_generation_session(self):
        if self._generation_session is None:
            self._log("Initializing background WanGP generation session.")
            self._generation_session = init_wangp_session(
                console_output=True,
                console_isatty=True,
            )
            self._log("Background WanGP generation session initialized.")
        return self._generation_session

    def _poll_once(self) -> bool:
        self._ensure_job_flow_enabled()
        if self._active_job is not None and not getattr(self._active_job, "done", True):
            return False
        connections = self._connection_contexts()
        if not connections:
            return False
        start_index = self._next_connection_poll_index % len(connections)
        ordered_connections = connections[start_index:] + connections[:start_index]
        for offset, connection in enumerate(ordered_connections):
            next_index = (start_index + offset + 1) % len(connections)
            try:
                if self._poll_connection_once(connection):
                    self._next_connection_poll_index = next_index
                    return True
            except Exception as exc:
                self._log(
                    "Candidate poll failed for project pairing; "
                    f"connection_id={connection.connection_id} worker_id={connection.worker_id} error={exc}",
                    force=True,
                )
            self._next_connection_poll_index = next_index
        return False

    def _poll_connection_once(self, connection: ConnectionContext) -> bool:
        response = requests.get(
            f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/candidates",
            headers=self._headers(connection),
            params={"limit": 5},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code in {401, 403} and self._forget_unauthorized_connection(
            connection,
            "Candidate poll",
            self._midom_error_message(response),
        ):
            return False
        response.raise_for_status()
        payload = response.json()
        jobs = payload.get("jobs") if isinstance(payload, dict) else []
        if not isinstance(jobs, list):
            self._log("Candidate poll returned a non-list jobs payload; ignoring response.")
            return False
        if jobs:
            self._log(
                f"Candidate poll returned {len(jobs)} candidate(s); "
                f"connection_id={connection.connection_id} worker_id={connection.worker_id} "
                f"order={self._candidate_order_log(jobs)}."
            )
        else:
            now = time.monotonic()
            if now - self._last_idle_poll_log_at >= IDLE_POLL_LOG_SECONDS:
                self._last_idle_poll_log_at = now
                self._log(f"Candidate poll returned no jobs; connection_id={connection.connection_id} worker_id={connection.worker_id}.")
        for candidate in jobs:
            incompatibility = self._candidate_incompatibility_reason(candidate, connection)
            if incompatibility:
                self._log(
                    "Skipping incompatible candidate; "
                    f"connection_id={connection.connection_id} worker_id={connection.worker_id} "
                    f"job_id={candidate.get('job_id') if isinstance(candidate, dict) else 'unknown'} "
                    f"model_id={candidate.get('model_id') if isinstance(candidate, dict) else 'unknown'} "
                    f"reason={incompatibility}."
                )
                continue
            candidate_media_type = str(candidate.get("media_type") or "").lower()
            if candidate_media_type == "audio":
                summary = candidate.get("summary") or {}
                self._log(
                    "Audio candidate is compatible; "
                    f"connection_id={connection.connection_id} worker_id={connection.worker_id} "
                    f"job_id={candidate.get('job_id')} model_id={candidate.get('model_id')} "
                    f"output_format={(summary.get('output_format') if isinstance(summary, dict) else None)!r} "
                    f"audio_task={(summary.get('audio_task') if isinstance(summary, dict) else None)!r} "
                    f"voice_mode={(summary.get('voice_mode') if isinstance(summary, dict) else None)!r} "
                    f"dialogue_mode={(summary.get('dialogue_mode') if isinstance(summary, dict) else None)!r} "
                    f"reference_audio_count={(summary.get('reference_audio_count') if isinstance(summary, dict) else None)!r} "
                    f"emotion_reference_audio_count={(summary.get('emotion_reference_audio_count') if isinstance(summary, dict) else None)!r} "
                    f"speaker_reference_audio_count={(summary.get('speaker_reference_audio_count') if isinstance(summary, dict) else None)!r} "
                    f"emotion_mode={(summary.get('emotion_mode') if isinstance(summary, dict) else None)!r}."
                )
            elif candidate_media_type == "video":
                summary = candidate.get("summary") or {}
                self._log(
                    "Video candidate is compatible; "
                    f"connection_id={connection.connection_id} worker_id={connection.worker_id} "
                    f"job_id={candidate.get('job_id')} model_id={candidate.get('model_id')} "
                    f"video_task={(summary.get('video_task') if isinstance(summary, dict) else None)!r} "
                    f"audio_video_mode={(summary.get('audio_video_mode') if isinstance(summary, dict) else None)!r} "
                    f"output_format={(summary.get('output_format') if isinstance(summary, dict) else None)!r} "
                    f"max_duration_seconds={(summary.get('max_duration_seconds') if isinstance(summary, dict) else None)!r} "
                    f"duration_mode={(summary.get('duration_mode') if isinstance(summary, dict) else None)!r} "
                    f"duration_seconds={(summary.get('duration_seconds') if isinstance(summary, dict) else None)!r} "
                    f"prompt_mode={(summary.get('prompt_mode') if isinstance(summary, dict) else None)!r} "
                    f"speed_profile_id={(summary.get('speed_profile_id') if isinstance(summary, dict) else None)!r} "
                    f"video_sync_profile_id={(summary.get('video_sync_profile_id') if isinstance(summary, dict) else None)!r} "
                    f"start_image_count={(summary.get('start_image_count') if isinstance(summary, dict) else None)!r} "
                    f"reference_image_count={(summary.get('reference_image_count') if isinstance(summary, dict) else None)!r} "
                    f"input_image_count={(summary.get('input_image_count') if isinstance(summary, dict) else None)!r} "
                    f"input_audio_count={(summary.get('input_audio_count') if isinstance(summary, dict) else None)!r}."
                )
            elif candidate_media_type == "image":
                summary = candidate.get("summary") or {}
                self._log(
                    "Image candidate is compatible; "
                    f"connection_id={connection.connection_id} worker_id={connection.worker_id} "
                    f"job_id={candidate.get('job_id')} model_id={candidate.get('model_id')} "
                    f"output_count={(summary.get('output_count') if isinstance(summary, dict) else None)!r} "
                    f"output_format={(summary.get('output_format') if isinstance(summary, dict) else None)!r} "
                    f"reference_image_count={(summary.get('reference_image_count') if isinstance(summary, dict) else None)!r} "
                    f"control_image_count={(summary.get('control_image_count') if isinstance(summary, dict) else None)!r}."
                )
            candidate_job_id = int(candidate["job_id"])
            self._log(f"Attempting explicit claim; connection_id={connection.connection_id} worker_id={connection.worker_id} job_id={candidate_job_id}.")
            claimed = self._claim_job(connection, candidate_job_id)
            if claimed is None:
                self._log(f"Claim conflict or unavailable candidate; job_id={candidate_job_id}.")
                continue
            self._reset_idle_backoff_for_active_use(f"claimed Midom job {candidate_job_id}")
            self._run_job(connection, claimed)
            return True
        return False

    @staticmethod
    def _candidate_order_log(jobs: list[Any]) -> str:
        parts = []
        for index, candidate in enumerate(jobs[:10]):
            if not isinstance(candidate, dict):
                parts.append(f"{index}:<non-object>")
                continue
            summary = candidate.get("summary") if isinstance(candidate.get("summary"), dict) else {}
            media_type = str(candidate.get("media_type") or "").strip() or "?"
            model_id = str(candidate.get("model_id") or candidate.get("processor_id") or "").strip() or "?"
            job_id = candidate.get("job_id")
            created_at = candidate.get("created_at") or candidate.get("queued_at") or summary.get("created_at") or summary.get("queued_at")
            priority = candidate.get("priority") or summary.get("priority")
            extra = []
            if created_at:
                extra.append(f"queued_at={created_at}")
            if priority is not None:
                extra.append(f"priority={priority}")
            suffix = f" {' '.join(extra)}" if extra else ""
            parts.append(f"{index}:job_id={job_id} media={media_type} model={model_id}{suffix}")
        if len(jobs) > 10:
            parts.append(f"...+{len(jobs) - 10} more")
        return "[" + "; ".join(parts) + "]"

    def _candidate_is_compatible(self, candidate: Any, config: Any) -> bool:
        return self._candidate_incompatibility_reason(candidate, config) is None

    def _candidate_incompatibility_reason(self, candidate: Any, config: Any) -> Optional[str]:
        if not isinstance(candidate, dict):
            return "candidate is not an object"
        try:
            job_id = int(candidate.get("job_id"))
            worker_id = int(candidate.get("worker_id"))
            org_id = int(candidate.get("org_id"))
            project_id = int(candidate.get("project_id"))
            requested_by_user_id = int(candidate.get("requested_by_user_id"))
        except (TypeError, ValueError):
            return "candidate has invalid ids"
        if job_id <= 0:
            return "job_id is missing or invalid"
        if worker_id != self._worker_id(config):
            return f"worker_id mismatch: {worker_id}"
        if org_id != int(self._connection_scope_value(config, "org_id") or 0):
            return f"org_id mismatch: {org_id}"
        if project_id != int(self._connection_scope_value(config, "project_id") or 0):
            return f"project_id mismatch: {project_id}"
        if requested_by_user_id != int(self._connection_scope_value(config, "paired_user_id") or 0):
            return f"requested_by_user_id mismatch: {requested_by_user_id}"
        media_type = str(candidate.get("media_type") or "").lower()
        model_id = str(candidate.get("model_id") or "").strip()
        if self._candidate_is_event_video_processing(candidate):
            return self._event_video_candidate_incompatibility_reason(candidate)
        if self._candidate_is_storyboard_ffmpeg_processing(candidate):
            return self._storyboard_ffmpeg_candidate_incompatibility_reason(candidate)
        if media_type not in {"image", "audio", "video"}:
            return f"unsupported media_type: {candidate.get('media_type')}"
        if media_type == "image" and model_id not in ALLOWED_MODEL_TYPES:
            return f"unsupported model_id: {candidate.get('model_id')}"
        summary = candidate.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        if media_type == "image":
            tool_id = str(candidate.get("tool_id") or summary.get("tool_id") or "").strip()
            if tool_id:
                if tool_id != QWEN_MULTI_ANGLE_TOOL_ID:
                    return f"unsupported image tool_id: {tool_id}"
                if model_id != QWEN_MULTI_ANGLE_BASE_MODEL_ID:
                    return f"{tool_id} requires model_id {QWEN_MULTI_ANGLE_BASE_MODEL_ID}"
                availability = self._qwen_multi_angle_tool_availability()
                if not availability["available"]:
                    return f"{tool_id} unavailable: {availability['unavailable_reason']}"
                accelerator_profile_id = str(
                    summary.get("accelerator_profile_id") or summary.get("speed_profile_id") or "standard"
                ).strip() or "standard"
                if accelerator_profile_id not in QWEN_MULTI_ANGLE_ALLOWED_ACCELERATOR_PROFILE_IDS:
                    return f"{tool_id} unsupported accelerator_profile_id: {accelerator_profile_id}"
                if accelerator_profile_id != "standard":
                    profile = ACCELERATOR_PROFILE_BY_ID.get(accelerator_profile_id)
                    if not profile:
                        return f"{tool_id} unknown accelerator_profile_id: {accelerator_profile_id}"
                    resolved_loras, missing_loras = self._resolve_accelerator_loras(profile)
                    if missing_loras or not resolved_loras:
                        return f"{tool_id} accelerator unavailable: {accelerator_profile_id}"
                view_change_strength = str(
                    summary.get("view_change_strength") or QWEN_MULTI_ANGLE_DEFAULT_VIEW_CHANGE_STRENGTH
                ).strip().lower()
                if view_change_strength not in QWEN_MULTI_ANGLE_VIEW_CHANGE_STRENGTHS:
                    return f"{tool_id} unsupported view_change_strength: {view_change_strength}"
        if media_type == "audio" and model_id not in ALLOWED_AUDIO_MODEL_TYPES:
            return f"unsupported audio model_id: {candidate.get('model_id')}"
        if media_type == "video" and model_id not in ALLOWED_VIDEO_MODEL_TYPES:
            return f"unsupported video model_id: {candidate.get('model_id')}"
        if isinstance(summary, dict):
            try:
                output_count = int(summary.get("output_count") or 1)
            except (TypeError, ValueError):
                return f"invalid output_count: {summary.get('output_count')}"
            max_outputs = _image_max_outputs_for_model(model_id) if media_type == "image" else 1
            if output_count < 1 or output_count > max_outputs:
                return f"unsupported output_count: {output_count}"
            if media_type == "audio":
                output_format = str(summary.get("output_format") or "mp3").strip().lower()
                if output_format not in {"mp3", "wav"}:
                    return f"unsupported audio output_format: {output_format}"
                if output_format == "mp3" and "audio/mpeg" not in self._audio_output_mime_types():
                    return "mp3 output requested but MP3 transcoding is unavailable"
                if model_id == SEEDVC_MODEL_ID:
                    if not self._seedvc_speech_available():
                        return "SeedVC speech voice replacement is not enabled"
                    audio_task = str(summary.get("audio_task") or "voice_conversion").strip().lower()
                    if audio_task != "voice_conversion":
                        return f"unsupported SeedVC audio_task: {audio_task}"
                    audio_category = str(summary.get("audio_category") or "voice_replacement").strip().lower()
                    if audio_category != "voice_replacement":
                        return f"unsupported SeedVC audio_category: {audio_category}"
                    voice_mode = str(summary.get("voice_mode") or "one_speaker_voice_replacement").strip().lower()
                    if voice_mode != "one_speaker_voice_replacement":
                        return f"unsupported SeedVC voice_mode: {voice_mode}"
                    source_audio_count = self._coerce_int(summary.get("source_audio_count"), 1, 0, 10)
                    reference_count = self._coerce_int(summary.get("reference_audio_count"), 1, 0, 10)
                    emotion_reference_count = self._coerce_int(summary.get("emotion_reference_audio_count"), 0, 0, 10)
                    speaker_reference_count = self._coerce_int(summary.get("speaker_reference_audio_count"), 0, 0, 10)
                    if source_audio_count != 1:
                        return f"unsupported source_audio_count: {source_audio_count}"
                    if reference_count != 1:
                        return f"unsupported reference_audio_count: {reference_count}"
                    if emotion_reference_count or speaker_reference_count:
                        return "SeedVC jobs do not support emotion or speaker reference inputs"
                    return None
                if model_id in STABLE_AUDIO3_MODEL_IDS or model_id == ACE_STEP15_MUSIC_MODEL_ID:
                    audio_task = str(summary.get("audio_task") or "generative_audio").strip().lower()
                    if audio_task != "generative_audio":
                        return f"unsupported generative audio_task: {audio_task}"
                    audio_category = str(summary.get("audio_category") or "music_and_sounds").strip().lower()
                    if audio_category != "music_and_sounds":
                        return f"unsupported generative audio_category: {audio_category}"
                    expected_subtype = "music" if model_id == ACE_STEP15_MUSIC_MODEL_ID else self._stable_audio3_subtype(model_id)
                    audio_subtype = str(summary.get("audio_subtype") or expected_subtype).strip().lower()
                    if audio_subtype != expected_subtype:
                        return f"unsupported generative audio_subtype: {audio_subtype}"
                    reference_count = self._coerce_int(summary.get("reference_audio_count"), 0, 0, 10)
                    emotion_reference_count = self._coerce_int(summary.get("emotion_reference_audio_count"), 0, 0, 10)
                    speaker_reference_count = self._coerce_int(summary.get("speaker_reference_audio_count"), 0, 0, 10)
                    if reference_count or emotion_reference_count or speaker_reference_count:
                        return "generative audio jobs do not support audio inputs"
                    return None
                default_audio_task = "dialogue_voice_clone_tts" if model_id == DRAMABOX_MODEL_ID else "voice_clone_tts"
                audio_task = str(summary.get("audio_task") or default_audio_task).strip().lower()
                if model_id == DRAMABOX_MODEL_ID:
                    if audio_task != "dialogue_voice_clone_tts":
                        return f"unsupported DramaBox audio_task: {audio_task}"
                elif audio_task != "voice_clone_tts":
                    return f"unsupported audio_task: {audio_task}"
                default_voice_mode = "two_voice_clone_n_voice_dialogue" if model_id == DRAMABOX_MODEL_ID else "single_reference"
                voice_mode = str(summary.get("voice_mode") or default_voice_mode).strip().lower()
                if model_id == DRAMABOX_MODEL_ID:
                    if voice_mode != "two_voice_clone_n_voice_dialogue":
                        return f"unsupported DramaBox voice_mode: {voice_mode}"
                elif voice_mode not in {"single_reference", "two_speaker_dialogue"}:
                    return f"unsupported voice_mode: {voice_mode}"
                if model_id == CHATTERBOX_MODEL_ID and voice_mode != "single_reference":
                    return f"unsupported Chatterbox voice_mode: {voice_mode}"
                if voice_mode in {"two_speaker_dialogue", "two_voice_clone_n_voice_dialogue"}:
                    expected_dialogue_mode = "dramabox_speaker_blocks" if model_id == DRAMABOX_MODEL_ID else "speaker_tags"
                    dialogue_mode = str(summary.get("dialogue_mode") or expected_dialogue_mode).strip().lower()
                    if dialogue_mode != expected_dialogue_mode:
                        return f"unsupported dialogue_mode: {dialogue_mode}"
                    speaker_reference_count = self._coerce_int(summary.get("speaker_reference_audio_count"), 2, 0, 10)
                    if speaker_reference_count != 2:
                        return f"unsupported speaker_reference_audio_count: {speaker_reference_count}"
                else:
                    reference_count = self._coerce_int(summary.get("reference_audio_count"), 1, 0, 10)
                    if reference_count != 1:
                        return f"unsupported reference_audio_count: {reference_count}"
                    emotion_reference_count = self._coerce_int(summary.get("emotion_reference_audio_count"), 0, 0, 10)
                    if model_id == CHATTERBOX_MODEL_ID and emotion_reference_count:
                        return f"unsupported Chatterbox emotion_reference_audio_count: {emotion_reference_count}"
                    if emotion_reference_count > 1:
                        return f"unsupported emotion_reference_audio_count: {emotion_reference_count}"
                    speaker_reference_count = self._coerce_int(summary.get("speaker_reference_audio_count"), 0, 0, 10)
                    if model_id == CHATTERBOX_MODEL_ID and speaker_reference_count:
                        return f"unsupported Chatterbox speaker_reference_audio_count: {speaker_reference_count}"
            elif media_type == "video":
                output_format = str(summary.get("output_format") or "mp4").strip().lower()
                if output_format != "mp4":
                    return f"unsupported video output_format: {output_format}"
                if model_id == "longcat_avatar_v1_5":
                    video_task = str(summary.get("video_task") or "talking_avatar").strip().lower()
                    if video_task != "talking_avatar":
                        return f"unsupported video_task: {video_task}"
                    duration_mode = str(summary.get("duration_mode") or LONGCAT_DURATION_MODE).strip().lower()
                    if duration_mode != LONGCAT_DURATION_MODE:
                        return f"unsupported duration_mode: {duration_mode}"
                elif model_id in LTX_VIDEO_MODEL_IDS:
                    video_task = str(summary.get("video_task") or "audio_conditioned_video").strip().lower()
                    if video_task not in {"audio_conditioned_video", "control_video_guided_video"}:
                        return f"unsupported video_task: {video_task}"
                    if video_task == "control_video_guided_video":
                        audio_video_mode = str(summary.get("audio_video_mode") or "control_video_audio_guided").strip().lower()
                        if audio_video_mode != "control_video_audio_guided":
                            return f"unsupported audio_video_mode: {audio_video_mode}"
                        duration_mode = str(summary.get("duration_mode") or LTX_CONTROL_VIDEO_DURATION_MODE).strip().lower()
                        if duration_mode != LTX_CONTROL_VIDEO_DURATION_MODE:
                            return f"unsupported duration_mode: {duration_mode}"
                        control_video_mode = str(summary.get("control_video_mode") or "").strip().lower()
                        if control_video_mode not in LTX_CONTROL_VIDEO_MODES:
                            return f"unsupported control_video_mode: {control_video_mode}"
                    else:
                        audio_video_mode = str(summary.get("audio_video_mode") or "driving_audio_guided").strip().lower()
                        if audio_video_mode != "driving_audio_guided":
                            return f"unsupported audio_video_mode: {audio_video_mode}"
                        duration_mode = str(summary.get("duration_mode") or LTX_DURATION_MODE).strip().lower()
                        if duration_mode != LTX_DURATION_MODE:
                            return f"unsupported duration_mode: {duration_mode}"
                    video_sync_profile_id = str(summary.get("video_sync_profile_id") or "standard").strip() or "standard"
                    if video_sync_profile_id not in {"standard", LTX_VIDEO_SYNC_OMNINFT_PROFILE_ID}:
                        return f"unsupported video_sync_profile_id: {video_sync_profile_id}"
                elif model_id == SVI_VIDEO_MODEL_ID:
                    video_task = str(summary.get("video_task") or "cinematic_i2v").strip().lower()
                    if video_task != "cinematic_i2v":
                        return f"unsupported video_task: {video_task}"
                    prompt_mode = str(summary.get("prompt_mode") or "plain").strip().lower()
                    if prompt_mode not in {"plain", "timed_cinematic_seconds"}:
                        return f"unsupported prompt_mode: {prompt_mode}"
                    duration_mode = str(summary.get("duration_mode") or SVI_DURATION_MODE).strip().lower()
                    if duration_mode != SVI_DURATION_MODE:
                        return f"unsupported duration_mode: {duration_mode}"
                    speed_profile_id = str(summary.get("speed_profile_id") or SVI_SPEED_PROFILE_ID).strip() or SVI_SPEED_PROFILE_ID
                    if speed_profile_id != SVI_SPEED_PROFILE_ID:
                        return f"unsupported speed_profile_id: {speed_profile_id}"
                if model_id in LTX_VIDEO_MODEL_IDS:
                    video_task = str(summary.get("video_task") or "audio_conditioned_video").strip().lower()
                    is_control_video_job = video_task == "control_video_guided_video"
                    input_audio_count = self._coerce_int(summary.get("input_audio_count"), 0 if is_control_video_job else 1, 0, 10)
                    # LTX candidate summaries may include aggregate image counts that are not as precise as
                    # the claimed input descriptors. Trust explicit start_image_count when present, then let
                    # the post-claim validator enforce the exact start_image/driving_audio contract.
                    if "start_image_count" in summary:
                        start_image_count = self._coerce_int(summary.get("start_image_count"), 1, 0, 10)
                        if start_image_count != 1:
                            return f"unsupported start_image_count: {start_image_count}"
                    end_image_count = self._coerce_int(summary.get("end_image_count"), 0, 0, 10)
                    if is_control_video_job and end_image_count:
                        return f"unsupported end_image_count: {end_image_count}"
                    if end_image_count > 1:
                        return f"unsupported end_image_count: {end_image_count}"
                    if is_control_video_job:
                        control_video_count = self._coerce_int(summary.get("control_video_count"), 1, 0, 10)
                        if control_video_count != 1:
                            return f"unsupported control_video_count: {control_video_count}"
                        if input_audio_count != 0:
                            return f"unsupported input_audio_count: {input_audio_count}"
                    elif input_audio_count != 1:
                        return f"unsupported input_audio_count: {input_audio_count}"
                elif model_id == SVI_VIDEO_MODEL_ID:
                    input_audio_count = self._coerce_int(summary.get("input_audio_count"), 0, 0, 10)
                    if "start_image_count" in summary:
                        start_image_count = self._coerce_int(summary.get("start_image_count"), 1, 0, 10)
                        if start_image_count != 1:
                            return f"unsupported start_image_count: {start_image_count}"
                        end_image_count = self._coerce_int(summary.get("end_image_count"), 0, 0, 10)
                        if end_image_count > 1:
                            return f"unsupported end_image_count: {end_image_count}"
                    else:
                        input_image_count = self._coerce_int(summary.get("input_image_count"), 1, 0, 10)
                        if input_image_count < 1 or input_image_count > 2:
                            return f"unsupported input_image_count: {input_image_count}"
                    if input_audio_count != 0:
                        return f"unsupported input_audio_count: {input_audio_count}"
                else:
                    input_audio_count = self._coerce_int(summary.get("input_audio_count"), 1, 0, 10)
                    input_image_count = self._coerce_int(summary.get("input_image_count"), 1, 0, 10)
                    if input_image_count != 1:
                        return f"unsupported input_image_count: {input_image_count}"
                    if input_audio_count != 1:
                        return f"unsupported input_audio_count: {input_audio_count}"
        return None

    def _candidate_is_event_video_processing(self, candidate: dict[str, Any]) -> bool:
        summary = candidate.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        processing_task = str(candidate.get("processing_task") or summary.get("processing_task") or "").strip().lower()
        processor_id = str(candidate.get("processor_id") or candidate.get("model_id") or summary.get("processor_id") or "").strip()
        operation_type = self._storyboard_operation_type(candidate)
        return (
            operation_type not in STORYBOARD_FFMPEG_OPERATION_TYPES
            and (processing_task == EVENT_VIDEO_PROCESSING_TASK or processor_id == EVENT_VIDEO_PROCESSOR_ID)
        )

    def _candidate_is_storyboard_ffmpeg_processing(self, candidate: dict[str, Any]) -> bool:
        summary = candidate.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        family = str(candidate.get("family") or summary.get("family") or "").strip().lower()
        processing_task = str(candidate.get("processing_task") or summary.get("processing_task") or "").strip().lower()
        processor_id = str(candidate.get("processor_id") or candidate.get("model_id") or summary.get("processor_id") or "").strip()
        operation_type = self._storyboard_operation_type(candidate)
        return (
            operation_type in STORYBOARD_FFMPEG_OPERATION_TYPES
            or processing_task == STORYBOARD_FFMPEG_PROCESSING_TASK
            or processor_id == STORYBOARD_FFMPEG_PROCESSOR_ID
            or (family == "media_processing" and operation_type in STORYBOARD_FFMPEG_OPERATION_TYPES)
        )

    def _event_video_candidate_incompatibility_reason(self, candidate: dict[str, Any]) -> Optional[str]:
        if not self._event_video_processing_capability():
            return "event video processing is unavailable on this worker"
        media_type = str(candidate.get("media_type") or "").strip().lower()
        if media_type != "video":
            return f"unsupported media processing media_type: {candidate.get('media_type')}"
        summary = candidate.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        family = str(candidate.get("family") or summary.get("family") or "").strip().lower()
        if family and family != "media_processing":
            return f"unsupported media processing family: {family}"
        processing_task = str(candidate.get("processing_task") or summary.get("processing_task") or EVENT_VIDEO_PROCESSING_TASK).strip().lower()
        if processing_task != EVENT_VIDEO_PROCESSING_TASK:
            return f"unsupported processing_task: {processing_task}"
        processor_id = str(candidate.get("processor_id") or candidate.get("model_id") or summary.get("processor_id") or EVENT_VIDEO_PROCESSOR_ID).strip()
        if processor_id != EVENT_VIDEO_PROCESSOR_ID:
            return f"unsupported processor_id: {processor_id}"
        output_format = str(summary.get("output_format") or summary.get("format") or "mp4").strip().lower()
        if output_format != "mp4":
            return f"unsupported event video output_format: {output_format}"
        output_profile = str(summary.get("output_profile") or summary.get("profile") or EVENT_VIDEO_OUTPUT_PROFILE).strip()
        if output_profile != EVENT_VIDEO_OUTPUT_PROFILE:
            return f"unsupported event video output_profile: {output_profile}"
        try:
            output_count = int(summary.get("output_count") or 1)
        except (TypeError, ValueError):
            return f"invalid event video output_count: {summary.get('output_count')}"
        if output_count != 1:
            return f"unsupported event video output_count: {output_count}"
        source_video_count = self._coerce_int(summary.get("source_video_count"), 1, 0, 10)
        if source_video_count != 1:
            return f"unsupported source_video_count: {source_video_count}"
        overlay_count = self._coerce_int(summary.get("overlay_png_count"), 0, 0, 10)
        bumper_count = self._coerce_int(summary.get("bumper_image_count"), 0, 0, 10)
        apply_overlay = self._coerce_bool(summary.get("apply_overlay"), bool(overlay_count))
        add_ending_bumper = self._coerce_bool(summary.get("add_ending_bumper"), bool(bumper_count))
        if apply_overlay and overlay_count <= 0:
            return "apply_overlay requires overlay_png inputs"
        if add_ending_bumper and bumper_count <= 0:
            return "add_ending_bumper requires bumper_image inputs"
        return None

    def _storyboard_ffmpeg_candidate_incompatibility_reason(self, candidate: dict[str, Any]) -> Optional[str]:
        if not self._storyboard_ffmpeg_processing_capability():
            return "storyboard ffmpeg processing is unavailable on this worker"
        media_type = str(candidate.get("media_type") or "").strip().lower()
        if media_type != "video":
            return f"unsupported storyboard media_type: {candidate.get('media_type')}"
        summary = candidate.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        family = str(candidate.get("family") or summary.get("family") or "").strip().lower()
        if family and family != "media_processing":
            return f"unsupported storyboard processing family: {family}"
        operation_type = self._storyboard_operation_type(candidate)
        if operation_type not in STORYBOARD_FFMPEG_OPERATION_TYPES:
            return f"unsupported storyboard operation_type: {operation_type}"
        processor_id = str(candidate.get("processor_id") or candidate.get("model_id") or summary.get("processor_id") or STORYBOARD_FFMPEG_PROCESSOR_ID).strip()
        if processor_id not in {"", STORYBOARD_FFMPEG_PROCESSOR_ID}:
            return f"unsupported storyboard processor_id: {processor_id}"
        output_format = str(summary.get("output_format") or summary.get("format") or "mp4").strip().lower()
        if output_format != "mp4":
            return f"unsupported storyboard output_format: {output_format}"
        try:
            output_count = int(summary.get("output_count") or 1)
        except (TypeError, ValueError):
            return f"invalid storyboard output_count: {summary.get('output_count')}"
        if output_count != 1:
            return f"unsupported storyboard output_count: {output_count}"
        video_count = self._coerce_int(
            summary.get("video_input_count", summary.get("source_video_count", summary.get("input_video_count"))),
            1,
            0,
            100,
        )
        if operation_type == "multicam_final_assembly":
            if video_count < 1:
                return "final assembly requires video inputs"
        elif operation_type == "replace_video_soundtrack":
            if video_count != 1:
                return f"replace_video_soundtrack requires exactly one source video input; got {video_count}"
            audio_count = self._coerce_int(summary.get("audio_input_count", summary.get("soundtrack_audio_count")), 1, 0, 10)
            if audio_count != 1:
                return f"replace_video_soundtrack requires exactly one soundtrack audio input; got {audio_count}"
        elif operation_type in STORYBOARD_SINGLE_VIDEO_OPERATION_TYPES and video_count < 1:
            return f"{operation_type} requires a video input"
        return None

    def _claim_job(self, connection: ConnectionContext, job_id: int) -> Optional[dict[str, Any]]:
        response = requests.post(
            f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{job_id}/claim",
            headers={**self._headers(connection), "Content-Type": "application/json"},
            json={"capabilities_revision": connection.capabilities_revision},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code == 409:
            return None
        if response.status_code in {401, 403} and self._forget_unauthorized_connection(
            connection,
            "Job claim",
            self._midom_error_message(response),
        ):
            return None
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Claim response was not a JSON object.")
        self._log(f"Claim succeeded; connection_id={connection.connection_id} worker_id={connection.worker_id} job_id={job_id}.")
        return payload

    def _run_job(self, connection: ConnectionContext, job: dict[str, Any]) -> None:
        self._ensure_job_flow_enabled()
        job_id = self._coerce_job_id(job)
        self._active_job_id = job_id
        self._active_job_context = connection
        self._cancel_requested_by_midom = False
        self._active_job_status = {
            "phase": "claimed",
            "status": "Midom job claimed; preparing WanGP generation.",
            "progress": 0,
            "current_step": None,
            "total_steps": None,
        }
        self._log(
            "Starting claimed job; "
            f"connection_id={connection.connection_id} worker_id={connection.worker_id} "
            f"job_id={job_id} model_id={job.get('model_id')} "
            f"media_type={job.get('media_type')} output={job.get('output')} "
            f"inputs={len(job.get('inputs') or [])}."
        )
        try:
            settings = self._validate_job(job, connection)
            media_type = str(settings.get("_midom_media_type") or "image")
            output_count = int(settings.pop("_midom_output_count"))
            downloaded_inputs = []
            with tempfile.TemporaryDirectory(prefix=f"midom-job-{job_id}-") as temp_dir:
                downloaded_inputs = self._download_job_inputs(connection, job, temp_dir)
                self._apply_inputs_to_settings(settings, downloaded_inputs, job)
                if media_type == "media_processing":
                    processor_id = str(settings.get("_midom_processor_id") or "")
                    if processor_id == EVENT_VIDEO_PROCESSOR_ID:
                        source_video_count = sum(1 for item in downloaded_inputs if item.get("kind") == "source_video")
                        overlay_count = sum(1 for item in downloaded_inputs if item.get("kind") == "overlay_png")
                        bumper_count = sum(1 for item in downloaded_inputs if item.get("kind") == "bumper_image")
                        self._log(
                            "Prepared Event Video Processing settings; "
                            f"job_id={job_id} processor_id={processor_id} "
                            f"output_profile={settings.get('_midom_output_profile')} "
                            f"processing={settings.get('_midom_processing')} "
                            f"source_video_count={source_video_count} overlay_png_count={overlay_count} "
                            f"bumper_image_count={bumper_count}."
                        )
                    elif processor_id == STORYBOARD_FFMPEG_PROCESSOR_ID:
                        self._log(
                            "Prepared Storyboard FFmpeg Processing settings; "
                            f"job_id={job_id} processor_id={processor_id} "
                            f"operation_type={settings.get('_midom_operation_type')} "
                            f"processing={settings.get('_midom_processing')} "
                            f"video_count={sum(1 for item in downloaded_inputs if item.get('category') == 'video')} "
                            f"image_count={sum(1 for item in downloaded_inputs if item.get('category') == 'image')} "
                            f"audio_count={sum(1 for item in downloaded_inputs if item.get('category') == 'audio')}."
                        )
                    else:
                        raise ValueError(f"Unsupported media processing processor_id: {processor_id}")
                elif media_type == "audio":
                    source_audio_count = sum(1 for item in downloaded_inputs if item.get("kind") == "source_audio")
                    reference_audio_count = sum(1 for item in downloaded_inputs if item.get("kind") == "reference_audio")
                    emotion_reference_audio_count = sum(1 for item in downloaded_inputs if item.get("kind") == "emotion_reference_audio")
                    speaker1_reference_audio_count = sum(1 for item in downloaded_inputs if item.get("kind") == "speaker1_reference_audio")
                    speaker2_reference_audio_count = sum(1 for item in downloaded_inputs if item.get("kind") == "speaker2_reference_audio")
                    self._log(
                        "Prepared WanGP audio settings; "
                        f"job_id={job_id} model_type={settings.get('model_type')} "
                        f"voice_mode={settings.get('_midom_voice_mode')!r} "
                        f"audio_prompt_type={settings.get('audio_prompt_type')!r} "
                        f"duration_seconds={settings.get('duration_seconds')} "
                        f"pause_seconds={settings.get('pause_seconds', 0)} "
                        f"prompt_processing_mode={settings.get('_midom_prompt_processing_mode')!r} "
                        f"model_mode={settings.get('model_mode')!r} "
                        f"custom_settings={settings.get('custom_settings') if settings.get('model_type') in (CHATTERBOX_MODEL_ID, DRAMABOX_MODEL_ID, ACE_STEP15_WANGP_MODEL_TYPE) else None!r} "
                        f"output_format={settings.get('_midom_output_format')} "
                        f"source_audio_count={source_audio_count} "
                        f"reference_audio_count={reference_audio_count} "
                        f"emotion_reference_audio_count={emotion_reference_audio_count} "
                        f"speaker1_reference_audio_count={speaker1_reference_audio_count} "
                        f"speaker2_reference_audio_count={speaker2_reference_audio_count}."
                    )
                elif media_type == "video":
                    reference_image_count = sum(1 for item in downloaded_inputs if item.get("kind") == "reference_image")
                    start_image_count = sum(1 for item in downloaded_inputs if item.get("kind") == "start_image")
                    driving_audio_count = sum(1 for item in downloaded_inputs if item.get("kind") == "driving_audio")
                    control_video_count = sum(1 for item in downloaded_inputs if item.get("kind") == "control_video")
                    self._log(
                        "Prepared WanGP video settings; "
                        f"job_id={job_id} model_type={settings.get('model_type')} "
                        f"video_task={settings.get('_midom_video_task')!r} "
                        f"audio_video_mode={settings.get('_midom_audio_video_mode')!r} "
                        f"duration_seconds={settings.get('duration_seconds')} "
                        f"video_length={settings.get('video_length')} resolution={settings.get('resolution')} "
                        f"steps={settings.get('num_inference_steps')} sample_solver={settings.get('sample_solver')!r} "
                        f"speed_profile_id={settings.get('_midom_speed_profile_id')!r} "
                        f"prompt_mode={settings.get('_midom_prompt_mode')!r} "
                        f"audio_prompt_type={settings.get('audio_prompt_type')!r} "
                        f"video_prompt_type={settings.get('video_prompt_type')!r} "
                        f"image_prompt_type={settings.get('image_prompt_type')!r} "
                        f"video_sync_profile_id={settings.get('_midom_video_sync_profile_id', 'standard')!r} "
                        f"reference_image_count={reference_image_count} start_image_count={start_image_count} "
                        f"driving_audio_count={driving_audio_count} control_video_count={control_video_count}."
                    )
                else:
                    reference_count = sum(1 for item in downloaded_inputs if item.get("kind") == "reference_image")
                    control_count = sum(1 for item in downloaded_inputs if item.get("kind") == "control_image")
                    self._log(
                        "Prepared WanGP settings; "
                        f"job_id={job_id} model_type={settings.get('model_type')} "
                        f"image_mode={settings.get('image_mode')} video_prompt_type={settings.get('video_prompt_type', '')!r} "
                        f"resolution={settings.get('resolution')} steps={settings.get('num_inference_steps', 'WanGP default')} "
                        f"accelerator_profile={settings.get('_midom_accelerator_profile_id', 'standard')} "
                        f"curated_tool_id={settings.get('_midom_curated_tool_id', '')!r} "
                        f"seed={settings.get('seed', 'none')} output_count={output_count} "
                        f"reference_count={reference_count} control_count={control_count}."
                    )
                self._set_active_job_status(
                    phase="starting",
                    status="Processing started; running local FFmpeg." if media_type == "media_processing" else "Generation started; submitting to WanGP.",
                    progress=0,
                )
                self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
                if self._cancel_requested_by_midom:
                    self._post_job_update(
                        connection,
                        job_id,
                        "fail",
                        {"reason": "cancel_requested", "message": "Midom requested cancellation"},
                    )
                    return
                if media_type == "media_processing":
                    process_handle = LocalProcessJob()
                    self._active_job = process_handle
                    processor_id = str(settings.get("_midom_processor_id") or "")
                    try:
                        try:
                            if processor_id == EVENT_VIDEO_PROCESSOR_ID:
                                output_path, processing_metadata = self._run_event_video_processing_job(
                                    connection,
                                    job_id,
                                    settings,
                                    downloaded_inputs,
                                    temp_dir,
                                    process_handle,
                                )
                                cancel_message = "Event Video Processing was cancelled."
                            elif processor_id == STORYBOARD_FFMPEG_PROCESSOR_ID:
                                output_path, processing_metadata = self._run_storyboard_ffmpeg_processing_job(
                                    connection,
                                    job_id,
                                    settings,
                                    downloaded_inputs,
                                    temp_dir,
                                    process_handle,
                                )
                                cancel_message = "Storyboard FFmpeg Processing was cancelled."
                            else:
                                raise ValueError(f"Unsupported media processing processor_id: {processor_id}")
                        except RuntimeError as exc:
                            if str(exc) == "cancel_requested":
                                self._post_job_update(
                                    connection,
                                    job_id,
                                    "fail",
                                    {"reason": "cancel_requested", "message": cancel_message},
                                )
                                return
                            raise
                    finally:
                        process_handle.mark_done()
                        if self._active_job is process_handle:
                            self._active_job = None
                    if self._cancel_requested_by_midom or process_handle.cancelled:
                        self._post_job_update(
                            connection,
                            job_id,
                            "fail",
                            {"reason": "cancel_requested", "message": f"{processing_metadata.get('processing_task', 'Media processing')} was cancelled."},
                        )
                        return
                    artifact = self._upload_video_artifact(connection, job_id, output_path, 0, settings)
                    complete_payload = {
                        "artifacts": [artifact],
                        "backend": "ffmpeg",
                        "model_id": processor_id,
                        "processor_id": processor_id,
                        "processing_metadata": processing_metadata,
                    }
                    self._post_job_update(connection, job_id, "complete", complete_payload)
                    self._log(f"Media processing complete accepted by Midom; job_id={job_id} processor_id={processor_id}.")
                    return
                callbacks = self._callbacks_for_job(connection, job_id)
                self._log(
                    "Submitting job to background WanGP generation session; "
                    f"job_id={job_id} media_type={media_type} output_count={output_count}."
                )
                job_handle = self._submit_wangp_job(self._get_generation_session(), settings, output_count, callbacks)
                self._active_job = job_handle
                try:
                    result = self._wait_for_wangp_result(connection, job_id, job_handle, callbacks)
                finally:
                    if self._active_job is job_handle:
                        self._active_job = None
                self._log(
                    "WanGP job returned; "
                    f"job_id={job_id} success={getattr(result, 'success', None)} "
                    f"cancelled={getattr(result, 'cancelled', None)} "
                    f"generated_files={len(getattr(result, 'generated_files', []) or [])} "
                    f"errors={len(getattr(result, 'errors', []) or [])}."
                )
                if self._cancel_requested_by_midom:
                    self._post_job_update(
                        connection,
                        job_id,
                        "fail",
                        {"reason": "cancel_requested", "message": "Midom requested cancellation"},
                    )
                    return
                if getattr(result, "cancelled", False):
                    self._post_job_update(
                        connection,
                        job_id,
                        "fail",
                        {"reason": "worker_stopped", "message": "WanGP worker stopped or cancelled the local generation"},
                    )
                    return
                if not result.success:
                    errors = list(result.errors or [])
                    message = str(errors[0] if errors else "WanGP generation failed.")
                    self._post_job_update(connection, job_id, "fail", {"reason": "generation_failed", "message": message})
                    return
                if media_type == "audio":
                    generated_files = self._result_audio_files(result)
                    if len(generated_files) > 1 and output_count == 1:
                        segment_count = len(generated_files)
                        combined_audio_path = self._combine_audio_segments(generated_files, temp_dir)
                        settings["_midom_combined_audio_segment_count"] = segment_count
                        generated_files = [combined_audio_path]
                elif media_type == "video":
                    generated_files = self._result_video_files(result, settings)
                else:
                    generated_files = self._result_image_files(result)
                if media_type == "audio":
                    self._log(
                        "Audio generation output check; "
                        f"job_id={job_id} generated_audio_files={len(generated_files)} "
                        f"expected_outputs={output_count} requested_format={settings.get('_midom_output_format')}."
                    )
                elif media_type == "video":
                    self._log(
                        "Video generation output check; "
                        f"job_id={job_id} generated_video_files={len(generated_files)} "
                        f"expected_outputs={output_count} requested_format={settings.get('_midom_output_format')}."
                    )
                if len(generated_files) < output_count:
                    self._post_job_update(
                        connection,
                        job_id,
                        "fail",
                        {
                            "reason": "generation_output_count_mismatch",
                            "message": f"WanGP returned {len(generated_files)} output(s), expected {output_count}.",
                        },
                    )
                    return
                artifacts = []
                for artifact_index, file_path in enumerate(generated_files[:output_count]):
                    if media_type == "audio":
                        artifacts.append(self._upload_audio_artifact(connection, job_id, file_path, artifact_index, settings, temp_dir))
                    elif media_type == "video":
                        artifacts.append(self._upload_video_artifact(connection, job_id, file_path, artifact_index, settings))
                    else:
                        artifacts.append(self._upload_artifact(connection, job_id, file_path, artifact_index, settings.get("resolution")))
                generation_metadata = self._build_generation_metadata(settings, result, generated_files[:output_count])
                self._log(f"All artifacts uploaded; calling complete; job_id={job_id} artifacts={len(artifacts)}.")
                complete_payload = {
                    "artifacts": artifacts,
                    "backend": "wangp",
                    "model_id": str(job.get("model_id") or settings.get("model_type") or ""),
                }
                if generation_metadata:
                    complete_payload["generation_metadata"] = generation_metadata
                    self._log(
                        "Prepared generation metadata for complete; "
                        f"job_id={job_id} resolved_seed={generation_metadata.get('resolved_seed')} "
                        f"seed_source={generation_metadata.get('seed_source')!r} "
                        f"artifact_seeds={len(generation_metadata.get('artifact_seeds') or [])}."
                    )
                self._post_job_update(
                    connection,
                    job_id,
                    "complete",
                    complete_payload,
                )
                self._log(f"Job complete accepted by Midom; job_id={job_id}.")
        except Exception as exc:
            self._log(f"Job failed locally; attempting fail report; job_id={job_id} error={exc}", force=True)
            try:
                self._post_job_update(connection, job_id, "fail", {"reason": "validation_or_runtime_error", "message": str(exc)})
                self._log(f"Fail report accepted by Midom; job_id={job_id}.")
            except Exception as fail_exc:
                self._log(f"Failed to report job {job_id} failure: {fail_exc}", force=True)
        finally:
            self._active_job = None
            self._active_job_id = None
            self._active_job_context = None
            self._cancel_requested_by_midom = False
            self._active_job_status = {}
            self._reset_idle_backoff_for_active_use(f"finished Midom job {job_id}", wake=True)
            self._log(f"Cleared active job state; job_id={job_id}.")

    def _run_event_video_processing_job(
        self,
        connection: ConnectionContext,
        job_id: int,
        settings: dict[str, Any],
        downloaded_inputs: list[dict[str, Any]],
        temp_dir: str,
        process_handle: LocalProcessJob,
    ) -> tuple[str, dict[str, Any]]:
        processing = settings.get("_midom_processing") or {}
        source_inputs = [item for item in downloaded_inputs if item.get("kind") == "source_video"]
        if len(source_inputs) != 1:
            raise ValueError(f"Event Video Processing requires exactly one source_video input; got {len(source_inputs)}.")
        source = source_inputs[0]
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else self._probe_event_video_metadata(Path(source["path"]))
        output_width, output_height = self._event_video_output_size(metadata, processing)
        output_orientation = "portrait" if output_height > output_width else "landscape"
        overlay_path = self._event_processing_asset_for_orientation(
            downloaded_inputs,
            "overlay_png",
            output_orientation,
            bool(processing.get("apply_overlay")),
        )
        bumper_path = self._event_processing_asset_for_orientation(
            downloaded_inputs,
            "bumper_image",
            output_orientation,
            bool(processing.get("add_ending_bumper")),
        )
        main_output = Path(temp_dir) / "event-video-main.mp4"
        final_output = Path(temp_dir) / "event-video-processed.mp4"
        source_duration = float(metadata.get("duration_seconds") or 0.0)
        self._set_active_job_status(
            phase="processing",
            status=f"Normalizing Event video to {output_width}x{output_height}.",
            progress=3,
        )
        self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
        self._run_event_video_main_ffmpeg(
            connection,
            job_id,
            Path(source["path"]),
            overlay_path,
            main_output,
            metadata,
            output_width,
            output_height,
            process_handle,
        )
        final_path = main_output
        bumper_added = False
        if bumper_path is not None:
            bumper_clip = Path(temp_dir) / "event-video-bumper.mp4"
            self._set_active_job_status(
                phase="processing",
                status="Rendering Event video ending bumper.",
                progress=76,
            )
            self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
            self._run_event_bumper_ffmpeg(
                connection,
                job_id,
                bumper_path,
                bumper_clip,
                output_width,
                output_height,
                process_handle,
            )
            self._set_active_job_status(
                phase="processing",
                status="Appending Event video ending bumper.",
                progress=86,
            )
            self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
            self._concat_event_video_segments(
                connection,
                job_id,
                [main_output, bumper_clip],
                final_output,
                source_duration + EVENT_VIDEO_BUMPER_SECONDS,
                process_handle,
            )
            final_path = final_output
            bumper_added = True
        if not final_path.is_file() or final_path.stat().st_size <= 0:
            raise ValueError("Event Video Processing did not produce a processed MP4.")
        expected_max_duration = source_duration + (EVENT_VIDEO_BUMPER_SECONDS if bumper_added else 0)
        output_metadata = self._validate_event_video_output(
            final_path,
            expected_width=output_width,
            expected_height=output_height,
            expected_max_duration_seconds=expected_max_duration,
            max_bytes=self._coerce_int(settings.get("_midom_max_artifact_bytes"), MAX_EVENT_VIDEO_OUTPUT_BYTES, 1, MAX_EVENT_VIDEO_OUTPUT_BYTES),
        )
        self._set_active_job_status(
            phase="uploading",
            status="Event Video Processing finished; uploading processed MP4.",
            progress=96,
        )
        self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
        return str(final_path), {
            "processing_task": EVENT_VIDEO_PROCESSING_TASK,
            "processor_id": EVENT_VIDEO_PROCESSOR_ID,
            "output_profile": EVENT_VIDEO_OUTPUT_PROFILE,
            "ffmpeg_encoder": EVENT_VIDEO_H264_ENCODER,
            "source_duration_seconds": source_duration,
            "output_duration_seconds": float(output_metadata.get("duration_seconds") or 0.0),
            "source_width": int(metadata.get("display_width") or metadata.get("width") or 0),
            "source_height": int(metadata.get("display_height") or metadata.get("height") or 0),
            "output_width": int(output_metadata.get("display_width") or output_width),
            "output_height": int(output_metadata.get("display_height") or output_height),
            "output_orientation": output_orientation,
            "overlay_applied": overlay_path is not None,
            "bumper_added": bumper_added,
            "worker_id": connection.worker_id,
        }

    def _validate_event_video_output(
        self,
        path: Path,
        *,
        expected_width: int,
        expected_height: int,
        expected_max_duration_seconds: float,
        max_bytes: int,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError("Event Video Processing output MP4 is missing.")
        file_size = path.stat().st_size
        if file_size <= 0 or file_size > max_bytes:
            raise ValueError(f"Event Video Processing output size is outside allowed bounds: {file_size} bytes.")
        with path.open("rb") as reader:
            header = reader.read(64)
        if not self._looks_like_mp4(header):
            raise ValueError("Event Video Processing output does not look like MP4 bytes.")
        metadata = self._probe_event_video_metadata(path)
        output_width = int(metadata.get("display_width") or metadata.get("width") or 0)
        output_height = int(metadata.get("display_height") or metadata.get("height") or 0)
        if (output_width, output_height) != (int(expected_width), int(expected_height)):
            raise ValueError(
                "Event Video Processing output dimensions do not match the selected profile; "
                f"expected {expected_width}x{expected_height}, got {output_width}x{output_height}."
            )
        duration = float(metadata.get("duration_seconds") or 0.0)
        duration_tolerance = max(2.0, float(expected_max_duration_seconds or 0.0) * 0.05)
        if duration <= 0:
            raise ValueError("Event Video Processing output duration could not be read.")
        if expected_max_duration_seconds > 0 and duration > expected_max_duration_seconds + duration_tolerance:
            raise ValueError(
                "Event Video Processing output duration exceeds expected bounds; "
                f"expected <= {expected_max_duration_seconds + duration_tolerance:.2f}s, got {duration:.2f}s."
            )
        if not metadata.get("has_audio"):
            raise ValueError("Event Video Processing output must contain an AAC audio stream, using silence when source has no audio.")
        video_codec = str(metadata.get("video_codec") or "").lower()
        audio_codec = str(metadata.get("audio_codec") or "").lower()
        if video_codec != "h264":
            raise ValueError(f"Event Video Processing output must use H.264 video; got {video_codec or 'unknown'}.")
        if audio_codec != "aac":
            raise ValueError(f"Event Video Processing output must use AAC audio; got {audio_codec or 'unknown'}.")
        if int(metadata.get("rotation_degrees") or 0) % 360 != 0:
            raise ValueError("Event Video Processing output must have physically rotated pixels and no rotation metadata.")
        self._log(
            "Validated Event Video Processing output; "
            f"filename={path.name!r} dimensions={output_width}x{output_height} "
            f"duration_seconds={duration:.2f} bytes={file_size} video_codec={video_codec} audio_codec={audio_codec}."
        )
        return metadata

    def _run_storyboard_ffmpeg_processing_job(
        self,
        connection: ConnectionContext,
        job_id: int,
        settings: dict[str, Any],
        downloaded_inputs: list[dict[str, Any]],
        temp_dir: str,
        process_handle: LocalProcessJob,
    ) -> tuple[str, dict[str, Any]]:
        processing = settings.get("_midom_processing") or {}
        operation_type = str(settings.get("_midom_operation_type") or "").strip()
        video_inputs = self._storyboard_video_inputs(downloaded_inputs)
        if not video_inputs:
            raise ValueError("Storyboard FFmpeg Processing requires at least one video input.")
        output_width, output_height = self._storyboard_output_size(video_inputs[0], processing)
        final_output = Path(temp_dir) / "storyboard-processed.mp4"
        self._set_active_job_status(
            phase="processing",
            status=f"Running storyboard FFmpeg operation {operation_type}.",
            progress=3,
        )
        self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
        if operation_type == "multicam_final_assembly":
            assembly_inputs = self._storyboard_ordered_assembly_inputs(video_inputs, processing)
            segment_paths = []
            progress_cursor = 5
            progress_span = max(1, 75 // max(1, len(assembly_inputs)))
            total_seconds = 0.0
            for index, video_input in enumerate(assembly_inputs):
                segment_output = Path(temp_dir) / f"storyboard-segment-{index}.mp4"
                segment_end = min(80, progress_cursor + progress_span)
                segment_processing = self._storyboard_segment_processing(video_input, processing, index)
                segment_duration = self._storyboard_effective_segment_duration(video_input, segment_processing)
                total_seconds += segment_duration
                self._run_storyboard_video_ffmpeg(
                    connection,
                    job_id,
                    video_input,
                    None,
                    None,
                    None,
                    segment_output,
                    output_width,
                    output_height,
                    process_handle,
                    processing=segment_processing,
                    progress_start=progress_cursor,
                    progress_end=segment_end,
                    status=f"Preparing storyboard assembly segment {index + 1} of {len(assembly_inputs)}.",
                )
                segment_paths.append(segment_output)
                progress_cursor = min(81, segment_end + 1)
            self._concat_event_video_segments(
                connection,
                job_id,
                segment_paths,
                final_output,
                total_seconds,
                process_handle,
                progress_start=82,
                progress_end=95,
                status="Assembling storyboard final MP4.",
            )
            video_inputs = assembly_inputs
        elif operation_type in {"multicam_optimize_video", "optimize_video"}:
            primary = self._storyboard_primary_video_input(video_inputs)
            output_width, output_height = self._storyboard_optimize_output_size(primary, processing)
            self._run_storyboard_optimize_video_ffmpeg(
                connection,
                job_id,
                primary,
                final_output,
                output_width,
                output_height,
                process_handle,
                processing=processing,
                progress_start=5,
                progress_end=95,
                status=f"Optimizing storyboard video operation {operation_type}.",
            )
        elif operation_type == "replace_video_soundtrack":
            primary = self._storyboard_primary_video_input(video_inputs)
            soundtrack_input = self._storyboard_audio_input(downloaded_inputs)
            if soundtrack_input is None:
                raise ValueError("Storyboard replace_video_soundtrack requires one soundtrack_audio input.")
            self._run_storyboard_replace_soundtrack_ffmpeg(
                connection,
                job_id,
                primary,
                soundtrack_input,
                final_output,
                output_width,
                output_height,
                process_handle,
                processing=processing,
                progress_start=5,
                progress_end=95,
                status="Replacing storyboard video soundtrack.",
            )
        else:
            primary = self._storyboard_primary_video_input(video_inputs)
            overlay = self._storyboard_overlay_input(downloaded_inputs, processing) if operation_type == "multicam_card_overlay_take" else None
            matte = self._storyboard_matte_input(downloaded_inputs, processing) if operation_type == "multicam_card_overlay_take" else None
            audio_input = self._storyboard_audio_input(downloaded_inputs)
            self._run_storyboard_video_ffmpeg(
                connection,
                job_id,
                primary,
                overlay,
                matte,
                audio_input,
                final_output,
                output_width,
                output_height,
                process_handle,
                processing=processing,
                progress_start=5,
                progress_end=95,
                status=f"Rendering storyboard operation {operation_type}.",
            )
        if not final_output.is_file() or final_output.stat().st_size <= 0:
            raise ValueError("Storyboard FFmpeg Processing did not produce a processed MP4.")
        output_metadata = self._validate_storyboard_ffmpeg_output(
            final_output,
            expected_width=output_width,
            expected_height=output_height,
            max_bytes=self._coerce_int(settings.get("_midom_max_artifact_bytes"), MAX_STORYBOARD_VIDEO_OUTPUT_BYTES, 1, MAX_STORYBOARD_VIDEO_OUTPUT_BYTES),
        )
        self._set_active_job_status(
            phase="uploading",
            status="Storyboard FFmpeg Processing finished; uploading processed MP4.",
            progress=96,
        )
        self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
        result_metadata = {
            "processing_task": STORYBOARD_FFMPEG_PROCESSING_TASK,
            "processor_id": STORYBOARD_FFMPEG_PROCESSOR_ID,
            "operation_type": operation_type,
            "ffmpeg_encoder": EVENT_VIDEO_H264_ENCODER,
            "output_width": int(output_metadata.get("display_width") or output_width),
            "output_height": int(output_metadata.get("display_height") or output_height),
            "output_duration_seconds": float(output_metadata.get("duration_seconds") or 0.0),
            "fps": self._coerce_int(processing.get("fps"), STORYBOARD_OUTPUT_FPS, 1, 120),
            "video_input_count": len(video_inputs),
            "image_input_count": sum(1 for item in downloaded_inputs if item.get("category") == "image"),
            "audio_input_count": sum(1 for item in downloaded_inputs if item.get("category") == "audio"),
            "worker_id": connection.worker_id,
        }
        if operation_type == "multicam_final_assembly":
            result_metadata["segment_count"] = len(video_inputs)
        return str(final_output), result_metadata

    def _storyboard_ordered_assembly_inputs(
        self,
        video_inputs: list[dict[str, Any]],
        processing: dict[str, Any],
    ) -> list[dict[str, Any]]:
        segments = processing.get("segments")
        if not isinstance(segments, list) or not segments:
            return list(video_inputs)
        input_by_id = {int(item.get("input_id") or 0): item for item in video_inputs}
        ordered = []
        missing_ids = []
        segment_entries = [
            (self._coerce_int(segment.get("order", segment.get("sequence", index)), index, 0, 10_000), index, segment)
            for index, segment in enumerate(segments)
            if isinstance(segment, dict)
        ]
        for _order, index, segment in sorted(segment_entries, key=lambda entry: (entry[0], entry[1])):
            input_id = self._coerce_int(segment.get("input_id"), 0, 0, 2_147_483_647)
            if input_id <= 0:
                continue
            match = input_by_id.get(input_id)
            if not match:
                missing_ids.append(input_id)
                continue
            ordered.append({**match, "_segment_index": index, "_segment_payload": segment})
        if missing_ids:
            raise ValueError(f"Storyboard final assembly referenced missing input_id(s): {missing_ids}")
        if not ordered:
            return list(video_inputs)
        return ordered

    def _storyboard_segment_processing(
        self,
        video_input: dict[str, Any],
        processing: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        segment = video_input.get("_segment_payload") if isinstance(video_input.get("_segment_payload"), dict) else {}
        trim_start = self._coerce_float(
            self._first_present(segment, "trim_start_seconds", "start_seconds", "start_time_seconds"),
            self._coerce_float(processing.get("trim_start_seconds"), 0.0, 0.0, MAX_STORYBOARD_VIDEO_DURATION_SECONDS),
            0.0,
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        trim_duration = self._optional_positive_float(
            self._first_present(segment, "trim_duration_seconds", "duration_seconds"),
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        trim_end = self._optional_positive_float(
            self._first_present(segment, "trim_end_seconds", "end_seconds", "end_time_seconds"),
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        if trim_duration is None and trim_end is None:
            trim_duration = self._optional_positive_float(processing.get("trim_duration_seconds"), MAX_STORYBOARD_VIDEO_DURATION_SECONDS)
            trim_end = self._optional_positive_float(processing.get("trim_end_seconds"), MAX_STORYBOARD_VIDEO_DURATION_SECONDS)
        return {
            **processing,
            "operation_type": "multicam_final_assembly_segment",
            "trim_start_seconds": trim_start,
            "trim_duration_seconds": trim_duration,
            "trim_end_seconds": trim_end,
            "segment_index": int(video_input.get("_segment_index", index)),
        }

    def _storyboard_effective_segment_duration(self, video_input: dict[str, Any], processing: dict[str, Any]) -> float:
        metadata = video_input.get("metadata") if isinstance(video_input.get("metadata"), dict) else {}
        source_duration = float(metadata.get("duration_seconds") or 1.0)
        trim_start = self._coerce_float(processing.get("trim_start_seconds"), 0.0, 0.0, MAX_STORYBOARD_VIDEO_DURATION_SECONDS)
        trim_duration = self._storyboard_trim_duration(source_duration, processing, trim_start)
        return max(0.1, float(trim_duration or max(0.1, source_duration - trim_start)))

    def _storyboard_optimize_output_size(self, video_input: dict[str, Any], processing: dict[str, Any]) -> tuple[int, int]:
        metadata = video_input.get("metadata") if isinstance(video_input.get("metadata"), dict) else {}
        source_width = self._coerce_int(metadata.get("display_width") or metadata.get("width"), 1280, 2, 8192)
        source_height = self._coerce_int(metadata.get("display_height") or metadata.get("height"), 720, 2, 8192)
        requested_width = self._coerce_int(processing.get("output_width") or processing.get("width"), 0, 0, 8192)
        requested_height = self._coerce_int(processing.get("output_height") or processing.get("height"), 0, 0, 8192)
        if requested_width > 0 and requested_height > 0:
            return self._even_video_size(requested_width, requested_height)
        max_dimension = self._coerce_int(processing.get("max_dimension"), 0, 0, 8192)
        if max_dimension <= 0 or max(source_width, source_height) <= max_dimension:
            return self._even_video_size(source_width, source_height)
        ratio = float(max_dimension) / float(max(source_width, source_height))
        return self._even_video_size(max(2, int(round(source_width * ratio))), max(2, int(round(source_height * ratio))))

    def _run_storyboard_optimize_video_ffmpeg(
        self,
        connection: ConnectionContext,
        job_id: int,
        video_input: dict[str, Any],
        output_path: Path,
        output_width: int,
        output_height: int,
        process_handle: LocalProcessJob,
        *,
        processing: dict[str, Any],
        progress_start: int,
        progress_end: int,
        status: str,
    ) -> None:
        source_path = Path(str(video_input["path"]))
        metadata = video_input.get("metadata") if isinstance(video_input.get("metadata"), dict) else self._probe_event_video_metadata(source_path)
        source_duration = float(metadata.get("duration_seconds") or 1.0)
        has_audio = bool(metadata.get("has_audio"))
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-v",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-i",
            str(source_path),
        ]
        audio_label = "0:a:0"
        if not has_audio:
            command.extend([
                "-f",
                "lavfi",
                "-t",
                f"{source_duration:.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ])
            audio_label = "1:a:0"
        video_filter = (
            f"[0:v]scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={self._coerce_int(processing.get('fps'), STORYBOARD_OUTPUT_FPS, 1, 120)},"
            "setsar=1,format=yuv420p[vout];"
            f"[{audio_label}]aresample=48000,aformat=channel_layouts=stereo[aout]"
        )
        command.extend([
            "-filter_complex",
            video_filter,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            *self._storyboard_encode_args(processing),
            str(output_path),
        ])
        self._run_ffmpeg_with_progress(
            connection,
            job_id,
            command,
            total_seconds=max(1.0, source_duration),
            progress_start=progress_start,
            progress_end=progress_end,
            phase="processing",
            status=status,
            process_handle=process_handle,
            timeout_seconds=self._storyboard_step_timeout(source_duration),
        )

    def _run_storyboard_replace_soundtrack_ffmpeg(
        self,
        connection: ConnectionContext,
        job_id: int,
        video_input: dict[str, Any],
        soundtrack_input: dict[str, Any],
        output_path: Path,
        output_width: int,
        output_height: int,
        process_handle: LocalProcessJob,
        *,
        processing: dict[str, Any],
        progress_start: int,
        progress_end: int,
        status: str,
    ) -> None:
        source_path = Path(str(video_input["path"]))
        soundtrack_path = Path(str(soundtrack_input["path"]))
        video_metadata = video_input.get("metadata") if isinstance(video_input.get("metadata"), dict) else self._probe_event_video_metadata(source_path)
        audio_metadata = soundtrack_input.get("metadata") if isinstance(soundtrack_input.get("metadata"), dict) else self._probe_audio_metadata(soundtrack_path)
        source_duration = float(video_metadata.get("duration_seconds") or 1.0)
        audio_duration = float(audio_metadata.get("duration_seconds") or 1.0)
        video_start = self._coerce_float(
            self._first_present(processing, "video_start_seconds", "trim_start_seconds", "start_seconds"),
            0.0,
            0.0,
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        requested_duration = self._optional_positive_float(
            self._first_present(processing, "output_duration_seconds", "duration_seconds", "trim_duration_seconds"),
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        remaining_video_duration = max(0.1, source_duration - video_start)
        effective_duration = min(requested_duration or remaining_video_duration, remaining_video_duration)
        effective_duration = max(0.1, effective_duration)
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-v",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-i",
            str(source_path),
            "-i",
            str(soundtrack_path),
        ]
        video_filter = (
            f"[0:v]trim=start={video_start:.3f}:duration={effective_duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={self._coerce_int(processing.get('fps'), STORYBOARD_OUTPUT_FPS, 1, 120)},"
            "setsar=1,format=yuv420p[vout]"
        )
        filter_parts = [video_filter]
        filter_parts.extend(self._storyboard_external_audio_filter_parts("1:a:0", processing, effective_duration))
        command.extend([
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-t",
            f"{effective_duration:.3f}",
            *self._storyboard_encode_args(processing),
            str(output_path),
        ])
        self._log(
            "Replacing storyboard video soundtrack; "
            f"job_id={job_id} source_input_id={video_input.get('input_id')} "
            f"soundtrack_input_id={soundtrack_input.get('input_id')} "
            f"soundtrack_kind={soundtrack_input.get('kind')} soundtrack_mime={soundtrack_input.get('mime_type')} "
            f"video_start={video_start:.3f} audio_duration={audio_duration:.3f} output_duration={effective_duration:.3f}."
        )
        self._run_ffmpeg_with_progress(
            connection,
            job_id,
            command,
            total_seconds=max(1.0, effective_duration),
            progress_start=progress_start,
            progress_end=progress_end,
            phase="processing",
            status=status,
            process_handle=process_handle,
            timeout_seconds=self._storyboard_step_timeout(effective_duration),
        )

    def _run_storyboard_video_ffmpeg(
        self,
        connection: ConnectionContext,
        job_id: int,
        video_input: dict[str, Any],
        overlay_input: Optional[dict[str, Any]],
        matte_input: Optional[dict[str, Any]],
        audio_input: Optional[dict[str, Any]],
        output_path: Path,
        output_width: int,
        output_height: int,
        process_handle: LocalProcessJob,
        *,
        processing: dict[str, Any],
        progress_start: int,
        progress_end: int,
        status: str,
    ) -> None:
        source_path = Path(str(video_input["path"]))
        metadata = video_input.get("metadata") if isinstance(video_input.get("metadata"), dict) else self._probe_event_video_metadata(source_path)
        source_duration = float(metadata.get("duration_seconds") or 1.0)
        trim_start = self._coerce_float(processing.get("trim_start_seconds"), 0.0, 0.0, MAX_STORYBOARD_VIDEO_DURATION_SECONDS)
        trim_duration = self._storyboard_trim_duration(source_duration, processing, trim_start)
        command = [self._ffmpeg_binary(), "-y", "-hide_banner", "-v", "error", "-progress", "pipe:1", "-nostats"]
        if trim_start > 0:
            command.extend(["-ss", f"{trim_start:.3f}"])
        command.extend(["-i", str(source_path)])
        next_input_index = 1
        audio_input_label = "0:a:0"
        effective_duration = float(trim_duration or max(0.1, source_duration - trim_start))
        external_audio = audio_input is not None
        if external_audio:
            command.extend(["-i", str(audio_input["path"])])
            audio_input_label = f"{next_input_index}:a:0"
            next_input_index += 1
        overlay_input_index = None
        matte_input_index = None
        if overlay_input is not None:
            if overlay_input.get("category") == "image":
                command.extend(["-loop", "1", "-t", f"{effective_duration:.3f}", "-i", str(overlay_input["path"])])
            else:
                command.extend(["-i", str(overlay_input["path"])])
            overlay_input_index = next_input_index
            next_input_index += 1
        if matte_input is not None and overlay_input_index is None:
            raise ValueError("Overlay matte was provided but could not be applied: no overlay input was provided.")
        if matte_input is not None:
            command.extend(["-loop", "1", "-t", f"{effective_duration:.3f}", "-i", str(matte_input["path"])])
            matte_input_index = next_input_index
            next_input_index += 1
        if overlay_input is not None and overlay_input.get("category") == "video":
            overlay_metadata = overlay_input.get("metadata") if isinstance(overlay_input.get("metadata"), dict) else {}
            overlay_duration = float(overlay_metadata.get("duration_seconds") or 0.0)
            if overlay_duration > 0:
                effective_duration = max(0.1, min(effective_duration, overlay_duration))
                trim_duration = effective_duration
        audio_source = str(processing.get("audio_source") or "").strip().lower()
        if overlay_input_index is not None and audio_source in {"base", "overlay"}:
            if audio_source == "overlay":
                audio_input_label = f"{overlay_input_index}:a:0"
                overlay_metadata = overlay_input.get("metadata") if isinstance(overlay_input.get("metadata"), dict) else {}
                selected_audio_has_stream = bool(overlay_metadata.get("has_audio"))
            else:
                audio_input_label = "0:a:0"
                selected_audio_has_stream = bool(metadata.get("has_audio"))
        else:
            selected_audio_has_stream = bool(external_audio or metadata.get("has_audio"))
        if overlay_input_index is not None and audio_source in {"base", "overlay"} and not selected_audio_has_stream:
            source_name = "overlay_video" if audio_source == "overlay" else "source_video"
            raise ValueError(
                f"Storyboard overlay audio_source={audio_source!r} requested, but {source_name} has no audio stream."
            )
        if not selected_audio_has_stream:
            command.extend([
                "-f",
                "lavfi",
                "-t",
                f"{effective_duration:.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ])
            audio_input_label = f"{next_input_index}:a:0"
            next_input_index += 1
        fit_mode = str(processing.get("fit_mode") or processing.get("fit") or "contain").strip().lower()
        if self._coerce_bool(processing.get("crop"), False) or fit_mode in {"cover", "crop"}:
            video_filter = (
                f"[0:v]scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
                f"crop={output_width}:{output_height},setsar=1,format=rgba[vbase]"
            )
        else:
            video_filter = (
                f"[0:v]scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                "setsar=1,format=rgba[vbase]"
            )
        filter_parts = [video_filter]
        if external_audio:
            filter_parts.extend(self._storyboard_external_audio_filter_parts(audio_input_label, processing, effective_duration))
        else:
            filter_parts.append(f"[{audio_input_label}]aresample=48000,aformat=channel_layouts=stereo[aout]")
        if overlay_input_index is not None:
            self._log_storyboard_overlay_matte_state(job_id, overlay_input, matte_input)
            try:
                overlay_filter_parts = self._storyboard_overlay_filter_parts(
                        overlay_input,
                        overlay_input_index,
                        matte_input_index,
                        output_width,
                        output_height,
                        processing,
                        effective_duration,
                )
            except Exception as exc:
                if matte_input is not None:
                    raise ValueError(f"Overlay matte was provided but could not be applied: {exc}") from exc
                raise
            filter_parts.extend(overlay_filter_parts)
        else:
            filter_parts.append("[vbase]format=yuv420p[vout]")
        command.extend([
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-r",
            str(self._coerce_int(processing.get("fps"), STORYBOARD_OUTPUT_FPS, 1, 120)),
            *(["-t", f"{trim_duration:.3f}"] if trim_duration is not None else []),
            *self._event_video_encode_args(),
            str(output_path),
        ])
        try:
            self._run_ffmpeg_with_progress(
                connection,
                job_id,
                command,
                total_seconds=max(1.0, effective_duration),
                progress_start=progress_start,
                progress_end=progress_end,
                phase="processing",
                status=status,
                process_handle=process_handle,
                timeout_seconds=self._storyboard_step_timeout(effective_duration),
            )
        except Exception as exc:
            if matte_input is not None and str(exc) != "cancel_requested":
                raise ValueError(f"Overlay matte was provided but could not be applied: {exc}") from exc
            raise

    def _validate_storyboard_ffmpeg_output(
        self,
        path: Path,
        *,
        expected_width: int,
        expected_height: int,
        max_bytes: int,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError("Storyboard FFmpeg output MP4 is missing.")
        file_size = path.stat().st_size
        if file_size <= 0 or file_size > max_bytes:
            raise ValueError(f"Storyboard FFmpeg output size is outside allowed bounds: {file_size} bytes.")
        with path.open("rb") as reader:
            header = reader.read(64)
        if not self._looks_like_mp4(header):
            raise ValueError("Storyboard FFmpeg output does not look like MP4 bytes.")
        metadata = self._probe_event_video_metadata(path)
        output_width = int(metadata.get("display_width") or metadata.get("width") or 0)
        output_height = int(metadata.get("display_height") or metadata.get("height") or 0)
        if expected_width > 0 and expected_height > 0 and (output_width, output_height) != (int(expected_width), int(expected_height)):
            raise ValueError(
                "Storyboard FFmpeg output dimensions do not match the requested output; "
                f"expected {expected_width}x{expected_height}, got {output_width}x{output_height}."
            )
        duration = float(metadata.get("duration_seconds") or 0.0)
        if duration <= 0:
            raise ValueError("Storyboard FFmpeg output duration could not be read.")
        if duration > MAX_STORYBOARD_VIDEO_DURATION_SECONDS + 0.05:
            raise ValueError(
                "Storyboard FFmpeg output duration exceeds bridge safety limit; "
                f"max={MAX_STORYBOARD_VIDEO_DURATION_SECONDS}s got={duration:.2f}s."
            )
        video_codec = str(metadata.get("video_codec") or "").lower()
        audio_codec = str(metadata.get("audio_codec") or "").lower()
        if video_codec != "h264":
            raise ValueError(f"Storyboard FFmpeg output must use H.264 video; got {video_codec or 'unknown'}.")
        if audio_codec != "aac":
            raise ValueError(f"Storyboard FFmpeg output must use AAC audio; got {audio_codec or 'unknown'}.")
        self._log(
            "Validated Storyboard FFmpeg output; "
            f"filename={path.name!r} dimensions={output_width}x{output_height} "
            f"duration_seconds={duration:.2f} bytes={file_size} video_codec={video_codec} audio_codec={audio_codec}."
        )
        return metadata

    @staticmethod
    def _storyboard_video_inputs(downloaded_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [item for item in downloaded_inputs if item.get("category") == "video"],
            key=lambda item: (int(item.get("order") or 0), int(item.get("input_id") or 0)),
        )

    @staticmethod
    def _storyboard_primary_video_input(video_inputs: list[dict[str, Any]]) -> dict[str, Any]:
        preferred_roles = {"primary", "source", "take", "card", "input"}
        for item in video_inputs:
            role = str(item.get("role") or "").strip().lower()
            kind = str(item.get("kind") or "").strip().lower()
            if role in preferred_roles or kind in {"source_video", "input_video", "take_video", "card_video", "video"}:
                return item
        return video_inputs[0]

    @staticmethod
    def _storyboard_overlay_input(downloaded_inputs: list[dict[str, Any]], processing: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        for item in downloaded_inputs:
            role = str(item.get("role") or "").strip().lower()
            kind = str(item.get("kind") or "").strip().lower()
            if kind == "overlay_video" or (role == "overlay" and item.get("category") == "video"):
                return item
        for item in downloaded_inputs:
            role = str(item.get("role") or "").strip().lower()
            kind = str(item.get("kind") or "").strip().lower()
            if AwsWorkerBridgePlugin._is_storyboard_matte_descriptor(item, processing):
                continue
            if role == "overlay" or kind in {"overlay_image", "overlay_png"}:
                return item
        return None

    @staticmethod
    def _storyboard_matte_input(downloaded_inputs: list[dict[str, Any]], processing: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        for item in downloaded_inputs:
            if item.get("category") == "image" and AwsWorkerBridgePlugin._is_storyboard_matte_descriptor(item, processing):
                return item
        return None

    @staticmethod
    def _is_storyboard_matte_descriptor(item: dict[str, Any], processing: Optional[dict[str, Any]] = None) -> bool:
        role = str(item.get("role") or "").strip().lower()
        kind = str(item.get("kind") or "").strip().lower()
        if role in {"matte", "mask"} or kind in {"matte_image", "mask_image", "overlay_matte", "overlay_mask"}:
            return True
        if not isinstance(processing, dict):
            return False
        selector_values = {
            processing.get("mask_dbfileid"),
            processing.get("mask_input_id"),
            processing.get("matte_dbfileid"),
            processing.get("matte_input_id"),
            processing.get("overlay_mask_dbfileid"),
            processing.get("overlay_matte_dbfileid"),
        }
        selector_ids = set()
        for value in selector_values:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                selector_ids.add(number)
        if not selector_ids:
            return False
        candidate_values = {
            item.get("input_id"),
            item.get("dbfileid"),
            item.get("dbfile_id"),
            item.get("file_id"),
        }
        for value in candidate_values:
            try:
                if int(value) in selector_ids:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _log_storyboard_overlay_matte_state(
        self,
        job_id: int,
        overlay_input: dict[str, Any],
        matte_input: Optional[dict[str, Any]],
    ) -> None:
        overlay_path = Path(str(overlay_input.get("path") or ""))
        if matte_input is None:
            self._log(
                "Storyboard overlay matte state; "
                f"job_id={job_id} matte_detected=False final_overlay_mode=no_matte "
                f"overlay_input_id={overlay_input.get('input_id')} overlay_filename={overlay_path.name!r}."
            )
            return
        matte_path = Path(str(matte_input.get("path") or ""))
        matte_width = self._coerce_int(matte_input.get("width"), 0, 0, 100_000)
        matte_height = self._coerce_int(matte_input.get("height"), 0, 0, 100_000)
        decoded_format = str(matte_input.get("decoded_format") or "").strip().upper()
        mime_type = str(matte_input.get("mime_type") or "").strip().lower()
        self._log(
            "Storyboard overlay matte state; "
            f"job_id={job_id} matte_detected=True matte_input_id={matte_input.get('input_id')} "
            f"matte_dbfileid={matte_input.get('dbfileid') or matte_input.get('dbfile_id') or ''} "
            f"matte_filename={matte_path.name!r} matte_mime={mime_type!r} "
            f"matte_dimensions={matte_width}x{matte_height} matte_format={decoded_format or 'unknown'} "
            "alphamerge_applied=True final_overlay_mode=luminance_matte."
        )

    def _storyboard_overlay_filter_parts(
        self,
        overlay_input: dict[str, Any],
        overlay_input_index: int,
        matte_input_index: Optional[int],
        output_width: int,
        output_height: int,
        processing: dict[str, Any],
        effective_duration: float,
    ) -> list[str]:
        variant = str(processing.get("overlay_variant") or "static_rectangle").strip().lower()
        if variant in {"animated_rectangle", "interpolated_rectangle"}:
            return self._storyboard_animated_overlay_filter_parts(
                overlay_input,
                overlay_input_index,
                matte_input_index,
                output_width,
                output_height,
                processing,
                effective_duration,
            )
        return self._storyboard_static_overlay_filter_parts(
            overlay_input,
            overlay_input_index,
            matte_input_index,
            output_width,
            output_height,
            processing,
        )

    def _storyboard_static_overlay_filter_parts(
        self,
        overlay_input: dict[str, Any],
        overlay_input_index: int,
        matte_input_index: Optional[int],
        output_width: int,
        output_height: int,
        processing: dict[str, Any],
    ) -> list[str]:
        overlay_x = self._coerce_int(self._first_present(processing, "overlay_x", "x", "left", "pip_x"), 0, -4096, 4096)
        overlay_y = self._coerce_int(self._first_present(processing, "overlay_y", "y", "top", "pip_y"), 0, -4096, 4096)
        overlay_width = self._coerce_int(self._first_present(processing, "overlay_width", "overlay_w", "pip_width", "pip_w"), 0, 0, 4096)
        overlay_height = self._coerce_int(self._first_present(processing, "overlay_height", "overlay_h", "pip_height", "pip_h"), 0, 0, 4096)
        if overlay_width <= 0 or overlay_height <= 0:
            preset = str(processing.get("overlay_preset") or "").strip().lower()
            scale = self._optional_positive_float(processing.get("overlay_scale"), 1.0)
            if preset and scale:
                rect = self._storyboard_overlay_rect(
                    output_width,
                    output_height,
                    overlay_input,
                    preset,
                    scale,
                    self._coerce_int(processing.get("margin_x"), 0, 0, 4096),
                    self._coerce_int(processing.get("margin_y"), 0, 0, 4096),
                )
                overlay_x, overlay_y, overlay_width, overlay_height = rect
        start_opacity = self._coerce_float(processing.get("start_opacity", processing.get("opacity")), 1.0, 0.0, 1.0)
        end_opacity = self._coerce_float(processing.get("end_opacity"), start_opacity, 0.0, 1.0)
        opacity = end_opacity
        parts = []
        overlay_chain = f"[{overlay_input_index}:v]"
        if overlay_width > 0 and overlay_height > 0:
            overlay_chain += f"scale={overlay_width}:{overlay_height},"
        overlay_chain += "format=rgb24[ovbase]" if matte_input_index is not None else "format=rgba[ovbase]"
        parts.append(overlay_chain)
        source_label = "ovbase"
        if matte_input_index is not None:
            if overlay_width > 0 and overlay_height > 0:
                parts.append(f"[{matte_input_index}:v]scale={overlay_width}:{overlay_height},format=rgb24,format=gray[ovmask]")
            else:
                parts.append(f"[{matte_input_index}:v]format=rgb24,format=gray[ovmask]")
            parts.append("[ovbase][ovmask]alphamerge,format=rgba[ovmatte]")
            source_label = "ovmatte"
        if opacity < 0.999:
            parts.append(
                f"[{source_label}]geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                f"a='alpha(X,Y)*{opacity:.6f}'[ov]"
            )
        else:
            parts.append(f"[{source_label}]copy[ov]")
        parts.append(f"[vbase][ov]overlay={overlay_x}:{overlay_y}:format=auto:eof_action=pass,format=yuv420p[vout]")
        return parts

    def _storyboard_animated_overlay_filter_parts(
        self,
        overlay_input: dict[str, Any],
        overlay_input_index: int,
        matte_input_index: Optional[int],
        output_width: int,
        output_height: int,
        processing: dict[str, Any],
        effective_duration: float,
    ) -> list[str]:
        start_preset = str(processing.get("overlay_preset") or "bottom_right").strip().lower()
        start_scale = self._coerce_float(processing.get("overlay_scale"), 0.28, 0.01, 4.0)
        start_margin_x = self._coerce_int(processing.get("margin_x"), 32, 0, 4096)
        start_margin_y = self._coerce_int(processing.get("margin_y"), 32, 0, 4096)
        end_preset = str(processing.get("end_overlay_preset") or processing.get("overlay_preset") or "bottom_right").strip().lower()
        end_scale = self._coerce_float(processing.get("end_overlay_scale"), start_scale, 0.01, 4.0)
        end_margin_x = self._coerce_int(processing.get("end_margin_x"), start_margin_x, 0, 4096)
        end_margin_y = self._coerce_int(processing.get("end_margin_y"), start_margin_y, 0, 4096)
        start_rect = self._storyboard_overlay_rect(
            output_width,
            output_height,
            overlay_input,
            start_preset,
            start_scale,
            start_margin_x,
            start_margin_y,
        )
        end_rect = self._storyboard_overlay_rect(
            output_width,
            output_height,
            overlay_input,
            end_preset,
            end_scale,
            end_margin_x,
            end_margin_y,
        )
        _sx, _sy, sw, sh = start_rect
        _ex, _ey, ew, eh = end_rect
        tile_width, tile_height = self._even_video_size(max(sw, ew), max(sh, eh))
        min_width_ratio = max(0.01, min(float(sw), float(ew)) / float(tile_width))
        min_height_ratio = max(0.01, min(float(sh), float(eh)) / float(tile_height))
        zoom_canvas_width, zoom_canvas_height = self._even_video_size(
            int(math.ceil(float(tile_width) / min_width_ratio)),
            int(math.ceil(float(tile_height) / min_height_ratio)),
        )
        if zoom_canvas_width > 8192 or zoom_canvas_height > 8192:
            raise ValueError(
                "Storyboard animated overlay scale range is too large for fixed-canvas rendering; "
                f"tile={tile_width}x{tile_height} zoom_canvas={zoom_canvas_width}x{zoom_canvas_height}."
            )
        progress_t = self._storyboard_overlay_progress_expr(effective_duration, str(processing.get("overlay_easing") or "ease_in_out"), "t")
        progress_t_upper = self._storyboard_overlay_progress_expr(effective_duration, str(processing.get("overlay_easing") or "ease_in_out"), "T")
        progress_on = self._storyboard_overlay_frame_progress_expr(
            effective_duration,
            self._coerce_int(processing.get("fps"), STORYBOARD_OUTPUT_FPS, 1, 120),
            str(processing.get("overlay_easing") or "ease_in_out"),
        )
        current_width_t = f"({sw}+({ew}-{sw})*({progress_t}))"
        current_height_t = f"({sh}+({eh}-{sh})*({progress_t}))"
        current_width_on = f"({sw}+({ew}-{sw})*({progress_on}))"
        current_height_on = f"({sh}+({eh}-{sh})*({progress_on}))"
        desired_x_expr = f"({start_rect[0]}+({end_rect[0]}-{start_rect[0]})*({progress_t}))"
        desired_y_expr = f"({start_rect[1]}+({end_rect[1]}-{start_rect[1]})*({progress_t}))"
        x_expr = f"({desired_x_expr}-(({tile_width})-({current_width_t}))/2)"
        y_expr = f"({desired_y_expr}-(({tile_height})-({current_height_t}))/2)"
        zoom_x_expr = (
            f"(({current_width_on})/{float(tile_width):.6f})/{min_width_ratio:.6f}"
        )
        zoom_y_expr = (
            f"(({current_height_on})/{float(tile_height):.6f})/{min_height_ratio:.6f}"
        )
        zoom_expr = f"min(({zoom_x_expr})\\,({zoom_y_expr}))"
        start_opacity = self._coerce_float(processing.get("start_opacity"), 1.0, 0.0, 1.0)
        end_opacity = self._coerce_float(processing.get("end_opacity"), start_opacity, 0.0, 1.0)
        opacity_expr = f"({start_opacity:.6f}+({end_opacity:.6f}-{start_opacity:.6f})*({progress_t_upper}))"
        parts = [
        ]
        if matte_input_index is not None:
            parts.extend([
                (
                    f"[{overlay_input_index}:v]setpts=PTS-STARTPTS,"
                    f"fps={self._coerce_int(processing.get('fps'), STORYBOARD_OUTPUT_FPS, 1, 120)},format=rgb24,"
                    f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
                    f"pad={zoom_canvas_width}:{zoom_canvas_height}:(ow-iw)/2:(oh-ih)/2:color=black[ovscaledrgb]"
                ),
                (
                    f"[{matte_input_index}:v]format=rgb24,"
                    f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
                    f"pad={zoom_canvas_width}:{zoom_canvas_height}:(ow-iw)/2:(oh-ih)/2:color=black,format=gray[ovmask]"
                ),
                "[ovscaledrgb][ovmask]alphamerge,format=rgba[ovmatte]",
            ])
            source_label = "ovmatte"
        else:
            parts.append(
                f"[{overlay_input_index}:v]setpts=PTS-STARTPTS,"
                f"fps={self._coerce_int(processing.get('fps'), STORYBOARD_OUTPUT_FPS, 1, 120)},format=rgba,"
                f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
                f"pad={zoom_canvas_width}:{zoom_canvas_height}:(ow-iw)/2:(oh-ih)/2:color=black@0[ovscaled]"
            )
            source_label = "ovscaled"
        parts.extend([
            (
                f"[{source_label}]zoompan=z='{zoom_expr}':"
                "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d=1:s={tile_width}x{tile_height}:fps={self._coerce_int(processing.get('fps'), STORYBOARD_OUTPUT_FPS, 1, 120)},"
                f"setpts=N/({self._coerce_int(processing.get('fps'), STORYBOARD_OUTPUT_FPS, 1, 120)}*TB),format=rgba[ovtile]"
            ),
            (
                "[ovtile]geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                f"a='alpha(X,Y)*({opacity_expr})'[ov]"
            ),
            (
                f"[vbase][ov]overlay=x='{x_expr}':y='{y_expr}':"
                "format=auto:eval=frame:eof_action=pass,format=yuv420p[vout]"
            ),
        ])
        return parts

    @staticmethod
    def _storyboard_overlay_tile_position(
        canvas_width: int,
        canvas_height: int,
        tile_width: int,
        tile_height: int,
        preset: str,
        margin_x: int,
        margin_y: int,
    ) -> tuple[int, int]:
        preset = preset if preset in {"bottom_right", "bottom_left", "top_right", "top_left", "center"} else "bottom_right"
        if preset == "bottom_right":
            x = canvas_width - tile_width - int(margin_x)
            y = canvas_height - tile_height - int(margin_y)
        elif preset == "bottom_left":
            x = int(margin_x)
            y = canvas_height - tile_height - int(margin_y)
        elif preset == "top_right":
            x = canvas_width - tile_width - int(margin_x)
            y = int(margin_y)
        elif preset == "top_left":
            x = int(margin_x)
            y = int(margin_y)
        else:
            x = (canvas_width - tile_width) // 2
            y = (canvas_height - tile_height) // 2
        return max(0, min(canvas_width - tile_width, x)), max(0, min(canvas_height - tile_height, y))

    def _storyboard_overlay_rect(
        self,
        canvas_width: int,
        canvas_height: int,
        overlay_input: dict[str, Any],
        preset: str,
        scale: float,
        margin_x: int,
        margin_y: int,
    ) -> tuple[int, int, int, int]:
        source_width, source_height = self._storyboard_input_display_size(overlay_input)
        aspect = max(0.01, float(source_width) / float(source_height or 1))
        rect_width = max(2, int(round(float(canvas_width) * float(scale))))
        rect_height = max(2, int(round(rect_width / aspect)))
        if rect_height > canvas_height:
            rect_height = max(2, int(round(float(canvas_height) * float(scale))))
            rect_width = max(2, int(round(rect_height * aspect)))
        rect_width, rect_height = self._even_video_size(min(rect_width, canvas_width), min(rect_height, canvas_height))
        preset = preset if preset in {"bottom_right", "bottom_left", "top_right", "top_left", "center"} else "bottom_right"
        if preset == "bottom_right":
            x = canvas_width - rect_width - int(margin_x)
            y = canvas_height - rect_height - int(margin_y)
        elif preset == "bottom_left":
            x = int(margin_x)
            y = canvas_height - rect_height - int(margin_y)
        elif preset == "top_right":
            x = canvas_width - rect_width - int(margin_x)
            y = int(margin_y)
        elif preset == "top_left":
            x = int(margin_x)
            y = int(margin_y)
        else:
            x = (canvas_width - rect_width) // 2
            y = (canvas_height - rect_height) // 2
        x = max(0, min(canvas_width - rect_width, x))
        y = max(0, min(canvas_height - rect_height, y))
        return x, y, rect_width, rect_height

    @staticmethod
    def _storyboard_input_display_size(item: dict[str, Any]) -> tuple[int, int]:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        width = int(metadata.get("display_width") or metadata.get("width") or item.get("width") or 0)
        height = int(metadata.get("display_height") or metadata.get("height") or item.get("height") or 0)
        return max(2, width or 2), max(2, height or 2)

    @staticmethod
    def _storyboard_overlay_progress_expr(duration_seconds: float, easing: str, time_var: str) -> str:
        duration = max(0.001, float(duration_seconds or 0.001))
        p = f"if(gte({time_var}\\,{duration:.6f})\\,1\\,if(lte({time_var}\\,0)\\,0\\,{time_var}/{duration:.6f}))"
        if str(easing or "").strip().lower() == "linear":
            return p
        return f"(({p})*({p})*(3-2*({p})))"

    @staticmethod
    def _storyboard_overlay_frame_progress_expr(duration_seconds: float, fps: int, easing: str) -> str:
        frame_count = max(1.0, float(duration_seconds or 0.001) * float(max(1, int(fps or STORYBOARD_OUTPUT_FPS))))
        p = f"if(gte(on\\,{frame_count:.6f})\\,1\\,if(lte(on\\,0)\\,0\\,on/{frame_count:.6f}))"
        if str(easing or "").strip().lower() == "linear":
            return p
        return f"(({p})*({p})*(3-2*({p})))"

    @staticmethod
    def _storyboard_audio_input(downloaded_inputs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        preferred_kinds = ("soundtrack_audio", "source_audio")
        for preferred_kind in preferred_kinds:
            for item in downloaded_inputs:
                if item.get("category") == "audio" and str(item.get("kind") or "").strip().lower() == preferred_kind:
                    return item
        for item in downloaded_inputs:
            role = str(item.get("role") or "").strip().lower()
            if item.get("category") == "audio" and role in {"soundtrack", "source", "audio", "primary"}:
                return item
        return None

    def _storyboard_external_audio_filter_parts(
        self,
        audio_input_label: str,
        processing: dict[str, Any],
        effective_duration: float,
    ) -> list[str]:
        soundtrack_start = self._coerce_float(
            self._first_present(processing, "soundtrack_start", "soundtrack_start_seconds", "audio_start_seconds", "audio_offset_seconds"),
            0.0,
            0.0,
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        head_silence = self._coerce_float(
            self._first_present(processing, "soundtrack_head_silence", "soundtrack_head_silence_seconds", "audio_head_silence_seconds"),
            0.0,
            0.0,
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        tail_silence = self._coerce_float(
            self._first_present(processing, "soundtrack_tail_silence", "soundtrack_tail_silence_seconds", "audio_tail_silence_seconds"),
            0.0,
            0.0,
            MAX_STORYBOARD_VIDEO_DURATION_SECONDS,
        )
        total_duration = max(0.1, float(effective_duration or 0.1))
        head_silence = min(head_silence, total_duration)
        tail_silence = min(tail_silence, max(0.0, total_duration - head_silence))
        body_duration = max(0.0, total_duration - head_silence - tail_silence)
        if body_duration <= 0.001:
            return [
                (
                    "anullsrc=channel_layout=stereo:sample_rate=48000,"
                    f"atrim=duration={total_duration:.3f},asetpts=PTS-STARTPTS,"
                    "aformat=channel_layouts=stereo[aout]"
                )
            ]
        if head_silence <= 0.001 and tail_silence <= 0.001:
            return [
                (
                    f"[{audio_input_label}]atrim=start={soundtrack_start:.3f}:duration={body_duration:.3f},"
                    "asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[aout]"
                )
            ]
        parts = []
        concat_labels = []
        if head_silence > 0.001:
            parts.append(
                "anullsrc=channel_layout=stereo:sample_rate=48000,"
                f"atrim=duration={head_silence:.3f},asetpts=PTS-STARTPTS,"
                "aformat=channel_layouts=stereo[ahead]"
            )
            concat_labels.append("[ahead]")
        parts.append(
            f"[{audio_input_label}]atrim=start={soundtrack_start:.3f}:duration={body_duration:.3f},"
            "asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[amain]"
        )
        concat_labels.append("[amain]")
        if tail_silence > 0.001:
            parts.append(
                "anullsrc=channel_layout=stereo:sample_rate=48000,"
                f"atrim=duration={tail_silence:.3f},asetpts=PTS-STARTPTS,"
                "aformat=channel_layouts=stereo[atail]"
            )
            concat_labels.append("[atail]")
        parts.append(f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=0:a=1,aformat=channel_layouts=stereo[aout]")
        return parts

    def _storyboard_output_size(self, video_input: dict[str, Any], processing: dict[str, Any]) -> tuple[int, int]:
        metadata = video_input.get("metadata") if isinstance(video_input.get("metadata"), dict) else {}
        width = self._coerce_int(processing.get("output_width") or processing.get("width"), 0, 0, 4096)
        height = self._coerce_int(processing.get("output_height") or processing.get("height"), 0, 0, 4096)
        if width <= 0 or height <= 0:
            width = self._coerce_int(metadata.get("display_width") or metadata.get("width"), 1280, 2, 4096)
            height = self._coerce_int(metadata.get("display_height") or metadata.get("height"), 720, 2, 4096)
        return self._even_video_size(width, height)

    @staticmethod
    def _even_video_size(width: int, height: int) -> tuple[int, int]:
        even_width = max(2, int(width) - (int(width) % 2))
        even_height = max(2, int(height) - (int(height) % 2))
        return even_width, even_height

    @staticmethod
    def _first_present(payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload and payload.get(key) is not None:
                return payload.get(key)
        return None

    def _storyboard_trim_duration(self, source_duration: float, processing: dict[str, Any], trim_start: float) -> Optional[float]:
        trim_duration = self._optional_positive_float(processing.get("trim_duration_seconds"), MAX_STORYBOARD_VIDEO_DURATION_SECONDS)
        if trim_duration is not None:
            return min(trim_duration, max(0.1, float(source_duration or 0.0) - trim_start))
        trim_end = self._optional_positive_float(processing.get("trim_end_seconds"), MAX_STORYBOARD_VIDEO_DURATION_SECONDS)
        if trim_end is not None and trim_end > trim_start:
            return min(trim_end - trim_start, max(0.1, float(source_duration or 0.0) - trim_start))
        if trim_start > 0:
            return max(0.1, float(source_duration or 0.0) - trim_start)
        return None

    @staticmethod
    def _storyboard_step_timeout(duration_seconds: float) -> int:
        try:
            duration = max(1.0, float(duration_seconds or 1.0))
        except (TypeError, ValueError):
            duration = 1.0
        return int(min(STORYBOARD_TIMEOUT_MAX_SECONDS, max(STORYBOARD_TIMEOUT_BASE_SECONDS, STORYBOARD_TIMEOUT_BASE_SECONDS + duration * STORYBOARD_TIMEOUT_MULTIPLIER)))

    def _event_video_output_size(self, metadata: dict[str, Any], processing: dict[str, Any]) -> tuple[int, int]:
        target_a = self._coerce_int(processing.get("target_max_width"), 720, 1, 4096)
        target_b = self._coerce_int(processing.get("target_max_height"), 1280, 1, 4096)
        short_edge = min(target_a, target_b)
        long_edge = max(target_a, target_b)
        orientation = str(metadata.get("orientation") or "landscape").strip().lower()
        if orientation == "portrait":
            return short_edge, long_edge
        return long_edge, short_edge

    def _event_processing_asset_for_orientation(
        self,
        downloaded_inputs: list[dict[str, Any]],
        kind: str,
        orientation: str,
        required: bool,
    ) -> Optional[Path]:
        matches = [
            Path(str(item.get("path")))
            for item in downloaded_inputs
            if item.get("kind") == kind and str(item.get("orientation") or "").strip().lower() == orientation
        ]
        if len(matches) > 1:
            raise ValueError(f"Event Video Processing received multiple {kind} assets for {orientation} output.")
        if matches:
            return matches[0]
        if required:
            label = "overlay" if kind == "overlay_png" else "bumper"
            raise ValueError(
                f"Event video {label} is enabled, but no {orientation} {label} asset was provided for {orientation} output."
            )
        return None

    def _run_event_video_main_ffmpeg(
        self,
        connection: ConnectionContext,
        job_id: int,
        source_path: Path,
        overlay_path: Optional[Path],
        output_path: Path,
        metadata: dict[str, Any],
        output_width: int,
        output_height: int,
        process_handle: LocalProcessJob,
    ) -> None:
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-v",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-i",
            str(source_path),
        ]
        next_input_index = 1
        audio_input_label = "0:a:0"
        if not metadata.get("has_audio"):
            command.extend([
                "-f",
                "lavfi",
                "-t",
                f"{float(metadata.get('duration_seconds') or 1.0):.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ])
            audio_input_label = f"{next_input_index}:a:0"
            next_input_index += 1
        overlay_input_index = None
        if overlay_path is not None:
            command.extend(["-i", str(overlay_path)])
            overlay_input_index = next_input_index
            next_input_index += 1
        filter_parts = [
            (
                f"[0:v]scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
                f"crop={output_width}:{output_height},"
                "setsar=1,format=rgba[vbase]"
            ),
            f"[{audio_input_label}]aresample=48000,aformat=channel_layouts=stereo[aout]",
        ]
        if overlay_input_index is not None:
            filter_parts.extend([
                f"[{overlay_input_index}:v]scale={output_width}:{output_height},format=rgba[ov]",
                "[vbase][ov]overlay=0:0:format=auto,format=yuv420p[vout]",
            ])
        else:
            filter_parts.append("[vbase]format=yuv420p[vout]")
        command.extend([
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            *self._event_video_encode_args(),
            str(output_path),
        ])
        self._run_ffmpeg_with_progress(
            connection,
            job_id,
            command,
            total_seconds=float(metadata.get("duration_seconds") or 1.0),
            progress_start=5,
            progress_end=75,
            phase="processing",
            status="Normalizing and branding Event video.",
            process_handle=process_handle,
            timeout_seconds=self._event_video_step_timeout(float(metadata.get("duration_seconds") or 1.0)),
        )

    def _run_event_bumper_ffmpeg(
        self,
        connection: ConnectionContext,
        job_id: int,
        bumper_path: Path,
        output_path: Path,
        output_width: int,
        output_height: int,
        process_handle: LocalProcessJob,
    ) -> None:
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-v",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-loop",
            "1",
            "-t",
            str(EVENT_VIDEO_BUMPER_SECONDS),
            "-i",
            str(bumper_path),
            "-f",
            "lavfi",
            "-t",
            str(EVENT_VIDEO_BUMPER_SECONDS),
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-filter_complex",
            (
                f"[0:v]scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,"
                f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                "setsar=1,format=yuv420p[vout];"
                "[1:a:0]aresample=48000,aformat=channel_layouts=stereo[aout]"
            ),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            *self._event_video_encode_args(),
            str(output_path),
        ]
        self._run_ffmpeg_with_progress(
            connection,
            job_id,
            command,
            total_seconds=float(EVENT_VIDEO_BUMPER_SECONDS),
            progress_start=76,
            progress_end=85,
            phase="processing",
            status="Rendering Event video ending bumper.",
            process_handle=process_handle,
            timeout_seconds=self._event_video_step_timeout(float(EVENT_VIDEO_BUMPER_SECONDS)),
        )

    def _concat_event_video_segments(
        self,
        connection: ConnectionContext,
        job_id: int,
        segment_paths: list[Path],
        output_path: Path,
        total_seconds: float,
        process_handle: LocalProcessJob,
        *,
        progress_start: int = 86,
        progress_end: int = 95,
        status: str = "Appending Event video ending bumper.",
    ) -> None:
        list_path = output_path.with_suffix(".txt")
        with list_path.open("w", encoding="utf-8") as writer:
            for path in segment_paths:
                escaped = str(path).replace("'", "'\\''")
                writer.write(f"file '{escaped}'\n")
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-v",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self._run_ffmpeg_with_progress(
            connection,
            job_id,
            command,
            total_seconds=max(1.0, float(total_seconds or 1.0)),
            progress_start=progress_start,
            progress_end=progress_end,
            phase="processing",
            status=status,
            process_handle=process_handle,
            timeout_seconds=self._event_video_step_timeout(max(1.0, float(total_seconds or 1.0))),
        )

    @staticmethod
    def _event_video_step_timeout(duration_seconds: float) -> int:
        try:
            duration = max(1.0, float(duration_seconds or 1.0))
        except (TypeError, ValueError):
            duration = 1.0
        return int(min(EVENT_VIDEO_TIMEOUT_MAX_SECONDS, max(EVENT_VIDEO_TIMEOUT_BASE_SECONDS, EVENT_VIDEO_TIMEOUT_BASE_SECONDS + duration * EVENT_VIDEO_TIMEOUT_MULTIPLIER)))

    @staticmethod
    def _event_video_encode_args() -> list[str]:
        return [
            "-c:v",
            EVENT_VIDEO_H264_ENCODER,
            "-preset",
            "veryfast",
            "-crf",
            str(EVENT_VIDEO_X264_CRF),
            "-maxrate",
            EVENT_VIDEO_X264_MAXRATE,
            "-bufsize",
            EVENT_VIDEO_X264_BUFSIZE,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-metadata:s:v:0",
            "rotate=0",
        ]

    def _storyboard_encode_args(self, processing: dict[str, Any]) -> list[str]:
        preset = str(processing.get("preset") or "veryfast").strip().lower()
        if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}:
            preset = "veryfast"
        tune = str(processing.get("tune") or "none").strip().lower()
        if tune not in {"none", "film", "animation", "grain", "stillimage", "fastdecode", "zerolatency"}:
            tune = "none"
        crf = self._coerce_int(processing.get("crf"), EVENT_VIDEO_X264_CRF, 0, 51)
        audio_bitrate = str(processing.get("audio_bitrate") or "128k").strip().lower()
        if not re.fullmatch(r"[1-9][0-9]{1,3}k", audio_bitrate):
            audio_bitrate = "128k"
        args = [
            "-c:v",
            EVENT_VIDEO_H264_ENCODER,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            "-metadata:s:v:0",
            "rotate=0",
        ]
        if tune != "none":
            args[6:6] = ["-tune", tune]
        return args

    def _run_ffmpeg_with_progress(
        self,
        connection: ConnectionContext,
        job_id: int,
        command: list[str],
        *,
        total_seconds: float,
        progress_start: int,
        progress_end: int,
        phase: str,
        status: str,
        process_handle: LocalProcessJob,
        timeout_seconds: int,
    ) -> None:
        self._log(
            "Starting FFmpeg subprocess; "
            f"job_id={job_id} phase={phase!r} progress_range={progress_start}-{progress_end} "
            f"timeout_seconds={timeout_seconds} "
            f"command={self._redacted_command_for_log(command)}."
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        process_handle.set_process(process)
        last_progress_at = 0.0
        output_lines: deque[str] = deque(maxlen=40)
        output_queue: queue.Queue = queue.Queue()

        def _reader() -> None:
            try:
                if process.stdout is not None:
                    for stream_line in process.stdout:
                        output_queue.put(stream_line)
            finally:
                output_queue.put(None)

        reader_thread = threading.Thread(target=_reader, name=f"{PLUGIN_ID}-ffmpeg-reader", daemon=True)
        reader_thread.start()
        deadline = time.monotonic() + max(1, int(timeout_seconds or EVENT_VIDEO_TIMEOUT_BASE_SECONDS))
        try:
            while True:
                if self._stop_event.is_set() or self._cancel_requested_by_midom or process_handle.cancelled:
                    process_handle.cancel()
                    raise RuntimeError("cancel_requested")
                now = time.monotonic()
                if now >= deadline:
                    self._terminate_ffmpeg_process(process)
                    raise TimeoutError(f"Event Video Processing FFmpeg step timed out after {timeout_seconds} seconds.")
                try:
                    line = output_queue.get(timeout=0.2)
                except queue.Empty:
                    if process.poll() is not None:
                        break
                    continue
                if line is None:
                    if process.poll() is not None:
                        break
                    continue
                if line:
                    progress_value = self._ffmpeg_progress_from_line(line, total_seconds, progress_start, progress_end)
                    now = time.monotonic()
                    if progress_value is not None and now - last_progress_at >= 2.0:
                        last_progress_at = now
                        self._set_active_job_status(phase=phase, status=status, progress=progress_value)
                        try:
                            self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
                        except Exception as exc:
                            self._log(f"FFmpeg progress update failed: {exc}", force=True)
                            if self._is_auth_error(exc):
                                process_handle.cancel()
                                raise
                    elif progress_value is None and line.strip():
                        output_lines.append(line.strip())
                    continue
            if process.returncode != 0:
                message = "\n".join(output_lines).strip() or f"ffmpeg exited with status {process.returncode}"
                raise ValueError(f"Event Video Processing FFmpeg step failed: {message[:1000]}")
        except RuntimeError:
            self._terminate_ffmpeg_process(process)
            raise
        finally:
            process_handle.clear_process(process)
        self._set_active_job_status(phase=phase, status=status, progress=progress_end)
        self._post_job_update(connection, job_id, "progress", dict(self._active_job_status))
        self._log(
            "FFmpeg subprocess finished; "
            f"job_id={job_id} phase={phase!r} returncode={process.returncode} progress={progress_end}."
        )

    @staticmethod
    def _ffmpeg_progress_from_line(line: str, total_seconds: float, progress_start: int, progress_end: int) -> Optional[int]:
        text = str(line or "").strip()
        if not text.startswith("out_time_"):
            return None
        value_text = text.split("=", 1)[1].strip() if "=" in text else ""
        seconds = 0.0
        if text.startswith("out_time_ms=") or text.startswith("out_time_us="):
            try:
                seconds = float(value_text) / 1_000_000.0
            except (TypeError, ValueError):
                seconds = 0.0
        elif text.startswith("out_time="):
            parts = value_text.split(":")
            if len(parts) == 3:
                try:
                    seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                except (TypeError, ValueError):
                    seconds = 0.0
        if seconds <= 0 or total_seconds <= 0:
            return None
        span = max(0, progress_end - progress_start)
        return max(progress_start, min(progress_end, int(progress_start + span * min(1.0, seconds / total_seconds))))

    @staticmethod
    def _terminate_ffmpeg_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    @staticmethod
    def _redacted_command_for_log(command: list[str]) -> str:
        parts = []
        for item in command:
            text = str(item)
            if len(text) > 180:
                text = f"{text[:177]}..."
            parts.append(text)
        return " ".join(parts)

    def _submit_wangp_job(self, api_session, settings: dict[str, Any], output_count: int, callbacks):
        if str(settings.get("_midom_media_type") or "") == "audio":
            self._log(
                "WanGP audio submit settings; "
                f"model_type={settings.get('model_type')} "
                f"voice_mode={settings.get('_midom_voice_mode')!r} "
                f"audio_prompt_type={settings.get('audio_prompt_type')!r} "
                f"duration_seconds={settings.get('duration_seconds')} "
                f"pause_seconds={settings.get('pause_seconds', 0)} "
                f"prompt_processing_mode={settings.get('_midom_prompt_processing_mode')!r} "
                f"model_mode={settings.get('model_mode')!r} "
                f"custom_settings={settings.get('custom_settings') if settings.get('model_type') in (CHATTERBOX_MODEL_ID, DRAMABOX_MODEL_ID, ACE_STEP15_WANGP_MODEL_TYPE) else None!r} "
                f"output_format={settings.get('_midom_output_format')} "
                f"{self._audio_prompt_summary(settings.get('prompt'))}."
            )
        elif str(settings.get("_midom_media_type") or "") == "video":
            self._log(
                "WanGP video submit settings; "
                f"model_type={settings.get('model_type')} "
                f"video_task={settings.get('_midom_video_task')!r} "
                f"duration_seconds={settings.get('duration_seconds')} "
                f"video_length={settings.get('video_length')} "
                f"resolution={settings.get('resolution')} "
                f"steps={settings.get('num_inference_steps')} "
                f"sample_solver={settings.get('sample_solver')!r} "
                f"speed_profile_id={settings.get('_midom_speed_profile_id')!r} "
                f"audio_video_mode={settings.get('_midom_audio_video_mode')!r} "
                f"video_prompt_type={settings.get('video_prompt_type')!r} "
                f"audio_prompt_type={settings.get('audio_prompt_type')!r} "
                f"video_sync_profile_id={settings.get('_midom_video_sync_profile_id', 'standard')!r} "
                f"output_format={settings.get('_midom_output_format')}."
            )
        model_type = str(settings.get("model_type") or "")
        layer_output_count = self._coerce_int(
            settings.get("_midom_layer_output_count"),
            output_count,
            1,
            _image_max_outputs_for_model(model_type),
        )
        media_type = str(settings.get("_midom_media_type") or "")
        prompt_processing_mode = str(settings.get("_midom_prompt_processing_mode") or DEFAULT_PROMPT_PROCESSING_MODE).strip().upper()
        if media_type == "audio" and str(settings.get("_midom_audio_task") or "") == "voice_conversion":
            audio_source = settings.get("audio_source")
            reference_audio = settings.get("replace_voice_sample")
            postprocess_audio = str(settings.get("postprocess_audio") or "").strip()
            if not audio_source or not reference_audio:
                raise ValueError("SeedVC voice conversion requires audio_source and replace_voice_sample paths.")
            if postprocess_audio != SEEDVC_METHOD_ONE_SPEAKER:
                raise ValueError(f"Unsupported SeedVC postprocess_audio method: {postprocess_audio}")
            self._log(
                "Using WanGP submit_audio_postprocessing for SeedVC voice replacement; "
                f"source_audio={Path(str(audio_source)).name!r} "
                f"reference_audio={Path(str(reference_audio)).name!r} "
                f"postprocess_audio={postprocess_audio!r}."
            )
            return api_session.submit_audio_postprocessing(
                audio_source,
                callbacks=callbacks,
                postprocess_audio=postprocess_audio,
                replace_voice_sample=reference_audio,
                replace_voice_sample2=None,
            )
        settings = {key: value for key, value in settings.items() if not str(key).startswith("_midom_")}
        if media_type == "audio" and output_count == 1 and prompt_processing_mode in {"G", "PG"}:
            prompt_chunks = self._prompt_chunks_for_mode(str(settings.get("prompt") or ""), prompt_processing_mode)
            if len(prompt_chunks) > 1:
                tasks = []
                for index, prompt_chunk in enumerate(prompt_chunks, start=1):
                    task = dict(settings)
                    task["prompt"] = prompt_chunk
                    task["multi_prompts_gen_type"] = DEFAULT_PROMPT_PROCESSING_MODE
                    tasks.append(task)
                self._log(
                    "Using WanGP submit_manifest for split audio prompt chunks; "
                    f"mode={prompt_processing_mode!r} chunks={len(tasks)}."
                )
                return api_session.submit_manifest(tasks, callbacks=callbacks)
        if model_type == "qwen_image_layered_20B":
            settings["batch_size"] = layer_output_count
            settings["repeat_generation"] = 1
            self._log(
                "Using WanGP submit_task for Qwen Image Layered decomposition; "
                f"batch_size={settings['batch_size']} output_count={output_count}."
            )
            return api_session.submit_task(settings, callbacks=callbacks)
        if output_count <= 1:
            self._log("Using WanGP submit_task for single output.")
            return api_session.submit_task(settings, callbacks=callbacks)
        tasks = []
        base_seed = settings.get("seed")
        for index in range(output_count):
            task = dict(settings)
            if isinstance(base_seed, int):
                task["seed"] = min(2_147_483_647, base_seed + index)
            tasks.append(task)
        self._log(f"Using WanGP submit_manifest for {len(tasks)} outputs.")
        return api_session.submit_manifest(tasks, callbacks=callbacks)

    def _wait_for_wangp_result(self, connection: ConnectionContext, job_id: int, job_handle, callbacks):
        next_keepalive_at = time.monotonic() + JOB_KEEPALIVE_SECONDS
        next_wait_log_at = time.monotonic() + JOB_WAIT_LOG_SECONDS
        cancel_requested_locally = False
        self._log(f"Waiting for WanGP result; keepalive_interval={JOB_KEEPALIVE_SECONDS}s; job_id={job_id}.")
        while not getattr(job_handle, "done", True):
            if self._stop_event.is_set():
                try:
                    job_handle.cancel()
                    self._log(f"Stop event set while waiting; local WanGP cancel requested; job_id={job_id}.")
                except Exception:
                    pass
            now = time.monotonic()
            if now >= next_keepalive_at:
                progress_value = self._callback_progress_value(callbacks)
                status_text = self._active_status_text("WanGP generation is still running.")
                phase_text = self._active_phase_text("running")
                try:
                    self._log(f"Sending progress keepalive; job_id={job_id} progress={progress_value} status={status_text!r}.")
                    self._post_job_update(
                        connection,
                        job_id,
                        "progress",
                        {
                            "phase": phase_text,
                            "status": status_text,
                            "progress": progress_value,
                        },
                    )
                except Exception as exc:
                    self._log(f"Job keepalive failed: {exc}", force=True)
                    if self._is_auth_error(exc):
                        self._handle_authorization_failure("Progress keepalive was rejected by Midom authorization.")
                        try:
                            job_handle.cancel()
                        except Exception:
                            pass
                        raise
                    next_keepalive_at = now + JOB_KEEPALIVE_RETRY_SECONDS
                    self._stop_event.wait(2.0)
                    continue
                try:
                    self._heartbeat_once(connection)
                except Exception as exc:
                    self._log(f"Busy heartbeat failed: {exc}", force=True)
                    if self._is_auth_error(exc):
                        self._handle_authorization_failure("Busy heartbeat was rejected by Midom authorization.")
                        try:
                            job_handle.cancel()
                        except Exception:
                            pass
                        raise
                next_keepalive_at = now + JOB_KEEPALIVE_SECONDS
            if now >= next_wait_log_at:
                progress_value = self._callback_progress_value(callbacks)
                status_snapshot = dict(self._active_job_status or {})
                status_text = self._active_status_text("WanGP generation is still running.")
                phase_text = self._active_phase_text("running")
                self._log(
                    "Waiting for WanGP generation; "
                    f"job_id={job_id} phase={phase_text!r} progress={progress_value} "
                    f"step={status_snapshot.get('current_step')}/{status_snapshot.get('total_steps')} "
                    f"status={status_text!r}."
                )
                next_wait_log_at = now + JOB_WAIT_LOG_SECONDS
            if self._cancel_requested_by_midom:
                try:
                    job_handle.cancel()
                    if not cancel_requested_locally:
                        cancel_requested_locally = True
                        self._set_active_job_status(
                            phase="canceling",
                            status="Midom cancellation received; waiting for WanGP to stop the local generation.",
                            progress=self._callback_progress_value(callbacks),
                        )
                        self._log(
                            "Midom cancellation flag set; local WanGP cancel requested; "
                            f"job_id={job_id}. Waiting for WanGP job handle to finish."
                        )
                except Exception:
                    pass
            self._stop_event.wait(2.0)
        return job_handle.result()

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        return int(getattr(response, "status_code", 0) or 0) in {401, 403}

    def _handle_authorization_failure(self, message: str) -> None:
        self._log(f"{message} Stopping worker and cancelling local WanGP job.", force=True)
        self._stop_event.set()
        job = self._active_job
        if job is not None and not getattr(job, "done", True):
            try:
                job.cancel()
            except Exception:
                pass

    @staticmethod
    def _callback_progress_value(callbacks) -> int:
        getter = getattr(callbacks, "progress_value", None)
        if callable(getter):
            return int(getter())
        return 0

    def _set_active_job_status(
        self,
        *,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        current_step: Any = None,
        total_steps: Any = None,
    ) -> None:
        snapshot = dict(self._active_job_status or {})
        if phase is not None:
            snapshot["phase"] = str(phase or "")
        if status is not None:
            snapshot["status"] = str(status or "")
        if progress is not None:
            try:
                snapshot["progress"] = max(0, min(100, int(progress)))
            except (TypeError, ValueError):
                pass
        if current_step is not None:
            snapshot["current_step"] = current_step
        if total_steps is not None:
            snapshot["total_steps"] = total_steps
        self._active_job_status = snapshot

    def _active_status_text(self, default: str) -> str:
        return str((self._active_job_status or {}).get("status") or default).strip()[:500]

    def _active_phase_text(self, default: str) -> str:
        return str((self._active_job_status or {}).get("phase") or default).strip()[:80]

    def _result_image_files(self, result) -> list[str]:
        files = []
        for file_path in list(getattr(result, "generated_files", []) or []):
            path = Path(str(file_path))
            if path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
                files.append(str(path))
            else:
                self._log(f"Ignoring non-image WanGP output file; path={path}.")
        for artifact in tuple(getattr(result, "artifacts", ()) or ()):
            file_path = getattr(artifact, "path", None)
            media_type = str(getattr(artifact, "media_type", "") or "")
            if file_path and media_type.startswith("image"):
                path = Path(str(file_path))
                if str(path) not in files and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
                    files.append(str(path))
            elif file_path:
                self._log(f"Ignoring non-image WanGP artifact; media_type={media_type!r} path={file_path}.")
        self._log(f"Resolved generated image files from WanGP result; count={len(files)}.")
        return files

    def _result_audio_files(self, result) -> list[str]:
        files = []
        for file_path in list(getattr(result, "generated_files", []) or []):
            path = Path(str(file_path))
            if path.suffix.lower() in ALLOWED_AUDIO_OUTPUT_SUFFIXES:
                files.append(str(path))
                self._log(
                    f"Accepted generated audio output file; filename={path.name!r} "
                    f"suffix={path.suffix.lower()!r} bytes={path.stat().st_size if path.is_file() else 'missing'}."
                )
            else:
                self._log(f"Ignoring non-audio WanGP output file; path={path}.")
        for artifact in tuple(getattr(result, "artifacts", ()) or ()):
            file_path = getattr(artifact, "path", None)
            media_type = str(getattr(artifact, "media_type", "") or "")
            if file_path and media_type.startswith("audio"):
                path = Path(str(file_path))
                if str(path) not in files and path.suffix.lower() in ALLOWED_AUDIO_OUTPUT_SUFFIXES:
                    files.append(str(path))
                    self._log(
                        f"Accepted generated audio artifact; media_type={media_type!r} filename={path.name!r} "
                        f"suffix={path.suffix.lower()!r} bytes={path.stat().st_size if path.is_file() else 'missing'}."
                    )
            elif file_path:
                self._log(f"Ignoring non-audio WanGP artifact; media_type={media_type!r} path={file_path}.")
        self._log(f"Resolved generated audio files from WanGP result; count={len(files)}.")
        return files

    def _combine_audio_segments(self, file_paths: list[str], temp_dir: str) -> str:
        if len(file_paths) <= 1:
            return str(file_paths[0]) if file_paths else ""
        paths = [Path(str(file_path)) for file_path in file_paths]
        for path in paths:
            if not path.is_file():
                raise ValueError(f"Generated audio segment path does not exist: {path}")
            if path.suffix.lower() not in ALLOWED_AUDIO_OUTPUT_SUFFIXES:
                raise ValueError(f"Unsupported generated audio segment extension: {path.suffix.lower()}")
        try:
            from shared.utils.audio_video import _ffmpeg_binary

            binary = _ffmpeg_binary()
        except Exception as exc:
            raise ValueError(f"Could not resolve WanGP ffmpeg binary for audio concatenation: {exc}")
        target_path = Path(temp_dir) / "midom-combined-audio.wav"
        command = [binary, "-y", "-v", "error"]
        filter_steps = []
        concat_inputs = []
        for index, path in enumerate(paths):
            command.extend(["-i", str(path)])
            filter_steps.append(
                f"[{index}:a]aresample=44100,aformat=sample_fmts=s16:channel_layouts=stereo[a{index}]"
            )
            concat_inputs.append(f"[a{index}]")
        filter_steps.append(f"{''.join(concat_inputs)}concat=n={len(paths)}:v=0:a=1[outa]")
        command.extend([
            "-filter_complex",
            ";".join(filter_steps),
            "-map",
            "[outa]",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(target_path),
        ])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = str(completed.stderr or completed.stdout or "").strip()
            raise ValueError(f"Audio segment concatenation failed: {stderr}")
        if not target_path.is_file() or target_path.stat().st_size <= 0:
            raise ValueError("Audio segment concatenation did not produce an output file.")
        self._log(
            "Combined WanGP audio segments for Midom upload; "
            f"segments={len(paths)} target={target_path.name!r} bytes={target_path.stat().st_size}."
        )
        return str(target_path)

    def _result_video_files(self, result, settings: Optional[dict[str, Any]] = None) -> list[str]:
        files = []
        for file_path in list(getattr(result, "generated_files", []) or []):
            path = Path(str(file_path))
            if path.suffix.lower() in ALLOWED_VIDEO_SUFFIXES:
                files.append(str(path))
                self._log(
                    f"Accepted generated video output file; filename={path.name!r} "
                    f"suffix={path.suffix.lower()!r} bytes={path.stat().st_size if path.is_file() else 'missing'}."
                )
            else:
                self._log(f"Ignoring non-video WanGP output file; path={path}.")
        for artifact in tuple(getattr(result, "artifacts", ()) or ()):
            file_path = getattr(artifact, "path", None)
            media_type = str(getattr(artifact, "media_type", "") or "")
            if file_path and media_type.startswith("video"):
                path = Path(str(file_path))
                if str(path) not in files and path.suffix.lower() in ALLOWED_VIDEO_SUFFIXES:
                    files.append(str(path))
                    self._log(
                        f"Accepted generated video artifact; media_type={media_type!r} filename={path.name!r} "
                        f"suffix={path.suffix.lower()!r} bytes={path.stat().st_size if path.is_file() else 'missing'}."
                    )
            elif file_path:
                self._log(f"Ignoring non-video WanGP artifact; media_type={media_type!r} path={file_path}.")
        files = self._select_video_output_files(files, settings or {})
        self._log(f"Resolved generated video files from WanGP result; count={len(files)}.")
        return files

    def _select_video_output_files(self, files: list[str], settings: dict[str, Any]) -> list[str]:
        unique_files = []
        seen = set()
        for file_path in files:
            path = Path(str(file_path))
            key = str(path.resolve()) if path.exists() else str(path)
            if key not in seen:
                unique_files.append(str(path))
                seen.add(key)
        unique_files = self._filter_input_echo_video_outputs(unique_files, settings)
        if len(unique_files) <= 1:
            return unique_files

        media_type = str(settings.get("_midom_media_type") or "").strip().lower()
        model_type = str(settings.get("model_type") or "").strip()
        output_count = self._coerce_int(settings.get("_midom_output_count"), 1, 1, 10)
        expected_duration = 0.0
        try:
            expected_duration = float(settings.get("duration_seconds") or 0)
        except (TypeError, ValueError):
            expected_duration = 0.0

        scored = []
        for file_path in unique_files:
            path = Path(file_path)
            size = path.stat().st_size if path.is_file() else 0
            mtime = path.stat().st_mtime if path.is_file() else 0.0
            duration = self._probe_video_duration_seconds(path)
            if expected_duration > 0 and duration is not None and duration > 0:
                rank = (0, abs(duration - expected_duration), -duration, -size, -mtime)
            elif duration is not None and duration > 0:
                rank = (1, -duration, -size, -mtime, 0)
            else:
                rank = (2, -size, -mtime, 0, 0)
            scored.append((rank, str(path), duration, size, mtime))
            self._log(
                "Video output candidate; "
                f"model_type={model_type!r} filename={path.name!r} "
                f"duration_seconds={duration if duration is not None else 'unknown'} "
                f"bytes={size} mtime={mtime:.0f}."
            )

        scored.sort(key=lambda item: item[0])
        selected = [item[1] for item in scored[:output_count]]
        selected_names = [Path(item).name for item in selected]
        self._log(
            "Selected video output file(s) for Midom upload; "
            f"media_type={media_type!r} model_type={model_type!r} "
            f"expected_duration_seconds={expected_duration or 'unknown'} selected={selected_names}."
        )
        return selected

    def _filter_input_echo_video_outputs(self, files: list[str], settings: dict[str, Any]) -> list[str]:
        input_paths = {
            str(Path(str(path)).resolve())
            for path in settings.get("_midom_input_video_paths", []) or []
            if str(path or "").strip()
        }
        input_sha256s = {
            str(value or "").strip().lower()
            for value in settings.get("_midom_input_video_sha256s", []) or []
            if str(value or "").strip()
        }
        if not input_paths and not input_sha256s:
            return files

        filtered = []
        for file_path in files:
            path = Path(str(file_path))
            resolved = str(path.resolve()) if path.exists() else str(path)
            if resolved in input_paths:
                self._log(
                    "Ignoring WanGP video output because it is the original input video path; "
                    f"filename={path.name!r}."
                )
                continue
            if input_sha256s and path.is_file():
                try:
                    with path.open("rb") as reader:
                        digest = hashlib.sha256(reader.read()).hexdigest()
                except Exception as exc:
                    self._log(f"Could not hash WanGP video output candidate; filename={path.name!r} error={exc}.")
                    digest = ""
                if digest and digest in input_sha256s:
                    self._log(
                        "Ignoring WanGP video output because it is byte-identical to the input control video; "
                        f"filename={path.name!r} sha256={digest[:12]}..."
                    )
                    continue
            filtered.append(file_path)
        if files and not filtered:
            self._log(
                "WanGP returned only input-echo video output(s); refusing to upload them as generated video.",
                force=True,
            )
        return filtered

    def _probe_video_duration_seconds(self, path: Path) -> Optional[float]:
        if not path.is_file():
            return None
        try:
            from shared.utils.video_decode import probe_video_stream_metadata

            metadata = probe_video_stream_metadata(str(path))
            if isinstance(metadata, dict):
                duration = float(metadata.get("duration") or 0)
                if duration > 0:
                    return duration
        except Exception as exc:
            self._log(f"Could not probe video duration; filename={path.name!r} error={exc}.")
        return None

    def _probe_control_video_metadata(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError(f"Control video file is missing: {path.name}")
        command = [
            self._ffprobe_binary(),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
        if completed.returncode != 0:
            stderr = str(completed.stderr or completed.stdout or "").strip()
            raise ValueError(f"Could not inspect control video metadata: {stderr}")
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse control video metadata: {exc}")
        streams = payload.get("streams") or []
        if not isinstance(streams, list):
            streams = []
        video_stream = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"), None)
        audio_stream = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"), None)
        if not isinstance(video_stream, dict):
            raise ValueError("Control video must contain a video stream.")
        width = self._coerce_int(video_stream.get("width"), 0, 0, 100_000)
        height = self._coerce_int(video_stream.get("height"), 0, 0, 100_000)
        if width <= 0 or height <= 0:
            raise ValueError("Control video width/height could not be read.")
        tags = video_stream.get("tags") if isinstance(video_stream.get("tags"), dict) else {}
        rotation_values = []
        if tags.get("rotate") not in {None, ""}:
            rotation_values.append(tags.get("rotate"))
        for side_data in video_stream.get("side_data_list") or []:
            if isinstance(side_data, dict) and side_data.get("rotation") not in {None, ""}:
                rotation_values.append(side_data.get("rotation"))
        for rotation in rotation_values:
            try:
                normalized_rotation = abs(float(rotation)) % 360.0
            except (TypeError, ValueError):
                normalized_rotation = 0.0
            if normalized_rotation:
                raise ValueError("Control video must have physically rotated pixels and no rotation metadata.")
        sample_aspect_ratio = str(video_stream.get("sample_aspect_ratio") or "").strip()
        if sample_aspect_ratio and sample_aspect_ratio not in {"1:1", "0:1"}:
            raise ValueError(f"Control video must use square pixels; got SAR {sample_aspect_ratio}.")
        format_block = payload.get("format") if isinstance(payload.get("format"), dict) else {}

        def _duration_seconds(block: Any) -> float:
            if not isinstance(block, dict):
                return 0.0
            try:
                value = float(block.get("duration") or 0)
            except (TypeError, ValueError):
                value = 0.0
            return value if value > 0 else 0.0

        video_duration = _duration_seconds(video_stream) or _duration_seconds(format_block)
        audio_duration = _duration_seconds(audio_stream) if isinstance(audio_stream, dict) else 0.0
        if audio_duration <= 0 and isinstance(audio_stream, dict):
            audio_duration = _duration_seconds(format_block)
        if video_duration <= 0:
            raise ValueError("Control video duration could not be read.")
        if not isinstance(audio_stream, dict) or audio_duration <= 0:
            raise ValueError("LTX control-video audio-guided mode requires embedded control-video audio.")
        fps_text = str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "").strip()
        fps = 0.0
        if "/" in fps_text:
            numerator, denominator = fps_text.split("/", 1)
            try:
                fps = float(numerator) / float(denominator)
            except (TypeError, ValueError, ZeroDivisionError):
                fps = 0.0
        else:
            try:
                fps = float(fps_text)
            except (TypeError, ValueError):
                fps = 0.0
        return {
            "width": width,
            "height": height,
            "video_duration_seconds": video_duration,
            "audio_duration_seconds": audio_duration,
            "fps": fps if fps > 0 else None,
        }

    def _probe_event_video_metadata(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError(f"Event source video file is missing: {path.name}")
        command = [
            self._ffprobe_binary(),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        if completed.returncode != 0:
            stderr = str(completed.stderr or completed.stdout or "").strip()
            raise ValueError(f"Could not inspect Event source video metadata: {stderr}")
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse Event source video metadata: {exc}")
        streams = payload.get("streams") or []
        if not isinstance(streams, list):
            streams = []
        video_stream = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"), None)
        audio_stream = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"), None)
        if not isinstance(video_stream, dict):
            raise ValueError("Event source video must contain a video stream.")
        format_block = payload.get("format") if isinstance(payload.get("format"), dict) else {}

        def _duration_seconds(block: Any) -> float:
            if not isinstance(block, dict):
                return 0.0
            try:
                value = float(block.get("duration") or 0)
            except (TypeError, ValueError):
                value = 0.0
            return value if value > 0 else 0.0

        width = self._coerce_int(video_stream.get("width"), 0, 0, 100_000)
        height = self._coerce_int(video_stream.get("height"), 0, 0, 100_000)
        if width <= 0 or height <= 0:
            raise ValueError("Event source video width/height could not be read.")
        rotation = self._video_rotation_degrees(video_stream)
        display_width, display_height = (height, width) if int(abs(rotation)) % 180 == 90 else (width, height)
        duration = _duration_seconds(video_stream) or _duration_seconds(format_block)
        if duration <= 0:
            raise ValueError("Event source video duration could not be read.")
        audio_duration = _duration_seconds(audio_stream) if isinstance(audio_stream, dict) else 0.0
        if audio_duration <= 0 and isinstance(audio_stream, dict):
            audio_duration = _duration_seconds(format_block)
        orientation = "portrait" if display_height > display_width else "landscape"
        fps_text = str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "").strip()
        fps = 0.0
        if "/" in fps_text:
            numerator, denominator = fps_text.split("/", 1)
            try:
                fps = float(numerator) / float(denominator)
            except (TypeError, ValueError, ZeroDivisionError):
                fps = 0.0
        else:
            try:
                fps = float(fps_text)
            except (TypeError, ValueError):
                fps = 0.0
        metadata = {
            "width": width,
            "height": height,
            "display_width": display_width,
            "display_height": display_height,
            "orientation": orientation,
            "duration_seconds": duration,
            "has_audio": isinstance(audio_stream, dict) and audio_duration > 0,
            "audio_duration_seconds": audio_duration,
            "video_codec": str(video_stream.get("codec_name") or "").strip().lower(),
            "audio_codec": str(audio_stream.get("codec_name") or "").strip().lower() if isinstance(audio_stream, dict) else "",
            "rotation_degrees": rotation,
            "fps": fps if fps > 0 else None,
        }
        self._log(
            "Probed Event source video; "
            f"filename={path.name!r} stored_size={width}x{height} display_size={display_width}x{display_height} "
            f"orientation={orientation} duration_seconds={duration:.2f} has_audio={metadata['has_audio']} "
            f"rotation_degrees={rotation} fps={metadata['fps'] if metadata['fps'] is not None else 'unknown'}."
        )
        return metadata

    @staticmethod
    def _video_rotation_degrees(video_stream: dict[str, Any]) -> int:
        rotation_values = []
        tags = video_stream.get("tags") if isinstance(video_stream.get("tags"), dict) else {}
        if tags.get("rotate") not in {None, ""}:
            rotation_values.append(tags.get("rotate"))
        for side_data in video_stream.get("side_data_list") or []:
            if isinstance(side_data, dict) and side_data.get("rotation") not in {None, ""}:
                rotation_values.append(side_data.get("rotation"))
        for rotation in rotation_values:
            try:
                return int(round(float(rotation))) % 360
            except (TypeError, ValueError):
                continue
        return 0

    def _probe_audio_metadata(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError(f"Audio file is missing: {path.name}")
        command = [
            self._ffprobe_binary(),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
        if completed.returncode != 0:
            stderr = str(completed.stderr or completed.stdout or "").strip()
            raise ValueError(f"Could not inspect audio metadata: {stderr}")
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse audio metadata: {exc}")
        streams = payload.get("streams") or []
        if not isinstance(streams, list):
            streams = []
        audio_stream = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"), None)
        if not isinstance(audio_stream, dict):
            raise ValueError("Audio input must contain an audio stream.")
        format_block = payload.get("format") if isinstance(payload.get("format"), dict) else {}

        def _duration_seconds(block: Any) -> float:
            if not isinstance(block, dict):
                return 0.0
            try:
                value = float(block.get("duration") or 0)
            except (TypeError, ValueError):
                value = 0.0
            return value if value > 0 else 0.0

        duration = _duration_seconds(audio_stream) or _duration_seconds(format_block)
        try:
            sample_rate = int(audio_stream.get("sample_rate") or 0)
        except (TypeError, ValueError):
            sample_rate = 0
        if duration <= 0:
            raise ValueError("Audio input duration could not be read.")
        return {
            "duration_seconds": duration,
            "sample_rate_hz": sample_rate,
        }

    def _download_job_inputs(self, connection: ConnectionContext, job: dict[str, Any], temp_dir: str) -> list[dict[str, Any]]:
        inputs = job.get("inputs") or []
        if not isinstance(inputs, list):
            raise ValueError("Job inputs must be a list.")
        if self._is_event_video_processing_job(job):
            return self._download_event_video_processing_inputs(connection, job, temp_dir, inputs)
        if self._is_storyboard_ffmpeg_processing_job(job):
            return self._download_storyboard_ffmpeg_processing_inputs(connection, job, temp_dir, inputs)
        media_type = str(job.get("media_type") or "image").strip().lower()
        if media_type == "audio":
            return self._download_audio_job_inputs(connection, job, temp_dir, inputs)
        if media_type == "video":
            return self._download_video_job_inputs(connection, job, temp_dir, inputs)
        downloaded = []
        job_id = self._coerce_job_id(job)
        model_id = str(job.get("model_id") or job.get("model_type") or "").strip()
        model_meta = MODEL_CAPABILITY_OVERRIDES.get(model_id) or {}
        supported_control_modes = {item["mode_id"] for item in self._control_modes_for_model(model_id)}
        max_reference_images = self._coerce_int((job.get("limits") or {}).get("max_reference_images"), 0, 0, 3)
        max_control_images = self._coerce_int((job.get("limits") or {}).get("max_control_images"), MAX_CONTROL_IMAGES, 0, MAX_CONTROL_IMAGES)
        tool_id = str(job.get("tool_id") or ((job.get("generation") or {}).get("tool_id") if isinstance(job.get("generation"), dict) else "") or "").strip()
        if tool_id == QWEN_MULTI_ANGLE_TOOL_ID:
            max_reference_images = 1
            max_control_images = 0
        reference_count = 0
        control_count = 0
        self._log(
            f"Downloading job inputs; job_id={job_id} descriptors={len(inputs)} "
            f"max_reference_images={max_reference_images} max_control_images={max_control_images}."
        )
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("Job input descriptor must be a JSON object.")
            input_id = self._coerce_input_id(item)
            kind = str(item.get("kind") or "reference_image").strip()
            if kind not in {"reference_image", "control_image"}:
                raise ValueError(f"Unsupported job input kind: {kind}")
            if kind == "reference_image":
                if not model_meta.get("image_reference"):
                    raise ValueError(f"Model {model_id} does not support ordinary reference images.")
                reference_count += 1
                if reference_count > max_reference_images:
                    raise ValueError("Job exceeds maximum reference image count.")
            if kind == "control_image":
                if not model_meta.get("control"):
                    raise ValueError(f"Model {model_id} does not support control images.")
                control_count += 1
                if control_count > max_control_images:
                    raise ValueError("Job exceeds maximum control image count.")
                control_mode = str(item.get("control_mode") or "raw").strip()
                if control_mode not in supported_control_modes:
                    raise ValueError(f"Unsupported control mode for {model_id}: {control_mode}")
            else:
                control_mode = ""
            response = requests.get(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{job_id}/inputs/{input_id}",
                headers=self._headers(connection),
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )
            response.raise_for_status()
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
                raise ValueError(f"Unsupported input MIME type: {mime_type}")
            data = self._read_limited_response_content(response, input_id, self._max_input_bytes_for_mime(mime_type))
            expected_size = item.get("bytes")
            if expected_size is not None and len(data) != int(expected_size):
                raise ValueError(f"Input {input_id} size mismatch.")
            actual_sha256 = hashlib.sha256(data).hexdigest()
            expected_sha256 = str(item.get("sha256") or response.headers.get("X-Midom-SHA256") or "").strip().lower()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(f"Input {input_id} SHA-256 mismatch.")
            filename = self._safe_input_filename(item.get("filename"), input_id, mime_type)
            path = Path(temp_dir) / filename
            with path.open("wb") as writer:
                writer.write(data)
            downloaded.append({
                "input_id": input_id,
                "kind": kind,
                "path": str(path),
                "mime_type": mime_type,
                "sha256": actual_sha256,
                "control_mode": control_mode,
            })
            self._log(
                f"Downloaded input; job_id={job_id} input_id={input_id} "
                f"kind={kind} mime_type={mime_type} bytes={len(data)} sha256={actual_sha256[:12]}..."
            )
        if model_id == "qwen_image_layered_20B":
            if reference_count != 0:
                raise ValueError("Qwen Image Layered does not support ordinary reference images.")
            if control_count != 1:
                raise ValueError("Qwen Image Layered requires exactly one Control Image.")
        return downloaded

    def _download_audio_job_inputs(
        self,
        connection: ConnectionContext,
        job: dict[str, Any],
        temp_dir: str,
        inputs: list[Any],
    ) -> list[dict[str, Any]]:
        downloaded = []
        job_id = self._coerce_job_id(job)
        limits = job.get("limits") or {}
        max_reference_audio_files = self._coerce_int(limits.get("max_reference_audio_files"), 2, 1, 2)
        required_reference_audio_files = self._coerce_int(limits.get("required_reference_audio_files"), 1, 1, 1)
        max_emotion_reference_audio_files = self._coerce_int(limits.get("max_emotion_reference_audio_files"), 1, 0, 1)
        max_dialogue_speakers = self._coerce_int(limits.get("max_dialogue_speakers"), 2, 1, DRAMABOX_MAX_DIALOGUE_SPEAKERS)
        voice_mode = str(((job.get("generation") or {}).get("voice_mode")) or "single_reference").strip().lower()
        model_id = str(job.get("model_id") or job.get("model_type") or job.get("model") or "").strip()
        if self._is_stable_audio3_model(model_id) or self._is_ace_step15_music_model(model_id):
            if inputs:
                raise ValueError(f"{model_id} first-pass jobs do not support inputs.")
            self._log(f"Generative audio job has no downloadable inputs; job_id={job_id} model_id={model_id}.")
            return []
        if model_id == SEEDVC_MODEL_ID:
            return self._download_seedvc_job_inputs(connection, job, temp_dir, inputs)
        reference_count = 0
        emotion_reference_count = 0
        speaker1_reference_count = 0
        speaker2_reference_count = 0
        self._log(
            f"Downloading audio job inputs; job_id={job_id} descriptors={len(inputs)} "
            f"voice_mode={voice_mode!r} "
            f"max_reference_audio_files={max_reference_audio_files} "
            f"max_emotion_reference_audio_files={max_emotion_reference_audio_files} "
            f"max_dialogue_speakers={max_dialogue_speakers}."
        )
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("Audio job input descriptor must be a JSON object.")
            input_id = self._coerce_input_id(item)
            kind = str(item.get("kind") or "").strip()
            if kind not in {"reference_audio", "emotion_reference_audio", "speaker1_reference_audio", "speaker2_reference_audio"}:
                raise ValueError(f"Unsupported audio job input kind: {kind}")
            if kind == "reference_audio":
                reference_count += 1
            elif kind == "emotion_reference_audio":
                emotion_reference_count += 1
            elif kind == "speaker1_reference_audio":
                speaker1_reference_count += 1
            elif kind == "speaker2_reference_audio":
                speaker2_reference_count += 1
            total_reference_count = reference_count + emotion_reference_count + speaker1_reference_count + speaker2_reference_count
            if total_reference_count > max_reference_audio_files:
                raise ValueError("Audio job exceeds maximum reference audio count.")
            if emotion_reference_count > max_emotion_reference_audio_files:
                raise ValueError("Audio job exceeds maximum emotion reference audio count.")
            if speaker1_reference_count > 1 or speaker2_reference_count > 1:
                raise ValueError("Dialogue job supports exactly one reference per speaker.")
            if (speaker1_reference_count + speaker2_reference_count) > max_dialogue_speakers:
                raise ValueError("Audio job exceeds maximum dialogue speaker count.")
            self._log(
                f"Requesting audio input download; job_id={job_id} input_id={input_id} "
                f"kind={kind} declared_mime={item.get('mime_type')!r} declared_bytes={item.get('bytes')!r}."
            )
            response = requests.get(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{job_id}/inputs/{input_id}",
                headers=self._headers(connection),
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )
            response.raise_for_status()
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime_type not in ALLOWED_AUDIO_INPUT_MIME_TYPES:
                raise ValueError(f"Unsupported audio input MIME type: {mime_type}")
            data = self._read_limited_response_content(response, input_id, self._max_input_bytes_for_mime(mime_type))
            expected_size = item.get("bytes")
            if expected_size is not None and len(data) != int(expected_size):
                raise ValueError(f"Audio input {input_id} size mismatch.")
            actual_sha256 = hashlib.sha256(data).hexdigest()
            expected_sha256 = str(item.get("sha256") or response.headers.get("X-Midom-SHA256") or "").strip().lower()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(f"Audio input {input_id} SHA-256 mismatch.")
            filename = self._safe_input_filename(item.get("filename"), input_id, mime_type)
            path = Path(temp_dir) / filename
            with path.open("wb") as writer:
                writer.write(data)
            downloaded.append({
                "input_id": input_id,
                "kind": kind,
                "path": str(path),
                "mime_type": mime_type,
                "sha256": actual_sha256,
            })
            self._log(
                f"Downloaded audio input; job_id={job_id} input_id={input_id} "
                f"kind={kind} mime_type={mime_type} bytes={len(data)} sha256={actual_sha256[:12]}..."
            )
        self._log(
            "Audio input download complete; "
            f"job_id={job_id} voice_mode={voice_mode!r} "
            f"reference_audio_count={reference_count} "
            f"emotion_reference_audio_count={emotion_reference_count} "
            f"speaker1_reference_audio_count={speaker1_reference_count} "
            f"speaker2_reference_audio_count={speaker2_reference_count}."
        )
        if voice_mode in {"two_speaker_dialogue", "two_voice_clone_n_voice_dialogue"}:
            if reference_count or emotion_reference_count:
                raise ValueError("Dialogue job must not include reference_audio or emotion_reference_audio inputs.")
            if speaker1_reference_count != 1 or speaker2_reference_count != 1:
                raise ValueError(
                    "Dialogue job requires exactly one speaker1_reference_audio and one speaker2_reference_audio input."
                )
        elif reference_count != required_reference_audio_files:
            raise ValueError(f"Audio job requires exactly one reference_audio input; got {reference_count}.")
        return downloaded

    def _download_seedvc_job_inputs(
        self,
        connection: ConnectionContext,
        job: dict[str, Any],
        temp_dir: str,
        inputs: list[Any],
    ) -> list[dict[str, Any]]:
        downloaded = []
        job_id = self._coerce_job_id(job)
        source_count = 0
        reference_count = 0
        self._log(f"Downloading SeedVC job inputs; job_id={job_id} descriptors={len(inputs)}.")
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("SeedVC job input descriptor must be a JSON object.")
            input_id = self._coerce_input_id(item)
            kind = str(item.get("kind") or "").strip()
            if kind not in {"source_audio", "reference_audio"}:
                raise ValueError(f"Unsupported SeedVC job input kind: {kind}")
            if kind == "source_audio":
                source_count += 1
                if source_count > 1:
                    raise ValueError("SeedVC requires exactly one source_audio input.")
            elif kind == "reference_audio":
                reference_count += 1
                if reference_count > 1:
                    raise ValueError("SeedVC requires exactly one reference_audio input.")
            self._log(
                f"Requesting SeedVC input download; job_id={job_id} input_id={input_id} "
                f"kind={kind} declared_mime={item.get('mime_type')!r} declared_bytes={item.get('bytes')!r}."
            )
            response = requests.get(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{job_id}/inputs/{input_id}",
                headers=self._headers(connection),
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )
            response.raise_for_status()
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime_type not in ALLOWED_AUDIO_INPUT_MIME_TYPES:
                raise ValueError(f"Unsupported SeedVC input MIME type: {mime_type}")
            data = self._read_limited_response_content(response, input_id, self._max_input_bytes_for_mime(mime_type))
            expected_size = item.get("bytes")
            if expected_size is not None and len(data) != int(expected_size):
                raise ValueError(f"SeedVC input {input_id} size mismatch.")
            actual_sha256 = hashlib.sha256(data).hexdigest()
            expected_sha256 = str(item.get("sha256") or response.headers.get("X-Midom-SHA256") or "").strip().lower()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(f"SeedVC input {input_id} SHA-256 mismatch.")
            filename = self._safe_input_filename(item.get("filename"), input_id, mime_type)
            path = Path(temp_dir) / filename
            with path.open("wb") as writer:
                writer.write(data)
            metadata = self._probe_audio_metadata(path)
            duration_seconds = float(metadata.get("duration_seconds") or 0.0)
            sample_rate_hz = int(metadata.get("sample_rate_hz") or 0)
            if kind == "source_audio" and duration_seconds > SEEDVC_MAX_SOURCE_AUDIO_SECONDS + 0.05:
                raise ValueError(
                    f"SeedVC source_audio exceeds {SEEDVC_MAX_SOURCE_AUDIO_SECONDS} seconds; "
                    f"got {duration_seconds:.2f}s."
                )
            if kind == "reference_audio":
                if duration_seconds > SEEDVC_MAX_REFERENCE_AUDIO_SECONDS + 0.05:
                    raise ValueError(
                        f"SeedVC reference_audio exceeds {SEEDVC_MAX_REFERENCE_AUDIO_SECONDS} seconds; "
                        f"got {duration_seconds:.2f}s."
                    )
                if sample_rate_hz > 0 and sample_rate_hz < MIN_REFERENCE_AUDIO_SAMPLE_RATE_HZ:
                    raise ValueError(
                        f"SeedVC reference_audio sample rate must be at least {MIN_REFERENCE_AUDIO_SAMPLE_RATE_HZ} Hz; "
                        f"got {sample_rate_hz} Hz."
                    )
            downloaded.append({
                "input_id": input_id,
                "kind": kind,
                "path": str(path),
                "mime_type": mime_type,
                "sha256": actual_sha256,
                "duration_seconds": duration_seconds,
                "sample_rate_hz": sample_rate_hz,
            })
            self._log(
                f"Downloaded SeedVC input; job_id={job_id} input_id={input_id} "
                f"kind={kind} mime_type={mime_type} bytes={len(data)} "
                f"duration_seconds={duration_seconds:.2f} sample_rate_hz={sample_rate_hz} "
                f"sha256={actual_sha256[:12]}..."
            )
        if source_count != 1:
            raise ValueError(f"SeedVC requires exactly one source_audio input; got {source_count}.")
        if reference_count != 1:
            raise ValueError(f"SeedVC requires exactly one reference_audio input; got {reference_count}.")
        self._log(
            "SeedVC input download complete; "
            f"job_id={job_id} source_audio_count={source_count} reference_audio_count={reference_count}."
        )
        return downloaded

    def _download_video_job_inputs(
        self,
        connection: ConnectionContext,
        job: dict[str, Any],
        temp_dir: str,
        inputs: list[Any],
    ) -> list[dict[str, Any]]:
        downloaded = []
        job_id = self._coerce_job_id(job)
        model_type = str(job.get("model_id") or job.get("model_type") or job.get("model") or "").strip()
        generation = job.get("generation") or {}
        video_task = str(generation.get("video_task") or "").strip().lower() if isinstance(generation, dict) else ""
        is_ltx_control_video = model_type in LTX_VIDEO_MODEL_IDS and video_task == "control_video_guided_video"
        if model_type in LTX_VIDEO_MODEL_IDS or model_type == SVI_VIDEO_MODEL_ID:
            required_image_kind = "start_image"
        else:
            required_image_kind = "reference_image"
        requires_driving_audio = model_type != SVI_VIDEO_MODEL_ID and not is_ltx_control_video
        requires_control_video = is_ltx_control_video
        model_label = "SVI" if model_type == SVI_VIDEO_MODEL_ID else ("LTX" if model_type in LTX_VIDEO_MODEL_IDS else "LongCat")
        reference_image_count = 0
        end_image_count = 0
        driving_audio_count = 0
        control_video_count = 0
        self._log(
            f"Downloading video job inputs; job_id={job_id} descriptors={len(inputs)} "
            f"required_{required_image_kind}=1 required_driving_audio={1 if requires_driving_audio else 0} "
            f"required_control_video={1 if requires_control_video else 0} "
            f"optional_end_image={1 if (model_type in LTX_VIDEO_MODEL_IDS or model_type == SVI_VIDEO_MODEL_ID) and not is_ltx_control_video else 0}."
        )
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("Video job input descriptor must be a JSON object.")
            input_id = self._coerce_input_id(item)
            kind = str(item.get("kind") or "").strip()
            allowed_kinds = {required_image_kind, "driving_audio"} if requires_driving_audio else {required_image_kind}
            if requires_control_video:
                allowed_kinds.add("control_video")
            if (model_type in LTX_VIDEO_MODEL_IDS or model_type == SVI_VIDEO_MODEL_ID) and not is_ltx_control_video:
                allowed_kinds.add("end_image")
            if kind not in allowed_kinds:
                raise ValueError(f"Unsupported video job input kind: {kind}")
            if kind == required_image_kind:
                reference_image_count += 1
                allowed_mime_types = ALLOWED_IMAGE_MIME_TYPES
            elif kind == "end_image":
                end_image_count += 1
                allowed_mime_types = ALLOWED_IMAGE_MIME_TYPES
            elif kind == "control_video":
                control_video_count += 1
                allowed_mime_types = ALLOWED_CONTROL_VIDEO_MIME_TYPES
            else:
                driving_audio_count += 1
                allowed_mime_types = ALLOWED_AUDIO_INPUT_MIME_TYPES
            if reference_image_count > 1:
                raise ValueError(f"{model_label} video job supports exactly one {required_image_kind} input.")
            if end_image_count > 1:
                raise ValueError(f"{model_label} video job supports at most one end_image input.")
            if driving_audio_count > 1:
                raise ValueError(f"{model_label} video job supports exactly one driving_audio input.")
            if control_video_count > 1:
                raise ValueError(f"{model_label} video job supports exactly one control_video input.")
            self._log(
                f"Requesting video input download; job_id={job_id} input_id={input_id} "
                f"kind={kind} declared_mime={item.get('mime_type')!r} declared_bytes={item.get('bytes')!r}."
            )
            response = requests.get(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{job_id}/inputs/{input_id}",
                headers=self._headers(connection),
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )
            response.raise_for_status()
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime_type not in allowed_mime_types:
                raise ValueError(f"Unsupported {kind} MIME type: {mime_type}")
            data = self._read_limited_response_content(response, input_id, self._max_input_bytes_for_mime(mime_type))
            expected_size = item.get("bytes")
            if expected_size is not None and len(data) != int(expected_size):
                raise ValueError(f"Video input {input_id} size mismatch.")
            actual_sha256 = hashlib.sha256(data).hexdigest()
            expected_sha256 = str(item.get("sha256") or response.headers.get("X-Midom-SHA256") or "").strip().lower()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(f"Video input {input_id} SHA-256 mismatch.")
            filename = self._safe_input_filename(item.get("filename"), input_id, mime_type)
            path = Path(temp_dir) / filename
            with path.open("wb") as writer:
                writer.write(data)
            downloaded.append({
                "input_id": input_id,
                "kind": kind,
                "path": str(path),
                "mime_type": mime_type,
                "sha256": actual_sha256,
            })
            self._log(
                f"Downloaded video input; job_id={job_id} input_id={input_id} "
                f"kind={kind} mime_type={mime_type} bytes={len(data)} sha256={actual_sha256[:12]}..."
            )
        if reference_image_count != 1:
            raise ValueError(f"{model_label} video job requires exactly one {required_image_kind} input; got {reference_image_count}.")
        if requires_driving_audio and driving_audio_count != 1:
            raise ValueError(f"{model_label} video job requires exactly one driving_audio input; got {driving_audio_count}.")
        if not requires_driving_audio and driving_audio_count != 0:
            raise ValueError(f"{model_label} video job does not support driving_audio inputs.")
        if requires_control_video and control_video_count != 1:
            raise ValueError(f"{model_label} control-video job requires exactly one control_video input; got {control_video_count}.")
        if not requires_control_video and control_video_count != 0:
            raise ValueError(f"{model_label} video job does not support control_video inputs.")
        self._log(
            "Video input download complete; "
            f"job_id={job_id} {required_image_kind}_count={reference_image_count} "
            f"end_image_count={end_image_count} driving_audio_count={driving_audio_count} "
            f"control_video_count={control_video_count}."
        )
        return downloaded

    def _download_event_video_processing_inputs(
        self,
        connection: ConnectionContext,
        job: dict[str, Any],
        temp_dir: str,
        inputs: list[Any],
    ) -> list[dict[str, Any]]:
        downloaded = []
        job_id = self._coerce_job_id(job)
        processing = job.get("processing") or {}
        apply_overlay = self._coerce_bool((processing if isinstance(processing, dict) else {}).get("apply_overlay"), False)
        add_ending_bumper = self._coerce_bool((processing if isinstance(processing, dict) else {}).get("add_ending_bumper"), False)
        source_count = 0
        overlay_count = 0
        bumper_count = 0
        source_mime_types = set(ALLOWED_EVENT_SOURCE_VIDEO_MIME_TYPES)
        if not self._ffmpeg_processing_probe().get("quicktime_demux_available"):
            source_mime_types.discard("video/quicktime")
        self._log(
            f"Downloading Event Video Processing inputs; job_id={job_id} descriptors={len(inputs)} "
            f"apply_overlay={apply_overlay} add_ending_bumper={add_ending_bumper} "
            f"source_mime_types={sorted(source_mime_types)}."
        )
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("Event Video Processing input descriptor must be a JSON object.")
            input_id = self._coerce_input_id(item)
            kind = str(item.get("kind") or "").strip()
            if kind not in ALLOWED_EVENT_INPUT_KINDS:
                raise ValueError(f"Unsupported Event Video Processing input kind: {kind}")
            orientation = ""
            if kind == "source_video":
                source_count += 1
                if source_count > 1:
                    raise ValueError("Event Video Processing supports exactly one source_video input.")
                allowed_mime_types = source_mime_types
                max_bytes = MAX_EVENT_VIDEO_INPUT_BYTES
            elif kind == "overlay_png":
                overlay_count += 1
                if overlay_count > 2:
                    raise ValueError("Event Video Processing supports at most one overlay_png per orientation.")
                orientation = str(item.get("orientation") or "").strip().lower()
                if orientation not in {"portrait", "landscape"}:
                    raise ValueError("overlay_png inputs require orientation portrait or landscape.")
                allowed_mime_types = ALLOWED_EVENT_OVERLAY_MIME_TYPES
                max_bytes = MAX_IMAGE_BYTES
            else:
                bumper_count += 1
                if bumper_count > 2:
                    raise ValueError("Event Video Processing supports at most one bumper_image per orientation.")
                orientation = str(item.get("orientation") or "").strip().lower()
                if orientation not in {"portrait", "landscape"}:
                    raise ValueError("bumper_image inputs require orientation portrait or landscape.")
                allowed_mime_types = ALLOWED_EVENT_BUMPER_MIME_TYPES
                max_bytes = MAX_IMAGE_BYTES
            self._log(
                f"Requesting Event Video Processing input download; job_id={job_id} input_id={input_id} "
                f"kind={kind} orientation={orientation!r} declared_mime={item.get('mime_type')!r} "
                f"declared_bytes={item.get('bytes')!r}."
            )
            response = requests.get(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{job_id}/inputs/{input_id}",
                headers=self._headers(connection),
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )
            response.raise_for_status()
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime_type not in allowed_mime_types:
                raise ValueError(f"Unsupported Event Video Processing {kind} MIME type: {mime_type}")
            data = self._read_limited_response_content(response, input_id, max_bytes)
            expected_size = item.get("bytes")
            if expected_size is not None and len(data) != int(expected_size):
                raise ValueError(f"Event Video Processing input {input_id} size mismatch.")
            actual_sha256 = hashlib.sha256(data).hexdigest()
            expected_sha256 = str(item.get("sha256") or response.headers.get("X-Midom-SHA256") or "").strip().lower()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(f"Event Video Processing input {input_id} SHA-256 mismatch.")
            filename = self._safe_input_filename(item.get("filename"), input_id, mime_type)
            path = Path(temp_dir) / filename
            with path.open("wb") as writer:
                writer.write(data)
            entry = {
                "input_id": input_id,
                "kind": kind,
                "path": str(path),
                "mime_type": mime_type,
                "sha256": actual_sha256,
            }
            if orientation:
                entry["orientation"] = orientation
            if kind == "source_video":
                metadata = self._probe_event_video_metadata(path)
                duration_seconds = float(metadata.get("duration_seconds") or 0.0)
                if duration_seconds > MAX_EVENT_VIDEO_DURATION_SECONDS + 0.05:
                    raise ValueError(
                        f"Event source video exceeds {MAX_EVENT_VIDEO_DURATION_SECONDS} seconds; "
                        f"got {duration_seconds:.2f}s."
                    )
                entry["metadata"] = metadata
            else:
                with Image.open(path) as image:
                    entry["width"], entry["height"] = image.size
                    entry["decoded_format"] = str(image.format or "").upper()
                self._validate_event_asset_dimensions(
                    kind,
                    int(entry["width"]),
                    int(entry["height"]),
                    orientation,
                )
                if kind == "overlay_png" and entry["decoded_format"] != "PNG":
                    raise ValueError(f"Event Video Processing overlay_png input must decode as PNG; got {entry['decoded_format']}.")
                if kind == "bumper_image" and entry["decoded_format"] not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError(f"Event Video Processing bumper_image decoded as unsupported format: {entry['decoded_format']}.")
            downloaded.append(entry)
            self._log(
                f"Downloaded Event Video Processing input; job_id={job_id} input_id={input_id} "
                f"kind={kind} orientation={orientation!r} mime_type={mime_type} bytes={len(data)} "
                f"sha256={actual_sha256[:12]}..."
            )
        if source_count != 1:
            raise ValueError(f"Event Video Processing requires exactly one source_video input; got {source_count}.")
        if apply_overlay and overlay_count <= 0:
            raise ValueError("Event Video Processing apply_overlay is true but no overlay_png inputs were downloaded.")
        if add_ending_bumper and bumper_count <= 0:
            raise ValueError("Event Video Processing add_ending_bumper is true but no bumper_image inputs were downloaded.")
        if not apply_overlay and overlay_count:
            raise ValueError("Event Video Processing received overlay_png inputs but apply_overlay is false.")
        if not add_ending_bumper and bumper_count:
            raise ValueError("Event Video Processing received bumper_image inputs but add_ending_bumper is false.")
        self._log(
            "Event Video Processing input download complete; "
            f"job_id={job_id} source_video_count={source_count} overlay_png_count={overlay_count} "
            f"bumper_image_count={bumper_count}."
        )
        return downloaded

    def _download_storyboard_ffmpeg_processing_inputs(
        self,
        connection: ConnectionContext,
        job: dict[str, Any],
        temp_dir: str,
        inputs: list[Any],
    ) -> list[dict[str, Any]]:
        downloaded = []
        job_id = self._coerce_job_id(job)
        operation_type = self._storyboard_operation_type(job)
        raw_processing = job.get("processing") if isinstance(job.get("processing"), dict) else {}
        operation_payload = job.get("operation_payload") if isinstance(job.get("operation_payload"), dict) else {}
        nested_operation_payload = raw_processing.get("operation_payload") if isinstance(raw_processing.get("operation_payload"), dict) else {}
        processing = {**operation_payload, **nested_operation_payload, **{key: value for key, value in raw_processing.items() if key != "operation_payload"}}
        source_video_mime_types = set(ALLOWED_STORYBOARD_SOURCE_VIDEO_MIME_TYPES)
        if not self._ffmpeg_processing_probe().get("quicktime_demux_available"):
            source_video_mime_types.discard("video/quicktime")
        self._log(
            f"Downloading Storyboard FFmpeg inputs; job_id={job_id} operation_type={operation_type} "
            f"descriptors={len(inputs)} video_mime_types={sorted(source_video_mime_types)}."
        )
        for index, item in enumerate(inputs):
            if not isinstance(item, dict):
                raise ValueError("Storyboard FFmpeg input descriptor must be a JSON object.")
            input_id = self._coerce_input_id(item)
            kind = str(item.get("kind") or "").strip()
            if kind not in ALLOWED_STORYBOARD_INPUT_KINDS:
                raise ValueError(f"Unsupported Storyboard FFmpeg input kind: {kind}")
            if kind in STORYBOARD_VIDEO_INPUT_KINDS:
                allowed_mime_types = source_video_mime_types
                max_bytes = MAX_STORYBOARD_VIDEO_INPUT_BYTES
                category = "video"
            elif kind in STORYBOARD_IMAGE_INPUT_KINDS:
                allowed_mime_types = ALLOWED_STORYBOARD_MATTE_MIME_TYPES if self._is_storyboard_matte_descriptor(item, processing) else ALLOWED_IMAGE_MIME_TYPES
                max_bytes = MAX_STORYBOARD_IMAGE_INPUT_BYTES
                category = "image"
            else:
                if kind in {"source_audio", "soundtrack_audio"}:
                    allowed_mime_types = ALLOWED_STORYBOARD_AUDIO_CONTAINER_MIME_TYPES
                    max_bytes = MAX_STORYBOARD_VIDEO_INPUT_BYTES
                else:
                    allowed_mime_types = ALLOWED_AUDIO_INPUT_MIME_TYPES
                    max_bytes = MAX_STORYBOARD_AUDIO_INPUT_BYTES
                category = "audio"
            self._log(
                f"Requesting Storyboard FFmpeg input download; job_id={job_id} input_id={input_id} "
                f"kind={kind} declared_mime={item.get('mime_type')!r} declared_bytes={item.get('bytes')!r}."
            )
            response = requests.get(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{job_id}/inputs/{input_id}",
                headers=self._headers(connection),
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )
            response.raise_for_status()
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime_type not in allowed_mime_types:
                if category == "image" and self._is_storyboard_matte_descriptor(item, processing):
                    raise ValueError(
                        "Overlay matte was provided but could not be applied: "
                        f"unsupported matte MIME type {mime_type or 'unknown'}."
                    )
                raise ValueError(f"Unsupported Storyboard FFmpeg {kind} MIME type: {mime_type}")
            data = self._read_limited_response_content(response, input_id, max_bytes)
            expected_size = item.get("bytes")
            if expected_size is not None and len(data) != int(expected_size):
                raise ValueError(f"Storyboard FFmpeg input {input_id} size mismatch.")
            actual_sha256 = hashlib.sha256(data).hexdigest()
            expected_sha256 = str(item.get("sha256") or response.headers.get("X-Midom-SHA256") or "").strip().lower()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(f"Storyboard FFmpeg input {input_id} SHA-256 mismatch.")
            filename = self._safe_input_filename(item.get("filename"), input_id, mime_type)
            path = Path(temp_dir) / filename
            with path.open("wb") as writer:
                writer.write(data)
            entry = {
                "input_id": input_id,
                "kind": kind,
                "category": category,
                "path": str(path),
                "mime_type": mime_type,
                "sha256": actual_sha256,
                "order": self._coerce_int(item.get("order", item.get("sequence", index)), index, 0, 10_000),
                "role": str(item.get("role") or "").strip(),
            }
            for id_key in ("dbfileid", "dbfile_id", "file_id"):
                if item.get(id_key) is not None:
                    entry[id_key] = item.get(id_key)
            if category == "video":
                metadata = self._probe_event_video_metadata(path)
                duration_seconds = float(metadata.get("duration_seconds") or 0.0)
                if duration_seconds > MAX_STORYBOARD_VIDEO_DURATION_SECONDS + 0.05:
                    raise ValueError(
                        f"Storyboard FFmpeg video input exceeds {MAX_STORYBOARD_VIDEO_DURATION_SECONDS} seconds; "
                        f"got {duration_seconds:.2f}s."
                    )
                entry["metadata"] = metadata
            elif category == "image":
                try:
                    with Image.open(path) as image:
                        entry["width"], entry["height"] = image.size
                        entry["decoded_format"] = str(image.format or "").upper()
                except Exception as exc:
                    if self._is_storyboard_matte_descriptor(entry, processing):
                        raise ValueError(f"Overlay matte was provided but could not be applied: decode failed: {exc}") from exc
                    raise
                if self._is_storyboard_matte_descriptor(entry, processing) and entry["decoded_format"] not in {"PNG", "JPEG"}:
                    raise ValueError(
                        "Overlay matte was provided but could not be applied: "
                        f"decoded matte format {entry['decoded_format'] or 'unknown'} is not PNG or JPEG."
                    )
            else:
                metadata = self._probe_audio_metadata(path)
                entry["metadata"] = metadata
                entry["duration_seconds"] = float(metadata.get("duration_seconds") or 0.0)
            downloaded.append(entry)
            self._log(
                f"Downloaded Storyboard FFmpeg input; job_id={job_id} input_id={input_id} "
                f"kind={kind} category={category} mime_type={mime_type} bytes={len(data)} "
                f"sha256={actual_sha256[:12]}..."
            )
        video_count = sum(1 for item in downloaded if item.get("category") == "video")
        if operation_type == "multicam_final_assembly":
            if video_count < 1:
                raise ValueError("Storyboard final assembly requires at least one downloaded video input.")
        elif operation_type == "replace_video_soundtrack":
            audio_count = sum(1 for item in downloaded if item.get("category") == "audio")
            if video_count != 1:
                raise ValueError(f"Storyboard replace_video_soundtrack requires exactly one downloaded source video input; got {video_count}.")
            if audio_count != 1:
                raise ValueError(f"Storyboard replace_video_soundtrack requires exactly one downloaded soundtrack audio input; got {audio_count}.")
        elif operation_type in STORYBOARD_SINGLE_VIDEO_OPERATION_TYPES and video_count < 1:
            raise ValueError(f"Storyboard operation {operation_type} requires at least one downloaded video input.")
        self._log(
            "Storyboard FFmpeg input download complete; "
            f"job_id={job_id} operation_type={operation_type} video_count={video_count} "
            f"image_count={sum(1 for item in downloaded if item.get('category') == 'image')} "
            f"audio_count={sum(1 for item in downloaded if item.get('category') == 'audio')}."
        )
        return downloaded

    def _validate_event_asset_dimensions(self, kind: str, width: int, height: int, orientation: str) -> None:
        if width < MIN_EVENT_ASSET_DIMENSION or height < MIN_EVENT_ASSET_DIMENSION:
            raise ValueError(
                f"Event Video Processing {kind} asset is too small: {width}x{height}. "
                f"Minimum dimension is {MIN_EVENT_ASSET_DIMENSION}px."
            )
        pixels = int(width) * int(height)
        if pixels > MAX_EVENT_ASSET_PIXELS:
            raise ValueError(
                f"Event Video Processing {kind} asset is too large: {width}x{height} "
                f"({pixels} pixels, max {MAX_EVENT_ASSET_PIXELS})."
            )
        expected_orientation = "portrait" if height > width else "landscape"
        if orientation in {"portrait", "landscape"} and expected_orientation != orientation:
            self._log(
                "Event Video Processing asset orientation metadata differs from decoded dimensions; "
                f"kind={kind} declared_orientation={orientation} decoded={width}x{height}. "
                "The bridge will still use the declared orientation selected by Midom.",
                force=True,
            )

    def _safe_input_filename(self, filename: Any, input_id: int, mime_type: str) -> str:
        name = Path(str(filename or "")).name
        suffix = Path(name).suffix.lower()
        if suffix not in (ALLOWED_IMAGE_SUFFIXES | ALLOWED_AUDIO_SUFFIXES | ALLOWED_VIDEO_SUFFIXES | {".mov"}):
            suffix = MIME_EXTENSION.get(mime_type, ".img")
        return f"input-{int(input_id)}{suffix}"

    def _validate_image_input_exact_size(self, path: Path, expected_size: tuple[int, int], label: str) -> None:
        if not path.is_file():
            raise ValueError(f"{label} file is missing: {path.name}")
        with Image.open(path) as image:
            actual_size = image.size
        if actual_size != expected_size:
            raise ValueError(
                f"{label} must match the selected output profile exactly; "
                f"expected {expected_size[0]}x{expected_size[1]}, got {actual_size[0]}x{actual_size[1]}."
            )

    def _ltx_requested_size(self, settings: dict[str, Any]) -> tuple[int, int]:
        requested_size = self._parse_resolution_size(settings.get("_midom_requested_resolution") or settings.get("resolution"))
        if requested_size is None:
            raise ValueError("LTX requested delivery resolution is missing or invalid.")
        return requested_size

    def _ltx_internal_size(self, settings: dict[str, Any]) -> tuple[int, int]:
        internal_size = self._parse_resolution_size(settings.get("_midom_ltx_internal_resolution") or settings.get("resolution"))
        if internal_size is None:
            raise ValueError("LTX internal render resolution is missing or invalid.")
        return internal_size

    def _prepare_ltx_overscan_image(
        self,
        source_path: Path,
        requested_size: tuple[int, int],
        internal_size: tuple[int, int],
        label: str,
    ) -> str:
        if requested_size == internal_size:
            return str(source_path)
        if internal_size[0] < requested_size[0] or internal_size[1] < requested_size[1]:
            raise ValueError(
                f"{label} LTX overscan target must be larger than the requested delivery size; "
                f"requested={requested_size[0]}x{requested_size[1]} internal={internal_size[0]}x{internal_size[1]}."
            )
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        if source.size != requested_size:
            raise ValueError(
                f"{label} must match the selected Midom delivery profile before LTX overscan; "
                f"expected {requested_size[0]}x{requested_size[1]}, got {source.size[0]}x{source.size[1]}."
            )
        scale = max(internal_size[0] / requested_size[0], internal_size[1] / requested_size[1])
        background_size = (
            max(internal_size[0], int(round(requested_size[0] * scale))),
            max(internal_size[1], int(round(requested_size[1] * scale))),
        )
        background = source.resize(background_size, Image.Resampling.LANCZOS)
        left = max(0, (background_size[0] - internal_size[0]) // 2)
        top = max(0, (background_size[1] - internal_size[1]) // 2)
        background = background.crop((left, top, left + internal_size[0], top + internal_size[1]))
        paste_x = (internal_size[0] - requested_size[0]) // 2
        paste_y = (internal_size[1] - requested_size[1]) // 2
        background.paste(source, (paste_x, paste_y))
        target_path = source_path.with_name(f"{source_path.stem}-ltx-overscan.png")
        background.save(target_path, format="PNG", optimize=True)
        self._log(
            "Prepared LTX overscan image input; "
            f"label={label!r} source={source_path.name!r} target={target_path.name!r} "
            f"requested={requested_size[0]}x{requested_size[1]} internal={internal_size[0]}x{internal_size[1]}."
        )
        return str(target_path)

    def _prepare_ltx_overscan_control_video(
        self,
        source_path: Path,
        requested_size: tuple[int, int],
        internal_size: tuple[int, int],
        duration_seconds: float,
    ) -> str:
        if requested_size == internal_size:
            return str(source_path)
        if internal_size[0] < requested_size[0] or internal_size[1] < requested_size[1]:
            raise ValueError(
                "LTX control-video overscan target must be larger than the requested delivery size; "
                f"requested={requested_size[0]}x{requested_size[1]} internal={internal_size[0]}x{internal_size[1]}."
            )
        paste_x = (internal_size[0] - requested_size[0]) // 2
        paste_y = (internal_size[1] - requested_size[1]) // 2
        target_path = source_path.with_name(f"{source_path.stem}-ltx-overscan.mp4")
        filter_complex = (
            "[0:v]split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={internal_size[0]}:{internal_size[1]}:force_original_aspect_ratio=increase,"
            f"crop={internal_size[0]}:{internal_size[1]},setsar=1[bg];"
            f"[fgsrc]scale={requested_size[0]}:{requested_size[1]},setsar=1[fg];"
            f"[bg][fg]overlay={paste_x}:{paste_y},format=yuv420p[vout];"
            "[0:a:0]aresample=48000,aformat=channel_layouts=stereo[aout]"
        )
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-r",
            str(LTX_VIDEO_FPS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-metadata:s:v:0",
            "rotate=0",
            str(target_path),
        ]
        timeout_seconds = int(max(120, min(600, float(duration_seconds or 1.0) * 20)))
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
        if completed.returncode != 0:
            stderr = str(completed.stderr or completed.stdout or "").strip()
            raise ValueError(f"LTX control-video overscan preparation failed: {stderr}")
        if not target_path.is_file() or target_path.stat().st_size <= 0:
            raise ValueError("LTX control-video overscan preparation did not produce an MP4.")
        metadata = self._probe_control_video_metadata(target_path)
        prepared_size = (int(metadata["width"]), int(metadata["height"]))
        if prepared_size != internal_size:
            raise ValueError(
                "LTX control-video overscan preparation produced the wrong size; "
                f"expected {internal_size[0]}x{internal_size[1]}, got {prepared_size[0]}x{prepared_size[1]}."
            )
        self._log(
            "Prepared LTX overscan control video input; "
            f"source={source_path.name!r} target={target_path.name!r} "
            f"requested={requested_size[0]}x{requested_size[1]} internal={internal_size[0]}x{internal_size[1]} "
            f"duration_seconds={metadata['video_duration_seconds']:.2f} fps={metadata['fps'] if metadata['fps'] is not None else 'unknown'}."
        )
        return str(target_path)

    def _apply_inputs_to_settings(self, settings: dict[str, Any], downloaded_inputs: list[dict[str, Any]], job: dict[str, Any]) -> None:
        if not downloaded_inputs:
            return
        media_type = str(job.get("media_type") or settings.get("_midom_media_type") or "image").strip().lower()
        if str(settings.get("_midom_media_type") or "").strip().lower() == "media_processing":
            return
        if media_type == "audio":
            voice_mode = str(settings.get("_midom_voice_mode") or ((job.get("generation") or {}).get("voice_mode")) or "single_reference").strip().lower()
            source_inputs = [item for item in downloaded_inputs if item.get("kind") == "source_audio"]
            audio_inputs = [item for item in downloaded_inputs if item.get("kind") == "reference_audio"]
            emotion_inputs = [item for item in downloaded_inputs if item.get("kind") == "emotion_reference_audio"]
            speaker1_inputs = [item for item in downloaded_inputs if item.get("kind") == "speaker1_reference_audio"]
            speaker2_inputs = [item for item in downloaded_inputs if item.get("kind") == "speaker2_reference_audio"]
            model_type = str(settings.get("model_type") or job.get("model_id") or "").strip()
            if model_type == SEEDVC_MODEL_ID:
                if len(source_inputs) != 1:
                    raise ValueError(f"SeedVC job requires exactly one source_audio input; got {len(source_inputs)}.")
                if len(audio_inputs) != 1:
                    raise ValueError(f"SeedVC job requires exactly one reference_audio input; got {len(audio_inputs)}.")
                if emotion_inputs or speaker1_inputs or speaker2_inputs:
                    raise ValueError("SeedVC jobs must not include emotion or speaker reference inputs.")
                settings["audio_source"] = source_inputs[0]["path"]
                settings["replace_voice_sample"] = audio_inputs[0]["path"]
                settings["replace_voice_sample2"] = None
                settings["postprocess_audio"] = SEEDVC_METHOD_ONE_SPEAKER
                self._log(
                    "Applied SeedVC audio inputs to WanGP settings; "
                    f"source_audio={Path(settings['audio_source']).name!r} "
                    f"reference_audio={Path(settings['replace_voice_sample']).name!r} "
                    f"postprocess_audio={settings.get('postprocess_audio')!r}."
                )
            elif voice_mode == "two_voice_clone_n_voice_dialogue":
                if len(speaker1_inputs) != 1 or len(speaker2_inputs) != 1:
                    raise ValueError("DramaBox job requires exactly one reference audio input for each voice anchor.")
                if source_inputs or audio_inputs or emotion_inputs:
                    raise ValueError("DramaBox jobs must not include reference_audio or emotion_reference_audio inputs.")
                settings["audio_guide"] = speaker1_inputs[0]["path"]
                settings["audio_guide2"] = speaker2_inputs[0]["path"]
                if settings.get("audio_prompt_type") not in {"AB", "AB0"}:
                    settings["audio_prompt_type"] = "AB"
            elif voice_mode == "two_speaker_dialogue":
                if len(speaker1_inputs) != 1 or len(speaker2_inputs) != 1:
                    raise ValueError("Dialogue job requires exactly one reference audio input for each speaker.")
                if source_inputs:
                    raise ValueError("Dialogue jobs must not include source_audio inputs.")
                settings["audio_guide"] = speaker1_inputs[0]["path"]
                settings["audio_guide2"] = speaker2_inputs[0]["path"]
                settings["audio_prompt_type"] = "AB2"
            else:
                if source_inputs:
                    raise ValueError("TTS audio jobs must not include source_audio inputs.")
                if len(audio_inputs) != 1:
                    raise ValueError(f"Audio job requires exactly one reference_audio input; got {len(audio_inputs)}.")
                if len(emotion_inputs) > 1:
                    raise ValueError(f"Audio job supports at most one emotion_reference_audio input; got {len(emotion_inputs)}.")
                settings["audio_guide"] = audio_inputs[0]["path"]
                if model_type == CHATTERBOX_MODEL_ID:
                    if emotion_inputs:
                        raise ValueError("Chatterbox does not support emotion_reference_audio inputs.")
                    settings["audio_guide2"] = None
                    settings["audio_prompt_type"] = "A"
                elif emotion_inputs:
                    settings["audio_guide2"] = emotion_inputs[0]["path"]
                    settings["audio_prompt_type"] = "AB"
                else:
                    settings["audio_guide2"] = None
                    settings["audio_prompt_type"] = "A"
            if model_type != SEEDVC_MODEL_ID:
                self._log(
                    "Applied reference audio to WanGP settings; "
                    f"model_id={settings.get('model_type')} voice_mode={voice_mode!r} "
                    f"audio_prompt_type={settings.get('audio_prompt_type')!r} "
                    f"audio_guide={Path(settings['audio_guide']).name!r} "
                    f"audio_guide2={(Path(settings['audio_guide2']).name if settings.get('audio_guide2') else None)!r}."
                )
            return
        if media_type == "video":
            model_type = str(settings.get("model_type") or job.get("model_id") or "").strip()
            driving_audio_inputs = [item for item in downloaded_inputs if item.get("kind") == "driving_audio"]
            if model_type == SVI_VIDEO_MODEL_ID:
                start_image_inputs = [item for item in downloaded_inputs if item.get("kind") == "start_image"]
                end_image_inputs = [item for item in downloaded_inputs if item.get("kind") == "end_image"]
                if len(start_image_inputs) != 1:
                    raise ValueError(f"SVI video job requires exactly one start_image input; got {len(start_image_inputs)}.")
                if len(end_image_inputs) > 1:
                    raise ValueError(f"SVI video job supports at most one end_image input; got {len(end_image_inputs)}.")
                if driving_audio_inputs:
                    raise ValueError(f"SVI video job does not support driving_audio inputs; got {len(driving_audio_inputs)}.")
                settings["image_start"] = [start_image_inputs[0]["path"]]
                settings["image_end"] = [end_image_inputs[0]["path"]] if end_image_inputs else None
                settings["audio_guide"] = None
                settings["audio_guide2"] = None
                settings["image_prompt_type"] = "SE" if end_image_inputs else "S"
                settings["video_prompt_type"] = ""
                settings["audio_prompt_type"] = ""
                self._log(
                    "Applied SVI cinematic I2V input to WanGP settings; "
                    f"model_id={model_type} "
                    f"image_start={Path(start_image_inputs[0]['path']).name!r} "
                    f"image_end={(Path(end_image_inputs[0]['path']).name if end_image_inputs else None)!r} "
                    f"image_prompt_type={settings.get('image_prompt_type')!r} "
                    f"video_prompt_type={settings.get('video_prompt_type')!r} "
                    f"audio_prompt_type={settings.get('audio_prompt_type')!r}."
                )
                return
            if model_type in LTX_VIDEO_MODEL_IDS:
                start_image_inputs = [item for item in downloaded_inputs if item.get("kind") == "start_image"]
                end_image_inputs = [item for item in downloaded_inputs if item.get("kind") == "end_image"]
                control_video_inputs = [item for item in downloaded_inputs if item.get("kind") == "control_video"]
                if settings.get("_midom_video_task") == "control_video_guided_video":
                    if len(start_image_inputs) != 1:
                        raise ValueError(f"LTX control-video job requires exactly one start_image input; got {len(start_image_inputs)}.")
                    if len(control_video_inputs) != 1:
                        raise ValueError(f"LTX control-video job requires exactly one control_video input; got {len(control_video_inputs)}.")
                    if driving_audio_inputs:
                        raise ValueError(f"LTX control-video job must not include driving_audio inputs; got {len(driving_audio_inputs)}.")
                    if end_image_inputs:
                        raise ValueError(f"LTX control-video first pass does not support end_image inputs; got {len(end_image_inputs)}.")
                    control_path = Path(control_video_inputs[0]["path"])
                    metadata = self._probe_control_video_metadata(control_path)
                    requested_size = self._ltx_requested_size(settings)
                    internal_size = self._ltx_internal_size(settings)
                    actual_control_size = (int(metadata["width"]), int(metadata["height"]))
                    if actual_control_size != requested_size:
                        raise ValueError(
                            "LTX control-video input must match the selected Midom delivery profile exactly; "
                            f"expected {requested_size[0]}x{requested_size[1]}, got {actual_control_size[0]}x{actual_control_size[1]}."
                        )
                    control_fps = metadata.get("fps")
                    if control_fps is None or abs(float(control_fps) - float(LTX_VIDEO_FPS)) > 0.05:
                        raise ValueError(
                            f"LTX control-video input must be constant {LTX_VIDEO_FPS} fps; "
                            f"got {control_fps if control_fps is not None else 'unknown'} fps."
                        )
                    start_path = Path(start_image_inputs[0]["path"])
                    self._validate_image_input_exact_size(
                        start_path,
                        requested_size,
                        "LTX control-video start_image",
                    )
                    max_available_duration = min(
                        float(metadata["video_duration_seconds"]),
                        float(metadata["audio_duration_seconds"]),
                        float(MAX_LTX_VIDEO_DURATION_SECONDS),
                    )
                    if max_available_duration < 1.0:
                        raise ValueError("Control video must contain at least one second of overlapping video and audio.")
                    requested_duration = self._coerce_int(settings.get("duration_seconds"), MAX_LTX_VIDEO_DURATION_SECONDS, 1, MAX_LTX_VIDEO_DURATION_SECONDS)
                    effective_duration = max(1, min(requested_duration, int(max_available_duration)))
                    if effective_duration != requested_duration:
                        self._log(
                            "Adjusted LTX control-video duration to downloaded media bounds; "
                            f"requested={requested_duration} effective={effective_duration} "
                            f"video_duration={metadata['video_duration_seconds']:.2f} "
                            f"audio_duration={metadata['audio_duration_seconds']:.2f}."
                        )
                    settings["duration_seconds"] = effective_duration
                    settings["video_length"] = self._ltx_video_length_for_duration(effective_duration)
                    overscan_start_path = self._prepare_ltx_overscan_image(
                        start_path,
                        requested_size,
                        internal_size,
                        "LTX control-video start_image",
                    )
                    overscan_control_path = self._prepare_ltx_overscan_control_video(
                        control_path,
                        requested_size,
                        internal_size,
                        float(metadata["video_duration_seconds"]),
                    )
                    settings["image_start"] = [overscan_start_path]
                    settings["image_end"] = None
                    settings["video_guide"] = overscan_control_path
                    settings["_midom_input_video_paths"] = [
                        str(control_path.resolve()),
                        str(Path(overscan_control_path).resolve()),
                    ]
                    settings["_midom_input_video_sha256s"] = [str(control_video_inputs[0].get("sha256") or "").strip().lower()]
                    settings["audio_guide"] = None
                    settings["audio_guide2"] = None
                    settings["image_prompt_type"] = "S"
                    settings["audio_prompt_type"] = "K"
                    control_mode = str(settings.get("_midom_control_video_mode") or "")
                    settings["video_prompt_type"] = LTX_CONTROL_VIDEO_MODES.get(control_mode, settings.get("video_prompt_type") or "VG")
                    self._log(
                        "Applied LTX control-video inputs to WanGP settings; "
                        f"model_id={model_type} "
                        f"image_start={Path(overscan_start_path).name!r} "
                        f"video_guide={Path(overscan_control_path).name!r} "
                        f"control_mode={control_mode!r} "
                        f"requested_size={requested_size[0]}x{requested_size[1]} "
                        f"internal_size={internal_size[0]}x{internal_size[1]} "
                        f"control_video_size={metadata['width']}x{metadata['height']} "
                        f"control_video_fps={metadata['fps'] if metadata['fps'] is not None else 'unknown'} "
                        f"duration_seconds={settings.get('duration_seconds')} "
                        f"video_length={settings.get('video_length')} "
                        f"image_prompt_type={settings.get('image_prompt_type')!r} "
                        f"video_prompt_type={settings.get('video_prompt_type')!r} "
                        f"audio_prompt_type={settings.get('audio_prompt_type')!r}."
                    )
                    return
                if len(start_image_inputs) != 1:
                    raise ValueError(f"LTX video job requires exactly one start_image input; got {len(start_image_inputs)}.")
                if len(end_image_inputs) > 1:
                    raise ValueError(f"LTX video job supports at most one end_image input; got {len(end_image_inputs)}.")
                if len(driving_audio_inputs) != 1:
                    raise ValueError(f"LTX video job requires exactly one driving_audio input; got {len(driving_audio_inputs)}.")
                requested_size = self._ltx_requested_size(settings)
                internal_size = self._ltx_internal_size(settings)
                start_path = Path(start_image_inputs[0]["path"])
                self._validate_image_input_exact_size(
                    start_path,
                    requested_size,
                    "LTX audio-guided start_image",
                )
                overscan_start_path = self._prepare_ltx_overscan_image(
                    start_path,
                    requested_size,
                    internal_size,
                    "LTX audio-guided start_image",
                )
                overscan_end_path = None
                if end_image_inputs:
                    end_path = Path(end_image_inputs[0]["path"])
                    self._validate_image_input_exact_size(
                        end_path,
                        requested_size,
                        "LTX audio-guided end_image",
                    )
                    overscan_end_path = self._prepare_ltx_overscan_image(
                        end_path,
                        requested_size,
                        internal_size,
                        "LTX audio-guided end_image",
                    )
                settings["image_start"] = [overscan_start_path]
                settings["image_end"] = [overscan_end_path] if overscan_end_path else None
                settings["audio_guide"] = driving_audio_inputs[0]["path"]
                settings["audio_guide2"] = None
                settings["image_prompt_type"] = "SE" if end_image_inputs else "S"
                settings["video_prompt_type"] = ""
                settings["audio_prompt_type"] = "A"
                self._log(
                    "Applied LTX video inputs to WanGP settings; "
                    f"model_id={model_type} "
                    f"image_start={Path(overscan_start_path).name!r} "
                    f"image_end={(Path(overscan_end_path).name if overscan_end_path else None)!r} "
                    f"audio_guide={Path(driving_audio_inputs[0]['path']).name!r} "
                    f"requested_size={requested_size[0]}x{requested_size[1]} "
                    f"internal_size={internal_size[0]}x{internal_size[1]} "
                    f"image_prompt_type={settings.get('image_prompt_type')!r} "
                    f"video_prompt_type={settings.get('video_prompt_type')!r} "
                    f"audio_prompt_type={settings.get('audio_prompt_type')!r}."
                )
                return
            reference_inputs = [item for item in downloaded_inputs if item.get("kind") == "reference_image"]
            if len(reference_inputs) != 1:
                raise ValueError(f"LongCat video job requires exactly one reference_image input; got {len(reference_inputs)}.")
            if len(driving_audio_inputs) != 1:
                raise ValueError(f"LongCat video job requires exactly one driving_audio input; got {len(driving_audio_inputs)}.")
            settings["image_refs"] = [reference_inputs[0]["path"]]
            settings["audio_guide"] = driving_audio_inputs[0]["path"]
            settings["audio_guide2"] = None
            settings["video_prompt_type"] = "KI"
            settings["audio_prompt_type"] = "A"
            self._log(
                "Applied LongCat video inputs to WanGP settings; "
                f"model_id={settings.get('model_type')} "
                f"image_ref={Path(reference_inputs[0]['path']).name!r} "
                f"audio_guide={Path(driving_audio_inputs[0]['path']).name!r} "
                f"video_prompt_type={settings.get('video_prompt_type')!r} "
                f"audio_prompt_type={settings.get('audio_prompt_type')!r}."
            )
            return
        model_id = str(job.get("model_id") or settings.get("model_type") or "")
        model_meta = MODEL_CAPABILITY_OVERRIDES.get(model_id) or {}
        paths = [item["path"] for item in downloaded_inputs if item.get("kind") == "reference_image"]
        control_inputs = [item for item in downloaded_inputs if item.get("kind") == "control_image"]
        if paths and not model_meta.get("image_reference"):
            raise ValueError(f"Model {model_id} does not support ordinary reference images.")
        if control_inputs and not model_meta.get("control"):
            raise ValueError(f"Model {model_id} does not support control images.")
        if model_id == "qwen_image_layered_20B":
            if paths:
                raise ValueError("Qwen Image Layered does not support ordinary reference images.")
            if len(control_inputs) != 1:
                raise ValueError("Qwen Image Layered requires exactly one Control Image.")
            control_mode = str(control_inputs[0].get("control_mode") or "raw")
            if control_mode != "raw":
                raise ValueError(f"Unsupported Qwen Image Layered control_mode: {control_mode}")
            guide_path = control_inputs[0]["path"]
            settings["image_guide"] = self._load_image_guide(guide_path)
            settings["video_prompt_type"] = self._control_prompt_type_for_model(model_id, control_mode)
            self._log(
                f"Applied Qwen layered image guide settings; model_id={model_id} "
                f"reference_count={len(paths)} control_count={len(control_inputs)} "
                f"control_mode={control_mode!r} "
                f"image_guide={Path(guide_path).name!r} video_prompt_type={settings['video_prompt_type']!r}."
            )
        elif model_meta.get("control") and control_inputs:
            control_input = control_inputs[0]
            control_mode = str(control_input.get("control_mode") or "raw")
            settings["image_guide"] = self._load_image_guide(control_input["path"])
            settings["video_prompt_type"] = self._control_prompt_type_for_model(model_id, control_mode)
            if paths and model_id.startswith("qwen_image_"):
                settings["image_refs"] = paths
                settings["video_prompt_type"] = _add_prompt_type_letter(settings["video_prompt_type"], "I")
            self._log(
                f"Applied explicit control image settings; model_id={model_id} "
                f"reference_count={len(paths)} control_count={len(control_inputs)} "
                f"control_mode={control_mode!r} "
                f"image_guide={Path(control_input['path']).name!r} "
                f"image_refs={[Path(path).name for path in paths] if paths else []} "
                f"video_prompt_type={settings['video_prompt_type']!r}."
            )
        elif model_meta.get("control") and paths:
            settings["image_guide"] = self._load_image_guide(paths[0])
            settings["video_prompt_type"] = self._control_prompt_type_for_model(model_id, "raw")
            self._log(
                "Applied fallback raw control image settings from reference input; "
                f"model_id={model_id} image_guide={Path(paths[0]).name!r} "
                f"video_prompt_type={settings['video_prompt_type']!r}."
            )
        else:
            settings["image_refs"] = paths
            settings["video_prompt_type"] = self._reference_prompt_type_for_model(model_id)
            self._log(
                "Applied reference images to WanGP settings; "
                f"model_id={model_id} reference_count={len(paths)} "
                f"image_refs={[Path(path).name for path in paths]} "
                f"video_prompt_type={settings['video_prompt_type']!r}."
            )
        if str(settings.get("_midom_curated_tool_id") or "") == QWEN_MULTI_ANGLE_TOOL_ID:
            self._log(
                "Applied curated Qwen multi-angle tool settings; "
                f"tool_id={QWEN_MULTI_ANGLE_TOOL_ID} "
                f"parameters={settings.get('_midom_curated_tool_parameters')} "
                f"lora={QWEN_MULTI_ANGLE_LORA_FILENAME!r} "
                f"lora_sha256={QWEN_MULTI_ANGLE_LORA_SHA256[:12]}... "
                f"lora_multiplier={settings.get('_midom_curated_tool_lora_multiplier')} "
                f"expanded_prompt={settings.get('_midom_curated_tool_expanded_prompt')!r}."
            )

    def _load_image_guide(self, path: str) -> Image.Image:
        with Image.open(path) as image:
            loaded = image.convert("RGB").copy()
        self._log(f"Loaded control image for WanGP image_guide; filename={Path(path).name!r} size={loaded.size}.")
        return loaded

    @staticmethod
    def _reference_prompt_type_for_model(model_id: str) -> str:
        if model_id == "qwen_image_edit_20B":
            return "KI"
        return "I"

    def _control_prompt_type_for_model(self, model_id: str, control_mode: Any) -> str:
        mode_id = str(control_mode or "raw").strip()
        supported_modes = {item["mode_id"] for item in self._control_modes_for_model(model_id)}
        if mode_id not in supported_modes:
            raise ValueError(f"Unsupported control mode for {model_id}: {mode_id}")
        return CONTROL_MODE_WANGP_CODES[mode_id]

    def _post_job_update(self, connection: ConnectionContext, job_id: str, status: str, payload: dict[str, Any]) -> None:
        self._ensure_job_flow_enabled()
        job_id_int = int(job_id)
        if job_id_int <= 0:
            raise ValueError("Job id is required.")
        if status not in {"progress", "complete", "fail"}:
            raise ValueError(f"Unsupported job update status: {status}")
        route_status = "complete" if status == "complete" else status
        self._log(f"Posting job update; job_id={job_id_int} status={status} payload_keys={sorted(payload.keys())}.")
        response = requests.post(
            f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{job_id_int}/{route_status}",
            headers={**self._headers(connection), "Content-Type": "application/json"},
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code in {401, 403}:
            self._handle_authorization_failure(f"Job {status} update was rejected by Midom authorization.")
        response.raise_for_status()
        self._log(f"Job update accepted; job_id={job_id_int} status={status} http_status={response.status_code}.")
        if status == "progress":
            response_payload = response.json() if response.content else {}
            if isinstance(response_payload, dict) and response_payload.get("cancel_requested"):
                self._cancel_requested_by_midom = True
                self._set_active_job_status(
                    phase="canceling",
                    status="Midom cancellation received; waiting for WanGP to stop the local generation.",
                    progress=(self._active_job_status or {}).get("progress"),
                )
                self._log(f"Midom requested cancellation in progress response; job_id={job_id_int}.", force=True)
                job = self._active_job
                if job is not None and not getattr(job, "done", True):
                    try:
                        job.cancel()
                    except Exception as exc:
                        self._log(f"Could not cancel WanGP job after Midom cancellation request: {exc}", force=True)

    def _upload_artifact(
        self,
        connection: ConnectionContext,
        job_id: int,
        file_path: str,
        artifact_index: int,
        expected_resolution: Any = None,
    ) -> dict[str, Any]:
        self._ensure_job_flow_enabled()
        path = Path(file_path)
        if not path.is_file():
            raise ValueError("Generated artifact path does not exist.")
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported generated artifact extension: {suffix}")
        file_size = path.stat().st_size
        if file_size <= 0 or file_size > MAX_IMAGE_BYTES:
            raise ValueError(f"Generated artifact size is outside allowed bounds: {file_size} bytes")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported generated artifact mime type: {mime_type}")
        decoded_width = decoded_height = None
        decoded_format = ""
        normalized_temp_path = None
        try:
            with Image.open(path) as image:
                decoded_width, decoded_height = image.size
                decoded_format = str(image.format or "").upper()
        except Exception as exc:
            raise ValueError(f"Generated artifact could not be decoded as an image: {exc}")
        expected_size = self._parse_resolution_size(expected_resolution)
        if expected_size is not None and (decoded_width, decoded_height) != expected_size:
            normalized_temp_path = self._normalize_artifact_dimensions(path, expected_size, mime_type)
            self._log(
                "Normalized generated artifact dimensions before upload; "
                f"job_id={job_id} artifact_index={artifact_index} "
                f"original={decoded_width}x{decoded_height} expected={expected_size[0]}x{expected_size[1]} "
                f"normalized_file={normalized_temp_path.name!r}.",
                force=True,
            )
            path = normalized_temp_path
            file_size = path.stat().st_size
            with Image.open(path) as image:
                decoded_width, decoded_height = image.size
                decoded_format = str(image.format or "").upper()
        with path.open("rb") as reader:
            data = reader.read()
        sha256 = hashlib.sha256(data).hexdigest()
        self._log(
            f"Uploading artifact; job_id={job_id} artifact_index={artifact_index} "
            f"filename={path.name!r} mime_type={mime_type} decoded_format={decoded_format!r} "
            f"dimensions={decoded_width}x{decoded_height} bytes={file_size} sha256={sha256[:12]}..."
        )
        try:
            with path.open("rb") as reader:
                response = requests.post(
                    f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{int(job_id)}/artifacts",
                    headers=self._headers(connection),
                    data={
                        "artifact_index": str(int(artifact_index)),
                        "sha256": sha256,
                        "mime_type": mime_type,
                        "filename": path.name,
                    },
                    files={"file": (path.name, reader, mime_type)},
                    timeout=UPLOAD_TIMEOUT_SECONDS,
                )
        finally:
            if normalized_temp_path is not None:
                try:
                    normalized_temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        if response.status_code >= 400:
            message = self._midom_error_message(response)
            self._log(
                f"Artifact upload rejected by Midom; job_id={job_id} artifact_index={artifact_index} "
                f"http_status={response.status_code} message={message!r}.",
                force=True,
            )
            raise ValueError(f"Midom rejected artifact upload with HTTP {response.status_code}: {message}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Artifact upload response was not a JSON object.")
        self._log(
            f"Artifact upload accepted; job_id={job_id} artifact_index={payload.get('artifact_index', artifact_index)} "
            f"artifact_id={payload.get('artifact_id')} file_id={payload.get('file_id')}."
        )
        return {
            "artifact_id": int(payload.get("artifact_id")),
            "file_id": int(payload.get("file_id")),
            "artifact_index": int(payload.get("artifact_index", artifact_index)),
        }

    def _upload_audio_artifact(
        self,
        connection: ConnectionContext,
        job_id: int,
        file_path: str,
        artifact_index: int,
        settings: dict[str, Any],
        temp_dir: str,
    ) -> dict[str, Any]:
        self._ensure_job_flow_enabled()
        if int(artifact_index) != 0:
            raise ValueError("Audio v1 supports only artifact_index 0.")
        requested_format = str(settings.get("_midom_output_format") or "mp3").strip().lower()
        upload_path, mime_type = self._prepare_audio_artifact_for_upload(file_path, requested_format, temp_dir)
        path = Path(upload_path)
        file_size = path.stat().st_size
        max_bytes = self._coerce_int(settings.get("_midom_max_artifact_bytes"), MAX_AUDIO_BYTES, 1, MAX_AUDIO_BYTES)
        if file_size <= 0 or file_size > max_bytes:
            raise ValueError(f"Generated audio artifact size is outside allowed bounds: {file_size} bytes")
        with path.open("rb") as reader:
            data = reader.read()
        sha256 = hashlib.sha256(data).hexdigest()
        self._log(
            f"Uploading audio artifact; job_id={job_id} artifact_index={artifact_index} "
            f"filename={path.name!r} requested_format={requested_format} mime_type={mime_type} "
            f"bytes={file_size} sha256={sha256[:12]}..."
        )
        with path.open("rb") as reader:
            response = requests.post(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{int(job_id)}/artifacts",
                headers=self._headers(connection),
                data={
                    "artifact_index": str(int(artifact_index)),
                    "sha256": sha256,
                    "mime_type": mime_type,
                    "filename": path.name,
                },
                files={"file": (path.name, reader, mime_type)},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        if response.status_code >= 400:
            message = self._midom_error_message(response)
            self._log(
                f"Audio artifact upload rejected by Midom; job_id={job_id} artifact_index={artifact_index} "
                f"http_status={response.status_code} message={message!r}.",
                force=True,
            )
            raise ValueError(f"Midom rejected audio artifact upload with HTTP {response.status_code}: {message}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Audio artifact upload response was not a JSON object.")
        self._log(
            f"Audio artifact upload accepted; job_id={job_id} artifact_index={payload.get('artifact_index', artifact_index)} "
            f"artifact_id={payload.get('artifact_id')} file_id={payload.get('file_id')}."
        )
        return {
            "artifact_id": int(payload.get("artifact_id")),
            "file_id": int(payload.get("file_id")),
            "artifact_index": int(payload.get("artifact_index", artifact_index)),
        }

    def _upload_video_artifact(
        self,
        connection: ConnectionContext,
        job_id: int,
        file_path: str,
        artifact_index: int,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        self._ensure_job_flow_enabled()
        if int(artifact_index) != 0:
            raise ValueError("Video v1 supports only artifact_index 0.")
        path = Path(file_path)
        if not path.is_file():
            raise ValueError("Generated video artifact path does not exist.")
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported generated video artifact extension: {suffix}")
        if settings.get("_midom_ltx_delivery_adapter") == "overscan_center_crop":
            path = Path(self._prepare_ltx_video_artifact_for_upload(path, settings, job_id, artifact_index))
            suffix = path.suffix.lower()
            if suffix not in ALLOWED_VIDEO_SUFFIXES:
                raise ValueError(f"Unsupported prepared LTX video artifact extension: {suffix}")
        file_size = path.stat().st_size
        max_bytes = self._coerce_int(settings.get("_midom_max_artifact_bytes"), MAX_VIDEO_BYTES, 1, MAX_VIDEO_BYTES)
        if file_size <= 0 or file_size > max_bytes:
            raise ValueError(f"Generated video artifact size is outside allowed bounds: {file_size} bytes")
        mime_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        if mime_type not in ALLOWED_VIDEO_OUTPUT_MIME_TYPES:
            raise ValueError(f"Unsupported generated video artifact MIME type: {mime_type}")
        with path.open("rb") as reader:
            data = reader.read()
        if not self._looks_like_mp4(data):
            raise ValueError("Generated video artifact does not look like MP4 bytes.")
        expected_size = self._parse_resolution_size(settings.get("_midom_requested_resolution") or settings.get("resolution"))
        if expected_size is not None:
            self._validate_generated_video_resolution(path, expected_size, job_id, artifact_index)
        sha256 = hashlib.sha256(data).hexdigest()
        self._log(
            f"Uploading video artifact; job_id={job_id} artifact_index={artifact_index} "
            f"filename={path.name!r} mime_type={mime_type} bytes={file_size} sha256={sha256[:12]}..."
        )
        with path.open("rb") as reader:
            response = requests.post(
                f"{connection.api_base_url}/b1/media-workers/{connection.worker_id}/jobs/{int(job_id)}/artifacts",
                headers=self._headers(connection),
                data={
                    "artifact_index": str(int(artifact_index)),
                    "sha256": sha256,
                    "mime_type": "video/mp4",
                    "filename": path.name,
                },
                files={"file": (path.name, reader, "video/mp4")},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        if response.status_code >= 400:
            message = self._midom_error_message(response)
            self._log(
                f"Video artifact upload rejected by Midom; job_id={job_id} artifact_index={artifact_index} "
                f"http_status={response.status_code} message={message!r}.",
                force=True,
            )
            raise ValueError(f"Midom rejected video artifact upload with HTTP {response.status_code}: {message}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Video artifact upload response was not a JSON object.")
        self._log(
            f"Video artifact upload accepted; job_id={job_id} artifact_index={payload.get('artifact_index', artifact_index)} "
            f"artifact_id={payload.get('artifact_id')} file_id={payload.get('file_id')}."
        )
        return {
            "artifact_id": int(payload.get("artifact_id")),
            "file_id": int(payload.get("file_id")),
            "artifact_index": int(payload.get("artifact_index", artifact_index)),
        }

    @staticmethod
    def _looks_like_mp4(data: bytes) -> bool:
        if len(data) < 12:
            return False
        return data[4:8] == b"ftyp" or b"ftyp" in data[:64]

    def _prepare_ltx_video_artifact_for_upload(
        self,
        source_path: Path,
        settings: dict[str, Any],
        job_id: int,
        artifact_index: int,
    ) -> str:
        requested_size = self._ltx_requested_size(settings)
        internal_size = self._ltx_internal_size(settings)
        info = self._probe_video_display_info(source_path)
        display_size = info["display_size"]
        if display_size == requested_size:
            self._log(
                "Generated LTX artifact already matches Midom delivery size; "
                f"job_id={job_id} artifact_index={artifact_index} filename={source_path.name!r} "
                f"size={display_size[0]}x{display_size[1]}."
            )
            return str(source_path)
        if display_size != internal_size:
            raise ValueError(
                "Generated LTX artifact does not match the expected internal overscan size; "
                f"expected {internal_size[0]}x{internal_size[1]} before crop, got {display_size[0]}x{display_size[1]}."
            )
        crop_x = (internal_size[0] - requested_size[0]) // 2
        crop_y = (internal_size[1] - requested_size[1]) // 2
        target_file = tempfile.NamedTemporaryFile(prefix="midom-ltx-cropped-", suffix=".mp4", delete=False)
        target_path = Path(target_file.name)
        target_file.close()
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            f"crop={requested_size[0]}:{requested_size[1]}:{crop_x}:{crop_y},setsar=1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-metadata:s:v:0",
            "rotate=0",
            str(target_path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            stderr = str(completed.stderr or completed.stdout or "").strip()
            raise ValueError(f"LTX generated video crop to Midom delivery size failed: {stderr}")
        if not target_path.is_file() or target_path.stat().st_size <= 0:
            raise ValueError("LTX generated video crop did not produce an MP4.")
        self._validate_generated_video_resolution(target_path, requested_size, job_id, artifact_index)
        self._log(
            "Cropped LTX overscan video artifact to Midom delivery size; "
            f"job_id={job_id} artifact_index={artifact_index} source={source_path.name!r} "
            f"target={target_path.name!r} internal={internal_size[0]}x{internal_size[1]} "
            f"delivery={requested_size[0]}x{requested_size[1]} crop={crop_x}:{crop_y}."
        )
        return str(target_path)

    def _probe_video_display_info(self, path: Path) -> dict[str, Any]:
        command = [
            self._ffprobe_binary(),
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"Could not inspect generated video resolution: {exc}")
        if completed.returncode != 0:
            stderr = str(completed.stderr or completed.stdout or "").strip()
            raise ValueError(f"Could not inspect generated video resolution: {stderr}")
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse generated video resolution metadata: {exc}")
        streams = payload.get("streams") or []
        if not isinstance(streams, list):
            streams = []
        video_stream = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"), None)
        if not isinstance(video_stream, dict):
            raise ValueError("Generated video artifact does not contain a video stream.")
        width = self._coerce_int(video_stream.get("width"), 0, 0, 100_000)
        height = self._coerce_int(video_stream.get("height"), 0, 0, 100_000)
        if width <= 0 or height <= 0:
            raise ValueError("Generated video artifact width/height could not be read.")
        rotation = self._video_rotation_degrees(video_stream)
        display_size = (height, width) if int(abs(rotation)) % 180 == 90 else (width, height)
        return {
            "stored_size": (width, height),
            "display_size": display_size,
            "rotation_degrees": rotation,
        }

    def _validate_generated_video_resolution(
        self,
        path: Path,
        expected_size: tuple[int, int],
        job_id: int,
        artifact_index: int,
    ) -> None:
        info = self._probe_video_display_info(path)
        width, height = info["stored_size"]
        display_size = info["display_size"]
        rotation = info["rotation_degrees"]
        if display_size != expected_size:
            if self._is_midom_repairable_video_stride_size(display_size, expected_size):
                self._log(
                    "Generated video artifact uses a Midom-repairable model stride size; "
                    "uploading raw output so Midom can create the official requested-size MP4. "
                    f"job_id={job_id} artifact_index={artifact_index} filename={path.name!r} "
                    f"display_size={display_size[0]}x{display_size[1]} expected={expected_size[0]}x{expected_size[1]}.",
                    force=True,
                )
                return
            raise ValueError(
                "Generated video artifact dimensions do not match the selected output profile; "
                f"expected {expected_size[0]}x{expected_size[1]}, got {display_size[0]}x{display_size[1]}."
            )
        self._log(
            "Validated generated video artifact dimensions; "
            f"job_id={job_id} artifact_index={artifact_index} filename={path.name!r} "
            f"stored_size={width}x{height} display_size={display_size[0]}x{display_size[1]} "
            f"expected={expected_size[0]}x{expected_size[1]} rotation_degrees={rotation}."
        )

    @staticmethod
    def _is_midom_repairable_video_stride_size(
        actual_size: tuple[int, int],
        expected_size: tuple[int, int],
    ) -> bool:
        actual_width, actual_height = actual_size
        expected_width, expected_height = expected_size
        if min(actual_width, actual_height, expected_width, expected_height) <= 0:
            return False
        if (actual_height > actual_width) != (expected_height > expected_width):
            return False
        allowed_pairs = {
            ((704, 1280), (720, 1280)),
            ((1280, 704), (1280, 720)),
        }
        if (actual_size, expected_size) in allowed_pairs:
            return True
        stride = 64
        nearest_lower_width = max(stride, (expected_width // stride) * stride)
        nearest_lower_height = max(stride, (expected_height // stride) * stride)
        return (
            (actual_width, actual_height) == (nearest_lower_width, expected_height)
            and nearest_lower_width < expected_width
        ) or (
            (actual_width, actual_height) == (expected_width, nearest_lower_height)
            and nearest_lower_height < expected_height
        )

    def _prepare_audio_artifact_for_upload(self, file_path: str, requested_format: str, temp_dir: str) -> tuple[str, str]:
        source_path = Path(file_path)
        if not source_path.is_file():
            raise ValueError("Generated audio artifact path does not exist.")
        suffix = source_path.suffix.lower()
        if suffix not in ALLOWED_AUDIO_OUTPUT_SUFFIXES:
            raise ValueError(f"Unsupported generated audio artifact extension: {suffix}")
        requested_format = str(requested_format or "mp3").strip().lower()
        if requested_format == "mp3":
            if suffix == ".mp3":
                self._log(f"Generated audio already matches requested MP3 format; filename={source_path.name!r}.")
                return str(source_path), "audio/mpeg"
            target_path = Path(temp_dir) / f"{source_path.stem}.mp3"
            self._transcode_audio(source_path, target_path, "mp3")
            self._log(
                f"Transcoded generated audio to MP3 for Midom upload; "
                f"source={source_path.name!r} source_bytes={source_path.stat().st_size} "
                f"target={target_path.name!r} target_bytes={target_path.stat().st_size}."
            )
            return str(target_path), "audio/mpeg"
        if requested_format == "wav":
            if suffix == ".wav":
                self._log(f"Generated audio already matches requested WAV format; filename={source_path.name!r}.")
                return str(source_path), "audio/wav"
            target_path = Path(temp_dir) / f"{source_path.stem}.wav"
            self._transcode_audio(source_path, target_path, "wav")
            self._log(
                f"Transcoded generated audio to WAV for Midom upload; "
                f"source={source_path.name!r} source_bytes={source_path.stat().st_size} "
                f"target={target_path.name!r} target_bytes={target_path.stat().st_size}."
            )
            return str(target_path), "audio/wav"
        raise ValueError(f"Unsupported requested audio format: {requested_format}")

    def _transcode_audio(self, source_path: Path, target_path: Path, requested_format: str) -> None:
        try:
            from shared.utils.audio_video import _ffmpeg_binary

            binary = _ffmpeg_binary()
        except Exception as exc:
            raise ValueError(f"Could not resolve WanGP ffmpeg binary for audio transcode: {exc}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if requested_format == "mp3":
            codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]
        elif requested_format == "wav":
            codec_args = ["-c:a", "pcm_s16le"]
        else:
            raise ValueError(f"Unsupported audio transcode format: {requested_format}")
        cmd = [
            binary,
            "-y",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-vn",
            *codec_args,
            str(target_path),
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = str(completed.stderr or completed.stdout or "").strip()
            raise ValueError(f"Audio transcode to {requested_format} failed: {stderr}")
        if not target_path.is_file() or target_path.stat().st_size <= 0:
            raise ValueError(f"Audio transcode to {requested_format} did not produce an output file.")

    @staticmethod
    def _parse_resolution_size(resolution: Any) -> Optional[tuple[int, int]]:
        text = str(resolution or "").strip().lower()
        match = re.fullmatch(r"(\d+)x(\d+)", text)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _normalize_artifact_dimensions(self, path: Path, expected_size: tuple[int, int], mime_type: str) -> Path:
        suffix = MIME_EXTENSION.get(mime_type, path.suffix.lower() or ".png")
        temp_file = tempfile.NamedTemporaryFile(prefix="midom-normalized-", suffix=suffix, delete=False)
        normalized_path = Path(temp_file.name)
        temp_file.close()
        with Image.open(path) as image:
            normalized = image.convert("RGB") if mime_type == "image/jpeg" else image.convert("RGBA")
            normalized = normalized.resize(expected_size, Image.Resampling.LANCZOS)
            if mime_type == "image/jpeg":
                normalized.save(normalized_path, format="JPEG", quality=95, optimize=True)
            elif mime_type == "image/webp":
                normalized.save(normalized_path, format="WEBP", quality=95, method=6)
            else:
                normalized.save(normalized_path, format="PNG", optimize=True)
        return normalized_path

    def _callbacks_for_job(self, connection: ConnectionContext, job_id: str):
        plugin = self

        class JobCallbacks:
            def __init__(self):
                self._last_progress_at = 0.0
                self._last_progress = 0
                self._last_logged_step = None
                self._last_stream_log_at = 0.0
                self._last_stream_text = ""

            def progress_value(self):
                return self._last_progress

            def on_progress(self, update):
                now = time.time()
                progress = int(getattr(update, "progress", 0) or self._last_progress or 0)
                phase = str(getattr(update, "phase", "") or "")
                status = str(getattr(update, "status", "") or "")
                current_step = getattr(update, "current_step", None)
                total_steps = getattr(update, "total_steps", None)
                step_key = (current_step, total_steps) if current_step is not None and total_steps is not None else None
                step_changed = step_key is not None and step_key != self._last_logged_step
                if not step_changed and now - self._last_progress_at < 2.0:
                    return
                self._last_progress_at = now
                self._last_progress = progress
                plugin._set_active_job_status(
                    phase=phase,
                    status=status,
                    progress=self._last_progress,
                    current_step=current_step,
                    total_steps=total_steps,
                )
                payload = {
                    "phase": phase,
                    "status": status,
                    "progress": self._last_progress,
                    "current_step": current_step,
                    "total_steps": total_steps,
                }
                if step_changed:
                    self._last_logged_step = step_key
                    plugin._log(
                        f"WanGP step advanced; job_id={job_id} "
                        f"step={payload['current_step']}/{payload['total_steps']} "
                        f"progress={payload['progress']} phase={payload['phase']!r} status={payload['status']!r}."
                    )
                else:
                    plugin._log(
                        f"WanGP progress callback; job_id={job_id} "
                        f"phase={payload['phase']!r} progress={payload['progress']} "
                        f"step={payload['current_step']}/{payload['total_steps']} status={payload['status']!r}."
                    )
                try:
                    plugin._post_job_update(connection, job_id, "progress", payload)
                except Exception as exc:
                    plugin._log(f"Progress update failed: {exc}", force=True)

            def on_status(self, status):
                text = str(status or "").strip()
                if not text:
                    return
                plugin._set_active_job_status(status=text, progress=self._last_progress)
                plugin._log(f"WanGP status callback; job_id={job_id} status={text!r} progress={self._last_progress}.")
                try:
                    plugin._post_job_update(connection, job_id, "progress", {"status": text, "progress": self._last_progress})
                except Exception:
                    pass

            def on_info(self, info):
                text = str(info or "").strip()
                if not text:
                    return
                plugin._log(f"WanGP info; job_id={job_id} message={text!r}.")

            def on_error(self, error):
                message = str(getattr(error, "message", None) or error or "").strip()
                if not message:
                    return
                stage = str(getattr(error, "stage", "") or "").strip()
                plugin._log(f"WanGP error callback; job_id={job_id} stage={stage!r} message={message!r}.", force=True)

            def on_output(self, output):
                plugin._log(f"WanGP output event; job_id={job_id} output_type={type(output).__name__}.")

            def on_stream(self, message):
                stream = str(getattr(message, "stream", "") or "stream").strip()
                if stream.lower() in {"stdout", "stderr"}:
                    return
                text = self._clean_stream_text(str(getattr(message, "text", "") or ""))
                if not text:
                    return
                if PLUGIN_NAME in text:
                    return
                if not self._should_log_stream_text(text):
                    return
                self._last_stream_text = text
                self._last_stream_log_at = time.time()
                plugin._log(f"WanGP {stream}; job_id={job_id} {text}")

            def _should_log_stream_text(self, text: str) -> bool:
                if text != self._last_stream_text:
                    return True
                return time.time() - self._last_stream_log_at >= 5.0

            @staticmethod
            def _clean_stream_text(text: str) -> str:
                text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", str(text or ""))
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 500:
                    text = f"{text[:497]}..."
                return text

        return JobCallbacks()
