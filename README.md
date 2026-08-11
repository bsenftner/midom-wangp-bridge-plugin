# Midom Remote Worker

Midom Remote Worker connects a local WanGP installation to Midom as a project-scoped media worker. The plugin folder/repository slug is `Midom-at-AWS-worker-bridge`.

WanGP provides local access to many open-source image, audio, music, and video models. Midom provides project organization, worker pairing, job queueing, file storage, provenance, and media-production workflows. This plugin is the bridge between them: it lets a local workstation running WanGP claim Midom media jobs, run selected WanGP models or deterministic FFmpeg processing operations, and upload the completed artifacts back to the correct Midom project.

The plugin intentionally exposes a curated model set instead of raw WanGP controls. Normal Midom users should not need to know WanGP prompt-type letters, local LoRA filenames, filesystem paths, sampler internals, or experimental model switches.

## Midom Media Generation Examples

These Midom Blog posts show the kinds of organized media workflows this bridge is designed to support:

- [The New Office Media Department Is Already in the Office](https://midombot.com/b1/blog/MI/the-new-office-media-department-is-already-in-the-office)
- [Media Storyboards: Guided Video Production, Even if Your First Customer Is a Caveman](https://midombot.com/b1/blog/MI/media-storyboards-guided-video-production-even-if-your-first-customer-is-a-caveman)
- [Video Collection Events](https://midombot.com/b1/blog/MI/video-collection-events)

## What This Plugin Does

- Pairs one local WanGP process with one or more Midom project worker records.
- Stores each pairing as an independent connection context with its own worker id, token, organization id, project id, user id, token expiry, and capabilities revision.
- Reports curated image, voice-audio, music/sound, video, and local media-processing capabilities to Midom.
- Polls Midom for queued jobs and claims at most one active local WanGP job at a time.
- Downloads only job-scoped inputs through Midom worker routes.
- Validates each claimed job against a local model-specific contract before calling WanGP internals.
- Runs generation through a background-safe WanGP session.
- Runs selected deterministic FFmpeg media-processing jobs locally for paired Midom projects.
- Uploads generated artifacts with SHA-256 metadata.
- Reports progress, completion, failure, heartbeat, revocation, and disconnect status through Midom worker APIs.

The plugin is outbound-only. It does not expose an inbound local API.

## Architecture

Midom owns:

- User identity and project membership.
- Worker pairing codes and scoped worker tokens.
- Job creation and job queueing.
- Project file storage and artifact validation.
- Server-side authorization, leases, revocation, rate limits, and audit logs.

This plugin owns:

- Local WanGP worker configuration.
- Capability reporting.
- Job polling and explicit claim.
- Local input validation and WanGP settings mapping.
- WanGP execution.
- Local FFmpeg processing for supported non-AI media jobs.
- Artifact discovery and upload.
- Local stop/disconnect controls.

The security boundary is Midom's server-side worker authorization and artifact validation, not the Python plugin. Open-source users can modify the plugin, so Midom must treat all worker requests as untrusted.

## Installation

1. Install and verify WanGP normally.
2. Copy or clone this repository into the WanGP plugins directory:

   ```text
   Wan2GP/plugins/Midom-at-AWS-worker-bridge/
   ```

3. Start or restart WanGP.
4. Open the WanGP UI and select the `Midom Remote Worker` plugin tab.
5. In Midom, create a project worker pairing code for the project you want this WanGP machine to serve.
6. In the plugin tab, enter:
   - Midom API Base URL.
   - Pairing Code.
   - Machine Name.
   - Local HTTP development checkbox only if you are intentionally using localhost/private LAN HTTP.
7. Click the pair/start action in the plugin UI.

The plugin writes local pairing state to:

```text
worker_config.json
```

The file contains scoped worker credentials. The plugin attempts to save it with `0600` permissions. Do not publish or share this file.

## HTTP And HTTPS

HTTPS is required for normal use.

HTTP is allowed only for explicit development modes:

- Localhost development.
- Private LAN / tailnet development.

The plugin validates the URL and rejects embedded credentials, query strings, fragments, and URL parameters. Public or production Midom deployments should use HTTPS.

## Pairing And Multi-Project Use

Pairing codes are generated in Midom and are project-scoped.

The current multi-project behavior is:

- One local WanGP process.
- Multiple stored project pairings are allowed.
- Pairings must use the same Midom API base URL.
- Pairings must belong to the same Midom user.
- Each pairing has its own worker token and project scope.
- The local WanGP engine runs one active job total.
- Jobs from different paired projects share the local execution queue.
- Non-active sibling pairings report a generic busy message while another project owns the active generation.

If Midom revokes a worker token, the plugin removes that stale local pairing when it receives HTTP 401/403 for non-active pairing operations. A valid sibling pairing can continue running.

## Supported AI Models And Local Processing

The exact model list is reported to Midom at pairing and capability update time. Midom should show only models reported by the worker and allowed by Midom's server-side model policy.

### Image Generation

Flux 2 family:

- `flux2_dev`
- `pi_flux2`
- `flux2_klein_4b`
- `flux2_klein_9b`

Qwen Image family:

- `qwen_image_20B`
- `qwen_image_edit_20B`
- `qwen_image_edit_plus_20B`
- `qwen_image_edit_plus2_20B`
- `qwen_image_layered_20B`

Z-Image family:

- `z_image`
- `z_image_base`
- `z_image_control`
- `z_image_control2`
- `z_image_control2_1`

Important image behavior:

- Ordinary image models are capped at 6 outputs.
- `qwen_image_layered_20B` is capped at 9 total outputs.
- Qwen Image Layered uses one raw control image and returns an ordered layer set.
- For Qwen Image Layered, artifact index `0` is the source/reconstruction echo and artifact indexes `1..8` are editable decomposition layers.
- Qwen Edit Plus control images are edit/conditioning inputs, not pure pose/depth guides. A control image with a competing subject can affect subject identity.
- Z-Image control variants are better suited for structure/layout control from pose, depth, edge, or raw control inputs.

Supported image output types:

- PNG
- JPEG
- WebP

Curated image resolutions include:

- `768x768`
- `1024x1024`
- `1280x720`
- `720x1280`

### Voice Audio Generation

`index_tts2`

- Voice-clone TTS.
- Single-reference mode.
- Optional emotion-reference mode.
- Two-speaker dialogue mode with `Speaker 1:` / `Speaker 2:` script blocks.
- Prompt processing modes: full prompt, split by non-empty lines, split by paragraphs.

`chatterbox`

- Single-reference voice-clone TTS.
- Requires one reference audio file.
- Exposes language, exaggeration, and pace controls.
- Prompt processing modes: full prompt, split by non-empty lines, split by paragraphs.
- Chatterbox prompt chunks should remain short; Midom validates the stricter chunk rules.

`dramabox_audio`

- Dialogue-first voice generation.
- Uses two uploaded voice anchors.
- Accepts `Speaker N:` script blocks.
- Supports N scripted speakers with a first-pass practical cap of 8 speaker labels.
- Outputs one concatenated dialogue audio artifact.

### Voice Conversion

`seedvc_voice_replacement`

- Converts an existing spoken source audio into a target reference voice.
- Not text-to-speech.
- No prompt, no prompt splitting, no dialogue script, no duration control.
- Requires exactly one `source_audio` input and one `reference_audio` input.
- First pass maps to WanGP `seedvc_one_speaker`.
- Reported only when the local WanGP SeedVC extension is enabled in a speech-compatible mode.
- Source audio first-pass cap: 120 seconds.
- Target reference audio max: 25 seconds.

### Music And Sound Generation

`stable_audio3_small_music`

- Prompt-generated music beds, loops, ambience, and soundtrack-style audio.
- Maps internally to WanGP `stable_audio3_small`.
- Output count is 1.

`stable_audio3_small_sfx`

- Prompt-generated sound effects, impacts, transitions, foley, UI sounds, and other short non-speech audio.
- Output count is 1.

`ace_step_v1_5_music`

- Instrumental music generation.
- First pass is instrumental-only.
- Midom prompt maps to WanGP music caption / `alt_prompt`.
- Lyrics are fixed to `[Instrumental]` for the first pass.
- Optional controls include BPM, key scale, time signature, and language when reported by the worker.

Audio output is truthful per worker:

- MP3 is reported only when ffmpeg/libmp3lame transcoding is available.
- WAV is reported when MP3 output cannot be guaranteed.

### Video Generation

`longcat_avatar_v1_5`

- Talking avatar / lip-sync video.
- One reference image.
- One driving audio input.
- Duration follows driving audio up to 20 seconds.
- Curated resolutions:
  - `832x480` compatibility mode.
  - `1280x720` landscape.
  - `720x1280` portrait.

`ltx2_22B_1_1`

- LTX-2.3 audio-guided video.
- One start image.
- One driving audio input.
- Optional Ending Image Target for ordinary audio-guided jobs.
- Duration follows driving audio up to 20 seconds.
- Curated resolutions:
  - `1280x720`
  - `720x1280`
- Optional sync profile:
  - `standard`
  - `omninft_rl_lora_sync` / Better Audio-Video Sync, reported only when the local OmniNFT LTX-2.3 LoRA is installed.

LTX-2.3 control-video mode:

- Same model id: `ltx2_22B_1_1`.
- One start image.
- One normalized control video with embedded audio.
- No separate driving audio in the current contract.
- No Ending Image Target in the first control-video pass.
- Duration is bounded by control-video audio/video duration and capped at 20 seconds.
- Control modes:
  - Human motion.
  - Human motion aligned.
  - Depth.
  - Canny edges.
  - Raw control video.

`i2v_2_2_Enhanced_Lightning_v2_svi2pro`

- Wan2.2 SVI 2 Pro cinematic image-to-video.
- One start image.
- Optional Ending Image Target.
- Fixed duration from 1 to 30 seconds.
- Curated resolutions:
  - `1280x720`
  - `720x1280`

Video output is MP4.

### Local Media Processing

Local media-processing capabilities are reported separately from AI generation. These jobs still use the same Midom worker pairing, token, candidate polling, claim, input download, progress, artifact upload, complete, fail, disconnect, and revoke routes.

These operations are deterministic FFmpeg work. They are not WanGP AI generation jobs, but they run inside the same local worker process so project-owned hardware can take work off the shared Midom server.

`event_video_ffmpeg_processor`

- Deterministic Video Collection Event processing, not AI generation.
- Reported separately from WanGP AI model capabilities under `media_processing`.
- Requires local FFmpeg and FFprobe.
- First pass accepts one `source_video` input plus optional orientation-specific `overlay_png` and `bumper_image` inputs.
- Uses existing claimed-job input download routes and canonical worker job input ids.
- Produces exactly one browser-friendly H.264/AAC MP4 artifact with the `mobile_public_720p` profile.
- Preserves/corrects source orientation, applies matching portrait/landscape branding assets when configured, optionally appends a still-image bumper, and optimizes the MP4 for web playback.
- Validates source duration, overlay/bumper dimensions, output MP4 bytes, exact output dimensions, H.264/AAC streams, duration bounds, and rotation metadata before upload.
- Applies local FFmpeg subprocess timeouts and cancellation handling so a stuck encode does not hold a claimed job indefinitely.

Midom remains responsible for poster generation, captions, moderation, approval, publication, gallery visibility, and final artifact validation.

`storyboard_ffmpeg_processor`

- Durable Storyboard FFmpeg processing for paired Midom projects.
- Used by Multi-Camera Storyboards and Media Storyboards when Midom routes eligible operations to a paired worker.
- Lets card-level and assembly-level video work continue after a browser tab is closed or refreshed.
- Produces exactly one H.264/AAC MP4 artifact per job.
- Uses the existing claimed-job input download route and canonical `inputs[].input_id` values.
- Validates downloaded input MIME types by input role before processing.
- Reports progress while FFmpeg runs, applies subprocess timeouts, and terminates FFmpeg on cancellation.

Currently advertised operation types include:

- `multicam_card_pass_through_take`
- `multicam_card_trim_take`
- `multicam_card_overlay_take`
- `multicam_final_assembly`
- `multicam_optimize_video`
- `optimize_video`
- `replace_video_soundtrack`
- `multicam_seekable_mp4`
- `multicam_ai_video_take_prepare`
- `mediastoryboard_card_pass_through_take`
- `mediastoryboard_card_local_video_take`
- `mediastoryboard_card_trim_take`
- `mediastoryboard_card_edge_trim_take`

Storyboard overlay support includes:

- Static rectangle picture-in-picture overlays.
- Animated and interpolated rectangle overlays.
- Animated position, scale, and opacity.
- Fixed-canvas animated scale handling for FFmpeg stability.
- Base-video or overlay-video audio selection.
- PNG and JPEG luminance mattes, where white reveals the overlay, black hides it, and gray partially reveals it.

Storyboard final assembly support includes:

- Ordered segment assembly.
- Per-segment trims.
- Normalization to one output size, frame rate, H.264 video, and AAC audio before concatenation.
- Silent AAC audio fill for no-audio segments.
- Web-ready MP4 output with `+faststart`.

Storyboard optimization support includes:

- H.264/AAC MP4 re-encode.
- `yuv420p` output.
- `+faststart`.
- Optional proportional downscaling by `max_dimension`.
- Configurable CRF, preset, and audio bitrate when sent by Midom.

Storyboard soundtrack replacement support includes:

- Replacing a video's audio with a separate AudioMass or soundtrack export.
- Accepting either audio files or video containers with audio streams for `source_audio` and `soundtrack_audio`.
- Mapping only the audio stream from video-container audio-role inputs.
- Honoring video start offset, soundtrack start offset, requested duration, and optional head/tail silence.

## WanGP Asset Requirements

The local WanGP operator is responsible for installing and testing WanGP models, model assets, and accelerator LoRAs.

The plugin detects and reports capabilities. It does not install arbitrary model assets or LoRAs in response to Midom jobs.

Some capabilities appear only when local runtime support is available:

- SeedVC appears only when WanGP SeedVC is enabled in a compatible speech mode.
- MP3 output appears only when ffmpeg/libmp3lame transcoding is available.
- LTX Better Audio-Video Sync appears only when the required OmniNFT LoRA is installed.
- Qwen Lightning accelerator profiles appear only when the required local Qwen accelerator files are installed.

## Basic Use

1. Start WanGP on the GPU workstation.
2. Open the `Midom Remote Worker` plugin tab.
3. Pair the worker to a Midom project with a fresh Midom pairing code.
4. Click Update Capabilities after changing local WanGP model configuration or installing new supported assets.
5. Start the worker loop if it is not already running.
6. Queue media generation jobs from Midom.
7. Monitor progress in Midom or in the plugin log panel.
8. Disconnect a project pairing when that project should no longer use this worker.

If the worker has been idle for a long time, the plugin gradually backs off polling. After long-idle standby, use the local Wake for Active Use button before expecting immediate job pickup.

For live Video Collection Events, use Event Video Processing Keep Awake before the upload window starts. The 4h, 8h, and 12h keep-awake buttons are local-only controls that keep compatible Event Video Processing pairings out of long-idle standby so Midom can queue bursty event uploads to the local processor instead of immediately falling back to hosted processing. Disable it after the event window if normal idle standby behavior is preferred.

For Storyboard FFmpeg work, Midom decides whether a job is eligible for a paired worker. If a compatible worker is available, Midom can route card rendering, overlay, trimming, final assembly, optimization, and soundtrack replacement jobs to this plugin; otherwise Midom can run its hosted fallback path.

## Troubleshooting

SeedVC does not appear:

- In WanGP, enable SeedVC Voice Replacement in Configuration > Extensions.
- Use a speech-compatible SeedVC mode.
- Click Update Capabilities in the plugin.

MP3 is not available:

- Verify WanGP can find ffmpeg.
- Verify ffmpeg includes `libmp3lame`.
- If MP3 transcoding is unavailable, the worker should report WAV-only output for affected models.

HTTP URL is rejected:

- Use HTTPS for normal deployments.
- For local development only, enable the matching localhost or private LAN / tailnet HTTP checkbox.
- Make sure the host resolves to localhost or a private LAN/tailnet address.

Pairing succeeds but an old worker reports 401/403:

- The old pairing was probably revoked or expired in Midom.
- The plugin removes stale non-active pairings automatically when Midom rejects them.
- Generate a fresh pairing code if the current project pairing was revoked.

Jobs queue but do not start quickly:

- Check whether the worker is running.
- Check whether the worker is in long-idle standby and needs Wake for Active Use.
- Check whether another paired project already owns the active local WanGP generation.
- Check whether the pairing was created with the job family enabled. Midom may offer separate pairing choices for AI Media Generation, Event Video Processing, and Storyboard FFmpeg processing.

Storyboard FFmpeg jobs run hosted instead of on the worker:

- Restart WanGP after updating the plugin so the new capability report is loaded.
- Create a fresh pairing if Midom still shows old capabilities for the worker.
- Confirm the pairing allows the relevant local processing job type.
- Confirm the plugin log reports `storyboard_ffmpeg_processor` capabilities.
- Confirm the specific operation type is advertised by the worker and allowed by Midom.

Storyboard matte overlays render as normal rectangles:

- Confirm the job includes an `overlay_image` input with a matte/mask role, or a `processing.operation_payload.mask_dbfileid` / `mask_input_id` value matching the downloaded matte input.
- Confirm the plugin log says `matte_detected=True` and `final_overlay_mode=luminance_matte`.
- If a matte is provided but cannot be applied, the plugin should fail the job clearly instead of silently rendering an unmasked rectangle.

## Security Notes

This plugin stores scoped worker credentials, not a user's normal Midom login credentials.

The worker token should only be able to:

- Register/update worker capabilities.
- Send heartbeats.
- Poll and claim assigned media jobs.
- Download inputs for claimed jobs.
- Report progress.
- Upload generated artifacts for claimed jobs.
- Complete, fail, disconnect, or respond to revocation for its own worker scope.

Midom must treat the plugin as an untrusted client. A modified plugin can lie about capabilities, progress, output metadata, and version. Server-side Midom validation remains authoritative.

Report suspected vulnerabilities privately according to [SECURITY.md](SECURITY.md). Do not post pairing codes, worker tokens, private project files, or full unreviewed logs in public issues.

This plugin is released under the MIT License. WanGP, model files, LoRAs, FFmpeg, Python packages, and other third-party dependencies remain governed by their own licenses.

Detailed Midom-side synchronization notes, internal threat-model checklists, and future planning documents are not published in this public plugin repository. Public releases should document the operator-facing behavior here and keep server-side hardening details in Midom's private engineering documentation.

## Development Notes

This plugin is a separate Git repository nested under:

```text
Wan2GP/plugins/Midom-at-AWS-worker-bridge/
```

Useful local checks:

```bash
python -m py_compile plugin.py
git diff --check
```

Do not commit local `worker_config.json`, generated media, pairing codes, bearer tokens, or OS sidecar files.

Private development checkouts may contain an ignored `internal-docs/` directory for Midom-side synchronization notes and planning records. Those files are intentionally excluded from the public plugin repository.
