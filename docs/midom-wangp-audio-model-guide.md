# Choosing WanGP Voice, Music And Sound Models For Midom

A practical guide to speech, dialogue, voice conversion, music and audio sound
effects for Midom users.

**Last checked: November 19, 2026**, against local Bridge 0.8.13 and WanGP 20.00.
The inventory contains ten supported audio entries, including both Fish
checkpoints and the separate Stable Audio Music/SFX choices. SeedVC is a
processing facility rather than an ordinary text-to-audio model selector.
The new ACE XL song route requires matching Midom integration and capability
refresh; a local Bridge implementation does not establish that a deployed
Midom site already offers that selector.

Midom offers the subset reported by your paired worker and permitted by its
current server policy. More models or features in WanGP do not automatically
mean they are supported through Midom. This guide does not cover hosted audio
providers or claim that all these models have been benchmarked against one
another on the same workstation.

## Start With The Kind Of Audio You Need

There are four different jobs:

- **Text-to-speech:** perform the words in an approved script.
- **Voice conversion:** change the vocal identity of an existing recording.
- **Music generation:** create a song, musical bed, cue or instrumental composition.
- **Sound-effects generation:** create an event, ambience, transition or texture.

A reference voice does not turn every model into a voice converter. A music
model is not a dependable reader of exact dialogue. A sound-effects model is
not a tool for repairing a noisy recording.

| Your goal | A useful starting choice | Why |
|---|---|---|
| Expressive narration or dialogue, with or without voice references | Fish Audio Drama 3 | Flexible performance and new-voice generation; allow time to audition and retry imperfect takes. |
| Clone a narrator and experiment with emotion guidance | Index TTS 2 | Voice reference plus optional emotion reference or emotion cues. |
| Two known voices taking turns | Index TTS 2 or Fish Audio | Both expose explicitly mapped speaker-reference workflows; compare actual delivery. |
| A multilingual single narrator with an explicit language choice | Chatterbox Multilingual | Language selection and a focused single-reference speech route. |
| A conversational or dramatic scene from screenplay-like descriptions | DramaBox Audio | Dialogue-first prompting with two voice anchors and additional scripted speakers. |
| Keep an existing performance but change its voice | SeedVC Voice Replacement | Converts recorded speech instead of generating new words from a script. |
| An instrumental music cue with tempo/key guidance | ACE-Step 1.5 Music | Exposes musical controls; the current Midom route is instrumental-only. |
| Sing your supplied lyrics, including personalized names or places | ACE-Step 1.5 XL Turbo | Separate lyrics and musical direction, with human review of every take; no reference-voice cloning. |
| A music bed, groove or loop candidate described in words | Stable Audio 3 Small Music | Music-focused prompt generation. |
| An impact, whoosh, mechanical sound, room tone or ambience | Stable Audio 3 Small SFX | Sound-event-focused generation, rather than musical composition. |

These are starting points, not promises that one model will always be faster,
more accurate or more natural. Choose by the performance you hear.

## What To Install In WanGP

You do not need every option. A practical starter collection is:

1. **One Fish Audio Drama 3 checkpoint** for expressive speech and dialogue,
   or **Index TTS 2** if your first requirement is reference-cloned narration.
2. **Chatterbox Multilingual** if explicit language selection is important.
3. **One music model:** ACE-Step for its musical controls, or Stable Audio
   Small Music for prompt-based beds and cues.
4. **Stable Audio Small SFX** if your productions need designed effects.
5. **SeedVC** only if you need to convert an existing spoken performance.

Install and initialize these inside WanGP. WanGP manages the weights and
supporting encoders/decoders; Midom does not install local model weights for
you. First loading can involve more files than the main checkpoint suggests.
Run a short test in WanGP, then refresh Bridge capabilities and reload Midom.

### Complete Supported Audio List

| WanGP selection or facility | Bridge ID | Best fit and main limitation |
|---|---|---|
| **TTS Fish Audio Drama 3** | `fish_drama3` | Default/FP8-expert expressive speech and dialogue; substantial resources and imperfect take reliability. |
| **TTS Fish Audio Drama 3 INT4 Experts** | `fish_drama3_int4` | Same exposed speech features with lower expert-memory requirements; audition for quality differences. |
| **TTS Index TTS 2** | `index_tts2` | Reference-cloned speech, emotion guidance and two-speaker dialogue; requires voice recordings. |
| **TTS Chatterbox Multilingual** | `chatterbox` | Language-selected, single-speaker cloning; short script chunks and no exposed multi-speaker mode. |
| **DramaBox Audio** | `dramabox_audio` | Expressive scene/dialogue generation; requires two voice anchors through this Bridge route. |
| **SeedVC audio processor: v1.0 Speech** | `seedvc_voice_replacement` | Change a recorded speaker's voice; not text-to-speech, singing conversion or multi-speaker replacement. |
| **Music ACE-Step v1.5 Turbo 2B** | `ace_step_v1_5_music` | Instrumental music with optional musical controls; lyrics and cloned singing voices are not exposed. |
| **Music ACE-Step v1.5 XL Turbo 4B** | `ace_step_v1_5_xl_music` | Supplied lyrics or explicit instrumental mode; eight steps, no planner, MP3 only, no cloned singing voices. |
| **Music Stable Audio 3 Small Music** | `stable_audio3_small_music` | Prompt-generated music; no supplied-audio editing or exact lyric performance through this route. |
| **Music Stable Audio 3 Small SFX** | `stable_audio3_small_sfx` | Prompt-generated effects and ambience; no source-audio repair/editing through this route. |

The Bridge's Stable Audio Music ID differs from its native WanGP architecture
name, `stable_audio3_small`. Use the Music/SFX selector names above rather than
substituting Medium or another upstream variant.

Likewise, a newer Index TTS version, another ACE-Step preset, or a different
SeedVC mode is not automatically interchangeable with the listed integration.

## Voice And Dialogue Models

### Fish Audio Drama 3: Expressive Performance, Reviewed Takes

Choose Fish when you want expressive delivery, conversational exchanges or
speech without first recording a reference. Voice references are optional;
when supplied, each needs an accurate transcript. The upstream releases offer
FP8 and INT4 expert variants. [Fish Audio model documentation](https://huggingface.co/fishaudio/drama-3),
[INT4 checkpoint documentation](https://huggingface.co/fishaudio/drama-3-int4)

The two Bridge checkpoints expose the same main features:

- Generate a model-selected voice without a reference.
- Clone one reference voice.
- Perform dialogue with up to **16 scripted speaker labels**.
- Supply cloned references for **Speakers 1-3 only**. Other speaker labels do
  not acquire their own uploaded reference voices.
- Add short delivery cues such as `[warm]`, `[calm]` or `[irritated]`.

**FP8 versus INT4:** start with FP8 as a comparison baseline if the workstation
fits it comfortably. Try INT4 when expert memory or offloading is a bottleneck.
INT4 does not mean every model component becomes four-bit, and it does not
remove the shared components. Neither variant is a universal speed/quality
winner; CPU memory bandwidth and offloading can matter as much as GPU size.

Example Fish dialogue:

```text
Speaker 1: [warm] Welcome to the community workshop.
Speaker 2: [curious] What can we make today?
Speaker 1: [encouraging] Let us start with something useful.
```

For three cloned references, map the recordings and their transcripts to
Speakers 1, 2 and 3 consistently. More scripted labels do not increase the
three-reference cloning limit.

**Review carefully:** Fish can stop early, omit or repeat words, drift between
voices, or assign similar voices to different speakers. Local testing has
observed both early stops and merged speaker identities. A different seed or
reference assignment can change the outcome, but neither is a guaranteed fix.
Verify each reference/transcript pairing before changing model settings.

A voice generated without a reference is not a durable named-voice identity.
For continuity, retain approved material and evaluate an authorized reference
workflow rather than assume a speaker label or repeated seed preserves a voice.

#### What "Completed - Review Needed" Means For Fish

On the explicit **Reviewable v2/v3** paths, a valid playable take can complete
even when generation stopped on silence or reached a length limit. Warnings
tell you what happened; they do not certify script completeness.

| Selected Fish path | Generation ceiling | Independent artifact safety ceiling |
|---|---|---|
| Historical v1 | Up to 120 seconds | Historical stricter acceptance rules; not the reviewable policy. |
| Reviewable v2 | Up to 120 seconds | 180 seconds and 50 MiB, plus decoded-audio safety checks. |
| Reviewable v3 | Up to 180 seconds | 240 seconds and 50 MiB, plus decoded-audio safety checks. |

The maximum is a ceiling, not a target. An 80-second performance is not too
short merely because the maximum was 120 seconds. A slight overrun can be
reviewable under v2/v3 within independent safety limits. An overrun beyond the
0.25-second warning deadband receives a warning; the deadband is not a relaxed
safety limit or proof of codec padding.

Listen, then Keep or Reject the take where those controls are available. Keep
records your editorial judgment, not verified identity or complete wording.
Malformed media, execution failure, authorization failure and cancellation
remain failures. Older jobs/workers do not silently acquire v2/v3 behavior.

Fish's current Bridge profile generates one combined recording in a whole-
script pass. It does not return individual speaker stems or word timestamps.
A retry is a deliberate new generation, not a guaranteed repair of the old
take. Preserve any usable recording before trying alternatives.

### Index TTS 2: Voice Cloning And Emotion Guidance

Use Index TTS 2 as another starting point for an existing narrator's voice,
especially when you want to compare emotion guidance. The model separates
voice identity and emotional conditioning. [IndexTTS project documentation](https://github.com/index-tts/index-tts)

Through Midom, you can use one voice reference with an optional emotion
reference, or two speaker references for two-speaker dialogue. An emotion
reference guides delivery; it is not an additional dialogue character.

Short inline emotion cues are supported, for example:

```text
[calm] The workshop opens at nine. [enthusiastic] Everyone is welcome.
```

For dialogue, each nonempty script line must use the correct label:

```text
Speaker 1: Welcome back to the workshop.
Speaker 2: Thank you. I am ready to begin.
```

This Bridge route supports two scripted speakers, not every dialogue facility
that may exist in a newer WanGP version. Single-speaker jobs can process the
full script or split it by lines/paragraphs where offered. Splitting may
introduce changes in rhythm, pauses or delivery between chunks.

Choose between Index and Fish by auditioning the same short, representative
passage with the same authorized voice sample. Do not assume either model
will reproduce every pronunciation or the speaker's identity perfectly.

### Chatterbox Multilingual: One Speaker, Explicit Language Choice

Choose Chatterbox when you want a single reference-cloned narrator and an
explicit language selector. It exposes language, exaggeration and pace
controls in the Bridge. [Chatterbox project documentation](https://github.com/resemble-ai/chatterbox)

Its focused scope can be useful for narration, localized announcements and
short explanatory passages. Changing language does not translate your script:
write the intended words in the intended language first.

The current route has **no multi-speaker dialogue, separate emotion reference
or Fish-style inline performance tags**. Do not paste one model's control
syntax into another and assume it will be interpreted rather than spoken.

Keep scripts short: the full-prompt limit is 350 characters. Line/paragraph
processing permits a longer combined script, but each chunk is limited to
300 characters, with a 2,000-character overall limit. Split at natural sentence
boundaries and listen for awkward joins. Audition pace/exaggeration changes;
these are not precise words-per-minute or emotion-intensity measurements.

### DramaBox Audio: A Scene Performed As Dialogue

Try DramaBox for a conversational or dramatic exchange, especially when scene
description is a natural way to express the delivery you want. The Bridge uses
its dialogue-first route with **two required voice anchors** and up to **eight
scripted speaker labels**. Eight labels do not mean eight uploaded cloned voices.

This is not the same as Fish's simple tagged speech. A useful DramaBox style
is a speaker block with a short performance description and quoted words:

```text
Speaker 1:
A welcoming host speaks clearly, "Welcome to the community workshop."
Speaker 2:
A curious visitor replies, "What can we make today?"
```

Both Speaker 1 and Speaker 2 must appear. Keep the spoken wording inside clear
quotation marks and performance description outside. This describes the
native screenplay-like approach; it does not guarantee every quotation will
be performed without additions or omissions.

The exposed controls include a duration multiplier and optional unexpected-
word removal. Those are assistance, not proof of transcript accuracy. Word
removal can add processing and should not replace listening. DramaBox's
timing controls are not a promise of exact commercial-length delivery.

Choose Fish or Index when you prefer their more explicit line-and-reference
workflow. Choose DramaBox when its scene interpretation produces the better
performance for your material. The similarly named DramaBox and Fish Audio
Drama 3 are different integrations, not two precision versions of one model.

## SeedVC: Convert A Performance You Already Have

Use SeedVC when the existing recording has the words, pauses and performance
you want, but you need a different authorized voice. It is **voice conversion**,
not TTS: supply a source recording and a separate target-voice reference.
[SeedVC project documentation](https://github.com/Plachtaa/seed-vc)

This can be a useful alternative to repeatedly asking TTS to recreate a
particular performance. Timing and expression come primarily from the source,
but inspect the converted result for artifacts, consonant changes and drift.
Do not treat conversion as a guarantee of sample-exact timing or perfect identity.

The current Bridge supports only **SeedVC v1.0 Speech**, one source and one
target reference, with source recordings up to 120 seconds. It is advertised
only when WanGP's SeedVC processor is configured for that supported speech
mode. It is not the singing route, SeedVC v2, a live conversion endpoint or
multi-person replacement.

There is no text prompt, script splitting or duration setting in this route.
It does not rewrite incorrect words, translate speech, isolate several speakers
or remove a music bed. Prepare a clean single-speaker source first.

## Prepare Good Voice References

| Route | Reference requirement through the Bridge |
|---|---|
| Fish Audio | Optional zero to three voice samples; each 3-30 seconds, at least 16 kHz, with an exact transcript. |
| Index TTS 2 | Voice samples 10-30 seconds, at least 16 kHz; one for cloning, two for two-speaker dialogue; optional separate emotion sample in the applicable single-speaker mode. |
| Chatterbox | One 10-30-second voice sample, at least 16 kHz. |
| DramaBox | Two 10-30-second voice anchors, at least 16 kHz; the native route uses a smaller internal reference budget. |
| SeedVC | One target reference 5-25 seconds, about 10 seconds recommended, at least 16 kHz; plus the source performance. |

Use one clearly audible speaker, natural delivery, little room echo, and no
overlapping voices or music. Avoid clipped peaks and recordings made mostly
of silence. More seconds of poor audio are not necessarily better than a
shorter clean sample within the supported bounds.

For Fish, the transcript describes **what is in the reference recording**, not
the new script you want generated. Include the actual words and keep the
sample paired with its own transcript. The worker does not transcribe it for
you. A transcript/sample mismatch is a conditioning problem, not something a
larger maximum duration will fix.

Use voices and recordings you are authorized to use. A cloned voice is not
proof that its owner approved the generated statement. Check the selected
model's current license as well as permissions for the reference material.

## Music Models

### ACE-Step 1.5 XL Turbo: Songs With Supplied Lyrics

Choose this separate XL model when the words matter: a short song, a jingle,
or personalized sung lines containing names and places. Enter **Lyrics** as
the words to sing, with optional section labels on their own lines, and
**Music Description** as genre, instrumentation, mood and vocal direction.
The Bridge does not rewrite the lyrics or certify that all words were sung.

Example lyrics:

```text
[Verse]
Morning lights in Denver glow
Blake is ready, here we go
```

Example music description:

```text
Warm acoustic pop with a clear English vocal, gentle guitar, piano and a
restrained rhythm section. Friendly, intimate delivery.
```

The route uses the plain eight-step XL Turbo preset, not a planner-enabled
variant. Optional controls are BPM, key, meter, language and seed. To request
an instrumental, select **Instrumental** explicitly and leave lyrics empty;
do not insert an instrumental marker into Song with Lyrics mode.

Requests span 5-360 seconds, defaulting to 30. This is musical guidance, not
an exact-length promise: valid shorter or longer takes within independent
safety limits remain reviewable. Delivery is one 48-kHz stereo MP3, bounded
at 362 decoded seconds and 80 MiB. There is no WAV delivery, reference audio,
cover mode or cloned singing voice in this first contract.

In Blake's short local tests, XL performed the supplied words and pronounced
"Blake" and "Denver" clearly. That is evidence for those inputs, not a general
accuracy, speed or hardware guarantee. Audition pronunciation, missing or
extra lyrics, musical structure and endings before using a take. Oversized
text fails clearly rather than silently losing tokens.

### ACE-Step 1.5: Instrumental Cues With Musical Controls

Choose ACE-Step for an instrumental cue where genre, instrumentation, groove
and musical direction matter. The upstream project supports broader song
creation, but **this older, smaller Bridge route deliberately accepts instrumentals
only**. [ACE-Step project documentation](https://github.com/ace-step/ACE-Step-1.5)

You describe the music in the caption. Where offered, you can also specify
BPM, key/scale and time signature. The initial route supports 5-120-second
requests, defaulting to 30 seconds. Musical settings guide generation; they
are not a score-level timing guarantee.

Example caption:

```text
Warm acoustic instrumental for a neighborhood workshop advertisement.
Gentle guitar, soft piano and restrained percussion. Steady moderate groove,
an optimistic mood, and plenty of space beneath a narrator. No vocals.
```

**Do not use this older route for personalized sung lyrics.** The Bridge fixes
its lyrics input to an instrumental marker. Upstream lyric entry, cover
generation, supplied-audio editing and voice cloning
are not exposed by this route. Use the separate XL song selection for lyrics.
A language control does not enable
singing in an instrumental-only route.

Listen for unwanted vocal-like sounds, unstable musical structure, abrupt
endings and a melody that competes with speech. An "instrumental" instruction
is not an automatic quality certificate.

### Stable Audio 3 Small Music: Beds, Grooves And Loop Candidates

Choose this for prompt-described background music, short grooves and cue
exploration. Its separate music checkpoint is intended for music generation.
[Stable Audio Small Music model documentation](https://huggingface.co/stabilityai/stable-audio-3-small-music)

Describe the sound: genre, instruments, tempo, energy, atmosphere and intended
structure. The Bridge permits 1-120-second requests, with a 30-second default.
A negative prompt is available; use it for unwanted characteristics such as
vocals or harsh distortion, not as a guarantee of their absence.

Example:

```text
A restrained electronic instrumental bed, soft synth pulses, warm bass and
light percussion, relaxed tempo, steady energy, no dramatic build, no vocals.
Suitable underneath a spoken product explanation.
```

Unlike ACE-Step's exposed BPM/key fields, the music direction here is chiefly
in the prompt. Try both models on the same production need and keep the better
cue. Neither is the universal music winner.

A requested loop is a **loop candidate**, not a verified seamless loop. Check
the join and use an appropriate edit or crossfade. The Bridge does not expose
Stable Audio's upstream audio inpainting/continuation capabilities or accept
reference audio through this route. It is not an exact-lyrics speech model.

## Stable Audio 3 Small SFX: Events And Ambience

Use SFX rather than the Music checkpoint for a concrete event or acoustic
environment: an impact, switch click, whoosh, room tone, machinery or weather.
It is a separate sound-effects-focused checkpoint. [Stable Audio Small SFX model documentation](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)

Describe the source, action, material, perspective and acoustic space. Start
with one clearly specified event rather than a long list of simultaneous
sounds. Requests are 1-120 seconds, with an eight-second default; a negative
prompt is available.

Example one-shot:

```text
One heavy metal workshop door closes with a solid latch click, close
perspective, a short indoor reverberation tail, otherwise quiet, no speech
or music.
```

Example ambience:

```text
Quiet interior workshop ambience, soft ventilation and distant mechanical
hum, steady background texture, no distinct speech, no music, no large impacts.
```

Inspect event count, unwanted speech/music, background hiss and whether the
sound has enough lead-in and decay for editing. A request for "one impact"
can still return several. Place or trim the chosen event during assembly;
do not assume generated events occur on exact video frames.

This route creates sound. It does not clean an existing recording, separate
voices from noise, follow an uploaded video or guarantee a technically accurate
recording of a particular product.

## Build The Audio Before The Video

For a narrated production, a useful order is:

1. **Finalize the script.** Resolve names, prices, dates, pronunciations and
   any personalized/template fields before generating the voice.
2. **Generate and review speech.** Listen to the whole recording and save the
   approved take as a Project asset. A normal model ending is not proof that
   all words were spoken.
3. **Generate music and effects separately.** Keep them independent of the
   approved voice so you can replace or remix one without regenerating all.
4. **Assemble and measure the audio.** Set levels, trim deliberately, add fades
   where appropriate and preserve the original takes.
5. **Use the approved audio as the Storyboard's audio spine.** Select cuts and
   prepare bounded segments for the chosen video model. Supplying audio does
   not itself guarantee accurate lip synchronization.

This supports consistent narration across visual takes. It avoids asking
every new video generation to invent the voice, wording and timing again.
These are production recommendations, not a claim that all planned
personalized-media/template applications are already implemented.

## Duration, Formats And Review

- **Measure the recording you received.** Speech pace depends on the script,
  reference and performance. Changing a maximum does not force an exact-length
  reading. To hit a fixed advertisement slot, revise wording or delivery and
  make an explicit edit rather than assume the ceiling solves timing.
- **Do not generalize Fish's policy.** Reviewable v2/v3 is an explicit Fish
  selection. Other routes retain their own duration and failure rules; a
  reviewable Fish take may still exceed a downstream Storyboard/provider limit.
- **WAV is usually the better working copy when offered.** It avoids an extra
  lossy MP3 stage. Prefer it for further editing or conversion, but do not
  assume every model's native codec, sample rate or channel count is identical.
- **MP3 is useful for compact delivery.** The Bridge offers it only when its
  encoder is available. Converting to WAV later does not restore lost detail.
- **Outputs are not stems.** These routes return one audio artifact per job;
  dialogue in one file does not mean independently editable speaker tracks.
- **Generation does not mix your production automatically.** A music bed may
  mask consonants even when the speech sounds fine in isolation. Review the
  assembled mix on the intended listening device.
- **Listen for more than pleasant tone.** Check every name, number, speaker
  change, sentence ending, pause and required phrase. Also check clipping,
  silence, repeated words, unintended vocalizations and abrupt endings.
- **Short trials save time.** Test a representative passage before a long
  script. Keep reference settings consistent when comparing checkpoints; a
  repeated seed across different models does not imply the same performance.
- **Retries are new takes.** Retain useful output and make deliberate retries.
  Do not assume a retry preserves a voice or fixes a particular error.

No universal generation-time or VRAM figures are promised here. Checkpoint
precision, shared components, CPU/RAM bandwidth, offloading, script length and
encoder/decoder configuration all affect performance. Do not apply a global
BF16 requirement simply because one model or previous experiment needed it.

## Further Reading And Maintenance

The inventory and route limits were checked against the Bridge's audio
capabilities/validators, WanGP's current model definitions and Midom's worker
registry. Recommendations are practical guidance, not a controlled benchmark.
Linked upstream documentation may describe features beyond this integration.

See also [Choosing WanGP Image Models For Midom](midom-wangp-image-model-guide.md)
and the [Bridge installation README](../README.md). Refresh this guide when
checkpoint selections, exposed audio modes or acceptance policies change.
