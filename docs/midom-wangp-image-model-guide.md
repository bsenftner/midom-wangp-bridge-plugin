# Choosing WanGP Image Models For Midom

A practical model-selection and download guide for Midom users.

**Last checked: November 18, 2026**, against the local Midom Bridge and WanGP
20.00. Model names below are WanGP names, followed by their Bridge identifiers.
Your Midom installation may use shorter display names or hide older options.

This guide covers the 16 image models currently listed by the Bridge. It does
not list every model in WanGP, hosted image-provider models, or proposed future
integrations. A feature available directly in WanGP is not necessarily exposed
through Midom.

## Start Here

**If you install only one image model for Midom, start with Qwen Image 2.1.** It
covers general generation, reference-based editing, and the specialized asset
and compositing workflows that make Midom more than a prompt-to-picture tool.

| What you want to do | Start with | Why |
|---|---|---|
| General images, reference edits, reusable Project assets | Qwen Image 2.1 | Broad coverage and the deepest Midom workflow integration. |
| Characters or props on transparent backgrounds | Qwen Image 2.1 through the appropriate transparent-output or Visual Variations mode | Native RGBA generation, rather than painting a checkerboard into an RGB image. |
| Posters, diagrams, infographics, text-heavy layouts | SenseNova U1.5 | A useful first choice when wording, layout and spatial relationships matter. Proofread every result. |
| Quick fresh concepts without reference images | Z-Image Turbo | A fast exploration-oriented option. |
| Quick concepts or edits that need reference images | Flux 2 Klein 4B | A compact, distilled generation-and-editing option. |
| A larger alternative to Klein 4B | Flux 2 Klein 9B | Worth comparing when the smaller model struggles, at a higher resource cost. |
| A different look for product, portrait or editorial imagery | Flux 2 Dev | A substantial general generation/editing alternative; compare its interpretation with Qwen. |
| Flux Dev-style work with fewer inference steps | pi-FLUX.2 | Distilled acceleration of the large Flux Dev model, not a small model. |
| Pose, depth or edge-guided image composition | A Z-Image Control variant | Dedicated structural conditioning, rather than ordinary reference-image editing. |
| Revise or repeat an established older Qwen workflow | The matching older Qwen model | Compatibility and a familiar result can matter more than changing models. |

These are practical starting recommendations, not a universal quality ranking
or a measured speed chart. The best choice depends on the image, prompt,
precision, accelerator, resolution and your workstation.

## What To Download In WanGP

You do **not** need all 16 models. A useful small collection is:

1. **Qwen Image 2.1 TextImage2Image 7B** for Midom's core image workflows.
2. **SenseNova U1.5 8B MoT** if you frequently create infographics or posters.
3. **Z-Image Turbo 6B** for fast text-only exploration, or **Flux 2 Klein 4B**
   if your fast workflow needs image references.
4. A **Z-Image Turbo Fun ControlNet** variant only if structural control is
   part of your work.

Install and initialize the models inside WanGP. Let WanGP manage its model
files, text encoders, autoencoders and additional control modules. Loading a
model may download several components; a model name does not describe the
whole installation size.

Midom does not install local model weights for you. After installing a model
or an optional accelerator, refresh the Bridge's capabilities and reload the
relevant Midom panel. A reported model is not proof that all its files have
already been downloaded; running a small test in WanGP first is useful.

### Complete Supported Model List

Use the name in the first column when looking in WanGP. IDs are included to
resolve naming differences, not because you normally need to enter them.

| WanGP model name | Bridge ID | Main purpose and tradeoff |
|---|---|---|
| **Qwen Image 2.1 TextImage2Image 7B** | `qwen_image_21_7B` | Main all-rounder and Midom visual-workflow model; small details and text still need review. |
| **SenseNova U1.5 8B MoT** | `sensenova_u1_5_8b_mot` | Text-rich compositions and high-detail layouts; Midom's large internal renders can be expensive. |
| **Flux 2 Dev 32B** | `flux2_dev` | Large general generation/editing model; substantial memory and loading requirements. |
| **pi-FLUX.2 Dev 32B** | `pi_flux2` | Four-step distilled Flux Dev route; faster sampling does not remove its large-model footprint. |
| **Flux 2 Klein 4B** | `flux2_klein_4b` | Compact, fast generation/editing; compare difficult fine details against larger options. |
| **Flux 2 Klein 9B** | `flux2_klein_9b` | Larger Klein generation/editing option; more demanding than 4B, not automatically better on every image. |
| **Z-Image Turbo 6B** | `z_image` | Fast text-to-image exploration; no ordinary image-reference editing through this route. |
| **Z-Image Base 6B** | `z_image_base` | Non-Turbo creative alternative; typically more sampling work, no ordinary reference editing. |
| **Z-Image Turbo Fun ControlNet v1 6B** | `z_image_control` | First structural-control variant; retain for established workflows rather than installing every version. |
| **Z-Image Turbo Fun ControlNet v2 6B** | `z_image_control2` | Revised structural-control variant; evaluate control strength and image quality on your material. |
| **Z-Image Turbo Fun ControlNet v2.1 6B** | `z_image_control2_1` | Latest of the three Bridge-listed control revisions; a reasonable new-workflow starting trial, not a guaranteed winner. |
| **Qwen Image Text2Image 20B** | `qwen_image_20B` | Older generation-only Qwen route, useful to compare for typography and established styles; heavier than the 7B visual model. |
| **Qwen Image Edit 20B** | `qwen_image_edit_20B` | Older reference-based editor; useful for existing recipes, not required for Qwen 2.1 editing. |
| **Qwen Image Edit Plus (2509) 20B** | `qwen_image_edit_plus_20B` | Older multi-reference editor with conditioning inputs; source appearance can leak into the result. |
| **Qwen Image Edit Plus (2511) 20B** | `qwen_image_edit_plus2_20B` | Later older-generation editor; not the model behind current Midom Visual Variations. |
| **Qwen Image Layered 20B** | `qwen_image_layered_20B` | Specialized decomposition into RGBA layers; a heavy specialist/legacy option, not the recommended transparent-asset starting point. |

## Understanding The Main Choices

### Qwen Image 2.1: The Main Midom Workhorse

Choose this first for visual assets you expect to edit, vary, compose or reuse.
The model combines generation and editing and supports native transparency.
Its official architecture includes an RGBA autoencoder; transparency is not
merely an instruction to paint a white background. [Qwen's model documentation](https://github.com/QwenLM/Qwen-Image-2.1)

**Good uses in Midom:** character concepts, prop designs, location plates,
reference-based revisions, staged scenes, transparent assets, local masked
changes and scene integration.

**Main limitations:**

- Treat editing as generative. An instruction to change lighting can also
  alter faces, lettering, costume details or other nearby content.
- Faces occupying few pixels are particularly vulnerable during full-scene
  edits. A good large portrait does not guarantee a good small face on a stage.
- Prefer short wording inside images. Dense infographics are not its strongest
  role in the current WanGP integration; try SenseNova for that work.
- Native alpha can have soft residue, halos, missing thin parts or unwanted
  partial opacity. Review it against different backgrounds.
- A reference is guidance, not a guarantee of identity or exact source pixels.

The ordinary Standard RGB route accepts up to **10 ordered reference images**.
Assign each reference a job: identity, wardrobe, prop, location, lighting or
style. Unrelated references can pull the result in conflicting directions.

Do not apply that ten-reference limit to every specialized tool: transparent
output, Visual Variations and masked operations have their own smaller limits.

#### Standard Versus Faster Qwen Profiles

Where the paired worker reports them, ordinary RGB generation/editing can use:

| Profile | Useful starting purpose | Important boundary |
|---|---|---|
| Standard | Quality baseline, complex edits, up to ten ordinary references | More sampling work; specialized tools may enforce their own Standard recipe. |
| Viggle Turbo, 6 steps | Faster ordinary RGB iterations | Up to three references; no negative prompt in this profile. |
| Pruna, 8 steps | A faster RGB alternative to compare | Up to three references; no negative prompt. |
| Pruna, 5 steps | Fastest Pruna exploration option | Lower expected quality than its eight-step profile; inspect small details. |

These accelerator profiles are not permission to accelerate every Qwen tool.
Visual Variations, native RGBA, compositor integration, masked editing and
Generative Reframing use explicitly supported recipes. A tool may therefore
offer Standard even when ordinary Image Generation offers faster profiles.

Also, the separate upstream **Qwen Image 2.1 Turbo checkpoint** is not listed
as a separate supported Bridge model here. Do not confuse it with the installed
Viggle/Pruna accelerator profiles for the existing Qwen 2.1 model.

### SenseNova U1.5: Text, Layout And Infographics

Start here for designed graphics involving headings, labeled sections,
diagrams, product explanations or multiple spatial requirements. The upstream
project specifically targets unified visual understanding, generation and
editing. [SenseNova project documentation](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/README.md)

In the current Midom integration, this is a useful first comparison when Qwen
2.1 makes image text too uncertain or simplifies a complicated layout. Supply
short, exact text in quotation marks and describe the hierarchy: title,
sections, labels and footer.

**Tradeoffs:** Midom's Bridge uses large internal renders before delivering the
selected canvas. Even a 720p deliverable can therefore take substantial time
and memory. The 8B label is not a promise that it is lighter in practice than
every larger-parameter model. Standard and an optional **Fast - 12 steps**
profile are exposed; the latter uses an accelerator whose filename says
eight-step, but Midom's curated setting is twelve steps.

The Bridge accepts up to three ordinary image references. It does not expose
the same specialized RGBA, Visual Variations or compositor contracts as Qwen
2.1. Upstream region-editing examples do not imply Midom offers an equivalent
SenseNova mask tool.

Always proofread spelling, figures, arrows, labels and relationships. An
attractive infographic is not evidence that its facts are correct. For exact
logos, legal wording, dense body text or revision-sensitive numbers, generate
the artwork and add authoritative text separately rather than rely entirely
on generated pixels.

### Flux 2: Alternative Appearance And Reference Editing

**Flux 2 Dev** is worth trying when you want another interpretation of a
portrait, product, illustration or scene, or when Qwen repeatedly misses the
look you want. It supports generation and editing; the upstream model is a
large 32B system. [Flux 2 Dev model card](https://huggingface.co/black-forest-labs/FLUX.2-dev)

**pi-FLUX.2** offers a four-step distilled route using the Flux Dev base plus
additional components. Choose it to evaluate faster iterations with that
family, not because you expect a small download or universally identical
results. The distillation changes the recipe and may change the result.
[pi-Flow project documentation](https://github.com/Lakonik/LakonLab)

**Klein 4B and Klein 9B** offer compact distilled generation/editing models.
Start with 4B when iteration speed and resources matter; compare 9B on tasks
where you want to see whether more capacity helps. Do not assume that returning
to the large Dev model will improve every composition. [Klein model documentation](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

All four Flux routes accept up to **three ordinary reference images** through
the Bridge. They do not expose the Qwen-specific native-RGBA, masked-edit,
Visual Variations or compositor integration tools. Choose them for generation
and reference editing, not as drop-in substitutes for those workflows.

### Z-Image: Fast Exploration Or Structural Guidance

**Z-Image Turbo** is a useful fast first pass for fresh concepts, portraits,
product ideas and compositions. Its distilled design targets few-evaluation
generation. **Z-Image Base** is the non-Turbo alternative for exploring a
different creativity/control balance. Neither provides ordinary reference
editing through these Bridge routes. [Z-Image model documentation](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)

The three **Fun ControlNet** variants add a different kind of input: a
structural control image. Through Midom, their advertised controls are pose,
depth, edges and raw/prepared control maps. You do not need to install all
three revisions. For a new workflow, trial v2.1 first; retain another version
when it produces a result you prefer or supports an established recipe.

Think of a pose control as guidance for arrangement and movement shape, not
proof of the person's identity. A depth map helps spatial structure; edges
constrain outlines. Strong structure can also constrain creative freedom or
carry unwanted details from the source. These routes are not character-copying
or pixel-preservation tools.

Upstream WanGP descriptions may mention additional control or inpainting
features. Use the modes Midom actually offers for the paired worker; do not
assume every upstream ControlNet feature is available through the Bridge.

### Older Qwen Models And Qwen Image Layered

The older **20B generation/editing family** remains listed for compatibility
and comparison. The Text2Image model is generation-only through this route;
the Edit/Plus/Plus2 routes accept up to three image references. Plus models
also expose pose, depth, shapes, recolorization and raw conditioning modes.

Their conditioning images can contribute appearance and content as well as
structure. A competing person in a control image can influence who appears
in the result. Do not assume they behave like a pure pose-only control model.

For new Midom character/prop workflows, prefer Qwen 2.1. In this project's
earlier testing, an older Qwen 2511 Alternate View recipe showed faint,
blurry/blotchy output under its tested INT8 setup. That observation concerns
that recipe and configuration, not a universal verdict on every 2511 render.

**Qwen Image Layered** is a specialist that decomposes a picture into generated
RGBA layers. The Bridge returns a source/reconstruction echo followed by the
decomposition layers, with up to nine artifacts total. This is not the same
thing as generating one isolated prop or character with native alpha.

Earlier Bridge testing found Layered consistently useful only with a heavy
BF16 configuration. It is therefore not the recommended first installation
for transparent assets. That historical limitation does **not** impose a
global BF16 requirement on Qwen Image 2.1 or ordinary Midom image work.

## Qwen Image 2.1 Inside Midom's Visual Tools

Installing Qwen 2.1 enables more than selecting it in Image Generation. It is
the model integrated into **Image Visual Variations** and the generative
operations in the **Image Compositor / 2D Compositing** workspace.

### Image Visual Variations

| Mode | Intended result | What to review |
|---|---|---|
| **Character Asset** | A character variation on a transparent RGBA background | Face, costume, hands, feet, complete silhouette and alpha edges. |
| **Location Plate** | An RGB environment/background variation | Composition, perspective, lighting and unwanted added subjects. |
| **Prop Asset** | An isolated RGBA prop or related object group | Thin parts, holes, materials, attached pieces and transparency. |
| **Dressed Set** | An RGB scene with characters, props, scenery and lighting interpreted together | Missing/duplicated people or props, altered blocking, perspective and connected shadows. |

These are dedicated Midom recipes, not merely the same prompt with a different
label. They use one primary source with up to two supporting references under
their current reference-enabled contracts. Give each supporting image a clear
purpose. Generating a variation does not guarantee exact camera rotation,
unchanged identity or retention of every object.

For Character and Prop variations, a valid but muddy alpha channel can arrive
as **needs alpha cleanup**. It is a review state, not proof of a clean cutout.
Midom's **Media Assembly Alpha Cleanup** tool lets you inspect and refine the
alpha and save a new derivative. The original generated RGBA file remains
available. Other transparent-output routes can apply stricter alpha checks.

### Image Compositor And Scene Integration

Placing, moving, scaling, rotating and layering assets in the compositor are
deterministic compositing operations. They do not require Qwen to regenerate
the whole picture. Qwen is used when you request generative scene integration.

**Draft Integration** can reinterpret the entire flattened scene. Use it to
explore a different overall look, but expect possible changes to faces,
lettering, props and small details.

**Production Integration** uses an editable binary matte around selected edge
and contact areas. White permits changes; black protects the saved composite.
The Bridge performs a higher-resolution generation pass and restores protected
black-matte pixels exactly before returning the final RGB PNG. The current
multi-layer path works from the flattened composition and combined matte.

Choose Production when preserving an approved face or costume matters more
than allowing unrestricted relighting. Paint the editable area carefully:
anything inside the white region can change. Protection is not a promise
that the newly generated shadow or edge will be aesthetically correct.

A request to relight the entire scene conflicts with protecting most of its
pixels. Use Draft when you accept broader reinterpretation, or adjust the
composition and selective integration goal. An experimental atmosphere-layer
operation is separate and should not be mistaken for guaranteed relighting.

### Masked Editing And Generative Reframing

Qwen 2.1 also underpins Midom's supported local masked-edit and Generative
Reframing routes. A masked edit targets an indicated region; it is different
from using a whole picture as an appearance reference. Generative Reframing
creates a new composition around a positioned source and may reinterpret
source content. It is not a promise to preserve the original source rectangle
pixel for pixel.

## A Useful Working Method

1. **Choose the task before choosing the model.** A transparent character,
   poster, pose-controlled composition and scene integration are different jobs.
2. **Start with one clear prompt and only useful references.** Explain what
   each image contributes and what should remain consistent.
3. **Generate a small comparison.** Change one variable at a time: model,
   accelerator, wording or reference set. Keep the seed when available, but do
   not expect the same seed to mean the same image across models or recipes.
4. **Inspect at delivery size and zoomed in.** Check faces, fingers, feet,
   lettering, object counts, perspective and transparent edges.
5. **Save promising work as Project assets.** Use Visual Variations for asset
   development and the compositor for deliberate arrangement instead of
   repeatedly asking a full-frame model to preserve everything.
6. **Use protected operations when preservation matters.** Review the matte
   rather than assuming a request to "keep the face" will enforce protection.

Example visual prompt for an asset: "One complete brushed-metal table lamp,
standing upright, matte blue housing, original handle retained, soft studio
lighting." Select the appropriate transparent/Prop workflow; users do not
need to supply the Bridge's internal RGBA instructions.

Example reference edit: "Keep the person and clothing from Image 1. Use
Image 2 for the studio's color palette and lighting, not for another person."
This clarifies intent; it is still a generative instruction, not a guarantee.

Example infographic request: "A three-section poster. Title: 'Community
Workshop'. Sections: 'Learn', 'Build', 'Share'. Large readable headings, ample
spacing, one simple illustration per section." Add complexity gradually.

## Performance, Formats And Expectations

- **Parameter count is not total memory.** Text encoders, reference encoding,
  autoencoders, control modules, internal resolution and offloading matter.
- **Fewer steps do not mean a proportionate speedup.** Loading, encoding,
  decoding and delivery preparation still take time. Measure your own worker.
- **A BF16 LoRA filename is not a global BF16 instruction.** Let the supported
  WanGP recipe manage precision; do not change every model to solve one issue.
- **Resolution means the delivered canvas.** Some Bridge recipes render a
  larger internal image and crop/downscale. More internal pixels do not
  guarantee identity preservation or correct detail.
- **PNG does not automatically mean transparency.** Use a supported RGBA
  operation. A visible checkerboard inside an RGB picture is not an alpha channel.
- **An upscale is not recovered ground truth.** Crop & Upscale's optional
  FlashVSR Restore Detail step is a separate processing operation and can alter
  fine detail; it is not another text-to-image model in the list above.
- **Technical support is not licensing permission.** Check the selected
  checkpoint's current license, reference-image rights and intended use.
  Different variants in the same family can have different terms.

## Further Reading And Maintenance

The model inventory and Midom-specific limits come from the Bridge's current
`MODEL_CAPABILITY_OVERRIDES`, capability builders and validators, cross-checked
against Midom's worker model registry and visual-tool implementations. The
selection advice is practical guidance, not a published all-model benchmark.

For upstream capabilities beyond this integration, consult the linked model
documentation. Follow Midom's actual offered controls for day-to-day use;
upstream examples may use different checkpoints, licenses or runtime settings.

For the Bridge's installation and compatibility instructions, see the
[plugin README](../README.md). Refresh this guide when the supported model
inventory, native names or specialized Midom workflows change.
