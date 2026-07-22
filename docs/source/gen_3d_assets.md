# 3D Asset Generation

GRAIL has two complementary 3D-asset pipelines: a **procedural** generator
for terrain primitives (curbs, slopes, stairs) and an **AI-generated**
pipeline using Tencent's [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
for textured object meshes from text.

## Procedural terrain

No GPU, no external services — direct OBJ generation.

```bash
# All terrain types (300 of each by default)
python -m grail.pipelines.gen_terrain --type all --num 300 --output_dir data/Terrain

# A specific type
python -m grail.pipelines.gen_terrain --type curb --num 40
python -m grail.pipelines.gen_terrain --type slope --num 100 --seed 1234
python -m grail.pipelines.gen_terrain --type stairs --num 50 --output_dir data/syn_stairs
```

Each asset lands in `data/<output>/<type>_<NNN>/` as `model.obj`,
`model.mtl`, and `texture.jpg`.

## Hunyuan3D (text → textured mesh)

Run the Hunyuan3D asset pipeline through its package module from the repo
root. This stage does not have a project-root wrapper script.

The Hunyuan3D pipeline uses the `hunyuan` conda env (separate from `grail`)
and calls the OpenAI API for prompt enhancement, height estimation,
image-prompt verification, and reference-image generation. Chat and vision
helpers default to `gpt-4o`; reference images use `gpt-image-1.5`.

```bash
conda run -n hunyuan python -m grail.pipelines.gen_3d_assets \
    -i configs/gen_3d/example_objects.yaml \
    -o data/gen_example
```

Pipeline per object:

1. Enhance the user prompt for 3D-friendly framing.
2. Generate a 1024×1024 reference image.
3. Verify image-prompt match; retry up to `--max-image-retries`.
4. Background remove and resize to 512×512.
5. Hunyuan3D shape generation → `mesh.glb`.
6. Hunyuan3D texture paint → textured `model.obj` + materials.
7. Estimate real-world height (vision chat) and rescale.

Each object takes ~2 minutes on an L40S.

### Sharded fan-out

`grail.pipelines.gen_3d_assets` accepts `--num_job_chunks N` and `--job_chunk_idx i` so the
work can fan out across N workers:

```bash
python -m grail.pipelines.gen_3d_assets \
    -i configs/gen_3d/chairs.yaml \
    -o data/gen_chairs \
    --job_chunk_idx <i> \
    --num_job_chunks <N>
```

## Bundled object lists

| Config | Count | Use |
|---|---|---|
| `configs/gen_3d/example_objects.yaml` | 6 | Smoke test |
| `configs/gen_3d/chairs.yaml` | 70 | Diverse rigid chairs (sitting demos) |

Each YAML is a plain list of object descriptions, one per line. Larger
object batches are not shipped in the repo; generate your own lists using
the guidelines below.

## Object selection guidelines

The downstream HOI pipeline (Hunyuan3D shape gen → Blender render → physics
sim → 4D reconstruction) is sensitive to the kind of object you generate.
When curating a list, follow these rules:

**Prefer:**

- Mid-sized objects suitable for human/robot manipulation. Typical largest
  dimension **5–60 cm** for table-top, **50–150 cm** for floor placement.
- Single rigid bodies — no cloth, fabric, hair, or flexible/deformable items.
- Visually asymmetric or feature-rich enough that orientation is well-defined
  (so FoundationPose can lock onto a canonical pose).

**Avoid:**

- **Too small** (< ~5 cm: spoons, paperclips, screws, single coins, dice) —
  get lost in 1280×720 renders, contact reasoning becomes ambiguous.
- **Too large** (> ~150 cm: fridges, sofas, full-size treadmills, vehicles) —
  won't fit in indoor scenes; exceed the character's reachable workspace.
- **Articulated** objects (fridges with doors, scissors, foldable chairs,
  drawers, robot arms with joints) — Hunyuan3D produces a single rigid mesh
  and articulation is lost.
- **Fully rotationally symmetric** objects (plain spheres, plain balls,
  generic orbs) — orientation is undefined, FoundationPose tracking is
  degenerate. *Asymmetric variants — soccer balls with panels, basketballs
  with seams — are fine.*
- **Phantom flat-base risk** — bare "cube", "pedestal", "sculpture display",
  or "modern minimalist block" descriptions tend to come out of Hunyuan3D
  with an extra flat slab/plinth baked under them. Specify natural feet or
  contact surface explicitly: e.g. *"concrete cube side table on four small
  recessed feet"*, *"sculpture with a rounded organic base"*, *"ottoman with
  hairpin legs"*.

`configs/gen_3d/chairs.yaml` is the in-repo example list that follows these
rules — mirror its style and granularity when authoring new lists.
