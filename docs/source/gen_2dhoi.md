# 2D HOI Generation

Synthesizes short videos of a human interacting with a 3D object, by
chaining physics simulation, multi-view Blender rendering, and a video
foundation model (default: [Kling AI](https://kling.ai/dev)).

## Quickstart

Run the 2D HOI pipeline through its package module from the repo root. This
stage does not have a project-root wrapper script.

```bash
python -m grail.pipelines.gen_2dhoi \
    --dataset ComAsset --category cordless_drill \
    --character kid \
    --results_dir results --video_model_api kling-ai
```

Outputs:

- `results/generation/initial_states/` — physics-stable orientations.
- `results/generation/asset_renders/` — rendered scene PNGs.
- `results/generation/cameras/` and `depth_maps/` — geometric ground truth.
- `results/generation/videos_kling/` — the generated MP4s.

## Pipeline steps

```{list-table}
:widths: 5 25 70
:header-rows: 1

* - #
  - Stage
  - Notes
* - 1
  - Object simulation
  - Drop the object from a small height in Blender + Bullet, settle, save the
    final orientation. Skipped via `--skip_step1` when an `obj_scale.json`
    cache exists.
* - 2
  - Scale optimization
  - Iterative Blender render + chat-vision evaluation (small/big/correct).
    Skipped by default in `manipulation.yaml` (`skip_step2: true`) since most
    object configs ship a hand-tuned scale.
* - 3
  - Multi-view rendering
  - `num_rand_scenes` random camera + lighting variants per object,
    1280×720, 32 samples. Outputs scene PNG + object/character masks +
    depth + camera parameters.
* - 4
  - Video generation
  - Refines the prompt via chat-vision, then calls Kling AI image→video
    (5 s, pro mode by default). Polls every 30 s up to 120 attempts.
```

## Required environment

```{list-table}
:widths: 30 70
:header-rows: 1

* - Variable
  - Why
* - `OPENAI_API_KEY`
  - Prompt refinement (step 4) and scale evaluation (step 2) through the OpenAI API. Defaults use `gpt-4o`.
* - `KLING_ACCESS_KEY` + `KLING_SECRET_KEY`
  - Kling AI HTTP API.
```

## Common variants

```bash
# Skip simulation (cached) and scale (already in obj_scale.json)
python -m grail.pipelines.gen_2dhoi --dataset ComAsset --category cordless_drill \
    --character kid --skip_step1 --skip_step2 \
    --results_dir results

# Render only — no Kling video gen
python -m grail.pipelines.gen_2dhoi --dataset ComAsset --category cordless_drill \
    --character kid --skip_step1 --skip_step2 --skip_step4 \
    --results_dir results

# Use a custom config (e.g., terrain stairs)
python -m grail.pipelines.gen_2dhoi --config configs/gen_2dhoi/terrain_stairs.yaml \
    --results_dir results
```

## Configs

```{list-table}
:widths: 35 65
:header-rows: 1

* - File
  - Purpose
* - `configs/gen_2dhoi/manipulation.yaml`
  - Standard table-top / handheld manipulation
* - `configs/gen_2dhoi/sitting.yaml`
  - Sitting interactions (chair-class objects)
* - `configs/gen_2dhoi/terrain_curbs.yaml`
  - Terrain traversal — curbs
* - `configs/gen_2dhoi/terrain_slope.yaml`
  - Terrain traversal — slopes
* - `configs/gen_2dhoi/terrain_stairs.yaml`
  - Terrain traversal — stairs
```

Object-specific overrides (scale, scene, etc.) live in `configs/objects/`.

## Sharded fan-out

```bash
# Run one shard per worker in your scheduler.
python -m grail.pipelines.gen_2dhoi \
    --dataset ComAsset \
    --character jason_rigged_001 \
    --results_dir results \
    --video_model_api kling-ai \
    --skip_done \
    --job_chunk_idx <i> \
    --num_job_chunks <N>
```
