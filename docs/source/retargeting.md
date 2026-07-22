# Retargeting

Convert GRAIL 4D HOI reconstructions (the output of `grail.pipelines.recon_4dhoi`) into
G1 robot motion trajectories consumable by the task-general tracking stack
under {src}`imports/SONIC/`.

The retargeting pipeline is a standalone subpackage:
{src}`grail/retargeting/`. All steps run as plain CLI tools.

## Install

1. Initialize the GMR submodule:
   ```bash
   git submodule update --init imports/GMR
   ```

2. In the published Docker image, activate the preinstalled environment:

   ```bash
   conda activate sonic
   ```

   For a source installation, run the one-shot installer from the GRAIL root:

   ```bash
   bash scripts/setup/install_env_sonic.sh
   ```

   This creates the `sonic` environment and installs Isaac Sim, Isaac Lab,
   public GMR, GRAIL, and the remaining retargeting/training dependencies.
   GRAIL-specific GMR behavior is applied at runtime by
   {blob}`grail.adapters.gmr <grail/adapters/gmr.py>`; the public GMR submodule
   is not modified. The script is idempotent.

   Override the env name via `GRAIL_SONIC_ENV=<name>`:
   ```bash
   GRAIL_SONIC_ENV=my_sonic_env bash scripts/setup/install_env_sonic.sh
   ```

## Running the pipeline

### End-to-end (recommended)

The first argument must point to **your own successful 4D-HOI reconstruction
output** inside the current checkout or container. Replace
`<your_results_dir>` with the value passed to reconstruction's `--results_dir`,
which defaults to `results`. The default SMPL-X reconstruction config writes
validated results to `generation/4dhoi_recon_smplx_valid/`. This directory
must contain at least one nested `hoi_data/hoi_data.pkl`; the second argument
only chooses the new folder name under `data/motion_lib/`. Use the validated
root to process every dataset, or append a dataset subdirectory to process only
that dataset.

```bash
conda activate sonic
export DISPLAY=:1                    # GMR uses mujoco viewer, needs a display

RECON_DIR="<your_results_dir>/generation/4dhoi_recon_smplx_valid"
OUTPUT_FOLDER="<your_output_folder>"

# This must print at least one file before retargeting.
find "$RECON_DIR" -type f -path '*/hoi_data/hoi_data.pkl' -print -quit

bash grail/retargeting/scripts/retarget_pipeline.sh \
    "$RECON_DIR" \
    "$OUTPUT_FOLDER"
```

If the `find` command prints nothing, correct `RECON_DIR` or verify that the
reconstruction stage produced valid results before continuing.

Outputs under `data/motion_lib/<your_output_folder>/`:

| Directory  | Contents                                          |
|------------|---------------------------------------------------|
| `robot/`   | G1 joint trajectories (one pkl per motion)        |
| `objects/` | Object 6-DOF trajectories                         |
| `object_usd/` | IsaacLab-ready USD assets                      |
| `meta/`    | Scene metadata (table pose, object name, …)       |

Plus a preprocessed twin at `data/motion_lib/<your_output_folder>_ha/`:

| Directory  | Contents                                                      |
|------------|---------------------------------------------------------------|
| `robot/`   | Robot motions with hand-action + table pose                   |
| `objects/` | Object motions, contact points filtered to ≥ lift frame       |
| `meta/`    | Per-motion meta (table pose/quat/size, object name)           |

And a BPS encoding at `data/motion_lib/<your_output_folder>/bps/` (multi-object
datasets only).

### Individual stages

Each stage is a plain Python CLI and can run in isolation.

```bash
RECON_DIR="<your_results_dir>/generation/4dhoi_recon_smplx_valid"
OUTPUT_BASE="data/motion_lib/<your_output_folder>"

# Stage 1 — retarget SMPL-X → G1
python -m grail.retargeting.retarget \
    --data_dir "$RECON_DIR" \
    --all --robot unitree_g1 --no_viewer \
    --output_dir "$OUTPUT_BASE"

# Stage 2 — hand-action + table-geometry processing
python -m grail.retargeting.process \
    --input  "$OUTPUT_BASE" \
    --output "${OUTPUT_BASE}_ha" \
    --meta_pkl data/g1_smplx/g1_skeleton_meta.pkl \
    --include_contact_points --grasp_from_lift \
    --lift_threshold 0.02 --grasp_anticipation_frames 10 \
    --skip_no_lift --per_object

# Add --treat_hands_equally to preserve both arms and derive left/right
# hand actions symmetrically from each hand's contacts.

# Stage 3 — BPS shape encoding (multi-object datasets only)
python -m grail.retargeting.compute_bps \
    --object_usd_dir "$OUTPUT_BASE/object_usd" \
    --output_dir     "$OUTPUT_BASE/bps"
```

The shell wrappers under
{src}`grail/retargeting/scripts/`
(`retarget.sh`, `process.sh`, `compute_bps.sh`) are thin convenience layers on
top of these CLIs — read them if you want to know the exact defaults.

(terrain-sitting-data)=
### Terrain / sitting data

Terrain (curbs, slopes, stairs) and sitting data involve whole-body interaction
with large environmental objects, not hand-held manipulation. Use
`--zero_out_wrist` to skip hand IK:

```bash
TERRAIN_RECON_DIR="<your_results_dir>/generation/4dhoi_recon_smplx_valid"
TERRAIN_OUTPUT_FOLDER="<your_terrain_output_folder>"

bash grail/retargeting/scripts/retarget.sh \
    "$TERRAIN_RECON_DIR" \
    "$TERRAIN_OUTPUT_FOLDER" \
    --zero_out_wrist
```

Then skip the `process.sh` step — terrain data does not need hand-action
preprocessing.

## How the pipeline works

1. **GMR (General Motion Retargeting)** — SMPL-X body model → Unitree G1 MJCF
   via inverse kinematics. The retarget engine is the public
   [YanjieZe/GMR](https://github.com/YanjieZe/GMR) submodule; GRAIL-specific
   compatibility behavior is applied at runtime by
   {blob}`grail.adapters.gmr <grail/adapters/gmr.py>`.
2. **Object mesh → USD** — `convert_mesh.py` runs IsaacLab's `MeshConverter`
   headlessly to produce simulation-ready USD assets with convex-hull collision.
3. **Hand-action + table geometry** — `process.py` derives hand open/close
   commands from object lift/contact timing and applies table geometry fixes.
   By default, the legacy right-hand pickup path zeroes the left arm and keeps
   `hand_action_left` open. Use `--treat_hands_equally` to preserve both arms
   and derive each hand action from that hand's contact timing.
4. **BPS encoding** — `compute_bps.py` samples surface points from each object
   USD and projects them onto a fixed basis-point set, producing a 10-D object
   shape embedding used as a policy observation in
   {blob}`pnp_table <imports/SONIC/gear_sonic/config/exp/manager/universal_token/hoi/pnp_table.yaml>`.

## Troubleshooting

| Symptom                                          | Likely cause / fix                                                                                    |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: general_motion_retargeting` | Rerun `bash scripts/setup/install_env_sonic.sh` — it installs GMR editable in the active env.         |
| `ModuleNotFoundError: pxr`                        | `pip install usd-core` (standalone PXR; Isaac Sim-vendored `pxr` is only importable inside kit apps). |
| Black mujoco viewer / `glfwInit failed`           | `export DISPLAY=:1` **before** activating conda (it is an env var, not a conda setting).              |
| `FileNotFoundError: .../robot` during processing  | Retargeting found no inputs. Update the first pipeline argument to your reconstruction directory and confirm `find "$RECON_DIR" -type f -path '*/hoi_data/hoi_data.pkl'` returns files. |
| Retarget skips motions as "no lift"              | `process.py` rejects motions where the object never rises 2 cm. Override with `--lift_threshold 0.01`. |
| `imports/GMR` is empty                            | Run `git submodule update --init imports/GMR`, then rerun the installer for a source installation.   |
