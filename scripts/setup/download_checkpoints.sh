#!/usr/bin/env bash
# ============================================================================
# Download all required checkpoints for GRAIL submodules.
#
# Usage:
#   bash scripts/setup/download_checkpoints.sh
#   bash scripts/setup/download_checkpoints.sh --skip-gem-smpl --skip-gem-soma
#   bash scripts/setup/download_checkpoints.sh --force   # re-download even if exists
#
# Sources:
#   GEM-SMPL       — HuggingFace nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  (checkpoint/GEM-SMPL/...)
#   GEM-SOMA       — HuggingFace nvidia/GEM-X + nvidia/soma-x
#   FoundationPose — HuggingFace nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  (checkpoint/FoundationPose/...)
#   Hunyuan3D      — GitHub release (RealESRGAN); shape/paint auto-download from HF
#   SONIC          — HuggingFace nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  (checkpoint/SONIC/models/...)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()      { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "  ${RED}[FAIL]${NC} $1"; }
section() { echo -e "\n${CYAN}── $1 ──${NC}"; }

# ── Parse arguments ────────────────────────────────────────────────────────
SKIP_GEM_SMPL=false
SKIP_GEM_SOMA=false
SKIP_FOUNDATIONPOSE=false
SKIP_HUNYUAN3D=false
SKIP_SONIC=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --skip-gem-smpl)       SKIP_GEM_SMPL=true ;;
        --skip-gem-soma)       SKIP_GEM_SOMA=true ;;
        --skip-foundationpose) SKIP_FOUNDATIONPOSE=true ;;
        --skip-hunyuan3d)      SKIP_HUNYUAN3D=true ;;
        --skip-sonic)          SKIP_SONIC=true ;;
        --force)               FORCE=true ;;
        --help|-h)
            echo "Usage: bash scripts/setup/download_checkpoints.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-gem-smpl        Skip GEM-SMPL checkpoint download"
            echo "  --skip-gem-soma        Skip GEM-SOMA checkpoint download"
            echo "  --skip-foundationpose  Skip FoundationPose weight download"
            echo "  --skip-hunyuan3d       Skip Hunyuan3D-2.1 checkpoint download"
            echo "  --skip-sonic           Skip SONIC base + reference checkpoint download"
            echo "  --force                Re-download even if files already exist"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Helper: download from HuggingFace ──────────────────────────────────────
hf_download() {
    local repo_id="$1"
    local filename="$2"
    local local_dir="$3"
    local local_path="$local_dir/$filename"

    if [ -f "$local_path" ] && [ "$FORCE" = false ]; then
        ok "$filename (already exists)"
        return
    fi

    mkdir -p "$local_dir"
    echo "  Downloading $filename from $repo_id..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$repo_id', '$filename', local_dir='$local_dir')
" 2>/dev/null
    if [ -f "$local_path" ]; then
        ok "$filename"
    else
        fail "$filename download failed"
    fi
}

# ── Helper: download from the GRAIL HF dataset repo ────────────────────────
# Remote layout `checkpoint/<rel>` mirrors local `imports/<rel>`, so callers
# only specify <rel> once. Hardlinks the file from the HF cache into the
# target path when possible to avoid duplicating multi-GB checkpoints.
GRAIL_HF_REPO="nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"

hf_grail_download() {
    local rel_path="$1"   # e.g. "GEM-SMPL/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    local local_path="imports/$rel_path"
    local remote_path="checkpoint/$rel_path"

    if [ -f "$local_path" ] && [ "$FORCE" = false ]; then
        ok "$rel_path (already exists)"
        return
    fi

    mkdir -p "$(dirname "$local_path")"
    echo "  Downloading $rel_path from $GRAIL_HF_REPO..."
    python3 - <<PYEOF
import os, shutil
from huggingface_hub import hf_hub_download
src = hf_hub_download(
    repo_id="$GRAIL_HF_REPO",
    filename="$remote_path",
    repo_type="dataset",
)
dst = "$local_path"
os.makedirs(os.path.dirname(dst), exist_ok=True)
if os.path.exists(dst):
    os.remove(dst)
try:
    os.link(src, dst)
except OSError:
    shutil.copy2(src, dst)
PYEOF
    if [ -f "$local_path" ]; then
        ok "$rel_path"
    else
        fail "$rel_path download failed"
    fi
}

# ════════════════════════════════════════════════════════════════════════════
print_body_model_instructions() {
    local smpl_dir="imports/GEM-SMPL/inputs/checkpoints/body_models/smpl"
    local smplx_dir="imports/GEM-SMPL/inputs/checkpoints/body_models/smplx"

    if [ -f "$smpl_dir/SMPL_NEUTRAL.pkl" ] && \
       [ -f "$smplx_dir/SMPLX_NEUTRAL.npz" ]; then
        ok "Licensed SMPL and SMPL-X body models"
        return
    fi

    mkdir -p "$smpl_dir" "$smplx_dir"
    warn "Please download the licensed body models manually:"
    echo "  SMPL:   https://smpl.is.tue.mpg.de/"
    echo "          SMPL_NEUTRAL.pkl -> $smpl_dir/"
    echo "  SMPL-X: https://smpl-x.is.tue.mpg.de/"
    echo "          SMPLX_NEUTRAL.npz -> $smplx_dir/"
    echo ""
}

# 1. GEM-SMPL checkpoints (HuggingFace nvidia/PhysicalAI-...-GRAIL)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_GEM_SMPL" = false ]; then
    section "GEM-SMPL checkpoints"

    python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
        fail "huggingface_hub is required. Install with: pip install huggingface_hub"
        exit 1
    }

    # These small, redistributable auxiliaries are checked on every run so an
    # interrupted or older installation can be repaired without --force.
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl/J_regressor_extra.npy"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl/J_regressor_h36m.npy"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl/J_regressor_wham.npy"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl/smpl_mean_params.npz"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smplx2smpl_sparse.pt"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/vimo/smpl_mean_params.npz"

    GEM_SMPL_MARKER="imports/GEM-SMPL/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    HMR4D_MARKER="imports/GEM-SMPL/outputs/mocap_mixed_v1/genmo/genmo_lg_jukebox_jukebox_new/version_0/checkpoints/last.ckpt"

    if [ -f "$GEM_SMPL_MARKER" ] && [ -f "$HMR4D_MARKER" ] && [ "$FORCE" = false ]; then
        ok "GEM-SMPL checkpoints (already exist)"
    else
        # Inputs bundle: pre-trained externals. Licensed SMPL/SMPL-X model
        # files are intentionally excluded and installed manually below.
        hf_grail_download "GEM-SMPL/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
        hf_grail_download "GEM-SMPL/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
        hf_grail_download "GEM-SMPL/inputs/checkpoints/vimo/vimo_checkpoint.pth.tar"
        hf_grail_download "GEM-SMPL/inputs/checkpoints/yolo/yolov8x.pt"
        # Outputs: GRAIL's own GENMO (HMR4D) checkpoint
        hf_grail_download "GEM-SMPL/outputs/mocap_mixed_v1/genmo/genmo_lg_jukebox_jukebox_new/version_0/checkpoints/last.ckpt"
    fi

    print_body_model_instructions
else
    section "GEM-SMPL checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 2. GEM-SOMA checkpoints (HuggingFace nvidia/GEM-X)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_GEM_SOMA" = false ]; then
    section "GEM-SOMA checkpoints"

    GEM_SOMA_INPUTS="imports/GEM-SOMA/inputs"
    GEM_SOMA_MARKER="$GEM_SOMA_INPUTS/pretrained/gem_soma.ckpt"

    if [ -f "$GEM_SOMA_MARKER" ] && [ "$FORCE" = false ]; then
        ok "GEM-SOMA checkpoints (already exist)"
    else
        python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
            fail "huggingface_hub is required. Install with: pip install huggingface_hub"
            exit 1
        }

        HF_REPO="nvidia/GEM-X"

        # Main checkpoint
        hf_download "$HF_REPO" "gem_soma.ckpt"    "$GEM_SOMA_INPUTS/pretrained"

        # ViTPose
        hf_download "$HF_REPO" "vitpose.pth"       "$GEM_SOMA_INPUTS/checkpoints/vitpose"

        # SAM-3D-Body
        hf_download "$HF_REPO" "sam3d_body.ckpt"   "$GEM_SOMA_INPUTS/checkpoints/sam-3d-body-dinov3"
        hf_download "$HF_REPO" "model_config.yaml" "$GEM_SOMA_INPUTS/checkpoints/sam-3d-body-dinov3"

        # MHR model
        hf_download "$HF_REPO" "mhr_model.pt"      "$GEM_SOMA_INPUTS/mhr_data"

        # SOMA scale data
        hf_download "$HF_REPO" "scale_mean.pth"    "$GEM_SOMA_INPUTS/soma_data"
        hf_download "$HF_REPO" "scale_comps.pth"   "$GEM_SOMA_INPUTS/soma_data"

        # SOMA body model assets from nvidia/soma-x
        SOMA_REPO="nvidia/soma-x"
        hf_download "$SOMA_REPO" "SOMA_neutral.npz"      "$GEM_SOMA_INPUTS/mhr_data"
        hf_download "$SOMA_REPO" "MHR/SOMA_wrap_lod1.obj" "$GEM_SOMA_INPUTS/mhr_data"
        hf_download "$SOMA_REPO" "MHR/base_body_lod6.obj" "$GEM_SOMA_INPUTS/mhr_data"
        hf_download "$SOMA_REPO" "MHR/mhr_model_lod6.pt" "$GEM_SOMA_INPUTS/mhr_data"
    fi
else
    section "GEM-SOMA checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 3. FoundationPose weights (HuggingFace nvidia/PhysicalAI-...-GRAIL)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_FOUNDATIONPOSE" = false ]; then
    section "FoundationPose weights"

    FP_SCORER="imports/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
    FP_SCORER_CFG="imports/FoundationPose/weights/2024-01-11-20-02-45/config.yml"
    FP_REFINER="imports/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"
    FP_REFINER_CFG="imports/FoundationPose/weights/2023-10-28-18-33-37/config.yml"

    if [ -f "$FP_SCORER" ] && [ -f "$FP_SCORER_CFG" ] && \
       [ -f "$FP_REFINER" ] && [ -f "$FP_REFINER_CFG" ] && [ "$FORCE" = false ]; then
        ok "FoundationPose weights (already exist)"
    else
        python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
            fail "huggingface_hub is required. Install with: pip install huggingface_hub"
            exit 1
        }

        # Two-network pipeline: scorer ranks pose hypotheses, refiner regresses pose deltas.
        # Timestamp dir names are load-path sensitive — FoundationPose's loader hardcodes them.
        hf_grail_download "FoundationPose/weights/2024-01-11-20-02-45/config.yml"
        hf_grail_download "FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
        hf_grail_download "FoundationPose/weights/2023-10-28-18-33-37/config.yml"
        hf_grail_download "FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"
    fi
else
    section "FoundationPose weights (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 4. Hunyuan3D-2.1 (RealESRGAN; shape/paint models auto-download on first use)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_HUNYUAN3D" = false ]; then
    section "Hunyuan3D-2.1 checkpoints"

    REALESRGAN="imports/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"

    if [ -f "$REALESRGAN" ] && [ "$FORCE" = false ]; then
        ok "RealESRGAN (already exists)"
    else
        mkdir -p "$(dirname "$REALESRGAN")"
        echo "  Downloading RealESRGAN_x4plus.pth..."
        wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
            -O "$REALESRGAN"
        if [ -f "$REALESRGAN" ]; then
            ok "RealESRGAN_x4plus.pth"
        else
            fail "RealESRGAN download failed"
        fi
    fi

    echo "  Note: Hunyuan3D shape/paint models will auto-download from HuggingFace on first use."
else
    section "Hunyuan3D-2.1 checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 5. SONIC reference checkpoints (HuggingFace nvidia/PhysicalAI-...-GRAIL)
# ════════════════════════════════════════════════════════════════════════════
SONIC_BASE_DIR="SONIC/models/sonic_manipulation_base"
SONIC_PNP_TABLE_WARMSTART_DIR="SONIC/models/pnp_table_warmstart"
SONIC_PNP_GROUND_WARMSTART_DIR="SONIC/models/pnp_ground_warmstart"
SONIC_STAIRS_WARMSTART_DIR="SONIC/models/terrain_stairs_warmstart"
SONIC_PNP_TABLE_RELEASE_DIR="SONIC/models/pnp_table_release"
SONIC_PNP_GROUND_RELEASE_DIR="SONIC/models/pnp_ground_release"
SONIC_TERRAIN_RELEASE_DIR="SONIC/models/terrain_release"

if [ "$SKIP_SONIC" = false ]; then
    section "SONIC checkpoints"

    SONIC_BASE_MARKER="imports/$SONIC_BASE_DIR/last.pt"

    if [ -f "$SONIC_BASE_MARKER" ] && [ "$FORCE" = false ]; then
        ok "SONIC checkpoints (already exist)"
    else
        python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
            fail "huggingface_hub is required. Install with: pip install huggingface_hub"
            exit 1
        }

        hf_grail_download "$SONIC_PNP_TABLE_WARMSTART_DIR/last.pt"
        hf_grail_download "$SONIC_PNP_TABLE_WARMSTART_DIR/config.yaml"
        hf_grail_download "$SONIC_PNP_GROUND_WARMSTART_DIR/last.pt"
        hf_grail_download "$SONIC_PNP_GROUND_WARMSTART_DIR/config.yaml"
        hf_grail_download "$SONIC_STAIRS_WARMSTART_DIR/last.pt"
        hf_grail_download "$SONIC_STAIRS_WARMSTART_DIR/config.yaml"
        hf_grail_download "$SONIC_PNP_TABLE_RELEASE_DIR/last.pt"
        hf_grail_download "$SONIC_PNP_TABLE_RELEASE_DIR/config.yaml"
        hf_grail_download "$SONIC_PNP_GROUND_RELEASE_DIR/last.pt"
        hf_grail_download "$SONIC_PNP_GROUND_RELEASE_DIR/config.yaml"
        hf_grail_download "$SONIC_TERRAIN_RELEASE_DIR/last.pt"
        hf_grail_download "$SONIC_TERRAIN_RELEASE_DIR/config.yaml"
        hf_grail_download "$SONIC_BASE_DIR/model_config.yaml"
        hf_grail_download "$SONIC_BASE_DIR/last.pt"
    fi
else
    section "SONIC checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 6. Validation
# ════════════════════════════════════════════════════════════════════════════
section "Validating checkpoint files"

ERRORS=0

check_file() {
    local path="$1"
    local description="$2"
    if [ -f "$path" ]; then
        ok "$description"
    else
        fail "$description: $path"
        ERRORS=$((ERRORS + 1))
    fi
}

# GEM-SMPL
check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"     "SMPL neutral body model (manual)"
check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smpl/J_regressor_extra.npy" "SMPL extra joint regressor"
check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt" "SMPL neutral joint regressor"
check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smplx2smpl_sparse.pt"     "SMPL-X to SMPL sparse mapper"
check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" "SMPL-X body model (manual)"
check_file "imports/GEM-SMPL/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"        "HMR2 checkpoint"
check_file "imports/GEM-SMPL/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"      "ViTPose (GEM-SMPL)"
check_file "imports/GEM-SMPL/inputs/checkpoints/vimo/smpl_mean_params.npz"             "VIMO SMPL mean params"
check_file "imports/GEM-SMPL/inputs/checkpoints/vimo/vimo_checkpoint.pth.tar"          "VIMO checkpoint"
check_file "imports/GEM-SMPL/inputs/checkpoints/yolo/yolov8x.pt"                       "YOLOv8x detector"
check_file "imports/GEM-SMPL/outputs/mocap_mixed_v1/genmo/genmo_lg_jukebox_jukebox_new/version_0/checkpoints/last.ckpt" "HMR4D (GENMO) checkpoint"

# GEM-SOMA
check_file "imports/GEM-SOMA/inputs/pretrained/gem_soma.ckpt"                          "GEM-SOMA pretrained"
check_file "imports/GEM-SOMA/inputs/mhr_data/mhr_model.pt"                             "SOMA MHR model"
check_file "imports/GEM-SOMA/inputs/checkpoints/vitpose/vitpose.pth"                   "ViTPose (GEM-SOMA)"
check_file "imports/GEM-SOMA/inputs/soma_data/scale_mean.pth"                          "SOMA scale params"
check_file "imports/GEM-SOMA/inputs/mhr_data/SOMA_neutral.npz"                         "SOMA core asset"
check_file "imports/GEM-SOMA/inputs/mhr_data/MHR/SOMA_wrap_lod1.obj"                   "SOMA SOMA wrap"
check_file "imports/GEM-SOMA/inputs/mhr_data/MHR/mhr_model_lod6.pt"                    "SOMA MHR model"
check_file "imports/GEM-SOMA/inputs/mhr_data/MHR/base_body_lod6.obj"                   "SOMA base body"

# FoundationPose
check_file "imports/FoundationPose/weights/2024-01-11-20-02-45/config.yml"            "FoundationPose ScorePredictor config"
check_file "imports/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"         "FoundationPose ScorePredictor"
check_file "imports/FoundationPose/weights/2023-10-28-18-33-37/config.yml"            "FoundationPose PoseRefinePredictor config"
check_file "imports/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"         "FoundationPose PoseRefinePredictor"

# Hunyuan3D-2.1
check_file "imports/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"               "RealESRGAN (Hunyuan3D)"

# SONIC
check_file "imports/$SONIC_BASE_DIR/last.pt"            "SONIC base behavior-model checkpoint"
check_file "imports/$SONIC_BASE_DIR/model_config.yaml"  "SONIC base behavior-model config"
check_file "imports/$SONIC_PNP_TABLE_WARMSTART_DIR/last.pt"       "SONIC pnp_table warmstart checkpoint"
check_file "imports/$SONIC_PNP_TABLE_WARMSTART_DIR/config.yaml"   "SONIC pnp_table warmstart config"
check_file "imports/$SONIC_PNP_GROUND_WARMSTART_DIR/last.pt"      "SONIC pnp_ground warmstart checkpoint"
check_file "imports/$SONIC_PNP_GROUND_WARMSTART_DIR/config.yaml"  "SONIC pnp_ground warmstart config"
check_file "imports/$SONIC_STAIRS_WARMSTART_DIR/last.pt"         "SONIC terrain stairs warmstart checkpoint"
check_file "imports/$SONIC_STAIRS_WARMSTART_DIR/config.yaml"     "SONIC terrain stairs warmstart config"
check_file "imports/$SONIC_PNP_TABLE_RELEASE_DIR/last.pt"       "SONIC pnp_table checkpoint"
check_file "imports/$SONIC_PNP_TABLE_RELEASE_DIR/config.yaml"   "SONIC pnp_table config"
check_file "imports/$SONIC_PNP_GROUND_RELEASE_DIR/last.pt"      "SONIC pnp_ground checkpoint"
check_file "imports/$SONIC_PNP_GROUND_RELEASE_DIR/config.yaml"  "SONIC pnp_ground config"
check_file "imports/$SONIC_TERRAIN_RELEASE_DIR/last.pt"         "SONIC terrain checkpoint"
check_file "imports/$SONIC_TERRAIN_RELEASE_DIR/config.yaml"     "SONIC terrain config"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checkpoints are present.${NC}"
else
    echo -e "${YELLOW}$ERRORS checkpoint file(s) missing. See above for details.${NC}"
fi
