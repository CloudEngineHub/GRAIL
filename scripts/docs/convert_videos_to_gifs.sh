#!/usr/bin/env bash
# Mirror the GRAIL project-page videos (research.nvidia.com/labs/dair/grail/)
# as looping GIFs under assets/videos/ for embedding in README.md and docs/.
#
# Usage:
#   bash scripts/docs/convert_videos_to_gifs.sh
#
# Re-running is idempotent: existing GIFs are overwritten.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_PAGE="https://research.nvidia.com/labs/dair/grail"
TMPDIR="${TMPDIR:-/tmp}/grail_mp4s"
OUT_DIR="${REPO_ROOT}/assets/videos"
ASSETS_DIR="${REPO_ROOT}/assets"

mkdir -p "${TMPDIR}" "${OUT_DIR}"

# Filename → staging-relative path. One representative MP4 per showcase row.
declare -A VIDEOS=(
    ["teaser"]="static/videos/teaser.mp4"
    ["pickup_table"]="static/videos/pickup/alcohol_11___11_loose_full_alcohol_11_jason_rigged_001_indoor2-v62_rand00001-20260308_202733__right-2699.mp4"
    ["pickup_ground"]="static/videos/pickup/apple_22__22_jason_rigged_001_indoor2-pickup-ground-v2_start00001-end00001-20260309_220504__front-3804.mp4"
    ["manip_small"]="static/videos/manipulation/ground_small/plyometric_jump_box_small___plyometric_jump_box_small_jason_rigged_001_indoor1-v7_rand00005-20260310_224749__right-4012.mp4"
    ["manip_large"]="static/videos/manipulation/ground_large/wooden_barrel_planter__full_wooden_barrel_planter_jason_rigged_001_indoor1-v7_rand00002-20260310_224951__right-3997.mp4"
    ["manip_tabletop"]="static/videos/manipulation/tabletop/watering_can_b-3913.mp4"
    ["sitting"]="static/videos/sitting/eames_molded_plastic_chair__eames_molded_plastic_chair_jason_rigged_001_indoor1-v7_rand00001-20260307_183841__back-3261.mp4"
    ["terrain_slopes"]="static/videos/terrain/slope_mesh_009-3347.mp4"
    ["terrain_curbs"]="static/videos/terrain/curb_004-3397.mp4"
    ["terrain_stairs"]="static/videos/terrain/stairs_002__341_stairs_002_stairs_002_jason_rigged_001_indoor1-v13_rand00001-20260306_140248__right-3250.mp4"
    ["deployment_egocentric_views"]="static/videos/collage_horizontal.mp4"
    ["deployment_pickup"]="static/videos/deployment/pickup/pickup_clip_10.mp4"
    ["deployment_stairs"]="static/videos/deployment/stairs/stair_clip_01.mp4"
    ["realworld"]="static/videos/real_deployment.mp4"
)

# Conversion knobs. Aggressive: 10 fps + 540px wide + 64-color palette +
# lossy gifsicle keeps most GIFs small. Deployment result GIFs use the full
# source sequence; other motion-gallery GIFs are capped to 5s.
FPS=10
WIDTH=540
MAX_DURATION=5

# Empty duration means use the full source clip.
declare -A MAX_DURATIONS=(
    ["deployment_pickup"]=""
    ["deployment_stairs"]=""
)

for name in "${!VIDEOS[@]}"; do
    rel="${VIDEOS[$name]}"
    mp4="${TMPDIR}/${name}.mp4"
    gif="${OUT_DIR}/${name}.gif"
    width="${WIDTH}"
    max_duration="${MAX_DURATIONS[$name]-$MAX_DURATION}"
    duration_args=()
    if [ -n "${max_duration}" ]; then
        duration_args=(-t "${max_duration}")
    fi
    if [ "${name}" = "deployment_egocentric_views" ]; then
        width=720
    fi

    if [ ! -f "${mp4}" ]; then
        echo "  fetch ${name} ← ${PROJECT_PAGE}/${rel}"
        curl -fsSL --output "${mp4}" "${PROJECT_PAGE}/${rel}"
    fi

    echo "  convert ${name} → ${gif}"
    # Two-pass palette generation; optional trim; small palette.
    palette="${TMPDIR}/${name}_palette.png"
    ffmpeg -y -loglevel error \
        "${duration_args[@]}" -i "${mp4}" \
        -vf "fps=${FPS},scale=${width}:-1:flags=lanczos,palettegen=stats_mode=diff:max_colors=64" \
        "${palette}"
    ffmpeg -y -loglevel error \
        "${duration_args[@]}" -i "${mp4}" -i "${palette}" \
        -lavfi "fps=${FPS},scale=${width}:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5" \
        -loop 0 \
        "${gif}"
    rm -f "${palette}"

    # Lossy optimize for further size reduction when gifsicle is available.
    if command -v gifsicle >/dev/null 2>&1; then
        gifsicle -O3 --lossy=80 --colors 64 "${gif}" -o "${gif}"
    else
        echo "    skip gifsicle optimization; gifsicle not found"
    fi
    sz=$(stat -c %s "${gif}")
    printf "    %s : %s bytes\n" "${name}.gif" "${sz}"
done

# Method overview diagram.
echo "  fetch method_overview.jpeg"
curl -fsSL --output "${ASSETS_DIR}/method_overview.jpeg" "${PROJECT_PAGE}/static/images/pipeline.jpeg"

echo
echo "Total assets size:"
du -sh "${ASSETS_DIR}"
echo
echo "Per-file:"
ls -lh "${OUT_DIR}" "${ASSETS_DIR}/method_overview.jpeg" 2>/dev/null
