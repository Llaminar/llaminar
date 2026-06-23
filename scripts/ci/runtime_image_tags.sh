#!/usr/bin/env bash
# Resolve Llaminar runtime image tags for a backend variant.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/ci/runtime_image_tags.sh --variant cpu|cuda|rocm|full --image IMAGE [options]

Options:
  --variant VARIANT      Runtime image variant.
  --image IMAGE          Image repository, for example ghcr.io/owner/repo.
  --branch BRANCH        Branch name. Default: GITHUB_REF_NAME or git branch.
  --sha SHA              Commit SHA. Default: GITHUB_SHA or git HEAD.
  --cuda-version VALUE   CUDA version for tag segment. Default: CUDA_VERSION or 13.0.
  --rocm-version VALUE   ROCm version for tag segment. Default: ROCM_VERSION or 7.1.1.
  --tag-suffix SUFFIX    Append a suffix to every emitted tag, for example -avx2.
  --format refs|tags     Output full refs or tag names only. Default: refs.
  -h, --help             Show this help.

Tag shape:
  cpu   -> branch-sha-cpu, branch-cpu-latest
  cuda  -> branch-sha-cuda13.0, branch-cuda13.0-latest
  rocm  -> branch-sha-rocm7.1.1, branch-rocm7.1.1-latest
  full  -> branch-sha-cuda13.0-rocm7.1.1,
           branch-cuda13.0-rocm7.1.1-latest,
           branch-latest

The CPU segment avoids colliding with the full image's branch-latest alias while
keeping CUDA/ROCm segments exclusive to images that actually contain those libs.
EOF
}

die() {
    echo "runtime_image_tags: error: $*" >&2
    exit 1
}

variant=""
image=""
branch="${GITHUB_REF_NAME:-}"
sha="${GITHUB_SHA:-}"
cuda_version="${CUDA_VERSION:-13.0}"
rocm_version="${ROCM_VERSION:-7.1.1}"
tag_suffix="${LLAMINAR_IMAGE_TAG_SUFFIX:-}"
format="refs"

while (($#)); do
    case "$1" in
        --variant)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            variant="$2"
            shift 2
            ;;
        --image)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            image="$2"
            shift 2
            ;;
        --branch)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            branch="$2"
            shift 2
            ;;
        --sha)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            sha="$2"
            shift 2
            ;;
        --cuda-version)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            cuda_version="$2"
            shift 2
            ;;
        --rocm-version)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            rocm_version="$2"
            shift 2
            ;;
        --tag-suffix)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            tag_suffix="$2"
            shift 2
            ;;
        --format)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            format="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -n "$variant" ]] || die "--variant is required"
[[ -n "$image" ]] || die "--image is required"

if [[ -z "$branch" ]]; then
    branch="$(git branch --show-current 2>/dev/null || true)"
fi
if [[ -z "$sha" ]]; then
    sha="$(git rev-parse HEAD 2>/dev/null || true)"
fi
[[ -n "$branch" ]] || die "branch could not be resolved"
[[ -n "$sha" ]] || die "sha could not be resolved"

sanitize_tag_part() {
    local value="$1"
    value="${value,,}"
    value="$(printf '%s' "$value" | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
    [[ -n "$value" ]] || value="unknown"
    printf '%s' "$value"
}

image="${image,,}"
branch="$(sanitize_tag_part "$branch")"
short_sha="$(sanitize_tag_part "${sha:0:7}")"
cuda_part="$(sanitize_tag_part "$cuda_version")"
rocm_part="$(sanitize_tag_part "$rocm_version")"
tag_suffix_part=""
if [[ -n "$tag_suffix" ]]; then
    tag_suffix_part="$(sanitize_tag_part "$tag_suffix")"
    [[ -n "$tag_suffix_part" && "$tag_suffix_part" != "unknown" ]] \
        || die "--tag-suffix resolved to an empty tag segment"
fi

case "$format" in
    refs|tags) ;;
    *) die "--format must be refs or tags" ;;
esac

case "$variant" in
    cpu)
        segments=("cpu")
        latest_segments=("cpu")
        ;;
    cuda)
        segments=("cuda${cuda_part}")
        latest_segments=("cuda${cuda_part}")
        ;;
    rocm)
        segments=("rocm${rocm_part}")
        latest_segments=("rocm${rocm_part}")
        ;;
    full)
        segments=("cuda${cuda_part}" "rocm${rocm_part}")
        latest_segments=("cuda${cuda_part}" "rocm${rocm_part}")
        ;;
    *)
        die "unsupported variant '$variant'"
        ;;
esac

segment_suffix="$(IFS=-; printf '%s' "${segments[*]}")"
latest_suffix="$(IFS=-; printf '%s' "${latest_segments[*]}")"

tags=(
    "${branch}-${short_sha}-${segment_suffix}"
    "${branch}-${latest_suffix}-latest"
)
if [[ "$variant" == "full" ]]; then
    tags+=("${branch}-latest")
fi

if [[ -n "$tag_suffix_part" ]]; then
    for idx in "${!tags[@]}"; do
        tags[$idx]="${tags[$idx]}-${tag_suffix_part}"
    done
fi

for tag in "${tags[@]}"; do
    if [[ "$format" == "refs" ]]; then
        printf '%s:%s\n' "$image" "$tag"
    else
        printf '%s\n' "$tag"
    fi
done
