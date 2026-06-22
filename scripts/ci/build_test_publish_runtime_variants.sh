#!/usr/bin/env bash
# Build release runtime image variants, run serialized E2E, then push tags.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/ci/build_test_publish_runtime_variants.sh [options]

Options:
  --image IMAGE          Image repository. Default: REGISTRY/IMAGE_NAME.
  --branch BRANCH        Branch name. Default: GITHUB_REF_NAME.
  --sha SHA              Commit SHA. Default: GITHUB_SHA.
  --push                 Push all tags after every E2E subset passes.
  --no-push              Build/test only. Default.
  --dry-run              Print build/test/push commands without running them.
  -h, --help             Show this help.

This script intentionally builds and tests variants serially on one runner to
avoid oversubscribing GPUs. It pushes no image until every requested container
E2E segment has passed.

EOF
}

die() {
    echo "build_test_publish_runtime_variants: error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

registry="${REGISTRY:-ghcr.io}"
image_name="${IMAGE_NAME:-${GITHUB_REPOSITORY:-}}"
image=""
branch="${GITHUB_REF_NAME:-}"
sha="${GITHUB_SHA:-}"
push=0
dry_run=false

# The self-hosted ARC runner invokes Docker builds from inside Kubernetes. Host
# networking avoids TLS stalls to codeload.github.com seen from BuildKit RUN
# layers while preserving the exact dependency versions fetched by the Dockerfile.
export LLAMINAR_DOCKER_BUILD_NETWORK="${LLAMINAR_DOCKER_BUILD_NETWORK:-host}"

while (($#)); do
    case "$1" in
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
        --push)
            push=1
            shift
            ;;
        --no-push)
            push=0
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
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

if [[ -z "$image" ]]; then
    [[ -n "$image_name" ]] || die "IMAGE_NAME or GITHUB_REPOSITORY is required when --image is omitted"
    image="${registry}/${image_name}"
fi
image="${image,,}"

if [[ -z "$branch" ]]; then
    branch="$(git -C "$repo_root" branch --show-current 2>/dev/null || true)"
fi
if [[ -z "$sha" ]]; then
    sha="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
fi
[[ -n "$branch" ]] || die "branch could not be resolved"
[[ -n "$sha" ]] || die "sha could not be resolved"

variants=(cpu cuda rocm full)
declare -A e2e_variant=(
    [cpu]=cpu
    [cuda]=cuda
    [rocm]=rocm
    [full]=hybrid
)
declare -A port=(
    [cpu]=20100
    [cuda]=20200
    [rocm]=20300
    [full]=20400
)

all_refs=()

cleanup_images() {
    if [[ "${LLAMINAR_KEEP_RUNTIME_IMAGES:-0}" == "1" ]]; then
        echo "[build-test-publish] keeping runtime image tags because LLAMINAR_KEEP_RUNTIME_IMAGES=1"
        return
    fi
    ((${#all_refs[@]} > 0)) || return
    echo "[build-test-publish] removing runtime image tags; BuildKit layer cache remains available"
    local ref
    for ref in "${all_refs[@]}"; do
        docker image rm "$ref" >/dev/null 2>&1 || true
    done
}
trap cleanup_images EXIT

run_cmd() {
    printf '[build-test-publish] '
    printf '%q ' "$@"
    printf '\n'
    if [[ "$dry_run" != "true" ]]; then
        "$@"
    fi
}

for variant in "${variants[@]}"; do
    mapfile -t refs < <("${script_dir}/runtime_image_tags.sh" \
        --variant "$variant" \
        --image "$image" \
        --branch "$branch" \
        --sha "$sha" \
        --format refs)
    ((${#refs[@]} > 0)) || die "no tags generated for variant ${variant}"
    all_refs+=("${refs[@]}")

    build_args=("${repo_root}/scripts/docker/build-runtime-image.sh" --variant "$variant" --load --no-verify)
    for ref in "${refs[@]}"; do
        build_args+=(--tag "$ref")
    done
    run_cmd "${build_args[@]}"

    run_cmd env LLAMINAR_E2E_DOCKER_NETWORK=bridge \
        "${repo_root}/scripts/ci/run_release_container_e2e.sh" \
        --variant "${e2e_variant[$variant]}" \
        --image "${refs[0]}" \
        --port "${port[$variant]}" \
        --log-dir "/tmp/llaminar-e2e-${e2e_variant[$variant]}-container"
done

{
    echo "## Runtime image tags"
    echo ""
    for ref in "${all_refs[@]}"; do
        echo "- \`${ref}\`"
    done
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}" || true

if [[ "$push" == "1" ]]; then
    for ref in "${all_refs[@]}"; do
        run_cmd docker push "$ref"
    done
else
    echo "[build-test-publish] push disabled; built and tested tags:"
    printf '  %s\n' "${all_refs[@]}"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
        echo "image_refs<<EOF"
        printf '%s\n' "${all_refs[@]}"
        echo "EOF"
    } >> "$GITHUB_OUTPUT"
fi
