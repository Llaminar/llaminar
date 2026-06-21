#!/usr/bin/env bash
# Build the Llaminar release runtime container from the repo Dockerfile.
#
# This is the local/CI entry point for packaging a Release `llaminar2` binary
# into the slim runtime image. The Dockerfile still owns the actual dependency
# install and binary copy; this script keeps the docker build invocation shared.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/docker/build-runtime-image.sh [options]

Builds the Dockerfile `runtime` target, which compiles Llaminar in the builder
stage and copies the Release `llaminar2` binary into /usr/local/bin/llaminar2.

Options:
  -t, --tag IMAGE          Image tag to apply. Repeatable.
                           Default: llaminar:local
      --push               Push tags to their registry. Requires an explicit tag.
      --load               Load the image into the local Docker daemon. Default.
      --verify             Run `llaminar2 --help` inside the image.
      --no-verify          Skip the local packaging smoke test.
      --version VALUE      VERSION Docker build arg. Default: git describe/local.
      --vcs-ref SHA        VCS_REF Docker build arg. Default: git HEAD/unknown.
      --build-date DATE    BUILD_DATE Docker build arg. Default: current UTC time.
      --build-type TYPE    LLAMINAR_BUILD_TYPE build arg. Default: Release.
      --variant VARIANT    Backend/runtime variant: full, cpu, cuda, rocm.
                           Default: full.
      --cpu-only           Alias for --variant cpu.
      --cuda-only          Alias for --variant cuda.
      --rocm-only          Alias for --variant rocm.
      --cuda-archs LIST    LLAMINAR_CUDA_ARCHS build arg. Dockerfile default if unset.
      --rccl-from-source   Build/source-package RCCL for ROCm multi-GPU.
                           Default.
      --no-rccl-from-source
                           Use packaged RCCL from the ROCm apt repository.
      --rccl-gpu-targets LIST
                           RCCL GPU_TARGETS for source builds. Default:
                           Dockerfile default (gfx906).
      --rocm-runtime-gpu-targets LIST
                           Prune ROCm runtime kernel payloads to these gfx
                           targets (comma/semicolon/space-separated). Default:
                           same value as --rccl-gpu-targets when provided.
      --rccl-msccl-kernel  Build RCCL's optional generated MSCCL kernels.
      --no-rccl-msccl-kernel
                           Disable RCCL's optional generated MSCCL kernels.
                           Default.
      --rccl-only-funcs PATTERN
                           RCCL ONLY_FUNCS generator pattern for source builds.
                           Default: Llaminar inference collectives/datatypes.
      --full-rccl-funcs    Do not pass ONLY_FUNCS; build RCCL's full generated
                           function matrix.
      --skip-integration-build
                           Skip Dockerfile integration build before packaging.
                           Default for this script.
      --include-integration-build
                           Build integration binaries too, matching the full
                           CI builder image behavior.
      --build-arg ARG      Extra Docker build arg, KEY=VALUE. Repeatable.
      --label LABEL        Extra image label, KEY=VALUE. Repeatable.
      --platform PLATFORM  Optional docker buildx platform.
      --build-network MODE Optional network mode for Dockerfile RUN steps.
                           Example: host.
      --cache-from SPEC    BuildKit cache import spec. Repeatable.
                           Example: type=registry,ref=ghcr.io/acme/llaminar-cache:runtime-full
      --cache-to SPEC      BuildKit cache export spec. Repeatable.
                           Example: type=registry,ref=ghcr.io/acme/llaminar-cache:runtime-full,mode=max
      --progress MODE      BuildKit progress mode. Default: auto.
      --dry-run            Print the docker command without running it.
  -h, --help               Show this help.

Environment:
  LLAMINAR_IMAGE_TAGS      Newline- or comma-separated image tags.
  LLAMINAR_IMAGE_LABELS    Newline-separated image labels.
  VERSION, VCS_REF, BUILD_DATE, LLAMINAR_BUILD_TYPE, LLAMINAR_IMAGE_VARIANT,
  LLAMINAR_ENABLE_CUDA, LLAMINAR_ENABLE_ROCM, LLAMINAR_CUDA_ARCHS,
  LLAMINAR_SKIP_INTEGRATION, LLAMINAR_BUILD_RCCL_FROM_SOURCE,
  RCCL_GPU_TARGETS, ROCM_RUNTIME_GPU_TARGETS, RCCL_ENABLE_MSCCL_KERNEL,
  RCCL_ONLY_FUNCS, LLAMINAR_DOCKER_BUILD_NETWORK,
  LLAMINAR_DOCKER_CACHE_FROM, LLAMINAR_DOCKER_CACHE_TO
                           Defaults for the matching options/build args.
                           Cache env vars are newline-separated BuildKit specs.

Examples:
  scripts/docker/build-runtime-image.sh --tag llaminar:local
  scripts/docker/build-runtime-image.sh --tag ghcr.io/acme/llaminar:v1 --push
EOF
}

die() {
    echo "build-runtime-image: error: $*" >&2
    exit 1
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

append_split_tags() {
    local raw="${1//$'\r'/}"
    raw="${raw//,/$'\n'}"
    local line
    while IFS= read -r line; do
        line="$(trim "${line}")"
        [[ -n "${line}" ]] && tags+=("${line}")
    done <<< "${raw}"
}

append_split_labels() {
    local raw="${1//$'\r'/}"
    local line
    while IFS= read -r line; do
        line="$(trim "${line}")"
        [[ -n "${line}" ]] && labels+=("${line}")
    done <<< "${raw}"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

default_rccl_only_funcs='AllReduce RING/TREE LL/LL128/SIMPLE Sum i8/i32/f16/f32/bf16|Reduce RING LL/LL128/SIMPLE Sum i8/i32/f16/f32/bf16|ReduceScatter RING/PAT LL/LL128/SIMPLE Sum i8/i32/f16/f32/bf16|AllGather RING/PAT LL/LL128/SIMPLE Sum i8/i32/f16/f32/bf16|Broadcast RING LL/LL128/SIMPLE Sum i8/i32/f16/f32/bf16|SendRecv RING SIMPLE Sum i8'

tags=()
labels=()
cache_from=()
cache_to=()
extra_build_args=()
output_mode="${LLAMINAR_IMAGE_OUTPUT:-load}"
verify_mode="auto"
dry_run=false
platform=""
build_network="${LLAMINAR_DOCKER_BUILD_NETWORK:-}"
progress="${BUILDKIT_PROGRESS:-auto}"
version="${VERSION:-}"
vcs_ref="${VCS_REF:-}"
build_date="${BUILD_DATE:-}"
build_type="${LLAMINAR_BUILD_TYPE:-Release}"
image_variant="${LLAMINAR_IMAGE_VARIANT:-full}"
enable_cuda="${LLAMINAR_ENABLE_CUDA:-}"
enable_rocm="${LLAMINAR_ENABLE_ROCM:-}"
cuda_archs="${LLAMINAR_CUDA_ARCHS:-}"
skip_integration="${LLAMINAR_SKIP_INTEGRATION:-1}"
build_rccl_from_source="${LLAMINAR_BUILD_RCCL_FROM_SOURCE:-ON}"
rccl_gpu_targets="${RCCL_GPU_TARGETS:-}"
rocm_runtime_gpu_targets="${ROCM_RUNTIME_GPU_TARGETS:-}"
rccl_enable_msccl_kernel="${RCCL_ENABLE_MSCCL_KERNEL:-OFF}"
rccl_only_funcs="${RCCL_ONLY_FUNCS:-${default_rccl_only_funcs}}"

if [[ -n "${LLAMINAR_IMAGE_TAGS:-}" ]]; then
    append_split_tags "${LLAMINAR_IMAGE_TAGS}"
fi
if [[ -n "${LLAMINAR_IMAGE_LABELS:-}" ]]; then
    append_split_labels "${LLAMINAR_IMAGE_LABELS}"
fi
if [[ -n "${LLAMINAR_DOCKER_CACHE_FROM:-}" ]]; then
    while IFS= read -r cache_spec; do
        cache_spec="$(trim "${cache_spec}")"
        [[ -n "${cache_spec}" ]] && cache_from+=("${cache_spec}")
    done <<< "${LLAMINAR_DOCKER_CACHE_FROM//$'\r'/}"
fi
if [[ -n "${LLAMINAR_DOCKER_CACHE_TO:-}" ]]; then
    while IFS= read -r cache_spec; do
        cache_spec="$(trim "${cache_spec}")"
        [[ -n "${cache_spec}" ]] && cache_to+=("${cache_spec}")
    done <<< "${LLAMINAR_DOCKER_CACHE_TO//$'\r'/}"
fi

while (($#)); do
    case "$1" in
        -t|--tag)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            tags+=("$2")
            shift 2
            ;;
        --push)
            output_mode="push"
            shift
            ;;
        --load)
            output_mode="load"
            shift
            ;;
        --verify)
            verify_mode="yes"
            shift
            ;;
        --no-verify)
            verify_mode="no"
            shift
            ;;
        --version)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            version="$2"
            shift 2
            ;;
        --vcs-ref)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            vcs_ref="$2"
            shift 2
            ;;
        --build-date)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            build_date="$2"
            shift 2
            ;;
        --build-type)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            build_type="$2"
            shift 2
            ;;
        --variant)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            image_variant="$2"
            shift 2
            ;;
        --cpu-only)
            image_variant="cpu"
            shift
            ;;
        --cuda-only)
            image_variant="cuda"
            shift
            ;;
        --rocm-only)
            image_variant="rocm"
            shift
            ;;
        --cuda-archs)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            cuda_archs="$2"
            shift 2
            ;;
        --rccl-from-source)
            build_rccl_from_source="ON"
            shift
            ;;
        --no-rccl-from-source)
            build_rccl_from_source="OFF"
            shift
            ;;
        --rccl-gpu-targets)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            rccl_gpu_targets="$2"
            shift 2
            ;;
        --rocm-runtime-gpu-targets)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            rocm_runtime_gpu_targets="$2"
            shift 2
            ;;
        --rccl-msccl-kernel)
            rccl_enable_msccl_kernel="ON"
            shift
            ;;
        --no-rccl-msccl-kernel)
            rccl_enable_msccl_kernel="OFF"
            shift
            ;;
        --rccl-only-funcs)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            rccl_only_funcs="$2"
            shift 2
            ;;
        --full-rccl-funcs)
            rccl_only_funcs=""
            shift
            ;;
        --skip-integration-build)
            skip_integration="1"
            shift
            ;;
        --include-integration-build)
            skip_integration="0"
            shift
            ;;
        --build-arg)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            extra_build_args+=("$2")
            shift 2
            ;;
        --label)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            labels+=("$2")
            shift 2
            ;;
        --platform)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            platform="$2"
            shift 2
            ;;
        --build-network)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            build_network="$2"
            shift 2
            ;;
        --cache-from)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            cache_from+=("$2")
            shift 2
            ;;
        --cache-to)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            cache_to+=("$2")
            shift 2
            ;;
        --progress)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            progress="$2"
            shift 2
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

case "${output_mode}" in
    load|push|none) ;;
    *) die "unsupported output mode '${output_mode}' (expected load, push, or none)" ;;
esac
case "${skip_integration}" in
    0|1) ;;
    *) die "LLAMINAR_SKIP_INTEGRATION must be 0 or 1" ;;
esac
case "${build_rccl_from_source}" in
    ON|OFF|on|off|1|0|TRUE|FALSE|true|false) ;;
    *) die "LLAMINAR_BUILD_RCCL_FROM_SOURCE must be ON or OFF" ;;
esac
case "${rccl_enable_msccl_kernel}" in
    ON|OFF|on|off|1|0|TRUE|FALSE|true|false) ;;
    *) die "RCCL_ENABLE_MSCCL_KERNEL must be ON or OFF" ;;
esac
case "${image_variant}" in
    full|cpu|cuda|rocm) ;;
    *) die "unsupported --variant '${image_variant}' (expected full, cpu, cuda, or rocm)" ;;
esac

normalize_bool_arg() {
    case "${1}" in
        ON|on|1|TRUE|true|YES|yes) printf 'ON' ;;
        OFF|off|0|FALSE|false|NO|no) printf 'OFF' ;;
        *) return 1 ;;
    esac
}

case "${image_variant}" in
    full)
        : "${enable_cuda:=ON}"
        : "${enable_rocm:=ON}"
        ;;
    cpu)
        : "${enable_cuda:=OFF}"
        : "${enable_rocm:=OFF}"
        ;;
    cuda)
        : "${enable_cuda:=ON}"
        : "${enable_rocm:=OFF}"
        ;;
    rocm)
        : "${enable_cuda:=OFF}"
        : "${enable_rocm:=ON}"
        ;;
esac
enable_cuda="$(normalize_bool_arg "${enable_cuda}")" || die "LLAMINAR_ENABLE_CUDA must be ON or OFF"
enable_rocm="$(normalize_bool_arg "${enable_rocm}")" || die "LLAMINAR_ENABLE_ROCM must be ON or OFF"
if [[ "${enable_rocm}" == "OFF" ]]; then
    build_rccl_from_source="OFF"
    rccl_only_funcs=""
    rccl_gpu_targets=""
    rocm_runtime_gpu_targets=""
fi

used_default_tag=false
if ((${#tags[@]} == 0)); then
    tags=("llaminar:local")
    used_default_tag=true
fi

if [[ "${output_mode}" == "push" && "${used_default_tag}" == "true" ]]; then
    die "--push requires --tag or LLAMINAR_IMAGE_TAGS"
fi

if [[ -z "${version}" ]]; then
    version="$(git -C "${repo_root}" describe --tags --always --dirty 2>/dev/null || printf 'local')"
fi
if [[ -z "${vcs_ref}" ]]; then
    vcs_ref="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
fi
if [[ -z "${build_date}" ]]; then
    build_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

verify=false
case "${verify_mode}" in
    auto)
        [[ "${output_mode}" == "load" ]] && verify=true
        ;;
    yes)
        [[ "${output_mode}" == "load" ]] || die "--verify requires --load"
        verify=true
        ;;
    no)
        verify=false
        ;;
esac

command -v docker >/dev/null 2>&1 || die "docker CLI is not available"
docker buildx version >/dev/null 2>&1 || die "docker buildx is not available"

cmd=(
    docker buildx build
    --file "${repo_root}/Dockerfile"
    --target runtime
    --build-arg "VERSION=${version}"
    --build-arg "VCS_REF=${vcs_ref}"
    --build-arg "BUILD_DATE=${build_date}"
    --build-arg "LLAMINAR_BUILD_TYPE=${build_type}"
    --build-arg "LLAMINAR_ENABLE_CUDA=${enable_cuda}"
    --build-arg "LLAMINAR_ENABLE_ROCM=${enable_rocm}"
    --build-arg "LLAMINAR_SKIP_INTEGRATION=${skip_integration}"
    --build-arg "LLAMINAR_BUILD_RCCL_FROM_SOURCE=${build_rccl_from_source}"
    --build-arg "RCCL_ENABLE_MSCCL_KERNEL=${rccl_enable_msccl_kernel}"
)

if [[ -n "${cuda_archs}" ]]; then
    cmd+=(--build-arg "LLAMINAR_CUDA_ARCHS=${cuda_archs}")
fi
if [[ -n "${rccl_gpu_targets}" ]]; then
    cmd+=(--build-arg "RCCL_GPU_TARGETS=${rccl_gpu_targets}")
fi
if [[ -z "${rocm_runtime_gpu_targets}" && -n "${rccl_gpu_targets}" ]]; then
    rocm_runtime_gpu_targets="${rccl_gpu_targets}"
fi
if [[ -n "${rocm_runtime_gpu_targets}" ]]; then
    cmd+=(--build-arg "ROCM_RUNTIME_GPU_TARGETS=${rocm_runtime_gpu_targets}")
fi
if [[ -n "${rccl_only_funcs}" ]]; then
    cmd+=(--build-arg "RCCL_ONLY_FUNCS=${rccl_only_funcs}")
fi
if [[ -n "${platform}" ]]; then
    cmd+=(--platform "${platform}")
fi
if [[ -n "${build_network}" ]]; then
    cmd+=(--network "${build_network}")
fi
if [[ -n "${progress}" ]]; then
    cmd+=(--progress "${progress}")
fi
for tag in "${tags[@]}"; do
    cmd+=(--tag "${tag}")
done
for label in "${labels[@]}"; do
    cmd+=(--label "${label}")
done
for cache_spec in "${cache_from[@]}"; do
    cmd+=(--cache-from "${cache_spec}")
done
for cache_spec in "${cache_to[@]}"; do
    cmd+=(--cache-to "${cache_spec}")
done
for build_arg in "${extra_build_args[@]}"; do
    cmd+=(--build-arg "${build_arg}")
done
case "${output_mode}" in
    load)
        cmd+=(--load)
        ;;
    push)
        cmd+=(--push)
        ;;
    none)
        ;;
esac
cmd+=("${repo_root}")

printf '[build-runtime-image] '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${dry_run}" == "true" ]]; then
    exit 0
fi

"${cmd[@]}"

if [[ "${verify}" == "true" ]]; then
    echo "[build-runtime-image] verifying ${tags[0]} can start /usr/local/bin/llaminar2"
    docker run --rm --entrypoint /usr/local/bin/llaminar2 "${tags[0]}" \
        --help >/dev/null
fi

echo "[build-runtime-image] built ${tags[*]} (variant=${image_variant}, cuda=${enable_cuda}, rocm=${enable_rocm})"
