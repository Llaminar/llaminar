#!/usr/bin/env bash
# @file build-runtime-image.sh
# @brief Build the Llaminar release runtime container from the repo Dockerfile.
#
# This is the local/CI entry point for packaging a Release `llaminar2` binary
# into the slim runtime image. The Dockerfile still owns the actual dependency
# install and binary copy; this script keeps the docker build invocation shared.
# Dry-run command planning requires neither a Docker client nor a daemon. Full
# production certification/publication is owned by run_production_pipeline.py.
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
      --cpu-isa ISA        LLAMINAR_CPU_ISA build arg: AVX512, AVX2 or NEON.
                           NEON selects linux/arm64 with the cpu or cuda variant.
                           On ARM64 the CUDA default is 121 (NVIDIA GB10).
                           Dockerfile default if unset.
      --cpu-only           Alias for --variant cpu.
      --cuda-only          Alias for --variant cuda.
      --rocm-only          Alias for --variant rocm.
      --cuda-archs LIST    Concrete CUDA targets for the app and NCCL
                           (semicolon-separated). Dockerfile default if unset.
      --rocm-runtime-gpu-targets LIST
                           Concrete ROCm GPU targets for the app and native
                           libraries (comma/semicolon/space-separated).
                           Default: shipping targets in rocm-release.env.
      --skip-integration-build
                           Skip Dockerfile integration build before packaging.
                           Default for this script.
      --include-integration-build
                           Build integration binaries too, matching the full
                           CI builder image behavior.
      --build-arg ARG      Extra Docker build arg, KEY=VALUE. Repeatable.
      --build-context ARG  Named BuildKit context, NAME=VALUE. Repeatable.
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
  LLAMINAR_CPU_ISA, LLAMINAR_ENABLE_CUDA, LLAMINAR_ENABLE_ROCM, LLAMINAR_CUDA_ARCHS,
  LLAMINAR_SKIP_INTEGRATION, ROCM_RUNTIME_GPU_TARGETS, LLAMINAR_DOCKER_BUILD_NETWORK,
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

tags=()
labels=()
cache_from=()
cache_to=()
extra_build_args=()
build_contexts=()
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
cpu_isa="${LLAMINAR_CPU_ISA:-}"
image_variant="${LLAMINAR_IMAGE_VARIANT:-full}"
enable_cuda="${LLAMINAR_ENABLE_CUDA:-}"
enable_rocm="${LLAMINAR_ENABLE_ROCM:-}"
cuda_archs="${LLAMINAR_CUDA_ARCHS:-}"
skip_integration="${LLAMINAR_SKIP_INTEGRATION:-1}"
rocm_runtime_gpu_targets="${ROCM_RUNTIME_GPU_TARGETS:-}"

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
        --cpu-isa)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            cpu_isa="$2"
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
        --rocm-runtime-gpu-targets)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            rocm_runtime_gpu_targets="$2"
            shift 2
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
        --build-context)
            [[ $# -ge 2 && "$2" == ?*=* ]] || die "$1 requires NAME=VALUE"
            build_contexts+=("$2")
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
case "${image_variant}" in
    full|cpu|cuda|rocm) ;;
    *) die "unsupported --variant '${image_variant}' (expected full, cpu, cuda, or rocm)" ;;
esac
if [[ -n "${cpu_isa}" ]]; then
    cpu_isa="${cpu_isa^^}"
    case "${cpu_isa}" in
        AVX512|AVX2|NEON) ;;
        *) die "unsupported --cpu-isa '${cpu_isa}' (expected AVX512, AVX2, or NEON)" ;;
    esac
fi

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
if [[ "${cpu_isa}" == "NEON" || "${platform}" == "linux/arm64" ]]; then
    : "${platform:=linux/arm64}"
    : "${cpu_isa:=NEON}"
    [[ "${platform}" == "linux/arm64" && "${cpu_isa}" == "NEON" ]] ||
        die "the NEON runtime requires --platform linux/arm64"
    [[ "${image_variant}" == "cpu" || "${image_variant}" == "cuda" ]] ||
        die "the ARM64 runtime requires --variant cpu or --variant cuda"
    [[ "${enable_rocm}" == "OFF" ]] || die "the pinned ROCm SDK does not support Linux ARM64"
    [[ "${skip_integration}" == "1" ]] || die "build the test-runner target with the arm64-pr inventory for ARM Unit and host tests"
    if [[ "${enable_cuda}" == "ON" ]]; then
        : "${cuda_archs:=121}"
    fi
fi
if [[ "${enable_rocm}" == "OFF" ]]; then
    rocm_runtime_gpu_targets=""
elif [[ -n "${rocm_runtime_gpu_targets}" ]]; then
    rocm_runtime_gpu_targets="$(bash "${script_dir}/rocm-image-targets.sh" "${rocm_runtime_gpu_targets}")" ||
        die "Invalid --rocm-runtime-gpu-targets"
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
)

if [[ -n "${cpu_isa}" ]]; then
    cmd+=(--build-arg "LLAMINAR_CPU_ISA=${cpu_isa}")
fi
if [[ -n "${cuda_archs}" ]]; then
    cmd+=(--build-arg "LLAMINAR_CUDA_ARCHS=${cuda_archs}")
fi
if [[ -n "${rocm_runtime_gpu_targets}" ]]; then
    cmd+=(--build-arg "ROCM_RUNTIME_GPU_TARGETS=${rocm_runtime_gpu_targets}")
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
for context in "${build_contexts[@]}"; do
    cmd+=(--build-context "${context}")
done
for cache_spec in "${cache_from[@]}"; do
    cmd+=(--cache-from "${cache_spec}")
done
for cache_spec in "${cache_to[@]}"; do
    cmd+=(--cache-to "${cache_spec}")
done
for build_arg in "${extra_build_args[@]}"; do
    case "${build_arg%%=*}" in
        LLAMINAR_BUILD_RCCL_FROM_SOURCE|RCCL_GIT_REF|RCCL_GIT_COMMIT|RCCL_ONLY_FUNCS|RCCL_ENABLE_MSCCL_KERNEL|RCCL_GPU_TARGETS)
            die "Unsupported collective dependency override '${build_arg%%=*}'"
            ;;
    esac
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

# A dry run validates and prints the build contract without requiring an
# installed Docker client or daemon. Execution alone needs those dependencies.
command -v docker >/dev/null 2>&1 || die "docker CLI is not available"
docker buildx version >/dev/null 2>&1 || die "docker buildx is not available"

"${cmd[@]}"

if [[ "${verify}" == "true" ]]; then
    echo "[build-runtime-image] verifying ${tags[0]} can start /usr/local/bin/llaminar2"
    docker run --rm --entrypoint /usr/local/bin/llaminar2 "${tags[0]}" \
        --help >/dev/null
fi

echo "[build-runtime-image] built ${tags[*]} (variant=${image_variant}, cpu_isa=${cpu_isa:-Dockerfile default}, cuda=${enable_cuda}, rocm=${enable_rocm})"
