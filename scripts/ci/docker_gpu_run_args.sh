#!/usr/bin/env bash
# Emit Docker run arguments for NVIDIA GPU access.
#
# Prefer the Docker/NVIDIA runtime when available. On ARC runners that use a
# privileged Docker-in-Docker sidecar, the daemon can see /dev/nvidia* but may
# not have nvidia-container-toolkit installed; in that case, fall back to
# explicit device nodes plus a staged host driver-library directory.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/ci/docker_gpu_run_args.sh [options]

Options:
  --probe-image IMAGE     Image used for Docker run probes. Default: ubuntu:24.04
  --lib-dir DIR           Host/DIND-visible NVIDIA driver lib directory.
                          Default: LLAMINAR_NVIDIA_DRIVER_LIB_DIR,
                          /opt/llaminar-nvidia-libs, then /usr/lib/x86_64-linux-gnu.
  --required              Fail if neither --gpus nor manual passthrough works.
  -h, --help              Show this help.

The output is one Docker argument per line so callers can safely use:

  mapfile -t gpu_args < <(scripts/ci/docker_gpu_run_args.sh --probe-image "$image" --required)
  docker run "${gpu_args[@]}" ...
EOF
}

die() {
    echo "docker_gpu_run_args: error: $*" >&2
    exit 1
}

probe_image="ubuntu:24.04"
lib_dir_override=""
required=0

while (($#)); do
    case "$1" in
        --probe-image)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            probe_image="$2"
            shift 2
            ;;
        --lib-dir)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            lib_dir_override="$2"
            shift 2
            ;;
        --required)
            required=1
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

emit() {
    printf '%s\n' "$@"
}

docker_supports_nvidia_runtime() {
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' && return 0
    docker run --rm --gpus all --entrypoint /bin/true "$probe_image" >/dev/null 2>&1
}

lib_dir_has_driver_libs() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    [[ -e "${dir}/libcuda.so.1" ]] || return 1
    [[ -e "${dir}/libnvidia-ml.so.1" ]] || return 1
}

resolve_driver_lib_dir() {
    local candidates=()
    if [[ -n "$lib_dir_override" ]]; then
        candidates+=("$lib_dir_override")
    fi
    if [[ -n "${LLAMINAR_NVIDIA_DRIVER_LIB_DIR:-}" ]]; then
        candidates+=("${LLAMINAR_NVIDIA_DRIVER_LIB_DIR}")
    fi
    candidates+=("/opt/llaminar-nvidia-libs" "/usr/lib/x86_64-linux-gnu")

    local dir
    for dir in "${candidates[@]}"; do
        if lib_dir_has_driver_libs "$dir"; then
            printf '%s' "$dir"
            return 0
        fi
    done
    return 1
}

daemon_nvidia_device_nodes() {
    docker run --rm \
        --entrypoint /bin/sh \
        -v /dev:/host-dev:ro \
        "$probe_image" \
        -lc '
set -eu
for path in \
    /host-dev/nvidiactl \
    /host-dev/nvidia-uvm \
    /host-dev/nvidia-uvm-tools \
    /host-dev/nvidia-modeset; do
    [ -e "$path" ] && printf "%s\n" "${path#/host-dev}"
done
for path in /host-dev/nvidia[0-9]* /host-dev/nvidia-caps/*; do
    [ -e "$path" ] && printf "%s\n" "${path#/host-dev}"
done
' 2>/dev/null | sort -u
}

if docker_supports_nvidia_runtime; then
    emit --gpus all
    exit 0
fi

driver_lib_dir="$(resolve_driver_lib_dir || true)"
mapfile -t device_nodes < <(daemon_nvidia_device_nodes || true)

if [[ -z "$driver_lib_dir" || "${#device_nodes[@]}" -eq 0 ]]; then
    if [[ "$required" == "1" ]]; then
        [[ -n "$driver_lib_dir" ]] || die "NVIDIA driver libs not found; stage them under /opt/llaminar-nvidia-libs or pass --lib-dir"
        ((${#device_nodes[@]} > 0)) || die "NVIDIA device nodes are not visible to the Docker daemon"
    fi
    exit 0
fi

emit -v "${driver_lib_dir}:/usr/local/nvidia/lib64:ro"
emit -e "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/lib:/usr/local/cuda/lib64:/opt/rocm/lib"
for node in "${device_nodes[@]}"; do
    emit --device="$node"
done
