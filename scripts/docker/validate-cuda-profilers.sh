#!/usr/bin/env bash
# validate-cuda-profilers.sh
#
# Prove that a CUDA development image can produce an Nsight Systems trace of a
# real Llaminar CUDA workload. Run inside the image with the GPU attached.
#
#   validate-cuda-profilers.sh                      # tool/package identity only
#   validate-cuda-profilers.sh -- <command> [args]  # also trace <command>
#
# The trace step records CUDA API and NVTX activity, then requires that the
# report contains at least one GPU kernel and at least one graph launch: an
# empty or capture-less trace cannot answer per-bucket replay questions.
# Profiles are diagnostic evidence; never compare their timings with an
# unprofiled benchmark. OUTPUT_DIR (default /tmp/llaminar-nsys-probe) receives
# the report and its CSV summaries.
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/llaminar-nsys-probe}"

fail() {
    echo "validate-cuda-profilers: $*" >&2
    exit 1
}

for package in cuda-nsight-systems-13-0 cuda-nsight-compute-13-0 cuda-cupti-13-0; do
    dpkg-query -W -f='${Package} ${Version}\n' "${package}" 2>/dev/null ||
        fail "package ${package} is not installed (MODE=profilers or INSTALL_CUDA_PROFILERS=1)"
done
command -v nsys >/dev/null || fail "nsys is not on PATH"
command -v ncu >/dev/null || fail "ncu is not on PATH"
nsys --version
ncu --version | tail -n 1
# Sampling/permission diagnostics are informative inside containers; the trace
# below needs only CUDA API and NVTX tracing.
nsys status --environment || true

if [[ $# -eq 0 ]]; then
    echo "validate-cuda-profilers: tool identity complete (no workload traced)"
    exit 0
fi
[[ "$1" == "--" ]] || fail "separate the traced command with --"
shift
[[ $# -gt 0 ]] || fail "no command follows --"

mkdir -p "${OUTPUT_DIR}"
report="${OUTPUT_DIR}/probe"
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --force-overwrite=true --output="${report}" -- "$@"

nsys stats --force-export=true --format=csv --output="${OUTPUT_DIR}/summary" \
    --report=cuda_gpu_kern_sum --report=cuda_api_sum "${report}.nsys-rep"

kernels="${OUTPUT_DIR}/summary_cuda_gpu_kern_sum.csv"
api="${OUTPUT_DIR}/summary_cuda_api_sum.csv"
[[ -s "${kernels}" ]] || fail "no kernel summary was exported"
[[ "$(wc -l < "${kernels}")" -gt 1 ]] || fail "the trace contains no GPU kernels"
grep -q 'cudaGraphLaunch' "${api}" || fail "the trace contains no CUDA graph launch"
echo "validate-cuda-profilers: trace complete: ${report}.nsys-rep"
