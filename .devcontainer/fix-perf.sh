#!/usr/bin/env bash
# Build perf from source matching the running kernel version
# Adapted from: https://askubuntu.com/questions/1539634

set -euo pipefail

# --- Helpers ---
log() { echo -e "\033[1;32m==>\033[0m $*"; }
err() { echo -e "\033[1;31m[ERR]\033[0m $*" >&2; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    err "This script must run as root. Re-executing with sudo..."
    exec sudo bash "$0" "$@"
fi

export DEBIAN_FRONTEND=noninteractive

FULLREL="$(uname -r)"                                   # e.g. 6.14.0-36-generic
MAJMIN="$(uname -r | awk -F'[.-]' '{print $1 "." $2}')" # e.g. 6.14
WRAP_DIR="/usr/lib/linux-tools/${FULLREL}"

log "Detected running kernel: ${FULLREL}"
log "Using upstream kernel source line: ${MAJMIN}"

# --- Check if perf already works ---
if [[ -x "${WRAP_DIR}/perf" ]] && ! [[ -L "${WRAP_DIR}/perf" ]] && "${WRAP_DIR}/perf" --version >/dev/null 2>&1; then
    log "perf already working: $(${WRAP_DIR}/perf --version)"
    log "Skipping build."
    exit 0
fi

# --- Install build dependencies ---
log "Installing build dependencies..."
apt-get update -y
apt-get install -y \
  build-essential make gcc g++ pkg-config flex bison \
  curl ca-certificates xz-utils tar \
  libelf-dev libdw-dev libdwarf-dev libunwind-dev \
  libnuma-dev libssl-dev libcap-dev \
  zlib1g-dev liblzma-dev libzstd-dev \
  libaio-dev libtraceevent-dev libpfm4-dev \
  libslang2-dev systemtap-sdt-dev \
  python3 python3-dev python3-setuptools \
  binutils-dev libiberty-dev libbabeltrace-dev \
  libbfd-dev clang llvm || true
apt-get install -y debuginfod || true

# --- Fetch kernel sources ---
WORKDIR="$(mktemp -d -t perfbuild-XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT
pushd "$WORKDIR" >/dev/null

BASE_URL="https://mirrors.edge.kernel.org/pub/linux/kernel/v${MAJMIN%%.*}.x"
ARCHIVE=""
for f in "linux-${MAJMIN}.tar.xz" "linux-${MAJMIN}.tar.gz"; do
  log "Downloading: ${BASE_URL}/${f}"
  if curl -fsSLO "${BASE_URL}/${f}"; then
    ARCHIVE="$f"
    break
  fi
done
[[ -n "$ARCHIVE" ]] || {
  err "Failed to download linux-${MAJMIN} from ${BASE_URL}"
  exit 2
}

log "Extracting ${ARCHIVE} ..."
tar -xf "${ARCHIVE}"
SRC_DIR="${WORKDIR}/linux-${MAJMIN}"
[[ -d "${SRC_DIR}/tools/perf" ]] || {
  err "tools/perf not found in sources"
  exit 3
}
[[ -d "${SRC_DIR}/tools/bpf/bpftool" ]] || {
  err "tools/bpf/bpftool not found in sources"
  exit 3
}

# --- Build perf ---
pushd "${SRC_DIR}/tools/perf" >/dev/null
log "Building perf (WERROR=0) ..."
make -j"$(nproc)" WERROR=0
log "Built perf version: $(./perf --version)"
popd >/dev/null

# --- Build bpftool ---
pushd "${SRC_DIR}/tools/bpf/bpftool" >/dev/null
log "Building bpftool ..."
make -j"$(nproc)"
log "Built bpftool version: $(./bpftool version || true)"
popd >/dev/null

# --- Install into wrapper path(s) ---
log "Installing perf to ${WRAP_DIR}/perf"
install -d "${WRAP_DIR}"
install -m 0755 "${SRC_DIR}/tools/perf/perf" "${WRAP_DIR}/perf"

log "Installing bpftool to ${WRAP_DIR}/bpftool"
install -m 0755 "${SRC_DIR}/tools/bpf/bpftool/bpftool" "${WRAP_DIR}/bpftool"

# --- Verify via Ubuntu wrappers ---
log "Verifying /usr/bin/perf (Ubuntu wrapper) ..."
if /usr/bin/perf --version >/dev/null 2>&1; then
  /usr/bin/perf --version
  log "✓ Success: wrapper found perf at ${WRAP_DIR}/perf"
else
  err "Wrapper did not run perf. Check that ${WRAP_DIR}/perf exists and is executable."
  ls -l "${WRAP_DIR}/perf" || true
  exit 4
fi

log "Verifying /usr/sbin/bpftool (Ubuntu wrapper) ..."
if /usr/sbin/bpftool version >/dev/null 2>&1; then
  /usr/sbin/bpftool version
  log "✓ Success: wrapper found bpftool at ${WRAP_DIR}/bpftool"
else
  err "Wrapper did not run bpftool. Check that ${WRAP_DIR}/bpftool exists and is executable."
  ls -l "${WRAP_DIR}/bpftool" || true
  exit 5
fi

log "✓ Done. perf and bpftool built from kernel ${MAJMIN} sources."
popd >/dev/null
