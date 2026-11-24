#!/usr/bin/env bash
# Isolated diagnostic dependency build, never an installer for the host or image.
# The versions match the initial image comparison. Hashes bind fetched release
# archives; this experiment is not a production dependency qualification receipt.
# CUDA 13 removed the unversioned driver-entrypoint typedef used by UCX 1.16.
# Bind that spelling to the v11070 five-argument ABI used at the UCX call site;
# its existing cuGetProcAddress version selection is unchanged. No transport behavior
# or allocator changes are made. Retain the initial failed compile as evidence.
set -euo pipefail
root="${1:?pass an absolute writable experiment root}"
[[ "$root" = /* ]] || { echo 'experiment root must be absolute' >&2; exit 2; }
mkdir -p "$root/deps-src" "$root/deps-build"
cd "$root/deps-src"
printf '%s\n' \
 'f73770d3b583c91aba5fb07557e655ead0786e057018bfe42f0ebe8716e9d28c  ucx-1.16.0.tar.gz' \
 '44da277b8cdc234e71c62473305a09d63f4dcca292ca40335aab7c4bf0e6a566  openmpi-4.1.6.tar.gz' | sha256sum -c -
[[ -d ucx-1.16.0 ]] || tar xf ucx-1.16.0.tar.gz
[[ -d openmpi-4.1.6 ]] || tar xf openmpi-4.1.6.tar.gz
# Keep the driver-hook test: CUDA 13's public context API adds a nullable
# creation-parameters argument. Earlier SDKs retain their original call.
patch_file="$(cd "$(dirname "$0")" && pwd)/ucx-1.16-cuda13-test.patch"
if patch -d ucx-1.16.0 -p1 --dry-run --reverse < "$patch_file" >/dev/null 2>&1; then
    : # Exact patched source already installed by this isolated builder.
else
    patch -d ucx-1.16.0 -p1 --forward < "$patch_file"
fi
mkdir -p "$root/deps-build/ucx" "$root/deps-build/mpi"
cd "$root/deps-build/ucx"
"$root/deps-src/ucx-1.16.0/configure" --prefix="$root/deps" \
 --with-cuda=/usr/local/cuda --without-rocm --without-gdrcopy \
 --disable-java --without-go --disable-static --enable-mt \
 CC=gcc-14 CXX=g++-14 \
 CFLAGS='-O3 -DPFN_cuMemGetHandleForAddressRange=PFN_cuMemGetHandleForAddressRange_v11070' CXXFLAGS='-O3'
make -j"$(nproc)"
make install
cd "$root/deps-build/mpi"
"$root/deps-src/openmpi-4.1.6/configure" --prefix="$root/deps" \
 --with-ucx="$root/deps" --with-cuda=/usr/local/cuda \
 --disable-mpi-fortran --disable-mpi-cxx --disable-static --with-devel-headers --without-verbs \
 --without-libfabric --without-psm --without-psm2 --without-slurm \
 CC=gcc-14 CXX=g++-14 CFLAGS='-O3' CXXFLAGS='-O3' \
 LDFLAGS="-Wl,-rpath,$root/deps/lib -Wl,-rpath,/usr/local/cuda/lib64"
make -j"$(nproc)"
make install
"$root/deps/bin/ompi_info" --parsable --all > "$root/deps-build/ompi-info.txt"
"$root/deps/bin/ucx_info" -v > "$root/deps-build/ucx-version.txt"
"$root/deps/bin/ucx_info" -d > "$root/deps-build/ucx-devices.txt"
echo 'EXPERIMENTAL_MPI_BUILD_PASS'
