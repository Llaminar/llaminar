#!/bin/bash
# nv-controls.sh <tools-dir>: N-1/N-2 (reviewer 91) and Q-2/Q-2d/Q-3/G-1/G-2 (reviewer 109) controls on a small git-committed CMake project with a REAL nvcc
# TU (canonical toolchain image, its nvcc, ccache launcher, sm_121 as production). Tool controls only.
# Libraries (each a production-like shared "core" + target):
#   gnufirst  g.cpp k.cu prog.cpp      nvcc weak copy of an audited helper DISCARDED (GNU first)  -> PASS
#   nvfirst   k.cu g.cpp prog.cpp      the nvcc copy SELECTED                                     -> FAIL (map)
#   reached   g.cpp k.cu reach.cpp     a program reaches a function defined only in the nvcc TU   -> FAIL (closure)
#   declared  g.cpp k2.cu prog.cpp     nvcc-exclusive definition named in an audited header        -> FAIL (B1)
set -euo pipefail
TOOLS=${1:?tools directory}
W=$HOME/pr73-production; C=$W/nv-controls; IMAGE=pr73-production/toolchain-neon:d3d00437f6cf91575d3d034d2eeb2836d07802623d8817c902a2d501868c6a16
docker run --rm -v $W:/w $IMAGE rm -rf /w/nv-controls; mkdir -p $C/nv $C/ccache $C/out
cat > $C/nv/CPUNativeQuantisedMini.h <<'H'
inline __attribute__((noinline)) int helper_copy(int x) { return x * 11 + 13; }
inline const int kTable[4] = {1, 2, 3, 4};
inline thread_local int tls_counter = 0;
#if defined(__CUDACC__) && defined(NV_HDR_KERNEL)
__global__ void hdr_kernel(int *p) { p[threadIdx.x] = 3; }
#endif
inline __attribute__((noinline)) int nv_header_only(int x) { return x * 7 + 1; }
int nv_declared(int x);
H
cat > $C/nv/g.cpp <<'G'
#include "CPUNativeQuantisedMini.h"
int uses_helper(int x) { return helper_copy(x) + 1; }
int thrower_g(int x) { if (x < 0) throw x; return x; }
G
cat > $C/nv/k.cu <<'K'
#include "CPUNativeQuantisedMini.h"
int nv_only(int x) { return helper_copy(x) + 2 + kTable[x & 3] + ++tls_counter; }
int thrower(int x) { if (x < 0) throw x; return x; }
K
cat > $C/nv/k2.cu <<'K'
#include "CPUNativeQuantisedMini.h"
int nv_declared(int x) { return helper_copy(x) + 3 + nv_header_only(x); }
K
echo "int plain_util(int x) { return x + 5; }" > $C/nv/plain.cpp
cat > $C/nv/kern.cu <<'K'
#include "CPUNativeQuantisedMini.h"
__global__ void scale_kernel(int *p) { p[threadIdx.x] *= 2; }
void launch_scale(int *p) { scale_kernel<<<1, 32>>>(p); }
K
cat > $C/nv/kern2.cu <<'K'
#include "CPUNativeQuantisedMini.h"
void launch_hdr(int *p) { hdr_kernel<<<1, 32>>>(p); }
K
cat > $C/nv/reachstub.cpp <<'R'
void __device_stub__Z12scale_kernelPi(int *);
namespace llaminar2::cpu::native_quantised::multi_scale {
template <int N> int rowsProgram(int x) { int v = x; __device_stub__Z12scale_kernelPi(&v); return v * N; }
template int rowsProgram<1>(int);
}
R
cat > $C/nv/prog.cpp <<'P'
#include "CPUNativeQuantisedMini.h"
namespace llaminar2::cpu::native_quantised::multi_scale {
template <int N> int rowsProgram(int x) { return helper_copy(x) * N + kTable[x & 3] + ++tls_counter; }
template int rowsProgram<1>(int);
}
P
cat > $C/nv/reach.cpp <<'R'
int nv_only(int);
namespace llaminar2::cpu::native_quantised::multi_scale {
template <int N> int rowsProgram(int x) { return nv_only(x) * N; }
template int rowsProgram<1>(int);
}
R
cat > $C/nv/CMakeLists.txt <<'L'
cmake_minimum_required(VERSION 3.20)
project(nv CXX CUDA)
add_library(gnufirst SHARED prog.cpp g.cpp k.cu plain.cpp)
add_library(nvfirst SHARED k.cu g.cpp prog.cpp)
add_library(reached SHARED g.cpp k.cu reach.cpp)
add_library(declared SHARED g.cpp k2.cu prog.cpp)
add_library(kernel SHARED g.cpp kern.cu prog.cpp)
add_library(hdrkernel SHARED g.cpp kern2.cu prog.cpp)
set_source_files_properties(kern2.cu PROPERTIES COMPILE_DEFINITIONS NV_HDR_KERNEL)
add_library(reachstub SHARED g.cpp kern.cu reachstub.cpp)
# Production cuda_backend compiles with relocatable device code (a global __fatbinwrap_ object).
set_target_properties(kernel hdrkernel reachstub PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
L
printf "%s\n" CPUNativeQuantisedMini.h prog.cpp reach.cpp reachstub.cpp > $C/nv/audited.txt
( cd $C/nv && git init -q && git add CPUNativeQuantisedMini.h kern.cu kern2.cu reachstub.cpp plain.cpp g.cpp k.cu k2.cu prog.cpp reach.cpp CMakeLists.txt \
  && git -c user.name=c -c user.email=c@c commit -qm nv && git ls-tree -r HEAD > manifest.txt )
docker run --rm -v $C/nv:/src/nv -v $C/ccache:/root/.ccache $IMAGE bash -c '
cmake -S /src/nv -B /src/nv/build -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=/usr/bin/ccache -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCMAKE_CXX_FLAGS="-O2 -g" -DCMAKE_CUDA_FLAGS="-O2 -g" >/dev/null && cmake --build /src/nv/build >/dev/null && echo NV_BUILT'
docker run --rm -v $C/nv:/src/nv -v $C/nv/build:/src/nv/build:ro -v $TOOLS:/tools:ro -v $C/out:/out $IMAGE python3 /tools/nv_controls.py
