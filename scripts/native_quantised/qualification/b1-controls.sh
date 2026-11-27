#!/bin/bash
# b1-controls.sh (v5): a small git-committed CMake/Ninja/ccache project in the canonical toolchain image, built three
# ways (Integration-like -O2 -g, Release-like -O2, LTO -O2 -flto), then b1_controls.py with every build read-only.
set -euo pipefail
TOOLS=${1:?tools directory}
W=$HOME/pr73-production; C=$W/b1-controls; IMAGE=pr73-production/toolchain-neon:d3d00437f6cf91575d3d034d2eeb2836d07802623d8817c902a2d501868c6a16
docker run --rm -v $W:/w $IMAGE rm -rf /w/b1-controls; mkdir -p $C/mini $C/evidence $C/ccache
cat > $C/mini/h.h <<'H'
inline int scale(int x) { return 3 * x + 1; }
H
cat > $C/mini/CPUNativeQuantisedMini.h <<'H'
inline int fully_inlined(int x) { return x * 7 + 5; }
inline __attribute__((noinline)) int helper_copy(int x) { return x * 11 + 13; }
H
cat > $C/mini/a.cpp <<'A'
#include "h.h"
#include "CPUNativeQuantisedMini.h"
namespace llaminar2::cpu::native_quantised::multi_scale {
__attribute__((noinline)) int scaled(int x) { return scale(x) + 1; }
template <int N> int rowsProgram(int x) { return scaled(x) * N; }
template int rowsProgram<1>(int);
}
A
cat > $C/mini/b.cpp <<'B'
int plain(int x) { return x + 2; }
B
cat > $C/mini/c.cpp <<'C'
#include "CPUNativeQuantisedMini.h"
int uses_inlined(const int *x, int n) { int s = 0; for (int i = 0; i < n; ++i) s += fully_inlined(x[i]) * x[i]; return s; }
C
cat > $C/mini/d.cpp <<'D'
#include "CPUNativeQuantisedMini.h"
int unrelated(int x) { return x - 4; }
D
cat > $C/mini/e.cpp <<'E'
#include "CPUNativeQuantisedMini.h"
int uses_helper(int x) { return helper_copy(x) + 1; }
E
mkdir -p $C/mini/external/dep/build
cat > $C/mini/external/dep/dep.h <<'X'
inline int dep_value(int x) { return x ^ 5; }
X
( cd $C/mini/external/dep && git init -q && git add dep.h && git -c user.name=c -c user.email=c@c commit -qm dep \
  && git remote add origin https://example.invalid/dep.git )
echo "inline int generated_value(int x) { return x + 9; }" > $C/mini/external/dep/build/gen.h
cat > $C/mini/f.cpp <<'F'
#include "external/dep/dep.h"
#include "external/dep/build/gen.h"
int uses_external(int x) { return dep_value(x) + generated_value(x); }
F
cat > $C/mini/ptr.h <<'Q'
#include <algorithm>
inline int unaudited_math(int x) { return std::max(x * x + 3, x); }
Q
cat > $C/mini/p.cpp <<'P'
#include "CPUNativeQuantisedMini.h"
#include "ptr.h"
int plain(int);
namespace llaminar2::cpu::native_quantised::multi_scale {
__attribute__((noinline)) int via_pointer(int x) { return x * 5 + 1; }
__attribute__((noinline)) int via_table_a(int x) { return x - 7; }
__attribute__((noinline)) int via_table_b(int x) { return x + 7; }
int (*table[2])(int) = {via_table_a, via_table_b};
template <int N> int rowsProgram(int x) {
  int (*volatile f)(int) = via_pointer;
  int s = 0;
#pragma omp parallel for reduction(+:s)
  for (int i = 0; i < x; ++i) s += fully_inlined(i);
  return f(s) * N + table[x & 1](x) + unaudited_math(x) + plain(x);
}
template int rowsProgram<2>(int);
}
P
cat > $C/mini/main.cpp <<'M'
int plain(int);
int main() { return plain(0) - 2; }
M
cat > $C/mini/CMakeLists.txt <<'L'
cmake_minimum_required(VERSION 3.20)
project(mini CXX)
add_library(prod SHARED a.cpp b.cpp c.cpp d.cpp e.cpp f.cpp p.cpp)
set_source_files_properties(p.cpp PROPERTIES COMPILE_OPTIONS -fopenmp)
target_link_options(prod PRIVATE -fopenmp)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE prod)
L
echo CPUNativeQuantisedMini.h > $C/mini/audited.txt
printf "%s\n" CPUNativeQuantisedMini.h a.cpp b.cpp h.h p.cpp ptr.h > $C/mini/audited-all.txt
mkdir -p $C/usr-mount && echo "int mounted_system_header;" > $C/usr-mount/mounted.h
( cd $C/mini && git init -q && git add h.h ptr.h CPUNativeQuantisedMini.h a.cpp b.cpp c.cpp d.cpp e.cpp f.cpp p.cpp main.cpp CMakeLists.txt \
  && git -c user.name=c -c user.email=c@c commit -qm mini && git ls-tree -r HEAD > manifest.txt )
docker run --rm -v $C/mini:/src/mini -v $C/ccache:/root/.ccache $IMAGE bash -c '
for v in "build:-O2 -g" "build-release:-O2" "build-lto:-O2 -flto"; do d=${v%%:*}; f=${v#*:}
  cmake -S /src/mini -B /src/mini/$d -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache -DCMAKE_CXX_FLAGS="$f" >/dev/null && cmake --build /src/mini/$d >/dev/null || exit 1; done; echo MINI_BUILT'
docker run --rm -v $C/mini:/src/mini -v $C/mini/build:/src/mini/build:ro -v $C/mini/build-release:/src/mini/build-release:ro \
  -v $C/mini/build-lto:/src/mini/build-lto:ro -v $C/usr-mount:/usr/local/ctl:ro -v $W/builds-r2/develop-63b053401-integration-neon:/prod:ro -v $TOOLS:/tools:ro -v $C/evidence:/evidence \
  $IMAGE python3 /tools/b1_controls.py
