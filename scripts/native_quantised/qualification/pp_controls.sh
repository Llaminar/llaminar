#!/bin/bash
# pp_controls.sh (v3): controls of postprocess.py on mini builds (tool controls only). Requires b1-controls.sh first
# (it creates ~/pr73-production/b1-controls/mini). Adds process-resolution mini builds: an executable `llaminar2`
# that imports a weak helper from `libllaminar2_core.so` and forks one descendant; build-proc-exit1 returns 1.
set -euo pipefail
TOOLS=${1:?tools directory}
W=$HOME/pr73-production; C=$W/b1-controls; IMAGE=pr73-production/toolchain-neon:d3d00437f6cf91575d3d034d2eeb2836d07802623d8817c902a2d501868c6a16
P=$C/proc; docker run --rm -v $W:/w $IMAGE rm -rf /w/b1-controls/proc /w/b1-controls/pp /w/b1-controls/ccache-proc; mkdir -p $P $C/pp $C/ccache-proc
cp $C/mini/h.h $C/mini/CPUNativeQuantisedMini.h $C/mini/a.cpp $C/mini/c.cpp $C/mini/e.cpp $P/
cat > $P/main2.cpp <<'M'
#include <unistd.h>
#include <sys/wait.h>
#include "CPUNativeQuantisedMini.h"
int uses_helper(int);
int main(int argc, char **argv) {
  pid_t child = fork();
  if (child == 0) { execl("/bin/true", "true", (char *)0); _exit(9); }
  int status = 0; waitpid(child, &status, 0);
  return uses_helper(argc) > 0 ? EXIT_CODE : 2;
}
M
cat > $P/CMakeLists.txt <<'L'
cmake_minimum_required(VERSION 3.20)
project(proc CXX)
add_library(llaminar2_core SHARED a.cpp c.cpp e.cpp)
add_executable(llaminar2 main2.cpp)
target_link_libraries(llaminar2 PRIVATE llaminar2_core)
target_compile_definitions(llaminar2 PRIVATE EXIT_CODE=${EXIT_CODE})
L
( cd $P && git init -q && git add . && git -c user.name=c -c user.email=c@c commit -qm proc && git ls-tree -r HEAD > manifest.txt && echo CPUNativeQuantisedMini.h > audited.txt )
echo BUILD_DONE > $C/pp/build.log
docker run --rm -v $C:/src/c -v $C/ccache-proc:/root/.ccache $IMAGE bash -c '
for v in "build-proc:0" "build-proc-exit1:1"; do d=${v%%:*}; e=${v#*:}
  cmake -S /src/c/proc -B /src/c/proc/$d -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache -DCMAKE_CXX_FLAGS="-O2 -g" -DEXIT_CODE=$e >/dev/null && cmake --build /src/c/proc/$d >/dev/null || exit 1; done; echo PROC_BUILT'
docker run --rm -v $C/mini:/src/mini -v $C/proc:/src/c/proc -v $C/mini/build:/src/mini/build:ro -v $C/mini/build-lto:/src/mini/build-lto:ro \
  -v $C/proc/build-proc:/src/c/proc/build-proc:ro -v $C/proc/build-proc-exit1:/src/c/proc/build-proc-exit1:ro \
  -v $TOOLS:/tools:ro -v $C/pp:/pp $IMAGE python3 /tools/pp_controls.py
