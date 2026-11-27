#!/bin/bash
# build-production-targets.sh <tree> <kind> <isa-lower> (B1-f): build `llaminar2 libllaminar2_core.so` explicitly in
# the SAME configured canonical tree after its gate build (which may be red): no reconfigure - the build.ninja and
# CMakeCache.txt sha256 must be unchanged across the step. Recorded as a separate step; the gate status stays as
# its own log reports it (a red gate is never qualified by this step).
set -uo pipefail
tree=$1; kind=$2; lisa=$3
W=$HOME/pr73-production; IMAGE=pr73-production/toolchain-neon:d3d00437f6cf91575d3d034d2eeb2836d07802623d8817c902a2d501868c6a16
b=$W/builds-r2/$tree-$kind-$lisa; r=$W/builds-r2/$tree-$kind-$lisa.production-targets.receipt
test -d $b && test -d $b.external || { echo "MISSING build $b"; exit 4; }
if docker ps --format "{{.Names}}" | grep -qi "^pr73-.*$tree-$kind-"; then echo "REFUSED: container running"; exit 3; fi
before="$(sha256sum $b/build.ninja $b/CMakeCache.txt)"
docker run --rm --name pr73-prodtargets-$tree-$kind-$lisa --network host -v $W/src/$tree:/snapshot:ro \
  -v $b:/src/build_v2_$kind -v $b.external:/src/external -v $W/ccache:/root/.ccache $IMAGE bash -c "
  set -euo pipefail; cd /src
  for p in src tests CMakeLists.txt .githooks jinja cmake python scripts .agents .github .devcontainer AGENTS.md README.md; do cp -a /snapshot/\$p /src/; done
  mkdir -p /src/benchmarks && cp -a /snapshot/benchmarks/production /src/benchmarks/  # /src/external is the build's persisted external dir
  export CCACHE_MAXSIZE=5G
  ninja -C build_v2_$kind llaminar2 libllaminar2_core.so" > $W/logs/r2-$tree-$kind-$lisa.production-targets.log 2>&1
status=$?
after="$(sha256sum $b/build.ninja $b/CMakeCache.txt)"
{ echo "step=explicit production targets (llaminar2 libllaminar2_core.so) in the same configured tree"; echo "exit=$status"
  echo "gate_log_build_done=$(grep -c BUILD_DONE $W/logs/r2-$tree-$kind-$lisa.log)"
  echo "before:"; echo "$before"; echo "after:"; echo "$after"
  [ "$before" = "$after" ] && echo "configuration=unchanged" || echo "configuration=CHANGED (FAIL)"; date -u +%FT%TZ; } > $r
cat $r
