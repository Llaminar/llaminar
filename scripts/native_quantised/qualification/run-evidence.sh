#!/bin/bash
# run-evidence.sh (v2) <tree> <kind> <isa-lower>: B1 replay then post-processing (link capture, selected copies,
# process resolution, facts) of one COMPLETED canonical build; the build directory is mounted read-only; refuses
# while its build or production-target container runs.
set -uo pipefail
tree=$1; kind=$2; lisa=$3
W=$HOME/pr73-production; IMAGE=pr73-production/toolchain-neon:d3d00437f6cf91575d3d034d2eeb2836d07802623d8817c902a2d501868c6a16
b=$W/builds-r2/$tree-$kind-$lisa; a=$W/artifacts-r2/$tree-$kind-$lisa; log=$W/logs/r2-$tree-$kind-$lisa.log
if docker ps --format "{{.Names}}" | grep -qi "^pr73-.*$tree-$kind-"; then echo "REFUSED: $tree $kind container still running"; exit 3; fi
test -d $b && test -d $b.external || { echo "MISSING build $b"; exit 4; }
test ! -e $a || { echo "REFUSED: $a exists"; exit 5; }
mkdir -p $a
image_id=$(docker image inspect $IMAGE --format "{{.Id}}")
docker run --rm --network host -v $W/src/$tree:/snapshot:ro -v $b:/src/build_v2_$kind:ro -v $b.external:/src/external:ro -v $W/tools:/tools:ro \
  -v $W/manifests:/manifests:ro -v $a:/artifacts -v $log:/build.log:ro -v $W/logs:/logs:ro $IMAGE bash -c "
  cd /src && for p in src tests CMakeLists.txt .githooks jinja cmake python scripts .agents .github .devcontainer AGENTS.md README.md; do cp -a /snapshot/\$p /src/; done
  mkdir -p /src/benchmarks && cp -a /snapshot/benchmarks/production /src/benchmarks/  # /src/external is the build's persisted external dir
  python3 /tools/b1_replay.py --build /src/build_v2_$kind --evidence /artifacts/b1 --target llaminar2 --target libllaminar2_core.so \
    --manifest /manifests/$tree.ls-tree.txt --identity /manifests/$tree.identity.txt --audited-headers /tools/audited-headers.txt \
    --image-id $image_id > /artifacts/b1.out 2>&1; echo \"b1 exit \$?\"
  python3 /tools/postprocess.py --build /src/build_v2_$kind --artifacts /artifacts/link --image-id $image_id \
    --build-log /build.log --b1 /artifacts/b1/b1-summary.json > /artifacts/postprocess.out 2>&1; echo \"postprocess exit \$?\"
  " > $a/evidence.out 2>&1
cat $a/evidence.out; tail -1 $a/b1.out; tail -1 $a/postprocess.out
( cd $a && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
