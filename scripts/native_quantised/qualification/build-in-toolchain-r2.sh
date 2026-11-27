#!/bin/bash
# build-in-toolchain.sh <tree-dir-name> <integration|release> <ISA> <toolchain-image>
# Round 2: /src/external (where configure clones and builds oneDNN, external to the build dir) is persisted per build at <out>.external, so the closure and link inputs outside the build dir survive for evidence; no flag or step changes.
# Runs the Dockerfile builder stage steps for one source-pinned tree inside the canonical toolchain image, exactly:
# the builder inventory check, the builder COPY set into /src, then the Integration RUN (arm64-pr inventory,
# target v2_arm64_pr_gate) or the Release RUN (target llaminar2, LLAMINAR_BUILD_TESTS=OFF) with the same cmake
# arguments and arm64 ccache size; the ONLY difference is that strip and intermediate deletion are NOT run (the
# unstripped objects are the evidence). Output: ~/pr73-production/builds/<tree>-<kind>-<isa>/ (bind-mounted at the
# canonical /src/build_v2_<kind>). Sources are copied from the read-only snapshot.
set -euo pipefail
tree=$1; kind=$2; isa=$3; image=$4
W=$HOME/pr73-production; out=$W/builds-r2/$tree-$kind-$(echo $isa | tr "[:upper:]" "[:lower:]")
test -d $W/src/$tree; test ! -e $out; mkdir -p $out $out.external
{ echo "tree=$tree kind=$kind isa=$isa image=$image image_id=$(docker image inspect $image --format {{.Id}})"; date -u +%FT%TZ; } > $out.receipt
exec docker run --rm --name pr73-$tree-$kind-$isa --network host \
  -v $W/src/$tree:/snapshot:ro -v $out:/src/build_v2_$kind -v $out.external:/src/external -v $W/ccache:/root/.ccache \
  -e KIND=$kind -e LLAMINAR_CPU_ISA=$isa -e TARGETARCH=arm64 \
  -e LLAMINAR_ENABLE_CUDA=ON -e LLAMINAR_ENABLE_ROCM=OFF -e LLAMINAR_CUDA_ARCHS=121 -e LLAMINAR_BUILD_TYPE=Release \
  -e LLAMINAR_SKIP_INTEGRATION=0 -e LLAMINAR_BUILD_MODEL_PARITY_MATRICES=OFF -e LLAMINAR_TEST_RUNNER_INVENTORY=arm64-pr \
  -e ROCM_RUNTIME_GPU_TARGETS= \
  $image bash -c "
set -euo pipefail
cd /src
case \"\${LLAMINAR_BUILD_MODEL_PARITY_MATRICES}:\${LLAMINAR_TEST_RUNNER_INVENTORY}\" in
  OFF:arm64-pr) test \"\${TARGETARCH}:\${LLAMINAR_ENABLE_CUDA}:\${LLAMINAR_ENABLE_ROCM}:\${LLAMINAR_SKIP_INTEGRATION}\" = arm64:ON:OFF:0 ;;
  *) exit 1 ;;
esac
for p in src tests CMakeLists.txt .githooks jinja cmake python scripts .agents .github .devcontainer AGENTS.md README.md; do cp -a /snapshot/\$p /src/; done
mkdir -p /src/external /src/benchmarks && cp -a /snapshot/external/vendor /src/external/ && cp -a /snapshot/benchmarks/production /src/benchmarks/
export CCACHE_MAXSIZE=5G; arm64_cmake_args=-DUSE_ONNX_HEURISTIC=OFF
rocm_gpu_targets=\"\$(bash scripts/docker/rocm-image-targets.sh \"\${ROCM_RUNTIME_GPU_TARGETS}\")\"
started=\$(date +%s)
if [ \"\$KIND\" = integration ]; then
  cmake -B build_v2_integration -S src/v2 -G Ninja -DCMAKE_MAKE_PROGRAM:FILEPATH=\"\$(command -v ninja)\" \
    -DCMAKE_BUILD_TYPE=Integration -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DHAVE_CUDA=ON -DHAVE_ROCM=OFF \
    -DLLAMINAR_CPU_ISA=\"\${LLAMINAR_CPU_ISA}\" -DLLAMINAR_SHARED_CORE=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DCMAKE_HIP_ARCHITECTURES=\"\${rocm_gpu_targets}\" \${arm64_cmake_args}
  cmake --build build_v2_integration --parallel --target v2_arm64_pr_gate
else
  cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_MAKE_PROGRAM:FILEPATH=\"\$(command -v ninja)\" \
    -DCMAKE_BUILD_TYPE=\${LLAMINAR_BUILD_TYPE} -DHAVE_CUDA=ON -DHAVE_ROCM=OFF \
    -DLLAMINAR_CPU_ISA=\"\${LLAMINAR_CPU_ISA}\" -DLLAMINAR_SHARED_CORE=ON -DLLAMINAR_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_HIP_ARCHITECTURES=\"\${rocm_gpu_targets}\" \${arm64_cmake_args}
  cmake --build build_v2_release --parallel --target llaminar2
  ninja -C build_v2_release -t compdb > build_v2_release/compile_commands.ninja-compdb.json
fi
echo \"BUILD_DONE kind=\$KIND elapsed_seconds=\$(( \$(date +%s) - started ))\"
"
