#!/bin/bash
# As build-in-toolchain.sh (unchanged for NEON), for NEON_DOTPROD / NEON_I8MM Integration: NON-CANONICAL (no CI job
# builds these envelopes). The Dockerfile admits the arm64-pr inventory only for NEON, so these use its admitted
# model-free inventory (v2_unit_gate v2_production_test_preflight_gate) plus, explicitly (B1-g), the production
# targets llaminar2 libllaminar2_core.so; every other step and argument is identical; same toolchain image id, the
# ISA only through -DLLAMINAR_CPU_ISA. Release is the Gate7 production subject for these envelopes; this
# Integration build is supplementary evidence.
set -euo pipefail
tree=$1; isa=$2; image=$3
W=$HOME/pr73-production; out=$W/builds/$tree-integration-$(echo $isa | tr "[:upper:]" "[:lower:]")
test -d $W/src/$tree; test ! -e $out; mkdir -p $out
{ echo "tree=$tree kind=integration isa=$isa inventory=model-free+production-targets label=non-canonical(no-CI) image=$image image_id=$(docker image inspect $image --format {{.Id}})"; date -u +%FT%TZ; } > $out.receipt
exec docker run --rm --name pr73-$tree-integration-$isa --network host \
  -v $W/src/$tree:/snapshot:ro -v $out:/src/build_v2_integration -v $W/ccache:/root/.ccache \
  -e LLAMINAR_CPU_ISA=$isa -e ROCM_RUNTIME_GPU_TARGETS= $image bash -c "
set -euo pipefail
cd /src
for p in src tests CMakeLists.txt .githooks jinja cmake python scripts .agents .github .devcontainer AGENTS.md README.md; do cp -a /snapshot/\$p /src/; done
mkdir -p /src/external /src/benchmarks && cp -a /snapshot/external/vendor /src/external/ && cp -a /snapshot/benchmarks/production /src/benchmarks/
export CCACHE_MAXSIZE=5G; arm64_cmake_args=-DUSE_ONNX_HEURISTIC=OFF
rocm_gpu_targets=\"\$(bash scripts/docker/rocm-image-targets.sh \"\${ROCM_RUNTIME_GPU_TARGETS}\")\"
started=\$(date +%s)
cmake -B build_v2_integration -S src/v2 -G Ninja -DCMAKE_MAKE_PROGRAM:FILEPATH=\"\$(command -v ninja)\" \
  -DCMAKE_BUILD_TYPE=Integration -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DHAVE_CUDA=ON -DHAVE_ROCM=OFF \
  -DLLAMINAR_CPU_ISA=\"\${LLAMINAR_CPU_ISA}\" -DLLAMINAR_SHARED_CORE=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCMAKE_HIP_ARCHITECTURES=\"\${rocm_gpu_targets}\" \${arm64_cmake_args}
cmake --build build_v2_integration --parallel --target v2_unit_gate v2_production_test_preflight_gate llaminar2 libllaminar2_core.so
echo \"BUILD_DONE kind=integration elapsed_seconds=\$(( \$(date +%s) - started ))\"
"
