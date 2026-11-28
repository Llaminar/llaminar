#!/bin/sh
# @file arm_native_integration_operation.sh
# @brief Canonical Integration operation shared by diagnostic export and gated builder.
    if [ "${LLAMINAR_SKIP_INTEGRATION}" = "1" ]; then \
        echo "==> [integration] skipped (LLAMINAR_SKIP_INTEGRATION=1)"; \
    else \
        integration_started_epoch="$(date +%s)"; \
        arm64_cmake_args=""; \
        if [ "${TARGETARCH}" = "arm64" ]; then \
            export CCACHE_MAXSIZE=5G; \
            arm64_cmake_args="-DUSE_ONNX_HEURISTIC=OFF"; \
        fi; \
        rocm_gpu_targets="$(bash scripts/docker/rocm-image-targets.sh "${ROCM_RUNTIME_GPU_TARGETS}")" || exit 1; \
        RCCL_CMAKE_ARGS=""; \
        if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ]; then \
            RCCL_CMAKE_ARGS="${RCCL_CMAKE_ARGS} -DRCCL_PROTOCOL_BUILD_DIR=/src/external/rccl/build -DRCCL_INCLUDE_DIR=/src/external/rccl/build/include -DRCCL_LIBRARY=/usr/local/lib/librccl.so.1 -DRCCL_FINAL_ALLOCATION_REPORT=/src/rccl-allocation-evidence/final-allocation.json -DRCCL_FINAL_ALLOCATION_RECEIPT=/src/rccl-allocation-evidence/build.txt"; \
        fi; \
        echo "==> [integration] cmake configure" \
     && cmake -B build_v2_integration -S src/v2 -G Ninja \
            -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" \
            -DCMAKE_BUILD_TYPE=Integration \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DHAVE_CUDA="${LLAMINAR_ENABLE_CUDA}" \
            -DHAVE_ROCM="${LLAMINAR_ENABLE_ROCM}" \
            -DLLAMINAR_CPU_ISA="${LLAMINAR_CPU_ISA}" \
            -DLLAMINAR_SHARED_CORE=ON \
            -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
            -DCMAKE_HIP_ARCHITECTURES="${rocm_gpu_targets}" \
            ${RCCL_CMAKE_ARGS} ${arm64_cmake_args} \
     && case "${LLAMINAR_TEST_RUNNER_INVENTORY}" in \
            full-matrix) integration_targets="v2_unit_gate v2_production_test_preflight_gate v2_model_parity_matrices" ;; \
            model-free) integration_targets="v2_unit_gate v2_production_test_preflight_gate" ;; \
            unit-only) integration_targets="v2_unit_gate" ;; \
            arm64-pr) integration_targets="v2_arm64_pr_gate" ;; \
            *) echo "Unsupported test-runner inventory: ${LLAMINAR_TEST_RUNNER_INVENTORY}" >&2; exit 1 ;; \
        esac \
     && echo "==> [integration] cmake build (parallel; matrices=${LLAMINAR_BUILD_MODEL_PARITY_MATRICES})" \
     && cmake --build build_v2_integration --parallel \
            --target ${integration_targets} \
     && echo "==> [integration] strip --strip-debug on executables/.a/.so (parallel, $(nproc) jobs)" \
     && { find build_v2_integration \
              \( -type f -executable -o -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
              -not -path '*/CMakeFiles/*' \
              -print0 \
          | xargs -0 -r -P "$(nproc)" -n 32 strip --strip-debug 2>/dev/null || true; } \
     && echo "==> [integration] removing intermediates (.o/.d/.gch/CMakeFiles)" \
     && mv build_v2_integration/CMakeFiles/rules.ninja build_v2_integration/installed-rules.ninja \
     && find build_v2_integration \
            ! -path '*/cpu_native_tile_registers/proof.o' \
            \( -name '*.o' -o -name '*.d' -o -name '*.gch' -o -name '*.cmake_pch.hxx' \) \
            -delete \
     && find build_v2_integration -depth -type d -name CMakeFiles -exec rm -rf {} + \
     && mkdir build_v2_integration/CMakeFiles \
     && mv build_v2_integration/installed-rules.ninja build_v2_integration/CMakeFiles/rules.ninja \
     && rm -rf build_v2_integration/Testing build_v2_integration/_deps/*-build/CMakeFiles \
     && echo "==> [integration] done; elapsed_seconds=$(( $(date +%s) - integration_started_epoch )); final size: $(du -sh build_v2_integration | cut -f1)"; \
    fi
