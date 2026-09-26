# GPU memory-spill contract for all project-owned compiled kernels.
# Apply before creating backend/test targets. Dependencies built in independent
# projects and prebuilt vendor libraries are outside this compilation authority.
include_guard(GLOBAL)

get_filename_component(LLAMINAR_SPILL_GUARD_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LLAMINAR_GPU_SPILL_GUARD "${LLAMINAR_SPILL_GUARD_ROOT}/scripts/build/gpu_spill_guard.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LLAMINAR_GPU_SPILL_GUARD}")
file(SHA256 "${LLAMINAR_GPU_SPILL_GUARD}" LLAMINAR_GPU_SPILL_GUARD_ID)

# ptxas distinguishes register spills from intentional local memory. Promote its
# spill warning, not the broader local-memory warning, to a compilation error.
# Keep Debug usable for unoptimized source-level debugging.
add_compile_options(
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:Integration>>>:-Xptxas=--warn-on-spills,--warning-as-error>"
    # HIP -g otherwise manufactures scratch saves for frame-unwinding even in
    # register-light leaf helpers. Integration measures optimized production
    # code with symbols/snapshots; reserve those intrusive CFI saves for Debug.
    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Integration>>:SHELL:-Xarch_device -mllvm=-amdgpu-spill-cfi-saved-regs=false>"
    # Final stack layouts classify spills versus explicit private variables,
    # including device callees. stdout is cached by ccache; no sidecar is needed.
    # '-' inherits stdout's file descriptor instead of reopening /dev/stdout and
    # potentially overwriting another target's remarks or compiler diagnostics.
    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<OR:$<CONFIG:Release>,$<CONFIG:Integration>>>:SHELL:-Xarch_device -fsave-optimization-record=yaml -Xarch_device -foptimization-record-passes=stack-frame-layout -Xarch_device -foptimization-record-file=- -Xarch_device -Rpass-analysis=stack-frame-layout>")

if(CMAKE_HIP_COMPILER)
    get_filename_component(LLAMINAR_HIP_LLVM_BIN "${CMAKE_HIP_COMPILER}" DIRECTORY)
    foreach(LLAMINAR_SPILL_TOOL llvm-readelf llvm-readobj llvm-objcopy clang-offload-bundler)
        if(NOT EXISTS "${LLAMINAR_HIP_LLVM_BIN}/${LLAMINAR_SPILL_TOOL}")
            message(FATAL_ERROR "HIP spill enforcement requires ${LLAMINAR_HIP_LLVM_BIN}/${LLAMINAR_SPILL_TOOL}")
        endif()
    endforeach()
    # Preserve the existing launcher (normally ccache), but inspect its output on
    # hits as well as misses. $<CONFIG> also covers multi-config Ninja builds.
    set(CMAKE_HIP_COMPILER_LAUNCHER
        "${Python3_EXECUTABLE};${LLAMINAR_GPU_SPILL_GUARD};--llvm-bin;${LLAMINAR_HIP_LLVM_BIN};--config;$<CONFIG>;--policy-id;${LLAMINAR_GPU_SPILL_GUARD_ID};--;${CMAKE_HIP_COMPILER_LAUNCHER}")
endif()
