## @file V2CPUCodegenTargetManifest.cmake
## @brief Write the final C++ compile view of one target for CPU code-generation evidence.
##
## The codegen-envelope witness compiles an isolated ISA motif with the exact
## options of a real test target. CMake evaluates file(GENERATE) once for every
## enabled language, and target options guarded by $<COMPILE_LANGUAGE:CUDA>
## (or any other language) evaluate differently in each view. Writing all views
## to one output is a configure-time error, so the manifest records only the
## C++ view the probe actually compiles with.

## @brief Generate the target manifest for @p target at @p output.
## @param target Existing C++ target whose final options/definitions are recorded.
## @param output Generated manifest path.
## @param flags Configuration-specific C++ flag string.
function(v2_write_cpu_codegen_target_manifest target output flags)
    file(GENERATE OUTPUT "${output}" CONTENT
        "compiler=${CMAKE_CXX_COMPILER}\narg1=${CMAKE_CXX_COMPILER_ARG1}\nflags=${flags}\ntarget=${CMAKE_CXX_COMPILER_TARGET}\nsysroot=${CMAKE_SYSROOT}\nosx_sysroot=${CMAKE_OSX_SYSROOT}\noptions=$<TARGET_GENEX_EVAL:${target},$<TARGET_PROPERTY:${target},COMPILE_OPTIONS>>\ndefinitions=$<TARGET_GENEX_EVAL:${target},$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>>\n"
        CONDITION "$<COMPILE_LANGUAGE:CXX>")
endfunction()
