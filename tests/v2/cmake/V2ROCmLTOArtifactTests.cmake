#[=[
@file V2ROCmLTOArtifactTests.cmake
@brief Device-free validation of exact external collective output and compiler-owned resource evidence.

Only the dependency builder compiles witnesses and captures physical frames.
Installed runners revalidate sealed typed observations and the selected DSO;
no SDK, GPU discovery, model loading or inference admission policy is involved.
]=]
include_guard(DIRECTORY)
get_filename_component(V2_LTO_TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(V2_LTO_UNIT_SCRIPT "${V2_LTO_TEST_ROOT}/unit/scripts/test_llvm_lto_artifact.py")
add_test(NAME V2_Unit_ROCmLTOArtifact
    COMMAND ${Python3_EXECUTABLE} "${V2_LTO_UNIT_SCRIPT}")
set_tests_properties(V2_Unit_ROCmLTOArtifact PROPERTIES
    LABELS "V2;Unit;DeviceFree;Build;Ownership;Regression" TIMEOUT 30)
foreach(V2_LTO_CASE IN ITEMS OutputRoleTest WrapperDriverTest CompilerProofTest FinalResourceEvidenceTest)
    set(V2_LTO_TEST "V2_Integration_ROCmLTOArtifact_${V2_LTO_CASE}")
    add_test(NAME ${V2_LTO_TEST}
        COMMAND ${Python3_EXECUTABLE} "${V2_LTO_UNIT_SCRIPT}" "${V2_LTO_CASE}")
    set_tests_properties(${V2_LTO_TEST} PROPERTIES
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Build;Ownership;Regression" TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${V2_LTO_TEST})
endforeach()

# Keep the filtered-diagnostic regression individually selectable. Its parser
# controls complement the actual GVN executions retained in the compiler proof.
add_test(NAME V2_Integration_ROCmGVNFilteredRemarks
    COMMAND ${Python3_EXECUTABLE} "${V2_LTO_UNIT_SCRIPT}"
        CompilerProofTest.test_actual_guard_proof_rejects_excluded_work_lost_remarks_or_changed_ir)
set_tests_properties(V2_Integration_ROCmGVNFilteredRemarks PROPERTIES
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Build;Regression" TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmGVNFilteredRemarks)

# SDK aliases may change spelling, never the physical tool or ld.lld basename.
add_test(NAME V2_Integration_ROCmLTOExecutableAliases
    COMMAND ${Python3_EXECUTABLE} "${V2_LTO_UNIT_SCRIPT}"
        WrapperDriverTest.test_sdk_and_fixed_linker_aliases_authenticate_same_files_without_rewriting_commands
        WrapperDriverTest.test_missing_foreign_and_basename_changing_aliases_reject)
set_tests_properties(V2_Integration_ROCmLTOExecutableAliases PROPERTIES
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Build;Ownership;Regression" TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmLTOExecutableAliases)

# The wrapper's preview stream must be serial even when the actual link is parallel.
add_test(NAME V2_Integration_ROCmLTOPreviewConcurrency
    COMMAND ${Python3_EXECUTABLE} "${V2_LTO_UNIT_SCRIPT}"
        WrapperDriverTest.test_preview_serializes_after_original_job_policy_without_mutating_link_argv
        WrapperDriverTest.test_torn_parallel_preview_commands_are_rejected_without_reassembly)
set_tests_properties(V2_Integration_ROCmLTOPreviewConcurrency PROPERTIES
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Build;Ownership;Regression" TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmLTOPreviewConcurrency)

if(HAVE_ROCM AND HAVE_RCCL)
    get_filename_component(V2_LTO_REPO_ROOT "${V2_LTO_TEST_ROOT}/../.." ABSOLUTE)
    get_filename_component(V2_LTO_LIBRARY_DIR "${RCCL_LIBRARY}" DIRECTORY)
    set(RCCL_FINAL_ALLOCATION_REPORT "${V2_LTO_LIBRARY_DIR}/llaminar-final-allocation.json"
        CACHE FILEPATH "Complete physical resource evidence for the exact selected external RCCL DSO")
    set(RCCL_FINAL_ALLOCATION_RECEIPT "${V2_LTO_LIBRARY_DIR}/.llaminar-rccl-commit"
        CACHE FILEPATH "Exact RCCL build receipt bound by its final allocation proof")
    set(V2_LTO_ARTIFACT_SCRIPT "${V2_LTO_REPO_ROOT}/scripts/docker/llvm-lto-linker/artifact.py")
    set(V2_LTO_GUARD "${V2_LTO_REPO_ROOT}/scripts/build/gpu_spill_guard.py")
    set(V2_LTO_STAGED_PROOF "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_final_allocation.json")
    set(V2_LTO_STAGED_RECEIPT "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_final_allocation_receipt.txt")
    list(JOIN CMAKE_HIP_ARCHITECTURES ";" V2_LTO_TARGETS)
    add_custom_command(OUTPUT "${V2_LTO_STAGED_PROOF}" "${V2_LTO_STAGED_RECEIPT}"
        COMMAND ${Python3_EXECUTABLE} "${V2_LTO_ARTIFACT_SCRIPT}" verify
            --report "${RCCL_FINAL_ALLOCATION_REPORT}" --library "${RCCL_LIBRARY}"
            --receipt "${RCCL_FINAL_ALLOCATION_RECEIPT}" --spill-guard "${V2_LTO_GUARD}"
            --architectures "${V2_LTO_TARGETS}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RCCL_FINAL_ALLOCATION_REPORT}" "${V2_LTO_STAGED_PROOF}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RCCL_FINAL_ALLOCATION_RECEIPT}" "${V2_LTO_STAGED_RECEIPT}"
        DEPENDS "${RCCL_FINAL_ALLOCATION_REPORT}" "${RCCL_FINAL_ALLOCATION_RECEIPT}" "${RCCL_LIBRARY}"
            "${V2_LTO_ARTIFACT_SCRIPT}" "${V2_LTO_GUARD}"
            "${V2_LTO_REPO_ROOT}/scripts/docker/llvm-lto-linker/build.py"
            "${V2_LTO_REPO_ROOT}/scripts/docker/llvm-lto-linker/GVNRemarkProbe.cpp"
            "${V2_LTO_REPO_ROOT}/scripts/docker/rocm-release.env"
            "${V2_LTO_REPO_ROOT}/scripts/docker/patches/llvm-parallel-lto-remarks.patch"
            "${V2_LTO_REPO_ROOT}/scripts/docker/patches/llvm-gvn-filtered-remarks.patch"
        VERBATIM)
    add_custom_target(v2_rccl_final_allocation_proof
        DEPENDS "${V2_LTO_STAGED_PROOF}" "${V2_LTO_STAGED_RECEIPT}")
    set_property(TARGET v2_rccl_final_allocation_proof PROPERTY V2_RELEASE_NATIVE_PROOF_OWNER TRUE)
    set_property(GLOBAL APPEND PROPERTY V2_PRODUCTION_TEST_PREFLIGHT_BUILD_TARGETS v2_rccl_final_allocation_proof)
    set_property(GLOBAL APPEND PROPERTY V2_UNIT_TEST_BUILD_TARGETS v2_rccl_final_allocation_proof)
    add_test(NAME V2_Integration_RCCLFinalParallelLTOResourceEvidence
        COMMAND ${Python3_EXECUTABLE} "${V2_LTO_ARTIFACT_SCRIPT}" verify
            --report "${V2_LTO_STAGED_PROOF}" --library "${RCCL_LIBRARY}"
            --receipt "${V2_LTO_STAGED_RECEIPT}" --spill-guard "${V2_LTO_GUARD}"
            --architectures "${V2_LTO_TARGETS}")
    set_tests_properties(V2_Integration_RCCLFinalParallelLTOResourceEvidence PROPERTIES
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;DeviceFree;Build;Ownership;Regression" TIMEOUT 30)
    set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_V2_Integration_RCCLFinalParallelLTOResourceEvidence
        v2_rccl_final_allocation_proof)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RCCLFinalParallelLTOResourceEvidence)
endif()
