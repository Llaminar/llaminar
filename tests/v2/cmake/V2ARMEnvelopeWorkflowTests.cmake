# @file V2ARMEnvelopeWorkflowTests.cmake
# @brief Register device-free ARM envelope orchestration failure controls.
include_guard(GLOBAL)
v2_add_core_free_test(V2_Unit_ARMEnvelopeWorkflow
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci"
        -p test_arm_envelope_checks.py
    LABELS V2 Unit DeviceFree CI)

v2_add_core_free_test(V2_Unit_NativeBuildEvidence
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci"
        -p test_arm_native_build_evidence.py
    LABELS V2 Unit DeviceFree CI)
v2_add_core_free_test(V2_Unit_NativeBuildCacheReceipt
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci"
        -p test_native_build_cache_receipt.py
    LABELS V2 Unit DeviceFree CI)
v2_add_core_free_test(V2_Unit_ArmWitnessToolchainManifest
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci"
        -p test_arm_witness_toolchain_manifest.py
    LABELS V2 Unit DeviceFree CI)
