# @file V2ARMEnvelopeWorkflowTests.cmake
# @brief Register device-free ARM envelope orchestration failure controls.
include_guard(GLOBAL)
v2_add_core_free_test(V2_Unit_ARMEnvelopeWorkflow
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci"
        -p test_arm_envelope_checks.py
    LABELS V2 Unit DeviceFree CI)
