# @file V2CPUHostCompilerTests.cmake
# @brief Register host ISA forwarding controls and actual build command admission.
include_guard(GLOBAL)
v2_add_core_free_test(V2_Unit_CPUHostCompilerOptions
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci"
        -p test_cpu_host_compiler_options.py
    LABELS V2 Unit DeviceFree CI)
v2_add_core_free_test(V2_Integration_CPUHostCompileContract
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci/cpu_host_compile_contract.py"
        --database "${CMAKE_BINARY_DIR}/compile_commands.json"
        "--flags=${LLAMINAR_CPU_COMPILE_FLAGS}"
    LABELS V2 Integration DeviceFree ProductionTestPreflight)
