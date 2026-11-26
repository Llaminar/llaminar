# @file V2CPUHostCompilerTests.cmake
# @brief Register host ISA forwarding controls and actual build command admission.
include_guard(GLOBAL)
v2_add_core_free_test(V2_Unit_CPUHostCompilerOptions
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci"
        -p test_cpu_host_compiler_options.py
    LABELS V2 Unit DeviceFree CI)
set(V2_CPU_HOST_DEFINITION_ARGS)
foreach(definition IN LISTS LLAMINAR_CPU_COMPILE_DEFINITIONS)
    list(APPEND V2_CPU_HOST_DEFINITION_ARGS "--definition=${definition}")
endforeach()
v2_add_core_free_test(V2_Integration_CPUHostCompileContract
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/ci/cpu_host_compile_contract.py"
        --database "${CMAKE_BINARY_DIR}/compile_commands.json"
        "--flags=${LLAMINAR_CPU_COMPILE_FLAGS}"
        ${V2_CPU_HOST_DEFINITION_ARGS}
    LABELS V2 Integration DeviceFree ProductionTestPreflight)
