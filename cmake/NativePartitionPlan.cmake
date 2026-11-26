# @file NativePartitionPlan.cmake
# @brief Sole link authority for the cold reporters of CPUNativeOrderedPartitionPlan.h.
#
# The header's fatal admissions call failSuperblockWalk() and
# failOrderedPartitionPlan(). They are defined out of line in
# CPUNativeOrderedPartitionPlan.cpp so that the NEON tile programs' dependency
# closure carries no logging or iostream initialization. That translation unit
# is compiled exactly once, as the OBJECT library
# llaminar2_native_partition_plan:
#
# - llaminar2_core links it privately, which places the object in the core
#   (archive member or shared-object code) exactly as an ordinary core source;
# - every core-free target whose sources reach the header links the same
#   library instead of listing the reporter source itself.
#
# LLAMINAR_NATIVE_PARTITION_PLAN_FLAGS_TARGET, when set before the first
# inclusion, names the target whose complete transitive compile definitions,
# include directories, options and features the object inherits. The complete
# build sets it to llaminar2_core and creates this library at the point where
# the core is created, so the reporter keeps the production core's compile
# command and ABI. Without it (the core-free portability project and the
# build-time link fixture) the object uses the including directory's flags,
# C++20 and the src/v2 include root, like the core-free tests themselves.
include_guard(GLOBAL)

get_filename_component(LLAMINAR_NATIVE_PARTITION_PLAN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

add_library(llaminar2_native_partition_plan OBJECT
    "${LLAMINAR_NATIVE_PARTITION_PLAN_ROOT}/src/v2/kernels/cpu/gemm/CPUNativeOrderedPartitionPlan.cpp")
if(LLAMINAR_NATIVE_PARTITION_PLAN_FLAGS_TARGET)
    # Generator expressions read the flags target's transitive build properties
    # (its own plus its link dependencies' usage requirements) at generation
    # time. Only PRIVATE properties are set here, so the flags target linking
    # this library forms no evaluation cycle.
    set(_llaminar_plan_flags "${LLAMINAR_NATIVE_PARTITION_PLAN_FLAGS_TARGET}")
    target_compile_definitions(llaminar2_native_partition_plan PRIVATE
        "$<TARGET_PROPERTY:${_llaminar_plan_flags},COMPILE_DEFINITIONS>")
    target_include_directories(llaminar2_native_partition_plan PRIVATE
        "$<TARGET_PROPERTY:${_llaminar_plan_flags},INCLUDE_DIRECTORIES>")
    target_compile_options(llaminar2_native_partition_plan PRIVATE
        "$<TARGET_PROPERTY:${_llaminar_plan_flags},COMPILE_OPTIONS>")
    target_compile_features(llaminar2_native_partition_plan PRIVATE
        "$<TARGET_PROPERTY:${_llaminar_plan_flags},COMPILE_FEATURES>")
    unset(_llaminar_plan_flags)
else()
    target_include_directories(llaminar2_native_partition_plan PRIVATE
        "${LLAMINAR_NATIVE_PARTITION_PLAN_ROOT}/src/v2")
    target_compile_features(llaminar2_native_partition_plan PRIVATE cxx_std_20)
endif()
