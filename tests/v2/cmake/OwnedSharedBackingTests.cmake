# These declarations own only functional targets and registrations. Release
# intentionally excludes them through V2TestTargetSelection/add_v2_test; return
# before attaching target properties or labels to omitted owners.
if(V2_PERF_TESTS_ONLY)
    return()
endif()

# Original-root shared file ownership and startup replay registration.
# All native work here is CPU MPI; no model or accelerator is initialized.
set(_owned_shared_cpu_env
    "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;HWLOC_COMPONENTS=-gl")
add_executable(v2_test_mpi_owned_shared_backing unit/utils/Test__MPIOwnedSharedBacking.cpp)
target_link_libraries(v2_test_mpi_owned_shared_backing PRIVATE llaminar_mpi_session_owner GTest::gtest)
add_v2_test(V2_Unit_MPIOwnedSharedBacking
    COMMAND $<TARGET_FILE:v2_test_mpi_owned_shared_backing>
    LABELS "V2;Unit;CPU;MPI;SharedMemory;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 ENVIRONMENT "${_owned_shared_cpu_env}")
add_v2_test(V2_Integration_MPIOwnedSharedFileIdentity
    COMMAND $<TARGET_FILE:v2_test_mpi_owned_shared_backing>
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MPI;SharedMemory;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 ENVIRONMENT "${_owned_shared_cpu_env}")
add_v2_test(V2_Integration_MPIIndependentLaunchFiles
    COMMAND $<TARGET_FILE:v2_test_mpi_owned_shared_backing>
        --gtest_filter=MPIOwnedSharedBacking.UniqueLaunchFilesCoexistAndRetireIndependently
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MPI;SharedMemory;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 ENVIRONMENT "${_owned_shared_cpu_env}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MPIIndependentLaunchFiles)
add_executable(v2_test_moe_owned_shared_startup unit/execution/moe/Test__MoEOverlaySharedBackingStartup.cpp)
target_include_directories(v2_test_moe_owned_shared_startup PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_owned_shared_startup PRIVATE llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_MoEOwnedSharedBackingStartup
    COMMAND $<TARGET_FILE:v2_test_moe_owned_shared_startup>
    LABELS "V2;Unit;CPU;MoE;MPI;SharedMemory;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 ENVIRONMENT "${_owned_shared_cpu_env}")
add_v2_test(V2_Integration_MoEOwnedSharedBackingStartupReplay
    COMMAND $<TARGET_FILE:v2_test_moe_owned_shared_startup>
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;MPI;SharedMemory;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 ENVIRONMENT "${_owned_shared_cpu_env}")
add_v2_test(V2_Integration_MoEOwnedSharedRankBatchStartup
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_node_local_rank_batch_transport>
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;MPI;SharedMemory;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 ENVIRONMENT "${_owned_shared_cpu_env}")

foreach(_owned_probe IN ITEMS BorrowProbe BorrowRepeat)
    string(TOLOWER ${_owned_probe} _owned_probe_lower)
    add_executable(v2_mpi_owned_shared_${_owned_probe_lower}
        integration/utils/MPIOwnedShared${_owned_probe}.cpp)
    target_include_directories(v2_mpi_owned_shared_${_owned_probe_lower} PRIVATE ${CMAKE_SOURCE_DIR}/../../external/vendor)
    target_link_libraries(v2_mpi_owned_shared_${_owned_probe_lower} PRIVATE llaminar_mpi_session_owner ${CMAKE_DL_LIBS})
endforeach()
add_library(v2_mpi_owned_shared_borrow_race MODULE integration/utils/MPIOwnedSharedBorrowRace.c)
target_link_libraries(v2_mpi_owned_shared_borrow_race PRIVATE ${CMAKE_DL_LIBS})
add_library(v2_mpi_owned_shared_pause MODULE integration/utils/MPIOwnedSharedBackingPause.c)
target_link_libraries(v2_mpi_owned_shared_pause PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(v2_mpi_owned_shared_borrow_race v2_mpi_owned_shared_pause PROPERTIES PREFIX "")
add_executable(v2_moe_owned_shared_startup_probe integration/utils/MoEOverlaySharedBackingStartupProbe.cpp)
target_link_libraries(v2_moe_owned_shared_startup_probe PRIVATE llaminar2_core)
add_v2_test(V2_Integration_MPIOwnedSharedBorrow
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_mpi_owned_shared_borrow.py
        --helper $<TARGET_FILE:llaminar_mpi_session>
        --probe $<TARGET_FILE:v2_mpi_owned_shared_borrowprobe>
        --repeat-probe $<TARGET_FILE:v2_mpi_owned_shared_borrowrepeat>
        --race-module $<TARGET_FILE:v2_mpi_owned_shared_borrow_race> --mpirun ${MPIEXEC_EXECUTABLE}
    LABELS "V2;Integration;ProductionTestPreflight;CPU;MPI;SharedMemory;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 90 ENVIRONMENT "${_owned_shared_cpu_env}")
add_v2_test(V2_Integration_MPIOwnedSharedBackingRetirement
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_mpi_owned_shared_backing.py
        --helper $<TARGET_FILE:llaminar_mpi_session> --runner $<TARGET_FILE:v2_test_shmem_spin_backend>
        --pause-module $<TARGET_FILE:v2_mpi_owned_shared_pause> --mpirun ${MPIEXEC_EXECUTABLE}
    LABELS "V2;Integration;ProductionTestPreflight;CPU;MPI;SharedMemory;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120 ENVIRONMENT "${_owned_shared_cpu_env}")
add_v2_test(V2_Integration_MoEOwnedSharedBackingNativeStartup
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_moe_owned_shared_startup.py
        --helper $<TARGET_FILE:llaminar_mpi_session> --probe $<TARGET_FILE:v2_moe_owned_shared_startup_probe>
        --mpirun ${MPIEXEC_EXECUTABLE}
    LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;MPI;SharedMemory;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120 ENVIRONMENT "${_owned_shared_cpu_env}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_MPIOwnedSharedFileIdentity V2_Integration_MoEOwnedSharedBackingStartupReplay
    V2_Integration_MoEOwnedSharedRankBatchStartup V2_Integration_MPIOwnedSharedBorrow
    V2_Integration_MPIOwnedSharedBackingRetirement V2_Integration_MoEOwnedSharedBackingNativeStartup)
unset(_owned_shared_cpu_env)

# Harness-owned diagnostic lifetime is independent of guardian-owned runtime roots.
foreach(_owned_evidence_suite Unit Integration)
    add_v2_test(V2_${_owned_evidence_suite}_OwnedSharedEvidenceLifetime
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_owned_shared_evidence_lifetime.py
        LABELS "V2;${_owned_evidence_suite};CPU;DeviceFree;MPI;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    add_v2_test(V2_${_owned_evidence_suite}_MPINativeDaemonDiagnosticLifetime
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_mpi_daemon_diagnostic_lifetime.py
        LABELS "V2;${_owned_evidence_suite};CPU;DeviceFree;MPI;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    if(_owned_evidence_suite STREQUAL "Integration")
        set_property(TEST V2_Integration_OwnedSharedEvidenceLifetime
            V2_Integration_MPINativeDaemonDiagnosticLifetime
            APPEND PROPERTY LABELS ProductionTestPreflight)
    endif()
endforeach()
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_OwnedSharedEvidenceLifetime V2_Integration_MPINativeDaemonDiagnosticLifetime)

# Actual process-entry loans must not share the guardian's enumeration cursor.
add_v2_test(V2_Integration_MPIRetirementCursorReplay
    COMMAND $<TARGET_FILE:v2_test_mpi_owned_shared_backing>
        --gtest_filter=MPIOwnedSharedBacking.GuardianRetiresAfterChildEnumeratesInheritedDirectoryLoans
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MPI;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
foreach(_retirement_suite Unit Integration)
    add_v2_test(V2_${_retirement_suite}_MPIRetirementDiagnostics
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_mpi_session_lifecycle.py
            --helper $<TARGET_FILE:llaminar_mpi_session> --probe $<TARGET_FILE:v2_mpi_session_probe>
            --mpirun ${MPIEXEC_EXECUTABLE} --case retirement
        LABELS "V2;${_retirement_suite};CPU;DeviceFree;MPI;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
endforeach()
set_property(TEST V2_Integration_MPIRetirementDiagnostics APPEND PROPERTY LABELS ProductionTestPreflight)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_MPIRetirementCursorReplay V2_Integration_MPIRetirementDiagnostics)

# Real owner reexec must preserve the admitted CPU domain before the new
# executable initializes its OpenMP runtime. Both aliases remain model-free.
add_executable(v2_mpi_process_affinity_probe integration/utils/MPIProcessAffinityProbe.cpp)
target_link_libraries(v2_mpi_process_affinity_probe PRIVATE llaminar_mpi_session_owner OpenMP::OpenMP_CXX)
add_dependencies(v2_mpi_process_affinity_probe llaminar_mpi_session)
foreach(_affinity_suite Unit Integration)
    add_v2_test(V2_${_affinity_suite}_MPIProcessAffinity
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_mpi_process_affinity.py
            --helper $<TARGET_FILE:llaminar_mpi_session>
            --probe $<TARGET_FILE:v2_mpi_process_affinity_probe>
        LABELS "V2;${_affinity_suite};CPU;DeviceFree;MPI;Threading;Topology;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
endforeach()
set_property(TEST V2_Integration_MPIProcessAffinity APPEND PROPERTY LABELS ProductionTestPreflight)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MPIProcessAffinity)

# Actual original-FD root retirement beyond unpublished/ECHILD frontiers.
# The body TU includes production source; it neither links the engine nor initializes MPI.
add_executable(v2_test_mpi_root_retirement_body unit/utils/Test__MPIRootRetirementBody.cpp)
target_include_directories(v2_test_mpi_root_retirement_body PRIVATE ${REPO_ROOT}/src/v2)
target_compile_features(v2_test_mpi_root_retirement_body PRIVATE cxx_std_20)
target_link_libraries(v2_test_mpi_root_retirement_body PRIVATE
    GTest::gtest_main OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
foreach(_root_suite Unit Integration)
    add_v2_test(V2_${_root_suite}_MPIRootRetirementBody
        COMMAND $<TARGET_FILE:v2_test_mpi_root_retirement_body> --gtest_repeat=20
        LABELS "V2;${_root_suite};CPU;DeviceFree;MPI;Lifecycle;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
endforeach()
set_property(TEST V2_Integration_MPIRootRetirementBody APPEND PROPERTY LABELS ProductionTestPreflight)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MPIRootRetirementBody)
add_v2_test(V2_Unit_MPIRootRetirementHelper
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_mpi_session_lifecycle.py
        --helper $<TARGET_FILE:llaminar_mpi_session> --probe $<TARGET_FILE:v2_mpi_session_probe>
        --mpirun ${MPIEXEC_EXECUTABLE} --case root_retirement
    LABELS "V2;Unit;CPU;DeviceFree;MPI;Lifecycle;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
foreach(_root_case root_unlink root_identity)
    foreach(_root_shard RANGE 0 3)
        set(_root_test V2_Integration_MPIRootRetirement_${_root_case}_Shard${_root_shard})
        add_v2_test(${_root_test}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_mpi_session_lifecycle.py
                --helper $<TARGET_FILE:llaminar_mpi_session> --probe $<TARGET_FILE:v2_mpi_session_probe>
                --mpirun ${MPIEXEC_EXECUTABLE} --case ${_root_case} --repeat 5
            LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MPI;Lifecycle;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_root_test})
    endforeach()
endforeach()
