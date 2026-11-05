#[[
@file V2CPUExpertDomainLifecycleTests.cmake
@brief Admit native CPU lifecycle gates only on discovered distinct physical sockets.
The canonical lscpu-derived V2_TEST_SOCKETS owns launch eligibility. Missing
populations remain unqualified; no oversubscribed process control substitutes
for physical sockets. Device-free membership and 1–8 production math remain.
]]
add_executable(v2_integration_cpu_expert_domain_lifecycle
    integration/collective/Test__CPUExpertDomainLifecycle.cpp mpi_session_gtest_main.cpp)
target_compile_features(v2_integration_cpu_expert_domain_lifecycle PRIVATE cxx_std_20)
target_link_libraries(v2_integration_cpu_expert_domain_lifecycle PRIVATE llaminar2_core GTest::gtest)

foreach(_cpu_socket_population IN ITEMS 2 4 8)
    if(NOT V2_TEST_SOCKETS OR V2_TEST_SOCKETS LESS _cpu_socket_population)
        message(STATUS "CPU expert native ${_cpu_socket_population}-socket lifecycle: unqualified (discovered ${V2_TEST_SOCKETS} physical sockets); no MPI launch")
        continue()
    endif()
    if(_cpu_socket_population EQUAL 2)
        set(_cpu_socket_case OriginalDomainRepeatedAndRetired)
    elseif(_cpu_socket_population EQUAL 4)
        set(_cpu_socket_case FourPhysicalSockets)
    else()
        set(_cpu_socket_case EightPhysicalSockets)
    endif()
    foreach(_cpu_socket_backend IN ITEMS UPI MPI)
        set(_cpu_socket_name V2_Integration_CPUExpertDomain_${_cpu_socket_backend}${_cpu_socket_case})
        add_v2_test(${_cpu_socket_name}
            COMMAND $<TARGET_FILE:v2_integration_cpu_expert_domain_lifecycle>
                --gtest_filter=CPUExpertDomainNative.${_cpu_socket_backend}${_cpu_socket_case}
            LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;MPI;Collective;Lifecycle;Replay"
            MPI_PROCS ${_cpu_socket_population} NO_MODELS TIMEOUT 120
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_cpu_socket_name})
    endforeach()
endforeach()

# This setup producer uses original source banks and one physical rank per
# socket. Its interval ends at the canonical output and excludes dispatch.
if(V2_TEST_SOCKETS AND V2_TEST_SOCKETS GREATER_EQUAL 2)
    add_v2_test(V2_Integration_CPUJointPreparedComputeService
        COMMAND $<TARGET_FILE:v2_integration_cpu_expert_domain_lifecycle>
            --gtest_filter=CPUExpertDomainNative.PreparedCPUJointComputeRootWallAndRetirement
        LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;MPI;Collective;Service;Measurement;Lifecycle;Replay"
        MPI_PROCS 2 NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointPreparedComputeService)

    # Negative jobs must fail through actual source/domain owners, and the
    # production reason must be visible before the mandatory original MPI abort.
    add_v2_test(V2_Integration_CPUPreparedServiceOriginalAbortDiagnostic
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_prepared_service_abort_diagnostic.py
            --helper $<TARGET_FILE:llaminar_mpi_session>
            --probe $<TARGET_FILE:v2_integration_cpu_expert_domain_lifecycle>
            --mpirun ${MPIEXEC_EXECUTABLE}
        LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;MPI;Collective;Service;Measurement;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 90
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUPreparedServiceOriginalAbortDiagnostic)
endif()

add_executable(v2_unit_cpu_expert_socket_membership unit/collective/Test__CPUExpertSocketMembership.cpp)
target_compile_features(v2_unit_cpu_expert_socket_membership PRIVATE cxx_std_20)
target_link_libraries(v2_unit_cpu_expert_socket_membership PRIVATE GTest::gtest_main)
add_v2_test(V2_Unit_CPUExpertSocketMembership
    COMMAND $<TARGET_FILE:v2_unit_cpu_expert_socket_membership>
    LABELS "V2;Unit;DeviceFree;CPU;Collective;Replay" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10)
add_v2_test(V2_Integration_CPUExpertSocketMembershipReplay
    COMMAND $<TARGET_FILE:v2_unit_cpu_expert_socket_membership> --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Collective;Replay"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUExpertSocketMembershipReplay)

# All phases reuse original prepared sources and retained native captures.
# Verifier rows compare byte-for-byte to the same inputs in actual serial decode;
# prefill and decode also exercise empty and partial CPU-cohort work.
foreach(_cpu_joint_backend IN ITEMS CUDA ROCm)
    string(TOUPPER "${_cpu_joint_backend}" _cpu_joint_backend_upper)
    if(HAVE_${_cpu_joint_backend_upper} AND V2_TEST_SOCKETS AND V2_TEST_SOCKETS GREATER_EQUAL 2)
        if(_cpu_joint_backend STREQUAL "CUDA")
            set(_cpu_joint_skip_cuda 0)
            set(_cpu_joint_skip_rocm 1)
        else()
            set(_cpu_joint_skip_cuda 1)
            set(_cpu_joint_skip_rocm 0)
        endif()
        foreach(_cpu_joint_phase IN ITEMS Prefill Decode Verifier)
            if(_cpu_joint_phase STREQUAL "Prefill")
                set(_cpu_joint_packet_name "V2_Integration_CPUJointCapturedRootedReplay_${_cpu_joint_backend}")
                set(_cpu_joint_packet_case "${_cpu_joint_backend}PreparedRootedPacketReplay")
            else()
                set(_cpu_joint_packet_name "V2_Integration_CPUJointCapturedRooted${_cpu_joint_phase}Replay_${_cpu_joint_backend}")
                set(_cpu_joint_packet_case "${_cpu_joint_backend}PreparedRooted${_cpu_joint_phase}Replay")
            endif()
            add_v2_test(${_cpu_joint_packet_name}
                COMMAND $<TARGET_FILE:v2_integration_cpu_expert_domain_lifecycle>
                    "--gtest_filter=CPUExpertDomainNative.${_cpu_joint_packet_case}"
                LABELS "V2;Integration;ProductionTestPreflight;${_cpu_joint_backend};CPU;MoE;MPI;Collective;Lifecycle;Replay;Contract"
                MPI_PROCS 2 NO_MODELS TIMEOUT 60
                ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0"
                    "LLAMINAR_SKIP_CUDA_STARTUP=${_cpu_joint_skip_cuda}"
                    "LLAMINAR_SKIP_ROCM_STARTUP=${_cpu_joint_skip_rocm}"
                    "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_cpu_joint_packet_name})
        endforeach()
    endif()
endforeach()
