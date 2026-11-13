# Source-only identity of compiled MTP controller semantics and trace interpretation.
# Producer and device-free replay share this generated header. These hashes cover
# small source files only, never models, caches, runtimes or other binary payloads.
include_guard(GLOBAL)
get_filename_component(_mtp_replay_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_mtp_replay_names
    policy_table learned_interpreter host_controller_header host_controller
    resident_controller control_layout transaction_trace runtime_policy
    policy_types policy_scope depth_defaults verifier_width_policy initialization device_policy_projection
    reversible_proposal_gate transaction_cost_profile publication_witness survival_score_interval survival_depth_selector)
set(_mtp_replay_paths
    src/v2/execution/mtp/MTPDepthPolicyGenerated.inc
    src/v2/execution/mtp/MTPDepthLearnedPolicy.h
    src/v2/execution/mtp/MTPDepthController.h
    src/v2/execution/mtp/MTPDepthController.cpp
    src/v2/kernels/common/SamplingMath.h
    src/v2/kernels/common/DeviceGenerationControlLayout.h
    src/v2/kernels/common/MTPDepthTransactionTrace.h
    src/v2/execution/config/RuntimeConfig.h
    src/v2/execution/config/MTPDepthPolicyTypes.h
    src/v2/execution/config/MTPDepthPolicyScope.h
    src/v2/execution/config/MTPDepthDefaults.h
    src/v2/execution/mtp/MTPVerifierPhysicalWidthPolicy.h
    src/v2/kernels/common/DeviceGenerationInitialization.h
    src/v2/execution/mtp/MTPDeviceGenerationPolicy.h
    src/v2/execution/mtp/MTPReversibleProposalGate.h
    src/v2/execution/mtp/MTPDepthCostProfile.h
    src/v2/execution/mtp/MTPPublicationWitness.h
    src/v2/execution/mtp/MTPSurvivalScoreInterval.h
    src/v2/execution/mtp/MTPSurvivalDepthSelector.h)
list(LENGTH _mtp_replay_names MTP_REPLAY_SOURCE_COUNT)
set(MTP_REPLAY_SOURCE_ROWS "")
math(EXPR _mtp_replay_last "${MTP_REPLAY_SOURCE_COUNT} - 1")
foreach(_mtp_replay_index RANGE ${_mtp_replay_last})
    list(GET _mtp_replay_names ${_mtp_replay_index} _mtp_replay_name)
    list(GET _mtp_replay_paths ${_mtp_replay_index} _mtp_replay_relative)
    set(_mtp_replay_path "${_mtp_replay_root}/${_mtp_replay_relative}")
    if(NOT EXISTS "${_mtp_replay_path}")
        message(FATAL_ERROR "Missing MTP replay identity source: ${_mtp_replay_path}")
    endif()
    # file(SHA256) alone does not tell Ninja to reconfigure after a policy edit.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_mtp_replay_path}")
    file(SHA256 "${_mtp_replay_path}" _mtp_replay_sha)
    string(APPEND MTP_REPLAY_SOURCE_ROWS "    {\"${_mtp_replay_name}\", \"${_mtp_replay_sha}\"},\n")
endforeach()
set(LLAMINAR_MTP_REPLAY_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/mtp-depth-replay")
file(MAKE_DIRECTORY "${LLAMINAR_MTP_REPLAY_INCLUDE_DIR}/execution/mtp")
configure_file("${_mtp_replay_root}/src/v2/execution/mtp/MTPDepthReplayBuildIdentity.h.in"
    "${LLAMINAR_MTP_REPLAY_INCLUDE_DIR}/execution/mtp/MTPDepthReplayBuildIdentity.h" @ONLY)
# This setup-time metadata header must also be visible to the core producer.
# The standalone replay target imports this directory explicitly, without core.
include_directories("${LLAMINAR_MTP_REPLAY_INCLUDE_DIR}")
