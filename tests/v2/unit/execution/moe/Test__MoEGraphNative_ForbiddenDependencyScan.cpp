/**
 * @file Test__MoEGraphNative_ForbiddenDependencyScan.cpp
 * @brief Source hygiene tests for graph-native MoE and backend-neutral MoE stages.
 *
 * These tests scan source files that are supposed to remain orchestration glue.
 * They catch accidental dependencies on legacy overlay runtime code and direct
 * CUDA/HIP runtime APIs before those dependencies can leak into compute stages.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        namespace fs = std::filesystem;

        fs::path findRepoRoot()
        {
            std::vector<fs::path> starts;
            starts.push_back(fs::current_path());
            starts.push_back(fs::path(__FILE__));

            for (auto start : starts)
            {
                if (fs::is_regular_file(start))
                    start = start.parent_path();

                for (fs::path candidate = start; !candidate.empty(); candidate = candidate.parent_path())
                {
                    if (fs::exists(candidate / "src/v2/execution/moe/MoEOverlaySparseCollective.h") &&
                        fs::exists(candidate / "tests/v2/CMakeLists.txt"))
                    {
                        return candidate;
                    }

                    if (candidate == candidate.root_path())
                        break;
                }
            }

            return fs::current_path();
        }

        std::string readFile(const fs::path &path)
        {
            std::ifstream input(path);
            if (!input)
                return {};

            std::ostringstream buffer;
            buffer << input.rdbuf();
            return buffer.str();
        }

        std::vector<fs::path> graphNativeFiles(const fs::path &root)
        {
            std::vector<fs::path> paths = {
                "src/v2/execution/compute_stages/stages/MoESparseDispatchStage.h",
                "src/v2/execution/compute_stages/stages/MoESparseDispatchStage.cpp",
                "src/v2/execution/compute_stages/stages/MoELocalExpertStage.h",
                "src/v2/execution/compute_stages/stages/MoELocalExpertStage.cpp",
                "src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.h",
                "src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.cpp",
                "src/v2/execution/moe/MoEOverlaySparseCollective.h",
                "src/v2/execution/moe/MoEOverlaySparseCollective.cpp",
                "src/v2/execution/moe/MoEExpertOwnerMap.h",
                "src/v2/execution/moe/MoEExpertOwnerMap.cpp",
                "src/v2/execution/moe/MoEGraphRoleRunner.h",
                "src/v2/execution/moe/MoEGraphRoleRunner.cpp",
            };

            const fs::path integration_dir = root / "tests/v2/integration/moe";
            if (fs::exists(integration_dir))
            {
                for (const auto &entry : fs::directory_iterator(integration_dir))
                {
                    if (!entry.is_regular_file())
                        continue;

                    const auto filename = entry.path().filename().string();
                    if (filename.rfind("Test__MoEGraphNative_", 0) == 0 && entry.path().extension() == ".cpp")
                        paths.push_back(fs::relative(entry.path(), root));
                }
            }

            return paths;
        }

        bool isStageFile(const fs::path &path)
        {
            return path.generic_string().find("src/v2/execution/compute_stages/stages/") != std::string::npos;
        }

        std::vector<fs::path> moeStageGlueFiles()
        {
            return {
                "src/v2/execution/compute_stages/stages/MoERoutingStage.h",
                "src/v2/execution/compute_stages/stages/MoERoutingStage.cpp",
                "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h",
                "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp",
                "src/v2/execution/compute_stages/stages/MoEExpertDispatchStage.h",
                "src/v2/execution/compute_stages/stages/MoEExpertDispatchStage.cpp",
                "src/v2/execution/compute_stages/stages/MoELocalExpertStage.h",
                "src/v2/execution/compute_stages/stages/MoELocalExpertStage.cpp",
                "src/v2/execution/compute_stages/stages/MoESparseDispatchStage.h",
                "src/v2/execution/compute_stages/stages/MoESparseDispatchStage.cpp",
                "src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.h",
                "src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.cpp",
            };
        }

    } // namespace

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GPUExpertTransferDoesNotExposePeerProbeAPI)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> files = {
            "src/v2/execution/moe/GPUExpertTransfer.h",
            "src/v2/execution/moe/GPUExpertTransfer.cpp",
            "src/v2/execution/moe/GPUExpertTransferBackend.h",
            "src/v2/execution/moe/GPUExpertTransferCUDA.cpp",
            "src/v2/execution/moe/GPUExpertTransferROCm.cpp",
        };
        const std::vector<std::string> forbidden_literals = {
            "canAccessPeer",
            "enablePeerAccess",
            "peer_access_cache",
            "peer_enable_cache",
            "host-staged",
            "P2P",
        };

        std::vector<std::string> failures;
        for (const auto &relative_path : files)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;
            for (const auto &token : forbidden_literals)
            {
                if (contents.find(token) != std::string::npos)
                    failures.push_back(relative_path.generic_string() + " contains obsolete transfer token " + token);
            }
        }

        EXPECT_TRUE(failures.empty()) << [&]
        {
            std::ostringstream out;
            for (const auto &failure : failures)
                out << failure << '\n';
            return out.str();
        }();
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, HomogeneousSendrecvMultiDelegatesToCoordinatorCopy)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> files = {
            "src/v2/collective/backends/NCCLBackend.cpp",
            "src/v2/collective/backends/RCCLBackend.cpp",
        };

        for (const auto &relative_path : files)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            const size_t fn_pos = contents.find("sendrecvMulti(");
            ASSERT_NE(fn_pos, std::string::npos) << relative_path;
            const size_t next_section = contents.find("// =========================================================================", fn_pos + 1);
            const std::string body = contents.substr(
                fn_pos,
                next_section == std::string::npos ? std::string::npos : next_section - fn_pos);

            EXPECT_NE(body.find("collectiveDataTypeByteSize(dtype)"), std::string::npos)
                << relative_path
                << " must convert typed counts to byte-exact transfers before copying packed blobs.";
            EXPECT_NE(body.find("coordinator_->copy(dst_buffer, dst_gpu, src_buffer, src_gpu, bytes)"),
                      std::string::npos)
                << relative_path
                << " sendrecvMulti must use the grouped coordinator copy path instead of a stub or bespoke P2P branch.";
            EXPECT_NE(body.find("requires distinct source and destination GPUs"), std::string::npos)
                << relative_path
                << " must fail same-GPU non-zero transfers rather than silently pretending an arrival moved.";
            EXPECT_EQ(body.find("not yet supported"), std::string::npos)
                << relative_path
                << " must not regress to a TODO stub.";
            EXPECT_EQ(body.find("not yet implemented"), std::string::npos)
                << relative_path
                << " must not regress to a TODO stub.";
            EXPECT_EQ(body.find("treating as no-op"), std::string::npos)
                << relative_path
                << " must not silently mark non-empty transfers as completed.";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceRebalanceSidebandsUseConfiguredRootParticipant)
    {
        const fs::path root = findRepoRoot();
        const fs::path path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(path)) << path;
        const std::string contents = readFile(path);
        ASSERT_FALSE(contents.empty()) << path;

        EXPECT_EQ(contents.find("root_device_index = 0"), std::string::npos)
            << "Graph-captured rebalance sidebands must use the configured domain root, not participant zero.";
        EXPECT_NE(contents.find("binding.config.root_participant"), std::string::npos)
            << "Qwen35 MoE graph rebalance sidebands must thread the device-side root participant.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PhaseSplitGpuPrefillCanUseLocalTPApportionedFastPath)
    {
        const fs::path root = findRepoRoot();
        const fs::path path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(path)) << path;
        const std::string contents = readFile(path);
        ASSERT_FALSE(contents.empty()) << path;

        const size_t candidate = contents.find("local_tp_apportioned_fast_candidate");
        ASSERT_NE(candidate, std::string::npos);
        const std::string gate = contents.substr(candidate, 1600);

        EXPECT_NE(gate.find("phase_split_local_tp_apportioned_gpu_prefill"), std::string::npos)
            << "Phase-split LocalTP GPU prefill must be allowed to use the grouped expert fast path; "
               "otherwise prefill falls back to the CPU sparse dispatch stage and rejects graph capture.";
        EXPECT_NE(gate.find("DenseParallelPolicy::PrefillTensorParallelDecodeReplicated"), std::string::npos);
        EXPECT_NE(gate.find("total_tokens > 1"), std::string::npos);
        EXPECT_NE(gate.find("phase_split_local_tp_apportioned_gpu_prefill"), std::string::npos);
        EXPECT_EQ(gate.find("(!device.is_gpu() || total_tokens == 1) &&"), std::string::npos)
            << "Do not regress to decode-only GPU LocalTP apportioned overlay fast path gating.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceRebalanceKernelsFilterResidentOnlyCandidatesBeforeRanking)
    {
        const fs::path root = findRepoRoot();
        const fs::path shared_path =
            root / "src/v2/execution/moe/DeviceMoERebalancePolicyShared.h";
        ASSERT_TRUE(fs::exists(shared_path)) << shared_path;
        const std::string shared_contents = readFile(shared_path);
        ASSERT_FALSE(shared_contents.empty()) << shared_path;
        EXPECT_NE(shared_contents.find("candidateCanAffectLocalCompute"), std::string::npos);
        EXPECT_NE(shared_contents.find("evaluateAddingResidentLoadSpread"), std::string::npos);
        EXPECT_NE(shared_contents.find("addingResidentImprovesLoadSpread"), std::string::npos);
        EXPECT_NE(shared_contents.find("evaluateAddingResidentLeastLoadedSpread"), std::string::npos);
        EXPECT_NE(shared_contents.find("addingResidentImprovesLeastLoadedSpread"), std::string::npos);
        EXPECT_NE(shared_contents.find("evaluateAddingResidentDynamicSpread"), std::string::npos);
        EXPECT_NE(shared_contents.find("addingResidentImprovesDynamicSpread"), std::string::npos);
        EXPECT_NE(shared_contents.find("dynamicMinimumProjectedShift"), std::string::npos);
        EXPECT_NE(shared_contents.find("expertCountCanMeetLoadSpreadFloor"),
                  std::string::npos);
        EXPECT_NE(shared_contents.find("requiredLoadSpreadImprovement"),
                  std::string::npos);
        EXPECT_NE(shared_contents.find("candidateValueIsBetter"), std::string::npos);
        EXPECT_NE(shared_contents.find("hasValidRootParticipant"), std::string::npos);
        EXPECT_NE(shared_contents.find("isRootParticipant"), std::string::npos);
        EXPECT_NE(shared_contents.find("DestinationChoice"), std::string::npos);
        EXPECT_NE(shared_contents.find("bestMissingResidentDestination"), std::string::npos);
        EXPECT_NE(shared_contents.find("bestLeastLoadedMissingResidentDestination"), std::string::npos);
        EXPECT_NE(shared_contents.find("bestDynamicMissingResidentDestination"), std::string::npos);
        EXPECT_NE(shared_contents.find("candidateCanAffectDomainCompute"), std::string::npos);
        EXPECT_NE(shared_contents.find("transferWaveMeetsSpreadImprovementFloor"), std::string::npos);
        EXPECT_NE(shared_contents.find("!plan_missing_arrivals || local_resident || owner_local"),
                  std::string::npos)
            << "ResidentOnly mode must not rank experts that need an arrival transfer.";

        const std::vector<fs::path> files = {
            "src/v2/execution/moe/DeviceMoERebalanceController.h",
            "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu",
            "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip",
        };

        for (const auto &relative_path : files)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            EXPECT_NE(contents.find("DeviceMoERebalancePolicyShared.h"), std::string::npos)
                << relative_path
                << " must use the shared host/device rebalance policy helpers.";
            EXPECT_NE(contents.find("moe_rebalance_policy::candidateCanAffectLocalCompute"),
                      std::string::npos)
                << relative_path
                << " must call the shared resident-only candidate filter.";
            EXPECT_NE(contents.find("moe_rebalance_policy::addingResidentImprovesDynamicSpread"),
                      std::string::npos)
                << relative_path
                << " must gate Dynamic arrivals through the shared Dynamic load-spread helper.";
            EXPECT_NE(contents.find("moe_rebalance_policy::evaluateAddingResidentDynamicSpread"),
                      std::string::npos)
                << relative_path
                << " must evaluate missing-arrival transfer candidates through the shared Dynamic projected load-spread helper.";
            EXPECT_NE(contents.find("moe_rebalance_policy::candidateValueIsBetter"),
                      std::string::npos)
                << relative_path
                << " must rank missing-arrival transfers by projected benefit rather than raw route count.";
            EXPECT_NE(contents.find("moe_rebalance_policy::candidateCanAffectDomainCompute"),
                      std::string::npos)
                << relative_path
                << " must expose root-domain candidate filtering for graph-captured rebalance.";
            EXPECT_NE(contents.find("moe_rebalance_policy::bestDynamicMissingResidentDestination"),
                      std::string::npos)
                << relative_path
                << " must select Dynamic missing-arrival destinations by the shared Dynamic policy.";
            if (relative_path.extension() == ".h")
            {
                EXPECT_NE(contents.find("moe_rebalance_policy::expertCountCanMeetLoadSpreadFloor"),
                          std::string::npos)
                    << relative_path
                    << " must prune missing arrivals that cannot meet the configured count-bound floor before ranking.";
            }
            else
            {
                EXPECT_NE(contents.find("shared_required_load_spread_improvement"),
                          std::string::npos)
                    << relative_path
                    << " must cache the configured count-bound floor before ranking missing arrivals.";
                EXPECT_NE(contents.find("moe_rebalance_policy::requiredLoadSpreadImprovement"),
                          std::string::npos)
                    << relative_path
                    << " must compute the count-bound floor through the shared policy helper.";
                EXPECT_NE(contents.find("shared_destination_replica_counts"),
                          std::string::npos)
                    << relative_path
                    << " must track root-domain replica budgets per destination participant.";
                EXPECT_NE(contents.find("domain_root_planning"),
                          std::string::npos)
                    << relative_path
                    << " must keep graph-captured root-domain planning explicit.";
                EXPECT_NE(contents.find("destination_choice.destination_participant"),
                          std::string::npos)
                    << relative_path
                    << " must not hard-code graph root arrivals to the local participant.";
            }
            EXPECT_NE(contents.find("root_participant"), std::string::npos)
                << relative_path
                << " must expose the domain root in the host/device rebalance config ABI.";
            EXPECT_NE(contents.find("hasValidRootParticipant"), std::string::npos)
                << relative_path
                << " must validate the root participant through the shared policy helper.";
            if (relative_path.extension() == ".cu" || relative_path.extension() == ".hip")
            {
                const size_t launch_start =
                    relative_path.extension() == ".cu"
                        ? contents.find("bool cudaMoE_device_rebalance_controller(")
                        : contents.find("bool hipMoE_device_rebalance_controller(");
                ASSERT_NE(launch_start, std::string::npos) << relative_path;
                const std::string launch_body = contents.substr(launch_start, 2000);
                EXPECT_NE(contents.find("shared_candidate_counts"), std::string::npos)
                    << relative_path
                    << " must parallelize hot-candidate ranking across a controller block.";
                EXPECT_NE(contents.find("__syncthreads()"), std::string::npos)
                    << relative_path
                    << " must synchronize the cooperative controller block.";
                EXPECT_NE(contents.find("kDeviceMoEMaxExperts"), std::string::npos)
                    << relative_path
                    << " should launch one lane per expert for the bounded MoE domain.";
                EXPECT_EQ(launch_body.find("device_rebalance_controller_kernel<<<1, 1"),
                          std::string::npos)
                    << relative_path
                    << " must not regress to a one-thread controller launch.";
                EXPECT_EQ(launch_body.find("dim3(1), dim3(1)"),
                          std::string::npos)
                    << relative_path
                    << " must not regress to a one-thread controller launch.";
                EXPECT_NE(contents.find("payload_bucket_requested_slots"), std::string::npos)
                    << relative_path
                    << " must publish device-side payload bucket scheduler state.";
                EXPECT_NE(contents.find("payload_slot_capacity"), std::string::npos)
                    << relative_path
                    << " must keep payload lane capacity separate from command metadata capacity.";
                EXPECT_NE(contents.find("shared_destination_transfer_slot_counts"), std::string::npos)
                    << relative_path
                    << " must allocate receiver staging slots per destination.";
                EXPECT_NE(contents.find("shared_source_payload_slot_counts"), std::string::npos)
                    << relative_path
                    << " must allocate compact transfer payload slots densely per source participant.";
                EXPECT_NE(contents.find("plan.payload_slot"), std::string::npos)
                    << relative_path
                    << " must carry sender payload slots separately from receiver transfer slots.";
                EXPECT_EQ(contents.find("payload_slots_per_destination"), std::string::npos)
                    << relative_path
                    << " must not reintroduce destination-major compact payload lanes.";
                EXPECT_NE(contents.find("payloadBucketSlots("), std::string::npos)
                    << relative_path
                    << " must use the shared bucket rounding helper.";
                EXPECT_NE(contents.find("rebalance_plan_requires_payload(plan.op)"),
                          std::string::npos)
                    << relative_path
                    << " must size payload buckets from payload-bearing expert movement, not resident metadata commands.";
                EXPECT_NE(contents.find("const bool hot_replica_cache"), std::string::npos)
                    << relative_path
                    << " must keep LLEP movement semantics separate from the optional hot-cache layer.";
                EXPECT_NE(contents.find("const bool durable_llep_ownership_wave"), std::string::npos)
                    << relative_path
                    << " must keep durable LLEP ownership waves out of the replica/cache post-spread ceiling.";
                EXPECT_NE(contents.find("ownership_transfer\n                                       ? kDeviceMoERebalancePlanOwnershipTransfer"),
                          std::string::npos)
                    << relative_path
                    << " LLEP without HotExpertReplicaCache must publish durable ownership transfers.";
            }

            size_t body_start = contents.find("applyDeviceMoERebalancePolicyHost(");
            if (body_start == std::string::npos)
                body_start = contents.find("device_rebalance_controller_kernel(");
            ASSERT_NE(body_start, std::string::npos) << relative_path;
            const std::string body = contents.substr(body_start);
            const size_t ranking_start = body.find("for (uint32_t rank");
            ASSERT_NE(ranking_start, std::string::npos) << relative_path;
            const std::string ranking_body = body.substr(ranking_start, 5000);

            const size_t host_filter_call = ranking_body.find("RebalanceCandidateCanAffectLocalCompute");
            const size_t shared_filter_call = ranking_body.find("candidateCanAffectLocalCompute");
            const size_t cuda_hip_filter_call =
                ranking_body.find("rebalance_candidate_can_affect_local_compute(");
            const size_t global_count = ranking_body.find("GlobalExpertCount");
            const size_t cuda_hip_global_count = ranking_body.find("rebalance_global_count");
            const size_t filter_call =
                host_filter_call != std::string::npos ? host_filter_call : shared_filter_call;
            const size_t filter = filter_call != std::string::npos ? filter_call : cuda_hip_filter_call;
            const size_t count = global_count != std::string::npos ? global_count : cuda_hip_global_count;
            ASSERT_NE(filter, std::string::npos) << relative_path;
            ASSERT_NE(count, std::string::npos) << relative_path;
            EXPECT_LT(filter, count)
                << relative_path
                << " must reject non-local missing experts before reading/ranking histogram counts.";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceRebalanceDomainRootPlanningSkipsDiscardedNonRootSearch)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> files = {
            "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu",
            "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip",
        };

        for (const auto &relative_path : files)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            const size_t root_planning =
                contents.find("const bool domain_root_planning = defer_runtime_apply && plan_missing_arrivals;");
            ASSERT_NE(root_planning, std::string::npos) << relative_path;
            const size_t non_root_fast_path =
                contents.find("if (domain_root_planning && config.participant_id != config.root_participant)",
                              root_planning);
            ASSERT_NE(non_root_fast_path, std::string::npos) << relative_path;
            const size_t ranking_loop =
                contents.find("for (uint32_t window_index = 0; window_index < layer_wave_count; ++window_index)",
                              root_planning);
            ASSERT_NE(ranking_loop, std::string::npos) << relative_path;

            EXPECT_LT(non_root_fast_path, ranking_loop)
                << relative_path
                << " must skip discarded non-root candidate search before scanning the rolling layer wave.";

            const std::string fast_path = contents.substr(non_root_fast_path,
                                                          ranking_loop - non_root_fast_path);
            const std::string planning_context = contents.substr(root_planning,
                                                                  ranking_loop - root_planning);
            EXPECT_NE(fast_path.find("command_header->command_count = 0u;"),
                      std::string::npos)
                << relative_path
                << " non-root root-domain planning must publish an empty local command header.";
            EXPECT_NE(fast_path.find("wave_state->next_start_layer"), std::string::npos)
                << relative_path
                << " non-root participants must still advance the rolling wave cursor.";
            EXPECT_NE(fast_path.find("return;"), std::string::npos)
                << relative_path
                << " non-root participants must not run the root-only policy search.";
            EXPECT_NE(planning_context.find("project_rebalance_domain_commands()"), std::string::npos)
                << relative_path
                << " the fast path comment must document why the root plan remains authoritative.";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceRebalanceCandidateRankingRespectsCompactPayloadSlots)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> files = {
            "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu",
            "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip",
        };

        for (const auto &relative_path : files)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            const size_t root_choice =
                contents.find("llaminar2::moe_rebalance_policy::bestDynamicMissingResidentDestination");
            ASSERT_NE(root_choice, std::string::npos) << relative_path;
            const std::string root_ranking = contents.substr(root_choice, 5200);
            const size_t root_slot_gate = root_ranking.find("destination_has_transfer_slot");
            const size_t root_source_slot_gate = root_ranking.find("source_has_payload_slot");
            const size_t root_value =
                root_ranking.find("candidate_value = source_delta.improvement");
            ASSERT_NE(root_slot_gate, std::string::npos) << relative_path;
            ASSERT_NE(root_source_slot_gate, std::string::npos) << relative_path;
            ASSERT_NE(root_value, std::string::npos) << relative_path;
            EXPECT_LT(root_slot_gate, root_value)
                << relative_path
                << " root-domain transfer candidates must check destination transfer-slot capacity before ranking.";
            EXPECT_LT(root_source_slot_gate, root_value)
                << relative_path
                << " root-domain transfer candidates must check source payload-slot capacity before ranking.";
            EXPECT_NE(root_ranking.find("shared_destination_transfer_slot_counts[\n"
                                        "                                                destination_choice.destination_participant] < payload_slot_capacity"),
                      std::string::npos)
                << relative_path
                << " root-domain ranking must use per-destination transfer-slot counts.";
            EXPECT_NE(root_ranking.find("shared_source_payload_slot_counts[\n"
                                        "                                                static_cast<uint32_t>(source_participant)] < payload_slot_capacity"),
                      std::string::npos)
                << relative_path
                << " root-domain ranking must use dense per-source payload-slot counts.";

            const size_t local_branch = contents.find("else if (plan_missing_arrivals");
            ASSERT_NE(local_branch, std::string::npos) << relative_path;
            const std::string local_ranking = contents.substr(local_branch, 4200);
            const size_t local_slot_gate =
                local_ranking.find("shared_destination_transfer_slot_counts[config.participant_id]");
            const size_t local_source_slot_gate = local_ranking.find("source_has_payload_slot");
            const size_t local_value = local_ranking.find("candidate_value = delta.improvement");
            ASSERT_NE(local_slot_gate, std::string::npos) << relative_path;
            ASSERT_NE(local_source_slot_gate, std::string::npos) << relative_path;
            ASSERT_NE(local_value, std::string::npos) << relative_path;
            EXPECT_LT(local_slot_gate, local_value)
                << relative_path
                << " local transfer candidates must check destination transfer-slot capacity before ranking.";
            EXPECT_LT(local_source_slot_gate, local_value)
                << relative_path
                << " local transfer candidates must check source payload-slot capacity before ranking.";

            const size_t bucket_slots = contents.find("payloadBucketSlots(\n                    requested_payload_slots");
            const size_t wave_floor = contents.find("transferWaveMeetsSpreadImprovementFloor");
            const size_t post_spread_ceiling = contents.find("transferWaveMeetsPostLoadSpreadCeiling");
            const size_t prune_arrivals =
                contents.find("prunePayloadArrivalsPreservingResidentAssignments");
            const size_t header_epoch = contents.find("command_header->epoch = defer_runtime_apply && command_count > 0u");
            ASSERT_NE(bucket_slots, std::string::npos) << relative_path;
            ASSERT_NE(wave_floor, std::string::npos) << relative_path;
            ASSERT_NE(post_spread_ceiling, std::string::npos) << relative_path;
            ASSERT_NE(prune_arrivals, std::string::npos) << relative_path;
            ASSERT_NE(header_epoch, std::string::npos) << relative_path;
            EXPECT_LT(bucket_slots, wave_floor)
                << relative_path
                << " transfer-wave value gating must happen after payload-slot request sizing.";
            EXPECT_LT(wave_floor, post_spread_ceiling)
                << relative_path
                << " residual post-policy imbalance gating must run with the same payload-slot economics.";
            EXPECT_LT(post_spread_ceiling, prune_arrivals)
                << relative_path
                << " rejected transfer waves must be value-gated before resident-only compaction.";
            EXPECT_LT(prune_arrivals, header_epoch)
                << relative_path
                << " rejected transfer waves must be compacted to resident-only commands before command header publication.";
            EXPECT_NE(contents.find("status->skipped_wave_cost_floor"), std::string::npos)
                << relative_path
                << " perfstats must expose wave-level value-gate skips.";
            EXPECT_NE(contents.find("transferWaveMeetsRealizedRouterBenefitFloor"), std::string::npos)
                << relative_path
                << " transfer-wave economics must also account for realized router hot-cache benefit.";
            EXPECT_NE(contents.find("status->skipped_low_router_benefit"), std::string::npos)
                << relative_path
                << " perfstats must expose realized-router-benefit gate skips.";
            EXPECT_NE(contents.find("status->skipped_post_load_spread_ceiling"), std::string::npos)
                << relative_path
                << " perfstats must expose residual post-policy load-spread gate skips.";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalanceSlotLeasesUseRAIINotRawTokens)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> files = {
            "src/v2/execution/moe/GpuExpertSlotPool.cpp",
            "src/v2/execution/moe/GpuExpertTransferStagingPool.cpp",
        };

        for (const auto &relative_path : files)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            EXPECT_EQ(contents.find("new SlotLeaseToken"), std::string::npos)
                << relative_path
                << " must not allocate active/transfer slot lease tokens manually.";
            EXPECT_EQ(contents.find("new StagingLeaseToken"), std::string::npos)
                << relative_path
                << " must not allocate transfer staging lease tokens manually.";
            EXPECT_EQ(contents.find("unique_ptr<SlotLeaseToken>"), std::string::npos)
                << relative_path
                << " lease release should be tied directly to shared lifetime, not a raw-token deleter.";
            EXPECT_EQ(contents.find("unique_ptr<StagingLeaseToken>"), std::string::npos)
                << relative_path
                << " lease release should be tied directly to shared lifetime, not a raw-token deleter.";
            EXPECT_NE(contents.find("std::make_shared<"), std::string::npos)
                << relative_path
                << " lease tokens should be created through RAII shared ownership.";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DynamicOwnershipSwapPolicyCountersStaySymmetric)
    {
        const fs::path root = findRepoRoot();
        const fs::path shared_path =
            root / "src/v2/execution/moe/DeviceMoERebalanceController.h";
        const fs::path perf_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(shared_path)) << shared_path;
        ASSERT_TRUE(fs::exists(perf_path)) << perf_path;

        const std::string shared = readFile(shared_path);
        const std::string perf = readFile(perf_path);
        ASSERT_FALSE(shared.empty()) << shared_path;
        ASSERT_FALSE(perf.empty()) << perf_path;

        for (const auto &token : {
                 "dynamic_ownership_swap_attempts",
                 "dynamic_ownership_swap_accepts",
                 "dynamic_ownership_swap_rejections",
             })
        {
            EXPECT_NE(shared.find(token), std::string::npos)
                << "Device rebalance status must expose " << token;
            EXPECT_NE(perf.find(token), std::string::npos)
                << "Maintenance trace JSON must expose " << token;
            EXPECT_NE(perf.find("device_rebalance_" + std::string(token)),
                      std::string::npos)
                << "Perfstats must expose " << token;
        }

        const std::vector<fs::path> kernel_files = {
            "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu",
            "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip",
        };
        for (const auto &relative_path : kernel_files)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            EXPECT_NE(contents.find("++dynamic_ownership_swap_attempts"), std::string::npos)
                << relative_path << " must count Dynamic ownership policy attempts.";
            EXPECT_NE(contents.find("++dynamic_ownership_swap_accepts"), std::string::npos)
                << relative_path << " must count accepted Dynamic ownership moves.";
            EXPECT_NE(contents.find("++dynamic_ownership_swap_rejections"), std::string::npos)
                << relative_path << " must count policy-declined Dynamic ownership attempts.";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, SharedDynamicPolicyKnobsReachGpuGraphConfig)
    {
        const fs::path root = findRepoRoot();
        const fs::path runtime_config_path = root / "src/v2/execution/config/RuntimeConfig.h";
        const fs::path debug_env_path = root / "src/v2/utils/DebugEnv.h";
        const fs::path parser_path = root / "src/v2/config/OrchestrationConfigParser.cpp";
        const fs::path factory_path = root / "src/v2/execution/factory/InferenceRunnerFactory.cpp";
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";

        const std::string runtime_config = readFile(runtime_config_path);
        const std::string debug_env = readFile(debug_env_path);
        const std::string parser = readFile(parser_path);
        const std::string factory = readFile(factory_path);
        const std::string graph = readFile(graph_path);

        ASSERT_FALSE(runtime_config.empty()) << runtime_config_path;
        ASSERT_FALSE(debug_env.empty()) << debug_env_path;
        ASSERT_FALSE(parser.empty()) << parser_path;
        ASSERT_FALSE(factory.empty()) << factory_path;
        ASSERT_FALSE(graph.empty()) << graph_path;

        for (const auto &token : {
                 "dynamic_imbalance_threshold_per_mille",
                 "dynamic_min_improvement_per_mille",
                 "dynamic_max_swaps_per_layer",
                 "dynamic_max_plan_entries_per_wave",
                 "dynamic_min_window_activations",
             })
        {
            EXPECT_NE(runtime_config.find(token), std::string::npos)
                << "Runtime config must carry shared Dynamic policy field " << token;
            EXPECT_NE(factory.find(token), std::string::npos)
                << "Inference runner factory must thread " << token
                << " into CPU/host rebalance config.";
            EXPECT_NE(graph.find("rebalance_config." + std::string(token)),
                      std::string::npos)
                << "Qwen35 graph must thread " << token
                << " into DeviceMoERebalanceConfig for CUDA/ROCm.";
        }

        for (const auto &env_name : {
                 "LLAMINAR_MOE_DYNAMIC_IMBALANCE_THRESHOLD_PERMILLE",
                 "LLAMINAR_MOE_DYNAMIC_MIN_IMPROVEMENT_PERMILLE",
                 "LLAMINAR_MOE_DYNAMIC_MAX_SWAPS_PER_LAYER",
                 "LLAMINAR_MOE_DYNAMIC_MAX_PLAN_ENTRIES_PER_WAVE",
                 "LLAMINAR_MOE_DYNAMIC_MIN_WINDOW_ACTIVATIONS",
             })
        {
            EXPECT_NE(debug_env.find(env_name), std::string::npos)
                << "DebugEnv must expose " << env_name;
        }

        for (const auto &cli_flag : {
                 "--moe-dynamic-imbalance-threshold-permille",
                 "--moe-dynamic-min-improvement-permille",
                 "--moe-dynamic-max-swaps-per-layer",
                 "--moe-dynamic-max-plan-entries-per-wave",
                 "--moe-dynamic-min-window-activations",
             })
        {
            EXPECT_NE(parser.find(cli_flag), std::string::npos)
                << "CLI parser must expose " << cli_flag;
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GraphNativeFilesDoNotReferenceLegacyOverlayRuntime)
    {
        const fs::path root = findRepoRoot();
        const std::vector<std::string> forbidden_literals = {
            "IOverlayDomainRuntime",
            "MoEOverlayDomainRuntimeStage",
            "MoEOverlayCPUFallbackParticipantRunner",
            "MoEExpertOverlayLocalTPStage",
            "MoEExpertOverlayLocalTPExecutor",
            "ILocalTPContext",
            "prepared_participants",
        };

        std::vector<std::string> failures;
        for (const auto &relative_path : graphNativeFiles(root))
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            for (const auto &token : forbidden_literals)
            {
                if (contents.find(token) != std::string::npos)
                    failures.push_back(relative_path.generic_string() + " contains forbidden token " + token);
            }

            if (isStageFile(relative_path))
            {
                const std::regex role_runner_pointer("\\bMoEGraphRoleRunner\\s*[*&]");
                if (std::regex_search(contents, role_runner_pointer))
                {
                    failures.push_back(relative_path.generic_string() +
                                       " contains a MoEGraphRoleRunner pointer/reference in stage code");
                }
            }
        }

        EXPECT_TRUE(failures.empty()) << [&]
        {
            std::ostringstream out;
            for (const auto &failure : failures)
                out << failure << '\n';
            return out.str();
        }();
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, LegacyOverlayRuntimeSourcesAndProductionReferencesAreRemoved)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> deleted_paths = {
            "src/v2/execution/compute_stages/stages/MoEOverlayDomainRuntimeStage.h",
            "src/v2/execution/compute_stages/stages/MoEOverlayDomainRuntimeStage.cpp",
            "src/v2/execution/compute_stages/stages/MoEExpertOverlayLocalTPStage.h",
            "src/v2/execution/compute_stages/stages/MoEExpertOverlayLocalTPStage.cpp",
            "src/v2/execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h",
            "src/v2/execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.cpp",
            "src/v2/execution/moe/IOverlayDomainRuntime.h",
            "src/v2/execution/moe/MoEOverlayDomainRuntime.h",
            "src/v2/execution/moe/MoEOverlayDomainRuntime.cpp",
            "src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.h",
            "src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.cpp",
            "src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.h",
            "src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.cpp",
            "src/v2/execution/moe/MoEOverlayDispatchCollective.h",
            "src/v2/execution/moe/MoEOverlayDispatchCollective.cpp",
            "src/v2/execution/moe/MoEOverlayMPIDispatchBackend.h",
            "src/v2/execution/moe/MoEOverlayMPIDispatchBackend.cpp",
            "src/v2/execution/moe/MoEExpertOverlayCPUFallback.h",
            "src/v2/execution/moe/MoEExpertOverlayCPUFallback.cpp",
        };

        for (const auto &relative_path : deleted_paths)
            EXPECT_FALSE(fs::exists(root / relative_path)) << relative_path;

        const std::vector<fs::path> production_paths = {
            "src/v2/CMakeLists.txt",
            "src/v2/models/GraphTypes.h",
            "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp",
            "src/v2/execution/factory/InferenceRunnerFactory.h",
            "src/v2/execution/factory/InferenceRunnerFactory.cpp",
            "src/v2/execution/runner/OrchestrationRunner.h",
            "src/v2/execution/runner/OrchestrationRunner.cpp",
            "src/v2/execution/local_execution/orchestrators/RankOrchestrator.h",
            "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp",
            "src/v2/execution/compute_stages/ComputeStageFactory.h",
            "src/v2/execution/compute_stages/ComputeStageFactory.cpp",
            "src/v2/execution/compute_stages/stages/MoERoutedExpertPartialReduceStage.h",
            "src/v2/execution/compute_stages/stages/MoERoutedExpertPartialReduceStage.cpp",
            "src/v2/execution/moe/MoEExpertOverlayProfiler.h",
            "src/v2/execution/moe/MoEExpertOverlayProfiler.cpp",
        };
        const std::vector<std::string> removed_tokens = {
            "LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME",
            "legacyOverlayDomainRuntimeEnabled",
            "IOverlayDomainRuntime",
            "overlay_domain_runtime",
            "MoEOverlayDomainRuntimeStage",
            "MoEOverlayDomainRuntime",
            "MoEOverlayDomainWorkResult",
            "MoEExpertOverlayLocalTPStage",
            "MoEExpertOverlayLocalTPExecutor",
            "MoEExpertOverlayCPUFallbackStage",
            "MoEExpertOverlayCPUFallback",
            "MoEOverlayCPUFallbackParticipantRunner",
            "MoEOverlayDispatchCollective",
            "MoEOverlayMPIDispatchBackend",
        };

        std::vector<std::string> failures;
        for (const auto &relative_path : production_paths)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            for (const auto &token : removed_tokens)
            {
                if (contents.find(token) != std::string::npos)
                    failures.push_back(relative_path.generic_string() + " contains removed token " + token);
            }
        }

        EXPECT_TRUE(failures.empty()) << [&]
        {
            std::ostringstream out;
            for (const auto &failure : failures)
                out << failure << '\n';
            return out.str();
        }();
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, MoEStageGlueDoesNotUseBackendRuntimeAPIs)
    {
        const fs::path root = findRepoRoot();
        const std::vector<std::string> forbidden_literals = {
            "#include <cuda",
            "#include \"cuda",
            "#include <hip/",
            "#include \"hip/",
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaMemset",
            "cudaStream",
            "cudaEvent",
            "cudaLaunch",
            "hipMalloc",
            "hipFree",
            "hipMemcpy",
            "hipMemset",
            "hipStream",
            "hipEvent",
            "hipLaunch",
        };

        std::vector<std::string> failures;
        for (const auto &relative_path : moeStageGlueFiles())
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            for (const auto &token : forbidden_literals)
            {
                if (contents.find(token) != std::string::npos)
                    failures.push_back(relative_path.generic_string() + " contains backend runtime token " + token);
            }
        }

        EXPECT_TRUE(failures.empty()) << [&]
        {
            std::ostringstream out;
            for (const auto &failure : failures)
                out << failure << '\n';
            return out.str();
        }();
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalanceCallSitesUseParticipantVocabulary)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> callsite_paths = {
            "src/v2/execution/runner/OrchestrationRunner.h",
            "src/v2/execution/runner/OrchestrationRunner.cpp",
            "src/v2/execution/local_execution/orchestrators/RankOrchestrator.h",
            "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp",
            "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp",
        };

        const std::vector<std::string> forbidden_tokens = {
            "masks_by_socket",
            ".owner_socket\"",
            "computeExpertMasks(socket",
        };

        std::vector<std::string> failures;
        for (const auto &relative_path : callsite_paths)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            for (const auto &token : forbidden_tokens)
            {
                if (contents.find(token) != std::string::npos)
                    failures.push_back(relative_path.generic_string() + " contains old rebalance vocabulary token " + token);
            }
        }

        EXPECT_TRUE(failures.empty()) << [&]
        {
            std::ostringstream out;
            for (const auto &failure : failures)
                out << failure << '\n';
            return out.str();
        }();
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ROCmTensorAwareMoEWrappersMarkDeviceOutputs)
    {
        const fs::path root = findRepoRoot();
        const fs::path path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(path)) << path;
        const std::string contents = readFile(path);
        ASSERT_FALSE(contents.empty()) << path;

        auto functionBody = [&](const std::string &start_marker,
                                const std::string &next_marker) -> std::string
        {
            const size_t start = contents.find(start_marker);
            EXPECT_NE(start, std::string::npos) << start_marker;
            if (start == std::string::npos)
                return {};
            const size_t end = contents.find(next_marker, start + start_marker.size());
            EXPECT_NE(end, std::string::npos) << next_marker;
            if (end == std::string::npos)
                return contents.substr(start);
            return contents.substr(start, end - start);
        };

        const std::string scatter_body = functionBody(
            "void ROCmMoEKernel::scatterAddWeightedFromTensors",
            "void ROCmMoEKernel::sharedExpertGateFromTensors");
        EXPECT_NE(scatter_body.find("output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE"),
                  std::string::npos);

        const std::string gate_add_body = functionBody(
            "void ROCmMoEKernel::sharedExpertGateAddFromTensors",
            "void ROCmMoEKernel::swiGLUFromTensors");
        EXPECT_NE(gate_add_body.find("markDeviceWritten(combined_output"),
                  std::string::npos);

        const std::string weighted_add_body = functionBody(
            "void ROCmMoEKernel::weightedAddFromTensors",
            "int ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable");
        EXPECT_NE(weighted_add_body.find("output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE"),
                  std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, HotCacheRouterStatsSkipStaticPlacements)
    {
        const fs::path root = findRepoRoot();
        const fs::path runtime_header = root / "src/v2/execution/moe/MoERuntimeTable.h";
        const fs::path runtime_source = root / "src/v2/execution/moe/MoERuntimeTable.cpp";
        const fs::path cuda_kernel = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_kernel = root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        ASSERT_TRUE(fs::exists(runtime_header)) << runtime_header;
        ASSERT_TRUE(fs::exists(runtime_source)) << runtime_source;
        ASSERT_TRUE(fs::exists(cuda_kernel)) << cuda_kernel;
        ASSERT_TRUE(fs::exists(rocm_kernel)) << rocm_kernel;

        const std::string header = readFile(runtime_header);
        const std::string runtime = readFile(runtime_source);
        const std::string cuda = readFile(cuda_kernel);
        const std::string rocm = readFile(rocm_kernel);
        ASSERT_FALSE(header.empty()) << runtime_header;
        ASSERT_FALSE(runtime.empty()) << runtime_source;
        ASSERT_FALSE(cuda.empty()) << cuda_kernel;
        ASSERT_FALSE(rocm.empty()) << rocm_kernel;

        EXPECT_NE(header.find("reserved[0]: count of experts resident on more than one participant"),
                  std::string::npos)
            << "The runtime ABI must document the cheap hot-cache stats gate.";
        EXPECT_NE(runtime.find("participantMaskCount(resident_mask) > 1u"), std::string::npos)
            << "Host-published placement banks must mark whether hot-cache stats can ever be useful.";

        auto expectStaticFastSkip = [](const std::string &source, const char *label)
        {
            const size_t helper_pos =
                source.find("runtime_resolve_decode_dispatch");
            ASSERT_NE(helper_pos, std::string::npos) << label;
            const size_t helper_end =
                source.find("int default_load[kDeviceMoEMaxParticipants]", helper_pos);
            ASSERT_NE(helper_end, std::string::npos) << label;
            const std::string prefix = source.substr(helper_pos, helper_end - helper_pos);
            EXPECT_NE(prefix.find("if (!has_multi_resident_experts)"), std::string::npos)
                << label << " must return before hot-cache load simulation when the layer has no replicas.";
            EXPECT_NE(prefix.find("const bool track_balance"), std::string::npos)
                << label << " must keep hot-cache balance accounting behind an explicit gate.";
            EXPECT_NE(prefix.find("runtime_multi_resident_expert_count(bank) != 0u"),
                      std::string::npos)
                << label << " must skip hot-cache balance accounting before top-k/load walks for static placements.";
            EXPECT_NE(source.find("runtime_compute_multi_resident_expert_count"),
                      std::string::npos)
                << label << " must maintain the gate after device-side rebalance applies.";
        };

        expectStaticFastSkip(cuda, "CUDA");
        expectStaticFastSkip(rocm, "ROCm");
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, CUDADecodeRuntimeTopKStaysBlockParallel)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_kernel = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        ASSERT_TRUE(fs::exists(cuda_kernel)) << cuda_kernel;

        const std::string cuda = readFile(cuda_kernel);
        ASSERT_FALSE(cuda.empty()) << cuda_kernel;

        const size_t helper = cuda.find("void moe_select_topk_probabilities_block(");
        ASSERT_NE(helper, std::string::npos)
            << "CUDA decode runtime routing should share the block-wide top-k helper.";

        const size_t begin = cuda.find("__global__ void softmax_topk_decode_runtime_kernel(");
        ASSERT_NE(begin, std::string::npos);
        const size_t end = cuda.find("__global__ void decode_route_select_runtime_kernel(", begin);
        ASSERT_NE(end, std::string::npos);
        const std::string body = cuda.substr(begin, end - begin);

        const size_t topk_call = body.find("moe_select_topk_probabilities_block(");
        ASSERT_NE(topk_call, std::string::npos)
            << "CUDA decode runtime top-k must use all lanes, matching the ROCm wave/block path.";
        const size_t publish = body.find("const bool shape_ok = runtime_shape_ok");
        ASSERT_NE(publish, std::string::npos);
        EXPECT_LT(topk_call, publish)
            << "Runtime placement publication should happen after block-parallel top-k selection.";
        EXPECT_NE(body.find("__shared__ int red_idx[kThreads]"), std::string::npos)
            << "The CUDA decode runtime kernel needs shared reduction indices for block top-k.";
        EXPECT_EQ(body.find("for (int expert = 0; expert < num_experts; ++expert)"),
                  std::string::npos)
            << "Do not reintroduce the serial thread-0 top-k expert walk in CUDA decode routing.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ROCmMoEHelpersSelectTheirOwningDevice)
    {
        const fs::path root = findRepoRoot();
        const fs::path path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(path)) << path;
        const std::string contents = readFile(path);
        ASSERT_FALSE(contents.empty()) << path;

        auto functionBody = [&](const std::string &start_marker,
                                const std::string &next_marker) -> std::string
        {
            const size_t start = contents.find(start_marker);
            EXPECT_NE(start, std::string::npos) << start_marker;
            if (start == std::string::npos)
                return {};
            const size_t end = contents.find(next_marker, start + start_marker.size());
            EXPECT_NE(end, std::string::npos) << next_marker;
            if (end == std::string::npos)
                return contents.substr(start);
            return contents.substr(start, end - start);
        };

        const std::vector<std::pair<std::string, std::string>> guarded_helpers = {
            {"void ROCmMoEKernel::gatherTokenBatch", "void ROCmMoEKernel::scatterAddWeighted"},
            {"void ROCmMoEKernel::scatterAddWeighted", "void ROCmMoEKernel::sharedExpertGate"},
            {"void ROCmMoEKernel::sharedExpertGate", "void ROCmMoEKernel::swiGLU"},
            {"void ROCmMoEKernel::swiGLU", "void ROCmMoEKernel::weightedAdd"},
            {"void ROCmMoEKernel::weightedAdd", "void ROCmMoEKernel::allocateHistogramBuffers"},
            {"bool ROCmMoEKernel::groupTokensByExpertDevice", "bool ROCmMoEKernel::ensureStagingCapacity"},
            {"void ROCmMoEKernel::zeroBuffer", "void ROCmMoEKernel::gatherTokenBatchFromTensors"},
            {"void ROCmMoEKernel::gatherTokenBatchFromTensors", "void ROCmMoEKernel::scatterAddWeightedFromTensors"},
            {"void ROCmMoEKernel::scatterAddWeightedFromTensors", "void ROCmMoEKernel::sharedExpertGateFromTensors"},
            {"void ROCmMoEKernel::sharedExpertGateFromTensors", "void ROCmMoEKernel::sharedExpertGateAddFromTensors"},
            {"void ROCmMoEKernel::sharedExpertGateAddFromTensors", "void ROCmMoEKernel::swiGLUFromTensors"},
            {"void ROCmMoEKernel::swiGLUFromTensors", "void ROCmMoEKernel::weightedAddFromTensors"},
            {"void ROCmMoEKernel::weightedAddFromTensors", "int ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable"},
            {"bool ROCmMoEKernel::groupPrefillRoutes", "bool ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime"},
            {"bool ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime",
             "bool ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime"},
            {"bool ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime",
             "bool ROCmMoEKernel::prepareExpertGroups"},
            {"bool ROCmMoEKernel::prepareExpertGroups", "int ROCmMoEKernel::getExpertTokenCount"},
            {"void ROCmMoEKernel::gatherExpertBatch", "void ROCmMoEKernel::scatterExpertResults"},
            {"void ROCmMoEKernel::scatterExpertResults", "bool ROCmMoEKernel::prepareExpertGroupsAsync"},
            {"bool ROCmMoEKernel::prepareExpertGroupsAsync",
             "bool ROCmMoEKernel::ensureGroupedPrefillScratchCapacity"},
            {"bool ROCmMoEKernel::ensureGroupedPrefillScratchCapacity",
             "bool ROCmMoEKernel::executeGroupedPrefillPipeline"},
            {"bool ROCmMoEKernel::executeGroupedPrefillPipeline", "} // namespace llaminar2"},
        };

        std::vector<std::string> failures;
        for (const auto &[start_marker, next_marker] : guarded_helpers)
        {
            const std::string body = functionBody(start_marker, next_marker);
            if (body.find("setMoEDevice(device_ordinal_") == std::string::npos)
                failures.push_back(start_marker + " does not select device_ordinal_");
        }

        EXPECT_TRUE(failures.empty()) << [&]
        {
            std::ostringstream out;
            for (const auto &failure : failures)
                out << failure << '\n';
            return out.str();
        }();
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GroupedVerifierPrefillSkipsPreZeroForOrderedScatter)
    {
        const fs::path root = findRepoRoot();
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;

        auto functionBody = [](const std::string &contents,
                               const std::string &start_marker,
                               const std::string &next_marker)
        {
            const size_t start = contents.find(start_marker);
            if (start == std::string::npos)
                return std::string{};
            const size_t end = contents.find(next_marker, start + start_marker.size());
            return contents.substr(start, end == std::string::npos ? std::string::npos : end - start);
        };

        const std::string rocm_body = functionBody(
            readFile(rocm_path),
            "bool ROCmMoEKernel::executeGroupedPrefillPipeline(",
            "} // namespace llaminar2");
        const std::string cuda_body = functionBody(
            readFile(cuda_path),
            "bool CUDAMoEKernel::executeGroupedPrefillPipeline(",
            "bool CUDAMoEKernel::groupedExpertGateUpDecodeFromTable(");
        ASSERT_FALSE(rocm_body.empty()) << rocm_path;
        ASSERT_FALSE(cuda_body.empty()) << cuda_path;

        for (const auto &[name, body] : std::vector<std::pair<std::string, std::string>>{
                 {"ROCm", rocm_body},
                 {"CUDA", cuda_body}})
        {
            EXPECT_NE(body.find("const bool ordered_scatter_overwrites_output"),
                      std::string::npos)
                << name << " grouped verifier prefill must explicitly model ordered scatter ownership";
            EXPECT_NE(body.find("active_expert_slots > 0 && d_group_original_to_grouped_ != nullptr"),
                      std::string::npos)
                << name << " ordered scatter validity must not rely on a stale workspace pointer alone";
            EXPECT_EQ(body.find("atomic scatter fallback"), std::string::npos)
                << name << " production grouped prefill must not retain an atomic publication fallback";
        }
        EXPECT_NE(rocm_body.find("if (!ordered_scatter_overwrites_output ||"),
                  std::string::npos)
            << "ROCm must fail hard when active routes lack ordered publication metadata";
        EXPECT_NE(cuda_body.find("active_expert_slots > 0 && !ordered_scatter_overwrites_output"),
                  std::string::npos)
            << "CUDA must fail hard when active routes lack ordered publication metadata";
        EXPECT_NE(cuda_body.find("if (active_expert_slots == 0)"),
                  std::string::npos)
            << "A CUDA participant with no local routes must publish a zero collective contribution";

        const fs::path cuda_kernels_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        ASSERT_TRUE(fs::exists(cuda_kernels_path)) << cuda_kernels_path;
        const std::string shared_kernel = functionBody(
            readFile(cuda_kernels_path),
            "__global__ void prepare_shared_expert_group_kernel(",
            "__global__ void gather_expert_fixed_kernel(");
        ASSERT_FALSE(shared_kernel.empty()) << cuda_kernels_path;
        EXPECT_NE(shared_kernel.find("int *__restrict__ original_to_grouped"),
                  std::string::npos);
        EXPECT_NE(shared_kernel.find("original_to_grouped[idx] = idx;"),
                  std::string::npos)
            << "CUDA shared verifier grouping must publish the same identity map ROCm uses";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, LocalExpertCompactRoutingUsesInvalidPadding)
    {
        const fs::path root = findRepoRoot();
        const fs::path local_path = root / "src/v2/execution/compute_stages/stages/MoELocalExpertStage.cpp";
        const fs::path compute_path = root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        ASSERT_TRUE(fs::exists(local_path)) << local_path;
        ASSERT_TRUE(fs::exists(compute_path)) << compute_path;
        const std::string local_contents = readFile(local_path);
        const std::string compute_contents = readFile(compute_path);
        ASSERT_FALSE(local_contents.empty()) << local_path;
        ASSERT_FALSE(compute_contents.empty()) << compute_path;

        EXPECT_NE(local_contents.find("constexpr int kCompactTopK = 1;"),
                  std::string::npos);
        EXPECT_NE(local_contents.find("std::fill_n(routing_indices, active_routes.size() * static_cast<size_t>(kCompactTopK), -1.0f)"),
                  std::string::npos);
        EXPECT_NE(local_contents.find("compute_params.top_k = kCompactTopK;"),
                  std::string::npos);
        EXPECT_EQ(local_contents.find("compact_output_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE"),
                  std::string::npos);
        EXPECT_NE(compute_contents.find("expert_id < 0 || expert_id >= num_experts"),
                  std::string::npos);
        EXPECT_NE(compute_contents.find("if (weight == 0.0f)"),
                  std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, SparseReturnReduceDeclaresCombinedOutputCoherence)
    {
        const fs::path root = findRepoRoot();
        const fs::path header_path = root / "src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.h";
        const fs::path impl_path = root / "src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.cpp";
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(header_path)) << header_path;
        ASSERT_TRUE(fs::exists(impl_path)) << impl_path;
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        const std::string header_contents = readFile(header_path);
        const std::string impl_contents = readFile(impl_path);
        const std::string graph_contents = readFile(graph_path);
        ASSERT_FALSE(header_contents.empty()) << header_path;
        ASSERT_FALSE(impl_contents.empty()) << impl_path;
        ASSERT_FALSE(graph_contents.empty()) << graph_path;

        EXPECT_NE(header_contents.find("std::optional<BufferId> dense_output_buffer_id;"),
                  std::string::npos);
        EXPECT_NE(header_contents.find("StageBufferContract bufferContract() const override;"),
                  std::string::npos);
        EXPECT_NE(impl_contents.find("StageBufferContract MoESparseReturnReduceStage::bufferContract() const"),
                  std::string::npos);
        EXPECT_NE(impl_contents.find("contract.addOutput(*params_.dense_output_buffer_id);"),
                  std::string::npos);
        EXPECT_NE(impl_contents.find("contract.addInOut(*params_.dense_output_buffer_id);"),
                  std::string::npos);
        EXPECT_NE(graph_contents.find("return_params.dense_output_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);"),
                  std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, LocalExpertPropagatesGpuStreamToNestedExpertCompute)
    {
        const fs::path root = findRepoRoot();
        const fs::path path = root / "src/v2/execution/compute_stages/stages/MoELocalExpertStage.cpp";
        ASSERT_TRUE(fs::exists(path)) << path;
        const std::string contents = readFile(path);
        ASSERT_FALSE(contents.empty()) << path;

        const size_t construct_stage = contents.find("MoEExpertComputeStage compute_stage(std::move(compute_params));");
        ASSERT_NE(construct_stage, std::string::npos);
        const size_t stream_bind = contents.find("compute_stage.setGPUStream(gpuStream());", construct_stage);
        const size_t execute_stage = contents.find("compute_stage.execute(ctx)", construct_stage);
        ASSERT_NE(stream_bind, std::string::npos);
        ASSERT_NE(execute_stage, std::string::npos);
        EXPECT_LT(stream_bind, execute_stage);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, WeightManagerUnpinsMmapWeightsBeforeMadvise)
    {
        const fs::path root = findRepoRoot();
        const fs::path manager_path = root / "src/v2/loaders/WeightManager.cpp";
        const fs::path tensor_path = root / "src/v2/tensors/TensorClasses.h";
        const fs::path slice_path = root / "src/v2/tensors/TensorSlice.h";
        ASSERT_TRUE(fs::exists(manager_path)) << manager_path;
        ASSERT_TRUE(fs::exists(tensor_path)) << tensor_path;
        ASSERT_TRUE(fs::exists(slice_path)) << slice_path;

        const std::string manager_contents = readFile(manager_path);
        const std::string tensor_contents = readFile(tensor_path);
        const std::string slice_contents = readFile(slice_path);
        ASSERT_FALSE(manager_contents.empty()) << manager_path;
        ASSERT_FALSE(tensor_contents.empty()) << tensor_path;
        ASSERT_FALSE(slice_contents.empty()) << slice_path;

        const size_t function_start = manager_contents.find("size_t WeightManager::adviseMmapDontneed()");
        ASSERT_NE(function_start, std::string::npos);
        const size_t release_call = manager_contents.find("releaseMmapHostRegistration()", function_start);
        const size_t madvise_call = manager_contents.find("return loader_.adviseMmapDontneed();", function_start);
        ASSERT_NE(release_call, std::string::npos);
        ASSERT_NE(madvise_call, std::string::npos);
        EXPECT_LT(release_call, madvise_call);

        EXPECT_NE(tensor_contents.find("virtual void releaseMmapHostRegistration()"), std::string::npos);
        EXPECT_NE(tensor_contents.find("if (is_mmap_data())\n                unpinHostMemory();"), std::string::npos);
        EXPECT_NE(slice_contents.find("void releaseMmapHostRegistration() override"), std::string::npos);
        EXPECT_NE(slice_contents.find("wrapped->releaseMmapHostRegistration();"), std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, MmapDontneedIsDeferredUntilAfterFirstPrefill)
    {
        const fs::path root = findRepoRoot();
        const fs::path dgo_path = root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        const fs::path rank_path = root / "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(dgo_path)) << dgo_path;
        ASSERT_TRUE(fs::exists(rank_path)) << rank_path;

        const std::string dgo_contents = readFile(dgo_path);
        const std::string rank_contents = readFile(rank_path);
        ASSERT_FALSE(dgo_contents.empty()) << dgo_path;
        ASSERT_FALSE(rank_contents.empty()) << rank_path;

        const size_t graph_ready_start = dgo_contents.find("void DeviceGraphOrchestrator::onFirstGraphReady()");
        ASSERT_NE(graph_ready_start, std::string::npos);
        const size_t graph_ready_end = dgo_contents.find("void DeviceGraphOrchestrator::adviseMmapDontneedAfterFirstPrefill()",
                                                         graph_ready_start);
        ASSERT_NE(graph_ready_end, std::string::npos);
        const std::string graph_ready_body = dgo_contents.substr(graph_ready_start, graph_ready_end - graph_ready_start);
        EXPECT_EQ(graph_ready_body.find("adviseMmapDontneed()"), std::string::npos);

        const size_t dgo_prefill_start = dgo_contents.find("void DeviceGraphOrchestrator::adviseMmapDontneedAfterFirstPrefill()");
        ASSERT_NE(dgo_prefill_start, std::string::npos);
        const size_t dgo_prefill_sync = dgo_contents.find("synchronizeDeviceBackendBeforeMmapRelease(state_.device_id)", dgo_prefill_start);
        const size_t dgo_prefill_advise = dgo_contents.find("weight_manager_->adviseMmapDontneed()", dgo_prefill_start);
        ASSERT_NE(dgo_prefill_sync, std::string::npos);
        ASSERT_NE(dgo_prefill_advise, std::string::npos);
        EXPECT_LT(dgo_prefill_sync, dgo_prefill_advise);

        const size_t rank_release = rank_contents.find("releaseHostResidentWeightData();");
        const size_t rank_sync = rank_contents.find("synchronizeGpuBackendsBeforeRankMmapRelease(config_)", rank_release);
        const size_t rank_advise = rank_contents.find("wm->adviseMmapDontneed()", rank_release);
        ASSERT_NE(rank_release, std::string::npos);
        ASSERT_NE(rank_sync, std::string::npos);
        ASSERT_NE(rank_advise, std::string::npos);
        EXPECT_LT(rank_release, rank_sync);
        EXPECT_LT(rank_sync, rank_advise);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DGOHostReleaseWaitsForMTPShiftedPrefill)
    {
        const fs::path root = findRepoRoot();
        const fs::path dgo_path = root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(dgo_path)) << dgo_path;

        const std::string dgo_contents = readFile(dgo_path);
        ASSERT_FALSE(dgo_contents.empty()) << dgo_path;

        const size_t forward_start = dgo_contents.find("const float *DeviceGraphOrchestrator::forward(");
        ASSERT_NE(forward_start, std::string::npos);
        const size_t forward_end = dgo_contents.find("bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule",
                                                     forward_start);
        ASSERT_NE(forward_end, std::string::npos);
        const std::string forward_body = dgo_contents.substr(forward_start, forward_end - forward_start);

        const size_t forward_mtp = forward_body.find("populateMTPShiftedCacheFromPrefill(");
        const size_t forward_terminal = forward_body.find("noteMainForwardHiddenProducedForMTP(");
        const size_t forward_release = forward_body.find("releaseHostResidentWeightData();");
        ASSERT_NE(forward_mtp, std::string::npos);
        ASSERT_NE(forward_terminal, std::string::npos);
        ASSERT_NE(forward_release, std::string::npos);
        EXPECT_LT(forward_mtp, forward_release);
        EXPECT_LT(forward_terminal, forward_release);

        const size_t chunk_start = dgo_contents.find("bool DeviceGraphOrchestrator::forwardPrefillChunkSchedule(");
        ASSERT_NE(chunk_start, std::string::npos);
        const size_t chunk_end = dgo_contents.find("bool DeviceGraphOrchestrator::ensureMTPTerminalHiddenBuffer",
                                                   chunk_start);
        ASSERT_NE(chunk_end, std::string::npos);
        const std::string chunk_body = dgo_contents.substr(chunk_start, chunk_end - chunk_start);

        const size_t chunk_mtp = chunk_body.find("populateMTPShiftedCacheFromPrefill(");
        const size_t chunk_terminal = chunk_body.find("noteMainForwardHiddenProducedForMTP(terminal_seq_len, 1)");
        const size_t chunk_release = chunk_body.find("releaseHostResidentWeightData();");
        ASSERT_NE(chunk_mtp, std::string::npos);
        ASSERT_NE(chunk_terminal, std::string::npos);
        ASSERT_NE(chunk_release, std::string::npos);
        EXPECT_LT(chunk_mtp, chunk_release);
        EXPECT_LT(chunk_terminal, chunk_release);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, MTPSidecarMoERuntimeTableIsSeparateFromMainHistogram)
    {
        const fs::path root = findRepoRoot();
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        const std::string contents = readFile(graph_path);
        ASSERT_FALSE(contents.empty()) << graph_path;

        const size_t ffn_start = contents.find("ComputeGraph Qwen35MoEGraph::buildFFNGraph(");
        ASSERT_NE(ffn_start, std::string::npos);
        const size_t ffn_end = contents.find("// =====================================================================",
                                             contents.find("Stage 3: MoE Expert Compute", ffn_start));
        ASSERT_NE(ffn_end, std::string::npos);
        const std::string ffn_body = contents.substr(ffn_start, ffn_end - ffn_start);

        EXPECT_NE(ffn_body.find("use_mtp_runtime_table = mtp_sidecar_context"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("std::max(config_.n_layers, layer_idx + 1)"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("\"mtp_depth\" + std::to_string(mtp_depth_idx)"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("register_runtime_histogram = !use_mtp_runtime_table"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("route_params.decode_histogram = mtp_sidecar_context ? nullptr : config_.moe.decode_histogram"),
                  std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceRebalanceStageSplitsCopyAndApplyForOverlap)
    {
        const fs::path root = findRepoRoot();
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        const std::string contents = readFile(graph_path);
        ASSERT_FALSE(contents.empty()) << graph_path;

        const size_t ffn_start = contents.find("ComputeGraph Qwen35MoEGraph::buildFFNGraph(");
        ASSERT_NE(ffn_start, std::string::npos);
        const size_t set_terminal = contents.find("graph.setTerminalNode(ffn_terminal)", ffn_start);
        ASSERT_NE(set_terminal, std::string::npos);
        const std::string ffn_body = contents.substr(ffn_start, set_terminal - ffn_start);

        EXPECT_NE(ffn_body.find("first_local_decode_layer"), std::string::npos);
        EXPECT_NE(ffn_body.find("last_local_decode_layer"), std::string::npos);
        EXPECT_NE(ffn_body.find("local_decode_layer"), std::string::npos);
        EXPECT_NE(ffn_body.find("total_tokens == 1"), std::string::npos);
        EXPECT_NE(ffn_body.find("layer_idx == config_.pp_layer_offset"), std::string::npos);
        EXPECT_NE(ffn_body.find("!config_.compute_all_position_logits"), std::string::npos)
            << "Graph-side rebalance must not run in all-position prefill/verifier graphs.";
        EXPECT_NE(ffn_body.find("device_side_graph_rebalance_candidate ="), std::string::npos);
        EXPECT_NE(ffn_body.find("local_decode_layer &&"), std::string::npos)
            << "Device-side graph rebalance must own every decode layer so later layers do not re-register host sync.";
        EXPECT_NE(ffn_body.find("env.moe_rebalance.device_rebalance_graph_controller &&"),
                  std::string::npos)
            << "Graph-stable placement and the graph-native controller must be separate policy gates.";
        EXPECT_NE(ffn_body.find("env.moe_rebalance.device_rebalance_collect_load_stats"),
                  std::string::npos)
            << "Projected load-spread diagnostics must be opt-in from DebugEnv.";
        EXPECT_NE(ffn_body.find("PerfStatsCollector::isEnabled()"),
                  std::string::npos)
            << "Perfstats exports must automatically enable projected load-spread diagnostics.";
        EXPECT_NE(ffn_body.find("DeviceMoERebalanceFlags::CollectLoadStats"),
                  std::string::npos)
            << "Load-spread diagnostics must flow into the device-controller config.";
        EXPECT_NE(ffn_body.find("shouldCollectGraphRebalanceTransferSpecs"), std::string::npos);
        EXPECT_NE(ffn_body.find("device_side_graph_rebalance_candidate"), std::string::npos);
        EXPECT_NE(ffn_body.find("register_runtime_histogram_for_decode"), std::string::npos)
            << "Device-side graph rebalance must not register host decode-histogram sync callbacks.";
        EXPECT_NE(ffn_body.find("!device_side_graph_rebalance_candidate"), std::string::npos);
        EXPECT_NE(ffn_body.find("config_.moe.rebalance_mode == MoERebalanceMode::DYNAMIC"), std::string::npos);
        EXPECT_EQ(ffn_body.find("hot_replica_cap > 0"), std::string::npos)
            << "Homogeneous GPU dynamic rebalance must not fall back to host publish/apply just because hot cache is off.";
        EXPECT_EQ(ffn_body.find("env.moe_rebalance.gpu_cache_experts_per_layer <= 0"), std::string::npos)
            << "Homogeneous GPU dynamic rebalance must not leave hot-cache decode on the host path.";
        EXPECT_NE(ffn_body.find("isHomogeneousGpuLocalTPRebalanceDomain"), std::string::npos);
        EXPECT_EQ(contents.find("#include \"../../execution/moe/GPUExpertTransfer.h\""),
                  std::string::npos);
        EXPECT_NE(contents.find("selectGraphRebalanceTransferMode("), std::string::npos);
        EXPECT_NE(contents.find("max_hot_replicas_per_participant == 0"), std::string::npos)
            << "No-hot device-side dynamic mode must be resident-only and avoid transfer-slot allocation.";
        EXPECT_NE(contents.find("DeviceMoERebalanceTransferMode::ResidentOnly"), std::string::npos);
        EXPECT_EQ(contents.find("GPUExpertTransfer::canAccessPeer(destination_device, source_device)"),
                  std::string::npos)
            << "Graph-side rebalance should use NCCL/RCCL sidebands instead of an untested direct peer-read branch.";
        EXPECT_EQ(contents.find("GPUExpertTransfer::enablePeerAccess(destination_device, source_device)"),
                  std::string::npos)
            << "Peer access setup is obsolete for graph-side rebalance; NCCL/RCCL own transport selection.";
        EXPECT_NE(contents.find("fixed-size collective payload arenas move empty expert slots"),
                  std::string::npos)
            << "Measured fixed-arena transfer moved mostly empty slots; graph-side rebalance must refuse it.";
        EXPECT_NE(contents.find("Use CompactTransferSlots async maintenance for non-empty transfer-slot arrivals"),
                  std::string::npos)
            << "The production path must use compact non-empty transfer-slot arrivals.";
        EXPECT_NE(ffn_body.find("graphRebalanceDecodeUsesMutableDescriptors"),
                  std::string::npos)
            << "Descriptor mutability must be tied to the rebalance transfer mode, not to dynamic rebalance itself.";
        const size_t mutable_policy_start =
            ffn_body.find("auto graphRebalanceDecodeUsesMutableDescriptors");
        ASSERT_NE(mutable_policy_start, std::string::npos);
        const size_t mutable_policy_end =
            ffn_body.find("auto ensureGraphRebalanceTransferMode", mutable_policy_start);
        ASSERT_NE(mutable_policy_end, std::string::npos);
        const std::string mutable_policy_body =
            ffn_body.substr(mutable_policy_start, mutable_policy_end - mutable_policy_start);
        EXPECT_NE(ffn_body.find("expert_params.runtime_decode_uses_mutable_descriptors =\n                    graphRebalanceDecodeUsesMutableDescriptors()"),
                  std::string::npos)
            << "ResidentOnly graph rebalance should keep using immutable grouped descriptor tables.";
        EXPECT_NE(ffn_body.find("ResidentOnly changes runtime top-k/local-compute masks"),
                  std::string::npos)
            << "The graph must document why ResidentOnly rebalance can use static descriptor tables.";
        EXPECT_NE(ffn_body.find("deviceMoERebalanceModeUsesTransferSlots(*graph_rebalance_transfer_mode)"),
                  std::string::npos)
            << "Any transfer-slot mode can publish new descriptors and must require mutable runtime descriptors.";
        EXPECT_NE(ffn_body.find("graphRebalanceEnsureTransferMode"), std::string::npos)
            << "Transfer-slot descriptor mutability must force transfer-mode selection before expert params are built.";
        EXPECT_NE(mutable_policy_body.find("return deviceMoERebalanceModeUsesTransferSlots(*graph_rebalance_transfer_mode);"),
                  std::string::npos)
            << "Once mode selection succeeds, mutable descriptor policy must not silently fall through to false.";
        EXPECT_EQ(mutable_policy_body.find("return graph_rebalance_transfer_mode.has_value() &&\n                   graphRebalanceMovesFixedPayloadCapacity(*graph_rebalance_transfer_mode);"),
                  std::string::npos)
            << "A no-mode fallback made payload-transfer decode tables immutable because mode selection happened later.";
        EXPECT_NE(ffn_body.find("masked_local_tp_apportioned_decode_runtime_table"),
                  std::string::npos)
            << "Plain homogeneous LocalTP apportioned decode must build masked runtime tables for graph-side rebalance.";
        EXPECT_NE(ffn_body.find("\"LocalTP expert-ID-apportioned masked GPU decode graph build\""),
                  std::string::npos)
            << "The standard LocalTP path must initialize the same masked runtime-table contract as overlay.";
        EXPECT_NE(contents.find("contiguousApportionedExpertOwners"),
                  std::string::npos)
            << "Plain LocalTP masked runtime tables must publish owner/resident metadata for every expert.";
        EXPECT_NE(contents.find("ownerParticipantsFromMap"),
                  std::string::npos)
            << "Overlay masked runtime tables must publish owner/resident metadata from the graph-native owner map.";
        EXPECT_NE(ffn_body.find("expert_params.my_socket_id = std::max(0, config_.tp_device_idx)"),
                  std::string::npos)
            << "Standard LocalTP expert stages must not default every participant to id 0.";
        EXPECT_NE(ffn_body.find("expert_params.participant_count =\n                    local_tp_ctx && local_tp_ctx->degree() > 0"),
                  std::string::npos)
            << "Expert stages must carry the full LocalTP domain size for variable 2+ card domains.";
        EXPECT_NE(ffn_body.find("const auto owner_participants =\n                            contiguousApportionedExpertOwners"),
                  std::string::npos)
            << "Standard LocalTP graph-side rebalance must know remote source owners for missing-arrival planning.";
        EXPECT_NE(contents.find("update.resident_participant_mask.assign"),
                  std::string::npos)
            << "Masked runtime tables must set resident masks for non-local owner experts, not only local payloads.";
        EXPECT_NE(ffn_body.find("graph_rebalance_producer_node_name"), std::string::npos)
            << "Graph-side rebalance collect-state must anchor to the path-specific expert producer.";
        EXPECT_NE(ffn_body.find("maybeInsertGraphSideRebalance(\n                        \"standard routed expert path\",\n                        prefix + \"moe_expert_ffn\")"),
                  std::string::npos)
            << "Plain LocalTP rebalance must anchor collect-state to the standard expert node.";
        EXPECT_EQ(ffn_body.find("graph.addDependency(graph_rebalance_collect_node,\n                                        prefix + \"moe_expert_ffn_overlay_fast\")"),
                  std::string::npos)
            << "A hardcoded overlay producer dependency breaks plain LocalTP apportioned expert rebalance.";
        EXPECT_NE(contents.find("supportsDeviceSideGraphRebalanceTransfer"), std::string::npos)
            << "The homogeneous-domain gate must be phrased in terms of the selected graph-side transfer path.";
        EXPECT_NE(contents.find("moe_env.device_rebalance_maintenance_graph"),
                  std::string::npos)
            << "Async maintenance must have an explicit transfer-mode branch.";
        EXPECT_NE(contents.find("!tp_ctx.supportsRawAllgatherOnStreamGraphCapture())"),
                  std::string::npos)
            << "Async maintenance must fail closed unless raw NCCL/RCCL allgather is graph-capturable on the maintenance stream.";
        const size_t selector_start =
            contents.find("selectGraphRebalanceTransferMode(");
        ASSERT_NE(selector_start, std::string::npos);
        const size_t selector_end =
            contents.find("int gpuOrdinalForGraphDevice", selector_start);
        ASSERT_NE(selector_end, std::string::npos);
        const std::string selector_body =
            contents.substr(selector_start, selector_end - selector_start);
        EXPECT_EQ(selector_body.find("DeviceMoERebalanceTransferMode::CollectiveSidebandPayload"),
                  std::string::npos)
            << "The graph-side transfer selector must not choose fixed-size payload arenas.";
        EXPECT_EQ(selector_body.find("DeviceMoERebalanceTransferMode::LegacyCollectiveAllGather"),
                  std::string::npos)
            << "The graph-side transfer selector must not choose legacy fixed-size allgather payload transfer.";
        EXPECT_NE(selector_body.find("DeviceMoERebalanceTransferMode::CompactTransferSlots"),
                  std::string::npos)
            << "The graph-side transfer selector should use compact transfer slots for same-backend maintenance.";
        EXPECT_NE(selector_body.find("moe_env.device_rebalance_payload_sideband"),
                  std::string::npos)
            << "Payload-sideband opt-in must be rejected explicitly.";
        EXPECT_NE(selector_body.find("moe_env.allow_legacy_collective_rebalance_transfer"),
                  std::string::npos)
            << "Legacy payload opt-in must be rejected explicitly.";
        EXPECT_NE(contents.find("graph-captured"),
                  std::string::npos);
        EXPECT_NE(contents.find("LLAMINAR_MOE_DEVICE_REBALANCE_MAINTENANCE_GRAPH=1"),
                  std::string::npos)
            << "Missing graph-side transport must direct production rebalance to the async maintenance lane.";
        EXPECT_NE(contents.find("async rolling-wave maintenance lane"),
                  std::string::npos)
            << "Histogram movement must be described as maintenance-wave traffic, not decode-side sideband traffic.";
        EXPECT_EQ(contents.find("command/payload sideband pipeline is under development"),
                  std::string::npos)
            << "The collective fallback should no longer describe command/payload sidebands as unfinished.";
        EXPECT_NE(contents.find("refusing to fall back to host publish/apply in device-side mode"),
                  std::string::npos)
            << "Missing peer access must fail closed instead of reintroducing host publish/apply.";
        EXPECT_NE(ffn_body.find("ensureGraphRebalanceTransferMode"), std::string::npos)
            << "Mode selection must not be hidden behind payload-transfer spec collection.";
        EXPECT_NE(ffn_body.find("graphRebalanceFixedPayloadTransferEnabled"), std::string::npos)
            << "Fixed-capacity transport must be an explicit mode check, not implicit in device-side rebalance.";
        EXPECT_NE(ffn_body.find("graphRebalanceUsesTransferSlots"), std::string::npos)
            << "Transfer-slot allocation must be an explicit mode check shared by compact and fixed diagnostic modes.";
        EXPECT_NE(ffn_body.find("DeviceMoERebalanceFlags::PlanMissingArrivals"), std::string::npos)
            << "Compact transfer slots should produce transfer commands for non-resident hot replicas.";
        EXPECT_NE(ffn_body.find("layer_window_start"), std::string::npos)
            << "Graph-side rebalance must plan a future-layer rolling wave for overlap.";
        EXPECT_NE(ffn_body.find("layer_window_count"), std::string::npos);
        EXPECT_NE(ffn_body.find("layer_wave_count"), std::string::npos)
            << "A captured maintenance replay must cap how many future layers it plans.";
        EXPECT_NE(ffn_body.find("env.moe_rebalance.device_rebalance_layer_wave_count"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("moe_graph_rebalance_bindings_"), std::string::npos)
            << "Route-boundary apply must consume the first layer's device command-buffer workspace.";
        EXPECT_NE(ffn_body.find("GraphSideRebalanceBinding{"), std::string::npos);
        EXPECT_NE(ffn_body.find("graphRebalanceCollectiveKey"), std::string::npos)
            << "Device-side rebalance collectives need a domain-stable label shared by every participant.";
        EXPECT_NE(ffn_body.find("\"moe_device_rebalance_\" + graphRebalanceCollectiveKey()"), std::string::npos)
            << "Collective workspace names must not include participant-local device ids.";
        EXPECT_EQ(ffn_body.find("\"moe_device_rebalance_\" + graphRebalanceDomainKey()"), std::string::npos)
            << "Using the participant key as the collective workspace label makes LocalTP allgather contracts diverge.";
        EXPECT_NE(ffn_body.find("deviceMoERebalanceModePlansMissingArrivals"),
                  std::string::npos)
            << "Arrival planning must be tied to transfer-slot capable modes.";
        EXPECT_NE(ffn_body.find("DeviceMoETransferSlotDirectory::create"), std::string::npos)
            << "Compact arrivals need stable device transfer slots.";
        EXPECT_NE(ffn_body.find("env.moe_rebalance.gpu_direct_transfer_wave_experts"), std::string::npos)
            << "Transfer slots must be bounded by the rolling-wave capacity, not the full hot-cache cap.";
        EXPECT_NE(ffn_body.find("sizeof(DeviceMoEExpertDirectoryEntry)"), std::string::npos)
            << "Legacy collective payload slots must carry the source descriptor needed by unpack.";
        EXPECT_NE(ffn_body.find("moe_rebalance_transfer_states_"), std::string::npos)
            << "The legacy split copy/apply path must share one transfer stream/event state.";
        EXPECT_NE(contents.find("ComputeGraph Qwen35MoEGraph::buildDeviceMoERebalanceMaintenanceGraph"),
                  std::string::npos)
            << "The maintenance graph hook owns rolling-window device rebalance replay when enabled.";
        EXPECT_EQ(contents.find("moe_device_rebalance_maintenance_apply"),
                  std::string::npos)
            << "Maintenance replay should not publish runtime placement; route-boundary apply owns ready-wave publication.";
        EXPECT_EQ(contents.find("moe_device_rebalance_maintenance_apply_drain"),
                  std::string::npos)
            << "The old maintenance drain stage reintroduces apply work on the async lane.";
        EXPECT_NE(contents.find("moe_device_rebalance_maintenance_collect"),
                  std::string::npos)
            << "Maintenance replay must snapshot histograms before planning the next command buffer.";
        EXPECT_NE(contents.find("DeviceMoERebalanceStagePhase::CollectAndGatherState"),
                  std::string::npos);
        EXPECT_NE(contents.find("moe_device_rebalance_maintenance_probe_after_snapshot"),
                  std::string::npos)
            << "Maintenance probe planning should consume the pre-apply histogram snapshot.";
        EXPECT_NE(contents.find("moe_device_rebalance_maintenance_metadata_payload"),
                  std::string::npos)
            << "Maintenance command metadata and payload movement should be a separate captured graph body.";
        EXPECT_NE(contents.find("graph.addDependency(plan_params.stage_name, collect_params.stage_name)"),
                  std::string::npos)
            << "The maintenance probe graph must gather the histogram window before planning commands.";
        EXPECT_EQ(contents.find("graph.addDependency(apply_params.stage_name, collect_params.stage_name)"),
                  std::string::npos);
        EXPECT_EQ(contents.find("graph.addDependency(apply_drain_params.stage_name, apply_params.stage_name)"),
                  std::string::npos);
        EXPECT_EQ(contents.find("graph.addDependency(plan_params.stage_name, apply_drain_params.stage_name)"),
                  std::string::npos);
        EXPECT_NE(contents.find("DeviceMoERebalanceStagePhase::PlanProbeAfterSideband"),
                  std::string::npos)
            << "Maintenance probe planning should reuse the histogram snapshot gathered before apply.";
        EXPECT_NE(contents.find("DeviceMoERebalanceStagePhase::GatherCommandsAndCopyPreparedPayload"),
                  std::string::npos)
            << "Command metadata and payload buckets must live in a separate graph so no-work waves skip them.";
        EXPECT_NE(ffn_body.find("producer_runs_in_maintenance_graph"),
                  std::string::npos)
            << "The graph builder must keep rolling maintenance ownership explicit instead of hiding host scheduling.";
        EXPECT_NE(ffn_body.find("env.moe_rebalance.device_rebalance_maintenance_graph"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("graph_rebalance_can_have_allreduce_anchor"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("!producer_runs_in_maintenance_graph"),
                  std::string::npos)
            << "Maintenance mode must not attach histogram sidebands to steady decode collectives.";
        EXPECT_NE(ffn_body.find("graph_rebalance_transfer_mode.value() ==\n                        DeviceMoERebalanceTransferMode::CollectiveSidebandPayload"),
                  std::string::npos)
            << "Decode-side histogram sidebands may be wired only behind the now-refused fixed-payload mode.";
        EXPECT_NE(ffn_body.find("makeGraphRebalanceStateSidebands"),
                  std::string::npos)
            << "The diagnostic sideband path remains source-visible for explicit transfer experiments.";
        EXPECT_NE(ffn_body.find("takeGraphRebalanceStateSidebands"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("state_sideband_enabled = true"),
                  std::string::npos)
            << "The maintenance graph may skip standalone state gather only after a decode allreduce consumed sidebands.";
        EXPECT_NE(ffn_body.find("TPAllreduceSidebandWorkspaceBinding"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("LocalTPCollectiveSidebandKind::Allgather"),
                  std::string::npos)
            << "Histogram sidebands and the legacy payload path use gathers, not activation allreduces.";
        EXPECT_NE(ffn_body.find("moe_rebalance_histogram_sideband"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("std::max<uint32_t>(1u, wave_count)"),
                  std::string::npos)
            << "Even diagnostic histogram sidebands must be sized by the rolling wave, not the full model.";
        EXPECT_EQ(ffn_body.find("moe_rebalance_directory_sideband"),
                  std::string::npos)
            << "Collective payload transfer should not allgather peer descriptor directories.";
        EXPECT_NE(ffn_body.find("takeGraphRebalanceTransferSidebands"),
                  std::string::npos)
            << "Legacy payload command and payload allgathers must stay isolated behind the transfer-mode gate.";
        EXPECT_NE(ffn_body.find("takeGraphRebalanceSidebandsForAllreduce"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("appendRebalanceSidebands(sidebands, takeGraphRebalanceTransferSidebands())"),
                  std::string::npos)
            << "The legacy helper should be wired only after the compact histogram sideband path.";
        EXPECT_NE(ffn_body.find("binding.transfer_mode != DeviceMoERebalanceTransferMode::CollectiveSidebandPayload"),
                  std::string::npos)
            << "Payload sidebands must be enabled only for the graph-captured sideband payload mode.";
        EXPECT_NE(ffn_body.find("graph_rebalance_transfer_command_sideband_taken"),
                  std::string::npos)
            << "Command/header and payload sidebands need independent once-per-layer guards.";
        EXPECT_NE(ffn_body.find("graph_rebalance_transfer_payload_sideband_taken"),
                  std::string::npos);
        EXPECT_EQ(ffn_body.find("graph_rebalance_transfer_sideband_taken"),
                  std::string::npos)
            << "A single transfer-sideband consumed flag drops the payload sideband after command/header allgather.";
        EXPECT_NE(ffn_body.find("moe_rebalance_transfer_plan_sideband"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("moe_rebalance_command_header_sideband"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("moe_rebalance_transfer_payload_sideband"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("maybeAddGraphRebalancePayloadStageAfterSideband"),
                  std::string::npos)
            << "Legacy sideband payload pack/unpack stages must be ordered after their allreduce anchors.";
        EXPECT_NE(ffn_body.find("PackCollectivePayloadAfterSideband"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("UnpackCollectivePayloadAfterSideband"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("graph.addDependency(graph_rebalance_pack_payload_node, anchor_name)"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("graph.addDependency(graph_rebalance_unpack_payload_node, anchor_name)"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("supportsCollectiveSidebandOnStreamGraphCapture"),
                  std::string::npos)
            << "Sideband mode must require graph-capturable backend support.";
        EXPECT_NE(ffn_body.find("if (!producer_runs_in_maintenance_graph)"),
                  std::string::npos)
            << "The default decode graph must skip the heavy rebalance producer.";
        EXPECT_NE(ffn_body.find("plan_params.local_transfer_slots = local_transfer_slots"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("plan_params.local_transfer_slot_count = local_transfer_slot_count"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("collect_params.phase = DeviceMoERebalanceStagePhase::CollectState"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("plan_params.phase = DeviceMoERebalanceStagePhase::PlanAndCopy"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband"),
                  std::string::npos);
        EXPECT_NE(ffn_body.find("plan_params.transfer_state = transfer_state"), std::string::npos);
        EXPECT_NE(ffn_body.find("plan_params.join_transfer_stream_after_copy"), std::string::npos)
            << "Legacy inline decode producers should fork transfer work and let a late graph node rejoin it.";
        EXPECT_NE(ffn_body.find("producer_runs_in_maintenance_graph"), std::string::npos);
        EXPECT_NE(ffn_body.find("ffn_terminal = graph_rebalance_plan_after_sideband_node"),
                  std::string::npos)
            << "The first-layer producer must be ordered before future layers, even though transfer completion is deferred.";
        EXPECT_NE(ffn_body.find("deviceMoERebalanceModeUsesTransferSlots(binding_it->second.transfer_mode) &&"),
                  std::string::npos)
            << "Resident-only rebalance must skip transfer apply/join nodes.";
        EXPECT_NE(ffn_body.find("apply_params.phase = DeviceMoERebalanceStagePhase::Apply"),
                  std::string::npos);
        EXPECT_EQ(ffn_body.find("apply_runs_in_maintenance_graph"),
                  std::string::npos)
            << "The heavy producer may run in the rolling maintenance graph without adding a separate apply-run mode.";
        EXPECT_NE(ffn_body.find("device_rebalance_decode_apply_poll"),
                  std::string::npos)
            << "Per-token decode apply polling must be an explicit opt-in when async maintenance owns ready-wave apply.";
        EXPECT_NE(ffn_body.find("route kernel"),
                  std::string::npos)
            << "The default maintenance path should piggyback ready-wave apply on routing instead of launching a decode apply stage.";
        EXPECT_NE(ffn_body.find("return {};"),
                  std::string::npos)
            << "Maintenance-graph mode must skip standalone decode apply-stage insertion.";
        EXPECT_NE(ffn_body.find("route_params.device_rebalance_route_apply = true"),
                  std::string::npos)
            << "Routing must receive the device-side rebalance binding so ready-wave apply can piggyback on the route kernel.";
        EXPECT_NE(ffn_body.find("boundary_apply"), std::string::npos)
            << "Inline/deferred decode apply should remain boundary-scoped when maintenance replay is disabled.";
        EXPECT_NE(ffn_body.find("apply_params.apply_layer_idx = boundary_apply ? -1 : layer_idx"),
                  std::string::npos)
            << "Payload and resident-only maintenance should apply ready waves at the routed-expert boundary.";
        EXPECT_NE(ffn_body.find("graph.addDependency(plan_node, prefix + \"ffn_norm\")"),
                  std::string::npos)
            << "The plan/copy phase must enqueue after FFN norm and before routed expert compute starts.";
        EXPECT_NE(ffn_body.find("graph.addDependency(graph_rebalance_collect_node"),
                  std::string::npos)
            << "Piggybacked collection must run after routed expert compute updates runtime histograms.";
        EXPECT_NE(ffn_body.find("DeviceMoERebalanceStagePhase::JoinTransfer"),
                  std::string::npos)
            << "Legacy split producer transfer work must rejoin late in the decode graph, not immediately after copy.";
        EXPECT_NE(ffn_body.find("moe_device_rebalance_transfer_join"), std::string::npos);
        EXPECT_NE(ffn_body.find("graph.addDependency(join_node, ffn_terminal)"), std::string::npos)
            << "The transfer join should happen after the local layer compute path has had a chance to overlap transfer.";
        EXPECT_NE(ffn_body.find("ffn_terminal = join_node"), std::string::npos);
        EXPECT_NE(ffn_body.find("graph.addDependency(ar_name, graph_rebalance_collect_node)"),
                  std::string::npos)
            << "Deferred combined allreduce anchors must also wait for packed sideband state.";
        EXPECT_NE(ffn_body.find("graph.addDependency(\n                                    graph_rebalance_plan_after_sideband_node"),
                  std::string::npos)
            << "Plan/copy-after-sideband must wait for the allreduce anchor to finish gathering state.";
        EXPECT_EQ(ffn_body.find("graph.addDependency(prefix + \"ffn_norm\", plan_node)"),
                  std::string::npos)
            << "Reversing this edge makes FFN norm depend on the rebalance producer.";
        EXPECT_NE(ffn_body.find("graph.addDependency(graph_rebalance_apply_node, prefix + \"ffn_norm\")"),
                  std::string::npos)
            << "The apply phase must run after FFN norm so it can be ordered before routing.";
        EXPECT_NE(ffn_body.find("graph.addDependency(prefix + \"moe_routing\", graph_rebalance_apply_node)"),
                  std::string::npos)
            << "Ready-wave apply must run before MoE routing, otherwise the router cannot account for hot replicas.";
        EXPECT_EQ(ffn_body.find("graph.addDependency(graph_rebalance_apply_node, prefix + \"moe_routing\")"),
                  std::string::npos)
            << "Applying after routing makes the hot-cache dispatch counters a false no-op.";
        EXPECT_NE(ffn_body.find("needs_deferred_resident_apply || needs_transfer_slot_apply"),
                  std::string::npos)
            << "Transfer-slot rebalance should use the same boundary apply as resident-only rebalance, "
               "not one poll kernel per layer.";
        EXPECT_NE(ffn_body.find("boundary_apply && !last_local_decode_layer"),
                  std::string::npos)
            << "Boundary apply must be inserted once at the last local decode layer so async transfers "
               "can overlap with routed compute before polling ready waves.";
        EXPECT_NE(ffn_body.find("rebalance_apply_dependency.empty()"),
                  std::string::npos)
            << "Routed expert stages must depend on split apply when rebalance is active.";
        EXPECT_EQ(ffn_body.find("ffn_terminal = rebalance_node"), std::string::npos)
            << "Graph-side rebalance must not remain a terminal serialized tail stage.";

        const fs::path stage_header_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.h";
        const fs::path stage_source_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.cpp";
        const fs::path controller_header_path =
            root / "src/v2/execution/moe/DeviceMoERebalanceController.h";
        ASSERT_TRUE(fs::exists(stage_header_path)) << stage_header_path;
        ASSERT_TRUE(fs::exists(stage_source_path)) << stage_source_path;
        ASSERT_TRUE(fs::exists(controller_header_path)) << controller_header_path;
        const std::string stage_header = readFile(stage_header_path);
        const std::string stage_source = readFile(stage_source_path);
        const std::string controller_header = readFile(controller_header_path);
        ASSERT_FALSE(stage_header.empty()) << stage_header_path;
        ASSERT_FALSE(stage_source.empty()) << stage_source_path;
        ASSERT_FALSE(controller_header.empty()) << controller_header_path;
        EXPECT_NE(controller_header.find("enum class DeviceMoERebalancePipelinePhase"),
                  std::string::npos);
        EXPECT_NE(controller_header.find("CollectState"), std::string::npos);
        EXPECT_NE(controller_header.find("PlanAssignments"), std::string::npos);
        EXPECT_NE(controller_header.find("StageArrivals"), std::string::npos);
        EXPECT_NE(controller_header.find("ApplyLayer"), std::string::npos);
        EXPECT_NE(controller_header.find("DeferRuntimeApply"), std::string::npos)
            << "Async maintenance must plan on its stream and defer runtime-table mutation to decode apply.";
        EXPECT_NE(controller_header.find("CollectLoadStats"), std::string::npos)
            << "Projected load-spread diagnostics must be an explicit controller flag.";
        EXPECT_NE(controller_header.find("ResidentExpertAssignment"), std::string::npos)
            << "Resident-only maintenance needs a command op that applies local replicas without transfer slots.";
        EXPECT_NE(controller_header.find("DeviceMoERebalanceCommandBufferHeader"),
                  std::string::npos)
            << "Device-side publish/apply must be modeled as a graph-captured command buffer.";
        EXPECT_NE(controller_header.find("DeviceMoERebalanceWaveState"),
                  std::string::npos)
            << "Rolling rebalance waves need explicit device-resident progress state.";
        EXPECT_NE(controller_header.find("layer_wave_count"), std::string::npos)
            << "The shared device-controller ABI needs a bounded per-replay layer wave.";
        EXPECT_NE(controller_header.find("window_index < layer_wave_count"),
                  std::string::npos)
            << "The policy must apply only the bounded wave, not the whole layer window.";
        EXPECT_NE(controller_header.find("enum class DeviceMoERebalanceWaveLifecycle"),
                  std::string::npos)
            << "Graph-captured maintenance rebalance needs explicit device-side wave states.";
        EXPECT_NE(controller_header.find("DeviceMoERebalanceWaveProgress"),
                  std::string::npos);
        EXPECT_NE(controller_header.find("requested_payload_slots"), std::string::npos)
            << "The maintenance scheduler must publish device-selected payload bucket inputs.";
        EXPECT_NE(controller_header.find("payload_bucket_slots"), std::string::npos)
            << "Bucketed transfer sizing must be represented in device-resident wave state.";
        EXPECT_NE(controller_header.find("payload_bucket_index"), std::string::npos)
            << "CUDA conditional graph switch bodies need a device-published bucket index.";
        EXPECT_NE(controller_header.find("payloadBucketSlots"), std::string::npos)
            << "CUDA, ROCm, and CPU mirrors must share one bucket rounding policy.";
        EXPECT_NE(controller_header.find("DeviceMoERebalanceGraphControllerState"),
                  std::string::npos)
            << "Maintenance and decode graphs must share persistent device controller state, not host callbacks.";
        EXPECT_NE(controller_header.find("pre_policy_imbalance_numerator"),
                  std::string::npos)
            << "Device-side controller status must expose before/after load imbalance.";
        EXPECT_NE(controller_header.find("post_policy_imbalance_numerator"),
                  std::string::npos);
        EXPECT_NE(controller_header.find("projectedParticipantLoadForExpert"),
                  std::string::npos)
            << "Host mirror must compute load-spread stats through the shared policy helper.";
        EXPECT_NE(controller_header.find("const bool collect_load_stats"), std::string::npos);
        EXPECT_NE(controller_header.find("if (collect_load_stats)"), std::string::npos)
            << "Projected load-spread stats are diagnostic and must not run on the default hot path.";
        EXPECT_NE(controller_header.find("waves[2]"), std::string::npos)
            << "Rebalance waves should be double-buffered so transfer and apply can overlap.";
        EXPECT_NE(controller_header.find("decode_apply_polls"), std::string::npos)
            << "Decode should poll ready epochs cheaply rather than run the full producer every token.";
        EXPECT_NE(controller_header.find("host may allocate and inspect the buffer for diagnostics"),
                  std::string::npos);
        EXPECT_NE(controller_header.find("must produce and consume these entries entirely from"),
                  std::string::npos);
        EXPECT_NE(stage_header.find("enum class DeviceMoERebalanceStagePhase"), std::string::npos);
        EXPECT_NE(stage_header.find("CollectState"), std::string::npos);
        EXPECT_NE(stage_header.find("PlanAndCopyAfterSideband"), std::string::npos);
        EXPECT_NE(stage_header.find("PlanProbeAfterSideband"), std::string::npos);
        EXPECT_NE(stage_header.find("GatherCommandsAndCopyPreparedPayload"), std::string::npos);
        EXPECT_NE(stage_header.find("CopyPreparedPayload"), std::string::npos);
        EXPECT_NE(stage_header.find("JoinTransfer"), std::string::npos);
        EXPECT_EQ(controller_header.find("PeerRead"), std::string::npos)
            << "Device-side rebalance must not expose an untested direct peer-read mode.";
        EXPECT_EQ(stage_header.find("DeviceMoERebalanceTransferMode::PeerRead"), std::string::npos);
        EXPECT_NE(stage_header.find("DeviceMoERebalanceTransferMode::ResidentOnly"), std::string::npos)
            << "The compute stage should default to resident-only graph-captured rebalance.";
        EXPECT_NE(stage_header.find("join_transfer_stream_after_copy"), std::string::npos);
        EXPECT_NE(stage_header.find("class DeviceMoERebalanceTransferState"), std::string::npos);
        EXPECT_NE(stage_header.find("WS_TRANSFER_PLAN"), std::string::npos);
        EXPECT_NE(stage_header.find("WS_TRANSFER_PLAN_COUNT"), std::string::npos);
        EXPECT_NE(stage_header.find("WS_COMMAND_HEADER"), std::string::npos);
        EXPECT_NE(stage_header.find("WS_CONTROLLER_STATE"), std::string::npos)
            << "Publish/apply handoff must use persistent device controller state.";
        EXPECT_NE(stage_header.find("WS_GATHERED_TRANSFER_PLAN"), std::string::npos);
        EXPECT_NE(stage_header.find("WS_GATHERED_COMMAND_HEADER"), std::string::npos);
        EXPECT_NE(stage_header.find("WS_LOCAL_TRANSFER_PAYLOAD"), std::string::npos);
        EXPECT_NE(stage_header.find("WS_GATHERED_TRANSFER_PAYLOAD"), std::string::npos);
        EXPECT_NE(stage_header.find("WS_WAVE_STATE"), std::string::npos);
        EXPECT_NE(stage_source.find("transferPlanEntries()"), std::string::npos);
        EXPECT_NE(stage_source.find("transferPlanCapacity()"), std::string::npos);
        EXPECT_NE(stage_source.find("commandBufferCount()"), std::string::npos)
            << "Transfer command metadata must be double-buffered for deferred graph-side apply.";
        EXPECT_NE(stage_header.find("commandBufferCount() const"), std::string::npos);
        EXPECT_NE(stage_source.find("collectsState()"), std::string::npos);
        EXPECT_NE(stage_source.find("gathersStateInline()"), std::string::npos);
        EXPECT_NE(stage_source.find("runsController()"), std::string::npos);
        EXPECT_NE(stage_source.find("usesReadyWaveApply()"), std::string::npos);
        EXPECT_NE(stage_source.find("DeviceMoERebalanceFlags::DeferRuntimeApply"),
                  std::string::npos)
            << "Resident-only deferred apply should allocate/poll the ready-wave mailbox without transfer slots.";
        EXPECT_NE(stage_source.find("histogramLayerCount()"), std::string::npos)
            << "Async maintenance should size histogram gathers by the rolling wave, not the full model.";
        EXPECT_NE(stage_source.find("packDeviceRebalanceHistograms(\n                        runtime_layers,\n                        local,\n                        params_.config,\n                        wave_state,\n                        controller_state"),
                  std::string::npos)
            << "Histogram packing must follow the device-owned wave cursor used by the controller.";
        EXPECT_NE(stage_source.find("\"device_rebalance_stage_mode\""), std::string::npos)
            << "Benchmarks need to distinguish sideband-state replay from standalone state allgather replay.";
        EXPECT_NE(stage_source.find("\"uses_sideband_state\""), std::string::npos);
        EXPECT_NE(stage_source.find("\"gathers_state_inline\""), std::string::npos);
        EXPECT_NE(stage_source.find("\"device_rebalance_histogram_payload_bytes\""), std::string::npos)
            << "Perfstats must expose histogram payload bytes so sideband/maintenance traffic is measurable.";
        EXPECT_NE(stage_source.find("\"histogram_layer_count\""), std::string::npos);
        EXPECT_NE(stage_source.find("\"local_bytes\""), std::string::npos);
        EXPECT_NE(stage_source.find("\"gathered_bytes\""), std::string::npos);
        EXPECT_NE(stage_source.find("packDeviceRebalanceCollectivePayloads("), std::string::npos);
        EXPECT_NE(stage_source.find("unpackDeviceRebalanceCollectivePayloads("), std::string::npos);
        EXPECT_EQ(stage_source.find("usesCollectiveTransfer"), std::string::npos)
            << "Collective all-gather is now the only graph-side transfer-slot path.";
        EXPECT_EQ(stage_source.find("gatheredDirectoryBufferName"), std::string::npos)
            << "Compact transfer slots must not allgather the full expert directory.";
        EXPECT_NE(stage_source.find("localSourceDescriptorsBufferName"), std::string::npos)
            << "Compact transfer slots must publish plan-sized source descriptor responses.";
        EXPECT_NE(stage_source.find("payloadSlotCapacity()"), std::string::npos)
            << "Compact transfer payload capacity must be split from command metadata capacity.";
        EXPECT_EQ(stage_source.find("gatheredSourceDescriptorsBufferName"), std::string::npos)
            << "Compact transfer slots must carry descriptors inside grouped payload slots, not a separate descriptor collective.";
        EXPECT_NE(stage_source.find("packDeviceRebalanceSourceDescriptors("), std::string::npos)
            << "Compact transfer slots must pack source descriptors after plan/header allgather.";
        EXPECT_EQ(stage_source.find("workspaceSuffix() + \"_source_descriptors\""), std::string::npos)
            << "Compact transfer slots must not launch a separate source-descriptor allgather.";
        EXPECT_NE(stage_source.find("packDeviceRebalanceCompactPayloads("), std::string::npos)
            << "Compact transfer slots must pack planned non-empty arrivals for NCCL/RCCL grouped payload collectives.";
        EXPECT_EQ(stage_source.find("copyDeviceRebalanceArrivals("), std::string::npos)
            << "Device-side compact arrivals must not use destination-side peer-pointer reads.";
        EXPECT_EQ(stage_header.find("gatheredDirectoryBufferName"), std::string::npos);
        EXPECT_NE(stage_header.find("localSourceDescriptorsBufferName"), std::string::npos);
        EXPECT_EQ(stage_header.find("gatheredSourceDescriptorsBufferName"), std::string::npos);
        EXPECT_NE(stage_source.find("usesCompactTransferSlots()"), std::string::npos);
        EXPECT_NE(stage_source.find("usesFixedPayloadTransfer()"), std::string::npos)
            << "Fixed payload arenas must stay isolated behind an explicit predicate.";
        const size_t pack_payload_call =
            stage_source.find("packDeviceRebalanceCollectivePayloads(");
        ASSERT_NE(pack_payload_call, std::string::npos);
        const std::string pack_payload_args =
            stage_source.substr(pack_payload_call, 900);
        EXPECT_NE(pack_payload_args.find("local_directory"), std::string::npos)
            << "Collective payload packing must use the local resident directory.";
        EXPECT_NE(pack_payload_args.find("local_transfer_payload"), std::string::npos);

        const size_t unpack_payload_call =
            stage_source.find("unpackDeviceRebalanceCollectivePayloads(");
        ASSERT_NE(unpack_payload_call, std::string::npos);
        const std::string unpack_payload_args =
            stage_source.substr(unpack_payload_call, 900);
        EXPECT_NE(unpack_payload_args.find("command_header"), std::string::npos);
        EXPECT_NE(unpack_payload_args.find("gathered_transfer_payload"), std::string::npos)
            << "Collective payload unpack must consume source descriptors embedded in payload slots.";
        EXPECT_EQ(stage_source.find("command_header,\n                            nullptr,\n                            gathered_transfer_payload"),
                  std::string::npos)
            << "Collective payload unpack must not retain a dummy gathered-directory parameter.";
        EXPECT_NE(stage_source.find("workspaceSuffix() + \"_transfer_payload\""), std::string::npos);
        EXPECT_NE(stage_source.find("runDeviceRebalanceController("), std::string::npos);
        EXPECT_NE(stage_source.find("plan_entries"), std::string::npos);
        EXPECT_NE(stage_source.find("plan_count"), std::string::npos);
        EXPECT_NE(stage_source.find("command_header"), std::string::npos);
        EXPECT_NE(stage_source.find("controller_state"), std::string::npos);
        EXPECT_NE(stage_source.find("initializeDeviceRebalanceGraphController("), std::string::npos)
            << "Captured replay must initialize/preserve device-side controller state without host reads.";
        EXPECT_NE(stage_source.find("publishDeviceRebalanceTransferComplete("), std::string::npos)
            << "Transfer completion must be published by a graph-capturable kernel on the transfer stream.";
        EXPECT_NE(stage_source.find("applyReadyDeviceRebalanceWave("), std::string::npos)
            << "Decode apply must poll a ready device wave rather than host-published state.";
        EXPECT_NE(stage_source.find("projectDeviceRebalanceDomainCommands("), std::string::npos)
            << "Gathered root command buffers must be projected into every participant's local apply ABI.";
        const size_t followup_phase =
            stage_source.find("DeviceMoERebalanceStagePhase::GatherCommandsAndCopyPreparedPayload");
        const size_t command_header_allgather =
            stage_source.find("workspaceSuffix() + \"_transfer_header\"", followup_phase);
        const size_t domain_projection =
            stage_source.find("if (!project_domain_commands())", command_header_allgather);
        const size_t probe_return =
            stage_source.find("DeviceMoERebalanceStagePhase::PlanProbeAfterSideband",
                              domain_projection);
        const size_t copy_payload_lambda =
            stage_source.find("auto copy_prepared_payload");
        const size_t source_descriptor_pack =
            stage_source.find("packDeviceRebalanceSourceDescriptors(", copy_payload_lambda);
        ASSERT_NE(followup_phase, std::string::npos);
        ASSERT_NE(command_header_allgather, std::string::npos);
        ASSERT_NE(domain_projection, std::string::npos);
        ASSERT_NE(probe_return, std::string::npos);
        ASSERT_NE(copy_payload_lambda, std::string::npos);
        ASSERT_NE(source_descriptor_pack, std::string::npos);
        EXPECT_LT(followup_phase, command_header_allgather)
            << "Only the metadata/payload follow-up graph should gather command headers.";
        EXPECT_LT(command_header_allgather, domain_projection)
            << "Command projection must run after command/header allgather.";
        EXPECT_LT(domain_projection, probe_return)
            << "The probe branch must return before command/header allgather and payload buckets.";
        EXPECT_LT(copy_payload_lambda, source_descriptor_pack)
            << "Source descriptor packing belongs to the payload graph body.";
        const std::string source_descriptor_args =
            stage_source.substr(source_descriptor_pack, 600);
        EXPECT_NE(source_descriptor_args.find("plan_entries"), std::string::npos)
            << "Compact source descriptors must consume the projected local command plan.";
        EXPECT_NE(source_descriptor_args.find("command_header"), std::string::npos)
            << "Compact source descriptors must consume the projected local command header.";
        EXPECT_EQ(source_descriptor_args.find("gathered_plan_entries"), std::string::npos)
            << "Compact source descriptors must not read stale pre-projection gathered plans.";
        EXPECT_EQ(source_descriptor_args.find("gathered_command_headers"), std::string::npos)
            << "Compact source descriptors must not read stale pre-projection gathered headers.";
        EXPECT_NE(stage_source.find("wave_state"), std::string::npos);
        EXPECT_NE(stage_header.find("prepareGraphLaunch"), std::string::npos)
            << "Transfer stream/event resources should be prepared before graph capture begins.";
        EXPECT_NE(stage_header.find("needsGraphLaunchPreparation() const override { return usesTransferSlotApply(); }"),
                  std::string::npos);
        const size_t prepare_launch =
            stage_source.find("bool MoEDeviceRebalanceStage::prepareGraphLaunch");
        ASSERT_NE(prepare_launch, std::string::npos);
        const size_t ensure_transfer_state =
            stage_source.find("ensureAsyncTransferState()", prepare_launch);
        const size_t precapture_fence =
            stage_source.find("synchronizeStreamChecked(transfer_state->transferStream())",
                              prepare_launch);
        const size_t fence_counter =
            stage_source.find("device_rebalance_precapture_transfer_stream_fence",
                              prepare_launch);
        ASSERT_NE(ensure_transfer_state, std::string::npos);
        ASSERT_NE(precapture_fence, std::string::npos)
            << "Transfer-slot rebalance must drain the auxiliary stream before graph capture begins.";
        EXPECT_LT(ensure_transfer_state, precapture_fence)
            << "The pre-capture fence must run after transfer stream/event allocation.";
        EXPECT_NE(fence_counter, std::string::npos)
            << "Perfstats should prove the pre-capture transfer-stream fence ran in e2e logs.";
        EXPECT_NE(stage_source.find("getOrCreateAuxiliaryStream"), std::string::npos)
            << "Transfer-slot arrivals must use a context-owned auxiliary stream.";
        EXPECT_NE(stage_source.find("gpu_ctx->recordEventChecked(transfer_state->computeReadyEvent(), stream)"),
                  std::string::npos)
            << "The compute->transfer edge must be captured as a graph event.";
        EXPECT_NE(stage_source.find("gpu_ctx->waitEventChecked(transfer_state->computeReadyEvent(), transfer_stream)"),
                  std::string::npos);
        EXPECT_NE(stage_source.find("moe_kernel->setGPUStream(transfer_stream)"), std::string::npos)
            << "Packed expert payload copies must run on the transfer stream.";
        EXPECT_EQ(stage_source.find("transfer_stream_is_stage_stream"), std::string::npos)
            << "CUDA and ROCm production paths must not retain a stage-stream fallback.";
        EXPECT_EQ(stage_source.find("stage_stream_fallback"), std::string::npos)
            << "Rebalance transfers should use the context-owned auxiliary stream on all GPU backends.";
        EXPECT_EQ(stage_source.find("HIP stream capture currently does not replay"), std::string::npos)
            << "The old ROCm correctness fallback must not reappear.";
        EXPECT_NE(stage_source.find("\"device_rebalance_transfer_stream_path\""), std::string::npos)
            << "Perfstats must expose that rebalance transfers use the auxiliary stream.";
        EXPECT_NE(stage_source.find("gpu_ctx->recordEventChecked(transfer_state->transferDoneEvent()"),
                  std::string::npos)
            << "The copy producer must record transfer completion.";
        EXPECT_NE(stage_source.find("isGraphCaptureActive()"), std::string::npos)
            << "Graph capture paths must make stream fork/join behavior explicit.";
        EXPECT_NE(stage_source.find("params_.join_transfer_stream_after_copy"), std::string::npos)
            << "Split decode producers must be able to defer the transfer-stream join.";
        EXPECT_NE(stage_source.find("join_transfer_stream_to_capture_stream"), std::string::npos);
        EXPECT_NE(stage_source.find("transfer stream back to the stage stream"),
                  std::string::npos);
        EXPECT_NE(stage_source.find("Failed to queue late transfer-stream join"),
                  std::string::npos)
            << "Late graph join keeps capture legal while allowing transfer/compute overlap.";
        EXPECT_EQ(stage_source.find("isGraphCaptureActive() &&\n                params_.join_transfer_stream_after_copy"),
                  std::string::npos)
            << "Maintenance metadata/payload replays may execute stream-only or direct; the transfer-stream join must not be capture-only.";
        EXPECT_EQ(stage_source.find("isGraphCaptureActive() &&\n                !gpu_ctx->waitEventChecked(transfer_state->transferDoneEvent(), stream)"),
                  std::string::npos)
            << "A stage that advertises join_transfer_stream_after_copy must make its public stream wait outside graph capture too.";
        EXPECT_NE(stage_source.find("must_wait_for_inline_transfer"), std::string::npos)
            << "Split decode apply should poll device-ready waves without an unconditional transfer wait.";
        EXPECT_NE(stage_source.find("DeviceMoERebalanceStagePhase::PlanCopyApply"),
                  std::string::npos)
            << "Only inline producer+consumer rebalance should wait on its just-queued transfer.";
        EXPECT_NE(stage_source.find("gpu_ctx->waitEventChecked(transfer_state->transferDoneEvent(), stream)"),
                  std::string::npos)
            << "Inline PlanCopyApply still needs a graph event dependency for its own transfer.";
        EXPECT_NE(stage_source.find("Failed to queue compute-to-transfer stream dependency"), std::string::npos);
        EXPECT_NE(stage_source.find("Failed to queue transfer-to-apply stream dependency"), std::string::npos);
        EXPECT_NE(stage_source.find("transfer-stream completion event"), std::string::npos);
        EXPECT_EQ(stage_source.find("synchronizeStream("), std::string::npos)
            << "Graph-side rebalance publish/apply must not block the host.";
        EXPECT_EQ(stage_source.find("LLAMINAR_MOE_REBALANCE_DEBUG_SYNC"), std::string::npos)
            << "Do not leave crash-localization sync knobs in the production rebalance path.";
        EXPECT_EQ(stage_source.find("transfer_event_backend_->recordEvent"), std::string::npos)
            << "IBackend::recordEvent intentionally no-ops during capture; use IWorkerGPUContext.";

        const fs::path cuda_kernel_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp";
        const fs::path rocm_kernel_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(cuda_kernel_path)) << cuda_kernel_path;
        ASSERT_TRUE(fs::exists(rocm_kernel_path)) << rocm_kernel_path;
        const std::string cuda_kernel = readFile(cuda_kernel_path);
        const std::string rocm_kernel = readFile(rocm_kernel_path);
        ASSERT_FALSE(cuda_kernel.empty()) << cuda_kernel_path;
        ASSERT_FALSE(rocm_kernel.empty()) << rocm_kernel_path;
        EXPECT_NE(cuda_kernel.find("(!plan_count && !command_header)"), std::string::npos)
            << "CUDA graph-side rebalance kernels must accept the device command header as the publish/apply ABI.";
        EXPECT_NE(rocm_kernel.find("(!plan_count && !command_header)"), std::string::npos)
            << "ROCm graph-side rebalance kernels must accept the device command header as the publish/apply ABI.";
        const size_t cuda_ready_wrapper =
            cuda_kernel.find("bool CUDAMoEKernel::applyReadyDeviceRebalanceWave");
        const size_t rocm_ready_wrapper =
            rocm_kernel.find("bool ROCmMoEKernel::applyReadyDeviceRebalanceWave");
        ASSERT_NE(cuda_ready_wrapper, std::string::npos);
        ASSERT_NE(rocm_ready_wrapper, std::string::npos);
        const std::string cuda_ready_wrapper_body =
            cuda_kernel.substr(cuda_ready_wrapper, 900);
        const std::string rocm_ready_wrapper_body =
            rocm_kernel.substr(rocm_ready_wrapper, 900);
        EXPECT_EQ(cuda_ready_wrapper_body.find("!local_transfer_slots"), std::string::npos)
            << "CUDA ready-wave apply must accept resident-only commands with no transfer slots.";
        EXPECT_EQ(rocm_ready_wrapper_body.find("!local_transfer_slots"), std::string::npos)
            << "ROCm ready-wave apply must accept resident-only commands with no transfer slots.";
        EXPECT_EQ(cuda_kernel.find("const DeviceMoEExpertDirectoryEntry *gathered_directory,\n        const uint8_t *gathered_payload"),
                  std::string::npos)
            << "CUDA collective payload unpack should not retain host-style gathered-directory input.";
        EXPECT_EQ(rocm_kernel.find("const DeviceMoEExpertDirectoryEntry *gathered_directory,\n        const uint8_t *gathered_payload"),
                  std::string::npos)
            << "ROCm collective payload unpack should not retain host-style gathered-directory input.";

        const fs::path cuda_kernel_header_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.h";
        const fs::path rocm_kernel_header_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.h";
        const fs::path cuda_kernel_impl_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_kernel_impl_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        ASSERT_TRUE(fs::exists(cuda_kernel_header_path)) << cuda_kernel_header_path;
        ASSERT_TRUE(fs::exists(rocm_kernel_header_path)) << rocm_kernel_header_path;
        ASSERT_TRUE(fs::exists(cuda_kernel_impl_path)) << cuda_kernel_impl_path;
        ASSERT_TRUE(fs::exists(rocm_kernel_impl_path)) << rocm_kernel_impl_path;
        const std::string cuda_kernel_header = readFile(cuda_kernel_header_path);
        const std::string rocm_kernel_header = readFile(rocm_kernel_header_path);
        const std::string cuda_kernel_impl = readFile(cuda_kernel_impl_path);
        const std::string rocm_kernel_impl = readFile(rocm_kernel_impl_path);
        ASSERT_FALSE(cuda_kernel_header.empty()) << cuda_kernel_header_path;
        ASSERT_FALSE(rocm_kernel_header.empty()) << rocm_kernel_header_path;
        ASSERT_FALSE(cuda_kernel_impl.empty()) << cuda_kernel_impl_path;
        ASSERT_FALSE(rocm_kernel_impl.empty()) << rocm_kernel_impl_path;
        for (const std::string &token : {
                 "initializeDeviceRebalanceGraphController",
                 "packDeviceRebalanceSourceDescriptors",
                 "projectDeviceRebalanceDomainCommands",
                 "publishDeviceRebalanceTransferComplete",
                 "applyReadyDeviceRebalanceWave",
                 "command_buffer_count",
             })
        {
            EXPECT_NE(cuda_kernel_header.find(token), std::string::npos)
                << "CUDA header missing " << token;
            EXPECT_NE(rocm_kernel_header.find(token), std::string::npos)
                << "ROCm header missing " << token;
            EXPECT_NE(cuda_kernel.find(token), std::string::npos)
                << "CUDA bridge missing " << token;
            EXPECT_NE(rocm_kernel.find(token), std::string::npos)
                << "ROCm bridge missing " << token;
        }
        for (const std::string &token : {
                 "DeviceMoERebalanceGraphControllerStateView",
                 "pack_rebalance_source_descriptors_kernel",
                 "project_rebalance_domain_commands_kernel",
                 "pack_rebalance_compact_payloads_kernel",
                 "init_rebalance_graph_controller_state_kernel",
                 "publish_rebalance_transfer_complete_kernel",
                 "apply_rebalance_arrivals_kernel",
                 "kDeviceMoERebalanceLifecycleReadyToApply",
                 "decode_apply_polls",
                 "decode_apply_hits",
                 "rebalance_active_command_wave_index",
                 "rebalance_active_wave_busy_for_new_plan",
                 "DeviceMoEReadyWaveSelection",
                 "layer_wave_count",
                 "window_index < layer_wave_count",
                 "start_offset + layer_wave_count",
                 "kDeviceMoERebalanceFlagDeferRuntimeApply",
                 "kDeviceMoERebalanceFlagCollectLoadStats",
                 "kDeviceMoERebalancePlanResidentExpertAssignment",
                 "rebalance_layer_wave_count",
                 "rebalance_actual_layer_for_wave_index",
                 "pre_policy_imbalance_numerator",
                 "post_policy_imbalance_numerator",
                 "candidate_arrivals_considered",
                 "candidate_arrivals_pruned_by_count_bound",
                 "skipped_busy_wave",
                 "window_ready_slots",
                 "window_required_slots",
                 "candidate_load_spread_improvement_total",
                 "shared_pre_policy_load",
                 "shared_post_policy_load",
                 "shared_current_policy_load",
                 "shared_candidate_policy_load",
	                 "evaluateAddingResidentDynamicSpread",
	                 "addingResidentImprovesDynamicSpread",
                 "requiredLoadSpreadImprovement",
                 "shared_required_load_spread_improvement",
                 "skipped_no_improvement",
                 "collect_load_stats",
                 "if (leader && collect_load_stats)",
                 "if (collect_load_stats)",
                 "projectedParticipantLoadForExpert",
                 "config.root_participant",
             })
        {
            EXPECT_NE(cuda_kernel_impl.find(token), std::string::npos)
                << "CUDA device implementation missing " << token;
            EXPECT_NE(rocm_kernel_impl.find(token), std::string::npos)
                << "ROCm device implementation missing " << token;
        }
        EXPECT_EQ(cuda_kernel_impl.find("rebalance_source_descriptor_gathered_index"), std::string::npos)
            << "CUDA compact transfer slots must not revive gathered source descriptor addressing.";
        EXPECT_EQ(rocm_kernel_impl.find("rebalance_source_descriptor_gathered_index"), std::string::npos)
            << "ROCm compact transfer slots must not revive gathered source descriptor addressing.";
        EXPECT_EQ(cuda_kernel_impl.find("if (rebalance_graph_controller_state_ok(controller_state, config))\n"
                                        "                        ++controller_state->decode_apply_polls"),
                  std::string::npos)
            << "CUDA decode apply poll counter must be gated by CollectLoadStats.";
        EXPECT_EQ(rocm_kernel_impl.find("if (rebalance_graph_controller_state_ok(controller_state, config))\n"
                                        "                        ++controller_state->decode_apply_polls"),
                  std::string::npos)
            << "ROCm decode apply poll counter must be gated by CollectLoadStats.";

        const size_t cuda_apply_kernel =
            cuda_kernel_impl.find("__global__ void apply_rebalance_arrivals_kernel(");
        const size_t rocm_apply_kernel =
            rocm_kernel_impl.find("__global__ void apply_rebalance_arrivals_kernel(");
        ASSERT_NE(cuda_apply_kernel, std::string::npos);
        ASSERT_NE(rocm_apply_kernel, std::string::npos);
        const size_t cuda_next_kernel =
            cuda_kernel_impl.find("__global__ void", cuda_apply_kernel + 1);
        const size_t rocm_next_kernel =
            rocm_kernel_impl.find("__global__ void", rocm_apply_kernel + 1);
        ASSERT_NE(cuda_next_kernel, std::string::npos);
        ASSERT_NE(rocm_next_kernel, std::string::npos);
        const std::string cuda_apply_body =
            cuda_kernel_impl.substr(cuda_apply_kernel, cuda_next_kernel - cuda_apply_kernel);
        const std::string rocm_apply_body =
            rocm_kernel_impl.substr(rocm_apply_kernel, rocm_next_kernel - rocm_apply_kernel);
        for (const auto &[label, body] : {
                 std::pair<std::string, std::string>{"CUDA", cuda_apply_body},
                 std::pair<std::string, std::string>{"ROCm", rocm_apply_body},
             })
        {
            EXPECT_NE(body.find("destination_local"), std::string::npos)
                << label << " arrival apply must distinguish metadata propagation from local slot install.";
            EXPECT_NE(body.find("destination_bit"), std::string::npos)
                << label << " arrival apply must publish the destination participant in every resident mask.";
            EXPECT_NE(body.find("plan.destination_participant >= config.participant_count"),
                      std::string::npos)
                << label << " arrival apply must validate destination range rather than treating it as a local filter.";
            EXPECT_EQ(body.find("plan.destination_participant != config.participant_id ||"),
                      std::string::npos)
                << label << " arrival apply must not skip non-destination plan entries.";
            EXPECT_EQ(body.find("++status->skipped_wrong_destination"), std::string::npos)
                << label << " arrival apply must propagate replica metadata on non-destination participants.";
        }

        const size_t cuda_ready_c_abi =
            cuda_kernel_impl.find("bool cudaMoE_apply_ready_rebalance_wave");
        const size_t rocm_ready_c_abi =
            rocm_kernel_impl.find("bool hipMoE_apply_ready_rebalance_wave");
        ASSERT_NE(cuda_ready_c_abi, std::string::npos);
        ASSERT_NE(rocm_ready_c_abi, std::string::npos);
        const std::string cuda_ready_c_abi_body =
            cuda_kernel_impl.substr(cuda_ready_c_abi, 900);
        const std::string rocm_ready_c_abi_body =
            rocm_kernel_impl.substr(rocm_ready_c_abi, 900);
        EXPECT_EQ(cuda_ready_c_abi_body.find("!local_transfer_slots"), std::string::npos)
            << "CUDA C ABI ready apply must allow resident-only command buffers.";
        EXPECT_EQ(rocm_ready_c_abi_body.find("!local_transfer_slots"), std::string::npos)
            << "ROCm C ABI ready apply must allow resident-only command buffers.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceSideRebalanceBypassesHostDecodeMaintenance)
    {
        const fs::path root = findRepoRoot();
        const fs::path runner_path = root / "src/v2/execution/runner/OrchestrationRunner.cpp";
        const fs::path dgo_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        const fs::path dgo_header_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h";
        const fs::path rank_path =
            root / "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp";
        const fs::path rank_header_path =
            root / "src/v2/execution/local_execution/orchestrators/RankOrchestrator.h";
        const fs::path graph_builder_path =
            root / "src/v2/execution/local_execution/graph/IGraphBuilder.h";
        const fs::path device_rebalance_stage_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.cpp";
        const fs::path host_rendezvous_path =
            root / "src/v2/execution/moe/MoEDeviceRebalanceHostRendezvous.h";
        const fs::path debug_env_path = root / "src/v2/utils/DebugEnv.h";
        const fs::path iface_path = root / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h";
        const fs::path chat_path = root / "src/v2/app/modes/ChatCompletionHandler.cpp";
        const fs::path benchmark_path = root / "src/v2/app/modes/BenchmarkMode.cpp";
        const fs::path benchmark_runner_path = root / "src/v2/utils/BenchmarkRunner.cpp";
        const fs::path adapter_path = root / "src/v2/app/InferenceRunnerAdapter.cpp";
        const fs::path adapter_header_path = root / "src/v2/app/InferenceRunnerAdapter.h";
        const fs::path qwen36_parity_path =
            root / "tests/v2/integration/parity/qwen36/Qwen36MoEParityTestBase.h";
        const fs::path server_e2e_path =
            root / "tests/v2/e2e/server/test_server_e2e.sh";
        ASSERT_TRUE(fs::exists(runner_path)) << runner_path;
        ASSERT_TRUE(fs::exists(dgo_path)) << dgo_path;
        ASSERT_TRUE(fs::exists(dgo_header_path)) << dgo_header_path;
        ASSERT_TRUE(fs::exists(rank_path)) << rank_path;
        ASSERT_TRUE(fs::exists(rank_header_path)) << rank_header_path;
        ASSERT_TRUE(fs::exists(graph_builder_path)) << graph_builder_path;
        ASSERT_TRUE(fs::exists(device_rebalance_stage_path)) << device_rebalance_stage_path;
        ASSERT_TRUE(fs::exists(host_rendezvous_path)) << host_rendezvous_path;
        ASSERT_TRUE(fs::exists(debug_env_path)) << debug_env_path;
        ASSERT_TRUE(fs::exists(iface_path)) << iface_path;
        ASSERT_TRUE(fs::exists(chat_path)) << chat_path;
        ASSERT_TRUE(fs::exists(benchmark_path)) << benchmark_path;
        ASSERT_TRUE(fs::exists(benchmark_runner_path)) << benchmark_runner_path;
        ASSERT_TRUE(fs::exists(adapter_path)) << adapter_path;
        ASSERT_TRUE(fs::exists(adapter_header_path)) << adapter_header_path;
        ASSERT_TRUE(fs::exists(qwen36_parity_path)) << qwen36_parity_path;
        ASSERT_TRUE(fs::exists(server_e2e_path)) << server_e2e_path;

        const std::string runner = readFile(runner_path);
        const std::string dgo = readFile(dgo_path);
        const std::string dgo_header = readFile(dgo_header_path);
        const std::string rank = readFile(rank_path);
        const std::string rank_header = readFile(rank_header_path);
        const std::string graph_builder = readFile(graph_builder_path);
        const std::string device_rebalance_stage = readFile(device_rebalance_stage_path);
        const std::string host_rendezvous = readFile(host_rendezvous_path);
        const std::string debug_env = readFile(debug_env_path);
        const std::string iface = readFile(iface_path);
        const std::string chat = readFile(chat_path);
        const std::string benchmark = readFile(benchmark_path);
        const std::string benchmark_runner = readFile(benchmark_runner_path);
        const std::string adapter = readFile(adapter_path);
        const std::string adapter_header = readFile(adapter_header_path);
        const std::string qwen36_parity = readFile(qwen36_parity_path);
        const std::string server_e2e = readFile(server_e2e_path);
        ASSERT_FALSE(runner.empty()) << runner_path;
        ASSERT_FALSE(dgo.empty()) << dgo_path;
        ASSERT_FALSE(dgo_header.empty()) << dgo_header_path;
        ASSERT_FALSE(rank.empty()) << rank_path;
        ASSERT_FALSE(rank_header.empty()) << rank_header_path;
        ASSERT_FALSE(graph_builder.empty()) << graph_builder_path;
        ASSERT_FALSE(device_rebalance_stage.empty()) << device_rebalance_stage_path;
        ASSERT_FALSE(host_rendezvous.empty()) << host_rendezvous_path;
        ASSERT_FALSE(debug_env.empty()) << debug_env_path;
        ASSERT_FALSE(iface.empty()) << iface_path;
        ASSERT_FALSE(chat.empty()) << chat_path;
        ASSERT_FALSE(benchmark.empty()) << benchmark_path;
        ASSERT_FALSE(benchmark_runner.empty()) << benchmark_runner_path;
        ASSERT_FALSE(adapter.empty()) << adapter_path;
        ASSERT_FALSE(adapter_header.empty()) << adapter_header_path;
        ASSERT_FALSE(qwen36_parity.empty()) << qwen36_parity_path;
        ASSERT_FALSE(server_e2e.empty()) << server_e2e_path;

        EXPECT_NE(iface.find("usesDeviceSideMoERebalanceController() const"), std::string::npos);
        EXPECT_NE(iface.find("drainCompletedDecodeBoundaryMaintenanceDiagnostics()"),
                  std::string::npos)
            << "IInferenceRunner should expose a non-reset epilogue hook for completed async maintenance diagnostics.";

        const size_t maybe_start = runner.find("bool OrchestrationRunner::maybeApplyMoERebalance()");
        ASSERT_NE(maybe_start, std::string::npos);
        const size_t maybe_end = runner.find("// =========================================================================", maybe_start);
        ASSERT_NE(maybe_end, std::string::npos);
        const std::string maybe_body = runner.substr(maybe_start, maybe_end - maybe_start);

        const size_t device_side_gate = maybe_body.find("usesDeviceSideMoERebalanceController()");
        const size_t host_controller_lookup = maybe_body.find("auto *controller = moeRebalanceController()");
        const size_t pending_host_publish = maybe_body.find("publishPendingMoERebalanceUpdate()");
        const size_t host_histogram_sync = maybe_body.find("histogram->syncRuntimeHistograms()");
        const size_t host_apply = maybe_body.find("applyMoERebalanceWithReplicas()");
        ASSERT_NE(device_side_gate, std::string::npos);
        ASSERT_NE(host_controller_lookup, std::string::npos);
        ASSERT_NE(pending_host_publish, std::string::npos);
        ASSERT_NE(host_histogram_sync, std::string::npos);
        ASSERT_NE(host_apply, std::string::npos);
        EXPECT_LT(device_side_gate, host_controller_lookup)
            << "The device-side controller path must return before consulting the host controller.";
        EXPECT_LT(device_side_gate, pending_host_publish)
            << "The device-side controller must bypass host pending-publish drain.";
        EXPECT_LT(device_side_gate, host_histogram_sync)
            << "The device-side controller must bypass host histogram sync.";
        EXPECT_LT(device_side_gate, host_apply)
            << "The device-side controller must bypass host publish/apply.";
        EXPECT_NE(maybe_body.find("pending_moe_rebalance_prepare_.has_value()"), std::string::npos)
            << "A leftover host-prepared publish in device-side mode is a state-machine bug.";
        EXPECT_NE(maybe_body.find("pending host-prepared publish while the device-side controller is active"),
                  std::string::npos);
        EXPECT_EQ(maybe_body.find("device_side_host_maintenance_skips"), std::string::npos)
            << "Device-side mode should not pay host-controller/stat bookkeeping overhead.";
        EXPECT_NE(dgo.find("ensureGraphStableMoERebalanceTransferOrThrow"), std::string::npos);
        EXPECT_EQ(dgo.find("GPUExpertTransfer::canAccessPeer(primary_device, source_device)"),
                  std::string::npos)
            << "Device-side mode should rely on graph-captured NCCL/RCCL sidebands, not direct peer-read preflight.";
        EXPECT_EQ(dgo.find("GPUExpertTransfer::enablePeerAccess(primary_device, source_device)"),
                  std::string::npos)
            << "NCCL/RCCL should own transport selection for graph-side rebalance.";
        EXPECT_NE(dgo.find("supportsRawAllgatherOnStreamGraphCapture()"), std::string::npos)
            << "The legacy collective staging capability must stay visible and explicitly gated.";
        EXPECT_NE(dgo.find("moe_env.device_rebalance_payload_sideband ||\n                moe_env.allow_legacy_collective_rebalance_transfer"),
                  std::string::npos)
            << "The orchestrator must reject both fixed-size payload transfer lanes as one fail-closed gate.";
        EXPECT_NE(dgo.find("fixed-size collective payload arenas move empty expert slots"),
                  std::string::npos)
            << "Device-side mode must never move known-empty fixed payload slot capacity.";
        EXPECT_NE(dgo.find("host publish/apply fallback is refused"), std::string::npos)
            << "Homogeneous GPU device-side rebalance must fail closed when graph-captured sideband collectives are unavailable.";
        EXPECT_NE(dgo.find("env.moe_rebalance.device_rebalance_graph_controller"), std::string::npos)
            << "The runner must bypass host publish/apply only when the graph-native controller is enabled.";
        EXPECT_NE(graph_builder.find("buildDeviceMoERebalanceMaintenanceGraph("),
                  std::string::npos)
            << "Graph builders expose first-class rolling maintenance replay for device-side rebalance.";
        EXPECT_NE(graph_builder.find("return {};"), std::string::npos)
            << "Non-MoE graph builders should opt out with an empty maintenance graph.";
        EXPECT_NE(dgo_header.find("DeviceMoERebalanceMaintenanceGraphCache"), std::string::npos);
        EXPECT_NE(dgo_header.find("device_moe_rebalance_maintenance_payload_graphs_"), std::string::npos)
            << "Payload buckets need independently captured graph bodies so no-work waves skip payload replay.";
        EXPECT_NE(dgo_header.find("std::unordered_map<uint64_t, DeviceMoERebalanceMaintenanceGraphCache>"),
                  std::string::npos)
            << "Payload maintenance graph variants must be keyed by directed edge mask.";
        EXPECT_NE(dgo_header.find("resetSessionStatePreservingGraphReplay"), std::string::npos)
            << "Request reset should preserve replay-safe maintenance graphs.";
        const size_t reset_inference_start =
            dgo_header.find("void resetInferenceState(const InferenceStateResetRequest &request) override");
        ASSERT_NE(reset_inference_start, std::string::npos);
        const size_t reset_inference_end = dgo_header.find("void clear_cache() override", reset_inference_start);
        ASSERT_NE(reset_inference_end, std::string::npos);
        const std::string reset_inference_body =
            dgo_header.substr(reset_inference_start, reset_inference_end - reset_inference_start);
        const size_t clear_cache_sync = reset_inference_body.find("ctx->synchronize()");
        const size_t clear_cache_drain =
            reset_inference_body.find("drainCompletedDeviceMoERebalanceMaintenanceForRequestReset()");
        const size_t clear_cache_reset =
            reset_inference_body.find("device_moe_rebalance_maintenance_graph_.resetSessionStatePreservingGraphReplay()");
        ASSERT_NE(clear_cache_sync, std::string::npos);
        ASSERT_NE(clear_cache_drain, std::string::npos)
            << "Request reset should export completed maintenance diagnostics after synchronizing, not leak them into the next request.";
        ASSERT_NE(clear_cache_reset, std::string::npos);
        EXPECT_LT(clear_cache_sync, clear_cache_drain);
        EXPECT_LT(clear_cache_drain, clear_cache_reset);
        const size_t prefix_restore_moe_comment =
            reset_inference_body.find("Prefix restore imports portable MoE runtime state");
        ASSERT_NE(prefix_restore_moe_comment, std::string::npos)
            << "Prefix restore must document why MoE maintenance graphs are request-topology "
               "objects rather than replay-safe captures.";
        const size_t prefix_restore_moe_invalidate =
            reset_inference_body.find("device_moe_rebalance_maintenance_graph_.invalidate()",
                                      prefix_restore_moe_comment);
        const size_t prefix_restore_payload_clear =
            reset_inference_body.find("device_moe_rebalance_maintenance_payload_graphs_.clear()",
                                      prefix_restore_moe_comment);
        ASSERT_NE(prefix_restore_moe_invalidate, std::string::npos)
            << "Prefix restore must rebuild graph-side MoE maintenance stages after cached "
               "runtime import rebinds transfer-slot descriptors.";
        ASSERT_NE(prefix_restore_payload_clear, std::string::npos)
            << "Prefix restore must drop payload-specific MoE maintenance graph variants, "
               "not recapture stale stage objects for a new directed edge mask.";
        EXPECT_LT(prefix_restore_moe_invalidate, prefix_restore_payload_clear);
        const size_t clear_cache_start = reset_inference_end;
        const size_t clear_cache_end = dgo_header.find("int get_position() const override", clear_cache_start);
        ASSERT_NE(clear_cache_end, std::string::npos);
        const std::string clear_cache_body =
            dgo_header.substr(clear_cache_start, clear_cache_end - clear_cache_start);
        EXPECT_NE(clear_cache_body.find("InferenceStateResetRequest::requestBoundary(\"clear_cache\")"),
                  std::string::npos)
            << "clear_cache must enter the request-boundary reset path that synchronizes and drains maintenance diagnostics.";
        const size_t maintenance_cache_start =
            dgo_header.find("struct DeviceMoERebalanceMaintenanceGraphCache\n        {");
        ASSERT_NE(maintenance_cache_start, std::string::npos);
        const size_t maintenance_cache_end =
            dgo_header.find("DeviceMoERebalanceMaintenanceGraphCache device_moe_rebalance_maintenance_graph_",
                            maintenance_cache_start);
        ASSERT_NE(maintenance_cache_end, std::string::npos);
        const std::string maintenance_cache_body =
            dgo_header.substr(maintenance_cache_start,
                              maintenance_cache_end - maintenance_cache_start);
        const size_t replay_preserving_reset =
            maintenance_cache_body.find("void resetSessionStatePreservingGraphReplay()");
        ASSERT_NE(replay_preserving_reset, std::string::npos);
        const size_t hard_reset =
            maintenance_cache_body.find("void resetSessionState()", replay_preserving_reset);
        ASSERT_NE(hard_reset, std::string::npos);
        const std::string replay_preserving_reset_body =
            maintenance_cache_body.substr(replay_preserving_reset,
                                          hard_reset - replay_preserving_reset);
        EXPECT_NE(replay_preserving_reset_body.find("completion_event_in_flight = false;"),
                  std::string::npos)
            << "Request reset synchronizes before preserving maintenance graph replay, so stale warmup/request events must not force a next-request diagnostics readback.";
        EXPECT_NE(replay_preserving_reset_body.find("skipped_inflight_count = 0;"),
                  std::string::npos)
            << "Request-local maintenance skip counters should not leak across replay-preserving request resets.";
        EXPECT_NE(dgo.find("drainCompletedDeviceMoERebalanceMaintenanceForRequestReset"),
                  std::string::npos);
        EXPECT_EQ(dgo.find("deviceMoERebalanceTracePathFromEnv().empty())\n        {\n            cache.completion_event_in_flight = false;"),
                  std::string::npos)
            << "Request reset must never silently discard in-flight device maintenance when diagnostics are disabled.";
        EXPECT_NE(dgo.find("gpu_ctx->synchronizeStreamChecked(maintenance_stream)"),
                  std::string::npos)
            << "Request reset must explicitly drain unfinished maintenance streams before preserving graph replay state.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_request_reset_exports\""),
                  std::string::npos)
            << "Final measured maintenance waves must remain visible even when the next request reset retires the event.";
        EXPECT_NE(dgo_header.find("drainCompletedDecodeBoundaryMaintenanceDiagnostics() override"),
                  std::string::npos)
            << "DeviceGraphOrchestrator must expose the non-reset diagnostics drain through IInferenceRunner.";
        EXPECT_NE(runner.find("OrchestrationRunner::drainCompletedDecodeBoundaryMaintenanceDiagnostics"),
                  std::string::npos);
        EXPECT_NE(runner.find("runner_->drainCompletedDecodeBoundaryMaintenanceDiagnostics()"),
                  std::string::npos)
            << "OrchestrationRunner must forward benchmark epilogue diagnostics to the active runner.";
        EXPECT_NE(adapter_header.find("drainCompletedDecodeBoundaryMaintenanceDiagnostics() override"),
                  std::string::npos);
        EXPECT_NE(adapter.find("orch_runner_->drainCompletedDecodeBoundaryMaintenanceDiagnostics()"),
                  std::string::npos)
            << "The benchmark adapter must not swallow the diagnostics epilogue hook.";
        EXPECT_NE(rank_header.find("drainCompletedDecodeBoundaryMaintenanceDiagnostics() override"),
                  std::string::npos)
            << "RankOrchestrator must expose the diagnostics epilogue hook for LocalTP domains.";
        EXPECT_NE(rank.find("RankOrchestrator::drainCompletedDecodeBoundaryMaintenanceDiagnostics"),
                  std::string::npos);
        EXPECT_NE(rank.find("runner->drainCompletedDecodeBoundaryMaintenanceDiagnostics()"),
                  std::string::npos)
            << "RankOrchestrator must fan final maintenance diagnostics out to child devices.";
        EXPECT_NE(benchmark_runner.find("runner_->drainCompletedDecodeBoundaryMaintenanceDiagnostics()"),
                  std::string::npos)
            << "Benchmark JSON/perfstats export should include the final measured maintenance wave.";
        EXPECT_NE(dgo.find("maybeRunDeviceMoERebalanceMaintenanceGraph(effective_input)"),
                  std::string::npos)
            << "The rolling maintenance hook must stay explicit while disabled by default.";
        const size_t stable_predicate_start =
            dgo.find("bool DeviceGraphOrchestrator::usesGraphStableGpuMoERebalance() const");
        ASSERT_NE(stable_predicate_start, std::string::npos);
        const size_t stable_predicate_end =
            dgo.find("uint64_t DeviceGraphOrchestrator::moePlacementEpoch() const",
                     stable_predicate_start);
        ASSERT_NE(stable_predicate_end, std::string::npos);
        const std::string stable_predicate =
            dgo.substr(stable_predicate_start, stable_predicate_end - stable_predicate_start);
        EXPECT_NE(stable_predicate.find("config.moe.rebalance_mode != MoERebalanceMode::DYNAMIC"),
                  std::string::npos)
            << "Graph-stable GPU MoE rebalance must be disabled for --moe-rebalance off/static configs.";

        const size_t device_controller_start =
            dgo.find("bool DeviceGraphOrchestrator::usesDeviceSideMoERebalanceController() const");
        ASSERT_NE(device_controller_start, std::string::npos);
        const size_t device_controller_end =
            dgo.find("uint64_t DeviceGraphOrchestrator::moePlacementEpoch() const",
                     device_controller_start);
        ASSERT_NE(device_controller_end, std::string::npos);
        const std::string device_controller_predicate =
            dgo.substr(device_controller_start, device_controller_end - device_controller_start);
        EXPECT_NE(device_controller_predicate.find("config.moe.rebalance_mode != MoERebalanceMode::DYNAMIC"),
                  std::string::npos)
            << "The device-side controller opt-in env must not override a static rebalance config.";
        EXPECT_EQ(device_controller_predicate.find("hot_replica_cap <= 0"),
                  std::string::npos)
            << "Homogeneous GPU dynamic rebalance must stay device-owned when --moe-hot-expert-cache is off.";
        EXPECT_EQ(device_controller_predicate.find("gpu_cache_experts_per_layer > 0"),
                  std::string::npos)
            << "Homogeneous GPU dynamic rebalance must stay device-owned when --moe-hot-expert-cache is on.";
        EXPECT_EQ(device_controller_predicate.find("resolveCap("),
                  std::string::npos)
            << "The runner-side device-controller predicate must not depend on hot replica capacity.";

        const size_t maintenance_start =
            dgo.find("bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph(");
        ASSERT_NE(maintenance_start, std::string::npos);
        const size_t maintenance_end =
            dgo.find("// =====================================================================",
                     maintenance_start);
        ASSERT_NE(maintenance_end, std::string::npos);
        const std::string maintenance_body =
            dgo.substr(maintenance_start, maintenance_end - maintenance_start);
        const size_t maintenance_env_gate =
            maintenance_body.find("if (!env.moe_rebalance.device_rebalance_maintenance_graph)");
        const size_t maintenance_dynamic_gate =
            maintenance_body.find("if (!isMoeRebalancingActive())");
        const size_t maintenance_device_gate =
            maintenance_body.find("!state_.device_id.is_gpu()");
        const size_t maintenance_controller_gate =
            maintenance_body.find("usesDeviceSideMoERebalanceController()");
        const size_t maintenance_host_timer =
            maintenance_body.find("\"device_maintenance_graph_host_window\"");
        ASSERT_NE(maintenance_env_gate, std::string::npos);
        ASSERT_NE(maintenance_dynamic_gate, std::string::npos)
            << "Static/off rebalance must return before the maintenance hook does any per-token work.";
        ASSERT_NE(maintenance_device_gate, std::string::npos);
        ASSERT_NE(maintenance_controller_gate, std::string::npos);
        ASSERT_NE(maintenance_host_timer, std::string::npos);
        EXPECT_LT(maintenance_env_gate, maintenance_dynamic_gate);
        EXPECT_LT(maintenance_dynamic_gate, maintenance_device_gate)
            << "Static/off rebalance must not pay GPU-state or position checks when maintenance env is set.";
        EXPECT_LT(maintenance_dynamic_gate, maintenance_controller_gate)
            << "Static/off rebalance must not walk controller/overlay state when maintenance env is set.";
        EXPECT_NE(debug_env.find("bool device_rebalance_graph_controller = true"),
                  std::string::npos)
            << "The graph-native controller should be the default homogeneous GPU rebalance path.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_GRAPH_CONTROLLER"),
                  std::string::npos);
        EXPECT_NE(debug_env.find("bool device_rebalance_maintenance_graph = true"),
                  std::string::npos)
            << "The graph-native controller should default to the async maintenance lane.";
        EXPECT_NE(debug_env.find("bool device_rebalance_decode_apply_poll = false"),
                  std::string::npos)
            << "Async maintenance should not add a per-token decode apply poll by default.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_DECODE_APPLY_POLL"),
                  std::string::npos);
        EXPECT_EQ(debug_env.find("device_rebalance_captured_transfer_stream"),
                  std::string::npos)
            << "Captured maintenance transfer stream selection is no longer configurable.";
        EXPECT_EQ(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_CAPTURED_TRANSFER_STREAM"),
                  std::string::npos)
            << "The obsolete captured transfer stream A/B knob should not be documented.";
        EXPECT_EQ(debug_env.find("LLAMINAR_MOE_REBALANCE_DEBUG_SYNC"), std::string::npos)
            << "Temporary crash-localization syncs must not become a permanent environment contract.";
        EXPECT_NE(debug_env.find("bool device_rebalance_collect_load_stats = false"),
                  std::string::npos)
            << "Projected load-spread diagnostics must not run by default.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_LOAD_STATS"),
                  std::string::npos)
            << "The explicit env knob for projected load-spread diagnostics should stay documented.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_TRANSFER"),
                  std::string::npos)
            << "The LLEP useful-work transfer gate should stay documented.";
        EXPECT_NE(debug_env.find("bool allow_legacy_collective_rebalance_transfer = false"),
                  std::string::npos)
            << "The obsolete fixed-arena transfer fallback must stay disabled by default.";
        EXPECT_NE(dgo.find("buildDeviceMoERebalanceMaintenanceGraph(\n                    state_.device_id"),
                  std::string::npos);
        EXPECT_NE(dgo.find("DeviceMoERebalanceMaintenanceGraphKind::MetadataAndPayload"),
                  std::string::npos)
            << "The maintenance scheduler must have a metadata/payload graph variant.";
        EXPECT_NE(dgo.find("domain_decision.launchPayloadGraph()"),
                  std::string::npos)
            << "Payload graph replay should be scheduled only through the shared payload-decision predicate.";
        EXPECT_NE(host_rendezvous.find("payload_bucket_slots != 0u"),
                  std::string::npos);
        EXPECT_NE(host_rendezvous.find("payload_edge_mask != 0ULL"),
                  std::string::npos)
            << "Payload graph replay must require directed edges so no-work waves do not run payload bodies.";
        EXPECT_NE(dgo.find("device_moe_rebalance_decode_tokens_seen_"), std::string::npos)
            << "Maintenance should run on a rebalance window, not every decode token.";
        EXPECT_NE(dgo.find("active_cache.segment_cache.ensureCaptureStream"), std::string::npos)
            << "Maintenance must use an explicit graph-capture stream.";
        EXPECT_NE(dgo.find("peekPendingLogitsStream(PendingLogitsStreamRole::MainDecode)"),
                  std::string::npos)
            << "Maintenance must order behind the actual main-decode replay stream when sync is deferred.";
        EXPECT_NE(dgo.find("gpu_ctx->insertStreamDependency(maintenance_stream, producer_stream)"),
                  std::string::npos)
            << "Maintenance should use a GPU-side event edge, not a host synchronize.";
        EXPECT_NE(dgo.find("policy.defer_final_sync = true"), std::string::npos)
            << "Maintenance replay should not force a host sync at the launch boundary.";
        EXPECT_NE(dgo.find("tryLaunchCapturedMoERebalanceMaintenanceGraphDirect"), std::string::npos)
            << "Steady maintenance replay should bypass generic decode replay bookkeeping once captured.";
        EXPECT_NE(dgo.find("segment.capture->launch()"), std::string::npos)
            << "The steady maintenance fast path should enqueue the captured graph executable directly.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_direct_replay\""), std::string::npos)
            << "Perfstats must expose when the maintenance graph uses the direct replay fast path.";
        EXPECT_NE(dgo.find("completion_event_in_flight"), std::string::npos)
            << "Maintenance replay must track in-flight work without host synchronization.";
        {
            const size_t inflight_gate =
                dgo.find("const bool has_inflight_maintenance");
            const size_t off_cadence_idle_return =
                dgo.find("!should_launch && !has_inflight_maintenance");
            const size_t plan_event_query =
                dgo.find("gpu_ctx->queryEventChecked(cache.completion_event.get(), previous_wave_ready)");
            const size_t off_cadence_after_drain_return =
                dgo.find("if (!launch_metadata_payload_graph && !should_launch)");
            ASSERT_NE(inflight_gate, std::string::npos);
            ASSERT_NE(off_cadence_idle_return, std::string::npos);
            ASSERT_NE(plan_event_query, std::string::npos);
            ASSERT_NE(off_cadence_after_drain_return, std::string::npos);
            EXPECT_LT(inflight_gate, off_cadence_idle_return);
            EXPECT_LT(off_cadence_idle_return, plan_event_query)
                << "Only completely idle off-cadence tokens may return before querying completed maintenance.";
            EXPECT_LT(plan_event_query, off_cadence_after_drain_return)
                << "A completed probe with payload work must be able to launch the metadata/payload graph before the next scheduled probe period.";
            EXPECT_NE(dgo.find("\"scheduled_plan_launch\""), std::string::npos)
                << "Perfstats should distinguish an off-cadence completion drain from a scheduled new plan launch.";
        }
        EXPECT_NE(dgo.find("gpu_ctx->queryEventChecked(cache.completion_event.get(), previous_wave_ready)"),
                  std::string::npos)
            << "Maintenance replay must use a nonblocking event query before launching another wave.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_skipped_inflight\""), std::string::npos)
            << "Skipped in-flight maintenance windows must be visible in perf counters.";
        EXPECT_NE(dgo.find("gpu_ctx->recordEventChecked(active_cache.completion_event.get(), maintenance_stream)"),
                  std::string::npos)
            << "Maintenance graph completion must be recorded on the explicit maintenance stream.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_launches\""), std::string::npos);
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_stream_path\""), std::string::npos)
            << "Measured maintenance replay perfstats must expose transfer path usage.";
        EXPECT_EQ(dgo.find("stage_stream_fallback"), std::string::npos)
            << "Orchestrator perfstats must not retain the obsolete ROCm fallback concept.";
        EXPECT_EQ(dgo.find("device_rebalance_captured_transfer_stream"), std::string::npos)
            << "The captured transfer stream A/B env knob should be removed now that aux is mandatory.";
        EXPECT_EQ(dgo.find("\"captured_transfer_stream_mode\""), std::string::npos)
            << "Perfstats should not tag a removed stream mode.";
        EXPECT_EQ(device_rebalance_stage.find("captured_transfer_stream_mode"), std::string::npos)
            << "Perfstats should not tag a removed stream mode.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_host_window\""), std::string::npos)
            << "Perfstats must time the host-scheduled maintenance boundary while this path exists.";
        EXPECT_NE(dgo.find("device_rebalance_maintenance_slack_tokens"), std::string::npos)
            << "Maintenance graph launch cadence must include slack so underfilled rolling-wave histograms do not replay uselessly.";
        EXPECT_NE(debug_env.find("int device_rebalance_maintenance_slack_tokens = 1"),
                  std::string::npos)
            << "Device-side maintenance should default to one-token slack to avoid near-full histogram no-op replays.";
        EXPECT_NE(debug_env.find("int device_rebalance_min_maintenance_period_tokens = 512"),
                  std::string::npos)
            << "Device-side maintenance should default to the shared CUDA2/ROCm2 cadence floor.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_MIN_MAINTENANCE_PERIOD_TOKENS"),
                  std::string::npos)
            << "The maintenance cadence floor must stay operator-tunable.";
        EXPECT_NE(dgo.find("device_rebalance_min_maintenance_period_tokens"), std::string::npos)
            << "The scheduler must apply the device-side maintenance cadence floor.";
        EXPECT_NE(debug_env.find("int device_rebalance_initial_maintenance_period_tokens = 321"),
                  std::string::npos)
            << "Device-side maintenance should default to the delayed first replay that stabilized ROCm without regressing clean CUDA.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_INITIAL_MAINTENANCE_PERIOD_TOKENS"),
                  std::string::npos)
            << "The early first maintenance period must stay operator-tunable.";
        EXPECT_NE(dgo.find("device_rebalance_initial_maintenance_period_tokens"), std::string::npos)
            << "The scheduler must honor the optional early first maintenance period.";
        EXPECT_NE(dgo.find("\"initial_maintenance_period_tokens\""), std::string::npos)
            << "Perfstats tags must expose the early first maintenance period.";
        EXPECT_NE(dgo.find("\"min_maintenance_period_tokens\""), std::string::npos)
            << "Perfstats tags must expose the maintenance cadence floor.";
        EXPECT_NE(dgo.find("\"requested_launch_period\""), std::string::npos)
            << "Perfstats tags must expose the window+slack period before the cadence floor.";
        EXPECT_NE(dgo.find("\"maintenance_slack_tokens\""), std::string::npos)
            << "Perfstats tags must expose the launch-cadence slack.";
        EXPECT_NE(dgo.find("\"launch_period\""), std::string::npos)
            << "Perfstats tags must expose the maintenance launch period.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_event_query\""), std::string::npos);
        EXPECT_NE(dgo.find("\"device_maintenance_graph_replay_enqueue\""), std::string::npos);
        EXPECT_NE(dgo.find("\"device_maintenance_graph_completion_record\""), std::string::npos);
        EXPECT_NE(dgo.find("createTimingEvent"), std::string::npos)
            << "Maintenance perfstats must use backend timing events to measure device work, not host timers only.";
        EXPECT_NE(dgo.find("eventElapsedTimeMs"), std::string::npos)
            << "Maintenance perfstats must read GPU elapsed time at the existing diagnostic completion boundary.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_gpu_elapsed\""), std::string::npos)
            << "Perfstats must expose maintenance graph GPU elapsed time so we can separate host overhead from device work.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_gpu_timing_record_failures\""), std::string::npos)
            << "Timing event record failures must be visible instead of silently losing device elapsed stats.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_gpu_timing_read_failures\""), std::string::npos)
            << "Timing event elapsed-read failures must be visible instead of silently losing device elapsed stats.";
        EXPECT_NE(dgo.find("exportCompletedDeviceMoERebalanceMaintenanceStats"),
                  std::string::npos)
            << "Completed maintenance waves must export policy/apply diagnostics while perfstats are enabled.";
        EXPECT_NE(dgo.find("\"device_rebalance_status_readback\""), std::string::npos)
            << "Diagnostic status export must time its tiny explicit-stream D2H copy.";
        EXPECT_NE(dgo.find("\"device_rebalance_policy_load_imbalance_ratio\""), std::string::npos)
            << "Perfstats must expose pre/post projected expert-load imbalance.";
        EXPECT_NE(dgo.find("\"device_rebalance_policy_load_spread_units\""), std::string::npos)
            << "Perfstats must expose absolute imbalance units so throughput loss can be regressed against load spread.";
        EXPECT_NE(dgo.find("\"device_rebalance_policy_load_spread_delta_units\""), std::string::npos)
            << "Perfstats must quantify how much each accepted rebalance wave improves projected spread.";
        EXPECT_NE(dgo.find("\"device_rebalance_candidate_arrivals_considered\""), std::string::npos)
            << "Perfstats must expose how much transfer opportunity the policy evaluated.";
        EXPECT_NE(dgo.find("\"device_rebalance_candidate_arrivals_pruned_by_count_bound\""),
                  std::string::npos)
            << "Perfstats must expose cheap count-bound pruning before full projected-spread evaluation.";
        EXPECT_NE(dgo.find("\"device_rebalance_skipped_busy_wave\""), std::string::npos)
            << "Perfstats must distinguish a busy transfer/apply wave from histogram cadence.";
        EXPECT_NE(dgo.find("\"device_rebalance_skipped_histogram_not_ready\""), std::string::npos)
            << "Perfstats must distinguish histogram cadence from a busy transfer/apply wave.";
        EXPECT_NE(dgo.find("\"device_rebalance_window_ready_slots\""), std::string::npos)
            << "Perfstats must expose the observed histogram slots at the readiness gate.";
        EXPECT_NE(dgo.find("\"device_rebalance_window_required_slots\""), std::string::npos)
            << "Perfstats must expose the required histogram slots at the readiness gate.";
        EXPECT_NE(dgo.find("\"device_rebalance_accepted_load_spread_improvement_total\""), std::string::npos)
            << "Perfstats must expose accepted projected imbalance improvement.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_improved_dispatches\""), std::string::npos)
            << "Perfstats must prove whether the decode router used hot-cache replicas to reduce dispatch imbalance.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_load_spread_improvement_total\""), std::string::npos)
            << "Perfstats must quantify router-side dispatch spread improvement from hot-cache replicas.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_active_dispatches\""), std::string::npos)
            << "Perfstats must expose hot-cache-capable decode dispatches after replicas exist.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_miss_dispatches\""), std::string::npos)
            << "Perfstats must expose hot-cache misses where selected top-k did not touch a replicated expert.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_selected_expert_slots\""), std::string::npos)
            << "Perfstats must expose the selected top-k slot denominator for hot-cache hit economics.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_replicated_selected_expert_slots\""), std::string::npos)
            << "Perfstats must expose selected top-k slots that were backed by hot-cache replicas.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_miss_ratio\""), std::string::npos)
            << "Perfstats must expose how often active hot-cache dispatches miss the replicated experts.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_selected_replica_slot_ratio\""), std::string::npos)
            << "Perfstats must expose what fraction of selected expert slots could consume the hot cache.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_spread_improvement_ratio\""), std::string::npos)
            << "Perfstats must expose whether hot-cache dispatch is materially reducing spread.";
        EXPECT_NE(dgo.find("\"device_rebalance_skipped_low_router_benefit\""), std::string::npos)
            << "Perfstats must expose waves rejected because prior hot-cache routing did not pay off.";
        EXPECT_NE(dgo.find("\"device_rebalance_skipped_post_load_spread_ceiling\""), std::string::npos)
            << "Perfstats must expose transfer waves rejected because residual projected imbalance stayed too high.";
        EXPECT_NE(dgo.find("\"device_rebalance_post_wave_load_spread_fraction\""), std::string::npos)
            << "Perfstats must expose the controller-visible residual imbalance signal used by the policy gate.";
        EXPECT_NE(dgo.find("\"device_rebalance_router_hot_cache_spread_improvement_per_requested_payload_slot\""), std::string::npos)
            << "Perfstats must expose realized router benefit in the same units used by the payload-slot gate.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_payload_gathered_capacity_bytes\""), std::string::npos)
            << "Perfstats must account for fixed collective payload bytes paid by each rebalance batch.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_selected_payload_gathered_capacity_bytes\""), std::string::npos)
            << "Perfstats must separately account for the device-selected payload bucket bytes.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_captured_payload_slack_bytes\""), std::string::npos)
            << "Perfstats must expose captured graph payload slack beyond the selected bucket.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_useful_payload_bytes\""), std::string::npos)
            << "Perfstats must account for useful expert bytes that actually land in transfer slots.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_current_copied_arrivals\""), std::string::npos)
            << "Perfstats must expose the copied-arrival count charged to the current replay.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_current_wave_index\""), std::string::npos)
            << "Perfstats must expose which wave, if any, was charged for current transfer bytes.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_current_wave_matched\""), std::string::npos)
            << "Perfstats must expose whether current_wave_index is meaningful instead of exporting UINT32_MAX sentinels.";
        EXPECT_EQ(dgo.find("uint32_t transfer_wave_index = UINT32_MAX"), std::string::npos)
            << "Perfstats must not export UINT32_MAX as a numeric wave index for participants with no matched wave.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_current_applied_arrivals\""), std::string::npos)
            << "Perfstats must expose current-wave applied arrivals from persistent wave state.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_current_applied_layer_count\""), std::string::npos)
            << "Perfstats must expose current-wave applied layers from persistent wave state.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_current_payload_bucket_slots\""), std::string::npos)
            << "Perfstats must expose the bucket selected by the device policy for the current wave.";
        EXPECT_NE(dgo.find("transfer_applied_arrivals"), std::string::npos)
            << "Current apply accounting must use persistent wave state, not transient apply_status.";
        EXPECT_NE(dgo.find("transfer_copied_arrivals"), std::string::npos)
            << "Useful payload accounting must use persistent wave copied-arrival counters.";
        EXPECT_NE(dgo.find("wave.epoch != status.last_epoch"), std::string::npos)
            << "Useful payload accounting must match persistent wave state to the current maintenance status epoch.";
        EXPECT_NE(dgo.find("header.epoch != 0u ? header.epoch : wave.epoch"), std::string::npos)
            << "Diagnostic command epoch may fall back to the persistent wave epoch after apply clears the header.";
        EXPECT_EQ(dgo.find("header_count == 0u ||"), std::string::npos)
            << "Applied waves clear command headers, so useful payload accounting must not reject a wave solely because the header count is zero.";
        EXPECT_NE(dgo.find("status.windows_applied != 0u"), std::string::npos)
            << "Skipped/no-op maintenance windows must not inherit stale useful-payload bytes from older waves.";
        EXPECT_EQ(dgo.find("wave_copied_arrivals_total > 0"), std::string::npos)
            << "Useful payload accounting must not sum stale copied-arrival counters across all waves.";
        {
            const size_t transfer_estimate = dgo.find("estimateDeviceMoERebalanceTransferCost(");
            ASSERT_NE(transfer_estimate, std::string::npos);
            const std::string transfer_estimate_call = dgo.substr(transfer_estimate, 700);
            EXPECT_NE(transfer_estimate_call.find("transfer_copied_arrivals"), std::string::npos)
                << "Do not derive useful payload bytes from transient apply_status after ready-wave apply resets it.";
            EXPECT_NE(transfer_estimate_call.find("transfer_payload_bucket_slots"), std::string::npos)
                << "Transfer accounting must charge selected bucket bytes separately from captured graph capacity.";
            EXPECT_EQ(transfer_estimate_call.find("apply_status.copied_arrivals"), std::string::npos)
                << "Useful payload accounting must not regress to the transient apply-status copy counter.";
        }
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_payload_collective_utilization_ratio\""), std::string::npos)
            << "Perfstats must expose whether fixed payload sidebands are mostly empty capacity.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_selected_payload_utilization_ratio\""), std::string::npos)
            << "Perfstats must expose useful payload utilization against the selected bucket.";
        EXPECT_NE(dgo.find("\"device_rebalance_payload_source_participant_mask\""), std::string::npos)
            << "Perfstats must expose which participants sourced non-empty payload slots.";
        EXPECT_NE(dgo.find("\"device_rebalance_payload_destination_participant_mask\""), std::string::npos)
            << "Perfstats must expose which participants receive non-empty payload slots.";
        EXPECT_NE(dgo.find("\"device_rebalance_payload_edge_mask_low32\""), std::string::npos)
            << "Perfstats must expose the low edge-mask bits without relying on lossy u64-as-double counters.";
        EXPECT_NE(dgo.find("\"device_rebalance_payload_edge_mask_high32\""), std::string::npos)
            << "Perfstats must expose the high edge-mask bits for the full 8x8 participant domain.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_current_payload_active_edge_count\""), std::string::npos)
            << "Perfstats must expose whether the current payload replay is one-sided, bidirectional, or broader.";
        EXPECT_NE(graph_builder.find("uint64_t payload_edge_mask = 0"), std::string::npos)
            << "Maintenance graph builders must expose payload edge-mask variants explicitly.";
        EXPECT_NE(dgo.find("pending_payload_edge_mask"), std::string::npos)
            << "The maintenance scheduler must carry the completed device-planned edge mask into payload graph launch.";
        EXPECT_NE(dgo.find("device_moe_rebalance_maintenance_payload_graphs_.try_emplace"),
                  std::string::npos)
            << "Payload maintenance graph replay must key captured variants by directed edge mask.";
        EXPECT_EQ(dgo.find("active_cache.payload_edge_mask != pending_payload_edge_mask"),
                  std::string::npos)
            << "Payload graph variants should be selected by key instead of evicting another captured edge mask.";
        EXPECT_NE(device_rebalance_stage.find("groupedP2PRawOnStream"), std::string::npos)
            << "Directed payload graphs must use grouped NCCL/RCCL send/recv instead of allgathering empty participant buckets.";
        EXPECT_NE(device_rebalance_stage.find("allgatherRawOnStream"), std::string::npos)
            << "The payload stage should retain the allgather path as an explicit metadata-only zero-edge path.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_selected_payload_transport_capacity_bytes\""), std::string::npos)
            << "Perfstats must price directed payload transport by active edge count, not only by allgather capacity.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_selected_payload_transport_utilization_ratio\""), std::string::npos)
            << "Perfstats must expose useful payload utilization against directed transport bytes.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_transport_bytes_per_accepted_spread_unit\""), std::string::npos)
            << "Policy tuning needs directed transport bytes per projected spread unit.";
        EXPECT_NE(dgo.find("\"device_rebalance_transfer_collective_bytes_per_accepted_spread_unit\""),
                  std::string::npos)
            << "Perfstats must let the policy compare transfer cost against projected imbalance improvement.";
        EXPECT_NE(dgo.find("DeviceMoERebalanceMaintenanceOutcome"), std::string::npos)
            << "The maintenance scheduler must consume completed device status, not guess no-work outcomes.";
        EXPECT_NE(dgo.find("outcome->useful_work"), std::string::npos)
            << "Completed maintenance exports must identify whether the replay produced useful work.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_no_work_completions\""), std::string::npos)
            << "Perfstats must expose zero-work maintenance completions that can trigger backoff.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_skipped_no_work_backoff\""), std::string::npos)
            << "Perfstats must prove when the scheduler avoids replaying an empty payload graph.";
        EXPECT_NE(dgo.find("device_rebalance_no_work_backoff_periods"), std::string::npos)
            << "No-work backoff must be controlled by DebugEnv while bucket graph scheduling is still interim.";
        EXPECT_NE(dgo.find("\"no_work_backoff_effective_periods\""), std::string::npos)
            << "No-work backoff perfstats must expose the progressive effective skip count.";
        EXPECT_NE(dgo.find("std::min<uint64_t>("), std::string::npos)
            << "Repeated no-work probe completions should progressively increase backoff instead of probing every other window.";
        EXPECT_NE(dgo.find("\"device_rebalance_wave_applied_arrivals_total\""), std::string::npos)
            << "Perfstats must expose wave aggregate apply counters that survive apply-status reuse.";
        EXPECT_NE(dgo.find("\"device_rebalance_selected_replicas\""), std::string::npos)
            << "Perfstats must make a no-op policy visible.";
        EXPECT_NE(dgo.find("\"device_rebalance_skipped_no_improvement\""), std::string::npos)
            << "Perfstats must distinguish no-op policy gates from missing transfer sources.";
        EXPECT_NE(dgo.find("\"device_rebalance_apply_changed_layers\""), std::string::npos)
            << "Perfstats must show whether decode-side apply actually changed runtime placement.";
        EXPECT_NE(dgo.find("\"device_rebalance_apply_post_apply_multi_resident_experts\""), std::string::npos)
            << "Perfstats must show whether apply produced hot-cache-visible multi-resident experts.";
        EXPECT_NE(dgo.find("\"post_apply_multi_resident_experts\""), std::string::npos)
            << "Trace JSON must expose post-apply hot-cache visibility for CUDA/ROCm diagnosis.";
        EXPECT_NE(dgo.find("MoEDeviceRebalanceStage::WS_STATUS"), std::string::npos);
        EXPECT_NE(dgo.find("MoEDeviceRebalanceStage::WS_COMMAND_HEADER"), std::string::npos);
        EXPECT_NE(dgo.find("MoEDeviceRebalanceStage::WS_CONTROLLER_STATE"), std::string::npos);
        EXPECT_NE(dgo.find("backend->deviceToHostOnStream"), std::string::npos)
            << "Status export should use explicit-stream D2H, not an implicit host sync.";
        EXPECT_NE(dgo.find("backend->synchronizeStream(maintenance_stream, device_ordinal)"),
                  std::string::npos)
            << "The diagnostic D2H handoff must synchronize only the explicit maintenance stream.";

        const size_t generate_start = runner.find("GenerationResult OrchestrationRunner::generate(");
        ASSERT_NE(generate_start, std::string::npos);
        const size_t generate_end = runner.find("bool OrchestrationRunner::maybeApplyMoERebalance()",
                                                generate_start);
        ASSERT_NE(generate_end, std::string::npos);
        const std::string generate_body = runner.substr(generate_start, generate_end - generate_start);

        const size_t device_side_fast_path =
            generate_body.find("const bool device_side_moe_rebalance");
        const size_t maintenance_guard =
            generate_body.find("!device_side_moe_rebalance && !maybeApplyMoERebalance()");
        const size_t epilogue_device_side_gate =
            generate_body.rfind("if (device_side_moe_rebalance)");
        const size_t epilogue_host_publish =
            generate_body.rfind("publishPendingMoERebalanceUpdate()");
        ASSERT_NE(device_side_fast_path, std::string::npos);
        ASSERT_NE(maintenance_guard, std::string::npos)
            << "Device-side rebalance must not call host decode-boundary maintenance after every token.";
        ASSERT_NE(epilogue_device_side_gate, std::string::npos);
        ASSERT_NE(epilogue_host_publish, std::string::npos);
        EXPECT_LT(device_side_fast_path, maintenance_guard);
        EXPECT_LT(epilogue_device_side_gate, epilogue_host_publish)
            << "The decode epilogue must not drain host pending publishes in device-side mode.";

        EXPECT_NE(chat.find("runner_.usesDeviceSideMoERebalanceController()"), std::string::npos);
        const size_t chat_rebalance_helper =
            chat.find("bool runChatMoERebalanceMaintenance(");
        ASSERT_NE(chat_rebalance_helper, std::string::npos)
            << "Chat serving should funnel host MoE maintenance through one request-boundary helper.";
        const size_t chat_rebalance_helper_end =
            chat.find("    }", chat_rebalance_helper);
        ASSERT_NE(chat_rebalance_helper_end, std::string::npos);
        const std::string chat_rebalance_helper_body =
            chat.substr(chat_rebalance_helper,
                        chat_rebalance_helper_end - chat_rebalance_helper);
        const size_t chat_helper_device_gate =
            chat_rebalance_helper_body.find("if (device_side_moe_rebalance)");
        const size_t chat_helper_host_maintenance =
            chat_rebalance_helper_body.find("runner.maybeApplyMoERebalance()");
        ASSERT_NE(chat_helper_device_gate, std::string::npos);
        ASSERT_NE(chat_helper_host_maintenance, std::string::npos);
        EXPECT_LT(chat_helper_device_gate, chat_helper_host_maintenance)
            << "Chat serving must skip host rebalance maintenance in device-side mode.";
        EXPECT_NE(chat.find("last_decode_window_had_moe_maintenance"), std::string::npos)
            << "Chat serving must not skip the final decode window's MoE maintenance.";
        EXPECT_NE(chat.find("runChatMoERebalanceMaintenance(runner_, device_side_moe_rebalance)"),
                  std::string::npos)
            << "Chat decode loops and epilogues must use the shared device-side gate.";
        const size_t benchmark_post_warmup =
            benchmark.find("benchmark.setPostWarmupCallback");
        ASSERT_NE(benchmark_post_warmup, std::string::npos);
        const size_t benchmark_device_side_gate =
            benchmark.find("orch_runner->usesDeviceSideMoERebalanceController()", benchmark_post_warmup);
        const size_t benchmark_host_apply =
            benchmark.find("orch_runner->applyMoERebalanceWithReplicas", benchmark_post_warmup);
        ASSERT_NE(benchmark_device_side_gate, std::string::npos);
        ASSERT_NE(benchmark_host_apply, std::string::npos);
        EXPECT_LT(benchmark_device_side_gate, benchmark_host_apply)
            << "Benchmark post-warmup setup must not call host publish/apply when the graph controller owns rebalance.";
        EXPECT_NE(benchmark.find("device-side graph controller owns publish/apply"), std::string::npos);
        EXPECT_NE(runner.find("drainPendingMoERebalanceBeforeCacheClear()"), std::string::npos)
            << "clearCache() must drain prepared MoE publishes before request/session reset.";
        EXPECT_NE(qwen36_parity.find("applyHostMoERebalanceIfNeeded"), std::string::npos);
        EXPECT_NE(qwen36_parity.find("runner.usesDeviceSideMoERebalanceController()"), std::string::npos)
            << "Parity direct decode loops must skip host rebalance maintenance in device-side mode.";
        EXPECT_EQ(qwen36_parity.find("runner->maybeApplyMoERebalance()"), std::string::npos)
            << "Qwen3.6 parity loops should not bypass the device-side maintenance gate.";
        EXPECT_NE(server_e2e.find("prefix-cache-rebalance-clear-probe"), std::string::npos)
            << "HTTP prefix-cache + MoE rebalance needs a dedicated server regression.";
        EXPECT_NE(server_e2e.find("run_prefix_cache_rebalance_clear_probe"), std::string::npos);
        EXPECT_NE(server_e2e.find("LLAMINAR_MOE_GPU_CACHE_EXPERTS_PER_LAYER"), std::string::npos)
            << "The HTTP regression should deterministically leave a prepared LocalTP MoE publish for cleanup.";
        EXPECT_NE(server_e2e.find("clear_cache_pending_publish_drains"), std::string::npos)
            << "The HTTP regression must assert the host request cleanup drain counter.";
        EXPECT_NE(server_e2e.find("device_maintenance_graph_request_reset_exports"), std::string::npos)
            << "The HTTP regression must accept the device-side request-reset export counter.";
        EXPECT_NE(server_e2e.find("mode != \"llep\""), std::string::npos)
            << "LLEP prefix-cache clear probes should not require dynamic publish drain/export counters.";
        EXPECT_NE(server_e2e.find("request-clear-cache"), std::string::npos)
            << "The HTTP regression must match the live-state mutation operation emitted by request-boundary cache clears.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DecodeCaptureFenceDrainsGraphStableMoERuntimeBeforeRendezvous)
    {
        const fs::path root = findRepoRoot();
        const fs::path dgo_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        const fs::path dgo_header_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h";
        ASSERT_TRUE(fs::exists(dgo_path)) << dgo_path;
        ASSERT_TRUE(fs::exists(dgo_header_path)) << dgo_header_path;

        const std::string dgo = readFile(dgo_path);
        const std::string dgo_header = readFile(dgo_header_path);
        ASSERT_FALSE(dgo.empty()) << dgo_path;
        ASSERT_FALSE(dgo_header.empty()) << dgo_header_path;

        EXPECT_NE(dgo_header.find("synchronizeGraphStableMoERuntimeBeforeDecodeCapture"),
                  std::string::npos)
            << "DeviceGraphOrchestrator must expose the decode-capture MoE runtime fence.";

        const size_t fence_start =
            dgo.find("bool DeviceGraphOrchestrator::synchronizeGraphStableMoERuntimeBeforeDecodeCapture(");
        ASSERT_NE(fence_start, std::string::npos);
        const size_t fence_end =
            dgo.find("bool DeviceGraphOrchestrator::waitAtDecodeGraphCaptureBoundary(",
                     fence_start);
        ASSERT_NE(fence_end, std::string::npos);
        const std::string fence_body = dgo.substr(fence_start, fence_end - fence_start);
        EXPECT_NE(fence_body.find("usesGraphStableGpuMoERebalance()"),
                  std::string::npos)
            << "The decode-capture fence should be limited to graph-stable MoE runtime-table movement.";
        EXPECT_NE(fence_body.find("gpu_ctx->synchronizeChecked()"),
                  std::string::npos)
            << "The decode-capture fence must fail fast on asynchronous CUDA/HIP errors.";
        EXPECT_NE(fence_body.find("decode_capture_boundary_moe_runtime_device_sync"),
                  std::string::npos)
            << "The fence must emit PerfStats so e2e can prove it was exercised.";

        const size_t boundary_start =
            dgo.find("bool DeviceGraphOrchestrator::waitAtDecodeGraphCaptureBoundary(");
        ASSERT_NE(boundary_start, std::string::npos);
        const size_t boundary_end =
            dgo.find("// =========================================================================", boundary_start);
        ASSERT_NE(boundary_end, std::string::npos);
        const std::string boundary_body =
            dgo.substr(boundary_start, boundary_end - boundary_start);
        const size_t fence_call =
            boundary_body.find("synchronizeGraphStableMoERuntimeBeforeDecodeCapture(");
        const size_t rendezvous_call =
            boundary_body.find("local_tp->graphCaptureBoundaryRendezvous(");
        ASSERT_NE(fence_call, std::string::npos);
        ASSERT_NE(rendezvous_call, std::string::npos);
        EXPECT_LT(fence_call, rendezvous_call)
            << "Graph-stable MoE runtime movement must be drained before any participant enters decode graph capture.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, SharedExpertGatePublishesGpuWritesWithStageStreamEvent)
    {
        const fs::path root = findRepoRoot();
        const fs::path stage_path = root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        ASSERT_TRUE(fs::exists(stage_path)) << stage_path;

        const std::string contents = readFile(stage_path);
        ASSERT_FALSE(contents.empty()) << stage_path;

        const size_t execute_start = contents.find("bool SharedExpertGateStage::execute(");
        ASSERT_NE(execute_start, std::string::npos);
        const size_t execute_end = contents.find("IMoEKernel *SharedExpertGateStage::ensureMoEKernel()",
                                                 execute_start);
        ASSERT_NE(execute_end, std::string::npos);
        const std::string execute_body = contents.substr(execute_start, execute_end - execute_start);

        const size_t fused_gate_call = execute_body.find("kernel->sharedExpertGateAddFromTensors(");
        const size_t fused_publish = execute_body.find("markGpuTensorWritten(params_.shared_output, params_.device_id, gpuStream())",
                                                       fused_gate_call);
        const size_t fused_combined_publish = execute_body.find("markGpuTensorWritten(params_.combined_output, params_.device_id, gpuStream())",
                                                               fused_gate_call);
        const size_t gate_call = execute_body.find("kernel->sharedExpertGateFromTensors(");
        const size_t publish = execute_body.find("markGpuTensorWritten(params_.shared_output, params_.device_id, gpuStream())",
                                                 gate_call);
        const size_t upload_fallback = execute_body.find("params_.shared_output->needsUpload()", publish);
        ASSERT_NE(fused_gate_call, std::string::npos);
        ASSERT_NE(fused_publish, std::string::npos);
        ASSERT_NE(fused_combined_publish, std::string::npos);
        ASSERT_NE(gate_call, std::string::npos);
        ASSERT_NE(publish, std::string::npos);
        ASSERT_NE(upload_fallback, std::string::npos);
        EXPECT_LT(fused_gate_call, fused_publish);
        EXPECT_LT(fused_gate_call, fused_combined_publish);
        EXPECT_LT(gate_call, publish);
        EXPECT_LT(publish, upload_fallback);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, SharedExpertGroupedDecodePublishesOutputBeforeReturning)
    {
        const fs::path root = findRepoRoot();
        const fs::path stage_path = root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        ASSERT_TRUE(fs::exists(stage_path)) << stage_path;

        const std::string contents = readFile(stage_path);
        ASSERT_FALSE(contents.empty()) << stage_path;

        const size_t execute_start = contents.find("bool SharedExpertFFNStage::execute(");
        ASSERT_NE(execute_start, std::string::npos);
        const size_t execute_end = contents.find("IMoEKernel *SharedExpertFFNStage::ensureMoEKernel()",
                                                 execute_start);
        ASSERT_NE(execute_end, std::string::npos);
        const std::string execute_body = contents.substr(execute_start, execute_end - execute_start);

        const size_t grouped_decode = execute_body.find("if (grouped_decode_required)");
        ASSERT_NE(grouped_decode, std::string::npos);
        const size_t grouped_call = execute_body.find("tryGroupedDecode(kernel, d_model, intermediate)", grouped_decode);
        ASSERT_NE(grouped_call, std::string::npos);
        const size_t publish = execute_body.find("markGpuTensorWritten(params_.output, params_.device_id, gpuStream())",
                                                 grouped_decode);
        const size_t return_true = execute_body.find("return true;", grouped_decode);
        ASSERT_NE(publish, std::string::npos);
        ASSERT_NE(return_true, std::string::npos);
        EXPECT_LT(grouped_call, publish);
        EXPECT_LT(publish, return_true);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, Qwen35MoEResetRestoresInitialRuntimePlacementBanks)
    {
        const fs::path root = findRepoRoot();
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        const std::string contents = readFile(graph_path);
        ASSERT_FALSE(contents.empty()) << graph_path;

        const size_t reset_start = contents.find("void Qwen35MoEGraph::resetState()");
        ASSERT_NE(reset_start, std::string::npos);
        const size_t reset_end = contents.find("void Qwen35MoEGraph::resetPrefixCacheRuntimeStateWithoutSnapshot",
                                               reset_start);
        ASSERT_NE(reset_end, std::string::npos);
        const std::string reset_body = contents.substr(reset_start, reset_end - reset_start);

        EXPECT_NE(reset_body.find("restoreInitialRuntimeState()"), std::string::npos)
            << "Request-boundary reset must restore canonical MoE placement so "
               "portable prefix blocks replay suffix prefill under the same "
               "logical expert ownership used by an uncached full prefill.";
        EXPECT_EQ(reset_body.find("resetDecodeRuntimeState()"), std::string::npos)
            << "Clearing runtime placement banks during session reset makes "
               "the next GPU decode route fall back to host/top-k state or fail "
               "before graph-captured MoE decode can run.";
        EXPECT_EQ(reset_body.find("moe_graph_rebalance_bindings_.clear()"), std::string::npos)
            << "Graph-stable device rebalance bindings must survive ordinary request reset because "
               "the captured decode graph may survive and the maintenance graph still needs the same "
               "runtime-table and transfer-slot pointers.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, Qwen35MoEDecodeGraphAcceptsLiveDynamicRuntimeBank)
    {
        const fs::path root = findRepoRoot();
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        const std::string contents = readFile(graph_path);
        ASSERT_FALSE(contents.empty()) << graph_path;

        const size_t init_start = contents.find("bool initializeMaskedLocalDecodeRuntimeTable(");
        ASSERT_NE(init_start, std::string::npos);
        const size_t init_end = contents.find("bool initializeFullLocalDecodeRuntimeTable(",
                                               init_start);
        ASSERT_NE(init_end, std::string::npos);
        const std::string init_body = contents.substr(init_start, init_end - init_start);

        const size_t dynamic_check = init_body.find("runtimeTableHasUsableLiveDynamicDecodeBank(");
        const size_t epoch_refusal = init_body.find("refusing to replace active MoE decode runtime bank");
        ASSERT_NE(dynamic_check, std::string::npos)
            << "Phase-split LLEP/Dynamic decode graph rebuilds must accept an already-published "
               "live dynamic runtime bank instead of demanding the original static owner mask.";
        ASSERT_NE(epoch_refusal, std::string::npos);
        EXPECT_LT(dynamic_check, epoch_refusal)
            << "The live dynamic bank check must run before the fail-fast stale-bank guard.";
        EXPECT_NE(init_body.find("allow_existing_dynamic_bank"), std::string::npos)
            << "Callers must make live-bank reuse explicit; accepted banks are still validated "
               "before the stale-bank guard.";

        const size_t dynamic_validator =
            contents.find("bool runtimeTableHasUsableLiveDynamicDecodeBank(");
        ASSERT_NE(dynamic_validator, std::string::npos);
        const size_t dynamic_validator_end =
            contents.find("bool initializeMaskedLocalDecodeRuntimeTable(",
                          dynamic_validator);
        ASSERT_NE(dynamic_validator_end, std::string::npos);
        const std::string dynamic_body =
            contents.substr(dynamic_validator,
                            dynamic_validator_end - dynamic_validator);
        EXPECT_NE(dynamic_body.find("effective_resident_mask"), std::string::npos)
            << "Dynamic decode-bank validation must use effective residency: "
               "owner metadata is optional, but every expert still needs a routeable resident participant.";
        EXPECT_EQ(dynamic_body.find("desc.owner_participant < 0"), std::string::npos)
            << "Portable prefix-restored dynamic banks may contain ownerless non-local entries; "
               "the graph guard should require resident participants, not global owner metadata on every slot.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, MoEExpertStageAcceptsOwnerlessRuntimeResidency)
    {
        const fs::path root = findRepoRoot();
        const fs::path stage_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        ASSERT_TRUE(fs::exists(stage_path)) << stage_path;

        const std::string contents = readFile(stage_path);
        ASSERT_FALSE(contents.empty()) << stage_path;

        const size_t check_start =
            contents.find("bool MoEExpertComputeStage::runtimeTableHasActiveGroupedDecodeBank() const");
        ASSERT_NE(check_start, std::string::npos);
        const size_t check_end =
            contents.find("bool MoEExpertComputeStage::supportsRequestedRoutedAssignmentPolicy() const",
                          check_start);
        ASSERT_NE(check_end, std::string::npos);
        const std::string check_body = contents.substr(check_start, check_end - check_start);

        EXPECT_NE(check_body.find("effective_resident_mask"), std::string::npos)
            << "Runtime decode readiness should use effective residency, including optional owner bits.";
        EXPECT_NE(check_body.find("owner_participant < -1"), std::string::npos)
            << "Only the explicit unknown-owner sentinel should be accepted.";
        EXPECT_EQ(check_body.find("owner_participant < 0 ||"), std::string::npos)
            << "Portable prefix-restored dynamic banks may omit owner metadata for non-local entries.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, Qwen35MoEDecodeUsesRuntimeDescriptorsForTransferSlots)
    {
        const fs::path root = findRepoRoot();
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        const std::string contents = readFile(graph_path);
        ASSERT_FALSE(contents.empty()) << graph_path;

        const size_t assignment =
            contents.find("expert_params.runtime_decode_uses_mutable_descriptors =");
        ASSERT_NE(assignment, std::string::npos);
        const size_t assignment_end =
            contents.find("expert_params.runtime_decode_has_explicit_owner_metadata", assignment);
        ASSERT_NE(assignment_end, std::string::npos);
        const std::string assignment_body =
            contents.substr(assignment, assignment_end - assignment);

        EXPECT_NE(assignment_body.find("graphRebalanceDecodeUsesMutableDescriptors()"),
                  std::string::npos);
        EXPECT_NE(assignment_body.find("activeRuntimeBankUsesTransientLocalPayload(layer_idx)"),
                  std::string::npos)
            << "Decode after LLEP prefill may use transfer-slot descriptors that are only present "
               "in the runtime placement bank; static descriptor tables would silently route stale "
               "or missing expert payloads.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ROCmMoEHardResetClearsDeclaredGroupedWorkspace)
    {
        const fs::path root = findRepoRoot();
        const fs::path kernel_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(kernel_path)) << kernel_path;

        const std::string contents = readFile(kernel_path);
        ASSERT_FALSE(contents.empty()) << kernel_path;

        const size_t reset_start = contents.find("void ROCmMoEKernel::resetDynamicState()");
        ASSERT_NE(reset_start, std::string::npos);
        const size_t reset_end = contents.find("void ROCmMoEKernel::syncBlasStream()", reset_start);
        ASSERT_NE(reset_end, std::string::npos);
        const std::string reset_body = contents.substr(reset_start, reset_end - reset_start);

        EXPECT_NE(reset_body.find("clearWorkspaceScratchBindings()"), std::string::npos)
            << "Hard ROCm MoE kernel-dynamic reset must drop workspace-backed "
               "grouping scratch. Replay-preserving request reset stays safe by "
               "not calling KernelFactory::resetAllDynamicState().";
        EXPECT_NE(reset_body.find("grouped_down_desc_tables_.clear()"), std::string::npos)
            << "Hard reset must clear stale grouped down descriptor handles.";
        EXPECT_NE(reset_body.find("grouped_gateup_desc_tables_.clear()"), std::string::npos)
            << "Hard reset must clear stale grouped gate/up descriptor handles.";

        const size_t clear_start = contents.find("void ROCmMoEKernel::clearWorkspaceScratchBindings()");
        ASSERT_NE(clear_start, std::string::npos);
        const size_t clear_end = contents.find("ROCmMoEKernel::~ROCmMoEKernel()",
                                               clear_start);
        ASSERT_NE(clear_end, std::string::npos);
        const std::string clear_body = contents.substr(clear_start, clear_end - clear_start);

        EXPECT_NE(clear_body.find("d_grouped_gate_ptrs_ = nullptr"), std::string::npos)
            << "Unbinding workspace must clear cached grouped pointer arrays.";
        EXPECT_NE(clear_body.find("d_grouped_swiglu_int8_ = nullptr"), std::string::npos)
            << "Unbinding workspace must clear cached grouped scratch pointers.";
        EXPECT_NE(clear_body.find("grouped_decode_active_cap_ = 0"), std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, MoEWorkspaceRebindResetContractIsSymmetric)
    {
        const fs::path root = findRepoRoot();

        struct BackendCase
        {
            fs::path path;
            std::string class_name;
            std::vector<std::string> required_clear_members;
        };

        const std::vector<BackendCase> cases = {
            {
                "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp",
                "CUDAMoEKernel",
                {
                    "d_grouped_gateup_gate_partials_ = nullptr",
                    "d_grouped_down_partials_ = nullptr",
                    "d_decode_swiglu_int8_ = nullptr",
                },
            },
            {
                "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp",
                "ROCmMoEKernel",
                {
                    "d_grouped_gate_ptrs_ = nullptr",
                    "d_grouped_swiglu_int8_ = nullptr",
                    "grouped_decode_active_cap_ = 0",
                },
            },
        };

        for (const auto &backend : cases)
        {
            SCOPED_TRACE(backend.class_name);
            const fs::path kernel_path = root / backend.path;
            ASSERT_TRUE(fs::exists(kernel_path)) << kernel_path;

            const std::string contents = readFile(kernel_path);
            ASSERT_FALSE(contents.empty()) << kernel_path;

            const std::string bind_signature =
                "void " + backend.class_name + "::bindWorkspace(DeviceWorkspaceManager *workspace)";
            const size_t bind_start = contents.find(bind_signature);
            ASSERT_NE(bind_start, std::string::npos);
            const size_t bind_end = contents.find(
                "bool " + backend.class_name + "::bindWorkspaceBuffer",
                bind_start);
            ASSERT_NE(bind_end, std::string::npos);
            const std::string bind_body = contents.substr(bind_start, bind_end - bind_start);

            EXPECT_NE(bind_body.find("workspace->id()"), std::string::npos)
                << "Workspace ABA protection must use DeviceWorkspaceManager::id().";
            EXPECT_NE(bind_body.find("bound_workspace_id_"), std::string::npos);
            EXPECT_NE(bind_body.find("workspace_ == workspace && bound_workspace_id_"), std::string::npos)
                << "Same-pointer rebinding is only a no-op when the workspace id also matches.";

            const size_t early_return = bind_body.find("return;");
            const size_t bind_base =
                bind_body.find(backend.class_name == "CUDAMoEKernel"
                                   ? "CUDAKernelBase::bindWorkspace(workspace)"
                                   : "ROCmKernelBase::bindWorkspace(workspace)");
            const size_t assign_id = bind_body.find("bound_workspace_id_", early_return + 1);
            const size_t clear_bind = bind_body.find("clearWorkspaceScratchBindings()", early_return + 1);
            ASSERT_NE(early_return, std::string::npos);
            ASSERT_NE(bind_base, std::string::npos);
            ASSERT_NE(assign_id, std::string::npos);
            ASSERT_NE(clear_bind, std::string::npos);
            EXPECT_LT(early_return, bind_base);
            EXPECT_LT(bind_base, assign_id);
            EXPECT_LT(assign_id, clear_bind)
                << "Scratch caches must be invalidated after binding the new workspace identity.";
            EXPECT_NE(bind_body.find("rebindGroupedDescriptorTablesToWorkspace(\"bindWorkspace\")"),
                      std::string::npos)
                << "Workspace rebinding must preserve and reupload graph-stable grouped descriptor tables.";

            const std::string reset_signature =
                "void " + backend.class_name + "::resetDynamicState()";
            const size_t reset_start = contents.find(reset_signature);
            ASSERT_NE(reset_start, std::string::npos);
            const size_t reset_end = contents.find(
                backend.class_name == "CUDAMoEKernel"
                    ? "void CUDAMoEKernel::releaseDeviceBuffers()"
                    : "void ROCmMoEKernel::syncBlasStream()",
                reset_start);
            ASSERT_NE(reset_end, std::string::npos);
            const std::string reset_body = contents.substr(reset_start, reset_end - reset_start);

            EXPECT_NE(reset_body.find("clearWorkspaceScratchBindings()"), std::string::npos)
                << "Hard dynamic reset must drop graph-owned MoE scratch. "
                   "Replay-preserving request boundaries must avoid this reset path.";
            EXPECT_NE(reset_body.find("grouped_down_desc_tables_.clear()"), std::string::npos)
                << "Hard dynamic reset must clear grouped down descriptor registries.";
            EXPECT_NE(reset_body.find("grouped_gateup_desc_tables_.clear()"), std::string::npos)
                << "Hard dynamic reset must clear grouped gate/up descriptor registries.";

            const std::string clear_signature =
                "void " + backend.class_name + "::clearWorkspaceScratchBindings()";
            const size_t clear_start = contents.find(clear_signature);
            ASSERT_NE(clear_start, std::string::npos);
            const size_t clear_end = contents.find(
                backend.class_name == "CUDAMoEKernel"
                    ? "void CUDAMoEKernel::resetDynamicState()"
                    : "ROCmMoEKernel::~ROCmMoEKernel()",
                clear_start);
            ASSERT_NE(clear_end, std::string::npos);
            const std::string clear_body = contents.substr(clear_start, clear_end - clear_start);

            for (const std::string &member_reset : backend.required_clear_members)
            {
                EXPECT_NE(clear_body.find(member_reset), std::string::npos)
                    << "Workspace lifetime handoff must invalidate " << member_reset;
            }
            EXPECT_EQ(clear_body.find("grouped_down_desc_tables_.clear()"), std::string::npos)
                << "Workspace handoff must preserve host grouped down descriptors; release/reset may clear them explicitly.";
            EXPECT_EQ(clear_body.find("grouped_gateup_desc_tables_.clear()"), std::string::npos)
                << "Workspace handoff must preserve host grouped gate/up descriptors; release/reset may clear them explicitly.";
            EXPECT_NE(clear_body.find("table.device_descs = nullptr"), std::string::npos)
                << "Workspace handoff should invalidate only the device down descriptor pointer.";
            EXPECT_NE(clear_body.find("table.device_gate_descs = nullptr"), std::string::npos)
                << "Workspace handoff should invalidate only the device gate descriptor pointer.";
            EXPECT_NE(clear_body.find("table.device_up_descs = nullptr"), std::string::npos)
                << "Workspace handoff should invalidate only the device up descriptor pointer.";
            EXPECT_NE(clear_body.find("scratch_workspace_bound_ = false"), std::string::npos);
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, KernelDynamicResetInvalidatesCachedMoEDescriptorHandles)
    {
        const fs::path root = findRepoRoot();
        const fs::path stage_base_path =
            root / "src/v2/execution/compute_stages/IComputeStage.h";
        const fs::path engine_header_path =
            root / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.h";
        const fs::path engine_cpp_path =
            root / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp";
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        const fs::path moe_header_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h";
        const fs::path moe_cpp_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        ASSERT_TRUE(fs::exists(stage_base_path)) << stage_base_path;
        ASSERT_TRUE(fs::exists(engine_header_path)) << engine_header_path;
        ASSERT_TRUE(fs::exists(engine_cpp_path)) << engine_cpp_path;
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;
        ASSERT_TRUE(fs::exists(moe_header_path)) << moe_header_path;
        ASSERT_TRUE(fs::exists(moe_cpp_path)) << moe_cpp_path;

        const std::string stage_base = readFile(stage_base_path);
        const std::string engine_header = readFile(engine_header_path);
        const std::string engine_cpp = readFile(engine_cpp_path);
        const std::string orchestrator = readFile(orchestrator_path);
        const std::string moe_header = readFile(moe_header_path);
        const std::string moe_cpp = readFile(moe_cpp_path);
        ASSERT_FALSE(stage_base.empty()) << stage_base_path;
        ASSERT_FALSE(engine_header.empty()) << engine_header_path;
        ASSERT_FALSE(engine_cpp.empty()) << engine_cpp_path;
        ASSERT_FALSE(orchestrator.empty()) << orchestrator_path;
        ASSERT_FALSE(moe_header.empty()) << moe_header_path;
        ASSERT_FALSE(moe_cpp.empty()) << moe_cpp_path;

        EXPECT_NE(stage_base.find("virtual void invalidateKernelDynamicState()"), std::string::npos)
            << "Cached stages need a first-class hook for invalidating handles into "
               "backend-owned kernel-dynamic state.";
        EXPECT_NE(engine_header.find("void forEachCachedStage(const std::function<void(IComputeStage *)> &visitor) const"),
                  std::string::npos)
            << "ForwardExecutionEngine must expose all cached stages, not only a type-filtered subset.";
        EXPECT_NE(engine_cpp.find("void ForwardExecutionEngine::forEachCachedStage(\n        const std::function<void(IComputeStage *)> &visitor) const"),
                  std::string::npos);

        const size_t reset_start =
            orchestrator.find("void DeviceGraphOrchestrator::resetKernelDynamicState()");
        ASSERT_NE(reset_start, std::string::npos);
        const size_t reset_end =
            orchestrator.find("void DeviceGraphOrchestrator::recordKernelDynamicStatePreservedForCapturedReplay",
                              reset_start);
        ASSERT_NE(reset_end, std::string::npos);
        const std::string reset_body = orchestrator.substr(reset_start, reset_end - reset_start);
        const size_t factory_reset = reset_body.find("KernelFactory::resetAllDynamicState()");
        const size_t store_reset = reset_body.find("prepared_weight_store_->resetDynamicState()");
        const size_t stage_invalidation = reset_body.find("invalidateKernelDynamicState()");
        ASSERT_NE(factory_reset, std::string::npos);
        ASSERT_NE(store_reset, std::string::npos);
        ASSERT_NE(stage_invalidation, std::string::npos)
            << "Hard kernel-dynamic reset must invalidate cached stage-local descriptor handles.";
        EXPECT_LT(factory_reset, stage_invalidation);
        EXPECT_LT(store_reset, stage_invalidation);
        EXPECT_NE(reset_body.find("forward_engine_->forEachCachedStage"), std::string::npos);
        EXPECT_NE(reset_body.find("layer_graph_cache_"), std::string::npos);
        EXPECT_NE(reset_body.find("mtp_sidecar_depth0_cache_"), std::string::npos);
        EXPECT_NE(reset_body.find("device_moe_rebalance_maintenance_payload_graphs_"), std::string::npos);

        const std::regex invalidate_hook_regex("void invalidateKernelDynamicState\\(\\) override");
        const auto invalidate_begin =
            std::sregex_iterator(moe_header.begin(), moe_header.end(), invalidate_hook_regex);
        const auto invalidate_end = std::sregex_iterator();
        EXPECT_EQ(std::distance(invalidate_begin, invalidate_end), 2)
            << "Both routed and shared MoE stages cache backend descriptor-table IDs.";
        for (const std::string &required_reset : {
                 "grouped_gateup_desc_table_id_ = -1",
                 "grouped_down_desc_table_id_ = -1",
                 "runtime_grouped_decode_warmed_ = false",
                 "shared_grouped_gateup_desc_table_id_ = -1",
                 "shared_grouped_down_desc_table_id_ = -1",
                 "grouped_decode_warmed_ = false",
             })
        {
            EXPECT_NE(moe_header.find(required_reset), std::string::npos)
                << "MoE kernel-dynamic invalidation must reset " << required_reset;
        }

        const size_t fixed_prefill_start =
            moe_cpp.find("const bool use_fixed_topology_grouped_prefill");
        ASSERT_NE(fixed_prefill_start, std::string::npos);
        const size_t fixed_prefill_end =
            moe_cpp.find("// Zero the output buffer via tensor-aware kernel",
                         fixed_prefill_start);
        ASSERT_NE(fixed_prefill_end, std::string::npos);
        const std::string fixed_prefill_body =
            moe_cpp.substr(fixed_prefill_start, fixed_prefill_end - fixed_prefill_start);
        EXPECT_NE(fixed_prefill_body.find("grouped_gateup_desc_table_dirty_"), std::string::npos)
            << "Grouped prefill must rebuild dirty descriptor tables even when table IDs are nonnegative.";
        EXPECT_NE(fixed_prefill_body.find("grouped_down_desc_table_dirty_"), std::string::npos)
            << "Grouped prefill must rebuild dirty descriptor tables even when table IDs are nonnegative.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PrefixTerminalRestoreUsesStreamfulTransfers)
    {
        const fs::path root = findRepoRoot();
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;

        const std::string contents = readFile(orchestrator_path);
        ASSERT_FALSE(contents.empty()) << orchestrator_path;

        const size_t populate_start = contents.find("bool DeviceGraphOrchestrator::populatePrefix(");
        ASSERT_NE(populate_start, std::string::npos);
        const size_t populate_end = contents.find("bool DeviceGraphOrchestrator::restorePrefixTerminalState(",
                                                  populate_start);
        ASSERT_NE(populate_end, std::string::npos);
        const std::string populate_body = contents.substr(populate_start, populate_end - populate_start);

        const size_t restore_start = populate_end;
        const size_t restore_end = contents.find("bool DeviceGraphOrchestrator::harvestPrefix(",
                                                 restore_start);
        ASSERT_NE(restore_end, std::string::npos);
        const std::string restore_body = contents.substr(restore_start, restore_end - restore_start);

        EXPECT_EQ(populate_body.find("TransferEngine::instance().upload("), std::string::npos)
            << "Prefix populate restores terminal hidden that MTP consumes immediately; use uploadFull(..., stream).";
        EXPECT_EQ(restore_body.find("TransferEngine::instance().upload("), std::string::npos)
            << "Terminal logits/hidden restore must be ordered on the explicit graph stream.";
        EXPECT_NE(populate_body.find("TransferEngine::instance().uploadFull("), std::string::npos);
        EXPECT_NE(restore_body.find("TransferEngine::instance().uploadFull("), std::string::npos);
        EXPECT_NE(restore_body.find("explicitGPUStreamForOperation(\"restorePrefixTerminalState\")"),
                  std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ROCmRuntimeNativeVNNIKernelsGuardSparseExpertTables)
    {
        const fs::path root = findRepoRoot();
        const fs::path decode_path =
            root / "src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip";
        const fs::path prefill_path =
            root / "src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip";
        ASSERT_TRUE(fs::exists(decode_path)) << decode_path;
        ASSERT_TRUE(fs::exists(prefill_path)) << prefill_path;

        const std::string decode_contents = readFile(decode_path);
        const std::string prefill_contents = readFile(prefill_path);
        ASSERT_FALSE(decode_contents.empty()) << decode_path;
        ASSERT_FALSE(prefill_contents.empty()) << prefill_path;

        const std::vector<std::pair<std::string, std::string>> decode_required_tokens = {
            {"descriptor validator", "native_vnni_desc_shape_ok"},
            {"runtime expert bounds guard", "if (expert_id < 0 || expert_id >= num_experts)"},
            {"negative descriptor index guard", "if (desc_idx < 0)"},
            {"blank descriptor guard", "!native_vnni_desc_shape_ok<FMT>(desc, N, K)"},
            {"invalid k-partial zero fill", "gate_partials[partial_index] = 0.0f;"},
            {"invalid up k-partial zero fill", "up_partials[partial_index] = 0.0f;"},
        };
        for (const auto &[label, token] : decode_required_tokens)
        {
            EXPECT_NE(decode_contents.find(token), std::string::npos)
                << "ROCm grouped decode NativeVNNI kernels must retain " << label
                << " for sparse rebalance descriptor tables";
        }

        const std::vector<std::pair<std::string, std::string>> prefill_required_tokens = {
            {"descriptor validator", "prefill_native_vnni_desc_shape_ok"},
            {"runtime expert bounds guard", "if (expert_id < 0 || expert_id >= num_experts)"},
            {"blank descriptor guard", "!prefill_native_vnni_desc_shape_ok<FMT>"},
        };
        for (const auto &[label, token] : prefill_required_tokens)
        {
            EXPECT_NE(prefill_contents.find(token), std::string::npos)
                << "ROCm grouped prefill NativeVNNI kernels must retain " << label
                << " for sparse rebalance descriptor tables";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ReadyWaveApplyDoesNotLaunchSeparateStatusInitKernel)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;

        const std::string cuda_contents = readFile(cuda_path);
        const std::string rocm_contents = readFile(rocm_path);
        ASSERT_FALSE(cuda_contents.empty()) << cuda_path;
        ASSERT_FALSE(rocm_contents.empty()) << rocm_path;

        auto require_fused_ready_apply =
            [](const std::string &contents,
               const std::string &begin_token,
               const std::string &end_token,
               const char *backend)
        {
            const size_t begin = contents.find(begin_token);
            ASSERT_NE(begin, std::string::npos) << backend << " ready-wave apply wrapper missing";
            const size_t end = contents.find(end_token, begin);
            ASSERT_NE(end, std::string::npos) << backend << " ready-wave apply wrapper end token missing";
            const std::string body = contents.substr(begin, end - begin);

            EXPECT_EQ(body.find("init_rebalance_apply_status_kernel"), std::string::npos)
                << backend << " ready-wave apply runs every decode replay and must not add a separate tiny init kernel";
            EXPECT_NE(body.find("apply_rebalance_arrivals_kernel"), std::string::npos)
                << backend << " ready-wave apply wrapper must still launch the fused apply/poll kernel";
            EXPECT_NE(contents.find("if (require_ready_wave)"), std::string::npos)
                << backend << " apply kernel should retain the ready-wave status reset guard";
            EXPECT_NE(contents.find("init_rebalance_apply_status_device(status)"), std::string::npos)
                << backend << " apply kernel should reset ready-wave status inside the poll/apply kernel";
        };

        require_fused_ready_apply(cuda_contents,
                                  "bool cudaMoE_apply_ready_rebalance_wave(",
                                  "bool cudaMoE_int_to_float",
                                  "CUDA");
        require_fused_ready_apply(rocm_contents,
                                  "bool hipMoE_apply_ready_rebalance_wave(",
                                  "bool hipMoE_count_per_expert",
                                  "ROCm");
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, HotCacheDispatchBalanceUsesSingleTopKResolutionPass)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;

        const std::string cuda_contents = readFile(cuda_path);
        const std::string rocm_contents = readFile(rocm_path);
        ASSERT_FALSE(cuda_contents.empty()) << cuda_path;
        ASSERT_FALSE(rocm_contents.empty()) << rocm_path;

        EXPECT_NE(cuda_contents.find("runtime_resolve_decode_dispatch("), std::string::npos)
            << "CUDA decode routing must keep local-compute decisions and hot-cache balance counters fused.";
        EXPECT_NE(rocm_contents.find("runtime_resolve_decode_dispatch("), std::string::npos)
            << "ROCm decode routing must keep local-compute decisions and hot-cache balance counters fused.";

        auto require_fused_publication =
            [](const std::string &contents,
               const std::string &begin_token,
               const std::string &end_token,
               const char *backend)
        {
            const size_t begin = contents.find(begin_token);
            ASSERT_NE(begin, std::string::npos) << backend << " decode publish section missing";
            const size_t end = contents.find(end_token, begin);
            ASSERT_NE(end, std::string::npos) << backend << " decode publish section end token missing";
            const std::string body = contents.substr(begin, end - begin);

            EXPECT_NE(body.find("runtime_resolve_decode_dispatch("), std::string::npos)
                << backend << " decode publish should resolve local dispatch and balance counters in one pass";
            EXPECT_EQ(body.find("runtime_record_hot_cache_dispatch_balance("), std::string::npos)
                << backend << " decode publish must not run a second top-k walk only for counters";
            EXPECT_EQ(body.find("runtime_selected_slot_local_compute("), std::string::npos)
                << backend << " decode publish must not rerun the participant chooser once per top-k slot";
        };

        require_fused_publication(cuda_contents,
                                  "__global__ void softmax_topk_decode_runtime_kernel(",
                                  "__global__ void decode_route_select_runtime_kernel(",
                                  "CUDA softmax-topk");
        require_fused_publication(cuda_contents,
                                  "__global__ void decode_route_select_runtime_kernel(",
                                  "__global__ void int_to_float_kernel",
                                  "CUDA route-copy");
        require_fused_publication(rocm_contents,
                                  "void runtime_publish_decode_dispatch(",
                                  "__global__ void device_rebalance_controller_kernel(",
                                  "ROCm");
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceRebalanceTransferBackedPlansAreNotReadyBeforeCopy)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;

        const std::string cuda_contents = readFile(cuda_path);
        const std::string rocm_contents = readFile(rocm_path);
        ASSERT_FALSE(cuda_contents.empty()) << cuda_path;
        ASSERT_FALSE(rocm_contents.empty()) << rocm_path;

        auto expect_backend = [](const std::string &contents, const char *backend)
        {
            EXPECT_NE(contents.find("const bool transfer_payload_required = requested_payload_slots > 0u"),
                      std::string::npos)
                << backend << " controller must distinguish resident-only ready waves from transfer-backed waves.";
            EXPECT_NE(contents.find("wave.copied_arrivals = transfer_payload_required ? 0u : command_count"),
                      std::string::npos)
                << backend << " transfer-backed waves must not advertise copied arrivals before the payload copy completes.";
            EXPECT_NE(contents.find("wave.state = transfer_payload_required"),
                      std::string::npos)
                << backend << " transfer-backed waves must not use the same ready-state assignment as resident-only waves.";
            EXPECT_NE(contents.find("? kDeviceMoERebalanceLifecyclePlanning\n                                 : kDeviceMoERebalanceLifecycleReadyToApply"),
                      std::string::npos)
                << backend << " transfer-backed waves must stay in Planning until publishDeviceRebalanceTransferComplete fences the copy.";
            EXPECT_NE(contents.find("wave.state = kDeviceMoERebalanceLifecycleTransferInFlight"),
                      std::string::npos)
                << backend << " transfer completion publisher must still move copied payload waves through TransferInFlight.";
        };

        expect_backend(cuda_contents, "CUDA");
        expect_backend(rocm_contents, "ROCm");
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalanceApplyKernelKeepsChunkyWorkBlockParallel)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;

        const std::string cuda_contents = readFile(cuda_path);
        const std::string rocm_contents = readFile(rocm_path);
        ASSERT_FALSE(cuda_contents.empty()) << cuda_path;
        ASSERT_FALSE(rocm_contents.empty()) << rocm_path;

        EXPECT_NE(cuda_contents.find("apply_rebalance_arrivals_kernel<<<1, kThreads"),
                  std::string::npos)
            << "CUDA rebalance apply should launch a full block, not one lane";
        EXPECT_NE(rocm_contents.find("dim3(kDeviceMoERebalanceApplyThreads)"),
                  std::string::npos)
            << "ROCm rebalance apply should launch a full block, not one lane";

        auto require_block_parallel_body =
            [](const std::string &contents, const char *backend)
        {
            EXPECT_EQ(contents.find("blockIdx.x != 0 || threadIdx.x != 0 || !status"),
                      std::string::npos)
                << backend << " apply kernel must not return all nonzero lanes before copying/resetting runtime banks";
            EXPECT_NE(contents.find("__shared__ uint8_t changed_layer"),
                      std::string::npos)
                << backend << " apply kernel should share changed-layer marks across the block";
            EXPECT_NE(contents.find("for (uint32_t expert = threadIdx.x; expert < config.num_experts; expert += blockDim.x)"),
                      std::string::npos)
                << backend << " apply kernel should parallelize per-expert bank copy/reset work";
            EXPECT_NE(contents.find("runtime_resident_count(mask, config.participant_count)"),
                      std::string::npos)
                << backend << " apply kernel should parallelize multi-resident recounts";
        };
        require_block_parallel_body(cuda_contents, "CUDA");
        require_block_parallel_body(rocm_contents, "ROCm");
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalanceSourcePackingUsesEffectiveRuntimeResidency)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;

        auto require_effective_residency =
            [](const std::string &contents, const char *backend)
        {
            const size_t begin = contents.find("DeviceMoEExpertDirectoryEntryView make_rebalance_source_entry(");
            ASSERT_NE(begin, std::string::npos) << backend << " source entry helper missing";
            const size_t end = contents.find("__global__ void pack_rebalance_directory_kernel", begin);
            ASSERT_NE(end, std::string::npos) << backend << " source entry helper end marker missing";
            const std::string body = contents.substr(begin, end - begin);

            EXPECT_NE(body.find("runtime_expert_resident_mask(&runtime, bank, static_cast<int>(expert))"),
                      std::string::npos)
                << backend << " transfer source packing must include owner/local-compute bits when deciding residency";
            EXPECT_EQ(body.find("const uint32_t resident_mask = bank.resident_participant_mask[expert] & valid_mask"),
                      std::string::npos)
                << backend << " raw resident masks can omit the owner and produce zero-byte transfer payloads";
        };

        require_effective_residency(readFile(cuda_path), "CUDA");
        require_effective_residency(readFile(rocm_path), "ROCm");
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, CompactSourceDescriptorKernelsUseProjectedCommandBuffers)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;

        auto require_projected_commands =
            [](const std::string &contents, const char *backend)
        {
            const size_t begin = contents.find("__global__ void pack_rebalance_source_descriptors_kernel(");
            ASSERT_NE(begin, std::string::npos) << backend << " compact source descriptor kernel missing";
            const size_t end = contents.find("__device__ void rebalance_copy_bytes", begin);
            ASSERT_NE(end, std::string::npos) << backend << " compact source descriptor kernel end marker missing";
            const std::string body = contents.substr(begin, end - begin);

            EXPECT_NE(body.find("command_headers[static_cast<unsigned long long>(command_buffer_index)]"),
                      std::string::npos)
                << backend << " compact source descriptor packing must read projected local command headers";
            EXPECT_NE(body.find("rebalance_command_header_ok(&header, config)"),
                      std::string::npos)
                << backend << " projected local headers should be validated against the local participant ABI";
            EXPECT_NE(body.find("plan_entries[static_cast<unsigned long long>(command_buffer_index)"),
                      std::string::npos)
                << backend << " compact source descriptor packing must read projected local plan entries";
            EXPECT_EQ(body.find("gathered_plan_entries"), std::string::npos)
                << backend << " compact source descriptor packing must not use stale gathered plans";
            EXPECT_EQ(body.find("gathered_command_headers"), std::string::npos)
                << backend << " compact source descriptor packing must not use stale gathered headers";
            EXPECT_EQ(body.find("rebalance_command_header_for_participant_ok"),
                      std::string::npos)
                << backend << " projected command headers are local ABI headers, not destination-participant gathered headers";
        };

        require_projected_commands(readFile(cuda_path), "CUDA");
        require_projected_commands(readFile(rocm_path), "ROCm");
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ROCmDecodeRouterFastPathsFailHard)
    {
        const fs::path root = findRepoRoot();
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;

        const std::string contents = readFile(rocm_path);
        auto require_strict_router_body =
            [&](const std::string &begin_marker,
                const std::string &end_marker,
                const char *label)
        {
            const size_t begin = contents.find(begin_marker);
            ASSERT_NE(begin, std::string::npos) << label << " body missing";
            const size_t end = contents.find(end_marker, begin);
            ASSERT_NE(end, std::string::npos) << label << " end marker missing";
            const std::string body = contents.substr(begin, end - begin);

            EXPECT_EQ(body.find("falling back to"), std::string::npos)
                << label << " must not silently choose a different decode router after a selected fast path fails";
            EXPECT_NE(body.find("K-part router was requested but scratch allocation failed"),
                      std::string::npos)
                << label << " must fail hard when explicit K-part routing cannot allocate scratch";
            EXPECT_NE(body.find("K-part router logits kernel failed"),
                      std::string::npos)
                << label << " must fail hard when explicit K-part logits fail";
            EXPECT_NE(body.find("fused K-part router runtime kernel failed"),
                      std::string::npos)
                << label << " must fail hard when explicit K-part top-k/rebalance routing fails";
            EXPECT_NE(body.find("FP16 router was requested but gate cache is unavailable"),
                      std::string::npos)
                << label << " must fail hard when explicit FP16 routing cannot build its cache";
            EXPECT_NE(body.find("FP16 router logits kernel failed"),
                      std::string::npos)
                << label << " must fail hard when explicit FP16 logits fail";
            EXPECT_NE(body.find("wave64 decode softmax/top-k runtime kernel failed"),
                      std::string::npos)
                << label << " must fail hard when the enabled wave top-k runtime path fails";
        };

        require_strict_router_body(
            "bool ROCmMoEKernel::decodeRouteSelect(",
            "bool ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply(",
            "ROCm decodeRouteSelect");
        require_strict_router_body(
            "bool ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply(",
            "bool ROCmMoEKernel::runDeviceRebalanceController(",
            "ROCm decodeRouteSelectWithReadyRebalanceApply");
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GpuGroupedDecodeFastPathsFailHard)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;

        auto body_between =
            [](const std::string &contents,
               const std::string &begin_marker,
               const std::string &end_marker,
               const char *label) -> std::string
        {
            const size_t begin = contents.find(begin_marker);
            EXPECT_NE(begin, std::string::npos) << label << " body missing";
            if (begin == std::string::npos)
                return {};
            const size_t end = end_marker.empty() ? std::string::npos : contents.find(end_marker, begin + 1);
            EXPECT_TRUE(end_marker.empty() || end != std::string::npos)
                << label << " end marker missing";
            if (!end_marker.empty() && end == std::string::npos)
                return {};
            return contents.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        };
        auto require_no_soft_retry =
            [](const std::string &body, const char *label)
        {
            ASSERT_FALSE(body.empty()) << label;
            EXPECT_EQ(body.find("falling back"), std::string::npos)
                << label << " must fail hard instead of retrying a serial path";
            EXPECT_EQ(body.find("using fallback"), std::string::npos)
                << label << " must not advertise a hidden fallback path";
            EXPECT_EQ(body.find("fall back"), std::string::npos)
                << label << " must keep optimized-path failure explicit";
        };

        const std::string cuda = readFile(cuda_path);
        const std::string rocm = readFile(rocm_path);

        const std::vector<std::pair<std::string, std::string>> cuda_bodies = {
            {"bool CUDAMoEKernel::groupedExpertGateUpDecodeFromTable(",
             "bool CUDAMoEKernel::groupedExpertDownDecodeFromTable("},
            {"bool CUDAMoEKernel::groupedExpertDownDecodeFromTable(",
             "bool CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting("},
            {"bool CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting(",
             "bool CUDAMoEKernel::groupedExpertDownDecodeFromRouting("},
            {"bool CUDAMoEKernel::groupedExpertDownDecodeFromRouting(",
             "bool CUDAMoEKernel::groupedExpertDecodeFromRuntime("},
            {"bool CUDAMoEKernel::groupedExpertDecodeFromRuntime(",
             "bool CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime("},
            {"bool CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime(",
             "bool CUDAMoEKernel::groupedExpertDownDecodeFromRuntime("},
            {"bool CUDAMoEKernel::groupedExpertDownDecodeFromRuntime(",
             "int CUDAMoEKernel::getExpertTokenCount("},
        };
        for (const auto &[begin, end] : cuda_bodies)
            require_no_soft_retry(body_between(cuda, begin, end, begin.c_str()), begin.c_str());

        EXPECT_NE(cuda.find("mandatory K-part gate/up scratch allocation failed"),
                  std::string::npos)
            << "CUDA grouped decode must fail hard when explicit gate/up K-part scratch is unavailable";
        EXPECT_NE(cuda.find("mandatory K-part down scratch allocation failed"),
                  std::string::npos)
            << "CUDA grouped decode must fail hard when explicit down K-part scratch is unavailable";

        const std::vector<std::pair<std::string, std::string>> rocm_bodies = {
            {"bool ROCmMoEKernel::groupedExpertGateUpDecodeFromTable(",
             "bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting("},
            {"bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting(",
             "bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime("},
            {"bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime(",
             "bool ROCmMoEKernel::groupedExpertDecodeFromRuntime("},
            {"bool ROCmMoEKernel::groupedExpertDecodeFromRuntime(",
             "bool ROCmMoEKernel::groupedExpertDownDecodeFromTable("},
            {"bool ROCmMoEKernel::groupedExpertDownDecodeFromTable(",
             "bool ROCmMoEKernel::groupedExpertDownDecodeFromRouting("},
            {"bool ROCmMoEKernel::groupedExpertDownDecodeFromRouting(",
             "bool ROCmMoEKernel::groupedExpertDownDecodeFromRuntime("},
            {"bool ROCmMoEKernel::groupedExpertDownDecodeFromRuntime(",
             "bool ROCmMoEKernel::groupedExpertDownDecode("},
            {"bool ROCmMoEKernel::groupedExpertDownDecode(",
             "bool ROCmMoEKernel::groupPrefillRoutes("},
        };
        for (const auto &[begin, end] : rocm_bodies)
            require_no_soft_retry(body_between(rocm, begin, end, begin.c_str()), begin.c_str());

        EXPECT_NE(rocm.find("K-part gate/up decode was requested but scratch allocation failed"),
                  std::string::npos)
            << "ROCm grouped decode must fail hard when explicit gate/up K-part scratch is unavailable";
        EXPECT_NE(rocm.find("K-part gate/up runtime kernel failed"), std::string::npos)
            << "ROCm grouped decode must fail hard when explicit K-part gate/up launch fails";
        EXPECT_NE(rocm.find("parallel down decode was requested but codebook"),
                  std::string::npos)
            << "ROCm grouped decode must fail hard when explicit parallel down cannot support the codebook";
        EXPECT_NE(rocm.find("K-part fused gate/up SwiGLU quant kernel failed"),
                  std::string::npos)
            << "ROCm fused grouped decode must fail hard when the explicit fused K-part path fails";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PrefillLLEPTransferModeIsExplicit)
    {
        const fs::path root = findRepoRoot();
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        const fs::path stage_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;
        ASSERT_TRUE(fs::exists(stage_path)) << stage_path;

        const std::string graph = readFile(graph_path);
        const std::string stage = readFile(stage_path);
        ASSERT_FALSE(graph.empty()) << graph_path;
        ASSERT_FALSE(stage.empty()) << stage_path;

        EXPECT_NE(graph.find("LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE"),
                  std::string::npos)
            << "LLEP prefill movement must be an explicit mode, not an implicit fallback.";
        EXPECT_NE(graph.find("parsedLLEPPrefillTransferMode"),
                  std::string::npos)
            << "Qwen35 MoE graph must consume a parsed LLEP prefill movement mode.";
        EXPECT_NE(graph.find("mode < 0 || mode > 1"),
                  std::string::npos)
            << "Invalid LLEP prefill movement modes must fail during graph construction.";
        EXPECT_NE(graph.find("const bool require_full_llep_prefill_transfer"),
                  std::string::npos)
            << "The graph must make full transfer-backed prefill an explicit requirement.";
        EXPECT_NE(graph.find("require_full_llep_prefill_transfer &&"),
                  std::string::npos)
            << "Compact transfer binding must only be attached for explicit full mode.";

        const fs::path debug_env_path = root / "src/v2/utils/DebugEnv.h";
        ASSERT_TRUE(fs::exists(debug_env_path)) << debug_env_path;
        const std::string debug_env = readFile(debug_env_path);
        ASSERT_FALSE(debug_env.empty()) << debug_env_path;
        EXPECT_NE(debug_env.find("int llep_prefill_transfer_mode = 1"),
                  std::string::npos)
            << "Production LLEP prefill must default to full transfer-backed movement; "
               "resident-only is a diagnostic mode.";
        EXPECT_NE(debug_env.find("moe_rebalance.llep_prefill_transfer_mode = 1"),
                  std::string::npos)
            << "DebugEnv reload/reset must preserve the full transfer-backed LLEP default.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS"),
                  std::string::npos)
            << "Prefill LLEP must expose an explicit current-batch cost gate.";
        EXPECT_NE(debug_env.find("uint64_t llep_prefill_min_routed_rows = 8192"),
                  std::string::npos)
            << "Prefill LLEP should default to a chunky routed-row gate instead of "
               "paying transfer-backed movement on tiny batches.";
        EXPECT_NE(debug_env.find("kDefaultDeviceMinLoadSpreadImprovementDivisor"),
                  std::string::npos)
            << "Device-side LLEP/Dynamic movement must share the relative "
               "load-spread default instead of silently disabling the gate.";
        EXPECT_NE(stage.find("device_rebalance_llep_prefill_policy_skips"),
                  std::string::npos)
            << "Prefill LLEP cost-gate skips must be visible in perfstats.";
        EXPECT_NE(stage.find("reason\", \"insufficient_routed_rows"),
                  std::string::npos)
            << "Prefill LLEP cost-gate skips must report why standard AE was selected.";

        EXPECT_NE(stage.find("assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers"),
                  std::string::npos)
            << "Resident-only prefill LLEP must use the guarded no-transfer apply kernel.";
        EXPECT_NE(stage.find("assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers"),
                  std::string::npos)
            << "Full transfer-backed prefill LLEP must use the after-transfer apply kernel.";
        EXPECT_NE(stage.find("prefill_llep_require_transfer_backing"),
                  std::string::npos)
            << "Full mode must fail hard when transfer backing is unavailable.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PrefillLLEPKernelHooksFailFastWhenUnsupported)
    {
        const fs::path root = findRepoRoot();
        const fs::path header_path = root / "src/v2/kernels/IMoEKernel.h";
        const fs::path impl_path = root / "src/v2/kernels/IMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(header_path)) << header_path;
        ASSERT_TRUE(fs::exists(impl_path)) << impl_path;

        const std::string header = readFile(header_path);
        const std::string impl = readFile(impl_path);
        ASSERT_FALSE(header.empty()) << header_path;
        ASSERT_FALSE(impl.empty()) << impl_path;

        const std::vector<std::string> required_hooks = {
            "materializePrefillLeastLoadedTransferCommands",
            "planPrefillRoutesLeastLoadedCurrentBatch",
            "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers",
            "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers",
        };
        for (const auto &hook : required_hooks)
        {
            const size_t declaration_start =
                header.find("virtual bool " + hook + "(");
            ASSERT_NE(declaration_start, std::string::npos) << hook;
            const size_t declaration_end =
                header.find(";", declaration_start);
            ASSERT_NE(declaration_end, std::string::npos) << hook;
            const std::string declaration =
                header.substr(declaration_start,
                              declaration_end - declaration_start + 1);

            EXPECT_EQ(declaration.find("{"), std::string::npos)
                << hook
                << " must not use an inline default body that can become a silent fallback.";
            EXPECT_EQ(declaration.find("return false"), std::string::npos)
                << hook
                << " is a required capability once selected by the graph; unsupported "
                   "backends must fail explicitly.";
        }

        EXPECT_NE(impl.find("LLEP prefill transfer command materialization is not implemented"),
                  std::string::npos);
        EXPECT_NE(impl.find("LLEP current-batch prefill route planning is not implemented"),
                  std::string::npos);
        EXPECT_NE(impl.find("LLEP resident-only current-batch prefill assignment is not implemented"),
                  std::string::npos);
        EXPECT_NE(impl.find("LLEP transfer-backed current-batch prefill assignment is not implemented"),
                  std::string::npos);
        EXPECT_NE(impl.find("throw std::logic_error"),
                  std::string::npos)
            << "Unsupported current-batch LLEP hooks must fail fast, not return false.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PrefixRuntimeStateDoesNotSerializeMoEPlacementPointers)
    {
        const fs::path root = findRepoRoot();
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        const std::string graph = readFile(graph_path);
        ASSERT_FALSE(graph.empty()) << graph_path;

        const size_t capture_start =
            graph.find("bool Qwen35MoEGraph::capturePrefixCacheRuntimeState(");
        ASSERT_NE(capture_start, std::string::npos);
        const size_t restore_start =
            graph.find("bool Qwen35MoEGraph::restorePrefixCacheRuntimeState(",
                       capture_start);
        ASSERT_NE(restore_start, std::string::npos);
        const std::string capture_body =
            graph.substr(capture_start, restore_start - capture_start);

        EXPECT_NE(capture_body.find("state.clear()"), std::string::npos);
        EXPECT_EQ(capture_body.find("syncRuntimeStateToHost"), std::string::npos)
            << "Prefix harvest must not synchronize and serialize pointer-bearing "
               "MoE runtime tables.";
        EXPECT_EQ(capture_body.find("sizeof(DeviceMoELayerRuntime)"), std::string::npos)
            << "MoE prefix runtime state must not embed DeviceMoELayerRuntime; "
               "the struct contains process-local device descriptors.";
        EXPECT_NE(graph.find("kMoEPrefixRuntimeVersion"),
                  std::string::npos)
            << "MoE prefix runtime state should be a versioned, portable "
               "logical runtime-state blob.";
        EXPECT_NE(graph.find("capturePortableRuntimeState"),
                  std::string::npos)
            << "MoE prefix capture should export logical placement and histogram "
               "state, not runtime descriptor pointers.";
        EXPECT_NE(graph.find("restorePortableRuntimeState"),
                  std::string::npos)
            << "MoE prefix restore should replay portable logical runtime state.";
        EXPECT_NE(graph.find("localPayloadDescriptorResolverForRuntimeTable"),
                  std::string::npos)
            << "MoE prefix restore with local-compute experts must rebind payload "
               "descriptors through the graph-owned transfer-slot owner, not "
               "through stale runtime banks.";
        EXPECT_NE(graph.find("descriptorForSlot"),
                  std::string::npos)
            << "Transfer-slot directories must expose a narrow descriptor resolver "
               "for prefix restore instead of forcing pointer-bearing runtime "
               "tables into prefix payloads.";
        EXPECT_NE(graph.find("resetPrefixCacheRuntimeStateWithoutSnapshot"),
                  std::string::npos)
            << "MoE prefix restore without a runtime payload must have its own "
               "model-runtime reset boundary instead of reusing request reset.";
        const size_t prefix_reset_start =
            graph.find("void Qwen35MoEGraph::resetPrefixCacheRuntimeStateWithoutSnapshot()");
        ASSERT_NE(prefix_reset_start, std::string::npos);
        const size_t prefix_reset_end =
            graph.find("ILocalTPContext *Qwen35MoEGraph::maintenanceTPContextForDomain",
                       prefix_reset_start);
        ASSERT_NE(prefix_reset_end, std::string::npos);
        const std::string prefix_reset_body =
            graph.substr(prefix_reset_start, prefix_reset_end - prefix_reset_start);
        EXPECT_NE(prefix_reset_body.find("resetDecodeRuntimeState()"),
                  std::string::npos)
            << "No-payload prefix restore must return MoE placement tables to "
               "the empty pre-decode baseline that split prefill observes.";
        EXPECT_EQ(prefix_reset_body.find("restoreInitialRuntimeState()"),
                  std::string::npos)
            << "No-payload prefix restore must not restore the first decode bank "
               "as an initial state; split prefill has no active decode bank.";
        EXPECT_NE(graph.find("moe_graph_rebalance_bindings_.clear()"),
                  std::string::npos)
            << "The no-payload prefix boundary must drop graph-side rebalance "
               "bindings so suffix prefill cannot reuse previous-request plans.";
        EXPECT_NE(graph.find("moe_rebalance_transfer_states_.clear()"),
                  std::string::npos)
            << "The no-payload prefix boundary must drop auxiliary transfer "
               "stream state owned by dynamic/LLEP movement.";
        EXPECT_NE(graph.find("moe_transfer_slot_directories_.clear()"),
                  std::string::npos)
            << "The no-payload prefix boundary must drop transfer-slot "
               "directories that may contain previous-request expert arrivals.";
        EXPECT_NE(graph.find("portableRuntimeLayerHasPrefixRestoreState"),
                  std::string::npos)
            << "Prefix harvest must preserve epoch-zero MoE controller history "
               "when logical placement or routing histograms carry request-owned state.";
        EXPECT_NE(graph.find("owner_participant"),
                  std::string::npos)
            << "MoE prefix runtime state must keep logical expert ownership even "
               "before a non-zero placement epoch is published.";
        EXPECT_NE(graph.find("local_compute"),
                  std::string::npos)
            << "MoE prefix runtime state must keep local-compute masks for "
               "phase-split suffix prefill restore.";
        EXPECT_NE(graph.find("resident_participant_mask"),
                  std::string::npos)
            << "MoE prefix runtime state must keep resident participant masks so "
               "zero-epoch portable placement descriptors are not dropped.";
        EXPECT_NE(graph.find("selected_histogram"),
                  std::string::npos)
            << "MoE prefix runtime state must include selected-expert routing "
               "histograms so dynamic/LLEP suffix prefill restores the same "
               "controller window as split prefill.";
        EXPECT_NE(graph.find("local_histogram"),
                  std::string::npos)
            << "MoE prefix runtime state must include local-compute histograms "
               "for phase-split restore parity.";
        EXPECT_EQ(graph.find("moe_portable_runtime_state_skipped_uninitialized"),
                  std::string::npos)
            << "Epoch-zero MoE runtime snapshots are not necessarily empty: "
               "prefix restore must not drop histogram-bearing state just "
               "because no placement-bank flip has occurred yet.";
        EXPECT_NE(graph.find("expectedPrefixRuntimeParticipantCount"),
                  std::string::npos)
            << "MoE prefix capture must know the graph restore domain before "
               "serializing runtime placement.";
        EXPECT_EQ(graph.find("if (layer.active_epoch != 0u)\n                return true;"),
                  std::string::npos)
            << "Portable restore flips every table layer; epoch-only placeholder "
               "layers in sparse MTP sidecar tables must not make the whole "
               "table look request-owned.";
        EXPECT_NE(graph.find("layer_has_restore_state &&\n                    layer.participant_count"),
                  std::string::npos)
            << "Participant-count compatibility must only be checked on layers "
               "that actually carry logical placement or histogram state. "
               "Unused layers in depth-scoped sidecar tables stay at the default "
               "single-participant shape.";
        EXPECT_NE(capture_body.find("portableRuntimeStateMatchesPrefixRestoreDomain"),
                  std::string::npos)
            << "Prefix harvest must not store a one-participant grouped-prefill "
               "runtime table for a multi-participant phase-split AE restore.";
        EXPECT_NE(graph.find("Refusing obsolete prefix-cache MoE runtime state"),
                  std::string::npos)
            << "Non-empty old pointer-bearing MoE runtime blobs must fail hard.";
        EXPECT_EQ(graph.find("Prefix-cache MoE runtime state omits transient"),
                  std::string::npos)
            << "Prefix cache should not special-case transient transfer slots by "
               "silently dropping runtime placement.";
        EXPECT_EQ(graph.find("DeviceMoELayerRuntime skipped{}"),
                  std::string::npos)
            << "Do not replace transient transfer-slot runtime layers with "
               "empty placeholders during prefix-cache capture.";

        const fs::path runner_path = root / "src/v2/execution/runner/OrchestrationRunner.cpp";
        ASSERT_TRUE(fs::exists(runner_path)) << runner_path;
        const std::string runner = readFile(runner_path);
        ASSERT_FALSE(runner.empty()) << runner_path;
        EXPECT_NE(runner.find("hit.restore_model_runtime_state = matched_tokens > 0"),
                  std::string::npos)
            << "Partial prefix hits must restore portable MoE controller history "
               "from the terminal matched block.";
        EXPECT_EQ(runner.find("prefix-cache-populate-fallback"),
                  std::string::npos)
            << "Prefix populate failures must fail the request; downgrading a "
               "hit to a miss can hide lost KV/GDN/MTP/MoE ownership state.";
        EXPECT_EQ(runner.find("prefix-cache-terminal-populate-fallback"),
                  std::string::npos)
            << "Terminal restore handling must not hide a failed prefix import "
               "behind a second populate attempt.";
        EXPECT_NE(runner.find("refusing to downgrade the hit to a miss"),
                  std::string::npos)
            << "The prefix-cache path should make populate failures explicit.";
        const size_t populate_call =
            runner.find("if (!runner_->populatePrefix(common_hit))");
        ASSERT_NE(populate_call, std::string::npos);
        const size_t initial_reset =
            runner.find("prefix-cache-initial-reset");
        ASSERT_NE(initial_reset, std::string::npos);
        EXPECT_LT(populate_call, initial_reset)
            << "Prefix hits must enter populatePrefix() before any ordinary "
               "request-boundary reset so model-runtime snapshots can restore "
               "against the live MoE payload owner.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PrefixPlacementFingerprintDoesNotKeyOnRuntimeMovement)
    {
        const fs::path root = findRepoRoot();
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;

        const std::string source = readFile(orchestrator_path);
        ASSERT_FALSE(source.empty()) << orchestrator_path;

        const size_t start =
            source.find("PrefixCacheFingerprintResult DeviceGraphOrchestrator::buildCurrentPrefixFingerprint(");
        ASSERT_NE(start, std::string::npos);
        const size_t end =
            source.find("bool DeviceGraphOrchestrator::ensurePrefixCacheReady()", start);
        ASSERT_NE(end, std::string::npos);
        const std::string body = source.substr(start, end - start);

        const size_t policy =
            body.find("prefix_config.moe_policy == PrefixCacheMoEPolicy::InvalidateOnRebalance");
        ASSERT_NE(policy, std::string::npos)
            << "MoE prefix fingerprinting must separate portable placement keys "
               "from explicit invalidate-on-rebalance keys.";

        const size_t movement = body.find("\"runtime_movement_epoch\"");
        ASSERT_NE(movement, std::string::npos);
        const size_t movement_guard = body.rfind("if (invalidate_on_rebalance)", movement);
        ASSERT_NE(movement_guard, std::string::npos)
            << "Runtime movement epochs should key prefix entries only for "
               "--prefix-cache-moe-policy invalidate-on-rebalance.";
        EXPECT_GT(movement_guard, policy);

        const size_t total_rebalances = body.find("\"controller.total_rebalances\"");
        ASSERT_NE(total_rebalances, std::string::npos);
        const size_t rebalance_guard = body.rfind("if (invalidate_on_rebalance)", total_rebalances);
        ASSERT_NE(rebalance_guard, std::string::npos)
            << "Controller movement counters must not churn placement-fingerprint keys.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PhaseSplitDecodeKeepsVocabParallelEmbeddingReduction)
    {
        const fs::path root = findRepoRoot();
        const fs::path base_path = root / "src/v2/models/qwen/QwenGraphBase.cpp";
        const fs::path mtp_path = root / "src/v2/models/qwen35/Qwen35Graph.cpp";
        ASSERT_TRUE(fs::exists(base_path)) << base_path;
        ASSERT_TRUE(fs::exists(mtp_path)) << mtp_path;

        const std::string base = readFile(base_path);
        const std::string mtp = readFile(mtp_path);
        ASSERT_FALSE(base.empty()) << base_path;
        ASSERT_FALSE(mtp.empty()) << mtp_path;

        EXPECT_NE(base.find("bool QwenGraphBase::usesVocabParallelEmbeddingForCurrentGraph() const"),
                  std::string::npos);
        EXPECT_NE(base.find("int QwenGraphBase::embeddingVocabOffsetForCurrentGraph(DeviceId device) const"),
                  std::string::npos);
        EXPECT_NE(base.find("return usesVocabParallelEmbeddingForCurrentGraph()"),
                  std::string::npos)
            << "Embedding offset must be based on the active table shape, not a requested policy bit.";
        EXPECT_NE(base.find("embeddingVocabOffsetForDevice(config_, device)"),
                  std::string::npos);
        EXPECT_NE(base.find("Full-vocab decode embedding was requested but no replicated embedding binding is available"),
                  std::string::npos)
            << "Phase-split decode must fail hard instead of falling back to a sharded embedding on one participant.";
        EXPECT_NE(base.find("Replicated dense decode was requested but no replicated LM head binding is available"),
                  std::string::npos)
            << "Phase-split decode must fail hard instead of falling back to a sharded LM head.";

        const std::vector<std::pair<std::string, std::string>> sources = {
            {"QwenGraphBase.cpp", base},
            {"Qwen35Graph.cpp", mtp},
        };
        for (const auto &[name, source] : sources)
        {
            EXPECT_EQ(source.find("vocab_offset = useFullVocabEmbeddingForCurrentGraph()"),
                      std::string::npos)
                << name << " must not force offset 0 when full-vocab decode bindings are unavailable.";
            EXPECT_EQ(source.find("embedding_is_sharded && needsTPAllreduce() && denseTPAllreduceEnabledForCurrentGraph()"),
                      std::string::npos)
                << name << " must reduce vocab-parallel embedding outputs even when phase-split decode "
                           "disables dense FFN/attention allreduces.";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GraphCapturedSnapshotsPublishOnlyFromImmutableDeviceManifest)
    {
        const fs::path root = findRepoRoot();
        const fs::path executor_path =
            root / "src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp";
        ASSERT_TRUE(fs::exists(executor_path)) << executor_path;

        const std::string source = readFile(executor_path);
        ASSERT_FALSE(source.empty()) << executor_path;

        const size_t contract_start =
            source.find("GPU graph capture has a two-phase snapshot contract");
        ASSERT_NE(contract_start, std::string::npos)
            << "Graph-captured snapshots need an explicit two-phase contract.";
        const size_t profiling_end =
            source.find("if (profiling)", contract_start);
        ASSERT_NE(profiling_end, std::string::npos);
        const std::string snapshot_block =
            source.substr(contract_start, profiling_end - contract_start);

        const size_t gpu_branch =
            snapshot_block.find("if (snapshot_device.is_gpu())");
        ASSERT_NE(gpu_branch, std::string::npos)
            << "GPU snapshots must have an explicit device-owned publication branch.";
        const size_t cpu_branch =
            snapshot_block.find("StageDumpInfo snapshot_dump_info", gpu_branch);
        ASSERT_NE(cpu_branch, std::string::npos);
        const std::string gpu_snapshot_only =
            snapshot_block.substr(gpu_branch, cpu_branch - gpu_branch);

        EXPECT_NE(gpu_snapshot_only.find("captureGraphSnapshotCopies"),
                  std::string::npos)
            << "Captured graphs must still record device-to-device snapshot copy nodes.";
        EXPECT_NE(gpu_snapshot_only.find("!graph_capture_active"),
                  std::string::npos)
            << "Only eager GPU execution may publish immediately; capture/replay publication belongs after launch.";
        EXPECT_NE(gpu_snapshot_only.find("publishGraphSnapshotCopies"),
                  std::string::npos)
            << "Eager GPU snapshots must use the same immutable device manifest as graph replay.";
        EXPECT_EQ(gpu_snapshot_only.find("refreshDumpInfoSnapshot"),
                  std::string::npos)
            << "GPU snapshot publication must never re-enter mutable stage dump descriptors.";
        EXPECT_EQ(gpu_snapshot_only.find("materializeGraphSnapshotCopies"),
                  std::string::npos)
            << "The retired live-descriptor merge path must not return.";
        EXPECT_EQ(gpu_snapshot_only.find("ensureOutputsOnHost"),
                  std::string::npos)
            << "Graph capture must not enqueue D2H snapshot publication inside the captured body.";
        EXPECT_EQ(gpu_snapshot_only.find("config_.snapshot_callback"),
                  std::string::npos)
            << "Host callbacks belong after graph launch, not inside capture.";

        EXPECT_EQ(source.find("materializeGraphSnapshotCopies"),
                  std::string::npos)
            << "GPU snapshots must not merge captured storage with live post-launch stage descriptors.";

        const size_t manifest_publish_start =
            source.find("bool DeviceGraphExecutor::publishGraphSnapshotCopies(");
        ASSERT_NE(manifest_publish_start, std::string::npos);
        const size_t post_graph_start =
            source.find("bool DeviceGraphExecutor::publishSnapshotsAfterGraphExecution(",
                        manifest_publish_start);
        ASSERT_NE(post_graph_start, std::string::npos);
        const std::string manifest_publish =
            source.substr(manifest_publish_start, post_graph_start - manifest_publish_start);
        EXPECT_NE(manifest_publish.find("graph_snapshot_copies_.find(stage_name)"),
                  std::string::npos)
            << "GPU publication must resolve the immutable per-stage captured-slot manifest.";
        EXPECT_NE(manifest_publish.find("output.tensor = copy.storage.get()"),
                  std::string::npos)
            << "Published descriptors must refer directly to graph-stable snapshot storage.";
        EXPECT_NE(manifest_publish.find("ensureOutputsOnHost"),
                  std::string::npos)
            << "The manifest publisher owns the post-launch D2H transfer.";
        EXPECT_NE(manifest_publish.find("config_.snapshot_callback"),
                  std::string::npos)
            << "The manifest publisher owns the post-launch host callback.";
        EXPECT_EQ(manifest_publish.find("refreshDumpInfoSnapshot"),
                  std::string::npos)
            << "Manifest publication must not consult mutable stage state.";

        const size_t publish_start =
            post_graph_start;
        ASSERT_NE(publish_start, std::string::npos);
        const size_t terminal_publish =
            source.find("bool DeviceGraphExecutor::publishCapturedTerminalStateAfterGraphExecution(",
                        publish_start);
        ASSERT_NE(terminal_publish, std::string::npos);
        const std::string post_graph_publish =
            source.substr(publish_start, terminal_publish - publish_start);
        EXPECT_NE(post_graph_publish.find("publishGraphSnapshotCopies"),
                  std::string::npos)
            << "Post-graph GPU publication must dispatch through the immutable manifest publisher.";
        const size_t post_graph_gpu_branch =
            post_graph_publish.find("if (snapshot_device.is_gpu())");
        ASSERT_NE(post_graph_gpu_branch, std::string::npos);
        const size_t post_graph_cpu_branch =
            post_graph_publish.find("else", post_graph_gpu_branch);
        ASSERT_NE(post_graph_cpu_branch, std::string::npos);
        const std::string post_graph_gpu_only =
            post_graph_publish.substr(post_graph_gpu_branch,
                                      post_graph_cpu_branch - post_graph_gpu_branch);
        EXPECT_NE(post_graph_gpu_only.find("publishGraphSnapshotCopies"),
                  std::string::npos);
        EXPECT_EQ(post_graph_gpu_only.find("refreshDumpInfoSnapshot"),
                  std::string::npos)
            << "Captured GPU replay must not rebuild snapshots from live stage state.";
        EXPECT_EQ(post_graph_gpu_only.find("ensureOutputsOnHost"),
                  std::string::npos)
            << "Only the manifest publisher may perform GPU D2H publication.";

        const size_t copy_start =
            source.find("bool DeviceGraphExecutor::prepareOrRecordGraphSnapshotCopies(");
        ASSERT_NE(copy_start, std::string::npos);
        const size_t materialize_start = manifest_publish_start;
        const std::string copy_body =
            source.substr(copy_start, materialize_start - copy_start);
        EXPECT_NE(copy_body.find("if (!record_device_copy && !capture_active)"),
                  std::string::npos)
            << "Allocation-only pre-capture preparation must validate the descriptor finalized by warmup execution.";
        EXPECT_NE(copy_body.find("reached capture preparation before warmup finalized its device manifest"),
                  std::string::npos)
            << "Missing warmup snapshot manifests must hard-fail instead of substituting pre-execution stage tensors.";
        EXPECT_NE(copy_body.find("graph_snapshot_outputless_stages_.contains(node.name)"),
                  std::string::npos)
            << "Warmup-observed outputless stages need an explicit finalized manifest state.";
        EXPECT_NE(copy_body.find("lost all tensor-backed outputs during graph capture"),
                  std::string::npos)
            << "A producer that loses warmed outputs during capture must hard-fail.";
        EXPECT_NE(copy_body.find("copy.descriptor_finalized = true"),
                  std::string::npos)
            << "A successful point-in-time D2D copy must finalize its production descriptor.";
        const size_t event_contract =
            copy_body.find("The D2D copy has been recorded as a graph node");
        ASSERT_NE(event_contract, std::string::npos)
            << "Graph snapshot copy coherence needs an explicit capture-time "
               "event contract.";
        const size_t capture_event_branch =
            copy_body.rfind("if (isGraphCaptureActive())", event_contract);
        ASSERT_NE(capture_event_branch, std::string::npos)
            << "Graph-captured snapshot copies need a capture-specific "
               "coherence branch.";
        const size_t eager_event_branch =
            copy_body.find("else", capture_event_branch);
        ASSERT_NE(eager_event_branch, std::string::npos);
        const std::string capture_event_only =
            copy_body.substr(capture_event_branch,
                             eager_event_branch - capture_event_branch);
        EXPECT_NE(capture_event_only.find("transitionTo("),
                  std::string::npos)
            << "Capture-time snapshot copies should update coherence flags.";
        EXPECT_EQ(capture_event_only.find("transitionToWithEvent"),
                  std::string::npos)
            << "Do not record host-waitable tensor events while a stream is "
               "being captured; the post-graph publisher records the real "
               "completion event after launch.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, FastCollectivePathRecordsGraphSnapshotCopiesBeforeReturn)
    {
        const fs::path root = findRepoRoot();
        const fs::path executor_path =
            root / "src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp";
        ASSERT_TRUE(fs::exists(executor_path)) << executor_path;

        const std::string source = readFile(executor_path);
        ASSERT_FALSE(source.empty()) << executor_path;

        const size_t collective_intercept =
            source.find("Collective Stage Intercept");
        ASSERT_NE(collective_intercept, std::string::npos);
        const size_t fast_branch =
            source.find("if (!policy.coherence && !force_contract_coherence)", collective_intercept);
        ASSERT_NE(fast_branch, std::string::npos)
            << "Fast collective execution must remain explicit so graph snapshot ownership is auditable.";
        const size_t fast_return = source.find("return ok;", fast_branch);
        ASSERT_NE(fast_return, std::string::npos);

        const std::string fast_collective_body =
            source.substr(fast_branch, fast_return - fast_branch);
        EXPECT_NE(fast_collective_body.find("config_.snapshot_callback && !policy.snapshot_callback"),
                  std::string::npos)
            << "Fast collectives must honor the graph snapshot copy contract while host callbacks are deferred.";
        EXPECT_NE(fast_collective_body.find("captureGraphSnapshotCopies"),
                  std::string::npos)
            << "Post-collective graph snapshots such as *_ALLREDUCED must record a D2D copy before the fast path returns.";
        EXPECT_EQ(fast_collective_body.find("ensureOutputsOnHost"),
                  std::string::npos)
            << "Fast collective graph snapshot capture must not materialize host data inside GPU capture.";
        EXPECT_EQ(fast_collective_body.find("config_.snapshot_callback(node.name"),
                  std::string::npos)
            << "Fast collective capture must defer host callbacks until post-graph publication.";
    }

    /**
     * @brief Prevent the retired umbrella terminology from returning.
     *
     * Routed-expert compute distribution, row assignment, placement topology,
     * and weight slicing are independent contracts. Reintroducing one generic
     * name for those contracts makes configuration and collective behavior
     * caller-dependent again, so this scan covers production sources, active
     * contributor guidance, and executable benchmark/E2E scripts. Parser tests
     * intentionally contain obsolete spellings as negative inputs and are
     * therefore outside this scan.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RoutedExpertPoliciesRemainExplicitlyNamed)
    {
        const fs::path root = findRepoRoot();
        std::vector<fs::path> files;

        const fs::path source_root = root / "src/v2";
        ASSERT_TRUE(fs::exists(source_root)) << source_root;
        for (const auto &entry : fs::recursive_directory_iterator(source_root))
        {
            if (!entry.is_regular_file())
                continue;

            const std::string extension = entry.path().extension().string();
            if (extension == ".h" || extension == ".hpp" ||
                extension == ".cpp" || extension == ".cu" ||
                extension == ".hip" ||
                entry.path().filename() == "CMakeLists.txt")
            {
                files.push_back(entry.path());
            }
        }

        files.push_back(
            root / ".github/instructions/llaminar-v2-architecture.instructions.md");
        files.push_back(root / "AGENTS.md");
        files.push_back(root / "README.md");
        files.push_back(root / ".agents/mtp-tuning/SKILL.md");
        files.push_back(root / "scripts/run_mtp_iteration_benchmark_matrix.sh");
        files.push_back(root / "scripts/run_qwen36_moe_gpu_rebalance_sprint.sh");
        files.push_back(root / "tests/v2/e2e/server/test_server_e2e.sh");

        const std::vector<std::string> retired_literals = {
            "ExpertParallel",
            "expert_parallel",
            "EXPERT_PARALLEL",
            "expert-parallel",
            "PhaseSplitHybridTP_AE",
            "phase_split_hybrid_tp_ae",
            "phase-split-hybrid-tp-ae",
            "--moe-expert-mode",
            "--moe-expert-overlay",
            "moe_expert_parallel",
            "apportioned-experts",
            "replicated-experts",
            "sharded-experts",
        };

        for (const auto &path : files)
        {
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string source = readFile(path);
            const std::string relative_path =
                fs::relative(path, root).generic_string();
            for (const auto &retired : retired_literals)
            {
                // The parser names this one obsolete YAML root solely to emit
                // an actionable migration error. It remains forbidden in every
                // other production file, including configuration consumers.
                if ((retired == "moe_expert_parallel" ||
                     retired == "expert_parallel") &&
                    relative_path == "src/v2/config/OrchestrationConfigParser.cpp")
                {
                    continue;
                }
                EXPECT_EQ(source.find(retired), std::string::npos)
                    << relative_path << " contains retired routed-expert term '"
                    << retired << "'";
            }
        }
    }

} // namespace llaminar2::test
