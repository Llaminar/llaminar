/**
 * @file Test__MoEGraphNative_ForbiddenDependencyScan.cpp
 * @brief Source hygiene tests for graph-native MoE and backend-neutral MoE stages.
 *
 * These tests scan source files that are supposed to remain orchestration glue.
 * They catch accidental dependencies on legacy overlay runtime code and direct
 * CUDA/HIP runtime APIs before those dependencies can leak into compute stages.
 */

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
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

        size_t countOccurrences(
            const std::string &contents,
            const std::string &needle)
        {
            if (needle.empty())
                return 0u;

            size_t count = 0u;
            size_t cursor = 0u;
            while ((cursor = contents.find(needle, cursor)) != std::string::npos)
            {
                ++count;
                cursor += needle.size();
            }
            return count;
        }

        /**
         * @brief Extract one out-of-line template method for source-contract checks.
         *
         * CUDA and ROCm KV-cache implementations place a `template` declaration
         * before every method. Using the next declaration as the boundary keeps
         * these tests independent of line numbers while avoiding false matches
         * elsewhere in the large backend files.
         */
        std::string templateMethodRegion(
            const std::string &contents,
            const std::string &signature)
        {
            const size_t start = contents.find(signature);
            if (start == std::string::npos)
                return {};
            const size_t end = contents.find("\n    template <", start + signature.size());
            return contents.substr(
                start,
                end == std::string::npos ? std::string::npos : end - start);
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

    /**
     * @brief Keep the transfer lifecycle gate symmetric across GPU backends.
     *
     * The production defect behind this suite survived because backend-local
     * one-wave tests covered same-layer slot reuse but not model-wide reuse.
     * This source contract prevents future test registration edits from
     * silently dropping either backend, the shared production-shaped model, or
     * the complete `DeviceRebalance*` lifecycle inventory.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         GPUTransferStateMachineSuitesRemainSymmetricAndComprehensive)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_test =
            root /
            "tests/v2/integration/kernels/cuda/Test__CUDAMoEKernel.cpp";
        const fs::path rocm_test =
            root /
            "tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp";
        const fs::path shared_model =
            root /
            "tests/v2/integration/kernels/moe/MoETransferStateMachineTestModel.h";
        const fs::path cmake_file =
            root / "tests/v2/CMakeLists.txt";

        for (const auto &path :
             {cuda_test, rocm_test, shared_model, cmake_file})
        {
            ASSERT_TRUE(fs::exists(path)) << path;
        }

        const std::string cuda = readFile(cuda_test);
        const std::string rocm = readFile(rocm_test);
        const std::string model = readFile(shared_model);
        const std::string cmake = readFile(cmake_file);
        for (const auto *contents :
             {&cuda, &rocm, &model, &cmake})
        {
            ASSERT_FALSE(contents->empty());
        }

        constexpr const char *kSharedInclude =
            "#include \"../moe/MoETransferStateMachineTestModel.h\"";
        constexpr const char *kStressCase =
            "DeviceRebalanceTransferStateMachineFullCapacityStress";
        EXPECT_NE(cuda.find(kSharedInclude), std::string::npos);
        EXPECT_NE(rocm.find(kSharedInclude), std::string::npos);
        EXPECT_EQ(countOccurrences(cuda, kStressCase), 1u);
        EXPECT_EQ(countOccurrences(rocm, kStressCase), 1u);

        EXPECT_NE(
            model.find("kLayerCount = 20"),
            std::string::npos);
        EXPECT_NE(
            model.find("kExpertCount = 128"),
            std::string::npos);
        EXPECT_NE(
            model.find("kTransferSlotCount = 42"),
            std::string::npos);
        EXPECT_NE(
            model.find("kStressRotationCount = 8"),
            std::string::npos);
        EXPECT_NE(
            model.find("kMaxCommandsPerWave = 4"),
            std::string::npos);
        EXPECT_NE(
            model.find("AdversarialWave makeAdversarialWave"),
            std::string::npos);
        EXPECT_NE(
            cuda.find("delay_last_publication"),
            std::string::npos);
        EXPECT_NE(
            rocm.find("delay_last_publication"),
            std::string::npos);
        EXPECT_NE(
            cuda.find("idempotent_replays"),
            std::string::npos);
        EXPECT_NE(
            rocm.find("idempotent_replays"),
            std::string::npos);
        EXPECT_NE(
            model.find("validateSnapshot"),
            std::string::npos);

        EXPECT_NE(
            cmake.find(
                "V2_Integration_CUDA_MoETransferStateMachine"),
            std::string::npos);
        EXPECT_NE(
            cmake.find(
                "V2_Integration_ROCm_MoETransferStateMachine"),
            std::string::npos);
        EXPECT_NE(
            cmake.find(
                "Test__CUDAMoEKernel.DeviceRebalance*"),
            std::string::npos);
        EXPECT_NE(
            cmake.find(
                "Test__ROCmMoEKernel.DeviceRebalance*"),
            std::string::npos);
        EXPECT_NE(
            cmake.find(
                "RepeatedRequestResetRestoresImmutableTransferDirectoryBaseline"),
            std::string::npos);
    }

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

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         GPUKVCacheWorkspaceBindingCannotPublishImmutableTopology)
    {
        const fs::path root = findRepoRoot();
        struct BackendSource
        {
            fs::path path;
            std::string class_name;
            std::array<std::string, 4> forbidden_bind_tokens;
        };
        const std::array<BackendSource, 2> backends = {{
            {
                "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu",
                "CUDARingKVCache",
                {"cudaMalloc(", "cudaMemcpy(", "cudaMemcpyAsync(", "batched_pointer_tables_ready_ = false"},
            },
            {
                "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp",
                "ROCmRingKVCache",
                {"hipMalloc(", "hipMemcpy(", "hipMemcpyAsync(", "batched_pointer_tables_ready_ = false"},
            },
        }};

        for (const auto &backend : backends)
        {
            const fs::path path = root / backend.path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            const std::string bind_region = templateMethodRegion(
                contents,
                "void " + backend.class_name + "<Precision>::bindWorkspace");
            ASSERT_FALSE(bind_region.empty())
                << backend.path << " is missing its workspace binding implementation";
            EXPECT_NE(bind_region.find("if (workspace_ == workspace)"), std::string::npos)
                << backend.path
                << " must make repeated hot-path workspace binding an identity no-op";
            EXPECT_NE(bind_region.find("isGraphCaptureActive()"), std::string::npos)
                << backend.path
                << " must reject ownership changes while an outer graph is recording";
            for (const auto &token : backend.forbidden_bind_tokens)
            {
                EXPECT_EQ(bind_region.find(token), std::string::npos)
                    << backend.path
                    << " must not perform immutable topology publication from bindWorkspace(): "
                    << token;
            }

            const std::string allocation_region = templateMethodRegion(
                contents,
                "void " + backend.class_name + "<Precision>::allocate_all_entries");
            ASSERT_FALSE(allocation_region.empty())
                << backend.path << " is missing transactional cache construction";
            EXPECT_NE(
                allocation_region.find("initializeBatchedEntryPointerTables()"),
                std::string::npos)
                << backend.path
                << " must publish grouped-gather topology during construction";
        }
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
            "src/v2/execution/compute_stages/stages/MoERoutedExpertPartialReduceStage.h",
            "src/v2/execution/compute_stages/stages/MoERoutedExpertPartialReduceStage.cpp",
            "src/v2/execution/moe/MoEExpertOverlayDenseReduce.h",
            "src/v2/execution/moe/MoEExpertOverlayDenseReduce.cpp",
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

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GPUMoETensorWritesRequireEventPublication)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> paths = {
            "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp",
            "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp",
        };

        for (const auto &relative_path : paths)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            EXPECT_NE(contents.find("void markDeviceWritten("), std::string::npos)
                << relative_path;
            EXPECT_NE(
                contents.find("TransferEngine::publishDeviceWrite("),
                std::string::npos)
                << relative_path;
            EXPECT_EQ(contents.find("dynamic_cast<"), std::string::npos)
                << relative_path
                << " must leave tensor placement and concrete coherence "
                   "validation inside TransferEngine";
            EXPECT_NE(
                contents.find("TransferEngine::prepareDeviceInput("),
                std::string::npos)
                << relative_path;
            EXPECT_NE(
                contents.find("TransferEngine::prepareDeviceOutput("),
                std::string::npos)
                << relative_path;
            EXPECT_NE(
                contents.find("transfer service owns concrete tensor validation"),
                std::string::npos)
                << relative_path;
            EXPECT_EQ(
                contents.find("->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE"),
                std::string::npos)
                << relative_path
                << " must publish every asynchronous MoE tensor write through "
                   "markDeviceWritten so consumers receive an ordering event";
            EXPECT_EQ(
                contents.find("->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE"),
                std::string::npos)
                << relative_path
                << " must not retain a namespace-qualified eventless publication";
            EXPECT_EQ(contents.find("::route(const float *"), std::string::npos)
                << relative_path
                << " must not restore the obsolete GPU host-returning routing API; "
                   "production routing is device-owned through routeWithTensors";
        }

        for (const auto &relative_path : std::vector<fs::path>{
                 "src/v2/kernels/cuda/moe/CUDAMoEKernel.h",
                 "src/v2/kernels/rocm/moe/ROCmMoEKernel.h"})
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;
            EXPECT_EQ(contents.find("bool route("), std::string::npos)
                << relative_path
                << " must expose only device-resident GPU routing contracts";
        }

        /*
         * The ROCm integration harness intentionally exercises asynchronous
         * tensor producers.  A plain DEVICE_AUTHORITATIVE transition in that
         * harness clears the producer event and converts a real ordering
         * contract into stale host metadata, so keep the regression fixtures
         * subject to the same publication rule as production code.
         */
        const fs::path rocm_test_path =
            root / "tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(rocm_test_path)) << rocm_test_path;
        const std::string rocm_test = readFile(rocm_test_path);
        ASSERT_FALSE(rocm_test.empty()) << rocm_test_path;
        EXPECT_EQ(
            rocm_test.find("->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE"),
            std::string::npos)
            << rocm_test_path
            << " must observe wrapper outputs through ensureOnHost(producer_stream), "
               "or publish raw launches through TransferEngine::publishDeviceWrite";
        EXPECT_NE(
            rocm_test.find("if (!kernel.hasExplicitGPUStream())"),
            std::string::npos)
            << rocm_test_path
            << " workspace binding must preserve a fixture's explicit producer stream";
        EXPECT_EQ(
            rocm_test.find(
                "EXPECT_TRUE(workspace->allocate(reqs));\n"
                "        static_cast<ITensorKernel &>(kernel).setGPUStream"),
            std::string::npos)
            << rocm_test_path
            << " must not silently replace stream ownership while binding a workspace";

        const fs::path cuda_test_path =
            root / "tests/v2/integration/kernels/cuda/Test__CUDAMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(cuda_test_path)) << cuda_test_path;
        const std::string cuda_test = readFile(cuda_test_path);
        ASSERT_FALSE(cuda_test.empty()) << cuda_test_path;
        EXPECT_EQ(
            cuda_test.find("transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE"),
            std::string::npos)
            << cuda_test_path
            << " must consume production completion events through ensureOnHost; "
               "an eventless rewrite makes stale host reads appear recoverable";

        const fs::path rocm_kernel_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(rocm_kernel_path)) << rocm_kernel_path;
        const std::string rocm_kernel = readFile(rocm_kernel_path);
        ASSERT_FALSE(rocm_kernel.empty()) << rocm_kernel_path;
        EXPECT_EQ(
            rocm_kernel.find("setGPUStream(ctx.defaultStream())"),
            std::string::npos)
            << rocm_kernel_path
            << " context fallback availability must not masquerade as explicit stream ownership";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GPUQuantisedTensorGEMMWritesRequireEventPublication)
    {
        const fs::path root = findRepoRoot();
        struct BackendContract
        {
            fs::path relative_path;
            std::string publication_helper;
            std::string stream_token;
        };
        const std::vector<BackendContract> contracts = {
            {
                "src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp",
                "publishCUDATensorWrite",
                "execution_stream",
            },
            {
                "src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp",
                "publishROCmTensorWrite",
                "gpu_stream_",
            },
        };

        /*
         * Quantised tensor GEMM is invoked directly by grouped verifier and
         * shared-expert composition.  It must therefore own both halves of the
         * asynchronous contract: join every tensor producer on the selected
         * stream before dereferencing its device pointer, then publish every
         * successful output write with an event on that same stream.
         */
        for (const auto &contract : contracts)
        {
            const fs::path path = root / contract.relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;

            EXPECT_NE(
                contents.find("void " + contract.publication_helper + "("),
                std::string::npos)
                << contract.relative_path
                << " must define one authoritative tensor-write publication helper";
            EXPECT_NE(
                contents.find("TransferEngine::publishDeviceWrite("),
                std::string::npos)
                << contract.relative_path;
            EXPECT_NE(
                contents.find("TransferEngine::prepareDeviceInput("),
                std::string::npos)
                << contract.relative_path
                << " must join input readiness through the transfer service";
            EXPECT_NE(contents.find(contract.stream_token), std::string::npos)
                << contract.relative_path;
            EXPECT_EQ(
                contents.find("->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE"),
                std::string::npos)
                << contract.relative_path
                << " must not clear a producer event after asynchronous GEMM";
            EXPECT_EQ(
                contents.find("ensureOnDevice(target_device)"),
                std::string::npos)
                << contract.relative_path
                << " must join tensor readiness on the actual GEMM consumer stream";
        }
    }

    /**
     * @brief Require compute stages to prepare and publish through their GPU token.
     *
     * TransferEngine's raw publication primitive validates that a stream is
     * non-null, but a free `(device, stream)` argument pair cannot prove that
     * the stream is the one which launched the producer kernel. The stage API
     * closes that gap by constructing StageGPUExecution exclusively from the
     * executor-bound stage state. Every compute stage must prepare inputs,
     * prepare outputs, and publish writes through that token; only
     * ComputeStageBase may translate it into raw TransferEngine primitives.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ComputeStagesCannotChoosePublicationStream)
    {
        const fs::path root = findRepoRoot();
        const fs::path stages_root =
            root / "src/v2/execution/compute_stages/stages";
        ASSERT_TRUE(fs::exists(stages_root)) << stages_root;

        for (const auto &entry : fs::recursive_directory_iterator(stages_root))
        {
            if (!entry.is_regular_file())
                continue;
            const std::string extension = entry.path().extension().string();
            if (extension != ".h" && extension != ".hpp" &&
                extension != ".cpp")
            {
                continue;
            }

            const std::string source = readFile(entry.path());
            ASSERT_FALSE(source.empty()) << entry.path();
            EXPECT_EQ(
                source.find("TransferEngine::publishDeviceWrite("),
                std::string::npos)
                << fs::relative(entry.path(), root)
                << " must publish through IComputeStage::gpuExecution(); "
                   "a raw stream argument can diverge from the producer stream";
            EXPECT_EQ(
                source.find("TransferEngine::prepareDeviceInput("),
                std::string::npos)
                << fs::relative(entry.path(), root)
                << " must prepare inputs through IComputeStage::gpuExecution(); "
                   "a raw stream argument can diverge from the consumer stream";
            EXPECT_EQ(
                source.find("TransferEngine::prepareDeviceOutput("),
                std::string::npos)
                << fs::relative(entry.path(), root)
                << " must prepare outputs through IComputeStage::gpuExecution(); "
                   "a raw stream argument can diverge from the producer stream";
        }

        const fs::path interface_path =
            root / "src/v2/execution/compute_stages/IComputeStage.h";
        const fs::path implementation_path =
            root / "src/v2/execution/compute_stages/ComputeStageBase.cpp";
        const std::string interface_source = readFile(interface_path);
        const std::string implementation_source = readFile(implementation_path);
        ASSERT_FALSE(interface_source.empty()) << interface_path;
        ASSERT_FALSE(implementation_source.empty()) << implementation_path;

        EXPECT_NE(interface_source.find("class StageGPUExecution final"),
                  std::string::npos);
        EXPECT_NE(interface_source.find(
                      "StageGPUExecution(DeviceId device, void *stream);"),
                  std::string::npos)
            << "Only IComputeStage may construct the device/stream authority";
        EXPECT_NE(interface_source.find(
                      "friend class IComputeStage;"),
                  std::string::npos);
        EXPECT_NE(interface_source.find(
                      "[[nodiscard]] StageGPUExecution gpuExecution() const"),
                  std::string::npos);
        EXPECT_EQ(
            countOccurrences(
                implementation_source,
                "TransferEngine::publishDeviceWrite("),
            1u)
            << "ComputeStageBase must contain the sole raw stage-publication boundary";
        EXPECT_EQ(
            countOccurrences(
                implementation_source,
                "TransferEngine::prepareDeviceInput("),
            1u)
            << "ComputeStageBase must contain the sole raw stage-input boundary";
        EXPECT_EQ(
            countOccurrences(
                implementation_source,
                "TransferEngine::prepareDeviceOutput("),
            1u)
            << "ComputeStageBase must contain the sole raw stage-output boundary";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RoutineCoherenceDiagnosticsRemainTraceOnly)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> paths = {
            "src/v2/tensors/TensorBase.cpp",
            "src/v2/transfer/TransferEngine.cpp",
            "src/v2/kernels/cuda/CUDAKernelBase.h",
            "src/v2/kernels/rocm/ROCmKernelBase.h",
        };

        /*
         * These ownership layers run once per tensor or stage edge. Successful
         * allocations, transfers, and state transitions are useful when
         * reconstructing a coherence timeline, but DEBUG makes an ordinary
         * inference log scale with graph edges and hides the first actionable
         * failure. WARN and ERROR remain available for anomalous and invalid
         * states; routine lifecycle narration belongs exclusively at TRACE.
         */
        for (const auto &relative_path : paths)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string contents = readFile(path);
            ASSERT_FALSE(contents.empty()) << path;
            EXPECT_EQ(contents.find("LOG_DEBUG("), std::string::npos)
                << relative_path
                << " must not emit per-buffer coherence narration at DEBUG";
        }
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

        EXPECT_NE(header.find("uint32_t multi_resident_expert_count = 0"),
                  std::string::npos)
            << "The runtime ABI must expose a typed cheap hot-cache stats gate.";
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

    /**
     * @brief Grouped GPU MoE publication must never sample its output on host.
     *
     * Production grouped prefill is shared by verifier and ordinary prefill
     * graph capture. A diagnostic D2H copy or stream synchronization in this
     * method changes the execution architecture when tracing is enabled and
     * can make stale host observations look authoritative. Device assertions
     * and graph-safe PerfStats counters are the supported diagnostics.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         GPUGroupedMoEPrefillHasNoHostSampling)
    {
        const fs::path root = findRepoRoot();
        const std::array<std::tuple<const char *, const char *, const char *>, 2>
            methods = {{
                {"src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp",
                 "bool CUDAMoEKernel::executeGroupedPrefillPipeline(",
                 "bool CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime("},
                {"src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp",
                 "bool ROCmMoEKernel::executeGroupedPrefillPipeline(",
                 "bool ROCmMoEKernel::executeGroupedPrefillPipelineFromRuntime("},
            }};

        for (const auto &[relative_path, method, next_method] : methods)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string source = readFile(path);
            const size_t begin = source.find(method);
            ASSERT_NE(begin, std::string::npos) << method;
            const size_t end = source.find(next_method, begin);
            ASSERT_NE(end, std::string::npos) << next_method;
            const std::string body = source.substr(begin, end - begin);

            EXPECT_EQ(body.find("DeviceToHost"), std::string::npos)
                << relative_path << " must not download grouped output";
            EXPECT_EQ(body.find("StreamSynchronize"), std::string::npos)
                << relative_path << " must not synchronize for host diagnostics";
            EXPECT_EQ(body.find("host_output"), std::string::npos)
                << relative_path << " must keep grouped output device-owned";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         ROCmGroupedFP32RouterUsesSerialDecodeGeometry)
    {
        const fs::path root = findRepoRoot();
        const fs::path bridge_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        const fs::path kernel_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(bridge_path)) << bridge_path;
        ASSERT_TRUE(fs::exists(kernel_path)) << kernel_path;

        const std::string bridge = readFile(bridge_path);
        const std::string kernel = readFile(kernel_path);
        ASSERT_FALSE(bridge.empty());
        ASSERT_FALSE(kernel.empty());

        EXPECT_NE(
            bridge.find(
                "rocm_moe_gate_logits_grouped_decode_router_rows_kernel"),
            std::string::npos)
            << "ROCm must expose one row-grid form of the selected M=1 decode router";
        EXPECT_NE(
            bridge.find("dim3(expert_blocks, seq_len)"),
            std::string::npos)
            << "Grouped rows must extend the grid without changing per-row block geometry";
        EXPECT_NE(
            kernel.find("bool launchDecodeEquivalentFP32Rows("),
            std::string::npos)
            << "FP32 router geometry policy must have one visible owner";
        EXPECT_GE(
            countOccurrences(kernel, "launchDecodeEquivalentFP32Rows("),
            3u)
            << "The policy owner, generic grouped route, and verifier route must share one dispatch";
        EXPECT_NE(
            kernel.find("A selected launch failure is returned directly"),
            std::string::npos)
            << "A selected arithmetic path must fail hard instead of retrying a different reduction";
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
            {"void ROCmMoEKernel::weightedAdd", "bool ROCmMoEKernel::groupTokensByExpertDevice"},
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
             "bool ROCmMoEKernel::prepareExpertGroupsAsync"},
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
        const size_t dgo_prefill_advise = dgo_contents.find("weight_manager_->adviseMmapDontneed()", dgo_prefill_start);
        ASSERT_NE(dgo_prefill_advise, std::string::npos);
        const std::string dgo_prefill_body =
            dgo_contents.substr(dgo_prefill_start, dgo_prefill_advise - dgo_prefill_start);
        EXPECT_EQ(dgo_prefill_body.find("synchronizeStream("), std::string::npos);
        EXPECT_EQ(dgo_prefill_body.find("synchronizeDevice("), std::string::npos)
            << "DeviceLoadPipeline completion owns the mmap-source lifetime edge; first prefill must not drain the GPU.";

        const size_t rank_release = rank_contents.find("releaseHostResidentWeightData();");
        const size_t rank_advise = rank_contents.find("wm->adviseMmapDontneed()", rank_release);
        ASSERT_NE(rank_release, std::string::npos);
        ASSERT_NE(rank_advise, std::string::npos);
        EXPECT_LT(rank_release, rank_advise);
        const std::string rank_release_body =
            rank_contents.substr(rank_release, rank_advise - rank_release);
        EXPECT_EQ(rank_release_body.find("synchronizeStream("), std::string::npos);
        EXPECT_EQ(rank_release_body.find("synchronizeDevice("), std::string::npos)
            << "Rank mmap reclamation must rely on completed load pipelines, not a rank-wide GPU drain.";
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
        const size_t chunk_terminal =
            chunk_body.find("noteMainForwardHiddenProducedForMTP(");
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

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceRebalanceMaintenanceIsOneAtomicGraph)
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

        EXPECT_NE(ffn_body.find("first_device_rebalance_decode_layer"), std::string::npos);
        EXPECT_NE(ffn_body.find("last_device_rebalance_decode_layer"), std::string::npos);
        EXPECT_NE(ffn_body.find("local_decode_layer"), std::string::npos);
        EXPECT_NE(ffn_body.find("grouped_main_verifier_layer"), std::string::npos);
        EXPECT_NE(ffn_body.find("device_rebalance_decode_layer"), std::string::npos);
        EXPECT_NE(ffn_body.find("total_tokens == 1"), std::string::npos);
        EXPECT_NE(ffn_body.find("layer_idx == config_.pp_layer_offset"), std::string::npos);
        EXPECT_NE(ffn_body.find("config_.compute_all_position_logits"), std::string::npos)
            << "Grouped all-position MTP rows must create the production decode-maintenance binding.";
        EXPECT_NE(ffn_body.find("device_side_graph_rebalance_candidate ="), std::string::npos);
        EXPECT_NE(ffn_body.find("device_rebalance_decode_layer &&"), std::string::npos)
            << "Device-side graph rebalance must own both serial and grouped decode layers so MTP does not depend on constructing an M=1 graph.";
        EXPECT_EQ(ffn_body.find("device_rebalance_graph_controller"),
                  std::string::npos)
            << "Homogeneous graph-stable GPU rebalance must not expose a host-controller selection toggle.";
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
        const size_t apportioned_table_start =
            ffn_body.find("const bool masked_local_tp_apportioned_decode_runtime_table");
        ASSERT_NE(apportioned_table_start, std::string::npos);
        const size_t apportioned_table_end =
            ffn_body.find("const bool decode_runtime_table_eligible", apportioned_table_start);
        ASSERT_NE(apportioned_table_end, std::string::npos);
        const std::string apportioned_table_policy =
            ffn_body.substr(
                apportioned_table_start,
                apportioned_table_end - apportioned_table_start);
        EXPECT_NE(
            apportioned_table_policy.find("local_decode_layer"),
            std::string::npos)
            << "Masked apportioned LocalTP decode is a topology contract, not a Dynamic-rebalance capability. "
               "Static, Dynamic, and LLEP decode must all enter the device-resident runtime table.";
        EXPECT_NE(
            apportioned_table_policy.find(
                "mtp_sidecar_context && total_tokens == 1"),
            std::string::npos)
            << "Single-row MTP sidecars share the same masked, device-resident apportioned topology.";
        EXPECT_NE(
            apportioned_table_policy.find("device.is_gpu()"),
            std::string::npos)
            << "The masked runtime table is the GPU graph-owned route contract.";
        EXPECT_EQ(
            apportioned_table_policy.find("device_side_graph_rebalance_candidate"),
            std::string::npos)
            << "Static LocalTP must not lose its masked runtime table merely because no rebalance wave is active.";
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
        EXPECT_NE(contents.find("params.stage_name = \"moe_device_rebalance_maintenance\""),
                  std::string::npos)
            << "Maintenance must have one stable graph-owned transaction name.";
        EXPECT_NE(contents.find("params.phase = DeviceMoERebalanceStagePhase::PlanCopyApply"),
                  std::string::npos)
            << "One stage must collect, plan, pack, transfer, and apply before its terminal event.";
        EXPECT_NE(contents.find("graph.addNode(\n            params.stage_name"),
                  std::string::npos);
        EXPECT_EQ(contents.find("moe_device_rebalance_maintenance_collect"),
                  std::string::npos);
        EXPECT_EQ(contents.find("moe_device_rebalance_maintenance_probe_after_snapshot"),
                  std::string::npos);
        EXPECT_EQ(contents.find("moe_device_rebalance_maintenance_metadata_payload"),
                  std::string::npos);
        EXPECT_EQ(contents.find("DeviceMoERebalanceStagePhase::PlanAndPreparePayloadAfterSideband"),
                  std::string::npos)
            << "Maintenance must not yield to the host after payload preparation.";
        EXPECT_EQ(contents.find("DeviceMoERebalanceStagePhase::TransferPreparedPayload"),
                  std::string::npos)
            << "Maintenance must not require a host-selected follow-up transport graph.";
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
        EXPECT_NE(ffn_body.find("ffn_terminal =\n                            graph_rebalance_plan_after_sideband_node"),
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
        EXPECT_NE(ffn_body.find("graph.addDependency(\n                        ar_name,\n                        prefix + \"moe_expert_ffn_overlay_fast\")"),
                  std::string::npos)
            << "The routed-expert allreduce must always wait directly for its tensor producer.";
        EXPECT_NE(ffn_body.find("if (!graph_rebalance_collect_node.empty())"),
                  std::string::npos)
            << "Rebalance sideband collection must be an additional allreduce producer.";
        EXPECT_NE(ffn_body.find("graph.addDependency(\n                            ar_name,\n                            graph_rebalance_collect_node)"),
                  std::string::npos)
            << "Allreduce anchors carrying rebalance sidebands must also wait for packed sideband state.";
        EXPECT_NE(ffn_body.find("graph.addDependency(\n                            graph_rebalance_plan_after_sideband_node,\n                            ar_name)"),
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
        EXPECT_NE(ffn_body.find("boundary_apply && !last_device_rebalance_decode_layer"),
                  std::string::npos)
            << "Boundary apply must be inserted once at the last serial/grouped decode layer so async transfers "
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
        EXPECT_EQ(stage_header.find("CollectAndGatherState"), std::string::npos)
            << "The retired probe-only maintenance graph must not remain selectable.";
        EXPECT_EQ(stage_header.find("PlanAndPreparePayloadAfterSideband"), std::string::npos)
            << "The retired host-separated planning graph must not remain selectable.";
        EXPECT_EQ(stage_header.find("TransferPreparedPayload"), std::string::npos)
            << "The retired host-selected transport graph must not remain selectable.";
        EXPECT_EQ(stage_header.find("PlanProbeAfterSideband"), std::string::npos)
            << "A plan-only maintenance phase permits the runtime directory to change before source packing.";
        EXPECT_EQ(stage_header.find("GatherCommandsAndCopyPreparedPayload"), std::string::npos)
            << "Transport must not gather stale plans or regenerate source payloads.";
        EXPECT_EQ(stage_header.find("CopyPreparedPayload"), std::string::npos)
            << "The old ambiguous copy phase did not encode immutable payload ownership.";
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
        EXPECT_NE(stage_source.find("packDeviceRebalanceHistograms(\n                        compute_launch,\n                        runtime_layers,\n                        local,\n                        params_.config,\n                        wave_state,\n                        controller_state"),
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
        const size_t planning_phase = stage_source.find("if (runsPlanning())");
        const size_t command_header_allgather =
            stage_source.find("workspaceSuffix() + \"_transfer_header\"", planning_phase);
        const size_t domain_projection =
            stage_source.find("if (!project_domain_commands(transfer_launch))",
                              command_header_allgather);
        const size_t planning_payload_prepare =
            stage_source.find("if (!prepare_local_payload())", domain_projection);
        const size_t planning_payload_transfer =
            stage_source.find("if (!transfer_prepared_payload(",
                              planning_payload_prepare);
        const size_t prepare_payload_lambda =
            stage_source.find("auto prepare_local_payload");
        const size_t source_descriptor_pack =
            stage_source.find("packDeviceRebalanceSourceDescriptors(", prepare_payload_lambda);
        ASSERT_NE(planning_phase, std::string::npos);
        ASSERT_NE(command_header_allgather, std::string::npos);
        ASSERT_NE(domain_projection, std::string::npos);
        ASSERT_NE(planning_payload_prepare, std::string::npos);
        ASSERT_NE(planning_payload_transfer, std::string::npos);
        ASSERT_NE(prepare_payload_lambda, std::string::npos);
        ASSERT_NE(source_descriptor_pack, std::string::npos);
        EXPECT_LT(planning_phase, command_header_allgather)
            << "The planning replay must gather command headers before it releases runtime ownership.";
        EXPECT_LT(command_header_allgather, domain_projection)
            << "Command projection must run after command/header allgather.";
        EXPECT_LT(domain_projection, planning_payload_prepare)
            << "The atomic replay must prepare payloads from projected local commands.";
        EXPECT_LT(planning_payload_prepare, planning_payload_transfer)
            << "The atomic replay must transport only after immutable source bytes are prepared.";
        EXPECT_LT(prepare_payload_lambda, source_descriptor_pack)
            << "Source descriptor packing belongs exclusively to payload preparation.";
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

        EXPECT_EQ(stage_source.find("CollectAndGatherState"), std::string::npos);
        EXPECT_EQ(stage_source.find("PlanAndPreparePayloadAfterSideband"), std::string::npos);
        EXPECT_EQ(stage_source.find("TransferPreparedPayload"), std::string::npos)
            << "No production branch may split payload transport into a host-selected follow-up replay.";
        EXPECT_NE(stage_source.find("wave_state"), std::string::npos);
        EXPECT_NE(stage_header.find("prepareGraphLaunch"), std::string::npos)
            << "Transfer stream/event resources should be prepared before graph capture begins.";
        EXPECT_NE(stage_header.find("needsGraphLaunchPreparation() const override { return usesTransferSlotApply(); }"),
                  std::string::npos);
        EXPECT_NE(stage_header.find("prepareForCapture("), std::string::npos)
            << "Every multi-stream MoE stage must share one typed pre-capture lane contract.";
        const size_t prepare_lane =
            stage_source.find("bool DeviceMoERebalanceTransferState::prepareForCapture");
        ASSERT_NE(prepare_lane, std::string::npos);
        const size_t ensure_transfer_state =
            stage_source.find("ensure(device, name_suffix)", prepare_lane);
        const size_t precapture_record =
            stage_source.find("recordEventChecked(",
                              ensure_transfer_state);
        const size_t precapture_wait =
            stage_source.find("waitEventChecked(",
                              precapture_record);
        ASSERT_NE(ensure_transfer_state, std::string::npos);
        ASSERT_NE(precapture_record, std::string::npos)
            << "Transfer-slot rebalance must publish prior auxiliary-stream work before graph capture.";
        ASSERT_NE(precapture_wait, std::string::npos)
            << "The capture stream must wait for the published auxiliary-stream boundary.";
        EXPECT_LT(ensure_transfer_state, precapture_record)
            << "The pre-capture event edge must run after transfer stream/event allocation.";
        EXPECT_LT(precapture_record, precapture_wait)
            << "The auxiliary stream must record completion before the capture stream waits.";

        const size_t prepare_launch =
            stage_source.find("bool MoEDeviceRebalanceStage::prepareGraphLaunch");
        ASSERT_NE(prepare_launch, std::string::npos);
        const size_t prepare_lane_call =
            stage_source.find("transfer_state->prepareForCapture(",
                              prepare_launch);
        const size_t fence_counter =
            stage_source.find("device_rebalance_precapture_transfer_stream_event_fence",
                              prepare_launch);
        ASSERT_NE(prepare_lane_call, std::string::npos)
            << "Maintenance capture must use the shared typed transfer-lane preflight.";
        EXPECT_NE(fence_counter, std::string::npos)
            << "Perfstats should prove the pre-capture event fence ran in e2e logs.";
        EXPECT_EQ(stage_source.find("synchronizeStreamChecked(transfer_state->transferStream())"),
                  std::string::npos)
            << "Capture preparation must never block the host on the auxiliary stream.";
        EXPECT_NE(stage_source.find("getOrCreateAuxiliaryStream"), std::string::npos)
            << "Transfer-slot arrivals must use a context-owned auxiliary stream.";
        EXPECT_NE(stage_source.find("gpu_ctx->recordEventChecked(transfer_state->computeReadyEvent(), stream)"),
                  std::string::npos)
            << "The compute->transfer edge must be captured as a graph event.";
        EXPECT_NE(stage_source.find("gpu_ctx->waitEventChecked(transfer_state->computeReadyEvent(), transfer_stream)"),
                  std::string::npos);
        EXPECT_NE(stage_source.find("MoEKernelLaunchContext transfer_launch"), std::string::npos)
            << "Packed expert payload copies must receive an immutable transfer-stream launch context.";
        EXPECT_NE(stage_source.find(".stream = transfer_stream"), std::string::npos)
            << "The transfer launch context must identify the context-owned auxiliary stream.";
        EXPECT_EQ(stage_source.find("moe_kernel->setGPUStream("), std::string::npos)
            << "Maintenance execution must never retarget the process-wide MoE kernel singleton.";
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
        EXPECT_NE(stage_source.find("inline_transfer_already_applied"), std::string::npos)
            << "The stage must encode transfer-owned apply as a completed state transition.";
        EXPECT_NE(stage_source.find("usesTransferSlotApply() && runsController()"),
                  std::string::npos)
            << "A controller phase with inline payload transport owns exactly one ready-wave apply.";
        EXPECT_NE(stage_source.find("!inline_transfer_already_applied"),
                  std::string::npos)
            << "The compute-stream apply path must exclude waves already applied on the transfer stream.";
        EXPECT_NE(stage_source.find("DeviceMoERebalanceStagePhase::PlanCopyApply"),
                  std::string::npos)
            << "Atomic maintenance remains an explicit stage phase.";
        EXPECT_NE(stage_source.find("gpu_ctx->waitEventChecked(transfer_state->transferDoneEvent(), stream)"),
                  std::string::npos)
            << "The atomic graph terminal stream must still join its auxiliary transfer work.";
        EXPECT_NE(stage_source.find("Failed to queue compute-to-transfer stream dependency"), std::string::npos);
        EXPECT_EQ(stage_source.find("Failed to queue transfer-to-apply stream dependency"), std::string::npos)
            << "Inline payload transfer applies on its owning stream; a second compute-stream apply wait is obsolete.";
        EXPECT_NE(stage_source.find("transfer-stream completion event"), std::string::npos);
        EXPECT_EQ(stage_source.find("synchronizeStream("), std::string::npos)
            << "Graph-side rebalance publish/apply must not block the host.";
        EXPECT_EQ(stage_source.find("LLAMINAR_MOE_REBALANCE_DEBUG_SYNC"), std::string::npos)
            << "Do not leave crash-localization sync knobs in the production rebalance path.";
        EXPECT_EQ(stage_source.find("transfer_event_backend_->recordEvent"), std::string::npos)
            << "The transfer stage must use its worker context so the event belongs to the exact graph-capture stream and device context.";

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

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, DeviceSideRebalanceRunsAtCommittedDecodeBoundary)
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
        const fs::path qwen_moe_graph_path =
            root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        const fs::path device_rebalance_stage_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.cpp";
        const fs::path debug_env_path = root / "src/v2/utils/DebugEnv.h";
        const fs::path iface_path = root / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h";
        const fs::path chat_path = root / "src/v2/app/modes/ChatCompletionHandler.cpp";
        const fs::path benchmark_path = root / "src/v2/app/modes/BenchmarkMode.cpp";
        const fs::path benchmark_runner_path = root / "src/v2/utils/BenchmarkRunner.cpp";
        const fs::path adapter_path = root / "src/v2/app/InferenceRunnerAdapter.cpp";
        const fs::path adapter_header_path = root / "src/v2/app/InferenceRunnerAdapter.h";
        const fs::path qwen36_parity_path =
            root / "tests/v2/integration/parity/qwen36/Qwen36MoEParityTestBase.h";
        const fs::path generic_parity_path =
            root / "tests/v2/integration/parity/ParityTestBase.h";
        const fs::path expert_overlay_parity_path =
            root / "tests/v2/integration/parity/qwen36/Test__Qwen36MoE_ExpertOverlay_MathParity.cpp";
        const fs::path expert_overlay_prefix_mtp_path =
            root / "tests/v2/integration/parity/qwen36/"
                   "Test__Qwen36MoE_ExpertOverlay_PrefixMTP_Parity.cpp";
        const fs::path server_e2e_path =
            root / "tests/v2/e2e/server/test_server_e2e.sh";
        ASSERT_TRUE(fs::exists(runner_path)) << runner_path;
        ASSERT_TRUE(fs::exists(dgo_path)) << dgo_path;
        ASSERT_TRUE(fs::exists(dgo_header_path)) << dgo_header_path;
        ASSERT_TRUE(fs::exists(rank_path)) << rank_path;
        ASSERT_TRUE(fs::exists(rank_header_path)) << rank_header_path;
        ASSERT_TRUE(fs::exists(graph_builder_path)) << graph_builder_path;
        ASSERT_TRUE(fs::exists(qwen_moe_graph_path)) << qwen_moe_graph_path;
        ASSERT_TRUE(fs::exists(device_rebalance_stage_path)) << device_rebalance_stage_path;
        ASSERT_TRUE(fs::exists(debug_env_path)) << debug_env_path;
        ASSERT_TRUE(fs::exists(iface_path)) << iface_path;
        ASSERT_TRUE(fs::exists(chat_path)) << chat_path;
        ASSERT_TRUE(fs::exists(benchmark_path)) << benchmark_path;
        ASSERT_TRUE(fs::exists(benchmark_runner_path)) << benchmark_runner_path;
        ASSERT_TRUE(fs::exists(adapter_path)) << adapter_path;
        ASSERT_TRUE(fs::exists(adapter_header_path)) << adapter_header_path;
        ASSERT_TRUE(fs::exists(qwen36_parity_path)) << qwen36_parity_path;
        ASSERT_TRUE(fs::exists(generic_parity_path)) << generic_parity_path;
        ASSERT_TRUE(fs::exists(expert_overlay_parity_path)) << expert_overlay_parity_path;
        ASSERT_TRUE(fs::exists(expert_overlay_prefix_mtp_path))
            << expert_overlay_prefix_mtp_path;
        ASSERT_TRUE(fs::exists(server_e2e_path)) << server_e2e_path;

        const std::string runner = readFile(runner_path);
        const std::string dgo = readFile(dgo_path);
        const std::string dgo_header = readFile(dgo_header_path);
        const std::string rank = readFile(rank_path);
        const std::string rank_header = readFile(rank_header_path);
        const std::string graph_builder = readFile(graph_builder_path);
        const std::string qwen_moe_graph = readFile(qwen_moe_graph_path);
        const std::string device_rebalance_stage = readFile(device_rebalance_stage_path);
        const std::string debug_env = readFile(debug_env_path);
        const std::string iface = readFile(iface_path);
        const std::string chat = readFile(chat_path);
        const std::string benchmark = readFile(benchmark_path);
        const std::string benchmark_runner = readFile(benchmark_runner_path);
        const std::string adapter = readFile(adapter_path);
        const std::string adapter_header = readFile(adapter_header_path);
        const std::string qwen36_parity = readFile(qwen36_parity_path);
        const std::string generic_parity = readFile(generic_parity_path);
        const std::string expert_overlay_parity = readFile(expert_overlay_parity_path);
        const std::string expert_overlay_prefix_mtp =
            readFile(expert_overlay_prefix_mtp_path);
        const std::string server_e2e = readFile(server_e2e_path);
        ASSERT_FALSE(runner.empty()) << runner_path;
        ASSERT_FALSE(dgo.empty()) << dgo_path;
        ASSERT_FALSE(dgo_header.empty()) << dgo_header_path;
        ASSERT_FALSE(rank.empty()) << rank_path;
        ASSERT_FALSE(rank_header.empty()) << rank_header_path;
        ASSERT_FALSE(graph_builder.empty()) << graph_builder_path;
        ASSERT_FALSE(qwen_moe_graph.empty()) << qwen_moe_graph_path;
        ASSERT_FALSE(device_rebalance_stage.empty()) << device_rebalance_stage_path;
        ASSERT_FALSE(debug_env.empty()) << debug_env_path;
        ASSERT_FALSE(iface.empty()) << iface_path;
        ASSERT_FALSE(chat.empty()) << chat_path;
        ASSERT_FALSE(benchmark.empty()) << benchmark_path;
        ASSERT_FALSE(benchmark_runner.empty()) << benchmark_runner_path;
        ASSERT_FALSE(adapter.empty()) << adapter_path;
        ASSERT_FALSE(adapter_header.empty()) << adapter_header_path;
        ASSERT_FALSE(qwen36_parity.empty()) << qwen36_parity_path;
        ASSERT_FALSE(generic_parity.empty()) << generic_parity_path;
        ASSERT_FALSE(expert_overlay_parity.empty()) << expert_overlay_parity_path;
        ASSERT_FALSE(expert_overlay_prefix_mtp.empty())
            << expert_overlay_prefix_mtp_path;
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
        EXPECT_EQ(dgo.find("device_rebalance_graph_controller"), std::string::npos)
            << "Topology, not a debug toggle, must select homogeneous GPU device-side rebalance.";
        EXPECT_NE(graph_builder.find("buildDeviceMoERebalanceMaintenanceGraph("),
                  std::string::npos)
            << "Graph builders expose first-class rolling maintenance replay for device-side rebalance.";
        EXPECT_NE(graph_builder.find("return {};"), std::string::npos)
            << "Non-MoE graph builders should opt out with an empty maintenance graph.";
        EXPECT_NE(dgo_header.find("DeviceMoERebalanceMaintenanceGraphCache"), std::string::npos);
        EXPECT_EQ(dgo_header.find("device_moe_rebalance_maintenance_payload_graphs_"), std::string::npos)
            << "Atomic maintenance must not retain host-selected payload graph variants.";
        EXPECT_EQ(dgo_header.find("std::unordered_map<uint64_t, DeviceMoERebalanceMaintenanceGraphCache>"),
                  std::string::npos)
            << "One device-owned maintenance transaction owns one graph cache.";
        EXPECT_NE(dgo_header.find("enum class ResetBoundary"), std::string::npos)
            << "Maintenance reset policy must classify stable contents separately "
               "from captured binding identity.";
        const size_t reset_inference_start =
            dgo_header.find("void resetInferenceState(const InferenceStateResetRequest &request) override");
        ASSERT_NE(reset_inference_start, std::string::npos);
        const size_t reset_inference_end = dgo_header.find("void clear_cache() override", reset_inference_start);
        ASSERT_NE(reset_inference_end, std::string::npos);
        const std::string reset_inference_body =
            dgo_header.substr(reset_inference_start, reset_inference_end - reset_inference_start);
        const size_t clear_cache_drain =
            reset_inference_body.find(
                "drainCompletedDeviceMoERebalanceMaintenanceDiagnostics(");
        const size_t clear_cache_reset =
            reset_inference_body.find(
                "device_moe_rebalance_maintenance_graph_.reset(\n"
                "                    DeviceMoERebalanceMaintenanceGraphCache::ResetBoundary::\n"
                "                        StableDeviceStateContents)");
        EXPECT_EQ(reset_inference_body.find("ctx->synchronize()"), std::string::npos)
            << "Request reset must join only explicitly owned transactions, never the whole GPU.";
        EXPECT_EQ(reset_inference_body.find("clearLastError()"), std::string::npos)
            << "Request reset must not erase an asynchronous backend failure before its owner attributes it.";
        EXPECT_EQ(dgo_header.find("clearLastError()"), std::string::npos)
            << "The GPU worker contract must not expose an API whose only effect is to erase an unattributed backend error.";
        EXPECT_EQ(dgo.find("clearLastError()"), std::string::npos)
            << "Graph orchestration must propagate checked backend errors instead of clearing them.";
        EXPECT_NE(reset_inference_body.find("\"request_reset_exact_device_epilogue\""),
                  std::string::npos)
            << "PerfStats must attribute the exact request-boundary device epilogue.";
        ASSERT_NE(clear_cache_drain, std::string::npos)
            << "Request reset should export maintenance diagnostics through the exact stream-owned epilogue.";
        const std::string clear_cache_drain_call =
            reset_inference_body.substr(clear_cache_drain, 220);
        EXPECT_NE(clear_cache_drain_call.find("\"request_reset\""),
                  std::string::npos);
        EXPECT_NE(clear_cache_drain_call.find("reset_reason"),
                  std::string::npos);
        ASSERT_NE(clear_cache_reset, std::string::npos);
        EXPECT_LT(clear_cache_drain, clear_cache_reset);
        EXPECT_NE(dgo.find("epilogue_tags[\"reset\"] = reset_operation;"),
                  std::string::npos)
            << "Request-reset maintenance exports must identify the concrete "
               "reset operation so E2E lifecycle evidence can distinguish "
               "clear_cache from other request boundaries.";
        const size_t prefix_restore_moe_comment =
            reset_inference_body.find("Both ordinary request reset and prefix restore mutate only");
        ASSERT_NE(prefix_restore_moe_comment, std::string::npos)
            << "Prefix restore must document that it mutates contents behind "
               "model-lifetime device addresses.";
        EXPECT_EQ(
            reset_inference_body.find(
                "device_moe_rebalance_maintenance_graph_.invalidate()",
                prefix_restore_moe_comment),
            std::string::npos)
            << "Prefix restore must not destroy a maintenance executable whose "
               "runtime table and transfer-slot addresses remain stable.";
        EXPECT_NE(
            reset_inference_body.find(
                "StableDeviceStateContents",
                prefix_restore_moe_comment),
            std::string::npos)
            << "Prefix restore must use the same typed stable-content boundary "
               "as an ordinary request reset.";
        EXPECT_EQ(
            reset_inference_body.find(
                "device_moe_rebalance_maintenance_payload_graphs_",
                prefix_restore_moe_comment),
            std::string::npos)
            << "Prefix restore must not resurrect the retired host-selected payload cache.";
        const size_t clear_cache_start = reset_inference_end;
        const size_t clear_cache_end = dgo_header.find("int get_position() const override", clear_cache_start);
        ASSERT_NE(clear_cache_end, std::string::npos);
        const std::string clear_cache_body =
            dgo_header.substr(clear_cache_start, clear_cache_end - clear_cache_start);
        EXPECT_NE(clear_cache_body.find("InferenceStateResetRequest::requestBoundary(\"clear_cache\")"),
                  std::string::npos)
            << "clear_cache must enter the request-boundary reset path that drains exact maintenance diagnostics.";
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
        const size_t typed_reset =
            maintenance_cache_body.find("void reset(ResetBoundary boundary)");
        ASSERT_NE(typed_reset, std::string::npos);
        const size_t invalidate =
            maintenance_cache_body.find("void invalidate()", typed_reset);
        ASSERT_NE(invalidate, std::string::npos);
        const std::string stable_contents_reset_body =
            maintenance_cache_body.substr(typed_reset,
                                          invalidate - typed_reset);
        EXPECT_NE(stable_contents_reset_body.find("completion_event_in_flight = false;"),
                  std::string::npos)
            << "The exact maintenance epilogue completes status readback before replay state is preserved, so stale request events must not force a next-request diagnostics readback.";
        EXPECT_EQ(maintenance_cache_body.find("completion_event_consumer_wait_pending"),
                  std::string::npos)
            << "A shared consumable boolean cannot represent the independent "
               "sidecar and main-forward ordering edges.";
        EXPECT_NE(stable_contents_reset_body.find("durable completion"),
                  std::string::npos)
            << "Stable request reset must preserve the durable event publication.";
        EXPECT_NE(stable_contents_reset_body.find(
                      "resetSessionStatePreservingCapturedReplay()"),
                  std::string::npos)
            << "A stable-content reset must preserve the captured maintenance "
               "executable and reset only replay-safe stage session state.";
        EXPECT_EQ(maintenance_cache_body.find("skipped_inflight_count"),
                  std::string::npos)
            << "Atomic maintenance no longer polls or skips host-observed in-flight waves.";
        EXPECT_EQ(maintenance_cache_body.find("timing_start_event"),
                  std::string::npos)
            << "Maintenance cache must not own per-wave timing-event allocations.";
        EXPECT_NE(dgo.find("drainCompletedDeviceMoERebalanceMaintenanceDiagnostics"),
                  std::string::npos);
        EXPECT_EQ(dgo.find("deviceMoERebalanceTracePathFromEnv().empty())\n        {\n            cache.completion_event_in_flight = false;"),
                  std::string::npos)
            << "Request reset must never silently discard in-flight device maintenance when diagnostics are disabled.";
        EXPECT_EQ(dgo.find("gpu_ctx->synchronizeStreamChecked(maintenance_stream)"),
                  std::string::npos)
            << "The diagnostic epilogue must not pre-synchronize the maintenance stream before its ordered status readback.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_epilogue_event_queries\""),
                  std::string::npos)
            << "PerfStats must record whether the maintenance event was already complete at the epilogue.";
        EXPECT_NE(dgo.find("backend->deviceToHostOnStream("),
                  std::string::npos)
            << "Maintenance status must be copied on its owning stream.";
        EXPECT_NE(dgo.find("backend->synchronizeStream(maintenance_stream, device_ordinal)"),
                  std::string::npos)
            << "The final status readback is the sole exact host boundary for maintenance diagnostics.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_request_reset_exports\""),
                  std::string::npos)
            << "Final measured maintenance waves must remain visible even when the next request reset retires the event.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_diagnostic_exports\""),
                  std::string::npos)
            << "Request epilogues need a non-reset counter for final device-status publication.";
        EXPECT_NE(dgo_header.find("last_status_publication_launch_count"),
                  std::string::npos)
            << "The epilogue must distinguish an unpublished device launch from a status already surfaced by profiling.";
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
        const size_t greedy_mtp_parity_start =
            qwen36_parity.find("inline void runMoEMTPParity(");
        ASSERT_NE(greedy_mtp_parity_start, std::string::npos);
        const size_t greedy_mtp_first_drain =
            qwen36_parity.find(
                "mtp->drainCompletedDecodeBoundaryMaintenanceDiagnostics();",
                greedy_mtp_parity_start);
        const size_t greedy_mtp_first_snapshot =
            qwen36_parity.find(
                "const auto first_records = PerfStatsCollector::snapshot(",
                greedy_mtp_parity_start);
        ASSERT_NE(greedy_mtp_first_drain, std::string::npos);
        ASSERT_NE(greedy_mtp_first_snapshot, std::string::npos);
        EXPECT_LT(greedy_mtp_first_drain, greedy_mtp_first_snapshot)
            << "Greedy MTP parity must export completed plan/payload maintenance "
               "diagnostics before certifying request-local movement counters.";
        const size_t stochastic_mtp_parity_start =
            qwen36_parity.find("inline void runMoEStochasticMTPVerifierParity(");
        ASSERT_NE(stochastic_mtp_parity_start, std::string::npos);
        const size_t stochastic_mtp_first_drain =
            qwen36_parity.find(
                "mtp->drainCompletedDecodeBoundaryMaintenanceDiagnostics();",
                stochastic_mtp_parity_start);
        const size_t stochastic_mtp_first_snapshot =
            qwen36_parity.find(
                "const auto first_mtp_records = PerfStatsCollector::snapshot(",
                stochastic_mtp_parity_start);
        ASSERT_NE(stochastic_mtp_first_drain, std::string::npos);
        ASSERT_NE(stochastic_mtp_first_snapshot, std::string::npos);
        EXPECT_LT(stochastic_mtp_first_drain, stochastic_mtp_first_snapshot)
            << "Stochastic MTP parity must use the same completed-maintenance "
               "diagnostic epilogue before its request-local PerfStats snapshot.";

        const size_t verifier_factory_start =
            qwen36_parity.find(
                "inline MoEVerifierProofRunner createMoEVerifierProofRunner(");
        const size_t verifier_factory_end =
            qwen36_parity.find(
                "inline std::shared_ptr<MoERoutedExpertPlacementPlan>",
                verifier_factory_start);
        ASSERT_NE(verifier_factory_start, std::string::npos);
        ASSERT_NE(verifier_factory_end, std::string::npos);
        const std::string verifier_factory =
            qwen36_parity.substr(
                verifier_factory_start,
                verifier_factory_end - verifier_factory_start);
        EXPECT_NE(
            verifier_factory.find(
                "effective_config.moe_rebalance = *test_case.moe_rebalance"),
            std::string::npos)
            << "Focused grouped-verifier proofs must inherit the fixture's "
               "declared Dynamic/LLEP policy instead of silently running the "
               "InferenceRunnerConfig default.";
        EXPECT_NE(
            verifier_factory.find(
                "effective_config.tp_allreduce_precision_override ="),
            std::string::npos)
            << "Focused verifier proofs must preserve the fixture's collective "
               "precision when proving byte-equivalent grouped execution.";

        for (const char *helper : {
                 "inline void runMoEMainVerifierAllPositionRowsMatchSerialDecode(",
                 "inline void runMoEMainVerifierDecodeEquivalentRowsMatchSerialDecode(",
                 "inline void runMoEMainVerifierGroupedRowsMatchSerialDecode("})
        {
            const size_t helper_start = qwen36_parity.find(helper);
            ASSERT_NE(helper_start, std::string::npos) << helper;
            const size_t helper_body = qwen36_parity.find('{', helper_start);
            ASSERT_NE(helper_body, std::string::npos) << helper;
            const size_t next_helper =
                qwen36_parity.find("\n    inline void ", helper_body + 1);
            const std::string region =
                qwen36_parity.substr(
                    helper_body,
                    next_helper == std::string::npos
                        ? std::string::npos
                        : next_helper - helper_body);
            EXPECT_NE(
                region.find(
                    "ScopedMoEPrefixCaseEnvironment case_env(test_case.env_overrides)"),
                std::string::npos)
                << helper
                << " must apply backend-specific cache, LLEP, and graph "
                   "environment policy for the full proof-runner lifetime.";
        }
        const size_t generic_drain_helper =
            generic_parity.find(
                "void activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics()");
        ASSERT_NE(generic_drain_helper, std::string::npos)
            << "Generic parity must expose the same non-reset diagnostics epilogue as focused MTP parity.";
        const size_t generic_due_helper =
            generic_parity.find(
                "bool parityMoERebalanceMaintenanceDue(");
        ASSERT_NE(generic_due_helper, std::string::npos);
        const size_t generic_drive_helper =
            generic_parity.find(
                "bool driveParityMoERebalanceMaintenance(");
        ASSERT_NE(generic_drive_helper, std::string::npos);
        ASSERT_LT(generic_due_helper, generic_drive_helper);
        const std::string generic_due_body =
            generic_parity.substr(
                generic_due_helper,
                generic_drive_helper - generic_due_helper);
        EXPECT_NE(
            generic_due_body.find(
                "activeUsesDeviceSideMoERebalanceController()"),
            std::string::npos)
            << "A device-owned scheduler needs one committed-boundary tick per decode transaction.";
        const size_t long_context_config_start =
            expert_overlay_parity.find(
                "ExpertOverlayParityConfig longContextConfig(");
        ASSERT_NE(long_context_config_start, std::string::npos);
        const size_t long_context_config_end =
            expert_overlay_parity.find(
                "const std::vector<ExpertOverlayParityConfig>",
                long_context_config_start);
        ASSERT_NE(long_context_config_end, std::string::npos);
        const std::string long_context_config_body =
            expert_overlay_parity.substr(
                long_context_config_start,
                long_context_config_end - long_context_config_start);
        EXPECT_NE(
            long_context_config_body.find(
                "rebalance_mode != MoERebalanceRuntimeMode::Dynamic"),
            std::string::npos);
        EXPECT_NE(
            long_context_config_body.find(
                "rebalance_mode != MoERebalanceRuntimeMode::LLEP"),
            std::string::npos)
            << "Long-context LLEP parity must enable the same committed-boundary "
               "maintenance exercise that its PerfStats gate certifies.";
        EXPECT_NE(
            long_context_config_body.find(
                "config.moe_rebalance_exercise.enabled = true;"),
            std::string::npos);
        const size_t long_context_perf_start =
            expert_overlay_parity.find(
                "void expectLongContextMoERebalancePerfPath(");
        ASSERT_NE(long_context_perf_start, std::string::npos);
        const size_t long_context_perf_end =
            expert_overlay_parity.find(
                "ExpertOverlayParityConfig baseConfig(",
                long_context_perf_start);
        ASSERT_NE(long_context_perf_end, std::string::npos);
        const std::string long_context_perf_body =
            expert_overlay_parity.substr(
                long_context_perf_start,
                long_context_perf_end - long_context_perf_start);
        for (const char *cumulative_counter : {
                 "device_rebalance_controller_decode_apply_hits",
                 "device_rebalance_wave_copied_arrivals_total",
                 "device_rebalance_wave_applied_arrivals_total",
                 "device_rebalance_wave_applied_layer_count_total"})
        {
            EXPECT_NE(
                long_context_perf_body.find(cumulative_counter),
                std::string::npos)
                << "Long-context movement certification must use cumulative "
                   "request evidence for "
                << cumulative_counter;
        }
        EXPECT_EQ(
            long_context_perf_body.find(
                "expectPerfCounterPositive(\n"
                "                records,\n"
                "                \"moe_rebalance\",\n"
                "                \"device_rebalance_changed_layers\""),
            std::string::npos)
            << "Per-wave planner status is not a stable request-level movement "
               "certificate; a final economical no-op may follow an applied wave.";
        EXPECT_NE(
            qwen36_parity.find(
                "inline std::string qwen36MoELongNeedleParityPrompt()"),
            std::string::npos)
            << "Long-context MoE suites must share one deterministic, "
               "self-regenerating ledger prompt.";
        EXPECT_EQ(
            expert_overlay_parity.find(
                "std::string qwen36MoELongNeedleParityPrompt()"),
            std::string::npos)
            << "The math parity file must not retain a private copy of the "
               "shared long-ledger builder.";
        EXPECT_NE(
            qwen36_parity.find(
                "test_case.minimum_prompt_tokens"),
            std::string::npos)
            << "Metadata authentication must include workload geometry, not "
               "only prompt string identity.";
        EXPECT_NE(
            expert_overlay_prefix_mtp.find(
                "test_case.prompt = qwen36MoELongNeedleParityPrompt();"),
            std::string::npos)
            << "Phase-split prefix/MTP fixtures must carry the complete ledger "
               "when metadata regeneration is required.";
        EXPECT_EQ(
            expert_overlay_prefix_mtp.find(
                "test_case.prompt = \"Task: read the ledger"),
            std::string::npos)
            << "A header-only long-context fixture can silently regenerate a "
               "short prompt and invalidate partial-prefix coverage.";
        EXPECT_NE(
            expert_overlay_prefix_mtp.find(
                "test_case.minimum_prompt_tokens = 640;"),
            std::string::npos);
        EXPECT_NE(
            expert_overlay_prefix_mtp.find(
                "MoEReferenceInputSource::ModelTokenizer"),
            std::string::npos)
            << "State-lifetime prefix/MTP tests should use production GGUF "
               "tokenization instead of regenerating unused PyTorch decode evidence.";
        EXPECT_NE(
            qwen36_parity.find(
                ".target_is_gpu = true"),
            std::string::npos)
            << "Metadata-only GGUF tokenization must remain demand-paged for "
               "large GPU fixtures.";
        EXPECT_NE(
            qwen36_parity.find(
                "expected_tokens->clear();"),
            std::string::npos)
            << "Model-tokenizer fixtures must not fabricate PyTorch oracle tokens.";
        const size_t overlay_model_context_start =
            expert_overlay_parity.find(
                "std::shared_ptr<ModelContext> getOrCreateOverlayModelContext(");
        ASSERT_NE(overlay_model_context_start, std::string::npos);
        const size_t overlay_model_context_end =
            expert_overlay_parity.find(
                "void evictOverlayPipelineCacheIfNeeded(",
                overlay_model_context_start);
        ASSERT_NE(overlay_model_context_end, std::string::npos);
        const std::string overlay_model_context_body =
            expert_overlay_parity.substr(
                overlay_model_context_start,
                overlay_model_context_end - overlay_model_context_start);
        EXPECT_NE(
            overlay_model_context_body.find(
                ".target_is_gpu = true"),
            std::string::npos)
            << "GPU-only expert-overlay parity must select demand-paged model mmap "
               "instead of whole-file CPU NUMA first-touch.";
        EXPECT_NE(
            overlay_model_context_body.find(
                ".mpi_ctx = mpi_ctx"),
            std::string::npos)
            << "The GPU model loader should use the active MPI topology rather "
               "than constructing a test-only single-rank TensorFactory.";
        EXPECT_NE(
            overlay_model_context_body.find(
                "ModelContext::create(config.model_path, model_config)"),
            std::string::npos);
        EXPECT_EQ(
            overlay_model_context_body.find(
                "ModelContext::create(\n            config.model_path,\n            nullptr"),
            std::string::npos)
            << "The legacy ModelContext factory silently selects CPU first-touch.";
        const size_t generic_assert_helper =
            generic_parity.find(
                "void assertParityMoERebalanceExercise(",
                generic_drive_helper);
        ASSERT_NE(generic_assert_helper, std::string::npos);
        const std::string generic_drive_body =
            generic_parity.substr(
                generic_drive_helper,
                generic_assert_helper - generic_drive_helper);
        EXPECT_NE(
            generic_drive_body.find(
                "runner->maybeApplyDecodeBoundaryMaintenance()"),
            std::string::npos)
            << "Direct parity runners must invoke the same post-commit device maintenance hook as serving.";
        EXPECT_NE(
            generic_drive_body.find(
                "orch_runner_->maybeApplyMoERebalance()"),
            std::string::npos)
            << "Orchestration parity runners must enter the shared committed-boundary hook.";
        const auto expect_final_movement_publication =
            [&](const char *decode_entry, const char *next_entry)
        {
            const size_t decode_start = generic_parity.find(decode_entry);
            ASSERT_NE(decode_start, std::string::npos) << decode_entry;
            const size_t decode_end =
                generic_parity.find(next_entry, decode_start);
            ASSERT_NE(decode_end, std::string::npos) << decode_entry;
            const std::string decode_body =
                generic_parity.substr(decode_start, decode_end - decode_start);
            EXPECT_NE(
                decode_body.find(
                    "parityMoERebalanceMaintenanceDue(step + 1)"),
                std::string::npos)
                << decode_entry
                << " must tick device-owned maintenance after every committed decode.";
            const size_t final_drain =
                decode_body.rfind(
                    "activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics();");
            const size_t movement_assert =
                decode_body.rfind("assertParityMoERebalanceExercise(");
            ASSERT_NE(final_drain, std::string::npos) << decode_entry;
            ASSERT_NE(movement_assert, std::string::npos) << decode_entry;
            EXPECT_LT(final_drain, movement_assert)
                << decode_entry
                << " must publish final graph-owned movement before reading the host-facing epoch.";
        };
        expect_final_movement_publication(
            "DecodeParitySummary runTPDecodeParity()",
            "void assertTPParity(");
        expect_final_movement_publication(
            "DecodeParitySummary runDecodeParity()",
            "void renderDecodeParityTable(");
        EXPECT_NE(dgo.find("DeviceGraphOrchestrator::maybeApplyDecodeBoundaryMaintenance()"),
                  std::string::npos)
            << "Device-owned rolling maintenance must remain explicit at the committed decode boundary.";
        EXPECT_EQ(dgo.find("maybeRunDeviceMoERebalanceMaintenanceGraph(effective_input)"),
                  std::string::npos)
            << "Raw grouped forward must not publish maintenance before verifier commit or rollback completes.";
        const size_t committed_maintenance_start =
            dgo.find("bool DeviceGraphOrchestrator::maybeApplyDecodeBoundaryMaintenance()");
        ASSERT_NE(committed_maintenance_start, std::string::npos);
        const size_t committed_maintenance_end =
            dgo.find("bool DeviceGraphOrchestrator::exportCompletedDeviceMoERebalanceMaintenanceStats",
                     committed_maintenance_start);
        ASSERT_NE(committed_maintenance_end, std::string::npos);
        const std::string committed_maintenance_body =
            dgo.substr(
                committed_maintenance_start,
                committed_maintenance_end - committed_maintenance_start);
        EXPECT_EQ(committed_maintenance_body.find("getPosition("), std::string::npos)
            << "Device-owned grouped MTP maintenance must not gate scheduling on the stale host position mirror.";
        EXPECT_NE(committed_maintenance_body.find("maybeRunDeviceMoERebalanceMaintenanceGraph()"),
                  std::string::npos)
            << "The committed boundary must tick the device-resident maintenance scheduler without host token metadata.";
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
            << "Static/off rebalance must never enter the device-side dynamic controller.";
        const size_t graph_stable_gate =
            device_controller_predicate.find("usesGraphStableGpuMoERebalance()");
        const size_t graph_stable_gate_end =
            device_controller_predicate.find("const DeviceId primary_device", graph_stable_gate);
        ASSERT_NE(graph_stable_gate, std::string::npos);
        ASSERT_NE(graph_stable_gate_end, std::string::npos);
        const std::string recognized_graph_stable_body =
            device_controller_predicate.substr(graph_stable_gate_end);
        EXPECT_EQ(recognized_graph_stable_body.find("return false;"), std::string::npos)
            << "Once a homogeneous graph-stable GPU domain is recognized, every invalid binding must fail hard instead of selecting host maintenance.";
        EXPECT_GE(
            countOccurrences(
                recognized_graph_stable_body,
                "host publish/apply fallback is refused"),
            6u)
            << "Every post-recognition topology/capability rejection must name the forbidden fallback.";
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
        const size_t maintenance_boundary_timer =
            maintenance_body.find("\"device_maintenance_graph_boundary_enqueue\"");
        ASSERT_NE(maintenance_env_gate, std::string::npos);
        ASSERT_NE(maintenance_dynamic_gate, std::string::npos)
            << "Static/off rebalance must return before the maintenance hook does any per-token work.";
        ASSERT_NE(maintenance_device_gate, std::string::npos);
        ASSERT_NE(maintenance_controller_gate, std::string::npos);
        ASSERT_NE(maintenance_boundary_timer, std::string::npos);
        EXPECT_LT(maintenance_env_gate, maintenance_dynamic_gate);
        EXPECT_LT(maintenance_dynamic_gate, maintenance_device_gate)
            << "Static/off rebalance must not pay GPU-state or position checks when maintenance env is set.";
        EXPECT_LT(maintenance_dynamic_gate, maintenance_controller_gate)
            << "Static/off rebalance must not walk controller/overlay state when maintenance env is set.";
        EXPECT_EQ(debug_env.find("device_rebalance_graph_controller"),
                  std::string::npos)
            << "The retired host-controller selection knob must not return.";
        EXPECT_EQ(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_GRAPH_CONTROLLER"),
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
        EXPECT_EQ(dgo.find("DeviceMoERebalanceMaintenanceGraphKind"),
                  std::string::npos)
            << "The scheduler must not select a host-owned maintenance graph kind.";
        EXPECT_EQ(dgo.find("domain_decision.launchPayloadGraph()"),
                  std::string::npos)
            << "The device transaction must not require a host payload decision.";
        EXPECT_EQ(dgo.find("MoEDeviceRebalanceHostRendezvous"),
                  std::string::npos)
            << "Homogeneous GPU maintenance must not rendezvous through host state.";
        EXPECT_EQ(dgo.find("device_moe_rebalance_decode_tokens_seen_"), std::string::npos)
            << "A host decode counter would make grouped MTP cadence depend on transaction depth.";
        EXPECT_NE(dgo.find("deviceMoERebalanceControllerStateDevice"), std::string::npos)
            << "Grouped summaries must bind the persistent device-owned maintenance clock.";
        EXPECT_NE(dgo.find("enqueueAdvanceSpeculativeCommitBoundary"), std::string::npos)
            << "The compact outcome must advance maintenance from committed device metadata.";
        EXPECT_NE(dgo.find("active_cache.segment_cache.ensureCaptureStream"), std::string::npos)
            << "Maintenance must use an explicit graph-capture stream.";
        const size_t maintenance_scheduler_begin =
            dgo.find("bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph()");
        const size_t maintenance_scheduler_end =
            dgo.find("// =====================================================================\n"
                     "    // IForwardExecutionHost interface implementations",
                     maintenance_scheduler_begin);
        ASSERT_NE(maintenance_scheduler_begin, std::string::npos);
        ASSERT_NE(maintenance_scheduler_end, std::string::npos);
        const std::string maintenance_scheduler =
            dgo.substr(
                maintenance_scheduler_begin,
                maintenance_scheduler_end - maintenance_scheduler_begin);
        EXPECT_NE(maintenance_scheduler.find(
                      "waitForLiveInferenceStateReadyForObservation(\n"
                      "                    maintenance_stream,\n"
                      "                    \"moe_device_rebalance_maintenance_before_capture\",\n"
                      "                    DeviceTimelineRole::MoERebalanceMaintenance)"),
                  std::string::npos)
            << "Maintenance must observe every committed device-state producer, including "
               "an all-position MTP verifier whose one-shot logits handoff was already consumed.";
        EXPECT_EQ(maintenance_scheduler.find(
                      "peekPendingLogitsStream(PendingLogitsStreamRole::MainDecode)"),
                  std::string::npos)
            << "A MainDecode-only dependency recreates the grouped-verifier histogram race.";
        const size_t live_state_wait =
            maintenance_scheduler.find(
                "waitForLiveInferenceStateReadyForObservation(\n"
                "                    maintenance_stream,\n"
                "                    \"moe_device_rebalance_maintenance_before_capture\",\n"
                "                    DeviceTimelineRole::MoERebalanceMaintenance)");
        const size_t maintenance_replay =
            maintenance_scheduler.find(
                "tryLaunchCapturedMoERebalanceMaintenanceGraphDirect");
        ASSERT_NE(live_state_wait, std::string::npos);
        ASSERT_NE(maintenance_replay, std::string::npos);
        EXPECT_LT(live_state_wait, maintenance_replay)
            << "The device event barrier must be queued before maintenance capture or replay begins.";
        const std::string steady_dependency_region =
            maintenance_scheduler.substr(
                live_state_wait,
                maintenance_replay - live_state_wait);
        EXPECT_EQ(steady_dependency_region.find("synchronizeStream"), std::string::npos)
            << "Steady maintenance ordering must remain device-side; request-reset diagnostics own "
               "the separate checked stream drain.";
        EXPECT_NE(dgo.find("policy.defer_final_sync = true"), std::string::npos)
            << "Maintenance replay should not force a host sync at the launch boundary.";
        EXPECT_NE(dgo.find("tryLaunchCapturedMoERebalanceMaintenanceGraphDirect"), std::string::npos)
            << "Steady maintenance replay should bypass generic decode replay bookkeeping once captured.";
        EXPECT_NE(dgo.find("segment.capture->launch()"), std::string::npos)
            << "The steady maintenance fast path should enqueue the captured graph executable directly.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_direct_replay\""), std::string::npos)
            << "Perfstats must expose when the maintenance graph uses the direct replay fast path.";
        EXPECT_NE(dgo.find("completion_event_in_flight"), std::string::npos)
            << "Maintenance replay must retain diagnostics ownership without host scheduling decisions.";
        {
            const size_t cadence_return =
                maintenance_scheduler.find("if (!should_launch)");
            const size_t retained_graph_contract =
                maintenance_scheduler.find("if (!active_cache.graph)");
            EXPECT_EQ(cadence_return, std::string::npos)
                << "The host must not decide whether a committed decode boundary is due.";
            ASSERT_NE(retained_graph_contract, std::string::npos);
            EXPECT_EQ(
                maintenance_scheduler.find(
                    "buildDeviceMoERebalanceMaintenanceGraph"),
                std::string::npos)
                << "The scheduled hot path must consume the graph retained by "
                   "eager family declaration, never discover topology lazily.";
            EXPECT_NE(
                dgo.find(
                    "materializeDeviceMoERebalanceMaintenanceGraphForFamily"),
                std::string::npos)
                << "Device maintenance must be a first-class pre-capture family participant.";
            const size_t forward_family_begin =
                dgo.find(
                    "bool DeviceGraphOrchestrator::materializeForwardGraphForShape");
            const size_t eager_maintenance =
                dgo.find(
                    "materializeDeviceMoERebalanceMaintenanceGraphForFamily()",
                    forward_family_begin);
            const size_t family_allocation =
                dgo.find(
                    "if (!ensureDeviceWorkspaceAllocated(",
                    eager_maintenance);
            ASSERT_NE(forward_family_begin, std::string::npos);
            ASSERT_NE(eager_maintenance, std::string::npos);
            ASSERT_NE(family_allocation, std::string::npos);
            EXPECT_LT(eager_maintenance, family_allocation)
                << "The production maintenance graph must join the manifest "
                   "before generation one is allocated.";
            EXPECT_NE(
                dgo.find(
                    "WorkspaceGraphParticipantRole::\n"
                    "                            MoERebalanceMaintenance",
                    eager_maintenance),
                std::string::npos)
                << "The manifest must preserve maintenance as an explicit typed role.";
            EXPECT_NE(
                dgo.find(
                    "WorkspaceGraphParticipantLifetime::\n"
                    "                            PersistentAcrossParticipants",
                    eager_maintenance),
                std::string::npos)
                << "Maintenance status/controller records outlive graph execution "
                   "and must remain disjoint from later prefill/decode workspace.";
            EXPECT_NE(dgo.find("\"scheduled_plan_launch\""), std::string::npos)
                << "Perfstats should identify scheduled device transactions.";
        }
        EXPECT_EQ(maintenance_scheduler.find("queryEventChecked"), std::string::npos)
            << "Steady maintenance must not poll device completion from the host.";
        EXPECT_EQ(maintenance_scheduler.find(
                      "exportCompletedDeviceMoERebalanceMaintenanceStats"),
                  std::string::npos)
            << "D2H status export belongs only to explicit request epilogues.";
        EXPECT_EQ(maintenance_scheduler.find("createTimingEvent"), std::string::npos)
            << "PerfStats must not allocate timing events per maintenance wave.";
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
        EXPECT_NE(dgo.find("\"device_maintenance_graph_boundary_enqueue\""), std::string::npos)
            << "Perfstats must time enqueue cost without describing the host as cadence authority.";
        EXPECT_NE(qwen_moe_graph.find("device_rebalance_maintenance_slack_tokens"), std::string::npos)
            << "Graph-declared device cadence must include its configured slack.";
        EXPECT_NE(debug_env.find("int device_rebalance_maintenance_slack_tokens = 1"),
                  std::string::npos)
            << "Device-side maintenance should default to one-token slack to avoid near-full histogram no-op replays.";
        EXPECT_NE(debug_env.find("int device_rebalance_min_maintenance_period_tokens = 512"),
                  std::string::npos)
            << "Device-side maintenance should default to the shared CUDA2/ROCm2 cadence floor.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_MIN_MAINTENANCE_PERIOD_TOKENS"),
                  std::string::npos)
            << "The maintenance cadence floor must stay operator-tunable.";
        EXPECT_NE(qwen_moe_graph.find("device_rebalance_min_maintenance_period_tokens"), std::string::npos)
            << "The graph config must publish the device-side maintenance cadence floor.";
        EXPECT_NE(debug_env.find("int device_rebalance_initial_maintenance_period_tokens = 321"),
                  std::string::npos)
            << "Device-side maintenance should default to the delayed first replay that stabilized ROCm without regressing clean CUDA.";
        EXPECT_NE(debug_env.find("LLAMINAR_MOE_DEVICE_REBALANCE_INITIAL_MAINTENANCE_PERIOD_TOKENS"),
                  std::string::npos)
            << "The early first maintenance period must stay operator-tunable.";
        EXPECT_NE(qwen_moe_graph.find("device_rebalance_initial_maintenance_period_tokens"), std::string::npos)
            << "The graph config must honor the optional early first maintenance period.";
        EXPECT_NE(qwen_moe_graph.find("initial_maintenance_period_tokens"), std::string::npos);
        EXPECT_NE(qwen_moe_graph.find("maintenance_period_tokens"), std::string::npos);
        EXPECT_EQ(dgo.find("\"requested_launch_period\""), std::string::npos)
            << "Hot-path host tags must not reconstruct the device-owned cadence.";
        EXPECT_EQ(dgo.find("\"launch_period\""), std::string::npos)
            << "Hot-path host tags must not advertise a second cadence authority.";
        EXPECT_EQ(dgo.find("\"device_maintenance_graph_event_query\""), std::string::npos)
            << "The host must not poll maintenance completion between device-side phases.";
        EXPECT_NE(dgo.find("\"device_maintenance_graph_replay_enqueue\""), std::string::npos);
        EXPECT_NE(dgo.find("\"device_maintenance_graph_completion_record\""), std::string::npos);
        EXPECT_EQ(dgo.find("\"device_maintenance_graph_gpu_elapsed\""), std::string::npos)
            << "Steady-state maintenance must not allocate per-wave GPU timing resources.";
        EXPECT_EQ(dgo.find("\"device_maintenance_graph_gpu_timing_record_failures\""),
                  std::string::npos)
            << "The retired per-wave timing-event path must not return.";
        EXPECT_EQ(dgo.find("\"device_maintenance_graph_gpu_timing_read_failures\""),
                  std::string::npos)
            << "The retired per-wave timing-event path must not return.";
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
        EXPECT_EQ(graph_builder.find("payload_edge_mask"), std::string::npos)
            << "The graph-builder contract must not expose a host-selected payload mask.";
        EXPECT_EQ(dgo.find("pending_payload_edge_mask"), std::string::npos)
            << "The maintenance scheduler must not read device planning state to choose another graph.";
        EXPECT_EQ(dgo.find("device_moe_rebalance_maintenance_payload_graphs_"),
                  std::string::npos)
            << "Atomic maintenance owns one captured graph, not an edge-mask cache.";
        EXPECT_EQ(device_rebalance_stage.find("groupedP2PRawOnStream"), std::string::npos)
            << "Atomic maintenance uses a fixed graph-captured NCCL/RCCL collective topology.";
        EXPECT_NE(device_rebalance_stage.find("allgatherRawOnStream"), std::string::npos)
            << "Payload bytes must move through the graph-captured collective lane.";
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
            << "Explicit epilogues must validate completed device status.";
        EXPECT_NE(dgo.find("outcome->useful_work"), std::string::npos)
            << "Completed maintenance exports must identify whether the replay produced useful work.";
        EXPECT_EQ(dgo.find("\"device_maintenance_graph_no_work_completions\""), std::string::npos)
            << "Steady scheduling must not D2H-read no-work outcomes.";
        EXPECT_EQ(dgo.find("\"device_maintenance_graph_skipped_no_work_backoff\""), std::string::npos)
            << "Host-managed no-work backoff is obsolete in the atomic device path.";
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

        const size_t maintenance_call =
            generate_body.find("if (!maybeApplyMoERebalance())");
        const size_t epilogue_device_side_gate =
            generate_body.rfind("if (usesDeviceSideMoERebalanceController())");
        const size_t epilogue_host_publish =
            generate_body.rfind("publishPendingMoERebalanceUpdate()");
        ASSERT_NE(maintenance_call, std::string::npos)
            << "Every completed serial or MTP transaction must enter the shared decode-boundary maintenance hook.";
        EXPECT_EQ(generate_body.find("!device_side_moe_rebalance && !maybeApplyMoERebalance()"),
                  std::string::npos)
            << "MTP must not skip captured device maintenance merely because the controller is device-owned.";
        ASSERT_NE(epilogue_device_side_gate, std::string::npos);
        ASSERT_NE(epilogue_host_publish, std::string::npos);
        EXPECT_LT(epilogue_device_side_gate, epilogue_host_publish)
            << "The decode epilogue must not drain host pending publishes in device-side mode.";

        const size_t chat_rebalance_helper =
            chat.find("bool runChatMoERebalanceMaintenance(");
        ASSERT_NE(chat_rebalance_helper, std::string::npos)
            << "Chat serving should funnel all MoE maintenance through one committed-boundary helper.";
        const size_t chat_rebalance_helper_end =
            chat.find("    }", chat_rebalance_helper);
        ASSERT_NE(chat_rebalance_helper_end, std::string::npos);
        const std::string chat_rebalance_helper_body =
            chat.substr(chat_rebalance_helper,
                        chat_rebalance_helper_end - chat_rebalance_helper);
        const size_t chat_helper_host_maintenance =
            chat_rebalance_helper_body.find("runner.maybeApplyMoERebalance()");
        ASSERT_NE(chat_helper_host_maintenance, std::string::npos);
        EXPECT_EQ(chat_rebalance_helper_body.find("device_side_moe_rebalance"),
                  std::string::npos)
            << "The shared boundary hook chooses captured-device versus host-prepared maintenance internally.";
        EXPECT_NE(chat.find("last_decode_window_had_moe_maintenance"), std::string::npos)
            << "Chat serving must not skip the final decode window's MoE maintenance.";
        EXPECT_NE(chat.find("runChatMoERebalanceMaintenance(runner_)"),
                  std::string::npos)
            << "Chat decode loops and epilogues must use the shared committed-boundary hook.";
        EXPECT_NE(runner.find("runner_->maybeApplyDecodeBoundaryMaintenance()"),
                  std::string::npos)
            << "Device-owned rebalance must schedule its captured maintenance graph instead of becoming a no-op.";
        EXPECT_NE(dgo.find("DeviceGraphOrchestrator::maybeApplyDecodeBoundaryMaintenance()"),
                  std::string::npos)
            << "A device runner must launch maintenance only after the verifier transaction closes.";
        EXPECT_NE(rank.find("RankOrchestrator::maybeApplyDecodeBoundaryMaintenance()"),
                  std::string::npos)
            << "LocalTP needs a rank-level maintenance fanout.";
        EXPECT_NE(rank.find("tp_worker_pool_->dispatch("), std::string::npos)
            << "LocalTP maintenance participants must enter NCCL/RCCL and host rendezvous concurrently.";
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
        EXPECT_NE(server_e2e.find("moe_rebalance_window_from_flags()"),
                  std::string::npos)
            << "Completion and maintenance probes must share one exact CLI window parser.";
        EXPECT_NE(server_e2e.find("mtp_draft_tokens_from_flags()"),
                  std::string::npos)
            << "MTP movement probes must derive their transaction capacity from the tested depth.";
        EXPECT_NE(
            server_e2e.find(
                "(rebalance_window - 1) * (mtp_draft_tokens + 1) + 1"),
            std::string::npos)
            << "The probe must request enough output to force one scheduler boundary per evidence-window token.";
        EXPECT_NE(
            server_e2e.find(
                "probe_rebalance_window=$(moe_rebalance_window_from_flags \"$extra_flags\")"),
            std::string::npos)
            << "The movement probe must derive readiness cadence from the tested server policy.";
        EXPECT_NE(
            server_e2e.find(
                "probe_effective_window + probe_maintenance_slack"),
            std::string::npos)
            << "The first diagnostic replay must use the production "
               "window-plus-slack boundary so grouped MTP has a complete "
               "histogram evidence window.";
        EXPECT_NE(
            server_e2e.find(
                "local probe_grouped_rows=$((probe_mtp_draft_tokens + 1))"),
            std::string::npos)
            << "The movement probe's device evidence window must be bounded by "
               "the actual grouped verifier row capacity.";
        EXPECT_NE(
            server_e2e.find(
                "LLAMINAR_MOE_DEVICE_REBALANCE_MAINTENANCE_SLACK_TOKENS=${probe_maintenance_slack}"),
            std::string::npos)
            << "The probe must configure the same named slack used to derive "
               "its initial maintenance period.";
        EXPECT_NE(
            server_e2e.find(
                "LLAMINAR_MOE_DEVICE_REBALANCE_INITIAL_MAINTENANCE_PERIOD_TOKENS=${probe_initial_maintenance_period}"),
            std::string::npos)
            << "The short HTTP movement probe must schedule maintenance before request cleanup resets request-local routing evidence.";
        EXPECT_NE(server_e2e.find("clear_cache_pending_publish_drains"), std::string::npos)
            << "The HTTP regression must assert the host request cleanup drain counter.";
        EXPECT_NE(server_e2e.find("device_maintenance_graph_request_reset_exports"), std::string::npos)
            << "The HTTP regression must accept the device-side request-reset export counter.";
        EXPECT_NE(
            server_e2e.find(
                "record_tags.get(\"reset\") in {\"clear_cache\", \"request-clear-cache\"}"),
            std::string::npos)
            << "The HTTP regression must accept both the IKVCache operation "
               "name and the request-boundary reset reason emitted by the "
               "device maintenance epilogue.";
        EXPECT_NE(server_e2e.find("mode != \"llep\""), std::string::npos)
            << "LLEP prefix-cache clear probes should not require dynamic publish drain/export counters.";
        EXPECT_NE(server_e2e.find("request-clear-cache"), std::string::npos)
            << "The HTTP regression must match the live-state mutation operation emitted by request-boundary cache clears.";
        EXPECT_NE(server_e2e.find("materialized_score = ("), std::string::npos)
            << "The movement gate must distinguish a policy proposal from a published transfer command.";
        const size_t planned_score = server_e2e.find("planned_score = (");
        ASSERT_NE(planned_score, std::string::npos);
        const size_t materialized_score =
            server_e2e.find("materialized_score = (", planned_score);
        ASSERT_NE(materialized_score, std::string::npos);
        const std::string planned_score_body =
            server_e2e.substr(planned_score, materialized_score - planned_score);
        EXPECT_NE(
            planned_score_body.find("device_rebalance_wave_planned_layer_count"),
            std::string::npos)
            << "Request-reset diagnostics must use the durable device wave's "
               "planned-layer publication after the transient planner status "
               "has advanced to WindowNotReady.";
        EXPECT_EQ(
            planned_score_body.find("device_rebalance_wave_command_count"),
            std::string::npos)
            << "A materialized command must not satisfy the planning gate.";
        EXPECT_EQ(
            planned_score_body.find("device_rebalance_wave_applied_arrivals"),
            std::string::npos)
            << "Applied movement must not substitute for independent planning evidence.";
        EXPECT_NE(
            server_e2e.find(
                "saw a policy proposal but no materialized transfer command/payload"),
            std::string::npos)
            << "The E2E probe must fail when LLEP proposes spans without publishing a transfer wave.";
        const size_t applied_score = server_e2e.find("applied_score = (");
        ASSERT_NE(applied_score, std::string::npos);
        const size_t applied_gate =
            server_e2e.find("if applied_score <= 0.0:", applied_score);
        ASSERT_NE(applied_gate, std::string::npos);
        const std::string applied_score_body =
            server_e2e.substr(applied_score, applied_gate - applied_score);
        EXPECT_EQ(
            applied_score_body.find("device_rebalance_llep_weight_transfer_count"),
            std::string::npos)
            << "An ideal LLEP transfer is planning evidence, not proof that the destination imported it.";
        EXPECT_EQ(
            applied_score_body.find("\"llep_weight_transfer_count\""),
            std::string::npos)
            << "LLEP planning tags must not satisfy the applied-movement gate.";
        EXPECT_EQ(
            applied_score_body.find("device_rebalance_transfer_useful_payload_bytes"),
            std::string::npos)
            << "Packing bytes is materialization evidence, not destination-side apply evidence.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ForwardConsumersQueueMoEMaintenanceEventsWithoutHostSynchronization)
    {
        const fs::path root = findRepoRoot();
        const fs::path dgo_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        const fs::path dgo_header_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h";
        const fs::path server_e2e_path =
            root / "tests/v2/e2e/server/test_server_e2e.sh";
        ASSERT_TRUE(fs::exists(dgo_path)) << dgo_path;
        ASSERT_TRUE(fs::exists(dgo_header_path)) << dgo_header_path;
        ASSERT_TRUE(fs::exists(server_e2e_path)) << server_e2e_path;

        const std::string dgo = readFile(dgo_path);
        const std::string dgo_header = readFile(dgo_header_path);
        const std::string server_e2e = readFile(server_e2e_path);
        ASSERT_FALSE(dgo.empty()) << dgo_path;
        ASSERT_FALSE(dgo_header.empty()) << dgo_header_path;
        ASSERT_FALSE(server_e2e.empty()) << server_e2e_path;

        EXPECT_NE(dgo_header.find("waitForPendingDeviceMoERebalanceMaintenance"),
                  std::string::npos)
            << "DeviceGraphOrchestrator must expose the device-side MoE maintenance handoff.";

        const size_t wait_start =
            dgo.find("bool DeviceGraphOrchestrator::waitForPendingDeviceMoERebalanceMaintenance(");
        ASSERT_NE(wait_start, std::string::npos);
        const size_t wait_end =
            dgo.find("bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph(",
                     wait_start);
        ASSERT_NE(wait_end, std::string::npos);
        const std::string wait_body = dgo.substr(wait_start, wait_end - wait_start);
        EXPECT_NE(wait_body.find("DeviceTimelinePoint::MoERebalanceMaintenanceReady"),
                  std::string::npos)
            << "Every consumer wait must be validated by the centralized timeline manifest.";
        EXPECT_NE(wait_body.find(".to(consumer_role)"),
                  std::string::npos)
            << "The handoff API must require a typed consumer role.";
        EXPECT_NE(wait_body.find(".enqueuePublishedWait("),
                  std::string::npos)
            << "Every consumer must queue a durable device event dependency on its exact execution stream.";
        EXPECT_NE(wait_body.find("device_maintenance_graph_consumer_event_waits"),
                  std::string::npos)
            << "PerfStats must prove that production graph consumers exercised the handoff.";
        EXPECT_NE(server_e2e.find("device_maintenance_graph_consumer_event_waits"),
                  std::string::npos)
            << "The canonical GPU movement gate must reject maintenance launches that no production graph consumes.";
        EXPECT_EQ(wait_body.find("synchronizeChecked()"), std::string::npos)
            << "Steady-state maintenance publication must not synchronize the full device.";
        EXPECT_EQ(wait_body.find("synchronizeStreamChecked("), std::string::npos)
            << "Steady-state maintenance publication must not synchronize a stream through the host.";
        EXPECT_EQ(wait_body.find("ensureOnHost("), std::string::npos)
            << "Maintenance publication ordering must not create a host mirror.";

        const size_t live_prepare_start =
            dgo.find("bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(");
        ASSERT_NE(live_prepare_start, std::string::npos);
        const size_t live_prepare_end =
            dgo.find("const float *DeviceGraphOrchestrator::getAllPositionLogits() const",
                     live_prepare_start);
        ASSERT_NE(live_prepare_end, std::string::npos);
        const std::string live_prepare =
            dgo.substr(live_prepare_start, live_prepare_end - live_prepare_start);
        const size_t maintenance_wait =
            live_prepare.find("waitForPendingDeviceMoERebalanceMaintenance(");
        const size_t optional_state_gate =
            live_prepare.find("if (!has_accepted_publication");
        ASSERT_NE(maintenance_wait, std::string::npos);
        ASSERT_NE(optional_state_gate, std::string::npos);
        EXPECT_LT(maintenance_wait, optional_state_gate)
            << "Main, prefill, and verifier graphs must consume MoE maintenance even when no MTP/prefix publication is pending.";

        const size_t sidecar_start =
            dgo.find("bool DeviceGraphOrchestrator::executeMTPDepth0Batched(");
        ASSERT_NE(sidecar_start, std::string::npos);
        const size_t sidecar_end =
            dgo.find("bool DeviceGraphOrchestrator::", sidecar_start + 64);
        ASSERT_NE(sidecar_end, std::string::npos);
        const std::string sidecar =
            dgo.substr(sidecar_start, sidecar_end - sidecar_start);
        EXPECT_NE(sidecar.find(
                      "waitForPendingDeviceMoERebalanceMaintenance(\n"
                      "                sidecar_dynamic_stream,\n"
                      "                DeviceTimelineRole::MTPSidecarGraph,\n"
                      "                \"mtp_sidecar_graph\")"),
                  std::string::npos)
            << "The MTP sidecar bypasses ForwardExecutionEngine and must queue the same event handoff explicitly.";
        EXPECT_NE(live_prepare.find(
                      "waitForPendingDeviceMoERebalanceMaintenance(\n"
                      "                execution_stream,\n"
                      "                DeviceTimelineRole::MainForwardGraph,\n"
                      "                \"forward_graph\")"),
                  std::string::npos)
            << "The main graph must identify itself as an independent typed consumer.";

        EXPECT_EQ(dgo.find("synchronizeGraphStableMoERuntimeBeforeDecodeCapture"),
                  std::string::npos)
            << "The obsolete capture-only full-device fence must remain retired.";
        EXPECT_EQ(dgo.find("decode_capture_boundary_moe_runtime_device_sync"),
                  std::string::npos)
            << "The obsolete host synchronization counter must remain retired.";
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
        const size_t fused_publish = execute_body.find("execution.publish(params_.shared_output)",
                                                       fused_gate_call);
        const size_t fused_combined_publish = execute_body.find("execution.publish(params_.combined_output)",
                                                               fused_gate_call);
        const size_t gate_call = execute_body.find("kernel->sharedExpertGateFromTensors(");
        const size_t publish = execute_body.find("gpuExecution().publish(params_.shared_output)",
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
        const size_t publish = execute_body.find("gpuExecution().publish(params_.output)",
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
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h";
        const fs::path runtime_table_path =
            root / "src/v2/execution/moe/MoERuntimeTable.cpp";
        const fs::path directory_path =
            root / "src/v2/execution/moe/DeviceMoETransferSlotDirectory.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;
        ASSERT_TRUE(fs::exists(runtime_table_path)) << runtime_table_path;
        ASSERT_TRUE(fs::exists(directory_path)) << directory_path;

        const std::string contents = readFile(graph_path);
        const std::string orchestrator = readFile(orchestrator_path);
        const std::string runtime_table = readFile(runtime_table_path);
        const std::string directory = readFile(directory_path);
        ASSERT_FALSE(contents.empty()) << graph_path;
        ASSERT_FALSE(orchestrator.empty()) << orchestrator_path;
        ASSERT_FALSE(runtime_table.empty()) << runtime_table_path;
        ASSERT_FALSE(directory.empty()) << directory_path;

        const size_t reset_start =
            contents.find("void Qwen35MoEGraph::resetState(void *execution_stream)");
        ASSERT_NE(reset_start, std::string::npos);
        const size_t reset_end = contents.find("void Qwen35MoEGraph::resetPrefixCacheRuntimeStateWithoutSnapshot",
                                               reset_start);
        ASSERT_NE(reset_end, std::string::npos);
        const std::string reset_body = contents.substr(reset_start, reset_end - reset_start);

        EXPECT_NE(reset_body.find("restoreInitialRuntimeState(execution_stream)"), std::string::npos)
            << "Request-boundary reset must restore canonical MoE placement so "
               "portable prefix blocks replay suffix prefill under the same "
               "logical expert ownership used by an uncached full prefill.";
        EXPECT_EQ(reset_body.find("restoreInitialRuntimeState()"), std::string::npos)
            << "GPU model reset must never select an implicit stream.";
        EXPECT_NE(reset_body.find("resetRequestPublications(execution_stream)"), std::string::npos)
            << "Runtime placement and transfer-directory occupancy are one "
               "stream-ordered reset transaction.";
        EXPECT_EQ(reset_body.find("resetDecodeRuntimeState("), std::string::npos)
            << "Clearing runtime placement banks during session reset makes "
               "the next GPU decode route fall back to host/top-k state or fail "
               "before graph-captured MoE decode can run.";
        EXPECT_EQ(reset_body.find("moe_graph_rebalance_bindings_.clear()"), std::string::npos)
            << "Graph-stable device rebalance bindings must survive ordinary request reset because "
               "the captured decode graph may survive and the maintenance graph still needs the same "
               "runtime-table and transfer-slot pointers.";

        const size_t kernel_reset =
            orchestrator.find("ResetKernelDynamicState);");
        const size_t model_reset =
            orchestrator.find("ResetModelRuntime);", kernel_reset);
        const size_t publish_reset =
            orchestrator.find("PublishResetReady);", model_reset);
        ASSERT_NE(kernel_reset, std::string::npos);
        ASSERT_NE(model_reset, std::string::npos);
        ASSERT_NE(publish_reset, std::string::npos);
        EXPECT_LT(kernel_reset, model_reset);
        EXPECT_LT(model_reset, publish_reset)
            << "Reset-ready must be the sole final publication after every "
               "kernel-owned and model-owned device mutation.";
        EXPECT_NE(orchestrator.find(
                      "graph_builder_->resetState(\n"
                      "                        reset_transaction.executionStream())"),
                  std::string::npos)
            << "The graph reset must consume the transaction's exact producer stream.";

        for (const auto &[owner, source] :
             std::array<std::pair<const char *, const std::string *>, 2>{
                 std::pair{"runtime table", &runtime_table},
                 std::pair{"transfer directory", &directory}})
        {
            EXPECT_NE(source->find("requires an explicit stream"), std::string::npos)
                << owner << " must reject null-stream request reset.";
            EXPECT_NE(source->find("deviceCopyAsync("), std::string::npos)
                << owner << " must enqueue immutable device-baseline restoration "
                            "on the request-reset transaction's exact stream.";
        }

        const size_t runtime_empty_reset =
            runtime_table.find("void DeviceMoERuntimeTable::resetDecodeRuntimeState(");
        const size_t runtime_initial_reset =
            runtime_table.find("void DeviceMoERuntimeTable::restoreInitialRuntimeState(",
                               runtime_empty_reset);
        const size_t runtime_reset_end =
            runtime_table.find("void DeviceMoERuntimeTable::syncRuntimeStateToHost(",
                               runtime_initial_reset);
        ASSERT_NE(runtime_empty_reset, std::string::npos);
        ASSERT_NE(runtime_initial_reset, std::string::npos);
        ASSERT_NE(runtime_reset_end, std::string::npos);
        const std::string runtime_reset_bodies =
            runtime_table.substr(runtime_empty_reset,
                                 runtime_reset_end - runtime_empty_reset);
        EXPECT_EQ(runtime_reset_bodies.find("createMirrorStream"), std::string::npos);
        EXPECT_EQ(runtime_reset_bodies.find("synchronizeMirror"), std::string::npos);
        EXPECT_EQ(runtime_reset_bodies.find("deviceToDevice("), std::string::npos)
            << "GPU request reset must not block the host on a legacy synchronous D2D copy.";
        EXPECT_NE(runtime_reset_bodies.find("copyMirrorToMirrorAsync("), std::string::npos)
            << "Both canonical runtime-bank restores must use the explicit-stream "
               "asynchronous D2D helper.";
        EXPECT_EQ(runtime_reset_bodies.find("host_layers_ ="), std::string::npos)
            << "GPU reset must not pretend a setup-time host template is a live coherence peer.";
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
        EXPECT_NE(reset_body.find("device_moe_rebalance_maintenance_graph_"), std::string::npos);
        EXPECT_EQ(reset_body.find("device_moe_rebalance_maintenance_payload_graphs_"), std::string::npos)
            << "Kernel-state invalidation must not retain the retired payload graph cache.";

        const std::regex invalidate_hook_regex("void invalidateKernelDynamicState\\(\\) override");
        const auto invalidate_begin =
            std::sregex_iterator(moe_header.begin(), moe_header.end(), invalidate_hook_regex);
        const auto invalidate_end = std::sregex_iterator();
        EXPECT_EQ(std::distance(invalidate_begin, invalidate_end), 3)
            << "Routed, shared-FFN, and shared-gate stages own independent backend state.";
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

        EXPECT_EQ(
            populate_body.find("TransferEngine::instance().upload"),
            std::string::npos)
            << "Prefix population owns explicit archive streams and must not "
               "re-enter host-mirror TransferEngine upload paths.";
        EXPECT_EQ(
            restore_body.find("TransferEngine::instance().upload"),
            std::string::npos)
            << "Terminal restore must consume explicit lower-tier ingress or "
               "device-hot D2D sources, never a TensorBase host mirror.";
        EXPECT_NE(
            populate_body.find("hostToDeviceOnStream("),
            std::string::npos)
            << "Cold RAM/disk ingress must use asynchronous H2D on the "
               "explicit prefix stream.";
        EXPECT_NE(
            populate_body.find("deviceCopyAsync("),
            std::string::npos)
            << "Device-hot KV/hidden restore must remain asynchronous direct D2D.";
        EXPECT_NE(
            restore_body.find("hostToDeviceOnStream("),
            std::string::npos)
            << "A terminal block larger than the configured hot tier still "
               "requires explicit-stream cold ingress.";
        EXPECT_NE(
            restore_body.find("deviceCopyAsync("),
            std::string::npos)
            << "Promoted terminal logits and hidden state must restore with asynchronous D2D.";
        EXPECT_NE(
            populate_body.find("TransferEngine::publishDeviceWrite("),
            std::string::npos)
            << "Restored terminal hidden ownership must publish stream order.";
        EXPECT_NE(
            restore_body.find("TransferEngine::publishDeviceWrite("),
            std::string::npos)
            << "Restored terminal tensors must publish stream order.";
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

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalanceTransferSlotsAreCapacityTypedAcrossModelFormats)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        auto require_capacity_typed_unpack =
            [](const std::string &contents, const char *backend)
        {
            EXPECT_NE(contents.find("rebalance_directory_fits_transfer_capacity(src, dst)"),
                      std::string::npos)
                << backend << " transfer unpack must admit any source format that fits the persistent allocation";
            EXPECT_NE(contents.find("rebalance_retarget_transfer_directory(dst, src)"),
                      std::string::npos)
                << backend << " transfer unpack must publish the arriving codebook before copying its format-dependent arrays";
            EXPECT_NE(contents.find("allocation_payload_bytes_per_block"),
                      std::string::npos)
                << backend << " descriptors must keep allocation capacity independent from active codebook metadata";
            EXPECT_EQ(contents.find("rebalance_directory_compatible("),
                      std::string::npos)
                << backend << " must not restore exact-codebook matching for model-wide transfer slots";
        };

        require_capacity_typed_unpack(readFile(cuda_path), "CUDA");
        require_capacity_typed_unpack(readFile(rocm_path), "ROCm");

        const std::string graph = readFile(graph_path);
        EXPECT_NE(graph.find("for (int scan_layer = 0;"),
                  std::string::npos)
            << "Decode-maintenance allocation capacity must be derived from every model layer";
        EXPECT_NE(graph.find("scan_layer < runtime_table_layers"),
                  std::string::npos)
            << "The model format preflight must cover the full runtime-table layer domain";
        EXPECT_NE(graph.find("DeviceMoETransferSlotDirectory::profileForLayerFormats"),
                  std::string::npos)
            << "Qwen MoE graph construction must merge exact layer formats into one explicit profile";
        EXPECT_NE(graph.find("collectGraphRebalanceTransferProfile"),
                  std::string::npos)
            << "Prefill LLEP and decode maintenance must share the same model-wide format preflight";
        EXPECT_NE(graph.find("graphRebalanceTransferDirectoryKey"),
                  std::string::npos)
            << "Prefill and maintenance must derive persistent transfer storage "
               "identity through one device-domain key builder";
        EXPECT_NE(graph.find("existing->second->requirePhysicalOwner("),
                  std::string::npos)
            << "Every cached transfer directory must prove exact device, ordinal, "
               "and participant ownership before graph binding";
        EXPECT_NE(
            graph.find(
                "const std::string transfer_state_key =\n"
                "                binding_key + \":workspace=\" + rebalance_workspace"),
            std::string::npos)
            << "Each layer must retain its own event state even though storage is domain-shared";
        EXPECT_NE(graph.find("transfer_directory->wirePayloadBytes()"),
                  std::string::npos)
            << "Collective wire sizing must use the largest realizable layer payload, not allocation maxima";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalancePublicationErrorsAreFatalAtRequestBoundary)
    {
        const fs::path root = findRepoRoot();
        const fs::path header_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h";
        const fs::path implementation_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(header_path)) << header_path;
        ASSERT_TRUE(fs::exists(implementation_path)) << implementation_path;

        const std::string header = readFile(header_path);
        const std::string implementation = readFile(implementation_path);
        EXPECT_NE(header.find("bool hasFatalError() const noexcept"),
                  std::string::npos)
            << "The device publication outcome must classify structural transfer errors centrally";
        EXPECT_NE(header.find("copy_missing_source_descriptors != 0u"),
                  std::string::npos);
        EXPECT_NE(header.find("copy_descriptor_mismatches != 0u"),
                  std::string::npos);
        EXPECT_NE(header.find("controller_last_error_code != 0u"),
                  std::string::npos);
        EXPECT_NE(header.find("plan_overflow != 0u"),
                  std::string::npos);
        EXPECT_NE(header.find("payload_bucket_overflow != 0u"),
                  std::string::npos);
        EXPECT_NE(header.find("copy_ready_local_arrivals !="),
                  std::string::npos);
        EXPECT_NE(implementation.find("if (outcome.hasFatalError())"),
                  std::string::npos)
            << "Request epilogues must stop inference rather than merely exporting error counters";
        EXPECT_NE(implementation.find(
                      "Device MoE rebalance maintenance violated a fatal publication invariant"),
                  std::string::npos)
            << "Fatal errors should report the exact device-owned publication contract";
    }

    /**
     * @brief Prevent device controller errors from being laundered by replay.
     *
     * CUDA and ROCm maintenance graphs repeatedly reuse two persistent command
     * waves. A failed transfer must poison the whole request-owned controller:
     * neither a later empty planner result nor a clean-looking publication may
     * return that wave to service. Keep the terminal transition and first-error
     * provenance symmetric across both backend kernels.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalanceControllerPoisonIsTerminalAndDiagnosable)
    {
        const fs::path root = findRepoRoot();
        const fs::path controller_path =
            root / "src/v2/execution/moe/DeviceMoERebalanceController.h";
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(controller_path)) << controller_path;
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;

        const std::string controller = readFile(controller_path);
        const std::string orchestrator = readFile(orchestrator_path);
        EXPECT_NE(controller.find("last_error_wave_index"),
                  std::string::npos);
        EXPECT_NE(controller.find("last_error_expected_arrivals"),
                  std::string::npos);
        EXPECT_NE(controller.find("last_error_copied_arrivals"),
                  std::string::npos);
        EXPECT_NE(controller.find("error_participant"),
                  std::string::npos);

        auto require_terminal_poison =
            [](const std::string &kernel, const char *backend)
        {
            EXPECT_NE(kernel.find("poison_rebalance_graph_controller("),
                      std::string::npos)
                << backend << " must publish first-error provenance on device";
            EXPECT_NE(kernel.find("state && state->last_error_code != 0u"),
                      std::string::npos)
                << backend << " planners must reject a poisoned controller";
            EXPECT_NE(kernel.find(
                          "wave.state == kDeviceMoERebalanceLifecycleError ||"),
                      std::string::npos)
                << backend << " empty-wave reset must preserve Error";
            EXPECT_NE(kernel.find(
                          "state->last_error_code != 0u ||\n            !command_headers"),
                      std::string::npos)
                << backend << " apply lookup must stop after any fatal error";
            EXPECT_NE(kernel.find("if (state->last_error_code != 0u)\n            return;"),
                      std::string::npos)
                << backend << " publication replay must stop after poison";
        };

        require_terminal_poison(readFile(cuda_path), "CUDA");
        require_terminal_poison(readFile(rocm_path), "ROCm");
        EXPECT_NE(orchestrator.find("controller_error_expected_arrivals="),
                  std::string::npos);
        EXPECT_NE(orchestrator.find("controller_error_copied_arrivals="),
                  std::string::npos);
        EXPECT_NE(orchestrator.find("controller_error_participant="),
                  std::string::npos);
    }

    /**
     * @brief Keep persistent graph lifetime separate from request state lifetime.
     *
     * Captured maintenance replay must preserve a live transaction, but a new
     * request must unconditionally replace its controller, command headers,
     * layer-window cursors, plan counts, poison, and counters on the ordered
     * reset stream. This regression prevents a future cleanup from resetting
     * only one coupled record and leaking Dynamic-MoE state across requests.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         RebalanceRequestResetIsExplicitStreamOrderedAndBackendSymmetric)
    {
        const fs::path root = findRepoRoot();
        const fs::path interface_path =
            root / "src/v2/kernels/IMoEKernel.h";
        const fs::path stage_header_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.h";
        const fs::path stage_source_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.cpp";
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h";
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";

        const std::string interface_source = readFile(interface_path);
        const std::string stage_header = readFile(stage_header_path);
        const std::string stage_source = readFile(stage_source_path);
        const std::string orchestrator = readFile(orchestrator_path);
        ASSERT_FALSE(interface_source.empty()) << interface_path;
        ASSERT_FALSE(stage_header.empty()) << stage_header_path;
        ASSERT_FALSE(stage_source.empty()) << stage_source_path;
        ASSERT_FALSE(orchestrator.empty()) << orchestrator_path;

        EXPECT_NE(
            interface_source.find(
                "resetDeviceRebalanceGraphTransactionForRequest("),
            std::string::npos)
            << "Request teardown needs a distinct unconditional kernel API";
        EXPECT_NE(
            stage_header.find("ownsRequestTransactionState() const noexcept"),
            std::string::npos)
            << "Phase-split graphs must declare one transaction owner";
        EXPECT_NE(
            stage_source.find(
                "resetRequestTransactionStateOnStream(void *stream)"),
            std::string::npos);
        EXPECT_NE(
            stage_source.find(
                "resetDeviceRebalanceGraphTransactionForRequest("),
            std::string::npos);
        EXPECT_NE(
            stage_source.find("commandHeaderBufferName()"),
            std::string::npos);
        EXPECT_NE(
            stage_source.find("waveStateBufferName()"),
            std::string::npos);
        EXPECT_NE(
            stage_source.find("transferPlanCountBufferName()"),
            std::string::npos)
            << "The typed reset must bind every transaction-root record";
        EXPECT_NE(
            stage_source.find("if (!stream)"),
            std::string::npos)
            << "A default-stream request reset would destroy ordering";

        const size_t reset_transaction =
            orchestrator.find("void resetInferenceState(");
        ASSERT_NE(reset_transaction, std::string::npos);
        const size_t reset_maintenance_state =
            orchestrator.find(
                "ResetMaintenanceRequestState",
                reset_transaction);
        const size_t reset_model_runtime =
            orchestrator.find("ResetModelRuntime", reset_maintenance_state);
        const size_t publish_reset_ready =
            orchestrator.find("PublishResetReady", reset_model_runtime);
        ASSERT_NE(reset_maintenance_state, std::string::npos);
        ASSERT_NE(reset_model_runtime, std::string::npos);
        ASSERT_NE(publish_reset_ready, std::string::npos);
        EXPECT_LT(reset_maintenance_state, reset_model_runtime);
        EXPECT_LT(reset_model_runtime, publish_reset_ready)
            << "The reset-ready event must transitively publish controller and model-runtime resets";
        EXPECT_NE(
            orchestrator.find(
                "transaction_owners.size() != 1u",
                reset_transaction),
            std::string::npos)
            << "Missing or ambiguous transaction ownership must fail closed";

        for (const auto &[backend, kernel] :
             std::array<std::pair<const char *, std::string>, 2>{
                 std::pair{"CUDA", readFile(cuda_path)},
                 std::pair{"ROCm", readFile(rocm_path)}})
        {
            const size_t reset_kernel =
                kernel.find(
                    "reset_rebalance_graph_transaction_for_request_kernel");
            ASSERT_NE(reset_kernel, std::string::npos)
                << backend << " is missing the request-lifetime reset kernel";
            const std::string body = kernel.substr(reset_kernel, 1800);
            EXPECT_NE(
                body.find(
                    "init_rebalance_graph_controller_state_device(state, config);"),
                std::string::npos)
                << backend << " request reset must be unconditional";
            EXPECT_EQ(
                body.find(
                    "if (!rebalance_graph_controller_state_ok(state, config))"),
                std::string::npos)
                << backend << " must not preserve valid state at a request boundary";
            EXPECT_NE(
                body.find(
                    "command_headers[wave] ="),
                std::string::npos)
                << backend << " must clear command headers at the same boundary";
            EXPECT_NE(
                body.find(
                    "wave_states[wave] ="),
                std::string::npos)
                << backend << " must clear layer-window cursors at the same boundary";
            EXPECT_NE(
                body.find("plan_counts[wave] = 0u;"),
                std::string::npos)
                << backend << " must clear stale plan cardinality";
        }
    }

    /**
     * @brief Lock in fail-fast semantics at the device publication boundary.
     *
     * Request-epilogue validation is a final safety net, not permission to run
     * more decode graphs after a malformed maintenance result. The producer
     * completion event must also exist before graph submission: after a graph
     * mutates the inactive ownership bank, failure to record its publication
     * event leaves no coherent state from which inference could continue.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalanceOrderingFailuresAreImmediatelyFatal)
    {
        const fs::path root = findRepoRoot();
        const fs::path implementation_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(implementation_path)) << implementation_path;

        const std::string implementation = readFile(implementation_path);
        ASSERT_FALSE(implementation.empty()) << implementation_path;

        const size_t wait_start =
            implementation.find(
                "bool DeviceGraphOrchestrator::waitForPendingDeviceMoERebalanceMaintenance(");
        const size_t materialization_name =
            implementation.find(
                "materializeDeviceMoERebalanceMaintenanceGraphForFamily()",
                wait_start);
        ASSERT_NE(wait_start, std::string::npos);
        ASSERT_NE(materialization_name, std::string::npos);
        const size_t materialization_start =
            implementation.rfind(
                "bool DeviceGraphOrchestrator::",
                materialization_name);
        ASSERT_NE(materialization_start, std::string::npos);
        const size_t maintenance_start =
            implementation.find(
                "bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph(",
                materialization_name);
        ASSERT_NE(maintenance_start, std::string::npos);
        const std::string wait_body =
            implementation.substr(
                wait_start,
                materialization_start - wait_start);
        EXPECT_NE(wait_body.find("throw std::runtime_error("),
                  std::string::npos)
            << "Missing producer events, streams, backends, and rejected event waits must throw at the ownership boundary";
        EXPECT_EQ(wait_body.find("return false;"), std::string::npos)
            << "The device publication wait must not expose an ignorable soft-failure result";

        const size_t maintenance_end =
            implementation.find(
                "// =====================================================================\n"
                "    // IForwardExecutionHost interface implementations",
                maintenance_start);
        ASSERT_NE(maintenance_end, std::string::npos);
        const std::string maintenance_body =
            implementation.substr(
                maintenance_start,
                maintenance_end - maintenance_start);
        EXPECT_EQ(
            maintenance_body.find(
                "exportCompletedDeviceMoERebalanceMaintenanceStats"),
            std::string::npos)
            << "Steady graph scheduling must not D2H-read status before the next decode.";
        const size_t epilogue_start =
            implementation.find(
                "void DeviceGraphOrchestrator::drainCompletedDeviceMoERebalanceMaintenanceDiagnostics(");
        ASSERT_NE(epilogue_start, std::string::npos);
        ASSERT_LT(epilogue_start, wait_start);
        const std::string epilogue_body =
            implementation.substr(epilogue_start, wait_start - epilogue_start);
        EXPECT_NE(epilogue_body.find("outcome.hasFatalError()"),
                  std::string::npos)
            << "The explicit diagnostic epilogue must validate the completed atomic transaction.";
        EXPECT_NE(epilogue_body.find(
                      "Device MoE rebalance maintenance violated a fatal publication invariant"),
                  std::string::npos);
        EXPECT_NE(epilogue_body.find("throw std::runtime_error(error.str())"),
                  std::string::npos)
            << "Malformed device publication must stop inference at the epilogue.";

        const size_t completion_preflight =
            maintenance_body.find("if (!active_cache.completion_event)");
        const size_t graph_launch =
            maintenance_body.find(
                "tryLaunchCapturedMoERebalanceMaintenanceGraphDirect(");
        ASSERT_NE(completion_preflight, std::string::npos);
        ASSERT_NE(graph_launch, std::string::npos);
        EXPECT_LT(completion_preflight, graph_launch)
            << "Completion-event allocation must happen before capture or replay submits any maintenance work";

        const size_t completion_record =
            maintenance_body.find(
                "gpu_ctx->recordEventChecked(active_cache.completion_event.get(), maintenance_stream)");
        ASSERT_NE(completion_record, std::string::npos);
        const std::string record_failure_contract =
            maintenance_body.substr(completion_record, 1800);
        EXPECT_NE(record_failure_contract.find("std::terminate();"),
                  std::string::npos)
            << "An already-submitted graph whose publication event cannot be recorded has no recoverable ownership state";
    }

    /**
     * @brief Prevent failed native graph capture from entering eager recovery.
     *
     * HIP/CUDA stream capture is a transaction. Re-executing selected stages
     * after a recording failure can omit recorded-only work, duplicate
     * collectives, and synchronize a stream that is still capturing. The
     * shared scoped owner must close every successful begin, and failure to
     * leave native capture mode must terminate the process.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GraphCaptureFailureCannotEnterEagerRecovery)
    {
        const fs::path root = findRepoRoot();
        const fs::path guard_path =
            root / "src/v2/execution/local_execution/graph/GraphCaptureGuard.h";
        const fs::path controller_path =
            root / "src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp";
        const fs::path executor_path =
            root / "src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp";
        ASSERT_TRUE(fs::exists(guard_path)) << guard_path;
        ASSERT_TRUE(fs::exists(controller_path)) << controller_path;
        ASSERT_TRUE(fs::exists(executor_path)) << executor_path;

        const std::string guard = readFile(guard_path);
        const std::string controller = readFile(controller_path);
        const std::string executor = readFile(executor_path);

        EXPECT_NE(guard.find("class ScopedBackendGraphCapture final"),
                  std::string::npos)
            << "All graph-capture call sites need one shared begin/end lifetime owner";
        EXPECT_NE(guard.find("if (!capture_.endCapture())"),
                  std::string::npos);
        EXPECT_NE(guard.find("std::terminate();"),
                  std::string::npos)
            << "An endCapture failure leaves the native stream state unknowable and cannot be recovered";

        EXPECT_NE(controller.find(
                      "ScopedBackendGraphCapture capture_transaction("),
                  std::string::npos)
            << "Segment capture and recapture must use the structural lifetime owner";
        EXPECT_EQ(controller.find("fallback_to_fast_decode"),
                  std::string::npos);
        EXPECT_EQ(controller.find("capture_abandoned"),
                  std::string::npos);
        EXPECT_EQ(controller.find("continuing without capture"),
                  std::string::npos);
        EXPECT_EQ(controller.find("Recovery execution"),
                  std::string::npos)
            << "A failed capture must stop inference rather than selectively replaying stages";

        const size_t single_capture_start =
            executor.find(
                "bool DeviceGraphExecutor::executeWithGraphCapture(");
        const size_t decode_policy_start =
            executor.find(
                "bool DeviceGraphExecutor::executeDecodeWithCapturePolicy(",
                single_capture_start);
        ASSERT_NE(single_capture_start, std::string::npos);
        ASSERT_NE(decode_policy_start, std::string::npos);
        const std::string single_capture =
            executor.substr(
                single_capture_start,
                decode_policy_start - single_capture_start);
        const size_t finish =
            single_capture.find("capture_transaction.finish();");
        const size_t execution_failure =
            single_capture.find("if (!exec_success)", finish);
        ASSERT_NE(finish, std::string::npos);
        ASSERT_NE(execution_failure, std::string::npos);
        EXPECT_LT(finish, execution_failure)
            << "The native stream must leave capture mode before stage failure propagates";
        EXPECT_EQ(single_capture.find(
                      "if (!exec_success || !capture->endCapture())"),
                  std::string::npos)
            << "Short-circuit evaluation must never skip endCapture after a stage failure";
        EXPECT_NE(single_capture.find("if (capture->nodeCount() == 0)"),
                  std::string::npos);
        EXPECT_NE(single_capture.find(
                      "zero nodes; refusing capture-time eager execution"),
                  std::string::npos)
            << "A selected graph path must reject zero-node capture instead of accepting capture-time eager work";
        EXPECT_EQ(single_capture.find("Skipping graph replay"),
                  std::string::npos)
            << "Zero-node capture must never be treated as successful execution";

        EXPECT_NE(
            controller.find(
                "unit produced zero native nodes; refusing manual execution"),
            std::string::npos)
            << "Cached graph capture must reject zero native nodes instead of "
               "changing the declared capturable unit into manual execution.";
        EXPECT_EQ(
            controller.find("captured 0 nodes (CPU-only), will execute manually"),
            std::string::npos);
        EXPECT_NE(
            controller.find("full_graph_capture_executable_nodes"),
            std::string::npos)
            << "The E2E gate requires publication-time evidence from the "
               "instantiated executable, not only a pre-capture graph plan.";
    }

    /**
     * @brief Require CUDA MoE integration captures to use production ownership.
     *
     * The kernel integration suite calls tensor-aware facades directly. A raw
     * cudaStreamBeginCapture/cudaStreamEndCapture pair does not establish the
     * thread-local graph lifecycle consumed by TensorBase, so an output facade
     * can incorrectly try to record an externally visible completion event
     * inside the graph. The shared test fixture must compose the same
     * CUDAGraphCapture and ScopedBackendGraphCapture owners as production.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         CUDAMoEIntegrationCaptureUsesProductionTransaction)
    {
        const fs::path root = findRepoRoot();
        const fs::path test_path =
            root / "tests/v2/integration/kernels/cuda/Test__CUDAMoEKernel.cpp";
        ASSERT_TRUE(fs::exists(test_path)) << test_path;

        const std::string test_source = readFile(test_path);
        ASSERT_FALSE(test_source.empty());
        EXPECT_NE(test_source.find("class ScopedCudaTestGraph final"),
                  std::string::npos);
        EXPECT_NE(test_source.find("llaminar2::CUDAGraphCapture graph_"),
                  std::string::npos);
        EXPECT_NE(test_source.find(
                      "llaminar2::ScopedBackendGraphCapture transaction_"),
                  std::string::npos);
        EXPECT_EQ(test_source.find("cudaStreamBeginCapture("),
                  std::string::npos)
            << "Tensor-aware CUDA MoE tests must not bypass the production "
               "capture owner";
        EXPECT_EQ(test_source.find("cudaStreamEndCapture("),
                  std::string::npos)
            << "Capture closure must remain structurally paired by RAII";
    }

    /**
     * @brief Keep monolithic prefill capture inside the same structural owner.
     *
     * Prefill historically exposed separate begin, abort, and end methods. Any
     * early return between those calls could strand the native stream in capture
     * mode and make a later synchronization illegal. The cache now accepts one
     * graph-body callback, closes capture before inspecting its result, and
     * publishes only a successfully instantiated executable.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PrefillCaptureIsOneScopedTransaction)
    {
        const fs::path root = findRepoRoot();
        const fs::path cache_header_path =
            root / "src/v2/execution/local_execution/engine/PrefillGraphCache.h";
        const fs::path cache_source_path =
            root / "src/v2/execution/local_execution/engine/PrefillGraphCache.cpp";
        const fs::path engine_path =
            root / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp";
        ASSERT_TRUE(fs::exists(cache_header_path)) << cache_header_path;
        ASSERT_TRUE(fs::exists(cache_source_path)) << cache_source_path;
        ASSERT_TRUE(fs::exists(engine_path)) << engine_path;

        const std::string cache_header = readFile(cache_header_path);
        const std::string cache_source = readFile(cache_source_path);
        const std::string engine = readFile(engine_path);
        ASSERT_FALSE(cache_header.empty());
        ASSERT_FALSE(cache_source.empty());
        ASSERT_FALSE(engine.empty());

        EXPECT_NE(cache_header.find("bool captureAndInstantiate("),
                  std::string::npos);
        EXPECT_EQ(cache_header.find("bool beginCapture("),
                  std::string::npos);
        EXPECT_EQ(cache_header.find("bool endCaptureAndInstantiate("),
                  std::string::npos);
        EXPECT_EQ(cache_header.find("bool abortCapture("),
                  std::string::npos)
            << "Callers must not own partial native capture lifecycles";

        const size_t transaction = cache_source.find(
            "ScopedBackendGraphCapture capture_transaction(");
        const size_t body = cache_source.find(
            "body_succeeded = record_graph_body();",
            transaction);
        const size_t finish = cache_source.find(
            "capture_transaction.finish();",
            body);
        const size_t failure = cache_source.find(
            "if (!body_succeeded)",
            finish);
        ASSERT_NE(transaction, std::string::npos);
        ASSERT_NE(body, std::string::npos);
        ASSERT_NE(finish, std::string::npos);
        ASSERT_NE(failure, std::string::npos);
        EXPECT_LT(transaction, body);
        EXPECT_LT(body, finish);
        EXPECT_LT(finish, failure)
            << "Capture closure must happen before graph-body failure propagates";

        EXPECT_NE(engine.find("cache.captureAndInstantiate("),
                  std::string::npos);
        EXPECT_EQ(engine.find("cache.abortCapture("),
                  std::string::npos);
        EXPECT_EQ(engine.find("cache.endCaptureAndInstantiate("),
                  std::string::npos);
    }

    /**
     * @brief Prevent graph-cache cleanup from tolerating poisoned GPU state.
     *
     * Once a graph cache owns a stream or event, failure to resolve its owner,
     * drain queued work, or destroy the resource makes its lifetime unknowable.
     * These conditions must remain centralized process-fatal invariants rather
     * than log-and-continue branches hidden in individual cleanup methods.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GraphCacheResourceLifecycleFailuresAreFatal)
    {
        const fs::path root = findRepoRoot();
        const fs::path executor_path =
            root / "src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp";
        ASSERT_TRUE(fs::exists(executor_path)) << executor_path;

        const std::string executor = readFile(executor_path);
        ASSERT_FALSE(executor.empty()) << executor_path;

        EXPECT_NE(executor.find(
                      "[[noreturn]] void terminateGraphSegmentCacheLifecycle("),
                  std::string::npos)
            << "All live stream/event lifecycle failures need one non-recoverable policy";
        EXPECT_NE(executor.find(
                      "requireLifecycleContext(\"capture stream event fence\")"),
                  std::string::npos);
        EXPECT_NE(executor.find(
                      "requireLifecycleContext(\"capture stream destruction\")"),
                  std::string::npos);
        EXPECT_NE(executor.find(
                      "requireLifecycleContext(\"sync event destruction\")"),
                  std::string::npos);
        EXPECT_NE(executor.find(
                      "capture stream event fence allocation was rejected by the backend"),
                  std::string::npos);
        EXPECT_NE(executor.find(
                      "capture stream event fence publication was rejected by the backend"),
                  std::string::npos);
        EXPECT_NE(executor.find(
                      "capture stream event fence wait was rejected by the backend"),
                  std::string::npos);
        EXPECT_EQ(executor.find("synchronizeStreamChecked(capture_stream)"),
                  std::string::npos)
            << "Graph-cache lifecycle ordering must use the explicit event fence.";
        EXPECT_EQ(executor.find(
                      "Dropping capture stream handle without destruction"),
                  std::string::npos);
        EXPECT_EQ(executor.find(
                      "Dropping sync event handle without destruction"),
                  std::string::npos);
        EXPECT_EQ(executor.find(
                      "Cannot synchronize capture stream: no live GPU context"),
                  std::string::npos);
    }

    /**
     * @brief Keep graph execution policy independent of prior launch failures.
     *
     * Backend capability and explicit configuration may select eager or graph
     * execution before a request runs. A failed attempted graph launch must not
     * accumulate a retry budget that silently changes a later request to eager
     * execution. Direct maintenance replay similarly needs a typed
     * applicability result so launch failure cannot masquerade as
     * `NotApplicable`.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GraphReplayHasNoFailureBudgetOrRuntimePathFallback)
    {
        const fs::path root = findRepoRoot();
        const fs::path executor_header =
            root / "src/v2/execution/local_execution/graph/DeviceGraphExecutor.h";
        const fs::path controller_header =
            root / "src/v2/execution/local_execution/graph/DeviceGraphCaptureController.h";
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(executor_header)) << executor_header;
        ASSERT_TRUE(fs::exists(controller_header)) << controller_header;
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;

        const std::string executor = readFile(executor_header);
        const std::string controller = readFile(controller_header);
        const std::string orchestrator = readFile(orchestrator_path);

        for (const auto &[name, contents] :
             std::array<std::pair<const char *, const std::string *>, 3>{
                 std::pair{"executor", &executor},
                 std::pair{"controller", &controller},
                 std::pair{"orchestrator", &orchestrator}})
        {
            EXPECT_EQ(contents->find("consecutive_failures"), std::string::npos)
                << name << " must not retain runtime graph-failure policy state";
            EXPECT_EQ(contents->find("max_segment_failures"), std::string::npos)
                << name << " must not retain graph-failure retry budgets";
            EXPECT_EQ(contents->find("launch_failure_fallback"), std::string::npos)
                << name << " must not expose a path-substitution hint after launch";
        }

        EXPECT_NE(orchestrator.find(
                      "enum class DirectMoEMaintenanceReplayResult"),
                  std::string::npos);
        EXPECT_NE(orchestrator.find(
                      "DirectMoEMaintenanceReplayResult::NotApplicable"),
                  std::string::npos);
        EXPECT_NE(orchestrator.find(
                      "Direct MoE rebalance maintenance graph launch failed "),
                  std::string::npos)
            << "A selected direct graph launch must throw rather than enter generic replay";
        EXPECT_NE(orchestrator.find(
                      "\"after the path was selected on \""),
                  std::string::npos);
        EXPECT_EQ(orchestrator.find(
                      "falling back to generic graph replay"),
                  std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, RebalanceCapacityBackpressureIsNotCommandOverflow)
    {
        const fs::path root = findRepoRoot();
        const fs::path controller_path =
            root / "src/v2/execution/moe/DeviceMoERebalanceController.h";
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(controller_path)) << controller_path;
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;

        const std::string controller = readFile(controller_path);
        const std::string cuda = readFile(cuda_path);
        const std::string rocm = readFile(rocm_path);
        const std::string orchestrator = readFile(orchestrator_path);
        EXPECT_NE(controller.find("uint32_t capacity_limited_candidates = 0;"),
                  std::string::npos)
            << "The stable status ABI must distinguish planned wave saturation from corruption";
        EXPECT_NE(controller.find("plan_overflow is"),
                  std::string::npos)
            << "The status contract must document that true command overflow remains fatal";
        for (const auto &[backend, source] :
             std::array<std::pair<const char *, const std::string *>, 2>{
                 std::pair{"CUDA", &cuda},
                 std::pair{"ROCm", &rocm}})
        {
            EXPECT_NE(source->find("++status->capacity_limited_candidates;"),
                      std::string::npos)
                << backend << " must stop candidate selection cleanly at the configured payload budget";
            EXPECT_NE(source->find("++status->plan_overflow;"),
                      std::string::npos)
                << backend << " must retain fatal accounting for true command-buffer overflow";
        }
        EXPECT_NE(orchestrator.find(
                      "\"device_rebalance_capacity_limited_candidates\""),
                  std::string::npos)
            << "Capacity pressure must remain observable instead of being silently discarded";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, PrefillMovementEvidenceIsDeviceResidentAndBackendSymmetric)
    {
        const fs::path root = findRepoRoot();
        const fs::path controller_path =
            root / "src/v2/execution/moe/DeviceMoERebalanceController.h";
        const fs::path runtime_path =
            root / "src/v2/execution/moe/MoERuntimeTable.h";
        const fs::path expert_stage_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        const fs::path e2e_path =
            root / "tests/v2/e2e/server/test_server_e2e.sh";
        const fs::path parity_base_path =
            root / "tests/v2/integration/parity/qwen36/Qwen36MoEParityTestBase.h";
        ASSERT_TRUE(fs::exists(controller_path)) << controller_path;
        ASSERT_TRUE(fs::exists(runtime_path)) << runtime_path;
        ASSERT_TRUE(fs::exists(expert_stage_path)) << expert_stage_path;
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;
        ASSERT_TRUE(fs::exists(e2e_path)) << e2e_path;
        ASSERT_TRUE(fs::exists(parity_base_path)) << parity_base_path;

        const std::string controller = readFile(controller_path);
        const std::string runtime = readFile(runtime_path);
        const std::string expert_stage = readFile(expert_stage_path);
        const std::string orchestrator = readFile(orchestrator_path);
        const std::string cuda = readFile(cuda_path);
        const std::string rocm = readFile(rocm_path);
        const std::string e2e = readFile(e2e_path);
        const std::string parity_base = readFile(parity_base_path);

        EXPECT_NE(controller.find(
                      "uint32_t prefill_current_batch_movement_layers = 0;"),
                  std::string::npos)
            << "The request-boundary status ABI must carry current-batch movement history";
        EXPECT_NE(runtime.find(
                      "uint32_t current_batch_llep_movement_observed = 0;"),
                  std::string::npos)
            << "Current-batch movement needs a device-owned marker distinct from portable prefix state";
        EXPECT_NE(orchestrator.find(
                      "\"device_rebalance_prefill_current_batch_movement_layers\""),
                  std::string::npos)
            << "Current-batch movement history must use the existing status readback";
        EXPECT_NE(e2e.find(
                      "prefill_applied_movement = record_value_sum("),
                  std::string::npos)
            << "The canonical E2E movement gate must consume stronger final-state evidence";
        EXPECT_NE(parity_base.find(
                      "expectLLEPAppliedPrefillMovementPositive(records, context)"),
                  std::string::npos)
            << "Focused Qwen 3.6 parity must consume the same durable movement proof as E2E";
        EXPECT_NE(parity_base.find(
                      "\"device_rebalance_prefill_current_batch_movement_layers\""),
                  std::string::npos)
            << "Parity movement proof must inspect current-batch device history";
        EXPECT_NE(expert_stage.find(
                      "params_.prefill_llep_rebalance_config\n"
                      "                            .min_wave_spread_improvement_per_payload_slot"),
                  std::string::npos)
            << "Current-batch planning must use the graph-owned typed economy policy";
        EXPECT_EQ(expert_stage.find(
                      "moe_env\n"
                      "                                    .device_rebalance_min_wave_spread_improvement_per_payload_slot"),
                  std::string::npos)
            << "Current-batch planning must not reread process-global economy defaults";
        EXPECT_EQ(parity_base.find(
                      "expectPerfCounterPositive(\n"
                      "                records,\n"
                      "                \"moe_rebalance\",\n"
                      "                \"device_rebalance_llep_weight_transfer_count\""),
                  std::string::npos)
            << "A transient per-layer planner count must not masquerade as request-final movement evidence";

        for (const auto &[backend, source] :
             std::array<std::pair<const char *, const std::string *>, 2>{
                 std::pair{"CUDA", &cuda},
                 std::pair{"ROCm", &rocm}})
        {
            EXPECT_NE(source->find(
                          "uint32_t prefill_current_batch_movement_layers;"),
                      std::string::npos)
                << backend << " must mirror the current-batch status ABI field";
            EXPECT_GE(countOccurrences(
                          *source,
                          "rebalance_publish_transfer_slot_claim_summary("),
                      3u)
                << backend << " must invoke one shared final-state publisher "
                   "from both the standard and graph-controller kernels.";
            EXPECT_NE(source->find(
                          "status->prefill_current_batch_movement_layers ="),
                      std::string::npos)
                << backend << " shared publisher must derive current-batch movement "
                   "from the complete device-resident runtime audit.";
            EXPECT_NE(source->find(
                          "runtime.current_batch_llep_movement_observed != 0u"),
                      std::string::npos)
                << backend << " must not confuse portable bank state with fresh movement";
            EXPECT_NE(source->find(
                          "kDeviceMoERebalancePlanFlagCurrentBatchLLEP"),
                      std::string::npos)
                << backend << " current-batch plan entries must carry semantic provenance";
            EXPECT_NE(source->find(
                          "prior_status.prefill_active_transfer_slot_experts"),
                      std::string::npos)
                << backend << " domain command projection must preserve final-state evidence";
        }

        EXPECT_EQ(orchestrator.find(
                      "deviceToHostOnStream(&prefill_current_batch_movement_layers"),
                  std::string::npos)
            << "Movement evidence must share the request-boundary status copy, not add hot-path D2H";
    }

    /**
     * @brief Keep current-batch LLEP planning cooperative and scratch-aware.
     *
     * The planner's status object is large enough to become private scratch on
     * gfx906 when the shared algorithm owns a temporary and copies it to an
     * optional output.  Production GPU callers must instead own the status,
     * and both backends must construct the canonical expert order
     * cooperatively before the leader performs the dependency-ordered policy
     * tail.  The backend regressions use the complete 256-expert codebook so a
     * future scalar-sort regression cannot hide behind a toy geometry.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, CurrentBatchLLEPPlannerOwnershipAndParallelSortAreStructural)
    {
        const fs::path root = findRepoRoot();
        const fs::path planner_path =
            root / "src/v2/execution/moe/LeastLoadedExpertAssignment.h";
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        const fs::path cuda_test_path =
            root / "tests/v2/integration/kernels/cuda/Test__CUDAMoEKernel.cpp";
        const fs::path rocm_test_path =
            root / "tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp";

        const std::string planner = readFile(planner_path);
        const std::string cuda = readFile(cuda_path);
        const std::string rocm = readFile(rocm_path);
        const std::string cuda_test = readFile(cuda_test_path);
        const std::string rocm_test = readFile(rocm_test_path);

        EXPECT_GE(countOccurrences(
                      planner,
                      "LeastLoadedExpertAssignmentStatus &status"),
                  2u)
            << "Full and transfer-only planners must write caller-owned status";
        EXPECT_EQ(planner.find("LeastLoadedExpertAssignmentStatus *status_out"),
                  std::string::npos)
            << "Nullable copy-back status recreates GPU private scratch";

        EXPECT_NE(cuda.find("sort_current_batch_llep_experts_parallel("),
                  std::string::npos)
            << "CUDA current-batch planning must cooperatively sort experts";
        EXPECT_NE(rocm.find("rebalance_sort_llep_experts_parallel("),
                  std::string::npos)
            << "ROCm current-batch planning must cooperatively sort experts";
        for (const auto &[backend, source] :
             std::array<std::pair<const char *, const std::string *>, 2>{
                 std::pair{"CUDA", &cuda},
                 std::pair{"ROCm", &rocm}})
        {
            EXPECT_NE(source->find(
                          "/*workspace_experts_are_sorted=*/true"),
                      std::string::npos)
                << backend << " must not repeat the canonical sort on its leader lane";
        }
        EXPECT_NE(rocm.find("planner_status_storage"), std::string::npos)
            << "gfx906 planner status must remain in explicit shared storage";

        for (const auto &[backend, source] :
             std::array<std::pair<const char *, const std::string *>, 2>{
                 std::pair{"CUDA", &cuda_test},
                 std::pair{"ROCm", &rocm_test}})
        {
            EXPECT_NE(source->find(
                          "RuntimePrefillLeastLoadedCurrentBatchPlannerMaterializesSpans"),
                      std::string::npos)
                << backend << " must retain focused current-batch planner coverage";
            EXPECT_NE(source->find("constexpr int num_experts = 256;"),
                      std::string::npos)
                << backend << " planner regression must exercise the production codebook width";
        }
    }

    /**
     * @brief Keep active occupancy distinct from directory addressability.
     *
     * Staging slots are spare capacity, not a permanently reserved index range.
     * Once a transfer is atomically published, any addressable directory slot
     * may hold a durable claim while the total number of claims remains bounded
     * by the active capacity. This sanitizer prevents either GPU backend or the
     * graph builder from collapsing those two independent quantities again.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         TransferSlotCapacityContractSeparatesOccupancyFromAddressability)
    {
        const fs::path root = findRepoRoot();
        const fs::path controller_path =
            root / "src/v2/execution/moe/DeviceMoERebalanceController.h";
        const fs::path policy_path =
            root / "src/v2/execution/moe/DeviceMoERebalancePolicyShared.h";
        const fs::path abi_path =
            root / "src/v2/execution/moe/DeviceMoERebalanceABI.h";
        const fs::path graph_path =
            root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        for (const auto &path :
             {controller_path,
              policy_path,
              abi_path,
              graph_path,
              cuda_path,
              rocm_path})
        {
            ASSERT_TRUE(fs::exists(path)) << path;
        }

        const std::string controller = readFile(controller_path);
        const std::string policy = readFile(policy_path);
        const std::string abi = readFile(abi_path);
        const std::string graph = readFile(graph_path);
        const std::string cuda = readFile(cuda_path);
        const std::string rocm = readFile(rocm_path);

        EXPECT_EQ(
            countOccurrences(
                controller,
                "uint32_t active_transfer_slot_capacity"),
            1u);
        EXPECT_EQ(
            countOccurrences(
                controller,
                "uint32_t transfer_slot_directory_capacity"),
            1u);
        EXPECT_NE(
            controller.find(
                "config.active_transfer_slot_capacity <=\n"
                "                   config.transfer_slot_directory_capacity"),
            std::string::npos)
            << "Host validation must reject an active budget larger than its directory";
        EXPECT_NE(
            controller.find(
                "config.transfer_slot_directory_capacity,\n"
                "                        valid_flag"),
            std::string::npos)
            << "Host claim classification must validate physical slot IDs against the directory";
        EXPECT_NE(
            policy.find(
                "TransferSlotClaimExceedsDirectoryCapacity"),
            std::string::npos);
        EXPECT_EQ(
            policy.find(
                "TransferSlotClaimExceedsPersistentCapacity"),
            std::string::npos);

        EXPECT_EQ(
            countOccurrences(
                graph,
                "rebalance_config.active_transfer_slot_capacity ="),
            2u);
        EXPECT_EQ(
            countOccurrences(
                graph,
                "rebalance_config.transfer_slot_directory_capacity ="),
            2u);
        EXPECT_EQ(
            countOccurrences(graph, "transfer_capacity.active_slots;"),
            2u);
        EXPECT_EQ(
            countOccurrences(graph, "transfer_capacity.total_slots;"),
            4u)
            << "Each path uses total slots for both config addressability and directory allocation";

        EXPECT_NE(
            abi.find("kVersion = 9u"),
            std::string::npos);
        EXPECT_NE(
            abi.find("kConfigBytes = 140u"),
            std::string::npos);

        for (const auto &[backend, source] :
             std::array<std::pair<const char *, const std::string *>, 2>{
                 std::pair{"CUDA", &cuda},
                 std::pair{"ROCm", &rocm}})
        {
            EXPECT_EQ(
                countOccurrences(
                    *source,
                    "uint32_t active_transfer_slot_capacity;"),
                1u)
                << backend << " config view must mirror active occupancy";
            EXPECT_EQ(
                countOccurrences(
                    *source,
                    "uint32_t transfer_slot_directory_capacity;"),
                1u)
                << backend << " config view must mirror directory addressability";
            EXPECT_NE(
                source->find(
                    "config.transfer_slot_directory_capacity,\n"
                    "                            kDeviceMoEFlagValid"),
                std::string::npos)
                << backend << " claim classifier must use the directory bound";
            EXPECT_GE(
                countOccurrences(
                    *source,
                    "config.active_transfer_slot_capacity"),
                5u)
                << backend << " planner and final active-count audit must use the occupancy bound";
            EXPECT_EQ(
                source->find("persistent_transfer_slot_capacity"),
                std::string::npos)
                << backend << " must not restore the overloaded capacity field";
            EXPECT_NE(
                source->find(
                    "moe_rebalance_abi::kConfigBytes"),
                std::string::npos)
                << backend << " must compile-time authenticate the config ABI";
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, CUDALLEPFatalContractsNameTheViolatedInvariant)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;

        const std::string cuda = readFile(cuda_path);
        EXPECT_NE(cuda.find(
                      "__assert_fail(invariant, __FILE__, line, function);"),
                  std::string::npos)
            << "Optimized CUDA builds must publish the fatal LLEP invariant through "
               "the device assertion channel";
        EXPECT_EQ(cuda.find("asm volatile(\"trap;\");"), std::string::npos)
            << "Anonymous PTX traps erase the first violated publication contract "
               "and leave only cudaErrorLaunchFailure";

        for (const char *required_reason :
             {
                 "post-transfer LLEP status is incomplete or inconsistent",
                 "default-owner LLEP assignment is not locally compute-ready",
                 "missing-span LLEP assignment is not locally compute-ready",
                 "planned LLEP span assigns a non-ready local expert",
                 "runtime active expert is not locally resident and compute-ready",
                 "runtime grouped route row does not match its original route slot",
             })
        {
            EXPECT_NE(cuda.find(required_reason), std::string::npos)
                << "Every fatal CUDA LLEP contract needs a stable diagnostic reason: "
                << required_reason;
        }
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GPUTransferSlotsHaveSingleDeviceOwnedLogicalOccupants)
    {
        const fs::path root = findRepoRoot();
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        const fs::path shared_policy_path =
            root /
            "src/v2/execution/moe/DeviceMoERebalancePolicyShared.h";
        ASSERT_TRUE(fs::exists(cuda_path)) << cuda_path;
        ASSERT_TRUE(fs::exists(rocm_path)) << rocm_path;
        ASSERT_TRUE(fs::exists(shared_policy_path))
            << shared_policy_path;

        const std::string cuda = readFile(cuda_path);
        const std::string rocm = readFile(rocm_path);
        const std::string shared_policy =
            readFile(shared_policy_path);
        ASSERT_FALSE(shared_policy.empty());
        for (const auto &[backend, source] :
             std::array<std::pair<const char *, const std::string *>, 2>{
                 std::pair{"CUDA", &cuda},
                 std::pair{"ROCm", &rocm}})
        {
            EXPECT_NE(source->find(
                          "rebalance_transfer_slot_has_active_runtime_claim("),
                      std::string::npos)
                << backend
                << " materialization must prove that a transfer-slot occupant is "
                   "still claimed by an active runtime bank";
            EXPECT_NE(source->find("occupancy.protected_from_reuse"),
                      std::string::npos)
                << backend
                << " materialization must protect authoritative and current-row slots";
            EXPECT_NE(source->find(
                          "moe_rebalance_policy::classifyTransferSlotOccupancy("),
                      std::string::npos)
                << backend
                << " storage lifetime must use the shared residency/ownership policy";
            EXPECT_NE(source->find(
                          "prefillAssignmentReadsTransferSlotOccupant("),
                      std::string::npos)
                << backend
                << " prefill leasing must share the forward-layer lifetime policy";
            EXPECT_EQ(source->find("prior.layer != plan.layer"),
                      std::string::npos)
                << backend
                << " the bounded transfer directory must reclaim event-ordered "
                   "non-owner replicas across layer boundaries";
            EXPECT_NE(source->find(
                          "const auto &occupant_runtime = runtime_layers[prior.layer];"),
                      std::string::npos)
                << backend
                << " cross-layer reuse must classify ownership and residency "
                   "against the prior occupant's runtime bank";
            EXPECT_NE(source->find(
                          "const bool has_active_runtime_claim ="),
                      std::string::npos)
                << backend
                << " prefill leasing must derive post-reset liveness from the "
                   "authoritative device runtime, not a stale directory epoch";
            EXPECT_NE(source->find(
                          "if (!has_active_runtime_claim)"),
                      std::string::npos)
                << backend
                << " event-ordered prefill must reclaim persistent directory "
                   "occupants whose request-local runtime claims were reset";
            EXPECT_EQ(source->find("const bool locally_live ="),
                      std::string::npos)
                << backend
                << " compute eligibility must never serve as a transfer-slot lease";
            EXPECT_NE(source->find(
                          "rebalance_lease_local_transfer_slot("),
                      std::string::npos)
                << backend
                << " destination-local projection must choose physical storage from its authoritative directory";
            EXPECT_NE(source->find(
                          "plan.destination_previous_layer = prior.layer;"),
                      std::string::npos)
                << backend
                << " each physical-slot lease must record the prior logical layer";
            EXPECT_NE(source->find(
                          "plan.destination_previous_expert = prior.expert;"),
                      std::string::npos)
                << backend
                << " each physical-slot lease must record the prior logical expert";
            EXPECT_NE(source->find(
                          "plan.destination_generation = prior.generation;"),
                      std::string::npos)
                << backend
                << " each physical-slot lease must carry a compare-and-replace generation";
            EXPECT_NE(source->find(
                          "rebalance_any_active_runtime_claims_slot("),
                      std::string::npos)
                << backend
                << " physical storage must remain protected while any active bank references it";
            EXPECT_NE(source->find(
                          "rebalance_transfer_slot_copy_complete_for_plan("),
                      std::string::npos)
                << backend
                << " apply must authenticate the generation transition and command epoch";
            EXPECT_NE(source->find(
                          "dst.generation == plan.destination_generation"),
                      std::string::npos)
                << backend
                << " unpack must compare the live directory generation before mutation";
            EXPECT_GE(countOccurrences(
                          *source,
                          "rebalance_ready_arrival_reuses_runtime_slot("),
                      3u)
                << backend
                << " both bulk apply and graph-controller apply must enforce slot reuse";
            EXPECT_GE(countOccurrences(
                          *source,
                          "rebalance_retire_reused_transfer_slot_replica("),
                      3u)
                << backend
                << " every apply entry point must retire the prior logical occupant";
            EXPECT_GE(
                countOccurrences(
                    *source,
                    "rebalance_retire_local_payload_publication("),
                3u)
                << backend
                << " every ownership and slot-reuse path must delegate one "
                   "atomic local-payload retirement transition";
            EXPECT_NE(
                source->find(
                    "rebalance_plan_duplicates_prior_local_arrival("),
                std::string::npos)
                << backend
                << " apply must reject duplicate physical-slot and logical-expert keys";
            EXPECT_NE(
                source->find(
                    "rebalance_apply_transaction_blocked(status)"),
                std::string::npos)
                << backend
                << " preflight must gate the entire multi-arrival bank commit";
            EXPECT_NE(
                source->find(
                    "permanent_plan_error"),
                std::string::npos)
                << backend
                << " malformed ready waves must poison instead of retrying forever";
        }

        EXPECT_NE(
            shared_policy.find(
                "void retireLocalPayloadPublication("),
            std::string::npos);
        EXPECT_NE(
            shared_policy.find("descriptor.local_slot = -1;"),
            std::string::npos);
        EXPECT_NE(
            shared_policy.find(
                "descriptor.gate = decltype(descriptor.gate){};"),
            std::string::npos);
        EXPECT_NE(
            shared_policy.find(
                "descriptor.up = decltype(descriptor.up){};"),
            std::string::npos);
        EXPECT_NE(
            shared_policy.find(
                "descriptor.down = decltype(descriptor.down){};"),
            std::string::npos);
        EXPECT_NE(
            shared_policy.find(
                "descriptor.flags &= ~local_payload_flags;"),
            std::string::npos);
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
             "} // namespace llaminar2"},
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
        EXPECT_NE(graph.find("kPrefillLLEPTransferWorkspaceLanes"),
                  std::string::npos)
            << "Current-batch LLEP needs a bounded persistent workspace ring.";
        EXPECT_NE(graph.find("layer_idx % kPrefillLLEPTransferWorkspaceLanes"),
                  std::string::npos)
            << "Layer workspaces must reuse the fixed ring after their graph event join.";
        EXPECT_EQ(graph.find("graphRebalancePrefillExecutionLifetimeKey"),
                  std::string::npos)
            << "Per-layer payload workspace lifetimes exhaust production GPU memory.";

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

        EXPECT_NE(stage.find("assignPrefillRoutesLeastLoadedResident"),
                  std::string::npos)
            << "Resident-only LLEP must use the planner whose candidate set is "
               "limited to the active runtime bank's resident masks.";
        EXPECT_EQ(stage.find("assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers"),
                  std::string::npos)
            << "Production resident-only LLEP must not first create a potentially "
               "foreign transfer plan and then conditionally decline to apply it.";
        EXPECT_NE(stage.find("assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers"),
                  std::string::npos)
            << "Full transfer-backed prefill LLEP must use the after-transfer apply kernel.";
        EXPECT_NE(stage.find("requestsTransferBackedCurrentBatchPrefillLLEP"),
                  std::string::npos)
            << "Current-batch migration must be selected by an explicit typed "
               "graph-build policy, not inferred from nullable transport pointers.";
        EXPECT_NE(graph.find("const bool current_batch_llep_transfer_candidate"),
                  std::string::npos)
            << "The graph must name current-batch migration independently from "
               "prefix-runtime payload rehydration.";
        EXPECT_NE(graph.find("!grouped_main_verifier_layer"),
                  std::string::npos)
            << "Grouped verifier rows must be structurally excluded from "
               "current-batch expert payload migration.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         RebalanceAndPrefillLLEPUseImmutableStageOwnedLaunchContexts)
    {
        const fs::path root = findRepoRoot();
        const fs::path interface_path = root / "src/v2/kernels/IMoEKernel.h";
        const fs::path cuda_path = root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp";
        const fs::path rocm_path = root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp";
        const fs::path expert_stage_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        const fs::path maintenance_stage_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.cpp";
        const fs::path graph_path =
            root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";

        for (const auto &path : {
                 interface_path,
                 cuda_path,
                 rocm_path,
                 expert_stage_path,
                 maintenance_stage_path,
                 graph_path})
        {
            ASSERT_TRUE(fs::exists(path)) << path;
        }

        const std::string interface_source = readFile(interface_path);
        const std::string cuda_source = readFile(cuda_path);
        const std::string rocm_source = readFile(rocm_path);
        const std::string expert_stage_source = readFile(expert_stage_path);
        const std::string maintenance_stage_source = readFile(maintenance_stage_path);
        const std::string graph_source = readFile(graph_path);
        ASSERT_FALSE(interface_source.empty());
        ASSERT_FALSE(cuda_source.empty());
        ASSERT_FALSE(rocm_source.empty());
        ASSERT_FALSE(expert_stage_source.empty());
        ASSERT_FALSE(maintenance_stage_source.empty());
        ASSERT_FALSE(graph_source.empty());

        EXPECT_NE(interface_source.find("struct MoEKernelLaunchContext"), std::string::npos)
            << "MoE launch ownership must be represented by an explicit immutable context.";
        EXPECT_NE(interface_source.find("const MoEKernelLaunchContext &launch"),
                  std::string::npos)
            << "Rebalance and LLEP interfaces must require the launch context.";
        EXPECT_EQ(interface_source.find("LaunchStateLease"), std::string::npos)
            << "A mutex lease would serialize independent graph capture and preserve mutable ownership.";
        EXPECT_EQ(interface_source.find("launch_state_mutex"), std::string::npos)
            << "MoE launch ownership must not regress to a process-wide lock.";

        for (const auto &[backend_name, backend_source] :
             std::vector<std::pair<std::string, std::string>>{
                 {"CUDA", cuda_source},
                 {"ROCm", rocm_source}})
        {
            EXPECT_NE(backend_source.find("explicitMoELaunchStream(launch"),
                      std::string::npos)
                << backend_name
                << " rebalance/LLEP kernels must consume the caller-owned stream.";
        }

        const size_t llep_begin = expert_stage_source.find(
            "bool MoEExpertComputeStage::executeTransferBackedPrefillLLEPMovement");
        const size_t llep_end = expert_stage_source.find(
            "bool MoEExpertComputeStage::executeFixedTopologyGroupedPrefill",
            llep_begin);
        ASSERT_NE(llep_begin, std::string::npos);
        ASSERT_NE(llep_end, std::string::npos);
        const std::string llep_body =
            expert_stage_source.substr(llep_begin, llep_end - llep_begin);

        EXPECT_NE(llep_body.find("const MoEKernelLaunchContext compute_launch"),
                  std::string::npos);
        EXPECT_NE(llep_body.find("const MoEKernelLaunchContext transfer_launch"),
                  std::string::npos);
        EXPECT_NE(llep_body.find("recordEventChecked"), std::string::npos)
            << "Compute/transfer publication must remain ordered by graph-captured events.";
        EXPECT_NE(llep_body.find("waitEventChecked"), std::string::npos)
            << "Compute/transfer publication must remain ordered by graph-captured events.";
        EXPECT_EQ(llep_body.find("kernel->setGPUStream("), std::string::npos)
            << "Prefill LLEP must not mutate inherited launch state.";
        EXPECT_EQ(llep_body.find("synchronizeStream"), std::string::npos)
            << "Prefill LLEP hot-path ordering must not involve a host stream fence.";
        EXPECT_EQ(llep_body.find("DeviceSynchronize"), std::string::npos)
            << "Prefill LLEP must not introduce a full-device synchronization.";
        EXPECT_EQ(llep_body.find("StreamSynchronize"), std::string::npos)
            << "Prefill LLEP must not introduce a host stream synchronization.";

        const size_t trace_begin = expert_stage_source.find(
            "bool tracePrefillLLEPStatus(");
        const size_t trace_end = expert_stage_source.find(
            "bool fusedSwigluDown(",
            trace_begin);
        ASSERT_NE(trace_begin, std::string::npos);
        ASSERT_NE(trace_end, std::string::npos);
        const std::string trace_body =
            expert_stage_source.substr(trace_begin, trace_end - trace_begin);
        const size_t capture_guard = trace_body.find("if (isGraphCaptureActive())");
        const size_t device_readback = trace_body.find("deviceToHostOnStream");
        ASSERT_NE(capture_guard, std::string::npos)
            << "Optional LLEP diagnostics must recognize native graph capture.";
        ASSERT_NE(device_readback, std::string::npos)
            << "The eager diagnostic still needs an exact status readback.";
        EXPECT_LT(capture_guard, device_readback)
            << "Graph capture must return before diagnostics enqueue any D2H readback.";
        EXPECT_NE(
            trace_body.find(
                "device_rebalance_llep_prefill_status_trace_deferred"),
            std::string::npos)
            << "PerfStats must expose that graph-owned status diagnostics were deferred.";

        const size_t assignment_trace_begin = expert_stage_source.find(
            "bool tracePrefillAssignmentRuntime(");
        const size_t assignment_trace_end = expert_stage_source.find(
            "bool tracePrefillLLEPStatus(",
            assignment_trace_begin);
        ASSERT_NE(assignment_trace_begin, std::string::npos);
        ASSERT_NE(assignment_trace_end, std::string::npos);
        const std::string assignment_trace_body =
            expert_stage_source.substr(
                assignment_trace_begin,
                assignment_trace_end - assignment_trace_begin);
        const size_t assignment_capture_guard =
            assignment_trace_body.find("if (isGraphCaptureActive())");
        const size_t assignment_readback =
            assignment_trace_body.find("copyTraceBuffer(");
        ASSERT_NE(assignment_capture_guard, std::string::npos);
        ASSERT_NE(assignment_readback, std::string::npos);
        EXPECT_LT(assignment_capture_guard, assignment_readback)
            << "Assignment diagnostics must return before graph capture can enqueue D2H work.";
        EXPECT_NE(
            assignment_trace_body.find(
                "device_rebalance_llep_prefill_assignment_trace_deferred"),
            std::string::npos);
        EXPECT_NE(
            expert_stage_source.find(
                "trace_runtime_assignment(\"after_pipeline\")"),
            std::string::npos)
            << "The final diagnostic boundary must attribute asynchronous grouped-kernel faults.";

        EXPECT_NE(
            graph_source.find(
                "transfer_key + \":workspace=\" + rebalance_workspace"),
            std::string::npos)
            << "Every layer must own distinct transfer-event handles even when "
               "the bounded stream/workspace lane is shared.";
        EXPECT_EQ(
            graph_source.find(
                "graphRebalanceDomainKey() + \":workspace=\""),
            std::string::npos)
            << "Domain-lane event aliasing makes multi-layer graph dependencies ambiguous.";

        const size_t llep_prepare_begin = expert_stage_source.find(
            "bool MoEExpertComputeStage::prepareGraphLaunch");
        const size_t llep_prepare_end = expert_stage_source.find(
            "StageBufferRequirements MoEExpertComputeStage::getBufferRequirements",
            llep_prepare_begin);
        ASSERT_NE(llep_prepare_begin, std::string::npos);
        ASSERT_NE(llep_prepare_end, std::string::npos);
        const std::string llep_prepare_body =
            expert_stage_source.substr(
                llep_prepare_begin,
                llep_prepare_end - llep_prepare_begin);
        EXPECT_NE(llep_prepare_body.find("prefill_llep_transfer_state->prepareForCapture("),
                  std::string::npos)
            << "Prefill LLEP must preflight its rolling transfer lane before graph capture.";
        EXPECT_NE(
            llep_prepare_body.find(
                "device_rebalance_llep_prefill_precapture_lane_event_fence"),
            std::string::npos)
            << "PerfStats must prove prefill LLEP used the pre-capture lane fence.";
        EXPECT_EQ(llep_prepare_body.find("synchronize"), std::string::npos)
            << "Prefill capture preparation must use event ordering, never a host fence.";

        const size_t maintenance_begin = maintenance_stage_source.find(
            "bool MoEDeviceRebalanceStage::execute(IDeviceContext *ctx)");
        const size_t maintenance_end = maintenance_stage_source.find(
            "bool MoEDeviceRebalanceStage::prepareGraphLaunch",
            maintenance_begin);
        ASSERT_NE(maintenance_begin, std::string::npos);
        ASSERT_NE(maintenance_end, std::string::npos);
        const std::string maintenance_body =
            maintenance_stage_source.substr(
                maintenance_begin,
                maintenance_end - maintenance_begin);

        EXPECT_NE(maintenance_body.find("const MoEKernelLaunchContext compute_launch"),
                  std::string::npos);
        EXPECT_NE(maintenance_body.find("MoEKernelLaunchContext transfer_launch"),
                  std::string::npos);
        EXPECT_EQ(maintenance_body.find("moe_kernel->setGPUStream("), std::string::npos)
            << "Maintenance must pass launch resources explicitly instead of retargeting inherited state.";
        EXPECT_EQ(maintenance_body.find("synchronizeStream("), std::string::npos)
            << "Steady-state maintenance must remain graph/event ordered.";

        const size_t rocm_rebalance_begin = rocm_source.find(
            "bool ROCmMoEKernel::runDeviceRebalanceController(");
        const size_t rocm_rebalance_end = rocm_source.find(
            "void ROCmMoEKernel::zeroBuffer(",
            rocm_rebalance_begin);
        ASSERT_NE(rocm_rebalance_begin, std::string::npos);
        ASSERT_NE(rocm_rebalance_end, std::string::npos);
        const std::string rocm_rebalance_body =
            rocm_source.substr(
                rocm_rebalance_begin,
                rocm_rebalance_end - rocm_rebalance_begin);
        EXPECT_NE(
            rocm_rebalance_body.find(
                "static_cast<hipStream_t>(stream)"),
            std::string::npos)
            << "ROCm rebalance profiling must use the immutable launch-context stream.";
        EXPECT_EQ(
            rocm_rebalance_body.find(
                "static_cast<hipStream_t>(getStream())"),
            std::string::npos)
            << "ROCm rebalance profiling must not consult unrelated mutable kernel stream state.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         ProductionMoEGraphsUseExplicitScopedKernelOwners)
    {
        const fs::path root = findRepoRoot();
        const fs::path factory_header_path = root / "src/v2/kernels/KernelFactory.h";
        const fs::path factory_source_path = root / "src/v2/kernels/KernelFactory.cpp";
        const fs::path routing_header_path =
            root / "src/v2/execution/compute_stages/stages/MoERoutingStage.h";
        const fs::path routing_source_path =
            root / "src/v2/execution/compute_stages/stages/MoERoutingStage.cpp";
        const fs::path expert_header_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h";
        const fs::path expert_source_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        const fs::path rebalance_header_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.h";
        const fs::path rebalance_source_path =
            root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.cpp";
        const fs::path graph_source_path =
            root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";

        for (const auto &path : {
                 factory_header_path,
                 factory_source_path,
                 routing_header_path,
                 routing_source_path,
                 expert_header_path,
                 expert_source_path,
                 rebalance_header_path,
                 rebalance_source_path,
                 graph_source_path})
        {
            ASSERT_TRUE(fs::exists(path)) << path;
        }

        const std::string factory_header = readFile(factory_header_path);
        const std::string factory_source = readFile(factory_source_path);
        const std::string routing_header = readFile(routing_header_path);
        const std::string routing_source = readFile(routing_source_path);
        const std::string expert_header = readFile(expert_header_path);
        const std::string expert_source = readFile(expert_source_path);
        const std::string rebalance_header = readFile(rebalance_header_path);
        const std::string rebalance_source = readFile(rebalance_source_path);
        const std::string graph_source = readFile(graph_source_path);

        EXPECT_NE(factory_header.find("std::unique_ptr<llaminar2::IMoEKernel> createMoEKernel"),
                  std::string::npos)
            << "The factory must transfer unique MoE kernel ownership to each stage.";
        for (const std::string &obsolete_singleton_symbol : {
                 "getOrCreateMoEKernel",
                 "MoECacheKey",
                 "moe_cache_",
                 "KernelKind::MOE",
             })
        {
            EXPECT_EQ(factory_header.find(obsolete_singleton_symbol), std::string::npos)
                << obsolete_singleton_symbol;
            EXPECT_EQ(factory_source.find(obsolete_singleton_symbol), std::string::npos)
                << obsolete_singleton_symbol;
        }

        EXPECT_NE(routing_header.find("std::unique_ptr<IMoEKernel> owned_moe_kernel_"),
                  std::string::npos);
        EXPECT_NE(routing_source.find("owned_moe_kernel_ = KernelFactory::createMoEKernel"),
                  std::string::npos);
        EXPECT_NE(routing_header.find(
                      "std::shared_ptr<MoERoutedPipelineKernelOwner> routed_pipeline_kernel_owner"),
                  std::string::npos)
            << "The router must accept an explicit graph-local producer/consumer owner.";

        const std::regex expert_owner_regex(
            "std::unique_ptr<IMoEKernel> owned_moe_kernel_");
        const auto expert_owner_begin =
            std::sregex_iterator(expert_header.begin(), expert_header.end(), expert_owner_regex);
        const auto expert_owner_end = std::sregex_iterator();
        EXPECT_EQ(std::distance(expert_owner_begin, expert_owner_end), 4)
            << "Routed experts, shared FFN, shared gate, and canonical route reduction require separate kernels.";
        EXPECT_NE(expert_source.find("owned_moe_kernel_ = KernelFactory::createMoEKernel"),
                  std::string::npos);
        EXPECT_NE(expert_header.find(
                      "std::shared_ptr<MoERoutedPipelineKernelOwner> routed_pipeline_kernel_owner"),
                  std::string::npos)
            << "The routed expert must consume the same graph-local owner as its router.";

        EXPECT_NE(rebalance_header.find("std::unique_ptr<IMoEKernel> owned_moe_kernel_"),
                  std::string::npos);
        EXPECT_NE(rebalance_source.find("owned_moe_kernel_ = KernelFactory::createMoEKernel"),
                  std::string::npos);
        EXPECT_NE(graph_source.find(
                      "std::make_shared<MoERoutedPipelineKernelOwner>()"),
                  std::string::npos)
            << "Every graph build must create a fresh routed-pipeline owner.";
        EXPECT_NE(graph_source.find(
                      "route_params.routed_pipeline_kernel_owner ="),
                  std::string::npos);
        EXPECT_NE(graph_source.find(
                      "expert_params.routed_pipeline_kernel_owner ="),
                  std::string::npos);
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         GroupedVerifierPublishesHistogramsOnlyAtAcceptedStateCommit)
    {
        const fs::path root = findRepoRoot();
        const fs::path interface_path = root / "src/v2/kernels/IMoEKernel.h";
        const fs::path stage_path =
            root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp";
        const fs::path orchestrator_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        const fs::path cuda_path =
            root / "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu";
        const fs::path rocm_path =
            root / "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip";
        for (const auto &path :
             {interface_path,
              stage_path,
              orchestrator_path,
              cuda_path,
              rocm_path})
        {
            ASSERT_TRUE(fs::exists(path)) << path;
        }

        const std::string interface_source = readFile(interface_path);
        const std::string stage_source = readFile(stage_path);
        const std::string orchestrator_source =
            readFile(orchestrator_path);
        const std::string cuda_source = readFile(cuda_path);
        const std::string rocm_source = readFile(rocm_path);

        EXPECT_EQ(
            interface_source.find("MoEGroupedHistogramUpdate"),
            std::string::npos)
            << "Speculative grouping must not expose a persistent-history "
               "side-effect switch.";
        EXPECT_NE(
            interface_source.find(
                "virtual bool commitGroupedVerifierHistograms("),
            std::string::npos);
        EXPECT_EQ(
            interface_source.find("snapshotGroupedVerifierRoutes"),
            std::string::npos)
            << "Route retention belongs inside the final group/regroup writer; "
               "a standalone snapshot API would restore one launch per layer.";
        EXPECT_EQ(
            stage_source.find("kernel->snapshotGroupedVerifierRoutes("),
            std::string::npos);
        EXPECT_NE(
            stage_source.find("retain_routes_during_initial_grouping"),
            std::string::npos);
        EXPECT_NE(
            stage_source.find(
                "top_k,\n                    retain_routes_for_deferred_commit)"),
            std::string::npos)
            << "Least-loaded assignment must retain only after final regroup.";
        EXPECT_NE(
            stage_source.find(
                "publishCommittedGroupedVerifierHistograms("),
            std::string::npos);
        EXPECT_NE(
            stage_source.find(
                "kernel->commitGroupedVerifierHistograms("),
            std::string::npos);

        const size_t commit_call =
            orchestrator_source.find(
                "publishCommittedMoEVerifierHistograms(",
                orchestrator_source.find(
                    "publishAcceptedMTPSpecStateBatchFromDeviceOutcome("));
        const size_t ready_event =
            orchestrator_source.find(
                "recordAcceptedSpecPublicationReady(",
                commit_call);
        ASSERT_NE(commit_call, std::string::npos);
        ASSERT_NE(ready_event, std::string::npos);
        EXPECT_LT(commit_call, ready_event)
            << "Committed routing history must be ordered before the "
               "accepted-publication ready event.";

        for (const auto &[backend, source] :
             std::array<std::pair<const char *, const std::string *>, 2>{
                 std::pair{"CUDA", &cuda_source},
                 std::pair{"ROCm", &rocm_source}})
        {
            const std::string grouping_kernel =
                backend == std::string("CUDA")
                    ? "prefill_group_cast_count_runtime_kernel("
                    : "rocm_moe_prefill_group_cast_count_kernel(";
            const std::string commit_kernel =
                backend == std::string("CUDA")
                    ? "commit_grouped_verifier_histograms_runtime_kernel("
                    : "rocm_moe_commit_grouped_verifier_histograms_kernel(";
            const size_t grouping_begin =
                source->find(grouping_kernel);
            const size_t commit_begin =
                source->find(commit_kernel);
            ASSERT_NE(grouping_begin, std::string::npos) << backend;
            ASSERT_NE(commit_begin, std::string::npos) << backend;
            ASSERT_LT(grouping_begin, commit_begin) << backend;
            const std::string speculative_grouping =
                source->substr(
                    grouping_begin,
                    commit_begin - grouping_begin);
            EXPECT_EQ(
                speculative_grouping.find("decode_histogram["),
                std::string::npos)
                << backend
                << " speculative grouping mutated selected-route history.";
            EXPECT_EQ(
                speculative_grouping.find("decode_local_histogram["),
                std::string::npos)
                << backend
                << " speculative grouping mutated local-route history.";
            EXPECT_NE(
                speculative_grouping.find(
                    "deferred_verifier_route_expert_ids[slot] ="),
                std::string::npos)
                << backend
                << " final route writers must fuse per-layer ledger retention.";
            EXPECT_NE(
                speculative_grouping.find(
                    "deferred_verifier_route_participant_ids[slot] ="),
                std::string::npos)
                << backend;
            EXPECT_EQ(
                source->find("snapshot_grouped_verifier_routes"),
                std::string::npos)
                << backend
                << " must not retain a standalone per-layer snapshot launch.";

            const std::string committed_publication =
                source->substr(commit_begin, 5000);
            EXPECT_NE(
                committed_publication.find(
                    "accepted_state_counts[request]"),
                std::string::npos)
                << backend
                << " commit must consume device-owned accepted prefixes.";
            EXPECT_NE(
                committed_publication.find(
                    "decode_histogram[expert_id]"),
                std::string::npos)
                << backend;
            EXPECT_NE(
                committed_publication.find(
                    "decode_local_histogram[expert_id]"),
                std::string::npos)
                << backend;
        }
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
        EXPECT_EQ(graph.find("localPayloadDescriptorResolverForRuntimeTable"),
                  std::string::npos)
            << "Portable restore must not resurrect rolling transfer-slot "
               "payloads through a resolver after those bytes may be reused.";
        EXPECT_NE(graph.find("Portable version 3 contains pointer-free logical placement"),
                  std::string::npos)
            << "The graph must document the durable placement boundary enforced "
               "by portable runtime version 3.";
        EXPECT_NE(graph.find("resetPrefixCacheRuntimeStateWithoutSnapshot"),
                  std::string::npos)
            << "MoE prefix restore without a runtime payload must have its own "
               "model-runtime reset boundary instead of reusing request reset.";
        const size_t prefix_reset_start =
            graph.find(
                "void Qwen35MoEGraph::resetPrefixCacheRuntimeStateWithoutSnapshot(\n"
                "        void *execution_stream)");
        ASSERT_NE(prefix_reset_start, std::string::npos);
        const size_t prefix_reset_end =
            graph.find("ILocalTPContext *Qwen35MoEGraph::maintenanceTPContextForDomain",
                       prefix_reset_start);
        ASSERT_NE(prefix_reset_end, std::string::npos);
        const std::string prefix_reset_body =
            graph.substr(prefix_reset_start, prefix_reset_end - prefix_reset_start);
        EXPECT_NE(prefix_reset_body.find("restoreInitialRuntimeState(execution_stream)"),
                  std::string::npos)
            << "No-payload prefix restore must republish the immutable "
               "model-lifetime MoE state on the reset transaction stream.";
        EXPECT_EQ(prefix_reset_body.find("resetDecodeRuntimeState();"),
                  std::string::npos)
            << "GPU prefix reset must never select an implicit stream.";
        EXPECT_EQ(prefix_reset_body.find("resetDecodeRuntimeState(execution_stream)"),
                  std::string::npos)
            << "An empty runtime table is not a valid prefix-restored model "
               "state and must not replace the immutable initial placement.";
        EXPECT_NE(prefix_reset_body.find("resetRequestPublications(execution_stream)"),
                  std::string::npos)
            << "Prefix reset must retire every transient transfer-slot "
               "publication on the same transaction stream.";
        EXPECT_EQ(prefix_reset_body.find("moe_graph_rebalance_bindings_.clear()"),
                  std::string::npos)
            << "Graph stages retain binding pointers across replay-executable "
               "replacement, so prefix reset must preserve their owners.";
        EXPECT_EQ(prefix_reset_body.find("moe_rebalance_transfer_states_.clear()"),
                  std::string::npos)
            << "Prefix reset must preserve graph-owned streams and events while "
               "runtime-table reset clears request-local movement claims.";
        EXPECT_EQ(prefix_reset_body.find("moe_transfer_slot_directories_.clear()"),
                  std::string::npos)
            << "Prefix reset must preserve transfer-directory pointer identity; "
               "portable state never claims the transient bytes in its slots.";
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

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         FluentFullGraphBuildUsesPublicPolicyBoundary)
    {
        const fs::path root = findRepoRoot();
        const fs::path orchestrator_path =
            root /
            "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(orchestrator_path)) << orchestrator_path;

        const std::string orchestrator = readFile(orchestrator_path);
        ASSERT_FALSE(orchestrator.empty()) << orchestrator_path;
        const size_t full_build_start = orchestrator.find(
            "DeviceGraphOrchestrator::GraphBuildSession::buildForward()");
        ASSERT_NE(full_build_start, std::string::npos);
        const size_t partial_build_start = orchestrator.find(
            "DeviceGraphOrchestrator::GraphBuildSession::buildPartial()",
            full_build_start);
        ASSERT_NE(partial_build_start, std::string::npos);
        const std::string full_build = orchestrator.substr(
            full_build_start,
            partial_build_start - full_build_start);

        EXPECT_NE(
            full_build.find(
                "graph_builder->buildForwardGraph(prepared_input, output)"),
            std::string::npos)
            << "The fluent full-graph session must enter through the public "
               "virtual builder contract so typed topology policies are adopted.";
        EXPECT_EQ(
            full_build.find(
                "graph_builder->buildFullForwardGraph(prepared_input, output)"),
            std::string::npos)
            << "Calling the model-internal full builder directly bypasses "
               "policy scopes such as prefix-runtime device rehydration.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         PrefixRuntimeRehydrationHasOneModelOwnedLifecycle)
    {
        const fs::path root = findRepoRoot();
        const fs::path orchestrator_source_path =
            root /
            "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        const fs::path orchestrator_header_path =
            root /
            "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h";
        ASSERT_TRUE(fs::exists(orchestrator_source_path))
            << orchestrator_source_path;
        ASSERT_TRUE(fs::exists(orchestrator_header_path))
            << orchestrator_header_path;

        const std::string source = readFile(orchestrator_source_path);
        const std::string header = readFile(orchestrator_header_path);
        ASSERT_FALSE(source.empty()) << orchestrator_source_path;
        ASSERT_FALSE(header.empty()) << orchestrator_header_path;

        EXPECT_EQ(
            header.find("prefix_runtime_device_rehydration_pending_"),
            std::string::npos)
            << "The model graph owns the pending prefix-runtime transaction; "
               "an orchestrator mirror becomes stale when a full prefix hit "
               "returns terminal logits without launching a main graph.";
        EXPECT_NE(
            source.find(
                "prefixCacheRuntimeStateRequiresDeviceRehydration();"),
            std::string::npos)
            << "Forward graph policy must be read from the model-runtime owner.";
        EXPECT_NE(
            source.find(
                "completePrefixCacheRuntimeStateDeviceRehydration();"),
            std::string::npos)
            << "A successful rehydration graph must retire that same "
               "model-runtime transaction.";
        EXPECT_EQ(
            source.find(
                "prefix-runtime rehydration policy disagrees with the model runtime owner"),
            std::string::npos)
            << "A duplicate-owner consistency check is not a lifecycle; make "
               "the duplicate state structurally impossible instead.";
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
        EXPECT_NE(
            manifest_publish.find(
                "snapshot_manifest.stage_copies.find(stage_name)"),
            std::string::npos)
            << "GPU publication must resolve the immutable graph-owned per-stage captured-slot manifest.";
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
        EXPECT_NE(
            copy_body.find(
                "snapshot_manifest.outputless_stages.contains(node.name)"),
            std::string::npos)
            << "Warmup-observed outputless stages need an explicit finalized manifest state.";
        EXPECT_NE(copy_body.find("lost all tensor-backed outputs during graph capture"),
                  std::string::npos)
            << "A producer that loses warmed outputs during capture must hard-fail.";
        EXPECT_NE(copy_body.find("copy.descriptor_finalized = true"),
                  std::string::npos)
            << "A successful point-in-time D2D copy must finalize its production descriptor.";
        EXPECT_NE(copy_body.find("FP32Tensor::createMapped("),
                  std::string::npos)
            << "Graph snapshots must preserve stage values in mapped host slots, "
               "not duplicate every selected output in scarce GPU memory.";
        EXPECT_NE(copy_body.find("copy.storage->isMapped()"),
                  std::string::npos)
            << "Mapped snapshot placement is a required contract, not a hint.";
        EXPECT_EQ(copy_body.find("std::make_unique<FP32Tensor>("),
                  std::string::npos)
            << "Graph snapshot manifests must not allocate ordinary device tensors.";
        EXPECT_EQ(copy_body.find("copy.storage->allocateOnDevice("),
                  std::string::npos)
            << "Graph snapshot recording must not lazily allocate HBM storage.";
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
        EXPECT_NE(
            capture_event_only.find(
                "TransferEngine::publishGraphOwnedDeviceWrite("),
                  std::string::npos)
            << "Capture-time snapshot copies should update coherence flags.";
        EXPECT_EQ(capture_event_only.find("TransferEngine::publishDeviceWrite("),
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
        EXPECT_NE(fast_collective_body.find(
                      "if (ok && config_.snapshot_callback)"),
                  std::string::npos)
            << "Fast collectives must always record a graph-stable D2D snapshot when snapshotting is enabled.";
        EXPECT_EQ(fast_collective_body.find(
                      "config_.snapshot_callback && !policy.snapshot_callback"),
                  std::string::npos)
            << "Snapshot-copy recording must not depend on whether immediate host publication is deferred.";
        EXPECT_NE(fast_collective_body.find("captureGraphSnapshotCopies"),
                  std::string::npos)
            << "Post-collective graph snapshots such as *_ALLREDUCED must record a D2D copy before the fast path returns.";
        EXPECT_NE(fast_collective_body.find("publishGraphSnapshotCopies"),
                  std::string::npos)
            << "Eager fast collectives must publish the immutable snapshot slot after its D2D copy completes.";
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

    /**
     * @brief Prevent graph capture from erasing unattributed CUDA/HIP failures.
     *
     * A sticky backend error is evidence that an earlier asynchronous launch
     * or runtime call failed. Clearing it before capture or instantiation turns
     * a broken ownership timeline into an apparently clean retry. The graph
     * system instead relies on checked stream/capture operations and treats an
     * explicit graph-update failure differently from the non-error
     * NeedsReinstantiate result.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, GraphCaptureNeverClearsBackendErrors)
    {
        const fs::path root = findRepoRoot();
        const std::array<fs::path, 8> production_files = {
            root / "src/v2/backends/IWorkerGPUContext.h",
            root / "src/v2/backends/cuda/NvidiaDeviceContext.h",
            root / "src/v2/backends/cuda/NvidiaDeviceContext.cu",
            root / "src/v2/backends/rocm/AMDDeviceContext.h",
            root / "src/v2/backends/rocm/AMDDeviceContext.cpp",
            root / "src/v2/execution/local_execution/graph/GraphCaptureGuard.h",
            root / "src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp",
            root / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp",
        };

        for (const auto &path : production_files)
        {
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string source = readFile(path);
            EXPECT_EQ(source.find("clearLastError"), std::string::npos)
                << fs::relative(path, root)
                << " must propagate checked backend failures rather than erase them";
        }

        const std::string controller = readFile(
            root / "src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp");
        const std::string executor = readFile(
            root / "src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp");
        const std::string interface_source = readFile(
            root / "src/v2/backends/IGPUGraphCapture.h");
        const std::string cuda_capture = readFile(
            root / "src/v2/backends/cuda/CUDAGraphCapture.h");
        const std::string rocm_capture = readFile(
            root / "src/v2/backends/rocm/HIPGraphCapture.h");

        EXPECT_NE(
            interface_source.find("supportsExecutableUpdate() const noexcept"),
            std::string::npos)
            << "Executable-update support must be an explicit graph-backend capability.";
        EXPECT_NE(
            cuda_capture.find("supportsExecutableUpdate() const noexcept override { return true; }"),
            std::string::npos);
        EXPECT_NE(
            rocm_capture.find("supportsExecutableUpdate() const noexcept override { return false; }"),
            std::string::npos);
        EXPECT_NE(controller.find("supportsExecutableUpdate()"), std::string::npos);
        EXPECT_NE(executor.find("supportsExecutableUpdate()"), std::string::npos);
        EXPECT_EQ(
            controller.find("deviceId().is_rocm()"),
            std::string::npos)
            << "Graph executable publication must use the backend capability, "
               "not a device-type special case.";

        const size_t failed_update =
            controller.find("if (update_result == GraphUpdateResult::Failed)");
        const size_t reinstantiate =
            controller.find(
                "update_result == GraphUpdateResult::NeedsReinstantiate",
                failed_update);
        ASSERT_NE(failed_update, std::string::npos);
        ASSERT_NE(reinstantiate, std::string::npos);
        EXPECT_LT(failed_update, reinstantiate)
            << "A failed graph update must stop before the non-error re-instantiation path.";
        EXPECT_NE(
            controller.find(
                "refusing to discard the backend error",
                failed_update),
            std::string::npos);
    }

    /**
     * @brief Keep graph lifecycle attribution attached to physical cache identity.
     *
     * Logical MTP operations may reuse one sidecar graph when their typed role,
     * stable device slots, geometry, and workspace bindings match. Relabeling
     * the cache on every invocation makes a capture appear under one context
     * and its replay under another, obscuring whether a real executable exists.
     */
    TEST(Test__MoEGraphNative_ForbiddenDependencyScan,
         MTPSidecarCaptureIdentityIsImmutableAcrossLogicalInvocations)
    {
        const fs::path root = findRepoRoot();
        const fs::path header_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h";
        const fs::path source_path =
            root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp";
        ASSERT_TRUE(fs::exists(header_path));
        ASSERT_TRUE(fs::exists(source_path));

        const std::string header = readFile(header_path);
        const std::string source = readFile(source_path);
        EXPECT_NE(header.find("std::string capture_perf_context;"), std::string::npos);
        EXPECT_NE(
            source.find(
                "sidecar_cache.capture_perf_context = sidecar_context;"),
            std::string::npos);
        EXPECT_NE(
            source.find(
                "sidecar_cache.segment_cache.perf_context =\n"
                "                    sidecar_cache.capture_perf_context;"),
            std::string::npos);
        EXPECT_EQ(
            source.find(
                "sidecar_cache.segment_cache.perf_context = sidecar_context;"),
            std::string::npos)
            << "A logical invocation context must never relabel an existing physical graph.";
        EXPECT_NE(source.find("{\"invocation_context\", sidecar_context}"),
                  std::string::npos);
        EXPECT_NE(
            source.find(
                "{\"graph_context\", sidecar_cache.capture_perf_context}"),
            std::string::npos);
    }

} // namespace llaminar2::test
