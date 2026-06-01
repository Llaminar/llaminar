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
            "src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.h",
            "src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.cpp",
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

        const std::string weighted_add_body = functionBody(
            "void ROCmMoEKernel::weightedAddFromTensors",
            "int ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable");
        EXPECT_NE(weighted_add_body.find("output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE"),
                  std::string::npos);
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
            {"bool ROCmMoEKernel::groupTokensByExpertDevice", "void ROCmMoEKernel::ensureStagingCapacity"},
            {"void ROCmMoEKernel::zeroBuffer", "void ROCmMoEKernel::gatherTokenBatchFromTensors"},
            {"void ROCmMoEKernel::gatherTokenBatchFromTensors", "void ROCmMoEKernel::scatterAddWeightedFromTensors"},
            {"void ROCmMoEKernel::scatterAddWeightedFromTensors", "void ROCmMoEKernel::sharedExpertGateFromTensors"},
            {"void ROCmMoEKernel::sharedExpertGateFromTensors", "void ROCmMoEKernel::swiGLUFromTensors"},
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

        EXPECT_NE(local_contents.find("std::fill_n(routing_indices, input.live_row_count * static_cast<size_t>(params_.top_k), -1.0f)"),
                  std::string::npos);
        EXPECT_EQ(local_contents.find("compact_output_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE"),
                  std::string::npos);
        EXPECT_NE(compute_contents.find("expert_id < 0 || expert_id >= num_experts || weight == 0.0f"),
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
        EXPECT_NE(graph_contents.find("return_params.dense_output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;"),
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
        const size_t dgo_prefill_sync = dgo_contents.find("synchronizeGpuBackendsBeforeMmapRelease()", dgo_prefill_start);
        const size_t dgo_prefill_advise = dgo_contents.find("weight_manager_->adviseMmapDontneed()", dgo_prefill_start);
        ASSERT_NE(dgo_prefill_sync, std::string::npos);
        ASSERT_NE(dgo_prefill_advise, std::string::npos);
        EXPECT_LT(dgo_prefill_sync, dgo_prefill_advise);

        const size_t rank_release = rank_contents.find("releaseHostResidentWeightData();");
        const size_t rank_sync = rank_contents.find("synchronizeGpuBackendsBeforeRankMmapRelease()", rank_release);
        const size_t rank_advise = rank_contents.find("wm->adviseMmapDontneed()", rank_release);
        ASSERT_NE(rank_release, std::string::npos);
        ASSERT_NE(rank_sync, std::string::npos);
        ASSERT_NE(rank_advise, std::string::npos);
        EXPECT_LT(rank_release, rank_sync);
        EXPECT_LT(rank_sync, rank_advise);
    }

} // namespace llaminar2::test
