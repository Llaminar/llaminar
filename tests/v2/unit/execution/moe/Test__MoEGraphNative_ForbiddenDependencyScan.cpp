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
            EXPECT_NE(body.find("if (!ordered_scatter_overwrites_output)"),
                      std::string::npos)
                << name << " must only pre-zero output for the atomic scatter fallback";
        }

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

        const size_t forward_mtp = forward_body.find("populateMTPShiftedCacheFromPrefill(tokens, seq_len, batch_size");
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

        const size_t chunk_mtp = chunk_body.find("populateMTPShiftedCacheFromPrefill(tokens, seq_len, 1");
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

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, Qwen35MoEResetPreservesRuntimePlacementBanks)
    {
        const fs::path root = findRepoRoot();
        const fs::path graph_path = root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp";
        ASSERT_TRUE(fs::exists(graph_path)) << graph_path;

        const std::string contents = readFile(graph_path);
        ASSERT_FALSE(contents.empty()) << graph_path;

        const size_t reset_start = contents.find("void Qwen35MoEGraph::resetState()");
        ASSERT_NE(reset_start, std::string::npos);
        const size_t reset_end = contents.find("void Qwen35MoEGraph::appendPrefixCacheFingerprintMaterial",
                                               reset_start);
        ASSERT_NE(reset_end, std::string::npos);
        const std::string reset_body = contents.substr(reset_start, reset_end - reset_start);

        EXPECT_NE(reset_body.find("resetDecodeHistogramCounts()"), std::string::npos)
            << "Session reset should clear per-session decode counters without "
               "dropping persistent device placement metadata.";
        EXPECT_EQ(reset_body.find("resetDecodeRuntimeState()"), std::string::npos)
            << "Clearing runtime placement banks during session reset makes "
               "the next GPU decode route fall back to host/top-k state or fail "
               "before graph-captured MoE decode can run.";
    }

    TEST(Test__MoEGraphNative_ForbiddenDependencyScan, ROCmMoEResetPreservesDeclaredGroupedWorkspace)
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

        EXPECT_EQ(reset_body.find("clearWorkspaceScratchBindings()"), std::string::npos)
            << "Session reset must not drop graph-owned ROCm MoE workspace bindings. "
               "Captured verifier graphs replay stable scratch addresses across requests.";
        EXPECT_EQ(reset_body.find("d_grouped_gate_ptrs_"), std::string::npos)
            << "Grouped pointer arrays are declared workspace, not per-session "
               "routing state. Resetting them here can make the next verifier "
               "replay use missing graph-captured addresses.";
        EXPECT_EQ(reset_body.find("d_grouped_swiglu_int8_"), std::string::npos)
            << "Grouped SwiGLU scratch is declared workspace and must keep stable "
               "addresses across request/session reset.";
        EXPECT_EQ(reset_body.find("grouped_decode_active_cap_"), std::string::npos)
            << "Grouped decode capacity metadata belongs to workspace binding "
               "lifetime, not request/session reset lifetime.";

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
            std::vector<std::string> representative_scratch_members;
            std::vector<std::string> required_clear_members;
        };

        const std::vector<BackendCase> cases = {
            {
                "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp",
                "CUDAMoEKernel",
                {
                    "d_grouped_gateup_gate_partials_",
                    "d_grouped_down_partials_",
                    "d_decode_swiglu_int8_",
                },
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
                    "d_grouped_gate_ptrs_",
                    "d_grouped_swiglu_int8_",
                    "grouped_decode_active_cap_",
                },
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

            EXPECT_EQ(reset_body.find("clearWorkspaceScratchBindings()"), std::string::npos)
                << "Dynamic reset is request/session metadata cleanup only; it must not "
                   "drop graph-owned scratch bindings.";
            for (const std::string &member : backend.representative_scratch_members)
            {
                EXPECT_EQ(reset_body.find(member), std::string::npos)
                    << "Dynamic reset must not directly mutate workspace-owned scratch member "
                    << member;
            }

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
            EXPECT_NE(clear_body.find("scratch_workspace_bound_ = false"), std::string::npos);
        }
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

} // namespace llaminar2::test
