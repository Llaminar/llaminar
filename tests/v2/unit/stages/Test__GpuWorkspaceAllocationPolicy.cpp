/**
 * @file Test__GpuWorkspaceAllocationPolicy.cpp
 * @brief Source-level guards for capture-sensitive GPU workspace allocation policy.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEWorkspaceRequirements.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
    std::filesystem::path repoRoot()
    {
#ifdef LLAMINAR_REPO_ROOT
        return std::filesystem::path(LLAMINAR_REPO_ROOT);
#else
        std::filesystem::path path = std::filesystem::current_path();
        while (!path.empty())
        {
            if (std::filesystem::exists(path / "src/v2") &&
                std::filesystem::exists(path / "tests/v2"))
            {
                return path;
            }
            path = path.parent_path();
        }
        return std::filesystem::current_path();
#endif
    }

    std::string readFile(const std::filesystem::path &path)
    {
        std::ifstream input(path);
        EXPECT_TRUE(input.good()) << "Could not open " << path;
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }

    std::string sliceBetween(const std::string &source,
                             const std::string &begin_marker,
                             const std::string &end_marker)
    {
        const size_t begin = source.find(begin_marker);
        EXPECT_NE(begin, std::string::npos) << "Missing begin marker: " << begin_marker;
        if (begin == std::string::npos)
        {
            return {};
        }
        const size_t end = source.find(end_marker, begin);
        EXPECT_NE(end, std::string::npos) << "Missing end marker: " << end_marker;
        if (end == std::string::npos)
        {
            return source.substr(begin);
        }
        return source.substr(begin, end - begin);
    }

    void expectNoRawGpuAllocationCalls(const std::string &source, const std::string &label)
    {
        const std::vector<std::string> forbidden = {
            "cudaMalloc(",
            "cudaMallocAsync(",
            "cudaFree(",
            "cudaFreeAsync(",
            "hipMalloc(",
            "hipMallocAsync(",
            "hipFree(",
            "hipFreeAsync(",
        };

        for (const auto &needle : forbidden)
        {
            EXPECT_EQ(source.find(needle), std::string::npos)
                << label << " must use workspace/backend-owned buffers, not " << needle;
        }
    }

    bool hasRawGpuAllocationCall(const std::string &source)
    {
        const std::vector<std::string> needles = {
            "cudaMalloc(",
            "cudaMallocAsync(",
            "hipMalloc(",
            "hipMallocAsync(",
        };
        for (const auto &needle : needles)
        {
            if (source.find(needle) != std::string::npos)
                return true;
        }
        return false;
    }

    std::string stripCommentsAndStringLiterals(const std::string &source)
    {
        std::string stripped;
        stripped.reserve(source.size());

        enum class State
        {
            Normal,
            LineComment,
            BlockComment,
            StringLiteral,
            CharLiteral,
        };

        State state = State::Normal;
        bool escaped = false;
        for (size_t i = 0; i < source.size(); ++i)
        {
            const char c = source[i];
            const char next = (i + 1 < source.size()) ? source[i + 1] : '\0';

            switch (state)
            {
            case State::Normal:
                if (c == '/' && next == '/')
                {
                    stripped.push_back(' ');
                    stripped.push_back(' ');
                    ++i;
                    state = State::LineComment;
                }
                else if (c == '/' && next == '*')
                {
                    stripped.push_back(' ');
                    stripped.push_back(' ');
                    ++i;
                    state = State::BlockComment;
                }
                else if (c == '"')
                {
                    stripped.push_back(' ');
                    state = State::StringLiteral;
                    escaped = false;
                }
                else if (c == '\'')
                {
                    stripped.push_back(' ');
                    state = State::CharLiteral;
                    escaped = false;
                }
                else
                {
                    stripped.push_back(c);
                }
                break;

            case State::LineComment:
                stripped.push_back(c == '\n' ? '\n' : ' ');
                if (c == '\n')
                    state = State::Normal;
                break;

            case State::BlockComment:
                if (c == '*' && next == '/')
                {
                    stripped.push_back(' ');
                    stripped.push_back(' ');
                    ++i;
                    state = State::Normal;
                }
                else
                {
                    stripped.push_back(c == '\n' ? '\n' : ' ');
                }
                break;

            case State::StringLiteral:
                stripped.push_back(c == '\n' ? '\n' : ' ');
                if (escaped)
                {
                    escaped = false;
                }
                else if (c == '\\')
                {
                    escaped = true;
                }
                else if (c == '"')
                {
                    state = State::Normal;
                }
                break;

            case State::CharLiteral:
                stripped.push_back(c == '\n' ? '\n' : ' ');
                if (escaped)
                {
                    escaped = false;
                }
                else if (c == '\\')
                {
                    escaped = true;
                }
                else if (c == '\'')
                {
                    state = State::Normal;
                }
                break;
            }
        }

        return stripped;
    }

    bool hasExecutableEnsureOnDeviceCall(const std::string &source)
    {
        const auto executable_source = stripCommentsAndStringLiterals(source);
        return executable_source.find("ensureOnDevice(") != std::string::npos ||
               executable_source.find("ensureOnDevice (") != std::string::npos;
    }

    bool hasNullStreamCudaKernelProfileScope(const std::string &source)
    {
        const auto executable_source = stripCommentsAndStringLiterals(source);
        return executable_source.find("CUDA_KERNEL_PROFILE_SCOPE(") != std::string::npos ||
               executable_source.find("CUDA_KERNEL_PROFILE_SCOPE (") != std::string::npos;
    }

    bool hasUncheckedSynchronizeStreamCall(const std::string &source)
    {
        const auto executable_source = stripCommentsAndStringLiterals(source);
        return executable_source.find("->synchronizeStream(") != std::string::npos ||
               executable_source.find("->synchronizeStream (") != std::string::npos;
    }

    bool isSourceFile(const std::filesystem::path &path)
    {
        const auto extension = path.extension().string();
        return extension == ".cpp" || extension == ".cu" || extension == ".cuh" ||
               extension == ".h" || extension == ".hpp";
    }

    bool isPhase138HygieneScannedFile(const std::filesystem::path &path)
    {
        if (isSourceFile(path))
            return true;
        const auto extension = path.extension().string();
        const auto filename = path.filename().string();
        return extension == ".md" || extension == ".cmake" ||
               filename == "CMakeLists.txt";
    }
} // namespace

TEST(Test__GpuWorkspaceAllocationPolicy, RawGpuMallocCallsStayInSanctionedSourceOwners)
{
    const auto root = repoRoot();
    const std::unordered_set<std::string> sanctioned = {
        "src/v2/backends/ComputeBackend.cpp",
        "src/v2/backends/IBackend.h",
        "src/v2/backends/benchmarks/CUDABenchmark.cu",
        "src/v2/backends/benchmarks/ROCmBenchmark.cpp",
        "src/v2/backends/cuda/CUDABackend.cu",
        "src/v2/backends/cuda/CUDATensorValidation.cu",
        "src/v2/backends/rocm/ROCmBackend.cpp",
        "src/v2/backends/rocm/ROCmTensorValidation.cpp",
        "src/v2/collective/backends/NCCLBackendCUDA.cu",
        "src/v2/collective/backends/RCCLBackendHIP.cpp",
        "src/v2/collective/coordinators/RCCLCoordinator.cpp",
        "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNetKernels.cu",
        "src/v2/kernels/cuda/gemm/CUDABatchGemmOps.cu",
        "src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu",
        "src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel_CUTLASS.cu",
        "src/v2/kernels/cuda/gemm/CUDAcuBLASQuantGemm.cu",
        "src/v2/kernels/cuda/gemm/CuBLASGemmKernel.cu",
        "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu",
        "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.cpp",
        "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu",
        "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTensorAdapter.cpp",
        "src/v2/kernels/cuda/kvcache/CUDATurboQuantKernels.cu",
        "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp",
        "src/v2/kernels/cuda/ops/CUDACastKernels.cu",
        "src/v2/kernels/cuda/ops/CUDARoPEKernels.cu",
        "src/v2/kernels/cuda/ops/CUDARowSelectKernels.cu",
        "src/v2/kernels/rocm/ROCmWeightPacker.cpp",
        "src/v2/kernels/rocm/gemm/HipBLASGemmKernel.cpp",
        "src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp",
        "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp",
        "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp",
        "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp",
        "src/v2/kernels/rocm/ops/ROCmEmbeddingKernelT.cpp",
    };

    std::vector<std::string> failures;
    const auto source_root = root / "src/v2";
    for (const auto &entry : std::filesystem::recursive_directory_iterator(source_root))
    {
        if (!entry.is_regular_file() || !isSourceFile(entry.path()))
            continue;
        const auto relative = std::filesystem::relative(entry.path(), root).generic_string();
        const auto source = readFile(entry.path());
        if (!hasRawGpuAllocationCall(source))
            continue;
        if (sanctioned.find(relative) == sanctioned.end())
            failures.push_back(relative);
    }

    EXPECT_TRUE(failures.empty()) << [&]
    {
        std::ostringstream out;
        out << "Raw cudaMalloc/hipMalloc calls must stay in sanctioned low-level allocation owners. "
               "Use DeviceWorkspaceManager/IWorkspaceConsumer or backend allocation APIs instead.\n";
        for (const auto &failure : failures)
            out << failure << '\n';
        return out.str();
    }();
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoEWorkspaceActiveExpertIdsCoversAllExperts)
{
    const int max_seq_len = 9;
    const int d_model = 2048;
    const int intermediate = 512;
    const int num_experts = 256;
    const int top_k = 8;

    const auto reqs = llaminar2::MoEWorkspaceBuffers::cudaMoE(
        max_seq_len, d_model, intermediate, num_experts, top_k);
    const auto *active_ids = reqs.find(llaminar2::MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS);
    ASSERT_NE(active_ids, nullptr);
    EXPECT_GE(active_ids->size_bytes, static_cast<size_t>(num_experts) * sizeof(int));
    EXPECT_GT(static_cast<size_t>(num_experts), static_cast<size_t>(max_seq_len) * top_k)
        << "fixture must cover the small-token, many-expert regression";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDAKernelProfilingScopesUseExplicitStreams)
{
    const auto root = repoRoot();
    std::vector<std::string> failures;
    const std::vector<std::filesystem::path> scan_roots = {
        root / "src/v2/kernels/cuda",
        root / "src/v2/backends/cuda",
    };

    for (const auto &scan_root : scan_roots)
    {
        for (const auto &entry : std::filesystem::recursive_directory_iterator(scan_root))
        {
            if (!entry.is_regular_file() || !isSourceFile(entry.path()))
                continue;
            const auto source = readFile(entry.path());
            if (!hasNullStreamCudaKernelProfileScope(source))
                continue;
            failures.push_back(std::filesystem::relative(entry.path(), root).generic_string());
        }
    }

    EXPECT_TRUE(failures.empty()) << [&]
    {
        std::ostringstream out;
        out << "CUDA kernel profiling scopes must use CUDA_KERNEL_PROFILE_SCOPE_STREAM(..., stream). "
               "The null-stream macro records events on CUDA's legacy default stream and can race "
               "graph-captured stage streams.\n";
        for (const auto &failure : failures)
            out << failure << '\n';
        return out.str();
    }();
}

TEST(Test__GpuWorkspaceAllocationPolicy, PerfStatsExportDoesNotEnableKernelTimingGate)
{
    const auto source = readFile(repoRoot() / "src/v2/utils/KernelProfiler.h");
    const auto kernel_profiler = sliceBetween(source, "class KernelProfiler", "static void setCurrentDevice");

    EXPECT_NE(kernel_profiler.find("PerfStatsCollector::gpuStageEventTimingEnabled()"), std::string::npos)
        << "KernelProfiler::isEnabled() must use the explicit GPU timing gate.";
    EXPECT_EQ(kernel_profiler.find("PerfStatsCollector::isEnabled()"), std::string::npos)
        << "Generic perf JSON/CSV export must stay passive and must not enable kernel/forward timing.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GraphCaptureControllerChecksStreamSynchronizeFailures)
{
    const auto source = readFile(repoRoot() / "src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp");

    EXPECT_EQ(hasUncheckedSynchronizeStreamCall(source), false)
        << "Graph replay/capture synchronization must use synchronizeStreamChecked() so async "
           "GPU failures are attributed to the graph segment that surfaced them.";
    EXPECT_NE(source.find("synchronizeStreamChecked("), std::string::npos);
    EXPECT_NE(source.find("Initial captured launch stream sync failed after segment starting at"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDARingKVCacheGatherHasNoRawAllocationFallback)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto gather_body = sliceBetween(
        source,
        "bool CUDARingKVCache<Precision>::launch_gather_kernel(",
        "// =========================================================================\n    // IWorkspaceConsumer Implementation");

    expectNoRawGpuAllocationCalls(gather_body, "CUDARingKVCache::launch_gather_kernel");
    EXPECT_NE(gather_body.find("Workspace is required for batched gather"), std::string::npos);
    EXPECT_NE(gather_body.find("Missing required workspace buffers"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDANativeVNNIWorkspaceCriticalFilesHaveNoRawAllocations)
{
    const std::vector<std::filesystem::path> files = {
        "src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillKernels.cu",
        "src/v2/kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh",
    };

    for (const auto &relative : files)
    {
        const auto source = readFile(repoRoot() / relative);
        expectNoRawGpuAllocationCalls(source, relative.string());
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDAMoEExecutionScratchUsesWorkspace)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto route_scratch = sliceBetween(
        source,
        "bool CUDAMoEKernel::ensureStagingCapacity(",
        "bool CUDAMoEKernel::ensureGroupingBufferCapacity(");
    const auto grouped_scratch = sliceBetween(
        source,
        "bool CUDAMoEKernel::ensureGroupingBufferCapacity(",
        "bool CUDAMoEKernel::ensureRuntimeGateUpPointerArrays(");

    expectNoRawGpuAllocationCalls(route_scratch, "CUDAMoEKernel route/staging scratch");
    expectNoRawGpuAllocationCalls(grouped_scratch, "CUDAMoEKernel grouped execution scratch");
    EXPECT_NE(route_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(grouped_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(source.find("requires graph-owned MoE workspace"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, ROCmMoEExecutionScratchUsesWorkspace)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp");
    const auto route_scratch = sliceBetween(
        source,
        "bool ROCmMoEKernel::ensureSharedGateScratchCapacity(",
        "const ROCmMoEKernel::RouterQ8GateCacheEntry *ROCmMoEKernel::getOrCreateQ8RouterGateCache(");
    const auto device_grouping_scratch = sliceBetween(
        source,
        "bool ROCmMoEKernel::groupTokensByExpertDevice(",
        "    // =========================================================================\n    // Tensor-aware API overrides");
    const auto decode_scratch = sliceBetween(
        source,
        "bool ROCmMoEKernel::ensureStagingCapacity(",
        "bool ROCmMoEKernel::ensureGroupedDecodeCapacity(");
    const auto sync_grouping_scratch = sliceBetween(
        source,
        "bool ROCmMoEKernel::prepareExpertGroups(",
        "int ROCmMoEKernel::getExpertTokenCount(");
    const auto async_grouping_scratch = sliceBetween(
        source,
        "bool ROCmMoEKernel::prepareExpertGroupsAsync(",
        "bool ROCmMoEKernel::executeGroupedPrefillPipeline(");

    expectNoRawGpuAllocationCalls(route_scratch, "ROCmMoEKernel route scratch");
    expectNoRawGpuAllocationCalls(device_grouping_scratch, "ROCmMoEKernel device grouping scratch");
    expectNoRawGpuAllocationCalls(decode_scratch, "ROCmMoEKernel decode scratch");
    expectNoRawGpuAllocationCalls(sync_grouping_scratch, "ROCmMoEKernel synchronous grouping scratch");
    expectNoRawGpuAllocationCalls(async_grouping_scratch, "ROCmMoEKernel asynchronous grouping/prefill scratch");

    EXPECT_NE(route_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(device_grouping_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(decode_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(sync_grouping_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(async_grouping_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(source.find("requires graph-owned MoE workspace"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, GDNDeinterleaveScratchUsesBoundWorkspace)
{
    const auto cuda_source = readFile(repoRoot() / "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNet.h");
    const auto cuda_deinterleave = sliceBetween(
        cuda_source,
        "bool deinterleave_qkv_device(",
        "        // =====================================================================\n        // IWorkspaceConsumer Interface");
    expectNoRawGpuAllocationCalls(cuda_deinterleave, "CUDAGatedDeltaNet deinterleave scratch");
    EXPECT_NE(cuda_deinterleave.find("requires bound graph workspace"), std::string::npos);

    const auto rocm_source = readFile(repoRoot() / "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h");
    const auto rocm_deinterleave = sliceBetween(
        rocm_source,
        "bool deinterleave_qkv_device(",
        "    private:");
    expectNoRawGpuAllocationCalls(rocm_deinterleave, "ROCmGatedDeltaNet deinterleave scratch");
    EXPECT_NE(rocm_deinterleave.find("requires bound graph workspace"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDAFlashAttentionDecodePartialsUseWorkspace)
{
    const auto kernel_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.cpp");
    const auto allocation_body = sliceBetween(
        kernel_source,
        "bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::allocateWorkspace(",
        "void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::freeWorkspace()");

    expectNoRawGpuAllocationCalls(allocation_body, "CUDAFlashAttention split-K partial workspace");
    EXPECT_NE(allocation_body.find("Flash decode requires bound graph workspace"), std::string::npos);
    EXPECT_NE(allocation_body.find("getBuffer(AttentionWorkspaceBuffers::PARTIAL_OUTPUT)"), std::string::npos);
    EXPECT_NE(allocation_body.find("getBufferSize(AttentionWorkspaceBuffers::PARTIAL_OUTPUT)"), std::string::npos);

    const auto cuda_kernel_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/attention/CUDAFlashAttentionKernels.cu");
    expectNoRawGpuAllocationCalls(cuda_kernel_source, "CUDAFlashAttention kernel wrappers");
    EXPECT_EQ(cuda_kernel_source.find("cudaFlashAttn_allocWorkspace"), std::string::npos);
    EXPECT_EQ(cuda_kernel_source.find("cudaFlashAttn_freeWorkspace"), std::string::npos);
    EXPECT_EQ(cuda_kernel_source.find("cudaFlashAttn_prefill_cublas_fp16kv"), std::string::npos);

    const auto debug_env_source =
        readFile(repoRoot() / "src/v2/utils/DebugEnv.h");
    EXPECT_EQ(debug_env_source.find("LLAMINAR_CUBLAS_ATTN"), std::string::npos);
    EXPECT_EQ(debug_env_source.find("cuda_cublas_attn"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, ProductionEnsureOnDeviceCallersStayExplicit)
{
    const auto root = repoRoot();
    const std::unordered_set<std::string> sanctioned = {
        // Tensor/coherence infrastructure owns the legacy tensor API while
        // callers migrate to TransferEngine, BufferArena, or workspace bindings.
        "src/v2/memory/CoherenceTracker.cpp",
        "src/v2/execution/local_execution/coherence/GpuCoherence.h",
        "src/v2/execution/local_execution/coherence/StageCoherence.cpp",
        "src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp",
        "src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp",
        "src/v2/tensors/TensorBase.cpp",
        "src/v2/tensors/cpu/CPUTensors.h",
        "src/v2/tensors/ITensor.h",
        "src/v2/tensors/TensorClasses.h",
        "src/v2/tensors/TensorSlice.h",

        // Multi-domain and loader paths are existing migration debt. New data
        // movement in these areas should go through TransferEngine.
        "src/v2/collective/LocalPPContext.cpp",
        "src/v2/collective/LocalTPContext.cpp",
        "src/v2/loaders/WeightManager.cpp",
        "src/v2/models/qwen/QwenGraphBase.cpp",

        // Current compute-stage debt tracked more narrowly below.
        "src/v2/execution/compute_stages/stages/AttentionOutputGateStage.cpp",
        "src/v2/execution/compute_stages/stages/EmbeddingStage.cpp",
        "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp",
        "src/v2/execution/compute_stages/stages/HiddenStateRowSelectStage.cpp",
        "src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp",
        "src/v2/execution/compute_stages/stages/KVCacheAppendStage.snapshot.cpp",
        "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp",
        "src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.cpp",
        "src/v2/execution/compute_stages/stages/MoELocalExpertStage.cpp",
        "src/v2/execution/compute_stages/stages/MoERoutingStage.cpp",
        "src/v2/execution/compute_stages/stages/QGateSplitStage.cpp",
        "src/v2/execution/compute_stages/stages/ResidualAddStage.cpp",
        "src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp",

        // Kernel adapter debt where direct tensors still feed legacy entrypoints.
        "src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp",
        "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTensorAdapter.cpp",
        "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp",
        "src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp",
        "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp",
        "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp",
    };

    std::vector<std::string> failures;
    const auto source_root = root / "src/v2";
    for (const auto &entry : std::filesystem::recursive_directory_iterator(source_root))
    {
        if (!entry.is_regular_file() || !isSourceFile(entry.path()))
            continue;
        const auto relative = std::filesystem::relative(entry.path(), root).generic_string();
        const auto source = readFile(entry.path());
        if (!hasExecutableEnsureOnDeviceCall(source))
            continue;
        if (sanctioned.find(relative) == sanctioned.end())
            failures.push_back(relative);
    }

    EXPECT_TRUE(failures.empty()) << [&]
    {
        std::ostringstream out;
        out << "New production ensureOnDevice() callers must not be added casually. "
               "Use TransferEngine for tensor movement, BufferArena/StageBufferContract "
               "coherence for graph stages, or IWorkspaceConsumer for graph-owned scratch. "
               "If this is truly infrastructure-owned legacy debt, add it here with a rationale.\n";
        for (const auto &failure : failures)
            out << failure << '\n';
        return out.str();
    }();
}

TEST(Test__GpuWorkspaceAllocationPolicy, ComputeStageEnsureOnDeviceDebtStaysExplicit)
{
    const auto root = repoRoot();
    const std::unordered_set<std::string> sanctioned = {
        "src/v2/execution/compute_stages/stages/AttentionOutputGateStage.cpp",
        "src/v2/execution/compute_stages/stages/EmbeddingStage.cpp",
        "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp",
        "src/v2/execution/compute_stages/stages/HiddenStateRowSelectStage.cpp",
        "src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp",
        "src/v2/execution/compute_stages/stages/KVCacheAppendStage.snapshot.cpp",
        "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp",
        "src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.cpp",
        "src/v2/execution/compute_stages/stages/MoELocalExpertStage.cpp",
        "src/v2/execution/compute_stages/stages/MoERoutingStage.cpp",
        "src/v2/execution/compute_stages/stages/QGateSplitStage.cpp",
        "src/v2/execution/compute_stages/stages/ResidualAddStage.cpp",
        "src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp",
    };

    std::vector<std::string> failures;
    const auto stages_root = root / "src/v2/execution/compute_stages/stages";
    for (const auto &entry : std::filesystem::recursive_directory_iterator(stages_root))
    {
        if (!entry.is_regular_file() || !isSourceFile(entry.path()))
            continue;
        const auto relative = std::filesystem::relative(entry.path(), root).generic_string();
        const auto source = readFile(entry.path());
        if (!hasExecutableEnsureOnDeviceCall(source))
            continue;
        if (sanctioned.find(relative) == sanctioned.end())
            failures.push_back(relative);
    }

    EXPECT_TRUE(failures.empty()) << [&]
    {
        std::ostringstream out;
        out << "New compute-stage ensureOnDevice() calls must not be added casually. "
               "Prefer BufferArena/StageBufferContract coherence for graph stages, or add an explicit "
               "sanction with rationale while migrating old debt.\n";
        for (const auto &failure : failures)
            out << failure << '\n';
        return out.str();
    }();
}

TEST(Test__GpuWorkspaceAllocationPolicy, HiddenStateRowSelectGraphManagedPathDoesNotSelfCohereArenaBuffers)
{
    const auto source = readFile(repoRoot() / "src/v2/execution/compute_stages/stages/HiddenStateRowSelectStage.cpp");
    const auto execute_gpu = sliceBetween(
        source,
        "bool HiddenStateRowSelectStage::executeGPU(",
        "    void HiddenStateRowSelectStage::releaseGpuParamState()");
    const auto launch_path = sliceBetween(
        execute_gpu,
        "        const auto *input_device =",
        "        return true;");

    EXPECT_NE(execute_gpu.find("const bool graph_managed"), std::string::npos);
    EXPECT_NE(execute_gpu.find("if (!graph_managed)"), std::string::npos);
    EXPECT_EQ(execute_gpu.find("output_base->ensureOnDevice("), std::string::npos)
        << "Row-select writes must allocate outputs without uploading stale host contents";
    EXPECT_EQ(launch_path.find("ensureOnDevice("), std::string::npos)
        << "Graph-managed row-select execution must rely on executor/arena input coherence";
    EXPECT_EQ(launch_path.find("allocateOnDevice("), std::string::npos)
        << "Graph-managed row-select execution must rely on executor/arena output allocation";
    EXPECT_NE(execute_gpu.find("if (!graph_managed)\n        {\n            output_base->transitionToWithEvent"), std::string::npos)
        << "Only direct, non-arena row-select execution should record tensor completion events";
}

TEST(Test__GpuWorkspaceAllocationPolicy, EmbeddingGraphManagedPathDoesNotSelfMarkArenaOutput)
{
    const auto source = readFile(repoRoot() / "src/v2/execution/compute_stages/stages/EmbeddingStage.cpp");
    const auto execute_body = sliceBetween(
        source,
        "bool EmbeddingStage::execute(IDeviceContext *ctx)",
        "    size_t EmbeddingStage::estimatedFlops() const");

    EXPECT_NE(execute_body.find("!params_.output_buffer_id.has_value()"), std::string::npos)
        << "Graph-managed embedding outputs must be marked written by DeviceGraphExecutor";
    EXPECT_EQ(execute_body.find("TensorCoherenceState::DEVICE_AUTHORITATIVE, std::nullopt"), std::string::npos)
        << "Embedding must record completion events against the explicit stage device, never an unspecified device";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDAEmbeddingDoesNotUploadDynamicTokensDuringGraphCapture)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/cuda/ops/CUDAOpsKernels.cpp");
    const auto apply_tensor = sliceBetween(
        source,
        "bool CUDAEmbeddingKernelT::apply_tensor(",
        "    WorkspaceRequirements CUDAEmbeddingKernelT::getWorkspaceRequirements(");

    EXPECT_NE(apply_tensor.find("isGraphCaptureActive()"), std::string::npos);
    EXPECT_NE(apply_tensor.find("Token IDs were not preloaded before graph capture"), std::string::npos);
    EXPECT_NE(apply_tensor.find("Token ID upload requires an explicit non-null stream"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, Qwen35MoECombineDoesNotForceFreshGraphSegment)
{
    const auto residual_source = readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ResidualAddStage.h");
    EXPECT_NE(residual_source.find("graph_capture_boundary_before"), std::string::npos)
        << "ResidualAddStage needs an opt-in graph-capture boundary for graph joins such as MoE combine.";
    EXPECT_NE(residual_source.find("requiresGraphCaptureSegmentBoundaryBefore()"), std::string::npos);

    const auto graph_source = readFile(repoRoot() / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto combine_section = sliceBetween(
        graph_source,
        "// Stage 5: Combine expert output + shared expert output",
        "// Stage 6: Explicit residual");

    EXPECT_EQ(combine_section.find("add_params.graph_capture_boundary_before = true"), std::string::npos)
        << "MoE combine should stay in the fused captured graph. Reintroducing this boundary "
           "splits Qwen3.6 MoE verifier replay into one graph segment per layer.";
    EXPECT_EQ(combine_section.find("copy_params.graph_capture_boundary_before = true"), std::string::npos)
        << "The no-shared-expert copy form must not reintroduce per-layer graph segmentation either.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, LiveHybridCheckpointStorageUsesReusablePool)
{
    const auto header_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    EXPECT_NE(header_source.find("LiveHybridCheckpointStorageSlot"), std::string::npos);
    EXPECT_NE(header_source.find("live_hybrid_checkpoint_storage_pool_"), std::string::npos);

    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto ensure_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::ensureLiveHybridCheckpointStorage(",
        "bool DeviceGraphOrchestrator::acquireLiveHybridCheckpointStorage(");
    EXPECT_NE(ensure_body.find("return acquireLiveHybridCheckpointStorage(handle);"), std::string::npos)
        << "The hot live-checkpoint path must not allocate fresh hybrid storage on every MTP decode step.";
    EXPECT_EQ(ensure_body.find("allocateDeviceByteStorage("), std::string::npos)
        << "Per-step checkpoint device allocation regresses CUDA MoE MTP decode.";

    const auto acquire_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::acquireLiveHybridCheckpointStorage(",
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(");
    EXPECT_NE(acquire_body.find("host_storage.use_count() == 1"), std::string::npos)
        << "Pool slots must not be reused while a PrefixStateSnapshot still owns host payload storage.";
    EXPECT_NE(acquire_body.find("device_storage.use_count() == 1"), std::string::npos)
        << "Pool slots must not be reused while a PrefixStateSnapshot still owns device payload storage.";
    EXPECT_NE(acquire_body.find("live_prefix_checkpoint_hybrid_storage_pool_hits"), std::string::npos);
    EXPECT_NE(acquire_body.find("live_prefix_checkpoint_hybrid_storage_pool_misses"), std::string::npos);
}
