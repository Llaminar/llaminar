/**
 * @file Test__GpuWorkspaceAllocationPolicy.cpp
 * @brief Source-level guards for capture-sensitive GPU workspace allocation policy.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEWorkspaceRequirements.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
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

    void expectNeedleBefore(const std::string &source,
                            const std::string &before,
                            const std::string &after,
                            const std::string &message)
    {
        const size_t before_pos = source.find(before);
        const size_t after_pos = source.find(after);
        ASSERT_NE(before_pos, std::string::npos) << "Missing required source marker: " << before;
        ASSERT_NE(after_pos, std::string::npos) << "Missing required source marker: " << after;
        EXPECT_LT(before_pos, after_pos) << message;
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

    std::string removeAsciiWhitespace(std::string source)
    {
        source.erase(
            std::remove_if(
                source.begin(),
                source.end(),
                [](unsigned char c)
                {
                    return std::isspace(c) != 0;
                }),
            source.end());
        return source;
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

    size_t countOccurrences(const std::string &source, const std::string &needle)
    {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = source.find(needle, pos)) != std::string::npos)
        {
            ++count;
            pos += needle.size();
        }
        return count;
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
        "src/v2/kernels/cuda/ops/CUDACastKernels.cu",
        "src/v2/kernels/cuda/ops/CUDARoPEKernels.cu",
        "src/v2/kernels/cuda/ops/CUDARowSelectKernels.cu",
        "src/v2/kernels/rocm/ROCmWeightPacker.cpp",
        "src/v2/kernels/rocm/gemm/HipBLASGemmKernel.cpp",
        "src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp",
        "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp",
        "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp",
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

TEST(Test__GpuWorkspaceAllocationPolicy, MoEKernelsDoNotOwnRawGpuAllocations)
{
    const auto root = repoRoot();
    expectNoRawGpuAllocationCalls(
        readFile(root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp"),
        "CUDA MoE kernel");
    expectNoRawGpuAllocationCalls(
        readFile(root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp"),
        "ROCm MoE kernel");
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
    const auto *expert_mask = reqs.find(llaminar2::MoEWorkspaceBuffers::GROUP_EXPERT_MASK);
    ASSERT_NE(expert_mask, nullptr);
    EXPECT_GE(expert_mask->size_bytes, static_cast<size_t>(num_experts) * sizeof(uint8_t));
    EXPECT_GT(static_cast<size_t>(num_experts), static_cast<size_t>(max_seq_len) * top_k)
        << "fixture must cover the small-token, many-expert regression";
}

TEST(Test__GpuWorkspaceAllocationPolicy, ROCmMoEWorkspaceOwnsRoutingStateAndMetadataCaches)
{
    const int max_seq_len = 9;
    const int d_model = 2048;
    const int intermediate = 512;
    const int num_experts = 256;
    const int top_k = 8;

    const auto reqs = llaminar2::MoEWorkspaceBuffers::rocmMoE(
        max_seq_len, d_model, intermediate, num_experts, top_k);
    const auto *histogram = reqs.find(llaminar2::MoEWorkspaceBuffers::ROCM_HISTOGRAM_COUNTS);
    ASSERT_NE(histogram, nullptr);
    EXPECT_GE(histogram->size_bytes,
              static_cast<size_t>(llaminar2::MoEWorkspaceBuffers::kHistogramLayerSlots) *
                  static_cast<size_t>(num_experts) * sizeof(uint64_t));
    const auto *expert_mask = reqs.find(llaminar2::MoEWorkspaceBuffers::ROCM_EXPERT_MASK);
    ASSERT_NE(expert_mask, nullptr);
    EXPECT_GE(expert_mask->size_bytes, static_cast<size_t>(num_experts) * sizeof(bool));
    EXPECT_NE(reqs.find(llaminar2::MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES), nullptr);
    EXPECT_NE(reqs.find(llaminar2::MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES), nullptr);
    EXPECT_NE(reqs.find(llaminar2::MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES), nullptr);
    EXPECT_NE(reqs.find(llaminar2::MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_WEIGHTS), nullptr);
    EXPECT_NE(reqs.find(llaminar2::MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_SCALES), nullptr);
    EXPECT_NE(reqs.find(llaminar2::MoEWorkspaceBuffers::ROCM_ROUTER_FP16_GATE_WEIGHTS), nullptr);
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

TEST(Test__GpuWorkspaceAllocationPolicy, StageVerifierUsesTensorDeviceOrdinalForGpuValidators)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/graph/StageVerifier.cpp");
    const auto executable_source = stripCommentsAndStringLiterals(source);

    EXPECT_NE(executable_source.find("getTensorValidator(device_opt->type, device_opt->ordinal)"),
              std::string::npos)
        << "Multi-GPU validation must fetch the validator for the tensor's actual "
           "device ordinal, not the ambient CUDA/HIP current device.";
    EXPECT_EQ(executable_source.find("getTensorValidator(device_opt->type);"),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoERebalanceMaintenanceSkipsSyncForDedicatedCollectiveLane)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto fn = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph(",
        "// =====================================================================\n    // IForwardExecutionHost interface implementations");
    const auto executable_fn = stripCommentsAndStringLiterals(fn);

    EXPECT_NE(fn.find("Dedicated maintenance lanes skip this guard"), std::string::npos)
        << "The maintenance scheduler must document why dedicated lanes are allowed to overlap decode.";
    EXPECT_NE(fn.find("Shared-communicator maintenance graphs must finish"), std::string::npos)
        << "The fallback guard must still document why shared-communicator overlap is forbidden.";
    EXPECT_NE(executable_fn.find("maintenance_graph_uses_decode_collective_lane()"), std::string::npos)
        << "Maintenance should only synchronize when the captured graph really uses the decode LocalTP context.";
    EXPECT_NE(fn.find("device_maintenance_graph_same_communicator_sync"), std::string::npos);
    EXPECT_NE(fn.find("dedicated_collective_lane"), std::string::npos)
        << "PerfStats must report whether the maintenance graph used a dedicated collective lane.";
    EXPECT_NE(executable_fn.find("gpu_ctx->synchronizeStreamChecked(maintenance_stream)"), std::string::npos)
        << "The shared-lane fallback must still fail safe.";
    EXPECT_NE(executable_fn.find("cache.completion_event_in_flight = false;"), std::string::npos)
        << "A synchronously completed maintenance wave must not be reported as still in flight.";
    expectNeedleBefore(
        fn,
        "device_maintenance_graph_replay_enqueue",
        "device_maintenance_graph_same_communicator_sync",
        "The shared-communicator guard must run after the maintenance graph is enqueued.");
    expectNeedleBefore(
        executable_fn,
        "maintenance_graph_uses_decode_collective_lane()",
        "gpu_ctx->synchronizeStreamChecked(maintenance_stream)",
        "The sync fallback must be gated by an actual shared LocalTP context.");
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPPendingLogitsStreamsUseOwnershipHelpers)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");

    /*
     * Pending logits streams are not general-purpose scratch pointers. They
     * encode ownership of an asynchronous graph replay stream from a logits
     * producer to exactly the next consumer. Remove the helper implementation
     * itself, then require every remaining production use to go through the
     * explicit publish/consume/peek/clear verbs.
     */
    std::string guarded_source = source;
    const size_t helper_begin = guarded_source.find(
        "const char *DeviceGraphOrchestrator::pendingLogitsStreamRoleName(");
    ASSERT_NE(helper_begin, std::string::npos);
    const size_t helper_end = guarded_source.find(
        "void DeviceGraphOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled",
        helper_begin);
    ASSERT_NE(helper_end, std::string::npos);
    guarded_source.erase(helper_begin, helper_end - helper_begin);

    const auto executable_source = stripCommentsAndStringLiterals(guarded_source);
    const std::vector<std::string> legacy_raw_fields = {
        "pending_mtp_logits_stream_",
        "pending_main_decode_logits_stream_",
        "pending_all_position_logits_stream_",
    };
    for (const auto &field : legacy_raw_fields)
    {
        EXPECT_EQ(executable_source.find(field), std::string::npos)
            << field << " must not come back as a role-specific raw stream field.";
        EXPECT_EQ(stripCommentsAndStringLiterals(header).find(field), std::string::npos)
            << field << " must not come back as a role-specific raw stream field.";
    }

    const auto executable_header = stripCommentsAndStringLiterals(header);
    const auto compact_executable_header = removeAsciiWhitespace(executable_header);
    EXPECT_NE(executable_header.find("struct PendingLogitsStreamHandoff"), std::string::npos)
        << "Pending stream storage should remain structurally wrapped.";
    EXPECT_NE(compact_executable_header.find("std::array<PendingLogitsStreamHandoff,3>pending_logits_streams_"),
              std::string::npos)
        << "Pending stream storage should stay role-indexed instead of scattered into raw fields.";
    EXPECT_NE(compact_executable_header.find("void*stream_=nullptr"), std::string::npos)
        << "The raw stream pointer must stay private to the handoff object.";
    EXPECT_NE(executable_header.find("bool canPublish(void *candidate) const"), std::string::npos)
        << "The one-shot overwrite rule should live on the handoff object.";
    EXPECT_EQ(executable_header.find("void *&pendingLogitsStreamSlot"), std::string::npos)
        << "Do not expose mutable raw stream references from the orchestrator.";
    EXPECT_EQ(executable_source.find("pending_logits_streams_"), std::string::npos)
        << "Production code outside the helper implementation must not touch the slot table directly.";
    EXPECT_EQ(executable_source.find("pendingLogitsStreamSlot("), std::string::npos)
        << "Production code must use the handoff object API, not a raw slot helper.";

    EXPECT_NE(source.find("publishPendingLogitsStream("), std::string::npos);
    EXPECT_NE(source.find("consumePendingLogitsStream("), std::string::npos);
    EXPECT_NE(source.find("peekPendingLogitsStream("), std::string::npos);
    EXPECT_NE(source.find("clearPendingLogitsStream("), std::string::npos);
    EXPECT_NE(source.find("Cannot replace unconsumed pending logits stream"), std::string::npos)
        << "Publishing a different stream over an unconsumed logits handoff must hard-fail.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPSidecarGraphCaptureInstallsLocalTPBoundaryHook)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    const size_t policy_build = sidecar_body.find("auto capture_policy = buildDecodeCapturePolicy(");
    ASSERT_NE(policy_build, std::string::npos)
        << "MTP sidecar execution must continue to build a decode capture policy.";
    const size_t boundary_hook =
        sidecar_body.find("capture_policy.before_begin_capture", policy_build);
    ASSERT_NE(boundary_hook, std::string::npos)
        << "MTP sidecar graph capture must join the same LocalTP capture boundary as main decode.";
    const size_t boundary_call =
        sidecar_body.find("waitAtDecodeGraphCaptureBoundary(", boundary_hook);
    ASSERT_NE(boundary_call, std::string::npos)
        << "The sidecar boundary hook must route through DeviceGraphOrchestrator so participant indices are validated.";
    const size_t execute_call =
        sidecar_body.find("executor_.executeDecodeWithCapturePolicy(", policy_build);
    ASSERT_NE(execute_call, std::string::npos)
        << "MTP sidecar execution must still use the centralized capture policy path.";

    EXPECT_LT(policy_build, boundary_hook)
        << "The sidecar capture policy should be built before installing the boundary hook.";
    EXPECT_LT(boundary_hook, execute_call)
        << "LocalTP participants must rendezvous before any sidecar stream begins graph capture.";
    EXPECT_LT(boundary_call, execute_call)
        << "The sidecar hook must be installed before entering the executor capture path.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPShiftedKVAsyncHandoffUsesEventBeforeConsumers)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto runner_interface =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto transaction_record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordDeviceResidentMTPTransactionMutation(",
        "bool DeviceGraphOrchestrator::recordRestoredDeviceResidentMTPTransaction(");
    const auto metadata_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");

    EXPECT_NE(header.find("struct PendingShiftedMTPKVReadyState"), std::string::npos)
        << "Deferred shifted MTP KV writes must be represented by an explicit owned state object.";
    EXPECT_NE(header.find("recordShiftedMTPKVReady"), std::string::npos);
    EXPECT_NE(header.find("waitForPendingShiftedMTPKVReady"), std::string::npos);
    EXPECT_NE(source.find("shifted_mtp_kv_ready_events"), std::string::npos);
    EXPECT_NE(source.find("shifted_mtp_kv_ready_waits"), std::string::npos);

    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    EXPECT_NE(sidecar_body.find("waitForPendingShiftedMTPKVReady"), std::string::npos)
        << "A new MTP sidecar run must wait before reading/appending shifted MTP KV.";
    EXPECT_NE(sidecar_body.find("recordShiftedMTPKVReady"), std::string::npos)
        << "KV-only sidecar replay must publish an event if it skips CPU stream sync.";
    EXPECT_NE(sidecar_body.find("shifted_mtp_kv_stream_syncs_deferred"), std::string::npos);
    EXPECT_NE(sidecar_body.find("recordDeviceResidentMTPTransactionMutation("),
              std::string::npos)
        << "Every successful GPU sidecar must advance the request transaction fence.";
    EXPECT_NE(transaction_record_body.find(
                  "cache->deviceSequenceCachedTokenCountPtr(0)"),
              std::string::npos)
        << "The request transaction must name each shifted cache's canonical "
           "device count, not a copied metadata row.";
    EXPECT_EQ(sidecar_body.find(
                  "static_cast<const int32_t *>(effective_position_ids_device)"),
              std::string::npos)
        << "Device position ids describe append geometry but do not own the "
           "post-append shifted-cache count.";
    const size_t kv_only_guard = sidecar_body.find("if (!kv_cache_only)");
    const size_t logits_defer = sidecar_body.find("deferredSamplingStream");
    ASSERT_NE(kv_only_guard, std::string::npos);
    ASSERT_NE(logits_defer, std::string::npos);
    EXPECT_LT(kv_only_guard, logits_defer)
        << "KV-only sidecar replay must not use the pending-logits stream handoff; it owns shifted KV.";

    const auto retired_host_publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatch(");
    EXPECT_NE(retired_host_publish_body.find(
                  "GPU MTP spec-state publication from host step plans is retired"),
              std::string::npos)
        << "GPU publication must stay on the compact device-resident path; "
           "the host step-plan publisher is CPU compatibility only.";

    const auto publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    const size_t publish_wait = publish_body.find("waitForPendingShiftedMTPKVReady");
    const size_t kv_publish = publish_body.find("publishSequenceStateFromDeviceMetadata");
    ASSERT_NE(publish_wait, std::string::npos);
    ASSERT_NE(kv_publish, std::string::npos);
    EXPECT_LT(publish_wait, kv_publish)
        << "Accepted-state publication truncates MTP KV and must wait for deferred shifted appends first.";
    const size_t terminal_hidden_publish =
        publish_body.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState");
    const size_t ready_event = publish_body.find("recordAcceptedSpecPublicationReady");
    ASSERT_NE(terminal_hidden_publish, std::string::npos);
    ASSERT_NE(ready_event, std::string::npos);
    EXPECT_LT(terminal_hidden_publish, ready_event)
        << "Publication readiness must cover the accepted verifier terminal hidden rows.";
    EXPECT_NE(publish_body.find("spec_state_terminal_hidden_publications"), std::string::npos)
        << "Terminal-hidden publication should be visible in perf stats.";
    EXPECT_NE(publish_body.find("recordDeviceResidentMTPTransactionMutation("),
              std::string::npos)
        << "Direct publication must advance the same request transaction after "
           "mutating canonical shifted-cache counts.";
    EXPECT_NE(metadata_body.find("request.outcome.mtp_transaction"),
              std::string::npos)
        << "Publication preflight must consume the child-local outcome lease.";
    expectNeedleBefore(
        metadata_body,
        "waitForDeviceResidentMTPTransaction(",
        "enqueueDeriveSpeculativePublicationMetadata(",
        "Publication metadata must wait for the transaction's latest mutation "
        "fence before reading its canonical device count rows.");

    const auto shifted_mutation_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::recordShiftedMTPKVReplayStateMutation(",
        "void DeviceGraphOrchestrator::handleLivePrefixReplayStateAfterMutation(");
    const auto executable_shifted_mutation_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(shifted_mutation_body));
    EXPECT_EQ(executable_shifted_mutation_body.find("clearPendingAllPositionVerifierStateReady()"),
              std::string::npos)
        << "Shifted-MTP KV epoch updates must not discard the verifier-state "
           "event that accepted-state publication still needs for row-snapshot "
           "restore ordering.";
    EXPECT_EQ(shifted_mutation_body.find(
                  "retargetDeviceResidentShiftedMTPKVStateMailboxAfterMutation"),
              std::string::npos)
        << "Request transactions keep stable identity; live-state epoch retargeting is obsolete.";

    const auto row_select_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPHiddenRowSelect(",
        "bool DeviceGraphOrchestrator::executeMTPTerminalHiddenRowSelect(");
    EXPECT_NE(row_select_body.find("cache.stage->setGPUStream(row_select_stream)"), std::string::npos)
        << "Publication must be able to bind terminal-hidden row-select to the publication stream.";

    const auto boundary_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareShiftedMTPKVCommitBoundary(",
        "DeviceResidentLogicalSequenceStateHandle");
    const auto compact_boundary =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(boundary_body));
    expectNeedleBefore(
        compact_boundary,
        "if(state_.device_id.is_gpu())",
        "cache.get_cached_tokens",
        "GPU transaction validation must be selected before the CPU-only count path.");
    EXPECT_NE(compact_boundary.find("transaction->state->cachedTokensForDepth(depth)"),
              std::string::npos)
        << "The boundary must validate the exact canonical count named by the lease.";
    EXPECT_NE(compact_boundary.find("waitForDeviceResidentMTPTransaction("),
              std::string::npos)
        << "Device-owned cache counts require an event-ordered transaction wait.";
    EXPECT_NE(boundary_body.find(
                  "device_resident_shifted_mtp_kv_commit_boundaries"),
              std::string::npos)
        << "Perfstats must prove that production commits consumed the resident boundary.";
    EXPECT_EQ(boundary_body.find("host_mirror_lag"), std::string::npos)
        << "A stale host count must never be accepted as a compatibility policy.";
    EXPECT_EQ(runner_interface.find("DeviceResidentShiftedMTPKVStateMailbox"),
              std::string::npos)
        << "The duplicate shifted-cache mailbox type must stay retired.";

    const auto sequential_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(");
    const auto device_target_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(");
    const auto initial_device_outcome_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPInitialShiftedRowFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(");
    const auto resident_logical_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromDeviceOutcome(");
    const auto device_outcome_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(");
    const auto partial_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(",
        "const float *DeviceGraphOrchestrator::mtpLogits() const");

    const auto sequential_executable = stripCommentsAndStringLiterals(sequential_body);
    const auto device_target_executable = stripCommentsAndStringLiterals(device_target_body);
    const auto initial_device_outcome_executable =
        stripCommentsAndStringLiterals(initial_device_outcome_body);
    const auto resident_logical_executable = stripCommentsAndStringLiterals(resident_logical_body);
    const auto device_outcome_executable = stripCommentsAndStringLiterals(device_outcome_body);
    const auto partial_executable = stripCommentsAndStringLiterals(partial_body);

    expectNeedleBefore(
        sequential_executable,
        "waitForPendingShiftedMTPKVReady",
        "prepareShiftedMTPKVCommitBoundary",
        "Sequential shifted-row commits must order deferred graph appends before selecting the cache-state owner.");
    expectNeedleBefore(
        device_target_executable,
        "waitForPendingShiftedMTPKVReady",
        "prepareShiftedMTPKVCommitBoundary",
        "Device-target shifted-row commits must order deferred graph appends before selecting the cache-state owner.");
    expectNeedleBefore(
        resident_logical_executable,
        "waitForDeviceResidentLogicalSequenceStateMailbox",
        "waitForPendingShiftedMTPKVReady",
        "Resident correction shifted commits must first receive the publication mailbox.");
    expectNeedleBefore(
        resident_logical_executable,
        "waitForPendingShiftedMTPKVReady",
        "prepareShiftedMTPKVCommitBoundary",
        "Resident correction shifted commits must order deferred graph appends before consuming the device boundary.");
    expectNeedleBefore(
        device_outcome_executable,
        "waitForPendingShiftedMTPKVReady",
        "prepareShiftedMTPKVCommitBoundary",
        "Device-outcome suffix commits must order deferred graph appends before consuming the device boundary.");
    expectNeedleBefore(
        partial_executable,
        "waitForPendingShiftedMTPKVReady",
        "prepareShiftedMTPKVCommitBoundary",
        "Partial-forward shifted-row commits must order deferred graph appends before selecting the cache-state owner.");

    for (const auto *body : {&sequential_executable,
                             &device_target_executable,
                             &resident_logical_executable,
                             &device_outcome_executable,
                             &partial_executable})
    {
        EXPECT_EQ(body->find("get_cached_tokens"), std::string::npos)
            << "Shifted-row commit entry points must not inspect host cache counters directly.";
        EXPECT_NE(body->find("prepareShiftedMTPKVCommitBoundary"),
                  std::string::npos)
            << "Every shifted-row commit entry point must use the centralized ownership contract.";
    }

    EXPECT_NE(resident_logical_body.find("waitForDeviceResidentLogicalSequenceStateMailbox"),
              std::string::npos)
        << "Resident correction shifted commits must wait on the publication mailbox before touching shifted KV.";
    EXPECT_NE(resident_logical_body.find("nextConditionTokenDeviceForRequest"),
              std::string::npos)
        << "Resident correction shifted commits must consume the device-derived next condition token.";
    EXPECT_NE(resident_logical_body.find("targetPositionDeviceForRequest"),
              std::string::npos)
        << "Resident correction shifted commits must consume the mailbox's "
           "request-local target position directly on device.";
    EXPECT_EQ(resident_logical_body.find("position_offset_override"),
              std::string::npos)
        << "A host scalar position override is not part of a fully resident "
           "logical-state handoff.";
    EXPECT_NE(resident_logical_body.find(
                  "/*device_position_offset=*/already_appended_tokens"),
              std::string::npos)
        << "Any already-appended suffix offset must be composed by the backend "
           "with the resident position row.";
    EXPECT_EQ(resident_logical_executable.find("state_.positions"),
              std::string::npos)
        << "Resident correction shifted commits must not derive positions from "
           "non-authoritative host mirrors.";
    EXPECT_EQ(resident_logical_executable.find("get_position()"),
              std::string::npos)
        << "Resident correction shifted commits must not call host logical getters.";
    EXPECT_NE(resident_logical_body.find("/*draft_condition_tokens=*/nullptr"),
              std::string::npos)
        << "Resident correction shifted commits must use the external device-token sidecar path.";
    expectNeedleBefore(
        resident_logical_executable,
        "executeMTPDepth0Batched",
        "selectMTPTerminalHiddenRowsFromDeviceAcceptedState",
        "Resident correction shifted commits must restore the accepted verifier terminal hidden after the sidecar append.");
    expectNeedleBefore(
        resident_logical_executable,
        "selectMTPTerminalHiddenRowsFromDeviceAcceptedState",
        "recordShiftedMTPKVReplayStateMutation",
        "Resident correction shifted commits must repair the terminal-hidden handoff before publishing the shifted-KV epoch.");
    EXPECT_NE(resident_logical_body.find(
                  "shifted_row_resident_terminal_hidden_reselects"),
              std::string::npos)
        << "Accepted-row terminal-hidden repair must remain visible in perf counters.";
    expectNeedleBefore(
        resident_logical_executable,
        "recordShiftedMTPKVReplayStateMutation",
        "retargetDeviceResidentLogicalSequenceStateMailboxAfterShiftedKVMutation",
        "Resident correction shifted commits must make the mailbox current after the shifted-KV epoch advances.");
    EXPECT_NE(resident_logical_body.find(
                  "retargetDeviceResidentLogicalSequenceStateMailboxAfterShiftedKVMutation"),
              std::string::npos)
        << "Resident correction shifted commits must keep the device mailbox usable "
           "for the next pending condition without a host-token fallback.";

    /*
     * The mandatory LocalTP publication route has two shifted-cache entry
     * points: an initial row when the chained sidecar could not retain row zero,
     * and a bounded suffix for the remaining accepted verifier rows. Both must
     * consume the same pre-verifier base-position snapshot on device. Keeping
     * these checks beside the shifted-cache mailbox contract makes it difficult
     * to accidentally split one transaction back across host and device owners.
     */
    for (const auto *body : {&initial_device_outcome_body,
                             &device_outcome_body})
    {
        EXPECT_NE(body->find("mtp_publication_base_cache_snapshot_ready_"),
                  std::string::npos)
            << "Device-outcome shifted commits require the resident pre-verifier base snapshot.";
        EXPECT_NE(body->find("metadata.base_cached_tokens"), std::string::npos)
            << "Device-outcome shifted commits must derive absolute positions from the resident base row.";
        EXPECT_NE(body->find("/*expected_cached_tokens=*/-1"),
                  std::string::npos)
            << "Device-outcome shifted commits must select the canonical device cache-count mailbox.";
        EXPECT_EQ(body->find("position_offset_override"), std::string::npos)
            << "The active device-outcome route must not accept or inspect a host position scalar.";
        EXPECT_EQ(body->find("already_appended_shifted_kv_tokens"),
                  std::string::npos)
            << "The active device-outcome route must not accept a host shadow of shifted-cache residency.";
    }
    EXPECT_NE(initial_device_outcome_body.find(
                  "/*device_position_offset=*/0"),
              std::string::npos)
        << "The initial row must use the resident verifier base without a host-computed offset.";
    EXPECT_NE(device_outcome_body.find(
                  "/*device_position_offset=*/already_appended_tokens"),
              std::string::npos)
        << "The suffix offset must be composed by the fused GPU preparation primitive.";
    EXPECT_EQ(initial_device_outcome_executable.find("checkpoint.cached_tokens"),
              std::string::npos)
        << "The checkpoint supplies terminal hidden state, not the authoritative GPU position.";
    EXPECT_EQ(device_outcome_executable.find("state_.positions"), std::string::npos)
        << "The suffix must not derive positions from a host-owned sequence mirror.";

    const auto initial_interface = sliceBetween(
        runner_interface,
        "virtual bool commitMTPInitialShiftedRowFromDeviceOutcome(",
        "virtual bool commitMTPShiftedRowFromDeviceTargetSample(");
    const auto suffix_interface = sliceBetween(
        runner_interface,
        "virtual bool commitMTPShiftedRowsFromDeviceOutcome(",
        "virtual const float *mtpLogits() const");
    for (const auto *interface_body : {&initial_interface, &suffix_interface})
    {
        const auto executable_interface =
            stripCommentsAndStringLiterals(*interface_body);
        EXPECT_EQ(executable_interface.find("position_offset_override"),
                  std::string::npos)
            << "Host position overrides are not part of the device-outcome API contract.";
        EXPECT_EQ(executable_interface.find("already_appended_shifted_kv_tokens"),
                  std::string::npos)
            << "Host shifted-cache count shadows are not part of the device-outcome API contract.";
    }
    EXPECT_NE(partial_body.find("waitForPendingShiftedMTPKVReady"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPTerminalHiddenRowSelectCachesTrackWorkspaceGeneration)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto executable_header = stripCommentsAndStringLiterals(header);
    const auto compact_header = removeAsciiWhitespace(executable_header);
    const auto executable_source = stripCommentsAndStringLiterals(source);

    EXPECT_GE(countOccurrences(compact_header, "uint64_tworkspace_generation=0;"), 2u)
        << "Both single-row and multi-row MTP terminal-hidden helper caches must remember "
           "the allocator generation they last ran under.";
    EXPECT_GE(countOccurrences(executable_source, "workspace_generation_changed"), 3u)
        << "All terminal-hidden helper paths must rebuild when the workspace allocator generation changes.";
    EXPECT_GE(countOccurrences(executable_source, "cache.workspace_generation = workspaceGeneration(state_.device_id);"), 3u)
        << "Each terminal-hidden helper execution path must publish the generation that validated its workspace bindings.";

    const auto reset_body = sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override");
    const auto reset_executable =
        stripCommentsAndStringLiterals(reset_body);
    EXPECT_NE(reset_executable.find("mtp_terminal_hidden_row_select_cache_.invalidate()"),
              std::string::npos);
    EXPECT_NE(reset_executable.find("mtp_terminal_hidden_rows_select_cache_.invalidate()"),
              std::string::npos);
    EXPECT_EQ(reset_executable.find("mtp_terminal_hidden_row_select_cache_.resetSessionState()"),
              std::string::npos)
        << "clear_cache() must not preserve tiny terminal-hidden helper graphs after request-state teardown.";
    EXPECT_EQ(reset_executable.find("mtp_terminal_hidden_rows_select_cache_.resetSessionState()"),
              std::string::npos)
        << "clear_cache() must not preserve tiny terminal-hidden helper graphs after request-state teardown.";

    const auto clear_inference_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearInferenceState()",
        "// =========================================================================\n"
        "    // Private Helpers");
    const auto clear_inference_executable =
        stripCommentsAndStringLiterals(clear_inference_body);
    EXPECT_NE(clear_inference_executable.find("mtp_terminal_hidden_row_select_cache_.invalidate()"),
              std::string::npos);
    EXPECT_NE(clear_inference_executable.find("mtp_terminal_hidden_rows_select_cache_.invalidate()"),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPVerifierGDNStateSnapshotsUseDecodeEquivalentRows)
{
    const auto root = repoRoot();
    const auto cuda_source =
        readFile(root / "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNetKernels.cu");
    const auto rocm_source =
        readFile(root / "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip");

    /*
     * All-position MTP state publication restores a captured GDN state row and
     * continues normal decode from it.  The current verifier path is deliberately
     * grouped, not a hidden host loop over one-row launches: it advances rows in
     * serial recurrence order inside one graph-capturable kernel route and
     * publishes post-row state snapshots from device memory.
     */
    const auto cuda_route = stripCommentsAndStringLiterals(sliceBetween(
        cuda_source,
        "bool cudaGDN_chunk_forward_kernel_route(",
        "bool cudaGDN_chunk_forward("));
    const auto cuda_batched_route = stripCommentsAndStringLiterals(sliceBetween(
        cuda_source,
        "bool cudaGDN_chunk_forward_batched_kernel_route(",
        "bool cudaGDN_chunk_forward_kernel_route("));
    EXPECT_NE(cuda_route.find("state_snapshots"), std::string::npos)
        << "CUDA grouped verifier GDN must publish post-row state snapshots.";
    EXPECT_NE(cuda_route.find("snapshot_stride_floats"), std::string::npos);
    EXPECT_NE(cuda_route.find("device_effective_seq_len"), std::string::npos)
        << "CUDA grouped verifier GDN must support padded verifier graphs.";
    EXPECT_NE(cuda_batched_route.find("!stream"), std::string::npos)
        << "CUDA grouped verifier GDN must reject the device-default stream.";
    EXPECT_TRUE(
        cuda_batched_route.find("static_cast<cudaStream_t>(stream)") != std::string::npos ||
        cuda_batched_route.find("(cudaStream_t)stream") != std::string::npos)
        << "CUDA grouped verifier GDN must launch on the explicit capture stream.";
    EXPECT_EQ(cuda_route.find("cudaDeviceSynchronize"), std::string::npos)
        << "CUDA grouped verifier GDN must stay graph-capturable.";

    const auto cuda_body = stripCommentsAndStringLiterals(sliceBetween(
        cuda_source,
        "bool cudaGDN_chunk_forward(",
        "bool cudaGDN_chunk_forward_effective("));
    EXPECT_NE(cuda_body.find("cudaGDN_chunk_forward_kernel_route("), std::string::npos);
    EXPECT_NE(cuda_body.find("state_snapshots"), std::string::npos);
    EXPECT_NE(cuda_body.find("nullptr"), std::string::npos)
        << "Non-padded CUDA verifier chunks should use the same grouped route "
           "without an effective-length guard.";
    const auto cuda_effective_body = stripCommentsAndStringLiterals(sliceBetween(
        cuda_source,
        "bool cudaGDN_chunk_forward_effective(",
        "bool cudaGDN_short_conv1d("));
    EXPECT_NE(cuda_effective_body.find("cudaGDN_chunk_forward_kernel_route("), std::string::npos);
    EXPECT_NE(cuda_effective_body.find("device_effective_seq_len"), std::string::npos)
        << "CUDA padded verifier snapshots must be guarded by the device effective length.";

    const auto rocm_route = stripCommentsAndStringLiterals(sliceBetween(
        rocm_source,
        "bool rocmGDN_chunk_forward_kernel_route(",
        "bool rocmGDN_chunk_forward("));
    const auto rocm_batched_route = stripCommentsAndStringLiterals(sliceBetween(
        rocm_source,
        "bool rocmGDN_chunk_forward_batched_kernel_route(",
        "bool rocmGDN_chunk_forward_kernel_route("));
    EXPECT_NE(rocm_route.find("state_snapshots"), std::string::npos)
        << "ROCm grouped verifier GDN must publish post-row state snapshots.";
    EXPECT_NE(rocm_route.find("snapshot_stride_floats"), std::string::npos);
    EXPECT_NE(rocm_route.find("device_effective_seq_len"), std::string::npos)
        << "ROCm grouped verifier GDN must support padded verifier graphs.";
    EXPECT_NE(rocm_batched_route.find("!stream"), std::string::npos)
        << "ROCm grouped verifier GDN must reject the device-default stream.";
    EXPECT_TRUE(
        rocm_batched_route.find("static_cast<hipStream_t>(stream)") != std::string::npos ||
        rocm_batched_route.find("(hipStream_t)stream") != std::string::npos)
        << "ROCm grouped verifier GDN must launch on the explicit capture stream.";
    EXPECT_EQ(rocm_route.find("hipDeviceSynchronize"), std::string::npos)
        << "ROCm grouped verifier GDN must stay graph-capturable.";

    const auto rocm_body = stripCommentsAndStringLiterals(sliceBetween(
        rocm_source,
        "bool rocmGDN_chunk_forward(",
        "bool rocmGDN_chunk_forward_effective("));
    EXPECT_NE(rocm_body.find("rocmGDN_chunk_forward_kernel_route("), std::string::npos);
    EXPECT_NE(rocm_body.find("state_snapshots"), std::string::npos);
    EXPECT_NE(rocm_body.find("nullptr"), std::string::npos)
        << "Non-padded ROCm verifier chunks should use the same grouped route "
           "without an effective-length guard.";
    const auto rocm_effective_body = stripCommentsAndStringLiterals(sliceBetween(
        rocm_source,
        "bool rocmGDN_chunk_forward_effective(",
        "bool rocmGDN_short_conv1d("));
    EXPECT_NE(rocm_effective_body.find("rocmGDN_chunk_forward_kernel_route("), std::string::npos);
    EXPECT_NE(rocm_effective_body.find("device_effective_seq_len"), std::string::npos)
        << "ROCm padded verifier snapshots must be guarded by the device effective length.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDANativeVNNIDispatchSweepUsesExplicitStream)
{
    const auto source =
        readFile(repoRoot() / "tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp");

    EXPECT_NE(source.find("cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)"), std::string::npos)
        << "The CUDA NativeVNNI dispatch trainer must create an explicit non-blocking stream.";
    EXPECT_NE(source.find("kernel->setGPUStream(static_cast<void *>(stream))"), std::string::npos)
        << "The CUDA NativeVNNI dispatch trainer must bind its GEMM kernel to the explicit stream.";
    EXPECT_NE(source.find("kernel->setGPUStream(nullptr)"), std::string::npos)
        << "The CUDA NativeVNNI dispatch trainer must unbind the stream before leaving the run.";
    EXPECT_NE(source.find("cudaEventRecord(start, stream)"), std::string::npos)
        << "Benchmark timing must record the start event on the same explicit stream as GEMV.";
    EXPECT_NE(source.find("cudaEventRecord(stop, stream)"), std::string::npos)
        << "Benchmark timing must record the stop event on the same explicit stream as GEMV.";
    EXPECT_EQ(source.find("cudaEventRecord(start);"), std::string::npos)
        << "A missing stream argument records on the default stream and invalidates capture-sensitive timing.";
    EXPECT_EQ(source.find("cudaEventRecord(stop);"), std::string::npos)
        << "A missing stream argument records on the default stream and invalidates capture-sensitive timing.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDANativeVNNISmallMDispatchSweepUsesRealCandidates)
{
    const auto tuned_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu");
    const auto kernel_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp");
    const auto sweep_source =
        readFile(repoRoot() / "tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp");

    const auto small_m_body = sliceBetween(
        tuned_source,
        "bool dispatchCodebookSmallMRowPar(",
        "template <int M>");

    EXPECT_NE(small_m_body.find("if (g_sweep.active)"), std::string::npos)
        << "CUDA M=2..4 NativeVNNI trainer rows must time the requested candidate, not the generated runtime route.";
    EXPECT_NE(small_m_body.find("tuning = GeneratedDispatchTuning{"), std::string::npos)
        << "The small-M sweep override must forward tile/family parameters into the real launch.";

    EXPECT_NE(kernel_source.find("if (cudaNativeVNNIGemvSweep_isActive())"), std::string::npos)
        << "Small-M training sweeps must fail closed instead of falling through to generic M>1 GEMM.";

    EXPECT_NE(sweep_source.find("candidateSupportedByGpuPreparedSweepPath"), std::string::npos)
        << "The CUDA NativeVNNI trainer must filter candidate families unsupported by the GPU-prepared harness.";
    EXPECT_NE(sweep_source.find("if (m > 1)\n            return candidate.family == SweepFamily::KPar;"),
              std::string::npos)
        << "M=2..4 trainer cases must not label WIDE/DIRECT/ROWPAR rows as specialized small-M timings.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDANativeVNNIFusedVerifierRowsPinCapturedStream)
{
    const auto kernel_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp");
    const auto header_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.h");

    const auto fused_raw = sliceBetween(
        kernel_source,
        "bool CUDAQuantisedGemmKernel::multiply_fused_tensor_impl(",
        "bool CUDAQuantisedGemmKernel::multiply_q8_to_fp32(");
    const auto fused_body = stripCommentsAndStringLiterals(fused_raw);
    const auto fused_compact = removeAsciiWhitespace(fused_body);

    EXPECT_NE(fused_body.find("void *execution_stream = gpu_stream_;"), std::string::npos)
        << "Fused CUDA projection launches must snapshot the stage stream once.";
    EXPECT_NE(fused_raw.find("Fused CUDA projection launch requires an explicit CUDA stream"),
              std::string::npos)
        << "The fused path must fail hard instead of falling onto the CUDA default stream.";
    EXPECT_EQ(fused_compact.find("cuda_device_id_,gpu_stream_"), std::string::npos)
        << "Fused projection work must pass the captured stream, not reread gpu_stream_.";
    EXPECT_NE(fused_compact.find("multiply_quantized_m1_decode_gemv("
                                 "d_A_int8,d_scales_A_blockwise,d_output,d_bias,n,k,1.0f,0.0f,execution_stream)"),
              std::string::npos)
        << "Canonical M=1 verifier replay must receive the fused transaction stream.";
    EXPECT_NE(fused_compact.find("binding.kernel->multiply_quantized_small_m_gemv("
                                 "d_A_int8,d_scales_A_blockwise,binding.output,binding.bias,"
                                 "m,binding.n,k,1.0f,0.0f,true,execution_stream)"),
              std::string::npos)
        << "Grouped verifier fused projections must pass their captured stream.";

    const auto small_m_raw = sliceBetween(
        kernel_source,
        "bool CUDAQuantisedGemmKernel::multiply_quantized_small_m_gemv(",
        "bool CUDAQuantisedGemmKernel::multiply_quantized_m1_decode_gemv(");
    const auto small_m_body = stripCommentsAndStringLiterals(small_m_raw);
    EXPECT_NE(small_m_body.find("void *stream_handle = execution_stream ? execution_stream : gpu_stream_;"),
              std::string::npos);
    EXPECT_NE(small_m_raw.find("NativeVNNI small-M GEMV requires an explicit CUDA stream"),
              std::string::npos)
        << "Small-M GEMV helpers must reject null streams instead of using stream 0.";

    const auto m1_raw = sliceBetween(
        kernel_source,
        "bool CUDAQuantisedGemmKernel::multiply_quantized_m1_decode_gemv(",
        "bool CUDAQuantisedGemmKernel::multiply_fp32_to_fp32(");
    const auto m1_body = stripCommentsAndStringLiterals(m1_raw);
    const auto m1_compact = removeAsciiWhitespace(m1_body);
    EXPECT_NE(m1_body.find("void *stream_handle = execution_stream ? execution_stream : gpu_stream_;"),
              std::string::npos);
    EXPECT_NE(m1_raw.find("Canonical M=1 decode GEMV requires an explicit CUDA stream"),
              std::string::npos);
    EXPECT_NE(m1_compact.find("cudaNativeVNNIGemvTuned_fp32("
                              "d_A_int8,impl_->d_weights_native_vnni,impl_->d_weights_native_scales,"
                              "impl_->d_weights_native_mins,impl_->d_weights_native_emins,d_C,"
                              "d_scales_A_blockwise,n,k,alpha,beta,nullptr,d_bias,"
                              "impl_->native_codebook_id,cuda_device_id_,stream_handle,impl_->gemv_ctx,nullptr)"),
              std::string::npos)
        << "The M=1 canonical path must launch the serial decode GEMV on the captured stream without lazy ROWPAR state.";
    EXPECT_EQ(m1_compact.find("cudaMemsetAsync("), std::string::npos)
        << "Canonical M=1 decode must not pad into a synthetic small-M transaction.";
    EXPECT_EQ(m1_compact.find("cudaMemcpyAsync("), std::string::npos)
        << "Canonical M=1 decode must write directly to the caller's output row.";

    EXPECT_NE(header_source.find("LocalTP workers may reset request-scoped kernel dynamic"),
              std::string::npos)
        << "The private helper contract should document why the explicit stream "
           "parameter exists.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDANativeVNNIDispatchSweepUsesFastDeterministicWeights)
{
    const auto source =
        readFile(repoRoot() / "tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp");
    const auto format_table = sliceBetween(
        source,
        "const std::vector<FormatSpec> kFormats",
        "const NativeVnniFormatInfo &requireNativeVnniInfo");

    EXPECT_NE(source.find("createFastSweepTensor"), std::string::npos)
        << "CUDA NativeVNNI dispatch sweeps should use deterministic packed payloads.";
    EXPECT_EQ(format_table.find("TestTensorFactory::create"), std::string::npos)
        << "Format weights in the dispatch trainer must not use random quantizers; "
           "giant LM-head refreshes need O(weight_bytes) deterministic construction.";

    const auto run_body = sliceBetween(
        source,
        "RunResult runTunedGemv(",
        "DeviceId device_ = DeviceId::cpu();");
    EXPECT_EQ(run_body.find("makeGpuPreparedGemm("), std::string::npos)
        << "Candidate measurements must not rebuild/repack the GPU weight payload.";
    EXPECT_NE(source.find("workspaceBudgetFor(reqs)"), std::string::npos)
        << "The dispatch trainer workspace budget must come from declared kernel requirements.";
    EXPECT_EQ(source.find("512ull * 1024ull * 1024ull"), std::string::npos)
        << "A fixed 512 MiB trainer workspace cap breaks giant LM-head small-M sweeps.";
    EXPECT_NE(source.find("Prepare/upload/repack once per format+shape"), std::string::npos)
        << "The dispatch sweep should document why preparation sits outside the candidate loop.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPGpuSidecarsStageConditionTokensInArenaBuffer)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    const auto executable_sidecar_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_body));

    /*
     * GPU graph capture records the embedding kernel's token pointer.  Host
     * token arrays can be stack-backed and can change address between decode
     * steps, so every GPU sidecar path stages condition tokens into the
     * arena-owned MTP_CONDITION_TOKEN device buffer before updating dynamic
     * params.  Each sidecar graph cache owns a fixed slot in that buffer; a
     * single shared pointer would let full-sidecar and catch-up graph replay
     * overwrite each other's mutable token input on different capture streams.
     */
    EXPECT_NE(header.find("mtp_sidecar_condition_token_capacity_"), std::string::npos)
        << "The condition-token buffer must expose its row capacity to runtime validation.";
    EXPECT_NE(source.find("kMTPSidecarConditionTokenSlotCount"), std::string::npos)
        << "Graph-captured sidecar roles must have distinct condition-token slots.";
    EXPECT_NE(source.find("BufferId::MTP_CONDITION_TOKEN,\n"
                          "                                        1,\n"
                          "                                        sampling_math::kSpeculativeBatchMaxRows *\n"
                          "                                            kMTPSidecarConditionTokenSlotCount"),
              std::string::npos)
        << "MTP_CONDITION_TOKEN must hold all catch-up rows for every sidecar slot.";
    EXPECT_NE(sidecar_body.find("sidecar_condition_token_slot"), std::string::npos)
        << "MTP sidecar caches must map to role-owned condition-token slots.";
    EXPECT_NE(sidecar_body.find("condition_token_slot * kConditionTokenSlotWidth"), std::string::npos)
        << "Device-token staging must use the cache-owned slot offset, not the buffer base.";
    EXPECT_NE(sidecar_body.find("condition_token_device"), std::string::npos)
        << "Graph construction and token staging must share the same slot pointer.";
    EXPECT_NE(sidecar_body.find("stage_host_condition_tokens_on_device"), std::string::npos)
        << "GPU host-token sidecars must be promoted to the graph-safe device-token path.";
    EXPECT_NE(sidecar_body.find("backend->hostToDeviceOnStream"), std::string::npos)
        << "Host condition tokens must be staged on the explicit sidecar stream.";
    EXPECT_NE(sidecar_body.find("sidecar_cache.token_ids.data()"), std::string::npos)
        << "Async host staging must source from cache-owned stable storage, not stack token arrays.";
    EXPECT_NE(executable_sidecar_body.find(
                  "external_device_condition_tokens||prepare_device_condition_tokens_from_speculative_outcome"),
              std::string::npos)
        << "Resident speculative-outcome catch-up rows must not copy from nullable host draft tokens.";
    EXPECT_NE(executable_sidecar_body.find(
                  "std::fill(sidecar_cache.token_ids.begin(),sidecar_cache.token_ids.end(),0);"),
              std::string::npos)
        << "Speculative-outcome token staging must keep cache-owned host storage valid for async sidecars.";
    EXPECT_EQ(executable_sidecar_body.find(
                  "external_device_condition_tokens&&!prepare_device_condition_tokens_from_speculative_outcome&&total_rows!=1"),
              std::string::npos)
        << "Request-major device proposal columns must no longer be restricted to one row.";
    EXPECT_NE(executable_sidecar_body.find("compose_batched_device_inputs"),
              std::string::npos);
    EXPECT_NE(executable_sidecar_body.find(
                  "backend->enqueuePrepareMTPBatchedSidecarInputs("),
              std::string::npos)
        << "Multi-request device columns must be gathered into contiguous arena rows.";
    EXPECT_NE(sidecar_body.find("enqueuePrepareSpeculativeShiftedKVTokens"),
              std::string::npos)
        << "Multi-row shifted catch-up tokens must come from resident compact outcome metadata.";
    EXPECT_EQ(executable_sidecar_body.find("use_device_condition_tokens&&token_count!=1"),
              std::string::npos)
        << "Batched GPU catch-up sidecars must be allowed to use the device-token staging path.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPTargetDistributionBuildPreservesDeferredFirstTokenReadyEvent)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto build_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::buildStochasticDistributionOnDevice(",
        "bool DeviceGraphOrchestrator::sampleStochasticDistributionOnDeviceImpl(");
    const auto executable_build_body = stripCommentsAndStringLiterals(build_body);

    /*
     * The penalty-free vLLM-style stochastic path samples the first generated
     * token into STOCHASTIC_TARGET_SAMPLE_TOKENS[0], then reuses target
     * distribution slot 0 for all-position verifier row 0.  Distribution-slot
     * reuse must not clear the sampled-token ready event; otherwise the batch
     * summary can read that first token from another stream without waiting for
     * the sampler kernel.
     */
    EXPECT_EQ(
        executable_build_body.find("clearStochasticTargetSampleReadySlot"),
        std::string::npos)
        << "Building a target distribution must preserve deferred first-token readiness.";
    EXPECT_NE(
        executable_build_body.find("clearStochasticDraftSampleReadySlot("),
        std::string::npos)
        << "Draft distribution builds still clear draft sample readiness because "
           "draft distribution slots and draft sampled-token slots share one "
           "step-local producer/consumer pair.";
    EXPECT_NE(
        executable_build_body.find("StochasticSampleReadyClearMode::Force"),
        std::string::npos)
        << "Draft distribution builds overwrite the sampled-token slot, so the clear must not preserve verifier-owned state.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPSidecarRestampsDeferredTargetTokenForVerifierConsumer)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::forwardMTP(");
    const auto compact_sidecar =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_body));

    /*
     * The first generated token is intentionally kept device-resident in the
     * vLLM-style greedy path. It is consumed once by the first sidecar and once
     * by the verifier token-row materializer, so sidecar staging must publish a
     * new ready event instead of treating the target sample slot as
     * single-consumer scratch.
     */
    EXPECT_NE(compact_sidecar.find("draft_condition_ready_is_target"),
              std::string::npos);
    EXPECT_NE(
        compact_sidecar.find(
            "recordStochasticTargetSampleReady(draft_condition_ready_slot,sidecar_dynamic_stream)"),
        std::string::npos)
        << "Device-target sidecar staging must restamp the deferred first-token ready event for the verifier.";
    EXPECT_EQ(
        compact_sidecar.find(
            "recordStochasticDraftSampleReady(draft_condition_ready_slot,sidecar_dynamic_stream)"),
        std::string::npos)
        << "Draft sample slots are still single-consumer for chained sidecars.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GPUDecodeSamplingCannotFallbackToFullLogitsDownload)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto sampler_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "GreedyLogitCandidate sampleGreedyCandidateFromTensor(",
        "int coordinateGreedyCandidate(")));

    const size_t gpu_guard = sampler_body.find("if(gpu_resident_logits)");
    const size_t host_download =
        sampler_body.find("TransferEngine::instance().download(tensor)");

    ASSERT_NE(gpu_guard, std::string::npos)
        << "GPU-resident logits must be guarded before host fallback sampling.";
    ASSERT_NE(host_download, std::string::npos)
        << "CPU sampling fallback may still download CPU/host-only tensors.";
    EXPECT_LT(gpu_guard, host_download)
        << "GPU logits sampling failures must return before full logits D2H fallback.";
    EXPECT_NE(sampler_body.find("gpu_ptr_for_guard!=nullptr"),
              std::string::npos)
        << "The guard must trigger for tensors with a GPU pointer even if the "
           "coherence device metadata is stale.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, DeferredSampleReadinessPreservesVerifierOwnedSlots)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto compact_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(source));
    const auto clear_target_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearStochasticTargetSampleReadySlot(",
        "void DeviceGraphOrchestrator::clearStochasticTargetSampleReadySlots(")));
    const auto clear_draft_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearStochasticDraftSampleReadySlot(",
        "void DeviceGraphOrchestrator::clearStochasticDraftSampleReadySlots(")));
    const auto sync_deferral_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "void DeviceGraphOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled)",
        "void DeviceGraphOrchestrator::setMTPMainDecodeSyncDeferralEnabled(bool enabled)")));
    const auto greedy_first_token_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::sampleGreedyFromMainLogitsToDeviceTargetSlot(",
        "int DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDevice(")));
    const auto greedy_outcome_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(",
        "bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDevice(")));
    const auto set_verifier_plan_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::setMTPSpecVerifierInputPlan(",
        "void DeviceGraphOrchestrator::clearMTPSpecVerifierInputPlan(")));
    const auto clear_verifier_plan_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearMTPSpecVerifierInputPlan(",
        "bool DeviceGraphOrchestrator::materializePendingMTPVerifierInputTokensOnDevice(")));
    const auto device_first_plan_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(",
        "const float *DeviceGraphOrchestrator::forwardImpl(")));

    EXPECT_NE(compact_header.find("boolverifier_consumer_pending=false"),
              std::string::npos)
        << "Deferred sample readiness must encode verifier transaction ownership.";
    EXPECT_NE(clear_target_body.find(
                  "mode==StochasticSampleReadyClearMode::PreserveVerifierConsumer&&ready.verifier_consumer_pending"),
              std::string::npos)
        << "Generic target-slot cleanup must preserve verifier-owned first-token samples.";
    EXPECT_NE(clear_draft_body.find(
                  "mode==StochasticSampleReadyClearMode::PreserveVerifierConsumer&&ready.verifier_consumer_pending"),
              std::string::npos)
        << "Generic draft-slot cleanup must preserve verifier-owned draft samples.";
    EXPECT_NE(sync_deferral_body.find("clearStochasticTargetSampleReadySlots()"),
              std::string::npos)
        << "All-position sync deferral cleanup should use preserving target-slot cleanup.";
    EXPECT_NE(sync_deferral_body.find("clearStochasticDraftSampleReadySlots()"),
              std::string::npos)
        << "All-position sync deferral cleanup should use preserving draft-slot cleanup.";
    EXPECT_NE(greedy_first_token_body.find(
                  "recordStochasticTargetSampleReady(target_sample_slot,stream,out_token==nullptr)"),
              std::string::npos)
        << "Deferred greedy first-token sampling must mark the target slot as verifier-owned.";
    EXPECT_NE(set_verifier_plan_body.find("pending_mtp_verifier_device_token_plan_.reset()"),
              std::string::npos)
        << "Installing a verifier row plan must drop any stale token-row copy plan.";
    EXPECT_NE(device_first_plan_body.find("!first_token_ready.valid"),
              std::string::npos)
        << "A device-first verifier row must fail before graph replay if its target token slot is not ready.";
    EXPECT_EQ(clear_verifier_plan_body.find("materialized_mtp_verifier_device_token_row_={}"),
              std::string::npos)
        << "Verifier metadata RAII cleanup runs before the compact outcome reducer, so it must not erase the materialized token row.";
    EXPECT_NE(greedy_outcome_body.find(
                  "clearStochasticTargetSampleReadySlot(materialized_first_target_sample_slot,StochasticSampleReadyClearMode::Force)"),
              std::string::npos)
        << "Device-resident greedy outcome consumption must force-clear the verifier-owned target slot.";
    EXPECT_NE(greedy_outcome_body.find("materialized_mtp_verifier_device_token_row_={}"),
              std::string::npos)
        << "The greedy reducer owns the materialized token row lifetime after verifier graph replay.";
    EXPECT_NE(compact_source.find(
                  "ready.verifier_consumer_pending=ready.verifier_consumer_pending||verifier_consumer_pending"),
              std::string::npos)
        << "Restamping a ready event must preserve an existing verifier-owner bit.";
    EXPECT_NE(compact_source.find("pending_mtp_verifier_device_token_plan_.reset()"),
              std::string::npos)
        << "Request-boundary clear_cache() must drop verifier token-row plans.";
    EXPECT_NE(compact_source.find("materialized_mtp_verifier_device_token_row_={}"),
              std::string::npos)
        << "Request-boundary clear_cache() must drop materialized verifier token rows.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, ClearCacheDropsStochasticDistributionSlotMetadata)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto reset_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override")));

    /*
     * Regression guard for seeded stochastic MTP reproducibility after
     * clearCache(): target/draft distribution slots are request-local metadata,
     * not persistent graph topology.  Leaving top-k values alive lets a later
     * request observe stale stochastic slots even after streams and row formats
     * were cleared.
     */
    const std::string clear_target_top_k =
        "std::fill(stochastic_target_top_k_.begin(),stochastic_target_top_k_.end(),0);";
    const std::string clear_draft_top_k =
        "std::fill(stochastic_draft_top_k_.begin(),stochastic_draft_top_k_.end(),0);";
    const std::string clear_ready =
        "clearStochasticTargetSampleReadySlots(StochasticSampleReadyClearMode::Force);";

    EXPECT_NE(reset_body.find(clear_target_top_k), std::string::npos)
        << "clear_cache() must reset request-local target stochastic top-k metadata.";
    EXPECT_NE(reset_body.find(clear_draft_top_k), std::string::npos)
        << "clear_cache() must reset request-local draft stochastic top-k metadata.";
    expectNeedleBefore(
        reset_body,
        clear_target_top_k,
        clear_ready,
        "clear_cache() must fully empty stochastic distribution slots before ready-event cleanup.");
    expectNeedleBefore(
        reset_body,
        clear_draft_top_k,
        clear_ready,
        "clear_cache() must fully empty stochastic distribution slots before ready-event cleanup.");
}

TEST(Test__GpuWorkspaceAllocationPolicy, ClearCachePreservesReplaySafeMTPGraphCaptures)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto reset_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override")));
    const auto clear_cache_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        header,
        "void clear_cache() override",
        "/**\n         * @brief Get current position")));
    const auto clear_cache_body_with_strings = removeAsciiWhitespace(sliceBetween(
        header,
        "void clear_cache() override",
        "/**\n         * @brief Get current position"));

    /*
     * Request boundaries should clear live KV/GDN/session state without forcing
     * every production request to pay GPU graph warmup/capture again.  Forward
     * engine policy still resets prefill and live-state-versioned multi-row
     * decode captures; this guard makes sure clear_cache() opts into preserving
     * the replay-safe single-token decode and all-position verifier classes.
     */
    EXPECT_NE(clear_cache_body_with_strings.find(
                  "resetInferenceState(InferenceStateResetRequest::requestBoundary(\"clear_cache\"));"),
              std::string::npos)
        << "clear_cache() must enter the typed request-boundary reset path.";
    EXPECT_NE(reset_body.find(
                  "!request.reset_model_runtime||!request.preserve_replay_safe_graphs"),
              std::string::npos)
        << "The request boundary must require replay-safe graph preservation.";
    EXPECT_NE(reset_body.find(
                  "forward_engine_->resetSessionReplayState(preserve_replay_safe_graphs);"),
              std::string::npos)
        << "clear_cache() must preserve replay-safe cached forward graph captures.";
    EXPECT_NE(reset_body.find(
                  "mtp_sidecar_depth0_cache_.resetSessionStatePreservingGraphReplay();"),
              std::string::npos)
        << "The ordinary MTP sidecar cache should stay replay-hot across requests.";
    EXPECT_NE(reset_body.find(
                  "mtp_sidecar_depth0_device_token_cache_.resetSessionStatePreservingGraphReplay();"),
              std::string::npos)
        << "Device-token sidecar replay is the served stochastic path and must not recapture every request.";
    EXPECT_NE(reset_body.find(
                  "cache.resetSessionStatePreservingGraphReplay();"),
              std::string::npos)
        << "Batched KV-only sidecar caches must use the same replay-preserving request reset.";
    EXPECT_NE(reset_body.find(
                  "recordKernelDynamicStatePreservedForCapturedReplay(reset_reason);"),
              std::string::npos)
        << "Request-boundary kernel-state preservation must remain visible in perf counters.";
    const auto reset_body_with_strings = removeAsciiWhitespace(sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override"));
    EXPECT_NE(reset_body_with_strings.find(
                  "recordLivePrefixSessionReset(reset_reason,preserve_replay_safe_graphs);"),
              std::string::npos)
        << "Live-prefix request-boundary telemetry must report replay/kernel "
           "state as preserved, not as a hard session reset.";

    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto record_reset_body = removeAsciiWhitespace(sliceBetween(
        source,
        "void DeviceGraphOrchestrator::recordLivePrefixSessionReset(",
        "void DeviceGraphOrchestrator::resetMTPSidecarDepth0ReplayState()"));
    EXPECT_NE(record_reset_body.find("if(preserve_gpu_replay_state)"),
              std::string::npos);
    EXPECT_NE(record_reset_body.find("request_boundary_preserve"),
              std::string::npos)
        << "Request-boundary reset perf counters should explain why replay stayed hot.";
    EXPECT_NE(record_reset_body.find("tags[\"kernel_dynamic_state\"]=\"preserved\";"),
              std::string::npos);
    EXPECT_NE(record_reset_body.find("tags[\"kernel_dynamic_state\"]=\"reset\";"),
              std::string::npos)
        << "Hard inference-state clears still need reset telemetry.";

    const auto clear_inference_body = removeAsciiWhitespace(sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearInferenceState()",
        "// ========================================================================="));
    EXPECT_NE(clear_inference_body.find("recordLivePrefixSessionReset(\"clearInferenceState\")"),
              std::string::npos)
        << "clearInferenceState() is still a hard state reset.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, PrefixRestoreWithoutModelRuntimeInvalidatesMTPSidecarGraphs)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto reset_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override")));
    const auto invalidation_body = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "void DeviceGraphOrchestrator::invalidateMTPSidecarDepth0GraphState(",
        "void DeviceGraphOrchestrator::resetMTPSidecarDepth0ReplayState()")));
    const auto invalidation_body_with_strings = removeAsciiWhitespace(sliceBetween(
        source,
        "void DeviceGraphOrchestrator::invalidateMTPSidecarDepth0GraphState(",
        "void DeviceGraphOrchestrator::resetMTPSidecarDepth0ReplayState()"));

    /*
     * Regression guard for prefix-cache + MTP + MoE LocalTP: a restored prefix
     * block may contain live KV/GDN/MTP payloads but no graph-owned
     * model-runtime placement snapshot.  That boundary resets MoE runtime
     * tables, so cached MTP sidecar graphs must be destroyed and rebuilt rather
     * than merely resetting replay handles while leaving stale stage metadata.
     */
    EXPECT_NE(compact_header.find("voidinvalidateMTPSidecarDepth0GraphState(constchar*reason);"),
              std::string::npos)
        << "MTP sidecar graph invalidation must remain a named coherence operation.";
    EXPECT_NE(reset_body.find(
                  "constboolprefix_restore_resets_model_runtime_owner=prefix_restore_boundary&&request.reset_model_runtime;"),
              std::string::npos)
        << "Prefix restore needs an explicit no-model-runtime-snapshot predicate.";
    EXPECT_NE(reset_body.find(
                  "elseif(prefix_restore_resets_model_runtime_owner){invalidateMTPSidecarDepth0GraphState(reset_reason);}"),
              std::string::npos)
        << "Prefix restore without model-runtime state must invalidate sidecar graphs.";
    expectNeedleBefore(
        reset_body,
        "invalidateMTPSidecarDepth0GraphState(reset_reason);",
        "mtp_sidecar_depth0_cache_.resetSessionState();",
        "The no-snapshot prefix restore branch must bypass stale-graph resetSessionState().");
    EXPECT_NE(reset_body.find(
                  "elseif(!prefix_restore_resets_model_runtime_owner)cache.resetSessionState();"),
              std::string::npos)
        << "Batched KV-only MTP sidecar caches must not survive the no-snapshot restore branch.";

    EXPECT_NE(invalidation_body.find("mtp_sidecar_depth0_device_token_cache_.invalidate();"),
              std::string::npos)
        << "Device-token sidecars were the observed stale cache hit and must be destroyed.";
    EXPECT_NE(invalidation_body.find("mtp_sidecar_depth0_kv_only_device_token_cache_.invalidate();"),
              std::string::npos)
        << "KV-only device-token sidecars follow the same runtime-table lifetime.";
    EXPECT_NE(invalidation_body.find("cache.invalidate();"), std::string::npos)
        << "Batched KV-only sidecar caches must be invalidated as a group.";
    EXPECT_NE(invalidation_body_with_strings.find(
                  "PerfStatsCollector::addCounter(\"mtp\",\"sidecar_graph_invalidations\""),
              std::string::npos)
        << "Perfstats must expose that prefix restore invalidated MTP sidecar graphs.";
    EXPECT_NE(invalidation_body_with_strings.find("tags[\"reason\"]=reason?reason:\"unknown\";"),
              std::string::npos)
        << "The invalidation counter should identify the reset boundary.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, DeviceResidentHostAdoptionApiIsRetired)
{
    const auto runner_interface =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto dgo_header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto dgo_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto rank_header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/RankOrchestrator.h");
    const auto rank_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
    const std::string production =
        runner_interface + dgo_header + dgo_source + rank_header + rank_source;
    const auto compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(production));

    EXPECT_EQ(compact.find("DeviceResidentHostStateAdoptionRequest"),
              std::string::npos)
        << "Resident MTP must not expose a host-adoption request object.";
    EXPECT_EQ(compact.find("adoptDeviceResidentMTPSpecPublishedHostState("),
              std::string::npos)
        << "Resident MTP publication must not expose a host-adoption method.";
    EXPECT_EQ(compact.find("device_resident_host_state_adoptions"),
              std::string::npos)
        << "Perf counters should not describe a retired adoption path.";
    EXPECT_EQ(compact.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "A resident mailbox is the authority; DGO must not retain a second host-mirror epoch.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPStochasticTopKPartialScratchIsSplitByStreamDomain)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto buffer_ids =
        readFile(repoRoot() / "src/v2/memory/BufferId.h");
    const auto processed_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::buildStochasticProcessedLogitRowsOnDevice(",
        "bool DeviceGraphOrchestrator::buildStochasticDistributionOnDevice(");
    const auto distribution_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::buildStochasticDistributionOnDevice(",
        "bool DeviceGraphOrchestrator::buildStochasticDistributionsOnDevice(");
    const auto compact_processed =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(processed_body));
    const auto compact_distribution =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(distribution_body));

    /*
     * MTP draft distributions and target/verifier distributions can be
     * produced on different graph-capture streams.  Sharing the same top-k
     * partial workspace makes the stochastic verifier timing-sensitive: a GPU
     * event or debug synchronization can accidentally serialize two producers
     * and hide the race.  Keep these arena buffers structurally separate.
     */
    EXPECT_NE(buffer_ids.find("STOCHASTIC_TOPK_PARTIAL_VALS"), std::string::npos);
    EXPECT_NE(buffer_ids.find("STOCHASTIC_DRAFT_TOPK_PARTIAL_VALS"), std::string::npos);
    EXPECT_NE(buffer_ids.find("STOCHASTIC_DRAFT_TOPK_PARTIAL_IDXS"), std::string::npos);
    EXPECT_NE(header.find("stochastic_draft_topk_partial_vals_dev_"), std::string::npos);
    EXPECT_NE(header.find("stochastic_draft_topk_partial_idxs_dev_"), std::string::npos);
    EXPECT_NE(source.find("BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_VALS"), std::string::npos);
    EXPECT_NE(source.find("BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_IDXS"), std::string::npos);

    const std::string selector =
        "buffer==DeviceDistributionBuffer::Target?stochastic_topk_partial_vals_dev_:stochastic_draft_topk_partial_vals_dev_";
    EXPECT_NE(compact_processed.find(selector), std::string::npos)
        << "Processed-logit builds must choose target or draft partial scratch by destination buffer.";
    EXPECT_NE(compact_distribution.find(selector), std::string::npos)
        << "Compact distribution builds must choose target or draft partial scratch by destination buffer.";
    EXPECT_NE(compact_processed.find("partial_vals,partial_idxs,partial_capacity"),
              std::string::npos)
        << "Processed-logit builds must pass the selected scratch to the backend.";
    EXPECT_NE(compact_distribution.find("partial_vals,partial_idxs,partial_capacity"),
              std::string::npos)
        << "Compact distribution builds must pass the selected scratch to the backend.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPStochasticVerifierDoesNotAllocateLegacyFullProbabilityRows)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto buffer_ids =
        readFile(repoRoot() / "src/v2/memory/BufferId.h");
    const auto build_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::buildStochasticDistributionOnDevice(",
        "bool DeviceGraphOrchestrator::sampleStochasticDistributionOnDeviceImpl(");
    const auto executable_build_body = stripCommentsAndStringLiterals(build_body);

    /*
     * The vLLM-style stochastic path keeps draft proposals as sampled tokens plus
     * q(token), and verifies target rows through processed logits. Reintroducing
     * full-vocab target/draft probability rows would allocate several extra
     * vocab-sized buffers per GPU runner and revive the removed scalar verifier.
     */
    EXPECT_EQ(buffer_ids.find("STOCHASTIC_TARGET_FULL_PROBS"), std::string::npos);
    EXPECT_EQ(buffer_ids.find("STOCHASTIC_DRAFT_FULL_PROBS"), std::string::npos);
    EXPECT_EQ(header.find("stochastic_target_full_probs_dev_"), std::string::npos);
    EXPECT_EQ(header.find("stochastic_draft_full_probs_dev_"), std::string::npos);
    EXPECT_EQ(source.find("BufferId::STOCHASTIC_TARGET_FULL_PROBS"), std::string::npos);
    EXPECT_EQ(source.find("BufferId::STOCHASTIC_DRAFT_FULL_PROBS"), std::string::npos);
    EXPECT_EQ(executable_build_body.find("buildStochasticProbabilityRowsOnDevice"),
              std::string::npos)
        << "Compact stochastic distribution builds must not materialize full "
           "softmax rows as a hidden side effect.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPStochasticBatchOutcomeCopiesOnlySemanticRowsByDefault)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(",
        "    // =========================================================================\n"
        "    // Batch Interface Implementation");
    const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(body));

    /*
     * The compact stochastic outcome has exactly one required host-visible
     * boundary: output tokens plus summary metadata. Per-row probabilities,
     * thresholds, and draft-token details are debug aids only.  Keeping those
     * copies behind copy_summary_to_host and capture_row_debug prevents
     * request-batched decode from adding per-request stream synchronizations on
     * ROCm/CUDA production runs.
     */
    EXPECT_NE(compact.find("constboolcapture_row_debug="), std::string::npos);
    const size_t debug_gate =
        compact.find("if(copy_summary_to_host&&copied_summary&&capture_row_debug){");
    ASSERT_NE(debug_gate, std::string::npos)
        << "Debug-only stochastic batch outcome copies must be gated.";

    const std::vector<std::string> debug_copies = {
        "deviceToHostFast(debug_accept_probs.data()",
        "deviceToHostFast(debug_accept_thresholds.data()",
        "deviceToHostFast(debug_draft_tokens.data()",
        "deviceToHostFast(debug_draft_probs.data()",
    };
    for (const auto &needle : debug_copies)
    {
        const size_t copy = compact.find(needle);
        ASSERT_NE(copy, std::string::npos) << needle;
        EXPECT_GT(copy, debug_gate)
            << needle << " must stay behind capture_row_debug.";
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPResidentOutcomeHostBridgeQueuesD2HBeforeOneSync)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::copyDeviceSpeculativeOutcomesToHost(",
        "    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(");
    const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(body));

    /*
     * The resident stochastic outcome bridge is still a host-visible boundary,
     * but it should enqueue the token and metadata copies on a response bridge
     * stream and then synchronize once.  Multiple deviceToHostFast() calls, or
     * synchronizing the verifier producer stream, would serialize the boundary
     * and make ROCm/CUDA wait more than needed.
     */
    EXPECT_EQ(compact.find("deviceToHostFast("), std::string::npos);
    EXPECT_NE(compact.find("ScopedTimertotal_timer("), std::string::npos);
    EXPECT_NE(compact.find("ScopedTimerenqueue_timer("), std::string::npos);
    EXPECT_NE(compact.find("ScopedTimerwait_timer("), std::string::npos);
    EXPECT_NE(compact.find("stochastic_batch_output_host_scratch_"),
              std::string::npos);
    EXPECT_NE(compact.find("deviceToHostOnStream(output_tokens,"),
              std::string::npos);
    EXPECT_NE(compact.find("deviceToHostOnStream(meta,"),
              std::string::npos);
    const size_t total_timer =
        compact.find("ScopedTimertotal_timer(");
    const size_t enqueue_timer =
        compact.find("ScopedTimerenqueue_timer(");
    const size_t wait_timer =
        compact.find("ScopedTimerwait_timer(");
    const size_t first_copy =
        compact.find("deviceToHostOnStream(output_tokens,");
    const size_t second_copy =
        compact.find("deviceToHostOnStream(meta,");
    const size_t sync = compact.find("synchronizeStream(copy_stream");
    ASSERT_NE(total_timer, std::string::npos);
    ASSERT_NE(enqueue_timer, std::string::npos);
    ASSERT_NE(wait_timer, std::string::npos);
    ASSERT_NE(first_copy, std::string::npos);
    ASSERT_NE(second_copy, std::string::npos);
    ASSERT_NE(sync, std::string::npos);
    EXPECT_EQ(compact.find("synchronizeStream(handle.stream"),
              std::string::npos);
    EXPECT_LT(total_timer, enqueue_timer);
    EXPECT_LT(enqueue_timer, first_copy);
    EXPECT_LT(first_copy, sync);
    EXPECT_LT(second_copy, sync);
    EXPECT_LT(wait_timer, sync);
    EXPECT_LT(first_copy, wait_timer);
    EXPECT_LT(second_copy, wait_timer);
}

TEST(Test__GpuWorkspaceAllocationPolicy, PrefixSnapshotsObserveAcceptedSpecPublicationWithoutConsumingIt)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto probe_body = sliceBetween(
        source,
        "PrefixRuntimeStateSnapshot DeviceGraphOrchestrator::prefixStateProbe() const",
        "void DeviceGraphOrchestrator::disablePrefixCacheForRunner");
    const auto payload_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixState(int seq_idx) const",
        "bool DeviceGraphOrchestrator::ensureLiveHybridCheckpointStorage");
    const auto checkpoint_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(int seq_idx) const",
        "bool DeviceGraphOrchestrator::restoreLivePrefixState");
    const auto live_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForLiveInferenceStateReadyForObservation(",
        "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(");
    const auto observation_wait_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingAcceptedSpecPublicationReadyForObservation(",
        "void DeviceGraphOrchestrator::clearPendingAcceptedSpecPublicationReady()");

    const auto compact_probe =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(probe_body));
    const auto compact_payload =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(payload_body));
    const auto compact_checkpoint =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(checkpoint_body));
    const auto compact_live_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(live_observation_body));
    const auto compact_observation_wait =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(observation_wait_body));
    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));

    EXPECT_NE(compact_header.find(
                  "waitForLiveInferenceStateReadyForObservation("
                  "void*observation_stream,constchar*observation_name)const"),
              std::string::npos)
        << "Host-visible live-state exports must use the shared observation boundary.";
    EXPECT_NE(compact_header.find(
                  "waitForPendingAcceptedSpecPublicationReadyForObservation("
                  "void*consumer_stream,constchar*consumer_name)const"),
              std::string::npos)
        << "Snapshot/probe observation waits must stay const and non-consuming.";
    EXPECT_NE(compact_observation_wait.find("accepted_spec_publication_ready_"),
              std::string::npos);
    EXPECT_NE(compact_observation_wait.find("streamWaitEvent("),
              std::string::npos);
    EXPECT_EQ(compact_observation_wait.find("ready.valid=false"),
              std::string::npos)
        << "Diagnostic snapshot/probe waits must not consume the publication event.";
    EXPECT_EQ(compact_observation_wait.find("accepted_spec_publication_ready_.valid=false"),
              std::string::npos)
        << "Diagnostic observation must leave the real forward handoff intact.";

    EXPECT_NE(compact_live_observation.find(
                  "waitForPendingAcceptedSpecPublicationReadyForObservation("),
              std::string::npos);
    EXPECT_NE(compact_live_observation.find(
                  "waitForPendingLivePrefixMutationReadyForObservation("),
              std::string::npos);
    EXPECT_NE(compact_live_observation.find(
                  "waitForPendingLiveGraphProducersForObservation("),
              std::string::npos);
    EXPECT_NE(compact_live_observation.find(
                  "waitForDeviceResidentLogicalSequenceStateMailboxForObservation("),
              std::string::npos);

    EXPECT_NE(compact_probe.find(
                  "waitForLiveInferenceStateReadyForObservation("),
              std::string::npos);
    EXPECT_NE(probe_body.find("\"prefix_state_probe\""),
              std::string::npos);
    EXPECT_NE(compact_payload.find(
                  "waitForLiveInferenceStateReadyForObservation("),
              std::string::npos);
    EXPECT_NE(payload_body.find("\"capture_live_prefix_state\""),
              std::string::npos);
    EXPECT_NE(compact_checkpoint.find(
                  "waitForLiveInferenceStateReadyForObservation("),
              std::string::npos);
    EXPECT_NE(checkpoint_body.find("\"capture_live_prefix_checkpoint\""),
              std::string::npos);

    expectNeedleBefore(
        compact_probe,
        "waitForLiveInferenceStateReadyForObservation(",
        "PrefixRuntimeStateSnapshotsnapshot;",
        "prefixStateProbe must order after all live inference-state producers before reading probes.");
    expectNeedleBefore(
        compact_payload,
        "waitForLiveInferenceStateReadyForObservation(",
        "snapshot.valid=true;",
        "captureLivePrefixState must order after all live inference-state producers before exporting payloads.");
    expectNeedleBefore(
        compact_checkpoint,
        "waitForLiveInferenceStateReadyForObservation(",
        "constintdraft_tokens=",
        "captureLivePrefixCheckpoint must order after all live inference-state producers before reading live metadata.");
}

TEST(Test__GpuWorkspaceAllocationPolicy, LiveStateObservationsWaitForGraphProducersWithoutConsumingThem)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");

    const auto logits_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLogitsStreamForObservation(",
        "void *DeviceGraphOrchestrator::peekPendingLogitsStream(");
    const auto graph_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLiveGraphProducersForObservation(",
        "void DeviceGraphOrchestrator::clearMTPVerifierTransactionStateForBoundary(");
    const auto shifted_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingShiftedMTPKVReadyForObservation(",
        "bool DeviceGraphOrchestrator::recordAllPositionVerifierStateReady(");
    const auto verifier_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingAllPositionVerifierStateReadyForObservation(",
        "void DeviceGraphOrchestrator::clearPendingAllPositionVerifierStateReady()");
    const auto mailbox_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateMailboxForObservation(",
        "bool DeviceGraphOrchestrator::retargetDeviceResidentLogicalSequenceStateMailboxAfterShiftedKVMutation(");
    const auto harvest_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::harvestPrefix(",
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixState(int seq_idx) const");

    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto compact_logits_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(logits_observation_body));
    const auto compact_graph_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(graph_observation_body));
    const auto compact_shifted_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(shifted_observation_body));
    const auto compact_verifier_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(verifier_observation_body));
    const auto compact_mailbox_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mailbox_observation_body));
    const auto compact_harvest =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(harvest_body));

    EXPECT_NE(compact_header.find("waitForPendingLogitsStreamForObservation("),
              std::string::npos);
    EXPECT_NE(compact_header.find("waitForPendingLiveGraphProducersForObservation("),
              std::string::npos);
    EXPECT_NE(compact_header.find("waitForPendingShiftedMTPKVReadyForObservation("),
              std::string::npos);
    EXPECT_NE(compact_header.find("waitForPendingAllPositionVerifierStateReadyForObservation("),
              std::string::npos);
    EXPECT_NE(compact_header.find("waitForDeviceResidentLogicalSequenceStateMailboxForObservation("),
              std::string::npos);

    EXPECT_NE(compact_logits_observation.find("peekPendingLogitsStream(role)"),
              std::string::npos)
        << "Observation must inspect the logits handoff without taking ownership.";
    EXPECT_EQ(compact_logits_observation.find("consumePendingLogitsStream("),
              std::string::npos)
        << "Observation must not steal sampler/reducer stream ownership.";
    EXPECT_NE(compact_logits_observation.find("insertStreamDependency("),
              std::string::npos);

    EXPECT_NE(compact_graph_observation.find("PendingLogitsStreamRole::MTPSidecar"),
              std::string::npos);
    EXPECT_NE(compact_graph_observation.find("PendingLogitsStreamRole::MainDecode"),
              std::string::npos);
    EXPECT_NE(compact_graph_observation.find("PendingLogitsStreamRole::AllPositionVerifier"),
              std::string::npos);
    EXPECT_NE(compact_graph_observation.find("waitForPendingShiftedMTPKVReadyForObservation("),
              std::string::npos);
    EXPECT_NE(compact_graph_observation.find("waitForPendingAllPositionVerifierStateReadyForObservation("),
              std::string::npos);
    EXPECT_EQ(compact_graph_observation.find("waitForPendingLogitsStream("),
              std::string::npos)
        << "Observation helper must use the non-consuming logits wait.";
    EXPECT_EQ(compact_graph_observation.find("waitForPendingShiftedMTPKVReady("),
              std::string::npos)
        << "Observation helper must not consume shifted-MTP-KV ownership.";
    EXPECT_EQ(compact_graph_observation.find("waitForPendingAllPositionVerifierStateReady("),
              std::string::npos)
        << "Observation helper must not consume verifier row-state ownership.";

    EXPECT_NE(compact_shifted_observation.find("streamWaitEvent("),
              std::string::npos);
    EXPECT_EQ(compact_shifted_observation.find("ready.valid=false"),
              std::string::npos);
    EXPECT_NE(compact_verifier_observation.find("streamWaitEvent("),
              std::string::npos);
    EXPECT_EQ(compact_verifier_observation.find("clearPendingAllPositionVerifierStateReady()"),
              std::string::npos);
    EXPECT_NE(compact_mailbox_observation.find("streamWaitEvent("),
              std::string::npos);
    EXPECT_EQ(compact_mailbox_observation.find("mailbox.clear()"),
              std::string::npos)
        << "Diagnostic observation must not adopt or clear device-resident logical state.";

    EXPECT_NE(compact_harvest.find("waitForLiveInferenceStateReadyForObservation("),
              std::string::npos)
        << "Prefix-cache harvest is a host-visible export and must use the shared observation boundary.";
    EXPECT_EQ(compact_harvest.find("waitForPendingLiveGraphProducersBeforePrefixMutation("),
              std::string::npos)
        << "Harvest must not consume graph-producer handoffs just to export payloads.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, LiveLogicalCheckpointsUseEventBackedSourceAndSnapshotHandoffs)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto snapshot_header =
        readFile(repoRoot() / "src/v2/execution/prefix_cache/PrefixStateSnapshot.h");
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto compact_header = removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto compact_snapshot = removeAsciiWhitespace(stripCommentsAndStringLiterals(snapshot_header));
    const auto capture_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(",
        "bool DeviceGraphOrchestrator::restoreLivePrefixState(");
    const auto restore_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::restoreLivePrefixState(",
        "bool DeviceGraphOrchestrator::truncateLivePrefixState(");
    const auto truncate_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::truncateLivePrefixState(",
        "bool DeviceGraphOrchestrator::mtpSpecStatePublicationRequiresCapturedStage(");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(",
        "const float *DeviceGraphOrchestrator::getAllPositionLogits()");
    const auto record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordLivePrefixCheckpointReady(",
        "bool DeviceGraphOrchestrator::waitForSnapshotReady(");
    const auto snapshot_wait_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForSnapshotReady(",
        "bool DeviceGraphOrchestrator::waitForPendingLivePrefixCheckpointReady(");
    const auto source_wait_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLivePrefixCheckpointReady(",
        "void DeviceGraphOrchestrator::clearPendingLivePrefixCheckpointReady()");

    const auto compact_capture = removeAsciiWhitespace(stripCommentsAndStringLiterals(capture_body));
    const auto compact_restore = removeAsciiWhitespace(stripCommentsAndStringLiterals(restore_body));
    const auto compact_truncate = removeAsciiWhitespace(stripCommentsAndStringLiterals(truncate_body));
    const auto compact_prepare = removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_record = removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));
    const auto compact_snapshot_wait = removeAsciiWhitespace(stripCommentsAndStringLiterals(snapshot_wait_body));
    const auto compact_source_wait = removeAsciiWhitespace(stripCommentsAndStringLiterals(source_wait_body));

    EXPECT_NE(compact_snapshot.find("std::shared_ptr<void>ready_event;"), std::string::npos);
    EXPECT_NE(compact_snapshot.find("void*ready_producer_stream=nullptr;"), std::string::npos);
    EXPECT_NE(compact_snapshot.find("DeviceIdready_device=DeviceId::invalid();"), std::string::npos);
    EXPECT_NE(compact_snapshot.find("boolready_event_valid=false;"), std::string::npos);
    EXPECT_NE(compact_snapshot.find("ready_event.swap(other.ready_event);"), std::string::npos)
        << "PrefixStateSnapshot move/swap must preserve event ownership.";

    EXPECT_NE(compact_header.find("PendingLivePrefixCheckpointReadyState"), std::string::npos);
    EXPECT_NE(compact_header.find("mutablePendingLivePrefixCheckpointReadyStatelive_prefix_checkpoint_ready_;"),
              std::string::npos);
    EXPECT_NE(compact_header.find("recordLivePrefixCheckpointReady(PrefixStateSnapshot*snapshot,void*producer_stream,constchar*producer_name)const"),
              std::string::npos);
    EXPECT_NE(compact_header.find("waitForSnapshotReady(constPrefixStateSnapshot&snapshot,void*consumer_stream,constchar*consumer_name)const"),
              std::string::npos);

    EXPECT_NE(compact_record.find("backend->createEvent("), std::string::npos);
    EXPECT_NE(compact_record.find("backend->recordEvent("), std::string::npos);
    EXPECT_NE(compact_record.find("snapshot->ready_event=event;"), std::string::npos);
    EXPECT_NE(compact_record.find("live_prefix_checkpoint_ready_.event=std::move(event);"),
              std::string::npos);
    EXPECT_NE(compact_snapshot_wait.find("streamWaitEvent("), std::string::npos);
    EXPECT_NE(compact_source_wait.find("streamWaitEvent("), std::string::npos);
    EXPECT_NE(compact_source_wait.find("clearPendingLivePrefixCheckpointReady();"), std::string::npos);

    EXPECT_NE(compact_capture.find("&handle,false,stream)"), std::string::npos)
        << "Hybrid logical checkpoints should stay async and be ordered by events, not host syncs.";
    EXPECT_NE(compact_capture.find("queued_async_device_checkpoint_payload"), std::string::npos);
    EXPECT_NE(compact_capture.find(
                  "constbooldevice_only_checkpoint=state_.device_id.is_gpu();"),
              std::string::npos)
        << "GPU hybrid checkpoints must serialize only canonical device-owned recurrent state.";
    EXPECT_EQ(compact_capture.find("constbooldevice_only_checkpoint=false;"),
              std::string::npos)
        << "A hard-coded host-mirror checkpoint policy would revive the retired GPU state seam.";
    EXPECT_NE(capture_body.find(
                  "live_prefix_checkpoint_hybrid_device_only_captures"),
              std::string::npos)
        << "Perfstats must prove that live GPU rollback checkpoints stayed device-only.";
    EXPECT_NE(compact_capture.find("recordLivePrefixCheckpointReady(&snapshot,stream,"), std::string::npos);
    expectNeedleBefore(
        compact_capture,
        "snapshot.blocks.push_back(std::move(handle));",
        "recordLivePrefixCheckpointReady(&snapshot,stream,",
        "The event must be recorded after the payload handle is owned by the snapshot.");

    EXPECT_NE(compact_prepare.find("live_prefix_checkpoint_ready_.valid"), std::string::npos);
    EXPECT_NE(compact_prepare.find("waitForPendingLivePrefixCheckpointReady("), std::string::npos);
    expectNeedleBefore(
        compact_prepare,
        "waitForPendingLivePrefixCheckpointReady(",
        "waitForPendingAcceptedSpecPublicationReady(",
        "Forward replay must wait for async checkpoint source copies before later live-state handoffs.");

    expectNeedleBefore(
        compact_restore,
        "waitForPendingLivePrefixCheckpointReady(",
        "state_.kv_cache->truncateSequence(",
        "Restore must not truncate live state while an async checkpoint export is still reading it.");
    expectNeedleBefore(
        compact_restore,
        "waitForSnapshotReady(",
        "state_.kv_cache->truncateSequence(",
        "Restore must wait for snapshot payload readiness before importing device-resident checkpoint bytes.");
    expectNeedleBefore(
        compact_truncate,
        "waitForPendingLivePrefixCheckpointReady(",
        "state_.kv_cache->truncateSequence(",
        "Truncate is a live-state mutation and must wait for pending async checkpoint exports.");
}

TEST(Test__GpuWorkspaceAllocationPolicy, LivePrefixRestoreAndTruncatePublishEventBackedMutationHandoffs)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto compact_header = removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto restore_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::restoreLivePrefixState(",
        "bool DeviceGraphOrchestrator::truncateLivePrefixState(");
    const auto truncate_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::truncateLivePrefixState(",
        "bool DeviceGraphOrchestrator::mtpSpecStatePublicationRequiresCapturedStage(");
    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(",
        "const float *DeviceGraphOrchestrator::getAllPositionLogits()");
    const auto probe_body = sliceBetween(
        source,
        "PrefixRuntimeStateSnapshot DeviceGraphOrchestrator::prefixStateProbe() const",
        "void DeviceGraphOrchestrator::disablePrefixCacheForRunner");
    const auto payload_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixState(int seq_idx) const",
        "bool DeviceGraphOrchestrator::ensureLiveHybridCheckpointStorage");
    const auto checkpoint_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(int seq_idx) const",
        "bool DeviceGraphOrchestrator::restoreLivePrefixState");
    const auto device_publication_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    const auto record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordLivePrefixMutationReady(",
        "bool DeviceGraphOrchestrator::waitForPendingLivePrefixMutationReady(");
    const auto wait_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLivePrefixMutationReady(",
        "bool DeviceGraphOrchestrator::waitForPendingLivePrefixMutationReadyForObservation(");
    const auto observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLivePrefixMutationReadyForObservation(",
        "void DeviceGraphOrchestrator::clearPendingLivePrefixMutationReady()");
    const auto restored_transaction_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordRestoredDeviceResidentMTPTransaction(",
        "bool DeviceGraphOrchestrator::waitForDeviceResidentMTPTransaction(");
    const auto transaction_record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordDeviceResidentMTPTransactionMutation(",
        "bool DeviceGraphOrchestrator::recordRestoredDeviceResidentMTPTransaction(");
    const auto retire_transaction_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::retireDeviceResidentMTPTransaction()",
        "DeviceResidentMTPTransactionLease");
    const auto clear_logical_mailbox_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearDeviceResidentLogicalSequenceStateMailbox()",
        "void DeviceGraphOrchestrator::retireDeviceResidentMTPTransaction()");
    const auto prepare_publication_metadata_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "DeviceGraphOrchestrator::forwardReplayCacheObservations() const");
    const auto clear_inference_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearInferenceState()",
        "} // namespace llaminar2");

    const auto compact_restore = removeAsciiWhitespace(stripCommentsAndStringLiterals(restore_body));
    const auto compact_restore_with_labels = removeAsciiWhitespace(restore_body);
    const auto compact_truncate = removeAsciiWhitespace(stripCommentsAndStringLiterals(truncate_body));
    const auto compact_sidecar = removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_body));
    const auto compact_prepare = removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_probe = removeAsciiWhitespace(stripCommentsAndStringLiterals(probe_body));
    const auto compact_payload = removeAsciiWhitespace(stripCommentsAndStringLiterals(payload_body));
    const auto compact_checkpoint = removeAsciiWhitespace(stripCommentsAndStringLiterals(checkpoint_body));
    const auto compact_device_publication = removeAsciiWhitespace(stripCommentsAndStringLiterals(device_publication_body));
    const auto compact_record = removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));
    const auto compact_wait = removeAsciiWhitespace(stripCommentsAndStringLiterals(wait_body));
    const auto compact_observation = removeAsciiWhitespace(stripCommentsAndStringLiterals(observation_body));
    const auto compact_restored_transaction =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(restored_transaction_body));
    const auto compact_transaction_record =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(transaction_record_body));
    const auto compact_retire_transaction =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(retire_transaction_body));
    const auto compact_clear_logical_mailbox =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(clear_logical_mailbox_body));
    const auto compact_prepare_publication_metadata =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(prepare_publication_metadata_body));
    const auto compact_clear_inference =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(clear_inference_body));

    EXPECT_NE(compact_header.find("PendingLivePrefixMutationReadyState"), std::string::npos);
    EXPECT_NE(compact_header.find("mutablePendingLivePrefixMutationReadyStatelive_prefix_mutation_ready_;"),
              std::string::npos);
    EXPECT_NE(compact_header.find("recordLivePrefixMutationReady(void*producer_stream,constchar*producer_name)"),
              std::string::npos);
    EXPECT_NE(compact_header.find("waitForPendingLivePrefixMutationReady(void*consumer_stream,constchar*consumer_name)"),
              std::string::npos);
    EXPECT_NE(compact_header.find("waitForPendingLivePrefixMutationReadyForObservation(void*consumer_stream,constchar*consumer_name)const"),
              std::string::npos);
    EXPECT_NE(compact_header.find("waitForPendingLiveGraphProducersBeforePrefixMutation(void*mutation_stream,constchar*mutation_name)"),
              std::string::npos);
    EXPECT_NE(compact_header.find(
                  "recordRestoredDeviceResidentMTPTransaction(void*producer_stream,constchar*producer_name)"),
              std::string::npos)
        << "GPU restore must advance the persistent shifted-cache transaction fence.";

    EXPECT_NE(compact_record.find("backend->createEvent("), std::string::npos);
    EXPECT_NE(compact_record.find("backend->recordEvent("), std::string::npos);
    EXPECT_NE(compact_record.find("live_prefix_mutation_ready_"), std::string::npos);
    EXPECT_NE(compact_wait.find("streamWaitEvent("), std::string::npos);
    EXPECT_NE(compact_wait.find("clearPendingLivePrefixMutationReady();"), std::string::npos);
    EXPECT_NE(compact_observation.find("streamWaitEvent("), std::string::npos);
    EXPECT_EQ(compact_observation.find("clearPendingLivePrefixMutationReady();"), std::string::npos)
        << "Diagnostic snapshot/probe waits must observe restore readiness without consuming it.";

    EXPECT_NE(compact_prepare.find("live_prefix_mutation_ready_.valid"), std::string::npos);
    EXPECT_NE(compact_prepare.find("waitForPendingLivePrefixMutationReady("), std::string::npos);
    EXPECT_NE(compact_sidecar.find("waitForPendingLivePrefixMutationReady("), std::string::npos);
    EXPECT_NE(compact_probe.find("waitForLiveInferenceStateReadyForObservation("), std::string::npos);
    EXPECT_NE(compact_payload.find("waitForLiveInferenceStateReadyForObservation("), std::string::npos);
    EXPECT_NE(compact_checkpoint.find("waitForLiveInferenceStateReadyForObservation("), std::string::npos);

    EXPECT_NE(compact_transaction_record.find(
                  "cache->deviceSequenceCachedTokenCountPtr(0)"),
              std::string::npos)
        << "The persistent transaction must point at each cache's canonical device counter.";
    EXPECT_NE(compact_restored_transaction.find(
                  "recordDeviceResidentMTPTransactionMutation("),
              std::string::npos)
        << "Restore ownership must be event-backed, not an untracked raw pointer.";
    EXPECT_EQ(compact_restored_transaction.find("get_cached_tokens"),
              std::string::npos)
        << "GPU restore must never reconstruct its shifted-cache boundary from a host mirror.";
    EXPECT_NE(compact_clear_logical_mailbox.find(
                  "device_resident_logical_sequence_state_mailbox_.clear()"),
              std::string::npos);
    EXPECT_EQ(compact_clear_logical_mailbox.find(
                  "retireDeviceResidentMTPTransaction"),
              std::string::npos)
        << "Retiring an old logical outcome must not destroy the request transaction.";
    EXPECT_NE(compact_prepare_publication_metadata.find(
                  "clearDeviceResidentLogicalSequenceStateMailbox()"),
              std::string::npos)
        << "A new compact outcome must retire the previous logical mailbox.";
    EXPECT_EQ(compact_prepare_publication_metadata.find(
                  "retireDeviceResidentMTPTransaction"),
              std::string::npos)
        << "Metadata preparation must preserve shifted-cache ownership through initial/suffix commits.";
    EXPECT_NE(compact_clear_inference.find(
                  "clearDeviceResidentLogicalSequenceStateMailbox()"),
              std::string::npos);
    EXPECT_NE(compact_clear_inference.find(
                  "retireDeviceResidentMTPTransaction()"),
              std::string::npos)
        << "Session teardown must retire the request-scoped transaction.";
    EXPECT_NE(compact_retire_transaction.find(
                  "device_resident_mtp_transaction_.reset()"),
              std::string::npos);
    EXPECT_EQ(countOccurrences(
                  compact_restore,
                  "recordRestoredDeviceResidentMTPTransaction("),
              3u)
        << "Logical, zero-token payload, and non-empty payload restores must all advance the transaction fence.";

    expectNeedleBefore(
        compact_restore,
        "waitForPendingLivePrefixMutationReady(",
        "state_.kv_cache->truncateSequence(",
        "A restore must wait for any older async prefix mutation before overwriting live state.");
    expectNeedleBefore(
        compact_restore,
        "waitForPendingLiveGraphProducersBeforePrefixMutation(",
        "state_.kv_cache->truncateSequence(",
        "A restore must order after pending forward/verifier graph producers before overwriting KV/GDN state.");
    expectNeedleBefore(
        compact_restore,
        "handleLivePrefixReplayStateAfterMutation(",
        "recordLivePrefixMutationReady(",
        "Restore readiness must be recorded after the live-state epoch changes.");
    expectNeedleBefore(
        compact_restore_with_labels,
        "recordLivePrefixMutationReady(stream,\"restore_logical_checkpoint\")",
        "recordRestoredDeviceResidentMTPTransaction(stream,\"restore_logical_checkpoint_shifted_kv\")",
        "Logical restore must advance its shifted-cache transaction after recording the restore mutation.");
    expectNeedleBefore(
        compact_restore_with_labels,
        "recordLivePrefixMutationReady(stream,\"restore_payload_checkpoint_zero\")",
        "recordRestoredDeviceResidentMTPTransaction(stream,\"restore_payload_checkpoint_zero_shifted_kv\")",
        "Zero-token restore must advance its shifted-cache transaction after recording the restore mutation.");
    expectNeedleBefore(
        compact_restore_with_labels,
        "recordLivePrefixMutationReady(stream,\"restore_payload_checkpoint\")",
        "recordRestoredDeviceResidentMTPTransaction(stream,\"restore_payload_checkpoint_shifted_kv\")",
        "Payload restore must advance its shifted-cache transaction after recording the restore mutation.");
    expectNeedleBefore(
        compact_truncate,
        "waitForPendingLivePrefixMutationReady(",
        "state_.kv_cache->truncateSequence(",
        "Truncate must wait for any older async prefix mutation before mutating live state.");
    expectNeedleBefore(
        compact_truncate,
        "waitForPendingLiveGraphProducersBeforePrefixMutation(",
        "state_.kv_cache->truncateSequence(",
        "Truncate must order after pending forward/verifier graph producers before mutating KV/GDN state.");
    expectNeedleBefore(
        compact_truncate,
        "handleLivePrefixReplayStateAfterMutation(",
        "recordLivePrefixMutationReady(",
        "Truncate readiness must be recorded after the live-state epoch changes.");
    expectNeedleBefore(
        compact_device_publication,
        "handleLivePrefixReplayStateAfterMutation(",
        "recordLivePrefixMutationReady(",
        "Device-resident accepted spec-state publication must publish the generic live mutation handoff.");
    expectNeedleBefore(
        compact_device_publication,
        "recordAcceptedSpecPublicationReady(",
        "recordLivePrefixMutationReady(",
        "Device-resident publication consumers wait accepted-state readiness before the generic live mutation handoff.");
    expectNeedleBefore(
        compact_device_publication,
        "recordDeviceResidentLogicalSequenceStateMailbox(",
        "recordLivePrefixMutationReady(",
        "Device-resident publication should record the live mutation after the mailbox event is queued.");
    expectNeedleBefore(
        compact_prepare,
        "waitForPendingAcceptedSpecPublicationReady(",
        "waitForPendingLivePrefixMutationReady(",
        "Forward consumers should order accepted publication first, then prefix restore/truncate handoffs.");
    expectNeedleBefore(
        compact_sidecar,
        "waitForPendingAcceptedSpecPublicationReady(",
        "waitForPendingLivePrefixMutationReady(",
        "Sidecar consumers should order accepted publication first, then prefix restore/truncate handoffs.");
}

TEST(Test__GpuWorkspaceAllocationPolicy, PrefixRestoreClearsDiscardedTimelineTransientHandoffs)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto reset_body = sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override");
    const auto helper_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearLivePrefixRestoreTransientHandoffs(",
        "void DeviceGraphOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled(");
    const auto mtp_transaction_helper_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearMTPVerifierTransactionStateForBoundary(",
        "void DeviceGraphOrchestrator::clearLivePrefixRestoreTransientHandoffs(");
    const auto wait_helper_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLiveGraphProducersBeforePrefixMutation(",
        "void DeviceGraphOrchestrator::clearLivePrefixRestoreTransientHandoffs(");
    const auto mutation_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::handleLivePrefixReplayStateAfterMutation(",
        "PrefixCacheFingerprintResult DeviceGraphOrchestrator::buildCurrentPrefixFingerprint(");
    const auto compact_helper =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(helper_body));
    const auto compact_mtp_transaction_helper =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mtp_transaction_helper_body));
    const auto compact_reset =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(reset_body));
    const auto compact_wait_helper =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(wait_helper_body));
    const auto compact_mutation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mutation_body));

    EXPECT_NE(compact_wait_helper.find("PendingLogitsStreamRole::MainDecode"),
              std::string::npos);
    EXPECT_NE(compact_wait_helper.find("PendingLogitsStreamRole::MTPSidecar"),
              std::string::npos);
    EXPECT_NE(compact_wait_helper.find("PendingLogitsStreamRole::AllPositionVerifier"),
              std::string::npos);
    EXPECT_NE(compact_wait_helper.find("waitForPendingShiftedMTPKVReady("),
              std::string::npos);
    EXPECT_NE(compact_wait_helper.find("waitForPendingAllPositionVerifierStateReady("),
              std::string::npos);
    EXPECT_NE(compact_header.find("clearLivePrefixRestoreTransientHandoffs(constchar*reason)"),
              std::string::npos);
    EXPECT_NE(compact_header.find("clearMTPVerifierTransactionStateForBoundary(constchar*reason)"),
              std::string::npos);
    EXPECT_NE(compact_reset.find("clearMTPVerifierTransactionStateForBoundary(reset_reason)"),
              std::string::npos)
        << "Request reset must clear the same MTP verifier transaction owner as prefix restore.";
    EXPECT_NE(compact_helper.find("clearAllPendingLogitsStreams(clear_reason)"),
              std::string::npos)
        << "A restored snapshot must not inherit logits stream ownership from the abandoned timeline.";
    EXPECT_NE(compact_helper.find("clearPendingAllPositionVerifierStateReady()"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("clearPendingAcceptedSpecPublicationReady()"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("clearPendingLivePrefixCheckpointReady()"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("shifted_mtp_kv_ready_.valid=false"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("shifted_mtp_kv_ready_.event.reset()"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("clearMTPVerifierTransactionStateForBoundary(clear_reason)"),
              std::string::npos)
        << "Prefix restore should clear MTP verifier transaction state through its first-class owner.";
    EXPECT_NE(compact_mtp_transaction_helper.find("pending_mtp_spec_verifier_input_plan_.reset()"),
              std::string::npos)
        << "Prefix restore must drop abandoned spec verifier input plans.";
    EXPECT_NE(compact_mtp_transaction_helper.find("pending_mtp_verifier_device_token_plan_.reset()"),
              std::string::npos);
    EXPECT_NE(compact_mtp_transaction_helper.find("materialized_mtp_verifier_device_token_row_={}"),
              std::string::npos);
    EXPECT_NE(compact_mtp_transaction_helper.find("mtp_publication_base_cache_snapshot_ready_=false"),
              std::string::npos)
        << "Prefix restore must not retain verifier publication base snapshots from the abandoned timeline.";
    EXPECT_NE(compact_mtp_transaction_helper.find("mtp_publication_base_cache_snapshot_request_count_=0"),
              std::string::npos);
    EXPECT_NE(compact_mtp_transaction_helper.find("request_batched_prefill_logits_row_count_=0"),
              std::string::npos);
    EXPECT_NE(compact_mtp_transaction_helper.find("setRowIndexedAllPositionLogitRows({})"),
              std::string::npos)
        << "Prefix restore must drop compact verifier row ownership from the abandoned timeline.";
    EXPECT_NE(compact_mtp_transaction_helper.find("setComputeRowIndexedAllPositionLogits(false,0)"),
              std::string::npos)
        << "A restored prefix checkpoint must not inherit row-indexed verifier logits mode.";
    EXPECT_NE(compact_mtp_transaction_helper.find("setComputeAllPositionLogits(false)"),
              std::string::npos)
        << "A restored prefix checkpoint resumes as ordinary decode, not an all-position verifier.";
    EXPECT_EQ(compact_helper.find("&& !compute_all_position_logits_"), std::string::npos)
        << "Restore cleanup must clear verifier mode even when restore happens while all-position mode is active.";
    EXPECT_NE(compact_helper.find(
                  "clearStochasticTargetSampleReadySlots(StochasticSampleReadyClearMode::PreserveVerifierConsumer)"),
              std::string::npos);
    EXPECT_NE(compact_helper.find(
                  "clearStochasticDraftSampleReadySlots(StochasticSampleReadyClearMode::PreserveVerifierConsumer)"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("clearDeviceResidentLogicalSequenceStateMailbox()"),
              std::string::npos);

    expectNeedleBefore(
        compact_mutation,
        "clearLivePrefixRestoreTransientHandoffs(operation)",
        "recordLivePrefixMutation(reason,operation)",
        "Prefix restore must drop stale timeline handoffs before publishing the new live-state epoch.");
    EXPECT_NE(compact_mutation.find("reason==LivePrefixMutationReason::PrefixRestore"),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, ROCmGDNVerifierRowsUseSerialDecodeEquivalentSnapshots)
{
    const auto source =
        readFile(repoRoot() / "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip");
    const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(source));
    const auto grouped_helper = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool rocmGDN_chunk_forward_kernel_route(",
        "bool rocmGDN_chunk_forward(")));
    const auto grouped_batched_helper = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool rocmGDN_chunk_forward_batched_kernel_route(",
        "bool rocmGDN_chunk_forward_kernel_route(")));
    const auto public_route = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool rocmGDN_chunk_forward(",
        "bool rocmGDN_chunk_forward_effective(")));
    const auto effective_route = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool rocmGDN_chunk_forward_effective(",
        "bool rocmGDN_short_conv1d(")));

    EXPECT_NE(compact.find("boolrocmGDN_chunk_forward_kernel_route("), std::string::npos)
        << "ROCm GDN should share one launch helper between normal and verifier-specific routes.";
    EXPECT_NE(public_route.find("rocmGDN_chunk_forward_kernel_route("), std::string::npos)
        << "MTP verifier chunks must use the grouped route instead of a hidden "
           "loop over single-row launches.";
    EXPECT_NE(effective_route.find("rocmGDN_chunk_forward_kernel_route("), std::string::npos)
        << "The effective-length route used by graph replay must share the grouped verifier path.";
    EXPECT_NE(grouped_helper.find("state_snapshots"), std::string::npos)
        << "Grouped verifier state snapshots must be produced inside the GPU route.";
    EXPECT_NE(grouped_helper.find("snapshot_stride_floats"), std::string::npos);
    EXPECT_NE(grouped_helper.find("device_effective_seq_len"), std::string::npos)
        << "Snapshot publication must be guarded by device-resident row metadata.";
    EXPECT_NE(grouped_batched_helper.find("!stream"), std::string::npos)
        << "Grouped verifier GDN must reject the device-default stream.";
    EXPECT_TRUE(
        grouped_batched_helper.find("static_cast<hipStream_t>(stream)") != std::string::npos ||
        grouped_batched_helper.find("(hipStream_t)stream") != std::string::npos)
        << "Grouped verifier GDN must run on the caller's explicit capture stream.";
    EXPECT_EQ(grouped_batched_helper.find("hipMemcpyAsync("), std::string::npos)
        << "Grouped verifier GDN snapshots must be written by kernels, not by "
           "ad hoc copy calls in the hot path.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GDNVerifierCaptureWorkspacesAreGraphRoleScoped)
{
    const auto gdn_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h");
    const auto gdn_stage =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp");
    const auto conv_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ShortConv1dStage.h");
    const auto conv_stage =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp");
    const auto qwen35 =
        readFile(repoRoot() / "src/v2/models/qwen35/Qwen35Graph.cpp");

    /*
     * Verifier-row snapshots are short-lived graph outputs.  Main all-position
     * verifier graphs and MTP sidecar graphs can have the same logical layer id,
     * so layer-only workspace keys let one graph overwrite another graph's
     * captured recurrent/conv state before publication consumes it.
     */
    for (const auto &header : {gdn_header, conv_header})
    {
        EXPECT_NE(header.find("std::string workspace_namespace"),
                  std::string::npos);
    }
    for (const auto &stage : {gdn_stage, conv_stage})
    {
        const auto stable_id = removeAsciiWhitespace(stripCommentsAndStringLiterals(
            sliceBetween(stage,
                         "std::string " +
                             std::string(stage.find("GDNRecurrenceStage") != std::string::npos
                                             ? "GDNRecurrenceStage"
                                             : "ShortConv1dStage") +
                             "::workspaceStableId() const",
                         "std::string " +
                             std::string(stage.find("GDNRecurrenceStage") != std::string::npos
                                             ? "GDNRecurrenceStage"
                                             : "ShortConv1dStage") +
                             "::effectiveSeqLenScalarBufferName() const")));
        EXPECT_NE(stable_id.find("params_.workspace_namespace"), std::string::npos);
        EXPECT_NE(stable_id.find("role_prefix+"), std::string::npos)
            << "Capture workspace ids must include the graph-role namespace.";
    }

    EXPECT_NE(qwen35.find("conv_params.workspace_namespace = workspace_namespace"),
              std::string::npos);
    EXPECT_NE(qwen35.find("rec_params.workspace_namespace = workspace_namespace"),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPStochasticAllPositionPathKeepsResidentOutcomeHandleVisible)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto body = sliceBetween(
        source,
        "if (batched_device_rejection)",
        "if (!used_device_batch_outcome &&");
    const auto publication_body = sliceBetween(
        source,
        "if (!state_published_from_device_outcome)\n"
        "            {",
        "int correction_forward_count = 0;");
    const auto correction_body = sliceBetween(
        source,
        "int correction_forward_count = 0;",
        "std::vector<int32_t> accepted_tokens =");
    const auto initial_shifted_body = sliceBetween(
        source,
        "const bool first_shifted_row_available_from_sidecar",
        "const MTPSpecDecodeVerifierInputPlan verifier_input_plan");
    const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(body));
    const auto compact_publication =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publication_body));
    const auto compact_correction =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(correction_body));
    const auto compact_initial_shifted =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(initial_shifted_body));

    /*
     * Phase 10's direct-publication path needs the compact device verifier
     * handle to remain visible in the runner.  Calling the legacy
     * host-returning verifier here would hide the ownership boundary inside the
     * runner implementation and make it much harder to move accepted-state
     * publication onto device later.
     */
    EXPECT_NE(compact.find("DeviceSpeculativeOutcomeHandledevice_outcome_handle;"),
              std::string::npos);
    EXPECT_NE(compact.find(
                  "verifyStochasticDistributionsBatchOutcomeOnDeviceResident("),
              std::string::npos);
    EXPECT_NE(compact.find(
                  "verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident("),
              std::string::npos);
    EXPECT_NE(compact.find("supportsDeviceResidentMTPSpecStatePublication()"),
              std::string::npos);
    EXPECT_NE(compact.find("publishAcceptedMTPSpecStateBatchFromDeviceOutcome("),
              std::string::npos);
    EXPECT_NE(compact.find("materializeDeviceSpeculativeOutcomesForHostResponse("),
              std::string::npos);
    const size_t direct_publish =
        compact.find("publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const size_t host_bridge = compact.find("copyDeviceSpeculativeOutcomesToHost(");
    const size_t host_response_materialize =
        compact.find("materializeDeviceSpeculativeOutcomesForHostResponse(");
    ASSERT_NE(direct_publish, std::string::npos);
    ASSERT_NE(host_response_materialize, std::string::npos);
    EXPECT_EQ(host_bridge, std::string::npos)
        << "The all-position path should call the named host-response materializer, "
           "not the low-level D2H copy hook directly.";
    EXPECT_LT(direct_publish, host_response_materialize)
        << "Device-resident state publication must run before the "
           "compatibility host-response materialization.";
    EXPECT_EQ(compact.find(
                  "verifyStochasticDistributionsBatchOutcomeOnDevice("),
              std::string::npos)
        << "The all-position stochastic path must enqueue a resident outcome "
           "first; the host-returning verifier API is compatibility-only.";
    EXPECT_EQ(compact.find(
                  "verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken("),
              std::string::npos)
        << "Device-first stochastic outcome reduction must keep the resident "
           "handle visible before bridging to host.";

    EXPECT_NE(compact_publication.find("publishAcceptedMTPSpecStateBatch("),
              std::string::npos)
        << "The non-resident branch still owns host-plan publication.";
    EXPECT_NE(compact_publication.find("deviceResidentLogicalSequenceState()"),
              std::string::npos)
        << "The resident branch must validate the typed resident mailbox instead "
           "of adopting backend host mirrors.";
    EXPECT_NE(compact_publication.find("!resident_state.valid()"),
              std::string::npos)
        << "Resident publication must fail closed if publication produced no "
           "typed logical-state mailbox.";
    EXPECT_EQ(compact_publication.find("adoptDeviceResidentMTPSpecPublishedHostState("),
              std::string::npos)
        << "Scalar resident publication must not require backend host-state adoption.";

    EXPECT_EQ(compact_correction.find(
                  "commitMTPShiftedRowFromDeviceResidentLogicalState("),
              std::string::npos)
        << "A rejected correction token is only a pending verifier condition. "
           "It must not append shifted MTP KV until the next step consumes it.";
    EXPECT_EQ(compact_correction.find(
                  "commitMTPShiftedRowFromCurrentTerminalHidden("),
              std::string::npos)
        << "The legacy host-token helper must not append shifted KV for a "
           "deferred correction row either.";
    EXPECT_NE(correction_body.find(
                  "\"all_position_deferred_correction_condition_tokens\""),
              std::string::npos)
        << "The correction branch should remain visible as pending-condition "
           "accounting, not as a shifted-cache mutation.";
    EXPECT_EQ(compact_initial_shifted.find(
                  "commitMTPShiftedRowFromCurrentTerminalHidden("),
              std::string::npos)
        << "All-position state publication must not synthesize the initial "
           "shifted-MTP row from a host-visible token; only explicit sidecar "
           "reuse or accepted verifier-row publication may own that state.";
    EXPECT_EQ(compact_initial_shifted.find(
                  "commitMTPShiftedRowFromDeviceTargetSample("),
              std::string::npos)
        << "The device-sampled first token is also only an output until a "
           "published verifier row proves the matching state boundary.";
    EXPECT_NE(initial_shifted_body.find(
                  "\"all_position_initial_shifted_deferred_to_verifier_rows\""),
              std::string::npos)
        << "The non-reuse branch should be explicit in perf stats so future "
           "profiling can distinguish sidecar reuse from verifier-row publication.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPStochasticAllPositionPathKeepsCompactBonusUntilProcessedBonusGate)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto body = sliceBetween(
        source,
        "if (batched_device_rejection)",
        "if (!used_device_batch_outcome &&");
    const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(body));

    /*
     * A Phase 10 experiment tried to replace the compact bonus distribution
     * with a lazily-sampled processed-logit bonus row.  The primitive is still
     * useful for isolated equivalence work, but the served stochastic stream
     * regressed badly because sampler trajectory and bonus-token consumption
     * were not equivalent end to end.  Until that equivalence is proven, the
     * production all-position path must build target rows plus the bonus row in
     * the compact distribution buffer and pass that bonus slot to the verifier.
     */
    EXPECT_NE(compact.find("constintbonus_row=compare_rows;"),
              std::string::npos);
    EXPECT_NE(compact.find(
                  "buildStochasticDistributionsOnDevice(DeviceLogitsSource::AllPosition,0,DeviceDistributionBuffer::Target,0,compare_rows+1,"),
              std::string::npos)
        << "Production stochastic MTP must keep the compact target+bonus row "
           "path until processed bonus equivalence has a dedicated gate.";
    EXPECT_NE(compact.find("bonus_row,bonus_threshold,"),
              std::string::npos)
        << "The compact bonus slot must remain part of the device verifier request.";
    EXPECT_EQ(compact.find("buildStochasticProcessedLogitRowsOnDevice("),
              std::string::npos)
        << "Processed-logit bonus rows are not a production replacement for "
           "compact target+bonus distributions yet.";
    EXPECT_EQ(compact.find("SampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus"),
              std::string::npos)
        << "Lazy processed bonus sampling must stay out of the served path "
           "until sampler-trajectory parity proves it is equivalent.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPResidentPublicationPrelaunchesBeforeHostBridge)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto body = sliceBetween(
        source,
        "if (batched_device_rejection)",
        "if (!used_device_batch_outcome &&");
    const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(body));
    const auto full_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(source));

    /*
     * vLLM overlaps host response materialization with already-resident GPU
     * state work.  Llaminar's Phase 10 bridge follows the same shape for
     * stochastic lanes: publish accepted state on device, enqueue the next
     * first sidecar from the resident mailbox, then run the compatibility
     * host-response materializer.  Stop handling is still host-visible at the
     * response boundary, so completed requests must discard prelaunch work.
     */
    const size_t direct_publish =
        compact.find("publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const size_t prelaunch_gate =
        compact.find("constboolcan_prelaunch_next_first_sidecar=");
    const size_t prelaunch_enqueue =
        compact.find("forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(",
                     prelaunch_gate);
    const size_t host_response_materialize =
        compact.find("materializeDeviceSpeculativeOutcomesForHostResponse(");

    ASSERT_NE(direct_publish, std::string::npos);
    ASSERT_NE(prelaunch_gate, std::string::npos);
    ASSERT_NE(prelaunch_enqueue, std::string::npos);
    ASSERT_NE(host_response_materialize, std::string::npos);
    EXPECT_LT(direct_publish, prelaunch_gate);
    EXPECT_LT(prelaunch_gate, prelaunch_enqueue);
    EXPECT_LT(prelaunch_enqueue, host_response_materialize)
        << "The first sidecar for the next step must be queued before the "
           "host outcome bridge can synchronize for served tokens.";
    EXPECT_NE(source.find(
                  "\"stochastic_first_sidecar_prelaunch_reuses\""),
              std::string::npos)
        << "The following decode step must have an explicit reuse path for "
           "the sidecar queued before host materialization.";
    EXPECT_NE(source.find(
                  "\"stochastic_first_sidecar_prelaunch_discarded_complete\""),
              std::string::npos)
        << "A stop-token completion must discard any sidecar prelaunched "
           "before host-visible response materialization.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GroupedGreedyResidentPublicationPrelaunchesBeforeHostBridge)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto body = sliceBetween(
        source,
        "if (!stochastic_verify && grouped_outcome_device_resident_publication)",
        "if (!catchup.ok)\n                {");
    const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(body));
    const auto compact_raw = removeAsciiWhitespace(body);

    const size_t direct_publish =
        compact.find("publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const size_t prelaunch_gate =
        compact.find("constboolcan_prelaunch_next_first_sidecar=");
    const size_t mailbox =
        compact.find("runner_->deviceResidentLogicalSequenceState()", prelaunch_gate);
    const size_t prelaunch_enqueue =
        compact.find("forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(",
                     prelaunch_gate);
    const size_t host_response_materialize =
        compact.find("materializeDeviceSpeculativeOutcomesForHostResponse(");

    ASSERT_NE(direct_publish, std::string::npos);
    ASSERT_NE(prelaunch_gate, std::string::npos);
    ASSERT_NE(mailbox, std::string::npos);
    ASSERT_NE(prelaunch_enqueue, std::string::npos);
    ASSERT_NE(host_response_materialize, std::string::npos);
    EXPECT_LT(direct_publish, prelaunch_gate)
        << "Grouped greedy prelaunch must wait until accepted state has been "
           "published from compact device metadata.";
    EXPECT_LT(prelaunch_gate, mailbox);
    EXPECT_LT(mailbox, prelaunch_enqueue);
    EXPECT_LT(prelaunch_enqueue, host_response_materialize)
        << "Grouped greedy should queue the resident continuation before the "
           "response-only D2H bridge.";
    EXPECT_NE(compact_raw.find("\"prelaunch_timing\",\"pre_bridge\""),
              std::string::npos);
    EXPECT_EQ(compact_raw.find("\"prelaunch_timing\",\"post_bridge\""),
              std::string::npos)
        << "The grouped greedy resident continuation should not wait for "
           "host-response materialization.";
    EXPECT_NE(compact_raw.find("\"timing\",\"post_publication_response_bridge\""),
              std::string::npos)
        << "The bridge must remain explicitly response-only after publication.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, StochasticOutcomeHostBridgeWaitsOnResponseReadyEvent)
{
    const auto interface_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto dgo_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto handle_body = sliceBetween(
        interface_source,
        "struct DeviceSpeculativeOutcomeHandle",
        "struct DeviceSpeculativePublicationRequest");
    const auto resident_verify_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(",
        "bool DeviceGraphOrchestrator::copyDeviceSpeculativeOutcomesToHost(");
    const auto host_bridge_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::copyDeviceSpeculativeOutcomesToHost(",
        "bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(");
    const auto compact_handle =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(handle_body));
    const auto compact_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(interface_source));
    const auto compact_verify =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(resident_verify_body));
    const auto compact_bridge =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(host_bridge_body));

    EXPECT_NE(compact_handle.find("std::shared_ptr<void>response_ready_event"),
              std::string::npos)
        << "Resident stochastic outcome handles must carry a response-ready event.";
    EXPECT_NE(compact_handle.find("response_ready_event!=nullptr"),
              std::string::npos)
        << "A handle without a response-ready event must not be considered valid.";
    EXPECT_EQ(compact_interface.find("materializeDeviceSpeculativeOutcomesForHostPlan("),
              std::string::npos)
        << "Resident MTP outcomes should have one host materialization surface: "
           "the response-only bridge. A host-plan bridge invites state "
           "publication to depend on D2H metadata again.";
    EXPECT_NE(compact_interface.find(
                  "returncopyDeviceSpeculativeOutcomesToHost(handle,outcomes);"),
              std::string::npos)
        << "The response-only materializer should delegate directly to the "
           "low-level compact D2H hook, not through a host-plan adapter.";
    EXPECT_NE(compact_verify.find("backend->createEvent("),
              std::string::npos);
    EXPECT_NE(compact_verify.find(
                  "backend->recordEvent(response_ready_event.get(),state_.device_id.gpu_ordinal(),stream)"),
              std::string::npos)
        << "The verifier must record response readiness on the producer stream "
           "before later live-state publication can enqueue behind it.";
    EXPECT_NE(compact_verify.find(
                  "out_handle->response_ready_event=std::move(response_ready_event)"),
              std::string::npos);

    EXPECT_NE(compact_bridge.find("backend->createStream("),
              std::string::npos)
        << "The compact D2H bridge should use an explicit bridge stream.";
    EXPECT_NE(compact_bridge.find("stochastic_outcome_response_bridge_stream_"),
              std::string::npos)
        << "The compact D2H bridge must reuse a persistent explicit stream; "
           "per-step HIP stream create/destroy is visible in fixed-depth MTP.";
    EXPECT_EQ(compact_bridge.find("std::shared_ptr<void>owned_copy_stream"),
              std::string::npos)
        << "The compact D2H bridge must not allocate a throwaway stream owner "
           "inside the decode hot path.";
    EXPECT_NE(compact_bridge.find(
                  "backend->waitForEvent(handle.response_ready_event.get(),state_.device_id.gpu_ordinal())"),
              std::string::npos)
        << "The compatibility bridge must wait only for the compact response "
           "event, not the full publication stream.";
    EXPECT_NE(dgo_source.find(
                  "\"stochastic_request_batch_summary_response_ready_wait\""),
              std::string::npos)
        << "Bridge accounting must split verifier dependency wait from the "
           "actual compact D2H copy wait.";
    EXPECT_EQ(compact_bridge.find(
                  "backend->streamWaitEvent(copy_stream,handle.response_ready_event.get(),state_.device_id.gpu_ordinal())"),
              std::string::npos)
        << "Do not hide verifier dependency time inside the D2H stream wait; "
           "wait for response readiness explicitly so perfstats stay honest.";
    EXPECT_NE(compact_bridge.find(
                  "backend->synchronizeStream(copy_stream,state_.device_id.gpu_ordinal())"),
              std::string::npos);
    EXPECT_EQ(compact_bridge.find("backend->synchronizeStream(handle.stream"),
              std::string::npos)
        << "Synchronizing the producer stream reintroduces a publication D2H barrier.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GreedyMTPDeviceDraftSlotPathDoesNotQuietlyFallback)
{
    const auto runner_source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto dgo_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto cuda_sampling =
        readFile(repoRoot() / "src/v2/kernels/cuda/ops/CUDASamplingKernels.cu");
    const auto rocm_sampling =
        readFile(repoRoot() / "src/v2/kernels/rocm/ops/ROCmSamplingKernels.hip");
    const auto first_token_body = sliceBetween(
        runner_source,
        "if (can_defer_greedy_first_host_read)",
        "if (first_token < 0 &&");
    const auto sample_body = sliceBetween(
        runner_source,
        "auto sample_mtp_token = [&](int draft_idx, bool defer_host_read) -> int32_t",
        "const bool use_sidecar_sample_fusion");
    const auto sidecar_body = sliceBetween(
        runner_source,
        "if (draft_idx == 0)",
        "else\n                {");
    const auto greedy_summary_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(",
        "const auto *base = static_cast<const float *>(gpu_ptr);");
    const auto greedy_runner_body = sliceBetween(
        runner_source,
        "const bool use_greedy_device_batch_outcome =",
        "else\n                {");
    const auto cuda_greedy_summary_kernel = sliceBetween(
        cuda_sampling,
        "__global__ void cuda_summarize_greedy_speculative_verify_batch_kernel(",
        "__global__ void cuda_derive_speculative_publication_metadata_kernel(");
    const auto rocm_greedy_summary_kernel = sliceBetween(
        rocm_sampling,
        "__global__ void rocm_summarize_greedy_speculative_verify_batch_kernel(",
        "__global__ void rocm_derive_speculative_publication_metadata_kernel(");

    const size_t first_token_device_sample =
        first_token_body.find("sampleGreedyFromMainLogitsToDeviceTargetSlot(");
    const size_t first_token_device_failure =
        first_token_body.find("device target-slot deferred sampling failed");
    const size_t first_token_legacy_sample =
        first_token_body.find("sampleGreedyOnDevice()");
    ASSERT_NE(first_token_device_sample, std::string::npos);
    ASSERT_NE(first_token_device_failure, std::string::npos);
    ASSERT_NE(first_token_legacy_sample, std::string::npos);
    EXPECT_LT(first_token_device_sample, first_token_device_failure);
    EXPECT_LT(first_token_device_failure, first_token_legacy_sample)
        << "Once the greedy first-token device-slot lane is selected, failure "
           "must abort instead of quietly using the old synchronized sampler.";

    const size_t slot_sample =
        sample_body.find("sampleGreedyFromMTPLogitsToDeviceDraftSlot(");
    const size_t slot_failure =
        sample_body.find("\"mtp_token_greedy_device_slot_failures\"");
    const size_t legacy_sample =
        sample_body.find("sampleGreedyFromMTPLogitsOnDevice()");
    ASSERT_NE(slot_sample, std::string::npos);
    ASSERT_NE(slot_failure, std::string::npos);
    ASSERT_NE(legacy_sample, std::string::npos);
    EXPECT_LT(slot_sample, slot_failure);
    EXPECT_LT(slot_failure, legacy_sample)
        << "If a backend opted into greedy device draft slots and sampling "
           "fails, the decode step must abort before the legacy sampler can "
           "hide the coherence bug.";

    const size_t expected_gate =
        greedy_summary_body.find("prepared_device_tokens_expected");
    const size_t missing_row_counter =
        greedy_summary_body.find(
            "\"greedy_verifier_missing_device_token_rows\"");
    const size_t missing_row_return =
        greedy_summary_body.find("return false;", missing_row_counter);
    const size_t legacy_upload_counter =
        greedy_summary_body.find("\"greedy_verifier_host_token_row_uploads\"");
    const size_t host_upload =
        greedy_summary_body.find("hostToDeviceOnStream(");
    ASSERT_NE(expected_gate, std::string::npos);
    ASSERT_NE(missing_row_counter, std::string::npos);
    ASSERT_NE(missing_row_return, std::string::npos);
    EXPECT_EQ(legacy_upload_counter, std::string::npos)
        << "Resident greedy verifier outcome must fail without prepared device "
           "tokens instead of staging a host verifier row.";
    EXPECT_EQ(host_upload, std::string::npos)
        << "Resident greedy verifier outcome must not have a hot-path H2D "
           "token-row upload.";
    EXPECT_LT(expected_gate, missing_row_counter);
    EXPECT_LT(missing_row_counter, missing_row_return);

    const auto compact_greedy_runner =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(greedy_runner_body));
    const size_t resident_verify =
        compact_greedy_runner.find(
            "verifyGreedyAllPositionBatchOutcomeOnDeviceResident(");
    const size_t legacy_verify =
        compact_greedy_runner.find(
            "verifyGreedyAllPositionBatchOutcomeOnDevice(");
    const size_t direct_publish =
        compact_greedy_runner.find(
            "publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
            resident_verify);
    const size_t host_materialize =
        compact_greedy_runner.find(
            "materializeDeviceSpeculativeOutcomesForHostResponse(",
            direct_publish);
    ASSERT_NE(resident_verify, std::string::npos);
    ASSERT_NE(direct_publish, std::string::npos);
    ASSERT_NE(host_materialize, std::string::npos);
    EXPECT_EQ(legacy_verify, std::string::npos)
        << "The greedy GPU all-position runner path must consume the resident "
           "compact outcome handle directly, not hide D2H inside the legacy "
           "host-returning verifier.";
    EXPECT_LT(resident_verify, direct_publish);
    EXPECT_LT(direct_publish, host_materialize)
        << "Device-resident greedy publication must happen before the "
           "compatibility host response bridge.";

    ASSERT_NE(sidecar_body.find(
                  "forwardMTPFromDeviceTargetAndSampleGreedyToDeviceDraftSlot("),
              std::string::npos)
        << "A deferred first token must feed the first greedy sidecar from "
           "the target device slot while preserving sidecar/sample fusion.";
    ASSERT_NE(cuda_greedy_summary_kernel.find("draft_tokens[0]"),
              std::string::npos);
    ASSERT_NE(rocm_greedy_summary_kernel.find("draft_tokens[0]"),
              std::string::npos);
    ASSERT_NE(cuda_greedy_summary_kernel.find("sampled_first_token"),
              std::string::npos);
    ASSERT_NE(rocm_greedy_summary_kernel.find("sampled_first_token"),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, RequestBatchResidentOutcomePublishesBeforeHostBridge)
{
    const auto runner_source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto producer_body = sliceBetween(
        runner_source,
        "auto produce_stochastic_outcomes =",
        "MTPOwnedDeviceOutcomeBatchTransactionResult tx;");
    const auto resident_branch = sliceBetween(
        runner_source,
        "MTPOwnedDeviceOutcomeBatchTransactionResult tx;\n            if (use_device_resident_request_batch_publication)",
        "else\n            {\n                tx =");
    const auto compact_producer =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(producer_body));
    const auto compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(resident_branch));

    EXPECT_NE(compact_producer.find("verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident("),
              std::string::npos)
        << "Request-batched GPU stochastic verification must keep the compact "
           "outcome in a typed device-resident handle.";
    EXPECT_NE(compact_producer.find("outcomes->clear()"),
              std::string::npos)
        << "The resident producer must not fabricate a host plan before "
           "device-state publication consumes the handle.";
    EXPECT_NE(compact.find("produce_stochastic_outcomes("),
              std::string::npos)
        << "The resident branch should enter through the shared producer so "
           "request admission and resident verification stay coupled.";
    EXPECT_NE(compact.find("publishAcceptedMTPSpecStateBatchFromDeviceOutcome("),
              std::string::npos)
        << "Resident request batches must publish live state from the device "
           "outcome before host response bookkeeping.";
    EXPECT_NE(compact.find("logical_state=runner_->deviceResidentLogicalSequenceState()"),
              std::string::npos)
        << "After resident publication, request batches must validate the "
           "resident logical-state mailbox without adopting backend host mirrors.";
    EXPECT_NE(compact.find("!logical_state.valid()"),
              std::string::npos)
        << "Request-batched resident publication must fail closed if the "
           "logical-state mailbox is missing.";
    EXPECT_EQ(compact.find("adoptDeviceResidentMTPSpecPublishedHostStateFromDeviceMetadata("),
              std::string::npos)
        << "Request-batched resident publication must not require backend "
           "host-state metadata adoption.";
    EXPECT_NE(compact.find("materializeDeviceSpeculativeOutcomesForHostResponse("),
              std::string::npos)
        << "The compact D2H bridge is allowed only for response tokens and "
           "sampler bookkeeping after publication.";
    EXPECT_EQ(compact.find("materializeDeviceSpeculativeOutcomesForHostPlan("),
              std::string::npos)
        << "Request-batched resident publication must not use the legacy "
           "host-plan bridge.";
    EXPECT_EQ(compact.find("copyDeviceSpeculativeOutcomesToHost("),
              std::string::npos)
        << "OrchestrationRunner should call the named response-only bridge, "
           "not the low-level D2H hook directly.";

    const size_t produce =
        compact.find("produce_stochastic_outcomes(");
    const size_t publish =
        compact.find("publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const size_t resident_plan =
        compact.find("logical_state=runner_->deviceResidentLogicalSequenceState()");
    const size_t response_bridge =
        compact.find("materializeDeviceSpeculativeOutcomesForHostResponse(");
    ASSERT_NE(produce, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    ASSERT_NE(resident_plan, std::string::npos);
    ASSERT_NE(response_bridge, std::string::npos);
    EXPECT_LT(produce, publish)
        << "The producer handle must exist before resident state publication.";
    EXPECT_LT(publish, resident_plan)
        << "Resident mailbox validation must run only after resident state publication.";
    EXPECT_LT(resident_plan, response_bridge)
        << "Response materialization is a post-publication bridge, not part "
           "of live-state mutation.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPSpecDeviceIndexedPublicationNeverFallsBackToHostRow)
{
    const auto publisher =
        readFile(repoRoot() / "src/v2/execution/mtp/MTPSpecStatePublisher.cpp");
    const auto device_publish_body = sliceBetween(
        publisher,
        "MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(\n"
        "        const MTPSpecStepPlan &plan,\n"
        "        const int *device_verifier_restore_row,\n"
        "        const std::vector<IComputeStage *> &state_stages,",
        "MTPSpecStatePublicationResult publishAcceptedMTPSpecState(");
    const auto compact_device_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(device_publish_body));

    /*
     * Phase 10's direct-publication path receives the accepted verifier row as
     * compact GPU metadata.  The publisher must forward that device pointer to
     * graph stages on an explicit stream; if it ever restores by host integer
     * row first, stochastic MTP pays the D2H sync that this phase is removing.
     */
    EXPECT_NE(compact_device_publish.find("!device.is_gpu()"), std::string::npos);
    EXPECT_NE(compact_device_publish.find("stream==nullptr"), std::string::npos);
    EXPECT_NE(compact_device_publish.find("!device_verifier_restore_row"), std::string::npos);
    EXPECT_NE(compact_device_publish.find(
                  "stage->restoreVerifierStateCaptureRowFromDeviceIndex("),
              std::string::npos);
    EXPECT_EQ(compact_device_publish.find("stage->restoreVerifierStateCaptureRow("),
              std::string::npos)
        << "Device-indexed publication must not force a host-visible row index.";

    const auto gdn_stage =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp");
    const auto conv_stage =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp");
    const auto gdn_restore = sliceBetween(
        gdn_stage,
        "bool GDNRecurrenceStage::restoreVerifierStateCaptureRowFromDeviceIndex(",
        "void GDNRecurrenceStage::onGraphReplayed()");
    const auto conv_restore = sliceBetween(
        conv_stage,
        "bool ShortConv1dStage::restoreVerifierStateCaptureRowFromDeviceIndex(",
        "void ShortConv1dStage::onGraphReplayed()");

    for (const auto &body : {gdn_restore, conv_restore})
    {
        const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(body));
        EXPECT_NE(compact.find("!device_row_index||!stream"), std::string::npos)
            << "Device-indexed stage restore must reject null row pointers and default streams.";
        EXPECT_NE(compact.find("restoreVerifierStateCaptureRowFromDeviceIndex("),
                  std::string::npos);
        EXPECT_NE(compact.find("nullptr,device_row_index,stream"),
                  std::string::npos)
            << "Device-indexed restore must not pass a host state mirror into the backend.";
        EXPECT_EQ(compact.find("stream?stream:gpuStream()"), std::string::npos)
            << "The device-indexed path must never silently fall back to a cached stream.";
    }

    const std::vector<std::filesystem::path> gpu_state_files = {
        "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNet.h",
        "src/v2/kernels/cuda/gdn/CUDAShortConvolution.h",
        "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h",
        "src/v2/kernels/rocm/gdn/ROCmShortConvolution.h",
    };
    for (const auto &relative : gpu_state_files)
    {
        const auto source = readFile(repoRoot() / relative);
        const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(source));
        EXPECT_NE(compact.find("restoreVerifierStateCaptureRowFromDeviceIndex("),
                  std::string::npos)
            << relative;
        EXPECT_NE(compact.find("!stream||"), std::string::npos)
            << relative << " must reject default/null stream publication.";
        EXPECT_TRUE(compact.find("cudaGDN_gpu_copy_capture_row_from_device_index(") != std::string::npos ||
                    compact.find("rocmGDN_gpu_copy_capture_row_from_device_index(") != std::string::npos)
            << relative << " must use the graph-capturable row-index copy kernel.";

        const auto device_restore = sliceBetween(
            source,
            "bool restoreVerifierStateCaptureRowFromDeviceIndex(",
            "bool supportsPaddedPrefillRealLength() const");
        const auto compact_device_restore =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(device_restore));
        EXPECT_EQ(compact_device_restore.find("gpu_memcpy_d2h"),
                  std::string::npos)
            << relative << " device-indexed restore must not refresh host state.";
        EXPECT_EQ(compact_device_restore.find("stream_synchronize"),
                  std::string::npos)
            << relative << " device-indexed restore must not synchronize for host visibility.";
    }

    const auto cuda_kernel =
        readFile(repoRoot() / "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNetKernels.cu");
    const auto rocm_kernel =
        readFile(repoRoot() / "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip");
    const auto cuda_helper = sliceBetween(
        cuda_kernel,
        "bool cudaGDN_gpu_copy_capture_row_from_device_index(",
        "} // extern \"C\"");
    const auto rocm_helper = sliceBetween(
        rocm_kernel,
        "bool rocmGDN_gpu_copy_capture_row_from_device_index(",
        "} // extern \"C\"");
    for (const auto &helper : {cuda_helper, rocm_helper})
    {
        const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(helper));
        EXPECT_NE(compact.find("!stream"), std::string::npos)
            << "GPU row-index copy helpers must require an explicit stream.";
        EXPECT_EQ(compact.find("deviceToHost"), std::string::npos);
        EXPECT_EQ(compact.find("DtoH"), std::string::npos);
        EXPECT_TRUE(compact.find("cudaStream_t") != std::string::npos ||
                    compact.find("hipStream_t") != std::string::npos)
            << "Row-index copy helpers must launch on the caller's stream.";
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPTerminalHiddenDeviceAcceptedRowsUseExternalMetadataWorkspace)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto external_rows_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelectFromDeviceMetadata(",
        "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRows(");
    const auto compact_external_rows =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(external_rows_body));

    EXPECT_NE(
        compact_external_rows.find(
            "params.device_row_index_source=HiddenStateRowsSelectStage::DeviceRowIndexSource::ExternalDeviceIndices"),
        std::string::npos)
        << "Device-produced row metadata must be an explicit external-device row source, not stage-owned host state.";
    EXPECT_NE(compact_external_rows.find("cache.stage->setGPUStream(rows_select_stream)"),
              std::string::npos)
        << "External row metadata selection must run on an explicit publication/verifier stream.";
    EXPECT_EQ(compact_external_rows.find("setSelectedRowsForReplay("),
              std::string::npos)
        << "The device-metadata helper must not mutate row indices through host replay setters.";

    const auto accepted_rows_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromDeviceAcceptedState(",
        "void DeviceGraphOrchestrator::noteMainForwardHiddenProducedForMTP(");
    const auto compact_accepted_rows =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(accepted_rows_body));
    EXPECT_NE(compact_accepted_rows.find("MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES"),
              std::string::npos)
        << "Accepted terminal-hidden publication must consume the row indices derived from compact device metadata.";
    EXPECT_NE(compact_accepted_rows.find("mtp_spec_decode_metadata_binding_.hasWorkspace()"),
              std::string::npos)
        << "The helper must fail clearly when the persistent MTP metadata workspace is not bound.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, RequestBatchedPrefillTerminalRowsStayDeviceOwned)
{
    const auto graph_source =
        readFile(repoRoot() / "src/v2/models/qwen/QwenGraphBase.cpp");
    const auto orchestrator_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/HiddenStateRowsSelectStage.cpp");

    const auto graph_row_select = sliceBetween(
        graph_source,
        "TensorBase *QwenGraphBase::maybeAddLMHeadRowSelect(",
        "bool GraphConfig::hasUnifiedPP() const");
    const auto compact_graph_row_select =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(graph_row_select));
    EXPECT_NE(
        compact_graph_row_select.find(
            "DeviceRowIndexSource::RequestTerminalLengths"),
        std::string::npos)
        << "Request-batched prefill must select terminal rows from resident request lengths.";
    EXPECT_NE(
        compact_graph_row_select.find(
            "row_params.request_sequence_lengths_device=request_sequence_lengths_device"),
        std::string::npos);
    EXPECT_NE(
        compact_graph_row_select.find(
            "row_params.request_row_stride=request_row_stride"),
        std::string::npos);

    const auto metadata_prepare = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(",
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(");
    const auto compact_metadata_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(metadata_prepare));
    EXPECT_NE(
        metadata_prepare.find(
            "\"request_batched_prefill_resident_lengths\""),
        std::string::npos)
        << "Production perf telemetry must identify the resident-length row path.";
    EXPECT_EQ(
        compact_metadata_prepare.find(
            "uploadMTPSpecDecodeVerifierLogitRows(request_batched"),
        std::string::npos)
        << "Request prefill must never upload host terminal rows into verifier metadata.";
    EXPECT_EQ(
        orchestrator_source.find("request_batched_prefill_logit_rows_"),
        std::string::npos)
        << "The retired host terminal-row shadow must not return.";

    const auto compact_stage =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(stage_source));
    EXPECT_NE(
        compact_stage.find("launchRequestTerminalRowsSelectFP32("),
        std::string::npos)
        << "The stage must dispatch the native device-length row-packing kernel.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPDeviceResidentPublicationMetadataStaysOnVerifierStream)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto runner_interface =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto direct_publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    const auto compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));

    EXPECT_NE(compact.find("!request.outcome.stream"), std::string::npos)
        << "Device-resident publication metadata must reject implicit/default GPU streams.";
    EXPECT_NE(compact.find("request.outcome.mtp_transaction"),
              std::string::npos)
        << "GPU resident publication must require the child-local transaction lease.";
    EXPECT_NE(compact.find("waitForDeviceResidentMTPTransaction("),
              std::string::npos)
        << "Publication must wait on the transaction's latest device fence.";
    EXPECT_NE(compact.find("forward_engine_->lastAllPositionVerifierForwardGraph()"),
              std::string::npos)
        << "The metadata helper must validate against the retained all-position "
           "verifier graph that produced the compact outcome.";
    EXPECT_NE(compact.find("mtp_spec_decode_metadata_binding_.setShape(shape)"),
              std::string::npos)
        << "Publication metadata must be declared through the persistent metadata workspace shape.";
    EXPECT_NE(compact.find("ensureDeviceWorkspaceAllocated(*verifier_graph->graph"),
              std::string::npos)
        << "Publication metadata buffers must come from the graph workspace allocator.";
    EXPECT_EQ(compact.find("hostToDeviceOnStream(ptrs.base_cached_tokens"),
              std::string::npos)
        << "Resident MTP publication must consume the pre-verifier device "
           "base-cache snapshot instead of uploading host base counts.";
    EXPECT_EQ(runner_interface.find("base_cached_tokens_device"),
              std::string::npos)
        << "Publication requests must not expose an alternate cache-count pointer.";
    EXPECT_NE(compact.find("mtp_publication_base_cache_snapshot_ready_"),
              std::string::npos)
        << "Publication must fail closed when the pre-verifier base-cache "
           "snapshot was not staged.";
    EXPECT_NE(compact.find("enqueueDeriveSpeculativePublicationMetadata("),
              std::string::npos)
        << "Accepted rows and target cache counts must be derived by the backend on device.";
    EXPECT_NE(compact.find("ptrs.accepted_state_slot_indices"),
              std::string::npos);
    EXPECT_NE(compact.find("ptrs.target_cached_tokens"),
              std::string::npos);
    EXPECT_NE(compact.find("ptrs.accepted_state_counts"),
              std::string::npos);
    EXPECT_NE(compact.find("ptrs.publication_ok_flags"),
              std::string::npos);
    EXPECT_NE(compact.find("ptrs.next_condition_tokens"),
              std::string::npos)
        << "The same derivation pass should stage the next decode/sidecar condition token.";
    EXPECT_NE(compact.find("ptrs.all_drafts_accepted_flags"),
              std::string::npos)
        << "The derivation pass must keep the all-drafts-accepted predicate device-resident.";
    EXPECT_NE(compact.find("ptrs.stopped_flags"),
              std::string::npos)
        << "The derivation pass must keep stop-token state device-resident.";
    EXPECT_NE(compact.find("request.outcome.output_tokens_device"),
              std::string::npos)
        << "Next-token staging must consume the resident compact output-token buffer, not the host bridge.";
    EXPECT_EQ(compact.find("recordDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos)
        << "The logical-state mailbox readiness event must be recorded by the "
           "publication endpoint after KV and shifted-MTP KV publication are enqueued.";
    EXPECT_EQ(compact.find("copyDeviceSpeculativeOutcomesToHost("),
              std::string::npos)
        << "This preflight is the replacement for the host bridge dependency, not another caller of it.";
    EXPECT_EQ(compact.find("synchronizeStream("), std::string::npos)
        << "Metadata preparation must enqueue work only; the owner decides when an output flush synchronizes.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPDeviceResidentPublicationRequiresAtomicKVAndLogicalState)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto support_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const auto publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    const auto compact_support =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(support_body));
    const auto compact_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publish_body));

    EXPECT_EQ(compact_support.find("supportsMTPSpecStatePublication()"),
              std::string::npos)
        << "Device-resident grouped-outcome publication is a resident handoff "
           "capability, not permission to promote the older direct all-position "
           "verifier policy.";
    EXPECT_NE(compact_support.find("supportsDeviceResidentLogicalSequenceStatePublication()"),
              std::string::npos)
        << "DGO must advertise resident publication only when the logical-state "
           "mailbox can expose the same device-owned sequence state as KV.";
    EXPECT_NE(compact_publish.find("prepareDeviceResidentMTPSpecPublicationMetadata("),
              std::string::npos)
        << "The direct endpoint should exercise the device metadata preflight "
           "before reporting the remaining unsupported handoff.";
    EXPECT_EQ(compact_publish.find("request.request_count!=1&&mtpSpecStatePublicationRequiresCapturedStage()"),
              std::string::npos)
        << "CUDA/ROCm GDN and short-conv now own per-request verifier live-state "
           "banks; resident request batches must use the batch restore hook "
           "instead of failing at the old scalar ownership guard.";
    EXPECT_EQ(publish_body.find("per-request GDN/short-conv live-state storage"),
              std::string::npos)
        << "The stale scalar-only ownership error must not remain in the "
           "promoted resident publication path.";
    EXPECT_NE(compact_publish.find("waitForPendingShiftedMTPKVReady("),
              std::string::npos)
        << "Direct publication mutates shifted MTP KV and must order after any "
           "deferred sidecar append on the verifier stream.";
    EXPECT_NE(compact_publish.find("supportsDeviceResidentSequenceStatePublication()"),
              std::string::npos)
        << "DGO must ask the KV cache whether it can consume device-derived "
           "target cache counts before advertising direct publication.";
    EXPECT_NE(compact_publish.find("supportsDeviceResidentLogicalSequenceStatePublication()"),
              std::string::npos)
        << "DGO must separately gate its host-owned logical positions and graph "
           "signature state before mutating KV from resident metadata.";
    EXPECT_NE(compact_publish.find("DeviceSequenceStatePublicationRequestkv_request;"),
              std::string::npos);
    EXPECT_NE(compact_publish.find("kv_request.target_cached_tokens_device=ptrs.target_cached_tokens"),
              std::string::npos);
    EXPECT_NE(compact_publish.find("kv_request.accepted_state_counts_device=ptrs.accepted_state_counts"),
              std::string::npos);
    EXPECT_NE(compact_publish.find("kv_request.publication_ok_flags_device=ptrs.publication_ok_flags"),
              std::string::npos);
    EXPECT_NE(compact_publish.find("publishSequenceStateFromDeviceMetadata("),
              std::string::npos)
        << "The direct endpoint should already be wired to the cache-side "
           "publication hook, even while DGO-level publication stays disabled.";
    EXPECT_LT(compact_publish.find("supportsDeviceResidentLogicalSequenceStatePublication()"),
              compact_publish.find("publishSequenceStateFromDeviceMetadata("))
        << "Logical-position publication must be supported before KV mutation is queued.";
    EXPECT_NE(compact_publish.find("enqueueDeriveShiftedSpeculativePublicationMetadata("),
              std::string::npos)
        << "Direct publication must derive per-depth shifted MTP KV target and delta counts on device.";
    EXPECT_NE(compact_publish.find("ptrs.shifted_target_cached_tokens"),
              std::string::npos);
    EXPECT_NE(compact_publish.find("ptrs.shifted_accepted_state_counts"),
              std::string::npos);
    EXPECT_NE(compact_publish.find("state_.mtp_kv_caches"),
              std::string::npos)
        << "Direct publication must update shifted MTP KV caches as part of the atomic handoff.";
    EXPECT_NE(compact_publish.find("MTPDeviceVerifierStatePublicationShapedevice_state_shape"),
              std::string::npos)
        << "Resident GPU publication should describe request shape without "
           "building synthetic host step plans.";
    EXPECT_EQ(compact_publish.find("MTPSpecStepPlandirect_state_plan"),
              std::string::npos)
        << "Resident GPU publication must not synthesize host scalar plans for recurrent state.";
    EXPECT_EQ(compact_publish.find("MTPSpecStepPlanBatchdirect_state_batch"),
              std::string::npos)
        << "Resident GPU publication must not synthesize host batch plans for recurrent state.";
    EXPECT_NE(compact_publish.find("publishAcceptedMTPSpecStateFromDeviceVerifierRow("),
              std::string::npos)
        << "Scalar direct publication keeps the single-row device-indexed helper.";
    EXPECT_NE(compact_publish.find("publishAcceptedMTPSpecStateFromDeviceVerifierRows("),
              std::string::npos)
        << "Request-batched resident publication must restore GDN/short-conv "
           "state through the batch hook once, not by looping the scalar helper.";
    EXPECT_NE(compact_publish.find("ptrs.accepted_state_slot_indices"),
              std::string::npos)
        << "Direct recurrent-state publication must consume compact GPU row metadata.";
    EXPECT_NE(compact_publish.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState("),
              std::string::npos)
        << "Direct publication must publish terminal hidden from device-derived "
           "accepted verifier rows before exposing the mailbox.";
    EXPECT_NE(compact_publish.find("handleLivePrefixReplayStateAfterMutation("),
              std::string::npos)
        << "Direct publication must advance the live replay epoch before "
           "recording a resident logical-state mailbox for consumers.";
    EXPECT_NE(compact_publish.find("recordDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos)
        << "The logical-state mailbox must be recorded after the full resident publication is enqueued.";
    EXPECT_LT(compact_publish.find("publishSequenceStateFromDeviceMetadata("),
              compact_publish.find("recordDeviceResidentLogicalSequenceStateMailbox("))
        << "Mailbox consumers must wait for KV publication, not just metadata derivation.";
    EXPECT_LT(compact_publish.find("publishAcceptedMTPSpecStateFromDeviceVerifierRow("),
              compact_publish.find("recordDeviceResidentLogicalSequenceStateMailbox("))
        << "Mailbox consumers must wait for recurrent-state publication.";
    EXPECT_LT(compact_publish.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState("),
              compact_publish.find("recordDeviceResidentLogicalSequenceStateMailbox("))
        << "Mailbox consumers must wait for terminal-hidden publication.";
    EXPECT_LT(compact_publish.find("handleLivePrefixReplayStateAfterMutation("),
              compact_publish.find("recordDeviceResidentLogicalSequenceStateMailbox("))
        << "Mailbox epoch must match the live state produced by direct publication.";
    EXPECT_NE(publish_body.find("device_resident_kv_sequence_state_publications"),
              std::string::npos)
        << "Successful direct publication should be visible in perf counters.";
    EXPECT_NE(compact_publish.find("returntrue;"),
              std::string::npos)
        << "After the gated device KV publication succeeds, the direct endpoint "
           "must succeed so the runner can adopt host mirrors from the plan.";
    EXPECT_EQ(compact_publish.find("copyDeviceSpeculativeOutcomesToHost("),
              std::string::npos)
        << "Direct publication must not quietly fall back to the "
           "compatibility host bridge.";
    EXPECT_EQ(compact_publish.find("materializeDeviceSpeculativeOutcomesForHostPlan("),
              std::string::npos)
        << "Direct publication must not depend on host-plan materialization.";
    EXPECT_EQ(compact_publish.find("materializeDeviceSpeculativeOutcomesForHostResponse("),
              std::string::npos)
        << "Direct publication must not depend on host-response materialization.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, DGODeviceLogicalStateMailboxWrapsResidentMetadata)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto orchestration_source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto orchestration_header =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.h");
    const auto runner_interface =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto stage_interface =
        readFile(repoRoot() / "src/v2/execution/compute_stages/IComputeStage.h");
    const auto attention_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/AttentionComputeStage.h");
    const auto attention_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp");
    const auto embedding_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/EmbeddingStage.h");
    const auto kv_append_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/KVCacheAppendStage.h");
    const auto gdn_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h");
    const auto shortconv_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ShortConv1dStage.h");
    const auto rope_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/RoPEStage.h");
    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto compact_orchestration_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(orchestration_header));
    const auto compact_runner_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(runner_interface));
    const auto compact_stage_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(stage_interface));
    const auto compact_attention_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(attention_header));
    const auto compact_attention_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(attention_source));
    const auto compact_embedding_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(embedding_header));
    const auto compact_kv_append_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(kv_append_header));
    const auto compact_gdn_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(gdn_header));
    const auto compact_shortconv_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(shortconv_header));
    const auto compact_rope_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rope_header));
    const auto view_body = sliceBetween(
        source,
        "DeviceGraphOrchestrator::deviceResidentLogicalSequenceState() const",
        "bool DeviceGraphOrchestrator::recordDeviceResidentLogicalSequenceStateMailbox(");
    const auto record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordDeviceResidentLogicalSequenceStateMailbox(",
        "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateMailbox(");
    const auto wait_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateMailbox(",
        "bool DeviceGraphOrchestrator::retargetDeviceResidentLogicalSequenceStateMailboxAfterShiftedKVMutation(");
    const auto retarget_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::retargetDeviceResidentLogicalSequenceStateMailboxAfterShiftedKVMutation(",
        "bool DeviceGraphOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const");
    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    const auto resident_sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(",
        "bool DeviceGraphOrchestrator::forwardMTPAndSampleGreedy(");
    const auto live_prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(",
        "const float *DeviceGraphOrchestrator::getAllPositionLogits() const");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto direct_publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    const auto compact_view =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(view_body));
    const auto compact_record =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));
    const auto compact_wait =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(wait_body));
    const auto compact_retarget =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(retarget_body));
    const auto compact_sidecar =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_body));
    const auto compact_resident_sidecar =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(resident_sidecar_body));
    const auto compact_live_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(live_prepare_body));
    const auto compact_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_direct_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(direct_publish_body));
    const auto clear_mailbox_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearDeviceResidentLogicalSequenceStateMailbox()",
        "DeviceResidentLogicalSequenceStateHandle");
    const auto compact_clear_mailbox =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(clear_mailbox_body));
    const auto planner_helper_body = sliceBetween(
        orchestration_source,
        "std::optional<int> OrchestrationRunner::currentMTPBaseSidecarPositionForPlanning(",
        "void OrchestrationRunner::recordMTPDepthZeroBypass()");
    const auto compact_planner_helper =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(planner_helper_body));
    const auto decode_mtp_body = sliceBetween(
        orchestration_source,
        "GenerationResult OrchestrationRunner::decodeStepMTP()",
        "GenerationResult OrchestrationRunner::decodeStep()");
    const auto compact_decode_mtp =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(decode_mtp_body));
    const auto decode_step_body = sliceBetween(
        orchestration_source,
        "GenerationResult OrchestrationRunner::decodeStep()",
        "void OrchestrationRunner::setDecodeStepTokenBudget(");
    const auto compact_decode_step =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(decode_step_body));

    EXPECT_NE(compact_runner_interface.find("structDeviceResidentLogicalSequenceStateHandle"),
              std::string::npos);
    EXPECT_NE(compact_runner_interface.find("virtualDeviceResidentLogicalSequenceStateHandledeviceResidentLogicalSequenceState()const"),
              std::string::npos)
        << "The resident logical state handoff must be a typed runner contract.";
    EXPECT_NE(compact_runner_interface.find("virtualboolforwardMTPFromDeviceResidentLogicalStateForDeviceSampling("),
              std::string::npos)
        << "The next sidecar row must have a typed resident-state entry point.";
    EXPECT_EQ(compact_runner_interface.find("virtualbooladoptDeviceResidentMTPSpecPublishedHostState("),
              std::string::npos)
        << "Direct device publication must not expose a backend host-adoption hook.";
    EXPECT_EQ(compact_runner_interface.find("hostLogicalStateMirrorsDeviceResidentState"),
              std::string::npos)
        << "Resident state ownership must not be weakened by a host-mirror freshness API.";
    EXPECT_NE(compact_runner_interface.find("stream!=nullptr"),
              std::string::npos)
        << "Consumers must not be able to treat the default/null stream as valid.";
    EXPECT_NE(compact_runner_interface.find("ready_event!=nullptr"),
              std::string::npos)
        << "The resident logical-state handle must expose an event for stream-ordered waits.";
    EXPECT_NE(compact_runner_interface.find("boolcoversRequest(intrequest_index)const"),
              std::string::npos)
        << "Resident logical-state handles should centralize validity and bounds checks.";
    EXPECT_NE(compact_runner_interface.find("nextConditionTokenDeviceForRequest(intrequest_index)const"),
              std::string::npos);
    EXPECT_NE(compact_runner_interface.find("targetPositionDeviceForRequest(intrequest_index)const"),
              std::string::npos);

    EXPECT_NE(compact_header.find("structDeviceResidentLogicalSequenceStateMailbox"),
              std::string::npos);
    EXPECT_EQ(compact_header.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "DGO must keep one logical-state owner, not a resident mailbox plus a host freshness epoch.";
    EXPECT_NE(compact_header.find("boolownsHandle(constDeviceResidentLogicalSequenceStateHandle&handle,uint64_tcurrent_live_state_epoch)const"),
              std::string::npos)
        << "Mailbox ownership must be checked structurally, not open-coded by each consumer.";
    EXPECT_NE(compact_header.find("retargetDeviceResidentLogicalSequenceStateMailboxAfterShiftedKVMutation("),
              std::string::npos)
        << "Resident shifted-KV commits need a typed way to keep mailbox ownership "
           "current after the live epoch advances.";
    EXPECT_NE(compact_header.find("DeviceResidentLogicalSequenceStateHandledeviceResidentLogicalSequenceState()constoverride"),
              std::string::npos);
    EXPECT_NE(compact_header.find("forwardMTPFromDeviceResidentLogicalStateForDeviceSampling("),
              std::string::npos);
    EXPECT_NE(compact_header.find("target_positions_device=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_header.find("target_sequence_lengths_device=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_header.find("accepted_state_counts_device=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_header.find("next_condition_tokens_device=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_header.find("all_drafts_accepted_flags_device=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_header.find("stopped_flags_device=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_header.find("publication_ok_flags_device=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_header.find("producer_stream=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_header.find("std::shared_ptr<void>ready_event"),
              std::string::npos);
    EXPECT_NE(compact_header.find("live_state_epoch=0"),
              std::string::npos);
    EXPECT_NE(compact_header.find("handle.target_sequence_lengths_device==target_sequence_lengths_device"),
              std::string::npos)
        << "Ownership checks must include the sequence-length pointer, not just positions.";
    EXPECT_NE(compact_header.find("handle.accepted_state_counts_device==accepted_state_counts_device"),
              std::string::npos)
        << "Ownership checks must include the accepted-state count pointer.";
    EXPECT_NE(compact_header.find("handle.all_drafts_accepted_flags_device==all_drafts_accepted_flags_device"),
              std::string::npos)
        << "Ownership checks must include the all-drafts-accepted predicate pointer.";
    EXPECT_NE(compact_header.find("handle.stopped_flags_device==stopped_flags_device"),
              std::string::npos)
        << "Ownership checks must include the stop-token predicate pointer.";
    EXPECT_NE(compact_header.find("handle.ready_event==ready_event.get()"),
              std::string::npos);

    EXPECT_NE(compact_view.find("mailbox.live_state_epoch!=live_replay_state_epoch_"),
              std::string::npos)
        << "Stale workspace mailbox pointers must not be exposed after live-state mutation.";
    EXPECT_NE(compact_view.find("handle.target_positions_device=mailbox.target_positions_device"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.target_sequence_lengths_device=mailbox.target_sequence_lengths_device"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.accepted_state_counts_device=mailbox.accepted_state_counts_device"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.next_condition_tokens_device=mailbox.next_condition_tokens_device"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.all_drafts_accepted_flags_device=mailbox.all_drafts_accepted_flags_device"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.stopped_flags_device=mailbox.stopped_flags_device"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.publication_ok_flags_device=mailbox.publication_ok_flags_device"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.device=state_.device_id"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.stream=mailbox.producer_stream"),
              std::string::npos);
    EXPECT_NE(compact_view.find("handle.ready_event=mailbox.ready_event.get()"),
              std::string::npos);
    EXPECT_EQ(compact_view.find("hostLogicalStateMirrorsDeviceResidentState"),
              std::string::npos);
    EXPECT_EQ(compact_view.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "Exposing a resident handle must not consult an obsolete host-mirror epoch.";

    EXPECT_NE(compact_record.find("!producer_stream"), std::string::npos)
        << "The mailbox must preserve explicit stream ownership.";
    EXPECT_NE(compact_record.find("backend->createEvent("),
              std::string::npos);
    EXPECT_NE(compact_record.find("backend->recordEvent("),
              std::string::npos)
        << "Mailbox readiness must be a stream-ordered event, not a host sync.";
    EXPECT_EQ(compact_record.find("synchronizeStream("),
              std::string::npos)
        << "Recording a resident logical-state mailbox must not synchronize.";
    EXPECT_NE(compact_record.find("mailbox.target_positions_device=ptrs.target_cached_tokens"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.target_sequence_lengths_device=ptrs.target_cached_tokens"),
              std::string::npos)
        << "For this phase, target cached tokens are both next position and sequence length.";
    EXPECT_NE(compact_record.find("mailbox.accepted_state_counts_device=ptrs.accepted_state_counts"),
              std::string::npos)
        << "The mailbox must expose the accepted-state count that defines the correction replay boundary.";
    EXPECT_NE(compact_record.find("mailbox.next_condition_tokens_device=ptrs.next_condition_tokens"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.all_drafts_accepted_flags_device=ptrs.all_drafts_accepted_flags"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.stopped_flags_device=ptrs.stopped_flags"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.publication_ok_flags_device=ptrs.publication_ok_flags"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.ready_event=std::move(ready_event)"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.live_state_epoch=live_replay_state_epoch_"),
              std::string::npos);
    EXPECT_NE(compact_record.find("device_resident_logical_sequence_state_mailbox_=mailbox"),
              std::string::npos);
    EXPECT_EQ(compact_record.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "Recording a resident mailbox must establish ownership directly.";
    EXPECT_EQ(compact_clear_mailbox.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "Mailbox cleanup must not maintain a retired host freshness owner.";
    EXPECT_NE(record_body.find("device_resident_logical_state_mailboxes"),
              std::string::npos)
        << "Mailbox creation should be visible in perf counters.";

    EXPECT_NE(compact_wait.find("mailbox.live_state_epoch!=live_replay_state_epoch_"),
              std::string::npos);
    EXPECT_NE(compact_wait.find("backend->streamWaitEvent("),
              std::string::npos)
        << "Mailbox consumers must wait on the producer event from their own explicit stream.";
    EXPECT_EQ(compact_wait.find("synchronizeStream("),
              std::string::npos)
        << "Mailbox consumption must not synchronize the host.";
    EXPECT_NE(wait_body.find("device_resident_logical_state_mailbox_waits"),
              std::string::npos)
        << "Mailbox waits should be visible in perf counters.";

    EXPECT_NE(compact_retarget.find("mailbox.ownsHandle(handle,previous_epoch)"),
              std::string::npos)
        << "Mailbox retarget must only accept the pre-mutation current mailbox.";
    EXPECT_NE(compact_retarget.find("backend->recordEvent("),
              std::string::npos)
        << "Retargeting must publish a new stream-ordered readiness event.";
    EXPECT_NE(compact_retarget.find("mailbox.producer_stream=producer_stream"),
              std::string::npos)
        << "Retargeting must hand ownership to the shifted-KV commit stream.";
    EXPECT_NE(compact_retarget.find("mailbox.live_state_epoch=live_replay_state_epoch_"),
              std::string::npos)
        << "Retargeting must refresh the mailbox epoch instead of weakening stale-handle checks.";
    EXPECT_EQ(compact_retarget.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "Shifted-KV retargets must carry only the resident ownership contract.";
    EXPECT_NE(retarget_body.find("device_resident_logical_state_mailbox_retargets"),
              std::string::npos)
        << "Retargets should be visible in perf counters.";

    EXPECT_NE(compact_sidecar.find("constvoid*position_ids_device_override"),
              std::string::npos)
        << "Sidecar replay must be able to consume device-resident position rows.";
    EXPECT_NE(compact_sidecar.find("input.position_ids_device=effective_position_ids_device"),
              std::string::npos);
    EXPECT_NE(compact_sidecar.find("cached_input.position_ids_device=effective_position_ids_device"),
              std::string::npos);
    EXPECT_NE(compact_sidecar.find("waitForDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos)
        << "Sidecar replay must wait before reading resident next-token/position rows.";
    EXPECT_NE(compact_sidecar.find("stage->updateDynamicDevicePositionIds("),
              std::string::npos)
        << "Device position rows must reach dynamic graph-replay stages.";
    EXPECT_NE(compact_sidecar.find("!stage->supportsDeviceResidentDynamicPositionReplay()"),
              std::string::npos)
        << "Resident position replay must hard-fail before dynamic stages that still need host scalar positions.";
    EXPECT_LT(compact_sidecar.find("!stage->supportsDeviceResidentDynamicPositionReplay()"),
              compact_sidecar.find("stage->updateDynamicDevicePositionIds("))
        << "The sidecar must validate device-position support before binding resident rows.";
    EXPECT_NE(compact_sidecar.find("stage->type()==ComputeStageType::ROPE"),
              std::string::npos)
        << "Device positions replace only RoPE host row uploads; other dynamic stages still refresh normally.";
    EXPECT_NE(compact_sidecar.find("scalar_position_for_dynamic_params=use_device_position_ids?0:position_id"),
              std::string::npos)
        << "Device-resident replay must not feed host-shadow positions into scalar dynamic params.";

    EXPECT_NE(compact_stage_interface.find("virtualboolsupportsDeviceResidentDynamicPositionReplay()const"),
              std::string::npos)
        << "Device-resident position replay needs an explicit stage contract.";
    EXPECT_NE(compact_stage_interface.find("returnfalse;"),
              std::string::npos)
        << "The default stage contract must be a hard-fail, not an optimistic fallback.";
    EXPECT_NE(compact_attention_header.find("boolsupportsDeviceResidentDynamicPositionReplay()constoverride"),
              std::string::npos)
        << "Attention must explicitly opt in only when it can derive metadata from device cache state.";
    EXPECT_NE(compact_attention_source.find("deviceCachedTokenCountPtr(params_.layer_idx,0)!=nullptr"),
              std::string::npos)
        << "Attention device-position replay must be backed by device-resident KV counters.";
    EXPECT_EQ(compact_attention_source.find("!tq_cache&&"),
              std::string::npos)
        << "TQ now shares the canonical device count path and must not be excluded by cache type.";
    EXPECT_NE(compact_embedding_header.find("boolsupportsDeviceResidentDynamicPositionReplay()constoverride"),
              std::string::npos);
    EXPECT_NE(compact_kv_append_header.find("boolsupportsDeviceResidentDynamicPositionReplay()constoverride"),
              std::string::npos);
    EXPECT_NE(compact_gdn_header.find("boolsupportsDeviceResidentDynamicPositionReplay()constoverride"),
              std::string::npos);
    EXPECT_NE(compact_shortconv_header.find("boolsupportsDeviceResidentDynamicPositionReplay()constoverride"),
              std::string::npos);
    EXPECT_NE(compact_rope_header.find("boolsupportsDeviceResidentDynamicPositionReplay()constoverride"),
              std::string::npos);

    EXPECT_NE(compact_resident_sidecar.find("logical_state.coversRequest(request_index)"),
              std::string::npos)
        << "The sidecar consumer must bounds-check through the resident handle.";
    EXPECT_NE(compact_resident_sidecar.find("mailbox.ownsHandle(logical_state,live_replay_state_epoch_)"),
              std::string::npos)
        << "The sidecar consumer must only accept the current runner-owned mailbox.";
    EXPECT_NE(compact_resident_sidecar.find("logical_state.nextConditionTokenDeviceForRequest(request_index)"),
              std::string::npos);
    EXPECT_NE(compact_resident_sidecar.find("logical_state.targetPositionDeviceForRequest(request_index)"),
              std::string::npos);
    EXPECT_EQ(compact_resident_sidecar.find("next_condition_tokens_device+request_index"),
              std::string::npos)
        << "Resident-state consumers should not open-code request-row pointer arithmetic.";
    EXPECT_EQ(compact_resident_sidecar.find("target_positions_device+request_index"),
              std::string::npos)
        << "Resident-state consumers should not open-code request-row pointer arithmetic.";
    EXPECT_EQ(compact_resident_sidecar.find("state_.positions"),
              std::string::npos)
        << "Resident-state sidecar replay must consume device position rows instead of host-shadow positions.";
    EXPECT_NE(compact_resident_sidecar.find("executeMTPDepth0Batched("),
              std::string::npos)
        << "The resident-state handoff should feed the normal sidecar graph path.";

    EXPECT_NE(compact_live_prepare.find("accepted_spec_publication_ready_.valid"),
              std::string::npos);
    EXPECT_NE(compact_live_prepare.find("device_resident_logical_sequence_state_mailbox_.valid()"),
              std::string::npos)
        << "Forward graph live-state preparation must not ignore mailbox-only handoffs.";
    EXPECT_NE(compact_live_prepare.find("waitForDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos);

    EXPECT_NE(compact_prepare.find("clearDeviceResidentLogicalSequenceStateMailbox();"),
              std::string::npos)
        << "Prepare must invalidate any stale mailbox before validation can fail.";
    EXPECT_EQ(compact_prepare.find("recordDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos)
        << "Prepare must not record readiness before KV and shifted-MTP KV publication are enqueued.";
    EXPECT_NE(compact_direct_publish.find("recordDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos);
    EXPECT_LT(compact_direct_publish.find("publishSequenceStateFromDeviceMetadata("),
              compact_direct_publish.find("recordDeviceResidentLogicalSequenceStateMailbox("))
        << "The mailbox readiness event must cover resident KV publication, not just metadata derivation.";
    EXPECT_EQ(compact_header.find("adoptDeviceResidentMTPSpecPublishedHostState("),
              std::string::npos);
    EXPECT_EQ(compact_header.find("adoptDeviceResidentMTPSpecPublishedHostStateFromDeviceMetadata("),
              std::string::npos);
    EXPECT_EQ(compact_direct_publish.find("adoptSequenceStateFromHostMetadata("),
              std::string::npos)
        << "Direct resident publication must not repair host KV mirrors after device publication.";
    EXPECT_EQ(source.find("device_resident_host_state_metadata_d2h_wait"),
              std::string::npos)
        << "Resident logical-state freshness must not require a host metadata D2H wait.";

    const auto reset_body = sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override");
    EXPECT_NE(removeAsciiWhitespace(stripCommentsAndStringLiterals(reset_body))
                  .find("clearDeviceResidentLogicalSequenceStateMailbox();"),
              std::string::npos)
        << "Session resets must invalidate mailbox pointers into workspace buffers.";

    const auto clear_state_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearInferenceState()",
        "// =========================================================================");
    EXPECT_NE(removeAsciiWhitespace(stripCommentsAndStringLiterals(clear_state_body))
                  .find("clearDeviceResidentLogicalSequenceStateMailbox();"),
              std::string::npos)
        << "Inference-state resets must invalidate mailbox pointers into workspace buffers.";

    EXPECT_NE(compact_orchestration_header.find("currentMTPBaseSidecarPositionForPlanning("),
              std::string::npos)
        << "MTP planning reads of get_position() should be centralized.";
    EXPECT_NE(compact_planner_helper.find("runner_->deviceResidentLogicalSequenceState()"),
              std::string::npos);
    EXPECT_EQ(compact_planner_helper.find("hostLogicalStateMirrorsDeviceResidentState"),
              std::string::npos)
        << "Planning must select the resident route solely from mailbox validity.";
    EXPECT_NE(compact_planner_helper.find("resident_state.valid()"),
              std::string::npos)
        << "A valid resident mailbox must unconditionally own MTP planning.";
    EXPECT_NE(compact_planner_helper.find("runner_->get_position()"),
              std::string::npos)
        << "The helper is the single allowed host-position read for MTP planning.";
    EXPECT_NE(planner_helper_body.find("sidecar_position_planning_host_reads"),
              std::string::npos);
    EXPECT_NE(compact_decode_mtp.find("currentMTPBaseSidecarPositionForPlanning("),
              std::string::npos);
    EXPECT_EQ(compact_decode_mtp.find("runner_->get_position()"),
              std::string::npos)
        << "decodeStepMTP must not bypass the host-mirror freshness guard.";
    EXPECT_NE(compact_decode_step.find("currentMTPBaseSidecarPositionForPlanning("),
              std::string::npos)
        << "Dynamic depth-zero shifted-cache maintenance must use the same planning guard.";
    EXPECT_EQ(compact_decode_step.find("runner_->get_position()"),
              std::string::npos)
        << "decodeStep must not bypass the host-mirror freshness guard for MTP planning.";
}

TEST(Test__GpuWorkspaceAllocationPolicy,
     RequestBatchedGPUMTPSidecarsRemainDeviceOwnedAcrossEveryDepth)
{
    const auto root = repoRoot();
    const auto runner_interface = readFile(
        root / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto orchestrator_source = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto runner_source = readFile(
        root / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto backend_interface = readFile(root / "src/v2/backends/IBackend.h");

    const auto compact_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(runner_interface));
    EXPECT_NE(compact_interface.find(
                  "forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots("),
              std::string::npos);
    EXPECT_NE(compact_interface.find(
                  "forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots("),
              std::string::npos);
    EXPECT_EQ(compact_interface.find(
                  "forwardMTPBatchAndSampleGreedyToDeviceDraftSlots("),
              std::string::npos)
        << "The host-token/device-output transitional API must stay retired.";
    EXPECT_EQ(compact_interface.find(
                  "forwardMTPBatchFromLastDraftAndSampleGreedyToDeviceDraftSlots("),
              std::string::npos)
        << "Chained GPU sidecars must not accept host token or position rows.";

    const auto first_sidecar = removeAsciiWhitespace(stripCommentsAndStringLiterals(
        sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(",
            "bool DeviceGraphOrchestrator::forwardMTPBatchFromLastDraftAndSampleGreedy(")));
    EXPECT_NE(first_sidecar.find("logical_state.next_condition_tokens_device"),
              std::string::npos);
    EXPECT_NE(first_sidecar.find("logical_state.target_positions_device"),
              std::string::npos);
    EXPECT_EQ(first_sidecar.find("out_tokens"), std::string::npos)
        << "The resident first-depth contract has no host token output.";
    EXPECT_EQ(first_sidecar.find("deviceToHost"), std::string::npos);
    EXPECT_EQ(first_sidecar.find("state_.positions"), std::string::npos);

    const auto chained_sidecar = removeAsciiWhitespace(stripCommentsAndStringLiterals(
        sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(",
            "bool DeviceGraphOrchestrator::forwardMTPFromLastDraftAndSampleGreedy(")));
    EXPECT_NE(chained_sidecar.find("stochastic_draft_sample_tokens_dev_"),
              std::string::npos);
    EXPECT_NE(chained_sidecar.find("condition_slot_stride"), std::string::npos);
    EXPECT_NE(chained_sidecar.find("position_offset"), std::string::npos);
    EXPECT_EQ(chained_sidecar.find("out_tokens"), std::string::npos)
        << "The device-slot chained contract has no host token output.";
    EXPECT_EQ(chained_sidecar.find("deviceToHost"), std::string::npos);

    const auto initial_publication = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::initializeDeviceResidentLogicalSequenceStateFromMainBatchSamples(",
            "bool DeviceGraphOrchestrator::sampleMainLogitsBatchRowsOnDevice(")));
    EXPECT_NE(initial_publication.find(
                  "backend->enqueueInitializeMTPDeviceLogicalState("),
              std::string::npos);
    EXPECT_NE(initial_publication.find(
                  "recordDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos);
    EXPECT_EQ(initial_publication.find("hostToDevice"), std::string::npos);
    EXPECT_EQ(initial_publication.find("synchronizeStream"), std::string::npos);

    const auto request_batch_sidecars = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            runner_source,
            "std::vector<int32_t> condition_tokens;",
            "MTPSpecRequestBatchOwner owner;")));
    EXPECT_NE(request_batch_sidecars.find(
                  "forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots("),
              std::string::npos);
    EXPECT_NE(request_batch_sidecars.find(
                  "forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots("),
              std::string::npos);
    EXPECT_NE(request_batch_sidecars.find(
                  "if(!use_request_batch_device_draft_slots&&!runner_->flushPendingMTPWork())"),
              std::string::npos)
        << "Only the CPU grouped lane may retain the explicit sidecar flush.";

    const auto compact_backend =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(backend_interface));
    EXPECT_NE(compact_backend.find("enqueuePrepareMTPBatchedSidecarInputs("),
              std::string::npos);
    EXPECT_NE(compact_backend.find("enqueueInitializeMTPDeviceLogicalState("),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy,
     RequestBatchedStochasticThresholdPositionsRemainDeviceOwned)
{
    const auto root = repoRoot();
    const auto runner_source = readFile(
        root / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto orchestrator_source = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto backend_interface = readFile(root / "src/v2/backends/IBackend.h");
    const auto cuda_sampling = readFile(
        root / "src/v2/kernels/cuda/ops/CUDASamplingKernels.cu");
    const auto rocm_sampling = readFile(
        root / "src/v2/kernels/rocm/ops/ROCmSamplingKernels.hip");

    const auto descriptor_body = sliceBetween(
        runner_source,
        "DeviceStochasticBatchOutcomeRequest descriptor;",
        "descriptor.stop_token_count =");
    const auto threshold_descriptor = sliceBetween(
        descriptor_body,
        "const int base_cached_tokens =",
        "descriptor.draft_tokens[static_cast<size_t>(row)] =");
    const auto resident_descriptor = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            threshold_descriptor,
            "if (use_device_resident_request_batch_publication)",
            "else\n                    {")));
    EXPECT_NE(resident_descriptor.find(
                  "descriptor.inverse_sample_first_logical_position=-1;"),
              std::string::npos)
        << "A resident GPU descriptor must not carry a host logical-position scalar.";
    EXPECT_NE(resident_descriptor.find(
                  "descriptor.derive_thresholds_from_seed=true;"),
              std::string::npos);
    EXPECT_NE(resident_descriptor.find(
                  "descriptor.draw_position_source=DeviceStochasticDrawPositionSource::ResidentLogicalState;"),
              std::string::npos);
    EXPECT_EQ(resident_descriptor.find("descriptor.accept_thresholds["),
              std::string::npos)
        << "GPU request batches must not populate host accept-threshold rows.";
    EXPECT_EQ(resident_descriptor.find("descriptor.residual_thresholds["),
              std::string::npos)
        << "GPU request batches must not populate host residual-threshold rows.";
    EXPECT_EQ(resident_descriptor.find("mtpSpecStochasticThresholdForPosition("),
              std::string::npos)
        << "Resident continuation draws belong to GPU kernels, not the host sampler.";

    const auto request_batch_verifier = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(",
            "bool DeviceGraphOrchestrator::copyDeviceSpeculativeOutcomesToHost(")));
    EXPECT_NE(request_batch_verifier.find(
                  "waitForDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos);
    EXPECT_NE(request_batch_verifier.find(
                  "targetPositionDeviceForRequest(request.request_id)"),
              std::string::npos);
    EXPECT_NE(request_batch_verifier.find(
                  "threshold_base_position_device"),
              std::string::npos);

    const auto common_verifier = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(",
        "bool DeviceGraphOrchestrator::forward_batch(");
    EXPECT_NE(common_verifier.find(
                  "stochastic_request_batch_resident_position_threshold_rows"),
              std::string::npos)
        << "Perfstats must prove that served request batches exercised resident-position draws.";

    const auto prefill_runner = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            runner_source,
            "if (has_ready_prefill_logits)",
            "if (!state.prefill_logits_ready)")));
    EXPECT_NE(prefill_runner.find("device_prefill_position_seeds"),
              std::string::npos);
    EXPECT_EQ(prefill_runner.find("device_prefill_thresholds"),
              std::string::npos);
    EXPECT_EQ(prefill_runner.find("mtp_spec_threshold_from_seed("),
              std::string::npos)
        << "First-token thresholds must be derived from resident positions in GPU kernels.";

    const auto prefill_admission = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::stageRequestBatchSequenceLengthsOnDevice(",
            "bool DeviceGraphOrchestrator::initializeDeviceResidentLogicalSequenceStateFromMainBatchSamples(")));
    EXPECT_NE(prefill_admission.find("request_sequence_lengths_dev_"),
              std::string::npos);
    EXPECT_NE(prefill_admission.find("hostToDeviceOnStream("),
              std::string::npos)
        << "Immutable prompt lengths must cross H2D once at request admission.";
    EXPECT_NE(prefill_admission.find("synchronizeStream("),
              std::string::npos)
        << "Request lengths must be fenced before any graph capture stream reads them.";
    EXPECT_EQ(prefill_admission.find("publishPendingLogitsStream("),
              std::string::npos)
        << "Request-length admission is a pre-capture ownership boundary, not a logits-stream handoff.";

    const auto initial_publication = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::initializeDeviceResidentLogicalSequenceStateFromMainBatchSamples(",
            "bool DeviceGraphOrchestrator::sampleMainLogitsBatchRowsOnDevice(")));
    EXPECT_NE(initial_publication.find(
                  "stochastic_target_sample_tokens_dev_,static_cast<constint32_t*>(request_sequence_lengths_dev_),request_count"),
              std::string::npos);
    EXPECT_NE(initial_publication.find("ptrs.target_cached_tokens"),
              std::string::npos)
        << "Initial publication must derive the device-owned target position mailbox.";
    EXPECT_EQ(initial_publication.find("state_.sequence_lengths"),
              std::string::npos)
        << "Mailbox publication must not re-adopt a host logical-position mirror.";

    const auto prefill_sampler_source = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::sampleMainLogitsBatchRowsOnDevice(",
        "bool DeviceGraphOrchestrator::applyPenaltiesOnDevice(");
    const auto prefill_sampler = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(prefill_sampler_source));
    EXPECT_NE(prefill_sampler.find("stochastic_position_seeds[row]"),
              std::string::npos);
    EXPECT_NE(prefill_sampler.find(
                  "static_cast<constint32_t*>(request_sequence_lengths_dev_)+row"),
              std::string::npos);
    EXPECT_NE(prefill_sampler_source.find(
                  "request_batch_prefill_resident_position_threshold_rows"),
              std::string::npos);
    EXPECT_EQ(prefill_sampler.find("stochastic_thresholds"),
              std::string::npos);

    const auto compact_backend =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(backend_interface));
    EXPECT_NE(compact_backend.find("threshold_base_position_device"),
              std::string::npos);
    EXPECT_NE(compact_backend.find("threshold_position_device"),
              std::string::npos);
    EXPECT_NE(compact_backend.find("target_positions_device"),
              std::string::npos);
    EXPECT_EQ(compact_backend.find("target_positions_host"),
              std::string::npos);

    for (const auto &source : {cuda_sampling, rocm_sampling})
    {
        const auto compact =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(source));
        EXPECT_NE(compact.find(
                      "*threshold_base_position+threshold_position_offset+row"),
                  std::string::npos)
            << "Every GPU verifier must consume the resident base position in-kernel.";
        EXPECT_NE(compact.find(
                      "*threshold_position+threshold_position_offset"),
                  std::string::npos)
            << "Every GPU bonus sampler must consume the resident position in-kernel.";
        EXPECT_NE(compact.find("mtp_spec_threshold_from_seed("),
                  std::string::npos);
        EXPECT_NE(compact.find("constintposition=target_positions[request];"),
                  std::string::npos)
            << "Initial publication must load logical positions from device memory.";
        EXPECT_EQ(compact.find("intposition_0"), std::string::npos)
            << "Host-derived position launch scalars are obsolete.";
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy,
     SeededSerialGroupedVerifierDrawsUseDeviceBaseSnapshot)
{
    const auto root = repoRoot();
    const auto runner_source = readFile(
        root / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto orchestrator_source = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto strict_descriptor = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            runner_source,
            "if (use_serial_sample_equivalent_stochastic)\n                    {",
            "else\n                    {")));
    EXPECT_NE(strict_descriptor.find(
                  "request.inverse_sample_seed=inverse_sample_seed;"),
              std::string::npos);
    EXPECT_NE(strict_descriptor.find(
                  "request.inverse_sample_first_logical_position=-1;"),
              std::string::npos)
        << "The GPU descriptor must not capture a host logical position.";
    EXPECT_NE(strict_descriptor.find(
                  "request.derive_thresholds_from_seed=true;"),
              std::string::npos);
    EXPECT_NE(strict_descriptor.find(
                  "request.draw_position_source=DeviceStochasticDrawPositionSource::VerifierBaseSnapshot;"),
              std::string::npos);
    EXPECT_EQ(strict_descriptor.find("request.sample_thresholds["),
              std::string::npos)
        << "Seeded serial-equivalent GPU rows must not carry host draw arrays.";

    const auto request_batch_reducer = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(",
            "bool DeviceGraphOrchestrator::copyDeviceSpeculativeOutcomesToHost(")));
    EXPECT_NE(request_batch_reducer.find(
                  "publication_metadata.base_cached_tokens+request.request_id"),
              std::string::npos)
        << "The reducer must bind the D2D pre-verifier KV base snapshot.";

    const auto common_reducer_source = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(",
        "bool DeviceGraphOrchestrator::forward_batch(");
    const auto common_reducer = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(common_reducer_source));
    EXPECT_NE(common_reducer.find(
                  "derive_sample_thresholds_from_device_position"),
              std::string::npos);
    EXPECT_NE(common_reducer.find(
                  "threshold_position_offset+row"),
              std::string::npos);
    EXPECT_NE(common_reducer.find(
                  "threshold_position_offset+row_count"),
              std::string::npos)
        << "The bonus row must use the same resident position sequence.";
    EXPECT_NE(common_reducer_source.find(
                  "stochastic_serial_equivalent_resident_position_sample_rows"),
              std::string::npos)
        << "Perfstats must prove that verifier and bonus draws remained resident.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPBudgetLimitedDirectEmitUsesCheckpointShiftedSidecarAnchor)
{
    const auto orchestration_source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto decode_mtp_body = sliceBetween(
        orchestration_source,
        "GenerationResult OrchestrationRunner::decodeStepMTP()",
        "GenerationResult OrchestrationRunner::decodeStep()");
    const auto direct_emit_body = sliceBetween(
        decode_mtp_body,
        "if (speculative_draft_count == 0)",
        "std::optional<PrefixStateSnapshot> verifier_replay_base_checkpoint;");
    const auto compact_direct_emit =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(direct_emit_body));

    EXPECT_NE(compact_direct_emit.find(
                  "constintbase_sidecar_position=snapshotShiftedMTPTokens(verifier_base_checkpoint)+1;"),
              std::string::npos)
        << "Budget-limited direct emit must anchor shifted-KV repair to the "
           "verifier-base checkpoint, because prefix-cache restore can leave "
           "the host runner position staged ahead of the shifted sidecar head.";
    EXPECT_NE(compact_direct_emit.find(
                  "commitMTPShiftedRowFromCurrentTerminalHidden(first_token,0,true,base_sidecar_position)"),
              std::string::npos)
        << "The direct emit shifted-row commit must consume the checkpoint-derived anchor.";
    EXPECT_EQ(compact_direct_emit.find("currentMTPBaseSidecarPositionForPlanning("),
              std::string::npos)
        << "This direct-emit block must not read host position for the shifted "
           "sidecar precondition; it already has the authoritative checkpoint.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPConditionForwardPublishesShiftedSidecarBeforeMainForward)
{
    const auto orchestration_source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto decode_mtp_body = sliceBetween(
        orchestration_source,
        "GenerationResult OrchestrationRunner::decodeStepMTP()",
        "GenerationResult OrchestrationRunner::decodeStep()");
    const auto condition_forward_body = sliceBetween(
        decode_mtp_body,
        "else if (!use_ready_logits)",
        "else if (use_ready_logits)");
    const auto compact_condition_forward =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(condition_forward_body));

    const size_t commit_pos = compact_condition_forward.find(
        "commitMTPShiftedRowFromCurrentTerminalHidden(condition_token,0,true,*condition_sidecar_position)");
    const size_t forward_pos = compact_condition_forward.find(
        "runner_->forward(&condition_token,1)");
    ASSERT_NE(commit_pos, std::string::npos)
        << "Condition-forward MTP decode must publish the shifted sidecar row "
           "for the condition token before advancing main state.";
    ASSERT_NE(forward_pos, std::string::npos);
    EXPECT_LT(commit_pos, forward_pos)
        << "The shifted sidecar row must be committed before the main condition forward.";
    EXPECT_NE(condition_forward_body.find("condition_forward_shifted_commits"),
              std::string::npos)
        << "The maintenance path should remain visible in perfstats.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, DGOResidentPublicationDoesNotMutateKVBeforeLogicalStateIsResident)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto support_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::supportsDeviceResidentLogicalSequenceStatePublication() const",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const auto publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    const auto compact_support =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(support_body));
    const auto compact_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publish_body));

    EXPECT_NE(header.find("supportsDeviceResidentLogicalSequenceStatePublication"),
              std::string::npos)
        << "DGO logical-state publication needs a named gate separate from IKVCache support.";
    EXPECT_NE(compact_support.find("state_.device_id.is_gpu()"),
              std::string::npos);
    EXPECT_NE(compact_support.find("state_.kv_cache!=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_support.find("state_.kv_cache->supportsDeviceResidentSequenceStatePublication()"),
              std::string::npos)
        << "DGO logical-state publication must stay tied to a cache that can "
           "publish device head/count mirrors.";

    const size_t kv_support = compact_publish.find("supportsDeviceResidentSequenceStatePublication()");
    const size_t dgo_support = compact_publish.find("supportsDeviceResidentLogicalSequenceStatePublication()");
    const size_t kv_publish = compact_publish.find("publishSequenceStateFromDeviceMetadata(");
    ASSERT_NE(kv_support, std::string::npos);
    ASSERT_NE(dgo_support, std::string::npos);
    ASSERT_NE(kv_publish, std::string::npos);
    EXPECT_LT(kv_support, dgo_support);
    EXPECT_LT(dgo_support, kv_publish)
        << "KV publication must remain after both the cache and DGO logical-state support gates.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, KVCacheDeviceResidentPublicationContractRequiresDeviceCounts)
{
    const auto source =
        readFile(repoRoot() / "src/v2/kernels/IKVCache.h");
    const auto compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(source));

    EXPECT_NE(compact.find("structDeviceSequenceStatePublicationRequest"),
              std::string::npos);
    EXPECT_EQ(compact.find("structHostSequenceStatePublicationRequest"),
              std::string::npos)
        << "GPU publication must not expose a host state-adoption request.";
    EXPECT_NE(compact.find("target_cached_tokens_device"),
              std::string::npos);
    EXPECT_NE(compact.find("accepted_state_counts_device"),
              std::string::npos);
    EXPECT_NE(compact.find("publication_ok_flags_device"),
              std::string::npos);
    EXPECT_NE(compact.find("stream!=nullptr"),
              std::string::npos)
        << "GPU sequence-state publication must not be expressible on the "
           "default/null stream.";
    EXPECT_NE(compact.find("supportsDeviceResidentSequenceStatePublication()const"),
              std::string::npos);
    EXPECT_EQ(compact.find("adoptSequenceStateFromHostMetadata("),
              std::string::npos)
        << "Direct device publication is authoritative and must not have a host adoption twin.";
    EXPECT_NE(compact.find("returnfalse;"),
              std::string::npos)
        << "The default IKVCache implementation must hard-fail until a cache "
           "owns device-readable count/head state.";
    EXPECT_NE(source.find("Long-context ring caches"), std::string::npos)
        << "The contract should document why target count alone is not enough "
           "to update wrapped ring-cache heads.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GPUKVLogicalBlockAccessRequiresExplicitStreams)
{
    const auto interface_source =
        readFile(repoRoot() / "src/v2/kernels/IKVCache.h");
    const auto cuda_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto rocm_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp");

    const auto cuda_export = sliceBetween(
        cuda_source,
        "bool CUDARingKVCache<Precision>::exportLogicalBlock(",
        "bool CUDARingKVCache<Precision>::importLogicalBlock(");
    const auto cuda_import = sliceBetween(
        cuda_source,
        "bool CUDARingKVCache<Precision>::importLogicalBlock(",
        "bool CUDARingKVCache<Precision>::truncateSequence(");
    const auto rocm_export = sliceBetween(
        rocm_source,
        "bool ROCmRingKVCache<Precision>::exportLogicalBlock(",
        "bool ROCmRingKVCache<Precision>::importLogicalBlock(");
    const auto rocm_import = sliceBetween(
        rocm_source,
        "bool ROCmRingKVCache<Precision>::importLogicalBlock(",
        "bool ROCmRingKVCache<Precision>::truncateSequence(");

    const auto compact_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(interface_source));
    const std::vector<std::string> bodies = {
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_export)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_import)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_export)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_import)),
    };

    EXPECT_NE(interface_source.find("GPU implementations require @ref stream"),
              std::string::npos)
        << "The API contract must tell callers that GPU logical KV access is stream-owned.";
    EXPECT_NE(compact_interface.find("void*stream=nullptr"),
              std::string::npos)
        << "CPU callers may still omit streams; GPU implementations enforce when streams are required.";

    for (const auto &body : bodies)
    {
        EXPECT_NE(body.find("if(!desc.stream)"), std::string::npos)
            << "GPU logical KV import/export must fail fast when no explicit stream is supplied.";
        EXPECT_EQ(body.find("getEffectiveStream(nullptr)"), std::string::npos)
            << "GPU logical KV import/export must not quietly fall back to a default stream.";
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, GPUKVSequenceStateIsCanonicalAndHasNoHostMirror)
{
    const auto cuda_cache_header =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.h");
    const auto cuda_tq_header =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.h");
    const auto rocm_cache_header =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.h");
    const auto rocm_tq_header =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.h");
    const auto cuda_base =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.h") +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.cpp") +
        cuda_cache_header +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu") +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTensorAdapter.cpp") +
        cuda_tq_header +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu");
    const auto rocm_base =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.h") +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp") +
        rocm_cache_header +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp") +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheKernels.hip") +
        rocm_tq_header +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.hip");
    const auto cuda_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_base));
    const auto rocm_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_base));

    for (const auto *compact : {&cuda_compact, &rocm_compact})
    {
        EXPECT_NE(compact->find("d_count_params_"), std::string::npos);
        EXPECT_EQ(compact->find("h_count_params_"), std::string::npos);
        EXPECT_EQ(compact->find("h_head_params_"), std::string::npos);
        EXPECT_EQ(compact->find("h_append_count_params_"), std::string::npos);
        EXPECT_EQ(compact->find("d_append_count_params_"), std::string::npos);
        EXPECT_NE(compact->find("deviceCachedTokenCountPtr("), std::string::npos);
        EXPECT_NE(compact->find("deviceRingHeadPtr("), std::string::npos);
        EXPECT_NE(compact->find("supportsDeviceResidentSequenceStatePublication()constoverride"),
                  std::string::npos);
        EXPECT_NE(compact->find("d_head_params_!=nullptr&&d_count_params_!=nullptr"),
                  std::string::npos)
            << "Caches may publish sequence state only when both canonical device rows exist.";
        EXPECT_EQ(compact->find("refreshHostDeviceParamMirror("), std::string::npos);
        EXPECT_EQ(compact->find("uploadHostDeviceParamMirror("), std::string::npos);
        EXPECT_EQ(compact->find("synchronizeHostSequenceStateFromDevice("),
                  std::string::npos)
            << "Production GPU state must never be adopted into a parallel host authority.";
        EXPECT_EQ(compact->find("entryHead("), std::string::npos);
        EXPECT_EQ(compact->find("entryCount("), std::string::npos);
        EXPECT_EQ(compact->find("setEntryHead("), std::string::npos);
        EXPECT_EQ(compact->find("setEntryCount("), std::string::npos);
        EXPECT_EQ(compact->find("resetEntry("), std::string::npos)
            << "Derived cache entries must not expose a second mutable ring ledger.";
        EXPECT_EQ(compact->find("entry.head"), std::string::npos);
        EXPECT_EQ(compact->find("entry.count"), std::string::npos);
        EXPECT_EQ(compact->find("read_past_host_count"), std::string::npos);
        EXPECT_EQ(compact->find("scratch_valid"), std::string::npos);
        EXPECT_EQ(compact->find("converted_count"), std::string::npos)
            << "GPU conversion validity must be decided from canonical device state.";
        EXPECT_NE(compact->find("append_count_sources_"), std::string::npos)
            << "Padded graphs must bind the persistent arena-owned length row directly.";
        EXPECT_NE(compact->find("kv_sequence_state_advance_kernel"),
                  std::string::npos)
            << "Every append must advance canonical device count/head in graph order.";
        EXPECT_NE(compact->find("kv_sequence_state_advance("),
                  std::string::npos);
        EXPECT_EQ(compact->find("voidring_append_kernel("), std::string::npos)
            << "GPU append must not retain a host-selected static-head kernel.";
        EXPECT_EQ(compact->find("voidring_append_verifier_rows_kernel("),
                  std::string::npos)
            << "Grouped verifier publication must read the canonical device head.";
        EXPECT_EQ(compact->find("voidring_gather_batched_kernel("),
                  std::string::npos)
            << "Grouped reads must not retain host tail/count gather kernels.";
    }

    for (const auto &name : {
             "cuda_ring_append_fp32(",
             "cuda_ring_append_fp16(",
             "cuda_ring_append_bf16(",
             "cuda_ring_append_q8_1("})
    {
        EXPECT_EQ(cuda_compact.find(name), std::string::npos)
            << "CUDA static-head append wrapper must remain retired: " << name;
    }
    for (const auto &name : {
             "hip_ring_append_fp32(",
             "hip_ring_append_fp16(",
             "hip_ring_append_bf16(",
             "hip_ring_append_q8_1(",
             "hip_ring_gather_batched_fp32(",
             "hip_ring_gather_batched_fp16(",
             "hip_ring_gather_batched_bf16(",
             "hip_ring_gather_batched_q8_1("})
    {
        EXPECT_EQ(rocm_compact.find(name), std::string::npos)
            << "ROCm static host-metadata wrapper must remain retired: " << name;
    }
    EXPECT_EQ(rocm_compact.find("allocate_entry("), std::string::npos);
    EXPECT_EQ(rocm_compact.find("free_entry("), std::string::npos)
        << "ROCm pooled KV allocation must fail closed rather than fall back per entry.";

    const auto assert_payload_entry_has_no_ring_ledger = [](const std::string &header,
                                                             const std::string &entry_begin,
                                                             const std::string &entry_end)
    {
        const auto entry = removeAsciiWhitespace(stripCommentsAndStringLiterals(
            sliceBetween(header, entry_begin, entry_end)));
        EXPECT_EQ(entry.find("inthead"), std::string::npos);
        EXPECT_EQ(entry.find("intcount"), std::string::npos);
        EXPECT_EQ(entry.find("tail("), std::string::npos);
        EXPECT_EQ(entry.find("is_wrapped("), std::string::npos);
    };
    assert_payload_entry_has_no_ring_ledger(
        cuda_cache_header, "struct CUDARingKVEntry", "class CUDARingKVCache");
    assert_payload_entry_has_no_ring_ledger(
        rocm_cache_header, "struct ROCmRingKVEntry", "class ROCmRingKVCache");
    assert_payload_entry_has_no_ring_ledger(
        cuda_tq_header, "struct TQEntry", "std::vector<std::vector<TQEntry>> entries_");
    assert_payload_entry_has_no_ring_ledger(
        rocm_tq_header, "struct TQEntry", "struct ScratchBuffer");

    const auto stage =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/KVCacheAppendStage.h") +
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp");
    const auto capture_guard =
        readFile(repoRoot() / "src/v2/execution/local_execution/graph/GraphCaptureGuard.h");
    EXPECT_EQ(stage.find("isGraphCaptureHostBookkeepingActive"), std::string::npos);
    EXPECT_EQ(stage.find("hostBookkeepingAppendTokensForRequest"), std::string::npos);
    EXPECT_EQ(stage.find("request_sequence_lengths ="), std::string::npos);
    EXPECT_EQ(capture_guard.find("tls_graph_capture_host_bookkeeping"),
              std::string::npos)
        << "Graph capture must not expose a switch that revives host-state mutation.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, KVCacheDevicePublicationPrimitiveIsWrappedRingSafe)
{
    const auto cuda_base =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.h") +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.cpp") +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto rocm_base =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.h") +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp") +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheKernels.hip");
    const auto cuda_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_base));
    const auto rocm_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_base));

    for (const auto *compact : {&cuda_compact, &rocm_compact})
    {
        EXPECT_NE(compact->find("publishSequenceStateFromDeviceMetadata("),
                  std::string::npos);
        EXPECT_NE(compact->find("kv_sequence_state_publish_kernel"),
                  std::string::npos);
        EXPECT_NE(compact->find("!stream"),
                  std::string::npos)
            << "Device publication must reject implicit/default streams.";
        EXPECT_NE(compact->find("first_seq_idx+request_count>batch_size"),
                  std::string::npos)
            << "Batch-range validation must happen before enqueueing a publication kernel.";
        EXPECT_NE(compact->find("publication_ok_flags[request_idx]"),
                  std::string::npos)
            << "Invalid verifier outcomes must not mutate KV sequence metadata.";
        EXPECT_NE(compact->find("accepted_state_counts[request_idx]"),
                  std::string::npos)
            << "Accepted-row counts must remain visible for validation/accounting.";
        EXPECT_NE(compact->find("target_cached_tokens[request_idx]"),
                  std::string::npos)
            << "The target cached-token count must drive the published valid window.";
        EXPECT_NE(compact->find("tail=old_head-old_count"),
                  std::string::npos)
            << "A wrapped-ring publication must preserve the current ring tail.";
        EXPECT_NE(compact->find("(tail+target_count)%max_seq_len"),
                  std::string::npos)
            << "A wrapped-ring publication must clamp the head to the accepted target length.";
        EXPECT_NE(compact->find("d_counts[entry_idx]=target_count"),
                  std::string::npos)
            << "Device publication must update the canonical count row for subsequent attention.";
        EXPECT_EQ(compact->find("adoptSequenceStateFromHostMetadata("),
                  std::string::npos)
            << "GPU cache state publication must have one device owner.";
        EXPECT_EQ(compact->find("HostSequenceStatePublicationRequest"),
                  std::string::npos)
            << "Backend implementations must not retain the retired host request type.";
    }

    const auto dgo_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto support_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const auto compact_support =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(support_body));
    EXPECT_NE(compact_support.find("supportsDeviceResidentLogicalSequenceStatePublication()"),
              std::string::npos);
    EXPECT_NE(compact_support.find("supportsDeviceResidentSequenceStatePublication()"),
              std::string::npos)
        << "DGO resident publication must stay tied to cache-side device publication support.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, AttentionDeviceParamsCanDeriveFromDeviceKVState)
{
    const auto interface_source =
        readFile(repoRoot() / "src/v2/tensors/TensorKernels.h");
    const auto stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp");
    const auto cuda_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.h") +
        readFile(repoRoot() / "src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.cpp") +
        readFile(repoRoot() / "src/v2/kernels/cuda/attention/CUDAFlashAttentionKernels.cu");
    const auto rocm_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/attention/ROCmFlashAttentionKernelT.h") +
        readFile(repoRoot() / "src/v2/kernels/rocm/attention/ROCmFlashAttentionKernelT.cpp") +
        readFile(repoRoot() / "src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip");

    const auto interface_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(interface_source));
    const auto stage_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(stage_source));
    const auto update_dynamic_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            stage_source,
            "void AttentionComputeStage::updateDynamicParams(",
            "bool AttentionComputeStage::supportsDeviceResidentDynamicPositionReplay(")));
    const auto execute_device_params_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            stage_source,
            "if (gpu_stage && params_.kv_cache && params_.layer_idx >= 0)",
            "// Get device index using proper ordinal for GPU devices")));
    const auto cuda_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_source));
    const auto rocm_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_source));

    EXPECT_NE(interface_compact.find("prepareDynamicAttnParamsFromDeviceSequenceState("),
              std::string::npos);
    EXPECT_NE(interface_compact.find("returnfalse;"), std::string::npos)
        << "Unsupported attention kernels must hard-fail device-owned sequence-state prep.";

    EXPECT_NE(stage_compact.find("deviceCachedTokenCountPtr(params_.layer_idx,0)"),
              std::string::npos);
    EXPECT_NE(stage_compact.find("prepareDynamicAttnParamsFromDeviceSequenceState("),
              std::string::npos);
    EXPECT_NE(stage_compact.find("dynamic_pre_append_cached_tokens_"),
              std::string::npos)
        << "Attention must make the pre-append KV history a request-local stage boundary.";
    EXPECT_NE(update_dynamic_body.find("dynamic_post_append_kv_len_=std::clamp(pre_append_cached_tokens+logical_seq_len,1,cache_capacity)"),
              std::string::npos);
    EXPECT_EQ(update_dynamic_body.find("kv_len>logical_seq_len"),
              std::string::npos)
        << "Reusable prefill graphs must record device-count derivation even when captured with no prefix history.";
    EXPECT_NE(execute_device_params_body.find("has_current_dynamic_sequence_state||effective_kv_len>logical_seq_len"),
              std::string::npos)
        << "Eager execution may still ignore hostile device counts unless the forward engine established a dynamic boundary.";
    EXPECT_EQ(stage_compact.find("!tq_cache"), std::string::npos)
        << "TQ attention must derive dynamic parameters from the same canonical device count as standard caches.";

    for (const auto *compact : {&cuda_compact, &rocm_compact})
    {
        EXPECT_NE(compact->find("prepare_device_params_from_count"),
                  std::string::npos);
        EXPECT_NE(compact->find("derive_attention_params_from_cached_tokens"),
                  std::string::npos);
        EXPECT_NE(compact->find("dynamic_attn_device_derived_"),
                  std::string::npos)
            << "Backend compute paths must not overwrite device-derived params "
               "with host scalar uploads.";
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPSpecStatePublicationSeparatesSidecarReplaySafety)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto mutation_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::handleLivePrefixReplayStateAfterMutation(",
        "PrefixCacheFingerprintResult DeviceGraphOrchestrator::buildCurrentPrefixFingerprint(");
    const auto sidecar_safety_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::preservesMTPSidecarReplayAfterSpecPublication() const",
        "void DeviceGraphOrchestrator::recordShiftedMTPKVReplayStateMutation(");

    /*
     * MTP spec-state publication replaces live main/MTP KV and recurrent state
     * with verifier-captured rows. Main/verifier replay caches need their own
     * correction boundary, and sidecar replay can only stay warm when the
     * sidecar transaction proves it does not capture stale metadata. GPU main
     * graph replay uses a typed lifetime policy: byte-proven single-token decode
     * and all-position verifier captures remain warm, while genuinely
     * live-state-versioned multi-row ordinary decode is invalidated. This guard
     * prevents blanket graph resets from returning and keeps sidecar safety an
     * independent decision.
     */
    EXPECT_NE(header.find("preservesMTPSidecarReplayAfterSpecPublication"),
              std::string::npos)
        << "DGO must expose a narrow sidecar replay-safety decision separate "
           "from main-state preservation.";
    const auto executable_publish_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publish_body));
    EXPECT_NE(executable_publish_body.find("plan.requiresCorrectionReplay()?"),
              std::string::npos)
        << "Spec-state publication diagnostics must distinguish accepted "
           "publication from rejected correction.";
    EXPECT_NE(executable_publish_body.find(
                  "handleLivePrefixReplayStateAfterMutation(mutation_reason,,false)"),
              std::string::npos)
        << "MTP accepted-state publication must keep a main/verifier replay-state "
           "mutation boundary.";

    const auto executable_mutation_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mutation_body));
    EXPECT_NE(executable_mutation_body.find(
                  "preservesMTPSidecarReplayAfterSpecPublication()"),
              std::string::npos)
        << "Spec-state publication must preserve sidecar replay only through "
           "the narrow replay-safety capability.";
    EXPECT_EQ(executable_mutation_body.find(
                  "if(state_.device_id.is_gpu()){forward_engine_->resetCapturedReplayState();"),
              std::string::npos)
        << "A GPU publication boundary must not blanket-reset byte-proven graph captures.";
    EXPECT_NE(executable_mutation_body.find(
                  "constboolpreserve_single_token_decode_replay=state_.device_id.is_gpu();"),
              std::string::npos)
        << "Only GPU domains with byte-proven stable device state may retain ordinary single-token replay.";
    EXPECT_NE(executable_mutation_body.find(
                  "resetCapturedReplayStateForCorrectionReplay(live_replay_state_epoch_,preserve_single_token_decode_replay)"),
              std::string::npos)
        << "Accepted publication must use the typed replay-lifetime policy on every backend.";
    EXPECT_NE(mutation_body.find("correction_replay_typed_live_state"),
              std::string::npos);
    EXPECT_NE(mutation_body.find("preserved_byte_proven_device_state"),
              std::string::npos)
        << "Perfstats must identify preservation backed by the persistent graph replay regressions.";
    EXPECT_EQ(mutation_body.find("correction_replay_all_gpu_segments"),
              std::string::npos)
        << "The obsolete blanket GPU reset policy must not return.";
    EXPECT_EQ(mutation_body.find("reset_until_device_state_indirection_proven"),
              std::string::npos)
        << "The device-state indirection debt was retired by byte-exact CUDA/ROCm lifetime tests.";
    EXPECT_NE(mutation_body.find("prefix_restore_discard_cached_graphs"),
              std::string::npos)
        << "Prefix restore is a full live-state replacement and must discard "
           "cached forward graph objects, not merely reset captured replay.";
    const auto executable_sidecar_safety_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_safety_body));
    EXPECT_NE(executable_sidecar_safety_body.find("if(!model_is_moe){"),
              std::string::npos)
        << "Dense MTP may keep the sidecar graph warm after spec publication "
           "because its shifted row reuse contract is proven.";
    EXPECT_NE(executable_sidecar_safety_body.find("constboolmodel_is_moe=config.isMoE()||isPrefixCacheMoEModel();"),
              std::string::npos)
        << "MoE detection must include Qwen3.6 fixtures whose graph config can "
           "arrive before every MoE field is populated.";
    EXPECT_NE(sidecar_safety_body.find("depth-scoped"),
              std::string::npos)
        << "MoE sidecar replay preservation must be justified by persistent "
           "sidecar-owned metadata, not by the broader main-state shortcut.";
    EXPECT_NE(executable_sidecar_safety_body.find("returntrue;"),
              std::string::npos)
        << "MoE MTP sidecar replay should stay warm once routed metadata is "
           "persistent and sidecar-owned.";
    EXPECT_NE(executable_mutation_body.find("resetMTPSidecarDepth0ReplayState()"),
              std::string::npos)
        << "Replay-unsafe sidecars must recapture instead of reusing stale "
           "captured metadata.";
    EXPECT_NE(mutation_body.find("reset_after_spec_publication"),
              std::string::npos)
        << "Perf stats must identify sidecar recapture caused by a spec "
           "publication boundary.";
    EXPECT_NE(mutation_body.find("sidecar_replay_reset"),
              std::string::npos)
        << "Replay-unsafe sidecar resets synchronize capture streams; Phase 10 "
           "tuning must keep that wall-clock cost visible.";
    EXPECT_NE(mutation_body.find("preserved_for_spec_publication"),
              std::string::npos)
        << "Perf stats must make the sidecar replay preservation explicit.";
    EXPECT_NE(executable_mutation_body.find(
                  "if(!preserve_gpu_replay_state&&preserves_correction_graph_replay)"),
              std::string::npos)
        << "Correction publication must not globally reset kernel dynamic state "
           "while verifier/sidecar graph executables are preserved.";
    EXPECT_NE(mutation_body.find("preserved_for_correction_graph_replay"),
              std::string::npos)
        << "Perf stats must make kernel dynamic-state preservation explicit.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoEMTPSidecarUsesPersistentDepthScopedMetadata)
{
    const auto graph_source =
        readFile(repoRoot() / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto dgo_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto ffn_body = sliceBetween(
        graph_source,
        "ComputeGraph Qwen35MoEGraph::buildFFNGraph(",
        "// =====================================================================\n"
        "        // Stage 1: Pre-FFN RMSNorm");
    const auto sidecar_replay_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::preservesMTPSidecarReplayAfterSpecPublication() const",
        "void DeviceGraphOrchestrator::recordShiftedMTPKVReplayStateMutation(");
    const auto spec_publication_support_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::supportsMTPSpecStatePublication() const",
        "MTPVerifierRowCapability DeviceGraphOrchestrator::mtpVerifierRowCapability() const");

    const auto compact_ffn =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(ffn_body));
    const auto compact_raw_ffn = removeAsciiWhitespace(ffn_body);
    const auto compact_replay =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_replay_body));
    const auto compact_spec_publication_support =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(spec_publication_support_body));

    EXPECT_NE(compact_ffn.find("constbooluse_mtp_runtime_table=mtp_sidecar_context;"),
              std::string::npos)
        << "Every MTP sidecar MoE fragment needs its own runtime table, even "
           "when the logical layer index aliases a main-model layer.";
    EXPECT_NE(compact_ffn.find("runtime_table_suffix=use_mtp_runtime_table?"),
              std::string::npos);
    EXPECT_NE(compact_raw_ffn.find("\"mtp_depth\"+std::to_string(mtp_depth_idx)"),
              std::string::npos)
        << "Sidecar metadata must be depth-scoped so captured graphs do not "
           "share main-decode top-k/runtime slots.";
    EXPECT_NE(compact_ffn.find("register_runtime_histogram=!use_mtp_runtime_table"),
              std::string::npos)
        << "Sidecar routing metadata is transient runtime state and must not "
           "feed request-level decode histograms.";
    EXPECT_NE(graph_source.find("moe_mtp_sidecar_runtime_table_creations"),
              std::string::npos)
        << "Real-model Phase 9.6 integration tests need a perf counter proving "
           "MTP sidecars use persistent MoE runtime tables.";
    EXPECT_NE(graph_source.find("moe_mtp_sidecar_runtime_table_reuses"),
              std::string::npos)
        << "Runtime-table reuse should remain observable when graph fragments "
           "stay warm across MTP sidecar calls.";
    EXPECT_NE(compact_replay.find("config.isMoE()||isPrefixCacheMoEModel()"),
              std::string::npos)
        << "Sidecar replay safety must use the same robust MoE detection as "
           "main-state preservation.";
    EXPECT_NE(compact_spec_publication_support.find("supportsMoEDirectAllPositionRows("),
              std::string::npos)
        << "MoE spec-state publication support must remain gated by the "
           "separate direct-row capability, not by the decode-equivalent fallback.";
    EXPECT_EQ(compact_spec_publication_support.find("state_.device_id.is_gpu()"),
              std::string::npos)
        << "Backend identity alone must not promote MoE direct all-position "
           "publication.";
    EXPECT_EQ(compact_spec_publication_support.find("supportsDeviceResidentLogicalSequenceStatePublication"),
              std::string::npos)
        << "The direct publication support query should not quietly compose a "
           "device-resident shortcut; the capability producer must earn and "
           "advertise that state explicitly.";
    EXPECT_NE(sidecar_replay_body.find("supportsMTPSidecarPreservesMainState()"),
              std::string::npos)
        << "The sidecar replay contract must remain explicitly narrower than "
           "main-state preservation and shifted-row reuse.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoERuntimeRefreshPublishesInitializedFlag)
{
    const auto stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp");
    const auto header_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h");

    const auto refresh_body = sliceBetween(
        stage_source,
        "bool MoEExpertComputeStage::refreshRuntimeGroupedDecodePlacement(bool preserve_capture_ready)",
        "bool MoEExpertComputeStage::refreshFixedTopologyGroupedPrefillPlacement()");
    const std::string compact_refresh =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(refresh_body));

    EXPECT_NE(compact_refresh.find("moe_runtime_table_initialized_=false;"),
              std::string::npos)
        << "Failed graph-stable refreshes must clear the stage-local readiness "
           "flag before the next warmup/replay observes stale placement state.";
    EXPECT_NE(compact_refresh.find("moe_runtime_table_initialized_=runtimeTableHasActiveGroupedDecodeBank();"),
              std::string::npos)
        << "A successful graph-stable placement refresh must publish the active "
           "runtime-bank check into the stage-local initialized flag.";
    EXPECT_NE(compact_refresh.find("&&moe_runtime_table_initialized_;"),
              std::string::npos)
        << "Capture-ready preservation must depend on the same initialized flag "
           "that decode execution will later read.";
    EXPECT_NE(compact_refresh.find("returnmoe_runtime_table_initialized_;"),
              std::string::npos)
        << "Explicit-owner LocalTP runtime tables must fail closed if the active "
           "bank disappears between graph build and replay.";
    expectNeedleBefore(
        compact_refresh,
        "moe_runtime_table_initialized_=runtimeTableHasActiveGroupedDecodeBank();",
        "runtime_grouped_decode_warmed_=preserve_capture_ready",
        "The initialized flag must be refreshed before capture-ready bookkeeping.");

    EXPECT_NE(header_source.find("@brief Revalidates the graph-owned one-token MoE runtime placement."),
              std::string::npos)
        << "This private refresh method owns a subtle graph/runtime coherence "
           "contract and needs durable Doxygen guidance.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoEVerifierRoutingTensorDecodeDoesNotRequireAllEnabledMask)
{
    const auto stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp");
    const auto imoe_source =
        readFile(repoRoot() / "src/v2/kernels/IMoEKernel.h");
    const auto cuda_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto rocm_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp");

    const auto verifier_replay_body = sliceBetween(
        stage_source,
        "struct ScopedSingleVerifierRow",
        "return executeSingleToken(ctx);");
    const auto routing_tensor_predicate = sliceBetween(
        stage_source,
        "const bool require_device_routing_tensor_decode =",
        "if (can_try_device_routing_tensor_decode)");
    const auto routing_tensor_body = sliceBetween(
        stage_source,
        "if (can_try_device_routing_tensor_decode)",
        "if (is_gpu && isGraphCaptureActive())");
    const auto compact_verifier_replay =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(verifier_replay_body));
    const auto compact_predicate =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(routing_tensor_predicate));
    const auto compact_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(routing_tensor_body));

    EXPECT_NE(compact_verifier_replay.find("params_.require_device_routing_tensor_decode=is_gpu;"),
              std::string::npos)
        << "GPU decode-equivalent verifier replay must use the explicit routing-tensor path.";
    EXPECT_NE(compact_predicate.find("hasFullLocalExpertOwnership();"),
              std::string::npos)
        << "The routing-tensor path still requires a complete local descriptor table.";
    EXPECT_EQ(compact_predicate.find("&&expertMaskAllEnabled()"),
              std::string::npos)
        << "Dynamic/LLEP masks can be participant scoped; all-enabled masks must "
           "not gate the decode-equivalent routing tensor path.";
    EXPECT_NE(compact_predicate.find("device_routing_expert_mask"),
              std::string::npos)
        << "Participant masks must be converted to byte masks before the device "
           "routing tensor path is considered usable.";
    EXPECT_NE(compact_body.find("required_expert_ids=device_routing_expert_mask_ptr?device_routing_required_expert_ids:all_expert_ids_;"),
              std::string::npos)
        << "Masked verifier routing must prepare only mask-active local experts "
           "instead of demanding a full replicated descriptor table.";
    EXPECT_NE(compact_body.find("ensureGemmEnginesForExperts(required_expert_ids)"),
              std::string::npos);
    EXPECT_NE(routing_tensor_body.find("device_routing_expert_mask_ptr"),
              std::string::npos)
        << "The device route calls must receive the participant mask so masked-off "
           "top-k slots become inactive on device.";
    EXPECT_NE(routing_tensor_predicate.find(
                  "masked-off top-k slots to -1"),
              std::string::npos)
        << "The non-obvious mask/ownership split needs inline guidance.";
    EXPECT_NE(imoe_source.find("const uint8_t *expert_mask = nullptr"),
              std::string::npos)
        << "The backend interface must expose an explicit optional local-compute "
           "mask for routing-tensor decode.";
    EXPECT_NE(cuda_source.find("cudaMoE_float_to_masked_int("),
              std::string::npos)
        << "CUDA routing-tensor decode must keep masked slots on device instead "
           "of round-tripping routing ids through the host.";
    EXPECT_NE(rocm_source.find("hipMoE_float_to_masked_int("),
              std::string::npos)
        << "ROCm routing-tensor decode must use the same masked device conversion.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GpuMoERebalanceProjectionRequiresPhysicalSourceResidency)
{
    const auto shared_policy_source =
        readFile(repoRoot() / "src/v2/execution/moe/DeviceMoERebalancePolicyShared.h");
    const auto domain_policy = sliceBetween(
        shared_policy_source,
        "LLAMINAR_MOE_REBALANCE_HD bool candidateCanAffectDomainCompute(",
        "} // namespace llaminar2::moe_rebalance_policy");
    const auto compact_domain_policy =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(domain_policy));

    EXPECT_EQ(compact_domain_policy.find("resident_mask|=participantBit"),
              std::string::npos)
        << "Domain-root planning must not promote descriptor ownership into "
           "physical source residency.";
    EXPECT_NE(compact_domain_policy.find(
                  "firstResidentParticipant(resident_mask,config.participant_count,desc.owner_participant,-1)"),
              std::string::npos)
        << "The owner may remain a source preference, but firstResidentParticipant "
           "must still require the bit in the physical resident mask.";

    const std::vector<std::pair<std::string, std::filesystem::path>> kernels = {
        {"CUDA", "src/v2/kernels/cuda/moe/CUDAMoEKernels.cu"},
        {"ROCm", "src/v2/kernels/rocm/moe/ROCmMoEKernels.hip"},
    };

    for (const auto &[backend, relative_path] : kernels)
    {
        const auto source = readFile(repoRoot() / relative_path);
        const auto compact =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(source));
        const auto root_projection = sliceBetween(
            source,
            "__global__ void project_rebalance_domain_commands_kernel(",
            "__global__ void project_prefill_llep_domain_commands_kernel(");
        const auto compact_root_projection =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(root_projection));
        const auto prefill_projection = sliceBetween(
            source,
            "__global__ void project_prefill_llep_domain_commands_kernel(",
            "__global__ void materialize_prefill_llep_transfer_commands_kernel(");
        const auto compact_prefill_projection =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(prefill_projection));
        const auto materialize_transfers = sliceBetween(
            source,
            "__global__ void materialize_prefill_llep_transfer_commands_kernel(",
            "__global__ void pack_rebalance_source_descriptors_kernel(");
        const auto compact_materialize =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(materialize_transfers));
        const auto source_entry_ready = sliceBetween(
            source,
            "__device__ __forceinline__ bool rebalance_source_entry_ready(",
            "__device__ __forceinline__ bool rebalance_transfer_slot_ready(");
        const auto compact_source_entry_ready =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(source_entry_ready));
        const auto copy_status = sliceBetween(
            source,
            "__device__ __forceinline__ bool rebalance_copy_status_ok(",
            "__device__ __forceinline__ uint32_t rebalance_expected_payload_arrivals(");
        const auto compact_copy_status =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(copy_status));

        EXPECT_EQ(compact.find("runtime_participant_bit(candidate_desc.owner_participant)"),
                  std::string::npos)
            << backend << " dynamic planning must not treat ownership as physical source residency.";
        EXPECT_EQ(compact.find("candidate_resident_mask|="),
                  std::string::npos)
            << backend << " dynamic planning must not inflate candidate residency masks.";
        EXPECT_EQ(compact.find("projected.source_resident_mask|="),
                  std::string::npos)
            << backend << " domain projection must not launder invalid root-plan source masks.";
        EXPECT_EQ(compact.find("shared_post_policy_resident_mask[transfer.expert]|source_bit"),
                  std::string::npos)
            << backend << " LLEP decode planning must not manufacture transfer-source residency.";
        EXPECT_NE(compact.find("shared_physical_source_resident_mask"),
                  std::string::npos)
            << backend << " must keep wave-start physical source residency separate "
               "from optimistic post-policy residency.";
        EXPECT_NE(compact.find("candidate_physical_source_mask"),
                  std::string::npos)
            << backend << " dynamic missing-resident scoring must pick copy sources "
               "from the wave-start physical mask.";
        EXPECT_NE(compact.find("heavy_physical_source_mask"),
                  std::string::npos)
            << backend << " ownership swaps must validate the heavy source against "
               "wave-start physical residency.";
        EXPECT_NE(compact.find("light_physical_source_mask"),
                  std::string::npos)
            << backend << " ownership swaps must validate the light source against "
               "wave-start physical residency.";
        EXPECT_EQ(compact.find("static_cast<int>(destination_choice.source_participant)"),
                  std::string::npos)
            << backend << " must not copy from a source participant chosen from "
               "the optimistic post-policy resident mask.";
        EXPECT_NE(compact.find("source_delta.meets_floor"),
                  std::string::npos)
            << backend << " dynamic candidate scoring must recompute economics for "
               "the physical source participant it will actually copy from.";
        EXPECT_GE(countOccurrences(compact, "source_is_physically_resident"),
                  2u)
            << backend << " dynamic root planning must check the selected copy source "
               "during both candidate selection and final command publication.";
        EXPECT_GE(countOccurrences(compact, "heavy_physical_source_mask"),
                  2u)
            << backend << " dynamic ownership swaps must validate the heavy expert's "
               "physical source residency in both controller paths.";
        EXPECT_GE(countOccurrences(compact, "light_physical_source_mask"),
                  2u)
            << backend << " dynamic ownership swaps must validate the light expert's "
               "physical source residency in both controller paths.";
        EXPECT_EQ(compact.find("source_resident_mask=runtime_participant_bit(static_cast<int>(heavy_source))"),
                  std::string::npos)
            << backend << " ownership-transfer commands must publish the actual "
               "resident mask, not a manufactured source bit.";
        EXPECT_EQ(compact.find("source_resident_mask=runtime_participant_bit(static_cast<int>(light_source))"),
                  std::string::npos)
            << backend << " ownership-transfer commands must publish the actual "
               "resident mask, not a manufactured source bit.";

        EXPECT_NE(compact_root_projection.find("(plan.source_resident_mask&source_participant_bit)!=0u"),
                  std::string::npos)
            << backend << " root-domain projection must reject source participants "
               "that are absent from the resident mask.";
        EXPECT_NE(root_projection.find("Compact root-domain transfer commands after checking physical source residency"),
                  std::string::npos)
            << backend << " needs an inline note for the source-residency projection contract.";
        EXPECT_NE(compact_root_projection.find("local_plan_entries[local_plan_index]=plan;"),
                  std::string::npos)
            << backend << " should compact only already-validated root commands.";
        EXPECT_EQ(compact_root_projection.find("entry=gathered_plan_entries"),
                  std::string::npos)
            << backend << " must not copy root commands verbatim before source-residency validation.";

        EXPECT_NE(compact_prefill_projection.find("(plan.source_resident_mask&source_participant_bit)!=0u"),
                  std::string::npos)
            << backend << " prefill-LLEP domain projection must reject non-resident sources.";
        EXPECT_NE(compact_materialize.find("(resident_mask&transfer_source_bit)!=0u"),
                  std::string::npos)
            << backend << " materialized prefill-LLEP transfers must require the "
               "selected source to be physically resident.";
        EXPECT_NE(compact_source_entry_ready.find("entry.slot_index!=kDeviceMoEInvalidSlot"),
                  std::string::npos)
            << backend << " source descriptors must require a participant-local "
               "slot before any payload bytes can be copied.";
        EXPECT_NE(compact_source_entry_ready.find("entry.descriptor.local_slot>=0"),
                  std::string::npos)
            << backend << " source descriptors must reject resident masks backed "
               "only by non-local descriptor metadata.";
        EXPECT_EQ(compact_materialize.find("resident_mask|=llaminar2::moe_rebalance_policy::participantBit(transfer_source)"),
                  std::string::npos)
            << backend << " materialization must not manufacture source residency.";
        EXPECT_NE(compact_copy_status.find("status.missing_source_descriptors==0u"),
                  std::string::npos)
            << backend << " transfer publication must reject source-side pack failures.";
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, Qwen35MoEDeviceRoutedDecodeTableGuardsOwnershipShape)
{
    const auto graph_source =
        readFile(repoRoot() / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto ffn_body = sliceBetween(
        graph_source,
        "ComputeGraph Qwen35MoEGraph::buildFFNGraph(",
        "// =====================================================================\n"
        "        // Stage 1: Pre-FFN RMSNorm");
    const std::string compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(ffn_body));

    EXPECT_NE(compact.find("has_static_full_local_expert_ownership"),
              std::string::npos)
        << "The one-token device-routed MoE runtime table is a full local "
           "expert ownership contract. LocalTP sharded experts must not receive "
           "that table until a sharded runtime table/reducer is implemented.";
    EXPECT_NE(compact.find("local_count!=config_.moe.num_experts"),
              std::string::npos)
        << "Partial ExpertParallel ranges must use the mask/range + allreduce "
           "path instead of the single-device full-owner runtime table.";
    EXPECT_NE(compact.find("debugEnv().moe_rebalance.gpu_cache_experts_per_layer>0"),
              std::string::npos)
        << "GPU expert-cache bootstrap masks also break full-owner runtime-table "
           "semantics and must keep decode capture disabled.";
    EXPECT_NE(compact.find("if(use_expert_overlay)returnfalse"),
              std::string::npos)
        << "Graph-native tiered overlays use per-participant expert masks even "
           "when the tier domain says compute=ApportionedExperts; they must not "
           "receive the full-owner decode runtime table.";
    EXPECT_NE(compact.find("constbooldecode_runtime_table_eligible="),
              std::string::npos)
        << "Runtime-table creation must use one explicit eligibility predicate "
           "before MoERoutingStage and MoEExpertComputeStage are built.";
    EXPECT_NE(compact.find("&&decode_runtime_table_eligible)"),
              std::string::npos)
        << "Runtime-table creation must be gated before MoERoutingStage and "
           "MoEExpertComputeStage are built; failing later in the expert stage "
           "turns E2E requests into parse errors instead of policy counters.";
    EXPECT_NE(compact.find("masked_local_tp_overlay_decode_runtime_table"),
              std::string::npos)
        << "Masked LocalTP overlays may use their own runtime table, but the "
           "masked path must stay separate from the full-owner table contract.";
    const std::string compact_graph =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(graph_source));
    EXPECT_NE(compact_graph.find("initializeMaskedLocalDecodeRuntimeTable("),
              std::string::npos)
        << "Masked LocalTP runtime tables must be initialized through the "
           "mask-aware helper, not the full local decode helper.";
    EXPECT_NE(compact_graph.find("route_params.allow_eager_gpu_single_row_route_for_partial_expert_owner=allow_eager_partial_owner_gpu_route"),
              std::string::npos)
        << "The temporary partial-owner LocalTP route path must stay explicit "
           "at the graph-builder boundary.";
    EXPECT_NE(compact_graph.find("config_.moe.expert_mode==MoEExpertMode::ApportionedExperts||use_expert_overlay"),
              std::string::npos)
        << "Tiered overlays must opt into the explicit eager partial-owner "
           "route path until a sharded runtime-table reducer exists.";

    const auto routing_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoERoutingStage.cpp");
    const std::string compact_routing =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(routing_source));
    EXPECT_NE(compact_routing.find("params_.allow_eager_gpu_single_row_route_for_partial_expert_owner"),
              std::string::npos);
    EXPECT_NE(compact_routing.find("if(isGraphCaptureActive())"),
              std::string::npos)
        << "Partial-owner routeWithTensors is a correctness bridge for eager "
           "TP execution only; graph capture must remain fail-closed.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, ProductionGpuMoERoutingRejectsHostTopKFallback)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoERoutingStage.cpp");
    const auto executable_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(source));

    EXPECT_NE(source.find("GPU single-row MoE routing requires"),
              std::string::npos)
        << "Production GPU decode must fail closed if runtime-table routing is "
           "unavailable, instead of silently materializing top-k rows on host.";
    EXPECT_NE(source.find("#ifdef ENABLE_PIPELINE_SNAPSHOTS"),
              std::string::npos)
        << "Host routing mirrors must remain snapshot-only diagnostics.";
    EXPECT_NE(executable_source.find(
                  "params_.device_id.is_gpu()&&params_.seq_len==1&&!isDeviceRoutedDecodeGraphCapturable()"),
              std::string::npos)
        << "The guard must cover every non-snapshot GPU single-row MoE route.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDARingKVCacheGroupedGatherHasNoRawAllocationFallback)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto gather_body = sliceBetween(
        source,
        "int CUDARingKVCache<Precision>::gather_kv_batched(",
        "bool CUDARingKVCache<Precision>::get_kv_batched_device_view(");

    expectNoRawGpuAllocationCalls(gather_body, "CUDARingKVCache::gather_kv_batched");
    const auto compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(gather_body));
    EXPECT_NE(compact.find("!stream"), std::string::npos)
        << "Grouped gather must reject the device-default stream.";
    EXPECT_NE(compact.find("d_head_params_"), std::string::npos);
    EXPECT_NE(compact.find("d_count_params_"), std::string::npos)
        << "Grouped gather must consume canonical device ring state.";
    EXPECT_EQ(compact.find("DeviceToHost"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, GPURequestBatchedKVViewsRemainDeviceOwned)
{
    const auto root = repoRoot();
    const auto cuda_source = readFile(
        root / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto rocm_source = readFile(
        root / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp");
    const auto rocm_kernels = readFile(
        root / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheKernels.hip");
    const auto cuda_hybrid = readFile(
        root / "src/v2/kernels/cuda/kvcache/CUDAHybridRingKVCache.h");
    const auto rocm_hybrid = readFile(
        root / "src/v2/kernels/rocm/kvcache/ROCmHybridRingKVCache.h");

    const auto cuda_view = sliceBetween(
        cuda_source,
        "bool CUDARingKVCache<Precision>::get_kv_batched_device_view(",
        "bool CUDARingKVCache<Precision>::get_kv_batched_converted_device_view(");
    const auto rocm_view = sliceBetween(
        rocm_source,
        "bool ROCmRingKVCache<Precision>::get_kv_batched_device_view(",
        "bool ROCmRingKVCache<Precision>::get_kv_batched_converted_device_view(");

    expectNoRawGpuAllocationCalls(cuda_view, "CUDA native request-batched KV view");
    expectNoRawGpuAllocationCalls(rocm_view, "ROCm native request-batched KV view");
    for (const auto &view : {cuda_view, rocm_view})
    {
        const auto compact =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(view));
        EXPECT_NE(compact.find("!gpu_stream"), std::string::npos)
            << "Native grouped KV reads must reject the device-default stream.";
        EXPECT_NE(compact.find("d_head_params_"), std::string::npos);
        EXPECT_NE(compact.find("d_count_params_"), std::string::npos)
            << "Grouped attention must read graph-ordered cache counts on device.";
        EXPECT_NE(compact.find("ensureConvScratch(required_bytes)"),
                  std::string::npos)
            << "The contiguous grouped view must use planner-owned cache scratch.";
        EXPECT_EQ(compact.find("synchronize"), std::string::npos);
        EXPECT_EQ(compact.find("DeviceToHost"), std::string::npos);
    }

    const auto compact_rocm_kernels =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_kernels));
    EXPECT_NE(compact_rocm_kernels.find(
                  "ring_gather_batched_device_state_kernel"),
              std::string::npos);
    EXPECT_NE(compact_rocm_kernels.find("d_heads[entry]"),
              std::string::npos);
    EXPECT_NE(compact_rocm_kernels.find("d_counts[entry]"),
              std::string::npos);

    for (const auto &hybrid : {cuda_hybrid, rocm_hybrid})
    {
        const auto compact =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(hybrid));
        EXPECT_NE(compact.find("get_kv_batched_device_view("),
                  std::string::npos);
        EXPECT_NE(compact.find("layer_map_.toKVIndex(normalizeLayerIndex(layer))"),
                  std::string::npos)
            << "Hybrid model-layer reads must remap to the compressed FA table.";
        EXPECT_NE(compact.find("Base::get_kv_batched_device_view("),
                  std::string::npos);
    }
}

/**
 * @brief Prevent GPU hybrid recurrent-state host mirrors from returning.
 *
 * CUDA and ROCm cache resources may retain logical shape metadata and kernel
 * objects, but only CPU caches allocate the FP32 recurrence/conv vectors. MTP
 * publication must likewise restore kernel-owned banks without a D2H sidecar.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, GPUHybridRecurrentStateHasNoHostMirror)
{
    const auto root = repoRoot();
    const std::vector<std::filesystem::path> hybrid_headers = {
        root / "src/v2/kernels/cuda/kvcache/CUDAHybridRingKVCache.h",
        root / "src/v2/kernels/rocm/kvcache/ROCmHybridRingKVCache.h",
    };
    for (const auto &path : hybrid_headers)
    {
        const auto source = readFile(path);
        const auto compact =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(source));
        EXPECT_NE(compact.find("state.initializeShape(qkv_dim)"), std::string::npos)
            << path;
        EXPECT_EQ(compact.find("state.initializeCPUState(qkv_dim)"), std::string::npos)
            << path;
        EXPECT_EQ(compact.find("metadata.host_bytes=gdnMemoryBytes()"), std::string::npos)
            << path;
        EXPECT_EQ(compact.find("state.reset();"), std::string::npos)
            << path;
    }

    const std::vector<std::filesystem::path> kernel_headers = {
        root / "src/v2/kernels/cuda/gdn/CUDAShortConvolution.h",
        root / "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNet.h",
        root / "src/v2/kernels/rocm/gdn/ROCmShortConvolution.h",
        root / "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h",
    };
    for (const auto &path : kernel_headers)
    {
        const auto source = readFile(path);
        const auto restore = sliceBetween(
            source,
            "bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream) override",
            "bool restoreVerifierStateCaptureRowFromDeviceIndex(");
        const auto compact =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(restore));
        EXPECT_NE(compact.find("(void)dst_state"), std::string::npos) << path;
        EXPECT_EQ(compact.find("memcpy_d2h"), std::string::npos) << path;
        EXPECT_EQ(compact.find("stream_synchronize"), std::string::npos) << path;
    }

    const auto orchestrator = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    EXPECT_EQ(orchestrator.find("denseLocalTPHybridStateRequiresHostOnlyPayload"),
              std::string::npos);
    EXPECT_EQ(orchestrator.find("applyDenseLocalTPHostOnlyHybridPayloadLayout"),
              std::string::npos);
    EXPECT_EQ(orchestrator.find("import_host_state_into_device_state"),
              std::string::npos);
    EXPECT_EQ(orchestrator.find("import_device_state_from_host_state"),
              std::string::npos);
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

TEST(Test__GpuWorkspaceAllocationPolicy, CUDAMoERuntimePointerArraysUseWorkspace)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto runtime_pointer_arrays = sliceBetween(
        source,
        "bool CUDAMoEKernel::ensureRuntimeGateUpPointerArrays(",
        "bool CUDAMoEKernel::routeCore(");

    expectNoRawGpuAllocationCalls(
        runtime_pointer_arrays,
        "CUDAMoEKernel grouped decode runtime pointer arrays");
    EXPECT_NE(runtime_pointer_arrays.find("CUDA_DECODE_GATEUP_GATE_PTRS"), std::string::npos);
    EXPECT_NE(runtime_pointer_arrays.find("CUDA_DECODE_GATEUP_UP_PTRS"), std::string::npos);
    EXPECT_NE(runtime_pointer_arrays.find("CUDA_DECODE_DOWN_GATE_PTRS"), std::string::npos);
    EXPECT_NE(runtime_pointer_arrays.find("CUDA_DECODE_DOWN_UP_PTRS"), std::string::npos);
    EXPECT_NE(runtime_pointer_arrays.find("runtimePointerWorkspaceSlot("),
              std::string::npos);
    EXPECT_NE(runtime_pointer_arrays.find("isCudaMoEDecodeCaptureActive(stream)"),
              std::string::npos);
    EXPECT_NE(runtime_pointer_arrays.find("was not staged before graph capture"),
              std::string::npos)
        << "Runtime pointer arrays must be uploaded before capture; direct "
           "stream capture must never record H2D copies from stack pointer arrays.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDAMoEWorkspaceRebindPreservesCaptureMetadata)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto header = readFile(repoRoot() / "src/v2/kernels/cuda/moe/CUDAMoEKernel.h");
    const auto bind_workspace = sliceBetween(
        source,
        "void CUDAMoEKernel::bindWorkspace(DeviceWorkspaceManager *workspace)",
        "bool CUDAMoEKernel::bindWorkspaceBuffer(");

    EXPECT_NE(header.find("uint64_t bound_workspace_id_"), std::string::npos)
        << "The CUDA MoE singleton must remember durable workspace identity, "
           "not just the manager host pointer.";
    EXPECT_NE(bind_workspace.find("workspace->id()"), std::string::npos)
        << "Same-pointer workspace reuse can be an ABA reallocation. "
           "Use DeviceWorkspaceManager::id() to distinguish real same-manager "
           "rebinds from new managers at the same host address.";
    EXPECT_NE(bind_workspace.find("workspace_ == workspace"), std::string::npos)
        << "Same-workspace rebinds happen as sibling MoE stages touch the singleton kernel. "
           "They must not clear runtime pointer arrays populated by graph warmup.";
    EXPECT_NE(bind_workspace.find("bound_workspace_id_ == next_workspace_id"), std::string::npos)
        << "The same-workspace no-op must also compare the manager's durable id.";
    EXPECT_NE(bind_workspace.find("bound_workspace_id_ = next_workspace_id"), std::string::npos)
        << "CUDA MoE workspace binding must record the durable manager id.";
    EXPECT_LT(bind_workspace.find("workspace_ == workspace"),
              bind_workspace.find("clearWorkspaceScratchBindings()"))
        << "The idempotent rebind guard must run before clearing graph-capture metadata.";
    EXPECT_LT(bind_workspace.find("bound_workspace_id_ = next_workspace_id"),
              bind_workspace.find("clearWorkspaceScratchBindings()"))
        << "The durable id must be updated before clearing stale scratch bindings.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDAMoERouteScratchReuseRequiresWorkspaceBinding)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto route_capacity = sliceBetween(
        source,
        "bool CUDAMoEKernel::ensureRouteBufferCapacity(",
        "bool CUDAMoEKernel::ensureGroupingBufferCapacity(");
    const auto executable_route_capacity =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(route_capacity));

    EXPECT_NE(executable_route_capacity.find("if(route_buffers_workspace_bound_&&"),
              std::string::npos)
        << "CUDA MoE routing scratch is owned by the graph workspace. Capacity "
           "alone must not make a cached route buffer reusable across singleton "
           "kernel rebinds.";
    EXPECT_NE(executable_route_capacity.find("d_route_logits_&&d_route_indices_&&d_route_weights_"),
              std::string::npos)
        << "Route-buffer reuse must also require non-null workspace pointers.";
    EXPECT_LT(executable_route_capacity.find("if(route_buffers_workspace_bound_&&"),
              executable_route_capacity.find("bindWorkspaceBuffer(&route_logits"))
        << "The binding-aware reuse guard must be checked before rebinding route scratch.";

    const auto bind_workspace_buffer = sliceBetween(
        source,
        "bool CUDAMoEKernel::bindWorkspaceBuffer(",
        "void CUDAMoEKernel::clearWorkspaceScratchBindings()");
    EXPECT_NE(bind_workspace_buffer.find("workspace_->device() != expected"), std::string::npos)
        << "CUDA MoE scratch binding must reject workspaces for any other device.";
    EXPECT_NE(bind_workspace_buffer.find("requireCudaDevicePointer(buffer"), std::string::npos)
        << "Workspace scratch must be validated as a CUDA device pointer at bind time.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, ROCmMoEDecodeRouteSelectStaysDeviceResident)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp");
    const auto decode_route_select = sliceBetween(
        source,
        "bool ROCmMoEKernel::decodeRouteSelect(",
        "void ROCmMoEKernel::zeroBuffer(");
    const auto executable_decode_route_select =
        stripCommentsAndStringLiterals(decode_route_select);

    EXPECT_EQ(executable_decode_route_select.find("hipMemcpyDeviceToHost"), std::string::npos)
        << "ROCm MoE decode routing is replayed inside MTP graphs. Runtime-table "
           "validation must happen before capture; the hot decode route path must "
           "not synchronize by copying metadata back to the host.";
    EXPECT_EQ(executable_decode_route_select.find("hipStreamSynchronize"), std::string::npos)
        << "ROCm MoE decode routing must remain graph-capturable and device-resident.";
    EXPECT_NE(decode_route_select.find("must stay entirely"), std::string::npos)
        << "Keep an inline note explaining why host validation is intentionally absent.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPVerifierGraphWaitsOnSidecarStreamWithoutHostFlush)
{
    const auto source = readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header = readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto wait_helper = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLogitsStream(",
        "void *DeviceGraphOrchestrator::peekPendingLogitsStream(");
    const auto verifier_metadata = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(",
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(");
    const auto executable_wait_helper = stripCommentsAndStringLiterals(wait_helper);
    const auto executable_verifier_metadata =
        stripCommentsAndStringLiterals(verifier_metadata);
    const auto compact_verifier_metadata =
        removeAsciiWhitespace(executable_verifier_metadata);

    EXPECT_NE(header.find("waitForPendingLogitsStream("), std::string::npos)
        << "The sidecar-to-verifier ordering edge should stay behind a named helper.";
    EXPECT_NE(wait_helper.find("insertStreamDependency(consumer_stream, producer_stream)"),
              std::string::npos)
        << "Verifier ordering must be a GPU-side event dependency, not a host sync.";
    EXPECT_EQ(executable_wait_helper.find("synchronizeStream"), std::string::npos)
        << "The pending-logits stream wait helper must not block the CPU.";
    EXPECT_NE(executable_verifier_metadata.find("PendingLogitsStreamRole::MTPSidecar"),
              std::string::npos)
        << "All-position verifier metadata preparation must consume any pending sidecar stream.";
    EXPECT_NE(executable_verifier_metadata.find("waitForPendingShiftedMTPKVReady"),
              std::string::npos)
        << "All-position verifier metadata preparation must consume deferred KV-only sidecar appends.";
    EXPECT_NE(compact_verifier_metadata.find("deviceSequenceCachedTokenCountPtr"),
              std::string::npos)
        << "All-position verifier metadata preparation must snapshot the "
           "device-owned pre-verifier cache length.";
    EXPECT_NE(compact_verifier_metadata.find("deviceCopyAsync(ptrs.base_cached_tokens"),
              std::string::npos)
        << "The pre-verifier base cache count must be staged device-to-device, "
           "not uploaded from the host after verifier replay.";
    EXPECT_NE(compact_verifier_metadata.find("mtp_publication_base_cache_snapshot_ready_=true"),
              std::string::npos)
        << "Publication should only trust BASE_CACHED_TOKENS after the verifier "
           "prep step marks the device snapshot ready.";
    EXPECT_EQ(compact_verifier_metadata.find("hostToDeviceOnStream(ptrs.base_cached_tokens"),
              std::string::npos)
        << "Verifier metadata preparation must not upload host base cache counts.";
    expectNeedleBefore(
        executable_verifier_metadata,
        "waitForPendingShiftedMTPKVReady",
        "uploadMTPSpecDecodeVerifierInputPlan",
        "Verifier graph metadata staging must be ordered after shifted MTP KV catch-up.");
    expectNeedleBefore(
        compact_verifier_metadata,
        "uploadMTPSpecDecodeVerifierInputPlan",
        "deviceCopyAsync(ptrs.base_cached_tokens",
        "The base-cache snapshot should reuse the verifier metadata workspace "
        "after row metadata has ensured it is bound.");
    EXPECT_NE(verifier_metadata.find("all_position_verifier_graph"),
              std::string::npos)
        << "The perf/debug label should make sidecar-to-verifier waits visible.";
    EXPECT_NE(verifier_metadata.find("all_position_verifier_shifted_mtp_kv"),
              std::string::npos)
        << "The perf/debug label should make shifted-KV-to-verifier waits visible.";
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
    const auto kernel_header =
        readFile(repoRoot() / "src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.h");
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

    const auto attention_wrapper_source = kernel_header + kernel_source;
    EXPECT_EQ(attention_wrapper_source.find("cudaMallocHost("), std::string::npos)
        << "Attention param staging is request-local metadata; it must use fixed "
           "host storage and upload into the workspace-owned DEVICE_PARAMS buffer.";
    EXPECT_EQ(attention_wrapper_source.find("cudaFreeHost("), std::string::npos)
        << "Attention param staging should not own pinned host allocations.";
    EXPECT_NE(kernel_header.find("std::array<attention::AttentionDeviceParams"), std::string::npos)
        << "The small-M attention param rows should stay in bounded member storage.";
    EXPECT_NE(kernel_source.find("getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS)"), std::string::npos)
        << "The CUDA graph-visible attention params must remain workspace-backed.";

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

TEST(Test__GpuWorkspaceAllocationPolicy, RoPECanConsumeDeviceResidentPositionIdsWithoutH2D)
{
    const auto graph_input =
        readFile(repoRoot() / "src/v2/execution/local_execution/graph/IGraphBuilder.h");
    const auto graph_types =
        readFile(repoRoot() / "src/v2/models/GraphTypes.h");
    const auto tensor_interface =
        readFile(repoRoot() / "src/v2/tensors/TensorKernels.h");
    const auto rope_stage_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/RoPEStage.h");
    const auto rope_stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/RoPEStage.cpp");
    const auto qwen_base =
        readFile(repoRoot() / "src/v2/models/qwen/QwenGraphBase.cpp") +
        readFile(repoRoot() / "src/v2/models/qwen/QwenGraphBase.h");
    const auto cuda_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/ops/CUDARoPEKernelT.h") +
        readFile(repoRoot() / "src/v2/kernels/cuda/ops/CUDAOpsKernels.cpp");
    const auto rocm_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/ops/ROCmRoPEKernelT.h") +
        readFile(repoRoot() / "src/v2/kernels/rocm/ops/ROCmRoPEKernelT.cpp");

    const auto compact_graph_input =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(graph_input));
    const auto compact_graph_types =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(graph_types));
    const auto compact_tensor_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(tensor_interface));
    const auto compact_stage_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rope_stage_header));
    const auto compact_stage_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rope_stage_source));
    const auto compact_qwen_base =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(qwen_base));

    EXPECT_NE(compact_graph_input.find("constvoid*position_ids_device=nullptr"),
              std::string::npos)
        << "ForwardInput must carry device-resident position rows beside host positions.";
    EXPECT_NE(compact_graph_types.find("constvoid*position_ids_device=nullptr"),
              std::string::npos)
        << "MTPForwardInput must carry the same resident position contract.";
    EXPECT_NE(compact_tensor_interface.find("setDynamicDevicePositionIds(constvoid*position_ids_device,intseq_len)"),
              std::string::npos);
    EXPECT_NE(compact_stage_header.find("constvoid*position_ids_device=nullptr"),
              std::string::npos);
    EXPECT_NE(compact_stage_header.find("updateDynamicDevicePositionIds("),
              std::string::npos);
    EXPECT_NE(compact_stage_source.find("kernel->setDynamicDevicePositionIds(params_.position_ids_device,seq_len)"),
              std::string::npos)
        << "RoPEStage execution must bind resident position rows directly.";
    EXPECT_NE(compact_stage_source.find("kernel->setDynamicDevicePositionIds(params_.position_ids_device,params_.seq_len)"),
              std::string::npos)
        << "Graph launch preparation must refresh resident position-row pointers before capture/replay.";
    EXPECT_NE(compact_qwen_base.find(".position_ids_device=position_ids_device"),
              std::string::npos)
        << "Qwen graph helpers must thread device positions into RoPEStage params.";
    EXPECT_NE(compact_qwen_base.find("qwen_input.position_ids_device=input.position_ids_device"),
              std::string::npos);

    const auto assert_backend_contract =
        [&](const std::string &source,
            const std::string &setter_marker,
            const std::string &setter_end_marker)
    {
        const auto compact =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(source));
        EXPECT_NE(compact.find("dynamic_position_ids_device_ptr_"),
                  std::string::npos);
        EXPECT_NE(compact.find("setDynamicDevicePositionIds(constvoid*position_ids_device,intseq_len)"),
                  std::string::npos);
        EXPECT_NE(source.find("Cannot bind device position_ids on a null/default"),
                  std::string::npos)
            << "Resident RoPE positions must not be accepted on the implicit default stream.";
        EXPECT_NE(compact.find("constboolhas_device_position_ids=dynamic_position_ids_device_ptr_!=nullptr"),
                  std::string::npos);
        EXPECT_NE(compact.find("constint*d_position_ids=dynamic_position_ids_device_ptr_"),
                  std::string::npos)
            << "Backend kernels must prefer the caller-owned device pointer over a workspace H2D buffer.";
        EXPECT_NE(compact.find("dynamic_position_ids_device_ptr_=nullptr;upload"),
                  std::string::npos)
            << "Host-position uploads must clear any previously bound resident pointer.";

        const auto setter = sliceBetween(
            source,
            setter_marker,
            setter_end_marker);
        EXPECT_EQ(setter.find("uploadCudaRoPEPositionIds("), std::string::npos);
        EXPECT_EQ(setter.find("uploadHIPRoPEPositionIds("), std::string::npos);
        EXPECT_EQ(setter.find("Memcpy"), std::string::npos)
            << "Resident position binding must not contain a hidden H2D copy.";
    };

    assert_backend_contract(
        cuda_source,
        "CUDARoPEKernelT<ActivationPrecision::FP32>::setDynamicDevicePositionIds",
        "bool CUDARoPEKernelT<ActivationPrecision::FP32>::apply_typed");
    assert_backend_contract(
        rocm_source,
        "ROCmRoPEKernelT<ActivationPrecision::FP32>::setDynamicDevicePositionIds",
        "void ROCmRoPEKernelT<ActivationPrecision::FP32>::bindWorkspace");
}

TEST(Test__GpuWorkspaceAllocationPolicy, DeviceResidentPositionIdsPropagateThroughGraphSessions)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto compact_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(source));

    EXPECT_NE(compact_header.find("GraphBuildSession&withDevicePositionIds(constvoid*position_ids_device)"),
              std::string::npos)
        << "Forward graph sessions need a first-class resident-position override.";
    EXPECT_NE(compact_source.find("GraphBuildSession::withDevicePositionIds(constvoid*position_ids_device)"),
              std::string::npos);
    EXPECT_NE(compact_source.find("explicit_position_ids_device_=position_ids_device"),
              std::string::npos);
    EXPECT_NE(compact_source.find("prepared.position_ids_device=explicit_position_ids_device_"),
              std::string::npos)
        << "Session preparation must carry the device pointer into ForwardInput.";

    EXPECT_NE(compact_header.find("AttentionGraphSession&withDevicePositionIds(constvoid*position_ids_device)"),
              std::string::npos)
        << "Layer-level attention graph sessions need the same resident-position contract.";
    EXPECT_NE(compact_source.find("AttentionGraphSession::withDevicePositionIds(constvoid*position_ids_device)"),
              std::string::npos);
    EXPECT_NE(compact_source.find("position_ids_device_=position_ids_device"),
              std::string::npos);
    EXPECT_NE(compact_source.find("kv_cache_,position_ids_,device_.value(),sequence_lengths_,position_ids_device_)"),
              std::string::npos)
        << "Attention graph builders must receive host and device position inputs separately.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CachedForwardReplayRefreshesResidentPositionRows)
{
    const auto stage_interface =
        readFile(repoRoot() / "src/v2/execution/compute_stages/IComputeStage.h");
    const auto rope_stage =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/RoPEStage.h");
    const auto forward_types =
        readFile(repoRoot() / "src/v2/execution/local_execution/engine/ForwardGraphTypes.h");
    const auto forward_engine =
        readFile(repoRoot() / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp");

    const auto compact_stage_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(stage_interface));
    const auto compact_rope_stage =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rope_stage));
    const auto compact_forward_types =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(forward_types));
    const auto compact_forward_engine =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(forward_engine));

    EXPECT_NE(compact_stage_interface.find("virtualvoidupdateDynamicPositionIds(constint*position_ids,intseq_len)"),
              std::string::npos);
    EXPECT_NE(compact_stage_interface.find("virtualvoidupdateDynamicDevicePositionIds(constvoid*position_ids_device,intseq_len)"),
              std::string::npos);
    EXPECT_NE(compact_rope_stage.find("updateDynamicPositionIds(constint*position_ids,intseq_len)override"),
              std::string::npos);
    EXPECT_NE(compact_rope_stage.find("updateDynamicDevicePositionIds(constvoid*position_ids_device,intseq_len)override"),
              std::string::npos);

    EXPECT_NE(compact_forward_types.find("booluses_device_position_ids=false"),
              std::string::npos)
        << "Forward graph signatures must distinguish resident-position graphs.";
    EXPECT_NE(compact_forward_types.find("uses_device_position_ids==other.uses_device_position_ids"),
              std::string::npos);
    EXPECT_NE(compact_forward_types.find("std::hash<bool>{}(sig.uses_device_position_ids)"),
              std::string::npos);
    EXPECT_NE(compact_forward_engine.find("uses_device_position_ids"),
              std::string::npos)
        << "Perf tags should expose whether a cached graph used resident positions.";

    EXPECT_NE(compact_forward_types.find("booluses_device_sequence_lengths=false"),
              std::string::npos)
        << "Forward graph signatures must distinguish request-length-owned row-selection graphs.";
    EXPECT_NE(compact_forward_types.find("uses_device_sequence_lengths==other.uses_device_sequence_lengths"),
              std::string::npos);
    EXPECT_NE(compact_forward_types.find("std::hash<bool>{}(sig.uses_device_sequence_lengths)"),
              std::string::npos);
    EXPECT_NE(compact_forward_engine.find("input.sequence_lengths_device!=nullptr"),
              std::string::npos)
        << "The cache signature and perf tags must record resident request-length ownership.";

    EXPECT_NE(compact_forward_engine.find("constboolhas_position_input=input.position_ids!=nullptr||(input.position_ids_device!=nullptr&&input.device.is_gpu())"),
              std::string::npos)
        << "GPU resident position rows must count as stable graph inputs.";
    EXPECT_NE(compact_forward_engine.find("input.position_ids_device!=nullptr"),
              std::string::npos)
        << "The cache signature should record resident-position mode.";
    EXPECT_NE(compact_forward_types.find("selectForwardReplayHostPositionIds"),
              std::string::npos)
        << "Cache-hit replay must use the current invocation's position rows before cache storage.";
    EXPECT_NE(compact_forward_types.find("forwardPositionRowCount"),
              std::string::npos)
        << "Explicit position rows need one overflow-checked flattened geometry helper.";
    EXPECT_NE(compact_forward_engine.find("stage->updateDynamicDevicePositionIds(input.position_ids_device,position_row_count)"),
              std::string::npos)
        << "Cache-hit replay must refresh every resident RoPE row before capture/replay.";
    EXPECT_NE(compact_forward_engine.find("stage->updateDynamicPositionIds(replay_position_ids,position_row_count)"),
              std::string::npos)
        << "Cache-hit replay must keep every host explicit position row fresh too.";
    EXPECT_NE(compact_forward_engine.find("stage->updateDynamicDevicePositionIds(effective_input.position_ids_device,position_row_count)"),
              std::string::npos)
        << "GPU cache-miss capture must refresh all resident RoPE rows after stream binding.";
    EXPECT_NE(compact_forward_engine.find("stage->updateDynamicPositionIds(effective_input.position_ids,position_row_count)"),
              std::string::npos)
        << "GPU cache-miss capture must refresh all host explicit RoPE rows after stream binding.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, Qwen35MoEMaintenanceGraphUsesDedicatedCollectiveLane)
{
    const auto header_source =
        readFile(repoRoot() / "src/v2/models/qwen35moe/Qwen35MoEGraph.h");
    const auto graph_source =
        readFile(repoRoot() / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto binding_section = sliceBetween(
        header_source,
        "struct GraphSideRebalanceBinding",
        "IMoERuntimeTable *moeRuntimeTableForDevice");
    const auto maintenance_build = sliceBetween(
        graph_source,
        "ComputeGraph Qwen35MoEGraph::buildDeviceMoERebalanceMaintenanceGraph(",
        "void Qwen35MoEGraph::appendPrefixCacheFingerprintMaterial");
    const auto insertion_section = sliceBetween(
        graph_source,
        "if (env.moe_rebalance.device_rebalance_maintenance_graph)",
        "moe_graph_rebalance_bindings_[domain_key]");

    EXPECT_NE(binding_section.find("decode_tp_ctx"), std::string::npos);
    EXPECT_NE(binding_section.find("maintenance_tp_ctx"), std::string::npos);
    EXPECT_NE(graph_source.find("maintenanceTPContextForDomain("), std::string::npos)
        << "Maintenance graph mode must own a second LocalTP context/communicator.";
    EXPECT_NE(graph_source.find("std::weak_ptr<ILocalTPContext>"), std::string::npos)
        << "Separate graph builders for different devices must share the same "
           "domain-wide maintenance context instead of creating one communicator per participant.";
    EXPECT_NE(graph_source.find("Reusing dedicated MoE rebalance maintenance collective lane"),
              std::string::npos);
    EXPECT_NE(graph_source.find("key << \":devices=\""), std::string::npos)
        << "The maintenance collective lane key must include concrete participants, "
           "not just backend/degree.";
    EXPECT_NE(insertion_section.find("maintenance_lane_key"),
              std::string::npos);
    EXPECT_EQ(insertion_section.find("maintenanceTPContextForDomain(domain_key, *local_tp_ctx)"),
              std::string::npos)
        << "The maintenance collective lane must be domain-wide, not participant-specific.";
    EXPECT_NE(maintenance_build.find("params.tp_ctx = binding.maintenance_tp_ctx"),
              std::string::npos)
        << "Standalone maintenance collectives must not ride the decode communicator.";
    EXPECT_EQ(maintenance_build.find("params.tp_ctx = binding.decode_tp_ctx"),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, Qwen35MoEMultiRowVerifierKeepsStrictPublicationGuards)
{
    const auto graph_source = readFile(repoRoot() / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto predicate_section = sliceBetween(
        graph_source,
        "auto forceDecodeEquivalentMoEVerifier = [&](DeviceId candidate)",
        "LayerWeightBindings layer_bindings");
    const std::string compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(predicate_section));

    EXPECT_NE(compact.find("returncandidate.is_cpu()&&"),
              std::string::npos)
        << "The decode-equivalent verifier lane is CPU-only now. GPU verifier "
           "rows must stay on the economical grouped publication routes instead "
           "of drifting back toward row replay.";
    EXPECT_NE(compact.find("total_tokens>=1&&total_tokens<=4"),
              std::string::npos)
        << "M=1 verifier publication uses the explicit decode-equivalent path, "
           "while M=2..4 can use the grouped verifier route.";

    const auto combined_shared_section = sliceBetween(
        graph_source,
        "const bool can_combine_shared_verifier =",
        "if (overlay_requested && !use_expert_overlay)");
    const std::string compact_combined_shared =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(combined_shared_section));
    EXPECT_NE(compact_combined_shared.find("false"),
              std::string::npos)
        << "The combined routed+shared verifier owner must remain disabled in "
           "production until full-model cosine/L2/KL/max-abs gates prove it. "
           "The accepted route is split routed grouped verifier plus standalone "
           "shared-expert GEMV-many.";

    const auto expert_stage_section = sliceBetween(
        graph_source,
        "auto expert_params = makeExpertParams(moe_output",
        "if (!prepareExpertParams(expert_params, device))");
    const std::string compact_expert_stage =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(expert_stage_section));
    EXPECT_EQ(compact_expert_stage.find("expert_params.combine_shared_expert_in_verifier=true;"),
              std::string::npos)
        << "Do not wire the unaccepted combined routed+shared verifier owner. "
           "It previously passed component microbenches while failing strict "
           "full-model continuation parity.";

    const auto shared_stage_section = sliceBetween(
        graph_source,
        "SharedExpertFFNStage::Params shared_params;",
        "graph.addNode(prefix + \"shared_expert_ffn\"");
    const auto shared_policy_section = sliceBetween(
        graph_source,
        "auto forceGroupedSharedMoEVerifierPrefill = [&](DeviceId candidate)",
        "/*\n         * MTP sidecars need their own persistent MoE metadata");
    const std::string compact_shared =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(shared_stage_section));
    const std::string compact_shared_policy =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(shared_policy_section));
    EXPECT_FALSE(compact_shared_policy.empty())
        << "Could not find the shared-expert verifier grouping policy.";
    EXPECT_EQ(compact_shared_policy.find("gpu_moe_env.grouped_prefill"),
              std::string::npos)
        << "Grouped shared-expert verifier rows are a hard GPU requirement now; "
           "do not route them through the removed grouped-prefill capability gate.";
    EXPECT_NE(compact_shared_policy.find("forceGroupedMoEVerifierPrefill(candidate)"),
              std::string::npos)
        << "The standalone grouped shared-expert verifier path is promoted for "
           "the same GPU small-M sidecar/main-verifier rows as the routed grouped "
           "prefill path. Do not recouple this with the failed combined "
           "routed+shared shortcut.";
    EXPECT_NE(compact_shared.find(
                  "constboolshared_gpu_table_verifier_prefill="
                  "shared_grouped_verifier_prefill&&"
                  "(shared_device.is_cuda()||shared_device.is_rocm());"),
              std::string::npos)
        << "CUDA and ROCm shared-expert verifier rows should use the promoted "
           "grouped table-prefill route rather than a row-replay verifier lane.";
    EXPECT_NE(compact_shared.find(
                  "shared_params.force_grouped_verifier_prefill_for_decode="
                  "shared_gpu_table_verifier_prefill;"),
              std::string::npos)
        << "The shared expert policy should expose the promoted GPU grouped "
           "table route for both CUDA and ROCm.";
    EXPECT_NE(compact_shared.find(
                  "shared_params.force_decode_equivalent_verifier_prefill="
                  "(!shared_gpu_table_verifier_prefill&&"
                  "(!shared_grouped_verifier_prefill&&"
                  "forceDecodeEquivalentMoEVerifier(shared_device)));"),
              std::string::npos)
        << "Decode-equivalent MoE routing must not force the independent GPU "
           "shared expert back onto row-serial verifier replay.";
    EXPECT_EQ(compact_shared.find("forceDecodeEquivalentMoERouting(shared_device)"),
              std::string::npos)
        << "Routing conservatism belongs to the router/routed-expert path only; "
           "recoupling it here reintroduces a large shared-expert replay cost.";

    const auto shared_dependency_section = sliceBetween(
        graph_source,
        "const bool main_verifier_rows =",
        "shared_ffn_last = prefix + \"shared_expert_ffn\";");
    const std::string compact_shared_dependency =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(shared_dependency_section));
    EXPECT_NE(compact_shared_dependency.find(
                  "constboolshared_verifier_owns_branch_local_math="
                  "main_verifier_rows&&(shared_gpu_table_verifier_prefill||"
                  "shared_params.force_decode_equivalent_verifier_prefill);"),
              std::string::npos)
        << "The promoted standalone shared verifier route must stay separated "
           "from backend MoE scratch ownership so routed and shared branches can "
           "be optimized independently.";
    EXPECT_NE(compact_shared_dependency.find(
                  "if(main_verifier_rows&&!shared_verifier_owns_branch_local_math&&"
                  "!ffn_terminal.empty())"),
              std::string::npos)
        << "Do not restore a blanket routed->shared verifier dependency. Only "
           "routes without branch-local shared verifier ownership should serialize.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, Qwen35GDNAllPositionVerifierBatchesCarryRequestShape)
{
    const auto graph_source = readFile(repoRoot() / "src/v2/models/qwen35/Qwen35Graph.cpp");
    const auto gdn_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h");
    const auto shortconv_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ShortConv1dStage.h");
    const auto tensor_header = readFile(repoRoot() / "src/v2/tensors/TensorKernels.h");
    const auto gdn_section = sliceBetween(
        graph_source,
        "ComputeGraph Qwen35Graph::buildGDNAttentionGraph(",
        "// Stage 1: Pre-attention RMSNorm");
    const std::string compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(gdn_section));

    EXPECT_EQ(compact.find("throwstd::runtime_error"),
              std::string::npos)
        << "GDN verifier request batches should fail at the backend capability "
           "boundary, not at graph construction.";
    EXPECT_NE(compact.find("per_request_verifier_state_capture_rows*std::max(1,batch_size)"),
              std::string::npos)
        << "Hybrid/GDN verifier state snapshots use flat transaction rows, so "
           "capture slots must scale by request count.";
    EXPECT_NE(gdn_header.find("int request_count = 1"),
              std::string::npos);
    EXPECT_NE(shortconv_header.find("int request_count = 1"),
              std::string::npos);
    EXPECT_NE(tensor_header.find("supportsRequestLiveStateBank"),
              std::string::npos)
        << "Request-batched GDN publication must be guarded by a backend "
           "live-state-bank capability.";
    EXPECT_NE(tensor_header.find("forwardBatchedRequests"),
              std::string::npos)
        << "Short-conv needs a request-batched execution hook, not a scalar loop.";
    EXPECT_NE(tensor_header.find("chunkForwardBatchedRequests"),
              std::string::npos)
        << "GDN recurrence needs a request-batched execution hook, not a scalar loop.";
    EXPECT_NE(graph_source.find("conv_params.request_count = batch_size"),
              std::string::npos)
        << "Short-conv verifier capture needs request shape, not only flattened token count.";
    EXPECT_NE(graph_source.find("rec_params.request_count = batch_size"),
              std::string::npos)
        << "GDN recurrence verifier capture needs request shape, not only flattened token count.";
    EXPECT_NE(graph_source.find("conv_params.request_seq_len = seq_len"),
              std::string::npos);
    EXPECT_NE(graph_source.find("rec_params.request_seq_len = seq_len"),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, VerifierStateBatchRestoreHasHardFailContract)
{
    const auto stage_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/IComputeStage.h");
    const auto tensor_header =
        readFile(repoRoot() / "src/v2/tensors/TensorKernels.h");
    const auto gdn_stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp");
    const auto shortconv_stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp");

    EXPECT_NE(stage_header.find("restoreVerifierStateCaptureRowsFromDeviceIndices"),
              std::string::npos)
        << "Request-batched verifier-state publication needs a first-class stage API.";
    EXPECT_NE(stage_header.find("This is not equivalent to looping"),
              std::string::npos)
        << "The stage contract must forbid silently looping scalar restore over a shared live state.";
    EXPECT_NE(tensor_header.find("restoreVerifierStateCaptureRowsFromDeviceIndices"),
              std::string::npos)
        << "GDN/short-conv tensor kernels need a backend hook for request-aware live-state banks.";
    EXPECT_NE(tensor_header.find("false is the correct behavior"),
              std::string::npos)
        << "Backend defaults must fail hard until capture layout and live state are request-aware.";
    EXPECT_NE(gdn_stage_source.find("GDNRecurrenceStage::restoreVerifierStateCaptureRowsFromDeviceIndices"),
              std::string::npos)
        << "GDN recurrence needs a stage-level batch restore hook, not an external special case.";
    EXPECT_NE(shortconv_stage_source.find("ShortConv1dStage::restoreVerifierStateCaptureRowsFromDeviceIndices"),
              std::string::npos)
        << "Short-conv needs a stage-level batch restore hook, not an external special case.";
    EXPECT_NE(removeAsciiWhitespace(stripCommentsAndStringLiterals(gdn_stage_source))
                  .find("params_.kernel->restoreVerifierStateCaptureRowsFromDeviceIndices("),
              std::string::npos)
        << "The GDN stage hook should delegate to the backend batch contract.";
    EXPECT_NE(removeAsciiWhitespace(stripCommentsAndStringLiterals(shortconv_stage_source))
                  .find("params_.kernel->restoreVerifierStateCaptureRowsFromDeviceIndices("),
              std::string::npos)
        << "The short-conv stage hook should delegate to the backend batch contract.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, PrefillReplayStagesPreserveCapturedResetState)
{
    const auto attention_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/AttentionComputeStage.h");
    const auto embedding_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/EmbeddingStage.h");
    const auto rope_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/RoPEStage.h");
    const auto kv_append_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/KVCacheAppendStage.h");
    const auto gdn_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h");
    const auto shortconv_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ShortConv1dStage.h");
    const auto moe_routing_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoERoutingStage.h");
    const auto moe_expert_header =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h");

    const std::vector<std::pair<std::string, std::string>> headers = {
        {"AttentionComputeStage", attention_header},
        {"EmbeddingStage", embedding_header},
        {"RoPEStage", rope_header},
        {"KVCacheAppendStage", kv_append_header},
        {"GDNRecurrenceStage", gdn_header},
        {"ShortConv1dStage", shortconv_header},
        {"MoERoutingStage", moe_routing_header},
        {"SharedExpertGateStage", moe_expert_header},
    };

    for (const auto &[name, source] : headers)
    {
        EXPECT_NE(source.find("resetSessionStatePreservingCapturedReplay"),
                  std::string::npos)
            << name << " must not fall back to resetSessionState() when a "
            << "Ready bucketed prefill graph is intentionally preserved.";
        EXPECT_NE(source.find("resetSessionStatePreservingLazyInitialization"),
                  std::string::npos)
            << name << " must keep warmup-created resources alive when an "
            << "Initialized bucket is captured after clear_cache().";
    }

    EXPECT_NE(attention_header.find("must not call resetDynamicState()"),
              std::string::npos)
        << "Attention reset comments should document why captured device-param "
           "storage survives request reset.";
    EXPECT_NE(gdn_header.find("Do not clear the verifier workspace binding"),
              std::string::npos)
        << "GDN reset comments should document why captured verifier-state "
           "workspace identities survive request reset.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPCatchupUsesOneGraphLifecycleContext)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    EXPECT_NE(source.find("kMTPDecodeCatchupContext"), std::string::npos)
        << "Accepted shifted-MTP KV catch-up needs a named logical graph lifecycle context.";
    EXPECT_EQ(source.find("\"mtp_decode_sequential_catchup\""), std::string::npos)
        << "Sequential and batched shifted-MTP catch-up must report the same graph lifecycle lane.";
    EXPECT_EQ(source.find("\"mtp_decode_sequential_catchup_device_target\""), std::string::npos)
        << "Device-token shifted-MTP catch-up must not fork graph lifecycle diagnostics.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, LocalTPMoEGroupedOutcomeRequiresTerminalHiddenCheckpoint)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto policy_body = sliceBetween(
        source,
        "const bool grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint",
        "if (use_device_publication_without_rollback_checkpoint)");
    const auto compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(policy_body));

    EXPECT_NE(compact.find(
                  "grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint="
                  "use_grouped_outcome_device_resident_publication_verifier&&"
                  "plan_.usesLocalTP()&&!runner_->supportsMTPShiftedRowReuseFromSidecar();"),
              std::string::npos)
        << "LocalTP MoE grouped-outcome publication must detect when the first "
           "shifted MTP row cannot be reused from the sidecar.";
    const size_t direct_publication_gate =
        compact.find("use_device_publication_without_rollback_checkpoint=");
    ASSERT_NE(direct_publication_gate, std::string::npos);
    const size_t synthesize_gate =
        compact.find("can_synthesize_verifier_base_checkpoint=");
    ASSERT_NE(synthesize_gate, std::string::npos);
    EXPECT_NE(compact.find(
                  "&&!grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint&&",
                  direct_publication_gate),
              std::string::npos)
        << "The initial direct-publication checkpoint path must not manufacture "
           "a token-count-only base when the shifted-row commit needs terminal hidden.";
    EXPECT_NE(compact.find(
                  "&&!grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint&&",
                  synthesize_gate),
              std::string::npos)
        << "The post-condition verifier-base checkpoint path must not synthesize "
           "a token-count-only base for this lane either.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, OrchestrationSnapshotsExposeTensorParallelSemanticViews)
{
    const auto header =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.h");
    const auto source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");

    EXPECT_NE(header.find("snapshot_combined_cache_"), std::string::npos)
        << "OrchestrationRunner must own TP-combined snapshot storage so getSnapshot() "
        << "can return a stable semantic view instead of a child-runner pointer.";
    EXPECT_NE(header.find("local-TP topology details"), std::string::npos)
        << "The snapshot cache member should document why topology-specific TP "
        << "assembly is hidden behind IOrchestrationRunner.";

    const auto get_snapshot_body = sliceBetween(
        source,
        "const float *OrchestrationRunner::getSnapshot(",
        "std::vector<std::string> OrchestrationRunner::getSnapshotKeys()");
    EXPECT_NE(get_snapshot_body.find("dynamic_cast<const RankOrchestrator *>"), std::string::npos)
        << "A RankOrchestrator-backed orchestration runner must publish TP-aware snapshots.";
    EXPECT_NE(get_snapshot_body.find("rank->getTPSnapshot(key)"), std::string::npos)
        << "Snapshot reads must use RankOrchestrator's sharding metadata instead of "
        << "returning only the primary participant snapshot.";
    EXPECT_NE(get_snapshot_body.find("snapshot_combined_cache_[key]"), std::string::npos)
        << "Combined snapshot data must live beyond the temporary TPSnapshot object.";

    const auto enable_body = sliceBetween(
        source,
        "void OrchestrationRunner::enableSnapshotCapture(",
        "void OrchestrationRunner::setSnapshotCaptureFilter(");
    const auto disable_body = sliceBetween(
        source,
        "void OrchestrationRunner::disableSnapshotCapture(",
        "void OrchestrationRunner::clearSnapshots()");
    const auto clear_body = sliceBetween(
        source,
        "void OrchestrationRunner::clearSnapshots()",
        "const float *OrchestrationRunner::getSnapshot(");
    EXPECT_NE(enable_body.find("snapshot_combined_cache_.clear()"), std::string::npos);
    EXPECT_NE(disable_body.find("snapshot_combined_cache_.clear()"), std::string::npos);
    EXPECT_NE(clear_body.find("snapshot_combined_cache_.clear()"), std::string::npos);
}
