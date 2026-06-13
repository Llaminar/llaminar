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

TEST(Test__GpuWorkspaceAllocationPolicy, MTPShiftedKVAsyncHandoffUsesEventBeforeConsumers)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");

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
    const size_t kv_only_guard = sidecar_body.find("if (!kv_cache_only)");
    const size_t logits_defer = sidecar_body.find("deferredSamplingStream");
    ASSERT_NE(kv_only_guard, std::string::npos);
    ASSERT_NE(logits_defer, std::string::npos);
    EXPECT_LT(kv_only_guard, logits_defer)
        << "KV-only sidecar replay must not use the pending-logits stream handoff; it owns shifted KV.";

    const auto publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const size_t publish_wait = publish_body.find("waitForPendingShiftedMTPKVReady");
    const size_t kv_publish = publish_body.find("publishAcceptedMTPSpecKVState");
    ASSERT_NE(publish_wait, std::string::npos);
    ASSERT_NE(kv_publish, std::string::npos);
    EXPECT_LT(publish_wait, kv_publish)
        << "Accepted-state publication truncates MTP KV and must wait for deferred shifted appends first.";
    const size_t terminal_hidden_publish = publish_body.find("selectMTPTerminalHiddenRow");
    const size_t ready_event = publish_body.find("recordAcceptedSpecPublicationReady");
    ASSERT_NE(terminal_hidden_publish, std::string::npos);
    ASSERT_NE(ready_event, std::string::npos);
    EXPECT_LT(terminal_hidden_publish, ready_event)
        << "Publication readiness must cover the accepted verifier terminal hidden row.";
    EXPECT_NE(publish_body.find("spec_state_terminal_hidden_publications"), std::string::npos)
        << "Terminal-hidden publication should be visible in perf stats.";

    const auto row_select_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPHiddenRowSelect(",
        "bool DeviceGraphOrchestrator::executeMTPTerminalHiddenRowSelect(");
    EXPECT_NE(row_select_body.find("cache.stage->setGPUStream(row_select_stream)"), std::string::npos)
        << "Publication must be able to bind terminal-hidden row-select to the publication stream.";

    const auto sequential_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(");
    const auto device_target_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(");
    const auto partial_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(",
        "const float *DeviceGraphOrchestrator::mtpLogits() const");
    EXPECT_NE(sequential_body.find("waitForPendingShiftedMTPKVReady"), std::string::npos);
    EXPECT_NE(device_target_body.find("waitForPendingShiftedMTPKVReady"), std::string::npos);
    EXPECT_NE(partial_body.find("waitForPendingShiftedMTPKVReady"), std::string::npos);
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
    EXPECT_NE(executable_sidecar_body.find("external_device_condition_tokens&&total_rows!=1"),
              std::string::npos)
        << "Only externally supplied device-token slots are limited to one row.";
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
        executable_build_body.find("clearStochasticDraftSampleReadySlot(slot)"),
        std::string::npos)
        << "Draft distribution builds still clear draft sample readiness because "
           "draft distribution slots and draft sampled-token slots share one "
           "step-local producer/consumer pair.";
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
     * but it should enqueue the token and metadata copies on the verifier
     * stream and then synchronize once.  Multiple deviceToHostFast() calls
     * would serialize the boundary and make ROCm/CUDA wait more than needed.
     */
    EXPECT_EQ(compact.find("deviceToHostFast("), std::string::npos);
    EXPECT_NE(compact.find("deviceToHostOnStream(output_tokens.data()"),
              std::string::npos);
    EXPECT_NE(compact.find("deviceToHostOnStream(meta.data()"),
              std::string::npos);
    const size_t first_copy =
        compact.find("deviceToHostOnStream(output_tokens.data()");
    const size_t second_copy =
        compact.find("deviceToHostOnStream(meta.data()");
    const size_t sync = compact.find("synchronizeStream(handle.stream");
    ASSERT_NE(first_copy, std::string::npos);
    ASSERT_NE(second_copy, std::string::npos);
    ASSERT_NE(sync, std::string::npos);
    EXPECT_LT(first_copy, sync);
    EXPECT_LT(second_copy, sync);
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPSpecStatePublicationPreservesSidecarReplayState)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto mutation_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::handleLivePrefixReplayStateAfterMutation(",
        "PrefixCacheFingerprintResult DeviceGraphOrchestrator::buildCurrentPrefixFingerprint(");

    /*
     * MTP spec-state publication replaces live main/MTP KV and recurrent state
     * with verifier-captured rows. Main/verifier replay caches still need a
     * correction boundary, but depth-0 sidecar graphs read stable arena buffers
     * and update dynamic token/position params before every launch. Resetting
     * sidecar segmented replay on every accepted token keeps ROCm/CUDA stuck in
     * warmup and destroys the vLLM-style speed path.
     */
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
    EXPECT_NE(executable_mutation_body.find("if(!correction_replay_boundary)"),
              std::string::npos)
        << "Spec-state publication must not reset sidecar replay caches.";
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
    EXPECT_NE(runtime_pointer_arrays.find("grouped gate/up pointer cache miss during graph capture"),
              std::string::npos);
    EXPECT_NE(runtime_pointer_arrays.find("grouped down pointer cache miss during graph capture"),
              std::string::npos);
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
