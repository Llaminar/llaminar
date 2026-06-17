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
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(");
    const auto resident_logical_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(");
    const auto partial_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(",
        "const float *DeviceGraphOrchestrator::mtpLogits() const");

    const auto sequential_executable = stripCommentsAndStringLiterals(sequential_body);
    const auto device_target_executable = stripCommentsAndStringLiterals(device_target_body);
    const auto resident_logical_executable = stripCommentsAndStringLiterals(resident_logical_body);
    const auto partial_executable = stripCommentsAndStringLiterals(partial_body);

    expectNeedleBefore(
        sequential_executable,
        "waitForPendingShiftedMTPKVReady",
        "get_cached_tokens",
        "Sequential shifted-row commits must order deferred graph appends before reading cache metadata.");
    expectNeedleBefore(
        device_target_executable,
        "waitForPendingShiftedMTPKVReady",
        "get_cached_tokens",
        "Device-target shifted-row commits must order deferred graph appends before reading cache metadata.");
    expectNeedleBefore(
        resident_logical_executable,
        "waitForDeviceResidentLogicalSequenceStateMailbox",
        "waitForPendingShiftedMTPKVReady",
        "Resident correction shifted commits must first receive the publication mailbox.");
    expectNeedleBefore(
        resident_logical_executable,
        "waitForPendingShiftedMTPKVReady",
        "get_cached_tokens",
        "Resident correction shifted commits must order deferred graph appends before reading cache metadata.");
    expectNeedleBefore(
        partial_executable,
        "waitForPendingShiftedMTPKVReady",
        "get_cached_tokens",
        "Partial-forward shifted-row commits must order deferred graph appends before reading cache metadata.");

    EXPECT_NE(resident_logical_body.find("waitForDeviceResidentLogicalSequenceStateMailbox"),
              std::string::npos)
        << "Resident correction shifted commits must wait on the publication mailbox before touching shifted KV.";
    EXPECT_NE(resident_logical_body.find("nextConditionTokenDeviceForRequest"),
              std::string::npos)
        << "Resident correction shifted commits must consume the device-derived next condition token.";
    EXPECT_NE(resident_logical_body.find("position_offset_override < 0"),
              std::string::npos)
        << "Resident correction shifted commits must require the caller's "
           "device-derived position offset.";
    EXPECT_EQ(resident_logical_executable.find("state_.positions"),
              std::string::npos)
        << "Resident correction shifted commits must not derive positions from "
           "host mirrors that may be stale until adoption.";
    EXPECT_EQ(resident_logical_executable.find("get_position"),
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

    const auto clear_cache_body = sliceBetween(
        header,
        "void clear_cache() override",
        "/**\n         * @brief Get current position");
    const auto clear_cache_executable =
        stripCommentsAndStringLiterals(clear_cache_body);
    EXPECT_NE(clear_cache_executable.find("mtp_terminal_hidden_row_select_cache_.invalidate()"),
              std::string::npos);
    EXPECT_NE(clear_cache_executable.find("mtp_terminal_hidden_rows_select_cache_.invalidate()"),
              std::string::npos);
    EXPECT_EQ(clear_cache_executable.find("mtp_terminal_hidden_row_select_cache_.resetSessionState()"),
              std::string::npos)
        << "clear_cache() must not preserve tiny terminal-hidden helper graphs after request-state teardown.";
    EXPECT_EQ(clear_cache_executable.find("mtp_terminal_hidden_rows_select_cache_.resetSessionState()"),
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
     * continues normal decode from it.  That captured row is only valid if it is
     * produced by the same recurrent-step math as serial decode.  The optimized
     * small-prefill route may use a different reduction geometry or in-place
     * preprocessing, so keep it out of this publishable-row helper.
     */
    const auto cuda_body = stripCommentsAndStringLiterals(sliceBetween(
        cuda_source,
        "bool cudaGDN_chunk_forward_via_single_row_chunks(",
        "bool cudaGDN_chunk_forward("));
    EXPECT_NE(cuda_body.find("cudaGDN_recurrent_step("), std::string::npos)
        << "CUDA publishable verifier rows must use serial decode recurrent_step.";
    EXPECT_EQ(cuda_body.find("cudaGDN_chunk_forward_kernel_route("), std::string::npos)
        << "CUDA verifier state snapshots must not come from the optimized prefill route.";

    const auto cuda_effective_body = stripCommentsAndStringLiterals(sliceBetween(
        cuda_source,
        "bool cudaGDN_chunk_forward_effective_via_single_row_chunks(",
        "bool cudaGDN_chunk_forward("));
    EXPECT_NE(cuda_effective_body.find("cudaGDN_recurrent_step_effective_row("), std::string::npos)
        << "CUDA padded verifier rows must remain serial decode-equivalent under graph replay.";
    EXPECT_NE(cuda_effective_body.find("cuda_gdn_copy_state_snapshot_if_active_kernel"), std::string::npos)
        << "CUDA padded verifier snapshots must be guarded by the device effective length.";
    EXPECT_EQ(cuda_effective_body.find("cudaGDN_chunk_forward_kernel_route("), std::string::npos)
        << "CUDA effective verifier snapshots must not come from the optimized prefill route.";

    const auto rocm_body = stripCommentsAndStringLiterals(sliceBetween(
        rocm_source,
        "bool rocmGDN_chunk_forward_via_single_row_chunks(",
        "bool rocmGDN_chunk_forward("));
    EXPECT_NE(rocm_body.find("rocmGDN_recurrent_step("), std::string::npos)
        << "ROCm publishable verifier rows must use serial decode recurrent_step.";
    EXPECT_EQ(rocm_body.find("rocmGDN_chunk_forward_kernel_route("), std::string::npos)
        << "ROCm verifier state snapshots must not come from the optimized prefill route.";

    const auto rocm_effective_body = stripCommentsAndStringLiterals(sliceBetween(
        rocm_source,
        "bool rocmGDN_chunk_forward_effective_via_single_row_chunks(",
        "bool rocmGDN_chunk_forward_effective("));
    EXPECT_NE(rocm_effective_body.find("rocmGDN_recurrent_step_effective_row("), std::string::npos)
        << "ROCm padded verifier rows must remain serial decode-equivalent under graph replay.";
    EXPECT_NE(rocm_effective_body.find("rocm_gdn_copy_state_snapshot_if_active_kernel"), std::string::npos)
        << "ROCm padded verifier snapshots must be guarded by the device effective length.";
    EXPECT_EQ(rocm_effective_body.find("rocmGDN_chunk_forward_kernel_route("), std::string::npos)
        << "ROCm effective verifier snapshots must not come from the optimized prefill route.";
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
    const auto compact_observation_wait =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(observation_wait_body));
    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));

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

    EXPECT_NE(compact_probe.find(
                  "waitForPendingAcceptedSpecPublicationReadyForObservation("),
              std::string::npos);
    EXPECT_NE(probe_body.find("\"prefix_state_probe\""),
              std::string::npos);
    EXPECT_NE(compact_payload.find(
                  "waitForPendingAcceptedSpecPublicationReadyForObservation("),
              std::string::npos);
    EXPECT_NE(payload_body.find("\"capture_live_prefix_state\""),
              std::string::npos);
    EXPECT_NE(compact_checkpoint.find(
                  "waitForPendingAcceptedSpecPublicationReadyForObservation("),
              std::string::npos);
    EXPECT_NE(checkpoint_body.find("\"capture_live_prefix_checkpoint\""),
              std::string::npos);

    expectNeedleBefore(
        compact_probe,
        "waitForPendingAcceptedSpecPublicationReadyForObservation(",
        "PrefixRuntimeStateSnapshotsnapshot;",
        "prefixStateProbe must order after accepted-state publication before reading probes.");
    expectNeedleBefore(
        compact_payload,
        "waitForPendingAcceptedSpecPublicationReadyForObservation(",
        "snapshot.valid=true;",
        "captureLivePrefixState must order after accepted-state publication before exporting payloads.");
    expectNeedleBefore(
        compact_checkpoint,
        "waitForPendingAcceptedSpecPublicationReadyForObservation(",
        "constintdraft_tokens=",
        "captureLivePrefixCheckpoint must order after accepted-state publication before reading live metadata.");
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
    EXPECT_NE(compact_capture.find("constbooldevice_only_checkpoint=false;"), std::string::npos)
        << "Hybrid logical checkpoints must include host GDN mirrors until a dedicated "
           "device-only replay equivalence test proves those mirrors are dead.";
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
        "bool DeviceGraphOrchestrator::adoptDeviceResidentMTPSpecPublishedHostState(");
    const auto single_publication_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatch(");
    const auto batch_publication_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatch(",
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(");
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

    const auto compact_restore = removeAsciiWhitespace(stripCommentsAndStringLiterals(restore_body));
    const auto compact_truncate = removeAsciiWhitespace(stripCommentsAndStringLiterals(truncate_body));
    const auto compact_sidecar = removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_body));
    const auto compact_prepare = removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_probe = removeAsciiWhitespace(stripCommentsAndStringLiterals(probe_body));
    const auto compact_payload = removeAsciiWhitespace(stripCommentsAndStringLiterals(payload_body));
    const auto compact_checkpoint = removeAsciiWhitespace(stripCommentsAndStringLiterals(checkpoint_body));
    const auto compact_device_publication = removeAsciiWhitespace(stripCommentsAndStringLiterals(device_publication_body));
    const auto compact_single_publication = removeAsciiWhitespace(stripCommentsAndStringLiterals(single_publication_body));
    const auto compact_batch_publication = removeAsciiWhitespace(stripCommentsAndStringLiterals(batch_publication_body));
    const auto compact_record = removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));
    const auto compact_wait = removeAsciiWhitespace(stripCommentsAndStringLiterals(wait_body));
    const auto compact_observation = removeAsciiWhitespace(stripCommentsAndStringLiterals(observation_body));

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
    EXPECT_NE(compact_probe.find("waitForPendingLivePrefixMutationReadyForObservation("), std::string::npos);
    EXPECT_NE(compact_payload.find("waitForPendingLivePrefixMutationReadyForObservation("), std::string::npos);
    EXPECT_NE(compact_checkpoint.find("waitForPendingLivePrefixMutationReadyForObservation("), std::string::npos);

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
        compact_single_publication,
        "handleLivePrefixReplayStateAfterMutation(",
        "recordLivePrefixMutationReady(",
        "Accepted spec-state publication mutates live KV/recurrent state and must publish the generic mutation handoff.");
    expectNeedleBefore(
        compact_single_publication,
        "recordAcceptedSpecPublicationReady(",
        "recordLivePrefixMutationReady(",
        "Single-request publication consumers wait accepted-state readiness before the generic live mutation handoff.");
    expectNeedleBefore(
        compact_batch_publication,
        "handleLivePrefixReplayStateAfterMutation(",
        "recordLivePrefixMutationReady(",
        "Batched accepted spec-state publication must publish the generic live mutation handoff.");
    expectNeedleBefore(
        compact_batch_publication,
        "recordAcceptedSpecPublicationReady(",
        "recordLivePrefixMutationReady(",
        "Batched publication consumers wait accepted-state readiness before the generic live mutation handoff.");
    expectNeedleBefore(
        compact_device_publication,
        "handleLivePrefixReplayStateAfterMutation(",
        "recordLivePrefixMutationReady(",
        "Device-resident accepted spec-state publication must publish the generic live mutation handoff.");
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
    const auto helper_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearLivePrefixRestoreTransientHandoffs(",
        "void DeviceGraphOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled(");
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
    EXPECT_NE(compact_helper.find("pending_mtp_verifier_device_token_plan_.reset()"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("request_batched_prefill_logits_row_count_=0"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("setRowIndexedAllPositionLogitRows({})"),
              std::string::npos)
        << "Prefix restore must drop compact verifier row ownership from the abandoned timeline.";
    EXPECT_NE(compact_helper.find("setComputeRowIndexedAllPositionLogits(false,0)"),
              std::string::npos)
        << "A restored prefix checkpoint must not inherit row-indexed verifier logits mode.";
    EXPECT_NE(compact_helper.find("setComputeAllPositionLogits(false)"),
              std::string::npos)
        << "A restored prefix checkpoint resumes as ordinary decode, not an all-position verifier.";
    EXPECT_EQ(compact_helper.find("&& !compute_all_position_logits_"), std::string::npos)
        << "Restore cleanup must clear verifier mode even when restore happens while all-position mode is active.";
    EXPECT_NE(compact_helper.find("clearStochasticTargetSampleReadySlots()"),
              std::string::npos);
    EXPECT_NE(compact_helper.find("clearStochasticDraftSampleReadySlots()"),
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
    const auto serial_helper = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool rocmGDN_chunk_forward_via_single_row_chunks(",
        "bool rocmGDN_chunk_forward(")));
    const auto public_route = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool rocmGDN_chunk_forward(",
        "bool rocmGDN_chunk_forward_effective(")));

    EXPECT_NE(compact.find("boolrocmGDN_chunk_forward_kernel_route("), std::string::npos)
        << "ROCm GDN should share one launch helper between normal and verifier-specific routes.";
    EXPECT_NE(compact.find("boolrocmGDN_chunk_forward_via_single_row_chunks("), std::string::npos)
        << "ROCm all-position verifier rows need the same serial-row path as CUDA.";
    EXPECT_NE(public_route.find("seq_len>0&&seq_len<=4"), std::string::npos)
        << "MTP verifier chunks are tiny and must publish serial-decode-equivalent GDN state.";
    EXPECT_NE(public_route.find("rocmGDN_chunk_forward_via_single_row_chunks("), std::string::npos);
    EXPECT_NE(public_route.find("rocmGDN_chunk_forward_effective_via_single_row_chunks("), std::string::npos)
        << "The effective-length route used by graph replay must share the serial-row verifier path.";
    EXPECT_NE(serial_helper.find("rocmGDN_recurrent_step("), std::string::npos)
        << "Each verifier row should run through the ordinary serial decode recurrent step.";
    EXPECT_EQ(serial_helper.find("rocmGDN_chunk_forward_kernel_route("), std::string::npos)
        << "Verifier state snapshots must not be produced by the optimized prefill route.";
    EXPECT_NE(serial_helper.find("hipMemcpyAsync("), std::string::npos);
    EXPECT_NE(serial_helper.find("hipMemcpyDeviceToDevice"), std::string::npos)
        << "Snapshot publication must stay device-resident and graph-capturable.";
    EXPECT_NE(serial_helper.find("static_cast<hipStream_t>(stream)"), std::string::npos)
        << "Snapshot copies must run on the caller's explicit capture stream.";
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
    EXPECT_NE(compact_publication.find("adoptDeviceResidentMTPSpecPublishedHostState("),
              std::string::npos)
        << "The resident branch must refresh host-visible positions without re-publishing state.";
    EXPECT_LT(compact_publication.find("publishAcceptedMTPSpecStateBatch("),
              compact_publication.find("adoptDeviceResidentMTPSpecPublishedHostState("))
        << "Host publication and resident host-state adoption must stay in separate branches.";

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
                  "backend->streamWaitEvent(copy_stream,handle.response_ready_event.get(),state_.device_id.gpu_ordinal())"),
              std::string::npos)
        << "The bridge stream must wait on the compact response event, not the "
           "full publication stream.";
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
        first_token_body.find("MTP greedy first-token GPU deferred sampling failed");
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
            "publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const size_t host_materialize =
        compact_greedy_runner.find(
            "materializeDeviceSpeculativeOutcomesForHostResponse(");
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

    EXPECT_NE(compact_external_rows.find("params.declare_selected_rows_workspace=false"),
              std::string::npos)
        << "Device-produced row metadata must be owned by the MTP metadata workspace, not by the row-select stage.";
    EXPECT_NE(compact_external_rows.find("params.upload_selected_rows_to_workspace=false"),
              std::string::npos)
        << "Device-produced row metadata must not be overwritten by host-side row uploads.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, MTPDeviceResidentPublicationMetadataStaysOnVerifierStream)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto direct_publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::adoptDeviceResidentMTPSpecPublishedHostState(");
    const auto compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));

    EXPECT_NE(compact.find("!request.outcome.stream"), std::string::npos)
        << "Device-resident publication metadata must reject implicit/default GPU streams.";
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

    EXPECT_NE(compact_support.find("supportsMTPSpecStatePublication()"),
              std::string::npos);
    EXPECT_NE(compact_support.find("supportsDeviceResidentLogicalSequenceStatePublication()"),
              std::string::npos)
        << "DGO must advertise direct publication only when the normal MTP "
           "state contract and resident logical-state handoff are both present.";
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
        "bool DeviceGraphOrchestrator::adoptDeviceResidentMTPSpecPublishedHostState(");
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
    const auto adopt_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::adoptDeviceResidentMTPSpecPublishedHostState(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    const auto compact_adopt =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(adopt_body));
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
    EXPECT_NE(compact_runner_interface.find("virtualbooladoptDeviceResidentMTPSpecPublishedHostState("),
              std::string::npos)
        << "Direct device publication needs a no-KV host mirror refresh hook.";
    EXPECT_NE(compact_runner_interface.find("virtualboolhostLogicalStateMirrorsDeviceResidentState()const"),
              std::string::npos)
        << "Planning code needs a typed freshness query before reading host logical getters.";
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
    EXPECT_NE(compact_header.find("device_resident_logical_sequence_host_adopted_epoch_"),
              std::string::npos)
        << "DGO must remember whether host mirrors adopted the current resident mailbox.";
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
    EXPECT_NE(compact_view.find("hostLogicalStateMirrorsDeviceResidentState()const"),
              std::string::npos);
    EXPECT_NE(compact_view.find("device_resident_logical_sequence_host_adopted_epoch_==mailbox.live_state_epoch"),
              std::string::npos)
        << "Host logical getters are fresh only after adopting the current mailbox epoch.";

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
    EXPECT_NE(compact_record.find("device_resident_logical_sequence_host_adopted_epoch_=0"),
              std::string::npos)
        << "Recording a resident mailbox must mark host logical mirrors stale.";
    EXPECT_NE(compact_clear_mailbox.find("device_resident_logical_sequence_host_adopted_epoch_=0"),
              std::string::npos)
        << "Clearing the mailbox must clear host-adoption freshness too.";
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
    EXPECT_NE(compact_retarget.find("device_resident_logical_sequence_host_adopted_epoch_=live_replay_state_epoch_"),
              std::string::npos)
        << "Shifted-KV-only retargets must keep already-adopted host logical mirrors fresh.";
    EXPECT_NE(retarget_body.find("device_resident_logical_state_mailbox_retargets"),
              std::string::npos)
        << "Retargets should be visible in perf counters.";

    EXPECT_NE(compact_sidecar.find("constvoid*position_ids_device_override"),
              std::string::npos)
        << "Sidecar replay must be able to consume device-resident position rows.";
    EXPECT_NE(compact_sidecar.find("input.position_ids_device=position_ids_device_override"),
              std::string::npos);
    EXPECT_NE(compact_sidecar.find("cached_input.position_ids_device=position_ids_device_override"),
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
    EXPECT_NE(compact_attention_source.find("!tq_cache&&"),
              std::string::npos)
        << "TQ/hybrid cache attention must stay guarded until it has an equivalent device metadata path.";
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
    EXPECT_NE(compact_header.find("adoptDeviceResidentMTPSpecPublishedHostState("),
              std::string::npos);
    EXPECT_NE(compact_adopt.find("device_resident_logical_sequence_state_mailbox_"),
              std::string::npos)
        << "Host mirror adoption must be tied to the current resident mailbox.";
    EXPECT_NE(compact_adopt.find("state_.positions[static_cast<size_t>(step.request_index)]=step.target_cached_tokens"),
              std::string::npos);
    EXPECT_NE(compact_adopt.find("state_.sequence_lengths[static_cast<size_t>(step.request_index)]=step.target_cached_tokens"),
              std::string::npos);
    EXPECT_NE(compact_adopt.find("HostSequenceStatePublicationRequestkv_host_request"),
              std::string::npos);
    EXPECT_NE(compact_adopt.find("adoptSequenceStateFromHostMetadata("),
              std::string::npos)
        << "DGO host adoption must refresh KV cache mirrors before positions are trusted.";
    EXPECT_NE(compact_adopt.find("state_.mtp_kv_caches"),
              std::string::npos)
        << "Host adoption must refresh shifted MTP KV cache mirrors too.";
    EXPECT_NE(compact_adopt.find("shifted_target_cached_tokens"),
              std::string::npos);
    EXPECT_NE(compact_adopt.find("shifted_accepted_state_counts"),
              std::string::npos);
    EXPECT_LT(compact_adopt.find("adoptSequenceStateFromHostMetadata("),
              compact_adopt.find("state_.positions[static_cast<size_t>(step.request_index)]=step.target_cached_tokens"))
        << "KV host mirrors must be adopted before DGO exposes the new host position.";
    EXPECT_NE(compact_adopt.find("device_resident_logical_sequence_host_adopted_epoch_=mailbox.live_state_epoch"),
              std::string::npos)
        << "Host mirror adoption must mark the current mailbox epoch as fresh.";
    EXPECT_EQ(compact_adopt.find("publishSequenceStateFromDeviceMetadata("),
              std::string::npos)
        << "Host mirror adoption must not republish KV/cache state.";
    EXPECT_EQ(compact_adopt.find("publishAcceptedMTPSpecKVState("),
              std::string::npos)
        << "Host mirror adoption must not call the host KV publication path.";
    EXPECT_EQ(compact_adopt.find("synchronizeStream("),
              std::string::npos)
        << "Host mirror adoption must use the existing host bridge plan, not add a sync.";

    const auto clear_cache_body = sliceBetween(
        header,
        "void clear_cache() override",
        "int get_position() const override");
    EXPECT_NE(removeAsciiWhitespace(stripCommentsAndStringLiterals(clear_cache_body))
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
    EXPECT_NE(compact_planner_helper.find("!runner_->hostLogicalStateMirrorsDeviceResidentState()"),
              std::string::npos)
        << "Planning must refuse stale host mirrors while a resident mailbox is current.";
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
    EXPECT_NE(compact.find("structHostSequenceStatePublicationRequest"),
              std::string::npos);
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
    EXPECT_NE(compact.find("adoptSequenceStateFromHostMetadata("),
              std::string::npos)
        << "Direct device publication must have a matching host mirror adoption contract.";
    EXPECT_NE(source.find("This method must not enqueue device work"),
              std::string::npos)
        << "Host mirror adoption must not smuggle a GPU sync or upload into the cache.";
    EXPECT_NE(compact.find("returnfalse;"),
              std::string::npos)
        << "The default IKVCache implementation must hard-fail until a cache "
           "owns device-readable count/head state.";
    EXPECT_NE(source.find("Long-context ring caches"), std::string::npos)
        << "The contract should document why target count alone is not enough "
           "to update wrapped ring-cache heads.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, KVCacheDeviceCountMirrorsAreSymmetricAndGraphAdvanced)
{
    const auto cuda_base =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.h") +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.cpp") +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto rocm_base =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.h") +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp") +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp") +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheKernels.hip");
    const auto cuda_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_base));
    const auto rocm_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_base));

    for (const auto *compact : {&cuda_compact, &rocm_compact})
    {
        EXPECT_NE(compact->find("d_count_params_"), std::string::npos);
        EXPECT_NE(compact->find("h_count_params_"), std::string::npos);
        EXPECT_NE(compact->find("deviceCachedTokenCountPtr("), std::string::npos);
        EXPECT_NE(compact->find("deviceRingHeadPtr("), std::string::npos);
        EXPECT_NE(compact->find("supportsDeviceResidentSequenceStatePublication()constoverride"),
                  std::string::npos);
        EXPECT_NE(compact->find("d_head_params_!=nullptr&&d_count_params_!=nullptr"),
                  std::string::npos)
            << "Caches must advertise direct publication only when both device mirrors exist.";
        EXPECT_NE(compact->find("refreshHostDeviceParamMirror("), std::string::npos);
        EXPECT_NE(compact->find("uploadHostDeviceParamMirror("), std::string::npos)
            << "Non-captured GPU cache mutations must refresh the device "
               "count/head mirror before attention derives params from it.";
        EXPECT_NE(compact->find("!uploadHostDeviceParamMirror("),
                  std::string::npos)
            << "A failed mirror upload on an explicit stream must not be ignored.";
        EXPECT_NE(compact->find("kv_sequence_state_advance_kernel"),
                  std::string::npos)
            << "Captured append replay must advance the device-side count/head "
               "mirror in the graph stream, not by a later host upload.";
        EXPECT_NE(compact->find("kv_sequence_state_advance("),
                  std::string::npos);
    }
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
            << "Wrapped ring heads must advance by the committed verifier-row count.";
        EXPECT_NE(compact->find("target_cached_tokens[request_idx]"),
                  std::string::npos)
            << "The target cached-token count is only the valid-token count.";
        EXPECT_NE(compact->find("(old_head+accepted_count)%max_seq_len"),
                  std::string::npos)
            << "A wrapped-ring head must be derived from the live device head plus accepted rows.";
        EXPECT_NE(compact->find("d_counts[entry_idx]=target_count"),
                  std::string::npos)
            << "Device publication must update the live count mirror for subsequent attention.";
        EXPECT_NE(compact->find("adoptSequenceStateFromHostMetadata("),
                  std::string::npos);
        EXPECT_NE(compact->find("setEntryHead(layer,seq_idx,(old_head+accepted_count)%max_seq_len_)"),
                  std::string::npos)
            << "Host adoption must mirror the wrapped-ring head advance.";
        EXPECT_NE(compact->find("setEntryCount(layer,seq_idx,target_count)"),
                  std::string::npos)
            << "Host adoption must mirror the target valid-token count.";
        EXPECT_NE(compact->find("refreshHostDeviceParamMirror(layer,seq_idx)"),
                  std::string::npos)
            << "Pinned host mirrors must match the host entry after adoption.";
        EXPECT_NE(compact->find("onAdvanceComplete(layer,seq_idx)"),
                  std::string::npos)
            << "Scratch/shadow hooks must see host mirror adoption too.";
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
    EXPECT_NE(stage_compact.find("!tq_cache"), std::string::npos)
        << "TQ/hybrid metadata must stay on its own guarded contract until it "
           "has the same device-owned append/count semantics.";

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
     * sidecar transaction proves it does not capture stale metadata. This guard
     * prevents the two contracts from collapsing into one broad shortcut.
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
    EXPECT_NE(executable_mutation_body.find(
                  "resetCapturedReplayStateForCorrectionReplay(live_replay_state_epoch_,preserve_single_token_decode_replay)"),
              std::string::npos)
        << "Correction publication must pass an explicit single-token decode "
           "replay policy instead of assuming the dense shortcut is always safe.";
    EXPECT_NE(executable_mutation_body.find(
                  "constboolpreserve_single_token_decode_replay=!isPrefixCacheMoEModel();"),
              std::string::npos)
        << "MoE accepted-state publication must recapture ordinary decode "
           "until its routed scratch/expert metadata replay is equivalence-tested.";
    EXPECT_NE(mutation_body.find("reset_for_moe"), std::string::npos)
        << "Perf stats must make MoE ordinary-decode recapture visible.";
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
        "void DeviceGraphOrchestrator::clearDeviceResidentLogicalSequenceStateMailbox()");

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
    EXPECT_NE(compact_spec_publication_support.find("config.isMoE()||isPrefixCacheMoEModel()"),
              std::string::npos)
        << "MoE direct all-position publication must stay disabled until "
           "verifier rows are serial-equivalent.";
    EXPECT_NE(compact_spec_publication_support.find("returnfalse;"),
              std::string::npos)
        << "MoE direct state publication should fail closed instead of using "
           "the unproven all-position verifier shortcut.";
    EXPECT_NE(sidecar_replay_body.find("supportsMTPSidecarPreservesMainState()"),
              std::string::npos)
        << "The sidecar replay contract must remain explicitly narrower than "
           "main-state preservation and shifted-row reuse.";
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

    EXPECT_NE(compact_forward_engine.find("constboolhas_position_input=input.position_ids!=nullptr||(input.position_ids_device!=nullptr&&input.device.is_gpu())"),
              std::string::npos)
        << "GPU resident position rows must count as stable graph inputs.";
    EXPECT_NE(compact_forward_engine.find("input.position_ids_device!=nullptr"),
              std::string::npos)
        << "The cache signature should record resident-position mode.";
    EXPECT_NE(compact_forward_engine.find("stage->updateDynamicDevicePositionIds(input.position_ids_device,input.seq_len)"),
              std::string::npos)
        << "Cache-hit replay must refresh the resident RoPE pointer before capture/replay.";
    EXPECT_NE(compact_forward_engine.find("stage->updateDynamicPositionIds(cached_position_ids,input.seq_len)"),
              std::string::npos)
        << "Cache-hit replay must keep host explicit position rows fresh too.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, Qwen35MoEMultiRowVerifierUsesDecodeEquivalentExpertPathOnAllBackends)
{
    const auto graph_source = readFile(repoRoot() / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto predicate_section = sliceBetween(
        graph_source,
        "auto forceDecodeEquivalentMoEVerifier = [&](DeviceId candidate)",
        "LayerWeightBindings layer_bindings");
    const std::string compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(predicate_section));

    EXPECT_NE(compact.find("(candidate.is_cpu()||candidate.is_cuda()||candidate.is_rocm())"),
              std::string::npos)
        << "Main-verifier MoE publication rows must reuse serial decode expert execution on CPU, CUDA, and ROCm. "
           "The grouped small-M path is not a valid publication contract until row parity proves it.";
    EXPECT_NE(compact.find("total_tokens>1&&total_tokens<=4"),
              std::string::npos)
        << "Only multi-row verifier publication should force the decode-equivalent path; one-token correction replay "
           "can keep the grouped verifier route.";

    const auto combined_shared_section = sliceBetween(
        graph_source,
        "const bool can_combine_shared_verifier =",
        "if (overlay_requested && !use_expert_overlay)");
    EXPECT_NE(removeAsciiWhitespace(stripCommentsAndStringLiterals(combined_shared_section))
                  .find("!forceDecodeEquivalentMoEVerifier(device)"),
              std::string::npos)
        << "Combined shared+routed verifier prefill is a grouped small-M route; it must stay disabled whenever "
           "the backend is publishing decode-equivalent all-position rows.";
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
