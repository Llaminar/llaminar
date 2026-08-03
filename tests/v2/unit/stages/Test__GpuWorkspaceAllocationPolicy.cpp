/**
 * @file Test__GpuWorkspaceAllocationPolicy.cpp
 * @brief Source-level guards for capture-sensitive GPU workspace allocation policy.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEWorkspaceRequirements.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
    std::string stripCommentsAndStringLiterals(const std::string &source);

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
        const auto executable_source = stripCommentsAndStringLiterals(source);
        const std::vector<std::string> needles = {
            "cudaMalloc(",
            "cudaMallocAsync(",
            "cudaFree(",
            "cudaFreeAsync(",
            "hipMalloc(",
            "hipMallocAsync(",
            "hipFree(",
            "hipFreeAsync(",
        };
        for (const auto &needle : needles)
        {
            if (executable_source.find(needle) != std::string::npos)
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

    bool hasExecutableStageDeviceAllocationCall(const std::string &source)
    {
        const auto executable_source = stripCommentsAndStringLiterals(source);
        const std::array<const char *, 14> forbidden = {
            "allocateOnDevice(",
            "allocateOnDevice (",
            "prepareDeviceInput(",
            "prepareDeviceInput (",
            "prepareDeviceOutput(",
            "prepareDeviceOutput (",
            "allocateDeviceStorage(",
            "allocateDeviceStorage (",
            ".prepareInput(",
            ".prepareInput (",
            ".prepareOutput(",
            ".prepareOutput (",
            "TransferEngine::instance().upload(",
            "TransferEngine::instance().upload (",
        };
        return std::any_of(
            forbidden.begin(),
            forbidden.end(),
            [&](const char *needle)
            {
                return executable_source.find(needle) != std::string::npos;
            });
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
        return extension == ".c" || extension == ".cc" ||
               extension == ".cpp" || extension == ".cxx" ||
               extension == ".cu" || extension == ".cuh" ||
               extension == ".hip" ||
               extension == ".h" || extension == ".hpp" ||
               extension == ".hxx" ||
               extension == ".inc" || extension == ".inl" ||
               extension == ".ipp";
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
        "src/v2/backends/benchmarks/CUDABenchmark.cu",
        "src/v2/backends/benchmarks/ROCmBenchmark.cpp",
        "src/v2/backends/cuda/CUDABackend.cu",
        "src/v2/backends/cuda/CUDATensorValidation.cu",
        "src/v2/backends/rocm/ROCmBackend.cpp",
        "src/v2/backends/rocm/ROCmTensorValidation.cpp",
        "src/v2/collective/backends/NCCLBackendCUDA.cu",
        "src/v2/collective/backends/RCCLBackendHIP.cpp",
        "src/v2/collective/coordinators/RCCLCoordinator.cpp",
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

TEST(Test__GpuWorkspaceAllocationPolicy, CUDARowMajorPreparationCannotRunInsideDispatch)
{
    const auto source = stripCommentsAndStringLiterals(readFile(
        repoRoot() /
        "src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvShardImpl.cu.inc"));
    ASSERT_FALSE(source.empty());

    const auto dispatch_source = sliceBetween(
        source,
        "bool dispatchCodebook(",
        "CUDAGemvContext *cudaGemvContext_create");
    ASSERT_FALSE(dispatch_source.empty());

    EXPECT_EQ(
        dispatch_source.find("cudaRowMajorWeights_create("),
        std::string::npos)
        << "ROWPAR weights must be prepared before dispatch; launch selectors "
           "cannot allocate or transpose persistent weight storage";
}

TEST(Test__GpuWorkspaceAllocationPolicy, KVCacheBatchSlicesCarryExactDeviceAndStream)
{
    const auto root = repoRoot();
    const auto stage = stripCommentsAndStringLiterals(readFile(
        root /
        "src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp"));
    const auto prepared_view = stripCommentsAndStringLiterals(readFile(
        root / "src/v2/tensors/GpuTensorView.h"));
    const auto cuda_adapter = stripCommentsAndStringLiterals(readFile(
        root /
        "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTensorAdapter.cpp"));
    const auto rocm_adapter = stripCommentsAndStringLiterals(readFile(
        root /
        "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp"));

    EXPECT_EQ(
        countOccurrences(stage, "PreparedGpuTensorView"),
        2u)
        << "Batched K and V slices must use the stream-qualified prepared view";
    EXPECT_FALSE(
        std::regex_search(
            stage,
            std::regex(R"((^|[^[:alnum:]_])GpuTensorView[[:space:]]+k_view[[:space:]]*\()")))
        << "The ordinal-only generic view loses backend and producer ordering";
    EXPECT_NE(
        prepared_view.find("friend class KVCacheAppendStage"),
        std::string::npos)
        << "Only KVCacheAppendStage may mint prepared device-slice authority";
    for (const auto &[backend, source] : {
             std::pair{"CUDA", &cuda_adapter},
             std::pair{"ROCm", &rocm_adapter}})
    {
        EXPECT_NE(
            source->find("dynamic_cast<const PreparedGpuTensorView *>"),
            std::string::npos)
            << backend;
        EXPECT_NE(
            source->find("isPreparedFor(target, gpu_stream)"),
            std::string::npos)
            << backend;
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, LocalTPCollectiveExecutionCannotAllocateScratch)
{
    const std::string source =
        readFile(repoRoot() / "src/v2/collective/LocalTPContext.cpp");
    ASSERT_FALSE(source.empty());

    const auto methodBody = [&](const std::string &begin_marker,
                                const std::string &end_marker)
    {
        const size_t begin = source.find(begin_marker);
        EXPECT_NE(begin, std::string::npos)
            << "Missing LocalTP allocation-policy marker: " << begin_marker;
        if (begin == std::string::npos)
            return std::string{};
        const size_t end = source.find(end_marker, begin + begin_marker.size());
        EXPECT_NE(end, std::string::npos)
            << "Missing LocalTP allocation-policy end marker: " << end_marker;
        return stripCommentsAndStringLiterals(
            end == std::string::npos
                ? source.substr(begin)
                : source.substr(begin, end - begin));
    };

    const std::string plain_allreduce = methodBody(
        "bool LocalTPContext::allreduceOnStream(",
        "bool LocalTPContext::allreduce(const TensorBase *input");
    const std::string bundled_allreduce = methodBody(
        "bool LocalTPContext::allreduceWithSidebandsOnStream(",
        "bool LocalTPContext::collectiveSidebandOnStream(");
    const std::string scratch_requirement = methodBody(
        "void *LocalTPContext::requireReservedFp16Scratch(",
        "bool LocalTPContext::reserveCollectiveResources(");

    for (const auto &[label, body] : std::array{
             std::pair{"allreduceOnStream", plain_allreduce},
             std::pair{"allreduceWithSidebandsOnStream", bundled_allreduce},
             std::pair{"requireReservedFp16Scratch", scratch_requirement}})
    {
        EXPECT_EQ(body.find("->allocate("), std::string::npos)
            << label << " must never allocate device storage";
        EXPECT_EQ(body.find("->free("), std::string::npos)
            << label << " must never release device storage";
        EXPECT_EQ(body.find("reserveCollectiveResources("), std::string::npos)
            << label << " must not repair an incomplete setup reservation";
    }

    EXPECT_EQ(source.find("cudaFP16ScratchAlloc"), std::string::npos);
    EXPECT_EQ(source.find("rocmFP16ScratchAlloc"), std::string::npos);
    EXPECT_EQ(source.find("cudaCollectiveControlAlloc"), std::string::npos);
    EXPECT_EQ(source.find("rocmCollectiveControlAlloc"), std::string::npos);
    EXPECT_NE(plain_allreduce.find("requireReservedFp16Scratch("), std::string::npos);
    EXPECT_NE(bundled_allreduce.find("requireReservedFp16Scratch("), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, ComputeStagesCannotAllocateOrUploadGpuStorage)
{
    const auto root = repoRoot();
    const auto stages_root =
        root / "src/v2/execution/compute_stages/stages";
    const std::string heterogeneous_transport_boundary =
        "src/v2/execution/compute_stages/stages/MoELocalExpertStage.cpp";
    std::vector<std::string> failures;

    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(stages_root))
    {
        if (!entry.is_regular_file() || !isSourceFile(entry.path()))
            continue;
        const auto source = readFile(entry.path());
        const auto relative =
            std::filesystem::relative(entry.path(), root).generic_string();
        if (hasExecutableStageDeviceAllocationCall(source))
        {
            /*
             * This is the one graph-stage-shaped transport boundary: sparse
             * rows cross a heterogeneous CPU/GPU participant domain through
             * host packets, and the stage is explicitly non-capturable. Keep
             * the exception exact and visible; homogeneous CUDA/ROCm stages
             * are never permitted to inherit it.
             */
            if (relative != heterogeneous_transport_boundary)
                failures.push_back(relative);
        }
    }

    EXPECT_TRUE(failures.empty()) << [&]
    {
        std::ostringstream out;
        out << "Compute stages may launch work and publish writes, but they may "
               "not allocate GPU storage or upload host data. Declare arena "
               "buffers in StageBufferContract or bind DeviceWorkspaceManager "
               "storage before execution. This guard applies equally to CUDA "
               "and ROCm.\n";
        for (const auto &failure : failures)
            out << failure << '\n';
        return out.str();
    }();
}

TEST(
    Test__GpuWorkspaceAllocationPolicy,
    GraphExecutionCannotRepairMissingWeightResidency)
{
    const auto root = repoRoot();
    const std::array<std::string, 2> executor_sources = {
        "src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp",
        "src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp",
    };

    for (const auto &relative : executor_sources)
    {
        const auto executable = stripCommentsAndStringLiterals(
            readFile(root / relative));
        EXPECT_EQ(
            executable.find("TransferEngine::prepareDeviceInput("),
            std::string::npos)
            << relative
            << " must fail on missing GPU weight residency instead of "
               "allocating or uploading during execution";
        EXPECT_NE(
            executable.find("TransferEngine::requireDeviceInput("),
            std::string::npos)
            << relative
            << " must validate exact residency and join producer ordering";
    }
}

TEST(
    Test__GpuWorkspaceAllocationPolicy,
    PreparedKernelStateCannotMasqueradeAsRawTransferResidency)
{
    const auto source = stripCommentsAndStringLiterals(
        readFile(repoRoot() / "src/v2/transfer/TransferEngine.cpp"));
    const auto upload_body = sliceBetween(
        source,
        "TransferResult TransferEngine::uploadFull(",
        "TransferResult TransferEngine::downloadFull(");

    EXPECT_EQ(
        upload_body.find("hasPreparedDeviceState("),
        std::string::npos)
        << "A backend-owned packed representation is not a raw TensorBase "
           "allocation. TransferEngine must perform the requested raw upload "
           "or fail; it may not advertise success from unrelated prepared state.";
}

TEST(
    Test__GpuWorkspaceAllocationPolicy,
    HeterogeneousSparseTransportIsTheOnlyStageOwnedDeviceTransferBoundary)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/compute_stages/stages/MoELocalExpertStage.cpp");
    const auto header = readFile(
        repoRoot() /
        "src/v2/execution/compute_stages/stages/MoELocalExpertStage.h");
    const auto executable = stripCommentsAndStringLiterals(source);

    EXPECT_EQ(countOccurrences(executable, ".prepareInput("), 3u);
    EXPECT_EQ(countOccurrences(executable, ".prepareOutput("), 1u);
    EXPECT_NE(
        header.find("bool isGraphCapturable() const override { return false; }"),
        std::string::npos)
        << "The heterogeneous host-packet bridge must remain visibly excluded "
           "from native graph capture.";
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

TEST(
    Test__GpuWorkspaceAllocationPolicy,
    GDNKernelsConsumeOnlyCacheOwnedPersistentState)
{
    const auto root = repoRoot();
    const std::array<std::string, 6> kernel_sources = {
        "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNet.h",
        "src/v2/kernels/cuda/gdn/CUDAShortConvolution.h",
        "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNetKernels.cu",
        "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h",
        "src/v2/kernels/rocm/gdn/ROCmShortConvolution.h",
        "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip",
    };
    const std::array<const char *, 8> forbidden = {
        "allocateGPUState(",
        "allocateGPUScratch(",
        "cudaGDN_gpu_malloc(",
        "cudaGDN_gpu_free(",
        "rocmGDN_gpu_malloc(",
        "rocmGDN_gpu_free(",
        "backend->allocate(",
        "backend->free(",
    };

    for (const auto &relative : kernel_sources)
    {
        const auto executable = stripCommentsAndStringLiterals(
            readFile(root / relative));
        for (const char *needle : forbidden)
        {
            EXPECT_EQ(executable.find(needle), std::string::npos)
                << relative
                << " must consume cache-owned persistent state and stage-owned "
                   "scratch; hidden capacity repair is forbidden: "
                << needle;
        }
    }

    const auto tensor_api = stripCommentsAndStringLiterals(
        readFile(root / "src/v2/tensors/TensorKernels.h"));
    EXPECT_EQ(tensor_api.find("allocateGPUState("), std::string::npos);
    EXPECT_EQ(tensor_api.find("allocateGPUScratch("), std::string::npos);
    EXPECT_EQ(countOccurrences(tensor_api, "bindDeviceState("), 2u)
        << "Short-conv and recurrence expose one typed persistent-state "
           "binding contract each";
}

/**
 * @brief Prevent host-side request-bank validity from re-entering GPU GDN state.
 *
 * CUDA/HIP graph replay executes the recorded device operations without
 * re-running the C++ wrapper that originally submitted them. A host field that
 * claims a request-state bank is current therefore cannot describe graph-replay
 * state. Likewise, swapping a kernel's primary state pointer while constructing
 * local/full LocalTP graphs makes graph build order alter later bindings.
 *
 * Request-bank publication must remain an explicit device copy on the producer
 * stream, and LocalTP handoff stages must name the state geometry they consume.
 */
TEST(
    Test__GpuWorkspaceAllocationPolicy,
    GDNRequestBanksHaveImmutableBindingsAndDevicePublishedValidity)
{
    const auto root = repoRoot();
    const std::array<std::string, 4> kernel_headers = {
        "src/v2/kernels/cuda/gdn/CUDAGatedDeltaNet.h",
        "src/v2/kernels/cuda/gdn/CUDAShortConvolution.h",
        "src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h",
        "src/v2/kernels/rocm/gdn/ROCmShortConvolution.h",
    };
    const std::array<const char *, 5> forbidden = {
        "request_state_bank_state_size_",
        "selectState(",
        "ensureActiveState(",
        "effective_state == gpu_state_",
        "active_state_size_",
    };

    for (const auto &relative : kernel_headers)
    {
        const auto executable = stripCommentsAndStringLiterals(
            readFile(root / relative));
        for (const char *needle : forbidden)
        {
            EXPECT_EQ(executable.find(needle), std::string::npos)
                << relative
                << " must not mirror device-state validity or mutate primary "
                   "state bindings on the host: "
                << needle;
        }
        EXPECT_NE(
            executable.find("publishLiveStateToRequestZero("),
            std::string::npos)
            << relative
            << " must publish the authoritative live row into request slot "
               "zero through an explicit stream-ordered device operation";
        EXPECT_NE(executable.find("stateForSize("), std::string::npos)
            << relative
            << " must resolve local/full state geometry without mutating the "
               "kernel's stable primary and secondary bindings";
    }

    const std::array<std::string, 2> handoff_sources = {
        "src/v2/execution/compute_stages/stages/GDNLiveStateLocalizeStage.cpp",
        "src/v2/execution/compute_stages/stages/GDNLiveStateAllGatherStage.cpp",
    };
    for (const auto &relative : handoff_sources)
    {
        const auto executable = stripCommentsAndStringLiterals(
            readFile(root / relative));
        EXPECT_EQ(executable.find("exportState("), std::string::npos)
            << relative
            << " must export the declared local/full geometry explicitly";
        EXPECT_EQ(executable.find("importState("), std::string::npos)
            << relative
            << " must import the declared local/full geometry explicitly";
        EXPECT_NE(executable.find("exportStateForSize("), std::string::npos)
            << relative
            << " must use the size-explicit state export contract";
        EXPECT_NE(executable.find("importStateForSize("), std::string::npos)
            << relative
            << " must use the size-explicit state import contract";
    }
}

TEST(
    Test__GpuWorkspaceAllocationPolicy,
    HybridCachePlansGDNStateBeforeKernelConstruction)
{
    const auto root = repoRoot();
    const auto arena = stripCommentsAndStringLiterals(readFile(
        root / "src/v2/kernels/HybridGDNDeviceStateArena.h"));
    const auto factory = stripCommentsAndStringLiterals(readFile(
        root / "src/v2/kernels/KernelFactory.cpp"));
    const auto handoff = stripCommentsAndStringLiterals(readFile(
        root /
        "src/v2/execution/compute_stages/stages/GDNLiveStateAllGatherStage.cpp"));

    EXPECT_NE(arena.find("DeviceWorkspaceManager"), std::string::npos);
    EXPECT_NE(arena.find("zeroAll(initialization_stream)"), std::string::npos);
    EXPECT_EQ(arena.find("synchronize"), std::string::npos)
        << "Persistent state initialization is stream ordered, never host blocked";

    EXPECT_EQ(countOccurrences(factory, "bindDeviceState("), 2u)
        << "KernelFactory must bind both short-conv and recurrence state";
    EXPECT_EQ(factory.find("allocateGPUState("), std::string::npos);

    EXPECT_EQ(countOccurrences(handoff, "importStateForSize("), 2u)
        << "LocalTP handoff must select the prebound full-size conv and "
           "recurrence banks by exact geometry";
    EXPECT_EQ(handoff.find("allocateGPUState("), std::string::npos);
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
    EXPECT_GE(
        expert_mask->size_bytes,
        static_cast<size_t>(llaminar2::MoEWorkspaceBuffers::kGroupedDescriptorTableSlots) *
            static_cast<size_t>(num_experts) * sizeof(uint8_t))
        << "Each graph-lifetime MoE owner needs an exclusive immutable mask slot";
    EXPECT_GT(static_cast<size_t>(num_experts), static_cast<size_t>(max_seq_len) * top_k)
        << "fixture must cover the small-token, many-expert regression";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoEPersistentMetadataCapacityCoversEveryLayerAndGraphOwner)
{
    using namespace llaminar2::MoEWorkspaceBuffers;
    constexpr int kExperts = 256;

    EXPECT_EQ(
        kGroupedDescriptorTableSlots,
        kMaximumMoELayersPerDevice *
            kMaximumGroupedDescriptorOwnersPerMoELayer)
        << "Persistent descriptor capacity must be derived from both model "
           "depth and simultaneously retained graph-role ownership";
    EXPECT_GE(kMaximumGroupedDescriptorOwnersPerMoELayer, 8)
        << "A layer can retain ordinary, grouped-verifier, MTP-sidecar, and "
           "graph-topology descriptor identities at the same time";
    EXPECT_GE(kRuntimePointerTableSlots, kGroupedDescriptorTableSlots)
        << "Every persistent descriptor identity must have a matching runtime "
           "pointer-array identity";
    EXPECT_EQ(kRouterGateCacheSlots, kMaximumMoELayersPerDevice)
        << "Router weights are one immutable cache owner per model layer, not "
           "one copy per graph role";

    const std::size_t expected_descriptor_bytes =
        static_cast<std::size_t>(kGroupedDescriptorTableSlots) *
        static_cast<std::size_t>(kExperts) *
        sizeof(llaminar2::DeviceNativeVNNIMatrixDesc);
    const auto cuda_requirements = cudaMoE(
        /*max_seq_len=*/16,
        /*d_model=*/2048,
        /*intermediate=*/512,
        kExperts,
        /*top_k=*/8);
    const auto rocm_requirements = rocmMoE(
        /*max_seq_len=*/16,
        /*d_model=*/2048,
        /*intermediate=*/512,
        kExperts,
        /*top_k=*/8);
    for (const auto *buffer_name : {
             CUDA_GROUPED_GATE_DESC_TABLES,
             CUDA_GROUPED_UP_DESC_TABLES,
             CUDA_GROUPED_DOWN_DESC_TABLES})
    {
        const auto *buffer = cuda_requirements.find(buffer_name);
        ASSERT_NE(buffer, nullptr) << buffer_name;
        EXPECT_EQ(buffer->size_bytes, expected_descriptor_bytes) << buffer_name;
    }
    for (const auto *buffer_name : {
             ROCM_GROUPED_GATE_DESC_TABLES,
             ROCM_GROUPED_UP_DESC_TABLES,
             ROCM_GROUPED_DOWN_DESC_TABLES})
    {
        const auto *buffer = rocm_requirements.find(buffer_name);
        ASSERT_NE(buffer, nullptr) << buffer_name;
        EXPECT_EQ(buffer->size_bytes, expected_descriptor_bytes) << buffer_name;
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, RuntimePrefillDescriptorsUseRetainedGraphOwnerSlots)
{
    using namespace llaminar2::MoEWorkspaceBuffers;
    constexpr int kExperts = 256;
    const std::size_t expected_descriptor_bytes =
        static_cast<std::size_t>(kGroupedDescriptorTableSlots) *
        static_cast<std::size_t>(kExperts) *
        sizeof(llaminar2::DeviceNativeVNNIMatrixDesc);

    /*
     * Runtime descriptors are rewritten on device for each placement epoch.
     * They nevertheless remain captured-graph state because every graph records
     * the destination address.  The workspace must therefore provide the same
     * owner count and fixed stride as the immutable descriptor tables.
     */
    const auto cuda_requirements = cudaMoE(
        /*max_seq_len=*/16,
        /*d_model=*/2048,
        /*intermediate=*/512,
        kExperts,
        /*top_k=*/8);
    const auto rocm_requirements = rocmMoE(
        /*max_seq_len=*/16,
        /*d_model=*/2048,
        /*intermediate=*/512,
        kExperts,
        /*top_k=*/8);
    for (const auto *buffer_name : {
             CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE,
             CUDA_RUNTIME_PREFILL_UP_DESC_TABLE,
             CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE})
    {
        const auto *buffer = cuda_requirements.find(buffer_name);
        ASSERT_NE(buffer, nullptr) << buffer_name;
        EXPECT_EQ(buffer->size_bytes, expected_descriptor_bytes) << buffer_name;
    }
    for (const auto *buffer_name : {
             ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
             ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
             ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE})
    {
        const auto *buffer = rocm_requirements.find(buffer_name);
        ASSERT_NE(buffer, nullptr) << buffer_name;
        EXPECT_EQ(buffer->size_bytes, expected_descriptor_bytes) << buffer_name;
    }

    const auto root = repoRoot();
    const auto cuda_source = readFile(
        root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto cuda_header = readFile(
        root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.h");
    const auto rocm_source = readFile(
        root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp");
    const auto rocm_header = readFile(
        root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.h");

    const auto cuda_runtime_execution = sliceBetween(
        cuda_source,
        "bool CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime(",
        "bool CUDAMoEKernel::groupedExpertGateUpDecodeFromTable(");
    const auto rocm_runtime_execution = sliceBetween(
        rocm_source,
        "bool ROCmMoEKernel::executeGroupedPrefillPipelineFromRuntime(",
        "\n} // namespace llaminar2");

    for (const auto &[backend, execution] : {
             std::pair{"CUDA", &cuda_runtime_execution},
             std::pair{"ROCm", &rocm_runtime_execution}})
    {
        SCOPED_TRACE(backend);
        EXPECT_NE(execution->find("gateup_table.workspace_slot"), std::string::npos)
            << backend << " runtime gate/up descriptors must inherit the "
                          "immutable gate/up table's retained lease identity";
        EXPECT_NE(execution->find("down_table.workspace_slot"), std::string::npos)
            << backend << " runtime down descriptors must inherit the immutable "
                          "down table's independently retained lease identity";
        EXPECT_NE(execution->find("bindGroupedDescriptorTableSlot("), std::string::npos)
            << backend << " runtime descriptor addresses must use validated "
                          "fixed-stride slot resolution";
        EXPECT_EQ(execution->find("acquirePersistentSlot("), std::string::npos)
            << backend << " graph execution must not acquire leases or enter "
                          "the host lease-registry mutex";
    }

    for (const auto &[backend, source, header] : {
             std::tuple{"CUDA", &cuda_source, &cuda_header},
             std::tuple{"ROCm", &rocm_source, &rocm_header}})
    {
        SCOPED_TRACE(backend);
        EXPECT_EQ(source->find("d_runtime_prefill_"), std::string::npos)
            << backend << " must not retain a device-wide mutable runtime "
                          "descriptor singleton";
        EXPECT_EQ(header->find("d_runtime_prefill_"), std::string::npos)
            << backend << " runtime descriptor ownership belongs to graph-local "
                          "execution pointers, not kernel-object members";
        EXPECT_EQ(source->find("ensureRuntimePrefillDescriptorCapacity"), std::string::npos);
        EXPECT_EQ(header->find("ensureRuntimePrefillDescriptorCapacity"), std::string::npos);
    }
}

/**
 * @brief Guard graph-captured MoE route scratch against pointer replacement.
 *
 * Runtime grouped kernels capture direct scratch addresses. A later prefill
 * bucket used to grow the runtime table, free those addresses, and leave older
 * verifier graphs reading recycled memory. Production graph construction must
 * now allocate one largest-participant arena per device, share it only among
 * stream/event-ordered graph roles, and treat every later capacity request as
 * validation rather than resize.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     MoERouteScratchIsSharedPlannedAndImmutableBeforeCapture)
{
    const auto root = repoRoot();
    const auto graph_source =
        readFile(root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto graph_header =
        readFile(root / "src/v2/models/qwen35moe/Qwen35MoEGraph.h");
    const auto runtime_source =
        readFile(root / "src/v2/execution/moe/MoERuntimeTable.cpp");
    const auto runtime_header =
        readFile(root / "src/v2/execution/moe/MoERuntimeTable.h");

    const auto factory = sliceBetween(
        graph_source,
        "IMoERuntimeTable *Qwen35MoEGraph::moeRuntimeTableForDevice(",
        "void Qwen35MoEGraph::registerRuntimeTableHistogramSyncIfNeeded(");
    const auto ensure = sliceBetween(
        runtime_source,
        "void DeviceMoERuntimeTable::ensurePrefillRouteScratchCapacity(",
        "bool DeviceMoERuntimeTable::prepareInactiveBank(");

    EXPECT_NE(
        factory.find("resolveActivationBufferSeqLen(config_.max_seq_len, device)"),
        std::string::npos)
        << "ordinary prefill must contribute its largest graph bucket";
    EXPECT_NE(
        factory.find("resolveMTPMaxTargetQueryRows(config_.mtp)"),
        std::string::npos)
        << "grouped verifier and request-batched MTP rows must contribute their "
           "complete configured capacity";
    EXPECT_NE(
        factory.find("moe_serial_route_scratch_arenas_"),
        std::string::npos)
        << "main and MTP graph roles need one device-scoped serial scratch owner";
    EXPECT_NE(
        factory.find("table_config.serial_route_scratch_arena = scratch_it->second"),
        std::string::npos)
        << "every production runtime table must bind the shared immutable arena";
    EXPECT_NE(
        graph_header.find("std::shared_ptr<DeviceMoESerialRouteScratchArena>"),
        std::string::npos);

    EXPECT_NE(
        ensure.find("if (serial_route_scratch_arena_)"),
        std::string::npos);
    EXPECT_NE(
        ensure.find("captured device addresses are immutable"),
        std::string::npos)
        << "capacity overflow must fail with an explicit graph-lifetime error";
    const size_t immutable_branch =
        ensure.find("if (serial_route_scratch_arena_)");
    const size_t mutable_resize =
        ensure.find("prefill_route_scratch_.resize");
    ASSERT_NE(immutable_branch, std::string::npos);
    ASSERT_NE(mutable_resize, std::string::npos);
    EXPECT_LT(immutable_branch, mutable_resize)
        << "the immutable production branch must return or throw before any "
           "legacy setup-only resize path";
    EXPECT_NE(
        runtime_header.find("usesImmutableSerialRouteScratch"),
        std::string::npos)
        << "integration tests need a semantic ownership query, not pointer guesses";
    EXPECT_NE(
        runtime_source.find(
            "\"moe_serial_route_scratch_arena_allocations\""),
        std::string::npos)
        << "The VRAM BOM must expose the single per-device route allocation.";
    EXPECT_NE(
        runtime_source.find(
            "\"moe_serial_route_scratch_runtime_table_bindings\""),
        std::string::npos)
        << "PerfStats must prove multiple graph roles reuse the allocation.";
    EXPECT_NE(
        runtime_source.find(
            "{\"ownership\", \"per_device_serial_graph_domain\"}"),
        std::string::npos)
        << "Sharing is legal only for the explicit serial graph domain.";
}

/**
 * @brief Guard graph stream handoffs against transient allocation and ignored errors.
 *
 * CUDA and ROCm graphs cross from capture streams to explicit collective and
 * sidecar streams. Losing one ordering edge can let one participant consume
 * stale device state while its peer consumes newly published state. The
 * handoff primitive is therefore a checked correctness operation.
 *
 * This source regression verifies both halves of the contract: contexts
 * preallocate a reusable event pool during initialization, and owners of real
 * cross-stream boundaries branch on publication failure. Ordinary executor
 * stages retain their scheduler-bound stream and therefore must not manufacture
 * an unnecessary handoff edge. Reintroducing per-replay event allocation,
 * fire-and-forget handoffs, or executor-side stream adoption fails this
 * unit-only gate before a timing-sensitive multi-GPU E2E can regress.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, StreamDependenciesArePersistentAndFailClosed)
{
    const auto root = repoRoot();
    const auto interface_source =
        readFile(root / "src/v2/backends/IWorkerGPUContext.h");
    const auto cuda_header =
        readFile(root / "src/v2/backends/cuda/NvidiaDeviceContext.h");
    const auto rocm_header =
        readFile(root / "src/v2/backends/rocm/AMDDeviceContext.h");
    const auto cuda_source =
        readFile(root / "src/v2/backends/cuda/NvidiaDeviceContext.cu");
    const auto rocm_source =
        readFile(root / "src/v2/backends/rocm/AMDDeviceContext.cpp");
    const auto capture_source = readFile(
        root /
        "src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp");
    const auto executor_source = readFile(
        root /
        "src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp");
    const auto orchestrator_source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto cuda_handoff = stripCommentsAndStringLiterals(sliceBetween(
        cuda_source,
        "bool NvidiaDeviceContext::insertStreamDependency(",
        "std::unique_ptr<IGPUGraphCapture> NvidiaDeviceContext::createGraphCapture("));
    const auto rocm_handoff = stripCommentsAndStringLiterals(sliceBetween(
        rocm_source,
        "bool AMDDeviceContext::insertStreamDependency(",
        "std::unique_ptr<IGPUGraphCapture> AMDDeviceContext::createGraphCapture("));

    EXPECT_NE(interface_source.find("virtual bool insertStreamDependency("),
              std::string::npos)
        << "A stream-ordering failure must be observable by every caller.";
    EXPECT_NE(cuda_header.find("stream_dependency_events_"),
              std::string::npos);
    EXPECT_NE(rocm_header.find("stream_dependency_events_"),
              std::string::npos);
    EXPECT_NE(cuda_source.find(
                  "cudaEventCreateWithFlags(&event, cudaEventDisableTiming)"),
              std::string::npos);
    EXPECT_NE(rocm_source.find(
                  "hipEventCreateWithFlags(&event, hipEventDisableTiming)"),
              std::string::npos);

    EXPECT_EQ(cuda_handoff.find("cudaEventCreate"), std::string::npos)
        << "CUDA graph replay must lease a persistent event, not allocate one.";
    EXPECT_EQ(cuda_handoff.find("cudaEventDestroy"), std::string::npos)
        << "CUDA graph replay must not destroy event resources.";
    EXPECT_EQ(rocm_handoff.find("hipEventCreate"), std::string::npos)
        << "ROCm graph replay must lease a persistent event, not allocate one.";
    EXPECT_EQ(rocm_handoff.find("hipEventDestroy"), std::string::npos)
        << "ROCm graph replay must not destroy event resources.";

    const auto compact_capture = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(capture_source));
    const auto compact_executor = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(executor_source));
    const auto compact_orchestrator = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(orchestrator_source));
    EXPECT_NE(compact_capture.find("if(!gpu_ctx->insertStreamDependency("),
              std::string::npos)
        << "Capture/collective handoffs must stop replay on publication failure.";
    EXPECT_EQ(compact_executor.find("insertStreamDependency("),
              std::string::npos)
        << "Executor stages retain their explicit scheduler binding; they must "
           "not invent a second producer/consumer handoff.";
    EXPECT_NE(compact_executor.find("hasGPUStream()"),
              std::string::npos)
        << "Executor binding decisions must use the nonthrowing typed query.";
    EXPECT_NE(
        compact_orchestrator.find(
            "if(!GPUDeviceContextPool::instance().getContext(state_.device_id).insertStreamDependency("),
        std::string::npos)
        << "MTP sidecar handoffs must stop on publication failure.";
}

/**
 * @brief Forbid using a stream handle as a nullable execution-policy signal.
 *
 * A GPU stage has exactly two legal questions:
 * - planning code asks hasGPUStream() whether the scheduler has bound it;
 * - execution code calls requireGPUStream() and fails fatally if it has not.
 *
 * Treating gpuStream() as a Boolean used to turn missing producer identity into
 * a recoverable false return and invited default-stream fallbacks. The API now
 * throws for unbound GPU stages; this scan keeps call sites aligned with that
 * contract and makes the ordering mistake visible in the unit gate.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, GPUStreamPresenceUsesTypedStageContract)
{
    const auto source_root = repoRoot() / "src/v2";
    const std::vector<std::string> forbidden = {
        "!gpuStream()",
        "gpuStream()!=nullptr",
        "gpuStream()==nullptr",
        "if(gpuStream())",
        "while(gpuStream())",
        "gpuStream()?",
        "gpuStream()&&",
        "&&gpuStream()",
        "gpuStream()||",
        "||gpuStream()",
    };

    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(source_root))
    {
        if (!entry.is_regular_file())
            continue;
        const auto extension = entry.path().extension().string();
        if (extension != ".cpp" && extension != ".h" &&
            extension != ".cu" && extension != ".hip")
        {
            continue;
        }

        const auto compact = removeAsciiWhitespace(
            stripCommentsAndStringLiterals(readFile(entry.path())));
        for (const auto &pattern : forbidden)
        {
            EXPECT_EQ(compact.find(pattern), std::string::npos)
                << entry.path().lexically_relative(repoRoot())
                << " uses gpuStream() as a nullable probe; use hasGPUStream() "
                   "during planning or requireGPUStream() during execution";
        }
    }
}

/**
 * @brief Guard explicit forward-role ownership of mirrored MTP diagnostics.
 *
 * A request-batched MTP condition graph and a grouped verifier can have the
 * same total row count. Shape-based labels therefore cannot identify which
 * graph produced the terminal hidden state consumed by the MTP head. This
 * source regression keeps the role typed at every forwardImpl call and ensures
 * grouped verifier graphs retain and publish their own diagnostic provenance:
 * accepted-state publication can promote one of their hidden rows into the
 * terminal mailbox.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, MirroredMTPDiagnosticsUseTypedForwardRoles)
{
    const auto root = repoRoot();
    const auto header = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto orchestrator = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto graph = readFile(
        root / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto graph_types = readFile(
        root / "src/v2/models/GraphTypes.h");
    const auto graph_interface = readFile(
        root /
        "src/v2/execution/local_execution/graph/IGraphBuilder.h");
    const auto runner_interface = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto forward_engine = readFile(
        root /
        "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp");
    const auto qwen_base = readFile(
        root / "src/v2/models/qwen/QwenGraphBase.h");
    const auto qwen_base_source = readFile(
        root / "src/v2/models/qwen/QwenGraphBase.cpp");

    EXPECT_NE(graph_interface.find("enum class ForwardExecutionRole"),
              std::string::npos);
    EXPECT_NE(graph_interface.find("GroupedMTPVerifier"),
              std::string::npos);
    EXPECT_NE(graph_interface.find("MTPCondition"),
              std::string::npos);
    EXPECT_NE(graph_interface.find("ForwardExecutionRole execution_role"),
              std::string::npos)
        << "The typed role must travel through both forward input and execution provenance.";
    const auto compact_runner_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(runner_interface));
    EXPECT_EQ(
        compact_runner_interface.find(
            "enumclassDeviceTokenForwardPurpose:uint8_t"),
        std::string::npos)
        << "A runtime purpose selector would reopen the token-only main-condition path.";
    EXPECT_NE(
        compact_runner_interface.find(
            "forwardGroupedMTPVerifierWithDeviceTokenIds("
            "constint*token_shadow,constvoid*token_ids_device,intseq_len)"),
        std::string::npos)
        << "Device-token-only forwarding must be verifier-specific at compile time.";
    EXPECT_EQ(
        compact_runner_interface.find(
            "forwardWithDeviceTokenIds("
            "constint*token_shadow,constvoid*token_ids_device,intseq_len)"),
        std::string::npos)
        << "The ambiguous device-token API must stay retired.";
    EXPECT_NE(
        compact_runner_interface.find(
            "advanceMTPMainConditionFromDeviceResidentLogicalState("
            "int32_ttoken_shadow,"
            "constDeviceResidentLogicalSequenceStateHandle&logical_state,"
            "intrequest_index=0)"),
        std::string::npos)
        << "GPU MTP condition replay must carry token and position ownership in "
           "one typed logical-state handle.";
    EXPECT_NE(
        compact_runner_interface.find(
            "advanceMTPMainConditionFromDeviceTargetSample("
            "int32_ttoken_shadow,inttarget_sample_slot)"),
        std::string::npos)
        << "A device target token must be composed with its live position before "
           "main-graph replay.";
    const auto executable_forward_engine =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(forward_engine));
    EXPECT_EQ(
        countOccurrences(
            executable_forward_engine,
            "host.prepareDeviceTokenInputsForForwardGraphExecution("),
        2u)
        << "Both cached replay and cache-miss execution must publish staged "
           "device-token rows on the exact graph stream.";
    EXPECT_NE(graph_types.find("bool grouped_mtp_verifier = false;"),
              std::string::npos);
    EXPECT_NE(graph_interface.find(
                  "virtual bool setGroupedMTPVerifier(bool enabled)"),
              std::string::npos);
    EXPECT_NE(qwen_base.find(
                  "bool setGroupedMTPVerifier(bool enabled) override"),
              std::string::npos);

    const auto forward_impl = sliceBetween(
        orchestrator,
        "const float *DeviceGraphOrchestrator::forwardImpl(",
        "bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule(");
    const auto compact_forward_impl =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(forward_impl));
    EXPECT_NE(
        forward_impl.find(
            "? \"grouped_verifier\""),
        std::string::npos)
        << "Grouped verifiers must select their own checkpoint bank after a successful forward.";
    EXPECT_NE(
        forward_impl.find("\"request_batch_decode\""),
        std::string::npos)
        << "Device-resident request-batch condition graphs require their own bank.";
    EXPECT_NE(
        forward_impl.find(
            "builder->setGroupedMTPVerifier(true)"),
        std::string::npos)
        << "Verifier ownership must be published to declarative graph builders.";
    EXPECT_NE(
        forward_impl.find(
            "builder->setGroupedMTPVerifier(false)"),
        std::string::npos)
        << "Verifier graph policy must be scoped to one execution.";
    EXPECT_NE(
        forward_impl.find("input.execution_role = execution_role"),
        std::string::npos)
        << "The orchestrator must publish its typed role into execution provenance.";
    EXPECT_NE(
        compact_forward_impl.find(
            "new_phase==InferencePhase::PREFILL&&"
            "execution_role==ForwardExecutionRole::MainInference&&"
            "!populateMTPShiftedCacheFromPrefill("),
        std::string::npos)
        << "Only a real main-inference prefill may populate shifted MTP state; "
           "a grouped verifier can share its shape and all-position output policy.";

    const auto single_verifier_forward = sliceBetween(
        orchestrator,
        "bool DeviceGraphOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds(",
        "advanceMTPMainConditionFromDeviceResidentLogicalState(");
    EXPECT_EQ(
        single_verifier_forward.find(
            "MTPCondition"),
        std::string::npos)
        << "A token-only verifier API must have no main-condition branch.";
    EXPECT_NE(
        single_verifier_forward.find(
            "ForwardExecutionRole::GroupedMTPVerifier"),
        std::string::npos)
        << "The device-token entry point must map grouped verifier rows explicitly.";

    const auto batched_verifier_forward = sliceBetween(
        orchestrator,
        "bool DeviceGraphOrchestrator::forwardBatchWithDeviceTokenIds(",
        "const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDevice(");
    EXPECT_NE(
        batched_verifier_forward.find(
            "ForwardExecutionRole::GroupedMTPVerifier"),
        std::string::npos)
        << "Request-batched device-token verifier forwards must declare their role.";
    EXPECT_NE(
        batched_verifier_forward.find(
            "actual_lengths"),
        std::string::npos)
        << "A ragged verifier batch must carry immutable logical row geometry "
           "instead of borrowing mutable runner progress.";

    const auto request_batch_forward = sliceBetween(
        orchestrator,
        "bool DeviceGraphOrchestrator::advanceMTPRequestBatchConditionOnDevice(",
        "bool DeviceGraphOrchestrator::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(");
    EXPECT_NE(
        request_batch_forward.find(
            "ForwardExecutionRole::MTPCondition"),
        std::string::npos)
        << "The request-batch condition call site must explicitly declare ownership.";

    EXPECT_NE(
        graph.find(
            "? \"grouped_verifier\""),
        std::string::npos)
        << "Qwen MoE graphs must retain a distinct grouped-verifier checkpoint bank.";
    EXPECT_NE(
        graph.find("? \"request_batch_decode\""),
        std::string::npos)
        << "Checkpoint names must distinguish request-batch decode from prefill.";
    EXPECT_NE(
        qwen_base_source.find(
            "\"final_norm_input\""),
        std::string::npos)
        << "Final-norm input must be an explicit diagnostic boundary.";
    EXPECT_NE(
        qwen_base_source.find(
            "\"final_norm_output\""),
        std::string::npos)
        << "Final-norm output must be an explicit diagnostic boundary.";
    EXPECT_NE(
        graph.find(
            "Qwen35MoEGraph::maybeAddFinalNormDiagnosticCheckpoint("),
        std::string::npos)
        << "Qwen MoE must retain final-norm checkpoints in graph-owned storage.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CachedGraphReplayFailureCannotRetryThroughEagerDecode)
{
    const auto root = repoRoot();
    const auto source = readFile(
        root /
        "src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp");
    const auto policy_body = sliceBetween(
        source,
        "bool DeviceGraphExecutor::executeDecodeWithCapturePolicy(",
        "// =========================================================================\n"
        "    // Cached GPU Graph Replay");
    const auto mandatory_replay_body = sliceBetween(
        policy_body,
        "if (policy.allow_cached_graph_replay)",
        "if (!bindGraphStagesForEagerExecution");
    const auto cached_replay_body = sliceBetween(
        source,
        "bool DeviceGraphExecutor::executeWithCachedGraphReplay(",
        "\n} // namespace llaminar2");

    EXPECT_NE(
        mandatory_replay_body.find(
            "failed under mandatory cached graph replay policy"),
        std::string::npos)
        << "Selecting cached replay must establish a fail-closed execution contract";
    EXPECT_EQ(
        stripCommentsAndStringLiterals(mandatory_replay_body)
            .find("executeFastDecode("),
        std::string::npos)
        << "A selected replay policy must never retry the request eagerly";
    EXPECT_EQ(
        stripCommentsAndStringLiterals(cached_replay_body)
            .find("executeFastDecode("),
        std::string::npos)
        << "Capture and replay failures must propagate to the request boundary";
    EXPECT_EQ(
        source.find("bindGraphStagesForEagerFallback"),
        std::string::npos)
        << "Eager execution is an explicit policy, not an error fallback";
}

TEST(Test__GpuWorkspaceAllocationPolicy, ROCmMoEWorkspaceOwnsLiveRoutingStateAndMetadataCaches)
{
    const int max_seq_len = 9;
    const int d_model = 2048;
    const int intermediate = 512;
    const int num_experts = 256;
    const int top_k = 8;

    const auto reqs = llaminar2::MoEWorkspaceBuffers::rocmMoE(
        max_seq_len, d_model, intermediate, num_experts, top_k);
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

/**
 * @brief Prevent diagnostics from selecting a different decode execution engine.
 *
 * GPU profiling used to disable cached graph replay so the eager stage timeline
 * could surround every stage. That made measurements unrepresentative and also
 * violated graph-owned MTP publication contracts. Whole-graph GPU events now
 * provide trustworthy production-path timing, so capture admission must remain
 * entirely independent of both unified and executor profiling switches.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, DecodeCapturePolicyCannotDependOnProfilingFlags)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto policy_source = sliceBetween(
        source,
        "DeviceGraphExecutor::DecodeCapturePolicy DeviceGraphOrchestrator::buildDecodeCapturePolicy(",
        "bool DeviceGraphOrchestrator::hasHeterogeneousCollectiveExecutionDomain()");
    const auto executable_policy = stripCommentsAndStringLiterals(policy_source);

    EXPECT_EQ(executable_policy.find("executor_profiling"), std::string::npos)
        << "Executor diagnostics must never disable or weaken production graph capture.";
    EXPECT_EQ(executable_policy.find("LLAMINAR_PROFILING"), std::string::npos)
        << "The deprecated unified profiling alias belongs in PerfStats, not capture policy.";
    EXPECT_NE(executable_policy.find("allow_cached_graph_replay"), std::string::npos)
        << "The guard must continue to inspect the production replay admission policy.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GraphCaptureControllerLimitsStreamSyncToDiagnostics)
{
    const auto source = readFile(repoRoot() / "src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp");
    const auto replay_phase = sliceBetween(
        source,
        "DeviceGraphCaptureController::ReplayPhaseResult DeviceGraphCaptureController::executeReplayPhase(",
        "bool DeviceGraphCaptureController::cohereReplaySegmentInputs(");
    const auto compact_replay_phase =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(replay_phase));

    EXPECT_EQ(hasUncheckedSynchronizeStreamCall(source), false)
        << "Reviewed graph diagnostics must use the checked synchronization API "
           "so asynchronous GPU failures are attributed to the operation that "
           "surfaced them.";
    EXPECT_NE(source.find("synchronizeStreamChecked("), std::string::npos);
    EXPECT_NE(source.find("const bool needs_segment_sync = verify_mode;"), std::string::npos)
        << "Intermediate replay waits are permitted only for explicit graph verification.";
    EXPECT_NE(source.find("Stream-only replay sync failed"), std::string::npos)
        << "The opt-in stream-only diagnostic may retain its terminal observation wait.";
    EXPECT_EQ(source.find("Initial captured launch stream sync failed"), std::string::npos)
        << "Initial graph capture must publish ordering without a blocking stream drain.";
    EXPECT_EQ(source.find("Re-capture stream sync failed"), std::string::npos)
        << "Graph recapture must flow into the final exact event fence instead of "
           "blocking after each rebuilt segment.";
    EXPECT_EQ(source.find("synchronizeEvent("), std::string::npos)
        << "GPU replay timing must never turn an event into a per-launch host fence.";
    EXPECT_EQ(compact_replay_phase.find("createEvent("), std::string::npos)
        << "Every replay timing event must be preallocated during warmup.";
    EXPECT_EQ(compact_replay_phase.find("destroyEvent("), std::string::npos)
        << "Replay must retain its fixed event ring instead of churning backend resources.";
    EXPECT_NE(source.find("queryEventChecked("), std::string::npos)
        << "Completed replay timing intervals must be reclaimed with nonblocking event queries.";
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

TEST(Test__GpuWorkspaceAllocationPolicy,
     BackendEventWaitsCannotDependOnAmbientGpuOrdinal)
{
    const auto cuda_source =
        readFile(repoRoot() / "src/v2/backends/cuda/CUDABackend.cu");
    const auto rocm_source =
        readFile(repoRoot() / "src/v2/backends/rocm/ROCmBackend.cpp");
    const auto cuda_wait = sliceBetween(
        cuda_source,
        "bool CUDABackend::streamWaitEvent(",
        "// ====================================================================\n    // Async H2D Without Sync");
    const auto rocm_wait = sliceBetween(
        rocm_source,
        "bool ROCmBackend::streamWaitEvent(",
        "// ====================================================================\n    // Async H2D Without Sync");

    EXPECT_NE(cuda_wait.find("cudaSetDevice(device_id)"), std::string::npos)
        << "CUDA event waits must select the stream/event owner's explicit ordinal.";
    EXPECT_NE(rocm_wait.find("hipSetDevice(device_id)"), std::string::npos)
        << "ROCm event waits must select the stream/event owner's explicit ordinal.";
    EXPECT_EQ(cuda_wait.find("(void)device_id"), std::string::npos);
    EXPECT_EQ(rocm_wait.find("(void)device_id"), std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoERebalanceMaintenanceAlwaysPublishesDeviceEvent)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto fn = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph(",
        "// =====================================================================\n    // IForwardExecutionHost interface implementations");
    const auto executable_fn = stripCommentsAndStringLiterals(fn);
    const auto wait_fn = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingDeviceMoERebalanceMaintenance(",
        "bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph(");
    const auto executable_wait_fn = stripCommentsAndStringLiterals(wait_fn);
    const auto materialize_fn = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::\n"
        "        materializeDeviceMoERebalanceMaintenanceGraphForFamily()",
        "DeviceMoERebalanceGraphControllerState *\n"
        "    DeviceGraphOrchestrator::deviceMoERebalanceControllerStateDevice()");
    const auto executable_materialize_fn =
        stripCommentsAndStringLiterals(materialize_fn);

    EXPECT_NE(fn.find("Publish every maintenance wave through its recorded device event"),
              std::string::npos)
        << "Maintenance publication must document the common device-event contract.";
    EXPECT_NE(fn.find("collective_lane_policy"), std::string::npos)
        << "PerfStats must publish the fixed shared-domain collective policy.";
    EXPECT_NE(fn.find("shared_domain_event_ordered"), std::string::npos);
    EXPECT_EQ(fn.find("dedicated_collective_lane"), std::string::npos)
        << "A dedicated maintenance communicator is no longer a production policy.";
    EXPECT_NE(fn.find("completion_handoff"), std::string::npos)
        << "PerfStats must identify the device-event publication contract.";
    EXPECT_EQ(executable_fn.find("gpu_ctx->synchronizeStreamChecked(maintenance_stream)"),
              std::string::npos)
        << "No maintenance lane may block the host after graph replay.";
    EXPECT_EQ(fn.find("device_maintenance_graph_same_communicator_sync"), std::string::npos)
        << "The obsolete host-side shared-communicator timer must not return.";
    EXPECT_NE(
        executable_materialize_fn.find(
            "stage->getParams().tp_ctx != domain_collective_ctx"),
        std::string::npos)
        << "Graph-family construction must reject maintenance collectives on a different communicator.";
    EXPECT_NE(
        materialize_fn.find("construction failure before capture or GPU submission"),
        std::string::npos)
        << "The single-communicator invariant must be checked before device work is submitted.";
    EXPECT_NE(executable_fn.find("active_cache.completion_event_in_flight = true;"),
              std::string::npos)
        << "Every launched maintenance wave must remain visible to the next graph consumer.";
    EXPECT_EQ(header.find("completion_event_consumer_wait_pending"),
              std::string::npos)
        << "A shared consumable flag cannot encode multi-stream publication fan-out.";
    EXPECT_NE(
        executable_wait_fn.find(
            "DeviceTimelinePoint::MoERebalanceMaintenanceReady"),
        std::string::npos)
        << "The consumer obligation must use the centralized typed timeline point.";
    EXPECT_NE(
        executable_wait_fn.find(".to(consumer_role)"),
        std::string::npos)
        << "Each independent graph consumer must declare its semantic role.";
    EXPECT_NE(
        executable_wait_fn.find(".enqueuePublishedWait("),
        std::string::npos)
        << "The durable publication must lower to a device-side event wait.";
    EXPECT_EQ(
        executable_wait_fn.find("synchronizeStreamChecked("),
        std::string::npos)
        << "The production event handoff must never block the host.";
    expectNeedleBefore(
        executable_fn,
        "gpu_ctx->recordEventChecked(active_cache.completion_event.get(), maintenance_stream)",
        "active_cache.completion_event_in_flight = true;",
        "The completion event must be recorded before it is published to graph consumers.");
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

TEST(Test__GpuWorkspaceAllocationPolicy, MainLogitsDeviceConsumersJoinDurableForwardEvent)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto helper = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "void *DeviceGraphOrchestrator::prepareMainLogitsDeviceConsumer(",
        "bool DeviceGraphOrchestrator::waitForPendingLogitsStream(")));
    const auto greedy = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "int DeviceGraphOrchestrator::sampleGreedyOnDevice()",
        "int DeviceGraphOrchestrator::sampleOnDevice(")));
    const auto stochastic = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "int DeviceGraphOrchestrator::sampleOnDeviceAtLogicalPosition(",
        "bool DeviceGraphOrchestrator::requiresMPICoordinatedDecodeSampling(")));
    const auto device_target = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::sampleGreedyFromMainLogitsToDeviceTargetSlot(",
        "int DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDevice(")));
    const auto batched = removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::sampleMainLogitsBatchRowsOnDevice(",
        "bool DeviceGraphOrchestrator::applyPenaltiesOnDevice(")));

    EXPECT_NE(
        helper.find(
            "waitForForwardGraphOutputReady(stream,DeviceTimelineRole::TargetSampler,require_main_forward?ForwardGraphOutputKind::MainForward:ForwardGraphOutputKind::Any,consumer_name)"),
        std::string::npos)
        << "The common main-logits consumer boundary must join the durable "
           "forward event on its exact sampler stream.";
    EXPECT_EQ(helper.find("synchronizeStream"), std::string::npos);
    EXPECT_EQ(helper.find("synchronizeDevice"), std::string::npos);

    for (const auto *body : {&greedy, &stochastic, &device_target})
    {
        EXPECT_NE(
            body->find("prepareMainLogitsDeviceConsumer("),
            std::string::npos)
            << "Every terminal main-logits sampler must use the typed durable rendezvous.";
        EXPECT_EQ(
            body->find(
                "consumePendingLogitsStream(PendingLogitsStreamRole::MainDecode"),
            std::string::npos)
            << "A sampler must not bypass the durable rendezvous by consuming "
               "the optional raw-stream handoff directly.";
    }

    EXPECT_NE(
        batched.find(
            "waitForForwardGraphOutputReady(stream,DeviceTimelineRole::TargetSampler,ForwardGraphOutputKind::Any,"),
        std::string::npos)
        << "Request-batched main logits may occupy either graph-declared "
           "surface, but still require the durable producer event.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPSidecarGraphCaptureInstallsLocalTPBoundaryHook)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    const auto collective_scan = sliceBetween(
        sidecar_body,
        "sidecar_cache.collective_nodes.clear();",
        "sidecar_cache.valid = true;");
    EXPECT_NE(
        collective_scan.find("node->stage->isCollectiveStage()"),
        std::string::npos)
        << "MTP sidecar discovery must consume the canonical stage-level "
           "collective contract so TP KV/GDN handoffs cannot be omitted.";
    EXPECT_EQ(
        collective_scan.find("ComputeStageType::ALLREDUCE"),
        std::string::npos)
        << "A sidecar-local enum list can silently force specialized "
           "collectives out of whole-graph capture.";

    const size_t policy_build = sidecar_body.find("auto capture_policy = buildDecodeCapturePolicy(");
    ASSERT_NE(policy_build, std::string::npos)
        << "MTP sidecar execution must continue to build a decode capture policy.";
    const size_t boundary_hook =
        sidecar_body.find("capture_policy.capture_boundary", policy_build);
    ASSERT_NE(boundary_hook, std::string::npos)
        << "MTP sidecar graph capture must join the same LocalTP capture lifecycle as main decode.";
    const size_t boundary_call =
        sidecar_body.find("waitAtDecodeGraphCaptureBoundary(", boundary_hook);
    ASSERT_NE(boundary_call, std::string::npos)
        << "The sidecar boundary hook must route through DeviceGraphOrchestrator so participant indices are validated.";
    const size_t execute_call =
        sidecar_body.find("executor_.executeDecodeWithCapturePolicy(", policy_build);
    ASSERT_NE(execute_call, std::string::npos)
        << "MTP sidecar execution must still use the centralized capture policy path.";

    EXPECT_LT(policy_build, boundary_hook)
        << "The sidecar capture policy should be built before installing the lifecycle hook.";
    EXPECT_LT(boundary_hook, execute_call)
        << "LocalTP participants must install capture entry/exit rendezvous before sidecar execution.";
    EXPECT_LT(boundary_call, execute_call)
        << "The sidecar hook must be installed before entering the executor capture path.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPFirstUseCaptureAndParentPublicationAreAtomic)
{
    const auto root = repoRoot();
    const auto executor_source = readFile(
        root / "src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp");
    const auto orchestrator_source = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto first_use = sliceBetween(
        executor_source,
        "if (phase_transition.phase == DeviceGraphCaptureController::Phase::Warmup)",
        "// ===== Invalid externally visible capture state =====");
    expectNeedleBefore(
        first_use,
        "runStages(",
        "graph.reset();",
        "First use must execute the logical payload before recording its reusable graph.");
    expectNeedleBefore(
        first_use,
        "graph.reset();",
        "DeviceGraphCaptureController::executeCapturePhase(",
        "Capture recording must be part of the same first-use transaction.");
    expectNeedleBefore(
        first_use,
        "MaterializeOnly",
        "segment_cache.needs_capture = false;",
        "First-use capture must suppress a second payload execution and publish replay readiness atomically.");
    const auto partial_state = sliceBetween(
        executor_source,
        "if (phase_transition.phase == DeviceGraphCaptureController::Phase::Capture)",
        "// Replay is handled by the fast path above;");
    EXPECT_EQ(
        partial_state.find("executeCapturePhase("),
        std::string::npos)
        << "A leaked partial cache state must fail hard, never finish capture as a fallback.";
    EXPECT_NE(
        partial_state.find("first-use materialization is atomic"),
        std::string::npos);

    const auto sidecar = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    EXPECT_EQ(
        sidecar.find("try_gpu_graph_capture && !rebuilt_graph"),
        std::string::npos)
        << "A freshly built GPU sidecar must capture during its first invocation.";
    EXPECT_EQ(sidecar.find("plain_after_build"), std::string::npos);
    EXPECT_NE(
        sidecar.find("eager execution is forbidden"),
        std::string::npos)
        << "GPU sidecars must fail closed when graph capture is unavailable.";
    EXPECT_NE(
        sidecar.find("sidecar_dynamic_stream,"),
        std::string::npos)
        << "Sidecar execution must use the exact capture-owned producer stream.";

    const auto publication = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    EXPECT_NE(
        publication.find("finalizeMTPSpeculativeStatePublicationLaunch("),
        std::string::npos);
    EXPECT_EQ(
        publication.find("materializeMTPDeviceGenerationLoopGraph("),
        std::string::npos)
        << "The exclusive logical-state writer must not also own fallible parent-graph composition.";
}

/**
 * @brief Keep collectives embedded in composite stages visible to graph policy.
 *
 * Most collectives have a dedicated stage type, but full LLEP prefill and MoE
 * maintenance intentionally compose several kernels and collectives inside one
 * stage. Their instance-level predicates are required so whole-graph capture
 * does not depend on another incomplete orchestration-side type list.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, CompositeMoEStagesPublishConditionalCollectiveContracts)
{
    const auto root = repoRoot();
    const auto expert_header = readFile(
        root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h");
    const auto expert_source = readFile(
        root / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp");
    const auto rebalance_header = readFile(
        root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.h");
    const auto rebalance_source = readFile(
        root / "src/v2/execution/compute_stages/stages/MoEDeviceRebalanceStage.cpp");

    EXPECT_NE(
        expert_header.find("bool isCollectiveStage() const override;"),
        std::string::npos);
    const auto expert_contract = sliceBetween(
        expert_source,
        "bool MoEExpertComputeStage::isCollectiveStage() const",
        "bool MoEExpertComputeStage::isGraphCapturable() const");
    EXPECT_NE(
        expert_contract.find(
            "requestsTransferBackedCurrentBatchPrefillLLEP()"),
        std::string::npos)
        << "The typed graph-build request must classify transfer-backed LLEP "
           "as collective before runtime binding validation.";

    EXPECT_NE(
        rebalance_header.find("bool isCollectiveStage() const override;"),
        std::string::npos);
    const auto rebalance_contract = sliceBetween(
        rebalance_source,
        "bool MoEDeviceRebalanceStage::isCollectiveStage() const",
        "bool MoEDeviceRebalanceStage::isGraphCapturable() const");
    EXPECT_NE(
        rebalance_contract.find("gathersStateInline()"),
        std::string::npos);
    EXPECT_NE(
        rebalance_contract.find("runsController() && usesTransferSlotApply()"),
        std::string::npos);
    EXPECT_EQ(
        rebalance_contract.find("TransferPreparedPayload"),
        std::string::npos)
        << "The retired host-selected transport graph must not remain a collective owner.";
}

/**
 * @brief Prevent auxiliary maintenance capture from hiding its collectives.
 *
 * The ordinary forward engine and the MoE maintenance scheduler used to build
 * independent collective classifications. The maintenance scheduler then
 * hardcoded `has_collective_nodes=false`, which disabled graph capture for two
 * otherwise capture-ready phase-split rebalance stages. Keep graph discovery,
 * cached ownership, capture policy, and executor classification tied to the
 * same instance-level node set.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, MoEMaintenanceCaptureUsesGraphOwnedCollectiveDiscovery)
{
    const auto root = repoRoot();
    const auto graph_header = readFile(
        root / "src/v2/execution/local_execution/graph/ComputeGraph.h");
    const auto graph_source = readFile(
        root / "src/v2/execution/local_execution/graph/ComputeGraph.cpp");
    const auto engine_source = readFile(
        root / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp");
    const auto orchestrator_header = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto orchestrator_source = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    EXPECT_NE(
        graph_header.find(
            "std::unordered_set<std::string> collectiveNodeNames() const;"),
        std::string::npos);
    EXPECT_NE(
        graph_source.find(
            "ComputeGraph::collectiveNodeNames() const"),
        std::string::npos);
    EXPECT_NE(
        graph_source.find(
            "node->stage->isCollectiveStage()"),
        std::string::npos);
    EXPECT_NE(
        engine_source.find(
            "auto collective_nodes = graph.collectiveNodeNames();"),
        std::string::npos);
    EXPECT_EQ(
        engine_source.find("collectCollectiveNodeNames"),
        std::string::npos)
        << "Forward execution must not grow a second collective classifier.";

    EXPECT_NE(
        orchestrator_header.find(
            "std::unordered_set<std::string> collective_nodes;"),
        std::string::npos);
    const auto maintenance_materialization = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::\n"
        "        materializeDeviceMoERebalanceMaintenanceGraphForFamily()",
        "bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph()");
    EXPECT_NE(
        maintenance_materialization.find(
            "cache.graph->collectiveNodeNames()"),
        std::string::npos)
        << "Collective ownership must be discovered once from the eagerly "
           "materialized graph family, before any maintenance launch.";

    const auto maintenance_launch = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph()",
        "// =====================================================================\n"
        "    // IForwardExecutionHost interface implementations");
    EXPECT_EQ(
        maintenance_launch.find("collectiveNodeNames()"),
        std::string::npos)
        << "The hot launch path must consume the graph-family declaration, "
           "not rediscover collective ownership.";
    EXPECT_NE(
        maintenance_launch.find(
            "!active_cache.collective_nodes.empty()"),
        std::string::npos);
    EXPECT_NE(
        maintenance_launch.find(
            "&active_cache.collective_nodes"),
        std::string::npos);
    EXPECT_EQ(
        maintenance_launch.find("/*has_collective_nodes=*/false"),
        std::string::npos)
        << "A phase-split maintenance graph must never suppress its embedded "
           "NCCL/RCCL stages when selecting capture policy.";
}

/**
 * @brief Locks MTP metadata planning to complete verifier-row capacity.
 *
 * The maximum MTP draft depth does not include the target verifier's bonus row.
 * Dynamic depth fifteen therefore materializes a sixteen-row graph. Declaring
 * only the draft depth leaves the strict serial-family preflight four bytes
 * short when that graph first publishes accepted-state indices.
 */
TEST(
    Test__GpuWorkspaceAllocationPolicy,
    MTPMetadataFamilyDeclarationIncludesBonusVerifierRow)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    EXPECT_NE(
        source.find(
            "metadata_shape.max_draft_tokens,\n"
            "                std::max(1, mtp_max_verifier_rows_)"),
        std::string::npos)
        << "The eager metadata participant must own every configured verifier "
           "row, including the all-drafts-accepted bonus row.";
    EXPECT_EQ(
        source.find(
            "metadata_shape.max_draft_tokens,\n"
            "                std::max(1, mtp_max_draft_depth_)"),
        std::string::npos)
        << "Draft-only capacity recreates the dynamic-depth-15 64-byte versus "
           "68-byte state-index mismatch.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPShiftedKVAsyncHandoffUsesEventBeforeConsumers)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto runner_interface =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto publication_stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.cpp");
    const auto transaction_record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordDeviceResidentMTPTransactionMutation(",
        "bool DeviceGraphOrchestrator::recordRestoredDeviceResidentMTPTransaction(");
    const auto metadata_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto publication_lifecycle_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::finalizeMTPSpeculativeStatePublicationLaunch(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");

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
    const size_t publication_launch =
        publish_body.find("executeMTPSpeculativeStatePublicationCaptured");
    ASSERT_NE(publish_wait, std::string::npos);
    ASSERT_NE(publication_launch, std::string::npos);
    EXPECT_LT(publish_wait, publication_launch)
        << "Accepted-state publication truncates MTP KV and must wait for deferred shifted appends first.";
    EXPECT_NE(
        publication_stage_source.find(
            "publishSequenceStateFromDeviceMetadata("),
        std::string::npos)
        << "The captured publication stage must own primary and shifted KV mutation.";
    const size_t terminal_hidden_publish =
        publish_body.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState");
    const size_t lifecycle_finalize =
        publish_body.find("finalizeMTPSpeculativeStatePublicationLaunch");
    const size_t ready_event =
        publication_lifecycle_body.find("recordAcceptedSpecPublicationReady");
    const size_t live_epoch_advance =
        publication_lifecycle_body.find("handleLivePrefixReplayStateAfterMutation");
    const size_t logical_mailbox_publish =
        publication_lifecycle_body.find("recordDeviceResidentLogicalSequenceStateMailbox");
    ASSERT_NE(terminal_hidden_publish, std::string::npos);
    ASSERT_NE(lifecycle_finalize, std::string::npos);
    ASSERT_NE(ready_event, std::string::npos);
    ASSERT_NE(live_epoch_advance, std::string::npos);
    ASSERT_NE(logical_mailbox_publish, std::string::npos);
    EXPECT_LT(terminal_hidden_publish, lifecycle_finalize)
        << "Publication readiness must cover the accepted verifier terminal hidden rows.";
    EXPECT_LT(live_epoch_advance, logical_mailbox_publish)
        << "Device-outcome publication must first advance the live-state epoch, "
           "then create a fresh arena-backed logical-state mailbox for that "
           "epoch. Retargeting a pre-publication mailbox would preserve stale "
           "ownership and add an unnecessary readiness event.";
    EXPECT_NE(publish_body.find("spec_state_terminal_hidden_publications"), std::string::npos)
        << "Terminal-hidden publication should be visible in perf stats.";
    EXPECT_NE(publication_lifecycle_body.find("recordDeviceResidentMTPTransactionMutation("),
              std::string::npos)
        << "Direct publication must advance the same request transaction after "
           "mutating canonical shifted-cache counts.";
    EXPECT_EQ(
        publish_body.find("prepareLivePrefixMutationReadyEvent("),
        std::string::npos)
        << "Accepted-state publication owns AcceptedSpecPublicationReady; it "
           "must not reserve the independent live-prefix mutation channel.";
    EXPECT_EQ(
        publish_body.find("recordLivePrefixMutationReady("),
        std::string::npos)
        << "Accepted-state publication must publish exactly one typed lifecycle "
           "edge instead of impersonating a prefix restore.";
    EXPECT_NE(metadata_body.find("request.outcome.mtp_transaction"),
              std::string::npos)
        << "Publication preflight must consume the child-local outcome lease.";
    expectNeedleBefore(
        metadata_body,
        "waitForDeviceResidentMTPTransaction(",
        "materializeMTPSpeculativeStatePublicationGraph(",
        "Publication metadata must wait for the transaction's latest mutation "
        "fence before freezing the captured publication bindings.");

    const auto shifted_mutation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordShiftedMTPKVReady(",
        "bool DeviceGraphOrchestrator::waitForPendingShiftedMTPKVReady(");
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
    EXPECT_EQ(executable_shifted_mutation_body.find(
                  "handleLivePrefixReplayStateAfterMutation("),
              std::string::npos)
        << "Shifted MTP KV owns a narrow event-ordered mutation on every GPU "
           "model. MoE must not destroy main/verifier/sidecar graph captures.";
    EXPECT_EQ(executable_shifted_mutation_body.find(
                  "recordLivePrefixMutation("),
              std::string::npos)
        << "Auxiliary shifted KV must never advance the main live-state epoch.";
    EXPECT_EQ(executable_shifted_mutation_body.find(
                  "++live_replay_state_epoch_"),
              std::string::npos)
        << "The main logical-state mailbox remains current across shifted-KV appends.";
    EXPECT_NE(executable_shifted_mutation_body.find(
                  "++shifted_mtp_kv_mutation_generation_"),
              std::string::npos)
        << "Shifted KV needs an independent, event-backed mutation generation.";
    expectNeedleBefore(
        executable_shifted_mutation_body,
        "backend->recordEvent(",
        "++shifted_mtp_kv_mutation_generation_",
        "The shifted-KV event and generation must be published atomically at the producer boundary.");
    EXPECT_NE(shifted_mutation_body.find("mutation_domain"),
              std::string::npos)
        << "PerfStats must identify the independent shifted-cache mutation domain.";

    const auto main_mutation_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "DeviceGraphOrchestrator::recordLivePrefixMutation(",
            "void DeviceGraphOrchestrator::recordLivePrefixSessionReset(")));
    EXPECT_NE(main_mutation_body.find(
                  "device_resident_logical_sequence_state_mailbox_.valid()"),
              std::string::npos)
        << "A main-state epoch transition must reject a still-live logical-state mailbox.";
    expectNeedleBefore(
        main_mutation_body,
        "device_resident_logical_sequence_state_mailbox_.valid()",
        "++live_replay_state_epoch_",
        "Mailbox retirement is a precondition of every main live-state epoch transition.");

    const auto row_select_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPHiddenRowSelect(",
        "bool DeviceGraphOrchestrator::executeMTPTerminalHiddenRowSelect(");
    EXPECT_NE(row_select_body.find("cache.stage->setGPUStream(stream)"), std::string::npos)
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
        "DeviceResidentLogicalStateReadScope",
        "waitForPendingShiftedMTPKVReady",
        "Resident correction shifted commits must first open a scoped publication reader.");
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

    EXPECT_NE(resident_logical_body.find("DeviceResidentLogicalStateReadScope"),
              std::string::npos)
        << "Resident correction shifted commits must retain mailbox ownership until all shifted-KV reads are enqueued.";
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
    EXPECT_EQ(device_target_executable.find("position_offset_override"),
              std::string::npos)
        << "The device-target API must not admit a host-owned position override.";
    EXPECT_EQ(device_target_executable.find("state_.positions"),
              std::string::npos)
        << "Device-target shifted commits must not inspect the host position mirror.";
    EXPECT_NE(device_target_body.find(
                  "transaction.state->cachedTokensForDepth(/*depth=*/0)"),
              std::string::npos)
        << "Device-target shifted commits must source positions from the canonical transaction count.";
    EXPECT_NE(device_target_body.find("/*device_position_offset=*/1"),
              std::string::npos)
        << "The shifted-cache position transform must be composed on device.";
    const auto host_resolved_target_publication_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::stageStochasticTargetTokenForDeviceSampling(",
        "bool DeviceGraphOrchestrator::publishDeviceResidentConditionTokenToTargetSampleSlot(");
    EXPECT_NE(
        host_resolved_target_publication_body.find(
            "enqueuePublishInt32ControlScalarDevice("),
        std::string::npos)
        << "Host-resolved request-policy tokens must acquire a persistent device "
           "owner through the explicit scalar publication kernel.";
    EXPECT_EQ(
        stripCommentsAndStringLiterals(
            host_resolved_target_publication_body)
            .find("hostToDevice"),
        std::string::npos)
        << "A stack-backed asynchronous H2D copy is not a valid token ownership boundary.";
    EXPECT_NE(
        host_resolved_target_publication_body.find(
            "recordStochasticTargetSampleReady("),
        std::string::npos)
        << "The target slot must publish the exact scalar-kernel producer stream.";
    expectNeedleBefore(
        resident_logical_executable,
        "executeMTPDepth0Batched",
        "selectMTPTerminalHiddenRowsFromDeviceAcceptedState",
        "Resident correction shifted commits must restore the accepted verifier terminal hidden after the sidecar append.");
    expectNeedleBefore(
        resident_logical_executable,
        "selectMTPTerminalHiddenRowsFromDeviceAcceptedState",
        "logical_state_read.complete",
        "Resident correction shifted commits must repair terminal hidden before releasing their logical-state read lease.");
    EXPECT_NE(resident_logical_body.find(
                  "shifted_row_resident_terminal_hidden_reselects"),
              std::string::npos)
        << "Accepted-row terminal-hidden repair must remain visible in perf counters.";
    expectNeedleBefore(
        resident_logical_executable,
        "logical_state_read.complete",
        "clearDeviceResidentLogicalSequenceStateMailbox",
        "Resident correction shifted commits must publish reader completion before consuming the mailbox.");
    EXPECT_NE(resident_logical_body.find(
                  "device_resident_logical_state_mailbox_consumptions"),
              std::string::npos)
        << "Resident correction mailbox consumption must remain visible in perfstats.";

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

TEST(Test__GpuWorkspaceAllocationPolicy, MTPTerminalHiddenMailboxUsesExplicitDeviceOwnership)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto engine_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp");
    const auto engine_header =
        readFile(repoRoot() / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.h");
    const auto graph_builder_header =
        readFile(repoRoot() / "src/v2/execution/local_execution/graph/IGraphBuilder.h");

    const auto archive_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(",
        "const float *DeviceGraphOrchestrator::logits() const");
    const auto main_forward_body = sliceBetween(
        source,
        "const float *DeviceGraphOrchestrator::forward(",
        "bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule(");
    const auto initialization_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::initializeMTPPrefillTerminalArchiveReadyEvent()",
        "bool DeviceGraphOrchestrator::beginMTPTerminalHiddenMailboxWrite(");
    const auto mailbox_begin_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::beginMTPTerminalHiddenMailboxWrite(",
        "bool DeviceGraphOrchestrator::publishMTPTerminalHiddenMailboxReady(");
    const auto mailbox_publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishMTPTerminalHiddenMailboxReady(",
        "bool DeviceGraphOrchestrator::waitForPendingMTPPrefillTerminalArchiveReady(");
    const auto consuming_wait_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingMTPPrefillTerminalArchiveReady(",
        "bool DeviceGraphOrchestrator::\n        waitForPendingMTPPrefillTerminalArchiveReadyForObservation(");
    const auto observation_wait_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::\n        waitForPendingMTPPrefillTerminalArchiveReadyForObservation(",
        "void DeviceGraphOrchestrator::clearPendingMTPPrefillTerminalArchiveReady()");
    const auto mutation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLiveGraphProducersBeforePrefixMutation(",
        "bool DeviceGraphOrchestrator::waitForPendingLiveGraphProducersForObservation(");
    const auto graph_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLiveGraphProducersForObservation(",
        "void DeviceGraphOrchestrator::clearMTPVerifierTransactionStateForBoundary(");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(",
        "const float *DeviceGraphOrchestrator::getAllPositionLogits() const");
    const auto refresh_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::refreshMTPTerminalHiddenState(",
        "bool DeviceGraphOrchestrator::ensureMTPCheckpointTerminalHidden()");
    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    const auto row_selectors_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRows(",
        "void DeviceGraphOrchestrator::noteMainForwardHiddenProducedForMTP(");
    const auto single_row_selector_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRow(",
        "bool DeviceGraphOrchestrator::executeMTPDepth0(");
    const auto checkpoint_import_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::restoreLiveCheckpointTerminalHidden(",
        "bool DeviceGraphOrchestrator::importMTPCheckpointTerminalHidden(");
    const auto checkpoint_import_entry_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::importMTPCheckpointTerminalHidden(",
        "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRow(");
    const auto device_outcome_suffix_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(");
    const auto partial_forward_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(",
        "const float *DeviceGraphOrchestrator::mtpLogits() const");
    const auto shifted_event_initialization_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::initializeShiftedMTPKVReadyEvent()",
        "bool DeviceGraphOrchestrator::recordShiftedMTPKVReady(");
    const auto shifted_event_record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordShiftedMTPKVReady(",
        "bool DeviceGraphOrchestrator::waitForPendingShiftedMTPKVReady(");

    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto compact_engine_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(engine_source));
    const auto compact_engine_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(engine_header));
    const auto compact_graph_builder_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(graph_builder_header));
    const auto compact_archive =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(archive_body));
    const auto compact_main_forward =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(main_forward_body));
    const auto compact_initialization =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(initialization_body));
    const auto compact_mailbox_begin =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mailbox_begin_body));
    const auto compact_mailbox_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mailbox_publish_body));
    const auto compact_consuming_wait =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(consuming_wait_body));
    const auto compact_observation_wait =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(observation_wait_body));
    const auto compact_mutation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mutation_body));
    const auto compact_graph_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(graph_observation_body));
    const auto compact_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_refresh =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(refresh_body));
    const auto compact_sidecar =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_body));
    const auto compact_row_selectors =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(row_selectors_body));
    const auto compact_single_row_selector =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(single_row_selector_body));
    const auto compact_checkpoint_import =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(checkpoint_import_body));
    const auto compact_checkpoint_import_entry =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(
            checkpoint_import_entry_body));
    const auto compact_device_outcome_suffix =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(device_outcome_suffix_body));
    const auto compact_partial_forward =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(partial_forward_body));
    const auto compact_shifted_event_initialization =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(
            shifted_event_initialization_body));
    const auto compact_shifted_event_record =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(
            shifted_event_record_body));

    EXPECT_NE(
        compact_header.find(
            "PendingMTPPrefillTerminalArchiveReadyState"
            "mtp_prefill_terminal_archive_ready_;"),
        std::string::npos)
        << "The asynchronous terminal-row copy needs an owned event handoff.";
    EXPECT_EQ(
        compact_engine_header.find("LastForwardExecution"),
        std::string::npos)
        << "Engine-global producer streams can outlive graph-cache ownership and are forbidden.";
    EXPECT_NE(
        compact_graph_builder_header.find(
            "structForwardExecutionProvenance"),
        std::string::npos)
        << "Each forward result must carry its own typed execution provenance.";
    EXPECT_NE(
        compact_graph_builder_header.find(
            "ForwardExecutionProvenanceexecution;"),
        std::string::npos)
        << "The producer stream belongs to the concrete ForwardOutput, not only engine-global state.";
    EXPECT_EQ(
        compact_engine_source.find("recordLastForwardExecution("),
        std::string::npos)
        << "Long-lived producer ordering belongs to a durable event, not a retained raw stream.";
    EXPECT_NE(
        removeAsciiWhitespace(stripCommentsAndStringLiterals(source)).find(
            "publishForwardGraphOutputReady(output.execution)"),
        std::string::npos)
        << "Every successful forward must publish its invocation-scoped stream before it can be invalidated.";
    EXPECT_NE(
        compact_archive.find(
            "hidden_selection_stream=producer.stream;"),
        std::string::npos)
        << "Shifted prefill must consume the stream carried by its exact ForwardOutput.";
    EXPECT_EQ(
        compact_archive.find("lastMainForwardHiddenProducerStream("),
        std::string::npos)
        << "Immediate shifted-prefill handoff must not rediscover its producer through mutable engine-global state.";
    EXPECT_NE(
        compact_main_forward.find(
            "input.position_offset,output.execution,"),
        std::string::npos)
        << "Main forward must pass invocation-scoped provenance directly into shifted-prefill publication.";
    EXPECT_NE(
        compact_archive.find(
            "selectMTPTerminalHiddenRow("
            "seq_len-1,seq_len,hidden_selection_stream)"),
        std::string::npos)
        << "GPU terminal-row archival must remain on the main-forward producer stream.";
    EXPECT_EQ(
        compact_archive.find("synchronizeOwnedGPUHelperStream("),
        std::string::npos)
        << "Terminal-row archival must not turn its stream dependency into a host wait.";

    EXPECT_NE(compact_initialization.find("backend->createEvent("), std::string::npos)
        << "The reusable event must be allocated during runner initialization.";
    EXPECT_NE(
        compact_mailbox_begin.find(
            "waitForPendingMTPPrefillTerminalArchiveReady("
            "writer_stream,"),
        std::string::npos)
        << "A writer must adopt any older terminal-hidden publication before replacing it.";
    EXPECT_NE(
        compact_mailbox_begin.find(
            "waitForPendingShiftedMTPKVReady("
            "writer_stream,"),
        std::string::npos)
        << "A writer must wait until the prior KV-only sidecar stops reading the mailbox.";
    EXPECT_EQ(compact_mailbox_publish.find("backend->createEvent("), std::string::npos)
        << "Hidden-row publication is a hot path and must not allocate events.";
    EXPECT_NE(compact_mailbox_publish.find("backend->recordEvent("), std::string::npos);
    EXPECT_NE(compact_consuming_wait.find("backend->streamWaitEvent("), std::string::npos);
    EXPECT_NE(compact_consuming_wait.find("ready.valid=false;"), std::string::npos)
        << "A forward or mutation consumer adopts terminal-buffer ownership.";
    EXPECT_NE(compact_observation_wait.find("backend->streamWaitEvent("), std::string::npos);
    EXPECT_EQ(compact_observation_wait.find("ready.valid=false;"), std::string::npos)
        << "Prefix harvest observes the terminal row without stealing the next forward's handoff.";

    EXPECT_NE(
        compact_prepare.find(
            "waitForPendingMTPPrefillTerminalArchiveReadyForObservation("
            "execution_stream,"),
        std::string::npos)
        << "The next graph protects the archive source without stealing the sidecar payload handoff.";
    EXPECT_NE(
        compact_sidecar.find(
            "waitForPendingMTPPrefillTerminalArchiveReady("
            "sidecar_dynamic_stream,"),
        std::string::npos)
        << "The sidecar must consume terminal-hidden readiness before reading the selected row.";
    EXPECT_NE(
        compact_refresh.find(
            "waitForForwardGraphOutputReady("
            "producer_stream,DeviceTimelineRole::MTPSidecarGraph,"
            "ForwardGraphOutputKind::Any,"),
        std::string::npos)
        << "Terminal refresh must wait on the exact latest main-model graph. "
           "Prefill and grouped verification can both own its hidden rows.";
    EXPECT_EQ(
        compact_refresh.find("synchronizeOwnedGPUHelperStream("),
        std::string::npos)
        << "Terminal refresh must remain device ordered instead of synchronizing the host.";
    EXPECT_GE(
        countOccurrences(
            compact_row_selectors,
            "beginMTPTerminalHiddenMailboxWrite("),
        3u)
        << "Every multi-row and device-metadata selector must acquire the mailbox itself.";
    EXPECT_GE(
        countOccurrences(
            compact_row_selectors,
            "publishMTPTerminalHiddenMailboxReady("),
        3u)
        << "Every multi-row and device-metadata selector must publish its own completion.";
    EXPECT_NE(
        compact_single_row_selector.find(
            "beginMTPTerminalHiddenMailboxWrite("),
        std::string::npos);
    EXPECT_NE(
        compact_single_row_selector.find(
            "publishMTPTerminalHiddenMailboxReady("),
        std::string::npos);
    EXPECT_NE(
        compact_checkpoint_import.find(
            "beginMTPTerminalHiddenMailboxWrite("),
        std::string::npos)
        << "Checkpoint D2D import is a mailbox writer and must acquire ownership.";
    EXPECT_NE(
        compact_checkpoint_import.find(
            "publishMTPTerminalHiddenMailboxReady("),
        std::string::npos)
        << "Checkpoint D2D import must publish completion for the sidecar.";
    EXPECT_NE(
        compact_checkpoint_import_entry.find(
            "restoreLiveCheckpointTerminalHidden("),
        std::string::npos)
        << "The public import entry point must use the domain-checked restore primitive.";
    EXPECT_NE(
        compact_device_outcome_suffix.find(
            "main_forward_token_count,outcome.stream)"),
        std::string::npos)
        << "Device-outcome suffix selection must execute on its verifier producer stream.";
    EXPECT_NE(
        compact_partial_forward.find(
            "waitForForwardGraphOutputReady("
            "hidden_producer_stream,"
            "DeviceTimelineRole::AcceptedStatePublication,"
            "ForwardGraphOutputKind::GroupedVerifier,"),
        std::string::npos)
        << "Grouped publication must consume the durable all-position verifier event.";
    EXPECT_NE(
        compact_partial_forward.find(
            "main_forward_token_count,hidden_producer_stream)"),
        std::string::npos);
    EXPECT_NE(
        compact_shifted_event_initialization.find("backend->createEvent("),
        std::string::npos)
        << "Deferred KV-only sidecars need a preallocated completion event.";
    EXPECT_EQ(
        compact_shifted_event_record.find("backend->createEvent("),
        std::string::npos)
        << "Shifted-KV publication must never allocate in the hot path.";
    EXPECT_NE(
        compact_sidecar.find(
            "if(kv_cache_only&&defer_final_sync)"
            "{if(!recordShiftedMTPKVReady("),
        std::string::npos)
        << "Deferred first-use and captured KV-only sidecars must publish an event instead of synchronizing.";
    EXPECT_NE(
        compact_mutation.find(
            "waitForPendingMTPPrefillTerminalArchiveReady("
            "mutation_stream,"),
        std::string::npos)
        << "Prefix restore/truncate must order itself after the archived row.";
    EXPECT_NE(
        compact_graph_observation.find(
            "waitForPendingMTPPrefillTerminalArchiveReadyForObservation("
            "observation_stream,"),
        std::string::npos)
        << "Prefix harvest must see the archived row without synchronizing the host.";
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
    const auto compact_source = removeAsciiWhitespace(executable_source);

    EXPECT_GE(countOccurrences(compact_header, "uint64_tworkspace_generation=0;"), 2u)
        << "Both single-row and multi-row MTP terminal-hidden helper caches must remember "
           "the allocator generation they last ran under.";
    EXPECT_GE(countOccurrences(executable_source, "workspace_generation_changed"), 2u)
        << "Legacy scalar/arbitrary-row helpers must still detect allocator-generation changes.";
    EXPECT_NE(executable_source.find("materializeMTPTerminalHiddenPublicationGraphs"),
              std::string::npos)
        << "GPU publication helpers must be materialized after graph-family workspace binding.";
    EXPECT_NE(compact_source.find("if(!selected_cache->valid||!selected_cache->graph||"),
              std::string::npos)
        << "Decode catchup must fail closed instead of rebuilding a stale publication graph.";
    EXPECT_NE(compact_source.find("if(!accepted_cache.valid||!accepted_cache.graph||"),
              std::string::npos)
        << "Accepted-state publication must fail closed instead of rebuilding a stale graph.";
    EXPECT_NE(compact_source.find("if(!request_cache.valid||!request_cache.graph||"),
              std::string::npos)
        << "Request-terminal publication must fail closed instead of rebuilding a stale graph.";

    const auto reset_body = sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override");
    const auto reset_executable =
        stripCommentsAndStringLiterals(reset_body);
    const auto compact_reset = removeAsciiWhitespace(reset_executable);
    EXPECT_NE(reset_executable.find("mtp_terminal_hidden_row_select_cache_.resetSessionState()"),
              std::string::npos);
    EXPECT_NE(reset_executable.find("mtp_terminal_hidden_rows_select_cache_.resetSessionState()"),
              std::string::npos);
    EXPECT_NE(reset_executable.find("mtp_terminal_hidden_contiguous_rows_select_caches_"),
              std::string::npos);
    EXPECT_NE(compact_reset.find("mtp_terminal_hidden_device_accepted_rows_select_caches_"),
              std::string::npos)
        << "Content-only request reset must preserve every exact-count GPU publication graph.";
    EXPECT_NE(
        compact_reset.find(
            "for(auto&cache:mtp_terminal_hidden_request_rows_select_caches_)"
            "{if(cache)cache->resetSessionState();}"),
        std::string::npos)
        << "Content-only request reset must preserve every request-count publication graph.";
    EXPECT_EQ(reset_executable.find("mtp_terminal_hidden_row_select_cache_.invalidate()"),
              std::string::npos);
    EXPECT_EQ(reset_executable.find("mtp_terminal_hidden_rows_select_cache_.invalidate()"),
              std::string::npos)
        << "Request reset must not destroy graph objects whose binding identities remain stable.";

    const auto buffer_replacement = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::ensureMTPTerminalHiddenBuffer(",
        "bool DeviceGraphOrchestrator::executeMTPHiddenRowSelect(");
    const auto compact_buffer_replacement =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(buffer_replacement));
    EXPECT_NE(compact_buffer_replacement.find(
                  "mtp_terminal_hidden_device_accepted_rows_select_caches_"),
              std::string::npos)
        << "Replacing the mailbox tensor is a real binding-identity change and must invalidate the exact-count graph family.";
    EXPECT_NE(compact_buffer_replacement.find(
                  "mtp_terminal_hidden_request_rows_select_caches_"),
              std::string::npos)
        << "Mailbox replacement must invalidate every request-terminal graph's captured output binding.";
    EXPECT_EQ(compact_buffer_replacement.find(
                  "mtp_terminal_hidden_request_rows_select_cache_"),
              std::string::npos)
        << "The prompt-width-specific singleton request cache must not return.";

}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPTerminalHiddenGpuPublicationHasTypedHostFreeCaches)
{
    const auto root = repoRoot();
    const auto source = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header = readFile(
        root / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto stage = readFile(
        root / "src/v2/execution/compute_stages/stages/HiddenStateRowsSelectStage.cpp");

    const auto materialize = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::materializeMTPTerminalHiddenRowsSelectGraph(",
            "bool DeviceGraphOrchestrator::materializeMTPTerminalHiddenPublicationGraphs(")));
    EXPECT_NE(materialize.find("DeviceRowIndexSource::FixedContiguousRange"),
              std::string::npos);
    EXPECT_NE(materialize.find("DeviceRowIndexSource::ExternalDeviceIndices"),
              std::string::npos);
    EXPECT_NE(materialize.find("DeviceRowIndexSource::RequestTerminalLengths"),
              std::string::npos);
    EXPECT_NE(materialize.find("DeviceRowIndexSource::ShiftedPrefillKVProgress"),
              std::string::npos)
        << "Shifted-prefill publication must derive its row range from canonical device KV progress.";
    EXPECT_NE(
        materialize.find(
            "deviceSequenceCachedTokenCountPtr(request_index)"),
        std::string::npos)
        << "Graph setup must bind the cache-owned device progress counters; a host cursor must not become graph state.";
    EXPECT_NE(
        materialize.find(
            "RequestRowStrideSource::ExternalDeviceScalar"),
        std::string::npos)
        << "Request-terminal graphs must read their current padded stride from device geometry.";
    EXPECT_NE(
        materialize.find(
            "params.request_row_stride_device=request_row_stride_device"),
        std::string::npos)
        << "Graph construction must bind the exact device-owned stride scalar.";
    EXPECT_EQ(
        materialize.find(
            "seq_capacity=request_input_row_capacity_"),
        std::string::npos)
        << "Request-input staging capacity must not define captured activation geometry.";
    EXPECT_NE(
        materialize.find(
            "constintseq_capacity=static_cast<int>(hidden_rows)"),
        std::string::npos)
        << "Each selector graph must be total over its stable hidden-state activation capacity.";
    EXPECT_EQ(materialize.find("DeviceRowIndexSource::StageOwnedIndices"),
              std::string::npos)
        << "Production GPU terminal-hidden publication graphs must never own a host row plan.";
    EXPECT_EQ(materialize.find("setSelectedRowsForReplay("), std::string::npos);

    const auto accepted = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromDeviceAcceptedState(",
            "void DeviceGraphOrchestrator::noteMainForwardHiddenProducedForMTP(")));
    EXPECT_NE(accepted.find("mtp_terminal_hidden_device_accepted_rows_select_caches_"),
              std::string::npos)
        << "Device-accepted rows need an exact-count cache family that cannot alias host-described catchup rows.";
    EXPECT_EQ(accepted.find("mtp_terminal_hidden_rows_select_cache_"),
              std::string::npos)
        << "The former shared cache allowed alternating producers to destroy pinned state in decode.";
    EXPECT_NE(accepted.find("static_cast<size_t>(row_count-1)"),
              std::string::npos)
        << "Accepted publication must select the graph whose captured geometry equals the live request count.";
    EXPECT_NE(accepted.find("ensureMTPTerminalHiddenBuffer(row_count)"),
              std::string::npos)
        << "Accepted publication must not size execution from inactive maximum-capacity rows.";
    EXPECT_NE(accepted.find("accepted_cache.selected_row_count!=row_count"),
              std::string::npos)
        << "Exact-count graph identity must be validated before publication.";
    EXPECT_EQ(accepted.find("publication_row_capacity,seq_len,stream"),
              std::string::npos)
        << "A smaller live batch must never replay the maximum-capacity selector.";

    const auto stage_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            stage,
            "bool HiddenStateRowsSelectStage::executeGPU(",
            "void HiddenStateRowsSelectStage::releaseGpuParamState(")));
    EXPECT_NE(stage_execute.find("launchFixedRowsSelectFP32("),
              std::string::npos);
    EXPECT_NE(
        stage_execute.find(
            "!request_terminal_rows&&!fixed_contiguous_rows&&!shifted_prefill_progress&&!uploadGpuSelectedRows()"),
        std::string::npos)
        << "Every device-owned row source must bypass pinned host metadata upload.";
    EXPECT_NE(stage_execute.find("launchDeviceKVProgressRowsSelectFP32("),
              std::string::npos)
        << "Shifted-prefill progress must be resolved by the captured backend kernel.";

    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    EXPECT_NE(compact_header.find(
                  "mtp_terminal_hidden_contiguous_rows_select_caches_"),
              std::string::npos);
    EXPECT_NE(compact_header.find(
                  "mtp_terminal_hidden_device_accepted_rows_select_caches_"),
              std::string::npos);
    EXPECT_NE(compact_header.find(
                  "mtp_terminal_hidden_request_rows_select_caches_"),
              std::string::npos);
    EXPECT_NE(compact_header.find(
                  "mtp_shifted_prefill_hidden_rows_select_caches_"),
              std::string::npos)
        << "Shifted-prefill publication needs one stable graph owner for each request/row geometry.";

    const auto shifted_prefill_materialization = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "const size_t expected_shifted_prefill_graphs =",
            "return true;\n    }\n\n    bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelect(")));
    EXPECT_NE(
        shifted_prefill_materialization.find(
            "for(intrequest_index=0;request_index<stochastic_batch_output_request_capacity_;++request_index)"),
        std::string::npos)
        << "Graph setup must cover every configured request index.";
    EXPECT_NE(
        shifted_prefill_materialization.find(
            "for(introw_count=1;row_count<=mtp_sidecar_condition_token_slot_width_;++row_count)"),
        std::string::npos)
        << "Graph setup must cover every legal grouped publication width.";

    const auto shifted_prefill_replay = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromShiftedPrefillProgress(",
            "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromDeviceAcceptedState(")));
    const size_t wait_position = shifted_prefill_replay.find(
        "waitForPendingShiftedMTPKVReady(");
    const size_t execute_position = shifted_prefill_replay.find(
        "executeMTPTerminalHiddenRowsCaptured(");
    const size_t publish_position = shifted_prefill_replay.find(
        "publishMTPTerminalHiddenMailboxReady(");
    ASSERT_NE(wait_position, std::string::npos);
    ASSERT_NE(execute_position, std::string::npos);
    ASSERT_NE(publish_position, std::string::npos);
    EXPECT_LT(wait_position, execute_position)
        << "The selector must consume the exact shifted-KV producer event before graph replay.";
    EXPECT_LT(execute_position, publish_position)
        << "The selector graph must complete on its stream before publishing the mailbox handoff.";
    EXPECT_EQ(
        shifted_prefill_replay.find(
            "materializeMTPTerminalHiddenRowsSelectGraph("),
        std::string::npos)
        << "Execution must never manufacture a replacement graph for live KV progress.";

    const auto shifted_prefill_population = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(",
            "const float *DeviceGraphOrchestrator::logits() const")));
    EXPECT_NE(
        shifted_prefill_population.find(
            "state_.device_id.is_gpu()?selectMTPTerminalHiddenRowsFromShiftedPrefillProgress("),
        std::string::npos)
        << "GPU shifted-prefill rows, including the terminal archive, must be selected from live device KV progress.";
    EXPECT_NE(
        shifted_prefill_population.find(
            "state_.device_id.is_gpu()?selectMTPTerminalHiddenRowsFromShiftedPrefillProgress("),
        std::string::npos)
        << "The terminal archive must choose device progress before its CPU-only host-indexed arm.";
    EXPECT_EQ(
        shifted_prefill_population.find(
            "if(!selectMTPTerminalHiddenRow(seq_len-1,seq_len,hidden_selection_stream))"),
        std::string::npos)
        << "An unconditional prompt-width host cursor must never return to terminal archiving.";

    const auto manifest = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::buildMTPWorkspaceFamilyManifest(",
            "bool DeviceGraphOrchestrator::materializeForwardGraphForShape(")));
    EXPECT_EQ(manifest.find("DeviceRowIndexSource::StageOwnedIndices"),
              std::string::npos)
        << "GPU workspace manifests must not keep retired host-owned terminal-row plans alive.";

    const auto fixed_rows_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelect(\n"
            "        TensorBase *input,\n"
            "        BufferId input_buffer_id,\n"
            "        TensorBase *output,\n"
            "        BufferId output_buffer_id,\n"
            "        MTPTerminalHiddenRowsSelectGraphCache &cache,\n"
            "        const char *node_name,\n"
            "        int row_start,",
            "bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelect(\n"
            "        TensorBase *input,\n"
            "        BufferId input_buffer_id,\n"
            "        TensorBase *output,\n"
            "        BufferId output_buffer_id,\n"
            "        MTPTerminalHiddenRowsSelectGraphCache &cache,\n"
            "        const char *node_name,\n"
            "        const std::vector<int> &selected_rows,")));
    EXPECT_NE(fixed_rows_execute.find("if(!prepared)"), std::string::npos)
        << "Runtime publication must validate the prebuilt graph binding.";
    EXPECT_EQ(
        fixed_rows_execute.find(
            "materializeMTPTerminalHiddenRowsSelectGraph("),
        std::string::npos)
        << "Runtime publication must never construct a replacement graph.";

    const auto scalar_selector = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRow(",
            "bool DeviceGraphOrchestrator::executeMTPDepth0(")));
    EXPECT_NE(
        scalar_selector.find(
            "if(state_.device_id.is_gpu())"
            "{returnselectMTPTerminalHiddenRows(row_idx,1,seq_len,stream);}"),
        std::string::npos)
        << "Every one-row GPU publication must use the host-free fixed-range graph.";

    const auto scalar_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::executeMTPHiddenRowSelect(",
            "bool DeviceGraphOrchestrator::executeMTPTerminalHiddenRowSelect(")));
    EXPECT_NE(
        scalar_execute.find(
            "if(state_.device_id.is_gpu()){LOG_ERROR();returnfalse;}"),
        std::string::npos)
        << "The host-authored scalar replay implementation must be CPU-only.";
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

/**
 * @brief Prevent scalar padded prefill from mutating only the request-state bank.
 *
 * The persistent device request-length allocation is shared metadata, not a
 * declaration that a one-request graph owns batched recurrent state.  Scalar
 * graph buckets must call the device-effective-length APIs, which advance the
 * primary short-conv and recurrence banks consumed by subsequent decode.
 * Genuine multi-request graphs use the request-batched APIs instead.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     ScalarPaddedGDNUsesPrimaryDeviceStateWhileBatchesUseRequestBanks)
{
    const auto root = repoRoot();
    const auto conv_source = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(readFile(
            root / "src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp")));
    const auto recurrence_source = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(readFile(
            root / "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp")));

    EXPECT_NE(
        conv_source.find(
            "if(request_batched){"),
        std::string::npos);
    EXPECT_NE(
        conv_source.find(
            "forwardBatchedRequestsWithDeviceSeqLens("),
        std::string::npos);
    EXPECT_NE(
        conv_source.find(
            "elseif(use_scalar_real_length_contract){"),
        std::string::npos);
    EXPECT_NE(
        conv_source.find(
            "forwardWithEffectiveSeqLen("),
        std::string::npos);
    EXPECT_EQ(
        conv_source.find(
            "request_batched||use_scalar_real_length_contract"),
        std::string::npos)
        << "Scalar graph buckets must not be folded into request-bank execution.";

    EXPECT_NE(
        recurrence_source.find(
            "if(request_batched){"),
        std::string::npos);
    EXPECT_NE(
        recurrence_source.find(
            "chunkForwardBatchedRequestsWithDeviceSeqLens("),
        std::string::npos);
    EXPECT_NE(
        recurrence_source.find(
            "elseif(use_scalar_real_length_contract){"),
        std::string::npos);
    EXPECT_NE(
        recurrence_source.find(
            "chunkForwardWithEffectiveSeqLen("),
        std::string::npos);
    EXPECT_EQ(
        recurrence_source.find(
            "request_batched||use_scalar_real_length_contract"),
        std::string::npos)
        << "Scalar graph buckets must commit the primary recurrence bank.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDANativeVNNIDispatchSweepUsesExplicitStream)
{
    const auto source =
        readFile(repoRoot() / "tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp");

    EXPECT_NE(source.find("cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)"), std::string::npos)
        << "The CUDA NativeVNNI dispatch trainer must create an explicit non-blocking stream.";
    EXPECT_NE(source.find("kernel->setGPUStream(static_cast<void *>(stream))"), std::string::npos)
        << "The CUDA NativeVNNI dispatch trainer must bind its GEMM kernel to the explicit stream.";
    EXPECT_NE(source.find("kernel->clearGPUStreamBinding()"), std::string::npos)
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
        readFile(repoRoot() / "src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvShardImpl.cu.inc");
    const auto kernel_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp");
    const auto sweep_source =
        readFile(repoRoot() / "tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp");

    const auto small_m_body = sliceBetween(
        tuned_source,
        "bool dispatchCodebookSmallMRowPar(",
        "bool dispatchSmallMByCodebook(");

    EXPECT_NE(small_m_body.find("if (g_sweep.active)"), std::string::npos)
        << "CUDA runtime-M NativeVNNI trainer rows must time the requested candidate, not the generated runtime route.";
    EXPECT_NE(small_m_body.find("tuning = GeneratedDispatchTuning{"), std::string::npos)
        << "The small-M sweep override must forward tile/family parameters into the real launch.";

    EXPECT_NE(kernel_source.find("if (cudaNativeVNNIGemvSweep_isActive())"), std::string::npos)
        << "Small-M training sweeps must fail closed instead of falling through to generic M>1 GEMM.";

    EXPECT_NE(sweep_source.find("candidateSupportedByGpuPreparedSweepPath"), std::string::npos)
        << "The CUDA NativeVNNI trainer must filter candidate families unsupported by the GPU-prepared harness.";
    EXPECT_NE(sweep_source.find("if (m > 1)\n            return candidate.family == SweepFamily::KPar;"),
              std::string::npos)
        << "Grouped trainer cases must not label WIDE/DIRECT/ROWPAR rows as specialized KPAR timings.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, MTPGpuSidecarsRequireDeviceOwnedConditionTokens)
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
    const auto compact_source = removeAsciiWhitespace(source);

    /*
     * GPU graph capture records the embedding kernel's token pointer. Every
     * GPU sidecar therefore requires a device-owned source and copies it into
     * the role-specific MTP_CONDITION_TOKEN slot. Host scalars are a CPU-only
     * contract and may never be uploaded from the sidecar execution path.
     */
    EXPECT_NE(header.find("mtp_sidecar_condition_token_capacity_"), std::string::npos)
        << "The condition-token buffer must expose its row capacity to runtime validation.";
    EXPECT_NE(header.find("MTPSidecarCaptureLayout"), std::string::npos)
        << "One typed layout must own sidecar cache indices and condition-token slots.";
    EXPECT_EQ(source.find("kMTPSidecarConditionTokenSlotCount"), std::string::npos)
        << "A fixed sidecar slot count cannot cover runtime-configured MTP depths.";
    EXPECT_NE(compact_source.find(
                  "BufferId::MTP_CONDITION_TOKEN,1,"
                  "static_cast<size_t>(mtp_sidecar_condition_token_slot_width_)*"
                  "static_cast<size_t>(mtp_sidecar_capture_layout_.conditionTokenSlotCount())"),
              std::string::npos)
        << "MTP_CONDITION_TOKEN must hold the configured runtime row capacity for every sidecar slot.";
    EXPECT_NE(sidecar_body.find("mtp_sidecar_capture_layout_.conditionTokenSlot"), std::string::npos)
        << "MTP sidecar caches and token slots must resolve from the same typed role.";
    EXPECT_EQ(sidecar_body.find("&cache =="), std::string::npos)
        << "Token-slot ownership must not be reverse-engineered from cache addresses.";
    EXPECT_NE(executable_sidecar_body.find(
                  "condition_token_slot*mtp_sidecar_condition_token_slot_width_"),
              std::string::npos)
        << "Device-token staging must use the cache-owned slot offset, not the buffer base.";
    EXPECT_NE(sidecar_body.find("condition_token_device"), std::string::npos)
        << "Graph construction and token staging must share the same slot pointer.";
    EXPECT_NE(sidecar_body.find(
                  "GPU MTP sidecars require a device-owned condition-token source"),
              std::string::npos)
        << "A missing GPU device token source must fail closed.";
    EXPECT_EQ(sidecar_body.find("backend->hostToDeviceOnStream"), std::string::npos)
        << "GPU sidecars must never reconstruct device state from a host scalar.";
    EXPECT_NE(executable_sidecar_body.find(
                  "external_device_condition_tokens||prepare_device_condition_tokens_from_speculative_outcome"),
              std::string::npos)
        << "Resident speculative-outcome catch-up rows must not copy from nullable host draft tokens.";
    EXPECT_NE(executable_sidecar_body.find(
                  "std::fill(sidecar_cache.token_ids.begin(),sidecar_cache.token_ids.end(),0);"),
              std::string::npos)
        << "The host shadow remains inert graph metadata for device-owned sidecars.";
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
    EXPECT_EQ(sidecar_body.find("deviceToHostFast("), std::string::npos)
        << "GPU sidecar staging must not inspect device-owned condition tokens through a validation-only D2H seam.";
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
                  "cache->resetSessionStatePreservingGraphReplay();"),
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
        << "Typed non-preserving reset boundaries still need reset telemetry.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, PrefixRestorePreservesStableMTPSidecarGraphs)
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
    const auto compact_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(source));

    /*
     * Regression guard for prefix-cache + MTP + MoE LocalTP: a restored prefix
     * block may contain live KV/GDN/MTP payloads but no graph-owned
     * model-runtime placement snapshot. The no-payload path restores the
     * immutable initial MoE state behind model-lifetime device addresses. Since
     * neither captured pointers nor graph topology change, sidecar executables
     * must survive the boundary and consume the reset-ready event.
     */
    EXPECT_EQ(compact_header.find("invalidateMTPSidecarDepth0GraphState"),
              std::string::npos)
        << "Stable-address prefix restore must not retain a graph-invalidation escape hatch.";
    EXPECT_EQ(compact_source.find("invalidateMTPSidecarDepth0GraphState"),
              std::string::npos)
        << "The obsolete prefix graph-invalidation implementation must stay retired.";
    EXPECT_NE(reset_body.find(
                  "if(!request.preserve_replay_safe_graphs){throwstd::invalid_argument("),
              std::string::npos)
        << "Prefix restore must reject requests that attempt to discard replay-safe captures.";
    EXPECT_NE(reset_body.find(
                  "if(preserve_replay_safe_graphs){mtp_sidecar_depth0_cache_.resetSessionStatePreservingGraphReplay();"),
              std::string::npos)
        << "Prefix restore must retain the ordinary MTP sidecar executable.";
    EXPECT_NE(reset_body.find(
                  "mtp_sidecar_depth0_device_token_cache_.resetSessionStatePreservingGraphReplay();"),
              std::string::npos)
        << "The production stochastic sidecar executable must remain replay-hot.";
    EXPECT_NE(reset_body.find(
                  "mtp_sidecar_depth0_kv_only_device_token_cache_.resetSessionStatePreservingGraphReplay();"),
              std::string::npos)
        << "KV-only device-token sidecars share the stable-address lifetime.";
    EXPECT_NE(reset_body.find(
                  "if(preserve_replay_safe_graphs)cache->resetSessionStatePreservingGraphReplay();"),
              std::string::npos)
        << "Batched KV-only sidecars must preserve their graph executables too.";
    EXPECT_EQ(source.find("sidecar_graph_invalidations"), std::string::npos)
        << "Prefix restore must not report graph invalidation after stable-address restoration.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GraphBuildPublicationStreamHasNoAmbientBuilderState)
{
    const auto interface_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/graph/IGraphBuilder.h");
    const auto qwen_header =
        readFile(repoRoot() / "src/v2/models/qwen/QwenGraphBase.h");
    const auto qwen_source =
        readFile(repoRoot() / "src/v2/models/qwen/QwenGraphBase.cpp");
    const auto moe_source =
        readFile(repoRoot() / "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp");
    const auto orchestrator_header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");

    const auto compact_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(interface_source));
    const auto compact_qwen_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(qwen_header));
    const auto compact_qwen_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(qwen_source));
    const auto compact_moe_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(moe_source));
    const auto compact_orchestrator_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(orchestrator_header));

    EXPECT_NE(
        compact_interface.find(
            "void*device_state_publication_stream=nullptr;constint*position_ids"),
        std::string::npos)
        << "LayerContext must carry graph-build publication ownership as an explicit value.";
    EXPECT_NE(
        compact_interface.find(
            "DeviceIddevice,void*device_state_publication_stream,constint32_t*sequence_lengths_device=nullptr"),
        std::string::npos)
        << "Every direct FFN graph build must name its publication stream.";
    EXPECT_EQ(
        compact_qwen_header.find("device_state_publication_stream_"),
        std::string::npos)
        << "The Qwen builder must not cache a producer stream as ambient mutable state.";
    EXPECT_EQ(
        compact_qwen_source.find("device_state_publication_stream_"),
        std::string::npos);
    EXPECT_EQ(
        compact_moe_source.find("device_state_publication_stream_"),
        std::string::npos);
    EXPECT_NE(
        compact_orchestrator_header.find(
            "FFNGraphSession&withDeviceStatePublicationStream(void*stream);"),
        std::string::npos)
        << "The fluent orchestrator API must expose the publication dependency at the call site.";
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
        "bool DeviceGraphOrchestrator::ensureLiveCheckpointStorage");
    const auto checkpoint_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(",
        "bool DeviceGraphOrchestrator::restoreLivePrefixState");
    const auto live_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForLiveInferenceStateReadyForObservation(",
        "bool DeviceGraphOrchestrator::prepareDeviceTokenInputsForForwardGraphExecution(");
    const auto published_join_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::joinPublishedLiveStateHandoffs(",
        "void DeviceGraphOrchestrator::clearPendingLivePrefixMutationReady()");

    const auto compact_probe =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(probe_body));
    const auto compact_payload =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(payload_body));
    const auto compact_checkpoint =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(checkpoint_body));
    const auto compact_live_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(live_observation_body));
    const auto compact_published_join =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(published_join_body));
    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));

    EXPECT_NE(compact_header.find(
                  "waitForLiveInferenceStateReadyForObservation("
                  "void*observation_stream,constchar*observation_name,"
                  "DeviceTimelineRoleobservation_role)const"),
              std::string::npos)
        << "Host-visible live-state exports must use the shared observation boundary.";
    EXPECT_NE(compact_header.find(
                  "joinPublishedLiveStateHandoffs("
                  "void*consumer_stream,DeviceTimelineRoleconsumer_role,"
                  "constchar*consumer_name)const"),
              std::string::npos)
        << "Published state may only be joined through the typed role API.";
    EXPECT_EQ(compact_header.find(
                  "waitForPendingAcceptedSpecPublicationReady("),
              std::string::npos);
    EXPECT_EQ(compact_header.find(
                  "waitForPendingAcceptedSpecPublicationReadyForObservation("),
              std::string::npos);
    EXPECT_NE(compact_published_join.find("accepted_spec_publication_ready_"),
              std::string::npos);
    EXPECT_NE(compact_published_join.find("live_prefix_mutation_ready_"),
              std::string::npos);
    EXPECT_NE(compact_published_join.find("enqueuePublishedWait("),
              std::string::npos)
        << "Cross-lifetime publications must always lower to durable event waits.";
    EXPECT_NE(compact_published_join.find(
                  "caseDeviceTimelineRole::MTPSidecarGraph:"),
              std::string::npos);
    EXPECT_NE(compact_published_join.find(
                  "caseDeviceTimelineRole::PrefixCheckpointArchive:"),
              std::string::npos);
    EXPECT_NE(compact_published_join.find(
                  "caseDeviceTimelineRole::Diagnostics:"),
              std::string::npos);
    EXPECT_NE(compact_published_join.find(
                  "if(consumes_publications){"),
              std::string::npos)
        << "Observation roles must not clear ownership publications.";

    EXPECT_NE(compact_live_observation.find(
                  "joinPublishedLiveStateHandoffs("
                  "observation_stream,observation_role,consumer)"),
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
    const auto live_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForLiveInferenceStateReadyForObservation(",
        "bool DeviceGraphOrchestrator::prepareDeviceTokenInputsForForwardGraphExecution(");
    const auto shifted_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingShiftedMTPKVReadyForObservation(",
        "bool DeviceGraphOrchestrator::recordAllPositionVerifierStateReady(");
    const auto verifier_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingAllPositionVerifierStateReadyForObservation(",
        "void DeviceGraphOrchestrator::clearPendingAllPositionVerifierStateReady()");
    const auto checkpoint_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::\n"
        "        waitForPendingLivePrefixCheckpointReadyForObservation(",
        "bool DeviceGraphOrchestrator::waitForPendingLivePrefixCheckpointReady(");
    const auto mailbox_observation_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateMailboxForObservation(",
        "bool DeviceGraphOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const");
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
    const auto compact_live_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(
            live_observation_body));
    const auto compact_shifted_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(shifted_observation_body));
    const auto compact_verifier_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(verifier_observation_body));
    const auto compact_checkpoint_observation =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(
            checkpoint_observation_body));
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
    EXPECT_NE(compact_header.find("waitForPendingLivePrefixCheckpointReadyForObservation("),
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
    EXPECT_NE(compact_graph_observation.find(
                  "waitForForwardGraphOutputReady("
                  "observation_stream,observation_role,"
                  "ForwardGraphOutputKind::Any,consumer)"),
              std::string::npos)
        << "Ordinary prefill/decode graphs mutate live KV/GDN state even when "
           "they publish no role-specific MTP logits handoff.";
    EXPECT_EQ(compact_graph_observation.find("lastForwardExecution("),
              std::string::npos)
        << "Observation must never retain or consult a cache-owned producer stream.";
    EXPECT_EQ(compact_graph_observation.find("insertStreamDependency("),
              std::string::npos)
        << "Observation consumes the event recorded while the producer stream was valid.";
    EXPECT_EQ(compact_graph_observation.find("synchronizeStream"),
              std::string::npos);
    EXPECT_EQ(compact_graph_observation.find("synchronizeDevice"),
              std::string::npos);
    EXPECT_NE(compact_live_observation.find(
                  "waitForPendingLivePrefixCheckpointReadyForObservation("),
              std::string::npos)
        << "A new pooled checkpoint writer must observe the previous copy before slot reuse.";
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
    EXPECT_NE(compact_checkpoint_observation.find("streamWaitEvent("),
              std::string::npos);
    EXPECT_EQ(compact_checkpoint_observation.find(
                  "clearPendingLivePrefixCheckpointReady()"),
              std::string::npos)
        << "Checkpoint observation must preserve the event for the next live-state mutation.";
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
    const auto terminal_restore_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::restoreLiveCheckpointTerminalHidden(",
        "bool DeviceGraphOrchestrator::importMTPCheckpointTerminalHidden(");

    const auto compact_capture = removeAsciiWhitespace(stripCommentsAndStringLiterals(capture_body));
    const auto compact_restore = removeAsciiWhitespace(stripCommentsAndStringLiterals(restore_body));
    const auto compact_truncate = removeAsciiWhitespace(stripCommentsAndStringLiterals(truncate_body));
    const auto compact_prepare = removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_record = removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));
    const auto compact_snapshot_wait = removeAsciiWhitespace(stripCommentsAndStringLiterals(snapshot_wait_body));
    const auto compact_source_wait = removeAsciiWhitespace(stripCommentsAndStringLiterals(source_wait_body));
    const auto compact_terminal_restore =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(terminal_restore_body));

    EXPECT_NE(compact_snapshot.find("std::shared_ptr<void>ready_event;"), std::string::npos);
    EXPECT_NE(compact_snapshot.find("structDeviceKVSequenceStateCheckpoint"), std::string::npos);
    EXPECT_NE(compact_snapshot.find(
                  "std::vector<DeviceKVSequenceStateCheckpoint>device_sequence_state_checkpoints;"),
              std::string::npos)
        << "A logical GPU checkpoint must own every opaque device metadata bank.";
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

    EXPECT_EQ(compact_record.find("backend->createEvent("), std::string::npos)
        << "Live checkpoint event creation belongs to setup, never decode.";
    EXPECT_NE(compact_record.find("handle.live_checkpoint_ready_event"),
              std::string::npos)
        << "Capture must record the event paired with its retained storage slot.";
    EXPECT_NE(compact_record.find("snapshot->mtp_blocks"),
              std::string::npos)
        << "A shifted-cache-only checkpoint still needs a preallocated completion event.";
    EXPECT_NE(compact_record.find("backend->recordEvent("), std::string::npos);
    EXPECT_NE(compact_record.find("snapshot->ready_event=event;"), std::string::npos);
    EXPECT_NE(compact_record.find("live_prefix_checkpoint_ready_.event=std::move(event);"),
              std::string::npos);
    EXPECT_NE(compact_snapshot_wait.find("streamWaitEvent("), std::string::npos);
    EXPECT_NE(compact_source_wait.find("streamWaitEvent("), std::string::npos);
    EXPECT_NE(compact_source_wait.find("clearPendingLivePrefixCheckpointReady();"), std::string::npos);

    EXPECT_NE(compact_capture.find("&handle,false,stream)"), std::string::npos)
        << "Hybrid logical checkpoints should stay async and be ordered by events, not host syncs.";
    EXPECT_EQ(compact_capture.find("payload_fallback_mtp_hybrid_state_failed"),
              std::string::npos)
        << "Shifted recurrent state is a normal logical-checkpoint payload, not a reason to replay KV through host RAM.";
    EXPECT_NE(compact_capture.find(
                  "snapshot.mtp_blocks.push_back(std::move(handle));"),
              std::string::npos)
        << "Every shifted hybrid bank must retain its own device checkpoint handle.";
    EXPECT_NE(compact_capture.find("captureDeviceSequenceStateCheckpoint("),
              std::string::npos)
        << "Live GPU rollback must capture canonical ring metadata on device.";
    EXPECT_NE(compact_restore.find("restoreDeviceSequenceStateCheckpoint("),
              std::string::npos)
        << "Live GPU rollback must restore canonical ring metadata on device.";
    EXPECT_EQ(compact_capture.find(
                  "deviceResidentLogicalTokenCountForObservation("),
              std::string::npos)
        << "Production rollback may not observe a device count through the host.";
    EXPECT_EQ(compact_capture.find(
                  "deviceResidentShiftedMTPKVTokenCountForObservation("),
              std::string::npos)
        << "Production rollback may not observe shifted-cache counts through the host.";
    EXPECT_EQ(compact_capture.find("deviceToHost"), std::string::npos);
    EXPECT_EQ(compact_capture.find("synchronizeStream("), std::string::npos);
    EXPECT_EQ(compact_capture.find("synchronizeDevice("), std::string::npos);
    EXPECT_EQ(compact_capture.find("captureLivePrefixState("), std::string::npos)
        << "The device checkpoint cannot fall back to host payload replay.";
    EXPECT_NE(compact_restore.find(
                  "importHybridPrefixPayload(*cache,*shifted_hybrid_handle,"
                  "seq_idx,false,false,stream)"),
              std::string::npos)
        << "Logical restore must import every shifted recurrent bank after truncation.";
    EXPECT_NE(compact_restore.find(
                  "snapshot.mtp_blocks.begin(),snapshot.mtp_blocks.end()"),
              std::string::npos)
        << "Restore publication must retain shifted checkpoint sources until the import stream completes.";
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
    EXPECT_NE(compact_capture.find("backend->deviceCopyAsync("), std::string::npos)
        << "GPU terminal-hidden rollback capture must remain an asynchronous "
           "device-to-device copy on the checkpoint stream.";
    EXPECT_EQ(compact_capture.find("TransferEngine::instance().downloadFull("),
              std::string::npos)
        << "A live GPU checkpoint must never materialize terminal hidden on the host.";
    EXPECT_NE(terminal_restore_body.find(
                  "gpu_live_checkpoint_requires_device_terminal_hidden"),
              std::string::npos)
        << "GPU live-checkpoint restore must reject a host-only terminal row.";
    EXPECT_NE(compact_terminal_restore.find("backend->deviceCopyAsync("),
              std::string::npos);
    EXPECT_EQ(compact_terminal_restore.find("TransferEngine::instance().uploadFull("),
              std::string::npos)
        << "A live GPU checkpoint must never upload a stale host mirror.";
    EXPECT_NE(compact_terminal_restore.find(
                  "TransferEngine::publishDeviceWrite("),
              std::string::npos)
        << "The D2D destination must publish device ownership and its exact "
           "producer-stream event through TransferEngine.";
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
        "joinPublishedLiveStateHandoffs(",
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
        "bool DeviceGraphOrchestrator::ensureLiveCheckpointStorage");
    const auto checkpoint_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(",
        "bool DeviceGraphOrchestrator::restoreLivePrefixState");
    const auto device_publication_lifecycle_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::finalizeMTPSpeculativeStatePublicationLaunch(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const auto record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordLivePrefixMutationReady(",
        "bool DeviceGraphOrchestrator::joinPublishedLiveStateHandoffs(");
    const auto event_preflight_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareLivePrefixMutationReadyEvent(",
        "bool DeviceGraphOrchestrator::recordLivePrefixMutationReady(");
    const auto published_join_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::joinPublishedLiveStateHandoffs(",
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
    const auto logical_state_writer_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateRowReuse(",
        "bool DeviceGraphOrchestrator::\n"
        "        recordDeviceResidentLogicalSequenceStateReadCompletion(");
    const auto reset_inference_body = sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override");

    const auto compact_restore = removeAsciiWhitespace(stripCommentsAndStringLiterals(restore_body));
    const auto compact_restore_with_labels = removeAsciiWhitespace(restore_body);
    const auto compact_truncate = removeAsciiWhitespace(stripCommentsAndStringLiterals(truncate_body));
    const auto compact_sidecar = removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_body));
    const auto compact_prepare = removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_probe = removeAsciiWhitespace(stripCommentsAndStringLiterals(probe_body));
    const auto compact_payload = removeAsciiWhitespace(stripCommentsAndStringLiterals(payload_body));
    const auto compact_checkpoint = removeAsciiWhitespace(stripCommentsAndStringLiterals(checkpoint_body));
    const auto compact_device_publication_lifecycle =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(
                device_publication_lifecycle_body));
    const auto compact_event_preflight =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(event_preflight_body));
    const auto compact_record = removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));
    const auto compact_published_join =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(published_join_body));
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
    const auto compact_logical_state_writer =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(logical_state_writer_body));
    const auto compact_reset_inference =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(reset_inference_body));

    EXPECT_NE(compact_header.find("PendingLivePrefixMutationReadyState"), std::string::npos);
    EXPECT_NE(compact_header.find("mutablePendingLivePrefixMutationReadyStatelive_prefix_mutation_ready_;"),
              std::string::npos);
    EXPECT_NE(
        compact_header.find(
            "recordLivePrefixMutationReady(void*producer_stream,constchar*producer_name,std::vector<PrefixBlockHandle>retained_payload_sources={},std::vector<std::shared_ptr<void>>retained_device_sources={})"),
              std::string::npos);
    EXPECT_NE(
        compact_header.find(
            "joinPublishedLiveStateHandoffs(void*consumer_stream,DeviceTimelineRoleconsumer_role,constchar*consumer_name)const"),
        std::string::npos);
    EXPECT_EQ(
        compact_header.find("waitForPendingLivePrefixMutationReady("),
        std::string::npos);
    EXPECT_EQ(
        compact_header.find(
            "waitForPendingLivePrefixMutationReadyForObservation("),
        std::string::npos);
    EXPECT_NE(compact_header.find("waitForPendingLiveGraphProducersBeforePrefixMutation(void*mutation_stream,constchar*mutation_name)"),
              std::string::npos);
    EXPECT_NE(compact_header.find(
                  "recordRestoredDeviceResidentMTPTransaction(void*producer_stream,constchar*producer_name)"),
              std::string::npos)
        << "GPU restore must advance the persistent shifted-cache transaction fence.";

    EXPECT_NE(compact_event_preflight.find("backend->createEvent("), std::string::npos)
        << "The completion event must exist before asynchronous restore payload reads begin.";
    EXPECT_NE(
        compact_record.find(
            "DeviceTimelinePoint::LivePrefixMutationReady"),
        std::string::npos);
    EXPECT_NE(compact_record.find(".publish("), std::string::npos);
    EXPECT_NE(compact_record.find("live_prefix_mutation_ready_"), std::string::npos);
    EXPECT_NE(compact_record.find("std::terminate()"), std::string::npos)
        << "A submitted restore payload may not outlive its owner when event publication fails.";
    EXPECT_NE(
        compact_published_join.find("enqueuePublishedWait("),
        std::string::npos);
    EXPECT_NE(
        compact_published_join.find(
            "if(consumes_publications){"),
        std::string::npos);
    EXPECT_NE(
        compact_published_join.find(
            "caseDeviceTimelineRole::TargetSampler:"),
        std::string::npos)
        << "The target sampler must observe restore/publication events without "
           "consuming the ownership needed by the following graph.";
    EXPECT_NE(
        compact_published_join.find(
            "clearPendingLivePrefixMutationReady();"),
        std::string::npos);
    EXPECT_EQ(
        compact_published_join.find("producer_stream==consumer_stream"),
        std::string::npos)
        << "Durable publication joins must never depend on raw stream identity.";

    EXPECT_NE(compact_prepare.find("live_prefix_mutation_ready_.valid"), std::string::npos);
    EXPECT_NE(
        compact_sidecar.find(
            "joinPublishedLiveStateHandoffs("
            "sidecar_dynamic_stream,"
            "DeviceTimelineRole::MTPSidecarGraph,"),
        std::string::npos)
        << "The MTP sidecar must identify its role to the shared join API.";
    EXPECT_NE(
        compact_prepare.find(
            "joinPublishedLiveStateHandoffs("
            "execution_stream,"
            "DeviceTimelineRole::MainForwardGraph,"),
        std::string::npos)
        << "The main graph must identify its consuming role to the shared join API.";
    EXPECT_NE(
        compact_restore.find(
            "joinPublishedLiveStateHandoffs("
            "stream,DeviceTimelineRole::PrefixRestoreMutation,"),
        std::string::npos)
        << "Prefix restore must consume the old live-state timeline through the typed API.";
    EXPECT_NE(
        compact_truncate.find(
            "joinPublishedLiveStateHandoffs("
            "stream,DeviceTimelineRole::PrefixRestoreMutation,"),
        std::string::npos)
        << "Prefix truncate must consume the old live-state timeline through the typed API.";
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
                  "waitForDeviceResidentLogicalSequenceStateRowReuse("),
              std::string::npos)
        << "A new compact outcome must begin the exclusive replacement transaction.";
    EXPECT_EQ(compact_prepare_publication_metadata.find(
                  "clearDeviceResidentLogicalSequenceStateMailbox()"),
              std::string::npos)
        << "Metadata preparation must not bypass the typed replacement transaction.";
    EXPECT_NE(compact_logical_state_writer.find(
                  "device_resident_logical_sequence_state_mailbox_.clear()"),
              std::string::npos)
        << "The exclusive writer transaction must retire the old mailbox after reader fan-in.";
    EXPECT_EQ(compact_prepare_publication_metadata.find(
                  "retireDeviceResidentMTPTransaction"),
              std::string::npos)
        << "Metadata preparation must preserve shifted-cache ownership through initial/suffix commits.";
    EXPECT_NE(compact_reset_inference.find(
                  "clearDeviceResidentLogicalSequenceStateMailbox()"),
              std::string::npos);
    EXPECT_NE(compact_reset_inference.find(
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
        "joinPublishedLiveStateHandoffs(",
        "state_.kv_cache->truncateSequence(",
        "A restore must wait for any older async prefix mutation before overwriting live state.");
    expectNeedleBefore(
        compact_restore,
        "waitForPendingLiveGraphProducersBeforePrefixMutation(",
        "state_.kv_cache->truncateSequence(",
        "A restore must order after pending forward/verifier graph producers before overwriting KV/GDN state.");
    expectNeedleBefore(
        compact_restore,
        "prepareLivePrefixMutationReadyEvent(",
        "state_.kv_cache->truncateSequence(",
        "A restore must preflight its completion event before submitting asynchronous payload reads.");
    expectNeedleBefore(
        compact_restore,
        "handleLivePrefixReplayStateAfterMutation(",
        "recordLivePrefixMutationReady(",
        "Restore readiness must be recorded after the live-state epoch changes.");
    expectNeedleBefore(
        compact_restore_with_labels,
        "recordLivePrefixMutationReady(stream,\"restore_logical_checkpoint\",std::move(retained_snapshot_sources),std::move(retained_device_sequence_sources))",
        "recordRestoredDeviceResidentMTPTransaction(stream,\"restore_logical_checkpoint_shifted_kv\")",
        "Logical restore must advance its shifted-cache transaction after recording the restore mutation.");
    expectNeedleBefore(
        compact_restore_with_labels,
        "recordLivePrefixMutationReady(stream,\"restore_payload_checkpoint_zero\")",
        "recordRestoredDeviceResidentMTPTransaction(stream,\"restore_payload_checkpoint_zero_shifted_kv\")",
        "Zero-token restore must advance its shifted-cache transaction after recording the restore mutation.");
    expectNeedleBefore(
        compact_restore_with_labels,
        "recordLivePrefixMutationReady(stream,\"restore_payload_checkpoint\",std::move(retained_snapshot_sources))",
        "recordRestoredDeviceResidentMTPTransaction(stream,\"restore_payload_checkpoint_shifted_kv\")",
        "Payload restore must advance its shifted-cache transaction after recording the restore mutation.");
    expectNeedleBefore(
        compact_truncate,
        "joinPublishedLiveStateHandoffs(",
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
    EXPECT_NE(
        compact_device_publication_lifecycle.find(
            "recordAcceptedSpecPublicationReady("),
        std::string::npos);
    EXPECT_EQ(
        compact_device_publication_lifecycle.find(
            "recordLivePrefixMutationReady("),
        std::string::npos)
        << "Accepted-state publication and prefix mutation are distinct timeline producers.";
    expectNeedleBefore(
        compact_device_publication_lifecycle,
        "recordDeviceResidentLogicalSequenceStateMailbox(",
        "recordAcceptedSpecPublicationReady(",
        "Accepted-state readiness must be recorded after the logical mailbox publication.");
    expectNeedleBefore(
        compact_published_join,
        "DeviceTimelinePoint::AcceptedSpecPublicationReady",
        "DeviceTimelinePoint::LivePrefixMutationReady",
        "The shared typed join must queue accepted publication before prefix mutation.");
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
    EXPECT_NE(compact_helper.find("shifted_mtp_kv_ready_.producer_stream=nullptr"),
              std::string::npos);
    EXPECT_EQ(compact_helper.find("shifted_mtp_kv_ready_.event.reset()"),
              std::string::npos)
        << "Boundary cleanup invalidates timeline ownership but preserves the preallocated event.";
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
                             "::speculativeStateSlotsBufferName() const")));
        EXPECT_NE(stable_id.find("params_.workspace_namespace"), std::string::npos);
        EXPECT_NE(stable_id.find("role_prefix+"), std::string::npos)
            << "Capture workspace ids must include the graph-role namespace.";
    }

    for (const auto &stage : {gdn_stage, conv_stage})
    {
        EXPECT_EQ(stage.find("device_value_uploaded"), std::string::npos);
        EXPECT_EQ(stage.find("host_effective_seq_len"), std::string::npos);
        EXPECT_EQ(stage.find("effective_seq_len_scalar"), std::string::npos)
            << "Padded GPU GDN replay must read request lengths from the "
               "persistent device request metadata allocation";
        EXPECT_NE(
            stage.find("request_seq_lens_device"),
            std::string::npos);
    }

    EXPECT_NE(qwen35.find("conv_params.workspace_namespace = workspace_namespace"),
              std::string::npos);
    EXPECT_NE(qwen35.find("rec_params.workspace_namespace = workspace_namespace"),
              std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, ResidentStochasticOutcomeControlIsDeviceAuthoritative)
{
    const auto runner_api = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto orchestrator = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto descriptor = sliceBetween(
        runner_api,
        "struct DeviceStochasticBatchOutcomeRequest",
        "struct DeviceMTPVerifierInputBatchRequest");
    const auto reducer = sliceBetween(
        orchestrator,
        "bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(",
        "bool DeviceGraphOrchestrator::forward_batch(");

    EXPECT_EQ(
        descriptor.find("leading_committed_output_count"),
        std::string::npos)
        << "Resident stochastic descriptors must not expose a host mirror of "
           "transaction carry; DeviceGenerationControlIndex is authoritative.";
    EXPECT_EQ(
        reducer.find("max_state_commit_rows_device"),
        std::string::npos)
        << "The resident reducer must read its commit budget from the complete "
           "device generation controller, not a separately threaded pointer.";
    EXPECT_EQ(
        reducer.find("leading_committed_output_count"),
        std::string::npos)
        << "The resident reducer must not accept, validate, or log a host carry scalar.";
    EXPECT_NE(
        reducer.find(
            "enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls("),
        std::string::npos)
        << "Every resident summary mode must consume the device generation controller.";
    EXPECT_EQ(
        reducer.find("enqueueSummarizeGreedySpeculativeVerifyBatch("),
        std::string::npos)
        << "The host-control summary API is not a valid resident production route.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, TerminalDeviceGenerationBridgeIsSingleAndPreallocated)
{
    const auto orchestrator = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto rank_orchestrator = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
    const auto terminal_bridge = sliceBetween(
        orchestrator,
        "bool DeviceGraphOrchestrator::finishDeviceResidentStochasticGeneration(",
        "bool DeviceGraphOrchestrator::joinPriorDeviceWorkForRequestStateReset(");
    const auto rank_bridge = sliceBetween(
        rank_orchestrator,
        "bool RankOrchestrator::finishDeviceResidentStochasticGeneration(",
        "bool RankOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(");

    EXPECT_NE(
        terminal_bridge.find("consumeDeviceGenerationStateReady("),
        std::string::npos)
        << "Terminal D2H must consume the exact final controller publication.";
    EXPECT_NE(
        terminal_bridge.find("DeviceTimelineRole::HostResultBridge"),
        std::string::npos);
    EXPECT_EQ(
        countOccurrences(terminal_bridge, "deviceToHostOnStream("),
        2u)
        << "Response tokens and controller words must be queued together.";
    EXPECT_EQ(
        countOccurrences(terminal_bridge, "synchronizeStream("),
        1u)
        << "The complete generation has exactly one host-blocking boundary.";
    EXPECT_EQ(
        terminal_bridge.find("deviceToHostFast("),
        std::string::npos);
    EXPECT_EQ(
        terminal_bridge.find("createStream("),
        std::string::npos)
        << "The terminal path must reuse its initialization-owned result stream.";
    EXPECT_EQ(
        terminal_bridge.find("allocatePinned("),
        std::string::npos)
        << "Pinned result storage must be allocated before inference.";
    EXPECT_NE(
        terminal_bridge.find(
            "device_generation_storage_.active_request_count = 0"),
        std::string::npos)
        << "Only successful terminal validation may release request admission.";
    EXPECT_EQ(
        terminal_bridge.find("handleLivePrefixReplayStateAfterMutation("),
        std::string::npos)
        << "The terminal bridge is a result observer. The captured parent graph "
           "already committed live state, so host terminal parsing must not "
           "invent a second mutation or retire the final device mailbox.";

    EXPECT_NE(
        rank_bridge.find("finishDeviceResidentStochasticGeneration("),
        std::string::npos);
    EXPECT_NE(
        rank_bridge.find("participant_results[participant_index].requests !="),
        std::string::npos)
        << "Mirrored participants must prove identical terminal ledgers.";
    EXPECT_NE(
        rank_bridge.find("describeTerminalLedgerMismatch("),
        std::string::npos)
        << "A mirrored-ledger failure must identify its first token or control "
           "difference rather than emitting an opaque struct mismatch.";
    EXPECT_EQ(
        rank_bridge.find("collective"),
        std::string::npos)
        << "Terminal validation must not add a tiny rank collective.";
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
     * The served stochastic stream must build every target row plus the bonus
     * row in one captured target-preparation graph.  The final row count is
     * therefore compare_rows + 1, while bonus_row names the last compact slot
     * consumed by the outcome graph.  Neither the retired eager row builder nor
     * the non-equivalent processed-logit bonus sampler may re-enter this path.
     */
    EXPECT_NE(compact.find("constintbonus_row=compare_rows;"),
              std::string::npos);
    EXPECT_NE(compact.find(
                  "buildCapturedStochasticVerifierTargetDistributions(compare_rows+1,"),
              std::string::npos)
        << "Production stochastic MTP must capture the compact target+bonus "
           "row transaction as one strict device graph.";
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
    EXPECT_EQ(compact.find("buildStochasticDistributionsOnDevice("),
              std::string::npos)
        << "The served verifier transaction must not escape the captured "
           "target-preparation graph through the retired eager row builder.";
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
    EXPECT_NE(compact_handle.find(
                  "boolmirrored_local_tp_locally_complete=false"),
              std::string::npos)
        << "The compact outcome handle must prove that this participant ran "
           "the complete mirrored verifier outcome transaction locally.";
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
    EXPECT_NE(compact_verify.find(
                  "acquirePersistentMTPOutcomeReadyEvent("),
              std::string::npos)
        << "The verifier must borrow a setup-owned response event instead of "
           "allocating backend resources during decode.";
    EXPECT_EQ(compact_verify.find("backend->createEvent("),
              std::string::npos);
    EXPECT_EQ(compact_verify.find("backend->createTimingEvent("),
              std::string::npos);
    EXPECT_EQ(compact_verify.find("backend->destroyEvent("),
              std::string::npos);
    EXPECT_NE(compact_verify.find(
                  "DeviceEventEdge::at(DeviceTimelinePoint::CompactSpeculativeResponseReady).from(DeviceTimelineRole::VerifierSummary).publish(*backend,state_.device_id,response_ready_event.get(),stream)"),
              std::string::npos)
        << "The verifier must publish response readiness through the typed "
           "cross-graph execution timeline.";
    EXPECT_NE(compact_verify.find(
                  "out_handle->response_ready_event=std::move(response_ready_event)"),
              std::string::npos);

    EXPECT_EQ(compact_bridge.find("backend->createStream("),
              std::string::npos)
        << "The compact D2H bridge must not allocate its explicit stream in "
           "the response hot path.";
    EXPECT_NE(compact_bridge.find("stochastic_outcome_response_bridge_stream_"),
              std::string::npos)
        << "The compact D2H bridge must reuse a persistent explicit stream; "
           "per-step HIP stream create/destroy is visible in fixed-depth MTP.";
    EXPECT_EQ(compact_bridge.find("std::shared_ptr<void>owned_copy_stream"),
              std::string::npos)
        << "The compact D2H bridge must not allocate a throwaway stream owner "
           "inside the decode hot path.";
    EXPECT_EQ(compact_bridge.find(
                  "backend->waitForEvent(handle.response_ready_event.get(),state_.device_id.gpu_ordinal())"),
              std::string::npos)
        << "The response bridge must not block the host before submitting its "
           "compact D2H copy.";
    EXPECT_NE(dgo_source.find(
                  "\"stochastic_request_batch_summary_response_dependency_enqueue\""),
              std::string::npos)
        << "Bridge accounting must identify the nonblocking event dependency "
           "separately from final host result materialization.";
    EXPECT_NE(compact_bridge.find(
                  "DeviceEventEdge::at(DeviceTimelinePoint::CompactSpeculativeResponseReady).from(DeviceTimelineRole::VerifierSummary).to(DeviceTimelineRole::HostResultBridge).enqueueWait(*backend,state_.device_id,handle.response_ready_event.get(),handle.stream,copy_stream)"),
              std::string::npos)
        << "The explicit D2H bridge stream must consume the participant-local "
           "verifier-summary edge directly; rank-owned response publication no "
           "longer exists for a mirrored head.";
    EXPECT_NE(compact_bridge.find(
                  "backend->synchronizeStream(copy_stream,state_.device_id.gpu_ordinal())"),
              std::string::npos);
    EXPECT_EQ(compact_bridge.find("backend->synchronizeStream(handle.stream"),
              std::string::npos)
        << "Synchronizing the producer stream reintroduces a publication D2H barrier.";
}

/**
 * @brief Forbids backend event allocation from every resident-MTP hot path.
 *
 * Events are durable execution-graph edges, not per-token scratch objects.
 * Setup preallocates fixed readiness events and bounded response/timing pools;
 * decode may only record or wait on those owners. Keeping the complete method
 * list here makes a newly introduced event allocation fail the source policy
 * gate before it can become an intermittent graph-capture or latency defect.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, ResidentMTPHotPathsUseOnlyPreallocatedEvents)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto initialization_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::initializePersistentMTPDeviceEvents()",
            "DeviceGraphOrchestrator::acquirePersistentMTPOutcomeReadyEvent(")));

    EXPECT_NE(initialization_body.find("allocatePersistentDeviceEvent("),
              std::string::npos)
        << "Persistent MTP readiness and response events must be allocated at setup.";
    EXPECT_NE(initialization_body.find("allocatePersistentDeviceTimingEvent("),
              std::string::npos)
        << "Persistent MTP PerfStats events must be allocated at setup.";

    const std::array<std::tuple<const char *, const char *, const char *>, 10>
        hot_paths = {{
            {
                "bool DeviceGraphOrchestrator::recordDeviceResidentMTPTransactionMutation(",
                "bool DeviceGraphOrchestrator::recordRestoredDeviceResidentMTPTransaction(",
                "resident MTP transaction publication",
            },
            {
                "bool DeviceGraphOrchestrator::recordDeviceResidentLogicalSequenceStateMailbox(",
                "bool DeviceGraphOrchestrator::admitDeviceResidentLogicalSequenceStateRead(",
                "resident logical-state publication",
            },
            {
                "bool DeviceGraphOrchestrator::recordStochasticDraftSampleReady(",
                "bool DeviceGraphOrchestrator::recordStochasticTargetSampleReady(",
                "draft sample publication",
            },
            {
                "bool DeviceGraphOrchestrator::recordStochasticTargetSampleReady(",
                "bool DeviceGraphOrchestrator::beginPendingGpuTimingMeasurement(",
                "target sample publication",
            },
            {
                "bool DeviceGraphOrchestrator::beginPendingGpuTimingMeasurement(",
                "void DeviceGraphOrchestrator::finishPendingGpuTimingMeasurement(",
                "deferred GPU timing publication",
            },
            {
                "bool DeviceGraphOrchestrator::recordAllPositionVerifierStateReady(",
                "bool DeviceGraphOrchestrator::waitForPendingAllPositionVerifierStateReady(",
                "all-position verifier publication",
            },
            {
                "bool DeviceGraphOrchestrator::recordAcceptedSpecPublicationReady(",
                "void DeviceGraphOrchestrator::clearPendingAcceptedSpecPublicationReady()",
                "accepted-state publication",
            },
            {
                "bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(",
                "bool DeviceGraphOrchestrator::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(",
                "single-request greedy compact outcome",
            },
            {
                "bool DeviceGraphOrchestrator::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(",
                "bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDevice(",
                "request-batch greedy compact outcome",
            },
            {
                "bool DeviceGraphOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(",
                "bool DeviceGraphOrchestrator::copyDeviceSpeculativeOutcomesToHost(",
                "request-batch stochastic compact outcome",
            },
        }};

    for (const auto &[begin_marker, end_marker, label] : hot_paths)
    {
        const auto body = removeAsciiWhitespace(
            stripCommentsAndStringLiterals(
                sliceBetween(source, begin_marker, end_marker)));
        EXPECT_EQ(body.find("createEvent("), std::string::npos)
            << label << " must not allocate an event";
        EXPECT_EQ(body.find("createTimingEvent("), std::string::npos)
            << label << " must not allocate a timing event";
        EXPECT_EQ(body.find("destroyEvent("), std::string::npos)
            << label << " must not retire backend event resources";
    }
}

/**
 * @brief Keep rooted collective execution free of workspace lookup and allocation.
 *
 * Rooted activation reduction and its rebalance sidebands are captured as one
 * fixed-stream transaction. Symbolic workspace names therefore have to become
 * stable device pointers during graph binding, before execute() can launch the
 * first collective. Resolving names or constructing descriptor vectors after
 * that launch would both allocate in a capture-sensitive path and permit a bad
 * sideband binding to split collective order across participants.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     RootedCollectivePrebindsSidebandsOutsideExecution)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/compute_stages/stages/TPAllreduceStage.cpp");
    const auto header = readFile(
        repoRoot() /
        "src/v2/execution/compute_stages/stages/TPAllreduceStage.h");

    const auto execute_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool TPLocalRootedCollectiveStage::execute(IDeviceContext *ctx)",
            "bool TPLocalRootedCollectiveStage::supportsBackend(")));
    const auto bind_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "void TPLocalRootedCollectiveStage::bindWorkspace(",
            "void TPLocalRootedCollectiveStage::unbindWorkspace()")));
    const auto compact_header = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(header));

    EXPECT_NE(
        compact_header.find(
            "std::vector<LocalTPCollectiveSidebandBuffer>bound_sidebands_;"),
        std::string::npos)
        << "The stage needs persistent prebound descriptors with graph-stable addresses.";
    EXPECT_NE(bind_body.find("bound_workspace_->getBuffer(buffer_name)"),
              std::string::npos)
        << "Workspace names must be resolved during binding.";
    EXPECT_NE(bind_body.find("bound_sidebands_.push_back("),
              std::string::npos)
        << "Binding must materialize the complete fixed sideband table.";

    EXPECT_NE(execute_body.find(
                  "collectiveSidebandSpanOnStream(bound_sidebands_,"),
              std::string::npos)
        << "Execution must submit the already-bound descriptor span.";
    EXPECT_EQ(execute_body.find("getBuffer("), std::string::npos);
    EXPECT_EQ(execute_body.find("std::vector<"), std::string::npos);
    EXPECT_EQ(execute_body.find(".reserve("), std::string::npos);
    EXPECT_EQ(execute_body.find(".push_back("), std::string::npos);
    EXPECT_EQ(execute_body.find("malloc("), std::string::npos);
    EXPECT_EQ(execute_body.find("cudaMalloc("), std::string::npos);
    EXPECT_EQ(execute_body.find("hipMalloc("), std::string::npos);
    EXPECT_EQ(execute_body.find("synchronize"), std::string::npos);
}

/**
 * @brief Locks in bounded profiler-event reuse and local MTP ownership.
 *
 * Every mirrored LocalTP participant owns a complete compact outcome. Per-kernel
 * timing events are queried and reclaimed without a stream/device synchronization
 * before the fixed pool can be reused. Rank orchestration validates those local
 * ownership contracts but may not move, compare, or republish compact results.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, MirroredMTPTimingEventsAndOutcomesRemainParticipantLocal)
{
    const auto dgo_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto rank_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
    const auto reclaim_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            dgo_source,
            "DeviceGraphOrchestrator::reclaimCompletedGpuTimingMeasurementsNonblocking(",
            "bool DeviceGraphOrchestrator::waitForStochasticDraftSampleReadyRange(")));
    const auto acquire_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            dgo_source,
            "bool DeviceGraphOrchestrator::acquirePersistentMTPGpuTimingEvents(",
            "bool DeviceGraphOrchestrator::recordShiftedMTPKVReady(")));
    const auto validation_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            rank_source,
            "bool RankOrchestrator::validateMirroredLocalTPChildOutcomesComplete(",
            "bool RankOrchestrator::buildRankStochasticDistributionFromLocalTP(")));

    EXPECT_NE(reclaim_body.find("queryEventChecked("), std::string::npos)
        << "Timing reuse must be gated by a nonblocking completion query.";
    EXPECT_NE(reclaim_body.find("eventElapsedTimeMs("), std::string::npos)
        << "Completed kernel timings must still become training/perf evidence.";
    EXPECT_EQ(reclaim_body.find("synchronizeStream("), std::string::npos);
    EXPECT_EQ(reclaim_body.find("synchronizeDevice("), std::string::npos);
    EXPECT_EQ(reclaim_body.find("waitForEvent("), std::string::npos);
    EXPECT_NE(acquire_body.find(
                  "reclaimCompletedGpuTimingMeasurementsNonblocking(backend)"),
              std::string::npos)
        << "Every fixed-pool borrow must first retire completed measurements.";

    EXPECT_NE(validation_body.find(
                  "child.mirrored_local_tp_locally_complete"),
              std::string::npos)
        << "Rank orchestration must fail closed unless every child proves local completion.";
    EXPECT_NE(validation_body.find("!child.stream||!child.response_ready_event"),
              std::string::npos)
        << "Every local outcome must retain its exact producer stream and event.";
    EXPECT_EQ(validation_body.find("collectiveSidebands"), std::string::npos);
    EXPECT_EQ(validation_body.find("deviceToHost"), std::string::npos);
    EXPECT_EQ(validation_body.find("hostToDevice"), std::string::npos);
    EXPECT_EQ(validation_body.find("publishRank"), std::string::npos);
    EXPECT_EQ(validation_body.find("getBackendFor("), std::string::npos)
        << "Rank orchestration must not manipulate child-owned event resources.";
    EXPECT_EQ(validation_body.find("synchronizeStream("), std::string::npos);
    EXPECT_EQ(validation_body.find("synchronizeDevice("), std::string::npos);
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
    const auto greedy_consumer_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(",
        "bool DeviceGraphOrchestrator::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(");
    const auto graph_outcome_stage = readFile(
        repoRoot() /
        "src/v2/execution/compute_stages/stages/MTPVerifierOutcomeStage.cpp");
    const auto verifier_metadata_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(",
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(");
    const auto device_token_input_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::prepareDeviceTokenInputsForForwardGraphExecution(",
        "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(");
    const auto greedy_runner_body = sliceBetween(
        runner_source,
        "const bool use_greedy_device_batch_outcome =",
        "else\n                {");
    const auto cuda_greedy_summary_kernel = sliceBetween(
        cuda_sampling,
        "cuda_summarize_greedy_speculative_verify_batch_device_controls_kernel(",
        "__global__ void cuda_derive_speculative_publication_metadata_kernel(");
    const auto rocm_greedy_summary_kernel = sliceBetween(
        rocm_sampling,
        "rocm_summarize_greedy_speculative_verify_batch_device_controls_kernel(",
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

    const auto compact_greedy_consumer = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(greedy_consumer_body));
    const auto compact_graph_outcome_stage = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(graph_outcome_stage));
    const auto compact_verifier_metadata = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(verifier_metadata_body));
    const auto compact_device_token_input = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(device_token_input_body));
    EXPECT_NE(
        compact_greedy_consumer.find(
            "GreedyVerifierOutcomeGraphState::Produced"),
        std::string::npos);
    EXPECT_NE(
        compact_greedy_consumer.find(
            "consumePendingLogitsStream("
            "PendingLogitsStreamRole::AllPositionVerifier"),
        std::string::npos);
    EXPECT_NE(
        compact_greedy_consumer.find(
            "forward_graph_output_ready_.event"),
        std::string::npos);
    EXPECT_EQ(
        compact_greedy_consumer.find(
            "enqueueArgmaxF32BatchedRowsDevice("),
        std::string::npos)
        << "The resident consumer must not launch a post-graph argmax.";
    EXPECT_EQ(
        compact_greedy_consumer.find(
            "enqueueSummarizeGreedySpeculativeVerifyBatch"),
        std::string::npos)
        << "The resident consumer must not launch a post-graph summary.";
    EXPECT_EQ(
        compact_greedy_consumer.find("explicitGPUStreamForOperation("),
        std::string::npos)
        << "A missing graph stream is fatal, not a fresh-stream fallback.";
    EXPECT_EQ(
        compact_greedy_consumer.find("hostToDeviceOnStream("),
        std::string::npos);
    EXPECT_NE(
        compact_graph_outcome_stage.find(
            "enqueueArgmaxF32BatchedRowsWithMTPPenaltiesDevice("),
        std::string::npos);
    EXPECT_NE(
        compact_graph_outcome_stage.find(
            "enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls("),
        std::string::npos);
    EXPECT_NE(
        compact_graph_outcome_stage.find(
            "MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal"),
        std::string::npos)
        << "The stage must lower an explicit graph ownership policy.";
    EXPECT_NE(
        compact_graph_outcome_stage.find("params_.logits->gpu_data_ptr()"),
        std::string::npos)
        << "Every graph participant must consume its own mirrored logits.";
    EXPECT_EQ(compact_graph_outcome_stage.find("isRootParticipant("),
              std::string::npos);
    EXPECT_EQ(compact_graph_outcome_stage.find("collectiveSideband"),
              std::string::npos);
    EXPECT_EQ(compact_graph_outcome_stage.find("publish_mirrored_local_tp"),
              std::string::npos)
        << "The terminal stage is a pure lowering of declarative ownership; it "
           "must not rediscover topology or conditionally add a collective.";

    const size_t stop_control_write =
        compact_verifier_metadata.find(
            "backend->hostToDeviceOnStream("
            "mtp_verifier_stop_tokens_dev_");
    const size_t stop_control_publication =
        compact_verifier_metadata.find(
            "publishPreparedArenaGraphInput("
            "BufferId::MTP_VERIFIER_STOP_TOKENS",
            stop_control_write);
    const size_t verifier_token_materialization =
        compact_device_token_input.find(
            "materializePendingMTPVerifierInputTokensOnDevice(");
    const size_t verifier_token_publication =
        compact_device_token_input.find(
            "publishPreparedArenaGraphInput("
            "BufferId::MTP_VERIFIER_INPUT_TOKENS",
            verifier_token_materialization);
    EXPECT_EQ(stop_control_write, std::string::npos)
        << "Request-constant stop controls must not be uploaded beside verifier "
           "replay.";
    EXPECT_EQ(stop_control_publication, std::string::npos)
        << "Verifier replay must consume the request-admitted stop-control row "
           "without republishing its ownership.";
    ASSERT_NE(verifier_token_materialization, std::string::npos);
    ASSERT_NE(verifier_token_publication, std::string::npos);
    EXPECT_LT(
        verifier_token_materialization,
        verifier_token_publication)
        << "Every staged device-token row must publish its exact arena binding "
           "and producer stream before any condition or verifier graph reads it.";
    EXPECT_EQ(
        compact_verifier_metadata.find(
            "materializePendingMTPVerifierInputTokensOnDevice("),
        std::string::npos)
        << "Token-row ownership belongs to the common graph-input hook, not the "
           "all-position verifier metadata path.";

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
                  "forwardMTPFromDeviceTargetAtLivePositionAndSampleGreedyToDeviceDraftSlot("),
              std::string::npos)
        << "A deferred first token must feed the first greedy sidecar from "
           "the target device slot while preserving sidecar/sample fusion.";
    ASSERT_NE(cuda_greedy_summary_kernel.find("draft_tokens[0]"),
              std::string::npos);
    ASSERT_NE(rocm_greedy_summary_kernel.find("draft_tokens[0]"),
              std::string::npos);
    ASSERT_NE(
        cuda_greedy_summary_kernel.find(
            "sampling_math::kSpeculativeBatchMaxStopTokens"),
        std::string::npos);
    ASSERT_NE(
        rocm_greedy_summary_kernel.find(
            "sampling_math::kSpeculativeBatchMaxStopTokens"),
        std::string::npos);
}

/**
 * @brief Keep commit-replay diagnostics on the GPU condition-token lifecycle.
 *
 * The replay checker knows the oracle token on the host, but a GPU shifted-row
 * commit still has to stage it into a persistent device sample slot, publish
 * readiness on the producer stream, and consume that slot through the normal
 * device-target commit API. Calling the scalar terminal-hidden helper would
 * reintroduce a host condition-token source that production GPU sidecars reject.
 */
TEST(
    Test__GpuWorkspaceAllocationPolicy,
    MTPCommitReplayDiagnosticPublishesGpuConditionTokensThroughDeviceSlots)
{
    const auto runner_source = readFile(
        repoRoot() /
        "src/v2/execution/runner/OrchestrationRunner.cpp");
    ASSERT_FALSE(runner_source.empty());

    const auto replay_commit = sliceBetween(
        runner_source,
        "auto commit_shifted_replay_row = [&](",
        "auto forward_replay_token = [&](");
    const auto compact = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(replay_commit));

    const size_t gpu_branch =
        compact.find("if(runner_->primaryDeviceId().is_gpu())");
    const size_t device_stage =
        compact.find(
            "stageStochasticTargetTokenForDeviceSampling(",
            gpu_branch);
    const size_t device_commit =
        compact.find(
            "commitMTPShiftedRowFromDeviceTargetSample(",
            device_stage);
    const size_t cpu_branch = compact.find("else{", device_commit);
    const size_t scalar_commit =
        compact.find(
            "commitMTPShiftedRowFromCurrentTerminalHidden(",
            cpu_branch);

    ASSERT_NE(gpu_branch, std::string::npos);
    ASSERT_NE(device_stage, std::string::npos);
    ASSERT_NE(device_commit, std::string::npos);
    ASSERT_NE(cpu_branch, std::string::npos);
    ASSERT_NE(scalar_commit, std::string::npos);
    EXPECT_LT(gpu_branch, device_stage);
    EXPECT_LT(device_stage, device_commit);
    EXPECT_LT(device_commit, cpu_branch);
    EXPECT_LT(cpu_branch, scalar_commit)
        << "The scalar replay oracle is CPU-only; GPU replay must consume the "
           "event-published device target slot.";
}

/**
 * @brief Keep sidecar-preservation verifier probes on device-owned input rows.
 *
 * GPU verifier graphs consume token IDs and logical positions as one published
 * arena row. The diagnostic must use the same preparation API and pass the
 * resulting pointer into executeMTPSpecVerifierForward(); invoking the host-token
 * overload would make a stale diagnostic fail before it can inspect main state.
 */
TEST(
    Test__GpuWorkspaceAllocationPolicy,
    MTPSidecarPreservationProbeUsesProductionDeviceVerifierInput)
{
    const auto runner_source = readFile(
        repoRoot() /
        "src/v2/execution/runner/OrchestrationRunner.cpp");
    ASSERT_FALSE(runner_source.empty());

    const auto preservation_probe = sliceBetween(
        runner_source,
        "auto verifier_rows_from_current_state = [&]()",
        "std::optional<std::vector<int32_t>> sidecar_rows");
    const auto compact = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(preservation_probe));

    const size_t gpu_branch =
        compact.find("if(runner_->primaryDeviceId().is_gpu())");
    const size_t typed_device_composer =
        compact.find(
            "prepare_grouped_gpu_verifier_input_tokens(",
            gpu_branch);
    const size_t bind_forward_options =
        compact.find(
            "forward_options.device_token_ids=verifier_input_tokens_device",
            typed_device_composer);
    const size_t verifier_forward =
        compact.find(
            "executeMTPSpecVerifierForward(*runner_,"
            "verifier_input_plan,forward_options)",
            bind_forward_options);

    ASSERT_NE(gpu_branch, std::string::npos);
    ASSERT_NE(typed_device_composer, std::string::npos);
    ASSERT_NE(bind_forward_options, std::string::npos);
    ASSERT_NE(verifier_forward, std::string::npos);
    EXPECT_LT(gpu_branch, typed_device_composer);
    EXPECT_LT(typed_device_composer, bind_forward_options);
    EXPECT_LT(bind_forward_options, verifier_forward);
    EXPECT_EQ(
        compact.find("prepareMTPVerifierInputTokensOnDevice("),
        std::string::npos);
    EXPECT_EQ(
        compact.find(
            "prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken("),
        std::string::npos);
    EXPECT_EQ(
        compact.find(
            "prepareMTPVerifierInputTokensOnDeviceFromHostRow("),
        std::string::npos);
}

/**
 * @brief Make host-authored GPU grouped-verifier rows structurally unreachable.
 *
 * Every production verifier path in OrchestrationRunner shares one typed RB=1
 * composer. It names either target sample slot zero or the current logical-state
 * mailbox and then enters the request-batch API. The three legacy scalar APIs
 * remain available to isolated diagnostic harnesses, but production runner
 * source may not call any of them.
 */
TEST(
    Test__GpuWorkspaceAllocationPolicy,
    ProductionGpuGroupedVerifierUsesOnlyTypedDeviceTokenSources)
{
    const auto runner_source = readFile(
        repoRoot() /
        "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto orchestrator_source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    ASSERT_FALSE(runner_source.empty());
    ASSERT_FALSE(orchestrator_source.empty());

    const auto compact_runner = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(runner_source));
    const auto composer = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            runner_source,
            "auto prepare_grouped_gpu_verifier_input_tokens =",
            "int speculative_draft_count =")));
    const auto device_batch_admission = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokenBatchOnDevice(",
            "const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromHostRow(")));

    EXPECT_NE(
        composer.find("DeviceMTPVerifierInputBatchRequestrequest{"),
        std::string::npos);
    EXPECT_NE(
        composer.find("first_token_from_device=true"),
        std::string::npos);
    EXPECT_NE(
        composer.find("first_token_device_target_slot_available"),
        std::string::npos);
    EXPECT_NE(
        composer.find("first_token_resident_state.has_value()"),
        std::string::npos);
    EXPECT_NE(
        composer.find("prepareMTPVerifierInputTokenBatchOnDevice("),
        std::string::npos);
    EXPECT_NE(
        device_batch_admission.find("!row.first_token_from_device"),
        std::string::npos)
        << "The backend API itself must reject host-owned grouped verifier rows.";
    EXPECT_EQ(
        compact_runner.find("prepareMTPVerifierInputTokensOnDevice("),
        std::string::npos)
        << "Production GPU verification may not compose row zero from a host scalar.";
    EXPECT_EQ(
        compact_runner.find(
            "prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken("),
        std::string::npos)
        << "Target slots must use the same typed request descriptor as resident mailboxes.";
    EXPECT_EQ(
        compact_runner.find(
            "prepareMTPVerifierInputTokensOnDeviceFromHostRow("),
        std::string::npos)
        << "Complete host verifier rows are diagnostic-only and cannot enter production wiring.";
    EXPECT_EQ(
        countOccurrences(
            compact_runner,
            "prepare_grouped_gpu_verifier_input_tokens"),
        5u)
        << "One definition plus all four verifier consumers must share the same ownership policy.";
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
        "size_t GDNRecurrenceStage::deinterleaveScratchFloats(");
    const auto conv_restore = sliceBetween(
        conv_stage,
        "bool ShortConv1dStage::restoreVerifierStateCaptureRowFromDeviceIndex(",
        "bool ShortConv1dStage::execute(");

    for (const auto &body : {gdn_restore, conv_restore})
    {
        const auto compact = removeAsciiWhitespace(stripCommentsAndStringLiterals(body));
        EXPECT_NE(compact.find("!device_row_index||!stream"), std::string::npos)
            << "Device-indexed stage restore must reject null row pointers and default streams.";
        EXPECT_NE(compact.find("restoreVerifierStateCaptureRowFromDeviceIndex("),
                  std::string::npos);
        EXPECT_NE(compact.find("bindKernelWorkspace()"), std::string::npos)
            << "Every publication entry point must establish its own verifier "
               "workspace binding; graph replay has no host callback.";
        EXPECT_NE(compact.find("nullptr,device_row_index,stream"),
                  std::string::npos)
            << "Device-indexed restore must not pass a host state mirror into the backend.";
        EXPECT_EQ(compact.find("stream?stream:gpuStream()"), std::string::npos)
            << "The device-indexed path must never silently fall back to a cached stream.";
    }

    const auto gdn_header = readFile(
        repoRoot() /
        "src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h");
    const auto conv_header = readFile(
        repoRoot() /
        "src/v2/execution/compute_stages/stages/ShortConv1dStage.h");
    EXPECT_EQ(gdn_header.find("onGraphReplayed"), std::string::npos);
    EXPECT_EQ(gdn_header.find("needsOnGraphReplayed"), std::string::npos);
    EXPECT_EQ(conv_header.find("onGraphReplayed"), std::string::npos);
    EXPECT_EQ(conv_header.find("needsOnGraphReplayed"), std::string::npos);

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
        EXPECT_EQ(compact.find("debugLogDeviceIndexedRestoreSamples"),
                  std::string::npos)
            << relative << " must not hide a D2H state sampler behind an opt-in diagnostic hook.";
        EXPECT_EQ(compact.find("LLAMINAR_MTP_PUBLICATION_DIAGNOSTICS"),
                  std::string::npos)
            << relative << " device-state publication diagnostics must remain device-resident.";

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

TEST(Test__GpuWorkspaceAllocationPolicy, MTPTerminalHiddenDeviceAcceptedRowsUseExactProducerPointer)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto external_rows_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelectFromDeviceMetadata(",
        "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRows(");
    const auto compact_external_rows =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(external_rows_body));

    EXPECT_NE(compact_external_rows.find(
                  "cache.external_device_row_indices!=row_indices_device"),
              std::string::npos)
        << "Execution must prove that the cached graph still names the exact metadata producer address.";
    EXPECT_EQ(compact_external_rows.find("createHiddenStateRowsSelect("),
              std::string::npos)
        << "A missing or stale external binding is fatal; the hot path must not rebuild a replacement graph.";
    EXPECT_EQ(compact_external_rows.find("make_unique<ComputeGraph>"),
              std::string::npos)
        << "Device-metadata execution must consume only the pre-materialized graph family.";
    EXPECT_NE(compact_external_rows.find("executeMTPTerminalHiddenRowsCaptured("),
              std::string::npos)
        << "External row metadata selection must use the reusable captured graph lifecycle.";
    EXPECT_NE(compact_external_rows.find("orderCaptureStreamAfter("),
              std::string::npos)
        << "The selector capture stream must wait for its exact metadata producer.";
    EXPECT_NE(compact_external_rows.find("orderStreamAfterCapture("),
              std::string::npos)
        << "Downstream publication must consume the exact selector capture stream.";
    EXPECT_NE(compact_external_rows.find("capture_policy.defer_final_sync=true"),
              std::string::npos)
        << "Terminal-hidden selection must never synchronize its capture stream in the hot path.";
    EXPECT_NE(
        compact_external_rows.find(
            "capture_policy.graph_replay_plan_policy=DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph"),
        std::string::npos)
        << "A segmented terminal-hidden selector cannot enter production.";
    EXPECT_EQ(compact_external_rows.find("synchronizeStream("),
              std::string::npos);
    EXPECT_EQ(compact_external_rows.find("setSelectedRowsForReplay("),
              std::string::npos)
        << "The device-metadata helper must not mutate row indices through host replay setters.";

    const auto materialize_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::materializeMTPTerminalHiddenRowsSelectGraph(",
        "bool DeviceGraphOrchestrator::materializeMTPTerminalHiddenPublicationGraphs(");
    const auto compact_materialize =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(materialize_body));
    EXPECT_NE(
        compact_materialize.find(
            "params.external_device_row_indices=external_device_row_indices"),
        std::string::npos)
        << "Graph setup must record the producer-owned row pointer directly in the captured stage.";
    EXPECT_EQ(
        compact_materialize.find(
            "params.workspace_buffer_name=stable_row_buffer"),
        std::string::npos)
        << "External producer ownership must not be inferred through a same-named graph workspace buffer.";

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
    EXPECT_NE(
        compact_accepted_rows.find(
            "mtp_spec_decode_metadata_binding_.devicePointers().accepted_state_slot_indices"),
        std::string::npos)
        << "Accepted-state publication must pass the metadata producer's exact device address.";

    const auto export_body = sliceBetween(
        source,
        "DeviceGraphOrchestrator::mtpAcceptedTerminalHiddenDeviceLoopGraphTemplate(",
        "bool DeviceGraphOrchestrator::executeMTPDepth0(");
    const auto compact_export =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(export_body));
    EXPECT_NE(
        compact_export.find(
            "cache.workspace_generation!=workspaceGeneration(state_.device_id)"),
        std::string::npos)
        << "Parent-loop composition must reject stale selector bindings.";
    EXPECT_NE(
        compact_export.find(
            "DeviceRowIndexSource::ExternalDeviceIndices"),
        std::string::npos);
    EXPECT_NE(
        compact_export.find(
            "MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES"),
        std::string::npos)
        << "Only compact device-outcome row indices may enter the parent loop.";
    EXPECT_NE(compact_export.find("deviceLoopGraphTemplate("),
              std::string::npos)
        << "The exported selector must pass the common strict monolithic graph contract.";
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
    EXPECT_NE(
        compact_graph_row_select.find(
            "row_params.selection_policy="
            "HiddenStateRowSelectStage::SelectionPolicy::"
            "DeviceResidentRequestLength"),
        std::string::npos)
        << "Single-request bucketed GPU LM-head selection must read the same resident length owner.";
    EXPECT_EQ(
        compact_graph_row_select.find(
            "WS_SELECTED_ROW_SCALAR"),
        std::string::npos)
        << "GPU LM-head graphs must not retain a pinned-scalar workspace seam.";

    const auto graph_kv_append = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            graph_source,
            "std::string QwenGraphBase::addKVCacheAppend(",
            "std::string QwenGraphBase::addKVCacheAndAttention(")));
    EXPECT_NE(
        graph_kv_append.find(
            "config_.grouped_mtp_verifier?"
            "KVCacheAppendSemantics::DecodeEquivalentVerifier:"
            "KVCacheAppendSemantics::Standard"),
        std::string::npos)
        << "Only the explicit grouped-verifier role may select speculative KV append semantics.";
    EXPECT_EQ(
        graph_kv_append.find(
            "config_.compute_all_position_logits&&seq_len"),
        std::string::npos)
        << "All-position logits are also used by compact ragged prefill and must "
           "never classify cache mutation semantics.";

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

    const auto refresh = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::refreshMTPTerminalHiddenState(",
            "bool DeviceGraphOrchestrator::ensureMTPCheckpointTerminalHidden()")));
    EXPECT_NE(
        refresh.find(
            "if(state_.device_id.is_gpu())"
            "{returnselectMTPTerminalHiddenRowsFromDeviceRequestLengths("
            "batch_size,seq_len,total_rows,producer_stream);}"),
        std::string::npos)
        << "Batched GPU terminal-hidden refresh must consume arena-resident request lengths directly.";

    const auto request_selector = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromDeviceRequestLengths(",
            "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromDeviceAcceptedState(")));
    EXPECT_NE(
        request_selector.find(
            "mtp_terminal_hidden_request_rows_select_caches_"),
        std::string::npos);
    EXPECT_NE(
        request_selector.find("static_cast<size_t>(request_count-1)"),
        std::string::npos)
        << "Request count, rather than prompt width, must select the captured graph identity.";
    EXPECT_NE(
        request_selector.find(
            "DeviceRowIndexSource::RequestTerminalLengths"),
        std::string::npos);
    EXPECT_NE(
        request_selector.find(
            "request_cache.request_sequence_lengths_device!="
            "static_cast<constint32_t*>(request_sequence_lengths_dev_)"),
        std::string::npos)
        << "Captured request-terminal graphs must authenticate the exact resident length binding.";
    EXPECT_NE(
        request_selector.find(
            "request_cache.request_row_stride_source!="
            "HiddenStateRowsSelectStage::RequestRowStrideSource::"
            "ExternalDeviceScalar"),
        std::string::npos)
        << "Captured request-terminal graphs must require device-owned dynamic geometry.";
    EXPECT_NE(
        request_selector.find(
            "request_cache.request_row_stride_device!="
            "request_row_stride_dev_"),
        std::string::npos)
        << "Captured request-terminal graphs must authenticate the exact resident stride binding.";
    EXPECT_EQ(request_selector.find("last_forward_request_lengths"),
              std::string::npos);
    EXPECT_EQ(request_selector.find("std::vector"), std::string::npos);

    const auto explicit_rows = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRows(\n"
            "        const std::vector<int> &row_indices,",
            "bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromDeviceRequestLengths(")));
    EXPECT_NE(
        explicit_rows.find(
            "if(state_.device_id.is_gpu()){LOG_ERROR();returnfalse;}"),
        std::string::npos)
        << "Host-authored row vectors must be structurally unavailable to GPU publication.";

    const auto compact_stage =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(stage_source));
    EXPECT_NE(
        compact_stage.find(
            "launchDeviceGeometryRequestTerminalRowsSelectFP32("),
        std::string::npos)
        << "The stage must dispatch the prompt-width-total device-geometry row-packing kernel.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPDeviceResidentPublicationMetadataStaysOnVerifierStream)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.cpp");
    const auto runner_interface =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto cuda_sampling =
        readFile(repoRoot() / "src/v2/kernels/cuda/ops/CUDASamplingKernels.cu");
    const auto rocm_sampling =
        readFile(repoRoot() / "src/v2/kernels/rocm/ops/ROCmSamplingKernels.hip");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_stage =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(stage_source));
    const auto materialize_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::materializeMTPSpeculativeStatePublicationGraph(",
            "bool DeviceGraphOrchestrator::executeMTPSpeculativeStatePublicationCaptured(")));
    const auto lifecycle_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::finalizeMTPSpeculativeStatePublicationLaunch(",
            "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(")));
    const auto compact_cuda_sampling =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_sampling));
    const auto compact_rocm_sampling =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_sampling));

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
    EXPECT_NE(compact.find("mtp_spec_decode_metadata_binding_.hasWorkspace()"),
              std::string::npos)
        << "Publication must consume the persistent workspace family declared during graph setup.";
    EXPECT_EQ(compact.find("ensureDeviceWorkspaceAllocated("),
              std::string::npos)
        << "The decode hot path must never allocate or rebind publication workspace.";
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
    EXPECT_NE(compact.find("materializeMTPSpeculativeStatePublicationGraph("),
              std::string::npos)
        << "Preflight must freeze a typed capture identity rather than enqueue ad hoc kernels.";
    EXPECT_NE(compact_stage.find("enqueueDeriveSpeculativePublicationMetadata("),
              std::string::npos);
    EXPECT_NE(compact_stage.find("params_.accepted_restore_rows_device"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("params_.target_cached_tokens_device"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("params_.accepted_state_counts_device"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("params_.publication_ok_flags_device"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("params_.next_condition_tokens_device"),
              std::string::npos)
        << "The same derivation pass should stage the next decode/sidecar condition token.";
    EXPECT_NE(
        materialize_body.find(
            "params.next_verifier_condition_tokens_device=static_cast<int32_t*>(stochastic_target_sample_tokens_dev_);"),
        std::string::npos)
        << "Accepted-state publication must refresh the fixed target-token bank "
           "consumed by the next verifier graph.";
    EXPECT_GE(
        countOccurrences(
            compact_stage,
            "params_.next_verifier_condition_tokens_device"),
        3u)
        << "Validation and both controller/non-controller publication routes "
           "must retain the canonical verifier-condition destination.";
    for (const auto *backend_source :
         {&compact_cuda_sampling, &compact_rocm_sampling})
    {
        EXPECT_NE(
            backend_source->find(
                "out_next_verifier_condition_tokens[request_index]=out_next_condition_tokens[request_index];"),
            std::string::npos)
            << "Each GPU backend must mirror the exact derived condition token "
               "inside its existing fused publication kernel.";
    }
    EXPECT_NE(compact_stage.find("params_.all_drafts_accepted_flags_device"),
              std::string::npos)
        << "The derivation pass must keep the all-drafts-accepted predicate device-resident.";
    EXPECT_NE(compact_stage.find("params_.stopped_flags_device"),
              std::string::npos)
        << "The derivation pass must keep stop-token state device-resident.";
    EXPECT_NE(compact_stage.find("params_.outcome_tokens_device"),
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
        << "Metadata preflight must not synchronize.";
    EXPECT_EQ(compact_stage.find("synchronize"), std::string::npos)
        << "The captured publication body must contain no host or device synchronization.";
    EXPECT_EQ(compact_stage.find("deviceToHost"), std::string::npos);
    EXPECT_EQ(compact_stage.find("hostToDevice"), std::string::npos);

    const auto canonical_ready = lifecycle_body.find(
        "recordStochasticTargetSampleReady(request_index,request.outcome.stream,true)");
    const auto logical_mailbox = lifecycle_body.find(
        "recordDeviceResidentLogicalSequenceStateMailbox(");
    ASSERT_NE(canonical_ready, std::string::npos)
        << "Finalization must publish the exact producer event for every refreshed target slot.";
    ASSERT_NE(logical_mailbox, std::string::npos);
    EXPECT_LT(canonical_ready, logical_mailbox)
        << "The canonical verifier condition must be visible before the accepted "
           "logical-state mailbox becomes consumable.";
    EXPECT_EQ(runner_interface.find("first_token_logical_state"), std::string::npos);
    EXPECT_EQ(runner_interface.find("first_token_request_index"), std::string::npos)
        << "Verifier input descriptors must not reintroduce transaction-varying "
           "logical-mailbox addresses as token sources.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPDeviceResidentPublicationRequiresAtomicKVAndLogicalState)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.cpp");
    const auto support_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const auto publish_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto lifecycle_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::finalizeMTPSpeculativeStatePublicationLaunch(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const auto compact_support =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(support_body));
    const auto compact_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publish_body));
    const auto compact_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_stage =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(stage_source));
    const auto compact_lifecycle =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(lifecycle_body));

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
        << "The endpoint must complete the resident publication preflight before replay.";
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
        << "Publication must join every deferred shifted sidecar before graph replay.";
    EXPECT_NE(compact_prepare.find("supportsDeviceResidentSequenceStatePublication()"),
              std::string::npos)
        << "Preflight must ask the primary cache whether it can consume device-derived state.";
    EXPECT_NE(compact_prepare.find("supportsDeviceResidentLogicalSequenceStatePublication()"),
              std::string::npos)
        << "Preflight must separately gate the resident logical-state mailbox.";
    EXPECT_LT(compact_prepare.find("supportsDeviceResidentSequenceStatePublication()"),
              compact_prepare.find("materializeMTPSpeculativeStatePublicationGraph("));
    EXPECT_LT(compact_prepare.find("supportsDeviceResidentLogicalSequenceStatePublication()"),
              compact_prepare.find("materializeMTPSpeculativeStatePublicationGraph("))
        << "Both ownership contracts must be proven before the publication graph is built.";

    EXPECT_NE(compact_stage.find("DeviceSequenceStatePublicationRequestrequest{"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("target_cached_tokens_device=params_.target_cached_tokens_device+request_index"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("accepted_state_counts_device=params_.accepted_state_counts_device+request_index"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("publication_ok_flags_device=params_.publication_ok_flags_device+request_index"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("publishSequenceStateFromDeviceMetadata("),
              std::string::npos)
        << "The captured stage must own primary and shifted KV mutation.";
    EXPECT_NE(compact_stage.find("enqueueDeriveShiftedSpeculativePublicationMetadataFromPrimary("),
              std::string::npos)
        << "The stage must derive each shifted-depth target and delta count on device.";
    EXPECT_NE(compact_stage.find("params_.shifted_target_cached_tokens_device"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("params_.shifted_accepted_state_counts_device"),
              std::string::npos);
    EXPECT_NE(compact_stage.find("params_.shifted_kv_caches"),
              std::string::npos)
        << "Shifted MTP caches must be part of the same captured transaction.";
    EXPECT_NE(compact_stage.find("MTPDeviceVerifierStatePublicationShapeshape{"),
              std::string::npos)
        << "Resident GPU publication should describe request shape without "
           "building synthetic host step plans.";
    EXPECT_EQ(compact_stage.find("MTPSpecStepPlan"),
              std::string::npos)
        << "Resident GPU publication must not synthesize host scalar plans for recurrent state.";
    EXPECT_EQ(compact_stage.find("MTPSpecStepPlanBatch"),
              std::string::npos)
        << "Resident GPU publication must not synthesize host batch plans for recurrent state.";
    EXPECT_NE(compact_stage.find("publishAcceptedMTPSpecStateFromDeviceVerifierRow("),
              std::string::npos)
        << "Scalar captured publication keeps the single-request device-indexed helper.";
    EXPECT_NE(compact_stage.find("publishAcceptedMTPSpecStateFromDeviceVerifierRows("),
              std::string::npos)
        << "Request-batched resident publication must restore GDN/short-conv "
           "state through the batch hook once, not by looping the scalar helper.";
    EXPECT_NE(compact_stage.find("params_.accepted_restore_rows_device"),
              std::string::npos)
        << "Captured recurrent-state publication must consume compact GPU row metadata.";

    EXPECT_NE(compact_publish.find("executeMTPSpeculativeStatePublicationCaptured("),
              std::string::npos)
        << "The endpoint must replay the strict captured publication graph.";
    EXPECT_NE(compact_publish.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState("),
              std::string::npos)
        << "Terminal hidden must be selected from the accepted device row.";
    EXPECT_NE(compact_publish.find("finalizeMTPSpeculativeStatePublicationLaunch("),
              std::string::npos);
    EXPECT_LT(compact_publish.find("executeMTPSpeculativeStatePublicationCaptured("),
              compact_publish.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState("));
    EXPECT_LT(compact_publish.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState("),
              compact_publish.find("finalizeMTPSpeculativeStatePublicationLaunch("))
        << "Lifecycle readiness cannot be published until KV, recurrent state, and terminal hidden are enqueued.";

    EXPECT_NE(compact_lifecycle.find("handleLivePrefixReplayStateAfterMutation("),
              std::string::npos)
        << "Finalization must advance the live epoch before publishing consumer readiness.";
    EXPECT_NE(compact_lifecycle.find("recordDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos)
        << "The lifecycle boundary must record the resident logical-state mailbox.";
    EXPECT_NE(compact_lifecycle.find("recordAcceptedSpecPublicationReady("),
              std::string::npos);
    EXPECT_LT(compact_lifecycle.find("handleLivePrefixReplayStateAfterMutation("),
              compact_lifecycle.find("recordDeviceResidentLogicalSequenceStateMailbox("))
        << "Mailbox epoch must match the live state produced by direct publication.";
    EXPECT_NE(lifecycle_body.find("device_resident_kv_sequence_state_publications"),
              std::string::npos)
        << "Successful captured publication should be visible in perf counters.";
    EXPECT_NE(compact_publish.find("returntrue;"),
              std::string::npos)
        << "The endpoint must complete without a host adoption phase.";
    EXPECT_EQ(compact_publish.find("copyDeviceSpeculativeOutcomesToHost("),
              std::string::npos)
        << "Captured publication must not quietly fall back to a compatibility host bridge.";
    EXPECT_EQ(compact_publish.find("materializeDeviceSpeculativeOutcomesForHostPlan("),
              std::string::npos)
        << "Captured publication must not depend on host-plan materialization.";
    EXPECT_EQ(compact_publish.find("materializeDeviceSpeculativeOutcomesForHostResponse("),
              std::string::npos)
        << "Captured publication must not depend on host-response materialization.";
    EXPECT_EQ(compact_stage.find("deviceToHost"), std::string::npos);
    EXPECT_EQ(compact_stage.find("hostToDevice"), std::string::npos);
    EXPECT_EQ(compact_stage.find("synchronize"), std::string::npos)
        << "The captured transaction body must remain allocation-free and stream ordered.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, DGODeviceLogicalStateMailboxOwnsArenaRowsAcrossWorkspaceRebind)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto publication_stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.cpp");
    const auto orchestration_source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto orchestration_header =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.h");
    const auto runner_interface =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto rank_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
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
        "bool DeviceGraphOrchestrator::admitDeviceResidentLogicalSequenceStateRead(");
    const auto wait_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::admitDeviceResidentLogicalSequenceStateRead(",
        "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateRowReuse(");
    const auto writer_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateRowReuse(",
        "bool DeviceGraphOrchestrator::\n"
        "        recordDeviceResidentLogicalSequenceStateReadCompletion(");
    const auto resident_shifted_commit_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(",
        "bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromDeviceOutcome(");
    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    const auto resident_sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(",
        "bool DeviceGraphOrchestrator::forwardMTPAndSampleGreedy(");
    const auto resident_target_publication_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::publishDeviceResidentConditionTokenToTargetSampleSlot(",
        "int DeviceGraphOrchestrator::sampleStochasticDistributionOnDevice(");
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
    const auto publication_lifecycle_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::finalizeMTPSpeculativeStatePublicationLaunch(",
        "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(");
    const auto workspace_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated(",
        "uint64_t DeviceGraphOrchestrator::workspaceGeneration(");
    const auto compact_view =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(view_body));
    const auto compact_record =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));
    const auto compact_wait =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(wait_body));
    const auto compact_writer =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(writer_body));
    const auto compact_resident_shifted_commit =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(resident_shifted_commit_body));
    const auto compact_sidecar =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sidecar_body));
    const auto compact_resident_sidecar =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(resident_sidecar_body));
    const auto compact_resident_target_publication =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(resident_target_publication_body));
    const auto compact_live_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(live_prepare_body));
    const auto compact_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_direct_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(direct_publish_body));
    const auto compact_publication_stage =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publication_stage_source));
    const auto compact_publication_lifecycle =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publication_lifecycle_body));
    const auto compact_workspace =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(workspace_body));
    const auto clear_mailbox_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::clearDeviceResidentLogicalSequenceStateMailbox()",
        "void DeviceGraphOrchestrator::retireDeviceResidentMTPTransaction()");
    const auto compact_clear_mailbox =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(clear_mailbox_body));
    const auto retire_transaction_body = sliceBetween(
        source,
        "void DeviceGraphOrchestrator::retireDeviceResidentMTPTransaction()",
        "DeviceResidentMTPTransactionLease");
    const auto compact_retire_transaction =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(retire_transaction_body));
    const auto planner_helper_body = sliceBetween(
        orchestration_source,
        "std::optional<int> OrchestrationRunner::currentDecodeTransactionPositionForPlanning(",
        "void OrchestrationRunner::recordMTPDepthZeroBypass()");
    const auto compact_planner_helper =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(planner_helper_body));
    const auto prefill_planning_initializer_body = sliceBetween(
        orchestration_source,
        "bool OrchestrationRunner::initializeDecodeTransactionPlanningPositionAfterPrefill(",
        "bool OrchestrationRunner::advanceDecodeTransactionPlanningPositionAfterForward(");
    const auto compact_prefill_planning_initializer =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(prefill_planning_initializer_body));
    const auto decode_position_advance_body = sliceBetween(
        orchestration_source,
        "bool OrchestrationRunner::advanceDecodeTransactionPlanningPositionAfterForward(",
        "bool OrchestrationRunner::publishDecodeTransactionPlanningPositionAfterMTPCommit(");
    const auto compact_decode_position_advance =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(decode_position_advance_body));
    const auto mtp_commit_position_publish_body = sliceBetween(
        orchestration_source,
        "bool OrchestrationRunner::publishDecodeTransactionPlanningPositionAfterMTPCommit(",
        "std::optional<int> OrchestrationRunner::currentDecodeTransactionPositionForPlanning(");
    const auto compact_mtp_commit_position_publish =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(mtp_commit_position_publish_body));
    const auto prefill_body = sliceBetween(
        orchestration_source,
        "bool OrchestrationRunner::prefill(const std::vector<int32_t> &prompt_tokens)",
        "bool OrchestrationRunner::supportsPrefillBatch(int request_batch) const");
    const auto compact_prefill =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prefill_body));
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
    const auto diagnostic_rebind_body = sliceBetween(
        source,
        "rebindDeviceResidentLogicalStateAfterDiagnosticRestore(",
        "bool DeviceGraphOrchestrator::\n"
        "        publishDeviceResidentLogicalSequenceStateFromTargetSample(");
    const auto compact_diagnostic_rebind =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(diagnostic_rebind_body));
    const auto rank_diagnostic_rebind_body = sliceBetween(
        rank_source,
        "rebindDeviceResidentLogicalStateAfterDiagnosticRestore(",
        "bool RankOrchestrator::commitMTPShiftedRowsFromLastForward(");
    const auto compact_rank_diagnostic_rebind =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(rank_diagnostic_rebind_body));

    EXPECT_NE(compact_runner_interface.find("structDeviceResidentLogicalSequenceStateHandle"),
              std::string::npos);
    EXPECT_NE(compact_runner_interface.find("virtualDeviceResidentLogicalSequenceStateHandledeviceResidentLogicalSequenceState()const"),
              std::string::npos)
        << "The resident logical state handoff must be a typed runner contract.";
    EXPECT_NE(
        compact_runner_interface.find(
            "virtualboolrebindDeviceResidentLogicalStateAfterDiagnosticRestore(intrequest_count)"),
        std::string::npos)
        << "Commit/replay diagnostics need an explicit fail-closed mailbox lifecycle API.";
    EXPECT_NE(
        compact_diagnostic_rebind.find(
            "recordDeviceResidentLogicalSequenceStateMailbox(request_count,stream,DeviceResidentLogicalStatePublicationKind::DiagnosticRestoreRebind,&mailbox_error)"),
        std::string::npos)
        << "Diagnostic restore must rebind the durable arena rows through a fresh event.";
    EXPECT_EQ(compact_diagnostic_rebind.find("synchronize"), std::string::npos)
        << "Diagnostic mailbox rebind must remain stream ordered.";
    EXPECT_EQ(compact_diagnostic_rebind.find("hostToDevice"), std::string::npos);
    EXPECT_EQ(compact_diagnostic_rebind.find("deviceToHost"), std::string::npos);
    EXPECT_NE(
        compact_rank_diagnostic_rebind.find(
            "child->rebindDeviceResidentLogicalStateAfterDiagnosticRestore(request_count)"),
        std::string::npos)
        << "LocalTP must refresh every child mailbox after a rank-wide restore.";
    EXPECT_NE(
        compact_rank_diagnostic_rebind.find(
            "adoptMirroredLocalTPResidentLogicalStateMailboxes("),
        std::string::npos)
        << "LocalTP must rebuild its aggregate only after all child rebinds pass.";
    EXPECT_NE(
        compact_decode_mtp.find(
            "rebindDeviceResidentLogicalStateAfterDiagnosticRestore("),
        std::string::npos)
        << "The opt-in commit/replay checker must repair mailbox lifecycle before returning.";
    EXPECT_NE(
        compact_decode_mtp.find(
            "refresh_resident_condition_handles_after_replay_diagnostic("),
        std::string::npos)
        << "Condition handles captured before diagnostic replay must adopt the refreshed epoch.";
    EXPECT_NE(
        decode_mtp_body.find(
            "commit_replay_check_serial_output_token_mismatches"),
        std::string::npos)
        << "The oracle must compare every grouped output transition with serial replay, not only the final continuation.";
    EXPECT_NE(
        compact_decode_mtp.find(
            "i+1<tokens_to_replay.size()?tokens_to_replay[i+1]:expected_next_token"),
        std::string::npos)
        << "Every replayed row must name its exact next grouped token or final ready token.";
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
    EXPECT_NE(compact_header.find("structDeviceResidentLogicalSequenceStateStorage"),
              std::string::npos)
        << "Published logical state needs an owner whose lifetime is independent of graph workspace.";
    EXPECT_NE(compact_header.find("staticconstexprsize_tkFieldCount=7"),
              std::string::npos)
        << "Every request-lifetime logical field and the independent initialization scratch must have a dedicated persistent row.";
    EXPECT_NE(compact_header.find("boolbindPublicationOutputs(MTPSpecDecodeMetadataDevicePointers*pointers,intrequest_count)const"),
              std::string::npos)
        << "Publication kernels must be redirected structurally to the persistent arena owner.";
    EXPECT_NE(compact_header.find("boolmarkPublished(intrequest_count,uint64_tlive_state_epoch)"),
              std::string::npos)
        << "The durable arena owner must track publication independently of the mailbox view.";
    EXPECT_NE(compact_header.find("boolhasLivePublication(uint64_tlive_state_epoch)const"),
              std::string::npos);
    EXPECT_NE(compact_header.find("boolaliasesSameRowsAs(constDeviceResidentLogicalSequenceStateStorage&other)const"),
              std::string::npos)
        << "Workspace growth needs a complete stable-row identity check.";
    EXPECT_EQ(compact_header.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "DGO must keep one logical-state owner, not a resident mailbox plus a host freshness epoch.";
    EXPECT_NE(compact_header.find("boolownsHandle(constDeviceResidentLogicalSequenceStateHandle&handle,uint64_tcurrent_live_state_epoch)const"),
              std::string::npos)
        << "Mailbox ownership must be checked structurally, not open-coded by each consumer.";
    EXPECT_EQ(compact_header.find(
                  "retargetDeviceResidentLogicalSequenceStateMailboxAfterShiftedKVMutation("),
              std::string::npos)
        << "Resident shifted-KV commits consume one-shot mailboxes instead of "
           "laundering old values into a fresh live-state epoch.";
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
    EXPECT_NE(
        compact_header.find(
            "enumclassDeviceResidentLogicalStatePublicationKind:uint8_t"),
        std::string::npos)
        << "Every resident logical-state writer must select typed publication provenance.";
    EXPECT_NE(compact_header.find("publication_generation=0"),
              std::string::npos)
        << "Stable arena addresses require a monotonic generation to disambiguate publications.";
    EXPECT_NE(compact_header.find(
                  "DeviceResidentLogicalStatePublicationKindpublication_kind"),
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
    EXPECT_EQ(compact_record.find("backend->createEvent("),
              std::string::npos)
        << "Logical-state publication is a decode hot path and must use its "
           "setup-owned readiness event.";
    EXPECT_NE(compact_record.find(
                  "device_resident_logical_sequence_state_ready_event_"),
              std::string::npos);
    EXPECT_NE(compact_record.find("backend->recordEvent("),
              std::string::npos)
        << "Mailbox readiness must be a stream-ordered event, not a host sync.";
    EXPECT_EQ(compact_record.find("synchronizeStream("),
              std::string::npos)
        << "Recording a resident logical-state mailbox must not synchronize.";
    EXPECT_NE(compact_record.find("device_resident_logical_sequence_state_storage_.validFor(request_count)"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.target_positions_device=device_resident_logical_sequence_state_storage_.target_cached_tokens_device"),
              std::string::npos)
        << "The mailbox must publish the arena row, never transient verifier workspace.";
    EXPECT_NE(compact_record.find("mailbox.target_sequence_lengths_device=device_resident_logical_sequence_state_storage_.target_cached_tokens_device"),
              std::string::npos)
        << "Target cached tokens are both next position and sequence length.";
    EXPECT_NE(compact_record.find("mailbox.accepted_state_counts_device=device_resident_logical_sequence_state_storage_.accepted_state_counts_device"),
              std::string::npos)
        << "The mailbox must expose the accepted-state count that defines the correction replay boundary.";
    EXPECT_NE(compact_record.find("mailbox.next_condition_tokens_device=device_resident_logical_sequence_state_storage_.next_condition_tokens_device"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.all_drafts_accepted_flags_device=device_resident_logical_sequence_state_storage_.all_drafts_accepted_flags_device"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.stopped_flags_device=device_resident_logical_sequence_state_storage_.stopped_flags_device"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.publication_ok_flags_device=device_resident_logical_sequence_state_storage_.publication_ok_flags_device"),
              std::string::npos);
    EXPECT_EQ(compact_record.find("ptrs."), std::string::npos)
        << "Mailbox publication must be incapable of retaining a reallocatable workspace pointer.";
    EXPECT_NE(compact_record.find(
                  "mailbox.ready_event=device_resident_logical_sequence_state_ready_event_"),
              std::string::npos);
    EXPECT_NE(compact_record.find("mailbox.live_state_epoch=live_replay_state_epoch_"),
              std::string::npos);
    EXPECT_NE(compact_record.find(
                  "mailbox.publication_generation=++device_resident_logical_state_publication_generation_"),
              std::string::npos)
        << "Each overwrite of the durable mailbox row must receive a distinct generation.";
    EXPECT_NE(compact_record.find("mailbox.publication_kind=publication_kind"),
              std::string::npos)
        << "Publication provenance must be recorded beside the producer event.";
    EXPECT_NE(compact_record.find("device_resident_logical_sequence_state_storage_.markPublished(request_count,live_replay_state_epoch_)"),
              std::string::npos)
        << "Publication liveness belongs to the arena owner, not only its clearable mailbox view.";
    EXPECT_NE(compact_record.find("device_resident_logical_sequence_state_mailbox_=mailbox"),
              std::string::npos);
    EXPECT_EQ(compact_record.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "Recording a resident mailbox must establish ownership directly.";
    EXPECT_EQ(compact_clear_mailbox.find("device_resident_logical_sequence_host_mirror_epoch_"),
              std::string::npos)
        << "Mailbox cleanup must not maintain a retired host freshness owner.";
    EXPECT_EQ(compact_clear_mailbox.find("retirePublication("),
              std::string::npos)
        << "Metadata preparation may clear a view while issued arena handles remain live.";
    EXPECT_NE(compact_retire_transaction.find(
                  "device_resident_logical_sequence_state_storage_.retirePublication()"),
              std::string::npos)
        << "Only the request-scoped transaction boundary retires durable logical values.";
    EXPECT_NE(record_body.find("device_resident_logical_state_mailboxes"),
              std::string::npos)
        << "Mailbox creation should be visible in perf counters.";
    EXPECT_NE(record_body.find("device_resident_logical_state_arena_publications"),
              std::string::npos)
        << "The real persistent-storage path should have a dedicated integration counter.";

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

    EXPECT_EQ(source.find(
                  "retargetDeviceResidentLogicalSequenceStateMailboxAfterShiftedKVMutation"),
              std::string::npos)
        << "A shifted-KV mutation must not launder a consumed logical mailbox "
           "into the next live-state epoch.";
    EXPECT_NE(compact_resident_shifted_commit.find(
                  "clearDeviceResidentLogicalSequenceStateMailbox()"),
              std::string::npos)
        << "The resident shifted-row committer must consume its one-shot mailbox.";
    EXPECT_NE(resident_shifted_commit_body.find(
                  "device_resident_logical_state_mailbox_consumptions"),
              std::string::npos)
        << "Mailbox consumption must remain visible to integration perfstats.";

    EXPECT_NE(compact_sidecar.find("constvoid*position_ids_device_override"),
              std::string::npos)
        << "Sidecar replay must be able to consume device-resident position rows.";
    EXPECT_NE(compact_sidecar.find("input.position_ids_device=effective_position_ids_device"),
              std::string::npos);
    EXPECT_NE(compact_sidecar.find("cached_input.position_ids_device=effective_position_ids_device"),
              std::string::npos);
    EXPECT_NE(compact_sidecar.find("DeviceResidentLogicalStateReadScope"),
              std::string::npos)
        << "Sidecar replay must own a scoped reader until resident next-token/position work is enqueued.";
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

    EXPECT_NE(
        compact_runner_interface.find(
            "virtualboolpublishDeviceResidentConditionTokenToTargetSampleSlot("),
        std::string::npos)
        << "Resident-token preservation needs a typed ownership transition.";
    EXPECT_NE(
        compact_resident_target_publication.find(
            "logical_state.nextConditionTokenDeviceForRequest(request_index)"),
        std::string::npos);
    EXPECT_NE(
        compact_resident_target_publication.find(
            "DeviceResidentLogicalStateReadScope"),
        std::string::npos)
        << "The D2D snapshot must retain a resident reader through its copy.";
    EXPECT_NE(compact_resident_target_publication.find("backend->deviceCopyAsync("),
              std::string::npos);
    EXPECT_NE(
        compact_resident_target_publication.find(
            "recordStochasticTargetSampleReady("),
        std::string::npos)
        << "The persistent target slot must publish an event for later consumers.";
    EXPECT_EQ(compact_resident_target_publication.find("hostToDevice"),
              std::string::npos)
        << "Resident condition tokens must never round-trip through a host shadow.";
    EXPECT_EQ(compact_resident_target_publication.find("deviceToHost"),
              std::string::npos);
    EXPECT_EQ(compact_resident_target_publication.find("synchronize"),
              std::string::npos)
        << "Resident-to-target publication is stream ordered, never host synchronized.";
    EXPECT_NE(
        compact_decode_mtp.find(
            "publishDeviceResidentConditionTokenToTargetSampleSlot("),
        std::string::npos)
        << "Budget-limited direct emit must use the resident D2D handoff.";

    EXPECT_NE(compact_live_prepare.find("accepted_spec_publication_ready_.valid"),
              std::string::npos);
    EXPECT_NE(compact_live_prepare.find("device_resident_logical_sequence_state_mailbox_.valid()"),
              std::string::npos)
        << "Forward graph live-state preparation must not ignore mailbox-only handoffs.";
    EXPECT_NE(compact_live_prepare.find("admitDeviceResidentLogicalSequenceStateRead("),
              std::string::npos);

    EXPECT_NE(
        compact_prepare.find(
            "waitForDeviceResidentLogicalSequenceStateRowReuse("
            "request.outcome.stream,)"),
        std::string::npos)
        << "A replacement writer must join every prior reader before retiring its handle.";
    EXPECT_EQ(compact_prepare.find(
                  "clearDeviceResidentLogicalSequenceStateMailbox();"),
              std::string::npos)
        << "Publication preparation must not retire the mailbox outside the exclusive writer transaction.";
    const size_t writer_begin = compact_writer.find(
        "pending_device_resident_logical_state_writer_.begin(");
    const size_t writer_clear = compact_writer.find(
        "device_resident_logical_sequence_state_mailbox_.clear()",
        writer_begin);
    ASSERT_NE(writer_begin, std::string::npos);
    ASSERT_NE(writer_clear, std::string::npos);
    EXPECT_LT(writer_begin, writer_clear)
        << "The replacement transaction must close reader admission before retiring the old mailbox view.";
    EXPECT_NE(compact_prepare.find("bindPublicationOutputs(&publication_ptrs,request.request_count)"),
              std::string::npos)
        << "The derive kernel must write request-lifetime outputs directly into arena storage.";
    EXPECT_NE(compact_header.find("pointers->target_cached_tokens=target_cached_tokens_device"),
              std::string::npos);
    EXPECT_NE(compact_header.find("pointers->next_condition_tokens=next_condition_tokens_device"),
              std::string::npos);
    EXPECT_NE(
        compact_prepare.find(
            "materializeMTPSpeculativeStatePublicationGraph(request,publication_ptrs,"),
        std::string::npos)
        << "Graph construction must consume the arena-rebound pointer bundle as one typed contract.";
    EXPECT_EQ(compact_prepare.find("recordDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos)
        << "Prepare must not record readiness before KV and shifted-MTP KV publication are enqueued.";
    EXPECT_NE(compact_publication_stage.find("publishSequenceStateFromDeviceMetadata("),
              std::string::npos)
        << "The captured stage must own every device-resident KV mutation.";
    EXPECT_NE(compact_publication_lifecycle.find("recordDeviceResidentLogicalSequenceStateMailbox("),
              std::string::npos);
    EXPECT_LT(compact_direct_publish.find("executeMTPSpeculativeStatePublicationCaptured("),
              compact_direct_publish.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState("));
    EXPECT_LT(compact_direct_publish.find("selectMTPTerminalHiddenRowsFromDeviceAcceptedState("),
              compact_direct_publish.find("finalizeMTPSpeculativeStatePublicationLaunch("))
        << "The mailbox finalizer must run only after the captured KV transaction and terminal-hidden selector.";
    EXPECT_EQ(compact_header.find("adoptDeviceResidentMTPSpecPublishedHostState("),
              std::string::npos);
    EXPECT_EQ(compact_header.find("adoptDeviceResidentMTPSpecPublishedHostStateFromDeviceMetadata("),
              std::string::npos);
    EXPECT_EQ(compact_publication_stage.find("adoptSequenceStateFromHostMetadata("),
              std::string::npos)
        << "Direct resident publication must not repair host KV mirrors after device publication.";
    EXPECT_EQ(source.find("device_resident_host_state_metadata_d2h_wait"),
              std::string::npos)
        << "Resident logical-state freshness must not require a host metadata D2H wait.";

    const auto reset_body = sliceBetween(
        header,
        "void resetInferenceState(const InferenceStateResetRequest &request) override",
        "void clear_cache() override");
    const auto verifier_boundary_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "void DeviceGraphOrchestrator::clearMTPVerifierTransactionStateForBoundary(",
            "void DeviceGraphOrchestrator::clearLivePrefixRestoreTransientHandoffs(")));
    EXPECT_NE(removeAsciiWhitespace(stripCommentsAndStringLiterals(reset_body))
                  .find("clearDeviceResidentLogicalSequenceStateMailbox();"),
              std::string::npos)
        << "Session resets must retire the event-fenced mailbox view.";
    EXPECT_NE(
        verifier_boundary_body.find(
            "greedy_verifier_outcome_graph_transaction_={};"
            "mtp_verifier_outcome_graph_mode_="
            "MTPVerifierOutcomeGraphMode::Disabled;"),
        std::string::npos)
        << "The centralized verifier boundary must retire any armed graph-owned outcome transaction.";

    EXPECT_NE(source.find("device_resident_logical_state_workspace_rebind_survivals"),
              std::string::npos)
        << "Integration tests need proof that a live arena publication survived an actual workspace generation change.";
    EXPECT_NE(compact_workspace.find("logical_state_storage_before.hasLivePublication(live_replay_state_epoch_)"),
              std::string::npos)
        << "Workspace replacement must observe durable publication even after its mailbox view was cleared.";
    EXPECT_NE(compact_workspace.find("device_resident_logical_sequence_state_storage_.aliasesSameRowsAs(logical_state_storage_before)"),
              std::string::npos)
        << "Workspace replacement must prove every durable arena row retained its address.";
    EXPECT_EQ(compact_workspace.find("logical_mailbox_before"),
              std::string::npos)
        << "A clearable mailbox view cannot be the publication-lifetime oracle.";

    EXPECT_NE(compact_orchestration_header.find("currentDecodeTransactionPositionForPlanning("),
              std::string::npos)
        << "MTP planning reads of get_position() should be centralized.";
    EXPECT_NE(compact_orchestration_header.find(
                  "initializeDecodeTransactionPlanningPositionAfterPrefill("),
              std::string::npos);
    EXPECT_NE(compact_orchestration_header.find(
                  "std::optional<int>decode_transaction_planning_position_"),
              std::string::npos)
        << "GPU MTP graph planning needs one scheduler-owned transaction position.";
    EXPECT_EQ(compact_orchestration_header.find(
                  "device_resident_mtp_planning_position_"),
              std::string::npos)
        << "The planning scalar must not be described as a backend device-state mirror.";

    EXPECT_NE(compact_prefill_planning_initializer.find(
                  "runner_->primaryDeviceId().is_gpu()"),
              std::string::npos);
    EXPECT_NE(compact_prefill_planning_initializer.find(
                  "decode_transaction_planning_position_=committed_tokens"),
              std::string::npos)
        << "Successful GPU prefill must initialize planning from the scheduler-known prompt length.";
    EXPECT_NE(prefill_planning_initializer_body.find(
                  "gpu_decode_transaction_position_initializations"),
              std::string::npos);
    EXPECT_EQ(compact_prefill_planning_initializer.find("mtp.enabled"),
              std::string::npos)
        << "Ordinary seeded GPU decode and MTP share the same logical-position contract.";
    EXPECT_NE(compact_decode_position_advance.find(
                  "++(*decode_transaction_planning_position_)"),
              std::string::npos)
        << "Every committed one-token GPU forward must advance the scheduler position.";
    EXPECT_EQ(compact_decode_position_advance.find("runner_->get_position()"),
              std::string::npos)
        << "Advancing GPU decode planning must never observe a backend host mirror.";
    EXPECT_NE(decode_position_advance_body.find(
                  "gpu_decode_transaction_position_forward_advances"),
              std::string::npos);
    EXPECT_NE(compact_mtp_commit_position_publish.find(
                  "*decode_transaction_planning_position_!=transaction_base_tokens"),
              std::string::npos)
        << "An MTP commit must prove that its validated base still matches the "
           "scheduler transaction before publishing the next position.";
    EXPECT_NE(compact_mtp_commit_position_publish.find(
                  "decode_transaction_planning_position_="
                  "transaction_base_tokens+committed_rows"),
              std::string::npos)
        << "Validated MTP rows must advance the scheduler-owned transaction coordinate.";
    EXPECT_EQ(compact_mtp_commit_position_publish.find(
                  "deviceResidentLogicalSequenceState()"),
              std::string::npos)
        << "Scheduler position publication must not depend on a transient "
           "device-outcome mailbox.";
    EXPECT_EQ(compact_mtp_commit_position_publish.find("runner_->get_position()"),
              std::string::npos)
        << "GPU MTP commits must never recover position from a backend host mirror.";
    EXPECT_NE(compact_decode_mtp.find(
                  "publishDecodeTransactionPlanningPositionAfterMTPCommit("),
              std::string::npos)
        << "Every successful MTP state commit must publish its scheduler position.";
    EXPECT_NE(compact_prefill.find(
                  "initializeDecodeTransactionPlanningPositionAfterPrefill("),
              std::string::npos)
        << "Every successful scalar prefill path, including prefix restore, must publish its planning position.";

    EXPECT_EQ(compact_planner_helper.find("hostLogicalStateMirrorsDeviceResidentState"),
              std::string::npos)
        << "Planning must not consult a retired backend host-mirror freshness contract.";
    EXPECT_EQ(compact_planner_helper.find("deviceResidentLogicalSequenceState()"),
              std::string::npos)
        << "The transaction position is scheduler-owned and must not depend on mailbox visibility.";
    const size_t gpu_planning_guard = compact_planner_helper.find(
        "runner_->primaryDeviceId().is_gpu()");
    const size_t cpu_position_read = compact_planner_helper.find(
        "runner_->get_position()");
    ASSERT_NE(gpu_planning_guard, std::string::npos);
    ASSERT_NE(cpu_position_read, std::string::npos);
    EXPECT_LT(gpu_planning_guard, cpu_position_read)
        << "GPU planning must return or fail closed before the CPU-only backend position read.";
    EXPECT_NE(compact_planner_helper.find(
                  "decode_transaction_planning_position_.has_value()"),
              std::string::npos);
    EXPECT_NE(compact_planner_helper.find("runner_->get_position()"),
              std::string::npos)
        << "CPU planning still reads the CPU runner's canonical scalar position.";
    EXPECT_NE(planner_helper_body.find("gpu_decode_transaction_position_reads"),
              std::string::npos);
    EXPECT_NE(planner_helper_body.find("cpu_decode_position_planning_reads"),
              std::string::npos);
    EXPECT_NE(compact_decode_mtp.find("currentDecodeTransactionPositionForPlanning("),
              std::string::npos);
    EXPECT_EQ(compact_decode_mtp.find("runner_->get_position()"),
              std::string::npos)
        << "decodeStepMTP must not bypass the host-mirror freshness guard.";
    EXPECT_NE(compact_decode_step.find("currentDecodeTransactionPositionForPlanning("),
              std::string::npos)
        << "Dynamic depth-zero shifted-cache maintenance must use the same planning guard.";
    EXPECT_NE(compact_decode_step.find(
                  "advanceDecodeTransactionPlanningPositionAfterForward("),
              std::string::npos)
        << "Ordinary GPU decode must advance scheduler-owned position after forwarding the prior token.";
    EXPECT_EQ(compact_decode_step.find("runner_->get_position()"),
              std::string::npos)
        << "decodeStep must not bypass the host-mirror freshness guard for MTP planning.";
}

/**
 * @brief Enforce fan-out/fan-in ownership for reusable logical-state rows.
 *
 * The resident logical-state mailbox is a single persistent row bank. Readers
 * must fan out from the immutable publication-ready event so independent GPU
 * streams can overlap. Each stream publishes completion to its own preallocated
 * lane, and a replacement writer waits every occupied lane before its first
 * store.
 *
 * This source contract also proves that ForwardExecutionEngine pairs every
 * admitted forward-graph read with a post-launch completion hook on cache hits
 * and cache misses. The check bans synchronization, transfers, and allocation
 * from the ownership protocol so a future correctness repair cannot quietly
 * turn the device-resident path into a serial or host-mediated one.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     LogicalStateMailboxReadersFanOutAndReplacementWritersFanIn)
{
    const auto root = repoRoot();
    const auto host_interface = readFile(
        root /
        "src/v2/execution/local_execution/engine/ForwardExecutionEngine.h");
    const auto orchestrator_header = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto engine_source = readFile(
        root /
        "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp");
    const auto orchestrator_source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto compact_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(host_interface));
    const auto compact_orchestrator_header =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(orchestrator_header));
    const auto compact_reader_wait =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::admitDeviceResidentLogicalSequenceStateRead(",
            "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateRowReuse(")));
    const auto compact_reader_completion =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::\n"
            "        recordDeviceResidentLogicalSequenceStateReadCompletion(",
            "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateMailboxForObservation(")));
    const auto compact_writer_wait =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateRowReuse(",
            "bool DeviceGraphOrchestrator::\n"
            "        recordDeviceResidentLogicalSequenceStateReadCompletion(")));
    const auto compact_forward_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(",
            "bool DeviceGraphOrchestrator::completeLiveStateForForwardGraphExecution(")));
    const auto compact_forward_complete =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::completeLiveStateForForwardGraphExecution(",
            "const float *DeviceGraphOrchestrator::getAllPositionLogits() const")));
    const auto compact_cache_hit =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            engine_source,
            "bool ForwardExecutionEngine::executeCacheHit(",
            "bool ForwardExecutionEngine::executePrefillWithGraphCache(")));
    const auto compact_cache_miss =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            engine_source,
            "bool ForwardExecutionEngine::executeCacheMiss(",
            "std::optional<ForwardExecutionEngine::PrefillGraphCacheSnapshot>")));
    const auto compact_batch_advance =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::advanceMTPRequestBatchConditionOnDevice(",
            "bool DeviceGraphOrchestrator::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(")));

    EXPECT_NE(
        compact_interface.find(
            "virtualboolcompleteLiveStateForForwardGraphExecution("),
        std::string::npos)
        << "Forward launch ownership needs a mandatory matching completion hook.";
    EXPECT_NE(
        compact_orchestrator_header.find(
            "structPendingDeviceResidentLogicalStateWriter"),
        std::string::npos)
        << "Reader exclusion must be represented by an explicit writer transaction.";
    EXPECT_NE(
        compact_orchestrator_header.find(
            "std::shared_lock<std::shared_mutex>admission_lock"),
        std::string::npos)
        << "Every asynchronous reader must carry its admission lease until completion publication.";

    EXPECT_NE(compact_reader_wait.find("mailbox.ready_event.get()"),
              std::string::npos)
        << "Readers must fan out from the immutable publication-ready event.";
    EXPECT_NE(
        compact_reader_wait.find(
            "std::shared_lock<std::shared_mutex>read_admission("),
        std::string::npos)
        << "Reader admission must atomically exclude replacement-writer admission.";
    EXPECT_NE(
        compact_reader_wait.find(
            "pending_device_resident_logical_state_writer_.valid()"),
        std::string::npos)
        << "No reader may enter while replacement publication is incomplete.";
    EXPECT_NE(
        compact_reader_wait.find(
            "*admission_lock=std::move(read_admission)"),
        std::string::npos)
        << "The shared admission lease must outlive the host admission method.";
    EXPECT_EQ(compact_reader_wait.find("access_fence.event.get()"),
              std::string::npos)
        << "Waiting any mutable access edge at admission serializes readers.";
    EXPECT_NE(
        compact_reader_wait.find(
            "mailbox.live_state_epoch!=live_replay_state_epoch_"),
        std::string::npos)
        << "Reader admission must reject a stale publication epoch.";
    EXPECT_EQ(compact_reader_wait.find("mailbox.clear()"),
              std::string::npos)
        << "A stale reader must fail immediately; silently clearing the "
           "mailbox hides a broken lifecycle edge.";
    EXPECT_NE(compact_reader_wait.find("returnfalse;"),
              std::string::npos)
        << "A stale reader must stop inference instead of tolerating the "
           "invalid publication.";

    EXPECT_EQ(compact_reader_completion.find("streamWaitEvent("),
              std::string::npos)
        << "Reader completion must not serialize behind another reader.";
    EXPECT_NE(
        compact_reader_completion.find(
            "device_resident_logical_state_reader_completion_events_[lane]"),
        std::string::npos)
        << "Every reader stream needs a fixed independent completion lane.";
    EXPECT_NE(compact_reader_completion.find(
                  "backend->recordEvent(completion_event.get(),"),
              std::string::npos)
        << "Reader completion must publish its own lane after enqueuing work.";

    EXPECT_NE(compact_writer_wait.find(
                  "access_epoch.publication_ready_event.get()"),
              std::string::npos)
        << "The writer must wait the publication root even when there are no readers.";
    EXPECT_NE(
        compact_writer_wait.find(
            "std::unique_lock<std::shared_mutex>write_admission("),
        std::string::npos)
        << "A replacement writer must close reader admission before inspecting completion lanes.";
    EXPECT_NE(compact_writer_wait.find(
                  "lane<access_epoch.reader_stream_count"),
              std::string::npos)
        << "The replacement writer must fan in every occupied reader lane.";
    EXPECT_NE(
        compact_writer_wait.find(
            "device_resident_logical_state_reader_completion_events_[lane]"),
        std::string::npos);

    const size_t forward_admit = compact_forward_prepare.find(
        "admitDeviceResidentLogicalSequenceStateRead("
        "execution_stream,");
    const size_t forward_arm = compact_forward_prepare.find(
        "pending_device_resident_logical_state_forward_read_.stream="
        "execution_stream;");
    ASSERT_NE(forward_admit, std::string::npos);
    ASSERT_NE(forward_arm, std::string::npos);
    EXPECT_LT(forward_admit, forward_arm);
    EXPECT_NE(
        compact_forward_complete.find(
            "execution_stream!=pending.stream"),
        std::string::npos)
        << "Completion must reject a guessed or substituted stream.";
    EXPECT_NE(
        compact_forward_complete.find(
            "recordDeviceResidentLogicalSequenceStateReadCompletion("
            "execution_stream,"),
        std::string::npos);

    const size_t cached_launch = compact_cache_hit.find(
        "executor_.executeDecodeWithCapturePolicy(");
    const size_t cached_complete = compact_cache_hit.find(
        "host.completeLiveStateForForwardGraphExecution(");
    ASSERT_NE(cached_launch, std::string::npos);
    ASSERT_NE(cached_complete, std::string::npos);
    EXPECT_LT(cached_launch, cached_complete);

    const size_t miss_launch = compact_cache_miss.find(
        "executor_.executeWithSnapshotManifest(");
    const size_t miss_complete = compact_cache_miss.find(
        "host.completeLiveStateForForwardGraphExecution(");
    ASSERT_NE(miss_launch, std::string::npos);
    ASSERT_NE(miss_complete, std::string::npos);
    EXPECT_LT(miss_launch, miss_complete);

    EXPECT_EQ(
        compact_batch_advance.find(
            "admitDeviceResidentLogicalSequenceStateRead("),
        std::string::npos)
        << "Request-batched advancement must use scoped readers, not an unpaired broad wait.";
    const size_t compose_read = compact_batch_advance.find(
        "DeviceResidentLogicalStateReadScopelogical_state_read(");
    const size_t compose_launch = compact_batch_advance.find(
        "backend->enqueuePrepareMTPBatchedSidecarInputs(",
        compose_read);
    const size_t compose_complete = compact_batch_advance.find(
        "logical_state_read.complete()",
        compose_launch);
    const size_t writer_wait = compact_batch_advance.find(
        "waitForDeviceResidentLogicalSequenceStateRowReuse(",
        compose_complete);
    const size_t replacement_write = compact_batch_advance.find(
        "backend->enqueueInitializeMTPDeviceLogicalState(",
        writer_wait);
    ASSERT_NE(compose_read, std::string::npos);
    ASSERT_NE(compose_launch, std::string::npos);
    ASSERT_NE(compose_complete, std::string::npos);
    ASSERT_NE(writer_wait, std::string::npos);
    ASSERT_NE(replacement_write, std::string::npos);
    EXPECT_LT(compose_read, compose_launch);
    EXPECT_LT(compose_launch, compose_complete);
    EXPECT_LT(compose_complete, writer_wait);
    EXPECT_LT(writer_wait, replacement_write);

    const std::string ownership_protocol =
        compact_reader_wait + compact_reader_completion + compact_writer_wait +
        compact_forward_prepare + compact_forward_complete;
    EXPECT_EQ(ownership_protocol.find("synchronize"), std::string::npos);
    EXPECT_EQ(ownership_protocol.find("deviceToHost"), std::string::npos);
    EXPECT_EQ(ownership_protocol.find("hostToDevice"), std::string::npos);
    EXPECT_EQ(ownership_protocol.find("malloc"), std::string::npos);
    EXPECT_EQ(ownership_protocol.find("free("), std::string::npos);
}

/**
 * @brief Keep the scalar target-to-sidecar transition fully device authoritative.
 *
 * The condition graph advances canonical main KV state before the first target
 * token is sampled. The first sidecar historically consumed the new token and
 * KV count directly while leaving the durable resident mailbox at the prior
 * position. That split ownership was timing-safe for the sidecar itself but
 * made later checkpoint and publication consumers observe stale logical state.
 *
 * This regression proves that the public target-slot transition performs one
 * explicit publication transaction before sidecar replay. It also bans every
 * tempting reconstruction path: host position mirrors, request-admission
 * lengths, guessed streams, host/device copies, synchronization, and hot-path
 * allocation.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     ScalarDeviceTargetSidecarRepublishesCanonicalLogicalState)
{
    const auto root = repoRoot();
    const auto header = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto compact_transition =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(",
            "bool DeviceGraphOrchestrator::forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(")));
    const auto compact_publication =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::\n"
            "        publishDeviceResidentLogicalSequenceStateFromTargetSample(",
            "const char *DeviceGraphOrchestrator::deviceResidentLogicalStatePublicationKindName(")));

    EXPECT_NE(
        compact_header.find(
            "boolpublishDeviceResidentLogicalSequenceStateFromTargetSample("
            "inttarget_sample_slot,constint32_t*live_position_device);"),
        std::string::npos)
        << "The target-slot ownership boundary needs one named typed helper.";

    const size_t canonical_count = compact_transition.find(
        "state_.kv_cache->deviceSequenceCachedTokenCountPtr(");
    const size_t publication = compact_transition.find(
        "publishDeviceResidentLogicalSequenceStateFromTargetSample("
        "target_sample_slot,live_position_device)");
    const size_t sidecar = compact_transition.find("executeMTPDepth0Batched(");
    ASSERT_NE(canonical_count, std::string::npos);
    ASSERT_NE(publication, std::string::npos);
    ASSERT_NE(sidecar, std::string::npos);
    EXPECT_LT(canonical_count, publication);
    EXPECT_LT(publication, sidecar)
        << "The resident token/position pair must be published before sidecar replay.";

    EXPECT_NE(
        compact_publication.find(
            "target_ready.producer_stream"),
        std::string::npos)
        << "Publication must append to the exact target sampler stream.";
    EXPECT_EQ(
        compact_publication.find("explicitGPUStreamForOperation("),
        std::string::npos)
        << "A guessed ambient stream would sever the target producer edge.";
    const size_t prior_mailbox_reuse = compact_publication.find(
        "waitForDeviceResidentLogicalSequenceStateRowReuse("
        "producer_stream,");
    const size_t target_wait = compact_publication.find(
        "waitForRequiredStochasticTargetSampleReady("
        "target_sample_slot,producer_stream,");
    const size_t initialize = compact_publication.find(
        "backend->enqueueInitializeMTPDeviceLogicalState("
        "target_token_device,live_position_device,");
    const size_t mailbox_event = compact_publication.find(
        "recordDeviceResidentLogicalSequenceStateMailbox(");
    ASSERT_NE(prior_mailbox_reuse, std::string::npos);
    ASSERT_NE(target_wait, std::string::npos);
    ASSERT_NE(initialize, std::string::npos);
    ASSERT_NE(mailbox_event, std::string::npos);
    EXPECT_LT(prior_mailbox_reuse, initialize)
        << "Single-buffered durable rows cannot be overwritten until every prior reader completes.";
    EXPECT_LT(target_wait, initialize)
        << "The publication kernel must consume a proven-ready target sample.";
    EXPECT_LT(initialize, mailbox_event)
        << "Mailbox readiness must cover the scratch write and all six durable row writes.";
    EXPECT_NE(
        compact_publication.find(
            "storage.initialization_base_cached_tokens_scratch_device"),
        std::string::npos)
        << "First-sidecar publication cannot borrow verifier workspace that does not exist yet.";

    EXPECT_EQ(compact_publication.find("state_.positions"), std::string::npos);
    EXPECT_EQ(compact_publication.find("request_sequence_lengths_dev_"),
              std::string::npos);
    EXPECT_EQ(compact_publication.find("get_position("), std::string::npos);
    EXPECT_EQ(compact_publication.find("hostToDevice"), std::string::npos);
    EXPECT_EQ(compact_publication.find("deviceToHost"), std::string::npos);
    EXPECT_EQ(compact_publication.find("synchronize"), std::string::npos);
    EXPECT_EQ(compact_publication.find("malloc"), std::string::npos);
    EXPECT_EQ(compact_publication.find("free("), std::string::npos);
}

/**
 * @brief Make post-restore GPU observations read canonical cache metadata.
 *
 * Restoring a checkpoint invalidates the prior speculative-outcome mailbox,
 * but the restored KV sequence count remains device authoritative. This guard
 * proves that host-visible cache snapshots and diagnostics share one observer:
 * use the event-fenced mailbox while it is current, otherwise read the
 * cache-owned count on the already ordered observation stream. GPU callers
 * must fail closed if neither device source is available; state_.positions is
 * never a coherence fallback.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     GPUCheckpointObservationUsesCanonicalKVCountWithoutMailbox)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto compact_observer =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            source,
            "std::optional<int> DeviceGraphOrchestrator::deviceResidentLogicalTokenCountForObservation(",
            "std::optional<int> DeviceGraphOrchestrator::deviceResidentShiftedMTPKVTokenCountForObservation(")));
    const auto capture_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixState(",
        "bool DeviceGraphOrchestrator::restoreLivePrefixState(");
    const auto compact_capture =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(capture_body));
    const auto probe_body = sliceBetween(
        source,
        "PrefixRuntimeStateSnapshot DeviceGraphOrchestrator::prefixStateProbe() const",
        "void DeviceGraphOrchestrator::disablePrefixCacheForRunner(");
    const auto compact_probe_fallback =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            probe_body,
            "else if (state_.device_id.is_gpu() &&",
            "const int sequence_count = state_.batch_size > 0 ? state_.batch_size : 1;")));

    EXPECT_NE(
        compact_observer.find(
            "mailbox.live_state_epoch==live_replay_state_epoch_"),
        std::string::npos);
    EXPECT_NE(
        compact_observer.find(
            "state_.kv_cache->supportsDeviceResidentSequenceStatePublication()"),
        std::string::npos);
    EXPECT_NE(
        compact_observer.find(
            "state_.kv_cache->deviceSequenceCachedTokenCountPtr(request_index)"),
        std::string::npos);
    EXPECT_NE(compact_observer.find("backend->deviceToHostFast("),
              std::string::npos)
        << "The scalar transfer belongs only to this explicit host-visible "
           "observation boundary.";
    EXPECT_EQ(compact_observer.find("state_.positions"), std::string::npos);
    EXPECT_EQ(compact_observer.find("get_position("), std::string::npos);
    EXPECT_EQ(compact_observer.find("synchronize"), std::string::npos);

    EXPECT_NE(
        compact_capture.find(
            "if(state_.device_id.is_gpu()){conststd::optional<int>"
            "resident_tokens=deviceResidentLogicalTokenCountForObservation("),
        std::string::npos);
    EXPECT_NE(
        capture_body.find(
            "canonical_device_cached_token_observation_failed"),
        std::string::npos);
    EXPECT_EQ(
        compact_capture.find(
            "intcached_tokens=restorablePrefixCachedTokens("),
        std::string::npos)
        << "GPU prefix payload capture must not initialize from a host mirror.";

    EXPECT_NE(
        compact_probe_fallback.find(
            "deviceResidentLogicalTokenCountForObservation("),
        std::string::npos);
    EXPECT_NE(compact_probe_fallback.find("throwstd::runtime_error("),
              std::string::npos)
        << "A missing canonical GPU observer must be fatal.";
    EXPECT_EQ(compact_probe_fallback.find("state_.positions"),
              std::string::npos);
    EXPECT_EQ(compact_probe_fallback.find("state_.sequence_lengths"),
              std::string::npos);
}

/**
 * @brief Keep ordinary GPU state summaries free of logical-metadata D2H.
 *
 * GPU sequence position is an execution-owned device publication. Benchmark,
 * server logging, and PerfStats summaries must not materialize it merely to
 * populate convenience vectors. A focused deep diagnostic is an explicit
 * exception: it first stages the complete packed publication D2D and exports
 * only the diagnostic owner at the host-visible result boundary.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     GPUStateProbeSummaryKeepsLogicalMetadataDeviceOwned)
{
    const auto root = repoRoot();
    const auto source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto policy_header = readFile(
        root / "src/v2/execution/prefix_cache/PrefixCacheStateProbe.h");
    const auto policy_source = readFile(
        root / "src/v2/execution/prefix_cache/PrefixCacheStateProbe.cpp");

    const auto probe_body = sliceBetween(
        source,
        "PrefixRuntimeStateSnapshot DeviceGraphOrchestrator::prefixStateProbe() const",
        "void DeviceGraphOrchestrator::disablePrefixCacheForRunner(");
    const auto compact_probe =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(probe_body));
    const auto compact_source =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(source));
    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto compact_policy_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(policy_header));
    const auto compact_policy_source = removeAsciiWhitespace(policy_source);

    EXPECT_NE(
        compact_policy_header.find("boolcapture_device_logical_state=false;"),
        std::string::npos)
        << "Logical-state export must be opt-in, never the runtime-summary default.";
    EXPECT_NE(
        compact_policy_source.find(
            "LLAMINAR_PREFIX_PROBE_CAPTURE_DEVICE_LOGICAL_STATE"),
        std::string::npos);
    EXPECT_NE(
        compact_source.find(
            "PrefixProbeCapturePolicy::fromEnvironment()."
            "capture_device_logical_state"),
        std::string::npos)
        << "Arena planning must know about the deep-probe policy before allocation.";

    const size_t summary_gate = compact_probe.find(
        "if(state_.device_id.is_gpu()&&!capture_policy.capture_device_logical_state)");
    ASSERT_NE(summary_gate, std::string::npos);
    EXPECT_NE(compact_probe.find("snapshot.positions.clear();", summary_gate),
              std::string::npos);
    EXPECT_NE(
        compact_probe.find("snapshot.sequence_lengths.clear();", summary_gate),
        std::string::npos);

    const auto compact_live_mailbox_probe =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            probe_body,
            "if (logical_state.valid() &&",
            "else if (state_.device_id.is_gpu() &&")));
    EXPECT_NE(
        compact_live_mailbox_probe.find(
            "capture_policy.capture_device_logical_state"),
        std::string::npos)
        << "The live mailbox export must remain behind the explicit deep-probe gate.";
    EXPECT_NE(
        compact_live_mailbox_probe.find(
            "diagnostic_storage.probeStagingBase()"),
        std::string::npos);
    const size_t stage_copy = compact_live_mailbox_probe.find(
        "backend->deviceCopyAsync(staging_device,storage."
        "initialization_base_cached_tokens_scratch_device,");
    const size_t staged_export = compact_live_mailbox_probe.find(
        "backend->deviceToHostFast(diagnostic_host_values.data(),"
        "diagnostic_storage.base_device,");
    ASSERT_NE(stage_copy, std::string::npos);
    ASSERT_NE(staged_export, std::string::npos);
    EXPECT_LT(stage_copy, staged_export)
        << "Deep diagnostics must snapshot D2D before exporting their isolated owner.";
    EXPECT_EQ(
        compact_live_mailbox_probe.find(
            "deviceToHostFast(packed_state.data(),storage."
            "initialization_base_cached_tokens_scratch_device"),
        std::string::npos)
        << "Direct host DMA from the live logical-state owner is forbidden.";

    EXPECT_NE(compact_header.find("kProbeStagingRowCount"), std::string::npos);
    EXPECT_NE(compact_header.find("probeStagingBase()const"), std::string::npos);
}

/**
 * @brief Prevent rollback capture from rediscovering a GPU logical cursor.
 *
 * The request scheduler owns the exact transaction position. The checkpoint
 * API carries that cursor explicitly through Rank/Global orchestrators into
 * every device participant, while the participant archives canonical KV
 * metadata D2D. A host mirror, scalar D2H observation, or stream sync in this
 * production path would recreate the stale-state failure this contract removes.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     GPURollbackCheckpointRequiresSchedulerOwnedCursor)
{
    const auto root = repoRoot();
    const auto snapshot_header = readFile(
        root / "src/v2/execution/prefix_cache/PrefixStateSnapshot.h");
    const auto runner_interface = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto orchestrator_source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto runner_source = readFile(
        root / "src/v2/execution/runner/OrchestrationRunner.cpp");

    const auto compact_snapshot =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(snapshot_header));
    const auto compact_interface =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(runner_interface));
    const auto compact_checkpoint =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(sliceBetween(
                orchestrator_source,
                "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(",
                "bool DeviceGraphOrchestrator::restoreLivePrefixState(")));
    const auto compact_decode =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(sliceBetween(
                runner_source,
                "GenerationResult OrchestrationRunner::decodeStepMTP(",
                "GenerationResult OrchestrationRunner::decodeStep(")));

    EXPECT_NE(
        compact_snapshot.find(
            "structPrefixCheckpointCaptureRequest{intsequence_index=-1;"
            "intlogical_cached_tokens=-1;"),
        std::string::npos)
        << "A default-constructed checkpoint cursor must remain invalid.";
    EXPECT_NE(
        compact_interface.find(
            "captureLivePrefixCheckpoint("
            "constPrefixCheckpointCaptureRequest&request)const"),
        std::string::npos);
    EXPECT_EQ(
        compact_interface.find(
            "captureLivePrefixCheckpoint(intseq_idx"),
        std::string::npos)
        << "The ambiguous host-position checkpoint API must stay retired.";

    EXPECT_NE(compact_checkpoint.find("if(!request.valid())"),
              std::string::npos);
    EXPECT_NE(
        compact_checkpoint.find(
            "constintcached_tokens=request.logical_cached_tokens;"),
        std::string::npos);
    EXPECT_EQ(compact_checkpoint.find("state_.positions"),
              std::string::npos);
    EXPECT_EQ(
        compact_checkpoint.find(
            "deviceResidentLogicalTokenCountForObservation("),
        std::string::npos);
    EXPECT_EQ(compact_checkpoint.find("deviceToHost"),
              std::string::npos);
    EXPECT_EQ(compact_checkpoint.find("synchronizeStream("),
              std::string::npos);
    EXPECT_EQ(compact_checkpoint.find("synchronizeDevice("),
              std::string::npos);

    EXPECT_NE(
        compact_decode.find(
            "currentDecodeTransactionPositionForPlanning(context,error)"),
        std::string::npos);
    EXPECT_NE(
        compact_decode.find(
            "captureLivePrefixCheckpoint(*capture_request)"),
        std::string::npos);
    EXPECT_EQ(
        compact_decode.find("captureLivePrefixCheckpoint()"),
        std::string::npos)
        << "Every scalar MTP checkpoint must name its scheduler cursor.";
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
                  "advanceMTPRequestBatchConditionOnDevice("),
              std::string::npos)
        << "GPU request batches must consume the prior response token in one "
           "resident grouped main decode before drafting.";
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

    const auto condition_advance = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::advanceMTPRequestBatchConditionOnDevice(",
            "bool DeviceGraphOrchestrator::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(")));
    EXPECT_NE(condition_advance.find(
                  "logical_state.next_condition_tokens_device"),
              std::string::npos);
    EXPECT_NE(condition_advance.find(
                  "logical_state.target_positions_device"),
              std::string::npos);
    EXPECT_NE(condition_advance.find(
                  "request_batch,ForwardExecutionRole::MTPCondition,false,true,logical_state.target_positions_device,logical_state.target_sequence_lengths_device"),
              std::string::npos)
        << "RB>1 condition rows must execute decode semantics with mailbox-owned "
           "positions and recurrent-state lengths, not padded prefill metadata.";
    EXPECT_NE(condition_advance.find(
                  "backend->enqueueInitializeMTPDeviceLogicalState("),
              std::string::npos)
        << "The sampled first target rows must republish the resident mailbox.";
    EXPECT_EQ(condition_advance.find("deviceToHost"), std::string::npos);
    EXPECT_EQ(condition_advance.find("hostToDevice"), std::string::npos);
    EXPECT_EQ(condition_advance.find("runner_->get_position("),
              std::string::npos);

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

    const auto request_batch_condition = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            runner_source,
            "bool request_batch_condition_advanced = false;",
            "std::vector<int32_t> condition_tokens;")));
    EXPECT_NE(request_batch_condition.find(
                  "runner_->advanceMTPRequestBatchConditionOnDevice("),
              std::string::npos);
    EXPECT_NE(request_batch_condition.find(
                  "runner_->deviceResidentLogicalSequenceState()"),
              std::string::npos)
        << "The sidecar must consume the mailbox republished after the main condition forward.";
    EXPECT_EQ(request_batch_condition.find("runner_->forward("),
              std::string::npos)
        << "The request-batched GPU lane must not replay scalar condition forwards.";

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
                  "DeviceResidentLogicalStateReadScope"),
              std::string::npos);
    EXPECT_NE(request_batch_verifier.find(
                  "threshold_position_read->ready()"),
              std::string::npos)
        << "Resident stochastic positions must be admitted through the scoped ready-event contract.";
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

    const auto decode_step_batch = sliceBetween(
        runner_source,
        "GenerationBatchResult OrchestrationRunner::decodeStepBatch(",
        "bool OrchestrationRunner::shouldUseMTPDecode() const");
    const auto prefill_runner = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            decode_step_batch,
            "if (has_ready_prefill_logits)",
            "if (!state.prefill_logits_ready)")));
    EXPECT_NE(prefill_runner.find("device_prefill_position_seeds"),
              std::string::npos);
    EXPECT_EQ(prefill_runner.find("device_prefill_thresholds"),
              std::string::npos);
    EXPECT_EQ(prefill_runner.find("mtp_spec_threshold_from_seed("),
              std::string::npos)
        << "First-token thresholds must be derived from resident positions in GPU kernels.";

    const auto prefill_admission_source = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::admitRequestInputsOnDevice(",
        "bool DeviceGraphOrchestrator::initializeDeviceResidentLogicalSequenceStateFromMainBatchSamples(");
    const auto prefill_admission = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(prefill_admission_source));
    EXPECT_NE(prefill_admission.find("request_token_ids_dev_"),
              std::string::npos);
    EXPECT_NE(prefill_admission.find("request_position_ids_dev_"),
              std::string::npos);
    EXPECT_NE(prefill_admission.find("request_sequence_lengths_dev_"),
              std::string::npos);
    EXPECT_NE(prefill_admission.find("request_real_lengths"),
              std::string::npos)
        << "Admission must receive logical request geometry as an explicit transaction input.";
    EXPECT_NE(prefill_admission.find("request_batch_geometry_host_"),
              std::string::npos)
        << "Asynchronous length-and-stride admission requires one stable source record.";
    EXPECT_NE(
        prefill_admission.find(
            "request_batch_geometry_layout_.rowStrideIndex()"),
        std::string::npos)
        << "Admission must publish the padded stride in the same geometry epoch as every request length.";
    EXPECT_NE(
        prefill_admission.find(
            "hostToDeviceOnStream(request_batch_geometry_dev_,"
            "request_batch_geometry_host_.data(),geometry_bytes"),
        std::string::npos)
        << "The complete request geometry must cross H2D as one ordered transfer.";
    EXPECT_EQ(prefill_admission.find("request_sequence_lengths_host_"),
              std::string::npos)
        << "The retired lengths-only host staging vector must not return.";
    EXPECT_EQ(prefill_admission.find("state_.sequence_lengths"),
              std::string::npos)
        << "Mutable request progress must never serve as an asynchronous H2D source.";
    EXPECT_NE(prefill_admission.find("hostToDeviceOnStream("),
              std::string::npos)
        << "External prompt inputs must cross H2D once at request admission.";
    EXPECT_NE(
        prefill_admission.find(
            "request_padding_token_ids_host_.data()"),
        std::string::npos)
        << "Admission must deterministically materialize every token row that a "
           "larger captured prefill bucket can expose to embedding";
    EXPECT_NE(
        prefill_admission.find(
            "request_input_row_capacity_-total_tokens"),
        std::string::npos)
        << "The padding proof must cover the complete stable request-input bank, "
           "not merely one currently selected bucket";
    EXPECT_NE(
        prefill_admission.find(
            "DeviceTimelinePoint::RequestInputAdmission"),
              std::string::npos)
        << "Request admission must publish its writer-to-reader edge through the declarative timeline.";
    EXPECT_NE(
        prefill_admission.find(
            "DeviceTimelinePoint::RequestInputReuseReady"),
        std::string::npos)
        << "The next admission must wait for the prior reader-to-writer release edge.";
    EXPECT_NE(
        prefill_admission.find(
            "waitForPendingShiftedMTPKVReadyForObservation("),
        std::string::npos)
        << "Input-bank release must join any shifted-MTP reader without stealing its live-state handoff.";
    EXPECT_NE(
        prefill_admission_source.find("device_input_reuse_publications"),
        std::string::npos)
        << "Perfstats must prove that the complete reader chain released the reusable input bank.";
    EXPECT_NE(
        prefill_admission_source.find("device_input_reader_admissions"),
        std::string::npos)
        << "Every host or device writer must enter one shared, countable reader transaction before release.";
    EXPECT_EQ(
        countOccurrences(
            removeAsciiWhitespace(
                stripCommentsAndStringLiterals(orchestrator_source)),
            "reuse.consumers_started=true;"),
        1u)
        << "Only beginRequestInputReaderTransaction may transfer the reusable bank to graph readers.";
    EXPECT_EQ(prefill_admission.find("createEvent("),
              std::string::npos)
        << "Request-input event ownership must be preallocated with the arena, never in the hot path.";
    EXPECT_EQ(prefill_admission.find("synchronizeStream("),
              std::string::npos)
        << "Request admission must never block the host before graph capture.";
    EXPECT_EQ(prefill_admission.find("synchronizeDevice("),
              std::string::npos);
    EXPECT_EQ(prefill_admission.find("deviceToHost"),
              std::string::npos)
        << "The request-input lifecycle is fully device ordered and has no host observation.";
    EXPECT_EQ(prefill_admission.find("publishPendingLogitsStream("),
              std::string::npos)
        << "Request-length admission is a pre-capture ownership boundary, not a logits-stream handoff.";

    const auto device_verifier_forward = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::forwardBatchWithDeviceTokenIds(",
            "const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDevice(")));
    EXPECT_EQ(device_verifier_forward.find(
                  "admitRequestInputsOnDevice("),
              std::string::npos)
        << "Grouped verifier geometry must not overwrite prefill-admitted logical positions or add a second H2D boundary.";

    const auto main_forward = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "const float *DeviceGraphOrchestrator::forwardImpl(",
            "bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule(")));
    EXPECT_NE(main_forward.find(
                  "admitRequestInputsOnDevice(tokens,position_ids.data(),request_real_lengths,total_tokens,batch_size,seq_len)"),
              std::string::npos)
        << "Ordinary GPU prefill must admit tokens, positions, and lengths as one transaction.";
    EXPECT_NE(main_forward.find(
                  "input.token_ids_device=effective_token_ids_device"),
              std::string::npos);
    EXPECT_NE(main_forward.find(
                  "input.position_ids_device=effective_position_ids_device"),
              std::string::npos);
    EXPECT_NE(
        main_forward.find(
            "effective_position_ids_device=request_position_ids_dev_"),
        std::string::npos)
        << "GPU serial decode must capture a stable arena position row.";
    EXPECT_NE(
        main_forward.find(
            "input.materialize_serial_decode_position_from_device_kv=materialize_serial_decode_position_from_device_kv"),
        std::string::npos)
        << "The execution prelude must receive explicit typed policy for the "
           "device-owned serial position snapshot.";
    EXPECT_NE(main_forward.find(
                  "publishRequestInputReuseReady(output.execution.stream,"),
              std::string::npos)
        << "Monolithic prefill must release the request-input bank from its exact final consumer stream.";

    const auto forward_live_state_prelude = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(",
            "const float *DeviceGraphOrchestrator::getAllPositionLogits() const")));
    EXPECT_NE(
        forward_live_state_prelude.find(
            "deviceSequenceCachedTokenCountPtr("),
        std::string::npos)
        << "Serial decode positions must derive from authoritative device KV state.";
    EXPECT_NE(
        forward_live_state_prelude.find(
            "enqueuePrepareMTPVerifierPositionIds(live_position_device,"),
        std::string::npos)
        << "The scalar decode snapshot must use the backend's device-only "
           "position materializer.";
    EXPECT_NE(
        forward_live_state_prelude.find(
            "DeviceTimelinePoint::RequestInputReuseReady"),
        std::string::npos)
        << "Position-row reuse must be ordered by the declarative event timeline.";
    EXPECT_EQ(forward_live_state_prelude.find("hostToDevice"),
              std::string::npos);
    EXPECT_EQ(forward_live_state_prelude.find("deviceToHost"),
              std::string::npos);
    EXPECT_EQ(forward_live_state_prelude.find("synchronizeStream("),
              std::string::npos);
    EXPECT_EQ(forward_live_state_prelude.find("synchronizeDevice("),
              std::string::npos)
        << "Serial decode position publication must remain fully device ordered.";

    const auto chunked_prefill = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::forwardPrefillChunkSchedule(",
            "bool DeviceGraphOrchestrator::ensureMTPTerminalHiddenBuffer(")));
    EXPECT_NE(chunked_prefill.find(
                  "publishRequestInputReuseReady(output.execution.stream,"),
              std::string::npos)
        << "Chunk-scheduled prefill must release the same reusable input bank.";
    EXPECT_EQ(chunked_prefill.find("synchronizeStream("),
              std::string::npos)
        << "Input-bank lifetime closure must stay fully event ordered.";

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
                  "commitMTPShiftedRowFromDeviceTargetSample(0,0,true)"),
              std::string::npos)
        << "GPU direct emit must publish shifted MTP state from its persistent "
           "target-sample slot without passing a host position.";
    EXPECT_NE(compact_direct_emit.find(
                  "commitMTPShiftedRowFromCurrentTerminalHidden(first_token,0,true,base_sidecar_position)"),
              std::string::npos)
        << "CPU direct emit must retain the checkpoint-derived scalar anchor.";
    EXPECT_NE(compact_direct_emit.find(
                  "advanceMTPMainConditionFromDeviceTargetSample(first_token,0)"),
              std::string::npos)
        << "GPU direct emit must publish and consume an inseparable resident "
           "token/position row.";
    EXPECT_EQ(
        compact_direct_emit.find(
            "forwardWithDeviceTokenIds(&first_token"),
        std::string::npos)
        << "Token-only main condition replay must stay retired.";
    EXPECT_EQ(compact_direct_emit.find("currentDecodeTransactionPositionForPlanning("),
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
    EXPECT_NE(
        compact_condition_forward.find(
            "advanceMTPMainConditionFromDeviceResidentLogicalState("
            "condition_token,condition_state,0)"),
        std::string::npos)
        << "GPU condition replay must consume the current resident token and "
           "logical-position publication together.";
    EXPECT_NE(condition_forward_body.find("condition_forward_shifted_commits"),
              std::string::npos)
        << "The maintenance path should remain visible in perfstats.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, ResidentRequestBatchConditionHasNoHostRowShadow)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto condition_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::advanceMTPRequestBatchConditionOnDevice(",
            "bool DeviceGraphOrchestrator::stageStochasticTargetTokenForDeviceSampling(")));
    const auto forward_body = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "const float *DeviceGraphOrchestrator::forwardImpl(",
            "bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule(")));

    EXPECT_EQ(condition_body.find("condition_shadows"), std::string::npos)
        << "A resident condition graph must not allocate a host token mirror.";
    EXPECT_NE(
        condition_body.find(
            "forwardImpl(nullptr,logical_state.next_condition_tokens_device,1,request_batch,ForwardExecutionRole::MTPCondition"),
        std::string::npos)
        << "The grouped condition forward must bind only its device token row.";
    EXPECT_NE(
        forward_body.find(
            "input.position_ids=effective_position_ids_device?nullptr:position_ids.data()"),
        std::string::npos)
        << "A device position owner must suppress the host position pointer.";
    EXPECT_NE(
        forward_body.find("if(!live_request_batch_condition)"),
        std::string::npos)
        << "Resident condition progress must not advance a competing host cursor.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GPUScalarMTPConditionRequiresPairedResidentTokenAndPosition)
{
    const auto orchestrator_source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto resident_advance = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "advanceMTPMainConditionFromDeviceResidentLogicalState(",
            "advanceMTPMainConditionFromDeviceTargetSample(")));
    const auto target_advance = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "advanceMTPMainConditionFromDeviceTargetSample(",
            "bool DeviceGraphOrchestrator::forwardBatchWithDeviceTokenIds(")));

    EXPECT_NE(
        resident_advance.find(
            "logical_state.nextConditionTokenDeviceForRequest(request_index)"),
        std::string::npos);
    EXPECT_NE(
        resident_advance.find(
            "logical_state.targetPositionDeviceForRequest(request_index)"),
        std::string::npos);
    EXPECT_NE(
        resident_advance.find(
            "logical_state.targetSequenceLengthDeviceForRequest(request_index)"),
        std::string::npos);
    EXPECT_NE(
        resident_advance.find(
            "forwardImpl(&token_shadow,condition_token_device,1,1,"
            "ForwardExecutionRole::MTPCondition,false,true,"
            "condition_position_device,condition_sequence_length_device)"),
        std::string::npos)
        << "The captured condition graph must bind the paired mailbox token, "
           "position, and sequence length.";

    EXPECT_NE(
        target_advance.find(
            "publishDeviceResidentLogicalSequenceStateFromTargetSample("
            "target_sample_slot,live_position_device)"),
        std::string::npos);
    EXPECT_NE(
        target_advance.find(
            "advanceMTPMainConditionFromDeviceResidentLogicalState("
            "token_shadow,logical_state,0)"),
        std::string::npos)
        << "Target slots must enter the same typed mailbox path rather than a "
           "token-only replay API.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, DGOResidentPublicationDoesNotMutateKVBeforeLogicalStateIsResident)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto stage_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.cpp");
    const auto support_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::supportsDeviceResidentLogicalSequenceStatePublication() const",
        "bool DeviceGraphOrchestrator::materializeMTPSpeculativeStatePublicationGraph(");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(",
        "std::vector<ForwardExecutionEngine::ReplayCacheObservation>");
    const auto compact_support =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(support_body));
    const auto compact_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_stage =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(stage_source));

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

    const size_t kv_support = compact_prepare.find("supportsDeviceResidentSequenceStatePublication()");
    const size_t dgo_support = compact_prepare.find("supportsDeviceResidentLogicalSequenceStatePublication()");
    const size_t graph_materialize = compact_prepare.find("materializeMTPSpeculativeStatePublicationGraph(");
    ASSERT_NE(kv_support, std::string::npos);
    ASSERT_NE(dgo_support, std::string::npos);
    ASSERT_NE(graph_materialize, std::string::npos);
    EXPECT_LT(kv_support, dgo_support);
    EXPECT_LT(dgo_support, graph_materialize)
        << "The strict publication graph cannot be materialized before both ownership gates pass.";
    EXPECT_NE(compact_stage.find("publishSequenceStateFromDeviceMetadata("),
              std::string::npos)
        << "KV mutation belongs to the captured stage proven by the preflight gates.";
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
    EXPECT_NE(compact.find("enumclassDeviceSequenceStatePublicationBasis"),
              std::string::npos)
        << "GPU publication must name the immutable state basis instead of "
           "inferring ring ownership from a target count.";
    EXPECT_NE(compact.find("CapturedBase"),
              std::string::npos);
    EXPECT_NE(compact.find("CurrentVisibleWindow"),
              std::string::npos);
    EXPECT_NE(compact.find("base_sequence_state_checkpoint_device"),
              std::string::npos);
    EXPECT_NE(compact.find("base_sequence_state_checkpoint_bytes"),
              std::string::npos)
        << "Captured-base publication must carry the opaque pre-verifier "
           "head/count checkpoint needed to reconstruct wrapped ring state.";
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
    EXPECT_NE(
        source.find(
            "cannot reconstruct the pre-verifier"),
        std::string::npos)
        << "The contract must document why target count alone cannot recover "
           "a wrapped ring's immutable pre-verifier state.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, GPUKVLogicalBlockAccessRequiresExplicitStreams)
{
    const auto interface_source =
        readFile(repoRoot() / "src/v2/kernels/IKVCache.h");
    const auto cuda_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto rocm_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp");
    const auto cuda_tq_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu");
    const auto rocm_tq_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.hip");

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
    const auto cuda_tq_export = sliceBetween(
        cuda_tq_source,
        "bool CUDARingKVCacheTQ::exportLogicalBlock(",
        "bool CUDARingKVCacheTQ::importLogicalBlock(");
    const auto cuda_tq_import = sliceBetween(
        cuda_tq_source,
        "bool CUDARingKVCacheTQ::importLogicalBlock(",
        "bool CUDARingKVCacheTQ::get_kv(");
    const auto rocm_tq_export = sliceBetween(
        rocm_tq_source,
        "bool ROCmRingKVCacheTQ::exportLogicalBlock(",
        "bool ROCmRingKVCacheTQ::importLogicalBlock(");
    const auto rocm_tq_import = sliceBetween(
        rocm_tq_source,
        "bool ROCmRingKVCacheTQ::importLogicalBlock(",
        "bool ROCmRingKVCacheTQ::get_kv(");

    const auto compact_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(interface_source));
    const std::vector<std::string> bodies = {
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_export)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_import)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_export)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_import)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_tq_export)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_tq_import)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_tq_export)),
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_tq_import)),
    };

    EXPECT_NE(interface_source.find("GPU implementations require @ref stream"),
              std::string::npos)
        << "The API contract must tell callers that GPU logical KV access is stream-owned.";
    EXPECT_NE(compact_interface.find("void*stream=nullptr"),
              std::string::npos)
        << "CPU callers may still omit streams; GPU implementations enforce when streams are required.";

    for (const auto &body : bodies)
    {
        EXPECT_NE(
            body.find("requireGPUExecutionStream(desc.stream,"),
            std::string::npos)
            << "Every ordinary and TurboQuant GPU logical KV operation must "
               "throw at its API boundary when no exact stream is supplied.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, GPUKVCheckpointAndTruncateStayDeviceResidentAcrossBackends)
{
    const auto interface_source =
        readFile(repoRoot() / "src/v2/kernels/IKVCache.h");
    const auto cuda_base_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.cpp");
    const auto rocm_base_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp");
    const auto cuda_cache_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto rocm_cache_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp");
    const auto cuda_tq_header =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.h");
    const auto rocm_tq_header =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.h");

    const auto compact_interface =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(interface_source));
    EXPECT_NE(compact_interface.find("deviceSequenceStateCheckpointBytes()const"),
              std::string::npos);
    EXPECT_NE(compact_interface.find("captureDeviceSequenceStateCheckpoint("),
              std::string::npos);
    EXPECT_NE(compact_interface.find("restoreDeviceSequenceStateCheckpoint("),
              std::string::npos);

    const std::array<std::pair<std::string, std::string>, 2> backend_sources = {{
        {"CUDA", cuda_base_source},
        {"ROCm", rocm_base_source},
    }};
    for (const auto &[backend_name, source] : backend_sources)
    {
        const std::string class_name =
            backend_name == "CUDA" ? "CUDARingKVCacheBase" : "ROCmRingKVCacheBase";
        const auto truncate = removeAsciiWhitespace(stripCommentsAndStringLiterals(
            sliceBetween(
                source,
                "bool " + class_name + "::truncateSequence(",
                "bool " + class_name + "::observeDeviceSequenceState(")));
        const auto checkpoint = removeAsciiWhitespace(stripCommentsAndStringLiterals(
            sliceBetween(
                source,
                "size_t " + class_name + "::deviceSequenceStateCheckpointBytes() const",
                "const int *" + class_name + "::deviceDynamicAppendCountPtr(")));

        EXPECT_NE(truncate.find("kv_sequence_state_truncate("), std::string::npos)
            << backend_name << " truncate must be one device kernel over every layer.";
        EXPECT_EQ(truncate.find("observeDeviceSequenceState("), std::string::npos);
        EXPECT_EQ(truncate.find("DeviceSynchronize("), std::string::npos);
        EXPECT_EQ(truncate.find("StreamSynchronize("), std::string::npos);
        EXPECT_EQ(truncate.find("Memcpy"), std::string::npos);
        EXPECT_EQ(truncate.find("std::vector"), std::string::npos);

        EXPECT_NE(checkpoint.find("kv_sequence_state_checkpoint_capture("),
                  std::string::npos);
        EXPECT_NE(checkpoint.find("kv_sequence_state_checkpoint_restore("),
                  std::string::npos);
        EXPECT_EQ(checkpoint.find("observeDeviceSequenceState("), std::string::npos);
        EXPECT_EQ(checkpoint.find("DeviceSynchronize("), std::string::npos);
        EXPECT_EQ(checkpoint.find("StreamSynchronize("), std::string::npos);
        EXPECT_EQ(checkpoint.find("Memcpy"), std::string::npos);
        EXPECT_EQ(checkpoint.find("std::vector"), std::string::npos);
    }

    const auto cuda_derived_truncate =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            cuda_cache_source,
            "bool CUDARingKVCache<Precision>::truncateSequence(",
            "template <ActivationPrecision Precision>")));
    const auto rocm_derived_truncate =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            rocm_cache_source,
            "bool ROCmRingKVCache<Precision>::truncateSequence(",
            "template <ActivationPrecision Precision>")));
    EXPECT_NE(cuda_derived_truncate.find("CUDARingKVCacheBase::truncateSequence("),
              std::string::npos);
    EXPECT_NE(rocm_derived_truncate.find("ROCmRingKVCacheBase::truncateSequence("),
              std::string::npos);

    EXPECT_EQ(cuda_tq_header.find("truncateSequence("), std::string::npos)
        << "CUDA TurboQuant must inherit the one canonical device truncate path.";
    EXPECT_EQ(rocm_tq_header.find("truncateSequence("), std::string::npos)
        << "ROCm TurboQuant must inherit the one canonical device truncate path.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, KVCacheDevicePublicationPrimitiveIsWrappedRingSafe)
{
    const auto cuda_base_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.cpp");
    const auto rocm_base_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp");
    const auto cuda_base =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.h") +
        cuda_base_source +
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto rocm_base =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.h") +
        rocm_base_source +
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheKernels.hip");
    const auto cuda_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cuda_base));
    const auto rocm_compact =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rocm_base));
    const auto cuda_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            cuda_base_source,
            "bool CUDARingKVCacheBase::publishSequenceStateFromDeviceMetadata(",
            "\n\n} // namespace llaminar2")));
    const auto rocm_publish =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(sliceBetween(
            rocm_base_source,
            "bool ROCmRingKVCacheBase::publishSequenceStateFromDeviceMetadata(",
            "\n\n} // namespace llaminar2")));

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

    for (const auto *publish : {&cuda_publish, &rocm_publish})
    {
        EXPECT_EQ(publish->find("deviceToHost"), std::string::npos);
        EXPECT_EQ(publish->find("hostToDevice"), std::string::npos);
        EXPECT_EQ(publish->find("Memcpy"), std::string::npos)
            << "Captured KV publication must enqueue only its device metadata kernel.";
        EXPECT_EQ(publish->find("synchronize"), std::string::npos)
            << "Diagnostic modes must not turn a graph-capturable publication primitive into a host barrier.";
        EXPECT_EQ(publish->find("malloc"), std::string::npos);
        EXPECT_EQ(publish->find("free("), std::string::npos)
            << "Publication storage belongs to setup-owned KV and graph workspaces.";
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
    /*
     * MTP spec-state publication replaces live main/MTP KV and recurrent state
     * with verifier-captured rows. GPU sidecar replay is mandatory because
     * graph construction gives dense and MoE sidecars persistent, non-aliasing
     * dynamic storage. GPU main graph replay uses a typed lifetime policy:
     * byte-proven single-token decode and all-position verifier captures remain
     * warm, while genuinely live-state-versioned multi-row ordinary decode is
     * invalidated.
     */
    EXPECT_EQ(header.find("preservesMTPSidecarReplayAfterSpecPublication"),
              std::string::npos)
        << "GPU sidecar replay is a hard ownership contract, not an optional "
           "capability advertisement.";
    const auto executable_publish_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publish_body));
    EXPECT_NE(executable_publish_body.find("plan.requiresCorrectionReplay()?"),
              std::string::npos)
        << "Spec-state publication diagnostics must distinguish accepted "
           "publication from rejected correction.";
    EXPECT_NE(executable_publish_body.find(
                  "handleLivePrefixReplayStateAfterMutation(mutation_reason,)"),
              std::string::npos)
        << "MTP accepted-state publication must keep a main/verifier replay-state "
           "mutation boundary.";
    EXPECT_EQ(mutation_body.find("preserve_gpu_replay_state"),
              std::string::npos)
        << "Callers must not bypass the typed graph-lifetime policy with a raw boolean.";

    const auto executable_mutation_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mutation_body));
    EXPECT_NE(executable_mutation_body.find(
                  "constboolpreserve_sidecar_replay=state_.device_id.is_gpu()&&(correction_replay_boundary||reason==LivePrefixMutationReason::PrefixRestore);"),
              std::string::npos)
        << "Accepted/rejected publication and prefix restore must preserve GPU "
           "sidecar captures through the persistent-buffer contract.";
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
    EXPECT_EQ(mutation_body.find("prefix_restore_discard_cached_graphs"),
              std::string::npos)
        << "Prefix restore must not destroy graph executables and recapture the "
           "first continuation token.";
    EXPECT_NE(executable_mutation_body.find(
                  "rebindCapturedReplayStateAfterPrefixRestore(live_replay_state_epoch_)"),
              std::string::npos)
        << "Prefix restore must use the typed replay policy and explicit stream "
           "rebind contract.";
    EXPECT_NE(mutation_body.find("prefix_restore_typed_live_state"),
              std::string::npos)
        << "Perf stats must identify typed graph reuse at prefix restore.";
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
    EXPECT_NE(mutation_body.find("preserved_for_prefix_restore"),
              std::string::npos)
        << "Perf stats must make prefix-restore sidecar reuse explicit.";
    EXPECT_NE(executable_mutation_body.find(
                  "if(preserves_live_graph_replay)"),
              std::string::npos)
        << "A typed live-state boundary must not globally reset kernel dynamic "
           "state while graph executables are preserved.";
    EXPECT_NE(mutation_body.find("preserved_for_live_state_graph_replay"),
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
    const auto shifted_replay_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::recordShiftedMTPKVReady(",
        "bool DeviceGraphOrchestrator::waitForPendingShiftedMTPKVReady(");
    const auto spec_publication_support_body = sliceBetween(
        dgo_source,
        "bool DeviceGraphOrchestrator::supportsMTPSpecStatePublication() const",
        "MTPVerifierRowCapability DeviceGraphOrchestrator::mtpVerifierRowCapability() const");

    const auto compact_ffn =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(ffn_body));
    const auto compact_raw_ffn = removeAsciiWhitespace(ffn_body);
    const auto compact_replay =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(shifted_replay_body));
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
    EXPECT_EQ(compact_replay.find("handleLivePrefixReplayStateAfterMutation("),
              std::string::npos)
        << "A shifted-KV update must never invoke the broad replay reset path.";
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
        "if (is_gpu)");
    const auto compact_verifier_replay =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(verifier_replay_body));
    const auto compact_predicate =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(routing_tensor_predicate));
    const auto compact_body =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(routing_tensor_body));

    EXPECT_NE(compact_verifier_replay.find("params_.require_device_routing_tensor_decode=is_gpu;"),
              std::string::npos)
        << "GPU decode-equivalent verifier replay must use the explicit routing-tensor path.";
    EXPECT_EQ(compact_predicate.find("hasFullLocalExpertOwnership()"),
              std::string::npos)
        << "The mask-aware routing-tensor path must admit static apportioned "
           "LocalTP participants that own only their assigned complete experts.";
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
    EXPECT_NE(compact_body.find("kernel->groupedExpertDecodeFromRouting("),
              std::string::npos)
        << "Explicit routing must use one fused workspace-native backend contract.";
    EXPECT_EQ(compact_body.find("groupedExpertGateUpDecodeFromRouting("),
              std::string::npos)
        << "The production stage must not recreate per-slot gate/up tensors.";
    EXPECT_EQ(compact_body.find("groupedExpertDownDecodeFromRouting("),
              std::string::npos)
        << "The production stage must not retain the split explicit-routing down path.";
    EXPECT_NE(routing_tensor_predicate.find(
                  "masked-off top-k slots to -1"),
              std::string::npos)
        << "The non-obvious mask/ownership split needs inline guidance.";
    EXPECT_NE(routing_tensor_predicate.find(
                  "Do not require full local ownership here"),
              std::string::npos)
        << "The apportioned ownership invariant must remain explicit beside "
           "the production-path predicate.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, TypedMTPLogitsReuseOnlyPreplannedDeviceOutputStorage)
{
    const auto orchestrator_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto helper_body = sliceBetween(
        orchestrator_source,
        "DeviceGraphOrchestrator::preplannedMTPLogitsBuffer(",
        "void DeviceGraphOrchestrator::releaseBuffers()");
    const auto initialization_body = sliceBetween(
        orchestrator_source,
        "bool DeviceGraphOrchestrator::initializeBuffers(",
        "bool DeviceGraphOrchestrator::initializeMTPKVCaches(");
    const auto forward_body = sliceBetween(
        orchestrator_source,
        "const float *DeviceGraphOrchestrator::forwardImpl(",
        "bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule(");
    const auto compact_helper =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(helper_body));
    const auto compact_initialization =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(initialization_body));
    const auto compact_forward =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(forward_body));

    EXPECT_NE(compact_helper.find(
                  "ForwardExecutionRole::GroupedMTPVerifier"),
              std::string::npos)
        << "Grouped verifier graphs must use the preplanned MTP output owner.";
    EXPECT_NE(compact_helper.find(
                  "ForwardExecutionRole::MTPCondition"),
              std::string::npos)
        << "Live MTP condition graphs must use the same capacity-owned output.";
    EXPECT_NE(compact_helper.find(
                  "shape.size()!=2||shape[0]<rows||shape[1]!=columns"),
              std::string::npos)
        << "Captured MTP outputs may use any runtime M within the preplanned "
           "row capacity.";
    EXPECT_NE(compact_helper.find("*device!=state_.device_id"),
              std::string::npos)
        << "Preplanned logits reuse must preserve exact device ownership.";
    EXPECT_NE(compact_helper.find("candidate->isMapped()"),
              std::string::npos)
        << "GPU MTP graph outputs must never adopt host-visible mapped storage.";
    EXPECT_NE(compact_helper.find("candidate->gpu_data_ptr()==nullptr"),
              std::string::npos);
    EXPECT_EQ(compact_helper.find("candidate->deviceValid()"),
              std::string::npos)
        << "Unwritten output storage is not required to contain valid input bytes.";
    EXPECT_NE(compact_initialization.find(
                  "arena_->allocateDeviceStorage("
                  "BufferId::MTP_LOGITS,state_.device_id)"),
              std::string::npos)
        << "The stable MTP logits pointer must be allocated before graph construction.";
    EXPECT_NE(compact_forward.find("preplannedMTPLogitsBuffer("),
              std::string::npos);
    EXPECT_NE(compact_forward.find("BufferId::MTP_LOGITS_GATHERED"),
              std::string::npos)
        << "Full-vocabulary grouped outputs should reuse a preplanned gathered "
           "sidecar allocation where that policy owns one.";
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
        const auto apply_transaction_gate = sliceBetween(
            source,
            "__device__ __forceinline__ bool rebalance_apply_transaction_blocked(",
            "__device__ __forceinline__ bool rebalance_config_ok(");
        const auto compact_apply_transaction_gate =
            removeAsciiWhitespace(
                stripCommentsAndStringLiterals(apply_transaction_gate));

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
        EXPECT_NE(compact_root_projection.find("local_plan_entries[local_plan_index]=projected_plan;"),
                  std::string::npos)
            << backend << " should publish the validated command plus its "
                          "participant-local destination-slot lease.";
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
        EXPECT_NE(
            compact_apply_transaction_gate.find(
                "status->missing_source_descriptors!=0u"),
                  std::string::npos)
            << backend << " transaction preflight must block all bank mutation "
                          "after a source-side pack failure.";
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
        << "Partial expert-ID ownership ranges must use the mask/range + allreduce "
           "path instead of the single-device full-owner runtime table.";
    EXPECT_NE(compact.find("debugEnv().moe_rebalance.gpu_cache_experts_per_layer>0"),
              std::string::npos)
        << "GPU expert-cache bootstrap masks also break full-owner runtime-table "
           "semantics and must keep decode capture disabled.";
    EXPECT_NE(compact.find("if(use_expert_overlay)returnfalse"),
              std::string::npos)
        << "Graph-native tiered overlays use per-participant expert masks even "
           "when the tier domain says routed_compute=apportioned; they must not "
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
    EXPECT_NE(
        compact_graph.find(
            "(local_decode_layer||(mtp_sidecar_context&&total_tokens==1))&&"
            "device.is_gpu()&&!use_expert_overlay"),
        std::string::npos)
        << "Static apportioned MTP sidecars must receive the same mask-aware "
           "device runtime-table contract as main decode.";
    EXPECT_EQ(
        compact_graph.find(
            "allow_eager_gpu_single_row_route_for_partial_expert_owner"),
        std::string::npos)
        << "Partial-owner GPU decode must never retain an eager route escape hatch.";

    const auto routing_source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoERoutingStage.cpp");
    const std::string compact_routing =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(routing_source));
    EXPECT_EQ(
        compact_routing.find(
            "allow_eager_gpu_single_row_route_for_partial_expert_owner"),
        std::string::npos)
        << "The routing stage must require a device runtime table for every "
           "single-row GPU topology.";
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
    EXPECT_NE(
        compact.find(
            "requireGPUExecutionStream(static_cast<void*>(stream),"),
        std::string::npos)
        << "Grouped gather must throw before a device-default stream can be used.";
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
        EXPECT_NE(
            compact.find("requireGPUExecutionStream(gpu_stream,"),
            std::string::npos)
            << "Native grouped KV reads must throw before a device-default "
               "stream can be used.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, MoEGraphMetadataUsesExplicitWorkspaceOwnership)
{
    const auto root = repoRoot();
    const auto workspace_header = readFile(
        root / "src/v2/execution/local_execution/device/DeviceWorkspaceManager.h");
    const auto cuda_source = readFile(
        root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto cuda_header = readFile(
        root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.h");
    const auto rocm_source = readFile(
        root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp");
    const auto rocm_header = readFile(
        root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.h");

    EXPECT_NE(
        workspace_header.find("acquirePersistentSlot("),
        std::string::npos)
        << "The device-wide workspace must arbitrate graph-lifetime metadata "
           "ownership instead of letting each kernel independently choose slot zero";
    EXPECT_NE(
        workspace_header.find("getPersistentSlotBuffer("),
        std::string::npos)
        << "Persistent payload addresses must use the workspace table's fixed "
           "stride instead of a request-specific payload stride";
    EXPECT_NE(
        workspace_header.find("getOrCreatePersistentPublication("),
        std::string::npos)
        << "Immutable weights shared by separately captured graph-local kernels "
           "need identity-keyed workspace ownership";

    for (const auto &[backend, source, header] :
         std::array<std::tuple<const char *, const std::string *, const std::string *>, 2>{
             std::tuple{"CUDA", &cuda_source, &cuda_header},
             std::tuple{"ROCm", &rocm_source, &rocm_header}})
    {
        SCOPED_TRACE(backend);
        EXPECT_NE(source->find("acquirePersistentSlot("), std::string::npos)
            << backend << " mutable fixed-topology masks must obtain exclusive "
                          "workspace slots from the shared manager";
        EXPECT_NE(
            source->find("GroupedExpertMaskLeaseDomain"),
            std::string::npos)
            << backend << " fixed-topology masks are immutable captured inputs "
                          "and need the same ownership discipline as descriptors";
        EXPECT_NE(
            source->find("RouterQ8GatePublicationDomain"),
            std::string::npos)
            << backend << " router weight conversions need a shared immutable "
                          "publication domain, not exclusive per-kernel leases";
        EXPECT_NE(
            source->find("getPersistentSlotBuffer("),
            std::string::npos)
            << backend << " router cache slots must use domain-wide fixed-stride addressing";
        EXPECT_EQ(header->find("source_tensor"), std::string::npos)
            << backend << " persistent router identity must not depend on ephemeral tensor wrappers";
        EXPECT_EQ(header->find("source_host_ptr"), std::string::npos)
            << backend << " device router identity must not adopt a host mirror pointer";
        EXPECT_NE(
            header->find(
                "std::shared_ptr<GroupedDescriptorWorkspacePublication>"),
            std::string::npos)
            << backend << " descriptor tables must retain their exact shared "
                          "publication for the entire graph lifetime";
        EXPECT_EQ(
            source->find("next_grouped_down_desc_workspace_slot_"),
            std::string::npos);
        EXPECT_EQ(
            source->find("next_grouped_gateup_desc_workspace_slot_"),
            std::string::npos);
        EXPECT_EQ(
            header->find("next_grouped_down_desc_workspace_slot_"),
            std::string::npos);
        EXPECT_EQ(
            header->find("next_grouped_gateup_desc_workspace_slot_"),
            std::string::npos);
        EXPECT_EQ(
            source->find("next_router_q8_gate_workspace_slot_"),
            std::string::npos);
        EXPECT_EQ(
            header->find("next_router_q8_gate_workspace_slot_"),
            std::string::npos);

        const auto router_publication = sliceBetween(
            *source,
            backend == std::string("CUDA")
                ? "const CUDAMoEKernel::RouterQ8GateCacheEntry *CUDAMoEKernel::getOrCreateQ8RouterGateCache("
                : "const ROCmMoEKernel::RouterQ8GateCacheEntry *ROCmMoEKernel::getOrCreateQ8RouterGateCache(",
            backend == std::string("CUDA")
                ? "bool CUDAMoEKernel::tryRouteDecodeLogitsQ8("
                : "const void *ROCmMoEKernel::getOrCreateFP16RouterGateCache(");
        const auto executable_router_publication =
            stripCommentsAndStringLiterals(router_publication);
        EXPECT_NE(
            executable_router_publication.find(
                "getOrCreatePersistentPublication("),
            std::string::npos)
            << backend << " graph-local router kernels must converge on one "
                          "workspace-owned immutable conversion";
        EXPECT_EQ(
            executable_router_publication.find("acquirePersistentSlot("),
            std::string::npos)
            << backend << " immutable router weights must not consume one "
                          "exclusive slot per graph-local kernel";
        EXPECT_NE(
            executable_router_publication.find(
                backend == std::string("CUDA")
                    ? "cudaStreamWaitEvent("
                    : "hipStreamWaitEvent("),
            std::string::npos)
            << backend << " first adoption must use an explicit device event edge";
        EXPECT_EQ(
            executable_router_publication.find("synchronize"),
            std::string::npos)
            << backend << " immutable publication must never introduce a host "
                          "or full-device synchronization";
        EXPECT_NE(
            executable_router_publication.find("std::terminate();"),
            std::string::npos)
            << backend << " event-record failure occurs after device submission "
                          "and must be process-fatal";

        const auto down_publication = sliceBetween(
            *source,
            backend == std::string("CUDA")
                ? "bool CUDAMoEKernel::publishGroupedDownDescriptorTable("
                : "bool ROCmMoEKernel::publishGroupedDownDescriptorTable(",
            backend == std::string("CUDA")
                ? "bool CUDAMoEKernel::publishGroupedGateUpDescriptorTable("
                : "bool ROCmMoEKernel::publishGroupedGateUpDescriptorTable(");
        const auto gateup_publication = sliceBetween(
            *source,
            backend == std::string("CUDA")
                ? "bool CUDAMoEKernel::publishGroupedGateUpDescriptorTable("
                : "bool ROCmMoEKernel::publishGroupedGateUpDescriptorTable(",
            backend == std::string("CUDA")
                ? "bool CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace("
                : "bool ROCmMoEKernel::rebindGroupedDescriptorTablesToWorkspace(");
        for (const auto *publication :
             {&down_publication, &gateup_publication})
        {
            const auto executable =
                stripCommentsAndStringLiterals(*publication);
            EXPECT_NE(
                executable.find("getOrCreatePersistentPublication("),
                std::string::npos)
                << backend << " graph variants must converge on one exact "
                              "prepared-weight descriptor publication";
            EXPECT_EQ(
                executable.find("acquirePersistentSlot("),
                std::string::npos)
                << backend << " descriptor ownership must not scale with "
                              "prefill buckets or MTP depths";
            EXPECT_NE(
                executable.find(
                    backend == std::string("CUDA")
                        ? "cudaStreamWaitEvent("
                        : "hipStreamWaitEvent("),
                std::string::npos)
                << backend << " descriptor adoption requires a device event edge";
            EXPECT_EQ(
                executable.find("synchronize"),
                std::string::npos);
            EXPECT_NE(
                executable.find("std::terminate();"),
                std::string::npos)
                << backend << " a submitted descriptor write without a "
                              "publishable event must be process-fatal";
        }
    }
    EXPECT_EQ(
        rocm_source.find("next_router_fp16_gate_workspace_slot_"),
        std::string::npos);
    EXPECT_EQ(
        rocm_header.find("next_router_fp16_gate_workspace_slot_"),
        std::string::npos);

    const auto rocm_fp16_publication = sliceBetween(
        rocm_source,
        "const void *ROCmMoEKernel::getOrCreateFP16RouterGateCache(",
        "bool ROCmMoEKernel::routeCore(");
    const auto executable_rocm_fp16_publication =
        stripCommentsAndStringLiterals(rocm_fp16_publication);
    EXPECT_NE(
        executable_rocm_fp16_publication.find(
            "getOrCreatePersistentPublication("),
        std::string::npos)
        << "The ROCm FP16 router preparation path has the same immutable "
           "workspace ownership contract as Q8";
    EXPECT_EQ(
        executable_rocm_fp16_publication.find("acquirePersistentSlot("),
        std::string::npos);
    EXPECT_NE(
        executable_rocm_fp16_publication.find("hipStreamWaitEvent("),
        std::string::npos);
    EXPECT_EQ(
        executable_rocm_fp16_publication.find("synchronize"),
        std::string::npos);
    EXPECT_NE(
        executable_rocm_fp16_publication.find("std::terminate();"),
        std::string::npos);
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoERuntimePointerArraysUseLeasedDescriptorIdentity)
{
    const auto root = repoRoot();
    const auto cuda_source = readFile(
        root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto cuda_header = readFile(
        root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.h");
    const auto rocm_source = readFile(
        root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp");
    const auto rocm_header = readFile(
        root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.h");

    for (const auto &[backend, source, header, slot_helper, execution_begin, execution_end] :
         std::array<
             std::tuple<const char *,
                        const std::string *,
                        const std::string *,
                        const char *,
                        const char *,
                        const char *>,
             2>{
             std::tuple{
                 "CUDA",
                 &cuda_source,
                 &cuda_header,
                 "bool CUDAMoEKernel::runtimePointerWorkspaceSlot(",
                 "bool CUDAMoEKernel::groupedExpertGateUpDecodeFromTable(",
                 "} // namespace llaminar2"},
             std::tuple{
                 "ROCm",
                 &rocm_source,
                 &rocm_header,
                 "bool ROCmMoEKernel::runtimePointerWorkspaceSlot(",
                 "bool ROCmMoEKernel::groupedExpertGateUpDecodeFromTable(",
                 "bool ROCmMoEKernel::groupedExpertDownDecode("}})
    {
        SCOPED_TRACE(backend);
        const auto helper_body = sliceBetween(
            *source,
            slot_helper,
            backend == std::string("CUDA")
                ? "bool CUDAMoEKernel::ensureRuntimeGateUpPointerArrays("
                : "bool ROCmMoEKernel::stageRuntimeGateUpPointerArrays(");
        const auto execution_region = sliceBetween(
            *source,
            execution_begin,
            execution_end);
        const auto executable_helper =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(helper_body));

        EXPECT_NE(
            header->find("std::size_t persistent_descriptor_slot"),
            std::string::npos)
            << backend << " pointer-array helpers must require a globally "
                          "leased descriptor slot, not a kernel-local table id";
        EXPECT_NE(
            executable_helper.find("persistent_descriptor_slot"),
            std::string::npos);
        EXPECT_EQ(executable_helper.find("table_id"), std::string::npos)
            << backend << " pointer-array address selection must not regress "
                          "to caller-local descriptor-table numbering";
        EXPECT_NE(
            execution_region.find(".workspace_slot"),
            std::string::npos)
            << backend << " grouped execution must derive pointer-array "
                          "addresses from the descriptor's retained lease";
        EXPECT_EQ(
            execution_region.find("acquirePersistentSlot("),
            std::string::npos)
            << backend << " graph execution/replay must never touch the host "
                          "lease registry or its mutex";
    }
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoEPersistentMetadataOwnershipClosesBeforeCapture)
{
    const auto root = repoRoot();
    const auto cuda_source = readFile(
        root / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto rocm_source = readFile(
        root / "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp");

    const auto cuda_rebind = sliceBetween(
        cuda_source,
        "bool CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace(",
        "void CUDAMoEKernel::resetDynamicState(");
    expectNeedleBefore(
        cuda_rebind,
        "rejectCudaPersistentMetadataMutationDuringCapture(",
        "publishGroupedDownDescriptorTable(",
        "CUDA must reject capture before changing descriptor ownership.");

    const auto cuda_down_upload = sliceBetween(
        cuda_source,
        "int CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable(",
        "int CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables(");
    expectNeedleBefore(
        cuda_down_upload,
        "rejectCudaPersistentMetadataMutationDuringCapture(",
        "publishGroupedDownDescriptorTable(",
        "CUDA down descriptors must be published before capture.");

    const auto cuda_gateup_upload = sliceBetween(
        cuda_source,
        "int CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables(",
        "bool CUDAMoEKernel::updateGroupedExpertDownDescriptorTable(");
    expectNeedleBefore(
        cuda_gateup_upload,
        "rejectCudaPersistentMetadataMutationDuringCapture(",
        "publishGroupedGateUpDescriptorTable(",
        "CUDA gate/up descriptors must be published before capture.");

    const auto rocm_rebind = sliceBetween(
        rocm_source,
        "bool ROCmMoEKernel::rebindGroupedDescriptorTablesToWorkspace(",
        "void ROCmMoEKernel::resetDynamicState(");
    expectNeedleBefore(
        rocm_rebind,
        "rejectDecodeStagingDuringCapture(",
        "publishGroupedDownDescriptorTable(",
        "ROCm must reject capture before changing descriptor ownership.");

    const auto rocm_down_upload = sliceBetween(
        rocm_source,
        "int ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable(",
        "int ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables(");
    expectNeedleBefore(
        rocm_down_upload,
        "rejectDecodeStagingDuringCapture(",
        "publishGroupedDownDescriptorTable(",
        "ROCm down descriptors must be published before capture.");

    const auto rocm_gateup_upload = sliceBetween(
        rocm_source,
        "int ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables(",
        "bool ROCmMoEKernel::updateGroupedExpertDownDescriptorTable(");
    expectNeedleBefore(
        rocm_gateup_upload,
        "rejectDecodeStagingDuringCapture(",
        "publishGroupedGateUpDescriptorTable(",
        "ROCm gate/up descriptors must be published before capture.");
}

TEST(Test__GpuWorkspaceAllocationPolicy, CUDAMoERouteScratchReuseRequiresWorkspaceBinding)
{
    const auto source = readFile(repoRoot() / "src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const auto header = readFile(repoRoot() / "src/v2/kernels/cuda/moe/CUDAMoEKernel.h");
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

    const auto grouping_capacity = sliceBetween(
        source,
        "bool CUDAMoEKernel::ensureGroupingBufferCapacity(",
        "bool CUDAMoEKernel::ensureGroupedPrefillScratchCapacity(");
    const auto executable_grouping_capacity =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(grouping_capacity));
    EXPECT_NE(header.find("bool group_buffers_workspace_bound_ = false"),
              std::string::npos)
        << "Grouped scratch needs an explicit current-workspace ownership proof.";
    EXPECT_NE(executable_grouping_capacity.find("if(group_buffers_workspace_bound_&&"),
              std::string::npos)
        << "Grouped scratch capacity alone must not authorize singleton-kernel reuse.";
    EXPECT_NE(
        executable_grouping_capacity.find(
            "d_group_int_indices_&&d_group_token_indices_&&d_group_original_to_grouped_&&d_group_original_expert_ids_&&d_group_weights_&&d_group_offsets_&&d_group_counts_&&d_group_active_expert_ids_&&d_group_write_heads_"),
        std::string::npos)
        << "Every grouped pointer touched by planning or the 0xff initialization must belong to the current workspace.";
    EXPECT_LT(executable_grouping_capacity.find("if(group_buffers_workspace_bound_&&"),
              executable_grouping_capacity.find("bindWorkspaceBuffer(&group_int_indices"))
        << "The binding-aware grouped reuse guard must precede workspace rebinding.";

    const auto clear_bindings = sliceBetween(
        source,
        "void CUDAMoEKernel::clearWorkspaceScratchBindings()",
        "bool CUDAMoEKernel::ensureStagingCapacity(");
    EXPECT_NE(clear_bindings.find("group_buffers_workspace_bound_ = false"),
              std::string::npos)
        << "Workspace replacement must revoke grouped scratch ownership.";
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
    const auto async_grouping_scratch = sliceBetween(
        source,
        "bool ROCmMoEKernel::prepareExpertGroupsAsync(",
        "bool ROCmMoEKernel::executeGroupedPrefillPipeline(");

    expectNoRawGpuAllocationCalls(route_scratch, "ROCmMoEKernel route scratch");
    expectNoRawGpuAllocationCalls(device_grouping_scratch, "ROCmMoEKernel device grouping scratch");
    expectNoRawGpuAllocationCalls(decode_scratch, "ROCmMoEKernel decode scratch");
    expectNoRawGpuAllocationCalls(async_grouping_scratch, "ROCmMoEKernel asynchronous grouping/prefill scratch");

    EXPECT_NE(route_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(device_grouping_scratch.find("bindWorkspaceBuffer"), std::string::npos);
    EXPECT_NE(decode_scratch.find("bindWorkspaceBuffer"), std::string::npos);
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

TEST(Test__GpuWorkspaceAllocationPolicy, ROCmVerifierDispatchPolicyIsWorkerLocalAndExplicit)
{
    const auto root = repoRoot();
    const auto launcher_source =
        readFile(root / "src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip");
    const auto kernel_source =
        readFile(root / "src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp");

    EXPECT_NE(
        launcher_source.find(
            "static thread_local int g_native_vnni_decode_equivalent_m1_config"),
        std::string::npos)
        << "ROCm LocalTP runs one verifier worker per GPU in one process. "
           "Decode-equivalent dispatch mode must therefore be worker-local.";
    EXPECT_EQ(
        launcher_source.find(
            "static std::atomic<int> g_native_vnni_decode_equivalent_m1_config"),
        std::string::npos)
        << "A process-global verifier mode lets one LocalTP worker clear a "
           "sibling worker's serial-M1 dispatch contract.";

    const auto grouped_policy = sliceBetween(
        kernel_source,
        "if (g_rocm_native_vnni_decode_equivalent_scope)",
        "if (PerfStatsCollector::isEnabled() && !batched_bypass_reason.empty())");
    EXPECT_FALSE(grouped_policy.empty());
    EXPECT_EQ(
        grouped_policy.find("projections.size() > 1"),
        std::string::npos)
        << "Single-projection grouped verifier stages such as LM_HEAD need the "
           "same explicit serial-M1 policy as fused projection groups.";
    EXPECT_NE(
        grouped_policy.find("rocmGemv_native_vnni_query_serial_m1_config"),
        std::string::npos);
    EXPECT_NE(grouped_policy.find("policy.kb"), std::string::npos);
    EXPECT_NE(grouped_policy.find("policy.target_waves"), std::string::npos)
        << "The resolved serial reduction geometry must be passed into the HIP "
           "launcher, not inferred from mutable process state.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, MTPVerifierGraphWaitsOnSidecarStreamWithoutHostFlush)
{
    const auto source = readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header = readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto runner_source =
        readFile(repoRoot() / "src/v2/execution/runner/OrchestrationRunner.cpp");
    const auto preparation_stage_source = readFile(
        repoRoot() /
        "src/v2/execution/compute_stages/stages/MTPVerifierPreparationStage.cpp");
    const auto sidecar_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::executeMTPDepth0Batched(",
        "bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(");
    const auto flush_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::flushPendingMTPWork()",
        "const char *DeviceGraphOrchestrator::pendingLogitsStreamRoleName(");
    const auto decode_body = sliceBetween(
        runner_source,
        "GenerationResult OrchestrationRunner::decodeStepMTP()",
        "GenerationResult OrchestrationRunner::decodeStep()");
    const auto wait_helper = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::waitForPendingLogitsStream(",
        "void *DeviceGraphOrchestrator::peekPendingLogitsStream(");
    const auto verifier_metadata = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(",
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(");
    const auto executable_wait_helper = stripCommentsAndStringLiterals(wait_helper);
    const auto compact_wait_helper =
        removeAsciiWhitespace(executable_wait_helper);
    const auto executable_verifier_metadata =
        stripCommentsAndStringLiterals(verifier_metadata);
    const auto compact_verifier_metadata =
        removeAsciiWhitespace(executable_verifier_metadata);
    const auto preparation_stage_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            preparation_stage_source,
            "bool MTPVerifierPreparationStage::execute(IDeviceContext *ctx)",
            "size_t MTPVerifierPreparationStage::estimatedFlops() const")));
    const auto executable_sidecar =
        stripCommentsAndStringLiterals(sidecar_body);
    const auto executable_flush =
        stripCommentsAndStringLiterals(flush_body);
    const auto compact_decode =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(decode_body));

    EXPECT_NE(header.find("waitForPendingLogitsStream("), std::string::npos)
        << "The sidecar-to-verifier ordering edge should stay behind a named helper.";
    EXPECT_NE(compact_wait_helper.find(
                  "insertStreamDependency(consumer_stream,producer_stream)"),
              std::string::npos)
        << "Verifier ordering must be a GPU-side event dependency, not a host sync.";
    EXPECT_EQ(executable_wait_helper.find("synchronizeStream"), std::string::npos)
        << "The pending-logits stream wait helper must not block the CPU.";
    EXPECT_EQ(executable_sidecar.find("synchronizeStream"), std::string::npos)
        << "A GPU sidecar must publish a logits/KV event even on first use.";
    EXPECT_EQ(executable_flush.find("synchronizeStream"), std::string::npos);
    EXPECT_EQ(executable_flush.find(".synchronize("), std::string::npos)
        << "The obsolete flush hook must never drain a GPU stream.";
    const auto compact_flush =
        removeAsciiWhitespace(executable_flush);
    EXPECT_NE(compact_flush.find("if(state_.device_id.is_gpu()){LOG_ERROR("),
              std::string::npos)
        << "Calling the old flush contract on GPU must report the invalid "
           "ordering contract.";
    EXPECT_NE(compact_flush.find("LOG_ERROR();returnfalse;"),
              std::string::npos)
        << "Calling the old flush contract on GPU must fail closed.";
    EXPECT_NE(compact_decode.find(
                  "if(runner_->primaryDeviceId().is_gpu()&&!runner_->supportsMTPSidecarLogitsStreamHandoff())"),
              std::string::npos)
        << "GPU MTP must require the event-owned sidecar contract up front.";
    EXPECT_NE(decode_body.find(
                  "\"sidecar_iteration_host_flushes_avoided\""),
              std::string::npos);
    EXPECT_EQ(decode_body.find(
                  "LLAMINAR_MTP_FORCE_SIDECAR_FLUSH_BEFORE_VERIFIER"),
              std::string::npos)
        << "Diagnostics may not revive a production GPU stream drain.";
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
    EXPECT_NE(preparation_stage_execute.find(
                  "deviceCopyAsync(params_.base_cached_tokens_snapshot_device,"),
              std::string::npos)
        << "The pre-verifier base cache count must be staged device-to-device, "
           "inside the typed captured preparation graph.";
    EXPECT_NE(compact_verifier_metadata.find("mtp_publication_base_cache_snapshot_ready_=true"),
              std::string::npos)
        << "Publication should only trust BASE_CACHED_TOKENS after the verifier "
           "prep step marks the device snapshot ready.";
    EXPECT_EQ(compact_verifier_metadata.find("hostToDeviceOnStream(ptrs.base_cached_tokens"),
              std::string::npos)
        << "Verifier metadata preparation must not upload host base cache counts.";
    EXPECT_EQ(compact_verifier_metadata.find(
                  "deviceCopyAsync(ptrs.base_cached_tokens"),
              std::string::npos)
        << "The orchestrator must not launch a loose base-snapshot operation.";
    EXPECT_NE(preparation_stage_execute.find(
                  "captureDeviceSequenceStateCheckpoint("),
              std::string::npos);
    EXPECT_NE(preparation_stage_execute.find(
                  "enqueuePrepareMTPVerifierGeometry("),
              std::string::npos);
    EXPECT_NE(compact_verifier_metadata.find(
                  "materializeMTPVerifierPreparationGraph("),
              std::string::npos);
    EXPECT_NE(compact_verifier_metadata.find(
                  "executeMTPVerifierPreparationCaptured("),
              std::string::npos);
    expectNeedleBefore(
        executable_verifier_metadata,
        "waitForPendingShiftedMTPKVReady",
        "uploadMTPSpecDecodeVerifierInputPlan",
        "Verifier graph metadata staging must be ordered after shifted MTP KV catch-up.");
    expectNeedleBefore(
        compact_verifier_metadata,
        "uploadMTPSpecDecodeVerifierInputPlan",
        "materializeMTPVerifierPreparationGraph",
        "The captured prelude should bind the verifier metadata workspace only "
        "after the row plan has established its resident identity.");
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

TEST(Test__GpuWorkspaceAllocationPolicy, GPUFlashAttentionDecodeStateIsWorkspaceAndDeviceOwned)
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
        << "CUDA attention params must not allocate pinned host staging.";
    EXPECT_EQ(attention_wrapper_source.find("cudaFreeHost("), std::string::npos)
        << "CUDA attention params must not own pinned host allocations.";
    EXPECT_EQ(attention_wrapper_source.find("cudaMemcpyHostToDevice"), std::string::npos)
        << "CUDA attention parameter publication must be a device kernel, not H2D.";
    EXPECT_EQ(kernel_header.find("std::array<attention::AttentionDeviceParams"),
              std::string::npos)
        << "CUDA attention must not retain a host parameter mirror.";
    EXPECT_NE(kernel_source.find("writeDynamicAttnParams("), std::string::npos)
        << "Explicit CUDA geometry must enter DEVICE_PARAMS through the device writer.";
    EXPECT_NE(kernel_source.find("getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS)"), std::string::npos)
        << "The CUDA graph-visible attention params must remain workspace-backed.";

    const auto cuda_kernel_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/attention/CUDAFlashAttentionKernels.cu");
    expectNoRawGpuAllocationCalls(cuda_kernel_source, "CUDAFlashAttention kernel wrappers");
    EXPECT_NE(cuda_kernel_source.find("cuda_write_attention_params_from_geometry_kernel"),
              std::string::npos)
        << "CUDA explicit geometry must be graph-capturably device-written.";
    EXPECT_EQ(cuda_kernel_source.find("cudaFlashAttn_allocWorkspace"), std::string::npos);
    EXPECT_EQ(cuda_kernel_source.find("cudaFlashAttn_freeWorkspace"), std::string::npos);
    EXPECT_EQ(cuda_kernel_source.find("cudaFlashAttn_prefill_cublas_fp16kv"), std::string::npos);

    const auto rocm_header =
        readFile(repoRoot() / "src/v2/kernels/rocm/attention/ROCmFlashAttentionKernelT.h");
    const auto rocm_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/attention/ROCmFlashAttentionKernelT.cpp");
    const auto rocm_kernel_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip");
    const auto rocm_wrapper_source = rocm_header + rocm_source;
    EXPECT_EQ(rocm_wrapper_source.find("h_attn_params_"), std::string::npos)
        << "ROCm attention must not retain a host parameter mirror.";
    EXPECT_EQ(rocm_wrapper_source.find("hipHostMalloc"), std::string::npos)
        << "ROCm attention params must not allocate pinned host staging.";
    EXPECT_EQ(rocm_wrapper_source.find("hipMemcpyHostToDevice"), std::string::npos)
        << "ROCm attention parameter publication must be a device kernel, not H2D.";
    EXPECT_NE(rocm_source.find("writeDynamicAttnParams("), std::string::npos)
        << "Explicit ROCm geometry must enter DEVICE_PARAMS through the device writer.";
    EXPECT_NE(rocm_kernel_source.find("hip_write_attention_params_from_geometry_kernel"),
              std::string::npos)
        << "ROCm explicit geometry must be graph-capturably device-written.";

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
        "src/v2/tensors/TensorBase.cpp",
        "src/v2/tensors/cpu/CPUTensors.h",
        "src/v2/tensors/ITensor.h",
        "src/v2/tensors/TensorClasses.h",
        "src/v2/tensors/TensorSlice.h",
        "src/v2/transfer/TransferEngine.cpp",
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
               "The legacy tensor methods are restricted to tensor and transfer "
               "internals; production callers must not extend this allowlist.\n";
        for (const auto &failure : failures)
            out << failure << '\n';
        return out.str();
    }();
}

TEST(Test__GpuWorkspaceAllocationPolicy, ComputeStageEnsureOnDeviceDebtStaysExplicit)
{
    const auto root = repoRoot();
    const std::unordered_set<std::string> sanctioned = {};

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
    const auto direct_publication = sliceBetween(
        execute_gpu,
        "        if (!graph_managed)\n        {\n            /*\n             * A direct-output row selector",
        "        return true;");
    EXPECT_NE(direct_publication.find("if (isGraphCaptureActive())"), std::string::npos);
    EXPECT_NE(
        direct_publication.find("TransferEngine::publishGraphOwnedDeviceWrite("),
        std::string::npos)
        << "Captured direct row-select execution must publish device authority without a standalone event.";
    EXPECT_NE(
        direct_publication.find("gpuExecution().publish(output_base);"),
        std::string::npos)
        << "Only eager direct row-select execution should publish a completion event "
           "through its exact stage device/stream capability.";
    EXPECT_EQ(direct_publication.find("output_base->transitionTo"), std::string::npos)
        << "Compute stages must not mutate tensor coherence outside TransferEngine.";
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

TEST(Test__GpuWorkspaceAllocationPolicy, GPUKVConversionScratchCoversResidentHistoryBeyondGraphBucket)
{
    const auto cuda_source =
        readFile(repoRoot() / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto rocm_source =
        readFile(repoRoot() / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp");

    const auto cuda_requirements = sliceBetween(
        cuda_source,
        "WorkspaceRequirements CUDARingKVCache<Precision>::getWorkspaceRequirements(",
        "    template <ActivationPrecision Precision>\n    void CUDARingKVCache<Precision>::bindWorkspace");
    const auto rocm_requirements = sliceBetween(
        rocm_source,
        "WorkspaceRequirements ROCmRingKVCache<Precision>::getWorkspaceRequirements(",
        "    template <ActivationPrecision Precision>\n    void ROCmRingKVCache<Precision>::bindWorkspace");

    for (const auto *requirements : {&cuda_requirements, &rocm_requirements})
    {
        EXPECT_NE(requirements->find("std::max(m, max_seq_len_)"), std::string::npos)
            << "A prefill graph bucket bounds new rows, not the resident KV horizon";
        EXPECT_NE(requirements->find("batch_size_"), std::string::npos)
            << "Scratch must retain the cache's full configured request capacity";
        EXPECT_NE(requirements->find("KVCacheWorkspaceBuffers::CONV_SCRATCH_K"), std::string::npos);
        EXPECT_NE(requirements->find("KVCacheWorkspaceBuffers::CONV_SCRATCH_V"), std::string::npos);
    }
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
    EXPECT_NE(compact_qwen_base.find("ForwardInputqwen_input=input;"),
              std::string::npos)
        << "The Qwen adapter must preserve the complete ForwardInput policy, "
           "including resident position rows and contiguous-position semantics.";

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

TEST(Test__GpuWorkspaceAllocationPolicy, PrefillCapturePrejoinsArenaInputsBeforeBeginCapture)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp");
    const auto capture_path = sliceBetween(
        source,
        "if (can_attempt_capture && capture_ready_reason == PrefillGraphRejectReason::None)",
        "const std::string capture_launch_boundary");

    expectNeedleBefore(
        capture_path,
        "executor_.prepareInputsForGraphCapture(",
        "host.waitAtPrefillGraphCaptureBoundary(",
        "Every external producer event must be joined before the multi-device "
        "capture-begin rendezvous.");
    expectNeedleBefore(
        capture_path,
        "executor_.prepareInputsForGraphCapture(",
        "cache.captureAndInstantiate(",
        "No arena input may discover an uncaptured producer event after "
        "beginCapture().");
}

TEST(Test__GpuWorkspaceAllocationPolicy, EveryNativeCapturePathPrejoinsInputsBeforeBeginCapture)
{
    const auto controller_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp");
    const auto executor_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp");
    const auto phase_two_capture = sliceBetween(
        controller_source,
        "DeviceGraphCaptureController::CapturePhaseResult DeviceGraphCaptureController::executeCapturePhase(",
        "DeviceGraphCaptureController::ReplayPhaseResult DeviceGraphCaptureController::executeReplayPhase(");
    const auto diagnostic_recapture = sliceBetween(
        controller_source,
        "bool DeviceGraphCaptureController::executeCapturedReplaySegmentRecapture(",
        "DeviceGraphCaptureController::VerifyReplayResult DeviceGraphCaptureController::executeCapturedReplaySegmentVerify(");
    const auto direct_capture = sliceBetween(
        executor_source,
        "bool DeviceGraphExecutor::executeWithGraphCapture(",
        "bool DeviceGraphExecutor::executeWithCachedGraphReplay(");

    expectNeedleBefore(
        phase_two_capture,
        "if (!hooks.cohere_inputs(seg))",
        "ScopedBackendGraphCapture capture_transaction(",
        "Phase-2 decode/sidecar capture must join every segment input before "
        "the backend capture transaction begins.");
    expectNeedleBefore(
        phase_two_capture,
        "if (!hooks.cohere_inputs(seg))",
        "hooks.capture_boundary(boundary_name, capture_stream)",
        "Input event joins must precede the LocalTP capture rendezvous so all "
        "participants enter capture with complete dependencies.");
    expectNeedleBefore(
        diagnostic_recapture,
        "if (!cohere_inputs_cb(segment))",
        "ScopedBackendGraphCapture capture_transaction(",
        "Diagnostic recapture is not allowed to bypass exact-stream input "
        "preparation.");
    expectNeedleBefore(
        direct_capture,
        "prepareInputsForGraphCapture(",
        "ScopedBackendGraphCapture capture_transaction(",
        "The direct single-graph API must enforce the same pre-capture event "
        "contract as cached decode capture.");
}

TEST(Test__GpuWorkspaceAllocationPolicy, MoERuntimeDecodeUsesFusedWorkspaceBeforeLegacyScratch)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp");
    const auto single_token = sliceBetween(
        source,
        "bool MoEExpertComputeStage::executeSingleToken(IDeviceContext *ctx)",
        "bool MoEExpertComputeStage::executeCPUGroupedDecodeEquivalentVerifierPrefill(");

    expectNeedleBefore(
        single_token,
        "kernel->groupedExpertDecodeFromRuntime(",
        "scratch_gate_batch_.resize(top_k)",
        "The workspace-native fused runtime route must return before any legacy "
        "per-slot scratch construction.");
    expectNeedleBefore(
        single_token,
        "kernel->groupedExpertDecodeFromRouting(",
        "scratch_gate_batch_.resize(top_k)",
        "The workspace-native fused explicit route must return before CPU-only "
        "per-slot scratch construction.");

    const auto runtime_route = sliceBetween(
        single_token,
        "if (can_try_device_routed_decode)",
        "const bool require_device_routing_tensor_decode =");
    EXPECT_EQ(
        runtime_route.find("groupedExpertGateUpDecodeFromRuntime("),
        std::string::npos)
        << "Production runtime decode must not retain a two-step scratch fallback.";
    EXPECT_EQ(
        runtime_route.find("using host-routed fallback"),
        std::string::npos)
        << "A failed fused runtime route is fatal, not permission to change execution modes.";

    const auto explicit_route = sliceBetween(
        single_token,
        "if (can_try_device_routing_tensor_decode)",
        "if (is_gpu)");
    EXPECT_EQ(
        explicit_route.find("groupedExpertGateUpDecodeFromRouting("),
        std::string::npos)
        << "Production explicit routing must not retain a two-step gate/up path.";
    EXPECT_EQ(
        explicit_route.find("groupedExpertDownDecodeFromRouting("),
        std::string::npos)
        << "Production explicit routing must not retain a two-step down path.";
    EXPECT_NE(
        explicit_route.find("Mandatory fused explicit-routing"),
        std::string::npos)
        << "A failed fused explicit route must be fatal.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, ResidualAddCannotSplitHomogeneousGpuGraph)
{
    const auto residual_source = readFile(repoRoot() / "src/v2/execution/compute_stages/stages/ResidualAddStage.h");
    EXPECT_EQ(residual_source.find("graph_capture_boundary_before"), std::string::npos)
        << "ResidualAddStage is ordinary capturable compute and must not expose "
           "a switch that can split a homogeneous GPU graph.";
    EXPECT_EQ(residual_source.find("requiresGraphCaptureSegmentBoundaryBefore()"), std::string::npos);

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

TEST(Test__GpuWorkspaceAllocationPolicy, Qwen35MoEMaintenanceGraphSharesDomainCollectiveLane)
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
        "moe_graph_rebalance_bindings_[domain_key] = GraphSideRebalanceBinding{",
        "graph_rebalance_plan_inserted = true;");

    EXPECT_NE(binding_section.find("collective_tp_ctx"), std::string::npos);
    EXPECT_EQ(binding_section.find("decode_tp_ctx"), std::string::npos);
    EXPECT_EQ(binding_section.find("maintenance_tp_ctx"), std::string::npos);
    EXPECT_EQ(graph_source.find("maintenanceTPContextForDomain("), std::string::npos)
        << "Maintenance must not create a second LocalTP communicator.";
    EXPECT_EQ(graph_source.find("std::weak_ptr<ILocalTPContext>"), std::string::npos)
        << "The obsolete process-wide maintenance communicator registry must stay deleted.";
    EXPECT_EQ(graph_source.find("createLocalTPContext("), std::string::npos)
        << "A declarative model graph must consume its supplied collective context, not construct another one.";
    EXPECT_NE(
        insertion_section.find(".collective_tp_ctx = local_tp_ctx"),
        std::string::npos)
        << "Decode maintenance must declare the enclosing graph-family communicator.";
    EXPECT_NE(
        maintenance_build.find(
            "params.tp_ctx = binding.collective_tp_ctx"),
        std::string::npos)
        << "The maintenance graph must lower the declared shared-domain context into its stage.";
    EXPECT_NE(
        maintenance_build.find(
            "params.phase = DeviceMoERebalanceStagePhase::PlanCopyApply"),
        std::string::npos)
        << "Maintenance must collect, plan, transfer, and apply in one captured transaction.";
    EXPECT_EQ(
        countOccurrences(maintenance_build, "graph.addNode("),
        1u)
        << "A host-selected probe/payload split would reintroduce an execution gap.";
    EXPECT_EQ(
        maintenance_build.find("PlanAndPreparePayloadAfterSideband"),
        std::string::npos);
    EXPECT_EQ(
        maintenance_build.find("TransferPreparedPayload"),
        std::string::npos);
    EXPECT_EQ(
        maintenance_build.find("payload_edge_mask"),
        std::string::npos)
        << "The graph topology must not depend on a D2H-planned edge mask.";
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
    EXPECT_NE(compact.find("total_tokens>=1&&"),
              std::string::npos)
        << "CPU verifier publication must accept every positive runtime row count.";
    EXPECT_EQ(compact.find("total_tokens<=4"), std::string::npos)
        << "Verifier admission must not encode the retired four-row MTP limit.";

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
    EXPECT_NE(compact.find("resolveMTPMaxTargetQueryRows(config_.mtp)"),
              std::string::npos)
        << "Hybrid/GDN verifier state snapshots must use the resolver's "
           "already-flattened request capacity.";
    EXPECT_EQ(compact.find("resolveMTPMaxTargetQueryRows(config_.mtp)*std::max(1,batch_size)"),
              std::string::npos)
        << "The MTP capacity resolver already includes max_request_batch; "
           "multiplying it again retains batch-squared snapshot storage.";
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
    EXPECT_NE(gdn_header.find("verifier workspace binding while a Ready"),
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

TEST(Test__GpuWorkspaceAllocationPolicy, LiveCheckpointStorageUsesReusableDomainCorrectPool)
{
    const auto header_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    EXPECT_NE(header_source.find("LiveCheckpointStorageSlot"), std::string::npos);
    EXPECT_NE(header_source.find("live_checkpoint_storage_pool_"), std::string::npos);
    EXPECT_NE(header_source.find("terminal_hidden_host_storage"), std::string::npos);
    EXPECT_NE(header_source.find("terminal_hidden_device_storage"), std::string::npos);
    EXPECT_NE(header_source.find("std::shared_ptr<void> ready_event"),
              std::string::npos)
        << "Every preallocated checkpoint slot must own a reusable event.";
    EXPECT_EQ(header_source.find(
                  "std::shared_ptr<TensorBase> terminal_hidden_device_storage"),
              std::string::npos)
        << "Transient GPU checkpoint slots must not carry TensorBase host mirrors.";

    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto ensure_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::ensureLiveCheckpointStorage(",
        "bool DeviceGraphOrchestrator::acquireLiveCheckpointStorage(");
    EXPECT_NE(ensure_body.find("return acquireLiveCheckpointStorage(handle);"), std::string::npos)
        << "The hot live-checkpoint path must acquire one reusable domain-correct slot.";
    EXPECT_EQ(ensure_body.find("allocateDeviceByteStorage("), std::string::npos)
        << "Per-step checkpoint device allocation regresses CUDA MoE MTP decode.";

    const auto acquire_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::acquireLiveCheckpointStorage(",
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(");
    const auto initialize_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::initializeLiveCheckpointStoragePool()",
        "bool DeviceGraphOrchestrator::acquireLiveCheckpointStorage(");
    const auto arena_initialization_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::initializeInferenceStateFromArena(",
        "bool DeviceGraphOrchestrator::initializeKVCaches(");
    EXPECT_NE(acquire_body.find("hybrid_host_storage.use_count() == 1"), std::string::npos)
        << "Pool slots must not be reused while a PrefixStateSnapshot still owns host payload storage.";
    EXPECT_NE(acquire_body.find("hybrid_device_storage.use_count() == 1"), std::string::npos)
        << "Pool slots must not be reused while a PrefixStateSnapshot still owns device payload storage.";
    EXPECT_NE(acquire_body.find("terminal_hidden_device_storage.use_count() == 1"),
              std::string::npos)
        << "A snapshot-owned terminal row must exclude its storage slot from reuse.";
    EXPECT_NE(acquire_body.find("ready_event.use_count() == 1"),
              std::string::npos)
        << "A retained completion proof must exclude its storage slot from reuse.";
    EXPECT_NE(acquire_body.find("allocatePureDeviceStorage("), std::string::npos);
    EXPECT_EQ(acquire_body.find("allocateDeviceByteStorage("), std::string::npos);
    EXPECT_NE(acquire_body.find("live_prefix_checkpoint_storage_pool_hits"), std::string::npos);
    EXPECT_NE(acquire_body.find("live_prefix_checkpoint_storage_pool_misses"), std::string::npos);
    EXPECT_NE(initialize_body.find("kMTPConcurrentLiveCheckpointSets"),
              std::string::npos)
        << "Runtime preallocation must consume the shared checkpoint lifetime policy.";
    EXPECT_NE(initialize_body.find("payload_layouts_per_checkpoint"),
              std::string::npos)
        << "Each retained checkpoint must reserve only payload layouts used by capture.";
    EXPECT_NE(initialize_body.find("shifted_layout.includes_hybrid_state"),
              std::string::npos)
        << "Full-attention shifted caches must not consume recurrent payload slots.";
    EXPECT_EQ(initialize_body.find("checkpoint_handles_per_snapshot"),
              std::string::npos)
        << "The pool must not count every shifted cache as a large payload owner.";
    EXPECT_NE(initialize_body.find("reservations.push_back(std::move(handle))"),
              std::string::npos)
        << "Prewarming must retain each reservation so acquisition creates distinct slots.";
    EXPECT_NE(initialize_body.find(
                  "live_checkpoint_storage_pool_initialized_ = true"),
              std::string::npos);
    EXPECT_NE(acquire_body.find(
                  "live_checkpoint_storage_pool_initialized_"),
              std::string::npos)
        << "Runtime checkpoint pool exhaustion must fail instead of allocating.";
    EXPECT_NE(acquire_body.find("allocatePersistentDeviceEvent("),
              std::string::npos)
        << "Reusable checkpoint events must be created only while the pool is initialized.";
    EXPECT_NE(arena_initialization_body.find(
                  "initializeLiveCheckpointStoragePool()"),
              std::string::npos)
        << "Checkpoint storage must be allocated before the first inference request.";

    const auto checkpoint_body = sliceBetween(
        source,
        "PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(",
        "bool DeviceGraphOrchestrator::restoreLivePrefixState(");
    EXPECT_EQ(checkpoint_body.find("captureLivePrefixState("), std::string::npos)
        << "The production transaction checkpoint must never switch to the host payload path.";
    EXPECT_EQ(checkpoint_body.find("live_prefix_checkpoint_payload_required"),
              std::string::npos)
        << "Headroom is an admission invariant, not a reason to change checkpoint architecture.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, QuantizedEmbeddingHasOnePlannedGPURepresentation)
{
    const auto weight_manager_source =
        readFile(repoRoot() / "src/v2/loaders/WeightManager.cpp");
    const auto frozen_non_gemm_body = sliceBetween(
        weight_manager_source,
        "bool WeightManager::uploadFrozenNonGemmWeights(",
        "bool WeightManager::uploadNonGemmWeights(");
    const auto embedding_branch =
        frozen_non_gemm_body.find(
            "binding.identity.role == WeightRole::Embedding");
    const auto prepared_call =
        frozen_non_gemm_body.find("store->prepareEmbedding(");
    const auto raw_upload =
        frozen_non_gemm_body.find(
            "TransferEngine::instance().upload(");

    ASSERT_NE(embedding_branch, std::string::npos);
    ASSERT_NE(prepared_call, std::string::npos);
    ASSERT_NE(raw_upload, std::string::npos);
    EXPECT_LT(embedding_branch, prepared_call);
    EXPECT_LT(prepared_call, raw_upload)
        << "Quantized embeddings must exit through preparation before the generic raw upload.";
    EXPECT_NE(
        frozen_non_gemm_body.find(
            "quantized embedding prepared without raw device upload"),
        std::string::npos);

    const auto orchestrator_source =
        readFile(
            repoRoot() /
            "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto initialization_body = sliceBetween(
        orchestrator_source,
        "void DeviceGraphOrchestrator::initializePreparedWeightStore(",
        "bool DeviceGraphOrchestrator::prepareMTPMoEExpertSlabs(");
    EXPECT_EQ(
        initialization_body.find(
            "prepared_weight_store_->prepareEmbedding("),
        std::string::npos)
        << "Graph construction must not allocate planner-invisible embedding storage.";
    EXPECT_NE(
        initialization_body.find(
            "without a prepared device handle"),
        std::string::npos)
        << "A missing prepared GPU embedding is a fatal lifecycle violation.";
}

/**
 * @brief Guards first-request ownership of persistent MTP sampling history.
 *
 * A fresh GPU runner may enter prefill and verifier execution without a
 * request-boundary reset.  The generated-token histogram therefore has to be
 * initialized and event-published during runner construction; merely allocating
 * its arena address leaves host-authoritative zeroes and makes graph input
 * admission fail on the first request.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     PersistentMTPGeneratedTokenHistoryIsPublishedBeforeFirstRequest)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto initialization_body = removeAsciiWhitespace(sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::initializeInferenceStateFromArena(",
        "bool DeviceGraphOrchestrator::initializeKVCaches("));
    const auto publication_body = removeAsciiWhitespace(sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::zeroAndPublishMTPGeneratedTokenHistoryOnStream(",
        "bool DeviceGraphOrchestrator::materializePendingMTPVerifierInputTokensOnDevice("));
    const auto reset_body = removeAsciiWhitespace(sliceBetween(
        header,
        "void resetInferenceState(",
        "void clear_cache() override"));

    EXPECT_NE(
        initialization_body.find(
            "explicitGPUStreamForOperation(\"initialize_mtp_generated_token_history\")"),
        std::string::npos)
        << "Fresh GPU runners need an explicit initialization producer stream.";
    EXPECT_NE(
        initialization_body.find(
            "zeroAndPublishMTPGeneratedTokenHistoryOnStream("),
        std::string::npos)
        << "Construction must publish initialized histogram bytes before the first request.";
    EXPECT_NE(publication_body.find("backend->memset("), std::string::npos)
        << "The histogram must be initialized directly on its owning GPU.";
    EXPECT_NE(
        publication_body.find("publishPreparedArenaGraphInput("),
        std::string::npos)
        << "The zero-fill must publish its exact producer event to BufferArena.";
    EXPECT_EQ(publication_body.find("copyToDevice("), std::string::npos)
        << "Persistent MTP history initialization must not round-trip through a host mirror.";
    EXPECT_EQ(publication_body.find("synchronize"), std::string::npos)
        << "Initialization ordering is an event edge, never a stream or device synchronization.";
    EXPECT_NE(
        reset_body.find(
            "zeroAndPublishMTPGeneratedTokenHistoryOnStream(reset_transaction.executionStream(),reset_reason)"),
        std::string::npos)
        << "Request reset and construction must use the same ownership boundary.";
}

TEST(Test__GpuWorkspaceAllocationPolicy, LiveCheckpointPublicationFailureIsFatal)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordLivePrefixCheckpointReady(",
        "bool DeviceGraphOrchestrator::waitForSnapshotReady(");
    const auto compact_record =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));

    EXPECT_NE(compact_record.find("std::terminate();"), std::string::npos)
        << "Post-submit checkpoint publication failure cannot release in-flight storage.";
    EXPECT_EQ(compact_record.find("synchronizeStream("), std::string::npos)
        << "Live checkpoint publication must never switch to a stream-sync fallback.";
}

/**
 * @brief Prevent live KV eviction from rebuilding a host metadata mirror.
 *
 * Ordinary and TurboQuant caches share the same canonical device metadata
 * helper. Every production eviction wrapper must call that helper directly and
 * must not observe head/count values or wait for the stream on the host.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     GpuKVEvictionIsDeviceOwnedForOrdinaryAndTurboQuantCaches)
{
    const auto root = repoRoot();
    const auto cuda_cache = readFile(
        root / "src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu");
    const auto cuda_tq = readFile(
        root / "src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu");
    const auto rocm_cache = readFile(
        root / "src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp");
    const auto rocm_tq = readFile(
        root / "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.hip");

    const std::array<std::pair<std::string, std::string>, 4> methods = {
        std::pair{
            sliceBetween(
                cuda_cache,
                "void CUDARingKVCache<Precision>::evict_oldest(",
                "void CUDARingKVCache<Precision>::evict_oldest_layer("),
            "CUDA ordinary"},
        std::pair{
            sliceBetween(
                cuda_tq,
                "void CUDARingKVCacheTQ::evict_oldest(",
                "} // namespace llaminar2"),
            "CUDA TurboQuant"},
        std::pair{
            sliceBetween(
                rocm_cache,
                "void ROCmRingKVCache<Precision>::evict_oldest(",
                "void ROCmRingKVCache<Precision>::evict_oldest_layer("),
            "ROCm ordinary"},
        std::pair{
            sliceBetween(
                rocm_tq,
                "void ROCmRingKVCacheTQ::evict_oldest(",
                "// =========================================================================\n"
                "    // Factory function"),
            "ROCm TurboQuant"},
    };

    for (const auto &[method, label] : methods)
    {
        const auto executable =
            removeAsciiWhitespace(stripCommentsAndStringLiterals(method));
        EXPECT_NE(
            executable.find("evictOldestDeviceSequenceState("),
            std::string::npos)
            << label << " eviction must launch the canonical device mutation.";
        EXPECT_EQ(
            executable.find("observeDeviceSequenceState("),
            std::string::npos)
            << label << " eviction must never download live ring metadata.";
        EXPECT_EQ(
            executable.find("StreamSynchronize("),
            std::string::npos)
            << label << " eviction must never block the host.";
        EXPECT_EQ(
            executable.find("synchronizeStream("),
            std::string::npos)
            << label << " eviction must never hide a backend stream wait.";
    }
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

/**
 * @brief Cached GPU graph snapshots must be owned by the captured graph class.
 *
 * Stage names such as `embedding` repeat across serial, grouped-verifier, and
 * prefill graph geometries. A DeviceGraphExecutor-wide map keyed by stage name
 * therefore lets one graph poison another graph's immutable capture descriptor.
 * Keep both decode segment caches and monolithic prefill caches as explicit
 * manifest owners, and forbid the retired executor-wide maps from returning.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, CachedGraphSnapshotsHavePerGraphOwners)
{
    const auto root = repoRoot();
    const std::string executor_header =
        readFile(root /
                 "src/v2/execution/local_execution/graph/DeviceGraphExecutor.h");
    const std::string forward_types =
        readFile(root /
                 "src/v2/execution/local_execution/engine/ForwardGraphTypes.h");
    const std::string capture_source =
        readFile(root /
                 "src/v2/execution/local_execution/graph/"
                 "DeviceGraphExecutor_GraphCapture.cpp");

    EXPECT_NE(
        executor_header.find(
            "GraphSnapshotManifest snapshot_manifest;"),
        std::string::npos)
        << "GraphSegmentCache must intrinsically own its decode snapshot manifest.";
    EXPECT_NE(
        forward_types.find(
            "DeviceGraphExecutor::GraphSnapshotManifest snapshot_manifest;"),
        std::string::npos)
        << "ForwardGraphCache must own the manifest used by monolithic prefill.";

    EXPECT_EQ(
        executor_header.find("graph_snapshot_copies_"),
        std::string::npos)
        << "The process-wide stage-name snapshot map must remain retired.";
    EXPECT_EQ(
        executor_header.find("graph_snapshot_outputless_stages_"),
        std::string::npos)
        << "Outputless stage metadata belongs to the same graph-owned manifest.";

    const std::string cached_replay = sliceBetween(
        capture_source,
        "bool DeviceGraphExecutor::executeWithCachedGraphReplay(",
        "} // namespace llaminar2");
    const std::string compact_cached_replay =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(cached_replay));
    EXPECT_NE(
        cached_replay.find("segment_cache.snapshot_manifest"),
        std::string::npos)
        << "Cached capture and replay must resolve the manifest from their exact segment cache.";
    EXPECT_NE(
        compact_cached_replay.find(
            "runStages(graph,ctx,warmup_policy,collective_nodes,"
            "&segment_cache.snapshot_manifest)"),
        std::string::npos)
        << "Phase-1 warmup must finalize descriptors in the same manifest that Phase-2 capture consumes.";
    EXPECT_EQ(
        cached_replay.find("transient_snapshot_manifest_"),
        std::string::npos)
        << "Cached graph execution must never fall through to the uncached eager manifest.";
}

/**
 * @brief Locks prefix archive DMA behind a preflighted event ownership edge.
 *
 * A prefix block owns pinned RAM and may also own a device-hot allocation.
 * Event creation after the first asynchronous copy leaves no safe recovery
 * path if allocation fails. This source-level policy test makes the required
 * order explicit and forbids reintroducing exceptional stream synchronization.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, PrefixArchiveReadinessIsPreflightedAndNeverSyncsOnFailure)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto prepare_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::preparePrefixArchivePayloadReady(",
        "bool DeviceGraphOrchestrator::recordPrefixArchivePayloadReady(");
    const auto record_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::recordPrefixArchivePayloadReady(",
        "bool DeviceGraphOrchestrator::promotePrefixBlockToDeviceHotForRestore(");
    const auto promotion_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::promotePrefixBlockToDeviceHotForRestore(",
        "bool DeviceGraphOrchestrator::ensurePrefixCacheReady()");
    const auto harvest_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::harvestPrefix(",
        "bool DeviceGraphOrchestrator::ensureLiveCheckpointStorage(");

    const auto compact_prepare =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(prepare_body));
    const auto compact_record =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(record_body));
    const auto compact_promotion =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(promotion_body));
    const auto compact_harvest =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(harvest_body));

    EXPECT_NE(compact_prepare.find("backend->createEvent("), std::string::npos);
    EXPECT_NE(compact_prepare.find("payload_readiness->prepare("), std::string::npos);
    expectNeedleBefore(
        compact_record,
        "backend->recordEvent(",
        "payload_readiness->publishRecorded()",
        "The completion event must be recorded before it becomes consumable.");
    EXPECT_NE(compact_record.find("std::terminate();"), std::string::npos)
        << "A post-submit event failure cannot safely return to ordinary cleanup.";
    EXPECT_EQ(compact_record.find("synchronizeStream("), std::string::npos)
        << "Event publication failure must not switch to a stream-sync fallback.";

    expectNeedleBefore(
        compact_promotion,
        "preparePrefixArchivePayloadReady(",
        "backend->hostToDeviceOnStream(",
        "Device-hot promotion must allocate its event before the first H2D.");
    expectNeedleBefore(
        compact_harvest,
        "preparePrefixArchivePayloadReady(&handle,stream,",
        "prefix_backend->deviceToHostOnStream(",
        "RAM archive event preparation must precede every D2H.");
    expectNeedleBefore(
        compact_harvest,
        "preparePrefixArchivePayloadReady(&device_hot_handle,stream,",
        "prefix_backend->deviceCopyAsync(",
        "Device-hot archive event preparation must precede every D2D.");
}

/**
 * @brief Locks every GPU logits host read behind the typed host bridge.
 *
 * A metadata-only LogitsLocalInfo view does not own graph completion ordering.
 * This policy guard prevents direct tensor downloads and prevents TP gather
 * code from silently returning to non-consuming views or guessed streams.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, GPUHostLogitsPublicationUsesTypedObservationBridge)
{
    const auto orchestrator_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto orchestrator_header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto runner_header =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto gatherer_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/LogitsGatherer.cpp");
    const auto rank_source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");

    const auto bridge_body = sliceBetween(
        orchestrator_source,
        "void *DeviceGraphOrchestrator::prepareLogitsHostObservation(",
        "const float *DeviceGraphOrchestrator::publishLogitsTensorToHost(");
    const auto main_accessor = sliceBetween(
        orchestrator_source,
        "const float *DeviceGraphOrchestrator::logits() const",
        "bool DeviceGraphOrchestrator::forwardMTP(");
    const auto mtp_accessor = sliceBetween(
        orchestrator_source,
        "const float *DeviceGraphOrchestrator::mtpLogits() const",
        "bool DeviceGraphOrchestrator::setComputeAllPositionLogits(");
    const auto verifier_accessor = sliceBetween(
        orchestrator_source,
        "const float *DeviceGraphOrchestrator::getAllPositionLogits() const",
        "std::string DeviceGraphOrchestrator::mtpDecodeUnsupportedReason() const");

    const auto compact_bridge =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(bridge_body));
    const auto compact_main =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(main_accessor));
    const auto compact_mtp =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(mtp_accessor));
    const auto compact_verifier =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(verifier_accessor));
    const auto compact_orchestrator_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(orchestrator_header));
    const auto compact_runner_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(runner_header));
    const auto compact_gatherer =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(gatherer_source));
    const auto compact_rank =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(rank_source));

    EXPECT_NE(compact_bridge.find("explicitGPUStreamForOperation("), std::string::npos);
    EXPECT_NE(compact_bridge.find("waitForForwardGraphOutputReady("), std::string::npos);
    EXPECT_NE(compact_bridge.find("waitForPendingLogitsStreamForObservation("), std::string::npos);
    EXPECT_NE(compact_bridge.find("waitForPendingAllPositionVerifierStateReadyForObservation("),
              std::string::npos);
    EXPECT_EQ(compact_bridge.find("waitForPendingLogitsStream("), std::string::npos);
    EXPECT_NE(compact_bridge.find("DeviceTimelineRole::HostResultBridge"), std::string::npos);

    EXPECT_NE(compact_main.find("publishLogitsTensorToHost("), std::string::npos);
    EXPECT_NE(compact_mtp.find("publishLogitsTensorToHost("), std::string::npos);
    EXPECT_NE(compact_verifier.find("publishLogitsTensorToHost("), std::string::npos);
    EXPECT_EQ(compact_main.find("fp32_data("), std::string::npos);
    EXPECT_EQ(compact_mtp.find("fp32_data("), std::string::npos);
    EXPECT_EQ(compact_verifier.find("fp32_data("), std::string::npos);

    EXPECT_NE(
        compact_runner_header.find("consumeLogitsLocalInfoForHostGather()"),
        std::string::npos);
    EXPECT_NE(
        compact_runner_header.find("consumeMTPLogitsLocalInfoForHostGather()"),
        std::string::npos);
    EXPECT_NE(
        compact_runner_header.find("consumeAllPositionLogitsLocalInfoForHostGather()"),
        std::string::npos);
    EXPECT_NE(
        compact_orchestrator_header.find("HostLogitsSurface::Main"),
        std::string::npos);
    EXPECT_NE(
        compact_orchestrator_header.find("HostLogitsSurface::MTPSidecar"),
        std::string::npos);
    EXPECT_NE(
        compact_orchestrator_header.find("HostLogitsSurface::AllPositionVerifier"),
        std::string::npos);

    EXPECT_NE(
        compact_gatherer.find("consumeLogitsLocalInfoForHostGather()"),
        std::string::npos);
    EXPECT_NE(
        compact_gatherer.find("if(!info.stream)"),
        std::string::npos);
    EXPECT_NE(
        compact_rank.find("consumeMTPLogitsLocalInfoForHostGather()"),
        std::string::npos);
    EXPECT_NE(
        compact_rank.find("consumeAllPositionLogitsLocalInfoForHostGather()"),
        std::string::npos);
}

/**
 * @brief Keep read-only verifier sampling from stealing producer readiness.
 *
 * The grouped verifier may be sampled on device and then surfaced to the host
 * for parity diagnostics or result gathering. Both operations are observers.
 * The transaction boundary, not the first observer, owns invalidation of the
 * request-scoped stream token.
 */
TEST(Test__GpuWorkspaceAllocationPolicy, AllPositionGreedyObserversPreserveProducerHandoff)
{
    const auto source =
        readFile(repoRoot() / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto single_sample = sliceBetween(
        source,
        "int DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDevice(",
        "bool DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows(");
    const auto batched_sample = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows(",
        "bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(");

    const auto compact_single =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(single_sample));
    const auto compact_batched =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(batched_sample));

    EXPECT_NE(compact_single.find(
                  "peekPendingLogitsStream("
                  "PendingLogitsStreamRole::AllPositionVerifier)"),
              std::string::npos);
    EXPECT_NE(compact_batched.find(
                  "peekPendingLogitsStream("
                  "PendingLogitsStreamRole::AllPositionVerifier)"),
              std::string::npos);
    EXPECT_EQ(compact_single.find("consumePendingLogitsStream("),
              std::string::npos);
    EXPECT_EQ(compact_batched.find("consumePendingLogitsStream("),
              std::string::npos);
    EXPECT_EQ(compact_single.find("clearPendingLogitsStream("),
              std::string::npos);
}

/**
 * @brief Eager forward-family declarations must retire their final event edge.
 *
 * Each forward graph build publishes device state. Intermediate declaration
 * edges are consumed before the next participant is built, so this source
 * contract protects the easy-to-miss terminal edge after the final participant.
 * Runtime graph construction must begin with unambiguous event ownership.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     ForwardWorkspaceFamilyManifestClosesFinalDeviceStatePublication)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto materialization_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::materializeForwardGraphForShape(",
        "ReceivedWeightsMap DeviceGraphOrchestrator::transferExpertWeights(");
    const auto compact =
        removeAsciiWhitespace(materialization_body);

    const std::string terminal_wait =
        "waitForPendingGraphBuildDeviceStateReady("
        "main_graph_build_device_state_ready_,"
        "publication_stream,"
        "DeviceTimelineRole::MainForwardGraph,"
        "\"forward_workspace_family_manifest_complete\")";
    ASSERT_NE(compact.find(terminal_wait), std::string::npos)
        << "The manifest must consume its final forward-build publication on "
           "the exact publication stream";

    expectNeedleBefore(
        materialization_body,
        "\"forward_workspace_family_manifest_complete\"",
        "buildMTPWorkspaceFamilyManifest(",
        "The forward publication must be retired before another graph-family "
        "manifest can publish device state");
}

/**
 * @brief The eager family manifest must include the live MTP condition graph.
 *
 * A condition graph has a role-specific GDN/short-conv namespace and consumes
 * durable device token, position, and sequence-length rows. Omitting that exact
 * participant lets first use discover new workspace names after addresses have
 * already been published to captured sibling graphs.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     ForwardWorkspaceFamilyManifestDeclaresMTPConditionParticipant)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto materialization_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::materializeForwardGraphForShape(",
        "ReceivedWeightsMap DeviceGraphOrchestrator::transferExpertWeights(");
    const auto compact =
        removeAsciiWhitespace(
            stripCommentsAndStringLiterals(materialization_body));

    EXPECT_NE(
        compact.find(
            "builder->setLiveMTPRequestBatchCondition(true)"),
        std::string::npos);
    EXPECT_NE(
        compact.find(
            "condition_input.token_ids_device="
            "logical_state.next_condition_tokens_device;"),
        std::string::npos);
    EXPECT_NE(
        compact.find(
            "condition_input.position_ids_device="
            "logical_state.target_cached_tokens_device;"),
        std::string::npos);
    EXPECT_NE(
        compact.find(
            "condition_input.sequence_lengths_device="
            "logical_state.target_cached_tokens_device;"),
        std::string::npos);
    EXPECT_NE(
        compact.find(
            "WorkspaceGraphParticipantRole::MTPCondition"),
        std::string::npos);
}

/**
 * @brief Bucket graph declarations bind the permanent request-length owner.
 *
 * Request admission publishes values only when a request exists, while graph
 * family declaration necessarily happens earlier.  This guard prevents future
 * code from treating runtime value readiness as pointer lifetime and restoring
 * a host scalar solely to make the bucket graph declarable.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     BucketPrefillManifestBindsPersistentDeviceRequestLengthOwner)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto materialization_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::materializeForwardGraphForShape(",
        "ReceivedWeightsMap DeviceGraphOrchestrator::transferExpertWeights(");
    const auto bucket_participant = sliceBetween(
        materialization_body,
        "ForwardInput bucket_input = input;",
        "forward_workspace_family_maximum_prefill_bucket");
    const auto compact = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(bucket_participant));

    EXPECT_NE(
        compact.find(
            "if(!request_sequence_lengths_dev_||"
            "request_sequence_lengths_capacity_<1)"),
        std::string::npos);
    EXPECT_NE(
        compact.find(
            "bucket_input.sequence_lengths_device="
            "static_cast<constint32_t*>(request_sequence_lengths_dev_);"),
        std::string::npos);
    EXPECT_EQ(compact.find("request_sequence_lengths_active_count_"),
              std::string::npos)
        << "Graph declaration authenticates the permanent address, not whether "
           "a runtime request has populated it yet.";
}

/**
 * @brief Maximum-prefill declaration uses the same durable row owner.
 *
 * The unbucketed maximum-shape participant is also constructed before request
 * admission. Diagnostic terminal-row selectors and any future graph-native
 * padded-row consumer must therefore capture the permanent arena address, not
 * depend on a host length or on the row already containing request data.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     MaximumPrefillManifestBindsPersistentDeviceRequestLengthOwner)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto materialization_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::materializeForwardGraphForShape(",
        "ReceivedWeightsMap DeviceGraphOrchestrator::transferExpertWeights(");
    const auto maximum_prefill_participant = sliceBetween(
        materialization_body,
        "ForwardInput maximum_prefill_input = input;",
        "forward_workspace_family_maximum_prefill");
    const auto compact = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(maximum_prefill_participant));

    EXPECT_NE(
        compact.find(
            "if(!request_sequence_lengths_dev_||"
            "request_sequence_lengths_capacity_<batch_size)"),
        std::string::npos);
    EXPECT_NE(
        compact.find(
            "maximum_prefill_input.sequence_lengths_device="
            "static_cast<constint32_t*>(request_sequence_lengths_dev_);"),
        std::string::npos);
    EXPECT_EQ(compact.find("request_sequence_lengths_active_count_"),
              std::string::npos)
        << "The manifest owns a stable address before any request publishes "
           "its length value.";
}

/**
 * @brief Grouped verifier graphs bind one participant-local geometry owner.
 *
 * LocalTP participants cannot borrow another device's request lengths, and the
 * prefill admission row describes prompt geometry rather than verifier padding.
 * The family manifest must therefore capture the dedicated arena row while the
 * replay prelude fills it from the same device valid-row metadata that selects
 * compact logits.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     GroupedVerifierManifestBindsDeviceDerivedRequestLengths)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto materialization_body = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::materializeForwardGraphForShape(",
        "ReceivedWeightsMap DeviceGraphOrchestrator::transferExpertWeights(");
    const auto grouped_participant = sliceBetween(
        materialization_body,
        "ForwardInput grouped_input = input;",
        "forward_workspace_family_grouped_verifier");
    const auto compact_grouped = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(grouped_participant));

    EXPECT_NE(
        compact_grouped.find(
            "grouped_input.sequence_lengths_device="
            "static_cast<constint32_t*>("
            "mtp_verifier_request_lengths_dev_);"),
        std::string::npos)
        << "The captured grouped graph must bind its own permanent length row.";

    const auto metadata_prelude = sliceBetween(
        source,
        "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(",
        "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(");
    const auto compact_prelude = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(metadata_prelude));
    const auto preparation_stage = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(readFile(
            repoRoot() /
            "src/v2/execution/compute_stages/stages/MTPVerifierPreparationStage.cpp")));
    EXPECT_NE(
        preparation_stage.find("enqueuePrepareMTPVerifierGeometry("),
        std::string::npos);
    EXPECT_EQ(
        compact_prelude.find("enqueuePrepareMTPVerifierGeometry("),
        std::string::npos)
        << "Geometry publication must not remain a loose pre-replay launch.";
    EXPECT_NE(
        compact_prelude.find("executeMTPVerifierPreparationCaptured("),
        std::string::npos);
    EXPECT_NE(
        compact_prelude.find("ptrs.verifier_logit_rows"),
        std::string::npos)
        << "Ragged recurrent masks must derive from the exact compact-row publication.";
    EXPECT_NE(
        compact_prelude.find("mtp_verifier_request_lengths_dev_"),
        std::string::npos);
    EXPECT_EQ(compact_prelude.find("hostToDevice"), std::string::npos)
        << "Grouped request lengths must not add a host metadata transfer.";
    EXPECT_EQ(compact_prelude.find("synchronizeStream("), std::string::npos);
    EXPECT_EQ(compact_prelude.find("synchronizeDevice("), std::string::npos);
}

/**
 * @brief Typed GPU MTP logits must resolve preplanned arena storage first.
 *
 * The generic all-position diagnostics path may still materialize a
 * row-specific tensor, but grouped verifier and condition execution must return
 * from the prepared-owner branch before that allocation site is reachable.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     TypedGPUMTPLogitsCannotReachDynamicTensorAllocation)
{
    const auto source = readFile(
        repoRoot() /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto forward_body = sliceBetween(
        source,
        "const float *DeviceGraphOrchestrator::forwardImpl(",
        "bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule(");

    expectNeedleBefore(
        forward_body,
        "if (typed_gpu_mtp_logits)",
        "tensor_factory_->createFP32(",
        "Typed GPU MTP logits must resolve arena-owned capacity before the "
        "generic row-specific allocation path");
    EXPECT_NE(
        forward_body.find(
            "auto prepared = preplannedMTPLogitsBuffer("),
        std::string::npos);
    EXPECT_NE(
        forward_body.find(
            "if (!prepared)"),
        std::string::npos);
    EXPECT_NE(
        forward_body.find(
        "return prepared;"),
        std::string::npos);
}

/**
 * @brief Lock the fused stochastic MTP outcome into a device-only graph node.
 *
 * Seeded stochastic MTP must sample all compact target rows and summarize the
 * transaction without returning to the host between those operations. This
 * source contract complements the CUDA/ROCm byte-exact integration test: it
 * prevents a later implementation from quietly reintroducing allocation,
 * transfer, callback, or synchronization work into either the typed capture
 * stage or its backend launch wrappers.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     FusedStochasticSerialOutcomeIsStrictDeviceOnlyGraphWork)
{
    const auto root = repoRoot();
    const auto stage_source = readFile(
        root /
        "src/v2/execution/compute_stages/stages/MTPStochasticSerialOutcomeStage.cpp");
    const auto cuda_source = readFile(
        root / "src/v2/kernels/cuda/ops/CUDASamplingKernels.cu");
    const auto rocm_source = readFile(
        root / "src/v2/kernels/rocm/ops/ROCmSamplingKernels.hip");
    const auto orchestrator_source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto stage_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            stage_source,
            "bool MTPStochasticSerialOutcomeStage::execute(IDeviceContext *ctx)",
            "size_t MTPStochasticSerialOutcomeStage::estimatedMemoryBytes() const")));
    const auto cuda_wrapper = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            cuda_source,
            "cudaOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(",
            "bool cudaOps_summarize_greedy_speculative_verify_batch(")));
    const auto rocm_wrapper = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            rocm_source,
            "rocmOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(",
            "bool rocmOps_summarize_greedy_speculative_verify_batch(")));
    const auto captured_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::executeMTPStochasticSerialOutcomeCaptured(",
            "bool DeviceGraphOrchestrator::materializeMTPSpeculativeStatePublicationGraph(")));

    EXPECT_NE(
        stage_execute.find(
            "enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls("),
        std::string::npos);
    EXPECT_NE(stage_execute.find("requireGPUStream()"), std::string::npos)
        << "The fused stage must reject a null/default stream.";

    const std::array<const char *, 14> forbidden = {
        "cudaMalloc",
        "hipMalloc",
        "cudaFree",
        "hipFree",
        "cudaMemcpy",
        "hipMemcpy",
        "hostToDevice",
        "deviceToHost",
        "synchronizeStream",
        "synchronizeDevice",
        "cudaStreamSynchronize",
        "hipStreamSynchronize",
        "addHostCallback",
        "enqueueHostCallback"};
    for (const char *needle : forbidden)
    {
        EXPECT_EQ(stage_execute.find(needle), std::string::npos)
            << "Captured stage contains forbidden host/runtime work: "
            << needle;
        EXPECT_EQ(cuda_wrapper.find(needle), std::string::npos)
            << "CUDA fused wrapper contains forbidden host/runtime work: "
            << needle;
        EXPECT_EQ(rocm_wrapper.find(needle), std::string::npos)
            << "ROCm fused wrapper contains forbidden host/runtime work: "
            << needle;
    }
    EXPECT_EQ(cuda_wrapper.find("atomic"), std::string::npos);
    EXPECT_EQ(rocm_wrapper.find("atomic"), std::string::npos);
    EXPECT_NE(cuda_wrapper.find("!stream"), std::string::npos);
    EXPECT_NE(rocm_wrapper.find("!stream"), std::string::npos);

    EXPECT_NE(
        captured_execute.find(
            "GraphReplayPlanPolicy::RequireFullGraph"),
        std::string::npos)
        << "Homogeneous GPU MTP outcome reduction must fail closed unless the "
           "whole typed stage is captured.";
    EXPECT_NE(
        captured_execute.find("capture_policy.defer_final_sync=true;"),
        std::string::npos);
    EXPECT_EQ(
        captured_execute.find("GraphReplayPlanPolicy::AllowSegmented"),
        std::string::npos);
    EXPECT_EQ(captured_execute.find("force_recapture=true"), std::string::npos);
    EXPECT_EQ(captured_execute.find("synchronizeStream("), std::string::npos);
    EXPECT_EQ(captured_execute.find("synchronizeDevice("), std::string::npos);
}

/**
 * @brief Keep grouped-verifier transaction preparation monolithic and resident.
 *
 * Token composition, complete main-KV checkpointing, verifier geometry, and
 * base-count publication describe one speculative state boundary. They must be
 * captured as one typed stage and connected to the forward graph by explicit
 * producer/consumer events. Request penalty policy is a separate admission-time
 * owner and this stage is forbidden from republishing it.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     GroupedVerifierPreparationIsOneStrictDeviceGraph)
{
    const auto root = repoRoot();
    const auto stage_source = readFile(
        root /
        "src/v2/execution/compute_stages/stages/MTPVerifierPreparationStage.cpp");
    const auto orchestrator_source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto stage_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            stage_source,
            "bool MTPVerifierPreparationStage::execute(IDeviceContext *ctx)",
            "size_t MTPVerifierPreparationStage::estimatedFlops() const")));
    const auto captured_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::executeMTPVerifierPreparationCaptured(",
            "bool DeviceGraphOrchestrator::\n        materializeMTPStochasticTargetDistributionGraph(")));
    const auto metadata_prelude = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(",
            "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(")));
    const auto token_prelude = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::prepareDeviceTokenInputsForForwardGraphExecution(",
            "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(")));

    EXPECT_NE(stage_execute.find("requireGPUStream()"), std::string::npos);
    EXPECT_NE(stage_execute.find("deviceCopyAsync("), std::string::npos);
    EXPECT_NE(
        stage_execute.find("captureDeviceSequenceStateCheckpoint("),
        std::string::npos);
    EXPECT_NE(
        stage_execute.find("enqueuePrepareMTPVerifierGeometry("),
        std::string::npos);
    EXPECT_EQ(
        stage_execute.find("enqueueConfigureMTPGreedyPenaltyPolicyDevice("),
        std::string::npos);
    EXPECT_NE(
        metadata_prelude.find("consumeMTPRequestPenaltyPolicyOnDevice("),
        std::string::npos);

    const std::array<const char *, 16> forbidden = {
        "cudaMalloc",
        "hipMalloc",
        "cudaFree",
        "hipFree",
        "hostToDevice",
        "deviceToHost",
        "cudaMemcpyHostToDevice",
        "hipMemcpyHostToDevice",
        "cudaMemcpyDeviceToHost",
        "hipMemcpyDeviceToHost",
        "synchronizeStream",
        "synchronizeDevice",
        "cudaStreamSynchronize",
        "hipStreamSynchronize",
        "addHostCallback",
        "enqueueHostCallback"};
    for (const char *needle : forbidden)
    {
        EXPECT_EQ(stage_execute.find(needle), std::string::npos)
            << "Captured verifier-preparation stage contains forbidden work: "
            << needle;
    }

    EXPECT_NE(
        captured_execute.find("GraphReplayPlanPolicy::RequireFullGraph"),
        std::string::npos);
    EXPECT_NE(
        captured_execute.find("orderCaptureStreamAfter("),
        std::string::npos);
    EXPECT_NE(
        captured_execute.find("orderStreamAfterCapture("),
        std::string::npos);
    EXPECT_EQ(
        captured_execute.find("GraphReplayPlanPolicy::AllowSegmented"),
        std::string::npos);
    EXPECT_EQ(captured_execute.find("force_recapture=true"), std::string::npos);
    EXPECT_EQ(captured_execute.find("synchronizeStream("), std::string::npos);
    EXPECT_EQ(captured_execute.find("synchronizeDevice("), std::string::npos);

    EXPECT_NE(
        metadata_prelude.find("materializeMTPVerifierPreparationGraph("),
        std::string::npos);
    EXPECT_NE(
        metadata_prelude.find("executeMTPVerifierPreparationCaptured("),
        std::string::npos);
    EXPECT_EQ(
        metadata_prelude.find("captureDeviceSequenceStateCheckpoint("),
        std::string::npos)
        << "KV checkpointing must be owned by the captured stage.";
    EXPECT_EQ(
        metadata_prelude.find("enqueuePrepareMTPVerifierGeometry("),
        std::string::npos)
        << "Geometry must be owned by the captured stage.";
    EXPECT_EQ(metadata_prelude.find("std::vector<"), std::string::npos)
        << "Verifier replay selection must use preallocated identity scratch.";
    EXPECT_NE(
        token_prelude.find(
            "ForwardExecutionRole::GroupedMTPVerifier"),
        std::string::npos);
    EXPECT_NE(
        token_prelude.find(
            "plan.all_tokens_from_host||!plan.first_token_from_device"),
        std::string::npos);
    EXPECT_NE(
        token_prelude.find("returntrue;"),
        std::string::npos)
        << "The token hook must defer the grouped transaction to metadata preparation.";
}

/**
 * @brief Give accepted-state publication sole ownership of MTP penalty history.
 *
 * Request penalty magnitudes are immutable admission data, while the pending
 * condition predicate and generated-token histogram evolve after each accepted
 * speculative transaction. The reducer must only describe an outcome. Exactly
 * one captured publication stage then commits both mutable values on the
 * outcome stream. Disabled penalties still retain the same single-request graph
 * topology: the resident policy makes the publication kernel a device no-op.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     MTPPenaltyHistoryHasOneDeviceWriterAndNoPolicyGraphIdentity)
{
    const auto root = repoRoot();
    const auto runner_interface = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/IInferenceRunner.h");
    const auto outcome_source = readFile(
        root /
        "src/v2/execution/compute_stages/stages/MTPVerifierOutcomeStage.cpp");
    const auto publication_header = readFile(
        root /
        "src/v2/execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.h");
    const auto publication_source = readFile(
        root /
        "src/v2/execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.cpp");
    const auto orchestrator_source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto request = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            runner_interface,
            "struct DeviceSpeculativePublicationRequest",
            "struct DeviceResidentLogicalSequenceStateHandle")));
    const auto outcome_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            outcome_source,
            "bool MTPVerifierOutcomeStage::execute(IDeviceContext *ctx)",
            "size_t MTPVerifierOutcomeStage::estimatedMemoryBytes() const")));
    const auto publication = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(publication_source));
    const auto capture_identity = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            publication_source,
            "bool MTPSpeculativeStatePublicationStage::hasSameCaptureIdentity(",
            "} // namespace llaminar2")));
    const auto device_publication = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
            "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(")));
    const auto lifecycle_finalize = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::finalizeMTPSpeculativeStatePublicationLaunch(",
            "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(")));

    EXPECT_NE(request.find("MTPRequestPenaltyPolicypenalty_policy{}"),
              std::string::npos);
    EXPECT_EQ(request.find("MTPGreedyPenaltyPolicy"), std::string::npos)
        << "Mutable device policy state must not be authorable by a host request.";
    EXPECT_EQ(
        outcome_execute.find("enqueueCommitMTPGreedyPenaltyHistoryDevice("),
        std::string::npos)
        << "Outcome reduction describes acceptance but may not mutate history.";
    EXPECT_EQ(
        countOccurrences(
            publication,
            "enqueueCommitMTPGreedyPenaltyHistoryDevice("),
        1U)
        << "Accepted-state publication must be the sole production history writer.";
    EXPECT_EQ(
        removeAsciiWhitespace(stripCommentsAndStringLiterals(publication_header))
            .find("commit_penalty_history"),
        std::string::npos)
        << "Penalty enabledness must never select a different captured topology.";
    EXPECT_EQ(capture_identity.find("presence_penalty"), std::string::npos);
    EXPECT_EQ(capture_identity.find("frequency_penalty"), std::string::npos);
    EXPECT_EQ(
        capture_identity.find("first_token_already_in_history"),
        std::string::npos);

    EXPECT_NE(
        device_publication.find(
            "request.request_count==1&&(!arena_"),
        std::string::npos)
        << "Every single-request publication must consume the resident policy, "
           "including the disabled-policy no-op topology.";
    EXPECT_EQ(
        device_publication.find("request.penalty_policy.enabled()"),
        std::string::npos)
        << "Host policy values must not select publication graph structure.";
    EXPECT_NE(
        lifecycle_finalize.find("if(request.request_count==1)"),
        std::string::npos);
    EXPECT_EQ(
        countOccurrences(
            lifecycle_finalize,
            "arena_->markWritten("),
        2U)
        << "The exact accepted-state stream must publish both policy and histogram readiness.";
}

/**
 * @brief Prevent parent composition from selecting a stale preparation capture.
 *
 * Geometry-only family lookup is unsafe because two captures can have equal
 * dimensions while naming different token rows, KV checkpoints, or arena
 * generations. The current transaction must publish one exact family/stage
 * identity after successful replay, and every failed replacement attempt must
 * retire the previously exportable identity first.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     GroupedVerifierPreparationExportsOnlyTheActiveExactCapture)
{
    const auto root = repoRoot();
    const auto header = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto compact_header =
        removeAsciiWhitespace(stripCommentsAndStringLiterals(header));
    const auto metadata_prelude = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(",
            "bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(")));
    const auto exporter = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "mtpVerifierPreparationDeviceLoopGraphTemplate(",
            "mtpAllPositionVerifierDeviceLoopGraphTemplate(")));
    const auto verifier_exporter = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "mtpAllPositionVerifierDeviceLoopGraphTemplate(",
            "mtpAcceptedTerminalHiddenDeviceLoopGraphTemplate(")));

    EXPECT_NE(
        compact_header.find("structActiveMTPVerifierPreparationGraph"),
        std::string::npos);
    EXPECT_NE(
        compact_header.find(
            "constMTPVerifierPreparationStage*stage=nullptr"),
        std::string::npos)
        << "The active identity must distinguish replacement at the same family index.";
    EXPECT_NE(
        compact_header.find("uint64_tworkspace_generation=0"),
        std::string::npos);

    const size_t clear_position = metadata_prelude.find(
        "active_mtp_verifier_preparation_graph_.clear()");
    const size_t replay_position = metadata_prelude.find(
        "executeMTPVerifierPreparationCaptured(");
    const size_t publish_position = metadata_prelude.find(
        "active_mtp_verifier_preparation_graph_={");
    ASSERT_NE(clear_position, std::string::npos);
    ASSERT_NE(replay_position, std::string::npos);
    ASSERT_NE(publish_position, std::string::npos);
    EXPECT_LT(clear_position, replay_position)
        << "A failed new transaction must not leave the old capture exportable.";
    EXPECT_LT(replay_position, publish_position)
        << "Only a successfully replayed preparation may become active.";

    EXPECT_NE(exporter.find("active.valid()"), std::string::npos);
    EXPECT_NE(exporter.find("active.request_count!=request_count"),
              std::string::npos);
    EXPECT_NE(exporter.find("active.padded_seq_len!=padded_seq_len"),
              std::string::npos);
    EXPECT_NE(
        exporter.find(
            "active.workspace_generation!=generation"),
        std::string::npos);
    EXPECT_NE(
        exporter.find("cache.stage!=active.stage"),
        std::string::npos)
        << "The exporter must reject a family slot repopulated with another stage.";
    EXPECT_NE(exporter.find("mtp_verifier_preparation_graphs_[active.family_index]"),
              std::string::npos);
    EXPECT_NE(exporter.find("deviceLoopGraphTemplate("),
              std::string::npos)
        << "The common strict capture validator must reject segmented graphs.";
    EXPECT_EQ(exporter.find("hasSameCaptureIdentity("), std::string::npos)
        << "Export is not a geometry search; it must use the retained producer identity.";

    EXPECT_NE(
        verifier_exporter.find(
            "mtpVerifierPreparationDeviceLoopGraphTemplate("),
        std::string::npos)
        << "The forward exporter must validate the exact preparation producer first.";
    EXPECT_NE(
        verifier_exporter.find(
            "lastAllPositionVerifierDeviceLoopGraphTemplate("),
        std::string::npos);
    EXPECT_NE(verifier_exporter.find("signature.seq_len!=padded_seq_len"),
              std::string::npos);
    EXPECT_NE(verifier_exporter.find("signature.batch_size!=request_count"),
              std::string::npos);
    EXPECT_NE(
        verifier_exporter.find(
            "signature.all_position_logit_rows!=preparation.valid_graph_row_count"),
        std::string::npos)
        << "The forward and preparation captures must name the same compact rows.";
    EXPECT_NE(verifier_exporter.find("!signature.uses_device_token_ids"),
              std::string::npos);
    EXPECT_NE(verifier_exporter.find("!signature.uses_device_position_ids"),
              std::string::npos);
    EXPECT_NE(
        verifier_exporter.find("!signature.uses_device_sequence_lengths"),
        std::string::npos);

    EXPECT_GE(
        countOccurrences(
            source,
            "active_mtp_verifier_preparation_graph_.clear();"),
        4U)
        << "Workspace setup, invalidation, and every new transaction must retire the active identity.";
}

/**
 * @brief Lock fixed-depth stochastic generation into one native device loop.
 *
 * The parent is valid only when every production fragment exports one strict
 * monolithic capture for the active request.  Depth D has D draft rows, D+1
 * verifier/bonus rows, and exactly 2D+6 ordered child graphs for static/LLEP.
 * Dynamic placement appends one device-gated maintenance transaction, making
 * 2D+7. Request admission and child replacement must retire prior parent
 * identity so a changed seed or sampling policy can never replay an older
 * executable.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     DeviceGenerationParentOwnsTheCompleteOrderedTransaction)
{
    const auto root = repoRoot();
    const auto header = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h");
    const auto source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto cuda_capture = readFile(
        root / "src/v2/backends/cuda/CUDAGraphCapture.cu");
    const auto runner_source = readFile(
        root / "src/v2/execution/runner/OrchestrationRunner.cpp");

    const auto compact_header = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(header));
    const auto composer = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::materializeMTPDeviceGenerationLoopGraph(",
            "bool DeviceGraphOrchestrator::executeMTPDepth0(")));
    const auto publication = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(",
            "bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(")));
    const auto admission = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            source,
            "bool DeviceGraphOrchestrator::beginDeviceResidentStochasticGeneration(",
            "bool DeviceGraphOrchestrator::finishDeviceResidentStochasticGeneration(")));
    const auto conditional_builder = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            cuda_capture,
            "bool CUDAGraphCapture::buildDeviceControlledWhileLoop(",
            "GraphUpdateResult CUDAGraphCapture::tryUpdate()")));
    const auto fixed_depth_runner = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            runner_source,
            "if (mtp.depth_policy.mode == MTPDepthPolicyMode::Fixed &&",
            "Device publication derives next-condition rows into the")));

    EXPECT_NE(
        compact_header.find(
            "std::vector<constIGPUGraphCapture*>source_fragments"),
        std::string::npos)
        << "The parent must retain the exact capture identities that it cloned.";
    EXPECT_NE(composer.find("request_count!=1"), std::string::npos);
    EXPECT_NE(
        composer.find("verifier_rows_per_request=draft_depth+1"),
        std::string::npos);
    EXPECT_NE(
        composer.find("expected_fragment_count=static_cast<size_t>(2*draft_depth+6)"),
        std::string::npos);
    EXPECT_NE(
        composer.find("include_device_moe_maintenance?1u:0u"),
        std::string::npos);
    EXPECT_NE(
        composer.find("source_fragments.capacity()<expected_fragment_count"),
        std::string::npos)
        << "Composition storage must be reserved before decode.";

    const size_t maintenance = composer.find(
        "maintenance.segment_cache.deviceLoopGraphTemplate(");
    const size_t full_sidecar = composer.find(
        "mtpSidecarDeviceLoopGraphTemplate(MTPSidecarCaptureRole::Full");
    const size_t draft_publication = composer.find(
        "mtpDraftTokenPublicationDeviceLoopGraphTemplate(");
    const size_t verifier_preparation = composer.find(
        "mtpVerifierPreparationDeviceLoopGraphTemplate(");
    const size_t verifier_forward = composer.find(
        "mtpAllPositionVerifierDeviceLoopGraphTemplate(");
    const size_t target_distribution = composer.find(
        "mtpStochasticTargetDistributionDeviceLoopGraphTemplate(");
    const size_t serial_outcome = composer.find(
        "mtpStochasticSerialOutcomeDeviceLoopGraphTemplate(");
    const size_t state_publication = composer.find(
        "mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(");
    const size_t terminal_hidden = composer.find(
        "mtpAcceptedTerminalHiddenDeviceLoopGraphTemplate(");
    ASSERT_NE(maintenance, std::string::npos);
    ASSERT_NE(full_sidecar, std::string::npos);
    ASSERT_NE(draft_publication, std::string::npos);
    ASSERT_NE(verifier_preparation, std::string::npos);
    ASSERT_NE(verifier_forward, std::string::npos);
    ASSERT_NE(target_distribution, std::string::npos);
    ASSERT_NE(serial_outcome, std::string::npos);
    ASSERT_NE(state_publication, std::string::npos);
    ASSERT_NE(terminal_hidden, std::string::npos);
    EXPECT_LT(full_sidecar, draft_publication);
    EXPECT_LT(draft_publication, verifier_preparation);
    EXPECT_LT(verifier_preparation, verifier_forward);
    EXPECT_LT(verifier_forward, target_distribution);
    EXPECT_LT(target_distribution, serial_outcome);
    EXPECT_LT(serial_outcome, state_publication);
    EXPECT_LT(state_publication, terminal_hidden);
    EXPECT_LT(terminal_hidden, maintenance)
        << "Dynamic maintenance must consume exactly the transaction committed by the preceding parent fragments.";
    EXPECT_NE(
        composer.find("MTPSidecarCaptureRole::Chained"),
        std::string::npos);

    EXPECT_NE(
        composer.find("supportsDeviceControlledWhileLoop()"),
        std::string::npos);
    EXPECT_NE(
        composer.find("buildDeviceControlledWhileLoop("),
        std::string::npos);
    EXPECT_NE(composer.find("capture->instantiate()"), std::string::npos);
    EXPECT_NE(composer.find("capture->hasExecutable()"), std::string::npos);
    EXPECT_NE(
        composer.find("healthy_index=kDeviceGenerationControlOk"),
        std::string::npos);
    EXPECT_NE(
        composer.find(
            "complete_index=kDeviceGenerationControlRequestComplete"),
        std::string::npos);
    EXPECT_NE(composer.find("source_identity_matches"), std::string::npos);

    const std::array<const char *, 18> forbidden = {
        "cudaMalloc",
        "hipMalloc",
        "cudaFree",
        "hipFree",
        "backend->allocate(",
        "createStream(",
        "hostToDevice",
        "deviceToHost",
        "cudaMemcpy",
        "hipMemcpy",
        "synchronizeStream",
        "synchronizeDevice",
        "cudaStreamSynchronize",
        "hipStreamSynchronize",
        "addHostCallback",
        "enqueueHostCallback",
        "copyDeviceSpeculativeOutcomesToHost",
        "GraphReplayPlanPolicy::AllowSegmented"};
    for (const char *needle : forbidden)
    {
        EXPECT_EQ(composer.find(needle), std::string::npos)
            << "Device-generation parent composition contains forbidden work: "
            << needle;
    }

    const size_t terminal_selection = publication.find(
        "selectMTPTerminalHiddenRowsFromDeviceAcceptedState(");
    const size_t lifecycle_finalize = publication.find(
        "finalizeMTPSpeculativeStatePublicationLaunch(");
    const size_t verifier_clear = publication.find(
        "clearLastAllPositionVerifierForwardGraph()");
    ASSERT_NE(terminal_selection, std::string::npos);
    ASSERT_NE(lifecycle_finalize, std::string::npos);
    ASSERT_NE(verifier_clear, std::string::npos);
    EXPECT_LT(terminal_selection, lifecycle_finalize);
    EXPECT_LT(lifecycle_finalize, verifier_clear);
    EXPECT_EQ(
        publication.find("materializeMTPDeviceGenerationLoopGraph("),
        std::string::npos)
        << "Parent composition is a distinct all-participant prelaunch phase, not part of mailbox publication.";
    EXPECT_NE(
        admission.find("mtp_device_generation_loop_graph_.invalidateGraph()"),
        std::string::npos)
        << "Every request must retire parent graphs containing old seeds or policy.";
    EXPECT_NE(
        admission.find("consumeDeviceGenerationStateReady("),
        std::string::npos);
    EXPECT_NE(admission.find("loop.capture->launch()"), std::string::npos);
    EXPECT_NE(
        admission.find("publishDeviceGenerationStateReady("),
        std::string::npos)
        << "The terminal bridge must consume one event published after the complete parent launch.";

    const size_t first_maintenance = fixed_depth_runner.find(
        "publishDeviceMoEMaintenanceBeforeMTPConsumer(");
    const size_t parent_prepare = fixed_depth_runner.find(
        "runner_->materializeDeviceResidentStochasticGeneration(");
    const size_t parent_launch = fixed_depth_runner.find(
        "runner_->launchDeviceResidentStochasticGeneration()");
    ASSERT_NE(first_maintenance, std::string::npos);
    ASSERT_NE(parent_prepare, std::string::npos);
    ASSERT_NE(parent_launch, std::string::npos);
    EXPECT_LT(first_maintenance, parent_prepare)
        << "First-transaction maintenance must make its child replay-ready before parent composition.";
    EXPECT_LT(parent_prepare, parent_launch)
        << "Every participant must finish parent composition before any parent launch.";
    EXPECT_NE(
        fixed_depth_runner.find(
            "runner_->finishDeviceResidentStochasticGeneration(&terminal)"),
        std::string::npos);
    EXPECT_EQ(
        fixed_depth_runner.find(
            "materializeDeviceSpeculativeOutcomesForHostResponse("),
        std::string::npos)
        << "Fixed-depth production generation must not enter the per-transaction compatibility bridge.";

    EXPECT_NE(
        conditional_builder.find(
            "cudaGraphConditionalHandleCreate(&condition,graph_,0,cudaGraphCondAssignDefault)"),
        std::string::npos);
    EXPECT_NE(
        conditional_builder.find("cudaGraphAddKernelNode(&initial_predicate_node"),
        std::string::npos);
    EXPECT_NE(
        conditional_builder.find(
            "cudaGraphAddNode(&conditional_node,graph_,&initial_predicate_node"),
        std::string::npos)
        << "The WHILE node must consume the live controller before iteration zero.";
}

/**
 * @brief Lock stochastic verifier target preparation into one device graph.
 *
 * Device-history penalties and compact target-distribution construction are a
 * single serial-sampling transaction. Splitting those calls lets a parent MTP
 * graph omit penalties, consume stale logits, or publish the wrong producer
 * stream. This source contract makes the typed stage device-only, requires a
 * strict monolithic graph, and forbids the canonical single-request production
 * lanes from invoking the two old operations independently.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     StochasticVerifierTargetPreparationIsOneStrictDeviceGraph)
{
    const auto root = repoRoot();
    const auto stage_source = readFile(
        root /
        "src/v2/execution/compute_stages/stages/MTPStochasticTargetDistributionStage.cpp");
    const auto orchestrator_source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const auto runner_source = readFile(
        root / "src/v2/execution/runner/OrchestrationRunner.cpp");

    const auto stage_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            stage_source,
            "bool MTPStochasticTargetDistributionStage::execute(IDeviceContext *ctx)",
            "size_t MTPStochasticTargetDistributionStage::estimatedFlops() const")));
    const auto captured_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "executeMTPStochasticTargetDistributionCaptured(",
            "bool DeviceGraphOrchestrator::materializeMTPStochasticSerialOutcomeGraph(")));
    const auto public_build = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "buildCapturedStochasticVerifierTargetDistributions(",
            "bool DeviceGraphOrchestrator::buildStochasticDistributionsOnDevice(")));
    const auto mtp_decode = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            runner_source,
            "GenerationResult OrchestrationRunner::decodeStepMTP()",
            "GenerationResult OrchestrationRunner::decodeStep()")));

    EXPECT_NE(stage_execute.find("requireGPUStream()"), std::string::npos);
    EXPECT_EQ(
        stage_execute.find(
            "enqueueConfigureMTPGreedyPenaltyPolicyDevice("),
        std::string::npos);
    EXPECT_NE(
        public_build.find("consumeMTPRequestPenaltyPolicyOnDevice("),
        std::string::npos);
    EXPECT_NE(
        stage_execute.find("enqueueApplyMTPPenaltiesToF32RowsDevice("),
        std::string::npos);
    EXPECT_NE(
        stage_execute.find(
            "enqueueBuildTopKTopPDistributionsF32Device("),
        std::string::npos);

    const std::array<const char *, 14> forbidden = {
        "cudaMalloc",
        "hipMalloc",
        "cudaFree",
        "hipFree",
        "cudaMemcpy",
        "hipMemcpy",
        "hostToDevice",
        "deviceToHost",
        "synchronizeStream",
        "synchronizeDevice",
        "cudaStreamSynchronize",
        "hipStreamSynchronize",
        "addHostCallback",
        "enqueueHostCallback"};
    for (const char *needle : forbidden)
    {
        EXPECT_EQ(stage_execute.find(needle), std::string::npos)
            << "Captured target-preparation stage contains forbidden work: "
            << needle;
    }

    EXPECT_NE(
        captured_execute.find("GraphReplayPlanPolicy::RequireFullGraph"),
        std::string::npos);
    EXPECT_NE(
        captured_execute.find("orderCaptureStreamAfter("),
        std::string::npos);
    EXPECT_NE(
        captured_execute.find("orderStreamAfterCapture("),
        std::string::npos);
    EXPECT_EQ(
        captured_execute.find("GraphReplayPlanPolicy::AllowSegmented"),
        std::string::npos);
    EXPECT_EQ(
        captured_execute.find("synchronizeStream("),
        std::string::npos);
    EXPECT_EQ(
        captured_execute.find("synchronizeDevice("),
        std::string::npos);

    EXPECT_NE(
        public_build.find(
            "peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier)"),
        std::string::npos)
        << "The captured transaction must consume the exact verifier producer.";
    EXPECT_EQ(
        public_build.find("explicitGPUStreamForOperation("),
        std::string::npos)
        << "A missing verifier producer stream is fatal, not replaceable.";

    EXPECT_GE(
        countOccurrences(
            mtp_decode,
            "buildCapturedStochasticVerifierTargetDistributions("),
        2U)
        << "Both canonical single-request stochastic verifier lanes must use "
           "the combined captured transaction.";
    EXPECT_EQ(
        mtp_decode.find(
            "applyDeviceOwnedMTPPenaltiesToLogitRows(DeviceLogitsSource::AllPosition"),
        std::string::npos)
        << "Verifier penalties may not remain a separately callable production step.";
}

/**
 * @brief Lock MTP proposal sampling behind one strict device graph.
 *
 * The sidecar-to-proposal boundary is latency-sensitive and ordering-critical.
 * This contract prevents a later edit from restoring eager argmax, inventing a
 * replacement stream, or adding host diagnostics to the deferred production
 * lane. Optional branch penalties and argmax publication remain one typed,
 * monolithic, exportable transaction.
 */
TEST(Test__GpuWorkspaceAllocationPolicy,
     MTPDraftPublicationIsOneStrictDeviceGraph)
{
    const auto root = repoRoot();
    const auto stage_source = readFile(
        root /
        "src/v2/execution/compute_stages/stages/MTPDraftTokenPublicationStage.cpp");
    const auto orchestrator_source = readFile(
        root /
        "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");

    const auto stage_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            stage_source,
            "bool MTPDraftTokenPublicationStage::execute(IDeviceContext *ctx)",
            "size_t MTPDraftTokenPublicationStage::estimatedFlops() const")));
    const auto captured_execute = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::executeMTPDraftTokenPublicationCaptured(",
            "materializeMTPStochasticTargetDistributionGraph(")));
    const auto public_publish = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::publishCapturedMTPDraftToken(",
            "bool DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDeviceImpl(")));
    const auto deferred_impl = removeAsciiWhitespace(
        stripCommentsAndStringLiterals(sliceBetween(
            orchestrator_source,
            "bool DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDeviceImpl(",
            "int DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDevice(")));

    EXPECT_NE(stage_execute.find("requireGPUStream()"), std::string::npos);
    EXPECT_NE(
        stage_execute.find(
            "enqueueApplyMTPBranchPenaltiesToF32RowDevice("),
        std::string::npos);
    EXPECT_NE(
        stage_execute.find(
            "enqueueArgmaxF32BatchedRowsAndPublishMTPChainDevice("),
        std::string::npos);
    EXPECT_NE(
        stage_execute.find("next_chain_condition_token_device"),
        std::string::npos)
        << "The exact argmax producer must publish the chained condition token.";
    EXPECT_NE(
        stage_execute.find("next_chain_position_id_device"),
        std::string::npos)
        << "The exact argmax producer must advance the chained position mailbox.";
    EXPECT_NE(
        stage_execute.find("chain_position_increment"),
        std::string::npos)
        << "Chained position advancement must be explicit capture-time policy.";
    EXPECT_EQ(
        stage_execute.find("enqueueArgmaxF32BatchedRowsDevice("),
        std::string::npos)
        << "Draft publication may not bypass producer-owned sidecar publication.";

    const std::array<const char *, 14> forbidden = {
        "cudaMalloc",
        "hipMalloc",
        "cudaFree",
        "hipFree",
        "cudaMemcpy",
        "hipMemcpy",
        "hostToDevice",
        "deviceToHost",
        "synchronizeStream",
        "synchronizeDevice",
        "cudaStreamSynchronize",
        "hipStreamSynchronize",
        "addHostCallback",
        "enqueueHostCallback"};
    for (const char *needle : forbidden)
    {
        EXPECT_EQ(stage_execute.find(needle), std::string::npos)
            << "Captured draft-publication stage contains forbidden work: "
            << needle;
    }

    EXPECT_NE(
        captured_execute.find("GraphReplayPlanPolicy::RequireFullGraph"),
        std::string::npos);
    EXPECT_NE(
        captured_execute.find("orderCaptureStreamAfter("),
        std::string::npos);
    EXPECT_NE(
        captured_execute.find("orderStreamAfterCapture("),
        std::string::npos);
    EXPECT_EQ(
        captured_execute.find("GraphReplayPlanPolicy::AllowSegmented"),
        std::string::npos);
    EXPECT_EQ(captured_execute.find("synchronizeStream("), std::string::npos);
    EXPECT_EQ(captured_execute.find("synchronizeDevice("), std::string::npos);

    EXPECT_NE(
        public_publish.find(
            "consumePendingLogitsStream(PendingLogitsStreamRole::MTPSidecar"),
        std::string::npos)
        << "Draft publication must take the exact sidecar producer.";
    EXPECT_EQ(
        public_publish.find("explicitGPUStreamForOperation("),
        std::string::npos)
        << "A missing sidecar producer is fatal, not replaceable.";
    EXPECT_NE(
        public_publish.find("recordStochasticDraftSampleReady("),
        std::string::npos);

    EXPECT_NE(
        deferred_impl.find("publishCapturedMTPDraftToken("),
        std::string::npos)
        << "The deferred production sampler must exercise the captured stage.";
    EXPECT_EQ(
        deferred_impl.find("enqueueArgmaxF32BatchedRowsDevice("),
        std::string::npos)
        << "Deferred production may not retain an eager argmax implementation.";
    EXPECT_EQ(
        deferred_impl.find("mtp_publication_diagnostics"),
        std::string::npos)
        << "Deferred production diagnostics may not introduce a D2H token read.";
}
