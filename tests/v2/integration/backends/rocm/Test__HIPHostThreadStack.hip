/**
 * @file Test__HIPHostThreadStack.hip
 * @brief Prove retained diagnostic/timeline graphs admit realistic client ELF TLS.
 *
 * glibc puts initial ELF TLS inside each pthread stack mapping. HIP previously
 * requested a fixed 256 KiB host-worker stack, which can be smaller than a
 * valid client's TLS and abort graph instantiation before a worker runs. Force
 * the native hostcall ABI used by diagnostic-enabled production kernels:
 * timeline commands alone do not create that worker under direct dispatch.
 * Retain that condition independently of the linked library inventory, then
 * replay the graph with the production channel's exact timeline dependencies.
 * This model-free native ABI fixture alone owns its direct HIP allocations;
 * production storage remains PhysicalMemoryAuthority/TransferEngine owned.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
// A valid client's TLS must not be constrained by a vendor's queue-stack cap.
// Volatile touches retain real ELF TLS, rather than a test-only heap substitute.
alignas(64) thread_local volatile unsigned char client_tls[384 * 1024];

/**
 * @brief Retain a native hostcall binding without a callback on successful work.
 * @tparam Word Exact production timeline width.
 * @param value Device-owned word published by the graph's producer.
 * @param expected Immutable value checked by the diagnostic branch.
 *
 * The cold printf branch requires the same hidden hostcall buffer as device
 * assertions. HIP must admit its listener when materializing the executable
 * even though correct replays never print. A plain empty/write/wait graph would
 * pass on the broken runtime and would not be a meaningful negative control.
 */
template<class Word>
__global__ void validateTimelineWord(const Word* value, Word expected) {
    if (*value != expected)
        printf("HIP host-stack timeline mismatch: %llx\n",
               static_cast<unsigned long long>(*value));
}

/**
 * @brief Fail at the exact rejected native ABI operation.
 * @param status Native HIP return value.
 * @param operation Lifecycle operation retained in the failure diagnostic.
 */
void requireHip(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

/**
 * @brief Own one retained timeline graph and its persistent native bindings.
 * @tparam Word Exact captured timeline width, either uint32_t or uint64_t.
 */
template<class Word>
class TimelineHostStackProof final {
public:
    /**
     * @brief Allocate before capture and record a device-owned write/wait DAG.
     * @throws std::runtime_error After retiring any partial setup on rejection.
     */
    TimelineHostStackProof() {
        static_assert(std::is_same_v<Word, std::uint32_t> ||
                      std::is_same_v<Word, std::uint64_t>);
        try {
            requireHip(hipSetDevice(0), "select real fixture device");
            for (auto& stream : streams_)
                requireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create explicit stream");
            requireHip(hipEventCreateWithFlags(&completion_, hipEventDisableTiming), "create completion event");
            requireHip(hipMalloc(reinterpret_cast<void**>(&value_), sizeof(Word)), "allocate timeline word");
            requireHip(hipMemsetAsync(value_, 0, sizeof(Word), streams_[0]), "initialize timeline word");
            requireHip(hipStreamBeginCapture(streams_[0], hipStreamCaptureModeGlobal), "begin native capture");
            if constexpr (std::is_same_v<Word, std::uint32_t>) {
                requireHip(hipStreamWriteValue32(streams_[0], value_, expected_, 0), "capture 32-bit producer");
                requireHip(hipStreamWaitValue32(streams_[0], value_, expected_, hipStreamWaitValueEq), "capture 32-bit consumer");
            } else {
                requireHip(hipStreamWriteValue64(streams_[0], value_, expected_, 0), "capture 64-bit producer");
                requireHip(hipStreamWaitValue64(streams_[0], value_, expected_, hipStreamWaitValueEq), "capture 64-bit consumer");
            }
            hipLaunchKernelGGL(HIP_KERNEL_NAME(validateTimelineWord<Word>),
                               dim3(1), dim3(1), 0, streams_[0], value_, expected_);
            requireHip(hipGetLastError(), "capture native hostcall diagnostic binding");
            requireHip(hipStreamEndCapture(streams_[0], &graph_), "finish native capture");
            // The diagnostic kernel requires a native hostcall listener here.
            // Its TLS admission must succeed before any graph replay can run.
            requireHip(hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0), "instantiate captured timeline");
        } catch (...) {
            retire();
            throw;
        }
    }

    /** @brief Retire the completed executable before its persistent word/streams. */
    ~TimelineHostStackProof() { retire(); }
    TimelineHostStackProof(const TimelineHostStackProof&) = delete;
    TimelineHostStackProof& operator=(const TimelineHostStackProof&) = delete;

    /**
     * @brief Replay without per-iteration host waits, then check the terminal word.
     *
     * Alternating launch streams consume the exact predecessor event. The only
     * host observation is one final result; no eager/serial replay replaces the
     * retained graph, and the host never writes the live timeline after setup.
     */
    void verify() {
        constexpr unsigned replays = 20;
        for (unsigned replay = 0; replay < replays; ++replay) {
            const auto stream = streams_[replay % streams_.size()];
            if (replay != 0)
                requireHip(hipStreamWaitEvent(stream, completion_, 0), "consume exact predecessor event");
            requireHip(hipGraphLaunch(executable_, stream), "replay retained timeline");
            requireHip(hipEventRecord(completion_, stream), "publish replay completion");
        }
        const auto terminal_stream = streams_[(replays - 1) % streams_.size()];
        Word result{};
        requireHip(hipMemcpyAsync(&result, value_, sizeof(Word), hipMemcpyDeviceToHost, terminal_stream), "read terminal word");
        requireHip(hipEventRecord(completion_, terminal_stream), "publish terminal result");
        requireHip(hipEventSynchronize(completion_), "observe terminal result");
        EXPECT_EQ(result, expected_);
    }

private:
    /** @brief Release native infrastructure, including a rejected partial setup. */
    void retire() noexcept {
        if (executable_) (void)hipGraphExecDestroy(executable_);
        if (graph_) (void)hipGraphDestroy(graph_);
        if (value_) (void)hipFree(value_);
        if (completion_) (void)hipEventDestroy(completion_);
        for (auto stream : streams_) if (stream) (void)hipStreamDestroy(stream);
    }

    static constexpr Word expected_ = std::is_same_v<Word, std::uint32_t>
        ? static_cast<Word>(0x10204080u) : static_cast<Word>(0x1020408010204080ull);
    std::array<hipStream_t, 2> streams_{};
    hipEvent_t completion_{};
    hipGraph_t graph_{};
    hipGraphExec_t executable_{};
    Word* value_{};
};

/** @brief Admit a real device and keep the oversized static TLS payload live. */
void admitClient() {
    client_tls[0] = 1;
    client_tls[sizeof(client_tls) - 1] = 2;
    int devices{};
    requireHip(hipGetDeviceCount(&devices), "enumerate real devices");
    if (devices <= 0) throw std::runtime_error("Explicit ROCm regression requires a real device");
}

/** @test A valid client with large ELF TLS retains captured 32-bit timeline work. */
TEST(HIPHostThreadStack, Captured32BitTimelineAdmitsClientTls) {
    admitClient();
    TimelineHostStackProof<std::uint32_t> proof;
    proof.verify();
}

/** @test The same admission contract covers 64-bit production channel epochs. */
TEST(HIPHostThreadStack, Captured64BitTimelineAdmitsClientTls) {
    admitClient();
    TimelineHostStackProof<std::uint64_t> proof;
    proof.verify();
}
} // namespace
