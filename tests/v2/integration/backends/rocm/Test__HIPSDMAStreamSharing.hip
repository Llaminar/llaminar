/**
 * @file Test__HIPSDMAStreamSharing.hip
 * @brief Prove NoCU copies remain byte-exact when streams outnumber DMA engines.
 *
 * Expert maintenance retains independent streams so transfers need not occupy
 * inference CUs. ROCclr must share its finite SDMA engines once exclusive
 * assignments fill; a completion event is not evidence that a rejected copy
 * moved bytes. This native-ABI regression checks all destinations and guards,
 * including mapped H2D/D2H round trips, device copies and retained graph replay.
 * Only this model-free infrastructure fixture owns direct native allocations;
 * production storage remains PhysicalMemoryAuthority/TransferEngine owned.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
/** @brief Native endpoints whose DMA admission must obey the same contract. */
enum class CopyEndpoints { MappedRoundTrip, DeviceToDevice };
/** @brief Explicitly distinguish maintenance submission from captured replay. */
enum class Submission { Async, Captured };

/**
 * @brief Reject the exact native operation that failed instead of trying a blit.
 * @param status Native HIP return value.
 * @param operation Lifecycle operation named in the failure diagnostic.
 */
void requireHip(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

/**
 * @brief Own persistent, independent DMA lanes and one exact terminal join.
 *
 * Thirty-two owners deliberately exceed the physical engine counts on the
 * supported cards. Different live byte counts exercise tails; guarded storage
 * is retained but never used as the maintenance-copy extent. An empty lane
 * submits no payload. There is no compute-copy escape hatch or default stream.
 */
class SDMAStreamSharingProof final {
public:
    /**
     * @brief Prepare every binding and optional captured executable before use.
     * @param endpoints Mapped round trips or same-device NoCU copies.
     * @param submission Ordinary async maintenance or retained native graphs.
     * @throws std::runtime_error After retiring a rejected partial setup.
     */
    SDMAStreamSharingProof(CopyEndpoints endpoints, Submission submission)
        : endpoints_(endpoints), submission_(submission) {
        try {
            requireHip(hipSetDevice(0), "select real DMA fixture device");
            for (auto& lane : lanes_) {
                requireHip(hipStreamCreateWithFlags(&lane.stream, hipStreamNonBlocking),
                           "create exact DMA owner stream");
                requireHip(hipEventCreateWithFlags(&lane.completion, hipEventDisableTiming),
                           "create exact lane completion");
            }
            requireHip(hipEventCreateWithFlags(&prepared_, hipEventDisableTiming),
                       "create persistent preparation frontier");
            requireHip(hipEventCreateWithFlags(&terminal_, hipEventDisableTiming),
                       "create terminal observation frontier");
            requireHip(hipHostMalloc(reinterpret_cast<void**>(&input_), storageBytes,
                                     hipHostMallocMapped), "allocate persistent mapped input");
            requireHip(hipHostMalloc(reinterpret_cast<void**>(&output_), storageBytes,
                                     hipHostMallocMapped), "allocate persistent mapped output");
            requireHip(hipHostGetDevicePointer(reinterpret_cast<void**>(&inputAlias_), input_, 0),
                       "bind exact input mapping");
            requireHip(hipHostGetDevicePointer(reinterpret_cast<void**>(&outputAlias_), output_, 0),
                       "bind exact output mapping");
            requireHip(hipMalloc(reinterpret_cast<void**>(&destination_), storageBytes),
                       "allocate guarded device destinations");
            std::fill_n(input_, storageBytes, sourceGuard);
            std::fill_n(output_, storageBytes, destinationGuard);
            for (std::size_t lane = 0; lane < laneCount; ++lane)
                for (std::size_t byte = 0; byte < liveBytes(lane); ++byte)
                    input_[payloadOffset(lane) + byte] = payloadByte(lane, byte);

            const auto setup = lanes_.front().stream;
            requireHip(hipMemsetAsync(destination_, destinationGuard, storageBytes, setup),
                       "initialize persistent destination guards");
            if (endpoints_ == CopyEndpoints::DeviceToDevice) {
                requireHip(hipMalloc(reinterpret_cast<void**>(&deviceInput_), storageBytes),
                           "allocate persistent device source");
                requireHip(hipMemcpyAsync(deviceInput_, input_, storageBytes,
                                          hipMemcpyHostToDevice, setup),
                           "prepare immutable device source");
            }
            requireHip(hipEventRecord(prepared_, setup), "publish exact preparation frontier");
            if (submission_ == Submission::Captured)
                for (std::size_t lane = 0; lane < laneCount; ++lane) capture(lane);
        } catch (...) {
            retire();
            throw;
        }
    }

    /** @brief Retire completed executables before their streams and bindings. */
    ~SDMAStreamSharingProof() { retire(); }
    SDMAStreamSharingProof(const SDMAStreamSharingProof&) = delete;
    SDMAStreamSharingProof& operator=(const SDMAStreamSharingProof&) = delete;

    /**
     * @brief Submit all owners without host joins and verify the actual bytes.
     *
     * Each FIFO orders its retained replays independently. Only the terminal
     * observer consumes all lane events; no per-copy or per-lane host wait
     * hides an allocator failure or turns maintenance into serial execution.
     */
    void verify() {
        for (auto& lane : lanes_)
            requireHip(hipStreamWaitEvent(lane.stream, prepared_, 0),
                       "consume exact source preparation");
        constexpr unsigned replays = 20;
        for (unsigned replay = 0; replay < replays; ++replay) {
            // Rotate submission order without changing stream ownership or
            // embedded pointers. No graph is rebuilt to accommodate a replay.
            for (std::size_t index = 0; index < laneCount; ++index) {
                const auto lane = (index + replay) % laneCount;
                if (submission_ == Submission::Captured)
                    requireHip(hipGraphLaunch(lanes_[lane].executable, lanes_[lane].stream),
                               "replay retained NoCU copies");
                else
                    enqueue(lane);
            }
        }
        for (auto& lane : lanes_)
            requireHip(hipEventRecord(lane.completion, lane.stream), "publish lane completion");
        const auto observer = lanes_.front().stream;
        for (const auto& lane : lanes_)
            requireHip(hipStreamWaitEvent(observer, lane.completion, 0),
                       "join exact completed DMA owners");
        if (endpoints_ == CopyEndpoints::DeviceToDevice)
            requireHip(hipMemcpyAsync(output_, destination_, storageBytes,
                                      hipMemcpyDeviceToHost, observer),
                       "observe all live destinations and guards");
        requireHip(hipEventRecord(terminal_, observer), "publish terminal result");
        requireHip(hipEventSynchronize(terminal_), "observe terminal result");

        for (std::size_t lane = 0; lane < laneCount; ++lane) {
            SCOPED_TRACE("DMA owner " + std::to_string(lane));
            for (std::size_t byte = 0; byte < stride; ++byte) {
                const bool live = byte >= guardBytes && byte - guardBytes < liveBytes(lane);
                const auto expected = live ? payloadByte(lane, byte - guardBytes) : destinationGuard;
                ASSERT_EQ(output_[lane * stride + byte], expected)
                    << "destination byte " << byte << "; completed events alone do not prove a copy";
            }
        }
    }

private:
    /** @brief One native FIFO, exact completion and retained capture binding. */
    struct Lane {
        hipStream_t stream{};
        hipEvent_t completion{};
        hipGraph_t graph{};
        hipGraphExec_t executable{};
    };
    static constexpr std::size_t laneCount = 32;
    static constexpr std::size_t guardBytes = 64;
    static constexpr std::size_t stride = 4096 + 2 * guardBytes;
    static constexpr std::size_t storageBytes = laneCount * stride;
    static constexpr std::uint8_t sourceGuard = 0x39;
    static constexpr std::uint8_t destinationGuard = 0xd3;

    /** @brief Include empty, odd tails, sub-page and full-page live payloads. */
    static constexpr std::size_t liveBytes(std::size_t lane) {
        constexpr std::array<std::size_t, 8> lengths{0, 1, 7, 31, 255, 1024, 4095, 4096};
        return lengths[lane % lengths.size()];
    }
    /** @brief Locate the live bytes without communicating either guard region. */
    static constexpr std::size_t payloadOffset(std::size_t lane) {
        return lane * stride + guardBytes;
    }
    /** @brief Keep each owner's source distinguishable from every empty guard. */
    static constexpr std::uint8_t payloadByte(std::size_t lane, std::size_t byte) {
        return static_cast<std::uint8_t>(lane * 17 + byte * 131 + 7);
    }
    /**
     * @brief Enqueue only required bytes using SDMA, never a compute substitute.
     * @param lane Exact stream owner and persistent payload geometry.
     */
    void enqueue(std::size_t lane) {
        const auto bytes = liveBytes(lane);
        if (bytes == 0) return;
        const auto offset = payloadOffset(lane);
        const auto stream = lanes_[lane].stream;
        const auto* source = endpoints_ == CopyEndpoints::MappedRoundTrip ? inputAlias_ : deviceInput_;
        requireHip(hipMemcpyAsync(destination_ + offset, source + offset, bytes,
                                  hipMemcpyDeviceToDeviceNoCU, stream), "submit required NoCU payload");
        if (endpoints_ == CopyEndpoints::MappedRoundTrip)
            requireHip(hipMemcpyAsync(outputAlias_ + offset, destination_ + offset, bytes,
                                      hipMemcpyDeviceToDeviceNoCU, stream), "submit required NoCU echo");
    }
    /** @brief Retain one independent graph with the owner's exact bindings. */
    void capture(std::size_t lane) {
        auto& owner = lanes_[lane];
        requireHip(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeGlobal), "begin NoCU capture");
        enqueue(lane);
        requireHip(hipStreamEndCapture(owner.stream, &owner.graph), "finish NoCU capture");
        requireHip(hipGraphInstantiate(&owner.executable, owner.graph, nullptr, nullptr, 0),
                   "instantiate independent DMA graph");
    }
    /** @brief Retire infrastructure, also handling a rejected partial setup. */
    void retire() noexcept {
        for (auto& lane : lanes_) {
            if (lane.executable) (void)hipGraphExecDestroy(lane.executable);
            if (lane.graph) (void)hipGraphDestroy(lane.graph);
        }
        if (deviceInput_) (void)hipFree(deviceInput_);
        if (destination_) (void)hipFree(destination_);
        if (input_) (void)hipHostFree(input_);
        if (output_) (void)hipHostFree(output_);
        if (prepared_) (void)hipEventDestroy(prepared_);
        if (terminal_) (void)hipEventDestroy(terminal_);
        for (auto& lane : lanes_) {
            if (lane.completion) (void)hipEventDestroy(lane.completion);
            if (lane.stream) (void)hipStreamDestroy(lane.stream);
        }
    }

    CopyEndpoints endpoints_;
    Submission submission_;
    std::array<Lane, laneCount> lanes_{};
    hipEvent_t prepared_{};
    hipEvent_t terminal_{};
    std::uint8_t* input_{};
    std::uint8_t* output_{};
    std::uint8_t* inputAlias_{};
    std::uint8_t* outputAlias_{};
    std::uint8_t* destination_{};
    std::uint8_t* deviceInput_{};
};

/** @test Background mapped traffic exceeds physical engine counts safely. */
TEST(HIPSDMAStreamSharing, AsyncMappedRoundTripsRemainByteExactAcross32Owners) {
    SDMAStreamSharingProof proof(CopyEndpoints::MappedRoundTrip, Submission::Async);
    proof.verify();
}
/** @test The same live extents survive retained multi-owner graph replay. */
TEST(HIPSDMAStreamSharing, CapturedMappedRoundTripsRemainByteExactAcross32Owners) {
    SDMAStreamSharingProof proof(CopyEndpoints::MappedRoundTrip, Submission::Captured);
    proof.verify();
}
/** @test Finite-engine sharing also covers captured same-device copies. */
TEST(HIPSDMAStreamSharing, CapturedDeviceCopiesRemainByteExactAcross32Owners) {
    SDMAStreamSharingProof proof(CopyEndpoints::DeviceToDevice, Submission::Captured);
    proof.verify();
}
} // namespace
