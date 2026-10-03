/**
 * @file Test__RCCLHostTransportStorage.cpp
 * @brief Prove native same-process SHM ownership through real captured collectives.
 *
 * File-backed USERPTR aliases can be invalidated by Linux compaction even while
 * a collective is using them. A single-process clique needs one native pinned
 * owner instead. The exact configured RCCL DSO reports its actual connected
 * descriptors; correct arithmetic alone cannot satisfy the storage assertion.
 * Retained graphs, reversed physical participant order, changing inputs, guard
 * tails and both retirement orders exercise the optimized native path. Native
 * allocations/readbacks belong only to this bounded infrastructure fixture.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "backends/rocm/ROCmRuntimeStartup.h"
#include "collective/backends/RCCLDynamicLoader.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    namespace native = llaminar2::rccl_dynamic;

    /** @brief Supported independent retirement orders, not a second transport mode. */
    enum class RetirementOrder { CreatorFirst, PeerFirst };

    /**
     * @brief Surface a native HIP failure while its owning fixture can still retire.
     * @param result Immediately preceding operation's status.
     * @param boundary Stable lifecycle label for the diagnostic.
     * @throws std::runtime_error on a failed native operation.
     */
    void requireHip(hipError_t result, const char* boundary)
    {
        if (result != hipSuccess)
            throw std::runtime_error(std::string(boundary) + ": " + hipGetErrorString(result));
    }

    /**
     * @brief Surface the exact configured vendor dependency's failure.
     * @param result Immediately preceding RCCL result.
     * @param boundary Setup, capture, query or teardown boundary.
     * @throws std::runtime_error on a failed vendor operation.
     */
    void requireRccl(native::ncclResult_t result, const char* boundary)
    {
        if (result != native::ncclSuccess)
            throw std::runtime_error(std::string(boundary) + ": " + native::ncclGetErrorString(result));
    }

    /** @brief Own one initialized clique and all captured pointer/stream lifetimes. */
    class CapturedHostTransport final
    {
    public:
        /**
         * @brief Construct reversed physical endpoints with persistent small buffers.
         * @param count Clique width; the test admits only available real devices.
         * @param order Normal communicator retirement ordering to exercise.
         * @throws std::runtime_error after retiring partial setup on failure.
         */
        CapturedHostTransport(unsigned count, RetirementOrder order)
            : endpoints_(count), communicators_(count, nullptr), order_(order)
        {
            try
            {
                llaminar2::requireROCmRuntimeStartup();
                if (!native::load())
                    throw std::runtime_error(native::getLastError());
                std::vector<int> ordinals(count);
                for (unsigned rank = 0; rank < count; ++rank)
                {
                    auto& endpoint = endpoints_[rank];
                    endpoint.device = static_cast<int>(count - rank - 1);
                    ordinals[rank] = endpoint.device;
                    requireHip(hipSetDevice(endpoint.device), "select physical endpoint");
                    requireHip(hipStreamCreateWithFlags(&endpoint.stream, hipStreamNonBlocking), "create exact stream");
                    requireHip(hipEventCreateWithFlags(&endpoint.terminal, hipEventDisableTiming), "create terminal event");
                    requireHip(hipMalloc(reinterpret_cast<void**>(&endpoint.input), kCapacity * sizeof(float)), "allocate input");
                    requireHip(hipMalloc(reinterpret_cast<void**>(&endpoint.output), kCapacity * sizeof(float)), "allocate output");
                }
                requireRccl(native::ncclCommInitAll(communicators_.data(), count, ordinals.data()), "initialize native clique");
                prepare(0);
                enqueue();
                observe(0); // Materialize native connectors before inspecting ownership.
            }
            catch (...)
            {
                release();
                throw;
            }
        }

        /** @brief Retire graphs before their bound streams, buffers and communicators. */
        ~CapturedHostTransport() { release(); }
        CapturedHostTransport(const CapturedHostTransport&) = delete;
        CapturedHostTransport& operator=(const CapturedHostTransport&) = delete;

        /**
         * @brief Authenticate the actual connected descriptors in the selected DSO.
         *
         * Counts are endpoint observations, not unique physical-byte accounting.
         * This fixture intentionally admits the no-P2P SHM contract; it must not
         * pass by silently selecting another collective or unregistered storage.
         */
        void verifyStorage()
        {
            for (unsigned rank = 0; rank < endpoints_.size(); ++rank)
            {
                SCOPED_TRACE("participant=" + std::to_string(rank));
                llaminar2::RCCLHostTransportStorage receipt;
                requireRccl(native::hostTransportStorage(communicators_[rank], receipt), "read actual native storage");
                ASSERT_TRUE(receipt.valid());
                EXPECT_GT(receipt.same_process_pinned_connections, 0u);
                EXPECT_GT(receipt.same_process_pinned_mapped_bytes, 0u);
                EXPECT_EQ(receipt.shared_process_registered_connections, 0u)
                    << "Same-process transport still registers migratable file-backed USERPTR aliases";
                EXPECT_EQ(receipt.shared_process_registered_mapped_bytes, 0u);
                EXPECT_EQ(receipt.cu_mem_connections, 0u);
            }
        }

        /**
         * @brief Record once, then replay twenty changed inputs without rebuilding.
         *
         * Every participant is submitted before any terminal observation. Poison
         * the output on its exact launch stream to catch omitted/stale replay and
         * require the unused allocation tail to remain untouched.
         */
        void verifyCapturedReplays()
        {
            for (auto& endpoint : endpoints_)
            {
                requireHip(hipSetDevice(endpoint.device), "select capture endpoint");
                requireHip(hipStreamBeginCapture(endpoint.stream, hipStreamCaptureModeGlobal), "begin native capture");
            }
            enqueue();
            for (auto& endpoint : endpoints_)
            {
                requireHip(hipSetDevice(endpoint.device), "select graph endpoint");
                requireHip(hipStreamEndCapture(endpoint.stream, &endpoint.graph), "end native capture");
                requireHip(hipGraphInstantiate(&endpoint.executable, endpoint.graph, nullptr, nullptr, 0), "instantiate native collective");
            }
            for (unsigned replay = 1; replay <= 20; ++replay)
            {
                prepare(replay);
                for (auto& endpoint : endpoints_)
                {
                    requireHip(hipSetDevice(endpoint.device), "select replay endpoint");
                    requireHip(hipGraphLaunch(endpoint.executable, endpoint.stream), "launch retained native collective");
                }
                observe(replay);
            }
            verifyStorage();
        }

        /**
         * @brief Prove successful native retirement in the declared owner order.
         *
         * Teardown is part of the regression, not best-effort destructor cleanup.
         * Withdraw each handle only after its native release succeeds so partial
         * failure remains available to the owning fixture's cleanup boundary.
         * @throws std::runtime_error on any graph, communicator or HIP release error.
         */
        void retire()
        {
            for (auto& endpoint : endpoints_)
            {
                requireHip(hipSetDevice(endpoint.device), "select graph retirement endpoint");
                requireHip(hipGraphExecDestroy(endpoint.executable), "retire native executable");
                endpoint.executable = nullptr;
                requireHip(hipGraphDestroy(endpoint.graph), "retire native graph");
                endpoint.graph = nullptr;
            }
            for (unsigned index = 0; index < communicators_.size(); ++index)
            {
                const auto rank = order_ == RetirementOrder::CreatorFirst ? index :
                    static_cast<unsigned>(communicators_.size()) - index - 1;
                requireHip(hipSetDevice(endpoints_[rank].device), "select communicator retirement endpoint");
                requireRccl(native::ncclCommDestroy(communicators_[rank]), "retire actual native owner/peer");
                communicators_[rank] = nullptr;
            }
            for (auto& endpoint : endpoints_)
            {
                requireHip(hipSetDevice(endpoint.device), "select backing retirement endpoint");
                requireHip(hipFree(endpoint.input), "release fixture input");
                endpoint.input = nullptr;
                requireHip(hipFree(endpoint.output), "release fixture output");
                endpoint.output = nullptr;
                requireHip(hipEventDestroy(endpoint.terminal), "release terminal event");
                endpoint.terminal = nullptr;
                requireHip(hipStreamDestroy(endpoint.stream), "release exact stream");
                endpoint.stream = nullptr;
            }
        }

    private:
        /** @brief One physical endpoint's persistent capture resources. */
        struct Endpoint
        {
            int device = 0;
            hipStream_t stream = nullptr;
            hipEvent_t terminal = nullptr;
            hipGraph_t graph = nullptr;
            hipGraphExec_t executable = nullptr;
            float* input = nullptr;
            float* output = nullptr;
            std::array<float, 256> sent{};
            std::array<float, 256> received{};
        };

        /** @brief Publish new fixture inputs before each retained replay. */
        void prepare(unsigned replay)
        {
            for (unsigned rank = 0; rank < endpoints_.size(); ++rank)
            {
                auto& endpoint = endpoints_[rank];
                requireHip(hipSetDevice(endpoint.device), "select input endpoint");
                for (unsigned column = 0; column < kCapacity; ++column)
                    endpoint.sent[column] = static_cast<float>(rank + 1 + column + replay);
                endpoint.received.fill(kGuard);
                requireHip(hipMemcpyAsync(endpoint.input, endpoint.sent.data(), sizeof(endpoint.sent), hipMemcpyHostToDevice, endpoint.stream), "publish fixture input");
                requireHip(hipMemcpyAsync(endpoint.output, endpoint.received.data(), sizeof(endpoint.received), hipMemcpyHostToDevice, endpoint.stream), "poison complete output");
            }
        }

        /** @brief Enqueue only the live extent through the ordinary native API. */
        void enqueue()
        {
            requireRccl(native::ncclGroupStart(), "begin native group");
            for (unsigned rank = 0; rank < endpoints_.size(); ++rank)
            {
                auto& endpoint = endpoints_[rank];
                requireHip(hipSetDevice(endpoint.device), "select collective endpoint");
                requireRccl(native::ncclAllReduce(endpoint.input, endpoint.output, kLive,
                    native::ncclFloat32, native::ncclSum, communicators_[rank], endpoint.stream), "enqueue live native allreduce");
            }
            requireRccl(native::ncclGroupEnd(), "end native group");
        }

        /** @brief Verify all real results at the fixture's sole terminal boundary. */
        void observe(unsigned replay)
        {
            for (auto& endpoint : endpoints_)
            {
                requireHip(hipSetDevice(endpoint.device), "select observation endpoint");
                requireHip(hipMemcpyAsync(endpoint.received.data(), endpoint.output,
                    sizeof(endpoint.received), hipMemcpyDeviceToHost, endpoint.stream), "read terminal result");
                requireHip(hipEventRecord(endpoint.terminal, endpoint.stream), "publish terminal event");
            }
            for (auto& endpoint : endpoints_)
            {
                requireHip(hipSetDevice(endpoint.device), "select terminal endpoint");
                requireHip(hipEventSynchronize(endpoint.terminal), "observe completed collective");
                for (unsigned column = 0; column < kCapacity; ++column)
                {
                    const auto ranks = static_cast<unsigned>(endpoints_.size());
                    const float expected = column < kLive ?
                        static_cast<float>(ranks * (ranks + 1) / 2 + ranks * (column + replay)) : kGuard;
                    EXPECT_EQ(endpoint.received[column], expected)
                        << "device=" << endpoint.device << " replay=" << replay << " column=" << column;
                }
            }
        }

        /** @brief Retire partial or completed setup without rebinding a captured owner. */
        void release() noexcept
        {
            for (auto& endpoint : endpoints_)
            {
                (void)hipSetDevice(endpoint.device);
                if (endpoint.executable) (void)hipGraphExecDestroy(endpoint.executable);
                if (endpoint.graph) (void)hipGraphDestroy(endpoint.graph);
            }
            // Every completed operation was observed before retirement. Vary the
            // communicator order so creator/alias ownership cannot depend on it.
            for (unsigned index = 0; index < communicators_.size(); ++index)
            {
                const auto rank = order_ == RetirementOrder::CreatorFirst ? index :
                    static_cast<unsigned>(communicators_.size()) - index - 1;
                if (communicators_[rank])
                {
                    (void)hipSetDevice(endpoints_[rank].device);
                    (void)native::ncclCommDestroy(communicators_[rank]);
                    communicators_[rank] = nullptr;
                }
            }
            for (auto& endpoint : endpoints_)
            {
                (void)hipSetDevice(endpoint.device);
                if (endpoint.input) (void)hipFree(endpoint.input);
                if (endpoint.output) (void)hipFree(endpoint.output);
                if (endpoint.terminal) (void)hipEventDestroy(endpoint.terminal);
                if (endpoint.stream) (void)hipStreamDestroy(endpoint.stream);
                endpoint = {};
            }
        }

        static constexpr unsigned kCapacity = 256;
        static constexpr unsigned kLive = 127;
        static constexpr float kGuard = -9999.0f;
        std::vector<Endpoint> endpoints_;
        std::vector<native::ncclComm_t> communicators_;
        RetirementOrder order_;
    };

    /** @brief Real clique widths and independent creator/peer lifetime order. */
    class RCCLHostTransportStorageTest : public ::testing::TestWithParam<
        std::tuple<unsigned, RetirementOrder>> {};

    TEST_P(RCCLHostTransportStorageTest, NativePinnedOwnershipSurvivesCapturedReplay)
    {
        const auto [width, order] = GetParam();
        llaminar2::requireROCmRuntimeStartup();
        int count = 0;
        requireHip(hipGetDeviceCount(&count), "enumerate fixture hardware");
        ASSERT_GE(count, static_cast<int>(width)) << "Required real ROCm clique is unavailable";
        CapturedHostTransport proof(width, order);
        proof.verifyStorage();
        proof.verifyCapturedReplays();
        proof.retire();
    }

    INSTANTIATE_TEST_SUITE_P(NativeSharedHost, RCCLHostTransportStorageTest,
        ::testing::Combine(::testing::Values(2u, 4u),
            ::testing::Values(RetirementOrder::CreatorFirst, RetirementOrder::PeerFirst)));
}

/** @brief Admit the explicit no-P2P storage contract before native runtime setup. */
int main(int argc, char** argv)
{
    if (setenv("NCCL_P2P_DISABLE", "1", 1) != 0) return 2;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
