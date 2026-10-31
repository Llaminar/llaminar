/**
 * @file Test__HIPProjectionCaptureLifetime.cpp
 * @brief Native replay of context-owned projection resources over whole HIP captures.
 *
 * Four branches in two separate fork/join stages borrow the production resource owner inside
 * one actual parent capture. Exact-stream fills preserve byte-visible ordering;
 * repeated poisoned replay, foreign-root rejection and aborted-capture cleanup
 * exercise native membership without adding a kernel or changing arithmetic.
 * All native operations execute on the owning context worker. Runtime reset is
 * deliberately left to the exclusive-retirement integration gates.
 */
#include "backends/GPUDeviceContextPool.h"
#include "backends/rocm/AMDDeviceContext.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <stdexcept>
#include <string>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Native proof failures cannot become another recording path. */
        void checked(hipError_t error)
        {
            if (error != hipSuccess)
                throw std::runtime_error(std::string("HIP projection lifetime replay: ") + hipGetErrorString(error));
        }
        /** @brief Own only test operands and parent streams; the context owns all side handles. */
        struct NativeOperands
        {
            hipStream_t root = nullptr, foreign = nullptr;
            unsigned char *output = nullptr;
            /** @brief Release after graph destruction and explicit terminal replay observation. */
            ~NativeOperands()
            {
                if (output && hipFree(output) != hipSuccess) std::terminate();
                if (foreign && hipStreamDestroy(foreign) != hipSuccess) std::terminate();
                if (root && hipStreamDestroy(root) != hipSuccess) std::terminate();
            }
        };
        /**
         * @brief Record the unchanged native fork/work/completion/join sequence under a borrowed capture lease.
         * @param context Exact worker context. @param root Parent recording stream.
         * @param output Persistent disjoint test bytes. @param branch Existing side-stream slot.
         * @param expected Stable cold-prepared resource identity.
         */
        void recordBranch(AMDDeviceContext &context, hipStream_t root,
            unsigned char *output, int branch, const GPUProjectionResourceView &expected)
        {
            auto submission = context.acquireProjectionSubmission(
                static_cast<void *>(root), GPUProjectionSubmissionKind::Eager, true);
            const auto &resources = submission.resources();
            EXPECT_EQ(resources.streams, expected.streams);
            EXPECT_EQ(resources.completion, expected.completion);
            EXPECT_EQ(resources.quant_ready, expected.quant_ready);
            const auto side = static_cast<hipStream_t>(resources.streams[branch]);
            const auto fork = static_cast<hipEvent_t>(resources.quant_ready);
            const auto completion = static_cast<hipEvent_t>(resources.completion[branch]);
            checked(hipEventRecord(fork, root));
            checked(hipStreamWaitEvent(side, fork, 0));
            checked(hipMemsetAsync(output + branch * 16, 0x11 * (branch + 1), 16, side));
            checked(hipEventRecord(completion, side));
            checked(hipStreamWaitEvent(root, completion, 0));
        }
    }

    /** @test Production whole-capture ownership survives separate stages, replay and explicit abort. */
    TEST(HIPProjectionCaptureLifetime, WholeCaptureOwnsForkJoinResourcesThroughEndAndAbort)
    {
        initializeROCmHardwareMetadata();
        const auto &inventory = rocmHardwareMetadata();
        if (inventory.size() == 0)
            GTEST_SKIP() << "UNQUALIFIED: HIP projection lifetime replay has no visible hardware";
        for (int ordinal = 0; ordinal < static_cast<int>(inventory.size()); ++ordinal)
        {
            SCOPED_TRACE(ordinal);
            auto context_owner = GPUDeviceContextPool::instance().retainAMDContext(ordinal);
            auto &context = static_cast<AMDDeviceContext &>(*context_owner);
            context.submitAndWait([&]
            {
                NativeOperands operands;
                checked(hipStreamCreateWithFlags(&operands.root, hipStreamNonBlocking));
                checked(hipStreamCreateWithFlags(&operands.foreign, hipStreamNonBlocking));
                checked(hipMalloc(reinterpret_cast<void **>(&operands.output), 64));
                context.prepareProjectionResources();
                const auto expected = [&]
                {
                    auto submission = context.acquireProjectionSubmission(
                        operands.root, GPUProjectionSubmissionKind::Eager);
                    return submission.resources();
                }();
                ASSERT_EQ(expected.count, 4);
                ASSERT_NE(expected.quant_ready, nullptr);
                for (int slot = 0; slot < expected.count; ++slot)
                {
                    ASSERT_NE(expected.streams[slot], nullptr);
                    ASSERT_NE(expected.completion[slot], nullptr);
                }
                // Maximum backing storage is not another native population.
                // Only the original four streams and five events may exist.
                for (std::size_t slot = expected.count; slot < expected.streams.size(); ++slot)
                {
                    EXPECT_EQ(expected.streams[slot], nullptr);
                    EXPECT_EQ(expected.completion[slot], nullptr);
                }
                {
                HIPGraphCapture graph(operands.root, ordinal, context);
                ASSERT_TRUE(graph.beginCapture());
                recordBranch(context, operands.root, operands.output, 0, expected);
                recordBranch(context, operands.root, operands.output, 1, expected);
                // The first stage's local borrower is gone, but the native
                // parent still owns all shared side-stream capture membership.
                EXPECT_THROW((void)context.acquireProjectionSubmission(
                    operands.foreign, GPUProjectionSubmissionKind::Eager, true), std::logic_error);
                EXPECT_THROW(context.resetAuxiliaryStreams(), std::logic_error);
                EXPECT_THROW(context.beginProjectionResourcesRetirement(), std::logic_error);
                recordBranch(context, operands.root, operands.output, 2, expected);
                recordBranch(context, operands.root, operands.output, 3, expected);
                ASSERT_TRUE(graph.endCapture());
                EXPECT_NO_THROW(context.assertProjectionResourcesRetirable());
                ASSERT_TRUE(graph.instantiate());
                for (int repeat = 0; repeat < 4; ++repeat)
                {
                    if (repeat == 2)
                    {
                        // Public named-stream reset must preserve the native
                        // handles embedded in this retained executable.
                        EXPECT_NO_THROW(context.resetAuxiliaryStreams());
                        auto submission = context.acquireProjectionSubmission(
                            operands.root, GPUProjectionSubmissionKind::Eager);
                        const auto &resources = submission.resources();
                        EXPECT_EQ(resources.count, expected.count);
                        EXPECT_EQ(resources.streams, expected.streams);
                        EXPECT_EQ(resources.completion, expected.completion);
                        EXPECT_EQ(resources.quant_ready, expected.quant_ready);
                    }
                    checked(hipMemsetAsync(operands.output, 0xa5, 64, operands.root));
                    ASSERT_TRUE(graph.launch());
                    std::array<unsigned char, 64> actual{};
                    checked(hipMemcpyAsync(actual.data(), operands.output, actual.size(),
                        hipMemcpyDeviceToHost, operands.root));
                    checked(hipStreamSynchronize(operands.root)); // Terminal test observation only.
                    for (int index = 0; index < 64; ++index)
                        EXPECT_EQ(actual[index], 0x11 * (index / 16 + 1));
                }
                graph.reset();
                ASSERT_TRUE(graph.beginCapture());
                recordBranch(context, operands.root, operands.output, 0, expected);
                graph.reset(); // Ends and destroys the unfinished native recording before lease release.
                hipStreamCaptureStatus status{};
                checked(hipStreamIsCapturing(operands.root, &status));
                EXPECT_EQ(status, hipStreamCaptureStatusNone);
                EXPECT_NO_THROW(context.assertProjectionResourcesRetirable());
                } // Destroy the graph object before context lifetime can retire.
                // Named reset never reconstructs fixed projection resources,
                // even after graph teardown. Only context cleanup retires them.
                EXPECT_NO_THROW(context.resetAuxiliaryStreams());
                EXPECT_NO_THROW(context.prepareProjectionResources());
                auto submission = context.acquireProjectionSubmission(
                    operands.root, GPUProjectionSubmissionKind::Eager);
                EXPECT_EQ(submission.resources().streams, expected.streams);
                EXPECT_EQ(submission.resources().completion, expected.completion);
                EXPECT_EQ(submission.resources().quant_ready, expected.quant_ready);
            });
        }
    }
}
