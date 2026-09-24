/**
 * @file Test__HostedDeviceGenerationContinuation.cpp
 * @brief Captured continuation ordering across an exhausted maintenance boundary.
 *
 * The ticket describes the completed body's maintenance tail and the NEXT
 * body's admission. This model-free CUDA/ROCm proof retains both graphs and
 * exercises the production selector and budget kernel without host repair of
 * controller state. Whole-model generation separately proves real movement.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "kernels/common/SamplingMath.h"
#include <gtest/gtest.h>
#include <array>
#include <functional>
#include <memory>

namespace
{
using namespace llaminar2;
using namespace llaminar2::sampling_math;

/** @brief The same retained-transaction protocol is required on both GPU APIs. */
class HostedContinuation : public ::testing::TestWithParam<std::string> {};

/**
 * @brief Retire due work before admitting the next verifier, including final tickets.
 *
 * A tiny captured device copy models the maintenance tail's publication, not
 * its planner/copy implementation. A deliberately exhausted boundary makes
 * body-first ordering fail in the actual production budget kernel. Sentinel
 * bytes and twenty resets cover both selected depths and terminal-only tails.
 */
TEST_P(HostedContinuation, RetiresDueTailBeforeNextVerifierAndAtTermination)
{
    IBackend *backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto run = [&](IWorkerGPUContext &context)
    {
        context.submitAndWait([&]
        {
            void *const stream = context.defaultStream();
            ASSERT_NE(stream, nullptr);
            struct Boundary { uint32_t rows, due, advanced; };
            struct State
            {
                std::array<int, kDeviceGenerationControlCount> control;
                Boundary pending;
                Boundary ready;
            };
            const auto free = [backend](State *p) { if (p) backend->free(p, 0); };
            std::unique_ptr<State, decltype(free)> storage(
                static_cast<State *>(backend->allocate(sizeof(State), 0)), free);
            ASSERT_NE(storage, nullptr);
            State *const device = storage.get();
            ASSERT_TRUE(backend->prepareMappedHostCopyKernels(0));

            // These graphs only borrow storage; their destructors run before
            // the allocation owner, even if a test assertion aborts the scope.
            auto maintenance = context.createGraphCapture(stream);
            auto verifier = context.createGraphCapture(stream);
            ASSERT_TRUE(maintenance->beginCapture());
            ASSERT_TRUE(backend->copyDeviceVisibleRegionByKernelOnStream(&device->pending, &device->ready,
                sizeof(Boundary), 0, stream));
            ASSERT_TRUE(maintenance->endCapture());
            ASSERT_TRUE(maintenance->instantiate());
            ASSERT_TRUE(verifier->beginCapture());
            ASSERT_TRUE(backend->enqueuePrepareDeviceGenerationTransactionBudget(
                device->control.data(), kDeviceGenerationControlCount, 1, 16,
                &device->pending.rows, &device->pending.due,
                &device->pending.advanced, 0, stream));
            ASSERT_TRUE(verifier->endCapture());
            ASSERT_TRUE(verifier->instantiate());
            const std::array<DeviceControlledLoopFragment, 2> branch{{
                {.name = "verifier", .capture = verifier.get(),
                 .execution = DeviceControlledLoopFragmentExecution::Always},
                {.name = "completed maintenance", .capture = maintenance.get(),
                 .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero,
                 .condition_word_device = &device->pending.due},
            }};
            const std::span<const DeviceControlledLoopFragment> ordered(branch);
            for (int repeat = 0; repeat < 20; ++repeat)
            for (int depth : {1, 2, 3, 15})
            for (bool terminal : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << GetParam() << " replay=" << repeat
                    << " depth=" << depth << " terminal=" << terminal);
                DeviceGenerationPolicy policy;
                policy.initial_depth = policy.minimum_depth = policy.maximum_depth = depth;
                State initial{};
                ASSERT_TRUE(initialize_device_generation_control(32, 32, policy, initial.control.data()));
                initial.control[kDeviceGenerationControlRequestComplete] = terminal;
                initial.pending = {.rows = 0, .due = 1, .advanced = 1};
                initial.ready = {.rows = 16, .due = 0, .advanced = 0};
                ASSERT_TRUE(backend->hostToDevice(device, &initial, sizeof(State), 0, stream));

                const DeviceControlledLoopTicketSelection continuation{
                    .next_iteration_admitted = !terminal,
                    .completed_iteration_word_nonzero = true, .selector = depth};
                const auto count = continuation.countSelected(ordered);
                ASSERT_EQ(count, terminal ? 1u : 2u);
                for (size_t ordinal = 0; ordinal < count; ++ordinal)
                {
                    const auto *fragment = continuation.selectOrdinal(ordered, ordinal);
                    ASSERT_NE(fragment, nullptr);
                    ASSERT_TRUE(fragment->capture->launchOnStream(stream));
                }
                State observed{};
                ASSERT_TRUE(backend->deviceToHost(&observed, device, sizeof(State), 0, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, 0)); // Terminal test oracle only.
                EXPECT_EQ(observed.control[kDeviceGenerationControlOk], 1);
                EXPECT_EQ(observed.control[kDeviceGenerationControlErrorCode], 0);
                EXPECT_EQ(observed.control[kDeviceGenerationControlRequestComplete], terminal);
                EXPECT_EQ(observed.control[kDeviceGenerationControlTransactionCommitBudget], terminal ? 0 : depth + 1);
                EXPECT_EQ(observed.pending.rows, 16u);
                EXPECT_EQ(observed.pending.due, 0u);
                EXPECT_EQ(observed.pending.advanced, 0u);
            }
        });
    };
    if (GetParam() == "CUDA") run(GPUDeviceContextPool::instance().getNvidiaContext(0));
    else run(GPUDeviceContextPool::instance().getAMDContext(0));
}

/**
 * @brief Later captured geometry failure cannot hide the original budget error.
 *
 * This is the exact cascade observed in the model failure: exhausted cadence
 * first poisons admission, then row preparation encounters the failed request.
 * Both GPU compilers must preserve the first error through all twenty replays.
 */
TEST_P(HostedContinuation, PreservesBudgetFailureThroughLaterGeometryFailure)
{
    IBackend *backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto run = [&](IWorkerGPUContext &context)
    {
        context.submitAndWait([&]
        {
            void *const stream = context.defaultStream();
            ASSERT_NE(stream, nullptr);
            struct State
            {
                std::array<int, kDeviceGenerationControlCount> control;
                uint32_t rows, due, advanced;
                int32_t first, draft, base, length, snapshot;
                std::array<int32_t, 2> tokens, positions;
            };
            const auto free = [backend](State *p) { if (p) backend->free(p, 0); };
            std::unique_ptr<State, decltype(free)> storage(
                static_cast<State *>(backend->allocate(sizeof(State), 0)), free);
            ASSERT_NE(storage, nullptr);
            auto *device = storage.get();
            auto capture = context.createGraphCapture(stream);
            ASSERT_TRUE(capture->beginCapture());
            ASSERT_TRUE(backend->enqueuePrepareDeviceGenerationTransactionBudget(
                device->control.data(), kDeviceGenerationControlCount, 1, 2,
                &device->rows, &device->due, &device->advanced, 0, stream));
            ASSERT_TRUE(backend->enqueuePrepareMTPVerifierControlledRow(
                &device->first, &device->draft, &device->base, device->control.data(),
                kDeviceGenerationControlCount, 2, 0, stream, device->tokens.data(),
                device->positions.data(), &device->length, &device->snapshot));
            ASSERT_TRUE(capture->endCapture());
            ASSERT_TRUE(capture->instantiate());
            for (int repeat = 0; repeat < 20; ++repeat)
            {
                State initial{};
                DeviceGenerationPolicy policy;
                policy.initial_depth = policy.minimum_depth = policy.maximum_depth = 1;
                ASSERT_TRUE(initialize_device_generation_control(32, 32, policy, initial.control.data()));
                initial.due = initial.advanced = 1;
                ASSERT_TRUE(backend->hostToDevice(device, &initial, sizeof(State), 0, stream));
                ASSERT_TRUE(capture->launch());
                State observed{};
                ASSERT_TRUE(backend->deviceToHost(&observed, device, sizeof(State), 0, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, 0)); // Terminal test oracle only.
                EXPECT_EQ(observed.control[kDeviceGenerationControlErrorCode],
                    static_cast<int>(DeviceGenerationError::InvalidController));
                EXPECT_EQ(observed.control[kDeviceGenerationControlOk], 0);
                EXPECT_EQ(observed.control[kDeviceGenerationControlRequestComplete], 1);
                EXPECT_EQ(observed.control[kDeviceGenerationControlTransactionCommitBudget], 0);
                EXPECT_EQ(observed.length, 0);
            }
        });
    };
    if (GetParam() == "CUDA") run(GPUDeviceContextPool::instance().getNvidiaContext(0));
    else run(GPUDeviceContextPool::instance().getAMDContext(0));
}

INSTANTIATE_TEST_SUITE_P(GPU, HostedContinuation, ::testing::Values("CUDA", "ROCm"),
    [](const auto &info) { return info.param; });
}
