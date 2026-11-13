/**
 * @file Test__HostedDeviceGenerationContinuation.cpp
 * @brief Retained hosted transactions preserve ordering and exact-row ownership.
 *
 * The ticket describes the completed body's maintenance tail and the NEXT
 * body's admission. This model-free CUDA/ROCm proof retains both graphs and
 * exercises the production selector and budget kernel without host repair of
 * controller state. Exact-row bodies also prove scratch isolation across retained
 * width transitions. Whole-model generation separately proves real movement.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "kernels/common/SamplingMath.h"
#include "execution/mtp/MTPServingForwardCaptureGeometry.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <vector>
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

/**
 * @brief Retained exact-row transactions preserve controller and scratch ownership.
 *
 * This component proof composes the production budget, row-preparation, greedy
 * outcome and atomic response/publication-metadata kernels. Immutable device
 * fixtures replace request data, never topology. The host selects an authored
 * fixture graph, not downloaded live state. Full model/KV and TP transport
 * certification remain separate from this model-free transaction proof.
 */
TEST_P(HostedContinuation, ExactRowFamilyReplaysWithoutTouchingInactiveScratch)
{
    IBackend *backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto run = [&](IWorkerGPUContext &context)
    {
        context.submitAndWait([&]
        {
            constexpr int rows = 16;
            constexpr int poison = -9127;
            // Sweep every exact width, then explicitly revisit large/small
            // transitions. The final two cases poison selector/count agreement.
            constexpr std::array widths{16, 2, 16, 4, 8, 4, 3, 5, 6, 7, 9, 10,
                                        11, 12, 13, 14, 15, 2, 16, 4, 4, 2};
            constexpr size_t cases = widths.size();
            struct Input
            {
                std::array<int, kDeviceGenerationControlCount> control;
                int32_t first, base;
                std::array<int32_t, rows - 1> draft;
                std::array<int32_t, rows> target;
                std::array<int32_t, kSpeculativeBatchMaxStopTokens> stops;
            };
            struct Scratch
            {
                std::array<int32_t, rows> tokens, positions, outcome;
                std::array<int, kSpeculativeBatchMetaCount> meta;
                std::array<int32_t, 64> response;
                int32_t length, base_snapshot, restore, cached, accepted, ok;
                int32_t next, next_verifier, all_accepted, stopped;
                MTPCommittedVerifierIdentityRecord identity;
            };
            struct State { Input input; Scratch scratch; };
            struct Storage
            {
                State active;
                std::array<Input, cases> inputs;
                std::array<State, cases> observed;
            };
            Storage initial{};
            auto &dirty = initial.active.scratch;
            dirty.tokens.fill(poison);
            dirty.positions.fill(poison);
            dirty.outcome.fill(poison);
            dirty.meta.fill(poison);
            dirty.response.fill(poison);
            dirty.length = dirty.base_snapshot = dirty.restore = dirty.cached =
                dirty.accepted = dirty.ok = dirty.next = dirty.next_verifier = dirty.all_accepted = dirty.stopped = poison;
            for (size_t index = 0; index < cases; ++index)
            {
                auto &input = initial.inputs[index];
                auto policy = DeviceGenerationPolicy::fixed(widths[index] - 1);
                policy.mode = DeviceGenerationPolicyMode::Dynamic;
                policy.minimum_depth = 1;
                policy.maximum_depth = rows - 1;
                const int budget = index == 18 ? 2 : 32;
                ASSERT_TRUE(initialize_device_generation_control(budget, 64, policy, input.control.data()));
                input.first = 1000 + static_cast<int>(index) * 100;
                input.base = 200 + static_cast<int>(index) * 20;
                input.stops.fill(-1);
                for (int row = 0; row < rows - 1; ++row)
                    input.draft[row] = input.first + row + 1;
                for (int row = 0; row < rows; ++row)
                    input.target[row] = input.first + row + 1;
                if (index == 7) input.target[1] += 77; // Actual rejection.
                if (index == 8) input.stops[0] = input.target[1]; // EOS in accepted prefix.
                if (index == 19)
                    input.control[kDeviceGenerationControlNextLeadingCommittedOutputCount] = 1;
                if (index == cases - 2)
                    input.control[kDeviceGenerationControlCurrentDraftDepth] = 1;
                if (index == cases - 1)
                    input.control[kDeviceGenerationControlActiveVerifierRowCount] = 3;
            }
            void *const stream = context.defaultStream();
            ASSERT_NE(stream, nullptr);
            const auto free = [backend](Storage *p) { if (p) backend->free(p, 0); };
            std::unique_ptr<Storage, decltype(free)> storage(
                static_cast<Storage *>(backend->allocate(sizeof(Storage), 0)), free);
            ASSERT_NE(storage, nullptr);
            auto *const device = storage.get();
            ASSERT_TRUE(backend->hostToDevice(device, &initial, sizeof(initial), 0, stream));
            ASSERT_TRUE(backend->prepareMappedHostCopyKernels(0));

            // Graphs borrow one stable allocation. Largest-first family capture
            // and every fixture-copy capture finish before the first replay.
            std::array<std::unique_ptr<IGPUGraphCapture>, rows + 1> bodies;
            std::array<std::unique_ptr<IGPUGraphCapture>, cases> setup, observe;
            std::array<size_t, rows + 1> body_nodes{};
            const auto role = MTPVerifierExecutionRole::HostedCapturedTransactions;
            const auto family = resolveMTPVerifierGraphFamily(role);
            const auto physical = resolveMTPVerifierPhysicalWidthPolicy(role, MTPDepthPolicyMode::Dynamic);
            ASSERT_EQ(mtpVerifierGraphFamilyWidthCount(rows, family), rows - 1u);
            for (size_t member = 0; member < mtpVerifierGraphFamilyWidthCount(rows, family); ++member)
            {
                const int width = mtpVerifierGraphFamilyWidthAt(member, rows, family);
                ASSERT_EQ(width, rows - static_cast<int>(member));
                ASSERT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, width, rows, physical), width);
                auto &body = bodies[width];
                body = context.createGraphCapture(stream);
                ASSERT_NE(body, nullptr);
                auto &in = device->active.input;
                auto &out = device->active.scratch;
                ASSERT_TRUE(body->beginCapture());
                ASSERT_TRUE(backend->enqueuePrepareDeviceGenerationTransactionBudget(
                    in.control.data(), kDeviceGenerationControlCount, 1, width,
                    nullptr, nullptr, nullptr, 0, stream));
                ASSERT_TRUE(backend->enqueuePrepareMTPVerifierControlledRow(
                    &in.first, in.draft.data(), &in.base, in.control.data(),
                    kDeviceGenerationControlCount, width, 0, stream,
                    out.tokens.data(), out.positions.data(), &out.length, &out.base_snapshot));
                ASSERT_TRUE(backend->enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls(
                    in.target.data(), out.tokens.data(), width - 1,
                    &in.control[kDeviceGenerationControlActiveVerifierRowCount],
                    in.stops.data(), 0, stream, rows, out.outcome.data(), out.meta.data(),
                    &in.control[kDeviceGenerationControlTransactionCommitBudget],
                    &in.control[kDeviceGenerationControlNextLeadingCommittedOutputCount]));
                ASSERT_TRUE(backend->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                    out.outcome.data(), rows, out.meta.data(), kSpeculativeBatchMetaCount,
                    &out.base_snapshot, 1, width, out.response.data(), 64,
                    in.control.data(), kDeviceGenerationControlCount, 0, stream,
                    &out.restore, &out.cached, &out.accepted, &out.ok,
                    &out.next, &out.all_accepted, &out.stopped, nullptr, nullptr, &out.next_verifier,
                    out.tokens.data(), width, &out.identity));
                ASSERT_TRUE(body->endCapture());
                ASSERT_TRUE(body->instantiate());
                std::vector<GPUGraphKernelNodeInfo> kernels;
                std::string diagnostic;
                ASSERT_TRUE(body->inspectKernelNodes(kernels, &diagnostic)) << diagnostic;
                // A flat four-kernel body has no transfer/callback/child nodes.
                ASSERT_EQ(kernels.size(), 4u);
                ASSERT_EQ(body->nodeCount(), kernels.size());
                for (const auto &kernel : kernels) EXPECT_TRUE(kernel.valid());
                body_nodes[width] = body->nodeCount();
            }
            for (size_t index = 0; index < cases; ++index)
            {
                setup[index] = context.createGraphCapture(stream);
                observe[index] = context.createGraphCapture(stream);
                ASSERT_NE(setup[index], nullptr);
                ASSERT_NE(observe[index], nullptr);
                ASSERT_TRUE(setup[index]->beginCapture());
                ASSERT_TRUE(backend->copyDeviceVisibleRegionByKernelOnStream(
                    &device->active.input, &device->inputs[index], sizeof(Input), 0, stream));
                ASSERT_TRUE(setup[index]->endCapture());
                ASSERT_TRUE(setup[index]->instantiate());
                ASSERT_TRUE(observe[index]->beginCapture());
                ASSERT_TRUE(backend->copyDeviceVisibleRegionByKernelOnStream(
                    &device->observed[index], &device->active, sizeof(State), 0, stream));
                ASSERT_TRUE(observe[index]->endCapture());
                ASSERT_TRUE(observe[index]->instantiate());
            }
            const auto before = backend->deviceAllocationAccounting(0);
            ASSERT_TRUE(before.supported) << before.diagnostic;
            std::array<State, cases> expected{};
            State oracle = initial.active;
            // Only retained graph submission occurs in this window: fixture
            // input/observations stay on device and dirty scratch persists.
            for (int repeat = 0; repeat < 20; ++repeat)
            for (size_t index = 0; index < cases; ++index)
            {
                const int width = widths[index];
                oracle.input = initial.inputs[index];
                auto &in = oracle.input;
                auto &out = oracle.scratch;
                auto *control = in.control.data();
                prepare_device_generation_transaction_budget(width, width, control);
                const int depth = control[kDeviceGenerationControlCurrentDraftDepth];
                const int live = control[kDeviceGenerationControlActiveVerifierRowCount];
                const bool valid = control[kDeviceGenerationControlOk] != 0 &&
                    control[kDeviceGenerationControlRequestComplete] == 0 &&
                    depth > 0 && live == depth + 1 && live <= width;
                if (!valid) fail_device_generation_control(control, DeviceGenerationError::InvalidDepthSelector);
                out.length = valid ? live : 0;
                out.base_snapshot = in.base;
                for (int row = 0; row < width; ++row)
                {
                    out.positions[row] = in.base + row;
                    out.tokens[row] = row == 0 ? in.first : valid && row < live ? in.draft[row - 1] : 0;
                }
                if (live <= 0 || live > width)
                    initialize_invalid_speculative_batch_outcome(out.outcome.data(), rows, out.meta.data());
                else
                    summarize_greedy_speculative_verify_batch_at_commit_boundary(
                        out.tokens[0], in.target.data(), out.tokens.data(), live - 1,
                        in.stops.data(), kSpeculativeBatchMaxStopTokens,
                        std::min(control[kDeviceGenerationControlTransactionCommitBudget], live),
                        out.outcome.data(), rows, out.meta.data(),
                        control[kDeviceGenerationControlNextLeadingCommittedOutputCount]);
                const bool committed = commit_device_generation_and_derive_speculative_publication_metadata(
                    out.outcome.data(), rows, out.meta.data(), kSpeculativeBatchMetaCount,
                    0, width, out.base_snapshot, out.response.data(), 64, control,
                    &out.restore, &out.cached, &out.accepted, &out.ok,
                    &out.next, &out.all_accepted, &out.stopped);
                if (committed && out.ok != 0)
                {
                    ASSERT_TRUE(publish_committed_verifier_identity(
                        out.tokens.data(), width, control, &out.identity));
                }
                // The backend transaction also publishes the next verifier's
                // separate persistent input word, including fatal-controller
                // replays that preserve the previously published condition.
                out.next_verifier = out.next;
                expected[index] = oracle;
                ASSERT_TRUE(setup[index]->launchOnStream(stream));
                ASSERT_TRUE(bodies[width]->launchOnStream(stream));
                ASSERT_TRUE(observe[index]->launchOnStream(stream));
            }
            std::array<State, cases> observed{};
            // One terminal device-to-host oracle, never a scheduler live read.
            ASSERT_TRUE(backend->deviceToHost(observed.data(), device->observed.data(), sizeof(observed), 0, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, 0));
            const auto after = backend->deviceAllocationAccounting(0);
            ASSERT_TRUE(after.supported) << after.diagnostic;
            EXPECT_EQ(before.active_allocations, after.active_allocations);
            EXPECT_EQ(before.active_bytes, after.active_bytes);
            for (int width = 2; width <= rows; ++width)
            {
                EXPECT_TRUE(bodies[width]->hasExecutable());
                EXPECT_EQ(bodies[width]->nodeCount(), body_nodes[width]);
            }
            for (size_t index = 0; index < cases; ++index)
            {
                SCOPED_TRACE(::testing::Message() << GetParam() << " width=" << widths[index] << " fixture=" << index);
                const auto &actual = observed[index];
                const auto &want = expected[index];
                EXPECT_EQ(actual.input.control, want.input.control);
                EXPECT_EQ(actual.scratch.tokens, want.scratch.tokens);
                EXPECT_EQ(actual.scratch.positions, want.scratch.positions);
                EXPECT_EQ(actual.scratch.outcome, want.scratch.outcome);
                EXPECT_EQ(actual.scratch.meta, want.scratch.meta);
                EXPECT_EQ(actual.scratch.response, want.scratch.response);
                EXPECT_EQ(actual.scratch.length, want.scratch.length);
                EXPECT_EQ(actual.scratch.base_snapshot, want.scratch.base_snapshot);
                EXPECT_EQ(actual.scratch.restore, want.scratch.restore);
                EXPECT_EQ(actual.scratch.cached, want.scratch.cached);
                EXPECT_EQ(actual.scratch.accepted, want.scratch.accepted);
                EXPECT_EQ(actual.scratch.ok, want.scratch.ok);
                EXPECT_EQ(actual.scratch.next, want.scratch.next);
                EXPECT_EQ(actual.scratch.next_verifier, want.scratch.next_verifier);
                EXPECT_EQ(actual.scratch.all_accepted, want.scratch.all_accepted);
                EXPECT_EQ(actual.scratch.stopped, want.scratch.stopped);
                EXPECT_EQ(actual.scratch.identity.valid, want.scratch.identity.valid);
                EXPECT_EQ(actual.scratch.identity.version, want.scratch.identity.version);
                EXPECT_EQ(actual.scratch.identity.transaction_count, want.scratch.identity.transaction_count);
                EXPECT_EQ(actual.scratch.identity.draft_depth, want.scratch.identity.draft_depth);
                for (int row = 0; row < kSpeculativeBatchMaxOutputTokens; ++row)
                    EXPECT_EQ(actual.scratch.identity.verifier_input_tokens[row],
                              want.scratch.identity.verifier_input_tokens[row]);
                if (index >= cases - 2)
                {
                    EXPECT_EQ(actual.input.control[kDeviceGenerationControlOk], 0);
                    EXPECT_EQ(actual.input.control[kDeviceGenerationControlErrorCode],
                        static_cast<int>(DeviceGenerationError::InvalidDepthSelector));
                    EXPECT_EQ(actual.scratch.length, 0);
                    EXPECT_EQ(actual.scratch.ok, 0);
                }
                else
                {
                    EXPECT_EQ(actual.input.control[kDeviceGenerationControlOk], 1);
                    EXPECT_EQ(actual.scratch.length, widths[index]);
                    EXPECT_EQ(actual.scratch.ok, 1);
                }
            }
        });
    };
    if (GetParam() == "CUDA") run(GPUDeviceContextPool::instance().getNvidiaContext(0));
    else run(GPUDeviceContextPool::instance().getAMDContext(0));
}

INSTANTIATE_TEST_SUITE_P(GPU, HostedContinuation, ::testing::Values("CUDA", "ROCm"),
    [](const auto &info) { return info.param; });
}
