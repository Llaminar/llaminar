/**
 * @file Test__StreamingMTPDepth.cpp
 * @brief Native captured proof that response publication preserves MTP learning.
 *
 * CUDA and ROCm execute consecutive short response windows in one retained
 * graph, with no intermediate host observation or upload of learner state.
 * Immutable compact verifier outcomes are test inputs. Every terminal word and
 * emitted token is compared with the shared serial arithmetic oracle; separate
 * assertions prove windows actually accumulate and request reset is repeatable.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/mtp/DeviceGenerationContract.h"
#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace sampling_math;
using Control = std::array<int, kDeviceGenerationControlCount>;

/** @brief Isolate native ownership and diagnostics for each backend. */
class StreamingMTPDepth : public ::testing::TestWithParam<std::string> {};

/** @test Partial windows, learned decisions and reset survive twenty native replays. */
TEST_P(StreamingMTPDepth, CapturedShortWindowsRetainDeviceLearning)
{
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &pool = GPUDeviceContextPool::instance();
    auto &context = cuda ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
    context.submitAndWait([&] {
        auto *stream = context.defaultStream();
        ASSERT_NE(stream, nullptr);
        const auto release = [backend](void *p) { if (p) backend->free(p, 0); };
        const auto allocate = [&](size_t bytes) {
            return std::unique_ptr<void, decltype(release)>(backend->allocate(bytes, 0), release);
        };
        constexpr int windows = 96, capacity = 16;
        constexpr int input_stride = capacity + kSpeculativeBatchMetaCount + 1;
        constexpr uint64_t epoch = 0x123456789abcdef0ULL, generation = 0xfedcba9876543210ULL;
        auto input_memory = allocate(windows * capacity * input_stride * sizeof(int));
        auto control_memory = allocate(sizeof(Control));
        auto response_memory = allocate(capacity * sizeof(int));
        auto ticket_memory = allocate(sizeof(DeviceGenerationDispatchTicket));
        auto publication_memory = allocate(6 * sizeof(int));
        auto snapshots_memory = allocate(windows * sizeof(Control));
        auto output_memory = allocate(windows * capacity * sizeof(int));
        ASSERT_TRUE(input_memory && control_memory && response_memory && ticket_memory &&
                    publication_memory && snapshots_memory && output_memory);
        auto *inputs = static_cast<int *>(input_memory.get());
        auto *control = static_cast<int *>(control_memory.get());
        auto *response = static_cast<int *>(response_memory.get());
        auto *ticket = static_cast<DeviceGenerationDispatchTicket *>(ticket_memory.get());
        auto *publication = static_cast<int *>(publication_memory.get());
        auto *snapshots = static_cast<int *>(snapshots_memory.get());
        auto *outputs = static_cast<int *>(output_memory.get());

        for (auto mode : {DeviceGenerationPolicyMode::Fixed, DeviceGenerationPolicyMode::Observe,
                          DeviceGenerationPolicyMode::Dynamic})
        for (int initial : {1, 3, 15})
        for (auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
        for (bool learned : {false, true})
        {
            if (learned && mode != DeviceGenerationPolicyMode::Dynamic)
                continue;
            SCOPED_TRACE(::testing::Message() << GetParam() << " mode=" << int(mode)
                << " initial=" << initial << " verify=" << int(verify) << " learned=" << learned);
            auto policy = DeviceGenerationPolicy::fixed(initial);
            policy.mode = mode;
            if (mode != DeviceGenerationPolicyMode::Fixed) {
                policy.minimum_depth = 1;
                policy.maximum_depth = 15;
            }
            policy.window_size = policy.minimum_samples = 32;
            policy.cooldown_steps = 0;
            policy.promote_consecutive_windows = 1;
            policy.learned = {.enabled = learned,
                .backend = cuda ? MTPDepthPolicyBackend::CUDA : MTPDepthPolicyBackend::ROCm,
                .model_class = MTPDepthPolicyModelClass::Dense, .verify_mode = verify};
            ASSERT_TRUE(policy.valid());
            std::vector<int> input(windows * capacity * input_stride);
            std::vector<int> expected_outputs(windows * capacity, -777);
            std::vector<Control> expected(windows);
            std::array<DeviceGenerationLeadingRowDisposition, windows> leading_rows{};
            std::array<std::vector<int>, windows> transaction_rows;
            Control reference{};
            std::array<int, capacity> reference_response{};
            DeviceGenerationDispatchTicket reference_ticket{};
            ASSERT_TRUE(initialize_device_generation_dispatch_ticket(epoch, generation, &reference_ticket));
            int base = 23, transaction = 0, eligible_observations = 0;
            int next_token = 100, previous_condition = -1;
            for (int window = 0; window < windows; ++window) {
                leading_rows[window] = window == 0 ? DeviceGenerationLeadingRowDisposition::PendingResponse
                    : static_cast<DeviceGenerationLeadingRowDisposition>(
                        reference[kDeviceGenerationControlNextLeadingCommittedOutputCount]);
                ASSERT_TRUE(initialize_device_generation_control(capacity, capacity, policy,
                    reference.data(), leading_rows[window],
                    {.kind = window == 0 ? DeviceGenerationAdmissionKind::NewRequest
                                         : DeviceGenerationAdmissionKind::ContinueResponse,
                     .session_epoch = epoch, .workspace_generation = generation,
                     .prior_tickets = &reference_ticket}));
                while (!reference[kDeviceGenerationControlRequestComplete]) {
                    const int depth = reference[kDeviceGenerationControlCurrentDraftDepth];
                    const int leading = reference[kDeviceGenerationControlNextLeadingCommittedOutputCount];
                    const int remaining = reference[kDeviceGenerationControlRemainingTokenCount];
                    const int budget = prepare_device_generation_transaction_budget(capacity, capacity,
                        reference.data());
                    ASSERT_GT(budget, 0);
                    eligible_observations += budget == depth + 1 ? 1 : 0;
                    const int output_count = std::min(depth + 1, remaining + leading);
                    const int row_index = transaction++;
                    ASSERT_LE(transaction, windows * capacity);
                    transaction_rows[window].push_back(row_index);
                    int *tokens = input.data() + row_index * input_stride;
                    for (int row = 0; row < output_count; ++row)
                        tokens[row] = row == 0 && leading ? previous_condition : next_token++;
                    int *meta = tokens + capacity;
                    meta[kSpecBatchMetaOk] = 1;
                    meta[kSpecBatchMetaOutputCount] = output_count;
                    meta[kSpecBatchMetaLeadingCommittedOutputCount] = leading;
                    meta[kSpecBatchMetaAcceptedSpeculativePrefix] = output_count - 1;
                    meta[kSpecBatchMetaTargetVerifierStateCommitCount] =
                        output_count == depth + 1 ? depth : output_count;
                    meta[kSpecBatchMetaConsumedVerifierRows] = output_count - 1;
                    meta[kSpecBatchMetaAllSpeculativeAccepted] = output_count == depth + 1 ? 1 : 0;
                    meta[kSpecBatchMetaSampledTerminal] = output_count == depth + 1 ? 1 : 0;
                    meta[kSpecBatchMetaCommitBoundaryClipped] = budget < depth + 1 ? 1 : 0;
                    meta[kSpecBatchMetaReadyToken] = -1;
                    tokens[input_stride - 1] = base;
                    const int prior_commits = reference[kDeviceGenerationControlPublishedStateCommitCount];
                    ASSERT_TRUE(append_speculative_outcome_to_device_generation(tokens, capacity, meta,
                        kSpeculativeBatchMetaCount, reference_response.data(), capacity, reference.data()));
                    base += reference[kDeviceGenerationControlPublishedStateCommitCount] - prior_commits;
                    previous_condition = tokens[output_count - 1];
                }
                EXPECT_EQ(validateDeviceGenerationTerminal(reference,
                    {.request_count = 1, .max_new_tokens = capacity, .depth_policy = policy,
                     .initial_leading_row_disposition = leading_rows[window]},
                    capacity, 0, 0), DeviceGenerationTerminalError::None);
                expected[window] = reference;
                std::copy(reference_response.begin(), reference_response.end(),
                    expected_outputs.begin() + window * capacity);
            }
            int expected_windows = 0;
            for (const auto &row : expected)
                expected_windows += row[kDeviceGenerationControlDepthEvaluatedWindows];
            ASSERT_EQ(expected_windows, mode == DeviceGenerationPolicyMode::Fixed ? 0 : eligible_observations / 32);
            if (mode != DeviceGenerationPolicyMode::Fixed)
                ASSERT_GT(expected_windows, 0);
            input.resize(transaction * input_stride);
            ASSERT_TRUE(backend->hostToDevice(inputs, input.data(), input.size() * sizeof(int), 0, stream));
            const std::vector<int> poison(windows * capacity, -777);
            ASSERT_TRUE(backend->hostToDevice(outputs, poison.data(), poison.size() * sizeof(int), 0, stream));
            auto graph = context.createGraphCapture(stream);
            ASSERT_TRUE(graph && graph->beginCapture());
            for (int window = 0; window < windows; ++window) {
                ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(1, capacity, policy,
                    leading_rows[window], capacity, response, kDeviceGenerationControlCount, control, 0, stream,
                    {.kind = window == 0 ? DeviceGenerationAdmissionKind::NewRequest
                                         : DeviceGenerationAdmissionKind::ContinueResponse,
                     .session_epoch = epoch, .workspace_generation = generation, .prior_tickets = ticket}));
                ASSERT_TRUE(backend->enqueueInitializeDeviceGenerationDispatchTicket(epoch, generation,
                    control, kDeviceGenerationControlCount, 1, ticket, 0, stream));
                for (const int row_index : transaction_rows[window]) {
                    ASSERT_TRUE(backend->enqueuePrepareDeviceGenerationTransactionBudget(control,
                        kDeviceGenerationControlCount, 1, capacity, nullptr, nullptr, nullptr, 0, stream));
                    auto *tokens = inputs + row_index * input_stride;
                    ASSERT_TRUE(backend->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                        tokens, capacity, tokens + capacity, kSpeculativeBatchMetaCount,
                        tokens + input_stride - 1, 1, capacity, response, capacity, control,
                        kDeviceGenerationControlCount, 0, stream, publication, publication + 1,
                        publication + 2, publication + 3, publication + 4, nullptr, nullptr,
                        nullptr, nullptr, publication + 5));
                }
                // Test observations remain entirely device-local until the
                // complete captured request returns. The retained graph owns
                // completion; enqueue only and copy the exact live extent.
                ASSERT_TRUE(backend->deviceCopyAsync(snapshots + window * kDeviceGenerationControlCount,
                    control, sizeof(Control), 0, stream));
                ASSERT_TRUE(backend->deviceCopyAsync(outputs + window * capacity, response,
                    capacity * sizeof(int), 0, stream));
            }
            ASSERT_TRUE(graph->endCapture() && graph->instantiate());
            for (int replay = 0; replay < 20; ++replay) {
                SCOPED_TRACE(replay);
                ASSERT_TRUE(graph->launch());
                std::vector<Control> actual(windows);
                std::vector<int> actual_outputs(windows * capacity);
                ASSERT_TRUE(backend->deviceToHost(actual.data(), snapshots, windows * sizeof(Control), 0, stream));
                ASSERT_TRUE(backend->deviceToHost(actual_outputs.data(), outputs,
                    actual_outputs.size() * sizeof(int), 0, stream));
                for (int window = 0; window < windows; ++window)
                    ASSERT_EQ(actual[window], expected[window]) << "window=" << window;
                ASSERT_EQ(actual_outputs, expected_outputs);
            }
        }
    });
}

/** @test One stale request cannot borrow another row's authenticated learner. */
TEST_P(StreamingMTPDepth, CapturedContinuationAuthenticatesEveryRequestRow)
{
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &pool = GPUDeviceContextPool::instance();
    auto &context = cuda ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
    context.submitAndWait([&] {
        auto *stream = context.defaultStream();
        const auto release = [backend](void *p) { if (p) backend->free(p, 0); };
        const auto allocate = [&](size_t bytes) {
            return std::unique_ptr<void, decltype(release)>(backend->allocate(bytes, 0), release);
        };
        auto controls = allocate(2 * sizeof(Control));
        auto tickets = allocate(2 * sizeof(DeviceGenerationDispatchTicket));
        auto responses = allocate(2 * 16 * sizeof(int));
        ASSERT_TRUE(controls && tickets && responses);
        auto policy = DeviceGenerationPolicy::fixed(3);
        policy.mode = DeviceGenerationPolicyMode::Dynamic;
        policy.minimum_depth = 1;
        policy.maximum_depth = 15;
        std::array<Control, 2> initial{};
        std::array<DeviceGenerationDispatchTicket, 2> identities{};
        for (int row = 0; row < 2; ++row) {
            ASSERT_TRUE(initialize_device_generation_control(16, 16, policy, initial[row].data()));
            auto &control = initial[row];
            control[kDeviceGenerationControlCurrentDraftDepth] = 5;
            control[kDeviceGenerationControlActiveVerifierRowCount] = 6;
            control[kDeviceGenerationControlDepthWindowVerifierRuns] = 7;
            control[kDeviceGenerationControlRequestComplete] = 1;
            control[kDeviceGenerationControlRemainingTokenCount] = 0;
            control[kDeviceGenerationControlResponseTokenCount] = 16;
            control[kDeviceGenerationControlTransactionCount] = 8;
            ASSERT_TRUE(initialize_device_generation_dispatch_ticket(17 + row, 41, &identities[row]));
        }
        const DeviceGenerationInitialization initialization{
            .kind = DeviceGenerationAdmissionKind::ContinueResponse,
            .session_epoch = 17, .workspace_generation = 41,
            .prior_tickets = static_cast<DeviceGenerationDispatchTicket *>(tickets.get())};
        EXPECT_FALSE(backend->enqueueInitializeDeviceGeneration(2, 16, policy,
            DeviceGenerationLeadingRowDisposition::PendingResponse, 16, responses.get(),
            kDeviceGenerationControlCount, controls.get(), 0, nullptr, initialization));
        auto graph = context.createGraphCapture(stream);
        ASSERT_TRUE(graph && graph->beginCapture());
        ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(2, 16, policy,
            DeviceGenerationLeadingRowDisposition::PendingResponse, 16, responses.get(),
            kDeviceGenerationControlCount, controls.get(), 0, stream, initialization));
        ASSERT_TRUE(graph->endCapture() && graph->instantiate());
        for (int replay = 0; replay < 20; ++replay) {
            ASSERT_TRUE(backend->hostToDevice(controls.get(), initial.data(), sizeof(initial), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(tickets.get(), identities.data(), sizeof(identities), 0, stream));
            ASSERT_TRUE(graph->launch());
            std::array<Control, 2> actual{};
            ASSERT_TRUE(backend->deviceToHost(actual.data(), controls.get(), sizeof(actual), 0, stream));
            EXPECT_EQ(actual[0][kDeviceGenerationControlOk], 1);
            EXPECT_EQ(actual[0][kDeviceGenerationControlCurrentDraftDepth], 5);
            EXPECT_EQ(actual[0][kDeviceGenerationControlDepthWindowVerifierRuns], 7);
            EXPECT_EQ(actual[0][kDeviceGenerationControlResponseTokenCount], 0);
            EXPECT_EQ(actual[1][kDeviceGenerationControlOk], 0);
            EXPECT_EQ(actual[1][kDeviceGenerationControlErrorCode], int(DeviceGenerationError::InvalidContinuation));
        }
    });
}

INSTANTIATE_TEST_SUITE_P(Backends, StreamingMTPDepth, ::testing::Values("CUDA", "ROCm"),
    [](const ::testing::TestParamInfo<std::string> &info) { return info.param; });
}
