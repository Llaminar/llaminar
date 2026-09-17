/**
 * @file Test__PipelineMTPStateOwnership.cpp
 * @brief Model-free CPU/CUDA/ROCm proofs of pipeline state and captured generation.
 *
 * The real DGO, arena and cache factory construct head, middle and terminal
 * participants. State tests exercise allocation and rollback lifetime. GPU
 * generation tests compose real embedding, KV and sampling kernels, checking
 * an independent token oracle and exact prefix-cache bytes across retained
 * native graphs. These focused proofs do not replace real-weight HTTP or
 * mathematical model certification.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/local_execution/orchestrators/PipelineForwardGraphEdges.h"
#include "execution/local_execution/orchestrators/TPWorkerPool.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/mtp/MTPCheckpointPolicy.h"
#include "loaders/WeightPlan.h"
#include "models/qwen35/Qwen35Graph.h"
#include "execution/compute_stages/stages/EmbeddingStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.h"
#include "execution/compute_stages/stages/MTPVerifierPreparationStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/TestTensorFactory.h"
#include "transfer/TransferEngine.h"
#include "collective/BackendRouter.h"
#include "collective/LocalTPContext.h"
#include "planning/MemoryPlanner.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/FP16Utils.h"
#include "../../../mocks/MockModelContext.h"
#include "../../backends/MTPMainForwardReadRetirementProof.h"

namespace llaminar2::test
{
/** @brief Explicit backend selection keeps CPU-only preflight device-free. */
class PipelineMTPStateOwnership : public ::testing::TestWithParam<std::string>
{
protected:
    /** @brief Small real Qwen schema; model weights are unnecessary for state admission. */
    static GraphConfig stateConfig(DeviceId device)
    {
        GraphConfig config;
        config.d_model = 32;
        config.d_ff = config.d_ff_local = 64;
        config.n_heads = config.local_n_heads = 1;
        config.n_kv_heads = config.local_n_kv_heads = 1;
        config.head_dim = 32;
        config.n_layers = 3;
        config.total_n_layers = 4;
        config.vocab_size = config.vocab_local = 32;
        config.max_seq_len = 64;
        config.default_device = device;
        config.layer_types.assign(4, "full_attention");
        config.gdn.conv_kernel_size = 4;
        config.gdn.state_size = config.head_dim;
        config.gdn.inner_size = config.d_model;
        config.gdn.group_count = config.n_kv_heads;
        config.gdn.time_step_rank = config.n_heads;
        // Match the production Qwen configuration builder.  A retained GPU
        // MTP family owns the shifted-KV append inside the captured prefill
        // transaction; the generic GraphConfig default is intentionally not
        // a valid substitute for graph-native MTP execution.
        config.mtp_request_terminal_hidden_publication =
            MTPRequestTerminalHiddenPublicationPolicy::GraphCapturedDeviceGeometry;
        config.mtp_shifted_prefill_hidden_publication =
            MTPShiftedPrefillHiddenPublicationPolicy::GraphIntegratedKVTransaction;
        return config;
    }
};

#include "Test__OrdinaryDGOGeneration.inc"
#include "Test__PipelineMTPPublication.inc"
#include "Test__PipelineMTPPublicationTransport.inc"
#include "Test__PipelineVerifierEdges.inc"
#include "Test__PipelineMTPForwardInput.inc"
#include "Test__PipelineMTPPublicationBindings.inc"
#include "Test__PipelineMTPMainForward.inc"

/**
 * @brief Retained MTP capacity is distinct from both request enablement and PP role.
 *
 * All stages keep the main cache. Only the terminal owns a shifted predictor;
 * an unpartitioned runner retains its existing predictor ownership. Depth-15
 * capacity must not duplicate learned predictor caches fifteen times.
 */
TEST_P(PipelineMTPStateOwnership, ConstructionAndRollbackMatchStageOwnership)
{
    initCPUBackend(-1);
    const auto device = GetParam() == "CPU" ? DeviceId::cpu()
                      : GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto prove = [&]
    {
        for (int capacity_mode = 0; capacity_mode < 3; ++capacity_mode)
            for (int stage = 0; stage < 4; ++stage)
            {
                SCOPED_TRACE(GetParam() + ":mode=" + std::to_string(capacity_mode) +
                             ":stage=" + std::to_string(stage));
                auto config = stateConfig(device);
                config.mtp.enabled = capacity_mode == 1;
                config.mtp.graph_capacity_draft_tokens = capacity_mode == 2 ? 15 : 0;
                // Use the production MTP-capable schema. Qwen2's ordinary
                // schema correctly has no shifted predictor output buffers.
                auto graph = std::make_shared<Qwen35Graph>(config, nullptr);
                DeviceGraphOrchestrator runner(graph, nullptr);
                if (stage < 3)
                    runner.setPPStageConfig({.first_layer = stage, .last_layer = stage + 1,
                        .has_embedding = stage == 0, .has_lm_head = stage == 2});
                ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 64, device));
                const auto &state = runner.inferenceState();
                ASSERT_NE(state.kv_cache, nullptr);
                const size_t shifted = capacity_mode != 0 && stage >= 2 ? 1u : 0u;
                ASSERT_EQ(state.mtp_kv_caches.size(), shifted);
                if (stage < 2)
                {
                    using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
                    EXPECT_TRUE(Peer::bindFollowerOutputs(runner, ForwardExecutionRole::MTPCondition, 1));
                    EXPECT_TRUE(Peer::bindFollowerOutputs(runner, ForwardExecutionRole::GroupedMTPVerifier, 16));
                    for (const auto transaction : {ForwardStateTransaction::Ordinary,
                            ForwardStateTransaction::CommittedMTPCondition,
                            ForwardStateTransaction::RestoredPrefixMTPDecodeBridge})
                        EXPECT_TRUE(Peer::bindFollowerCondition(runner, transaction));
                }
                if (config.mtp.enabled)
                {
                    // The initial empty frontier is a real rollback boundary.
                    // Retain all concurrent checkpoint leases together, then
                    // release and repeat: removing MTP on followers cannot pass.
                    for (int replay = 0; replay < 3; ++replay)
                    {
                        // Exercise both same-request reuse and a request reset.
                        // Neither may depend on the host observing GPU completion:
                        // the next pool writer must wait for the prior restore.
                        if (replay == 2)
                            runner.resetInferenceState(InferenceStateResetRequest::requestBoundary(
                                "pipeline-state-ownership-replay"));
                        std::vector<PrefixStateSnapshot> snapshots;
                        for (size_t slot = 0; slot < kMTPConcurrentLiveCheckpointSets; ++slot)
                        {
                            auto snapshot = runner.captureLivePrefixCheckpoint({
                                .sequence_index = 0, .logical_cached_tokens = 0,
                                .maximum_main_append_tokens = 16,
                                .maximum_shifted_append_tokens = stage >= 2 ? 16 : 0});
                            ASSERT_TRUE(snapshot.valid);
                            EXPECT_EQ(snapshot.mtp_cached_tokens.size(), shifted);
                            if (device.is_gpu())
                                EXPECT_EQ(snapshot.device_sequence_state_checkpoints.size(), 1u + shifted);
                            snapshots.push_back(std::move(snapshot));
                        }
                        // Restoring also joins the producer event before these
                        // pooled destinations can be released or reused.
                        ASSERT_TRUE(runner.restoreLivePrefixState(snapshots.front()));
                    }
                }
            }
    };
    if (device.is_cpu())
        prove();
    else
    {
        ASSERT_NE(device.is_cuda() ? getCUDABackend() : getROCmBackend(), nullptr);
        auto &pool = GPUDeviceContextPool::instance();
        auto &context = device.is_cuda() ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
        context.submitAndWait(prove);
    }
}

/**
 * @brief Ordinary admission initializes and resets the existing GPU controller.
 *
 * No MTP capacity, predictor, or synthetic inference is needed to admit an
 * ordinary request. Inspect only at deliberate test boundaries. Twenty resets
 * prove that an untouched admission is retired through the same event-owned
 * lifecycle as an executed request, with stable physical buffer addresses.
 */
TEST_P(PipelineMTPStateOwnership, NonSpeculativeAdmissionNeedsNoPredictor)
{
    if (GetParam() == "CPU")
        return; // CPU generation has no device controller or native graph bank.
    using namespace sampling_math;
    initCPUBackend(-1);
    const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    ASSERT_NE(device.is_cuda() ? getCUDABackend() : getROCmBackend(), nullptr);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        auto config = stateConfig(device);
        config.mtp.enabled = false;
        config.mtp.graph_capacity_draft_tokens = 0;
        auto graph = std::make_shared<Qwen35Graph>(config, nullptr);
        DeviceGraphOrchestrator runner(graph, nullptr);
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 64, device));
        const auto &state = runner.inferenceState();
        ASSERT_TRUE(state.mtp_kv_caches.empty());
        const auto control = std::dynamic_pointer_cast<INT32Tensor>(
            state.extension_buffers.at(BufferId::MTP_GENERATION_CONTROL));
        const auto response = std::dynamic_pointer_cast<INT32Tensor>(
            state.extension_buffers.at(BufferId::MTP_GENERATION_RESPONSE_TOKENS));
        ASSERT_TRUE(control && response);
        const void *const control_address = control->gpu_data_ptr();
        const void *const response_address = response->gpu_data_ptr();
        ASSERT_NE(control_address, nullptr);
        ASSERT_NE(response_address, nullptr);
        // Typed access itself is not a transfer. This deliberate diagnostic
        // readback follows the production tensor's exact initialization event.
        ASSERT_TRUE(control->ensureOnHost(context.defaultStream()));
        ASSERT_TRUE(response->ensureOnHost(context.defaultStream()));
        for (int i = 0; i < kDeviceGenerationControlCount; ++i)
            ASSERT_EQ(control->typed_data()[i], 0);
        for (int i = 0; i < config.max_seq_len; ++i)
            ASSERT_EQ(response->typed_data()[i], -1);

        // Admission requires a real prepared workspace generation. Use DGO's
        // canonical host interface to materialize its existing KV workspace;
        // no model arithmetic or forged generation number is necessary.
        EXPECT_FALSE(runner.beginDeviceResidentGeneration({
            .request_count = 1, .max_new_tokens = 17,
            .depth_policy = DeviceGenerationPolicy::ordinary()}));
        IForwardExecutionHost &forward_host = runner;
        ASSERT_TRUE(forward_host.ensureDeviceWorkspaceAllocated(ComputeGraph{}, 1));

        // The ordinary algorithm must not grant missing speculative capacity.
        EXPECT_FALSE(runner.beginDeviceResidentGeneration({
            .request_count = 1, .max_new_tokens = 8,
            .depth_policy = DeviceGenerationPolicy::fixed(1)}));
        void *frontier_event_identity = nullptr;
        for (int reset = 0; reset < 20; ++reset)
        {
            SCOPED_TRACE(reset);
            const bool forward_only = reset % 2 != 0;
            const DeviceGenerationAdmissionRequest request{
                .request_count = 1, .max_new_tokens = forward_only ? 0 : 17,
                .depth_policy = forward_only ? DeviceGenerationPolicy::forwardOnly()
                                            : DeviceGenerationPolicy::ordinary(),
                .initial_leading_row_disposition = forward_only
                    ? DeviceGenerationLeadingRowDisposition::AlreadyEmitted
                    : DeviceGenerationLeadingRowDisposition::PendingResponse};
            ASSERT_TRUE(runner.beginDeviceResidentGeneration(request));
            EXPECT_FALSE(runner.beginDeviceResidentGeneration(request));
            if (!forward_only)
            {
                const auto frontier = runner.deviceResidentLogicalSequenceState();
                ASSERT_TRUE(frontier.coversRequest(0));
                ASSERT_NE(frontier.ready_event, nullptr);
                if (!frontier_event_identity)
                    frontier_event_identity = frontier.ready_event;
                EXPECT_EQ(frontier.ready_event, frontier_event_identity)
                    << "Request reset must reuse the setup-owned publication event.";
                // No model was run: the initial request must have an empty
                // canonical cache and no fictitious pending input token.
                int32_t position = -77, token = -77, healthy = -77;
                auto *const backend = device.is_cuda() ? getCUDABackend() : getROCmBackend();
                ASSERT_TRUE(backend->deviceToHost(&position, frontier.target_positions_device,
                    sizeof(position), device.gpu_ordinal(), context.defaultStream()));
                ASSERT_TRUE(backend->deviceToHost(&token, frontier.next_condition_tokens_device,
                    sizeof(token), device.gpu_ordinal(), context.defaultStream()));
                ASSERT_TRUE(backend->deviceToHost(&healthy, frontier.publication_ok_flags_device,
                    sizeof(healthy), device.gpu_ordinal(), context.defaultStream()));
                EXPECT_EQ(position, 0);
                EXPECT_EQ(token, -1);
                EXPECT_EQ(healthy, 1);
            }
            ASSERT_TRUE(control->ensureOnHost(context.defaultStream()));
            const int *const row = control->typed_data();
            EXPECT_EQ(row[kDeviceGenerationControlOk], 1);
            EXPECT_EQ(row[kDeviceGenerationControlDepthPolicyMode], static_cast<int>(request.depth_policy.mode));
            EXPECT_EQ(row[kDeviceGenerationControlRemainingTokenCount], request.max_new_tokens);
            EXPECT_EQ(row[kDeviceGenerationControlTransactionCount], 0);
            EXPECT_EQ(row[kDeviceGenerationControlCurrentDraftDepth], 0);
            EXPECT_EQ(row[kDeviceGenerationControlActiveVerifierRowCount], 0);
            EXPECT_EQ(control->gpu_data_ptr(), control_address);
            EXPECT_EQ(response->gpu_data_ptr(), response_address);
            runner.resetInferenceState(InferenceStateResetRequest::requestBoundary(
                "non-speculative-admission-replay"));
        }
    });
}

/**
 * @test Ordinary admission changes seed data, never the captured sampler identity.
 *
 * Capture once, then alternate request widths, high-bit seeds and seedless
 * admission twenty times. Real GPU draws must match the shared scalar oracle;
 * inactive rows must be invalid rather than inheriting the preceding request.
 */
TEST_P(PipelineMTPStateOwnership, OrdinarySamplingRetainsSeedBankWithoutMTP)
{
    if (GetParam() == "CPU")
        return; // CPU sampling has no resident random-seed bank.
    initCPUBackend(-1);
    const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto *const backend = device.is_cuda() ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        auto config = stateConfig(device);
        config.mtp.enabled = false;
        config.mtp.graph_capacity_draft_tokens = 0;
        config.mtp.max_request_batch = 3;
        auto graph = std::make_shared<Qwen35Graph>(config, nullptr);
        DeviceGraphOrchestrator runner(graph, nullptr);
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(3, 64, device));
        const auto &buffers = runner.inferenceState().extension_buffers;
        ASSERT_TRUE(buffers.contains(BufferId::SAMPLING_REQUEST_SEEDS));
        const auto seeds = std::dynamic_pointer_cast<INT32Tensor>(buffers.at(BufferId::SAMPLING_REQUEST_SEEDS));
        ASSERT_TRUE(seeds);
        ASSERT_NE(seeds->gpu_data_ptr(), nullptr);
        EXPECT_EQ(seeds->numel() * sizeof(int32_t), 3u * sizeof(uint64_t));
        ASSERT_TRUE(seeds->deviceValid());
        ASSERT_TRUE(seeds->ensureOnHost(context.defaultStream()));
        EXPECT_TRUE(std::all_of(seeds->typed_data(), seeds->typed_data() + seeds->numel(),
            [](int32_t word) { return word == 0; }));
        const auto *const seed_address = static_cast<const uint64_t *>(seeds->gpu_data_ptr());
        IForwardExecutionHost &forward_host = runner;
        ASSERT_TRUE(forward_host.ensureDeviceWorkspaceAllocated(ComputeGraph{}, 1));

        const auto ids = std::dynamic_pointer_cast<INT32Tensor>(buffers.at(BufferId::STOCHASTIC_TARGET_TOKEN_IDS));
        const auto probs = std::dynamic_pointer_cast<FP32Tensor>(buffers.at(BufferId::STOCHASTIC_TARGET_PROBS));
        const auto position = std::dynamic_pointer_cast<INT32Tensor>(buffers.at(BufferId::MTP_POSITION_IDS));
        const auto samples = std::dynamic_pointer_cast<INT32Tensor>(buffers.at(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS));
        ASSERT_TRUE(ids && probs && position && samples);
        const std::array<int32_t, 2> support{3, 7};
        const std::array<float, 2> probabilities{0.5F, 0.5F};
        std::memcpy(ids->raw_mutable_data(), support.data(), sizeof(support));
        std::memcpy(probs->raw_mutable_data(), probabilities.data(), sizeof(probabilities));
        const int32_t logical_position = 17;
        std::memcpy(position->raw_mutable_data(), &logical_position, sizeof(logical_position));
        for (const auto &tensor : {std::static_pointer_cast<TensorBase>(ids),
                 std::static_pointer_cast<TensorBase>(probs), std::static_pointer_cast<TensorBase>(position)})
            TransferEngine::prepareDeviceInput(tensor.get(), device, context.defaultStream());
        auto capture = context.createGraphCapture(context.defaultStream());
        ASSERT_TRUE(capture && capture->beginCapture());
        for (int row = 0; row < 3; ++row)
            ASSERT_TRUE(backend->enqueueSampleDistributionF32Device(
                ids->gpu_data_ptr(), probs->gpu_data_ptr(), 2, 0.0F, device.gpu_ordinal(),
                context.defaultStream(), static_cast<int32_t *>(samples->gpu_data_ptr()) + row,
                nullptr, 0, position->gpu_data_ptr(), 0, seed_address + row));
        ASSERT_TRUE(capture->endCapture() && capture->instantiate());

        for (int reset = 0; reset < 20; ++reset)
        {
            SCOPED_TRACE(reset);
            const int rows = reset % 3 + 1;
            std::vector<uint64_t> admitted(rows);
            for (int row = 0; row < rows; ++row)
                admitted[row] = (uint64_t{1} << 63) + 19 + 97 * reset + row;
            DeviceGenerationAdmissionRequest request{
                .request_count = rows, .max_new_tokens = 9,
                .depth_policy = sampling_math::DeviceGenerationPolicy::ordinary()};
            if (reset % 4 != 3)
                request.sampling_seeds.emplace(admitted);
            else
                admitted.assign(rows, 0); // Greedy/no-seed admission must clear preceding randomness.
            auto malformed = request;
            malformed.request_count = 4;
            EXPECT_FALSE(runner.beginDeviceResidentGeneration(malformed));
            ASSERT_TRUE(runner.beginDeviceResidentGeneration(request));
            // The request's caller-owned storage may disappear immediately.
            request.sampling_seeds.reset();
            ASSERT_EQ(seeds->gpu_data_ptr(), seed_address);
            TransferEngine::requireDeviceInput(seeds.get(), device, context.defaultStream());
            ASSERT_TRUE(capture->launch());
            TransferEngine::publishDeviceWrite(samples, device, context.defaultStream());
            ASSERT_TRUE(samples->ensureOnHost(context.defaultStream()));
            ASSERT_TRUE(seeds->ensureOnHost(context.defaultStream()));
            std::array<uint64_t, 3> observed{};
            std::memcpy(observed.data(), seeds->typed_data(), sizeof(observed));
            for (int row = 0; row < 3; ++row)
            {
                const uint64_t seed = row < rows ? admitted[row] : 0;
                EXPECT_EQ(observed[row], seed);
                const int expected = seed == 0 ? -1 : sampling_math::sample_distribution_with_threshold(
                    support.data(), probabilities.data(), 2,
                    sampling_math::mtp_spec_threshold_from_seed(seed, logical_position, 0));
                EXPECT_EQ(samples->typed_data()[row], expected);
            }
            runner.resetInferenceState(InferenceStateResetRequest::requestBoundary(
                "ordinary-seed-admission-reset"));
            ASSERT_TRUE(seeds->deviceValid());
            ASSERT_TRUE(seeds->ensureOnHost(context.defaultStream()));
            EXPECT_TRUE(std::all_of(seeds->typed_data(), seeds->typed_data() + seeds->numel(),
                [](int32_t word) { return word == 0; }));
        }
    });
}

/**
 * @test Every admitted request owns fresh history, independently of MTP enablement.
 *
 * Exercise actual arena construction and reset, not a sampler-owned allocation.
 * Both ordinary request reset and prefix replacement must clear every retained
 * row without replacing the address embedded in captured consumers.
 */
TEST_P(PipelineMTPStateOwnership, SamplingHistoryIsRequestOwnedWithoutMTP)
{
    if (GetParam() == "CPU")
        return; // CPU sampling owns its host history rather than this GPU bank.
    initCPUBackend(-1);
    const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto *backend = device.is_cuda() ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        for (const bool mtp : {false, true})
        for (const int requests : {1, 3})
        {
            SCOPED_TRACE(::testing::Message() << "mtp=" << mtp << " requests=" << requests);
            auto config = stateConfig(device);
            config.mtp.enabled = mtp;
            config.mtp.max_request_batch = requests;
            auto graph = std::make_shared<Qwen35Graph>(config, nullptr);
            DeviceGraphOrchestrator runner(graph, nullptr);
            ASSERT_TRUE(runner.initializeInferenceStateFromArena(requests, 64, device));
            const auto history = std::dynamic_pointer_cast<INT32Tensor>(
                runner.inferenceState().extension_buffers.at(BufferId::MTP_GENERATED_TOKEN_COUNTS));
            ASSERT_TRUE(history);
            ASSERT_EQ(history->shape(), (std::vector<size_t>{size_t(requests), size_t(config.vocab_size)}));
            void *const address = history->gpu_data_ptr();
            ASSERT_NE(address, nullptr);
            // Require initialization to have published coherence before any
            // request/reset. Raw zero-looking allocator bytes are not proof.
            ASSERT_TRUE(history->deviceValid());
            ASSERT_TRUE(history->ensureOnHost(context.defaultStream()));
            const size_t elements = size_t(requests) * size_t(config.vocab_size);
            EXPECT_TRUE(std::all_of(history->typed_data(), history->typed_data() + elements,
                                    [](int32_t count) { return count == 0; }));
            std::vector<int32_t> observed(elements, -1);
            for (int replay = 0; replay < 20; ++replay)
            {
                SCOPED_TRACE(replay);
                ASSERT_TRUE(backend->memset(address, 1, elements * sizeof(int32_t),
                                           device.gpu_ordinal(), context.defaultStream()));
                // Verify poison really reached all rows. The subsequent read
                // bypasses any cached host copy so a missing reset cannot pass.
                ASSERT_TRUE(backend->deviceToHost(observed.data(), address,
                    elements * sizeof(int32_t), device.gpu_ordinal(), context.defaultStream()));
                ASSERT_TRUE(std::all_of(observed.begin(), observed.end(),
                                       [](int32_t count) { return count == 0x01010101; }));
                runner.resetInferenceState(replay % 2 == 0
                    ? InferenceStateResetRequest::requestBoundary("sampling-history-new-request")
                    : InferenceStateResetRequest::prefixRestoreBoundary("sampling-history-prefix", true));
                ASSERT_TRUE(history->ensureOnHost(context.defaultStream()));
                ASSERT_TRUE(backend->deviceToHost(observed.data(), address,
                    elements * sizeof(int32_t), device.gpu_ordinal(), context.defaultStream()));
                EXPECT_TRUE(std::all_of(observed.begin(), observed.end(),
                                       [](int32_t count) { return count == 0; }));
                EXPECT_EQ(history->gpu_data_ptr(), address);
            }
        }
    });
}

/** @test Ordinary controller admission publishes the same stop/penalty banks as MTP. */
TEST_P(PipelineMTPStateOwnership, OrdinaryAdmissionPublishesRequestSamplingPolicy)
{
    if (GetParam() == "CPU")
        return;
    initCPUBackend(-1);
    const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    ASSERT_NE(device.is_cuda() ? getCUDABackend() : getROCmBackend(), nullptr);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        auto config = stateConfig(device);
        config.mtp.enabled = false;
        auto graph = std::make_shared<Qwen35Graph>(config, nullptr);
        DeviceGraphOrchestrator runner(graph, nullptr);
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 64, device));
        IForwardExecutionHost &forward_host = runner;
        ASSERT_TRUE(forward_host.ensureDeviceWorkspaceAllocated(ComputeGraph{}, 1));
        const auto &buffers = runner.inferenceState().extension_buffers;
        const auto stop = std::dynamic_pointer_cast<INT32Tensor>(buffers.at(BufferId::MTP_VERIFIER_STOP_TOKENS));
        const auto policy = std::dynamic_pointer_cast<INT32Tensor>(buffers.at(BufferId::MTP_GREEDY_PENALTY_POLICY));
        ASSERT_TRUE(stop && policy);
        const void *const stop_address = stop->gpu_data_ptr();
        const void *const policy_address = policy->gpu_data_ptr();
        for (int replay = 0; replay < 20; ++replay)
        {
            SCOPED_TRACE(replay);
            const std::vector<int32_t> stops = replay % 2 ? std::vector<int32_t>{} : std::vector<int32_t>{3, 7};
            const MTPRequestPenaltyPolicy requested{.presence_penalty = replay % 3 ? -0.75F : 0.0F,
                .frequency_penalty = replay % 2 ? 0.125F : 0.0F};
            ASSERT_TRUE(runner.configureMTPRequestStopTokens(stops));
            ASSERT_TRUE(runner.configureMTPRequestPenaltyPolicy(requested));
            ASSERT_TRUE(runner.beginDeviceResidentGeneration({.request_count = 1, .max_new_tokens = 9,
                .depth_policy = sampling_math::DeviceGenerationPolicy::ordinary()}));
            ASSERT_TRUE(stop->deviceValid());
            ASSERT_TRUE(policy->deviceValid());
            ASSERT_TRUE(stop->ensureOnHost(context.defaultStream()));
            ASSERT_TRUE(policy->ensureOnHost(context.defaultStream()));
            for (int i = 0; i < sampling_math::kSpeculativeBatchMaxStopTokens; ++i)
                EXPECT_EQ(stop->typed_data()[i], size_t(i) < stops.size() ? stops[size_t(i)] : -1);
            MTPGreedyPenaltyPolicy observed;
            std::memcpy(&observed, policy->typed_data(), sizeof(observed));
            EXPECT_EQ(observed.presence_penalty, requested.presence_penalty);
            EXPECT_EQ(observed.frequency_penalty, requested.frequency_penalty);
            EXPECT_EQ(observed.enabled, requested.enabled() ? 1 : 0);
            EXPECT_EQ(observed.first_token_already_in_history, 0);
            EXPECT_EQ(stop->gpu_data_ptr(), stop_address);
            EXPECT_EQ(policy->gpu_data_ptr(), policy_address);
            runner.resetInferenceState(replay % 2 == 0
                ? InferenceStateResetRequest::requestBoundary("ordinary-policy-new-request")
                : InferenceStateResetRequest::prefixRestoreBoundary("ordinary-policy-prefix", true));
        }
    });
}

INSTANTIATE_TEST_SUITE_P(Backends, PipelineMTPStateOwnership,
    ::testing::Values("CPU"
#ifdef HAVE_CUDA
        , "CUDA"
#endif
#ifdef HAVE_ROCM
        , "ROCm"
#endif
    ), [](const auto &info) { return info.param; });
}
