/**
 * @file Test__OrdinaryGenerationSampling.cpp
 * @brief Captured ordinary sampling versus the established serial sampling law.
 *
 * CUDA and HIP run the same production stage with persistent bindings. Tests
 * vary request admission, stop bytes and live position without recapture and
 * compare every token and controller word. This certifies the sampler boundary,
 * not a real-model pipeline: model/server certification remains separate.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/compute_stages/stages/OrdinaryGenerationSamplingStage.h"
#include "execution/mtp/DeviceGenerationContract.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "execution/mtp/MTPVerifierOutcomeGraph.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace llaminar2::sampling_math;
using Stage = OrdinaryGenerationSamplingStage;

/** @brief Separate backend processes prevent mixed-runtime fixture ownership. */
class OrdinaryDeviceGenerationSampling : public ::testing::TestWithParam<std::string> {};

/** @test The seed-to-publication boundary reuses one two-node capture, including fatal admission. */
TEST_P(OrdinaryDeviceGenerationSampling, ResidentSeedResetRetainsGraph)
{
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &pool = GPUDeviceContextPool::instance();
    auto &context = cuda ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
    context.submitAndWait([&]
    {
        void *const stream = context.defaultStream();
        const auto release = [backend](void *p) { if (p) backend->free(p, 0); };
        // One test-owned persistent block, with naturally aligned typed views.
        auto storage = std::unique_ptr<void, decltype(release)>(backend->allocate(1024, 0), release);
        ASSERT_TRUE(storage);
        auto *seed = static_cast<uint64_t *>(storage.get());
        auto *tokens = reinterpret_cast<int32_t *>(seed + 1);
        auto *probs = reinterpret_cast<float *>(tokens + 2);
        auto *live = reinterpret_cast<int32_t *>(probs + 2);
        auto *sample = live + 4;
        auto *response = sample + 1;
        auto *control = response + 1;
        auto *history = control + kDeviceGenerationControlCount;
        const int32_t host_tokens[] = {11, 19};
        const float host_probs[] = {0.5F, 0.5F};
        ASSERT_TRUE(backend->hostToDevice(tokens, host_tokens, sizeof(host_tokens), 0, stream));
        ASSERT_TRUE(backend->hostToDevice(probs, host_probs, sizeof(host_probs), 0, stream));
        const OrdinaryGenerationPublication publication{
            .request_count = 1, .sampled_tokens = sample,
            .source = OrdinaryGenerationSampleSource::PrefillLogits,
            .response_tokens = response, .response_token_stride = 1,
            .control = control, .control_stride = kDeviceGenerationControlCount,
            .frontier = {.cached_tokens = live, .next_condition_tokens = live + 1,
                .stopped_flags = live + 2, .publication_ok_flags = live + 3,
                .request_capacity = 1, .context_capacity = 8192},
            .history = {history, 32, 32, 1}};
        auto graph = context.createGraphCapture(stream);
        ASSERT_TRUE(graph && graph->beginCapture());
        ASSERT_TRUE(backend->enqueueSampleDistributionF32Device(
            tokens, probs, 2, 0, 0, stream, sample, nullptr, 0, live, 0, seed));
        ASSERT_TRUE(backend->enqueuePublishOrdinaryGenerationSample(publication, 0, stream));
        ASSERT_TRUE(graph->endCapture() && graph->instantiate());
        ASSERT_EQ(graph->nodeCount(), 2u);
        // Keep the same executable for valid reset, malformed random state,
        // invalid history and out-of-range sampled IDs. No rejection may
        // partially publish a response or write beyond its histogram row.
        struct AdmissionCase {
            uint64_t seed;
            int32_t initial_count;
            bool invalid_token;
        };
        for (const auto admission : std::array<AdmissionCase, 6>{{
                 {19, 0, false}, {142, 3, false}, {0, 0, false},
                 {19, -1, false}, {19, INT32_MAX, false}, {19, 0, true}}})
        for (int repeat = 0; repeat < 20; ++repeat) {
            SCOPED_TRACE(::testing::Message() << "seed=" << admission.seed << " initial_count="
                << admission.initial_count << " invalid_token=" << admission.invalid_token << " reset=" << repeat);
            const int32_t initial[] = {7, 99, 0, 1};
            std::array<int32_t, 32> expected_history{};
            expected_history[11] = expected_history[19] = admission.initial_count;
            const int32_t admitted_tokens[] = {admission.invalid_token ? 32 : 11,
                admission.invalid_token ? 32 : 19};
            ASSERT_TRUE(backend->hostToDevice(tokens, admitted_tokens, sizeof(admitted_tokens), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(history, expected_history.data(), sizeof(expected_history), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(seed, &admission.seed, sizeof(admission.seed), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(live, initial, sizeof(initial), 0, stream));
            ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(1, 1, DeviceGenerationPolicy::ordinary(),
                DeviceGenerationLeadingRowDisposition::PendingResponse, 1, response,
                kDeviceGenerationControlCount, control, 0, stream));
            ASSERT_TRUE(graph->launch());
            std::array<int32_t, kDeviceGenerationControlCount> observed_control;
            std::array<int32_t, 4> observed_live;
            ASSERT_TRUE(backend->deviceToHost(observed_control.data(), control, sizeof(observed_control), 0, stream));
            ASSERT_TRUE(backend->deviceToHost(observed_live.data(), live, sizeof(observed_live), 0, stream));
            const bool valid = admission.seed && admission.initial_count >= 0 &&
                admission.initial_count < INT32_MAX && !admission.invalid_token;
            if (valid) {
                const int expected = sample_distribution_with_threshold_and_probability(
                    host_tokens, host_probs, 2, mtp_spec_threshold_from_seed(admission.seed, 7, 0), nullptr);
                EXPECT_EQ(observed_live, (std::array<int32_t, 4>{7, expected, 0, 1}));
                EXPECT_EQ(observed_control[kDeviceGenerationControlResponseTokenCount], 1);
                EXPECT_EQ(observed_control[kDeviceGenerationControlOk], 1);
                ++expected_history[expected];
            } else {
                EXPECT_EQ(observed_live, (std::array<int32_t, 4>{7, 99, 0, 0}));
                EXPECT_EQ(observed_control[kDeviceGenerationControlResponseTokenCount], 0);
                EXPECT_EQ(observed_control[kDeviceGenerationControlOk], 0);
                EXPECT_EQ(observed_control[kDeviceGenerationControlErrorCode], int(DeviceGenerationError::InvalidOrdinarySample));
            }
            // A second replay is absorbing for success and failure alike.
            ASSERT_TRUE(graph->launch());
            std::array<int32_t, 32> observed_history;
            ASSERT_TRUE(backend->deviceToHost(observed_history.data(), history, sizeof(observed_history), 0, stream));
            EXPECT_EQ(observed_history, expected_history);
        }
    });
}

/** @test Ordinary greedy history shares verifier arithmetic without fabricated branch data. */
TEST_P(OrdinaryDeviceGenerationSampling, CommittedPenaltyArgmaxMatchesSerialArithmetic)
{
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &pool = GPUDeviceContextPool::instance();
    auto &context = cuda ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
    context.submitAndWait([&]
    {
        void *const stream = context.defaultStream();
        const auto release = [backend](void *p) { if (p) backend->free(p, 0); };
        const auto allocate = [&](size_t bytes)
        { return std::unique_ptr<void, decltype(release)>(backend->allocate(bytes, 0), release); };
        constexpr int max_vocab = 152064, partial_capacity = 128;
        auto logits = allocate(max_vocab * sizeof(float));
        auto counts = allocate(max_vocab * sizeof(int32_t));
        auto policy = allocate(sizeof(MTPGreedyPenaltyPolicy));
        auto values = allocate(partial_capacity * sizeof(float));
        auto indices = allocate(partial_capacity * sizeof(int32_t));
        auto output = allocate(2 * sizeof(int32_t));
        ASSERT_TRUE(logits && counts && policy && values && indices && output);
        std::vector<float> host_logits(max_vocab);
        std::vector<int32_t> host_counts(max_vocab);
        for (int token = 0; token < max_vocab; ++token) {
            // Binary fractions make the independent host oracle exactly
            // representable, including negative rewards and lowest-ID ties.
            host_logits[token] = float(token % 97) * 0.0625F;
            host_counts[token] = token % 7;
        }
        ASSERT_TRUE(backend->hostToDevice(logits.get(), host_logits.data(), max_vocab * sizeof(float), 0, stream));
        ASSERT_TRUE(backend->hostToDevice(counts.get(), host_counts.data(), max_vocab * sizeof(int32_t), 0, stream));
        const auto history = GenerationPenaltyHistory::committed(counts.get(), policy.get());
        for (int vocab : {513, 4096, max_vocab}) {
            SCOPED_TRACE(::testing::Message() << GetParam() << " vocab=" << vocab);
            // A single committed histogram cannot impersonate multiple
            // independent requests or a verifier's branch history.
            EXPECT_FALSE(backend->enqueueArgmaxF32RowsWithHistoryDevice(logits.get(), 2, vocab, history,
                0, stream, output.get(), static_cast<int32_t *>(output.get()) + 1,
                values.get(), indices.get(), partial_capacity));
            auto graph = context.createGraphCapture(stream);
            ASSERT_TRUE(graph && graph->beginCapture());
            ASSERT_TRUE(backend->enqueueArgmaxF32RowsWithHistoryDevice(logits.get(), 1, vocab, history,
                0, stream, output.get(), static_cast<int32_t *>(output.get()) + 1,
                values.get(), indices.get(), partial_capacity));
            ASSERT_TRUE(graph->endCapture());
            EXPECT_EQ(graph->nodeCount(), 2u) << "History is fused into the existing two-pass argmax";
            std::vector<GPUGraphKernelNodeInfo> kernels;
            std::string error;
            ASSERT_TRUE(graph->inspectKernelNodes(kernels, &error)) << error;
            for (const auto &kernel : kernels) {
                EXPECT_EQ(kernel.local_memory_bytes_per_thread, 0u) << kernel.name;
                EXPECT_GT(kernel.max_active_blocks_per_sm, 0u) << kernel.name;
            }
            ASSERT_TRUE(graph->instantiate());
            for (int reset = 0; reset < 20; ++reset) {
                const float presence = reset % 3 == 0 ? 0 : (reset % 3 == 1 ? 0.75F : -0.5F);
                const float frequency = reset % 3 == 0 ? 0 : (reset % 3 == 1 ? 0.125F : -0.25F);
                ASSERT_TRUE(backend->enqueueConfigureMTPGreedyPenaltyPolicyDevice(policy.get(),
                    presence, frequency, false, 0, stream));
                ASSERT_TRUE(graph->launch());
                float actual_value;
                int32_t actual_token;
                ASSERT_TRUE(backend->deviceToHost(&actual_value, output.get(), sizeof(actual_value), 0, stream));
                ASSERT_TRUE(backend->deviceToHost(&actual_token, static_cast<int32_t *>(output.get()) + 1,
                    sizeof(actual_token), 0, stream));
                float best = -std::numeric_limits<float>::infinity();
                int best_token = -1;
                for (int token = 0; token < vocab; ++token) {
                    const float value = host_logits[token] - (host_counts[token] > 0
                        ? presence + frequency * float(host_counts[token]) : 0.0F);
                    if (value > best) { best = value; best_token = token; }
                }
                EXPECT_EQ(actual_value, best);
                EXPECT_EQ(actual_token, best_token);
            }
        }
    });
}

/** @test Real seeded/greedy samplers replay without host token or position publication. */
TEST_P(OrdinaryDeviceGenerationSampling, CapturedSamplesMatchSerialLawAcrossReset)
{
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &pool = GPUDeviceContextPool::instance();
    auto &context = cuda ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
    const DeviceId device = cuda ? DeviceId::cuda(0) : DeviceId::rocm(0);
    context.submitAndWait([&]
    {
        void *const stream = context.defaultStream();
        ASSERT_NE(stream, nullptr);
        auto execution = IDeviceContext::create(device);
        ASSERT_TRUE(execution);
        const auto release = [backend](void *p) { if (p) backend->free(p, 0); };
        const auto allocate = [&](size_t bytes)
        { return std::unique_ptr<void, decltype(release)>(backend->allocate(bytes, 0), release); };
        constexpr int requests = 3, vocab = 513, stride = vocab + 7, budget = 17;
        constexpr int response_stride = budget + 3, stop_stride = 5;
        constexpr int scratch_capacity = 64 * kMaxTopK;
        std::vector<float> logits(requests * stride, -999.0F);
        for (int request = 0; request < requests; ++request)
            for (int token = 0; token < vocab; ++token)
                logits[request * stride + token] = std::sin(float(token * 7 + request * 31) * 0.13F);
        // A real tie proves the canonical lower-token argmax rule as well.
        for (int request = 0; request < requests; ++request)
            logits[request * stride + 11] = logits[request * stride + 19] = 2.0F;
        auto d_logits = allocate(logits.size() * sizeof(float));
        auto d_scratch_values = allocate(scratch_capacity * sizeof(float));
        auto d_scratch_indices = allocate(scratch_capacity * sizeof(int32_t));
        auto d_values = allocate(kMaxTopK * sizeof(float));
        auto d_indices = allocate(kMaxTopK * sizeof(int32_t));
        auto d_samples = allocate(requests * sizeof(int32_t));
        auto d_live = allocate(4 * requests * sizeof(int32_t));
        auto d_control = allocate(requests * kDeviceGenerationControlCount * sizeof(int32_t));
        auto d_response = allocate(requests * response_stride * sizeof(int32_t));
        auto d_stops = allocate(requests * stop_stride * sizeof(int32_t));
        auto d_seeds = allocate(requests * sizeof(uint64_t));
        auto d_processed = allocate(logits.size() * sizeof(float));
        constexpr int history_stride = vocab + 3;
        auto d_history = allocate(requests * history_stride * sizeof(int32_t));
        auto d_fresh_logits = allocate(logits.size() * sizeof(float));
        auto d_penalties = allocate(sizeof(MTPGreedyPenaltyPolicy));
        ASSERT_TRUE(d_logits && d_scratch_values && d_scratch_indices && d_values && d_indices &&
            d_samples && d_live && d_control && d_response && d_stops && d_seeds && d_processed && d_history &&
            d_fresh_logits && d_penalties);
        ASSERT_TRUE(backend->hostToDevice(d_logits.get(), logits.data(), logits.size() * sizeof(float), 0, stream));
        ASSERT_TRUE(backend->hostToDevice(d_fresh_logits.get(), logits.data(), logits.size() * sizeof(float), 0, stream));
        auto *live = static_cast<int32_t *>(d_live.get());
        const OrdinaryGenerationPublication publication{
            .request_count = requests, .sampled_tokens = static_cast<int32_t *>(d_samples.get()),
            .stop_tokens = {static_cast<const int32_t *>(d_stops.get()), 3, stop_stride},
            .source = OrdinaryGenerationSampleSource::DecodeLogits,
            .response_tokens = static_cast<int32_t *>(d_response.get()), .response_token_stride = response_stride,
            .control = static_cast<int *>(d_control.get()), .control_stride = kDeviceGenerationControlCount,
            .frontier = {.cached_tokens = live, .next_condition_tokens = live + requests,
                .stopped_flags = live + 2 * requests, .publication_ok_flags = live + 3 * requests,
                .request_capacity = requests, .context_capacity = 8192},
            .history = {static_cast<int32_t *>(d_history.get()), vocab, history_stride, requests}};
        Stage::Params params{
            .device_id = device, .backend = backend, .logits = static_cast<float *>(d_logits.get()),
            .vocab_size = vocab, .logits_row_stride = stride,
            .workspace = {.values = static_cast<float *>(d_values.get()),
                .indices = static_cast<int32_t *>(d_indices.get()), .capacity = kMaxTopK,
                .partial_values = static_cast<float *>(d_scratch_values.get()),
                .partial_indices = static_cast<int32_t *>(d_scratch_indices.get()), .partial_capacity = scratch_capacity},
            .publication = publication};

        for (bool apply_penalties : {false, true})
        for (int top_k : {0, 1, 7, 32, 192, 193, 255, 256})
        {
            SCOPED_TRACE(::testing::Message() << GetParam() << " top_k=" << top_k << " penalties=" << apply_penalties);
            ASSERT_TRUE(backend->hostToDevice(d_logits.get(), logits.data(), logits.size() * sizeof(float), 0, stream));
            params.penalties = apply_penalties ? static_cast<const MTPGreedyPenaltyPolicy *>(d_penalties.get()) : nullptr;
            if (top_k)
                params.policy = Stage::Stochastic{top_k, 0.87F, 0.73F, static_cast<const uint64_t *>(d_seeds.get())};
            else
                params.policy = Stage::Greedy{};
            Stage decode(params);
            decode.setGPUStream(stream);
            auto prefill_params = params;
            prefill_params.publication.source = OrdinaryGenerationSampleSource::PrefillLogits;
            Stage prefill(prefill_params);
            prefill.setGPUStream(stream);
            EXPECT_TRUE(decode.hasSameCaptureIdentity(params));
            EXPECT_FALSE(decode.hasSameCaptureIdentity(prefill_params));
            auto changed = params;
            changed.publication.stop_tokens.request_stride = 0;
            EXPECT_FALSE(decode.hasSameCaptureIdentity(changed));
            changed = params;
            changed.publication.sampled_tokens = live + requests;
            EXPECT_THROW(Stage{changed}, std::invalid_argument);
            changed = params;
            changed.publication.history.counts++;
            EXPECT_FALSE(decode.hasSameCaptureIdentity(changed));
            changed = params;
            changed.publication.history.vocab_size--;
            EXPECT_THROW(Stage{changed}, std::invalid_argument);
            changed = params;
            changed.penalties = apply_penalties ? nullptr : static_cast<const MTPGreedyPenaltyPolicy *>(d_penalties.get());
            EXPECT_FALSE(decode.hasSameCaptureIdentity(changed));
            if (top_k) {
                changed = params;
                std::get<Stage::Stochastic>(changed.policy).seeds++;
                EXPECT_FALSE(decode.hasSameCaptureIdentity(changed));
                std::get<Stage::Stochastic>(changed.policy).temperature = std::numeric_limits<float>::quiet_NaN();
                EXPECT_THROW(Stage{changed}, std::invalid_argument);
                EXPECT_FALSE(backend->enqueueSampleDistributionF32Device(
                    d_indices.get(), d_values.get(), top_k, 0, 0, stream, d_samples.get(), nullptr,
                    19, live, 1, static_cast<const uint64_t *>(d_seeds.get())))
                    << "Two seed authorities must be rejected before launch";
                EXPECT_FALSE(backend->enqueueSampleDistributionF32Device(
                    d_indices.get(), d_values.get(), top_k, 0, 0, stream, d_samples.get(), nullptr,
                    0, nullptr, 1, static_cast<const uint64_t *>(d_seeds.get())))
                    << "Resident seed requires the matching resident position";
            }
            auto prefill_graph = context.createGraphCapture(stream);
            ASSERT_TRUE(prefill_graph && prefill_graph->beginCapture());
            // This fixture substitutes an exact copy for the model's fresh
            // logits producer. Production forward overwrites its logits on
            // every transaction; testing reuse of already-penalized logits
            // would exercise a different, invalid producer contract.
            if (apply_penalties)
                ASSERT_TRUE(backend->deviceCopyAsync(d_logits.get(), d_fresh_logits.get(),
                    logits.size() * sizeof(float), 0, stream));
            ASSERT_TRUE(prefill.execute(execution.get()));
            ASSERT_TRUE(prefill_graph->endCapture());
            auto decode_graph = context.createGraphCapture(stream);
            ASSERT_TRUE(decode_graph && decode_graph->beginCapture());
            if (apply_penalties)
                ASSERT_TRUE(backend->deviceCopyAsync(d_logits.get(), d_fresh_logits.get(),
                    logits.size() * sizeof(float), 0, stream));
            ASSERT_TRUE(decode.execute(execution.get()));
            ASSERT_TRUE(decode_graph->endCapture());
            std::vector<GPUGraphKernelNodeInfo> kernels;
            std::string inspection_error;
            ASSERT_TRUE(decode_graph->inspectKernelNodes(kernels, &inspection_error)) << inspection_error;
            int publishers = 0;
            for (const auto &kernel : kernels) {
                if (kernel.name.find("publish_ordinary_generation_sample_kernel") != std::string::npos) {
                    ++publishers;
                    EXPECT_EQ(kernel.local_memory_bytes_per_thread, 0u) << kernel.name;
                    EXPECT_GT(kernel.max_active_blocks_per_sm, 0u);
                }
            }
            EXPECT_EQ(publishers, 1) << "One fused stop/response/frontier publication per transaction";
            if (top_k > 0) {
                // Exercise every public generic form at CUDA's 48-KiB boundary,
                // not only the compact distribution used by this new stage.
                std::vector<int> sorted(vocab);
                for (int i = 0; i < vocab; ++i) sorted[i] = i;
                std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
                });
                std::vector<int> selected(top_k);
                std::vector<float> selected_values(top_k);
                ASSERT_TRUE(backend->topKF32(d_logits.get(), vocab, top_k, 0,
                    selected_values.data(), selected.data(), stream));
                for (int i = 0; i < top_k; ++i) {
                    EXPECT_EQ(selected[i], sorted[i]);
                    EXPECT_FLOAT_EQ(selected_values[i], logits[sorted[i]]);
                }
                auto processed_graph = context.createGraphCapture(stream);
                ASSERT_TRUE(processed_graph && processed_graph->beginCapture());
                ASSERT_TRUE(backend->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    d_logits.get(), requests, vocab, stride, top_k, 1.0F, 1.0F, 0, stream,
                    d_processed.get(), stride, d_scratch_values.get(), d_scratch_indices.get(), scratch_capacity));
                ASSERT_TRUE(processed_graph->endCapture() && processed_graph->instantiate());
                ASSERT_TRUE(processed_graph->launch());
                std::vector<float> processed(logits.size());
                ASSERT_TRUE(backend->deviceToHost(processed.data(), d_processed.get(), processed.size() * sizeof(float), 0, stream));
                for (int i = 0; i < vocab; ++i) {
                    if (i < top_k) EXPECT_FLOAT_EQ(processed[sorted[i]], logits[sorted[i]]);
                    else EXPECT_EQ(processed[sorted[i]], -std::numeric_limits<float>::infinity());
                }
            }
            std::unique_ptr<IGPUGraphCapture> parent, prefill_parent;
            if (cuda) {
                parent = context.createGraphCapture(stream);
                const DeviceControlledLoopFragment fragments[] = {
                    {.name = "ordinary decode sampling", .capture = decode_graph.get()}};
                ASSERT_NE(parent, nullptr);
                std::string construction_error;
                ASSERT_TRUE(DeviceGenerationGraphProgram::native(*parent,
                    {.policy = DeviceGenerationPolicy::ordinary(), .rows = publication.control,
                     .stride = kDeviceGenerationControlCount, .requests = requests},
                    {.iteration = fragments}, construction_error)) << construction_error;
                prefill_parent = context.createGraphCapture(stream);
                ASSERT_NE(prefill_parent, nullptr);
                const DeviceControlledLoopFragment initialization[] = {
                    {.name = "sample prefill exactly once", .capture = prefill_graph.get()}};
                ASSERT_TRUE(DeviceGenerationGraphProgram::native(*prefill_parent,
                    {.policy = DeviceGenerationPolicy::ordinary(), .rows = publication.control,
                     .stride = kDeviceGenerationControlCount, .requests = requests},
                    {.initialization = initialization, .iteration = fragments}, construction_error)) << construction_error;
                EXPECT_FALSE(prefill_graph->hasExecutable());
                EXPECT_FALSE(decode_graph->hasExecutable()) << "Only the retained parent owns execution";
            } else {
                ASSERT_TRUE(prefill_graph->instantiate());
                ASSERT_TRUE(decode_graph->instantiate());
            }

            for (auto leading : {DeviceGenerationLeadingRowDisposition::PendingResponse,
                                 DeviceGenerationLeadingRowDisposition::AlreadyEmitted})
            {
                // A new request changes seed bytes behind the same graph-bound
                // address. Both source graphs and CUDA's parent remain retained.
                const uint64_t epoch = leading == DeviceGenerationLeadingRowDisposition::PendingResponse ? 0 : 100;
                const uint64_t seeds[] = {19 + epoch, 42 + epoch, 1234567 + epoch};
                ASSERT_TRUE(backend->hostToDevice(d_seeds.get(), seeds, sizeof(seeds), 0, stream));
                // Request values change without replacing their captured
                // address. Exercise positive penalties and negative rewards.
                const float presence = epoch ? -0.5F : 0.75F;
                const float frequency = epoch ? -0.25F : 0.125F;
                ASSERT_TRUE(backend->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                    d_penalties.get(), presence, frequency, false, 0, stream));
                // Freeze an independent serial-sampler oracle outside capture.
                // The existing serial API uses the exact public seed/offset law.
                std::array<std::array<int, budget>, requests> expected_samples{};
                for (int request = 0; request < requests; ++request)
                {
                    std::array<int, vocab> oracle_history{};
                    for (int row = 0; row < budget; ++row)
                    {
                        std::array<float, vocab> oracle_logits;
                        for (int token = 0; token < vocab; ++token)
                        {
                            oracle_logits[token] = logits[request * stride + token];
                            if (apply_penalties && oracle_history[token] != 0)
                                oracle_logits[token] -= presence + frequency * static_cast<float>(oracle_history[token]);
                        }
                        const int position = 7 + request + row +
                            (leading == DeviceGenerationLeadingRowDisposition::AlreadyEmitted ? 1 : 0);
                        int token = static_cast<int>(std::max_element(oracle_logits.begin(), oracle_logits.end()) - oracle_logits.begin());
                        if (top_k) {
                            ASSERT_TRUE(backend->hostToDevice(d_processed.get(), oracle_logits.data(),
                                sizeof(oracle_logits), 0, stream));
                            const auto &policy = std::get<Stage::Stochastic>(params.policy);
                            ASSERT_TRUE(backend->sampleTopKTopPF32(
                                static_cast<const float *>(d_processed.get()),
                                vocab, top_k, policy.top_p, policy.temperature, seeds[request],
                                static_cast<uint64_t>(position) * kMTPSpecDrawPurposesPerToken,
                                0, &token, stream));
                        }
                        expected_samples[request][row] = token;
                        ++oracle_history[token];
                    }
                }
                for (int repeat = 0; repeat < 20; ++repeat)
                {
                    SCOPED_TRACE(::testing::Message() << "repeat=" << repeat << " leading=" << int(leading));
                    std::array<int32_t, 4 * requests> initial{};
                    std::array<int32_t, requests * history_stride> initial_history{};
                    for (int request = 0; request < requests; ++request)
                        for (int pad = vocab; pad < history_stride; ++pad)
                            initial_history[request * history_stride + pad] = -77;
                    ASSERT_TRUE(backend->hostToDevice(d_history.get(), initial_history.data(),
                        sizeof(initial_history), 0, stream));
                    std::array<int32_t, requests * stop_stride> stops;
                    stops.fill(-1);
                    for (int request = 0; request < requests; ++request) {
                        initial[request] = 7 + request;
                        initial[requests + request] = 99;
                        initial[3 * requests + request] = 1;
                        if ((repeat + request) % 3 != 0)
                            stops[request * stop_stride + 1] = expected_samples[request][4];
                        stops[request * stop_stride + 3] = -77; // Ignored padding, not stop policy.
                    }
                    ASSERT_TRUE(backend->hostToDevice(d_live.get(), initial.data(), sizeof(initial), 0, stream));
                    ASSERT_TRUE(backend->hostToDevice(d_stops.get(), stops.data(), sizeof(stops), 0, stream));
                    ASSERT_TRUE(backend->memset(d_response.get(), 0xff, requests * response_stride * sizeof(int32_t), 0, stream));
                    ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(requests, budget,
                        DeviceGenerationPolicy::ordinary(), leading, response_stride, publication.response_tokens,
                        kDeviceGenerationControlCount, publication.control, 0, stream));
                    const int remaining = budget - (leading == DeviceGenerationLeadingRowDisposition::PendingResponse ? 1 : 0);
                    if (parent) {
                        // CUDA consumes live positions and independent EOS rows
                        // in one parent, including prefill sampling and an
                        // absorbing terminal replay. Admission selects the
                        // retained policy; no device state is observed here.
                        auto &program = leading == DeviceGenerationLeadingRowDisposition::PendingResponse
                            ? prefill_parent : parent;
                        ASSERT_TRUE(program->launch());
                        ASSERT_TRUE(program->launch());
                    } else {
                        // This bounded HIP test driver proves the retained stage,
                        // not a production scheduler. Authenticated ticket tests
                        // independently own the host-submission protocol proof.
                        if (leading == DeviceGenerationLeadingRowDisposition::PendingResponse)
                            ASSERT_TRUE(prefill_graph->launch());
                        for (int row = 0; row < remaining + 1; ++row)
                            ASSERT_TRUE(decode_graph->launch());
                    }
                    std::array<int32_t, requests * response_stride> actual_response;
                    std::array<int32_t, requests * kDeviceGenerationControlCount> actual_control;
                    std::array<int32_t, 4 * requests> actual_live;
                    std::array<int32_t, requests * history_stride> actual_history;
                    ASSERT_TRUE(backend->deviceToHost(actual_response.data(), d_response.get(), sizeof(actual_response), 0, stream));
                    ASSERT_TRUE(backend->deviceToHost(actual_control.data(), d_control.get(), sizeof(actual_control), 0, stream));
                    ASSERT_TRUE(backend->deviceToHost(actual_live.data(), d_live.get(), sizeof(actual_live), 0, stream));
                    ASSERT_TRUE(backend->deviceToHost(actual_history.data(), d_history.get(), sizeof(actual_history), 0, stream));
                    for (int request = 0; request < requests; ++request) {
                        int count = 0;
                        bool stopped = false;
                        for (; count < budget && !stopped; ++count) {
                            const int expected = expected_samples[request][count];
                            EXPECT_EQ(actual_response[request * response_stride + count], expected);
                            ++initial_history[request * history_stride + expected];
                            stopped = expected == stops[request * stop_stride + 1];
                        }
                        for (int row = count; row < response_stride; ++row)
                            EXPECT_EQ(actual_response[request * response_stride + row], -1);
                        const DeviceGenerationAdmissionRequest admission{
                            .request_count = requests, .max_new_tokens = budget,
                            .depth_policy = DeviceGenerationPolicy::ordinary(), .initial_leading_row_disposition = leading};
                        const auto control_row = std::span<const int, kDeviceGenerationControlCount>(
                            actual_control.data() + request * kDeviceGenerationControlCount, kDeviceGenerationControlCount);
                        EXPECT_EQ(validateDeviceGenerationTerminal(control_row, admission, response_stride, request, 0),
                            DeviceGenerationTerminalError::None);
                        EXPECT_EQ(control_row[kDeviceGenerationControlResponseTokenCount], count);
                        const int consumed = count - (leading == DeviceGenerationLeadingRowDisposition::PendingResponse ? 1 : 0);
                        EXPECT_EQ(actual_live[request], 7 + request + consumed);
                        EXPECT_EQ(actual_live[requests + request], expected_samples[request][count - 1]);
                        EXPECT_EQ(actual_live[2 * requests + request], stopped);
                        EXPECT_EQ(actual_live[3 * requests + request], 1);
                    }
                    EXPECT_EQ(actual_history, initial_history)
                        << "Every emitted token, including EOS, must enter its own history once; padding is untouched";
                }
            }
            if (top_k) {
                // Bad admitted bytes must fail, not silently use a scalar seed
                // or produce a plausible token through a different RNG stream.
                std::array<int32_t, 4 * requests> initial{};
                for (int request = 0; request < requests; ++request) {
                    initial[request] = 7 + request;
                    initial[requests + request] = 99;
                    initial[3 * requests + request] = 1;
                }
                ASSERT_TRUE(backend->hostToDevice(d_live.get(), initial.data(), sizeof(initial), 0, stream));
                ASSERT_TRUE(backend->memset(d_seeds.get(), 0, requests * sizeof(uint64_t), 0, stream));
                ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(requests, budget,
                    DeviceGenerationPolicy::ordinary(), DeviceGenerationLeadingRowDisposition::AlreadyEmitted,
                    response_stride, publication.response_tokens, kDeviceGenerationControlCount, publication.control, 0, stream));
                ASSERT_TRUE(parent ? parent->launch() : decode_graph->launch());
                std::array<int32_t, requests * kDeviceGenerationControlCount> controls;
                ASSERT_TRUE(backend->deviceToHost(controls.data(), d_control.get(), sizeof(controls), 0, stream));
                std::array<int32_t, 4 * requests> observed;
                ASSERT_TRUE(backend->deviceToHost(observed.data(), d_live.get(), sizeof(observed), 0, stream));
                for (int request = 0; request < requests; ++request) {
                    const auto *control = controls.data() + request * kDeviceGenerationControlCount;
                    EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlErrorCode], int(DeviceGenerationError::InvalidOrdinarySample));
                    EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 0);
                    EXPECT_EQ(observed[request], 7 + request);
                    EXPECT_EQ(observed[requests + request], 99);
                    EXPECT_EQ(observed[3 * requests + request], 0);
                }
            }
        }
    });
}

INSTANTIATE_TEST_SUITE_P(Backends, OrdinaryDeviceGenerationSampling,
    ::testing::Values("CUDA", "ROCm"),
    [](const auto &info) { return info.param; });
} // namespace
