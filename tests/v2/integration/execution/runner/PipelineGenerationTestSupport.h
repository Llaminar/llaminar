/**
 * @file PipelineGenerationTestSupport.h
 * @brief Shared model-free graph, admission fixture and independent pipeline oracles.
 *
 * Separate translation units reuse these declarations without recompiling every
 * state, ordinary-generation and speculative-generation test together. All
 * cache and GPU operations remain production implementations.
 */
#pragma once
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
        config.mtp_verifier_outcome_ownership = MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal;
        return config;
    }
};

/** @brief Observe retained graph construction without enabling stage profiling. */
class OrdinaryGraphEvidence final
{
public:
    /** @brief Save and install the narrowly scoped diagnostic collection policy. */
    explicit OrdinaryGraphEvidence(const char *domains = "generation,forward_graph")
    {
        for (const char *key : {"LLAMINAR_PERF_STATS_SUMMARY", "LLAMINAR_PERF_STATS_FILTER"})
            previous.emplace_back(key, std::getenv(key) ? std::optional<std::string>(std::getenv(key)) : std::nullopt);
        setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
        setenv("LLAMINAR_PERF_STATS_FILTER", domains, 1);
        PerfStatsCollector::reset();
    }
    /** @brief Restore the process policy after every assertion/exception exit. */
    ~OrdinaryGraphEvidence()
    {
        for (const auto &[key, value] : previous)
            if (value) setenv(key.c_str(), value->c_str(), 1); else unsetenv(key.c_str());
        PerfStatsCollector::reset();
    }
    /** @return Number of parent constructions, never an inference decision input. */
    double materializations(const char *name = "ordinary_graph_materializations") const
    {
        double count = 0;
        for (const auto &record : PerfStatsCollector::snapshot())
            if (record.domain == "generation" && record.name == name)
                count += record.value;
        return count;
    }
    /** @return Terminal-observed forwards executed inside a complete generation parent.
     * The first prefill sample consumes no model row. Evidence must count real
     * model executions, not sampler transactions or unlaunched setup graphs. */
    double completedForwards(DeviceId device) const
    {
        double count = 0;
        for (const auto &record : PerfStatsCollector::snapshot())
            if (record.domain == "forward_graph" && record.name == "decode_graph_phase" &&
                record.device == device.toString() &&
                record.tags.contains("context") && record.tags.at("context") == "main_decode" &&
                record.tags.contains("phase") && record.tags.at("phase") == "replay")
                count += record.value;
        return count;
    }
private:
    std::vector<std::pair<std::string, std::optional<std::string>>> previous;
};

/** @brief A token-dependent numerical graph with actual canonical KV advancement. */
class OrdinaryDGOGraph final : public Qwen35Graph
{
public:
    /** @brief Own immutable weight bytes before any graph borrows their address. */
    explicit OrdinaryDGOGraph(const GraphConfig &config, bool embedding = true, bool head = true)
        : Qwen35Graph(config, nullptr), table({32, 32}), owns_embedding(embedding), owns_head(head)
    {
        auto *data = table.mutable_data();
        for (int input = 0; input < 32; ++input)
            for (int token = 0; token < 32; ++token)
                data[input * 32 + token] = (token - input - 1 + 32) % 32 < 4 ? 4.0F : -16.0F;
    }
    /** @return This model-free graph owns its only immutable weight. */
    bool isInitialized() const override { return true; }
    /** @brief Bind actual resident token input, logits and KV append in one DAG.
     * @param input Exact production invocation, never a host token shadow.
     * @param output Receives the arena-owned terminal logits.
     * @return Participant-local graph with no mocked kernels or callbacks. */
    ComputeGraph buildForwardGraph(const ForwardInput &input, ForwardOutput &output) override
    {
        if (input.seq_len <= 0 || input.batch_size != 1 || !input.kv_cache)
            throw std::invalid_argument("Ordinary DGO probe requires one canonical KV row");
        ++builds;
        EmbeddingStage::Params embedding;
        embedding.device_id = input.device;
        embedding.embed_table = &table;
        embedding.token_ids = input.token_ids;
        embedding.token_ids_device = input.token_ids_device;
        embedding.output = buffers_.current_hidden;
        embedding.num_tokens = input.seq_len;
        embedding.d_model = embedding.vocab_size = 32;
        embedding.output_buffer_id = BufferId::HIDDEN_STATE;
        KVCacheAppendStage::Params append;
        append.device_id = input.device;
        append.K = append.V = buffers_.current_hidden;
        append.kv_cache = input.kv_cache;
        append.layer_idx = append.seq_idx = 0;
        append.num_tokens = append.seq_len = input.seq_len;
        append.batch_size = 1;
        // A scalar condition's length bank contains its absolute KV position,
        // not an append extent. Grouped verification/prefill use real row counts.
        append.request_sequence_lengths_device = input.execution_role == ForwardExecutionRole::MTPCondition
            ? nullptr : input.sequence_lengths_device;
        if (input.execution_role == ForwardExecutionRole::GroupedMTPVerifier)
            append.append_semantics = KVCacheAppendSemantics::DecodeEquivalentVerifier;
        append.head_dim = 32;
        append.k_buffer_id = append.v_buffer_id = BufferId::HIDDEN_STATE;
        // Use the real bucketed LM-head selection policy: padded rows may
        // execute embedding, but only the resident real length owns KV and
        // the terminal logits row. No fixed row or host length replaces it.
        HiddenStateRowSelectStage::Params select;
        select.device_id = input.device;
        select.input = buffers_.current_hidden;
        select.output = buffers_.logits;
        select.seq_len = input.seq_len;
        select.d_model = 32;
        select.input_buffer_id = BufferId::HIDDEN_STATE;
        select.output_buffer_id = BufferId::LOGITS;
        select.selected_row_idx = input.seq_len - 1;
        select.selection_policy = input.sequence_lengths_device && input.execution_role != ForwardExecutionRole::MTPCondition
            ? HiddenStateRowSelectStage::SelectionPolicy::DeviceResidentRequestLength
            : HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow;
        select.request_sequence_length_device = input.sequence_lengths_device;
        ComputeGraph graph;
        if (owns_embedding)
            graph.addNode("ordinary_embedding", std::make_unique<EmbeddingStage>(embedding), input.device);
        graph.addNode("ordinary_kv_append", std::make_unique<KVCacheAppendStage>(append), input.device);
        if (owns_embedding)
            graph.addDependency("ordinary_kv_append", "ordinary_embedding");
        if (owns_head)
        {
            if (input.execution_role == ForwardExecutionRole::GroupedMTPVerifier)
            {
                // This table-defined model computes each verifier distribution
                // directly from its received token row. The tests separately
                // inspect the stage-owned KV payload to prove the activation
                // edge, since a token-only oracle cannot prove that transport.
                auto distribution = embedding;
                distribution.output = buffers_.logits;
                distribution.output_buffer_id = BufferId::LOGITS;
                graph.addNode("ordinary_terminal_row", std::make_unique<EmbeddingStage>(distribution), input.device);
            }
            else
                graph.addNode("ordinary_terminal_row", std::make_unique<HiddenStateRowSelectStage>(select), input.device);
            graph.addDependency("ordinary_terminal_row", "ordinary_kv_append");
            if (input.execution_role == ForwardExecutionRole::GroupedMTPVerifier)
                addMTPVerifierOutcomeToGraph(graph, "ordinary_terminal_row",
                    buffers_.logits, input.seq_len, input.device);
        }
        output.logits = owns_head ? buffers_.logits : nullptr;
        output.hidden = buffers_.current_hidden;
        return graph;
    }
    /** @brief Full graph setup has the same exact probe, not an alternate implementation. */
    ComputeGraph buildFullForwardGraph(const ForwardInput &input, ForwardOutput &output) override
    {
        return buildForwardGraph(input, output);
    }
    /**
     * @brief Declare this fixture's real shifted-KV sidecar transaction.
     *
     * The predictor uses the same exact four-token table as the main model.
     * Its real device embedding supplies draft logits while a real KV append
     * maintains shifted history. This isolates lifecycle from transformer
     * arithmetic without substituting precomputed tokens or a host sampler.
     *
     * @param depth_idx Predictor depth requested by the retained family.
     * @param bindings Frozen marker proving that the terminal owns a sidecar
     *        binding; this tiny numerical graph has no learned projection.
     * @param input Production MTP sidecar input and shifted cache owner.
     * @param output Tail-owned logits destination supplied by the production sidecar.
     * @return One stateful production KV-append stage, or an empty graph for
     *         an invalid sidecar contract.
     */
    ComputeGraph buildMTPGraph(
        int depth_idx,
        const MTPDepthWeightBindings &bindings,
        const MTPForwardInput &input,
        MTPForwardOutput &output) override
    {
        (void)depth_idx;
        (void)bindings;
        ComputeGraph graph;
        if (!input.kv_cache || !input.terminal_hidden ||
            input.batch_size <= 0 || input.seq_len <= 0)
            return graph;

        KVCacheAppendStage::Params append;
        append.device_id = input.device;
        append.K = append.V = input.terminal_hidden;
        append.kv_cache = input.kv_cache;
        append.layer_idx = append.seq_idx = 0;
        append.num_tokens = input.batch_size * input.seq_len;
        append.seq_len = input.seq_len;
        append.batch_size = input.batch_size;
        append.head_dim = 32;
        append.k_buffer_id = append.v_buffer_id = input.terminal_hidden_buffer_id;
        graph.addNode("ordinary_mtp_shifted_kv_append",
            std::make_unique<KVCacheAppendStage>(append), input.device);
        if (!input.kv_cache_only)
        {
            EmbeddingStage::Params prediction;
            prediction.device_id = input.device;
            prediction.embed_table = &table;
            prediction.token_ids = input.draft_token_ids;
            prediction.token_ids_device = input.draft_token_ids_device;
            prediction.output = output.logits;
            prediction.num_tokens = input.seq_len * input.batch_size;
            prediction.d_model = prediction.vocab_size = 32;
            prediction.output_buffer_id = BufferId::MTP_LOGITS;
            graph.addNode("ordinary_mtp_predictor", std::make_unique<EmbeddingStage>(prediction), input.device);
            graph.addDependency("ordinary_mtp_predictor", "ordinary_mtp_shifted_kv_append");
            // Later draft depths consume the previous predictor's hidden row.
            // Publish it through a real graph output, exactly as the learned
            // sidecar does; a logits-only toy would leave that input undefined.
            prediction.output = output.hidden;
            prediction.output_buffer_id = BufferId::MTP_HIDDEN;
            graph.addNode("ordinary_mtp_hidden", std::make_unique<EmbeddingStage>(prediction), input.device);
            graph.addDependency("ordinary_mtp_hidden", "ordinary_mtp_predictor");
        }
        return graph;
    }
    /** @brief Honor the production PP entrypoint with this fixture's owned stages.
     * @param input Exact invocation, including the participant-local hidden bank.
     * @param output Receives this stage's hidden/logit publications.
     * @param first_layer First fixture layer (one local KV layer).
     * @param last_layer Exclusive fixture layer bound.
     * @param embedding Must match the entry stage selected at construction.
     * @param head Must match the vocabulary owner selected at construction.
     * @return The same real embedding/KV/row-selection kernels, scoped to this stage. */
    ComputeGraph buildPartialForwardGraph(const ForwardInput &input, ForwardOutput &output,
        int first_layer, int last_layer, bool embedding, bool head) override
    {
        if (first_layer != 0 || last_layer != 1 || embedding != owns_embedding || head != owns_head)
            throw std::invalid_argument("Ordinary pipeline fixture role differs from the runner plan");
        return buildForwardGraph(input, output);
    }
    FP32Tensor table; ///< Outlives the runner's retained embedding graph.
    int builds = 0; ///< Setup-only evidence of cache reuse, not inference authority.
    bool owns_embedding; ///< Only the pipeline entry embeds the tail's token.
    bool owns_head; ///< Only the tail produces vocabulary logits for sampling.
};

/**
 * @brief Freeze the terminal-only sidecar binding required by the MTP family.
 *
 * The ordinary graph overrides sidecar arithmetic above, so one immutable
 * marker tensor is sufficient.  Routing the marker through the same frozen
 * authority used by production is intentional: a terminal graph family must
 * never infer sidecar ownership from an incidental cache pointer.
 *
 * @param graph Fixture that owns the tensor for every retained graph lifetime.
 * @param resident_device Participant on which the terminal graph is captured.
 * @return Frozen graph-time authority with one depth-zero sidecar binding.
 */
inline std::unique_ptr<FrozenModelWeightSet> makeOrdinaryMTPFrozenWeightSet(
    OrdinaryDGOGraph &graph, DeviceId resident_device)
{
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.devices.push_back(resident_device);

    ModelWeightSetBuilder builder(strategy);
    WeightBinding binding;
    binding.identity.canonical_name = "mtp.fc.weight";
    binding.identity.role = WeightRole::Other;
    binding.identity.logical_id = stableWeightLogicalId(
        binding.identity.canonical_name);
    binding.residency.home_device = DeviceId::cpu();
    binding.residency.resident_device = resident_device;
    binding.tensor = &graph.table;
    binding.immutable = true;
    builder.addBinding(std::move(binding));

    auto frozen = std::make_unique<FrozenModelWeightSet>(
        strategy, builder.freezeBindings());
    frozen->validateForGraph();
    return frozen;
}

/**
 * @brief Independently verify every archived prompt KV element at terminal observation.
 * @param rank Production pipeline owner after generation and continuation finish.
 * @param prompt Exact request tokens, including a changed partial-hit suffix.
 * @param cached_tokens Expected live prompt plus consumed generated history.
 * @param participants Number of independently owned leaf participant caches.
 *
 * This toy model appends its known embedding to K and V. Checking only tokens
 * would miss a broken cache import because its logits do not read old KV.
 * Inspect only after generation: an earlier diagnostic wait could hide a
 * missing restore-to-generation event edge.
 */
inline void expectOrdinaryPipelinePromptKV(IInferenceRunner &rank, const std::vector<int> &prompt,
    int cached_tokens, size_t participants)
{
    std::vector<uint16_t> expected(prompt.size() * 32);
    for (size_t row = 0; row < prompt.size(); ++row)
        for (int column = 0; column < 32; ++column)
            expected[row * 32 + column] = fp32_to_fp16(
                (column - prompt[row] - 1 + 32) % 32 < 4 ? 4.0F : -16.0F);
    const auto snapshot = rank.captureLivePrefixState();
    ASSERT_TRUE(snapshot.valid);
    ASSERT_EQ(snapshot.cached_tokens, cached_tokens);
    // TP+PP snapshots preserve domain nesting. Walk the ownership tree rather
    // than mistaking each domain for one physical cache; no leaf may be skipped.
    std::vector<const PrefixStateSnapshot *> leaves;
    const auto visit = [&](const auto &self, const PrefixStateSnapshot &node) -> void {
        ASSERT_TRUE(node.valid);
        ASSERT_EQ(node.cached_tokens, cached_tokens);
        if (node.participant_snapshots.empty())
            leaves.push_back(&node);
        else
        {
            EXPECT_TRUE(node.blocks.empty()) << "Only the leaf owns physical KV payload";
            for (const auto &child : node.participant_snapshots) self(self, child);
        }
    };
    visit(visit, snapshot);
    ASSERT_EQ(leaves.size(), participants);
    for (const auto *leaf : leaves)
    {
        const auto &participant = *leaf;
        ASSERT_EQ(participant.blocks.size(), 1u);
        const auto &block = participant.blocks.front();
        ASSERT_EQ(block.layout.fa_layers, 1);
        ASSERT_EQ(block.layout.local_kv_heads, 1);
        ASSERT_EQ(block.layout.head_dim, 32);
        ASSERT_EQ(block.layout.k_precision, ActivationPrecision::FP16);
        ASSERT_EQ(block.layout.v_precision, ActivationPrecision::FP16);
        const size_t bytes = expected.size() * sizeof(uint16_t);
        ASSERT_GE(block.layout.bytes_per_fa_layer_k, bytes);
        ASSERT_GE(block.layout.bytes_per_fa_layer_v, bytes);
        ASSERT_NE(block.kvKData(), nullptr);
        ASSERT_NE(block.kvVData(), nullptr);
        EXPECT_EQ(std::memcmp(block.kvKData(), expected.data(), bytes), 0);
        EXPECT_EQ(std::memcmp(block.kvVData(), expected.data(), bytes), 0);
    }
}


} // namespace llaminar2::test
