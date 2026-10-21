/**
 * @file Qwen4ExpFAOutputStageFixture.h
 * @brief Actual main FA Stage5 composer sharing existing original-source preparation.
 *
 * N2560/K6144 native bytes and real prepared owners come from the existing
 * projection fixture. Only the original canonical role/name and actual FA gate
 * producer differ; no new arithmetic, preparation path or model payload exists.
 */
#pragma once
#include "Qwen4ExpGDNOutputStageFixture.h"
#include "execution/compute_stages/stages/AttentionOutputGateStage.h"
namespace llaminar2::test
{
/** @brief Expose exactly the production FA Stage5 composer and original query scopes. */
class Qwen4ExpFAOutputStageBuilder final : public Qwen35Graph
{
public:
    /** @brief Freeze the same immutable policy as the production graph. */
    explicit Qwen4ExpFAOutputStageBuilder(const GraphConfig &config) : Qwen35Graph(config, nullptr) {}
    /**
     * @brief Compose the actual main Stage5 after its canonical sigmoid gate.
     * @param graph Participant graph with its real gate producer.
     * @param input Complete original query and construction intent.
     * @param buffers Canonical original activation roles.
     * @param source Exact real prepared main source binding.
     * @return Actual production node name.
     */
    std::string append(ComputeGraph &graph, const ForwardInput &input,
        ActivationBuffers &buffers, const WeightBinding &source)
    {
        ForwardInputConstructionScope invocation(*this, input);
        PrefillCollectiveRowsScope query(*this, input);
        return addFAOutputProjection(graph, "layer3_", buffers, source.tensor, &source,
            3, input.seq_len, input.batch_size, input.device, input.sequence_lengths_device,
            "layer3_attn_output_gate");
    }
};
/** @brief One unchanged original source; preparation and retirement reuse the existing fixture. */
struct Qwen4ExpFAOutputStageFixture final
{
    /** @brief Prepare the original main FA source on the actual participant/stream. */
    explicit Qwen4ExpFAOutputStageFixture(DeviceId device, void *stream = nullptr)
        : source(device, stream, "blk.3.attn_output.weight")
    {
        source.config.qfn_gdn_output_activation = QuantizedActivationOperandPrecision::SingleQ8;
        source.config.qfn_fa_output_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
    }
    /** @return New production builder sharing the sole unchanged prepared source. */
    std::unique_ptr<Qwen4ExpFAOutputStageBuilder> builder(const GraphConfig &policy) const
    {
        auto value = std::make_unique<Qwen4ExpFAOutputStageBuilder>(policy);
        value->setPreparedWeightStore(source.prepared.store.get());
        return value;
    }
    Qwen4ExpGDNOutputStageFixture source; ///< Sole original weight/prepared/source lifetime authority.
};
/** @brief Canonical FA activation owners and actual sigmoid-output-gate predecessor. */
struct Qwen4ExpFAOutputActivationFixture final
{
    /** @brief Declare exact existing FA input/output/gate planes. */
    explicit Qwen4ExpFAOutputActivationFixture(int rows)
        : input({static_cast<std::size_t>(rows), 6144}), output({static_cast<std::size_t>(rows), 2560}),
          gate({static_cast<std::size_t>(rows), 6144})
    {
        buffers.attn_output = &input; buffers.attn_proj = &output;
    }
    /** @brief Add the same in-place sigmoid gate node consumed by production Stage5. */
    void addProducer(ComputeGraph &graph, int rows, DeviceId device)
    {
        graph.addNode("layer3_attn_output_gate", std::make_unique<AttentionOutputGateStage>(AttentionOutputGateStage::Params{
            .device_id=device, .input=&input, .gate=&gate, .output=&input,
            .seq_len=rows, .feature_dim=6144, .input_buffer_id=BufferId::ATTN_OUTPUT,
            .gate_buffer_id=BufferId::FA_GATE, .output_buffer_id=BufferId::ATTN_OUTPUT}), device);
    }
    FP32Tensor input, output, gate; ///< Original live owners; no additional projection plane.
    ActivationBuffers buffers; ///< Canonical FA arena-role projection.
};
} // namespace llaminar2::test
