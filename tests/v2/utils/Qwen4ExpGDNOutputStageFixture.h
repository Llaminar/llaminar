/**
 * @file Qwen4ExpGDNOutputStageFixture.h
 * @brief Actual Stage6 construction using one original native Q8_0 prepared source.
 *
 * The fixture does not fabricate a prepared entry or substitute source geometry.
 * CPU tests prepare a real CPU engine; explicit CUDA/ROCm tests prepare the same
 * original source on their actual backend. Stage construction inspects original
 * query ownership only; native arithmetic/capture execution is a separate gate.
 */
#pragma once
#include "CPUPreparedStoreTestAdmission.h"
#include "../unit/models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "PreparedWeightTestHarness.h"
#include "GpuPreparedGemmHarness.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "models/qwen35/Qwen35Graph.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "transfer/TransferEngine.h"
#include "backends/BackendManager.h"
#include "tensors/Tensors.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llaminar2::test
{
/** @brief Expose the production Stage6 composer with its actual complete forward scopes. */
class Qwen4ExpGDNOutputStageBuilder final : public Qwen35Graph
{
public:
    /** @brief Copy immutable model policy; this creates no model/backend allocation. */
    explicit Qwen4ExpGDNOutputStageBuilder(const GraphConfig &config) : Qwen35Graph(config, nullptr) {}
    /**
     * @brief Append the production Stage6 after a real gated-normalization producer.
     * @param graph Caller-owned graph, including the producer node.
     * @param input Exact original query/controller binding whose scope is retained.
     * @param buffers Existing activation owners consumed/written by Stage6.
     * @param source Actual original immutable prepared output weight binding.
     * @return The same production Stage6 node name used by buildGDNAttentionGraph.
     */
    std::string append(ComputeGraph &graph, const ForwardInput &input,
        ActivationBuffers &buffers, const WeightBinding &source)
    {
        ForwardInputConstructionScope invocation(*this, input);
        PrefillCollectiveRowsScope query(*this, input);
        return addGDNOutputProjection(graph, "layer0_", buffers, source.tensor, &source,
            input.seq_len, input.batch_size, input.device, input.sequence_lengths_device,
            "layer0_gated_norm");
    }
};

/** @brief One real N2560/K6144 original Q8_0 source retained through many Stage6 graphs. */
class Qwen4ExpGDNOutputStageFixture final
{
public:
    static constexpr int N = 2560; ///< Original GDN output projection width.
    static constexpr int K = 6144; ///< Original GDN value-head output width.
    /**
     * @brief Prepare exactly the backend-native source once, before graph construction.
     * @param device Exact actual preparation participant.
     * @param stream Non-default actual producer stream for GPU preparation; null for CPU.
     * @param source_name Exact original projection name; shared native shape also supports FA fixtures.
     */
    explicit Qwen4ExpGDNOutputStageFixture(DeviceId device, void *stream = nullptr,
        const std::string &source_name = "blk.0.ssm_out.weight")
        : device(device), original(static_cast<std::size_t>(N) * (K / 32) * sizeof(Q8_0Block))
    {
        if (device.is_gpu() && !stream)
            throw std::invalid_argument("Native Stage6 fixture requires the exact source preparation stream");
        auto *blocks = reinterpret_cast<Q8_0Block *>(original.data());
        for (std::size_t index = 0; index < original.size() / sizeof(Q8_0Block); ++index)
        {
            blocks[index].d = fp32_to_fp16(1.0f / 128.0f);
            for (int lane = 0; lane < 32; ++lane)
                blocks[index].qs[lane] = static_cast<int8_t>(static_cast<int>((index + lane) % 23) - 11);
        }
        weight = std::make_unique<Q8_0Tensor>(std::vector<std::size_t>{N, K}, original);
        prepared.model_id = ModelContextId{99317};
        prepared.binding = makePreparedWeightTestBinding(weight.get(), device,
            source_name, prepared.model_id);
        prepared.binding.slice = {.source_rows=N, .source_cols=K, .row_count=N, .col_count=K};
        stream_ = stream;
        backend_ = device.is_gpu() ? getBackendFor(device) : nullptr;
        if (device.is_gpu() && !backend_)
            throw std::invalid_argument("Native Stage6 fixture has no actual backend");
        try
        {
            prepared.store = std::make_unique<PreparedWeightStore>(prepared.model_id);
            if (device.is_gpu())
            {
                // GPU preparation belongs to the production upload/repack
                // pipeline. Keep its original native pool alive alongside the
                // real store handle, rather than uploading an unused raw copy.
                prepared.ref=registerGpuPreparedGemmInStore(weight.get(),device,source_name,
                    prepared.model_id,prepared.store.get(),&gpu_loaders_,&prepared.binding);
                // The loader helper's empty slice canonically means the whole
                // original tensor. Stage6 explicitly proves that same complete
                // matrix; authenticate the equivalent full slice through the
                // store's public adoption contract, never a sliced fake source.
                prepared.binding.slice={.source_rows=N,.source_cols=K,.row_count=N,.col_count=K};
                prepared.binding.prepared=prepared.ref;
                if(!prepared.store->adoptPreparedGemmForBinding(prepared.binding,device))
                    throw std::runtime_error("Native Stage6 fixture lost its original prepared source binding");
            }
            else
            {
                // Admit this actual original matrix before its CPU native pack;
                // the shared FA fixture inherits the same source-owned claim.
                admitCPUPreparedStoreForTest(*prepared.store, {weight.get()}, device);
                prepared.ref = prepared.store->prepareGemm(prepared.binding);
            }
            prepared.binding.prepared = prepared.ref;
            Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), config);
            config.default_device = device;
            config.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
            config.qfn_gdn_output_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
        }
        catch (...)
        {
            retirePreparation();
            throw;
        }
    }
    /** @brief Join diagnostic preparation before prepared/source owners are destroyed, including assertion unwinds. */
    ~Qwen4ExpGDNOutputStageFixture() { retirePreparation(); }
    /** @return A fresh production builder sharing only the immutable prepared engine/store. */
    std::unique_ptr<Qwen4ExpGDNOutputStageBuilder> builder(const GraphConfig &policy) const
    {
        auto result = std::make_unique<Qwen4ExpGDNOutputStageBuilder>(policy);
        result->setPreparedWeightStore(prepared.store.get());
        return result;
    }
    /** @return True only if preparation/composition left every original native source byte untouched. */
    bool sourceUnchanged() const
    {
        return weight->native_type() == TensorType::Q8_0 && weight->size_bytes() == original.size() &&
            weight->raw_data() && std::memcmp(weight->raw_data(), original.data(), original.size()) == 0;
    }
    DeviceId device; ///< Actual backend used for source preparation.
    std::vector<uint8_t> original; ///< Exact native original Q8_0 blocks; never a requantized format.
    std::unique_ptr<Q8_0Tensor> weight; ///< Source outlives every constructed stage.
    std::vector<std::shared_ptr<LoadOrchestrator>> gpu_loaders_; ///< Real native pool owners outlive the prepared store, with no separate raw GPU source.
    PreparedGemmFixture prepared; ///< Store/ref/immutable source identity, never registerPreparedForTest.
    GraphConfig config; ///< Authenticated actual QFN widths and frozen independent policy.
private:
    /** @brief Diagnostic-only terminal join; a failed join cannot permit unsafe source destruction. */
    void retirePreparation() noexcept
    {
        if (backend_ && !backend_->synchronizeStream(stream_, device.ordinal)) std::terminate();
    }
    IBackend *backend_ = nullptr; ///< Borrowed setup authority, absent on device-free CPU tests.
    void *stream_ = nullptr; ///< Exact producer stream; never a default-stream substitute.
};

/** @brief Existing Stage5/6 activation owners with no alternate view or scratch allocation. */
struct Qwen4ExpGDNOutputActivationFixture final
{
    /** @brief Allocate one physical shape; device preparation belongs to explicit native tests. */
    explicit Qwen4ExpGDNOutputActivationFixture(int rows)
        : input({static_cast<std::size_t>(rows), Qwen4ExpGDNOutputStageFixture::K}),
          output({static_cast<std::size_t>(rows), Qwen4ExpGDNOutputStageFixture::N}),
          gate({static_cast<std::size_t>(rows), Qwen4ExpGDNOutputStageFixture::K}), gamma({128})
    {
        buffers.attn_output = &input;
        buffers.attn_proj = &output;
        buffers.extensions[BufferId::GDN_Z] = &gate;
        std::fill_n(gamma.mutable_data(), gamma.numel(), 1.0f);
    }
    /**
     * @brief Declare the real gated-norm predecessor, matching production Stage5's graph edge.
     * @param graph Graph receiving the existing input producer.
     * @param rows Exact same physical query rows used by Stage6.
     * @param device Actual participant whose prepared Stage6 will consume this output.
     */
    void addProducer(ComputeGraph &graph, int rows, DeviceId device)
    {
        graph.addNode("layer0_gated_norm", std::make_unique<GatedRMSNormStage>(GatedRMSNormStage::Params{
            .device_id=device, .input=&input, .gate=&gate, .output=&input, .gamma=&gamma,
            .seq_len=rows, .feature_dim=Qwen4ExpGDNOutputStageFixture::K, .norm_dim=128,
            .gate_activation=GatedNormActivation::Sigmoid, .input_buffer_id=BufferId::ATTN_OUTPUT,
            .gate_buffer_id=BufferId::GDN_Z, .output_buffer_id=BufferId::ATTN_OUTPUT}), device);
    }
    FP32Tensor input; ///< Existing gated-norm/output-projection input, K6144.
    FP32Tensor output; ///< Original projection output, N2560.
    FP32Tensor gate; ///< Existing Stage5 Z gate.
    FP32Tensor gamma; ///< Original per-head Stage5 gain.
    ActivationBuffers buffers; ///< Exact existing role bindings, with no secondary projection output.
};
} // namespace llaminar2::test
