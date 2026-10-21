/**
 * @file Qwen4ExpHCQueryFixture.h
 * @brief Actual original-Q8 prepared HC graph sites and their existing activation owners.
 *
 * The fixture calls production branch/read-only/head composition, not a copied
 * row resolver. Its named weights have real native tensors and executable
 * PreparedWeightStore handles. No model or projection is executed by these
 * metadata/capture-identity contracts.
 */
#pragma once
#include "../unit/models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "PreparedWeightTestHarness.h"
#include "GpuPreparedGemmHarness.h"
#include "models/qwen35/Qwen35Graph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "backends/BackendManager.h"
#include "transfer/TransferEngine.h"
#include "tensors/Tensors.h"
#include <algorithm>
#include <array>
#include <deque>
#include <stdexcept>
namespace llaminar2::test
{
/** @brief The three main composition sites, with both block-weight roles explicit. */
enum class HCQuerySite { Attention, FeedForward, ReadOnlyAttention, Head };
/** @brief Admitted original branch scratch, kept distinct exactly as the real builder requires. */
struct HCQueryActivations final
{
    static constexpr int H=2560, C=4*H, R=320;
    /** @brief Allocate only the existing physical row roles; no alternate projection output is added. */
    explicit HCQueryActivations(int rows)
        : trunk({static_cast<size_t>(rows),C}), normalized({static_cast<size_t>(rows),C}),
          down({static_cast<size_t>(rows),R}), logits({static_cast<size_t>(rows),C}),
          mixed({static_cast<size_t>(rows),H}), block({static_cast<size_t>(rows),H}),
          inject({static_cast<size_t>(rows),4}), blockGain({128})
    {
        buffers.current_hidden=&trunk; buffers.normalized=&mixed; buffers.attn_proj=&block;
        buffers.extensions[BufferId::QFN_HC_NORMALIZED]=&normalized;
        buffers.extensions[BufferId::QFN_HC_DOWN]=&down;
        buffers.extensions[BufferId::QFN_HC_READ_LOGITS]=&logits;
        buffers.extensions[BufferId::QFN_HC_WRITE_LOGITS]=&inject;
        std::fill_n(blockGain.mutable_data(),blockGain.numel(),1.0f);
    }
    FP32Tensor trunk,normalized,down,logits,mixed,block,inject,blockGain;
    ActivationBuffers buffers; ///< Borrowed exact declared roles; no tensor row views or rebinding.
};
/** @brief Expose the real main HC sites under their complete original construction scopes. */
class HCQueryBuilder final : public Qwen35Graph
{
public:
    /** @brief Retain only frozen model/setup policy.
     * @param config Authenticated model shape and original arithmetic selection.
     */
    explicit HCQueryBuilder(const GraphConfig &config):Qwen35Graph(config,nullptr) {}
    /**
     * @brief Compose an actual branch/read-only/head HC, including the paired original injection when present.
     * @param graph Participant-local destination; invalid query ownership adds no nodes.
     * @param input Original complete main invocation, never a host live-count substitute.
     * @param activation Existing arena-shaped roles, retained throughout graph inspection.
     * @param site Exact production composition entrypoint.
     * @return Canonical HC node prefix, used only to inspect actual resulting stages.
     */
    std::string append(ComputeGraph &graph,const ForwardInput &input,HCQueryActivations &activation,HCQuerySite site)
    {
        ForwardInputConstructionScope invocation(*this,input);
        PrefillCollectiveRowsScope prefill(*this,input);
        ModelBuffers model; model.current_hidden=&activation.trunk; model.layer_buffers=activation.buffers;
        setBuffers(model);
        if(site==HCQuerySite::Head)
        {
            addBranchHeadRead(graph,&activation.trunk,&activation.mixed,"",input.seq_len,input.device,BufferId::HIDDEN_STATE);
            return "hc_head_";
        }
        if(site==HCQuerySite::ReadOnlyAttention)
        {
            addBranchAttentionReadOnly(graph,"layer0_",activation.buffers,0,input.seq_len,input.device);
            return "layer0_hc_attn_";
        }
        auto read=addBranchBlockRead(graph,"layer0_",activation.buffers,0,input.seq_len,input.device,
            site==HCQuerySite::Attention?ResidualBlock::Attention:ResidualBlock::FeedForward);
        // A real typed block producer permits inspecting paired injection Params
        // without an invented dependency or executing attention/MoE arithmetic.
        graph.addNode("completed_block",std::make_unique<GatedRMSNormStage>(GatedRMSNormStage::Params{
            .device_id=input.device,.input=&activation.mixed,.gate=&activation.mixed,.output=&activation.block,
            .gamma=&activation.blockGain,.seq_len=input.seq_len,.feature_dim=HCQueryActivations::H,
            .norm_dim=128,.gate_activation=GatedNormActivation::Sigmoid}),input.device);
        graph.addDependency("completed_block",read.terminal());
        addBranchBlockWrite(graph,std::move(read),activation.buffers,"completed_block");
        return site==HCQuerySite::Attention?"layer0_hc_attn_":"layer0_hc_ffn_";
    }
};
/** @brief Shared original native source/store for all main HC sites and policy combinations. */
class HCQuerySource final
{
public:
    /**
     * @brief Prepare original native Q8 tensors before composing any graph.
     * @param device Actual original source/prepared-engine participant.
     * @param stream Non-default preparation stream on GPU; absent for CPU.
     */
    explicit HCQuerySource(DeviceId device,void *stream=nullptr)
        : device_(device),stream_(stream),gain_({4,HCQueryActivations::H}),store_(ModelContextId{99191})
    {
        if(device.is_gpu() && !stream)throw std::invalid_argument("HC fixture requires its actual source preparation stream");
        backend_=device.is_gpu()?getBackendFor(device):nullptr;
        if(device.is_gpu() && !backend_)throw std::invalid_argument("HC fixture has no native backend");
        std::fill_n(gain_.mutable_data(),gain_.numel(),1.0f);
        try
        {
            down_=matrix(HCQueryActivations::R,HCQueryActivations::C);
            up_=matrix(HCQueryActivations::C,HCQueryActivations::R);
            inject_=matrix(4,HCQueryActivations::C);
            attention_=weights("blk.0.hc_attn_");
            ffn_=weights("blk.0.hc_ffn_");
            head_=weights("output_hc_");
            Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false),config);
            config.default_device=device;
            config.qfn_hc_up_activation=QuantizedActivationOperandPrecision::TwoTermQ8;
        }
        catch(...){retire();throw;}
    }
    /** @brief Retire diagnostic setup before any source/prepared owner destruction, even on assertion unwind. */
    ~HCQuerySource(){retire();}
    /** @brief Bind one actual graph builder to the same immutable prepared source authority.
     * @param policy Frozen caller selection; no environment is reread here.
     * @return A fresh builder retaining only original store/binding borrows.
     */
    std::unique_ptr<HCQueryBuilder> builder(const GraphConfig &policy)
    {
        auto result=std::make_unique<HCQueryBuilder>(policy);
        result->setPreparedWeightStore(&store_);
        ModelWeightBindings bindings;
        bindings.get_layer_weights=[this](int layer) {
            if(layer!=0)throw std::invalid_argument("HC fixture was asked for another physical source layer");
            LayerWeightBindings value;value.flash_next.attention=attention_;value.flash_next.ffn=ffn_;return value;
        };
        bindings.hyper_connection_head=head_;
        result->setWeightBindings(bindings);
        return result;
    }
    /** @brief Observe the actual original source selected by a semantic site.
     * @param site Exact branch/head role under test.
     * @return Immutable store-owned source binding.
     */
    const WeightBinding &upBinding(HCQuerySite site) const
    {
        return *(site==HCQuerySite::Head?head_.up:site==HCQuerySite::FeedForward?ffn_.up:attention_.up);
    }
    GraphConfig config; ///< Frozen authenticated QFN geometry and explicit policy.
private:
    /** @brief Create deterministic original native blocks; no prepared-weight requantization is substituted.
     * @param n Native source output rows.
     * @param k Native source K32-aligned reduction columns.
     * @return Original source tensor; GPU prepared payload is owned by the load pipeline in bind().
     * @throws std::exception For source allocation failure.
     */
    std::unique_ptr<Q8_0Tensor> matrix(int n,int k)
    {
        std::vector<uint8_t> bytes(static_cast<size_t>(n)*k/32*sizeof(Q8_0Block));
        auto *blocks=reinterpret_cast<Q8_0Block *>(bytes.data());
        for(size_t b=0;b<bytes.size()/sizeof(Q8_0Block);++b)
        {
            blocks[b].d=fp32_to_fp16(1.0f/128.0f);
            for(int lane=0;lane<32;++lane)blocks[b].qs[lane]=static_cast<int8_t>(static_cast<int>((b+lane)%17)-8);
        }
        return std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(n),static_cast<size_t>(k)},bytes);
    }
    /** @brief Bind a native named source to its sole immutable store entry.
     * @param tensor Original native tensor retained by this fixture.
     * @param name Original semantic source name.
     * @param projection Whether this is a matrix requiring an executable prepared handle.
     * @return Stable immutable source binding, retained in the fixture deque.
     * @throws std::exception For actual preparation failure.
     */
    const WeightBinding *bind(TensorBase &tensor,const std::string &name,bool projection)
    {
        WeightBinding value;
        if(projection&&device_.is_gpu())
        {
            // The existing fixture helper drives the real upload/repack pool
            // and registers its executable handle. Tensor upload alone is not
            // prepared GEMM authority; the CPU-local factory rejects that use.
            const auto ref=registerGpuPreparedGemmInStore(&tensor,device_,name,store_.modelId(),
                &store_,&gpu_loaders_,&value);
            value.prepared=ref;
        }
        else
        {
            value=makePreparedWeightTestBinding(&tensor,device_,name,store_.modelId());
            if(projection)value.prepared=store_.prepareGemm(value);
        }
        bindings_.push_back(std::move(value));return &bindings_.back();
    }
    /** @brief Create exact named block/head roles; only a block consumes its injection.
     * @param prefix Original source-role prefix including its underscore.
     * @return Complete immutable actual norm/down/up/inject binding group.
     */
    HyperConnectionWeights<const WeightBinding *> weights(const std::string &prefix)
    {
        return {bind(gain_,prefix+"norm.weight",false),bind(*down_,prefix+"down.weight",true),
            bind(*up_,prefix+"up.weight",true),bind(*inject_,prefix+"inject.weight",true)};
    }
    /** @brief Diagnostic terminal retirement only; inference graph construction submits no work. */
    void retire() noexcept
    {
        if(backend_ && !backend_->synchronizeStream(stream_,device_.ordinal))std::terminate();
    }
    DeviceId device_;void *stream_;IBackend *backend_=nullptr;
    FP32Tensor gain_;
    std::unique_ptr<Q8_0Tensor> down_,up_,inject_;
    std::vector<std::shared_ptr<LoadOrchestrator>> gpu_loaders_; ///< Actual upload/pool owners outlive prepared handles and retire before original tensors.
    PreparedWeightStore store_; ///< Destroyed before original source owners, after diagnostic retirement.
    std::deque<WeightBinding> bindings_;
    HyperConnectionWeights<const WeightBinding *> attention_,ffn_,head_;
};
/** @brief Inspect the actual resulting prepared GEMMStage, with no copied resolver/Params.
 * @param graph Real HC composition result.
 * @param name Exact actual projection node name.
 * @return The graph-owned projection, borrowed through graph lifetime.
 * @throws std::runtime_error For missing or wrongly typed production output.
 */
inline const GEMMStage &hcQueryStage(const ComputeGraph &graph,const std::string &name)
{
    const auto *node=graph.getNode(name);
    const auto *stage=node?dynamic_cast<const GEMMStage *>(node->stage.get()):nullptr;
    if(!stage)throw std::runtime_error("Actual HC graph omitted its prepared projection");
    return *stage;
}
inline constexpr std::array hcQuerySites{HCQuerySite::Attention,HCQuerySite::FeedForward,
    HCQuerySite::ReadOnlyAttention,HCQuerySite::Head};
} // namespace llaminar2::test
