/**
 * @file RetainedCUDAQSAAttentionStage.cpp
 * @brief Participant-local CUDA QSA Pre-Attention and Post-Attention megakernel graph stages.
 *
 * Replaces all 35 participant-local CUDA QSA kernels with:
 *   - RetainedCUDAQSAPreAttentionStage (1 cooperative kernel)
 *   - Standalone qsa_indexer
 *   - RetainedCUDAQSAPostAttentionStage (selectedAttentionFP16Tiles + 1 cooperative kernel)
 */
#include "RetainedQSAAttentionStage.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDAHardwareMetadata.h"
#include "execution/compute_stages/IGraphOutputDemand.h"
#include "execution/compute_stages/stages/AttentionOutputGateStage.h"
#include "execution/compute_stages/stages/FullNativeHCFFNOutputDemandResolver.h"
#include "execution/compute_stages/stages/FusedQKVGEMMStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/compute_stages/stages/QGateSplitStage.h"
#include "execution/compute_stages/stages/QKNormStage.h"
#include "execution/compute_stages/stages/RoPEStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "execution/compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/LocalMegakernelLowering.h"
#include "kernels/IKVCache.h"
#include "kernels/cuda/attention/CUDAQwenSparseAttention.h"
#include "kernels/cuda/gemm/CUDANativeQuantisedPrefillHostPolicy.h"
#include "kernels/cuda/qwen4exp/CUDARetainedQSA.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
namespace
{
    namespace native = cuda::qwen4exp;
    constexpr auto kCudaQSAScratchBank = "qfn_cuda_retained_qsa";
    constexpr int kHidden = 2560;
    constexpr int kHeadDim = 256;

    void require(bool condition, const char *message)
    {
        if (!condition)
            throw std::invalid_argument(message);
    }

    struct OutputLease
    {
        TensorBase *tensor = nullptr;
        BufferId id{};
        std::size_t byte_offset = 0;
        std::size_t byte_extent = 0;
        std::shared_ptr<const CapturedDeviceStorageLease> lease;
    };

    /** @brief Resolve canonical M1 K-partition schedule for a prepared CUDA NativeQuantised matrix. */
    native::CUDARetainedQSAProjection bindNativeCUDAProj(
        ITensorGemm *eng,
        float *out,
        int expected_n,
        int policy_n,
        int expected_k,
        int sm_count,
        const std::optional<OriginalOutputSlice> &slice)
    {
        native::CUDARetainedQSAProjection proj{};
        require(eng && eng->inspectPreparedNativeQuantisedMatrixDesc(proj.weights) &&
                proj.weights.valid() && proj.weights.n == expected_n && proj.weights.k == expected_k &&
                proj.weights.source_identity_present,
            "Retained CUDA QSA requires prepared NativeQuantised matrices");
        const std::uint8_t policy_cb = slice
            ? slice->sourceIdentity().codebook_id
            : proj.weights.source_codebook_id;
        int ordered = 0, partitions = 0;
        require(cudaNativeQuantisedGemvTuned_queryCanonicalM1Schedule(
                    policy_cb, policy_n, expected_k, sm_count, &ordered, &partitions) &&
                partitions > 0 && partitions <= expected_k / 32 && (ordered || partitions == 1),
            "Retained CUDA QSA has no canonical M1 schedule for projection");
        proj.output = out;
        proj.k_partitions = partitions;
        proj.blocks_per_partition = (expected_k / 32 + partitions - 1) / partitions;
        proj.ordered_kpar = ordered != 0;
        return proj;
    }

    /** @brief Bind an indexer projection supporting BOTH all 18 NativeQuantised codebooks and BF16/FP16/FP32. */
    native::CUDARetainedQSAIndexerProj bindCUDAIndexerProj(
        const std::optional<GEMMStage::Params> &gemm_opt,
        ITensorGemm *&engine_out,
        int expected_n,
        int sm_count)
    {
        native::CUDARetainedQSAIndexerProj dst{};
        if (!gemm_opt)
            return dst;
        const auto &g = *gemm_opt;
        const auto binding = g.prepared_store->binding(*g.prepared_ref);
        engine_out = g.prepared_store->gemmKernel(*g.prepared_ref);
        require(binding && binding->immutable && binding->tensor == g.B && engine_out,
            "Retained CUDA QSA indexer projection lost its immutable prepared binding");
        auto *out_tensor = dynamic_cast<TensorBase *>(g.C);
        require(out_tensor && out_tensor->gpu_data_ptr(), "Retained CUDA QSA indexer output is not prepared");
        auto *out_ptr = static_cast<float *>(out_tensor->gpu_data_ptr());
        dst.n = expected_n;
        dst.ranking_only_guard = g.projection_row_work.has_value();

        const TensorType wt = g.B->native_type();
        if (wt == TensorType::FP32 || wt == TensorType::FP16 || wt == TensorType::BF16)
        {
            ContiguousFloatingPointWeightDescriptor view;
            require(engine_out->exportContiguousFloatingPointWeights(view) && view.valid() &&
                    view.type == wt && view.n == expected_n && view.k == kHidden,
                "Retained CUDA QSA floating indexer projection could not export contiguous weights");
            dst.format = (wt == TensorType::BF16) ? native::CUDARetainedQSAIndexerFormat::BF16
                       : (wt == TensorType::FP16) ? native::CUDARetainedQSAIndexerFormat::FP16
                                                  : native::CUDARetainedQSAIndexerFormat::FP32;
            dst.floating_weights = view.data;
            dst.floating_output = out_ptr;
            return dst;
        }

        dst.format = native::CUDARetainedQSAIndexerFormat::NativeQuantised;
        dst.quantised = bindNativeCUDAProj(
            engine_out, out_ptr, expected_n, expected_n, kHidden, sm_count, std::nullopt);
        return dst;
    }

    /** @brief Sealed pre-attention subgraph declaration consumed by RetainedCUDAQSAPreAttentionStage. */
    struct CUDAQSAPreDeclaration
    {
        FusedQKVGEMMStage::Params qkv;
        QGateSplitStage::Params split;
        std::optional<QKNormStage::Params> q_norm;
        std::optional<QKNormStage::Params> k_norm;
        RoPEStage::Params rope;
        QwenSparseAttentionStage::Params replay_stage;
        KVCacheAppendStage::Params kv_append;
        GEMMStage::Params key_proj;
        std::optional<GEMMStage::Params> query_proj;
    };

    class RetainedCUDAQSAPreAttentionStage final
        : public IComputeStage, public IWorkspaceConsumer, public IGraphOutputDemand
    {
    public:
        RetainedCUDAQSAPreAttentionStage(std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc,
                                         CUDAQSAPreDeclaration decl)
            : IComputeStage(decl.qkv.device_id),
              hc_(std::move(hc)),
              decl_(std::move(decl))
        {
            const auto &qkv = decl_.qkv;
            const auto &sp = decl_.split;
            const auto &rp = decl_.rope;
            const auto &prep = std::get<QwenSparseAttentionStage::PrepareReplay>(decl_.replay_stage.operation);
            const auto &kva = decl_.kv_append;
            require(hc_ && device().is_cuda() && hc_->phaseDeclaration().device() == device() &&
                    qkv.m == 1 && qkv.k == kHidden &&
                    sp.head_dim == kHeadDim && sp.n_heads >= 1 &&
                    static_cast<unsigned>(sp.n_heads) <= native::kMaxCUDARetainedQSAQHeads &&
                    qkv.n_q == sp.n_heads * 2 * kHeadDim &&
                    qkv.n_k == kva.kv_cache->local_n_kv_heads() * kHeadDim &&
                    qkv.n_v == qkv.n_k &&
                    static_cast<unsigned>(kva.kv_cache->local_n_kv_heads()) <= native::kMaxCUDARetainedQSAKVHeads &&
                    rp.position_ids_device != nullptr &&
                    kva.kv_cache->isGraphCaptureReady() &&
                    kva.kv_cache->k_precision() == ActivationPrecision::FP16 &&
                    kva.kv_cache->v_precision() == ActivationPrecision::FP16,
                "Retained CUDA QSA Pre-Attention requires valid ordinary M1 CUDA FP16 QSA declarations");

            for (const auto &output : hc_->phaseOutputs())
                if (output.tensor == qkv.input && output.buffer == *qkv.input_buffer_id && output.byteOffset == 0)
                    mixed_ = output.tensor;
            require(mixed_, "Retained CUDA QSA Pre-Attention must consume its exact HC mixed publication");

            const auto optional = hc_->optionalPhaseOutputs();
            for (const auto &opt : optional)
                require(opt.tensor && opt.byteOffset == 0,
                    "Retained CUDA QSA Pre-Attention optional outputs require original arena bindings");
            optional_ = optional;
            hc_->sealFullOutputDemand(FullNativeHCFFNOutputDemand::none());
            for (const auto &output : hc_->phaseOutputs())
                if (std::none_of(phase_outputs_.begin(), phase_outputs_.end(),
                    [&](const auto &prior) { return prior.tensor == output.tensor; }))
                    phase_outputs_.push_back(output);
            phase_inputs_ = hc_->phaseInputs();
            for (auto *w : hc_->phaseDeclaration().bufferContract().weight_tensors)
                phase_weights_.push_back(w);

            auto addOut = [&](ITensor *t, BufferId id, std::size_t off, std::size_t bytes) {
                auto *tb = dynamic_cast<TensorBase *>(t);
                require(tb && bytes > 0, "Retained CUDA QSA Pre-Attention output tensor invalid");
                outputs_.push_back({tb, id, off, bytes, {}});
            };
            addOut(sp.output_q, *sp.output_q_buffer_id, 0,
                   static_cast<std::size_t>(sp.n_heads) * kHeadDim * sizeof(float));
            addOut(sp.output_gate, *sp.output_gate_buffer_id, 0,
                   static_cast<std::size_t>(sp.n_heads) * kHeadDim * sizeof(float));
            addOut(qkv.output_k, *qkv.output_k_buffer_id, 0,
                   static_cast<std::size_t>(qkv.n_k) * sizeof(float));
            addOut(qkv.output_v, *qkv.output_v_buffer_id, 0,
                   static_cast<std::size_t>(qkv.n_v) * sizeof(float));
            addOut(decl_.replay_stage.replay.tensor, decl_.replay_stage.replay.buffer,
                   decl_.replay_stage.replay.byteOffset, sizeof(llaminar2::qwen4exp::QwenSparseReplay));
            addOut(prep.rowParameters->tensor, prep.rowParameters->buffer,
                   prep.rowParameters->byteOffset, sizeof(attention::AttentionDeviceParams));
            addOut(decl_.key_proj.C, *decl_.key_proj.c_buffer_id, 0, 128u * sizeof(float));
            if (decl_.query_proj)
                addOut(decl_.query_proj->C, *decl_.query_proj->c_buffer_id, 0, 512u * sizeof(float));

            if (decl_.q_norm && decl_.q_norm->gamma)
                raw_weights_.push_back(const_cast<ITensor *>(decl_.q_norm->gamma));
            if (decl_.k_norm && decl_.k_norm->gamma)
                raw_weights_.push_back(const_cast<ITensor *>(decl_.k_norm->gamma));
        }

        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_QKV; }
        bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::GPU_CUDA; }
        bool isGraphCapturable() const override { return admission_.has_value(); }
        bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return GraphLaunchPreparationPolicy::CaptureOnly;
        }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }

        StageBufferContract bufferContract() const override
        {
            auto result = hc_->phaseDeclaration().bufferContract();
            for (const auto &o : outputs_)
                result.addOutput(o.id);
            for (auto *w : raw_weights_)
                result.addWeight(w);
            const auto &qkv = decl_.qkv;
            result.addPreparedWeight(const_cast<ITensor *>(qkv.wq), qkv.prepared_store, *qkv.prepared_ref_q);
            result.addPreparedWeight(const_cast<ITensor *>(qkv.wk), qkv.prepared_store, *qkv.prepared_ref_k);
            result.addPreparedWeight(const_cast<ITensor *>(qkv.wv), qkv.prepared_store, *qkv.prepared_ref_v);
            result.addPreparedWeight(const_cast<ITensor *>(decl_.key_proj.B),
                                     decl_.key_proj.prepared_store, *decl_.key_proj.prepared_ref);
            if (decl_.query_proj)
                result.addPreparedWeight(const_cast<ITensor *>(decl_.query_proj->B),
                                         decl_.query_proj->prepared_store, *decl_.query_proj->prepared_ref);
            return result;
        }

        StageBufferRequirements getBufferRequirements() const override
        {
            return hc_->phaseDeclaration().getBufferRequirements();
        }

        std::optional<StageDumpInfo> coldSnapshotPublicationDeclaration() const override
        {
            auto info = buildDumpInfoImpl();
            info.addOutput("activated_down", (*optional_)[0].tensor, 1, (*optional_)[0].tensor->cols());
            info.addOutput("read_logits", (*optional_)[1].tensor, 1, (*optional_)[1].tensor->cols());
            return info;
        }

        void sealGraphOutputDemand(const ComputeGraph &graph, const BufferArena &arena,
                                   std::span<const std::string> selected, std::uint64_t epoch) override
        {
            const auto order = graph.getExecutionOrder();
            const auto producer = std::find_if(order.begin(), order.end(),
                [&](const auto &name) { return graph.getNode(name)->stage.get() == this; });
            require(producer != order.end(), "Retained CUDA QSA Pre-Attention has no producer node");
            for (const auto &name : selected)
                require(name != "activated_down" && name != "read_logits",
                    "Retained CUDA QSA Pre-Attention cannot publish optional HC diagnostics");
            const std::array views{
                FullNativeHCFFNOutputBinding{FullNativeHCFFNOutputRole::HCActivated,
                    *producer, (*optional_)[0].tensor, (*optional_)[0].buffer, 0, (*optional_)[0].tensor->size_bytes()},
                FullNativeHCFFNOutputBinding{FullNativeHCFFNOutputRole::HCUp,
                    *producer, (*optional_)[1].tensor, (*optional_)[1].buffer, 0, (*optional_)[1].tensor->size_bytes()}};
            require(resolveFullNativeHCFFNOutputDemand(graph, arena, *producer, views).bits() == 0,
                "Retained CUDA QSA Pre-Attention cannot omit an HC intermediate with an active reader");
            const auto identity = std::pair{graph.topologyGeneration(), epoch};
            require(!seal_ || *seal_ == identity, "Retained CUDA QSA Pre-Attention observation identity changed");
            seal_ = identity;
        }

        WorkspaceRequirements getWorkspaceRequirements(int, int, int) const override
        {
            auto result = hc_->phaseWorkspaceRequirements();
            result.merge(retainedCUDAQSAAttentionWorkspace());
            return result;
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            require(!admission_ || workspace_ == workspace,
                "Retained CUDA QSA Pre-Attention workspace changed while captured");
            hc_->bindPhaseWorkspace(workspace);
            workspace_ = workspace;
        }

        void unbindWorkspace() override
        {
            require(!admission_, "Retained CUDA QSA Pre-Attention must retire before workspace release");
            bindWorkspace(nullptr);
        }

        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            require(seal_ && ctx && ctx->deviceId() == device() && workspace_ &&
                    workspace_->device() == device() && reinterpret_cast<std::uintptr_t>(stream) > 2 &&
                    !isGraphCaptureActive(),
                "Retained CUDA QSA Pre-Attention requires sealed cold context, workspace, and stream");
            if (admission_)
            {
                validateStorage(stream);
                return true;
            }
            setGPUStream(stream);
            hc_->preparePhaseBindings(ctx, stream);
            const auto &hc_prefix = hc_->cudaPrefixArguments();
            require(hc_prefix.mixed == mixed_->gpu_data_ptr(),
                "Retained CUDA QSA Pre-Attention HC mixed pointer mismatch");
            require(hc_prefix.up.rows.capacity() == 1 && hc_prefix.up.rows.physicalRows() == 1 &&
                    hc_prefix.up.rows.countOwner() == nullptr,
                "Retained CUDA QSA Pre-Attention requires fully-active M1 row authority");

            for (auto &o : outputs_)
            {
                o.lease = TransferEngine::instance().retainCapturedDeviceStorage(
                    *o.tensor, device(), o.byte_offset + o.byte_extent);
            }

            const auto &qkv = decl_.qkv;
            auto resolveQKVEngine = [&](const std::optional<PreparedWeightRef> &ref,
                                        const std::optional<OriginalOutputSlice> &slice,
                                        const ITensor *weight) -> ITensorGemm * {
                if (slice)
                    return slice->retainedEngine();
                const auto b = qkv.prepared_store->binding(*ref);
                auto *eng = qkv.prepared_store->gemmKernel(*ref);
                require(b && b->immutable && b->tensor == weight && eng,
                    "Retained CUDA QSA QKV engine missing or mutable");
                return eng;
            };
            q_engine_ = resolveQKVEngine(qkv.prepared_ref_q, qkv.original_output_slice_q, qkv.wq);
            k_engine_ = resolveQKVEngine(qkv.prepared_ref_k, qkv.original_output_slice_k, qkv.wk);
            v_engine_ = resolveQKVEngine(qkv.prepared_ref_v, qkv.original_output_slice_v, qkv.wv);

            const auto &hw = cudaHardwareMetadata().at(device().ordinal);
            const auto &sp = decl_.split;
            const auto &rp = decl_.rope;
            const auto &prep = std::get<QwenSparseAttentionStage::PrepareReplay>(decl_.replay_stage.operation);
            const auto &kva = decl_.kv_append;

            auto outPtr = [](const OutputLease &o) -> void * {
                return static_cast<std::byte *>(o.tensor->gpu_data_ptr()) + o.byte_offset;
            };
            auto weightPtr = [](const ITensor *t) -> const float * {
                const auto *tb = dynamic_cast<const TensorBase *>(t);
                return tb ? static_cast<const float *>(tb->gpu_data_ptr()) : nullptr;
            };

            const auto *row_params_ptr = static_cast<const attention::AttentionDeviceParams *>(outPtr(outputs_[5]));
            const auto kv_view = kva.kv_cache->acquireSparseAttentionNativeView(
                kva.layer_idx, kva.seq_idx, row_params_ptr, attention::AttentionKeyCachePolicy{}, stream);
            require(kv_view.keys.codec == TensorType::FP16 && kv_view.values.codec == TensorType::FP16 &&
                    kv_view.keys.data && kv_view.values.data &&
                    kv_view.keys.token_stride_bytes == kv_view.values.token_stride_bytes &&
                    kv_view.keys.head_stride_bytes == kv_view.values.head_stride_bytes,
                "Retained CUDA QSA Pre-Attention requires native FP16 ring KV banks");

            int *ring_head = kva.kv_cache->mutableDeviceRingHeadPtr(kva.layer_idx, kva.seq_idx);
            int *cached_tokens = kva.kv_cache->mutableDeviceCachedTokenCountPtr(kva.layer_idx, kva.seq_idx);
            require(ring_head && cached_tokens,
                "Retained CUDA QSA Pre-Attention requires mutable canonical device KV ring metadata");

            const int rotary_dim = std::max(2, static_cast<int>(std::lround(
                static_cast<double>(kHeadDim) * static_cast<double>(rp.partial_rotary_factor))));
            scratch_ = workspace_->retainBuffer(kCudaQSAScratchBank, 0, native::retainedCUDAQSAScratchBytes());

            native::CUDARetainedQSAPreInvocation inv{.hc = hc_prefix};
            inv.wq = bindNativeCUDAProj(
                q_engine_, nullptr, qkv.n_q,
                qkv.original_output_slice_q ? qkv.original_output_slice_q->policyColumns() : qkv.n_q,
                kHidden, hw.sm_count, qkv.original_output_slice_q);
            inv.wk = bindNativeCUDAProj(
                k_engine_, static_cast<float *>(outPtr(outputs_[2])), qkv.n_k,
                qkv.original_output_slice_k ? qkv.original_output_slice_k->policyColumns() : qkv.n_k,
                kHidden, hw.sm_count, qkv.original_output_slice_k);
            inv.wv = bindNativeCUDAProj(
                v_engine_, static_cast<float *>(outPtr(outputs_[3])), qkv.n_v,
                qkv.original_output_slice_v ? qkv.original_output_slice_v->policyColumns() : qkv.n_v,
                kHidden, hw.sm_count, qkv.original_output_slice_v);
            inv.qsa_key = bindCUDAIndexerProj(decl_.key_proj, key_engine_, 128, hw.sm_count);
            inv.qsa_query = bindCUDAIndexerProj(decl_.query_proj, query_engine_, 512, hw.sm_count);

            inv.Q = static_cast<float *>(outPtr(outputs_[0]));
            inv.fa_gate = static_cast<float *>(outPtr(outputs_[1]));
            inv.q_norm_gamma = decl_.q_norm ? weightPtr(decl_.q_norm->gamma) : nullptr;
            inv.k_norm_gamma = decl_.k_norm ? weightPtr(decl_.k_norm->gamma) : nullptr;
            inv.qk_norm_eps = decl_.q_norm ? decl_.q_norm->eps : (decl_.k_norm ? decl_.k_norm->eps : 1e-6f);
            inv.local_q_heads = sp.n_heads;
            inv.local_kv_heads = kva.kv_cache->local_n_kv_heads();
            inv.rotary_dim = rotary_dim;
            inv.rope_neg_log_theta = -std::log(rp.theta_base);
            inv.skip_k_rope = rp.skip_k;

            inv.position_ids_device = static_cast<const std::int32_t *>(rp.position_ids_device);
            inv.d_ring_head = ring_head;
            inv.d_cached_tokens = cached_tokens;
            inv.kv_token_capacity = kva.kv_cache->max_seq_len();
            inv.kv_keys_data = const_cast<void *>(kv_view.keys.data);
            inv.kv_values_data = const_cast<void *>(kv_view.values.data);
            inv.kv_token_stride_bytes = kv_view.keys.token_stride_bytes;
            inv.kv_head_stride_bytes = kv_view.keys.head_stride_bytes;

            inv.replay_out = static_cast<llaminar2::qwen4exp::QwenSparseReplay *>(outPtr(outputs_[4]));
            inv.row_params_out = static_cast<attention::AttentionDeviceParams *>(outPtr(outputs_[5]));
            inv.scratch = scratch_->data();
            inv.scratch_bytes = native::retainedCUDAQSAScratchBytes();

            workspace_id_ = workspace_->id();
            admission_ = native::prepareCUDARetainedQSAPreAttention(device(), inv, workspace_id_);
            stream_ = stream;
            validateStorage(stream);
            return true;
        }

        bool execute(IDeviceContext *ctx) override
        {
            require(ctx && ctx->deviceId() == device(), "Retained CUDA QSA Pre-Attention changed participant");
            const auto execution = gpuExecution();
            validateStorage(execution.nativeStream());
            for (const auto &in : phase_inputs_)
                execution.requirePreparedInput(in.tensor);
            for (auto *w : phase_weights_)
                execution.requirePreparedInput(w);
            for (auto *w : raw_weights_)
                execution.requirePreparedInput(w);
            for (const auto &o : phase_outputs_)
                execution.requirePreparedOutput(o.tensor);
            for (const auto &o : outputs_)
                execution.requirePreparedOutput(o.tensor);

            native::launchCUDARetainedQSAPreAttention(*admission_, execution.nativeStream(), workspace_id_);

            for (const auto &o : phase_outputs_)
                execution.publish(o.tensor);
            for (const auto &o : outputs_)
                execution.publish(o.tensor);

            PerfStatsCollector::addCounter("kernel", "cuda_retained_qsa_pre_megakernel_calls",
                1.0, "graph_capture", device().toString(),
                {{"m", "1"}, {"q_heads", std::to_string(decl_.split.n_heads)}});
            return true;
        }

        std::uint64_t graphCaptureVariantSignature() const override
        {
            if (admission_)
                validateStorage(stream_, /*validate_hc=*/false);
            std::uint64_t h = 0x4355444151534150ull;
            auto mix = [&h](std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
            mix(hc_->phaseDeclaration().graphCaptureVariantSignature());
            mix(static_cast<std::uint64_t>(decl_.split.n_heads));
            mix(static_cast<std::uint64_t>(decl_.qkv.n_k));
            mix(decl_.rope.skip_k ? 1ull : 0ull);
            mix(reinterpret_cast<std::uintptr_t>(decl_.rope.position_ids_device));
            mix(static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(decl_.rope.theta_base)));
            mix(decl_.query_proj ? 1ull : 0ull);
            if (decl_.query_proj)
                mix(decl_.query_proj->projection_row_work ? 1ull : 0ull);
            for (const auto &o : outputs_)
                mix(reinterpret_cast<std::uintptr_t>(o.tensor) ^ o.byte_offset);
            return h ? h : 1ull;
        }

        void invalidateKernelDynamicState() override
        {
            admission_.reset();
            scratch_.reset();
            for (auto &o : outputs_)
                o.lease.reset();
            q_engine_ = k_engine_ = v_engine_ = key_engine_ = query_engine_ = nullptr;
            stream_ = nullptr;
            seal_.reset();
            hc_->invalidatePhaseBindings();
        }

        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            const bool ownsWrite = hc_->trunkSource() == Qwen4ExpHCTrunkSource::PreviousWrite;
            static constexpr std::array<const char *, 8> kHcLabels{
                "hc_post", "normalized", "raw_down", "mixed", "hc_injection",
                "hc_extra_0", "hc_extra_1", "hc_extra_2"};
            for (std::size_t i = 0; i < phase_outputs_.size(); ++i)
            {
                const std::size_t label_idx = i + (ownsWrite ? 0u : 1u);
                const char *label = kHcLabels[std::min(label_idx, kHcLabels.size() - 1u)];
                info.addOutput(label, phase_outputs_[i].tensor, 1, phase_outputs_[i].tensor->cols());
            }
            static constexpr std::array<const char *, 8> kPreLabels{
                "q", "fa_gate", "k", "v", "replay", "row_params", "qsa_key", "qsa_query"};
            for (std::size_t i = 0; i < outputs_.size(); ++i)
            {
                const char *label = kPreLabels[std::min(i, kPreLabels.size() - 1u)];
                info.addOutput(label, outputs_[i].tensor, 1, outputs_[i].byte_extent / sizeof(float));
            }
            return info;
        }

    private:
        void validateStorage(void *stream, bool validate_hc = true) const
        {
            require(admission_ && stream == stream_ && workspace_ &&
                    workspace_->id() == workspace_id_ &&
                    scratch_ && scratch_->data() != nullptr,
                "Retained CUDA QSA Pre-Attention captured stream or workspace changed");
            if (validate_hc)
                hc_->validatePhaseBindings(stream);
            for (const auto &o : outputs_)
                TransferEngine::instance().requireCapturedDeviceStorage(
                    *o.lease, *o.tensor, device(), o.byte_offset + o.byte_extent);
        }

        std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc_;
        CUDAQSAPreDeclaration decl_;
        std::optional<std::array<Qwen4ExpFloatBinding, 2>> optional_;
        std::vector<Qwen4ExpFloatBinding> phase_inputs_;
        std::vector<Qwen4ExpFloatBinding> phase_outputs_;
        std::vector<ITensor *> phase_weights_;
        std::vector<ITensor *> raw_weights_;
        std::vector<OutputLease> outputs_;
        TensorBase *mixed_ = nullptr;
        ITensorGemm *q_engine_ = nullptr;
        ITensorGemm *k_engine_ = nullptr;
        ITensorGemm *v_engine_ = nullptr;
        ITensorGemm *key_engine_ = nullptr;
        ITensorGemm *query_engine_ = nullptr;
        std::optional<std::pair<std::uint64_t, std::uint64_t>> seal_;
        DeviceWorkspaceManager *workspace_ = nullptr;
        std::uint64_t workspace_id_ = 0;
        void *stream_ = nullptr;
        std::shared_ptr<const WorkspaceBufferLease> scratch_;
        std::optional<native::CUDARetainedQSAPreAdmission> admission_;
    };

    /** @brief Sealed post-attention subgraph declaration consumed by RetainedCUDAQSAPostAttentionStage. */
    struct CUDAQSAPostDeclaration
    {
        QwenSparseAttentionStage::Params attn_stage;
        std::optional<QwenSparseAttentionStage::CommitPartial> commit_partial;
        AttentionOutputGateStage::Params gate;
        GEMMStage::Params wo;
    };

    class RetainedCUDAQSAPostAttentionStage final
        : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        explicit RetainedCUDAQSAPostAttentionStage(CUDAQSAPostDeclaration decl)
            : IComputeStage(decl.wo.device_id),
              decl_(std::move(decl)),
              attn_sel_(std::get<QwenSparseAttentionStage::SelectedAttention>(decl_.attn_stage.operation))
        {
            require(device().is_cuda() && decl_.attn_stage.rowCapacity == 1 &&
                    (attn_sel_.arithmetic == llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals ||
                     attn_sel_.arithmetic == llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33) &&
                    attn_sel_.rowParameters && attn_sel_.gpuPartials &&
                    decl_.gate.seq_len == 1 && decl_.gate.feature_dim == attn_sel_.queryHeads * kHeadDim &&
                    decl_.wo.m == 1 && decl_.wo.n == kHidden && decl_.wo.k == attn_sel_.queryHeads * kHeadDim &&
                    decl_.wo.prepared_store && decl_.wo.prepared_ref &&
                    decl_.wo.prepared_ref->device == device(),
                "Retained CUDA QSA Post-Attention requires valid FP16 tiled M1 QSA declarations");
            if (attn_sel_.native.keyPolicy.transformsOnRead() &&
                attn_sel_.keyRotarySource == llaminar2::qwen4exp::QwenSparseRotarySource::CanonicalEngineTheta)
                key_neg_log_theta_ = -std::log(attn_sel_.native.keyPolicy.rope_theta);

            auto addIn = [&](TensorBase *t, BufferId id) {
                require(t != nullptr, "Retained CUDA QSA Post-Attention input missing");
                inputs_.push_back({t, id});
            };
            addIn(decl_.attn_stage.replay.tensor, decl_.attn_stage.replay.buffer);
            addIn(attn_sel_.query.tensor, attn_sel_.query.buffer);
            addIn(attn_sel_.selected.tensor, attn_sel_.selected.buffer);
            addIn(attn_sel_.selectedCount.tensor, attn_sel_.selectedCount.buffer);
            addIn(attn_sel_.rowParameters->tensor, attn_sel_.rowParameters->buffer);
            addIn(dynamic_cast<TensorBase *>(decl_.gate.gate), *decl_.gate.gate_buffer_id);
            if (decl_.commit_partial)
            {
                const auto &cp = *decl_.commit_partial;
                addIn(cp.priorPartial.tensor, cp.priorPartial.buffer);
                addIn(cp.rawKey.tensor, cp.rawKey.buffer);
                if (const auto *b = std::get_if<Qwen4ExpIntBinding>(&cp.acceptedRows.value))
                    addIn(b->tensor, b->buffer);
            }

            const std::size_t parts = static_cast<std::size_t>(
                llaminar2::qwen4exp::qwenSparseAttentionPartialCount(attn_sel_.arithmetic));
            outputs_.push_back({attn_sel_.gpuPartials->tensor, attn_sel_.gpuPartials->buffer,
                attn_sel_.gpuPartials->byteOffset,
                static_cast<std::size_t>(attn_sel_.queryHeads) * parts * sizeof(llaminar2::qwen4exp::QwenSparseAttentionPartial), {}});
            outputs_.push_back({attn_sel_.output.tensor, attn_sel_.output.buffer,
                attn_sel_.output.byteOffset,
                static_cast<std::size_t>(attn_sel_.queryHeads) * kHeadDim * sizeof(float), {}});
            outputs_.push_back({dynamic_cast<TensorBase *>(decl_.wo.C), *decl_.wo.c_buffer_id, 0,
                static_cast<std::size_t>(kHidden) * sizeof(float), {}});
            if (attn_sel_.observations)
            {
                outputs_.push_back({attn_sel_.observations->tensor, attn_sel_.observations->buffer,
                    attn_sel_.observations->byteOffset,
                    static_cast<std::size_t>(attn_sel_.queryHeads) * sizeof(llaminar2::qwen4exp::QwenSparseAttentionObservation), {}});
            }
            if (decl_.commit_partial)
            {
                const auto &cp = *decl_.commit_partial;
                outputs_.push_back({cp.outputPartial.tensor, cp.outputPartial.buffer,
                    cp.outputPartial.byteOffset, 4u * 128u * sizeof(float), {}});
            }
        }

        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::GPU_CUDA; }
        bool isGraphCapturable() const override { return admission_.has_value(); }
        bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return GraphLaunchPreparationPolicy::CaptureOnly;
        }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }

        bool transitionGraphCaptureActivity(IDeviceContext *context, void *stream,
                                            GraphCaptureActivityTransition transition) override
        {
            if (!context || context->deviceId() != device() || !stream)
                return false;
            if (transition == GraphCaptureActivityTransition::Completed &&
                PerfStatsCollector::isDomainEnabled("gpu_graph_inventory"))
            {
                PerfStatsCollector::addCounter("gpu_graph_inventory", "qsa_attention_program_captures",
                    1.0, "capture_setup", device().to_string(),
                    {{"policy", std::string(llaminar2::qwen4exp::qwenSparseAttentionArithmeticName(attn_sel_.arithmetic))},
                     {"rows", "1"},
                     {"query_heads", std::to_string(attn_sel_.queryHeads)},
                     {"kv_heads", std::to_string(attn_sel_.kvHeads)},
                     {"phase", "decode"},
                     {"program", std::string(llaminar2::qwen4exp::qwenSparseAttentionArithmeticName(attn_sel_.arithmetic))},
                     {"parts", std::to_string(llaminar2::qwen4exp::qwenSparseAttentionPartialCount(attn_sel_.arithmetic))},
                     {"capture_signature", std::to_string(graphCaptureVariantSignature())}});
            }
            return true;
        }

        StageBufferContract bufferContract() const override
        {
            StageBufferContract c;
            auto hasOut = [&](BufferId id) {
                return std::any_of(outputs_.begin(), outputs_.end(),
                    [&](const auto &o) { return o.id == id; });
            };
            auto hasIn = [&](BufferId id) {
                return std::any_of(inputs_.begin(), inputs_.end(),
                    [&](const auto &in) { return in.second == id; });
            };
            auto added = [&](const auto &views, BufferId id) {
                return std::any_of(views.begin(), views.end(),
                    [&](const auto &v) { return v.id == id; });
            };
            for (const auto &in : inputs_)
            {
                if (hasOut(in.second))
                {
                    if (!added(c.inouts, in.second))
                        c.addPreallocatedInOut(in.second);
                }
                else if (!added(c.inputs, in.second))
                {
                    c.addInput(in.second);
                }
            }
            for (const auto &o : outputs_)
            {
                if (!hasIn(o.id) && !added(c.outputs, o.id))
                    c.addOutput(o.id);
            }
            c.addPreparedWeight(const_cast<ITensor *>(decl_.wo.B),
                                decl_.wo.prepared_store, *decl_.wo.prepared_ref);
            if (attn_sel_.keyCosine)
                c.addWeight(const_cast<TensorBase *>(attn_sel_.keyCosine->tensor));
            if (attn_sel_.keySine)
                c.addWeight(const_cast<TensorBase *>(attn_sel_.keySine->tensor));
            return c;
        }

        StageBufferRequirements getBufferRequirements() const override { return {}; }

        WorkspaceRequirements getWorkspaceRequirements(int, int, int) const override
        {
            return retainedCUDAQSAAttentionWorkspace();
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            require(!admission_ || workspace_ == workspace,
                "Retained CUDA QSA Post-Attention workspace changed while captured");
            workspace_ = workspace;
        }

        void unbindWorkspace() override
        {
            require(!admission_, "Retained CUDA QSA Post-Attention must retire before workspace release");
            workspace_ = nullptr;
        }

        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            require(ctx && ctx->deviceId() == device() && workspace_ && workspace_->device() == device() &&
                    reinterpret_cast<std::uintptr_t>(stream) > 2 && !isGraphCaptureActive(),
                "Retained CUDA QSA Post-Attention requires cold context, workspace, and stream");
            if (admission_)
            {
                validateStorage(stream);
                return true;
            }
            setGPUStream(stream);
            for (auto &o : outputs_)
            {
                o.lease = TransferEngine::instance().retainCapturedDeviceStorage(
                    *o.tensor, device(), o.byte_offset + o.byte_extent);
            }

            const auto wo_binding = decl_.wo.prepared_store->binding(*decl_.wo.prepared_ref);
            wo_engine_ = decl_.wo.prepared_store->gemmKernel(*decl_.wo.prepared_ref);
            require(wo_binding && wo_binding->immutable && wo_binding->tensor == decl_.wo.B && wo_engine_,
                "Retained CUDA QSA Post-Attention lost immutable Wo prepared source");

            auto ptrAt = [](TensorBase *t, std::size_t off) -> void * {
                return static_cast<std::byte *>(t->gpu_data_ptr()) + off;
            };
            const auto *replay_ptr = static_cast<const llaminar2::qwen4exp::QwenSparseReplay *>(
                ptrAt(decl_.attn_stage.replay.tensor, decl_.attn_stage.replay.byteOffset));
            const auto *row_params_ptr = static_cast<const attention::AttentionDeviceParams *>(
                ptrAt(attn_sel_.rowParameters->tensor, attn_sel_.rowParameters->byteOffset));

            auto kv = attn_sel_.native.cache.acquireSparseAttentionNativeView(
                attn_sel_.native.layer, attn_sel_.native.sequence,
                row_params_ptr, attn_sel_.native.keyPolicy, stream);
            kv.key_rotary_source = attn_sel_.keyRotarySource;
            kv.key_rotary_dim = attn_sel_.keyRotaryDim;
            kv.key_negative_log_theta = key_neg_log_theta_;
            if (attn_sel_.keyCosine)
            {
                kv.key_cosine = static_cast<const float *>(attn_sel_.keyCosine->tensor->gpu_data_ptr());
                kv.key_sine = static_cast<const float *>(attn_sel_.keySine->tensor->gpu_data_ptr());
                kv.key_position_capacity = attn_sel_.keyPositionCapacity;
            }

            llaminar2::qwen4exp::QwenSparseAttention attn_args{
                .replay = replay_ptr,
                .query = static_cast<const float *>(ptrAt(attn_sel_.query.tensor, attn_sel_.query.byteOffset)),
                .kv = kv,
                .selected = static_cast<const std::int32_t *>(ptrAt(attn_sel_.selected.tensor, attn_sel_.selected.byteOffset)),
                .selected_count = static_cast<const std::int32_t *>(ptrAt(attn_sel_.selectedCount.tensor, attn_sel_.selectedCount.byteOffset)),
                .gate = nullptr,
                .output = static_cast<float *>(ptrAt(attn_sel_.output.tensor, attn_sel_.output.byteOffset)),
                .gpu_partials = static_cast<llaminar2::qwen4exp::QwenSparseAttentionPartial *>(
                    ptrAt(attn_sel_.gpuPartials->tensor, attn_sel_.gpuPartials->byteOffset)),
                .observations = attn_sel_.observations
                    ? static_cast<llaminar2::qwen4exp::QwenSparseAttentionObservation *>(
                          ptrAt(attn_sel_.observations->tensor, attn_sel_.observations->byteOffset))
                    : nullptr,
                .row_capacity = 1,
                .token_capacity = decl_.attn_stage.tokenCapacity,
                .query_heads = attn_sel_.queryHeads,
                .kv_heads = attn_sel_.kvHeads,
                .query_head_start = attn_sel_.queryHeadStart,
                .kv_head_start = attn_sel_.kvHeadStart,
                .arithmetic = attn_sel_.arithmetic};

            scratch_ = workspace_->retainBuffer(kCudaQSAScratchBank, 0, native::retainedCUDAQSAScratchBytes());
            const auto &hw = cudaHardwareMetadata().at(device().ordinal);

            llaminar2::qwen4exp::QwenSparsePartialCommit commit_args{
                .replay = nullptr,
                .prior_partial = nullptr,
                .raw_keys = nullptr,
                .acceptance = AcceptedRequestPrefixes::fixed(DeviceRequestRowRanges::fullyActive(1), 1),
                .partial_keys = nullptr};
            if (decl_.commit_partial)
            {
                const auto &cp = *decl_.commit_partial;
                const std::int32_t *accepted_ptr = nullptr;
                if (const auto *b = std::get_if<Qwen4ExpIntBinding>(&cp.acceptedRows.value))
                    accepted_ptr = static_cast<const std::int32_t *>(ptrAt(b->tensor, b->byteOffset));
                else
                    accepted_ptr = std::get<Qwen4ExpNativeIntInput>(cp.acceptedRows.value).data;
                commit_args = {
                    .replay = replay_ptr,
                    .prior_partial = static_cast<const float *>(ptrAt(cp.priorPartial.tensor, cp.priorPartial.byteOffset)),
                    .raw_keys = static_cast<const float *>(ptrAt(cp.rawKey.tensor, cp.rawKey.byteOffset)),
                    .acceptance = AcceptedRequestPrefixes::counted(
                        DeviceRequestRowRanges::deviceCounted(1, 1, &replay_ptr->live_rows),
                        accepted_ptr),
                    .partial_keys = static_cast<float *>(ptrAt(cp.outputPartial.tensor, cp.outputPartial.byteOffset))};
            }

            native::CUDARetainedQSAPostInvocation inv{
                .attention = attn_args,
                .commit = commit_args,
                .has_commit = decl_.commit_partial.has_value(),
                .fa_gate = static_cast<const float *>(dynamic_cast<const TensorBase *>(decl_.gate.gate)->gpu_data_ptr()),
                .wo = bindNativeCUDAProj(
                    wo_engine_,
                    static_cast<float *>(dynamic_cast<TensorBase *>(decl_.wo.C)->gpu_data_ptr()),
                    kHidden, kHidden, attn_sel_.queryHeads * kHeadDim, hw.sm_count, std::nullopt),
                .scratch = scratch_->data(),
                .scratch_bytes = native::retainedCUDAQSAScratchBytes()};

            workspace_id_ = workspace_->id();
            admission_ = native::prepareCUDARetainedQSAPostAttention(device(), inv, workspace_id_);
            stream_ = stream;
            validateStorage(stream);
            return true;
        }

        bool execute(IDeviceContext *ctx) override
        {
            require(ctx && ctx->deviceId() == device(), "Retained CUDA QSA Post-Attention changed participant");
            const auto execution = gpuExecution();
            validateStorage(execution.nativeStream());
            for (const auto &in : inputs_)
                execution.requirePreparedInput(in.first);
            for (const auto &o : outputs_)
                execution.requirePreparedOutput(o.tensor);

            auto *cuda_stream = static_cast<cudaStream_t>(execution.nativeStream());
            llaminar2::qwen4exp::cudaQwenSparseFP16TilesOnly(admission_->invocation.attention, cuda_stream);
            native::launchCUDARetainedQSAPostAttention(*admission_, execution.nativeStream(), workspace_id_);

            for (const auto &o : outputs_)
                execution.publish(o.tensor);

            PerfStatsCollector::addCounter("kernel", "cuda_retained_qsa_post_megakernel_calls",
                1.0, "graph_capture", device().toString(),
                {{"m", "1"}, {"q_heads", std::to_string(attn_sel_.queryHeads)}});
            return true;
        }

        std::uint64_t graphCaptureVariantSignature() const override
        {
            if (admission_)
                validateStorage(stream_);
            std::uint64_t h = 0x435544415153414full;
            auto mix = [&h](std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
            mix(static_cast<std::uint64_t>(attn_sel_.arithmetic));
            mix(static_cast<std::uint64_t>(attn_sel_.queryHeads));
            mix(static_cast<std::uint64_t>(attn_sel_.queryHeadStart));
            mix(static_cast<std::uint64_t>(attn_sel_.kvHeads));
            mix(decl_.commit_partial ? 1ull : 0ull);
            mix(decl_.wo.prepared_ref ? decl_.wo.prepared_ref->binding_id : 0ull);
            for (const auto &o : outputs_)
                mix(reinterpret_cast<std::uintptr_t>(o.tensor) ^ o.byte_offset);
            return h ? h : 1ull;
        }

        void invalidateKernelDynamicState() override
        {
            admission_.reset();
            scratch_.reset();
            for (auto &o : outputs_)
                o.lease.reset();
            wo_engine_ = nullptr;
            stream_ = nullptr;
        }

        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            static constexpr std::array<const char *, 6> kPostLabels{
                "gpu_partials", "attn_output", "attn_proj", "observations", "commit_partial", "post_extra"};
            for (std::size_t i = 0; i < outputs_.size(); ++i)
            {
                const char *label = (i == 3 && !attn_sel_.observations)
                    ? "commit_partial"
                    : kPostLabels[std::min(i, kPostLabels.size() - 1u)];
                info.addOutput(label, outputs_[i].tensor, 1, outputs_[i].byte_extent / sizeof(float));
            }
            return info;
        }

    private:
        void validateStorage(void *stream) const
        {
            require(admission_ && stream == stream_ && workspace_ &&
                    workspace_->id() == workspace_id_ &&
                    workspace_->getBuffer(kCudaQSAScratchBank) == scratch_->data(),
                "Retained CUDA QSA Post-Attention captured stream or workspace changed");
            for (const auto &o : outputs_)
                TransferEngine::instance().requireCapturedDeviceStorage(
                    *o.lease, *o.tensor, device(), o.byte_offset + o.byte_extent);
            require(decl_.wo.prepared_store->gemmKernel(*decl_.wo.prepared_ref) == wo_engine_,
                "Retained CUDA QSA Post-Attention Wo engine changed");
        }

        CUDAQSAPostDeclaration decl_;
        QwenSparseAttentionStage::SelectedAttention attn_sel_;
        float key_neg_log_theta_ = 0.0f;
        std::vector<std::pair<TensorBase *, BufferId>> inputs_;
        std::vector<OutputLease> outputs_;
        ITensorGemm *wo_engine_ = nullptr;
        DeviceWorkspaceManager *workspace_ = nullptr;
        std::uint64_t workspace_id_ = 0;
        void *stream_ = nullptr;
        std::shared_ptr<const WorkspaceBufferLease> scratch_;
        std::optional<native::CUDARetainedQSAPostAdmission> admission_;
    };

    bool isNativeQuantisedTensor(const ITensor *t) noexcept
    {
        if (!t)
            return false;
        const auto nt = t->native_type();
        return nt != TensorType::FP32 && nt != TensorType::FP16 && nt != TensorType::BF16;
    }
} // namespace

WorkspaceRequirements retainedCUDAQSAAttentionWorkspace()
{
    WorkspaceRequirements req;
    req.buffers.emplace_back(kCudaQSAScratchBank, native::retainedCUDAQSAScratchBytes(), 16, true,
        WorkspaceExecutionRegime::Any, WorkspaceContentLifetime::SerialGraphFamily);
    return req;
}

void lowerRetainedCUDAQSAAttention(ComputeGraph &graph,
                                   const std::string &layerPrefix,
                                   const std::string &previousInputBoundary)
{
    using Primitive = Qwen4ExpPrimitiveStage;
    const std::string qkv_name = layerPrefix + "qkv_proj";
    const std::string split_name = layerPrefix + "q_gate_split";
    const std::string q_norm_name = layerPrefix + "q_norm";
    const std::string k_norm_name = layerPrefix + "k_norm";
    const std::string rope_name = layerPrefix + "rope";
    const std::string replay_name = layerPrefix + "qsa_prepare_replay";
    const std::string kv_append_name = layerPrefix + "kv_append";
    const std::string key_proj_name = layerPrefix + "qsa_key_proj";
    const std::string query_proj_name = layerPrefix + "qsa_query_proj";
    const std::string attn_name = layerPrefix + "attention";
    const std::string commit_name = layerPrefix + "qsa_commit_partial";
    const std::string gate_name = layerPrefix + "attn_output_gate";
    const std::string wo_name = layerPrefix + "wo_proj";

    const auto *qkv_node = graph.getNode(qkv_name);
    const auto *split_node = graph.getNode(split_name);
    const auto *rope_node = graph.getNode(rope_name);
    const auto *replay_node = graph.getNode(replay_name);
    const auto *kva_node = graph.getNode(kv_append_name);
    const auto *key_node = graph.getNode(key_proj_name);
    const auto *query_node = graph.getNode(query_proj_name);
    const auto *q_norm_node = graph.getNode(q_norm_name);
    const auto *k_norm_node = graph.getNode(k_norm_name);

    const auto *qkv_stage = qkv_node ? dynamic_cast<const FusedQKVGEMMStage *>(qkv_node->stage.get()) : nullptr;
    const auto *split_stage = split_node ? dynamic_cast<const QGateSplitStage *>(split_node->stage.get()) : nullptr;
    const auto *rope_stage = rope_node ? dynamic_cast<const RoPEStage *>(rope_node->stage.get()) : nullptr;
    const auto *replay_stage = replay_node ? dynamic_cast<const QwenSparseAttentionStage *>(replay_node->stage.get()) : nullptr;
    const auto *kva_stage = kva_node ? dynamic_cast<const KVCacheAppendStage *>(kva_node->stage.get()) : nullptr;
    const auto *key_stage = key_node ? dynamic_cast<const GEMMStage *>(key_node->stage.get()) : nullptr;
    const auto *query_stage = query_node ? dynamic_cast<const GEMMStage *>(query_node->stage.get()) : nullptr;
    const auto *q_norm_stage = q_norm_node ? dynamic_cast<const QKNormStage *>(q_norm_node->stage.get()) : nullptr;
    const auto *k_norm_stage = k_norm_node ? dynamic_cast<const QKNormStage *>(k_norm_node->stage.get()) : nullptr;

    if (!qkv_stage || !split_stage || !rope_stage || !replay_stage || !kva_stage || !key_stage)
        return;

    const auto &qkv = qkv_stage->getParams();
    const auto &sp = split_stage->getParams();
    const auto &rp = rope_stage->getParams();
    const auto &rprep_params = replay_stage->getParams();
    const auto *prep = std::get_if<QwenSparseAttentionStage::PrepareReplay>(&rprep_params.operation);
    const auto &kva = kva_stage->getParams();
    const auto &kp = key_stage->getParams();

    // Check Pre-Attention eligibility (ordinary M1 CUDA, FP16 ring KV, NativeQuantised Q/K/V, no biases).
    if (!qkv.device_id.is_cuda() || qkv.m != 1 || qkv.k != kHidden ||
        qkv.projection_set != AttentionProjectionSet::QueryKeyValue ||
        qkv.bias_q || qkv.bias_k || qkv.bias_v ||
        !qkv.prepared_store || !qkv.prepared_ref_q || !qkv.prepared_ref_k || !qkv.prepared_ref_v ||
        !isNativeQuantisedTensor(qkv.wq) || !isNativeQuantisedTensor(qkv.wk) || !isNativeQuantisedTensor(qkv.wv) ||
        !qkv.input || !qkv.input_buffer_id || !qkv.output_k_buffer_id || !qkv.output_v_buffer_id ||
        sp.seq_len != 1 || sp.head_dim != kHeadDim || !sp.output_q_buffer_id || !sp.output_gate_buffer_id ||
        !rp.position_ids_device || rp.Q_out || rp.K_out || rp.operand_set != RoPEOperandSet::QueryAndOptionalKey ||
        !prep || !prep->rowParameters || rprep_params.rowCapacity != 1 || prep->liveRows.has_value() ||
        !kva.kv_cache || !kva.kv_cache->isGraphCaptureReady() ||
        kva.kv_cache->k_precision() != ActivationPrecision::FP16 ||
        kva.kv_cache->v_precision() != ActivationPrecision::FP16 ||
        kva.num_tokens != 1 || kva.batch_size != 1 || kva.request_sequence_lengths_device != nullptr ||
        kva.append_semantics != KVCacheAppendSemantics::Standard ||
        !kva.kv_cache->mutableDeviceRingHeadPtr(kva.layer_idx, kva.seq_idx) ||
        !kva.kv_cache->mutableDeviceCachedTokenCountPtr(kva.layer_idx, kva.seq_idx) ||
        kp.m != 1 || kp.n != 128 || kp.k != kHidden || kp.bias || kp.bias_tensor ||
        !kp.prepared_store || !kp.prepared_ref || !kp.c_buffer_id || kp.A != qkv.input)
        return;

    if (query_stage)
    {
        const auto &qp = query_stage->getParams();
        if (qp.m != 1 || qp.n != 512 || qp.k != kHidden || qp.bias || qp.bias_tensor ||
            !qp.prepared_store || !qp.prepared_ref || !qp.c_buffer_id || qp.A != qkv.input)
            return;
    }

    // Locate the direct HC ResidualRead producer feeding qkv_proj.
    std::string readName;
    for (const auto &dep : qkv_node->dependencies)
    {
        const auto *node = graph.getNode(dep);
        const auto *prim = node ? dynamic_cast<const Primitive *>(node->stage.get()) : nullptr;
        const auto *read = prim ? std::get_if<Primitive::ResidualRead>(&prim->getParams().operation) : nullptr;
        if (read && read->output.tensor == qkv.input && read->output.buffer == *qkv.input_buffer_id &&
            read->output.byteOffset == 0)
        {
            if (!readName.empty())
                return;
            readName = dep;
        }
    }
    if (readName.empty())
        return;

    // Lower Pre-Attention into ONE cooperative megakernel.
    const auto *boundary = previousInputBoundary.empty() ? nullptr : graph.getNode(previousInputBoundary);
    const auto *boundary_prim = boundary ? dynamic_cast<const Primitive *>(boundary->stage.get()) : nullptr;
    const auto *boundary_write = boundary_prim
        ? std::get_if<Primitive::ResidualWrite>(&boundary_prim->getParams().operation) : nullptr;
    const auto inject_policy = boundary_write
        ? Qwen4ExpHCInjectionOwnership::DiscoverInRegion
        : Qwen4ExpHCInjectionOwnership::KeepOriginalStage;
    auto region = Qwen4ExpHyperconnectionBuilder::describeFullNativeReadRegion(
        graph, previousInputBoundary, readName, inject_policy);
    const auto *normNode = graph.getNode(region.branch_norm);
    const auto &norm = std::get<Primitive::BranchRMSNorm>(
        static_cast<const Primitive &>(*normNode->stage).getParams().operation);

    auto source = Qwen4ExpHCTrunkSource::PreviousWrite;
    if (!boundary_write || boundary_write->output.tensor != norm.input.tensor ||
        boundary_write->output.buffer != norm.input.buffer ||
        boundary_write->output.byteOffset != norm.input.byteOffset)
    {
        source = Qwen4ExpHCTrunkSource::PublishedTrunk;
        region.previous_write.clear();
        region.injection.clear();
    }
    if (region.injection == graph.terminalNode())
        region.injection.clear();

    auto phase = Qwen4ExpHyperconnectionBuilder::retainFullNativeReadPhase(graph, region, source);
    CUDAQSAPreDeclaration pre_decl{
        .qkv = qkv,
        .split = sp,
        .q_norm = q_norm_stage ? std::optional{q_norm_stage->getParams()} : std::nullopt,
        .k_norm = k_norm_stage ? std::optional{k_norm_stage->getParams()} : std::nullopt,
        .rope = rp,
        .replay_stage = rprep_params,
        .kv_append = kva,
        .key_proj = kp,
        .query_proj = query_stage ? std::optional{query_stage->getParams()} : std::nullopt};
    auto pre_stage = std::make_unique<RetainedCUDAQSAPreAttentionStage>(std::move(phase), std::move(pre_decl));

    {
        GraphComposer composer(graph);
        std::vector<GraphNodeRef> members;
        if (!region.previous_write.empty())
            members.push_back(composer.ref(region.previous_write));
        members.push_back(composer.ref(region.branch_norm));
        members.push_back(composer.ref(region.down));
        members.push_back(composer.ref(region.up));
        members.push_back(composer.ref(region.read));
        if (!region.injection.empty())
            members.push_back(composer.ref(region.injection));
        members.push_back(composer.ref(qkv_name));
        members.push_back(composer.ref(split_name));
        if (q_norm_stage)
            members.push_back(composer.ref(q_norm_name));
        if (k_norm_stage)
            members.push_back(composer.ref(k_norm_name));
        members.push_back(composer.ref(rope_name));
        members.push_back(composer.ref(replay_name));
        members.push_back(composer.ref(key_proj_name));
        if (query_stage)
            members.push_back(composer.ref(query_proj_name));
        GraphNodeRef pre_terminal = composer.ref(kv_append_name);
        members.push_back(pre_terminal);
        (void)lowerLocalMegakernelRegion(graph, LocalMegakernelRegionSpec{
            .members = std::move(members),
            .terminal = pre_terminal,
            .replacement = std::move(pre_stage),
        });
    }

    // Lower Post-Attention (attention + commit_partial + attn_output_gate + wo_proj).
    const auto *attn_node = graph.getNode(attn_name);
    const auto *commit_node = graph.getNode(commit_name);
    const auto *gate_node = graph.getNode(gate_name);
    const auto *wo_node = graph.getNode(wo_name);
    const auto *attn_stage = attn_node ? dynamic_cast<const QwenSparseAttentionStage *>(attn_node->stage.get()) : nullptr;
    const auto *commit_stage = commit_node ? dynamic_cast<const QwenSparseAttentionStage *>(commit_node->stage.get()) : nullptr;
    const auto *gate_stage = gate_node ? dynamic_cast<const AttentionOutputGateStage *>(gate_node->stage.get()) : nullptr;
    const auto *wo_stage = wo_node ? dynamic_cast<const GEMMStage *>(wo_node->stage.get()) : nullptr;
    if (!attn_stage || !gate_stage || !wo_stage)
        return;

    const auto &ap = attn_stage->getParams();
    const auto *sel = std::get_if<QwenSparseAttentionStage::SelectedAttention>(&ap.operation);
    const auto &gp = gate_stage->getParams();
    const auto &wp = wo_stage->getParams();
    if (!sel || ap.rowCapacity != 1 ||
        (sel->arithmetic != llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals &&
         sel->arithmetic != llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33) ||
        !sel->rowParameters || !sel->gpuPartials || sel->gate.has_value() ||
        gp.seq_len != 1 || gp.feature_dim != sel->queryHeads * kHeadDim ||
        gp.input != sel->output.tensor || gp.output != sel->output.tensor || !gp.gate_buffer_id ||
        wp.m != 1 || wp.n != kHidden || wp.k != sel->queryHeads * kHeadDim ||
        wp.A != gp.output || !isNativeQuantisedTensor(wp.B) ||
        wp.bias || wp.bias_tensor || wp.alpha != 1.0f || wp.beta != 0.0f ||
        !wp.prepared_store || !wp.prepared_ref || !wp.c_buffer_id)
        return;

    std::optional<QwenSparseAttentionStage::CommitPartial> commit_partial;
    if (commit_stage)
    {
        const auto *cp = std::get_if<QwenSparseAttentionStage::CommitPartial>(&commit_stage->getParams().operation);
        if (!cp)
            return;
        commit_partial = *cp;
    }

    auto post_stage = std::make_unique<RetainedCUDAQSAPostAttentionStage>(CUDAQSAPostDeclaration{
        .attn_stage = ap,
        .commit_partial = std::move(commit_partial),
        .gate = gp,
        .wo = wp});

    {
        GraphComposer composer(graph);
        std::vector<GraphNodeRef> members;
        members.push_back(composer.ref(attn_name));
        if (commit_stage)
            members.push_back(composer.ref(commit_name));
        members.push_back(composer.ref(gate_name));
        GraphNodeRef post_terminal = composer.ref(wo_name);
        members.push_back(post_terminal);
        (void)lowerLocalMegakernelRegion(graph, LocalMegakernelRegionSpec{
            .members = std::move(members),
            .terminal = post_terminal,
            .replacement = std::move(post_stage),
        });
    }
}
} // namespace llaminar2
#endif
