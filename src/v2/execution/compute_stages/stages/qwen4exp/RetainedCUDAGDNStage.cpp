/**
 * @file RetainedCUDAGDNStage.cpp
 * @brief Participant-local CUDA Full GDN Attention megakernel graph stage and lowering.
 *
 * Replaces the entire 10-stage / 25-kernel CUDA GDN attention sequence
 * (HC ResidualWrite/BranchRMSNorm/Down/Injection/Up/Read + GDNProjection +
 * ShortConv1d + GDNRecurrence + GatedRMSNorm + ssm_out GEMM) with one single
 * cooperative CUDA megakernel across all CUDA topologies and MoE placements.
 */
#include "RetainedGDNPreAttentionStage.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDAHardwareMetadata.h"
#include "execution/compute_stages/IGraphOutputDemand.h"
#include "execution/compute_stages/stages/FullNativeHCFFNOutputDemandResolver.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/LocalMegakernelLowering.h"
#include "kernels/IKVCache.h"
#include "kernels/cuda/gdn/CUDAGatedDeltaNet.h"
#include "kernels/cuda/gdn/CUDAShortConvolution.h"
#include "kernels/cuda/gemm/CUDANativeQuantisedPrefillHostPolicy.h"
#include "kernels/cuda/qwen4exp/CUDARetainedGDN.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
namespace
{
    namespace native = cuda::qwen4exp;
    constexpr auto kCudaGDNScratchBank = "qfn_cuda_retained_gdn";
    constexpr int kHidden = 2560;
    constexpr std::array<const char *, 4> kProjNames{"qkv", "z", "alpha", "beta"};

    /** @brief Authenticate a complete original local owner before native recording. */
    void require(bool condition, const char *message)
    {
        if (!condition)
            throw std::invalid_argument(message);
    }

    /** @brief Immutable downstream CUDA GDN stages folded into one resident megakernel. */
    struct FullCUDAGDNDeclaration
    {
        ShortConv1dStage::Params conv;
        GDNRecurrenceStage::Params rec;
        GatedRMSNormStage::Params norm;
        GEMMStage::Params out_proj;
    };

    class RetainedCUDAGDNStage final : public IComputeStage, public IWorkspaceConsumer, public IGraphOutputDemand
    {
    public:
        /** @brief Seal all original HC and GDN declarations before graph mutation. */
        RetainedCUDAGDNStage(const GDNProjectionStage::Params &params,
                             std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc,
                             FullCUDAGDNDeclaration fullGdn)
            : IComputeStage(params.device_id),
              projection_(params),
              hc_(std::move(hc)),
              columns_{params.n_qkv, params.n_z, params.n_a, params.n_b},
              inputs_{params.w_qkv, params.w_z, params.w_a, params.w_b},
              refs_{params.prepared_ref_qkv, params.prepared_ref_z,
                    params.prepared_ref_a, params.prepared_ref_b},
              cached_{params.gemm_qkv, params.gemm_z, params.gemm_a, params.gemm_b},
              full_gdn_(std::move(fullGdn))
        {
            require(hc_ && device().is_cuda() && hc_->phaseDeclaration().device() == device() &&
                    params.m == 1 && params.k == kHidden &&
                    columns_[0] > 0 && (columns_[0] % 4) == 0 &&
                    columns_[1] > 0 && (columns_[1] % 4) == 0 &&
                    columns_[2] > 0 && columns_[2] == columns_[3] &&
                    !params.force_decode_equivalent_verifier_prefill && !params.verifier_row_range &&
                    params.qkv_operand_precision == QuantizedActivationOperandPrecision::SingleQ8 &&
                    params.prepared_store && params.input && params.input_buffer_id,
                "Retained CUDA GDN requires valid ordinary CUDA M1 QKV/Z/A/B declarations");

            for (const auto &output : hc_->phaseOutputs())
                if (output.tensor == params.input && output.buffer == *params.input_buffer_id &&
                    output.byteOffset == 0)
                    mixed_ = output.tensor;
            require(mixed_, "Retained CUDA GDN must consume its exact original HC mixed publication");

            const std::array output_tensors{params.output_qkv, params.output_z, params.output_a, params.output_b};
            const std::array ids{params.output_qkv_buffer_id, params.output_z_buffer_id,
                                 params.output_a_buffer_id, params.output_b_buffer_id};
            for (std::size_t i = 0; i < outputs_.size(); ++i)
            {
                require(inputs_[i] && refs_[i] && refs_[i]->device == device(),
                    "Retained CUDA GDN requires exact original participant projection references");
                outputs_[i].tensor = dynamic_cast<TensorBase *>(output_tensors[i]);
                const auto *tensor = outputs_[i].tensor;
                require(ids[i] && tensor && tensor != mixed_ && tensor->native_type() == TensorType::FP32 &&
                        tensor->rows() >= 1 && tensor->cols() == static_cast<std::size_t>(columns_[i]),
                    "Retained CUDA GDN requires original distinct FP32 projection output geometry");
                require(std::none_of(outputs_.begin(), outputs_.begin() + i,
                    [tensor](const auto &prior) { return prior.tensor == tensor; }),
                    "Retained CUDA GDN projection outputs cannot alias");
                outputs_[i].id = *ids[i];
            }

            const auto optional = hc_->optionalPhaseOutputs();
            for (const auto &opt : optional)
                require(opt.tensor && opt.byteOffset == 0,
                    "Retained CUDA GDN HC optional outputs require original arena bindings");
            optional_ = optional;
            hc_->sealFullOutputDemand(FullNativeHCFFNOutputDemand::none());
            for (const auto &output : hc_->phaseOutputs())
                if (std::none_of(phase_outputs_.begin(), phase_outputs_.end(),
                    [&](const auto &prior) { return prior.tensor == output.tensor; }))
                    phase_outputs_.push_back(output);
            phase_inputs_ = hc_->phaseInputs();
            for (const auto &output : outputs_)
            {
                const auto usesOutput = [&](const auto &view) { return view.tensor == output.tensor; };
                require(std::none_of(phase_inputs_.begin(), phase_inputs_.end(), usesOutput) &&
                        std::none_of(phase_outputs_.begin(), phase_outputs_.end(), usesOutput),
                    "Retained CUDA GDN projections cannot share an original HC port");
            }
            for (auto *weight : hc_->phaseDeclaration().bufferContract().weight_tensors)
                phase_weights_.push_back(weight);

            const auto &c = full_gdn_.conv;
            const auto &r = full_gdn_.rec;
            const auto &gn = full_gdn_.norm;
            const auto &o = full_gdn_.out_proj;
            const bool raw_qk = std::holds_alternative<GDNRawQKInput>(c.qk_contract) &&
                                std::holds_alternative<GDNRawQKInput>(r.qk_contract);
            const bool prepared_qk = std::holds_alternative<GDNPreparedQKBinding>(c.qk_contract) &&
                                     std::holds_alternative<GDNPreparedQKBinding>(r.qk_contract);
            require(c.device_id == device() && r.device_id == device() &&
                    gn.device_id == device() && o.device_id == device() &&
                    c.seq_len == 1 && c.request_count == 1 && c.kernel_size == 4 &&
                    c.channels == columns_[0] && (columns_[0] & 7) == 0 &&
                    c.input == outputs_[0].tensor && c.output && c.output_buffer_id && c.weight &&
                    !c.request_seq_lens_device && c.verifier_state_capture_rows == 0 &&
                    c.speculative_state_slot_rows == 0 && (raw_qk || prepared_qk) &&
                    dynamic_cast<const CUDAShortConvolution *>(c.kernel) != nullptr,
                "Retained CUDA GDN requires exact ordinary M1 CUDA ShortConv1d");
            require(r.seq_len == 1 && r.request_count == 1 &&
                    r.d_k == 128 && r.d_v == 128 &&
                    r.n_heads == columns_[2] && r.n_heads >= 1 &&
                    static_cast<unsigned>(r.n_heads) <= native::kMaxCUDARetainedGDNVHeads &&
                    r.n_heads * 128 == columns_[1] && (columns_[1] & 7) == 0 &&
                    r.n_k_heads > 0 && 2 * r.n_k_heads * 128 + r.n_heads * 128 == columns_[0] &&
                    r.Q == c.output && r.K == c.output && r.V == c.output &&
                    r.alpha == outputs_[2].tensor && r.beta == outputs_[3].tensor &&
                    r.A_log && r.dt_bias && r.output && r.output_buffer_id &&
                    !r.request_seq_lens_device && !r.gpu_retained_workspace &&
                    r.verifier_state_capture_rows == 0 && r.speculative_state_slot_rows == 0 &&
                    dynamic_cast<const CUDAGatedDeltaNet *>(r.kernel) != nullptr,
                "Retained CUDA GDN requires exact ordinary M1 CUDA GDNRecurrence");
            require(gn.seq_len == 1 && gn.feature_dim == columns_[1] && gn.norm_dim == 128 &&
                    gn.input == r.output && gn.output == r.output && gn.gate == outputs_[1].tensor &&
                    gn.gamma && gn.gamma->numel() == 128 &&
                    std::isfinite(gn.eps) && gn.eps > 0.0f &&
                    validGatedNormActivation(gn.gate_activation),
                "Retained CUDA GDN requires exact ordinary M1 in-place GatedRMSNorm");
            require(o.m == 1 && o.n == kHidden && o.k == columns_[1] &&
                    o.A == gn.output && o.B && o.C && o.c_buffer_id &&
                    !o.bias && !o.bias_tensor && o.alpha == 1.0f && o.beta == 0.0f &&
                    o.prepared_store && o.prepared_ref && o.prepared_ref->device == device(),
                "Retained CUDA GDN requires exact M1 Q8_0 ssm_out");

            full_outputs_[0] = {dynamic_cast<TensorBase *>(c.output), *c.output_buffer_id, {}};
            full_outputs_[1] = {dynamic_cast<TensorBase *>(r.output), *r.output_buffer_id, {}};
            full_outputs_[2] = {dynamic_cast<TensorBase *>(o.C), *o.c_buffer_id, {}};
            for (const auto &fo : full_outputs_)
                require(fo.tensor && fo.tensor->native_type() == TensorType::FP32,
                    "Retained CUDA GDN downstream outputs must be FP32 tensors");
            full_weights_.push_back(const_cast<ITensor *>(c.weight));
            if (c.bias)
                full_weights_.push_back(const_cast<ITensor *>(c.bias));
            full_weights_.push_back(const_cast<ITensor *>(r.A_log));
            full_weights_.push_back(const_cast<ITensor *>(r.dt_bias));
            full_weights_.push_back(const_cast<ITensor *>(gn.gamma));
        }

        ComputeStageType type() const override { return ComputeStageType::GDN_PROJECTION; }
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
            for (std::size_t i = 0; i < outputs_.size(); ++i)
            {
                result.addOutput(outputs_[i].id);
                result.addPreparedWeight(const_cast<ITensor *>(inputs_[i]), params().prepared_store, *refs_[i]);
            }
            for (const auto &fo : full_outputs_)
                result.addOutput(fo.id);
            for (auto *weight : full_weights_)
                result.addWeight(weight);
            result.addPreparedWeight(const_cast<ITensor *>(full_gdn_.out_proj.B),
                full_gdn_.out_proj.prepared_store, *full_gdn_.out_proj.prepared_ref);
            return result;
        }

        StageBufferRequirements getBufferRequirements() const override
        {
            auto result = hc_->phaseDeclaration().getBufferRequirements();
            for (std::size_t i = 0; i < outputs_.size(); ++i)
                result.addOutput(kProjNames[i], outputs_[i].tensor->shape(), BufferTensorType::FP32);
            result.addOutput("gdn_recurrence_in", full_outputs_[0].tensor->shape(), BufferTensorType::FP32);
            result.addOutput("attn_output", full_outputs_[1].tensor->shape(), BufferTensorType::FP32);
            result.addOutput("attn_proj", full_outputs_[2].tensor->shape(), BufferTensorType::FP32);
            return result;
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
            require(producer != order.end(), "Retained CUDA GDN output seal has no original producer node");
            for (const auto &name : selected)
                require(name != "activated_down" && name != "read_logits",
                    "Retained CUDA GDN cannot publish optional HC activated or up diagnostics");

            const std::array views{
                FullNativeHCFFNOutputBinding{FullNativeHCFFNOutputRole::HCActivated,
                    *producer, (*optional_)[0].tensor, (*optional_)[0].buffer, 0, (*optional_)[0].tensor->size_bytes()},
                FullNativeHCFFNOutputBinding{FullNativeHCFFNOutputRole::HCUp,
                    *producer, (*optional_)[1].tensor, (*optional_)[1].buffer, 0, (*optional_)[1].tensor->size_bytes()}};
            require(resolveFullNativeHCFFNOutputDemand(graph, arena, *producer, views).bits() == 0,
                "Retained CUDA GDN cannot omit an HC intermediate with an active graph reader");
            const auto identity = std::pair{graph.topologyGeneration(), epoch};
            require(!seal_ || *seal_ == identity, "Retained CUDA GDN observation identity changed");
            seal_ = identity;
        }

        WorkspaceRequirements getWorkspaceRequirements(int, int, int) const override
        {
            auto result = hc_->phaseWorkspaceRequirements();
            result.merge(retainedCUDAGDNWorkspace());
            return result;
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            require(!admission_ || workspace_ == workspace, "Retained CUDA GDN workspace changed while captured");
            hc_->bindPhaseWorkspace(workspace);
            workspace_ = workspace;
        }

        void unbindWorkspace() override
        {
            require(!admission_, "Retained CUDA GDN executable must retire before workspace release");
            bindWorkspace(nullptr);
        }

        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            require(seal_ && ctx && ctx->deviceId() == device() && workspace_ &&
                    workspace_->device() == device() && reinterpret_cast<std::uintptr_t>(stream) > 2 &&
                    !isGraphCaptureActive(),
                "Retained CUDA GDN requires a sealed original cold participant, workspace, and stream");
            if (admission_)
            {
                validateStorage(stream);
                return true;
            }
            setGPUStream(stream);
            hc_->preparePhaseBindings(ctx, stream);
            const auto &hc_prefix = hc_->cudaPrefixArguments();
            require(hc_prefix.mixed == mixed_->gpu_data_ptr(),
                "Retained CUDA GDN HC prefix mixed pointer mismatch");
            require(hc_prefix.up.rows.capacity() == 1 && hc_prefix.up.rows.physicalRows() == 1 &&
                    hc_prefix.up.rows.countOwner() == nullptr,
                "Retained CUDA GDN requires fully-active M1 row authority");

            engines_ = resolveEngines();
            for (std::size_t i = 0; i < outputs_.size(); ++i)
            {
                outputs_[i].lease = TransferEngine::instance().retainCapturedDeviceStorage(
                    *outputs_[i].tensor, device(), static_cast<std::size_t>(columns_[i]) * sizeof(float));
            }
            for (std::size_t i = 2; i < engines_.size(); ++i)
            {
                ContiguousFloatingPointWeightDescriptor view;
                require(engines_[i]->floatingActivationOperandPrecision() == FloatingActivationOperandPrecision::FP16RNE &&
                        engines_[i]->exportContiguousFloatingPointWeights(view) && view.valid() &&
                        view.type == TensorType::FP16 && view.n == columns_[i] && view.k == kHidden,
                    "Retained CUDA GDN requires the original FP16RNE alpha/beta subgroup");
                half_weights_[i - 2] = view;
            }

            const auto &c = full_gdn_.conv;
            const auto &r = full_gdn_.rec;
            const auto &o = full_gdn_.out_proj;
            const std::array<int, 3> full_cols{columns_[0], columns_[1], o.n};
            for (std::size_t i = 0; i < full_outputs_.size(); ++i)
            {
                full_outputs_[i].lease = TransferEngine::instance().retainCapturedDeviceStorage(
                    *full_outputs_[i].tensor, device(), static_cast<std::size_t>(full_cols[i]) * sizeof(float));
            }
            const auto ssm_source = o.prepared_store->binding(*o.prepared_ref);
            ssm_out_engine_ = o.prepared_store->gemmKernel(*o.prepared_ref);
            require(ssm_source && ssm_source->immutable && ssm_source->tensor == o.B && ssm_out_engine_,
                "Retained CUDA GDN lost its immutable ssm_out prepared source or engine");

            const auto *conv_kernel = dynamic_cast<const CUDAShortConvolution *>(c.kernel);
            const auto *rec_kernel = dynamic_cast<const CUDAGatedDeltaNet *>(r.kernel);
            const int conv_state_size = columns_[0] * (c.kernel_size - 1);
            const int rec_state_size = r.n_heads * r.d_k * r.d_v;
            conv_state_ = conv_kernel ? conv_kernel->borrowedLiveDeviceState(conv_state_size) : nullptr;
            rec_state_ = rec_kernel ? rec_kernel->borrowedLiveDeviceState(rec_state_size) : nullptr;
            require(conv_state_ && rec_state_,
                "Retained CUDA GDN requires already-bound CUDA conv and recurrence device state");

            const auto &hardware = cudaHardwareMetadata().at(device().ordinal);
            auto bindNativeProjection = [&](ITensorGemm *eng, float *out, int expected_n, int expected_k)
                -> native::CUDARetainedGDNProjection
            {
                native::CUDARetainedGDNProjection proj{};
                require(eng->inspectPreparedNativeQuantisedMatrixDesc(proj.weights) &&
                        proj.weights.valid() && proj.weights.n == expected_n && proj.weights.k == expected_k &&
                        proj.weights.source_identity_present,
                    "Retained CUDA GDN requires prepared NativeQuantised matrices");
                int ordered = 0, partitions = 0;
                require(cudaNativeQuantisedGemvTuned_queryCanonicalM1Schedule(
                            proj.weights.source_codebook_id, expected_n, expected_k,
                            hardware.sm_count, &ordered, &partitions) &&
                        partitions > 0 && partitions <= expected_k / 32 && (ordered || partitions == 1),
                    "Retained CUDA GDN has no canonical M1 schedule for projection");
                proj.output = out;
                proj.k_partitions = partitions;
                proj.blocks_per_partition = (expected_k / 32 + partitions - 1) / partitions;
                proj.ordered_kpar = ordered != 0;
                return proj;
            };

            auto weightPtr = [](const ITensor *t) -> const float * {
                const auto *tb = dynamic_cast<const TensorBase *>(t);
                return tb ? static_cast<const float *>(tb->gpu_data_ptr()) : nullptr;
            };

            scratch_ = workspace_->retainBuffer(kCudaGDNScratchBank, 0, native::retainedCUDAGDNScratchBytes());
            native::CUDARetainedGDNInvocation inv{.hc = hc_prefix};
            inv.qkv = bindNativeProjection(engines_[0], static_cast<float *>(outputs_[0].tensor->gpu_data_ptr()),
                columns_[0], kHidden);
            inv.z = bindNativeProjection(engines_[1], static_cast<float *>(outputs_[1].tensor->gpu_data_ptr()),
                columns_[1], kHidden);
            inv.half = {
                static_cast<const std::uint16_t *>(half_weights_[0].data),
                static_cast<const std::uint16_t *>(half_weights_[1].data),
                static_cast<float *>(outputs_[2].tensor->gpu_data_ptr()),
                static_cast<float *>(outputs_[3].tensor->gpu_data_ptr()),
                columns_[2]};
            inv.conv_weight = weightPtr(c.weight);
            inv.conv_bias = c.bias ? weightPtr(c.bias) : nullptr;
            inv.conv_state = conv_state_;
            inv.gdn_recurrence_in = static_cast<float *>(full_outputs_[0].tensor->gpu_data_ptr());
            inv.A_log = weightPtr(r.A_log);
            inv.dt_bias = weightPtr(r.dt_bias);
            inv.recurrence_state = rec_state_;
            inv.attn_output = static_cast<float *>(full_outputs_[1].tensor->gpu_data_ptr());
            const auto &gn = full_gdn_.norm;
            inv.ssm_norm_gamma = weightPtr(gn.gamma);
            inv.ssm_norm_eps = gn.eps;
            inv.ssm_norm_subtract_one = gn.subtract_one;
            inv.ssm_norm_activation = gn.gate_activation;
            inv.n_k_heads = r.n_k_heads;
            inv.n_v_heads = r.n_heads;
            inv.global_v_offset = r.global_v_head_offset;
            inv.use_qk_l2norm = r.use_qk_l2norm;
            inv.state_layout = r.kernel->recurrenceStateLayout();
            inv.ssm_out = bindNativeProjection(ssm_out_engine_,
                static_cast<float *>(full_outputs_[2].tensor->gpu_data_ptr()), o.n, columns_[1]);
            inv.scratch = scratch_->data();
            inv.scratch_bytes = native::retainedCUDAGDNScratchBytes();

            workspace_id_ = workspace_->id();
            same_prepared_store_ = (params().prepared_store == full_gdn_.out_proj.prepared_store);
            validated_store_epoch_ = 0;
            admission_ = native::prepareCUDARetainedGDN(device(), inv, workspace_id_);
            stream_ = stream;
            validateStorage(stream);
            if (same_prepared_store_)
                validated_store_epoch_ = params().prepared_store->mutationEpoch();
            return true;
        }

        bool execute(IDeviceContext *ctx) override
        {
            require(ctx && ctx->deviceId() == device(), "Retained CUDA GDN execution changed participant");
            const auto execution = gpuExecution();
            validateStorage(execution.nativeStream());
            for (const auto &input : phase_inputs_)
                execution.requirePreparedInput(input.tensor);
            for (auto *weight : phase_weights_)
                execution.requirePreparedInput(weight);
            for (const auto &output : phase_outputs_)
                execution.requirePreparedOutput(output.tensor);
            for (const auto &output : outputs_)
                execution.requirePreparedOutput(output.tensor);
            for (auto *weight : full_weights_)
                execution.requirePreparedInput(weight);
            for (const auto &output : full_outputs_)
                execution.requirePreparedOutput(output.tensor);

            native::launchCUDARetainedGDN(*admission_, execution.nativeStream(), workspace_id_);
            if (full_gdn_.rec.sequence_state_cache &&
                !full_gdn_.rec.sequence_state_cache->advanceRecurrentSequenceState(
                    full_gdn_.rec.layer_idx, 1, nullptr, 1, execution.nativeStream()))
            {
                throw std::runtime_error("Retained CUDA GDN recurrent sequence frontier publication failed");
            }
            for (const auto &output : phase_outputs_)
                execution.publish(output.tensor);
            for (const auto &output : outputs_)
                execution.publish(output.tensor);
            for (const auto &output : full_outputs_)
                execution.publish(output.tensor);

            PerfStatsCollector::addCounter("kernel", "cuda_retained_gdn_full_megakernel_calls",
                1.0, "graph_capture", device().toString(),
                {{"m", "1"}, {"h", "2560"}, {"n_v_heads", std::to_string(full_gdn_.rec.n_heads)}});
            return true;
        }

        std::uint64_t graphCaptureVariantSignature() const override
        {
            // Complete declaration identity, stable across cold physical materialization
            // (identical before and after sealGraphOutputDemand / prepareGraphLaunch).
            if (admission_)
                validateStorage(stream_, /*validate_hc=*/false);
            auto identity = projection_.graphCaptureVariantSignature();
            const auto mix = [&identity](std::uint64_t item) { identity = (identity ^ item) * 1099511628211ull; };
            mix(hc_->phaseDeclaration().graphCaptureVariantSignature());
            mix(0x4355444147444e31ull);
            for (const auto &output : outputs_)
                mix(reinterpret_cast<std::uintptr_t>(output.tensor));
            for (const auto &fo : full_outputs_)
                mix(reinterpret_cast<std::uintptr_t>(fo.tensor));
            for (auto *weight : full_weights_)
                mix(reinterpret_cast<std::uintptr_t>(weight));
            const auto &c = full_gdn_.conv;
            const auto &r = full_gdn_.rec;
            const auto &gn = full_gdn_.norm;
            const auto &o = full_gdn_.out_proj;
            mix(reinterpret_cast<std::uintptr_t>(c.kernel));
            mix(reinterpret_cast<std::uintptr_t>(r.kernel));
            mix(static_cast<std::uint64_t>(r.kernel->recurrenceStateLayout()));
            mix(static_cast<std::uint64_t>(r.n_k_heads));
            mix(static_cast<std::uint64_t>(r.n_heads));
            mix(static_cast<std::uint64_t>(r.global_v_head_offset));
            mix(r.use_qk_l2norm ? 1ull : 0ull);
            mix(static_cast<std::uint64_t>(gn.gate_activation));
            mix(gn.subtract_one ? 1ull : 0ull);
            mix(static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(gn.eps)));
            mix(reinterpret_cast<std::uintptr_t>(o.B));
            mix(reinterpret_cast<std::uintptr_t>(o.prepared_store));
            mix(o.prepared_ref ? o.prepared_ref->binding_id : 0ull);
            return identity ? identity : 1ull;
        }

        void invalidateKernelDynamicState() override
        {
            admission_.reset();
            scratch_.reset();
            for (auto &output : outputs_)
                output.lease.reset();
            for (auto &output : full_outputs_)
                output.lease.reset();
            ssm_out_engine_ = nullptr;
            conv_state_ = nullptr;
            rec_state_ = nullptr;
            stream_ = nullptr;
            validated_store_epoch_ = 0;
            seal_.reset();
            hc_->invalidatePhaseBindings();
        }

        StageDumpInfo buildDumpInfoImpl() const override
        {
            auto info = projection_.getDumpInfoSnapshot();
            info.inputs.clear();
            const bool ownsWrite = hc_->trunkSource() == Qwen4ExpHCTrunkSource::PreviousWrite;
            for (std::size_t i = 0; i < phase_inputs_.size(); ++i)
            {
                constexpr std::array labels{"previous_residual", "previous_block", "previous_gate"};
                const auto &input = phase_inputs_[i];
                info.addInput(ownsWrite ? labels[i] : "published_trunk", input.tensor, 1, input.tensor->cols());
            }
            for (std::size_t i = 0; i < phase_outputs_.size(); ++i)
            {
                constexpr std::array labels{"hc_post", "normalized", "raw_down", "mixed", "hc_injection"};
                const auto &output = phase_outputs_[i];
                info.addOutput(labels[i + (ownsWrite ? 0 : 1)], output.tensor, 1, output.tensor->cols());
            }
            constexpr std::array full_labels{"gdn_recurrence_in", "attn_output", "attn_proj"};
            for (std::size_t i = 0; i < full_outputs_.size(); ++i)
                info.addOutput(full_labels[i], full_outputs_[i].tensor, 1, full_outputs_[i].tensor->cols());
            return info;
        }

    private:
        struct Output
        {
            TensorBase *tensor = nullptr;
            BufferId id{};
            std::shared_ptr<const CapturedDeviceStorageLease> lease;
        };

        const GDNProjectionStage::Params &params() const { return projection_.getParams(); }

        std::array<ITensorGemm *, 4> resolveEngines() const
        {
            std::array<ITensorGemm *, 4> engines{};
            for (std::size_t i = 0; i < engines.size(); ++i)
            {
                require(refs_[i] && refs_[i]->device == device(),
                    "Retained CUDA GDN lost its exact prepared projection ref");
                const auto source = params().prepared_store->binding(*refs_[i]);
                engines[i] = params().prepared_store->gemmKernel(*refs_[i]);
                require(source && source->immutable && source->tensor == inputs_[i] &&
                        engines[i] && (!cached_[i] || cached_[i] == engines[i]),
                    "Retained CUDA GDN frozen prepared source or engine changed");
            }
            return engines;
        }

        void validateStorage(void *stream, bool validate_hc = true) const
        {
            require(admission_ && stream == stream_ && workspace_ &&
                    workspace_->id() == workspace_id_ &&
                    scratch_ && scratch_->data() != nullptr,
                "Retained CUDA GDN captured stream or workspace generation changed");
            if (validate_hc)
                hc_->validatePhaseBindings(stream);
            const bool store_unchanged = same_prepared_store_ && validated_store_epoch_ != 0 &&
                params().prepared_store->mutationEpoch() == validated_store_epoch_;
            if (!store_unchanged)
                require(resolveEngines() == engines_, "Retained CUDA GDN prepared engine changed");
            for (std::size_t i = 0; i < outputs_.size(); ++i)
                TransferEngine::instance().requireCapturedDeviceStorage(*outputs_[i].lease,
                    *outputs_[i].tensor, device(), static_cast<std::size_t>(columns_[i]) * sizeof(float));
            const auto &c = full_gdn_.conv;
            const auto &r = full_gdn_.rec;
            const auto &o = full_gdn_.out_proj;
            const std::array<int, 3> full_cols{columns_[0], columns_[1], o.n};
            for (std::size_t i = 0; i < full_outputs_.size(); ++i)
                TransferEngine::instance().requireCapturedDeviceStorage(*full_outputs_[i].lease,
                    *full_outputs_[i].tensor, device(), static_cast<std::size_t>(full_cols[i]) * sizeof(float));
            if (!store_unchanged)
            {
                require(o.prepared_store->gemmKernel(*o.prepared_ref) == ssm_out_engine_,
                    "Retained CUDA GDN ssm_out engine changed");
                if (same_prepared_store_)
                    validated_store_epoch_ = params().prepared_store->mutationEpoch();
            }
            const auto *conv_kernel = dynamic_cast<const CUDAShortConvolution *>(c.kernel);
            const auto *rec_kernel = dynamic_cast<const CUDAGatedDeltaNet *>(r.kernel);
            require(conv_kernel &&
                    conv_kernel->borrowedLiveDeviceState(columns_[0] * (c.kernel_size - 1)) == conv_state_ &&
                    rec_kernel &&
                    rec_kernel->borrowedLiveDeviceState(r.n_heads * r.d_k * r.d_v) == rec_state_,
                "Retained CUDA GDN live device state owner changed");
        }

        GDNProjectionStage projection_;
        std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc_;
        const std::array<int, 4> columns_;
        const std::array<const ITensor *, 4> inputs_;
        const std::array<std::optional<PreparedWeightRef>, 4> refs_;
        const std::array<ITensorGemm *, 4> cached_;
        FullCUDAGDNDeclaration full_gdn_;
        std::array<Output, 4> outputs_{};
        std::array<Output, 3> full_outputs_{};
        std::optional<std::array<Qwen4ExpFloatBinding, 2>> optional_;
        std::vector<Qwen4ExpFloatBinding> phase_inputs_;
        std::vector<Qwen4ExpFloatBinding> phase_outputs_;
        std::vector<ITensor *> phase_weights_;
        std::vector<ITensor *> full_weights_;
        std::optional<std::pair<std::uint64_t, std::uint64_t>> seal_;
        std::array<ITensorGemm *, 4> engines_{};
        ITensorGemm *ssm_out_engine_ = nullptr;
        float *conv_state_ = nullptr;
        float *rec_state_ = nullptr;
        std::array<ContiguousFloatingPointWeightDescriptor, 2> half_weights_{};
        DeviceWorkspaceManager *workspace_ = nullptr;
        std::uint64_t workspace_id_ = 0;
        bool same_prepared_store_ = false;
        mutable std::uint64_t validated_store_epoch_ = 0;
        void *stream_ = nullptr;
        std::shared_ptr<const WorkspaceBufferLease> scratch_;
        std::optional<native::CUDARetainedGDNAdmission> admission_;
        TensorBase *mixed_ = nullptr;
    };
} // namespace

WorkspaceRequirements retainedCUDAGDNWorkspace()
{
    WorkspaceRequirements requirements;
    requirements.buffers.emplace_back(kCudaGDNScratchBank, native::retainedCUDAGDNScratchBytes(), 16, true,
        WorkspaceExecutionRegime::Any, WorkspaceContentLifetime::SerialGraphFamily);
    return requirements;
}

void lowerRetainedCUDAGDN(ComputeGraph &graph, const std::string &gdnProjectionName,
                          const std::string &previousInputBoundary)
{
    using Primitive = Qwen4ExpPrimitiveStage;
    const auto *target = graph.getNode(gdnProjectionName);
    const auto *projection = target ? dynamic_cast<const GDNProjectionStage *>(target->stage.get()) : nullptr;
    require(projection, "Retained CUDA GDN lowerer requires the original GDN projection");
    const auto &params = projection->getParams();
    require(params.device_id.is_cuda() && params.input && params.input_buffer_id,
        "Retained CUDA GDN lost its CUDA participant or mixed binding");

    // Verify closed downstream chain:
    //   gdn_proj -> short_conv -> gdn_recurrence -> gated_norm -> gdn_out_proj
    // (or 4-node variant if gated_norm is already attached to gdn_out_proj).
    const auto proj_consumers = graph.directConsumersOf(gdnProjectionName);
    if (proj_consumers.size() != 1)
        return;
    const auto *conv_node = graph.getNode(proj_consumers[0]);
    const auto *conv_stage = conv_node ? dynamic_cast<const ShortConv1dStage *>(conv_node->stage.get()) : nullptr;
    if (!conv_stage)
        return;
    const auto conv_consumers = graph.directConsumersOf(proj_consumers[0]);
    const auto *rec_node = conv_consumers.size() == 1 ? graph.getNode(conv_consumers[0]) : nullptr;
    const auto *rec_stage = rec_node ? dynamic_cast<const GDNRecurrenceStage *>(rec_node->stage.get()) : nullptr;
    if (!rec_stage)
        return;

    const auto rec_consumers = graph.directConsumersOf(conv_consumers[0]);
    if (rec_consumers.size() != 1)
        return;
    std::string gated_norm_name;
    std::string out_proj_name;
    const ComputeNode *out_node = nullptr;
    const GEMMStage *out_stage = nullptr;
    GatedRMSNormStage::Params gn{};

    const auto *rec_next_node = graph.getNode(rec_consumers[0]);
    if (const auto *gn_stage = rec_next_node
            ? dynamic_cast<const GatedRMSNormStage *>(rec_next_node->stage.get()) : nullptr)
    {
        gated_norm_name = rec_consumers[0];
        gn = gn_stage->getParams();
        const auto gn_consumers = graph.directConsumersOf(gated_norm_name);
        if (gn_consumers.size() != 1)
            return;
        out_proj_name = gn_consumers[0];
        out_node = graph.getNode(out_proj_name);
        out_stage = out_node ? dynamic_cast<const GEMMStage *>(out_node->stage.get()) : nullptr;
    }
    else if (const auto *direct_out = rec_next_node
            ? dynamic_cast<const GEMMStage *>(rec_next_node->stage.get()) : nullptr)
    {
        if (!direct_out->getParams().gated_norm.has_value())
            return;
        out_proj_name = rec_consumers[0];
        out_node = rec_next_node;
        out_stage = direct_out;
        const auto &embedded = *direct_out->getParams().gated_norm;
        gn.device_id = params.device_id;
        gn.input = rec_stage->getParams().output;
        gn.output = rec_stage->getParams().output;
        gn.gate = const_cast<ITensor *>(embedded.gate);
        gn.gamma = embedded.gamma;
        gn.eps = embedded.parameters.eps;
        gn.subtract_one = embedded.parameters.subtract_one;
        gn.seq_len = 1;
        gn.feature_dim = params.n_z;
        gn.norm_dim = embedded.parameters.norm_dim;
        gn.gate_activation = embedded.parameters.activation;
    }
    if (!out_node || !out_stage)
        return;

    const auto &cp = conv_stage->getParams();
    const auto &rp = rec_stage->getParams();
    const auto &op = out_stage->getParams();
    const bool raw_qk = std::holds_alternative<GDNRawQKInput>(cp.qk_contract) &&
                        std::holds_alternative<GDNRawQKInput>(rp.qk_contract);
    const bool prep_qk = std::holds_alternative<GDNPreparedQKBinding>(cp.qk_contract) &&
                         std::holds_alternative<GDNPreparedQKBinding>(rp.qk_contract);
    const bool out_deps_closed = std::all_of(
        out_node->dependencies.begin(), out_node->dependencies.end(),
        [&](const std::string &dep) {
            return dep == gdnProjectionName || dep == proj_consumers[0] ||
                   dep == conv_consumers[0] || (!gated_norm_name.empty() && dep == gated_norm_name);
        });
    if (!out_deps_closed ||
        cp.device_id != params.device_id || rp.device_id != params.device_id ||
        gn.device_id != params.device_id || op.device_id != params.device_id ||
        cp.seq_len != 1 || cp.request_count != 1 || cp.kernel_size != 4 ||
        cp.channels != params.n_qkv || (params.n_qkv & 7) != 0 ||
        cp.input != params.output_qkv || !cp.output || !cp.output_buffer_id || !cp.weight ||
        cp.request_seq_lens_device || cp.verifier_state_capture_rows != 0 ||
        cp.speculative_state_slot_rows != 0 || (!raw_qk && !prep_qk) ||
        dynamic_cast<const CUDAShortConvolution *>(cp.kernel) == nullptr ||
        rp.seq_len != 1 || rp.request_count != 1 || rp.d_k != 128 || rp.d_v != 128 ||
        rp.n_heads != params.n_a || rp.n_heads < 1 ||
        static_cast<unsigned>(rp.n_heads) > native::kMaxCUDARetainedGDNVHeads ||
        rp.n_heads * 128 != params.n_z || (params.n_z & 7) != 0 ||
        rp.n_k_heads <= 0 || 2 * rp.n_k_heads * 128 + rp.n_heads * 128 != params.n_qkv ||
        rp.Q != cp.output || rp.K != cp.output || rp.V != cp.output ||
        rp.alpha != params.output_a || rp.beta != params.output_b ||
        !rp.A_log || !rp.dt_bias || !rp.output || !rp.output_buffer_id ||
        rp.request_seq_lens_device || rp.gpu_retained_workspace ||
        rp.verifier_state_capture_rows != 0 || rp.speculative_state_slot_rows != 0 ||
        dynamic_cast<const CUDAGatedDeltaNet *>(rp.kernel) == nullptr ||
        gn.seq_len != 1 || gn.feature_dim != params.n_z || gn.norm_dim != 128 ||
        gn.input != rp.output || gn.output != rp.output || gn.gate != params.output_z ||
        !gn.gamma || gn.gamma->numel() != 128 ||
        !std::isfinite(gn.eps) || gn.eps <= 0.0f ||
        !validGatedNormActivation(gn.gate_activation) ||
        op.m != 1 || op.n != kHidden || op.k != params.n_z ||
        op.A != gn.output || !op.B ||
        op.B->native_type() == TensorType::FP32 || op.B->native_type() == TensorType::FP16 ||
        op.B->native_type() == TensorType::BF16 ||
        !op.C || !op.c_buffer_id ||
        op.bias || op.bias_tensor || op.alpha != 1.0f || op.beta != 0.0f ||
        !op.prepared_store || !op.prepared_ref || op.prepared_ref->device != params.device_id)
    {
        return;
    }

    std::string readName;
    for (const auto &dependency : target->dependencies)
    {
        const auto *node = graph.getNode(dependency);
        const auto *primitive = node ? dynamic_cast<const Primitive *>(node->stage.get()) : nullptr;
        const auto *read = primitive ? std::get_if<Primitive::ResidualRead>(&primitive->getParams().operation) : nullptr;
        if (read && read->output.tensor == params.input && read->output.buffer == *params.input_buffer_id &&
            read->output.byteOffset == 0)
        {
            require(readName.empty(), "Retained CUDA GDN has ambiguous original mixed producers");
            readName = dependency;
        }
    }
    require(!readName.empty(), "Retained CUDA GDN requires its original direct HC residual-read producer");

    const auto *boundary = previousInputBoundary.empty() ? nullptr : graph.getNode(previousInputBoundary);
    const auto *boundary_prim = boundary ? dynamic_cast<const Primitive *>(boundary->stage.get()) : nullptr;
    const auto *boundary_write = boundary_prim
        ? std::get_if<Primitive::ResidualWrite>(&boundary_prim->getParams().operation) : nullptr;
    const auto injection_policy = boundary_write
        ? Qwen4ExpHCInjectionOwnership::DiscoverInRegion
        : Qwen4ExpHCInjectionOwnership::KeepOriginalStage;
    auto region = Qwen4ExpHyperconnectionBuilder::describeFullNativeReadRegion(
        graph, previousInputBoundary, readName, injection_policy);
    const auto *normNode = graph.getNode(region.branch_norm);
    const auto &norm = std::get<Primitive::BranchRMSNorm>(
        static_cast<const Primitive &>(*normNode->stage).getParams().operation);
    if (!previousInputBoundary.empty())
        require(boundary && std::find(normNode->dependencies.begin(), normNode->dependencies.end(),
            previousInputBoundary) != normNode->dependencies.end(),
            "Retained CUDA GDN lost its original merged input frontier");

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
    auto replacement = std::make_unique<RetainedCUDAGDNStage>(
        params, std::move(phase), FullCUDAGDNDeclaration{cp, rp, gn, op});

    GraphComposer composer(graph);
    std::vector<GraphNodeRef> members;
    if (!region.previous_write.empty())
        members.push_back(composer.ref(region.previous_write));
    members.push_back(composer.ref(region.branch_norm));
    members.push_back(composer.ref(region.down));
    members.push_back(composer.ref(region.up));
    members.push_back(composer.ref(region.read));
    members.push_back(composer.ref(gdnProjectionName));
    if (!region.injection.empty())
        members.push_back(composer.ref(region.injection));
    members.push_back(composer.ref(proj_consumers[0]));
    members.push_back(composer.ref(conv_consumers[0]));
    if (!gated_norm_name.empty())
        members.push_back(composer.ref(gated_norm_name));
    GraphNodeRef replacement_terminal = composer.ref(out_proj_name);
    members.push_back(replacement_terminal);

    (void)lowerLocalMegakernelRegion(graph, LocalMegakernelRegionSpec{
        .members = std::move(members),
        .terminal = replacement_terminal,
        .replacement = std::move(replacement),
    });
}
} // namespace llaminar2
#endif
