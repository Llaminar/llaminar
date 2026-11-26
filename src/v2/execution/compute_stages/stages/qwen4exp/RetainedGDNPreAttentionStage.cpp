/**
 * @file RetainedGDNPreAttentionStage.cpp
 * @brief Preserve original POST/HC and GDN publications across one resident ROCm prefix.
 *
 * The resident prefix consumes the already prepared Q8_0 and FP16 matrix
 * views and produces the original raw GDN projections on one stream.
 * Original HC leases, prepared tensors, and the executor workspace own every byte.
 */
#include "RetainedGDNPreAttentionStage.h"

#ifdef HAVE_ROCM
#include "execution/compute_stages/IGraphOutputDemand.h"
#include "execution/compute_stages/stages/FullNativeHCFFNOutputDemandResolver.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/LocalMegakernelLowering.h"
#include "kernels/IKVCache.h"
#include "kernels/rocm/gdn/ROCmGatedDeltaNet.h"
#include "kernels/rocm/gdn/ROCmShortConvolution.h"
#include "kernels/rocm/gemm/RetainedPreAttentionDevice.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        namespace native = rocm::pre_attention_probe;
        constexpr auto scratchBank = "qfn_retained_pre_attention";
        constexpr int hidden = 2560;
        constexpr std::array<const char *, 4> names{"qkv", "z", "alpha", "beta"};

        /** @brief Authenticate a complete original local owner before native recording. */
        void require(bool condition, const char *message)
        {
            if (!condition) throw std::invalid_argument(message);
        }

        /** @brief Immutable downstream GDN stages folded into one resident megakernel. */
        struct FullGDNDeclaration
        {
            ShortConv1dStage::Params conv;
            GDNRecurrenceStage::Params rec;
            GEMMStage::Params out_proj;
        };

        class RetainedGDNPreAttentionStage final : public IComputeStage, public IWorkspaceConsumer, public IGraphOutputDemand
        {
        public:
            /** @brief Seal the original source and output declarations before any graph mutation. */
            RetainedGDNPreAttentionStage(const GDNProjectionStage::Params &params,
                                        std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc, native::HalfGDNPolicy halfPolicy,
                                        std::optional<FullGDNDeclaration> fullGdn = std::nullopt)
                : IComputeStage(params.device_id), projection_(params), hc_(std::move(hc)),
                  columns_{params.n_qkv, params.n_z, params.n_a, params.n_b},
                  inputs_{params.w_qkv, params.w_z, params.w_a, params.w_b},
                  refs_{params.prepared_ref_qkv, params.prepared_ref_z,
                        params.prepared_ref_a, params.prepared_ref_b},
                  cached_{params.gemm_qkv, params.gemm_z, params.gemm_a, params.gemm_b},
                  half_policy_(halfPolicy), full_gdn_(std::move(fullGdn))
            {
                require(hc_ && device().is_rocm() && hc_->phaseDeclaration().device() == device() &&
                    params.m == 1 && params.k == hidden &&
                    columns_[0] > 0 && (columns_[0] % 4) == 0 &&
                    columns_[1] > 0 && (columns_[1] % 4) == 0 &&
                    columns_[2] > 0 && columns_[2] == columns_[3] &&
                    !params.force_decode_equivalent_verifier_prefill && !params.verifier_row_range &&
                    params.qkv_operand_precision == QuantizedActivationOperandPrecision::SingleQ8 &&
                    params.prepared_store && params.input && params.input_buffer_id,
                    "Retained GDN pre-attention requires valid ordinary ROCm M1 QKV/Z/A/B declarations");

                for (const auto &output : hc_->phaseOutputs())
                    if (output.tensor == params.input && output.buffer == *params.input_buffer_id &&
                        output.byteOffset == 0)
                        mixed_ = output.tensor;
                require(mixed_, "Retained GDN must consume its exact original HC mixed publication");

                const std::array output_tensors{params.output_qkv, params.output_z, params.output_a, params.output_b};
                const std::array ids{params.output_qkv_buffer_id, params.output_z_buffer_id,
                                     params.output_a_buffer_id, params.output_b_buffer_id};
                for (std::size_t i = 0; i < outputs_.size(); ++i)
                {
                    require(inputs_[i] && refs_[i] && refs_[i]->device == device(),
                        "Retained GDN requires exact original participant projection references");
                    outputs_[i].tensor = dynamic_cast<TensorBase *>(output_tensors[i]);
                    const auto *tensor = outputs_[i].tensor;
                    require(ids[i] && tensor && tensor != mixed_ && tensor->native_type() == TensorType::FP32 &&
                        tensor->rows() >= 1 && tensor->cols() == std::size_t(columns_[i]),
                        "Retained GDN requires the original distinct FP32 projection output geometry");
                    require(std::none_of(outputs_.begin(), outputs_.begin() + i,
                        [tensor](const auto &prior) { return prior.tensor == tensor; }),
                        "Retained GDN projection outputs cannot alias");
                    outputs_[i].id = *ids[i];
                }

                const auto optional = hc_->optionalPhaseOutputs();
                for (std::size_t i = 0; i < optional.size(); ++i)
                    require(optional[i].tensor && optional[i].byteOffset == 0,
                        "Retained GDN HC optional outputs require original arena bindings");
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
                        "Retained GDN projections cannot share an original HC port");
                }
                const bool ownsWrite = hc_->trunkSource() == Qwen4ExpHCTrunkSource::PreviousWrite;
                const std::size_t expected_base_outputs = ownsWrite ? 4u : 3u;
                require(phase_inputs_.size() == (ownsWrite ? 3u : 1u) &&
                    (phase_outputs_.size() == expected_base_outputs ||
                     (ownsWrite && phase_outputs_.size() == expected_base_outputs + 1u)),
                    "Retained GDN requires original POST/HC outputs (with optional absorbed injection)");
                for (auto *weight : hc_->phaseDeclaration().bufferContract().weight_tensors)
                    phase_weights_.push_back(weight);
                if (full_gdn_)
                {
                    const auto &c = full_gdn_->conv;
                    const auto &r = full_gdn_->rec;
                    const auto &o = full_gdn_->out_proj;
                    const bool raw_qk = std::holds_alternative<GDNRawQKInput>(c.qk_contract) &&
                                        std::holds_alternative<GDNRawQKInput>(r.qk_contract);
                    const bool prepared_qk = std::holds_alternative<GDNPreparedQKBinding>(c.qk_contract) &&
                                             std::holds_alternative<GDNPreparedQKBinding>(r.qk_contract);
                    require(retainsHalf() &&
                        c.device_id == device() && r.device_id == device() && o.device_id == device() &&
                        c.seq_len == 1 && c.request_count == 1 && c.kernel_size == 4 &&
                        c.channels == columns_[0] && (columns_[0] & 7) == 0 &&
                        c.input == outputs_[0].tensor && c.output && c.output_buffer_id && c.weight &&
                        !c.request_seq_lens_device && c.verifier_state_capture_rows == 0 &&
                        c.speculative_state_slot_rows == 0 &&
                        (raw_qk || prepared_qk) &&
                        dynamic_cast<const ROCmShortConvolution *>(c.kernel) != nullptr,
                        "Retained Full GDN requires exact ordinary M1 ROCm ShortConv1d");
                    require(r.seq_len == 1 && r.request_count == 1 &&
                        r.d_k == 128 && r.d_v == 128 &&
                        r.n_heads == columns_[2] && r.n_heads >= 4 && r.n_heads <= 48 &&
                        r.n_heads * 128 == columns_[1] && (columns_[1] & 7) == 0 &&
                        r.n_k_heads > 0 && 2 * r.n_k_heads * 128 + r.n_heads * 128 == columns_[0] &&
                        r.Q == c.output && r.K == c.output && r.V == c.output &&
                        r.alpha == outputs_[2].tensor && r.beta == outputs_[3].tensor &&
                        r.A_log && r.dt_bias && r.output && r.output_buffer_id &&
                        !r.request_seq_lens_device && !r.gpu_retained_workspace &&
                        r.verifier_state_capture_rows == 0 && r.speculative_state_slot_rows == 0 &&
                        dynamic_cast<const ROCmGatedDeltaNet *>(r.kernel) != nullptr &&
                        r.kernel->recurrenceStateLayout() == GDNRecurrenceStateLayout::ValueMajorKeyContiguous,
                        "Retained Full GDN requires exact ordinary M1 ValueMajor ROCm GDNRecurrence");
                    require(o.m == 1 && o.n == hidden && o.k == columns_[1] &&
                        o.A == r.output && o.B && o.C && o.c_buffer_id &&
                        !o.bias && !o.bias_tensor && o.alpha == 1.0f && o.beta == 0.0f &&
                        o.gated_norm.has_value() && o.gated_norm->gate == outputs_[1].tensor &&
                        o.gated_norm->gamma && o.gated_norm->parameters.norm_dim == 128 &&
                        o.gated_norm->parameters.gamma_period == 128 &&
                        o.prepared_store && o.prepared_ref && o.prepared_ref->device == device(),
                        "Retained Full GDN requires exact M1 GatedRMSNorm + Q8_0 ssm_out");
                    full_outputs_[0] = {dynamic_cast<TensorBase *>(c.output), *c.output_buffer_id, {}};
                    full_outputs_[1] = {dynamic_cast<TensorBase *>(r.output), *r.output_buffer_id, {}};
                    full_outputs_[2] = {dynamic_cast<TensorBase *>(o.C), *o.c_buffer_id, {}};
                    for (const auto &fo : full_outputs_)
                        require(fo.tensor && fo.tensor->native_type() == TensorType::FP32,
                            "Retained Full GDN downstream outputs must be FP32 tensors");
                    full_weights_.push_back(const_cast<ITensor *>(c.weight));
                    if (c.bias) full_weights_.push_back(const_cast<ITensor *>(c.bias));
                    full_weights_.push_back(const_cast<ITensor *>(r.A_log));
                    full_weights_.push_back(const_cast<ITensor *>(r.dt_bias));
                    full_weights_.push_back(const_cast<ITensor *>(o.gated_norm->gamma));
                }
            }

            ComputeStageType type() const override { return ComputeStageType::GDN_PROJECTION; }
            bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::GPU_ROCM; }
            bool isGraphCapturable() const override { return admission_.has_value(); }
            bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
            bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
            GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
            { return GraphLaunchPreparationPolicy::CaptureOnly; }
            CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }

            /** @return Original output and weight lifetimes with optional HC observations excluded. */
            StageBufferContract bufferContract() const override
            {
                auto result = hc_->phaseDeclaration().bufferContract();
                for (std::size_t i = 0; i < outputs_.size(); ++i)
                {
                    result.addOutput(outputs_[i].id);
                    result.addPreparedWeight(const_cast<ITensor *>(inputs_[i]), params().prepared_store, *refs_[i]);
                }
                if (full_gdn_)
                {
                    for (const auto &fo : full_outputs_) result.addOutput(fo.id);
                    for (auto *weight : full_weights_) result.addWeight(weight);
                    result.addPreparedWeight(const_cast<ITensor *>(full_gdn_->out_proj.B),
                        full_gdn_->out_proj.prepared_store, *full_gdn_->out_proj.prepared_ref);
                }
                return result;
            }

            /** @return Existing HC and GDN geometries without a new activation allocation. */
            StageBufferRequirements getBufferRequirements() const override
            {
                auto result = hc_->phaseDeclaration().getBufferRequirements();
                for (std::size_t i = 0; i < outputs_.size(); ++i)
                    result.addOutput(names[i], outputs_[i].tensor->shape(), BufferTensorType::FP32);
                if (full_gdn_)
                {
                    result.addOutput("gdn_recurrence_in", full_outputs_[0].tensor->shape(), BufferTensorType::FP32);
                    result.addOutput("attn_output", full_outputs_[1].tensor->shape(), BufferTensorType::FP32);
                    result.addOutput("attn_proj", full_outputs_[2].tensor->shape(), BufferTensorType::FP32);
                }
                return result;
            }

            /** @return Advertise the omitted original observations so explicit selections fail at the seal. */
            std::optional<StageDumpInfo> coldSnapshotPublicationDeclaration() const override
            {
                auto info = buildDumpInfoImpl();
                info.addOutput("activated_down", (*optional_)[0].tensor, 1, (*optional_)[0].tensor->cols());
                info.addOutput("read_logits", (*optional_)[1].tensor, 1, (*optional_)[1].tensor->cols());
                return info;
            }

            /** @brief Reject live omitted intermediates and authenticate the observation generation. */
            void sealGraphOutputDemand(const ComputeGraph &graph, const BufferArena &arena,
                                       std::span<const std::string> selected, std::uint64_t epoch) override
            {
                const auto order = graph.getExecutionOrder();
                const auto producer = std::find_if(order.begin(), order.end(),
                    [&](const auto &name) { return graph.getNode(name)->stage.get() == this; });
                require(producer != order.end(), "Retained GDN output seal has no original producer node");
                for (const auto &name : selected)
                    require(name != "activated_down" && name != "read_logits",
                        "Retained GDN cannot publish optional HC activated or up diagnostics");

                const std::array views{
                    FullNativeHCFFNOutputBinding{FullNativeHCFFNOutputRole::HCActivated,
                        *producer, (*optional_)[0].tensor, (*optional_)[0].buffer, 0, (*optional_)[0].tensor->size_bytes()},
                    FullNativeHCFFNOutputBinding{FullNativeHCFFNOutputRole::HCUp,
                        *producer, (*optional_)[1].tensor, (*optional_)[1].buffer, 0, (*optional_)[1].tensor->size_bytes()}};
                require(resolveFullNativeHCFFNOutputDemand(graph, arena, *producer, views).bits() == 0,
                    "Retained GDN cannot omit an HC intermediate with an active graph reader");
                const auto identity = std::pair{graph.topologyGeneration(), epoch};
                require(!seal_ || *seal_ == identity, "Retained GDN observation identity changed");
                seal_ = identity;
            }

            /** @return Exact scratch quotes for the original HC and surviving FP16 subgroup. */
            WorkspaceRequirements getWorkspaceRequirements(int, int, int) const override
            {
                auto result = hc_->phaseWorkspaceRequirements();
                result.merge(retainedGDNPreAttentionWorkspace());
                if (retainsHalf()) return result;
                const auto engines = resolveEngines();
                for (std::size_t i = 2; i < engines.size(); ++i)
                {
                    auto *workspace = dynamic_cast<IWorkspaceConsumer *>(engines[i]);
                    require(workspace, "Retained GDN half engine lost its original workspace declaration");
                    result.merge(workspace->getWorkspaceRequirements(1, columns_[i], hidden));
                }
                return result;
            }

            /** @brief Bind the exact executor workspace before any captures use it. */
            void bindWorkspace(DeviceWorkspaceManager *workspace) override
            {
                require(!admission_ || workspace_ == workspace, "Retained GDN workspace changed while captured");
                hc_->bindPhaseWorkspace(workspace);
                workspace_ = workspace;
            }

            /** @brief Release workspace only after the original executable retires. */
            void unbindWorkspace() override
            {
                require(!admission_, "Retained GDN executable must retire before workspace release");
                bindWorkspace(nullptr);
            }

            bool hasWorkspace() const override { return workspace_ != nullptr; }
            DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

            /** @brief Retain all native bindings and initialize the shared serial instance before capture. */
            bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
            {
                require(seal_ && ctx && ctx->deviceId() == device() && workspace_ &&
                    workspace_->device() == device() && reinterpret_cast<std::uintptr_t>(stream) > 2 &&
                    !isGraphCaptureActive(), "Retained GDN requires a sealed original cold participant, workspace and stream");
                if (admission_)
                {
                    validateStorage(stream);
                    return true;
                }
                setGPUStream(stream);
                hc_->preparePhaseBindings(ctx, stream);
                auto prefix = hc_->fullPrefixArguments();
                require((bool(prefix.hc.injection.weights) == bool(prefix.hc.injection.output)) &&
                    prefix.hc.mixed == mixed_->gpu_data_ptr(),
                    "Retained GDN requires paired injection bindings and its own mixed output");
                rows_ = prefix.hc.up.rows;
                require(rows_.capacity() == 1 && rows_.physicalRows() == 1,
                    "Retained GDN requires the original whole M1 prefix");
                const auto &original = params();
                require(!original.qkv_row_range ||
                    original.qkv_row_range->captureIdentity() == rows_.captureIdentity(),
                    "Retained GDN QKV count owner differs from the HC prefix");

                engines_ = resolveEngines();
                for (std::size_t i = 0; i < outputs_.size(); ++i)
                {
                    auto &output = outputs_[i];
                    output.lease = TransferEngine::instance().retainCapturedDeviceStorage(
                        *output.tensor, device(), std::size_t(columns_[i]) * sizeof(float));
                }
                half_.clear();
                for (std::size_t i = 2; i < engines_.size(); ++i)
                {
                    ContiguousFloatingPointWeightDescriptor view;
                    require(engines_[i]->floatingActivationOperandPrecision() == FloatingActivationOperandPrecision::FP16RNE &&
                        engines_[i]->exportContiguousFloatingPointWeights(view) && view.valid() &&
                        view.type == TensorType::FP16 && view.n == columns_[i] && view.k == hidden,
                        "Retained GDN requires the original FP16RNE alpha/beta subgroup");
                    if (!retainsHalf())
                    {
                        engines_[i]->setGPUStream(stream);
                        require(engines_[i]->prepareFusedProjectionGraphCapture(2),
                            "Retained GDN half subgroup cold preparation failed");
                        half_.push_back({engines_[i], outputs_[i].tensor, columns_[i], nullptr, names[i]});
                    }
                    half_weights_[i - 2] = view;
                }

                scratch_ = workspace_->retainBuffer(scratchBank, 0, native::scratchBytes());
                if (!retainsHalf())
                    half_bank_ = workspace_->retainBuffer(GemmWorkspaceBuffers::FULL_A_FP16, 0,
                        std::size_t(hidden) * sizeof(std::uint16_t));
                native::Invocation invocation{.prefix = prefix, .projections = {},
                    .geometry = native::EarlyQ8Geometry::GDN, .scratch = scratch_->data(),
                    .scratch_bytes = native::scratchBytes(),
                    .trunk_source = hc_->trunkSource() == Qwen4ExpHCTrunkSource::PreviousWrite
                        ? native::TrunkSource::PreviousWrite : native::TrunkSource::PublishedTrunk,
                    .half_policy = half_policy_};
                if (retainsHalf())
                {
                    invocation.half = {
                        {static_cast<const std::uint16_t *>(half_weights_[0].data), half_weights_[0].bytes},
                        {static_cast<const std::uint16_t *>(half_weights_[1].data), half_weights_[1].bytes},
                        static_cast<float *>(outputs_[2].tensor->gpu_data_ptr()),
                        static_cast<float *>(outputs_[3].tensor->gpu_data_ptr()),
                        columns_[2]};
                }
                for (std::size_t i = 0; i < 2; ++i)
                {
                    auto &view = invocation.projections[i].weights;
                    require(engines_[i]->exportNativeQuantisedMatrixDesc(view) &&
                        view.valid() && view.n == columns_[i] && view.k == hidden &&
                        view.source_identity_present,
                        "Retained GDN requires valid NativeQuantised QKV and Z matrices");
                    invocation.projections[i].output = static_cast<float *>(outputs_[i].tensor->gpu_data_ptr());
                }
                if (full_gdn_)
                {
                    const auto &c = full_gdn_->conv;
                    const auto &r = full_gdn_->rec;
                    const auto &o = full_gdn_->out_proj;
                    const std::array<int, 3> full_cols{columns_[0], columns_[1], o.n};
                    for (std::size_t i = 0; i < full_outputs_.size(); ++i)
                    {
                        full_outputs_[i].lease = TransferEngine::instance().retainCapturedDeviceStorage(
                            *full_outputs_[i].tensor, device(), std::size_t(full_cols[i]) * sizeof(float));
                    }
                    const auto ssm_source = o.prepared_store->binding(*o.prepared_ref);
                    ssm_out_engine_ = o.prepared_store->gemmKernel(*o.prepared_ref);
                    require(ssm_source && ssm_source->immutable && ssm_source->tensor == o.B && ssm_out_engine_,
                        "Retained Full GDN lost its immutable ssm_out prepared source or engine");
                    const auto *conv_kernel = dynamic_cast<const ROCmShortConvolution *>(c.kernel);
                    const auto *rec_kernel = dynamic_cast<const ROCmGatedDeltaNet *>(r.kernel);
                    const int conv_state_size = columns_[0] * (c.kernel_size - 1);
                    const int rec_state_size = r.n_heads * r.d_k * r.d_v;
                    conv_state_ = conv_kernel ? conv_kernel->borrowedLiveDeviceState(conv_state_size) : nullptr;
                    rec_state_ = rec_kernel ? rec_kernel->borrowedLiveDeviceState(rec_state_size) : nullptr;
                    require(conv_state_ && rec_state_,
                        "Retained Full GDN requires already-bound ROCm conv and recurrence device state");

                    auto weightPtr = [](const ITensor *t) -> const float * {
                        const auto *tb = dynamic_cast<const TensorBase *>(t);
                        return tb ? static_cast<const float *>(tb->gpu_data_ptr()) : nullptr;
                    };
                    auto &fg = invocation.full_gdn;
                    fg.enabled = true;
                    fg.conv_weight = weightPtr(c.weight);
                    fg.conv_bias = c.bias ? weightPtr(c.bias) : nullptr;
                    fg.conv_state = conv_state_;
                    fg.gdn_recurrence_in = static_cast<float *>(full_outputs_[0].tensor->gpu_data_ptr());
                    fg.A_log = weightPtr(r.A_log);
                    fg.dt_bias = weightPtr(r.dt_bias);
                    fg.recurrence_state = rec_state_;
                    fg.attn_output = static_cast<float *>(full_outputs_[1].tensor->gpu_data_ptr());
                    fg.ssm_norm_gamma = weightPtr(o.gated_norm->gamma);
                    fg.ssm_norm_eps = o.gated_norm->parameters.eps;
                    fg.ssm_norm_subtract_one = o.gated_norm->parameters.subtract_one;
                    fg.ssm_norm_activation = o.gated_norm->parameters.activation;
                    fg.n_k_heads = r.n_k_heads;
                    fg.n_v_heads = r.n_heads;
                    fg.global_v_offset = r.global_v_head_offset;
                    fg.use_qk_l2norm = r.use_qk_l2norm;
                    require(ssm_out_engine_->exportNativeQuantisedMatrixDesc(fg.ssm_out.weights) &&
                        fg.ssm_out.weights.valid() && fg.ssm_out.weights.n == o.n &&
                        fg.ssm_out.weights.k == columns_[1] &&
                        fg.ssm_out.weights.source_identity_present,
                        "Retained Full GDN requires a valid NativeQuantised ssm_out matrix");
                    fg.ssm_out.output = static_cast<float *>(full_outputs_[2].tensor->gpu_data_ptr());
                }

                native::Admission admitted{.invocation = invocation, .function = nullptr, .resources = {}, .device = -1};
                require(native::prepare(invocation, device().ordinal, admitted) == hipSuccess,
                    "Retained GDN native POST/HC/projection admission failed");
                TransferEngine::instance().initializeWorkspaceMetadata(*scratch_,
                    std::vector<std::byte>(native::scratchBytes(), std::byte{0}), stream);
                workspace_id_ = workspace_->id();
                stream_ = stream;
                admission_ = admitted;
                validateStorage(stream);
                return true;
            }

            /** @brief Submit the native HC/Q8 prefix (or full GDN megakernel) on one stream. */
            bool execute(IDeviceContext *ctx) override
            {
                require(ctx && ctx->deviceId() == device(), "Retained GDN execution changed participant");
                const auto execution = gpuExecution();
                validateStorage(execution.nativeStream());
                for (const auto &input : phase_inputs_) execution.requirePreparedInput(input.tensor);
                for (auto *weight : phase_weights_) execution.requirePreparedInput(weight);
                for (const auto &output : phase_outputs_) execution.requirePreparedOutput(output.tensor);
                for (const auto &output : outputs_) execution.requirePreparedOutput(output.tensor);
                if (full_gdn_)
                {
                    for (auto *weight : full_weights_) execution.requirePreparedInput(weight);
                    for (const auto &output : full_outputs_) execution.requirePreparedOutput(output.tensor);
                }

                if (native::launch(*admission_, static_cast<hipStream_t>(execution.nativeStream())) != hipSuccess)
                    throw std::runtime_error("Retained GDN native prefix enqueue failed");
                if (!retainsHalf())
                {
                    execution.bind(engines_[2]);
                    execution.bind(engines_[3]);
                    auto row_scope = engines_[2]->beginVerifierDecodeEquivalentScope(rows_);
                    if (!engines_[2]->multiply_fused_tensor(mixed_, half_, 1, hidden, nullptr, workspace_))
                        throw std::runtime_error("Retained GDN original FP16 alpha/beta subgroup enqueue failed");
                }
                if (full_gdn_ && full_gdn_->rec.sequence_state_cache &&
                    !full_gdn_->rec.sequence_state_cache->advanceRecurrentSequenceState(
                        full_gdn_->rec.layer_idx, 1, nullptr, 1, execution.nativeStream()))
                {
                    throw std::runtime_error("Retained Full GDN recurrent sequence frontier publication failed");
                }
                for (const auto &output : phase_outputs_) execution.publish(output.tensor);
                for (const auto &output : outputs_) execution.publish(output.tensor);
                if (full_gdn_)
                {
                    for (const auto &output : full_outputs_) execution.publish(output.tensor);
                    PerfStatsCollector::addCounter("kernel", "rocm_retained_gdn_full_megakernel_calls",
                        1., "graph_capture", device().toString(),
                        {{"m", "1"}, {"h", "2560"}, {"n_v_heads", std::to_string(full_gdn_->rec.n_heads)}});
                }
                else
                {
                    PerfStatsCollector::addCounter("kernel", "rocm_retained_gdn_pre_attention_calls",
                        1., "graph_capture", device().toString(), {{"m", "1"}, {"h", "2560"},
                            {"half", retainsHalf() ? "retained_fp16rne" : "separate_fp16rne"}});
                }
                return true;
            }

            /** @return Original declaration, storage, count, workspace generation and registered function identity. */
            std::uint64_t graphCaptureVariantSignature() const override
            {
                auto identity = projection_.graphCaptureVariantSignature();
                const auto mix = [&identity](std::uint64_t item) { identity = (identity ^ item) * 1099511628211ull; };
                mix(hc_->phaseDeclaration().graphCaptureVariantSignature());
                mix(static_cast<std::uint64_t>(half_policy_));
                mix(full_gdn_.has_value() ? 0x46554c4c47444e31ull : 0ull);
                if (seal_) { mix(seal_->first); mix(seal_->second); }
                if (admission_)
                {
                    validateStorage(stream_);
                    mix(reinterpret_cast<std::uintptr_t>(admission_->function));
                    mix(static_cast<std::uint64_t>(admission_->invocation.trunk_source));
                    mix(reinterpret_cast<std::uintptr_t>(scratch_->data()));
                    if (half_bank_) mix(reinterpret_cast<std::uintptr_t>(half_bank_->data()));
                    mix(reinterpret_cast<std::uintptr_t>(stream_));
                    mix(workspace_id_);
                    for (const auto &output : outputs_)
                        mix(reinterpret_cast<std::uintptr_t>(output.tensor->gpu_data_ptr()));
                    for (const auto &weight : half_weights_) mix(reinterpret_cast<std::uintptr_t>(weight.data));
                    for (std::size_t i = 0; i < 2; ++i)
                    {
                        const auto &weight = admission_->invocation.projections[i].weights;
                        mix(reinterpret_cast<std::uintptr_t>(weight.payload));
                        mix(reinterpret_cast<std::uintptr_t>(weight.scales));
                    }
                    if (full_gdn_)
                    {
                        const auto &fg = admission_->invocation.full_gdn;
                        mix(reinterpret_cast<std::uintptr_t>(fg.conv_weight));
                        mix(reinterpret_cast<std::uintptr_t>(fg.conv_bias));
                        mix(reinterpret_cast<std::uintptr_t>(fg.conv_state));
                        mix(reinterpret_cast<std::uintptr_t>(fg.A_log));
                        mix(reinterpret_cast<std::uintptr_t>(fg.dt_bias));
                        mix(reinterpret_cast<std::uintptr_t>(fg.recurrence_state));
                        mix(reinterpret_cast<std::uintptr_t>(fg.ssm_norm_gamma));
                        mix(reinterpret_cast<std::uintptr_t>(fg.ssm_out.weights.payload));
                        mix(reinterpret_cast<std::uintptr_t>(fg.ssm_out.weights.scales));
                        for (const auto &fo : full_outputs_)
                            mix(reinterpret_cast<std::uintptr_t>(fo.tensor->gpu_data_ptr()));
                    }
                }
                return identity;
            }

            /** @brief Drop cold bindings only after the enclosing executable is retired. */
            void invalidateKernelDynamicState() override
            {
                admission_.reset();
                scratch_.reset();
                half_bank_.reset();
                half_.clear();
                for (auto &output : outputs_) output.lease.reset();
                for (auto &output : full_outputs_) output.lease.reset();
                ssm_out_engine_ = nullptr;
                conv_state_ = nullptr;
                rec_state_ = nullptr;
                stream_ = nullptr;
                seal_.reset();
                hc_->invalidatePhaseBindings();
            }

            /** @return Original observable retained outputs; no omitted intermediary is reported as produced. */
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
                if (full_gdn_)
                {
                    constexpr std::array full_labels{"gdn_recurrence_in", "attn_output", "attn_proj"};
                    for (std::size_t i = 0; i < full_outputs_.size(); ++i)
                        info.addOutput(full_labels[i], full_outputs_[i].tensor, 1, full_outputs_[i].tensor->cols());
                }
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

            /** @return Whether the native prefix owns the original half projections. */
            bool retainsHalf() const noexcept
            {
                return half_policy_ == native::HalfGDNPolicy::RetainedPreparedFP16RNE;
            }

            /** @return Engines from immutable original prepared source bindings. */
            std::array<ITensorGemm *, 4> resolveEngines() const
            {
                std::array<ITensorGemm *, 4> engines{};
                for (std::size_t i = 0; i < engines.size(); ++i)
                {
                    require(refs_[i] && refs_[i]->device == device(),
                        "Retained GDN lost its exact prepared projection ref");
                    const auto source = params().prepared_store->binding(*refs_[i]);
                    engines[i] = params().prepared_store->gemmKernel(*refs_[i]);
                    require(source && source->immutable && source->tensor == inputs_[i] &&
                        engines[i] && (!cached_[i] || cached_[i] == engines[i]),
                        "Retained GDN original frozen prepared source or engine changed");
                }
                return engines;
            }

            /** @brief Check exact graph storage and immutable engine authority before enqueue. */
            void validateStorage(void *stream) const
            {
                require(admission_ && stream == stream_ && workspace_ &&
                    workspace_->id() == workspace_id_ &&
                    workspace_->getBuffer(scratchBank) == scratch_->data() &&
                    (retainsHalf() || (half_bank_ &&
                    workspace_->getBuffer(GemmWorkspaceBuffers::FULL_A_FP16) == half_bank_->data())),
                    "Retained GDN captured stream or workspace generation changed");
                hc_->validatePhaseBindings(stream);
                require(resolveEngines() == engines_, "Retained GDN prepared scalar engine changed");
                for (std::size_t i = 0; i < outputs_.size(); ++i)
                    TransferEngine::instance().requireCapturedDeviceStorage(*outputs_[i].lease,
                        *outputs_[i].tensor, device(), std::size_t(columns_[i]) * sizeof(float));
                if (full_gdn_)
                {
                    const auto &c = full_gdn_->conv;
                    const auto &r = full_gdn_->rec;
                    const auto &o = full_gdn_->out_proj;
                    const std::array<int, 3> full_cols{columns_[0], columns_[1], o.n};
                    for (std::size_t i = 0; i < full_outputs_.size(); ++i)
                        TransferEngine::instance().requireCapturedDeviceStorage(*full_outputs_[i].lease,
                            *full_outputs_[i].tensor, device(), std::size_t(full_cols[i]) * sizeof(float));
                    require(o.prepared_store->gemmKernel(*o.prepared_ref) == ssm_out_engine_,
                        "Retained Full GDN ssm_out engine changed");
                    const auto *conv_kernel = dynamic_cast<const ROCmShortConvolution *>(c.kernel);
                    const auto *rec_kernel = dynamic_cast<const ROCmGatedDeltaNet *>(r.kernel);
                    require(conv_kernel &&
                        conv_kernel->borrowedLiveDeviceState(columns_[0] * (c.kernel_size - 1)) == conv_state_ &&
                        rec_kernel &&
                        rec_kernel->borrowedLiveDeviceState(r.n_heads * r.d_k * r.d_v) == rec_state_,
                        "Retained Full GDN live device state owner changed");
                }
            }

            GDNProjectionStage projection_;
            std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc_;
            const std::array<int, 4> columns_;
            const std::array<const ITensor *, 4> inputs_;
            const std::array<std::optional<PreparedWeightRef>, 4> refs_;
            const std::array<ITensorGemm *, 4> cached_;
            const native::HalfGDNPolicy half_policy_;
            std::optional<FullGDNDeclaration> full_gdn_;
            std::array<Output, 4> outputs_{};
            std::array<Output, 3> full_outputs_{};
            std::optional<std::array<Qwen4ExpFloatBinding, 2>> optional_; ///< Original omitted diagnostic ports.
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
            std::vector<ITensorGemm::TensorProjectionDesc> half_;
            DeviceRowRange rows_ = DeviceRowRange::fullyActive(1);
            DeviceWorkspaceManager *workspace_ = nullptr;
            std::uint64_t workspace_id_ = 0;
            void *stream_ = nullptr;
            std::shared_ptr<const WorkspaceBufferLease> scratch_;
            std::shared_ptr<const WorkspaceBufferLease> half_bank_;
            std::optional<native::Admission> admission_;
            TensorBase *mixed_ = nullptr;
        };
    }

    WorkspaceRequirements retainedGDNPreAttentionWorkspace()
    {
        WorkspaceRequirements requirements;
        requirements.buffers.emplace_back(scratchBank, native::scratchBytes(), 16, true,
            WorkspaceExecutionRegime::Any, WorkspaceContentLifetime::SerialGraphFamily);
        return requirements;
    }

    std::unique_ptr<IComputeStage> makeRetainedGDNPreAttentionStage(
        const GDNProjectionStage::Params &params, std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc)
    {
        return std::make_unique<RetainedGDNPreAttentionStage>(params, std::move(hc),
            native::HalfGDNPolicy::RetainedPreparedFP16RNE);
    }
    void lowerRetainedQwenGDNPreAttention(ComputeGraph &graph, const std::string &gdnProjectionName,
                                        const std::string &previousInputBoundary)
    {
        using Primitive = Qwen4ExpPrimitiveStage;
        const auto *target = graph.getNode(gdnProjectionName);
        const auto *projection = target ? dynamic_cast<const GDNProjectionStage *>(target->stage.get()) : nullptr;
        require(projection, "Retained pre-attention lowerer requires the original GDN projection");
        const auto &params = projection->getParams();
        require(params.input && params.input_buffer_id, "Retained GDN lost its original mixed binding");

        std::string readName;
        for (const auto &dependency : target->dependencies)
        {
            const auto *node = graph.getNode(dependency);
            const auto *primitive = node ? dynamic_cast<const Primitive *>(node->stage.get()) : nullptr;
            const auto *read = primitive ? std::get_if<Primitive::ResidualRead>(&primitive->getParams().operation) : nullptr;
            if (read && read->output.tensor == params.input && read->output.buffer == *params.input_buffer_id &&
                read->output.byteOffset == 0)
            {
                require(readName.empty(), "Retained GDN has ambiguous original mixed producers");
                readName = dependency;
            }
        }
        require(!readName.empty(), "Retained GDN requires its original direct HC residual-read producer");
        const auto *boundary = previousInputBoundary.empty() ? nullptr : graph.getNode(previousInputBoundary);
        const auto *boundary_prim = boundary ? dynamic_cast<const Primitive *>(boundary->stage.get()) : nullptr;
        const auto *boundary_write = boundary_prim
            ? std::get_if<Primitive::ResidualWrite>(&boundary_prim->getParams().operation) : nullptr;
        const auto injection_policy = boundary_write
            ? Qwen4ExpHCInjectionOwnership::DiscoverInRegion
            : Qwen4ExpHCInjectionOwnership::KeepOriginalStage;
        auto region = Qwen4ExpHyperconnectionBuilder::describeFullNativeReadRegion(graph, previousInputBoundary,
            readName, injection_policy);
        const auto *normNode = graph.getNode(region.branch_norm);
        const auto &norm = std::get<Primitive::BranchRMSNorm>(
            static_cast<const Primitive &>(*normNode->stage).getParams().operation);
        if (!previousInputBoundary.empty())
            require(boundary && std::find(normNode->dependencies.begin(), normNode->dependencies.end(),
                previousInputBoundary) != normNode->dependencies.end(),
                "Retained GDN lost its original merged input frontier");
        auto source = Qwen4ExpHCTrunkSource::PreviousWrite;
        if (!boundary_write || boundary_write->output.tensor != norm.input.tensor ||
            boundary_write->output.buffer != norm.input.buffer ||
            boundary_write->output.byteOffset != norm.input.byteOffset)
        {
            source = Qwen4ExpHCTrunkSource::PublishedTrunk;
            region.previous_write.clear();
            region.injection.clear();
        }

        // Detect an eligible closed downstream GDN chain
        // (gdn_proj -> short_conv -> gdn_recurrence -> gdn_out_proj) so the
        // entire local GDN attention block fuses into one resident megakernel.
        std::optional<FullGDNDeclaration> full_gdn;
        std::string conv_name, rec_name, out_proj_name;
        const auto proj_consumers = graph.directConsumersOf(gdnProjectionName);
        if (proj_consumers.size() == 1)
        {
            const auto *conv_node = graph.getNode(proj_consumers[0]);
            const auto *conv_stage = conv_node ? dynamic_cast<const ShortConv1dStage *>(conv_node->stage.get()) : nullptr;
            if (conv_stage)
            {
                const auto conv_consumers = graph.directConsumersOf(proj_consumers[0]);
                const auto *rec_node = conv_consumers.size() == 1 ? graph.getNode(conv_consumers[0]) : nullptr;
                const auto *rec_stage = rec_node ? dynamic_cast<const GDNRecurrenceStage *>(rec_node->stage.get()) : nullptr;
                if (rec_stage)
                {
                    const auto rec_consumers = graph.directConsumersOf(conv_consumers[0]);
                    const auto *out_node = rec_consumers.size() == 1 ? graph.getNode(rec_consumers[0]) : nullptr;
                    const auto *out_stage = out_node ? dynamic_cast<const GEMMStage *>(out_node->stage.get()) : nullptr;
                    if (out_stage)
                    {
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
                                return dep == gdnProjectionName || dep == proj_consumers[0] || dep == conv_consumers[0];
                            });
                        if (out_deps_closed &&
                            cp.device_id == params.device_id && rp.device_id == params.device_id &&
                            op.device_id == params.device_id &&
                            cp.seq_len == 1 && cp.request_count == 1 && cp.kernel_size == 4 &&
                            cp.channels == params.n_qkv && (params.n_qkv & 7) == 0 &&
                            cp.input == params.output_qkv && cp.output && cp.output_buffer_id && cp.weight &&
                            !cp.request_seq_lens_device && cp.verifier_state_capture_rows == 0 &&
                            cp.speculative_state_slot_rows == 0 && (raw_qk || prep_qk) &&
                            dynamic_cast<const ROCmShortConvolution *>(cp.kernel) != nullptr &&
                            rp.seq_len == 1 && rp.request_count == 1 && rp.d_k == 128 && rp.d_v == 128 &&
                            rp.n_heads == params.n_a && rp.n_heads >= 4 && rp.n_heads <= 48 &&
                            rp.n_heads * 128 == params.n_z && (params.n_z & 7) == 0 &&
                            rp.n_k_heads > 0 && 2 * rp.n_k_heads * 128 + rp.n_heads * 128 == params.n_qkv &&
                            rp.Q == cp.output && rp.K == cp.output && rp.V == cp.output &&
                            rp.alpha == params.output_a && rp.beta == params.output_b &&
                            rp.A_log && rp.dt_bias && rp.output && rp.output_buffer_id &&
                            !rp.request_seq_lens_device && !rp.gpu_retained_workspace &&
                            rp.verifier_state_capture_rows == 0 && rp.speculative_state_slot_rows == 0 &&
                            dynamic_cast<const ROCmGatedDeltaNet *>(rp.kernel) != nullptr &&
                            rp.kernel->recurrenceStateLayout() == GDNRecurrenceStateLayout::ValueMajorKeyContiguous &&
                            op.m == 1 && op.n == hidden && op.k == params.n_z &&
                            op.A == rp.output && op.B &&
                            op.B->native_type() != TensorType::FP32 &&
                            op.B->native_type() != TensorType::FP16 &&
                            op.B->native_type() != TensorType::BF16 &&
                            op.C && op.c_buffer_id &&
                            !op.bias && !op.bias_tensor && op.alpha == 1.0f && op.beta == 0.0f &&
                            op.gated_norm.has_value() && op.gated_norm->gate == params.output_z &&
                            op.gated_norm->gamma && op.gated_norm->parameters.norm_dim == 128 &&
                            op.gated_norm->parameters.gamma_period == 128 &&
                            op.prepared_store && op.prepared_ref && op.prepared_ref->device == params.device_id)
                        {
                            conv_name = proj_consumers[0];
                            rec_name = conv_consumers[0];
                            out_proj_name = rec_consumers[0];
                            full_gdn = FullGDNDeclaration{cp, rp, op};
                        }
                    }
                }
            }
        }

        // Only absorb hc_attn_write_projection when lowering the Full GDN
        // megakernel and injection is not the graph's sealed terminal node.
        if (!full_gdn.has_value() || region.injection == graph.terminalNode())
            region.injection.clear();

        auto phase = Qwen4ExpHyperconnectionBuilder::retainFullNativeReadPhase(graph, region, source);
        auto replacement = std::make_unique<RetainedGDNPreAttentionStage>(
            params, std::move(phase), native::HalfGDNPolicy::RetainedPreparedFP16RNE, std::move(full_gdn));
        GraphComposer composer(graph);
        std::vector<GraphNodeRef> members;
        if (!region.previous_write.empty()) members.push_back(composer.ref(region.previous_write));
        members.push_back(composer.ref(region.branch_norm));
        members.push_back(composer.ref(region.down));
        members.push_back(composer.ref(region.up));
        members.push_back(composer.ref(region.read));
        members.push_back(composer.ref(gdnProjectionName));
        if (!region.injection.empty()) members.push_back(composer.ref(region.injection));
        GraphNodeRef replacement_terminal = composer.ref(gdnProjectionName);
        if (!out_proj_name.empty())
        {
            members.push_back(composer.ref(conv_name));
            members.push_back(composer.ref(rec_name));
            replacement_terminal = composer.ref(out_proj_name);
            members.push_back(replacement_terminal);
        }
        (void)lowerLocalMegakernelRegion(graph, LocalMegakernelRegionSpec{
            .members = std::move(members),
            .terminal = replacement_terminal,
            .replacement = std::move(replacement),
        });
    }

}
#endif
