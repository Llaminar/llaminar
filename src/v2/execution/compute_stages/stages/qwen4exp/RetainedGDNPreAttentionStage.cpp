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
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
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
        constexpr std::array<int, 4> columns{2560, 1536, 12, 12};
        constexpr std::array<const char *, 4> names{"qkv", "z", "alpha", "beta"};

        /** @brief Authenticate a complete original local owner before native recording. */
        void require(bool condition, const char *message)
        {
            if (!condition) throw std::invalid_argument(message);
        }

        class RetainedGDNPreAttentionStage final : public IComputeStage, public IWorkspaceConsumer, public IGraphOutputDemand
        {
        public:
            /** @brief Seal the original source and output declarations before any graph mutation. */
            RetainedGDNPreAttentionStage(const GDNProjectionStage::Params &params,
                                        std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc, native::HalfGDNPolicy halfPolicy)
                : IComputeStage(params.device_id), projection_(params), hc_(std::move(hc)),
                  inputs_{params.w_qkv, params.w_z, params.w_a, params.w_b},
                  refs_{params.prepared_ref_qkv, params.prepared_ref_z,
                        params.prepared_ref_a, params.prepared_ref_b},
                  cached_{params.gemm_qkv, params.gemm_z, params.gemm_a, params.gemm_b}, half_policy_(halfPolicy)
            {
                require(hc_ && device().is_rocm() && hc_->phaseDeclaration().device() == device() &&
                    params.m == 1 && params.k == hidden &&
                    std::array{params.n_qkv, params.n_z, params.n_a, params.n_b} == columns &&
                    !params.force_decode_equivalent_verifier_prefill && !params.verifier_row_range &&
                    params.qkv_operand_precision == QuantizedActivationOperandPrecision::SingleQ8 &&
                    params.prepared_store && params.input && params.input_buffer_id,
                    "Retained GDN pre-attention requires the original ordinary ROCm M1 QKV/Z/A/B declarations");

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
                        tensor->rows() >= 1 && tensor->cols() == std::size_t(columns[i]),
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
                require(phase_inputs_.size() == (ownsWrite ? 3u : 1u) &&
                    phase_outputs_.size() == (ownsWrite ? 4u : 3u),
                    "Retained GDN requires original POST/HC outputs with independent injection outside");
                for (auto *weight : hc_->phaseDeclaration().bufferContract().weight_tensors)
                    phase_weights_.push_back(weight);
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
                return result;
            }

            /** @return Existing HC and GDN geometries without a new activation allocation. */
            StageBufferRequirements getBufferRequirements() const override
            {
                auto result = hc_->phaseDeclaration().getBufferRequirements();
                for (std::size_t i = 0; i < outputs_.size(); ++i)
                    result.addOutput(names[i], outputs_[i].tensor->shape(), BufferTensorType::FP32);
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
                    result.merge(workspace->getWorkspaceRequirements(1, columns[i], hidden));
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
                require(!prefix.hc.injection.weights && !prefix.hc.injection.output &&
                    prefix.hc.mixed == mixed_->gpu_data_ptr(),
                    "Retained GDN cannot replace independent injection or a foreign mixed output");
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
                        *output.tensor, device(), std::size_t(columns[i]) * sizeof(float));
                }
                half_.clear();
                for (std::size_t i = 2; i < engines_.size(); ++i)
                {
                    ContiguousFloatingPointWeightDescriptor view;
                    require(engines_[i]->floatingActivationOperandPrecision() == FloatingActivationOperandPrecision::FP16RNE &&
                        engines_[i]->exportContiguousFloatingPointWeights(view) && view.valid() &&
                        view.type == TensorType::FP16 && view.n == columns[i] && view.k == hidden,
                        "Retained GDN requires the original FP16RNE alpha/beta subgroup");
                    if (!retainsHalf())
                    {
                        engines_[i]->setGPUStream(stream);
                        require(engines_[i]->prepareFusedProjectionGraphCapture(2),
                            "Retained GDN half subgroup cold preparation failed");
                        half_.push_back({engines_[i], outputs_[i].tensor, columns[i], nullptr, names[i]});
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
                        static_cast<float *>(outputs_[3].tensor->gpu_data_ptr())};
                }
                for (std::size_t i = 0; i < 2; ++i)
                {
                    auto &view = invocation.projections[i].weights;
                    require(engines_[i]->exportNativeQuantisedMatrixDesc(view) &&
                        view.valid() && view.n == columns[i] && view.k == hidden &&
                        view.codebook_id == 19 && view.arithmeticPolicyCodebookId() == 19,
                        "Retained GDN requires the original Q8_0 QKV and Z matrices");
                    invocation.projections[i].output = static_cast<float *>(outputs_[i].tensor->gpu_data_ptr());
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

            /** @brief Submit the native HC/Q8 prefix followed by the unchanged half subgroup. */
            bool execute(IDeviceContext *ctx) override
            {
                require(ctx && ctx->deviceId() == device(), "Retained GDN execution changed participant");
                const auto execution = gpuExecution();
                validateStorage(execution.nativeStream());
                for (const auto &input : phase_inputs_) execution.requirePreparedInput(input.tensor);
                for (auto *weight : phase_weights_) execution.requirePreparedInput(weight);
                for (const auto &output : phase_outputs_) execution.requirePreparedOutput(output.tensor);
                for (const auto &output : outputs_) execution.requirePreparedOutput(output.tensor);

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
                for (const auto &output : phase_outputs_) execution.publish(output.tensor);
                for (const auto &output : outputs_) execution.publish(output.tensor);
                PerfStatsCollector::addCounter("kernel", "rocm_retained_gdn_pre_attention_calls",
                    1., "graph_capture", device().toString(), {{"m", "1"}, {"h", "2560"},
                        {"half", retainsHalf() ? "retained_fp16rne" : "separate_fp16rne"}});
                return true;
            }

            /** @return Original declaration, storage, count, workspace generation and registered function identity. */
            std::uint64_t graphCaptureVariantSignature() const override
            {
                auto identity = projection_.graphCaptureVariantSignature();
                const auto mix = [&identity](std::uint64_t item) { identity = (identity ^ item) * 1099511628211ull; };
                mix(hc_->phaseDeclaration().graphCaptureVariantSignature());
                mix(static_cast<std::uint64_t>(half_policy_));
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
                    constexpr std::array labels{"hc_post", "normalized", "raw_down", "mixed"};
                    const auto &output = phase_outputs_[i];
                    info.addOutput(labels[i + (ownsWrite ? 0 : 1)], output.tensor, 1, output.tensor->cols());
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
                        *outputs_[i].tensor, device(), std::size_t(columns[i]) * sizeof(float));
            }

            GDNProjectionStage projection_;
            std::shared_ptr<Qwen4ExpHCReadPhaseOwner> hc_;
            const std::array<const ITensor *, 4> inputs_;
            const std::array<std::optional<PreparedWeightRef>, 4> refs_;
            const std::array<ITensorGemm *, 4> cached_;
            const native::HalfGDNPolicy half_policy_;
            std::array<Output, 4> outputs_{};
            std::optional<std::array<Qwen4ExpFloatBinding, 2>> optional_; ///< Original omitted diagnostic ports.
            std::vector<Qwen4ExpFloatBinding> phase_inputs_;
            std::vector<Qwen4ExpFloatBinding> phase_outputs_;
            std::vector<ITensor *> phase_weights_;
            std::optional<std::pair<std::uint64_t, std::uint64_t>> seal_;
            std::array<ITensorGemm *, 4> engines_{};
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
        auto region = Qwen4ExpHyperconnectionBuilder::describeFullNativeReadRegion(graph, previousInputBoundary,
            readName, Qwen4ExpHCInjectionOwnership::KeepOriginalStage);
        const auto *normNode = graph.getNode(region.branch_norm);
        const auto &norm = std::get<Primitive::BranchRMSNorm>(
            static_cast<const Primitive &>(*normNode->stage).getParams().operation);
        const auto *boundary = previousInputBoundary.empty() ? nullptr : graph.getNode(previousInputBoundary);
        if (!previousInputBoundary.empty())
            require(boundary && std::find(normNode->dependencies.begin(), normNode->dependencies.end(),
                previousInputBoundary) != normNode->dependencies.end(),
                "Retained GDN lost its original merged input frontier");
        const auto *primitive = boundary ? dynamic_cast<const Primitive *>(boundary->stage.get()) : nullptr;
        const auto *write = primitive ? std::get_if<Primitive::ResidualWrite>(&primitive->getParams().operation) : nullptr;
        auto source = Qwen4ExpHCTrunkSource::PreviousWrite;
        if (!write || write->output.tensor != norm.input.tensor || write->output.buffer != norm.input.buffer ||
            write->output.byteOffset != norm.input.byteOffset)
        {
            source = Qwen4ExpHCTrunkSource::PublishedTrunk;
            region.previous_write.clear();
        }

        auto phase = Qwen4ExpHyperconnectionBuilder::retainFullNativeReadPhase(graph, region, source);
        auto replacement = makeRetainedGDNPreAttentionStage(params, std::move(phase));
        std::vector<std::string> members{region.branch_norm, region.down, region.up, region.read, gdnProjectionName};
        if (!region.previous_write.empty()) members.insert(members.begin(), region.previous_write);

        // One outside-reader pass also captures the original independent
        // norm→injection edge. The graph authenticates the exported endpoints
        // and rejects contractions that would introduce a cycle.
        std::vector<LocalRegionExport> exports;
        for (const auto &member : members)
            if (member != gdnProjectionName) exports.push_back({member, {}});
        for (const auto &name : graph.getExecutionOrder())
        {
            if (std::find(members.begin(), members.end(), name) != members.end()) continue;
            const auto &dependencies = graph.getNode(name)->dependencies;
            for (auto &port : exports)
                if (std::find(dependencies.begin(), dependencies.end(), port.producer) != dependencies.end())
                    port.consumers.push_back(name);
        }
        std::erase_if(exports, [](const auto &port) { return port.consumers.empty(); });
        graph.replaceClosedLocalRegion(members, gdnProjectionName, std::move(replacement), exports);
    }

}
#endif
