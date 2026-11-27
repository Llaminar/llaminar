/**
 * @file MoEExpertComputeStageOverlay.cpp
 * @brief MoEExpertComputeStage workspace binding, sparse overlay invocation, and auxiliary stage execution.
 */
#include "MoEExpertComputeStageInternal.h"

namespace llaminar2
{

    DeviceWorkspaceManager *MoEExpertComputeStage::getWorkspace() const
    {
        return bound_workspace_;
    }

    // =========================================================================
    // SharedExpertFFNStage — Dense SwiGLU on shared expert
    // =========================================================================

    namespace
    {
        /** @brief Validate the numerical scope without inspecting GPU data or allocating storage. */
        void validateSharedPublication(SharedExpertOutputRole role, MoEOverlayServicePhaseHint phase, DeviceId device)
        {
            if ((role != SharedExpertOutputRole::CompleteContribution && role != SharedExpertOutputRole::ParticipantLocalPartial) ||
                (phase != MoEOverlayServicePhaseHint::Auto && phase != MoEOverlayServicePhaseHint::Decode &&
                 phase != MoEOverlayServicePhaseHint::Prefill && phase != MoEOverlayServicePhaseHint::GroupedVerifier) ||
                (role == SharedExpertOutputRole::ParticipantLocalPartial && (!device.is_gpu() || phase == MoEOverlayServicePhaseHint::Auto)))
                throw std::invalid_argument("Shared local-partial publication requires an explicit GPU numerical phase");
        }
        /** @brief Hash only immutable role/phase policy; zero preserves the existing complete-row identity. */
        uint64_t sharedPublicationSignature(SharedExpertOutputRole role, MoEOverlayServicePhaseHint phase)
        {
            if (role == SharedExpertOutputRole::CompleteContribution && phase == MoEOverlayServicePhaseHint::Auto) return 0;
            auto identity = (kFNV1a64OffsetBasis ^ static_cast<uint64_t>(role)) * kFNV1a64Prime;
            identity = (identity ^ static_cast<uint64_t>(phase)) * kFNV1a64Prime;
            return identity ? identity : 1;
        }
    }

    SharedExpertFFNStage::SharedExpertFFNStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        validateSharedPublication(params_.output_role, params_.phase, params_.device_id);
        scratch_gate_ = dynamic_cast<FP32Tensor *>(params_.gate_scratch);
        scratch_up_ = dynamic_cast<FP32Tensor *>(params_.up_scratch);

        /*
         * Standalone CPU stages have no BufferArena executor. Materialize their
         * host scratch once at construction. GPU stages are intentionally not
         * given an analogous path: their stable device buffers are a mandatory
         * graph-construction contract.
         */
        if (params_.device_id.is_cpu() &&
            params_.seq_len > 0 &&
            params_.intermediate > 0)
        {
            if (!scratch_gate_)
            {
                owned_cpu_scratch_gate_ = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{
                        static_cast<size_t>(params_.seq_len),
                        static_cast<size_t>(params_.intermediate)});
                scratch_gate_ = owned_cpu_scratch_gate_.get();
            }
            if (!scratch_up_)
            {
                owned_cpu_scratch_up_ = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{
                        static_cast<size_t>(params_.seq_len),
                        static_cast<size_t>(params_.intermediate)});
                scratch_up_ = owned_cpu_scratch_up_.get();
            }
        }

        if (scratch_gate_ && scratch_up_ &&
            scratch_gate_->shape().size() >= 2 &&
            scratch_up_->shape().size() >= 2)
        {
            scratch_seq_len_ = static_cast<int>(std::min(
                scratch_gate_->shape()[0],
                scratch_up_->shape()[0]));
        }
    }

    uint64_t SharedExpertFFNStage::graphCaptureVariantSignature() const
    {
        auto signature = sharedPublicationSignature(params_.output_role, params_.phase);
        const bool original_m1=params_.required_router_q8_publication&&params_.required_router_q8_publication->requiredOriginalM1Source();
        if(original_m1) {
            if(moe_kernel_)(void)moe_kernel_->originalM1HiddenQ8BindingSignature();
            signature=(signature^0x4d31513800000001ULL)*kFNV1a64Prime;
        }
        if (!signature) return 0;
        const auto mix = [&](uint64_t value) { signature = (signature ^ value) * kFNV1a64Prime; };
        mix(params_.seq_len); mix(params_.d_model); mix(params_.intermediate);
        mix(params_.force_grouped_verifier_prefill_for_decode); mix(params_.force_decode_equivalent_verifier_prefill);
        mix(params_.disable_grouped_decode_shortcut);
        mix(reinterpret_cast<uintptr_t>(params_.active_row_count_device));
        mix(reinterpret_cast<uintptr_t>(params_.required_router_q8_publication.get()));
        mix(reinterpret_cast<uintptr_t>(params_.prepared_store));
        mix(static_cast<uint64_t>(params_.device_id.type)); mix(params_.device_id.ordinal);
        for (auto *tensor : {params_.input, params_.gate_w, params_.up_w, params_.down_w, params_.output,
                params_.gate_scratch, params_.up_scratch})
        {
            mix(reinterpret_cast<uintptr_t>(tensor));
            mix(reinterpret_cast<uintptr_t>(tensor ? tensor->gpu_data_ptr() : nullptr));
        }
        for (auto id : {params_.input_buffer_id, params_.output_buffer_id,
                params_.gate_scratch_buffer_id, params_.up_scratch_buffer_id}) mix(static_cast<uint64_t>(id));
        for (const auto &ref : {params_.prepared_ref_gate, params_.prepared_ref_up, params_.prepared_ref_down})
            if (ref) {
                mix(ref->model_id.value); mix(ref->binding_id); mix(static_cast<uint64_t>(ref->kind));
                mix(static_cast<uint64_t>(ref->device.type)); mix(ref->device.ordinal);
            }
        return signature ? signature : 1;
    }

    bool SharedExpertFFNStage::validatePlannedScratch(
        int rows,
        int intermediate) const
    {
        if (!scratch_gate_ || !scratch_up_)
        {
            LOG_ERROR(
                "[SharedExpertFFNStage] Missing graph-owned FP32 gate/up scratch; "
                "GPU execution cannot allocate scratch dynamically");
            return false;
        }

        const auto has_capacity = [rows, intermediate](
                                      const FP32Tensor *tensor)
        {
            const auto &shape = tensor->shape();
            return shape.size() >= 2 &&
                   shape[0] >= static_cast<size_t>(rows) &&
                   shape[1] >= static_cast<size_t>(intermediate);
        };
        if (!has_capacity(scratch_gate_) || !has_capacity(scratch_up_))
        {
            LOG_ERROR(
                "[SharedExpertFFNStage] Graph-owned gate/up scratch is undersized: "
                "required_rows="
                << rows
                << " required_intermediate=" << intermediate
                << " gate_shape="
                << (scratch_gate_->shape().size() >= 2
                        ? std::to_string(scratch_gate_->shape()[0]) + "x" +
                              std::to_string(scratch_gate_->shape()[1])
                        : "invalid")
                << " up_shape="
                << (scratch_up_->shape().size() >= 2
                        ? std::to_string(scratch_up_->shape()[0]) + "x" +
                              std::to_string(scratch_up_->shape()[1])
                        : "invalid"));
            return false;
        }

        if (params_.device_id.is_gpu() &&
            (!scratch_gate_->gpu_data_ptr() ||
             !scratch_up_->gpu_data_ptr()))
        {
            LOG_ERROR(
                "[SharedExpertFFNStage] Graph executor did not establish "
                "device storage for arena-owned gate/up scratch");
            return false;
        }
        return true;
    }

    bool SharedExpertFFNStage::validatePreparedWeights(std::string *error) const
    {
        auto fail = [error](const std::string &message)
        {
            if (error)
                *error = message;
            return false;
        };

        if (!params_.gate_w && !params_.up_w && !params_.down_w)
        {
            if (error)
                error->clear();
            return true;
        }

        if (!params_.prepared_store)
            return fail("PreparedWeightStore is required for SharedExpertFFNStage weights");

        auto check = [&](const char *name, const TensorBase *weight, const std::optional<PreparedWeightRef> &ref)
        {
            if (!weight)
                return true;
            if (!ref.has_value())
                return fail(std::string("missing PreparedWeightRef for ") + name);
            if (!params_.prepared_store->contains(ref.value()))
                return fail(std::string("PreparedWeightStore does not contain ref for ") + name);
            return true;
        };

        if (!check("gate_w", params_.gate_w, params_.prepared_ref_gate))
            return false;
        if (!check("up_w", params_.up_w, params_.prepared_ref_up))
            return false;
        if (!check("down_w", params_.down_w, params_.prepared_ref_down))
            return false;

        if (error)
            error->clear();
        return true;
    }

    void SharedExpertFFNStage::ensureGemmEnginesCached() const
    {
        if (cached_gate_gemm_)
            return;

        if (!params_.prepared_store ||
            !params_.prepared_ref_gate.has_value() ||
            !params_.prepared_ref_up.has_value() ||
            !params_.prepared_ref_down.has_value())
        {
            LOG_ERROR("[SharedExpertFFNStage] PreparedWeightStore and gate/up/down PreparedWeightRefs are required");
            return;
        }

        cached_gate_gemm_ = params_.prepared_store->gemmKernel(params_.prepared_ref_gate.value());
        cached_up_gemm_ = params_.prepared_store->gemmKernel(params_.prepared_ref_up.value());
        cached_down_gemm_ = params_.prepared_store->gemmKernel(params_.prepared_ref_down.value());
        if (!cached_gate_gemm_ || !cached_up_gemm_ || !cached_down_gemm_)
        {
            LOG_ERROR("[SharedExpertFFNStage] PreparedWeightRefs were provided but shared expert kernel(s) were missing from PreparedWeightStore. "
                      "gate="
                      << (void *)cached_gate_gemm_ << " up=" << (void *)cached_up_gemm_
                      << " down=" << (void *)cached_down_gemm_);
            return;
        }

        auto bind_if_needed = [this](ITensorGemm *gemm)
        {
            if (!gemm)
                return;
            /*
             * Workspace planning resolves prepared engines before the graph
             * scheduler binds a GPU stream. Propagate a stream only when one
             * already exists; execute() performs the mandatory propagation
             * immediately before launching any kernel.
             */
            if (!params_.device_id.is_gpu() || hasGPUStream())
                bindStageStream(gemm);
            if (bound_workspace_)
            {
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer && !consumer->hasWorkspace())
                    consumer->bindWorkspace(bound_workspace_);
            }
        };
        bind_if_needed(cached_gate_gemm_);
        bind_if_needed(cached_up_gemm_);
        bind_if_needed(cached_down_gemm_);
    }

    bool SharedExpertFFNStage::ensureSharedGroupedGateUpDescriptorTable(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        if (!kernel || !cached_gate_gemm_ || !cached_up_gemm_ || d_model <= 0 || intermediate <= 0)
        {
            LOG_ERROR("[SharedExpertFFNStage] Cannot publish the shared gate/up "
                      "descriptor table: kernel="
                      << perfBool(kernel != nullptr)
                      << " gate_gemm=" << perfBool(cached_gate_gemm_ != nullptr)
                      << " up_gemm=" << perfBool(cached_up_gemm_ != nullptr)
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }

        if (shared_grouped_gateup_desc_table_id_ >= 0 &&
            shared_grouped_gateup_desc_table_d_model_ == d_model &&
            shared_grouped_gateup_desc_table_intermediate_ == intermediate)
        {
            return true;
        }

        /*
         * A shared expert is simply an always-selected one-entry expert table.
         * Export the complete projection triple through the same typed boundary
         * as routed experts so floating model weights do not get mistaken for
         * failed NativeQuantised preparation.  Validating down here as part of the
         * triple also prevents gate/up and down tables from recording different
         * precision families in one captured launch.
         */
        DeviceMoEExpertDescriptor exported;
        if (!exportDeviceMoEExpertWeightDescriptors(
                cached_gate_gemm_,
                cached_up_gemm_,
                cached_down_gemm_,
                d_model,
                intermediate,
                exported))
        {
            LOG_ERROR("[SharedExpertFFNStage] Prepared shared gate/up/down "
                      "weights do not export one supported descriptor family"
                      << " device=" << params_.device_id.to_string()
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate
                      << " gate_type="
                      << (params_.gate_w ? params_.gate_w->dtype_name() : "null")
                      << " up_type="
                      << (params_.up_w ? params_.up_w->dtype_name() : "null")
                      << " down_type="
                      << (params_.down_w ? params_.down_w->dtype_name() : "null"));
            return false;
        }

        const int table_id =
            exported.weight_format == DeviceMoEWeightFormat::NativeQuantised
                ? kernel->uploadGroupedExpertGateUpDescriptorTables(
                      &exported.gate,
                      &exported.up,
                      1,
                      d_model,
                      intermediate)
                : kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(
                      &exported.floating_gate,
                      &exported.floating_up,
                      exported.weight_format,
                      1,
                      d_model,
                      intermediate);
        if (table_id < 0)
        {
            LOG_ERROR("[SharedExpertFFNStage] Backend rejected the typed shared "
                      "gate/up descriptor table"
                      << " device=" << params_.device_id.to_string()
                      << " weight_format="
                      << static_cast<std::uint32_t>(exported.weight_format)
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }

        shared_grouped_gateup_desc_table_id_ = table_id;
        shared_grouped_gateup_desc_table_d_model_ = d_model;
        shared_grouped_gateup_desc_table_intermediate_ = intermediate;
        return true;
    }

    bool SharedExpertFFNStage::ensureSharedGroupedDownDescriptorTable(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        if (!kernel || !cached_down_gemm_ || d_model <= 0 || intermediate <= 0)
        {
            LOG_ERROR("[SharedExpertFFNStage] Cannot publish the shared down "
                      "descriptor table: kernel="
                      << perfBool(kernel != nullptr)
                      << " down_gemm=" << perfBool(cached_down_gemm_ != nullptr)
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }

        if (shared_grouped_down_desc_table_id_ >= 0 &&
            shared_grouped_down_desc_table_d_model_ == d_model &&
            shared_grouped_down_desc_table_intermediate_ == intermediate)
        {
            return true;
        }

        /* Keep the down-table format coupled to the validated gate/up triple. */
        DeviceMoEExpertDescriptor exported;
        if (!exportDeviceMoEExpertWeightDescriptors(
                cached_gate_gemm_,
                cached_up_gemm_,
                cached_down_gemm_,
                d_model,
                intermediate,
                exported))
        {
            LOG_ERROR("[SharedExpertFFNStage] Prepared shared gate/up/down "
                      "weights do not export one supported descriptor family"
                      << " device=" << params_.device_id.to_string()
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate
                      << " gate_type="
                      << (params_.gate_w ? params_.gate_w->dtype_name() : "null")
                      << " up_type="
                      << (params_.up_w ? params_.up_w->dtype_name() : "null")
                      << " down_type="
                      << (params_.down_w ? params_.down_w->dtype_name() : "null"));
            return false;
        }

        const int table_id =
            exported.weight_format == DeviceMoEWeightFormat::NativeQuantised
                ? kernel->uploadGroupedExpertDownDescriptorTable(
                      &exported.down,
                      1,
                      d_model,
                      intermediate)
                : kernel->uploadGroupedExpertFloatingDownDescriptorTable(
                      &exported.floating_down,
                      exported.weight_format,
                      1,
                      d_model,
                      intermediate);
        if (table_id < 0)
        {
            LOG_ERROR("[SharedExpertFFNStage] Backend rejected the typed shared "
                      "down descriptor table"
                      << " device=" << params_.device_id.to_string()
                      << " weight_format="
                      << static_cast<std::uint32_t>(exported.weight_format)
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }

        shared_grouped_down_desc_table_id_ = table_id;
        shared_grouped_down_desc_table_d_model_ = d_model;
        shared_grouped_down_desc_table_intermediate_ = intermediate;
        return true;
    }

    bool SharedExpertFFNStage::shouldUseGroupedVerifierPrefillRoute() const
    {
        return params_.device_id.is_gpu() &&
               !shouldUseDecodeEquivalentVerifierPrefill() &&
               params_.force_grouped_verifier_prefill_for_decode &&
               /*
                * The verifier-row kernels own the complete small-M contract.
                * M=1 still represents an MTP verifier bucket whose counters and
                * publication math must stay aligned with M=2..4 instead of
                * silently taking the ordinary decode shortcut.
                */
               params_.seq_len >= 1 &&
               supportsGroupedPrefillExecutionBackend(params_.device_id);
    }

    bool SharedExpertFFNStage::usesGroupedVerifierPrefillRouteForTesting() const
    {
        return shouldUseGroupedVerifierPrefillRoute();
    }

    bool SharedExpertFFNStage::shouldUseDecodeEquivalentVerifierPrefill() const
    {
        return (params_.device_id.is_cpu() ||
                params_.device_id.is_cuda() ||
                params_.device_id.is_rocm()) &&
               params_.force_decode_equivalent_verifier_prefill &&
               params_.seq_len >= 1;
    }

    bool SharedExpertFFNStage::usesDecodeEquivalentVerifierPrefillForTesting() const
    {
        return shouldUseDecodeEquivalentVerifierPrefill();
    }

    bool SharedExpertFFNStage::usesCPUDecodeEquivalentVerifierPrefillForTesting() const
    {
        return usesDecodeEquivalentVerifierPrefillForTesting();
    }

    bool SharedExpertFFNStage::shouldUseGroupedDecodeRoute() const
    {
        return params_.device_id.is_gpu() &&
               params_.seq_len == 1 &&
               !shouldUseDecodeEquivalentVerifierPrefill() &&
               !params_.disable_grouped_decode_shortcut &&
               shouldUseSharedExpertGroupedDecode(params_.device_id);
    }

    bool SharedExpertFFNStage::usesGroupedDecodeForTesting() const
    {
        return shouldUseGroupedDecodeRoute();
    }

    bool SharedExpertFFNStage::executeDecodeEquivalentVerifierPrefill(
        IDeviceContext *ctx, IMoEKernel *kernel, int d_model, int intermediate)
    {
        if (!kernel || !params_.input || !params_.output ||
            !cached_gate_gemm_ || !cached_up_gemm_ || !cached_down_gemm_)
        {
            return false;
        }
        if (params_.device_id.is_gpu() && isGraphCaptureActive())
        {
            LOG_ERROR("[SharedExpertFFNStage] Decode-equivalent shared verifier prefill is not graph-capturable yet; "
                      "multi-row publication requires a device-resident row copy primitive before capture");
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "moe_decode_equivalent_verifier_prefill_runs",
            1.0,
            "verifier",
            params_.device_id.toString(),
            {{"stage", "shared_expert"},
             {"route", params_.seq_len == 1
                           ? "single_row_decode"
                           : ((params_.device_id.is_cuda() ||
                               params_.device_id.is_rocm())
                                  ? "gemm_grouped_verifier_hooks"
                                  : "cpu_grouped_verifier_hooks")},
             {"seq_len", std::to_string(params_.seq_len)}});

        if (params_.seq_len == 1)
        {
            /*
             * M=1 has no grouped dimension to exploit. Execute the ordinary
             * production one-row GEMM decode contract once.  Verifier
             * publication uses this row as the byte-for-byte oracle for grouped
             * M=2..4 rows, so the ordinary shared-expert grouped-table shortcut
             * is deliberately suppressed here even when normal decode would use
             * it.  Otherwise the grouped side and the serial witness can choose
             * different reduction kernels for the same TP-sharded shared expert
             * projection, which creates tiny layer-0 drift that later flips MoE
             * routes.
             */
            struct ScopedRuntimeDecodeSharedExpert
            {
                SharedExpertFFNStage::Params &params;
                bool force_grouped_verifier_prefill_for_decode;
                bool force_decode_equivalent_verifier_prefill;
                bool disable_grouped_decode_shortcut;
                int seq_len;

                ~ScopedRuntimeDecodeSharedExpert()
                {
                    params.force_grouped_verifier_prefill_for_decode =
                        force_grouped_verifier_prefill_for_decode;
                    params.force_decode_equivalent_verifier_prefill =
                        force_decode_equivalent_verifier_prefill;
                    params.disable_grouped_decode_shortcut =
                        disable_grouped_decode_shortcut;
                    params.seq_len = seq_len;
                }
            } restore{
                params_,
                params_.force_grouped_verifier_prefill_for_decode,
                params_.force_decode_equivalent_verifier_prefill,
                params_.disable_grouped_decode_shortcut,
                params_.seq_len};

            params_.seq_len = 1;
            params_.force_decode_equivalent_verifier_prefill = false;
            params_.force_grouped_verifier_prefill_for_decode = false;
            params_.disable_grouped_decode_shortcut = true;
            return execute(ctx);
        }

        if (!validatePlannedScratch(params_.seq_len, intermediate))
            return false;

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_, intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_, intermediate, nullptr, "shared_up"}};
        auto verifier_scopes = beginVerifierDecodeEquivalentScopes(
            {cached_gate_gemm_, cached_up_gemm_, cached_down_gemm_});
        if (!cached_gate_gemm_->multiply_fused_verifier_rows_decode_equivalent(
                params_.input,
                projections,
                params_.seq_len,
                d_model,
                nullptr,
                getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Grouped verifier shared gate/up projection failed"
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }
        if (params_.device_id.is_gpu())
        {
            const StageGPUExecution execution = gpuExecution();
            for (const auto &projection : projections)
                execution.publish(projection.output);
        }

        if (!cached_down_gemm_->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                scratch_gate_,
                scratch_up_,
                params_.output,
                params_.seq_len,
                d_model,
                intermediate,
                1.0f,
                0.0f,
                getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Grouped verifier shared SwiGLU/down projection failed"
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }
        if (params_.device_id.is_gpu())
            gpuExecution().publish(params_.output);

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_shared_grouped_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(params_.seq_len),
            "verifier",
             params_.device_id.toString(),
            {{"stage", "shared_expert"},
             {"route", (params_.device_id.is_cuda() ||
                         params_.device_id.is_rocm())
                           ? "gemm_grouped_verifier_hooks"
                           : "cpu_grouped_verifier_hooks"}});
        return true;
    }

    bool SharedExpertFFNStage::tryGroupedVerifierPrefill(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        (void)kernel;
        if (!shouldUseGroupedVerifierPrefillRoute())
            return false;

        if (params_.device_id.is_gpu())
            (void)requireGPUStream();

        /*
         * The serial CUDA shared-expert decode route uses the MoE grouped table
         * kernels even though there is only one always-active shared expert.
         * Verifier rows therefore need to be grouped through the same table
         * descriptor family, reduction order, and split-K partial layout rather
         * than through the generic dense GEMM verifier hooks.  This keeps the
         * implementation genuinely grouped for M=2..4 while making its math
         * target the exact path that ordinary one-token decode exercises.
         */
        if (!ensureSharedGroupedGateUpDescriptorTable(kernel, d_model, intermediate) ||
            !ensureSharedGroupedDownDescriptorTable(kernel, d_model, intermediate))
        {
            LOG_ERROR("[SharedExpertFFNStage] Failed to prepare shared expert grouped descriptor tables"
                      << " device=" << params_.device_id.to_string()
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }
        if (!kernel->prepareSharedExpertPrefillGroup(
                params_.active_row_count_device
                    ? DeviceRowRange::deviceCounted(params_.seq_len, params_.active_row_count_device)
                    : DeviceRowRange::fullyActive(params_.seq_len),
                MoESharedExpertGroupKind::DecodeEquivalentVerifier))
        {
            LOG_ERROR("[SharedExpertFFNStage] Failed to prepare shared expert verifier group"
                      << " device=" << params_.device_id.to_string()
                      << " m=" << params_.seq_len);
            return false;
        }
        if (!kernel->executeGroupedPrefillPipeline(
                params_.input,
                params_.output,
                shared_grouped_gateup_desc_table_id_,
                shared_grouped_down_desc_table_id_,
                params_.seq_len,
                d_model,
                intermediate,
                /*num_experts=*/1,
                /*top_k=*/1))
        {
            LOG_ERROR("[SharedExpertFFNStage] Shared expert grouped verifier pipeline failed"
                      << " device=" << params_.device_id.to_string()
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_shared_grouped_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(params_.seq_len),
            "verifier",
            params_.device_id.toString(),
            {{"stage", "shared_expert"},
             {"route", "grouped_table_prefill"}});
        return true;
    }

    bool SharedExpertFFNStage::tryGroupedDecode(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        if (!params_.device_id.is_gpu() || params_.seq_len != 1 ||
            !shouldUseSharedExpertGroupedDecode(params_.device_id))
        {
            return false;
        }

        if (!ensureSharedGroupedGateUpDescriptorTable(kernel, d_model, intermediate) ||
            !ensureSharedGroupedDownDescriptorTable(kernel, d_model, intermediate))
        {
            return false;
        }

        constexpr int expert_id = 0;
        constexpr float expert_weight = 1.0f;
        ITensor *gate_outputs[1] = {scratch_gate_};
        ITensor *up_outputs[1] = {scratch_up_};
        if (!kernel->groupedExpertGateUpDecodeFromTable(
                params_.input,
                &expert_id,
                shared_grouped_gateup_desc_table_id_,
                1,
                gate_outputs,
                up_outputs,
                d_model,
                intermediate))
        {
            return false;
        }

        ITensor *gate_tensors[1] = {scratch_gate_};
        ITensor *up_tensors[1] = {scratch_up_};
        const bool ok = kernel->groupedExpertDownDecodeFromTable(
            gate_tensors,
            up_tensors,
            &expert_id,
            &expert_weight,
            shared_grouped_down_desc_table_id_,
            1,
            params_.output,
            d_model,
            intermediate);
        if (ok)
            grouped_decode_launch_state_prepared_ = true;
        return ok;
    }

    bool SharedExpertFFNStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[SharedExpertFFNStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.gate_w || !params_.up_w || !params_.down_w || !params_.output)
        {
            LOG_ERROR("[SharedExpertFFNStage] Null tensor parameter");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int intermediate = params_.intermediate;

        // Cache GEMM engines on first call
        ensureGemmEnginesCached();
        if (!cached_gate_gemm_ || !cached_up_gemm_ || !cached_down_gemm_)
        {
            LOG_ERROR("[SharedExpertFFNStage] Missing shared expert GEMM engine");
            return false;
        }
        bindStageStream(cached_gate_gemm_);
        bindStageStream(cached_up_gemm_);
        bindStageStream(cached_down_gemm_);

        if (!validatePlannedScratch(seq_len, intermediate))
            return false;

        IMoEKernel *kernel = ensureMoEKernel();
        if (shouldUseDecodeEquivalentVerifierPrefill())
        {
            if (!executeDecodeEquivalentVerifierPrefill(ctx, kernel, d_model, intermediate))
            {
                LOG_ERROR("[SharedExpertFFNStage] Decode-equivalent verifier shared expert path failed");
                return false;
            }
            return true;
        }

        if (shouldUseGroupedVerifierPrefillRoute())
        {
            if (!tryGroupedVerifierPrefill(kernel, d_model, intermediate))
            {
                LOG_ERROR("[SharedExpertFFNStage] Verifier grouped shared expert path failed");
                return false;
            }
            gpuExecution().publish(params_.output);
            return true;
        }

        const bool grouped_decode_required =
            shouldUseGroupedDecodeRoute();
        if (grouped_decode_required)
        {
            if (!tryGroupedDecode(kernel, d_model, intermediate))
            {
                LOG_ERROR("[SharedExpertFFNStage] Grouped shared expert decode path failed");
                return false;
            }
            gpuExecution().publish(params_.output);
            return true;
        }

        // Gate+Up projections via fused multi-projection (quantizes input once)
        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_, intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_, intermediate, nullptr, "shared_up"}};
        if (!cached_gate_gemm_->multiply_fused_tensor(
                params_.input, projections,
                seq_len, d_model,
                nullptr, getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Shared gate/up projection failed");
            return false;
        }
        if (params_.device_id.is_gpu())
        {
            const StageGPUExecution execution = gpuExecution();
            for (const auto &projection : projections)
                execution.publish(projection.output);
        }

        // Mandatory fused SwiGLU/down grouped implementation.
        if (!fusedSwigluDown(
                *this,
                scratch_gate_, scratch_up_, params_.output,
                cached_down_gemm_, seq_len, d_model, intermediate,
                getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Shared SwiGLU/down projection failed");
            return false;
        }

        return true;
    }

    IMoEKernel *SharedExpertFFNStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
        {
            owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        if (params_.device_id.is_gpu() && params_.required_router_q8_publication&&
            (shouldUseGroupedVerifierPrefillRoute()||params_.required_router_q8_publication->requiredOriginalM1Source()))
        {
            if (!moe_kernel_ ||
                !moe_kernel_->bindRouterQ8HiddenPublication(
                    params_.required_router_q8_publication,
                    MoERouterQ8PublicationAccess::RequiredConsumer))
            {
                LOG_ERROR("[SharedExpertFFNStage] Failed to bind the required "
                          "router Q8 publication on "
                          << params_.device_id.to_string());
                return nullptr;
            }
        }
        auto *kernel = bindStageStream(moe_kernel_);
        if (bound_workspace_)
        {
            // The shared expert can be the first MoE stage in focused tests
            // and verifier-only paths. Bind its private workspace explicitly
            // before any pointer array or descriptor table is materialized.
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel))
                consumer->bindWorkspace(bound_workspace_);
        }
        return kernel;
    }

    size_t SharedExpertFFNStage::estimatedFlops() const
    {
        return static_cast<size_t>(6) * params_.seq_len * params_.d_model * params_.intermediate;
    }

    bool SharedExpertFFNStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool SharedExpertFFNStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!params_.device_id.is_gpu())
            return true;
        if (!stream)
        {
            LOG_ERROR("[SharedExpertFFNStage] Graph launch preparation requires "
                      "the exact non-null producer stream");
            return false;
        }
        setGPUStream(stream);

        ensureGemmEnginesCached();
        if (!cached_gate_gemm_ || !cached_up_gemm_ || !cached_down_gemm_)
        {
            LOG_ERROR("[SharedExpertFFNStage] Persistent shared-expert GEMM "
                      "engines are unavailable before graph capture");
            return false;
        }
        bindStageStream(cached_gate_gemm_);
        bindStageStream(cached_up_gemm_);
        bindStageStream(cached_down_gemm_);
        if (!validatePlannedScratch(
                std::max(1, params_.seq_len), params_.intermediate))
        {
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
        {
            LOG_ERROR("[SharedExpertFFNStage] Backend MoE kernel is unavailable "
                      "after exact-stream binding"
                      << " device=" << params_.device_id.to_string()
                      << " seq_len=" << params_.seq_len);
            return false;
        }

        if (shouldUseGroupedVerifierPrefillRoute())
        {
            // Cold preparation owns fixed launch metadata only. The retained
            // verifier's controller has not published a live row count yet
            // (and its reused allocation may contain arbitrary prior bytes).
            // Captured tryGroupedVerifierPrefill() republishes this metadata
            // from the device-owned count before performing any model math.
            if (!ensureSharedGroupedGateUpDescriptorTable(
                    kernel, params_.d_model, params_.intermediate) ||
                !ensureSharedGroupedDownDescriptorTable(
                    kernel, params_.d_model, params_.intermediate) ||
                !kernel->prepareSharedExpertPrefillGroup(
                    DeviceRowRange::fullyActive(params_.seq_len),
                    MoESharedExpertGroupKind::DecodeEquivalentVerifier))
            {
                LOG_ERROR("[SharedExpertFFNStage] Failed to prepare the fixed "
                          "grouped verifier launch state before graph capture");
                return false;
            }
            return true;
        }

        if (!shouldUseGroupedDecodeRoute())
            return true;

        if (!ensureSharedGroupedGateUpDescriptorTable(
                kernel, params_.d_model, params_.intermediate) ||
            !ensureSharedGroupedDownDescriptorTable(
                kernel, params_.d_model, params_.intermediate))
        {
            LOG_ERROR("[SharedExpertFFNStage] Failed to prepare typed shared "
                      "expert descriptor tables before grouped-decode capture"
                      << " device=" << params_.device_id.to_string()
                      << " d_model=" << params_.d_model
                      << " intermediate=" << params_.intermediate);
            return false;
        }

        constexpr int expert_id = 0;
        constexpr float expert_weight = 1.0f;
        ITensor *gate_outputs[1] = {scratch_gate_};
        ITensor *up_outputs[1] = {scratch_up_};
        if (!kernel->prepareGroupedTableDecodeLaunchState(
                &expert_id,
                &expert_weight,
                shared_grouped_gateup_desc_table_id_,
                shared_grouped_down_desc_table_id_,
                /*num_active=*/1,
                gate_outputs,
                up_outputs,
                params_.output,
                params_.d_model,
                params_.intermediate))
        {
            LOG_ERROR("[SharedExpertFFNStage] Failed to prepare persistent "
                      "shared-expert grouped-decode pointer tables");
            grouped_decode_launch_state_prepared_ = false;
            return false;
        }

        if(params_.required_router_q8_publication&&params_.required_router_q8_publication->requiredOriginalM1Source()) {
            if(params_.seq_len!=1||params_.active_row_count_device||params_.phase!=MoEOverlayServicePhaseHint::Decode)
                throw std::invalid_argument("Required original hidden Q8 shared reader supports only fully-active ordinary M1 Decode");
            if(!kernel->prepareOriginalM1HiddenQ8Publication(dynamic_cast<const TensorBase*>(params_.input),DeviceRowRange::fullyActive(1)))
                throw std::logic_error("Required original hidden Q8 shared reader preparation failed");
        }
        grouped_decode_launch_state_prepared_ = true;
        return true;
    }

    bool SharedExpertFFNStage::isGraphCapturable() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (!params_.device_id.is_gpu())
            return false;

        /*
         * Forced verifier replay is a small-M grouped-prefill path for M=1..4.
         * Do not test grouped_decode_launch_state_prepared_ here: that flag is only written by
         * tryGroupedDecode(), while verifier replay deliberately executes
         * tryGroupedVerifierPrefill() and owns its own warmup contract.
         */
        if (shouldUseGroupedVerifierPrefillRoute())
        {
            if (!supportsGroupedPrefillGraphCaptureBackend(params_.device_id))
                return false;
            return scratch_seq_len_ >= params_.seq_len && moe_kernel_ != nullptr;
        }

        /*
         * Normal grouped decode bakes device-side pointer arrays into CUDA graph
         * replay.  A cold stage can be warmed by Phase 1, but Phase 2 capture
         * must not start until that warmup has populated arrays for this stage's
         * current scratch buffers and workspace binding.
         */
        if (params_.seq_len == 1)
        {
            if (shouldUseGroupedDecodeRoute())
            {
                if (!supportsSharedExpertGroupedDecodeGraphCaptureBackend(params_.device_id))
                    return false;
                return grouped_decode_launch_state_prepared_ && scratch_seq_len_ >= 1 && moe_kernel_ != nullptr;
            }
            return scratch_seq_len_ >= 1;
        }

        // Prefill capture is safe only after warmup has allocated scratch and
        // resolved the MoE kernel used by fused SwiGLU/down.
        if (!supportsGroupedPrefillGraphCaptureBackend(params_.device_id))
            return false;
        return scratch_seq_len_ >= params_.seq_len && moe_kernel_ != nullptr;
#endif
    }

    bool SharedExpertFFNStage::supportsGraphCaptureAfterLaunchPreparation() const
    {
        if (supportsPaddedPrefillGraphCapturePreflight())
            return true;

        return shouldUseGroupedDecodeRoute() &&
               supportsSharedExpertGroupedDecodeGraphCaptureBackend(params_.device_id) &&
               params_.d_model > 0 &&
               params_.intermediate > 0 &&
               params_.input &&
               params_.gate_w &&
               params_.up_w &&
               params_.down_w &&
               params_.output;
    }

    std::string SharedExpertFFNStage::graphCaptureReadinessDebugString() const
    {
        std::ostringstream out;
        const bool prefill_graph_backend =
            supportsGroupedPrefillGraphCaptureBackend(params_.device_id);
        const bool decode_graph_backend =
            supportsSharedExpertGroupedDecodeGraphCaptureBackend(params_.device_id);
        const bool verifier_route = shouldUseGroupedVerifierPrefillRoute();
        const bool decode_equivalent_route = shouldUseDecodeEquivalentVerifierPrefill();
        const bool grouped_decode_route = shouldUseGroupedDecodeRoute();

        out << "device=" << params_.device_id.toString()
            << " seq_len=" << params_.seq_len
            << " d_model=" << params_.d_model
            << " intermediate=" << params_.intermediate
            << " prefill_graph_backend=" << perfBool(prefill_graph_backend)
            << " decode_graph_backend=" << perfBool(decode_graph_backend)
            << " verifier_route=" << perfBool(verifier_route)
            << " router_q8_publication_bound="
            << perfBool(params_.required_router_q8_publication != nullptr)
            << " decode_equivalent_route=" << perfBool(decode_equivalent_route)
            << " grouped_decode_route=" << perfBool(grouped_decode_route)
            << " scratch_seq_len=" << scratch_seq_len_
            << " scratch_ready=" << perfBool(scratch_seq_len_ >= std::max(1, params_.seq_len))
            << " moe_kernel=" << perfBool(moe_kernel_ != nullptr)
            << " grouped_decode_warmed=" << perfBool(grouped_decode_launch_state_prepared_)
            << " workspace_bound=" << perfBool(bound_workspace_ != nullptr)
            << " gate_gemm=" << perfBool(cached_gate_gemm_ != nullptr)
            << " up_gemm=" << perfBool(cached_up_gemm_ != nullptr)
            << " down_gemm=" << perfBool(cached_down_gemm_ != nullptr)
            << " input=" << perfBool(params_.input != nullptr)
            << " gate_w=" << perfBool(params_.gate_w != nullptr)
            << " up_w=" << perfBool(params_.up_w != nullptr)
            << " down_w=" << perfBool(params_.down_w != nullptr)
            << " output=" << perfBool(params_.output != nullptr);
        return out.str();
    }

    bool SharedExpertFFNStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        /*
         * Physical M=1 is a valid exact-shape prefill bucket. This occurs when
         * prefix restore leaves one uncached token. The shared FFN has no
         * position-dependent arithmetic, so its prepared one-row production
         * kernel is equally valid in decode and prefill transactions; the
         * enclosing graph owns the lifecycle distinction.
         */
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               params_.seq_len >= 1 &&
               params_.d_model > 0 &&
               params_.intermediate > 0 &&
               params_.input &&
               params_.gate_w &&
               params_.up_w &&
               params_.down_w &&
               params_.output;
    }

    bool SharedExpertFFNStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        if (params_.force_grouped_verifier_prefill_for_decode)
        {
            return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
                   params_.seq_len >= 1 &&
                   params_.d_model > 0 &&
                   params_.intermediate > 0 &&
                   params_.input &&
                   params_.gate_w &&
                   params_.up_w &&
                   params_.down_w &&
                   params_.output;
        }
        return supportsLazyPrefillGraphCapturePreflight();
    }

    StageBufferRequirements SharedExpertFFNStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        if (params_.gate_scratch)
            reqs.addScratch(
                "gate_scratch",
                params_.gate_scratch->shape(),
                toBufferTensorType(params_.gate_scratch->native_type()));
        if (params_.up_scratch)
            reqs.addScratch(
                "up_scratch",
                params_.up_scratch->shape(),
                toBufferTensorType(params_.up_scratch->native_type()));
        return reqs;
    }

    StageBufferContract SharedExpertFFNStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();

        contract.addInput(params_.input_buffer_id);
        contract.addOutput(params_.output_buffer_id);
        if (params_.gate_scratch)
            contract.addOutput(params_.gate_scratch_buffer_id);
        if (params_.up_scratch)
            contract.addOutput(params_.up_scratch_buffer_id);

        // Shared-expert GEMMs consume store-owned prepared representations.
        if (params_.gate_w)
            contract.addPreparedWeight(
                params_.gate_w,
                params_.prepared_store,
                params_.prepared_ref_gate.value_or(PreparedWeightRef{}));
        if (params_.up_w)
            contract.addPreparedWeight(
                params_.up_w,
                params_.prepared_store,
                params_.prepared_ref_up.value_or(PreparedWeightRef{}));
        if (params_.down_w)
            contract.addPreparedWeight(
                params_.down_w,
                params_.prepared_store,
                params_.prepared_ref_down.value_or(PreparedWeightRef{}));

        return contract;
    }

    StageDumpInfo SharedExpertFFNStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("shared_output_role", static_cast<int>(params_.output_role));
        info.addScalarInt("shared_phase", static_cast<int>(params_.phase));
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_w)
            info.addWeight("gate_w", params_.gate_w);
        if (params_.up_w)
            info.addWeight("up_w", params_.up_w);
        if (params_.down_w)
            info.addWeight("down_w", params_.down_w);
        if (params_.output)
            info.addOutput(params_.output_role == SharedExpertOutputRole::ParticipantLocalPartial ? "shared_local_partial" : "output", params_.output, params_.seq_len, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("intermediate", params_.intermediate);
        return info;
    }

    // =========================================================================
    // SharedExpertFFNStage — IWorkspaceConsumer Implementation
    // =========================================================================

    WorkspaceRequirements SharedExpertFFNStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        auto *self = const_cast<SharedExpertFFNStage *>(this);
        self->ensureGemmEnginesCached();

        const int rows = std::max({1, m, params_.seq_len});
        const int d_model =
            params_.d_model > 0 ? params_.d_model : (n > 0 ? n : 0);
        const int intermediate =
            params_.intermediate > 0 ? params_.intermediate : (k > 0 ? k : 0);

        /*
         * Gate/up and down have transposed logical shapes:
         *
         *   gate/up: [rows, d_model]       x [intermediate, d_model]^T
         *   down:    [rows, intermediate]  x [d_model, intermediate]^T
         *
         * The workspace names are intentionally shared by the GEMM engines and the
         * manager keeps the maximum buffer per name.  Passing the down shape to
         * gate/up under-sized QUANT_A/TEMP_A for M=2/3 verifier serial replay
         * (rows * intermediate < 1 * d_model), which corrupted ROCm M=1 oracle
         * rows.  Request each projection with its true shape so both grouped and
         * row-wise decode-equivalent paths are covered by the declared workspace.
         */
        WorkspaceRequirements combined;
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
            combined.merge(c->getWorkspaceRequirements(rows, intermediate, d_model));
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_up_gemm_))
            combined.merge(c->getWorkspaceRequirements(rows, intermediate, d_model));
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_down_gemm_))
        {
            combined.merge(c->getWorkspaceRequirements(rows, d_model, intermediate));
            c->appendSwiGLUWorkspaceRequirements(combined, rows, intermediate);
        }
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
        {
            const std::array<int, 2> fused_columns = {
                intermediate, intermediate};
            c->appendFusedProjectionWorkspaceRequirements(
                combined, rows, fused_columns, d_model);
        }

        /*
         * M=1 decode may overlap shared gate/up projections on the fixed CUDA
         * stream pool. Grouped verifier execution has a different stable
         * contract: it quantizes all rows once and launches each projection
         * in-order on the graph stream, so its dedicated grouped KPAR buffer
         * makes this helper a no-op.
         */
        addCudaConcurrentDecodeGemvSideStreamWorkspace(
            combined,
            params_.device_id,
            rows,
            /*projection_count=*/2);

        /*
         * Grouped shared-expert decode and verifier prefill use IMoEKernel
         * scratch in addition to the three projection GEMM workspaces.  Declare
         * those buffers here so a standalone shared-expert stage cannot rely on
         * a sibling routed-MoE stage to happen to reserve pointer arrays,
         * metadata, and INT8 activation scratch for graph capture.
         */
        const bool may_use_grouped_moe =
            params_.device_id.is_gpu() &&
            (shouldUseGroupedDecodeRoute() || shouldUseGroupedVerifierPrefillRoute());
        if (may_use_grouped_moe)
        {
            const int workspace_seq_len = rows;
            if (params_.device_id.is_cuda())
            {
                combined.merge(MoEWorkspaceBuffers::cudaMoE(
                    workspace_seq_len,
                    params_.d_model,
                    params_.intermediate,
                    /*num_experts=*/1,
                    /*top_k=*/1));
            }
            else if (params_.device_id.is_rocm())
            {
                combined.merge(MoEWorkspaceBuffers::rocmMoE(
                    workspace_seq_len,
                    params_.d_model,
                    params_.intermediate,
                    /*num_experts=*/1,
                    /*top_k=*/1));
            }
        }
        return combined;
    }

    void SharedExpertFFNStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        const bool workspace_changed = bound_workspace_ != workspace;
        ensureGemmEnginesCached();

        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
            c->bindWorkspace(workspace);
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_up_gemm_))
            c->bindWorkspace(workspace);
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_down_gemm_))
            c->bindWorkspace(workspace);

        bound_workspace_ = workspace;
        if (workspace_changed)
            grouped_decode_launch_state_prepared_ = false;
        LOG_TRACE("[SharedExpertFFNStage] Bound workspace to gate/up/down GEMM engines");
    }

    void SharedExpertFFNStage::unbindWorkspace()
    {
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
            c->unbindWorkspace();
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_up_gemm_))
            c->unbindWorkspace();
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_down_gemm_))
            c->unbindWorkspace();

        bound_workspace_ = nullptr;
        grouped_decode_launch_state_prepared_ = false;
    }

    bool SharedExpertFFNStage::hasWorkspace() const
    {
        return bound_workspace_ != nullptr;
    }

    DeviceWorkspaceManager *SharedExpertFFNStage::getWorkspace() const
    {
        return bound_workspace_;
    }

    // =========================================================================
    // SharedExpertGateStage — Sigmoid gating on shared expert output
    // =========================================================================

    SharedExpertGateStage::SharedExpertGateStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        validateSharedPublication(params_.output_role, params_.phase, params_.device_id);
        if (params_.output_role == SharedExpertOutputRole::ParticipantLocalPartial &&
            (!params_.routed_residual || !params_.combined_output || params_.execution_role != ExecutionRole::ReplicatedOwner))
            throw std::invalid_argument("Local FFN gating requires both local producers and participant-owned fused publication");
        if (params_.active_row_count_device && !params_.device_id.is_gpu())
            throw std::invalid_argument(
                "[SharedExpertGateStage] Device-owned live rows require a GPU execution owner");
        if (!params_.gate_inp ||
            params_.gate_inp->native_type() == TensorType::FP32)
        {
            return;
        }

        throw std::invalid_argument(
            "[SharedExpertGateStage] Shared-expert input gate must be normalized "
            "to FP32 by model-weight preparation");
    }

    uint64_t SharedExpertGateStage::graphCaptureVariantSignature() const
    {
        auto signature = sharedPublicationSignature(params_.output_role, params_.phase);
        if (!signature) return 0;
        const auto mix = [&](uint64_t value) { signature = (signature ^ value) * kFNV1a64Prime; };
        mix(params_.seq_len); mix(params_.d_model); mix(static_cast<uint64_t>(params_.execution_role));
        mix(reinterpret_cast<uintptr_t>(params_.active_row_count_device));
        mix(static_cast<uint64_t>(params_.device_id.type)); mix(params_.device_id.ordinal);
        for (auto *tensor : {params_.input, params_.gate_inp, params_.shared_output, params_.routed_residual, params_.combined_output})
        {
            mix(reinterpret_cast<uintptr_t>(tensor));
            mix(reinterpret_cast<uintptr_t>(tensor ? tensor->gpu_data_ptr() : nullptr));
        }
        for (auto id : {params_.input_buffer_id, params_.output_buffer_id, params_.residual_buffer_id, params_.combined_output_buffer_id})
            mix(static_cast<uint64_t>(id));
        return signature ? signature : 1;
    }

    void SharedExpertGateStage::invalidateKernelDynamicState()
    {
        if (!owned_moe_kernel_)
            return;

        owned_moe_kernel_->resetDynamicState();
        owned_moe_kernel_->clearGPUStreamBinding();
    }

    bool SharedExpertGateStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
    {
        (void)ctx;
        if (params_.device_id.is_gpu() && !stream)
        {
            LOG_ERROR("[SharedExpertGateStage] Graph launch preparation requires "
                      "the exact non-null producer stream");
            return false;
        }
        if (stream)
            setGPUStream(stream);

        /* A rooted-reduction observer exists only to keep participant graphs
         * structurally symmetric. It owns no tensor and launches no kernel. */
        if (!ownsGateArithmetic())
            return true;

        if (params_.device_id.is_gpu())
        {
            TensorBase *gate_input = effectiveGateInput();
            if (!gate_input || !ensureMoEKernel())
                return false;
            if (!gateInputReadyForGraphCapture())
            {
                LOG_ERROR("[SharedExpertGateStage] Shared gate vector is not "
                          "resident on the graph device; model-weight preparation "
                          "must complete before graph launch preparation");
                return false;
            }
        }

        if (params_.active_row_count_device &&
            PerfStatsCollector::isDomainEnabled("moe"))
        {
            PerfStatsCollector::addCounter(
                "moe",
                "shared_gate_device_row_count_bound",
                1.0,
                "prefill",
                params_.device_id.toString(),
                PerfStatsCollector::Tags{
                    {"physical_rows", std::to_string(params_.seq_len)},
                    {"row_count_source", "device_request_geometry"}});
        }
        return true;
    }

    TensorBase *SharedExpertGateStage::effectiveGateInput() const
    {
        if (!params_.gate_inp)
            return nullptr;

        return params_.gate_inp->native_type() == TensorType::FP32
                   ? params_.gate_inp
                   : nullptr;
    }

    bool SharedExpertGateStage::gateInputReadyForGraphCapture() const
    {
        if (!params_.gate_inp)
            return false;

        const TensorBase *gate_input = effectiveGateInput();

        if (!gate_input)
            return false;

        /**
         * Graph replay may not perform a hidden H2D for model weights. Warmup is
         * responsible for materializing the FP32 gate vector and making the
         * selected tensor resident on the exact backend device used by this
         * stage; capture readiness is only true after that device-side handoff.
         */
        return !params_.device_id.is_gpu() || gate_input->is_on_device(params_.device_id);
    }

    bool SharedExpertGateStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[SharedExpertGateStage] Null device context");
            return false;
        }

        if (!ownsGateArithmetic())
            return true;

        if (!params_.input || !params_.gate_inp || !params_.shared_output)
        {
            LOG_ERROR("[SharedExpertGateStage] Null tensor parameter");
            return false;
        }

        const bool fused_combine = params_.routed_residual || params_.combined_output;
        if (fused_combine && (!params_.routed_residual || !params_.combined_output))
        {
            LOG_ERROR("[SharedExpertGateStage] Fused combine requires both routed_residual and combined_output");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;

        // Delegate sigmoid gating to device-appropriate MoE kernel.
        // Tensor-aware API handles CPU/GPU dispatch internally.
        TensorBase *gate_inp = effectiveGateInput();
        if (!gate_inp)
        {
            LOG_ERROR("[SharedExpertGateStage] Failed to prepare gate input");
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();
        // M=1 can be an empty retained transaction too. The stable device
        // scalar, rather than physical width, owns whether gate math is live.
        const int32_t *device_active_rows = params_.device_id.is_gpu()
            ? params_.active_row_count_device : nullptr;
        if (device_active_rows &&
            PerfStatsCollector::isDomainEnabled("moe"))
        {
            PerfStatsCollector::addCounter(
                "moe",
                "shared_gate_device_row_count_execute",
                1.0,
                "prefill",
                params_.device_id.toString(),
                PerfStatsCollector::Tags{
                    {"physical_rows", std::to_string(seq_len)},
                    {"row_count_source", "device_request_geometry"}});
        }

        if (fused_combine)
        {
            if (device_active_rows)
            {
                if (!kernel->sharedExpertGateAddFromTensorsEffectiveSeqLen(
                        params_.input, gate_inp, params_.shared_output,
                        params_.routed_residual, params_.combined_output,
                        seq_len, d_model,
                        device_active_rows))
                {
                    LOG_ERROR("[SharedExpertGateStage] Effective-length fused shared gate-add failed");
                    return false;
                }
            }
            else
            {
                kernel->sharedExpertGateAddFromTensors(
                    params_.input, gate_inp, params_.shared_output,
                    params_.routed_residual, params_.combined_output,
                    seq_len, d_model);
            }
            // Fused gate-add materializes both semantic outputs: the gated
            // shared contribution and the final routed+shared combined row.
            if (params_.device_id.is_gpu())
            {
                const StageGPUExecution execution = gpuExecution();
                execution.publish(params_.shared_output);
                execution.publish(params_.combined_output);
            }
            return true;
        }

        if (device_active_rows)
        {
            if (!kernel->sharedExpertGateFromTensorsEffectiveSeqLen(
                    params_.input, gate_inp, params_.shared_output,
                    seq_len, d_model,
                    device_active_rows))
            {
                LOG_ERROR("[SharedExpertGateStage] Effective-length shared gate failed");
                return false;
            }
        }
        else
        {
            kernel->sharedExpertGateFromTensors(
                params_.input, gate_inp, params_.shared_output,
                seq_len, d_model);
        }
        if (params_.device_id.is_gpu())
            gpuExecution().publish(params_.shared_output);

        return true;
    }

    IMoEKernel *SharedExpertGateStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
        {
            owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        auto *kernel = bindStageStream(moe_kernel_);
        if (bound_workspace_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel))
                consumer->bindWorkspace(bound_workspace_);
        }
        return kernel;
    }

    size_t SharedExpertGateStage::estimatedFlops() const
    {
        if (!ownsGateArithmetic())
            return 0u;
        // Dot product + sigmoid + elementwise multiply, plus one add in fused-combine mode.
        const size_t per_token = static_cast<size_t>(2 * params_.d_model + params_.d_model);
        const size_t fused_add = (params_.routed_residual && params_.combined_output)
                                     ? static_cast<size_t>(params_.d_model)
                                     : 0u;
        return static_cast<size_t>(params_.seq_len) * (per_token + fused_add);
    }

    bool SharedExpertGateStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool SharedExpertGateStage::isGraphCapturable() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (!ownsGateArithmetic())
            return supportsGroupedPrefillGraphCaptureBackend(params_.device_id);
        // Single kernel call (sigmoid gating): pure kernel launch after explicit
        // launch-state preparation.
        // Capturable for both decode (seq_len==1) and prefill (seq_len>1) on supported GPU backends
        // because the stage is just a kernel launch with stable device pointers.
        // MoE kernel and stable input placement must already be bound by
        // prepareGraphLaunch().
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               moe_kernel_ != nullptr &&
               gateInputReadyForGraphCapture();
#endif
    }

    bool SharedExpertGateStage::supportsGraphCaptureAfterLaunchPreparation() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (!ownsGateArithmetic())
            return supportsGroupedPrefillGraphCaptureBackend(params_.device_id);
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               params_.seq_len > 0 &&
               params_.d_model > 0 &&
               params_.input &&
               params_.gate_inp &&
               params_.shared_output &&
               ((!params_.routed_residual && !params_.combined_output) ||
                (params_.routed_residual && params_.combined_output));
#endif
    }

    bool SharedExpertGateStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        /*
         * Prefix restore may produce a one-row exact-shape prefill. Shared
         * gating is row-local, and its M=1 launch is the same stable device
         * kernel used by decode. The serving graph, not physical row count,
         * supplies the prefill lifecycle identity.
         */
        if (!ownsGateArithmetic())
        {
            return supportsGroupedPrefillGraphCaptureBackend(
                       params_.device_id) &&
                   params_.seq_len >= 1;
        }
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               params_.seq_len >= 1 &&
               params_.d_model > 0 &&
               params_.input &&
               params_.gate_inp &&
               params_.shared_output &&
               ((!params_.routed_residual && !params_.combined_output) ||
                (params_.routed_residual && params_.combined_output));
    }

    bool SharedExpertGateStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        if (!ownsGateArithmetic())
            return supportsLazyPrefillGraphCapturePreflight();
        return supportsLazyPrefillGraphCapturePreflight() &&
               params_.active_row_count_device != nullptr;
    }

    bool SharedExpertGateStage::supportsPaddedPrefillRealLengthContract() const
    {
        return supportsPaddedPrefillGraphCapturePreflight();
    }

    WorkspaceRequirements SharedExpertGateStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        return WorkspaceRequirements{};
    }

    void SharedExpertGateStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
    }

    void SharedExpertGateStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
    }

    bool SharedExpertGateStage::hasWorkspace() const
    {
        return bound_workspace_ != nullptr;
    }

    DeviceWorkspaceManager *SharedExpertGateStage::getWorkspace() const
    {
        return bound_workspace_;
    }

    StageBufferRequirements SharedExpertGateStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (!ownsGateArithmetic())
            return reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.shared_output)
        {
            if (params_.routed_residual && params_.combined_output)
            {
                reqs.addInput("shared_output", params_.shared_output->shape(), toBufferTensorType(params_.shared_output->native_type()));
                reqs.addOutput("shared_output", params_.shared_output->shape(), toBufferTensorType(params_.shared_output->native_type()));
            }
            else
                reqs.addOutput("shared_output", params_.shared_output->shape(), toBufferTensorType(params_.shared_output->native_type()));
        }
        if (params_.routed_residual)
            reqs.addInput("routed_residual", params_.routed_residual->shape(), toBufferTensorType(params_.routed_residual->native_type()));
        if (params_.combined_output)
            reqs.addOutput("combined_output", params_.combined_output->shape(), toBufferTensorType(params_.combined_output->native_type()));
        return reqs;
    }

    StageBufferContract SharedExpertGateStage::bufferContract() const
    {
        if (!ownsGateArithmetic())
            return {};
        auto contract = StageBufferContract::build();

        contract.addInput(params_.input_buffer_id);
        if (params_.routed_residual && params_.combined_output)
        {
            contract.addInOut(params_.output_buffer_id);
            contract.addInput(params_.residual_buffer_id);
            contract.addOutput(params_.combined_output_buffer_id);
        }
        else
        {
            contract.addInOut(params_.output_buffer_id);
        }

        // Gate vector is a model weight, not arena-managed
        if (params_.gate_inp)
            contract.addWeight(params_.gate_inp);

        return contract;
    }

    StageDumpInfo SharedExpertGateStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const bool local = params_.output_role == SharedExpertOutputRole::ParticipantLocalPartial;
        info.addScalarInt("shared_output_role", static_cast<int>(params_.output_role));
        info.addScalarInt("shared_phase", static_cast<int>(params_.phase));
        if (!ownsGateArithmetic())
        {
            info.addScalarInt(
                "execution_role",
                static_cast<int>(params_.execution_role));
            return info;
        }
        info.addScalarInt(
            "execution_role",
            static_cast<int>(params_.execution_role));
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_inp)
            info.addWeight("gate_inp", params_.gate_inp);
        if (params_.shared_output)
        {
            if (params_.routed_residual && params_.combined_output)
                info.addOutput(local ? "shared_local_partial" : "shared_output", params_.shared_output, params_.seq_len, params_.d_model);
            else
                info.addOutput(local ? "shared_local_partial" : "shared_output", params_.shared_output, params_.seq_len, params_.d_model);
        }
        if (params_.routed_residual)
        {
            /*
             * The fused epilogue reads but does not mutate the complete routed
             * row.  Publish that already-live value as a diagnostic output as
             * well as an execution input: post-graph snapshot capture copies
             * outputs only, and this stage is the sole root-owned observation
             * point when a heterogeneous overlay defers its routed broadcast
             * until the final shared+routed publication.  This declaration
             * adds no production copy or buffer; it only authenticates the
             * existing pointer for the diagnostic manifest.
             */
            info.addInput("routed_residual", params_.routed_residual, params_.seq_len, params_.d_model);
            info.addOutput(local ? "routed_local_partial" : "routed_output", params_.routed_residual, params_.seq_len, params_.d_model);
        }
        if (params_.combined_output)
            info.addOutput(local ? "ffn_local_partial" : "combined_output", params_.combined_output, params_.seq_len, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        return info;
    }

    // =========================================================================
    // MoECanonicalRouteReduceStage
    // =========================================================================

    MoECanonicalRouteReduceStage::MoECanonicalRouteReduceStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
        if (params_.canonical_route_layout ==
                MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows &&
            params_.seq_len > 0 && params_.top_k > 0)
        {
            packed_record_for_route_slot_.resize(
                static_cast<size_t>(params_.seq_len) *
                static_cast<size_t>(params_.top_k),
                -1);
        }

        /*
         * The route scratch is shared serially by every MoE layer, so a host
         * read after the complete forward can observe only the last layer.
         * Bind a pure-device view at graph construction instead. Diagnostic
         * capture then publishes each layer's authenticated ledger at the
         * reducer boundary, on the same stream/event edge that consumed it.
         */
        const std::uint64_t route_slots =
            static_cast<std::uint64_t>(std::max(0, params_.seq_len)) *
            static_cast<std::uint64_t>(std::max(0, params_.top_k));
        if (params_.device_id.is_gpu() &&
            params_.reduction_role ==
                MoECanonicalRouteReductionRole::RootOwner &&
            route_slots > 0u &&
            route_slots <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::uint32_t>::max()) &&
            params_.domain_route_assignment.validFor(
                static_cast<std::uint32_t>(route_slots)))
        {
            /* Global placement and domain scheduling are two projections of
             * one pinned epoch. The shared view type keeps their geometry and
             * selected-bank semantics identical at every producer boundary. */
            pinned_route_evidence_views_ =
                std::make_unique<MoEOverlayPinnedRouteEvidenceViews>(
                    MoEOverlayPinnedRouteEvidenceViews::Params{
                        .device = params_.device_id,
                        .domain_route_assignment =
                            params_.domain_route_assignment,
                        .runtime_route_weights =
                            params_.runtime_route_weights,
                        .overlay_route_placement =
                            params_.overlay_route_placement,
                        .physical_rows =
                            static_cast<std::uint32_t>(params_.seq_len),
                        .top_k =
                            static_cast<std::uint32_t>(params_.top_k),
                    });
        }
    }

    void MoECanonicalRouteReduceStage::bindGatedSharedResidual(
        TensorBase *shared_output,
        TensorBase *combined_output,
        BufferId shared_id,
        BufferId combined_id)
    {
        if (!shared_output || !combined_output || !params_.node_local_route_exchange ||
            params_.seq_len != 1 || params_.top_k > 32)
        {
            throw std::invalid_argument(
                "MoECanonicalRouteReduceStage::bindGatedSharedResidual requires "
                "single-row node-local mapped route reduction with valid shared/combined tensors");
        }
        params_.gated_shared_residual = shared_output;
        params_.gated_shared_residual_buffer_id = shared_id;
        params_.output = combined_output;
        params_.output_buffer_id = combined_id;
    }

    namespace
    {
        /**
         * @brief Return whether this graph participant owns ordered reduction.
         *
         * Callers must validate that the role is not `Unspecified` before
         * consulting this helper. The helper deliberately contains no
         * topology arithmetic: graph construction already made the ownership
         * decision and the stage executes that immutable policy.
         */
        bool ownsCanonicalRouteReduction(
            MoECanonicalRouteReductionRole role) noexcept
        {
            return role == MoECanonicalRouteReductionRole::RootOwner ||
                   role ==
                       MoECanonicalRouteReductionRole::AllreduceParticipant;
        }

        /**
         * @brief Validate the graph-bound final route publication.
         *
         * The stable pointer/capacity prove setup binding. Stream-ordered graph
         * dependencies from route publication through expert completion prove
         * that the current invocation's contents are ready for this reducer.
         */
        bool hasCompleteNodeLocalRouteAssignment(
            const MoECanonicalRouteReduceStage::Params &params) noexcept
        {
            const std::uint64_t slots =
                static_cast<std::uint64_t>(std::max(0, params.seq_len)) *
                static_cast<std::uint64_t>(std::max(0, params.top_k));
            return slots > 0u &&
                   slots <=
                       static_cast<std::uint64_t>(
                           std::numeric_limits<std::uint32_t>::max()) &&
                   params.domain_route_assignment.validFor(
                       static_cast<std::uint32_t>(slots));
        }

        /**
         * @brief Upload immutable root-side mapped-lane descriptors once.
         *
         * INT32Tensor is only the aligned RAII storage owner. The bytes retain
         * their strongly typed ABI at construction and launch, and preparation
         * happens before native capture on the stage's exact stream.
         */
        bool uploadNodeLocalRoutePeerBindings(
            const std::vector<MoENodeLocalRoutePeerDeviceBinding> &bindings,
            DeviceId device,
            void *stream,
            std::shared_ptr<MoEOverlayPersistentGraphStorage> *storage)
        {
            static_assert(std::is_trivially_copyable_v<
                          MoENodeLocalRoutePeerDeviceBinding>);
            static_assert(
                sizeof(MoENodeLocalRoutePeerDeviceBinding) %
                    sizeof(std::int32_t) ==
                0u);
            if (!storage || bindings.empty() || !device.is_gpu() || !stream)
                return false;
            try
            {
                const std::size_t bytes =
                    bindings.size() *
                    sizeof(MoENodeLocalRoutePeerDeviceBinding);
                const std::size_t words =
                    bytes / sizeof(std::int32_t);
                if (!*storage)
                {
                    *storage = std::make_shared<
                        MoEOverlayPersistentGraphStorage>(
                        MoEOverlayPersistentGraphStorage::Config{
                            .device = device,
                            .type = MoEOverlayGraphStorageType::Int32,
                            .shape = {words},
                            .immutable_input = true,
                            .identity =
                                "node_local_route_peer_bindings",
                        });
                }
                return (*storage)->sizeBytes() == bytes &&
                       (*storage)->publishImmutableBytes(
                           bindings.data(), bytes, stream);
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[MoECanonicalRouteReduceStage] Failed to prepare "
                    "node-local peer bindings: "
                    << error.what());
                return false;
            }
        }

    } // namespace

    bool MoECanonicalRouteReduceStage::execute(IDeviceContext *ctx)
    {
        const bool uses_node_local_exchange =
            params_.node_local_route_exchange != nullptr;
        if (!validMoENodeLocalRouteOwnership(params_.node_local_route_ownership) ||
            (params_.node_local_route_ownership == MoENodeLocalRouteOwnership::ContinuationCohort &&
             (!uses_node_local_exchange ||
              params_.external_route_source != MoEExternalCanonicalRouteSource::RootCanonicalRouteBank)))
        {
            LOG_ERROR("[MoECanonicalRouteReduceStage] Cohort reduction requires the original mapped root bank");
            return false;
        }
        if (!ctx ||
            !params_.canonical_route_contributions ||
            !params_.output ||
            params_.seq_len <= 0 ||
            params_.top_k <= 0 ||
            params_.d_model <= 0 ||
            params_.canonical_route_arithmetic ==
                MoECanonicalRouteArithmeticPolicy::Unspecified ||
            params_.canonical_route_layout ==
                MoECanonicalRoutePublicationLayout::Unspecified ||
            params_.reduction_role ==
                MoECanonicalRouteReductionRole::Unspecified ||
            (uses_node_local_exchange &&
             (!hasCompleteNodeLocalRouteAssignment(params_) ||
              params_.external_route_source ==
                  MoEExternalCanonicalRouteSource::Unspecified ||
              params_.route_participant_id < 0 ||
              !params_.node_local_route_exchange->materialized())))
        {
            LOG_ERROR("[MoECanonicalRouteReduceStage] Invalid execution contract"
                      << " ctx=" << (ctx != nullptr)
                      << " contributions="
                      << (params_.canonical_route_contributions != nullptr)
                      << " output=" << (params_.output != nullptr)
                      << " seq_len=" << params_.seq_len
                      << " top_k=" << params_.top_k
                      << " d_model=" << params_.d_model
                      << " arithmetic="
                      << moeCanonicalRouteArithmeticPolicyToString(
                             params_.canonical_route_arithmetic)
                      << " layout="
                      << moeCanonicalRoutePublicationLayoutToString(
                             params_.canonical_route_layout)
                      << " reduction_role="
                      << static_cast<int>(params_.reduction_role)
                      << " node_local_exchange="
                      << uses_node_local_exchange
                      << " domain_route_assignment_bound="
                      << (params_.domain_route_assignment.participant_ids !=
                          nullptr)
                      << " domain_route_assignment_capacity="
                      << params_.domain_route_assignment.capacity
                      << " domain_route_assignment_complete="
                      << hasCompleteNodeLocalRouteAssignment(params_)
                      << " external_route_source="
                      << static_cast<std::uint32_t>(
                             params_.external_route_source)
                      << " route_participant_id="
                      << params_.route_participant_id);
            return false;
        }

        /*
         * The preceding rooted collective leaves the canonical route slots
         * valid only on the fixed root. Every graph keeps this node so the DAG
         * remains symmetric, but non-root participants must not read those
         * slots or launch redundant arithmetic before the compact broadcast.
         */
        if (!uses_node_local_exchange &&
            !ownsCanonicalRouteReduction(params_.reduction_role))
            return true;

        if (params_.device_id.is_cpu())
        {
            const bool dense_allreduce_slots =
                params_.canonical_route_layout ==
                    MoECanonicalRoutePublicationLayout::
                        DenseOriginalRouteSlots;
            const bool packed_rooted_rows =
                params_.canonical_route_layout ==
                    MoECanonicalRoutePublicationLayout::
                        PackedIndexedRouteRows;
            if ((!dense_allreduce_slots && !packed_rooted_rows) ||
                (dense_allreduce_slots &&
                 params_.reduction_role !=
                     MoECanonicalRouteReductionRole::
                         AllreduceParticipant) ||
                (packed_rooted_rows &&
                 params_.reduction_role !=
                     MoECanonicalRouteReductionRole::RootOwner) ||
                params_.canonical_route_arithmetic !=
                    MoECanonicalRouteArithmeticPolicy::
                        UnweightedExpertRowThenOrderedFMA ||
                !params_.routing_weights)
            {
                LOG_ERROR(
                    "[MoECanonicalRouteReduceStage] CPU canonical reduction "
                    "received incompatible ownership, arithmetic, or layout"
                    << " role=" << static_cast<int>(params_.reduction_role)
                    << " arithmetic="
                    << moeCanonicalRouteArithmeticPolicyToString(
                           params_.canonical_route_arithmetic)
                    << " layout="
                    << moeCanonicalRoutePublicationLayoutToString(
                           params_.canonical_route_layout)
                    << " routing_weights="
                    << (params_.routing_weights != nullptr));
                return false;
            }

            const float *route_rows =
                params_.canonical_route_contributions->data();
            const float *route_weights = params_.routing_weights->data();
            float *output = params_.output->mutable_data();
            if (!route_rows || !route_weights || !output)
            {
                LOG_ERROR(
                    "[MoECanonicalRouteReduceStage] CPU canonical reduction "
                    "could not access host tensors");
                return false;
            }

            const size_t route_slot_count =
                static_cast<size_t>(params_.seq_len) *
                static_cast<size_t>(params_.top_k);
            const bool publication_timing_enabled =
                PerfStatsCollector::isDomainEnabled(
                    "moe_canonical_publication");
            const auto publication_start = publication_timing_enabled
                                               ? PerfStatsCollector::Clock::now()
                                               : PerfStatsCollector::Clock::time_point{};
            size_t gathered_record_count = 0u;
            if (packed_rooted_rows)
            {
                const size_t record_capacity =
                    canonical_moe_route_record::recordCapacity(
                        params_.canonical_route_contributions->numel(),
                        params_.d_model);
                gathered_record_count =
                    canonical_moe_route_record::readRecordCount(
                        route_rows,
                        params_.canonical_route_contributions->numel());
                if (record_capacity < route_slot_count ||
                    gathered_record_count > route_slot_count ||
                    packed_record_for_route_slot_.size() != route_slot_count)
                {
                    LOG_ERROR(
                        "[MoECanonicalRouteReduceStage] Packed CPU route "
                        "publication has invalid capacity/count"
                        << " capacity=" << record_capacity
                        << " gathered=" << gathered_record_count
                        << " route_slots=" << route_slot_count
                        << " lookup="
                        << packed_record_for_route_slot_.size());
                    return false;
                }

                std::fill(
                    packed_record_for_route_slot_.begin(),
                    packed_record_for_route_slot_.end(),
                    -1);
                size_t expected_records = 0u;
                for (size_t record_index = 0;
                     record_index < gathered_record_count;
                     ++record_index)
                {
                    const float *const route_record =
                        canonical_moe_route_record::record(
                            route_rows,
                            record_index,
                            params_.d_model);
                    const size_t flat_slot = static_cast<size_t>(
                        canonical_moe_route_record::readFlatRouteSlot(
                            route_record,
                            params_.d_model));
                    if (flat_slot >= route_slot_count ||
                        route_weights[flat_slot] == 0.0f ||
                        packed_record_for_route_slot_[flat_slot] >= 0)
                    {
                        LOG_ERROR(
                            "[MoECanonicalRouteReduceStage] Packed CPU route "
                            "record has an invalid, inert, or duplicate slot"
                            << " record=" << record_index
                            << " slot=" << flat_slot
                            << " route_slots=" << route_slot_count
                            << " weight="
                            << (flat_slot < route_slot_count
                                    ? route_weights[flat_slot]
                                    : 0.0f));
                        return false;
                    }
                    packed_record_for_route_slot_[flat_slot] =
                        static_cast<int32_t>(record_index);
                }
                for (size_t flat_slot = 0;
                     flat_slot < route_slot_count;
                     ++flat_slot)
                {
                    if (route_weights[flat_slot] == 0.0f)
                        continue;
                    ++expected_records;
                    if (packed_record_for_route_slot_[flat_slot] < 0)
                    {
                        LOG_ERROR(
                            "[MoECanonicalRouteReduceStage] Packed CPU route "
                            "publication is missing a live original slot"
                            << " slot=" << flat_slot
                            << " weight=" << route_weights[flat_slot]);
                        return false;
                    }
                }
                if (expected_records != gathered_record_count)
                {
                    LOG_ERROR(
                        "[MoECanonicalRouteReduceStage] Packed CPU route "
                        "record total does not match live router slots"
                        << " expected=" << expected_records
                        << " gathered=" << gathered_record_count);
                    return false;
                }
            }

            cpu::moe::CanonicalRouteFoldPlan fold_plan;
            if (!cpu::moe::reduceCanonicalRouteRowsOrderedFMA(
                    {
                        .route_rows = route_rows,
                        .route_weights = route_weights,
                        .record_for_route_slot = packed_rooted_rows
                                                     ? packed_record_for_route_slot_.data()
                                                     : nullptr,
                        .output = output,
                        .rows = params_.seq_len,
                        .top_k = params_.top_k,
                        .columns = params_.d_model,
                    },
                    &fold_plan))
            {
                LOG_ERROR(
                    "[MoECanonicalRouteReduceStage] CPU canonical ordered "
                    "route fold rejected a validated publication"
                    << " rows=" << params_.seq_len
                    << " top_k=" << params_.top_k
                    << " d_model=" << params_.d_model
                    << " packed=" << packed_rooted_rows);
                return false;
            }

            if (publication_timing_enabled)
            {
                const PerfStatsCollector::Tags publication_tags{
                    {"top_k", std::to_string(params_.top_k)},
                    {"d_model", std::to_string(params_.d_model)},
                    {"layout",
                     moeCanonicalRoutePublicationLayoutToString(
                         params_.canonical_route_layout)},
                    {"gathered_records",
                     std::to_string(gathered_record_count)},
                    {"executor",
                     cpu::moe::canonicalRouteFoldExecutorName(
                         fold_plan.executor)},
                    {"column_tile",
                     std::to_string(fold_plan.column_tile)},
                    {"parallel_tasks",
                     std::to_string(fold_plan.parallel_tasks)},
                    {"team_threads",
                     std::to_string(fold_plan.team_threads)}};
                PerfStatsCollector::addCounter(
                    "moe_canonical_publication",
                    packed_rooted_rows
                        ? "cpu_packed_route_root_ordered_fma_rows"
                        : "cpu_unweighted_route_slot_ordered_fma_rows",
                    static_cast<double>(params_.seq_len),
                    params_.seq_len == 1 ? "decode" : "prefill",
                    params_.device_id.to_string(),
                    publication_tags);
                const auto publication_finish =
                    PerfStatsCollector::Clock::now();
                const auto publication_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        publication_finish - publication_start)
                        .count();
                PerfStatsCollector::recordTimingNs(
                    "moe_canonical_publication",
                    packed_rooted_rows
                        ? "cpu_packed_route_root_ordered_fma"
                        : "cpu_dense_route_root_ordered_fma",
                    publication_ns > 0
                        ? static_cast<uint64_t>(publication_ns)
                        : 0u,
                    params_.seq_len == 1 ? "decode" : "prefill",
                    params_.device_id.to_string(),
                    publication_tags);
            }
            return true;
        }

        if (params_.canonical_route_arithmetic !=
                MoECanonicalRouteArithmeticPolicy::
                    PreweightedContributionThenOrderedAdd ||
            params_.canonical_route_layout !=
                MoECanonicalRoutePublicationLayout::
                    DenseOriginalRouteSlots)
        {
            LOG_ERROR(
                "[MoECanonicalRouteReduceStage] GPU canonical reduction "
                "requires dense preweighted route contributions"
                << " arithmetic="
                << moeCanonicalRouteArithmeticPolicyToString(
                       params_.canonical_route_arithmetic)
                << " layout="
                << moeCanonicalRoutePublicationLayoutToString(
                       params_.canonical_route_layout));
            return false;
        }

        if (!moe_kernel_)
        {
            owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        IMoEKernel *kernel = bindStageStream(moe_kernel_);
        if (!kernel)
        {
            LOG_ERROR(
                "[MoECanonicalRouteReduceStage] Device kernel is unavailable"
                << " device=" << params_.device_id.toString());
            return false;
        }

        if (uses_node_local_exchange)
        {
            const StageGPUExecution execution = gpuExecution();
            execution.requirePreparedInput(
                params_.canonical_route_contributions);
            const auto *const route_contributions =
                static_cast<const float *>(
                    params_.canonical_route_contributions->gpu_data_ptr());
            const std::uint64_t live_route_slots =
                static_cast<std::uint64_t>(params_.seq_len) *
                static_cast<std::uint64_t>(params_.top_k);
            if (!route_contributions || live_route_slots == 0u ||
                live_route_slots >
                    std::numeric_limits<std::uint32_t>::max())
            {
                LOG_ERROR(
                    "[MoECanonicalRouteReduceStage] Sparse node-local route "
                    "publication has invalid device input or geometry");
                return false;
            }
            const MoEDomainRouteAssignmentLedger domain_assignment =
                params_.domain_route_assignment;
            if (!domain_assignment.validFor(
                    static_cast<std::uint32_t>(live_route_slots)))
            {
                LOG_ERROR(
                    "[MoECanonicalRouteReduceStage] Sparse route assignment "
                    "did not resolve to a complete device binding");
                return false;
            }
            const MoEKernelLaunchContext launch{
                .stream = execution.nativeStream(),
            };
                if (params_.reduction_role ==
                    MoECanonicalRouteReductionRole::RootOwner)
                {
                    const bool stage_only =
                        params_.node_local_sparse_phase == MoENodeLocalSparsePhase::StageOnly;
                    const bool fold_staged =
                        params_.node_local_sparse_phase == MoENodeLocalSparsePhase::FoldStaged;
                    if (!stage_only)
                        execution.requirePreparedOutput(params_.output);
                    if (!node_local_peer_bindings_storage_ ||
                        !node_local_validation_storage_ ||
                        node_local_peer_count_ == 0u)
                    {
                        LOG_ERROR(
                            "[MoECanonicalRouteReduceStage] Root sparse route "
                            "consumer has no prepared peer descriptors/status");
                        return false;
                    }
                    const bool raw_cohort = params_.node_local_route_ownership ==
                        MoENodeLocalRouteOwnership::ContinuationCohort;
                    const auto &weights = params_.runtime_route_weights;
                    if (raw_cohort && !weights.validFor(static_cast<std::uint32_t>(params_.seq_len),
                                                          static_cast<std::uint32_t>(params_.top_k)))
                        throw std::logic_error("mixed-tier cohort lost its original router weight publication");
                    if (!stage_only && params_.gated_shared_residual)
                        execution.requirePreparedInput(params_.gated_shared_residual);
                    const float *gated_shared_ptr =
                        (!stage_only && params_.gated_shared_residual)
                            ? static_cast<const float *>(params_.gated_shared_residual->gpu_data_ptr())
                            : nullptr;
                    if (!stage_only && params_.gated_shared_residual && !gated_shared_ptr)
                    {
                        LOG_ERROR(
                            "[MoECanonicalRouteReduceStage] Root fused gated shared "
                            "residual has no device allocation");
                        return false;
                    }
                    MoENodeLocalRouteConsumeLaunch consumption{
                        .peers = static_cast<
                            const MoENodeLocalRoutePeerDeviceBinding *>(
                            node_local_peer_bindings_storage_->deviceData()),
                        .peer_count = node_local_peer_count_,
                        .root_canonical_route_contributions =
                            route_contributions,
                        .domain_assignment = domain_assignment,
                        .external_route_source =
                            params_.external_route_source,
                        .dense_output = static_cast<float *>(
                            params_.output->gpu_data_ptr()),
                        .validation_status = static_cast<std::int32_t *>(
                            node_local_validation_storage_->deviceData()),
                        .root_participant = params_.route_participant_id,
                        .physical_rows =
                            static_cast<std::uint32_t>(params_.seq_len),
                        .top_k = static_cast<std::uint32_t>(params_.top_k),
                        .d_model = static_cast<std::uint32_t>(params_.d_model),
                        .ownership = params_.node_local_route_ownership,
                        .cohort_route_weights = raw_cohort ? weights.weights : nullptr,
                        .cohort_route_capacity = raw_cohort ? weights.capacity : 0u,
                        .gated_shared_residual = gated_shared_ptr,
                        .payload_precision = params_.node_local_route_precision,
                    };
                    if (!stage_only && !fold_staged &&
                        consumption.physical_rows == 1u &&
                        consumption.top_k <= 32u &&
                        consumption.peer_count <= 8u)
                    {
                        if (!kernel->consumeSingleRowNodeLocalCanonicalRoutes(
                                launch, consumption))
                        {
                            LOG_ERROR(
                                "[MoECanonicalRouteReduceStage] Root fused single-row "
                                "sparse route consume failed");
                            return false;
                        }
                        execution.publish(params_.output);
                        return true;
                    }
                    if (!stage_only && params_.gated_shared_residual)
                    {
                        LOG_ERROR(
                            "[MoECanonicalRouteReduceStage] Fused gated shared residual "
                            "requires single-row consumeSingleRowNodeLocalCanonicalRoutes");
                        return false;
                    }
                    if (!fold_staged)
                    {
                        if (!kernel->acquireNodeLocalCanonicalRoutes(
                                launch, consumption))
                        {
                            LOG_ERROR(
                                "[MoECanonicalRouteReduceStage] Root sparse route "
                                "epoch acquisition failed");
                            return false;
                        }
                        /*
                         * The acquire kernel above is the device-owned producer edge.
                         * When all peers write sparse rows directly into Root VRAM via
                         * P2P, no Host RAM staging pass is required; otherwise pull
                         * only owner-selected mapped rows into stable root VRAM.
                         */
                        if (!params_.node_local_route_exchange->allPeersDirectP2P())
                        {
                            if (!kernel->stageNodeLocalCanonicalRoutes(
                                    launch, consumption))
                            {
                                LOG_ERROR(
                                    "[MoECanonicalRouteReduceStage] Root sparse route "
                                    "owner-selected payload staging failed");
                                return false;
                            }
                        }
                        if (stage_only)
                            return true;
                    }
                    if (!kernel->foldNodeLocalCanonicalRoutes(
                            launch, consumption))
                    {
                        LOG_ERROR(
                            "[MoECanonicalRouteReduceStage] Root sparse route "
                            "validation/fold failed");
                        return false;
                    }
                    execution.publish(params_.output);
                    return true;
                }

            MoENodeLocalRoutePublishLaunch publication{
                .lane = node_local_producer_binding_,
                .canonical_route_contributions = route_contributions,
                .domain_assignment = domain_assignment,
                .live_route_slots =
                    static_cast<std::uint32_t>(live_route_slots),
                .ownership = params_.node_local_route_ownership,
                .payload_precision = params_.node_local_route_precision,
            };
            if (!kernel->publishNodeLocalCanonicalRoutes(
                    launch, publication))
            {
                LOG_ERROR(
                    "[MoECanonicalRouteReduceStage] Peer sparse node-local "
                    "route publication failed");
                return false;
            }
            return true;
        }

        if (!kernel->reduceCanonicalRouteContributions(
                params_.canonical_route_contributions,
                params_.output,
                params_.seq_len,
                params_.top_k,
                params_.d_model))
        {
            LOG_ERROR("[MoECanonicalRouteReduceStage] Device reducer failed"
                      << " device=" << params_.device_id.to_string()
                      << " seq_len=" << params_.seq_len
                      << " top_k=" << params_.top_k
                      << " d_model=" << params_.d_model);
            return false;
        }

        const StageGPUExecution execution = gpuExecution();
        execution.publish(params_.output);
        return true;
    }

    uint64_t MoECanonicalRouteReduceStage::graphCaptureVariantSignature() const
    {
        return static_cast<uint64_t>(params_.node_local_route_ownership) |
               (params_.gated_shared_residual ? (uint64_t{1} << 8) : 0u) |
               (static_cast<uint64_t>(params_.node_local_sparse_phase) << 16) |
               (static_cast<uint64_t>(params_.node_local_route_precision) << 24);
    }

    size_t MoECanonicalRouteReduceStage::estimatedFlops() const
    {
        if (!ownsCanonicalRouteReduction(params_.reduction_role))
            return 0;
        return static_cast<size_t>(params_.seq_len) *
               static_cast<size_t>(params_.d_model) *
               static_cast<size_t>(std::max(0, params_.top_k - 1));
    }

    bool MoECanonicalRouteReduceStage::supportsBackend(
        ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool MoECanonicalRouteReduceStage::isGraphCapturable() const
    {
        if (!supportsGraphCaptureAfterLaunchPreparation())
            return false;
        if (params_.node_local_route_exchange)
            return moe_kernel_ != nullptr;
        return !ownsCanonicalRouteReduction(params_.reduction_role) ||
               moe_kernel_ != nullptr;
    }

    bool MoECanonicalRouteReduceStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR("[MoECanonicalRouteReduceStage] Graph launch preparation "
                      "requires the exact non-null producer stream");
            return false;
        }
        setGPUStream(stream);
        const bool uses_node_local_exchange =
            params_.node_local_route_exchange != nullptr;
        if (!uses_node_local_exchange &&
            !ownsCanonicalRouteReduction(params_.reduction_role))
            return true;

        if (!moe_kernel_)
        {
            owned_moe_kernel_ =
                KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        if (!bindStageStream(moe_kernel_))
            return false;

        if (!uses_node_local_exchange)
            return true;
        if (!hasCompleteNodeLocalRouteAssignment(params_) ||
            params_.external_route_source ==
                MoEExternalCanonicalRouteSource::Unspecified ||
            params_.route_participant_id < 0 ||
            !params_.node_local_route_exchange->materialized())
        {
            LOG_ERROR(
                "[MoECanonicalRouteReduceStage] Sparse route launch "
                "preparation requires a materialized fabric and stable "
                "typed device-owned route assignment");
            return false;
        }
        const bool is_root =
            params_.reduction_role ==
            MoECanonicalRouteReductionRole::RootOwner;
        if (is_root !=
                (params_.route_participant_id ==
                 params_.node_local_route_exchange->rootParticipant()) ||
            is_root !=
                (params_.device_id ==
                 params_.node_local_route_exchange->rootDevice()))
        {
            LOG_ERROR(
                "[MoECanonicalRouteReduceStage] Sparse route role disagrees "
                "with fabric root identity");
            return false;
        }

        if (is_root)
        {
            if (params_.node_local_sparse_shared_state &&
                params_.node_local_sparse_shared_state->peer_bindings_storage &&
                params_.node_local_sparse_shared_state->validation_storage &&
                params_.node_local_sparse_shared_state->peer_count > 0u)
            {
                node_local_peer_bindings_storage_ =
                    params_.node_local_sparse_shared_state->peer_bindings_storage;
                node_local_validation_storage_ =
                    params_.node_local_sparse_shared_state->validation_storage;
                node_local_peer_count_ =
                    params_.node_local_sparse_shared_state->peer_count;
                if (!node_local_peer_bindings_storage_->requireInput(stream) ||
                    !node_local_validation_storage_->requireOutput(stream))
                {
                    return false;
                }
                return true;
            }
            const auto bindings =
                params_.node_local_route_exchange->rootPeerBindings(
                    params_.device_id);
            if (bindings.empty())
            {
                return false;
            }
            node_local_peer_count_ =
                static_cast<std::uint32_t>(bindings.size());
            if (!node_local_peer_bindings_storage_)
            {
                if (!uploadNodeLocalRoutePeerBindings(
                        bindings,
                        params_.device_id,
                        stream,
                        &node_local_peer_bindings_storage_))
                {
                    return false;
                }
            }
            else
            {
                if (!node_local_peer_bindings_storage_->requireInput(stream))
                    return false;
            }
            if (!node_local_validation_storage_)
            {
                node_local_validation_storage_ = std::make_shared<
                    MoEOverlayPersistentGraphStorage>(
                    MoEOverlayPersistentGraphStorage::Config{
                        .device = params_.device_id,
                        .type = MoEOverlayGraphStorageType::Int32,
                        .shape = {1u},
                        .immutable_input = false,
                        .identity = "node_local_route_validation",
                    });
                if (!node_local_validation_storage_->requireOutput(stream))
                    return false;
            }
            else
            {
                if (!node_local_validation_storage_->requireOutput(stream))
                    return false;
            }
            if (params_.node_local_sparse_shared_state)
            {
                params_.node_local_sparse_shared_state->peer_bindings_storage =
                    node_local_peer_bindings_storage_;
                params_.node_local_sparse_shared_state->validation_storage =
                    node_local_validation_storage_;
                params_.node_local_sparse_shared_state->peer_count =
                    node_local_peer_count_;
            }
        }
        else
        {
            node_local_producer_binding_ =
                params_.node_local_route_exchange->producerBinding(
                    params_.device_id);
            if (!node_local_producer_binding_.valid() ||
                node_local_producer_binding_.producer_participant !=
                    params_.route_participant_id)
            {
                LOG_ERROR(
                    "[MoECanonicalRouteReduceStage] Producer mapped alias "
                    "does not match graph participant identity");
                return false;
            }
        }
        return true;
    }

    bool MoECanonicalRouteReduceStage::supportsGraphCaptureAfterLaunchPreparation() const
    {
        return params_.device_id.is_gpu() &&
               params_.canonical_route_contributions &&
               params_.output &&
               params_.seq_len > 0 &&
               params_.top_k > 0 &&
               params_.d_model > 0 &&
               (!params_.node_local_route_exchange ||
                (hasCompleteNodeLocalRouteAssignment(params_) &&
                 params_.external_route_source !=
                     MoEExternalCanonicalRouteSource::Unspecified &&
                 params_.route_participant_id >= 0)) &&
               params_.reduction_role !=
                   MoECanonicalRouteReductionRole::Unspecified;
    }

    bool MoECanonicalRouteReduceStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        if (!supportsGraphCaptureAfterLaunchPreparation())
            return false;

        if (params_.device_id.is_cuda())
            return supportsBackend(ComputeBackendType::GPU_CUDA);
        if (params_.device_id.is_rocm())
            return supportsBackend(ComputeBackendType::GPU_ROCM);
        return false;
    }

    bool MoECanonicalRouteReduceStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        /*
         * Every bucket row is reduced independently and overwritten in full.
         * Unlike KV, GDN, and short-convolution stages, this epilogue owns no
         * cross-row state whose lifetime would be advanced by padding rows.
         */
        return supportsLazyPrefillGraphCapturePreflight();
    }

    StageBufferRequirements
    MoECanonicalRouteReduceStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        const bool participates =
            params_.node_local_route_exchange ||
            ownsCanonicalRouteReduction(params_.reduction_role);
        if (!participates)
            return reqs;
        if (params_.canonical_route_contributions)
        {
            reqs.addInput(
                "canonical_route_contributions",
                params_.canonical_route_contributions->shape(),
                toBufferTensorType(
                    params_.canonical_route_contributions->native_type()));
        }
        if (params_.canonical_route_arithmetic ==
                MoECanonicalRouteArithmeticPolicy::
                    UnweightedExpertRowThenOrderedFMA &&
            params_.routing_weights)
        {
            reqs.addInput(
                "routing_weights",
                params_.routing_weights->shape(),
                toBufferTensorType(
                    params_.routing_weights->native_type()));
        }
        const bool writes_output =
            ownsCanonicalRouteReduction(params_.reduction_role) &&
            params_.node_local_sparse_phase != MoENodeLocalSparsePhase::StageOnly;
        if (writes_output &&
            params_.gated_shared_residual)
        {
            reqs.addInput(
                "gated_shared_residual",
                params_.gated_shared_residual->shape(),
                toBufferTensorType(
                    params_.gated_shared_residual->native_type()));
        }
        if (writes_output &&
            params_.output)
        {
            reqs.addOutput(
                "output",
                params_.output->shape(),
                toBufferTensorType(params_.output->native_type()));
        }
        return reqs;
    }

    StageBufferContract MoECanonicalRouteReduceStage::bufferContract() const
    {
        const bool participates =
            params_.node_local_route_exchange ||
            ownsCanonicalRouteReduction(params_.reduction_role);
        if (!participates)
            return {};
        const bool writes_output =
            ownsCanonicalRouteReduction(params_.reduction_role) &&
            params_.node_local_sparse_phase != MoENodeLocalSparsePhase::StageOnly;
        auto contract = StageBufferContract::build();
        contract.addInput(params_.canonical_route_contributions_buffer_id);
        if (params_.canonical_route_arithmetic ==
            MoECanonicalRouteArithmeticPolicy::
                UnweightedExpertRowThenOrderedFMA)
        {
            contract.addInput(params_.routing_weights_buffer_id);
        }
        if (writes_output &&
            params_.gated_shared_residual &&
            params_.gated_shared_residual_buffer_id != BufferId::_COUNT)
        {
            contract.addInput(params_.gated_shared_residual_buffer_id);
        }
        if (writes_output)
            contract.addOutput(params_.output_buffer_id);
        return contract;
    }

    StageDumpInfo MoECanonicalRouteReduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const bool owns_reduction =
            ownsCanonicalRouteReduction(params_.reduction_role);
        const bool participates =
            owns_reduction || params_.node_local_route_exchange;
        const bool publishes_complete_canonical_bank =
            owns_reduction && !params_.node_local_route_exchange &&
            params_.external_route_source ==
                MoEExternalCanonicalRouteSource::RootCanonicalRouteBank;
        if (participates && params_.canonical_route_contributions)
        {
            info.addInput(
                "canonical_route_contributions",
                params_.canonical_route_contributions,
                static_cast<size_t>(params_.seq_len * params_.top_k),
                static_cast<size_t>(params_.d_model));

            /* A mapped single-participant continuation materializes every
             * returned route in this bank before the ordered fold. Publish a
             * diagnostic output alias at that final boundary so parity sees
             * the completed bank, not the local-only value emitted by the
             * earlier expert stage. Node-local peer folding deliberately does
             * not qualify: peer slots stream directly into the dense result
             * and are not all resident in this tensor. */
            if (publishes_complete_canonical_bank)
            {
                info.addOutput(
                    "canonical_route_contributions",
                    params_.canonical_route_contributions,
                    static_cast<size_t>(params_.seq_len * params_.top_k),
                    static_cast<size_t>(params_.d_model));
            }
        }
        if (owns_reduction && params_.routing_weights &&
            params_.canonical_route_arithmetic ==
                MoECanonicalRouteArithmeticPolicy::
                    UnweightedExpertRowThenOrderedFMA)
        {
            info.addInput(
                "routing_weights",
                params_.routing_weights,
                params_.seq_len,
                params_.top_k);
        }
        if (owns_reduction && params_.output)
        {
            info.addOutput(
                "output",
                params_.output,
                params_.seq_len,
                params_.d_model);
        }
        if (owns_reduction && pinned_route_evidence_views_)
            pinned_route_evidence_views_->appendOutputs(info);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt(
            "reduction_role",
            static_cast<int>(params_.reduction_role));
        info.addScalarInt(
            "domain_route_assignment_capacity",
            static_cast<int>(params_.domain_route_assignment.capacity));
        info.addScalarInt(
            "external_route_source",
            static_cast<int>(params_.external_route_source));
        info.addScalarBool("owns_reduction", owns_reduction);
        return info;
    }

    // =========================================================================
    // MoECanonicalRouteGatherStage
    // =========================================================================

    MoECanonicalRouteGatherStage::MoECanonicalRouteGatherStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MoECanonicalRouteGatherStage::execute(IDeviceContext *ctx)
    {
        auto *const global_tp =
            dynamic_cast<IGlobalTPContext *>(params_.tp_ctx);
        if (!ctx || !params_.device_id.is_cpu() || !global_tp ||
            !params_.packed_route_records || params_.seq_len <= 0 ||
            params_.top_k <= 0 || params_.d_model <= 0 ||
            params_.root_participant < 0 ||
            params_.root_participant >= global_tp->degree() ||
            global_tp->degree() <= 1)
        {
            LOG_ERROR("[MoECanonicalRouteGatherStage] Invalid rooted packed "
                      "gather contract"
                      << " ctx=" << (ctx != nullptr)
                      << " device=" << params_.device_id.to_string()
                      << " global_tp=" << (global_tp != nullptr)
                      << " records="
                      << (params_.packed_route_records != nullptr)
                      << " seq_len=" << params_.seq_len
                      << " top_k=" << params_.top_k
                      << " d_model=" << params_.d_model
                      << " root=" << params_.root_participant);
            return false;
        }

        float *const records = params_.packed_route_records->mutable_data();
        const size_t route_slot_count =
            static_cast<size_t>(params_.seq_len) *
            static_cast<size_t>(params_.top_k);
        const size_t record_capacity =
            canonical_moe_route_record::recordCapacity(
                params_.packed_route_records->numel(),
                params_.d_model);
        const size_t local_record_count =
            canonical_moe_route_record::readRecordCount(
                records,
                params_.packed_route_records->numel());
        if (!records || record_capacity < route_slot_count ||
            local_record_count > route_slot_count)
        {
            LOG_ERROR("[MoECanonicalRouteGatherStage] Packed publication "
                      "capacity/count is invalid"
                      << " stage="
                      << (params_.stage_name.empty()
                              ? "(unnamed)"
                              : params_.stage_name)
                      << " capacity=" << record_capacity
                      << " local_records=" << local_record_count
                      << " route_slots=" << route_slot_count);
            return false;
        }

        const bool gather_timing_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_canonical_publication");
        const auto gather_start = gather_timing_enabled
                                      ? PerfStatsCollector::Clock::now()
                                      : PerfStatsCollector::Clock::time_point{};
        size_t gathered_record_count = 0u;
        if (!global_tp->gatherVariableFloatRecordsToRoot(
                records,
                local_record_count,
                records,
                route_slot_count,
                canonical_moe_route_record::recordWidth(params_.d_model),
                params_.root_participant,
                gathered_record_count,
                params_.stage_name))
        {
            LOG_ERROR("[MoECanonicalRouteGatherStage] Native packed rooted "
                      "gather failed"
                      << " stage="
                      << (params_.stage_name.empty()
                              ? "(unnamed)"
                              : params_.stage_name)
                      << " participant=" << global_tp->myIndex()
                      << " root=" << params_.root_participant);
            return false;
        }
        if (global_tp->myIndex() == params_.root_participant &&
            (gathered_record_count > route_slot_count ||
             !canonical_moe_route_record::writeRecordCount(
                 records,
                 params_.packed_route_records->numel(),
                 gathered_record_count)))
        {
            LOG_ERROR("[MoECanonicalRouteGatherStage] Root could not publish "
                      "the gathered packed-record count"
                      << " gathered=" << gathered_record_count
                      << " route_slots=" << route_slot_count);
            return false;
        }

        if (gather_timing_enabled)
        {
            const size_t local_elements =
                local_record_count *
                canonical_moe_route_record::recordWidth(params_.d_model);
            const PerfStatsCollector::Tags tags{
                {"stage",
                 params_.stage_name.empty() ? "unnamed" : params_.stage_name},
                {"root", std::to_string(params_.root_participant)},
                {"participant", std::to_string(global_tp->myIndex())},
                {"degree", std::to_string(global_tp->degree())},
                {"local_records", std::to_string(local_record_count)},
                {"record_width",
                 std::to_string(
                     canonical_moe_route_record::recordWidth(params_.d_model))}};
            const auto gather_finish = PerfStatsCollector::Clock::now();
            const auto gather_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    gather_finish - gather_start)
                    .count();
            PerfStatsCollector::recordTimingNs(
                "moe_canonical_publication",
                "cpu_packed_route_gather",
                gather_ns > 0 ? static_cast<uint64_t>(gather_ns) : 0u,
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.to_string(),
                tags);
            PerfStatsCollector::addCounter(
                "moe_canonical_publication",
                "cpu_packed_route_gather_calls",
                1.0,
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.to_string(),
                tags);
            PerfStatsCollector::addCounter(
                "moe_canonical_publication",
                "cpu_packed_route_gather_local_bytes",
                static_cast<double>(local_elements * sizeof(float)),
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.to_string(),
                tags);
        }
        return true;
    }

    bool MoECanonicalRouteGatherStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements
    MoECanonicalRouteGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.packed_route_records)
        {
            reqs.addInout(
                "packed_route_records",
                params_.packed_route_records->shape(),
                toBufferTensorType(
                    params_.packed_route_records->native_type()));
        }
        return reqs;
    }

    StageBufferContract MoECanonicalRouteGatherStage::bufferContract() const
    {
        if (!params_.packed_route_records)
            return {};
        return StageBufferContract::build().addPreallocatedInOut(
            params_.packed_route_records_buffer_id);
    }

    StageDumpInfo MoECanonicalRouteGatherStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.packed_route_records)
        {
            info.addInput(
                "packed_route_records",
                params_.packed_route_records,
                params_.packed_route_records->rows(),
                params_.packed_route_records->cols());
        }
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("root_participant", params_.root_participant);
        return info;
    }

    // =========================================================================
    // MoECanonicalOutputBroadcastStage
    // =========================================================================

    MoECanonicalOutputBroadcastStage::
        MoECanonicalOutputBroadcastStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MoECanonicalOutputBroadcastStage::execute(IDeviceContext *ctx)
    {
        auto *const global_tp =
            dynamic_cast<IGlobalTPContext *>(params_.tp_ctx);
        if (!ctx || !params_.device_id.is_cpu() || !global_tp ||
            !params_.output || params_.seq_len <= 0 ||
            params_.d_model <= 0 || params_.root_participant < 0 ||
            params_.root_participant >= global_tp->degree() ||
            global_tp->degree() <= 1)
        {
            LOG_ERROR("[MoECanonicalOutputBroadcastStage] Invalid compact "
                      "broadcast contract"
                      << " ctx=" << (ctx != nullptr)
                      << " device=" << params_.device_id.to_string()
                      << " global_tp=" << (global_tp != nullptr)
                      << " output=" << (params_.output != nullptr)
                      << " seq_len=" << params_.seq_len
                      << " d_model=" << params_.d_model
                      << " root=" << params_.root_participant);
            return false;
        }
        const size_t element_count =
            static_cast<size_t>(params_.seq_len) *
            static_cast<size_t>(params_.d_model);
        const bool broadcast_timing_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_canonical_publication");
        const auto broadcast_start = broadcast_timing_enabled
                                         ? PerfStatsCollector::Clock::now()
                                         : PerfStatsCollector::Clock::time_point{};
        if (!global_tp->broadcastFloatElements(
                params_.output,
                element_count,
                params_.root_participant,
                params_.stage_name))
        {
            LOG_ERROR("[MoECanonicalOutputBroadcastStage] Compact rooted "
                      "broadcast failed"
                      << " stage="
                      << (params_.stage_name.empty()
                              ? "(unnamed)"
                              : params_.stage_name)
                      << " participant=" << global_tp->myIndex()
                      << " root=" << params_.root_participant);
            return false;
        }
        if (broadcast_timing_enabled)
        {
            const PerfStatsCollector::Tags broadcast_tags{
                {"stage",
                 params_.stage_name.empty() ? "unnamed" : params_.stage_name},
                {"root", std::to_string(params_.root_participant)},
                {"participant", std::to_string(global_tp->myIndex())},
                {"degree", std::to_string(global_tp->degree())},
                {"bytes", std::to_string(element_count * sizeof(float))}};
            const auto broadcast_finish = PerfStatsCollector::Clock::now();
            const auto broadcast_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    broadcast_finish - broadcast_start)
                    .count();
            PerfStatsCollector::recordTimingNs(
                "moe_canonical_publication",
                "cpu_compact_output_broadcast",
                broadcast_ns > 0
                    ? static_cast<uint64_t>(broadcast_ns)
                    : 0u,
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.to_string(),
                broadcast_tags);
            PerfStatsCollector::addCounter(
                "moe_canonical_publication",
                "cpu_compact_output_broadcast_calls",
                1.0,
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.to_string(),
                broadcast_tags);
        }
        return true;
    }

    bool MoECanonicalOutputBroadcastStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements
    MoECanonicalOutputBroadcastStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.output)
        {
            reqs.addInout(
                "output",
                params_.output->shape(),
                toBufferTensorType(params_.output->native_type()));
        }
        return reqs;
    }

    StageBufferContract
    MoECanonicalOutputBroadcastStage::bufferContract() const
    {
        if (!params_.output)
            return {};
        return StageBufferContract::build().addPreallocatedInOut(
            params_.output_buffer_id);
    }

    StageDumpInfo
    MoECanonicalOutputBroadcastStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.output)
        {
            info.addOutput(
                "output",
                params_.output,
                params_.seq_len,
                params_.d_model);
        }
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("root_participant", params_.root_participant);
        return info;
    }

    // =========================================================================
    // MoESharedExpertRankBankPublishStage
    // =========================================================================

    MoESharedExpertRankBankPublishStage::
        MoESharedExpertRankBankPublishStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MoESharedExpertRankBankPublishStage::execute(
        IDeviceContext *ctx)
    {
        if (!ctx || !params_.shared_output ||
            !params_.canonical_publication || params_.seq_len <= 0 ||
            params_.top_k <= 0 || params_.d_model <= 0 ||
            params_.participant_count <= 0 ||
            params_.participant_device_index < 0 ||
            params_.participant_device_index >= params_.participant_count)
        {
            LOG_ERROR(
                "[MoESharedExpertRankBankPublishStage] Invalid execution "
                "contract device="
                << params_.device_id.to_string()
                << " seq_len=" << params_.seq_len
                << " top_k=" << params_.top_k
                << " d_model=" << params_.d_model
                << " participant=" << params_.participant_device_index
                << " participants=" << params_.participant_count);
            return false;
        }

        if (!moe_kernel_)
        {
            owned_moe_kernel_ =
                KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        IMoEKernel *kernel = bindStageStream(moe_kernel_);
        if (!kernel ||
            !kernel->publishSharedExpertRankBank(
                params_.shared_output,
                params_.canonical_publication,
                params_.seq_len,
                params_.top_k,
                params_.d_model,
                params_.participant_device_index,
                params_.participant_count,
                reinterpret_cast<const int *>(
                    params_.active_row_count_device)))
        {
            LOG_ERROR(
                "[MoESharedExpertRankBankPublishStage] Device publication "
                "failed device="
                << params_.device_id.to_string()
                << " participant=" << params_.participant_device_index
                << " participants=" << params_.participant_count);
            return false;
        }

        const StageGPUExecution execution = gpuExecution();
        execution.publish(params_.canonical_publication);
        return true;
    }

    bool MoESharedExpertRankBankPublishStage::supportsBackend(
        ComputeBackendType backend) const
    {
        switch (backend)
        {
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool MoESharedExpertRankBankPublishStage::isGraphCapturable() const
    {
        return supportsGraphCaptureAfterLaunchPreparation() &&
               moe_kernel_ != nullptr;
    }

    bool MoESharedExpertRankBankPublishStage::
        supportsGraphCaptureAfterLaunchPreparation() const
    {
        return params_.device_id.is_gpu() && params_.shared_output &&
               params_.canonical_publication && params_.seq_len > 0 &&
               params_.top_k > 0 && params_.d_model > 0 &&
               params_.participant_count > 0 &&
               params_.participant_device_index >= 0 &&
               params_.participant_device_index < params_.participant_count;
    }

    bool MoESharedExpertRankBankPublishStage::
        supportsLazyPrefillGraphCapturePreflight() const
    {
        if (!supportsGraphCaptureAfterLaunchPreparation())
            return false;
        if (params_.device_id.is_cuda())
            return supportsBackend(ComputeBackendType::GPU_CUDA);
        if (params_.device_id.is_rocm())
            return supportsBackend(ComputeBackendType::GPU_ROCM);
        return false;
    }

    bool MoESharedExpertRankBankPublishStage::
        supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsLazyPrefillGraphCapturePreflight();
    }

    bool MoESharedExpertRankBankPublishStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR(
                "[MoESharedExpertRankBankPublishStage] Graph preparation "
                "requires the exact non-null producer stream");
            return false;
        }
        setGPUStream(stream);
        if (!moe_kernel_)
        {
            owned_moe_kernel_ =
                KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        return bindStageStream(moe_kernel_) != nullptr;
    }

    StageBufferRequirements
    MoESharedExpertRankBankPublishStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.shared_output)
        {
            reqs.addInput(
                "shared_output",
                params_.shared_output->shape(),
                toBufferTensorType(params_.shared_output->native_type()));
        }
        if (params_.canonical_publication)
        {
            reqs.addOutput(
                "canonical_publication",
                params_.canonical_publication->shape(),
                toBufferTensorType(
                    params_.canonical_publication->native_type()));
        }
        return reqs;
    }

    StageBufferContract
    MoESharedExpertRankBankPublishStage::bufferContract() const
    {
        return StageBufferContract::build()
            .addInput(params_.shared_output_buffer_id)
            .addInOut(params_.canonical_publication_buffer_id);
    }

    StageDumpInfo
    MoESharedExpertRankBankPublishStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.shared_output)
        {
            info.addInput(
                "shared_output",
                params_.shared_output,
                params_.seq_len,
                params_.d_model);
        }
        if (params_.canonical_publication)
        {
            info.addOutput(
                "canonical_publication",
                params_.canonical_publication,
                static_cast<size_t>(params_.seq_len) *
                    static_cast<size_t>(
                        params_.top_k + params_.participant_count),
                params_.d_model);
        }
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt(
            "participant_device_index",
            params_.participant_device_index);
        info.addScalarInt("participant_count", params_.participant_count);
        return info;
    }

    // =========================================================================
    // MoECanonicalPublicationFinalizeStage
    // =========================================================================

    MoECanonicalPublicationFinalizeStage::
        MoECanonicalPublicationFinalizeStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MoECanonicalPublicationFinalizeStage::execute(
        IDeviceContext *ctx)
    {
        if (!ctx || !params_.input || !params_.gate_inp ||
            !params_.canonical_publication || !params_.routed_output ||
            !params_.shared_output || !params_.combined_output ||
            params_.seq_len <= 0 || params_.top_k <= 0 ||
            params_.d_model <= 0 || params_.participant_count <= 0 ||
            params_.participant_device_index < 0 ||
            params_.participant_device_index >= params_.participant_count ||
            params_.root_device_index < 0 ||
            params_.root_device_index >= params_.participant_count)
        {
            LOG_ERROR(
                "[MoECanonicalPublicationFinalizeStage] Invalid execution "
                "contract device="
                << params_.device_id.to_string()
                << " seq_len=" << params_.seq_len
                << " top_k=" << params_.top_k
                << " d_model=" << params_.d_model
                << " participant=" << params_.participant_device_index
                << " root=" << params_.root_device_index
                << " participants=" << params_.participant_count);
            return false;
        }

        // The rooted reduction owns valid publication bytes only on this rank.
        if (params_.participant_device_index != params_.root_device_index)
            return true;

        if (!moe_kernel_)
        {
            owned_moe_kernel_ =
                KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        IMoEKernel *kernel = bindStageStream(moe_kernel_);
        if (!kernel ||
            !kernel->finalizeCanonicalMoEPublication(
                params_.input,
                params_.gate_inp,
                params_.canonical_publication,
                params_.routed_output,
                params_.shared_output,
                params_.combined_output,
                params_.seq_len,
                params_.top_k,
                params_.d_model,
                params_.participant_count,
                reinterpret_cast<const int *>(
                    params_.active_row_count_device)))
        {
            LOG_ERROR(
                "[MoECanonicalPublicationFinalizeStage] Device finalizer "
                "failed device="
                << params_.device_id.to_string()
                << " root=" << params_.root_device_index);
            return false;
        }

        const StageGPUExecution execution = gpuExecution();
        execution.publish(params_.routed_output);
        execution.publish(params_.shared_output);
        execution.publish(params_.combined_output);
        return true;
    }

    size_t MoECanonicalPublicationFinalizeStage::estimatedFlops() const
    {
        if (params_.participant_device_index != params_.root_device_index)
            return 0;
        const size_t rows = static_cast<size_t>(params_.seq_len);
        const size_t columns = static_cast<size_t>(params_.d_model);
        const size_t route_adds = static_cast<size_t>(
            std::max(0, params_.top_k - 1));
        const size_t rank_adds = static_cast<size_t>(
            std::max(0, params_.participant_count - 1));
        return rows * columns *
               (size_t{4} + route_adds + rank_adds);
    }

    bool MoECanonicalPublicationFinalizeStage::supportsBackend(
        ComputeBackendType backend) const
    {
        switch (backend)
        {
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool MoECanonicalPublicationFinalizeStage::isGraphCapturable() const
    {
        if (!supportsGraphCaptureAfterLaunchPreparation())
            return false;
        return params_.participant_device_index !=
                   params_.root_device_index ||
               moe_kernel_ != nullptr;
    }

    bool MoECanonicalPublicationFinalizeStage::
        supportsGraphCaptureAfterLaunchPreparation() const
    {
        return params_.device_id.is_gpu() && params_.input &&
               params_.gate_inp && params_.canonical_publication &&
               params_.routed_output && params_.shared_output &&
               params_.combined_output && params_.seq_len > 0 &&
               params_.top_k > 0 && params_.d_model > 0 &&
               params_.participant_count > 0 &&
               params_.participant_device_index >= 0 &&
               params_.participant_device_index < params_.participant_count &&
               params_.root_device_index >= 0 &&
               params_.root_device_index < params_.participant_count;
    }

    bool MoECanonicalPublicationFinalizeStage::
        supportsLazyPrefillGraphCapturePreflight() const
    {
        if (!supportsGraphCaptureAfterLaunchPreparation())
            return false;
        if (params_.device_id.is_cuda())
            return supportsBackend(ComputeBackendType::GPU_CUDA);
        if (params_.device_id.is_rocm())
            return supportsBackend(ComputeBackendType::GPU_ROCM);
        return false;
    }

    bool MoECanonicalPublicationFinalizeStage::
        supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsLazyPrefillGraphCapturePreflight();
    }

    bool MoECanonicalPublicationFinalizeStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR(
                "[MoECanonicalPublicationFinalizeStage] Graph preparation "
                "requires the exact non-null producer stream");
            return false;
        }
        setGPUStream(stream);
        if (params_.participant_device_index != params_.root_device_index)
            return true;
        if (!moe_kernel_)
        {
            owned_moe_kernel_ =
                KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        return bindStageStream(moe_kernel_) != nullptr;
    }

    StageBufferRequirements
    MoECanonicalPublicationFinalizeStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.participant_device_index != params_.root_device_index)
            return reqs;
        if (params_.input)
        {
            reqs.addInput(
                "input",
                params_.input->shape(),
                toBufferTensorType(params_.input->native_type()));
        }
        if (params_.canonical_publication)
        {
            reqs.addInput(
                "canonical_publication",
                params_.canonical_publication->shape(),
                toBufferTensorType(
                    params_.canonical_publication->native_type()));
        }
        if (params_.routed_output)
        {
            reqs.addOutput(
                "routed_output",
                params_.routed_output->shape(),
                toBufferTensorType(params_.routed_output->native_type()));
        }
        if (params_.shared_output)
        {
            reqs.addOutput(
                "shared_output",
                params_.shared_output->shape(),
                toBufferTensorType(params_.shared_output->native_type()));
        }
        if (params_.combined_output)
        {
            reqs.addOutput(
                "combined_output",
                params_.combined_output->shape(),
                toBufferTensorType(params_.combined_output->native_type()));
        }
        return reqs;
    }

    StageBufferContract
    MoECanonicalPublicationFinalizeStage::bufferContract() const
    {
        if (params_.participant_device_index != params_.root_device_index)
            return {};
        return StageBufferContract::build()
            .addInput(params_.input_buffer_id)
            .addWeight(params_.gate_inp)
            .addInput(params_.canonical_publication_buffer_id)
            .addOutput(params_.routed_output_buffer_id)
            .addOutput(params_.shared_output_buffer_id)
            .addOutput(params_.combined_output_buffer_id);
    }

    StageDumpInfo
    MoECanonicalPublicationFinalizeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const bool owns_finalization =
            params_.participant_device_index == params_.root_device_index;
        if (owns_finalization && params_.input)
        {
            info.addInput(
                "input", params_.input, params_.seq_len, params_.d_model);
        }
        if (owns_finalization && params_.gate_inp)
            info.addWeight("gate_inp", params_.gate_inp);
        if (owns_finalization && params_.canonical_publication)
        {
            info.addInput(
                "canonical_publication",
                params_.canonical_publication,
                static_cast<size_t>(params_.seq_len) *
                    static_cast<size_t>(
                        params_.top_k + params_.participant_count),
                params_.d_model);
        }
        if (owns_finalization && params_.routed_output)
        {
            info.addOutput(
                "routed_output",
                params_.routed_output,
                params_.seq_len,
                params_.d_model);
        }
        if (owns_finalization && params_.shared_output)
        {
            info.addOutput(
                "shared_output",
                params_.shared_output,
                params_.seq_len,
                params_.d_model);
        }
        if (owns_finalization && params_.combined_output)
        {
            info.addOutput(
                "combined_output",
                params_.combined_output,
                params_.seq_len,
                params_.d_model);
        }
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt(
            "participant_device_index",
            params_.participant_device_index);
        info.addScalarInt("root_device_index", params_.root_device_index);
        info.addScalarInt("participant_count", params_.participant_count);
        info.addScalarBool("owns_finalization", owns_finalization);
        return info;
    }

} // namespace llaminar2
