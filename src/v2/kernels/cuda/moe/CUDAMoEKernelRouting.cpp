/**
 * @file CUDAMoEKernelRouting.cpp
 * @brief CUDAMoEKernel routing, rebalance controller, token grouping, and descriptor table setup.
 */
#include "CUDAMoEKernelInternal.h"

namespace llaminar2
{
    bool CUDAMoEKernel::tryRouteDecodeLogitsQ8(
        const float *d_hidden,
        const float *d_gate,
        int d_model,
        int num_experts,
        int top_k,
        const char *context)
    {
        invalidateRouterQ8HiddenPublication();
        if (!debugEnv().gemm.cuda_moe_router_q8)
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] Q8 router requested while disabled");
            return false;
        }
        if (!d_hidden || !d_gate || d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] invalid Q8 router shape or pointer state");
            return false;
        }
        if (!ensureGroupedGateUpDecodeCapacity(top_k, d_model))
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] Q8 router hidden scratch unavailable");
            return false;
        }

        const auto *q8_gate =
            getOrCreateQ8RouterGateCache(d_gate, d_model, num_experts);
        if (!q8_gate)
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] Q8 router gate cache unavailable");
            return false;
        }

        void *stream = requireStream(context ? context : "CUDAMoEKernel::tryRouteDecodeLogitsQ8");
        const bool ok = cudaMoE_gate_logits_single_token_q8_weights(
            d_hidden,
            d_decode_hidden_int8_,
            d_decode_hidden_scales_,
            q8_gate->d_gate_weights_q8,
            q8_gate->d_gate_scales,
            d_route_logits_,
            d_model,
            num_experts,
            device_ordinal_,
            stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        if (!ok)
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] Q8 router logits kernel failed");
            return false;
        }

        publishRouterQ8Hidden(
            d_hidden,
            /*rows=*/1,
            isCudaMoEDecodeCaptureActive(stream),
            routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        PerfStatsCollector::addCounter(
            "kernel", "cuda_moe_router_q8_decode_calls", 1.0, {}, {},
            {{"num_experts", std::to_string(num_experts)},
             {"d_model", std::to_string(d_model)}});
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedGateUpKPartScratchCapacity(
        int active_slots,
        int k_partitions,
        int intermediate)
    {
        // Only the discrete partition counts the kpart launcher accepts are valid.
        if (active_slots <= 0 || intermediate <= 0 ||
            !(k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
              k_partitions == 16 || k_partitions == 32))
            return false;

        // Fast path: existing buffers already cover the requested shape.
        if (d_grouped_gateup_gate_partials_ && d_grouped_gateup_up_partials_ &&
            grouped_gateup_kpart_active_cap_ >= active_slots &&
            grouped_gateup_kpart_partitions_cap_ >= k_partitions &&
            grouped_gateup_kpart_intermediate_cap_ >= intermediate)
            return true;

        const size_t partial_count = static_cast<size_t>(active_slots) *
                                     static_cast<size_t>(k_partitions) *
                                     static_cast<size_t>(intermediate);
        void *gate_partials = nullptr;
        void *up_partials = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&gate_partials, MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,
                                partial_count * sizeof(float), "gate/up gate partials") &&
            bindWorkspaceBuffer(&up_partials, MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
                                partial_count * sizeof(float), "gate/up up partials");
        if (!ok)
        {
            d_grouped_gateup_gate_partials_ = nullptr;
            d_grouped_gateup_up_partials_ = nullptr;
            grouped_gateup_kpart_active_cap_ = 0;
            grouped_gateup_kpart_partitions_cap_ = 0;
            grouped_gateup_kpart_intermediate_cap_ = 0;
            return false;
        }

        d_grouped_gateup_gate_partials_ = static_cast<float *>(gate_partials);
        d_grouped_gateup_up_partials_ = static_cast<float *>(up_partials);
        grouped_gateup_kpart_active_cap_ = active_slots;
        grouped_gateup_kpart_partitions_cap_ = k_partitions;
        grouped_gateup_kpart_intermediate_cap_ = intermediate;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedDownKPartScratchCapacity(int k_partitions, int d_model, int slots)
    {
        // Only the discrete partition counts the kpart launcher accepts are valid.
        if (d_model <= 0 || slots <= 0 ||
            !(k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
              k_partitions == 16))
            return false;

        // Every split-K down path stores one partial row per original route.
        // M=1 decode therefore requests top_k rows, while grouped verifier
        // prefill requests tile_rows * top_k rows.  Collective and direct
        // publication deliberately share this capacity contract so changing
        // the publication destination cannot change the arithmetic layout or
        // silently make an otherwise valid graph write beyond its workspace.
        if (d_grouped_down_partials_ &&
            grouped_down_kpart_partitions_cap_ >= k_partitions &&
            grouped_down_kpart_d_model_cap_ >= d_model &&
            grouped_down_kpart_slots_cap_ >= slots)
            return true;

        const size_t partial_count =
            static_cast<size_t>(slots) *
            static_cast<size_t>(k_partitions) *
            static_cast<size_t>(d_model);
        void *down_partials = nullptr;
        if (!bindWorkspaceBuffer(&down_partials, MoEWorkspaceBuffers::DOWN_PARTIALS,
                                 partial_count * sizeof(float), "down partials"))
        {
            d_grouped_down_partials_ = nullptr;
            grouped_down_kpart_partitions_cap_ = 0;
            grouped_down_kpart_d_model_cap_ = 0;
            grouped_down_kpart_slots_cap_ = 0;
            return false;
        }

        d_grouped_down_partials_ = static_cast<float *>(down_partials);
        grouped_down_kpart_partitions_cap_ = k_partitions;
        grouped_down_kpart_d_model_cap_ = d_model;
        grouped_down_kpart_slots_cap_ = slots;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedDownDecodeCapacity(int top_k, int intermediate)
    {
        if (top_k <= 0 || top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            intermediate <= 0 || (intermediate % 32) != 0)
            return false;

        const bool need_growth = top_k > decode_down_topk_cap_ ||
                                 intermediate > decode_down_intermediate_cap_ ||
                                 !d_decode_swiglu_int8_ || !d_decode_swiglu_scales_;
        if (!need_growth)
            return true;

        const int blocks_per_row = intermediate / 32;
        void *decode_swiglu_int8 = nullptr;
        void *decode_swiglu_scales = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&decode_swiglu_int8, MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,
                                static_cast<size_t>(top_k) * intermediate * sizeof(int8_t), "decode SwiGLU int8") &&
            bindWorkspaceBuffer(&decode_swiglu_scales, MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
                                static_cast<size_t>(top_k) * blocks_per_row * sizeof(float), "decode SwiGLU scales");
        if (!ok)
        {
            d_decode_swiglu_int8_ = nullptr;
            d_decode_swiglu_scales_ = nullptr;
            decode_down_topk_cap_ = 0;
            decode_down_intermediate_cap_ = 0;
            return false;
        }

        d_decode_swiglu_int8_ = static_cast<int8_t *>(decode_swiglu_int8);
        d_decode_swiglu_scales_ = static_cast<float *>(decode_swiglu_scales);
        decode_down_topk_cap_ = top_k;
        decode_down_intermediate_cap_ = intermediate;
        return true;
    }

    bool CUDAMoEKernel::ensureRoutingDecodeMetadataCapacity(int num_active)
    {
        if (num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }
        if (d_routing_decode_expert_ids_ &&
            routing_decode_metadata_cap_ >= num_active)
        {
            return true;
        }

        void *routing_expert_ids = nullptr;
        if (!bindWorkspaceBuffer(
                &routing_expert_ids,
                MoEWorkspaceBuffers::CUDA_ROUTING_DECODE_EXPERT_IDS,
                static_cast<size_t>(num_active) * sizeof(int),
                "CUDA routing decode expert ids"))
        {
            d_routing_decode_expert_ids_ = nullptr;
            routing_decode_metadata_cap_ = 0;
            return false;
        }

        d_routing_decode_expert_ids_ = static_cast<int *>(routing_expert_ids);
        routing_decode_metadata_cap_ = num_active;
        return true;
    }

    bool CUDAMoEKernel::resolveFixedTableGateUpMetadata(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        const int *expert_ids,
        int num_active,
        const int **device_expert_ids)
    {
        if (!expert_ids || !device_expert_ids || num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        void *raw_stream = requireStream(
            "CUDAMoEKernel::resolveFixedTableGateUpMetadata");
        const bool capture_active = isCudaMoEDecodeCaptureActive(raw_stream);
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::GateUp,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "fixed-table gate/up metadata"))
        {
            return false;
        }

        auto *slot_ids = static_cast<int *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_GATEUP_EXPERT_IDS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(int)));
        if (!slot_ids)
            return false;

        const auto state = fixed_gateup_metadata_states_.find(workspace_slot);
        if (state != fixed_gateup_metadata_states_.end())
        {
            const bool matches =
                state->second.num_active == num_active &&
                std::equal(
                    state->second.expert_ids.begin(),
                    state->second.expert_ids.begin() + num_active,
                    expert_ids);
            if (!matches)
            {
                LOG_ERROR("[CUDAMoEKernel] Fixed-table gate/up metadata is "
                          "immutable for one captured owner slot; use the "
                          "device-routed API for changing expert ids");
                return false;
            }
            *device_expert_ids = slot_ids;
            return true;
        }
        if (capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel] Fixed-table gate/up metadata was not "
                      "published before graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "resolveFixedTableGateUpMetadata"))
            return false;

        const cudaError_t err = cudaMemcpyAsync(
            slot_ids,
            expert_ids,
            static_cast<std::size_t>(num_active) * sizeof(int),
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(raw_stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] Fixed-table gate/up metadata publication failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        FixedGateUpMetadataState published;
        published.num_active = num_active;
        std::copy(expert_ids, expert_ids + num_active, published.expert_ids.begin());
        fixed_gateup_metadata_states_.emplace(workspace_slot, std::move(published));
        *device_expert_ids = slot_ids;
        return true;
    }

    bool CUDAMoEKernel::resolveFixedTableDownMetadata(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        const int *expert_ids,
        const float *expert_weights,
        int num_active,
        const int **device_expert_ids,
        const float **device_expert_weights)
    {
        if (!expert_ids || !expert_weights || !device_expert_ids ||
            !device_expert_weights || num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }

        void *raw_stream = requireStream(
            "CUDAMoEKernel::resolveFixedTableDownMetadata");
        const bool capture_active = isCudaMoEDecodeCaptureActive(raw_stream);
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::Down,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "fixed-table down metadata"))
        {
            return false;
        }

        auto *slot_ids = static_cast<int *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_DOWN_EXPERT_IDS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(int)));
        auto *slot_weights = static_cast<float *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_DOWN_WEIGHTS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(float)));
        if (!slot_ids || !slot_weights)
            return false;

        const auto state = fixed_down_metadata_states_.find(workspace_slot);
        if (state != fixed_down_metadata_states_.end())
        {
            const bool matches =
                state->second.num_active == num_active &&
                std::equal(
                    state->second.expert_ids.begin(),
                    state->second.expert_ids.begin() + num_active,
                    expert_ids) &&
                std::equal(
                    state->second.expert_weights.begin(),
                    state->second.expert_weights.begin() + num_active,
                    expert_weights);
            if (!matches)
            {
                LOG_ERROR("[CUDAMoEKernel] Fixed-table down metadata is "
                          "immutable for one captured owner slot; use the "
                          "device-routed API for changing expert ids or weights");
                return false;
            }
            *device_expert_ids = slot_ids;
            *device_expert_weights = slot_weights;
            return true;
        }
        if (capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel] Fixed-table down metadata was not "
                      "published before graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "resolveFixedTableDownMetadata"))
            return false;

        cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
        cudaError_t err = cudaMemcpyAsync(
            slot_ids,
            expert_ids,
            static_cast<std::size_t>(num_active) * sizeof(int),
            cudaMemcpyHostToDevice,
            stream);
        if (err == cudaSuccess)
        {
            err = cudaMemcpyAsync(
                slot_weights,
                expert_weights,
                static_cast<std::size_t>(num_active) * sizeof(float),
                cudaMemcpyHostToDevice,
                stream);
        }
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] Fixed-table down metadata publication failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        FixedDownMetadataState published;
        published.num_active = num_active;
        std::copy(expert_ids, expert_ids + num_active, published.expert_ids.begin());
        std::copy(
            expert_weights,
            expert_weights + num_active,
            published.expert_weights.begin());
        fixed_down_metadata_states_.emplace(workspace_slot, std::move(published));
        *device_expert_ids = slot_ids;
        *device_expert_weights = slot_weights;
        return true;
    }

    bool CUDAMoEKernel::runtimePointerWorkspaceSlot(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        MoERuntimePointerArrayRole role,
        MoERuntimePointerWorkspaceAccess access,
        std::size_t *workspace_slot,
        const char *context)
    {
        if (!workspace_slot)
            return false;

        const auto slot = runtime_pointer_workspace_owners_.resolve(
            workspace_,
            role,
            persistent_descriptor_slot,
            static_cast<std::size_t>(scope),
            access,
            context);
        if (!slot || *slot >= kRuntimePointerArrayWorkspaceEntries)
        {
            LOG_ERROR("[CUDAMoEKernel] "
                      << (context ? context : "runtime pointer table")
                      << " could not resolve an exclusive pointer workspace slot");
            return false;
        }

        *workspace_slot = *slot;
        return true;
    }

    bool CUDAMoEKernel::ensureRuntimeGateUpPointerArrays(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        int top_k,
        const std::array<float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        float ***d_gate_ptrs,
        float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        cudaStream_t stream = static_cast<cudaStream_t>(
            requireStream("CUDAMoEKernel::ensureRuntimeGateUpPointerArrays"));
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::GateUp,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "grouped gate/up"))
            return false;
        if (!setMoEDevice(device_ordinal_, "ensureRuntimeGateUpPointerArrays"))
            return false;

        void *gate_ptr_workspace = nullptr;
        void *up_ptr_workspace = nullptr;
        const size_t workspace_bytes =
            kRuntimePointerArrayWorkspaceEntries *
            kRuntimePointerArrayMaxTopK *
            sizeof(float *);
        if (!bindWorkspaceBuffer(&gate_ptr_workspace,
                                 MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_GATE_PTRS,
                                 workspace_bytes,
                                 "CUDA grouped gate/up gate pointer arrays") ||
            !bindWorkspaceBuffer(&up_ptr_workspace,
                                 MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_UP_PTRS,
                                 workspace_bytes,
                                 "CUDA grouped gate/up up pointer arrays"))
        {
            return false;
        }

        float **slot_gate_ptrs =
            static_cast<float **>(gate_ptr_workspace) +
            workspace_slot * kRuntimePointerArrayMaxTopK;
        float **slot_up_ptrs =
            static_cast<float **>(up_ptr_workspace) +
            workspace_slot * kRuntimePointerArrayMaxTopK;

        if (capture_active)
        {
            if (!gateup_pointer_slot_ready_[workspace_slot])
            {
                LOG_ERROR("[CUDAMoEKernel] grouped gate/up pointer workspace slot "
                          << workspace_slot << " was not staged before graph capture");
                return false;
            }
            *d_gate_ptrs = slot_gate_ptrs;
            *d_up_ptrs = slot_up_ptrs;
            return true;
        }

        cudaError_t err = cudaMemcpyAsync(slot_gate_ptrs, gate_ptrs.data(),
                                          static_cast<size_t>(top_k) * sizeof(float *),
                                          cudaMemcpyHostToDevice, stream);
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(slot_up_ptrs, up_ptrs.data(),
                                  static_cast<size_t>(top_k) * sizeof(float *),
                                  cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped gate/up pointer staging failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        gateup_pointer_slot_ready_[workspace_slot] = true;
        *d_gate_ptrs = slot_gate_ptrs;
        *d_up_ptrs = slot_up_ptrs;
        return true;
    }

    bool CUDAMoEKernel::ensureRuntimeDownPointerArrays(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        int top_k,
        const std::array<const float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<const float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        const float ***d_gate_ptrs,
        const float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        cudaStream_t stream = static_cast<cudaStream_t>(
            requireStream("CUDAMoEKernel::ensureRuntimeDownPointerArrays"));
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::Down,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "grouped down"))
            return false;
        if (!setMoEDevice(device_ordinal_, "ensureRuntimeDownPointerArrays"))
            return false;

        void *gate_ptr_workspace = nullptr;
        void *up_ptr_workspace = nullptr;
        const size_t workspace_bytes =
            kRuntimePointerArrayWorkspaceEntries *
            kRuntimePointerArrayMaxTopK *
            sizeof(const float *);
        if (!bindWorkspaceBuffer(&gate_ptr_workspace,
                                 MoEWorkspaceBuffers::CUDA_DECODE_DOWN_GATE_PTRS,
                                 workspace_bytes,
                                 "CUDA grouped down gate pointer arrays") ||
            !bindWorkspaceBuffer(&up_ptr_workspace,
                                 MoEWorkspaceBuffers::CUDA_DECODE_DOWN_UP_PTRS,
                                 workspace_bytes,
                                 "CUDA grouped down up pointer arrays"))
        {
            return false;
        }

        const float **slot_gate_ptrs =
            static_cast<const float **>(gate_ptr_workspace) +
            workspace_slot * kRuntimePointerArrayMaxTopK;
        const float **slot_up_ptrs =
            static_cast<const float **>(up_ptr_workspace) +
            workspace_slot * kRuntimePointerArrayMaxTopK;

        if (capture_active)
        {
            if (!down_pointer_slot_ready_[workspace_slot])
            {
                LOG_ERROR("[CUDAMoEKernel] grouped down pointer workspace slot "
                          << workspace_slot << " was not staged before graph capture");
                return false;
            }
            *d_gate_ptrs = slot_gate_ptrs;
            *d_up_ptrs = slot_up_ptrs;
            return true;
        }

        cudaError_t err = cudaMemcpyAsync(slot_gate_ptrs, gate_ptrs.data(),
                                          static_cast<size_t>(top_k) * sizeof(const float *),
                                          cudaMemcpyHostToDevice, stream);
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(slot_up_ptrs, up_ptrs.data(),
                                  static_cast<size_t>(top_k) * sizeof(const float *),
                                  cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped down pointer staging failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        down_pointer_slot_ready_[workspace_slot] = true;
        *d_gate_ptrs = slot_gate_ptrs;
        *d_up_ptrs = slot_up_ptrs;
        return true;
    }

    bool CUDAMoEKernel::routeCore(const float *hidden, const void *gate_weights, TensorType gate_type,
                                  int seq_len, int d_model, int num_experts, int top_k,
                                  bool normalize_weights,
                                  float *output_indices, float *output_weights,
                                  const int *device_effective_seq_len,
                                  const MoERouterOwnedRowOutputs *owned)
    {
        if ((!owned && (!output_indices || !output_weights)) ||
            seq_len <= 0 || d_model <= 0 || num_experts <= 0 ||
            top_k <= 0 || top_k > num_experts)
            return false;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        const bool gate_is_bf16 = (gate_type == TensorType::BF16);
        if (!gate_is_fp32 && !gate_is_bf16 && gate_type != TensorType::FP16)
        {
            LOG_ERROR("[CUDAMoEKernel] CUDA router supports FP32, FP16 or BF16 gate weights, got tensor type "
                      << static_cast<int>(gate_type));
            return false;
        }
        if (num_experts > 1024 || top_k > static_cast<int>(kDeviceMoEMaxTopK))
        {
            LOG_ERROR("[CUDAMoEKernel] Unsupported routing shape num_experts=" << num_experts
                                                                                << " top_k=" << top_k);
            return false;
        }

        const size_t logits_count = static_cast<size_t>(seq_len) * num_experts;
        if (!ensureRouteBufferCapacity(logits_count))
            return false;

        void *stream = getStream();
        if (!requireAlignedPointer(hidden, 16, "hidden", "routeCore") ||
            !requireAlignedPointer(d_route_logits_, 16, "route logits", "routeCore") ||
            (!owned && (!requireAlignedPointer(output_indices, 16, "output indices", "routeCore") ||
                        !requireAlignedPointer(output_weights, 16, "output weights", "routeCore"))))
            return false;
        if (gate_is_fp32 && !requireAlignedPointer(gate_weights, 16, "FP32 gate", "routeCore"))
            return false;
        if (!requireCudaDevicePointer(hidden, device_ordinal_, "hidden", "routeCore", stream) ||
            !requireCudaDevicePointer(gate_weights, device_ordinal_, "gate weights", "routeCore", stream) ||
            !requireCudaDevicePointer(d_route_logits_, device_ordinal_, "route logits", "routeCore", stream) ||
            (!owned && (!requireCudaDevicePointer(output_indices, device_ordinal_, "output indices", "routeCore", stream) ||
                        !requireCudaDevicePointer(output_weights, device_ordinal_, "output weights", "routeCore", stream))))
            return false;
        if (device_effective_seq_len &&
            !requireCudaDevicePointer(device_effective_seq_len, device_ordinal_,
                                      "effective prefill sequence length",
                                      "routeCore", stream))
        {
            return false;
        }

        // The explicit Q8 prefill policy uses the same canonical block quantizer
        // and row-local reduction as decode. Setup has already admitted the gate
        // cache and hidden bank; capture only consumes their retained addresses.
        const bool q8_prefill = gate_is_fp32 &&
            debugEnv().gemm.cuda_moe_router_prefill == CUDARouterPrefillArithmetic::Q8;
        invalidateRouterQ8HiddenPublication();
        if (!prepareRequiredRouterQ8Hidden(seq_len, d_model, top_k))
            return false;
        if (!q8_prefill && !publishRequiredRouterQ8Hidden(
                hidden, seq_len, d_model, top_k, device_effective_seq_len))
            return false;
        if (q8_prefill)
        {
            if (!debugEnv().gemm.cuda_moe_router_q8 || d_model % 32 != 0 ||
                !ensureGroupedGateUpDecodeCapacity(top_k, d_model, seq_len)) return false;
            const auto *gate = getOrCreateQ8RouterGateCache(
                static_cast<const float *>(gate_weights), d_model, num_experts);
            if (!gate) return false;
            const auto basis = routerQ8Basis(debugEnv().gemm.moe_router_arithmetic);
            invalidateRouterQ8HiddenPublication();
            if (owned)
            {
                const MoERouterOwnedRowsLaunch launch{
                    .partition = owned->partition, .capacity = seq_len, .width = d_model,
                    .experts = num_experts, .top_k = top_k, .normalize = normalize_weights,
                    .live_rows = device_effective_seq_len, .hidden = hidden,
                    .format = MoERouterPreparedFormat::BlockQ8, .q8_basis = basis,
                    .gate = gate->d_gate_weights_q8, .gate_scales = gate->d_gate_scales,
                    .hidden_q8 = d_decode_hidden_int8_, .hidden_scales = d_decode_hidden_scales_,
                    .logits = d_route_logits_, .selected = owned->selected,
                    .selected_bytes = owned->selected_bytes};
                if (!cuda::routeOwnedRows(launch, stream, router_q8_prefill_hardware_)) return false;
            }
            else
            {
                if (!cudaMoE_gate_logits_q8_weights_prefill_rows(hidden,
                        d_decode_hidden_int8_, d_decode_hidden_scales_, gate->d_gate_weights_q8,
                        gate->d_gate_scales, d_route_logits_, seq_len, d_model, num_experts,
                        device_ordinal_, stream, device_effective_seq_len, basis, router_q8_prefill_hardware_) ||
                    !cudaMoE_softmax_topk(d_route_logits_, output_indices, output_weights,
                        seq_len, num_experts, top_k, normalize_weights, device_ordinal_, stream,
                        device_effective_seq_len)) return false;
            }
            publishRouterQ8Hidden(hidden, seq_len, isCudaMoEDecodeCaptureActive(stream),
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
            const auto geometry = cuda::routerQ8PrefillGeometry(
                owned ? owned->partition.capacityFor(seq_len) : seq_len, d_model, num_experts, router_q8_prefill_hardware_);
            PerfStatsCollector::addCounter("kernel", "cuda_moe_router_q8_prefill_calls", 1.0,
                "router", "cuda:" + std::to_string(device_ordinal_),
                {{"arithmetic", "int8_tensor_core"}, {"tile_rows", std::to_string(geometry.rows)},
                 {"tile_experts", std::to_string(geometry.experts)}, {"warps", std::to_string(geometry.warps)},
                 {"k_warps", std::to_string(geometry.k_warps)}, {"constant_width", std::to_string(geometry.constant_width)},
                 {"sm_count", std::to_string(router_q8_prefill_hardware_.multiprocessors)},
                 {"rows", std::to_string(seq_len)}, {"basis", basis == RouterQ8Basis::Original ? "original" : "hadamard32"},
                 {"ownership", owned ? "partitioned" : "complete"}, {"expert_hidden_basis", "original"}});
            return true;
        }

        // Row distribution changes ownership, not the prepared representation or
        // arithmetic. This same workspace/gate owner serves both destinations.
        if (owned)
        {
            const MoERouterOwnedRowsLaunch launch{
                .partition = owned->partition,
                .capacity = seq_len, .width = d_model, .experts = num_experts,
                .top_k = top_k, .normalize = normalize_weights,
                .live_rows = device_effective_seq_len, .hidden = hidden,
                .format = gate_is_fp32 ? MoERouterPreparedFormat::FP32 :
                    (gate_type == TensorType::FP16 ? MoERouterPreparedFormat::FP16 : MoERouterPreparedFormat::BF16),
                .gate = gate_weights, .logits = d_route_logits_,
                .selected = owned->selected, .selected_bytes = owned->selected_bytes};
            const bool submitted = cuda::routeOwnedRows(launch, stream, router_q8_prefill_hardware_);
            if (submitted && gate_type == TensorType::FP16)
                PerfStatsCollector::addCounter("cuda_moe", "router_fp16_owned_operands", 1.0, {}, {},
                    {{"m", std::to_string(seq_len)}, {"activation", "fp16_rne"}, {"weight", "fp16"}, {"accumulation", "fp32"}});
            return submitted;
        }
        if (gate_type == TensorType::FP16)
            PerfStatsCollector::addCounter("cuda_moe", "router_fp16_operands", 1.0, {}, {},
                {{"m", std::to_string(seq_len)}, {"k", std::to_string(d_model)},
                 {"weight", "fp16"}, {"activation", "fp16_rne"}, {"accumulation", "fp32"}});
        const bool route_ok = gate_is_fp32
                                  ? cudaMoE_route_logits(hidden, static_cast<const float *>(gate_weights), d_route_logits_,
                                                         seq_len, d_model, num_experts,
                                                         device_ordinal_, stream)
                                  : gate_type == TensorType::FP16
                                  ? cudaMoE_route_logits_fp16(hidden, gate_weights, d_route_logits_,
                                        seq_len, d_model, num_experts, device_ordinal_, stream, device_effective_seq_len)
                                  : cudaMoE_route_logits_bf16(hidden, gate_weights, d_route_logits_,
                                                              seq_len, d_model, num_experts,
                                                              device_ordinal_, stream);
        if (!route_ok)
            return false;
        if (gate_is_fp32 && seq_len > 1)
        {
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_router_tiled_prefill_calls", 1.0, {}, {},
                {{"seq_len", std::to_string(seq_len)},
                 {"d_model", std::to_string(d_model)},
                 {"num_experts", std::to_string(num_experts)}});
        }
        if (!cudaMoE_softmax_topk(d_route_logits_, output_indices, output_weights,
                                  seq_len, num_experts, top_k, normalize_weights,
                                  device_ordinal_, getStream(),
                                  device_effective_seq_len))
            return false;
        return true;
    }

    void CUDAMoEKernel::gatherTokenBatch(const float *hidden, float *batch_buffer,
                                         const int *token_indices, int num_tokens, int d_model)
    {
        if (num_tokens <= 0)
            return;
        cudaMoE_gather_tokens(hidden, batch_buffer, token_indices,
                              num_tokens, d_model, device_ordinal_, getStream());
    }

    void CUDAMoEKernel::scatterAddWeighted(float *output, const float *expert_output,
                                           const int *token_indices, const float *weights,
                                           int num_tokens, int d_model)
    {
        if (num_tokens <= 0)
            return;
        cudaMoE_scatter_add(output, expert_output, token_indices, weights,
                            num_tokens, d_model, device_ordinal_, getStream());
    }

    void CUDAMoEKernel::sharedExpertGate(const float *input, const float *gate_inp,
                                         float *shared_output, int seq_len, int d_model)
    {
        if (seq_len <= 0)
            return;
        cudaMoE_shared_expert_gate(input, gate_inp, shared_output,
                                   seq_len, d_model, device_ordinal_, getStream());
    }

    void CUDAMoEKernel::swiGLU(float *gate, const float *up, int count)
    {
        if (count <= 0)
            return;
        cudaMoE_swiglu(gate, up, count, device_ordinal_, getStream());
    }

    void CUDAMoEKernel::weightedAdd(float *output, const float *input, float weight, int count)
    {
        if (count <= 0)
            return;
        cudaMoE_weighted_add(output, input, weight, count, device_ordinal_, getStream());
    }

    bool CUDAMoEKernel::routeWithTensorsImpl(ITensor *hidden, ITensor *gate_weights,
                                             int seq_len, int d_model, int num_experts, int top_k,
                                             bool normalize_weights,
                                             ITensor *output_indices, ITensor *output_weights,
                                             MoERoutingResult &host_result,
                                             const int *device_effective_seq_len,
                                             const char *context)
    {
        void *stream = requireStream(context);
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(hidden, device, stream, "hidden") ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights") ||
            !requireOutputOnDevice(output_indices, device, stream, "output_indices") ||
            !requireOutputOnDevice(output_weights, device, stream, "output_weights"))
            return false;

        if (seq_len <= 0 || d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[" << context << "] invalid routing shape seq_len="
                      << seq_len << " d_model=" << d_model
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }

        auto *hidden_base = requireTensor(hidden, "routeWithTensors hidden");
        auto *gate_base = requireTensor(gate_weights, "routeWithTensors gate_weights");
        auto *indices_base = requireTensor(output_indices, "routeWithTensors output_indices");
        auto *weights_base = requireTensor(output_weights, "routeWithTensors output_weights");
        if (!hidden_base || !gate_base || !indices_base || !weights_base)
            return false;

        if (!requireTensorType(hidden_base, TensorType::FP32, "hidden", "routeWithTensors") ||
            !(gate_base->native_type() == TensorType::FP16 || requireTensorTypeOneOf(gate_base, TensorType::FP32, TensorType::BF16, "gate_weights", "routeWithTensors")) ||
            !requireTensorType(indices_base, TensorType::FP32, "output_indices", "routeWithTensors") ||
            !requireTensorType(weights_base, TensorType::FP32, "output_weights", "routeWithTensors"))
            return false;

        const size_t required_hidden = static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_gate = static_cast<size_t>(num_experts) * static_cast<size_t>(d_model);
        const size_t required_topk = static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (!requireMatrixCapacity(hidden, seq_len, d_model, "hidden", "routeWithTensors") ||
            !requireMatrixCapacity(gate_weights, num_experts, d_model, "gate_weights", "routeWithTensors") ||
            !requireTensorElements(hidden, required_hidden, "hidden", "routeWithTensors") ||
            !requireTensorElements(gate_weights, required_gate, "gate_weights", "routeWithTensors") ||
            !requireTensorElements(output_indices, required_topk, "output_indices", "routeWithTensors") ||
            !requireTensorElements(output_weights, required_topk, "output_weights", "routeWithTensors"))
            return false;

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_hidden || !d_gate || !d_idx || !d_wt)
            return false;

        if (!routeCore(d_hidden, d_gate, gate_base->native_type(),
                       seq_len, d_model, num_experts, top_k, normalize_weights,
                       d_idx, d_wt,
                       device_effective_seq_len))
            return false;

        host_result.expert_indices.clear();
        host_result.expert_weights.clear();
        host_result.router_logits.clear();

        /*
         * Routing tensors are authoritative on the device for both eager
         * execution and graph replay.  Snapshotting drains the published
         * device tensors at its own explicit host-observation boundary; this
         * production routing API must not create a second host mirror.
         */
        markDeviceWritten(output_indices, device, stream);
        markDeviceWritten(output_weights, device, stream);

        return true;
    }

    bool CUDAMoEKernel::routeWithTensors(ITensor *hidden, ITensor *gate_weights,
                                         int seq_len, int d_model, int num_experts, int top_k,
                                         bool normalize_weights,
                                         ITensor *output_indices, ITensor *output_weights,
                                         MoERoutingResult &host_result)
    {
        return routeWithTensorsImpl(hidden, gate_weights,
                                    seq_len, d_model, num_experts, top_k,
                                    normalize_weights,
                                    output_indices, output_weights,
                                    host_result,
                                    nullptr,
                                    "CUDAMoEKernel::routeWithTensors");
    }

    bool CUDAMoEKernel::routeWithTensorsEffectiveSeqLen(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result,
        const int *device_effective_seq_len)
    {
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[CUDAMoEKernel::routeWithTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }
        return routeWithTensorsImpl(hidden, gate_weights,
                                    seq_len, d_model, num_experts, top_k,
                                    normalize_weights,
                                    output_indices, output_weights,
                                    host_result,
                                    device_effective_seq_len,
                                    "CUDAMoEKernel::routeWithTensorsEffectiveSeqLen");
    }

    bool CUDAMoEKernel::publishOwnedRouteRows(const MoERouterRowPublication &publication)
    {
        return cuda::publishRouterOwnedRows(publication, getStream());
    }

    bool CUDAMoEKernel::routeOwnedRowsWithTensors(
        ITensor *hidden, ITensor *gate_weights,
        const MoERouterRowPacketLayout &layout, int d_model, int num_experts,
        bool normalize_weights, ITensor *packet, std::uint64_t *selected_bytes,
        const std::int32_t *live_rows)
    {
        constexpr const char *context = "CUDAMoEKernel::routeOwnedRowsWithTensors";
        void *stream = requireStream(context);
        const auto device = deviceId();
        if (!layout.valid() || d_model <= 0 || num_experts < layout.top_k ||
            !live_rows || !selected_bytes || !hidden || !gate_weights || !packet ||
            hidden->native_type() != TensorType::FP32 ||
            !requireTensorOnDevice(hidden, device, stream, "hidden") ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights") ||
            !requireOutputOnDevice(packet, device, stream, "router packet") ||
            !requireMatrixCapacity(hidden, layout.capacity, d_model, "hidden", context) ||
            !requireMatrixCapacity(gate_weights, num_experts, d_model, "gate", context) ||
            packet->size_bytes() < layout.packetBytes() ||
            !requireCudaDevicePointer(selected_bytes, device_ordinal_, "packet extent", context, stream))
            return false;
        const MoERouterOwnedRowOutputs owned{layout.partition,
            static_cast<MoERouterSelectedRoute *>(packet->gpu_data_ptr()), selected_bytes};
        if (!routeCore(static_cast<const float *>(hidden->gpu_data_ptr()), gate_weights->gpu_data_ptr(),
                       gate_weights->native_type(), layout.capacity, d_model, num_experts, layout.top_k,
                       normalize_weights, nullptr, nullptr, live_rows, &owned))
            return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool CUDAMoEKernel::prepareRouteLaunch(
        ITensor *gate_weights,
        const MoERouteLaunchPlan &plan)
    {
        constexpr const char *kContext = "CUDAMoEKernel::prepareRouteLaunch";
        void *stream = requireStream(kContext);
        if (!stream)
            return false;
        if (!plan.valid() ||
            (plan.kind == MoERouteLaunchKind::RuntimeDecode &&
             plan.physical_rows != 1))
        {
            LOG_ERROR("[" << kContext << "] invalid route launch plan"
                      << " kind=" << static_cast<int>(plan.kind)
                      << " rows=" << plan.physical_rows
                      << " d_model=" << plan.d_model
                      << " num_experts=" << plan.num_experts
                      << " top_k=" << plan.top_k);
            return false;
        }
        if (plan.kind == MoERouteLaunchKind::RuntimeDecode)
            prepared_decode_resident_set_policy_ = plan.resident_set_policy;
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(
                gate_weights, device, stream, "gate_weights"))
        {
            return false;
        }

        auto *gate_base = requireTensor(
            gate_weights, "prepared route gate_weights");
        if (!gate_base ||
            !(gate_base->native_type() == TensorType::FP16 || requireTensorTypeOneOf(
                gate_base,
                TensorType::FP32,
                TensorType::BF16,
                "gate_weights",
                kContext)) ||
            !requireMatrixCapacity(
                gate_weights,
                plan.num_experts,
                plan.d_model,
                "gate_weights",
                kContext))
        {
            return false;
        }

        if (debugEnv().gemm.cuda_moe_router_prefill == CUDARouterPrefillArithmetic::Q8 &&
            gate_base->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[" << kContext << "] explicit CUDA Q8 prefill requires an FP32 "
                      "prepared router gate; a floating operand override cannot replace it");
            return false;
        }

        if (debugEnv().gemm.cuda_moe_router_prefill == CUDARouterPrefillArithmetic::Q8 &&
            !router_q8_prefill_hardware_.valid() &&
            !cuda::prepareRouterQ8PrefillHardware(device_ordinal_, stream, router_q8_prefill_hardware_))
        {
            LOG_ERROR("[" << kContext << "] Q8 router hardware limits must be prepared on the owning stream before capture");
            return false;
        }

        if (!prepareRequiredRouterQ8Hidden(plan.physical_rows, plan.d_model, plan.top_k))
            return false;

        const size_t logits_count =
            static_cast<size_t>(plan.physical_rows) *
            static_cast<size_t>(plan.num_experts);
        if (!ensureRouteBufferCapacity(logits_count))
        {
            LOG_ERROR("[" << kContext << "] failed to bind persistent route scratch");
            return false;
        }

        const bool uses_decode_row_math =
            plan.kind == MoERouteLaunchKind::RuntimeDecode ||
            plan.kind == MoERouteLaunchKind::DecodeEquivalentVerifier;
        const bool q8_router_requested =
            (uses_decode_row_math || debugEnv().gemm.cuda_moe_router_prefill == CUDARouterPrefillArithmetic::Q8) &&
            gate_base->native_type() == TensorType::FP32 &&
            debugEnv().gemm.cuda_moe_router_q8;
        if (!q8_router_requested)
            return true;
        if ((plan.d_model % 32) != 0)
        {
            LOG_ERROR("[" << kContext << "] Q8 decode-row routing requires "
                      "d_model to be a multiple of 32, got " << plan.d_model);
            return false;
        }
        if (!ensureGroupedGateUpDecodeCapacity(
                plan.top_k,
                plan.d_model,
                plan.physical_rows))
        {
            LOG_ERROR("[" << kContext << "] failed to bind persistent Q8 hidden-row scratch");
            return false;
        }

        const auto *q8_gate = getOrCreateQ8RouterGateCache(
            static_cast<const float *>(gate_weights->gpu_data_ptr()),
            plan.d_model,
            plan.num_experts);
        if (!q8_gate)
        {
            LOG_ERROR("[" << kContext << "] failed to publish immutable Q8 router gate");
            return false;
        }
        return true;
    }

    bool CUDAMoEKernel::routeVerifierRowsDecodeEquivalent(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        const int *device_effective_seq_len,
        DeviceMoELayerRuntime *deferred_selected_route_ledger)
    {
        constexpr const char *kContext = "CUDAMoEKernel::routeVerifierRowsDecodeEquivalent";
        void *stream = requireStream(kContext);
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(hidden, device, stream, "hidden") ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights") ||
            !requireOutputOnDevice(output_indices, device, stream, "output_indices") ||
            !requireOutputOnDevice(output_weights, device, stream, "output_weights"))
        {
            return false;
        }

        if (seq_len < 1 || d_model <= 0 ||
            num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[" << kContext << "] invalid verifier routing shape seq_len="
                          << seq_len << " d_model=" << d_model
                          << " num_experts=" << num_experts
                          << " top_k=" << top_k);
            return false;
        }

        auto *hidden_base = requireTensor(hidden, "verifier route hidden");
        auto *gate_base = requireTensor(gate_weights, "verifier route gate_weights");
        auto *indices_base = requireTensor(output_indices, "verifier route output_indices");
        auto *weights_base = requireTensor(output_weights, "verifier route output_weights");
        if (!hidden_base || !gate_base || !indices_base || !weights_base)
            return false;

        if (!requireTensorType(hidden_base, TensorType::FP32, "hidden", kContext) ||
            !(gate_base->native_type() == TensorType::FP16 || requireTensorTypeOneOf(gate_base, TensorType::FP32, TensorType::BF16, "gate_weights", kContext)) ||
            !requireTensorType(indices_base, TensorType::FP32, "output_indices", kContext) ||
            !requireTensorType(weights_base, TensorType::FP32, "output_weights", kContext))
        {
            return false;
        }

        const size_t required_hidden =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_gate =
            static_cast<size_t>(num_experts) * static_cast<size_t>(d_model);
        const size_t required_topk =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (!requireMatrixCapacity(hidden, seq_len, d_model, "hidden", kContext) ||
            !requireMatrixCapacity(gate_weights, num_experts, d_model, "gate_weights", kContext) ||
            !requireTensorElements(hidden, required_hidden, "hidden", kContext) ||
            !requireTensorElements(gate_weights, required_gate, "gate_weights", kContext) ||
            !requireTensorElements(output_indices, required_topk, "output_indices", kContext) ||
            !requireTensorElements(output_weights, required_topk, "output_weights", kContext))
        {
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_hidden || !d_gate || !d_idx || !d_wt)
        {
            LOG_ERROR("[" << kContext << "] null device pointer hidden="
                          << static_cast<const void *>(d_hidden)
                          << " gate=" << d_gate
                          << " indices=" << static_cast<void *>(d_idx)
                          << " weights=" << static_cast<void *>(d_wt));
            return false;
        }

        const bool gate_is_fp32 = (gate_base->native_type() == TensorType::FP32);
        const bool q8_router_requested = gate_is_fp32 && debugEnv().gemm.cuda_moe_router_q8;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[" << kContext << "] Q8 verifier router requires d_model to be a multiple of 32, got "
                          << d_model);
            return false;
        }

        const size_t logits_count =
            static_cast<size_t>(seq_len) * static_cast<size_t>(num_experts);
        if (!ensureRouteBufferCapacity(logits_count))
        {
            LOG_ERROR("[" << kContext << "] route scratch allocation failed");
            return false;
        }
        invalidateRouterQ8HiddenPublication();
        if (!prepareRequiredRouterQ8Hidden(seq_len, d_model, top_k))
            return false;
        if (!q8_router_requested && !publishRequiredRouterQ8Hidden(
                d_hidden, seq_len, d_model, top_k, device_effective_seq_len))
            return false;

        /*
         * Route all verifier rows as one grouped transaction while preserving
         * serial decode math inside each row.  The Q8, FP32, and BF16 kernels
         * all use expert-owned grouped blocks that share gate loads across
         * verifier rows while preserving each row's serial K/reduction order.
         */
        if (q8_router_requested)
        {
            if (!ensureGroupedGateUpDecodeCapacity(top_k, d_model, seq_len))
            {
                LOG_ERROR("[" << kContext << "] Q8 router hidden scratch unavailable");
                return false;
            }
            const auto *q8_gate = getOrCreateQ8RouterGateCache(
                static_cast<const float *>(d_gate), d_model, num_experts);
            if (!q8_gate)
            {
                LOG_ERROR("[" << kContext << "] Q8 router gate cache unavailable");
                return false;
            }
            if (!cudaMoE_gate_logits_q8_weights_decode_equivalent_rows(
                    d_hidden,
                    d_decode_hidden_int8_,
                    d_decode_hidden_scales_,
                    q8_gate->d_gate_weights_q8,
                    q8_gate->d_gate_scales,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic)))
            {
                LOG_ERROR("[" << kContext << "] grouped Q8 decode-equivalent logits failed");
                return false;
            }

            publishRouterQ8Hidden(
                d_hidden,
                seq_len,
                isCudaMoEDecodeCaptureActive(stream),
            routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        }
        else if (gate_is_fp32)
        {
            if (!cudaMoE_route_logits_decode_equivalent_rows(
                    d_hidden,
                    static_cast<const float *>(d_gate),
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    stream))
            {
                LOG_ERROR("[" << kContext << "] grouped FP32 decode-equivalent logits failed");
                return false;
            }
        }
        else if (gate_base->native_type() == TensorType::FP16)
        {
            if (!cudaMoE_route_logits_fp16(d_hidden, d_gate, d_route_logits_, seq_len, d_model,
                    num_experts, device_ordinal_, stream, device_effective_seq_len)) return false;
            PerfStatsCollector::addCounter("cuda_moe", "router_fp16_grouped_operands", 1.0, {}, {},
                {{"m", std::to_string(seq_len)}, {"activation", "fp16_rne"}, {"weight", "fp16"}, {"accumulation", "fp32"}});
        }
        else if (!cudaMoE_route_logits_bf16_decode_equivalent_rows(
                     d_hidden,
                     d_gate,
                     d_route_logits_,
                     seq_len,
                     d_model,
                     num_experts,
                     device_ordinal_,
                     stream))
        {
            LOG_ERROR("[" << kContext << "] grouped BF16 decode-equivalent logits failed");
            return false;
        }

        if (!cudaMoE_softmax_topk(
                d_route_logits_,
                d_idx,
                d_wt,
                seq_len,
                num_experts,
                top_k,
                normalize_weights,
                device_ordinal_,
                stream,
                device_effective_seq_len,
                static_cast<void *>(deferred_selected_route_ledger)))
        {
            LOG_ERROR("[" << kContext << "] grouped decode-equivalent softmax/top-k failed");
            return false;
        }

        markDeviceWritten(output_indices, device, stream);
        markDeviceWritten(output_weights, device, stream);
        PerfStatsCollector::addCounter(
            "kernel",
            "cuda_moe_router_decode_equivalent_runtime_m_calls",
            1.0,
            {},
            {},
            {{"seq_len", std::to_string(seq_len)},
             {"d_model", std::to_string(d_model)},
             {"num_experts", std::to_string(num_experts)},
             {"top_k", std::to_string(top_k)},
             {"route", q8_router_requested ? "grouped_decode_equivalent_q8"
                                            : "grouped_decode_equivalent"}});
        return true;
    }

    bool CUDAMoEKernel::decodeRouteSelect(DeviceMoELayerRuntime *runtime_layer,
                                          ITensor *hidden, ITensor *gate_weights,
                                          int d_model, int num_experts, int top_k,
                                          bool normalize_weights,
                                          ITensor *output_indices, ITensor *output_weights,
                                          bool write_legacy_outputs,
                                          bool update_runtime_histogram,
                                          const int32_t *absolute_position_ids_device,
                                          RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t *active_row_count_device)
    {
        void *stream = requireStream("CUDAMoEKernel::decodeRouteSelect");
        const DeviceId device = deviceId();
        if (!runtime_layer || !hidden || !gate_weights)
            return false;
        if (!requireTensorOnDevice(hidden, device, stream, "hidden") ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights"))
            return false;

        auto *hidden_base = requireTensor(hidden, "decodeRouteSelect hidden");
        auto *gate_base = requireTensor(gate_weights, "decodeRouteSelect gate_weights");
        if (!hidden_base || !gate_base)
            return false;
        if (hidden_base->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelect] hidden must be FP32, got "
                      << tensorTypeName(hidden_base->native_type()));
            return false;
        }
        const TensorType gate_type = gate_base->native_type();
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        const bool gate_is_bf16 = (gate_type == TensorType::BF16);
        if (!gate_is_fp32 && !gate_is_bf16 && gate_type != TensorType::FP16)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelect] CUDA runtime router supports FP32, FP16 or BF16 gate weights, got "
                      << tensorTypeName(gate_type));
            return false;
        }

        float *legacy_indices = nullptr;
        float *legacy_weights = nullptr;
        if (write_legacy_outputs)
        {
            if (!output_indices || !output_weights ||
                !requireOutputOnDevice(output_indices, device, stream, "output_indices") ||
                !requireOutputOnDevice(output_weights, device, stream, "output_weights"))
                return false;
            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        if (!d_hidden || !d_gate)
            return false;

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts)))
            return false;
        if (!requireAlignedPointer(d_hidden, 16, "hidden", "decodeRouteSelect") ||
            !requireAlignedPointer(d_route_logits_, 16, "route logits", "decodeRouteSelect"))
            return false;
        if (gate_is_fp32 && !requireAlignedPointer(d_gate, 16, "FP32 gate", "decodeRouteSelect"))
            return false;

        bool logits_ready = false;
        const bool q8_router_requested = gate_is_fp32 && debugEnv().gemm.cuda_moe_router_q8;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelect] Q8 router requires d_model to be a multiple of 32, got "
                      << d_model);
            return false;
        }
        if (q8_router_requested)
        {
            if (!tryRouteDecodeLogitsQ8(
                d_hidden, static_cast<const float *>(d_gate),
                d_model, num_experts, top_k, "CUDAMoEKernel::decodeRouteSelect"))
            {
                return false;
            }
            logits_ready = true;
        }
        if (!logits_ready)
        {
            if (gate_type == TensorType::FP16)
                PerfStatsCollector::addCounter("cuda_moe", "router_fp16_operands", 1.0, {}, {},
                    {{"m", "1"}, {"k", std::to_string(d_model)},
                     {"weight", "fp16"}, {"activation", "fp16_rne"}, {"accumulation", "fp32"}});
            invalidateRouterQ8HiddenPublication();
            // Runtime M1 can serve the grouped MTP bridge. Preserve floating
            // router arithmetic while honoring its cold shared-consumer Q8
            // demand; ordinary decode with no such consumer launches nothing.
            if (!publishRequiredRouterQ8Hidden(d_hidden, /*rows=*/1, d_model,
                    top_k, active_row_count_device))
                return false;
            const bool route_ok = gate_is_fp32
                                      ? cudaMoE_route_logits(d_hidden, static_cast<const float *>(d_gate), d_route_logits_,
                                                             /*seq_len=*/1, d_model, num_experts,
                                                             device_ordinal_, stream)
                                      : gate_type == TensorType::FP16
                                      ? cudaMoE_route_logits_fp16(d_hidden, d_gate, d_route_logits_,
                                            1, d_model, num_experts, device_ordinal_, stream)
                                      : cudaMoE_route_logits_bf16(d_hidden, d_gate, d_route_logits_,
                                                                  /*seq_len=*/1, d_model, num_experts,
                                                                  device_ordinal_, stream);
            if (!route_ok)
                return false;
        }
        const bool unique_owner_route =
            row_execution_policy == RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
            prepared_decode_resident_set_policy_ ==
            RoutedExpertResidentSetPolicy::UniqueOwner;
        const bool route_ready = unique_owner_route
                                     ? cudaMoE_softmax_topk_decode_unique_owner_runtime(
                                           d_route_logits_,
                                           runtime_layer,
                                           legacy_indices,
                                           legacy_weights,
                                           num_experts,
                                           top_k,
                                           normalize_weights,
                                           write_legacy_outputs,
                                           update_runtime_histogram,
                                           device_ordinal_,
                                           stream,
                                           active_row_count_device)
                                     : cudaMoE_softmax_topk_decode_runtime(
                                           d_route_logits_,
                                           runtime_layer,
                                           legacy_indices,
                                           legacy_weights,
                                           num_experts,
                                           top_k,
                                           normalize_weights,
                                           write_legacy_outputs,
                                           update_runtime_histogram,
                                           row_execution_policy,
                                           nullptr,
                                           nullptr,
                                           0u,
                                           nullptr,
                                           nullptr,
                                           0u,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           -1,
                                           1u,
                                           absolute_position_ids_device,
                                           device_ordinal_,
                                           stream,
                                           active_row_count_device);
        if (!route_ready)
            return false;
        if (write_legacy_outputs)
        {
            markDeviceWritten(output_indices, device, stream);
            markDeviceWritten(output_weights, device, stream);
        }
        return true;
    }

    bool CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply(
        DeviceMoELayerRuntime *runtime_layers,
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *gate_weights,
        int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        const DeviceMoERebalancePlanEntry *rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *rebalance_command_header,
        const DeviceMoEExpertDirectoryEntry *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        const DeviceMoERebalanceConfig &rebalance_config,
        DeviceMoERebalanceApplyStatus *rebalance_apply_status,
        DeviceMoERebalanceGraphControllerState *rebalance_controller_state,
        int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count,
        const int32_t *absolute_position_ids_device,
        RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t *active_row_count_device)
    {
        void *stream = requireStream("CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply");
        const DeviceId device = deviceId();
        if (!runtime_layers || !runtime_layer || !hidden || !gate_weights ||
            !rebalance_plan_entries || rebalance_plan_capacity == 0 ||
            !rebalance_command_header || !rebalance_apply_status ||
            !rebalance_controller_state ||
            !validateDeviceMoERebalanceConfig(rebalance_config))
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply] invalid runtime or rebalance binding");
            return false;
        }
        if (!requireTensorOnDevice(hidden, device, stream, "hidden") ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights"))
            return false;

        auto *hidden_base = requireTensor(hidden, "decodeRouteSelectWithReadyRebalanceApply hidden");
        auto *gate_base = requireTensor(gate_weights, "decodeRouteSelectWithReadyRebalanceApply gate_weights");
        if (!hidden_base || !gate_base)
            return false;
        if (hidden_base->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply] hidden must be FP32, got "
                      << tensorTypeName(hidden_base->native_type()));
            return false;
        }
        const TensorType gate_type = gate_base->native_type();
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        const bool gate_is_bf16 = (gate_type == TensorType::BF16);
        if (!gate_is_fp32 && !gate_is_bf16 && gate_type != TensorType::FP16)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply] CUDA runtime router supports FP32, FP16 or BF16 gate weights, got "
                      << tensorTypeName(gate_type));
            return false;
        }

        float *legacy_indices = nullptr;
        float *legacy_weights = nullptr;
        if (write_legacy_outputs)
        {
            if (!output_indices || !output_weights ||
                !requireOutputOnDevice(output_indices, device, stream, "output_indices") ||
                !requireOutputOnDevice(output_weights, device, stream, "output_weights"))
                return false;
            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        if (!d_hidden || !d_gate)
            return false;

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts)))
            return false;
        if (!requireAlignedPointer(d_hidden, 16, "hidden", "decodeRouteSelectWithReadyRebalanceApply") ||
            !requireAlignedPointer(d_route_logits_, 16, "route logits", "decodeRouteSelectWithReadyRebalanceApply"))
            return false;
        if (gate_is_fp32 && !requireAlignedPointer(d_gate, 16, "FP32 gate", "decodeRouteSelectWithReadyRebalanceApply"))
            return false;

        bool logits_ready = false;
        const bool q8_router_requested = gate_is_fp32 && debugEnv().gemm.cuda_moe_router_q8;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router requires d_model to be a multiple of 32, got "
                      << d_model);
            return false;
        }
        if (q8_router_requested)
        {
            if (!tryRouteDecodeLogitsQ8(
                d_hidden, static_cast<const float *>(d_gate),
                d_model, num_experts, top_k,
                "CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply"))
            {
                return false;
            }
            logits_ready = true;
        }
        if (!logits_ready)
        {
            if (gate_type == TensorType::FP16)
                PerfStatsCollector::addCounter("cuda_moe", "router_fp16_operands", 1.0, {}, {},
                    {{"m", "1"}, {"k", std::to_string(d_model)},
                     {"weight", "fp16"}, {"activation", "fp16_rne"}, {"accumulation", "fp32"}});
            invalidateRouterQ8HiddenPublication();
            // Runtime M1 can serve the grouped MTP bridge. Preserve floating
            // router arithmetic while honoring its cold shared-consumer Q8
            // demand; ordinary decode with no such consumer launches nothing.
            if (!publishRequiredRouterQ8Hidden(d_hidden, /*rows=*/1, d_model,
                    top_k, active_row_count_device))
                return false;
            const bool route_ok = gate_is_fp32
                                      ? cudaMoE_route_logits(d_hidden, static_cast<const float *>(d_gate), d_route_logits_,
                                                             /*seq_len=*/1, d_model, num_experts,
                                                             device_ordinal_, stream)
                                      : gate_type == TensorType::FP16
                                      ? cudaMoE_route_logits_fp16(d_hidden, d_gate, d_route_logits_,
                                            1, d_model, num_experts, device_ordinal_, stream)
                                      : cudaMoE_route_logits_bf16(d_hidden, d_gate, d_route_logits_,
                                                                  /*seq_len=*/1, d_model, num_experts,
                                                                  device_ordinal_, stream);
            if (!route_ok)
                return false;
        }
        if (!cudaMoE_softmax_topk_decode_runtime(
                d_route_logits_,
                runtime_layer,
                legacy_indices, legacy_weights,
                num_experts, top_k, normalize_weights,
                write_legacy_outputs, update_runtime_histogram,
                row_execution_policy,
                runtime_layers,
                rebalance_plan_entries,
                rebalance_plan_capacity,
                rebalance_command_header,
                rebalance_local_transfer_slots,
                rebalance_local_transfer_slot_count,
                &rebalance_config,
                rebalance_apply_status,
                rebalance_controller_state,
                rebalance_target_layer,
                rebalance_command_buffer_count,
                absolute_position_ids_device,
                device_ordinal_, stream,
                active_row_count_device))
            return false;

        if (write_legacy_outputs)
        {
            markDeviceWritten(output_indices, device, stream);
            markDeviceWritten(output_weights, device, stream);
        }
        return true;
    }

    bool CUDAMoEKernel::runDeviceRebalanceController(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const uint64_t *gathered_histograms,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceWaveState *wave_state,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoELLEPLayerPlanScratch *llep_layer_plans,
        DeviceMoEPlacementBank *placement_plan_scratch)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::runDeviceRebalanceController] invalid device rebalance config");
            return false;
        }
        const bool deferred_planning = hasDeviceMoERebalanceFlag(
            config.flags,
            DeviceMoERebalanceFlags::DeferRuntimeApply);
        if (!runtime_layers || !gathered_histograms || !status ||
            !controller_state ||
            (deferred_planning && !placement_plan_scratch))
        {
            LOG_ERROR("[CUDAMoEKernel::runDeviceRebalanceController] runtime layers, gathered histograms, status, controller state, and placement planner scratch must be non-null");
            return false;
        }
        if (config.routed_assignment_policy ==
                kDeviceMoERebalanceAssignmentLeastLoadedResident &&
            !llep_layer_plans)
        {
            LOG_ERROR("[CUDAMoEKernel::runDeviceRebalanceController] LLEP requires graph-lifetime parallel planner scratch");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "runDeviceRebalanceController");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "runDeviceRebalanceController"))
            return false;

        return cudaMoE_device_rebalance_controller(
            runtime_layers,
            reinterpret_cast<const unsigned long long *>(gathered_histograms),
            status,
            &config,
            plan_entries,
            plan_count,
            plan_capacity,
            payload_slot_capacity,
            command_header,
            wave_state,
            controller_state,
            llep_layer_plans,
            placement_plan_scratch,
            command_buffer_count,
            local_transfer_slots,
            local_transfer_slot_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::finalizeMoEOverlayRebalancePublication(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        std::uint32_t layer_count,
        std::uint32_t expert_count,
        DeviceMoEOverlayEpochControl *control,
        std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *reservation_and_publication_status,
        const DeviceMoERebalanceApplyStatus *apply_status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_headers,
        std::uint32_t command_buffer_count,
        const DeviceMoERebalancePlanEntry *plan_entries,
        std::uint32_t plan_capacity,
        std::uint64_t estimated_expert_bytes,
        DeviceMoERebalanceMovementJournalView journal)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finalizeMoEOverlayRebalancePublication");
        if (!runtime_layers || layer_count == 0u || expert_count == 0u ||
            !control || !candidate_epoch ||
            !reservation_and_publication_status || !apply_status || !stream ||
            !controller_state || !command_headers || command_buffer_count == 0u ||
            command_buffer_count > 2u || !plan_entries || plan_capacity == 0u ||
            !journal.state || !journal.waves || !journal.edges ||
            journal.wave_capacity == 0u || journal.edge_capacity == 0u)
        {
            LOG_ERROR(
                "[CUDAMoEKernel::finalizeMoEOverlayRebalancePublication] "
                "runtime family, epoch state, apply status, and explicit stream are required");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "finalizeMoEOverlayRebalancePublication"))
        {
            return false;
        }
        return cudaMoE_finalize_overlay_rebalance_publication(
            runtime_layers,
            layer_count,
            expert_count,
            control,
            candidate_epoch,
            reservation_and_publication_status,
            apply_status,
            controller_state,
            command_headers,
            command_buffer_count,
            plan_entries,
            plan_capacity,
            estimated_expert_bytes,
            journal,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceHistograms(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        uint64_t *local_histograms,
        const DeviceMoERebalanceConfig &config,
        const DeviceMoERebalanceWaveState *wave_state,
        const DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count,
        uint32_t histogram_source_mask,
        uint64_t *previous_activation_counts)
    {
        if (!validateDeviceMoERebalanceConfig(config) ||
            !moe_runtime_abi::validHistogramSourceMask(
                histogram_source_mask))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceHistograms] invalid device rebalance config or histogram source mask");
            return false;
        }
        if (!runtime_layers || !local_histograms)
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceHistograms] runtime layers and local histograms must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceHistograms");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceHistograms"))
            return false;

        return cudaMoE_pack_rebalance_histograms(
            runtime_layers,
            reinterpret_cast<unsigned long long *>(local_histograms),
            &config,
            wave_state,
            controller_state,
            command_buffer_count,
            histogram_source_mask,
            reinterpret_cast<unsigned long long *>(
                previous_activation_counts),
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceDirectory(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        DeviceMoEExpertDirectoryEntry *local_directory,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceDirectory] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !local_directory)
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceDirectory] runtime layers and local directory must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceDirectory");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceDirectory"))
            return false;

        return cudaMoE_pack_rebalance_directory(
            runtime_layers,
            local_directory,
            &config,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceSourceDescriptors(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *command_headers,
        uint32_t plan_capacity,
        DeviceMoEExpertDirectoryEntry *local_source_descriptors,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceSourceDescriptors] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || !command_headers ||
            !local_source_descriptors)
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceSourceDescriptors] runtime, projected plans, headers, and source descriptor output must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceSourceDescriptors");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceSourceDescriptors"))
            return false;

        return cudaMoE_pack_rebalance_source_descriptors(
            runtime_layers,
            plan_entries,
            command_headers,
            plan_capacity,
            local_source_descriptors,
            &config,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::projectDeviceRebalanceDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntry *local_plan_entries,
        DeviceMoERebalanceCommandBufferHeader *local_command_headers,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceWaveState *gathered_wave_states,
        DeviceMoERebalanceWaveState *local_wave_states,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::projectDeviceRebalanceDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_plan_entries || !local_command_headers ||
            !runtime_layers || !local_transfer_slots ||
            !transfer_slot_claim_index ||
            plan_capacity == 0 || local_transfer_slot_count == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::projectDeviceRebalanceDomainCommands] gathered/local commands, runtime layers, and transfer directory must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "projectDeviceRebalanceDomainCommands");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "projectDeviceRebalanceDomainCommands"))
            return false;

        return cudaMoE_project_rebalance_domain_commands(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_plan_entries,
            local_command_headers,
            &config,
            status,
            payload_slot_capacity,
            command_buffer_count,
            gathered_wave_states,
            local_wave_states,
            runtime_layers,
            local_transfer_slots,
            local_transfer_slot_count,
            transfer_slot_claim_index,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::projectPrefillLeastLoadedDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntry *local_plan_entries,
        uint32_t *local_plan_count,
        DeviceMoERebalanceCommandBufferHeader *local_command_header,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        uint32_t payload_slot_capacity,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::projectPrefillLeastLoadedDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_plan_entries || !local_plan_count || !local_command_header ||
            !status || !runtime_layers || !local_transfer_slots ||
            !transfer_slot_claim_index ||
            plan_capacity == 0 || payload_slot_capacity == 0 ||
            local_transfer_slot_count == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::projectPrefillLeastLoadedDomainCommands] gathered commands, local command output, runtime layers, transfer directory, status, and payload capacity are required");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "projectPrefillLeastLoadedDomainCommands");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "projectPrefillLeastLoadedDomainCommands"))
            return false;

        return cudaMoE_project_prefill_llep_domain_commands(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_plan_entries,
            local_plan_count,
            local_command_header,
            &config,
            status,
            payload_slot_capacity,
            command_buffer_count,
            runtime_layers,
            local_transfer_slots,
            local_transfer_slot_count,
            transfer_slot_claim_index,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoELayerRuntime *runtime_layer,
        DeviceMoERebalancePlanEntry *mirrored_plan_entries,
        DeviceMoERebalanceCommandBufferHeader *mirrored_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!runtime_layer || !mirrored_plan_entries ||
            !mirrored_command_headers || !status || plan_capacity == 0 ||
            payload_slot_capacity == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] runtime, mirrored command buffers, status, and payload capacity are required");
            return false;
        }
        if (layer_idx >= config.num_layers)
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] layer index out of range"
                      << " layer=" << layer_idx
                      << " num_layers=" << config.num_layers);
            return false;
        }
        void *stream = explicitMoELaunchStream(
            launch,
            "materializePrefillLeastLoadedMirroredDomainCommands");
        if (!stream)
            return false;
        if (!setMoEDevice(
                device_ordinal_,
                "materializePrefillLeastLoadedMirroredDomainCommands"))
        {
            return false;
        }

        return cudaMoE_materialize_prefill_llep_mirrored_domain_commands(
            runtime_layer,
            mirrored_plan_entries,
            mirrored_command_headers,
            plan_capacity,
            status,
            &config,
            payload_slot_capacity,
            layer_idx,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoELayerRuntime *runtime_layer,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands] invalid device rebalance config");
            return false;
        }
        if (!runtime_layer || !plan_entries || !plan_count || !command_header || !status ||
            plan_capacity == 0 || payload_slot_capacity == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands] runtime, command buffers, status, and payload capacity are required");
            return false;
        }
        if (layer_idx >= config.num_layers)
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands] layer index out of range"
                      << " layer=" << layer_idx << " num_layers=" << config.num_layers);
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "materializePrefillLeastLoadedTransferCommands");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "materializePrefillLeastLoadedTransferCommands"))
            return false;

        return cudaMoE_materialize_prefill_llep_transfer_commands(
            runtime_layer,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            status,
            &config,
            payload_slot_capacity,
            layer_idx,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceCompactPayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_source_descriptors,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceCompactPayloads] invalid device rebalance config");
            return false;
        }
        if (!plan_entries || !command_headers ||
            !local_source_descriptors || !local_payload || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntry))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceCompactPayloads] plans, headers, source descriptors, payload, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceCompactPayloads");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceCompactPayloads"))
            return false;

        return cudaMoE_pack_rebalance_compact_payloads(
            plan_entries,
            command_headers,
            plan_capacity,
            local_source_descriptors,
            local_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::applyDeviceRebalanceArrivals(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        int target_layer)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::applyDeviceRebalanceArrivals] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !local_transfer_slots || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::applyDeviceRebalanceArrivals] runtime, plan, transfer slots, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "applyDeviceRebalanceArrivals");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "applyDeviceRebalanceArrivals"))
            return false;

        return cudaMoE_apply_rebalance_arrivals(
            runtime_layers,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            target_layer,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceCollectivePayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_directory,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceCollectivePayloads] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_directory || !local_payload || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceCollectivePayloads] gathered plans, headers, local directory, payload, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceCollectivePayloads");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceCollectivePayloads"))
            return false;

        return cudaMoE_pack_rebalance_collective_payloads(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_directory,
            local_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::unpackDeviceRebalanceCollectivePayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        const uint8_t *gathered_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::unpackDeviceRebalanceCollectivePayloads] invalid device rebalance config");
            return false;
        }
        if (!plan_entries || (!plan_count && !command_header) ||
            !gathered_payload || !local_transfer_slots || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntry))
        {
            LOG_ERROR("[CUDAMoEKernel::unpackDeviceRebalanceCollectivePayloads] plan, gathered payload, transfer slots, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "unpackDeviceRebalanceCollectivePayloads");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "unpackDeviceRebalanceCollectivePayloads"))
            return false;

        return cudaMoE_unpack_rebalance_collective_payloads(
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            gathered_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::initializeDeviceRebalanceGraphController(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::initializeDeviceRebalanceGraphController] invalid device rebalance config");
            return false;
        }
        if (!controller_state)
        {
            LOG_ERROR("[CUDAMoEKernel::initializeDeviceRebalanceGraphController] controller state must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "initializeDeviceRebalanceGraphController");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "initializeDeviceRebalanceGraphController"))
            return false;

        return cudaMoE_init_rebalance_graph_controller_state(
            controller_state,
            &config,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::resetDeviceRebalanceGraphTransactionForRequest(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_headers,
        DeviceMoERebalanceWaveState *wave_states,
        uint32_t *plan_counts,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::resetDeviceRebalanceGraphTransactionForRequest] invalid device rebalance config");
            return false;
        }
        if (!controller_state || !command_headers || !wave_states ||
            !plan_counts || command_buffer_count == 0u ||
            command_buffer_count > 2u)
        {
            LOG_ERROR("[CUDAMoEKernel::resetDeviceRebalanceGraphTransactionForRequest] transaction buffers must be non-null and command-buffer count must be one or two");
            return false;
        }
        void *stream = explicitMoELaunchStream(
            launch,
            "resetDeviceRebalanceGraphTransactionForRequest");
        if (!stream)
            return false;
        if (!setMoEDevice(
                device_ordinal_,
                "resetDeviceRebalanceGraphTransactionForRequest"))
        {
            return false;
        }

        return cudaMoE_reset_rebalance_graph_transaction_for_request(
            controller_state,
            command_headers,
            wave_states,
            plan_counts,
            command_buffer_count,
            &config,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::publishDeviceRebalanceTransferComplete(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        const DeviceMoERebalanceWaveState *wave_state,
        const DeviceMoERebalanceApplyStatus *copy_status,
        const DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t plan_capacity,
        const DeviceMoERebalanceApplyStatus *gathered_copy_status,
        const DeviceMoERebalanceConfig &config,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::publishDeviceRebalanceTransferComplete] invalid device rebalance config");
            return false;
        }
        if (!controller_state || !command_header || !wave_state || !copy_status ||
            !plan_entries || plan_capacity == 0u || !gathered_copy_status)
        {
            LOG_ERROR("[CUDAMoEKernel::publishDeviceRebalanceTransferComplete] controller state, command header, wave state, copy status, plan entries, and gathered copy status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "publishDeviceRebalanceTransferComplete");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "publishDeviceRebalanceTransferComplete"))
            return false;

        return cudaMoE_publish_rebalance_transfer_complete(
            controller_state,
            command_header,
            wave_state,
            copy_status,
            plan_entries,
            plan_capacity,
            gathered_copy_status,
            &config,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::applyReadyDeviceRebalanceWave(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        int target_layer,
        uint32_t command_buffer_count,
        const DeviceMoEOverlayEpochStatus *overlay_reservation_status)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::applyReadyDeviceRebalanceWave] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !status || !controller_state)
        {
            LOG_ERROR("[CUDAMoEKernel::applyReadyDeviceRebalanceWave] runtime, plan, status, controller state, and command ABI must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "applyReadyDeviceRebalanceWave");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "applyReadyDeviceRebalanceWave"))
            return false;

        return cudaMoE_apply_ready_rebalance_wave(
            runtime_layers,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            controller_state,
            target_layer,
            command_buffer_count,
            overlay_reservation_status,
            device_ordinal_,
            stream);
    }

    void CUDAMoEKernel::zeroBuffer(ITensor *tensor, size_t bytes)
    {
        void *stream = requireStream("CUDAMoEKernel::zeroBuffer");
        if (!requireOutputOnDevice(tensor, deviceId(), stream, "zeroBuffer"))
            return;
        void *ptr = tensor->gpu_data_ptr();
        cudaError_t err = cudaMemsetAsync(ptr, 0, bytes, static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::zeroBuffer] cudaMemsetAsync failed: " << cudaGetErrorString(err));
            return;
        }
        markDeviceWritten(tensor, deviceId(), stream);
    }

    void CUDAMoEKernel::gatherTokenBatchFromTensors(ITensor *hidden, ITensor *batch_buffer,
                                                    const int *host_token_indices, int num_tokens, int d_model)
    {
        if (num_tokens <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::gatherTokenBatchFromTensors");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(hidden, device, stream, "hidden") ||
            !requireOutputOnDevice(batch_buffer, device, stream, "batch_buffer") ||
            !ensureStagingCapacity(num_tokens))
            return;

        cudaError_t err = cudaMemcpyAsync(d_staging_indices_, host_token_indices,
                                          static_cast<size_t>(num_tokens) * sizeof(int),
                                          cudaMemcpyHostToDevice,
                                          static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return;
        gatherTokenBatch(static_cast<const float *>(hidden->gpu_data_ptr()),
                         static_cast<float *>(batch_buffer->gpu_data_ptr()),
                         d_staging_indices_, num_tokens, d_model);
        markDeviceWritten(batch_buffer, device, stream);
    }

    bool CUDAMoEKernel::copyTokenRowFromTensor(
        ITensor *source, ITensor *row_buffer,
        int row_index, int row_width)
    {
        if (row_index < 0 || row_width <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::copyTokenRowFromTensor");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(source, device, stream, "source") ||
            !requireOutputOnDevice(row_buffer, device, stream, "row_buffer"))
        {
            return false;
        }

        const auto *src = static_cast<const float *>(source->gpu_data_ptr());
        auto *dst = static_cast<float *>(row_buffer->gpu_data_ptr());
        if (!cudaMoE_copy_token_row(src, dst, row_index, row_width, device.ordinal, stream))
            return false;

        markDeviceWritten(row_buffer, device, stream);
        return true;
    }

    void CUDAMoEKernel::scatterAddWeightedFromTensors(ITensor *output, ITensor *expert_output,
                                                      const int *host_token_indices, const float *host_weights,
                                                      int num_tokens, int d_model)
    {
        if (num_tokens <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::scatterAddWeightedFromTensors");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(output, device, stream, "output") ||
            !requireTensorOnDevice(expert_output, device, stream, "expert_output") ||
            !ensureStagingCapacity(num_tokens))
            return;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaMemcpyAsync(d_staging_indices_, host_token_indices,
                                          static_cast<size_t>(num_tokens) * sizeof(int),
                                          cudaMemcpyHostToDevice, cuda_stream);
        if (err != cudaSuccess)
            return;
        err = cudaMemcpyAsync(d_staging_weights_, host_weights,
                              static_cast<size_t>(num_tokens) * sizeof(float),
                              cudaMemcpyHostToDevice, cuda_stream);
        if (err != cudaSuccess)
            return;

        scatterAddWeighted(static_cast<float *>(output->gpu_data_ptr()),
                           static_cast<const float *>(expert_output->gpu_data_ptr()),
                           d_staging_indices_, d_staging_weights_, num_tokens, d_model);
        markDeviceWritten(output, device, stream);
    }

    bool CUDAMoEKernel::writeTokenRowToTensor(
        ITensor *destination, ITensor *row_buffer,
        int row_index, int row_width)
    {
        if (row_index < 0 || row_width <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::writeTokenRowToTensor");
        const DeviceId device = deviceId();
        if (!requireOutputOnDevice(destination, device, stream, "destination") ||
            !requireTensorOnDevice(row_buffer, device, stream, "row_buffer"))
        {
            return false;
        }

        auto *dst = static_cast<float *>(destination->gpu_data_ptr());
        const auto *src = static_cast<const float *>(row_buffer->gpu_data_ptr());
        if (!cudaMoE_write_token_row(dst, src, row_index, row_width, device.ordinal, stream))
            return false;

        markDeviceWritten(destination, device, stream);
        return true;
    }

    void CUDAMoEKernel::sharedExpertGateFromTensors(ITensor *input, ITensor *gate_inp, ITensor *shared_output,
                                                    int seq_len, int d_model)
    {
        if (seq_len <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::sharedExpertGateFromTensors");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(input, device, stream, "input") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output"))
            return;
        sharedExpertGate(static_cast<const float *>(input->gpu_data_ptr()),
                         static_cast<const float *>(gate_inp->gpu_data_ptr()),
                         static_cast<float *>(shared_output->gpu_data_ptr()),
                         seq_len, d_model);
        markDeviceWritten(shared_output, device, stream);
    }

    bool CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (seq_len <= 0)
            return true;
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(input, device, stream, "input") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output"))
            return false;

        const auto *in = static_cast<const float *>(input->gpu_data_ptr());
        const auto *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        auto *so = static_cast<float *>(shared_output->gpu_data_ptr());
        if (!in || !gi || !so)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] null device pointer");
            return false;
        }

        if (!cudaMoE_shared_expert_gate_effective_seq_len(
                in, gi, so, seq_len, d_model, device_effective_seq_len,
                device_ordinal_, stream))
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] kernel launch failed");
            return false;
        }
        markDeviceWritten(shared_output, device, stream);
        return true;
    }

    void CUDAMoEKernel::sharedExpertGateAddFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model)
    {
        if (seq_len <= 0)
            return;

        void *stream = requireStream("CUDAMoEKernel::sharedExpertGateAddFromTensors");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(input, device, stream, "input") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output") ||
            !requireTensorOnDevice(routed_residual, device, stream, "routed_residual") ||
            !requireOutputOnDevice(combined_output, device, stream, "combined_output"))
            return;

        const auto *in = static_cast<const float *>(input->gpu_data_ptr());
        const auto *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        auto *so = static_cast<float *>(shared_output->gpu_data_ptr());
        const auto *rr = static_cast<const float *>(routed_residual->gpu_data_ptr());
        auto *co = static_cast<float *>(combined_output->gpu_data_ptr());
        if (!in || !gi || !so || !rr || !co)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensors] null device pointer");
            return;
        }

        if (!cudaMoE_shared_expert_gate_add(in, gi, so, rr, co,
                                            seq_len, d_model, device_ordinal_, stream))
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensors] fused gate-add kernel launch failed");
            return;
        }
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
    }

    bool CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (seq_len <= 0)
            return true;
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(input, device, stream, "input") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output") ||
            !requireTensorOnDevice(routed_residual, device, stream, "routed_residual") ||
            !requireOutputOnDevice(combined_output, device, stream, "combined_output"))
            return false;

        const auto *in = static_cast<const float *>(input->gpu_data_ptr());
        const auto *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        auto *so = static_cast<float *>(shared_output->gpu_data_ptr());
        const auto *rr = static_cast<const float *>(routed_residual->gpu_data_ptr());
        auto *co = static_cast<float *>(combined_output->gpu_data_ptr());
        if (!in || !gi || !so || !rr || !co)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] null device pointer");
            return false;
        }

        if (!cudaMoE_shared_expert_gate_add_effective_seq_len(
                in, gi, so, rr, co, seq_len, d_model,
                device_effective_seq_len, device_ordinal_, stream))
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] fused gate-add kernel launch failed");
            return false;
        }
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
        return true;
    }

    void CUDAMoEKernel::swiGLUFromTensors(ITensor *gate, ITensor *up, int count)
    {
        if (count <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::swiGLUFromTensors");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(gate, device, stream, "gate") ||
            !requireTensorOnDevice(up, device, stream, "up"))
            return;
        swiGLU(static_cast<float *>(gate->gpu_data_ptr()),
               static_cast<const float *>(up->gpu_data_ptr()), count);
        markDeviceWritten(gate, device, stream);
    }

    void CUDAMoEKernel::weightedAddFromTensors(ITensor *output, ITensor *input, float weight, int count)
    {
        if (count <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::weightedAddFromTensors");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(output, device, stream, "output") ||
            !requireTensorOnDevice(input, device, stream, "input"))
            return;
        weightedAdd(static_cast<float *>(output->gpu_data_ptr()),
                    static_cast<const float *>(input->gpu_data_ptr()), weight, count);
        markDeviceWritten(output, device, stream);
    }

    /**
     * @brief Reset and publish stable expert grouping on one retained stream.
     * @param d_routing_indices Original expert IDs, including -1 inactive slots.
     * @param d_routing_weights Unmodified per-route probability bits.
     * @param seq_len Admitted token extent, including a padded bucket tail.
     * @param num_experts Number of experts represented by counts and offsets.
     * @param top_k Route slots per token.
     * @param d_expert_offsets Output exclusive count prefix.
     * @param d_expert_counts Output integer expert counts.
     * @param d_grouped_token_indices Output stable compact route IDs.
     * @param d_grouped_weights Output compact probabilities; unused slots are zero.
     * @return False if initialization or any grouping launch fails.
     */
    bool CUDAMoEKernel::groupTokensByExpertDevice(const int *d_routing_indices,
                                                  const float *d_routing_weights,
                                                  int seq_len, int num_experts, int top_k,
                                                  int *d_expert_offsets, int *d_expert_counts,
                                                  int *d_grouped_token_indices, float *d_grouped_weights)
    {
        grouped_prefill_grouping_.invalidate();
        runtime_gate_plan_.reset();
        const int total_slots = seq_len * top_k;
        void *stream = requireStream("CUDAMoEKernel::groupTokensByExpertDevice");
        // One graph node resets every publication, including padded/remote
        // slots. Preserve -1 maps and exact-zero compact tails across replay.
        if (!cudaMoE_initialize_grouping({
                .expert_counts = d_expert_counts,
                .write_heads = d_group_write_heads_,
                .original_to_grouped = d_group_original_to_grouped_,
                .original_expert_ids = d_group_original_expert_ids_,
                .grouped_token_indices = d_grouped_token_indices,
                .grouped_weights = d_grouped_weights,
                .num_experts = num_experts,
                .total_slots = total_slots}, device_ordinal_, stream))
            return false;
        return cudaMoE_count_per_expert(d_routing_indices, d_expert_counts, total_slots,
                                        num_experts, device_ordinal_, stream) &&
               cudaMoE_exclusive_scan(d_expert_counts, d_expert_offsets,
                                       num_experts, device_ordinal_, stream) &&
               cudaMoE_build_active_expert_list(
                   d_expert_counts, d_group_active_expert_ids_,
                   num_experts, std::min(total_slots, num_experts),
                   device_ordinal_, stream) &&
               cudaMoE_scatter_tokens_deterministic(
                   d_routing_indices, d_routing_weights,
                   d_expert_offsets, d_expert_counts,
                   d_grouped_token_indices,
                   d_group_original_to_grouped_,
                   d_group_original_expert_ids_,
                   d_grouped_weights,
                   total_slots, top_k, num_experts,
                   device_ordinal_, stream);
    }

    bool CUDAMoEKernel::groupPrefillRoutes(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        bool filter_to_local_runtime_experts,
        bool retain_routes_for_deferred_commit)
    {
        grouped_prefill_grouping_.invalidate();
        runtime_gate_plan_.reset();
        if (!runtime_layer || !routing_indices || !routing_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::groupPrefillRoutes] null runtime or routing tensor");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::groupPrefillRoutes] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupPrefillRoutes");
        const DeviceId device = deviceId();
        const int max_slots = max_tokens * top_k;
        if (!requireTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !requireTensorOnDevice(routing_weights, device, stream, "routing_weights") ||
            !ensureGroupingBufferCapacity(max_slots, num_experts))
        {
            return false;
        }

        const float *d_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_indices || !d_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::groupPrefillRoutes] routing tensors have no device pointers");
            return false;
        }

        if (traceMoEPrefillGroupingWrapperEnabled())
        {
            LOG_TRACE("[CUDAMoEKernel] prefill grouping wrapper trace"
                     << " tag=groupPrefillRoutes"
                     << " device=" << device.to_string()
                     << " current_tokens=" << current_tokens
                     << " max_tokens=" << max_tokens
                     << " num_experts=" << num_experts
                     << " top_k=" << top_k
                     << " runtime_layer=" << static_cast<void *>(runtime_layer)
                     << " routing_indices=" << static_cast<const void *>(d_indices)
                     << " routing_weights=" << static_cast<const void *>(d_weights));
        }

        return cudaMoE_group_prefill_routes_runtime(
            d_indices,
            d_weights,
            static_cast<void *>(runtime_layer),
            d_group_original_to_grouped_,
            current_tokens * top_k,
            max_slots,
            num_experts,
            top_k,
            filter_to_local_runtime_experts ? 1 : 0,
            retain_routes_for_deferred_commit ? 1 : 0,
            device_ordinal_,
            stream,
            DeviceMoEProjectionSet::CompleteExpert);
    }

    bool CUDAMoEKernel::regroupPrefillRoutesForDiagnostics(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        bool retain_routes_for_deferred_commit)
    {
        grouped_prefill_grouping_.invalidate();
        runtime_gate_plan_.reset();
        if (!runtime_layer ||
            current_tokens < 0 || max_tokens <= 0 ||
            current_tokens > max_tokens || num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR(
                "[CUDAMoEKernel::regroupPrefillRoutesForDiagnostics] "
                "invalid route-only diagnostic contract");
            return false;
        }

        void *stream = requireStream(
            "CUDAMoEKernel::regroupPrefillRoutesForDiagnostics");
        const int max_slots = max_tokens * top_k;
        if (!ensureGroupingBufferCapacity(max_slots, num_experts))
            return false;
        return cudaMoE_regroup_prefill_routes_runtime_assignments(
            static_cast<void *>(runtime_layer),
            d_group_original_to_grouped_,
            current_tokens * top_k,
            max_slots,
            num_experts,
            top_k,
            retain_routes_for_deferred_commit ? 1 : 0,
            device_ordinal_,
            stream);
    }

    CUDAMoERuntimeGateContract CUDAMoEKernel::runtimeGateContract(
        const GroupedGateUpDescriptorTable &table, int rows, int capacity, int top_k) const
    {
        if (rows != CUDAMoEBatchInvariantPolicy::qfn_verifier_gate_plan_rows ||
            capacity != rows || top_k != CUDAMoEBatchInvariantPolicy::qfn_verifier_routes_per_token ||
            table.weight_format != DeviceMoEWeightFormat::NativeQuantised)
            return {};
        const auto contract = CUDAMoERuntimeGateContract::qfn(
            table.intermediate, table.d_model, table.codebook_id);
        if (!contract.enabled() || table.codebook_mask != (1u << contract.codebook()))
            return {};
        const auto &hardware = cudaHardwareMetadata().at(device_ordinal_);
        return hardware.sm_major == 8 && hardware.sm_minor == 6
                   ? contract : CUDAMoERuntimeGateContract{};
    }

    bool CUDAMoEKernel::publishCompleteGroupedPrefillPlanFromRouter(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        int gateup_desc_table_id,
        int down_desc_table_id,
        bool filter_to_local_runtime_experts,
        MoEGroupedPlanDemand demand,
        DeviceMoEProjectionSet expected_projections)
    {
        grouped_prefill_grouping_.invalidate();
        runtime_gate_plan_.reset();
        if (!validMoEGroupedPlanDemand(demand) || !runtime_layer || !routing_indices || !routing_weights ||
            current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0 ||
            max_tokens > std::numeric_limits<int>::max() / top_k ||
            !deviceMoEProjectionSetValid(expected_projections) ||
            gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[CUDAMoEKernel::publishCompleteGroupedPrefillPlanFromRouter] "
                      "invalid complete-plan contract");
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard))
            return false;

        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts)
        {
            LOG_ERROR("[CUDAMoEKernel::publishCompleteGroupedPrefillPlanFromRouter] "
                      "descriptor table does not match the expert domain");
            return false;
        }

        void *stream = requireStream(
            "CUDAMoEKernel::publishCompleteGroupedPrefillPlanFromRouter");
        const DeviceId device = deviceId();
        const int max_slots = max_tokens * top_k;
        if (!requireTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !requireTensorOnDevice(routing_weights, device, stream, "routing_weights") ||
            !ensureGroupingBufferCapacity(max_slots, num_experts))
        {
            return false;
        }

        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "CUDA complete runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "CUDA complete runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "CUDA complete runtime prefill down descriptors"))
        {
            return false;
        }

        const auto *device_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        const auto *device_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        const int max_active_experts = std::min(max_slots, num_experts);
        const auto gate_contract = runtimeGateContract(gateup_table, current_tokens, max_tokens, top_k);
        const bool submitted = device_indices && device_weights &&
               cudaMoE_group_prefill_routes_and_materialize_plan_runtime(
                   device_indices,
                   device_weights,
                   static_cast<void *>(runtime_layer),
                   d_group_original_to_grouped_,
                   runtime_gate_descs,
                   runtime_up_descs,
                   runtime_down_descs,
                   d_group_active_expert_ids_,
                   current_tokens * top_k,
                   max_slots,
                   num_experts,
                   top_k,
                   max_active_experts,
                   filter_to_local_runtime_experts ? 1 : 0,
                   static_cast<int>(demand),
                   gateup_table.weight_format,
                   device_ordinal_,
                   stream, expected_projections, gate_contract);
        if (submitted && gate_contract.enabled())
            runtime_gate_plan_ = RuntimeGatePlanPublication{
                gate_contract, runtime_layer, runtime_gate_descs, runtime_up_descs,
                d_group_original_to_grouped_, stream, grouped_prefill_grouping_.generation(), num_experts};
        return submitted;
    }

    bool CUDAMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        int gateup_desc_table_id,
        int down_desc_table_id,
        MoEGroupedPlanDemand demand,
        DeviceMoEProjectionSet expected_projections)
    {
        grouped_prefill_grouping_.invalidate();
        runtime_gate_plan_.reset();
        if (!validMoEGroupedPlanDemand(demand) || !runtime_layer ||
            current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0 ||
            max_tokens > std::numeric_limits<int>::max() / top_k ||
            !deviceMoEProjectionSetValid(expected_projections) ||
            gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[CUDAMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments] "
                      "invalid complete-plan contract");
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard))
            return false;

        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts)
        {
            LOG_ERROR("[CUDAMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments] "
                      "descriptor table does not match the expert domain");
            return false;
        }

        void *stream = requireStream(
            "CUDAMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments");
        const int max_slots = max_tokens * top_k;
        if (!ensureGroupingBufferCapacity(max_slots, num_experts))
            return false;

        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "CUDA assigned runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "CUDA assigned runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "CUDA assigned runtime prefill down descriptors"))
        {
            return false;
        }

        const auto gate_contract = runtimeGateContract(gateup_table, current_tokens, max_tokens, top_k);
        const bool submitted = cudaMoE_regroup_prefill_routes_and_materialize_plan_runtime(
            static_cast<void *>(runtime_layer),
            d_group_original_to_grouped_,
            runtime_gate_descs,
            runtime_up_descs,
            runtime_down_descs,
            d_group_active_expert_ids_,
            current_tokens * top_k,
            max_slots,
            num_experts,
            top_k,
            std::min(max_slots, num_experts),
            static_cast<int>(demand),
            gateup_table.weight_format,
            device_ordinal_,
            stream, expected_projections, gate_contract);
        if (submitted && gate_contract.enabled())
            runtime_gate_plan_ = RuntimeGatePlanPublication{
                gate_contract, runtime_layer, runtime_gate_descs, runtime_up_descs,
                d_group_original_to_grouped_, stream, grouped_prefill_grouping_.generation(), num_experts};
        return submitted;
    }

    bool CUDAMoEKernel::commitGroupedVerifierHistograms(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        const int32_t *accepted_state_counts_device,
        const int32_t *publication_ok_flags_device,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "commitGroupedVerifierHistograms");
        if (!stream)
            return false;
        if (!runtime_layer ||
            !accepted_state_counts_device ||
            !publication_ok_flags_device ||
            request_count <= 0 ||
            rows_per_request <= 0 ||
            total_rows != request_count * rows_per_request ||
            num_experts <= 0 ||
            top_k <= 0)
        {
            LOG_ERROR(
                "[CUDAMoEKernel::commitGroupedVerifierHistograms] invalid "
                "device publication contract");
            return false;
        }

        return cudaMoE_commit_grouped_verifier_histograms(
            static_cast<void *>(runtime_layer),
            accepted_state_counts_device,
            publication_ok_flags_device,
            request_count,
            rows_per_request,
            total_rows,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::commitGroupedVerifierHistogramsBatch(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        int layer_count,
        size_t layer_stride_bytes,
        const int32_t *accepted_state_counts_device,
        const int32_t *publication_ok_flags_device,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "commitGroupedVerifierHistogramsBatch");
        if (!stream)
            return false;
        if (!runtime_layers ||
            layer_count <= 0 ||
            (layer_count > 1 && layer_stride_bytes < sizeof(DeviceMoELayerRuntime)) ||
            !accepted_state_counts_device ||
            !publication_ok_flags_device ||
            request_count <= 0 ||
            rows_per_request <= 0 ||
            total_rows != request_count * rows_per_request ||
            num_experts <= 0 ||
            top_k <= 0)
        {
            LOG_ERROR(
                "[CUDAMoEKernel::commitGroupedVerifierHistogramsBatch] invalid "
                "device publication contract");
            return false;
        }

        return cudaMoE_commit_grouped_verifier_histograms_batch(
            static_cast<void *>(runtime_layers),
            layer_count,
            layer_stride_bytes,
            accepted_state_counts_device,
            publication_ok_flags_device,
            request_count,
            rows_per_request,
            total_rows,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::assignPrefillRoutesLeastLoadedResident(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        const int32_t *absolute_position_ids_device,
        const int32_t *active_row_count_device)
    {
        if (!runtime_layer || !absolute_position_ids_device ||
            !active_row_count_device)
        {
            LOG_ERROR(
                "[CUDAMoEKernel::assignPrefillRoutesLeastLoadedResident] "
                "runtime, device position row, and device active-row count "
                "must be non-null");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesLeastLoadedResident] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "assignPrefillRoutesLeastLoadedResident");
        if (!stream)
            return false;
        return cudaMoE_assign_prefill_routes_least_loaded_resident(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            absolute_position_ids_device,
            active_row_count_device,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        const least_loaded_ep::LeastLoadedExpertAssignmentConfig &config)
    {
        if (!runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (config.expert_count != static_cast<uint32_t>(num_experts) ||
            config.participant_count == 0u ||
            config.alpha_numerator == 0u ||
            config.alpha_denominator == 0u ||
            config.lambda_numerator == 0u ||
            config.lambda_denominator == 0u)
        {
            LOG_ERROR("[CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] invalid LLEP config");
            return false;
        }

        void *stream = explicitMoELaunchStream(launch, "planPrefillRoutesLeastLoadedCurrentBatch");
        if (!stream)
            return false;
        return cudaMoE_plan_prefill_routes_least_loaded_current_batch(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            config.min_chunk_tokens,
            config.alpha_numerator,
            config.alpha_denominator,
            config.lambda_numerator,
            config.lambda_denominator,
            config.min_spread_improvement,
            config.min_spread_improvement_divisor,
            config.min_spread_improvement_per_critical_path_slot,
            config.min_foreign_rows_per_critical_path_slot,
            config.max_weight_transfers,
            config.max_non_owner_experts_per_participant,
            config.enable_balanced_skip ? 1 : 0,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k)
    {
        if (!runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }

        void *stream = explicitMoELaunchStream(
            launch,
            "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers");
        if (!stream)
            return false;
        return cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        const DeviceMoERebalanceStatus *transfer_status,
        const DeviceMoERebalanceApplyStatus *apply_status)
    {
        if (!runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] null runtime");
            return false;
        }
        if (!transfer_status || !apply_status)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] transfer and apply status are required");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }

        void *stream = explicitMoELaunchStream(
            launch,
            "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers");
        if (!stream)
            return false;
        if (!requireCudaDevicePointer(
                runtime_layer,
                device_ordinal_,
                "runtime_layer",
                "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers",
                stream) ||
            !requireCudaDevicePointer(
                transfer_status,
                device_ordinal_,
                "transfer_status",
                "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers",
                stream) ||
            !requireCudaDevicePointer(
                apply_status,
                device_ordinal_,
                "apply_status",
                "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers",
                stream))
        {
            return false;
        }
        return cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            transfer_status,
            apply_status,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden,
        ITensor *batch_buffer,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        if (!runtime_layer || !hidden || !batch_buffer)
        {
            LOG_ERROR("[CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime] invalid arguments expert_id="
                      << expert_id << " max_tokens=" << max_tokens << " d_model=" << d_model);
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime");
        const DeviceId device = deviceId();
        if (!requireTensorOnDevice(hidden, device, stream, "hidden") ||
            !requireOutputOnDevice(batch_buffer, device, stream, "batch_buffer"))
        {
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_batch = static_cast<float *>(batch_buffer->gpu_data_ptr());
        if (!d_hidden || !d_batch)
        {
            LOG_ERROR("[CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = cudaMoE_prefill_gather_expert_runtime(
            static_cast<const void *>(runtime_layer),
            d_hidden,
            d_batch,
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            stream);
        if (ok)
            markDeviceWritten(batch_buffer, device, stream);
        return ok;
    }

    bool CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime(
        ITensor *output,
        ITensor *expert_results,
        DeviceMoELayerRuntime *runtime_layer,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        if (!output || !expert_results || !runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime] invalid arguments expert_id="
                      << expert_id << " max_tokens=" << max_tokens << " d_model=" << d_model);
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime");
        const DeviceId device = deviceId();
        if (!requireOutputOnDevice(output, device, stream, "output") ||
            !requireTensorOnDevice(expert_results, device, stream, "expert_results"))
        {
            return false;
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        const float *d_expert_output = static_cast<const float *>(expert_results->gpu_data_ptr());
        if (!d_output || !d_expert_output)
        {
            LOG_ERROR("[CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = cudaMoE_prefill_scatter_expert_runtime(
            d_output,
            d_expert_output,
            static_cast<const void *>(runtime_layer),
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            stream);
        if (ok)
            markDeviceWritten(output, device, stream);
        return ok;
    }

    int CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable(
        const DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::cuda(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != DeviceMoEWeightFormat::NativeQuantised)) return -1;

        if (!down_descs || num_experts <= 0 || d_model <= 0 || intermediate <= 0 || (intermediate % 32) != 0)
            return -1;
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "upload grouped down descriptor table"))
        {
            return -1;
        }
        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertDownDescriptorTable"))
            return -1;

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if (!runtime_mutable &&
            descriptor_source !=
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            return -1;
        }
        NativeQuantisedExecutionFormatEnvelope format_envelope = original_family
            ? original_family->nativeDownEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;
            if (original_family && !original_family->matchesNativeDown(desc)) return -1;

            if (!validateCudaGroupedDescShape(desc, d_model, intermediate))
            {
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }
            if (!original_family && !addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable))
            {
                return -1;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return -1;
        for (uint8_t codebook = 0u; codebook < 32u; ++codebook)
        {
            if ((codebook_mask & nativeQuantisedCodebookMaskBit(codebook)) != 0u &&
                !cudaGroupedPrefillSupportsCodebook(codebook))
            {
                return -1;
            }
        }
        const int sole_codebook = singleCUDAMoECodebook(codebook_mask);
        const uint8_t codebook_id =
            sole_codebook >= 0
                ? static_cast<uint8_t>(sole_codebook)
                : kCudaMoEMixedCodebookSentinel;

        if (cudaGroupedPrefillMaskNeedsIQTables(codebook_mask))
        {
            if (!ensureCudaMoEIQGridTablesInitialized(device_ordinal_))
            {
                LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] IQ grid table init failed");
                return -1;
            }
        }

        /*
         * Descriptor tables are weight-shaped, not request-shaped.  Graph
         * recapture can ask for the same table repeatedly; returning the
         * existing id keeps CUDA aligned with ROCm's deterministic slot model
         * and prevents pointer-array workspace slots from growing with replay
         * attempts.
         */
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) ||
                existing.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
                !existing.device_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.codebook_id != codebook_id ||
                existing.codebook_mask != codebook_mask ||
                existing.policy_codebook_mask != policy_codebook_mask ||
                existing.descriptor_source != descriptor_source ||
                existing.host_descs.size() != static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(existing.host_descs.data(), down_descs, desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedDownDescriptorTable table;
        table.host_descs.assign(down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;
        table.codebook_mask = codebook_mask;
        table.policy_codebook_mask = policy_codebook_mask;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.weight_format = DeviceMoEWeightFormat::NativeQuantised;
        table.valid = true;
        if (!publishGroupedDownDescriptorTable(
                table,
                "upload grouped expert down descriptor table"))
        {
            return -1;
        }
        grouped_down_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_down_desc_tables_.size() - 1);
    }

    int CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables(
        const DeviceNativeQuantisedMatrixDesc *gate_descs,
        const DeviceNativeQuantisedMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::cuda(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != DeviceMoEWeightFormat::NativeQuantised)) return -1;

        if (!gate_descs || !up_descs || num_experts <= 0 || d_model <= 0 || intermediate <= 0 || (d_model % 32) != 0)
            return -1;
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "upload grouped gate/up descriptor tables"))
        {
            return -1;
        }
        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertGateUpDescriptorTables"))
            return -1;

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if (!runtime_mutable &&
            descriptor_source !=
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            return -1;
        }
        NativeQuantisedExecutionFormatEnvelope format_envelope = original_family
            ? original_family->nativeGateUpEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            const bool gate_blank = isBlankGroupedDesc(gate_desc);
            const bool up_blank = isBlankGroupedDesc(up_desc);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Incomplete sparse descriptor pair for expert "
                          << expert_id);
                return -1;
            }

            if (original_family && !original_family->matchesNativeGateUp(gate_desc,up_desc)) return -1;
            if (!gate_desc.valid() || !up_desc.valid() ||
                !areNativeQuantisedGateUpCodebooksCompatible(gate_desc.codebook_id, up_desc.codebook_id) ||
                !validateCudaGroupedDescShape(gate_desc, intermediate, d_model) ||
                !validateCudaGroupedDescShape(up_desc, intermediate, d_model))
            {
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Invalid gate/up descriptor pair for expert "
                          << expert_id);
                return -1;
            }
            const auto add_descriptor = [&](
                                            const DeviceNativeQuantisedMatrixDesc &desc)
            {
                return addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable);
            };
            if (!original_family && (!add_descriptor(gate_desc) || !add_descriptor(up_desc)))
            {
                return -1;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return -1;
        for (uint8_t codebook = 0u; codebook < 32u; ++codebook)
        {
            if ((codebook_mask & nativeQuantisedCodebookMaskBit(codebook)) != 0u &&
                !cudaGroupedPrefillSupportsCodebook(codebook))
            {
                return -1;
            }
        }
        const int sole_codebook = singleCUDAMoECodebook(codebook_mask);
        const uint8_t codebook_id =
            sole_codebook >= 0
                ? static_cast<uint8_t>(sole_codebook)
                : kCudaMoEMixedCodebookSentinel;

        if (cudaGroupedPrefillMaskNeedsIQTables(codebook_mask))
        {
            if (!ensureCudaMoEIQGridTablesInitialized(device_ordinal_))
            {
                LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] IQ grid table init failed");
                return -1;
            }
        }

        /*
         * Gate/up descriptor tables are persistent prepared-weight metadata.
         * Reusing identical tables gives CUDA the same deterministic descriptor
         * id behavior as ROCm and keeps graph-owned pointer slots stable across
         * replay-state resets.
         */
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) ||
                existing.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
                !existing.device_gate_descs ||
                !existing.device_up_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.codebook_id != codebook_id ||
                existing.codebook_mask != codebook_mask ||
                existing.policy_codebook_mask != policy_codebook_mask ||
                existing.descriptor_source != descriptor_source ||
                existing.host_gate_descs.size() != static_cast<size_t>(num_experts) ||
                existing.host_up_descs.size() != static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(existing.host_gate_descs.data(), gate_descs, desc_bytes) == 0 &&
                std::memcmp(existing.host_up_descs.data(), up_descs, desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedGateUpDescriptorTable table;
        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;
        table.codebook_mask = codebook_mask;
        table.policy_codebook_mask = policy_codebook_mask;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.weight_format = DeviceMoEWeightFormat::NativeQuantised;
        table.valid = true;
        if (!publishGroupedGateUpDescriptorTable(
                table,
                "upload grouped expert gate/up descriptor tables"))
        {
            return -1;
        }
        grouped_gateup_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    int CUDAMoEKernel::uploadGroupedExpertFloatingDownDescriptorTable(
        const DeviceMoEFloatingMatrixDesc *down_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::cuda(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != weight_format)) return -1;

        if (!down_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return -1;
        }
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "upload grouped floating down descriptor table") ||
            !setMoEDevice(
                device_ordinal_,
                "uploadGroupedExpertFloatingDownDescriptorTable"))
        {
            return -1;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto &desc = down_descs[expert];
            if (isBlankGroupedFloatingDesc(desc))
                continue;
            if (original_family && !original_family->matchesFloatingDown(desc)) return -1;
            if (!validateGroupedFloatingDesc(desc, d_model, intermediate))
                return -1;
            any_live = true;
        }
        if (!any_live && !original_family)
            return -1;

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) || existing.weight_format != weight_format ||
                !existing.device_floating_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.descriptor_source != descriptor_source ||
                existing.host_floating_descs.size() !=
                    static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(
                    existing.host_floating_descs.data(),
                    down_descs,
                    desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedDownDescriptorTable table;
        table.host_floating_descs.assign(
            down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.weight_format = weight_format;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.valid = true;
        if (!publishGroupedDownDescriptorTable(
                table,
                "upload grouped floating expert down descriptor table"))
        {
            return -1;
        }
        grouped_down_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_down_desc_tables_.size() - 1);
    }

    int CUDAMoEKernel::uploadGroupedExpertFloatingGateUpDescriptorTables(
        const DeviceMoEFloatingMatrixDesc *gate_descs,
        const DeviceMoEFloatingMatrixDesc *up_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::cuda(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != weight_format)) return -1;

        if (!gate_descs || !up_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return -1;
        }
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "upload grouped floating gate/up descriptor tables") ||
            !setMoEDevice(
                device_ordinal_,
                "uploadGroupedExpertFloatingGateUpDescriptorTables"))
        {
            return -1;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const bool gate_blank = isBlankGroupedFloatingDesc(gate_descs[expert]);
            const bool up_blank = isBlankGroupedFloatingDesc(up_descs[expert]);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                return -1;
            }
            if (original_family && !original_family->matchesFloatingGateUp(gate_descs[expert],up_descs[expert])) return -1;
            if (!validateGroupedFloatingDesc(
                    gate_descs[expert], intermediate, d_model) ||
                !validateGroupedFloatingDesc(
                    up_descs[expert], intermediate, d_model))
            {
                return -1;
            }
            any_live = true;
        }
        if (!any_live && !original_family)
            return -1;

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) || existing.weight_format != weight_format ||
                !existing.device_floating_gate_descs ||
                !existing.device_floating_up_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.descriptor_source != descriptor_source ||
                existing.host_floating_gate_descs.size() !=
                    static_cast<size_t>(num_experts) ||
                existing.host_floating_up_descs.size() !=
                    static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(
                    existing.host_floating_gate_descs.data(),
                    gate_descs,
                    desc_bytes) == 0 &&
                std::memcmp(
                    existing.host_floating_up_descs.data(),
                    up_descs,
                    desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedGateUpDescriptorTable table;
        table.host_floating_gate_descs.assign(
            gate_descs, gate_descs + num_experts);
        table.host_floating_up_descs.assign(
            up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.weight_format = weight_format;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.valid = true;
        if (!publishGroupedGateUpDescriptorTable(
                table,
                "upload grouped floating expert gate/up descriptor tables"))
        {
            return -1;
        }
        grouped_gateup_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    bool CUDAMoEKernel::updateGroupedExpertDownDescriptorTable(
        int descriptor_table_id,
        const DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!down_descs || descriptor_table_id < 0 ||
            descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
            (intermediate % 32) != 0)
        {
            return false;
        }
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "update grouped down descriptor table"))
        {
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "updateGroupedExpertDownDescriptorTable"))
            return false;

        auto &table = grouped_down_desc_tables_[static_cast<size_t>(descriptor_table_id)];
        if (!table.valid ||
            table.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            !table.device_descs ||
            !table.workspace_publication ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(table.descriptor_source);
        NativeQuantisedExecutionFormatEnvelope format_envelope = table.original_family.valid()
            ? table.original_family.nativeDownEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;
            if (table.original_family.valid() && !table.original_family.matchesNativeDown(desc)) return false;
            if (!validateCudaGroupedDescShape(desc, d_model, intermediate))
                return false;
            if (!table.original_family.valid() && !addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable))
            {
                return false;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return false;
        const int sole_codebook = singleCUDAMoECodebook(codebook_mask);
        const uint8_t codebook_id =
            sole_codebook >= 0
                ? static_cast<uint8_t>(sole_codebook)
                : kCudaMoEMixedCodebookSentinel;
        if (table.codebook_id != codebook_id ||
            table.codebook_mask != codebook_mask ||
            table.policy_codebook_mask != policy_codebook_mask)
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertDownDescriptorTable] refusing in-place update "
                      "with changed codebook semantics");
            return false;
        }

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(getStream());
        if (!workspace_->rewritePersistentPublication(
                kCudaGroupedDownDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    down_descs,
                    nullptr,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    cudaError_t err = cudaMemcpyAsync(
                        table.device_descs,
                        down_descs,
                        desc_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertDownDescriptorTable] "
                                  "descriptor refresh failed: "
                                  << cudaGetErrorString(err));
                        return false;
                    }
                    err = cudaEventRecord(
                        static_cast<cudaEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertDownDescriptorTable] "
                                  "cannot publish refreshed descriptor readiness: "
                                  << cudaGetErrorString(err));
                        std::terminate();
                    }
                    return true;
                }))
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertDownDescriptorTable] "
                      "cannot rewrite refreshed descriptor publication");
            return false;
        }

        table.host_descs.assign(down_descs, down_descs + num_experts);
        return true;
    }

} // namespace llaminar2
