/**
 * @file IMoEKernel.h
 * @brief Device-agnostic Mixture-of-Experts kernel interface
 *
 * Defines the kernel contract for MoE-specific operations that are not
 * covered by ITensorGemm (which already handles device-agnostic GEMM).
 *
 * Operations:
 * - Router: gate logits → softmax → top-k selection
 * - Token gather: collect tokens for an expert batch
 * - Scatter-add: weighted accumulation of expert outputs
 * - Shared expert gate: sigmoid dot + elementwise scale
 * - SwiGLU fallback: activation when fused GEMM+SwiGLU is unavailable
 *
 * CPU implementation: CPUMoEKernel (src/v2/kernels/cpu/moe/)
 * GPU implementations can override for device-native execution.
 */

#pragma once

#include "../tensors/TensorKernels.h"

#include <cstdint>
#include <vector>

namespace llaminar2
{

    struct DeviceMoELayerRuntime;

    /**
     * @brief Routing result from MoE gate computation
     *
     * Contains per-token expert assignments and weights after
     * softmax + top-k selection.
     */
    struct MoERoutingResult
    {
        std::vector<int> expert_indices;   ///< [seq_len * top_k] selected expert IDs
        std::vector<float> expert_weights; ///< [seq_len * top_k] normalized weights
        std::vector<float> router_logits;  ///< [seq_len * num_experts] post-softmax probs
    };

    /**
     * @brief Device-agnostic MoE kernel interface
     *
     * Encapsulates all non-GEMM MoE operations. GEMM (gate/up/down projections)
     * is already device-agnostic via ITensorGemm engines. This kernel handles
     * the MoE-specific orchestration primitives:
     *
     * - Routing (softmax top-k)
     * - Token gather/scatter for expert batching
     * - Shared expert sigmoid gating
     * - SwiGLU activation fallback
     */
    class IMoEKernel : public ITensorKernel
    {
    public:
        ~IMoEKernel() override = default;

        // =================================================================
        // Router: gate logits → softmax → top-k per token
        // =================================================================

        /**
         * @brief Compute MoE routing: gate logits, softmax, top-k selection
         *
         * For each token t in [0, seq_len):
         *   1. logits[e] = dot(hidden[t], gate_weights[e]) for all experts
         *   2. probs = softmax(logits)
         *   3. Select top-k experts by probability
         *   4. Optionally normalize top-k weights to sum to 1
         *
         * @param hidden           Input hidden states [seq_len, d_model]
         * @param gate_weights     Router gate matrix [num_experts, d_model]
         * @param seq_len          Number of tokens
         * @param d_model          Hidden dimension
         * @param num_experts      Total number of experts
         * @param top_k            Experts selected per token
         * @param normalize_weights Renormalize top-k weights to sum to 1
         * @param[out] result      Routing assignments and weights
         * @return true on success
         */
        virtual bool route(
            const float *hidden,
            const float *gate_weights,
            int seq_len, int d_model,
            int num_experts, int top_k,
            bool normalize_weights,
            MoERoutingResult &result) = 0;

        // =================================================================
        // Token gather/scatter for expert batching
        // =================================================================

        /**
         * @brief Gather tokens into a contiguous batch buffer for one expert
         *
         * Copies rows from hidden[token_indices[i]] into batch_buffer[i]
         * for i in [0, num_tokens).
         *
         * @param hidden        Full hidden states [seq_len, d_model]
         * @param batch_buffer  Output batch [num_tokens, d_model]
         * @param token_indices Token row indices to gather [num_tokens]
         * @param num_tokens    Number of tokens in this expert batch
         * @param d_model       Hidden dimension
         */
        virtual void gatherTokenBatch(
            const float *hidden,
            float *batch_buffer,
            const int *token_indices,
            int num_tokens, int d_model) = 0;

        /**
         * @brief Scatter weighted expert outputs back to combined output
         *
         * For each token i in [0, num_tokens):
         *   output[token_indices[i]] += weights[i] * expert_output[i]
         *
         * @param output         Accumulated output [seq_len, d_model] (must be pre-zeroed)
         * @param expert_output  Expert's output [num_tokens, d_model]
         * @param token_indices  Token row indices [num_tokens]
         * @param weights        Per-token routing weights [num_tokens]
         * @param num_tokens     Number of tokens in this expert batch
         * @param d_model        Hidden dimension
         */
        virtual void scatterAddWeighted(
            float *output,
            const float *expert_output,
            const int *token_indices,
            const float *weights,
            int num_tokens, int d_model) = 0;

        // =================================================================
        // Shared expert sigmoid gating
        // =================================================================

        /**
         * @brief Apply sigmoid gating to shared expert output (in-place)
         *
         * For each token t in [0, seq_len):
         *   gate = sigmoid(dot(gate_inp, input[t]))
         *   shared_output[t] *= gate
         *
         * @param input          Hidden states [seq_len, d_model]
         * @param gate_inp       Gate vector [d_model]
         * @param shared_output  Shared expert output, modified in-place [seq_len, d_model]
         * @param seq_len        Number of tokens
         * @param d_model        Hidden dimension
         */
        virtual void sharedExpertGate(
            const float *input,
            const float *gate_inp,
            float *shared_output,
            int seq_len, int d_model) = 0;

        // =================================================================
        // SwiGLU activation fallback
        // =================================================================

        /**
         * @brief SwiGLU activation: gate = silu(gate) * up
         *
         * In-place into gate buffer. Used as fallback when the GEMM engine
         * does not support fused SwiGLU+Down projection.
         *
         * @param gate  Gate projection output, modified in-place [count]
         * @param up    Up projection output [count]
         * @param count Total number of elements (batch_size * intermediate_dim)
         */
        virtual void swiGLU(float *gate, const float *up, int count) = 0;

        // =================================================================
        // Weighted vector addition (for expert output accumulation)
        // =================================================================

        /**
         * @brief Weighted add: output += weight * input
         *
         * Used to accumulate expert outputs into the combined MoE output.
         * CPU implementation is a simple loop. GPU implementations use
         * device-native kernels.
         *
         * @param output  Accumulated output buffer [count] (read+write)
         * @param input   Expert output buffer [count] (read-only)
         * @param weight  Scalar routing weight
         * @param count   Number of elements
         */
        virtual void weightedAdd(float *output, const float *input,
                                 float weight, int count)
        {
            // Default CPU implementation
            for (int i = 0; i < count; ++i)
                output[i] += weight * input[i];
        }

        // =================================================================
        // Tensor-aware API (device-agnostic)
        //
        // These methods accept ITensor* and handle coherence internally.
        // CPU defaults (in IMoEKernel.cpp) use data()/mutable_data().
        // GPU implementations override to use gpu_data_ptr() +
        // transitionTo(), keeping data on-device without H2D round-trips.
        //
        // Compute stages should use ONLY these methods — never raw pointers
        // or CUDA/HIP APIs directly.
        // =================================================================

        /// Tensor-aware route: computes routing, writes results to output
        /// tensors on the active device, and returns a host copy in
        /// host_result for CPU-side expert dispatch.  GPU implementations
        /// use D2D for tensors (no intermediate H2D).
        virtual bool routeWithTensors(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result);

        /// Decode-only runtime-table routing path. GPU implementations may
        /// keep top-k results entirely device-resident and optionally fill the
        /// legacy routing tensors for existing staged consumers.
        virtual bool decodeRouteSelect(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *hidden, ITensor *gate_weights,
            int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            bool write_legacy_outputs,
            bool update_runtime_histogram)
        {
            (void)runtime_layer;
            (void)hidden;
            (void)gate_weights;
            (void)d_model;
            (void)num_experts;
            (void)top_k;
            (void)normalize_weights;
            (void)output_indices;
            (void)output_weights;
            (void)write_legacy_outputs;
            (void)update_runtime_histogram;
            return false;
        }

        /// Zero a tensor's data buffer on the active device.
        /// GPU: zeros device memory, marks DEVICE_AUTHORITATIVE.
        /// CPU: zeros via mutable_data().
        virtual void zeroBuffer(ITensor *tensor, size_t bytes);

        /// Tensor-aware gather.  host_token_indices lives on the host;
        /// GPU implementations upload it to device staging internally.
        virtual void gatherTokenBatchFromTensors(
            ITensor *hidden, ITensor *batch_buffer,
            const int *host_token_indices, int num_tokens, int d_model);

        /// Tensor-aware scatter-add.  host indices/weights live on the host;
        /// GPU implementations upload them internally.
        virtual void scatterAddWeightedFromTensors(
            ITensor *output, ITensor *expert_output,
            const int *host_token_indices, const float *host_weights,
            int num_tokens, int d_model);

        /// Tensor-aware shared expert gate (sigmoid gating in-place).
        virtual void sharedExpertGateFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model);

        /// Tensor-aware SwiGLU: gate = silu(gate) * up, on active device.
        virtual void swiGLUFromTensors(ITensor *gate, ITensor *up, int count);

        /// Tensor-aware weighted add: output += weight * input.
        virtual void weightedAddFromTensors(
            ITensor *output, ITensor *input, float weight, int count);

        /**
         * @brief Grouped decode path for routed expert down projections.
         *
         * Implementations may consume per-active-expert gate/up scratch tensors,
         * native-VNNI down-weight descriptors, and routing weights to produce the
         * final weighted MoE output in routing order. The default returns false so
         * compute stages can fall back to the established per-expert path.
         */
        virtual bool groupedExpertDownDecode(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            const int *expert_ids,
            const float *expert_weights,
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_active,
            ITensor *output,
            int d_model,
            int intermediate)
        {
            (void)gate_tensors;
            (void)up_tensors;
            (void)expert_ids;
            (void)expert_weights;
            (void)down_descs;
            (void)num_active;
            (void)output;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Upload a persistent all-expert descriptor table for grouped decode.
         *
         * Returns an opaque table id owned by the kernel implementation, or -1 if
         * the backend cannot use a persistent descriptor table. Entries may be
         * invalid for non-local experts; grouped table decode validates active
         * expert ids before launching.
         */
        virtual int uploadGroupedExpertDownDescriptorTable(
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_experts,
            int d_model,
            int intermediate)
        {
            (void)down_descs;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            return -1;
        }

        /**
         * @brief Upload persistent all-expert descriptor tables for grouped gate/up decode projections.
         *
         * Returns an opaque table id owned by the kernel implementation, or -1 if
         * the backend cannot use the descriptor tables. Entries may be invalid
         * for non-local experts; grouped table decode validates active expert ids
         * before launching.
         */
        virtual int uploadGroupedExpertGateUpDescriptorTables(
            const DeviceNativeVNNIMatrixDesc *gate_descs,
            const DeviceNativeVNNIMatrixDesc *up_descs,
            int num_experts,
            int d_model,
            int intermediate)
        {
            (void)gate_descs;
            (void)up_descs;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            return -1;
        }

        /**
         * @brief Grouped single-token gate/up projections using persistent descriptor tables.
         *
         * Implementations should write gate_outputs[i] and up_outputs[i] for each
         * active expert slot i. The default returns false so stages can fall back
         * to multiply_fused_tensor().
         */
        virtual bool groupedExpertGateUpDecodeFromTable(
            const TensorBase *input,
            const int *expert_ids,
            int descriptor_table_id,
            int num_active,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate)
        {
            (void)input;
            (void)expert_ids;
            (void)descriptor_table_id;
            (void)num_active;
            (void)gate_outputs;
            (void)up_outputs;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Grouped single-token gate/up projections from device routing indices.
         *
         * This variant consumes FP32 routing index tensors directly on device and
         * avoids the decode-time D2H top-k synchronization. Implementations may
         * return false to let stages fall back to the host-routed table path.
         */
        virtual bool groupedExpertGateUpDecodeFromRouting(
            const TensorBase *input,
            ITensor *routing_indices,
            int descriptor_table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate)
        {
            (void)input;
            (void)routing_indices;
            (void)descriptor_table_id;
            (void)top_k;
            (void)gate_outputs;
            (void)up_outputs;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Grouped single-token gate/up projections from runtime-table top-k state.
         *
         * This variant consumes DeviceMoELayerRuntime::topk_expert_ids directly
         * on device after decodeRouteSelect(), avoiding legacy FP32 routing
         * tensors and decode-time float-to-int conversion.
         */
        virtual bool groupedExpertGateUpDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int descriptor_table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate)
        {
            (void)runtime_layer;
            (void)input;
            (void)descriptor_table_id;
            (void)top_k;
            (void)gate_outputs;
            (void)up_outputs;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Grouped decode path using a persistent descriptor table.
         *
         * The active expert ids are uploaded as tiny per-call routing metadata;
         * the down projection descriptor is selected on device from the table.
         */
        virtual bool groupedExpertDownDecodeFromTable(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            const int *expert_ids,
            const float *expert_weights,
            int descriptor_table_id,
            int num_active,
            ITensor *output,
            int d_model,
            int intermediate)
        {
            (void)gate_tensors;
            (void)up_tensors;
            (void)expert_ids;
            (void)expert_weights;
            (void)descriptor_table_id;
            (void)num_active;
            (void)output;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Grouped decode down path from device routing tensors.
         *
         * Reads FP32 routing_indices and routing_weights directly on device,
         * selecting expert descriptors by expert id without host-side dynamic
         * expert dispatch.
         */
        virtual bool groupedExpertDownDecodeFromRouting(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            ITensor *routing_indices,
            ITensor *routing_weights,
            int descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate)
        {
            (void)gate_tensors;
            (void)up_tensors;
            (void)routing_indices;
            (void)routing_weights;
            (void)descriptor_table_id;
            (void)top_k;
            (void)output;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Grouped decode down path from runtime-table top-k state.
         *
         * Reads DeviceMoELayerRuntime::topk_expert_ids and topk_weights directly
         * on device, selecting expert descriptors by runtime expert id without
         * consuming legacy FP32 routing tensors.
         */
        virtual bool groupedExpertDownDecodeFromRuntime(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            DeviceMoELayerRuntime *runtime_layer,
            int descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate)
        {
            (void)gate_tensors;
            (void)up_tensors;
            (void)runtime_layer;
            (void)descriptor_table_id;
            (void)top_k;
            (void)output;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Fused runtime-table single-token expert decode.
         *
         * Backends may fuse runtime-routed gate/up projection, SwiGLU
         * quantization, and down projection into one graph-capturable launch
         * sequence. The default returns false so existing backends keep the
         * established gate/up plus down path.
         */
        virtual bool groupedExpertDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate)
        {
            (void)runtime_layer;
            (void)input;
            (void)gateup_descriptor_table_id;
            (void)down_descriptor_table_id;
            (void)top_k;
            (void)output;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        // =================================================================
        // Device-resident histogram + expert mask (Phase 2)
        //
        // Default no-op implementations so CPU kernels compile unchanged.
        // GPU kernels override for on-device execution.
        // =================================================================

        /**
         * @brief Record routing decisions into device-resident histogram
         *
         * Atomically increments per-expert counters based on routing indices.
         *
         * @param d_routing_indices Device pointer: [seq_len * top_k] expert indices
         * @param seq_len           Number of tokens
         * @param top_k             Number of experts per token
         * @param layer_idx         Layer index (for per-layer histograms)
         */
        virtual void recordHistogramDevice(
            const int *d_routing_indices, int seq_len, int top_k, int layer_idx) {}

        /**
         * @brief Sync device histogram to host (async D2H copy + stream sync)
         *
         * @param host_counts Host buffer to receive counts: [num_experts] uint64_t
         * @param layer_idx   Which layer's histogram to sync
         * @param num_experts  Number of experts
         */
        virtual void syncHistogramToHost(
            uint64_t *host_counts, int layer_idx, int num_experts) {}

        /**
         * @brief Reset device histogram counters for a specific layer to zero
         *
         * @param layer_idx   Layer index
         * @param num_experts  Number of experts
         */
        virtual void resetHistogramDevice(int layer_idx, int num_experts) {}

        /**
         * @brief Upload expert mask to device (H2D)
         *
         * mask[e] = true means expert e is active (local to this device).
         *
         * @param mask        Host pointer: [num_experts] booleans
         * @param num_experts Number of experts
         */
        virtual void updateExpertMaskDevice(const bool *mask, int num_experts) {}

        /**
         * @brief Apply expert mask: zero out routing weights for masked-off experts
         *
         * For each token/expert slot, if the expert is masked off,
         * set its routing weight to 0.
         *
         * @param d_routing_weights  Device pointer: [seq_len * top_k] weights (modified in-place)
         * @param d_routing_indices  Device pointer: [seq_len * top_k] expert indices
         * @param seq_len            Number of tokens
         * @param top_k              Number of experts per token
         */
        virtual void applyExpertMaskDevice(
            float *d_routing_weights, const int *d_routing_indices,
            int seq_len, int top_k) {}

        // =================================================================
        // Device-side token grouping (Phase 3 — prefill optimization)
        //
        // Default no-op returning false so CPU kernels compile unchanged.
        // GPU kernels override for on-device execution.
        // =================================================================

        /**
         * @brief Group tokens by expert on-device (prefill optimization)
         *
         * After this call, tokens for expert e are at:
         *   grouped_token_indices[expert_offsets[e] .. expert_offsets[e] + expert_counts[e])
         *   grouped_weights[expert_offsets[e] .. expert_offsets[e] + expert_counts[e])
         * where grouped_token_indices[i] is the original token index (0..seq_len-1).
         *
         * All pointers are device pointers. expert_offsets and expert_counts
         * are output device buffers.
         *
         * @param d_routing_indices       Device pointer: [seq_len * top_k] expert indices
         * @param d_routing_weights       Device pointer: [seq_len * top_k] routing weights
         * @param seq_len                 Number of tokens
         * @param num_experts             Total number of experts
         * @param top_k                   Experts selected per token
         * @param d_expert_offsets        Output device pointer: [num_experts] exclusive prefix sums
         * @param d_expert_counts         Output device pointer: [num_experts] per-expert token counts
         * @param d_grouped_token_indices Output device pointer: [seq_len * top_k] grouped token indices
         * @param d_grouped_weights       Output device pointer: [seq_len * top_k] grouped weights
         * @return true on success
         */
        virtual bool groupTokensByExpertDevice(
            const int *d_routing_indices,
            const float *d_routing_weights,
            int seq_len, int num_experts, int top_k,
            int *d_expert_offsets,
            int *d_expert_counts,
            int *d_grouped_token_indices,
            float *d_grouped_weights) { return false; }

        /**
         * @brief Populate DeviceMoELayerRuntime prefill grouping scratch from routing tensors.
         *
         * routing_indices and routing_weights are FP32 tensors with
         * current_tokens * top_k entries. GPU implementations should keep all
         * route/group state device-resident in runtime_layer. The default
         * returns false so callers can use the established host/grouping path.
         */
        virtual bool groupPrefillRoutes(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *routing_indices, ITensor *routing_weights,
            int current_tokens, int max_tokens,
            int num_experts, int top_k)
        {
            (void)runtime_layer;
            (void)routing_indices;
            (void)routing_weights;
            (void)current_tokens;
            (void)max_tokens;
            (void)num_experts;
            (void)top_k;
            return false;
        }

        /**
         * @brief Gather one expert's fixed-capacity prefill batch from runtime grouping.
         *
         * The default returns false; GPU implementations can use runtime_layer
         * scratch without reading counts or routing data back to host.
         */
        virtual bool gatherPrefillExpertBatchFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *hidden, ITensor *batch_buffer,
            int expert_id, int max_tokens, int d_model)
        {
            (void)runtime_layer;
            (void)hidden;
            (void)batch_buffer;
            (void)expert_id;
            (void)max_tokens;
            (void)d_model;
            return false;
        }

        /**
         * @brief Scatter one expert's fixed-capacity prefill output from runtime grouping.
         *
         * The default returns false; GPU implementations can consume runtime
         * grouped token ids and weights entirely on device.
         */
        virtual bool scatterPrefillExpertResultsFromRuntime(
            ITensor *output, ITensor *expert_results,
            DeviceMoELayerRuntime *runtime_layer,
            int expert_id, int max_tokens, int d_model)
        {
            (void)output;
            (void)expert_results;
            (void)runtime_layer;
            (void)expert_id;
            (void)max_tokens;
            (void)d_model;
            return false;
        }

        // =================================================================
        // Phase 4: GPU-side expert dispatch for prefill
        //
        // prepareExpertGroups() does GPU-side token grouping from tensor
        // routing results (float→int conversion + grouping kernel).
        // After this, gatherExpertBatch/scatterExpertResults use device-
        // resident grouped indices — no per-call H2D staging needed.
        //
        // CPU default: falls back to host-side grouping.
        // =================================================================

        /**
         * @brief Prepare device-side token groups from routing tensor results.
         *
         * Converts float routing indices to int, runs GPU token grouping,
         * and D2H's the per-expert counts (small: num_experts ints).
         * After this call, gatherExpertBatch/scatterExpertResults use
         * pre-computed device-side offsets.
         *
         * @return true if GPU grouping succeeded; false → caller should
         *         fall back to host-side grouping.
         */
        virtual bool prepareExpertGroups(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k);

        /**
         * @brief Get number of tokens routed to a specific expert.
         * Only valid after prepareExpertGroups().
         */
        virtual int getExpertTokenCount(int expert_id) const;

        /**
         * @brief Gather tokens for expert_id using pre-grouped device indices.
         * Only valid after prepareExpertGroups().
         */
        virtual void gatherExpertBatch(
            ITensor *hidden, ITensor *batch_buffer,
            int expert_id, int d_model);

        /**
         * @brief Scatter weighted expert results using pre-grouped device data.
         * Only valid after prepareExpertGroups().
         */
        virtual void scatterExpertResults(
            ITensor *output, ITensor *expert_results,
            int expert_id, int d_model);

        // =================================================================
        // Phase 5: Fully-grouped MoE prefill pipeline (graph-capturable)
        //
        // Runs ALL experts in a single pipeline with NO host-device sync:
        //   1. gather + quantize  (all experts, single launch)
        //   2. gate+up GEMM      (all experts, single launch)
        //   3. SwiGLU + quantize (all experts, single launch)
        //   4. down GEMM         (all experts, single launch)
        //   5. weighted scatter  (all experts, single launch)
        //
        // Requires prepareExpertGroupsAsync() (no D2H) to have been called.
        // =================================================================

        /**
         * @brief Prepare device-side expert groups WITHOUT D2H synchronization.
         *
         * Same as prepareExpertGroups() but omits the hipStreamSynchronize
         * and D2H copy of counts/offsets. The grouped data stays entirely
         * on device for consumption by executeGroupedPrefillPipeline().
         *
         * @return true if GPU grouping succeeded.
         */
        virtual bool prepareExpertGroupsAsync(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k)
        {
            (void)routing_indices;
            (void)routing_weights;
            (void)seq_len;
            (void)num_experts;
            (void)top_k;
            return false;
        }

        /**
         * @brief Prepare a graph-capturable grouped prefill layout for a shared expert.
         *
         * Shared experts are always active for every token and have an implicit
         * route weight of 1. GPU implementations can populate the same grouping
         * scratch used by executeGroupedPrefillPipeline() without materializing
         * synthetic routing tensors. The default returns false.
         */
        virtual bool prepareSharedExpertPrefillGroup(int seq_len)
        {
            (void)seq_len;
            return false;
        }

        /**
         * @brief Execute the full grouped MoE prefill pipeline (graph-capturable).
         *
         * Runs all 5 kernels (gather+quant, gate+up GEMM, SwiGLU+quant,
         * down GEMM, weighted scatter) in a single function with zero
         * host-device synchronization.
         *
         * @param hidden         Input hidden states [seq_len, d_model]
         * @param output         Output buffer [seq_len, d_model] (pre-zeroed)
         * @param gate_desc_table_id  Descriptor table ID for gate weights
         * @param up_desc_table_id    Descriptor table ID for up weights (same table)
         * @param down_desc_table_id  Descriptor table ID for down weights
         * @param seq_len        Number of tokens in the sequence
         * @param d_model        Model dimension
         * @param intermediate   Expert intermediate dimension
         * @param num_experts    Total number of experts
         * @param top_k          Experts per token
         * @return true on success
         */
        virtual bool executeGroupedPrefillPipeline(
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k)
        {
            (void)hidden;
            (void)output;
            (void)gateup_desc_table_id;
            (void)down_desc_table_id;
            (void)seq_len;
            (void)d_model;
            (void)intermediate;
            (void)num_experts;
            (void)top_k;
            return false;
        }

    protected:
        // State for CPU-default prepareExpertGroups / getExpertTokenCount /
        // gatherExpertBatch / scatterExpertResults.
        // GPU overrides manage their own device-resident equivalents.
        std::vector<int> host_expert_counts_;
        std::vector<int> host_expert_offsets_;
        std::vector<int> host_grouped_indices_;
        std::vector<float> host_grouped_weights_;
        int prepared_num_experts_ = 0;
    };

} // namespace llaminar2
