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

#include "../execution/config/RoutedExpertPolicy.h"
#include "../execution/moe/DeviceMoERebalanceController.h"
#include "../tensors/TensorKernels.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace llaminar2
{

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

    enum class MoEDecodeDescriptorSource : uint8_t
    {
        /// Top-k ids/weights come from the runtime table, but expert weight
        /// descriptors are read from the immutable per-stage descriptor tables.
        StaticDescriptorTable = 0,

        /// Top-k ids/weights and expert weight descriptors are both read from
        /// the mutable runtime placement table. Use this only when captured
        /// graph replay must observe in-place expert ownership/replica changes.
        RuntimePlacementTable = 1,
    };

    /**
     * @brief Arithmetic route selected for one prepared MoE routing graph.
     *
     * The route kind is capture identity.  Backends use it to bind exactly the
     * persistent scratch and immutable converted weights consumed by the
     * corresponding production kernel before graph capture begins.
     */
    enum class MoERouteLaunchKind : uint8_t
    {
        /// One-token routing that publishes the live device runtime table.
        RuntimeDecode = 0,
        /// Ordinary multi-row routing with a stable physical row capacity.
        GroupedPrefill = 1,
        /// Grouped MTP routing with serial-decode-equivalent row arithmetic.
        DecodeEquivalentVerifier = 2,
    };

    /**
     * @brief Immutable geometry and arithmetic policy for router preparation.
     *
     * This value contains no live request state.  The physical row count is a
     * graph-capture capacity; a separate device scalar owns the logical row
     * count at replay time.
     */
    struct MoERouteLaunchPlan
    {
        MoERouteLaunchKind kind = MoERouteLaunchKind::GroupedPrefill;
        int physical_rows = 0;
        int d_model = 0;
        int num_experts = 0;
        int top_k = 0;

        /** @brief Return whether every launch dimension is internally valid. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return physical_rows > 0 &&
                   d_model > 0 &&
                   num_experts > 0 &&
                   top_k > 0 &&
                   top_k <= num_experts;
        }
    };

    /**
     * @brief Immutable host launch metadata for one device-resident MoE call.
     *
     * GPU MoE execution may be captured concurrently by the main graph, an MTP
     * sidecar graph, and a graph-native rebalance maintenance graph. A
     * device-wide kernel object must therefore never carry a mutable "current
     * stream" or "current workspace" that another graph can retarget between
     * related launches. Callers construct this value from stage-owned bindings
     * and pass it directly to each backend primitive.
     *
     * The context contains stable addresses only; it does not mirror live
     * device values and it does not introduce host synchronization. The
     * workspace pointer is included now even where a primitive receives all
     * scratch buffers explicitly, so subsequent grouped routing/expert APIs can
     * migrate to the same contract without changing its ownership model.
     */
    struct MoEKernelLaunchContext
    {
        void *stream = nullptr;                         ///< Explicit CUDA/HIP stream for this launch.
        DeviceWorkspaceManager *workspace = nullptr;   ///< Stage-owned persistent workspace view.

        /**
         * @brief Return whether this context can issue GPU work.
         */
        [[nodiscard]] constexpr bool hasExplicitStream() const noexcept
        {
            return stream != nullptr;
        }
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
        // GPU implementations override to use gpu_data_ptr() and publish
        // writes through TransferEngine, keeping data on-device without H2D.
        //
        // Compute stages should use ONLY these methods — never raw pointers
        // or CUDA/HIP APIs directly.
        // =================================================================

        /**
         * @brief Compute softmax/top-k routing into backend-owned tensors.
         *
         * This is the sole production routing contract. CPU implementations
         * populate @p host_result because CPU expert dispatch consumes host
         * routing rows. GPU implementations must leave @p host_result empty
         * and publish @p output_indices and @p output_weights with a completion
         * event on their explicit stream; snapshots perform any D2H transfer
         * later at an explicit observation boundary.
         *
         * @return true when all output rows were published successfully.
         */
        virtual bool routeWithTensors(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result) = 0;

        /**
         * @brief Graph-capturable padded-prefill routing contract.
         *
         * Bucketed prefill graphs launch with a fixed @p seq_len, while the
         * real prompt length can be smaller on replay.  GPU implementations
         * read @p device_effective_seq_len from device memory inside the
         * top-k kernel and mark padded rows as invalid routes
         * (`expert=-1`, `weight=0`).  That keeps graph launch dimensions fixed
         * without letting padded rows mutate grouped-expert state.
         *
         * CPU/default implementations deliberately fail for non-null device
         * scalars because they cannot safely read backend-owned memory.  Call
         * routeWithTensors() for ordinary non-padded routing.
         */
        virtual bool routeWithTensorsEffectiveSeqLen(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result,
            const int *device_effective_seq_len);

        /**
         * @brief Route MTP verifier rows with grouped serial-row-equivalent math.
         *
         * MTP verifier batches contain a runtime number of logical decode rows,
         * and any accepted prefix may later be published into live state. That
         * makes ordinary
         * small-prefill router math unsafe: a batched GEMM can accumulate gate
         * logits in a different order than serial decode, changing top-k weights
         * enough to drift downstream MoE outputs.  Backends that support this
         * method must therefore compute every row with the same per-row math,
         * K traversal, and reduction contract as M=1 decode while batching the
         * verifier rows economically.  Row replay is a diagnostic oracle only;
         * production implementations must publish the per-row top-k tensors
         * directly from grouped backend work.
         *
         * Implementations must be graph-capturable on GPU backends:
         * - no host top-k mirrors or stream synchronization,
         * - no H2D copies from stack-owned row indices,
         * - no allocation after graph warmup has declared the route workspace.
         *
         * @param device_effective_seq_len Optional device INT32 logical row
         *        count for a fixed-width graph. Rows at or above this value
         *        must publish `expert=-1, weight=0` and cannot reach expert
         *        state. The pointer is replay data, not graph identity.
         *
         * The default deliberately returns false; verifier correctness should fail
         * loudly on a backend that has not implemented the rowwise contract.
         */
        virtual bool routeVerifierRowsDecodeEquivalent(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            const int *device_effective_seq_len = nullptr)
        {
            (void)hidden;
            (void)gate_weights;
            (void)seq_len;
            (void)d_model;
            (void)num_experts;
            (void)top_k;
            (void)normalize_weights;
            (void)output_indices;
            (void)output_weights;
            (void)device_effective_seq_len;
            return false;
        }

        /**
         * @brief Prepare every persistent resource used by a captured router.
         *
         * GPU implementations must bind route scratch, row-quantization
         * scratch, and any immutable Q8/FP16 gate publication required by
         * @p plan on the kernel's exact producer stream.  Conversion kernels
         * may run here because this is a setup boundary before capture; routing
         * arithmetic and output publication must not run here.
         *
         * The default is a hard unsupported result.  A GPU backend cannot
         * become graph-capturable merely by inheriting a no-op preparation.
         *
         * @param gate_weights Device-resident router matrix whose stable
         *        address participates in immutable-cache identity.
         * @param plan Typed graph geometry and arithmetic route.
         * @return `true` only when the subsequent capture performs no lazy
         *         allocation, conversion, or host-to-device publication.
         */
        virtual bool prepareRouteLaunch(
            ITensor *gate_weights,
            const MoERouteLaunchPlan &plan)
        {
            (void)gate_weights;
            (void)plan;
            return false;
        }

        /**
         * @brief Decode one row into the device-resident runtime route table.
         *
         * `absolute_position_ids_device` is the same graph-local position row
         * consumed by RoPE. GPU implementations require it whenever the active
         * placement contains a replicated expert: the logical position is the
         * stable tie-break key that makes serial and grouped route partitions
         * independent of speculative workload history.
         *
         * `row_execution_policy` is mandatory because physical replication
         * alone does not specify whether one participant or every participant
         * executes a selected row. Fully replicated local routing must preserve
         * every locally resident route and must never apply LLEP assignment.
         */
        virtual bool decodeRouteSelect(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *hidden, ITensor *gate_weights,
            int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            bool write_legacy_outputs,
            bool update_runtime_histogram,
            const int32_t *absolute_position_ids_device,
            RoutedExpertRowExecutionPolicy row_execution_policy)
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
            (void)absolute_position_ids_device;
            (void)row_execution_policy;
            return false;
        }

        /// Decode-only runtime-table routing path with graph-capturable
        /// ready-wave rebalance apply piggybacked into the existing route
        /// kernel launch. Implementations must not launch a separate apply
        /// kernel for this method.
        virtual bool decodeRouteSelectWithReadyRebalanceApply(
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
            RoutedExpertRowExecutionPolicy row_execution_policy)
        {
            (void)runtime_layers;
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
            (void)rebalance_plan_entries;
            (void)rebalance_plan_capacity;
            (void)rebalance_command_header;
            (void)rebalance_local_transfer_slots;
            (void)rebalance_local_transfer_slot_count;
            (void)rebalance_config;
            (void)rebalance_apply_status;
            (void)rebalance_controller_state;
            (void)rebalance_target_layer;
            (void)rebalance_command_buffer_count;
            (void)absolute_position_ids_device;
            (void)row_execution_policy;
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

        /**
         * @brief Diagnostic/test helper that copies one logical row between tensors.
         *
         * This API exists to probe backend row-addressing and tensor-residency
         * handoffs in focused tests.  It must not be used as a production MTP
         * verifier path: grouped verifier execution is required to consume the
         * full verifier row set directly, preserve serial-decode math order, and
         * publish state without looping through one-row scratch tensors.
         *
         * GPU backends still implement this as a tiny row-copy kernel so tests
         * can validate device-side addressing without staging host-owned index
         * arrays.  Production code should prefer grouped gather/scatter or
         * explicit grouped verifier kernels.
         */
        virtual bool copyTokenRowFromTensor(
            ITensor *source, ITensor *row_buffer,
            int row_index, int row_width);

        /// Tensor-aware scatter-add.  host indices/weights live on the host;
        /// GPU implementations upload them internally.
        virtual void scatterAddWeightedFromTensors(
            ITensor *output, ITensor *expert_output,
            const int *host_token_indices, const float *host_weights,
            int num_tokens, int d_model);

        /**
         * @brief Diagnostic/test helper that writes one scratch row into a tensor.
         *
         * The destination row is overwritten, not accumulated.  This is useful
         * for focused coherence tests that need to seed a device tensor row, but
         * it is not a verifier publication primitive.  Production MTP verifier
         * rows must use grouped decode-equivalent execution and grouped
         * publication paths rather than row-by-row scratch writes.
         */
        virtual bool writeTokenRowToTensor(
            ITensor *destination, ITensor *row_buffer,
            int row_index, int row_width);

        /// Tensor-aware shared expert gate (sigmoid gating in-place).
        virtual void sharedExpertGateFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model);

        /**
         * @brief Graph-capturable shared expert gate with a device-owned real length.
         *
         * Padded prefill graphs execute at bucket length, but only the first
         * `*device_effective_seq_len` rows are semantically live. GPU backends
         * must read that scalar on device and zero padded output rows so stale
         * graph-replay tail state cannot leak into later layers.
         *
         * The default CPU implementation only accepts a null scalar because CPU
         * callers cannot safely dereference a device pointer.
         */
        virtual bool sharedExpertGateFromTensorsEffectiveSeqLen(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model,
            const int *device_effective_seq_len);

        /**
         * @brief Gate shared expert output and add routed MoE output in one step.
         *
         * Computes, for each token row:
         *   shared_output[t, j] = sigmoid(dot(gate_inp, input[t])) * shared_output[t, j]
         *   combined[t, j] = routed_residual[t, j] + shared_output[t, j]
         *
         * The in-place write to @p shared_output is intentional: the fused path
         * still publishes the same diagnostic and parity-visible intermediate as
         * the standalone shared-gate stage while avoiding a separate residual-add
         * node.
         */
        virtual void sharedExpertGateAddFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            ITensor *routed_residual, ITensor *combined_output,
            int seq_len, int d_model);

        /**
         * @brief Graph-capturable shared gate plus routed residual combine.
         *
         * For rows beyond `*device_effective_seq_len`, both the gated shared
         * output and the final combined output are written as zero. This makes
         * padded prefill bucket rows neutral at the MoE output boundary while
         * keeping the launch shape stable for graph capture.
         */
        virtual bool sharedExpertGateAddFromTensorsEffectiveSeqLen(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            ITensor *routed_residual, ITensor *combined_output,
            int seq_len, int d_model,
            const int *device_effective_seq_len);

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
         * @brief Refresh an existing persistent down descriptor table in place.
         *
         * Graph-captured grouped decode records the device pointer for the table,
         * so dynamic expert rebalancing must update the existing allocation
         * rather than allocate a new table id when shape/codebook semantics are
         * unchanged.
         */
        virtual bool updateGroupedExpertDownDescriptorTable(
            int descriptor_table_id,
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_experts,
            int d_model,
            int intermediate)
        {
            (void)descriptor_table_id;
            (void)down_descs;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Refresh existing persistent gate/up descriptor tables in place.
         *
         * Implementations must preserve the device table pointer associated with
         * descriptor_table_id; returning false means graph-stable rebalance cannot
         * safely publish this placement without recapture.
         */
        virtual bool updateGroupedExpertGateUpDescriptorTables(
            int descriptor_table_id,
            const DeviceNativeVNNIMatrixDesc *gate_descs,
            const DeviceNativeVNNIMatrixDesc *up_descs,
            int num_experts,
            int d_model,
            int intermediate)
        {
            (void)descriptor_table_id;
            (void)gate_descs;
            (void)up_descs;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            return false;
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
         *
         * @param expert_mask Optional host-side local-compute mask with one byte
         * per logical expert.  When present, backends must convert route ids for
         * masked-off experts to `-1` in their tiny device metadata buffer before
         * launching descriptor-table kernels.  The original routing tensor is not
         * modified because histograms and runtime placement publication still need
         * the model's true top-k ids.
         */
        virtual bool groupedExpertGateUpDecodeFromRouting(
            const TensorBase *input,
            ITensor *routing_indices,
            int descriptor_table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate,
            const uint8_t *expert_mask = nullptr)
        {
            (void)input;
            (void)routing_indices;
            (void)descriptor_table_id;
            (void)top_k;
            (void)gate_outputs;
            (void)up_outputs;
            (void)d_model;
            (void)intermediate;
            (void)expert_mask;
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
         * expert dispatch.  If @p expert_mask is present, masked-off experts must
         * be converted to inactive `-1` route ids in backend-owned scratch before
         * the down kernel reads descriptor tables.
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
            int intermediate,
            const uint8_t *expert_mask = nullptr)
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
            (void)expert_mask;
            return false;
        }

        /**
         * @brief Fused single-token expert decode from device routing tensors.
         *
         * This is the explicit-routing counterpart to
         * groupedExpertDecodeFromRuntime().  The backend consumes one FP32
         * routing-index row and one FP32 routing-weight row directly on the
         * producer stream, converts the indices into backend-owned integer
         * metadata, and executes gate/up, SwiGLU, and down projection through
         * persistent workspace.  No per-route TensorBase intermediates may be
         * created by the caller.
         *
         * @param input Device-resident hidden row with shape [1, d_model].
         * @param routing_indices Device-resident FP32 expert ids with top_k
         *        entries.
         * @param routing_weights Device-resident FP32 route weights with top_k
         *        entries.
         * @param gateup_descriptor_table_id Persistent gate/up descriptor table.
         * @param down_descriptor_table_id Persistent down descriptor table.
         * @param top_k Number of routed expert slots in this row.
         * @param output Weighted MoE output consumed by the later canonical
         *        reducer when @p canonical_route_contributions is non-null;
         *        otherwise this is the device-resident publication target.
         *        A canonical producer must not require, allocate, read, or
         *        write this later reducer target.
         * @param d_model Model hidden width.
         * @param intermediate Expert intermediate width.
         * @param expert_mask Optional immutable participant-local ownership
         *        mask. Masked routes must become inactive device metadata
         *        without modifying the original routing tensors.
         * @param canonical_route_contributions Optional device-resident
         *        [1, top_k, d_model] publication target. When supplied, this
         *        tensor is the producer's sole output and @p output may not yet
         *        have device storage.
         * @return true after the output write has been published on the exact
         *         producer stream; false on any contract or launch failure.
         */
        virtual bool groupedExpertDecodeFromRouting(
            const TensorBase *input,
            ITensor *routing_indices,
            ITensor *routing_weights,
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            const uint8_t *expert_mask = nullptr,
            ITensor *canonical_route_contributions = nullptr)
        {
            (void)input;
            (void)routing_indices;
            (void)routing_weights;
            (void)gateup_descriptor_table_id;
            (void)down_descriptor_table_id;
            (void)top_k;
            (void)output;
            (void)d_model;
            (void)intermediate;
            (void)expert_mask;
            (void)canonical_route_contributions;
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
         * @brief Prepare immutable backend state for fused runtime-table decode capture.
         *
         * A captured grouped decode launch embeds persistent descriptor-table and
         * scratch-pointer addresses. Backends must bind every required workspace
         * slice and publish those pointer arrays on their exact producer stream
         * before native graph capture begins. This method performs only that
         * launch-topology preparation: it must not execute expert arithmetic,
         * consume runtime routing values, allocate transient device memory, copy
         * model activations, or synchronize a stream or device.
         *
         * @return true only when a subsequent groupedExpertDecodeFromRuntime()
         *         call is capture-ready without eager arithmetic.
         */
        virtual bool prepareGroupedRuntimeDecodeLaunchState(
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int top_k,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source)
        {
            (void)gateup_descriptor_table_id;
            (void)down_descriptor_table_id;
            (void)top_k;
            (void)d_model;
            (void)intermediate;
            (void)descriptor_source;
            return false;
        }

        /**
         * @brief Prepare immutable backend state for fixed-table grouped decode capture.
         *
         * Shared-expert decode uses fixed host-known expert metadata but device-
         * resident projection scratch. Before capture, the backend publishes the
         * fixed metadata and exact scratch addresses into persistent workspace.
         * No GEMV, GEMM, SwiGLU, or down operation may run here; the first model
         * transaction is executed only by the captured graph.
         *
         * @return true only when both table-decode launches are capture-ready.
         */
        virtual bool prepareGroupedTableDecodeLaunchState(
            const int *expert_ids,
            const float *expert_weights,
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int num_active,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            ITensor *output,
            int d_model,
            int intermediate)
        {
            (void)expert_ids;
            (void)expert_weights;
            (void)gateup_descriptor_table_id;
            (void)down_descriptor_table_id;
            (void)num_active;
            (void)gate_outputs;
            (void)up_outputs;
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
         * sequence. When @p canonical_route_contributions is supplied, it is
         * the sole publication target; @p output belongs to the later
         * canonical reducer and may not yet have device storage. The producer
         * must not require, allocate, read, or write that inactive target. The
         * default returns false so existing backends keep the established
         * gate/up plus down path.
         */
        virtual bool groupedExpertDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::RuntimePlacementTable,
            ITensor *canonical_route_contributions = nullptr)
        {
            (void)runtime_layer;
            (void)input;
            (void)gateup_descriptor_table_id;
            (void)down_descriptor_table_id;
            (void)top_k;
            (void)output;
            (void)d_model;
            (void)intermediate;
            (void)descriptor_source;
            (void)canonical_route_contributions;
            return false;
        }

        /**
         * @brief Reduce canonical routed-expert slots in router order.
         *
         * LocalTP publishes one independently allreduced FP32 row for every
         * original router slot.  This device-only epilogue is the sole owner of
         * the observable routed output and performs exactly the serial decode
         * accumulation `route 0, route 1, ...`.  Backends must overwrite every
         * output element and must not use floating-point atomics.
         *
         * @param canonical_route_contributions FP32 [seq_len, top_k, d_model].
         * @param output FP32 [seq_len, d_model] overwrite destination.
         * @param seq_len Number of original token rows.
         * @param top_k Number of router slots per row.
         * @param d_model Hidden width.
         * @return true after publication on the kernel's exact bound stream.
         */
        virtual bool reduceCanonicalRouteContributions(
            ITensor *canonical_route_contributions,
            ITensor *output,
            int seq_len,
            int top_k,
            int d_model)
        {
            (void)canonical_route_contributions;
            (void)output;
            (void)seq_len;
            (void)top_k;
            (void)d_model;
            return false;
        }

        /**
         * @brief Graph-capturable device-side MoE rebalance publish/apply.
         *
         * Backends consume a device-resident gathered histogram buffer with
         * layout [participant][layer][expert] and mutate the stable
         * DeviceMoELayerRuntime placement banks in-place. This intentionally
         * performs only assignment publication; any expert payload movement
         * must already have populated resident slots and resident masks before
         * the captured graph observes them.
         */
        virtual bool runDeviceRebalanceController(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            const uint64_t *gathered_histograms,
            DeviceMoERebalanceStatus *status,
            const DeviceMoERebalanceConfig &config,
            DeviceMoERebalancePlanEntry *plan_entries = nullptr,
            uint32_t *plan_count = nullptr,
            uint32_t plan_capacity = 0,
            uint32_t payload_slot_capacity = 0,
            DeviceMoERebalanceCommandBufferHeader *command_header = nullptr,
            DeviceMoERebalanceWaveState *wave_state = nullptr,
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1,
            const DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr,
            uint32_t local_transfer_slot_count = 0,
            DeviceMoELLEPLayerPlanScratch *llep_layer_plans = nullptr)
        {
            (void)launch;
            (void)runtime_layers;
            (void)gathered_histograms;
            (void)status;
            (void)config;
            (void)plan_entries;
            (void)plan_count;
            (void)plan_capacity;
            (void)payload_slot_capacity;
            (void)command_header;
            (void)wave_state;
            (void)controller_state;
            (void)command_buffer_count;
            (void)local_transfer_slots;
            (void)local_transfer_slot_count;
            (void)llep_layer_plans;
            return false;
        }

        /**
         * @brief Pack per-layer runtime histograms into a contiguous device buffer.
         *
         * The graph-captured rebalance path all-gathers a flat
         * [wave_layer][expert] histogram from every participant before launching
         * runDeviceRebalanceController().  GPU backends implement this as a
         * small device kernel that reads DeviceMoELayerRuntime::decode_histogram
         * directly from the mirrored runtime table; no host sync or host copy is
         * permitted on this path.  The optional wave/controller pointers let the
         * device packer follow the same rolling wave cursor as the controller.
         */
        virtual bool packDeviceRebalanceHistograms(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            uint64_t *local_histograms,
            const DeviceMoERebalanceConfig &config,
            const DeviceMoERebalanceWaveState *wave_state = nullptr,
            const DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1)
        {
            (void)launch;
            (void)runtime_layers;
            (void)local_histograms;
            (void)config;
            (void)wave_state;
            (void)controller_state;
            (void)command_buffer_count;
            return false;
        }

        /**
         * @brief Pack local resident expert descriptors into a graph-visible directory.
         *
         * The directory layout is [layer][expert] for this participant.  Each
         * valid entry describes bytes that are resident on the current device
         * and can be used as a source by the graph-side peer-copy consumer.
         * Entries for experts that are known but not locally resident remain
         * invalid.  No host synchronization or default stream work is allowed.
         */
        virtual bool packDeviceRebalanceDirectory(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            DeviceMoEExpertDirectoryEntry *local_directory,
            const DeviceMoERebalanceConfig &config)
        {
            (void)launch;
            (void)runtime_layers;
            (void)local_directory;
            (void)config;
            return false;
        }

        /**
         * @brief Pack source descriptors only for projected transfer-plan arrivals.
         *
         * Compact transfer-slot rebalance first projects the domain-root command
         * buffer into each participant's local command ABI, and then asks each
         * participant to publish descriptors only for arrivals where it is the
         * source.  The local source-descriptor layout is
         * [destination_participant][command_buffer][plan_index]. After LocalTP
         * payload allgather, consumers read
         * [source_participant][destination_participant][command_buffer][plan_index].
         */
        virtual bool packDeviceRebalanceSourceDescriptors(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            const DeviceMoERebalancePlanEntry *plan_entries,
            const DeviceMoERebalanceCommandBufferHeader *command_headers,
            uint32_t plan_capacity,
            DeviceMoEExpertDirectoryEntry *local_source_descriptors,
            const DeviceMoERebalanceConfig &config,
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1)
        {
            (void)launch;
            (void)runtime_layers;
            (void)plan_entries;
            (void)command_headers;
            (void)plan_capacity;
            (void)local_source_descriptors;
            (void)config;
            (void)controller_state;
            (void)command_buffer_count;
            return false;
        }

        /**
         * @brief Project the domain-root gathered command buffer into the local apply ABI.
         *
         * The compact graph path allgathers command buffers before transfer. The
         * root participant owns placement decisions; every participant then needs
         * the same command list locally so apply can update replicated metadata
         * deterministically, while only destination participants install transfer
         * slots. This graph-capturable projection copies the root command buffers
         * from the gathered layout into local plan/header buffers and rewrites the
         * header participant id to the local participant.
         */
        virtual bool projectDeviceRebalanceDomainCommands(
            const MoEKernelLaunchContext &launch,
            const DeviceMoERebalancePlanEntry *gathered_plan_entries,
            const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
            uint32_t plan_capacity,
            DeviceMoERebalancePlanEntry *local_plan_entries,
            DeviceMoERebalanceCommandBufferHeader *local_command_headers,
            const DeviceMoERebalanceConfig &config,
            DeviceMoERebalanceStatus *status = nullptr,
            uint32_t payload_slot_capacity = 0,
            uint32_t command_buffer_count = 1,
            const DeviceMoERebalanceWaveState *gathered_wave_states = nullptr,
            DeviceMoERebalanceWaveState *local_wave_states = nullptr,
            DeviceMoELayerRuntime *runtime_layers = nullptr,
            const DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr,
            uint32_t local_transfer_slot_count = 0)
        {
            (void)launch;
            (void)gathered_plan_entries;
            (void)gathered_command_headers;
            (void)plan_capacity;
            (void)local_plan_entries;
            (void)local_command_headers;
            (void)config;
            (void)status;
            (void)payload_slot_capacity;
            (void)command_buffer_count;
            (void)gathered_wave_states;
            (void)local_wave_states;
            (void)runtime_layers;
            (void)local_transfer_slots;
            (void)local_transfer_slot_count;
            return false;
        }

        /**
         * @brief Project per-participant prefill LLEP transfer requests into one
         *        domain-visible compact payload plan.
         *
         * Unlike decode maintenance, prefill LLEP can publish destination-local
         * transfer requests from every participant after current-batch route
         * planning.  This graph-capturable projection gathers those requests,
         * emits the same merged plan on every participant, and remaps payload
         * slots so each source has a unique compact payload lane across all
         * destinations.  The router top-k choices are not changed here.
         */
        virtual bool projectPrefillLeastLoadedDomainCommands(
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
            uint32_t command_buffer_count = 1)
        {
            (void)launch;
            (void)gathered_plan_entries;
            (void)gathered_command_headers;
            (void)plan_capacity;
            (void)local_plan_entries;
            (void)local_plan_count;
            (void)local_command_header;
            (void)config;
            (void)status;
            (void)payload_slot_capacity;
            (void)runtime_layers;
            (void)local_transfer_slots;
            (void)local_transfer_slot_count;
            (void)command_buffer_count;
            return false;
        }

        /**
         * @brief Materialize current-batch LLEP weight-transfer requirements
         *        into the standard rebalance command buffer ABI.
         *
         * planPrefillRoutesLeastLoadedCurrentBatch() writes Algorithm-4 style
         * expert-weight transfers into DeviceMoELayerRuntime::reserved_ptrs[2].
         * This graph-capturable bridge converts those records into
         * ExpertPayloadArrival commands so the existing compact payload
         * movement path can stage/import them. Commands produced here are
         * logical, including prefix-runtime rehydration:
         * `destination_slot` remains invalid until
         * projectPrefillLeastLoadedDomainCommands() leases physical storage on
         * the destination participant from its complete transfer directory.
         * Rolling directory indices are graph-lifetime allocator state and
         * must never be serialized into a RAM/disk prefix checkpoint.
         * It must not rewrite router top-k choices or perform host readback.
         */
        virtual bool materializePrefillLeastLoadedTransferCommands(
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
            uint32_t command_buffer_count = 1);

        /**
         * @brief Pack compact planned-arrival payload slots for a grouped collective.
         *
         * The source descriptor buffer is produced by
         * packDeviceRebalanceSourceDescriptors().  This method packs only the
         * entries where this participant is the source into the local payload
         * lane indexed by the command's source-local compact payload slot.
         * Before reading any command, the implementation must select exactly
         * one root-projected nonempty wave and publish its immutable
         * wave/epoch/count ticket in `status`. NCCL/RCCL allgather then moves
         * these staging slots; the destination unpacks locally. Device kernels
         * must not read peer device pointers directly or resample the mutable
         * controller cursor after the ticket is published. The matching unpack
         * and transfer-completion publisher append counters to this same record
         * and consume only the ticketed wave.
         */
        virtual bool packDeviceRebalanceCompactPayloads(
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
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1)
        {
            (void)launch;
            (void)plan_entries;
            (void)command_headers;
            (void)plan_capacity;
            (void)local_source_descriptors;
            (void)local_payload;
            (void)local_payload_slot_count;
            (void)payload_slot_bytes;
            (void)config;
            (void)status;
            (void)controller_state;
            (void)command_buffer_count;
            return false;
        }

        virtual bool packDeviceRebalanceCollectivePayloads(
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
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1)
        {
            (void)launch;
            (void)gathered_plan_entries;
            (void)gathered_command_headers;
            (void)plan_capacity;
            (void)local_directory;
            (void)local_payload;
            (void)local_payload_slot_count;
            (void)payload_slot_bytes;
            (void)config;
            (void)status;
            (void)controller_state;
            (void)command_buffer_count;
            return false;
        }

        /**
         * @brief Unpack gathered expert payload slots into local transfer slots.
         *
         * This call intentionally preserves the `status` record written by the
         * preceding pack call and consumes its immutable wave/epoch/count
         * ticket. Source-side pack errors and destination-side unpack errors
         * are published as one transfer-wave status so the graph controller can
         * fail the wave before apply observes incomplete payloads. Sampling
         * `controller_state->active_wave` here is forbidden because planning may
         * already have advanced that cursor for the next overlapped transaction.
         */
        virtual bool unpackDeviceRebalanceCollectivePayloads(
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
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1)
        {
            (void)launch;
            (void)plan_entries;
            (void)plan_count;
            (void)plan_capacity;
            (void)command_header;
            (void)gathered_payload;
            (void)local_payload_slot_count;
            (void)payload_slot_bytes;
            (void)local_transfer_slots;
            (void)local_transfer_slot_count;
            (void)config;
            (void)status;
            (void)controller_state;
            (void)command_buffer_count;
            return false;
        }

        /**
         * @brief Initialize persistent device-side rebalance graph controller state.
         *
         * The state object is shared by the captured maintenance rebalance graph
         * and the captured decode graph. Implementations must be graph-capturable:
         * no host reads, no stream synchronization, and no default stream work.
         * Replays may call this every launch; kernels should preserve an already
         * valid matching state rather than resetting wave progress.
         */
        virtual bool initializeDeviceRebalanceGraphController(
            const MoEKernelLaunchContext &launch,
            DeviceMoERebalanceGraphControllerState *controller_state,
            const DeviceMoERebalanceConfig &config)
        {
            (void)launch;
            (void)controller_state;
            (void)config;
            return false;
        }

        /**
         * @brief Begin a new request lifetime for a persistent rebalance transaction.
         *
         * Unlike initializeDeviceRebalanceGraphController(), this operation is
         * unconditional. It discards the previous request's controller epoch,
         * active wave, terminal error poison, command headers, layer-window
         * cursors, plan counts, and diagnostic counters while preserving the
         * model-lifetime allocations whose addresses are embedded in captured
         * graph executables. These records form one transaction root and must
         * never be reset independently.
         *
         * The caller must enqueue this operation on the explicit request-reset
         * stream after joining every producer from the old request and before
         * publishing the reset-ready event. Implementations must not synchronize,
         * allocate, transfer state through the host, or use a default stream.
         */
        virtual bool resetDeviceRebalanceGraphTransactionForRequest(
            const MoEKernelLaunchContext &launch,
            DeviceMoERebalanceGraphControllerState *controller_state,
            DeviceMoERebalanceCommandBufferHeader *command_headers,
            DeviceMoERebalanceWaveState *wave_states,
            uint32_t *plan_counts,
            uint32_t command_buffer_count,
            const DeviceMoERebalanceConfig &config)
        {
            (void)launch;
            (void)controller_state;
            (void)command_headers;
            (void)wave_states;
            (void)plan_counts;
            (void)command_buffer_count;
            (void)config;
            return false;
        }

        /**
         * @brief Publish a completed transfer wave from the transfer stream.
         *
         * This kernel is queued after peer/collective transfer-slot copies and
         * performs the device-side fence and lifecycle transition to
         * ReadyToApply. Decode-side apply stages poll this state instead of a
         * host callback or host-visible publication path.
         */
        virtual bool publishDeviceRebalanceTransferComplete(
            const MoEKernelLaunchContext &launch,
            DeviceMoERebalanceGraphControllerState *controller_state,
            const DeviceMoERebalanceCommandBufferHeader *command_header,
            const DeviceMoERebalanceWaveState *wave_state,
            const DeviceMoERebalanceApplyStatus *copy_status,
            const DeviceMoERebalancePlanEntry *plan_entries,
            uint32_t plan_capacity,
            const DeviceMoERebalanceApplyStatus *gathered_copy_status,
            const DeviceMoERebalanceConfig &config,
            uint32_t command_buffer_count = 1)
        {
            (void)launch;
            (void)controller_state;
            (void)command_header;
            (void)wave_state;
            (void)copy_status;
            (void)plan_entries;
            (void)plan_capacity;
            (void)gathered_copy_status;
            (void)config;
            (void)command_buffer_count;
            return false;
        }

        /**
         * @brief Poll and apply one ready wave entirely on device.
         *
         * A poll miss is not an error: the kernel leaves runtime placement
         * untouched and returns success. A ready matching wave applies arrivals
         * for @p target_layer and advances the wave lifecycle after all planned
         * layers have passed the apply boundary.
         */
        virtual bool applyReadyDeviceRebalanceWave(
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
            DeviceMoERebalanceCommandBufferHeader *command_header = nullptr,
            int target_layer = -1,
            uint32_t command_buffer_count = 1)
        {
            (void)launch;
            (void)runtime_layers;
            (void)plan_entries;
            (void)plan_count;
            (void)plan_capacity;
            (void)local_transfer_slots;
            (void)local_transfer_slot_count;
            (void)config;
            (void)status;
            (void)controller_state;
            (void)command_header;
            (void)target_layer;
            (void)command_buffer_count;
            return false;
        }

        virtual bool applyDeviceRebalanceArrivals(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            const DeviceMoERebalancePlanEntry *plan_entries,
            const uint32_t *plan_count,
            uint32_t plan_capacity,
            const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
            uint32_t local_transfer_slot_count,
            const DeviceMoERebalanceConfig &config,
            DeviceMoERebalanceApplyStatus *status,
            const DeviceMoERebalanceCommandBufferHeader *command_header = nullptr,
            int target_layer = -1)
        {
            (void)launch;
            (void)runtime_layers;
            (void)plan_entries;
            (void)plan_count;
            (void)plan_capacity;
            (void)local_transfer_slots;
            (void)local_transfer_slot_count;
            (void)config;
            (void)status;
            (void)command_header;
            (void)target_layer;
            return false;
        }

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
         * route/group state device-resident in runtime_layer. When
         * filter_to_local_runtime_experts is true, routes whose expert is not
         * local-compute ready in the active placement bank are written as
         * inactive route slots and are omitted from the local grouped batch.
         * That mode is the StaticOwner LocalTP contract: every participant
         * groups only the shard it will actually compute, while the later TP
         * reduction combines the partial MoE outputs. Dynamic/LLEP planning
         * leaves this flag false because it first needs all routed slots before
         * the assignment kernels choose participants. The default returns false
         * so a missing grouped implementation fails loudly.
         *
         * Grouping is deliberately free of persistent routing-history side
         * effects. A speculative verifier computes routes for rows that may be
         * rejected, so it cannot know which rows belong to the serial-visible
         * timeline. When @p retain_routes_for_deferred_commit is true, the
         * grouping kernel must copy each final route and participant assignment
         * into the layer's immutable-address device ledger while those values
         * are already in-register. Accepted demand is published later by
         * commitGroupedVerifierHistograms() from device-owned acceptance
         * metadata. This fused retention must not add a kernel launch.
         */
        virtual bool groupPrefillRoutes(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *routing_indices, ITensor *routing_weights,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            bool filter_to_local_runtime_experts = false,
            bool retain_routes_for_deferred_commit = false)
        {
            (void)runtime_layer;
            (void)routing_indices;
            (void)routing_weights;
            (void)current_tokens;
            (void)max_tokens;
            (void)num_experts;
            (void)top_k;
            (void)filter_to_local_runtime_experts;
            (void)retain_routes_for_deferred_commit;
            return false;
        }

        /**
         * @brief Rebuild runtime prefill groups from device-owned route assignments.
         *
         * LLEP/device-side assignment kernels may rewrite
         * DeviceMoELayerRuntime::route_participant_ids after routing while
         * preserving the original route_expert_ids and route_weights. This
         * method clears only counts/offsets/grouped scratch, then groups rows
         * assigned to runtime_layer->participant_id. It must not read route
         * metadata back to host and must be graph-capturable.
         *
         * When @p retain_routes_for_deferred_commit is true, the regrouping
         * kernel must retain the final route and participant assignment in the
         * per-layer device ledger without launching a separate copy kernel.
         *
         * Like initial grouping, regrouping must not mutate persistent decode
         * history. It may run before the verifier outcome exists and therefore
         * cannot distinguish committed rows from rejected speculative rows.
         */
        virtual bool regroupPrefillRoutesFromRuntimeAssignments(
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            bool retain_routes_for_deferred_commit = false)
        {
            (void)runtime_layer;
            (void)current_tokens;
            (void)max_tokens;
            (void)num_experts;
            (void)top_k;
            (void)retain_routes_for_deferred_commit;
            return false;
        }

        /**
         * @brief Commit routing demand for serial-visible grouped-verifier rows.
         *
         * Grouped MTP verification computes a padded request-major matrix, but
         * only a device-selected prefix of each request becomes part of the
         * main-model timeline. This operation reads those accepted prefix
         * lengths and publishes both global selected-expert demand and
         * participant-local assigned demand from the per-layer route ledger
         * retained by @p runtime_layer. Transient route scratch is not a valid
         * publication source because later layers intentionally reuse it.
         *
         * The accepted-state commit is intentionally separate from route
         * retention in groupPrefillRoutes() and
         * regroupPrefillRoutesFromRuntimeAssignments(). Grouping runs before
         * acceptance is known and may only retain immutable route evidence; it
         * must not mutate routing history. Callers must enqueue this commit on
         * the exact stream that owns accepted-state publication. The default
         * hard failure keeps an unimplemented backend from silently losing or
         * overcounting decode evidence.
         *
         * @param launch Explicit producer stream and persistent workspace.
         * @param runtime_layer Device-resident per-layer routing scratch.
         * @param accepted_state_counts_device Per-request committed row counts.
         * @param publication_ok_flags_device Per-request metadata validity flags.
         * @param request_count Number of active request rows.
         * @param rows_per_request Physical padded verifier rows per request.
         * @param total_rows Total physical rows represented by routing scratch.
         * @param num_experts Number of routed experts in this layer.
         * @param top_k Number of route slots per physical row.
         * @return true when the commit kernel was enqueued successfully.
         */
        virtual bool commitGroupedVerifierHistograms(
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
            (void)launch;
            (void)runtime_layer;
            (void)accepted_state_counts_device;
            (void)publication_ok_flags_device;
            (void)request_count;
            (void)rows_per_request;
            (void)total_rows;
            (void)num_experts;
            (void)top_k;
            return false;
        }

        /**
         * @brief Assign prefill routes to least-loaded resident participants.
         *
         * This is the graph-capturable resident-expert bridge used by grouped
         * MTP verification. It preserves the router's selected experts and
         * weights, then rewrites only
         * DeviceMoELayerRuntime::route_participant_ids.
         *
         * Every token row reproduces the production serial-decode transaction:
         * participant loads start at zero, route slots are consumed in router
         * order, and equal-load resident replicas use the row's absolute
         * position plus route identity as a deterministic tie key. The same
         * policy runs in serial decode, so a logical row receives identical
         * participant ids at every grouped M regardless of rejected work.
         *
         * This is a numerical-correctness requirement. Moving one expert
         * contribution to a different collective participant changes the
         * floating-point partial-sum grouping before the allreduce and can
         * change stochastic generation even when every expert kernel is
         * individually byte exact.
         *
         * Candidate destinations are limited to the active placement bank's
         * resident_participant_mask for each expert. The method therefore
         * never schedules a route to a device that lacks the expert weights
         * and never performs an implicit weight transfer. Implementations must
         * remain allocation-free, host-free, and graph-capturable.
         *
         * @param absolute_position_ids_device Device-resident INT32 absolute
         *        position for every grouped row. This must be the exact
         *        graph-local row consumed by RoPE and must never be null.
         * @param active_row_count_device Device-resident INT32 logical row
         *        count within the physically captured grouped width. This is
         *        the same request-geometry scalar consumed by routing and
         *        downstream row-masked epilogues. Implementations must read it
         *        on the launch stream and leave every physical suffix row
         *        untouched; host reconstruction of this value is forbidden.
         */
        virtual bool assignPrefillRoutesLeastLoadedResident(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            const int32_t *absolute_position_ids_device,
            const int32_t *active_row_count_device)
        {
            (void)launch;
            (void)runtime_layer;
            (void)current_tokens;
            (void)max_tokens;
            (void)num_experts;
            (void)top_k;
            (void)absolute_position_ids_device;
            (void)active_row_count_device;
            return false;
        }

        /**
         * @brief Plan full current-batch LLEP route spans without changing top-k.
         *
         * This graph-capturable planner materializes Algorithm-4 style
         * assignment spans and expert-weight transfer requirements into
         * DeviceMoELayerRuntime::reserved_ptrs[1..2]. It does not execute the
         * token/weight exchange and must not rewrite route_participant_ids:
         * callers may consume the plan only through a transfer-aware grouped
         * collective path.
         */
        virtual bool planPrefillRoutesLeastLoadedCurrentBatch(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            const least_loaded_ep::LeastLoadedExpertAssignmentConfig &config);

        /**
         * @brief Apply a previously materialized current-batch LLEP plan only
         * when it does not require missing expert-weight transfers.
         *
         * This is the guarded execution bridge for already-resident plans. It
         * rewrites only route_participant_ids from reserved_ptrs[1] spans and
         * preserves router-selected expert ids/weights. If reserved_u64[3]
         * reports any required weight transfers, the kernel deliberately leaves
         * executable route assignments unchanged; the full transfer-aware path
         * must import those expert payloads before applying foreign spans.
         */
        virtual bool assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k);

        /**
         * @brief Apply current-batch LLEP spans after required expert movement
         * has been staged into the runtime table.
         *
         * Unlike the zero-transfer guard above, this method is allowed to
         * consume spans whose destination participant required an expert
         * arrival. Callers must only invoke it after the transfer/arrival path
         * has populated the destination descriptors. It still preserves router
         * top-k expert ids and weights; only route_participant_ids are balanced.
         */
        virtual bool assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            const DeviceMoERebalanceStatus *transfer_status,
            const DeviceMoERebalanceApplyStatus *apply_status);

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
         * Builds counts, offsets, and grouped route rows entirely on device for
         * direct consumption by executeGroupedPrefillPipeline(). There is no
         * synchronous host-metadata counterpart: production GPU grouping is
         * device-owned by construction.
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
         * @brief Prepare device-side groups using a previously published device mask.
         *
         * This is the graph-execution half of the fixed-topology masked grouping
         * contract. The backend must consume only its persistent device-resident
         * mask; this method deliberately accepts no host pointer and may never
         * perform mask publication. The original routing tensors are preserved so
         * rebalance histograms can still observe the model's true top-k choices
         * while this participant computes only its locally enabled experts.
         *
         * Call updateGroupedPrefillExpertMask() on the same explicit stream
         * before graph capture begins. Calling this method without a matching
         * publication is a contract violation and must fail rather than silently
         * scheduling every expert or uploading transient host state.
         */
        virtual bool prepareExpertGroupsAsyncUsingPublishedMask(
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
         * @brief Publish the persistent device expert mask before graph capture.
         *
         * Captured prefill graphs read the backend-owned mask buffer by device
         * address. The control plane must therefore upload the complete mask on
         * the same explicit stream used by the stage before capture or replay is
         * admitted. Implementations must reject this call while graph capture is
         * active: host publication is a lifecycle transition, never a graph node
         * or token-hot-path operation.
         */
        virtual bool updateGroupedPrefillExpertMask(
            const uint8_t *expert_mask,
            int num_experts)
        {
            (void)expert_mask;
            (void)num_experts;
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
         * Runs a fixed device-side pipeline with grouped gather/quantization,
         * grouped gate/up, grouped SwiGLU quantization, and direct ordered down
         * publication.  The final dispatch owns one token/output lane and walks
         * only that token's tiny top-k route list in serial-decode order.  It
         * therefore remains economical and graph-capturable without per-route
         * down materialization, atomic accumulation, or host-device
         * synchronization.
         *
         * @param hidden         Input hidden states [seq_len, d_model]
         * @param output         Final output [seq_len, d_model]. This is the
         *        publication target only when
         *        @p canonical_route_contributions is null; otherwise it is a
         *        later reducer target that may not yet have device storage.
         * @param gate_desc_table_id  Descriptor table ID for gate weights
         * @param up_desc_table_id    Descriptor table ID for up weights (same table)
         * @param down_desc_table_id  Descriptor table ID for down weights
         * @param seq_len        Number of tokens in the sequence
         * @param d_model        Model dimension
         * @param intermediate   Expert intermediate dimension
         * @param num_experts    Total number of experts
         * @param top_k          Experts per token
         * @param canonical_route_contributions Optional device-resident
         *        [seq_len, top_k, d_model] publication target. When supplied,
         *        the producer must not require, allocate, read, or write
         *        @p output.
         * @return true on success
         */
        virtual bool executeGroupedPrefillPipeline(
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k,
            ITensor *canonical_route_contributions = nullptr)
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
            (void)canonical_route_contributions;
            return false;
        }

        /**
         * @brief Execute grouped MoE prefill from DeviceMoELayerRuntime scratch.
         *
         * @p device_runtime_layer is the graph-stable device runtime table
         * pointer. @p runtime_host_layer supplies stable scratch pointer values
         * for counts, offsets, grouped token ids, and grouped route weights.
         * Implementations must not read device-side count values on the host;
         * all route and mutable descriptor data remains device-resident and
         * graph-capturable. Publication ownership matches
         * executeGroupedPrefillPipeline(): a non-null
         * @p canonical_route_contributions is the sole producer output, while
         * @p output belongs to the later canonical reducer and may not yet
         * have device storage.
         */
        virtual bool executeGroupedPrefillPipelineFromRuntime(
            DeviceMoELayerRuntime *device_runtime_layer,
            const DeviceMoELayerRuntime &runtime_host_layer,
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k,
            ITensor *canonical_route_contributions = nullptr)
        {
            (void)device_runtime_layer;
            (void)runtime_host_layer;
            (void)hidden;
            (void)output;
            (void)gateup_desc_table_id;
            (void)down_desc_table_id;
            (void)seq_len;
            (void)d_model;
            (void)intermediate;
            (void)num_experts;
            (void)top_k;
            (void)canonical_route_contributions;
            return false;
        }

    };

    /**
     * @brief Graph-local owner for one routed MoE producer/consumer pipeline.
     *
     * Routing and routed-expert execution are two sequential stages of one
     * transaction. The router publishes reusable Q8 hidden rows and device
     * route metadata that the immediately following expert stage consumes.
     * Keeping one kernel inside this explicit graph-local owner preserves that
     * zero-copy publication while preventing unrelated main-graph, MTP-sidecar,
     * shared-expert, and maintenance graphs from aliasing mutable launch state.
     *
     * Graph builders create one owner per layer graph. The first stage to warm
     * the graph constructs @ref kernel lazily through KernelFactory; no backend
     * allocation or device interaction is required during declarative graph
     * construction.
     */
    struct MoERoutedPipelineKernelOwner
    {
        std::unique_ptr<IMoEKernel> kernel; ///< Backend object shared only by the paired routed stages.
    };

} // namespace llaminar2
