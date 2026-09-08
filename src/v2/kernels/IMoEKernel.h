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
#include "../execution/moe/MoEOverlayActivationPacketABI.h"
#include "../execution/moe/MoEOverlayNodeLocalRouteExchangeABI.h"
#include "../execution/moe/MoEOverlayDeviceControllerKernels.h"
#include "../execution/moe/DeviceMoEOverlayServiceTelemetry.h"
#include "../execution/moe/DeviceMoEOverlayEpochABI.h"
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

    /**
     * @brief Select the sole weight-descriptor authority for one captured MoE
     *        stage.
     *
     * Routing ids and weights may come from device runtime state in either
     * mode. This value controls only expert projection descriptors and is
     * immutable graph identity: setup, decode, grouped prefill, and MTP replay
     * must all carry the same selection.
     */
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
     * @brief Access granted to one graph-local router Q8 publication binding.
     *
     * The routed router/expert pair owns both sides of the publication.  A
     * sibling shared expert is deliberately a required consumer: it may read
     * the published device rows, but it may neither invalidate them nor
     * silently quantize a second copy when publication is unavailable.
     */
    enum class MoERouterQ8PublicationAccess : uint8_t
    {
        ProducerAndConsumer = 0,
        RequiredConsumer = 1,
    };

    /**
     * @brief Graph-construction metadata for reusable device-resident Q8 rows.
     *
     * This object contains no tensor values and is never consulted by graph
     * replay.  During capture, the router records the stable device addresses
     * that its quantization kernel writes; later capture-time consumers embed
     * those same addresses in their own launches.  The source identity and
     * geometry prevent a sibling stage from consuming another layer's rows.
     *
     * The object is shared independently of backend launch state.  In
     * particular, routed and shared experts retain separate grouping scratch,
     * descriptor tables, and work directories even though both consume the
     * router's immutable Q8 row publication.
     */
    struct MoERouterQ8HiddenPublication
    {
        DeviceType backend = DeviceType::CPU; ///< Producing backend; ordinal < 0 means unbound.
        int device_ordinal = -1;              ///< Exact producer device ordinal.
        const float *source_rows = nullptr;   ///< FP32 source pointer used by the router.
        const int8_t *quantized_rows = nullptr; ///< Device Q8 row payload.
        const float *row_scales = nullptr;      ///< Device scale payload, one per 32 columns.
        int published_rows = 0;               ///< Number of contiguous source rows published.
        int d_model_capacity = 0;              ///< Hidden-width capacity of the payload.
        int blocks_per_row_capacity = 0;       ///< Scale blocks available per row.
        bool capture_recorded = false;         ///< Producer is recorded in the active capture.

        /**
         * @brief Clear payload provenance while preserving the device binding.
         */
        void clearPayload() noexcept
        {
            source_rows = nullptr;
            quantized_rows = nullptr;
            row_scales = nullptr;
            published_rows = 0;
            d_model_capacity = 0;
            blocks_per_row_capacity = 0;
            capture_recorded = false;
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
         * @param deferred_selected_route_ledger Optional per-layer runtime
         *        whose immutable-address verifier ledger receives selected
         *        expert IDs in the same top-k kernel. Participant IDs are
         *        written as `-1`; this contract is reserved for a manual
         *        heterogeneous overlay boundary where execution ownership is
         *        resolved outside this device graph. Passing a non-null ledger
         *        requires capacity for `seq_len * top_k` route slots.
         *
         * The default deliberately returns false; verifier correctness should fail
         * loudly on a backend that has not implemented the rowwise contract.
         */
        virtual bool routeVerifierRowsDecodeEquivalent(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            const int *device_effective_seq_len = nullptr,
            DeviceMoELayerRuntime *deferred_selected_route_ledger = nullptr)
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
            (void)deferred_selected_route_ledger;
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
         * @brief Bind the graph-local Q8 row publication used by this kernel.
         *
         * GPU graph builders call this before capture.  Producers publish
         * stable device addresses into @p publication while required consumers
         * only read them.  Implementations must reject backend/device mismatch
         * and a required consumer must fail execution when the exact source
         * rows are not present; duplicate quantization is not an allowed
         * substitute.
         *
         * @param publication Graph-local publication shared with the router.
         * @param access Typed producer/consumer authority for this kernel.
         * @return True when the binding belongs to this exact backend device.
         */
        virtual bool bindRouterQ8HiddenPublication(
            std::shared_ptr<MoERouterQ8HiddenPublication> publication,
            MoERouterQ8PublicationAccess access)
        {
            (void)publication;
            (void)access;
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
         * expert ids before launching. A runtime-placement table must certify
         * the complete capture-stable execution-format envelope reachable from
         * each descriptor's authenticated source identity.
         *
         * @param down_descs Sparse host descriptor table indexed by expert id.
         * @param num_experts Logical expert count and table entry count.
         * @param d_model Down-projection row count.
         * @param intermediate Down-projection reduction width.
         * @param descriptor_source Immutable graph-selected descriptor authority.
         * @return Persistent backend table id, or -1 on contract failure.
         */
        virtual int uploadGroupedExpertDownDescriptorTable(
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_experts,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            (void)down_descs;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            (void)descriptor_source;
            return -1;
        }

        /**
         * @brief Upload persistent all-expert descriptor tables for grouped gate/up decode projections.
         *
         * Returns an opaque table id owned by the kernel implementation, or -1 if
         * the backend cannot use the descriptor tables. Entries may be invalid
         * for non-local experts; grouped table decode validates active expert ids
         * before launching. Runtime-placement tables widen their physical
         * decoder envelope before capture without changing source arithmetic.
         *
         * @param gate_descs Sparse gate descriptors indexed by expert id.
         * @param up_descs Sparse up descriptors indexed by expert id.
         * @param num_experts Logical expert count and table entry count.
         * @param d_model Gate/up reduction width.
         * @param intermediate Gate/up row count.
         * @param descriptor_source Immutable graph-selected descriptor authority.
         * @return Persistent backend table id, or -1 on contract failure.
         */
        virtual int uploadGroupedExpertGateUpDescriptorTables(
            const DeviceNativeVNNIMatrixDesc *gate_descs,
            const DeviceNativeVNNIMatrixDesc *up_descs,
            int num_experts,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            (void)gate_descs;
            (void)up_descs;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            (void)descriptor_source;
            return -1;
        }

        /**
         * @brief Publish a persistent contiguous-floating down table.
         *
         * Blank entries represent experts not resident on this participant.
         * Every non-blank entry uses @p weight_format and exact row-major
         * geometry. The returned id occupies the same typed table namespace as
         * NativeVNNI tables so captured callers cannot accidentally combine
         * descriptor families.
         */
        virtual int uploadGroupedExpertFloatingDownDescriptorTable(
            const DeviceMoEFloatingMatrixDesc *down_descs,
            DeviceMoEWeightFormat weight_format,
            int num_experts,
            int d_model,
            int intermediate)
        {
            (void)down_descs;
            (void)weight_format;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            return -1;
        }

        /**
         * @brief Publish persistent contiguous-floating gate/up tables.
         *
         * Gate and up entries must be blank together or form a complete pair
         * with the same @p weight_format and projection geometry.
         */
        virtual int uploadGroupedExpertFloatingGateUpDescriptorTables(
            const DeviceMoEFloatingMatrixDesc *gate_descs,
            const DeviceMoEFloatingMatrixDesc *up_descs,
            DeviceMoEWeightFormat weight_format,
            int num_experts,
            int d_model,
            int intermediate)
        {
            (void)gate_descs;
            (void)up_descs;
            (void)weight_format;
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
         * @brief Refresh a floating down table without changing its address.
         *
         * Runtime movement may replace payload pointers, but captured table
         * identity, precision, and geometry remain immutable.
         */
        virtual bool updateGroupedExpertFloatingDownDescriptorTable(
            int descriptor_table_id,
            const DeviceMoEFloatingMatrixDesc *down_descs,
            DeviceMoEWeightFormat weight_format,
            int num_experts,
            int d_model,
            int intermediate)
        {
            (void)descriptor_table_id;
            (void)down_descs;
            (void)weight_format;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Refresh paired floating gate/up tables in their stable slots.
         */
        virtual bool updateGroupedExpertFloatingGateUpDescriptorTables(
            int descriptor_table_id,
            const DeviceMoEFloatingMatrixDesc *gate_descs,
            const DeviceMoEFloatingMatrixDesc *up_descs,
            DeviceMoEWeightFormat weight_format,
            int num_experts,
            int d_model,
            int intermediate)
        {
            (void)descriptor_table_id;
            (void)gate_descs;
            (void)up_descs;
            (void)weight_format;
            (void)num_experts;
            (void)d_model;
            (void)intermediate;
            return false;
        }

        /**
         * @brief Grouped single-token gate/up projections using persistent descriptor tables.
         *
         * Implementations write gate_outputs[i] and up_outputs[i] for each active
         * expert slot. Host-known expert metadata is immutable after its owner
         * slot is prepared for capture; callers with changing routes use
         * groupedExpertGateUpDecodeFromRouting() or the runtime-table variant.
         * The default returns false.
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
         * The active expert ids and weights belong to the immutable captured
         * owner slot prepared during setup; the down descriptor is selected on
         * device from the table. Callers with changing routing metadata use
         * groupedExpertDownDecodeFromRouting() or the runtime-table variant.
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
         * @param runtime_layer Optional live placement/histogram authority. When
         *        @p descriptor_source selects RuntimePlacementTable, the backend
         *        filters the explicit route through this epoch-pinned bank and
         *        records selected/local demand before resolving descriptors.
         * @param descriptor_source Select immutable setup descriptors or the
         *        mutable runtime placement bank. Runtime placement requires a
         *        non-null @p runtime_layer and forbids a host ownership mask.
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
            ITensor *canonical_route_contributions = nullptr,
            DeviceMoELayerRuntime *runtime_layer = nullptr,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable)
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
            (void)runtime_layer;
            (void)descriptor_source;
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
         * @brief Publish one participant's shared-expert partial into a rank bank.
         *
         * The canonical publication tensor has the contiguous layout
         * `[route slots][participant shared banks]`, where route slots occupy
         * `seq_len * top_k * d_model` FP32 elements and every shared bank occupies
         * `seq_len * d_model` elements. The implementation overwrites every
         * shared-bank element on every invocation: the calling participant's bank
         * receives its shared partial and every other bank receives exact zero.
         * Consequently a later rooted sum transports the evidence without
         * allowing the collective library to choose the shared branch's FP32
         * addition order.
         *
         * @param shared_output Participant-local input-parallel shared FFN row.
         * @param canonical_publication Route slots followed by shared rank banks.
         * @param seq_len Physical graph row capacity.
         * @param top_k Number of canonical routed slots per row.
         * @param d_model Hidden width.
         * @param participant_index Communicator-local bank written by this graph.
         * @param participant_count Number of rank banks in the publication.
         * @param device_effective_seq_len Optional device-owned live row count.
         * @return true when the complete bank region was published.
         */
        virtual bool publishSharedExpertRankBank(
            ITensor *shared_output,
            ITensor *canonical_publication,
            int seq_len,
            int top_k,
            int d_model,
            int participant_index,
            int participant_count,
            const int *device_effective_seq_len = nullptr);

        /**
         * @brief Finalize one rooted canonical MoE publication in fixed order.
         *
         * Only the collective root calls this operation. For every output element
         * it folds route slots in original router order, folds shared banks in
         * ascending participant order, computes the existing shared sigmoid gate,
         * and publishes the routed, gated-shared, and final combined tensors. The
         * arithmetic order is independent of collective algorithm, M, and expert
         * placement; GPU implementations must use explicit FP32 rounding boundaries
         * and may not use atomics.
         *
         * @param input Normalized hidden rows used by the shared gate dot product.
         * @param gate_inp FP32 shared-expert gate vector.
         * @param canonical_publication Root-owned route slots and shared rank banks.
         * @param routed_output Router-order reduced branch output.
         * @param shared_output Rank-order reduced and sigmoid-gated shared output.
         * @param combined_output Final routed plus gated-shared output.
         * @param seq_len Physical graph row capacity.
         * @param top_k Number of routed slots per row.
         * @param d_model Hidden width.
         * @param participant_count Number of shared rank banks.
         * @param device_effective_seq_len Optional device-owned live row count.
         * @return true when every output was published.
         */
        virtual bool finalizeCanonicalMoEPublication(
            ITensor *input,
            ITensor *gate_inp,
            ITensor *canonical_publication,
            ITensor *routed_output,
            ITensor *shared_output,
            ITensor *combined_output,
            int seq_len,
            int top_k,
            int d_model,
            int participant_count,
            const int *device_effective_seq_len = nullptr);

        // =================================================================
        // Captured ExpertOverlay sparse activation transactions
        // =================================================================

        /**
         * @brief Compact one target participant's routes into shared pages.
         *
         * GPU implementations enqueue metadata and payload kernels on the
         * exact @p launch stream. The caller subsequently publishes the lane's
         * dispatch timeline through TransferEngine on that same stream. No
         * method may synchronize or inspect live counts on the host.
         */
        virtual bool packMoEOverlayActivationDispatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationDispatchPackLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Validate and expand a dispatch after its exact timeline wait.
         *
         * Unused fixed-capacity rows and route slots are cleared on device.
         * Semantic failures leave the live-row scalar zero so stale packet
         * bytes can never reach participant-local expert compute.
         */
        virtual bool consumeMoEOverlayActivationDispatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationDispatchConsumeLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Compact follower-local expert output into shared return pages.
         *
         * The caller publishes the matching return timeline through
         * TransferEngine after this enqueue on the same exact stream.
         */
        virtual bool packMoEOverlayActivationReturn(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationReturnPackLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Validate and accumulate one participant return in fixed order.
         *
         * Graph construction invokes this once per participant in canonical
         * planner order on one stream. One thread owns each output element and
         * no atomics are used, retaining deterministic FP32 addition order.
         */
        virtual bool consumeMoEOverlayActivationReturn(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationReturnConsumeLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Materialize a colocated CPU ticket into canonical GPU slots.
         *
         * Implementations acquire the next setup-owned mapped publication
         * sequence on the exact caller stream, copy each preweighted compact
         * row to its authenticated original route slot, perform no reduction,
         * and release-acknowledge that sequence only after materialization.
         */
        virtual bool consumeMoEOverlayCanonicalRouteTicket(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayCanonicalRouteTicketConsumeLaunch &ticket)
        {
            (void)launch;
            (void)ticket;
            return false;
        }

        /**
         * @brief Pack one direct-mapped row and publish it in one graph node.
         *
         * The fixed one-row geometry lets one cooperative block preserve the
         * ordinary packet arithmetic while issuing the final system-release
         * timeline store itself. This method is not a fallback for wider
         * geometry; unsupported or incomplete launches must fail.
         */
        virtual bool packSingleRowMoEOverlayActivationDispatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowDispatchPackLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Acquire, validate, and expand one direct-mapped row atomically.
         *
         * A single cooperative block waits on the exact mapped lease, validates
         * the authenticated descriptor, and materializes hidden/routes without
         * a scheduler-visible gap or host-owned state.
         */
        virtual bool consumeSingleRowMoEOverlayActivationDispatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowDispatchConsumeLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Pack and system-release one direct-mapped return row.
         *
         * The release store occurs only after every cooperative thread has
         * written its assigned columns, making the one kernel the complete
         * follower publication edge.
         */
        virtual bool packSingleRowMoEOverlayActivationReturn(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowReturnPackLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Acquire and fold one direct-mapped return row in one graph node.
         *
         * Parent graph order still supplies canonical participant ordering, so
         * fusing the wait and fold does not alter FP32 addition order.
         */
        virtual bool consumeSingleRowMoEOverlayActivationReturn(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowReturnConsumeLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Publish all independent one-row continuation lanes in one launch.
         *
         * The persistent descriptor array is topology-sized during setup. GPU
         * implementations assign one block to each lane; there is no host loop,
         * participant-count specialization, or cross-lane arithmetic.
         */
        virtual bool packSingleRowMoEOverlayActivationDispatchBatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowDispatchBatchLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Acquire remote rows concurrently and fold them deterministically.
         *
         * Implementations enqueue a topology-parallel gather followed by one
         * planner-ordered FP32 fold on the same exact stream. Both operations
         * are graph-capturable and allocation-free.
         */
        virtual bool consumeSingleRowMoEOverlayActivationReturnBatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowReturnBatchLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Validate multi-row lanes in parallel and fold them once.
         *
         * The persistent descriptor array is planner ordered. Implementations
         * must retain that exact order in FP32 while assigning validation and
         * row-index construction independently across lanes.
         */
        virtual bool consumeMultiRowMoEOverlayActivationReturnBatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationMultiRowReturnBatchLaunch &packet)
        {
            (void)launch;
            (void)packet;
            return false;
        }

        /**
         * @brief Publish this participant's assigned canonical routes locally.
         *
         * The producer copies only route slots selected by the authoritative
         * device runtime table into its mapped node-local lane and advances the
         * lane epoch on @p launch.stream. Implementations must not synchronize
         * the stream or materialize route ownership on the host.
         */
        virtual bool publishNodeLocalCanonicalRoutes(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalRoutePublishLaunch &publication)
        {
            (void)launch;
            (void)publication;
            return false;
        }

        /**
         * @brief Acquire all local peer-route epochs on the exact root stream.
         *
         * This operation deliberately ends before bulk payload consumption so
         * TransferEngine can insert captured H2D DMA nodes on the same stream.
         */
        virtual bool acquireNodeLocalCanonicalRoutes(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalRouteConsumeLaunch &consumption)
        {
            (void)launch;
            (void)consumption;
            return false;
        }

        /**
         * @brief Stage only peer-owned mapped rows into root-device scratch.
         *
         * Implementations enqueue a fixed-shape sparse copy kernel after the
         * epoch acquire and before validation/fold. No host-visible route count
         * or variable graph node is permitted.
         */
        virtual bool stageNodeLocalCanonicalRoutes(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalRouteConsumeLaunch &consumption)
        {
            (void)launch;
            (void)consumption;
            return false;
        }

        /**
         * @brief Validate staged peer routes, fold in order, and acknowledge.
         *
         * The continuation root consumes peer rows from stable root-device
         * scratch, validates every mapped slot tag, and writes the dense routed
         * output using the fixed FP32 route order. Participants outside this
         * node-local fabric contribute zero and are folded by their explicit
         * heterogeneous return stages later in the captured graph.
         */
        virtual bool foldNodeLocalCanonicalRoutes(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalRouteConsumeLaunch &consumption)
        {
            (void)launch;
            (void)consumption;
            return false;
        }

        /**
         * @brief Gate root payload reuse on device-owned peer acknowledgements.
         *
         * Implementations enqueue exactly one graph-capturable wait kernel on
         * @p launch.stream. TransferEngine inserts the dense D2H copy after
         * this edge; no host progress or device synchronization is permitted.
         */
        virtual bool beginNodeLocalDensePublication(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalDensePublicationLaunch &publication)
        {
            (void)launch;
            (void)publication;
            return false;
        }

        /**
         * @brief Release a root D2H-complete dense payload to every peer.
         *
         * The implementation advances the monotonic root epoch with a
         * system-release store on the exact stream after TransferEngine's copy.
         */
        virtual bool finishNodeLocalDensePublication(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalDensePublicationLaunch &publication)
        {
            (void)launch;
            (void)publication;
            return false;
        }

        /**
         * @brief Acquire the next dense root publication on one peer stream.
         *
         * TransferEngine inserts the peer's H2D copy only after this device
         * wait, preserving root-to-peer visibility inside the captured graph.
         */
        virtual bool beginNodeLocalDensePublicationConsume(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalDensePublicationLaunch &publication)
        {
            (void)launch;
            (void)publication;
            return false;
        }

        /**
         * @brief Acknowledge peer H2D completion to the root producer.
         *
         * The peer is the sole writer of its cache-line acknowledgement. The
         * root may not overwrite the shared bank until every peer publishes it.
         */
        virtual bool finishNodeLocalDensePublicationConsume(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalDensePublicationLaunch &publication)
        {
            (void)launch;
            (void)publication;
            return false;
        }

        // =================================================================
        // Captured ExpertOverlay epoch admission and maintenance
        // =================================================================

        /**
         * @brief Enqueue the start marker for one real routed-expert stage.
         *
         * GPU implementations write only the graph-local device sample. They
         * perform no mapped write, allocation, copy, or synchronization.
         *
         * @param launch Exact non-null inference stream.
         * @param sample Stable graph-workspace cursor for this stage.
         * @return True when the backend accepted the graph-capturable launch.
         */
        virtual bool beginMoEOverlayServiceTelemetry(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayServiceTelemetrySample *sample)
        {
            (void)launch;
            (void)sample;
            return false;
        }

        /**
         * @brief Enqueue service accumulation after routed expert publication.
         *
         * The finish kernel derives exact locally executed activations from
         * the runtime histogram generation (or grouped expert counts), converts
         * the device's steady clock to nanoseconds, and atomically adds one
         * coherent sample to @p layer_telemetry.
         *
         * @param launch Exact producer stream containing all expert work.
         * @param runtime_layer Device runtime whose route evidence was consumed.
         * @param layer_telemetry Three phase cells for this model layer.
         * @param sample Matching graph-local start marker and route baselines.
         * @param num_experts Exact runtime expert count.
         * @param hint Semantic phase identity or device-detected Auto.
         * @param runtime_graph_role Optional device-owned authenticated phase
         *        authority. When non-null it supersedes @p hint at execution.
         * @return True when the backend accepted the graph-capturable launch.
         */
        virtual bool finishMoEOverlayServiceTelemetry(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            DeviceMoEOverlayServiceTelemetryCell *layer_telemetry,
            DeviceMoEOverlayServiceTelemetrySample *sample,
            std::uint32_t num_experts,
            MoEOverlayServicePhaseHint hint,
            const MoEOverlayInferenceGraphRole *runtime_graph_role = nullptr)
        {
            (void)launch;
            (void)runtime_layer;
            (void)layer_telemetry;
            (void)sample;
            (void)num_experts;
            (void)hint;
            (void)runtime_graph_role;
            return false;
        }

        /**
         * @brief Publish cumulative service totals at a maintenance boundary.
         *
         * The finite kernel copies device-local cells into this participant's
         * mapped record and release-publishes a new generation after a system
         * fence. It is never launched from an inference graph.
         *
         * @param launch Exact dedicated maintenance stream.
         * @param telemetry First local `[layer][phase]` accumulator.
         * @param samples First per-layer timing cursor.
         * @param layer_count Exact model layer count.
         * @param participant_id Global dense overlay participant id.
         * @param publication Participant-owned mapped destination.
         * @return True when the backend accepted the graph-capturable launch.
         */
        virtual bool publishMoEOverlayServiceTelemetry(
            const MoEKernelLaunchContext &launch,
            const DeviceMoEOverlayServiceTelemetryCell *telemetry,
            const DeviceMoEOverlayServiceTelemetrySample *samples,
            std::uint32_t layer_count,
            std::int32_t participant_id,
            MoEOverlayDeviceServiceTelemetryPublicationHeader *publication)
        {
            (void)launch;
            (void)telemetry;
            (void)samples;
            (void)layer_count;
            (void)participant_id;
            (void)publication;
            return false;
        }

        /**
         * @brief Execute one topology-wide mapped controller transition.
         *
         * CUDA and HIP implementations enqueue one fixed graph-capturable
         * kernel on @p launch.stream. The leader GPU is the sole writer of
         * global lifecycle/command state; a follower may write only its own
         * group record. Semantic completion remains device-resident and is
         * consumed by the next action or a terminal diagnostic read.
         *
         * @param launch Exact non-null graph or maintenance stream.
         * @param action Immutable mapped aliases and typed transition.
         * @return True when the backend accepted the enqueue.
         */
        virtual bool runMoEOverlayDeviceControllerAction(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayDeviceControllerActionLaunch &action)
        {
            (void)launch;
            (void)action;
            return false;
        }

        /**
         * @brief Acquire the currently published immutable placement bank.
         *
         * GPU implementations enqueue one bounded, graph-capturable kernel on
         * @p launch.stream.  The kernel installs a reader before publishing the
         * request-lifetime @p ticket.  CPU executes the identical lifecycle
         * directly against its authoritative host control block.
         *
         * @param launch Exact stage-owned stream/workspace binding.
         * @param control Authoritative backend-resident epoch control block.
         * @param ticket Persistent request ticket, overwritten on success.
         * @param status Persistent semantic completion record.
         * @param external_admission_epoch Optional system-visible global epoch.
         *        When present, acquire selects that exact Published/Retiring
         *        bank instead of the participant-local selector. This closes
         *        multi-participant publication fan-out without blocking.
         * @param admission_barrier Optional node-local continuation barrier.
         *        When present, every symmetric participant raises its local RCU
         *        guard before the publisher freezes @p external_admission_epoch.
         * @param peer_placement_epoch Optional heterogeneous-follower source.
         *        The device waits for an activation identity newer than the
         *        endpoint's retained grant, authenticates the matching packet
         *        descriptor, and selects its exact placement epoch. It is
         *        mutually exclusive with the continuation admission inputs.
         * @return True when the operation executed or was enqueued; inspect
         *         @p status through an ordered consumer for semantic success.
         */
        virtual bool acquireMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            DeviceMoEOverlayEpochTicket *ticket,
            DeviceMoEOverlayEpochStatus *status,
            const std::uint64_t *external_admission_epoch = nullptr,
            DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier = {},
            MoEOverlayPeerPlacementEpochBinding peer_placement_epoch = {})
        {
            (void)launch;
            (void)control;
            (void)ticket;
            (void)status;
            (void)external_admission_epoch;
            (void)admission_barrier;
            (void)peer_placement_epoch;
            return false;
        }

        /**
         * @brief Release the reader held by one request-lifetime placement ticket.
         *
         * A successful release clears @p ticket so replay can acquire a fresh
         * epoch without a host reset.  A stale, malformed, or already released
         * ticket is a semantic failure recorded in @p status.
         */
        virtual bool releaseMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            DeviceMoEOverlayEpochTicket *ticket,
            DeviceMoEOverlayEpochStatus *status)
        {
            (void)launch;
            (void)control;
            (void)ticket;
            (void)status;
            return false;
        }

        /**
         * @brief Reserve the reusable non-published bank for a newer epoch.
         *
         * @p candidate_epoch points to backend-owned live state so a captured
         * maintenance graph can process successive epochs without recapture.
         */
        virtual bool reserveMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status)
        {
            (void)launch;
            (void)control;
            (void)candidate_epoch;
            (void)status;
            return false;
        }

        /**
         * @brief Normalize and atomically publish a completed durable rebalance.
         *
         * The prior reserve status selects the one writable peer bank. A
         * successful apply may have changed only a subset of layers, so this
         * operation clones every untouched published layer into that peer,
         * stamps one common epoch, and switches ticket admission only after the
         * complete family is visible. A no-work apply aborts the candidate; a
         * busy/failed reservation is a device-side no-op. On successful
         * publication @p candidate_epoch advances in place for graph replay.
         *
         * @param launch Exact maintenance stream/workspace binding.
         * @param runtime_layers Canonical main-model placement table.
         * @param layer_count Complete durable model-layer count.
         * @param expert_count Logical experts per layer.
         * @param control Authoritative epoch publication control.
         * @param candidate_epoch Persistent next-epoch scalar.
         * @param reservation_and_publication_status Reserve input and terminal output.
         * @param apply_status Device-owned result of the immediately preceding apply.
         * @return True when the operation executed or was enqueued.
         */
        virtual bool finalizeMoEOverlayRebalancePublication(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            std::uint32_t layer_count,
            std::uint32_t expert_count,
            DeviceMoEOverlayEpochControl *control,
            std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *reservation_and_publication_status,
            const DeviceMoERebalanceApplyStatus *apply_status)
        {
            (void)launch;
            (void)runtime_layers;
            (void)layer_count;
            (void)expert_count;
            (void)control;
            (void)candidate_epoch;
            (void)reservation_and_publication_status;
            (void)apply_status;
            return false;
        }

        /**
         * @brief Mark a reserved epoch ready after every transfer event is joined.
         */
        virtual bool markMoEOverlayEpochCandidateReady(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status)
        {
            (void)launch;
            (void)control;
            (void)candidate_epoch;
            (void)status;
            return false;
        }

        /**
         * @brief Atomically publish a ready epoch and begin retiring its predecessor.
         *
         * This is the sole device ticket-admission linearization point.  The
         * host residency authority must already accept the candidate epoch
         * before a GPU maintenance stream invokes this operation.
         */
        virtual bool publishMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status)
        {
            (void)launch;
            (void)control;
            (void)candidate_epoch;
            (void)status;
            return false;
        }

        /** @brief Abort one unpublished Candidate or Ready epoch. */
        virtual bool abortMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status)
        {
            (void)launch;
            (void)control;
            (void)candidate_epoch;
            (void)status;
            return false;
        }

        /**
         * @brief Reclaim a retiring bank only after its device grace period.
         *
         * A Busy status is expected while either an admission is in flight or
         * an old request still holds the named epoch.  Callers poll with events;
         * inference never waits for this maintenance operation.
         */
        virtual bool retireMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *retiring_epoch,
            DeviceMoEOverlayEpochStatus *status)
        {
            (void)launch;
            (void)control;
            (void)retiring_epoch;
            (void)status;
            return false;
        }

        /**
         * @brief Graph-capturable device-side MoE rebalance publish/apply.
         *
         * Backends consume a device-resident gathered histogram buffer with
         * layout [participant][layer][expert]. Immediate resident-only policy
         * may publish directly into the inactive durable runtime bank. A
         * deferred movement policy must instead derive its candidate into
         * @p placement_plan_scratch, emit immutable commands, and leave both
         * durable banks untouched until the arrival/apply transaction has
         * completed. This separation is the device-side RCU guarantee: active
         * readers cannot observe a merely planned placement and the inactive
         * bank remains available to clone the next committed epoch.
         *
         * Expert payload movement is never performed by this call. Arrival
         * kernels must first populate the preallocated directory slots named by
         * the commands; the later apply kernel validates those leases, copies
         * changed rows from scratch, clones unchanged rows from the active
         * bank, and only then makes the candidate eligible for publication.
         *
         * @param placement_plan_scratch Required graph-owned candidate storage
         *        when `DeferRuntimeApply` is set; ignored by immediate policy.
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
            DeviceMoELLEPLayerPlanScratch *llep_layer_plans = nullptr,
            DeviceMoEPlacementBank *placement_plan_scratch = nullptr)
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
            (void)placement_plan_scratch;
            return false;
        }

        /**
         * @brief Pack per-layer runtime histograms into a contiguous device buffer.
         *
         * The graph-captured rebalance path all-gathers a flat
         * [wave_layer][expert] histogram from every participant before launching
         * runDeviceRebalanceController(). GPU backends implement this as a small
         * device kernel that reads the selected phase counters directly from the
         * mirrored runtime table; no host sync or host copy is permitted on this
         * path. The optional wave/controller pointers let the device packer follow
         * the same rolling wave cursor as the controller.
         *
         * @param histogram_source_mask Bit mask of
         *        `moe_runtime_abi::HistogramSource` values to combine. The default
         *        preserves the graph-native same-domain aggregate policy.
         * @param previous_activation_counts Optional device-owned cumulative
         *        baseline indexed by `[layer][expert]`. When present, the packer
         *        publishes `current - previous` and advances the baseline in the
         *        same kernel. A counter-generation reset is represented by
         *        `current < previous` and begins a fresh window at `current`.
         */
        virtual bool packDeviceRebalanceHistograms(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            uint64_t *local_histograms,
            const DeviceMoERebalanceConfig &config,
            const DeviceMoERebalanceWaveState *wave_state = nullptr,
            const DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1,
            uint32_t histogram_source_mask =
                moe_runtime_abi::kAllHistogramSourcesMask,
            uint64_t *previous_activation_counts = nullptr)
        {
            (void)launch;
            (void)runtime_layers;
            (void)local_histograms;
            (void)config;
            (void)wave_state;
            (void)controller_state;
            (void)command_buffer_count;
            (void)histogram_source_mask;
            (void)previous_activation_counts;
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
            uint32_t local_transfer_slot_count = 0,
            DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index = nullptr)
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
            (void)transfer_slot_claim_index;
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
            DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index,
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
            (void)transfer_slot_claim_index;
            (void)command_buffer_count;
            return false;
        }

        /**
         * @brief Materialize a mirrored current-batch LLEP domain request set.
         *
         * Homogeneous LocalTP prefill participants consume byte-identical
         * router choices and mirrored runtime ownership, so the deterministic
         * current-batch planner publishes the same logical transfer list on
         * every participant. This graph-capturable operation expands that
         * shared list into the participant-major layout normally produced by
         * metadata allgathers. The following domain projection still leases
         * destination-local physical slots and the payload collective still
         * transports packed bytes; only redundant plan/header collectives are
         * removed.
         *
         * Prefix-runtime rehydration must not use this operation because its
         * restored transfer requests are participant-owned rather than a
         * deterministic result of replicated current-batch routing.
         *
         * @param launch Explicit graph-owned launch stream and workspace.
         * @param runtime_layer Current layer runtime containing the shared
         *        planner transfer publication in `reserved_ptrs[2]`.
         * @param mirrored_plan_entries Participant-major logical requests with
         *        `participant_count * plan_capacity` writable entries.
         * @param mirrored_command_headers One writable header per participant.
         * @param plan_capacity Capacity of each participant request slice.
         * @param status Device-owned transaction status publication.
         * @param config Immutable participant topology and planner policy.
         * @param payload_slot_capacity Fixed compact slots available per
         *        source/destination lane.
         * @param layer_idx Logical routed-expert layer being materialized.
         * @return `true` when the backend enqueued the exact operation.
         */
        virtual bool materializePrefillLeastLoadedMirroredDomainCommands(
            const MoEKernelLaunchContext &launch,
            const DeviceMoELayerRuntime *runtime_layer,
            DeviceMoERebalancePlanEntry *mirrored_plan_entries,
            DeviceMoERebalanceCommandBufferHeader *mirrored_command_headers,
            uint32_t plan_capacity,
            DeviceMoERebalanceStatus *status,
            const DeviceMoERebalanceConfig &config,
            uint32_t payload_slot_capacity,
            uint32_t layer_idx);

        /**
         * @brief Materialize current-batch LLEP weight-transfer requirements
         *        into the standard rebalance command buffer ABI.
         *
         * planPrefillRoutesLeastLoadedCurrentBatch() writes Algorithm-4 style
         * expert-weight transfers into DeviceMoELayerRuntime::reserved_ptrs[2].
         * This graph-capturable bridge converts those records into
         * ExpertPayloadArrival commands so the existing compact payload
         * movement path can stage/import them. Each participant emits only
         * commands whose destination equals `config.participant_id`; the
         * subsequent domain projection reconstructs the common source-packing
         * plan after these destination-local request lists are gathered.
         * Commands produced here are logical, including prefix-runtime rehydration:
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
            uint32_t command_buffer_count = 1,
            const DeviceMoEOverlayEpochStatus *overlay_reservation_status = nullptr)
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
            (void)overlay_reservation_status;
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
         * @brief Publish a complete grouped plan directly from router outputs.
         *
         * Static-owner and fully replicated execution have no intervening
         * participant assignment phase. Their final publication therefore owns
         * route conversion, deterministic grouping, inverse-map construction,
         * compact descriptor materialization, and stable active-expert ids as
         * one stream-ordered transaction. Implementations may mode-shift by
         * geometry, but the method must publish every product before returning.
         */
        virtual bool publishCompleteGroupedPrefillPlanFromRouter(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *routing_indices,
            ITensor *routing_weights,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            int gateup_desc_table_id,
            int down_desc_table_id,
            bool filter_to_local_runtime_experts,
            bool retain_routes_for_deferred_commit = false)
        {
            (void)runtime_layer;
            (void)routing_indices;
            (void)routing_weights;
            (void)current_tokens;
            (void)max_tokens;
            (void)num_experts;
            (void)top_k;
            (void)gateup_desc_table_id;
            (void)down_desc_table_id;
            (void)filter_to_local_runtime_experts;
            (void)retain_routes_for_deferred_commit;
            return false;
        }

        /**
         * @brief Publish a complete grouped plan from final device assignments.
         *
         * LLEP rewrites participant ids after route-only grouping. This method
         * consumes that final device ledger and atomically defines the grouped
         * rows, inverse map, compact descriptor tables, active ids, and optional
         * deferred verifier ledger consumed by the following grouped compute.
         * No host observation or separate descriptor side effect is permitted.
         */
        virtual bool publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens,
            int max_tokens,
            int num_experts,
            int top_k,
            int gateup_desc_table_id,
            int down_desc_table_id,
            bool retain_routes_for_deferred_commit = false)
        {
            (void)runtime_layer;
            (void)current_tokens;
            (void)max_tokens;
            (void)num_experts;
            (void)top_k;
            (void)gateup_desc_table_id;
            (void)down_desc_table_id;
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
         * retention in groupPrefillRoutes() and complete-plan publication from
         * final runtime assignments. Grouping runs before acceptance is known
         * and may only retain immutable route evidence; it must not mutate
         * routing history. Callers must enqueue this commit on the exact stream
         * that owns accepted-state publication. The default hard failure keeps
         * an unimplemented backend from silently losing or overcounting decode
         * evidence.
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
        virtual bool executeGroupedPrefillPipelineFromPublishedRuntimePlan(
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
        /**
         * @brief Immutable-row publication shared with sibling consumers.
         *
         * This is intentionally separate from @ref kernel.  A shared expert
         * may consume the router's Q8 rows without gaining access to, or
         * aliasing, the routed pipeline's mutable grouping and descriptor
         * state.
         */
        std::shared_ptr<MoERouterQ8HiddenPublication> router_q8_publication =
            std::make_shared<MoERouterQ8HiddenPublication>();
    };

} // namespace llaminar2
