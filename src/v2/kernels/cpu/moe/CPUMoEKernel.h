/**
 * @file CPUMoEKernel.h
 * @brief CPU implementation of MoE kernel operations
 *
 * Implements IMoEKernel for CPU execution using ISA-dispatched vector
 * primitives (AVX2/AVX-512). This is the reference implementation;
 * GPU implementations (CUDA/ROCm) can override for device-native execution.
 */

#pragma once

#include "../../IMoEKernel.h"
#include "../CPUKernelBase.h"
#include "../../../tensors/BlockStructures.h"

#include <span>
#include <vector>

namespace llaminar2
{

    /**
     * @brief CPU implementation of MoE kernel operations
     *
     * Uses ISA-dispatched vector primitives for:
     * - Router: socket-wide row/expert dot products, SIMD softmax, partial-sort top-k
     * - Gather: memcpy-based token collection
     * - Scatter: vec_axpy for weighted accumulation
     * - Shared expert gate: vec_dot + sigmoid + vec_scale
     * - SwiGLU: ISA-dispatched silu(gate) * up
     */
    class CPUMoEKernel : public IMoEKernel, public CPUKernelBase
    {
    public:
        CPUMoEKernel() = default;
        ~CPUMoEKernel() override = default;

        // =================================================================
        // IMoEKernel interface
        // =================================================================

        /**
         * @brief Route one or more hidden rows with serial-decode arithmetic.
         *
         * Independent row/expert dot products occupy the complete OpenMP team.
         * Each row is then finalized by one worker using the same softmax,
         * top-k ordering, summation, and normalization as M=1 decode. The
         * method also publishes canonical Q8_1 hidden rows for the paired
         * NativeVNNI expert consumer.
         *
         * @param hidden Contiguous FP32 hidden rows `[seq_len, d_model]`.
         * @param gate_weights FP32 router matrix `[num_experts, d_model]`.
         * @param seq_len Positive runtime row count.
         * @param d_model Hidden width shared by inputs and gate rows.
         * @param num_experts Number of router output columns.
         * @param top_k Number of unique winning experts retained per row.
         * @param normalize_weights Whether selected probabilities sum to one.
         * @param result Persistent host-owned output storage.
         * @return True after every route and Q8 publication is complete.
         */
        bool route(
            const float *hidden,
            const float *gate_weights,
            int seq_len, int d_model,
            int num_experts, int top_k,
            bool normalize_weights,
            MoERoutingResult &result);

        /**
         * @brief Route CPU tensor rows and publish both tensor and host outputs.
         *
         * CPU expert dispatch is host-owned, so the CPU implementation is the
         * one backend where @p host_result is a production output rather than
         * an obsolete GPU mirror.
         */
        bool routeWithTensors(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result) override;

        /** @brief Acquire the current CPU-owned ExpertOverlay placement epoch. */
        bool acquireMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            DeviceMoEOverlayEpochTicket *ticket,
            DeviceMoEOverlayEpochStatus *status,
            const std::uint64_t *external_admission_epoch = nullptr,
            DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier = {},
            MoEOverlayPeerPlacementEpochBinding peer_placement_epoch = {}) override;

        /** @brief Release and clear one CPU ExpertOverlay placement ticket. */
        bool releaseMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            DeviceMoEOverlayEpochTicket *ticket,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @brief Reserve the CPU control block's reusable placement bank. */
        bool reserveMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @brief Mark a CPU placement candidate ready for publication. */
        bool markMoEOverlayEpochCandidateReady(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @brief Publish a ready CPU placement candidate without waiting on readers. */
        bool publishMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @brief Abort an unpublished CPU placement candidate. */
        bool abortMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @brief Reclaim a CPU placement bank after its complete grace period. */
        bool retireMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *retiring_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        void gatherTokenBatch(
            const float *hidden,
            float *batch_buffer,
            const int *token_indices,
            int num_tokens, int d_model) override;

        void scatterAddWeighted(
            float *output,
            const float *expert_output,
            const int *token_indices,
            const float *weights,
            int num_tokens, int d_model) override;

        void sharedExpertGate(
            const float *input,
            const float *gate_inp,
            float *shared_output,
            int seq_len, int d_model) override;

        void swiGLU(float *gate, const float *up, int count) override;

        /**
         * @brief Return the router-owned Q8_1 rows for the next grouped expert pass.
         *
         * CPU NativeVNNI expert projections consume Q8_1 activations.  Routing
         * and expert execution therefore share one canonical transform instead
         * of quantizing the same hidden row again for every routed expert.  A
         * publication is valid only for the exact FP32 source address, requested
         * row count, and model width that produced it.  Callers must treat a null
         * result as a broken production ordering contract; grouped CPU verifier
         * execution has no requantization fallback.
         *
         * @param source FP32 hidden-row base passed to route().
         * @param rows Number of contiguous verifier rows the consumer needs.
         * @param d_model Number of FP32 values in each row.
         * @return Contiguous `[rows, ceil(d_model / 32)]` Q8_1 blocks, or nullptr
         *         when no matching router publication exists.
         */
        const Q8_1Block *publishedRouterQ8Hidden(
            const float *source,
            int rows,
            int d_model) const noexcept;

        /**
         * @brief Publish canonical Q8_1 rows received across a host boundary.
         *
         * ExpertOverlay routing may execute on a GPU while a participant-local
         * NativeVNNI expert executes on CPU. The fixed-capacity ticket carries
         * the authoritative FP32 rows, so the CPU boundary must perform the
         * same one-time canonical transform normally owned by CPU route(). This
         * is an explicit transport adaptation, not a missing-publication
         * recovery path; callers select it through a typed stage policy.
         *
         * @param source Authoritative transported FP32 row base.
         * @param rows Number of contiguous transported rows.
         * @param d_model Logical values per row.
         * @return true after publishing byte-identical canonical Q8_1 blocks.
         */
        bool publishTransportedRouterQ8Hidden(
            const float *source,
            int rows,
            int d_model);

        /**
         * @brief Publish transported rows directly in expert-major route order.
         *
         * A heterogeneous CPU endpoint already knows its stable expert-major
         * route schedule before it prepares NativeVNNI input. Materializing a
         * row-major Q8_1 tensor and then gathering those same bytes performs an
         * avoidable second activation pass. This transport-specific operation
         * applies the canonical Q8_1 transform directly into caller-owned
         * expert-major storage. Repeated source indices deliberately produce
         * repeated byte-identical rows; no quantized value is shared or
         * arithmetically combined.
         *
         * This method revokes any ordinary row-major publication owned by this
         * kernel. It is not a fallback for a missing CPU router publication;
         * only a stage with the typed transported-row policy may call it.
         *
         * @param source Authoritative contiguous transported FP32 rows.
         * @param source_rows Number of valid rows in @p source.
         * @param d_model Logical FP32 values per source row.
         * @param expert_major_source_rows Source row index for each output row.
         * @param destination Caller-owned expert-major Q8_1 block storage.
         * @return True after every selected row is published exactly once.
         */
        bool publishTransportedRouterQ8HiddenExpertMajor(
            const float *source,
            int source_rows,
            int d_model,
            std::span<const int> expert_major_source_rows,
            std::span<Q8_1Block> destination);

        /**
         * @brief Reserve and first-touch transported-router publication storage.
         *
         * Serial ExpertOverlay graph families know their maximum compact row
         * geometry during setup. Reserving here prevents the first production
         * request from changing the CPU kernel's memory topology.
         *
         * @param rows Maximum positive compact row count.
         * @param d_model Positive hidden width.
         * @return True when the complete Q8_1 publication fits the reservation.
         */
        bool reserveRouterQ8HiddenCapacity(int rows, int d_model);

        // =================================================================
        // ITensorKernel interface
        // =================================================================

        bool supports_device(int device_idx) const override
        {
            return device_idx < 0; // CPU only (device_idx == -1)
        }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::passthrough();
        }

    private:
        /** Revoke publication metadata before beginning a new routing call. */
        void invalidateRouterQ8HiddenPublication() noexcept;

        /**
         * @brief Quantize and publish runtime-M router inputs for NativeVNNI experts.
         *
         * The implementation deliberately uses the same Q8_1 block primitives
         * as serial NativeVNNI decode.  Keeping this transform in the router
         * makes grouped expert execution both economical and byte-equivalent.
         */
        bool publishRouterQ8Hidden(
            const float *source,
            int rows,
            int d_model);

        /**
         * @brief Quantize contiguous or indexed rows into an exact Q8_1 target.
         *
         * @param source Contiguous FP32 source-row base.
         * @param source_rows Number of addressable source rows.
         * @param d_model Logical values per source row.
         * @param source_row_indices Optional output-to-source row map; null is identity.
         * @param output_rows Number of Q8_1 rows to write.
         * @param destination Destination block base.
         * @param destination_blocks Available destination block count.
         * @return True when the complete validated publication was written.
         */
        bool quantizeRouterQ8Rows(
            const float *source,
            int source_rows,
            int d_model,
            const int *source_row_indices,
            int output_rows,
            Q8_1Block *destination,
            size_t destination_blocks);

        std::vector<Q8_1Block> router_q8_hidden_; ///< Canonical contiguous verifier rows.
        const float *router_q8_hidden_source_ = nullptr; ///< FP32 producer identity.
        int router_q8_hidden_rows_ = 0; ///< Number of published rows.
        int router_q8_hidden_d_model_ = 0; ///< Published row width.
        bool router_q8_hidden_valid_ = false; ///< Metadata and payload are current.
    };

} // namespace llaminar2
