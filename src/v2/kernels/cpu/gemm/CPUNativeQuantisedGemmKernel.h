/**
 * @file CPUNativeQuantisedGemmKernel.h
 * @brief ITensorGemm implementation for CPU NativeQuantised GEMM/GEMV.
 *
 * Immutable prepared weights use the canonical NativeQuantised format registry and
 * AVX2/AVX512 runtime dispatch. Batched descriptor objects occupy only the live
 * prefix of invocation-owned fixed stack storage, with no heap allocation.
 * Serial decode and grouped verification retain their explicit numerical policy
 * and reduction order. Method implementations live in `CPUNativeQuantisedGemmKernel.cpp`
 * so callers including this header (such as `TensorSlice.h`) do not pull in the
 * templated AVX2/AVX512/NEON GEMV kernels.
 */

#pragma once

#include "../../../execution/moe/DeviceMoERuntimeABI.h"
#include "CPUQ8ResidualCorrectionWorkspace.h"
#include "CPUProjectionWorkspaceContract.h"
#include "CPUNativeQuantisedExecutionContract.h"
#include "CPUNativeQuantisedPackedWeights.h"
#include "CPUPackedWeights.h"
#include "tensors/TensorKernels.h"
#include "utils/PerfStatsCollector.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace llaminar2
{
    class ActivationRotation;
}

namespace llaminar2::cpu::native_quantised
{
    struct FusedVerifierRowsDesc;

    /**
     * @brief RAII publication of serial TP output geometry for CPU NativeQuantised.
     */
    class CPUNativeQuantisedOutputPartitionEquivalenceScope final
        : public ITensorGemm::OutputPartitionEquivalenceScope
    {
    public:
        /** Publish @p serial_partition_n until this object is destroyed. */
        explicit CPUNativeQuantisedOutputPartitionEquivalenceScope(int serial_partition_n);

        /** Restore the enclosing output-partition policy. */
        ~CPUNativeQuantisedOutputPartitionEquivalenceScope() override;

        CPUNativeQuantisedOutputPartitionEquivalenceScope(
            const CPUNativeQuantisedOutputPartitionEquivalenceScope &) = delete;
        CPUNativeQuantisedOutputPartitionEquivalenceScope &operator=(
            const CPUNativeQuantisedOutputPartitionEquivalenceScope &) = delete;

    private:
        int previous_ = 0; ///< Scope value restored at transaction completion.
    };

    /** @brief Bind one authenticated source interval; sibling projections never enter this scope. */
    class CPUNativeQuantisedOriginalOutputSliceScope final
        : public ITensorGemm::OutputPartitionEquivalenceScope
    {
    public:
        /** @brief Preserve both enclosing APIs while recording this leaf's original arithmetic. */
        explicit CPUNativeQuantisedOriginalOutputSliceScope(const OriginalOutputSlice &slice);

        /** @brief Restore even after a failed launch or nested legacy scope. */
        ~CPUNativeQuantisedOriginalOutputSliceScope() override;

        CPUNativeQuantisedOriginalOutputSliceScope(const CPUNativeQuantisedOriginalOutputSliceScope &) = delete;
        CPUNativeQuantisedOriginalOutputSliceScope &operator=(const CPUNativeQuantisedOriginalOutputSliceScope &) = delete;

    private:
        struct SavedPolicy
        {
            int physical_columns = 0;
            int policy_columns = 0;
        };
        SavedPolicy previous_{}; ///< Immutable predecessor geometry.
        int previous_serial_ = 0; ///< Existing two-int predecessor policy.
    };

    /** @brief Prepared quantized weights with participant-owned invocation scratch. */
    class CPUNativeQuantisedGemmKernel : public ITensorGemm, public CPUInvocationWorkspaceConsumer
    {
    public:
        /**
         * @brief Construct from a quantized weight tensor.
         *
         * Packs weights into the CPU NativeQuantised layout at construction time.
         * The tensor must implement IINT8Unpackable and provide vnniFormatInfo().
         *
         * @param weights Source weight tensor [N, K]
         * @param row_start Start row for TP slicing (default 0)
         * @param row_end End row for TP slicing (default -1 = all)
         * @param numerical_policy Arithmetic contract carried by the prepared engine.
         * @param placement Final CPU storage placement established before packing.
         */
        explicit CPUNativeQuantisedGemmKernel(const TensorBase *weights,
                                         int row_start = 0, int row_end = -1,
                                         CPUProjectionNumericalPolicy numerical_policy =
                                             CPUProjectionNumericalPolicy::BackendNative,
                                         CPUWeightStoragePlacement placement = CPUWeightStoragePlacement::local());

        /**
         * @brief Construct from pre-packed weights (move).
         */
        explicit CPUNativeQuantisedGemmKernel(
            CPUNativeQuantisedPackedWeights &&packed,
            CPUProjectionNumericalPolicy numerical_policy =
                CPUProjectionNumericalPolicy::BackendNative);

        ~CPUNativeQuantisedGemmKernel() override = default;

        /**
         * @brief Declare the input bank without retaining a participant binding.
         * @param m Maximum simultaneous input rows for this projection.
         * @param k Explicit K geometry, or zero to use the prepared matrix width.
         * @return Q8 and ordered-partial banks, plus the optional rotated input tile.
         */
        WorkspaceRequirements getWorkspaceRequirements(int m, int = 0, int k = 0) const override;

        /** @brief Preserve the beta addend only when the owning stage selects accumulation. */
        void appendOutputAccumulationWorkspaceRequirements(
            WorkspaceRequirements &requirements, int m, int n) const override;

        /** @copydoc ITensorGemm::q8ResidualCorrectionWorkspaceRequirements */
        WorkspaceRequirements q8ResidualCorrectionWorkspaceRequirements(int m,int n,int k) const override;
        /** @copydoc ITensorGemm::multiply_q8_residual_corrected */
        void multiply_q8_residual_corrected(const TensorBase*input,TensorBase*output,DeviceRowRange rows,
            int n,int k,DeviceWorkspaceManager*workspace) override;

        /** @copydoc ITensorGemm::multiply_q8_residual_corrected */
        void multiply_q8_residual_corrected(HCResidualDownInput hc,TensorBase*output,DeviceRowRange rows,
            int n,int k,DeviceWorkspaceManager*workspace) override;

        // -------------------------------------------------------------------
        // ITensorKernel interface
        // -------------------------------------------------------------------

        bool supports_device(int device_idx) const override;

        /** @brief Export exact CPU-packed source codebook and superblock identity. */
        bool exportNativeQuantisedSourceIdentity(
            NativeQuantisedSourceIdentity &out) const override;

        // -------------------------------------------------------------------
        // ITensorGemm interface
        // -------------------------------------------------------------------

        /**
         * @brief Bind mirrored full-width execution to serial TP shard geometry.
         *
         * CPU NativeQuantised output columns are independent once activation Q8_1
         * quantization is complete. Reusing the serial shard's generated policy,
         * N-task width, and K-partition tree therefore produces exactly the same
         * bytes while one replicated invocation writes all vocabulary columns.
         *
         * @param actual_output_columns Full output width owned by this kernel.
         * @param serial_partition_columns Column width used by serial TP decode.
         * @return A caller-thread RAII scope restored after the LM-head launch.
         */
        std::unique_ptr<OutputPartitionEquivalenceScope>
        beginOutputPartitionEquivalenceScope(
            int actual_output_columns,
            int serial_partition_columns) override;

        /** @brief Bind store-authenticated original N without changing packed bytes or physical extents. */
        std::unique_ptr<OutputPartitionEquivalenceScope>
        beginOriginalOutputSliceScope(const OriginalOutputSlice &slice) override;

        /**
         * @brief C[m×n] = A[m×k] @ B_packed[n×k]^T
         *
         * @param A Immutable FP32 input with original physical row coordinates.
         * @param C Physical-M output; unused query rows remain unwritten.
         * @param m Original physical matrix row capacity.
         * @param n Original prepared output width.
         * @param k Original prepared input width.
         * @param transpose_B Existing native prepared orientation.
         * @param alpha Original product multiplier.
         * @param beta Original destination multiplier.
         * @param bias Optional immutable bias; typed query construction admits none.
         * @param mpi_ctx Existing CPU/MPI execution context.
         * @param device_idx CPU execution selector, required to be -1.
         * @param workspace Participant-admitted stable scratch.
         * @param activation_row_offset Original input row offset.
         * @return Whether the original prepared operation completed successfully.
         * @throws std::invalid_argument For work-bearing non-M1 ordinary calls.
         * Primary tensor-aware GEMM entry point.
         * For M=1: dispatches to optimized GEMV path.
         * For M>1: dispatches to tiled GEMM path.
         */
        bool multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const TensorBase *bias = nullptr,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr,
            int activation_row_offset = 0) override;

        /** @brief Borrow the engine-owned final CPU migration representation. */
        const CPUNativeQuantisedPackedWeights *
        exportCPUNativeQuantisedPackedWeights() const override;

        /**
         * @brief Expose final storage to the ticket-gated overlay slot arena.
         *
         * The arena holds this engine for the model lifetime and writes only
         * while its physical slot is inactive. Returning the existing range
         * does not resize, allocate, change the GEMM object's address or the
         * current publication: an adopted static engine keeps serving until its
         * slot is acquired, which opens a rewrite through
         * beginRetiredCPUNativeQuantisedRewrite().
         */
        std::span<std::uint8_t>
        exportRetiredCPUNativeQuantisedStorage() noexcept override;

        /**
         * @brief Open a new publication generation for an acquired overlay slot.
         * @return The same final byte range; it stays unadmitted until resealed.
         */
        std::span<std::uint8_t> beginRetiredCPUNativeQuantisedRewrite() noexcept override;

        /**
         * @brief Admit completely written slot bytes before their engine becomes reachable.
         * @throws std::invalid_argument For reserved metadata bits or non-canonical storage; the engine stays unadmitted.
         * @throws std::overflow_error For unrepresentable geometry.
         * @throws std::logic_error For a retired engine or owner fields that disagree with the canonical geometry.
         * @details A current seal is left untouched, so republishing an unchanged
         * engine never writes state that concurrent readers observe.
         */
        void admitCPUNativeQuantisedPublication() override;

        // -------------------------------------------------------------------
        // Weight Lifecycle (IPackedWeights integration)
        // -------------------------------------------------------------------

        std::unique_ptr<IPackedWeights> detachWeights() override;

        std::unique_ptr<IPackedWeights> cloneWeights() const override;

        /**
         * @brief Clone published bytes through the explicit destination authority.
         * @param destination Admitted final CPU placement for the new owner.
         * @return Independent packed weights, or empty when this engine is retired.
         */
        std::unique_ptr<IPackedWeights> cloneWeightsTo(CPUWeightStoragePlacement destination) const override;

        bool attachWeights(std::unique_ptr<IPackedWeights> weights) override;

        void releaseWeights() override;

        bool hasWeights() const override { return valid_; }

        size_t packedWeightBytes() const override;

        bool canReleaseSourceWeightTensor() const override;

        // -------------------------------------------------------------------
        // Accessors
        // -------------------------------------------------------------------

        bool isValid() const { return valid_; }

        const CPUNativeQuantisedPackedWeights &packedWeights() const { return packed_; }

        uint8_t codebookId() const { return packed_.codebook_id; }

        int get_n() const override { return packed_.N; }
        int get_k() const override { return packed_.K; }

        /** @brief Return whether this kernel applies an activation rotation before quantization. */
        bool hasActivationRotation() const noexcept { return activation_rotation_ != nullptr; }

        /**
         * @brief Pre-resolved single-row pointers and scratch spans for retained OpenMP team execution.
         *
         * Populated on Thread 0 before entering an enclosing @c OMP_LAYER_PARALLEL region so
         * all workers invoke @ref executeSingleRowInTeam without locks, workspace map lookups,
         * or host tensor metadata queries.
         */
        struct RetainedSingleRowBinding
        {
            const float *input = nullptr;
            float *output = nullptr;
            float *hc_activated = nullptr;
            std::span<Q8_1Block> q8_primary;
            std::span<Q8_1Block> q8_residual;
            std::span<float> partials;
            int serial_partition_n = 0; ///< Optional mirrored-TP serial partition policy_n (0 = none).
            bool two_term = false;
        };

        /**
         * @brief Validate and pre-resolve raw single-row pointers and workspace spans on Thread 0.
         * @return True when @p out is complete and safe for collective @ref executeSingleRowInTeam.
         */
        bool prepareSingleRowRetainedBinding(
            const TensorBase *A,
            TensorBase *C,
            TensorBase *hc_activated,
            int n,
            int k,
            bool two_term,
            DeviceWorkspaceManager *workspace,
            RetainedSingleRowBinding &out,
            int serial_partition_n = 0) const;

        /**
         * @brief Execute one single-row projection collectively across an already-active OpenMP team.
         *
         * Must be entered by every thread in the active team with an identical @p binding produced
         * by @ref prepareSingleRowRetainedBinding. Ends on an implicit OpenMP barrier publishing
         * @c binding.output (and @c binding.hc_activated for HC two-term projections).
         */
        void executeSingleRowInTeam(const RetainedSingleRowBinding &binding) const;

        // -------------------------------------------------------------------
        // Fused SwiGLU + GEMM: output = W @ (silu(gate) * up)
        // -------------------------------------------------------------------

        /**
         * @brief Fused SwiGLU activation + GEMM on CPU.
         *
         * Computes: output = W_down @ (silu(gate) * up)
         * SwiGLU is applied to the input BEFORE quantization and GEMM,
         * which is mathematically correct (gate and up share dimension K,
         * while output has dimension N ≠ K).
         */
        bool multiply_tensor_with_fused_swiglu(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha = 1.0f, float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr) override;

        /**
         * @brief Grouped MTP verifier SwiGLU + down projection.
         *
         * This follows the same SwiGLU math as the ordinary CPU down path, but
         * M=1 is the valid no-accepted-draft verifier extent, including predictor
         * bootstrap. It remains on this same prepared primitive. This path
         * sends M1 through the prepared decode GEMV and M>=2 through
         * gemm_native_quantised_preq_decode_equivalent_rows(). The grouped helper shares
         * Q8_1 activation quantization across rows and parallelizes the work,
         * while each row keeps the same K-tile reduction order as M=1 decode.
         */
        bool multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha = 1.0f, float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr) override;

        // -------------------------------------------------------------------
        // Fused multi-projection with quantize-once + epilogues
        // -------------------------------------------------------------------

        bool supports_fused_projection() const override;

        bool multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override;

        /**
         * @brief Group MTP verifier rows while preserving M=1 decode GEMV math.
         *
         * This is the CPU implementation of ITensorGemm's Phase 9.8 verifier
         * contract, including M=1 with no accepted draft. Zero physical rows
         * remain invalid; compact CPU predictor admission owns live-row checks.
         * It quantizes the M verifier activation rows once, then
         * uses the prepared decode GEMV for one projection at M1; grouped
         * rows use gemm_native_quantised_preq_decode_equivalent_rows(), while fused
         * projection bundles use their existing positive-M descriptor scheduler.
         * That helper parallelizes across rows and N/K tasks, but each row uses
         * the same chunk kernels and reduction order as serial decode.  The
         * result is grouped/concurrent execution without the 2-row GEMM
         * accumulation drift that can poison GDN/short-conv state publication.
         */
        bool multiply_fused_verifier_rows_decode_equivalent(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override;

        /**
         * @brief One independently routed pre-quantized projection.
         *
         * A CPU MoE layer contains many small expert matrices with unequal row
         * counts. Describing each matrix explicitly lets the NativeQuantised
         * scheduler execute the complete layer under one persistent OpenMP
         * team while preserving each row's serial-decode arithmetic contract.
         */
        struct BatchedPrequantizedProjectionDesc
        {
            CPUNativeQuantisedGemmKernel *kernel = nullptr; ///< Prepared weight owner.
            const Q8_1Block *input_q8 = nullptr;       ///< Projection-local rows.
            float *output = nullptr;                   ///< Row-major destination.
            const float *bias = nullptr;               ///< Optional projection bias.
            int rows = 0;                              ///< Runtime rows for this expert.
            int n = 0;                                 ///< Logical output columns.
            int ldc = 0;                               ///< Destination row stride.
            /**
             * Grouped-row policy override used by byte-exact certification.
             * Production descriptors retain Auto. One-row projections ignore
             * legacy grouped overrides because M=1 has a distinct generated
             * authority. OrderedThreeRows requires M>=3 and is rejected before
             * singleton normalization, never advertised as an executed override.
             */
            VerifierRowsPolicy verifier_schedule = VerifierRowsPolicy::Auto;
            /** Distinguish prompt work from equally large speculative batches. */
            ProjectionRowsPurpose purpose = ProjectionRowsPurpose::Decode;
            CPUProjectionOutputRows output_rows; ///< Checked final row binding for indexed publication.
        };

        /** Maximum gate/up descriptors for routed experts plus one shared expert. */
        static constexpr int kMaxBatchedPrequantizedProjections =
            2 * (moe_runtime_abi::kMaxExperts + 1);

        /** @brief Original route ownership for one indexed Down publication bank. */
        struct GroupedDownRouteOrder
        {
            std::span<const int> expert_major_destinations; ///< Final destination row for each expert-major row.
            std::span<const int> local_to_expert; ///< Inverse of the transport's compact local route ordering.
            std::span<const int> destination_to_local; ///< Original slot-to-local map, empty for compact packed records.
        };

        /**
         * @brief Resolve the measured standalone schedule before binding phase operands.
         * @param hidden Original hidden width.
         * @param intermediate Original expert width.
         * @param routes Number of originally admitted live local routes.
         * @return Fixed worker count on the supported physical schedule, or zero for another schedule.
         */
        static int standaloneMoEOneRowWorkers(int hidden, int intermediate, size_t routes) noexcept;

        /**
         * @brief Execute the prepared one-token native expert phases on their original fixed OpenMP team.
         * @param gate Gate and Up sources and destinations, adjacent in original local route order.
         * @param down Original Down sources, their activation Q8 rows and their route destinations.
         * @param activation Exact admitted middle Q8 workspace, with one row per live route.
         * @param hidden Original hidden width.
         * @param intermediate Original expert width.
         * @return Empty for other supported schedules, or whether the native team completed its publication.
         */
        static std::optional<bool> tryExecuteStandaloneMoEOneRow(std::span<const BatchedPrequantizedProjectionDesc> gate,
            std::span<const BatchedPrequantizedProjectionDesc> down, std::span<Q8_1Block> activation,
            int hidden, int intermediate);

        /**
         * @brief Check whether a one-token native MoE execution qualifies for in-team execution.
         * @details Same geometry and numerical contract as @ref standaloneMoEOneRowWorkers without
         *          rejecting an enclosing @ref OMP_LAYER_PARALLEL team.
         */
        static bool supportsStandaloneMoEOneRowInTeam(int hidden, int intermediate, size_t routes) noexcept;

        /**
         * @brief Minimum float capacity required for split-K scratch passed to @ref executeStandaloneMoEOneRowInTeam.
         */
        static constexpr size_t kStandaloneMoESplitKScratchFloats = 160u * 12u * 64u;

        /**
         * @brief Pre-resolved single-row standalone MoE tile plan prepared once per layer on one thread.
         *
         * Allows an enclosing @ref OMP_LAYER_PARALLEL megakernel to prepare the active experts'
         * packed pointers, K-partition counts, and codebook LUTs once inside an existing
         * @ref OMP_SINGLE block and then execute Gate/Up/SwiGLU/Q8_1/Down across all workers
         * with zero redundant per-thread setup or extra barriers.
         */
        struct StandaloneMoEOneRowPlan
        {
            struct alignas(64) RowEntry
            {
                const CPUNativeQuantisedPackedWeights *weights = nullptr;
                const Q8_1Block *input = nullptr;
                float *output = nullptr;
                void (*tile)(const RowEntry &, int) = nullptr;
                void (*part_range)(const RowEntry &, int, int, int, float *, bool) = nullptr;
                alignas(64) std::array<std::uint8_t, 64> lut{};
                int blocks = 0;
                int partitions = 0;
                ISALevel isa = ISALevel::Scalar;
            };
            static constexpr size_t kMaxRoutes = 16;
            std::array<RowEntry, 2 * kMaxRoutes> gate_plans{};
            std::array<RowEntry, kMaxRoutes> down_plans{};
            size_t route_count = 0;
            int hidden = 0;
            int intermediate = 0;
            int workers = 1;
            int gate_partitions = 0;
            int k_split = 1;
            ISALevel isa = ISALevel::Scalar;
            bool valid = false;
        };

        /**
         * @brief Pre-resolve immutable RowEntry metadata (weights, tile fn, LUT, partitions) once at setup.
         */
        static bool prepareStandaloneMoEOneRowEntry(
            StandaloneMoEOneRowPlan::RowEntry &out,
            const CPUNativeQuantisedGemmKernel *kernel,
            int width, int k, int workers, ISALevel isa);

        /**
         * @brief Finalize a @ref StandaloneMoEOneRowPlan whose `gate_plans`/`down_plans` were populated from cached entries.
         */
        static bool finalizeStandaloneMoEOneRowPlan(
            StandaloneMoEOneRowPlan &plan,
            size_t route_count,
            int hidden, int intermediate,
            int workers, ISALevel isa,
            size_t splitk_partials_capacity) noexcept;

        /**
         * @brief Prepare a @ref StandaloneMoEOneRowPlan once on one thread (e.g. inside `OMP_SINGLE`).
         */
        static bool prepareStandaloneMoEOneRowPlan(
            StandaloneMoEOneRowPlan &plan,
            std::span<const BatchedPrequantizedProjectionDesc> gate,
            std::span<const BatchedPrequantizedProjectionDesc> down,
            int hidden, int intermediate,
            int workers, ISALevel isa,
            size_t splitk_partials_capacity);

        /**
         * @brief Execute a pre-built @ref StandaloneMoEOneRowPlan inside an already-active OpenMP team.
         * @pre Every thread in the active OpenMP team observes the published @p plan identically.
         */
        static bool executePreparedStandaloneMoEOneRowInTeam(
            const StandaloneMoEOneRowPlan &plan,
            std::span<Q8_1Block> activation,
            float *splitk_partials, size_t splitk_partials_capacity);

        /**
         * @brief Execute the prepared one-token native MoE phases inside an already-active OpenMP team.
         * @pre All threads in the active OpenMP team call this identically; @p splitk_partials is
         *      shared across all threads with at least @ref kStandaloneMoESplitKScratchFloats floats.
         * @return True when every Gate/Up/SwiGLU/Q8_1/Down tile completed and synchronized.
         */
        static bool executeStandaloneMoEOneRowInTeam(
            std::span<const BatchedPrequantizedProjectionDesc> gate,
            std::span<const BatchedPrequantizedProjectionDesc> down,
            std::span<Q8_1Block> activation,
            int hidden, int intermediate,
            float *splitk_partials, size_t splitk_partials_capacity);

        /**
         * @brief Execute unequal-M pre-quantized projections in one CPU team.
         *
         * Every descriptor independently resolves its typed scheduling purpose:
         * sealed M=1/grouped decode policy, or the ordinary compact-prefill
         * policy, for its codebook, geometry, ISA, and row count. Both retain
         * the same serial-decode arithmetic tree. The
         * underlying launcher shares one persistent OpenMP team across the
         * entire descriptor set, eliminating per-expert team reconstruction.
         * No descriptor is replayed row by row and no alternate arithmetic
         * path is available.
         *
         * @param descriptors Complete projection set for one layer phase.
         * @param descriptor_count Number of valid entries in @p descriptors.
         * @param k Shared logical activation width.
         * @param partial_storage Admitted invocation bank shared by every team worker.
         * @param job_ownership Independent explicit ordinary-prefill output ownership.
         * @return `true` after every projection completes; `false` when the
         *         explicit eager/unrotated NativeQuantised contract is invalid.
         * @throws std::invalid_argument Before phase writes for unsupported explicit ownership.
         */
        static bool multiply_batched_preq_decode_equivalent(
            const BatchedPrequantizedProjectionDesc *descriptors,
            int descriptor_count,
            int k,
            std::span<float> partial_storage,
            FusedOutputJobOwnership job_ownership = FusedOutputJobOwnership::Auto);

        /**
         * @brief Execute one complete grouped CPU MoE FFN in a stable team.
         *
         * Sparse routed experts expose three dependent parallel phases. Starting
         * a fresh OpenMP region for gate/up, activation publication, and down
         * spends a material fraction of verifier latency in libgomp handoff and
         * completion barriers. This transaction makes the phase DAG explicit
         * and keeps one physical-core team alive across all three operations.
         *
         * Every worker enters every nested-safe workshare. Gate/up completion
         * therefore happens before the exact fused SwiGLU-to-Q8_1 publication,
         * and that publication completes before down projection begins. No
         * arithmetic is fused across those boundaries: each row retains the
         * same Q8 blocks, FP32 accumulation order, and rounding points as the
         * ordinary serial-decode-equivalent kernels.
         *
         * @param gate_up_descriptors Gate and up projections for all local experts.
         * @param gate_up_count Number of valid gate/up descriptors.
         * @param hidden_width Logical K dimension of gate/up matrices.
         * @param gate_rows Contiguous gate projection output rows.
         * @param up_rows Contiguous up projection output rows.
         * @param activation_q8 Destination for exact Q8_1 SwiGLU publication.
         * @param route_rows Total expert-major route rows in the transaction.
         * @param intermediate Expert hidden width and SwiGLU row width.
         * @param activation_blocks_per_row Q8_1 blocks in one SwiGLU row.
         * @param down_descriptors Down projections for all local experts.
         * @param down_count Number of valid down descriptors.
         * @param partial_storage Admitted bank reused only after each phase's readers retire.
         * @param gate_up_ownership Independent requested gate/up ownership; defaults to typed Auto.
         * @param down_ownership Independent requested down ownership; defaults to typed Auto.
         * @param down_route_order Optional original routing order for a complete shared Down bank.
         * @return `true` only when every team participant completes every phase.
         * @throws std::invalid_argument Before any phase writes for unsupported explicit ownership.
         */
        static bool execute_moe_grouped_ffn_transaction_preq_decode_equivalent(
            const BatchedPrequantizedProjectionDesc *gate_up_descriptors,
            int gate_up_count,
            int hidden_width,
            const float *gate_rows,
            const float *up_rows,
            Q8_1Block *activation_q8,
            int route_rows,
            int intermediate,
            int activation_blocks_per_row,
            const BatchedPrequantizedProjectionDesc *down_descriptors,
            int down_count,
            std::span<float> partial_storage,
            FusedOutputJobOwnership gate_up_ownership = FusedOutputJobOwnership::Auto,
            FusedOutputJobOwnership down_ownership = FusedOutputJobOwnership::Auto,
            std::optional<GroupedDownRouteOrder> down_route_order = std::nullopt);

        /**
         * @brief Execute a complete grouped MoE FFN (Gate+Up -> SwiGLU -> Q8_1 -> Down)
         *        inside one persistent OpenMP region with zero expert-major Q8 copy
         *        and zero gate/up DRAM round-trip.
         *
         * Consumes the unique row-major Q8_1 hidden rows directly via
         * `router_row_indices` (eliminating expert-major Q8 gathers), computes
         * Gate and Up into L1d-resident strips with exact ascending K-partition
         * order, immediately fuses SwiGLU and Q8_1 quantization into `activation_q8`,
         * synchronizes once, and executes weight-stationary Down directly into
         * `down_descriptors[e].output_rows`. Supports all 18 NativeQuantised
         * codebooks and runtime Scalar / AVX2 / AVX-512 ISA dispatch.
         *
         * @param gate_up_descriptors Gate and Up descriptors (2 per active expert).
         * @param unique_router_q8 Contiguous unique Q8_1 hidden rows `[unique_router_rows, router_q8_blocks_per_row]`.
         * @param unique_router_rows Number of unique source rows in `unique_router_q8`.
         * @param router_q8_blocks_per_row Number of Q8_1 blocks per hidden row.
         * @param router_row_indices Expert-major map from route row `[0, route_rows)` to unique row `[0, unique_router_rows)`.
         * @param hidden_width Logical K dimension of Gate/Up matrices (`d_model`).
         * @param activation_q8 Destination bank for exact Q8_1 SwiGLU publication.
         * @param route_rows Total expert-major route rows across active experts.
         * @param intermediate Expert intermediate width.
         * @param activation_blocks_per_row Number of Q8_1 blocks per intermediate row.
         * @param down_descriptors Down descriptors (1 per active expert).
         * @param down_route_order Optional route-partition validation for indexed Down publication.
         * @return `true` when every phase completes cleanly; `false` on contract violation.
         */
        static bool execute_moe_grouped_fused_ffn(
            std::span<const BatchedPrequantizedProjectionDesc> gate_up_descriptors,
            const Q8_1Block *unique_router_q8,
            int unique_router_rows,
            int router_q8_blocks_per_row,
            std::span<const int> router_row_indices,
            int hidden_width,
            Q8_1Block *activation_q8,
            int route_rows,
            int intermediate,
            int activation_blocks_per_row,
            std::span<const BatchedPrequantizedProjectionDesc> down_descriptors,
            std::optional<GroupedDownRouteOrder> down_route_order = std::nullopt);

        /**
         * @brief Project router-published Q8_1 rows without requantizing hidden state.
         *
         * Routed MoE prefill and verifier execution group route slots by expert.
         * A single hidden row can therefore appear in several expert batches;
         * quantizing each batch independently is both redundant and a source of
         * accidental divergence from serial decode. The CPU router publishes
         * the canonical Q8_1 row set once, and this method consumes gathered
         * blocks from that publication directly.
         *
         * M=1 batches use the ordinary fused decode GEMV kernel. Multi-row
         * batches use the grouped decode-equivalent kernel, whose per-row K
         * reduction order is identical to M=1 decode. The method has no FP32
         * alternative: every projection must be an eager, unrotated
         * NativeQuantised kernel with a compatible matrix shape.
         *
         * @param input_q8 Contiguous Q8_1 rows in `[m, ceil(k / 32)]` layout.
         * @param projections Gate/up projection bundle sharing the same rows.
         * @param m Number of gathered route rows. Runtime M is bounded only by
         *          the caller's declared graph and scratch capacity.
         * @param k FP32 logical width represented by each Q8_1 row.
         * @param partial_storage Caller-owned admitted partial bank for the complete bundle.
         */
        bool multiply_fused_router_q8_hidden_grouped_decode_equivalent(
            const Q8_1Block *input_q8,
            const std::vector<TensorProjectionDesc> &projections,
            int m,
            int k,
            std::span<float> partial_storage);

        // =====================================================================
        // Fused multi-input GEMV for MoE expert down projections
        // =====================================================================

        bool multiply_fused_expert_down(
            const FusedExpertDownDesc *descs, int num_descs,
            int m, int k, DeviceWorkspaceManager *workspace = nullptr) override;

    private:
        /**
         * @brief Prove shared-bank Down ownership using the router's existing inverse.
         * @param projections Validated Down descriptors in expert-bucket order.
         * @param route_rows Exact number of live local routes in the FFN.
         * @param order Original forward and inverse route maps, immutable through the workshare.
         * @throws std::invalid_argument When descriptors or route maps disagree.
         *
         * The descriptors must partition the same expert-major index span and
         * share a physical bank and pitch. An exact inverse proves that no two
         * rows can name the same slot, even when expert buckets interleave.
         */
        static void requireGroupedDownRoutePartition(
            std::span<const BatchedPrequantizedProjectionDesc> projections,
            int route_rows, const GroupedDownRouteOrder &order);

        /**
         * @brief Prove a complete indexed bundle's output rows are exclusive before worksharing.
         * @param projections Validated descriptors whose row addresses increase within each member.
         * @throws std::invalid_argument If any two live writes overlap.
         *
         * Merge the original member schedules with one cursor per live expert.
         * Visiting only neighboring intervals avoids comparing every pair of
         * expert row lists in large prefill bundles with overlapping envelopes.
         */
        static void requireDisjointIndexedBatchedOutputs(
            std::span<const BatchedPrequantizedProjectionDesc> projections);

        /**
         * @brief Preserve the original prepared descriptor projection in one authority.
         * @param source Already validated original batched member.
         * @return Identical fused metadata for execution and explicit cold preflight.
         */
        static FusedVerifierRowsDesc fusedPrequantizedDescriptor(const BatchedPrequantizedProjectionDesc &source);

        struct PreparedMoERow; ///< Immutable admitted operands in one invocation-owned tile plan.
        template<bool Nibble>
        static void preparedMoERowTile(const PreparedMoERow &row, int chunk);
        template<bool Nibble>
        static void preparedMoERowPartitionRange(const PreparedMoERow &row, int chunk,
            int p_start, int p_end, float *dst, bool fold_into_dst);
        static bool prepareMoERows(std::span<const BatchedPrequantizedProjectionDesc> original,
            std::span<PreparedMoERow> prepared, int width, int blocks, int workers, ISALevel isa);
        template<bool Join>
        static void executeMoERows(std::span<const PreparedMoERow> rows, int width);
        template<int KSplit>
        static bool runPreparedFixedMoETilesSplitK(std::span<const PreparedMoERow> gate_plans,
            std::span<const PreparedMoERow> down_plans, std::span<Q8_1Block> activation,
            int hidden, int intermediate, int gate_partitions, int workers, ISALevel isa);
        static std::optional<bool> runPreparedFixedMoETiles(std::span<const BatchedPrequantizedProjectionDesc> gate,
            std::span<const BatchedPrequantizedProjectionDesc> down, std::span<Q8_1Block> activation,
            int hidden, int intermediate, int workers);

        /**
         * @brief Reject an unsupported explicit phase before any FFN payload write.
         * @param descriptors Original already validated source members.
         * @param count Live descriptor prefix.
         * @param ownership Explicit control or Auto; Auto/Current perform no diagnostic preplanning.
         * @throws std::invalid_argument Unless the original planner admits the requested diagnostic.
         *
         * Uses invocation-local fixed storage and the original live-prefix planner.
         * It creates no heap, shared cache, retained plan or additional barrier.
         */
        static void requireBatchedOutputJobOwnership(const BatchedPrequantizedProjectionDesc *descriptors,
            int count, FusedOutputJobOwnership ownership);

        /**
         * @brief Borrow and authenticate query row work before any OpenMP team starts.
         * @param physicalRows Original matrix/scratch row capacity.
         * @return Invocation-local immutable metadata, absent for unchanged operations.
         * @throws std::invalid_argument For inconsistent inherited publication geometry.
         * @details TLS is read on the invoking recorder only. Workers receive
         * explicit metadata pointers; work never enables verifier arithmetic.
         */
        static std::optional<ProjectionRowWork> projectionWorkFor(int physicalRows);
        /**
         * @brief Resolve only the canonical host-owned CPU live prefix.
         * @param physical_rows Immutable matrix capacity used by tensor/workspace admission.
         * @return Valid active rows; absent count means every physical row.
         * @throws std::invalid_argument If the host count is outside its admitted extent.
         * @throws std::logic_error If enclosing row geometry belongs to another matrix.
         *
         * CPU request state is host-owned. This read is not a GPU count readback.
         * Scheduling retains the original serial K tree; no inactive activation,
         * Q8 block (including half scales/sums), bias, or output row is touched.
         */
        static int verifierLiveRows(int physical_rows);
        /** @return The invocation's declared slab; the fused plan checks its used prefix. */
        static std::span<float> fusedPartialWorkspace(DeviceWorkspaceManager *workspace);
        /** @return The nonzero-beta product bank; zero-beta execution borrows no extra bytes. */
        static std::span<float> productWorkspace(DeviceWorkspaceManager *workspace, int rows, int columns, float beta);

        /** @return Shared producer/reducer storage from this invocation's arena. */
        std::span<float> partialWorkspace(DeviceWorkspaceManager *workspace, int rows) const;

        /** @brief Borrow the complete input bank before any quantizer writes to it. */
        static std::span<Q8_1Block> q8Workspace(DeviceWorkspaceManager *workspace, int rows, int columns);

        CPUNativeQuantisedPackedWeights packed_;
        bool valid_ = false;

        /** Arithmetic identity retained across detach/attach lifecycle events. */
        CPUProjectionNumericalPolicy numerical_policy_ =
            CPUProjectionNumericalPolicy::BackendNative;

        /**
         * @brief Optional block-diagonal activation transform paired with weights.
         *
         * The pointer is immutable after construction and is shared safely by
         * concurrent callers. Each participant supplies its own admitted scratch.
         */
        const ActivationRotation *activation_rotation_ = nullptr;

        /**
         * @brief Copy and rotate each input row exactly once into borrowed storage.
         * @param input Immutable source; never modified by this projection.
         * @param m Simultaneously live rows.
         * @param k Logical row width paired with the prepared rotation.
         * @param workspace Participant-owned storage declared by this engine.
         * @param work Optional original-row admission borrowed before the workshare.
         * @return Original input for an unrotated engine, otherwise the rotated bank.
         *
         * Copy and rotation share one row owner. Existing OpenMP teams enter
         * the same workshare and its completion barrier publishes the full bank
         * before quantization. A standalone single row needs no extra team.
         */
        const float *maybe_rotate_activation(const float *input, int m, int k, DeviceWorkspaceManager *workspace,
            const ProjectionRowWork *work = nullptr) const;

        /**
         * @brief Record one coarse verifier-kernel timing sample.
         *
         * These timers intentionally sit at the projection bundle boundary, not
         * inside the NativeQuantised block kernels.  That keeps ordinary inference
         * cheap while still telling us whether Phase 9.8 verifier time is spent
         * in activation preparation, grouped projection GEMV, or SwiGLU.
         */
        static void recordVerifierTiming(
            const char *name,
            PerfStatsCollector::Clock::time_point start,
            int m,
            int n,
            int k,
            int projections);
    };

} // namespace llaminar2::cpu::native_quantised
