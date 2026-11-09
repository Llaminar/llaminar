/**
 * @file CPUNativeQuantisedGemmKernel.h
 * @brief ITensorGemm implementation for CPU NativeQuantised GEMM/GEMV.
 *
 * Immutable prepared weights use the canonical NativeQuantised format registry and
 * AVX2/AVX512 runtime dispatch. Batched descriptor objects occupy only the live
 * prefix of invocation-owned fixed stack storage, with no heap allocation.
 * Serial decode and grouped verification retain
 * their explicit numerical policy and reduction order. A shared prepared
 * engine never retains another participant's workspace binding: the owning
 * stage supplies admitted activation scratch with each invocation.
 * Query projection work is borrowed before OpenMP and retains original request
 * coordinates through rotation, quantization and fixed-order products.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 * Imported optional source-order archive payloads are retired at engine
 * publication; scalar and SIMD execution share the canonical interleaved bank.
 * Packed ownership handoffs retain physical-allocation leases; explicit clone
 * destinations allocate new bytes through their own admitted storage authority.
 * Explicit output-job ownership is invocation-local and independent of row
 * arithmetic; both stable-FFN phases are authenticated before payload mutation.
 * Empty retained-team work keeps its completion edge before a caller can publish
 * new request/frontier data, just as a nonempty workshare does.
 */

#pragma once

#include "../../../execution/moe/DeviceMoERuntimeABI.h"
#include "CPUQ8ResidualCorrectionWorkspace.h"
#include "CPUProjectionWorkspaceContract.h"

#include "CPUNativeQuantisedWeightPacker.h"
#include "CPUNativeQuantisedGemv.h"
#include "CPUQ8ResidualCorrection.h"
#include "CPUPackedWeights.h"
#include "tensors/TensorKernels.h"
#include "tensors/TensorClasses.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "kernels/cpu/rotation/ActivationRotation.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>

namespace llaminar2::cpu::native_quantised
{

    /**
     * @brief RAII publication of serial TP output geometry for CPU NativeQuantised.
     *
     * The scope changes dispatch geometry only on the caller thread. The full
     * replicated weight, complete output buffer, and physical N traversal stay
     * unchanged. Nested graph/stage scopes restore their predecessor exactly,
     * which keeps concurrent rank workers independent without locks.
     */
    class CPUNativeQuantisedOutputPartitionEquivalenceScope final
        : public ITensorGemm::OutputPartitionEquivalenceScope
    {
    public:
        /** Publish @p serial_partition_n until this object is destroyed. */
        explicit CPUNativeQuantisedOutputPartitionEquivalenceScope(
            int serial_partition_n)
            : previous_(cpuNativeQuantisedSerialOutputPartitionN())
        {
            setCPUNativeQuantisedSerialOutputPartitionN(serial_partition_n);
        }

        /** Restore the enclosing output-partition policy. */
        ~CPUNativeQuantisedOutputPartitionEquivalenceScope() override
        {
            setCPUNativeQuantisedSerialOutputPartitionN(previous_);
        }

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
        explicit CPUNativeQuantisedOriginalOutputSliceScope(const OriginalOutputSlice &slice)
            : previous_(detail::originalOutputSlicePolicy()),
              previous_serial_(cpuNativeQuantisedSerialOutputPartitionN())
        {
            detail::originalOutputSlicePolicy() = {slice.physicalColumns(), slice.policyColumns()};
            setCPUNativeQuantisedSerialOutputPartitionN(slice.policyColumns());
        }
        /** @brief Restore even after a failed launch or nested legacy scope. */
        ~CPUNativeQuantisedOriginalOutputSliceScope() override
        {
            setCPUNativeQuantisedSerialOutputPartitionN(previous_serial_);
            detail::originalOutputSlicePolicy() = previous_;
        }
        CPUNativeQuantisedOriginalOutputSliceScope(const CPUNativeQuantisedOriginalOutputSliceScope &) = delete;
        CPUNativeQuantisedOriginalOutputSliceScope &operator=(const CPUNativeQuantisedOriginalOutputSliceScope &) = delete;
    private:
        detail::OriginalOutputSlicePolicy previous_; ///< Immutable predecessor geometry.
        int previous_serial_; ///< Existing two-int predecessor policy.
    };


#if defined(__aarch64__) && defined(__ARM_NEON)
    /**
     * @brief Publish one semantic launch record per distinct executed native schedule.
     *
     * Installed as the native launcher's observer. The record family and its
     * 21 tags are shared with the x86 launcher, so
     * PerfStats consumers read one schema. Identity is geometry and policy,
     * never the member's ordinal in a transient bundle: identical members
     * share one key and contribute their multiplicity as observations.
     * Shared-row I8MM members report isa NEON_I8MM and a grouped_neon_i8mm_*
     * route, proving row-pair SMMLA ran; their lone rows report the SDOT leaf.
     * @param descriptors Complete executed members.
     * @param count Positive member count.
     * @param plan The native schedule the launcher executed.
     * @param tasks Producer tasks per member, from the bound batches.
     * @param dot Native dot instruction used for the bundle.
     */
    inline void recordNativeNEONFusedLaunches(const FusedVerifierRowsDesc *descriptors, int count,
        const NativeQuantisedFusedRowsPlan &plan, const std::array<std::int64_t, NativeQuantisedFusedRowsPlan::capacity> &tasks,
        multi_scale::NEONByteDot dot)
    {
        if (!PerfStatsCollector::isDomainEnabled("kernel") || (omp_in_parallel() && omp_get_thread_num() != 0))
            return;
        std::vector<std::pair<PerfStatsCollector::Tags, std::uint64_t>> groups;
        groups.reserve(std::size_t(count));
        for (int p = 0; p < count; ++p)
        {
            const auto &d = descriptors[p];
            const int rows = plan.rows[p];
            const int tile = rows == 1 ? 1 : (plan.tiles[p] == NEONProjectionRowTile::Pair ? 2 : 4);
            // A lone I8MM row executes the SDOT leaf; shared-row tiles run SMMLA pairs.
            const bool row_pairs = dot == multi_scale::NEONByteDot::I8MM && rows > 1;
            const char *isa = row_pairs ? "NEON_I8MM"
                : dot == multi_scale::NEONByteDot::ASIMD ? "NEON" : "NEON_DOTPROD";
            const char *effective_verifier = rows == 1 ? verifierRowsPolicyName(d.verifier_schedule)
                : (tile == 2 ? verifierRowsPolicyName(VerifierRowsPolicy::Pairwise)
                             : verifierRowsPolicyName(VerifierRowsPolicy::WideRows));
            PerfStatsCollector::Tags tags{
                {"m", std::to_string(rows)}, {"n", std::to_string(d.N)},
                {"policy_n", std::to_string(d.policy_n ? d.policy_n : cpuNativeQuantisedSerialEquivalentPolicyN(d.N))},
                {"k", std::to_string(d.packed->K)}, {"codebook", std::to_string(d.packed->codebook_id)},
                {"purpose", projectionRowsPurposeName(d.purpose)},
                {"isa", isa},
                // ARM register-K tiles traverse full K: one K tile per output owner.
                {"k_tiles", "1"}, {"k_tile_blocks", std::to_string(d.packed->blocks_per_row)},
                {"ambient_n_block_chunks", std::to_string(plan.chunks[p])},
                {"n_block_chunks", std::to_string(plan.chunks[p])},
                {"requested_decode_policy", decodeSchedulePolicyName(d.decode_schedule)},
                {"effective_decode_policy", "Nbc" + std::to_string(plan.chunks[p])},
                {"requested_verifier_policy", verifierRowsPolicyName(d.verifier_schedule)},
                {"effective_verifier_policy", effective_verifier},
                {"physical_row_tile", std::to_string(tile)},
                {"route", rows == 1 ? "decode_neon_full_k_row"
                    : row_pairs ? (tile == 2 ? "grouped_neon_i8mm_pair_rows" : "grouped_neon_i8mm_four_rows")
                    : (tile == 2 ? "grouped_neon_full_k_pair_rows" : "grouped_neon_full_k_four_rows")},
                {"task_grid", "row_tile_n_block"},
                {"producer_tasks", std::to_string(tasks[p])},
                {"reduction_tasks", "0"},
                {"bundle_scheduler", "layer_global_full_k"}};
            const auto found = std::find_if(groups.begin(), groups.end(),
                [&](const auto &group) { return group.first == tags; });
            if (found == groups.end()) groups.emplace_back(std::move(tags), 1u);
            else ++found->second;
        }
        for (const auto &[tags, observations] : groups)
            PerfStatsCollector::addCounter("kernel", "cpu_native_quantised_fused_verifier_rows_projection_launch",
                PerfStatsCollector::CounterObservationBatch{
                    .observations = observations, .value = static_cast<double>(observations)},
                "gemm", "cpu", tags);
    }

    /** Installs the PerfStats publisher in every core program that can issue a native launch. */
    inline const bool kNativeNEONFusedLaunchEvidenceInstalled =
        (nativeNEONFusedLaunchObserver().store(&recordNativeNEONFusedLaunches, std::memory_order_release), true);
#endif

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
                                         CPUWeightStoragePlacement placement = CPUWeightStoragePlacement::local())
            : numerical_policy_(numerical_policy)
        {
            // Pick up activation rotation from the weight tensor (if set).
            // When present, activations will be rotated before Q8_1 quantization
            // to reduce kurtosis and improve int8 fidelity.
            // The rotation is also fused into weight packing (dequant→rotate→requant)
            // so that the original tensor format is preserved.
            activation_rotation_ = weights->activationRotation();

            if (!packWeightsCPUNativeQuantised(weights, packed_, row_start, row_end,
                                          activation_rotation_, placement))
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] Failed to pack weights");
                valid_ = false;
                return;
            }
            packed_.numerical_policy = numerical_policy_;
            valid_ = true;

            LOG_TRACE("[CPUNativeQuantisedGemmKernel] Packed "
                      << packed_.N << "×" << packed_.K
                      << " weights (codebook=" << (int)packed_.codebook_id
                      << ", payload=" << packed_.payload_bytes << " B/block"
                      << ", asymmetric=" << packed_.is_asymmetric
                      << ", rotation=" << (activation_rotation_ != nullptr)
                      << ", permanent_interleaved=true)");
        }

        /**
         * @brief Construct from pre-packed weights (move).
         */
        explicit CPUNativeQuantisedGemmKernel(
            CPUNativeQuantisedPackedWeights &&packed,
            CPUProjectionNumericalPolicy numerical_policy =
                CPUProjectionNumericalPolicy::BackendNative)
            : packed_(std::move(packed)),
              valid_(packed_.hasInterleavedData()),
              numerical_policy_(numerical_policy)
        {
            packed_.numerical_policy = numerical_policy_;
            packed_.releaseSourcePayload();
            if (!valid_)
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] Pre-packed CPU_NATIVE_QUANTISED weights are missing eager interleaved data");
        }

        ~CPUNativeQuantisedGemmKernel() override = default;

        /**
         * @brief Declare the input bank without retaining a participant binding.
         * @param m Maximum simultaneous input rows for this projection.
         * @param k Explicit K geometry, or zero to use the prepared matrix width.
         * @return Q8 and ordered-partial banks, plus the optional rotated input tile.
         */
        WorkspaceRequirements getWorkspaceRequirements(int m, int = 0, int k = 0) const override
        {
            const int columns = k > 0 ? k : packed_.K;
            auto requirements = CPUProjectionWorkspaceContract::prepared(packed_.preparedFootprint(),
                packed_.codebook_id, {.rows = m, .n = packed_.N, .k = packed_.K,
                    .serial_n = cpuNativeQuantisedSerialEquivalentPolicyN(packed_.N),
                    .workers = nativeQuantisedInvocationWorkerCount(),
                    .execution = CPUExecutionGeometry::local(), .numerical_policy = numerical_policy_});
            // An explicit wider input bank may be shared with another serial
            // projection; it cannot shrink this engine's physical requirement.
            if (columns != packed_.K) requirements.merge(cpuProjectionQ8WorkspaceRequirements(m, columns));
            if (activation_rotation_)
                requirements.merge(cpuProjectionFP32WorkspaceRequirements(kCPUProjectionRotation, m, columns));
            return requirements;
        }

        /** @brief Preserve the beta addend only when the owning stage selects accumulation. */
        void appendOutputAccumulationWorkspaceRequirements(
            WorkspaceRequirements &requirements, int m, int n) const override
        {
            requirements.merge(cpuProjectionFP32WorkspaceRequirements(
                kCPUProjectionProduct, m, n > 0 ? n : packed_.N));
        }

        /** @copydoc ITensorGemm::q8ResidualCorrectionWorkspaceRequirements */
        WorkspaceRequirements q8ResidualCorrectionWorkspaceRequirements(int m,int n,int k) const override {
            cpu::native_quantised::validateCPUQ8ResidualSource(packed_,m,n,k);
            if(activation_rotation_)throw std::invalid_argument("Two-term quantized activation cannot silently change a rotated source contract");
            return cpuQ8ResidualCorrectionWorkspaceRequirements(m,n,k);
        }
        /** @copydoc ITensorGemm::multiply_q8_residual_corrected */
        void multiply_q8_residual_corrected(const TensorBase*input,TensorBase*output,DeviceRowRange rows,
            int n,int k,DeviceWorkspaceManager*workspace) override {
            cpu::native_quantised::validateCPUQ8ResidualSource(packed_,rows.physicalRows(),n,k);
            if(activation_rotation_)throw std::invalid_argument("Two-term quantized activation requires its unrotated prepared source");
            if(!input||!output||input->native_type()!=TensorType::FP32||output->native_type()!=TensorType::FP32||
                !input->isHostValid()||!output->isHostValid())
                throw std::invalid_argument("Two-term CPU Q8 requires prepared host-owned FP32 tensors");
            validateQ8ResidualTensorCapacities(rows.physicalRows(),n,k,input->numel(),output->numel());
            auto primary=CPUInvocationWorkspace::require<Q8_1Block>(workspace,kCPUProjectionQ8,cpuProjectionQ8Blocks(rows.physicalRows(),k));
            auto residual=CPUInvocationWorkspace::require<Q8_1Block>(workspace,kCPUProjectionResidualQ8,cpuProjectionQ8Blocks(rows.physicalRows(),k));
            cpuQ8ResidualCorrectedProjection(packed_,input->data(),output->mutable_data(),rows,primary,residual);
        }

        /** @copydoc ITensorGemm::multiply_q8_residual_corrected */
        void multiply_q8_residual_corrected(HCResidualDownInput hc,TensorBase*output,DeviceRowRange rows,
            int n,int k,DeviceWorkspaceManager*workspace) override {
            const auto*input=&hc.raw;auto*activated=&hc.activated;
            validateHCResidualDownRows(rows,rows.physicalRows());
            cpu::native_quantised::validateCPUQ8ResidualSource(packed_,rows.physicalRows(),n,k);
            if(activation_rotation_)throw std::invalid_argument("Two-term quantized activation requires its unrotated prepared source");
            if(!input||!output||input->native_type()!=TensorType::FP32||output->native_type()!=TensorType::FP32||
                !input->isHostValid()||!output->isHostValid())
                throw std::invalid_argument("Two-term CPU Q8 requires prepared host-owned FP32 tensors");
            validateQ8ResidualTensorCapacities(rows.physicalRows(),n,k,input->numel(),output->numel());
            auto primary=CPUInvocationWorkspace::require<Q8_1Block>(workspace,kCPUProjectionQ8,cpuProjectionQ8Blocks(rows.physicalRows(),k));
            auto residual=CPUInvocationWorkspace::require<Q8_1Block>(workspace,kCPUProjectionResidualQ8,cpuProjectionQ8Blocks(rows.physicalRows(),k));
            if(activated==output||activated->native_type()!=TensorType::FP32||!activated->isHostValid()||
               activated->numel()<static_cast<std::size_t>(rows.physicalRows())*k)
                throw std::invalid_argument("HC CPU requires its prepared FP32 activation output");
            cpuHCResidualDownQ8Projection(packed_,input->data(),activated->mutable_data(),output->mutable_data(),rows,primary,residual);
        }

        // -------------------------------------------------------------------
        // ITensorKernel interface
        // -------------------------------------------------------------------

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        /** @brief Export exact CPU-packed source codebook and superblock identity. */
        bool exportNativeQuantisedSourceIdentity(
            NativeQuantisedSourceIdentity &out) const override
        {
            if (!valid_)
            {
                out = {};
                return false;
            }
            out = {
                .codebook_id = packed_.codebook_id,
                .is_superblock = packed_.is_superblock,
                .present = true,
            };
            return native_quantised_formats::forSourceIdentity(
                       out.codebook_id, out.is_superblock) != nullptr;
        }

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
            int serial_partition_columns) override
        {
            if (actual_output_columns <= 0 ||
                serial_partition_columns <= 0 ||
                actual_output_columns != packed_.N ||
                serial_partition_columns > actual_output_columns)
            {
                throw std::invalid_argument(
                    "[CPUNativeQuantisedGemmKernel] Invalid replicated-output serial partition contract");
            }
            return std::make_unique<
                CPUNativeQuantisedOutputPartitionEquivalenceScope>(
                serial_partition_columns);
        }

        /** @brief Bind store-authenticated original N without changing packed bytes or physical extents. */
        std::unique_ptr<OutputPartitionEquivalenceScope>
        beginOriginalOutputSliceScope(const OriginalOutputSlice &slice) override
        {
            slice.require(this, packed_.N, packed_.K);
            if (slice.device() != DeviceId::cpu())
                throw std::invalid_argument("Original output slice belongs to a foreign CPU endpoint");
            return std::make_unique<CPUNativeQuantisedOriginalOutputSliceScope>(slice);
        }

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
            int activation_row_offset = 0) override
        {
            (void)transpose_B;
            (void)mpi_ctx;
            (void)workspace;

            if (!valid_ || device_idx != -1)
                return false;

            if (n != packed_.N || k != packed_.K)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] Dimension mismatch: "
                          << "requested n=" << n << " k=" << k
                          << " packed N=" << packed_.N << " K=" << packed_.K);
                return false;
            }

            const auto projection_work=projectionWorkFor(m);
            const auto *work=projection_work?&*projection_work:nullptr;
            if(work&&m!=1)throw std::invalid_argument("CPU ordinary query work requires explicit M1 arithmetic");
            if(work&&!CPUProjectionRowGrid(m,work).beginWorkshare())return true;
            const float *A_data = A->data() + static_cast<size_t>(activation_row_offset) * k;
            float *C_data = C->mutable_data();

            // Apply activation rotation for kurtosis reduction (if configured)
            const auto activation_storage = q8Workspace(workspace, m, k);
            const auto product_storage = productWorkspace(workspace, m, n, beta);
            const auto partial_storage = partialWorkspace(workspace, m);
            A_data = maybe_rotate_activation(A_data, m, k, workspace,work);

            multiply_native_quantised_with_epilogue(
                packed_, A_data, C_data, m, n, alpha, beta,
                activation_storage, product_storage, partial_storage,work);

            // Apply bias epilogue: C[m, j] += bias[j]
            if (bias)
            {
                const float *bias_data = bias->data();
                apply_bias_epilogue(C_data, bias_data, m, n, n,work);
            }

            return true;
        }

        /** @brief Borrow the engine-owned final CPU migration representation. */
        const CPUNativeQuantisedPackedWeights *
        exportCPUNativeQuantisedPackedWeights() const override
        {
            return valid_ ? &packed_ : nullptr;
        }

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
        exportRetiredCPUNativeQuantisedStorage() noexcept override
        {
            if (!valid_ || packed_.native_interleaved.empty())
                return {};
            return packed_.native_interleaved.retainedSlotRoot();
        }

        /**
         * @brief Open a new publication generation for an acquired overlay slot.
         * @return The same final byte range; it stays unadmitted until resealed.
         */
        std::span<std::uint8_t> beginRetiredCPUNativeQuantisedRewrite() noexcept override
        {
            if (!valid_ || packed_.native_interleaved.empty())
                return {};
            return packed_.native_interleaved.beginRewrite();
        }

        /**
         * @brief Admit completely written slot bytes before their engine becomes reachable.
         * @throws std::invalid_argument For reserved metadata bits or non-canonical storage; the engine stays unadmitted.
         * @throws std::overflow_error For unrepresentable geometry.
         * @throws std::logic_error For a retired engine or owner fields that disagree with the canonical geometry.
         * @details A current seal is left untouched, so republishing an unchanged
         * engine never writes state that concurrent readers observe.
         */
        void admitCPUNativeQuantisedPublication() override
        {
            if (!valid_)
                throw std::logic_error("Retired CPU NativeQuantised engine cannot publish weights");
            if (!packed_.publicationSealed())
            {
                // The arrival wrote live columns only; the publisher defines the
                // final chunk's padding in the same generation, then seals.
                packed_.normalizePaddingMetadata();
                packed_.sealPublication();
            }
        }

        // -------------------------------------------------------------------
        // Weight Lifecycle (IPackedWeights integration)
        // -------------------------------------------------------------------

        std::unique_ptr<IPackedWeights> detachWeights() override
        {
            if (!valid_)
                return nullptr;

            auto result = std::make_unique<CPUPackedWeights>(std::move(packed_));

            // Invalidate kernel
            packed_ = CPUNativeQuantisedPackedWeights{};
            valid_ = false;
            return result;
        }

        std::unique_ptr<IPackedWeights> cloneWeights() const override
        {
            if (!valid_)
                return nullptr;

            return std::make_unique<CPUPackedWeights>(packed_);
        }

        /**
         * @brief Clone published bytes through the explicit destination authority.
         * @param destination Admitted final CPU placement for the new owner.
         * @return Independent packed weights, or empty when this engine is retired.
         */
        std::unique_ptr<IPackedWeights> cloneWeightsTo(CPUWeightStoragePlacement destination) const override
        {
            if (!valid_)
                return nullptr;
            // Read-only wrapper is unnecessary: destination copying operates
            // on a metadata view and never transfers this engine's storage.
            return CPUPackedWeights::copyTo(packed_, destination);
        }

        bool attachWeights(std::unique_ptr<IPackedWeights> weights) override
        {
            if (!weights || weights->format() != PackedWeightsFormat::CPU_NATIVE_QUANTISED)
                return false;

            auto *cpu_packed = dynamic_cast<CPUPackedWeights *>(weights.get());
            if (!cpu_packed)
                return false;

            // Portable cross-backend records may carry an additional native
            // block section.  CPU attachment consumes only the eager
            // interleaved representation and lets the wrapper release the
            // unneeded cross-backend section after this move.
            packed_ = cpu_packed->takePacked();
            packed_.numerical_policy = numerical_policy_;
            packed_.releaseSourcePayload();
            if (!packed_.hasInterleavedData())
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] Attached CPU_NATIVE_QUANTISED weights have no eager interleaved data");
                packed_ = CPUNativeQuantisedPackedWeights{};
                return false;
            }
            // Attachment publishes received or detached bytes to execution.
            try
            {
                if (!packed_.publicationSealed())
                    packed_.sealPublication();
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] Attached CPU_NATIVE_QUANTISED weights failed metadata admission: " << error.what());
                packed_ = CPUNativeQuantisedPackedWeights{};
                return false;
            }

            valid_ = true;
            return true;
        }

        void releaseWeights() override
        {
            {
                CPUNativeQuantisedPackedWeights empty;
                packed_ = std::move(empty);
            }
            valid_ = false;
        }

        bool hasWeights() const override { return valid_; }

        size_t packedWeightBytes() const override
        {
            if (!valid_)
                return 0;
            return packed_.native_interleaved.size() +
                   packed_.payload.size() +
                   packed_.int8_flat.size();
        }

        bool canReleaseSourceWeightTensor() const override
        {
            return valid_ && !packed_.native_interleaved.empty();
        }

        // -------------------------------------------------------------------
        // Accessors
        // -------------------------------------------------------------------

        bool isValid() const { return valid_; }

        const CPUNativeQuantisedPackedWeights &packedWeights() const { return packed_; }

        uint8_t codebookId() const { return packed_.codebook_id; }

        int get_n() const override { return packed_.N; }
        int get_k() const override { return packed_.K; }

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
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)workspace;
            if (!valid_)
                return false;
            if (!gate || !up || !output)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] fused SwiGLU received null tensor");
                return false;
            }
            if (m <= 0 || n <= 0 || k <= 0)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] fused SwiGLU invalid dimensions m="
                          << m << " n=" << n << " k=" << k);
                return false;
            }
            if (packed_.N != n || packed_.K != k)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] fused SwiGLU dimension mismatch: packed N="
                          << packed_.N << " K=" << packed_.K << ", call n=" << n
                          << " k=" << k);
                return false;
            }
            const size_t input_size = static_cast<size_t>(m) * static_cast<size_t>(k);
            const size_t output_size = static_cast<size_t>(m) * static_cast<size_t>(n);
            if (gate->numel() < input_size || up->numel() < input_size || output->numel() < output_size)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] fused SwiGLU tensor capacity mismatch: gate="
                          << gate->numel() << " up=" << up->numel() << " output=" << output->numel()
                          << ", required input=" << input_size << " output=" << output_size);
                return false;
            }

            const float *gate_fp32 = gate->data();
            const float *up_fp32 = up->data();
            float *output_fp32 = output->mutable_data();

            // Apply SwiGLU to get the GEMM input: temp = silu(gate) * up  [m, k]

            // Participants can share weights, but never this live intermediate.
            // The arena owns the admitted tile before inference starts.
            auto swiglu_scratch = CPUInvocationWorkspace::require<float>(
                workspace, kCPUSwiGLUInput, input_size);

            // M=1 decode: use serial SwiGLU to avoid OMP fork/join overhead.
            // For MoE experts with intermediate=512, the 512-element SwiGLU
            // takes ~0.1µs in SIMD vs ~6µs OMP barrier cost.
            if (numerical_policy_ ==
                CPUProjectionNumericalPolicy::GPUAlignedExpert)
            {
                if (m == 1)
                {
                    primitives::compute_swiglu_gpu_aligned_expert_serial(
                        gate_fp32,
                        up_fp32,
                        swiglu_scratch.data(),
                        static_cast<int>(input_size));
                }
                else
                {
                    primitives::compute_swiglu_gpu_aligned_expert(
                        gate_fp32,
                        up_fp32,
                        swiglu_scratch.data(),
                        static_cast<int>(input_size));
                }
            }
            else if (m == 1)
            {
                primitives::compute_swiglu_serial(
                    gate_fp32,
                    up_fp32,
                    swiglu_scratch.data(),
                    static_cast<int>(input_size));
            }
            else
            {
                primitives::compute_swiglu(
                    gate_fp32,
                    up_fp32,
                    swiglu_scratch.data(),
                    static_cast<int>(input_size));
            }

            // Apply activation rotation for kurtosis reduction (if configured)
            const float *gemm_input = maybe_rotate_activation(swiglu_scratch.data(), m, k, workspace);

            multiply_native_quantised_with_epilogue(
                packed_, gemm_input, output_fp32, m, n, alpha, beta,
                q8Workspace(workspace, m, k), productWorkspace(workspace, m, n, beta),
                partialWorkspace(workspace, m));
            return true;
        }

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
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)workspace;
            if (!valid_ || !gate || !up || !output || m < 1 || n <= 0 || k <= 0)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier SwiGLU rejected: valid="
                          << valid_ << " gate=" << (gate != nullptr)
                          << " up=" << (up != nullptr)
                          << " output=" << (output != nullptr)
                          << " m=" << m << " n=" << n << " k=" << k);
                return false;
            }
            if (alpha != 1.0f || beta != 0.0f)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier SwiGLU only supports alpha=1,beta=0; got alpha="
                          << alpha << " beta=" << beta);
                return false;
            }
            if (packed_.N != n || packed_.K != k)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier SwiGLU dimension mismatch: packed N="
                          << packed_.N << " K=" << packed_.K << ", call n=" << n << " k=" << k);
                return false;
            }

            const size_t input_size = static_cast<size_t>(m) * static_cast<size_t>(k);
            const size_t output_size = static_cast<size_t>(m) * static_cast<size_t>(n);
            if (gate->numel() < input_size || up->numel() < input_size || output->numel() < output_size)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier SwiGLU tensor capacity mismatch");
                return false;
            }

            const int live_rows = verifierLiveRows(m);
            // The arena remains admitted for physical m, including empty calls.
            auto swiglu_scratch = CPUInvocationWorkspace::require<float>(
                workspace, kCPUSwiGLUInput, input_size);
            auto shared_q8 = q8Workspace(workspace, m, k);
            (void)partialWorkspace(workspace, m);
            if (activation_rotation_)
                (void)CPUInvocationWorkspace::require<float>(workspace, kCPUProjectionRotation, input_size);
            const auto live_input_size = static_cast<size_t>(live_rows) * k;
            const float *gate_fp32 = gate->data();
            const float *up_fp32 = up->data();
            float *output_fp32 = output->mutable_data();
            if (!gate_fp32 || !up_fp32 || !output_fp32)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier SwiGLU requires FP32 host tensors");
                return false;
            }

            if (live_rows == 0) return true;
            const bool perf_enabled =
                PerfStatsCollector::isDomainEnabled("kernel");
            auto perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                           : PerfStatsCollector::Clock::time_point{};
            if (numerical_policy_ ==
                CPUProjectionNumericalPolicy::GPUAlignedExpert)
            {
                primitives::compute_swiglu_gpu_aligned_expert(
                    gate_fp32,
                    up_fp32,
                    swiglu_scratch.data(),
                    static_cast<int>(live_input_size));
            }
            else
            {
                primitives::compute_swiglu(
                    gate_fp32,
                    up_fp32,
                    swiglu_scratch.data(),
                    static_cast<int>(live_input_size));
            }
            recordVerifierTiming(
                "cpu_native_quantised_verifier_swiglu_compute",
                perf_start,
                live_rows,
                n,
                k,
                1);

            const float *gemm_input = maybe_rotate_activation(swiglu_scratch.data(), live_rows, k, workspace);
            const int K_blocks = (k + 31) / 32;
            perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                      : PerfStatsCollector::Clock::time_point{};
            quantize_activations_to_q8_1(
                gemm_input,
                shared_q8.data(),
                live_rows,
                k,
                K_blocks,
                numerical_policy_);
            recordVerifierTiming(
                "cpu_native_quantised_verifier_activation_quantize",
                perf_start,
                live_rows,
                n,
                k,
                1);

            perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                      : PerfStatsCollector::Clock::time_point{};
            // One verifier row belongs to the existing decode schedule, not
            // the M>=2 grouped policy table. Consume the already-published Q8
            // bank once; no tensor API reentry or activation requantization.
            if (live_rows == 1)
                gemv_native_quantised_preq(packed_, shared_q8.data(), output_fp32,
                    partialWorkspace(workspace, m));
            else
                gemm_native_quantised_preq_decode_equivalent_rows(
                    packed_, shared_q8.data(), output_fp32,
                    partialWorkspace(workspace, m), live_rows, n);
            recordVerifierTiming(
                "cpu_native_quantised_verifier_swiglu_down_gemv",
                perf_start,
                live_rows,
                n,
                k,
                1);
            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_quantised_grouped_verifier_swiglu_down_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(live_rows)},
                        {"n", std::to_string(n)},
                        {"k", std::to_string(k)}});
            }
            return true;
        }

        // -------------------------------------------------------------------
        // Fused multi-projection with quantize-once + epilogues
        // -------------------------------------------------------------------

        bool supports_fused_projection() const override
        {
            return true;
        }

        bool multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)mpi_ctx;
            (void)workspace;
            for (const auto &projection : projections)
                projection.requireOriginalOutputSlice(k);

            constexpr size_t kMaxFusedProjections = NativeQuantisedFusedRowsPlan::capacity;
            if (!valid_ || !input || m <= 0 || k <= 0 ||
                k != packed_.K || projections.empty() ||
                projections.size() > kMaxFusedProjections)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] fused projection rejected: valid="
                          << valid_ << " input=" << (input != nullptr)
                          << " m=" << m << " k=" << k
                          << " packed_k=" << packed_.K
                          << " projections=" << projections.size());
                return false;
            }

            const float *input_data = input->data();
            if (!input_data || input->numel() < static_cast<size_t>(m) * k)
                return false;

            std::array<CPUNativeQuantisedGemmKernel *, kMaxFusedProjections>
                vnni_kernels = {};
            std::array<float *, kMaxFusedProjections> outputs = {};
            std::array<const float *, kMaxFusedProjections> biases = {};
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &projection = projections[i];
                auto *vnni = dynamic_cast<CPUNativeQuantisedGemmKernel *>(
                    projection.kernel);
                if (!vnni || !vnni->valid_ || !projection.output ||
                    projection.n <= 0 || vnni->packed_.K != k ||
                    vnni->packed_.N != projection.n ||
                    vnni->activation_rotation_ != activation_rotation_ ||
                    projection.output->numel() <
                        static_cast<size_t>(m) * projection.n ||
                    (projection.bias &&
                     projection.bias->numel() < static_cast<size_t>(projection.n)))
                {
                    LOG_ERROR("[CPUNativeQuantisedGemmKernel] fused projection contract "
                              "mismatch at index " << i);
                    return false;
                }

                vnni_kernels[i] = vnni;
                outputs[i] = projection.output->mutable_data();
                biases[i] = projection.bias ? projection.bias->data() : nullptr;
                if (!outputs[i] || (projection.bias && !biases[i]))
                    return false;
            }

            input_data = maybe_rotate_activation(input_data, m, k, workspace);
            const int K_blocks = (k + Q8_1Block::BLOCK_SIZE - 1) /
                                 Q8_1Block::BLOCK_SIZE;
            auto shared_q8 = q8Workspace(workspace, m, k);
            quantize_activations_to_q8_1(
                input_data,
                shared_q8.data(),
                m,
                k,
                K_blocks,
                numerical_policy_);

            if (m == 1)
            {
                std::array<FusedGemvDesc, kMaxFusedProjections> descriptors = {};
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    descriptors[i] = {
                        .packed = &vnni_kernels[i]->packed_,
                        .output = outputs[i],
                        .bias = biases[i],
                        .N = projections[i].n,
                        .policy_n = projections[i].original_output_slice
                            ? projections[i].original_output_slice->policyColumns() : 0,
                    };
                }
                gemv_native_quantised_fused_preq(
                    shared_q8.data(),
                    descriptors.data(),
                    fusedPartialWorkspace(workspace),
                    static_cast<int>(projections.size()));
                return true;
            }

            for (size_t i = 0; i < projections.size(); ++i)
            {
                auto original_scope = projections[i].beginOriginalOutputSliceScope();
                gemm_native_quantised_preq(
                    vnni_kernels[i]->packed_,
                    shared_q8.data(),
                    outputs[i],
                    vnni_kernels[i]->partialWorkspace(workspace, m),
                    m,
                    projections[i].n);
                if (biases[i])
                {
                    apply_bias_epilogue(
                        outputs[i],
                        biases[i],
                        m,
                        projections[i].n,
                        projections[i].n);
                }
            }
            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_quantised_small_m_fused_projection_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(m)},
                        {"k", std::to_string(k)},
                        {"projections", std::to_string(projections.size())}});
            }
            return true;
        }

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
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)mpi_ctx;
            (void)workspace;

            for (const auto &projection : projections)
                projection.requireOriginalOutputSlice(k);

            constexpr size_t kMaxGroupedProjections = NativeQuantisedFusedRowsPlan::capacity;
            if (!valid_ || !input || m < 1 || k <= 0 || k != packed_.K ||
                projections.empty() ||
                projections.size() > kMaxGroupedProjections)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier projection rejected: valid="
                          << valid_ << " input=" << (input != nullptr)
                          << " m=" << m << " k=" << k
                          << " projections=" << projections.size());
                return false;
            }

            const auto projection_work=projectionWorkFor(m);
            const auto *work=projection_work?&*projection_work:nullptr;
            // Query work is one prepared projection. Fused bundles retain their
            // existing no-work contract; do not advertise an unwired work mode.
            if(work&&projections.size()!=1)
                throw std::invalid_argument("CPU query row work requires one prepared projection, without fused bundle");
            const float *input_data = input->data();
            if (!input_data || input->numel() < static_cast<size_t>(m) * k)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier projection rejected: input has no complete host FP32 rows");
                return false;
            }

            std::array<CPUNativeQuantisedGemmKernel *, kMaxGroupedProjections>
                vnni_kernels = {};
            std::array<float *, kMaxGroupedProjections> outputs = {};
            std::array<const float *, kMaxGroupedProjections> biases = {};
            // Validate every descriptor before quantization or projection writes,
            // matching the ordinary fused contract. Compute consumes packed.N/K,
            // so the advertised dimensions cannot substitute smaller capacities.
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &proj = projections[i];
                auto *vnni = dynamic_cast<CPUNativeQuantisedGemmKernel *>(proj.kernel);
                if (!vnni || !vnni->valid_ || !proj.output || proj.n <= 0 ||
                    vnni->packed_.K != k || vnni->packed_.N != proj.n ||
                    proj.output->numel() < static_cast<size_t>(m) * proj.n ||
                    (proj.bias && proj.bias->numel() < static_cast<size_t>(proj.n)))
                {
                    LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier projection rejected at projection "
                              << i << ": native_quantised=" << (vnni != nullptr)
                              << " valid=" << (vnni ? vnni->valid_ : false)
                              << " output=" << (proj.output != nullptr)
                              << " n=" << proj.n);
                    return false;
                }

                /*
                 * The fused projection API shares one activation transform.
                 * If a future model ever mixes differently rotated weights in
                 * one fused group, that is a real contract gap and must be
                 * handled explicitly rather than silently producing drift.
                 */
                if (vnni->activation_rotation_ != activation_rotation_)
                {
                    LOG_ERROR("[CPUNativeQuantisedGemmKernel] grouped verifier projection rejected at projection "
                              << i << ": mixed activation rotation contracts");
                    return false;
                }
                vnni_kernels[i] = vnni;
            }
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &proj = projections[i];
                outputs[i] = proj.output->mutable_data();
                biases[i] = proj.bias ? proj.bias->data() : nullptr;
                if (!outputs[i] || (proj.bias && !biases[i]))
                    return false;
            }

            std::array<FusedVerifierRowsDesc, kMaxGroupedProjections>
                fused_descs = {};
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &proj = projections[i];
                float *out_data = outputs[i];

                fused_descs[i] = {
                    &vnni_kernels[i]->packed_,
                    out_data,
                    biases[i],
                    proj.n,
                    proj.n};
                fused_descs[i].policy_n = proj.original_output_slice
                    ? proj.original_output_slice->policyColumns() : 0;
            }

            const int live_rows = work ? m : verifierLiveRows(m);
            auto shared_q8 = q8Workspace(workspace, m, k);
            if (projections.size() == 1)
            {
                auto original_scope = projections.front().beginOriginalOutputSliceScope();
                (void)vnni_kernels.front()->partialWorkspace(workspace, m);
            }
            std::span<float> fused_partials;
            if (projections.size() >= 2)
            {
                fused_partials = fusedPartialWorkspace(workspace);
                // The complete physical admission and current live plan share
                // original per-leaf policy_n; neither may overrun this arena.
                const auto descriptors = std::span<const FusedVerifierRowsDesc>(
                    fused_descs.data(), projections.size());
                const auto physical_plan = planNativeQuantisedFusedRows(
                    descriptors, m, nativeQuantisedInvocationWorkerCount(), ISAPath::AUTO);
                auto required = physical_plan.sharedPartialFloats();
                if (live_rows > 0 && live_rows != m)
                {
                    const auto live_plan = planNativeQuantisedFusedRows(
                        descriptors, live_rows, nativeQuantisedInvocationWorkerCount(), ISAPath::AUTO);
                    required = std::max(required, live_plan.sharedPartialFloats());
                }
                if (fused_partials.size() < required)
                    throw std::runtime_error("CPU fused projection requires admitted ordered-partial storage");
            }
            if (activation_rotation_)
                (void)CPUInvocationWorkspace::require<float>(workspace, kCPUProjectionRotation,
                    static_cast<size_t>(m) * k);
            if(live_rows==0) {
                (void)llaminar::v2::beginWorkshare(false);
                return true;
            }
            if(work&&!CPUProjectionRowGrid(m,work).beginWorkshare())return true;

            const bool perf_enabled =
                PerfStatsCollector::isDomainEnabled("kernel");
            auto perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                           : PerfStatsCollector::Clock::time_point{};
            input_data = maybe_rotate_activation(input_data, live_rows, k, workspace,work);

            const int K_blocks = (k + 31) / 32;
            quantize_activations_to_q8_1(
                input_data,
                shared_q8.data(),
                live_rows,
                k,
                K_blocks,
                numerical_policy_,work);
            recordVerifierTiming(
                "cpu_native_quantised_verifier_projection_activation_quantize",
                perf_start,
                live_rows,
                /*n=*/0,
                k,
                static_cast<int>(projections.size()));

            /*
             * Multi-projection verifier path.
             *
             * GDN/QKV verifier graphs commonly have several projections fed by
             * the same runtime-M hidden-state rows. The older implementation ran
             * one grouped-row GEMV per projection, which was correct but paid
             * an OpenMP team entry and scheduling cost for every projection.
             * This fused descriptor path keeps the exact M=1 decode chunk math
             * and schedules every projection under one OpenMP team.
             */
            if (projections.size() >= 2)
            {
                perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                          : PerfStatsCollector::Clock::time_point{};
                if (gemm_native_quantised_fused_verifier_rows_preq(
                        shared_q8.data(),
                        fused_descs.data(),
                        fused_partials,
                        static_cast<int>(projections.size()),
                        live_rows,
                        K_blocks))
                {
                    recordVerifierTiming(
                        "cpu_native_quantised_verifier_projection_fused_gemv",
                        perf_start,
                        live_rows,
                        /*n=*/0,
                        k,
                        static_cast<int>(projections.size()));
                    if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "cpu_native_quantised_fused_grouped_verifier_projection_calls",
                            1.0,
                            "gemm",
                            "cpu",
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(live_rows)},
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(projections.size())}});
                    }
                    return true;
                }

                LOG_ERROR("[CPUNativeQuantisedGemmKernel] fused grouped verifier "
                          "kernel rejected an advertised multi-projection "
                          "NativeQuantised contract");
                recordVerifierTiming(
                    "cpu_native_quantised_verifier_projection_fused_gemv_rejected",
                    perf_start,
                    live_rows,
                    /*n=*/0,
                    k,
                    static_cast<int>(projections.size()));

                return false;
            }

            for (size_t i = 0; i < projections.size(); ++i)
            {
                auto *vnni = vnni_kernels[i];
                const auto &proj = projections[i];
                auto original_scope = proj.beginOriginalOutputSliceScope();
                float *out_data = outputs[i];

                perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                          : PerfStatsCollector::Clock::time_point{};
                // The single-descriptor M1 transaction uses its actual
                // prepared decode scheduler. Multi-row transactions retain
                // the grouped scheduler and its unchanged M>=2 contract.
                if (live_rows == 1)
                    gemv_native_quantised_preq(vnni->packed_, shared_q8.data(), out_data,
                        vnni->partialWorkspace(workspace, m));
                else
                    gemm_native_quantised_preq_decode_equivalent_rows(
                        vnni->packed_, shared_q8.data(), out_data,
                        vnni->partialWorkspace(workspace, m), live_rows, proj.n,
                        ISAPath::AUTO,VerifierRowsPolicy::Auto,true,0,work);
                recordVerifierTiming(
                    "cpu_native_quantised_verifier_projection_per_gemv",
                    perf_start,
                    live_rows,
                    proj.n,
                    k,
                    1);

                if (proj.bias)
                {
                    apply_bias_epilogue(out_data, biases[i], live_rows, proj.n, proj.n,work);
                }
            }

            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_quantised_grouped_verifier_projection_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(live_rows)},
                        {"k", std::to_string(k)},
                        {"projections", std::to_string(projections.size())}});
            }
            return true;
        }

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
        };

        /** Maximum gate/up descriptors for routed experts plus one shared expert. */
        static constexpr int kMaxBatchedPrequantizedProjections =
            NativeQuantisedFusedRowsPlan::capacity;

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
            FusedOutputJobOwnership job_ownership = FusedOutputJobOwnership::Auto)
        {
            if (!descriptors || descriptor_count <= 0 ||
                descriptor_count > kMaxBatchedPrequantizedProjections ||
                k <= 0)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] batched pre-quantized "
                          "projection received invalid geometry: descriptors="
                          << descriptor_count << " k=" << k);
                return false;
            }

            // The maximum stack capacity already belongs to this invocation.
            // Standard non-allocating placement array-new starts exactly the
            // live contiguous objects; no heap allocation or array cookie is
            // permitted by this void* placement overload. Default member
            // initializers still establish every field of every live object.
            // Trivial destruction lets their lifetimes end with this backing.
            static_assert(std::is_trivially_destructible_v<FusedVerifierRowsDesc>);
            alignas(FusedVerifierRowsDesc) std::array<std::byte,
                sizeof(FusedVerifierRowsDesc) * kMaxBatchedPrequantizedProjections>
                fused_descriptor_storage;
            auto *const fused_descriptors = ::new (static_cast<void*>(fused_descriptor_storage.data()))
                FusedVerifierRowsDesc[static_cast<size_t>(descriptor_count)];
            int total_rows = 0;
            int max_rows = 0;
            for (int projection = 0;
                 projection < descriptor_count;
                 ++projection)
            {
                const auto &source = descriptors[projection];
                CPUNativeQuantisedGemmKernel *kernel = source.kernel;
                if (!kernel || !kernel->valid_ || !source.input_q8 ||
                    !source.output || source.rows <= 0 || source.n <= 0 ||
                    source.ldc < source.n || kernel->packed_.K != k ||
                    kernel->packed_.N != source.n ||
                    kernel->activation_rotation_ != nullptr)
                {
                    LOG_ERROR("[CPUNativeQuantisedGemmKernel] batched pre-quantized "
                              "projection contract mismatch at descriptor "
                              << projection);
                    return false;
                }

                // The explicit three-row mechanism has no singleton/pair alias.
                // Validate the entire source set before any output is touched.
                if (source.verifier_schedule == VerifierRowsPolicy::OrderedThreeRows && source.rows < 3)
                {
                    LOG_ERROR("[CPUNativeQuantisedGemmKernel] OrderedThreeRows requires at least three rows at descriptor "
                              << projection);
                    return false;
                }

                fused_descriptors[static_cast<size_t>(projection)] = fusedPrequantizedDescriptor(source);
                total_rows += source.rows;
                max_rows = std::max(max_rows, source.rows);
            }

            const bool team_observer =
                !omp_in_parallel() || omp_get_thread_num() == 0;
            const bool perf_enabled =
                team_observer &&
                PerfStatsCollector::isDomainEnabled("kernel");
            const auto perf_start = perf_enabled
                                        ? PerfStatsCollector::Clock::now()
                                        : PerfStatsCollector::Clock::time_point{};
            const int blocks_per_row =
                (k + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            if (!gemm_native_quantised_fused_verifier_rows_preq(
                    nullptr,
                    fused_descriptors,
                    partial_storage,
                    descriptor_count,
                    /*default_rows unused by explicit descriptors=*/1,
                    blocks_per_row, ISAPath::AUTO, job_ownership))
            {
                return false;
            }

            // Guard before constructing strings/maps: addCounter's own guard
            // cannot prevent allocations while evaluating its arguments.
            if (perf_enabled)
            {
                recordVerifierTiming(
                    "cpu_native_quantised_batched_preq_decode_equivalent",
                    perf_start,
                    total_rows,
                    descriptor_count,
                    k,
                    descriptor_count);
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_quantised_batched_preq_decode_equivalent_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    {{"descriptors", std::to_string(descriptor_count)},
                     {"total_rows", std::to_string(total_rows)},
                     {"max_rows", std::to_string(max_rows)},
                     {"k", std::to_string(k)},
                     {"team_lifetime", "layer_phase"}});
            }
            return true;
        }

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
            FusedOutputJobOwnership down_ownership = FusedOutputJobOwnership::Auto)
        {
            if (!gate_up_descriptors || gate_up_count <= 0 ||
                !down_descriptors || down_count <= 0 ||
                !gate_rows || !up_rows || !activation_q8 ||
                hidden_width <= 0 || route_rows <= 0 || intermediate <= 0 ||
                activation_blocks_per_row <= 0)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] MoE verifier transaction "
                          "received an incomplete descriptor or buffer contract");
                return false;
            }

            auto validate_bundle = [](const BatchedPrequantizedProjectionDesc *descriptors,
                                      int count,
                                      int expected_k,
                                      const char *phase) -> bool
            {
                if (count > kMaxBatchedPrequantizedProjections)
                    return false;
                for (int projection = 0; projection < count; ++projection)
                {
                    const auto &source = descriptors[projection];
                    CPUNativeQuantisedGemmKernel *kernel = source.kernel;
                    if (!kernel || !kernel->valid_ || !source.input_q8 ||
                        !source.output || source.rows <= 0 || source.n <= 0 ||
                        source.ldc < source.n ||
                        kernel->packed_.K != expected_k ||
                        kernel->packed_.N != source.n ||
                        kernel->activation_rotation_ != nullptr)
                    {
                        LOG_ERROR("[CPUNativeQuantisedGemmKernel] MoE grouped FFN "
                                  << phase << " descriptor " << projection
                                  << " violates the eager pre-quantized contract");
                        return false;
                    }
                }
                return true;
            };
            if (!validate_bundle(
                    gate_up_descriptors,
                    gate_up_count,
                    hidden_width,
                    "gate/up") ||
                !validate_bundle(
                    down_descriptors,
                    down_count,
                    intermediate,
                    "down"))
            {
                return false;
            }

            // Both explicit phases must be legal before gate/up can mutate any
            // payload. The same original planner is used again under the actual
            // team; no alternate eligibility model or arithmetic is introduced.
            requireBatchedOutputJobOwnership(gate_up_descriptors, gate_up_count, gate_up_ownership);
            requireBatchedOutputJobOwnership(down_descriptors, down_count, down_ownership);

            std::atomic<bool> transaction_ok{true};
            auto execute_participant = [&]()
            {
                if (!multiply_batched_preq_decode_equivalent(
                        gate_up_descriptors,
                        gate_up_count,
                        hidden_width, partial_storage, gate_up_ownership))
                {
                    transaction_ok.store(false, std::memory_order_relaxed);
                }

                swiglu_quantize_activations_to_q8_1(
                    gate_rows,
                    up_rows,
                    activation_q8,
                    route_rows,
                    intermediate,
                    activation_blocks_per_row,
                    gate_up_descriptors[0].kernel->numerical_policy_);

                if (!multiply_batched_preq_decode_equivalent(
                        down_descriptors,
                        down_count,
                        intermediate, partial_storage, down_ownership))
                {
                    transaction_ok.store(false, std::memory_order_relaxed);
                }
            };
            OMP_WORKSHARE_REGION(execute_participant);

            const bool completed =
                transaction_ok.load(std::memory_order_relaxed);
            if (!completed)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] MoE verifier transaction "
                          "failed inside its stable OpenMP team");
                return false;
            }

            if ((!omp_in_parallel() || omp_get_thread_num() == 0) &&
                PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_quantised_moe_grouped_ffn_transactions",
                    1.0,
                    "gemm",
                    "cpu",
                    {{"gate_up_descriptors", std::to_string(gate_up_count)},
                     {"down_descriptors", std::to_string(down_count)},
                     {"route_rows", std::to_string(route_rows)},
                     {"hidden_width", std::to_string(hidden_width)},
                     {"intermediate", std::to_string(intermediate)},
                     {"team_lifetime", "complete_moe_ffn"}});
            }
            return true;
        }

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
            std::span<float> partial_storage)
        {
            for (const auto &projection : projections)
                projection.requireOriginalOutputSlice(k);

            constexpr size_t kMaxRouterQ8Projections = NativeQuantisedFusedRowsPlan::capacity;
            if (!valid_ || !input_q8 || m < 1 || k <= 0 ||
                projections.empty() ||
                projections.size() > kMaxRouterQ8Projections)
            {
                LOG_ERROR("[CPUNativeQuantisedGemmKernel] router-Q8 projection rejected: valid="
                          << valid_ << " input_q8=" << (input_q8 != nullptr)
                          << " m=" << m << " k=" << k
                          << " projections=" << projections.size());
                return false;
            }

            const int K_blocks = (k + Q8_1Block::BLOCK_SIZE - 1) /
                                 Q8_1Block::BLOCK_SIZE;
            std::array<CPUNativeQuantisedGemmKernel *, kMaxRouterQ8Projections>
                vnni_kernels = {};
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &projection = projections[i];
                auto *vnni = dynamic_cast<CPUNativeQuantisedGemmKernel *>(projection.kernel);
                if (!vnni || !vnni->valid_ || !projection.output ||
                    projection.n <= 0 || vnni->packed_.K != k ||
                    vnni->packed_.N != projection.n)
                {
                    LOG_ERROR("[CPUNativeQuantisedGemmKernel] router-Q8 projection "
                              "contract mismatch at index "
                              << i);
                    return false;
                }
                if (vnni->activation_rotation_ != nullptr)
                {
                    LOG_ERROR("[CPUNativeQuantisedGemmKernel] router-Q8 publication "
                              "cannot feed a rotated activation contract at projection "
                              << i);
                    return false;
                }
                vnni_kernels[i] = vnni;
            }

            const bool perf_enabled =
                PerfStatsCollector::isDomainEnabled("kernel");
            const auto perf_start = perf_enabled
                                        ? PerfStatsCollector::Clock::now()
                                        : PerfStatsCollector::Clock::time_point{};

            if (m == 1)
            {
                std::array<FusedGemvDesc, kMaxRouterQ8Projections>
                    descriptors = {};
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &projection = projections[i];
                    auto &descriptor = descriptors[i];
                    descriptor.policy_n = projection.original_output_slice
                        ? projection.original_output_slice->policyColumns() : 0;
                    descriptor.packed = &vnni_kernels[i]->packed_;
                    descriptor.output = projection.output->mutable_data();
                    descriptor.bias = projection.bias ? projection.bias->data() : nullptr;
                    descriptor.N = projection.n;
                    if (!descriptor.output || (projection.bias && !descriptor.bias))
                        return false;
                }
                gemv_native_quantised_fused_preq(
                    input_q8,
                    descriptors.data(),
                    partial_storage,
                    static_cast<int>(projections.size()));
            }
            else
            {
                std::array<FusedVerifierRowsDesc, kMaxRouterQ8Projections>
                    descriptors = {};
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &projection = projections[i];
                    float *output = projection.output->mutable_data();
                    const float *bias = projection.bias ? projection.bias->data() : nullptr;
                    if (!output || (projection.bias && !bias))
                        return false;
                    descriptors[i] = {
                        &vnni_kernels[i]->packed_,
                        output,
                        bias,
                        projection.n,
                        projection.n};
                    descriptors[i].policy_n = projection.original_output_slice
                        ? projection.original_output_slice->policyColumns() : 0;
                }

                if (!gemm_native_quantised_fused_verifier_rows_preq(
                        input_q8,
                        descriptors.data(),
                        partial_storage,
                        static_cast<int>(projections.size()),
                        m,
                        K_blocks))
                {
                    LOG_ERROR("[CPUNativeQuantisedGemmKernel] router-Q8 fused verifier "
                              "kernel rejected an advertised NativeQuantised contract");
                    return false;
                }
            }

            recordVerifierTiming(
                "cpu_native_quantised_router_q8_reused_projection_gemv",
                perf_start,
                m,
                /*n=*/0,
                k,
                static_cast<int>(projections.size()));
            if (perf_enabled)
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_quantised_router_q8_grouped_decode_equivalent_projection_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    {{"m", std::to_string(m)},
                     {"k", std::to_string(k)},
                     {"projections", std::to_string(projections.size())},
                     {"path", m == 1 ? "decode" : "grouped_rows"}});
            }
            return true;
        }

        // =====================================================================
        // Fused multi-input GEMV for MoE expert down projections
        // =====================================================================

        bool multiply_fused_expert_down(
            const FusedExpertDownDesc *descs, int num_descs,
            int m, int k, DeviceWorkspaceManager *workspace = nullptr) override
        {
            if (!descs || m != 1 || num_descs < 1 ||
                num_descs > kMaxBatchedPrequantizedProjections || k <= 0)
                return false;

            // Verify all kernels are CPUNativeQuantisedGemmKernel
            for (int i = 0; i < num_descs; ++i)
            {
                auto *vnni = dynamic_cast<CPUNativeQuantisedGemmKernel *>(descs[i].kernel);
                if (!vnni || !vnni->valid_ || !descs[i].input ||
                    !descs[i].output || descs[i].n <= 0 ||
                    vnni->packed_.K != k || vnni->packed_.N != descs[i].n ||
                    vnni->activation_rotation_ != nullptr)
                    return false;
            }

            const int K_blocks = (k + 31) / 32;

            // Quantize each expert's FP32 input to Q8_1
            // Use a contiguous buffer for all experts' Q8_1 blocks
            auto multi_q8 = q8Workspace(workspace, num_descs, k);

            auto quantize_inputs = [&]() {
                for (int i = 0; i < num_descs; ++i)
                {
                    auto *vnni = static_cast<CPUNativeQuantisedGemmKernel *>(
                        descs[i].kernel);
                    Q8_1Block *A_q8 = multi_q8.data() + static_cast<size_t>(i) * K_blocks;
                    const float *input_data = descs[i].input;
                    int kb = 0;
#if defined(__AVX512F__)
                    for (; kb + 1 < K_blocks; kb += 2)
                        quantizeTwoActivationBlocks(
                            input_data + kb * 32,
                            A_q8[kb],
                            A_q8[kb + 1],
                            vnni->numerical_policy_);
#endif
                    for (; kb < K_blocks; ++kb)
                        quantizeActivationBlock(
                            input_data + kb * 32,
                            A_q8[kb],
                            std::min(32, k - kb * 32),
                            vnni->numerical_policy_);
                }
            };
            // One retained team shares the complete distinct-input bank.
            // Publish it once, then let every worker enter the fused launcher.
            if (omp_in_parallel())
            {
#pragma omp single
                { quantize_inputs(); }
            }
            else
                quantize_inputs();

            std::array<FusedGemvMultiInputDesc,
                       kMaxBatchedPrequantizedProjections>
                mi_descs = {};
            for (int i = 0; i < num_descs; ++i)
            {
                auto *vnni = static_cast<CPUNativeQuantisedGemmKernel *>(descs[i].kernel);
                auto &d = mi_descs[static_cast<size_t>(i)];
                d.A_q8 = multi_q8.data() + static_cast<size_t>(i) * K_blocks;
                d.packed = &vnni->packed_;
                d.output = descs[i].output;
                d.N = descs[i].n;
            }

            // Single OMP region with nowait between expert projections
            gemv_fused_multi_input_preq(
                mi_descs.data(),
                    fusedPartialWorkspace(workspace),
                num_descs);

            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_quantised_fused_expert_down_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(m)},
                        {"k", std::to_string(k)},
                        {"experts", std::to_string(num_descs)}});
            }

            return true;
        }

    private:
        /**
         * @brief Preserve the original prepared descriptor projection in one authority.
         * @param source Already validated original batched member.
         * @return Identical fused metadata for execution and explicit cold preflight.
         */
        static FusedVerifierRowsDesc fusedPrequantizedDescriptor(const BatchedPrequantizedProjectionDesc &source)
        {
            return {
                .packed = &source.kernel->packed_,
                .output = source.output,
                .bias = source.bias,
                .N = source.n,
                .ldc = source.ldc,
                .rows = source.rows,
                .input = source.input_q8,
                .decode_schedule = DecodeSchedulePolicy::Auto,
                .verifier_schedule = source.rows > 1
                    ? source.verifier_schedule
                    : VerifierRowsPolicy::Auto,
                .purpose = source.purpose,
            };
        }

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
            int count, FusedOutputJobOwnership ownership)
        {
#if defined(__aarch64__) && defined(__ARM_NEON)
            // The native ARM schedule owns N directly; only Auto/Current exist.
            (void)descriptors; (void)count;
            requireNativeNEONFusedOutputJobOwnership(ownership);
#else
            (void)fusedOutputJobOwnershipName(ownership);
            if (ownership == FusedOutputJobOwnership::Current || ownership == FusedOutputJobOwnership::Auto) return;
            alignas(FusedVerifierRowsDesc) std::array<std::byte,
                sizeof(FusedVerifierRowsDesc) * kMaxBatchedPrequantizedProjections> descriptor_storage;
            auto *const fused = ::new (static_cast<void *>(descriptor_storage.data()))
                FusedVerifierRowsDesc[static_cast<size_t>(count)];
            for (int member = 0; member < count; ++member)
                fused[member] = fusedPrequantizedDescriptor(descriptors[member]);
            alignas(FusedVerifierRowsPlan) std::array<std::byte,
                sizeof(FusedVerifierRowsPlan) * kMaxBatchedPrequantizedProjections> plan_storage;
            detail::NativeQuantisedFusedRowsLiveView live_plan(plan_storage, static_cast<size_t>(count));
            const auto bundle = detail::fillNativeQuantisedFusedRowsPlans(
                live_plan.mutableProjections(), {fused, static_cast<size_t>(count)},
                1, nativeQuantisedInvocationWorkerCount());
            detail::requireFusedOutputJobOwnership(ownership,
                {fused, static_cast<size_t>(count)}, live_plan.projections(), bundle.partial_route);
#endif
        }

        /**
         * @brief Borrow and authenticate query row work before any OpenMP team starts.
         * @param physicalRows Original matrix/scratch row capacity.
         * @return Invocation-local immutable metadata, absent for unchanged operations.
         * @throws std::invalid_argument For inconsistent inherited publication geometry.
         * @details TLS is read on the invoking recorder only. Workers receive
         * explicit metadata pointers; work never enables verifier arithmetic.
         */
        static std::optional<ProjectionRowWork> projectionWorkFor(int physicalRows)
        {
            const auto *scoped=currentProjectionRowWork(physicalRows);
            if(!scoped)return std::nullopt;
            auto work=*scoped;
            if(const auto *requests=ITensorGemm::VerifierKernelModeScope::requestsFor(physicalRows))
                work=work.withInheritedRequests(*requests);
            else if(const auto *prefix=ITensorGemm::VerifierKernelModeScope::prefixForFloating(physicalRows))
                work=work.withInheritedPrefix(*prefix);
            (void)CPUProjectionRowGrid(physicalRows,&work);
            return work;
        }
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
        static int verifierLiveRows(int physical_rows)
        {
            const auto* rows = ITensorGemm::VerifierKernelModeScope::rowsFor(physical_rows);
            if (!rows) return physical_rows;
            const int count = rows->countOwner() ? *rows->countOwner() : rows->countCapacity();
            const int live = rows->activeRowsFor(count);
            if (live < 0)
                throw std::invalid_argument("CPU verifier live count exceeds its admitted row extent");
            return live;
        }
        /** @return The invocation's declared slab; the fused plan checks its used prefix. */
        static std::span<float> fusedPartialWorkspace(DeviceWorkspaceManager *workspace)
        {
            return CPUInvocationWorkspace::available<float>(workspace, kCPUProjectionPartials);
        }
        /** @return The nonzero-beta product bank; zero-beta execution borrows no extra bytes. */
        static std::span<float> productWorkspace(DeviceWorkspaceManager *workspace, int rows, int columns, float beta)
        {
            if (beta == 0.f) return {};
            return CPUInvocationWorkspace::require<float>(workspace, kCPUProjectionProduct,
                CPUInvocationWorkspace::multiply(static_cast<size_t>(rows), static_cast<size_t>(columns)));
        }

        /** @return Shared producer/reducer storage from this invocation's arena. */
        std::span<float> partialWorkspace(DeviceWorkspaceManager *workspace, int rows) const
        {
            return CPUInvocationWorkspace::require<float>(workspace, kCPUProjectionPartials,
                nativeQuantisedProjectionPartialFloats(packed_, rows));
        }

        /** @brief Borrow the complete input bank before any quantizer writes to it. */
        static std::span<Q8_1Block> q8Workspace(DeviceWorkspaceManager *workspace, int rows, int columns)
        {
            return CPUInvocationWorkspace::require<Q8_1Block>(workspace, kCPUProjectionQ8,
                cpuProjectionQ8Blocks(rows, columns));
        }

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
            const ProjectionRowWork *work = nullptr) const
        {
            if (!activation_rotation_)
                return input;

            const CPUProjectionRowGrid row_grid(m,work);
            if(!row_grid.beginWorkshare())return input;
            const size_t len = static_cast<size_t>(m) * k;
            auto rotated = CPUInvocationWorkspace::require<float>(workspace, kCPUProjectionRotation, len);
            auto rotate_row = [&](int row) {
                if(!row_grid.active(row))return;
                float *destination = rotated.data() + static_cast<size_t>(row) * k;
                std::memcpy(destination, input + static_cast<size_t>(row) * k, static_cast<size_t>(k) * sizeof(float));
                activation_rotation_->rotate_inplace(destination, k);
            };
            if (m == 1 && !omp_in_parallel()) rotate_row(0);
            else
            {
                auto work = [&]() {
#pragma omp for schedule(static)
                    for (int row = 0; row < m; ++row) rotate_row(row);
                };
                OMP_WORKSHARE_REGION(work);
            }
            return rotated.data();
        }

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
            int projections)
        {
            if (!PerfStatsCollector::isDomainEnabled("kernel"))
                return;

            const auto end = PerfStatsCollector::Clock::now();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            PerfStatsCollector::Tags tags{
                {"m", std::to_string(m)},
                {"k", std::to_string(k)},
                {"projections", std::to_string(projections)}};
            if (n > 0)
                tags.emplace("n", std::to_string(n));
            PerfStatsCollector::recordTimingNs(
                "kernel",
                name ? name : "cpu_native_quantised_verifier_unknown",
                ns > 0 ? static_cast<uint64_t>(ns) : 0,
                "gemm",
                "cpu",
                std::move(tags));
        }
    };

} // namespace llaminar2::cpu::native_quantised
