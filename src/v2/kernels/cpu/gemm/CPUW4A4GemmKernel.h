/**
 * @file CPUW4A4GemmKernel.h
 * @brief Prepared CPU GEMM interface for native W4A4/H operands.
 *
 * A kernel retains immutable source storage, not a source tensor or converted
 * floating weights. Each invocation borrows the owning stage's admitted banks.
 * Ordinary, fused and verifier projections use the same grouped integer body;
 * output slicing changes ownership only, never the whole-token activation unit
 * or the per-group shift. CPU row publications retain original coordinates.
 */
#pragma once

#include "CPUW4A4Workspace.h"
#include "tensors/TensorKernels.h"
#include "tensors/W4A4WeightStorage.h"

namespace llaminar2::w4a4
{
    /** @brief Immutable prepared engine with invocation-owned native A4 scratch. */
    class CPUW4A4GemmKernel final : public ITensorGemm, public CPUInvocationWorkspaceConsumer
    {
    public:
        /**
         * @brief Retain admitted native source bytes without repacking or copying.
         * @param weights Complete-K matrix or checked complete-output-row lease.
         */
        explicit CPUW4A4GemmKernel(WeightStorage weights);

        /** @return True only for the CPU execution selector. */
        bool supports_device(int device_idx) const override;
        /** @return Complete output width of the retained native row lease. */
        int get_n() const override;
        /** @return Complete reduction width used for the online token unit. */
        int get_k() const override;
        /** @return Native payload bytes retained by this engine; PMA remains the physical ledger. */
        std::size_t packedWeightBytes() const override;
        /** @return True: source tensor retirement cannot retire this retained storage. */
        bool canReleaseSourceWeightTensor() const override;
        /** @return FP32 graph boundaries; snapshots never invent floating weight equivalents. */
        KernelSnapshotInfo getKernelSnapshotInfo() const override;

        /**
         * @brief Declare the exact native activation and optional transform banks.
         * @param m Maximum original input rows for this invocation family.
         * @param n Exact output width, or zero for the retained width.
         * @param k Exact complete input width, or zero for the retained width.
         * @return Typed requirements for the existing arena/PMA authority.
         * @throws std::invalid_argument For geometry belonging to another matrix.
         */
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        /** @copydoc IWorkspaceConsumer::appendOutputAccumulationWorkspaceRequirements */
        void appendOutputAccumulationWorkspaceRequirements(WorkspaceRequirements &, int m, int n) const override;
        /** @copydoc IWorkspaceConsumer::appendFusedProjectionWorkspaceRequirements */
        void appendFusedProjectionWorkspaceRequirements(WorkspaceRequirements &, int m,
            std::span<const int> projection_columns, int k) const override;

        /** @copydoc ITensorGemm::beginOutputPartitionEquivalenceScope */
        std::unique_ptr<OutputPartitionEquivalenceScope>
        beginOutputPartitionEquivalenceScope(int actual_output_columns, int serial_partition_columns) override;
        /** @copydoc ITensorGemm::beginOriginalOutputSliceScope */
        std::unique_ptr<OutputPartitionEquivalenceScope>
        beginOriginalOutputSliceScope(const OriginalOutputSlice &) override;

        /** @copydoc ITensorGemm::multiply_tensor */
        bool multiply_tensor(const TensorBase *input, TensorBase *output,
            int m, int n, int k, bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f, const TensorBase *bias = nullptr,
            const IMPIContext *mpi_ctx = nullptr, int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr, int activation_row_offset = 0) override;
        /** @copydoc ITensorGemm::multiply_tensor_with_fused_swiglu */
        bool multiply_tensor_with_fused_swiglu(const TensorBase *gate, const TensorBase *up,
            TensorBase *output, int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr) override;
        /** @copydoc ITensorGemm::multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent */
        bool multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
            const TensorBase *gate, const TensorBase *up, TensorBase *output,
            int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr) override;
        /** @return True for the implemented native multi-projection workshare. */
        bool supports_fused_projection() const override;
        /** @copydoc ITensorGemm::multiply_fused_tensor */
        bool multiply_fused_tensor(const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections, int m, int k,
            const IMPIContext *mpi_ctx = nullptr, DeviceWorkspaceManager *workspace = nullptr) override;
        /** @copydoc ITensorGemm::multiply_fused_verifier_rows_decode_equivalent */
        bool multiply_fused_verifier_rows_decode_equivalent(const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections, int m, int k,
            const IMPIContext *mpi_ctx = nullptr, DeviceWorkspaceManager *workspace = nullptr) override;

    private:
        /**
         * @brief Reject invalid physical geometry before observing any data pointer.
         * @param m Positive original input rows; live selections may be empty.
         * @param n Exact retained output width.
         * @param k Exact complete reduction width.
         * @throws std::invalid_argument For invalid or inconsistent geometry.
         */
        void requireGeometry(int m, int n, int k) const;
        const WeightStorage weights_; ///< Shared lifetime, never a mutable source-tensor dependency.
    };
} // namespace llaminar2::w4a4
