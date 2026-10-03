/**
 * @file TPColumnReduceScatterStage.h
 * @brief Native shared-row sum whose result stays with each output-column owner.
 *
 * Two existing arena banks alternate roles: row partials are packed into the
 * later column-allgather bank, summed by NCCL/RCCL, and returned as contiguous
 * local columns in the original partial bank. FP16 transport reuses these same
 * banks for conversion. No allocation, host barrier or extra tensor is needed.
 */
#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "backends/GlobalDeviceAddress.h"
#include "memory/BufferId.h"
#include "kernels/common/DeviceRowRange.h"

#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    class ILocalTPContext;
    class AcquiredDeviceTransferInput;

    /** @brief Capturable sum with column ownership and an exact-stream publication. */
    class TPColumnReduceScatterStage final : public IComputeStage
    {
    public:
        /** @brief Immutable geometry over two disjoint, already admitted FP32 banks. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            ILocalTPContext *tp_ctx = nullptr;
            ITensor *tensor = nullptr; ///< Full row partial in; packed local columns out.
            ITensor *packing = nullptr; ///< Full-row scratch, reused later by column allgather.
            int rows = 0;
            int model_columns = 0; ///< Original row width, including every participant's columns.
            int participant = -1; ///< Communicator coordinate, never physical ordinal.
            std::string precision; ///< Resolved allreduce wire policy, retained unchanged.
            std::string stage_name;
            std::optional<BufferId> tensor_buffer_id;
            std::optional<BufferId> packing_buffer_id;
            std::optional<DeviceRowRange> live_rows; ///< Canonical query prefix; physical rank strides remain fixed.
        };
        static_assert(StageParamsRequired<Params>);

        /** @brief Freeze native membership, precision and safe disjoint buffer geometry. */
        explicit TPColumnReduceScatterStage(Params params);
        /** @brief Submit on the exact graph stream and publish both written banks. */
        bool execute(IDeviceContext *ctx) override;
        /** @brief Authenticate complete resident storage before an ordinary or forked enqueue. */
        void validateEnqueue(const StageGPUExecution &execution) const;
        /** @brief Enqueue using TransferEngine's acquired producer frontier; does not publish. */
        bool enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const;
        /** @return Immutable declaration shared by the paired overlap builder. */
        const Params &params() const noexcept { return params_; }
        /** @return Explicit native reduction classification for graph capture/scheduling. */
        ComputeStageType type() const override { return ComputeStageType::NATIVE_REDUCE_SCATTER; }
        /** @return Frozen graph identity. */
        std::string name() const override { return params_.stage_name; }
        /** @return True: this is an explicit collective, not disguised compute. */
        bool requiresAllreduce() const override { return true; }
        /** @return Only the declared compiled GPU backend. */
        bool supportsBackend(ComputeBackendType backend) const override;
        /** @return Native graph support after checking unchanged membership. */
        bool isGraphCapturable() const override;
        /** @return Exact original row inout and the disjoint packing-bank write. */
        StageBufferContract bufferContract() const override;
        /** @return Borrowed full-bank geometry; no new workspace reservation. */
        StageBufferRequirements getBufferRequirements() const override;
        /** @return Only the locally owned result columns, never a replicated full row. */
        StageDumpInfo buildDumpInfoImpl() const override;
        /** @return The executor prepares both output banks before recording. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }

    private:
        /** @brief Reject changed/native-incompatible membership without GPU discovery. */
        void validateMembership() const;
        /** @brief Validate capacities and physical ranges; PMA remains the byte authority. */
        void validateBuffers() const;
        /** @brief Ordered pack, native sum and optional casts on one non-null stream. */
        bool enqueue(void *stream) const;

        const Params params_;
        const std::vector<GlobalDeviceAddress> members_;
        std::size_t elements_ = 0;
        bool fp16_ = false; ///< Frozen wire arithmetic, never request-time state.
    };
}
