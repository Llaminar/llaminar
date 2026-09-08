/**
 * @file ExpertTierMigrationOperations.cpp
 * @brief Non-blocking lifecycle adaptation for physical tier-transfer lanes.
 */

#include "ExpertTierMigrationOperations.h"

#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Store an adapter diagnostic only when requested. */
        void assignMigrationOperationError(
            std::string *error,
            const std::string &message) noexcept
        {
            if (error)
                *error = message;
        }

        /** @brief Map one CPU-edge lane result onto the composite wave ABI. */
        MoEOverlayResidencyWaveProgress mapWeightProgress(
            ExpertTierWeightTransferProgress progress) noexcept
        {
            switch (progress)
            {
            case ExpertTierWeightTransferProgress::Pending:
                return MoEOverlayResidencyWaveProgress::Pending;
            case ExpertTierWeightTransferProgress::Ready:
                return MoEOverlayResidencyWaveProgress::Ready;
            case ExpertTierWeightTransferProgress::Idle:
            case ExpertTierWeightTransferProgress::Failed:
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        /** @brief Map one heterogeneous blob result onto the composite wave ABI. */
        MoEOverlayResidencyWaveProgress mapBlobProgress(
            ExpertTierGpuBlobTransferProgress progress) noexcept
        {
            switch (progress)
            {
            case ExpertTierGpuBlobTransferProgress::Pending:
                return MoEOverlayResidencyWaveProgress::Pending;
            case ExpertTierGpuBlobTransferProgress::Ready:
                return MoEOverlayResidencyWaveProgress::Ready;
            case ExpertTierGpuBlobTransferProgress::Idle:
            case ExpertTierGpuBlobTransferProgress::Failed:
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        /** @brief Map one same-backend peer result onto the composite ABI. */
        MoEOverlayResidencyWaveProgress mapPeerProgress(
            ExpertTierGpuPeerTransferProgress progress) noexcept
        {
            switch (progress)
            {
            case ExpertTierGpuPeerTransferProgress::Pending:
                return MoEOverlayResidencyWaveProgress::Pending;
            case ExpertTierGpuPeerTransferProgress::Ready:
                return MoEOverlayResidencyWaveProgress::Ready;
            case ExpertTierGpuPeerTransferProgress::Idle:
            case ExpertTierGpuPeerTransferProgress::Failed:
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            return MoEOverlayResidencyWaveProgress::Failed;
        }
    } // namespace

    ExpertTierWeightLaneOperation::ExpertTierWeightLaneOperation(
        std::shared_ptr<ExpertTierWeightTransferLane> lane)
        : lane_(std::move(lane))
    {
        if (!lane_)
            throw std::invalid_argument(
                "CPU-edge migration operation requires a retained lane");
        const auto progress = lane_->progress();
        if (progress != ExpertTierWeightTransferProgress::Pending &&
            progress != ExpertTierWeightTransferProgress::Ready)
        {
            throw std::invalid_argument(
                "CPU-edge migration operation requires an already-started lane");
        }
    }

    MoEOverlayResidencyWaveProgress ExpertTierWeightLaneOperation::poll(
        std::string *error) noexcept
    {
        if (aborted_)
        {
            assignMigrationOperationError(
                error,
                "Cannot poll normal progress after CPU-edge operation abort");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        const auto progress = lane_->poll(error);
        if (progress == ExpertTierWeightTransferProgress::Ready)
        {
            const auto observed = lane_->stats().last_measurement;
            if (observed.valid())
                measurement_ = observed;
        }
        return mapWeightProgress(progress);
    }

    void ExpertTierWeightLaneOperation::abort() noexcept
    {
        /*
         * GPU APIs do not provide a portable safe cancellation primitive for
         * submitted kernels/DMA. Marking discard while retaining the lane is
         * the asynchronous cancellation protocol.
         */
        aborted_ = true;
    }

    MoEOverlayResidencyWaveProgress
    ExpertTierWeightLaneOperation::pollAbort(std::string *error) noexcept
    {
        if (!aborted_)
        {
            assignMigrationOperationError(
                error,
                "CPU-edge abort cleanup was polled before abort");
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        auto progress = lane_->progress();
        if (progress == ExpertTierWeightTransferProgress::Pending)
            progress = lane_->poll(error);
        if (progress == ExpertTierWeightTransferProgress::Pending)
            return MoEOverlayResidencyWaveProgress::Pending;
        if (lane_->quiescent())
            return MoEOverlayResidencyWaveProgress::Ready;

        assignMigrationOperationError(
            error,
            "CPU-edge lane lost its completion authority during abort");
        return MoEOverlayResidencyWaveProgress::Failed;
    }

    std::optional<ExpertTierProjectionTransferMeasurement>
    ExpertTierWeightLaneOperation::completedMeasurement() const noexcept
    {
        return measurement_;
    }

    ExpertTierGpuBlobLaneOperation::ExpertTierGpuBlobLaneOperation(
        std::shared_ptr<ExpertTierGpuBlobTransferLane> lane)
        : lane_(std::move(lane))
    {
        if (!lane_)
            throw std::invalid_argument(
                "Heterogeneous GPU migration operation requires a retained lane");
        const auto progress = lane_->progress();
        if (progress != ExpertTierGpuBlobTransferProgress::Pending &&
            progress != ExpertTierGpuBlobTransferProgress::Ready)
        {
            throw std::invalid_argument(
                "Heterogeneous GPU migration operation requires an already-started lane");
        }
    }

    MoEOverlayResidencyWaveProgress ExpertTierGpuBlobLaneOperation::poll(
        std::string *error) noexcept
    {
        if (aborted_)
        {
            assignMigrationOperationError(
                error,
                "Cannot poll normal progress after heterogeneous GPU operation abort");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        const auto progress = lane_->poll(error);
        if (progress == ExpertTierGpuBlobTransferProgress::Ready)
        {
            const auto observed = lane_->stats().last_measurement;
            if (observed.valid())
                measurement_ = observed;
        }
        return mapBlobProgress(progress);
    }

    void ExpertTierGpuBlobLaneOperation::abort() noexcept
    {
        /* Both runtimes keep their own fenced slots; drain instead of cancel. */
        aborted_ = true;
    }

    MoEOverlayResidencyWaveProgress
    ExpertTierGpuBlobLaneOperation::pollAbort(std::string *error) noexcept
    {
        if (!aborted_)
        {
            assignMigrationOperationError(
                error,
                "Heterogeneous GPU abort cleanup was polled before abort");
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        auto progress = lane_->progress();
        if (progress == ExpertTierGpuBlobTransferProgress::Pending)
            progress = lane_->poll(error);
        if (progress == ExpertTierGpuBlobTransferProgress::Pending)
            return MoEOverlayResidencyWaveProgress::Pending;
        if (lane_->quiescent())
            return MoEOverlayResidencyWaveProgress::Ready;

        assignMigrationOperationError(
            error,
            "Heterogeneous GPU lane lost a completion authority during abort");
        return MoEOverlayResidencyWaveProgress::Failed;
    }

    std::optional<ExpertTierProjectionTransferMeasurement>
    ExpertTierGpuBlobLaneOperation::completedMeasurement() const noexcept
    {
        return measurement_;
    }

    ExpertTierGpuPeerLaneOperation::ExpertTierGpuPeerLaneOperation(
        std::shared_ptr<ExpertTierGpuPeerTransferLane> lane)
        : lane_(std::move(lane))
    {
        if (!lane_)
            throw std::invalid_argument(
                "Same-backend GPU migration operation requires a retained lane");
        const auto progress = lane_->progress();
        if (progress != ExpertTierGpuPeerTransferProgress::Pending &&
            progress != ExpertTierGpuPeerTransferProgress::Ready)
        {
            throw std::invalid_argument(
                "Same-backend GPU migration operation requires an already-started lane");
        }
    }

    MoEOverlayResidencyWaveProgress ExpertTierGpuPeerLaneOperation::poll(
        std::string *error) noexcept
    {
        if (aborted_)
        {
            assignMigrationOperationError(
                error,
                "Cannot poll normal progress after GPU peer operation abort");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        const auto progress = lane_->poll(error);
        if (progress == ExpertTierGpuPeerTransferProgress::Ready)
        {
            const auto observed = lane_->stats().last_measurement;
            if (observed.valid())
                measurement_ = observed;
        }
        return mapPeerProgress(progress);
    }

    void ExpertTierGpuPeerLaneOperation::abort() noexcept
    {
        /* Runtime peer DMA cannot be cancelled safely; retain and drain it. */
        aborted_ = true;
    }

    MoEOverlayResidencyWaveProgress ExpertTierGpuPeerLaneOperation::pollAbort(
        std::string *error) noexcept
    {
        if (!aborted_)
        {
            assignMigrationOperationError(
                error,
                "GPU peer abort cleanup was polled before abort");
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        auto progress = lane_->progress();
        if (progress == ExpertTierGpuPeerTransferProgress::Pending)
            progress = lane_->poll(error);
        if (progress == ExpertTierGpuPeerTransferProgress::Pending)
            return MoEOverlayResidencyWaveProgress::Pending;
        if (lane_->quiescent())
            return MoEOverlayResidencyWaveProgress::Ready;

        assignMigrationOperationError(
            error,
            "GPU peer lane lost its completion authority during abort");
        return MoEOverlayResidencyWaveProgress::Failed;
    }

    std::optional<ExpertTierProjectionTransferMeasurement>
    ExpertTierGpuPeerLaneOperation::completedMeasurement() const noexcept
    {
        return measurement_;
    }
} // namespace llaminar2
