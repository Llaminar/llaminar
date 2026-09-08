/**
 * @file ExpertTierMigrationOperations.h
 * @brief Composite-wave adapters for persistent ExpertOverlay transfer lanes.
 *
 * Physical lanes own streams, events, pinned buffers, and device staging while
 * `MoEOverlayTierMigrationTransport` works in terms of independently pollable
 * projection operations. These adapters join the two lifecycles without adding
 * a wait or cancellation race: abort means "discard the unpublished result and
 * keep pumping its exact events until quiescent," never "destroy pending DMA."
 */

#pragma once

#include "ExpertTierGpuBlobTransferLane.h"
#include "ExpertTierGpuPeerTransferLane.h"
#include "ExpertTierWeightTransferLane.h"
#include "MoEOverlayTierMigrationTransport.h"

#include <memory>
#include <string>

namespace llaminar2
{
    /**
     * @brief Pollable composite-wave operation for one GPU/CPU projection edge.
     *
     * The lane is shared with its topology-owned pool, which materializes it
     * before inference. The operation retains that lane through publication,
     * retirement, or asynchronous abort cleanup. External scheduling must not
     * start another transfer on the lane until this operation is destroyed.
     */
    class ExpertTierWeightLaneOperation final
        : public IMoEOverlayTierTransferOperation
    {
    public:
        /**
         * @brief Bind an already-started persistent CPU-edge lane.
         * @param lane Non-null lane in Pending or Ready state.
         * @throws std::invalid_argument When ownership or lifecycle is invalid.
         */
        explicit ExpertTierWeightLaneOperation(
            std::shared_ptr<ExpertTierWeightTransferLane> lane);

        /** @brief Poll normal transfer progress without waiting. */
        MoEOverlayResidencyWaveProgress poll(
            std::string *error) noexcept override;

        /** @brief Mark the unpublished destination discarded exactly once. */
        void abort() noexcept override;

        /** @brief Pump exact lane events until abort reclamation is safe. */
        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error) noexcept override;

        /** @brief Return the observation captured before the lane can be reused. */
        [[nodiscard]] std::optional<
            ExpertTierProjectionTransferMeasurement>
        completedMeasurement() const noexcept override;

        /** @brief Return the retained lane for evidence and pool accounting. */
        [[nodiscard]] const std::shared_ptr<ExpertTierWeightTransferLane> &lane()
            const noexcept
        {
            return lane_;
        }

    private:
        std::shared_ptr<ExpertTierWeightTransferLane> lane_;
        std::optional<ExpertTierProjectionTransferMeasurement> measurement_;
        bool aborted_ = false;
    };

    /**
     * @brief Pollable composite-wave operation for one CUDA/ROCm blob edge.
     *
     * The adapter preserves the lane's double-buffered source/host/destination
     * event DAG. An abort never attempts cross-runtime cancellation; it drains
     * the already-fenced work and then permits the owning pool to reuse it.
     */
    class ExpertTierGpuBlobLaneOperation final
        : public IMoEOverlayTierTransferOperation
    {
    public:
        /**
         * @brief Bind an already-started persistent heterogeneous GPU lane.
         * @param lane Non-null lane in Pending or Ready state.
         * @throws std::invalid_argument When ownership or lifecycle is invalid.
         */
        explicit ExpertTierGpuBlobLaneOperation(
            std::shared_ptr<ExpertTierGpuBlobTransferLane> lane);

        /** @brief Poll normal cross-runtime transfer progress without waiting. */
        MoEOverlayResidencyWaveProgress poll(
            std::string *error) noexcept override;

        /** @brief Mark the unpublished destination discarded exactly once. */
        void abort() noexcept override;

        /** @brief Pump both runtime event sets until reclamation is safe. */
        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error) noexcept override;

        /** @brief Return the completed cross-runtime transfer observation. */
        [[nodiscard]] std::optional<
            ExpertTierProjectionTransferMeasurement>
        completedMeasurement() const noexcept override;

        /** @brief Return the retained lane for evidence and pool accounting. */
        [[nodiscard]] const std::shared_ptr<ExpertTierGpuBlobTransferLane> &lane()
            const noexcept
        {
            return lane_;
        }

    private:
        std::shared_ptr<ExpertTierGpuBlobTransferLane> lane_;
        std::optional<ExpertTierProjectionTransferMeasurement> measurement_;
        bool aborted_ = false;
    };

    /**
     * @brief Pollable composite-wave operation for one same-backend GPU edge.
     *
     * The retained lane owns its destination stream and event. Abort is a
     * discard protocol: already-submitted peer DMA is event-polled to quiescence
     * before the topology pool may reuse the lane or destination slot.
     */
    class ExpertTierGpuPeerLaneOperation final
        : public IMoEOverlayTierTransferOperation
    {
    public:
        /**
         * @brief Bind one already-started persistent peer lane.
         * @param lane Non-null lane in Pending or Ready state.
         * @throws std::invalid_argument For absent or idle ownership.
         */
        explicit ExpertTierGpuPeerLaneOperation(
            std::shared_ptr<ExpertTierGpuPeerTransferLane> lane);

        /** @brief Poll destination completion without waiting. */
        MoEOverlayResidencyWaveProgress poll(
            std::string *error) noexcept override;

        /** @brief Mark the unpublished destination discarded. */
        void abort() noexcept override;

        /** @brief Drain the exact peer completion event after abort. */
        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error) noexcept override;

        /** @brief Return the completed peer-copy transfer observation. */
        [[nodiscard]] std::optional<
            ExpertTierProjectionTransferMeasurement>
        completedMeasurement() const noexcept override;

        /** @brief Return retained lane ownership for evidence. */
        [[nodiscard]] const std::shared_ptr<ExpertTierGpuPeerTransferLane> &lane()
            const noexcept
        {
            return lane_;
        }

    private:
        std::shared_ptr<ExpertTierGpuPeerTransferLane> lane_;
        std::optional<ExpertTierProjectionTransferMeasurement> measurement_;
        bool aborted_ = false;
    };
} // namespace llaminar2
