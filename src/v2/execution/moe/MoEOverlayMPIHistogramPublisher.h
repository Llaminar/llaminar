/**
 * @file MoEOverlayMPIHistogramPublisher.h
 * @brief Coordinator-to-rank publication of frozen ExpertOverlay histograms.
 *
 * Dynamic placement has one routing-evidence authority: the world rank that
 * owns the logical continuation/router root.  This model-lifetime lane sends
 * that rank's immutable histogram window to every peer over a private MPI
 * communicator. Receives are posted before serving and remain passive across
 * idle/calibration intervals; an initiated coordinator publication owns the
 * bounded progress deadline. Maintenance polls MPI_Test without involving
 * inference.
 */

#pragma once

#include "MoEOverlayHistogramPublisher.h"

#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;

    /** @brief Observable local lifecycle of one histogram publication lane. */
    enum class MoEOverlayMPIHistogramPublisherState
    {
        Idle,
        Receiving,
        Acknowledging,
        Publishing,
        Failed,
        Stopped,
    };

    /**
     * @brief Return whether the local lane owns a bounded publication deadline.
     *
     * A peer receive is a persistent passive mailbox. It may be posted during
     * model setup and remain idle through economy calibration or a long gap
     * between inference windows; no publication has been promised yet, so that
     * dwell is not a stalled collective. Once the coordinator starts sends, or
     * a peer has decoded a generation and started its readiness acknowledgement,
     * that rank owns the canonical progress deadline. The acknowledgement
     * prevents the coordinator from entering residency consensus merely because
     * bytes reached a peer's preposted mailbox.
     *
     * @param state Current process-local histogram lane state.
     * @return True only for an initiated coordinator publication.
     */
    [[nodiscard]] constexpr bool
    moeOverlayHistogramOwnsProgressDeadline(
        MoEOverlayMPIHistogramPublisherState state) noexcept
    {
        return state ==
                   MoEOverlayMPIHistogramPublisherState::Publishing ||
               state ==
                   MoEOverlayMPIHistogramPublisherState::Acknowledging;
    }

    /** @brief Process-local proof counters for coordinator histogram traffic. */
    struct MoEOverlayMPIHistogramPublisherStats
    {
        std::uint64_t receives_armed = 0;
        std::uint64_t publications_started = 0;
        std::uint64_t publications_completed = 0;
        std::uint64_t windows_received = 0;
        std::uint64_t acknowledgements_started = 0;
        std::uint64_t acknowledgements_completed = 0;
        std::uint64_t acknowledgements_received = 0;
        std::uint64_t progress_polls = 0;
        std::uint64_t bytes_sent = 0;
        std::uint64_t bytes_received = 0;
        std::uint64_t validation_failures = 0;
        std::uint64_t mpi_failures = 0;
        std::uint64_t blocking_inference_waits = 0;
    };

    /**
     * @brief Private point-to-point lane for one authoritative frozen window.
     *
     * The coordinator may be any world rank; no device or socket number is
     * inferred from it. Peers keep one exact-size Irecv posted. After a peer
     * decodes a window it acknowledges that exact generation before returning
     * the immutable object to the maintenance service. The coordinator returns
     * Ready only after every peer acknowledgement has arrived. A peer then
     * remains Idle until @ref armReceive is called for the next epoch,
     * preventing a new proposal from overwriting retained evidence.
     */
    class MoEOverlayMPIHistogramPublisher final
        : public IMoEOverlayHistogramPublisher
    {
    public:
        /** @brief Immutable communicator, coordinator, and model geometry. */
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_context;
            int coordinator_world_rank = -1;
            int num_layers = 0;
            int num_experts = 0;
            std::string perf_device;
        };

        /**
         * @brief Materialize private communicator and fixed packet buffers.
         * @param config Multi-rank MPI_THREAD_MULTIPLE context and exact model.
         * @throws std::invalid_argument For invalid rank or model geometry.
         * @throws std::runtime_error For MPI setup/thread-support failures.
         *
         * Non-coordinator ranks post their first receive before returning.
         */
        explicit MoEOverlayMPIHistogramPublisher(Config config);

        /** @brief Drain setup/control-plane MPI and free the communicator. */
        ~MoEOverlayMPIHistogramPublisher();

        MoEOverlayMPIHistogramPublisher(
            const MoEOverlayMPIHistogramPublisher &) = delete;
        MoEOverlayMPIHistogramPublisher &operator=(
            const MoEOverlayMPIHistogramPublisher &) = delete;

        /**
         * @brief Publish one coordinator-frozen generation to every peer.
         * @param window Valid exact-geometry immutable routing evidence.
         * @param error Optional validation or MPI diagnostic.
         * @return True after all non-blocking sends are owned by MPI.
         */
        bool beginPublish(
            const DecodeExpertHistogramWindow &window,
            std::string *error = nullptr) override;

        /**
         * @brief Poll the active send or receive generation once.
         * @param received_window On a peer, receives an authenticated window
         *        only when Ready is returned; coordinator leaves it empty.
         * @param error Optional failure diagnostic.
         * @return Pending, Ready, or Failed without waiting.
         */
        MoEOverlayResidencyWaveProgress poll(
            std::shared_ptr<const DecodeExpertHistogramWindow>
                *received_window,
            std::string *error = nullptr) override;

        /**
         * @brief Post the next fixed receive after the prior wave releases it.
         * @param error Optional state or MPI diagnostic.
         * @return True only when a peer owns one live Irecv.
         *
         * Coordinator calls are rejected. Production calls this after the
         * retained transaction commits or aborts, never while its window is live.
         */
        bool armReceive(std::string *error = nullptr) override;

        /**
         * @brief Cancel/drain setup-control traffic and free no resources yet.
         *
         * This runner-shutdown operation may wait for an already-matched
         * point-to-point request. It is never called from inference or steady
         * maintenance and is idempotent.
         */
        void stopAndDrain() override;

        /** @return Whether this rank is the declared routing coordinator. */
        [[nodiscard]] bool isCoordinator() const noexcept override;

        /** @return Current local lane state. */
        [[nodiscard]] MoEOverlayMPIHistogramPublisherState state()
            const noexcept
        {
            return state_;
        }

        /** @return Process-local lane counters. */
        [[nodiscard]] MoEOverlayMPIHistogramPublisherStats stats()
            const noexcept
        {
            return stats_;
        }

    private:
        /** @brief Convert MPI status into a stable diagnostic. */
        [[nodiscard]] static std::string mpiError(
            const char *operation,
            int mpi_error);

        /** @brief Record one low-frequency PerfStats event. */
        void recordCounter(const char *name, double value = 1.0) const;

        /** @brief Fail the lane once and retain a caller-facing diagnostic. */
        MoEOverlayResidencyWaveProgress fail(
            std::string message,
            std::string *error);

        Config config_;
        MPI_Comm private_communicator_ = MPI_COMM_NULL;
        MPI_Request receive_request_ = MPI_REQUEST_NULL;
        MPI_Request acknowledgement_send_request_ = MPI_REQUEST_NULL;
        std::vector<MPI_Request> send_requests_;
        std::vector<MPI_Request> acknowledgement_receive_requests_;
        std::vector<std::uint64_t> acknowledgement_receive_generations_;
        std::vector<std::uint8_t> wire_buffer_;
        std::shared_ptr<const DecodeExpertHistogramWindow>
            received_window_pending_acknowledgement_;
        std::uint64_t publishing_generation_ = 0;
        std::uint64_t acknowledgement_send_generation_ = 0;
        std::chrono::steady_clock::time_point operation_started_at_{};
        MoEOverlayMPIHistogramPublisherState state_ =
            MoEOverlayMPIHistogramPublisherState::Idle;
        std::string failure_;
        MoEOverlayMPIHistogramPublisherStats stats_;
    };
} // namespace llaminar2
