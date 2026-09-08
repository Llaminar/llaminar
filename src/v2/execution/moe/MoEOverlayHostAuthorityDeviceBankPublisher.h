/**
 * @file MoEOverlayHostAuthorityDeviceBankPublisher.h
 * @brief Event-driven GPU bank publication for host-authority ExpertOverlay.
 *
 * A heterogeneous or multi-tier overlay keeps placement policy on the host,
 * but each CUDA/ROCm participant still owns its live placement banks and RCU
 * selector on device.  This file defines the one bridge between those
 * authorities.  It prepares complete inactive banks from immutable participant
 * residency leases, uploads them on persistent background streams, publishes
 * selectors only after semantic readiness, and retires old device banks by
 * polling exact events.  Inference streams are never synchronized or blocked.
 */

#pragma once

#include "MoEOverlayDeviceControllerRuntimeBinding.h"
#include "MoEOverlayTierMigrationTransport.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    struct MoEOverlayResidencySnapshot;
    class MoEOverlayParticipantResidencyRegistry;

    /**
     * @brief Factory for one process-local host-to-device epoch transaction.
     *
     * The returned object uses the same typed lifecycle as the host participant
     * banks: prepare, publish, abort, device-reader retirement, then reclaim.
     * Keeping one contract lets the outer participant transaction compose CPU,
     * CUDA, and ROCm endpoints without parallel flag sets or caller discipline.
     */
    class IMoEOverlayHostAuthorityDeviceBankPublisher
    {
    public:
        virtual ~IMoEOverlayHostAuthorityDeviceBankPublisher() = default;

        /**
         * @brief Reserve one device transaction for an already-installed host candidate.
         * @param previous Immutable currently published residency snapshot.
         * @param candidate Immutable successor whose host banks are exact-addressable.
         * @param error Optional validation or overlap diagnostic.
         * @return One inactive-bank lifecycle, or null before any device work starts.
         */
        [[nodiscard]] virtual std::unique_ptr<
            IMoEOverlayInactiveBankTransaction>
        createTransaction(
            std::shared_ptr<const MoEOverlayResidencySnapshot> previous,
            std::shared_ptr<const MoEOverlayResidencySnapshot> candidate,
            std::string *error = nullptr) noexcept = 0;
    };

    /**
     * @brief Persistent CUDA/ROCm publisher for a host-resident overlay authority.
     *
     * Model setup allocates one mapped staging page, exact background stream,
     * semantic-status allocation, and terminal event per local GPU.  A wave
     * reuses those resources and advances every endpoint independently, so
     * CUDA and ROCm preparation/publication overlap instead of serializing by
     * participant.  The class supports exactly one wave at a time because the
     * residency authority itself admits exactly one unpublished candidate.
     */
    class MoEOverlayHostAuthorityDeviceBankPublisher final
        : public IMoEOverlayHostAuthorityDeviceBankPublisher
    {
    public:
        /** @brief Immutable model topology and participant-bank source. */
        struct Config
        {
            std::vector<MoEOverlayDeviceControllerRuntimeBinding>
                runtime_bindings;
            std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
            std::string perf_device;
        };

        /**
         * @brief Materialize every persistent publication resource at model setup.
         * @throws std::invalid_argument For empty, duplicate, or incomplete bindings.
         * @throws std::runtime_error When a backend resource cannot be retained.
         */
        explicit MoEOverlayHostAuthorityDeviceBankPublisher(Config config);

        /** @brief Release setup resources after the maintenance service drains. */
        ~MoEOverlayHostAuthorityDeviceBankPublisher() override;

        MoEOverlayHostAuthorityDeviceBankPublisher(
            const MoEOverlayHostAuthorityDeviceBankPublisher &) = delete;
        MoEOverlayHostAuthorityDeviceBankPublisher &operator=(
            const MoEOverlayHostAuthorityDeviceBankPublisher &) = delete;

        /** @copydoc IMoEOverlayHostAuthorityDeviceBankPublisher::createTransaction */
        [[nodiscard]] std::unique_ptr<IMoEOverlayInactiveBankTransaction>
        createTransaction(
            std::shared_ptr<const MoEOverlayResidencySnapshot> previous,
            std::shared_ptr<const MoEOverlayResidencySnapshot> candidate,
            std::string *error = nullptr) noexcept override;

        /** @return Number of process-local GPU endpoints retained at setup. */
        [[nodiscard]] std::size_t endpointCount() const noexcept;

        /** @brief Opaque shared implementation retained by active wave proxies. */
        struct State;

    private:
        std::shared_ptr<State> state_;
    };
} // namespace llaminar2
