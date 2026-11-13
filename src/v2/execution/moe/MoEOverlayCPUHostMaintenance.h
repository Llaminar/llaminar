/**
 * @file MoEOverlayCPUHostMaintenance.h
 * @brief Stage-local lifetime of admitted CPU residency maintenance.
 *
 * A pipeline stage retains the same physical admission, global layer interval,
 * NUMA endpoints and histogram as its prepared expert banks. This owner never
 * installs a parent-wide authority or constructs a second memory ledger. Setup
 * samples are temporary; the physical fabric outlives its wave factory,
 * transport, calibration and sole background worker in that dependency order.
 */
#pragma once
#include "MoEOverlayCapacityAdmission.h"
#include "MoEOverlayPhysicalResidencyFabric.h"
#include "MoEOverlayResidencyMaintenanceService.h"

namespace llaminar2
{
    class MoEOverlayCPUHostMaintenance final
    {
    public:
        /** @brief Complete frozen scope and existing owners; construction allocates no payload. */
        struct Config
        {
            MoERoutedExpertModelMetadata metadata;
            MoEOverlayCapacityAdmissionPolicy storage;
            MoERebalanceRuntimeConfig rebalance;
            ExpertHistogramProductionTopology topology;
            std::shared_ptr<PhysicalMemoryAuthority> memory;
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> admission;
            std::shared_ptr<MoEOverlayResidencyAuthority> authority;
            std::shared_ptr<MoEOverlayParticipantResidencyRegistry> residency;
            std::shared_ptr<DecodeExpertHistogram> histogram;
            std::shared_ptr<const MoEOverlaySealedMigrationMeasurements> presealed_measurements;
            std::string perf_device;
        };

        /** @brief Authenticate one complete process-local CPU-only mutable stage. */
        explicit MoEOverlayCPUHostMaintenance(Config config);
        /** @brief Drain the worker before destroying the physical owners it borrows. */
        ~MoEOverlayCPUHostMaintenance();
        /** @brief Materialize existing admitted roots and observe prepared service on the affined setup thread. */
        void prepare();
        /** @brief Activate exactly once after serving graph publication. */
        void start();
        /** @brief Signal committed model work, never routing or placement evidence. */
        void notify(MoEOverlayInferenceProgress progress) noexcept;
        /** @brief Quiesce all asynchronous work before graph/physical retirement; repeatable. */
        void stopAndDrain();
        /** @return Whether physical preparation completed and the service remains healthy. */
        [[nodiscard]] bool ready() const noexcept;
        /** @brief Require the exact authority, bank registry and global interval supplied at declaration. */
        void requireScope(const MoEOverlayResidencyAuthority *authority,
            const MoEOverlayParticipantResidencyRegistry *residency,
            const MoERoutedExpertModelMetadata &metadata) const;
        /** @return Sole worker, absent until successful prepare; retained for nonpipeline lifecycle reuse. */
        MoEOverlayResidencyMaintenanceService *service() const noexcept { return service_.get(); }
        /** @return Setup calibration retained by this same physical lifetime. */
        MoEOverlayEconomyCalibrationController *calibration() const noexcept { return calibration_.get(); }
        /** @brief Restore loader-era placement and seal its physical banks after drain, within the standard deadline. */
        [[nodiscard]] MoEOverlayReusableContextSeal restorePreparedPlacement();
        /** @return Exact stage placement authority for passive publication. */
        const std::shared_ptr<MoEOverlayResidencyAuthority> &authority() const noexcept { return config_.authority; }

    private:
        Config config_;
        std::shared_ptr<MoEOverlayPhysicalResidencyFabric> fabric_;
        std::unique_ptr<MoEOverlayParticipantPreparedWaveFactory> factory_;
        std::shared_ptr<MoEOverlayTierMigrationTransport> transport_;
        std::shared_ptr<MoEOverlayEconomyCalibrationController> calibration_;
        std::shared_ptr<MoEOverlayResidencyMaintenanceService> service_;
        bool drained_ = false;
    };
}
