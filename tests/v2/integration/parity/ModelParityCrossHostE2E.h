/**
 * @file ModelParityCrossHostE2E.h
 * @brief Provider-independent declarations for public-frontend MPI certification.
 *
 * A selected model cell supplies weights, precision, MTP and prefix policy.
 * This declaration adds remote CPU compute hosts, not fabricated local NUMA
 * devices. Azure location, credentials, VM SKU and network addresses belong to
 * resource provisioning and never become a second model/topology matrix.
 */
#pragma once

#include <array>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace llaminar2::test::parity
{
    /** @brief Public routes that must independently execute every remote topology. */
    enum class ModelParityCrossHostFrontend
    {
        PlanApply, ///< Default-auto plan, then serve the exact emitted configuration.
        AutoServe, ///< Default-auto serve with the same hostfile and constraints.
    };

    /** @return Stable report spelling; an unknown route cannot silently omit a proof. */
    inline std::string_view modelParityCrossHostFrontendName(ModelParityCrossHostFrontend route)
    {
        switch (route)
        {
        case ModelParityCrossHostFrontend::PlanApply: return "plan-apply";
        case ModelParityCrossHostFrontend::AutoServe: return "auto-serve";
        }
        throw std::invalid_argument("invalid cross-host E2E frontend");
    }

    /** Both routes are mandatory; drivers consume their expansion, not another list. */
    inline constexpr std::array kModelParityCrossHostFrontends{
        ModelParityCrossHostFrontend::PlanApply,
        ModelParityCrossHostFrontend::AutoServe,
    };

    /**
     * @brief Positive remote-host count for one-GPU/remote-CPU overlay intent.
     *
     * There is one CPU execution rank on each distinct remote physical host.
     * The continuation GPU lives on a different host and owns one rank. This
     * is intentionally independent of CPU socket indices and cloud VM names.
     * Every remote rank must execute routed experts and exchange MPI payloads;
     * merely launching extra ranks or choosing an all-local plan is not proof.
     */
    class ModelParityRemoteCPUHosts final
    {
    public:
        /**
         * @brief Seal the host count before matrix expansion or cloud admission.
         * @param count Number of distinct remote CPU hosts, each with one rank.
         * @throws std::invalid_argument if the complete communicator cannot fit.
         */
        explicit ModelParityRemoteCPUHosts(int count) : count_(count)
        {
            if (count <= 0 || count == std::numeric_limits<int>::max())
                throw std::invalid_argument("cross-host E2E requires positive CPU hosts and space for the continuation rank");
        }

        /** @return Remote physical hosts; never a launcher-local CPU device index. */
        [[nodiscard]] int count() const noexcept { return count_; }
        /** @return Complete selected execution membership including the GPU owner. */
        [[nodiscard]] int executionRanks() const noexcept { return count_ + 1; }
        friend bool operator==(const ModelParityRemoteCPUHosts &,
                               const ModelParityRemoteCPUHosts &) = default;

    private:
        int count_;
    };
}
