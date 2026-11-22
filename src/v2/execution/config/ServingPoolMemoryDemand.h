/**
 * @file ServingPoolMemoryDemand.h
 * @brief Typed physical-memory demand for the accepted serving response pool.
 *
 * The serving capacity contract (design §9.4) bounds accepted response leases,
 * resident slots and queue capacity. Their hosting state (pool records,
 * delivery staging and detached-response retention) is real host memory, so
 * the capacity must be admitted by the same authority as every other
 * allocation (PhysicalMemoryAuthority / PhysicalMemoryBOM): this header is
 * the typed estimator input that converts one resolved serving capacity into
 * the exact owner charge. PRODUCTION state: MemoryPlanner writes the
 * ServingResponsePool line directly from servingPoolMemoryDemand-computed
 * bytes (the runner/candidate BOM gates); the addServingPoolDemand helper
 * below is the BOM-builder path (tests/future aggregate builders). The
 * candidate-admission and MoE pipeline/overlay aggregate wiring is still
 * INCOMPLETE (recorded open) — no all-entrypoints claim is made here.
 */

#pragma once

#include "execution/config/ServingCapacity.h"
#include "planning/PhysicalMemoryBOM.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief Per-accepted-response hosting budget for the serving pool. */
    struct ServingPoolPerResponseBudget
    {
        /** @brief Bytes admitted per accepted response lease (pool record + retained delivery staging). */
        std::uint64_t bytes_per_response = 0;

        /** @brief Starting POLICY default for the per-response hosting budget.
         *
         *  One accepted response lease hosts the pool record plus its detached
         *  delivery staging (bounded response window). This is a policy input to
         *  the typed BOM estimation — the value is expected to become plan/
         *  configuration-derived; it is a starting default, not certified
         *  geometry. */
        [[nodiscard]] static ServingPoolPerResponseBudget defaultPolicy()
        {
            return ServingPoolPerResponseBudget{64 * 1024};
        }
    };

    /**
     * @brief Compute the exact serving-pool physical demand for one resolved capacity.
     *
     * The pool hosts one budgeted region per accepted response lease (the
     * response bound R already covers resident slots plus queue capacity by
     * the §9.4 invariant). The result is the single typed BOM charge for
     * @ref PhysicalMemoryOwner::ServingResponsePool; it is never recomputed
     * elsewhere.
     *
     * @param capacity Resolved serving capacity (R/S/Q, already envelope-checked).
     * @param budget Typed per-response hosting budget.
     * @return Exact total demand in bytes.
     * @throws std::invalid_argument For a non-positive budget.
     * @throws std::overflow_error When the demand cannot be represented.
     */
    [[nodiscard]] inline std::uint64_t servingPoolMemoryDemand(
        const ResolvedServingCapacity &capacity,
        const ServingPoolPerResponseBudget &budget)
    {
        if (budget.bytes_per_response == 0)
            throw std::invalid_argument("Serving pool per-response budget must be positive");
        const auto leases = static_cast<std::uint64_t>(capacity.responseLeases());
        if (leases == 0)
            throw std::invalid_argument("Serving pool response leases must be positive");
        if (budget.bytes_per_response >
            std::numeric_limits<std::uint64_t>::max() / leases)
            throw std::overflow_error("Serving pool memory demand overflows uint64");
        return budget.bytes_per_response * leases;
    }

    /**
     * @brief FAIL-CLOSED serving-pool demand decision for one rank (BOM gate).
     *
     * Serving requires a SEALED request authority before BOM admission:
     *   - not serving           -> no demand;
     *   - serving, UNSEALED     -> fatal (never a silent non-owner);
     *   - serving, rank != owner-> no demand (follower, no pool charge);
     *   - serving, rank == owner-> the exact typed demand.
     * The runner's seal phase engages the authority beforehand; the runner BOM
     * gate calls this helper as the single decision point (candidate/MoE
     * aggregate wiring remains open — no all-entrypoints claim).
     * @throws std::logic_error When serving is selected without a sealed
     *         request authority.
     */
    [[nodiscard]] inline std::optional<std::uint64_t> servingPoolDemandForRank(
        bool serve_mode,
        const std::optional<int> &sealed_authority,
        int rank,
        const ServingCapacityRequest &request,
        const ServingPoolPerResponseBudget &budget = ServingPoolPerResponseBudget::defaultPolicy())
    {
        if (!serve_mode)
            return std::nullopt;
        if (!sealed_authority)
            throw std::logic_error(
                "Serve mode requires a sealed request authority before BOM admission");
        if (rank != *sealed_authority)
            return std::nullopt; // follower rank: no serving pool charge
        return servingPoolMemoryDemand(resolveInstalledServingCapacity(request), budget);
    }

    /**
     * @brief Charge one serving pool demand on a physical-memory BOM builder.
     *  The builder is the sole charge authority; this helper writes the owner
     *  line through the builder API. The PRODUCTION ordinary path writes the
     *  same typed bytes directly in MemoryPlanner; candidate-admission and MoE
     *  pipeline/overlay aggregate wiring is still INCOMPLETE (open).
     */
    inline void addServingPoolDemand(PhysicalMemoryBOMBuilder &builder,
                                     const ResolvedServingCapacity &capacity,
                                     const ServingPoolPerResponseBudget &budget)
    {
        builder.add(PhysicalMemoryOwner::ServingResponsePool,
                    servingPoolMemoryDemand(capacity, budget));
    }
} // namespace llaminar2
