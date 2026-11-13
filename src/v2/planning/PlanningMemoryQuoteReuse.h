/**
 * @file PlanningMemoryQuoteReuse.h
 * @brief Source-bound lexical reuse of pure workspace quotes during one exhaustive search.
 *
 * Quotes contain no device allocation, admission, quota, capacity or certificate.
 * Exact typed geometry equality is authoritative. Original physical bindings
 * still reach PMA separately for every candidate and every resource.
 */
#pragma once
#include "planning/PlanningModelMetadata.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "planning/ActivationMemoryEstimator.h"
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include <memory>
namespace llaminar2
{
    /** @brief Device-free expensive-work counts retained by the lexical search owner. */
    struct PlanningMemoryQuoteReuseCounts
    {
        size_t workspace_requests=0,workspace_builds=0,workspace_hits=0;
        size_t projection_arena_requests=0,projection_arena_builds=0,projection_arena_hits=0;
        size_t prepared_service_requests=0,prepared_service_builds=0,prepared_service_hits=0;
        size_t activation_requests=0,activation_builds=0,activation_hits=0;
    };
    /** @brief One original source/immutable index lifetime, never a process-global persistent cache. */
    class PlanningMemoryQuoteReuse final
    {
    public:
        /** @brief Bind this original source on the current planning thread; nested scopes are rejected. */
        explicit PlanningMemoryQuoteReuse(std::shared_ptr<const PlanningTensorIndex> source);
        /** @brief Revoke all quotes before their source/search lifetime ends. */
        ~PlanningMemoryQuoteReuse();
        PlanningMemoryQuoteReuse(const PlanningMemoryQuoteReuse&)=delete;
        PlanningMemoryQuoteReuse& operator=(const PlanningMemoryQuoteReuse&)=delete;
        /** @return Exact canonical quote, compiled once per full actual function input inside the bound scope. */
        static size_t workspace(const ModelMemoryProfile &profile,const WorkspaceMemoryGeometry &geometry);
        /** @return Canonical metadata-only arena geometry, reused only for this original source and exact degree. */
        static MoEProjectionArenaGeometry projectionArena(const ModelMemoryProfile &profile,int participants);
        /** @return Canonical prepared-service bytes, keyed by the complete original workspace geometry. */
        static size_t preparedExpertService(const ModelMemoryProfile &profile,const WorkspaceMemoryGeometry &geometry);
        /** @return Canonical activation bytes for the complete geometry and exact participant device. */
        static size_t activation(const ModelMemoryProfile &profile,const ActivationGraphMemoryGeometry &geometry,DeviceId device);
        /** @return Original source index for canonical estimators, or empty outside a scoped search. */
        static std::shared_ptr<const PlanningTensorIndex> sourceIndex(const ModelMemoryProfile &profile);
        /** @return Pure compilation counts; no measured hardware performance is asserted. */
        PlanningMemoryQuoteReuseCounts counts() const noexcept;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        static thread_local PlanningMemoryQuoteReuse *active_;
    };
}
