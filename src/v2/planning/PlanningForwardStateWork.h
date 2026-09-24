/**
 * @file PlanningForwardStateWork.h
 * @brief Bounded main-forward attention/recurrent work from admitted ownership.
 *
 * This is an analytic work projection, not a graph simulator or memory ledger.
 * It reuses canonical TP assignments, layer classification, KV payload codecs
 * and GDN geometry. Live positions, not reserved cache horizons, determine work.
 * Pricing uses an explicitly qualified FP32 arithmetic proxy and independent
 * streaming service; it does not advertise GEMM timing as measured attention.
 */
#pragma once
#include "PlanningForwardWeightWork.h"
#include "PlanningMemoryBandwidthMeasurement.h"
#include "PlanningProjectionMeasurement.h"
#include "kernels/HybridGDNStateGeometry.h"
#include <variant>

namespace llaminar2
{
    class PlanningModelMetadata;
    struct DevicePlanConfig;

    /** @brief One nonempty main-forward invocation, possibly at the mean decode context. */
    class PlanningForwardInvocation final
    {
    public:
        /**
         * @brief Reject empty rows and negative/nonfinite preceding context.
         * @param rows Newly appended query rows, not retained graph capacity.
         * @param preceding_context Previously live positions; fractional means
         *        analytically averaged decode contexts, never fractional inference.
         */
        PlanningForwardInvocation(int rows, double preceding_context);
        /** @return Newly appended logical rows. */
        int rows() const noexcept { return rows_; }
        /** @return Positions live before this invocation. */
        double precedingContext() const noexcept { return preceding_context_; }
        /** @return Exact causal query/key pair count, including each query's own key. */
        double causalPairs() const noexcept;
    private:
        int rows_;
        double preceding_context_;
    };

    /** @brief Attention algebra and minimum logical traffic, excluding its weight projections. */
    struct PlanningAttentionStateWork
    {
        int query_heads;
        int kv_heads;
        int head_dim;
        double causal_pairs;
        double operations; ///< QK/PV plus scalar softmax proxy; one FMA counts twice.
        double kv_read_bytes; ///< One logical traversal, with ideal reuse across query rows/heads.
        double kv_write_bytes; ///< Only new native K/V positions, never cache capacity.
        double activation_bytes; ///< FP32 Q/K/V input and attention output.
    };

    /** @brief Equivalent recurrent algebra; no claim that chunked GDN executes a serial-row loop. */
    struct PlanningGDNStateWork
    {
        HybridGDNStateGeometry geometry;
        double operations;
        double state_bytes; ///< Read/write local banks once with ideal invocation-local reuse.
        double activation_bytes;
    };

    /** @brief One typed layer's state kernel family and explicit logical invocation. */
    struct PlanningLayerStateWork
    {
        int layer;
        int rows;
        std::variant<PlanningAttentionStateWork, PlanningGDNStateWork> operation;
    };

    /**
     * @brief Project only main continuation state from the compiled participant.
     * @param model Immutable GGUF metadata with its authenticated main-layer boundary.
     * @param device Canonical admitted participant, including exact TP assignment/PP interval.
     * @param phase Select primary prefill or installed replicated-dense decode ownership.
     * @param invocation Live work inside admitted row/context capacity.
     * @return One entry per owned main layer; none for expert-only participants.
     * @throws std::exception for incomplete geometry, unsupported codecs or invalid invocation.
     *
     * Retained MTP predictors, graph capacity, prefix archives and replicated
     * serialization banks are not extra main-forward work. This projection
     * deliberately excludes weight projections, norms/FFNs and communication;
     * those remain dependent work for the request-level composer.
     */
    std::vector<PlanningLayerStateWork> compilePlanningForwardStateWork(
        const PlanningModelMetadata &model, const DevicePlanConfig &device,
        PlanningMainForwardPhase phase, const PlanningForwardInvocation &invocation);

    /**
     * @brief Price one state family using qualified FP32 and streaming proxies.
     * @param work Analytic work, not sampled model latency or memory capacity.
     * @param arithmetic Dedicated source-free FP32 evidence; quantized GOPS cannot substitute.
     * @param memory Independent streaming observation for the same device/workshare.
     * @return The larger arithmetic/byte service demand in seconds.
     * @throws std::exception for mismatched devices, CPU workshares, units or missing phases.
     *
     * This is a bounded roofline proxy, not a measured attention/GDN invocation.
     * No speculative acceptance, migration benefit or fitted kernel-launch tax
     * is invented. Distributed callers must additionally authenticate both
     * observations through the same physical observer in their catalogs.
     */
    double planningStateServiceSeconds(const PlanningLayerStateWork &work,
        const PlanningFP32ArithmeticObservations &arithmetic,
        const PlanningMemoryBandwidthObservation &memory);
}
