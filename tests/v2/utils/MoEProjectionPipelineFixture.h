/**
 * @file MoEProjectionPipelineFixture.h
 * @brief One captured projection fixture shared by functional and economy tests.
 *
 * Functional preflight runs all formats and adversarial lifecycles without
 * collecting timing. Explicit performance tests request one geometry/format;
 * they reuse the identical graph lowerer and retain byte-exact checks. The
 * synthetic source weights and redundant oracle are not a model certificate.
 * Explicit A1 geometry timing retains ordinal native membership and checks
 * producer-owned useful packet extents after each timed pair, outside timing.
 */
#pragma once
#include "backends/DeviceId.h"
#include "execution/moe/DeviceMoEOverlayServiceTelemetry.h"
#include <string>
#include <vector>

namespace llaminar2::test
{
/** @brief Select an explicit functional history sweep, never performance instrumentation. */
enum class ProjectionHistoryProof { Disabled, AllDemandBoundaries };

/** @brief Select routed-only math, prefill column completion, or decode's unchanged full-row sum. */
enum class ProjectionOutputProof { RoutedOnly, SharedColumnCompletion, SharedFullRowsOverlap };

/** @brief Explicit native baseline or production counted graph; never chosen on failure. */
enum class ProjectionExchangeProof { Native, DeviceCounted };

/** @brief Preserve legacy reversed order or select an explicit ordinal-native timing domain. */
enum class ProjectionParticipantOrder { Reversed, Ordinal };

/** @brief Synthetic complete-expert geometry; output columns shard over peers. */
struct ProjectionPipelineFixtureGeometry
{
    int experts = 4, width = 256, intermediate = 512, top_k = 2;
};

/** @brief Explicit diagnostic inputs; never an installed execution policy. */
struct ProjectionPipelineFixtureConfig
{
    ProjectionPipelineFixtureGeometry geometry;
    std::vector<int> rows{1, 16, 129};
    std::string format; ///< Empty selects every supported source format.
    ProjectionHistoryProof history = ProjectionHistoryProof::Disabled;
    ProjectionOutputProof output = ProjectionOutputProof::RoutedOnly;
    ProjectionExchangeProof exchange = ProjectionExchangeProof::Native;
    ProjectionParticipantOrder participant_order = ProjectionParticipantOrder::Reversed; ///< Legacy fixtures retain their original order.
    MoEOverlayServicePhaseHint service_phase = MoEOverlayServicePhaseHint::Prefill; ///< Explicit graph identity, never inferred from rows.
    int timing_warmups = 5;
    int timing_samples = 0; ///< Zero keeps functional preflight timing-free.
};

/** @brief Critical participant interval, not a sum or whole-model speed claim. */
struct ProjectionPipelineTiming
{
    double median_us = 0, p10_us = 0, p90_us = 0;
    double unsharded_compute_us = 0; ///< Complete-expert math without inter-device exchange.
    bool counted_extent_checks = false; ///< Device-published useful packet extents checked after every timed pair, not a primitive link-byte receipt.
};

/**
 * @brief Execute the lowerer across native peers and prove retained replay.
 * @param first Backend selector; ordering follows the explicit fixture declaration.
 * @param participants Number of native-domain GPUs, at least two.
 * @param config Geometry, format selection and optional timing cohort.
 * @param timing Non-null iff one format/row is explicitly timed.
 *
 * Every invocation covers twenty changing-owner/full/empty/partial replays,
 * ordinary request reset and retired-graph hard reset. Timing is an additional
 * paired cohort and never weakens those functional checks.
 */
void runProjectionPipelineFixture(DeviceId first, int participants,
    const ProjectionPipelineFixtureConfig &config = {}, ProjectionPipelineTiming *timing = nullptr);
}
