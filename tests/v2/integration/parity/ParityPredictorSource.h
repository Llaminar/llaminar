/**
 * @file ParityPredictorSource.h
 * @brief Preserve explicit optional predictor identity through parity setup.
 *
 * A predictor is a second admitted GGUF source, including MTP-off controls.
 * The existing campaign manifest authenticates and stages both artifacts; no
 * environment variable selects a predictor and no filename discovers one.
 */
#pragma once
#include "ProductionParityModelPath.h"
#include "loaders/ModelContextConfig.h"
#include "loaders/SharedPredictorGGUFSource.h"
#include <memory>
#include <utility>
#include <optional>
#include <stdexcept>
#include <string>

namespace llaminar2::test::parity
{
/**
 * @brief Resolve a selected predictor through the same manifest as the main GGUF.
 * @param source Explicit selection, absent for main-only or inline-MTP sources.
 * @return Authenticated staged path, or the unchanged focused path.
 * @throws std::runtime_error Before materialization if a campaign did not declare it.
 */
inline std::optional<SharedPredictorGGUFSource> resolveParityPredictorSource(
    const std::optional<SharedPredictorGGUFSource> &source)
{
    if (!source) return std::nullopt;
    return SharedPredictorGGUFSource(productionParityResolvedModelPath(source->path()));
}
/**
 * @brief Bind the fixture-owned predictor while rejecting a foreign runner source.
 * @param selected Resolved immutable fixture selection.
 * @param requested Resolved optional source authored by a runner configuration.
 * @return Fixture source; an omitted runner field inherits that explicit value.
 * @throws std::invalid_argument If a runner requests a different or unselected source.
 */
inline std::optional<SharedPredictorGGUFSource> bindParityPredictorSource(
    const std::optional<SharedPredictorGGUFSource> &selected,
    const std::optional<SharedPredictorGGUFSource> &requested)
{
    if (requested && requested != selected)
        throw std::invalid_argument("Parity runner selected a foreign shared predictor GGUF");
    return selected;
}
/**
 * @brief Extend an immutable cache identity only when a predictor is selected.
 * @param source Exact selected artifact, never a guessed sidecar filename.
 * @return Empty for legacy sources; otherwise a length-delimited path identity.
 */
inline std::string parityPredictorIdentity(
    const std::optional<SharedPredictorGGUFSource> &source)
{
    return source ? "|predictor_bytes=" + std::to_string(source->path().size()) +
        ":" + source->path() : std::string{};
}
/**
 * @brief Build the actual loader contract before ModelContext materialization.
 * @param mpi Canonical fixture MPI authority.
 * @param strategy Existing target weight distribution policy.
 * @param source Resolved explicit predictor source, absent for legacy loading.
 * @param retained_mtp_capacity Physical predictor capacity, including MTP-off controls.
 * @return One ModelContextConfig carrying both source and usage policy.
 * @throws std::invalid_argument If the declared capacity is negative.
 */
inline ModelContextConfig parityModelContextConfig(std::shared_ptr<IMPIContext> mpi,
    WeightDistributionStrategy strategy,
    const std::optional<SharedPredictorGGUFSource> &source, int retained_mtp_capacity)
{
    if (retained_mtp_capacity < 0)
        throw std::invalid_argument("Parity model context has negative retained MTP capacity");
    ModelContextConfig config;
    config.mpi_ctx = std::move(mpi);
    config.strategy = strategy;
    config.shared_predictor = source;
    config.predictor_materialization = source && retained_mtp_capacity > 0 ?
        SharedPredictorMaterialization::Active : SharedPredictorMaterialization::Inactive;
    return config;
}
}
