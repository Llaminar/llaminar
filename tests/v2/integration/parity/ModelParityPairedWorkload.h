/**
 * @file ModelParityPairedWorkload.h
 * @brief Source-bound document obligations pairing diagnostic accuracy and timing.
 *
 * These small metadata references neither grant topology eligibility nor prove
 * approval, accuracy, prerequisites or performance. Consumers authenticate the
 * referenced documents before collection/qualification. Original matrix cases
 * remain the only model, topology, arithmetic and placement authority.
 */
#pragma once
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include "config/GDNProjectionArithmeticPolicy.h"
#include "config/MoERouterArithmeticPolicy.h"
#include "kernels/common/QuantizedActivationOperandPrecision.h"

namespace llaminar2::test::parity
{
/** @brief Required coverage or explicitly expanded coverage; neither is a pass. */
enum class ModelParityPairRole { Required, Expanded };

/** @brief Exact small-document locator and content identity, never a model hash. */
struct ModelParityDocumentReference
{
    std::string path;
    std::string sha256;

    /** @brief Reject incomplete metadata without opening files or admitting memory. */
    void validate() const
    {
        if (path.empty() || path.find('\0') != std::string::npos ||
            sha256.size() != 64 || !std::all_of(sha256.begin(), sha256.end(),
                [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
            throw std::invalid_argument("paired workload requires document path and lowercase SHA256");
    }
    friend bool operator==(const ModelParityDocumentReference &,
                           const ModelParityDocumentReference &) = default;
};

/**
 * @brief Initial absolute-pilot document obligation without an approved baseline.
 *
 * This branch declares the exact pilot document; it neither certifies the whole
 * corpus nor supplies baseline, calibration, image or CI eligibility.
 */
struct ModelParityInitialAbsolutePilot
{
    ModelParityDocumentReference pilot;

    /** @brief Reject an absent/incomplete pilot locator before case expansion. */
    void validate() const { pilot.validate(); }
    friend bool operator==(const ModelParityInitialAbsolutePilot &,
                           const ModelParityInitialAbsolutePilot &) = default;
};

/**
 * @brief Calibrated-revision obligations including the actual baseline locator.
 *
 * Approval/calibration digest bindings cannot locate their baseline report.
 * Retaining its own reference prevents directory search or invented self-baselines.
 * These are required document references, not evidence that their contents pass.
 */
struct ModelParityCalibratedRevision
{
    ModelParityDocumentReference baseline;
    ModelParityDocumentReference approval;
    ModelParityDocumentReference calibration;

    /** @brief Require all three pinned documents without opening model/files. */
    void validate() const { baseline.validate(); approval.validate(); calibration.validate(); }
    friend bool operator==(const ModelParityCalibratedRevision &,
                           const ModelParityCalibratedRevision &) = default;
};

/** @brief Exclusive document policies; no contradictory optional approval flags. */
using ModelParityAccuracyPolicy =
    std::variant<ModelParityCalibratedRevision, ModelParityInitialAbsolutePilot>;

/**
 * @brief Explicit candidate arithmetic owned by this source profile, never the corpus environment.
 *
 * Absence does not imply installed defaults or a candidate selection. A QFN
 * paired consumer requires this complete value; non-QFN documents stay unchanged.
 */
struct ModelParityQFNArithmetic
{
    GDNProjectionArithmetic gdn_projection;
    MoERouterArithmetic moe_router;
    QuantizedActivationOperandPrecision hc_up, gdn_output, fa_output, gdn_qkv;

    /**
     * @brief Require every candidate selector explicitly; no default candidate is minted.
     * @param gdn Original GDN preparation/operand policy.
     * @param router Concrete original router preparation policy.
     * @param hc Original hyperconnection-up operand policy.
     * @param output Original GDN output operand policy.
     * @param attention Original full-attention output operand policy.
     * @param qkv Original GDN QKV operand policy.
     */
    ModelParityQFNArithmetic(GDNProjectionArithmetic gdn, MoERouterArithmetic router,
        QuantizedActivationOperandPrecision hc, QuantizedActivationOperandPrecision output,
        QuantizedActivationOperandPrecision attention, QuantizedActivationOperandPrecision qkv)
        : gdn_projection(gdn), moe_router(router), hc_up(hc), gdn_output(output), fa_output(attention), gdn_qkv(qkv) {}

    /** @brief Reject unknown or unresolved selectors before expansion/export. */
    void validate() const
    {
        (void)gdnProjectionArithmeticName(gdn_projection);
        if (moe_router == MoERouterArithmetic::ModelDefault)
            throw std::invalid_argument("paired QFN arithmetic needs an explicit resolved router policy");
        (void)moeRouterArithmeticName(moe_router);
        for (auto value : {hc_up, gdn_output, fa_output, gdn_qkv})
            (void)quantizedActivationOperandPrecisionName(value);
    }
    friend bool operator==(const ModelParityQFNArithmetic &, const ModelParityQFNArithmetic &) = default;
};

/**
 * @brief Reviewed references for both projections of each original expanded case.
 *
 * Source locators bind the review to the original main/optional predictor. They
 * never select a second model: validation compares them with the containing
 * case, and export derives source fields from that case. Every document is
 * mandatory even for Expanded coverage. The typed policy requires either the
 * initial pilot document or the complete calibrated revision; neither branch
 * invents numerical approval. Unprofiled definitions remain unchanged.
 */
struct ModelParityPairedWorkload
{
    ModelParityPairRole role = ModelParityPairRole::Required;
    std::string reviewed_main_source;
    std::optional<std::string> reviewed_predictor_source;
    ModelParityDocumentReference accuracy_catalog;
    ModelParityDocumentReference accuracy_workload;
    ModelParityAccuracyPolicy accuracy_policy;
    ModelParityDocumentReference benchmark_profile;
    std::optional<ModelParityQFNArithmetic> qfn_arithmetic; ///< Explicit source-owned candidate; no case/default mutation.

    /**
     * @brief Validate complete references against the sole original source roles.
     * @param main_source Containing definition/case original GGUF locator.
     * @param predictor_source Its explicit paired predictor, including MTP-off.
     * @throws std::invalid_argument For unknown role, incomplete or foreign review.
     */
    void validate(std::string_view main_source,
                  const std::optional<std::string> &predictor_source) const
    {
        if (role != ModelParityPairRole::Required && role != ModelParityPairRole::Expanded)
            throw std::invalid_argument("invalid paired workload coverage role");
        if (main_source.empty() || reviewed_main_source != main_source ||
            reviewed_main_source.find('\0') != std::string::npos ||
            reviewed_predictor_source != predictor_source ||
            (predictor_source && (predictor_source->empty() ||
                predictor_source->find('\0') != std::string::npos)))
            throw std::invalid_argument("paired workload review names a foreign original source");
        accuracy_catalog.validate();
        accuracy_workload.validate();
        if (accuracy_policy.valueless_by_exception())
            throw std::invalid_argument("paired workload has no accuracy document policy");
        std::visit([](const auto &policy) { policy.validate(); }, accuracy_policy);
        benchmark_profile.validate();
        if (qfn_arithmetic) qfn_arithmetic->validate();
    }
    friend bool operator==(const ModelParityPairedWorkload &,
                           const ModelParityPairedWorkload &) = default;
};
} // namespace llaminar2::test::parity
