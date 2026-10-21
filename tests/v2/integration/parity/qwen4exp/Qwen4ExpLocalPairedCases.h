/**
 * @file Qwen4ExpLocalPairedCases.h
 * @brief Opt-in build-frozen paired documents on exact original QFN cases.
 *
 * Ordinary cases remain the sole topology/storage producer. The optional local
 * document attaches candidate obligations and an explicitly declared inactive
 * predictor to exact existing IDs; it never expands axes or changes arithmetic
 * defaults. All requested IDs must be consumed before any family is published.
 * Local source stat identities and artifact locators are not image eligibility.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"
#include "Qwen4ExpFixedHeadParityCases.h"
#include "Qwen4ExpGPUAdaptiveParityCases.h"
#include "Qwen4ExpFullROCmTP4ParityCases.h"
#include <nlohmann/json.hpp>
#include <map>
#include <set>
#include <stdexcept>
#ifdef LLAMINAR_QFN_LOCAL_PAIRED_BINDINGS
#include "QFNLocalPairedBindings.generated.h"
#endif

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Existing suite families, not another topology or policy expander. */
using QFNCaseFamilies = std::map<std::string, std::vector<ModelParityCase>>;

/** @return Original case producers in their unchanged registration families. */
inline QFNCaseFamilies qfnOriginalCaseFamilies()
{
    return {{"Qwen4ExpMain", qwen4ExpProductionCases()},
            {"Qwen4ExpHostAdaptiveRAMSSD", qwen4ExpHostAdaptiveProductionCases()},
            {"Qwen4ExpFixedHeadTP", qwen4ExpFixedHeadProductionCases()},
            {"Qwen4ExpGPUAdaptive", qwen4ExpGPUAdaptiveProductionCases()},
            {"Qwen4ExpFullROCmTP4", qwen4ExpFullROCmTP4ProductionCases()}};
}

/** @brief Reject malformed local metadata before a case can enter registration. */
inline void qfnLocalRequire(bool value, const char *why)
{
    if (!value) throw std::invalid_argument(why);
}

/**
 * @brief Parse one complete original document reference through its canonical validator.
 * @param value Build-frozen small metadata object; no files are opened here.
 * @return Exact path and digest obligation, never an approval.
 */
inline ModelParityDocumentReference qfnLocalDocument(const nlohmann::json &value)
{
    qfnLocalRequire(value.is_object() && value.size() == 2 && value.contains("path") && value.contains("sha256"),
                    "Local QFN profile has an incomplete document reference");
    ModelParityDocumentReference result{value.at("path").get<std::string>(), value.at("sha256").get<std::string>()};
    result.validate();
    return result;
}

/**
 * @brief Attach frozen candidate obligations to exact original cases transactionally.
 * @param original Complete actual source families; every untouched case is preserved.
 * @param document Explicit configured local document; no environment lookup or registry.
 * @return Owned metadata copies, after every binding has matched exactly one source ID.
 * @throws std::invalid_argument For unknown/unused/duplicate IDs, foreign source roles or invalid policies.
 * @note No source bytes, model, prepared tensor or backend is accessed by this operation.
 */
inline QFNCaseFamilies qfnAttachLocalPairedCases(QFNCaseFamilies original, const nlohmann::json &document)
{
    qfnLocalRequire(document.is_object() && document.size() == 3 &&
        document.at("schema") == "llaminar.qfn-local-paired-profiles.v1" &&
        document.contains("source") && document.contains("bindings"), "Invalid local QFN paired document");
    const auto &source = document.at("source");
    qfnLocalRequire(source.at("architecture") == "qwen4exp" && source.at("main_layers") == 48 &&
        source.at("expert_count") == 512 && source.at("shards").is_array() && source.at("shards").size() == 4 &&
        source.at("shards").at(0).at("path") == kMainModelPath &&
        source.contains("shared_predictor") && source.at("shared_predictor").at("path") == kPredictorModelPath,
        "Local QFN binding names a foreign original model/predictor source");
    const auto &bindings = document.at("bindings");
    qfnLocalRequire(bindings.is_array() && !bindings.empty() && bindings.size() <= 128,
                    "Local QFN profile requires bounded explicit case bindings");
    std::map<std::string, ModelParityCase *> available;
    for (auto &[family, cases] : original)
        for (auto &cell : cases)
            qfnLocalRequire(available.emplace(family + "/Qwen4ExpMainParityTest.ProductionParity/" + cell.testName(), &cell).second,
                            "Original QFN registration repeats an exact source ID");
    std::set<std::string> seen;
    for (const auto &binding : bindings)
    {
        qfnLocalRequire(binding.is_object() && binding.size() == 7,
                        "Local QFN binding has missing or foreign fields");
        const auto exact = binding.at("exact_cell").get<std::string>();
        const auto found = available.find(exact);
        qfnLocalRequire(found != available.end() && seen.insert(exact).second,
                        "Local QFN binding has an unknown, unused or duplicate exact source case");
        auto &cell = *found->second;
        qfnLocalRequire(cell.model.model_path == kMainModelPath && cell.mtp == ModelParityMTP::Off &&
            cell.activation_precision == ActivationPrecision::FP32 && cell.kv_cache_precision == KVCachePrecision::FP16 &&
            !cell.paired_workload, "Local QFN binding cannot replace an existing profile or change ordinary source/math");
        if (cell.model.shared_predictor)
            qfnLocalRequire(cell.model.shared_predictor->path() == kPredictorModelPath,
                            "Local QFN binding cannot replace an original predictor");
        // This explicit opt-in source declaration is inactive with MTP-off.
        // No ordinary case producer or production default is modified.
        cell.model.shared_predictor = SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
        const auto role = binding.at("role").get<std::string>();
        qfnLocalRequire(role == "required" || role == "expanded", "Unknown local QFN coverage role");
        const auto &a = binding.at("arithmetic");
        qfnLocalRequire(a.is_object() && a.size() == 6, "Local QFN candidate requires all six explicit selectors");
        ModelParityPairedWorkload profile;
        profile.role = role == "required" ? ModelParityPairRole::Required : ModelParityPairRole::Expanded;
        profile.reviewed_main_source = cell.model.model_path;
        profile.reviewed_predictor_source = std::string(kPredictorModelPath);
        profile.accuracy_catalog = qfnLocalDocument(binding.at("catalog"));
        profile.accuracy_workload = qfnLocalDocument(binding.at("workload"));
        profile.accuracy_policy = ModelParityInitialAbsolutePilot{qfnLocalDocument(binding.at("pilot"))};
        profile.benchmark_profile = qfnLocalDocument(binding.at("benchmark"));
        profile.qfn_arithmetic = ModelParityQFNArithmetic{
            parseGDNProjectionArithmetic(a.at("gdn_projection").get<std::string>()),
            parseMoERouterArithmetic(a.at("moe_router").get<std::string>()),
            parseQuantizedActivationOperandPrecision(a.at("hc_up").get<std::string>()),
            parseQuantizedActivationOperandPrecision(a.at("gdn_output").get<std::string>()),
            parseQuantizedActivationOperandPrecision(a.at("fa_output").get<std::string>()),
            parseQuantizedActivationOperandPrecision(a.at("gdn_qkv").get<std::string>())};
        profile.validate(cell.model.model_path, std::optional<std::string>(cell.model.shared_predictor->path()));
        cell.paired_workload = std::move(profile);
    }
    return original;
}

/**
 * @brief Read one existing registration family, applying opt-in metadata only in configured builds.
 * @param family Original suite prefix used by its unchanged GTest registration.
 * @param ordinary Original source vector, returned directly when configuration is absent.
 * @return Original vector or the completely validated local metadata projection.
 * @throws std::invalid_argument Before discovery if any configured case is unused or malformed.
 */
inline const std::vector<ModelParityCase> &qfnLocalPairedFamily(const std::string &family,
    const std::vector<ModelParityCase> &ordinary)
{
#ifdef LLAMINAR_QFN_LOCAL_PAIRED_BINDINGS
    (void)ordinary;
    static const auto selected = [] {
        const auto frozen = nlohmann::json::parse(local_paired_build::input);
        qfnLocalRequire(frozen.is_object() && frozen.size() == 4 &&
            frozen.contains("binding_path") && frozen.contains("binding_sha256") &&
            frozen.contains("source_sha256") && frozen.contains("document") &&
            !frozen.at("binding_path").get<std::string>().empty() &&
            frozen.at("binding_sha256").get<std::string>().size() == 64 &&
            frozen.at("source_sha256").get<std::string>().size() == 64,
            "Incomplete build-frozen local QFN input identity");
        return qfnAttachLocalPairedCases(qfnOriginalCaseFamilies(),frozen.at("document"));
    }();
    const auto found = selected.find(family);
    qfnLocalRequire(found != selected.end(), "Unknown original QFN registration family");
    return found->second;
#else
    (void)family;
    return ordinary;
#endif
}
} // namespace llaminar2::test::parity::qwen4exp
