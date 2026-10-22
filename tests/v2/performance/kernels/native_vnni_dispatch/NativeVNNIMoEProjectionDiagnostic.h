/**
 * @file NativeVNNIMoEProjectionDiagnostic.h
 * @brief Explicit descriptor-derived diagnostic operation for the MoE trainer.
 *
 * This declaration adds no release inventory or installable overlay key. It
 * borrows a retained metadata-only descriptor document and uses the existing
 * routed source triple and immutable output-column ownership. Original source
 * names remain distinct from physical decoder IDs. Synthetic seeded weights
 * are diagnostic inputs, never an authenticated model-weight corpus.
 */
#pragma once

#include "NativeVNNIMoEPrefillManifest.h"
#include "NativeVNNIMoERoutingProfiles.h"
#include "execution/moe/MoEExpertProjectionOwnership.h"

#include <limits>


namespace llaminar2::test::native_vnni_dispatch
{
/** @brief One M1 owner gate/up contest and separately named fixed-down control. */
struct MoEProjectionDiagnostic
{
    MoERoutedPrefillCase source;
    int layer = -1;
    int rows = 1;
    int participant = -1;
    int top_k = 0;
    std::vector<int> device_ordinals;
    MoERoutingProfile route_profile{};
    int warmups = 31;
    int samples = 101;
    int operations_per_sample = 1;
    std::string descriptor_document;
    nlohmann::json descriptor_evidence;

    /** @return Original full-K ownership; device ordinal is a separate fact. */
    [[nodiscard]] MoEExpertProjectionOwnership ownership() const
    {
        return MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
            {source.expert_count, source.hidden_size, source.routed_expert_width},
            participant, static_cast<int>(device_ordinals.size()));
    }

    /** @return Complete diagnostic identity; never a complete-operation corpus ID. */
    [[nodiscard]] std::string evidenceId() const
    {
        const auto down = ownership().projection(WeightRole::MoEExpertDown);
        return source.evidenceId() + "_descriptor_layer" + std::to_string(layer) + "_m1_owner_ordinal_p" + std::to_string(participant) +
            "of" + std::to_string(device_ordinals.size()) + "_down" +
            std::to_string(down.first_row) + "x" + std::to_string(down.rows);
    }
};

/**
 * @brief Validate one explicit metadata-only diagnostic request before GPU setup.
 * @param request Authored diagnostic record; no inherited legacy pair settings.
 * @param descriptors Existing bounded descriptor document, never tensor data.
 * @return Typed original geometry/source and exact local physical interval.
 * @throws std::invalid_argument for unknown/oversized/inconsistent declarations.
 * @note Top-k is explicit because tensor descriptors do not encode router policy.
 */
inline MoEProjectionDiagnostic parseMoEProjectionDiagnostic(
    const nlohmann::json &request, const nlohmann::json &descriptors)
{
    const std::set<std::string> allowed{
        "schema", "operation", "owner_map", "provenance", "m", "layer", "participant",
        "top_k", "device_ordinals", "route_profile", "warmups", "samples",
        "operations_per_sample", "descriptor_document"};
    if (!request.is_object())
        throw std::invalid_argument("MoE projection diagnostic request must be an object");
    for (const auto &[name, value] : request.items())
        if (!allowed.contains(name))
            throw std::invalid_argument("unknown MoE projection diagnostic field: " + name);
    const auto integer = [](const nlohmann::json &value, const std::string &name) -> int
    {
        if (!value.is_number_integer() ||
            (value.is_number_unsigned() && value.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<int>::max())))
            throw std::invalid_argument("MoE projection field must be an exact bounded integer: " + name);
        const auto result = value.get<std::int64_t>();
        if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max())
            throw std::invalid_argument("MoE projection integer overflows: " + name);
        return static_cast<int>(result);
    };
    const auto optional_integer = [&](const char *name, int default_value) {
        return request.contains(name) ? integer(request.at(name), name) : default_value;
    };
    const auto integers = [&](const nlohmann::json &values, const std::string &name) {
        if (!values.is_array()) throw std::invalid_argument("MoE projection field must be an array: " + name);
        std::vector<int> result;
        for (const auto &value : values) result.push_back(integer(value, name));
        return result;
    };
    if (request.value("schema", "") != "native-vnni-moe-projection-diagnostic-v1" ||
        request.value("operation", "") != "owner-gate-up-pack-m1" ||
        request.value("owner_map", "") != "ordinal" ||
        request.value("provenance", "") != "retained-descriptors-synthetic-seeded" ||
        integer(request.at("m"), "m") != 1)
        throw std::invalid_argument("unknown MoE projection diagnostic operation/provenance");
    MoEProjectionDiagnostic result;
    result.layer = integer(request.at("layer"), "layer");
    result.participant = integer(request.at("participant"), "participant");
    result.top_k = integer(request.at("top_k"), "top_k");
    result.device_ordinals = integers(request.at("device_ordinals"), "device_ordinals");
    result.route_profile = parseMoERoutingProfile(request.at("route_profile").get<std::string>());
    result.warmups = optional_integer("warmups", 31);
    result.samples = optional_integer("samples", 101);
    result.operations_per_sample = optional_integer("operations_per_sample", 1);
    result.descriptor_document = request.at("descriptor_document").get<std::string>();
    const auto degree = result.device_ordinals.size();
    if (result.layer < 0 || degree < 2 || degree > 8 || result.participant < 0 ||
        static_cast<std::size_t>(result.participant) >= degree ||
        result.top_k <= 0 || result.top_k > 16 || result.warmups < 0 || result.warmups > 10001 ||
        result.samples < 3 || result.samples > 10001 ||
        (result.operations_per_sample != 1 && result.operations_per_sample != 32) ||
        result.descriptor_document.empty())
        throw std::invalid_argument("invalid MoE projection diagnostic bounds");
    std::set<int> devices;
    for (int ordinal : result.device_ordinals)
        if (ordinal < 0 || !devices.insert(ordinal).second)
            throw std::invalid_argument("MoE projection diagnostic has invalid native membership");
    std::map<std::string, nlohmann::json> roles;
    const std::string prefix = "blk." + std::to_string(result.layer) + ".ffn_";
    if (!descriptors.is_object() || !descriptors.contains("files") || !descriptors.at("files").is_array())
        throw std::invalid_argument("MoE projection diagnostic requires retained tensor descriptors");
    for (const auto &file : descriptors.at("files"))
        for (const auto &tensor : file.at("tensors"))
            for (const std::string role : {"gate", "up", "down"})
                if (tensor.value("name", "") == prefix + role + "_exps.weight")
                {
                    auto evidence = tensor;
                    evidence["source_file"] = file.at("path");
                    evidence["source_stat"] = file.at("stat");
                    evidence["header_data_offset"] = file.at("header_data_offset");
                    if (!roles.emplace(role, std::move(evidence)).second)
                        throw std::invalid_argument("duplicate MoE source descriptor role");
                }
    if (roles.size() != 3)
        throw std::invalid_argument("MoE diagnostic layer lacks gate/up/down source descriptors");
    const auto gate = integers(roles.at("gate").at("dimensions"), "gate.dimensions");
    const auto up = integers(roles.at("up").at("dimensions"), "up.dimensions");
    const auto down = integers(roles.at("down").at("dimensions"), "down.dimensions");
    if (gate.size() != 3 || up != gate || down.size() != 3 ||
        gate[0] <= 0 || gate[1] <= 0 || gate[0] > (1 << 20) || gate[1] > (1 << 20) ||
        gate[2] < result.top_k ||
        gate[2] > 1024 || down != std::vector<int>{gate[1], gate[0], gate[2]} ||
        gate[0] % 32 || gate[1] % 32 || gate[0] % degree)
        throw std::invalid_argument("inconsistent MoE original geometry or equal down interval");
    result.source = {gate[0], gate[1], gate[2], result.top_k,
        {roles.at("gate").at("format_label").get<std::string>(),
         roles.at("up").at("format_label").get<std::string>(),
         roles.at("down").at("format_label").get<std::string>()}, {}};
    if (result.source.routed.gate != result.source.routed.up ||
        !detail::isNativeVnniTriple(result.source.routed))
        throw std::invalid_argument("MoE diagnostic requires a supported original native gate/up pair");
    result.descriptor_evidence = roles;
    (void)result.ownership();
    return result;
}

/**
 * @brief Load only two small explicit metadata documents for the native branch.
 * @param path Existing trainer's explicit diagnostic request path.
 * @return Validated M1 operation; source-file paths are evidence only, never opened.
 */
inline MoEProjectionDiagnostic loadMoEProjectionDiagnostic(const std::string &path)
{
    const auto request = detail::readJsonDocument(path, "MoE projection diagnostic request");
    return parseMoEProjectionDiagnostic(request, detail::readJsonDocument(
        request.at("descriptor_document").get<std::string>(), "retained MoE tensor descriptors"));
}
} // namespace llaminar2::test::native_vnni_dispatch
