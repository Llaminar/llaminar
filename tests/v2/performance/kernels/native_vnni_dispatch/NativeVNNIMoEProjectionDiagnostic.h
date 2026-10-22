/**
 * @file NativeVNNIMoEProjectionDiagnostic.h
 * @brief Explicit original-source diagnostic operations for the canonical MoE trainer.
 *
 * This declaration adds no release inventory or installable overlay key. It
 * borrows a retained metadata-only descriptor document and uses the existing
 * routed source triple and immutable output-column ownership. Original source
 * names remain distinct from physical decoder IDs. Synthetic seeded weights
 * are diagnostic inputs, never an authenticated model-weight corpus. Explicit
 * source fixtures use the canonical source registry and normal preparation;
 * they carry no fictitious GGUF descriptor, layer, file or model identity.
 * An explicitly tagged promoted fixture retains that original source identity
 * while requiring genuine expanded asymmetric CPU preparation and transfer.
 */
#pragma once

#include "NativeVNNIMoEPrefillManifest.h"
#include "NativeVNNIMoERoutingProfiles.h"
#include "execution/moe/MoEExpertProjectionOwnership.h"

#include "tensors/NativeVnniFormatInfo.h"

#include <cstdint>
#include <limits>


namespace llaminar2::test::native_vnni_dispatch
{
/** @brief Authored physical contest; partition execution never changes the original source operation. */
enum class MoEProjectionContest : std::uint8_t
{
    GateUpGeometry, ///< Existing inline TN64/128/256 contest.
    Inline64VsParallelPartitions64, ///< Exact source partition work sharing at fixed TN64/K_LANES1.
    AutomaticVsParallelPartitions64 ///< Actual unscoped production decoder versus four worker planes.
};

/**
 * @brief Resolve a closed request contest without a device or source payload.
 * @param contest Authored partition execution/geometry contest.
 * @return Stable diagnostic identity, distinct from installed policy provenance.
 * @throws std::invalid_argument For an unknown enum; no default substitution.
 */
[[nodiscard]] constexpr const char *moeProjectionContestName(MoEProjectionContest contest)
{
    switch (contest)
    {
    case MoEProjectionContest::GateUpGeometry: return "gate-up-geometry";
    case MoEProjectionContest::Inline64VsParallelPartitions64: return "inline64-vs-parallel-partitions64";
    case MoEProjectionContest::AutomaticVsParallelPartitions64: return "automatic-vs-parallel-partitions64";
    }
    throw std::invalid_argument("unknown MoE projection diagnostic contest");
}

/** @brief Original metadata authority; synthetic fixtures never impersonate a model descriptor. */
enum class MoEProjectionSourceKind : std::uint8_t
{
    RetainedDescriptors, ///< Existing retained model tensor metadata, seeded diagnostic weights.
    SyntheticSourceFixture ///< Explicit canonical source-format fixture, no model or GGUF identity.
};

/** @brief Actual prepared representation; original source labels never name promoted execution codebooks. */
enum class MoEProjectionPreparedRepresentation : std::uint8_t
{
    OriginalGpu, ///< Existing source-native GPU preparation and independent oracle.
    CpuPreparedPromoted23 ///< Genuine expanded asymmetric CPU preparation followed by ordinary transfer-lane promotion.
};

/**
 * @param representation Closed immutable preparation declaration.
 * @return Stable request/evidence name, orthogonal to original source and physical contest.
 * @throws std::invalid_argument For an unknown enum; no implicit representation substitution.
 */
[[nodiscard]] constexpr const char *moeProjectionPreparedRepresentationName(MoEProjectionPreparedRepresentation representation)
{
    switch (representation)
    {
    case MoEProjectionPreparedRepresentation::OriginalGpu: return "original-gpu-prepared";
    case MoEProjectionPreparedRepresentation::CpuPreparedPromoted23: return "cpu-prepared-promoted23";
    }
    throw std::invalid_argument("unknown MoE diagnostic prepared representation");
}

/**
 * @param original_format Canonical original source alias, never an authored physical23 label.
 * @return Whether canonical CPU promotion yields actual expanded asymmetric23 for this original source.
 */
[[nodiscard]] inline bool moeProjectionSupportsPromoted23(std::string_view original_format) noexcept
{
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        if (source.quant_type == original_format)
            return source.metadata->is_asymmetric &&
                migrationStableDeviceVnniFormat(*source.metadata).codebook_id == kNativeVnniExpandedInt8MinCodebook;
    return false;
}

/** @brief One M1 owner gate/up contest and separately named fixed-down control. */
struct MoEProjectionDiagnostic
{
    MoERoutedPrefillCase source;
    MoEProjectionSourceKind source_kind = MoEProjectionSourceKind::RetainedDescriptors;
    MoEProjectionPreparedRepresentation prepared_representation = MoEProjectionPreparedRepresentation::OriginalGpu; ///< Immutable explicit source-to-prepared contract.
    int layer = -1;
    int rows = 1;
    int participant = -1;
    int top_k = 0;
    std::vector<int> device_ordinals;
    MoERoutingProfile route_profile{};
    int warmups = 31;
    int samples = 101;
    int operations_per_sample = 1;
    MoEProjectionContest contest = MoEProjectionContest::GateUpGeometry; ///< Omission retains the existing inline geometry contest.
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
        return source.evidenceId() + (source_kind == MoEProjectionSourceKind::SyntheticSourceFixture ?
                "_synthetic_source_fixture" : "_descriptor_layer" + std::to_string(layer)) + "_m1_owner_ordinal_p" + std::to_string(participant) +
            "of" + std::to_string(device_ordinals.size()) + "_down" +
            std::to_string(down.first_row) + "x" + std::to_string(down.rows) +
            (prepared_representation == MoEProjectionPreparedRepresentation::OriginalGpu ? std::string{} :
                "_prepared_" + std::string(moeProjectionPreparedRepresentationName(prepared_representation))) +
            (contest == MoEProjectionContest::GateUpGeometry ? std::string{} :
                "_contest_" + std::string(moeProjectionContestName(contest)));
    }
};

/**
 * @brief Validate one explicit metadata-only diagnostic request before GPU setup.
 * @param request Authored diagnostic record; no inherited legacy pair settings.
 * @param descriptors Existing bounded descriptor document; ignored by explicitly tagged source fixtures.
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
        "operations_per_sample", "descriptor_document", "contest", "source_fixture", "prepared_representation"};
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
        integer(request.at("m"), "m") != 1)
        throw std::invalid_argument("unknown MoE projection diagnostic operation/provenance");
    MoEProjectionDiagnostic result;
    const auto provenance = request.value("provenance", "");
    if (provenance == "synthetic-source-fixture")
    {
        result.source_kind = MoEProjectionSourceKind::SyntheticSourceFixture;
        if (request.contains("layer") || request.contains("descriptor_document") || !request.contains("source_fixture"))
            throw std::invalid_argument("synthetic MoE fixture must not claim model descriptor/layer identity");
    }
    else if (provenance == "retained-descriptors-synthetic-seeded")
    {
        if (request.contains("source_fixture"))
            throw std::invalid_argument("retained MoE descriptor request cannot mix synthetic source authority");
        result.layer = integer(request.at("layer"), "layer");
        result.descriptor_document = request.at("descriptor_document").get<std::string>();
    }
    else throw std::invalid_argument("unknown MoE projection source provenance");
    result.participant = integer(request.at("participant"), "participant");
    result.top_k = integer(request.at("top_k"), "top_k");
    result.device_ordinals = integers(request.at("device_ordinals"), "device_ordinals");
    result.route_profile = parseMoERoutingProfile(request.at("route_profile").get<std::string>());
    result.warmups = optional_integer("warmups", 31);
    result.samples = optional_integer("samples", 101);
    result.operations_per_sample = optional_integer("operations_per_sample", 1);
    if (request.contains("contest"))
    {
        if (!request.at("contest").is_string())
            throw std::invalid_argument("MoE projection contest must be an explicit string");
        const auto &name = request.at("contest").get_ref<const std::string &>();
        if (name == "gate-up-geometry") result.contest = MoEProjectionContest::GateUpGeometry;
        else if (name == "inline64-vs-parallel-partitions64") result.contest = MoEProjectionContest::Inline64VsParallelPartitions64;
        else if (name == "automatic-vs-parallel-partitions64") result.contest = MoEProjectionContest::AutomaticVsParallelPartitions64;
        else throw std::invalid_argument("unknown MoE projection diagnostic contest: " + name);
    }
    if (request.contains("prepared_representation"))
    {
        if (!request.at("prepared_representation").is_string())
            throw std::invalid_argument("MoE prepared representation must be an explicit string");
        const auto &name = request.at("prepared_representation").get_ref<const std::string &>();
        if (name == "original-gpu-prepared") result.prepared_representation = MoEProjectionPreparedRepresentation::OriginalGpu;
        else if (name == "cpu-prepared-promoted23") result.prepared_representation = MoEProjectionPreparedRepresentation::CpuPreparedPromoted23;
        else throw std::invalid_argument("unknown MoE prepared representation: " + name);
    }
    if (result.prepared_representation == MoEProjectionPreparedRepresentation::CpuPreparedPromoted23 &&
        (result.source_kind != MoEProjectionSourceKind::SyntheticSourceFixture ||
         result.contest != MoEProjectionContest::AutomaticVsParallelPartitions64))
        throw std::invalid_argument("promoted23 is an explicit synthetic automatic-versus-worker fixture only");
    const auto degree = result.device_ordinals.size();
    if ((result.source_kind == MoEProjectionSourceKind::RetainedDescriptors &&
            (result.layer < 0 || result.descriptor_document.empty())) || degree < 2 || degree > 8 || result.participant < 0 ||
        static_cast<std::size_t>(result.participant) >= degree ||
        result.top_k <= 0 || result.top_k > 16 || result.warmups < 0 || result.warmups > 10001 ||
        result.samples < 3 || result.samples > 10001 ||
        (result.operations_per_sample != 1 && result.operations_per_sample != 32))
        throw std::invalid_argument("invalid MoE projection diagnostic bounds");
    std::set<int> devices;
    for (int ordinal : result.device_ordinals)
        if (ordinal < 0 || !devices.insert(ordinal).second)
            throw std::invalid_argument("MoE projection diagnostic has invalid native membership");
    if (result.source_kind == MoEProjectionSourceKind::SyntheticSourceFixture)
    {
        const auto &fixture = request.at("source_fixture");
        const std::set<std::string> keys{"hidden_size", "expert_width", "expert_count", "formats"};
        if (!fixture.is_object() || fixture.size() != keys.size())
            throw std::invalid_argument("synthetic MoE source fixture needs exactly geometry and original formats");
        for (const auto &[key, value] : fixture.items())
            if (!keys.contains(key)) throw std::invalid_argument("unknown synthetic MoE source field: " + key);
        const int h = integer(fixture.at("hidden_size"), "source_fixture.hidden_size");
        const int f = integer(fixture.at("expert_width"), "source_fixture.expert_width");
        const int e = integer(fixture.at("expert_count"), "source_fixture.expert_count");
        const auto &formats = fixture.at("formats");
        if (!formats.is_object() || formats.size() != 3 || !formats.contains("gate") ||
            !formats.contains("up") || !formats.contains("down"))
            throw std::invalid_argument("synthetic MoE source requires exactly gate/up/down formats");
        const MoEPrefillFormatTriple triple{formats.at("gate").get<std::string>(),
            formats.at("up").get<std::string>(), formats.at("down").get<std::string>()};
        if (h <= 0 || f <= 0 || h > (1 << 20) || f > (1 << 20) || h % 32 || f % 32 ||
            h % degree || e < result.top_k || e > 1024 || e < static_cast<int>(degree) ||
            triple.gate != triple.up || !detail::isNativeVnniTriple(triple))
            throw std::invalid_argument("invalid synthetic MoE source geometry or native format triple");
        if (result.prepared_representation == MoEProjectionPreparedRepresentation::CpuPreparedPromoted23 &&
            (!moeProjectionSupportsPromoted23(triple.gate) || f % 64))
            throw std::invalid_argument("promoted23 requires a canonical asymmetric source and complete live64-column CPU stream units");
        nlohmann::json identities = nlohmann::json::object();
        for (const std::string role : {"gate", "up", "down"})
        {
            const std::string label = formats.at(role).get<std::string>();
            const NativeVnniFormatInfo *metadata = nullptr;
            for (const auto &entry : native_vnni_formats::kAllSourceFormats)
                if (entry.quant_type == label) metadata = entry.metadata;
            if (!metadata || ((role == "down" ? f : h) % (metadata->is_superblock ? 256 : 32)))
                throw std::invalid_argument("synthetic MoE source is absent or misaligned in the canonical registry");
            identities[role] = {{"source_format", label}, {"source_codebook", metadata->codebook_id},
                {"source_is_superblock", metadata->is_superblock},
                {"canonical_native_payload_bytes_per_k32", metadata->payload_bytes},
                {"canonical_execution_codebook", canonicalDeviceVnniFormat(*metadata).codebook_id},
                {"migration_execution_codebook", migrationStableDeviceVnniFormat(*metadata).codebook_id},
                {"original_n", role == "down" ? h : f}, {"original_k", role == "down" ? f : h}};
        }
        result.source = {h, f, e, result.top_k, triple, {}};
        result.descriptor_evidence = {{"authority", "canonical-native-vnni-source-fixture"},
            {"model_identity", nullptr}, {"original_source_roles", identities},
            {"expert_count", e}, {"normal_preparation", true}};
        if (result.prepared_representation == MoEProjectionPreparedRepresentation::CpuPreparedPromoted23)
            result.descriptor_evidence["prepared_representation"] = moeProjectionPreparedRepresentationName(result.prepared_representation);
        (void)result.ownership();
        return result;
    }
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
 * @brief Load the explicit request and its tagged metadata authority for the native branch.
 * @param path Existing trainer's explicit diagnostic request path.
 * @return Validated M1 operation; synthetic fixtures open no descriptor document and source-file paths remain evidence only.
 */
inline MoEProjectionDiagnostic loadMoEProjectionDiagnostic(const std::string &path)
{
    const auto request = detail::readJsonDocument(path, "MoE projection diagnostic request");
    if (request.value("provenance", "") == "synthetic-source-fixture")
        return parseMoEProjectionDiagnostic(request, nlohmann::json::object());
    return parseMoEProjectionDiagnostic(request, detail::readJsonDocument(
        request.at("descriptor_document").get<std::string>(), "retained MoE tensor descriptors"));
}
} // namespace llaminar2::test::native_vnni_dispatch
