/**
 * @file PrefixCacheFingerprint.h
 * @brief Canonical cache keys bound to one request-time placement observation.
 *
 * Fingerprints authenticate model, topology and payload compatibility. The
 * returned epoch is the observation used to build the key, not a live mirror
 * of expert placement; callers retain both through lookup and harvest.
 */
#pragma once

#include "execution/config/RuntimeConfig.h"
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMoERuntimeTable;

    /** @brief Independently hashed compatibility categories of one cache key. */
    struct PrefixFingerprintParts
    {
        uint64_t model = 0;
        uint64_t tokenizer = 0;
        uint64_t runtime = 0;
        uint64_t topology = 0;
        uint64_t hybrid = 0;
        uint64_t moe = 0;
        uint64_t mtp = 0;
    };

    /** @brief Named, serialized model property with deterministic ordering. */
    struct PrefixFingerprintField
    {
        std::string name;
        std::string value;
    };

    /** @brief Compatibility inputs contributed by the model and execution layout. */
    struct PrefixFingerprintMaterial
    {
        std::vector<PrefixFingerprintField> model;
        std::vector<PrefixFingerprintField> tokenizer;
        std::vector<PrefixFingerprintField> runtime;
        std::vector<PrefixFingerprintField> topology;
        std::vector<PrefixFingerprintField> hybrid;
        std::vector<PrefixFingerprintField> moe;
        std::vector<PrefixFingerprintField> mtp;
    };

    /** @brief One fingerprint and the exact placement observation it describes. */
    struct PrefixCacheFingerprintResult
    {
        bool bypass = false;
        std::string bypass_reason;
        PrefixFingerprintParts parts;
        uint64_t key = 0;
        uint64_t placement_epoch = 0; ///< Epoch supplied when constructing this key.
    };

    /**
     * @brief Hash a named category independently of input field order.
     * @param part_name Stable category name, included in the hash.
     * @param fields Named compatibility values.
     * @return Deterministic category hash.
     */
    uint64_t hashPrefixFingerprintFields(
        const std::string &part_name,
        const std::vector<PrefixFingerprintField> &fields);

    /**
     * @brief Hash every compatibility category.
     * @param material Complete named input fields.
     * @return Category hashes for diagnostics and final key construction.
     */
    PrefixFingerprintParts buildPrefixFingerprintParts(
        const PrefixFingerprintMaterial &material);

    /**
     * @brief Join category hashes in fixed order under the cache schema version.
     * @param parts Per-category hashes.
     * @return Stable combined cache key.
     */
    uint64_t combinePrefixFingerprintParts(const PrefixFingerprintParts &parts);

    /**
     * @brief Bind the key and admission epoch using one immutable observation.
     * @param material Model-owned compatibility fields; moved by production.
     * @param model_is_moe Whether MoE compatibility policy applies.
     * @param moe_policy Selected cache compatibility policy.
     * @param placement_epoch Single authority observation for this identity.
     * @return Fingerprint and that same epoch, or the explicit disabled result.
     * @throws std::invalid_argument If material duplicates the owned epoch field.
     *
     * InvalidateOnRebalance incorporates the epoch into the key. Other policies
     * still retain its admission provenance without changing key compatibility.
     */
    PrefixCacheFingerprintResult buildPrefixCacheFingerprint(
        PrefixFingerprintMaterial material,
        bool model_is_moe,
        PrefixCacheMoEPolicy moe_policy,
        uint64_t placement_epoch = 0);

    /**
     * @brief Append published logical placement, excluding histogram telemetry.
     * @param fields Destination compatibility fields.
     * @param table Runtime owner of the published placement banks.
     * @param layer_count Number of logical layers to include.
     * @param scope Stable namespace for this table's fields.
     * @throws std::logic_error If a layer's placement publication is incomplete.
     */
    void appendMoEPlacementFingerprintFields(
        std::vector<PrefixFingerprintField> &fields,
        const IMoERuntimeTable &table,
        int layer_count,
        const std::string &scope);

} // namespace llaminar2
