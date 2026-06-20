#pragma once

#include "execution/config/RuntimeConfig.h"
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMoERuntimeTable;

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

    struct PrefixFingerprintField
    {
        std::string name;
        std::string value;
    };

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

    struct PrefixCacheFingerprintResult
    {
        bool bypass = false;
        std::string bypass_reason;
        PrefixFingerprintParts parts;
        uint64_t key = 0;
    };

    uint64_t hashPrefixFingerprintFields(
        const std::string &part_name,
        const std::vector<PrefixFingerprintField> &fields);

    PrefixFingerprintParts buildPrefixFingerprintParts(
        const PrefixFingerprintMaterial &material);

    uint64_t combinePrefixFingerprintParts(const PrefixFingerprintParts &parts);

    PrefixCacheFingerprintResult buildPrefixCacheFingerprint(
        const PrefixFingerprintMaterial &material,
        bool model_is_moe,
        PrefixCacheMoEPolicy moe_policy);

    void appendMoEPlacementFingerprintFields(
        std::vector<PrefixFingerprintField> &fields,
        const IMoERuntimeTable &table,
        int layer_count,
        const std::string &scope);

} // namespace llaminar2
