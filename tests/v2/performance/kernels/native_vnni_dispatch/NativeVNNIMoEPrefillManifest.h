/**
 * @file NativeVNNIMoEPrefillManifest.h
 * @brief Shared C++ view of production-derived Qwen MoE prefill mixtures.
 *
 * The Python refresh transaction reads bounded GGUF headers from immutable
 * Hugging Face revisions and checks in a compact source manifest. This header
 * gives CUDA and ROCm performance harnesses one common, model-agnostic view of
 * those records. Runtime dispatch keys use only formats, projection geometry,
 * and work size; release names, quantization filenames, and layer numbers are
 * retained solely as evidence provenance.
 *
 * No production inference code includes this file or parses JSON. Trainer
 * binaries validate it before launching expensive sweeps, then generated C++
 * exact-overlay tables carry the selected launch configurations into runtime.
 */

#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef LLAMINAR_NATIVE_VNNI_MOE_GGUF_PATTERN_MANIFEST_PATH
#error "MoE prefill trainer must define the production GGUF pattern manifest path"
#endif

#ifndef LLAMINAR_NATIVE_VNNI_QWEN_RELEASE_CATALOG_PATH
#error "MoE prefill trainer must define the Qwen release catalog path"
#endif

namespace llaminar2::test::native_vnni_dispatch
{
    /** @brief Gate, up, and down source formats for one expert family. */
    struct MoEPrefillFormatTriple
    {
        std::string gate;
        std::string up;
        std::string down;

        /** @brief Return fields in the policy ABI's stable role order. */
        std::array<std::string, 3> ordered() const
        {
            return {gate, up, down};
        }
    };

    /** @brief Checkpoint layers represented by one model-independent mixture. */
    struct MoEPrefillPatternUse
    {
        std::string release_id;
        std::string variant_id;
        std::vector<int> layers;
    };

    /**
     * @brief One unique routed/shared prefill geometry and source-format tuple.
     *
     * ``native_vnni_sweepable`` covers all six role tensors. A false value does
     * not remove the record: the all-format harness still owns it through the
     * applicable floating-point or future-format path.
     */
    struct MoEPrefillMixtureSpec
    {
        int hidden_size = 0;
        int routed_expert_width = 0;
        int shared_expert_width = 0;
        int expert_count = 0;
        int experts_per_token = 0;
        MoEPrefillFormatTriple routed;
        MoEPrefillFormatTriple shared;
        bool native_vnni_sweepable = false;
        std::vector<MoEPrefillPatternUse> uses;

        /** @brief Stable human-readable identity used only in evidence rows. */
        std::string evidenceId() const
        {
            return
                "h" + std::to_string(hidden_size) +
                "_r" + std::to_string(routed_expert_width) +
                "_s" + std::to_string(shared_expert_width) +
                "_e" + std::to_string(expert_count) +
                "_k" + std::to_string(experts_per_token) +
                "_rg" + routed.gate +
                "_ru" + routed.up +
                "_rd" + routed.down +
                "_sg" + shared.gate +
                "_su" + shared.up +
                "_sd" + shared.down;
        }
    };

    /**
     * @brief One distinct source-format key for the routed grouped launcher.
     *
     * Shared-expert formats are deliberately absent from the execution key:
     * they use the ordinary dense prefill surface. `mixture_evidence_ids`
     * retains every full six-role GGUF tuple that depends on this routed case.
     */
    struct MoERoutedPrefillCase
    {
        int hidden_size = 0;
        int routed_expert_width = 0;
        int expert_count = 0;
        int experts_per_token = 0;
        MoEPrefillFormatTriple routed;
        std::vector<std::string> mixture_evidence_ids;

        /** @brief Stable source-key identity for resumable evidence. */
        std::string evidenceId() const
        {
            return
                "h" + std::to_string(hidden_size) +
                "_r" + std::to_string(routed_expert_width) +
                "_e" + std::to_string(expert_count) +
                "_k" + std::to_string(experts_per_token) +
                "_g" + routed.gate +
                "_u" + routed.up +
                "_d" + routed.down;
        }
    };

    namespace detail
    {
        /** @brief Read one required JSON document with a contextual failure. */
        inline nlohmann::json readJsonDocument(
            const std::string &path,
            const char *description)
        {
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error(
                    "Unable to open " + std::string(description) + ": " + path);
            }
            try
            {
                nlohmann::json document;
                input >> document;
                return document;
            }
            catch (const nlohmann::json::exception &error)
            {
                throw std::runtime_error(
                    "Unable to parse " + std::string(description) + " " +
                    path + ": " + error.what());
            }
        }

        /** @brief Return the reviewed NativeVNNI source-format inventory. */
        inline const std::set<std::string> &nativeVnniSourceFormats()
        {
            static const std::set<std::string> formats = {
                "Q4_0", "IQ4_NL", "IQ4_XS", "Q4_1", "Q4_K",
                "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q3_K", "Q2_K",
                "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS",
                "IQ1_S", "IQ1_M", "Q8_0", "Q8_1", "Q8_K",
            };
            return formats;
        }

        /** @brief Parse and validate one gate/up/down format object. */
        inline MoEPrefillFormatTriple parseFormatTriple(
            const nlohmann::json &record,
            const std::string &context)
        {
            if (!record.is_object() || record.size() != 3 ||
                !record.contains("gate") || !record.contains("up") ||
                !record.contains("down"))
            {
                throw std::runtime_error(
                    context + " must contain exactly gate/up/down formats");
            }
            MoEPrefillFormatTriple result{
                record.at("gate").get<std::string>(),
                record.at("up").get<std::string>(),
                record.at("down").get<std::string>(),
            };
            if (result.gate.empty() || result.up.empty() || result.down.empty())
                throw std::runtime_error(context + " contains an empty format");
            return result;
        }

        /** @brief Test all six roles against the common NativeVNNI registry. */
        inline bool isNativeVnniMixture(
            const MoEPrefillFormatTriple &routed,
            const MoEPrefillFormatTriple &shared)
        {
            const auto &formats = nativeVnniSourceFormats();
            const auto owns = [&formats](const MoEPrefillFormatTriple &triple)
            {
                const auto ordered = triple.ordered();
                return std::all_of(
                    ordered.begin(),
                    ordered.end(),
                    [&formats](const std::string &format)
                    {
                        return formats.contains(format);
                    });
            };
            return owns(routed) && owns(shared);
        }

        /** @brief Test one projection triple against the NativeVNNI registry. */
        inline bool isNativeVnniTriple(
            const MoEPrefillFormatTriple &triple)
        {
            const auto &formats = nativeVnniSourceFormats();
            const auto ordered = triple.ordered();
            return std::all_of(
                ordered.begin(),
                ordered.end(),
                [&formats](const std::string &format)
                {
                    return formats.contains(format);
                });
        }
    }

    /**
     * @brief Load the deduplicated production MoE prefill mixture inventory.
     *
     * @return Immutable format/geometry records shared by CUDA and ROCm.
     * @throws std::runtime_error for a stale schema, unknown release, malformed
     * layer partition, or inconsistent GGUF/release geometry.
     */
    inline const std::vector<MoEPrefillMixtureSpec> &
    nativeVnniMoEPrefillMixtureManifest()
    {
        static const std::vector<MoEPrefillMixtureSpec> manifest = []
        {
            const std::string pattern_path =
                LLAMINAR_NATIVE_VNNI_MOE_GGUF_PATTERN_MANIFEST_PATH;
            const auto patterns = detail::readJsonDocument(
                pattern_path, "Qwen MoE GGUF pattern manifest");
            if (!patterns.is_object() ||
                patterns.value("schema_version", "") !=
                    "qwen-moe-gguf-codebook-patterns-v1" ||
                !patterns.contains("variants") ||
                !patterns.at("variants").is_array() ||
                patterns.at("variants").empty())
            {
                throw std::runtime_error(
                    "Qwen MoE GGUF pattern manifest has an invalid root schema: " +
                    pattern_path);
            }

            const std::string release_path =
                LLAMINAR_NATIVE_VNNI_QWEN_RELEASE_CATALOG_PATH;
            const auto releases = detail::readJsonDocument(
                release_path, "Qwen release catalog");
            if (!releases.is_object() ||
                releases.value("schema_version", "") !=
                    "qwen35-qwen36-release-models-v1" ||
                !releases.contains("releases") ||
                !releases.at("releases").is_array())
            {
                throw std::runtime_error(
                    "Qwen release catalog has an invalid root schema: " +
                    release_path);
            }

            std::unordered_map<std::string, std::pair<int, int>> geometry;
            for (const auto &release : releases.at("releases"))
            {
                if (release.value("kind", "") != "moe")
                    continue;
                const std::string id = release.at("release_id").get<std::string>();
                const int hidden = release.at("hidden_size").get<int>();
                const int expert = release.at("feed_forward_size").get<int>();
                if (id.empty() || hidden <= 0 || expert <= 0 ||
                    !geometry.emplace(id, std::pair{hidden, expert}).second)
                {
                    throw std::runtime_error(
                        "Qwen release catalog contains invalid duplicate MoE geometry");
                }
            }

            using MixtureKey = std::tuple<
                int, int, int, int, int,
                std::string, std::string, std::string,
                std::string, std::string, std::string>;
            std::map<MixtureKey, MoEPrefillMixtureSpec> unique;
            for (const auto &variant : patterns.at("variants"))
            {
                const std::string release_id =
                    variant.at("release_id").get<std::string>();
                const auto geometry_it = geometry.find(release_id);
                if (geometry_it == geometry.end())
                {
                    throw std::runtime_error(
                        "GGUF pattern references unknown MoE release " + release_id);
                }
                const int hidden = geometry_it->second.first;
                const int expert = variant.at("expert_width").get<int>();
                const int shared = variant.at("shared_expert_width").get<int>();
                const int expert_count = variant.at("expert_count").get<int>();
                const int experts_per_token =
                    variant.at("experts_per_token").get<int>();
                const int blocks = variant.at("block_count").get<int>();
                const std::string variant_id =
                    variant.at("variant_id").get<std::string>();
                if (expert != geometry_it->second.second || shared <= 0 ||
                    expert_count <= 0 || experts_per_token <= 0 ||
                    experts_per_token > expert_count ||
                    blocks <= 0 || variant_id.empty() ||
                    !variant.contains("layer_patterns") ||
                    !variant.at("layer_patterns").is_array())
                {
                    throw std::runtime_error(
                        release_id + "/" + variant_id +
                        " has inconsistent MoE geometry");
                }

                std::set<int> covered_layers;
                for (const auto &pattern : variant.at("layer_patterns"))
                {
                    const auto routed = detail::parseFormatTriple(
                        pattern.at("routed"), release_id + " routed pattern");
                    const auto shared_formats = detail::parseFormatTriple(
                        pattern.at("shared"), release_id + " shared pattern");
                    const MixtureKey key{
                        hidden, expert, shared, expert_count, experts_per_token,
                        routed.gate, routed.up, routed.down,
                        shared_formats.gate, shared_formats.up,
                        shared_formats.down,
                    };
                    auto [entry, inserted] = unique.try_emplace(
                        key,
                        MoEPrefillMixtureSpec{
                            hidden,
                            expert,
                            shared,
                            expert_count,
                            experts_per_token,
                            routed,
                            shared_formats,
                            detail::isNativeVnniMixture(routed, shared_formats),
                            {},
                        });
                    (void)inserted;
                    std::vector<int> layers =
                        pattern.at("layers").get<std::vector<int>>();
                    if (layers.empty() ||
                        !std::is_sorted(layers.begin(), layers.end()) ||
                        std::adjacent_find(layers.begin(), layers.end()) != layers.end())
                    {
                        throw std::runtime_error(
                            release_id + "/" + variant_id +
                            " has invalid pattern layers");
                    }
                    for (const int layer : layers)
                    {
                        if (layer < 0 || layer >= blocks ||
                            !covered_layers.insert(layer).second)
                        {
                            throw std::runtime_error(
                                release_id + "/" + variant_id +
                                " has overlapping or out-of-range layers");
                        }
                    }
                    entry->second.uses.push_back(
                        {release_id, variant_id, std::move(layers)});
                }
                if (covered_layers.size() != static_cast<size_t>(blocks))
                {
                    throw std::runtime_error(
                        release_id + "/" + variant_id +
                        " does not cover every model layer");
                }
            }

            std::vector<MoEPrefillMixtureSpec> result;
            result.reserve(unique.size());
            for (auto &[key, mixture] : unique)
            {
                (void)key;
                std::sort(
                    mixture.uses.begin(),
                    mixture.uses.end(),
                    [](const auto &left, const auto &right)
                    {
                        return std::tie(
                                   left.release_id,
                                   left.variant_id,
                                   left.layers) <
                               std::tie(
                                   right.release_id,
                                   right.variant_id,
                                   right.layers);
                    });
                result.push_back(std::move(mixture));
            }
            if (result.empty())
            {
                throw std::runtime_error(
                    "Qwen MoE GGUF mixture manifest contains no execution records");
            }
            return result;
        }();
        return manifest;
    }

    /**
     * @brief Collapse full mixtures onto distinct grouped routed source keys.
     */
    inline const std::vector<MoERoutedPrefillCase> &
    nativeVnniMoERoutedPrefillCases()
    {
        static const std::vector<MoERoutedPrefillCase> cases = []
        {
            using Key = std::tuple<
                int, int, int, int,
                std::string, std::string, std::string>;
            std::map<Key, MoERoutedPrefillCase> unique;
            for (const auto &mixture : nativeVnniMoEPrefillMixtureManifest())
            {
                if (!detail::isNativeVnniTriple(mixture.routed))
                    continue;
                const Key key{
                    mixture.hidden_size,
                    mixture.routed_expert_width,
                    mixture.expert_count,
                    mixture.experts_per_token,
                    mixture.routed.gate,
                    mixture.routed.up,
                    mixture.routed.down,
                };
                auto [entry, inserted] = unique.try_emplace(
                    key,
                    MoERoutedPrefillCase{
                        mixture.hidden_size,
                        mixture.routed_expert_width,
                        mixture.expert_count,
                        mixture.experts_per_token,
                        mixture.routed,
                        {},
                    });
                (void)inserted;
                entry->second.mixture_evidence_ids.push_back(
                    mixture.evidenceId());
            }

            std::vector<MoERoutedPrefillCase> result;
            result.reserve(unique.size());
            for (auto &[key, routed_case] : unique)
            {
                (void)key;
                auto &ids = routed_case.mixture_evidence_ids;
                std::sort(ids.begin(), ids.end());
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                result.push_back(std::move(routed_case));
            }
            if (result.empty())
                throw std::runtime_error("GGUF manifest has no NativeVNNI routed cases");
            return result;
        }();
        return cases;
    }
}
