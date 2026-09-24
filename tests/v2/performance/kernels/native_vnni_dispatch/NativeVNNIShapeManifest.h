/**
 * @file NativeVNNIShapeManifest.h
 * @brief Structured loader for the canonical cross-backend NativeVNNI shapes.
 *
 * Trainer binaries consume the same checked-in JSON manifest as the common
 * Python compiler and refresh transaction. This keeps model dimensions,
 * generic-certification points, exact-overlay ownership, and independent
 * Fast/verifier/prefill sealed assignments identical on CPU, CUDA, and ROCm.
 *
 * This header belongs only to performance and integration tooling. Production
 * inference consumes the generated policy artifact and never parses JSON.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef LLAMINAR_NATIVE_VNNI_SHAPE_MANIFEST_PATH
#error "NativeVNNI trainer target must define LLAMINAR_NATIVE_VNNI_SHAPE_MANIFEST_PATH"
#endif

#ifndef LLAMINAR_NATIVE_VNNI_QWEN_RELEASE_CATALOG_PATH
#error "NativeVNNI trainer target must define LLAMINAR_NATIVE_VNNI_QWEN_RELEASE_CATALOG_PATH"
#endif

namespace llaminar2::test::native_vnni_dispatch
{
    /**
     * @brief One validated model or generic-certification matrix geometry.
     *
     * The trainer kernels need only the name and matrix dimensions. The
     * remaining fields are retained so malformed or incomplete records fail at
     * the trainer boundary instead of producing a corpus that the compiler must
     * reject after an expensive sweep.
     */
    struct NativeVNNIShapeSpec
    {
        std::string name;
        int N = 0;
        int K = 0;
        std::string model_family;
        std::string role;
        bool exact_overlay = false;
        std::string aspect_bucket;
        /** Fast M=1 partition, or nullopt when this geometry is verifier-only. */
        std::optional<std::string> fast_partition;

        /** Grouped-verifier partition, or nullopt when this geometry is Fast-only. */
        std::optional<std::string> verifier_partition;

        /**
         * Ordinary-prefill partition, or nullopt for shapes owned only by the
         * decode or grouped-verifier contracts.
         *
         * This field was introduced after the canonical v5 decode lattice was
         * committed. Older records therefore omit it rather than spelling out
         * JSON null; new prefill-only records must declare it explicitly.
         */
        std::optional<std::string> prefill_partition;
    };

    /**
     * @brief Parse and validate the canonical NativeVNNI shape manifest once.
     *
     * @return Immutable manifest-order shape records shared by every trainer.
     * @throws std::runtime_error when the file is absent, malformed, contains a
     * duplicate name, or specifies dimensions incompatible with blockwise
     * NativeVNNI preparation.
     */
    inline const std::vector<NativeVNNIShapeSpec> &nativeVnniShapeManifest()
    {
        static const std::vector<NativeVNNIShapeSpec> manifest = []
        {
            const std::string path = LLAMINAR_NATIVE_VNNI_SHAPE_MANIFEST_PATH;
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error(
                    "Unable to open NativeVNNI shape manifest: " + path);
            }

            nlohmann::json root;
            try
            {
                input >> root;
            }
            catch (const nlohmann::json::exception &error)
            {
                throw std::runtime_error(
                    "Unable to parse NativeVNNI shape manifest " + path +
                    ": " + error.what());
            }
            if (!root.is_object() || !root.contains("schema_version") ||
                !root.contains("description") ||
                !root.contains("maximum_supported_weight_elements") ||
                !root.contains("maximum_cpu_measurement_weight_elements") ||
                !root.contains("shapes") ||
                !root.at("shapes").is_array() || root.at("shapes").empty())
            {
                throw std::runtime_error(
                    "NativeVNNI shape manifest has an invalid root schema: " + path);
            }

            std::vector<NativeVNNIShapeSpec> shapes;
            std::unordered_set<std::string> names;
            std::unordered_set<std::string> role_dimensions;
            std::unordered_set<std::string> production_dimensions;
            shapes.reserve(root.at("shapes").size());
            const uint64_t maximum_supported_weight_elements =
                root.at("maximum_supported_weight_elements").get<uint64_t>();
            if (maximum_supported_weight_elements == 0)
            {
                throw std::runtime_error(
                    "NativeVNNI shape manifest has a zero projection-work "
                    "envelope: " + path);
            }
            const uint64_t maximum_cpu_measurement_weight_elements =
                root.at("maximum_cpu_measurement_weight_elements").get<uint64_t>();
            if (maximum_cpu_measurement_weight_elements == 0 ||
                maximum_cpu_measurement_weight_elements >
                    maximum_supported_weight_elements)
            {
                throw std::runtime_error(
                    "NativeVNNI shape manifest has an invalid CPU measurement "
                    "envelope: " + path);
            }

            /**
             * Decode an explicit surface assignment without inventing a
             * development default. JSON null is a reviewed statement that the
             * geometry does not belong to that semantic contract.
             */
            const auto parse_partition = [&path](
                                             const nlohmann::json &record,
                                             const char *field,
                                             const std::string &shape_name,
                                             bool field_is_optional = false)
                -> std::optional<std::string>
            {
                if (field_is_optional && !record.contains(field))
                    return std::nullopt;
                const auto &value = record.at(field);
                if (value.is_null())
                    return std::nullopt;
                const std::string partition = value.get<std::string>();
                if (partition != "development" && partition != "sealed")
                {
                    throw std::runtime_error(
                        "Invalid NativeVNNI " + std::string(field) +
                        " for " + shape_name + " in " + path);
                }
                return partition;
            };

            for (const auto &record : root.at("shapes"))
            {
                NativeVNNIShapeSpec shape;
                try
                {
                    shape.name = record.at("name").get<std::string>();
                    shape.N = record.at("n").get<int>();
                    shape.K = record.at("k").get<int>();
                    shape.model_family = record.at("model_family").get<std::string>();
                    shape.role = record.at("role").get<std::string>();
                    shape.exact_overlay = record.at("exact_overlay").get<bool>();
                    shape.aspect_bucket = record.at("aspect_bucket").get<std::string>();
                    shape.fast_partition = parse_partition(
                        record, "fast_partition", shape.name);
                    shape.verifier_partition = parse_partition(
                        record, "verifier_partition", shape.name);
                    shape.prefill_partition = parse_partition(
                        record,
                        "prefill_partition",
                        shape.name,
                        /*field_is_optional=*/true);
                }
                catch (const nlohmann::json::exception &error)
                {
                    throw std::runtime_error(
                        "Invalid NativeVNNI shape record in " + path + ": " +
                        error.what());
                }

                if (shape.name.empty() || shape.model_family.empty() ||
                    shape.N <= 0 || shape.K <= 0 || shape.K % 32 != 0)
                {
                    throw std::runtime_error(
                        "Invalid NativeVNNI shape dimensions or identity: " +
                        shape.name);
                }
                const uint64_t weight_elements =
                    static_cast<uint64_t>(shape.N) *
                    static_cast<uint64_t>(shape.K);
                if (weight_elements > maximum_supported_weight_elements)
                {
                    throw std::runtime_error(
                        "NativeVNNI shape exceeds the supported projection-work "
                        "envelope: " + shape.name + "=" +
                        std::to_string(shape.N) + "x" +
                        std::to_string(shape.K));
                }
                if (!shape.fast_partition && !shape.verifier_partition &&
                    !shape.prefill_partition)
                {
                    throw std::runtime_error(
                        "NativeVNNI shape has no applicable semantic surface: " +
                        shape.name);
                }
                if (!names.insert(shape.name).second)
                {
                    throw std::runtime_error(
                        "Duplicate NativeVNNI shape name: " + shape.name);
                }
                const std::string dimension_key =
                    std::to_string(shape.N) + "x" + std::to_string(shape.K);
                if (!role_dimensions.insert(
                        shape.role + ":" + dimension_key).second)
                {
                    throw std::runtime_error(
                        "Duplicate NativeVNNI " + shape.role +
                        " shape dimensions: " + dimension_key);
                }
                if (shape.role == "production")
                    production_dimensions.insert(dimension_key);
                shapes.push_back(std::move(shape));
            }

            /**
             * Derive exact-overlay geometries for every released Qwen 3.5 and
             * Qwen 3.6 text backbone from the same declarative model catalog
             * consumed by Python planning.  Keeping model parameters in one
             * JSON file prevents a subtle split-brain failure where Python
             * schedules an overlay that CPU, CUDA, and ROCm trainer binaries
             * cannot resolve by name.
             *
             * Runtime dispatch remains model agnostic.  Release names are
             * discarded here after they authorize projection geometries; the
             * generated policy sees only codebook, N, K, and work size.
             */
            const std::string release_catalog_path =
                LLAMINAR_NATIVE_VNNI_QWEN_RELEASE_CATALOG_PATH;
            std::ifstream release_catalog_input(release_catalog_path);
            if (!release_catalog_input)
            {
                throw std::runtime_error(
                    "Unable to open Qwen release catalog: " +
                    release_catalog_path);
            }
            nlohmann::json release_catalog;
            try
            {
                release_catalog_input >> release_catalog;
            }
            catch (const nlohmann::json::exception &error)
            {
                throw std::runtime_error(
                    "Unable to parse Qwen release catalog " +
                    release_catalog_path + ": " + error.what());
            }
            if (!release_catalog.is_object() ||
                release_catalog.value("schema_version", "") !=
                    "qwen35-qwen36-release-models-v1" ||
                !release_catalog.contains("releases") ||
                !release_catalog.at("releases").is_array() ||
                release_catalog.at("releases").empty())
            {
                throw std::runtime_error(
                    "Qwen release catalog has an invalid root schema: " +
                    release_catalog_path);
            }

            int vocabulary_size = 0;
            int attention_head_dimension = 0;
            int gdn_key_head_count = 0;
            int gdn_key_head_dimension = 0;
            try
            {
                vocabulary_size =
                    release_catalog.at("vocabulary_size").get<int>();
                attention_head_dimension =
                    release_catalog.at("attention_head_dimension").get<int>();
                gdn_key_head_count =
                    release_catalog.at("gdn_key_head_count").get<int>();
                gdn_key_head_dimension =
                    release_catalog.at("gdn_key_head_dimension").get<int>();
            }
            catch (const nlohmann::json::exception &error)
            {
                throw std::runtime_error(
                    "Invalid Qwen release catalog constants in " +
                    release_catalog_path + ": " + error.what());
            }
            if (vocabulary_size <= 0 || attention_head_dimension <= 0 ||
                gdn_key_head_count <= 0 || gdn_key_head_dimension <= 0)
            {
                throw std::runtime_error(
                    "Qwen release catalog constants must be positive: " +
                    release_catalog_path);
            }

            std::set<std::pair<int, int>> release_geometries;
            std::unordered_set<std::string> release_ids;
            for (const auto &record : release_catalog.at("releases"))
            {
                try
                {
                    const std::string release_id =
                        record.at("release_id").get<std::string>();
                    const std::string kind =
                        record.at("kind").get<std::string>();
                    const int hidden = record.at("hidden_size").get<int>();
                    const int feed_forward =
                        record.at("feed_forward_size").get<int>();
                    const int attention_heads =
                        record.at("attention_head_count").get<int>();
                    const int attention_kv_heads =
                        record.at("attention_kv_head_count").get<int>();
                    const int gdn_value_heads =
                        record.at("gdn_value_head_count").get<int>();
                    const int gdn_time_rank =
                        record.at("gdn_time_step_rank").get<int>();
                    if (release_id.empty() ||
                        !release_ids.insert(release_id).second ||
                        (kind != "dense" && kind != "moe") || hidden <= 0 ||
                        feed_forward <= 0 || attention_heads <= 0 ||
                        attention_kv_heads <= 0 || gdn_value_heads <= 0 ||
                        gdn_time_rank <= 0)
                    {
                        throw std::runtime_error(
                            "Invalid or duplicate Qwen release model: " +
                            release_id);
                    }

                    const int query_width =
                        attention_heads * attention_head_dimension;
                    const int kv_width =
                        attention_kv_heads * attention_head_dimension;
                    const int gdn_inner_width =
                        gdn_value_heads * gdn_key_head_dimension;
                    const int gdn_qkv_width =
                        2 * gdn_key_head_count * gdn_key_head_dimension +
                        gdn_inner_width;
                    release_geometries.insert({2 * query_width, hidden});
                    release_geometries.insert({kv_width, hidden});
                    release_geometries.insert({hidden, query_width});
                    release_geometries.insert({gdn_qkv_width, hidden});
                    release_geometries.insert({gdn_inner_width, hidden});
                    release_geometries.insert({gdn_time_rank, hidden});
                    release_geometries.insert({hidden, gdn_inner_width});
                    release_geometries.insert({hidden, 2 * hidden});
                    release_geometries.insert({feed_forward, hidden});
                    release_geometries.insert({hidden, feed_forward});
                    release_geometries.insert({vocabulary_size, hidden});
                }
                catch (const nlohmann::json::exception &error)
                {
                    throw std::runtime_error(
                        "Invalid Qwen release model in " +
                        release_catalog_path + ": " + error.what());
                }
            }

            for (const auto &[n, k] : release_geometries)
            {
                const std::string dimension_key =
                    std::to_string(n) + "x" + std::to_string(k);
                if (production_dimensions.count(dimension_key) != 0)
                    continue;
                const uint64_t weight_elements =
                    static_cast<uint64_t>(n) * static_cast<uint64_t>(k);
                if (k % 32 != 0 || weight_elements >
                                       maximum_supported_weight_elements)
                {
                    throw std::runtime_error(
                        "Qwen release geometry is unsupported: " +
                        dimension_key);
                }

                NativeVNNIShapeSpec shape;
                shape.name = "Qwen35Release_" + dimension_key;
                shape.N = n;
                shape.K = k;
                shape.model_family = "qwen35-qwen36-release-geometries";
                shape.role = "production";
                shape.exact_overlay = true;
                const double aspect =
                    static_cast<double>(n) / static_cast<double>(k);
                shape.aspect_bucket = aspect >= 16.0   ? "very_wide"
                                      : aspect >= 2.0  ? "wide"
                                      : aspect >= 0.75 ? "balanced"
                                                       : "tall";
                shape.fast_partition = "development";
                shape.verifier_partition = "development";
                shape.prefill_partition = std::nullopt;
                if (!names.insert(shape.name).second ||
                    !role_dimensions.insert(
                        shape.role + ":" + dimension_key).second)
                {
                    throw std::runtime_error(
                        "Duplicate Qwen release geometry: " + dimension_key);
                }
                production_dimensions.insert(dimension_key);
                shapes.push_back(std::move(shape));
            }
            return shapes;
        }();
        return manifest;
    }
} // namespace llaminar2::test::native_vnni_dispatch
