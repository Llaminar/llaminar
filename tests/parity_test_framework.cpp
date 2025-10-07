/**
 * @file parity_test_framework.cpp
 * @brief Implementation of the parity test framework
 * @author David Sanftenberg
 */

#include "parity_test_framework.h"
#include "npz_loader.h"
#include "logger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <queue>
#include <map>
#include <fstream>
#include <unistd.h> // for getpid()
#include <cstdlib>  // for system()

namespace llaminar
{
    namespace parity
    {
        // Static member initialization
        bool LlaminarSnapshotHook::enabled_ = false;

        // ==================== SnapshotRegistry Implementation ====================

        SnapshotRegistry &SnapshotRegistry::instance()
        {
            static SnapshotRegistry registry;
            return registry;
        }

        void SnapshotRegistry::clear()
        {
            snapshots_.clear();
        }

        void SnapshotRegistry::register_snapshot(const std::string &key, const TensorSnapshot &snapshot)
        {
            snapshots_[key] = snapshot;
        }

        bool SnapshotRegistry::has_snapshot(const std::string &key) const
        {
            return snapshots_.find(key) != snapshots_.end();
        }

        bool SnapshotRegistry::get_snapshot(const std::string &key, TensorSnapshot &out_snapshot) const
        {
            auto it = snapshots_.find(key);
            if (it != snapshots_.end())
            {
                out_snapshot = it->second;
                return true;
            }
            return false;
        }

        std::vector<std::string> SnapshotRegistry::list_keys() const
        {
            std::vector<std::string> keys;
            keys.reserve(snapshots_.size());
            for (const auto &pair : snapshots_)
            {
                keys.push_back(pair.first);
            }
            return keys;
        }

        std::string SnapshotRegistry::make_key(const std::string &source, PipelineStage stage, int layer) const
        {
            return make_key(source, llaminar::stage_to_string(stage), layer);
        }

        std::string SnapshotRegistry::make_key(const std::string &source, const std::string &stage_name, int layer) const
        {
            std::ostringstream oss;
            oss << source;
            if (layer >= 0)
            {
                oss << "_layer_" << layer;
            }
            oss << "_" << stage_name;
            return oss.str();
        }

        // ==================== SnapshotComparator Implementation ====================

        ComparisonResult SnapshotComparator::compare(
            const TensorSnapshot &expected,
            const TensorSnapshot &actual,
            const ComparisonTolerance &tolerance)
        {
            ComparisonResult result;
            result.stage_name = expected.metadata.stage_name;
            result.layer_index = expected.metadata.layer_index;
            result.tolerance = tolerance;

            // Size validation
            if (expected.data.size() != actual.data.size())
            {
                std::ostringstream oss;
                oss << "Size mismatch: expected " << expected.data.size()
                    << " but got " << actual.data.size();
                result.error_message = oss.str();
                result.metrics.passed = false;
                return result;
            }

            if (expected.data.empty())
            {
                result.error_message = "Empty tensor data";
                result.metrics.passed = false;
                return result;
            }

            // Compute metrics
            result.metrics = compute_metrics(expected.data, actual.data);

            // Check tolerances
            bool pass_abs = result.metrics.max_abs_diff <= tolerance.max_abs;
            bool pass_l2 = result.metrics.rel_l2 <= tolerance.rel_l2;
            result.metrics.passed = pass_abs && pass_l2;

            if (!result.metrics.passed)
            {
                std::ostringstream oss;
                oss << "Tolerance exceeded: max_abs=" << result.metrics.max_abs_diff
                    << " (tol=" << tolerance.max_abs << "), rel_l2=" << result.metrics.rel_l2
                    << " (tol=" << tolerance.rel_l2 << ")";
                result.error_message = oss.str();
            }

            return result;
        }

        ComparisonMetrics SnapshotComparator::compute_metrics(
            const std::vector<float> &expected,
            const std::vector<float> &actual)
        {
            ComparisonMetrics metrics;

            if (expected.size() != actual.size() || expected.empty())
            {
                return metrics;
            }

            double sum_abs_diff = 0.0;
            double sum_sq_diff = 0.0;
            double sum_sq_ref = 0.0;

            for (size_t i = 0; i < expected.size(); ++i)
            {
                double exp = static_cast<double>(expected[i]);
                double act = static_cast<double>(actual[i]);
                double diff = act - exp;
                double abs_diff = std::fabs(diff);

                if (abs_diff > metrics.max_abs_diff)
                {
                    metrics.max_abs_diff = static_cast<float>(abs_diff);
                    metrics.worst_index = i;
                    metrics.worst_expected = expected[i];
                    metrics.worst_actual = actual[i];
                }

                sum_abs_diff += abs_diff;
                sum_sq_diff += diff * diff;
                sum_sq_ref += exp * exp;
            }

            metrics.mean_abs_diff = static_cast<float>(sum_abs_diff / expected.size());

            if (sum_sq_ref > 0.0)
            {
                metrics.rel_l2 = std::sqrt(sum_sq_diff) / std::sqrt(sum_sq_ref);
            }
            else
            {
                metrics.rel_l2 = (sum_sq_diff > 0.0) ? 1.0 : 0.0;
            }

            return metrics;
        }

        void SnapshotComparator::log_top_differences(
            const std::vector<float> &expected,
            const std::vector<float> &actual,
            int cols,
            int top_k,
            const std::string &label)
        {
            if (expected.size() != actual.size() || expected.empty() || cols <= 0 || top_k <= 0)
            {
                return;
            }

            struct DiffEntry
            {
                size_t index;
                float diff;
                float expected_val;
                float actual_val;
            };

            auto cmp = [](const DiffEntry &a, const DiffEntry &b)
            { return a.diff > b.diff; };
            std::priority_queue<DiffEntry, std::vector<DiffEntry>, decltype(cmp)> min_heap(cmp);

            for (size_t i = 0; i < expected.size(); ++i)
            {
                float diff = std::fabs(actual[i] - expected[i]);

                if (min_heap.size() < static_cast<size_t>(top_k))
                {
                    min_heap.push({i, diff, expected[i], actual[i]});
                }
                else if (diff > min_heap.top().diff)
                {
                    min_heap.pop();
                    min_heap.push({i, diff, expected[i], actual[i]});
                }
            }

            std::vector<DiffEntry> top_diffs;
            while (!min_heap.empty())
            {
                top_diffs.push_back(min_heap.top());
                min_heap.pop();
            }
            std::reverse(top_diffs.begin(), top_diffs.end());

            std::cout << "[PARITY_TOP_DIFF] " << label << " top_k=" << top_k << std::endl;
            for (const auto &entry : top_diffs)
            {
                size_t row = entry.index / static_cast<size_t>(cols);
                size_t col = entry.index % static_cast<size_t>(cols);
                std::cout << "  [" << row << "," << col << "] diff=" << entry.diff
                          << " expected=" << entry.expected_val
                          << " actual=" << entry.actual_val << std::endl;
            }
        }

        // ==================== LlaminarSnapshotHook Implementation ====================

        void LlaminarSnapshotHook::capture(
            PipelineStage stage,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim)
        {
            capture(llaminar::stage_to_string(stage), layer_index, data, seq_len, feature_dim);
        }

        void LlaminarSnapshotHook::capture(
            const std::string &stage_name,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim)
        {
            if (!enabled_ || !data || seq_len <= 0 || feature_dim <= 0)
            {
                return;
            }

            SnapshotMetadata meta;
            meta.stage_name = stage_name;
            meta.stage = llaminar::string_to_stage(stage_name);
            meta.layer_index = layer_index;
            meta.seq_len = seq_len;
            meta.feature_dim = feature_dim;
            meta.source = "llaminar";

            size_t count = static_cast<size_t>(seq_len) * static_cast<size_t>(feature_dim);
            TensorSnapshot snapshot(meta, data, count);

            auto &registry = SnapshotRegistry::instance();
            std::string key = registry.make_key("llaminar", stage_name, layer_index);
            registry.register_snapshot(key, snapshot);
        }

        void LlaminarSnapshotHook::set_enabled(bool enabled)
        {
            enabled_ = enabled;
        }

        bool LlaminarSnapshotHook::is_enabled()
        {
            return enabled_;
        }

        // ==================== PyTorch Snapshot Loader ====================

        size_t PytorchSnapshotLoader::load_from_npz(const std::string &filepath,
                                                    const std::string &source_name)
        {
            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open())
            {
                LOG_ERROR("Failed to open NPZ file: " << filepath);
                return 0;
            }
            file.close();

            size_t count = 0;
            auto &registry = SnapshotRegistry::instance();

            try
            {
                LOG_INFO("Loading PyTorch snapshots from: " << filepath);

                // NPZ is a ZIP file containing .npy files
                // Strategy: Extract to temp directory, load each .npy, then cleanup

                // Create temporary extraction directory
                std::string temp_dir = "/tmp/pytorch_npz_extract_" + std::to_string(getpid());
                std::string mkdir_cmd = "mkdir -p " + temp_dir;
                if (system(mkdir_cmd.c_str()) != 0)
                {
                    LOG_ERROR("Failed to create temp directory: " << temp_dir);
                    return 0;
                }

                // Extract NPZ using unzip (NPZ is just a ZIP file)
                std::string unzip_cmd = "unzip -q -o " + filepath + " -d " + temp_dir + " 2>/dev/null";
                if (system(unzip_cmd.c_str()) != 0)
                {
                    LOG_ERROR("Failed to extract NPZ file (is unzip installed?)");
                    system(("rm -rf " + temp_dir).c_str());
                    return 0;
                }

                // List extracted .npy files
                std::string ls_cmd = "ls " + temp_dir + "/*.npy 2>/dev/null";
                FILE *ls_pipe = popen(ls_cmd.c_str(), "r");
                if (!ls_pipe)
                {
                    LOG_ERROR("Failed to list extracted files");
                    system(("rm -rf " + temp_dir).c_str());
                    return 0;
                }

                // Read file list
                std::vector<std::string> npy_files;
                char buffer[512];
                while (fgets(buffer, sizeof(buffer), ls_pipe))
                {
                    std::string filename(buffer);
                    // Remove trailing newline
                    if (!filename.empty() && filename.back() == '\n')
                    {
                        filename.pop_back();
                    }
                    npy_files.push_back(filename);
                }
                pclose(ls_pipe);

                LOG_INFO("Found " << npy_files.size() << " .npy files in NPZ");

                // Load each .npy file and convert to snapshot
                for (const auto &npy_path : npy_files)
                {
                    // Extract key name from filename (remove path and .npy extension)
                    size_t last_slash = npy_path.rfind('/');
                    std::string filename = (last_slash != std::string::npos)
                                               ? npy_path.substr(last_slash + 1)
                                               : npy_path;

                    std::string key = filename.substr(0, filename.length() - 4); // Remove .npy

                    // Load the .npy file
                    NpyArray array;
                    if (!NpzLoader::load_npy(npy_path, array))
                    {
                        LOG_WARN("Failed to load " << npy_path << ", skipping");
                        continue;
                    }

                    // Parse key to get layer info
                    int layer_index;
                    std::string stage_name;
                    if (!parse_key(key, layer_index, stage_name))
                    {
                        LOG_WARN("Failed to parse key: " << key << ", skipping");
                        continue;
                    }

                    // Convert to TensorSnapshot and store in registry
                    SnapshotMetadata metadata;
                    metadata.source = source_name;
                    metadata.layer_index = layer_index;
                    metadata.stage_name = stage_name;
                    metadata.stage = PipelineStage::CUSTOM; // PyTorch doesn't use our enum

                    // Set dimensions based on shape
                    if (array.shape.size() >= 2)
                    {
                        metadata.seq_len = static_cast<int>(array.shape[array.shape.size() - 2]);
                        metadata.feature_dim = static_cast<int>(array.shape.back());
                    }
                    else if (array.shape.size() == 1)
                    {
                        metadata.seq_len = 1;
                        metadata.feature_dim = static_cast<int>(array.shape[0]);
                    }
                    metadata.total_elements = static_cast<int64_t>(array.data.size());

                    TensorSnapshot snapshot(metadata, array.data.data(), array.data.size());

                    // Create key for registry
                    std::string registry_key = registry.make_key(source_name, stage_name, layer_index);
                    registry.register_snapshot(registry_key, snapshot);
                    count++;

                    LOG_DEBUG("Loaded snapshot: " << key << " -> layer=" << layer_index
                                                  << " stage=" << stage_name << " shape=["
                                                  << array.shape[0]);
                    for (size_t i = 1; i < array.shape.size(); ++i)
                    {
                        LOG_DEBUG("," << array.shape[i]);
                    }
                    LOG_DEBUG("]");
                }

                // Cleanup temp directory
                system(("rm -rf " + temp_dir).c_str());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to load NPZ: " << e.what());
                return 0;
            }

            LOG_INFO("Loaded " << count << " PyTorch snapshots from " << filepath);
            return count;
        }

        bool PytorchSnapshotLoader::parse_key(const std::string &key,
                                              int &layer_index,
                                              std::string &stage_name)
        {
            // Handle special keys
            if (key == "embeddings" || key == "EMBEDDING")
            {
                layer_index = -1;
                stage_name = "EMBEDDING";
                return true;
            }
            if (key == "final_norm_out" || key == "FINAL_NORM")
            {
                layer_index = -1;
                stage_name = "FINAL_NORM";
                return true;
            }
            if (key == "logits" || key == "LM_HEAD")
            {
                layer_index = -1;
                stage_name = "LM_HEAD";
                return true;
            }

            // Try parsing layer_N_stage format first (old convention)
            std::string prefix = "layer_";
            if (key.substr(0, prefix.length()) == prefix)
            {
                // Find next underscore after "layer_"
                size_t first_underscore = prefix.length();
                size_t second_underscore = key.find('_', first_underscore);
                if (second_underscore != std::string::npos)
                {
                    // Extract layer number
                    std::string layer_str = key.substr(first_underscore,
                                                       second_underscore - first_underscore);
                    try
                    {
                        layer_index = std::stoi(layer_str);
                        stage_name = key.substr(second_underscore + 1);
                        return true;
                    }
                    catch (...)
                    {
                        // Fall through to try STAGE_N format
                    }
                }
            }

            // Try parsing STAGE_N format (new convention: "ATTENTION_OUTPUT_0")
            size_t last_underscore = key.rfind('_');
            if (last_underscore != std::string::npos && last_underscore > 0)
            {
                std::string potential_number = key.substr(last_underscore + 1);
                try
                {
                    layer_index = std::stoi(potential_number);
                    stage_name = key.substr(0, last_underscore);
                    return true;
                }
                catch (...)
                {
                    // Not a valid number
                }
            }

            return false;
        }

        // ==================== Layer-by-Layer Comparator ====================

        std::map<std::string, ComparisonResult> LayerByLayerComparator::compare_all(
            const std::string &source1,
            const std::string &source2,
            const ComparisonTolerance &tolerance)
        {
            std::map<std::string, ComparisonResult> results;
            auto &registry = SnapshotRegistry::instance();
            auto all_keys = registry.list_keys();

            // Group keys by layer and stage
            std::map<std::string, std::pair<std::string, std::string>> layer_keys;

            for (const auto &key : all_keys)
            {
                // Extract source from key (format: "source_layer_N_stage")
                size_t first_underscore = key.find('_');
                if (first_underscore == std::string::npos)
                    continue;

                std::string source = key.substr(0, first_underscore);
                std::string layer_stage = key.substr(first_underscore + 1);

                if (source == source1)
                {
                    layer_keys[layer_stage].first = key;
                }
                else if (source == source2)
                {
                    layer_keys[layer_stage].second = key;
                }
            }

            // Compare matching pairs
            for (const auto &[layer_stage, key_pair] : layer_keys)
            {
                if (!key_pair.first.empty() && !key_pair.second.empty())
                {
                    TensorSnapshot snap1, snap2;
                    if (registry.get_snapshot(key_pair.first, snap1) &&
                        registry.get_snapshot(key_pair.second, snap2))
                    {

                        auto result = SnapshotComparator::compare(snap1, snap2, tolerance);
                        results[layer_stage] = result;
                    }
                }
            }

            return results;
        }
        std::string LayerByLayerComparator::find_first_divergence(
            const std::string &source1,
            const std::string &source2,
            const ComparisonTolerance &tolerance)
        {
            auto results = compare_all(source1, source2, tolerance);

            // Sort by layer number to find first divergence
            std::vector<std::pair<std::string, ComparisonResult>> sorted_results(
                results.begin(), results.end());

            std::sort(sorted_results.begin(), sorted_results.end(),
                      [](const auto &a, const auto &b)
                      {
                          // Extract layer numbers for sorting
                          int layer_a = -1, layer_b = -1;
                          std::string stage_a, stage_b;
                          PytorchSnapshotLoader::parse_key(a.first, layer_a, stage_a);
                          PytorchSnapshotLoader::parse_key(b.first, layer_b, stage_b);
                          return layer_a < layer_b;
                      });

            for (const auto &[key, result] : sorted_results)
            {
                if (!result.passed())
                {
                    return key;
                }
            }

            return ""; // All layers match
        }

        void LayerByLayerComparator::print_report(
            const std::map<std::string, ComparisonResult> &results,
            bool verbose)
        {
            std::cout << "\n"
                      << std::string(80, '=') << "\n";
            std::cout << "LAYER-BY-LAYER COMPARISON REPORT\n";
            std::cout << std::string(80, '=') << "\n";
            std::cout << "Total layers compared: " << results.size() << "\n";

            size_t passed = 0, failed = 0;
            for (const auto &[key, result] : results)
            {
                if (result.passed())
                    ++passed;
                else
                    ++failed;
            }

            std::cout << "Passed: " << passed << "\n";
            std::cout << "Failed: " << failed << "\n";
            std::cout << std::string(80, '=') << "\n\n";

            // Print details
            for (const auto &[key, result] : results)
            {
                if (!verbose && result.passed())
                    continue;

                std::string status = result.passed() ? "✓" : "❌";
                std::cout << status << " " << key << ":\n";
                std::cout << "   Max abs diff: " << result.metrics.max_abs_diff << "\n";
                std::cout << "   Mean abs diff: " << result.metrics.mean_abs_diff << "\n";
                std::cout << "   Rel L2: " << result.metrics.rel_l2 << "\n";

                if (!result.passed())
                {
                    std::cout << "   🔍 DIVERGENCE DETECTED\n";
                    if (result.metrics.worst_index >= 0)
                    {
                        std::cout << "   Worst element: idx="
                                  << result.metrics.worst_index << "\n";
                    }
                }
                std::cout << "\n";
            }
        }

        // ==================== Utility Functions ====================
        // Note: stage_to_string() and string_to_stage() are now provided by
        // the core pipeline_stages.h header (inline functions in llaminar namespace).
        // We removed the duplicate implementations from this file.

    } // namespace parity
} // namespace llaminar
