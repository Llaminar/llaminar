/**
 * @file parity_test_framework.cpp
 * @brief Implementation of the parity test framework
 * @author David Sanftenberg
 */

#include "parity_test_framework.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <queue>

namespace llaminar
{
    namespace parity
    {
        // Static member initialization
        bool LlaminarSnapshotHook::enabled_ = false;

        // ==================== SnapshotRegistry Implementation ====================

        SnapshotRegistry& SnapshotRegistry::instance()
        {
            static SnapshotRegistry registry;
            return registry;
        }

        void SnapshotRegistry::clear()
        {
            snapshots_.clear();
        }

        void SnapshotRegistry::register_snapshot(const std::string& key, const TensorSnapshot& snapshot)
        {
            snapshots_[key] = snapshot;
        }

        bool SnapshotRegistry::has_snapshot(const std::string& key) const
        {
            return snapshots_.find(key) != snapshots_.end();
        }

        bool SnapshotRegistry::get_snapshot(const std::string& key, TensorSnapshot& out_snapshot) const
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
            for (const auto& pair : snapshots_)
            {
                keys.push_back(pair.first);
            }
            return keys;
        }

        std::string SnapshotRegistry::make_key(const std::string& source, PipelineStage stage, int layer) const
        {
            return make_key(source, stage_to_string(stage), layer);
        }

        std::string SnapshotRegistry::make_key(const std::string& source, const std::string& stage_name, int layer) const
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
            const TensorSnapshot& expected,
            const TensorSnapshot& actual,
            const ComparisonTolerance& tolerance)
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
            const std::vector<float>& expected,
            const std::vector<float>& actual)
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
            const std::vector<float>& expected,
            const std::vector<float>& actual,
            int cols,
            int top_k,
            const std::string& label)
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

            auto cmp = [](const DiffEntry& a, const DiffEntry& b) { return a.diff > b.diff; };
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
            for (const auto& entry : top_diffs)
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
            const float* data,
            int seq_len,
            int feature_dim)
        {
            capture(stage_to_string(stage), layer_index, data, seq_len, feature_dim);
        }

        void LlaminarSnapshotHook::capture(
            const std::string& stage_name,
            int layer_index,
            const float* data,
            int seq_len,
            int feature_dim)
        {
            if (!enabled_ || !data || seq_len <= 0 || feature_dim <= 0)
            {
                return;
            }

            SnapshotMetadata meta;
            meta.stage_name = stage_name;
            meta.stage = string_to_stage(stage_name);
            meta.layer_index = layer_index;
            meta.seq_len = seq_len;
            meta.feature_dim = feature_dim;
            meta.source = "llaminar";

            size_t count = static_cast<size_t>(seq_len) * static_cast<size_t>(feature_dim);
            TensorSnapshot snapshot(meta, data, count);

            auto& registry = SnapshotRegistry::instance();
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

        // ==================== Utility Functions ====================

        std::string stage_to_string(PipelineStage stage)
        {
            switch (stage)
            {
                case PipelineStage::EMBEDDING: return "embedding";
                case PipelineStage::ATTENTION_NORM: return "attention_norm";
                case PipelineStage::QKV_PROJECTION: return "qkv_projection";
                case PipelineStage::ROPE_APPLICATION: return "rope";
                case PipelineStage::ATTENTION_SCORES: return "attention_scores";
                case PipelineStage::ATTENTION_SOFTMAX: return "attention_softmax";
                case PipelineStage::ATTENTION_CONTEXT: return "attention_context";
                case PipelineStage::ATTENTION_OUTPUT: return "attention_output";
                case PipelineStage::ATTENTION_RESIDUAL: return "attention_residual";
                case PipelineStage::FFN_NORM: return "ffn_norm";
                case PipelineStage::FFN_GATE: return "ffn_gate";
                case PipelineStage::FFN_UP: return "ffn_up";
                case PipelineStage::FFN_SWIGLU: return "ffn_swiglu";
                case PipelineStage::FFN_DOWN: return "ffn_down";
                case PipelineStage::FFN_RESIDUAL: return "ffn_residual";
                case PipelineStage::FINAL_NORM: return "final_norm";
                case PipelineStage::LM_HEAD: return "lm_head";
                case PipelineStage::CUSTOM: return "custom";
                default: return "unknown";
            }
        }

        PipelineStage string_to_stage(const std::string& str)
        {
            if (str == "embedding") return PipelineStage::EMBEDDING;
            if (str == "attention_norm") return PipelineStage::ATTENTION_NORM;
            if (str == "qkv_projection") return PipelineStage::QKV_PROJECTION;
            if (str == "rope") return PipelineStage::ROPE_APPLICATION;
            if (str == "attention_scores") return PipelineStage::ATTENTION_SCORES;
            if (str == "attention_softmax") return PipelineStage::ATTENTION_SOFTMAX;
            if (str == "attention_context") return PipelineStage::ATTENTION_CONTEXT;
            if (str == "attention_output") return PipelineStage::ATTENTION_OUTPUT;
            if (str == "attention_residual") return PipelineStage::ATTENTION_RESIDUAL;
            if (str == "ffn_norm") return PipelineStage::FFN_NORM;
            if (str == "ffn_gate") return PipelineStage::FFN_GATE;
            if (str == "ffn_up") return PipelineStage::FFN_UP;
            if (str == "ffn_swiglu") return PipelineStage::FFN_SWIGLU;
            if (str == "ffn_down") return PipelineStage::FFN_DOWN;
            if (str == "ffn_residual") return PipelineStage::FFN_RESIDUAL;
            if (str == "final_norm") return PipelineStage::FINAL_NORM;
            if (str == "lm_head") return PipelineStage::LM_HEAD;
            return PipelineStage::CUSTOM;
        }

    } // namespace parity
} // namespace llaminar
