/**
 * @file prefill_diagnostics.cpp
 * @brief Implementation of prefill diagnostics helpers.
 * @author David Sanftenberg
 *
 * Extracted from distributed_transformer_pipeline.cpp to improve modularity.
 * Preserves all existing log tags and behavior.
 */

#include "prefill_diagnostics.h"
#include "debug_utils.h"
#include "utils/debug_env.h"
#include "logger.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace llaminar
{
    BufferStats computeBufferStats(const float *data, size_t size)
    {
        BufferStats stats;

        // Guard against invalid input
        if (!data || size == 0)
            return stats;

        // Initialize accumulators and extrema
        double sum = 0.0;
        double sumsq = 0.0;
        double min_v = static_cast<double>(data[0]);
        double max_v = static_cast<double>(data[0]);

        // Single-pass accumulation of statistics
        for (size_t i = 0; i < size; ++i)
        {
            double v = static_cast<double>(data[i]);

            // Accumulate moments
            sum += v;
            sumsq += v * v;

            // Track extrema
            if (v < min_v)
                min_v = v;
            if (v > max_v)
                max_v = v;
        }

        // Compute derived statistics
        double mean = sum / static_cast<double>(size);
        double variance = std::max(0.0, sumsq / static_cast<double>(size) - mean * mean);

        // Populate result structure
        stats.min = min_v;
        stats.max = max_v;
        stats.mean = mean;
        stats.rms = std::sqrt(sumsq / static_cast<double>(size));
        stats.l2 = std::sqrt(sumsq);
        stats.stddev = std::sqrt(variance);

        return stats;
    }

    DiffSummary computeDiffSummary(const float *a, const float *b, size_t size)
    {
        DiffSummary summary;

        // Guard against invalid input
        if (!a || !b || size == 0)
            return summary;

        // Initialize tracking variables for worst-case element
        double max_abs = 0.0;
        size_t worst = 0;
        float worst_a = 0.0f;
        float worst_b = 0.0f;

        // Initialize accumulators for aggregate metrics
        double sum_abs = 0.0;
        long double sum_sq = 0.0L;
        long double denom_sq = 0.0L;

        // Single-pass computation of all diff metrics
        for (size_t i = 0; i < size; ++i)
        {
            // Compute element-wise difference
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            double abs_diff = std::fabs(diff);

            // Accumulate absolute error
            sum_abs += abs_diff;

            // Accumulate squared error for L2 norm
            sum_sq += diff * diff;

            // Accumulate denominator for relative L2
            double base = static_cast<double>(b[i]);
            denom_sq += base * base;

            // Track worst-case element
            if (abs_diff > max_abs)
            {
                max_abs = abs_diff;
                worst = i;
                worst_a = a[i];
                worst_b = b[i];
            }
        }

        // Compute final metrics
        summary.max_abs = max_abs;
        summary.mean_abs = sum_abs / static_cast<double>(size);
        summary.rel_l2 = std::sqrt(static_cast<double>(sum_sq)) / (std::sqrt(static_cast<double>(denom_sq)) + 1e-30);

        // Record worst-case element details for debugging
        summary.worst_index = worst;
        summary.value_a = worst_a;
        summary.value_b = worst_b;

        return summary;
    }

    PrefillBaselineRegistry &PrefillBaselineRegistry::instance()
    {
        static PrefillBaselineRegistry inst;
        return inst;
    }

    void PrefillBaselineRegistry::clear()
    {
        std::lock_guard<std::mutex> g(mutex_);
        storage_.clear();
    }

    bool PrefillBaselineRegistry::ensure(const std::string &name, const float *data, size_t count)
    {
        if (!data || count == 0)
            return false;
        std::lock_guard<std::mutex> g(mutex_);
        auto it = storage_.find(name);
        if (it == storage_.end() || it->second.size() != count)
        {
            storage_[name] = std::vector<float>(data, data + count);
            return true;
        }
        return false;
    }

    bool PrefillBaselineRegistry::fetch(const std::string &name, std::vector<float> &out) const
    {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = storage_.find(name);
        if (it == storage_.end())
            return false;
        out = it->second;
        return true;
    }

    void handle_prefill_stage_snapshot(int rank, const std::string &name, const float *data, size_t count,
                                       int cols, double warn_threshold, bool capture_enabled, bool compare_enabled)
    {
        if (rank != 0 || !data || count == 0)
            return;
        auto &reg = PrefillBaselineRegistry::instance();
        if (capture_enabled)
        {
            reg.ensure(name, data, count);
            LOG_TRACE("[PrefillBaseline] captured " << name << " count=" << count);
        }
        if (compare_enabled)
        {
            std::vector<float> baseline;
            if (!reg.fetch(name, baseline) || baseline.size() != count)
            {
                LOG_WARN("[PrefillBaseline] missing baseline for compare name=" << name);
                return;
            }
            auto diff = computeDiffSummary(data, baseline.data(), count);
            if (diff.rel_l2 > warn_threshold)
            {
                LOG_WARN("[PrefillBaseline] rel_l2=" << diff.rel_l2 << " max_abs=" << diff.max_abs << " mean_abs=" << diff.mean_abs << " name=" << name);
            }
            else
            {
                LOG_DEBUG("[PrefillBaseline] OK name=" << name << " rel_l2=" << diff.rel_l2);
            }
        }
    }

    bool isFFNShardTracingEnabledFor(const std::string &label)
    {
        const auto &cfg = debugEnv().ffn_shard_trace;
        if (!cfg.enabled)
            return false;
        if (cfg.match_all)
            return true;
        if (cfg.shards_spec.empty())
            return true;
        std::stringstream ss(cfg.shards_spec);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            // lightweight trim
            size_t b = tok.find_first_not_of(" \t\n\r");
            if (b == std::string::npos)
                continue;
            size_t e = tok.find_last_not_of(" \t\n\r");
            std::string t = tok.substr(b, e - b + 1);
            if (t == label)
                return true;
        }
        return false;
    }

    void logFFNRowPreviewIfEnabled(const std::string &label,
                                   const std::shared_ptr<TensorBase> &tensor,
                                   size_t default_preview_cols)
    {
        if (!tensor)
            return;
        const auto &cfg = debugEnv().ffn_shard_trace;
        if (!cfg.enabled || !isFFNShardTracingEnabledFor(label))
            return;
        const auto &shape = tensor->shape();
        if (shape.size() < 2)
            return;
        int total_rows = shape[0], total_cols = shape[1];
        if (total_rows <= 0 || total_cols <= 0)
            return;
        std::vector<int> rows;
        if (!cfg.rows_spec.empty())
        {
            std::stringstream ss(cfg.rows_spec);
            std::string tok;
            while (std::getline(ss, tok, ','))
            {
                int r = std::atoi(tok.c_str());
                if (r >= 0 && r < total_rows)
                    rows.push_back(r);
            }
        }
        if (rows.empty())
        {
            rows.push_back(0);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        if (rows.empty())
            return;
        size_t preview_cols = default_preview_cols;
        if (!cfg.cols.empty())
        {
            preview_cols = std::min<size_t>(cfg.cols.size(), (size_t)total_cols);
        }
        preview_cols = std::max<size_t>(1, std::min(preview_cols, (size_t)total_cols));
        logTensorRowPreview(tensor, label, rows, preview_cols, "PrefillFFNPreview");
    }

} // namespace llaminar
