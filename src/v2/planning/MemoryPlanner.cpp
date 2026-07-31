#include "planning/MemoryPlanner.h"
#include "planning/WeightMemoryEstimator.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "planning/ActivationMemoryEstimator.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "utils/Logger.h"

#include "fort.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>

namespace llaminar2
{

namespace
{

std::string formatMB(size_t bytes)
{
    double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::ostringstream ss;
    if (mb >= 1024.0)
    {
        ss << std::fixed << std::setprecision(1) << (mb / 1024.0) << " GB";
    }
    else
    {
        ss << std::fixed << std::setprecision(0) << mb << " MB";
    }
    return ss.str();
}

} // anonymous namespace

MemoryPlan MemoryPlanner::plan(
    const ModelMemoryProfile& profile,
    const std::vector<DevicePlanConfig>& device_configs)
{
    MemoryPlan result;
    result.devices.reserve(device_configs.size());

    for (const auto& cfg : device_configs)
    {
        DeviceMemoryPlan dev_plan;
        dev_plan.device = cfg.device;
        dev_plan.device_total_bytes = cfg.device_total_bytes;
        dev_plan.device_free_bytes = cfg.device_free_bytes;
        dev_plan.headroom_bytes = cfg.headroom_bytes;

        int last_layer = cfg.last_layer >= 0 ? cfg.last_layer : profile.n_layers - 1;
        int max_seq = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;
        int activation_seq = cfg.activation_seq_len > 0 ? cfg.activation_seq_len : max_seq;
        if (max_seq > 0)
            activation_seq = std::min(activation_seq, max_seq);
        activation_seq = std::max(1, activation_seq);
        int local_kv_heads = cfg.local_kv_heads > 0 ? cfg.local_kv_heads : profile.n_kv_heads;

        dev_plan.max_seq_len = max_seq;
        dev_plan.activation_seq_len = activation_seq;

        // If TP sharded, divide KV heads
        if (cfg.total_shards > 1 && cfg.local_kv_heads <= 0)
        {
            local_kv_heads = std::max(1, profile.n_kv_heads / cfg.total_shards);
        }

        /*
         * The execution plan's layer interval names main-model blocks. MTP
         * predictors are trailing model-file blocks owned by the participant
         * that owns the terminal main layer, so include them explicitly in
         * that participant's persistent weight inventory.
         */
        const int main_layer_count =
            std::max(0, profile.n_layers - profile.mtp_layer_count);
        const bool owns_terminal_main_layer =
            main_layer_count > 0 &&
            cfg.first_layer <= main_layer_count - 1 &&
            last_layer >= main_layer_count - 1;
        const int weight_last_layer =
            cfg.mtp_enabled &&
                    owns_terminal_main_layer &&
                    profile.mtp_layer_count > 0
                ? profile.n_layers - 1
                : last_layer;

        // Weight estimation
        auto weight_est = WeightMemoryEstimator::estimate(
            profile, cfg.device,
            cfg.shard_index, cfg.total_shards,
            cfg.first_layer, weight_last_layer);
        dev_plan.weight_bytes = weight_est.device_bytes;

        /*
         * Cache planning follows the actual hybrid layer inventory. Full
         * attention contributes sequence-length-scaled KV storage; GDN and
         * MTP rollback state are exposed separately instead of hidden in a
         * workspace reserve.
         */
        const auto persistent_state =
            PersistentStateMemoryEstimator::estimate(
                profile,
                cfg.device,
                cfg.batch_size,
                max_seq,
                local_kv_heads,
                cfg.total_shards,
                cfg.first_layer,
                last_layer,
                cfg.kv_precision,
                cfg.mtp_enabled);
        dev_plan.kv_cache_bytes =
            persistent_state.kv_cache_bytes;
        dev_plan.live_recurrent_state_bytes =
            persistent_state.live_recurrent_state_bytes;
        dev_plan.checkpoint_state_bytes =
            persistent_state.checkpoint_state_bytes;
        dev_plan.persistent_state_bytes =
            persistent_state.stateBytes();

        // Activation estimation
        int local_n_heads = profile.n_heads;
        if (cfg.total_shards > 1)
        {
            local_n_heads = std::max(1, profile.n_heads / cfg.total_shards);
        }
        int local_d_ff = profile.d_ff;
        if (cfg.total_shards > 1)
        {
            local_d_ff = std::max(1, profile.d_ff / cfg.total_shards);
        }

        dev_plan.activation_bytes = ActivationMemoryEstimator::estimate(
            profile,
            cfg.batch_size,
            activation_seq,
            local_d_ff,
            local_n_heads,
            local_kv_heads,
            cfg.first_layer,
            last_layer,
            cfg.total_shards,
            cfg.device);

        // Workspace estimation
        dev_plan.workspace_bytes = WorkspaceMemoryEstimator::estimate(
            profile,
            cfg.batch_size,
            activation_seq,
            local_d_ff,
            cfg.first_layer,
            last_layer,
            cfg.total_shards,
            cfg.device);

        // Diagnostics
        if (!dev_plan.fits())
        {
            std::ostringstream msg;
            msg << cfg.device.to_string() << ": need "
                << formatMB(dev_plan.total_bytes() + dev_plan.headroom_bytes)
                << " but only " << formatMB(dev_plan.device_free_bytes) << " available"
                << " (deficit: " << formatMB(dev_plan.deficit()) << ")";
            result.diagnostics.push_back(msg.str());
        }
        else if (dev_plan.remaining() < 256ULL * 1024 * 1024)
        {
            std::ostringstream msg;
            msg << cfg.device.to_string() << ": tight fit — only "
                << formatMB(dev_plan.remaining()) << " remaining after headroom";
            result.diagnostics.push_back(msg.str());
        }

        result.devices.push_back(std::move(dev_plan));
    }

    return result;
}

ResidentGraphMemoryPlan MemoryPlanner::planLargestFittingResidentGraphRows(
    const ModelMemoryProfile& profile,
    const std::vector<DevicePlanConfig>& device_configs,
    const std::vector<int>& candidate_rows)
{
    ResidentGraphMemoryPlan selection;

    const bool has_gpu = std::any_of(
        device_configs.begin(),
        device_configs.end(),
        [](const DevicePlanConfig& config)
        {
            return config.device.is_gpu();
        });
    if (!has_gpu)
    {
        selection.memory_plan = plan(profile, device_configs);
        selection.resident_graph_rows =
            device_configs.empty()
                ? 0
                : std::max(
                      1,
                      device_configs.front().activation_seq_len > 0
                          ? device_configs.front().activation_seq_len
                          : device_configs.front().max_seq_len);
        return selection;
    }

    int common_max_rows = std::numeric_limits<int>::max();
    for (const auto& config : device_configs)
    {
        if (!config.device.is_gpu())
            continue;
        const int max_rows =
            config.max_seq_len > 0 ? config.max_seq_len : profile.max_seq_len;
        if (max_rows > 0)
            common_max_rows = std::min(common_max_rows, max_rows);
    }

    std::vector<int> candidates;
    candidates.reserve(candidate_rows.size() + 1);
    for (int rows : candidate_rows)
    {
        if (rows > 0 && rows <= common_max_rows)
            candidates.push_back(rows);
    }
    /*
     * A context shorter than the smallest configured bucket still needs one
     * exact graph shape. Otherwise do not invent a graph geometry merely
     * because the KV horizon is larger than the configured bucket inventory.
     */
    if (candidates.empty() && common_max_rows > 0)
        candidates.push_back(common_max_rows);
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    if (candidates.empty())
    {
        selection.memory_plan = plan(profile, device_configs);
        return selection;
    }

    /*
     * Descending evaluation makes the first successful plan the most
     * economical throughput choice that respects the measured memory
     * contract. Keep the smallest failed plan for a useful hard-failure
     * diagnostic when even the minimum captured graph cannot be resident.
     */
    for (auto candidate = candidates.rbegin(); candidate != candidates.rend(); ++candidate)
    {
        std::vector<DevicePlanConfig> evaluated = device_configs;
        for (auto& config : evaluated)
        {
            if (config.device.is_gpu())
                config.activation_seq_len = *candidate;
        }

        MemoryPlan candidate_plan = plan(profile, evaluated);
        selection.resident_graph_rows = *candidate;
        selection.memory_plan = std::move(candidate_plan);
        if (selection.memory_plan.fits())
            return selection;
    }

    return selection;
}

std::string MemoryPlan::renderTable() const
{
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);

    // Header
    table << fort::header
          << "Device" << "Context" << "Act.Seq" << "Weights" << "KV Cache" << "State"
          << "Activ." << "Wkspace" << "Total" << "Avail." << "OK"
          << fort::endr;

    // Column alignments
    table.column(0).set_cell_text_align(fort::text_align::left);
    for (int c = 1; c <= 9; ++c)
    {
        table.column(c).set_cell_text_align(fort::text_align::right);
    }
    table.column(10).set_cell_text_align(fort::text_align::center);

    // Data rows
    for (const auto& d : devices)
    {
        table << d.device.to_string()
              << d.max_seq_len
              << d.activation_seq_len
              << formatMB(d.weight_bytes)
              << formatMB(d.kv_cache_bytes)
              << formatMB(d.persistent_state_bytes)
              << formatMB(d.activation_bytes)
              << formatMB(d.workspace_bytes)
              << formatMB(d.total_bytes())
              << formatMB(d.device_free_bytes)
              << (d.fits() ? "\xe2\x9c\x93" : "\xe2\x9c\x97")  // ✓ / ✗
              << fort::endr;
    }

    std::string output = table.to_string();

    // Append diagnostics
    if (!diagnostics.empty())
    {
        output += "\n";
        for (const auto& diag : diagnostics)
        {
            output += "  " + diag + "\n";
        }
    }

    return output;
}

} // namespace llaminar2
