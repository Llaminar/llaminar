#include "planning/MemoryPlanner.h"
#include "planning/WeightMemoryEstimator.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/ActivationMemoryEstimator.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "utils/Logger.h"

#include "fort.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>

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
        int n_layers_local = last_layer - cfg.first_layer + 1;
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

        // Weight estimation
        auto weight_est = WeightMemoryEstimator::estimate(
            profile, cfg.device,
            cfg.shard_index, cfg.total_shards,
            cfg.first_layer, last_layer);
        dev_plan.weight_bytes = weight_est.device_bytes;

        // KV cache estimation
        dev_plan.kv_cache_bytes = KVCacheMemoryEstimator::estimate(
            n_layers_local, cfg.batch_size, max_seq,
            local_kv_heads, profile.head_dim,
            cfg.kv_precision, cfg.device);

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
            cfg.batch_size, activation_seq,
            profile.d_model, local_d_ff,
            local_n_heads, local_kv_heads,
            profile.head_dim, profile.vocab_size,
            cfg.device);

        // Workspace estimation
        dev_plan.workspace_bytes = WorkspaceMemoryEstimator::estimate(
            cfg.batch_size, activation_seq,
            profile.d_model, local_d_ff,
            profile.vocab_size, cfg.device);

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

std::string MemoryPlan::renderTable() const
{
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);

    // Header
    table << fort::header
          << "Device" << "Context" << "Act.Seq" << "Weights" << "KV Cache" << "Activ."
          << "Wkspace" << "Total" << "Avail." << "OK"
          << fort::endr;

    // Column alignments
    table.column(0).set_cell_text_align(fort::text_align::left);
    for (int c = 1; c <= 8; ++c)
    {
        table.column(c).set_cell_text_align(fort::text_align::right);
    }
    table.column(9).set_cell_text_align(fort::text_align::center);

    // Data rows
    for (const auto& d : devices)
    {
        table << d.device.to_string()
              << d.max_seq_len
              << d.activation_seq_len
              << formatMB(d.weight_bytes)
              << formatMB(d.kv_cache_bytes)
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
