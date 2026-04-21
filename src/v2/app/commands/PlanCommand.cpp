/**
 * @file PlanCommand.cpp
 * @brief 'llaminar plan' — cluster analysis and execution plan generation.
 *
 * Lightweight command that reads the GGUF model header (no weight data),
 * gathers the cluster device inventory, runs the MemoryPlanner to evaluate
 * candidate strategies, and outputs a memory breakdown table + YAML config.
 */

#include "app/commands/PlanCommand.h"
#include "config/CliSpec.h"
#include "utils/Logger.h"
#include "loaders/ModelLoader.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/MemoryPlanner.h"
#include "planning/ClusterInventoryGatherer.h"
#include "backends/DeviceAddressAdapter.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace llaminar2
{

    namespace
    {
        struct PlanConfig
        {
            bool show_help = false;
            std::string model_path;
            std::string strategy = "auto";
            std::string output_file;
            std::string format = "table";
            std::string kv_precision = "fp16";
            int max_seq_len = 0;   // 0 = use model default
            int batch_size = 1;
            int headroom_mb = 128;
        };

        CliSpec<PlanConfig> buildPlanSpec()
        {
            CliSpec<PlanConfig> spec;
            spec.addCategory("Model");
            spec.addCategory("Strategy");
            spec.addCategory("Output");

            spec.add({
                .short_name  = "-h",
                .long_name   = "--help",
                .category    = "Output",
                .description = "Show this help message",
                .setter      = setters::assignBoolTrue(&PlanConfig::show_help),
            });
            spec.add({
                .short_name  = "-m",
                .long_name   = "--model",
                .category    = "Model",
                .value_label = "<path>",
                .description = "Path to GGUF model file (required)",
                .setter      = setters::assignString(&PlanConfig::model_path),
            });
            spec.add({
                .short_name  = "-s",
                .long_name   = "--strategy",
                .category    = "Strategy",
                .value_label = "<type>",
                .description = "Placement strategy: auto, single-gpu, tp, pp, hybrid, cpu-only",
                .valid_values = {"auto", "single-gpu", "tp", "pp", "hybrid", "cpu-only"},
                .setter      = setters::assignString(&PlanConfig::strategy),
            });
            spec.add({
                .long_name   = "--max-seq-len",
                .category    = "Model",
                .value_label = "<n>",
                .description = "Context length for KV cache sizing (default: model max)",
                .setter      = setters::parseInt(&PlanConfig::max_seq_len, "max-seq-len"),
            });
            spec.add({
                .long_name   = "--batch-size",
                .category    = "Model",
                .value_label = "<n>",
                .description = "Batch size (default: 1)",
                .setter      = setters::parseInt(&PlanConfig::batch_size, "batch-size"),
            });
            spec.add({
                .long_name   = "--kv-precision",
                .category    = "Model",
                .value_label = "<type>",
                .description = "KV cache precision: fp16, fp32, q8_1, auto",
                .setter      = setters::assignString(&PlanConfig::kv_precision),
            });
            spec.add({
                .long_name   = "--headroom",
                .category    = "Strategy",
                .value_label = "<mb>",
                .description = "Reserved headroom per device in MB (default: 128)",
                .setter      = setters::parseInt(&PlanConfig::headroom_mb, "headroom"),
            });
            spec.add({
                .short_name  = "-o",
                .long_name   = "--output",
                .category    = "Output",
                .value_label = "<file>",
                .description = "Write plan to file (default: stdout)",
                .setter      = setters::assignString(&PlanConfig::output_file),
            });
            spec.add({
                .long_name    = "--format",
                .category     = "Output",
                .value_label  = "<fmt>",
                .description  = "Output format: table, yaml, json",
                .valid_values = {"table", "yaml", "json"},
                .setter       = setters::assignString(&PlanConfig::format),
            });

            return spec;
        }

        /// Build a YAML plan string from a MemoryPlan and PlanConfig.
        std::string buildYAML(const PlanConfig& cfg, const ModelMemoryProfile& profile,
                              const MemoryPlan& plan, const std::string& strategy_label)
        {
            auto mb = [](size_t bytes) -> int
            {
                return static_cast<int>(bytes / (1024ULL * 1024));
            };

            std::ostringstream yaml;
            yaml << "# Llaminar execution plan\n"
                 << "# Generated by: llaminar2 plan\n"
                 << "# Model: " << cfg.model_path << "\n"
                 << "# Architecture: " << profile.architecture << "\n"
                 << "# Layers: " << profile.n_layers
                 << ", d_model: " << profile.d_model
                 << ", vocab: " << profile.vocab_size << "\n"
                 << "# Strategy: " << strategy_label << "\n"
                 << "\n"
                 << "model_path: \"" << cfg.model_path << "\"\n"
                 << "strategy: " << strategy_label << "\n";

            int seq = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;
            yaml << "max_seq_len: " << seq << "\n"
                 << "batch_size: " << cfg.batch_size << "\n"
                 << "kv_precision: " << cfg.kv_precision << "\n"
                 << "\n";

            if (plan.devices.size() == 1)
            {
                const auto& d = plan.devices[0];
                yaml << "device: " << d.device.to_string() << "\n";
                yaml << "memory:\n"
                     << "  weights_mb: " << mb(d.weight_bytes) << "\n"
                     << "  kv_cache_mb: " << mb(d.kv_cache_bytes) << "\n"
                     << "  activations_mb: " << mb(d.activation_bytes) << "\n"
                     << "  workspace_mb: " << mb(d.workspace_bytes) << "\n"
                     << "  total_mb: " << mb(d.total_bytes()) << "\n"
                     << "  device_free_mb: " << mb(d.device_free_bytes) << "\n"
                     << "  headroom_mb: " << mb(d.remaining()) << "\n";
            }
            else
            {
                yaml << "devices:\n";
                for (size_t i = 0; i < plan.devices.size(); ++i)
                {
                    const auto& d = plan.devices[i];
                    yaml << "  - id: " << d.device.to_string() << "\n"
                         << "    weights_mb: " << mb(d.weight_bytes) << "\n"
                         << "    kv_cache_mb: " << mb(d.kv_cache_bytes) << "\n"
                         << "    activations_mb: " << mb(d.activation_bytes) << "\n"
                         << "    workspace_mb: " << mb(d.workspace_bytes) << "\n"
                         << "    total_mb: " << mb(d.total_bytes()) << "\n"
                         << "    device_free_mb: " << mb(d.device_free_bytes) << "\n";
                }
            }

            return yaml.str();
        }

        /// Build DevicePlanConfig for a single-device strategy.
        DevicePlanConfig buildSingleDeviceConfig(
            const PlanConfig& cfg, const ModelMemoryProfile& profile,
            const DeviceInfo& gpu, DeviceId device)
        {
            DevicePlanConfig dc;
            dc.device = device;
            dc.device_total_bytes = gpu.memory_bytes;
            dc.device_free_bytes = gpu.free_memory_bytes;
            dc.first_layer = 0;
            dc.last_layer = profile.n_layers - 1;
            dc.batch_size = cfg.batch_size;
            dc.max_seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;
            dc.kv_precision = cfg.kv_precision;
            dc.headroom_bytes = static_cast<size_t>(cfg.headroom_mb) * 1024ULL * 1024;
            return dc;
        }

        /// Build DevicePlanConfigs for TP across N GPUs.
        std::vector<DevicePlanConfig> buildTPConfigs(
            const PlanConfig& cfg, const ModelMemoryProfile& profile,
            const std::vector<DeviceInfo>& gpus, int tp_degree)
        {
            std::vector<DevicePlanConfig> configs;
            int actual_tp = std::min(tp_degree, static_cast<int>(gpus.size()));

            for (int i = 0; i < actual_tp; ++i)
            {
                DevicePlanConfig dc;
                dc.device = DeviceId(gpus[i].type, gpus[i].local_device_id);
                dc.device_total_bytes = gpus[i].memory_bytes;
                dc.device_free_bytes = gpus[i].free_memory_bytes;
                dc.shard_index = i;
                dc.total_shards = actual_tp;
                dc.first_layer = 0;
                dc.last_layer = profile.n_layers - 1;
                dc.batch_size = cfg.batch_size;
                dc.max_seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;
                dc.kv_precision = cfg.kv_precision;
                dc.headroom_bytes = static_cast<size_t>(cfg.headroom_mb) * 1024ULL * 1024;
                configs.push_back(dc);
            }
            return configs;
        }

        /// Build DevicePlanConfigs for PP across N GPUs (equal layer split).
        std::vector<DevicePlanConfig> buildPPConfigs(
            const PlanConfig& cfg, const ModelMemoryProfile& profile,
            const std::vector<DeviceInfo>& gpus, int pp_degree)
        {
            std::vector<DevicePlanConfig> configs;
            int actual_pp = std::min(pp_degree, static_cast<int>(gpus.size()));
            int layers_per_stage = profile.n_layers / actual_pp;
            int remainder = profile.n_layers % actual_pp;

            int layer_offset = 0;
            for (int i = 0; i < actual_pp; ++i)
            {
                int stage_layers = layers_per_stage + (i < remainder ? 1 : 0);
                DevicePlanConfig dc;
                dc.device = DeviceId(gpus[i].type, gpus[i].local_device_id);
                dc.device_total_bytes = gpus[i].memory_bytes;
                dc.device_free_bytes = gpus[i].free_memory_bytes;
                dc.first_layer = layer_offset;
                dc.last_layer = layer_offset + stage_layers - 1;
                dc.batch_size = cfg.batch_size;
                dc.max_seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;
                dc.kv_precision = cfg.kv_precision;
                dc.headroom_bytes = static_cast<size_t>(cfg.headroom_mb) * 1024ULL * 1024;
                configs.push_back(dc);
                layer_offset += stage_layers;
            }
            return configs;
        }

        /// Build DevicePlanConfig for CPU-only.
        DevicePlanConfig buildCPUConfig(
            const PlanConfig& cfg, const ModelMemoryProfile& profile,
            size_t cpu_memory_bytes)
        {
            DevicePlanConfig dc;
            dc.device = DeviceId::cpu();
            dc.device_total_bytes = cpu_memory_bytes;
            dc.device_free_bytes = cpu_memory_bytes;
            dc.first_layer = 0;
            dc.last_layer = profile.n_layers - 1;
            dc.batch_size = cfg.batch_size;
            dc.max_seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;
            dc.kv_precision = cfg.kv_precision;
            dc.headroom_bytes = static_cast<size_t>(cfg.headroom_mb) * 1024ULL * 1024;
            return dc;
        }

        struct StrategyCandidate
        {
            std::string name;
            std::vector<DevicePlanConfig> configs;
        };

    } // anonymous namespace

    int PlanCommand::execute(int argc, char *argv[])
    {
        initializeLogging();

        auto spec = buildPlanSpec();
        PlanConfig cfg;

        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i)
            args.emplace_back(argv[i]);

        try
        {
            spec.parse(args, cfg);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << "\n\n";
            std::cout << spec.getHelpText("Usage: llaminar2 plan [options]") << std::endl;
            return 1;
        }

        if (cfg.show_help)
        {
            std::cout << spec.getHelpText(
                             "Usage: llaminar2 plan -m <model> [options]\n\n"
                             "Analyze the cluster and produce an execution plan.\n"
                             "The plan can be consumed via: llaminar2 serve --config <plan.yaml>")
                      << std::endl;
            return 0;
        }

        if (cfg.model_path.empty())
        {
            std::cerr << "Error: --model/-m is required for 'plan'.\n";
            return 1;
        }

        // --- Step 1: Load GGUF header (metadata only, no weight data) ---
        ModelLoader loader(nullptr);  // creates internal factory
        loader.setUseMmap(false);
        if (!loader.loadModel(cfg.model_path))
        {
            std::cerr << "Error: Failed to load model: " << cfg.model_path << "\n";
            return 1;
        }

        auto profile = ModelMemoryProfile::fromGGUF(loader.getModel());
        int seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;

        // --- Step 2: Gather cluster inventory (local-only, no MPI) ---
        auto inventory = gatherClusterInventory(nullptr);

        // Collect available GPUs
        std::vector<DeviceInfo> gpus;
        size_t cpu_memory = 0;
        if (!inventory.ranks.empty())
        {
            gpus = inventory.ranks[0].gpus;
            cpu_memory = inventory.ranks[0].cpu_memory_bytes;
            if (cpu_memory == 0)
            {
                cpu_memory = 64ULL * 1024 * 1024 * 1024; // fallback: 64 GB
            }
        }

        // --- Step 3: Build and evaluate strategies ---
        std::vector<StrategyCandidate> candidates;

        if (cfg.strategy == "auto")
        {
            // Enumerate candidates in preference order: single-gpu > tp > pp > cpu-only
            if (!gpus.empty())
            {
                // Single-GPU on the largest GPU
                auto best_gpu = *std::max_element(gpus.begin(), gpus.end(),
                    [](const DeviceInfo& a, const DeviceInfo& b)
                    { return a.free_memory_bytes < b.free_memory_bytes; });
                DeviceId best_id(best_gpu.type, best_gpu.local_device_id);
                candidates.push_back({"single-gpu",
                    {buildSingleDeviceConfig(cfg, profile, best_gpu, best_id)}});

                // TP across all GPUs
                if (gpus.size() >= 2)
                {
                    candidates.push_back({"tp-" + std::to_string(gpus.size()),
                        buildTPConfigs(cfg, profile, gpus, static_cast<int>(gpus.size()))});
                }

                // PP across 2 GPUs
                if (gpus.size() >= 2)
                {
                    candidates.push_back({"pp-2",
                        buildPPConfigs(cfg, profile, gpus, 2)});
                }

                // PP across all GPUs
                if (gpus.size() > 2)
                {
                    candidates.push_back({"pp-" + std::to_string(gpus.size()),
                        buildPPConfigs(cfg, profile, gpus, static_cast<int>(gpus.size()))});
                }
            }

            // CPU-only fallback
            candidates.push_back({"cpu-only",
                {buildCPUConfig(cfg, profile, cpu_memory)}});
        }
        else if (cfg.strategy == "single-gpu")
        {
            if (gpus.empty())
            {
                std::cerr << "Error: No GPUs available for single-gpu strategy.\n";
                return 1;
            }
            auto best_gpu = *std::max_element(gpus.begin(), gpus.end(),
                [](const DeviceInfo& a, const DeviceInfo& b)
                { return a.free_memory_bytes < b.free_memory_bytes; });
            DeviceId best_id(best_gpu.type, best_gpu.local_device_id);
            candidates.push_back({"single-gpu",
                {buildSingleDeviceConfig(cfg, profile, best_gpu, best_id)}});
        }
        else if (cfg.strategy == "tp")
        {
            if (gpus.size() < 2)
            {
                std::cerr << "Error: Need at least 2 GPUs for tp strategy.\n";
                return 1;
            }
            candidates.push_back({"tp-" + std::to_string(gpus.size()),
                buildTPConfigs(cfg, profile, gpus, static_cast<int>(gpus.size()))});
        }
        else if (cfg.strategy == "pp")
        {
            if (gpus.size() < 2)
            {
                std::cerr << "Error: Need at least 2 GPUs for pp strategy.\n";
                return 1;
            }
            candidates.push_back({"pp-" + std::to_string(gpus.size()),
                buildPPConfigs(cfg, profile, gpus, static_cast<int>(gpus.size()))});
        }
        else if (cfg.strategy == "hybrid")
        {
            if (gpus.size() < 2)
            {
                std::cerr << "Error: Need at least 2 GPUs for hybrid strategy.\n";
                return 1;
            }
            // Hybrid: TP across all GPUs (PP+TP would need more config)
            candidates.push_back({"hybrid-tp" + std::to_string(gpus.size()),
                buildTPConfigs(cfg, profile, gpus, static_cast<int>(gpus.size()))});
        }
        else if (cfg.strategy == "cpu-only")
        {
            candidates.push_back({"cpu-only",
                {buildCPUConfig(cfg, profile, cpu_memory)}});
        }

        // Evaluate each candidate
        MemoryPlan best_plan;
        std::string best_strategy;
        bool found_feasible = false;

        for (const auto& candidate : candidates)
        {
            auto plan = MemoryPlanner::plan(profile, candidate.configs);
            if (plan.fits() && !found_feasible)
            {
                best_plan = std::move(plan);
                best_strategy = candidate.name;
                found_feasible = true;
            }
            else if (!found_feasible)
            {
                // Keep the last one if nothing fits (for diagnostics)
                best_plan = std::move(plan);
                best_strategy = candidate.name;
            }
        }

        // --- Step 4: Output results ---
        std::ostringstream output;

        // Header
        output << "=== Memory Plan ===\n";
        output << "Model: " << cfg.model_path << "\n";
        output << "Architecture: " << profile.architecture
               << " (" << profile.n_layers << " layers, d=" << profile.d_model
               << ", vocab=" << profile.vocab_size << ")\n";
        output << "Strategy: " << best_strategy
               << (found_feasible ? "" : " [DOES NOT FIT]") << "\n";
        output << "Context: " << seq_len << " tokens, batch=" << cfg.batch_size
               << ", kv=" << cfg.kv_precision << "\n\n";

        // Memory table
        output << best_plan.renderTable();

        // YAML section (always appended unless format is table-only)
        if (cfg.format == "yaml" || cfg.format == "json")
        {
            output << "\n--- plan.yaml ---\n";
            output << buildYAML(cfg, profile, best_plan, best_strategy);
        }

        // Emit output
        if (!cfg.output_file.empty())
        {
            std::ofstream out(cfg.output_file);
            if (!out.is_open())
            {
                std::cerr << "Error: Cannot write to " << cfg.output_file << "\n";
                return 1;
            }
            // Write YAML to file regardless of format flag
            out << buildYAML(cfg, profile, best_plan, best_strategy);
            std::cout << output.str();
            std::cout << "\nPlan written to: " << cfg.output_file << std::endl;
        }
        else
        {
            std::cout << output.str();
        }

        return found_feasible ? 0 : 1;
    }

} // namespace llaminar2
