/**
 * @file OrchestrationConfigParser.cpp
 * @brief Implementation of OrchestrationConfigParser
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "OrchestrationConfigParser.h"
#include "ParallelismTreeParser.h"          // For --topology parsing
#include "execution/config/RuntimeConfig.h" // For parseFusedAttentionBackend
#include "utils/Logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <set>

namespace llaminar2
{

    // =========================================================================
    // Factory function
    // =========================================================================

    std::unique_ptr<IOrchestrationConfigParser> createOrchestrationConfigParser()
    {
        return std::make_unique<OrchestrationConfigParser>();
    }

    // =========================================================================
    // Helper functions
    // =========================================================================

    namespace
    {
        std::string trim(const std::string &str)
        {
            size_t start = str.find_first_not_of(" \t\n\r");
            if (start == std::string::npos)
                return "";
            size_t end = str.find_last_not_of(" \t\n\r");
            return str.substr(start, end - start + 1);
        }

        std::vector<std::string> split(const std::string &str, char delim)
        {
            std::vector<std::string> parts;
            std::stringstream ss(str);
            std::string part;
            while (std::getline(ss, part, delim))
            {
                std::string trimmed = trim(part);
                if (!trimmed.empty())
                {
                    parts.push_back(trimmed);
                }
            }
            return parts;
        }

        std::string toLower(const std::string &str)
        {
            std::string lower = str;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            return lower;
        }
    } // anonymous namespace

    // =========================================================================
    // Static helpers
    // =========================================================================

    std::vector<GlobalDeviceAddress> OrchestrationConfigParser::parseDeviceList(const std::string &spec)
    {
        std::vector<GlobalDeviceAddress> devices;
        auto parts = split(spec, ',');

        for (const auto &part : parts)
        {
            auto addr = GlobalDeviceAddress::tryParse(part);
            if (!addr)
            {
                throw std::invalid_argument("Invalid device specification: '" + part + "'");
            }
            devices.push_back(*addr);
        }

        return devices;
    }

    std::vector<float> OrchestrationConfigParser::parseWeightList(const std::string &spec)
    {
        std::vector<float> weights;
        auto parts = split(spec, ',');

        for (const auto &part : parts)
        {
            try
            {
                float w = std::stof(part);
                weights.push_back(w);
            }
            catch (const std::exception &)
            {
                throw std::invalid_argument("Invalid weight value: '" + part + "'");
            }
        }

        return weights;
    }

    std::vector<std::pair<int, GlobalDeviceAddress>> OrchestrationConfigParser::parseDeviceMap(
        const std::string &spec)
    {
        std::vector<std::pair<int, GlobalDeviceAddress>> device_map;
        auto parts = split(spec, ',');

        for (const auto &part : parts)
        {
            size_t eq_pos = part.find('=');
            if (eq_pos == std::string::npos)
            {
                throw std::invalid_argument(
                    "Invalid device map entry: '" + part + "'. Expected format: rank=device");
            }

            int rank;
            try
            {
                rank = std::stoi(part.substr(0, eq_pos));
            }
            catch (const std::exception &)
            {
                throw std::invalid_argument("Invalid rank in device map: '" + part.substr(0, eq_pos) + "'");
            }

            std::string device_spec = part.substr(eq_pos + 1);
            std::string lower = toLower(device_spec);

            if (lower == "cpu")
            {
                device_map.emplace_back(rank, GlobalDeviceAddress::cpu(0));
            }
            else if (lower.rfind("cpu:", 0) == 0)
            {
                try
                {
                    int numa = std::stoi(device_spec.substr(4));
                    if (numa < 0)
                    {
                        throw std::invalid_argument("negative NUMA");
                    }
                    device_map.emplace_back(rank, GlobalDeviceAddress::cpu(numa));
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid CPU device in device map: '" + device_spec + "' (expected cpu or cpu:<numa>)");
                }
            }
            else
            {
                auto addr = GlobalDeviceAddress::tryParse(device_spec);
                if (!addr)
                {
                    throw std::invalid_argument("Invalid device in device map: '" + device_spec + "'");
                }

                device_map.emplace_back(rank, *addr);
            }
        }

        return device_map;
    }

    namespace
    {
        std::vector<std::pair<int, bool>> parseDeviceMapNumaExplicit(const std::string &spec)
        {
            std::vector<std::pair<int, bool>> explicitness;
            auto parts = split(spec, ',');

            for (const auto &part : parts)
            {
                size_t eq_pos = part.find('=');
                if (eq_pos == std::string::npos)
                {
                    throw std::invalid_argument(
                        "Invalid device map entry: '" + part + "'. Expected format: rank=device");
                }

                int rank;
                try
                {
                    rank = std::stoi(part.substr(0, eq_pos));
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid rank in device map: '" + part.substr(0, eq_pos) + "'");
                }

                const std::string device_spec = part.substr(eq_pos + 1);
                const std::string lower = toLower(device_spec);

                if (lower.rfind("cpu:", 0) == 0)
                {
                    explicitness.emplace_back(rank, true);
                }
                else
                {
                    const size_t colon_count = static_cast<size_t>(std::count(device_spec.begin(), device_spec.end(), ':'));
                    explicitness.emplace_back(rank, colon_count >= 2);
                }
            }

            return explicitness;
        }
    }

    bool OrchestrationConfigParser::matchesFlag(const std::string &arg,
                                                const std::string &short_form,
                                                const std::string &long_form)
    {
        // Check exact match
        if (arg == short_form || arg == long_form)
        {
            return true;
        }

        // Check --flag=value format
        if (!long_form.empty() && arg.substr(0, long_form.size() + 1) == long_form + "=")
        {
            return true;
        }

        return false;
    }

    std::string OrchestrationConfigParser::getFlagValue(const std::vector<std::string> &args, size_t &idx)
    {
        const std::string &arg = args[idx];

        // Check --flag=value format
        size_t eq_pos = arg.find('=');
        if (eq_pos != std::string::npos)
        {
            return arg.substr(eq_pos + 1);
        }

        // Check --flag value format
        if (idx + 1 < args.size() && !args[idx + 1].empty() && args[idx + 1][0] != '-')
        {
            ++idx;
            return args[idx];
        }

        return "";
    }

    // =========================================================================
    // parseArgs
    // =========================================================================

    OrchestrationConfig OrchestrationConfigParser::parseArgs(int argc, char **argv)
    {
        OrchestrationConfig config;

        // Convert to vector for easier handling
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }

        for (size_t i = 0; i < args.size(); ++i)
        {
            const std::string &arg = args[i];

            // ===== Help and info flags =====
            if (arg == "-h" || arg == "--help")
            {
                config.show_help = true;
            }
            else if (arg == "-v")
            {
                config.verbose_level++;
            }
            else if (arg == "-vv")
            {
                config.verbose_level = 2;
            }
            else if (arg == "-vvv")
            {
                config.verbose_level = 3;
            }
            else if (arg == "--list-devices")
            {
                config.list_devices = true;
            }

            // ===== Introspection flags =====
            else if (arg == "--dry-run")
            {
                config.dry_run = true;
            }
            else if (arg == "--explain-placement")
            {
                config.explain_placement = true;
            }
            else if (arg == "--show-topology")
            {
                config.show_topology = true;
            }
            else if (arg == "--show-numa")
            {
                config.show_numa = true;
            }
            else if (arg == "--validate-only")
            {
                config.validate_only = true;
            }

            // ===== Model Configuration =====
            else if (matchesFlag(arg, "-m", "--model"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--model requires a path");
                }
                config.model_path = value;
            }
            else if (matchesFlag(arg, "-c", "--context-length"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--context-length requires a value");
                }
                try
                {
                    config.max_seq_len = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid context length: '" + value + "'");
                }
            }
            else if (arg == "--mmap")
            {
                config.use_mmap = true;
            }
            else if (arg == "--no-mmap")
            {
                config.use_mmap = false;
            }

            // ===== Inference Configuration =====
            else if (matchesFlag(arg, "-p", "--prompt"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--prompt requires a value");
                }
                config.prompt = value;
            }
            else if (matchesFlag(arg, "-n", "--n-predict"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--n-predict requires a value");
                }
                try
                {
                    config.n_predict = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid n-predict value: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--batch-size"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--batch-size requires a value");
                }
                try
                {
                    config.batch_size = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid batch size: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--threads"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--threads requires a value");
                }
                try
                {
                    config.n_threads = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid threads value: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "-s", "--seed"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--seed requires a value");
                }
                try
                {
                    config.seed = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid seed value: '" + value + "'");
                }
            }

            // ===== Sampling Configuration =====
            else if (matchesFlag(arg, "-t", "--temperature"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--temperature requires a value");
                }
                try
                {
                    config.temperature = std::stof(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid temperature value: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--top-k"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--top-k requires a value");
                }
                try
                {
                    config.top_k = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid top-k value: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--top-p"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--top-p requires a value");
                }
                try
                {
                    config.top_p = std::stof(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid top-p value: '" + value + "'");
                }
            }
            else if (arg == "--deterministic")
            {
                config.deterministic = true;
            }

            // ===== Chat Configuration =====
            else if (arg == "--chat")
            {
                config.chat_mode = true;
            }
            else if (arg == "--chat-single")
            {
                config.single_shot_chat = true;
            }
            else if (matchesFlag(arg, "", "--system"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--system requires a value");
                }
                config.system_prompt = value;
            }
            else if (matchesFlag(arg, "", "--chat-template"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--chat-template requires a value");
                }
                config.chat_template_override = value;
            }

            // ===== Benchmark Configuration =====
            else if (arg == "--benchmark")
            {
                config.benchmark_mode = true;
            }

            // ===== Server Configuration =====
            else if (arg == "--serve")
            {
                config.serve_mode = true;
            }
            else if (matchesFlag(arg, "", "--port"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                    throw std::invalid_argument("--port requires a value");
                try
                {
                    config.serve_port = std::stoi(value);
                }
                catch (...)
                {
                    throw std::invalid_argument("Invalid port value: " + value);
                }
            }
            else if (matchesFlag(arg, "", "--host"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                    throw std::invalid_argument("--host requires a value");
                config.serve_host = value;
            }

            // ===== Fused Attention Configuration =====
            else if (arg == "--fused-attention")
            {
                config.use_fused_attention = true;
            }
            else if (matchesFlag(arg, "", "--fused-attention-backend"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--fused-attention-backend requires a value");
                }
                config.fused_attention_backend = parseFusedAttentionBackend(value);
            }

            // ===== MPI Bootstrap Configuration =====
            else if (matchesFlag(arg, "", "--mpi-procs"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--mpi-procs requires a value");
                }
                try
                {
                    config.mpi_procs = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid mpi-procs value: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--hostfile"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--hostfile requires a path");
                }
                config.hostfile = value;
            }
            else if (arg == "--mpi-dry-run")
            {
                config.mpi_dry_run = true;
            }
            else if (arg == "--mpi-verbose")
            {
                config.mpi_verbose = true;
            }
            else if (arg == "--no-mpi-bootstrap")
            {
                config.mpi_no_bootstrap = true;
            }
            else if (arg == "--mpi-oversubscribe")
            {
                config.mpi_oversubscribe = true;
            }
            else if (matchesFlag(arg, "", "--mpi-profile"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--mpi-profile requires a value (auto|tuned)");
                }
                auto profile = parseMPIProfile(value);
                if (!profile)
                {
                    throw std::invalid_argument("Invalid mpi profile: '" + value + "' (valid: auto, tuned)");
                }
                config.mpi_profile = *profile;
            }

            // ===== Device assignment =====
            else if (matchesFlag(arg, "-d", "--device"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--device requires a device specification");
                }
                const std::string lower = toLower(value);

                // CPU shorthand semantics:
                // - "cpu"   => all local NUMA CPU devices (global CPU TP intent)
                // - "cpu:N" => explicit NUMA-node CPU target
                if (lower == "cpu")
                {
                    config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
                    config.device_for_this_rank_numa_explicit = false;
                    config.cpu_global_tp_all_local = true;
                }
                else if (lower.rfind("cpu:", 0) == 0)
                {
                    try
                    {
                        int numa = std::stoi(value.substr(4));
                        if (numa < 0)
                        {
                            throw std::invalid_argument("negative NUMA");
                        }
                        config.device_for_this_rank = GlobalDeviceAddress::cpu(numa);
                        config.device_for_this_rank_numa_explicit = true;
                        config.cpu_global_tp_all_local = false;
                    }
                    catch (const std::exception &)
                    {
                        throw std::invalid_argument("Invalid CPU device specification: '" + value + "' (expected cpu or cpu:<numa>)");
                    }
                }
                else
                {
                    auto addr = GlobalDeviceAddress::tryParse(value);
                    if (!addr)
                    {
                        throw std::invalid_argument("Invalid device specification: '" + value + "'");
                    }
                    config.device_for_this_rank = *addr;

                    // Short form "type:ordinal" is NUMA-ambiguous; forms with >=3 fields are explicit.
                    size_t colon_count = static_cast<size_t>(std::count(value.begin(), value.end(), ':'));
                    config.device_for_this_rank_numa_explicit = (colon_count >= 2);
                    config.cpu_global_tp_all_local = false;
                }
            }
            else if (matchesFlag(arg, "", "--device-mode"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--device-mode requires a mode");
                }
                auto mode = parseDeviceAssignmentMode(value);
                if (!mode)
                {
                    throw std::invalid_argument("Invalid device mode: '" + value + "'");
                }
                config.device_mode = *mode;
            }
            else if (matchesFlag(arg, "", "--device-map"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--device-map requires a mapping");
                }
                config.device_map = parseDeviceMap(value);
                config.device_map_numa_explicit = parseDeviceMapNumaExplicit(value);
                config.device_mode = DeviceAssignmentMode::EXPLICIT;
            }

            // ===== Named domains =====
            else if (matchesFlag(arg, "", "--define-domain"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--define-domain requires a specification");
                }
                config.domain_definitions.push_back(DomainDefinition::parse(value));
            }
            else if (matchesFlag(arg, "", "--pp-stage"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--pp-stage requires a specification");
                }
                config.pp_stage_definitions.push_back(PPStageDefinition::parse(value));
            }

            // ===== Simple TP options =====
            else if (matchesFlag(arg, "-tp", "--tensor-parallelism-degree"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tensor-parallelism-degree requires a degree");
                }
                try
                {
                    config.tp_degree = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid TP degree: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--tp-scope"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-scope requires a scope");
                }
                auto scope = parseTpScope(value);
                if (!scope)
                {
                    throw std::invalid_argument("Invalid TP scope: '" + value + "'");
                }
                config.tp_scope = *scope;
            }
            else if (matchesFlag(arg, "", "--tp-devices"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-devices requires a device list");
                }
                config.tp_devices = parseDeviceList(value);
            }
            else if (matchesFlag(arg, "", "--tp-weights"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-weights requires a weight list");
                }
                config.tp_weights = parseWeightList(value);
            }
            else if (matchesFlag(arg, "", "--tp-local"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-local requires a degree");
                }
                try
                {
                    config.tp_local_degree = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid TP local degree: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--tp-global"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-global requires a degree");
                }
                try
                {
                    config.tp_global_degree = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid TP global degree: '" + value + "'");
                }
            }

            // ===== Simple PP options =====
            else if (matchesFlag(arg, "-pp", "--pipeline-parallelism-degree"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--pipeline-parallelism-degree requires a degree");
                }
                try
                {
                    config.pp_degree = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid PP degree: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--pp-split"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--pp-split requires a mode");
                }
                auto mode = parsePpSplitMode(value);
                if (!mode)
                {
                    throw std::invalid_argument("Invalid PP split mode: '" + value + "'");
                }
                config.pp_split = *mode;
            }

            // ===== Layer placement =====
            else if (matchesFlag(arg, "", "--cpu-layers"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--cpu-layers requires a count");
                }
                try
                {
                    config.cpu_layers = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid CPU layers count: '" + value + "'");
                }
            }
            else if (arg == "--cpu-layers-first")
            {
                config.cpu_layers_first = true;
            }

            // ===== Backend =====
            else if (matchesFlag(arg, "-b", "--backend"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--backend requires a type");
                }
                auto backend = parseCollectiveBackendType(value);
                if (!backend)
                {
                    throw std::invalid_argument("Invalid backend type: '" + value + "'");
                }
                config.default_backend = *backend;
            }

            // ===== Config file =====
            else if (matchesFlag(arg, "", "--config"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--config requires a file path");
                }
                config.config_file_path = value;

                // Parse the config file and merge
                OrchestrationConfig file_config = parseYamlFile(value);

                // Merge: CLI options override file options
                // Only override fields that weren't explicitly set via CLI
                // For now, just use file config as base if we got this far
                // (A more sophisticated merge would track which CLI flags were set)
                if (config.mpi_profile == MPIProfile::AUTO && file_config.mpi_profile != MPIProfile::AUTO)
                {
                    config.mpi_profile = file_config.mpi_profile;
                }
            }

            // ===== Topology Tree (Global PP Phase 8) =====
            else if (matchesFlag(arg, "", "--topology"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--topology requires an inline topology specification");
                }
                config.topology_string = value;
                // Note: Tree parsing deferred until we know model n_layers
                // The factory will parse when n_layers is available from model
            }
            else if (matchesFlag(arg, "", "--topology-file"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--topology-file requires a file path");
                }
                config.topology_file_path = value;
                // Note: Tree parsing deferred until we know model n_layers
                // The factory will parse when n_layers is available from model
            }

            // ===== Memory Constraints =====
            else if (matchesFlag(arg, "", "--max-gpu-memory"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--max-gpu-memory requires a value in MB");
                }
                config.max_gpu_memory_mb = std::stoull(value);
            }
            else if (matchesFlag(arg, "", "--max-cpu-memory"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--max-cpu-memory requires a value in MB");
                }
                config.max_cpu_memory_mb = std::stoull(value);
            }

            // ===== MoE Configuration =====
            else if (arg == "--moe-shared-gpu")
            {
                config.moe_shared_experts_gpu = true;
            }
            else if (arg == "--moe-shared-cpu")
            {
                config.moe_shared_experts_gpu = false;
            }
            else if (arg == "--moe-sparse-gpu")
            {
                config.moe_sparse_experts_cpu = false;
            }
            else if (arg == "--moe-sparse-cpu")
            {
                config.moe_sparse_experts_cpu = true;
            }

            // ===== Activation Precision =====
            else if (arg == "--activation-precision" || arg == "--activation-prec" || arg == "--act-prec")
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--activation-precision requires a value");
                }
                static const std::set<std::string> valid_precisions = {"fp32", "bf16", "fp16", "q8_1"};
                if (valid_precisions.find(value) == valid_precisions.end())
                {
                    throw std::invalid_argument("Invalid activation precision: '" + value + "'. Valid: fp32, bf16, fp16, q8_1");
                }
                config.activation_precision = value;
            }
            else if (arg == "--kv-cache-precision" || arg == "--kv-prec")
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--kv-cache-precision requires a value");
                }
                value = toLower(value);
                static const std::set<std::string> valid_precisions = {"auto", "fp32", "f32", "fp16", "f16", "q8_1", "q8", "q81"};
                if (valid_precisions.find(value) == valid_precisions.end())
                {
                    throw std::invalid_argument("Invalid KV cache precision: '" + value + "'. Valid: auto, fp32, fp16, q8_1");
                }
                config.kv_cache_precision = value;
            }

            // ===== Weight Sharding =====
            else if (arg == "--shard-weights")
            {
                config.shard_weights = true;
            }
            else if (arg == "--no-shard-weights")
            {
                config.disable_weight_sharding = true;
            }

            // ===== Heterogeneous Mode =====
            else if (arg == "--heterogeneous")
            {
                config.heterogeneous_mode = true;
            }
            else if (matchesFlag(arg, "", "--cpu-fraction"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--cpu-fraction requires a value");
                }
                config.cpu_compute_fraction = std::stof(value);
                if (config.cpu_compute_fraction < 0.0f || config.cpu_compute_fraction > 1.0f)
                {
                    throw std::invalid_argument("--cpu-fraction must be between 0.0 and 1.0");
                }
            }
            else if (arg == "--no-gpu-tp")
            {
                config.disable_gpu_tp = true;
            }
            else if (arg == "--no-cpu-tp")
            {
                config.disable_cpu_tp = true;
            }
            else if (matchesFlag(arg, "", "--min-layers-per-domain"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--min-layers-per-domain requires a value");
                }
                config.min_layers_per_domain = std::stoi(value);
                if (config.min_layers_per_domain < 1)
                {
                    throw std::invalid_argument("--min-layers-per-domain must be >= 1");
                }
            }

            // Unknown argument - throw error
            else
            {
                throw std::invalid_argument("Unknown argument: '" + arg + "'. Use --help to see available options.");
            }
        }

        // Validate heterogeneous mode
        if (config.heterogeneous_mode && config.disable_gpu_tp && config.disable_cpu_tp)
        {
            throw std::invalid_argument("Cannot use --heterogeneous with both --no-gpu-tp and --no-cpu-tp");
        }

        return config;
    }

    // =========================================================================
    // parseYamlFile
    // =========================================================================

    OrchestrationConfig OrchestrationConfigParser::parseYamlFile(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::invalid_argument("Failed to open config file: " + path);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return parseYamlString(buffer.str());
    }

    // =========================================================================
    // parseYamlString
    // =========================================================================

    OrchestrationConfig OrchestrationConfigParser::parseYamlString(const std::string &yaml)
    {
        OrchestrationConfig config;

        // Simple line-by-line YAML parser (sufficient for our flat structure)
        // For production, consider using a proper YAML library like yaml-cpp

        std::istringstream stream(yaml);
        std::string line;
        std::string current_section;

        while (std::getline(stream, line))
        {
            std::string trimmed = trim(line);

            // Skip empty lines and comments
            if (trimmed.empty() || trimmed[0] == '#')
            {
                continue;
            }

            // Check for section headers
            if (trimmed.back() == ':' && trimmed.find(':') == trimmed.size() - 1)
            {
                current_section = trimmed.substr(0, trimmed.size() - 1);
                continue;
            }

            // Parse key: value pairs
            size_t colon_pos = trimmed.find(':');
            if (colon_pos == std::string::npos)
            {
                continue;
            }

            std::string key = trim(trimmed.substr(0, colon_pos));
            std::string value = trim(trimmed.substr(colon_pos + 1));

            // Remove quotes if present
            if (value.size() >= 2 &&
                ((value.front() == '"' && value.back() == '"') ||
                 (value.front() == '\'' && value.back() == '\'')))
            {
                value = value.substr(1, value.size() - 2);
            }

            // Map YAML keys to config fields
            if (key == "dry_run")
            {
                config.dry_run = (toLower(value) == "true" || value == "1");
            }
            else if (key == "explain_placement")
            {
                config.explain_placement = (toLower(value) == "true" || value == "1");
            }
            else if (key == "show_topology")
            {
                config.show_topology = (toLower(value) == "true" || value == "1");
            }
            else if (key == "show_numa")
            {
                config.show_numa = (toLower(value) == "true" || value == "1");
            }
            else if (key == "validate_only")
            {
                config.validate_only = (toLower(value) == "true" || value == "1");
            }
            else if (key == "device_mode")
            {
                auto mode = parseDeviceAssignmentMode(value);
                if (mode)
                    config.device_mode = *mode;
            }
            else if (key == "device")
            {
                const std::string lower = toLower(value);
                if (lower == "cpu")
                {
                    config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
                    config.device_for_this_rank_numa_explicit = false;
                    config.cpu_global_tp_all_local = true;
                }
                else if (lower.rfind("cpu:", 0) == 0)
                {
                    try
                    {
                        int numa = std::stoi(value.substr(4));
                        if (numa >= 0)
                        {
                            config.device_for_this_rank = GlobalDeviceAddress::cpu(numa);
                            config.device_for_this_rank_numa_explicit = true;
                            config.cpu_global_tp_all_local = false;
                        }
                    }
                    catch (const std::exception &)
                    {
                        // Keep prior behavior for invalid entries (ignore in config-file parse path)
                    }
                }
                else
                {
                    auto addr = GlobalDeviceAddress::tryParse(value);
                    if (addr)
                    {
                        config.device_for_this_rank = *addr;
                        size_t colon_count = static_cast<size_t>(std::count(value.begin(), value.end(), ':'));
                        config.device_for_this_rank_numa_explicit = (colon_count >= 2);
                        config.cpu_global_tp_all_local = false;
                    }
                }
            }
            else if (key == "tp_degree" || key == "tp")
            {
                config.tp_degree = std::stoi(value);
            }
            else if (key == "tp_scope")
            {
                auto scope = parseTpScope(value);
                if (scope)
                    config.tp_scope = *scope;
            }
            else if (key == "tp_devices")
            {
                // Handle array format [device1, device2]
                std::string devices_str = value;
                if (!devices_str.empty() && devices_str.front() == '[')
                {
                    devices_str = devices_str.substr(1);
                    if (!devices_str.empty() && devices_str.back() == ']')
                    {
                        devices_str = devices_str.substr(0, devices_str.size() - 1);
                    }
                }
                config.tp_devices = parseDeviceList(devices_str);
            }
            else if (key == "tp_weights")
            {
                std::string weights_str = value;
                if (!weights_str.empty() && weights_str.front() == '[')
                {
                    weights_str = weights_str.substr(1);
                    if (!weights_str.empty() && weights_str.back() == ']')
                    {
                        weights_str = weights_str.substr(0, weights_str.size() - 1);
                    }
                }
                config.tp_weights = parseWeightList(weights_str);
            }
            else if (key == "tp_local_degree" || key == "tp_local")
            {
                config.tp_local_degree = std::stoi(value);
            }
            else if (key == "tp_global_degree" || key == "tp_global")
            {
                config.tp_global_degree = std::stoi(value);
            }
            else if (key == "pp_degree" || key == "pp")
            {
                config.pp_degree = std::stoi(value);
            }
            else if (key == "pp_split")
            {
                auto mode = parsePpSplitMode(value);
                if (mode)
                    config.pp_split = *mode;
            }
            else if (key == "cpu_layers")
            {
                config.cpu_layers = std::stoi(value);
            }
            else if (key == "cpu_layers_first")
            {
                config.cpu_layers_first = (toLower(value) == "true" || value == "1");
            }
            else if (key == "backend" || key == "default_backend")
            {
                auto backend = parseCollectiveBackendType(value);
                if (backend)
                    config.default_backend = *backend;
            }
            else if (key == "activation_precision")
            {
                config.activation_precision = value;
            }
            else if (key == "kv_cache_precision")
            {
                config.kv_cache_precision = value;
            }
            else if (key == "mpi_profile" || key == "mpi-profile")
            {
                auto profile = parseMPIProfile(value);
                if (profile)
                {
                    config.mpi_profile = *profile;
                }
            }
        }

        return config;
    }

    // =========================================================================
    // Help text
    // =========================================================================

    std::string OrchestrationConfigParser::getHelpText()
    {
        return R"HELPTEXT(
Llaminar V2 LLM Inference Engine

Usage: llaminar2 [OPTIONS]

Model Configuration:
  -m, --model <path>     Path to GGUF model file (required)
  -c, --context-length <n>  Maximum context/sequence length (default: 2048)
  --mmap                 Use memory-mapped file loading (default)
  --no-mmap              Disable memory-mapped file loading

Inference Configuration:
  -p, --prompt <text>    Input prompt text
  -n, --n-predict <n>    Tokens to generate (-1 = until EOS, default: -1)
  --batch-size <n>       Batch size (default: 1)
  --threads <n>          Thread count (-1 = auto, default: -1)
  -s, --seed <n>         Random seed (-1 = random, default: -1)

Sampling Configuration:
  -t, --temperature <f>  Sampling temperature (default: 0.8)
  --top-k <n>            Top-K sampling (default: 40)
  --top-p <f>            Top-P (nucleus) sampling (default: 0.9)
  --deterministic        Force deterministic mode (temperature=0)

Chat Configuration:
  --chat                 Interactive chat mode
  --chat-single          Single prompt with chat template applied
  --system <text>        System prompt for chat
  --chat-template <name> Override chat template (chatml, llama3, etc.)

Benchmark Configuration:
  --benchmark            Run benchmark (warmup + multiple timed runs)

Server Configuration:
  --serve                Start HTTP server (OpenAI-compatible REST API)
  --port <n>             Server port (default: 8080)
  --host <addr>          Server bind address (default: 127.0.0.1)

Fused Attention:
  --fused-attention      Enable fused attention+Wo kernel
  --fused-attention-backend <type>  Backend: jit (default), reference, tiled, q16

MPI Bootstrap:
  --mpi-procs <n>        Number of MPI processes (0 = auto)
  --hostfile <path>      MPI hostfile path
  --mpi-dry-run          Print MPI launch command and exit
  --mpi-verbose          Verbose MPI output
  --no-mpi-bootstrap     Disable automatic MPI bootstrap
  --mpi-oversubscribe    Allow MPI oversubscription
    --mpi-profile <mode>   MPI bootstrap profile: auto (default), tuned

Device Assignment:
  -d, --device <spec>    Device for this rank (e.g., cuda:0, rocm:0, cpu)
  --device-mode <mode>   Assignment mode: auto, local_gpu, round_robin, explicit
  --device-map <map>     Explicit mapping: "0=cuda:0,1=cuda:1"

Tensor Parallelism:
  -tp, --tensor-parallelism-degree <n>  TP parallelism degree
  --tp-scope <scope>     Scope: auto, local, global, hybrid
  --tp-devices <list>    Device list: "cuda:0,cuda:1"
  --tp-weights <list>    Weight distribution: "0.73,0.27"
  --tp-local <degree>    Local TP degree for hybrid
  --tp-global <degree>   Global TP degree for hybrid

Pipeline Parallelism:
  -pp, --pipeline-parallelism-degree <n>  PP parallelism degree
  --pp-split <mode>      Layer split: equal, weighted, manual

Layer Placement:
  --cpu-layers <n>       Number of layers on CPU
  --cpu-layers-first     Put CPU layers at beginning (default: end)

Named Domains (advanced):
  --define-domain <spec> Define domain: "name=device1,device2[;weights=w1,w2][;backend=type]"
  --pp-stage <spec>      Define PP stage: "stage_id=domain:first_layer-last_layer"

Collective Backend:
  -b, --backend <type>   Default collective: auto, nccl, rccl, pcie_bar, upi, mpi, host

Introspection:
  --dry-run              Show configuration without executing
  --explain-placement    Explain device placement decisions
  --show-topology        Show detected topology and exit
  --show-numa            Show NUMA configuration and exit
  --validate-only        Validate configuration without running

Config File:
  --config <path>        Load configuration from YAML file
                                                 (supports mpi_profile: auto|tuned)

Topology Tree (Global PP):
  --topology <spec>      Inline topology: "PP(name, Device(cpu,0), Device(cpu,0))"
  --topology-file <path> Load topology from YAML file

Memory Constraints:
  --max-gpu-memory <mb>  Maximum GPU memory in MB
  --max-cpu-memory <mb>  Maximum CPU memory in MB

MoE Configuration:
  --moe-shared-gpu       Place shared experts on GPU (default)
  --moe-shared-cpu       Place shared experts on CPU
  --moe-sparse-gpu       Place sparse experts on GPU
  --moe-sparse-cpu       Place sparse experts on CPU (default)

Precision:
  --activation-precision <type>  Activation precision: fp32, bf16, fp16, q8_1
  --act-prec <type>      Alias for --activation-precision
    --kv-cache-precision <type>  KV cache precision: auto (fp16), fp32, fp16, q8_1
    --kv-prec <type>       Alias for --kv-cache-precision

Weight Sharding:
  --shard-weights        Enable weight sharding
  --no-shard-weights     Disable weight sharding

Heterogeneous Mode:
  --heterogeneous        Enable heterogeneous mode
  --cpu-fraction <f>     CPU compute fraction (0.0-1.0, default: 0.2)
  --no-gpu-tp            Disable GPU tensor parallelism
  --no-cpu-tp            Disable CPU tensor parallelism
  --min-layers-per-domain <n>  Minimum layers per domain (default: 2)

Verbosity:
  -v                     Increase verbosity (-v = DEBUG, -vv = TRACE)
  --list-devices         List available devices and exit
  -h, --help             Show this help message

Examples:
  llaminar2 -m model.gguf -p "Hello world" -n 50
  llaminar2 -m model.gguf --chat
  llaminar2 -m model.gguf --benchmark -n 100
  llaminar2 -m model.gguf --tp 2 --tp-devices "cuda:0,cuda:1"
  llaminar2 -m model.gguf -d cuda:0 --fused-attention
)HELPTEXT";
    }

} // namespace llaminar2
