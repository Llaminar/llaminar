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
#include <cstdlib>
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

        std::string normalizeToken(const std::string &str)
        {
            std::string normalized = toLower(trim(str));
            std::replace(normalized.begin(), normalized.end(), '-', '_');
            return normalized;
        }

        size_t leadingWhitespace(const std::string &line)
        {
            size_t count = 0;
            while (count < line.size() && (line[count] == ' ' || line[count] == '\t'))
            {
                ++count;
            }
            return count;
        }

        std::string stripOuterQuotes(std::string value)
        {
            value = trim(value);
            if (value.size() >= 2 &&
                ((value.front() == '"' && value.back() == '"') ||
                 (value.front() == '\'' && value.back() == '\'')))
            {
                value = value.substr(1, value.size() - 2);
            }
            return value;
        }

        bool parseBoolValue(const std::string &value)
        {
            const std::string normalized = normalizeToken(value);
            if (normalized == "true" || normalized == "1" || normalized == "yes" || normalized == "on")
                return true;
            if (normalized == "false" || normalized == "0" || normalized == "no" || normalized == "off")
                return false;
            throw std::invalid_argument("Invalid boolean value: '" + value + "'");
        }

        MoEExpertMode parseMoEExpertModeValue(const std::string &value)
        {
            auto parsed = parseMoEExpertMode(value);
            if (!parsed)
            {
                throw std::invalid_argument(
                    "Invalid MoE expert mode: '" + value +
                    "' (valid: expert-parallel, tensor-parallel, replicated)");
            }
            return *parsed;
        }

        MoERebalanceRuntimeMode parseMoERebalanceModeValue(const std::string &value)
        {
            auto parsed = parseMoERebalanceRuntimeMode(value);
            if (!parsed)
            {
                throw std::invalid_argument(
                    "Invalid MoE rebalance mode: '" + value +
                    "' (valid: off, observe, dynamic)");
            }
            return *parsed;
        }

        MoEHotExpertCacheConfig parseMoEHotExpertCacheValue(const std::string &value)
        {
            const std::string normalized = normalizeToken(value);
            MoEHotExpertCacheConfig config;

            if (normalized == "off" || normalized == "disabled" || normalized == "none" || normalized == "false")
            {
                config.kind = MoEHotExpertCacheConfig::Kind::Off;
                config.count = 0;
                config.percent = 0.0f;
                return config;
            }

            std::string trimmed_value = trim(value);
            if (!trimmed_value.empty() && trimmed_value.back() == '%')
            {
                const std::string number = trim(trimmed_value.substr(0, trimmed_value.size() - 1));
                try
                {
                    config.kind = MoEHotExpertCacheConfig::Kind::Percent;
                    config.percent = std::stof(number);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid percent value for --moe-hot-expert-cache: '" + value + "'");
                }
                if (config.percent < 0.0f || config.percent > 100.0f)
                {
                    throw std::invalid_argument("--moe-hot-expert-cache percent must be in [0, 100], got '" + value + "'");
                }
                return config;
            }

            try
            {
                config.kind = MoEHotExpertCacheConfig::Kind::Count;
                config.count = std::stoi(trimmed_value);
            }
            catch (const std::exception &)
            {
                throw std::invalid_argument(
                    "Invalid value for --moe-hot-expert-cache: '" + value +
                    "' (expected count, percent like 10%, or off)");
            }
            if (config.count < 0)
            {
                throw std::invalid_argument("--moe-hot-expert-cache count must be >= 0");
            }
            return config;
        }

        void applyMoEYamlKey(OrchestrationConfig &config,
                             const std::string &key,
                             const std::string &value)
        {
            const std::string normalized_key = normalizeToken(key);
            if (normalized_key == "expert_mode")
            {
                config.moe_expert_mode = parseMoEExpertModeValue(value);
            }
            else if (normalized_key == "hot_expert_cache")
            {
                config.moe_hot_expert_cache = parseMoEHotExpertCacheValue(value);
            }
            else if (normalized_key == "rebalance")
            {
                config.moe_rebalance.mode = parseMoERebalanceModeValue(value);
            }
            else if (normalized_key == "rebalance_window")
            {
                config.moe_rebalance.window_size = std::stoi(value);
            }
            else if (normalized_key == "rebalance_max_window")
            {
                config.moe_rebalance.max_window_size = std::stoi(value);
            }
            else if (normalized_key == "rebalance_window_growth")
            {
                config.moe_rebalance.window_growth_factor = std::stof(value);
            }
            else if (normalized_key == "release_raw_expert_weights")
            {
                config.moe_rebalance.release_raw_expert_weights = parseBoolValue(value);
            }
        }

        std::shared_ptr<MoEExpertParallelPlan> ensureMoEExpertParallelPlan(OrchestrationConfig &config)
        {
            if (!config.moe_expert_parallel_plan)
            {
                config.moe_expert_parallel_plan = std::make_shared<MoEExpertParallelPlan>();
            }
            return config.moe_expert_parallel_plan;
        }

        MoEExpertExecutionKind parseMoEExpertExecutionKind(const std::string &value, bool &enabled)
        {
            const std::string normalized = normalizeToken(value);
            if (normalized == "off" || normalized == "disabled" || normalized == "false")
            {
                enabled = false;
                return MoEExpertExecutionKind::TieredExpertOverlay;
            }
            if (normalized == "tiered" || normalized == "tiered_expert_overlay")
            {
                enabled = true;
                return MoEExpertExecutionKind::TieredExpertOverlay;
            }
            if (normalized == "single_domain" || normalized == "single_domain_expert_sharded")
            {
                enabled = true;
                return MoEExpertExecutionKind::SingleDomainExpertSharded;
            }
            throw std::invalid_argument("Invalid MoE expert overlay kind: '" + value + "' (valid: off, single-domain, tiered)");
        }

        void applyMoEExpertOverlayKind(OrchestrationConfig &config, const std::string &value)
        {
            bool enabled = false;
            const auto kind = parseMoEExpertExecutionKind(value, enabled);
            auto plan = ensureMoEExpertParallelPlan(config);
            plan->enabled = enabled;
            if (enabled)
            {
                plan->execution_kind = kind;
            }
        }

        ExpertResidencyPolicy parseExpertResidencyPolicyValue(const std::string &value)
        {
            const std::string normalized = normalizeToken(value);
            if (normalized == "disabled" || normalized == "off" || normalized == "none")
                return ExpertResidencyPolicy::Disabled;
            if (normalized == "static_by_id")
                return ExpertResidencyPolicy::StaticById;
            if (normalized == "histogram" || normalized == "histogram_tiered_cache")
                return ExpertResidencyPolicy::HistogramTieredCache;
            if (normalized == "explicit_masks")
                return ExpertResidencyPolicy::ExplicitMasks;
            throw std::invalid_argument("Invalid MoE expert overlay residency policy: '" + value + "' (valid: static-by-id, histogram, explicit-masks)");
        }

        ExpertComputeDomain parseMoEExpertOverlayDomainSpec(const std::string &spec)
        {
            ExecutionDomainParseOptions options;
            options.context = "MoE expert overlay domain";
            options.require_scope = true;
            options.allow_global_scope = false;
            options.require_compute = true;
            return ExpertComputeDomain::fromExecutionDomainDefinition(
                ExecutionDomainDefinition::parse(spec, options));
        }

        ExpertRoutedTier parseMoEExpertOverlayTierSpec(const std::string &spec)
        {
            const auto sections = split(spec, ';');
            if (sections.empty())
            {
                throw std::invalid_argument("MoE expert overlay tier spec is empty");
            }

            const auto at_pos = sections[0].find('@');
            if (at_pos == std::string::npos)
            {
                throw std::invalid_argument("Invalid MoE expert overlay tier spec: '" + spec + "' (expected name@domain;priority=N)");
            }

            ExpertRoutedTier tier;
            tier.name = trim(sections[0].substr(0, at_pos));
            tier.domain = trim(sections[0].substr(at_pos + 1));
            if (tier.name.empty() || tier.domain.empty())
            {
                throw std::invalid_argument("MoE expert overlay tier must include non-empty name and domain");
            }

            bool saw_priority = false;
            for (size_t i = 1; i < sections.size(); ++i)
            {
                const auto eq_pos = sections[i].find('=');
                if (eq_pos == std::string::npos)
                {
                    throw std::invalid_argument("Invalid MoE expert overlay tier option: '" + sections[i] + "'");
                }

                const std::string key = normalizeToken(sections[i].substr(0, eq_pos));
                const std::string value = trim(sections[i].substr(eq_pos + 1));

                if (key == "priority")
                {
                    tier.priority = std::stoi(value);
                    saw_priority = true;
                }
                else if (key == "max_experts_per_layer")
                {
                    tier.max_experts_per_layer = std::stoi(value);
                    if (tier.max_experts_per_layer < 0)
                    {
                        throw std::invalid_argument("MoE expert overlay tier max-experts-per-layer must be >= 0");
                    }
                }
                else if (key == "memory_mb")
                {
                    if (normalizeToken(value) == "auto")
                    {
                        tier.memory_budget_bytes = 0;
                    }
                    else
                    {
                        const auto mb = std::stoull(value);
                        tier.memory_budget_bytes = mb * 1024ULL * 1024ULL;
                    }
                }
                else if (key == "fallback")
                {
                    tier.fallback = parseBoolValue(value);
                }
                else
                {
                    throw std::invalid_argument("Unknown MoE expert overlay tier option: '" + key + "'");
                }
            }

            if (!saw_priority)
            {
                throw std::invalid_argument("MoE expert overlay tier '" + tier.name + "' is missing priority=<n>");
            }

            return tier;
        }

        std::string formatMoEOverlayValidationErrors(const std::vector<std::string> &errors)
        {
            std::ostringstream message;
            message << "Invalid MoE expert overlay configuration:";
            for (const auto &error : errors)
            {
                message << "\n - " << error;
            }
            return message.str();
        }

        void parseMoEExpertParallelYamlBlock(const std::string &yaml, OrchestrationConfig &config)
        {
            std::istringstream stream(yaml);
            std::string line;
            bool in_moe_block = false;
            std::string current_moe_section;

            while (std::getline(stream, line))
            {
                const std::string trimmed = trim(line);
                if (trimmed.empty() || trimmed[0] == '#')
                {
                    continue;
                }

                const size_t indent = leadingWhitespace(line);
                if (!in_moe_block)
                {
                    if (indent == 0 && trimmed == "moe_expert_parallel:")
                    {
                        in_moe_block = true;
                        current_moe_section.clear();
                        ensureMoEExpertParallelPlan(config);
                    }
                    continue;
                }

                if (indent == 0)
                {
                    in_moe_block = false;
                    current_moe_section.clear();
                    if (trimmed == "moe_expert_parallel:")
                    {
                        in_moe_block = true;
                        ensureMoEExpertParallelPlan(config);
                    }
                    continue;
                }

                auto plan = ensureMoEExpertParallelPlan(config);

                if (trimmed.rfind("-", 0) == 0)
                {
                    const std::string item = stripOuterQuotes(trim(trimmed.substr(1)));
                    if (current_moe_section == "domains")
                    {
                        plan->domains.push_back(parseMoEExpertOverlayDomainSpec(item));
                    }
                    else if (current_moe_section == "routed_tiers")
                    {
                        plan->routed_tiers.push_back(parseMoEExpertOverlayTierSpec(item));
                    }
                    continue;
                }

                if (trimmed.back() == ':' && trimmed.find(':') == trimmed.size() - 1)
                {
                    current_moe_section = normalizeToken(trimmed.substr(0, trimmed.size() - 1));
                    continue;
                }

                const size_t colon_pos = trimmed.find(':');
                if (colon_pos == std::string::npos)
                {
                    continue;
                }

                const std::string key = normalizeToken(trimmed.substr(0, colon_pos));
                const std::string value = stripOuterQuotes(trim(trimmed.substr(colon_pos + 1)));

                if (current_moe_section == "residency" && key == "mode")
                {
                    plan->residency_policy = parseExpertResidencyPolicyValue(value);
                    continue;
                }

                if (key == "enabled")
                {
                    plan->enabled = parseBoolValue(value);
                }
                else if (key == "execution_kind" || key == "kind")
                {
                    applyMoEExpertOverlayKind(config, value);
                }
                else if (key == "continuation_domain")
                {
                    plan->continuation_domain = value;
                }
                else if (key == "base_model_domain" || key == "base_domain")
                {
                    plan->base_model_domain = value;
                }
                else if (key == "shared_expert_domain" || key == "shared_domain")
                {
                    plan->shared_expert_domain = value;
                }
                else if (key == "residency_mode")
                {
                    plan->residency_policy = parseExpertResidencyPolicyValue(value);
                }
            }
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

    // =========================================================================
    // Structured CLI specification
    // =========================================================================
    //
    // Every flag lives in one place: a `CliSpec<OrchestrationConfig>` built by
    // `buildSpec()` below. The spec drives both `parseArgs()` (generic
    // parse / validation loop) and `getHelpText()` (auto-formatted) so the
    // two can't drift apart.
    //
    // A handful of flags still need custom setters because they either touch
    // multiple fields (`--device`, `--device-map`, `--deterministic`) or push
    // into a vector (`--define-domain`, `--pp-stage`). Those are wrapped in
    // `setters::custom(...)` inline; everything else uses the generic helpers
    // from `CliSpec.h`.

    namespace
    {
        using Opt = CliOption<OrchestrationConfig>;

        // Small helpers to bridge existing enum parsers (which return
        // std::optional) into CliSpec setters.
        template <typename EnumT, typename Parser>
        Opt::Setter enumSetter(EnumT OrchestrationConfig::*member,
                               Parser parser,
                               std::string option_name,
                               std::string valid_list)
        {
            return [member, parser, option_name, valid_list](OrchestrationConfig &c,
                                                             const std::string &v)
            {
                auto parsed = parser(v);
                if (!parsed)
                {
                    throw std::invalid_argument("Invalid value for " + option_name +
                                                ": '" + v + "' (valid: " + valid_list + ")");
                }
                c.*member = *parsed;
            };
        }
    } // namespace

    CliSpec<OrchestrationConfig> OrchestrationConfigParser::buildSpec()
    {
        CliSpec<OrchestrationConfig> spec;

        // Preserve the order in which categories appear in --help.
        spec.addCategory("Model Configuration")
            .addCategory("Inference Configuration")
            .addCategory("Sampling Configuration")
            .addCategory("Chat Configuration")
            .addCategory("Benchmark Configuration")
            .addCategory("Server Configuration")
            .addCategory("Fused Attention")
            .addCategory("MPI Bootstrap")
            .addCategory("Device Assignment")
            .addCategory("Tensor Parallelism")
            .addCategory("Pipeline Parallelism")
            .addCategory("Named Domains (advanced)")
            .addCategory("Collective Backend")
            .addCategory("Introspection")
            .addCategory("Config File")
            .addCategory("MoE Configuration")
            .addCategory("Precision")
            .addCategory("Heterogeneous Mode")
            .addCategory("Verbosity");

        // --- Model Configuration ---------------------------------------------
        spec.add({
            .short_name = "-m",
            .long_name = "--model",
            .category = "Model Configuration",
            .value_label = "<path>",
            .description = "Path to GGUF model file (required)",
            .setter = setters::assignString(&OrchestrationConfig::model_path),
        });
        spec.add({
            .short_name = "-c",
            .long_name = "--context-length",
            .category = "Model Configuration",
            .value_label = "<n>",
            .description = "Maximum context/sequence length (default: 4096)",
            .setter = setters::parseInt(&OrchestrationConfig::max_seq_len, "--context-length"),
        });
        spec.add({
            .long_name = "--mmap",
            .category = "Model Configuration",
            .description = "Use memory-mapped file loading (default)",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::use_mmap),
        });
        spec.add({
            .long_name = "--no-mmap",
            .category = "Model Configuration",
            .description = "Disable memory-mapped file loading",
            .setter = setters::assignBoolFalse(&OrchestrationConfig::use_mmap),
        });

        // --- Inference Configuration -----------------------------------------
        spec.add({
            .short_name = "-p",
            .long_name = "--prompt",
            .category = "Inference Configuration",
            .value_label = "<text>",
            .description = "Input prompt text",
            .setter = setters::assignString(&OrchestrationConfig::prompt),
        });
        spec.add({
            .short_name = "-n",
            .long_name = "--n-predict",
            .category = "Inference Configuration",
            .value_label = "<n>",
            .description = "Tokens to generate (-1 = until EOS, default: -1)",
            .setter = setters::parseInt(&OrchestrationConfig::n_predict, "--n-predict"),
        });
        spec.add({
            .long_name = "--batch-size",
            .category = "Inference Configuration",
            .value_label = "<n>",
            .description = "Batch size (default: 1)",
            .setter = setters::parseInt(&OrchestrationConfig::batch_size, "--batch-size"),
        });
        spec.add({
            .long_name = "--threads",
            .category = "Inference Configuration",
            .value_label = "<n>",
            .description = "Thread count override for OpenMP / BLAS (-1 = auto)",
            .setter = setters::parseInt(&OrchestrationConfig::n_threads, "--threads"),
        });
        spec.add({
            .short_name = "-s",
            .long_name = "--seed",
            .category = "Inference Configuration",
            .value_label = "<n>",
            .description = "Random seed (-1 = random, default: -1)",
            .setter = setters::parseInt(&OrchestrationConfig::seed, "--seed"),
        });

        // --- Sampling Configuration ------------------------------------------
        spec.add({
            .short_name = "-t",
            .long_name = "--temperature",
            .category = "Sampling Configuration",
            .value_label = "<f>",
            .description = "Sampling temperature (default: 0.8; 0 = greedy)",
            .setter = setters::parseFloat(&OrchestrationConfig::temperature, "--temperature"),
        });
        spec.add({
            .long_name = "--top-k",
            .category = "Sampling Configuration",
            .value_label = "<n>",
            .description = "Top-K sampling (default: 40)",
            .setter = setters::parseInt(&OrchestrationConfig::top_k, "--top-k"),
        });
        spec.add({
            .long_name = "--top-p",
            .category = "Sampling Configuration",
            .value_label = "<f>",
            .description = "Top-P (nucleus) sampling (default: 0.9)",
            .setter = setters::parseFloat(&OrchestrationConfig::top_p, "--top-p"),
        });
        spec.add({
            .long_name = "--deterministic",
            .category = "Sampling Configuration",
            .description = "Force greedy sampling (temperature=0) and export "
                           "LLAMINAR_DETERMINISTIC=1 for kernel-level determinism",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &)
                {
                    c.deterministic = true;
                    c.temperature = 0.0f;
                    setenv("LLAMINAR_DETERMINISTIC", "1", 1);
                }),
        });

        // --- Chat Configuration ----------------------------------------------
        spec.add({
            .long_name = "--chat",
            .category = "Chat Configuration",
            .description = "Interactive chat mode",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::chat_mode),
        });
        spec.add({
            .long_name = "--chat-single",
            .category = "Chat Configuration",
            .description = "Single prompt with chat template applied",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::single_shot_chat),
        });
        spec.add({
            .long_name = "--system",
            .category = "Chat Configuration",
            .value_label = "<text>",
            .description = "System prompt for chat",
            .setter = setters::assignString(&OrchestrationConfig::system_prompt),
        });
        spec.add({
            .long_name = "--chat-template",
            .category = "Chat Configuration",
            .value_label = "<name>",
            .description = "Override chat template (chatml, llama3, etc.)",
            .setter = setters::assignString(&OrchestrationConfig::chat_template_override),
        });

        // --- Benchmark Configuration -----------------------------------------
        // NOTE: --benchmark is accepted for backward compatibility but is
        // deprecated.  Use 'llaminar2 benchmark' subcommand instead.
        spec.add({
            .long_name = "--benchmark",
            .category = "Benchmark Configuration",
            .description = "(deprecated) Use 'llaminar2 benchmark' subcommand instead",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::benchmark_mode),
        });

        // --- Server Configuration --------------------------------------------
        spec.add({
            .long_name = "--serve",
            .category = "Server Configuration",
            .description = "Start HTTP server (OpenAI-compatible REST API)",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::serve_mode),
        });
        spec.add({
            .long_name = "--port",
            .category = "Server Configuration",
            .value_label = "<n>",
            .description = "Server port (default: 8080)",
            .setter = setters::parseInt(&OrchestrationConfig::serve_port, "--port"),
        });
        spec.add({
            .long_name = "--host",
            .category = "Server Configuration",
            .value_label = "<addr>",
            .description = "Server bind address (default: 127.0.0.1)",
            .setter = setters::assignString(&OrchestrationConfig::serve_host),
        });

        // --- Fused Attention -------------------------------------------------
        spec.add({
            .long_name = "--fused-attention-backend",
            .category = "Fused Attention",
            .value_label = "<type>",
            .description = "Backend: jit (default), reference, tiled, q16",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.fused_attention_backend = parseFusedAttentionBackend(v);
                }),
        });

        // --- MPI Bootstrap ---------------------------------------------------
        spec.add({
            .long_name = "--mpi-procs",
            .category = "MPI Bootstrap",
            .value_label = "<n>",
            .description = "Number of MPI processes (0 = auto)",
            .setter = setters::parseInt(&OrchestrationConfig::mpi_procs, "--mpi-procs"),
        });
        spec.add({
            .long_name = "--mpi-hostfile",
            .category = "MPI Bootstrap",
            .value_label = "<path>",
            .description = "MPI hostfile path (also used for node detection)",
            .setter = setters::assignString(&OrchestrationConfig::hostfile),
        });
        spec.add({
            .long_name = "--mpi-dry-run",
            .category = "MPI Bootstrap",
            .description = "Print MPI launch command and exit",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::mpi_dry_run),
        });
        spec.add({
            .long_name = "--mpi-verbose",
            .category = "MPI Bootstrap",
            .description = "Verbose MPI output",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::mpi_verbose),
        });
        spec.add({
            .long_name = "--no-mpi-bootstrap",
            .category = "MPI Bootstrap",
            .description = "Disable auto MPI bootstrap (DEBUG ONLY; use for ncu/nsys/perf only)",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::mpi_no_bootstrap),
        });
        spec.add({
            .long_name = "--mpi-oversubscribe",
            .category = "MPI Bootstrap",
            .description = "Allow MPI oversubscription",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::mpi_oversubscribe),
        });
        spec.add({
            .long_name = "--mpi-profile",
            .category = "MPI Bootstrap",
            .value_label = "<mode>",
            .description = "MPI bootstrap profile: auto (default), tuned",
            .valid_values = {"auto", "tuned"},
            .setter = enumSetter(&OrchestrationConfig::mpi_profile,
                                 parseMPIProfile, "--mpi-profile", "auto, tuned"),
        });

        // --- Device Assignment -----------------------------------------------
        spec.add({
            .short_name = "-d",
            .long_name = "--device",
            .category = "Device Assignment",
            .value_label = "<spec>",
            .description = "Device for this rank. cuda:N / rocm:N for GPUs; "
                           "cpu selects all local NUMA nodes; cpu:N selects one",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &value)
                {
                    std::string lower = toLower(value);
                    if (lower == "cpu")
                    {
                        c.device_for_this_rank = GlobalDeviceAddress::cpu(0);
                        c.device_for_this_rank_numa_explicit = false;
                        c.cpu_global_tp_all_local = true;
                        return;
                    }
                    if (lower.rfind("cpu:", 0) == 0)
                    {
                        try
                        {
                            int numa = std::stoi(value.substr(4));
                            if (numa < 0)
                                throw std::invalid_argument("negative NUMA");
                            c.device_for_this_rank = GlobalDeviceAddress::cpu(numa);
                            c.device_for_this_rank_numa_explicit = true;
                            c.cpu_global_tp_all_local = false;
                        }
                        catch (const std::exception &)
                        {
                            throw std::invalid_argument(
                                "Invalid CPU device specification: '" + value +
                                "' (expected cpu or cpu:<numa>)");
                        }
                        return;
                    }
                    auto addr = GlobalDeviceAddress::tryParse(value);
                    if (!addr)
                        throw std::invalid_argument(
                            "Invalid device specification: '" + value + "'");
                    c.device_for_this_rank = *addr;
                    size_t colon_count = static_cast<size_t>(
                        std::count(value.begin(), value.end(), ':'));
                    c.device_for_this_rank_numa_explicit = (colon_count >= 2);
                    c.cpu_global_tp_all_local = false;
                }),
        });
        spec.add({
            .long_name = "--device-mode",
            .category = "Device Assignment",
            .value_label = "<mode>",
            .description = "Assignment mode: auto, local_gpu, round_robin, explicit "
                           "(dashes also accepted: local-gpu, round-robin)",
            .setter = enumSetter(&OrchestrationConfig::device_mode,
                                 parseDeviceAssignmentMode, "--device-mode",
                                 "auto, local_gpu, round_robin, explicit"),
        });
        spec.add({
            .long_name = "--device-map",
            .category = "Device Assignment",
            .value_label = "<map>",
            .description = "Explicit rank->device mapping, e.g. \"0=cuda:0,1=cuda:1\"",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.device_map = parseDeviceMap(v);
                    c.device_map_numa_explicit = parseDeviceMapNumaExplicit(v);
                    c.device_mode = DeviceAssignmentMode::EXPLICIT;
                }),
        });

        // --- Tensor Parallelism ----------------------------------------------
        spec.add({
            .short_name = "-tp",
            .long_name = "--tensor-parallelism-degree",
            .category = "Tensor Parallelism",
            .value_label = "<n>",
            .description = "TP parallelism degree",
            .setter = setters::parseInt(&OrchestrationConfig::tp_degree,
                                        "--tensor-parallelism-degree"),
        });
        spec.add({
            .long_name = "--tp-scope",
            .category = "Tensor Parallelism",
            .value_label = "<scope>",
            .description = "Scope: auto, local, node_local, global, hybrid",
            .setter = enumSetter(&OrchestrationConfig::tp_scope,
                                 parseTpScope, "--tp-scope",
                                 "auto, local, node_local, global, hybrid"),
        });
        spec.add({
            .long_name = "--tp-devices",
            .category = "Tensor Parallelism",
            .value_label = "<list>",
            .description = "Device list, e.g. \"cuda:0,cuda:1\"",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.tp_devices = parseDeviceList(v);
                }),
        });
        spec.add({
            .long_name = "--tp-weights",
            .category = "Tensor Parallelism",
            .value_label = "<list>",
            .description = "Weight distribution (requires --tp-devices), e.g. \"0.73,0.27\"",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.tp_weights = parseWeightList(v);
                }),
        });

        // --- Pipeline Parallelism --------------------------------------------
        spec.add({
            .short_name = "-pp",
            .long_name = "--pipeline-parallelism-degree",
            .category = "Pipeline Parallelism",
            .value_label = "<n>",
            .description = "PP parallelism degree",
            .setter = setters::parseInt(&OrchestrationConfig::pp_degree,
                                        "--pipeline-parallelism-degree"),
        });
        spec.add({
            .long_name = "--pp-split",
            .category = "Pipeline Parallelism",
            .value_label = "<mode>",
            .description = "Layer split: equal, weighted, manual",
            .setter = enumSetter(&OrchestrationConfig::pp_split,
                                 parsePpSplitMode, "--pp-split",
                                 "equal, weighted, manual"),
        });

        // --- Named Domains (advanced) ----------------------------------------
        spec.add({
            .long_name = "--define-domain",
            .category = "Named Domains (advanced)",
            .value_label = "<spec>",
            .description = "Define domain: \"name=dev1,dev2[;weights=w1,w2][;backend=type][;scope=local|node_local|global][;owner=N][;ranks=0,1,...]\"",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.domain_definitions.push_back(DomainDefinition::parse(v));
                }),
        });
        spec.add({
            .long_name = "--pp-stage",
            .category = "Named Domains (advanced)",
            .value_label = "<spec>",
            .description = "Define PP stage: \"stage_id=domain:first_layer-last_layer\"",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.pp_stage_definitions.push_back(PPStageDefinition::parse(v));
                }),
        });

        // --- Collective Backend ----------------------------------------------
        spec.add({
            .short_name = "-b",
            .long_name = "--backend",
            .category = "Collective Backend",
            .value_label = "<type>",
            .description = "Default collective: auto, nccl, rccl, upi, mpi, host, heterogeneous",
            .setter = enumSetter(&OrchestrationConfig::default_backend,
                                 parseCollectiveBackendType, "--backend",
                                 "auto, nccl, rccl, upi, mpi, host, heterogeneous"),
        });

        // --- Introspection ---------------------------------------------------
        spec.add({
            .long_name = "--dry-run",
            .category = "Introspection",
            .description = "Validate configuration and cluster inventory, then exit",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::dry_run),
        });
        spec.add({
            .long_name = "--explain-placement",
            .category = "Introspection",
            .description = "Dump resolved orchestration plan on rank 0",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::explain_placement),
        });
        spec.add({
            .long_name = "--show-topology",
            .category = "Introspection",
            .description = "Show detected CPU topology and exit",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::show_topology),
        });
        spec.add({
            .long_name = "--show-numa",
            .category = "Introspection",
            .description = "Show NUMA configuration and exit",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::show_numa),
        });
        spec.add({
            .long_name = "--validate-only",
            .category = "Introspection",
            .description = "Validate configuration and exit",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::validate_only),
        });
        spec.add({
            .long_name = "--list-devices",
            .category = "Introspection",
            .description = "List available devices and exit",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::list_devices),
        });

        // --- Config File -----------------------------------------------------
        // Loaded as the base config in the first pass of parseArgs; on the
        // second pass we just record the path so downstream code sees it.
        spec.add({
            .long_name = "--config",
            .category = "Config File",
            .value_label = "<path>",
            .description = "Load base configuration from YAML; subsequent CLI flags override it",
            .setter = setters::assignString(&OrchestrationConfig::config_file_path),
        });

        // --- MoE Configuration -----------------------------------------------
        spec.add({
            .long_name = "--moe-shared-gpu",
            .category = "MoE Configuration",
            .description = "Place shared experts on GPU (default)",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::moe_shared_experts_gpu),
        });
        spec.add({
            .long_name = "--moe-shared-cpu",
            .category = "MoE Configuration",
            .description = "Place shared experts on CPU",
            .setter = setters::assignBoolFalse(&OrchestrationConfig::moe_shared_experts_gpu),
        });
        spec.add({
            .long_name = "--moe-sparse-gpu",
            .category = "MoE Configuration",
            .description = "Place sparse experts on GPU",
            .setter = setters::assignBoolFalse(&OrchestrationConfig::moe_sparse_experts_cpu),
        });
        spec.add({
            .long_name = "--moe-sparse-cpu",
            .category = "MoE Configuration",
            .description = "Place sparse experts on CPU (default)",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::moe_sparse_experts_cpu),
        });
        spec.add({
            .long_name = "--moe-expert-mode",
            .category = "MoE Configuration",
            .value_label = "<mode>",
            .description = "Routed expert execution: expert-parallel (default), tensor-parallel, replicated",
            .valid_values = {"expert-parallel", "tensor-parallel", "replicated"},
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.moe_expert_mode = parseMoEExpertModeValue(v);
                }),
        });
        spec.add({
            .long_name = "--moe-hot-expert-cache",
            .category = "MoE Configuration",
            .value_label = "<count|percent|off>",
            .description = "Remote hot expert replica cap per rank/device (default: 10%)",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.moe_hot_expert_cache = parseMoEHotExpertCacheValue(v);
                }),
        });
        spec.add({
            .long_name = "--moe-rebalance",
            .category = "MoE Configuration",
            .value_label = "<mode>",
            .description = "MoE decode rebalance mode: off, observe, dynamic (default)",
            .valid_values = {"off", "observe", "dynamic"},
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.moe_rebalance.mode = parseMoERebalanceModeValue(v);
                }),
        });
        spec.add({
            .long_name = "--moe-rebalance-window",
            .category = "MoE Configuration",
            .value_label = "<tokens>",
            .description = "Decode histogram window size for MoE rebalance (default: 256)",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.moe_rebalance.window_size = std::stoi(v);
                }),
        });
        spec.add({
            .long_name = "--moe-rebalance-max-window",
            .category = "MoE Configuration",
            .value_label = "<tokens>",
            .description = "Maximum adaptive MoE rebalance window (default: 4096; 0 disables growth)",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.moe_rebalance.max_window_size = std::stoi(v);
                }),
        });
        spec.add({
            .long_name = "--moe-rebalance-window-growth",
            .category = "MoE Configuration",
            .value_label = "<factor>",
            .description = "Adaptive MoE rebalance window growth factor (default: 1.5)",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.moe_rebalance.window_growth_factor = std::stof(v);
                }),
        });
        spec.add({
            .long_name = "--moe-release-raw-expert-weights",
            .category = "MoE Configuration",
            .description = "Release raw routed expert tensors after prepared weights are resident",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &)
                {
                    c.moe_rebalance.release_raw_expert_weights = true;
                }),
        });
        spec.add({
            .long_name = "--moe-expert-overlay",
            .category = "MoE Configuration",
            .value_label = "<kind>",
            .description = "Same-layer MoE expert overlay: off, single-domain, tiered",
            .valid_values = {"off", "single-domain", "tiered"},
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    applyMoEExpertOverlayKind(c, v);
                }),
        });
        spec.add({
            .long_name = "--moe-expert-overlay-continuation",
            .category = "MoE Configuration",
            .value_label = "<domain>",
            .description = "MoE overlay domain that receives the final reduced output",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    ensureMoEExpertParallelPlan(c)->continuation_domain = v;
                }),
        });
        spec.add({
            .long_name = "--moe-expert-overlay-base-domain",
            .aliases = {"--base-model-domain"},
            .category = "MoE Configuration",
            .value_label = "<domain>",
            .description = "MoE overlay domain for dense/non-expert model placement (defaults to continuation)",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    ensureMoEExpertParallelPlan(c)->base_model_domain = v;
                }),
        });
        spec.add({
            .long_name = "--moe-expert-overlay-shared-domain",
            .category = "MoE Configuration",
            .value_label = "<domain>",
            .description = "MoE overlay domain where shared experts execute",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    ensureMoEExpertParallelPlan(c)->shared_expert_domain = v;
                }),
        });
        spec.add({
            .long_name = "--moe-expert-overlay-residency",
            .category = "MoE Configuration",
            .value_label = "<policy>",
            .description = "MoE overlay residency: static-by-id, histogram, explicit-masks",
            .valid_values = {"static-by-id", "histogram", "explicit-masks"},
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    ensureMoEExpertParallelPlan(c)->residency_policy = parseExpertResidencyPolicyValue(v);
                }),
        });
        spec.add({
            .long_name = "--moe-expert-overlay-domain",
            .category = "MoE Configuration",
            .value_label = "<spec>",
            .description = "Define MoE overlay domain: \"name=devices;scope=single|local|node_local;backend=type;compute=replicated_experts|expert_id_sharded|tensor_parallel_experts[;owner=N][;ranks=0,1]\"",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    ensureMoEExpertParallelPlan(c)->domains.push_back(parseMoEExpertOverlayDomainSpec(v));
                }),
        });
        spec.add({
            .long_name = "--moe-expert-overlay-tier",
            .category = "MoE Configuration",
            .value_label = "<spec>",
            .description = "Define MoE overlay routed tier: \"name@domain;priority=N[;max-experts-per-layer=N][;memory-mb=N|auto][;fallback=true]\"",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    ensureMoEExpertParallelPlan(c)->routed_tiers.push_back(parseMoEExpertOverlayTierSpec(v));
                }),
        });

        // --- Precision -------------------------------------------------------
        spec.add({
            .long_name = "--activation-precision",
            .aliases = {"--activation-prec", "--act-prec"},
            .category = "Precision",
            .value_label = "<type>",
            .description = "Activation precision: fp32, bf16, fp16, q8_1",
            .valid_values = {"fp32", "bf16", "fp16", "q8_1"},
            .setter = setters::assignString(&OrchestrationConfig::activation_precision),
        });
        // --kv-cache-precision accepts many short aliases; normalise to
        // lowercase and validate against the full alias list.
        spec.add({
            .long_name = "--kv-cache-precision",
            .aliases = {"--kv-prec"},
            .category = "Precision",
            .value_label = "<type>",
            .description = "KV cache precision: auto (default), fp32, fp16, q8_1, q16_1, tq4, tq "
                           "(short aliases: f32, f16, q8, q16, i16)",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &value)
                {
                    std::string lower = toLower(value);
                    static const std::set<std::string> valid_precisions = {
                        "auto", "fp32", "f32", "fp16", "f16",
                        "q8_1", "q8", "q81",
                        "q16_1", "q16", "q161",
                        "i16", "int16", "tq4", "tq"};
                    if (valid_precisions.find(lower) == valid_precisions.end())
                    {
                        throw std::invalid_argument(
                            "Invalid value for --kv-cache-precision: '" + value +
                            "' (valid: auto, fp32, fp16, q8_1, q16_1, tq4, tq)");
                    }
                    c.kv_cache_precision = lower;
                }),
        });

        // --- Heterogeneous Mode ----------------------------------------------
        spec.add({
            .long_name = "--cpu-fraction",
            .category = "Heterogeneous Mode",
            .value_label = "<f>",
            .description = "CPU compute fraction (0.0-1.0, default: 0.2)",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    try
                    {
                        c.cpu_compute_fraction = std::stof(v);
                    }
                    catch (const std::exception &)
                    {
                        throw std::invalid_argument(
                            "Invalid value for --cpu-fraction: '" + v + "'");
                    }
                    if (c.cpu_compute_fraction < 0.0f || c.cpu_compute_fraction > 1.0f)
                        throw std::invalid_argument(
                            "--cpu-fraction must be between 0.0 and 1.0");
                }),
        });
        spec.add({
            .long_name = "--min-layers-per-domain",
            .category = "Heterogeneous Mode",
            .value_label = "<n>",
            .description = "Minimum layers per domain (default: 2)",
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    try
                    {
                        c.min_layers_per_domain = std::stoi(v);
                    }
                    catch (const std::exception &)
                    {
                        throw std::invalid_argument(
                            "Invalid value for --min-layers-per-domain: '" + v + "'");
                    }
                    if (c.min_layers_per_domain < 1)
                        throw std::invalid_argument(
                            "--min-layers-per-domain must be >= 1");
                }),
        });

        // --- Verbosity -------------------------------------------------------
        spec.add({
            .short_name = "-v",
            .category = "Verbosity",
            .description = "Increase verbosity (-v, -vv, -vvv)",
            .setter = setters::incrementInt(&OrchestrationConfig::verbose_level, 3),
        });
        spec.add({
            .short_name = "-vv",
            .category = "Verbosity",
            .description = "Shorthand for verbosity level 2 (DEBUG)",
            .setter = setters::assignIntLiteral(&OrchestrationConfig::verbose_level, 2),
        });
        spec.add({
            .short_name = "-vvv",
            .category = "Verbosity",
            .description = "Shorthand for verbosity level 3 (TRACE)",
            .setter = setters::assignIntLiteral(&OrchestrationConfig::verbose_level, 3),
        });
        spec.add({
            .short_name = "-h",
            .long_name = "--help",
            .category = "Verbosity",
            .description = "Show this help message",
            .setter = setters::assignBoolTrue(&OrchestrationConfig::show_help),
        });

        // =====================================================================
        // Not yet implemented (parsed for back-compat, rendered in NYI section)
        // =====================================================================
        spec.add({
            .long_name = "--fused-attention",
            .category = "Fused Attention",
            .description = "Superseded by --fused-attention-backend",
            .not_yet_implemented = true,
            .setter = setters::assignBoolTrue(&OrchestrationConfig::use_fused_attention),
        });
        spec.add({
            .long_name = "--shard-weights",
            .category = "Tensor Parallelism",
            .description = "Weight sharding is automatic based on TP",
            .not_yet_implemented = true,
            .setter = setters::assignBoolTrue(&OrchestrationConfig::shard_weights),
        });
        spec.add({
            .long_name = "--no-shard-weights",
            .category = "Tensor Parallelism",
            .description = "Weight sharding is automatic based on TP",
            .not_yet_implemented = true,
            .setter = setters::assignBoolTrue(&OrchestrationConfig::disable_weight_sharding),
        });
        spec.add({
            .long_name = "--heterogeneous",
            .category = "Heterogeneous Mode",
            .description = "No-op; use --define-domain for heterogeneous setups",
            .not_yet_implemented = true,
            .setter = setters::assignBoolTrue(&OrchestrationConfig::heterogeneous_mode),
        });
        spec.add({
            .long_name = "--no-gpu-tp",
            .category = "Heterogeneous Mode",
            .description = "No-op under the current heterogeneous path",
            .not_yet_implemented = true,
            .setter = setters::assignBoolTrue(&OrchestrationConfig::disable_gpu_tp),
        });
        spec.add({
            .long_name = "--no-cpu-tp",
            .category = "Heterogeneous Mode",
            .description = "No-op under the current heterogeneous path",
            .not_yet_implemented = true,
            .setter = setters::assignBoolTrue(&OrchestrationConfig::disable_cpu_tp),
        });
        spec.add({
            .long_name = "--tp-local",
            .category = "Tensor Parallelism",
            .value_label = "<degree>",
            .description = "Hybrid-TP local subdegree (not yet consumed)",
            .not_yet_implemented = true,
            .setter = setters::parseInt(&OrchestrationConfig::tp_local_degree, "--tp-local"),
        });
        spec.add({
            .long_name = "--tp-global",
            .category = "Tensor Parallelism",
            .value_label = "<degree>",
            .description = "Hybrid-TP global subdegree (not yet consumed)",
            .not_yet_implemented = true,
            .setter = setters::parseInt(&OrchestrationConfig::tp_global_degree, "--tp-global"),
        });
        spec.add({
            .long_name = "--cpu-layers",
            .category = "Layer Placement",
            .value_label = "<n>",
            .description = "Legacy layer-placement; use --pp-stage with CPU domains",
            .not_yet_implemented = true,
            .setter = setters::parseInt(&OrchestrationConfig::cpu_layers, "--cpu-layers"),
        });
        spec.add({
            .long_name = "--cpu-layers-first",
            .category = "Layer Placement",
            .description = "Legacy layer-placement; use --pp-stage with CPU domains",
            .not_yet_implemented = true,
            .setter = setters::assignBoolTrue(&OrchestrationConfig::cpu_layers_first),
        });
        spec.add({
            .long_name = "--max-gpu-memory",
            .category = "Memory Constraints",
            .value_label = "<mb>",
            .description = "Legacy DeviceOrchestrator is inactive",
            .not_yet_implemented = true,
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.max_gpu_memory_mb = std::stoull(v);
                }),
        });
        spec.add({
            .long_name = "--max-cpu-memory",
            .category = "Memory Constraints",
            .value_label = "<mb>",
            .description = "Legacy DeviceOrchestrator is inactive",
            .not_yet_implemented = true,
            .setter = setters::custom<OrchestrationConfig>(
                [](OrchestrationConfig &c, const std::string &v)
                {
                    c.max_cpu_memory_mb = std::stoull(v);
                }),
        });
        spec.add({
            .long_name = "--topology",
            .category = "Topology",
            .value_label = "<spec>",
            .description = "Tree parsing not yet wired into runner",
            .not_yet_implemented = true,
            .setter = setters::assignString(&OrchestrationConfig::topology_string),
        });
        spec.add({
            .long_name = "--topology-file",
            .category = "Topology",
            .value_label = "<path>",
            .description = "Tree parsing not yet wired into runner",
            .not_yet_implemented = true,
            .setter = setters::assignString(&OrchestrationConfig::topology_file_path),
        });

        return spec;
    }

    // =========================================================================
    // parseArgs
    // =========================================================================

    OrchestrationConfig OrchestrationConfigParser::parseArgs(int argc, char **argv)
    {
        // Convert argc/argv into a vector for ergonomic iteration.
        std::vector<std::string> args;
        args.reserve(static_cast<size_t>(std::max(0, argc - 1)));
        for (int i = 1; i < argc; ++i)
            args.emplace_back(argv[i]);

        // First pass: if --config <path> was supplied, load the YAML file as
        // the base config. The second pass (full CLI parse) will override any
        // individual fields specified on the command line.
        OrchestrationConfig config;
        for (size_t i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] == "--config")
            {
                config = parseYamlFile(args[i + 1]);
                config.config_file_path = args[i + 1];
                break;
            }
            // Also handle --config=<path>
            static const std::string kPrefix = "--config=";
            if (args[i].compare(0, kPrefix.size(), kPrefix) == 0)
            {
                std::string path = args[i].substr(kPrefix.size());
                if (!path.empty())
                {
                    config = parseYamlFile(path);
                    config.config_file_path = path;
                }
                break;
            }
        }

        // Second pass: apply every flag through the structured spec.
        static const CliSpec<OrchestrationConfig> spec = buildSpec();
        spec.parse(args, config);

        // Cross-flag validation that doesn't fit cleanly into per-option setters.
        if (config.heterogeneous_mode && config.disable_gpu_tp && config.disable_cpu_tp)
        {
            throw std::invalid_argument(
                "Cannot use --heterogeneous with both --no-gpu-tp and --no-cpu-tp");
        }

        auto normalize_errors = normalizeMoEExpertOverlayDomains(config);
        if (!normalize_errors.empty())
        {
            throw std::invalid_argument(formatMoEOverlayValidationErrors(normalize_errors));
        }

        auto overlay_errors = validateMoEExpertOverlayConfig(config);
        if (!overlay_errors.empty())
        {
            throw std::invalid_argument(formatMoEOverlayValidationErrors(overlay_errors));
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

        parseMoEExpertParallelYamlBlock(yaml, config);

        // Simple line-by-line YAML parser (sufficient for our flat structure)
        // For production, consider using a proper YAML library like yaml-cpp

        std::istringstream stream(yaml);
        std::string line;
        std::string current_section;
        bool skipping_moe_block = false;

        while (std::getline(stream, line))
        {
            std::string trimmed = trim(line);

            // Skip empty lines and comments
            if (trimmed.empty() || trimmed[0] == '#')
            {
                continue;
            }

            const size_t indent = leadingWhitespace(line);
            if (skipping_moe_block)
            {
                if (indent > 0)
                {
                    continue;
                }
                skipping_moe_block = false;
            }

            if (indent == 0 && trimmed == "moe_expert_parallel:")
            {
                skipping_moe_block = true;
                current_section.clear();
                continue;
            }

            // Minimal YAML list support for named-domain configs:
            // domains:
            //   - "gpu=0:cuda:0;scope=local;owner=0"
            // pp_stages:
            //   - "0=gpu:0-11"
            if (trimmed.rfind("-", 0) == 0)
            {
                std::string item = trim(trimmed.substr(1));
                if (item.size() >= 2 &&
                    ((item.front() == '"' && item.back() == '"') ||
                     (item.front() == '\'' && item.back() == '\'')))
                {
                    item = item.substr(1, item.size() - 2);
                }

                if (current_section == "domains")
                {
                    auto domain = DomainDefinition::tryParse(item);
                    if (domain)
                    {
                        config.domain_definitions.push_back(*domain);
                    }
                    continue;
                }
                if (current_section == "pp_stages")
                {
                    auto stage = PPStageDefinition::tryParse(item);
                    if (stage)
                    {
                        config.pp_stage_definitions.push_back(*stage);
                    }
                    continue;
                }
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

            const std::string normalized_section = normalizeToken(current_section);
            const std::string normalized_key = normalizeToken(key);
            if (normalized_section == "moe")
            {
                applyMoEYamlKey(config, normalized_key, value);
                continue;
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
            else if (normalized_key == "moe_expert_mode")
            {
                config.moe_expert_mode = parseMoEExpertModeValue(value);
            }
            else if (normalized_key == "moe_hot_expert_cache")
            {
                config.moe_hot_expert_cache = parseMoEHotExpertCacheValue(value);
            }
            else if (normalized_key == "moe_rebalance")
            {
                config.moe_rebalance.mode = parseMoERebalanceModeValue(value);
            }
            else if (normalized_key == "moe_rebalance_window")
            {
                config.moe_rebalance.window_size = std::stoi(value);
            }
            else if (normalized_key == "moe_rebalance_max_window")
            {
                config.moe_rebalance.max_window_size = std::stoi(value);
            }
            else if (normalized_key == "moe_rebalance_window_growth")
            {
                config.moe_rebalance.window_growth_factor = std::stof(value);
            }
            else if (normalized_key == "moe_release_raw_expert_weights")
            {
                config.moe_rebalance.release_raw_expert_weights = parseBoolValue(value);
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

        auto normalize_errors = normalizeMoEExpertOverlayDomains(config);
        if (!normalize_errors.empty())
        {
            throw std::invalid_argument(formatMoEOverlayValidationErrors(normalize_errors));
        }

        return config;
    }

    // =========================================================================
    // Help text
    // =========================================================================

    std::string OrchestrationConfigParser::getHelpText()
    {
        // Help text is auto-generated from the structured CLI spec so flag
        // wiring and help text can never drift apart.
        static const CliSpec<OrchestrationConfig> spec = buildSpec();

        const std::string header =
            "\nLlaminar V2 LLM Inference Engine\n\n"
            "Usage: llaminar2 [OPTIONS]\n";

        const std::string footer =
            "Examples:\n"
            "  llaminar2 -m model.gguf -p \"Hello world\" -n 50\n"
            "  llaminar2 -m model.gguf --chat\n"
            "  llaminar2 -m model.gguf --benchmark -n 100\n"
            "  llaminar2 -m model.gguf --tp 2 --tp-devices \"cuda:0,cuda:1\"\n"
            "  llaminar2 -m model.gguf -d cuda:0 --fused-attention-backend jit\n";

        return spec.getHelpText(header, footer);
    }

} // namespace llaminar2
