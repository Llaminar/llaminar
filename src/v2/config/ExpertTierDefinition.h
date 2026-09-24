/**
 * @file ExpertTierDefinition.h
 * @brief Compact user intent for one hardware domain and its expert tier.
 *
 * This is a configuration adapter, not another placement controller. It splits
 * the tier's priority/capacity from the canonical execution-domain declaration,
 * then returns the existing typed runtime inputs. Omitted ownership, scope and
 * collective remain unresolved until cluster inventory binding. No device,
 * rank, memory quota or backend is guessed from a friendly tier name.
 */
#pragma once

#include "config/ExecutionDomainDefinition.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"

#include <charconv>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2
{
    /** @brief A single parsed declaration, consumed by the existing plan owner. */
    struct ExpertTierDefinition
    {
        RoutedExpertDomain domain;
        RoutedExpertTier tier;

        /**
         * @brief Parse `name=devices;priority=N` with optional domain/capacity fields.
         * @param declaration Canonical device selectors and integer tier priority.
         * @return Domain and tier with automatic capacity and explicit overrides.
         * @throws std::invalid_argument for missing, duplicate or malformed fields.
         *
         * GPU and CPU participants use the same existing apportioned-expert
         * default. Tensor-sharded and replicated compute remain explicit,
         * independently selectable policies rather than backend-name guesses.
         */
        static ExpertTierDefinition parse(const std::string &declaration)
        {
            std::istringstream input(declaration);
            std::string head;
            std::getline(input, head, ';');
            const auto equal = head.find('=');
            if (equal == std::string::npos)
                throw std::invalid_argument(
                    "Expert tier requires name=devices;priority=N");
            const std::string name = trim(head.substr(0, equal));
            std::string hardware = head;
            std::string placement = name + "@" + name;
            std::string field;
            std::set<std::string> keys;
            while (std::getline(input, field, ';'))
            {
                field = trim(field);
                if (field.empty())
                    throw std::invalid_argument("Empty expert tier option");
                const auto separator = field.find('=');
                if (separator == std::string::npos)
                    throw std::invalid_argument("Expert tier option requires key=value: " + field);
                const auto key = normalizedKey(field.substr(0, separator));
                if (!keys.insert(key).second)
                    throw std::invalid_argument("Duplicate expert tier option: " + key);
                if (key == "priority" || key == "memory_mb" ||
                    key == "max_experts_per_layer")
                    placement += ";" + field;
                else
                    hardware += ";" + field;
            }
            return {parseDomain(hardware), parsePlacement(placement)};
        }

        /**
         * @brief Parse hardware intent using the shared execution-domain grammar.
         * @return Existing routed-domain type; AUTO scope awaits inventory binding.
         * @throws std::invalid_argument for unsupported or malformed declarations.
         */
        static RoutedExpertDomain parseDomain(const std::string &declaration)
        {
            ExecutionDomainParseOptions options;
            options.context = "MoE routed-expert domain";
            options.allow_global_scope = false;
            return RoutedExpertDomain::fromExecutionDomainDefinition(
                ExecutionDomainDefinition::parse(declaration, options));
        }

        /**
         * @brief Parse `tier@domain;priority=N` for compact and expanded surfaces.
         * @return Tier whose zero budget/cap is resolved by physical admission.
         * @throws std::invalid_argument for unknown fields, overflow or duplicates.
         */
        static RoutedExpertTier parsePlacement(const std::string &declaration)
        {
            std::istringstream input(declaration);
            std::string head;
            std::getline(input, head, ';');
            const auto at = head.find('@');
            if (at == std::string::npos)
                throw std::invalid_argument("Expert tier requires name@domain;priority=N");
            RoutedExpertTier tier;
            tier.name = trim(head.substr(0, at));
            tier.domain = trim(head.substr(at + 1));
            if (tier.name.empty() || tier.domain.empty())
                throw std::invalid_argument("Expert tier requires a nonempty name and domain");
            std::set<std::string> keys;
            std::string field;
            while (std::getline(input, field, ';'))
            {
                const auto equal = field.find('=');
                if (equal == std::string::npos)
                    throw std::invalid_argument("Expert tier option requires key=value: " + field);
                const auto key = normalizedKey(field.substr(0, equal));
                const auto value = trim(field.substr(equal + 1));
                if (!keys.insert(key).second)
                    throw std::invalid_argument("Duplicate expert tier option: " + key);
                if (key == "priority")
                    tier.priority = integer<int>(value, key);
                else if (key == "max_experts_per_layer")
                {
                    tier.max_experts_per_layer = integer<int>(value, key);
                    if (tier.max_experts_per_layer < 0)
                        throw std::invalid_argument(key + " must be nonnegative");
                }
                else if (key == "memory_mb")
                {
                    if (normalizedKey(value) == "auto")
                        continue;
                    const auto mb = integer<std::size_t>(value, key);
                    constexpr std::size_t mib = 1024u * 1024u;
                    if (mb > std::numeric_limits<std::size_t>::max() / mib)
                        throw std::invalid_argument("Expert tier memory-mb overflows bytes");
                    tier.memory_budget_bytes = mb * mib;
                }
                else
                    throw std::invalid_argument("Unknown expert tier option: " + key);
            }
            if (!keys.contains("priority"))
                throw std::invalid_argument("Expert tier '" + tier.name + "' is missing priority=N");
            return tier;
        }

    private:
        /** @return Whitespace-free ends without changing embedded device syntax. */
        static std::string trim(const std::string &value)
        {
            const auto first = value.find_first_not_of(" \t\r\n");
            if (first == std::string::npos)
                return {};
            return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
        }

        /** @return Canonical option key shared by dash/underscore CLI spellings. */
        static std::string normalizedKey(const std::string &value)
        {
            auto key = trim(value);
            for (auto &character : key)
            {
                if (character == '-') character = '_';
                if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
            }
            return key;
        }

        /** @brief Parse one whole integer; never accept a numeric prefix or wrap. */
        template <typename Integer>
        static Integer integer(const std::string &value, const std::string &key)
        {
            Integer result{};
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
                throw std::invalid_argument("Invalid expert tier " + key + ": " + value);
            return result;
        }
    };
}
