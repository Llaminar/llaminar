/**
 * @file BenchmarkGeometrySelection.h
 * @brief Strict shared admission for benchmark row, capacity and device inventories.
 *
 * An absent selector uses the caller's declared inventory. A malformed explicit
 * selector must fail instead of silently measuring that default or dropping a
 * requested shape. This device-free helper is shared by CUDA and ROCm probes;
 * it preserves requested order and rejects duplicates that would bias samples.
 */
#pragma once

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Admit one exact positive-integer inventory without accessing hardware.
     * @param name Selector identity retained in every admission error.
     * @param value Explicit CSV, or nullopt when the selector was not supplied.
     * @param defaults Declared inventory used only for an absent selector.
     * @return Distinct positive integers in the declared/requested order.
     * @throws std::runtime_error Empty, malformed, overflowing or duplicate entries.
     */
    inline std::vector<int> parseBenchmarkGeometry(
        std::string_view name,
        std::optional<std::string_view> value,
        std::initializer_list<int> defaults)
    {
        const auto invalid = [name]() -> void {
            throw std::runtime_error(std::string(name) +
                " requires a nonempty comma-separated inventory of distinct positive integers");
        };
        std::vector<int> result;
        const auto append = [&](int number) {
            if (number <= 0 || std::find(result.begin(), result.end(), number) != result.end())
                invalid();
            result.push_back(number);
        };
        if (!value)
        {
            for (const int number : defaults)
                append(number);
        }
        else
        {
            // A supplied empty/malformed token is not absence. Parse every
            // byte, including a trailing comma, before admitting any workload.
            std::string_view remaining = *value;
            for (;;)
            {
                const auto comma = remaining.find(',');
                auto token = remaining.substr(0, comma);
                while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front())))
                    token.remove_prefix(1);
                while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back())))
                    token.remove_suffix(1);
                if (token.empty())
                    invalid();
                int number = 0;
                const auto parsed = std::from_chars(token.data(), token.data() + token.size(), number);
                if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
                    invalid();
                append(number);
                if (comma == std::string_view::npos)
                    break;
                remaining.remove_prefix(comma + 1);
            }
        }
        if (result.empty())
            invalid();
        return result;
    }

    /**
     * @brief Read one environment selector through the same strict admission contract.
     * @param name Exact environment key; an unset key alone selects defaults.
     * @param defaults Caller-owned standard geometry inventory.
     * @return Fully admitted geometry before allocation, capture or timing.
     * @throws std::runtime_error An invalid explicit or default inventory.
     */
    inline std::vector<int> benchmarkGeometryIntegers(
        const char *name, std::initializer_list<int> defaults)
    {
        const char *value = std::getenv(name);
        return parseBenchmarkGeometry(name,
            value ? std::optional<std::string_view>(value) : std::nullopt, defaults);
    }
}
