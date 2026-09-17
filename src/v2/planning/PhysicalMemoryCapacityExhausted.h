/**
 * @file PhysicalMemoryCapacityExhausted.h
 * @brief Typed rejection of a valid physical-memory candidate that does not fit.
 *
 * Automatic admission may compare another topology or retained graph capacity
 * after this outcome. Malformed geometry, missing owners, arithmetic overflow
 * and stale lifecycle grants are different failures and must not be swallowed
 * as memory pressure. This exception carries a diagnostic, never a byte ledger.
 */
#pragma once
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief A complete, valid BOM exceeds its observed or explicitly limited capacity. */
    class PhysicalMemoryCapacityExhausted : public std::invalid_argument
    {
    public:
        /** @brief Preserve the canonical authority's limiting-resource diagnostic. */
        explicit PhysicalMemoryCapacityExhausted(const std::string &message)
            : std::invalid_argument(message) {}
    };
}
