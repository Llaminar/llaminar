/**
 * @file RetentionIntentResolver.h
 * @brief §8.4 standard-field retention intent resolution (one typed path).
 *
 * Clients express retention ONLY through standard fields of a served API
 * (Chat Completions `prompt_cache_retention`, `prompt_cache_key`). This
 * resolver is the single typed path from those fields to a §8.4 intent:
 * it validates the value against the code-owned class map (RetentionClassMap),
 * applies the operator ceiling (P0 baseline §4: a class rank GREATER than the
 * ceiling is refused with a precise diagnostic — never silently demoted),
 * hashes an explicit cache key into a retention namespace (keyless requests
 * carry no namespace and are reachable only through their own request), and
 * returns the exact typed intent or a precise refusal. No mutation, no
 * allocation policy: the caller (serving path or tests) owns the
 * RetentionOwner and its bounded metadata.
 *
 * The namespace is BYTE IDENTITY ONLY: it partitions retention and
 * accounting (design §8.4) and never makes incompatible content shareable or
 * grants access across trust domains. Per-domain trust isolation is the
 * WIRING's job (the surge path applies the §8.1 trust-domain key alongside
 * the namespace); this resolver performs no access-control decision.
 */
#pragma once
#include "execution/prefix_cache/RetentionOwner.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace llaminar2
{
/** @brief One resolved standard-field intent (or a precise refusal). */
struct RetentionIntentResolution
{
    std::optional<RetentionIntent> intent;
    /** @brief Namespace derived from the explicit cache key; nullopt when the
     * request has no key (keyless: reachable only through its own request). */
    std::optional<std::uint64_t> namespace_id;
    std::string refusal; ///< Nonempty exactly when intent is empty.
    bool ok() const noexcept { return intent.has_value(); }
};

class RetentionIntentResolver final
{
public:
    /**
     * @param ceiling Optional operator ceiling on accepted classes.
     * @throws std::invalid_argument For an unrepresentable ceiling value
     * (only explicit, valid RetentionClass values are accepted).
     */
    explicit RetentionIntentResolver(std::optional<RetentionClass> ceiling);

    /**
     * @brief Resolve the two standard fields.
     * @param cache_retention The standard `prompt_cache_retention` value, or
     * empty when absent (absent -> code-owned default class).
     * @param cache_key The standard `prompt_cache_key` value, or empty when
     * absent (absent -> keyless, nullopt namespace).
     * @return The typed intent (default when the value is absent) or a
     * precise refusal — never a silent remap, never a demoted class.
     */
    RetentionIntentResolution resolve(std::string_view cache_retention,
        std::string_view cache_key) const;

    /** @return The operator ceiling (nullopt = no ceiling). */
    std::optional<RetentionClass> ceiling() const noexcept { return ceiling_; }

private:
    std::optional<RetentionClass> ceiling_;
};
} // namespace llaminar2
