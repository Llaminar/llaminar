/**
 * @file MTPDepthLearnedPolicy.h
 * @brief One generated-rule interpreter for CPU and device-resident MTP.
 *
 * The trainer owns the rule rows. Admission supplies immutable backend/model/
 * sampling keys; the live controller supplies integer window observations.
 * No clocks, host polling, tensor transfers or additional allocations occur.
 */
#pragma once

#include "../config/MTPDepthPolicyTypes.h"
#include "../config/MTPDepthPolicyScope.h"
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_POLICY_HD __host__ __device__ inline
#else
#define LLAMINAR_MTP_POLICY_HD inline
#endif

namespace llaminar2
{
    /** @brief Offline trainer ABI; rule order defines first-match precedence. */
    struct MTPGeneratedDepthPolicyRule
    {
        MTPVerifyMode verify_mode;
        MTPDepthPolicyBackend backend;
        MTPDepthPolicyModelClass model_class;
        int current_depth;
        double min_acceptance_rate;
        double max_acceptance_rate;
        double max_zero_accept_rate;
        double min_full_accept_rate;
        int depth_delta;
        const char *label;
        MTPDepthPolicyScope scope{}; ///< Absent only for preserved generic rows.
    };

    /**
     * @brief Offline-measured admission choice, separate from live transitions.
     *
     * Several depths can have profitable hold intervals. Their numeric order
     * does not rank request-wide economics. The trainer chooses this warm start
     * from measured training throughput; admission publishes it through the
     * existing host/device depth contract without changing adaptive capacity.
     */
    struct MTPGeneratedDepthPolicyStartup
    {
        MTPVerifyMode verify_mode;
        MTPDepthPolicyBackend backend;
        MTPDepthPolicyModelClass model_class;
        int initial_depth;
        MTPDepthPolicyScope scope{}; ///< Admission and transitions select the same domain.
    };

#include "MTPDepthPolicyGenerated.inc"

    /** @brief Immutable learned-policy identity, carried in the resident row. */
    struct MTPLearnedDepthPolicyContext
    {
        bool enabled = false;
        MTPDepthPolicyBackend backend = MTPDepthPolicyBackend::Any;
        MTPDepthPolicyModelClass model_class = MTPDepthPolicyModelClass::Any;
        MTPVerifyMode verify_mode = MTPVerifyMode::Greedy;
        MTPDepthPolicyScope scope;

        /** @return Complete immutable learned-policy identity. */
        bool operator==(const MTPLearnedDepthPolicyContext &) const = default;

        /** @return Whether every key is a supported value, even when disabled. */
        LLAMINAR_MTP_POLICY_HD bool valid() const
        {
            return (backend == MTPDepthPolicyBackend::Any ||
                    backend == MTPDepthPolicyBackend::CPU ||
                    backend == MTPDepthPolicyBackend::CUDA ||
                    backend == MTPDepthPolicyBackend::ROCm) &&
                   (model_class == MTPDepthPolicyModelClass::Any ||
                    model_class == MTPDepthPolicyModelClass::Dense ||
                    model_class == MTPDepthPolicyModelClass::MoE) &&
                   scope.valid() &&
                   (!scope.specified() ||
                    ((backend == MTPDepthPolicyBackend::CUDA || backend == MTPDepthPolicyBackend::ROCm) &&
                     (model_class == MTPDepthPolicyModelClass::Dense || model_class == MTPDepthPolicyModelClass::MoE) &&
                     ((model_class == MTPDepthPolicyModelClass::MoE) == (scope.expert_count > 0)))) &&
                   (verify_mode == MTPVerifyMode::Greedy ||
                    verify_mode == MTPVerifyMode::SpeculativeSampling);
        }
    };

    /** @brief Installed tables; metadata tests may provide synthetic tables without installing defaults. */
    struct MTPInstalledDepthPolicyTables
    {
        static constexpr const auto &rules = kMTPGeneratedDepthPolicyRules;
        static constexpr const auto &startups = kMTPGeneratedDepthPolicyStartups;
        /** @return One embedded rule value without a device read of a host reference member. */
        template <size_t Index>
        LLAMINAR_MTP_POLICY_HD static constexpr MTPGeneratedDepthPolicyRule ruleAt()
        {
            constexpr auto row = kMTPGeneratedDepthPolicyRules[Index];
            return row;
        }
        /** @return One embedded startup value without materializing a host table pointer. */
        template <size_t Index>
        LLAMINAR_MTP_POLICY_HD static constexpr MTPGeneratedDepthPolicyStartup startupAt()
        {
            constexpr auto row = kMTPGeneratedDepthPolicyStartups[Index];
            return row;
        }
    };

    /** @return An installed embedded startup, or a synthetic host-test table row. */
    template <size_t Index, typename Tables>
    LLAMINAR_MTP_POLICY_HD constexpr MTPGeneratedDepthPolicyStartup mtpLearnedStartupAt()
    {
        if constexpr (requires { Tables::template startupAt<Index>(); })
            return Tables::template startupAt<Index>();
        else
            return Tables::startups[Index];
    }

    /** @return An installed embedded transition, or a synthetic host-test table row. */
    template <size_t Index, typename Tables>
    LLAMINAR_MTP_POLICY_HD constexpr MTPGeneratedDepthPolicyRule mtpLearnedRuleAt()
    {
        if constexpr (requires { Tables::template ruleAt<Index>(); })
            return Tables::template ruleAt<Index>();
        else
            return Tables::rules[Index];
    }

    /** @return Whether immutable backend/model/law keys select this row. */
    template <typename Row>
    LLAMINAR_MTP_POLICY_HD bool mtpLearnedDomainKeysMatch(
        const Row &row, const MTPLearnedDepthPolicyContext &context)
    {
        return row.verify_mode == context.verify_mode &&
            (row.backend == MTPDepthPolicyBackend::Any || row.backend == context.backend) &&
            (row.model_class == MTPDepthPolicyModelClass::Any || row.model_class == context.model_class);
    }

    /**
     * @brief Select a complete measured scope before any rule or startup predicate.
     * @tparam Index Compile-time table row; no device table pointer is materialized.
     * @tparam Tables One immutable generated artifact containing startup and transition rows.
     * @param context Complete frozen runtime identity.
     * @return Whether this identity has a measured scoped domain.
     *
     * Every generated transition domain must have a startup row. Its existence,
     * independent of current depth, bounds, or acceptance, suppresses all generic
     * rows in that domain. A new interval miss cannot reactivate an old hold.
     */
    template <size_t Index = 0, typename Tables = MTPInstalledDepthPolicyTables>
    LLAMINAR_MTP_POLICY_HD bool hasMTPMeasuredScope(const MTPLearnedDepthPolicyContext &context)
    {
        if constexpr (Index < sizeof(Tables::startups) / sizeof(Tables::startups[0]))
        {
            constexpr auto choice = mtpLearnedStartupAt<Index, Tables>();
            if (choice.scope.specified() && choice.scope == context.scope &&
                mtpLearnedDomainKeysMatch(choice, context))
                return true;
            return hasMTPMeasuredScope<Index + 1, Tables>(context);
        }
        return false;
    }

    /** @return Whether the row belongs to the already selected whole domain. */
    template <typename Row>
    LLAMINAR_MTP_POLICY_HD bool mtpLearnedSelectedDomainMatch(
        const Row &row, const MTPLearnedDepthPolicyContext &context, bool measured_scope)
    {
        return mtpLearnedDomainKeysMatch(row, context) &&
            (measured_scope ? row.scope == context.scope : !row.scope.specified());
    }

    /**
     * @brief Resolve measured startup in the same domain used by live windows.
     * @param context Immutable admitted identity; callers enforce dynamic/enabled intent.
     * @param minimum Minimum admitted selector.
     * @param maximum Maximum admitted selector.
     * @return First applicable bounded startup, or -1 without entering another domain.
     */
    template <size_t Index = 0, typename Tables = MTPInstalledDepthPolicyTables>
    LLAMINAR_MTP_POLICY_HD int resolveMTPLearnedStartupInDomain(
        const MTPLearnedDepthPolicyContext &context, int minimum, int maximum, bool measured_scope)
    {
        if constexpr (Index < sizeof(Tables::startups) / sizeof(Tables::startups[0]))
        {
            constexpr auto choice = mtpLearnedStartupAt<Index, Tables>();
            if (mtpLearnedSelectedDomainMatch(choice, context, measured_scope) &&
                choice.initial_depth >= minimum && choice.initial_depth <= maximum)
                return choice.initial_depth;
            return resolveMTPLearnedStartupInDomain<Index + 1, Tables>(context, minimum, maximum, measured_scope);
        }
        return -1;
    }

    /** @brief Validate and select the whole startup domain once before table traversal. */
    template <size_t Index = 0, typename Tables = MTPInstalledDepthPolicyTables>
    LLAMINAR_MTP_POLICY_HD int resolveMTPLearnedStartup(
        const MTPLearnedDepthPolicyContext &context, int minimum, int maximum)
    {
        if (!context.valid()) return -1;
        const bool measured_scope = hasMTPMeasuredScope<0, Tables>(context);
        return resolveMTPLearnedStartupInDomain<Index, Tables>(context, minimum, maximum, measured_scope);
    }

    /** @brief Integer observations; zero-denominator windows never match. */
    struct MTPLearnedDepthWindow
    {
        uint64_t attempted = 0;
        uint64_t accepted = 0;
        uint64_t runs = 0;
        uint64_t zero_accepts = 0;
        uint64_t full_accepts = 0;
    };

    /** @brief A hold is a real match, distinct from no learned recommendation. */
    struct MTPLearnedDepthMatch
    {
        bool matched = false;
        int depth_delta = 0;
    };

    /**
     * @brief Validated learned recommendation without retaining the large scope context.
     * Invalid immutable identity remains distinct from a valid learned miss/hold.
     * This value is temporary arithmetic output, never a second controller state.
     */
    struct MTPLearnedDepthDecision
    {
        bool valid = false;
        MTPLearnedDepthMatch match;
    };

    /** @brief Convert the trainer's six-decimal rate to an exact integer key. */
    LLAMINAR_MTP_POLICY_HD constexpr uint64_t mtpLearnedRatePPM(double rate)
    {
        return static_cast<uint64_t>(rate * 1'000'000.0 + 0.5);
    }

    /**
     * @brief Interpret generated rows in their original order on either owner.
     * @tparam Index Compile-time row ordinal, starting at zero.
     * @param context Immutable admitted policy identity.
     * @param depth Current live speculative depth.
     * @param window Completed verifier observations at that depth.
     * @return First learned match, including delta-zero holds.
     *
     * Compile-time traversal embeds scalar constants in GPU instructions. It
     * avoids a host table pointer or a per-thread array that could spill. Both
     * owners compare integer cross-products against the trainer's PPM rates;
     * neither rounds an observed acceptance ratio at a threshold boundary.
     * A positive learned delta also needs one actually observed full prefix:
     * a high partial-prefix average cannot prove that another draft can help.
     */
    template <size_t Index = 0, typename Tables = MTPInstalledDepthPolicyTables>
    LLAMINAR_MTP_POLICY_HD MTPLearnedDepthMatch matchMTPLearnedDepthPolicyInDomain(
        const MTPLearnedDepthPolicyContext &context, int depth,
        const MTPLearnedDepthWindow &window, bool measured_scope)
    {
        if constexpr (Index < sizeof(Tables::rules) / sizeof(Tables::rules[0]))
        {
            constexpr auto rule = mtpLearnedRuleAt<Index, Tables>();
            if (mtpLearnedSelectedDomainMatch(rule, context, measured_scope) &&
                rule.current_depth == depth &&
                (rule.depth_delta <= 0 || window.full_accepts > 0) &&
                window.accepted * 1'000'000 >= window.attempted * mtpLearnedRatePPM(rule.min_acceptance_rate) &&
                window.accepted * 1'000'000 <= window.attempted * mtpLearnedRatePPM(rule.max_acceptance_rate) &&
                window.zero_accepts * 1'000'000 <= window.runs * mtpLearnedRatePPM(rule.max_zero_accept_rate) &&
                window.full_accepts * 1'000'000 >= window.runs * mtpLearnedRatePPM(rule.min_full_accept_rate))
            {
                return {.matched = true, .depth_delta = rule.depth_delta};
            }
            return matchMTPLearnedDepthPolicyInDomain<Index + 1, Tables>(context, depth, window, measured_scope);
        }
        return {};
    }
    /**
     * @brief Validate and select the measured domain once, then interpret ordered predicates.
     * @param context Frozen admission keys; no mutable execution state.
     * @param depth Current device- or CPU-owned live selector.
     * @param window Completed integer observations from that same owner.
     * @return Typed identity validity plus the exact recommendation, including valid misses.
     */
    template <size_t Index = 0, typename Tables = MTPInstalledDepthPolicyTables>
    LLAMINAR_MTP_POLICY_HD MTPLearnedDepthDecision evaluateMTPLearnedDepthPolicy(
        const MTPLearnedDepthPolicyContext &context, int depth, const MTPLearnedDepthWindow &window)
    {
        if (!context.valid()) return {};
        if (!context.enabled || window.attempted == 0 || window.runs == 0)
            return {.valid = true};
        const bool measured_scope = hasMTPMeasuredScope<0, Tables>(context);
        return {.valid = true,
            .match = matchMTPLearnedDepthPolicyInDomain<Index, Tables>(context, depth, window, measured_scope)};
    }

    /**
     * @brief Preserve the ordinary CPU matcher API over the single validated evaluator.
     * @param context Complete immutable admission identity.
     * @param depth Current owner-selected depth.
     * @param window Exact integer observations; no rates or predicates are changed.
     * @return Learned match, or an ordinary miss for invalid/disabled identity.
     */
    template <size_t Index = 0, typename Tables = MTPInstalledDepthPolicyTables>
    LLAMINAR_MTP_POLICY_HD MTPLearnedDepthMatch matchMTPLearnedDepthPolicy(
        const MTPLearnedDepthPolicyContext &context, int depth, const MTPLearnedDepthWindow &window)
    {
        return evaluateMTPLearnedDepthPolicy<Index, Tables>(context, depth, window).match;
    }
}

#undef LLAMINAR_MTP_POLICY_HD
