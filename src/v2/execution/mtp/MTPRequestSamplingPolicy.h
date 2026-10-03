/**
 * @file MTPRequestSamplingPolicy.h
 * @brief Resolve MTP execution semantics from the admitted request sampling law.
 *
 * Verification capability belongs to the retained/request policy. The sampler
 * determines which specialization executes: greedy requests do not build
 * probability distributions and must not inherit stochastic depth economics.
 * This projection is ephemeral; it never mutates graph capacity, placement or
 * the authored policy, so successive greedy/stochastic requests share one
 * retained graph family without a second configuration authority.
 */
#pragma once

#include "../config/RuntimeConfig.h"
#include "../../utils/Sampler.h"
#include "../../backends/DeviceId.h"

#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief A request selected an unimplemented GPU MTP sampling feature.
     *
     * This typed admission failure is distinct from a broken graph or backend.
     * HTTP may report it as a client error; ordinary execution failures remain
     * fatal. Its code and parameter are a stable protocol, not a parsed log.
     */
    class UnsupportedMTPSamplingRequest final : public std::invalid_argument
    {
    public:
        /** @brief Name the missing device-owned DRY sequence-history contract. */
        UnsupportedMTPSamplingRequest()
            : std::invalid_argument(
                  "GPU MTP does not implement device-owned DRY sequence history; "
                  "dry_multiplier must be zero or dry_penalty_last_n must be zero")
        {
        }

        static constexpr const char *code = "unsupported_mtp_sampling_policy"; ///< HTTP error identity.
        static constexpr const char *parameter = "dry_multiplier"; ///< Unsupported request field.
    };

    /**
     * @brief Admit a sampling policy before modifying request state or dispatching ranks.
     * @param device Authoritative continuation endpoint, not the offload tier.
     * @param mtp Active request policy; ordinary inference is not restricted here.
     * @param sampling Effective sampler after model-default/request merging.
     * @throws UnsupportedMTPSamplingRequest For enabled GPU MTP with active DRY.
     *
     * Greedy and stochastic MTP have the same missing history contract. CPU DRY
     * remains first class, and explicitly disabling DRY does not reject a request.
     * No request is silently converted to non-MTP or a host sampler.
     */
    inline void requireSupportedMTPSamplingRequest(
        const DeviceId &device, const MTPRuntimeConfig &mtp, const SamplingParams &sampling)
    {
        if (device.is_gpu() && mtp.enabled && sampling.dry_multiplier != 0.0f &&
            sampling.dry_penalty_last_n != 0)
            throw UnsupportedMTPSamplingRequest{};
    }

    /**
     * @brief Compose an active request with its exact sampling specialization.
     * @param retained Immutable physical graph/weight envelope.
     * @param requested Request-selectable verification and depth intent.
     * @param sampling Admitted sampler, including argmax-equivalent top-k rules.
     * @return Execution view; stochastic capability specializes only for argmax.
     *
     * An explicitly greedy-only policy remains greedy even for an incompatible
     * sampler. The runner rejects that pair before decode instead of upgrading
     * an operator's explicit policy or silently bypassing MTP.
     */
    [[nodiscard]] inline MTPRuntimeConfig resolveMTPSamplingRequestConfig(
        const MTPRuntimeConfig &retained,
        const MTPRequestPolicy &requested,
        const SamplingParams &sampling)
    {
        auto active = composeMTPRequestConfig(retained, requested);
        if (active.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            sampling.is_greedy())
        {
            active.verify_mode = MTPVerifyMode::Greedy;
        }
        return active;
    }
}
