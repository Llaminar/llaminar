/**
 * @file OrchestrationStartupPolicy.cpp
 * @brief Shared setup-only publication for CLI, saved config, and MPI admission.
 *
 * Serialize the canonical bucket ladder, never a second arithmetic estimate.
 * Refresh DebugEnv after publication because logging may have read its snapshot
 * before argument parsing. Kernels and capacity planning then observe exactly
 * the same policy as newly exec'd MPI children inheriting the environment.
 */
#include "OrchestrationStartupPolicy.h"
#include "OrchestrationConfig.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "planning/ModelMemoryProfile.h"
#include "utils/DebugEnv.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    void resolveMTPStartupPolicy(OrchestrationConfig &config, const ModelMemoryProfile &model)
    {
        switch (config.mtp_activation_policy)
        {
        case MTPActivationPolicy::Automatic:
            // A programmatically supplied enabled runtime policy is explicit.
            // Otherwise the learned-block count selects the automatic intent;
            // directory completeness is required before any BOM is admitted.
            config.mtp.enabled = config.mtp.enabled || model.mtp_layer_count > 0;
            break;
        case MTPActivationPolicy::Enabled:
            config.mtp.enabled = true;
            break;
        case MTPActivationPolicy::Disabled:
            config.mtp.enabled = false;
            break;
        }
        if (retainsMTPGraphCapacity(config.mtp))
        {
            const auto manifest = discoverMTPWeightManifest(model, true);
            if (!manifest.available)
            {
                auto diagnostic = manifest.diagnostic;
                if (!manifest.missing_required.empty())
                    diagnostic += "; missing required tensor: " + manifest.missing_required.front();
                throw std::invalid_argument(diagnostic +
                    "; supply complete learned MTP/NextN weights or use --no-mtp "
                    "without retained MTP graph capacity");
            }
        }
        config.mtp_activation_policy = config.mtp.enabled
            ? MTPActivationPolicy::Enabled : MTPActivationPolicy::Disabled;
    }

    void publishOrchestrationStartupPolicy(const OrchestrationConfig &config)
    {
        std::optional<std::string> buckets;
        if (config.prefill_max_bucket_size)
        {
            // Validate before any publication, including deterministic policy.
            std::ostringstream serialized;
            for (const int rows : prefillGraphBucketSizes(*config.prefill_max_bucket_size))
            {
                if (serialized.tellp() > 0) serialized << ',';
                serialized << rows;
            }
            buckets = serialized.str();
        }
        if (buckets && setenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", buckets->c_str(), 1) != 0)
            throw std::runtime_error("Could not publish captured-prefill startup policy");
        if (config.deterministic && setenv("LLAMINAR_DETERMINISTIC", "1", 1) != 0)
            throw std::runtime_error("Could not publish deterministic startup policy");
        if (buckets || config.deterministic) mutableDebugEnv().reload();
    }
}
