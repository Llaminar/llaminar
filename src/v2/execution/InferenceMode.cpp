/**
 * @file InferenceMode.cpp
 * @brief Implementation of InferenceMode
 */

#include "InferenceMode.h"
#include "../models/qwen/Qwen2Graph.h" // For Qwen2ActivationBuffers
#include <sstream>

namespace llaminar2
{

    InferenceMode::InferenceMode(ActivationPrecision precision)
        : precision_(precision)
    {
    }

    std::string InferenceMode::name() const
    {
        switch (precision_)
        {
        case ActivationPrecision::FP32:
            return "FP32";
        case ActivationPrecision::BF16:
            return "BF16";
        case ActivationPrecision::FP16:
            return "FP16";
        case ActivationPrecision::Q8_1:
            return "Q8_1";
        case ActivationPrecision::Hybrid:
            return "Hybrid";
        default:
            return "Unknown";
        }
    }

    std::vector<std::string> InferenceMode::extraRequiredBuffers() const
    {
        std::vector<std::string> buffers;
        if (needsQRope())
            buffers.push_back("Q_rope");
        if (needsKRope())
            buffers.push_back("K_rope");
        if (needsVDequant())
            buffers.push_back("V_dequant");
        return buffers;
    }

    InferenceMode::ValidationResult InferenceMode::validateBuffers(
        const Qwen2ActivationBuffers &buffers) const
    {
        ValidationResult result;

        // Core buffers required by all modes
        if (!buffers.Q)
        {
            result.valid = false;
            result.missing_buffers.push_back("Q");
        }
        if (!buffers.K)
        {
            result.valid = false;
            result.missing_buffers.push_back("K");
        }
        if (!buffers.V)
        {
            result.valid = false;
            result.missing_buffers.push_back("V");
        }
        if (!buffers.attn_output)
        {
            result.valid = false;
            result.missing_buffers.push_back("attn_output");
        }

        // Mode-specific buffers
        if (needsQRope() && !buffers.Q_rope)
        {
            result.valid = false;
            result.missing_buffers.push_back("Q_rope");
        }
        if (needsKRope() && !buffers.K_rope)
        {
            result.valid = false;
            result.missing_buffers.push_back("K_rope");
        }
        if (needsVDequant() && !buffers.V_dequant)
        {
            result.valid = false;
            result.missing_buffers.push_back("V_dequant");
        }

        // Build error message
        if (!result.valid)
        {
            std::ostringstream oss;
            oss << "InferenceMode::" << name() << " requires buffers: ";
            for (size_t i = 0; i < result.missing_buffers.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << result.missing_buffers[i];
            }
            result.error_message = oss.str();
        }

        return result;
    }

    bool isHybridModeActive(const InferenceMode &mode, const Qwen2ActivationBuffers &buffers)
    {
        // Hybrid mode is active when:
        // 1. Mode is any Hybrid variant (Hybrid or HybridQ16)
        // 2. Required Hybrid buffers are allocated (Q_rope, K_rope for RoPE)
        return mode.isAnyHybrid() && buffers.Q_rope && buffers.K_rope;
    }

} // namespace llaminar2
