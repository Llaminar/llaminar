/**
 * @file ActivationPrecisionPolicy.h
 * @brief Admission policy for production model activation precision.
 *
 * Tensor and kernel support for a dtype does not establish model-graph support.
 * Production graphs currently expose FP32 activations. CLI, configuration
 * validation, and parity discovery share this check so an unsupported request
 * cannot be advertised as a successfully tested precision.
 */
#pragma once

#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2
{
    /**
     * @brief Reject activation modes without a production implementation.
     * @param precision Requested activation spelling, case insensitive.
     * @throws std::invalid_argument When the request is not FP32.
     *
     * KV storage, weight formats, and kernel-local quantized operands have
     * independent contracts and must not use this model activation check.
     */
    inline void requireImplementedActivationPrecision(std::string_view precision)
    {
        std::string normalized(precision);
        for (char &character : normalized)
            character = static_cast<char>(
                std::tolower(static_cast<unsigned char>(character)));
        if (normalized != "fp32")
            throw std::invalid_argument(
                "Activation precision '" + std::string(precision) +
                "' is unimplemented for production inference; only fp32 is supported");
    }
}
