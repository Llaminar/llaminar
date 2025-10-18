/**
 * @file TensorHealthCheck.h
 * @brief Shared tensor validation utilities for detecting NaN, Inf, and uninitialized data
 * @author David Sanftenberg
 *
 * This header provides utilities for granular tensor health validation, used across
 * attention operators to detect numerical issues early.
 */

#pragma once

#include "../../Logger.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace llaminar
{

    /**
     * @brief Granular tensor validation (detects NaN, Inf, uninitialized data)
     *
     * Used to validate tensor health at critical stages of computation.
     * Provides detailed statistics and heuristics for detecting common issues:
     * - NaN values (computation errors)
     * - Inf values (overflow)
     * - Uninitialized data (extreme values or all zeros)
     */
    struct TensorHealthCheck
    {
        std::string name;
        int nan_count = 0;
        int inf_count = 0;
        int zero_count = 0;
        int normal_count = 0;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float abs_sum = 0.0f;

        TensorHealthCheck(const std::string &n) : name(n) {}

        /**
         * @brief Check tensor data for health issues
         * @param data Pointer to float array
         * @param size Number of elements
         */
        void check(const float *data, size_t size)
        {
            for (size_t i = 0; i < size; ++i)
            {
                float val = data[i];
                if (std::isnan(val))
                {
                    nan_count++;
                }
                else if (std::isinf(val))
                {
                    inf_count++;
                }
                else if (val == 0.0f)
                {
                    zero_count++;
                }
                else
                {
                    normal_count++;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    abs_sum += std::abs(val);
                }
            }
        }

        /**
         * @brief Check if tensor has no NaN/Inf and some normal values
         */
        bool is_healthy() const
        {
            return nan_count == 0 && inf_count == 0 && normal_count > 0;
        }

        /**
         * @brief Heuristic detection of uninitialized data
         *
         * Detects:
         * - Extreme values (>1e10) suggesting garbage memory
         * - All zeros when some values expected
         */
        bool is_uninitialized() const
        {
            return (normal_count > 0 && (std::abs(min_val) > 1e10f || std::abs(max_val) > 1e10f)) ||
                   (normal_count == 0 && zero_count > 0);
        }

        /**
         * @brief Log health status with optional rank prefix
         * @param rank MPI rank (-1 for no rank prefix)
         */
        void log(int rank = -1) const
        {
            std::string prefix = (rank >= 0) ? "[Rank " + std::to_string(rank) + "] " : "";
            if (!is_healthy())
            {
                LOG_ERROR(prefix << "UNHEALTHY " << name << ": NaN=" << nan_count
                                 << " Inf=" << inf_count << " Zero=" << zero_count
                                 << " Normal=" << normal_count);
                if (normal_count > 0)
                {
                    LOG_ERROR(prefix << "  Range: [" << min_val << ", " << max_val << "], Sum=" << abs_sum);
                }
            }
            else if (is_uninitialized())
            {
                LOG_WARN(prefix << "SUSPICIOUS " << name << ": Range [" << min_val << ", " << max_val
                                << "] suggests uninitialized data");
            }
            else
            {
                LOG_DEBUG(prefix << "HEALTHY " << name << ": " << normal_count << " values in ["
                                 << min_val << ", " << max_val << "], sum=" << abs_sum);
            }
        }
    };

} // namespace llaminar
