#pragma once

#include "logger.h"
#include "tensors/tensor_base.h"
#include <memory>
#include <cmath>
#include <string>

namespace llaminar
{

// Debug assertion that only fires in debug builds
#ifdef NDEBUG
#define DEBUG_ASSERT(condition, message) ((void)0)
#else
#define DEBUG_ASSERT(condition, message)                                                            \
    do                                                                                              \
    {                                                                                               \
        if (!(condition))                                                                           \
        {                                                                                           \
            LOG_ERROR("DEBUG_ASSERT FAILED: " << message << " at " << __FILE__ << ":" << __LINE__); \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)
#endif

// Tensor validation assertions
#define ASSERT_TENSOR_NOT_NULL(tensor, name) \
    DEBUG_ASSERT(tensor != nullptr, "Tensor " << name << " is null")

#define ASSERT_TENSOR_VALID(tensor, name)                                               \
    do                                                                                  \
    {                                                                                   \
        ASSERT_TENSOR_NOT_NULL(tensor, name);                                           \
        DEBUG_ASSERT(tensor->data() != nullptr, "Tensor " << name << " has null data"); \
        DEBUG_ASSERT(tensor->size() > 0, "Tensor " << name << " has zero size");        \
    } while (0)

#define ASSERT_TENSOR_NOT_NAN(tensor, message)                                             \
    do                                                                                     \
    {                                                                                      \
        if (tensor && tensor->data())                                                      \
        {                                                                                  \
            const float *data = tensor->data();                                            \
            int max_check = std::min(tensor->size(), 10);                                  \
            for (int i = 0; i < max_check; ++i)                                            \
            {                                                                              \
                if (std::isnan(data[i]))                                                   \
                {                                                                          \
                    LOG_ERROR("NaN detected in tensor at index " << i << ": " << message); \
                    abort();                                                               \
                }                                                                          \
            }                                                                              \
        }                                                                                  \
    } while (0)

#define ASSERT_TENSOR_NOT_ALL_ZEROS(tensor, name)                                                                \
    do                                                                                                           \
    {                                                                                                            \
        ASSERT_TENSOR_VALID(tensor, name);                                                                       \
        const float *data = tensor->data();                                                                      \
        bool has_nonzero = false;                                                                                \
        for (size_t i = 0; i < std::min(tensor->size(), size_t(100)); ++i)                                       \
        {                                                                                                        \
            if (data[i] != 0.0f)                                                                                 \
            {                                                                                                    \
                has_nonzero = true;                                                                              \
                break;                                                                                           \
            }                                                                                                    \
        }                                                                                                        \
        DEBUG_ASSERT(has_nonzero, "Tensor " << name << " appears to be all zeros (checked first 100 elements)"); \
    } while (0)

    // Tensor logging utilities
    class TensorLogger
    {
    public:
        static void logTensorStats(const std::shared_ptr<TensorBase> &tensor, const std::string &name, const std::string &stage = "")
        {
            if (!tensor)
            {
                LOG_TRACE("[" << stage << "] Tensor " << name << " is NULL");
                return;
            }

            const float *data = tensor->data();
            if (!data)
            {
                LOG_TRACE("[" << stage << "] Tensor " << name << " has NULL data");
                return;
            }

            size_t size = tensor->size();
            auto shape = tensor->shape();

            // Shape info
            std::string shape_str = "[";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (i > 0)
                    shape_str += ", ";
                shape_str += std::to_string(shape[i]);
            }
            shape_str += "]";

            // First few values
            std::string values_str = "";
            size_t max_show = std::min(size, size_t(5));
            for (size_t i = 0; i < max_show; ++i)
            {
                if (i > 0)
                    values_str += ", ";
                values_str += std::to_string(data[i]);
            }
            if (size > max_show)
                values_str += "...";

            // Statistics
            float min_val = data[0], max_val = data[0];
            double sum = 0.0;
            int nan_count = 0, inf_count = 0, zero_count = 0;

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
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    sum += val;
                }
            }

            double mean = (size - nan_count - inf_count) > 0 ? sum / (size - nan_count - inf_count) : 0.0;

            LOG_TRACE("[" << stage << "] " << name << " shape=" << shape_str
                      << " size=" << size << " values=[" << values_str << "]");
            LOG_TRACE("[" << stage << "] " << name << " stats: min=" << min_val << " max=" << max_val
                      << " mean=" << mean << " zeros=" << zero_count << " NaN=" << nan_count << " Inf=" << inf_count);
        }

        static void logMatMulOperation(const std::shared_ptr<TensorBase> &input,
                                       const std::shared_ptr<TensorBase> &weight,
                                       const std::shared_ptr<TensorBase> &output,
                                       const std::string &operation_name)
        {
            LOG_TRACE("=== MatMul Operation: " << operation_name << " ===");
            logTensorStats(input, "input", operation_name);
            logTensorStats(weight, "weight", operation_name);
            logTensorStats(output, "output", operation_name);
        }

        static void logNormalizationOperation(const std::shared_ptr<TensorBase> &input,
                                              const std::shared_ptr<TensorBase> &weight,
                                              const std::shared_ptr<TensorBase> &output,
                                              const std::string &operation_name)
        {
            LOG_INFO("=== Normalization Operation: " << operation_name << " ===");
            logTensorStats(input, "input", operation_name);
            logTensorStats(weight, "norm_weight", operation_name);
            logTensorStats(output, "output", operation_name);
        }
    };

} // namespace llaminar