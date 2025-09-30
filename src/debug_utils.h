#pragma once

#include "logger.h"
#include "tensors/tensor_base.h"
#include <memory>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <unordered_set>

#include <cctype>

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
                else if (!Logger::getInstance().shouldLog(LogLevel::TRACE))
                {
                    return;
                }
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

    inline std::vector<int> parseRowSpecification(const std::string &spec, size_t max_rows)
    {
        constexpr size_t kMaxEntries = 4096;
        std::vector<int> rows;
        if (spec.empty() || max_rows == 0)
        {
            return rows;
        }
        rows.reserve(8);
        std::unordered_set<int> seen;

        auto trim = [](const std::string &token) -> std::string
        {
            size_t start = 0;
            while (start < token.size() && std::isspace(static_cast<unsigned char>(token[start])))
            {
                ++start;
            }
            size_t end = token.size();
            while (end > start && std::isspace(static_cast<unsigned char>(token[end - 1])))
            {
                --end;
            }
            return token.substr(start, end - start);
        };

        size_t pos = 0;
        while (pos < spec.size() && rows.size() < kMaxEntries)
        {
            size_t next = pos;
            while (next < spec.size() && spec[next] != ',' && spec[next] != ';')
            {
                ++next;
            }
            std::string token = trim(spec.substr(pos, next - pos));
            if (!token.empty())
            {
                size_t range_delim = token.find_first_of(":-");
                auto push_value = [&](int value)
                {
                    if (value >= 0 && static_cast<size_t>(value) < max_rows && seen.insert(value).second)
                    {
                        rows.push_back(value);
                    }
                };

                if (range_delim != std::string::npos)
                {
                    std::string start_token = trim(token.substr(0, range_delim));
                    std::string end_token = trim(token.substr(range_delim + 1));
                    if (!start_token.empty() && !end_token.empty())
                    {
                        int start_value = std::atoi(start_token.c_str());
                        int end_value = std::atoi(end_token.c_str());
                        if (start_value <= end_value)
                        {
                            for (int v = start_value; v <= end_value && rows.size() < kMaxEntries; ++v)
                            {
                                push_value(v);
                            }
                        }
                        else
                        {
                            for (int v = start_value; v >= end_value && rows.size() < kMaxEntries; --v)
                            {
                                push_value(v);
                            }
                        }
                    }
                }
                else
                {
                    int value = std::atoi(token.c_str());
                    push_value(value);
                }
            }

            if (next == spec.size())
            {
                break;
            }
            pos = next + 1;
        }

        std::sort(rows.begin(), rows.end());
        return rows;
    }

    inline void logTensorRowPreview(const std::shared_ptr<TensorBase> &tensor,
                                    const std::string &name,
                                    const std::vector<int> &rows,
                                    size_t preview_cols,
                                    const std::string &stage)
    {
        if (!tensor || !tensor->data() || rows.empty())
        {
            return;
        }

        const auto &shape = tensor->shape();
        if (shape.size() < 2)
        {
            LOG_WARN("[" << stage << "] Cannot preview tensor '" << name << "' with rank " << shape.size());
            return;
        }

        size_t total_rows = static_cast<size_t>(shape[0]);
        size_t total_cols = static_cast<size_t>(shape[1]);
        if (total_rows == 0 || total_cols == 0)
        {
            LOG_WARN("[" << stage << "] Tensor '" << name << "' has empty dimensions");
            return;
        }

        const float *data = tensor->data();
        preview_cols = std::min(preview_cols, total_cols);

        for (int row : rows)
        {
            if (row < 0 || static_cast<size_t>(row) >= total_rows)
            {
                continue;
            }

            const float *row_ptr = data + static_cast<size_t>(row) * total_cols;
            float row_min = row_ptr[0];
            float row_max = row_ptr[0];
            double sum = 0.0;
            double sum_sq = 0.0;

            for (size_t c = 0; c < total_cols; ++c)
            {
                float value = row_ptr[c];
                row_min = std::min(row_min, value);
                row_max = std::max(row_max, value);
                sum += value;
                sum_sq += static_cast<double>(value) * static_cast<double>(value);
            }

            double mean = sum / static_cast<double>(total_cols);
            double rms = std::sqrt(sum_sq / static_cast<double>(total_cols));

            std::ostringstream preview_stream;
            preview_stream << "[";
            for (size_t c = 0; c < preview_cols; ++c)
            {
                if (c)
                {
                    preview_stream << ", ";
                }
                preview_stream << row_ptr[c];
            }
            if (total_cols > preview_cols)
            {
                preview_stream << ", ...";
            }
            preview_stream << "]";

            LOG_INFO("[" << stage << "] " << name
                         << " row=" << row
                         << " preview=" << preview_stream.str()
                         << " stats(min=" << row_min
                         << ", max=" << row_max
                         << ", mean=" << mean
                         << ", rms=" << rms << ")");
        }
    }

} // namespace llaminar