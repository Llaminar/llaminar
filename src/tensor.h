#pragma once

#include <vector>

namespace llaminar
{

    // Legacy tensor data structure for transformer operations
    // This is the original Tensor struct used throughout the codebase
    struct Tensor
    {
        std::vector<float> data;
        std::vector<int> shape;

        Tensor() = default;
        Tensor(const std::vector<int> &dims) : shape(dims)
        {
            int size = 1;
            for (int dim : dims)
                size *= dim;
            data.resize(size);
        }

        Tensor(const std::vector<int> &dims, const std::vector<float> &values) : shape(dims), data(values) {}

        // Constructor for compatibility with double data
        Tensor(const std::vector<int> &dims, const std::vector<double> &values) : shape(dims)
        {
            data.reserve(values.size());
            for (double val : values)
            {
                data.push_back(static_cast<float>(val));
            }
        }

        int size() const
        {
            int s = 1;
            for (int dim : shape)
                s *= dim;
            return s;
        }

        float *ptr() { return data.data(); }
        const float *ptr() const { return data.data(); }
    };

} // namespace llaminar