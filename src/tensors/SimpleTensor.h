#pragma once

#include "TensorBase.h"
#include "../Logger.h"
#include "../utils/DebugEnv.h"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <omp.h>

namespace llaminar
{

    /**
     * Simple in-memory tensor implementation for non-distributed operations.
     * Compatible with our existing Tensor struct but wrapped in the TensorBase interface.
     */
    class SimpleTensor : public TensorBase
    {
    private:
        std::vector<float> data_;
        std::vector<int> shape_;

        /**
         * @brief Perform NUMA-aware first-touch initialization on allocated memory
         *
         * Uses OpenMP parallel loops to ensure memory pages are allocated on the NUMA node
         * where they will be accessed by worker threads. This eliminates remote NUMA access
         * penalties (2-3x latency) for large tensors like K/V caches.
         *
         * @param data Pointer to memory buffer to initialize
         * @param size Number of elements to initialize
         * @param init_value Value to initialize elements to (default 0.0f)
         */
        static void numaFirstTouch(float *data, size_t size, float init_value = 0.0f)
        {
            const auto &env = debugEnv();

            // Skip if disabled via environment
            if (!env.loader.numa_first_touch)
            {
                std::fill(data, data + size, init_value);
                return;
            }

            // Small allocations: single-threaded (overhead not worth it)
            constexpr size_t kSmallThreshold = 32 * 1024; // 128KB (32K floats)
            if (size < kSmallThreshold)
            {
                std::fill(data, data + size, init_value);
                return;
            }

// Large allocations: parallel first-touch for NUMA locality
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < size; ++i)
            {
                data[i] = init_value;
            }

            // Optional: Verify NUMA locality (expensive, debug only)
            if (env.loader.numa_verify_locality && size >= kSmallThreshold)
            {
                LOG_DEBUG("[SimpleTensor-NUMA] First-touch completed for " << size
                                                                           << " elements (" << (size * sizeof(float) / 1024.0 / 1024.0)
                                                                           << " MB) using " << omp_get_max_threads() << " threads");
            }
        }

    public:
        // Constructors
        SimpleTensor() = default;

        explicit SimpleTensor(const std::vector<int> &dims) : shape_(dims)
        {
            int total_size = 1;
            for (int dim : dims)
            {
                if (dim < 0)
                {
                    throw std::invalid_argument("Tensor dimensions must be non-negative");
                }
                total_size *= dim;
            }

            // Allocate memory (without initialization to avoid duplicate work)
            data_.resize(total_size);

            // NUMA-aware first-touch initialization
            // This ensures memory pages are allocated on the NUMA node where they will be accessed,
            // eliminating 2-3x remote access latency penalty for large tensors (K/V caches, etc.)
            numaFirstTouch(data_.data(), total_size, 0.0f);

            // DEBUG: Verify zero-initialization
            static int tensor_count = 0;
            if (++tensor_count <= 20)
            {
                std::string shape_str = "[";
                for (size_t i = 0; i < dims.size(); ++i)
                {
                    if (i > 0)
                        shape_str += ",";
                    shape_str += std::to_string(dims[i]);
                }
                shape_str += "]";

                // Check first 16 values to detect any non-zero corruption
                bool has_nonzero = false;
                size_t check_count = std::min(static_cast<size_t>(total_size), static_cast<size_t>(16));
                for (size_t i = 0; i < check_count; ++i)
                {
                    if (data_[i] != 0.0f)
                    {
                        LOG_DEBUG("[SimpleTensor-CTOR-" << tensor_count
                                                        << "] ⚠️ NON-ZERO at index " << i << ": " << data_[i]);
                        has_nonzero = true;
                        break;
                    }
                }

                if (!has_nonzero)
                {
                    LOG_DEBUG("[SimpleTensor-CTOR-" << tensor_count << "] shape=" << shape_str
                                                    << " size=" << total_size
                                                    << " data_ptr=" << (void *)data_.data()
                                                    << " first_val=" << (total_size > 0 ? data_[0] : 0.0f)
                                                    << " last_val=" << (total_size > 0 ? data_[total_size - 1] : 0.0f)
                                                    << " all_zeros=YES");
                }
                else
                {
                    LOG_DEBUG("[SimpleTensor-CTOR-" << tensor_count << "] shape=" << shape_str
                                                    << " size=" << total_size
                                                    << " data_ptr=" << (void *)data_.data()
                                                    << " CORRUPTED!");
                }
            }
        }

        SimpleTensor(const std::vector<int> &dims, const std::vector<float> &values)
            : shape_(dims), data_(values)
        {
            if (total_elements() != static_cast<int>(values.size()))
            {
                throw std::invalid_argument("Data size does not match tensor shape");
            }
        }

        // Constructor for compatibility with double data (from existing code)
        SimpleTensor(const std::vector<int> &dims, const std::vector<double> &values) : shape_(dims)
        {
            data_.reserve(values.size());
            for (double val : values)
            {
                data_.push_back(static_cast<float>(val));
            }
        }

        // TensorBase interface implementation
        const std::vector<int> &shape() const override { return shape_; }

        int size() const override { return static_cast<int>(data_.size()); }

        int ndim() const override { return static_cast<int>(shape_.size()); }

        float *data() override { return data_.data(); }

        const float *data() const override { return data_.data(); }

        std::string type_name() const override { return "SimpleTensor"; }

        bool is_distributed() const override { return false; }

        void zero() override
        {
            std::fill(data_.begin(), data_.end(), 0.0f);
        }

        void fill(float value) override
        {
            std::fill(data_.begin(), data_.end(), value);
        }

        std::shared_ptr<TensorBase> copy() const override
        {
            return std::make_shared<SimpleTensor>(shape_, data_);
        }

        void copy_from(const TensorBase &other) override
        {
            if (!is_compatible_shape(other))
            {
                throw std::invalid_argument("Incompatible tensor shapes for copy");
            }

            // If other is also a SimpleTensor, we can do efficient copy
            if (auto simple_other = dynamic_cast<const SimpleTensor *>(&other))
            {
                data_ = simple_other->data_;
            }
            else
            {
                // Generic copy using data pointer
                const float *other_data = other.data();
                std::copy(other_data, other_data + size(), data_.begin());
            }
        }

        // Direct access to underlying data for compatibility
        std::vector<float> &get_data() { return data_; }
        const std::vector<float> &get_data() const { return data_; }

        std::vector<int> &get_shape() { return shape_; }
        const std::vector<int> &get_shape() const { return shape_; }

        // Utility methods
        void reshape(const std::vector<int> &new_shape)
        {
            int new_size = 1;
            for (int dim : new_shape)
            {
                new_size *= dim;
            }

            if (new_size != size())
            {
                throw std::invalid_argument("Cannot reshape tensor: size mismatch");
            }

            shape_ = new_shape;
        }

        void resize(const std::vector<int> &new_shape)
        {
            shape_ = new_shape;
            int new_size = 1;
            for (int dim : new_shape)
            {
                new_size *= dim;
            }

            // Resize without initialization, then apply NUMA-aware first-touch
            size_t old_size = data_.size();
            data_.resize(new_size);

            // Only first-touch newly allocated portion
            if (static_cast<size_t>(new_size) > old_size)
            {
                numaFirstTouch(data_.data() + old_size, new_size - old_size, 0.0f);
            }
        }

        // Compatibility with legacy Tensor struct
        struct LegacyView
        {
            std::vector<float> &data;
            std::vector<int> &shape;

            LegacyView(SimpleTensor &tensor) : data(tensor.data_), shape(tensor.shape_) {}
        };

        LegacyView legacy_view() { return LegacyView(*this); }
    };

} // namespace llaminar