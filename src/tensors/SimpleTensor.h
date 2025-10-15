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
        
        /**
         * @brief Get batch size (first dimension) - 1 if tensor has no batch dimension
         */
        size_t batch_size() const
        {
            return shape_.empty() ? 1 : static_cast<size_t>(shape_[0]);
        }
        
        /**
         * @brief Get sequence length dimension
         * For 2D tensors: [seq_len, d_model] -> seq_len
         * For 3D tensors: [batch, seq_len, d_model] -> seq_len
         */
        size_t seq_len() const
        {
            if (shape_.size() < 2) return 1;
            return static_cast<size_t>(shape_[1]);
        }
        
        /**
         * @brief Reshape tensor to new shape (returns new tensor)
         * @param new_shape New tensor shape
         * @return Shared pointer to new tensor with reshaped data
         */
        std::shared_ptr<SimpleTensor> reshape_copy(const std::vector<int> &new_shape) const
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

            auto result = std::make_shared<SimpleTensor>();
            result->shape_ = new_shape;
            result->data_ = data_;  // Copy data
            return result;
        }
        
        /**
         * @brief Reshape tensor in-place
         */
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
        
        /**
         * @brief Extract a single sequence from a batched tensor
         * @param batch_idx Index of the batch to extract
         * @return Shared pointer to tensor containing just that batch element
         * 
         * Example: [batch=4, seq_len=8, d_model=896] -> [seq_len=8, d_model=896]
         */
        std::shared_ptr<SimpleTensor> slice_batch(size_t batch_idx) const
        {
            if (shape_.empty())
            {
                throw std::invalid_argument("Cannot slice batch from scalar tensor");
            }
            
            size_t batch_dim = static_cast<size_t>(shape_[0]);
            if (batch_idx >= batch_dim)
            {
                throw std::out_of_range("Batch index out of range");
            }
            
            // Calculate size of one batch element
            size_t batch_elem_size = 1;
            std::vector<int> new_shape;
            for (size_t i = 1; i < shape_.size(); ++i)
            {
                new_shape.push_back(shape_[i]);
                batch_elem_size *= shape_[i];
            }
            
            // Create new tensor with sliced data
            auto result = std::make_shared<SimpleTensor>();
            result->shape_ = new_shape;
            result->data_.resize(batch_elem_size);
            
            // Copy data for this batch index
            const float* src = data_.data() + batch_idx * batch_elem_size;
            std::copy(src, src + batch_elem_size, result->data_.begin());
            
            return result;
        }
        
        /**
         * @brief Stack multiple tensors along a new batch dimension
         * @param sequences Vector of tensors to stack
         * @return Shared pointer to batched tensor
         * 
         * Example: 4× [seq_len=8, d_model=896] -> [batch=4, seq_len=8, d_model=896]
         */
        static std::shared_ptr<SimpleTensor> stack_batch(
            const std::vector<std::shared_ptr<SimpleTensor>>& sequences)
        {
            if (sequences.empty())
            {
                throw std::invalid_argument("Cannot stack empty sequence vector");
            }
            
            // Verify all sequences have same shape
            const auto& ref_shape = sequences[0]->shape_;
            for (size_t i = 1; i < sequences.size(); ++i)
            {
                if (sequences[i]->shape_ != ref_shape)
                {
                    throw std::invalid_argument("All sequences must have same shape for stacking");
                }
            }
            
            // Create new shape with batch dimension
            std::vector<int> new_shape;
            new_shape.push_back(static_cast<int>(sequences.size()));  // batch dimension
            new_shape.insert(new_shape.end(), ref_shape.begin(), ref_shape.end());
            
            // Calculate sizes
            size_t batch_size = sequences.size();
            size_t elem_size = sequences[0]->data_.size();
            size_t total_size = batch_size * elem_size;
            
            // Create result tensor
            auto result = std::make_shared<SimpleTensor>();
            result->shape_ = new_shape;
            result->data_.resize(total_size);
            
            // Copy data from each sequence
            for (size_t b = 0; b < batch_size; ++b)
            {
                float* dst = result->data_.data() + b * elem_size;
                const float* src = sequences[b]->data_.data();
                std::copy(src, src + elem_size, dst);
            }
            
            return result;
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