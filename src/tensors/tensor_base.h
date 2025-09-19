#pragma once

#include <vector>
#include <memory>
#include <string>

namespace llaminar
{
    // Forward declaration
    struct Tensor;

    /**
     * Abstract base class for all tensor types in the llaminar system.
     * Provides a common interface that can be implemented by both simple
     * in-memory tensors and COSMA-optimized distributed tensors.
     */
    class TensorBase
    {
    public:
        virtual ~TensorBase() = default;

        // Shape and size information
        virtual const std::vector<int> &shape() const = 0;
        virtual int size() const = 0;
        virtual int ndim() const = 0;

        // Data access - returns pointer to the local data portion
        virtual float *data() = 0;
        virtual const float *data() const = 0;

        // Tensor type identification
        virtual std::string type_name() const = 0;
        virtual bool is_distributed() const = 0;

        // Utility methods
        virtual void zero() = 0;
        virtual void fill(float value) = 0;

        // Basic tensor operations
        virtual std::shared_ptr<TensorBase> copy() const = 0;
        virtual void copy_from(const TensorBase &other) = 0;

        // Shape utilities
        int total_elements() const
        {
            int total = 1;
            const auto &dims = shape();
            for (int dim : dims)
            {
                total *= dim;
            }
            return total;
        }

        // Matrix-specific accessors (for 2D tensors)
        int rows() const
        {
            const auto &dims = shape();
            return dims.size() >= 1 ? dims[0] : 0;
        }

        int cols() const
        {
            const auto &dims = shape();
            return dims.size() >= 2 ? dims[1] : 1;
        }

        // Validation
        bool is_matrix() const { return ndim() == 2; }
        bool is_vector() const { return ndim() == 1; }
        bool is_scalar() const { return ndim() == 0; }

        bool is_compatible_shape(const TensorBase &other) const
        {
            return shape() == other.shape();
        }
    };

    // Forward declaration - full definition in tensor_factory.h
    class TensorFactory;

    // Graph Compute Compatibility Bridge
    class GraphTensorBridge
    {
    public:
        // Convert legacy Tensor to optimal TensorBase for kernels
        static std::shared_ptr<TensorBase> optimize_for_kernel(std::shared_ptr<llaminar::Tensor> legacy_tensor,
                                                               const std::string &operation_type = "");

        // Convert TensorBase back to legacy Tensor for graph system
        static std::shared_ptr<llaminar::Tensor> to_graph_tensor(std::shared_ptr<TensorBase> tensor);

        // Auto-upgrade tensor for kernel execution (zero-copy when possible)
        static std::shared_ptr<TensorBase> auto_upgrade(std::shared_ptr<llaminar::Tensor> legacy_tensor,
                                                        const std::string &kernel_type = "");
    };

} // namespace llaminar