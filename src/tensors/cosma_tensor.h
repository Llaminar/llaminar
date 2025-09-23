#pragma once

#include "tensor_base.h"
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>
#include <cosma/context.hpp>
#include <mpi.h>
#include <memory>
#include <stdexcept>

namespace llaminar
{

    /**
     * COSMA-optimized tensor implementation for distributed matrix operations.
     * Provides zero-copy access to COSMA's CosmaMatrix while implementing
     * the common TensorBase interface.
     */
    class COSMATensor : public TensorBase
    {
    private:
        std::unique_ptr<cosma::CosmaMatrix<float>> cosma_matrix_;
        std::unique_ptr<cosma::Strategy> strategy_;
        std::vector<int> shape_;
        std::string label_;
        int mpi_rank_;
        int mpi_size_;

        // Cache for shape and size calculations
        mutable int cached_size_ = -1;

        void validate_matrix_dimensions(int m, int n, int k) const
        {
            if (m <= 0 || n <= 0 || k <= 0)
            {
                throw std::invalid_argument("Matrix dimensions must be positive");
            }
        }

    public:
        // Constructor for matrix multiplication (requires strategy and MPI context)
        COSMATensor(const std::vector<int> &dims,
                    const std::string &label,
                    const cosma::Strategy &strategy,
                    int mpi_rank,
                    MPI_Comm comm = MPI_COMM_WORLD)
            : shape_(dims), label_(label), mpi_rank_(mpi_rank)
        {

            if (dims.size() != 2)
            {
                throw std::invalid_argument("COSMATensor currently supports only 2D matrices");
            }

            MPI_Comm_size(comm, &mpi_size_);

            // Create a copy of the strategy
            strategy_ = std::make_unique<cosma::Strategy>(strategy);

            // Create COSMA matrix with the provided strategy
            cosma_matrix_ = std::make_unique<cosma::CosmaMatrix<float>>(
                label[0], *strategy_, mpi_rank_);
        }

        // Constructor that creates strategy automatically
        COSMATensor(const std::vector<int> &dims,
                    const std::string &label,
                    int mpi_rank,
                    MPI_Comm comm = MPI_COMM_WORLD)
            : shape_(dims), label_(label), mpi_rank_(mpi_rank)
        {

            if (dims.size() != 2)
            {
                throw std::invalid_argument("COSMATensor currently supports only 2D matrices");
            }

            MPI_Comm_size(comm, &mpi_size_);

            int m = dims[0], n = dims[1];

            // Validate that dimensions are suitable for distributed processing
            // COSMA requires matrices large enough to distribute effectively
            if (m <= 0 || n <= 0)
            {
                throw std::invalid_argument("Matrix dimensions must be positive");
            }

            // For small matrices or when dimensions don't distribute well, reject COSMA
            if (m < mpi_size_ || n < mpi_size_)
            {
                throw std::invalid_argument("Matrix too small for distributed processing with " +
                                            std::to_string(mpi_size_) + " processes");
            }

            // For auto strategy with tensor allocation, we assume square-ish operations
            // Use the larger dimension as a conservative estimate for k
            int k = std::max(m, n);
            validate_matrix_dimensions(m, n, k);

            // Create auto strategy with conservative parameters
            try
            {
                strategy_ = std::make_unique<cosma::Strategy>(m, n, k, mpi_size_);
            }
            catch (const std::exception &e)
            {
                throw std::invalid_argument("Failed to create COSMA strategy: " + std::string(e.what()));
            }

            // Create COSMA matrix
            cosma_matrix_ = std::make_unique<cosma::CosmaMatrix<float>>(
                label[0], *strategy_, mpi_rank_);
        }

        // TensorBase interface implementation
        const std::vector<int> &shape() const override
        {
            return shape_;
        }

        int size() const override
        {
            if (cached_size_ < 0)
            {
                cached_size_ = cosma_matrix_ ? static_cast<int>(cosma_matrix_->matrix_size()) : 0;
            }
            return cached_size_;
        }

        int ndim() const override
        {
            return static_cast<int>(shape_.size());
        }

        float *data() override
        {
            return cosma_matrix_ ? cosma_matrix_->matrix_pointer() : nullptr;
        }

        const float *data() const override
        {
            return cosma_matrix_ ? cosma_matrix_->matrix_pointer() : nullptr;
        }

        std::string type_name() const override
        {
            return "COSMATensor";
        }

        bool is_distributed() const override
        {
            return true;
        }

        void zero() override
        {
            if (cosma_matrix_)
            {
                float *ptr = cosma_matrix_->matrix_pointer();
                int sz = size();
                std::fill(ptr, ptr + sz, 0.0f);
            }
        }

        void fill(float value) override
        {
            if (cosma_matrix_)
            {
                float *ptr = cosma_matrix_->matrix_pointer();
                int sz = size();
                std::fill(ptr, ptr + sz, value);
            }
        }

        std::shared_ptr<TensorBase> copy() const override
        {
            // Create new COSMATensor with same configuration
            auto copy_tensor = std::make_shared<COSMATensor>(shape_, label_ + "_copy", mpi_rank_);
            copy_tensor->copy_from(*this);
            return copy_tensor;
        }

        void copy_from(const TensorBase &other) override
        {
            if (!is_compatible_shape(other))
            {
                throw std::invalid_argument("Incompatible tensor shapes for copy");
            }

            const float *other_data = other.data();
            float *this_data = data();
            int copy_size = std::min(size(), other.size());

            if (other_data && this_data)
            {
                std::copy(other_data, other_data + copy_size, this_data);
            }
        }

        // COSMA-specific methods for zero-copy operations
        cosma::CosmaMatrix<float> &cosma_matrix()
        {
            if (!cosma_matrix_)
            {
                throw std::runtime_error("COSMATensor not properly initialized");
            }
            return *cosma_matrix_;
        }

        const cosma::CosmaMatrix<float> &cosma_matrix() const
        {
            if (!cosma_matrix_)
            {
                throw std::runtime_error("COSMATensor not properly initialized");
            }
            return *cosma_matrix_;
        }

        const cosma::Strategy &strategy() const
        {
            if (!strategy_)
            {
                throw std::runtime_error("COSMATensor strategy not available");
            }
            return *strategy_;
        }

        const std::string &label() const { return label_; }
        int mpi_rank() const { return mpi_rank_; }
        int mpi_size() const { return mpi_size_; }

        // Advanced COSMA operations
        void initialize_data(const std::vector<float> &global_data)
        {
            // Initialize COSMA matrix with global data
            // This involves distributing data according to COSMA's layout
            if (!cosma_matrix_)
            {
                throw std::runtime_error("COSMATensor not initialized");
            }

            // Get local portion size
            size_t local_size = cosma_matrix_->matrix_size();
            float *local_ptr = cosma_matrix_->matrix_pointer();

            // Simple initialization - copy what we can from global data
            // In a full implementation, this would handle proper data distribution
            size_t copy_size = std::min(local_size, global_data.size());
            if (copy_size > 0)
            {
                std::copy(global_data.begin(), global_data.begin() + copy_size, local_ptr);
            }

            // Zero out remaining elements
            if (local_size > copy_size)
            {
                std::fill(local_ptr + copy_size, local_ptr + local_size, 0.0f);
            }
        }

        // Get global matrix dimensions (not just local portion)
        std::vector<int> global_shape() const
        {
            if (!cosma_matrix_)
            {
                return shape_;
            }

            // COSMA matrices know their global dimensions
            return {cosma_matrix_->m(), cosma_matrix_->n()};
        }

        // Check if this tensor can be used directly in COSMA operations
        bool is_cosma_compatible() const
        {
            return cosma_matrix_ != nullptr && strategy_ != nullptr;
        }
    };

} // namespace llaminar