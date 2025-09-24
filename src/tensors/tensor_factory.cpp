#include "tensor_factory.h"
#include "tensor_base.h"
#include "simple_tensor.h"
#include "cosma_tensor.h"
#include "../logger.h"
#include <mpi.h>
#include <atomic>
#include <array>
#include <cctype>

namespace
{
    // COSMA currently recognizes only matrix labels 'A', 'B', or 'C'.
    // When callers provide arbitrary labels (e.g. "auto_512"), normalize
    // them to a valid single-character label, cycling through A/B/C to
    // minimize collisions when possible.
    std::string normalize_cosma_label(const std::string &label)
    {
        static const std::array<char, 3> kAllowedLabels = {'A', 'B', 'C'};
        static std::atomic<size_t> next_label{0};

        if (!label.empty())
        {
            char c = static_cast<char>(std::toupper(label.front()));
            for (char allowed : kAllowedLabels)
            {
                if (c == allowed)
                {
                    return std::string(1, allowed);
                }
            }
        }

        size_t idx = next_label.fetch_add(1, std::memory_order_relaxed) % kAllowedLabels.size();
        return std::string(1, kAllowedLabels[idx]);
    }
}

namespace llaminar
{

    // Factory function implementations

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_simple(const std::vector<int> &shape)
    {
        return std::make_shared<SimpleTensor>(shape);
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_simple(const std::vector<int> &shape,
                                                                       const std::vector<float> &data)
    {
        return std::make_shared<SimpleTensor>(shape, data);
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_cosma(const std::vector<int> &shape,
                                                                      const std::string &label,
                                                                      int mpi_rank)
    {
        if (shape.size() != 2)
        {
            LOG_WARN("COSMATensor requires 2D shape, received " + std::to_string(shape.size()) + "D. Falling back to SimpleTensor.");
            return create_simple(shape);
        }

        std::string sanitized_label = normalize_cosma_label(label);

        try
        {
            auto tensor = std::make_shared<COSMATensor>(shape, sanitized_label, mpi_rank);
            if (!tensor->data() || tensor->size() == 0)
            {
                LOG_WARN("COSMATensor allocated with zero local storage for label '" << sanitized_label
                                                                                     << "' on rank " << mpi_rank << ". Falling back to SimpleTensor.");
                return create_simple(shape);
            }
            return tensor;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Failed to create COSMATensor: " + std::string(e.what()) +
                      ". Falling back to SimpleTensor.");
            return create_simple(shape);
        }
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_auto(const std::vector<int> &shape,
                                                                     bool prefer_distributed)
    {
        // Heuristics for automatic tensor type selection

        // Check if MPI is available and we have multiple processes
        int mpi_size = 1;
        int mpi_rank = 0;
        int mpi_initialized;
        MPI_Initialized(&mpi_initialized);

        if (mpi_initialized)
        {
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        }

        // Decision criteria for using COSMA tensor:
        // 1. MPI environment with multiple processes
        // 2. 2D matrices (COSMA's strength)
        // 3. Large enough matrices to benefit from distribution
        // 4. User preference for distributed

        bool should_use_cosma = false;

        if (mpi_initialized && mpi_size > 1)
        {
            long total_elements = 1;
            for (int dim : shape)
            {
                total_elements *= dim;
            }

            // Use COSMA for large matrices or when explicitly preferred
            const long COSMA_THRESHOLD = 256 * 256; // 64K elements threshold

            // Additional constraints for COSMA - minimum dimensions for distribution
            bool has_sufficient_rows = (shape.size() >= 2 && shape[0] >= mpi_size);
            bool has_sufficient_cols = (shape.size() >= 2 && shape[1] >= mpi_size);
            bool dimensions_suitable = has_sufficient_rows || has_sufficient_cols;

            should_use_cosma = (total_elements >= COSMA_THRESHOLD) &&
                               dimensions_suitable &&
                               prefer_distributed;
        }

        if (should_use_cosma)
        {
            LOG_DEBUG("Auto-selecting COSMATensor for shape [" +
                      std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
                      "] with " + std::to_string(mpi_size) + " MPI processes");

            // Generate a label based on tensor characteristics
            std::string auto_label = "auto_" + std::to_string(shape[0]) + "x" + std::to_string(shape[1]);
            return create_cosma(shape, auto_label, mpi_rank);
        }
        else
        {
            LOG_DEBUG("Auto-selecting SimpleTensor for shape of size " + std::to_string(shape.size()));
            return create_simple(shape);
        }
    }

    std::shared_ptr<TensorBase> TensorFactory::convert_to_cosma(const std::shared_ptr<TensorBase> &tensor,
                                                                const std::string &label,
                                                                int mpi_rank)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a COSMA tensor, return as-is
        if (auto cosma_tensor = std::dynamic_pointer_cast<COSMATensor>(tensor))
        {
            return tensor;
        }

        // Create new COSMA tensor and copy data
        try
        {
            auto cosma_tensor = create_cosma(tensor->shape(), label, mpi_rank);
            cosma_tensor->copy_from(*tensor);

            LOG_DEBUG("Converted " + tensor->type_name() + " to COSMATensor");
            return cosma_tensor;
        }
        catch (const std::exception &e)
        {
            LOG_WARN("Failed to convert to COSMATensor: " + std::string(e.what()) +
                     ". Returning original tensor.");
            return tensor;
        }
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::convert_to_simple(const std::shared_ptr<llaminar::TensorBase> &tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a simple tensor, return as-is
        if (auto simple_tensor = std::dynamic_pointer_cast<SimpleTensor>(tensor))
        {
            return tensor;
        }

        // Create new simple tensor and copy data
        auto simple_tensor = create_simple(tensor->shape());
        simple_tensor->copy_from(*tensor);

        LOG_DEBUG("Converted " + tensor->type_name() + " to SimpleTensor");
        return simple_tensor;
    }

    // Legacy compatibility methods

    std::shared_ptr<SimpleTensor> TensorFactory::to_simple_tensor(std::shared_ptr<llaminar::TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a SimpleTensor, cast and return
        if (auto simple = std::dynamic_pointer_cast<SimpleTensor>(tensor))
        {
            return simple;
        }

        // Convert to SimpleTensor
        auto simple_tensor = std::make_shared<SimpleTensor>(tensor->shape());
        std::copy(tensor->data(), tensor->data() + tensor->size(), simple_tensor->data());
        return simple_tensor;
    }

    std::shared_ptr<COSMATensor> TensorFactory::to_cosma_tensor(std::shared_ptr<llaminar::TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a COSMATensor, cast and return
        if (auto cosma = std::dynamic_pointer_cast<COSMATensor>(tensor))
        {
            return cosma;
        }

        // Convert to COSMATensor
        int mpi_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        try
        {
            auto cosma_tensor = std::make_shared<COSMATensor>(tensor->shape(), normalize_cosma_label("converted"), mpi_rank);
            std::copy(tensor->data(), tensor->data() + tensor->size(), cosma_tensor->data());
            return cosma_tensor;
        }
        catch (const std::exception &e)
        {
            LOG_WARN("Failed to convert to COSMATensor: " + std::string(e.what()));
            return nullptr;
        }
    }

} // namespace llaminar