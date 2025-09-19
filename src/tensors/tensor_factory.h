#pragma once

#include "tensor_base.h"
#include "simple_tensor.h"
#include "cosma_tensor.h"
#include "../tensor.h"
#include <memory>
#include <vector>
#include <string>

// Forward declarations are now replaced with full includes above

namespace llaminar
{
    /**
     * TensorFactory provides utilities for creating and converting between
     * different tensor types (SimpleTensor, COSMATensor) and the legacy Tensor struct.
     */
    class TensorFactory
    {
    public:
        // Simple tensor creation
        static std::shared_ptr<llaminar::TensorBase> create_simple(const std::vector<int> &shape);
        static std::shared_ptr<llaminar::TensorBase> create_simple(const std::vector<int> &shape,
                                                                   const std::vector<float> &data);

        // COSMA tensor creation
        static std::shared_ptr<llaminar::TensorBase> create_cosma(const std::vector<int> &shape,
                                                                  const std::string &label = "",
                                                                  int mpi_rank = -1);

        // Automatic tensor type selection
        static std::shared_ptr<llaminar::TensorBase> create_auto(const std::vector<int> &shape,
                                                                 bool prefer_distributed = false);
        static std::shared_ptr<llaminar::TensorBase> create_auto(const std::vector<int> &shape,
                                                                 const std::vector<float> &data = {});

        // Tensor type conversion
        static std::shared_ptr<llaminar::TensorBase> convert_to_cosma(const std::shared_ptr<llaminar::TensorBase> &tensor,
                                                                      const std::string &label = "",
                                                                      int mpi_rank = -1);
        static std::shared_ptr<llaminar::TensorBase> convert_to_simple(const std::shared_ptr<llaminar::TensorBase> &tensor);

        // Type-specific accessors
        static std::shared_ptr<SimpleTensor> to_simple_tensor(std::shared_ptr<llaminar::TensorBase> tensor);
        static std::shared_ptr<COSMATensor> to_cosma_tensor(std::shared_ptr<llaminar::TensorBase> tensor);

        // Legacy tensor conversion
        static std::shared_ptr<llaminar::TensorBase> from_tensor(std::shared_ptr<llaminar::Tensor> legacy_tensor);
        static std::shared_ptr<llaminar::Tensor> to_tensor(std::shared_ptr<llaminar::TensorBase> tensor);

        // Utility functions
        static void copy_from_simple_tensor(std::shared_ptr<llaminar::Tensor> source,
                                            std::shared_ptr<llaminar::TensorBase> destination);

    private:
        TensorFactory() = default; // Static class, no instantiation
    };

} // namespace llaminar