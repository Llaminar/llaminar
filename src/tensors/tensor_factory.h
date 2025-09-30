#pragma once

#include "tensor_base.h"
#include "simple_tensor.h"
#include "cosma_tensor.h"
#include "sharded_simple_tensor.h"
#include "shard_spec.h"
#include <functional>
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

        // Iterate currently registered sharded tensors (non-owning). Callback receives const ShardSpec&.
        static void for_each_sharded(const std::function<void(const ShardSpec &, const std::vector<int> &)> &fn);

        // Assign role metadata to last created sharded tensor (helper for loader/pipeline)
        static void set_last_shard_role(const std::string &role);

        // Sharded tensor creation (1D partition along given axis index within shape)
        // axis_kind selects semantic axis (Hidden or Heads); axis_dim_index indicates which dimension in shape is split.
        static std::shared_ptr<llaminar::TensorBase> create_sharded(const std::vector<int> &shape,
                                                                    ShardSpec::Axis axis_kind,
                                                                    int axis_dim_index,
                                                                    int world,
                                                                    int rank);

        // Head-aligned sharded creation: ensure partition boundaries align to whole heads (head_dim elements).
        // global_dim = n_heads * head_dim, axis_dim_index indicates which dimension encodes that product.
        // Distributes heads with near-even strategy (base + remainder) so some ranks may have +1 head.
        static std::shared_ptr<llaminar::TensorBase> create_heads_sharded(const std::vector<int> &shape,
                                                                          int axis_dim_index,
                                                                          int n_heads,
                                                                          int head_dim,
                                                                          int world,
                                                                          int rank);

        // Tensor type conversion
        static std::shared_ptr<llaminar::TensorBase> convert_to_cosma(const std::shared_ptr<llaminar::TensorBase> &tensor,
                                                                      const std::string &label = "",
                                                                      int mpi_rank = -1);
        static std::shared_ptr<llaminar::TensorBase> convert_to_simple(const std::shared_ptr<llaminar::TensorBase> &tensor);

        // Type-specific accessors
        static std::shared_ptr<SimpleTensor> to_simple_tensor(std::shared_ptr<llaminar::TensorBase> tensor);
        static std::shared_ptr<COSMATensor> to_cosma_tensor(std::shared_ptr<llaminar::TensorBase> tensor);

    private:
        TensorFactory() = default; // Static class, no instantiation
    };

} // namespace llaminar