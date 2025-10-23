/**
 * @file DistributedTensorView.h
 * @brief Lightweight non-owning view that associates a TensorBase with sharding metadata.
 */
#pragma once

#include "TensorBase.h"
#include "ShardedSimpleTensor.h"
#include "ShardSpec.h"
#include <memory>

namespace llaminar
{

    /**
     * DistributedTensorView provides a uniform interface for kernels to query
     * sharding metadata regardless of whether the underlying tensor is sharded
     * or replicated. It never allocates or owns storage.
     */
    class DistributedTensorView
    {
    public:
        DistributedTensorView() = default;
        explicit DistributedTensorView(std::shared_ptr<TensorBase> t) : tensor_(std::move(t)) {}

        bool valid() const { return (bool)tensor_; }
        TensorBase *get() const { return tensor_.get(); }
        std::shared_ptr<TensorBase> shared() const { return tensor_; }

        const std::vector<int> &shape() const { return tensor_->shape(); }
        float *data() const { return tensor_->data(); }
        size_t size() const { return tensor_->size(); }

        const ShardSpec &shard_spec() const
        {
            if (auto sharded = std::dynamic_pointer_cast<ShardedSimpleTensor>(tensor_))
            {
                return sharded->shard_spec();
            }
            // Fallback static replicated spec (thread-safe)
            static ShardSpec replicated{}; // defaults to replicated/none
            return replicated;
        }

        bool is_sharded() const { return shard_spec().is_sharded(); }
        ShardSpec::Axis axis() const { return shard_spec().axis; }
        int world() const { return shard_spec().world; }
        int rank() const { return shard_spec().rank; }
        size_t global_dim() const { return shard_spec().global_dim; }
        size_t local_dim() const { return shard_spec().local_dim; }
        size_t local_offset() const { return shard_spec().local_offset; }

    private:
        std::shared_ptr<TensorBase> tensor_{};
    };

} // namespace llaminar
