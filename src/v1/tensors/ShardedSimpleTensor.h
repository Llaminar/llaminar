/**
 * @file ShardedSimpleTensor.h
 * @brief Simple in-memory tensor carrying a ShardSpec.
 */
#pragma once

#include "SimpleTensor.h"
#include "ShardSpec.h"
#include <memory>

namespace llaminar
{

    class ShardedSimpleTensor : public SimpleTensor
    {
    public:
        explicit ShardedSimpleTensor(const std::vector<int> &shape, const ShardSpec &spec)
            : SimpleTensor(shape), spec_(spec) {}
        ShardedSimpleTensor(const std::vector<int> &shape, const std::vector<float> &data, const ShardSpec &spec)
            : SimpleTensor(shape, data), spec_(spec) {}

        const ShardSpec &shard_spec() const { return spec_; }
        ShardSpec &shard_spec() { return spec_; }

        std::string type_name() const override
        {
            return std::string("ShardedSimpleTensor(") + spec_.to_string() + ")";
        }

        bool is_distributed() const override { return spec_.is_sharded(); }

    private:
        ShardSpec spec_;
    };

} // namespace llaminar
