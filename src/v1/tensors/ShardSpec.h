/**
 * @file ShardSpec.h
 * @brief Metadata definition for tensor sharding (hidden/head partitions).
 */
#pragma once

#include <cstddef>
#include <string>
#include <sstream>

namespace llaminar
{

    struct ShardSpec
    {
        enum class Type
        {
            Replicated,
            Sharded
        };
        enum class Axis
        {
            None,
            Hidden,
            Heads
        };

        Type type = Type::Replicated;
        Axis axis = Axis::None;
        int world = 1;                // MPI world size
        int rank = 0;                 // MPI rank
        std::size_t global_dim = 0;   // size along sharded axis (0 if replicated)
        std::size_t local_offset = 0; // starting offset (elements) along axis
        std::size_t local_dim = 0;    // local slice size along axis
        std::string role;             // optional: semantic role (e.g., W_Q, W_K, W_V, W_O, W1, W2, Embedding)

        bool is_sharded() const { return type == Type::Sharded; }
        std::string axis_name() const
        {
            switch (axis)
            {
            case Axis::None:
                return "none";
            case Axis::Hidden:
                return "hidden";
            case Axis::Heads:
                return "heads";
            }
            return "?";
        }
        std::string to_string() const
        {
            std::ostringstream oss;
            if (!is_sharded())
            {
                oss << "replicated";
            }
            else
            {
                oss << "shard axis=" << axis_name() << " rank=" << rank << "/" << world
                    << " offset=" << local_offset << " size=" << local_dim << " global=" << global_dim;
                if (!role.empty())
                    oss << " role=" << role;
            }
            return oss.str();
        }
    };

} // namespace llaminar
