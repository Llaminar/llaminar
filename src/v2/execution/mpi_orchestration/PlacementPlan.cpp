/**
 * @file PlacementPlan.cpp
 * @brief Implementation of PlacementPlan utility methods
 * @author David Sanftenberg
 * @date December 2025
 */

#include "PlacementPlan.h"
#include <sstream>

namespace llaminar2
{

    std::string PlacementPlan::toString() const
    {
        std::ostringstream oss;
        oss << "PlacementPlan {\n";
        oss << "  strategy: " << strategy_name << "\n";
        oss << "  model: " << architecture << " (" << n_layers << " layers)\n";
        oss << "  topology: " << world_size << " ranks, " << node_count << " nodes, "
            << ranks_per_node << " ranks/node\n";
        oss << "  gpu: " << (has_gpu ? "yes" : "no");
        if (has_gpu)
        {
            oss << " (" << (total_gpu_memory / (1024 * 1024 * 1024)) << " GB total)";
        }
        oss << "\n";

        oss << "  global:\n";
        oss << "    embedding: device=" << global.embedding_device.toString()
            << " shard=" << (global.shard_embedding ? "yes" : "no") << "\n";
        oss << "    lm_head: device=" << global.lm_head_device.toString()
            << " shard=" << (global.shard_lm_head ? "yes" : "no") << "\n";

        oss << "  layers:\n";

        // Group consecutive layers with same placement for compact output
        if (!layers.empty())
        {
            int start_layer = 0;
            LayerPlacement current = layers[0];

            for (int i = 1; i <= static_cast<int>(layers.size()); ++i)
            {
                bool same = (i < static_cast<int>(layers.size()) &&
                             layers[i].owner_rank == current.owner_rank &&
                             layers[i].device == current.device);

                if (!same)
                {
                    // Print range
                    if (i - 1 == start_layer)
                    {
                        oss << "    layer " << start_layer << ": ";
                    }
                    else
                    {
                        oss << "    layers " << start_layer << "-" << (i - 1) << ": ";
                    }
                    oss << "rank=" << current.owner_rank
                        << " device=" << current.device.toString() << "\n";

                    if (i < static_cast<int>(layers.size()))
                    {
                        start_layer = i;
                        current = layers[i];
                    }
                }
            }
        }

        oss << "}";
        return oss.str();
    }

} // namespace llaminar2
