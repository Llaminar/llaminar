#pragma once

#include "../graph/ComputeGraph.h"

#include <string>

namespace llaminar2
{
    namespace mtp_sidecar
    {
        inline bool bindStagesToCaptureStream(ComputeGraph &graph,
                                              void *capture_stream,
                                              std::string *error = nullptr)
        {
            if (!capture_stream)
            {
                if (error)
                    *error = "MTP sidecar capture stream is null";
                return false;
            }

            for (const auto &node_name : graph.getExecutionOrder())
            {
                ComputeNode *node = graph.getNode(node_name);
                if (!node)
                {
                    if (error)
                        *error = "MTP sidecar graph execution order references missing node '" + node_name + "'";
                    return false;
                }
                if (!node->stage)
                {
                    if (error)
                        *error = "MTP sidecar graph node '" + node_name + "' has no stage";
                    return false;
                }

                node->stage->setGPUStream(capture_stream);
            }

            return true;
        }

        inline bool allStagesBoundToStream(const ComputeGraph &graph,
                                           void *expected_stream,
                                           std::string *mismatch = nullptr)
        {
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const ComputeNode *node = graph.getNode(node_name);
                if (!node)
                {
                    if (mismatch)
                        *mismatch = "missing node '" + node_name + "'";
                    return false;
                }
                if (!node->stage)
                {
                    if (mismatch)
                        *mismatch = "node '" + node_name + "' has no stage";
                    return false;
                }
                if (node->stage->gpuStream() != expected_stream)
                {
                    if (mismatch)
                        *mismatch = "node '" + node_name + "' is not bound to the expected stream";
                    return false;
                }
            }

            return true;
        }

        inline void *deferredSamplingStream(bool forward_ok,
                                            bool defer_final_sync,
                                            void *capture_stream)
        {
            if (!forward_ok || !defer_final_sync)
                return nullptr;
            return capture_stream;
        }
    }
}
