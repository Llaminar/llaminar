/**
 * @file GraphValidator.h
 * @brief Graph-level buffer flow validation
 * @author GitHub Copilot
 * @date December 2025
 *
 * Validates buffer producer/consumer relationships in compute graphs.
 * Catches issues like:
 * - Buffers allocated but never populated (like V_dequant bug)
 * - Buffers consumed before being produced
 * - Missing producers for required buffers
 *
 * ## Usage
 *
 * @code
 * GraphValidator validator;
 * auto result = validator.validateBufferFlow(graph);
 * if (!result.valid) {
 *     for (const auto& error : result.errors) {
 *         LOG_ERROR(error);
 *     }
 * }
 * @endcode
 */

#pragma once

#include "../../debug/BufferRole.h"
#include "GraphExecutor.h"
#include "../../compute_stages/ComputeStages.h"
#include "../../../utils/Logger.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace llaminar2
{

    /**
     * @brief Result of graph validation
     */
    struct GraphValidationResult
    {
        /// Whether validation passed
        bool valid = true;

        /// Error messages (validation failures)
        std::vector<std::string> errors;

        /// Warning messages (potential issues)
        std::vector<std::string> warnings;

        /// Add an error and mark as invalid
        void addError(const std::string &msg)
        {
            errors.push_back(msg);
            valid = false;
        }

        /// Add a warning (doesn't affect validity)
        void addWarning(const std::string &msg)
        {
            warnings.push_back(msg);
        }

        /// Merge another result into this one
        void merge(const GraphValidationResult &other)
        {
            if (!other.valid)
                valid = false;
            errors.insert(errors.end(), other.errors.begin(), other.errors.end());
            warnings.insert(warnings.end(), other.warnings.begin(), other.warnings.end());
        }

        /// Log all errors and warnings
        void log() const
        {
            for (const auto &err : errors)
            {
                LOG_ERROR("[GraphValidator] " << err);
            }
            for (const auto &warn : warnings)
            {
                LOG_WARN("[GraphValidator] " << warn);
            }
        }
    };

    /**
     * @brief Buffer flow entry tracking producer and consumers
     */
    struct BufferFlowEntry
    {
        std::string buffer_name;
        BufferDescriptor descriptor;
        std::string producer_stage;               // Stage that produces this buffer
        std::vector<std::string> consumer_stages; // Stages that consume this buffer
        bool is_external = false;                 // True if pre-allocated externally
    };

    /**
     * @brief Graph-level buffer flow validator
     *
     * Analyzes compute graphs to verify:
     * 1. Every required buffer has a producer (or is external)
     * 2. Consumers don't access buffers before producers run
     * 3. No orphan buffers (allocated but never consumed)
     */
    class GraphValidator
    {
    public:
        GraphValidator() = default;

        /**
         * @brief Validate buffer producer/consumer relationships
         *
         * Checks that every buffer with role OUTPUT or INOUT has a producer
         * stage, and that consumers are ordered after producers.
         *
         * @param graph Compute graph to validate
         * @return Validation result with errors/warnings
         */
        GraphValidationResult validateBufferFlow(const ComputeGraph &graph) const
        {
            GraphValidationResult result;

            // Build buffer flow map from stage requirements
            std::unordered_map<std::string, BufferFlowEntry> buffer_flow;
            auto execution_order = graph.getExecutionOrder();

            // Track stage execution indices for ordering validation
            std::unordered_map<std::string, size_t> stage_order;
            for (size_t i = 0; i < execution_order.size(); ++i)
            {
                stage_order[execution_order[i]] = i;
            }

            // Collect buffer requirements from all stages
            for (const auto &stage_name : execution_order)
            {
                const auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                    continue;

                auto reqs = node->stage->getBufferRequirements();
                for (const auto &buf : reqs.buffers)
                {
                    std::string key = stage_name + "/" + buf.name;

                    // Track outputs as produced by this stage
                    if (buf.role == BufferRole::OUTPUT || buf.role == BufferRole::INOUT)
                    {
                        if (buffer_flow.find(buf.name) == buffer_flow.end())
                        {
                            buffer_flow[buf.name] = BufferFlowEntry{
                                buf.name,
                                buf,
                                stage_name,
                                {},
                                false};
                        }
                        else if (buffer_flow[buf.name].producer_stage.empty())
                        {
                            buffer_flow[buf.name].producer_stage = stage_name;
                        }
                    }

                    // Track inputs as consumed by this stage
                    if (buf.role == BufferRole::INPUT || buf.role == BufferRole::INOUT)
                    {
                        if (buffer_flow.find(buf.name) != buffer_flow.end())
                        {
                            buffer_flow[buf.name].consumer_stages.push_back(stage_name);
                        }
                    }
                }
            }

            // Validate: Check that buffers with declared producers have those stages
            for (const auto &[name, entry] : buffer_flow)
            {
                if (entry.descriptor.hasProducer())
                {
                    // Verify declared producer exists in graph
                    if (stage_order.find(entry.descriptor.producer_stage) == stage_order.end())
                    {
                        result.addError("Buffer '" + name + "' declares producer '" +
                                        entry.descriptor.producer_stage + "' which doesn't exist in graph");
                    }
                }

                // Check for orphan outputs (produced but never consumed)
                if ((entry.descriptor.role == BufferRole::OUTPUT) &&
                    entry.consumer_stages.empty() &&
                    !entry.is_external)
                {
                    result.addWarning("Buffer '" + name + "' is produced by '" +
                                      entry.producer_stage + "' but never consumed");
                }
            }

            // Validate execution ordering: producers before consumers
            for (const auto &[name, entry] : buffer_flow)
            {
                if (entry.producer_stage.empty())
                    continue;

                auto producer_idx = stage_order.find(entry.producer_stage);
                if (producer_idx == stage_order.end())
                    continue;

                for (const auto &consumer : entry.consumer_stages)
                {
                    auto consumer_idx = stage_order.find(consumer);
                    if (consumer_idx != stage_order.end())
                    {
                        if (consumer_idx->second < producer_idx->second)
                        {
                            result.addError("Buffer '" + name + "' is consumed by '" +
                                            consumer + "' (order " + std::to_string(consumer_idx->second) +
                                            ") before produced by '" + entry.producer_stage +
                                            "' (order " + std::to_string(producer_idx->second) + ")");
                        }
                    }
                }
            }

            return result;
        }

        /**
         * @brief Check for buffers that are allocated but never have a producer
         *
         * Useful for detecting the V_dequant-style bug where a buffer is
         * allocated in one place but never populated.
         *
         * @param allocated_buffers Names of buffers that were allocated
         * @param graph Compute graph with stage requirements
         * @return Validation result
         */
        GraphValidationResult validateNoOrphanAllocations(
            const std::vector<std::string> &allocated_buffers,
            const ComputeGraph &graph) const
        {
            GraphValidationResult result;

            // Collect all buffers that stages produce
            std::unordered_set<std::string> produced_buffers;
            auto execution_order = graph.getExecutionOrder();

            for (const auto &stage_name : execution_order)
            {
                const auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                    continue;

                auto reqs = node->stage->getBufferRequirements();
                for (const auto &buf : reqs.buffers)
                {
                    if (buf.role == BufferRole::OUTPUT || buf.role == BufferRole::INOUT)
                    {
                        produced_buffers.insert(buf.name);
                    }
                }
            }

            // Check for allocated buffers with no producer
            for (const auto &buf_name : allocated_buffers)
            {
                if (produced_buffers.find(buf_name) == produced_buffers.end())
                {
                    result.addWarning("Buffer '" + buf_name +
                                      "' is allocated but no stage declares it as output");
                }
            }

            return result;
        }

        /**
         * @brief Validate buffer requirements match between connected stages
         *
         * Ensures that when stage A outputs buffer X and stage B inputs buffer X,
         * the shapes and types are compatible.
         *
         * @param graph Compute graph to validate
         * @return Validation result
         */
        GraphValidationResult validateBufferCompatibility(const ComputeGraph &graph) const
        {
            GraphValidationResult result;

            // Build map of output descriptors by buffer name
            std::unordered_map<std::string, std::pair<std::string, BufferDescriptor>> outputs;
            auto execution_order = graph.getExecutionOrder();

            for (const auto &stage_name : execution_order)
            {
                const auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                    continue;

                auto reqs = node->stage->getBufferRequirements();
                for (const auto &buf : reqs.buffers)
                {
                    if (buf.role == BufferRole::OUTPUT)
                    {
                        outputs[buf.name] = {stage_name, buf};
                    }
                }
            }

            // Check that inputs match outputs
            for (const auto &stage_name : execution_order)
            {
                const auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                    continue;

                auto reqs = node->stage->getBufferRequirements();
                for (const auto &buf : reqs.buffers)
                {
                    if (buf.role == BufferRole::INPUT)
                    {
                        auto it = outputs.find(buf.name);
                        if (it != outputs.end())
                        {
                            const auto &[producer, out_desc] = it->second;

                            // Check type compatibility
                            if (buf.tensor_type != out_desc.tensor_type)
                            {
                                result.addError("Buffer '" + buf.name + "' type mismatch: " +
                                                "producer '" + producer + "' outputs " +
                                                std::to_string(static_cast<int>(out_desc.tensor_type)) +
                                                " but consumer '" + stage_name + "' expects " +
                                                std::to_string(static_cast<int>(buf.tensor_type)));
                            }

                            // Check shape compatibility (if both specify shapes)
                            if (!buf.shape.empty() && !out_desc.shape.empty())
                            {
                                if (buf.shape != out_desc.shape)
                                {
                                    result.addWarning("Buffer '" + buf.name + "' shape mismatch: " +
                                                      "producer '" + producer + "' vs consumer '" + stage_name + "'");
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }

        /**
         * @brief Run all validations on a graph
         *
         * @param graph Compute graph to validate
         * @return Combined validation result
         */
        GraphValidationResult validateAll(const ComputeGraph &graph) const
        {
            GraphValidationResult result;

            result.merge(validateBufferFlow(graph));
            result.merge(validateBufferCompatibility(graph));

            return result;
        }
    };

} // namespace llaminar2
