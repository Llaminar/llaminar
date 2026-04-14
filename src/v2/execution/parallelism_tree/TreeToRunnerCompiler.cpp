/**
 * @file TreeToRunnerCompiler.cpp
 * @brief Implementation of tree-to-runner compilation
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "TreeToRunnerCompiler.h"
#include "TransferSpec.h"
#include "../../utils/Logger.h"
#include <algorithm>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Main API
    // =========================================================================

    std::unique_ptr<IInferenceRunner> TreeToRunnerCompiler::compile(
        const ParallelismTree &tree,
        const CompileContext &ctx)
    {
        // Validate tree has layers assigned
        if (tree.root.first_layer < 0)
        {
            throw std::invalid_argument("TreeToRunnerCompiler: tree must have layers assigned (call assignLayers first)");
        }

        // Validate tree
        auto errors = tree.validate();
        if (!errors.empty())
        {
            std::string msg = "TreeToRunnerCompiler: invalid tree:\n";
            for (const auto &e : errors)
            {
                msg += "  - " + e + "\n";
            }
            throw std::invalid_argument(msg);
        }

        LOG_DEBUG("TreeToRunnerCompiler: compiling tree for rank " << ctx.my_rank
                                                                   << " (world_size=" << ctx.world_size << ")");

        // Compile from root
        return compileNode(tree.root, ctx);
    }

    bool TreeToRunnerCompiler::shouldCompile(const ParallelismNode &node, int my_rank)
    {
        // For DEVICE leaf: compile if we own it
        if (node.type == ParallelismNodeType::DEVICE)
        {
            return node.owning_rank == my_rank;
        }

        // For interior nodes: compile if any descendant is owned by us
        auto ranks = node.leafRanks();
        return ranks.find(my_rank) != ranks.end();
    }

    bool TreeToRunnerCompiler::isCrossRankPP(const ParallelismNode &node)
    {
        if (node.type != ParallelismNodeType::PIPELINE_PARALLEL)
        {
            return false;
        }

        // Check if children span multiple ranks
        std::set<int> all_ranks;
        for (const auto &child : node.children)
        {
            auto child_ranks = child.leafRanks();
            all_ranks.insert(child_ranks.begin(), child_ranks.end());
        }

        return all_ranks.size() > 1;
    }

    std::set<int> TreeToRunnerCompiler::getSubtreeRanks(const ParallelismNode &node)
    {
        return node.leafRanks();
    }

    // =========================================================================
    // Private Implementation
    // =========================================================================

    std::unique_ptr<IInferenceRunner> TreeToRunnerCompiler::compileNode(
        const ParallelismNode &node,
        const CompileContext &ctx)
    {
        // Skip nodes not owned by this rank
        if (!shouldCompile(node, ctx.my_rank))
        {
            LOG_DEBUG("TreeToRunnerCompiler: skipping node '" << node.name
                                                              << "' (not owned by rank " << ctx.my_rank << ")");
            return nullptr;
        }

        switch (node.type)
        {
        case ParallelismNodeType::DEVICE:
            return compileDevice(node, ctx);

        case ParallelismNodeType::TENSOR_PARALLEL:
            return compileTP(node, ctx);

        case ParallelismNodeType::PIPELINE_PARALLEL:
            if (isCrossRankPP(node))
            {
                return compileCrossRankPP(node, ctx);
            }
            else
            {
                return compileLocalPP(node, ctx);
            }

        default:
            throw std::logic_error("TreeToRunnerCompiler: unknown node type");
        }
    }

    std::unique_ptr<IInferenceRunner> TreeToRunnerCompiler::compileDevice(
        const ParallelismNode &node,
        const CompileContext &ctx)
    {
        LOG_DEBUG("TreeToRunnerCompiler: compiling DEVICE node '" << node.name
                                                                  << "' (device=" << node.device.toString()
                                                                  << ", layers=" << node.first_layer << "-" << node.last_layer << ")");

        // Use factory if provided (for testing)
        if (ctx.device_runner_factory)
        {
            return ctx.device_runner_factory(node, ctx.model_ctx);
        }

        // Default: Would create DeviceGraphOrchestrator
        // For now, return nullptr to indicate need for real implementation
        LOG_WARN("TreeToRunnerCompiler: no device_runner_factory provided, returning nullptr");
        return nullptr;
    }

    std::unique_ptr<IInferenceRunner> TreeToRunnerCompiler::compileTP(
        const ParallelismNode &node,
        const CompileContext &ctx)
    {
        LOG_DEBUG("TreeToRunnerCompiler: compiling TP node '" << node.name
                                                              << "' (" << node.children.size() << " children"
                                                              << ", layers=" << node.first_layer << "-" << node.last_layer << ")");

        // Compile non-DEVICE children only.
        // DEVICE children are metadata (specifying which devices the TP group
        // uses). The TP orchestrator (MDO) creates its own TP-aware device
        // runners internally — compiling DEVICE children here would create
        // standalone runners that load all weights unsharded, polluting the
        // shared WeightManager/KernelFactory/GPU state, and which MDO would
        // then discard in favor of its own properly-sharded runners.
        std::vector<std::unique_ptr<IInferenceRunner>> child_runners;
        for (const auto &child : node.children)
        {
            if (child.type == ParallelismNodeType::DEVICE)
            {
                LOG_DEBUG("TreeToRunnerCompiler: skipping DEVICE child '" << child.name
                                                                          << "' of TP node '" << node.name
                                                                          << "' — TP orchestrator manages device runners internally");
                continue;
            }
            auto runner = compileNode(child, ctx);
            if (runner)
            {
                child_runners.push_back(std::move(runner));
            }
        }

        // Use factory if provided
        if (ctx.tp_runner_factory)
        {
            return ctx.tp_runner_factory(node, std::move(child_runners), ctx.model_ctx);
        }

        // Default: Would create MultiDeviceOrchestrator(TP mode)
        LOG_WARN("TreeToRunnerCompiler: no tp_runner_factory provided, returning nullptr");
        return nullptr;
    }

    std::unique_ptr<IInferenceRunner> TreeToRunnerCompiler::compileLocalPP(
        const ParallelismNode &node,
        const CompileContext &ctx)
    {
        LOG_DEBUG("TreeToRunnerCompiler: compiling local PP node '" << node.name
                                                                    << "' (" << node.children.size() << " children"
                                                                    << ", layers=" << node.first_layer << "-" << node.last_layer << ")");

        // Use factory if provided — do NOT pre-compile children.
        // The factory (e.g. MDO constructor) creates its own runners internally,
        // so pre-compiling would create duplicate runners on the same GPUs,
        // wasting memory and potentially corrupting shared device state.
        if (ctx.local_pp_runner_factory)
        {
            std::vector<std::unique_ptr<IInferenceRunner>> no_children;
            return ctx.local_pp_runner_factory(node, std::move(no_children), ctx.model_ctx);
        }

        // No factory: compile children for default handling
        std::vector<std::unique_ptr<IInferenceRunner>> child_runners;
        for (const auto &child : node.children)
        {
            auto runner = compileNode(child, ctx);
            if (runner)
            {
                child_runners.push_back(std::move(runner));
            }
        }

        // Default: Would create MultiDeviceOrchestrator(PP mode)
        LOG_WARN("TreeToRunnerCompiler: no local_pp_runner_factory provided, returning nullptr");
        return nullptr;
    }

    std::unique_ptr<IInferenceRunner> TreeToRunnerCompiler::compileCrossRankPP(
        const ParallelismNode &node,
        const CompileContext &ctx)
    {
        LOG_DEBUG("TreeToRunnerCompiler: compiling cross-rank PP node '" << node.name
                                                                         << "' (" << node.children.size() << " children)");

        // Build stage infos (one per child)
        auto stages = buildStageInfos(node, ctx);

        // Build transfer infos (between adjacent stages)
        auto transfers = buildTransferInfos(node);

        // Create PipelineRunner
        return std::make_unique<PipelineRunner>(
            ctx.my_rank,
            ctx.world_size,
            std::move(stages),
            std::move(transfers),
            ctx.hidden_dim,
            ctx.vocab_size);
    }

    std::vector<PipelineRunner::StageInfo> TreeToRunnerCompiler::buildStageInfos(
        const ParallelismNode &node,
        const CompileContext &ctx)
    {
        std::vector<PipelineRunner::StageInfo> stages;
        stages.reserve(node.children.size());

        for (size_t i = 0; i < node.children.size(); ++i)
        {
            const auto &child = node.children[i];

            PipelineRunner::StageInfo stage;
            stage.stage_index = static_cast<int>(i);
            stage.first_layer = child.first_layer;
            stage.last_layer = child.last_layer;
            stage.has_embedding = child.has_embedding;
            stage.has_lm_head = child.has_lm_head;

            // Determine owning rank
            auto child_ranks = child.leafRanks();
            if (!child_ranks.empty())
            {
                // Use the minimum rank as the canonical owner
                stage.owning_rank = *child_ranks.begin();
            }
            else
            {
                stage.owning_rank = -1;
            }

            // Compile the runner if this rank owns this stage
            if (stage.owning_rank == ctx.my_rank)
            {
                stage.runner = compileNode(child, ctx);
            }

            stages.push_back(std::move(stage));
        }

        return stages;
    }

    std::vector<PipelineRunner::TransferInfo> TreeToRunnerCompiler::buildTransferInfos(
        const ParallelismNode &node)
    {
        std::vector<PipelineRunner::TransferInfo> transfers;

        if (node.children.size() < 2)
        {
            return transfers;
        }

        transfers.reserve(node.children.size() - 1);

        // Create transfer between each adjacent pair of children
        for (size_t i = 0; i + 1 < node.children.size(); ++i)
        {
            const auto &from = node.children[i];
            const auto &to = node.children[i + 1];

            // Use TransferSpec to derive mechanism
            auto spec = TransferSpec::derive(from, to, static_cast<int>(i));

            PipelineRunner::TransferInfo transfer;
            transfer.from_stage = static_cast<int>(i);
            transfer.to_stage = static_cast<int>(i + 1);
            transfer.mpi_tag = spec.mpi_tag;
            transfer.mechanism = spec.mechanism;
            transfer.sender_rank = spec.sender_rank;
            transfer.receiver_rank = spec.receiver_rank;

            transfers.push_back(transfer);
        }

        return transfers;
    }

} // namespace llaminar2
