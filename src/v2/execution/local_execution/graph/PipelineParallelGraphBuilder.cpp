/**
 * @file PipelineParallelGraphBuilder.cpp
 * @brief Implementation of IPipelineParallelGraphBuilder for pipeline parallelism
 *
 * Part of Phase 3: Pipeline Parallelism Integration
 *
 * Inserts Send/Recv stages at pipeline boundaries based on the execution plan.
 * Uses unique MPI tags per PP stage to ensure correct message matching.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "IPipelineParallelGraphBuilder.h"
#include "DeviceGraphExecutor.h" // For full ComputeGraph definition
#include "../../compute_stages/ComputeStageFactory.h"
#include "../../compute_stages/stages/SendActivationsStage.h"
#include "../../compute_stages/stages/ReceiveActivationsStage.h"
#include "../../../tensors/TensorClasses.h" // For TensorBase -> ITensor
#include "../../../utils/Logger.h"
#include "../../../utils/MPITags.h"
#include <sstream>

namespace llaminar2
{

    namespace
    {
        /**
         * @brief Generate unique MPI tag for PP stage communication
         *
         * Tag encoding: base_tag + stage_id
         * Base tag of 1000 avoids conflicts with other MPI tags in the system.
         *
         * @param stage_id PP stage identifier
         * @return Unique MPI tag for this stage's forward communication
         */
        int makePPTag(int stage_id)
        {
            // PP tags start at 1000 to avoid conflicts with TP allreduce tags
            constexpr int PP_TAG_BASE = 1000;
            return PP_TAG_BASE + stage_id;
        }

        /**
         * @brief Generate stage name for receive stage
         */
        std::string makeRecvStageName(int from_stage_id)
        {
            std::ostringstream oss;
            oss << "recv_from_stage_" << from_stage_id;
            return oss.str();
        }

        /**
         * @brief Generate stage name for send stage
         */
        std::string makeSendStageName(int to_stage_id)
        {
            std::ostringstream oss;
            oss << "send_to_stage_" << to_stage_id;
            return oss.str();
        }
    } // anonymous namespace

    // =========================================================================
    // PipelineParallelGraphBuilder Implementation
    // =========================================================================

    class PipelineParallelGraphBuilder : public IPipelineParallelGraphBuilder
    {
    public:
        explicit PipelineParallelGraphBuilder(MPIContext *mpi_ctx)
            : mpi_ctx_(mpi_ctx), async_mode_(false)
        {
        }

        bool insertReceiveStage(
            ComputeGraph &graph,
            const RankExecutionPlan &plan,
            TensorBase *input_buffer) override
        {
            // Only insert if we have a previous rank to receive from
            if (!plan.prev_rank.has_value())
            {
                LOG_DEBUG("[PPGraphBuilder] Rank " << plan.rank
                                                   << " is first PP stage, no receive needed");
                return false;
            }

            if (!input_buffer)
            {
                LOG_ERROR("[PPGraphBuilder] Cannot insert receive stage: null input buffer");
                return false;
            }

            if (!mpi_ctx_)
            {
                LOG_ERROR("[PPGraphBuilder] Cannot insert receive stage: null MPI context");
                return false;
            }

            int src_rank = *plan.prev_rank;
            int from_stage_id = plan.pp_stage_id - 1;
            std::string stage_name = makeRecvStageName(from_stage_id);

            // Create receive stage parameters
            ReceiveActivationsStage::Params params{
                .device_id = plan.primary_device.toLocalDeviceId(),
                .mpi_ctx = mpi_ctx_,
                .buffer = input_buffer,
                .src_rank = src_rank,
                .tag = makePPTag(plan.pp_stage_id), // Tag for messages TO this stage
                .async = async_mode_,
                .stage_name = stage_name,
            };

            auto stage = ComputeStageFactory::createReceiveActivations(params);

            // Get existing root nodes before adding the receive stage
            auto existing_roots = graph.getRootNodes();

            // Add the receive stage as a new root node
            graph.addNode(stage_name, std::move(stage), DeviceId::cpu());

            // Make all existing root nodes depend on the receive stage
            // This ensures receive completes before any computation starts
            for (const auto &root : existing_roots)
            {
                graph.addDependency(root, stage_name);
            }

            LOG_INFO("[PPGraphBuilder] Inserted receive stage '" << stage_name
                                                                 << "' (from rank " << src_rank
                                                                 << ", tag " << makePPTag(plan.pp_stage_id) << ")");
            return true;
        }

        bool insertSendStage(
            ComputeGraph &graph,
            const RankExecutionPlan &plan,
            TensorBase *output_buffer) override
        {
            // Only insert if we have a next rank to send to
            if (!plan.next_rank.has_value())
            {
                LOG_DEBUG("[PPGraphBuilder] Rank " << plan.rank
                                                   << " is last PP stage, no send needed");
                return false;
            }

            if (!output_buffer)
            {
                LOG_ERROR("[PPGraphBuilder] Cannot insert send stage: null output buffer");
                return false;
            }

            if (!mpi_ctx_)
            {
                LOG_ERROR("[PPGraphBuilder] Cannot insert send stage: null MPI context");
                return false;
            }

            int dest_rank = *plan.next_rank;
            int to_stage_id = plan.pp_stage_id + 1;
            std::string stage_name = makeSendStageName(to_stage_id);

            // Create send stage parameters
            SendActivationsStage::Params params{
                .device_id = plan.primary_device.toLocalDeviceId(),
                .mpi_ctx = mpi_ctx_,
                .buffer = output_buffer,
                .dest_rank = dest_rank,
                .tag = makePPTag(to_stage_id), // Tag for messages TO the next stage
                .async = async_mode_,
                .stage_name = stage_name,
            };

            auto stage = ComputeStageFactory::createSendActivations(params);

            // Get existing leaf nodes before adding the send stage
            auto existing_leaves = graph.getLeafNodes();

            // Add the send stage
            graph.addNode(stage_name, std::move(stage), DeviceId::cpu());

            // Make the send stage depend on all existing leaf nodes
            // This ensures all computation completes before sending
            for (const auto &leaf : existing_leaves)
            {
                graph.addDependency(stage_name, leaf);
            }

            LOG_INFO("[PPGraphBuilder] Inserted send stage '" << stage_name
                                                              << "' (to rank " << dest_rank
                                                              << ", tag " << makePPTag(to_stage_id) << ")");
            return true;
        }

        int insertPPStages(
            ComputeGraph &graph,
            const RankExecutionPlan &plan,
            TensorBase *input_buffer,
            TensorBase *output_buffer) override
        {
            int stages_inserted = 0;

            // Insert receive stage first (it becomes a dependency for computation)
            if (insertReceiveStage(graph, plan, input_buffer))
            {
                stages_inserted++;
            }

            // Insert send stage (it depends on computation completion)
            if (insertSendStage(graph, plan, output_buffer))
            {
                stages_inserted++;
            }

            LOG_DEBUG("[PPGraphBuilder] Inserted " << stages_inserted
                                                   << " PP stages for rank " << plan.rank
                                                   << " (PP stage " << plan.pp_stage_id << ")");

            return stages_inserted;
        }

        void setAsyncMode(bool async) override
        {
            async_mode_ = async;
        }

        bool asyncMode() const override
        {
            return async_mode_;
        }

    private:
        MPIContext *mpi_ctx_;
        bool async_mode_;
    };

    // =========================================================================
    // Factory Implementation
    // =========================================================================

    std::unique_ptr<IPipelineParallelGraphBuilder> createPipelineParallelGraphBuilder(
        MPIContext *mpi_ctx)
    {
        return std::make_unique<PipelineParallelGraphBuilder>(mpi_ctx);
    }

} // namespace llaminar2
