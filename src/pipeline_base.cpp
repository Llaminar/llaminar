#include "pipeline_base.h"
#include "debug_utils.h"
#include "performance_timer.h"
#include "utils/perf_counters.h"
#include <chrono>
#include <algorithm>
#include <iomanip>

namespace llaminar
{

    PipelineBase::PipelineBase() : MPIKernelBase()
    {
        initializePipeline();
        LOG_DEBUG("PipelineBase initialized on rank " << getRank() << "/" << getSize());
    }

    PipelineBase::PipelineBase(const MPIContext &ctx) : MPIKernelBase(ctx)
    {
        initializePipeline();
        LOG_DEBUG("PipelineBase initialized from MPIContext: " << mpi_ctx_.toString());
    }

    bool PipelineBase::registerKernel(const std::string &name, std::unique_ptr<MPIKernelBase> kernel)
    {
        if (hasKernel(name))
        {
            LOG_ERROR("PipelineBase: Kernel '" << name << "' already registered");
            return false;
        }

        if (!kernel)
        {
            LOG_ERROR("PipelineBase: Cannot register null kernel '" << name << "'");
            return false;
        }

        kernels_[name] = std::move(kernel);
        kernel_execution_times_[name] = 0.0;
        kernel_execution_counts_[name] = 0;

        LOG_DEBUG("PipelineBase: Registered kernel '" << name << "'");
        return true;
    }

    MPIKernelBase *PipelineBase::getKernel(const std::string &name)
    {
        auto it = kernels_.find(name);
        if (it == kernels_.end())
        {
            LOG_ERROR("PipelineBase: Kernel '" << name << "' not found");
            return nullptr;
        }
        return it->second.get();
    }

    bool PipelineBase::hasKernel(const std::string &name) const
    {
        return kernels_.find(name) != kernels_.end();
    }

    std::vector<std::string> PipelineBase::getKernelNames() const
    {
        std::vector<std::string> names;
        names.reserve(kernels_.size());

        for (const auto &pair : kernels_)
        {
            names.push_back(pair.first);
        }

        std::sort(names.begin(), names.end());
        return names;
    }

    std::shared_ptr<TensorBase> PipelineBase::createBroadcastTensor(const std::vector<int> &shape)
    {
        // Create tensor suitable for MPI broadcasting
        // Use simple tensor for reliable cross-rank communication
        return TensorFactory::create_simple(shape);
    }

    std::shared_ptr<TensorBase> PipelineBase::createLocalTensor(const std::vector<int> &shape)
    {
        // Create tensor for local computation on current rank
        return TensorFactory::create_simple(shape);
    }

    std::shared_ptr<TensorBase> PipelineBase::createDistributedTensor(const std::vector<int> &shape,
                                                                      const std::string &operation_name)
    {
        // Create tensor optimized for distributed computation
        if (!operation_name.empty())
        {
            return TensorFactory::create_cosma(shape, operation_name, getRank());
        }
        else
        {
            return TensorFactory::create_cosma(shape, "distributed_op", getRank());
        }
    }

    std::shared_ptr<TensorBase> PipelineBase::createAutoTensor(const std::vector<int> &shape)
    {
        // Create tensor with automatic backend selection
        return TensorFactory::create_auto(shape, true);
    }

    void PipelineBase::syncAllRanks(const std::string &operation_name, double timeout_seconds)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Use the base class sync with timeout
        PerfBarrier(MPI_COMM_WORLD);

        auto end_time = std::chrono::high_resolution_clock::now();
        double sync_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        if (sync_time > 100.0)
        { // Log if sync takes more than 100ms
            LOG_DEBUG("PipelineBase: Sync for '" << operation_name << "' took "
                                                 << std::fixed << std::setprecision(2) << sync_time << "ms");
        }
    }

    bool PipelineBase::broadcastTensor(std::shared_ptr<TensorBase> &tensor, int root_rank)
    {
        if (!tensor)
        {
            LOG_ERROR("PipelineBase: Cannot broadcast null tensor");
            return false;
        }

        try
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Ensure all ranks are ready for broadcast
            syncAllRanks("pre-broadcast");

            // Perform broadcast
            size_t total_elements = tensor->total_elements();
            checkMPIError(PerfBcast(tensor->data(), total_elements, MPI_FLOAT, root_rank, getComm()),
                          "Tensor broadcast");

            // Ensure broadcast completed on all ranks
            syncAllRanks("post-broadcast");

            auto end_time = std::chrono::high_resolution_clock::now();
            double broadcast_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            LOG_DEBUG("PipelineBase: Broadcast " << total_elements << " elements in "
                                                 << std::fixed << std::setprecision(2) << broadcast_time << "ms");

            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("PipelineBase: Broadcast failed: " << e.what());
            return false;
        }
    }

    bool PipelineBase::gatherTensors(const std::shared_ptr<TensorBase> &local_tensor,
                                     std::vector<std::shared_ptr<TensorBase>> &gathered_tensors,
                                     int root_rank)
    {
        if (!local_tensor)
        {
            LOG_ERROR("PipelineBase: Cannot gather null tensor");
            return false;
        }

        try
        {
            int rank = getRank();
            int size = getSize();
            size_t local_elements = local_tensor->total_elements();

            // Ensure all ranks are ready
            syncAllRanks("pre-gather");

            if (rank == root_rank)
            {
                // Root rank prepares gather buffers
                gathered_tensors.resize(size);
                for (int i = 0; i < size; ++i)
                {
                    gathered_tensors[i] = createLocalTensor(local_tensor->shape());
                }

                // Gather data from all ranks
                std::vector<float *> recv_buffers(size);
                for (int i = 0; i < size; ++i)
                {
                    recv_buffers[i] = gathered_tensors[i]->data();
                }

                checkMPIError(MPI_Gather(local_tensor->data(), local_elements, MPI_FLOAT,
                                         recv_buffers[rank], local_elements, MPI_FLOAT,
                                         root_rank, getComm()),
                              "Tensor gather");
            }
            else
            {
                // Non-root ranks just send their data
                checkMPIError(MPI_Gather(local_tensor->data(), local_elements, MPI_FLOAT,
                                         nullptr, 0, MPI_FLOAT, root_rank, getComm()),
                              "Tensor gather");
            }

            syncAllRanks("post-gather");
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("PipelineBase: Gather failed: " << e.what());
            return false;
        }
    }

    bool PipelineBase::allGatherTensors(const std::shared_ptr<TensorBase> &local_tensor,
                                        std::vector<std::shared_ptr<TensorBase>> &all_tensors)
    {
        if (!local_tensor)
        {
            LOG_ERROR("PipelineBase: Cannot all-gather null tensor");
            return false;
        }

        try
        {
            int size = getSize();
            size_t local_elements = local_tensor->total_elements();

            // Prepare buffers for all ranks
            all_tensors.resize(size);
            for (int i = 0; i < size; ++i)
            {
                all_tensors[i] = createLocalTensor(local_tensor->shape());
            }

            // Ensure all ranks are ready
            syncAllRanks("pre-allgather");

            // Prepare receive buffer pointers
            std::vector<float *> recv_buffers(size);
            for (int i = 0; i < size; ++i)
            {
                recv_buffers[i] = all_tensors[i]->data();
            }

            // Perform all-gather
            checkMPIError(MPI_Allgather(local_tensor->data(), local_elements, MPI_FLOAT,
                                        recv_buffers[0], local_elements, MPI_FLOAT, getComm()),
                          "Tensor all-gather");

            syncAllRanks("post-allgather");
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("PipelineBase: All-gather failed: " << e.what());
            return false;
        }
    }

    bool PipelineBase::reduceTensors(const std::shared_ptr<TensorBase> &local_tensor,
                                     std::shared_ptr<TensorBase> &result_tensor,
                                     MPI_Op op, int root_rank)
    {
        if (!local_tensor)
        {
            LOG_ERROR("PipelineBase: Cannot reduce null tensor");
            return false;
        }

        try
        {
            int rank = getRank();
            size_t local_elements = local_tensor->total_elements();

            // Ensure all ranks are ready
            syncAllRanks("pre-reduce");

            if (rank == root_rank)
            {
                // Root rank prepares result buffer
                if (!result_tensor)
                {
                    result_tensor = createLocalTensor(local_tensor->shape());
                }

                // Perform reduction
                checkMPIError(MPI_Reduce(local_tensor->data(), result_tensor->data(),
                                         local_elements, MPI_FLOAT, op, root_rank, getComm()),
                              "Tensor reduce");
            }
            else
            {
                // Non-root ranks just send their data
                checkMPIError(MPI_Reduce(local_tensor->data(), nullptr,
                                         local_elements, MPI_FLOAT, op, root_rank, getComm()),
                              "Tensor reduce");
            }

            syncAllRanks("post-reduce");
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("PipelineBase: Reduce failed: " << e.what());
            return false;
        }
    }

    bool PipelineBase::allReduceTensors(std::shared_ptr<TensorBase> &local_tensor, MPI_Op op)
    {
        if (!local_tensor)
        {
            LOG_ERROR("PipelineBase: Cannot all-reduce null tensor");
            return false;
        }

        try
        {
            size_t local_elements = local_tensor->total_elements();

            // Create temporary buffer for result
            auto result_tensor = createLocalTensor(local_tensor->shape());

            // Ensure all ranks are ready
            syncAllRanks("pre-allreduce");

            // Perform all-reduce
            checkMPIError(PerfAllreduce(local_tensor->data(), result_tensor->data(),
                                        (int)local_elements, MPI_FLOAT, op, getComm()),
                          "Tensor all-reduce");

            // Copy result back to input tensor
            std::copy(result_tensor->data(), result_tensor->data() + local_elements, local_tensor->data());

            syncAllRanks("post-allreduce");
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("PipelineBase: All-reduce failed: " << e.what());
            return false;
        }
    }

    bool PipelineBase::executeKernel(const std::string &kernel_name,
                                     const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                     std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        auto kernel = getKernel(kernel_name);
        if (!kernel)
        {
            LOG_ERROR("PipelineBase: Kernel '" << kernel_name << "' not found for execution");
            return false;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        bool success = kernel->execute(inputs, outputs);

        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // Update performance tracking
        kernel_execution_times_[kernel_name] += execution_time;
        kernel_execution_counts_[kernel_name]++;

        if (success)
        {
            LOG_DEBUG("PipelineBase: Kernel '" << kernel_name << "' executed in "
                                               << std::fixed << std::setprecision(2) << execution_time << "ms");
        }
        else
        {
            LOG_ERROR("PipelineBase: Kernel '" << kernel_name << "' execution failed");
        }

        return success;
    }

    bool PipelineBase::executeKernelChain(const std::vector<std::string> &kernel_chain,
                                          const std::vector<std::shared_ptr<TensorBase>> &initial_inputs,
                                          std::vector<std::shared_ptr<TensorBase>> &final_outputs,
                                          std::vector<std::vector<std::shared_ptr<TensorBase>>> *intermediate_tensors)
    {
        if (kernel_chain.empty())
        {
            LOG_ERROR("PipelineBase: Empty kernel chain provided");
            return false;
        }

        auto current_inputs = initial_inputs;

        for (size_t i = 0; i < kernel_chain.size(); ++i)
        {
            const std::string &kernel_name = kernel_chain[i];

            std::vector<std::shared_ptr<TensorBase>> current_outputs;

            if (!executeKernel(kernel_name, current_inputs, current_outputs))
            {
                LOG_ERROR("PipelineBase: Kernel chain failed at '" << kernel_name << "' (step " << i << ")");
                return false;
            }

            // Store intermediate results if requested
            if (intermediate_tensors && i < kernel_chain.size() - 1)
            {
                intermediate_tensors->push_back(current_outputs);
            }

            // Prepare inputs for next kernel (or final outputs)
            if (i == kernel_chain.size() - 1)
            {
                final_outputs = std::move(current_outputs);
            }
            else
            {
                current_inputs = std::move(current_outputs);
            }
        }

        LOG_DEBUG("PipelineBase: Kernel chain completed successfully (" << kernel_chain.size() << " steps)");
        return true;
    }

    bool PipelineBase::validateTensorShapes(const std::vector<std::shared_ptr<TensorBase>> &tensors,
                                            const std::vector<std::vector<size_t>> &expected_shapes,
                                            const std::string &operation_name)
    {
        if (tensors.size() != expected_shapes.size())
        {
            LOG_ERROR("PipelineBase: " << operation_name << " tensor count mismatch - expected: "
                                       << expected_shapes.size() << ", got: " << tensors.size());
            return false;
        }

        for (size_t i = 0; i < tensors.size(); ++i)
        {
            if (!tensors[i])
            {
                LOG_ERROR("PipelineBase: " << operation_name << " tensor " << i << " is null");
                return false;
            }

            const auto &actual_shape = tensors[i]->shape();
            const auto &expected_shape = expected_shapes[i];

            if (actual_shape.size() != expected_shape.size())
            {
                LOG_ERROR("PipelineBase: " << operation_name << " tensor " << i
                                           << " dimension mismatch - expected: " << expected_shape.size()
                                           << "D, got: " << actual_shape.size() << "D");
                return false;
            }

            for (size_t j = 0; j < actual_shape.size(); ++j)
            {
                if (actual_shape[j] != expected_shape[j])
                {
                    LOG_ERROR("PipelineBase: " << operation_name << " tensor " << i
                                               << " shape mismatch at dimension " << j
                                               << " - expected: " << expected_shape[j]
                                               << ", got: " << actual_shape[j]);
                    return false;
                }
            }
        }

        return true;
    }

    bool PipelineBase::validateTensorContents(const std::vector<std::shared_ptr<TensorBase>> &tensors,
                                              const std::string &operation_name)
    {
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            if (!tensors[i])
            {
                LOG_ERROR("PipelineBase: " << operation_name << " tensor " << i << " is null");
                return false;
            }

            const float *data = tensors[i]->data();
            size_t total_elements = tensors[i]->total_elements();

            for (size_t j = 0; j < total_elements; ++j)
            {
                if (std::isnan(data[j]))
                {
                    LOG_ERROR("PipelineBase: " << operation_name << " tensor " << i
                                               << " contains NaN at index " << j);
                    return false;
                }
                if (std::isinf(data[j]))
                {
                    LOG_ERROR("PipelineBase: " << operation_name << " tensor " << i
                                               << " contains inf at index " << j);
                    return false;
                }
            }
        }

        return true;
    }

    void PipelineBase::logExecutionStats(const std::string &operation_name,
                                         double execution_time_ms,
                                         const std::vector<std::vector<size_t>> &tensor_shapes)
    {
        std::string shape_info = "";
        for (size_t i = 0; i < tensor_shapes.size(); ++i)
        {
            if (i > 0)
                shape_info += ", ";
            shape_info += "[";
            for (size_t j = 0; j < tensor_shapes[i].size(); ++j)
            {
                if (j > 0)
                    shape_info += "x";
                shape_info += std::to_string(tensor_shapes[i][j]);
            }
            shape_info += "]";
        }

        LOG_INFO("PipelineBase: " << operation_name << " executed in "
                                  << std::fixed << std::setprecision(2) << execution_time_ms
                                  << "ms, shapes: " << shape_info);
    }

    void PipelineBase::initializePipeline()
    {
        if (!validateMPIEnvironment())
        {
            throw std::runtime_error("PipelineBase: Invalid MPI environment");
        }

        configureTensorAllocation();

        LOG_DEBUG("PipelineBase: Pipeline initialization completed");
    }

    bool PipelineBase::validateMPIEnvironment()
    {
        // Check MPI initialization
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            LOG_ERROR("PipelineBase: MPI not initialized");
            return false;
        }

        // Check valid communicator
        if (getComm() == MPI_COMM_NULL)
        {
            LOG_ERROR("PipelineBase: Invalid MPI communicator");
            return false;
        }

        // Check rank and size
        if (getRank() < 0 || getSize() <= 0)
        {
            LOG_ERROR("PipelineBase: Invalid MPI rank/size: " << getRank() << "/" << getSize());
            return false;
        }

        return true;
    }

    void PipelineBase::configureTensorAllocation()
    {
        // Configure NUMA-aware allocation strategies
        // This could be extended with specific NUMA binding policies
        LOG_DEBUG("PipelineBase: Configured tensor allocation for rank " << getRank());
    }

} // namespace llaminar