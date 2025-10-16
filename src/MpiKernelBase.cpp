#include "MpiKernelBase.h"
#include "Logger.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include "utils/PerfCounters.h"

namespace llaminar
{

    MPIOperatorBase::MPIOperatorBase(MPI_Comm comm, bool init_mpi)
        : comm_(comm), rank_(0), size_(1), mpi_initialized_(false)
    {

        int provided, required = MPI_THREAD_MULTIPLE;
        int mpi_init_flag;

        // Check if MPI is already initialized
        checkMPIError(MPI_Initialized(&mpi_init_flag), "MPI_Initialized");

        if (!mpi_init_flag && init_mpi)
        {
            // Initialize MPI with thread support
            checkMPIError(MPI_Init_thread(nullptr, nullptr, required, &provided), "MPI_Init_thread");
            mpi_initialized_ = true;

            if (provided < required)
            {
                LOG_WARN("MPI thread support level " << provided << " is less than required " << required);
            }

            LOG_DEBUG("MPI initialized by MPIOperatorBase");
        }

        // Get rank and size for the communicator
        checkMPIError(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
        checkMPIError(MPI_Comm_size(comm_, &size_), "MPI_Comm_size");

        // Populate MPIContext
        mpi_ctx_ = MPIContext(rank_, size_, comm_);

        LOG_DEBUG("MPIOperatorBase initialized: rank=" << rank_ << ", size=" << size_);

        // Call virtual initialization method
        initializeMPI();
    }

    MPIOperatorBase::MPIOperatorBase(const MPIContext &ctx, bool init_mpi)
        : mpi_ctx_(ctx), comm_(ctx.comm), rank_(ctx.rank), size_(ctx.size), mpi_initialized_(false)
    {
        int provided, required = MPI_THREAD_MULTIPLE;
        int mpi_init_flag;

        // Check if MPI is already initialized
        checkMPIError(MPI_Initialized(&mpi_init_flag), "MPI_Initialized");

        if (!mpi_init_flag && init_mpi)
        {
            // Initialize MPI with thread support
            checkMPIError(MPI_Init_thread(nullptr, nullptr, required, &provided), "MPI_Init_thread");
            mpi_initialized_ = true;

            if (provided < required)
            {
                LOG_WARN("MPI thread support level " << provided << " is less than required " << required);
            }

            LOG_DEBUG("MPI initialized by MPIOperatorBase with MPIContext");
        }

        LOG_DEBUG("MPIOperatorBase initialized from MPIContext: " << mpi_ctx_.toString());

        // Call virtual initialization method
        initializeMPI();
    }

    MPIOperatorBase::~MPIOperatorBase()
    {
        try
        {
            // Call virtual cleanup method
            finalizeMPI();

            // Finalize MPI if we initialized it
            if (mpi_initialized_)
            {
                // In test environments we sometimes need MPI to persist across multiple
                // test cases even after individual kernels (that performed the initial
                // MPI_Init_thread) are destroyed. A test harness can set the environment
                // variable LLAMINAR_TEST_MPI_NO_FINALIZE=1 (or any non-empty value) to
                // delegate responsibility for MPI_Finalize() to a higher-level fixture.
                // This avoids scenarios where the first test's kernel destructors call
                // MPI_Finalize(), leaving subsequent tests attempting MPI calls after
                // finalization (triggering MPI standard violations).
                const char *no_finalize = std::getenv("LLAMINAR_TEST_MPI_NO_FINALIZE");
                if (no_finalize)
                {
                    // Skip finalization here; a global test environment will finalize
                    // once all tests (and their kernels) have completed.
                    return;
                }
                int mpi_finalized;
                MPI_Finalized(&mpi_finalized);
                if (!mpi_finalized)
                {
                    MPI_Finalize();
                    LOG_DEBUG("MPI finalized by MPIOperatorBase");
                }
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Exception in MPIOperatorBase destructor: " << e.what());
        }
    }

    MPIOperatorBase::MPIOperatorBase(MPIOperatorBase &&other) noexcept
        : comm_(other.comm_), rank_(other.rank_), size_(other.size_),
          mpi_initialized_(other.mpi_initialized_),
          send_buffer_(std::move(other.send_buffer_)),
          recv_buffer_(std::move(other.recv_buffer_))
    {

        // Reset other object to prevent double cleanup
        other.mpi_initialized_ = false;
    }

    MPIOperatorBase &MPIOperatorBase::operator=(MPIOperatorBase &&other) noexcept
    {
        if (this != &other)
        {
            // Cleanup current state
            if (mpi_initialized_)
            {
                finalizeMPI();
                int mpi_finalized;
                MPI_Finalized(&mpi_finalized);
                if (!mpi_finalized)
                {
                    MPI_Finalize();
                }
            }

            // Move from other
            comm_ = other.comm_;
            rank_ = other.rank_;
            size_ = other.size_;
            mpi_initialized_ = other.mpi_initialized_;
            send_buffer_ = std::move(other.send_buffer_);
            recv_buffer_ = std::move(other.recv_buffer_);

            // Reset other
            other.mpi_initialized_ = false;
        }
        return *this;
    }

    std::pair<int, int> MPIOperatorBase::getRowDistribution(int global_size, int rank) const
    {
        if (rank < 0)
            rank = rank_;

        int base_size = global_size / size_;
        int remainder = global_size % size_;

        int local_size = base_size + (rank < remainder ? 1 : 0);
        int offset = rank * base_size + std::min(rank, remainder);

        return {local_size, offset};
    }

    std::pair<int, int> MPIOperatorBase::getColDistribution(int global_size, int rank) const
    {
        // Column distribution uses the same logic as row distribution
        return getRowDistribution(global_size, rank);
    }

    std::tuple<int, int, int, int> MPIOperatorBase::getBlockDistribution(int rows, int cols, int rank) const
    {
        if (rank < 0)
            rank = rank_;

        // Create a 2D process grid as close to square as possible
        int proc_rows = static_cast<int>(std::sqrt(size_));
        while (size_ % proc_rows != 0)
        {
            proc_rows--;
        }
        int proc_cols = size_ / proc_rows;

        // Determine position in process grid
        int grid_row = rank / proc_cols;
        int grid_col = rank % proc_cols;

        // Calculate block distribution
        auto [local_rows, row_offset] = getRowDistribution(rows, grid_row);
        auto [local_cols, col_offset] = getColDistribution(cols, grid_col);

        return {local_rows, local_cols, row_offset, col_offset};
    }

    void MPIOperatorBase::allGather(const float *send_data, int send_count, std::vector<float> &recv_data) const
    {
        int total_count = send_count * size_;
        recv_data.resize(total_count);

        checkMPIError(MPI_Allgather(send_data, send_count, MPI_FLOAT,
                                    recv_data.data(), send_count, MPI_FLOAT, comm_),
                      "MPI_Allgather");
    }

    void MPIOperatorBase::allReduceSum(const float *send_data, float *recv_data, int count) const
    {
        checkMPIError(PerfAllreduce(send_data, recv_data, count, MPI_FLOAT, MPI_SUM, comm_),
                      "MPI_Allreduce (SUM)");
    }

    void MPIOperatorBase::allReduceMax(const float *send_data, float *recv_data, int count) const
    {
        checkMPIError(PerfAllreduce(send_data, recv_data, count, MPI_FLOAT, MPI_MAX, comm_),
                      "MPI_Allreduce (MAX)");
    }

    void MPIOperatorBase::reduceScatter(const float *send_data, float *recv_data, const int *recv_counts) const
    {
        checkMPIError(MPI_Reduce_scatter(send_data, recv_data, recv_counts, MPI_FLOAT, MPI_SUM, comm_),
                      "MPI_Reduce_scatter");
    }

    void MPIOperatorBase::broadcast(float *data, int count, int root) const
    {
        checkMPIError(MPI_Bcast(data, count, MPI_FLOAT, root, comm_),
                      "MPI_Bcast");
    }

    void MPIOperatorBase::allGatherStart(const float *send_data, int send_count,
                                         std::vector<float> &recv_data, MPI_Request *request) const
    {
        int total_count = send_count * size_;
        recv_data.resize(total_count);

        checkMPIError(MPI_Iallgather(send_data, send_count, MPI_FLOAT,
                                     recv_data.data(), send_count, MPI_FLOAT, comm_, request),
                      "MPI_Iallgather");
    }

    void MPIOperatorBase::waitCompletion(MPI_Request *request) const
    {
        checkMPIError(MPI_Wait(request, MPI_STATUS_IGNORE), "MPI_Wait");
    }

    void MPIOperatorBase::checkMPIError(int error_code, const std::string &operation_name) const
    {
        if (error_code != MPI_SUCCESS)
        {
            char error_string[MPI_MAX_ERROR_STRING];
            int length;
            MPI_Error_string(error_code, error_string, &length);

            std::ostringstream oss;
            oss << "MPI Error in " << operation_name << ": " << error_string
                << " (rank " << rank_ << ")";

            LOG_ERROR(oss.str());
            throw std::runtime_error(oss.str());
        }
    }

    void MPIOperatorBase::synchronize() const
    {
        checkMPIError(MPI_Barrier(comm_), "MPI_Barrier");
    }

    std::shared_ptr<TensorBase> MPIOperatorBase::createLocalTensor(const std::vector<size_t> &shape) const
    {
        // Convert size_t vector to int vector for TensorFactory
        std::vector<int> int_shape(shape.begin(), shape.end());

        // For multi-process MPI execution, prefer COSMA tensors for optimal distributed performance
        if (size_ > 1 && shape.size() == 2)
        {
            // Create COSMA tensor for matrix operations
            std::string label = "rank_" + std::to_string(rank_) + "_tensor";
            return TensorFactory::create_cosma(int_shape, label, rank_);
        }
        else
        {
            // Single process or non-matrix tensors use SimpleTensor
            return TensorFactory::create_simple(int_shape);
        }
    }

    std::shared_ptr<TensorBase> MPIOperatorBase::createBroadcastTensor(const std::vector<size_t> &shape) const
    {
        // Convert size_t vector to int vector for TensorFactory
        std::vector<int> int_shape(shape.begin(), shape.end());

        // Always use SimpleTensor for broadcast tensors to ensure compatibility
        // between all MPI ranks (avoids buffer size mismatches during MPI_Bcast)
        return TensorFactory::create_simple(int_shape);
    }

    std::shared_ptr<TensorBase> MPIOperatorBase::createDistributedTensor(const std::vector<size_t> &shape,
                                                                         const std::string &label,
                                                                         int m, int n, int k) const
    {
        // Convert size_t vector to int vector for TensorFactory
        std::vector<int> int_shape(shape.begin(), shape.end());

        if (shape.size() != 2)
        {
            LOG_WARN("createDistributedTensor called for non-matrix shape, falling back to simple tensor");
            return TensorFactory::create_simple(int_shape);
        }

        // Create COSMA strategy optimized for the specific matrix multiplication
        cosma::Strategy strategy(m, n, k, size_);

        // Create COSMA tensor with optimal strategy
        auto cosma_tensor = std::make_shared<COSMATensor>(int_shape, label, strategy, rank_, comm_);

        LOG_DEBUG("Created distributed COSMA tensor [" << m << "x" << n << "] with strategy for " << size_ << " processes");
        return cosma_tensor;
    }

} // namespace llaminar