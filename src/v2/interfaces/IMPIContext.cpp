/**
 * @file IMPIContext.cpp
 * @brief Implementation of IMPIContext factory methods
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "interfaces/IMPIContext.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace llaminar2
{

    /**
     * @brief Minimal mock MPI context for testing
     *
     * This is a simple in-library mock that provides basic functionality
     * for testing without the full MockMPIContext from the test utilities.
     * For more comprehensive mocking (call tracking, failure injection),
     * use MockMPIContext from tests/v2/mocks/MockMPIContext.h
     */
    class SimpleMockMPIContext : public IMPIContext
    {
    public:
        SimpleMockMPIContext(int rank, int world_size)
            : rank_(rank), world_size_(world_size)
        {
            if (rank < 0 || rank >= world_size)
            {
                throw std::invalid_argument("SimpleMockMPIContext: rank must be in [0, world_size)");
            }
            if (world_size < 1)
            {
                throw std::invalid_argument("SimpleMockMPIContext: world_size must be >= 1");
            }
        }

        int rank() const override { return rank_; }
        int world_size() const override { return world_size_; }
        bool is_root() const override { return rank_ == 0; }
        int local_rank() const override { return rank_; }
        MPI_Comm communicator() const override { return MPI_COMM_NULL; }

        void barrier() const override
        {
            // No-op in mock
        }

        void allreduce_sum(const float *send_data, float *recv_data, size_t count) const override
        {
            // In single-process mock, output = input
            std::memcpy(recv_data, send_data, count * sizeof(float));
        }

        void allreduce_sum_inplace(float * /*data*/, size_t /*count*/) const override
        {
            // No-op in single-process mock - data unchanged
        }

        void allreduce_q8_1_inplace(Q8_1Block * /*data*/, size_t /*n_blocks*/) const override
        {
            // No-op
        }

        void allreduce_q16_1_inplace(Q16_1Block * /*data*/, size_t /*n_blocks*/) const override
        {
            // No-op
        }

        void allreduce_fp16_inplace(uint16_t * /*data*/, size_t /*count*/) const override
        {
            // No-op
        }

        void allreduce_bf16_inplace(uint16_t * /*data*/, size_t /*count*/) const override
        {
            // No-op
        }

        void broadcast(float * /*data*/, size_t /*count*/, int /*root*/) const override
        {
            // No-op in mock
        }

        void broadcast_int32(int32_t * /*data*/, size_t /*count*/, int /*root*/) const override
        {
            // No-op in mock
        }

        void allgather(const float *send_data, float *recv_data, size_t count) const override
        {
            // Replicate send_data for each rank
            for (int r = 0; r < world_size_; ++r)
            {
                std::memcpy(recv_data + r * count, send_data, count * sizeof(float));
            }
        }

        void allgather_bytes(const void *send_data, void *recv_data, size_t byte_count) const override
        {
            auto *recv_bytes = static_cast<char *>(recv_data);
            for (int r = 0; r < world_size_; ++r)
            {
                std::memcpy(recv_bytes + r * byte_count, send_data, byte_count);
            }
        }

        void allgatherv_bytes(const void *send_data, int send_count,
                              void *recv_data, const int *recv_counts,
                              const int *displs) const override
        {
            auto *recv_bytes = static_cast<char *>(recv_data);
            for (int r = 0; r < world_size_; ++r)
            {
                int copy_count = std::min(send_count, recv_counts[r]);
                std::memcpy(recv_bytes + displs[r], send_data, copy_count);
            }
        }

        std::pair<size_t, size_t> get_local_slice(size_t total_elements) const override
        {
            size_t base_count = total_elements / world_size_;
            size_t remainder = total_elements % world_size_;

            size_t start = rank_ * base_count +
                           std::min(static_cast<size_t>(rank_), remainder);
            size_t count = base_count + (rank_ < static_cast<int>(remainder) ? 1 : 0);

            return {start, count};
        }

        std::pair<size_t, size_t> distribute_rows(size_t total_rows) const override
        {
            return get_local_slice(total_rows);
        }

        // Point-to-point operations (no-op in mock)
        void send(const void * /*data*/, size_t /*count*/, MPI_Datatype /*type*/,
                  int /*dest*/, int /*tag*/) const override
        {
            // No-op in mock
        }

        void recv(void * /*data*/, size_t /*count*/, MPI_Datatype /*type*/,
                  int /*src*/, int /*tag*/, MPI_Status * /*status*/) const override
        {
            // No-op in mock
        }

        MPI_Request isend(const void * /*data*/, size_t /*count*/, MPI_Datatype /*type*/,
                          int /*dest*/, int /*tag*/) const override
        {
            return MPI_REQUEST_NULL;
        }

        MPI_Request irecv(void * /*data*/, size_t /*count*/, MPI_Datatype /*type*/,
                          int /*src*/, int /*tag*/) const override
        {
            return MPI_REQUEST_NULL;
        }

        void wait(MPI_Request *request, MPI_Status * /*status*/) const override
        {
            if (request)
            {
                *request = MPI_REQUEST_NULL;
            }
        }

        void waitAll(std::vector<MPI_Request> &requests) const override
        {
            for (auto &req : requests)
            {
                req = MPI_REQUEST_NULL;
            }
        }

        void probe(int /*src*/, int /*tag*/, MPI_Status *status) const override
        {
            if (status)
            {
                status->MPI_SOURCE = rank_ == 0 ? 1 : 0;
                status->MPI_TAG = 0;
                status->MPI_ERROR = MPI_SUCCESS;
            }
        }

        bool iprobe(int /*src*/, int /*tag*/, MPI_Status *status) const override
        {
            if (status)
            {
                status->MPI_SOURCE = MPI_ANY_SOURCE;
                status->MPI_TAG = MPI_ANY_TAG;
                status->MPI_ERROR = MPI_SUCCESS;
            }
            return false;
        }

        void sendFloat(const float *data, size_t count, int dest, int tag) const override
        {
            send(data, count, MPI_FLOAT, dest, tag);
        }

        void recvFloat(float *data, size_t count, int src, int tag,
                       MPI_Status *status) const override
        {
            recv(data, count, MPI_FLOAT, src, tag, status);
        }

        void sendBytes(const void *data, size_t byte_count, int dest, int tag) const override
        {
            send(data, byte_count, MPI_BYTE, dest, tag);
        }

        void recvBytes(void *data, size_t byte_count, int src, int tag,
                       MPI_Status *status) const override
        {
            recv(data, byte_count, MPI_BYTE, src, tag, status);
        }

        int getCount(const MPI_Status & /*status*/, MPI_Datatype /*type*/) const override
        {
            return 0; // Mock returns 0
        }

    private:
        int rank_;
        int world_size_;
    };

    std::shared_ptr<IMPIContext> IMPIContext::createMock(int rank, int world_size)
    {
        return std::make_shared<SimpleMockMPIContext>(rank, world_size);
    }

} // namespace llaminar2
