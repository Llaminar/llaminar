/**
 * @file MPIContext.h
 * @brief MPI coordination abstraction for distributed inference
 *
 * Separates MPI coordination from device execution, enabling:
 * - Distributed inference without GPU (OpenBLAS + MPI)
 * - Single-GPU inference without MPI
 * - Clean separation of concerns
 *
 * @author David Sanftenberg
 */

#pragma once

#include <mpi.h>
#include <memory>
#include <utility>
#include <cstddef>
#include <vector>
#include "../tensors/BlockStructures.h"
#include "../tensors/SIMDHelpers.h"

namespace llaminar2
{

    /**
     * @brief MPI context for distributed coordination
     *
     * Encapsulates MPI state and provides clean API for collective operations.
     * Thread-safe for MPI_THREAD_MULTIPLE environments.
     */
    class MPIContext
    {
    public:
        /**
         * @brief Construct MPI context
         *
         * @param rank MPI rank (0-indexed)
         * @param world_size Total number of ranks
         * @param comm MPI communicator (default: MPI_COMM_WORLD)
         */
        MPIContext(int rank, int world_size, MPI_Comm comm = MPI_COMM_WORLD)
            : rank_(rank), world_size_(world_size), comm_(comm) {}

        // Accessors
        int rank() const { return rank_; }
        int world_size() const { return world_size_; }
        MPI_Comm comm() const { return comm_; }

        /**
         * @brief All-reduce sum operation
         *
         * @param send_data Input data (local contribution)
         * @param recv_data Output data (global sum)
         * @param count Number of elements
         */
        void allreduce_sum(const float *send_data, float *recv_data, size_t count) const
        {
            MPI_Allreduce(send_data, recv_data, count, MPI_FLOAT, MPI_SUM, comm_);
        }

        /**
         * @brief All-reduce sum operation (in-place)
         *
         * Uses MPI_IN_PLACE to perform in-place reduction, avoiding memory allocation.
         *
         * @param data Data buffer (input and output)
         * @param count Number of elements
         */
        void allreduce_sum_inplace(float *data, size_t count) const
        {
            MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, comm_);
        }

        /**
         * @brief All-reduce sum operation for Q8_1 blocks (in-place)
         *
         * Performs N-way reduction of Q8_1 blocks across all MPI ranks without
         * FP32 conversion overhead. Uses allgather + vectorized local reduction.
         *
         * Algorithm:
         * 1. Allgather all Q8_1 blocks from all ranks (total: n_blocks * world_size)
         * 2. Sum blocks using AVX512-vectorized reduction (dequant→sum→requant per block)
         * 3. Store result back to input buffer
         *
         * Memory: Temporarily allocates n_blocks * world_size * sizeof(Q8_1Block) bytes.
         * Bandwidth: Q8_1 is 36 bytes/32 elements vs FP32's 128 bytes/32 elements = 3.5x less.
         *
         * @param data Q8_1 block buffer (input and output), size = n_blocks
         * @param n_blocks Number of Q8_1 blocks per rank
         */
        void allreduce_q8_1_inplace(Q8_1Block *data, size_t n_blocks) const
        {
            if (world_size_ == 1)
            {
                return; // No reduction needed for single rank
            }

            const size_t block_bytes = sizeof(Q8_1Block); // 36 bytes
            const size_t total_bytes = n_blocks * block_bytes;
            const size_t gathered_blocks = n_blocks * world_size_;

            // Allocate buffer for all ranks' contributions
            std::vector<Q8_1Block> gathered(gathered_blocks);

            // Allgather Q8_1 blocks from all ranks
            MPI_Allgather(data, total_bytes, MPI_BYTE,
                          gathered.data(), total_bytes, MPI_BYTE, comm_);

            // Build pointer array for q8_1_sum_n
            std::vector<const Q8_1Block *> inputs(world_size_);
            for (int r = 0; r < world_size_; ++r)
            {
                inputs[r] = gathered.data() + r * n_blocks;
            }

            // Sum all contributions using AVX512-vectorized reduction
            simd::q8_1_sum_n(inputs.data(), world_size_, data, n_blocks);
        }

        /**
         * @brief All-reduce sum operation for Q16_1 blocks (in-place)
         *
         * Performs N-way reduction of Q16_1 quantized blocks across all MPI ranks.
         * Uses allgather + vectorized local reduction with AVX512/AVX2 SIMD.
         *
         * Memory: Temporarily allocates n_blocks * world_size * sizeof(Q16_1Block) bytes.
         *
         * @param data Q16_1 block buffer (input and output)
         * @param n_blocks Number of Q16_1 blocks per rank
         */
        void allreduce_q16_1_inplace(Q16_1Block *data, size_t n_blocks) const
        {
            if (world_size_ == 1)
            {
                return; // No reduction needed for single rank
            }

            const size_t block_bytes = sizeof(Q16_1Block); // 72 bytes (32*int16 + float + int32)
            const size_t total_bytes = n_blocks * block_bytes;
            const size_t gathered_blocks = n_blocks * world_size_;

            // Allocate buffer for all ranks' contributions
            std::vector<Q16_1Block> gathered(gathered_blocks);

            // Allgather Q16_1 blocks from all ranks
            MPI_Allgather(data, total_bytes, MPI_BYTE,
                          gathered.data(), total_bytes, MPI_BYTE, comm_);

            // Build pointer array for q16_1_sum_n
            std::vector<const Q16_1Block *> inputs(world_size_);
            for (int r = 0; r < world_size_; ++r)
            {
                inputs[r] = gathered.data() + r * n_blocks;
            }

            // Sum all contributions using AVX512/AVX2 vectorized reduction
            simd::q16_1_sum_n(inputs.data(), world_size_, data, n_blocks);
        }

        /**
         * @brief All-reduce sum operation for variable-size Q16 blocks (in-place)
         *
         * Templated version that supports Q16_1Block (32), Q16_1Block_64 (64),
         * and Q16_1Block_128 (128) element blocks.
         *
         * Uses allgather + vectorized local reduction with AVX512/AVX2 SIMD.
         *
         * Memory: Temporarily allocates n_blocks * world_size * sizeof(BlockType) bytes.
         *
         * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, Q16_1Block_128)
         * @param data Q16 block buffer (input and output)
         * @param n_blocks Number of Q16 blocks per rank
         */
        template <typename BlockType>
        void allreduce_q16_inplace(BlockType *data, size_t n_blocks) const
        {
            if (world_size_ == 1)
            {
                return; // No reduction needed for single rank
            }

            const size_t block_bytes = sizeof(BlockType);
            const size_t total_bytes = n_blocks * block_bytes;
            const size_t gathered_blocks = n_blocks * world_size_;

            // Allocate buffer for all ranks' contributions
            std::vector<BlockType> gathered(gathered_blocks);

            // Allgather Q16 blocks from all ranks
            MPI_Allgather(data, total_bytes, MPI_BYTE,
                          gathered.data(), total_bytes, MPI_BYTE, comm_);

            // Sum all contributions using AVX512/AVX2 vectorized reduction
            // q16_sum_n processes one block at a time, so loop over blocks
            std::vector<const BlockType *> inputs(world_size_);
            for (size_t b = 0; b < n_blocks; ++b)
            {
                // Build pointer array for this block position across all ranks
                for (int r = 0; r < world_size_; ++r)
                {
                    inputs[r] = gathered.data() + r * n_blocks + b;
                }
                simd::q16_sum_n<BlockType>(inputs.data(), world_size_, data + b);
            }
        }

        /**
         * @brief All-reduce sum operation for FP16 elements (in-place)
         *
         * Performs N-way reduction of FP16 values across all MPI ranks without
         * FP32 conversion overhead in communication. Uses allgather + vectorized
         * local reduction with AVX512/AVX2 F16C instructions.
         *
         * Memory: Temporarily allocates count * world_size * sizeof(uint16_t) bytes.
         * Bandwidth: FP16 is 2 bytes/element vs FP32's 4 bytes/element = 2x less.
         *
         * @param data FP16 buffer (uint16_t, input and output)
         * @param count Number of FP16 elements per rank
         */
        void allreduce_fp16_inplace(uint16_t *data, size_t count) const
        {
            if (world_size_ == 1)
            {
                return; // No reduction needed for single rank
            }

            const size_t total_bytes = count * sizeof(uint16_t);
            const size_t gathered_count = count * world_size_;

            // Allocate buffer for all ranks' contributions
            std::vector<uint16_t> gathered(gathered_count);

            // Allgather FP16 elements from all ranks
            MPI_Allgather(data, total_bytes, MPI_BYTE,
                          gathered.data(), total_bytes, MPI_BYTE, comm_);

            // Build pointer array for fp16_sum_n
            std::vector<const uint16_t *> inputs(world_size_);
            for (int r = 0; r < world_size_; ++r)
            {
                inputs[r] = gathered.data() + r * count;
            }

            // Sum all contributions using AVX512-vectorized reduction
            simd::fp16_sum_n(inputs.data(), world_size_, data, count);
        }

        /**
         * @brief All-reduce sum operation for BF16 elements (in-place)
         *
         * Performs N-way reduction of BF16 values across all MPI ranks without
         * FP32 conversion overhead in communication. Uses allgather + vectorized
         * local reduction with AVX512 BF16 conversion.
         *
         * Memory: Temporarily allocates count * world_size * sizeof(uint16_t) bytes.
         * Bandwidth: BF16 is 2 bytes/element vs FP32's 4 bytes/element = 2x less.
         *
         * @param data BF16 buffer (uint16_t, input and output)
         * @param count Number of BF16 elements per rank
         */
        void allreduce_bf16_inplace(uint16_t *data, size_t count) const
        {
            if (world_size_ == 1)
            {
                return; // No reduction needed for single rank
            }

            const size_t total_bytes = count * sizeof(uint16_t);
            const size_t gathered_count = count * world_size_;

            // Allocate buffer for all ranks' contributions
            std::vector<uint16_t> gathered(gathered_count);

            // Allgather BF16 elements from all ranks
            MPI_Allgather(data, total_bytes, MPI_BYTE,
                          gathered.data(), total_bytes, MPI_BYTE, comm_);

            // Build pointer array for bf16_sum_n
            std::vector<const uint16_t *> inputs(world_size_);
            for (int r = 0; r < world_size_; ++r)
            {
                inputs[r] = gathered.data() + r * count;
            }

            // Sum all contributions using AVX512-vectorized reduction
            simd::bf16_sum_n(inputs.data(), world_size_, data, count);
        }

        /**
         * @brief Broadcast data from root to all ranks
         *
         * @param data Data buffer (input on root, output on others)
         * @param count Number of elements
         * @param root Root rank (default: 0)
         */
        void broadcast(float *data, size_t count, int root = 0) const
        {
            MPI_Bcast(data, count, MPI_FLOAT, root, comm_);
        }

        /**
         * @brief All-gather operation
         *
         * @param send_data Local data
         * @param recv_data Global data (concatenated from all ranks)
         * @param count Number of elements per rank
         */
        void allgather(const float *send_data, float *recv_data, size_t count) const
        {
            MPI_Allgather(send_data, count, MPI_FLOAT, recv_data, count, MPI_FLOAT, comm_);
        }

        /**
         * @brief All-gather bytes operation (type-agnostic)
         *
         * Used for Q8_1 block communication where we want to avoid
         * dequant→allreduce→requant round trips. Each rank sends its
         * local bytes to all other ranks.
         *
         * @param send_data Local data (raw bytes)
         * @param recv_data Global data (concatenated from all ranks)
         * @param byte_count Number of bytes per rank
         */
        void allgather_bytes(const void *send_data, void *recv_data, size_t byte_count) const
        {
            MPI_Allgather(send_data, byte_count, MPI_BYTE, recv_data, byte_count, MPI_BYTE, comm_);
        }

        /**
         * @brief All-gather with variable counts per rank (bytes)
         *
         * Each rank can contribute a different number of bytes.
         * Used for Q8_1 attention where ranks may have different head counts.
         *
         * @param send_data Local data (raw bytes)
         * @param send_count Number of bytes this rank is sending
         * @param recv_data Global data buffer (must be pre-sized)
         * @param recv_counts Array of counts from each rank
         * @param displs Array of displacements in recv_data for each rank
         */
        void allgatherv_bytes(const void *send_data, int send_count,
                              void *recv_data, const int *recv_counts, const int *displs) const
        {
            MPI_Allgatherv(send_data, send_count, MPI_BYTE,
                           recv_data, recv_counts, displs, MPI_BYTE, comm_);
        }

        /**
         * @brief Synchronization barrier
         *
         * Blocks until all ranks reach this point.
         */
        void barrier() const
        {
            MPI_Barrier(comm_);
        }

        /**
         * @brief Get local slice for distributed work
         *
         * @param total_elements Total number of elements to distribute
         * @return {start_index, count} for this rank
         */
        std::pair<size_t, size_t> get_local_slice(size_t total_elements) const
        {
            size_t base_count = total_elements / world_size_;
            size_t remainder = total_elements % world_size_;

            size_t start = rank_ * base_count + std::min(static_cast<size_t>(rank_), remainder);
            size_t count = base_count + (rank_ < remainder ? 1 : 0);

            return {start, count};
        }

        /**
         * @brief Distribute rows across ranks (for matrix sharding)
         *
         * @param total_rows Total number of rows
         * @return {start_row, num_rows} for this rank
         */
        std::pair<size_t, size_t> distribute_rows(size_t total_rows) const
        {
            return get_local_slice(total_rows);
        }

    private:
        int rank_;
        int world_size_;
        MPI_Comm comm_;
    };

    /**
     * @brief Factory for creating MPI contexts
     */
    class MPIContextFactory
    {
    public:
        /**
         * @brief Get global MPI context (MPI_COMM_WORLD)
         *
         * @return Shared pointer to global context (singleton)
         */
        static std::shared_ptr<MPIContext> global()
        {
            static std::shared_ptr<MPIContext> instance;
            if (!instance)
            {
                int rank, world_size;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);
                instance = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
            }
            return instance;
        }

        /**
         * @brief Create mock MPI context for testing
         *
         * @param rank Simulated rank
         * @param world_size Simulated world size
         * @return Mock context (no actual MPI calls)
         */
        static std::shared_ptr<MPIContext> create_mock(int rank, int world_size)
        {
            return std::make_shared<MPIContext>(rank, world_size, MPI_COMM_SELF);
        }
    };

} // namespace llaminar2
