/**
 * @file Test__UPIBackendMPI.cpp
 * @brief Integration tests for UPICollectiveBackend with actual MPI
 *
 * These tests require running with mpirun -np 2 (or more).
 * They test actual MPI collective operations over UPI.
 *
 * Test categories:
 * - Collective operations (AllReduce, AllGather, Broadcast, Barrier)
 * - Domain communicator splitting
 * - Performance benchmarks (latency and bandwidth)
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "v2/collective/backends/UPIBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/utils/Logger.h"
#include <vector>
#include <numeric>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace llaminar2::test
{

    // =============================================================================
    // Test Fixture with MPI
    // =============================================================================

    class Test__UPIBackendMPI : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Get world rank and size
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

            // Skip tests if running single rank
            if (world_size_ < 2)
            {
                GTEST_SKIP() << "UPI backend tests require at least 2 MPI ranks";
            }

            // Create UPI backend with MPI_COMM_WORLD as domain communicator
            // In real usage, this would be a split communicator
            backend_ = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD);

            // Initialize with a local CPU group
            auto group = createLocalCPUGroup();
            ASSERT_TRUE(backend_->initialize(group));
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
            backend_.reset();
        }

        // Helper to create a local-scope CPU group
        DeviceGroup createLocalCPUGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("local_cpu_upi_test")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cpu())
                .setLocalRank(world_rank_)
                .build();
        }

        std::unique_ptr<UPICollectiveBackend> backend_;
        int world_rank_ = 0;
        int world_size_ = 0;
    };

    // =============================================================================
    // AllReduce Tests
    // =============================================================================

    TEST_F(Test__UPIBackendMPI, AllReduceSumFP32)
    {
        // Each rank has values [1, 2, 3, 4] * (rank + 1)
        std::vector<float> buffer(4);
        for (int i = 0; i < 4; ++i)
        {
            buffer[i] = static_cast<float>((i + 1) * (world_rank_ + 1));
        }

        // AllReduce SUM
        ASSERT_TRUE(backend_->allreduce(
            buffer.data(), 4, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Expected: sum of (rank+1) from rank 0 to world_size-1 = world_size*(world_size+1)/2
        // For each element i: sum = (i+1) * [1 + 2 + ... + world_size] = (i+1) * world_size*(world_size+1)/2
        float sum_of_ranks = static_cast<float>(world_size_ * (world_size_ + 1)) / 2.0f;
        for (int i = 0; i < 4; ++i)
        {
            float expected = static_cast<float>(i + 1) * sum_of_ranks;
            EXPECT_FLOAT_EQ(buffer[i], expected) << "Mismatch at index " << i;
        }
    }

    TEST_F(Test__UPIBackendMPI, AllReduceSumFP16AsUINT16)
    {
        // Test with FLOAT16 type (treated as UINT16 for byte transfer)
        // This tests the raw byte transfer, not actual FP16 reduction
        std::vector<uint16_t> buffer(4);
        for (int i = 0; i < 4; ++i)
        {
            buffer[i] = static_cast<uint16_t>(1000 * (world_rank_ + 1) + i);
        }

        uint16_t original_val = buffer[0];

        // Note: This uses MPI_SUM on UINT16 values, which works for integer addition
        // but NOT for actual FP16 reduction (which would need custom MPI_Op)
        ASSERT_TRUE(backend_->allreduce(
            buffer.data(), 4, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_SUM));

        // For testing, we just verify the operation completed without error
        // and the values changed (since it's a sum across ranks)
        if (world_size_ > 1)
        {
            // Value should be larger than original (sum of multiple ranks)
            EXPECT_GT(buffer[0], original_val);
        }
    }

    TEST_F(Test__UPIBackendMPI, AllReduceMaxFP32)
    {
        // Each rank sets buffer[i] = (i+1) * (rank+1)
        std::vector<float> buffer(4);
        for (int i = 0; i < 4; ++i)
        {
            buffer[i] = static_cast<float>((i + 1) * (world_rank_ + 1));
        }

        // AllReduce MAX
        ASSERT_TRUE(backend_->allreduce(
            buffer.data(), 4, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_MAX));

        // Expected: max value comes from highest rank
        for (int i = 0; i < 4; ++i)
        {
            float expected = static_cast<float>((i + 1) * world_size_);
            EXPECT_FLOAT_EQ(buffer[i], expected) << "Mismatch at index " << i;
        }
    }

    TEST_F(Test__UPIBackendMPI, AllReduceMinFP32)
    {
        // Each rank sets buffer[i] = (i+1) * (rank+1)
        std::vector<float> buffer(4);
        for (int i = 0; i < 4; ++i)
        {
            buffer[i] = static_cast<float>((i + 1) * (world_rank_ + 1));
        }

        // AllReduce MIN
        ASSERT_TRUE(backend_->allreduce(
            buffer.data(), 4, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_MIN));

        // Expected: min value comes from rank 0
        for (int i = 0; i < 4; ++i)
        {
            float expected = static_cast<float>(i + 1); // rank 0's values
            EXPECT_FLOAT_EQ(buffer[i], expected) << "Mismatch at index " << i;
        }
    }

    // =============================================================================
    // AllGather Tests
    // =============================================================================

    TEST_F(Test__UPIBackendMPI, AllGatherFP32)
    {
        const size_t send_count = 4;

        // Each rank sends [rank*10, rank*10+1, rank*10+2, rank*10+3]
        std::vector<float> send_buf(send_count);
        for (size_t i = 0; i < send_count; ++i)
        {
            send_buf[i] = static_cast<float>(world_rank_ * 10 + i);
        }

        // Receive buffer for all ranks
        std::vector<float> recv_buf(send_count * world_size_);

        ASSERT_TRUE(backend_->allgather(
            send_buf.data(), recv_buf.data(), send_count, CollectiveDataType::FLOAT32));

        // Verify: recv_buf should contain all ranks' data concatenated
        for (int rank = 0; rank < world_size_; ++rank)
        {
            for (size_t i = 0; i < send_count; ++i)
            {
                float expected = static_cast<float>(rank * 10 + i);
                float actual = recv_buf[rank * send_count + i];
                EXPECT_FLOAT_EQ(actual, expected)
                    << "Mismatch at rank " << rank << ", index " << i;
            }
        }
    }

    TEST_F(Test__UPIBackendMPI, AllGatherVVariableSizes)
    {
        // Each rank sends (rank+1) elements
        size_t send_count = static_cast<size_t>(world_rank_ + 1);

        // Send buffer: [rank * 100 + 0, rank * 100 + 1, ...]
        std::vector<float> send_buf(send_count);
        for (size_t i = 0; i < send_count; ++i)
        {
            send_buf[i] = static_cast<float>(world_rank_ * 100 + i);
        }

        // Calculate total receive size and displacements
        std::vector<int> recv_counts(world_size_);
        std::vector<int> displacements(world_size_);
        int total_recv = 0;
        for (int r = 0; r < world_size_; ++r)
        {
            recv_counts[r] = r + 1;
            displacements[r] = total_recv;
            total_recv += recv_counts[r];
        }

        std::vector<float> recv_buf(total_recv);

        ASSERT_TRUE(backend_->allgatherv(
            send_buf.data(), send_count,
            recv_buf.data(), recv_counts, displacements,
            CollectiveDataType::FLOAT32));

        // Verify: each rank's data should be at its displacement
        for (int rank = 0; rank < world_size_; ++rank)
        {
            int count = recv_counts[rank];
            int disp = displacements[rank];
            for (int i = 0; i < count; ++i)
            {
                float expected = static_cast<float>(rank * 100 + i);
                float actual = recv_buf[disp + i];
                EXPECT_FLOAT_EQ(actual, expected)
                    << "Mismatch at rank " << rank << ", index " << i;
            }
        }
    }

    // =============================================================================
    // Broadcast Tests
    // =============================================================================

    TEST_F(Test__UPIBackendMPI, BroadcastFromRank0)
    {
        std::vector<float> buffer(4);

        if (world_rank_ == 0)
        {
            // Root sets values
            for (int i = 0; i < 4; ++i)
            {
                buffer[i] = static_cast<float>(100 + i);
            }
        }
        else
        {
            // Non-root starts with zeros
            for (int i = 0; i < 4; ++i)
            {
                buffer[i] = 0.0f;
            }
        }

        ASSERT_TRUE(backend_->broadcast(buffer.data(), 4, CollectiveDataType::FLOAT32, 0));

        // All ranks should have root's values
        for (int i = 0; i < 4; ++i)
        {
            EXPECT_FLOAT_EQ(buffer[i], static_cast<float>(100 + i))
                << "Mismatch at index " << i << " on rank " << world_rank_;
        }
    }

    TEST_F(Test__UPIBackendMPI, BroadcastFromRank1)
    {
        if (world_size_ < 2)
        {
            GTEST_SKIP() << "Need at least 2 ranks for broadcast from rank 1";
        }

        std::vector<float> buffer(4);

        if (world_rank_ == 1)
        {
            // Root (rank 1) sets values
            for (int i = 0; i < 4; ++i)
            {
                buffer[i] = static_cast<float>(200 + i);
            }
        }
        else
        {
            // Non-root starts with zeros
            for (int i = 0; i < 4; ++i)
            {
                buffer[i] = 0.0f;
            }
        }

        ASSERT_TRUE(backend_->broadcast(buffer.data(), 4, CollectiveDataType::FLOAT32, 1));

        // All ranks should have rank 1's values
        for (int i = 0; i < 4; ++i)
        {
            EXPECT_FLOAT_EQ(buffer[i], static_cast<float>(200 + i))
                << "Mismatch at index " << i << " on rank " << world_rank_;
        }
    }

    // =============================================================================
    // Barrier Test
    // =============================================================================

    TEST_F(Test__UPIBackendMPI, Barrier)
    {
        // Simple barrier test - all ranks synchronize
        ASSERT_TRUE(backend_->synchronize());

        // If we got here, barrier succeeded on all ranks
        SUCCEED();
    }

    // =============================================================================
    // Domain Communicator Split Test
    // =============================================================================

    TEST_F(Test__UPIBackendMPI, DomainCommunicatorSplit)
    {
        // Test creating a split communicator (simulates TP domain creation)
        // Split ranks into two groups: even and odd
        int color = world_rank_ % 2;

        MPI_Comm split_comm;
        MPI_Comm_split(MPI_COMM_WORLD, color, world_rank_, &split_comm);

        // Create backend with split communicator
        auto split_backend = std::make_unique<UPICollectiveBackend>(split_comm);

        auto group = createLocalCPUGroup();
        ASSERT_TRUE(split_backend->initialize(group));

        // Verify domain size is half (or close to half) of world size
        int expected_domain_size = (world_size_ + 1 - color) / 2;
        EXPECT_EQ(split_backend->domainSize(), expected_domain_size);

        // Test allreduce within split domain
        float buffer = static_cast<float>(world_rank_ + 1);
        ASSERT_TRUE(split_backend->allreduce(
            &buffer, 1, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify result is sum of (rank+1) for ranks in this domain
        float expected_sum = 0.0f;
        for (int r = color; r < world_size_; r += 2)
        {
            expected_sum += static_cast<float>(r + 1);
        }
        EXPECT_FLOAT_EQ(buffer, expected_sum);

        // Cleanup
        split_backend->shutdown();
        MPI_Comm_free(&split_comm);
    }

    // =============================================================================
    // Performance Benchmarks
    // =============================================================================

    TEST_F(Test__UPIBackendMPI, LatencyBenchmarkSmall)
    {
        // Measure latency for small allreduce (typical for scalar reductions)
        const int num_warmup = 10;
        const int num_iters = 100;
        float buffer = 1.0f;

        // Warmup
        for (int i = 0; i < num_warmup; ++i)
        {
            backend_->allreduce(&buffer, 1, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        }
        backend_->synchronize();

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iters; ++i)
        {
            backend_->allreduce(&buffer, 1, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        }
        backend_->synchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double avg_latency_us = total_us / num_iters;

        if (world_rank_ == 0)
        {
            LOG_INFO("UPI AllReduce latency (1 float): " << avg_latency_us << " μs");
        }

        // Target: <10μs for small messages on fast interconnect
        // Relaxed to <100μs for CI environments with oversubscription
        EXPECT_LT(avg_latency_us, 100.0)
            << "AllReduce latency " << avg_latency_us << " μs exceeds 100 μs threshold";
    }

    TEST_F(Test__UPIBackendMPI, BandwidthBenchmarkLarge)
    {
        // Measure bandwidth for large allreduce (typical for tensor reductions)
        const size_t num_elements = 1024 * 1024; // 4 MB of floats
        const int num_warmup = 3;
        const int num_iters = 10;

        std::vector<float> buffer(num_elements, 1.0f);

        // Warmup
        for (int i = 0; i < num_warmup; ++i)
        {
            backend_->allreduce(buffer.data(), num_elements,
                                CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        }
        backend_->synchronize();

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iters; ++i)
        {
            backend_->allreduce(buffer.data(), num_elements,
                                CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        }
        backend_->synchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double total_sec = std::chrono::duration<double>(end - start).count();
        double bytes_per_iter = num_elements * sizeof(float);
        double total_bytes = bytes_per_iter * num_iters;
        double bandwidth_gbps = (total_bytes / total_sec) / 1e9;

        if (world_rank_ == 0)
        {
            LOG_INFO("UPI AllReduce bandwidth (" << num_elements << " floats): "
                                                 << bandwidth_gbps << " GB/s");
            LOG_INFO("Estimated UPI bandwidth from topology: "
                     << backend_->estimatedBandwidthGBps() << " GB/s");
        }

        // Note: Actual bandwidth depends on many factors (shared memory, network, etc.)
        // In CI with oversubscription, we won't hit real UPI bandwidth
        // Just verify it's positive and operations succeed
        EXPECT_GT(bandwidth_gbps, 0.0) << "Measured bandwidth should be positive";
    }

    /**
     * @test Compare the timeout-aware UPI request-progress loop with blocking MPI
     *       at the exact Qwen3.6-35B prefill reduction geometry.
     *
     * Qwen3.6-35B publishes 434 rows of 2,048 FP32 hidden values at each
     * tensor-parallel reduction.  A historical timeout hardening change replaced
     * blocking MPI progress with `MPI_Test()` followed by a one-millisecond sleep.
     * Open MPI may require many `MPI_Test()` calls to advance a segmented large
     * reduction, so that sleep multiplied protocol progress into roughly 85 ms
     * per collective.  A generic "bandwidth is positive" assertion did not catch
     * the resulting order-of-magnitude prefill regression.
     *
     * This test measures a blocking `MPI_Allreduce` control and the production
     * `UPICollectiveBackend` path back-to-back with the same communicator, payload,
     * process placement, and warmed buffers.  The production path retains its
     * timeout and fatal failure semantics, but its progress machinery must remain
     * close to the MPI transport floor.  A relative gate is used so overloaded CI
     * hosts do not need to meet this workstation's absolute UPI bandwidth.
     */
    TEST_F(Test__UPIBackendMPI, PrefillPayloadProgressTracksBlockingMPI)
    {
        constexpr size_t kPrefillRows = 434;
        constexpr size_t kHiddenDimension = 2048;
        constexpr size_t kElementCount = kPrefillRows * kHiddenDimension;
        constexpr int kWarmupIterations = 3;
        constexpr int kMeasuredIterations = 10;

        std::vector<float> buffer(kElementCount, static_cast<float>(world_rank_ + 1));

        const auto reset_buffer = [&]()
        {
            std::fill(
                buffer.begin(),
                buffer.end(),
                static_cast<float>(world_rank_ + 1));
        };

        const auto measure_blocking_mpi = [&]() -> double
        {
            for (int iteration = 0; iteration < kWarmupIterations; ++iteration)
            {
                reset_buffer();
                EXPECT_EQ(
                    MPI_Allreduce(
                        MPI_IN_PLACE,
                        buffer.data(),
                        static_cast<int>(kElementCount),
                        MPI_FLOAT,
                        MPI_SUM,
                        MPI_COMM_WORLD),
                    MPI_SUCCESS);
            }

            EXPECT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < kMeasuredIterations; ++iteration)
            {
                reset_buffer();
                EXPECT_EQ(
                    MPI_Allreduce(
                        MPI_IN_PLACE,
                        buffer.data(),
                        static_cast<int>(kElementCount),
                        MPI_FLOAT,
                        MPI_SUM,
                        MPI_COMM_WORLD),
                    MPI_SUCCESS);
            }
            const auto stop = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::micro>(stop - start).count() /
                   static_cast<double>(kMeasuredIterations);
        };

        const auto measure_production_upi = [&]() -> double
        {
            for (int iteration = 0; iteration < kWarmupIterations; ++iteration)
            {
                reset_buffer();
                EXPECT_TRUE(backend_->allreduce(
                    buffer.data(),
                    kElementCount,
                    CollectiveDataType::FLOAT32,
                    CollectiveOp::ALLREDUCE_SUM));
            }

            EXPECT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < kMeasuredIterations; ++iteration)
            {
                reset_buffer();
                EXPECT_TRUE(backend_->allreduce(
                    buffer.data(),
                    kElementCount,
                    CollectiveDataType::FLOAT32,
                    CollectiveOp::ALLREDUCE_SUM));
            }
            const auto stop = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::micro>(stop - start).count() /
                   static_cast<double>(kMeasuredIterations);
        };

        const double blocking_local_us = measure_blocking_mpi();
        const double production_local_us = measure_production_upi();
        double blocking_max_us = 0.0;
        double production_max_us = 0.0;
        ASSERT_EQ(
            MPI_Allreduce(
                &blocking_local_us,
                &blocking_max_us,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD),
            MPI_SUCCESS);
        ASSERT_EQ(
            MPI_Allreduce(
                &production_local_us,
                &production_max_us,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD),
            MPI_SUCCESS);

        if (world_rank_ == 0)
        {
            LOG_INFO("Qwen3.6-35B prefill allreduce control="
                     << blocking_max_us << " us, production UPI="
                     << production_max_us << " us, ratio="
                     << production_max_us / blocking_max_us);
        }

        const double allowed_production_us =
            std::max(blocking_max_us * 3.0, blocking_max_us + 250.0);
        EXPECT_LE(production_max_us, allowed_production_us)
            << "Timeout-aware UPI progress is materially slower than blocking MPI "
            << "for the production 434x2048 FP32 prefill payload";
    }

    // =============================================================================
    // ReduceScatter Test
    // =============================================================================

    TEST_F(Test__UPIBackendMPI, ReduceScatterFP32)
    {
        // Each rank has a full array [1, 2, 3, 4] * (rank + 1)
        // After reduce-scatter, each rank gets one element of the sum
        const size_t total_elements = 4;
        const size_t recv_count = total_elements / world_size_;

        if (total_elements % world_size_ != 0)
        {
            GTEST_SKIP() << "Test requires total_elements divisible by world_size";
        }

        std::vector<float> send_buf(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            send_buf[i] = static_cast<float>((i + 1) * (world_rank_ + 1));
        }

        std::vector<float> recv_buf(recv_count);

        ASSERT_TRUE(backend_->reduceScatter(
            send_buf.data(), recv_buf.data(), recv_count,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify: each rank gets elements [rank*recv_count, (rank+1)*recv_count)
        // summed across all ranks
        float sum_of_ranks = static_cast<float>(world_size_ * (world_size_ + 1)) / 2.0f;
        for (size_t i = 0; i < recv_count; ++i)
        {
            size_t global_idx = world_rank_ * recv_count + i;
            float expected = static_cast<float>(global_idx + 1) * sum_of_ranks;
            EXPECT_FLOAT_EQ(recv_buf[i], expected)
                << "Mismatch at local index " << i << " on rank " << world_rank_;
        }
    }

} // namespace llaminar2::test
