/**
 * @file Perf__CPURootedPublicationTransport.cpp
 * @brief Measure same-node CPU rooted-publication transport crossovers.
 *
 * This focused two-rank harness compares the bounded native shared-memory
 * protocols with MPI payload transport for the two communication operations
 * used by exact CPU MoE publication: variable packed-record gather and compact
 * output broadcast. It sweeps payloads from decode-sized messages through
 * long-prefill messages and reports slowest-rank p50/p95 latency and effective
 * payload bandwidth.
 *
 * Profiler or benchmark output from this executable is evidence for a typed
 * transport policy. It is not a correctness substitute: production graph and
 * all-format byte-equivalence tests remain the authority for arithmetic.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "collective/DeviceGroup.h"
#include "collective/backends/ShmemSpinBackend.h"
#include "collective/backends/UPIBackend.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string_view>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        /** Return this process's rank in the benchmark communicator. */
        int worldRank()
        {
            int rank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank;
        }

        /** Return the process count in the benchmark communicator. */
        int worldSize()
        {
            int size = 0;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return size;
        }

        /** Construct the CPU device group consumed by both transport backends. */
        DeviceGroup makeCpuGroup(int size)
        {
            DeviceGroup group;
            group.name = "perf_cpu_rooted_publication_transport";
            group.scope = CollectiveScope::LOCAL;
            for (int rank = 0; rank < size; ++rank)
                group.devices.push_back(DeviceId::cpu());
            return group;
        }

        /** Choose enough samples for stable latency without making large sweeps slow. */
        int repetitionsForBytes(size_t bytes)
        {
            if (bytes <= 256u * 1024u)
                return 200;
            if (bytes <= 4u * 1024u * 1024u)
                return 60;
            return 20;
        }

        /** Select a nearest-rank percentile from an already sorted sample set. */
        double percentile(
            const std::vector<double> &sorted_samples,
            size_t numerator)
        {
            const size_t rank =
                (numerator * sorted_samples.size() + 99u) / 100u;
            return sorted_samples[std::max<size_t>(1u, rank) - 1u];
        }

        /** Slowest-rank latency summary for one exact transport and payload. */
        struct LatencySummary
        {
            double p50_us = 0.0;
            double p95_us = 0.0;
        };

        /**
         * Combine rank-local samples by maximum before calculating percentiles.
         *
         * A rooted transaction is complete only when its slowest participant is
         * complete. Reducing the full sample vector once after measurement avoids
         * placing an MPI timing reduction inside every canonical sample.
         */
        LatencySummary summarizeSlowestRank(
            const std::vector<double> &local_samples)
        {
            std::vector<double> slowest_samples(local_samples.size(), 0.0);
            EXPECT_EQ(
                MPI_Allreduce(
                    local_samples.data(),
                    slowest_samples.data(),
                    static_cast<int>(local_samples.size()),
                    MPI_DOUBLE,
                    MPI_MAX,
                    MPI_COMM_WORLD),
                MPI_SUCCESS);
            std::sort(slowest_samples.begin(), slowest_samples.end());
            return {
                .p50_us = percentile(slowest_samples, 50u),
                .p95_us = percentile(slowest_samples, 95u),
            };
        }

        /** Emit one stable CSV row from rank zero. */
        void printResult(
            std::string_view operation,
            size_t bytes,
            std::string_view transport,
            const LatencySummary &summary)
        {
            if (worldRank() != 0)
                return;
            const double p50_gbps =
                static_cast<double>(bytes) / summary.p50_us / 1000.0;
            std::cout << operation << ',' << bytes << ',' << transport << ','
                      << std::fixed << std::setprecision(3)
                      << summary.p50_us << ',' << summary.p95_us << ','
                      << p50_gbps << '\n';
        }
    } // namespace

    /**
     * @test SweepNativeSharedMemoryAndMPIByPayloadSize
     *
     * Measures each candidate after independent warmup. Gather payload bytes are
     * the sum of both participant contributions; broadcast bytes are the compact
     * result size delivered to the peer. Canonical timing excludes initialization,
     * allocation, validation output, and cross-rank sample reduction.
     */
    TEST(
        Perf__CPURootedPublicationTransport,
        SweepNativeSharedMemoryAndMPIByPayloadSize)
    {
        const int rank = worldRank();
        const int size = worldSize();
        if (size != 2)
            GTEST_SKIP() << "Rooted publication crossover requires exactly two ranks";

        auto upi =
            std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
        ShmemSpinBackend shmem(/*domain_id=*/73001, rank, std::move(upi));
        ASSERT_TRUE(shmem.initialize(makeCpuGroup(size))) << shmem.lastError();

        const std::vector<size_t> payload_bytes = {
            8u * 1024u,
            16u * 1024u,
            32u * 1024u,
            64u * 1024u,
            128u * 1024u,
            256u * 1024u,
            512u * 1024u,
            1u * 1024u * 1024u,
            2u * 1024u * 1024u,
            4u * 1024u * 1024u,
            8u * 1024u * 1024u,
            16u * 1024u * 1024u,
            32u * 1024u * 1024u,
        };

        if (rank == 0)
        {
            std::cout << "operation,bytes,transport,p50_us,p95_us,p50_gbps\n";
        }

        for (const size_t bytes : payload_bytes)
        {
            ASSERT_EQ(bytes % (2u * sizeof(float)), 0u);
            const size_t total_elements = bytes / sizeof(float);
            const size_t local_elements = total_elements / 2u;
            std::vector<float> local(local_elements);
            std::iota(
                local.begin(),
                local.end(),
                static_cast<float>(rank * 1024));
            std::vector<float> root_records;
            if (rank == 0)
                root_records.resize(total_elements);

            const int repetitions = repetitionsForBytes(bytes);
            const auto run_gather = [&](bool native_shared_memory) {
                std::vector<double> local_samples(
                    static_cast<size_t>(repetitions));
                for (int warmup = 0; warmup < 3; ++warmup)
                {
                    size_t gathered = 0u;
                    if (native_shared_memory)
                    {
                        EXPECT_TRUE(shmem.gatherVariableFloatRecordsToRoot(
                            local.data(),
                            local_elements,
                            rank == 0 ? root_records.data() : nullptr,
                            total_elements,
                            /*record_width_elements=*/1u,
                            /*root_rank=*/0,
                            gathered));
                    }
                    else if (rank == 0)
                    {
                        std::memcpy(
                            root_records.data(),
                            local.data(),
                            local_elements * sizeof(float));
                        MPI_Request request = MPI_REQUEST_NULL;
                        EXPECT_EQ(
                            MPI_Irecv(
                                root_records.data() + local_elements,
                                static_cast<int>(local_elements),
                                MPI_FLOAT,
                                1,
                                7301,
                                MPI_COMM_WORLD,
                                &request),
                            MPI_SUCCESS);
                        EXPECT_EQ(
                            MPI_Wait(&request, MPI_STATUS_IGNORE),
                            MPI_SUCCESS);
                    }
                    else
                    {
                        MPI_Request request = MPI_REQUEST_NULL;
                        EXPECT_EQ(
                            MPI_Isend(
                                local.data(),
                                static_cast<int>(local_elements),
                                MPI_FLOAT,
                                0,
                                7301,
                                MPI_COMM_WORLD,
                                &request),
                            MPI_SUCCESS);
                        EXPECT_EQ(
                            MPI_Wait(&request, MPI_STATUS_IGNORE),
                            MPI_SUCCESS);
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                for (int repetition = 0; repetition < repetitions; ++repetition)
                {
                    size_t gathered = 0u;
                    const auto start = Clock::now();
                    if (native_shared_memory)
                    {
                        EXPECT_TRUE(shmem.gatherVariableFloatRecordsToRoot(
                            local.data(),
                            local_elements,
                            rank == 0 ? root_records.data() : nullptr,
                            total_elements,
                            /*record_width_elements=*/1u,
                            /*root_rank=*/0,
                            gathered));
                    }
                    else if (rank == 0)
                    {
                        std::memcpy(
                            root_records.data(),
                            local.data(),
                            local_elements * sizeof(float));
                        MPI_Request request = MPI_REQUEST_NULL;
                        EXPECT_EQ(
                            MPI_Irecv(
                                root_records.data() + local_elements,
                                static_cast<int>(local_elements),
                                MPI_FLOAT,
                                1,
                                7301,
                                MPI_COMM_WORLD,
                                &request),
                            MPI_SUCCESS);
                        EXPECT_EQ(
                            MPI_Wait(&request, MPI_STATUS_IGNORE),
                            MPI_SUCCESS);
                    }
                    else
                    {
                        MPI_Request request = MPI_REQUEST_NULL;
                        EXPECT_EQ(
                            MPI_Isend(
                                local.data(),
                                static_cast<int>(local_elements),
                                MPI_FLOAT,
                                0,
                                7301,
                                MPI_COMM_WORLD,
                                &request),
                            MPI_SUCCESS);
                        EXPECT_EQ(
                            MPI_Wait(&request, MPI_STATUS_IGNORE),
                            MPI_SUCCESS);
                    }
                    const auto finish = Clock::now();
                    local_samples[static_cast<size_t>(repetition)] =
                        std::chrono::duration<double, std::micro>(
                            finish - start)
                            .count();
                }
                return summarizeSlowestRank(local_samples);
            };

            printResult("gather", bytes, "shmem", run_gather(true));
            printResult("gather", bytes, "mpi", run_gather(false));

            std::vector<float> broadcast_payload(total_elements, 0.0f);
            if (rank == 0)
                std::iota(broadcast_payload.begin(), broadcast_payload.end(), 1.0f);
            const auto run_broadcast = [&](bool native_shared_memory) {
                std::vector<double> local_samples(
                    static_cast<size_t>(repetitions));
                for (int warmup = 0; warmup < 3; ++warmup)
                {
                    if (native_shared_memory)
                    {
                        EXPECT_TRUE(shmem.broadcast(
                            broadcast_payload.data(),
                            total_elements,
                            CollectiveDataType::FLOAT32,
                            0));
                    }
                    else
                    {
                        EXPECT_EQ(
                            MPI_Bcast(
                                broadcast_payload.data(),
                                static_cast<int>(total_elements),
                                MPI_FLOAT,
                                0,
                                MPI_COMM_WORLD),
                            MPI_SUCCESS);
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                for (int repetition = 0; repetition < repetitions; ++repetition)
                {
                    const auto start = Clock::now();
                    if (native_shared_memory)
                    {
                        EXPECT_TRUE(shmem.broadcast(
                            broadcast_payload.data(),
                            total_elements,
                            CollectiveDataType::FLOAT32,
                            0));
                    }
                    else
                    {
                        EXPECT_EQ(
                            MPI_Bcast(
                                broadcast_payload.data(),
                                static_cast<int>(total_elements),
                                MPI_FLOAT,
                                0,
                                MPI_COMM_WORLD),
                            MPI_SUCCESS);
                    }
                    const auto finish = Clock::now();
                    local_samples[static_cast<size_t>(repetition)] =
                        std::chrono::duration<double, std::micro>(
                            finish - start)
                            .count();
                }
                return summarizeSlowestRank(local_samples);
            };

            printResult("broadcast", bytes, "shmem", run_broadcast(true));
            printResult("broadcast", bytes, "mpi", run_broadcast(false));
        }

        shmem.shutdown();
        MPI_Barrier(MPI_COMM_WORLD);
    }
} // namespace llaminar2::test
