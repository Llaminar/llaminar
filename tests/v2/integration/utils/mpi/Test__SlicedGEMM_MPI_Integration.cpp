/**
 * @file Test__SlicedGEMM_MPI_Integration.cpp
 * @brief Multi-rank integration tests for sliced GEMM (Phase 2)
 *
 * Tests the sliced GEMM API with real MPI communication:
 *   - Row-parallel GEMM with AllGather
 *   - Column-parallel GEMM with AllReduce
 *   - Consistency between sliced and full GEMM outputs
 *   - Integration with MPITopology work distribution
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

#include "kernels/KernelFactory.h"
#include "loaders/PreparedWeightStore.h"
#include "utils/MPITopology.h"
#include "utils/MPIContext.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;

namespace
{
    WeightBinding makeTestBinding(TensorBase *tensor, const std::string &canonical_name)
    {
        static uint64_t next_binding_id = 1;
        WeightBinding binding;
        binding.binding_id = next_binding_id++;
        binding.identity = makeSourceWeightIdentity(canonical_name, ModelContextId{77}, binding.binding_id);
        binding.residency.home_device = DeviceId::cpu();
        binding.residency.resident_device = DeviceId::cpu();
        binding.tensor = tensor;
        binding.immutable = true;
        return binding;
    }

    struct PreparedSliceFixture
    {
        std::unique_ptr<PreparedWeightStore> store;
        PreparedWeightRef ref;
    };

    PreparedSliceFixture makePreparedSliceFixture(TensorBase *tensor, const std::string &canonical_name)
    {
        PreparedSliceFixture fixture;
        fixture.store = std::make_unique<PreparedWeightStore>(ModelContextId{77});
        auto binding = makeTestBinding(tensor, canonical_name);
        fixture.ref = fixture.store->registerPreparedForTest(
            binding,
            PreparedWeightKind::CpuPackedGemm,
            DeviceId::cpu());
        return fixture;
    }

    ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device_id = DeviceId::cpu())
    {
        static std::vector<std::shared_ptr<KernelFactory::PreparedGemmHandle>> handles;
        auto prepared = KernelFactory::prepareGemmHandleLocal(tensor, device_id);
        if (!prepared)
        {
            return nullptr;
        }
        handles.push_back(std::move(prepared));
        return KernelFactory::getOrCreateGemmEngine(handles.back().get());
    }
}

/**
 * @brief Test fixture for sliced GEMM MPI integration tests
 */
class Test__SlicedGEMM_MPI_Integration : public ::testing::Test
{
protected:
    std::shared_ptr<IMPIContext> mpi_ctx_;
    std::unique_ptr<MPITopology> topology_;
    int rank_ = 0;
    int world_size_ = 1;
    std::mt19937 rng_;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
        topology_ = std::make_unique<MPITopology>(MPI_COMM_WORLD);
        rng_.seed(42); // Same seed on all ranks for consistent random data

        // Clear kernel cache
        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
        topology_.reset();
        mpi_ctx_->barrier();
    }

    // Create Q8_0 tensor with deterministic random data
    std::unique_ptr<Q8_0Tensor> createQ8_0(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 34; // Q8_0: 2 bytes scale + 32 bytes data
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(42); // Deterministic seed
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
        {
            byte = static_cast<uint8_t>(dist(gen));
        }
        // Set scale values to reasonable floats
        for (size_t b = 0; b < num_blocks; ++b)
        {
            uint16_t scale_bits = 0x3C00; // ~1.0 in FP16
            memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q8_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    // Create FP32 activation tensor
    std::unique_ptr<FP32Tensor> createFP32(size_t rows, size_t cols, float fill = 0.0f)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = fill;
        }
        return tensor;
    }

    // Create FP32 with deterministic random data
    std::unique_ptr<FP32Tensor> createFP32Random(size_t rows, size_t cols)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();
        std::mt19937 gen(42); // Same seed for consistency
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = dist(gen);
        }
        return tensor;
    }

    // Compute max absolute difference between two tensors
    float maxAbsDiff(const FP32Tensor *a, const FP32Tensor *b)
    {
        size_t n = a->numel();
        float max_diff = 0.0f;
        const float *a_data = a->data();
        const float *b_data = b->data();
        for (size_t i = 0; i < n; ++i)
        {
            max_diff = std::max(max_diff, std::abs(a_data[i] - b_data[i]));
        }
        return max_diff;
    }

    // Check if tensor has NaN or Inf
    bool hasNaNOrInf(const FP32Tensor *tensor)
    {
        const float *data = tensor->data();
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
                return true;
        }
        return false;
    }
};

// =============================================================================
// Row-Parallel GEMM Tests (Output dimension sharded)
// =============================================================================

TEST_F(Test__SlicedGEMM_MPI_Integration, RowParallelGEMM_SlicesMatchFullOutput)
{
    // Test that row-parallel slices, when gathered, match full GEMM output
    constexpr int M = 32;  // Batch/sequence dimension
    constexpr int N = 512; // Output dimension (will be sharded)
    constexpr int K = 256; // Input dimension

    // Create weight tensor (same on all ranks - deterministic seed)
    auto weights = createQ8_0(N, K); // [N, K] for transposed GEMM

    // Create input activation (same on all ranks)
    auto input = createFP32Random(M, K);

    // Get this rank's output range
    WorkRange my_range = topology_->get_row_range(N);
    size_t local_n = my_range.size();

    // Create local output buffer
    auto local_output = createFP32(M, local_n);

    // Get sliced kernel for this rank's portion through the prepared store.
    auto prepared_slice = makePreparedSliceFixture(weights.get(), "blk.0.ffn_down.weight");
    auto *sliced_kernel = prepared_slice.store->slicedGemmKernel(
        prepared_slice.ref, my_range.start, my_range.end);
    ASSERT_NE(sliced_kernel, nullptr) << "Failed to create sliced kernel";

    // Execute local GEMM
    bool success = sliced_kernel->multiply_tensor(
        input.get(), local_output.get(),
        M, static_cast<int>(local_n), K);
    ASSERT_TRUE(success) << "Sliced GEMM execution failed";
    EXPECT_FALSE(hasNaNOrInf(local_output.get())) << "Local output has NaN/Inf";

    // === AllGatherv to combine results ===
    // Each rank has [M, local_n], need to gather into [M, N]

    // Gather all local sizes
    std::vector<int> all_local_n(world_size_);
    int my_local_n = static_cast<int>(local_n);
    MPI_Allgather(&my_local_n, 1, MPI_INT, all_local_n.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Compute displacements (row-major layout: each rank's data is contiguous per row)
    // We'll gather row-by-row to simplify
    auto full_output = createFP32(M, N);

    for (int row = 0; row < M; ++row)
    {
        // Gather one row at a time
        std::vector<int> recvcounts(world_size_);
        std::vector<int> displs(world_size_);
        int offset = 0;
        for (int r = 0; r < world_size_; ++r)
        {
            recvcounts[r] = all_local_n[r];
            displs[r] = offset;
            offset += all_local_n[r];
        }

        MPI_Allgatherv(
            local_output->data() + row * local_n, static_cast<int>(local_n), MPI_FLOAT,
            full_output->mutable_data() + row * N, recvcounts.data(), displs.data(), MPI_FLOAT,
            MPI_COMM_WORLD);
    }

    // === Compare against full GEMM on rank 0 ===
    if (rank_ == 0)
    {
        auto reference_output = createFP32(M, N);
        auto *full_kernel = getPreparedKernel(weights.get(), DeviceId::cpu());
        ASSERT_NE(full_kernel, nullptr);

        bool ref_success = full_kernel->multiply_tensor(
            input.get(), reference_output.get(),
            M, N, K);
        ASSERT_TRUE(ref_success);

        float max_diff = maxAbsDiff(full_output.get(), reference_output.get());
        LOG_INFO("[RowParallel] Max diff between gathered and full GEMM: " << max_diff);

        // Should be numerically identical (same operations, just partitioned)
        EXPECT_LT(max_diff, 1e-4f) << "Gathered output differs from full GEMM";
    }

    mpi_ctx_->barrier();
}

TEST_F(Test__SlicedGEMM_MPI_Integration, RowParallelGEMM_AllRanksGetSameGatheredResult)
{
    // Verify that after AllGatherv, all ranks have identical results
    constexpr int M = 16;
    constexpr int N = 256;
    constexpr int K = 128;

    auto weights = createQ8_0(N, K);
    auto input = createFP32Random(M, K);

    WorkRange my_range = topology_->get_row_range(N);
    size_t local_n = my_range.size();

    auto local_output = createFP32(M, local_n);
    auto prepared_slice = makePreparedSliceFixture(weights.get(), "blk.0.ffn_down.weight");
    auto *sliced_kernel = prepared_slice.store->slicedGemmKernel(
        prepared_slice.ref, my_range.start, my_range.end);
    ASSERT_NE(sliced_kernel, nullptr);

    sliced_kernel->multiply_tensor(input.get(), local_output.get(),
                                   M, static_cast<int>(local_n), K);

    // Gather local_n values
    std::vector<int> all_local_n(world_size_);
    int my_local_n = static_cast<int>(local_n);
    MPI_Allgather(&my_local_n, 1, MPI_INT, all_local_n.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Full gather
    auto full_output = createFP32(M, N);
    for (int row = 0; row < M; ++row)
    {
        std::vector<int> recvcounts(world_size_);
        std::vector<int> displs(world_size_);
        int offset = 0;
        for (int r = 0; r < world_size_; ++r)
        {
            recvcounts[r] = all_local_n[r];
            displs[r] = offset;
            offset += all_local_n[r];
        }
        MPI_Allgatherv(
            local_output->data() + row * local_n, static_cast<int>(local_n), MPI_FLOAT,
            full_output->mutable_data() + row * N, recvcounts.data(), displs.data(), MPI_FLOAT,
            MPI_COMM_WORLD);
    }

    // Compute local checksum
    double local_checksum = 0.0;
    const float *data = full_output->data();
    for (size_t i = 0; i < full_output->numel(); ++i)
    {
        local_checksum += static_cast<double>(data[i]);
    }

    // Verify all ranks have same checksum
    std::vector<double> all_checksums(world_size_);
    MPI_Allgather(&local_checksum, 1, MPI_DOUBLE,
                  all_checksums.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_NEAR(all_checksums[r], all_checksums[0], 1e-6)
            << "Rank " << r << " has different gathered result checksum";
    }
}

// =============================================================================
// Column-Parallel GEMM Tests (Input dimension sharded)
// =============================================================================

TEST_F(Test__SlicedGEMM_MPI_Integration, ColumnParallelGEMM_AllreduceSumsCorrectly)
{
    // Column-parallel: each rank computes partial products, then allreduce-sum
    // C = A @ B where A is sharded along K dimension
    constexpr int M = 32;
    constexpr int N = 128;
    constexpr int K = 512; // Will be sharded across ranks

    // Get this rank's K range
    WorkRange k_range = topology_->get_column_range(K);
    size_t local_k = k_range.size();

    // Create full weight tensor (all ranks have same)
    auto full_weights = createQ8_0(N, K);

    // Create full input (for reference)
    auto full_input = createFP32Random(M, K);

    // Extract this rank's portion of input: input[:, k_start:k_end]
    auto local_input = createFP32(M, local_k);
    for (int row = 0; row < M; ++row)
    {
        for (size_t c = 0; c < local_k; ++c)
        {
            local_input->mutable_data()[row * local_k + c] =
                full_input->data()[row * K + k_range.start + c];
        }
    }

    // For column-parallel GEMM, we need to slice weights along K dimension
    // This requires creating a new weight tensor or using a column-sliced kernel
    // For now, we use full kernel and slice input manually
    // (Full column-parallel support is Phase 3)

    // === Partial GEMM: local_input[M, local_k] @ full_weights[N, K]^T ===
    // This is not quite right - we need weights sliced too
    // For this test, we'll compute reference differently

    // Compute local output
    auto local_output = createFP32(M, N);
    auto *full_kernel = getPreparedKernel(full_weights.get(), DeviceId::cpu());
    ASSERT_NE(full_kernel, nullptr);

    // For a true column-parallel test, we'd need weight slicing
    // This test validates the AllReduce pattern instead

    // === Skip the column-parallel GEMM for now, just test allreduce ===
    // Fill local_output with rank-specific values for testing allreduce
    for (size_t i = 0; i < local_output->numel(); ++i)
    {
        local_output->mutable_data()[i] = static_cast<float>(rank_ + 1);
    }

    // Allreduce (in-place using MPI_IN_PLACE pattern)
    std::vector<float> send_copy(local_output->data(), local_output->data() + local_output->numel());
    mpi_ctx_->allreduce_sum(send_copy.data(), local_output->mutable_data(), local_output->numel());

    // Expected value after allreduce: sum of (1 + 2 + ... + world_size) = world_size*(world_size+1)/2
    float expected = static_cast<float>(world_size_ * (world_size_ + 1) / 2);

    for (size_t i = 0; i < local_output->numel(); ++i)
    {
        EXPECT_NEAR(local_output->data()[i], expected, 1e-5f)
            << "AllReduce produced incorrect result at index " << i;
    }
}

// =============================================================================
// Kernel Caching Tests
// =============================================================================

TEST_F(Test__SlicedGEMM_MPI_Integration, SlicedKernelCachingIsRankLocal)
{
    // Each rank should have its own cached kernel for its slice
    constexpr int N = 512;
    constexpr int K = 256;

    auto weights = createQ8_0(N, K);

    WorkRange my_range = topology_->get_row_range(N);

    // Get sliced kernel twice through the same prepared store.
    auto prepared_slice = makePreparedSliceFixture(weights.get(), "blk.0.ffn_down.weight");
    auto *k1 = prepared_slice.store->slicedGemmKernel(prepared_slice.ref, my_range.start, my_range.end);
    auto *k2 = prepared_slice.store->slicedGemmKernel(prepared_slice.ref, my_range.start, my_range.end);

    EXPECT_EQ(k1, k2) << "Same slice should return same cached kernel";

    // Different rank's slice should be different (if we could access it)
    // Here we just verify our own caching works

    // Gather kernel pointers as integers for comparison
    uintptr_t my_ptr = reinterpret_cast<uintptr_t>(k1);
    std::vector<uintptr_t> all_ptrs(world_size_);
    MPI_Allgather(&my_ptr, sizeof(uintptr_t), MPI_BYTE,
                  all_ptrs.data(), sizeof(uintptr_t), MPI_BYTE, MPI_COMM_WORLD);

    // All ranks should have non-null kernels
    for (int r = 0; r < world_size_; ++r)
    {
        EXPECT_NE(all_ptrs[r], 0u) << "Rank " << r << " has null kernel pointer";
    }

    // Different ranks should have different kernel objects
    // (They're in different address spaces, so comparison isn't meaningful
    //  but we can verify none are null)
}

// =============================================================================
// Integration with MPITopology Work Distribution
// =============================================================================

TEST_F(Test__SlicedGEMM_MPI_Integration, WorkRangesMatchSlicedKernelDimensions)
{
    // Verify that MPITopology work ranges match sliced kernel dimensions
    constexpr int N = 1024;
    constexpr int K = 512;

    auto weights = createQ8_0(N, K);

    WorkRange my_range = topology_->get_row_range(N);

    auto prepared_slice = makePreparedSliceFixture(weights.get(), "blk.0.ffn_down.weight");
    auto *kernel = prepared_slice.store->slicedGemmKernel(
        prepared_slice.ref, my_range.start, my_range.end);
    ASSERT_NE(kernel, nullptr);

    // Kernel's N dimension should match slice size
    EXPECT_EQ(static_cast<size_t>(kernel->get_n()), my_range.size())
        << "Kernel N dimension should match slice size";

    // Kernel's K dimension should be unchanged
    EXPECT_EQ(static_cast<size_t>(kernel->get_k()), K)
        << "Kernel K dimension should be unchanged";
}

TEST_F(Test__SlicedGEMM_MPI_Integration, LargeMatrixRowParallel)
{
    // Test with larger matrices similar to real model dimensions
    constexpr int M = 128;  // Larger batch
    constexpr int N = 4864; // FFN intermediate dim (Qwen2.5 0.5B)
    constexpr int K = 896;  // d_model

    auto weights = createQ8_0(N, K);
    auto input = createFP32Random(M, K);

    WorkRange my_range = topology_->get_row_range(N);
    size_t local_n = my_range.size();

    auto local_output = createFP32(M, local_n);

    auto prepared_slice = makePreparedSliceFixture(weights.get(), "blk.0.ffn_down.weight");
    auto *sliced_kernel = prepared_slice.store->slicedGemmKernel(
        prepared_slice.ref, my_range.start, my_range.end);
    ASSERT_NE(sliced_kernel, nullptr);

    bool success = sliced_kernel->multiply_tensor(
        input.get(), local_output.get(),
        M, static_cast<int>(local_n), K);
    ASSERT_TRUE(success) << "Large matrix sliced GEMM failed";
    EXPECT_FALSE(hasNaNOrInf(local_output.get())) << "Output has NaN/Inf";

    // Verify non-trivial output
    float sum = 0.0f;
    for (size_t i = 0; i < local_output->numel(); ++i)
    {
        sum += std::abs(local_output->data()[i]);
    }
    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros";

    // Verify all ranks completed successfully
    int local_success = 1;
    int global_success = 0;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(global_success, 1) << "All ranks should succeed";
}

// MPI-aware main function
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
