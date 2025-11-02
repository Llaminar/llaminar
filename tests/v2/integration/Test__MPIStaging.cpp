/**
 * @file Test__MPIStaging.cpp
 * @brief Integration tests for MPIStager utility
 *
 * Tests the MPI host staging infrastructure for GPU↔Host transfers.
 * Since GPU backends are currently disabled, we test:
 * 1. CPU tensor staging (no-op path)
 * 2. API correctness (buffer sizes, error handling)
 * 3. MPI integration (staging around Allreduce/Allgather)
 *
 * Future: Add GPU tensor tests when CUDA/ROCm backends enabled.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>

#include "utils/MPIStager.h"
#include "utils/MPIContext.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"

using namespace llaminar2;

class MPIStaging : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI context
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        rank_ = rank;
        world_size_ = world_size;
        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size);

        // Create tensor factory
        tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int rank_;
    int world_size_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> tensor_factory_;
};

// ============================================================================
// Test 1: CPU Tensor Staging (No-Op Path)
// ============================================================================

TEST_F(MPIStaging, CPUTensorNoOp)
{
    // Create CPU tensor (device_index = -1 by default in V2)
    std::vector<size_t> shape = {4, 8}; // 32 elements
    auto tensor = tensor_factory_->createFP32(shape);

    // Verify it's a CPU tensor
    EXPECT_EQ(tensor->device_index(), -1) << "Tensor should be on CPU (device -1)";

    // Fill with test data
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 32; ++i)
    {
        data[i] = static_cast<float>(rank_ * 100 + i);
    }

    // Stage to host (should be no-op for CPU tensor)
    auto host_buffer = MPIStager::toHost(tensor.get());

    // Verify buffer size
    EXPECT_EQ(host_buffer.size(), 32) << "Host buffer size mismatch";

    // Verify data matches (no-op should preserve values)
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(host_buffer[i], data[i])
            << "Data mismatch at index " << i << " on rank " << rank_;
    }

    // Modify host buffer
    for (auto &val : host_buffer)
    {
        val *= 2.0f;
    }

    // Stage back to device (should be no-op for CPU tensor)
    MPIStager::toDevice(host_buffer, tensor.get());

    // Verify data updated
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(rank_ * 100 + i) * 2.0f)
            << "Data not updated at index " << i << " on rank " << rank_;
    }
}

// ============================================================================
// Test 2: CPU Tensor with MPI Allreduce
// ============================================================================

TEST_F(MPIStaging, CPUTensorWithAllreduce)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Create tensor with rank-specific values
    std::vector<size_t> shape = {16}; // 16 elements
    auto tensor = tensor_factory_->createFP32(shape);

    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 16; ++i)
    {
        data[i] = static_cast<float>(rank_ + 1); // Rank 0→1.0, Rank 1→2.0
    }

    // Stage to host
    auto host_buffer = MPIStager::toHost(tensor.get());

    // MPI Allreduce (sum across ranks)
    std::vector<float> result(16);
    MPI_Allreduce(host_buffer.data(), result.data(), 16, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Stage result back
    MPIStager::toDevice(result, tensor.get());

    // Verify sum (rank 0 + rank 1 = 1 + 2 = 3 for 2 ranks)
    float expected_sum = 0.0f;
    for (int r = 0; r < world_size_; ++r)
    {
        expected_sum += static_cast<float>(r + 1);
    }

    for (size_t i = 0; i < 16; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], expected_sum)
            << "Allreduce result incorrect at index " << i << " on rank " << rank_;
    }
}

// ============================================================================
// Test 3: Error Handling - Buffer Size Mismatch
// ============================================================================

TEST_F(MPIStaging, BufferSizeMismatch)
{
    // Create tensor
    std::vector<size_t> shape = {8};
    auto tensor = tensor_factory_->createFP32(shape);

    // Create mismatched buffer (wrong size)
    std::vector<float> wrong_size_buffer(16, 0.0f); // 16 instead of 8

    // Expect exception when staging wrong-sized buffer
    EXPECT_THROW(
        {
            MPIStager::toDevice(wrong_size_buffer, tensor.get());
        },
        std::invalid_argument)
        << "Should throw on buffer size mismatch";
}

// ============================================================================
// Test 4: Error Handling - Null Tensor
// ============================================================================

TEST_F(MPIStaging, NullTensorHandling)
{
    // Null tensor should throw
    EXPECT_THROW(
        {
            MPIStager::toHost(nullptr);
        },
        std::invalid_argument)
        << "Should throw on null tensor in toHost()";

    std::vector<float> buffer(8, 0.0f);
    EXPECT_THROW(
        {
            MPIStager::toDevice(buffer, nullptr);
        },
        std::invalid_argument)
        << "Should throw on null tensor in toDevice()";
}

// ============================================================================
// Test 5: RequiresStaging Helper
// ============================================================================

TEST_F(MPIStaging, RequiresStagingHelper)
{
    // CPU tensor should NOT require staging
    auto cpu_tensor = tensor_factory_->createFP32({4, 4});
    EXPECT_FALSE(MPIStager::requiresStaging(cpu_tensor.get()))
        << "CPU tensor should not require staging";

    // Verify it's actually a CPU tensor
    EXPECT_EQ(cpu_tensor->device_index(), -1)
        << "TensorFactory should create CPU tensors by default";

    // Null tensor should return false
    EXPECT_FALSE(MPIStager::requiresStaging(nullptr))
        << "Null tensor should return false";

    // Future: GPU tensor should require staging (when backends enabled)
    // auto gpu_tensor = tensor_factory_->createFP32({4, 4}, 0);  // device 0
    // EXPECT_TRUE(MPIStager::requiresStaging(gpu_tensor.get()))
    //     << "GPU tensor should require staging";
}

// ============================================================================
// Test 6: Large Tensor Staging (Performance Smoke Test)
// ============================================================================

TEST_F(MPIStaging, LargeTensorStaging)
{
    // Create large tensor (1MB = 262144 floats)
    std::vector<size_t> shape = {512, 512}; // 262144 elements
    auto tensor = tensor_factory_->createFP32(shape);

    // Fill with sequential data
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 262144; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Measure staging overhead
    auto start = std::chrono::high_resolution_clock::now();

    auto host_buffer = MPIStager::toHost(tensor.get());

    auto mid = std::chrono::high_resolution_clock::now();

    MPIStager::toDevice(host_buffer, tensor.get());

    auto end = std::chrono::high_resolution_clock::now();

    // Verify data integrity (spot check)
    for (size_t i = 0; i < 262144; i += 1000)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i))
            << "Data corruption at index " << i;
    }

    // Report timing (informational, not a hard requirement)
    auto to_host_us = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
    auto to_device_us = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();

    if (rank_ == 0)
    {
        std::cout << "[MPIStaging] Large tensor (1MB) staging times:\n";
        std::cout << "  toHost:   " << to_host_us << " μs\n";
        std::cout << "  toDevice: " << to_device_us << " μs\n";
        std::cout << "  Total:    " << (to_host_us + to_device_us) << " μs\n";
    }
}

// ============================================================================
// Test 7: MPI Allgather with Staging
// ============================================================================

TEST_F(MPIStaging, AllgatherWithStaging)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Each rank has a small tensor with rank-specific value
    std::vector<size_t> shape = {4}; // 4 elements per rank
    auto tensor = tensor_factory_->createFP32(shape);

    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 4; ++i)
    {
        data[i] = static_cast<float>(rank_ * 10 + i); // Rank 0: [0,1,2,3], Rank 1: [10,11,12,13]
    }

    // Stage to host
    auto host_buffer = MPIStager::toHost(tensor.get());

    // Allgather (collect all ranks' data)
    std::vector<float> gathered(4 * world_size_);
    MPI_Allgather(host_buffer.data(), 4, MPI_FLOAT,
                  gathered.data(), 4, MPI_FLOAT, MPI_COMM_WORLD);

    // Verify gathered data
    for (int r = 0; r < world_size_; ++r)
    {
        for (int i = 0; i < 4; ++i)
        {
            float expected = static_cast<float>(r * 10 + i);
            EXPECT_FLOAT_EQ(gathered[r * 4 + i], expected)
                << "Allgather data mismatch at rank " << r << " index " << i;
        }
    }
}

// ============================================================================
// Test 8: 2D Tensor Staging (Typical Activation Shape)
// ============================================================================

TEST_F(MPIStaging, TwoDimensionalTensor)
{
    // Typical activation shape [seq_len, d_model]
    std::vector<size_t> shape = {8, 896}; // 7168 elements
    auto tensor = tensor_factory_->createFP32(shape);

    // Fill with row/col pattern
    float *data = tensor->mutable_data();
    for (size_t row = 0; row < 8; ++row)
    {
        for (size_t col = 0; col < 896; ++col)
        {
            data[row * 896 + col] = static_cast<float>(row * 1000 + col);
        }
    }

    // Stage round-trip
    auto host_buffer = MPIStager::toHost(tensor.get());
    EXPECT_EQ(host_buffer.size(), 7168) << "Buffer size mismatch for 2D tensor";

    // Modify and stage back
    for (auto &val : host_buffer)
    {
        val += 1.0f;
    }
    MPIStager::toDevice(host_buffer, tensor.get());

    // Verify modification
    for (size_t row = 0; row < 8; ++row)
    {
        for (size_t col = 0; col < 896; ++col)
        {
            float expected = static_cast<float>(row * 1000 + col) + 1.0f;
            EXPECT_FLOAT_EQ(data[row * 896 + col], expected)
                << "Mismatch at [" << row << "," << col << "]";
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
