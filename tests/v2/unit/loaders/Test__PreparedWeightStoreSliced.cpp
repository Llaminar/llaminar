/**
 * @file Test__PreparedWeightStoreSliced.cpp
 * @brief Unit tests for PreparedWeightStore ref-based sliced GEMM resolution.
 */

#include <gtest/gtest.h>

#include "loaders/PreparedWeightStore.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "tensors/Tensors.h"
#include "../../utils/PreparedWeightTestHarness.h"

#include <cstring>
#include <memory>
#include <random>

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

class PreparedWeightStoreSlicedTest : public ::testing::Test
{
protected:
    std::unique_ptr<Q8_0Tensor> createQ8_0(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 34;
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
            byte = static_cast<uint8_t>(dist(gen));

        for (size_t block = 0; block < num_blocks; ++block)
        {
            uint16_t scale_bits = 0x3C00;
            std::memcpy(&raw_data[block * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q8_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    std::unique_ptr<Q4_0Tensor> createQ4_0(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 18;
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
            byte = static_cast<uint8_t>(dist(gen));

        for (size_t block = 0; block < num_blocks; ++block)
        {
            uint16_t scale_bits = 0x3C00;
            std::memcpy(&raw_data[block * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q4_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    std::unique_ptr<FP32Tensor> createFP32(size_t rows, size_t cols, float fill_value = 0.0f)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < rows * cols; ++i)
            data[i] = fill_value;
        return tensor;
    }

    std::unique_ptr<FP32Tensor> createFP32Random(size_t rows, size_t cols)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < rows * cols; ++i)
            data[i] = dist(gen);
        return tensor;
    }

    PreparedGemmFixture prepare(TensorBase *tensor)
    {
        return makePreparedGemmFixture(
            tensor,
            DeviceId::cpu(),
            "blk.0.ffn_down.weight");
    }
};

TEST_F(PreparedWeightStoreSlicedTest, CreatesSlicedKernel)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());

    auto *kernel = fixture.store->slicedGemmKernel(fixture.ref, 0, 512);
    ASSERT_NE(kernel, nullptr);

    auto *qgemm = dynamic_cast<CPUNativeVNNIGemmKernel *>(kernel);
    ASSERT_NE(qgemm, nullptr);
    EXPECT_EQ(qgemm->get_n(), 512);
    EXPECT_EQ(qgemm->get_k(), 896);
}

TEST_F(PreparedWeightStoreSlicedTest, SlicedKernelSecondHalf)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());

    auto *kernel = fixture.store->slicedGemmKernel(fixture.ref, 512, 1024);
    ASSERT_NE(kernel, nullptr);

    auto *qgemm = dynamic_cast<CPUNativeVNNIGemmKernel *>(kernel);
    ASSERT_NE(qgemm, nullptr);
    EXPECT_EQ(qgemm->get_n(), 512);
    EXPECT_EQ(qgemm->get_k(), 896);
}

TEST_F(PreparedWeightStoreSlicedTest, CachesSlicedKernelsWithinStore)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());

    auto *first = fixture.store->slicedGemmKernel(fixture.ref, 0, 512);
    auto *again = fixture.store->slicedGemmKernel(fixture.ref, 0, 512);

    EXPECT_EQ(first, again);
}

TEST_F(PreparedWeightStoreSlicedTest, DifferentSlicesGetDifferentKernels)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());

    auto *first_half = fixture.store->slicedGemmKernel(fixture.ref, 0, 512);
    auto *second_half = fixture.store->slicedGemmKernel(fixture.ref, 512, 1024);

    EXPECT_NE(first_half, second_half);
}

TEST_F(PreparedWeightStoreSlicedTest, FullKernelDifferentFromSliced)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());

    auto *full = fixture.store->gemmKernel(fixture.ref);
    auto *sliced = fixture.store->slicedGemmKernel(fixture.ref, 0, 1024);

    ASSERT_NE(full, nullptr);
    ASSERT_NE(sliced, nullptr);
    EXPECT_NE(full, sliced);

    auto *full_qgemm = dynamic_cast<CPUNativeVNNIGemmKernel *>(full);
    auto *sliced_qgemm = dynamic_cast<CPUNativeVNNIGemmKernel *>(sliced);
    ASSERT_NE(full_qgemm, nullptr);
    ASSERT_NE(sliced_qgemm, nullptr);
    EXPECT_EQ(full_qgemm->get_n(), sliced_qgemm->get_n());
    EXPECT_EQ(full_qgemm->get_k(), sliced_qgemm->get_k());
}

TEST_F(PreparedWeightStoreSlicedTest, ReleaseAllPreparedStateClearsSlicedEntries)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());

    auto *kernel = fixture.store->slicedGemmKernel(fixture.ref, 0, 512);
    ASSERT_NE(kernel, nullptr);

    fixture.store->releaseAllPreparedState();
    EXPECT_EQ(fixture.store->slicedGemmKernel(fixture.ref, 0, 512), nullptr);
}

TEST_F(PreparedWeightStoreSlicedTest, MissingPreparedRefReturnsNull)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());
    auto missing_ref = fixture.ref;
    missing_ref.binding_id += 1000000;

    EXPECT_EQ(fixture.store->slicedGemmKernel(missing_ref, 0, 512), nullptr);
}

TEST_F(PreparedWeightStoreSlicedTest, InvalidRangeThrows)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());

    EXPECT_THROW(
        fixture.store->slicedGemmKernel(fixture.ref, 512, 512),
        std::runtime_error);
    EXPECT_THROW(
        fixture.store->slicedGemmKernel(fixture.ref, 600, 500),
        std::runtime_error);
}

TEST_F(PreparedWeightStoreSlicedTest, RangeExceedsTensorThrows)
{
    auto weights = createQ8_0(1024, 896);
    auto fixture = prepare(weights.get());

    EXPECT_THROW(
        fixture.store->slicedGemmKernel(fixture.ref, 0, 2048),
        std::runtime_error);
}

TEST_F(PreparedWeightStoreSlicedTest, SlicedGemmProducesCorrectOutput)
{
    const size_t N = 64;
    const size_t K = 32;
    const size_t M = 1;

    auto weights = createQ8_0(N, K);
    auto fixture = prepare(weights.get());
    auto input = createFP32Random(M, K);
    auto output_full = createFP32(M, N, 0.0f);
    auto output_sliced1 = createFP32(M, N / 2, 0.0f);
    auto output_sliced2 = createFP32(M, N / 2, 0.0f);

    auto *full_kernel = fixture.store->gemmKernel(fixture.ref);
    ASSERT_NE(full_kernel, nullptr);
    ASSERT_TRUE(full_kernel->multiply_tensor(
        input.get(), output_full.get(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));

    auto *sliced1 = fixture.store->slicedGemmKernel(fixture.ref, 0, N / 2);
    auto *sliced2 = fixture.store->slicedGemmKernel(fixture.ref, N / 2, N);
    ASSERT_NE(sliced1, nullptr);
    ASSERT_NE(sliced2, nullptr);

    ASSERT_TRUE(sliced1->multiply_tensor(
        input.get(), output_sliced1.get(),
        static_cast<int>(M), static_cast<int>(N / 2), static_cast<int>(K)));
    ASSERT_TRUE(sliced2->multiply_tensor(
        input.get(), output_sliced2.get(),
        static_cast<int>(M), static_cast<int>(N / 2), static_cast<int>(K)));

    const float *full_data = output_full->data();
    const float *slice1_data = output_sliced1->data();
    const float *slice2_data = output_sliced2->data();

    for (size_t m = 0; m < M; ++m)
    {
        for (size_t n = 0; n < N / 2; ++n)
        {
            EXPECT_NEAR(full_data[m * N + n], slice1_data[m * (N / 2) + n], 1e-3f)
                << "Mismatch at row " << m << ", col " << n << " (first half)";
        }
        for (size_t n = 0; n < N / 2; ++n)
        {
            EXPECT_NEAR(full_data[m * N + N / 2 + n], slice2_data[m * (N / 2) + n], 1e-3f)
                << "Mismatch at row " << m << ", col " << (N / 2 + n) << " (second half)";
        }
    }
}

TEST_F(PreparedWeightStoreSlicedTest, WorksWithQ4_0)
{
    auto weights = createQ4_0(1024, 896);
    auto fixture = prepare(weights.get());

    auto *kernel = fixture.store->slicedGemmKernel(fixture.ref, 0, 512);
    ASSERT_NE(kernel, nullptr);

    auto *qgemm = dynamic_cast<CPUNativeVNNIGemmKernel *>(kernel);
    ASSERT_NE(qgemm, nullptr);
    EXPECT_EQ(qgemm->get_n(), 512);
}
