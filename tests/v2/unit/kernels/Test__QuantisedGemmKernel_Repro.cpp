#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include "v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"

using namespace llaminar2;

// Mock Tensor for Asymmetric Quantization (simulating Q4_1 behavior)
class MockAsymmetricTensor : public TensorBase, public IINT8Unpackable
{
public:
    int rows_, cols_;
    std::vector<float> scales_;
    std::vector<float> mins_;
    std::vector<int8_t> quants_;

    MockAsymmetricTensor(int rows, int cols) : rows_(rows), cols_(cols)
    {
        int blocks_per_row = (cols + 31) / 32;
        scales_.resize(rows * blocks_per_row, 1.0f);
        mins_.resize(rows * blocks_per_row, 0.0f); // Default 0, will set to non-zero
        quants_.resize(rows * cols, 1);
    }

    // TensorBase implementation
    const std::vector<size_t> &shape() const override
    {
        static std::vector<size_t> s;
        s = {static_cast<size_t>(rows_), static_cast<size_t>(cols_)};
        return s;
    }
    TensorType native_type() const override { return TensorType::Q4_1; } // Pretend to be Q4_1
    int device_index() const override { return -1; }
    bool set_device(int) override { return false; }
    bool is_on_device(int) const override { return false; }
    const float *data() const override { return nullptr; } // Not needed for this test
    float *mutable_data() override { return nullptr; }
    bool copyFrom(const TensorBase *) override { return false; }
    std::unique_ptr<ITensorGemm> createGemm() override { return nullptr; }
    void to_fp32(float *) const override {}
    void to_bf16(uint16_t *) const override {}
    void to_fp16(uint16_t *) const override {}

    // Missing implementations
    void to_int8_blocked(int8_t *, float *, size_t) const override {}
    bool to_int8_perchannel(int8_t *, float *, float *) const override { return false; }
    void to_fp32_row(size_t, float *) const override {}
    void to_fp32_span(size_t, size_t, float *) const override {}
    std::shared_ptr<TensorBase> create_view(const std::vector<size_t> &, size_t) override { return nullptr; }

    // IINT8Unpackable implementation
    void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
    {
        size_t start_col = k_block_offset * 32;
        for (int i = 0; i < 32; ++i)
        {
            if (start_col + i < cols_)
            {
                output[i] = quants_[row_idx * cols_ + start_col + i];
            }
            else
            {
                output[i] = 0;
            }
        }
    }

    float get_block_scale(size_t row_idx, size_t k_block_offset) const override
    {
        size_t blocks_per_row = (cols_ + 31) / 32;
        return scales_[row_idx * blocks_per_row + k_block_offset];
    }

    // Helper to set min
    void set_block_min(size_t row_idx, size_t k_block_offset, float min_val)
    {
        size_t blocks_per_row = (cols_ + 31) / 32;
        mins_[row_idx * blocks_per_row + k_block_offset] = min_val;
    }

    float get_block_min(size_t row_idx, size_t k_block_offset) const
    {
        size_t blocks_per_row = (cols_ + 31) / 32;
        return mins_[row_idx * blocks_per_row + k_block_offset];
    }
};

TEST(Test__QuantisedGemmKernel_Repro, AsymmetricQuantizationError)
{
    int M = 1;
    int N = 1;
    int K = 32;

    MockAsymmetricTensor weights(N, K);
    // Set scale = 1.0, quant = 1, min = 10.0
    // Expected value = 1.0 * 1 + 10.0 = 11.0
    weights.set_block_min(0, 0, 10.0f);

    // Create kernel (packs weights)
    gemm_v4::QuantisedGemmKernel kernel(&weights);

    // Input A (all 1s)
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    // Run multiply
    MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);
    kernel.multiply(A.data(), C.data(), M, N, K, false, 1.0f, 0.0f, &mpi_ctx, -1);

    // Expected result: sum(A[i] * W[i]) = sum(1.0 * 11.0) = 32 * 11.0 = 352.0
    // Actual result (without min): sum(1.0 * 1.0) = 32.0

    // Tolerance increased to 0.1f due to FP16 quantization of scale factor
    EXPECT_NEAR(C[0], 352.0f, 0.1f) << "Kernel should account for min value in asymmetric quantization";
}

TEST(Test__QuantisedGemmKernel_Repro, TailTruncationBug)
{
    int M = 1;
    int N = 1;
    int K = 33; // Not multiple of 32

    MockAsymmetricTensor weights(N, K);
    // Set scale = 1.0, quant = 1, min = 0.0
    // Expected value = 1.0 * 1 + 0.0 = 1.0

    // Create kernel (packs weights)
    gemm_v4::QuantisedGemmKernel kernel(&weights);

    // Input A (all 1s)
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    // Run multiply
    MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);
    kernel.multiply(A.data(), C.data(), M, N, K, false, 1.0f, 0.0f, &mpi_ctx, -1);

    // Expected result: sum(A[i] * W[i]) = sum(1.0 * 1.0) = 33.0
    // Actual result (with truncation): sum(1.0 * 1.0) for first 32 elements = 32.0

    // Tolerance increased to 0.1f due to FP16 quantization of scale factor
    EXPECT_NEAR(C[0], 33.0f, 0.1f) << "Kernel should process all K columns, including tail";
}
