#include <gtest/gtest.h>
#include "../src/kernels/MPIAttentionKernel.h"
#include "../src/tensors/tensor_factory.h"
#include <memory>
#include <cmath>
#include <random>
#include <mpi.h>

using namespace llaminar;

class MPIAttentionKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // MPI initialization is handled by main function
        int flag;
        MPI_Initialized(&flag);
        if (!flag)
        {
            throw std::runtime_error("MPI should be initialized before running tests");
        }

        // Initialize attention kernels with test configuration
        const int n_head = 8;
        const int n_head_kv = 8; // Simplified: assume same as n_head for testing
        const int head_dim = 64;

        mpi_kernel = std::make_unique<MPIAttentionKernel>(n_head, n_head_kv, head_dim);

        // Initialize random generator with fixed seed for reproducibility
        generator.seed(42);
    }

    void TearDown() override
    {
        // MPI finalization is handled by main function
    }

    void fillRandomData(std::shared_ptr<TensorBase> &tensor, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (int i = 0; i < tensor->size(); ++i)
        {
            tensor->data()[i] = dist(generator);
        }
    }

    std::shared_ptr<TensorBase> createTensor(const std::vector<size_t> &shape)
    {
        // Convert size_t to int for TensorFactory
        std::vector<int> int_shape;
        int_shape.reserve(shape.size());
        for (const auto &dim : shape)
        {
            int_shape.push_back(static_cast<int>(dim));
        }

        auto tensor = llaminar::TensorFactory::create_simple(int_shape);
        return tensor;
    }

    std::shared_ptr<TensorBase> createTensor(const std::vector<size_t> &shape, const std::vector<float> &data)
    {
        // Convert size_t to int for TensorFactory
        std::vector<int> int_shape;
        int_shape.reserve(shape.size());
        for (const auto &dim : shape)
        {
            int_shape.push_back(static_cast<int>(dim));
        }

        auto tensor = llaminar::TensorFactory::create_simple(int_shape, data);
        return tensor;
    }

    void assertTensorNear(const TensorBase &actual, const TensorBase &expected, float tolerance = 1e-4f)
    {
        ASSERT_EQ(actual.shape(), expected.shape());
        ASSERT_EQ(actual.size(), expected.size());

        for (int i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(actual.data()[i], expected.data()[i], tolerance)
                << "Mismatch at index " << i << ": expected " << expected.data()[i]
                << ", got " << actual.data()[i];
        }
    }

    std::unique_ptr<MPIAttentionKernel> mpi_kernel;
    // Legacy non-MPI AttentionKernel removed.
    std::mt19937 generator;
};

TEST_F(MPIAttentionKernelTest, BasicFunctionality)
{
    // Test basic attention computation with simple configuration
    const size_t seq_len = 4;
    const size_t d_model = 512; // 8 heads * 64 head_dim
    const size_t n_head = 8;
    const size_t head_dim = 64;

    // Create input tensors
    auto input = createTensor({seq_len, d_model});
    auto wq = createTensor({d_model, n_head * head_dim});
    auto wk = createTensor({d_model, n_head * head_dim});
    auto wv = createTensor({d_model, n_head * head_dim});
    auto wo = createTensor({n_head * head_dim, d_model});
    auto k_cache = createTensor({seq_len, n_head * head_dim}); // Simplified cache
    auto v_cache = createTensor({seq_len, n_head * head_dim}); // Simplified cache
    auto output = createTensor({seq_len, d_model});

    // Fill with simple test data
    for (int i = 0; i < input->size(); ++i)
    {
        input->data()[i] = 0.01f * (i + 1);
    }
    fillRandomData(wq, -0.01f, 0.01f); // Scale down from -0.1, 0.1 to -0.01, 0.01
    fillRandomData(wk, -0.01f, 0.01f);
    fillRandomData(wv, -0.01f, 0.01f);
    fillRandomData(wo, -0.01f, 0.01f);

    std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    ASSERT_TRUE(mpi_kernel->execute(inputs, outputs));

    // Verify output is not zero and has reasonable values
    bool has_nonzero = false;
    bool has_reasonable_values = true;
    for (int i = 0; i < output->size(); ++i)
    {
        float val = output->data()[i];
        if (std::abs(val) > 1e-6f)
        {
            has_nonzero = true;
        }
        if (std::abs(val) > 100.0f)
        { // Check for exploding values
            has_reasonable_values = false;
        }
    }
    EXPECT_TRUE(has_nonzero);
    EXPECT_TRUE(has_reasonable_values);
}

TEST_F(MPIAttentionKernelTest, ValidationTests)
{
    // Test input validation
    const size_t seq_len = 2;
    const size_t d_model = 512;
    const size_t n_head = 8;
    const size_t head_dim = 64;

    auto input = createTensor({seq_len, d_model});
    auto wq = createTensor({d_model, n_head * head_dim});
    auto wk = createTensor({d_model, n_head * head_dim});
    auto wv = createTensor({d_model, n_head * head_dim});
    auto wo = createTensor({n_head * head_dim, d_model});
    auto k_cache = createTensor({seq_len, n_head * head_dim});
    auto v_cache = createTensor({seq_len, n_head * head_dim});
    auto output = createTensor({seq_len, d_model});

    // Test wrong number of inputs
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache}; // Missing v_cache
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        EXPECT_FALSE(mpi_kernel->validate(inputs, outputs));
    }

    // Test wrong number of outputs
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs = {}; // No outputs
        EXPECT_FALSE(mpi_kernel->validate(inputs, outputs));
    }

    // Test dimension mismatches
    {
        auto wrong_wq = createTensor({d_model, 100}); // Wrong dimension
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, wrong_wq, wk, wv, wo, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        EXPECT_FALSE(mpi_kernel->validate(inputs, outputs));
    }

    // Test correct validation
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        EXPECT_TRUE(mpi_kernel->validate(inputs, outputs));
    }
}

TEST_F(MPIAttentionKernelTest, HeadDistribution)
{
    // Test head distribution logic with different MPI sizes
    const int n_head = 8;
    const int n_head_kv = 8;
    const int head_dim = 64;

    MPIAttentionKernel kernel(n_head, n_head_kv, head_dim);

    // For this test, we'll verify the distribution logic by checking
    // that all heads are covered and no overlaps exist
    int total_heads_covered = 0;
    std::vector<bool> head_covered(n_head, false);

    // Get current MPI size
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int rank = 0; rank < size; ++rank)
    {
        auto [local_heads, head_offset] = kernel.getHeadDistribution(rank);

        // Verify no negative values
        EXPECT_GE(local_heads, 0);
        EXPECT_GE(head_offset, 0);

        // Verify bounds
        EXPECT_LE(head_offset + local_heads, n_head);

        // Mark covered heads
        for (int h = head_offset; h < head_offset + local_heads; ++h)
        {
            EXPECT_FALSE(head_covered[h]) << "Head " << h << " covered by multiple ranks";
            head_covered[h] = true;
        }

        total_heads_covered += local_heads;
    }

    // Verify all heads are covered exactly once
    EXPECT_EQ(total_heads_covered, n_head);
    for (int h = 0; h < n_head; ++h)
    {
        EXPECT_TRUE(head_covered[h]) << "Head " << h << " not covered by any rank";
    }
}

TEST_F(MPIAttentionKernelTest, DifferentSequenceLengths)
{
    // Test with various sequence lengths to ensure attention works correctly
    std::vector<size_t> test_seq_lens = {1, 2, 4, 8};
    const size_t d_model = 512;
    const size_t n_head = 8;
    const size_t head_dim = 64;

    for (size_t seq_len : test_seq_lens)
    {
        auto input = createTensor({seq_len, d_model});
        auto wq = createTensor({d_model, n_head * head_dim});
        auto wk = createTensor({d_model, n_head * head_dim});
        auto wv = createTensor({d_model, n_head * head_dim});
        auto wo = createTensor({n_head * head_dim, d_model});
        auto k_cache = createTensor({seq_len, n_head * head_dim});
        auto v_cache = createTensor({seq_len, n_head * head_dim});
        auto output = createTensor({seq_len, d_model});

        fillRandomData(input, -0.5f, 0.5f);
        fillRandomData(wq, -0.1f, 0.1f);
        fillRandomData(wk, -0.1f, 0.1f);
        fillRandomData(wv, -0.1f, 0.1f);
        fillRandomData(wo, -0.1f, 0.1f);

        std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        ASSERT_TRUE(mpi_kernel->execute(inputs, outputs))
            << "Failed for seq_len=" << seq_len;

        // Verify output has expected shape and non-zero values
        EXPECT_EQ(output->shape()[0], seq_len);
        EXPECT_EQ(output->shape()[1], d_model);

        bool has_nonzero = false;
        for (int i = 0; i < output->size(); ++i)
        {
            if (std::abs(output->data()[i]) > 1e-6f)
            {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Output is all zeros for seq_len=" << seq_len;
    }
}

TEST_F(MPIAttentionKernelTest, ConsistencyAcrossRuns)
{
    // Test that multiple runs with same input produce same output
    const size_t seq_len = 3;
    const size_t d_model = 512;
    const size_t n_head = 8;
    const size_t head_dim = 64;

    auto input = createTensor({seq_len, d_model});
    auto wq = createTensor({d_model, n_head * head_dim});
    auto wk = createTensor({d_model, n_head * head_dim});
    auto wv = createTensor({d_model, n_head * head_dim});
    auto wo = createTensor({n_head * head_dim, d_model});
    auto k_cache = createTensor({seq_len, n_head * head_dim});
    auto v_cache = createTensor({seq_len, n_head * head_dim});
    auto output1 = createTensor({seq_len, d_model});
    auto output2 = createTensor({seq_len, d_model});

    // Set deterministic input data
    std::fill(input->data(), input->data() + input->size(), 0.5f);
    std::fill(wq->data(), wq->data() + wq->size(), 0.1f);
    std::fill(wk->data(), wk->data() + wk->size(), 0.1f);
    std::fill(wv->data(), wv->data() + wv->size(), 0.1f);
    std::fill(wo->data(), wo->data() + wo->size(), 0.1f);

    std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};

    // Run 1
    std::vector<std::shared_ptr<TensorBase>> outputs1 = {output1};
    ASSERT_TRUE(mpi_kernel->execute(inputs, outputs1));

    // Run 2
    std::vector<std::shared_ptr<TensorBase>> outputs2 = {output2};
    ASSERT_TRUE(mpi_kernel->execute(inputs, outputs2));

    // Compare results - they should be identical
    assertTensorNear(*output1, *output2, 1e-6f);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI for testing
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}