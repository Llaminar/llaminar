#include <gtest/gtest.h>
#include "TestTimeoutGuard.h"
#include "../src/operators/MPIRMSNormOperator.h"
#include "../src/tensors/TensorFactory.h"
#include <memory>
#include <chrono>
#include <cmath>
#include <random>
#include "TestMpiUtils.h"

using namespace llaminar;

class MPIRMSNormOperatorTest : public ::testing::Test
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

        mpi_kernel = std::make_unique<MPIRMSNormOperator>();

        // Set the same epsilon for both kernels
        float epsilon = 1e-6f;
        mpi_kernel->setEpsilon(epsilon);

        // Initialize random generator with fixed seed for reproducibility
        generator.seed(42);
    }

    void TearDown() override
    {
        // MPI finalization is handled by MPIOperatorBase destructor
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
        auto tensor = llaminar::TensorFactory::create_simple(std::vector<int>(shape.begin(), shape.end()));
        return tensor;
    }

    std::shared_ptr<TensorBase> createTensor(const std::vector<size_t> &shape, const std::vector<float> &data)
    {
        auto tensor = llaminar::TensorFactory::create_simple(std::vector<int>(shape.begin(), shape.end()), data);
        return tensor;
    }

    void assertTensorNear(const TensorBase &actual, const TensorBase &expected, float tolerance = 1e-5f)
    {
        ASSERT_EQ(actual.shape().size(), expected.shape().size());
        for (size_t i = 0; i < actual.shape().size(); ++i)
        {
            ASSERT_EQ(actual.shape()[i], expected.shape()[i]);
        }

        auto actual_data = actual.data();
        auto expected_data = expected.data();
        size_t total_size = 1;
        for (int dim : actual.shape())
        {
            total_size *= dim;
        }

        for (size_t i = 0; i < total_size; ++i)
        {
            EXPECT_NEAR(actual_data[i], expected_data[i], tolerance)
                << "Mismatch at index " << i << ": expected " << expected_data[i]
                << ", got " << actual_data[i];
        }
    }

    std::unique_ptr<MPIRMSNormOperator> mpi_kernel;
    // Legacy non-MPI RMSNormKernel removed.
    std::mt19937 generator;
};

TEST_F(MPIRMSNormOperatorTest, BasicFunctionality)
{
    // Test basic RMS normalization with simple data
    const size_t seq_len = 4;
    const size_t hidden_size = 6;

    // Create test tensors
    auto input = createTensor({seq_len, hidden_size});
    auto weight = createTensor({hidden_size});
    auto output = createTensor({seq_len, hidden_size});

    // Fill with test data
    for (int i = 0; i < input->size(); ++i)
    {
        input->data()[i] = static_cast<float>(i + 1); // 1, 2, 3, 4, 5, 6, 7, 8, ...
    }
    std::fill(weight->data(), weight->data() + weight->size(), 1.0f); // All weights = 1

    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    ASSERT_TRUE(mpi_kernel->execute(inputs, outputs));

    // Verify output is not zero and has reasonable values
    bool has_nonzero = false;
    for (int i = 0; i < output->size(); ++i)
    {
        if (std::abs(output->data()[i]) > 1e-6f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(MPIRMSNormOperatorTest, ValidationTests)
{
    // Test input validation
    auto input = createTensor({2, 4});
    auto weight = createTensor({4});
    auto output = createTensor({2, 4});

    // Test wrong number of inputs
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {input}; // Missing weight
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        EXPECT_FALSE(mpi_kernel->validate(inputs, outputs));
    }

    // Test wrong number of outputs
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {}; // No outputs
        EXPECT_FALSE(mpi_kernel->validate(inputs, outputs));
    }

    // Test dimension mismatches
    {
        auto wrong_weight = createTensor({3}); // Wrong hidden size
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, wrong_weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        EXPECT_FALSE(mpi_kernel->validate(inputs, outputs));
    }

    // Test correct validation
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        EXPECT_TRUE(mpi_kernel->validate(inputs, outputs));
    }
}

TEST_F(MPIRMSNormOperatorTest, DifferentSequenceLengths)
{
    // Test with various sequence lengths to ensure distribution works correctly
    std::vector<size_t> test_seq_lens = {1, 3, 7, 16, 32};
    const size_t hidden_size = 8;

    for (size_t seq_len : test_seq_lens)
    {
        auto input = createTensor({seq_len, hidden_size});
        auto weight = createTensor({hidden_size});
        auto output = createTensor({seq_len, hidden_size});

        fillRandomData(input);
        fillRandomData(weight, 0.5f, 1.5f);

        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        ASSERT_TRUE(mpi_kernel->execute(inputs, outputs))
            << "Failed for seq_len=" << seq_len;

        // Verify output has expected shape and non-zero values
        EXPECT_EQ(output->shape()[0], seq_len);
        EXPECT_EQ(output->shape()[1], hidden_size);

        bool has_nonzero = false;
        const auto *output_data = output->data();
        size_t output_size = output->shape()[0] * output->shape()[1];
        for (size_t i = 0; i < output_size; ++i)
        {
            if (std::abs(output_data[i]) > 1e-6f)
            {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Output is all zeros for seq_len=" << seq_len;
    }
}

TEST_F(MPIRMSNormOperatorTest, EpsilonConfiguration)
{
    // Test that epsilon configuration works correctly
    const size_t seq_len = 4;
    const size_t hidden_size = 4;

    auto input = createTensor({seq_len, hidden_size});
    auto weight = createTensor({hidden_size});
    auto output1 = createTensor({seq_len, hidden_size});
    auto output2 = createTensor({seq_len, hidden_size});

    // Fill with very small values to test epsilon effect
    auto input_data = const_cast<float *>(input->data());
    auto weight_data = const_cast<float *>(weight->data());
    auto input_size = input->shape()[0] * input->shape()[1];
    auto weight_size = weight->shape()[0];
    std::fill(input_data, input_data + input_size, 1e-8f);
    std::fill(weight_data, weight_data + weight_size, 1.0f);

    // Test with different epsilon values
    mpi_kernel->setEpsilon(1e-6f);
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs1 = {output1};
    ASSERT_TRUE(mpi_kernel->execute(inputs, outputs1));

    mpi_kernel->setEpsilon(1e-4f);
    std::vector<std::shared_ptr<TensorBase>> outputs2 = {output2};
    ASSERT_TRUE(mpi_kernel->execute(inputs, outputs2));

    // Results should be different due to different epsilon values
    bool results_different = false;
    auto output1_data = output1->data();
    auto output2_data = output2->data();
    auto output_size = output1->shape()[0] * output1->shape()[1];
    for (size_t i = 0; i < output_size; ++i)
    {
        if (std::abs(output1_data[i] - output2_data[i]) > 1e-6f)
        {
            results_different = true;
            break;
        }
    }
    EXPECT_TRUE(results_different);
}

LLAMINAR_DEFINE_GTEST_MPI_MAIN();