/**
 * @file test_operator_batch_interfaces.cpp
 * @brief Interface validation tests for batch-aware operators
 * @author David Sanftenberg
 * @date October 15, 2025
 * 
 * Test Coverage:
 * - Dimension detection (2D vs 3D tensors)
 * - Backward compatibility (batch=1 case)
 * - Shape validation
 * - Batch dimension propagation
 * - Error handling for invalid shapes
 */

#include <gtest/gtest.h>
#include "../src/operators/MPIEmbeddingOperator.h"
#include "../src/operators/MPILinearOperator.h"
#include "../src/operators/MPILinearBatchOperator.h"
#include "../src/operators/MPIRMSNormOperator.h"
#include "../src/operators/MPISwiGLUBatchOperator.h"
#include "../src/operators/MPIAttentionOperator.h"
#include "../src/tensors/SimpleTensor.h"
#include "../src/tensors/BatchedKVCache.h"
#include <mpi.h>

using namespace llaminar;

class OperatorBatchInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        
        // Standard test dimensions
        batch_size_ = 4;
        seq_len_ = 16;
        hidden_dim_ = 512;
        output_dim_ = 1024;
        vocab_size_ = 32000;
        num_heads_ = 8;
        head_dim_ = hidden_dim_ / num_heads_;
    }
    
    // Helper: Create 2D tensor [seq, hidden]
    std::shared_ptr<SimpleTensor> create2DTensor(int seq, int hidden, float value = 0.0f) {
        auto tensor = std::make_shared<SimpleTensor>(std::vector<int>{seq, hidden});
        std::fill(tensor->data(), tensor->data() + seq * hidden, value);
        return tensor;
    }
    
    // Helper: Create 3D tensor [batch, seq, hidden]
    std::shared_ptr<SimpleTensor> create3DTensor(int batch, int seq, int hidden, float value = 0.0f) {
        auto tensor = std::make_shared<SimpleTensor>(std::vector<int>{batch, seq, hidden});
        std::fill(tensor->data(), tensor->data() + batch * seq * hidden, value);
        return tensor;
    }
    
    // Helper: Create token tensor 1D [seq] or 2D [batch, seq]
    std::shared_ptr<SimpleTensor> createTokenTensor(int seq, int batch = 1) {
        std::shared_ptr<SimpleTensor> tensor;
        if (batch == 1) {
            tensor = std::make_shared<SimpleTensor>(std::vector<int>{seq});
            for (int i = 0; i < seq; ++i) {
                tensor->data()[i] = static_cast<float>(i % vocab_size_);
            }
        } else {
            tensor = std::make_shared<SimpleTensor>(std::vector<int>{batch, seq});
            for (int b = 0; b < batch; ++b) {
                for (int s = 0; s < seq; ++s) {
                    tensor->data()[b * seq + s] = static_cast<float>((b * seq + s) % vocab_size_);
                }
            }
        }
        return tensor;
    }
    
    int rank_;
    int world_size_;
    int batch_size_;
    int seq_len_;
    int hidden_dim_;
    int output_dim_;
    int vocab_size_;
    int num_heads_;
    int head_dim_;
};

// ========================================
// MPIEmbeddingOperator Interface Tests
// ========================================

TEST_F(OperatorBatchInterfaceTest, EmbeddingDetects2DInput) {
    MPIEmbeddingOperator op(vocab_size_, hidden_dim_);
    
    // Create 1D token input [seq_len]
    auto tokens = createTokenTensor(seq_len_, 1);
    
    // Create embedding table (full table for now)
    auto table = create2DTensor(vocab_size_, hidden_dim_, 1.0f);
    
    // Pre-allocate output tensor
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len_, hidden_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {tokens, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Output should be [seq_len, hidden_dim]
    ASSERT_NE(outputs[0], nullptr);
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], seq_len_);
    EXPECT_EQ(shape[1], hidden_dim_);
}

TEST_F(OperatorBatchInterfaceTest, EmbeddingDetects3DInput) {
    MPIEmbeddingOperator op(vocab_size_, hidden_dim_);
    
    // Create 2D token input [batch, seq_len]
    auto tokens = createTokenTensor(seq_len_, batch_size_);
    
    // Create embedding table
    auto table = create2DTensor(vocab_size_, hidden_dim_, 1.0f);
    
    // Pre-allocate output tensor
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{batch_size_, seq_len_, hidden_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {tokens, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Output should be [batch, seq_len, hidden_dim]
    ASSERT_NE(outputs[0], nullptr);
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], batch_size_);
    EXPECT_EQ(shape[1], seq_len_);
    EXPECT_EQ(shape[2], hidden_dim_);
}

// ========================================
// MPILinearOperator Interface Tests
// ========================================

TEST_F(OperatorBatchInterfaceTest, LinearDetects2DInput) {
    MPILinearOperator op;
    
    // Create 2D input [seq, hidden]
    auto input = create2DTensor(seq_len_, hidden_dim_, 1.0f);
    
    // Create weight [output_dim, hidden_dim] per operator convention
    auto weight = create2DTensor(output_dim_, hidden_dim_, 0.01f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len_, output_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Output should be [seq, output_dim]
    ASSERT_NE(outputs[0], nullptr);
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], seq_len_);
    EXPECT_EQ(shape[1], output_dim_);
}

TEST_F(OperatorBatchInterfaceTest, LinearDetects3DInput) {
    MPILinearBatchOperator op;  // Use batch-aware operator for 3D inputs
    
    // Create 3D input [batch, seq, hidden]
    auto input = create3DTensor(batch_size_, seq_len_, hidden_dim_, 1.0f);
    
    // Create weight [output_dim, hidden_dim] per operator convention
    auto weight = create2DTensor(output_dim_, hidden_dim_, 0.01f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{batch_size_, seq_len_, output_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Output should be [batch, seq, output_dim]
    ASSERT_NE(outputs[0], nullptr);
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], batch_size_);
    EXPECT_EQ(shape[1], seq_len_);
    EXPECT_EQ(shape[2], output_dim_);
}

TEST_F(OperatorBatchInterfaceTest, LinearWithBias2D) {
    MPILinearOperator op;
    
    auto input = create2DTensor(seq_len_, hidden_dim_, 1.0f);
    auto weight = create2DTensor(output_dim_, hidden_dim_, 0.01f);
    
    // Create bias [output_dim]
    auto bias = std::make_shared<SimpleTensor>(std::vector<int>{output_dim_});
    bias->fill(0.1f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len_, output_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    ASSERT_NE(outputs[0], nullptr);
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape[0], seq_len_);
    EXPECT_EQ(shape[1], output_dim_);
}

TEST_F(OperatorBatchInterfaceTest, LinearWithBias3D) {
    MPILinearBatchOperator op;  // Use batch-aware operator for 3D inputs
    
    auto input = create3DTensor(batch_size_, seq_len_, hidden_dim_, 1.0f);
    auto weight = create2DTensor(output_dim_, hidden_dim_, 0.01f);
    auto bias = std::make_shared<SimpleTensor>(std::vector<int>{output_dim_});
    bias->fill(0.0f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{batch_size_, seq_len_, output_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    ASSERT_NE(outputs[0], nullptr);
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape[0], batch_size_);
    EXPECT_EQ(shape[1], seq_len_);
    EXPECT_EQ(shape[2], output_dim_);
}

// ========================================
// MPIRMSNormOperator Interface Tests
// ========================================

TEST_F(OperatorBatchInterfaceTest, RMSNormDetects2DInput) {
    MPIRMSNormOperator op;
    
    // Create 2D input [seq, hidden]
    auto input = create2DTensor(seq_len_, hidden_dim_, 1.0f);
    
    // Create 1D weight [hidden_dim]
    auto weight = std::make_shared<SimpleTensor>(std::vector<int>{hidden_dim_});
    weight->fill(1.0f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len_, hidden_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Output should be [seq, hidden]
    ASSERT_NE(outputs[0], nullptr);
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], seq_len_);
    EXPECT_EQ(shape[1], hidden_dim_);
}

TEST_F(OperatorBatchInterfaceTest, RMSNormDetects3DInput) {
    MPIRMSNormOperator op;
    
    // Create 3D input [batch, seq, hidden]
    auto input = create3DTensor(batch_size_, seq_len_, hidden_dim_, 1.0f);
    
    // Create 1D weight [hidden_dim]
    auto weight = std::make_shared<SimpleTensor>(std::vector<int>{hidden_dim_});
    weight->fill(1.0f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{batch_size_, seq_len_, hidden_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Output should be [batch, seq, hidden]
    ASSERT_NE(outputs[0], nullptr);
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], batch_size_);
    EXPECT_EQ(shape[1], seq_len_);
    EXPECT_EQ(shape[2], hidden_dim_);
}

// ========================================
// Backward Compatibility Tests (batch=1)
// ========================================

TEST_F(OperatorBatchInterfaceTest, EmbeddingBackwardCompatible) {
    MPIEmbeddingOperator op(vocab_size_, hidden_dim_);
    
    // Single sequence case (existing behavior)
    auto tokens = createTokenTensor(seq_len_, 1);
    auto table = create2DTensor(vocab_size_, hidden_dim_, 1.0f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len_, hidden_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {tokens, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Should still produce 2D output for backward compatibility
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 2);
}

TEST_F(OperatorBatchInterfaceTest, LinearBackwardCompatible) {
    MPILinearOperator op;
    
    // Single sequence case
    auto input = create2DTensor(seq_len_, hidden_dim_, 1.0f);
    auto weight = create2DTensor(output_dim_, hidden_dim_, 0.01f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len_, output_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Should produce 2D output
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 2);
}

TEST_F(OperatorBatchInterfaceTest, RMSNormBackwardCompatible) {
    MPIRMSNormOperator op;
    
    auto input = create2DTensor(seq_len_, hidden_dim_, 1.0f);
    auto weight = std::make_shared<SimpleTensor>(std::vector<int>{hidden_dim_});
    weight->fill(1.0f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len_, hidden_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    // Should produce 2D output
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape.size(), 2);
}

// ========================================
// Shape Validation Tests
// ========================================

TEST_F(OperatorBatchInterfaceTest, LinearRejectsMismatchedDimensions) {
    MPILinearOperator op;
    
    // Create incompatible input and weight
    auto input = create2DTensor(seq_len_, hidden_dim_, 1.0f);
    auto weight = create2DTensor(output_dim_, hidden_dim_ + 10, 0.01f); // Wrong input dim (intentional)
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs(1);
    
    // Should fail validation
    bool success = op.execute(inputs, outputs);
    EXPECT_FALSE(success);
}

TEST_F(OperatorBatchInterfaceTest, RMSNormRejectsMismatchedWeight) {
    MPIRMSNormOperator op;
    
    auto input = create2DTensor(seq_len_, hidden_dim_, 1.0f);
    auto weight = std::make_shared<SimpleTensor>(std::vector<int>{hidden_dim_ + 5}); // Wrong size (intentional)
    weight->fill(1.0f);
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs(1);
    
    bool success = op.execute(inputs, outputs);
    EXPECT_FALSE(success);
}

// ========================================
// Batch Dimension Propagation Tests
// ========================================

TEST_F(OperatorBatchInterfaceTest, BatchDimensionPreservedThroughLinear) {
    MPILinearBatchOperator op;  // Use batch-aware operator for 3D inputs
    
    // Test various batch sizes
    for (int batch : {1, 2, 4, 8, 16}) {
        auto input = create3DTensor(batch, seq_len_, hidden_dim_, 1.0f);
        // Weight is [output_dim, hidden_dim] for linear projection
        auto weight = create2DTensor(output_dim_, hidden_dim_, 0.01f);
        
        // Pre-allocate output
        auto output = std::make_shared<SimpleTensor>(std::vector<int>{batch, seq_len_, output_dim_});
        
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        
        bool success = op.execute(inputs, outputs);
        EXPECT_TRUE(success) << "Failed for batch=" << batch;
        
        auto shape = outputs[0]->shape();
        EXPECT_EQ(shape[0], batch) << "Batch dimension not preserved for batch=" << batch;
        EXPECT_EQ(shape[1], seq_len_);
        EXPECT_EQ(shape[2], output_dim_);
    }
}

TEST_F(OperatorBatchInterfaceTest, BatchDimensionPreservedThroughRMSNorm) {
    MPIRMSNormOperator op;
    
    // Test various batch sizes
    for (int batch : {1, 2, 4, 8, 16}) {
        auto input = create3DTensor(batch, seq_len_, hidden_dim_, 1.0f);
        // RMSNorm weight must be 1D [hidden_dim]
        auto weight = std::make_shared<SimpleTensor>(std::vector<int>{hidden_dim_});
        weight->fill(1.0f);
        
        // Pre-allocate output (RMSNorm preserves input shape)
        auto output = std::make_shared<SimpleTensor>(std::vector<int>{batch, seq_len_, hidden_dim_});
        
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        
        bool success = op.execute(inputs, outputs);
        EXPECT_TRUE(success) << "Failed for batch=" << batch;
        
        auto shape = outputs[0]->shape();
        EXPECT_EQ(shape[0], batch) << "Batch dimension not preserved for batch=" << batch;
        EXPECT_EQ(shape[1], seq_len_);
        EXPECT_EQ(shape[2], hidden_dim_);
    }
}

// ========================================
// Edge Cases
// ========================================

TEST_F(OperatorBatchInterfaceTest, SingleTokenSequence2D) {
    MPILinearOperator op;
    
    // Single token case [1, hidden]
    auto input = create2DTensor(1, hidden_dim_, 1.0f);
    auto weight = create2DTensor(output_dim_, hidden_dim_, 0.01f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{1, output_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], output_dim_);
}

TEST_F(OperatorBatchInterfaceTest, SingleTokenSequence3D) {
    MPILinearBatchOperator op;  // Use batch operator for 3D inputs
    
    // Single token, single batch [1, 1, hidden]
    auto input = create3DTensor(1, 1, hidden_dim_, 1.0f);
    auto weight = create2DTensor(output_dim_, hidden_dim_, 0.01f);
    
    // Pre-allocate output
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{1, 1, output_dim_});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    EXPECT_TRUE(success);
    
    auto shape = outputs[0]->shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 1);
    EXPECT_EQ(shape[2], output_dim_);
}

// ========================================
// Main
// ========================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
