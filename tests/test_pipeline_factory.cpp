/**
 * @file test_pipeline_factory.cpp
 * @brief Tests for multi-architecture PipelineFactory.
 * @author David Sanftenberg
 */
#include <gtest/gtest.h>
#include "abstract_pipeline.h"
#include "llama_pipeline_adapter.h"
#include "qwen_pipeline_adapter.h"
#include "transformer_config.h"

using namespace llaminar;

// Prevent MPI_Finalize during test execution (Google Test calls MPI_Init)
class MPIEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        // Set environment variable to prevent individual test MPI finalizations
        setenv("LLAMINAR_TEST_MPI_NO_FINALIZE", "1", 1);
    }
};

class PipelineFactoryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Register both architectures
        registerQwenPipeline();
        registerLlamaPipeline();
    }

    ModelConfig createTestConfig(const std::string &arch)
    {
        TransformerLayerConfig layer_config;
        layer_config.n_layers = 2;
        layer_config.n_head = 4;
        layer_config.n_head_kv = 4; // Must be <= n_head
        layer_config.head_dim = 32; // d_model / n_head = 128 / 4 = 32
        layer_config.d_model = 128;
        layer_config.d_ff = 512;
        layer_config.vocab_size = 1000;
        layer_config.max_seq_len = 256;
        layer_config.eps = 1e-5f;

        ModelConfig config(layer_config);
        config.architecture = arch;
        return config;
    }
};

TEST_F(PipelineFactoryTest, CreateQwenPipeline)
{
    auto config = createTestConfig("qwen");
    auto pipeline = PipelineFactory::instance().create(config);

    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(pipeline->name(), "QwenPipelineAdapter");
    EXPECT_EQ(pipeline->config().architecture, "qwen");
}

TEST_F(PipelineFactoryTest, CreateLlamaPipeline)
{
    auto config = createTestConfig("llama");
    auto pipeline = PipelineFactory::instance().create(config);

    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(pipeline->name(), "LlamaPipelineAdapter");
    EXPECT_EQ(pipeline->config().architecture, "llama");
}

TEST_F(PipelineFactoryTest, UnknownArchitectureReturnsNull)
{
    auto config = createTestConfig("unknown_arch");
    auto pipeline = PipelineFactory::instance().create(config);

    // Factory should return nullptr for unknown architectures
    EXPECT_EQ(pipeline, nullptr);
}

TEST_F(PipelineFactoryTest, MultipleCreationsWork)
{
    auto qwen_config = createTestConfig("qwen");
    auto llama_config = createTestConfig("llama");

    auto qwen1 = PipelineFactory::instance().create(qwen_config);
    auto llama1 = PipelineFactory::instance().create(llama_config);
    auto qwen2 = PipelineFactory::instance().create(qwen_config);

    ASSERT_NE(qwen1, nullptr);
    ASSERT_NE(llama1, nullptr);
    ASSERT_NE(qwen2, nullptr);

    EXPECT_EQ(qwen1->name(), "QwenPipelineAdapter");
    EXPECT_EQ(llama1->name(), "LlamaPipelineAdapter");
    EXPECT_EQ(qwen2->name(), "QwenPipelineAdapter");

    // Each should be a separate instance
    EXPECT_NE(qwen1.get(), qwen2.get());
}

TEST_F(PipelineFactoryTest, ConfigurationPreserved)
{
    auto config = createTestConfig("llama");
    config.getLayerConfig().n_layers = 8;
    config.getLayerConfig().n_head = 16;

    auto pipeline = PipelineFactory::instance().create(config);

    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(pipeline->config().getLayerConfig().n_layers, 8);
    EXPECT_EQ(pipeline->config().getLayerConfig().n_head, 16);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}
