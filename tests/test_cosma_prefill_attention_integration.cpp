#include "../src/mpi_transformer_pipeline.h"
#include "../src/cosma_prefill_manager.h"
#include "../src/tensors/tensor_factory.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <random>
#include <vector>
#include <limits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "test_timeout_guard.h"

using namespace llaminar;

namespace
{
    struct MPIFinalizerPrefillAttention
    {
        ~MPIFinalizerPrefillAttention()
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (initialized)
            {
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized)
                {
                    MPI_Finalize();
                }
            }
        }
    };

    std::shared_ptr<TensorBase> makeTensor(const std::vector<int> &shape, std::mt19937 &rng)
    {
        auto tensor = TensorFactory::create_simple(shape);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        float *ptr = tensor->data();
        int total = tensor->size();
        for (int i = 0; i < total; ++i)
        {
            ptr[i] = dist(rng);
        }
        return tensor;
    }
} // namespace

TEST(CosmaPrefillAttentionIntegration, PrefillPathMatchesBaseline)
{
    static MPIFinalizerPrefillAttention finalizer;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    int world = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2)
    {
        GTEST_SKIP() << "Need >=2 ranks";
    }

    auto timeout = llaminar::test_util::TestTimeoutGuard::ResolveTimeout(
        {"LLAMINAR_TEST_TIMEOUT_MS", "LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS"},
        std::chrono::milliseconds(60000));
    llaminar::test_util::TestTimeoutGuard watchdog("CosmaPrefillAttentionIntegration.PrefillPathMatchesBaseline", timeout);

    setenv("LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT", "1", 1);
    setenv("LLAMINAR_COSMA_FORCE_DIRECT", "1", 1);

    TransformerLayerConfig config{};
    config.n_head = 2;
    config.n_head_kv = 2;
    config.head_dim = 32;
    config.d_model = config.n_head * config.head_dim;
    config.d_ff = config.d_model * 2;
    config.vocab_size = 64;
    config.max_seq_len = 128;
    config.n_layers = 2;
    config.eps = 1e-5f;

    MPITransformerPipeline pipeline(config);

    std::mt19937 rng(42);
    MPITransformerPipeline::ModelWeights weights;
    weights.token_embedding = makeTensor({config.vocab_size, config.d_model}, rng);
    weights.output_norm_weight = makeTensor({config.d_model}, rng);
    weights.lm_head = makeTensor({config.d_model, config.vocab_size}, rng);

    weights.attn_norm_weight.resize(config.n_layers);
    weights.wq.resize(config.n_layers);
    weights.wk.resize(config.n_layers);
    weights.wv.resize(config.n_layers);
    weights.wo.resize(config.n_layers);
    weights.ffn_norm_weight.resize(config.n_layers);
    weights.w_gate.resize(config.n_layers);
    weights.w_up.resize(config.n_layers);
    weights.w_down.resize(config.n_layers);

    for (int layer = 0; layer < config.n_layers; ++layer)
    {
        weights.attn_norm_weight[layer] = makeTensor({config.d_model}, rng);
        weights.wq[layer] = makeTensor({config.d_model, config.n_head * config.head_dim}, rng);
        weights.wk[layer] = makeTensor({config.d_model, config.n_head_kv * config.head_dim}, rng);
        weights.wv[layer] = makeTensor({config.d_model, config.n_head_kv * config.head_dim}, rng);
        weights.wo[layer] = makeTensor({config.n_head * config.head_dim, config.d_model}, rng);
        weights.ffn_norm_weight[layer] = makeTensor({config.d_model}, rng);
        weights.w_gate[layer] = makeTensor({config.d_model, config.d_ff}, rng);
        weights.w_up[layer] = makeTensor({config.d_model, config.d_ff}, rng);
        weights.w_down[layer] = makeTensor({config.d_ff, config.d_model}, rng);
    }

    const int seq_len = 64;
    std::vector<int> token_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
    {
        token_ids[i] = (i * 7) % config.vocab_size;
    }

    CosmaPrefillManager &manager = CosmaPrefillManager::instance();
    manager.reset_stats();

    manager.set_force_cosma(false);
    manager.set_threshold(std::numeric_limits<int>::max());

    std::shared_ptr<TensorBase> baseline_output;
    ASSERT_TRUE(pipeline.execute(token_ids, weights, baseline_output));

    const int output_elems = seq_len * config.vocab_size;
    std::vector<float> baseline_values(output_elems, 0.f);
    if (baseline_output && baseline_output->data())
    {
        std::memcpy(baseline_values.data(), baseline_output->data(), output_elems * sizeof(float));
    }
    MPI_Bcast(baseline_values.data(), output_elems, MPI_FLOAT, 0, MPI_COMM_WORLD);

    manager.set_threshold(1);
    manager.set_force_cosma(true);
    manager.reset_stats();

    std::shared_ptr<TensorBase> cosma_output;
    ASSERT_TRUE(pipeline.execute(token_ids, weights, cosma_output));

    std::vector<float> cosma_values(output_elems, 0.f);
    if (cosma_output && cosma_output->data())
    {
        std::memcpy(cosma_values.data(), cosma_output->data(), output_elems * sizeof(float));
    }
    MPI_Bcast(cosma_values.data(), output_elems, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < output_elems; ++i)
        {
            EXPECT_NEAR(cosma_values[i], baseline_values[i], 1e-3f);
        }
    }

    manager.set_threshold(4096);
    manager.set_force_cosma(false);
    manager.reset_stats();
    unsetenv("LLAMINAR_COSMA_FORCE_DIRECT");
    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED");
    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED_DIAG");
    unsetenv("LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT");

    watchdog.disarm();
}
