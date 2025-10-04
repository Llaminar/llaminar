#include "../src/distributed_transformer_pipeline.h"
#include "../src/cosma_prefill_manager.h"
#include "../src/tensors/tensor_factory.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <tuple>
#include <sstream>
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

    DistributedTransformerPipeline pipeline(config);

    std::mt19937 rng(42);
    DistributedTransformerPipeline::ModelWeights weights;
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

    setenv("LLAMINAR_PREFILL_CAPTURE_BASELINE", "1", 1);

    std::shared_ptr<TensorBase> baseline_output;
    ASSERT_TRUE(pipeline.execute(token_ids, weights, baseline_output));

    const int output_elems = seq_len * config.vocab_size;
    std::vector<float> baseline_values(output_elems, 0.f);
    if (baseline_output && baseline_output->data())
    {
        std::memcpy(baseline_values.data(), baseline_output->data(), output_elems * sizeof(float));
    }
    MPI_Bcast(baseline_values.data(), output_elems, MPI_FLOAT, 0, MPI_COMM_WORLD);

    unsetenv("LLAMINAR_PREFILL_CAPTURE_BASELINE");

    manager.set_threshold(1);
    manager.set_force_cosma(true);
    manager.reset_stats();

    setenv("LLAMINAR_PREFILL_COMPARE_BASELINE", "1", 1);

    std::shared_ptr<TensorBase> cosma_output;
    ASSERT_TRUE(pipeline.execute(token_ids, weights, cosma_output));

    std::vector<float> cosma_values(output_elems, 0.f);
    if (cosma_output && cosma_output->data())
    {
        std::memcpy(cosma_values.data(), cosma_output->data(), output_elems * sizeof(float));
    }
    MPI_Bcast(cosma_values.data(), output_elems, MPI_FLOAT, 0, MPI_COMM_WORLD);

    unsetenv("LLAMINAR_PREFILL_COMPARE_BASELINE");

    auto dump_row = [&](int inspect_row, int inspect_count, const char *label)
    {
        if (rank != 0 || inspect_row < 0 || inspect_row >= seq_len)
        {
            return;
        }
        inspect_count = std::max(1, std::min(inspect_count, config.vocab_size));
        std::ostringstream oss;
        oss << "[DebugDiff] label=" << label << " row=" << inspect_row << " baseline=";
        for (int c = 0; c < inspect_count; ++c)
        {
            int idx = inspect_row * config.vocab_size + c;
            oss << baseline_values[idx];
            if (c + 1 < inspect_count)
            {
                oss << ',';
            }
        }
        oss << " cosma=";
        for (int c = 0; c < inspect_count; ++c)
        {
            int idx = inspect_row * config.vocab_size + c;
            oss << cosma_values[idx];
            if (c + 1 < inspect_count)
            {
                oss << ',';
            }
        }
        oss << " diff=";
        for (int c = 0; c < inspect_count; ++c)
        {
            int idx = inspect_row * config.vocab_size + c;
            oss << (cosma_values[idx] - baseline_values[idx]);
            if (c + 1 < inspect_count)
            {
                oss << ',';
            }
        }
        oss << '\n';
        std::fprintf(stderr, "%s", oss.str().c_str());
        std::fflush(stderr);
    };

    if (const char *dump_env = std::getenv("LLAMINAR_DEBUG_COSMA_DIFF"))
    {
        int inspect_row = 0;
        int inspect_count = std::min(config.vocab_size, 16);
        if (const char *row_env = std::getenv("LLAMINAR_DEBUG_COSMA_DIFF_ROW"))
        {
            inspect_row = std::atoi(row_env);
        }
        dump_row(inspect_row, inspect_count, dump_env);
    }

    if (rank == 0)
    {
        const float threshold = 1e-3f;
        size_t mismatch_count = 0;
        double max_abs = 0.0;
        double sum_abs = 0.0;
        long double sum_sq = 0.0L;
        long double denom_sq = 0.0L;
        size_t worst_index = 0;
        float worst_cosma = 0.0f;
        float worst_baseline = 0.0f;
        std::vector<std::tuple<int, float, float, double>> top_diffs;
        top_diffs.reserve(5);

        auto consider_top = [&](int idx, float cosma, float baseline, double diff)
        {
            if (top_diffs.size() < 5)
            {
                top_diffs.emplace_back(idx, cosma, baseline, diff);
            }
            else
            {
                auto min_it = std::min_element(top_diffs.begin(), top_diffs.end(),
                                               [](const auto &lhs, const auto &rhs)
                                               { return std::get<3>(lhs) < std::get<3>(rhs); });
                if (diff > std::get<3>(*min_it))
                {
                    *min_it = std::make_tuple(idx, cosma, baseline, diff);
                }
            }
        };

        for (int i = 0; i < output_elems; ++i)
        {
            float cosma_v = cosma_values[i];
            float base_v = baseline_values[i];
            double diff = std::fabs(static_cast<double>(cosma_v) - static_cast<double>(base_v));
            sum_abs += diff;
            sum_sq += diff * diff;
            double base_d = static_cast<double>(base_v);
            denom_sq += base_d * base_d;
            if (diff > max_abs)
            {
                max_abs = diff;
                worst_index = static_cast<size_t>(i);
                worst_cosma = cosma_v;
                worst_baseline = base_v;
            }
            if (diff > threshold)
            {
                ++mismatch_count;
                consider_top(i, cosma_v, base_v, diff);
            }
        }

        double mean_abs = output_elems > 0 ? sum_abs / static_cast<double>(output_elems) : 0.0;
        double rel_l2 = std::sqrt(static_cast<double>(sum_sq)) / (std::sqrt(static_cast<double>(denom_sq)) + 1e-30);

        if (mismatch_count > 0)
        {
            std::sort(top_diffs.begin(), top_diffs.end(),
                      [](const auto &lhs, const auto &rhs)
                      { return std::get<3>(lhs) > std::get<3>(rhs); });
            std::ostringstream oss;
            oss << "COSMA prefill attention comparison failed: mismatches=" << mismatch_count
                << " threshold=" << threshold
                << " max_abs=" << max_abs
                << " mean_abs=" << mean_abs
                << " rel_l2=" << rel_l2
                << " worst_index=" << worst_index
                << " cosma=" << worst_cosma
                << " baseline=" << worst_baseline;
            oss << " top_diffs=";
            for (size_t i = 0; i < top_diffs.size(); ++i)
            {
                int idx = std::get<0>(top_diffs[i]);
                int row = idx / config.vocab_size;
                int col = idx % config.vocab_size;
                oss << " [i=" << idx << " r=" << row << " c=" << col
                    << " cosma=" << std::get<1>(top_diffs[i])
                    << " baseline=" << std::get<2>(top_diffs[i])
                    << " diff=" << std::get<3>(top_diffs[i]) << ']';
            }
            ADD_FAILURE() << oss.str();

            int worst_row = static_cast<int>(worst_index / config.vocab_size);
            dump_row(worst_row, config.vocab_size, "worst_row");
            if (!top_diffs.empty())
            {
                int top_row = std::get<0>(top_diffs.front()) / config.vocab_size;
                dump_row(top_row, config.vocab_size, "top_diff_row");
            }
        }

        EXPECT_EQ(static_cast<size_t>(0), mismatch_count) << "COSMA output diverged from baseline";
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
