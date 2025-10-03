#include "src/kernels/MPIAttentionKernel.h"
#include "src/tensors/tensor_factory.h"
#include "src/utils/debug_env.h"
#include "src/logger.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <mpi.h>

using namespace llaminar;

struct BenchCase
{
    int seq_len;
    int n_head;
    int n_head_kv;
    int head_dim;
};

static std::shared_ptr<TensorBase> make_tensor(const std::vector<int> &shape)
{
    return TensorFactory::create_simple(shape);
}

static void fill_random(float *data, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        data[i] = float((i * 1315423911u) % 1000) / 1000.f - 0.5f; // cheap deterministic
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    std::vector<BenchCase> cases = {
        {64, 12, 12, 64},
        {128, 12, 12, 64},
        {256, 16, 16, 64},
        {512, 16, 16, 64},
        {512, 32, 32, 64},
        {1024, 32, 32, 64}};

    std::cout << "seq_len,n_head,head_dim,mode,ms" << std::endl;

    for (auto bc : cases)
    {
        int d_model = bc.n_head * bc.head_dim;
        // Allocate tensors: input, wq, wk, wv, wo, k_cache, v_cache
        auto input = make_tensor({bc.seq_len, d_model});
        auto wq = make_tensor({d_model, d_model});
        auto wk = make_tensor({d_model, bc.n_head_kv * bc.head_dim});
        auto wv = make_tensor({d_model, bc.n_head_kv * bc.head_dim});
        auto wo = make_tensor({d_model, d_model});
        auto k_cache = make_tensor({bc.seq_len, bc.n_head_kv * bc.head_dim});
        auto v_cache = make_tensor({bc.seq_len, bc.n_head_kv * bc.head_dim});
        auto output = make_tensor({bc.seq_len, d_model});

        // Populate deterministic data
        fill_random(input->data(), input->size());
        fill_random(wq->data(), wq->size());
        fill_random(wk->data(), wk->size());
        fill_random(wv->data(), wv->size());
        fill_random(wo->data(), wo->size());

        // Warmup + measure for primitives ON and OFF
        for (int mode = 0; mode < 2; ++mode)
        {
            // Force env toggle via direct snapshot hack (not thread-safe but fine for single-thread bench)
            auto &env = const_cast<DebugEnvSnapshot &>(debugEnv());
            env.attention.use_primitives = (mode == 0); // 0 -> primitives ON, 1 -> OFF

            MPIAttentionKernel kernel(bc.n_head, bc.n_head_kv, bc.head_dim);
            kernel.setSequencePosition(0);
            kernel.setOutputMode(MPIAttentionKernel::AttentionOutputMode::LocalHeads);

            std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
            std::vector<std::shared_ptr<TensorBase>> outputs = {output};

            // Warmup
            kernel.execute(inputs, outputs);
            kernel.execute(inputs, outputs);

            auto t0 = std::chrono::high_resolution_clock::now();
            kernel.execute(inputs, outputs);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            std::cout << bc.seq_len << ',' << bc.n_head << ',' << bc.head_dim << ',' << (mode == 0 ? "prims" : "legacy") << ',' << ms << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}
