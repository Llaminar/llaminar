// Benchmark fused vs unfused quant linear (decode+GEMM) across shapes & formats.
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include "../src/tensors/TensorFactory.h"
#include "../src/operators/QuantLinearKernel.h"
#include "../src/AdaptiveMatmul.h"
#include "../src/operators/QuantLinearInstrumentation.h"
#include "../src/operators/QuantSlabCache.h"
#include "../llama.cpp/ggml/src/ggml-quants.h"

using namespace llaminar;

struct Case
{
    QuantFormat fmt;
    int M, K, N;
};

static std::string fmt_name(QuantFormat f)
{
    switch (f)
    {
    case QuantFormat::Q4_0:
        return "Q4_0";
    case QuantFormat::Q5_0:
        return "Q5_0";
    case QuantFormat::Q8_0:
        return "Q8_0";
    case QuantFormat::Q4_K:
        return "Q4_K";
    case QuantFormat::Q5_K:
        return "Q5_K";
    case QuantFormat::Q6_K:
        return "Q6_K";
    case QuantFormat::Q8_K:
        return "Q8_K";
    default:
        return "UNK";
    }
}

static std::shared_ptr<QuantizedTensor> make_random_quant(int K, int N, QuantFormat fmt, std::mt19937 &rng)
{
    QuantBlockDescriptor desc;
    bool kfmt = false;
    int block_elems = 32;
    int bytes = 0;
    int bits = 0;
    switch (fmt)
    {
    case QuantFormat::Q4_0:
        block_elems = 32;
        bytes = 18;
        bits = 4;
        break;
    case QuantFormat::Q5_0:
        block_elems = 32;
        bytes = 20;
        bits = 5;
        break;
    case QuantFormat::Q8_0:
        block_elems = 32;
        bytes = 34;
        bits = 8;
        break;
    case QuantFormat::Q4_K:
        block_elems = QK_K;
        bytes = sizeof(block_q4_K);
        bits = 4;
        kfmt = true;
        break;
    case QuantFormat::Q5_K:
        block_elems = QK_K;
        bytes = sizeof(block_q5_K);
        bits = 5;
        kfmt = true;
        break;
    case QuantFormat::Q6_K:
        block_elems = QK_K;
        bytes = sizeof(block_q6_K);
        bits = 6;
        kfmt = true;
        break;
    case QuantFormat::Q8_K:
        block_elems = QK_K;
        bytes = sizeof(block_q8_K);
        bits = 8;
        kfmt = true;
        break;
    default:
        block_elems = 32;
        bytes = 18;
        bits = 4;
        break;
    }
    desc.elements_per_block = block_elems;
    desc.bytes_per_block = bytes;
    desc.bits_per_value = bits;
    desc.scale_count = kfmt ? 0 : 1;
    desc.is_k_quant = kfmt;
    int blocks_per_row = (N + block_elems - 1) / block_elems;
    size_t total_blocks = (size_t)K * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * bytes);
    std::uniform_real_distribution<float> dist(-4.f, 4.f);
    // Build each block by reference quantization where possible for K formats; else random bytes with scale=1
    for (int k = 0; k < K; ++k)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            size_t bi = (size_t)k * blocks_per_row + b;
            uint8_t *ptr = raw.data() + bi * bytes;
            if (kfmt)
            {
                std::vector<float> buf(block_elems, 0.f);
                for (int i = 0; i < block_elems; ++i)
                    buf[i] = dist(rng);
                switch (fmt)
                {
                case QuantFormat::Q4_K:
                {
                    block_q4_K blk{};
                    quantize_row_q4_K_ref(buf.data(), &blk, block_elems);
                    std::memcpy(ptr, &blk, sizeof(blk));
                    break;
                }
                case QuantFormat::Q5_K:
                {
                    block_q5_K blk{};
                    quantize_row_q5_K_ref(buf.data(), &blk, block_elems);
                    std::memcpy(ptr, &blk, sizeof(blk));
                    break;
                }
                case QuantFormat::Q6_K:
                {
                    block_q6_K blk{};
                    quantize_row_q6_K_ref(buf.data(), &blk, block_elems);
                    std::memcpy(ptr, &blk, sizeof(blk));
                    break;
                }
                case QuantFormat::Q8_K:
                {
                    block_q8_K blk{};
                    quantize_row_q8_K_ref(buf.data(), &blk, block_elems);
                    std::memcpy(ptr, &blk, sizeof(blk));
                    break;
                }
                default:
                    break;
                }
            }
            else
            {
                uint16_t h = 0x3C00;
                std::memcpy(ptr, &h, 2); // scale
                for (int i = 2; i < bytes; ++i)
                    ptr[i] = (uint8_t)rng();
            }
        }
    }
    QuantStorageLayout layout{fmt, {K, N}, total_blocks, desc};
    return std::make_shared<QuantizedTensor>(layout, raw);
}

static void dequant_full(const QuantizedTensor &Wq, std::vector<float> &Wfull)
{
    const auto &layout = Wq.layout();
    int K = layout.original_shape[0];
    int N = layout.original_shape[1];
    int block_elems = layout.block_desc.elements_per_block;
    int blocks_per_row = (N + block_elems - 1) / block_elems;
    std::vector<float> tmp(block_elems);
    for (int k = 0; k < K; ++k)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            int col0 = b * block_elems;
            int span = std::min(block_elems, N - col0);
            size_t bi = (size_t)k * blocks_per_row + b;
            Wq.decodeBlock(bi, tmp.data());
            std::memcpy(Wfull.data() + (size_t)k * N + col0, tmp.data(), span * sizeof(float));
        }
    }
}

int main(int argc, char **argv)
{
    std::vector<Case> cases = {
        // Small baseline shapes
        {QuantFormat::Q4_0, 32, 256, 384},
        {QuantFormat::Q8_0, 32, 256, 384},
        {QuantFormat::Q4_K, 32, 256, 384},
        {QuantFormat::Q5_K, 32, 256, 384},
        {QuantFormat::Q6_K, 32, 256, 384},
        {QuantFormat::Q8_K, 32, 256, 384},
        {QuantFormat::Q4_0, 64, 512, 512},
        {QuantFormat::Q4_K, 64, 512, 512},
        // Larger projection-like shapes (simulate FFN/attention projections)
        {QuantFormat::Q4_0, 64, 4096, 4096},
        {QuantFormat::Q8_0, 64, 4096, 4096},
        {QuantFormat::Q4_0, 128, 4096, 2048},
        {QuantFormat::Q8_0, 128, 4096, 2048}};
    int iters = 5;
    if (argc > 1)
        iters = std::max(1, std::atoi(argv[1]));
    std::mt19937 rng(123);
    std::cout << "Format,M,K,N,iters,fused_ms,dequant_ms,gemm_ms,slab_first_ms,slab_hit_ms,total_unfused_ms,speedup,decoded_blocks,applied_blocks,simd_path,scalar_path,slab_reused" << std::endl;
    for (const auto &cs : cases)
    {
        auto Wq = make_random_quant(cs.K, cs.N, cs.fmt, rng);
        std::vector<float> A((size_t)cs.M * cs.K);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto &v : A)
            v = dist(rng);
        // Warm decode once
        std::vector<float> Wfull((size_t)cs.K * cs.N, 0.f);
        dequant_full(*Wq, Wfull);
        // Fused timing
        double fused_ms = 0, unfused_decode_ms = 0, gemm_ms = 0;
        double slab_first_ms = 0, slab_hit_ms = 0;
        long long slab_reuse_hits = 0;
        quantLinearInstr().reset();
        if (debugEnv().quant.fused_enable)
        {
            for (int it = 0; it < iters; ++it)
            {
                std::vector<float> C_fused((size_t)cs.M * cs.N, 0.f);
                QuantLinearParams p;
                p.A = A.data();
                p.M = cs.M;
                p.K = cs.K;
                p.N = cs.N;
                p.C = C_fused.data();
                p.zero_C = true;
                auto t0 = std::chrono::high_resolution_clock::now();
                quant_linear_fused(*Wq, p);
                auto t1 = std::chrono::high_resolution_clock::now();
                fused_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            fused_ms /= iters;
        }
        // Slab first decode (no reuse)
        {
            QuantSlabCache::instance().clear();
            auto t0 = std::chrono::high_resolution_clock::now();
            QuantSlab slab;
            bool reused = QuantSlabCache::instance().getOrDecode(*Wq, 0, cs.N, slab, false);
            auto t1 = std::chrono::high_resolution_clock::now();
            slab_first_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            (void)reused;
        }
        // Subsequent hits (reuse allowed)
        {
            double accum = 0;
            int hits = 0;
            slab_reuse_hits = 0;
            for (int it = 0; it < iters; ++it)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                QuantSlab slab;
                bool reused = QuantSlabCache::instance().getOrDecode(*Wq, 0, cs.N, slab, true);
                auto t1 = std::chrono::high_resolution_clock::now();
                accum += std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (reused)
                    slab_reuse_hits++;
                hits++;
            }
            slab_hit_ms = accum / hits;
        }
        for (int it = 0; it < iters; ++it)
        {
            // Unfused: full dequant then adaptiveMatMul (routes OpenBLAS/COSMA)
            std::vector<float> Wtmp((size_t)cs.K * cs.N, 0.f);
            auto d0 = std::chrono::high_resolution_clock::now();
            dequant_full(*Wq, Wtmp);
            auto d1 = std::chrono::high_resolution_clock::now();
            std::vector<float> C((size_t)cs.M * cs.N, 0.f);
            auto g0 = std::chrono::high_resolution_clock::now();
            // adaptiveMatMul arguments: A (M x K), B (K x N), C (M x N)
            llaminar::adaptiveMatMul(A.data(), Wtmp.data(), C.data(),
                                     cs.M, cs.N, cs.K,
                                     /*is_prefill*/ false,
                                     /*distributed_partition*/ false,
                                     /*transpose_A*/ false,
                                     /*transpose_B*/ false,
                                     1.0f, 0.0f);
            auto g1 = std::chrono::high_resolution_clock::now();
            unfused_decode_ms += std::chrono::duration<double, std::milli>(d1 - d0).count();
            gemm_ms += std::chrono::duration<double, std::milli>(g1 - g0).count();
        }
        unfused_decode_ms /= iters;
        gemm_ms /= iters;
        double unfused_total = unfused_decode_ms + gemm_ms;
        double speedup = (fused_ms > 0 ? unfused_total / fused_ms : 0.0);
        auto &instr = quantLinearInstr();
        std::cout << fmt_name(cs.fmt) << "," << cs.M << "," << cs.K << "," << cs.N << "," << iters
                  << "," << fused_ms << "," << unfused_decode_ms << "," << gemm_ms
                  << "," << slab_first_ms << "," << slab_hit_ms
                  << "," << unfused_total << "," << speedup
                  << "," << instr.decoded_blocks.load() << "," << instr.applied_blocks.load()
                  << "," << instr.simd_path.load() << "," << instr.scalar_path.load()
                  << "," << slab_reuse_hits << std::endl;
    }
    return 0;
}
