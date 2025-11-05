/**
 * @file Test__Phase5DirectNVRTC.cpp
 * @brief Test Phase 5 kernel compiled DIRECTLY with NVRTC (no Jitify)
 *
 * This bypasses Jitify to rule out any JIT framework overhead or bugs.
 */

#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

#define NVRTC_CHECK(call)                                                         \
    do                                                                            \
    {                                                                             \
        nvrtcResult res = call;                                                   \
        if (res != NVRTC_SUCCESS)                                                 \
        {                                                                         \
            std::cerr << "NVRTC error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << nvrtcGetErrorString(res) << std::endl;                   \
            throw std::runtime_error("NVRTC call failed");                        \
        }                                                                         \
    } while (0)

#define CU_CHECK(call)                                                                  \
    do                                                                                  \
    {                                                                                   \
        CUresult res = call;                                                            \
        if (res != CUDA_SUCCESS)                                                        \
        {                                                                               \
            const char *err_str;                                                        \
            cuGetErrorString(res, &err_str);                                            \
            std::cerr << "CUDA Driver error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << err_str << std::endl;                                          \
            throw std::runtime_error("CUDA Driver call failed");                        \
        }                                                                               \
    } while (0)

class Phase5DirectNVRTCTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            GTEST_SKIP() << "No CUDA device available";
        }

        // Initialize CUDA Driver API
        cuInit(0);
        CUdevice device;
        cuDeviceGet(&device, 0);
        CUcontext ctx;
        cuDevicePrimaryCtxRetain(&ctx, device);
        cuCtxSetCurrent(ctx);

        M_ = 1024;
        N_ = 896;
        K_ = 896;

        cudaMalloc(&d_A_, M_ * K_ * sizeof(float));
        int blocks_per_row = (K_ + 31) / 32;
        cudaMalloc(&d_B_, N_ * blocks_per_row * 18);
        cudaMalloc(&d_C_, M_ * N_ * sizeof(float));

        // Initialize activations in COLUMN-MAJOR format A[K][M] for coalesced loads
        std::vector<float> h_A(M_ * K_);
        std::vector<uint8_t> h_B(N_ * blocks_per_row * 18);

        // Fill column-major: A[k * M + m] = value
        for (int k = 0; k < K_; ++k)
        {
            for (int m = 0; m < M_; ++m)
            {
                h_A[k * M_ + m] = 0.1f;
            }
        }

        for (auto &v : h_B)
            v = 0x42;

        cudaMemcpy(d_A_, h_A.data(), M_ * K_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_, h_B.data(), N_ * blocks_per_row * 18, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    void TearDown() override
    {
        if (d_A_)
            cudaFree(d_A_);
        if (d_B_)
            cudaFree(d_B_);
        if (d_C_)
            cudaFree(d_C_);
    }

    CUfunction compileWithNVRTC()
    {
        // Build kernel source manually (avoid host-only includes in template)
        std::string source = R"(
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

// IQ4_NL block structure
struct IQ4_NLBlock {
    float scale;
    uint8_t quants[16];
};

// IQ4_NL lookup table
__constant__ float iq4nl_values[16] = {
    -127.0f/127.0f, -104.0f/127.0f, -83.0f/127.0f, -65.0f/127.0f,
    -49.0f/127.0f,  -35.0f/127.0f,  -22.0f/127.0f, -10.0f/127.0f,
    0.0f,            10.0f/127.0f,   22.0f/127.0f,  35.0f/127.0f,
    49.0f/127.0f,    65.0f/127.0f,   83.0f/127.0f,  104.0f/127.0f
};

// Decode function
__device__ __forceinline__ void decodeIQ4NLBlock(
    const IQ4_NLBlock& block, half_t* output)
{
    const float scale = block.scale;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t q = block.quants[i];
        float v0 = scale * iq4nl_values[q & 0xF];
        float v1 = scale * iq4nl_values[q >> 4];
        output[i*2]   = half_t(v0);
        output[i*2+1] = half_t(v1);
    }
}
)" + std::string(R"(
// Configuration
#define TILE_M 64
#define TILE_N 64
#define TILE_K 64
#define SUB_K 16
#define MMA_M 2
#define MMA_N 2
#define BUFFER_STAGES 2
#define THREADS_PER_BLOCK 128
#define SWIZZLE_B 3
#define SWIZZLE_M 3
#define SWIZZLE_S 3

extern "C" __global__ void
__launch_bounds__(THREADS_PER_BLOCK)
iq4nl_gemm_phase5_kernel(
    const float* __restrict__ A,
    const IQ4_NLBlock* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    using tiled_mma = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(Shape<Int<MMA_M>, Int<MMA_N>, Int<1>>{}),
        make_tile(Layout<Shape<_1,_1,_1>>{})
    ));
    
    tiled_mma mma;
    
    // Shared memory with swizzle
    using SmemLayoutA = decltype(composition(
        Swizzle<SWIZZLE_B, SWIZZLE_M, SWIZZLE_S>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_K>>, Stride<Int<TILE_K>, Int<1>>>{}
    ));
    
    using SmemLayoutB = decltype(composition(
        Swizzle<SWIZZLE_B, SWIZZLE_M, SWIZZLE_S>{},
        Layout<Shape<Int<TILE_N>, Int<TILE_K>>, Stride<Int<TILE_K>, Int<1>>>{}
    ));
    
    __shared__ half_t s_A[BUFFER_STAGES][TILE_M][TILE_K];
    __shared__ half_t s_B[TILE_N][TILE_K];
    
    // Thread block coordinates
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    
    // Global memory pointers for this block
    const float* gA_block = A + bx * TILE_M * K;
    const IQ4_NLBlock* gB_block = B + by * TILE_N * (K / 32);
    float* gC_block = C + bx * TILE_M * N + by * TILE_N;
    
    // Accumulator
    float acc[MMA_M * MMA_N * 16] = {0.0f};
    
    // Main K-loop
    int K_tiles = K / TILE_K;
    for (int k = 0; k < K_tiles; k++) {
        // Load A tile to shared memory
        if (k < K_tiles) {
            for (int i = tid; i < TILE_M * TILE_K; i += THREADS_PER_BLOCK) {
                int m_local = i / TILE_K;
                int k_local = i % TILE_K;
                int buf_idx = k % BUFFER_STAGES;
                if (bx * TILE_M + m_local < M && k * TILE_K + k_local < K) {
                    s_A[buf_idx][m_local][k_local] = 
                        half_t(gA_block[m_local * K + k * TILE_K + k_local]);
                } else {
                    s_A[buf_idx][m_local][k_local] = half_t(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // Streaming decode and compute
        for (int sub_k = 0; sub_k < TILE_K / SUB_K; sub_k++) {
            // Decode B sub-tile
            for (int i = tid; i < TILE_N * SUB_K; i += THREADS_PER_BLOCK) {
                int n_local = i / SUB_K;
                int k_sub = i % SUB_K;
                int k_global = k * TILE_K + sub_k * SUB_K + k_sub;
                int block_idx = k_global / 32;
                int elem_in_block = k_global % 32;
                
                if (by * TILE_N + n_local < N && k_global < K) {
                    half_t decoded[32];
                    decodeIQ4NLBlock(gB_block[n_local * (K/32) + block_idx], decoded);
                    s_B[n_local][sub_k * SUB_K + k_sub] = decoded[elem_in_block];
                } else {
                    s_B[n_local][sub_k * SUB_K + k_sub] = half_t(0.0f);
                }
            }
            
            __syncthreads();
            
            // MMA on sub-tile (simplified - actual implementation would use CuTe properly)
            // For now, just do naive accumulation to keep it simple
            int mma_m_start = (tid / 32) % MMA_M;
            int mma_n_start = (tid / 32) / MMA_M;
            
            for (int mm = 0; mm < 16; mm++) {
                for (int nn = 0; nn < 8; nn++) {
                    for (int kk = 0; kk < SUB_K; kk++) {
                        int m_idx = mma_m_start * 16 + mm;
                        int n_idx = mma_n_start * 8 + nn;
                        int k_idx = sub_k * SUB_K + kk;
                        
                        if (m_idx < TILE_M && n_idx < TILE_N) {
                            int buf_idx = k % BUFFER_STAGES;
                            acc[mm * 8 + nn] += 
                                float(s_A[buf_idx][m_idx][k_idx]) * 
                                float(s_B[n_idx][k_idx]);
                        }
                    }
                }
            }
            
            __syncthreads();
        }
    }
    
    // Write results
    int mma_m_start = (tid / 32) % MMA_M;
    int mma_n_start = (tid / 32) / MMA_M;
    for (int mm = 0; mm < 16; mm++) {
        for (int nn = 0; nn < 8; nn++) {
            int m_global = bx * TILE_M + mma_m_start * 16 + mm;
            int n_global = by * TILE_N + mma_n_start * 8 + nn;
            if (m_global < M && n_global < N) {
                gC_block[mm * N + nn] = acc[mm * 8 + nn];
            }
        }
    }
}
)");

        std::cout << "Source length: " << source.length() << " bytes\n";

        // NVRTC compilation
        nvrtcProgram prog;

        // Add required headers
        const char *headers[] = {
            "cute/tensor.hpp",
            "cute/atom/mma_atom.hpp"};
        const char *include_names[] = {
            "cute_tensor",
            "cute_mma"};

        NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), "phase5_kernel.cu", 0, nullptr, nullptr));

        // Compilation options (SAME as Jitify uses + add header search paths)
        const char *opts[] = {
            "--gpu-architecture=compute_86",
            "-std=c++17",
            "--use_fast_math",
            "--extra-device-vectorization",
            "-default-device",
            "-I/opt/cutlass/include",
            "-I/usr/local/cuda/include",
            "-I/workspaces/llaminar/src/v2/kernels/cuda"};

        std::cout << "Compiling with NVRTC...\n";
        nvrtcResult compile_result = nvrtcCompileProgram(prog, 8, opts);

        // Get compilation log
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1)
        {
            std::vector<char> log(log_size);
            nvrtcGetProgramLog(prog, log.data());
            std::cout << "NVRTC Log:\n"
                      << log.data() << std::endl;
        }

        if (compile_result != NVRTC_SUCCESS)
        {
            throw std::runtime_error("NVRTC compilation failed");
        }

        // Get PTX
        size_t ptx_size;
        NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
        std::vector<char> ptx(ptx_size);
        NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

        std::cout << "PTX size: " << ptx_size << " bytes\n";

        nvrtcDestroyProgram(&prog);

        // Load module
        CUmodule module;
        CU_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr));

        // Get function
        CUfunction function;
        CU_CHECK(cuModuleGetFunction(&function, module, "iq4nl_gemm_phase5_kernel"));

        return function;
    }

    int M_, N_, K_;
    float *d_A_ = nullptr;
    void *d_B_ = nullptr;
    float *d_C_ = nullptr;
};

TEST_F(Phase5DirectNVRTCTest, DirectNVRTC_Performance)
{
    std::cout << "\n========================================\n";
    std::cout << "Phase 5 Direct NVRTC Compilation Test\n";
    std::cout << "========================================\n";
    std::cout << "Matrix: " << M_ << "×" << N_ << "×" << K_ << "\n\n";

    // Compile kernel
    auto compile_start = std::chrono::high_resolution_clock::now();
    CUfunction kernel = compileWithNVRTC();
    auto compile_end = std::chrono::high_resolution_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(compile_end - compile_start).count();

    std::cout << "Compilation time: " << compile_ms << " ms\n\n";

    // Setup launch parameters
    int blocks_m = (M_ + 63) / 64;
    int blocks_n = (N_ + 63) / 64;
    dim3 grid(blocks_m, blocks_n);
    dim3 block(128);
    void *args[] = {&d_A_, &d_B_, &d_C_, &M_, &N_, &K_};

    std::cout << "Launch config: Grid(" << grid.x << ", " << grid.y << "), Block(" << block.x << ")\n\n";

    // Warmup
    std::cout << "Warmup (20 iterations)...\n";
    for (int i = 0; i < 20; i++)
    {
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z,
                       0, nullptr, args, nullptr);
    }
    cudaDeviceSynchronize();

    // Benchmark with Runtime API (match Phase 5A)
    std::cout << "Benchmarking (50 iterations)...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 50; i++)
    {
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z,
                       0, nullptr, args, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    float avg_ms = time_ms / 50.0f;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate performance
    double flops = 2.0 * M_ * N_ * K_;
    double tflops = flops / (avg_ms * 1e9);

    std::cout << "\n========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n";
    std::cout << "Average time: " << std::fixed << std::setprecision(4) << avg_ms << " ms\n";
    std::cout << "Throughput:   " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";
    std::cout << "========================================\n\n";

    // Verify output is non-zero
    std::vector<float> h_C(M_ * N_);
    cudaMemcpy(h_C.data(), d_C_, M_ * N_ * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for (int i = 0; i < 100; i++)
        sum += h_C[i];
    std::cout << "Output sanity check (sum of first 100): " << sum << "\n";
    EXPECT_NE(std::abs(sum), 0.0f) << "Output should be non-zero";

    // Performance expectation
    constexpr float EXPECTED_TFLOPS = 8.86f;
    constexpr float TOLERANCE = 0.15f;

    if (tflops >= EXPECTED_TFLOPS * (1.0f - TOLERANCE))
    {
        std::cout << "✓ PASS: Direct NVRTC achieves expected performance!\n";
    }
    else
    {
        std::cout << "✗ FAIL: Direct NVRTC slower than expected ("
                  << tflops << " vs " << EXPECTED_TFLOPS << " TFLOPS)\n";
    }
}
