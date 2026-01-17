/**
 * @file debug_ck_k_crash.hip
 * @brief Debug tool to trace the CK GEMM kernel crash at K > 1408
 * 
 * This file adds printf debugging to understand why DeviceGemmDl crashes
 * when K exceeds ~1408 on MI50 (gfx906).
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>

// Include CK headers
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#define HIP_CHECK(expr) do { \
    hipError_t err = (expr); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace ck_types {
    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;
}

using namespace ck;
using namespace ck::tensor_operation::device;
using namespace ck::tensor_operation::element_wise;
using namespace ck_types;

// Same kernel configuration as in the benchmark
using ALayout = tensor_layout::gemm::ColumnMajor;
using BLayout = tensor_layout::gemm::RowMajor;
using CLayout = tensor_layout::gemm::RowMajor;
using ElementPassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGemmInstance = DeviceGemmDl<
    int8_t, int8_t, int8_t, int32_t,
    ALayout, BLayout, CLayout,
    ElementPassThrough, ElementPassThrough, ElementPassThrough,
    GemmSpecialization::MNPadding,
    256, 128, 128, 16, 4, 4, 4, 1,
    S<2, 8>, S<2, 8>,
    S<2, 1, 4, 4>, S<8, 1, 32, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>,
    S<1, 1, 4, 1>, S<0, 3, 1, 2>, S<1, 1, 4, 4>,
    S<2, 1, 4, 4>, S<8, 1, 32, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>,
    S<1, 1, 4, 1>, S<0, 3, 1, 2>, S<1, 1, 4, 4>,
    S<0, 1, 2, 3, 4, 5>, 5, 4
>;

void printKernelParameters(int M, int N, int K) {
    // Print the loop control parameters that CK calculates
    int K1 = 4;  // From template
    int K0PerBlock = 16;  // From template
    int K0 = K / K1;
    
    int hasMainKBlockLoop = ((K0 + K0PerBlock) / (2 * K0PerBlock)) > 1;
    int hasDoubleTailKBlockLoop = ((K0 / K0PerBlock) % 2) == 0;
    
    int numK0Blocks = K0 / K0PerBlock;
    int mainLoopIterations = hasMainKBlockLoop ? (K0 - 2 * K0PerBlock) / (2 * K0PerBlock) : 0;
    
    printf("\n=== Kernel Parameters for M=%d, N=%d, K=%d ===\n", M, N, K);
    printf("K1 = %d\n", K1);
    printf("K0 = K / K1 = %d\n", K0);
    printf("K0PerBlock = %d\n", K0PerBlock);
    printf("NumK0Blocks = K0 / K0PerBlock = %d\n", numK0Blocks);
    printf("HasMainKBlockLoop = ((%d + %d) / (2 * %d)) > 1 = %d\n", 
           K0, K0PerBlock, K0PerBlock, hasMainKBlockLoop);
    printf("HasDoubleTailKBlockLoop = ((%d / %d) %% 2) == 0 = %d\n",
           K0, K0PerBlock, hasDoubleTailKBlockLoop);
    printf("Main loop iterations (if has main loop): %d\n", mainLoopIterations);
    printf("\n");
    
    // Trace the loop execution
    if (hasMainKBlockLoop) {
        int k_block_data_begin = 0;
        int iteration = 0;
        printf("=== Simulating Main Loop ===\n");
        do {
            printf("  Iteration %d: k_block_data_begin=%d, condition: %d < %d = %s\n",
                   iteration, k_block_data_begin, k_block_data_begin, K0 - 2*K0PerBlock,
                   k_block_data_begin < K0 - 2*K0PerBlock ? "true" : "false");
            
            // Even iteration: accesses block at (k_block_data_begin + K0PerBlock) / K0PerBlock
            int evenBlock = (k_block_data_begin + K0PerBlock) / K0PerBlock;
            // Odd iteration: accesses block at (k_block_data_begin + 2*K0PerBlock) / K0PerBlock
            int oddBlock = (k_block_data_begin + 2*K0PerBlock) / K0PerBlock;
            
            printf("    Even iteration accesses block ~%d, Odd iteration accesses block ~%d\n",
                   evenBlock, oddBlock);
            
            k_block_data_begin += 2 * K0PerBlock;
            iteration++;
            
            if (iteration > 100) {
                printf("    ... (truncated, too many iterations)\n");
                break;
            }
        } while(k_block_data_begin < K0 - 2 * K0PerBlock);
        printf("  Loop exited after %d iterations\n", iteration);
        printf("  Final k_block_data_begin = %d\n", k_block_data_begin);
    }
    
    printf("\n");
}

bool testGemm(int M, int N, int K, bool verbose = true) {
    if (verbose) {
        printKernelParameters(M, N, K);
    }
    
    auto gemm = DeviceGemmInstance{};
    
    size_t size_A = static_cast<size_t>(K) * M;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;
    
    int8_t* d_A = nullptr;
    int8_t* d_B = nullptr;
    int8_t* d_C = nullptr;
    
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&d_B, size_B * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&d_C, size_C * sizeof(int8_t)));
    
    // Initialize with pattern data (easier to debug)
    std::vector<int8_t> h_A(size_A, 1);
    std::vector<int8_t> h_B(size_B, 1);
    
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, size_C * sizeof(int8_t)));
    
    // For column-major M×K matrix: element (m,k) at offset m + k*StrideA
    // StrideA must equal M for contiguous storage (M elements per column)
    // BUG FIX: Was K, which caused element_space_size > allocated buffer
    int StrideA = M;
    int StrideB = N;
    int StrideC = N;
    
    auto argument = gemm.MakeArgument(
        d_A, d_B, d_C,
        M, N, K,
        StrideA, StrideB, StrideC,
        ElementPassThrough{}, ElementPassThrough{}, ElementPassThrough{}
    );
    
    if (!gemm.IsSupportedArgument(argument)) {
        printf("ERROR: Argument not supported by CK\n");
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
        return false;
    }
    
    printf("CK reports argument is supported. Launching kernel...\n");
    fflush(stdout);
    
    auto invoker = gemm.MakeInvoker();
    
    // Enable trap handler for better error reporting
    // Note: This requires HIP_VISIBLE_DEVICES and HSA_TOOLS_LIB to be set properly
    
    hipStream_t stream = nullptr;
    
    // Single kernel launch with sync
    printf("Invoking kernel...\n");
    fflush(stdout);
    
    float kernel_time = invoker.Run(argument, StreamConfig{stream, true}); // timing=true
    
    printf("Kernel returned (time=%f ms), synchronizing...\n", kernel_time);
    fflush(stdout);
    
    hipError_t sync_err = hipDeviceSynchronize();
    
    if (sync_err != hipSuccess) {
        printf("ERROR: hipDeviceSynchronize failed: %s\n", hipGetErrorString(sync_err));
        
        // Try to get more error info
        hipError_t last_err = hipGetLastError();
        printf("Last HIP error: %s\n", hipGetErrorString(last_err));
        
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
        return false;
    }
    
    printf("SUCCESS: Kernel completed without error\n");
    
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    return true;
}

int main(int argc, char* argv[]) {
    // Print device info
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    printf("Found %d HIP devices\n", device_count);
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("Device 0: %s\n", props.name);
    printf("Compute capability: %d.%d\n", props.major, props.minor);
    printf("Total memory: %zu MB\n", props.totalGlobalMem / (1024*1024));
    printf("\n");
    
    int M = 1024;
    int N = 896;
    
    // Test sequence of K values, looking for the boundary
    std::vector<int> K_values;
    
    if (argc > 1) {
        // Test specific K value
        K_values.push_back(atoi(argv[1]));
    } else {
        // Default: test around the boundary
        K_values = {
            896,   // Known working
            1344,  // Known working  
            1408,  // Last known working
            1472,  // First known failing
            1536,  // Failing
            2432,  // Failing (target K for FFN Down)
        };
    }
    
    printf("Testing K values: ");
    for (int k : K_values) printf("%d ", k);
    printf("\n\n");
    
    for (int K : K_values) {
        printf("========================================\n");
        printf("Testing M=%d, N=%d, K=%d\n", M, N, K);
        printf("========================================\n");
        
        bool success = testGemm(M, N, K);
        
        printf("Result: %s\n\n", success ? "PASSED" : "FAILED");
        
        if (!success) {
            printf("Stopped at first failure.\n");
            break;
        }
    }
    
    return 0;
}
