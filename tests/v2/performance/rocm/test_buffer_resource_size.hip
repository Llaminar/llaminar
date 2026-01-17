/**
 * @file test_buffer_resource_size.hip
 * @brief Test to verify buffer resource sizes and check for overflow
 * 
 * The hypothesis is that the buffer resource might overflow when K is large.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

#define HIP_CHECK(expr) do { \
    hipError_t err = (expr); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Test kernel that mimics what CK does with buffer resources
__global__ void testBufferAccess(
    const int8_t* __restrict__ A,  // [K, M] column-major
    int M, int K,
    int K0PerBlock,  // = 16
    int K1,          // = 4
    int* result)     // Output: 0 = success, >0 = problem iteration
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid != 0) return;  // Only one thread does the test
    
    int K0 = K / K1;  // Number of K0 blocks
    
    // Calculate buffer size as CK does
    // For A[K,M] column-major: element_space_size = K * M
    long element_space_size = (long)K * M;
    
    printf("[GPU] K=%d, M=%d, K0=%d, K0PerBlock=%d, K1=%d\n", K, M, K0, K0PerBlock, K1);
    printf("[GPU] element_space_size = %ld\n", element_space_size);
    printf("[GPU] buffer_size_bytes = %ld\n", element_space_size * sizeof(int8_t));
    
    // Check if buffer size fits in 32-bit
    long buffer_size_bytes = element_space_size * sizeof(int8_t);
    if (buffer_size_bytes > 0xFFFFFFFF) {
        printf("[GPU] ERROR: buffer_size_bytes exceeds 32-bit!\n");
        *result = -1;
        return;
    }
    printf("[GPU] buffer_size_bytes fits in 32-bit: OK\n");
    
    // Simulate the loop and check maximum offset accessed
    // The kernel accesses elements at various offsets during the K loop
    
    // For A[K,M] column-major with K0_M0_M1_K1 descriptor:
    // - K0 dimension: 0 to K/K1 - 1
    // - M0 dimension: derived from M
    // - M1 dimension: derived from M  
    // - K1 dimension: 0 to K1 - 1
    
    // Maximum K0 index accessed in the loop
    int hasMainKBlockLoop = ((K0 + K0PerBlock) / (2 * K0PerBlock)) > 1;
    int hasDoubleTailKBlockLoop = ((K0 / K0PerBlock) % 2) == 0;
    
    printf("[GPU] HasMainKBlockLoop=%d, HasDoubleTailKBlockLoop=%d\n", 
           hasMainKBlockLoop, hasDoubleTailKBlockLoop);
    
    // Maximum K0 block accessed
    int max_k0_block = 0;
    
    // Initial prefetch: block 0
    max_k0_block = 0;
    
    if (hasMainKBlockLoop) {
        int k_block_data_begin = 0;
        int iteration = 0;
        
        do {
            // Even iteration accesses: k_block_data_begin + K0PerBlock
            int even_k0 = k_block_data_begin + K0PerBlock;
            // Odd iteration accesses: k_block_data_begin + 2*K0PerBlock
            int odd_k0 = k_block_data_begin + 2 * K0PerBlock;
            
            if (even_k0 > max_k0_block) max_k0_block = even_k0;
            if (odd_k0 > max_k0_block) max_k0_block = odd_k0;
            
            k_block_data_begin += 2 * K0PerBlock;
            iteration++;
        } while(k_block_data_begin < K0 - 2 * K0PerBlock);
        
        printf("[GPU] Main loop iterations: %d, final k_block_data_begin=%d\n", 
               iteration, k_block_data_begin);
    }
    
    // Tail iterations
    if (hasDoubleTailKBlockLoop) {
        // Accesses block at K0 - K0PerBlock
        int tail_k0 = K0 - K0PerBlock;
        if (tail_k0 > max_k0_block) max_k0_block = tail_k0;
    }
    
    // Final tail accesses block at K0 - 1
    if (K0 - 1 > max_k0_block) max_k0_block = K0 - 1;
    
    printf("[GPU] max_k0_block (K0 index) accessed: %d\n", max_k0_block);
    printf("[GPU] K0 total: %d\n", K0);
    
    // Check if max_k0_block is in bounds
    if (max_k0_block >= K0) {
        printf("[GPU] ERROR: max_k0_block (%d) >= K0 (%d)!\n", max_k0_block, K0);
        *result = max_k0_block;
        return;
    }
    printf("[GPU] max_k0_block < K0: OK\n");
    
    // Calculate maximum byte offset that would be accessed
    // For each thread, it accesses multiple elements within a K block
    // The maximum offset is approximately: max_k0_block * K1 * M + (M-1) * K + (K1-1)
    // But this depends on the thread's position...
    
    // Let's calculate the maximum linear element offset
    // For A[K0,M0,M1,K1] with K0 being the slowest changing:
    // Linear offset = k0 * (M * K1) + m0 * (M1 * K1) + m1 * K1 + k1
    // Maximum offset = (K0-1) * (M * K1) + (M0-1) * (M1 * K1) + (M1-1) * K1 + (K1-1)
    // Simplified: approximately K * M - 1
    
    long max_element_offset = (long)K * M - 1;
    printf("[GPU] max_element_offset = %ld\n", max_element_offset);
    
    // Verify we can actually read the maximum offset
    printf("[GPU] Attempting to read at offset 0...\n");
    int8_t v0 = A[0];
    printf("[GPU] A[0] = %d OK\n", (int)v0);
    
    printf("[GPU] Attempting to read at last valid offset (%ld)...\n", max_element_offset);
    int8_t vmax = A[max_element_offset];
    printf("[GPU] A[%ld] = %d OK\n", max_element_offset, (int)vmax);
    
    *result = 0;
    printf("[GPU] All checks passed!\n");
}

int main(int argc, char* argv[]) {
    int M = 1024;
    int K = 1472;  // Default: first failing K
    int K0PerBlock = 16;
    int K1 = 4;
    
    if (argc > 1) K = atoi(argv[1]);
    
    printf("Testing M=%d, K=%d\n", M, K);
    printf("K0 = K/K1 = %d\n", K / K1);
    printf("K0PerBlock = %d, K1 = %d\n", K0PerBlock, K1);
    
    size_t size_A = (size_t)K * M;
    printf("Buffer size: %zu bytes\n", size_A);
    
    int8_t* d_A = nullptr;
    int* d_result = nullptr;
    
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));
    
    // Initialize with non-zero pattern
    std::vector<int8_t> h_A(size_A, 42);
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_result, 0xFF, sizeof(int)));
    
    printf("\nLaunching test kernel...\n");
    fflush(stdout);
    
    testBufferAccess<<<1, 1>>>(d_A, M, K, K0PerBlock, K1, d_result);
    
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("ERROR: Kernel failed: %s\n", hipGetErrorString(err));
    }
    
    int h_result;
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    printf("\nResult: %d\n", h_result);
    
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_result));
    
    return h_result != 0 ? 1 : 0;
}
