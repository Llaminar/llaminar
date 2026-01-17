/*
 * Debug CK tensor descriptor - print element_space_size and coordinates
 * at various K values to understand where the crash occurs
 */
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstring>

// Include CK headers to get the actual types
#include "ck/ck.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"

using namespace ck;

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Test descriptor creation and offset calculation
void test_descriptor(int M, int K) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing M=" << M << " K=" << K << std::endl;
    std::cout << "========================================" << std::endl;
    
    constexpr int K1 = 4;
    constexpr int MPerBlock = 128;
    constexpr int K0PerBlock = 16;
    
    const int K0 = K / K1;
    
    // Create the base descriptor (M x K row-major)
    const auto a_grid_desc_m_k = make_naive_tensor_descriptor(
        make_tuple(M, K),
        make_tuple(K, 1)  // row-major: stride M is K, stride K is 1
    );
    
    std::cout << "Base descriptor (M x K):" << std::endl;
    std::cout << "  Dimensions: [" << a_grid_desc_m_k.GetLength(Number<0>{}) 
              << " x " << a_grid_desc_m_k.GetLength(Number<1>{}) << "]" << std::endl;
    std::cout << "  ElementSpaceSize: " << a_grid_desc_m_k.GetElementSpaceSize() << std::endl;
    
    // Transform to K0 x M x K1 (like CK does)
    const auto a_grid_desc_k0_m_k1 = transform_tensor_descriptor(
        a_grid_desc_m_k,
        make_tuple(
            make_unmerge_transform(make_tuple(K0, Number<K1>{})),  // K -> K0 x K1
            make_pass_through_transform(M)                         // M -> M
        ),
        make_tuple(Sequence<1>{}, Sequence<0>{}),   // old dims: K, M
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}) // new dims: K0, K1 at 0,2; M at 1
    );
    
    std::cout << "\nTransformed descriptor (K0 x M x K1):" << std::endl;
    std::cout << "  K0 = " << a_grid_desc_k0_m_k1.GetLength(Number<0>{}) << std::endl;
    std::cout << "  M  = " << a_grid_desc_k0_m_k1.GetLength(Number<1>{}) << std::endl;
    std::cout << "  K1 = " << a_grid_desc_k0_m_k1.GetLength(Number<2>{}) << std::endl;
    std::cout << "  ElementSpaceSize: " << a_grid_desc_k0_m_k1.GetElementSpaceSize() << std::endl;
    
    // Further transform to K0 x M0 x M1 x K1 (block tiling)
    const int M1 = MPerBlock;
    const int M0 = M / M1;
    
    const auto a_grid_desc_k0_m0_m1_k1 = transform_tensor_descriptor(
        a_grid_desc_k0_m_k1,
        make_tuple(
            make_pass_through_transform(K0),
            make_unmerge_transform(make_tuple(M0, Number<MPerBlock>{})),
            make_pass_through_transform(Number<K1>{})
        ),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{})
    );
    
    std::cout << "\nBlock-tiled descriptor (K0 x M0 x M1 x K1):" << std::endl;
    std::cout << "  K0 = " << a_grid_desc_k0_m0_m1_k1.GetLength(Number<0>{}) << std::endl;
    std::cout << "  M0 = " << a_grid_desc_k0_m0_m1_k1.GetLength(Number<1>{}) << std::endl;
    std::cout << "  M1 = " << a_grid_desc_k0_m0_m1_k1.GetLength(Number<2>{}) << std::endl;
    std::cout << "  K1 = " << a_grid_desc_k0_m0_m1_k1.GetLength(Number<3>{}) << std::endl;
    std::cout << "  ElementSpaceSize: " << a_grid_desc_k0_m0_m1_k1.GetElementSpaceSize() << std::endl;
    
    // Calculate coordinates at various K0 positions
    std::cout << "\nCoordinate test at various K0 positions:" << std::endl;
    
    int num_k_blocks = K0 / K0PerBlock;  // How many K blocks
    int main_loop_iters = num_k_blocks / 2;  // Main loop does 2 blocks per iter
    bool has_tail = (num_k_blocks % 2) != 0;
    
    std::cout << "  num_k_blocks = " << num_k_blocks << std::endl;
    std::cout << "  main_loop_iters = " << main_loop_iters << std::endl;
    std::cout << "  has_tail = " << (has_tail ? "yes" : "no") << std::endl;
    
    // Check coordinate at max k0 position
    int max_k0 = K0 - 1;
    std::cout << "\n  Max K0 index: " << max_k0 << std::endl;
    
    // Create coordinate at origin
    auto coord = make_tensor_coordinate(a_grid_desc_k0_m0_m1_k1, make_multi_index(0, 0, 0, 0));
    std::cout << "  Offset at (0,0,0,0): " << coord.GetOffset() << std::endl;
    
    // Move to max K0
    auto coord_max = make_tensor_coordinate(a_grid_desc_k0_m0_m1_k1, make_multi_index(max_k0, 0, 0, 0));
    std::cout << "  Offset at (" << max_k0 << ",0,0,0): " << coord_max.GetOffset() << std::endl;
    
    // Check if max offset is within bounds
    int64_t element_space_size = a_grid_desc_k0_m0_m1_k1.GetElementSpaceSize();
    std::cout << "\n  Max offset (" << coord_max.GetOffset() << ") < element_space_size (" 
              << element_space_size << "): " 
              << (coord_max.GetOffset() < element_space_size ? "YES" : "NO!!!") << std::endl;
    
    // Check corner cases
    auto coord_corner = make_tensor_coordinate(a_grid_desc_k0_m0_m1_k1, 
        make_multi_index(max_k0, M0-1, MPerBlock-1, K1-1));
    std::cout << "  Offset at (" << max_k0 << "," << (M0-1) << "," << (MPerBlock-1) << "," << (K1-1) 
              << "): " << coord_corner.GetOffset() << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "CK Tensor Descriptor Debug Tool" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Test various K values
    test_descriptor(128, 896);   // Works
    test_descriptor(128, 1344);  // Works  
    test_descriptor(128, 1408);  // Works (boundary)
    test_descriptor(128, 1472);  // CRASHES
    test_descriptor(128, 1536);  // CRASHES
    
    return 0;
}
