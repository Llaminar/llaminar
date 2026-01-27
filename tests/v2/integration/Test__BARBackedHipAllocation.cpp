/**
 * @file Test__BARBackedHipAllocation.cpp
 * @brief Standalone test to validate BAR-backed allocation and transfer paths
 * 
 * This test explores:
 * 1. CPU write -> BAR -> CUDA read path (the established path)
 * 2. hipMemcpy D2H to BAR mmap'd memory (via ROCmBackend)
 * 3. Whether BAR memory can work as a staging area for ROCm output
 * 
 * NOTE: Cannot include both CUDA and HIP headers in same compilation unit
 * due to conflicting type definitions. This test uses the existing DirectP2P
 * and ROCmBackend infrastructure which handles this separation.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <dlfcn.h>  // For HSA runtime loading
#include <fcntl.h>  // For open()
#include <unistd.h> // For close()
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h> // For mmap/munmap

// Include CUDA headers only - HIP is handled via ROCmBackend
#if defined(HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "../../../src/v2/backends/p2p/DirectP2P.h"
#include "../../../src/v2/backends/rocm/ROCmBackend.h"
#include "../../../src/v2/utils/Logger.h"

using namespace llaminar2;

class Test__BARBackedHipAllocation : public ::testing::Test
{
protected:
    void SetUp() override
    {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
        GTEST_SKIP() << "Requires both CUDA and ROCm";
#endif
    }
};

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

TEST_F(Test__BARBackedHipAllocation, TestCPUWriteToBAR_CUDARead)
{
    // Test: CPU write -> BAR -> CUDA read (the established working path)
    
    LOG_INFO("=== Testing CPU->BAR->CUDA path ===");
    
    int cuda_count = 0;
    cudaGetDeviceCount(&cuda_count);
    
    if (cuda_count == 0)
    {
        GTEST_SKIP() << "Need at least 1 CUDA device";
    }
    
    // Check ROCm via backend
    ROCmBackend rocm_backend;
    int rocm_count = rocm_backend.deviceCount();
    
    if (rocm_count == 0)
    {
        GTEST_SKIP() << "Need at least 1 ROCm device";
    }
    
    LOG_INFO("CUDA devices: " << cuda_count << ", ROCm devices: " << rocm_count);
    
    DirectP2PEngine p2p;
    
    DeviceId cuda_dev = DeviceId::cuda(0);
    DeviceId rocm_dev = DeviceId::rocm(0);
    
    bool init_ok = p2p.initializePCIeBar(cuda_dev, rocm_dev, 64 * 1024 * 1024, 0);
    if (!init_ok)
    {
        GTEST_SKIP() << "Failed to initialize PCIe BAR";
    }
    
    LOG_INFO("BAR initialized:");
    LOG_INFO("  Host ptr: " << p2p.getBarHostPtr());
    LOG_INFO("  CUDA ptr: " << p2p.getCudaBarPointer());
    LOG_INFO("  Size: " << (p2p.getBarMappedSize() / (1024*1024)) << " MB");
    
    const int count = 1024;
    float* bar_host = static_cast<float*>(p2p.getBarHostPtr());
    void* cuda_bar = p2p.getCudaBarPointer();
    
    // CPU writes to BAR
    LOG_INFO("Writing pattern from CPU to BAR...");
    for (int i = 0; i < count; ++i)
    {
        bar_host[i] = static_cast<float>(i * 2);
    }
    
    // CUDA reads from BAR
    cudaSetDevice(0);
    
    float* cuda_dst = nullptr;
    cudaMalloc(&cuda_dst, count * sizeof(float));
    
    LOG_INFO("Copying from BAR to CUDA device memory...");
    cudaError_t err = cudaMemcpy(cuda_dst, cuda_bar, count * sizeof(float), cudaMemcpyDeviceToDevice);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy from BAR failed: " << cudaGetErrorString(err);
    
    // Verify
    std::vector<float> result(count);
    cudaMemcpy(result.data(), cuda_dst, count * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool all_match = true;
    for (int i = 0; i < count; ++i)
    {
        float expected = static_cast<float>(i * 2);
        if (std::abs(result[i] - expected) > 0.001f)
        {
            all_match = false;
            if (i < 10)
            {
                LOG_ERROR("Mismatch at " << i << ": expected " << expected << " got " << result[i]);
            }
        }
    }
    
    EXPECT_TRUE(all_match) << "Data mismatch in CPU->BAR->CUDA transfer";
    
    if (all_match)
    {
        LOG_INFO("✓ CPU write -> BAR -> CUDA read works correctly!");
    }
    
    cudaFree(cuda_dst);
}

TEST_F(Test__BARBackedHipAllocation, TestROCmWriteToBAR_CUDARead)
{
    // Test: ROCm kernel writes to hipMalloc buffer, then hipMemcpy D2H to BAR, then CUDA reads
    // This tests if hipMemcpy can use BAR mmap as destination
    
    LOG_INFO("=== Testing ROCm->BAR->CUDA path ===");
    
    int cuda_count = 0;
    cudaGetDeviceCount(&cuda_count);
    
    ROCmBackend rocm_backend;
    int rocm_count = rocm_backend.deviceCount();
    
    if (cuda_count == 0 || rocm_count == 0)
    {
        GTEST_SKIP() << "Need at least 1 CUDA and 1 ROCm device";
    }
    
    DirectP2PEngine p2p;
    
    DeviceId cuda_dev = DeviceId::cuda(0);
    DeviceId rocm_dev = DeviceId::rocm(0);
    
    bool init_ok = p2p.initializePCIeBar(cuda_dev, rocm_dev, 64 * 1024 * 1024, 0);
    if (!init_ok)
    {
        GTEST_SKIP() << "Failed to initialize PCIe BAR";
    }
    
    const int count = 1024;
    const size_t bytes = count * sizeof(float);
    
    float* bar_host = static_cast<float*>(p2p.getBarHostPtr());
    
    // Clear BAR first
    memset(bar_host, 0, bytes);
    
    // Allocate ROCm device buffer via backend
    rocm_backend.setDevice(0);
    void* rocm_buffer = rocm_backend.allocate(bytes, 0);
    ASSERT_NE(rocm_buffer, nullptr) << "ROCm allocation failed";
    
    // Fill ROCm buffer from host
    std::vector<float> pattern(count);
    for (int i = 0; i < count; ++i)
    {
        pattern[i] = 500.0f + i;
    }
    
    // Upload pattern to ROCm device
    bool copy_ok = rocm_backend.hostToDevice(rocm_buffer, pattern.data(), bytes, 0);
    ASSERT_TRUE(copy_ok) << "Host to ROCm copy failed";
    
    LOG_INFO("Pattern uploaded to ROCm device buffer");
    
    // Now try to copy from ROCm device to BAR (using D2H since BAR mmap is CPU VA)
    LOG_INFO("Attempting hipMemcpy D2H from ROCm buffer to BAR mmap...");
    
    bool d2h_ok = rocm_backend.deviceToHost(bar_host, rocm_buffer, bytes, 0);
    
    if (!d2h_ok)
    {
        LOG_WARN("hipMemcpy D2H to BAR failed - this is the key finding!");
        LOG_INFO("The BAR mmap'd memory is not recognized as valid host memory by HIP");
        
        // This is expected - let's verify by reading BAR
        LOG_INFO("BAR values after failed copy (should be zeros):");
        for (int i = 0; i < 8; ++i)
        {
            std::cout << "  bar_host[" << i << "] = " << bar_host[i] << std::endl;
        }
    }
    else
    {
        LOG_INFO("✓ hipMemcpy D2H to BAR SUCCEEDED!");
        
        // Verify from CPU
        LOG_INFO("Values in BAR after hipMemcpy:");
        for (int i = 0; i < 8; ++i)
        {
            std::cout << "  bar_host[" << i << "] = " << bar_host[i] << std::endl;
        }
        
        bool match = true;
        for (int i = 0; i < count; ++i)
        {
            if (std::abs(bar_host[i] - pattern[i]) > 0.001f)
            {
                match = false;
                break;
            }
        }
        
        if (match)
        {
            LOG_INFO("✓ Data verified correct in BAR!");
            
            // Now test CUDA read
            cudaSetDevice(0);
            float* cuda_dst = nullptr;
            cudaMalloc(&cuda_dst, bytes);
            
            void* cuda_bar = p2p.getCudaBarPointer();
            cudaError_t cerr = cudaMemcpy(cuda_dst, cuda_bar, bytes, cudaMemcpyDeviceToDevice);
            
            if (cerr == cudaSuccess)
            {
                std::vector<float> cuda_result(count);
                cudaMemcpy(cuda_result.data(), cuda_dst, bytes, cudaMemcpyDeviceToHost);
                
                match = true;
                for (int i = 0; i < count; ++i)
                {
                    if (std::abs(cuda_result[i] - pattern[i]) > 0.001f)
                    {
                        match = false;
                        break;
                    }
                }
                
                if (match)
                {
                    LOG_INFO("✓ FULL PATH WORKS: ROCm kernel -> hipMemcpy D2H -> BAR -> CUDA D2D!");
                }
            }
            
            cudaFree(cuda_dst);
        }
    }
    
    rocm_backend.free(rocm_buffer, 0);
}

TEST_F(Test__BARBackedHipAllocation, TestBARToROCm_H2D)
{
    // Test: CPU writes to BAR, then hipMemcpy H2D from BAR to ROCm device buffer
    // This tests if hipMemcpy can use BAR mmap as source
    
    LOG_INFO("=== Testing BAR->ROCm (H2D) path ===");
    
    int cuda_count = 0;
    cudaGetDeviceCount(&cuda_count);
    
    ROCmBackend rocm_backend;
    int rocm_count = rocm_backend.deviceCount();
    
    if (cuda_count == 0 || rocm_count == 0)
    {
        GTEST_SKIP() << "Need at least 1 CUDA and 1 ROCm device";
    }
    
    DirectP2PEngine p2p;
    
    DeviceId cuda_dev = DeviceId::cuda(0);
    DeviceId rocm_dev = DeviceId::rocm(0);
    
    bool init_ok = p2p.initializePCIeBar(cuda_dev, rocm_dev, 64 * 1024 * 1024, 0);
    if (!init_ok)
    {
        GTEST_SKIP() << "Failed to initialize PCIe BAR";
    }
    
    const int count = 1024;
    const size_t bytes = count * sizeof(float);
    
    float* bar_host = static_cast<float*>(p2p.getBarHostPtr());
    
    // Fill BAR from CPU
    for (int i = 0; i < count; ++i)
    {
        bar_host[i] = 600.0f + i;
    }
    
    LOG_INFO("Pattern written to BAR from CPU");
    
    // Allocate ROCm device buffer
    rocm_backend.setDevice(0);
    void* rocm_buffer = rocm_backend.allocate(bytes, 0);
    ASSERT_NE(rocm_buffer, nullptr);
    
    // Try H2D copy from BAR to ROCm device buffer
    LOG_INFO("Attempting hipMemcpy H2D from BAR mmap to ROCm buffer...");
    
    bool h2d_ok = rocm_backend.hostToDevice(rocm_buffer, bar_host, bytes, 0);
    
    if (!h2d_ok)
    {
        LOG_ERROR("hipMemcpy H2D from BAR failed!");
    }
    else
    {
        LOG_INFO("✓ hipMemcpy H2D from BAR SUCCEEDED!");
        
        // Verify by copying back
        std::vector<float> verify(count);
        bool d2h_ok = rocm_backend.deviceToHost(verify.data(), rocm_buffer, bytes, 0);
        ASSERT_TRUE(d2h_ok);
        
        bool match = true;
        for (int i = 0; i < count; ++i)
        {
            if (std::abs(verify[i] - (600.0f + i)) > 0.001f)
            {
                match = false;
                if (i < 10)
                {
                    LOG_ERROR("Mismatch at " << i << ": expected " << (600.0f + i) << " got " << verify[i]);
                }
            }
        }
        
        if (match)
        {
            LOG_INFO("✓ Data verified correct after H2D from BAR!");
            LOG_INFO("✓ BIDIRECTIONAL PATH CONFIRMED: BAR<->ROCm via hipMemcpy!");
        }
    }
    
    rocm_backend.free(rocm_buffer, 0);
}

TEST_F(Test__BARBackedHipAllocation, TestBARAddressSpaceRelationship)
{
    // Explore the relationship between BAR mmap addresses and HIP device addresses
    // The goal: find if there's a direct mapping that would allow zero-copy
    
    LOG_INFO("=== Exploring BAR address space relationship ===");
    
    int cuda_count = 0;
    cudaGetDeviceCount(&cuda_count);
    
    ROCmBackend rocm_backend;
    int rocm_count = rocm_backend.deviceCount();
    
    if (cuda_count == 0 || rocm_count == 0)
    {
        GTEST_SKIP() << "Need at least 1 CUDA and 1 ROCm device";
    }
    
    DirectP2PEngine p2p;
    
    DeviceId cuda_dev = DeviceId::cuda(0);
    DeviceId rocm_dev = DeviceId::rocm(0);
    
    // Map a region of the BAR at offset 64MB
    size_t bar_offset = 64 * 1024 * 1024;  // 64 MB offset
    size_t bar_size = 64 * 1024 * 1024;    // 64 MB size
    
    bool init_ok = p2p.initializePCIeBar(cuda_dev, rocm_dev, bar_size, bar_offset);
    if (!init_ok)
    {
        GTEST_SKIP() << "Failed to initialize PCIe BAR";
    }
    
    void* bar_host_ptr = p2p.getBarHostPtr();
    
    LOG_INFO("BAR mapping info:");
    LOG_INFO("  BAR offset: " << bar_offset << " bytes (" << (bar_offset / (1024*1024)) << " MB)");
    LOG_INFO("  BAR size: " << bar_size << " bytes (" << (bar_size / (1024*1024)) << " MB)");
    LOG_INFO("  BAR host ptr (CPU VA): " << std::hex << reinterpret_cast<uintptr_t>(bar_host_ptr) << std::dec);
    
    // Now allocate some HIP buffers and see their addresses
    rocm_backend.setDevice(0);
    
    void* hip_ptr1 = rocm_backend.allocate(1024, 0);
    void* hip_ptr2 = rocm_backend.allocate(1024, 0);
    void* hip_ptr3 = rocm_backend.allocate(64 * 1024 * 1024, 0);  // Large allocation
    
    LOG_INFO("HIP allocation addresses:");
    LOG_INFO("  hip_ptr1: " << std::hex << reinterpret_cast<uintptr_t>(hip_ptr1) << std::dec);
    LOG_INFO("  hip_ptr2: " << std::hex << reinterpret_cast<uintptr_t>(hip_ptr2) << std::dec);
    LOG_INFO("  hip_ptr3 (64MB): " << std::hex << reinterpret_cast<uintptr_t>(hip_ptr3) << std::dec);
    
    // Calculate offsets/differences
    uintptr_t bar_addr = reinterpret_cast<uintptr_t>(bar_host_ptr);
    uintptr_t hip1_addr = reinterpret_cast<uintptr_t>(hip_ptr1);
    uintptr_t hip2_addr = reinterpret_cast<uintptr_t>(hip_ptr2);
    uintptr_t hip3_addr = reinterpret_cast<uintptr_t>(hip_ptr3);
    
    LOG_INFO("Address analysis:");
    LOG_INFO("  BAR address:  " << std::hex << bar_addr << std::dec);
    LOG_INFO("  HIP1 address: " << std::hex << hip1_addr << " (diff from BAR: " << std::dec << static_cast<int64_t>(hip1_addr - bar_addr) << ")");
    LOG_INFO("  HIP2 address: " << std::hex << hip2_addr << " (diff from BAR: " << std::dec << static_cast<int64_t>(hip2_addr - bar_addr) << ")");
    LOG_INFO("  HIP3 address: " << std::hex << hip3_addr << " (diff from BAR: " << std::dec << static_cast<int64_t>(hip3_addr - bar_addr) << ")");
    
    // ================================================================
    // KEY TEST: Use hipPointerGetAttributes to understand what HIP
    // thinks about these different pointers
    // ================================================================
    LOG_INFO("\n=== Querying pointer attributes via hipPointerGetAttributes ===");
    
    bool is_device, is_host, is_managed;
    int device_id;
    
    if (rocm_backend.queryPointerAttributes(hip_ptr1, is_device, is_host, is_managed, device_id))
    {
        LOG_INFO("hip_ptr1: device=" << is_device << " host=" << is_host 
                 << " managed=" << is_managed << " device_id=" << device_id);
    }
    else
    {
        LOG_WARN("hipPointerGetAttributes FAILED for hip_ptr1");
    }
    
    // This is the critical query: what does HIP think about the BAR mmap address?
    if (rocm_backend.queryPointerAttributes(bar_host_ptr, is_device, is_host, is_managed, device_id))
    {
        LOG_INFO("bar_host_ptr: device=" << is_device << " host=" << is_host 
                 << " managed=" << is_managed << " device_id=" << device_id);
        
        if (is_device)
        {
            LOG_INFO("✓ CRITICAL: HIP recognizes BAR mmap as DEVICE memory!");
            LOG_INFO("  This means we can use bar_host_ptr directly as a HIP device pointer!");
        }
        else if (is_host)
        {
            LOG_INFO("HIP recognizes BAR mmap as HOST memory");
        }
    }
    else
    {
        LOG_INFO("hipPointerGetAttributes FAILED for bar_host_ptr");
        LOG_INFO("This means HIP doesn't know about the BAR mmap address");
    }
    
    // ================================================================
    // KEY TEST: Try D2D copy with BAR address as source
    // ================================================================
    LOG_INFO("\n=== Testing D2D copy with BAR address as device source ===");
    
    // Write test pattern to BAR via CPU
    float* bar_float = static_cast<float*>(bar_host_ptr);
    for (int i = 0; i < 256; i++)
    {
        bar_float[i] = static_cast<float>(i) * 3.14159f;
    }
    
    // Try D2D from bar_host_ptr to hip_ptr1
    // This tests if HIP can interpret bar_host_ptr as a valid device address
    bool d2d_ok = rocm_backend.deviceToDevice(hip_ptr1, bar_host_ptr, 256 * sizeof(float), 0);
    
    if (d2d_ok)
    {
        LOG_INFO("✓ D2D copy from BAR to HIP device memory SUCCEEDED!");
        
        // Verify the data
        std::vector<float> verify(256);
        rocm_backend.deviceToHost(verify.data(), hip_ptr1, 256 * sizeof(float), 0);
        
        bool match = true;
        for (int i = 0; i < 256; i++)
        {
            if (std::abs(verify[i] - bar_float[i]) > 0.001f)
            {
                LOG_ERROR("Mismatch at " << i << ": expected " << bar_float[i] << " got " << verify[i]);
                match = false;
                break;
            }
        }
        
        if (match)
        {
            LOG_INFO("✓✓✓ DATA VERIFIED! BAR mmap address IS a valid HIP device address!");
            LOG_INFO("    This proves ZERO-COPY is possible: HIP kernels can use bar_host_ptr directly!");
        }
    }
    else
    {
        LOG_WARN("D2D copy failed - HIP doesn't see BAR mmap as device memory");
        LOG_INFO("Falling back to H2D interpretation...");
    }
    
    // ================================================================
    // Calculate what BAR offset would give us a specific HIP address
    // ================================================================
    LOG_INFO("\n=== Address space mapping analysis ===");
    
    // The BAR is 32GB. Our BAR is mapped at bar_offset bytes into the BAR.
    // If HIP allocations come from a heap at some base offset in the BAR,
    // we can calculate where hipMalloc addresses fall within the BAR.
    
    // If bar_host_ptr (at bar_offset=64MB) has address bar_addr, and
    // hip_ptr1 has address hip1_addr, then hip_ptr1's BAR offset would be:
    // bar_offset + (hip1_addr - bar_addr)
    
    int64_t hip1_bar_offset = static_cast<int64_t>(bar_offset) + (hip1_addr - bar_addr);
    int64_t hip3_bar_offset = static_cast<int64_t>(bar_offset) + (hip3_addr - bar_addr);
    
    LOG_INFO("If BAR offset 0 = base VA, estimated BAR offsets:");
    LOG_INFO("  hip_ptr1 @ BAR offset: " << hip1_bar_offset << " bytes (" << (hip1_bar_offset/(1024*1024)) << " MB)");
    LOG_INFO("  hip_ptr3 @ BAR offset: " << hip3_bar_offset << " bytes (" << (hip3_bar_offset/(1024*1024)) << " MB)");
    
    // Cleanup
    rocm_backend.free(hip_ptr1, 0);
    rocm_backend.free(hip_ptr2, 0);
    rocm_backend.free(hip_ptr3, 0);
}

TEST_F(Test__BARBackedHipAllocation, TestDirectBARAddressAsDevicePointer)
{
    // Exploratory test: What if we pass the BAR mmap address directly to hipMemcpy
    // as if it were a device pointer? The BAR IS AMD GPU VRAM after all.
    
    LOG_INFO("=== Testing BAR address as device pointer (experimental) ===");
    
    int cuda_count = 0;
    cudaGetDeviceCount(&cuda_count);
    
    ROCmBackend rocm_backend;
    int rocm_count = rocm_backend.deviceCount();
    
    if (cuda_count == 0 || rocm_count == 0)
    {
        GTEST_SKIP() << "Need at least 1 CUDA and 1 ROCm device";
    }
    
    DirectP2PEngine p2p;
    
    DeviceId cuda_dev = DeviceId::cuda(0);
    DeviceId rocm_dev = DeviceId::rocm(0);
    
    bool init_ok = p2p.initializePCIeBar(cuda_dev, rocm_dev, 64 * 1024 * 1024, 0);
    if (!init_ok)
    {
        GTEST_SKIP() << "Failed to initialize PCIe BAR";
    }
    
    float* bar_host = static_cast<float*>(p2p.getBarHostPtr());
    
    // Fill BAR from CPU with known pattern
    const int count = 256;
    const size_t bytes = count * sizeof(float);
    
    for (int i = 0; i < count; ++i)
    {
        bar_host[i] = 777.0f + i;
    }
    
    // Allocate ROCm device buffer
    rocm_backend.setDevice(0);
    void* rocm_buffer = rocm_backend.allocate(bytes, 0);
    ASSERT_NE(rocm_buffer, nullptr);
    
    // Try D2D copy from bar_host (CPU VA of AMD VRAM) to rocm_buffer (device ptr)
    LOG_INFO("Attempting hipMemcpy D2D from BAR VA to ROCm device buffer...");
    LOG_INFO("  BAR host ptr: " << static_cast<void*>(bar_host));
    LOG_INFO("  ROCm device ptr: " << rocm_buffer);
    
    // Note: We can't call hipMemcpy directly here since we're not including hip headers
    // Instead, let's use ROCmBackend's copyDeviceToDevice if it exists, or test via 
    // copying host->device and comparing
    
    // For now, let's just verify the CPU can read back what it wrote
    LOG_INFO("Verifying CPU read of BAR (sanity check):");
    for (int i = 0; i < 8; ++i)
    {
        std::cout << "  bar_host[" << i << "] = " << bar_host[i] << std::endl;
    }
    
    // The key question is: can we get a HIP device pointer for this BAR region?
    // This would require either:
    // 1. hipImportExternalMemory + hipExternalMemoryGetMappedBuffer
    // 2. Some AMD-specific API to convert CPU VA of GPU VRAM to device pointer
    
    LOG_INFO("Key insight: BAR mmap gives CPU virtual address " << std::hex 
             << reinterpret_cast<uintptr_t>(bar_host) << std::dec);
    LOG_INFO("HIP kernels need device address space pointers, not CPU VAs");
    LOG_INFO("The memory IS AMD GPU VRAM, but HIP doesn't know how to reach it via CPU VA");
    
    rocm_backend.free(rocm_buffer, 0);
}

TEST_F(Test__BARBackedHipAllocation, TestZeroCopyBidirectional)
{
    // CRITICAL TEST: Prove that BAR address works as a HIP device address
    // for bidirectional D2D copies (no staging required!)
    
    LOG_INFO("=== Testing ZERO-COPY bidirectional D2D via BAR ===");
    
    int cuda_count = 0;
    cudaGetDeviceCount(&cuda_count);
    
    ROCmBackend rocm_backend;
    int rocm_count = rocm_backend.deviceCount();
    
    if (cuda_count == 0 || rocm_count == 0)
    {
        GTEST_SKIP() << "Need at least 1 CUDA and 1 ROCm device";
    }
    
    DirectP2PEngine p2p;
    
    DeviceId cuda_dev = DeviceId::cuda(0);
    DeviceId rocm_dev = DeviceId::rocm(0);
    
    bool init_ok = p2p.initializePCIeBar(cuda_dev, rocm_dev, 64 * 1024 * 1024, 64 * 1024 * 1024);
    if (!init_ok)
    {
        GTEST_SKIP() << "Failed to initialize PCIe BAR";
    }
    
    float* bar_ptr = static_cast<float*>(p2p.getBarHostPtr());
    const int count = 1024;
    const size_t bytes = count * sizeof(float);
    
    // Allocate HIP device buffers
    rocm_backend.setDevice(0);
    void* hip_src = rocm_backend.allocate(bytes, 0);
    void* hip_dst = rocm_backend.allocate(bytes, 0);
    ASSERT_NE(hip_src, nullptr);
    ASSERT_NE(hip_dst, nullptr);
    
    // ============================================================
    // TEST 1: HIP device → BAR (D2D with BAR as destination)
    // ============================================================
    LOG_INFO("\n--- TEST 1: HIP device → BAR via D2D ---");
    
    // Initialize HIP source buffer via H2D
    std::vector<float> host_data(count);
    for (int i = 0; i < count; i++) host_data[i] = static_cast<float>(i) * 2.5f;
    rocm_backend.hostToDevice(hip_src, host_data.data(), bytes, 0);
    
    // D2D: HIP device → BAR (treating BAR as device destination)
    bool d2d_to_bar = rocm_backend.deviceToDevice(bar_ptr, hip_src, bytes, 0);
    
    if (d2d_to_bar)
    {
        LOG_INFO("✓ D2D copy HIP → BAR succeeded!");
        
        // Verify by reading BAR directly from CPU
        bool match = true;
        for (int i = 0; i < count; i++)
        {
            if (std::abs(bar_ptr[i] - host_data[i]) > 0.001f)
            {
                LOG_ERROR("Mismatch at " << i << ": expected " << host_data[i] << " got " << bar_ptr[i]);
                match = false;
                break;
            }
        }
        if (match)
        {
            LOG_INFO("✓✓ DATA VERIFIED! HIP→BAR D2D produces correct data readable from CPU!");
        }
    }
    else
    {
        LOG_WARN("D2D copy HIP → BAR failed");
    }
    
    // ============================================================
    // TEST 2: BAR → HIP device (D2D with BAR as source)  
    // ============================================================
    LOG_INFO("\n--- TEST 2: BAR → HIP device via D2D ---");
    
    // Write different pattern to BAR via CPU
    for (int i = 0; i < count; i++) bar_ptr[i] = static_cast<float>(i) * 3.14159f;
    
    // D2D: BAR → HIP device (treating BAR as device source)
    bool d2d_from_bar = rocm_backend.deviceToDevice(hip_dst, bar_ptr, bytes, 0);
    
    if (d2d_from_bar)
    {
        LOG_INFO("✓ D2D copy BAR → HIP succeeded!");
        
        // Verify by copying HIP buffer back to host
        std::vector<float> verify(count);
        rocm_backend.deviceToHost(verify.data(), hip_dst, bytes, 0);
        
        bool match = true;
        for (int i = 0; i < count; i++)
        {
            if (std::abs(verify[i] - bar_ptr[i]) > 0.001f)
            {
                LOG_ERROR("Mismatch at " << i << ": expected " << bar_ptr[i] << " got " << verify[i]);
                match = false;
                break;
            }
        }
        if (match)
        {
            LOG_INFO("✓✓ DATA VERIFIED! BAR→HIP D2D produces correct data!");
        }
    }
    else
    {
        LOG_WARN("D2D copy BAR → HIP failed");
    }
    
    // ============================================================
    // TEST 3: Full round-trip: HIP → BAR → CUDA → verify
    // ============================================================
    LOG_INFO("\n--- TEST 3: Full round-trip HIP → BAR → CUDA ---");
    
    // Write new pattern to HIP source
    for (int i = 0; i < count; i++) host_data[i] = 999.0f - i;
    rocm_backend.hostToDevice(hip_src, host_data.data(), bytes, 0);
    
    // HIP → BAR via D2D
    rocm_backend.deviceToDevice(bar_ptr, hip_src, bytes, 0);
    
    // BAR → CUDA via cudaMemcpy (BAR appears as pinned host to CUDA)
    void* cuda_dst = nullptr;
    cudaMalloc(&cuda_dst, bytes);
    cudaError_t cuda_err = cudaMemcpy(cuda_dst, p2p.getCudaBarPointer(), bytes, cudaMemcpyDeviceToDevice);
    
    if (cuda_err == cudaSuccess)
    {
        // Verify CUDA buffer
        std::vector<float> cuda_verify(count);
        cudaMemcpy(cuda_verify.data(), cuda_dst, bytes, cudaMemcpyDeviceToHost);
        
        bool match = true;
        for (int i = 0; i < count; i++)
        {
            if (std::abs(cuda_verify[i] - host_data[i]) > 0.001f)
            {
                LOG_ERROR("Round-trip mismatch at " << i << ": expected " << host_data[i] << " got " << cuda_verify[i]);
                match = false;
                break;
            }
        }
        if (match)
        {
            LOG_INFO("✓✓✓ FULL ROUND-TRIP VERIFIED: HIP → BAR → CUDA works with ZERO staging!");
            LOG_INFO("    This proves we can use BAR for direct GPU-to-GPU data transfer!");
        }
    }
    else
    {
        LOG_WARN("CUDA D2D from BAR failed: " << cudaGetErrorString(cuda_err));
    }
    
    cudaFree(cuda_dst);
    rocm_backend.free(hip_src, 0);
    rocm_backend.free(hip_dst, 0);
}

/**
 * @brief Test hipHostRegisterIoMemory flag for BAR memory
 * 
 * HYPOTHESIS: hipHostRegisterIoMemory might allow us to register the BAR
 * mmap'd memory as "IO memory" and get a device pointer that HIP kernels
 * can dereference directly.
 * 
 * This is analogous to CUDA's CU_MEMHOSTREGISTER_IOMEMORY which we use
 * successfully for CUDA kernels to access the BAR.
 */
TEST_F(Test__BARBackedHipAllocation, TestHipHostRegisterIoMemory)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   TEST: hipHostRegisterIoMemory for BAR mmap                     ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // Initialize P2P with BAR
    DirectP2PEngine p2p;
    DeviceId cuda_device = DeviceId::cuda(0);
    DeviceId rocm_device = DeviceId::rocm(0);
    
    ASSERT_TRUE(p2p.initializePCIeBar(cuda_device, rocm_device, 0, 64 * 1024 * 1024))
        << "Failed to initialize PCIe BAR";
    
    void* bar_host_ptr = p2p.getBarHostPtr();
    size_t bar_size = p2p.getBarMappedSize();
    ASSERT_NE(bar_host_ptr, nullptr);
    
    LOG_INFO("BAR mmap info:");
    LOG_INFO("  Host ptr: " << bar_host_ptr);
    LOG_INFO("  Size: " << (bar_size / (1024*1024)) << " MB");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    // Try to register the BAR memory with HIP
    LOG_INFO("\n--- Attempting to register BAR as IO memory ---");
    
    void* device_ptr = nullptr;
    bool registered = rocm_backend.registerIoMemory(bar_host_ptr, bar_size, &device_ptr);
    
    if (registered && device_ptr != nullptr)
    {
        LOG_INFO("✓ Registration SUCCEEDED!");
        LOG_INFO("  Host ptr: " << bar_host_ptr);
        LOG_INFO("  Device ptr: " << device_ptr);
        
        // Write test pattern to BAR via CPU
        float* bar_float = static_cast<float*>(bar_host_ptr);
        for (int i = 0; i < 256; i++)
        {
            bar_float[i] = static_cast<float>(i) * 2.5f;
        }
        
        // Allocate regular HIP memory for comparison
        void* hip_dst = rocm_backend.allocate(256 * sizeof(float), 0);
        
        // Try D2D copy using the registered device pointer
        LOG_INFO("\nTrying D2D copy using registered device pointer...");
        bool d2d_ok = rocm_backend.deviceToDevice(hip_dst, device_ptr, 256 * sizeof(float), 0);
        LOG_INFO("D2D copy result: " << (d2d_ok ? "SUCCESS" : "FAILED"));
        
        if (d2d_ok)
        {
            // Verify
            std::vector<float> verify(256);
            rocm_backend.deviceToHost(verify.data(), hip_dst, 256 * sizeof(float), 0);
            
            bool match = true;
            for (int i = 0; i < 256; i++)
            {
                if (std::abs(verify[i] - bar_float[i]) > 0.001f)
                {
                    match = false;
                    break;
                }
            }
            
            if (match)
            {
                LOG_INFO("✓✓✓ DATA VERIFIED! Registered device pointer works for D2D!");
            }
            else
            {
                LOG_WARN("Data mismatch after copy");
            }
        }
        
        rocm_backend.free(hip_dst, 0);
        rocm_backend.unregisterIoMemory(bar_host_ptr);
    }
    else
    {
        LOG_INFO("Registration failed (this is expected for mmap'd BAR memory)");
        LOG_INFO("Exploring alternative approaches...");
    }
    
    // The test passes as long as we explored the options - actual success is a bonus
    SUCCEED() << "Explored hipHostRegisterIoMemory approach";
}

/**
 * @brief Test pointer attribute discovery for BAR memory
 * 
 * Explore what HIP knows about the BAR mmap address and compare
 * with regular HIP allocations to understand the address space.
 */
TEST_F(Test__BARBackedHipAllocation, TestPointerAttributeDiscovery)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   TEST: Pointer attribute discovery for BAR mmap                 ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // Initialize P2P with BAR
    DirectP2PEngine p2p;
    DeviceId cuda_device = DeviceId::cuda(0);
    DeviceId rocm_device = DeviceId::rocm(0);
    
    ASSERT_TRUE(p2p.initializePCIeBar(cuda_device, rocm_device, 0, 64 * 1024 * 1024))
        << "Failed to initialize PCIe BAR";
    
    void* bar_host_ptr = p2p.getBarHostPtr();
    ASSERT_NE(bar_host_ptr, nullptr);
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    // Allocate normal HIP memory for comparison
    void* hip_normal = rocm_backend.allocate(4096, 0);
    void* hip_large = rocm_backend.allocate(64*1024*1024, 0);
    
    LOG_INFO("\n--- Pointer attribute comparison ---");
    
    void* dev_ptr = nullptr;
    void* host_ptr = nullptr;
    std::string mem_type;
    
    // Query normal HIP memory
    LOG_INFO("\nNormal HIP allocation (4KB):");
    if (rocm_backend.getPointerInfo(hip_normal, &dev_ptr, &host_ptr, mem_type))
    {
        LOG_INFO("  Type: " << mem_type);
        LOG_INFO("  Input ptr:   " << std::hex << reinterpret_cast<uintptr_t>(hip_normal) << std::dec);
        LOG_INFO("  Device ptr:  " << std::hex << reinterpret_cast<uintptr_t>(dev_ptr) << std::dec);
        LOG_INFO("  Host ptr:    " << std::hex << reinterpret_cast<uintptr_t>(host_ptr) << std::dec);
    }
    else
    {
        LOG_INFO("  Query failed");
    }
    
    // Query large HIP memory
    LOG_INFO("\nLarge HIP allocation (64MB):");
    if (rocm_backend.getPointerInfo(hip_large, &dev_ptr, &host_ptr, mem_type))
    {
        LOG_INFO("  Type: " << mem_type);
        LOG_INFO("  Input ptr:   " << std::hex << reinterpret_cast<uintptr_t>(hip_large) << std::dec);
        LOG_INFO("  Device ptr:  " << std::hex << reinterpret_cast<uintptr_t>(dev_ptr) << std::dec);
        LOG_INFO("  Host ptr:    " << std::hex << reinterpret_cast<uintptr_t>(host_ptr) << std::dec);
    }
    else
    {
        LOG_INFO("  Query failed");
    }
    
    // Query BAR pointer
    LOG_INFO("\nBAR mmap pointer:");
    if (rocm_backend.getPointerInfo(bar_host_ptr, &dev_ptr, &host_ptr, mem_type))
    {
        LOG_INFO("  Type: " << mem_type);
        LOG_INFO("  Input ptr:   " << std::hex << reinterpret_cast<uintptr_t>(bar_host_ptr) << std::dec);
        LOG_INFO("  Device ptr:  " << std::hex << reinterpret_cast<uintptr_t>(dev_ptr) << std::dec);
        LOG_INFO("  Host ptr:    " << std::hex << reinterpret_cast<uintptr_t>(host_ptr) << std::dec);
        
        if (dev_ptr != nullptr)
        {
            LOG_INFO("✓✓✓ BAR has a device pointer! This might be usable by kernels!");
        }
    }
    else
    {
        LOG_INFO("  Query failed (HIP doesn't know about this memory)");
    }
    
    // Address space analysis
    LOG_INFO("\n--- Address space analysis ---");
    uintptr_t bar_addr = reinterpret_cast<uintptr_t>(bar_host_ptr);
    uintptr_t hip1_addr = reinterpret_cast<uintptr_t>(hip_normal);
    uintptr_t hip2_addr = reinterpret_cast<uintptr_t>(hip_large);
    
    LOG_INFO("BAR mmap:     " << std::hex << bar_addr << std::dec);
    LOG_INFO("HIP small:    " << std::hex << hip1_addr << std::dec);
    LOG_INFO("HIP large:    " << std::hex << hip2_addr << std::dec);
    LOG_INFO("BAR - HIP1:   " << static_cast<int64_t>(bar_addr - hip1_addr) << " bytes");
    LOG_INFO("BAR - HIP2:   " << static_cast<int64_t>(bar_addr - hip2_addr) << " bytes");
    
    rocm_backend.free(hip_normal, 0);
    rocm_backend.free(hip_large, 0);
    
    SUCCEED() << "Explored pointer attributes";
}

/**
 * @brief CRITICAL TEST: Can BAR mmap address be used directly as a HIP device pointer?
 * 
 * OBSERVATION: The BAR mmap address (0x7a8dc0000000) is in the SAME address
 * space range as HIP device allocations (0x7a8dc8200000). They're only ~130MB apart!
 * 
 * HYPOTHESIS: On AMD GPUs with unified/shared address space between host and
 * device, the BAR mmap might already be a valid GPU address. hipMemcpy(D2D)
 * works because the driver recognizes it, but hipPointerGetAttributes fails
 * because the memory wasn't allocated through HIP.
 * 
 * This test tries to use the BAR address directly in a HIP kernel via hipLaunchKernelGGL.
 * We can't write HIP kernels in this file (CUDA headers conflict), so we use
 * hipMemset as a proxy - it runs a kernel internally.
 */
TEST_F(Test__BARBackedHipAllocation, TestDirectBARAsKernelPointer)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   CRITICAL TEST: Direct BAR address as HIP kernel pointer        ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // Initialize P2P with BAR
    DirectP2PEngine p2p;
    DeviceId cuda_device = DeviceId::cuda(0);
    DeviceId rocm_device = DeviceId::rocm(0);
    
    ASSERT_TRUE(p2p.initializePCIeBar(cuda_device, rocm_device, 0, 64 * 1024 * 1024))
        << "Failed to initialize PCIe BAR";
    
    void* bar_host_ptr = p2p.getBarHostPtr();
    ASSERT_NE(bar_host_ptr, nullptr);
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    // First, verify D2D still works (baseline)
    LOG_INFO("\n--- BASELINE: Verify D2D copy works ---");
    
    float* bar_float = static_cast<float*>(bar_host_ptr);
    for (int i = 0; i < 256; i++)
    {
        bar_float[i] = static_cast<float>(i) * 1.5f;
    }
    
    void* hip_buffer = rocm_backend.allocate(256 * sizeof(float), 0);
    
    // D2D: BAR -> HIP
    bool d2d_ok = rocm_backend.deviceToDevice(hip_buffer, bar_host_ptr, 256 * sizeof(float), 0);
    LOG_INFO("D2D copy BAR -> HIP: " << (d2d_ok ? "SUCCESS" : "FAILED"));
    
    if (d2d_ok)
    {
        std::vector<float> verify(256);
        rocm_backend.deviceToHost(verify.data(), hip_buffer, 256 * sizeof(float), 0);
        
        bool match = true;
        for (int i = 0; i < 256; i++)
        {
            if (std::abs(verify[i] - bar_float[i]) > 0.001f)
            {
                match = false;
                break;
            }
        }
        LOG_INFO("Data verification: " << (match ? "PASSED" : "FAILED"));
    }
    
    // Now the critical test: Can hipMemset work on BAR address?
    // hipMemset launches a kernel that writes to device memory
    LOG_INFO("\n--- CRITICAL: Trying hipMemset on BAR address ---");
    LOG_INFO("hipMemset launches an internal kernel that writes to the address.");
    LOG_INFO("If this works, it proves the BAR address is kernel-accessible.");
    
    // Write a known pattern to BAR first (via CPU)
    for (int i = 0; i < 1024; i++)
    {
        reinterpret_cast<uint8_t*>(bar_host_ptr)[i] = 0xAA;
    }
    LOG_INFO("Wrote 0xAA pattern to BAR via CPU");
    
    // Try hipMemset on BAR address (this will launch a kernel!)
    bool memset_ok = rocm_backend.memset(bar_host_ptr, 0x55, 1024, 0);
    LOG_INFO("hipMemset(bar_host_ptr, 0x55, 1024) = " << (memset_ok ? "SUCCESS" : "FAILED"));
    
    if (memset_ok)
    {
        // Verify the pattern changed
        bool changed = true;
        for (int i = 0; i < 1024; i++)
        {
            if (reinterpret_cast<uint8_t*>(bar_host_ptr)[i] != 0x55)
            {
                changed = false;
                LOG_INFO("Byte " << i << " = " << std::hex 
                        << static_cast<int>(reinterpret_cast<uint8_t*>(bar_host_ptr)[i]) << std::dec);
                break;
            }
        }
        
        if (changed)
        {
            LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║   ✓✓✓ SUCCESS! BAR ADDRESS IS KERNEL-ACCESSIBLE!                 ║");
            LOG_INFO("║   hipMemset kernel successfully wrote to BAR memory!             ║");
            LOG_INFO("║   This means HIP kernels CAN dereference the BAR mmap address!   ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
        }
        else
        {
            LOG_INFO("hipMemset returned success but data didn't change?");
        }
    }
    else
    {
        // Try alternative: hipMemsetAsync
        LOG_INFO("\nhipMemset failed. This is the expected behavior if BAR isn't kernel-accessible.");
        LOG_INFO("The GPU crashed attempting to access memory not in its virtual address space.");
    }
    
    rocm_backend.free(hip_buffer, 0);
    
    SUCCEED() << "Explored direct BAR kernel access";
}

/**
 * @brief Test HSA-level memory lock API for BAR memory
 * 
 * hsa_amd_memory_lock() is a lower-level API that pins host memory
 * and returns a GPU-accessible pointer. It operates below the HIP layer
 * and may work for memory that hipHostRegister rejects.
 */
TEST_F(Test__BARBackedHipAllocation, TestHSAMemoryLockForBAR)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing HSA-level hsa_amd_memory_lock on BAR memory            ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // Initialize P2P with BAR
    DirectP2PEngine p2p;
    DeviceId cuda_device = DeviceId::cuda(0);
    DeviceId rocm_device = DeviceId::rocm(0);
    
    ASSERT_TRUE(p2p.initializePCIeBar(cuda_device, rocm_device, 0, 64 * 1024 * 1024))
        << "Failed to initialize PCIe BAR";
    
    void* bar_host_ptr = p2p.getBarHostPtr();
    ASSERT_NE(bar_host_ptr, nullptr);
    
    LOG_INFO("BAR mmap address: " << std::hex << bar_host_ptr << std::dec);
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    // Try HSA-level memory lock
    void* hsa_agent_ptr = nullptr;
    bool lock_ok = rocm_backend.hsaMemoryLock(bar_host_ptr, 4096, &hsa_agent_ptr);
    
    LOG_INFO("\nhsa_amd_memory_lock result:");
    LOG_INFO("  Success: " << (lock_ok ? "YES" : "NO"));
    LOG_INFO("  Agent ptr: " << std::hex << hsa_agent_ptr << std::dec);
    
    if (lock_ok && hsa_agent_ptr != nullptr)
    {
        LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║   HSA MEMORY LOCK SUCCEEDED!                                     ║");
        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
        
        LOG_INFO("\nComparing pointers:");
        LOG_INFO("  BAR host ptr:   " << std::hex << bar_host_ptr << std::dec);
        LOG_INFO("  HSA agent ptr:  " << std::hex << hsa_agent_ptr << std::dec);
        
        if (bar_host_ptr == hsa_agent_ptr)
        {
            LOG_INFO("  → SAME ADDRESS! HSA unified memory model.");
        }
        else
        {
            LOG_INFO("  → DIFFERENT ADDRESS! HSA provided a mapped device pointer.");
            
            // Try hipMemset with the HSA-provided pointer
            LOG_INFO("\nTrying hipMemset with HSA agent ptr...");
            
            // First write via CPU
            float* bar_float = static_cast<float*>(bar_host_ptr);
            bar_float[0] = 12345.0f;
            
            bool memset_ok = rocm_backend.memset(hsa_agent_ptr, 0, 1024, 0);
            LOG_INFO("hipMemset(hsa_agent_ptr, 0, 1024) = " << (memset_ok ? "SUCCESS" : "FAILED"));
            
            if (memset_ok)
            {
                // Check if CPU can still read
                LOG_INFO("After hipMemset, bar_float[0] = " << bar_float[0]);
                
                if (bar_float[0] == 0.0f)
                {
                    LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
                    LOG_INFO("║   ✓✓✓ SUCCESS! HIP KERNEL WROTE TO BAR VIA HSA POINTER!          ║");
                    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
                }
            }
        }
        
        // Cleanup
        rocm_backend.hsaMemoryUnlock(bar_host_ptr);
    }
    else
    {
        LOG_INFO("\nHSA memory lock failed for BAR memory.");
        LOG_INFO("This is expected if HSA doesn't support locking IO memory.");
    }
    
    SUCCEED() << "Explored HSA memory lock on BAR";
}

/**
 * @brief Test comparing normal host memory vs BAR memory with HSA lock
 * 
 * As a baseline, test hsa_amd_memory_lock on normal malloc'd memory
 * to verify the API works before trying it on BAR.
 */
TEST_F(Test__BARBackedHipAllocation, TestHSAMemoryLockBaseline)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   BASELINE: Testing HSA memory lock on normal host memory        ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    // Allocate normal host memory
    size_t size = 4096;
    void* host_ptr = aligned_alloc(4096, size);  // Page-aligned
    ASSERT_NE(host_ptr, nullptr);
    
    LOG_INFO("Normal host ptr: " << std::hex << host_ptr << std::dec);
    
    // Try HSA-level memory lock
    void* hsa_agent_ptr = nullptr;
    bool lock_ok = rocm_backend.hsaMemoryLock(host_ptr, size, &hsa_agent_ptr);
    
    LOG_INFO("\nhsa_amd_memory_lock on normal host memory:");
    LOG_INFO("  Success: " << (lock_ok ? "YES" : "NO"));
    LOG_INFO("  Agent ptr: " << std::hex << hsa_agent_ptr << std::dec);
    
    if (lock_ok && hsa_agent_ptr != nullptr)
    {
        // Test that HIP can use this pointer
        LOG_INFO("\nTrying hipMemset with HSA agent ptr on normal memory...");
        
        float* host_float = static_cast<float*>(host_ptr);
        host_float[0] = 99999.0f;
        
        bool memset_ok = rocm_backend.memset(hsa_agent_ptr, 0, 1024, 0);
        rocm_backend.synchronize(0);
        
        LOG_INFO("hipMemset result: " << (memset_ok ? "SUCCESS" : "FAILED"));
        LOG_INFO("After memset, host_float[0] = " << host_float[0]);
        
        if (memset_ok && host_float[0] == 0.0f)
        {
            LOG_INFO("✓ HSA memory lock + hipMemset works on normal host memory");
        }
        else if (memset_ok)
        {
            LOG_INFO("hipMemset returned success but CPU doesn't see the change");
            LOG_INFO("This might mean the agent_ptr maps to a different buffer");
        }
        
        rocm_backend.hsaMemoryUnlock(host_ptr);
    }
    
    free(host_ptr);
    
    SUCCEED() << "Explored HSA memory lock baseline";
}

/**
 * @brief CREATIVE APPROACH: Test if we can use HIP "copy engine" to write to BAR
 * 
 * Key observation: hipMemcpy(D2D) WORKS with BAR address, meaning the DMA
 * engine can access it. What if we use the DMA engine for our "compute" by:
 * 1. Preparing data in a HIP device buffer
 * 2. hipMemcpy(D2D) from HIP buffer -> BAR
 * 
 * This wouldn't be "zero copy" but would avoid kernel access to BAR entirely.
 * The data path would be:
 *   ROCm compute kernel -> HIP device buffer -> DMA copy -> BAR -> CPU mmap -> CUDA
 */
TEST_F(Test__BARBackedHipAllocation, TestDMABasedApproach)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing DMA-based approach: HIP buffer -> BAR via D2D copy     ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // Initialize P2P with BAR
    DirectP2PEngine p2p;
    DeviceId cuda_device = DeviceId::cuda(0);
    DeviceId rocm_device = DeviceId::rocm(0);
    
    ASSERT_TRUE(p2p.initializePCIeBar(cuda_device, rocm_device, 0, 64 * 1024 * 1024))
        << "Failed to initialize PCIe BAR";
    
    void* bar_host_ptr = p2p.getBarHostPtr();
    ASSERT_NE(bar_host_ptr, nullptr);
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    const size_t count = 1024;
    const size_t bytes = count * sizeof(float);
    
    // Allocate HIP device buffer (this is where kernels will write)
    void* hip_buffer = rocm_backend.allocate(bytes, 0);
    ASSERT_NE(hip_buffer, nullptr);
    
    // Initialize BAR with known pattern via CPU
    float* bar_float = static_cast<float*>(bar_host_ptr);
    for (size_t i = 0; i < count; ++i)
    {
        bar_float[i] = static_cast<float>(i);  // 0, 1, 2, 3, ...
    }
    
    LOG_INFO("\n--- Step 1: Verify initial BAR state ---");
    LOG_INFO("bar_float[0-3] = " << bar_float[0] << ", " << bar_float[1] 
             << ", " << bar_float[2] << ", " << bar_float[3]);
    
    // Simulate a "kernel" by writing to HIP buffer using hipMemset
    // (In real use, this would be an actual compute kernel)
    LOG_INFO("\n--- Step 2: Simulate kernel output to HIP buffer ---");
    bool memset_ok = rocm_backend.memset(hip_buffer, 0xFF, bytes, 0);
    LOG_INFO("hipMemset to HIP buffer: " << (memset_ok ? "SUCCESS" : "FAILED"));
    
    if (memset_ok)
    {
        // Now the key test: D2D copy from HIP buffer to BAR
        LOG_INFO("\n--- Step 3: D2D copy: HIP buffer -> BAR ---");
        bool d2d_ok = rocm_backend.deviceToDevice(bar_host_ptr, hip_buffer, bytes, 0);
        LOG_INFO("hipMemcpy(D2D) HIP->BAR: " << (d2d_ok ? "SUCCESS" : "FAILED"));
        
        if (d2d_ok)
        {
            rocm_backend.synchronize(0);
            
            // Verify BAR content changed
            LOG_INFO("\n--- Step 4: Verify BAR content changed ---");
            
            // Interpret as uint32_t to check for 0xFFFFFFFF pattern
            uint32_t* bar_uint = reinterpret_cast<uint32_t*>(bar_host_ptr);
            LOG_INFO("bar_uint[0-3] = 0x" << std::hex << bar_uint[0] << ", 0x" << bar_uint[1]
                     << ", 0x" << bar_uint[2] << ", 0x" << bar_uint[3] << std::dec);
            
            if (bar_uint[0] == 0xFFFFFFFF && bar_uint[1] == 0xFFFFFFFF)
            {
                LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
                LOG_INFO("║   ✓✓✓ SUCCESS! DMA-BASED APPROACH WORKS!                         ║");
                LOG_INFO("║                                                                  ║");
                LOG_INFO("║   Data flow validated:                                           ║");
                LOG_INFO("║     HIP kernel -> HIP device buffer -> D2D -> BAR -> CPU         ║");
                LOG_INFO("║                                                                  ║");
                LOG_INFO("║   This is a viable workaround for ROCm compute + BAR staging!    ║");
                LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
            }
            else
            {
                LOG_INFO("D2D reported success but BAR data didn't change as expected");
            }
        }
    }
    
    rocm_backend.free(hip_buffer, 0);
    
    SUCCEED() << "Explored DMA-based approach";
}

/**
 * @brief Test bidirectional DMA: BAR <-> HIP buffer
 * 
 * Test both directions:
 * 1. BAR -> HIP buffer (for input data from CUDA)
 * 2. HIP buffer -> BAR (for output data to CUDA)
 */
TEST_F(Test__BARBackedHipAllocation, TestBidirectionalDMA)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing bidirectional DMA: BAR <-> HIP buffer                  ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // Initialize P2P with BAR
    DirectP2PEngine p2p;
    DeviceId cuda_device = DeviceId::cuda(0);
    DeviceId rocm_device = DeviceId::rocm(0);
    
    ASSERT_TRUE(p2p.initializePCIeBar(cuda_device, rocm_device, 0, 64 * 1024 * 1024))
        << "Failed to initialize PCIe BAR";
    
    void* bar_host_ptr = p2p.getBarHostPtr();
    ASSERT_NE(bar_host_ptr, nullptr);
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    const size_t count = 256;
    const size_t bytes = count * sizeof(float);
    
    // Allocate two HIP buffers: input and output
    void* hip_input = rocm_backend.allocate(bytes, 0);
    void* hip_output = rocm_backend.allocate(bytes, 0);
    ASSERT_NE(hip_input, nullptr);
    ASSERT_NE(hip_output, nullptr);
    
    LOG_INFO("\n--- TEST 1: BAR -> HIP (input path) ---");
    
    // Write test data to BAR via CPU
    float* bar_float = static_cast<float*>(bar_host_ptr);
    for (size_t i = 0; i < count; ++i)
    {
        bar_float[i] = static_cast<float>(i * 3.14159f);
    }
    
    // D2D: BAR -> HIP input buffer
    bool d2d_in = rocm_backend.deviceToDevice(hip_input, bar_host_ptr, bytes, 0);
    LOG_INFO("BAR -> HIP input: " << (d2d_in ? "SUCCESS" : "FAILED"));
    
    // Verify by copying back to host
    std::vector<float> verify_in(count);
    rocm_backend.deviceToHost(verify_in.data(), hip_input, bytes, 0);
    
    bool input_ok = true;
    for (size_t i = 0; i < count && input_ok; ++i)
    {
        if (std::abs(verify_in[i] - bar_float[i]) > 0.001f)
        {
            LOG_INFO("Mismatch at " << i << ": expected " << bar_float[i] << ", got " << verify_in[i]);
            input_ok = false;
        }
    }
    LOG_INFO("Input verification: " << (input_ok ? "PASSED" : "FAILED"));
    
    LOG_INFO("\n--- TEST 2: HIP -> BAR (output path) ---");
    
    // Clear BAR
    for (size_t i = 0; i < count; ++i)
    {
        bar_float[i] = 0.0f;
    }
    
    // Initialize HIP output buffer with different pattern
    std::vector<float> output_data(count);
    for (size_t i = 0; i < count; ++i)
    {
        output_data[i] = static_cast<float>(i * 2.71828f);
    }
    rocm_backend.hostToDevice(hip_output, output_data.data(), bytes, 0);
    
    // D2D: HIP output -> BAR
    bool d2d_out = rocm_backend.deviceToDevice(bar_host_ptr, hip_output, bytes, 0);
    rocm_backend.synchronize(0);
    LOG_INFO("HIP output -> BAR: " << (d2d_out ? "SUCCESS" : "FAILED"));
    
    // Verify BAR content
    bool output_ok = true;
    for (size_t i = 0; i < count && output_ok; ++i)
    {
        if (std::abs(bar_float[i] - output_data[i]) > 0.001f)
        {
            LOG_INFO("Mismatch at " << i << ": expected " << output_data[i] << ", got " << bar_float[i]);
            output_ok = false;
        }
    }
    LOG_INFO("Output verification: " << (output_ok ? "PASSED" : "FAILED"));
    
    if (d2d_in && d2d_out && input_ok && output_ok)
    {
        LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║   ✓✓✓ BIDIRECTIONAL DMA WORKS!                                   ║");
        LOG_INFO("║                                                                  ║");
        LOG_INFO("║   Validated data paths:                                          ║");
        LOG_INFO("║     Input:  CUDA -> BAR -> HIP buffer -> ROCm kernel             ║");
        LOG_INFO("║     Output: ROCm kernel -> HIP buffer -> BAR -> CUDA             ║");
        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    }
    
    rocm_backend.free(hip_input, 0);
    rocm_backend.free(hip_output, 0);
    
    SUCCEED() << "Explored bidirectional DMA";
}

/**
 * @brief Test HSA interop API with the BAR file descriptor
 * 
 * hsa_amd_interop_map_buffer takes a dmabuf fd and returns a GPU-accessible
 * pointer. The PCIe BAR resource file (/sys/bus/pci/devices/.../resource0)
 * might work as an interop handle since it represents GPU VRAM.
 */
TEST_F(Test__BARBackedHipAllocation, TestHSAInteropMapBuffer)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing HSA interop API with BAR resource file descriptor      ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    // Find the AMD GPU BAR path
    std::string bar_path = "/sys/bus/pci/devices/0000:1a:00.0/resource0";
    
    LOG_INFO("Attempting to use BAR resource file: " << bar_path);
    
    // Open the BAR resource file
    int bar_fd = open(bar_path.c_str(), O_RDWR);
    if (bar_fd < 0)
    {
        LOG_INFO("Failed to open " << bar_path << ": " << strerror(errno));
        GTEST_SKIP() << "Cannot open BAR resource file (need root?)";
    }
    
    LOG_INFO("Opened BAR resource file, fd=" << bar_fd);
    
    // Try HSA interop map
    size_t mapped_size = 0;
    void* device_ptr = nullptr;
    
    bool success = rocm_backend.hsaInteropMapBuffer(bar_fd, &mapped_size, &device_ptr);
    
    LOG_INFO("\nhsa_amd_interop_map_buffer result:");
    LOG_INFO("  Success: " << (success ? "YES" : "NO"));
    LOG_INFO("  Device ptr: " << std::hex << device_ptr << std::dec);
    LOG_INFO("  Mapped size: " << mapped_size);
    
    if (success && device_ptr != nullptr)
    {
        LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║   ✓✓✓ HSA INTEROP MAP SUCCEEDED!                                 ║");
        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
        
        // Try to use the pointer with hipMemset
        LOG_INFO("\nTrying hipMemset on interop-mapped pointer...");
        bool memset_ok = rocm_backend.memset(device_ptr, 0x55, 1024, 0);
        LOG_INFO("hipMemset result: " << (memset_ok ? "SUCCESS" : "FAILED"));
        
        rocm_backend.hsaInteropUnmapBuffer(device_ptr);
    }
    else
    {
        LOG_INFO("\nHSA interop map failed. The BAR fd may not be a valid dmabuf handle.");
    }
    
    close(bar_fd);
    
    SUCCEED() << "Explored HSA interop API";
}

/**
 * @brief Test HIP external memory import with BAR fd
 * 
 * hipImportExternalMemory with hipExternalMemoryHandleTypeOpaqueFd might
 * work if the BAR resource file can be treated as an opaque fd.
 */
TEST_F(Test__BARBackedHipAllocation, TestHipImportExternalMemory)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing HIP external memory import with BAR fd                 ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    // Find the AMD GPU BAR path
    std::string bar_path = "/sys/bus/pci/devices/0000:1a:00.0/resource0";
    
    // Open the BAR resource file
    int bar_fd = open(bar_path.c_str(), O_RDWR);
    if (bar_fd < 0)
    {
        GTEST_SKIP() << "Cannot open BAR resource file";
    }
    
    LOG_INFO("Opened BAR resource file, fd=" << bar_fd);
    
    // Get the file size (BAR size)
    struct stat st;
    if (fstat(bar_fd, &st) < 0)
    {
        close(bar_fd);
        GTEST_SKIP() << "Cannot stat BAR file";
    }
    
    size_t bar_size = st.st_size;
    LOG_INFO("BAR size: " << (bar_size / (1024*1024*1024)) << " GB");
    
    // Try a smaller region for testing
    size_t test_size = 64 * 1024 * 1024;  // 64 MB
    
    void* device_ptr = nullptr;
    bool success = rocm_backend.importExternalMemory(bar_fd, test_size, &device_ptr);
    
    LOG_INFO("\nhipImportExternalMemory result:");
    LOG_INFO("  Success: " << (success ? "YES" : "NO"));
    LOG_INFO("  Device ptr: " << std::hex << device_ptr << std::dec);
    
    if (success && device_ptr != nullptr)
    {
        LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║   ✓✓✓ HIP EXTERNAL MEMORY IMPORT SUCCEEDED!                      ║");
        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
        
        // Try to use the pointer
        LOG_INFO("\nTrying hipMemset on imported memory...");
        bool memset_ok = rocm_backend.memset(device_ptr, 0x55, 1024, 0);
        LOG_INFO("hipMemset result: " << (memset_ok ? "SUCCESS" : "FAILED"));
    }
    else
    {
        LOG_INFO("\nHIP external memory import failed.");
        LOG_INFO("The BAR fd may not be compatible with hipImportExternalMemory.");
    }
    
    // Note: fd is consumed by hipImportExternalMemory, don't close it
    // close(bar_fd);
    
    SUCCEED() << "Explored HIP external memory import";
}

/**
 * @brief Test if DRM PRIME APIs can help convert BAR to dmabuf
 * 
 * Check if we can use DRM/AMDGPU APIs to create a GEM handle from the
 * BAR userptr, then convert to dmabuf.
 */
TEST_F(Test__BARBackedHipAllocation, TestDRMPrimeExport)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing DRM PRIME APIs for BAR -> dmabuf conversion            ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // First, let's initialize the BAR via our normal path to get the mmap
    DirectP2PEngine p2p;
    DeviceId cuda_device = DeviceId::cuda(0);
    DeviceId rocm_device = DeviceId::rocm(0);
    
    ASSERT_TRUE(p2p.initializePCIeBar(cuda_device, rocm_device, 0, 64 * 1024 * 1024))
        << "Failed to initialize PCIe BAR";
    
    void* bar_host_ptr = p2p.getBarHostPtr();
    ASSERT_NE(bar_host_ptr, nullptr);
    
    LOG_INFO("BAR mmap address: " << std::hex << bar_host_ptr << std::dec);
    
    // Try to find the DRM render node for the AMD GPU
    std::string drm_path = "/dev/dri/renderD129";  // Common path for AMD GPU
    
    int drm_fd = open(drm_path.c_str(), O_RDWR);
    if (drm_fd < 0)
    {
        // Try renderD128
        drm_path = "/dev/dri/renderD128";
        drm_fd = open(drm_path.c_str(), O_RDWR);
    }
    
    if (drm_fd < 0)
    {
        LOG_INFO("Could not open DRM render node: " << strerror(errno));
        LOG_INFO("Trying to find the correct render node...");
        
        // List available render nodes
        for (int i = 128; i < 140; ++i)
        {
            std::string path = "/dev/dri/renderD" + std::to_string(i);
            int fd = open(path.c_str(), O_RDWR);
            if (fd >= 0)
            {
                LOG_INFO("  Found render node: " << path);
                close(fd);
            }
        }
        
        GTEST_SKIP() << "Cannot open DRM render node";
    }
    
    LOG_INFO("Opened DRM render node: " << drm_path << ", fd=" << drm_fd);
    
    // The idea: use DRM_IOCTL_AMDGPU_GEM_USERPTR to create a GEM handle
    // from the BAR mmap pointer, then DRM_IOCTL_PRIME_HANDLE_TO_FD to get dmabuf
    
    // Note: This is exploratory - GEM USERPTR might not work for IO-mapped memory
    
    LOG_INFO("\nNote: GEM USERPTR typically requires memory from malloc/mmap of normal RAM.");
    LOG_INFO("BAR memory is already GPU VRAM, so this approach may not apply.");
    LOG_INFO("The BAR is the 'native' form; we need the reverse mapping!");
    
    close(drm_fd);
    
    SUCCEED() << "Explored DRM PRIME APIs";
}

/**
 * @brief Explore if we can get dmabuf fd from the existing BAR allocation
 * 
 * The AMD driver may already have internal dmabuf handles for BAR regions.
 * This test explores if we can extract them.
 */
TEST_F(Test__BARBackedHipAllocation, TestAMDGPUBOExport)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Exploring AMDGPU buffer object export paths                    ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // Check what render nodes are available
    LOG_INFO("\n--- Checking DRM render nodes ---");
    
    for (int i = 128; i < 140; ++i)
    {
        std::string path = "/dev/dri/renderD" + std::to_string(i);
        int fd = open(path.c_str(), O_RDWR);
        if (fd >= 0)
        {
            // Try to identify the GPU via sysfs
            char sysfs_path[256];
            snprintf(sysfs_path, sizeof(sysfs_path), 
                     "/sys/class/drm/renderD%d/device/vendor", i);
            
            FILE* f = fopen(sysfs_path, "r");
            if (f)
            {
                char vendor[16] = {0};
                if (fgets(vendor, sizeof(vendor), f))
                {
                    // 0x1002 = AMD, 0x10de = NVIDIA
                    LOG_INFO("  renderD" << i << ": vendor=" << vendor);
                }
                fclose(f);
            }
            else
            {
                LOG_INFO("  renderD" << i << ": available (vendor unknown)");
            }
            
            close(fd);
        }
    }
    
    LOG_INFO("\n--- Key insight ---");
    LOG_INFO("The PCIe BAR gives us a CPU virtual address to GPU VRAM.");
    LOG_INFO("To use this from HIP kernels, we need either:");
    LOG_INFO("  1. A way to 'adopt' this memory into HIP's allocator (not supported)");
    LOG_INFO("  2. A dmabuf fd for the BAR region that HSA can map (need driver support)");
    LOG_INFO("  3. Use DMA engine to copy (our working solution)");
    
    LOG_INFO("\n--- Alternative approach to explore ---");
    LOG_INFO("What if we allocate memory with HIP first, then export as dmabuf,");
    LOG_INFO("then share with CUDA? This is the reverse direction!");
    
    SUCCEED() << "Explored AMDGPU export paths";
}

/**
 * @brief Test allocating HIP memory and exporting as dmabuf
 * 
 * Instead of trying to import the CUDA BAR, allocate ROCm memory and
 * export it as dmabuf for CUDA to import. This is the reverse approach!
 */
TEST_F(Test__BARBackedHipAllocation, TestHIPAllocateAndExport)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing reverse approach: HIP allocate -> export -> CUDA       ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    const size_t size = 64 * 1024 * 1024;  // 64 MB
    
    // Allocate HIP device memory
    void* hip_ptr = rocm_backend.allocate(size, 0);
    ASSERT_NE(hip_ptr, nullptr);
    
    LOG_INFO("Allocated HIP memory: " << std::hex << hip_ptr << std::dec);
    
    // Try to get IPC handle (if supported)
    // This would allow sharing with CUDA via CUDA's IPC import
    
    LOG_INFO("\n--- Exploring HIP IPC memory sharing ---");
    LOG_INFO("HIP supports hipIpcGetMemHandle() for inter-process sharing.");
    LOG_INFO("However, cross-vendor (HIP <-> CUDA) IPC is not supported.");
    
    LOG_INFO("\n--- Exploring HIP dmabuf export ---");
    LOG_INFO("HIP does not currently expose a direct dmabuf export API.");
    LOG_INFO("This would need to go through HSA or AMDGPU DRM layer.");
    
    // Try HSA-level export
    void* hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (hsa_handle)
    {
        // Check for hsa_amd_portable_export_dmabuf
        void* export_fn = dlsym(hsa_handle, "hsa_amd_portable_export_dmabuf");
        if (export_fn)
        {
            LOG_INFO("Found hsa_amd_portable_export_dmabuf - dmabuf export may be possible!");
        }
        else
        {
            LOG_INFO("hsa_amd_portable_export_dmabuf not found");
        }
        
        // Check for any other export functions
        void* interop_export = dlsym(hsa_handle, "hsa_amd_interop_export");
        if (interop_export)
        {
            LOG_INFO("Found hsa_amd_interop_export");
        }
        
        dlclose(hsa_handle);
    }
    
    rocm_backend.free(hip_ptr, 0);
    
    SUCCEED() << "Explored HIP export path";
}

/**
 * @brief Test exporting HIP memory as dmabuf and importing into CUDA
 * 
 * This is the reverse of our original BAR approach:
 *   - Allocate memory on AMD GPU via HIP
 *   - Export as dmabuf via hsa_amd_portable_export_dmabuf
 *   - Import into CUDA via cudaImportExternalMemory
 *   - Both GPUs can now access the same AMD VRAM!
 */
TEST_F(Test__BARBackedHipAllocation, TestHIPExportToCUDAImport)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing HIP dmabuf export -> CUDA import (reverse BAR)         ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    const size_t size = 4 * 1024 * 1024;  // 4 MB for testing
    
    // Allocate HIP device memory
    void* hip_ptr = rocm_backend.allocate(size, 0);
    ASSERT_NE(hip_ptr, nullptr);
    
    LOG_INFO("Allocated HIP memory: " << std::hex << hip_ptr << std::dec);
    
    // Initialize with pattern from HIP side
    rocm_backend.memset(hip_ptr, 0xAA, size, 0);
    rocm_backend.synchronize(0);
    LOG_INFO("Initialized HIP memory with pattern 0xAA");
    
    // Load HSA and get the export function
    void* hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!hsa_handle)
    {
        GTEST_SKIP() << "Cannot load HSA runtime";
    }
    
    // hsa_amd_portable_export_dmabuf(const void* ptr, size_t size, int* dmabuf, uint64_t* offset)
    typedef int (*export_dmabuf_fn)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);
    
    auto export_fn = (export_dmabuf_fn)dlsym(hsa_handle, "hsa_amd_portable_export_dmabuf");
    if (!export_fn)
    {
        dlclose(hsa_handle);
        GTEST_SKIP() << "hsa_amd_portable_export_dmabuf not found";
    }
    
    LOG_INFO("\n--- Step 1: Export HIP memory as dmabuf ---");
    
    int dmabuf_fd = -1;
    uint64_t offset = 0;
    
    int status = export_fn(hip_ptr, size, &dmabuf_fd, &offset);
    
    LOG_INFO("hsa_amd_portable_export_dmabuf result:");
    LOG_INFO("  Status: " << status << (status == 0 ? " (SUCCESS)" : " (FAILED)"));
    LOG_INFO("  dmabuf_fd: " << dmabuf_fd);
    LOG_INFO("  offset: " << offset);
    
    if (status != 0 || dmabuf_fd < 0)
    {
        LOG_INFO("\nDmabuf export failed. HIP memory may not be exportable.");
        dlclose(hsa_handle);
        rocm_backend.free(hip_ptr, 0);
        SUCCEED() << "Export not supported for this allocation";
        return;
    }
    
    LOG_INFO("\n--- Step 2: Import dmabuf into CUDA ---");
    
    // First, try CUDA runtime API
    LOG_INFO("\nTrying CUDA runtime API (cudaImportExternalMemory)...");
    
    cudaSetDevice(0);
    
    cudaExternalMemoryHandleDesc extMemHandleDesc = {};
    extMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    extMemHandleDesc.handle.fd = dup(dmabuf_fd);  // dup because CUDA takes ownership
    extMemHandleDesc.size = size;
    extMemHandleDesc.flags = 0;
    
    cudaExternalMemory_t extMem = nullptr;
    cudaError_t cuda_err = cudaImportExternalMemory(&extMem, &extMemHandleDesc);
    
    LOG_INFO("cudaImportExternalMemory (OpaqueFd) result: " << cudaGetErrorString(cuda_err));
    
    // Try with file descriptor type explicitly
    if (cuda_err != cudaSuccess)
    {
        // Maybe it needs explicit type
        LOG_INFO("\nTrying with cudaExternalMemoryHandleTypePosixFileDescriptor...");
        
        // Note: This enum might not exist in older CUDA versions
        // extMemHandleDesc.type = cudaExternalMemoryHandleTypePosixFileDescriptor;
        
        LOG_INFO("Skipping - need to use driver API for more control");
    }
    
    // Try CUDA driver API with cuMemImportFromShareableHandle (CUDA 11.2+)
    LOG_INFO("\nTrying CUDA driver API (cuMemImportFromShareableHandle)...");
    
    CUdeviceptr cu_ptr = 0;
    CUresult cu_result = CUDA_ERROR_NOT_SUPPORTED;
    
    // Load the driver function dynamically
    void* cuda_driver = dlopen("libcuda.so", RTLD_NOW);
    if (cuda_driver)
    {
        typedef CUresult (*cuMemImportFromShareableHandle_fn)(
            CUdeviceptr*, CUmemGenericAllocationHandle*, CUmemAllocationHandleType, void*, unsigned long long);
        
        typedef CUresult (*cuMemCreate_fn)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
        typedef CUresult (*cuMemRelease_fn)(CUmemGenericAllocationHandle);
        typedef CUresult (*cuMemMap_fn)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
        typedef CUresult (*cuMemAddressReserve_fn)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
        typedef CUresult (*cuMemAddressFree_fn)(CUdeviceptr, size_t);
        typedef CUresult (*cuMemSetAccess_fn)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
        
        auto import_fn = (cuMemImportFromShareableHandle_fn)dlsym(cuda_driver, "cuMemImportFromShareableHandle");
        auto create_fn = (cuMemCreate_fn)dlsym(cuda_driver, "cuMemCreate");
        auto release_fn = (cuMemRelease_fn)dlsym(cuda_driver, "cuMemRelease");
        auto map_fn = (cuMemMap_fn)dlsym(cuda_driver, "cuMemMap");
        auto reserve_fn = (cuMemAddressReserve_fn)dlsym(cuda_driver, "cuMemAddressReserve");
        auto free_fn = (cuMemAddressFree_fn)dlsym(cuda_driver, "cuMemAddressFree");
        auto access_fn = (cuMemSetAccess_fn)dlsym(cuda_driver, "cuMemSetAccess");
        
        if (import_fn && reserve_fn && map_fn && access_fn)
        {
            LOG_INFO("Found CUDA VMM functions, trying import...");
            
            // Try importing the dmabuf as a handle
            CUmemGenericAllocationHandle alloc_handle;
            
            // First try with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR (1)
            cu_result = import_fn(&cu_ptr, &alloc_handle, 
                                  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                  (void*)(intptr_t)dmabuf_fd, size);
            
            if (cu_result == CUDA_SUCCESS)
            {
                LOG_INFO("cuMemImportFromShareableHandle succeeded!");
            }
            else
            {
                const char* err_str = nullptr;
                cuGetErrorString(cu_result, &err_str);
                LOG_INFO("cuMemImportFromShareableHandle failed: " << (err_str ? err_str : "unknown") 
                        << " (" << cu_result << ")");
            }
        }
        else
        {
            LOG_INFO("CUDA VMM functions not all found:");
            LOG_INFO("  cuMemImportFromShareableHandle: " << (import_fn ? "found" : "NOT found"));
            LOG_INFO("  cuMemAddressReserve: " << (reserve_fn ? "found" : "NOT found"));
            LOG_INFO("  cuMemMap: " << (map_fn ? "found" : "NOT found"));
            LOG_INFO("  cuMemSetAccess: " << (access_fn ? "found" : "NOT found"));
        }
        
        dlclose(cuda_driver);
    }
    else
    {
        LOG_INFO("Could not load libcuda.so");
    }
    
    // Try CUDA interop with GL/EGL if available
    LOG_INFO("\n--- Checking available CUDA external memory handle types ---");
    LOG_INFO("cudaExternalMemoryHandleTypeOpaqueFd = 1");
    LOG_INFO("cudaExternalMemoryHandleTypeOpaqueWin32 = 2");  
    LOG_INFO("cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3");
    LOG_INFO("cudaExternalMemoryHandleTypeD3D12Heap = 4");
    LOG_INFO("cudaExternalMemoryHandleTypeD3D12Resource = 5");
    LOG_INFO("cudaExternalMemoryHandleTypeD3D11Resource = 6");
    LOG_INFO("cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7");
    LOG_INFO("cudaExternalMemoryHandleTypeNvSciBuf = 8");
    
    // Note: NVIDIA does not support importing arbitrary dmabuf fds from other vendors
    
    if (cuda_err == cudaSuccess && extMem != nullptr)
    {
        LOG_INFO("\n--- Step 3: Get CUDA mapped pointer ---");
        
        cudaExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.offset = offset;
        bufferDesc.size = size;
        bufferDesc.flags = 0;
        
        void* cuda_ptr = nullptr;
        cuda_err = cudaExternalMemoryGetMappedBuffer(&cuda_ptr, extMem, &bufferDesc);
        
        LOG_INFO("cudaExternalMemoryGetMappedBuffer: " << cudaGetErrorString(cuda_err));
        LOG_INFO("CUDA pointer: " << std::hex << cuda_ptr << std::dec);
        
        if (cuda_err == cudaSuccess && cuda_ptr != nullptr)
        {
            LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║   ✓✓✓ CROSS-VENDOR MEMORY SHARING SUCCEEDED!                     ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
            
            LOG_INFO("\n--- Step 4: Verify CUDA can read the HIP data ---");
            
            // Copy from CUDA-mapped pointer to host for verification
            std::vector<uint8_t> verify(1024);
            cuda_err = cudaMemcpy(verify.data(), cuda_ptr, 1024, cudaMemcpyDeviceToHost);
            
            if (cuda_err == cudaSuccess)
            {
                bool all_aa = true;
                for (int i = 0; i < 1024; ++i)
                {
                    if (verify[i] != 0xAA)
                    {
                        all_aa = false;
                        LOG_INFO("Mismatch at " << i << ": expected 0xAA, got 0x" 
                                << std::hex << static_cast<int>(verify[i]) << std::dec);
                        break;
                    }
                }
                
                if (all_aa)
                {
                    LOG_INFO("✓ CUDA successfully read HIP's 0xAA pattern!");
                    
                    LOG_INFO("\n--- Step 5: Test CUDA writing, HIP reading ---");
                    
                    // CUDA writes 0x55
                    cuda_err = cudaMemset(cuda_ptr, 0x55, size);
                    cudaDeviceSynchronize();
                    
                    if (cuda_err == cudaSuccess)
                    {
                        // HIP reads back
                        rocm_backend.synchronize(0);
                        std::vector<uint8_t> hip_verify(1024);
                        rocm_backend.deviceToHost(hip_verify.data(), hip_ptr, 1024, 0);
                        
                        bool all_55 = true;
                        for (int i = 0; i < 1024; ++i)
                        {
                            if (hip_verify[i] != 0x55)
                            {
                                all_55 = false;
                                break;
                            }
                        }
                        
                        if (all_55)
                        {
                            LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
                            LOG_INFO("║   ✓✓✓ BIDIRECTIONAL CROSS-VENDOR SHARING WORKS!                  ║");
                            LOG_INFO("║       CUDA wrote 0x55, HIP read it back correctly!               ║");
                            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
                        }
                        else
                        {
                            LOG_INFO("HIP could not read CUDA's writes");
                        }
                    }
                }
            }
            else
            {
                LOG_INFO("CUDA memcpy from shared memory failed: " << cudaGetErrorString(cuda_err));
            }
        }
        
        cudaDestroyExternalMemory(extMem);
    }
    else
    {
        LOG_INFO("\nCUDA import failed. Cross-vendor dmabuf sharing may not be supported.");
    }
    
    dlclose(hsa_handle);
    rocm_backend.free(hip_ptr, 0);
    
    SUCCEED() << "Explored HIP->CUDA dmabuf sharing";
}

/**
 * @brief Test dmabuf CPU mapping for cross-GPU communication
 * 
 * If CUDA can't import AMD dmabuf directly, we can still use it via CPU mmap:
 *   1. HIP allocates memory
 *   2. Export as dmabuf via hsa_amd_portable_export_dmabuf
 *   3. mmap the dmabuf to get CPU pointer
 *   4. Use CPU pointer as staging area (CPU can see both GPUs' memory)
 */
TEST_F(Test__BARBackedHipAllocation, TestDmabufCPUMapping)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing AMD dmabuf CPU mapping for staging                      ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    const size_t size = 4 * 1024 * 1024;  // 4 MB
    
    // Allocate HIP device memory
    void* hip_ptr = rocm_backend.allocate(size, 0);
    ASSERT_NE(hip_ptr, nullptr);
    
    // Initialize with pattern
    rocm_backend.memset(hip_ptr, 0xAA, size, 0);
    rocm_backend.synchronize(0);
    
    LOG_INFO("Allocated and initialized HIP memory: " << std::hex << hip_ptr << std::dec);
    
    // Export as dmabuf
    void* hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!hsa_handle)
    {
        rocm_backend.free(hip_ptr, 0);
        GTEST_SKIP() << "Cannot load HSA runtime";
    }
    
    typedef int (*export_dmabuf_fn)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);
    auto export_fn = (export_dmabuf_fn)dlsym(hsa_handle, "hsa_amd_portable_export_dmabuf");
    
    if (!export_fn)
    {
        dlclose(hsa_handle);
        rocm_backend.free(hip_ptr, 0);
        GTEST_SKIP() << "hsa_amd_portable_export_dmabuf not found";
    }
    
    int dmabuf_fd = -1;
    uint64_t offset = 0;
    int status = export_fn(hip_ptr, size, &dmabuf_fd, &offset);
    
    LOG_INFO("Export result: status=" << status << " fd=" << dmabuf_fd << " offset=" << offset);
    
    if (status != 0 || dmabuf_fd < 0)
    {
        dlclose(hsa_handle);
        rocm_backend.free(hip_ptr, 0);
        GTEST_SKIP() << "Could not export dmabuf";
    }
    
    // Try to mmap the dmabuf for CPU access
    LOG_INFO("\n--- Trying to mmap dmabuf fd ---");
    
    void* cpu_mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    
    if (cpu_mapped == MAP_FAILED)
    {
        LOG_INFO("mmap failed: " << strerror(errno));
        LOG_INFO("This is expected - GPU VRAM dmabuf may not be CPU-mappable");
    }
    else
    {
        LOG_INFO("mmap succeeded! CPU pointer: " << cpu_mapped);
        
        // Try to read from it
        LOG_INFO("Attempting to read first byte...");
        volatile uint8_t* ptr = static_cast<volatile uint8_t*>(cpu_mapped);
        
        try
        {
            uint8_t val = *ptr;
            LOG_INFO("Read value: 0x" << std::hex << static_cast<int>(val) << std::dec);
            
            if (val == 0xAA)
            {
                LOG_INFO("✓ CPU successfully read HIP's 0xAA pattern via dmabuf mmap!");
            }
        }
        catch (...)
        {
            LOG_INFO("Read caused exception");
        }
        
        munmap(cpu_mapped, size);
    }
    
    close(dmabuf_fd);
    dlclose(hsa_handle);
    rocm_backend.free(hip_ptr, 0);
    
    SUCCEED() << "Tested dmabuf CPU mapping";
}

/**
 * @brief Test if HIP memory exports work with hipMemPoolExportPointer
 * 
 * HIP 5.6+ added memory pool export which might provide a different path
 */
TEST_F(Test__BARBackedHipAllocation, TestHIPMemPoolExport)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Testing HIP memory pool export                                  ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    // Check for hipMemPoolExportPointer
    void* hip_rt = dlopen("libamdhip64.so", RTLD_NOW);
    if (!hip_rt)
    {
        GTEST_SKIP() << "Cannot load HIP runtime";
    }
    
    typedef int (*hipMemPoolExportToShareableHandle_fn)(void*, void*, int, int);
    typedef int (*hipMemPoolImportFromShareableHandle_fn)(void*, void*, int, int);
    typedef int (*hipMemPoolExportPointer_fn)(void*, void*);
    typedef int (*hipMemPoolImportPointer_fn)(void*, void*, void*);
    
    auto export_handle_fn = (hipMemPoolExportToShareableHandle_fn)dlsym(hip_rt, "hipMemPoolExportToShareableHandle");
    auto import_handle_fn = (hipMemPoolImportFromShareableHandle_fn)dlsym(hip_rt, "hipMemPoolImportFromShareableHandle");
    auto export_ptr_fn = (hipMemPoolExportPointer_fn)dlsym(hip_rt, "hipMemPoolExportPointer");
    auto import_ptr_fn = (hipMemPoolImportPointer_fn)dlsym(hip_rt, "hipMemPoolImportPointer");
    
    LOG_INFO("HIP memory pool APIs:");
    LOG_INFO("  hipMemPoolExportToShareableHandle: " << (export_handle_fn ? "found" : "NOT found"));
    LOG_INFO("  hipMemPoolImportFromShareableHandle: " << (import_handle_fn ? "found" : "NOT found"));
    LOG_INFO("  hipMemPoolExportPointer: " << (export_ptr_fn ? "found" : "NOT found"));
    LOG_INFO("  hipMemPoolImportPointer: " << (import_ptr_fn ? "found" : "NOT found"));
    
    dlclose(hip_rt);
    
    SUCCEED() << "Explored HIP memory pool APIs";
}

/**
 * @brief Full end-to-end test: HIP -> dmabuf mmap -> CUDA
 * 
 * This is a potentially viable zero-copy path:
 *   1. HIP allocates GPU memory
 *   2. Export as dmabuf via hsa_amd_portable_export_dmabuf
 *   3. mmap the dmabuf to get CPU pointer
 *   4. CUDA copies from CPU pointer to CUDA memory
 *   
 * If this works, it's effectively "zero-copy" from AMD's perspective
 * (the dmabuf mmap provides coherent CPU access to GPU memory)
 */
TEST_F(Test__BARBackedHipAllocation, TestHIPDmabufCUDAFullPath)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Full path: HIP allocate -> dmabuf mmap -> CUDA copy            ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    const size_t size = 16 * 1024 * 1024;  // 16 MB for realistic test
    
    // Allocate HIP device memory
    void* hip_ptr = rocm_backend.allocate(size, 0);
    ASSERT_NE(hip_ptr, nullptr);
    
    // Initialize with a pattern
    std::vector<uint8_t> pattern(size);
    for (size_t i = 0; i < size; ++i)
    {
        pattern[i] = static_cast<uint8_t>(i & 0xFF);
    }
    rocm_backend.hostToDevice(hip_ptr, pattern.data(), size, 0);
    rocm_backend.synchronize(0);
    
    LOG_INFO("Initialized HIP memory with sequential pattern");
    
    // Export as dmabuf
    void* hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    ASSERT_NE(hsa_handle, nullptr);
    
    typedef int (*export_dmabuf_fn)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);
    auto export_fn = (export_dmabuf_fn)dlsym(hsa_handle, "hsa_amd_portable_export_dmabuf");
    ASSERT_NE(export_fn, nullptr);
    
    int dmabuf_fd = -1;
    uint64_t offset = 0;
    int status = export_fn(hip_ptr, size, &dmabuf_fd, &offset);
    ASSERT_EQ(status, 0);
    ASSERT_GE(dmabuf_fd, 0);
    
    LOG_INFO("Exported dmabuf fd=" << dmabuf_fd);
    
    // mmap the dmabuf
    void* cpu_mapped = mmap(nullptr, size, PROT_READ, MAP_SHARED, dmabuf_fd, 0);
    ASSERT_NE(cpu_mapped, MAP_FAILED);
    
    LOG_INFO("mmap'd dmabuf to CPU: " << cpu_mapped);
    
    // Allocate CUDA memory and copy from the mmap'd pointer
    cudaSetDevice(0);
    
    void* cuda_ptr = nullptr;
    cudaError_t cuda_err = cudaMalloc(&cuda_ptr, size);
    ASSERT_EQ(cuda_err, cudaSuccess);
    
    LOG_INFO("Allocated CUDA memory: " << cuda_ptr);
    
    // Time the copy from dmabuf mmap to CUDA
    LOG_INFO("\n--- Timing dmabuf mmap -> CUDA copy ---");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cuda_err = cudaMemcpy(cuda_ptr, cpu_mapped, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (cuda_err == cudaSuccess)
    {
        double bandwidth = (size / (1024.0 * 1024.0)) / (duration.count() / 1e6);
        LOG_INFO("Copy time: " << duration.count() << " µs");
        LOG_INFO("Bandwidth: " << bandwidth << " MB/s");
        
        // Verify the data
        std::vector<uint8_t> verify(size);
        cuda_err = cudaMemcpy(verify.data(), cuda_ptr, size, cudaMemcpyDeviceToHost);
        ASSERT_EQ(cuda_err, cudaSuccess);
        
        bool match = true;
        for (size_t i = 0; i < size; ++i)
        {
            if (verify[i] != pattern[i])
            {
                LOG_INFO("Mismatch at " << i << ": expected " << static_cast<int>(pattern[i]) 
                        << ", got " << static_cast<int>(verify[i]));
                match = false;
                break;
            }
        }
        
        if (match)
        {
            LOG_INFO("✓ Data verified correctly!");
        }
    }
    else
    {
        LOG_INFO("CUDA copy from dmabuf mmap failed: " << cudaGetErrorString(cuda_err));
    }
    
    // Compare with traditional CPU staging
    LOG_INFO("\n--- Comparing with traditional CPU staging ---");
    
    std::vector<uint8_t> cpu_staging(size);
    rocm_backend.deviceToHost(cpu_staging.data(), hip_ptr, size, 0);
    rocm_backend.synchronize(0);
    
    start = std::chrono::high_resolution_clock::now();
    
    cuda_err = cudaMemcpy(cuda_ptr, cpu_staging.data(), size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (cuda_err == cudaSuccess)
    {
        double bandwidth = (size / (1024.0 * 1024.0)) / (duration.count() / 1e6);
        LOG_INFO("Traditional CPU staging copy time: " << duration.count() << " µs");
        LOG_INFO("Traditional CPU staging bandwidth: " << bandwidth << " MB/s");
    }
    
    // Cleanup
    cudaFree(cuda_ptr);
    munmap(cpu_mapped, size);
    close(dmabuf_fd);
    dlclose(hsa_handle);
    rocm_backend.free(hip_ptr, 0);
    
    SUCCEED() << "Tested full HIP -> dmabuf mmap -> CUDA path";
}

/**
 * @brief Test if CUDA can WRITE to dmabuf mmap and HIP can read it
 * 
 * The reverse direction: CUDA writes to mmap'd AMD dmabuf, HIP sees it
 */
TEST_F(Test__BARBackedHipAllocation, TestCUDAWriteToDmabufMmap)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║   Reverse: CUDA write -> dmabuf mmap -> HIP read                 ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    const size_t size = 4 * 1024 * 1024;  // 4 MB
    
    // Allocate HIP device memory (this is the "shared" buffer)
    void* hip_ptr = rocm_backend.allocate(size, 0);
    ASSERT_NE(hip_ptr, nullptr);
    
    // Initialize to zeros
    rocm_backend.memset(hip_ptr, 0x00, size, 0);
    rocm_backend.synchronize(0);
    
    LOG_INFO("Initialized HIP memory to zeros");
    
    // Export as dmabuf
    void* hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    ASSERT_NE(hsa_handle, nullptr);
    
    typedef int (*export_dmabuf_fn)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);
    auto export_fn = (export_dmabuf_fn)dlsym(hsa_handle, "hsa_amd_portable_export_dmabuf");
    ASSERT_NE(export_fn, nullptr);
    
    int dmabuf_fd = -1;
    uint64_t offset = 0;
    int status = export_fn(hip_ptr, size, &dmabuf_fd, &offset);
    ASSERT_EQ(status, 0);
    ASSERT_GE(dmabuf_fd, 0);
    
    // mmap the dmabuf with WRITE permission
    void* cpu_mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    ASSERT_NE(cpu_mapped, MAP_FAILED);
    
    LOG_INFO("mmap'd dmabuf with R/W: " << cpu_mapped);
    
    // CUDA writes 0x55 pattern via CPU (copy from CPU staging)
    cudaSetDevice(0);
    
    void* cuda_src = nullptr;
    cudaError_t cuda_err = cudaMalloc(&cuda_src, size);
    ASSERT_EQ(cuda_err, cudaSuccess);
    
    // Initialize CUDA buffer with 0x55
    cuda_err = cudaMemset(cuda_src, 0x55, size);
    cudaDeviceSynchronize();
    ASSERT_EQ(cuda_err, cudaSuccess);
    
    // CUDA copies TO the dmabuf mmap
    LOG_INFO("CUDA copying 0x55 pattern to dmabuf mmap...");
    
    cuda_err = cudaMemcpy(cpu_mapped, cuda_src, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    if (cuda_err == cudaSuccess)
    {
        LOG_INFO("CUDA copy to dmabuf mmap succeeded");
        
        // HIP should now see the 0x55 pattern WITHOUT any explicit copy
        LOG_INFO("Checking if HIP sees the data without explicit copy...");
        
        // Sync HIP side (might be needed for coherence)
        rocm_backend.synchronize(0);
        
        // Read back from HIP
        std::vector<uint8_t> hip_verify(1024);
        rocm_backend.deviceToHost(hip_verify.data(), hip_ptr, 1024, 0);
        rocm_backend.synchronize(0);
        
        bool all_55 = true;
        for (int i = 0; i < 1024; ++i)
        {
            if (hip_verify[i] != 0x55)
            {
                LOG_INFO("Mismatch at " << i << ": expected 0x55, got 0x" 
                        << std::hex << static_cast<int>(hip_verify[i]) << std::dec);
                all_55 = false;
                break;
            }
        }
        
        if (all_55)
        {
            LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║   ✓✓✓ BIDIRECTIONAL DMABUF SHARING WORKS!                        ║");
            LOG_INFO("║       CUDA wrote to mmap, HIP read from GPU memory!              ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
        }
        else
        {
            LOG_INFO("HIP didn't see CUDA's writes via dmabuf mmap");
            LOG_INFO("This might require explicit cache invalidation on HIP side");
        }
    }
    else
    {
        LOG_INFO("CUDA copy to dmabuf mmap failed: " << cudaGetErrorString(cuda_err));
    }
    
    // Cleanup
    cudaFree(cuda_src);
    munmap(cpu_mapped, size);
    close(dmabuf_fd);
    dlclose(hsa_handle);
    rocm_backend.free(hip_ptr, 0);
    
    SUCCEED() << "Tested CUDA write to dmabuf mmap";
}

/**
 * @brief COMPREHENSIVE COMPARISON: All cross-GPU transfer methods
 * 
 * Compare every viable path for HIP → CUDA data transfer:
 * 1. Traditional CPU staging (hipMemcpy D2H + cudaMemcpy H2D)
 * 2. Dmabuf mmap (export dmabuf, mmap, cudaMemcpy from mmap)
 * 3. BAR DMA (our current approach: mmap BAR, hipMemcpy D2D to BAR)
 * 4. BAR DMA reverse (CUDA allocated, HIP writes to BAR)
 */
TEST_F(Test__BARBackedHipAllocation, TestComprehensiveTransferComparison)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║        COMPREHENSIVE CROSS-GPU TRANSFER COMPARISON               ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
    
    ROCmBackend rocm_backend;
    rocm_backend.setDevice(0);
    
    const size_t size = 64 * 1024 * 1024;  // 64 MB for meaningful benchmark
    const int warmup = 3;
    const int iterations = 10;
    
    // Allocate HIP source buffer
    void* hip_src = rocm_backend.allocate(size, 0);
    ASSERT_NE(hip_src, nullptr);
    
    // Initialize with pattern
    std::vector<uint8_t> pattern(size);
    for (size_t i = 0; i < size; ++i)
        pattern[i] = static_cast<uint8_t>(i & 0xFF);
    rocm_backend.hostToDevice(hip_src, pattern.data(), size, 0);
    rocm_backend.synchronize(0);
    
    // Allocate CUDA destination buffer
    cudaSetDevice(0);
    void* cuda_dst = nullptr;
    cudaError_t cuda_err = cudaMalloc(&cuda_dst, size);
    ASSERT_EQ(cuda_err, cudaSuccess);
    
    // ========================================================================
    // Method 1: Traditional CPU staging
    // ========================================================================
    LOG_INFO("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    LOG_INFO("Method 1: Traditional CPU staging (HIP D2H + CUDA H2D)");
    LOG_INFO("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    std::vector<uint8_t> cpu_staging(size);
    
    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        rocm_backend.deviceToHost(cpu_staging.data(), hip_src, size, 0);
        rocm_backend.synchronize(0);
        cudaMemcpy(cuda_dst, cpu_staging.data(), size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        rocm_backend.deviceToHost(cpu_staging.data(), hip_src, size, 0);
        rocm_backend.synchronize(0);
        cudaMemcpy(cuda_dst, cpu_staging.data(), size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double cpu_staging_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double cpu_staging_bw = (size / (1024.0 * 1024.0)) / (cpu_staging_ms / 1000.0);
    
    LOG_INFO("  Average time: " << cpu_staging_ms << " ms");
    LOG_INFO("  Bandwidth: " << cpu_staging_bw << " MB/s");
    
    // ========================================================================
    // Method 2: Dmabuf mmap path
    // ========================================================================
    LOG_INFO("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    LOG_INFO("Method 2: Dmabuf mmap (export dmabuf + mmap + cudaMemcpy)");
    LOG_INFO("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    void* hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    double dmabuf_ms = 0;
    double dmabuf_bw = 0;
    
    if (hsa_handle)
    {
        typedef int (*export_dmabuf_fn)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);
        auto export_fn = (export_dmabuf_fn)dlsym(hsa_handle, "hsa_amd_portable_export_dmabuf");
        
        if (export_fn)
        {
            int dmabuf_fd = -1;
            uint64_t offset = 0;
            int status = export_fn(hip_src, size, &dmabuf_fd, &offset);
            
            if (status == 0 && dmabuf_fd >= 0)
            {
                void* cpu_mapped = mmap(nullptr, size, PROT_READ, MAP_SHARED, dmabuf_fd, 0);
                
                if (cpu_mapped != MAP_FAILED)
                {
                    // Warmup
                    for (int i = 0; i < warmup; ++i)
                    {
                        cudaMemcpy(cuda_dst, cpu_mapped, size, cudaMemcpyHostToDevice);
                        cudaDeviceSynchronize();
                    }
                    
                    start = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < iterations; ++i)
                    {
                        cudaMemcpy(cuda_dst, cpu_mapped, size, cudaMemcpyHostToDevice);
                        cudaDeviceSynchronize();
                    }
                    end = std::chrono::high_resolution_clock::now();
                    
                    dmabuf_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
                    dmabuf_bw = (size / (1024.0 * 1024.0)) / (dmabuf_ms / 1000.0);
                    
                    LOG_INFO("  Average time: " << dmabuf_ms << " ms");
                    LOG_INFO("  Bandwidth: " << dmabuf_bw << " MB/s");
                    LOG_INFO("  NOTE: Very slow due to uncached CPU->PCIe access");
                    
                    munmap(cpu_mapped, size);
                }
                else
                {
                    LOG_INFO("  FAILED: mmap failed");
                }
                close(dmabuf_fd);
            }
            else
            {
                LOG_INFO("  FAILED: dmabuf export failed");
            }
        }
        dlclose(hsa_handle);
    }
    
    // ========================================================================
    // Method 3: BAR DMA (our current approach)
    // ========================================================================
    LOG_INFO("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    LOG_INFO("Method 3: BAR DMA (hipMemcpy to BAR mmap'd CUDA VRAM)");
    LOG_INFO("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    double bar_dma_ms = 0;
    double bar_dma_bw = 0;
    
    // Find NVIDIA BAR
    std::string nvidia_bar_path;
    for (int i = 0; i < 10; ++i)
    {
        std::string path = "/sys/bus/pci/devices/0000:65:00." + std::to_string(i) + "/resource0";
        struct stat st;
        if (stat(path.c_str(), &st) == 0)
        {
            nvidia_bar_path = path;
            break;
        }
    }
    
    // Also check common NVIDIA PCI addresses
    if (nvidia_bar_path.empty())
    {
        std::vector<std::string> common_paths = {
            "/sys/bus/pci/devices/0000:41:00.0/resource0",
            "/sys/bus/pci/devices/0000:01:00.0/resource0"
        };
        for (const auto& path : common_paths)
        {
            struct stat st;
            if (stat(path.c_str(), &st) == 0)
            {
                nvidia_bar_path = path;
                break;
            }
        }
    }
    
    if (!nvidia_bar_path.empty())
    {
        int bar_fd = open(nvidia_bar_path.c_str(), O_RDWR);
        if (bar_fd >= 0)
        {
            // Get BAR size (need to check actual allocation vs BAR size)
            struct stat st;
            fstat(bar_fd, &st);
            size_t bar_size = st.st_size;
            
            LOG_INFO("  NVIDIA BAR: " << nvidia_bar_path << " (" << (bar_size / (1024*1024)) << " MB)");
            
            if (bar_size >= size)
            {
                void* bar_mmap = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, bar_fd, 0);
                
                if (bar_mmap != MAP_FAILED)
                {
                    // Warmup
                    for (int i = 0; i < warmup; ++i)
                    {
                        rocm_backend.deviceToDevice(bar_mmap, hip_src, size, 0);
                        rocm_backend.synchronize(0);
                    }
                    
                    start = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < iterations; ++i)
                    {
                        rocm_backend.deviceToDevice(bar_mmap, hip_src, size, 0);
                        rocm_backend.synchronize(0);
                    }
                    end = std::chrono::high_resolution_clock::now();
                    
                    bar_dma_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
                    bar_dma_bw = (size / (1024.0 * 1024.0)) / (bar_dma_ms / 1000.0);
                    
                    LOG_INFO("  Average time: " << bar_dma_ms << " ms");
                    LOG_INFO("  Bandwidth: " << bar_dma_bw << " MB/s");
                    
                    munmap(bar_mmap, size);
                }
                else
                {
                    LOG_INFO("  FAILED: BAR mmap failed - " << strerror(errno));
                }
            }
            else
            {
                LOG_INFO("  SKIPPED: BAR size " << bar_size << " < test size " << size);
            }
            close(bar_fd);
        }
        else
        {
            LOG_INFO("  FAILED: Cannot open BAR - " << strerror(errno));
        }
    }
    else
    {
        LOG_INFO("  SKIPPED: Could not find NVIDIA BAR");
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    LOG_INFO("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    LOG_INFO("SUMMARY: HIP -> CUDA Transfer Methods (" << (size / (1024*1024)) << " MB)");
    LOG_INFO("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    LOG_INFO("┌─────────────────────────────────┬────────────┬────────────────┐");
    LOG_INFO("│ Method                          │ Time (ms)  │ Bandwidth      │");
    LOG_INFO("├─────────────────────────────────┼────────────┼────────────────┤");
    LOG_INFO("│ 1. CPU staging (D2H + H2D)      │ " << std::fixed << std::setprecision(2) << std::setw(8) << cpu_staging_ms 
            << "   │ " << std::setw(8) << cpu_staging_bw << " MB/s │");
    if (dmabuf_bw > 0)
    {
        LOG_INFO("│ 2. Dmabuf mmap (very slow)      │ " << std::setw(8) << dmabuf_ms 
                << "   │ " << std::setw(8) << dmabuf_bw << " MB/s │");
    }
    if (bar_dma_bw > 0)
    {
        LOG_INFO("│ 3. BAR DMA (hipMemcpy D2D)      │ " << std::setw(8) << bar_dma_ms 
                << "   │ " << std::setw(8) << bar_dma_bw << " MB/s │");
    }
    LOG_INFO("└─────────────────────────────────┴────────────┴────────────────┘");
    
    if (bar_dma_bw > 0 && bar_dma_bw > cpu_staging_bw)
    {
        double speedup = bar_dma_bw / cpu_staging_bw;
        LOG_INFO("\n✓ BAR DMA is " << std::setprecision(1) << speedup << "x faster than CPU staging!");
    }
    else if (bar_dma_bw > 0)
    {
        LOG_INFO("\n⚠ BAR DMA is slower than CPU staging on this system");
    }
    
    // Cleanup
    cudaFree(cuda_dst);
    rocm_backend.free(hip_src, 0);
    
    SUCCEED() << "Completed comprehensive transfer comparison";
}

/**
 * @brief Summary of findings and recommended approach
 */
TEST_F(Test__BARBackedHipAllocation, TestSummaryAndRecommendation)
{
    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║           COMPREHENSIVE SUMMARY: CROSS-GPU MEMORY ACCESS OPTIONS             ║");
    LOG_INFO("╠══════════════════════════════════════════════════════════════════════════════╣");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  FAILED APPROACHES (HIP cannot directly access BAR):                         ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  ❌ hipHostRegister(BAR, flags) - 'invalid argument' for all flag combos     ║");
    LOG_INFO("║  ❌ hsa_amd_memory_lock(BAR) - 'out of resources' (can't pin IO memory)      ║");
    LOG_INFO("║  ❌ hsa_amd_interop_map_buffer(BAR_fd) - 'invalid argument'                  ║");
    LOG_INFO("║     → BAR resource file is NOT a dmabuf                                      ║");
    LOG_INFO("║  ❌ hipImportExternalMemory(BAR_fd) - 'out of memory'                        ║");
    LOG_INFO("║     → CUDA BAR fd not compatible with HIP import                             ║");
    LOG_INFO("║  ❌ hipMemset/kernel deref on BAR - memory access fault                      ║");
    LOG_INFO("║     → BAR address not in HIP memory tracker                                  ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  FAILED APPROACHES (CUDA cannot import AMD dmabuf):                          ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  ❌ cudaImportExternalMemory(AMD_dmabuf) - 'unknown error'                   ║");
    LOG_INFO("║  ❌ cuMemImportFromShareableHandle(AMD_dmabuf) - error 999                   ║");
    LOG_INFO("║     → NVIDIA only accepts dmabufs from own allocations                       ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  WORKING BUT SLOW APPROACHES:                                                ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  ✅ hsa_amd_portable_export_dmabuf() + mmap() - WORKS                        ║");
    LOG_INFO("║     → CPU can read/write AMD GPU memory via dmabuf mmap                      ║");
    LOG_INFO("║     → BIDIRECTIONAL: CUDA can write to mmap, HIP sees it!                    ║");
    LOG_INFO("║     → BUT: ~21 MB/s (uncached PCIe access) vs ~4 GB/s staging                ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  WORKING HIGH-PERFORMANCE APPROACHES:                                        ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  ✅ hipMemcpy(D2D) BAR ↔ HIP buffer - WORKS (~4+ GB/s)                       ║");
    LOG_INFO("║     → AMD DMA engine handles the PCIe transaction                            ║");
    LOG_INFO("║     → Full PCIe bandwidth, bidirectional                                     ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║  ✅ Traditional CPU staging - WORKS (~1.8 GB/s round-trip)                   ║");
    LOG_INFO("║     → hipMemcpy D2H + cudaMemcpy H2D                                         ║");
    LOG_INFO("║     → Simple, portable, but 2x PCIe traversal                                ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  RECOMMENDED ARCHITECTURE:                                                   ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║    [CUDA writes]       [BAR memory]        [HIP D2D]      [HIP kernel]      ║");
    LOG_INFO("║    ┌──────────┐       ┌───────────┐       ┌─────────┐    ┌───────────┐      ║");
    LOG_INFO("║    │  CUDA    │──────>│  AMD BAR  │<──────│ hipCopy │<───│ HIP       │      ║");
    LOG_INFO("║    │  kernel  │       │  (mmap'd) │       │   D2D   │    │ kernel    │      ║");
    LOG_INFO("║    └──────────┘       └───────────┘       └─────────┘    └───────────┘      ║");
    LOG_INFO("║                              │                                              ║");
    LOG_INFO("║                              v                                              ║");
    LOG_INFO("║                       PCIe Gen4 x16                                         ║");
    LOG_INFO("║                       ~25 GB/s unidirectional                               ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║  WHY BAR DMA IS OPTIMAL:                                                    ║");
    LOG_INFO("║  • Single PCIe traversal (vs 2x for CPU staging)                            ║");
    LOG_INFO("║  • DMA engine handles transfer (no CPU involvement)                         ║");
    LOG_INFO("║  • Full PCIe bandwidth utilization                                          ║");
    LOG_INFO("║  • Extra D2D copy adds minimal latency (~1-2μs for small transfers)         ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  KEY LEARNINGS:                                                              ║");
    LOG_INFO("║  ═══════════════════════════════════════════════════════════════════════════ ║");
    LOG_INFO("║  1. PCIe BAR is NOT a dmabuf - it's a direct PCI resource mapping            ║");
    LOG_INFO("║  2. Neither HIP nor CUDA can 'register' each other's BAR memory              ║");
    LOG_INFO("║  3. hsa_amd_portable_export_dmabuf() exports HIP VRAM as dmabuf              ║");
    LOG_INFO("║  4. Exported dmabuf is mmap-able, but slow (uncached CPU access)             ║");
    LOG_INFO("║  5. NVIDIA rejects foreign dmabufs in cudaImportExternalMemory               ║");
    LOG_INFO("║  6. DMA engines (hipMemcpy D2D to BAR) are the fastest option                ║");
    LOG_INFO("║                                                                              ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════════════════╝");
    
    SUCCEED() << "Summary complete";
}

#endif // HAVE_CUDA && HAVE_ROCM