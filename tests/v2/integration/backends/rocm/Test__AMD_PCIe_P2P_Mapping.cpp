/**
 * @file Test__AMD_PCIe_P2P_Mapping.cpp
 * @brief Experimental test for AMD-side PCIe P2P mapping (tinygrad-style)
 *
 * This test explores mapping NVIDIA GPU BAR memory into AMD GPU's address space
 * using direct page table manipulation via MMIO. This is the approach used by
 * tinygrad to enable bidirectional P2P between AMD and NVIDIA GPUs.
 *
 * Key insight from tinygrad:
 * 1. Get the PCIe BAR physical address from /sys/bus/pci/devices/{bus}/resource
 * 2. Program AMD GPU's page tables to map a virtual address to the PCIe BAR address
 * 3. Use aspace=AddrSpace.SYS with snooped=True, uncached=True for PCIe coherent access
 *
 * References:
 * - tinygrad/runtime/support/system.py: LNXPCIIfaceBase.map() method
 * - tinygrad/runtime/support/am/amdev.py: AMMemoryManager, AMPageTableEntry
 * - tinygrad/runtime/support/am/ip.py: AM_GMC for page table setup
 *
 * NOTE: This test uses pure sysfs/mmap - no CUDA or HIP runtime headers needed.
 * For runtime operations, use the Llaminar backend abstractions (CUDABackend, ROCmBackend)
 * which isolate vendor headers in their respective compilation units.
 */

#include <gtest/gtest.h>

#include "utils/Logger.h"
#include "backends/DeviceId.h"

// Optional: Use Llaminar backends for runtime queries (avoids header conflicts)
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#include <hip/hip_runtime.h>  // For raw HIP API calls
#endif

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <memory>
#include <dlfcn.h>

// This test works with sysfs parsing regardless of backend availability
// The HAVE_CUDA/HAVE_ROCM guards are only for optional runtime verification

namespace llaminar2::test::experimental
{

    //==========================================================================
    // PCIe BAR Info Structure
    //==========================================================================

    struct PCIeBarInfo
    {
        std::string pci_address; // e.g., "0000:b1:00.0"

        // All BARs
        uint64_t bar_phys_addr[6] = {0}; // Physical address for each BAR
        uint64_t bar_size[6] = {0};      // Size in bytes for each BAR

        // Convenience: largest BAR (typically VRAM)
        int vram_bar_index = -1;     // Which BAR is VRAM (largest), -1 if none
        uint64_t vram_phys_addr = 0; // Physical address of VRAM BAR
        uint64_t vram_size = 0;      // Size of VRAM BAR

        bool is_nvidia = false;     // Vendor check
        bool is_amd = false;        // Vendor check
        bool resizable_bar = false; // Whether BAR is resizable (>256MB typically)
    };

    //==========================================================================
    // PCI Resource Parser (mimics tinygrad's PCIDevice)
    //==========================================================================

    class PCIResourceParser
    {
    public:
        static std::vector<std::string> scanPCIBus(uint16_t vendor_id)
        {
            std::vector<std::string> devices;
            const std::string pci_path = "/sys/bus/pci/devices";

            DIR *dir = opendir(pci_path.c_str());
            if (!dir)
            {
                LOG_WARN("Cannot open " << pci_path);
                return devices;
            }

            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr)
            {
                if (entry->d_name[0] == '.')
                    continue;

                std::string device_path = pci_path + "/" + entry->d_name;
                std::string vendor_path = device_path + "/vendor";

                std::ifstream vendor_file(vendor_path);
                if (!vendor_file)
                    continue;

                std::string vendor_str;
                vendor_file >> vendor_str;

                uint16_t vendor = std::stoul(vendor_str, nullptr, 16);
                if (vendor == vendor_id)
                {
                    devices.push_back(entry->d_name);
                }
            }

            closedir(dir);
            return devices;
        }

        static PCIeBarInfo parseResourceFile(const std::string &pci_address)
        {
            PCIeBarInfo info{};
            info.pci_address = pci_address;

            std::string resource_path = "/sys/bus/pci/devices/" + pci_address + "/resource";
            std::ifstream resource_file(resource_path);

            if (!resource_file)
            {
                LOG_WARN("Cannot open " << resource_path);
                return info;
            }

            // Parse resource file - each line is: start_addr end_addr flags
            // Lines 0-5 are BAR0-BAR5, line 6+ are expansion ROM and other resources
            std::string line;
            int line_index = 0;

            while (std::getline(resource_file, line) && line_index < 6)
            {
                std::istringstream iss(line);
                uint64_t start, end, flags;
                iss >> std::hex >> start >> end >> flags;

                uint64_t size = (end >= start && start != 0) ? (end - start + 1) : 0;

                info.bar_phys_addr[line_index] = start;
                info.bar_size[line_index] = size;

                line_index++;
            }

            // Find the largest BAR - this is typically VRAM
            info.vram_bar_index = -1;
            info.vram_size = 0;
            info.vram_phys_addr = 0;

            for (int i = 0; i < 6; i++)
            {
                if (info.bar_size[i] > info.vram_size)
                {
                    info.vram_size = info.bar_size[i];
                    info.vram_phys_addr = info.bar_phys_addr[i];
                    info.vram_bar_index = i;
                }
            }

            // Resizable BAR typically means VRAM > 256MB
            info.resizable_bar = (info.vram_size > 256 * 1024 * 1024);

            // Check vendor
            std::string vendor_path = "/sys/bus/pci/devices/" + pci_address + "/vendor";
            std::ifstream vendor_file(vendor_path);
            if (vendor_file)
            {
                std::string vendor_str;
                vendor_file >> vendor_str;
                uint16_t vendor = std::stoul(vendor_str, nullptr, 16);
                info.is_nvidia = (vendor == 0x10de);
                info.is_amd = (vendor == 0x1002);
            }

            return info;
        }

        static void *mmapBar(const std::string &pci_address, int bar_num, size_t size)
        {
            std::string resource_path = "/sys/bus/pci/devices/" + pci_address + "/resource" + std::to_string(bar_num);

            int fd = open(resource_path.c_str(), O_RDWR | O_SYNC);
            if (fd < 0)
            {
                LOG_WARN("Cannot open " << resource_path << " (need root?)");
                return nullptr;
            }

            void *mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);

            if (mapped == MAP_FAILED)
            {
                LOG_WARN("mmap failed for " << resource_path);
                return nullptr;
            }

            return mapped;
        }
    };

    //==========================================================================
    // AMD GPU Page Table Manipulation (simplified)
    //==========================================================================

    /**
     * AMD GPU Memory Manager - Simplified version of tinygrad's AMMemoryManager
     *
     * Key concepts from tinygrad:
     * - Page tables are stored in GPU VRAM, accessed via BAR0
     * - MMIO registers (BAR5) control VM and page table configuration
     * - AddrSpace.SYS means use system (PCIe) address space
     * - The page table entry format includes flags for system, snooped, uncached
     */
    class AMDPageTableManager
    {
    public:
        // AMD GPU PTE flags (from amdgpu_vm.h and tinygrad)
        static constexpr uint64_t AMDGPU_PTE_VALID = (1ULL << 0);
        static constexpr uint64_t AMDGPU_PTE_SYSTEM = (1ULL << 1);
        static constexpr uint64_t AMDGPU_PTE_SNOOPED = (1ULL << 2);
        static constexpr uint64_t AMDGPU_PTE_READABLE = (1ULL << 5);
        static constexpr uint64_t AMDGPU_PTE_WRITEABLE = (1ULL << 6);
        static constexpr uint64_t AMDGPU_PTE_EXECUTABLE = (1ULL << 7);
        static constexpr uint64_t AMDGPU_PTE_MTYPE_NV10_MASK = (0x7ULL << 57); // MTYPE field
        static constexpr uint64_t AMDGPU_MTYPE_UC = 0;                         // Uncached

        /**
         * Build a PTE entry for mapping a PCIe BAR address
         *
         * From tinygrad's AMPageTableEntry.set_entry():
         * - For system addresses: set AMDGPU_PTE_SYSTEM flag
         * - Physical address goes in bits [47:12]
         * - Valid, readable, writeable, snooped flags for PCIe access
         */
        static uint64_t buildPTE(uint64_t pcie_phys_addr, bool snooped = true, bool uncached = true)
        {
            uint64_t pte = 0;

            // Address field (bits 47:12, mask 0x0000FFFFFFFFF000)
            pte |= (pcie_phys_addr & 0x0000FFFFFFFFF000ULL);

            // Flags
            pte |= AMDGPU_PTE_VALID;
            pte |= AMDGPU_PTE_SYSTEM; // This is a PCIe system address
            pte |= AMDGPU_PTE_READABLE;
            pte |= AMDGPU_PTE_WRITEABLE;

            if (snooped)
            {
                pte |= AMDGPU_PTE_SNOOPED;
            }

            if (uncached)
            {
                // Set MTYPE to UC (uncached)
                pte |= (AMDGPU_MTYPE_UC << 57);
            }

            return pte;
        }
    };

    //==========================================================================
    // Test Fixture
    //==========================================================================

    class Test__AMD_PCIe_P2P_Mapping : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Scan for NVIDIA GPUs (vendor 0x10de)
            nvidia_devices_ = PCIResourceParser::scanPCIBus(0x10de);

            // Scan for AMD GPUs (vendor 0x1002)
            amd_devices_ = PCIResourceParser::scanPCIBus(0x1002);

            LOG_INFO("Found " << nvidia_devices_.size() << " NVIDIA GPUs");
            LOG_INFO("Found " << amd_devices_.size() << " AMD GPUs");
        }

        std::vector<std::string> nvidia_devices_;
        std::vector<std::string> amd_devices_;
    };

    //==========================================================================
    // Tests
    //==========================================================================

    TEST_F(Test__AMD_PCIe_P2P_Mapping, DiscoverGPUBARs)
    {
        LOG_INFO("\n=== GPU PCIe BAR Discovery ===\n");

        // List NVIDIA GPU BARs
        for (const auto &dev : nvidia_devices_)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            LOG_INFO("NVIDIA GPU: " << info.pci_address);
            LOG_INFO("  VRAM BAR" << info.vram_bar_index << ": 0x" << std::hex << info.vram_phys_addr
                                  << " size=" << std::dec << (info.vram_size / (1024 * 1024 * 1024)) << " GB"
                                  << (info.resizable_bar ? " (resizable)" : ""));
            // Show all BARs for completeness
            for (int i = 0; i < 6; i++)
            {
                if (info.bar_size[i] > 0)
                {
                    LOG_INFO("    BAR" << i << ": 0x" << std::hex << info.bar_phys_addr[i]
                                       << " size=" << std::dec << (info.bar_size[i] / (1024 * 1024)) << " MB");
                }
            }
        }

        // List AMD GPU BARs
        for (const auto &dev : amd_devices_)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            LOG_INFO("AMD GPU: " << info.pci_address);
            LOG_INFO("  VRAM BAR" << info.vram_bar_index << ": 0x" << std::hex << info.vram_phys_addr
                                  << " size=" << std::dec << (info.vram_size / (1024 * 1024 * 1024)) << " GB"
                                  << (info.resizable_bar ? " (resizable)" : ""));
            // Show all BARs for completeness
            for (int i = 0; i < 6; i++)
            {
                if (info.bar_size[i] > 0)
                {
                    LOG_INFO("    BAR" << i << ": 0x" << std::hex << info.bar_phys_addr[i]
                                       << " size=" << std::dec << (info.bar_size[i] / (1024 * 1024)) << " MB");
                }
            }
        }

        // Basic assertions
        EXPECT_GT(nvidia_devices_.size() + amd_devices_.size(), 0u) << "Need at least one GPU";
    }

    TEST_F(Test__AMD_PCIe_P2P_Mapping, DISABLED_ExplainTinygradP2PApproach)
    {
        /**
         * This test documents tinygrad's P2P mapping approach.
         *
         * From system.py LNXPCIIfaceBase.map():
         *
         * ```python
         * def map(self, b:HCQBuffer):
         *     # Case 1: CPU memory - lock pages and get physical addresses
         *     if b.owner._is_cpu():
         *         System.lock_memory(int(b.va_addr), b.size)
         *         paddrs = [(x, 0x1000) for x in System.system_paddrs(va_addr, size)]
         *         aspace = AddrSpace.SYS
         *
         *     # Case 2: Another PCI GPU
         *     elif isinstance(ifa, LNXPCIIfaceBase):
         *         if b.meta.mapping.aspace is AddrSpace.SYS:
         *             # Already system memory
         *             paddrs, aspace = b.meta.mapping.paddrs, AddrSpace.SYS
         *         elif hasattr(ifa.dev_impl, 'paddr2xgmi') and ifa.dev_impl.gmc.xgmi_seg_sz > 0:
         *             # XGMI link (AMD GPU to AMD GPU)
         *             paddrs = [(ifa.dev_impl.paddr2xgmi(p), sz) for p, sz in b.meta.mapping.paddrs]
         *             aspace = AddrSpace.PEER
         *         else:
         *             # PCIe BAR P2P - KEY INSIGHT!
         *             paddrs = [(p + ifa.p2p_base_addr, sz) for p, sz in b.meta.mapping.paddrs]
         *             aspace = AddrSpace.SYS
         *
         *     # Program the page tables
         *     self.dev_impl.mm.map_range(va_addr, size, paddrs, aspace=aspace, snooped=True, uncached=True)
         * ```
         *
         * Where p2p_base_addr comes from:
         * ```python
         * self.p2p_base_addr = self.pci_dev.bar_info[vram_bar].addr
         * ```
         *
         * And bar_info[vram_bar].addr is the physical PCIe BAR address from /sys/bus/pci/devices/{bus}/resource
         *
         * So to map NVIDIA VRAM into AMD GPU's address space:
         * 1. Get NVIDIA's BAR0 physical address (e.g., 0x3bc000000000)
         * 2. Get the offset within NVIDIA VRAM of the allocation
         * 3. Compute pcie_addr = nvidia_bar0_phys + offset
         * 4. Program AMD's page tables to map a virtual address to pcie_addr
         * 5. Use aspace=SYS, snooped=True, uncached=True for PCIe coherent access
         */

        LOG_INFO("See test comments for tinygrad P2P approach documentation");
        SUCCEED();
    }

    TEST_F(Test__AMD_PCIe_P2P_Mapping, DISABLED_MapNVIDIABarFromAMD_Experimental)
    {
        /**
         * EXPERIMENTAL: Attempt to map NVIDIA BAR from AMD GPU
         *
         * This is a proof-of-concept that requires:
         * 1. Root access to mmap GPU BARs
         * 2. Understanding of AMD GPU's page table format
         * 3. Proper initialization of AMD GPU's MMU
         *
         * WARNING: This can crash the system if done incorrectly!
         * Only run on a development machine with proper precautions.
         */

        if (nvidia_devices_.empty() || amd_devices_.empty())
        {
            GTEST_SKIP() << "Need both NVIDIA and AMD GPUs for this test";
        }

        // Find NVIDIA GPU with largest VRAM BAR
        PCIeBarInfo nvidia_info;
        for (const auto &dev : nvidia_devices_)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            if (info.vram_size > nvidia_info.vram_size)
                nvidia_info = info;
        }
        LOG_INFO("NVIDIA VRAM BAR" << nvidia_info.vram_bar_index << " physical address: 0x"
                                   << std::hex << nvidia_info.vram_phys_addr);

        // Find AMD GPU with largest VRAM BAR
        PCIeBarInfo amd_info;
        for (const auto &dev : amd_devices_)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            if (info.vram_size > amd_info.vram_size)
                amd_info = info;
        }
        LOG_INFO("AMD VRAM BAR" << amd_info.vram_bar_index << " physical address: 0x"
                                << std::hex << amd_info.vram_phys_addr);

        // To actually do the mapping, we would need to:
        // 1. mmap AMD's MMIO BAR (for page table manipulation)
        // 2. mmap AMD's VRAM BAR (for page table storage)
        // 3. Find or allocate a page table
        // 4. Write PTEs that point to NVIDIA's VRAM BAR + offset
        // 5. Flush the TLB
        //
        // This requires deep knowledge of the specific AMD GPU's page table format,
        // which varies by generation (GFX9, GFX10, GFX11, etc.)

        // For now, just compute what the PTE would look like
        uint64_t test_offset = 0;  // Offset into NVIDIA VRAM
        uint64_t page_size = 4096; // 4KB pages
        uint64_t nvidia_vram_phys_page = nvidia_info.vram_phys_addr + test_offset;

        uint64_t pte = AMDPageTableManager::buildPTE(nvidia_vram_phys_page, true, true);

        LOG_INFO("\nComputed PTE for mapping NVIDIA VRAM into AMD:");
        LOG_INFO("  Target physical address: 0x" << std::hex << nvidia_vram_phys_page);
        LOG_INFO("  PTE value: 0x" << std::hex << pte);
        LOG_INFO("  PTE flags:");
        LOG_INFO("    VALID: " << ((pte & AMDPageTableManager::AMDGPU_PTE_VALID) ? "yes" : "no"));
        LOG_INFO("    SYSTEM: " << ((pte & AMDPageTableManager::AMDGPU_PTE_SYSTEM) ? "yes" : "no"));
        LOG_INFO("    SNOOPED: " << ((pte & AMDPageTableManager::AMDGPU_PTE_SNOOPED) ? "yes" : "no"));
        LOG_INFO("    READABLE: " << ((pte & AMDPageTableManager::AMDGPU_PTE_READABLE) ? "yes" : "no"));
        LOG_INFO("    WRITEABLE: " << ((pte & AMDPageTableManager::AMDGPU_PTE_WRITEABLE) ? "yes" : "no"));

        SUCCEED() << "PTE computation successful (actual mapping not attempted)";
    }

    TEST_F(Test__AMD_PCIe_P2P_Mapping, DISABLED_CompareWithCurrentApproach)
    {
        /**
         * Compare tinygrad's approach with our current DirectP2P approach:
         *
         * CURRENT APPROACH (DirectP2P.cpp):
         * - Map AMD BAR0 (VRAM) into CPU address space via mmap
         * - Register that CPU pointer with CUDA using cuMemHostRegister(CU_MEMHOSTREGISTER_IOMEMORY)
         * - CUDA can then access AMD VRAM through the registered pointer
         * - Direction: CUDA reads/writes AMD VRAM
         *
         * TINYGRAD APPROACH:
         * - Program AMD GPU's page tables to map a virtual address to NVIDIA's BAR0
         * - HIP/ROCm can then access NVIDIA VRAM through that virtual address
         * - Direction: HIP reads/writes NVIDIA VRAM
         *
         * ADVANTAGES OF TINYGRAD'S APPROACH:
         * 1. Bidirectional: Both GPUs can initiate transfers
         * 2. No CUDA API dependency: Works at hardware level
         * 3. Can work with any PCIe device that exposes a BAR
         *
         * DISADVANTAGES:
         * 1. Requires root access and deep GPU knowledge
         * 2. GPU-specific page table format
         * 3. Risk of system crashes if done incorrectly
         * 4. Not officially supported by AMD
         *
         * IMPLEMENTATION COMPLEXITY:
         * - Current: ~200 lines, uses official CUDA API
         * - Tinygrad: ~2000+ lines across multiple files, direct hardware manipulation
         */

        LOG_INFO("See test comments for approach comparison");
        SUCCEED();
    }

    TEST_F(Test__AMD_PCIe_P2P_Mapping, MmapNVIDIABar_RequiresRoot)
    {
        if (nvidia_devices_.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        // Find GPU with largest VRAM BAR
        PCIeBarInfo best_info;
        for (const auto &dev : nvidia_devices_)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            if (info.vram_size > best_info.vram_size)
            {
                best_info = info;
            }
        }

        if (best_info.vram_size == 0)
        {
            GTEST_SKIP() << "No NVIDIA GPU with valid VRAM BAR found";
        }

        LOG_INFO("Found NVIDIA GPU: " << best_info.pci_address);
        LOG_INFO("  VRAM BAR" << best_info.vram_bar_index << ": 0x" << std::hex << best_info.vram_phys_addr
                              << " size=" << std::dec << (best_info.vram_size / (1024 * 1024 * 1024)) << " GB");

        // Try to mmap a portion of the NVIDIA VRAM BAR
        size_t map_size = std::min(best_info.vram_size, (uint64_t)(64 * 1024 * 1024)); // 64MB max

        void *mapped = PCIResourceParser::mmapBar(best_info.pci_address, best_info.vram_bar_index, map_size);

        if (mapped == nullptr)
        {
            LOG_WARN("Cannot mmap NVIDIA VRAM BAR - this is expected without root access");
            LOG_INFO("To enable: run as root or set CAP_SYS_RAWIO capability");
            GTEST_SKIP() << "Need root access to mmap GPU BARs";
        }

        LOG_INFO("Successfully mapped " << (map_size / (1024 * 1024)) << " MB of NVIDIA VRAM");

        // Read first 4 bytes (just to verify access works)
        uint32_t first_word = *reinterpret_cast<volatile uint32_t *>(mapped);
        LOG_INFO("First word of NVIDIA VRAM: 0x" << std::hex << first_word);

        munmap(mapped, map_size);

        SUCCEED();
    }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    TEST_F(Test__AMD_PCIe_P2P_Mapping, VerifyBackendsAvailable)
    {
        // Use Llaminar backends to verify GPU availability at runtime
        // This demonstrates using the backend abstraction instead of raw CUDA/HIP headers
        CUDABackend cuda_backend;
        ROCmBackend rocm_backend;

        LOG_INFO("CUDA devices via backend: " << cuda_backend.deviceCount());
        LOG_INFO("ROCm devices via backend: " << rocm_backend.deviceCount());

        for (int i = 0; i < cuda_backend.deviceCount(); i++)
        {
            LOG_INFO("  CUDA[" << i << "]: " << cuda_backend.deviceName(i));
        }
        for (int i = 0; i < rocm_backend.deviceCount(); i++)
        {
            LOG_INFO("  ROCm[" << i << "]: " << rocm_backend.deviceName(i));
        }

        EXPECT_GT(cuda_backend.deviceCount(), 0) << "Expected at least one CUDA device";
        EXPECT_GT(rocm_backend.deviceCount(), 0) << "Expected at least one ROCm device";
    }

    TEST_F(Test__AMD_PCIe_P2P_Mapping, TinygradStyle_HsaMemoryLock)
    {
        /**
         * Attempt tinygrad-style P2P: mmap NVIDIA BAR and lock it for HIP access
         *
         * This uses hsa_amd_memory_lock() to pin the mmap'd PCIe BAR memory and
         * get a pointer that HIP kernels can use to access NVIDIA VRAM directly.
         */
        if (nvidia_devices_.empty() || amd_devices_.empty())
        {
            GTEST_SKIP() << "Need both NVIDIA and AMD GPUs for this test";
        }

        // Find the actual GPU with largest BAR (VRAM)
        std::string nvidia_gpu;
        PCIeBarInfo nvidia_info;
        for (const auto &dev : nvidia_devices_)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            if (info.vram_size > nvidia_info.vram_size)
            {
                nvidia_gpu = dev;
                nvidia_info = info;
            }
        }

        if (nvidia_gpu.empty() || nvidia_info.vram_size == 0)
        {
            GTEST_SKIP() << "No NVIDIA GPU with valid VRAM BAR found";
        }

        LOG_INFO("Using NVIDIA GPU: " << nvidia_gpu);
        LOG_INFO("  VRAM BAR" << nvidia_info.vram_bar_index << " phys: 0x" << std::hex << nvidia_info.vram_phys_addr);
        LOG_INFO("  VRAM size: " << std::dec << (nvidia_info.vram_size / (1024 * 1024 * 1024)) << " GB"
                                 << (nvidia_info.resizable_bar ? " (resizable BAR)" : ""));

        // Step 1: mmap NVIDIA's VRAM BAR
        size_t map_size = std::min(nvidia_info.vram_size, (uint64_t)(64 * 1024 * 1024)); // 64MB max for test
        void *nvidia_bar_mapped = PCIResourceParser::mmapBar(nvidia_gpu, nvidia_info.vram_bar_index, map_size);

        if (nvidia_bar_mapped == nullptr)
        {
            GTEST_SKIP() << "Cannot mmap NVIDIA VRAM BAR - need root access";
        }

        LOG_INFO("Step 1: mmap'd " << (map_size / (1024 * 1024)) << " MB of NVIDIA VRAM BAR"
                                   << nvidia_info.vram_bar_index << " at " << nvidia_bar_mapped);

        // Verify we can read from the mmap
        volatile uint32_t test_read = *reinterpret_cast<volatile uint32_t *>(nvidia_bar_mapped);
        LOG_INFO("  Verification read: 0x" << std::hex << test_read);

        // Step 2: Try to lock this memory for HIP access using HSA
        ROCmBackend rocm_backend;

        if (rocm_backend.deviceCount() == 0)
        {
            munmap(nvidia_bar_mapped, map_size);
            GTEST_SKIP() << "No ROCm devices available";
        }

        LOG_INFO("Step 2: Attempting hsa_amd_memory_lock on NVIDIA VRAM BAR...");

        void *hip_accessible_ptr = nullptr;
        bool lock_success = rocm_backend.hsaMemoryLock(nvidia_bar_mapped, map_size, &hip_accessible_ptr);

        if (lock_success && hip_accessible_ptr != nullptr)
        {
            LOG_INFO("  SUCCESS! HSA memory lock worked!");
            LOG_INFO("  Host ptr:   " << nvidia_bar_mapped);
            LOG_INFO("  Agent ptr:  " << hip_accessible_ptr);

            // Step 3: Try to access it from HIP
            LOG_INFO("Step 3: Testing HIP access to NVIDIA VRAM...");

            // Allocate a small buffer on AMD GPU
            void *amd_buffer = rocm_backend.allocate(4096, 0);
            if (amd_buffer)
            {
                // Try to copy from NVIDIA BAR to AMD buffer
                bool copy_success = rocm_backend.hostToDevice(amd_buffer, hip_accessible_ptr, 256, 0);
                if (copy_success)
                {
                    LOG_INFO("  hostToDevice copy succeeded!");

                    // Read back to verify
                    uint32_t readback[64];
                    rocm_backend.deviceToHost(readback, amd_buffer, 256, 0);
                    LOG_INFO("  First word from NVIDIA via HIP: 0x" << std::hex << readback[0]);

                    EXPECT_EQ(readback[0], test_read) << "Data mismatch between CPU and HIP reads";
                }
                else
                {
                    LOG_WARN("  hostToDevice copy failed");
                }
                rocm_backend.free(amd_buffer, 0);
            }

            // Cleanup: unlock HSA memory
            rocm_backend.hsaMemoryUnlock(nvidia_bar_mapped);
        }
        else
        {
            LOG_INFO("  HSA memory lock failed (expected for PCIe BAR memory)");
            LOG_INFO("  This means tinygrad uses a different approach - direct page table manipulation");
        }

        // Step 4: Try registerIoMemory as alternative
        LOG_INFO("Step 4: Trying hipHostRegister with IOMEMORY flag...");
        void *io_device_ptr = nullptr;
        bool io_success = rocm_backend.registerIoMemory(nvidia_bar_mapped, map_size, &io_device_ptr);

        if (io_success && io_device_ptr != nullptr)
        {
            LOG_INFO("  registerIoMemory SUCCESS!");
            LOG_INFO("  Device ptr: " << io_device_ptr);

            // Test access
            void *amd_buffer = rocm_backend.allocate(4096, 0);
            if (amd_buffer)
            {
                // Try D2D copy from registered IO memory to AMD buffer
                bool copy_ok = rocm_backend.deviceToDevice(amd_buffer, io_device_ptr, 256, 0);
                if (copy_ok)
                {
                    LOG_INFO("  D2D copy from NVIDIA BAR succeeded!");
                    uint32_t readback[64];
                    rocm_backend.deviceToHost(readback, amd_buffer, 256, 0);
                    LOG_INFO("  Data: 0x" << std::hex << readback[0]);
                }
                rocm_backend.free(amd_buffer, 0);
            }

            rocm_backend.unregisterIoMemory(nvidia_bar_mapped);
        }
        else
        {
            LOG_INFO("  registerIoMemory failed");
        }

        // Cleanup
        munmap(nvidia_bar_mapped, map_size);

        LOG_INFO("\n=== Summary ===");
        LOG_INFO("HSA memory lock: " << (lock_success ? "SUCCESS" : "FAILED"));
        LOG_INFO("Register IO memory: " << (io_success ? "SUCCESS" : "FAILED"));

        // At least one method should work for tinygrad-style P2P
        if (lock_success || io_success)
        {
            SUCCEED() << "At least one P2P registration method worked!";
        }
        else
        {
            LOG_INFO("Neither method worked - tinygrad may use direct page table manipulation");
            SUCCEED() << "P2P registration methods not available (expected for some configurations)";
        }
    }

    /**
     * @brief Try direct hipMemcpy with NVIDIA BAR mmap as destination
     *
     * The AMD GPU's DMA engine might accept any physical address, even ones
     * pointing to NVIDIA VRAM via the PCIe BAR. This is different from kernel
     * access - DMA engines just push data to addresses.
     *
     * Hypothesis: If the AMD DMA engine treats the mmap'd NVIDIA BAR as just
     * another system memory address, it should be able to DMA data there.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryDirectDMA_NvidiaBarAsDest)
    {
        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        // Find NVIDIA and AMD GPUs
        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        auto amd_gpus = PCIResourceParser::scanPCIBus(0x1002);

        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }
        if (amd_gpus.empty())
        {
            GTEST_SKIP() << "No AMD GPUs found";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: Direct DMA to NVIDIA BAR via hipMemcpy                ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Get NVIDIA GPU info
        std::string nvidia_gpu = nvidia_gpus[0];
        PCIeBarInfo nvidia_info = PCIResourceParser::parseResourceFile(nvidia_gpu);

        LOG_INFO("NVIDIA GPU: " << nvidia_gpu);
        LOG_INFO("  VRAM BAR" << nvidia_info.vram_bar_index << " phys: 0x" << std::hex
                              << nvidia_info.vram_phys_addr << std::dec);
        LOG_INFO("  VRAM size: " << (nvidia_info.vram_size / (1024 * 1024 * 1024)) << " GB");

        // mmap a portion of NVIDIA's VRAM BAR
        size_t map_size = 64 * 1024 * 1024; // 64MB
        void *nvidia_bar = PCIResourceParser::mmapBar(nvidia_gpu, nvidia_info.vram_bar_index, map_size);

        if (nvidia_bar == nullptr)
        {
            GTEST_SKIP() << "Cannot mmap NVIDIA VRAM BAR - need root access";
        }

        LOG_INFO("mmap'd " << (map_size / (1024 * 1024)) << " MB of NVIDIA VRAM at " << nvidia_bar);

        // Allocate source buffer on AMD GPU
        rocm_backend.setDevice(0);
        const size_t test_size = 4096; // Start small
        void *hip_src = rocm_backend.allocate(test_size, 0);
        ASSERT_NE(hip_src, nullptr) << "Failed to allocate HIP memory";

        // Initialize source with pattern
        std::vector<float> pattern(test_size / sizeof(float));
        for (size_t i = 0; i < pattern.size(); ++i)
        {
            pattern[i] = 3.14159f * (i + 1);
        }
        rocm_backend.hostToDevice(hip_src, pattern.data(), test_size, 0);
        rocm_backend.synchronize(0);

        LOG_INFO("\nAttempt 1: hipMemcpyDeviceToHost with NVIDIA BAR as 'host'...");
        LOG_INFO("  This treats NVIDIA BAR mmap as a host pointer destination");

        // Try deviceToHost (HIP treats the mmap as host memory)
        bool d2h_ok = rocm_backend.deviceToHost(nvidia_bar, hip_src, test_size, 0);
        rocm_backend.synchronize(0);

        if (d2h_ok)
        {
            LOG_INFO("  ✓ hipMemcpyDeviceToHost returned success!");

            // Verify by reading back from the BAR via CPU
            float *bar_data = static_cast<float *>(nvidia_bar);
            bool match = true;
            for (size_t i = 0; i < std::min(pattern.size(), (size_t)8); ++i)
            {
                LOG_INFO("    bar[" << i << "] = " << bar_data[i] << " (expected " << pattern[i] << ")");
                if (std::abs(bar_data[i] - pattern[i]) > 0.001f)
                {
                    match = false;
                }
            }

            if (match)
            {
                LOG_INFO("  ✓✓✓ DATA VERIFIED! AMD DMA can write to NVIDIA VRAM via BAR!");
            }
            else
            {
                LOG_INFO("  Data mismatch - DMA may have gone to wrong location");
            }
        }
        else
        {
            LOG_INFO("  ✗ hipMemcpyDeviceToHost failed");
        }

        LOG_INFO("\nAttempt 2: hipMemcpyDeviceToDevice with NVIDIA BAR as destination...");
        LOG_INFO("  This tries to use HIP D2D with BAR address as a 'device' pointer");

        // Try deviceToDevice (HIP might recognize it as device memory)
        bool d2d_ok = rocm_backend.deviceToDevice(nvidia_bar, hip_src, test_size, 0);
        rocm_backend.synchronize(0);

        if (d2d_ok)
        {
            LOG_INFO("  ✓ hipMemcpyDeviceToDevice returned success!");

            // Verify
            float *bar_data = static_cast<float *>(nvidia_bar);
            bool match = true;
            for (size_t i = 0; i < std::min(pattern.size(), (size_t)8); ++i)
            {
                LOG_INFO("    bar[" << i << "] = " << bar_data[i] << " (expected " << pattern[i] << ")");
                if (std::abs(bar_data[i] - pattern[i]) > 0.001f)
                {
                    match = false;
                }
            }

            if (match)
            {
                LOG_INFO("  ✓✓✓ DATA VERIFIED! AMD D2D can write to NVIDIA BAR!");
            }
        }
        else
        {
            LOG_INFO("  ✗ hipMemcpyDeviceToDevice failed");
        }

        // Cleanup
        rocm_backend.free(hip_src, 0);
        munmap(nvidia_bar, map_size);

        SUCCEED() << "Completed DMA experiments";
    }

    /**
     * @brief Try HSA SVM (Shared Virtual Memory) APIs
     *
     * HSA 2.0 supports SVM where the same virtual address is valid on both
     * CPU and GPU. If we can make the NVIDIA BAR appear in a shared address
     * space, AMD GPU might be able to access it.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryHSA_SVM_Coarse)
    {
        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: HSA SVM Coarse-Grain Memory                           ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Load HSA runtime
        void *hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hsa_handle)
        {
            hsa_handle = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_GLOBAL);
        }
        if (!hsa_handle)
        {
            GTEST_SKIP() << "Cannot load HSA runtime";
        }

        // Get NVIDIA BAR
        std::string nvidia_gpu = nvidia_gpus[0];
        PCIeBarInfo nvidia_info = PCIResourceParser::parseResourceFile(nvidia_gpu);

        size_t map_size = 4 * 1024 * 1024; // 4MB test
        void *nvidia_bar = PCIResourceParser::mmapBar(nvidia_gpu, nvidia_info.vram_bar_index, map_size);

        if (nvidia_bar == nullptr)
        {
            dlclose(hsa_handle);
            GTEST_SKIP() << "Cannot mmap NVIDIA VRAM BAR";
        }

        LOG_INFO("NVIDIA BAR mmap'd at " << nvidia_bar);

        // Try hsa_amd_memory_pool_allocate with specific address
        typedef int (*hsa_init_fn)(void);
        typedef int (*hsa_status_string_fn)(int status, const char **str);

        auto hsa_init = (hsa_init_fn)dlsym(hsa_handle, "hsa_init");
        auto hsa_status_string = (hsa_status_string_fn)dlsym(hsa_handle, "hsa_status_string");

        if (hsa_init)
        {
            int status = hsa_init();
            LOG_INFO("hsa_init() returned: " << status);
        }

        // Try hsa_amd_svm_attributes_set to make the BAR accessible
        typedef int (*hsa_amd_svm_attributes_set_fn)(void *ptr, size_t size, void *attrs, size_t num_attrs);
        auto hsa_amd_svm_attributes_set = (hsa_amd_svm_attributes_set_fn)dlsym(hsa_handle, "hsa_amd_svm_attributes_set");

        if (hsa_amd_svm_attributes_set)
        {
            LOG_INFO("Found hsa_amd_svm_attributes_set - trying to set SVM attributes...");

            // SVM attribute structure (from HSA headers)
            // HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG = 1
            // HSA_AMD_SVM_ATTRIB_ACCESS = 2
            struct
            {
                int attribute;
                uint64_t value;
            } attrs[2] = {
                {1, 1},  // GLOBAL_FLAG = 1 (accessible from any agent)
                {2, 0x7} // ACCESS = read|write|exec
            };

            int status = hsa_amd_svm_attributes_set(nvidia_bar, map_size, attrs, 2);
            LOG_INFO("hsa_amd_svm_attributes_set returned: " << status);

            if (status == 0)
            {
                LOG_INFO("✓ SVM attributes set successfully!");

                // Try DMA now
                rocm_backend.setDevice(0);
                void *hip_src = rocm_backend.allocate(4096, 0);

                std::vector<float> data(1024, 42.0f);
                rocm_backend.hostToDevice(hip_src, data.data(), 4096, 0);

                bool ok = rocm_backend.deviceToHost(nvidia_bar, hip_src, 4096, 0);
                LOG_INFO("D2H after SVM setup: " << (ok ? "SUCCESS" : "FAILED"));

                rocm_backend.free(hip_src, 0);
            }
        }
        else
        {
            LOG_INFO("hsa_amd_svm_attributes_set not found in HSA runtime");
        }

        // Try hsa_amd_vmem_map if available (virtual memory mapping)
        typedef int (*hsa_amd_vmem_map_fn)(void *va, size_t size, void *phys_addr, uint64_t offset, uint64_t permissions);
        auto hsa_amd_vmem_map = (hsa_amd_vmem_map_fn)dlsym(hsa_handle, "hsa_amd_vmem_map");

        if (hsa_amd_vmem_map)
        {
            LOG_INFO("\nFound hsa_amd_vmem_map - trying virtual memory mapping...");
            // Note: This would require knowing the exact physical address
            LOG_INFO("  (Would need physical address mapping support)");
        }

        munmap(nvidia_bar, map_size);
        dlclose(hsa_handle);

        SUCCEED() << "Completed HSA SVM experiments";
    }

    /**
     * @brief Try KFD (Kernel Fusion Driver) direct ioctls
     *
     * AMD's KFD provides low-level memory management ioctls. We might be able
     * to directly register the NVIDIA BAR physical address as a GPU-accessible
     * memory region.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryKFD_DirectMapping)
    {
        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: KFD Direct Memory Mapping                             ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Open KFD device
        int kfd_fd = open("/dev/kfd", O_RDWR);
        if (kfd_fd < 0)
        {
            GTEST_SKIP() << "Cannot open /dev/kfd: " << strerror(errno);
        }

        LOG_INFO("Opened /dev/kfd");

        // Get NVIDIA BAR physical address
        std::string nvidia_gpu = nvidia_gpus[0];
        PCIeBarInfo nvidia_info = PCIResourceParser::parseResourceFile(nvidia_gpu);

        LOG_INFO("NVIDIA BAR physical address: 0x" << std::hex << nvidia_info.vram_phys_addr << std::dec);
        LOG_INFO("NVIDIA BAR size: " << (nvidia_info.vram_size / (1024 * 1024 * 1024)) << " GB");

// KFD ioctl numbers (from linux kernel headers: drivers/gpu/drm/amd/amdkfd/kfd_ioctl.h)
// AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = 0x16
// AMDKFD_IOC_MAP_MEMORY_TO_GPU = 0x18

// Try to find GPU ID first
// AMDKFD_IOC_GET_PROCESS_APERTURES_NEW = 0x14
#define AMDKFD_IOCTL_BASE 'K'
#define AMDKFD_IOWR(nr, type) _IOWR(AMDKFD_IOCTL_BASE, nr, type)

        // Structure for getting apertures (simplified)
        struct kfd_ioctl_get_process_apertures_new_args
        {
            uint64_t kfd_process_device_apertures_ptr;
            uint32_t num_of_nodes;
            uint32_t pad;
        };

        struct kfd_process_device_apertures
        {
            uint64_t lds_base;
            uint64_t lds_limit;
            uint64_t scratch_base;
            uint64_t scratch_limit;
            uint64_t gpuvm_base;
            uint64_t gpuvm_limit;
            uint32_t gpu_id;
            uint32_t pad;
        };

        // Query apertures to get GPU ID
        struct kfd_process_device_apertures apertures[8] = {};
        struct kfd_ioctl_get_process_apertures_new_args aperture_args = {};
        aperture_args.kfd_process_device_apertures_ptr = (uint64_t)apertures;
        aperture_args.num_of_nodes = 8;

        int ioctl_num = _IOWR(AMDKFD_IOCTL_BASE, 0x14, struct kfd_ioctl_get_process_apertures_new_args);
        int ret = ioctl(kfd_fd, ioctl_num, &aperture_args);

        if (ret == 0)
        {
            LOG_INFO("Found " << aperture_args.num_of_nodes << " KFD nodes:");
            for (uint32_t i = 0; i < aperture_args.num_of_nodes && i < 8; ++i)
            {
                LOG_INFO("  Node " << i << ": gpu_id=" << apertures[i].gpu_id
                                   << " gpuvm_base=0x" << std::hex << apertures[i].gpuvm_base << std::dec);
            }
        }
        else
        {
            LOG_INFO("GET_PROCESS_APERTURES ioctl failed: " << strerror(errno));
        }

// Memory allocation flags from KFD
#define KFD_IOC_ALLOC_MEM_FLAGS_VRAM (1 << 0)
#define KFD_IOC_ALLOC_MEM_FLAGS_GTT (1 << 1)
#define KFD_IOC_ALLOC_MEM_FLAGS_USERPTR (1 << 2)
#define KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL (1 << 3)
#define KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP (1 << 4)
#define KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE (1 << 5)
#define KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE (1 << 6)
#define KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC (1 << 7)
#define KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE (1 << 8)
#define KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM (1 << 9)
#define KFD_IOC_ALLOC_MEM_FLAGS_COHERENT (1 << 10)
#define KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED (1 << 11)

        // Try MMIO_REMAP flag - this might allow mapping external MMIO regions!
        LOG_INFO("\nTrying KFD MMIO_REMAP allocation with NVIDIA BAR address...");

        struct kfd_ioctl_alloc_memory_of_gpu_args
        {
            uint64_t va_addr;     // Virtual address (in/out)
            uint64_t size;        // Size
            uint64_t handle;      // Handle (out)
            uint64_t mmap_offset; // mmap offset (out)
            uint32_t gpu_id;      // GPU ID
            uint32_t flags;       // Flags
        };

        if (aperture_args.num_of_nodes > 0)
        {
            // mmap the NVIDIA BAR first
            size_t map_size = 4 * 1024 * 1024;
            void *nvidia_bar = PCIResourceParser::mmapBar(nvidia_gpu, nvidia_info.vram_bar_index, map_size);

            if (nvidia_bar != nullptr)
            {
                struct kfd_ioctl_alloc_memory_of_gpu_args alloc_args = {};
                alloc_args.va_addr = (uint64_t)nvidia_bar; // Try to use the mmap'd address
                alloc_args.size = map_size;
                alloc_args.gpu_id = apertures[0].gpu_id;
                alloc_args.flags = KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;

                ioctl_num = _IOWR(AMDKFD_IOCTL_BASE, 0x16, struct kfd_ioctl_alloc_memory_of_gpu_args);
                ret = ioctl(kfd_fd, ioctl_num, &alloc_args);

                if (ret == 0)
                {
                    LOG_INFO("✓ KFD MMIO_REMAP allocation succeeded!");
                    LOG_INFO("  handle: 0x" << std::hex << alloc_args.handle << std::dec);
                    LOG_INFO("  mmap_offset: 0x" << std::hex << alloc_args.mmap_offset << std::dec);

                    // Try to map to GPU
                    struct kfd_ioctl_map_memory_to_gpu_args
                    {
                        uint64_t handle;
                        uint64_t device_ids_array_ptr;
                        uint32_t n_devices;
                        uint32_t n_success;
                    };

                    uint32_t gpu_ids[1] = {apertures[0].gpu_id};
                    struct kfd_ioctl_map_memory_to_gpu_args map_args = {};
                    map_args.handle = alloc_args.handle;
                    map_args.device_ids_array_ptr = (uint64_t)gpu_ids;
                    map_args.n_devices = 1;

                    ioctl_num = _IOWR(AMDKFD_IOCTL_BASE, 0x18, struct kfd_ioctl_map_memory_to_gpu_args);
                    ret = ioctl(kfd_fd, ioctl_num, &map_args);

                    if (ret == 0 && map_args.n_success > 0)
                    {
                        LOG_INFO("✓✓✓ KFD MAP_MEMORY_TO_GPU succeeded!");
                        LOG_INFO("  Now the NVIDIA BAR should be accessible from AMD GPU!");

                        // Try a copy!
                        rocm_backend.setDevice(0);
                        void *hip_src = rocm_backend.allocate(4096, 0);
                        std::vector<float> test_data(1024, 99.0f);
                        rocm_backend.hostToDevice(hip_src, test_data.data(), 4096, 0);

                        bool ok = rocm_backend.deviceToDevice(nvidia_bar, hip_src, 4096, 0);
                        LOG_INFO("D2D after KFD map: " << (ok ? "SUCCESS" : "FAILED"));

                        rocm_backend.free(hip_src, 0);
                    }
                    else
                    {
                        LOG_INFO("MAP_MEMORY_TO_GPU failed: " << strerror(errno));
                    }
                }
                else
                {
                    LOG_INFO("KFD MMIO_REMAP allocation failed: " << strerror(errno) << " (errno=" << errno << ")");
                }

                munmap(nvidia_bar, map_size);
            }
        }

        close(kfd_fd);
        SUCCEED() << "Completed KFD experiments";
    }

    /**
     * @brief Try hipMallocManaged with hints
     *
     * HIP's managed memory might have options to include external memory regions.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryManagedMemoryPrefetch)
    {
        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: hipMemAdvise on NVIDIA BAR                            ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Get NVIDIA BAR
        std::string nvidia_gpu = nvidia_gpus[0];
        PCIeBarInfo nvidia_info = PCIResourceParser::parseResourceFile(nvidia_gpu);

        size_t map_size = 4 * 1024 * 1024;
        void *nvidia_bar = PCIResourceParser::mmapBar(nvidia_gpu, nvidia_info.vram_bar_index, map_size);

        if (nvidia_bar == nullptr)
        {
            GTEST_SKIP() << "Cannot mmap NVIDIA VRAM BAR";
        }

        LOG_INFO("NVIDIA BAR mmap'd at " << nvidia_bar);

        // Load HIP symbols
        void *hip_handle = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hip_handle)
        {
            munmap(nvidia_bar, map_size);
            GTEST_SKIP() << "Cannot load HIP runtime";
        }

        // hipMemAdvise - give memory hints to the runtime
        typedef int (*hipMemAdvise_fn)(const void *, size_t, int, int);
        auto hipMemAdvise = (hipMemAdvise_fn)dlsym(hip_handle, "hipMemAdvise");

        if (hipMemAdvise)
        {
            LOG_INFO("Found hipMemAdvise - trying memory advices...");

            // hipMemAdviseSetReadMostly = 1
            // hipMemAdviseSetPreferredLocation = 2
            // hipMemAdviseSetAccessedBy = 3
            // hipMemAdviseSetCoarseGrain = 4

            int status = hipMemAdvise(nvidia_bar, map_size, 3, 0); // SetAccessedBy device 0
            LOG_INFO("hipMemAdvise(SetAccessedBy, device=0) returned: " << status);

            status = hipMemAdvise(nvidia_bar, map_size, 4, 0); // SetCoarseGrain
            LOG_INFO("hipMemAdvise(SetCoarseGrain) returned: " << status);
        }

        // hipMemPrefetchAsync - try to prefetch to device
        typedef int (*hipMemPrefetchAsync_fn)(const void *, size_t, int, void *);
        auto hipMemPrefetchAsync = (hipMemPrefetchAsync_fn)dlsym(hip_handle, "hipMemPrefetchAsync");

        if (hipMemPrefetchAsync)
        {
            LOG_INFO("\nTrying hipMemPrefetchAsync...");
            int status = hipMemPrefetchAsync(nvidia_bar, map_size, 0, nullptr);
            LOG_INFO("hipMemPrefetchAsync to device 0 returned: " << status);
        }

        // hipMemRangeGetAttribute - check attributes
        typedef int (*hipMemRangeGetAttribute_fn)(void *, size_t, int, const void *, size_t);
        auto hipMemRangeGetAttribute = (hipMemRangeGetAttribute_fn)dlsym(hip_handle, "hipMemRangeGetAttribute");

        if (hipMemRangeGetAttribute)
        {
            LOG_INFO("\nQuerying memory range attributes...");
            int location = -1;
            int status = hipMemRangeGetAttribute(&location, sizeof(location), 3, nvidia_bar, map_size);
            LOG_INFO("hipMemRangeGetAttribute(LastPrefetchLocation) returned: " << status << " value: " << location);
        }

        // Try DMA after advices
        LOG_INFO("\nTrying DMA after memory advices...");
        rocm_backend.setDevice(0);
        void *hip_src = rocm_backend.allocate(4096, 0);

        std::vector<float> data(1024, 77.0f);
        rocm_backend.hostToDevice(hip_src, data.data(), 4096, 0);

        bool ok = rocm_backend.deviceToHost(nvidia_bar, hip_src, 4096, 0);
        LOG_INFO("D2H after advices: " << (ok ? "SUCCESS" : "FAILED"));

        rocm_backend.free(hip_src, 0);
        dlclose(hip_handle);
        munmap(nvidia_bar, map_size);

        SUCCEED() << "Completed managed memory experiments";
    }

    /**
     * @brief Try using RDMA/GPUDirect APIs if available
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryRDMA_Verbs)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: Check for RDMA/GPUDirect Support                      ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Check if libibverbs is available
        void *verbs_handle = dlopen("libibverbs.so", RTLD_NOW);
        if (!verbs_handle)
        {
            verbs_handle = dlopen("libibverbs.so.1", RTLD_NOW);
        }

        if (verbs_handle)
        {
            LOG_INFO("✓ libibverbs found - RDMA support available");

            typedef void *(*ibv_get_device_list_fn)(int *);
            auto ibv_get_device_list = (ibv_get_device_list_fn)dlsym(verbs_handle, "ibv_get_device_list");

            if (ibv_get_device_list)
            {
                int num_devices = 0;
                void *devices = ibv_get_device_list(&num_devices);
                LOG_INFO("  Found " << num_devices << " RDMA devices");

                if (num_devices > 0)
                {
                    LOG_INFO("  RDMA could potentially be used for cross-GPU P2P");
                }
            }

            dlclose(verbs_handle);
        }
        else
        {
            LOG_INFO("✗ libibverbs not found - no RDMA support");
        }

        // Check for AMD's ROCr GPUDirect RDMA
        void *rocr_handle = dlopen("libhsa-runtime64.so", RTLD_NOW);
        if (rocr_handle)
        {
            // hsa_amd_ipc_memory_create / hsa_amd_ipc_memory_attach
            typedef int (*hsa_amd_ipc_memory_create_fn)(void *, size_t, void *, size_t *);
            auto hsa_amd_ipc_memory_create = (hsa_amd_ipc_memory_create_fn)dlsym(rocr_handle, "hsa_amd_ipc_memory_create");

            if (hsa_amd_ipc_memory_create)
            {
                LOG_INFO("✓ hsa_amd_ipc_memory_create found - IPC memory support available");
            }

            dlclose(rocr_handle);
        }

        // Check for NVIDIA GPUDirect peer memory
        std::ifstream gpudirect("/sys/kernel/mm/memory_peers/nv_mem/status");
        if (gpudirect.is_open())
        {
            std::string status;
            std::getline(gpudirect, status);
            LOG_INFO("NVIDIA GPUDirect peer memory status: " << status);
            gpudirect.close();
        }
        else
        {
            LOG_INFO("NVIDIA GPUDirect peer memory not available");
        }

        SUCCEED() << "Completed RDMA/GPUDirect check";
    }

    /**
     * @brief Comprehensive test: CPU write to NVIDIA BAR, verify, then try AMD GPU read
     *
     * This test verifies that:
     * 1. CPU can write to NVIDIA BAR (via mmap)
     * 2. CPU can read back from NVIDIA BAR (verify mmap works)
     * 3. AMD GPU can read from NVIDIA BAR via hipMemcpy
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryBidirectional_CPUWriteGPURead)
    {
        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: CPU Write → NVIDIA BAR → AMD GPU Read                 ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Get NVIDIA BAR
        std::string nvidia_gpu = nvidia_gpus[0];
        PCIeBarInfo nvidia_info = PCIResourceParser::parseResourceFile(nvidia_gpu);

        LOG_INFO("NVIDIA GPU: " << nvidia_gpu);
        LOG_INFO("  VRAM BAR" << nvidia_info.vram_bar_index << " phys: 0x" << std::hex
                              << nvidia_info.vram_phys_addr << std::dec);

        size_t map_size = 4 * 1024 * 1024; // 4MB
        size_t test_size = 4096;           // 4KB test

        void *nvidia_bar = PCIResourceParser::mmapBar(nvidia_gpu, nvidia_info.vram_bar_index, map_size);
        if (nvidia_bar == nullptr)
        {
            GTEST_SKIP() << "Cannot mmap NVIDIA VRAM BAR - need root access";
        }

        LOG_INFO("mmap'd " << (map_size / (1024 * 1024)) << " MB of NVIDIA VRAM at " << nvidia_bar);

        // Step 1: CPU writes test pattern to NVIDIA BAR
        LOG_INFO("\nStep 1: CPU writing test pattern to NVIDIA BAR...");
        float *bar_data = static_cast<float *>(nvidia_bar);
        for (size_t i = 0; i < test_size / sizeof(float); ++i)
        {
            bar_data[i] = 2.71828f * (i + 1); // e * (i+1) as test pattern
        }
        __sync_synchronize(); // Memory barrier

        LOG_INFO("  Wrote " << (test_size / sizeof(float)) << " floats");

        // Step 2: CPU reads back from NVIDIA BAR to verify mmap works
        LOG_INFO("\nStep 2: CPU reading back from NVIDIA BAR...");
        bool cpu_verify_ok = true;
        for (size_t i = 0; i < std::min((size_t)8, test_size / sizeof(float)); ++i)
        {
            float expected = 2.71828f * (i + 1);
            float actual = bar_data[i];
            LOG_INFO("  bar[" << i << "] = " << actual << " (expected " << expected << ")");
            if (std::abs(actual - expected) > 0.001f)
            {
                cpu_verify_ok = false;
            }
        }
        LOG_INFO("  CPU verification: " << (cpu_verify_ok ? "✓ PASSED" : "✗ FAILED"));

        // Step 3: AMD GPU reads from NVIDIA BAR (hipMemcpyHostToDevice)
        LOG_INFO("\nStep 3: AMD GPU reading from NVIDIA BAR (hipMemcpyHostToDevice)...");
        rocm_backend.setDevice(0);
        void *hip_dst = rocm_backend.allocate(test_size, 0);
        ASSERT_NE(hip_dst, nullptr) << "Failed to allocate HIP memory";

        // HIP treats the mmap'd BAR as "host" memory
        bool h2d_ok = rocm_backend.hostToDevice(hip_dst, nvidia_bar, test_size, 0);
        rocm_backend.synchronize(0);
        LOG_INFO("  hipMemcpyHostToDevice returned: " << (h2d_ok ? "SUCCESS" : "FAILED"));

        // Step 4: Read back from AMD GPU to CPU to verify
        LOG_INFO("\nStep 4: Reading back from AMD GPU to verify...");
        std::vector<float> readback(test_size / sizeof(float));
        bool d2h_ok = rocm_backend.deviceToHost(readback.data(), hip_dst, test_size, 0);
        rocm_backend.synchronize(0);

        bool gpu_verify_ok = true;
        for (size_t i = 0; i < std::min((size_t)8, readback.size()); ++i)
        {
            float expected = 2.71828f * (i + 1);
            LOG_INFO("  gpu_data[" << i << "] = " << readback[i] << " (expected " << expected << ")");
            if (std::abs(readback[i] - expected) > 0.001f)
            {
                gpu_verify_ok = false;
            }
        }

        if (gpu_verify_ok)
        {
            LOG_INFO("  ✓✓✓ AMD GPU SUCCESSFULLY READ FROM NVIDIA BAR! ✓✓✓");
        }
        else
        {
            LOG_INFO("  ✗ GPU data mismatch - P2P read may not be working");
        }

        // Cleanup
        rocm_backend.free(hip_dst, 0);
        munmap(nvidia_bar, map_size);

        SUCCEED() << "Completed bidirectional test";
    }

    /**
     * @brief Test using mmap with different flags (MAP_POPULATE, MAP_LOCKED)
     *
     * Maybe different mmap flags will make the mapping visible to HIP/HSA.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryMmapFlags)
    {
        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: Different mmap flags for NVIDIA BAR                   ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        std::string nvidia_gpu = nvidia_gpus[0];
        PCIeBarInfo nvidia_info = PCIResourceParser::parseResourceFile(nvidia_gpu);

        std::string resource_path = "/sys/bus/pci/devices/" + nvidia_gpu + "/resource" + std::to_string(nvidia_info.vram_bar_index);

        int fd = open(resource_path.c_str(), O_RDWR | O_SYNC);
        if (fd < 0)
        {
            GTEST_SKIP() << "Cannot open " << resource_path << " - need root";
        }

        size_t map_size = 4 * 1024 * 1024;

        // Try different mmap flag combinations
        struct MmapTest
        {
            int flags;
            const char *name;
        } tests[] = {
            {MAP_SHARED, "MAP_SHARED"},
            {MAP_SHARED | MAP_LOCKED, "MAP_SHARED | MAP_LOCKED"},
            {MAP_SHARED | MAP_POPULATE, "MAP_SHARED | MAP_POPULATE"},
            {MAP_SHARED | MAP_LOCKED | MAP_POPULATE, "MAP_SHARED | MAP_LOCKED | MAP_POPULATE"},
        };

        for (const auto &test : tests)
        {
            LOG_INFO("\nTrying " << test.name << "...");

            void *mapped = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, test.flags, fd, 0);
            if (mapped == MAP_FAILED)
            {
                LOG_INFO("  mmap failed: " << strerror(errno));
                continue;
            }

            LOG_INFO("  mmap succeeded at " << mapped);

            // Try writing and reading back
            float *data = static_cast<float *>(mapped);
            data[0] = 123.456f;
            __sync_synchronize();
            float readback = data[0];
            LOG_INFO("  Write 123.456, read back: " << readback);

            if (std::abs(readback - 123.456f) < 0.001f)
            {
                LOG_INFO("  ✓ R/W verified!");
            }
            else
            {
                LOG_INFO("  Data mismatch");
            }

            // Try mlock
            int mlock_result = mlock(mapped, map_size);
            LOG_INFO("  mlock result: " << mlock_result << " (errno=" << (mlock_result ? errno : 0) << ")");
            if (mlock_result == 0)
            {
                munlock(mapped, map_size);
            }

            munmap(mapped, map_size);
        }

        close(fd);
        SUCCEED() << "Completed mmap flags test";
    }

    /**
     * @brief Try registering the physical address of NVIDIA BAR via /dev/mem
     *
     * If we can mmap the physical address directly, maybe HSA will accept it.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryDevMem)
    {
        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: /dev/mem direct physical address mapping              ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        std::string nvidia_gpu = nvidia_gpus[0];
        PCIeBarInfo nvidia_info = PCIResourceParser::parseResourceFile(nvidia_gpu);

        LOG_INFO("NVIDIA BAR physical address: 0x" << std::hex << nvidia_info.vram_phys_addr << std::dec);

        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0)
        {
            LOG_INFO("/dev/mem not available: " << strerror(errno));
            LOG_INFO("This is expected - /dev/mem is restricted on modern kernels");
            SUCCEED() << "/dev/mem not available (expected)";
            return;
        }

        LOG_INFO("Opened /dev/mem successfully");

        // Try to mmap the NVIDIA BAR physical address
        size_t map_size = 4096; // Start with a small page
        void *mapped = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                            fd, nvidia_info.vram_phys_addr);

        if (mapped == MAP_FAILED)
        {
            LOG_INFO("mmap of physical address failed: " << strerror(errno));
            close(fd);
            SUCCEED() << "Physical address mmap failed (expected on secure systems)";
            return;
        }

        LOG_INFO("mmap'd physical address at " << mapped);

        // Try a read
        uint32_t *data = static_cast<uint32_t *>(mapped);
        LOG_INFO("First word: 0x" << std::hex << data[0] << std::dec);

        munmap(mapped, map_size);
        close(fd);

        SUCCEED() << "Completed /dev/mem test";
    }

    /**
     * @brief Try resource1_wc (write-combining) instead of resource1
     *
     * The kernel exposes a write-combining version of the BAR that might
     * allow actual writes to persist.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryWriteCombiningBAR)
    {
        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: Write-Combining BAR (resource1_wc)                    ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        std::string nvidia_gpu = nvidia_gpus[0];
        PCIeBarInfo nvidia_info = PCIResourceParser::parseResourceFile(nvidia_gpu);

        // Try resource1_wc (write-combining) first
        std::string wc_path = "/sys/bus/pci/devices/" + nvidia_gpu + "/resource1_wc";
        std::string normal_path = "/sys/bus/pci/devices/" + nvidia_gpu + "/resource1";

        LOG_INFO("Trying write-combining BAR: " << wc_path);

        int fd = open(wc_path.c_str(), O_RDWR | O_SYNC);
        if (fd < 0)
        {
            LOG_INFO("Cannot open resource1_wc: " << strerror(errno));
            LOG_INFO("Falling back to resource1...");
            fd = open(normal_path.c_str(), O_RDWR | O_SYNC);
            if (fd < 0)
            {
                GTEST_SKIP() << "Cannot open either resource1_wc or resource1";
            }
        }
        else
        {
            LOG_INFO("Opened resource1_wc successfully");
        }

        size_t map_size = 256 * 1024 * 1024; // 256MB (file size is 268MB)
        size_t test_size = 4096;

        void *nvidia_bar = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);

        if (nvidia_bar == MAP_FAILED)
        {
            GTEST_SKIP() << "mmap failed: " << strerror(errno);
        }

        LOG_INFO("mmap'd " << (map_size / (1024 * 1024)) << " MB at " << nvidia_bar);

        // Step 1: CPU writes test pattern
        LOG_INFO("\nStep 1: CPU writing test pattern...");
        float *bar_data = static_cast<float *>(nvidia_bar);
        for (size_t i = 0; i < test_size / sizeof(float); ++i)
        {
            bar_data[i] = 2.71828f * (i + 1);
        }
        // Memory barrier and flush
        __sync_synchronize();
        asm volatile("sfence" ::: "memory");

        LOG_INFO("  Wrote " << (test_size / sizeof(float)) << " floats with sfence");

        // Step 2: CPU reads back
        LOG_INFO("\nStep 2: CPU reading back...");
        bool cpu_verify_ok = true;
        for (size_t i = 0; i < std::min((size_t)8, test_size / sizeof(float)); ++i)
        {
            float expected = 2.71828f * (i + 1);
            volatile float actual = bar_data[i]; // volatile to prevent caching
            LOG_INFO("  bar[" << i << "] = " << actual << " (expected " << expected << ")");
            if (std::abs(actual - expected) > 0.001f)
            {
                cpu_verify_ok = false;
            }
        }
        LOG_INFO("  CPU verification: " << (cpu_verify_ok ? "✓ PASSED" : "✗ FAILED"));

        if (cpu_verify_ok)
        {
            // Step 3: AMD GPU reads from the BAR
            LOG_INFO("\nStep 3: AMD GPU reading from NVIDIA BAR...");
            rocm_backend.setDevice(0);
            void *hip_dst = rocm_backend.allocate(test_size, 0);

            bool h2d_ok = rocm_backend.hostToDevice(hip_dst, nvidia_bar, test_size, 0);
            rocm_backend.synchronize(0);
            LOG_INFO("  hipMemcpyHostToDevice: " << (h2d_ok ? "SUCCESS" : "FAILED"));

            // Step 4: Read back from GPU
            std::vector<float> readback(test_size / sizeof(float));
            rocm_backend.deviceToHost(readback.data(), hip_dst, test_size, 0);
            rocm_backend.synchronize(0);

            bool gpu_verify_ok = true;
            for (size_t i = 0; i < std::min((size_t)8, readback.size()); ++i)
            {
                float expected = 2.71828f * (i + 1);
                LOG_INFO("  gpu_data[" << i << "] = " << readback[i] << " (expected " << expected << ")");
                if (std::abs(readback[i] - expected) > 0.001f)
                {
                    gpu_verify_ok = false;
                }
            }

            if (gpu_verify_ok)
            {
                LOG_INFO("  ✓✓✓ SUCCESS! AMD GPU can read from NVIDIA VRAM via WC BAR! ✓✓✓");
            }
            else
            {
                LOG_INFO("  GPU verification failed");
            }

            rocm_backend.free(hip_dst, 0);
        }

        munmap(nvidia_bar, map_size);
        SUCCEED() << "Completed write-combining BAR test";
    }

    //==========================================================================
    // CUDA API Exploration Tests
    //==========================================================================

    /**
     * @brief Explore CUDA driver API for memory export capabilities
     *
     * CUDA provides several mechanisms for exporting memory:
     * 1. cuMemExportToShareableHandle - Export to OS handle (fd on Linux)
     * 2. cuIpcGetMemHandle - IPC memory handle for cross-process sharing
     * 3. cuMemGetAllocationGranularity - Query allocation requirements
     * 4. cuMemCreate with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, CUDA_ExploreMemoryExportAPIs)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: CUDA Memory Export APIs                               ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Load CUDA driver API
        void *cuda_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        if (!cuda_handle)
        {
            cuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        }
        if (!cuda_handle)
        {
            GTEST_SKIP() << "Cannot load CUDA driver library";
        }

        LOG_INFO("Loaded libcuda.so successfully");

        // Get function pointers
        typedef int (*cuInit_fn)(unsigned int);
        typedef int (*cuDeviceGet_fn)(int *, int);
        typedef int (*cuCtxCreate_fn)(void **, unsigned int, int);
        typedef int (*cuCtxDestroy_fn)(void *);
        typedef int (*cuMemAlloc_fn)(uint64_t *, size_t);
        typedef int (*cuMemFree_fn)(uint64_t);
        typedef int (*cuMemsetD8_fn)(uint64_t, unsigned char, size_t);
        typedef int (*cuGetErrorString_fn)(int, const char **);
        typedef int (*cuDeviceGetAttribute_fn)(int *, int, int);
        typedef int (*cuPointerGetAttribute_fn)(void *, int, uint64_t);

        // IPC APIs
        typedef int (*cuIpcGetMemHandle_fn)(void *, uint64_t);
        typedef int (*cuIpcOpenMemHandle_fn)(uint64_t *, void *, void *, unsigned int);

        // Virtual memory management APIs (CUDA 10.2+)
        typedef int (*cuMemCreate_fn)(uint64_t *, size_t, void *, unsigned long long);
        typedef int (*cuMemExportToShareableHandle_fn)(void *, uint64_t, int, unsigned long long);
        typedef int (*cuMemGetAllocationGranularity_fn)(size_t *, void *, int);
        typedef int (*cuMemAddressReserve_fn)(uint64_t *, size_t, size_t, uint64_t, unsigned long long);
        typedef int (*cuMemMap_fn)(uint64_t, size_t, size_t, uint64_t, unsigned long long);
        typedef int (*cuMemSetAccess_fn)(uint64_t, size_t, void *, size_t);
        typedef int (*cuMemRelease_fn)(uint64_t);
        typedef int (*cuMemUnmap_fn)(uint64_t, size_t);
        typedef int (*cuMemAddressFree_fn)(uint64_t, size_t);

        auto cuInit = (cuInit_fn)dlsym(cuda_handle, "cuInit");
        auto cuDeviceGet = (cuDeviceGet_fn)dlsym(cuda_handle, "cuDeviceGet");
        auto cuCtxCreate = (cuCtxCreate_fn)dlsym(cuda_handle, "cuCtxCreate_v2");
        auto cuCtxDestroy = (cuCtxDestroy_fn)dlsym(cuda_handle, "cuCtxDestroy_v2");
        auto cuMemAlloc = (cuMemAlloc_fn)dlsym(cuda_handle, "cuMemAlloc_v2");
        auto cuMemFree = (cuMemFree_fn)dlsym(cuda_handle, "cuMemFree_v2");
        auto cuMemsetD8 = (cuMemsetD8_fn)dlsym(cuda_handle, "cuMemsetD8_v2");
        auto cuGetErrorString = (cuGetErrorString_fn)dlsym(cuda_handle, "cuGetErrorString");
        auto cuDeviceGetAttribute = (cuDeviceGetAttribute_fn)dlsym(cuda_handle, "cuDeviceGetAttribute");
        auto cuPointerGetAttribute = (cuPointerGetAttribute_fn)dlsym(cuda_handle, "cuPointerGetAttribute");
        auto cuIpcGetMemHandle = (cuIpcGetMemHandle_fn)dlsym(cuda_handle, "cuIpcGetMemHandle");
        auto cuMemCreate = (cuMemCreate_fn)dlsym(cuda_handle, "cuMemCreate");
        auto cuMemExportToShareableHandle = (cuMemExportToShareableHandle_fn)dlsym(cuda_handle, "cuMemExportToShareableHandle");
        auto cuMemGetAllocationGranularity = (cuMemGetAllocationGranularity_fn)dlsym(cuda_handle, "cuMemGetAllocationGranularity");
        auto cuMemAddressReserve = (cuMemAddressReserve_fn)dlsym(cuda_handle, "cuMemAddressReserve");
        auto cuMemMap = (cuMemMap_fn)dlsym(cuda_handle, "cuMemMap");
        auto cuMemSetAccess = (cuMemSetAccess_fn)dlsym(cuda_handle, "cuMemSetAccess");
        auto cuMemRelease = (cuMemRelease_fn)dlsym(cuda_handle, "cuMemRelease");
        auto cuMemUnmap = (cuMemUnmap_fn)dlsym(cuda_handle, "cuMemUnmap");
        auto cuMemAddressFree = (cuMemAddressFree_fn)dlsym(cuda_handle, "cuMemAddressFree");

        // Helper lambda for error checking
        auto check_cuda = [&](int result, const char *call)
        {
            if (result != 0)
            {
                const char *err_str = "unknown";
                if (cuGetErrorString)
                    cuGetErrorString(result, &err_str);
                LOG_INFO("  " << call << " failed: " << result << " (" << err_str << ")");
                return false;
            }
            LOG_INFO("  " << call << " succeeded");
            return true;
        };

        // Initialize CUDA
        if (!cuInit || !check_cuda(cuInit(0), "cuInit"))
        {
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuInit failed";
        }

        // Get device
        int device = 0;
        if (!cuDeviceGet || !check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet"))
        {
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuDeviceGet failed";
        }

        // Check device attributes
        LOG_INFO("\nDevice Attributes:");
        if (cuDeviceGetAttribute)
        {
            int value = 0;

            // CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
            if (cuDeviceGetAttribute(&value, 102, device) == 0)
            {
                LOG_INFO("  Virtual Memory Management: " << (value ? "YES" : "NO"));
            }

            // CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
            if (cuDeviceGetAttribute(&value, 103, device) == 0)
            {
                LOG_INFO("  POSIX FD Handle Type: " << (value ? "YES" : "NO"));
            }

            // CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
            if (cuDeviceGetAttribute(&value, 41, device) == 0)
            {
                LOG_INFO("  Unified Addressing: " << (value ? "YES" : "NO"));
            }

            // CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116
            if (cuDeviceGetAttribute(&value, 116, device) == 0)
            {
                LOG_INFO("  GPUDirect RDMA: " << (value ? "YES" : "NO"));
            }

            // CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 119
            if (cuDeviceGetAttribute(&value, 119, device) == 0)
            {
                LOG_INFO("  GPUDirect RDMA with VMM: " << (value ? "YES" : "NO"));
            }
        }

        // Create context
        void *ctx = nullptr;
        if (!cuCtxCreate || !check_cuda(cuCtxCreate(&ctx, 0, device), "cuCtxCreate"))
        {
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuCtxCreate failed";
        }

        // Test 1: Traditional cuMemAlloc + cuIpcGetMemHandle
        LOG_INFO("\n--- Test 1: cuIpcGetMemHandle ---");
        uint64_t dptr = 0;
        size_t alloc_size = 64 * 1024 * 1024; // 64MB

        if (cuMemAlloc && check_cuda(cuMemAlloc(&dptr, alloc_size), "cuMemAlloc(64MB)"))
        {
            LOG_INFO("  Allocated at device ptr: 0x" << std::hex << dptr << std::dec);

            // Initialize with pattern
            if (cuMemsetD8)
            {
                cuMemsetD8(dptr, 0xAB, alloc_size);
            }

            // Try to get IPC handle
            if (cuIpcGetMemHandle)
            {
                char ipc_handle[64] = {0}; // CUipcMemHandle is 64 bytes
                int result = cuIpcGetMemHandle(ipc_handle, dptr);
                if (result == 0)
                {
                    LOG_INFO("  ✓ cuIpcGetMemHandle succeeded!");
                    LOG_INFO("    IPC handle (first 16 bytes): " << std::hex);
                    for (int i = 0; i < 16; i++)
                    {
                        LOG_INFO("      [" << i << "] = 0x" << (unsigned)(unsigned char)ipc_handle[i]);
                    }
                    LOG_INFO(std::dec);

                    // The IPC handle could potentially be imported by HIP
                    // using hipIpcOpenMemHandle if there was cross-vendor support
                }
                else
                {
                    const char *err_str = "unknown";
                    if (cuGetErrorString)
                        cuGetErrorString(result, &err_str);
                    LOG_INFO("  cuIpcGetMemHandle failed: " << result << " (" << err_str << ")");
                }
            }

            // Get pointer attributes
            if (cuPointerGetAttribute)
            {
                // CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
                unsigned int mem_type = 0;
                if (cuPointerGetAttribute(&mem_type, 2, dptr) == 0)
                {
                    LOG_INFO("  Memory type: " << mem_type << " (1=host, 2=device, 3=array, 4=unified)");
                }

                // CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 13
                unsigned int rdma_capable = 0;
                if (cuPointerGetAttribute(&rdma_capable, 13, dptr) == 0)
                {
                    LOG_INFO("  GPUDirect RDMA capable: " << (rdma_capable ? "YES" : "NO"));
                }
            }

            if (cuMemFree)
                cuMemFree(dptr);
        }

        // Test 2: Virtual Memory Management API (cuMemCreate + cuMemExportToShareableHandle)
        LOG_INFO("\n--- Test 2: cuMemCreate + cuMemExportToShareableHandle ---");

        if (cuMemCreate && cuMemExportToShareableHandle && cuMemGetAllocationGranularity)
        {
            // CU_MEM_ALLOCATION_PROP structure (simplified)
            struct
            {
                int type;                 // CU_MEM_ALLOCATION_TYPE_PINNED = 1
                int requestedHandleTypes; // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
                struct
                {
                    int type; // CU_MEM_LOCATION_TYPE_DEVICE = 1
                    int id;   // device id
                } location;
                void *win32HandleMetaData;
                uint64_t reserved[8];
            } alloc_prop = {0};

            alloc_prop.type = 1;                 // CU_MEM_ALLOCATION_TYPE_PINNED
            alloc_prop.requestedHandleTypes = 1; // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            alloc_prop.location.type = 1;        // CU_MEM_LOCATION_TYPE_DEVICE
            alloc_prop.location.id = device;

            // Get allocation granularity
            size_t granularity = 0;
            int result = cuMemGetAllocationGranularity(&granularity, &alloc_prop, 1); // CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            if (result == 0)
            {
                LOG_INFO("  Allocation granularity: " << granularity << " bytes");
            }
            else
            {
                const char *err_str = "unknown";
                if (cuGetErrorString)
                    cuGetErrorString(result, &err_str);
                LOG_INFO("  cuMemGetAllocationGranularity failed: " << result << " (" << err_str << ")");
            }

            // Round up allocation size
            size_t vmm_size = ((alloc_size + granularity - 1) / granularity) * granularity;
            if (vmm_size == 0)
                vmm_size = 64 * 1024 * 1024;

            // Create allocation
            uint64_t alloc_handle = 0;
            result = cuMemCreate(&alloc_handle, vmm_size, &alloc_prop, 0);
            if (result == 0)
            {
                LOG_INFO("  ✓ cuMemCreate succeeded! handle=0x" << std::hex << alloc_handle << std::dec);

                // Try to export to file descriptor
                int fd = -1;
                // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
                result = cuMemExportToShareableHandle(&fd, alloc_handle, 1, 0);
                if (result == 0)
                {
                    LOG_INFO("  ✓✓ cuMemExportToShareableHandle succeeded! fd=" << fd);

                    // This fd could potentially be imported by HIP!
                    // Let's try some operations on it

                    struct stat st;
                    if (fstat(fd, &st) == 0)
                    {
                        LOG_INFO("    fd stat: mode=0" << std::oct << st.st_mode << std::dec
                                                       << " size=" << st.st_size);
                    }

                    // Try to mmap the fd
                    void *mapped = mmap(nullptr, vmm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                    if (mapped != MAP_FAILED)
                    {
                        LOG_INFO("    ✓ mmap of exported fd succeeded at " << mapped);

                        // Try reading
                        uint32_t *data = static_cast<uint32_t *>(mapped);
                        LOG_INFO("    First word: 0x" << std::hex << data[0] << std::dec);

                        munmap(mapped, vmm_size);
                    }
                    else
                    {
                        LOG_INFO("    mmap failed: " << strerror(errno));
                    }

                    // Save the fd for HIP import test
                    LOG_INFO("\n    Now attempting to import this fd into HIP...");

#ifdef HAVE_ROCM
                    ROCmBackend rocm_backend;
                    if (rocm_backend.deviceCount() > 0)
                    {
                        rocm_backend.setDevice(0);

                        // Try hipImportExternalMemory
                        void *hip_handle = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
                        if (hip_handle)
                        {
                            typedef int (*hipImportExternalMemory_fn)(void **, void *);
                            typedef int (*hipExternalMemoryGetMappedBuffer_fn)(void **, void *, void *);

                            auto hipImportExternalMemory = (hipImportExternalMemory_fn)dlsym(hip_handle, "hipImportExternalMemory");
                            auto hipExternalMemoryGetMappedBuffer = (hipExternalMemoryGetMappedBuffer_fn)dlsym(hip_handle, "hipExternalMemoryGetMappedBuffer");

                            if (hipImportExternalMemory)
                            {
                                // hipExternalMemoryHandleDesc structure
                                struct
                                {
                                    int type; // hipExternalMemoryHandleTypeOpaqueFd = 1
                                    union
                                    {
                                        int fd;
                                        struct
                                        {
                                            void *handle;
                                            void *name;
                                        } win32;
                                    } handle;
                                    uint64_t size;
                                    unsigned int flags;
                                    unsigned int reserved[16];
                                } ext_mem_desc = {0};

                                ext_mem_desc.type = 1; // hipExternalMemoryHandleTypeOpaqueFd
                                ext_mem_desc.handle.fd = fd;
                                ext_mem_desc.size = vmm_size;
                                ext_mem_desc.flags = 0;

                                void *ext_mem = nullptr;
                                int hip_result = hipImportExternalMemory(&ext_mem, &ext_mem_desc);
                                if (hip_result == 0 && ext_mem != nullptr)
                                {
                                    LOG_INFO("    ✓✓✓ hipImportExternalMemory SUCCEEDED!");
                                    LOG_INFO("    External memory handle: " << ext_mem);

                                    // Try to get mapped buffer
                                    if (hipExternalMemoryGetMappedBuffer)
                                    {
                                        struct
                                        {
                                            uint64_t offset;
                                            uint64_t size;
                                            unsigned int flags;
                                            unsigned int reserved[16];
                                        } buffer_desc = {0, vmm_size, 0};

                                        void *hip_ptr = nullptr;
                                        hip_result = hipExternalMemoryGetMappedBuffer(&hip_ptr, ext_mem, &buffer_desc);
                                        if (hip_result == 0)
                                        {
                                            LOG_INFO("    ✓✓✓✓ hipExternalMemoryGetMappedBuffer SUCCEEDED!");
                                            LOG_INFO("    HIP device pointer: " << hip_ptr);

                                            // This would be AMAZING - we'd have cross-vendor P2P!
                                        }
                                        else
                                        {
                                            LOG_INFO("    hipExternalMemoryGetMappedBuffer failed: " << hip_result);
                                        }
                                    }
                                }
                                else
                                {
                                    LOG_INFO("    hipImportExternalMemory failed: " << hip_result);
                                }
                            }

                            dlclose(hip_handle);
                        }
                    }
#endif

                    close(fd);
                }
                else
                {
                    const char *err_str = "unknown";
                    if (cuGetErrorString)
                        cuGetErrorString(result, &err_str);
                    LOG_INFO("  cuMemExportToShareableHandle failed: " << result << " (" << err_str << ")");
                }

                if (cuMemRelease)
                    cuMemRelease(alloc_handle);
            }
            else
            {
                const char *err_str = "unknown";
                if (cuGetErrorString)
                    cuGetErrorString(result, &err_str);
                LOG_INFO("  cuMemCreate failed: " << result << " (" << err_str << ")");
            }
        }
        else
        {
            LOG_INFO("  Virtual memory management APIs not available");
            if (!cuMemCreate)
                LOG_INFO("    - cuMemCreate not found");
            if (!cuMemExportToShareableHandle)
                LOG_INFO("    - cuMemExportToShareableHandle not found");
            if (!cuMemGetAllocationGranularity)
                LOG_INFO("    - cuMemGetAllocationGranularity not found");
        }

        // Cleanup
        if (cuCtxDestroy)
            cuCtxDestroy(ctx);
        dlclose(cuda_handle);

        SUCCEED() << "Completed CUDA memory export API exploration";
    }

    /**
     * @brief Try CUDA's dmabuf export (newer API)
     *
     * CUDA 11.4+ supports exporting memory as a Linux dmabuf,
     * which is a more standard format that might be accepted by HIP.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, CUDA_DmabufExport)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: CUDA dmabuf Export                                    ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        void *cuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!cuda_handle)
        {
            GTEST_SKIP() << "Cannot load CUDA driver library";
        }

        // Function pointers
        typedef int (*cuInit_fn)(unsigned int);
        typedef int (*cuDeviceGet_fn)(int *, int);
        typedef int (*cuCtxCreate_fn)(void **, unsigned int, int);
        typedef int (*cuCtxDestroy_fn)(void *);
        typedef int (*cuMemAlloc_fn)(uint64_t *, size_t);
        typedef int (*cuMemFree_fn)(uint64_t);
        typedef int (*cuGetErrorString_fn)(int, const char **);
        typedef int (*cuMemGetHandleForAddressRange_fn)(void *, uint64_t, size_t, int, unsigned long long);

        auto cuInit = (cuInit_fn)dlsym(cuda_handle, "cuInit");
        auto cuDeviceGet = (cuDeviceGet_fn)dlsym(cuda_handle, "cuDeviceGet");
        auto cuCtxCreate = (cuCtxCreate_fn)dlsym(cuda_handle, "cuCtxCreate_v2");
        auto cuCtxDestroy = (cuCtxDestroy_fn)dlsym(cuda_handle, "cuCtxDestroy_v2");
        auto cuMemAlloc = (cuMemAlloc_fn)dlsym(cuda_handle, "cuMemAlloc_v2");
        auto cuMemFree = (cuMemFree_fn)dlsym(cuda_handle, "cuMemFree_v2");
        auto cuGetErrorString = (cuGetErrorString_fn)dlsym(cuda_handle, "cuGetErrorString");
        auto cuMemGetHandleForAddressRange = (cuMemGetHandleForAddressRange_fn)dlsym(cuda_handle, "cuMemGetHandleForAddressRange");

        if (!cuMemGetHandleForAddressRange)
        {
            LOG_INFO("cuMemGetHandleForAddressRange not found (requires CUDA 11.4+)");
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuMemGetHandleForAddressRange not available";
        }

        LOG_INFO("Found cuMemGetHandleForAddressRange - trying dmabuf export...");

        // Initialize
        if (cuInit(0) != 0)
        {
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuInit failed";
        }

        int device = 0;
        cuDeviceGet(&device, 0);

        void *ctx = nullptr;
        if (cuCtxCreate(&ctx, 0, device) != 0)
        {
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuCtxCreate failed";
        }

        // Allocate memory
        uint64_t dptr = 0;
        size_t alloc_size = 64 * 1024 * 1024;

        if (cuMemAlloc(&dptr, alloc_size) != 0)
        {
            cuCtxDestroy(ctx);
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuMemAlloc failed";
        }

        LOG_INFO("Allocated CUDA memory at 0x" << std::hex << dptr << std::dec);

        // Try to get dmabuf handle
        // CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1
        int dmabuf_fd = -1;
        int result = cuMemGetHandleForAddressRange(&dmabuf_fd, dptr, alloc_size, 1, 0);

        if (result == 0)
        {
            LOG_INFO("✓ cuMemGetHandleForAddressRange succeeded! dmabuf_fd=" << dmabuf_fd);

            // Try to mmap the dmabuf
            void *mapped = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
            if (mapped != MAP_FAILED)
            {
                LOG_INFO("  ✓ mmap of dmabuf succeeded at " << mapped);
                uint32_t *data = static_cast<uint32_t *>(mapped);
                LOG_INFO("  First word: 0x" << std::hex << data[0] << std::dec);
                munmap(mapped, alloc_size);
            }
            else
            {
                LOG_INFO("  mmap of dmabuf failed: " << strerror(errno));
            }

#ifdef HAVE_ROCM
            // Try importing into HIP
            LOG_INFO("\nTrying to import dmabuf into HIP...");

            ROCmBackend rocm_backend;
            if (rocm_backend.deviceCount() > 0)
            {
                // Try hsa_amd_interop_map_buffer
                void *hsa_handle = dlopen("libhsa-runtime64.so.1", RTLD_NOW);
                if (hsa_handle)
                {
                    typedef int (*hsa_amd_interop_map_buffer_fn)(int, int, uint64_t *, size_t *, void *);
                    auto hsa_amd_interop_map_buffer = (hsa_amd_interop_map_buffer_fn)dlsym(hsa_handle, "hsa_amd_interop_map_buffer");

                    if (hsa_amd_interop_map_buffer)
                    {
                        uint64_t mapped_ptr = 0;
                        size_t mapped_size = 0;

                        // num_agents=1, agents=first AMD GPU, dmabuf_fd
                        int hsa_result = hsa_amd_interop_map_buffer(1, dmabuf_fd, &mapped_ptr, &mapped_size, nullptr);
                        if (hsa_result == 0)
                        {
                            LOG_INFO("  ✓✓✓ hsa_amd_interop_map_buffer SUCCEEDED!");
                            LOG_INFO("  Mapped ptr: 0x" << std::hex << mapped_ptr << std::dec);
                            LOG_INFO("  Mapped size: " << mapped_size);
                        }
                        else
                        {
                            LOG_INFO("  hsa_amd_interop_map_buffer failed: " << hsa_result);
                        }
                    }

                    dlclose(hsa_handle);
                }
            }
#endif

            close(dmabuf_fd);
        }
        else
        {
            const char *err_str = "unknown";
            if (cuGetErrorString)
                cuGetErrorString(result, &err_str);
            LOG_INFO("cuMemGetHandleForAddressRange failed: " << result << " (" << err_str << ")");
        }

        cuMemFree(dptr);
        cuCtxDestroy(ctx);
        dlclose(cuda_handle);

        SUCCEED() << "Completed CUDA dmabuf export test";
    }

    /**
     * @brief Check if CUDA exposes the physical BAR address
     *
     * If we can get the actual physical address that CUDA is using,
     * we might be able to use that for AMD page table mapping.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, CUDA_GetPhysicalAddress)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: Query CUDA Physical Addresses                         ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        void *cuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!cuda_handle)
        {
            GTEST_SKIP() << "Cannot load CUDA driver library";
        }

        typedef int (*cuInit_fn)(unsigned int);
        typedef int (*cuDeviceGet_fn)(int *, int);
        typedef int (*cuCtxCreate_fn)(void **, unsigned int, int);
        typedef int (*cuCtxDestroy_fn)(void *);
        typedef int (*cuMemAlloc_fn)(uint64_t *, size_t);
        typedef int (*cuMemFree_fn)(uint64_t);
        typedef int (*cuPointerGetAttribute_fn)(void *, int, uint64_t);
        typedef int (*cuPointerGetAttributes_fn)(unsigned int, int *, void **, uint64_t);

        auto cuInit = (cuInit_fn)dlsym(cuda_handle, "cuInit");
        auto cuDeviceGet = (cuDeviceGet_fn)dlsym(cuda_handle, "cuDeviceGet");
        auto cuCtxCreate = (cuCtxCreate_fn)dlsym(cuda_handle, "cuCtxCreate_v2");
        auto cuCtxDestroy = (cuCtxDestroy_fn)dlsym(cuda_handle, "cuCtxDestroy_v2");
        auto cuMemAlloc = (cuMemAlloc_fn)dlsym(cuda_handle, "cuMemAlloc_v2");
        auto cuMemFree = (cuMemFree_fn)dlsym(cuda_handle, "cuMemFree_v2");
        auto cuPointerGetAttribute = (cuPointerGetAttribute_fn)dlsym(cuda_handle, "cuPointerGetAttribute");
        auto cuPointerGetAttributes = (cuPointerGetAttributes_fn)dlsym(cuda_handle, "cuPointerGetAttributes");

        if (cuInit(0) != 0)
        {
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuInit failed";
        }

        int device = 0;
        cuDeviceGet(&device, 0);

        void *ctx = nullptr;
        cuCtxCreate(&ctx, 0, device);

        uint64_t dptr = 0;
        size_t alloc_size = 64 * 1024 * 1024;
        cuMemAlloc(&dptr, alloc_size);

        LOG_INFO("CUDA device pointer: 0x" << std::hex << dptr << std::dec);

        if (cuPointerGetAttribute)
        {
            LOG_INFO("\nQuerying pointer attributes...");

            // Try all known attribute types
            struct
            {
                int id;
                const char *name;
            } attrs[] = {
                {1, "CONTEXT"},
                {2, "MEMORY_TYPE"},
                {3, "DEVICE_POINTER"},
                {4, "HOST_POINTER"},
                {5, "P2P_TOKENS"},
                {6, "SYNC_MEMOPS"},
                {7, "BUFFER_ID"},
                {8, "IS_MANAGED"},
                {9, "DEVICE_ORDINAL"},
                {10, "IS_LEGACY_CUDA_IPC_CAPABLE"},
                {11, "RANGE_START_ADDR"},
                {12, "RANGE_SIZE"},
                {13, "MAPPED"},
                {14, "ALLOWED_HANDLE_TYPES"},
                {15, "IS_GPU_DIRECT_RDMA_CAPABLE"},
                {16, "ACCESS_FLAGS"},
                {17, "MEMPOOL_HANDLE"},
            };

            for (const auto &attr : attrs)
            {
                uint64_t value = 0;
                int result = cuPointerGetAttribute(&value, attr.id, dptr);
                if (result == 0)
                {
                    LOG_INFO("  " << attr.name << " (" << attr.id << "): 0x" << std::hex << value << std::dec);
                }
            }
        }

        // Check /proc/self/maps for CUDA mappings
        LOG_INFO("\nChecking /proc/self/maps for CUDA-related mappings...");
        std::ifstream maps("/proc/self/maps");
        std::string line;
        int cuda_maps = 0;
        while (std::getline(maps, line))
        {
            if (line.find("nvidia") != std::string::npos ||
                line.find("cuda") != std::string::npos ||
                line.find("/dev/") != std::string::npos)
            {
                if (cuda_maps < 10)
                {
                    LOG_INFO("  " << line);
                }
                cuda_maps++;
            }
        }
        LOG_INFO("  (Found " << cuda_maps << " CUDA-related mappings)");

        // Check if we can find the BAR address in the device pointer
        // NVIDIA device pointers are virtual addresses in GPU space
        // The physical BAR address is at 0x3a0000000000 for our GPU
        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (!nvidia_gpus.empty())
        {
            PCIeBarInfo info = PCIResourceParser::parseResourceFile(nvidia_gpus[0]);
            LOG_INFO("\nComparing addresses:");
            LOG_INFO("  BAR1 physical address: 0x" << std::hex << info.vram_phys_addr);
            LOG_INFO("  CUDA device pointer:   0x" << dptr);

            // The difference tells us the offset into VRAM
            if (dptr > info.vram_phys_addr)
            {
                LOG_INFO("  Difference: 0x" << (dptr - info.vram_phys_addr) << std::dec);
            }
        }

        cuMemFree(dptr);
        cuCtxDestroy(ctx);
        dlclose(cuda_handle);

        SUCCEED() << "Completed CUDA physical address query";
    }

    /**
     * @brief Try CUDA P2P APIs even though we don't expect them to work cross-vendor
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, CUDA_P2P_Status)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: CUDA P2P Capabilities                                 ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        void *cuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!cuda_handle)
        {
            GTEST_SKIP() << "Cannot load CUDA driver library";
        }

        typedef int (*cuInit_fn)(unsigned int);
        typedef int (*cuDeviceGet_fn)(int *, int);
        typedef int (*cuDeviceGetCount_fn)(int *);
        typedef int (*cuDeviceCanAccessPeer_fn)(int *, int, int);
        typedef int (*cuDeviceGetP2PAttribute_fn)(int *, int, int, int);
        typedef int (*cuDeviceGetName_fn)(char *, int, int);

        auto cuInit = (cuInit_fn)dlsym(cuda_handle, "cuInit");
        auto cuDeviceGet = (cuDeviceGet_fn)dlsym(cuda_handle, "cuDeviceGet");
        auto cuDeviceGetCount = (cuDeviceGetCount_fn)dlsym(cuda_handle, "cuDeviceGetCount");
        auto cuDeviceCanAccessPeer = (cuDeviceCanAccessPeer_fn)dlsym(cuda_handle, "cuDeviceCanAccessPeer");
        auto cuDeviceGetP2PAttribute = (cuDeviceGetP2PAttribute_fn)dlsym(cuda_handle, "cuDeviceGetP2PAttribute");
        auto cuDeviceGetName = (cuDeviceGetName_fn)dlsym(cuda_handle, "cuDeviceGetName");

        if (cuInit(0) != 0)
        {
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuInit failed";
        }

        int device_count = 0;
        cuDeviceGetCount(&device_count);
        LOG_INFO("CUDA device count: " << device_count);

        for (int i = 0; i < device_count; i++)
        {
            int dev;
            cuDeviceGet(&dev, i);

            char name[256] = {0};
            if (cuDeviceGetName)
                cuDeviceGetName(name, sizeof(name), dev);
            LOG_INFO("\nDevice " << i << ": " << name);

            for (int j = 0; j < device_count; j++)
            {
                if (i == j)
                    continue;

                int peer_dev;
                cuDeviceGet(&peer_dev, j);

                if (cuDeviceCanAccessPeer)
                {
                    int can_access = 0;
                    cuDeviceCanAccessPeer(&can_access, dev, peer_dev);
                    LOG_INFO("  Can access device " << j << ": " << (can_access ? "YES" : "NO"));
                }

                if (cuDeviceGetP2PAttribute)
                {
                    // CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0
                    // CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 1
                    // CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 2
                    // CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 3
                    // CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4
                    int perf_rank = 0;
                    cuDeviceGetP2PAttribute(&perf_rank, 0, dev, peer_dev);
                    LOG_INFO("    P2P performance rank: " << perf_rank);
                }
            }
        }

        dlclose(cuda_handle);
        SUCCEED() << "Completed CUDA P2P status check";
    }

    /**
     * @brief Try using HSA's dmabuf import capability with CUDA's exported fd
     *
     * The hsa_amd_interop_map_buffer API is designed for importing dmabufs.
     * The CUDA fd is not a dmabuf but a POSIX file descriptor - maybe HSA can still use it.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, CUDA_To_HSA_FD_Import)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: Import CUDA Exported FD via HSA                       ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // First, get a CUDA fd
        void *cuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!cuda_handle)
        {
            GTEST_SKIP() << "Cannot load CUDA driver library";
        }

        // CUDA function pointers
        typedef int (*cuInit_fn)(unsigned int);
        typedef int (*cuDeviceGet_fn)(int *, int);
        typedef int (*cuCtxCreate_fn)(void **, unsigned int, int);
        typedef int (*cuCtxDestroy_fn)(void *);
        typedef int (*cuGetErrorString_fn)(int, const char **);
        typedef int (*cuMemCreate_fn)(uint64_t *, size_t, void *, unsigned long long);
        typedef int (*cuMemExportToShareableHandle_fn)(void *, uint64_t, int, unsigned long long);
        typedef int (*cuMemGetAllocationGranularity_fn)(size_t *, void *, int);
        typedef int (*cuMemRelease_fn)(uint64_t);

        auto cuInit = (cuInit_fn)dlsym(cuda_handle, "cuInit");
        auto cuDeviceGet = (cuDeviceGet_fn)dlsym(cuda_handle, "cuDeviceGet");
        auto cuCtxCreate = (cuCtxCreate_fn)dlsym(cuda_handle, "cuCtxCreate_v2");
        auto cuCtxDestroy = (cuCtxDestroy_fn)dlsym(cuda_handle, "cuCtxDestroy_v2");
        auto cuGetErrorString = (cuGetErrorString_fn)dlsym(cuda_handle, "cuGetErrorString");
        auto cuMemCreate = (cuMemCreate_fn)dlsym(cuda_handle, "cuMemCreate");
        auto cuMemExportToShareableHandle = (cuMemExportToShareableHandle_fn)dlsym(cuda_handle, "cuMemExportToShareableHandle");
        auto cuMemGetAllocationGranularity = (cuMemGetAllocationGranularity_fn)dlsym(cuda_handle, "cuMemGetAllocationGranularity");
        auto cuMemRelease = (cuMemRelease_fn)dlsym(cuda_handle, "cuMemRelease");

        if (!cuInit || !cuMemCreate || !cuMemExportToShareableHandle)
        {
            dlclose(cuda_handle);
            GTEST_SKIP() << "CUDA VMM APIs not available";
        }

        cuInit(0);
        int device = 0;
        cuDeviceGet(&device, 0);
        void *ctx = nullptr;
        cuCtxCreate(&ctx, 0, device);

        // Create exportable allocation
        struct
        {
            int type;                 // CU_MEM_ALLOCATION_TYPE_PINNED = 1
            int requestedHandleTypes; // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
            struct
            {
                int type; // CU_MEM_LOCATION_TYPE_DEVICE = 1
                int id;
            } location;
            void *win32HandleMetaData;
            uint64_t reserved[8];
        } alloc_prop = {0};

        alloc_prop.type = 1;
        alloc_prop.requestedHandleTypes = 1;
        alloc_prop.location.type = 1;
        alloc_prop.location.id = device;

        size_t granularity = 0;
        cuMemGetAllocationGranularity(&granularity, &alloc_prop, 1);

        size_t alloc_size = 64 * 1024 * 1024;
        alloc_size = ((alloc_size + granularity - 1) / granularity) * granularity;

        uint64_t alloc_handle = 0;
        int result = cuMemCreate(&alloc_handle, alloc_size, &alloc_prop, 0);
        if (result != 0)
        {
            cuCtxDestroy(ctx);
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuMemCreate failed";
        }

        int fd = -1;
        result = cuMemExportToShareableHandle(&fd, alloc_handle, 1, 0);
        if (result != 0 || fd < 0)
        {
            cuMemRelease(alloc_handle);
            cuCtxDestroy(ctx);
            dlclose(cuda_handle);
            GTEST_SKIP() << "cuMemExportToShareableHandle failed";
        }

        LOG_INFO("Got CUDA exported fd: " << fd);

        // Check what type of fd this is
        struct stat st;
        fstat(fd, &st);
        LOG_INFO("  fd mode: 0" << std::oct << st.st_mode << std::dec);
        LOG_INFO("  S_ISREG: " << S_ISREG(st.st_mode));
        LOG_INFO("  S_ISCHR: " << S_ISCHR(st.st_mode));
        LOG_INFO("  S_ISBLK: " << S_ISBLK(st.st_mode));
        LOG_INFO("  S_ISFIFO: " << S_ISFIFO(st.st_mode));
        LOG_INFO("  S_ISSOCK: " << S_ISSOCK(st.st_mode));

        // Try to read the fd type via /proc
        char fd_link[256];
        snprintf(fd_link, sizeof(fd_link), "/proc/self/fd/%d", fd);
        char fd_target[512] = {0};
        ssize_t link_len = readlink(fd_link, fd_target, sizeof(fd_target) - 1);
        if (link_len > 0)
        {
            LOG_INFO("  fd points to: " << fd_target);
        }

#ifdef HAVE_ROCM
        // Try to import via HSA
        LOG_INFO("\nTrying HSA import methods...");

        void *hsa_handle = dlopen("libhsa-runtime64.so.1", RTLD_NOW);
        if (hsa_handle)
        {
            // hsa_amd_interop_map_buffer expects dmabuf fds
            typedef int (*hsa_amd_interop_map_buffer_fn)(int, int, uint64_t *, size_t *, void *);
            auto hsa_amd_interop_map_buffer = (hsa_amd_interop_map_buffer_fn)dlsym(hsa_handle, "hsa_amd_interop_map_buffer");

            if (hsa_amd_interop_map_buffer)
            {
                LOG_INFO("Trying hsa_amd_interop_map_buffer with fd=" << fd);
                uint64_t mapped_ptr = 0;
                size_t mapped_size = 0;
                int hsa_result = hsa_amd_interop_map_buffer(1, fd, &mapped_ptr, &mapped_size, nullptr);
                if (hsa_result == 0)
                {
                    LOG_INFO("  ✓ hsa_amd_interop_map_buffer succeeded!");
                    LOG_INFO("    mapped_ptr: 0x" << std::hex << mapped_ptr << std::dec);
                    LOG_INFO("    mapped_size: " << mapped_size);
                }
                else
                {
                    LOG_INFO("  hsa_amd_interop_map_buffer failed: " << hsa_result);
                }
            }

            dlclose(hsa_handle);
        }

        // Try to dup the fd and register as external semaphore (different API path)
        void *hip_handle = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
        if (hip_handle)
        {
            // hipExternalMemoryHandleType values:
            // 1 = hipExternalMemoryHandleTypeOpaqueFd
            // 2 = hipExternalMemoryHandleTypeOpaqueWin32
            // 3 = hipExternalMemoryHandleTypeOpaqueWin32Kmt
            // 4 = hipExternalMemoryHandleTypeD3D12Heap
            // 5 = hipExternalMemoryHandleTypeD3D12Resource
            // 6 = hipExternalMemoryHandleTypeD3D11Resource
            // 7 = hipExternalMemoryHandleTypeD3D11ResourceKmt
            // 16 = hipExternalMemoryHandleTypeDmaBuf (!)

            typedef int (*hipImportExternalMemory_fn)(void **, void *);
            auto hipImportExternalMemory = (hipImportExternalMemory_fn)dlsym(hip_handle, "hipImportExternalMemory");

            if (hipImportExternalMemory)
            {
                LOG_INFO("\nTrying hipImportExternalMemory with different handle types...");

                struct
                {
                    int type;
                    union
                    {
                        int fd;
                        struct
                        {
                            void *handle;
                            void *name;
                        } win32;
                    } handle;
                    uint64_t size;
                    unsigned int flags;
                    unsigned int reserved[16];
                } ext_mem_desc = {0};

                int types_to_try[] = {1, 16}; // OpaqueFd and DmaBuf
                const char *type_names[] = {"OpaqueFd", "DmaBuf"};

                for (int i = 0; i < 2; i++)
                {
                    memset(&ext_mem_desc, 0, sizeof(ext_mem_desc));
                    ext_mem_desc.type = types_to_try[i];
                    ext_mem_desc.handle.fd = fd;
                    ext_mem_desc.size = alloc_size;
                    ext_mem_desc.flags = 0;

                    void *ext_mem = nullptr;
                    int hip_result = hipImportExternalMemory(&ext_mem, &ext_mem_desc);
                    LOG_INFO("  hipImportExternalMemory(type=" << type_names[i] << "): "
                                                               << (hip_result == 0 ? "SUCCESS" : "failed with " + std::to_string(hip_result)));

                    if (hip_result == 0)
                    {
                        LOG_INFO("    ✓✓✓ GOT EXTERNAL MEMORY HANDLE!");
                    }
                }
            }

            dlclose(hip_handle);
        }
#endif

        close(fd);
        cuMemRelease(alloc_handle);
        cuCtxDestroy(ctx);
        dlclose(cuda_handle);

        SUCCEED() << "Completed CUDA to HSA fd import test";
    }

    /**
     * @brief Try using nvidia-uvm character device for P2P
     *
     * nvidia-uvm provides a unified memory interface that might expose
     * more control over memory mappings.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryNvidiaUVM)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: NVIDIA UVM Device Interface                           ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Check if nvidia-uvm is available
        int uvm_fd = open("/dev/nvidia-uvm", O_RDWR);
        if (uvm_fd < 0)
        {
            LOG_INFO("Cannot open /dev/nvidia-uvm: " << strerror(errno));
            GTEST_SKIP() << "nvidia-uvm not available";
        }

        LOG_INFO("Opened /dev/nvidia-uvm, fd=" << uvm_fd);

        // Try ioctl to query capabilities
        // UVM ioctl numbers are defined in nvidia-uvm/uvm_ioctl.h (proprietary)
        // We'll try some common ioctl probing

        struct
        {
            char data[4096];
        } ioctl_data = {0};

        // UVM_INITIALIZE = 0x30000001 (typical)
        int result = ioctl(uvm_fd, 0x30000001, &ioctl_data);
        LOG_INFO("ioctl(UVM_INITIALIZE): " << result << " errno=" << errno);

        // Try mapping some UVM memory
        void *uvm_map = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, uvm_fd, 0);
        if (uvm_map != MAP_FAILED)
        {
            LOG_INFO("mmap of nvidia-uvm succeeded at " << uvm_map);
            munmap(uvm_map, 4096);
        }
        else
        {
            LOG_INFO("mmap of nvidia-uvm failed: " << strerror(errno));
        }

        close(uvm_fd);

        SUCCEED() << "Completed nvidia-uvm exploration";
    }

    /**
     * @brief Try to use cuda-gdr (GPUDirect RDMA) kernel module if available
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, TryGDRCopy)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  EXPERIMENT: GPUDirect RDMA (gdrdrv) Interface                     ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════╝");

        // Check if gdrdrv is loaded
        std::ifstream modules("/proc/modules");
        std::string line;
        bool has_gdrdrv = false;
        while (std::getline(modules, line))
        {
            if (line.find("gdrdrv") != std::string::npos ||
                line.find("nvidia_peermem") != std::string::npos)
            {
                LOG_INFO("Found module: " << line);
                has_gdrdrv = true;
            }
        }

        if (!has_gdrdrv)
        {
            LOG_INFO("gdrdrv/nvidia_peermem not loaded - GPUDirect RDMA not available");
        }

        // Check for gdrcopy device
        int gdr_fd = open("/dev/gdrdrv", O_RDWR);
        if (gdr_fd >= 0)
        {
            LOG_INFO("Opened /dev/gdrdrv, fd=" << gdr_fd);

            // gdrcopy provides gdr_open, gdr_map, etc. via library
            void *gdr_lib = dlopen("libgdrapi.so", RTLD_NOW);
            if (gdr_lib)
            {
                LOG_INFO("Found libgdrapi.so - GPUDirect RDMA library available");
                // gdr_open(), gdr_map(), gdr_copy_to_mapping() etc.
                dlclose(gdr_lib);
            }
            else
            {
                LOG_INFO("libgdrapi.so not found");
            }

            close(gdr_fd);
        }
        else
        {
            LOG_INFO("/dev/gdrdrv not available: " << strerror(errno));
        }

        // Check nvidia_peermem for IB verbs integration
        int peermem_fd = open("/dev/nvidia-peermem", O_RDWR);
        if (peermem_fd >= 0)
        {
            LOG_INFO("Opened /dev/nvidia-peermem");
            close(peermem_fd);
        }

        SUCCEED() << "Completed GPUDirect RDMA exploration";
    }

    /**
     * @brief Test hipHostRegisterIoMemory (ROCm 7.1.1+) for NVIDIA BAR P2P
     *
     * ROCm 7.1.1 added hipHostRegisterIoMemory flag which is AMD's equivalent of
     * CUDA's CU_MEMHOSTREGISTER_IOMEMORY. This flag registers I/O memory (like
     * PCIe BARs) with the HIP runtime so AMD GPUs can access it.
     *
     * This test attempts the REVERSE of what DirectP2P.cpp does:
     * - DirectP2P.cpp: mmap AMD BAR → cuMemHostRegister(IOMEMORY) → CUDA accesses AMD
     * - This test: mmap NVIDIA BAR → hipHostRegister(IoMemory) → HIP accesses NVIDIA
     *
     * If this works, we can achieve TRUE BIDIRECTIONAL P2P:
     * - NVIDIA initiates DMA via CUDA accessing AMD's BAR (already working)
     * - AMD initiates DMA via HIP accessing NVIDIA's BAR (testing here)
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, DISABLED_HipHostRegisterIoMemory_NvidiaBAR)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  TEST: hipHostRegisterIoMemory for NVIDIA BAR (ROCm 7.1.1+)                         ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  This tests AMD's IOMEMORY equivalent for cross-vendor P2P                          ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════════════════════╝\n");

        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        // Find NVIDIA GPU with largest VRAM BAR
        std::string nvidia_gpu;
        PCIeBarInfo nvidia_info;
        for (const auto &dev : nvidia_gpus)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            if (info.vram_size > nvidia_info.vram_size)
            {
                nvidia_gpu = dev;
                nvidia_info = info;
            }
        }

        if (nvidia_info.vram_size == 0)
        {
            GTEST_SKIP() << "No NVIDIA GPU with valid VRAM BAR found";
        }

        LOG_INFO("NVIDIA GPU: " << nvidia_gpu);
        LOG_INFO("  VRAM BAR" << nvidia_info.vram_bar_index << " physical: 0x"
                              << std::hex << nvidia_info.vram_phys_addr << std::dec);
        LOG_INFO("  VRAM size: " << (nvidia_info.vram_size / (1024 * 1024 * 1024)) << " GB"
                                 << (nvidia_info.resizable_bar ? " (Resizable BAR enabled)" : " (256MB BAR)"));

        // Step 1: mmap NVIDIA's VRAM BAR
        size_t map_size = 64 * 1024 * 1024; // 64MB test region
        if (!nvidia_info.resizable_bar)
        {
            map_size = std::min(map_size, (size_t)(64 * 1024 * 1024)); // Limit for small BAR
        }

        LOG_INFO("\n=== Step 1: mmap NVIDIA VRAM BAR via sysfs ===");
        void *nvidia_bar = PCIResourceParser::mmapBar(nvidia_gpu, nvidia_info.vram_bar_index, map_size);

        if (nvidia_bar == nullptr)
        {
            LOG_ERROR("Cannot mmap NVIDIA VRAM BAR - need root access (sudo)");
            GTEST_SKIP() << "Cannot mmap NVIDIA VRAM BAR - need root access";
        }

        LOG_INFO("✓ mmap'd " << (map_size / (1024 * 1024)) << " MB of NVIDIA VRAM at " << nvidia_bar);

        // Verify we can read from the BAR via CPU
        volatile uint32_t cpu_read = *reinterpret_cast<volatile uint32_t *>(nvidia_bar);
        LOG_INFO("  CPU read verification: 0x" << std::hex << cpu_read << std::dec);

        // Step 2: Try hipHostRegisterIoMemory (ROCm 7.1.1+ feature)
        LOG_INFO("\n=== Step 2: hipHostRegister with hipHostRegisterIoMemory flag ===");
        LOG_INFO("This is the AMD equivalent of CUDA's CU_MEMHOSTREGISTER_IOMEMORY");

        void *hip_device_ptr = nullptr;
        bool io_register_success = rocm_backend.registerIoMemory(nvidia_bar, map_size, &hip_device_ptr);

        if (io_register_success && hip_device_ptr != nullptr)
        {
            LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║  ✓✓✓ hipHostRegisterIoMemory SUCCEEDED!                          ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
            LOG_INFO("  Host BAR ptr:   " << nvidia_bar);
            LOG_INFO("  HIP device ptr: " << hip_device_ptr);

            // Step 3: Test data transfer - AMD GPU reading from NVIDIA VRAM!
            LOG_INFO("\n=== Step 3: Test AMD→NVIDIA read via HIP ===");

            rocm_backend.setDevice(0);
            const size_t test_size = 4096;
            void *amd_buffer = rocm_backend.allocate(test_size, 0);
            ASSERT_NE(amd_buffer, nullptr) << "Failed to allocate AMD GPU buffer";

            // Try to read from NVIDIA BAR into AMD buffer
            LOG_INFO("Attempting hipMemcpy from NVIDIA BAR to AMD buffer...");
            bool read_ok = rocm_backend.deviceToDevice(amd_buffer, hip_device_ptr, test_size, 0);
            rocm_backend.synchronize(0);

            if (read_ok)
            {
                LOG_INFO("  ✓ hipMemcpy D2D (NVIDIA→AMD) succeeded!");

                // Verify by reading back to CPU
                std::vector<uint32_t> readback(test_size / sizeof(uint32_t));
                rocm_backend.deviceToHost(readback.data(), amd_buffer, test_size, 0);

                LOG_INFO("  First 4 words via HIP: 0x" << std::hex
                                                       << readback[0] << " " << readback[1] << " "
                                                       << readback[2] << " " << readback[3] << std::dec);
            }
            else
            {
                LOG_WARN("  ✗ hipMemcpy D2D (NVIDIA→AMD) failed");
            }

            // Step 4: Test data transfer - AMD GPU writing to NVIDIA VRAM!
            LOG_INFO("\n=== Step 4: Test AMD→NVIDIA write via HIP ===");

            // Prepare test pattern on AMD GPU
            std::vector<float> test_pattern(test_size / sizeof(float));
            for (size_t i = 0; i < test_pattern.size(); ++i)
            {
                test_pattern[i] = 42.0f + static_cast<float>(i);
            }
            rocm_backend.hostToDevice(amd_buffer, test_pattern.data(), test_size, 0);
            rocm_backend.synchronize(0);

            // Try to write from AMD buffer to NVIDIA BAR
            LOG_INFO("Attempting hipMemcpy from AMD buffer to NVIDIA BAR...");
            bool write_ok = rocm_backend.deviceToDevice(hip_device_ptr, amd_buffer, test_size, 0);
            rocm_backend.synchronize(0);

            if (write_ok)
            {
                LOG_INFO("  ✓ hipMemcpy D2D (AMD→NVIDIA) succeeded!");

                // Verify by reading from the BAR via CPU
                float *bar_floats = reinterpret_cast<float *>(nvidia_bar);
                bool data_matches = true;
                for (size_t i = 0; i < std::min((size_t)8, test_pattern.size()); ++i)
                {
                    LOG_INFO("    BAR[" << i << "] = " << bar_floats[i]
                                        << " (expected " << test_pattern[i] << ")");
                    if (std::abs(bar_floats[i] - test_pattern[i]) > 0.001f)
                    {
                        data_matches = false;
                    }
                }

                if (data_matches)
                {
                    LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
                    LOG_INFO("║  ✓✓✓ BIDIRECTIONAL P2P CONFIRMED!                                ║");
                    LOG_INFO("║                                                                   ║");
                    LOG_INFO("║  AMD GPU can read from AND write to NVIDIA VRAM via PCIe BAR!    ║");
                    LOG_INFO("║  This enables TRUE bidirectional cross-vendor GPU P2P!           ║");
                    LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
                }
                else
                {
                    LOG_WARN("  Data mismatch - write may have gone to wrong location");
                }
            }
            else
            {
                LOG_WARN("  ✗ hipMemcpy D2D (AMD→NVIDIA) failed");
            }

            // Cleanup
            rocm_backend.free(amd_buffer, 0);
            rocm_backend.unregisterIoMemory(nvidia_bar);

            // Final result
            if (read_ok || write_ok)
            {
                SUCCEED() << "hipHostRegisterIoMemory enables AMD access to NVIDIA VRAM!";
            }
            else
            {
                LOG_WARN("Registration succeeded but DMA transfers failed");
                SUCCEED() << "hipHostRegisterIoMemory worked but transfers need investigation";
            }
        }
        else
        {
            LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║  hipHostRegisterIoMemory FAILED                                   ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
            LOG_INFO("This could mean:");
            LOG_INFO("  1. ROCm version < 7.1.1 (hipHostRegisterIoMemory not available)");
            LOG_INFO("  2. NVIDIA driver blocks BAR access for external registration");
            LOG_INFO("  3. IOMMU blocking the mapping");
            LOG_INFO("  4. Need CAP_SYS_ADMIN for IOMEMORY registration");

            // The existing registerIoMemory already tries fallback methods
            SUCCEED() << "hipHostRegisterIoMemory not available for NVIDIA BAR";
        }

        // Cleanup
        munmap(nvidia_bar, map_size);
    }

    /**
     * @brief Validate hipHostRegisterIoMemory works for AMD's own BAR
     *
     * This baseline test confirms that ROCm 7.1.1's hipHostRegisterIoMemory
     * flag actually works by registering AMD's own VRAM BAR.
     *
     * FINDING: hipHostRegisterIoMemory does NOT work for mmap'd PCIe BARs
     * (even AMD's own) because:
     * 1. The flag is designed for DEVICE-ALLOCATED I/O memory (e.g., FPGA)
     * 2. mmap'd sysfs resources are VM_IO regions that can't be DMA mapped
     * 3. tinygrad bypasses this entirely by writing to GPU page tables directly
     *
     * The correct approach for BAR access is either:
     * - KFD ioctl for memory allocation (which gives you a GART mapping)
     * - Direct page table manipulation (tinygrad's approach)
     * - Use nvidia-peermem/amd-gpudirect for kernel-level DMA registration
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, DISABLED_HipHostRegisterIoMemory_AMD_Baseline)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  BASELINE: hipHostRegisterIoMemory for AMD's own BAR                                ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  Validates that the flag works at all before testing cross-vendor                    ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════════════════════╝\n");

        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        auto amd_gpus = PCIResourceParser::scanPCIBus(0x1002);
        if (amd_gpus.empty())
        {
            GTEST_SKIP() << "No AMD GPUs found";
        }

        // Find AMD GPU with largest VRAM BAR
        std::string amd_gpu;
        PCIeBarInfo amd_info;
        for (const auto &dev : amd_gpus)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            if (info.vram_size > amd_info.vram_size)
            {
                amd_gpu = dev;
                amd_info = info;
            }
        }

        if (amd_info.vram_size == 0)
        {
            GTEST_SKIP() << "No AMD GPU with valid VRAM BAR found";
        }

        LOG_INFO("AMD GPU: " << amd_gpu);
        LOG_INFO("  VRAM BAR" << amd_info.vram_bar_index << " physical: 0x"
                              << std::hex << amd_info.vram_phys_addr << std::dec);
        LOG_INFO("  VRAM size: " << (amd_info.vram_size / (1024 * 1024 * 1024)) << " GB"
                                 << (amd_info.resizable_bar ? " (Resizable BAR enabled)" : " (256MB BAR)"));

        // mmap AMD's own VRAM BAR
        size_t map_size = 4 * 1024 * 1024; // 4MB test region
        void *amd_bar = PCIResourceParser::mmapBar(amd_gpu, amd_info.vram_bar_index, map_size);

        if (amd_bar == nullptr)
        {
            GTEST_SKIP() << "Cannot mmap AMD VRAM BAR - need root access";
        }

        LOG_INFO("mmap'd " << (map_size / (1024 * 1024)) << " MB of AMD VRAM at " << amd_bar);

        // Try hipHostRegisterIoMemory on AMD's own BAR
        LOG_INFO("\n=== Testing hipHostRegisterIoMemory on AMD's own BAR ===");
        LOG_INFO("NOTE: This is EXPECTED TO FAIL because:");
        LOG_INFO("  - hipHostRegisterIoMemory is for DEVICE-ALLOCATED I/O memory");
        LOG_INFO("  - mmap'd sysfs BARs are VM_IO regions that can't be DMA mapped");
        LOG_INFO("  - The HIP runtime needs physical page backing for DMA");

        void *hip_device_ptr = nullptr;
        bool success = rocm_backend.registerIoMemory(amd_bar, map_size, &hip_device_ptr);

        if (success && hip_device_ptr != nullptr)
        {
            LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║  ✓ hipHostRegisterIoMemory works for AMD BAR!                     ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
            LOG_INFO("  This is UNEXPECTED - the flag might work differently than documented");

            rocm_backend.unregisterIoMemory(amd_bar);
            SUCCEED() << "hipHostRegisterIoMemory baseline works";
        }
        else
        {
            LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║  hipHostRegisterIoMemory failed (EXPECTED for mmap'd BARs)        ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
            LOG_INFO("");
            LOG_INFO("This confirms that hipHostRegisterIoMemory is NOT suitable for:");
            LOG_INFO("  - Accessing PCIe BARs via mmap (sysfs resource files)");
            LOG_INFO("  - Cross-vendor P2P via BAR mapping");
            LOG_INFO("");
            LOG_INFO("Alternative approaches for cross-vendor P2P:");
            LOG_INFO("  1. KFD ioctl: DRM_AMDGPU_GEM_DGMA (direct GMA to external device)");
            LOG_INFO("  2. Page table manipulation: tinygrad's AMMemoryManager approach");
            LOG_INFO("  3. Kernel driver: nvidia-peermem + amd-gpudirect (RDMA style)");
            LOG_INFO("  4. DMA-BUF: hsa_amd_vmem_export_shareable_handle (same-vendor only)");

            // This is actually the expected behavior - not a skip
            SUCCEED() << "hipHostRegisterIoMemory correctly rejects mmap'd BAR memory";
        }

        munmap(amd_bar, map_size);
    }

    /**
     * @brief Benchmark hipHostRegisterIoMemory P2P if available
     *
     * If the IOMEMORY registration works, benchmark the actual throughput
     * to compare with our existing CUDA-side P2P implementation.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, DISABLED_Benchmark_HipIoMemory_P2P)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  BENCHMARK: hipHostRegisterIoMemory P2P Throughput                                  ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════════════════════╝\n");

        ROCmBackend rocm_backend;
        if (rocm_backend.deviceCount() == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        auto nvidia_gpus = PCIResourceParser::scanPCIBus(0x10de);
        if (nvidia_gpus.empty())
        {
            GTEST_SKIP() << "No NVIDIA GPUs found";
        }

        // Find NVIDIA GPU with largest BAR
        std::string nvidia_gpu;
        PCIeBarInfo nvidia_info;
        for (const auto &dev : nvidia_gpus)
        {
            auto info = PCIResourceParser::parseResourceFile(dev);
            if (info.vram_size > nvidia_info.vram_size)
            {
                nvidia_gpu = dev;
                nvidia_info = info;
            }
        }

        // mmap NVIDIA BAR
        size_t map_size = 64 * 1024 * 1024;
        void *nvidia_bar = PCIResourceParser::mmapBar(nvidia_gpu, nvidia_info.vram_bar_index, map_size);
        if (!nvidia_bar)
        {
            GTEST_SKIP() << "Cannot mmap NVIDIA BAR - need root";
        }

        // Register with HIP
        void *hip_device_ptr = nullptr;
        bool success = rocm_backend.registerIoMemory(nvidia_bar, map_size, &hip_device_ptr);

        if (!success || !hip_device_ptr)
        {
            munmap(nvidia_bar, map_size);
            GTEST_SKIP() << "hipHostRegisterIoMemory not available";
        }

        LOG_INFO("hipHostRegisterIoMemory succeeded - running benchmarks...\n");

        rocm_backend.setDevice(0);
        const size_t bench_size = 16 * 1024 * 1024; // 16MB transfers
        const int iterations = 10;

        void *amd_buffer = rocm_backend.allocate(bench_size, 0);
        if (!amd_buffer)
        {
            rocm_backend.unregisterIoMemory(nvidia_bar);
            munmap(nvidia_bar, map_size);
            GTEST_SKIP() << "Cannot allocate AMD buffer";
        }

        // Initialize AMD buffer
        std::vector<float> data(bench_size / sizeof(float), 1.0f);
        rocm_backend.hostToDevice(amd_buffer, data.data(), bench_size, 0);
        rocm_backend.synchronize(0);

        // Warmup
        rocm_backend.deviceToDevice(hip_device_ptr, amd_buffer, bench_size, 0);
        rocm_backend.deviceToDevice(amd_buffer, hip_device_ptr, bench_size, 0);
        rocm_backend.synchronize(0);

        // Benchmark WRITE (AMD→NVIDIA)
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            rocm_backend.deviceToDevice(hip_device_ptr, amd_buffer, bench_size, 0);
        }
        rocm_backend.synchronize(0);
        auto end = std::chrono::high_resolution_clock::now();

        double write_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double write_gbps = (bench_size * iterations / (1024.0 * 1024.0 * 1024.0)) / (write_time_ms / 1000.0);

        // Benchmark READ (NVIDIA→AMD)
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            rocm_backend.deviceToDevice(amd_buffer, hip_device_ptr, bench_size, 0);
        }
        rocm_backend.synchronize(0);
        end = std::chrono::high_resolution_clock::now();

        double read_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double read_gbps = (bench_size * iterations / (1024.0 * 1024.0 * 1024.0)) / (read_time_ms / 1000.0);

        LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  HIP-initiated P2P Benchmark Results (" << (bench_size / (1024 * 1024)) << " MB x " << iterations << " iterations)");
        LOG_INFO("╠══════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  AMD→NVIDIA (write to BAR):  " << std::fixed << std::setprecision(2) << write_gbps << " GB/s");
        LOG_INFO("║  NVIDIA→AMD (read from BAR): " << std::fixed << std::setprecision(2) << read_gbps << " GB/s");
        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
        LOG_INFO("");
        LOG_INFO("Compare with CUDA-initiated P2P (DirectP2P.cpp):");
        LOG_INFO("  Expected: ~2.65 GB/s write, ~2.67 GB/s read with rBAR");

        // Cleanup
        rocm_backend.free(amd_buffer, 0);
        rocm_backend.unregisterIoMemory(nvidia_bar);
        munmap(nvidia_bar, map_size);

        SUCCEED() << "Benchmark complete";
    }

    /**
     * @brief Test DIRECT AMD-to-AMD P2P via PCIe BAR mapping
     *
     * This test explores true direct P2P between AMD GPUs by mapping one GPU's
     * VRAM BAR and having another GPU write to it directly - the same approach
     * we use for CUDA↔ROCm in DirectP2P.cpp.
     *
     * Approach:
     * 1. mmap AMD GPU 1's VRAM BAR via sysfs
     * 2. Use HSA to register that memory for AMD GPU 0 to access
     * 3. Have GPU 0 write directly to GPU 1's BAR (true P2P over PCIe)
     *
     * This bypasses the broken hipDeviceCanAccessPeer() and avoids host staging.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, AMD_to_AMD_DirectP2P_via_BAR)
    {
        ROCmBackend rocm_backend;
        int device_count = rocm_backend.deviceCount();
        
        if (device_count < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ROCm devices (found " << device_count << ")";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  AMD-to-AMD DIRECT P2P via PCIe BAR Mapping                                         ║");
        LOG_INFO("╠════════════════════════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  Goal: True direct P2P without host staging                                         ║");
        LOG_INFO("║  Method: mmap GPU1's BAR, have GPU0 write to it directly                            ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════════════════════╝\n");

        // Find AMD GPUs via sysfs
        auto amd_gpus = PCIResourceParser::scanPCIBus(0x1002);
        if (amd_gpus.size() < 2)
        {
            GTEST_SKIP() << "Need at least 2 AMD GPUs in sysfs (found " << amd_gpus.size() << ")";
        }

        // Parse BAR info for first two AMD GPUs
        PCIeBarInfo gpu0_info = PCIResourceParser::parseResourceFile(amd_gpus[0]);
        PCIeBarInfo gpu1_info = PCIResourceParser::parseResourceFile(amd_gpus[1]);

        LOG_INFO("=== GPU BAR Information ===");
        LOG_INFO("GPU 0: " << amd_gpus[0]);
        LOG_INFO("  VRAM BAR" << gpu0_info.vram_bar_index << ": 0x" << std::hex << gpu0_info.vram_phys_addr
                              << " size=" << std::dec << (gpu0_info.vram_size / (1024ULL * 1024 * 1024)) << " GB"
                              << (gpu0_info.resizable_bar ? " (rBAR)" : ""));
        LOG_INFO("GPU 1: " << amd_gpus[1]);
        LOG_INFO("  VRAM BAR" << gpu1_info.vram_bar_index << ": 0x" << std::hex << gpu1_info.vram_phys_addr
                              << " size=" << std::dec << (gpu1_info.vram_size / (1024ULL * 1024 * 1024)) << " GB"
                              << (gpu1_info.resizable_bar ? " (rBAR)" : ""));

        // Step 1: mmap GPU 1's VRAM BAR
        LOG_INFO("\n=== Step 1: mmap GPU 1's VRAM BAR ===");
        const size_t map_size = 64 * 1024 * 1024;  // 64 MB test region
        void* gpu1_bar = PCIResourceParser::mmapBar(amd_gpus[1], gpu1_info.vram_bar_index, map_size);
        
        if (gpu1_bar == nullptr)
        {
            LOG_WARN("Cannot mmap GPU 1's BAR - need root access");
            LOG_INFO("Run with: sudo ./v2_integration_amd_pcie_p2p_mapping");
            GTEST_SKIP() << "Cannot mmap AMD BAR - need root";
        }
        LOG_INFO("  ✓ mmap'd " << (map_size / (1024 * 1024)) << " MB of GPU 1's VRAM at " << gpu1_bar);

        // Step 2: Try to register the BAR memory with HSA for GPU 0 to access
        LOG_INFO("\n=== Step 2: Register BAR with HSA for GPU 0 access ===");
        
        // Load HSA dynamically to avoid header conflicts
        void* hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hsa_handle)
        {
            hsa_handle = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_GLOBAL);
        }
        
        if (!hsa_handle)
        {
            LOG_WARN("Cannot load HSA runtime: " << dlerror());
            munmap(gpu1_bar, map_size);
            GTEST_SKIP() << "HSA runtime not available";
        }

        // Define HSA types and function pointers
        typedef uint32_t hsa_status_t;
        typedef struct { uint64_t handle; } hsa_agent_t;
        
        typedef hsa_status_t (*hsa_init_fn)();
        typedef hsa_status_t (*hsa_shut_down_fn)();
        typedef hsa_status_t (*hsa_iterate_agents_fn)(
            hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data);
        typedef hsa_status_t (*hsa_agent_get_info_fn)(hsa_agent_t agent, int attribute, void* value);
        typedef hsa_status_t (*hsa_amd_memory_lock_fn)(
            void* host_ptr, size_t size, hsa_agent_t* agents, int num_agents, void** agent_ptr);
        typedef hsa_status_t (*hsa_amd_memory_unlock_fn)(void* host_ptr);

        auto hsa_init = (hsa_init_fn)dlsym(hsa_handle, "hsa_init");
        auto hsa_shut_down = (hsa_shut_down_fn)dlsym(hsa_handle, "hsa_shut_down");
        auto hsa_iterate_agents = (hsa_iterate_agents_fn)dlsym(hsa_handle, "hsa_iterate_agents");
        auto hsa_agent_get_info = (hsa_agent_get_info_fn)dlsym(hsa_handle, "hsa_agent_get_info");
        auto hsa_amd_memory_lock = (hsa_amd_memory_lock_fn)dlsym(hsa_handle, "hsa_amd_memory_lock");
        auto hsa_amd_memory_unlock = (hsa_amd_memory_unlock_fn)dlsym(hsa_handle, "hsa_amd_memory_unlock");

        if (!hsa_init || !hsa_amd_memory_lock)
        {
            LOG_WARN("Cannot find HSA functions");
            dlclose(hsa_handle);
            munmap(gpu1_bar, map_size);
            GTEST_SKIP() << "HSA functions not available";
        }

        // Initialize HSA
        hsa_status_t status = hsa_init();
        if (status != 0)
        {
            LOG_WARN("hsa_init failed: " << status);
            dlclose(hsa_handle);
            munmap(gpu1_bar, map_size);
            GTEST_SKIP() << "HSA init failed";
        }
        LOG_INFO("  ✓ HSA initialized");

        // Collect GPU agents
        std::vector<hsa_agent_t> gpu_agents;
        auto agent_callback = [](hsa_agent_t agent, void* data) -> hsa_status_t {
            auto* agents = static_cast<std::vector<hsa_agent_t>*>(data);
            agents->push_back(agent);
            return 0; // HSA_STATUS_SUCCESS
        };
        
        hsa_iterate_agents(agent_callback, &gpu_agents);
        LOG_INFO("  Found " << gpu_agents.size() << " HSA agents");

        // Filter to GPU agents only (device_type == 1 is GPU)
        std::vector<hsa_agent_t> gpus_only;
        for (const auto& agent : gpu_agents)
        {
            uint32_t device_type = 0;
            hsa_agent_get_info(agent, 17, &device_type); // 17 = HSA_AGENT_INFO_DEVICE
            if (device_type == 1) // HSA_DEVICE_TYPE_GPU
            {
                gpus_only.push_back(agent);
            }
        }
        LOG_INFO("  Found " << gpus_only.size() << " GPU agents");

        if (gpus_only.size() < 2)
        {
            hsa_shut_down();
            dlclose(hsa_handle);
            munmap(gpu1_bar, map_size);
            GTEST_SKIP() << "Need at least 2 GPU agents";
        }

        // Try hsa_amd_memory_lock on the mmap'd BAR
        // This should pin the memory and return a GPU-accessible pointer
        LOG_INFO("\n=== Step 3: hsa_amd_memory_lock on mmap'd BAR ===");
        LOG_INFO("  Attempting to lock " << (map_size / (1024 * 1024)) << " MB for GPU 0 access...");
        
        void* gpu0_accessible_ptr = nullptr;
        hsa_agent_t target_gpu = gpus_only[0];
        
        status = hsa_amd_memory_lock(gpu1_bar, map_size, &target_gpu, 1, &gpu0_accessible_ptr);
        
        if (status == 0 && gpu0_accessible_ptr != nullptr)
        {
            LOG_INFO("  ✓ hsa_amd_memory_lock SUCCEEDED!");
            LOG_INFO("    GPU 0 accessible pointer: " << gpu0_accessible_ptr);
            LOG_INFO("    Original BAR pointer: " << gpu1_bar);
            
            // Step 4: Try to use this pointer for GPU 0 kernel/memcpy
            LOG_INFO("\n=== Step 4: Test GPU 0 write to GPU 1's BAR ===");
            
            // Allocate source buffer on GPU 0
            rocm_backend.setDevice(0);
            const size_t test_size = 4 * 1024 * 1024;  // 4 MB
            void* gpu0_src = rocm_backend.allocate(test_size, 0);
            
            if (gpu0_src)
            {
                // Initialize with pattern
                std::vector<float> pattern(test_size / sizeof(float));
                for (size_t i = 0; i < pattern.size(); ++i)
                {
                    pattern[i] = static_cast<float>(i % 1000) * 0.001f;
                }
                rocm_backend.hostToDevice(gpu0_src, pattern.data(), test_size, 0);
                rocm_backend.synchronize(0);
                
                LOG_INFO("  Attempting hipMemcpy from GPU 0 to locked BAR pointer...");
                
                // Try device-to-device copy from GPU 0 to the locked BAR pointer
                auto start = std::chrono::high_resolution_clock::now();
                bool copy_success = rocm_backend.deviceToDevice(gpu0_accessible_ptr, gpu0_src, test_size, 0);
                rocm_backend.synchronize(0);
                auto end = std::chrono::high_resolution_clock::now();
                
                if (copy_success)
                {
                    double ms = std::chrono::duration<double, std::milli>(end - start).count();
                    double gbps = (test_size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
                    
                    LOG_INFO("  ✓ Copy succeeded!");
                    LOG_INFO("    Time: " << std::fixed << std::setprecision(2) << ms << " ms");
                    LOG_INFO("    Throughput: " << std::fixed << std::setprecision(2) << gbps << " GB/s");
                    
                    // Verify by reading back from BAR
                    LOG_INFO("\n=== Step 5: Verify data in GPU 1's BAR ===");
                    std::vector<float> verify(test_size / sizeof(float));
                    memcpy(verify.data(), gpu1_bar, test_size);  // Direct read from mmap'd BAR
                    
                    int mismatches = 0;
                    for (size_t i = 0; i < verify.size() && mismatches < 10; ++i)
                    {
                        if (std::abs(verify[i] - pattern[i]) > 1e-6f)
                        {
                            mismatches++;
                        }
                    }
                    
                    if (mismatches == 0)
                    {
                        LOG_INFO("  ✓✓✓ DATA VERIFIED! True direct AMD-to-AMD P2P works!");
                        LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
                        LOG_INFO("║  SUCCESS: Direct AMD-to-AMD P2P via BAR mapping!                  ║");
                        LOG_INFO("║  Throughput: " << std::fixed << std::setprecision(2) << gbps << " GB/s                                        ║");
                        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
                    }
                    else
                    {
                        LOG_WARN("  Data verification failed: " << mismatches << " mismatches");
                    }
                }
                else
                {
                    LOG_WARN("  Copy failed");
                }
                
                rocm_backend.free(gpu0_src, 0);
            }
            
            // Unlock
            hsa_amd_memory_unlock(gpu1_bar);
        }
        else
        {
            LOG_WARN("  ✗ hsa_amd_memory_lock FAILED (status=" << status << ")");
            LOG_INFO("");
            LOG_INFO("This is EXPECTED for mmap'd PCIe BARs because:");
            LOG_INFO("  - hsa_amd_memory_lock is for pinning HOST memory, not I/O memory");
            LOG_INFO("  - mmap'd sysfs BARs are VM_IO regions without backing pages");
            LOG_INFO("");
            LOG_INFO("ALTERNATIVE APPROACHES for direct AMD-to-AMD P2P:");
            LOG_INFO("  1. tinygrad-style: Program AMD GPU page tables directly via MMIO");
            LOG_INFO("  2. KFD ioctl: Use DRM_AMDGPU_GEM_DGMA for direct GMA mapping");
            LOG_INFO("  3. Kernel module: Custom driver to set up inter-GPU DMA");
        }

        // Cleanup
        hsa_shut_down();
        dlclose(hsa_handle);
        munmap(gpu1_bar, map_size);
        
        SUCCEED() << "Direct P2P exploration complete";
    }

    /**
     * @brief Test AMD-to-AMD DIRECT P2P via KFD MAP_MEMORY_TO_GPU
     *
     * This is the CORRECT approach for direct AMD-to-AMD P2P:
     * 1. Allocate VRAM on GPU 1 via ALLOC_MEMORY_OF_GPU
     * 2. Map that memory to GPU 0 via MAP_MEMORY_TO_GPU
     * 3. GPU 0 can now access GPU 1's VRAM directly (true P2P)
     *
     * This is what RCCL/ROCm should be using internally, but the standard
     * hipDeviceCanAccessPeer() check may be returning 0 incorrectly on our
     * hardware configuration.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, AMD_to_AMD_DirectP2P_via_KFD_MapMemory)
    {
        ROCmBackend rocm_backend;
        int device_count = rocm_backend.deviceCount();
        
        if (device_count < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ROCm devices (found " << device_count << ")";
        }

        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  AMD-to-AMD DIRECT P2P via KFD MAP_MEMORY_TO_GPU                                    ║");
        LOG_INFO("╠════════════════════════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  This test uses the CORRECT approach:                                               ║");
        LOG_INFO("║  1. Allocate VRAM on GPU 1 via KFD ALLOC_MEMORY_OF_GPU                              ║");
        LOG_INFO("║  2. Map to GPU 0 via KFD MAP_MEMORY_TO_GPU                                          ║");
        LOG_INFO("║  3. GPU 0 writes directly to GPU 1's VRAM (true P2P)                                ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════════════════════╝\n");

        // Open KFD
        int kfd_fd = open("/dev/kfd", O_RDWR);
        if (kfd_fd < 0)
        {
            GTEST_SKIP() << "Cannot open /dev/kfd: " << strerror(errno);
        }
        LOG_INFO("Opened /dev/kfd");

        // KFD ioctl definitions (from linux/kfd_ioctl.h)
        // We define them locally to avoid extern "C" conflicts
        #define AMDKFD_IOCTL_BASE 'K'
        #define AMDKFD_IOWR_LOCAL(nr, type) _IOWR(AMDKFD_IOCTL_BASE, nr, type)
        #define AMDKFD_IOW_LOCAL(nr, type) _IOW(AMDKFD_IOCTL_BASE, nr, type)

        // Memory flags
        constexpr uint32_t KFD_ALLOC_MEM_FLAGS_VRAM = (1 << 0);
        constexpr uint32_t KFD_ALLOC_MEM_FLAGS_WRITABLE = (1 << 5);
        constexpr uint32_t KFD_ALLOC_MEM_FLAGS_PUBLIC = (1 << 7);

        // Local structure definitions (matching kernel kfd_ioctl.h)
        struct kfd_alloc_mem_args {
            uint64_t va_addr;
            uint64_t size;
            uint64_t handle;
            uint64_t mmap_offset;
            uint32_t gpu_id;
            uint32_t flags;
        };

        struct kfd_map_mem_args {
            uint64_t handle;
            uint64_t device_ids_array_ptr;
            uint32_t n_devices;
            uint32_t n_success;
        };

        struct kfd_free_mem_args {
            uint64_t handle;
        };

        // Structure for getting apertures
        struct kfd_ioctl_get_process_apertures_new_args_local
        {
            uint64_t kfd_process_device_apertures_ptr;
            uint32_t num_of_nodes;
            uint32_t pad;
        };

        struct kfd_process_device_apertures_local
        {
            uint64_t lds_base;
            uint64_t lds_limit;
            uint64_t scratch_base;
            uint64_t scratch_limit;
            uint64_t gpuvm_base;
            uint64_t gpuvm_limit;
            uint32_t gpu_id;
            uint32_t pad;
        };

        // Get GPU IDs via KFD apertures query
        LOG_INFO("\n=== Step 1: Query KFD GPU IDs ===");
        
        kfd_process_device_apertures_local apertures[8] = {};
        kfd_ioctl_get_process_apertures_new_args_local aperture_args = {};
        aperture_args.kfd_process_device_apertures_ptr = reinterpret_cast<uint64_t>(apertures);
        aperture_args.num_of_nodes = 8;

        // ioctl number 0x14 = GET_PROCESS_APERTURES_NEW
        int ioctl_apertures = _IOWR(AMDKFD_IOCTL_BASE, 0x14, kfd_ioctl_get_process_apertures_new_args_local);
        int ret = ioctl(kfd_fd, ioctl_apertures, &aperture_args);
        if (ret != 0)
        {
            close(kfd_fd);
            GTEST_SKIP() << "GET_PROCESS_APERTURES failed: " << strerror(errno);
        }

        LOG_INFO("Found " << aperture_args.num_of_nodes << " KFD nodes:");
        for (uint32_t i = 0; i < aperture_args.num_of_nodes && i < 8; ++i)
        {
            LOG_INFO("  Node " << i << ": gpu_id=" << apertures[i].gpu_id
                               << " gpuvm_base=0x" << std::hex << apertures[i].gpuvm_base << std::dec);
        }

        if (aperture_args.num_of_nodes < 2)
        {
            close(kfd_fd);
            GTEST_SKIP() << "Need at least 2 KFD GPU nodes";
        }

        uint32_t gpu0_id = apertures[0].gpu_id;
        uint32_t gpu1_id = apertures[1].gpu_id;
        LOG_INFO("\nUsing GPU 0 (id=" << gpu0_id << ") and GPU 1 (id=" << gpu1_id << ")");

        // Step 2: Allocate memory on GPU 1
        LOG_INFO("\n=== Step 2: Allocate VRAM on GPU 1 ===");
        
        const size_t alloc_size = 4 * 1024 * 1024;  // 4 MB
        
        kfd_alloc_mem_args alloc_args = {};
        alloc_args.va_addr = 0;  // Let KFD choose
        alloc_args.size = alloc_size;
        alloc_args.gpu_id = gpu1_id;
        alloc_args.flags = KFD_ALLOC_MEM_FLAGS_VRAM | 
                          KFD_ALLOC_MEM_FLAGS_WRITABLE |
                          KFD_ALLOC_MEM_FLAGS_PUBLIC;  // PUBLIC allows cross-GPU access

        // ioctl number 0x16 = ALLOC_MEMORY_OF_GPU
        int ioctl_alloc = _IOWR(AMDKFD_IOCTL_BASE, 0x16, kfd_alloc_mem_args);
        ret = ioctl(kfd_fd, ioctl_alloc, &alloc_args);
        if (ret != 0)
        {
            LOG_WARN("ALLOC_MEMORY_OF_GPU failed: " << strerror(errno) << " (errno=" << errno << ")");
            close(kfd_fd);
            GTEST_SKIP() << "ALLOC_MEMORY_OF_GPU failed";
        }

        uint64_t handle = alloc_args.handle;
        uint64_t mmap_offset = alloc_args.mmap_offset;
        LOG_INFO("  ✓ Allocated " << (alloc_size / (1024 * 1024)) << " MB on GPU 1");
        LOG_INFO("    handle=0x" << std::hex << handle << std::dec);
        LOG_INFO("    mmap_offset=0x" << std::hex << mmap_offset << std::dec);

        // Step 3: Map memory to BOTH GPUs (required for P2P)
        LOG_INFO("\n=== Step 3: Map memory to both GPUs ===");
        
        uint32_t device_ids[2] = {gpu0_id, gpu1_id};
        kfd_map_mem_args map_args = {};
        map_args.handle = handle;
        map_args.device_ids_array_ptr = reinterpret_cast<uint64_t>(device_ids);
        map_args.n_devices = 2;
        map_args.n_success = 0;

        // ioctl number 0x18 = MAP_MEMORY_TO_GPU
        int ioctl_map = _IOWR(AMDKFD_IOCTL_BASE, 0x18, kfd_map_mem_args);
        ret = ioctl(kfd_fd, ioctl_map, &map_args);
        if (ret != 0)
        {
            LOG_WARN("MAP_MEMORY_TO_GPU failed: " << strerror(errno) << " (errno=" << errno << ")");
            LOG_INFO("n_success=" << map_args.n_success);
            
            // Cleanup
            kfd_free_mem_args free_args = {};
            free_args.handle = handle;
            // ioctl number 0x17 = FREE_MEMORY_OF_GPU
            int ioctl_free = _IOW(AMDKFD_IOCTL_BASE, 0x17, kfd_free_mem_args);
            ioctl(kfd_fd, ioctl_free, &free_args);
            close(kfd_fd);
            
            LOG_INFO("\nThis failure indicates the kernel doesn't support P2P mapping between these GPUs.");
            LOG_INFO("Possible causes:");
            LOG_INFO("  1. GPUs on different PCIe root complexes without P2P support");
            LOG_INFO("  2. IOMMU blocking cross-device DMA");
            LOG_INFO("  3. Driver doesn't support P2P for this GPU generation");
            GTEST_SKIP() << "MAP_MEMORY_TO_GPU failed";
        }

        LOG_INFO("  ✓ Mapped to " << map_args.n_success << " GPUs");

        // Step 4: mmap the memory for CPU access (to initialize)
        LOG_INFO("\n=== Step 4: mmap for CPU access ===");
        
        // Open render node
        int render_fd = open("/dev/dri/renderD128", O_RDWR);
        if (render_fd < 0)
        {
            // Try other render nodes
            for (int i = 129; i < 140; ++i)
            {
                char path[32];
                snprintf(path, sizeof(path), "/dev/dri/renderD%d", i);
                render_fd = open(path, O_RDWR);
                if (render_fd >= 0) break;
            }
        }

        void* cpu_ptr = nullptr;
        if (render_fd >= 0)
        {
            cpu_ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, render_fd, mmap_offset);
            if (cpu_ptr == MAP_FAILED)
            {
                LOG_WARN("mmap failed: " << strerror(errno));
                cpu_ptr = nullptr;
            }
            else
            {
                LOG_INFO("  ✓ mmap'd at " << cpu_ptr);
            }
            close(render_fd);
        }

        // Step 5: Test P2P access
        LOG_INFO("\n=== Step 5: Test P2P access ===");
        
        if (cpu_ptr)
        {
            // Initialize via CPU
            float* data = static_cast<float*>(cpu_ptr);
            for (size_t i = 0; i < alloc_size / sizeof(float); ++i)
            {
                data[i] = static_cast<float>(i % 1000) * 0.001f;
            }
            LOG_INFO("  Initialized " << (alloc_size / sizeof(float)) << " floats via CPU");

            // Now use HIP to have GPU 0 read from this memory
            // The va_addr should be accessible from GPU 0 now
            LOG_INFO("  Virtual address for HIP: 0x" << std::hex << alloc_args.va_addr << std::dec);
            
            // Allocate output buffer on GPU 0
            rocm_backend.setDevice(0);
            void* gpu0_out = rocm_backend.allocate(alloc_size, 0);
            
            if (gpu0_out)
            {
                LOG_INFO("\n  Attempting hipMemcpy from mapped address to GPU 0...");
                
                // The mapped memory should be accessible at va_addr
                void* mapped_ptr = reinterpret_cast<void*>(alloc_args.va_addr);
                
                auto start = std::chrono::high_resolution_clock::now();
                bool copy_ok = rocm_backend.deviceToDevice(gpu0_out, mapped_ptr, alloc_size, 0);
                rocm_backend.synchronize(0);
                auto end = std::chrono::high_resolution_clock::now();
                
                if (copy_ok)
                {
                    double ms = std::chrono::duration<double, std::milli>(end - start).count();
                    double gbps = (alloc_size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
                    
                    LOG_INFO("  ✓ hipMemcpy succeeded!");
                    LOG_INFO("    Time: " << std::fixed << std::setprecision(2) << ms << " ms");
                    LOG_INFO("    Throughput: " << std::fixed << std::setprecision(2) << gbps << " GB/s");
                    
                    // Verify
                    std::vector<float> verify(alloc_size / sizeof(float));
                    rocm_backend.deviceToHost(verify.data(), gpu0_out, alloc_size, 0);
                    
                    int mismatches = 0;
                    for (size_t i = 0; i < verify.size() && mismatches < 5; ++i)
                    {
                        float expected = static_cast<float>(i % 1000) * 0.001f;
                        if (std::abs(verify[i] - expected) > 1e-6f)
                        {
                            mismatches++;
                        }
                    }
                    
                    if (mismatches == 0)
                    {
                        LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
                        LOG_INFO("║  ✓✓✓ DIRECT AMD-to-AMD P2P WORKS via KFD!                        ║");
                        LOG_INFO("║  Throughput: " << std::fixed << std::setprecision(2) << gbps << " GB/s");
                        LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
                    }
                    else
                    {
                        LOG_WARN("  Data verification had " << mismatches << " mismatches");
                    }
                }
                else
                {
                    LOG_WARN("  hipMemcpy failed - va_addr may not be valid for HIP");
                }
                
                rocm_backend.free(gpu0_out, 0);
            }
            
            munmap(cpu_ptr, alloc_size);
        }

        // Cleanup
        LOG_INFO("\n=== Cleanup ===");
        
        // Define unmap struct locally (ioctl 0x19)
        struct kfd_unmap_mem_args
        {
            uint64_t handle;
            uint64_t device_ids_array_ptr;
            uint32_t n_devices;
            uint32_t n_success;
        };
        
        kfd_unmap_mem_args unmap_args = {};
        unmap_args.handle = handle;
        unmap_args.device_ids_array_ptr = reinterpret_cast<uint64_t>(device_ids);
        unmap_args.n_devices = 2;
        int ioctl_unmap = _IOWR(AMDKFD_IOCTL_BASE, 0x19, kfd_unmap_mem_args);
        ioctl(kfd_fd, ioctl_unmap, &unmap_args);

        kfd_free_mem_args free_args = {};
        free_args.handle = handle;
        int ioctl_free_final = _IOW(AMDKFD_IOCTL_BASE, 0x17, kfd_free_mem_args);
        ioctl(kfd_fd, ioctl_free_final, &free_args);

        close(kfd_fd);
        
        LOG_INFO("  Done");
        SUCCEED() << "KFD P2P exploration complete";
    }

    /**
     * @brief Test AMD-to-AMD P2P via RAW BAR MAPPING (tinygrad approach)
     *
     * Since HSA/KFD P2P mapping fails, this test uses direct PCIe BAR mapping:
     * 1. mmap GPU 1's VRAM BAR from CPU (via sysfs resource1)
     * 2. Use hipHostRegister with hipHostRegisterIoMemory to make it GPU-accessible
     * 3. GPU 0 writes to the registered pointer → writes directly to GPU 1's VRAM
     *
     * This bypasses the HSA/KFD P2P machinery entirely and uses raw PCIe transactions.
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, AMD_to_AMD_DirectP2P_via_RawBAR)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  AMD-to-AMD DIRECT P2P via RAW BAR MAPPING (tinygrad approach)                      ║");
        LOG_INFO("╠════════════════════════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  This bypasses HSA/KFD entirely:                                                     ║");
        LOG_INFO("║  1. mmap GPU 1's VRAM BAR (sysfs resource1)                                          ║");
        LOG_INFO("║  2. hipHostRegister with hipHostRegisterIoMemory                                     ║");
        LOG_INFO("║  3. GPU 0 writes to the pointer → direct PCIe write to GPU 1 VRAM                   ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════════════════════╝");

        ROCmBackend rocm_backend;
        int device_count = rocm_backend.deviceCount();
        
        if (device_count < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ROCm devices (found " << device_count << ")";
        }

        // Step 1: Find AMD GPU 1's PCIe address and VRAM BAR
        LOG_INFO("\n=== Step 1: Find GPU 1's PCIe BAR ===");
        
        std::string amd_pci_addr;
        {
            // Enumerate AMD GPUs via sysfs
            DIR* gpu_dir = opendir("/sys/class/drm");
            if (!gpu_dir)
            {
                GTEST_SKIP() << "Cannot open /sys/class/drm";
            }
            
            std::vector<std::string> amd_addrs;
            struct dirent* entry;
            while ((entry = readdir(gpu_dir)) != nullptr)
            {
                std::string name = entry->d_name;
                if (name.find("card") != 0 || name.find("-") != std::string::npos) continue;
                
                std::string path = "/sys/class/drm/" + name + "/device/vendor";
                std::ifstream vendor_file(path);
                std::string vendor;
                if (vendor_file >> vendor && vendor == "0x1002")
                {  // AMD vendor ID
                    // Get PCI address from device symlink
                    char device_path[PATH_MAX];
                    std::string link_path = "/sys/class/drm/" + name + "/device";
                    ssize_t len = readlink(link_path.c_str(), device_path, PATH_MAX - 1);
                    if (len > 0)
                    {
                        device_path[len] = '\0';
                        std::string full_path = device_path;
                        size_t pos = full_path.rfind('/');
                        if (pos != std::string::npos)
                        {
                            amd_addrs.push_back(full_path.substr(pos + 1));
                        }
                    }
                }
            }
            closedir(gpu_dir);
            
            if (amd_addrs.size() < 2)
            {
                GTEST_SKIP() << "Need at least 2 AMD GPUs (found " << amd_addrs.size() << ")";
            }
            
            // Use second AMD GPU
            amd_pci_addr = amd_addrs[1];
            LOG_INFO("  Found AMD GPUs:");
            for (size_t i = 0; i < amd_addrs.size(); ++i)
            {
                LOG_INFO("    GPU " << i << ": " << amd_addrs[i]);
            }
            LOG_INFO("  Using GPU 1 (" << amd_pci_addr << ") as target");
        }

        // Step 2: mmap GPU 1's VRAM BAR
        LOG_INFO("\n=== Step 2: mmap GPU 1's VRAM BAR ===");
        
        // AMD GPUs use BAR0 for VRAM (unlike NVIDIA which uses BAR1)
        std::string bar_path = "/sys/bus/pci/devices/" + amd_pci_addr + "/resource0_wc";
        int bar_fd = open(bar_path.c_str(), O_RDWR);
        if (bar_fd < 0)
        {
            LOG_WARN("  Cannot open " << bar_path << ": " << strerror(errno));
            // Try non-write-combining variant
            bar_path = "/sys/bus/pci/devices/" + amd_pci_addr + "/resource0";
            bar_fd = open(bar_path.c_str(), O_RDWR);
            if (bar_fd < 0)
            {
                GTEST_SKIP() << "Cannot open VRAM BAR: " << strerror(errno);
            }
        }
        LOG_INFO("  Opened " << bar_path);
        
        // Get BAR size
        struct stat st;
        fstat(bar_fd, &st);
        size_t bar_size = st.st_size;
        LOG_INFO("  BAR size: " << (bar_size / (1024 * 1024)) << " MB");
        
        // Map a portion (4MB for testing)
        size_t map_size = 4 * 1024 * 1024;
        void* bar_ptr = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, bar_fd, 0);
        close(bar_fd);
        
        if (bar_ptr == MAP_FAILED)
        {
            GTEST_SKIP() << "mmap failed: " << strerror(errno);
        }
        LOG_INFO("  ✓ Mapped 4 MB at " << bar_ptr);

        // Step 3: Try hipHostRegister with hipHostRegisterIoMemory
        LOG_INFO("\n=== Step 3: Register with hipHostRegisterIoMemory ===");
        
        rocm_backend.setDevice(0);  // Use GPU 0 as source
        
        // hipHostRegisterIoMemory = 0x10 (from HIP headers)
        constexpr unsigned int HIP_HOST_REGISTER_IO_MEMORY = 0x10;
        
        hipError_t err = hipHostRegister(bar_ptr, map_size, 
                                          hipHostRegisterMapped | HIP_HOST_REGISTER_IO_MEMORY);
        if (err != hipSuccess)
        {
            LOG_WARN("  hipHostRegister(hipHostRegisterIoMemory) failed: " << hipGetErrorString(err));
            LOG_INFO("\n  This indicates hipHostRegisterIoMemory doesn't support AMD BAR mapping.");
            
            // Let's test if the BAR itself works via CPU access
            LOG_INFO("\n=== Step 3b: Test direct CPU access to BAR ===");
            LOG_INFO("  Writing pattern via CPU to GPU 1's BAR...");
            
            float* cpu_ptr = static_cast<float*>(bar_ptr);
            // Write a pattern
            for (size_t i = 0; i < 256; ++i)
            {
                cpu_ptr[i] = static_cast<float>(i) * 0.01f;
            }
            
            // Readback
            bool mismatch = false;
            for (size_t i = 0; i < 256; ++i)
            {
                float expected = static_cast<float>(i) * 0.01f;
                if (std::abs(cpu_ptr[i] - expected) > 1e-6f)
                {
                    LOG_WARN("  CPU BAR access mismatch at " << i << ": wrote " << expected << ", read " << cpu_ptr[i]);
                    mismatch = true;
                    break;
                }
            }
            
            if (!mismatch)
            {
                LOG_INFO("  ✓ Direct CPU access to AMD BAR WORKS!");
                LOG_INFO("  This confirms the BAR mapping is valid, but HIP cannot register it.");
            }
            else
            {
                LOG_WARN("  CPU BAR access failed - the BAR mapping may not work.");
            }
            
            munmap(bar_ptr, map_size);
            
            LOG_INFO("\n  Next step would be to use the AMDPageTableManager approach");
            LOG_INFO("  (direct GPU page table manipulation) to make GPU 0 access this memory.");
            
            GTEST_SKIP() << "hipHostRegister with hipHostRegisterIoMemory failed";
        }
        
        LOG_INFO("  ✓ hipHostRegister succeeded!");
        
        // Get device pointer
        void* dev_ptr = nullptr;
        err = hipHostGetDevicePointer(&dev_ptr, bar_ptr, 0);
        if (err != hipSuccess)
        {
            hipHostUnregister(bar_ptr);
            munmap(bar_ptr, map_size);
            GTEST_SKIP() << "hipHostGetDevicePointer failed: " << hipGetErrorString(err);
        }
        LOG_INFO("  Device pointer: " << dev_ptr);
        
        // Step 4: Test write from GPU 0 to GPU 1's VRAM via the mapped BAR
        LOG_INFO("\n=== Step 4: Test P2P write ===");
        
        // Initialize with test pattern via CPU
        float* cpu_ptr = static_cast<float*>(bar_ptr);
        for (size_t i = 0; i < 1024; ++i)
        {
            cpu_ptr[i] = 0.0f;
        }
        
        // Allocate source buffer on GPU 0
        void* src_ptr = rocm_backend.allocate(4096, 0);
        if (!src_ptr)
        {
            hipHostUnregister(bar_ptr);
            munmap(bar_ptr, map_size);
            GTEST_SKIP() << "Failed to allocate source buffer";
        }
        
        // Initialize source with test pattern
        std::vector<float> src_data(1024);
        for (size_t i = 0; i < 1024; ++i)
        {
            src_data[i] = static_cast<float>(i) * 0.001f;
        }
        rocm_backend.hostToDevice(src_ptr, src_data.data(), 4096, 0);
        
        // Copy from GPU 0's buffer to GPU 1's VRAM via the registered BAR pointer
        LOG_INFO("  Copying from GPU 0 buffer to GPU 1 VRAM via registered BAR...");
        auto start = std::chrono::high_resolution_clock::now();
        err = hipMemcpy(dev_ptr, src_ptr, 4096, hipMemcpyDeviceToDevice);
        rocm_backend.synchronize(0);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (err != hipSuccess)
        {
            LOG_WARN("  hipMemcpy failed: " << hipGetErrorString(err));
        }
        else
        {
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            LOG_INFO("  hipMemcpy completed in " << std::fixed << std::setprecision(2) << ms << " ms");
            
            // Verify via CPU read
            bool verified = true;
            for (size_t i = 0; i < 1024 && verified; ++i)
            {
                if (std::abs(cpu_ptr[i] - src_data[i]) > 1e-6f)
                {
                    LOG_WARN("  Mismatch at " << i << ": expected " << src_data[i] << ", got " << cpu_ptr[i]);
                    verified = false;
                }
            }
            
            if (verified)
            {
                LOG_INFO("\n╔══════════════════════════════════════════════════════════════════╗");
                LOG_INFO("║  ✓✓✓ AMD-to-AMD P2P via RAW BAR MAPPING WORKS!                    ║");
                LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
            }
        }
        
        // Cleanup
        rocm_backend.free(src_ptr, 0);
        hipHostUnregister(bar_ptr);
        munmap(bar_ptr, map_size);
        
        SUCCEED() << "Raw BAR P2P test complete";
    }

    /**
     * @brief Final summary and recommendations
     */
    TEST_F(Test__AMD_PCIe_P2P_Mapping, Summary_And_Recommendations)
    {
        LOG_INFO("\n╔════════════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║                    P2P INVESTIGATION SUMMARY                                         ║");
        LOG_INFO("╠════════════════════════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  AMD-to-AMD P2P FINDINGS:                                                            ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  1. hipDeviceCanAccessPeer() returns 0 for all AMD GPU pairs                         ║");
        LOG_INFO("║     - GPUs are on separate PCIe root complexes                                       ║");
        LOG_INFO("║     - No hardware P2P fabric (xGMI, Infinity Fabric, etc.)                           ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  2. hipMemcpyPeer() WORKS but uses HOST STAGING (~0.2 GB/s)                          ║");
        LOG_INFO("║     - Data goes: GPU0 → Host RAM → GPU1                                              ║");
        LOG_INFO("║     - This is NOT true P2P                                                           ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  3. HSA reports P2P access is NEVER ALLOWED                                          ║");
        LOG_INFO("║     - hsa_amd_agents_allow_access() fails with status 4104                           ║");
        LOG_INFO("║     - This is a fundamental driver/hardware limitation                               ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  4. KFD MAP_MEMORY_TO_GPU ioctl fails with EINVAL                                    ║");
        LOG_INFO("║     - The kernel driver doesn't support P2P mapping between these GPUs               ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  5. Direct BAR mmap from CPU WORKS ✓                                                 ║");
        LOG_INFO("║     - Can read/write GPU 1's VRAM via /sys/.../resource0                             ║");
        LOG_INFO("║     - BUT: hipHostRegisterIoMemory fails to register this for GPU access             ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  CONCLUSION:                                                                         ║");
        LOG_INFO("║  True direct AMD-to-AMD P2P is NOT possible on this hardware via standard APIs.      ║");
        LOG_INFO("║  The ONLY remaining option is:                                                       ║");
        LOG_INFO("║    - Direct GPU page table manipulation (tinygrad approach)                          ║");
        LOG_INFO("║    - Map GPU 1's BAR physical address into GPU 0's page tables                       ║");
        LOG_INFO("║    - This bypasses HSA/KFD entirely but is complex and fragile                       ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("╠════════════════════════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  WORKING DIRECTION: AMD BAR → NVIDIA                                                 ║");
        LOG_INFO("║    - AMD's amdgpu driver exposes real VRAM via sysfs BAR                             ║");
        LOG_INFO("║    - NVIDIA can read this memory via standard PCIe transactions                      ║");
        LOG_INFO("║    - hipMemcpy D2D works perfectly                                                   ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  BLOCKED DIRECTION: NVIDIA BAR → AMD                                                 ║");
        LOG_INFO("║    - NVIDIA's proprietary driver intercepts all BAR access                           ║");
        LOG_INFO("║    - sysfs resource1/resource1_wc provide dummy mappings (writes don't persist)      ║");
        LOG_INFO("║    - /dev/mem access is blocked by driver                                            ║");
        LOG_INFO("║    - All standard memory export APIs (IPC, VMM, dmabuf) cannot be imported by HIP    ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  NEW IN ROCm 7.1.1: hipHostRegisterIoMemory                                          ║");
        LOG_INFO("║    - AMD's equivalent of CUDA's CU_MEMHOSTREGISTER_IOMEMORY                          ║");
        LOG_INFO("║    - May enable HIP to access mmap'd NVIDIA BAR                                      ║");
        LOG_INFO("║    - Run HipHostRegisterIoMemory_NvidiaBAR test to check!                            ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("║  POTENTIAL WORKAROUNDS:                                                              ║");
        LOG_INFO("║    1. Use working direction (AMD→NVIDIA) and restructure data flow                   ║");
        LOG_INFO("║    2. Stage data through CPU RAM (lose P2P benefits)                                 ║");
        LOG_INFO("║    3. Use nvidia-peermem with RDMA-capable NIC for indirect path                     ║");
        LOG_INFO("║    4. Wait for cross-vendor DMA-BUF support (unlikely in near term)                  ║");
        LOG_INFO("║                                                                                      ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════════════════════════╝");

        SUCCEED() << "Investigation complete";
    }
#endif

} // namespace llaminar2::test::experimental
