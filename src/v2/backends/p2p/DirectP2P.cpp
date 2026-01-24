/**
 * @file DirectP2P.cpp
 * @brief Direct cross-vendor P2P via PCIe BAR mapping
 *
 * Implements PCIe BAR-based direct P2P between CUDA and ROCm devices.
 * This is true peer-to-peer - data flows directly over PCIe with no host memory bounce.
 *
 * ## PCIe BAR Implementation
 *
 * The key insight is that AMD GPU's VRAM is exposed via PCIe BAR0, and
 * CUDA's cuMemHostRegister with IOMEMORY flag can register this "device I/O
 * memory" for DMA access. This bypasses the normal P2P restrictions.
 *
 * Steps:
 * 1. Find AMD GPU's sysfs path (e.g., /sys/bus/pci/devices/0000:b1:00.0)
 * 2. Open and mmap() resource0 (BAR0)
 * 3. cuMemHostRegister() with CU_MEMHOSTREGISTER_IOMEMORY flag
 * 4. cuMemHostGetDevicePointer() to get CUDA-usable address
 * 5. cuMemcpyDtoD() between CUDA memory and BAR-mapped AMD memory
 */

#include "DirectP2P.h"
#include "../../utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <sstream>
#include <iomanip>
#include <sys/utsname.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    //--------------------------------------------------------------------------
    // PCIe BAR Discovery Functions
    //--------------------------------------------------------------------------

    namespace
    {
        /**
         * @brief Check if a PCI device is an AMD GPU
         */
        bool isAmdGpu(const std::string &pci_path)
        {
            // Read vendor ID
            std::string vendor_path = pci_path + "/vendor";
            std::ifstream vendor_file(vendor_path);
            if (!vendor_file.is_open())
                return false;

            std::string vendor_id;
            vendor_file >> vendor_id;

            // AMD vendor ID is 0x1002
            return vendor_id == "0x1002";
        }

        /**
         * @brief Get BAR0 size from sysfs resource file
         */
        size_t getBar0Size(const std::string &pci_path)
        {
            // Parse /sys/bus/pci/devices/XXXX:XX:XX.X/resource
            std::string resource_path = pci_path + "/resource";
            std::ifstream resource_file(resource_path);
            if (!resource_file.is_open())
                return 0;

            // First line is BAR0
            std::string line;
            if (!std::getline(resource_file, line))
                return 0;

            // Format: "0xSTART 0xEND 0xFLAGS"
            unsigned long long start, end, flags;
            if (sscanf(line.c_str(), "%llx %llx %llx", &start, &end, &flags) != 3)
                return 0;

            if (end <= start)
                return 0;
            return static_cast<size_t>(end - start + 1);
        }

        /**
         * @brief Discover all AMD GPU PCIe BARs
         */
        std::vector<PCIeBarInfo> discoverAmdBars()
        {
            std::vector<PCIeBarInfo> bars;

            const char *pci_path = "/sys/bus/pci/devices";
            DIR *dir = opendir(pci_path);
            if (!dir)
            {
                LOG_DEBUG("Cannot open " << pci_path);
                return bars;
            }

            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr)
            {
                if (entry->d_name[0] == '.')
                    continue;

                std::string device_path = std::string(pci_path) + "/" + entry->d_name;

                if (isAmdGpu(device_path))
                {
                    PCIeBarInfo info;
                    info.pci_address = entry->d_name;
                    info.sysfs_path = device_path + "/resource0";
                    info.bar_size = getBar0Size(device_path);
                    info.is_amd = true;

                    if (info.bar_size > 0)
                    {
                        LOG_DEBUG("Found AMD GPU BAR: " << info.pci_address
                                                        << " size=" << (info.bar_size / (1024 * 1024 * 1024)) << " GB");
                        bars.push_back(std::move(info));
                    }
                }
            }

            closedir(dir);
            return bars;
        }

        /**
         * @brief Check if we have permission to access PCIe BAR
         */
        bool canAccessBar(const std::string &bar_path)
        {
            int fd = open(bar_path.c_str(), O_RDWR);
            if (fd >= 0)
            {
                close(fd);
                return true;
            }
            return false;
        }

        /**
         * @brief Get PCIe bus ID for a ROCm device ordinal
         * @param rocm_ordinal ROCm device ordinal (0, 1, ...)
         * @return PCIe bus ID string (e.g., "0000:1a:00.0") or empty if failed
         */
        std::string getROCmPCIBusID(int rocm_ordinal)
        {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, rocm_ordinal) != hipSuccess)
            {
                return "";
            }

            // Format: domain:bus:device.function
            char bus_id[32];
            snprintf(bus_id, sizeof(bus_id), "%04x:%02x:%02x.0",
                     prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
            return std::string(bus_id);
        }

        /**
         * @brief Normalize PCIe bus ID for comparison (handles case differences)
         * Converts "0000:1a:00.0" to lowercase without leading zeros in domain
         */
        std::string normalizePCIBusID(const std::string &bus_id)
        {
            std::string result = bus_id;
            // Convert to lowercase for comparison
            std::transform(result.begin(), result.end(), result.begin(), ::tolower);
            return result;
        }

        /**
         * @brief Check if CUDA supports IOMEMORY registration
         */
        bool checkCudaIomemorySupport()
        {
            // Check CUDA driver version - IOMEMORY support is in most modern drivers
            int driverVersion = 0;
            if (cuDriverGetVersion(&driverVersion) == CUDA_SUCCESS)
            {
                // IOMEMORY flag exists in CUDA 11.0+ (version 11000+)
                return driverVersion >= 11000;
            }
            return false;
        }

        /**
         * @brief Get CUDA driver version string
         */
        std::string getCudaDriverVersion()
        {
            int driver_version = 0;
            cuDriverGetVersion(&driver_version);
            std::ostringstream ss;
            ss << driver_version / 1000 << "." << (driver_version % 1000) / 10;
            return ss.str();
        }

        /**
         * @brief Get ROCm driver version string
         */
        std::string getRocmDriverVersion()
        {
            int runtime_version = 0;
            hipRuntimeGetVersion(&runtime_version);

            // HIP version format: MAJOR * 10000000 + MINOR * 100000 + PATCH
            int major = runtime_version / 10000000;
            int minor = (runtime_version % 10000000) / 100000;
            int patch = (runtime_version % 100000);

            std::ostringstream ss;
            ss << major << "." << minor << "." << patch;
            return ss.str();
        }
    } // anonymous namespace

    //--------------------------------------------------------------------------
    // DirectP2PCapability Implementation
    //--------------------------------------------------------------------------

    std::string DirectP2PCapability::describe() const
    {
        std::ostringstream ss;
        ss << "Direct P2P Capabilities (PCIe BAR only):\n";
        ss << "  Kernel: " << kernel_version << "\n";
        ss << "  CUDA Driver: " << cuda_driver_version << "\n";
        ss << "  ROCm Driver: " << rocm_driver_version << "\n";
        ss << "\n  === PCIe BAR ===\n";
        ss << "  PCIe BAR Access: " << (pcie_bar_accessible ? "YES" : "NO (need root?)") << "\n";
        ss << "  CUDA IOMEMORY Support: " << (pcie_bar_iomemory_supported ? "YES" : "NO") << "\n";
        ss << "  AMD BARs Found: " << discovered_bars.size() << "\n";
        for (const auto &bar : discovered_bars)
        {
            ss << "    - " << bar.pci_address << ": "
               << (bar.bar_size / (1024 * 1024 * 1024)) << " GB\n";
        }
        ss << "\n  IOMMU: " << (iommu_enabled ? "ENABLED (may affect DMA)" : "disabled") << "\n";
        ss << "  Direct P2P Possible: " << (canDoDirectP2P() ? "YES" : "NO") << "\n";
        return ss.str();
    }

    //--------------------------------------------------------------------------
    // DirectP2PEngine Implementation
    //--------------------------------------------------------------------------

    struct DirectP2PEngine::Impl
    {
        // Streams for async operations (multiple for concurrent transfers)
        void *cuda_stream_read = nullptr;  // For overlapped reads
        void *cuda_stream_write = nullptr; // For overlapped writes
        int cuda_device = -1;
        int rocm_device = -1;

        // Single BAR (legacy API)
        PCIeBarInfo bar_info;
        CUdeviceptr cuda_bar_ptr = 0; // CUDA device pointer to BAR
        bool bar_mapped = false;

        // Multi-GPU BAR resources (new API)
        std::vector<PCIeBarInfo> mapped_bars;          // All mapped AMD GPU BARs
        std::map<int, size_t> rocm_ordinal_to_bar_idx; // Map ordinal to bar index

        // CUDA context (retained)
        CUcontext cuda_ctx = nullptr;
        CUdevice cu_device = 0;

        ~Impl()
        {
            cleanup();
        }

        void cleanup()
        {
            // Cleanup single BAR
            if (bar_mapped && bar_info.mapped_ptr)
            {
                cuMemHostUnregister(bar_info.mapped_ptr);
                munmap(bar_info.mapped_ptr, bar_info.mapped_size);
                bar_info.mapped_ptr = nullptr;
                bar_info.mapped_size = 0;
                cuda_bar_ptr = 0;
                bar_mapped = false;
            }

            if (bar_info.bar_fd >= 0)
            {
                close(bar_info.bar_fd);
                bar_info.bar_fd = -1;
            }

            // Cleanup multi-GPU BARs
            for (auto &bar : mapped_bars)
            {
                if (bar.mapped_ptr)
                {
                    cuMemHostUnregister(bar.mapped_ptr);
                    munmap(bar.mapped_ptr, bar.mapped_size);
                    bar.mapped_ptr = nullptr;
                }
                if (bar.bar_fd >= 0)
                {
                    close(bar.bar_fd);
                    bar.bar_fd = -1;
                }
            }
            mapped_bars.clear();
            rocm_ordinal_to_bar_idx.clear();

            // Cleanup streams
            if (cuda_stream_read)
            {
                cuStreamDestroy(reinterpret_cast<CUstream>(cuda_stream_read));
                cuda_stream_read = nullptr;
            }
            if (cuda_stream_write)
            {
                cuStreamDestroy(reinterpret_cast<CUstream>(cuda_stream_write));
                cuda_stream_write = nullptr;
            }

            // Release CUDA context
            if (cuda_ctx && cu_device >= 0)
            {
                cuDevicePrimaryCtxRelease(cu_device);
                cuda_ctx = nullptr;
            }
        }
    };

    DirectP2PEngine::DirectP2PEngine()
        : impl_(std::make_unique<Impl>())
    {
    }

    DirectP2PEngine::~DirectP2PEngine() = default;

    DirectP2PCapability DirectP2PEngine::probeCapabilities()
    {
        DirectP2PCapability caps;

        // Get kernel version
        struct utsname uname_data;
        if (uname(&uname_data) == 0)
        {
            caps.kernel_version = uname_data.release;
        }

        // Check for IOMMU
        FILE *f = fopen("/sys/kernel/iommu_groups/0/type", "r");
        if (f)
        {
            caps.iommu_enabled = true;
            fclose(f);
        }

        // Discover AMD GPU BARs
        caps.discovered_bars = discoverAmdBars();

        // Check if we can access any BAR with usable size (need root or udev rule)
        for (const auto &bar : caps.discovered_bars)
        {
            if (bar.bar_size > 0 && canAccessBar(bar.sysfs_path))
            {
                caps.pcie_bar_accessible = true;
                break;
            }
        }

        caps.cuda_driver_version = getCudaDriverVersion();
        caps.rocm_driver_version = getRocmDriverVersion();
        caps.pcie_bar_iomemory_supported = checkCudaIomemorySupport();

        return caps;
    }

    //--------------------------------------------------------------------------
    // Single-GPU PCIe BAR P2P Implementation
    //--------------------------------------------------------------------------

    bool DirectP2PEngine::initializePCIeBar(DeviceId cuda_device, DeviceId rocm_device,
                                            size_t bar_offset, size_t map_size)
    {
        if (cuda_device.type != DeviceType::CUDA)
        {
            LOG_ERROR("First device must be CUDA for PCIe BAR P2P");
            return false;
        }
        if (rocm_device.type != DeviceType::ROCm)
        {
            LOG_ERROR("Second device must be ROCm for PCIe BAR P2P");
            return false;
        }

        impl_->cleanup();
        impl_->cuda_device = cuda_device.ordinal;
        impl_->rocm_device = rocm_device.ordinal;

        // Get the PCIe bus ID for the requested ROCm device
        std::string requested_pci = getROCmPCIBusID(rocm_device.ordinal);
        if (requested_pci.empty())
        {
            LOG_ERROR("Cannot get PCIe bus ID for ROCm device " << rocm_device.ordinal);
            return false;
        }
        LOG_INFO("ROCm device " << rocm_device.ordinal << " has PCIe bus ID: " << requested_pci);

        // Discover AMD GPU BARs
        auto bars = discoverAmdBars();
        if (bars.empty())
        {
            LOG_ERROR("No AMD GPU BARs found");
            return false;
        }

        // Find the BAR matching the requested ROCm device
        // The BAR's pci_address is the full PCI address (e.g., "0000:1a:00.0")
        // We need to match it to the device's PCI address
        PCIeBarInfo *best_bar = nullptr;
        std::string norm_requested = normalizePCIBusID(requested_pci);

        // First try to find an exact match for the requested ROCm device
        for (auto &bar : bars)
        {
            // The bar.pci_address might be different format, normalize for comparison
            // sysfs uses lowercase, hip uses uppercase in some fields
            std::string norm_bar = normalizePCIBusID(bar.pci_address);

            if (bar.bar_size > 0 && norm_bar == norm_requested)
            {
                best_bar = &bar;
                LOG_INFO("Found matching BAR for ROCm device " << rocm_device.ordinal
                                                               << " at " << bar.pci_address);
                break;
            }
        }

        // If no exact match, try to find the BAR on the same PCIe switch/bus
        // by comparing the bus portion (e.g., "0000:1a" from "0000:1a:00.0")
        if (!best_bar)
        {
            LOG_WARN("No exact BAR match for " << requested_pci << ", searching for same-bus BAR");
            std::string req_bus_prefix = norm_requested.substr(0, norm_requested.find_last_of(':'));

            for (auto &bar : bars)
            {
                std::string norm_bar = normalizePCIBusID(bar.pci_address);
                std::string bar_bus_prefix = norm_bar.substr(0, norm_bar.find_last_of(':'));

                if (bar.bar_size > 0 && bar_bus_prefix == req_bus_prefix)
                {
                    best_bar = &bar;
                    LOG_INFO("Found same-bus BAR for ROCm device " << rocm_device.ordinal
                                                                   << " at " << bar.pci_address);
                    break;
                }
            }
        }

        // If still no match, fall back to largest BAR (legacy behavior) with warning
        if (!best_bar)
        {
            LOG_WARN("No matching BAR found for ROCm device " << rocm_device.ordinal
                                                              << " at " << requested_pci << ", falling back to largest BAR");
            for (auto &bar : bars)
            {
                if (bar.bar_size > 0 && (!best_bar || bar.bar_size > best_bar->bar_size))
                {
                    best_bar = &bar;
                }
            }
        }

        if (!best_bar)
        {
            LOG_ERROR("No AMD GPU BAR with usable size found");
            return false;
        }

        impl_->bar_info = *best_bar;
        LOG_INFO("Using AMD BAR: " << impl_->bar_info.pci_address
                                   << " (" << (impl_->bar_info.bar_size / (1024 * 1024 * 1024)) << " GB)");

        // Open BAR file
        impl_->bar_info.bar_fd = open(impl_->bar_info.sysfs_path.c_str(), O_RDWR | O_SYNC);
        if (impl_->bar_info.bar_fd < 0)
        {
            LOG_ERROR("Cannot open BAR at " << impl_->bar_info.sysfs_path
                                            << " (errno=" << errno << ")");
            LOG_ERROR("PCIe BAR access requires elevated privileges. Options:");
            LOG_ERROR("  1. Run with sudo");
            LOG_ERROR("  2. Create udev rule: sudo tee /etc/udev/rules.d/99-amd-gpu-bar.rules << 'EOF'");
            LOG_ERROR("     SUBSYSTEM==\"pci\", ATTR{vendor}==\"0x1002\", ATTR{class}==\"0x030000\", \\");
            LOG_ERROR("         RUN+=\"/bin/chmod 0666 /sys/bus/pci/devices/%k/resource0\"");
            LOG_ERROR("     EOF");
            LOG_ERROR("     sudo udevadm control --reload-rules && sudo udevadm trigger");
            LOG_ERROR("  3. One-time: sudo chmod 666 " << impl_->bar_info.sysfs_path);
            return false;
        }

        // Determine map size
        size_t actual_map_size = map_size;
        if (actual_map_size == 0)
        {
            // Default to 64MB (reasonable for transfers, not the whole BAR)
            actual_map_size = 64 * 1024 * 1024;
        }
        actual_map_size = std::min(actual_map_size, impl_->bar_info.bar_size - bar_offset);

        // mmap the BAR
        void *mapped = mmap(nullptr, actual_map_size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, impl_->bar_info.bar_fd, bar_offset);
        if (mapped == MAP_FAILED)
        {
            LOG_ERROR("Failed to mmap BAR (errno=" << errno << ")");
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        impl_->bar_info.mapped_ptr = mapped;
        impl_->bar_info.mapped_size = actual_map_size;
        LOG_INFO("Mapped " << (actual_map_size / (1024 * 1024)) << " MB of BAR at offset " << bar_offset);

        // Initialize CUDA context
        CUresult err;
        err = cuInit(0);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuInit failed: " << err);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        CUdevice cu_device;
        err = cuDeviceGet(&cu_device, cuda_device.ordinal);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuDeviceGet failed: " << err);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }
        impl_->cu_device = cu_device;

        // Use primary context instead of creating a new one
        CUcontext cu_ctx;
        err = cuDevicePrimaryCtxRetain(&cu_ctx, cu_device);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuDevicePrimaryCtxRetain failed: " << err);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }
        impl_->cuda_ctx = cu_ctx;

        err = cuCtxSetCurrent(cu_ctx);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuCtxSetCurrent failed: " << err);
            cuDevicePrimaryCtxRelease(cu_device);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        // Register BAR memory with CUDA using IOMEMORY flag
        // This is the key - IOMEMORY tells CUDA this is device I/O memory
        // NOTE: This requires root privileges - CUDA's IOMEMORY registration
        // checks for CAP_SYS_ADMIN, not just file permissions
        unsigned int flags = CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY;
        err = cuMemHostRegister(mapped, actual_map_size, flags);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuMemHostRegister with IOMEMORY failed: " << err);
            if (err == 800) // CUDA_ERROR_INVALID_VALUE
            {
                LOG_ERROR("CUDA IOMEMORY registration requires root privileges.");
                LOG_ERROR("The NVIDIA driver checks CAP_SYS_ADMIN for DMA to device I/O memory.");
                LOG_ERROR("Options:");
                LOG_ERROR("  1. Run with sudo");
                LOG_ERROR("  2. Run as root user");
            }
            cuDevicePrimaryCtxRelease(cu_device);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        // Get device pointer for the registered memory
        err = cuMemHostGetDevicePointer(&impl_->cuda_bar_ptr, mapped, 0);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuMemHostGetDevicePointer failed: " << err);
            cuMemHostUnregister(mapped);
            cuDevicePrimaryCtxRelease(cu_device);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        impl_->bar_mapped = true;
        pcie_bar_active_ = true;

        LOG_INFO("PCIe BAR P2P initialized successfully!");
        LOG_INFO("  CUDA device pointer to AMD BAR: " << std::hex << impl_->cuda_bar_ptr << std::dec);
        LOG_INFO("  Mapped size: " << (actual_map_size / (1024 * 1024)) << " MB");

        return true;
    }

    void *DirectP2PEngine::getCudaBarPointer() const
    {
        return reinterpret_cast<void *>(impl_->cuda_bar_ptr);
    }

    size_t DirectP2PEngine::getBarMappedSize() const
    {
        return impl_->bar_info.mapped_size;
    }

    void *DirectP2PEngine::getBarHostPtr() const
    {
        return impl_->bar_info.mapped_ptr;
    }

    size_t DirectP2PEngine::getBarOffset() const
    {
        return 0; // We always map from offset 0 in this implementation
    }

    DirectP2PResult DirectP2PEngine::transferViaPCIeBar(void *cuda_ptr, size_t bar_offset,
                                                        size_t num_bytes, Direction direction)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes;

        if (!pcie_bar_active_)
        {
            result.error_message = "PCIe BAR not initialized - call initializePCIeBar() first";
            return result;
        }

        if (bar_offset + num_bytes > impl_->bar_info.mapped_size)
        {
            result.error_message = "Transfer exceeds mapped BAR region";
            return result;
        }

        CUdeviceptr bar_addr = impl_->cuda_bar_ptr + bar_offset;
        CUdeviceptr cuda_addr = reinterpret_cast<CUdeviceptr>(cuda_ptr);

        auto start = std::chrono::high_resolution_clock::now();

        CUresult err;
        if (direction == Direction::ToNVIDIA)
        {
            // Read from AMD BAR → CUDA memory
            err = cuMemcpyDtoD(cuda_addr, bar_addr, num_bytes);
            result.transfer_path = "AMD BAR → NVIDIA (read from BAR)";
        }
        else
        {
            // Write CUDA memory → AMD BAR
            err = cuMemcpyDtoD(bar_addr, cuda_addr, num_bytes);
            result.transfer_path = "NVIDIA → AMD BAR (write to BAR)";
        }

        if (err != CUDA_SUCCESS)
        {
            result.error_message = "cuMemcpyDtoD failed: " + std::to_string(err);
            return result;
        }

        // Sync
        cuCtxSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.throughput_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / (result.transfer_time_ms / 1000.0);
        result.success = true;

        return result;
    }

    DirectP2PResult DirectP2PEngine::benchmarkPCIeBar(size_t num_bytes, int iterations)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes;

        if (!pcie_bar_active_)
        {
            result.error_message = "PCIe BAR not initialized";
            return result;
        }

        if (num_bytes > impl_->bar_info.mapped_size)
        {
            result.error_message = "Transfer size exceeds mapped BAR region";
            return result;
        }

        LOG_INFO("=== PCIe BAR Direct P2P Benchmark ===");
        LOG_INFO("Transfer size: " << (num_bytes / (1024 * 1024)) << " MB");
        LOG_INFO("Iterations: " << iterations);

        // Allocate CUDA test buffer
        CUdeviceptr cuda_buf;
        CUresult err = cuMemAlloc(&cuda_buf, num_bytes);
        if (err != CUDA_SUCCESS)
        {
            result.error_message = "Failed to allocate CUDA buffer";
            return result;
        }

        // Initialize CUDA buffer
        cuMemsetD8(cuda_buf, 0xAB, num_bytes);
        cuCtxSynchronize();

        // Warmup
        LOG_INFO("Warming up...");
        cuMemcpyDtoD(cuda_buf, impl_->cuda_bar_ptr, num_bytes);
        cuMemcpyDtoD(impl_->cuda_bar_ptr, cuda_buf, num_bytes);
        cuCtxSynchronize();

        // Benchmark AMD → NVIDIA (read from BAR)
        double read_total_ms = 0.0;
        for (int i = 0; i < iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuMemcpyDtoD(cuda_buf, impl_->cuda_bar_ptr, num_bytes);
            cuCtxSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            read_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_read_ms = read_total_ms / iterations;
        result.read_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_read_ms / 1000.0);

        // Benchmark NVIDIA → AMD (write to BAR)
        double write_total_ms = 0.0;
        for (int i = 0; i < iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuMemcpyDtoD(impl_->cuda_bar_ptr, cuda_buf, num_bytes);
            cuCtxSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            write_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_write_ms = write_total_ms / iterations;
        result.write_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_write_ms / 1000.0);

        // Overall average
        result.throughput_gbps = (result.read_gbps + result.write_gbps) / 2.0;
        result.transfer_time_ms = (avg_read_ms + avg_write_ms) / 2.0;
        result.success = true;
        result.transfer_path = "PCIe BAR Direct P2P";

        LOG_INFO("\n╔══════════════════════════════════════════════════════════════╗");
        LOG_INFO("║           PCIe BAR DIRECT P2P BENCHMARK RESULTS              ║");
        LOG_INFO("╠══════════════════════════════════════════════════════════════╣");
        LOG_INFO("║ AMD → NVIDIA (read from BAR):  " << std::fixed << std::setprecision(2)
                                                     << std::setw(6) << result.read_gbps << " GB/s                    ║");
        LOG_INFO("║ NVIDIA → AMD (write to BAR):   " << std::fixed << std::setprecision(2)
                                                     << std::setw(6) << result.write_gbps << " GB/s                    ║");
        LOG_INFO("╚══════════════════════════════════════════════════════════════╝");

        cuMemFree(cuda_buf);
        return result;
    }

    //--------------------------------------------------------------------------
    // Multi-GPU PCIe BAR P2P Implementation
    //--------------------------------------------------------------------------

    bool DirectP2PEngine::initializePCIeBarMultiGPU(DeviceId cuda_device,
                                                    const std::vector<DeviceId> &rocm_devices,
                                                    size_t map_size)
    {
        if (cuda_device.type != DeviceType::CUDA)
        {
            LOG_ERROR("First device must be CUDA for PCIe BAR P2P");
            return false;
        }

        for (const auto &dev : rocm_devices)
        {
            if (dev.type != DeviceType::ROCm)
            {
                LOG_ERROR("All target devices must be ROCm for PCIe BAR P2P");
                return false;
            }
        }

        impl_->cleanup();
        impl_->cuda_device = cuda_device.ordinal;
        impl_->mapped_bars.clear();
        impl_->rocm_ordinal_to_bar_idx.clear();

        // Discover all AMD GPU BARs
        auto discovered_bars = discoverAmdBars();
        if (discovered_bars.empty())
        {
            LOG_ERROR("No AMD GPU BARs found");
            return false;
        }

        LOG_INFO("Found " << discovered_bars.size() << " AMD GPU BARs");

        // Initialize CUDA context
        CUresult err = cuInit(0);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuInit failed: " << err);
            return false;
        }

        CUdevice cu_device;
        err = cuDeviceGet(&cu_device, cuda_device.ordinal);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuDeviceGet failed: " << err);
            return false;
        }
        impl_->cu_device = cu_device;

        CUcontext cu_ctx;
        err = cuDevicePrimaryCtxRetain(&cu_ctx, cu_device);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuDevicePrimaryCtxRetain failed: " << err);
            return false;
        }
        impl_->cuda_ctx = cu_ctx;

        err = cuCtxSetCurrent(cu_ctx);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuCtxSetCurrent failed: " << err);
            cuDevicePrimaryCtxRelease(cu_device);
            return false;
        }

        // Create streams for overlapped operations
        CUstream read_stream, write_stream;
        err = cuStreamCreate(&read_stream, CU_STREAM_NON_BLOCKING);
        if (err == CUDA_SUCCESS)
        {
            impl_->cuda_stream_read = reinterpret_cast<void *>(read_stream);
        }
        err = cuStreamCreate(&write_stream, CU_STREAM_NON_BLOCKING);
        if (err == CUDA_SUCCESS)
        {
            impl_->cuda_stream_write = reinterpret_cast<void *>(write_stream);
        }

        // Determine map size per BAR
        size_t actual_map_size = map_size > 0 ? map_size : 64 * 1024 * 1024; // Default 64 MB

        // Map BAR for each requested ROCm device
        size_t bar_idx = 0;
        for (const auto &rocm_dev : rocm_devices)
        {
            // Find matching BAR by device ordinal
            // Note: This is heuristic - BAR order may not match device ordinal
            if (bar_idx >= discovered_bars.size())
            {
                LOG_WARN("Not enough AMD GPU BARs for all requested devices");
                break;
            }

            PCIeBarInfo bar = discovered_bars[bar_idx];
            LOG_INFO("Mapping BAR for ROCm device " << rocm_dev.ordinal
                                                    << " from " << bar.pci_address);

            // Open BAR file
            bar.bar_fd = open(bar.sysfs_path.c_str(), O_RDWR | O_SYNC);
            if (bar.bar_fd < 0)
            {
                LOG_ERROR("Cannot open BAR at " << bar.sysfs_path << " (errno=" << errno << ")");
                LOG_ERROR("Run: sudo chmod 666 " << bar.sysfs_path);
                bar_idx++;
                continue;
            }

            // Map the BAR
            size_t to_map = std::min(actual_map_size, bar.bar_size);
            void *mapped = mmap(nullptr, to_map, PROT_READ | PROT_WRITE,
                                MAP_SHARED, bar.bar_fd, 0);
            if (mapped == MAP_FAILED)
            {
                LOG_ERROR("Failed to mmap BAR for " << bar.pci_address);
                close(bar.bar_fd);
                bar.bar_fd = -1;
                bar_idx++;
                continue;
            }

            bar.mapped_ptr = mapped;
            bar.mapped_size = to_map;
            bar.device_ordinal = rocm_dev.ordinal;

            // Register with CUDA
            unsigned int flags = CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY;
            err = cuMemHostRegister(mapped, to_map, flags);
            if (err != CUDA_SUCCESS)
            {
                LOG_ERROR("cuMemHostRegister failed for " << bar.pci_address << ": " << err);
                munmap(mapped, to_map);
                close(bar.bar_fd);
                bar.bar_fd = -1;
                bar.mapped_ptr = nullptr;
                bar_idx++;
                continue;
            }

            // Get CUDA device pointer
            CUdeviceptr cuda_ptr;
            err = cuMemHostGetDevicePointer(&cuda_ptr, mapped, 0);
            if (err != CUDA_SUCCESS)
            {
                LOG_ERROR("cuMemHostGetDevicePointer failed: " << err);
                cuMemHostUnregister(mapped);
                munmap(mapped, to_map);
                close(bar.bar_fd);
                bar.bar_fd = -1;
                bar.mapped_ptr = nullptr;
                bar_idx++;
                continue;
            }

            bar.cuda_device_ptr = reinterpret_cast<void *>(cuda_ptr);

            // Store the mapping
            impl_->rocm_ordinal_to_bar_idx[rocm_dev.ordinal] = impl_->mapped_bars.size();
            impl_->mapped_bars.push_back(std::move(bar));

            LOG_INFO("  Mapped " << (to_map / (1024 * 1024)) << " MB, CUDA ptr: "
                                 << std::hex << cuda_ptr << std::dec);

            bar_idx++;
        }

        if (impl_->mapped_bars.empty())
        {
            LOG_ERROR("Failed to map any AMD GPU BARs");
            return false;
        }

        pcie_bar_active_ = true;
        LOG_INFO("PCIe BAR Multi-GPU initialized: " << impl_->mapped_bars.size() << " BARs mapped");

        return true;
    }

    size_t DirectP2PEngine::getNumMappedBars() const
    {
        return impl_->mapped_bars.size();
    }

    void *DirectP2PEngine::getCudaBarPointerForDevice(int rocm_ordinal) const
    {
        auto it = impl_->rocm_ordinal_to_bar_idx.find(rocm_ordinal);
        if (it == impl_->rocm_ordinal_to_bar_idx.end())
        {
            return nullptr;
        }
        return impl_->mapped_bars[it->second].cuda_device_ptr;
    }

    DirectP2PResult DirectP2PEngine::broadcastToAMD(const void *cuda_src, size_t num_bytes,
                                                    void *stream)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes * impl_->mapped_bars.size();

        if (!pcie_bar_active_ || impl_->mapped_bars.empty())
        {
            result.error_message = "PCIe BAR not initialized";
            return result;
        }

        CUstream cu_stream = stream ? reinterpret_cast<CUstream>(stream)
                                    : reinterpret_cast<CUstream>(impl_->cuda_stream_write);

        auto start = std::chrono::high_resolution_clock::now();

        // Write to all AMD BARs (uses fast posted writes ~2.65 GB/s per target)
        for (auto &bar : impl_->mapped_bars)
        {
            if (num_bytes > bar.mapped_size)
            {
                LOG_WARN("Transfer size exceeds BAR mapped size for device " << bar.device_ordinal);
                continue;
            }

            CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar.cuda_device_ptr);
            CUdeviceptr src_ptr = reinterpret_cast<CUdeviceptr>(cuda_src);

            CUresult err = cuMemcpyDtoDAsync(bar_ptr, src_ptr, num_bytes, cu_stream);
            if (err != CUDA_SUCCESS)
            {
                LOG_ERROR("cuMemcpyDtoDAsync to BAR " << bar.device_ordinal << " failed: " << err);
            }
        }

        // Wait for all writes to complete
        cuStreamSynchronize(cu_stream);

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Total throughput = total bytes / time
        result.throughput_gbps = (result.bytes_transferred / (1024.0 * 1024.0 * 1024.0)) /
                                 (result.transfer_time_ms / 1000.0);
        result.write_gbps = result.throughput_gbps;
        result.dual_write_gbps = result.throughput_gbps;
        result.success = true;
        result.transfer_path = "NVIDIA -> " + std::to_string(impl_->mapped_bars.size()) + "x AMD (broadcast)";

        return result;
    }

    DirectP2PResult DirectP2PEngine::gatherFromAMDOverlapped(void *cuda_dst, size_t num_bytes,
                                                             void *stream)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes * impl_->mapped_bars.size();

        if (!pcie_bar_active_ || impl_->mapped_bars.empty())
        {
            result.error_message = "PCIe BAR not initialized";
            return result;
        }

        // Use separate streams for overlapped reads
        CUstream read_stream = reinterpret_cast<CUstream>(impl_->cuda_stream_read);
        CUstream write_stream = reinterpret_cast<CUstream>(impl_->cuda_stream_write);

        auto start = std::chrono::high_resolution_clock::now();

        // Launch reads from all AMD BARs using different streams for overlap
        size_t offset = 0;
        for (size_t i = 0; i < impl_->mapped_bars.size(); ++i)
        {
            auto &bar = impl_->mapped_bars[i];
            if (num_bytes > bar.mapped_size)
            {
                LOG_WARN("Transfer size exceeds BAR mapped size for device " << bar.device_ordinal);
                continue;
            }

            CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar.cuda_device_ptr);
            CUdeviceptr dst_ptr = reinterpret_cast<CUdeviceptr>(cuda_dst) + offset;

            // Alternate streams for overlap
            CUstream s = (i % 2 == 0) ? read_stream : write_stream;
            if (!s)
                s = read_stream;

            CUresult err = cuMemcpyDtoDAsync(dst_ptr, bar_ptr, num_bytes, s);
            if (err != CUDA_SUCCESS)
            {
                LOG_ERROR("cuMemcpyDtoDAsync from BAR " << bar.device_ordinal << " failed: " << err);
            }

            offset += num_bytes;
        }

        // Wait for all reads to complete
        if (read_stream)
            cuStreamSynchronize(read_stream);
        if (write_stream)
            cuStreamSynchronize(write_stream);

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.throughput_gbps = (result.bytes_transferred / (1024.0 * 1024.0 * 1024.0)) /
                                 (result.transfer_time_ms / 1000.0);
        result.read_gbps = result.throughput_gbps;
        result.dual_read_gbps = result.throughput_gbps;
        result.success = true;
        result.used_overlap = true;
        result.transfer_path = std::to_string(impl_->mapped_bars.size()) + "x AMD -> NVIDIA (overlapped gather)";

        return result;
    }

    DirectP2PResult DirectP2PEngine::transferOverlapped(void *cuda_read_dst, size_t read_bar_offset,
                                                        size_t read_bytes,
                                                        const void *cuda_write_src, size_t write_bar_offset,
                                                        size_t write_bytes)
    {
        DirectP2PResult result;
        result.bytes_transferred = read_bytes + write_bytes;

        if (!pcie_bar_active_ || impl_->mapped_bars.size() < 2)
        {
            result.error_message = "Need at least 2 AMD BARs for overlapped transfer";
            return result;
        }

        CUstream read_stream = reinterpret_cast<CUstream>(impl_->cuda_stream_read);
        CUstream write_stream = reinterpret_cast<CUstream>(impl_->cuda_stream_write);

        if (!read_stream || !write_stream)
        {
            result.error_message = "Streams not initialized";
            return result;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Simultaneously:
        // - Read from BAR 0 (AMD->NVIDIA) on read_stream
        // - Write to BAR 1 (NVIDIA->AMD) on write_stream

        if (cuda_read_dst && read_bytes > 0)
        {
            auto &bar0 = impl_->mapped_bars[0];
            if (read_bar_offset + read_bytes <= bar0.mapped_size)
            {
                CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar0.cuda_device_ptr) + read_bar_offset;
                CUdeviceptr dst_ptr = reinterpret_cast<CUdeviceptr>(cuda_read_dst);
                cuMemcpyDtoDAsync(dst_ptr, bar_ptr, read_bytes, read_stream);
            }
        }

        if (cuda_write_src && write_bytes > 0)
        {
            auto &bar1 = impl_->mapped_bars[1];
            if (write_bar_offset + write_bytes <= bar1.mapped_size)
            {
                CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar1.cuda_device_ptr) + write_bar_offset;
                CUdeviceptr src_ptr = reinterpret_cast<CUdeviceptr>(cuda_write_src);
                cuMemcpyDtoDAsync(bar_ptr, src_ptr, write_bytes, write_stream);
            }
        }

        // Wait for both to complete
        cuStreamSynchronize(read_stream);
        cuStreamSynchronize(write_stream);

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.throughput_gbps = (result.bytes_transferred / (1024.0 * 1024.0 * 1024.0)) /
                                 (result.transfer_time_ms / 1000.0);
        result.concurrent_gbps = result.throughput_gbps;
        result.success = true;
        result.used_overlap = true;
        result.transfer_path = "Overlapped: AMD->NVIDIA + NVIDIA->AMD";

        return result;
    }

    DirectP2PResult DirectP2PEngine::benchmarkAllModes(size_t num_bytes, int iterations)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes;

        if (!pcie_bar_active_ || impl_->mapped_bars.empty())
        {
            result.error_message = "PCIe BAR not initialized";
            return result;
        }

        LOG_INFO("");
        LOG_INFO("================================================================");
        LOG_INFO("       COMPREHENSIVE PCIe BAR P2P BENCHMARK                     ");
        LOG_INFO("================================================================");
        LOG_INFO(" Transfer size: " << (num_bytes / (1024 * 1024)) << " MB per operation");
        LOG_INFO(" Iterations: " << iterations);
        LOG_INFO(" Mapped BARs: " << impl_->mapped_bars.size());
        LOG_INFO("----------------------------------------------------------------");

        // Allocate test buffer
        CUdeviceptr cuda_buf;
        CUresult err = cuMemAlloc(&cuda_buf, num_bytes);
        if (err != CUDA_SUCCESS)
        {
            result.error_message = "Failed to allocate CUDA buffer";
            return result;
        }
        cuMemsetD8(cuda_buf, 0xAB, num_bytes);

        // Warmup
        for (auto &bar : impl_->mapped_bars)
        {
            CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar.cuda_device_ptr);
            cuMemcpyDtoD(bar_ptr, cuda_buf, std::min(num_bytes, bar.mapped_size));
            cuMemcpyDtoD(cuda_buf, bar_ptr, std::min(num_bytes, bar.mapped_size));
        }
        cuCtxSynchronize();

        // 1. Single write (NVIDIA -> AMD)
        double write_total = 0;
        auto &bar0 = impl_->mapped_bars[0];
        CUdeviceptr bar0_ptr = reinterpret_cast<CUdeviceptr>(bar0.cuda_device_ptr);
        for (int i = 0; i < iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuMemcpyDtoD(bar0_ptr, cuda_buf, num_bytes);
            cuCtxSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            write_total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        result.write_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / ((write_total / iterations) / 1000.0);
        LOG_INFO(" Single WRITE (NVIDIA->AMD):  " << std::fixed << std::setprecision(2)
                                                  << result.write_gbps << " GB/s");

        // 2. Single read (AMD -> NVIDIA)
        double read_total = 0;
        for (int i = 0; i < iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuMemcpyDtoD(cuda_buf, bar0_ptr, num_bytes);
            cuCtxSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            read_total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        result.read_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / ((read_total / iterations) / 1000.0);
        LOG_INFO(" Single READ (AMD->NVIDIA):   " << std::fixed << std::setprecision(2)
                                                  << result.read_gbps << " GB/s");

        // 3. Dual write (if 2+ BARs)
        if (impl_->mapped_bars.size() >= 2)
        {
            CUstream s1, s2;
            cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING);
            cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING);

            CUdeviceptr bar1_ptr = reinterpret_cast<CUdeviceptr>(impl_->mapped_bars[1].cuda_device_ptr);

            double dual_write_total = 0;
            for (int i = 0; i < iterations; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                cuMemcpyDtoDAsync(bar0_ptr, cuda_buf, num_bytes, s1);
                cuMemcpyDtoDAsync(bar1_ptr, cuda_buf, num_bytes, s2);
                cuStreamSynchronize(s1);
                cuStreamSynchronize(s2);
                auto end = std::chrono::high_resolution_clock::now();
                dual_write_total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            result.dual_write_gbps = (2 * num_bytes / (1024.0 * 1024.0 * 1024.0)) /
                                     ((dual_write_total / iterations) / 1000.0);
            LOG_INFO(" Dual WRITE (NVIDIA->2xAMD):  " << std::fixed << std::setprecision(2)
                                                      << result.dual_write_gbps << " GB/s (total)");

            // 4. Dual read
            double dual_read_total = 0;
            CUdeviceptr cuda_buf2;
            cuMemAlloc(&cuda_buf2, num_bytes);
            for (int i = 0; i < iterations; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                cuMemcpyDtoDAsync(cuda_buf, bar0_ptr, num_bytes, s1);
                cuMemcpyDtoDAsync(cuda_buf2, bar1_ptr, num_bytes, s2);
                cuStreamSynchronize(s1);
                cuStreamSynchronize(s2);
                auto end = std::chrono::high_resolution_clock::now();
                dual_read_total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            result.dual_read_gbps = (2 * num_bytes / (1024.0 * 1024.0 * 1024.0)) /
                                    ((dual_read_total / iterations) / 1000.0);
            LOG_INFO(" Dual READ (2xAMD->NVIDIA):   " << std::fixed << std::setprecision(2)
                                                      << result.dual_read_gbps << " GB/s (total)");

            // 5. Concurrent read+write
            double concurrent_total = 0;
            for (int i = 0; i < iterations; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                cuMemcpyDtoDAsync(cuda_buf, bar0_ptr, num_bytes, s1);  // Read from BAR0
                cuMemcpyDtoDAsync(bar1_ptr, cuda_buf2, num_bytes, s2); // Write to BAR1
                cuStreamSynchronize(s1);
                cuStreamSynchronize(s2);
                auto end = std::chrono::high_resolution_clock::now();
                concurrent_total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            result.concurrent_gbps = (2 * num_bytes / (1024.0 * 1024.0 * 1024.0)) /
                                     ((concurrent_total / iterations) / 1000.0);
            LOG_INFO(" Concurrent READ+WRITE:       " << std::fixed << std::setprecision(2)
                                                      << result.concurrent_gbps << " GB/s (effective)");
            result.used_overlap = true;

            cuMemFree(cuda_buf2);
            cuStreamDestroy(s1);
            cuStreamDestroy(s2);
        }

        LOG_INFO("----------------------------------------------------------------");
        LOG_INFO(" Write/Read asymmetry: " << std::fixed << std::setprecision(1)
                                           << (result.write_gbps / result.read_gbps) << "x");
        if (result.concurrent_gbps > 0)
        {
            double seq_time = (num_bytes / result.read_gbps + num_bytes / result.write_gbps);
            double conc_time = (2 * num_bytes / result.concurrent_gbps);
            double speedup = seq_time / conc_time;
            LOG_INFO(" Overlap speedup vs sequential: " << std::fixed << std::setprecision(1)
                                                        << ((speedup - 1) * 100) << "%");
        }
        LOG_INFO("================================================================");

        result.success = true;
        result.throughput_gbps = (result.read_gbps + result.write_gbps) / 2.0;

        cuMemFree(cuda_buf);
        return result;
    }

#endif // HAVE_CUDA && HAVE_ROCM

} // namespace llaminar2
