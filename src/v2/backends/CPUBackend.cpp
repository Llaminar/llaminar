/**
 * @file CPUBackend.cpp
 * @brief CPU/NUMA backend implementation
 *
 * Implements IBackend for CPU execution with NUMA-aware memory allocation.
 * Reads memory info from /sys/devices/system/node/nodeN/meminfo for NUMA nodes
 * or /proc/meminfo as fallback.
 *
 * @author David Sanftenberg
 */

#include "CPUBackend.h"
#include "../utils/Logger.h"

#include <cerrno>
#include <cstring> // memcpy, memset
#include <cstdlib> // aligned_alloc, free
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>

#include <numa.h>
#include <numaif.h>

#include <omp.h>

namespace llaminar2
{

    // ====================================================================
    // Constructor / Destructor
    // ====================================================================

    CPUBackend::CPUBackend(int local_numa_node)
        : local_numa_node_(local_numa_node)
    {
        // Validate NUMA node if specified
        if (local_numa_node_ >= 0)
        {
            if (numa_available() == -1)
            {
                throw std::runtime_error("[CPUBackend] libnuma reports NUMA unavailable; NUMA binding is required");
            }
            else if (local_numa_node_ >= numa_max_node() + 1)
            {
                throw std::runtime_error("[CPUBackend] NUMA node " + std::to_string(local_numa_node_) +
                                         " exceeds max " + std::to_string(numa_max_node()));
            }
            // Check if /sys/devices/system/node/nodeN exists
            std::string path = "/sys/devices/system/node/node" + std::to_string(local_numa_node_);
            std::ifstream test(path);
            if (!test.good())
            {
                throw std::runtime_error("[CPUBackend] NUMA node sysfs path not found: " + path);
            }
        }

        LOG_DEBUG("[CPUBackend] Initialized for NUMA node " << local_numa_node_
                                                            << " (total: " << (deviceMemoryTotal(0) / (1024 * 1024)) << " MB)");
    }

    CPUBackend::~CPUBackend()
    {
        // No cleanup needed - memory is freed by caller via free()
    }

    // ====================================================================
    // Device Enumeration (rank-local: always returns 1)
    // ====================================================================

    int CPUBackend::deviceCount() const
    {
        return 1; // Rank-local view - each rank sees 1 "device"
    }

    std::string CPUBackend::backendName() const
    {
        return "CPU";
    }

    std::string CPUBackend::deviceName(int device_id) const
    {
        if (!isValidDeviceId(device_id))
        {
            return "Invalid Device";
        }

        std::ostringstream oss;
        oss << "CPU:NUMA" << (local_numa_node_ >= 0 ? std::to_string(local_numa_node_) : "ALL");
        return oss.str();
    }

    // ====================================================================
    // Memory Query
    // ====================================================================

    size_t CPUBackend::deviceMemoryTotal(int device_id) const
    {
        if (!isValidDeviceId(device_id))
        {
            return 0;
        }

        if (local_numa_node_ >= 0)
        {
            size_t numa_total = readNumaMemTotal();
            if (numa_total > 0)
            {
                return numa_total;
            }
        }

        return readSystemMemTotal();
    }

    size_t CPUBackend::deviceMemoryFree(int device_id) const
    {
        if (!isValidDeviceId(device_id))
        {
            return 0;
        }

        if (local_numa_node_ >= 0)
        {
            size_t numa_free = readNumaMemFree();
            if (numa_free > 0)
            {
                return numa_free;
            }
        }

        return readSystemMemFree();
    }

    // ====================================================================
    // Memory Operations
    // ====================================================================

    void *CPUBackend::allocate(size_t bytes, int device_id)
    {
        if (!isValidDeviceId(device_id))
        {
            LOG_ERROR("[CPUBackend] Invalid device ID " << device_id << " (must be 0)");
            return nullptr;
        }

        if (bytes == 0)
        {
            return nullptr;
        }

        void *ptr = nullptr;

        // Use page alignment when installing NUMA memory policy; otherwise
        // 64-byte alignment is enough for cache line / AVX-512 alignment.
        // NUMA placement is installed with mbind() before first-touch so the
        // pointer remains compatible with std::free().
        const size_t alignment = local_numa_node_ >= 0 ? static_cast<size_t>(4096) : static_cast<size_t>(64);
        size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);

        ptr = std::aligned_alloc(alignment, aligned_bytes);
        if (!ptr)
        {
            LOG_ERROR("[CPUBackend] aligned_alloc failed for " << bytes << " bytes");
            return nullptr;
        }

        if (local_numa_node_ >= 0)
        {
            struct bitmask *nodemask = numa_allocate_nodemask();
            if (!nodemask)
            {
                LOG_ERROR("[CPUBackend] Failed to allocate NUMA nodemask");
                std::free(ptr);
                return nullptr;
            }

            numa_bitmask_clearall(nodemask);
            numa_bitmask_setbit(nodemask, local_numa_node_);
            errno = 0;
            int rc = mbind(ptr, aligned_bytes, MPOL_BIND, nodemask->maskp, nodemask->size, 0);
            const int bind_errno = errno;
            numa_free_nodemask(nodemask);

            if (rc != 0)
            {
                LOG_ERROR("[CPUBackend] mbind(" << ptr << ", " << aligned_bytes
                                                << ", node=" << local_numa_node_
                                                << ") failed: errno=" << bind_errno
                                                << " (" << std::strerror(bind_errno) << ")");
                std::free(ptr);
                return nullptr;
            }
        }

        // NUMA first-touch: parallel initialization to bind memory to local node
        // This ensures the memory is allocated on the socket where threads touch it first
        if (bytes >= 128 * 1024) // Only parallel init for >= 128KB
        {
            char *data = static_cast<char *>(ptr);
            const size_t page_size = 4096;
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < bytes; i += page_size)
            {
                data[i] = 0; // Touch first byte of each page
            }
        }
        else
        {
            std::memset(ptr, 0, bytes);
        }

        LOG_TRACE("[CPUBackend] aligned_alloc(" << bytes << ") = " << ptr);
        return ptr;
    }

    void CPUBackend::free(void *ptr, int device_id)
    {
        if (ptr == nullptr)
        {
            return; // Freeing nullptr is a no-op
        }

        if (!isValidDeviceId(device_id))
        {
            LOG_ERROR("[CPUBackend] Invalid device ID " << device_id << " for free");
            return;
        }

        std::free(ptr);
    }

    bool CPUBackend::memset(void *ptr, int value, size_t bytes, int device_id, void *stream)
    {
        if (ptr == nullptr || bytes == 0)
        {
            return true; // No-op for null pointer or zero bytes
        }

        if (!isValidDeviceId(device_id))
        {
            LOG_ERROR("[CPUBackend] Invalid device ID " << device_id << " for memset");
            return false;
        }

        std::memset(ptr, value, bytes);
        return true;
    }

    void *CPUBackend::allocateMapped(size_t bytes, int device_id, void **device_ptr)
    {
        // On CPU, "mapped" memory is just regular memory
        void *ptr = allocate(bytes, device_id);
        if (device_ptr)
        {
            *device_ptr = ptr; // Same pointer, no GPU
        }
        return ptr;
    }

    void CPUBackend::freeMapped(void *host_ptr, int device_id)
    {
        // On CPU, just use regular free
        free(host_ptr, device_id);
    }

    // ====================================================================
    // Transfer Operations (memcpy for CPU)
    // ====================================================================

    bool CPUBackend::deviceToHost(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (!isValidDeviceId(device_id))
        {
            return false;
        }

        if (dst == nullptr || src == nullptr || bytes == 0)
        {
            return (bytes == 0); // Empty copy is success
        }

        std::memcpy(dst, src, bytes);
        return true;
    }

    bool CPUBackend::hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (!isValidDeviceId(device_id))
        {
            return false;
        }

        if (dst == nullptr || src == nullptr || bytes == 0)
        {
            return (bytes == 0); // Empty copy is success
        }

        std::memcpy(dst, src, bytes);
        return true;
    }

    bool CPUBackend::synchronize(int device_id)
    {
        if (!isValidDeviceId(device_id))
        {
            return false;
        }

        // CPU execution is always synchronous - no-op
        return true;
    }

    bool CPUBackend::streamSynchronize(int device_id)
    {
        if (!isValidDeviceId(device_id))
        {
            return false;
        }

        // CPU execution is always synchronous - no-op
        return true;
    }

    bool CPUBackend::setDevice(int device_id)
    {
        if (!isValidDeviceId(device_id))
        {
            LOG_ERROR("[CPUBackend] setDevice(" << device_id << ") failed - only device 0 exists");
            return false;
        }

        // No-op for CPU - single "device"
        return true;
    }

    // ====================================================================
    // Event Operations (no-op for CPU - always synchronous)
    // ====================================================================

    void *CPUBackend::createEvent(int device_id)
    {
        if (!isValidDeviceId(device_id))
        {
            return nullptr;
        }

        // Return a dummy non-null pointer
        // CPU execution is always synchronous, so events are no-ops
        static int dummy_event = 1;
        return reinterpret_cast<void *>(&dummy_event);
    }

    void *CPUBackend::createTimingEvent(int device_id)
    {
        return createEvent(device_id);
    }

    void CPUBackend::destroyEvent(void *event, int device_id)
    {
        // No-op for CPU - nothing to destroy
        (void)event;
        (void)device_id;
    }

    bool CPUBackend::recordEvent(void *event, int device_id, void *stream)
    {
        if (!event || !isValidDeviceId(device_id))
        {
            return false;
        }

        (void)stream; // CPU has no streams
        // No-op for CPU - always synchronous
        return true;
    }

    bool CPUBackend::waitForEvent(void *event, int device_id)
    {
        if (!event || !isValidDeviceId(device_id))
        {
            return false;
        }

        // No-op for CPU - always synchronous
        return true;
    }

    bool CPUBackend::eventElapsedTimeMs(
        void *start_event,
        void *stop_event,
        int device_id,
        float *out_ms)
    {
        if (!start_event || !stop_event || !out_ms ||
            !isValidDeviceId(device_id))
        {
            return false;
        }

        *out_ms = 0.0f;
        return true;
    }

    // ====================================================================
    // Capability Queries
    // ====================================================================

    bool CPUBackend::supportsBF16(int device_id) const
    {
        if (!isValidDeviceId(device_id))
        {
            return false;
        }

        // CPU supports BF16 via AVX-512 BF16 or software emulation
        // Conservative: always return true since we can emulate
        return true;
    }

    bool CPUBackend::supportsFP16(int device_id) const
    {
        if (!isValidDeviceId(device_id))
        {
            return false;
        }

        // CPU supports FP16 via F16C extension or software emulation
        return true;
    }

    bool CPUBackend::supportsINT8(int device_id) const
    {
        if (!isValidDeviceId(device_id))
        {
            return false;
        }

        // CPU supports INT8 via VNNI or scalar operations
        return true;
    }

    // ====================================================================
    // Compute Operations
    // ====================================================================

    bool CPUBackend::gemmIQ4NL(
        const void *A_device,
        const void *B_device,
        void *C_device,
        int m,
        int n,
        int k,
        int device_id)
    {
        // CPU kernels should be called directly, not through the backend interface
        (void)A_device;
        (void)B_device;
        (void)C_device;
        (void)m;
        (void)n;
        (void)k;
        (void)device_id;

        LOG_WARN("[CPUBackend] gemmIQ4NL not implemented - use CPU kernel directly");
        return false;
    }

    // ====================================================================
    // Private Helper Methods
    // ====================================================================

    size_t CPUBackend::readNumaMemTotal() const
    {
        if (local_numa_node_ < 0)
        {
            return 0;
        }

        std::string path = "/sys/devices/system/node/node" + std::to_string(local_numa_node_) + "/meminfo";
        std::ifstream file(path);
        if (!file.is_open())
        {
            return 0;
        }

        std::string line;
        while (std::getline(file, line))
        {
            // Format: "Node 0 MemTotal:       12345678 kB"
            if (line.find("MemTotal:") != std::string::npos)
            {
                size_t value_kb = 0;
                // Parse the value (skip "Node N MemTotal:" prefix)
                size_t pos = line.find("MemTotal:");
                if (pos != std::string::npos)
                {
                    std::istringstream iss(line.substr(pos + 9)); // Skip "MemTotal:"
                    iss >> value_kb;
                    return value_kb * 1024; // Convert KB to bytes
                }
            }
        }

        return 0;
    }

    size_t CPUBackend::readNumaMemFree() const
    {
        if (local_numa_node_ < 0)
        {
            return 0;
        }

        std::string path = "/sys/devices/system/node/node" + std::to_string(local_numa_node_) + "/meminfo";
        std::ifstream file(path);
        if (!file.is_open())
        {
            return 0;
        }

        std::string line;
        while (std::getline(file, line))
        {
            // Format: "Node 0 MemFree:        12345678 kB"
            if (line.find("MemFree:") != std::string::npos)
            {
                size_t value_kb = 0;
                size_t pos = line.find("MemFree:");
                if (pos != std::string::npos)
                {
                    std::istringstream iss(line.substr(pos + 8)); // Skip "MemFree:"
                    iss >> value_kb;
                    return value_kb * 1024; // Convert KB to bytes
                }
            }
        }

        return 0;
    }

    size_t CPUBackend::readSystemMemTotal() const
    {
        std::ifstream file("/proc/meminfo");
        if (!file.is_open())
        {
            return 0;
        }

        std::string line;
        while (std::getline(file, line))
        {
            // Format: "MemTotal:       12345678 kB"
            if (line.find("MemTotal:") == 0)
            {
                size_t value_kb = 0;
                std::istringstream iss(line.substr(9)); // Skip "MemTotal:"
                iss >> value_kb;
                return value_kb * 1024; // Convert KB to bytes
            }
        }

        return 0;
    }

    size_t CPUBackend::readSystemMemFree() const
    {
        std::ifstream file("/proc/meminfo");
        if (!file.is_open())
        {
            return 0;
        }

        std::string line;
        while (std::getline(file, line))
        {
            // Format: "MemFree:        12345678 kB"
            if (line.find("MemFree:") == 0)
            {
                size_t value_kb = 0;
                std::istringstream iss(line.substr(8)); // Skip "MemFree:"
                iss >> value_kb;
                return value_kb * 1024; // Convert KB to bytes
            }
        }

        return 0;
    }

    bool CPUBackend::isValidDeviceId(int device_id) const
    {
        // Rank-local view: only device 0 is valid
        return device_id == 0;
    }

    // ====================================================================
    // Async Operations (Trivial for CPU - immediate completion)
    // ====================================================================

    std::future<bool> CPUBackend::deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id)
    {
        std::promise<bool> p;
        p.set_value(deviceToHost(dst, src, bytes, device_id));
        return p.get_future();
    }

    std::future<bool> CPUBackend::hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id)
    {
        std::promise<bool> p;
        p.set_value(hostToDevice(dst, src, bytes, device_id));
        return p.get_future();
    }

    std::future<bool> CPUBackend::synchronizeAsync(int device_id)
    {
        std::promise<bool> p;
        p.set_value(synchronize(device_id));
        return p.get_future();
    }

    std::future<void *> CPUBackend::allocateAsync(size_t bytes, int device_id)
    {
        std::promise<void *> p;
        p.set_value(allocate(bytes, device_id));
        return p.get_future();
    }

    std::future<void> CPUBackend::freeAsync(void *ptr, int device_id)
    {
        free(ptr, device_id);
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }

    std::future<bool> CPUBackend::memsetAsync(void *ptr, int value, size_t bytes, int device_id)
    {
        std::promise<bool> p;
        p.set_value(memset(ptr, value, bytes, device_id));
        return p.get_future();
    }

} // namespace llaminar2
