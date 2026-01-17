/**
 * @file DirectP2P_ROCm.cpp
 * @brief ROCm/HIP implementation of DMA-BUF export/import for direct P2P
 *
 * Uses HIP APIs for external memory handling:
 * - hipMemCreate (if available) for shareable allocations
 * - hipImportExternalMemory to import DMA-BUF from CUDA
 * - hipExternalMemoryGetMappedBuffer to get device pointer
 *
 * Note: ROCm's DMA-BUF support is more limited than CUDA's.
 * We may need to fall back to KFD (Kernel Fusion Driver) APIs.
 */

#include "DirectP2P_ROCm.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

// ROCm KFD headers for low-level access (if available)
#ifdef __has_include
#if __has_include(<hsakmt.h>)
#include <hsakmt.h>
#define HAVE_KFD_HEADERS 1
#endif
#endif

namespace llaminar2
{
    namespace rocm_direct_p2p
    {
        // Static version string buffer
        static char s_version_str[64] = {0};

        bool supportsDmaBufExport()
        {
            // ROCm 5.0+ has experimental DMA-BUF export
            int runtime_version = 0;
            hipRuntimeGetVersion(&runtime_version);

            // ROCm 5.0 = 50000000 in HIP versioning
            // Actually HIP version is like 50421134 for 5.4.2
            if (runtime_version < 50000000)
                return false;

            // Check device capabilities
            int device = 0;
            hipSetDevice(device);

            hipDeviceProp_t props;
            hipGetDeviceProperties(&props, device);

            // ROCm on most AMD GPUs should support this
            // GCN 3+ (Fiji, Polaris, Vega, RDNA)
            return true;  // Optimistic - will fail gracefully if not supported
        }

        bool supportsDmaBufImport()
        {
            // External memory import available since ROCm 4.0
            int runtime_version = 0;
            hipRuntimeGetVersion(&runtime_version);
            return runtime_version >= 40000000;
        }

        const char *getDriverVersion()
        {
            int runtime_version = 0;
            hipRuntimeGetVersion(&runtime_version);

            // HIP version format: MAJOR * 10000000 + MINOR * 100000 + PATCH
            int major = runtime_version / 10000000;
            int minor = (runtime_version % 10000000) / 100000;
            int patch = (runtime_version % 100000);

            snprintf(s_version_str, sizeof(s_version_str), "%d.%d.%d",
                     major, minor, patch);
            return s_version_str;
        }

        void *allocateExportable(int device_ordinal, size_t size, int *out_dmabuf_fd)
        {
            *out_dmabuf_fd = -1;

            hipSetDevice(device_ordinal);

            // ROCm doesn't have direct hipMemCreate equivalent to CUDA's cuMemCreate
            // We need to use the lower-level approach via /dev/kfd

            // First, try standard HIP allocation and see if we can export it
            void *ptr = nullptr;
            hipError_t err = hipMalloc(&ptr, size);
            if (err != hipSuccess)
            {
                fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
                return nullptr;
            }

            // ROCm 5.3+ has hipMemExportToShareableHandle (experimental)
            // For now, we'll return the pointer without export capability
            // The import side is more likely to work

            fprintf(stderr, "ROCm DMA-BUF export not fully implemented - returning regular allocation\n");
            fprintf(stderr, "Pointer: %p, Size: %zu\n", ptr, size);

            // On Linux with KFD, we could potentially:
            // 1. Open /dev/kfd
            // 2. Use AMDKFD_IOC_EXPORT_DMABUF ioctl
            // But this requires root and specific kernel version

            return ptr;
        }

        void freeExportable(int device_ordinal, void *ptr, int dmabuf_fd)
        {
            if (dmabuf_fd >= 0)
            {
                close(dmabuf_fd);
            }

            if (ptr)
            {
                hipSetDevice(device_ordinal);
                hipFree(ptr);
            }
        }

        void *importDmaBuf(int device_ordinal, int dmabuf_fd, size_t size)
        {
            fprintf(stderr, "\n=== ROCm importDmaBuf Debug ===");
            fprintf(stderr, "\n  device_ordinal: %d", device_ordinal);
            fprintf(stderr, "\n  dmabuf_fd: %d", dmabuf_fd);
            fprintf(stderr, "\n  size: %zu bytes (%.2f MB)\n", size, size / (1024.0 * 1024.0));

            if (dmabuf_fd < 0)
            {
                fprintf(stderr, "  ERROR: Invalid DMA-BUF fd\n");
                return nullptr;
            }

            hipError_t set_err = hipSetDevice(device_ordinal);
            if (set_err != hipSuccess)
            {
                fprintf(stderr, "  ERROR: hipSetDevice(%d) failed: %s\n",
                        device_ordinal, hipGetErrorString(set_err));
                return nullptr;
            }
            fprintf(stderr, "  hipSetDevice(%d): OK\n", device_ordinal);

            // Use hipImportExternalMemory for importing DMA-BUF from CUDA
            hipExternalMemoryHandleDesc ext_mem_desc = {};
            ext_mem_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
            ext_mem_desc.handle.fd = dmabuf_fd;
            ext_mem_desc.size = size;
            ext_mem_desc.flags = 0;

            fprintf(stderr, "  Attempting hipImportExternalMemory with:\n");
            fprintf(stderr, "    type: hipExternalMemoryHandleTypeOpaqueFd (%d)\n", (int)ext_mem_desc.type);
            fprintf(stderr, "    fd: %d\n", ext_mem_desc.handle.fd);
            fprintf(stderr, "    size: %zu\n", ext_mem_desc.size);
            fprintf(stderr, "    flags: %u\n", ext_mem_desc.flags);

            hipExternalMemory_t ext_mem = nullptr;
            hipError_t err = hipImportExternalMemory(&ext_mem, &ext_mem_desc);
            if (err != hipSuccess)
            {
                fprintf(stderr, "  FAILED: hipImportExternalMemory: %s (code %d)\n",
                        hipGetErrorString(err), (int)err);
                fprintf(stderr, "  Error codes: hipErrorOutOfMemory=2, hipErrorInvalidValue=1\n");
                fprintf(stderr, "  This often means:\n");
                fprintf(stderr, "    - IOMMU is blocking the cross-device mapping\n");
                fprintf(stderr, "    - Handle type incompatibility between CUDA and ROCm\n");
                fprintf(stderr, "    - Kernel doesn't support this DMA-BUF operation\n");
                return nullptr;
            }

            // Map the external memory as a buffer
            hipExternalMemoryBufferDesc buf_desc = {};
            buf_desc.offset = 0;
            buf_desc.size = size;
            buf_desc.flags = 0;

            void *mapped_ptr = nullptr;
            err = hipExternalMemoryGetMappedBuffer(&mapped_ptr, ext_mem, &buf_desc);
            if (err != hipSuccess)
            {
                fprintf(stderr, "hipExternalMemoryGetMappedBuffer failed: %s\n",
                        hipGetErrorString(err));
                hipDestroyExternalMemory(ext_mem);
                return nullptr;
            }

            fprintf(stderr, "Successfully imported DMA-BUF fd %d as ROCm ptr %p\n",
                    dmabuf_fd, mapped_ptr);

            return mapped_ptr;
        }

        void releaseImported(int device_ordinal, void *ptr)
        {
            // TODO: Track and destroy hipExternalMemory_t handle
            (void)device_ordinal;
            (void)ptr;
        }

        bool asyncCopy(int device_ordinal, void *dst, const void *src,
                       size_t size, void *stream)
        {
            hipSetDevice(device_ordinal);
            hipError_t err = hipMemcpyAsync(dst, src, size,
                                             hipMemcpyDefault,
                                             static_cast<hipStream_t>(stream));
            return err == hipSuccess;
        }

        void syncStream(void *stream)
        {
            hipStreamSynchronize(static_cast<hipStream_t>(stream));
        }

        void *createStream(int device_ordinal)
        {
            fprintf(stderr, "ROCm createStream: attempting device %d\n", device_ordinal);
            hipError_t err = hipSetDevice(device_ordinal);
            if (err != hipSuccess)
            {
                fprintf(stderr, "ROCm createStream: hipSetDevice(%d) FAILED: %s (code %d)\n",
                        device_ordinal, hipGetErrorString(err), (int)err);
                return nullptr;
            }
            fprintf(stderr, "ROCm createStream: hipSetDevice OK\n");
            
            hipStream_t stream;
            err = hipStreamCreate(&stream);
            if (err != hipSuccess)
            {
                fprintf(stderr, "ROCm createStream: hipStreamCreate FAILED: %s (code %d)\n",
                        hipGetErrorString(err), (int)err);
                return nullptr;
            }
            fprintf(stderr, "ROCm createStream: SUCCESS (stream=%p)\n", (void*)stream);
            return static_cast<void *>(stream);
        }

        void destroyStream(void *stream)
        {
            if (stream)
            {
                hipStreamDestroy(static_cast<hipStream_t>(stream));
            }
        }

    } // namespace rocm_direct_p2p
} // namespace llaminar2
