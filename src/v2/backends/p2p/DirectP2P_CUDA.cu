/**
 * @file DirectP2P_CUDA.cu
 * @brief CUDA implementation of DMA-BUF export/import for direct P2P
 *
 * Uses CUDA Driver API for memory handle export/import:
 * - cuMemCreate with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
 * - cuMemExportToShareableHandle to get DMA-BUF fd
 * - cuMemImportFromShareableHandle to import external DMA-BUF
 */

#include "DirectP2P_CUDA.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>  // for close()

namespace llaminar2
{
    namespace cuda_direct_p2p
    {
        // Static version string buffer
        static char s_version_str[64] = {0};

        bool supportsDmaBufExport()
        {
            // Requires CUDA 11.4+ and driver support
            int driver_version = 0;
            cudaDriverGetVersion(&driver_version);

            // CUDA 11.4 = 11040
            if (driver_version < 11040)
                return false;

            // Check if the device supports memory handles
            int device = 0;
            cudaSetDevice(device);

            // Try to query handle type support
            int handle_type_supported = 0;
            cudaError_t err = cudaDeviceGetAttribute(
                &handle_type_supported,
                cudaDevAttrMemoryPoolSupportedHandleTypes,
                device);

            // If attribute exists and includes POSIX FD support
            return (err == cudaSuccess && (handle_type_supported & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        }

        bool supportsDmaBufImport()
        {
            // External memory import has been supported since CUDA 10
            int driver_version = 0;
            cudaDriverGetVersion(&driver_version);
            return driver_version >= 10000;
        }

        const char *getDriverVersion()
        {
            int driver_version = 0;
            cudaDriverGetVersion(&driver_version);
            snprintf(s_version_str, sizeof(s_version_str), "%d.%d",
                     driver_version / 1000, (driver_version % 1000) / 10);
            return s_version_str;
        }

        void *allocateExportable(int device_ordinal, size_t size, int *out_dmabuf_fd)
        {
            *out_dmabuf_fd = -1;

            cudaSetDevice(device_ordinal);

            // Try the modern cuMemCreate path first
            CUdevice cu_device;
            CUresult cu_err = cuDeviceGet(&cu_device, device_ordinal);
            if (cu_err != CUDA_SUCCESS)
            {
                fprintf(stderr, "cuDeviceGet failed: %d\n", cu_err);
                return nullptr;
            }

            // Check if device supports shareable handles
            int supports_posix_fd = 0;
            cu_err = cuDeviceGetAttribute(&supports_posix_fd,
                                          CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
                                          cu_device);
            if (cu_err != CUDA_SUCCESS || !supports_posix_fd)
            {
                fprintf(stderr, "Device doesn't support POSIX FD handles\n");
                return nullptr;
            }

            fprintf(stderr, "\n=== CUDA allocateExportable Debug ===\n");
            fprintf(stderr, "  device_ordinal: %d\n", device_ordinal);
            fprintf(stderr, "  requested size: %zu bytes (%.2f MB)\n", size, size / (1024.0 * 1024.0));
            fprintf(stderr, "  POSIX FD handles supported: YES\n");

            // Set up allocation properties for shareable memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device_ordinal;
            prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

            // Get allocation granularity
            size_t granularity = 0;
            cu_err = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            if (cu_err != CUDA_SUCCESS)
            {
                fprintf(stderr, "cuMemGetAllocationGranularity failed: %d\n", cu_err);
                return nullptr;
            }
            fprintf(stderr, "  allocation granularity: %zu bytes\n", granularity);

            // Round up size to granularity
            size_t aligned_size = ((size + granularity - 1) / granularity) * granularity;
            fprintf(stderr, "  aligned size: %zu bytes (%.2f MB)\n", aligned_size, aligned_size / (1024.0 * 1024.0));

            // Create the allocation
            CUmemGenericAllocationHandle alloc_handle;
            cu_err = cuMemCreate(&alloc_handle, aligned_size, &prop, 0);
            if (cu_err != CUDA_SUCCESS)
            {
                fprintf(stderr, "cuMemCreate failed: %d\n", cu_err);
                return nullptr;
            }
            fprintf(stderr, "  cuMemCreate: OK (handle=%llu)\n", (unsigned long long)alloc_handle);

            // Reserve virtual address range
            CUdeviceptr dptr = 0;
            cu_err = cuMemAddressReserve(&dptr, aligned_size, 0, 0, 0);
            if (cu_err != CUDA_SUCCESS)
            {
                fprintf(stderr, "cuMemAddressReserve failed: %d\n", cu_err);
                cuMemRelease(alloc_handle);
                return nullptr;
            }
            fprintf(stderr, "  cuMemAddressReserve: OK (dptr=%p)\n", (void*)dptr);

            // Map allocation to virtual address
            cu_err = cuMemMap(dptr, aligned_size, 0, alloc_handle, 0);
            if (cu_err != CUDA_SUCCESS)
            {
                fprintf(stderr, "cuMemMap failed: %d\n", cu_err);
                cuMemAddressFree(dptr, aligned_size);
                cuMemRelease(alloc_handle);
                return nullptr;
            }

            // Set access permissions
            CUmemAccessDesc access_desc = {};
            access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id = device_ordinal;
            access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            cu_err = cuMemSetAccess(dptr, aligned_size, &access_desc, 1);
            if (cu_err != CUDA_SUCCESS)
            {
                fprintf(stderr, "cuMemSetAccess failed: %d\n", cu_err);
                cuMemUnmap(dptr, aligned_size);
                cuMemAddressFree(dptr, aligned_size);
                cuMemRelease(alloc_handle);
                return nullptr;
            }
            fprintf(stderr, "  cuMemSetAccess: OK\n");

            // Export as DMA-BUF file descriptor
            int fd = -1;
            cu_err = cuMemExportToShareableHandle(&fd, alloc_handle,
                                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
            if (cu_err != CUDA_SUCCESS)
            {
                fprintf(stderr, "  cuMemExportToShareableHandle FAILED: %d\n", cu_err);
                cuMemUnmap(dptr, aligned_size);
                cuMemAddressFree(dptr, aligned_size);
                cuMemRelease(alloc_handle);
                return nullptr;
            }

            fprintf(stderr, "  cuMemExportToShareableHandle: OK (fd=%d)\n", fd);
            fprintf(stderr, "  SUCCESS: CUDA exportable buffer ready\\n\");\n");
            fprintf(stderr, "    device_ptr: %p\n", (void*)dptr);
            fprintf(stderr, "    dmabuf_fd: %d\n", fd);
            fprintf(stderr, "    size: %zu bytes\n", aligned_size);

            *out_dmabuf_fd = fd;
            return reinterpret_cast<void *>(dptr);
            return reinterpret_cast<void *>(dptr);
        }

        void freeExportable(int device_ordinal, void *ptr, int dmabuf_fd)
        {
            if (dmabuf_fd >= 0)
            {
                close(dmabuf_fd);
            }

            if (ptr)
            {
                cudaSetDevice(device_ordinal);
                CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);

                // We'd need to track the size to properly unmap...
                // For now, just release (will leak virtual address range)
                // TODO: Track allocation metadata properly
                cuMemUnmap(dptr, 0);  // Size 0 = unmap all
            }
        }

        void *importDmaBuf(int device_ordinal, int dmabuf_fd, size_t size)
        {
            fprintf(stderr, "\n=== CUDA importDmaBuf Debug ===");
            fprintf(stderr, "\n  device_ordinal: %d", device_ordinal);
            fprintf(stderr, "\n  dmabuf_fd: %d", dmabuf_fd);
            fprintf(stderr, "\n  size: %zu bytes (%.2f MB)\n", size, size / (1024.0 * 1024.0));

            if (dmabuf_fd < 0)
            {
                fprintf(stderr, "  ERROR: Invalid DMA-BUF fd\n");
                return nullptr;
            }

            cudaError_t set_err = cudaSetDevice(device_ordinal);
            if (set_err != cudaSuccess)
            {
                fprintf(stderr, "  ERROR: cudaSetDevice(%d) failed: %s\n",
                        device_ordinal, cudaGetErrorString(set_err));
                return nullptr;
            }
            fprintf(stderr, "  cudaSetDevice(%d): OK\n", device_ordinal);

            // Use cudaImportExternalMemory for importing DMA-BUF
            cudaExternalMemoryHandleDesc ext_mem_desc = {};
            ext_mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
            ext_mem_desc.handle.fd = dmabuf_fd;
            ext_mem_desc.size = size;
            ext_mem_desc.flags = 0;

            fprintf(stderr, "  Attempting cudaImportExternalMemory with:\n");
            fprintf(stderr, "    type: cudaExternalMemoryHandleTypeOpaqueFd (%d)\n", (int)ext_mem_desc.type);
            fprintf(stderr, "    fd: %d\n", ext_mem_desc.handle.fd);
            fprintf(stderr, "    size: %zu\n", ext_mem_desc.size);
            fprintf(stderr, "    flags: %u\n", ext_mem_desc.flags);

            cudaExternalMemory_t ext_mem = nullptr;
            cudaError_t err = cudaImportExternalMemory(&ext_mem, &ext_mem_desc);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "  FAILED: cudaImportExternalMemory: %s (code %d)\n",
                        cudaGetErrorString(err), (int)err);
                return nullptr;
            }

            // Map the external memory as a buffer
            cudaExternalMemoryBufferDesc buf_desc = {};
            buf_desc.offset = 0;
            buf_desc.size = size;
            buf_desc.flags = 0;

            void *mapped_ptr = nullptr;
            err = cudaExternalMemoryGetMappedBuffer(&mapped_ptr, ext_mem, &buf_desc);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "cudaExternalMemoryGetMappedBuffer failed: %s\n",
                        cudaGetErrorString(err));
                cudaDestroyExternalMemory(ext_mem);
                return nullptr;
            }

            // Note: We should track ext_mem handle for cleanup...
            // For now, return the mapped pointer
            return mapped_ptr;
        }

        void releaseImported(int device_ordinal, void *ptr)
        {
            // TODO: Need to track and destroy the cudaExternalMemory_t handle
            (void)device_ordinal;
            (void)ptr;
        }

        bool asyncCopy(int device_ordinal, void *dst, const void *src,
                       size_t size, void *stream)
        {
            cudaSetDevice(device_ordinal);
            cudaError_t err = cudaMemcpyAsync(dst, src, size,
                                               cudaMemcpyDefault,
                                               static_cast<cudaStream_t>(stream));
            return err == cudaSuccess;
        }

        void syncStream(void *stream)
        {
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        }

        void *createStream(int device_ordinal)
        {
            cudaSetDevice(device_ordinal);
            cudaStream_t stream;
            if (cudaStreamCreate(&stream) != cudaSuccess)
                return nullptr;
            return static_cast<void *>(stream);
        }

        void destroyStream(void *stream)
        {
            if (stream)
            {
                cudaStreamDestroy(static_cast<cudaStream_t>(stream));
            }
        }

    } // namespace cuda_direct_p2p
} // namespace llaminar2
