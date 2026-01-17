/**
 * @file DirectP2P_ROCm.h
 * @brief ROCm-side DMA-BUF export/import for direct cross-vendor P2P
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    namespace rocm_direct_p2p
    {
        /**
         * @brief Check if this ROCm driver supports DMA-BUF export
         */
        bool supportsDmaBufExport();

        /**
         * @brief Check if this ROCm driver supports external memory import
         */
        bool supportsDmaBufImport();

        /**
         * @brief Get ROCm driver version string
         */
        const char *getDriverVersion();

        /**
         * @brief Allocate ROCm memory that can be exported as DMA-BUF
         *
         * Uses hipMemCreate with shareable handle support.
         *
         * @param device_ordinal ROCm device
         * @param size Allocation size
         * @param out_dmabuf_fd Output: DMA-BUF file descriptor
         * @return Device pointer or nullptr on failure
         */
        void *allocateExportable(int device_ordinal, size_t size, int *out_dmabuf_fd);

        /**
         * @brief Free exportable ROCm memory
         */
        void freeExportable(int device_ordinal, void *ptr, int dmabuf_fd);

        /**
         * @brief Import external DMA-BUF as ROCm memory
         *
         * @param device_ordinal ROCm device
         * @param dmabuf_fd DMA-BUF file descriptor from another GPU
         * @param size Buffer size
         * @return Device pointer accessible by ROCm or nullptr on failure
         */
        void *importDmaBuf(int device_ordinal, int dmabuf_fd, size_t size);

        /**
         * @brief Release imported external memory
         */
        void releaseImported(int device_ordinal, void *ptr);

        /**
         * @brief Async copy using HIP (for use after import)
         */
        bool asyncCopy(int device_ordinal, void *dst, const void *src,
                       size_t size, void *stream);

        /**
         * @brief Sync HIP stream
         */
        void syncStream(void *stream);

        /**
         * @brief Create HIP stream
         */
        void *createStream(int device_ordinal);

        /**
         * @brief Destroy HIP stream
         */
        void destroyStream(void *stream);

    } // namespace rocm_direct_p2p
} // namespace llaminar2
