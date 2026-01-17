/**
 * @file DirectP2P_CUDA.h
 * @brief CUDA-side DMA-BUF export/import for direct cross-vendor P2P
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    namespace cuda_direct_p2p
    {
        /**
         * @brief Check if this CUDA driver supports DMA-BUF export
         */
        bool supportsDmaBufExport();

        /**
         * @brief Check if this CUDA driver supports external memory import
         */
        bool supportsDmaBufImport();

        /**
         * @brief Get CUDA driver version string
         */
        const char *getDriverVersion();

        /**
         * @brief Allocate CUDA memory that can be exported as DMA-BUF
         *
         * Uses cuMemCreate with shareable handle support.
         *
         * @param device_ordinal CUDA device
         * @param size Allocation size
         * @param out_dmabuf_fd Output: DMA-BUF file descriptor
         * @return Device pointer or nullptr on failure
         */
        void *allocateExportable(int device_ordinal, size_t size, int *out_dmabuf_fd);

        /**
         * @brief Free exportable CUDA memory
         */
        void freeExportable(int device_ordinal, void *ptr, int dmabuf_fd);

        /**
         * @brief Import external DMA-BUF as CUDA memory
         *
         * @param device_ordinal CUDA device
         * @param dmabuf_fd DMA-BUF file descriptor from another GPU
         * @param size Buffer size
         * @return Device pointer accessible by CUDA or nullptr on failure
         */
        void *importDmaBuf(int device_ordinal, int dmabuf_fd, size_t size);

        /**
         * @brief Release imported external memory
         */
        void releaseImported(int device_ordinal, void *ptr);

        /**
         * @brief Async copy using CUDA (for use after import)
         */
        bool asyncCopy(int device_ordinal, void *dst, const void *src,
                       size_t size, void *stream);

        /**
         * @brief Sync CUDA stream
         */
        void syncStream(void *stream);

        /**
         * @brief Create CUDA stream
         */
        void *createStream(int device_ordinal);

        /**
         * @brief Destroy CUDA stream
         */
        void destroyStream(void *stream);

    } // namespace cuda_direct_p2p
} // namespace llaminar2
