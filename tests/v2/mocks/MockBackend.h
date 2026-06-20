/**
 * @file MockBackend.h
 * @brief Mock IBackend implementation with transfer tracking for testing
 * @author GitHub Copilot
 * @date January 2026
 *
 * Provides a mock backend that:
 * 1. Implements all IBackend interface methods
 * 2. Tracks all H2D (hostToDevice) and D2H (deviceToHost) calls
 * 3. Simulates device memory allocations with a simple memory pool
 * 4. Provides transfer statistics for verifying GPU-resident data flow
 *
 * Usage:
 *   auto backend = std::make_shared<MockBackend>();
 *   // ... run code that uses the backend ...
 *   auto stats = backend->getTransferStats();
 *   EXPECT_EQ(stats.h2d_count, expected_uploads);
 *   EXPECT_EQ(stats.d2h_count, expected_downloads);
 */

#pragma once

#include "backends/IBackend.h"
#include "utils/Logger.h"
#include <algorithm>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <vector>

namespace llaminar2
{
    namespace test
    {

        /**
         * @brief Record of a single transfer operation
         */
        struct TransferRecord
        {
            enum Type
            {
                H2D, ///< Host to Device
                D2H  ///< Device to Host
            };

            Type type;
            size_t bytes;
            void *dst;
            const void *src;
            int device_id;
        };

        /**
         * @brief Aggregated transfer statistics
         */
        struct TransferStats
        {
            size_t h2d_count = 0;                ///< Number of Host→Device transfers
            size_t d2h_count = 0;                ///< Number of Device→Host transfers
            size_t h2d_bytes = 0;                ///< Total bytes transferred Host→Device
            size_t d2h_bytes = 0;                ///< Total bytes transferred Device→Host
            size_t d2d_count = 0;                ///< Number of same-device Device→Device copies
            size_t d2d_bytes = 0;                ///< Total bytes transferred Device→Device
            std::vector<TransferRecord> records; ///< Detailed transfer log

            void clear()
            {
                h2d_count = 0;
                d2h_count = 0;
                h2d_bytes = 0;
                d2h_bytes = 0;
                d2d_count = 0;
                d2d_bytes = 0;
                records.clear();
            }
        };

        /**
         * @brief Mock IBackend implementation with transfer tracking
         *
         * Simulates GPU memory operations using host memory allocations.
         * Tracks all transfer operations for test verification.
         *
         * Thread Safety: All public methods are thread-safe.
         */
        class MockBackend : public IBackend
        {
        public:
            MockBackend() = default;
            explicit MockBackend(DeviceType device_type) : mock_device_type_(device_type) {}
            ~MockBackend() override
            {
                // Free all allocations
                std::lock_guard<std::mutex> lock(mutex_);
                for (auto &[ptr, info] : allocations_)
                {
                    std::free(ptr);
                }
                allocations_.clear();

                for (auto &[ptr, info] : mapped_allocations_)
                {
                    std::free(ptr);
                }
                mapped_allocations_.clear();
            }

            // =================================================================
            // IBackend Memory Transfer Operations
            // =================================================================

            bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr) override
            {
                std::lock_guard<std::mutex> lock(mutex_);

                // Record the transfer
                stats_.d2h_count++;
                stats_.d2h_bytes += bytes;
                stats_.records.push_back({TransferRecord::D2H, bytes, dst, src, device_id});

                LOG_DEBUG("[MockBackend] D2H transfer: " << bytes << " bytes from device "
                                                         << device_id << " (total D2H: " << stats_.d2h_count << ")");

                // Verify source is a valid device allocation
                if (!isValidDevicePtr(src))
                {
                    LOG_WARN("[MockBackend] D2H from untracked device pointer: " << src);
                }

                // Simulate the transfer (copy from our "device" memory to host)
                if (bytes > 0 && dst && src)
                {
                    std::memcpy(dst, src, bytes);
                }

                return true;
            }

            bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr) override
            {
                std::lock_guard<std::mutex> lock(mutex_);

                // Record the transfer
                stats_.h2d_count++;
                stats_.h2d_bytes += bytes;
                stats_.records.push_back({TransferRecord::H2D, bytes, dst, src, device_id});

                LOG_DEBUG("[MockBackend] H2D transfer: " << bytes << " bytes to device "
                                                         << device_id << " (total H2D: " << stats_.h2d_count << ")");

                // Verify destination is a valid device allocation
                if (!isValidDevicePtr(dst))
                {
                    LOG_WARN("[MockBackend] H2D to untracked device pointer: " << dst);
                }

                // Simulate the transfer (copy from host to our "device" memory)
                if (bytes > 0 && dst && src)
                {
                    std::memcpy(dst, src, bytes);
                }

                return true;
            }

            bool deviceToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr) override
            {
                (void)stream;
                std::lock_guard<std::mutex> lock(mutex_);

                // Record the same-device device-to-device copy
                stats_.d2d_count++;
                stats_.d2d_bytes += bytes;

                LOG_DEBUG("[MockBackend] D2D transfer: " << bytes << " bytes on device "
                                                         << device_id << " (total D2D: " << stats_.d2d_count << ")");

                // Simulate the copy (mock "device" memory is host memory)
                if (bytes > 0 && dst && src)
                {
                    std::memcpy(dst, src, bytes);
                }

                return true;
            }

            bool synchronize(int device_id) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                sync_count_++;
                (void)device_id;
                return true;
            }

            bool streamSynchronize(int device_id) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stream_sync_count_++;
                (void)device_id;
                return true;
            }

            // =================================================================
            // IBackend Event Operations (with tracking)
            // =================================================================

            /**
             * @brief Record of a single event operation for test verification
             */
            struct EventRecord
            {
                enum Type
                {
                    CREATE,
                    DESTROY,
                    RECORD,
                    WAIT
                };
                Type type;
                void *event;
                int device_id;
                void *stream; ///< Only valid for RECORD operations
            };

            void *createEvent(int device_id) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                void *event = reinterpret_cast<void *>(next_event_id_++);
                event_records_.push_back({EventRecord::CREATE, event, device_id, nullptr});
                return event;
            }

            void destroyEvent(void *event, int device_id) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                event_records_.push_back({EventRecord::DESTROY, event, device_id, nullptr});
            }

            bool recordEvent(void *event, int device_id, void *stream = nullptr) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                event_records_.push_back({EventRecord::RECORD, event, device_id, stream});
                return true;
            }

            bool waitForEvent(void *event, int device_id) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                event_records_.push_back({EventRecord::WAIT, event, device_id, nullptr});
                return true;
            }

            bool setDevice(int device_id) override
            {
                current_device_ = device_id;
                return true;
            }

            // =================================================================
            // IBackend Memory Allocation Operations
            // =================================================================

            void *allocate(size_t bytes, int device_id) override
            {
                std::lock_guard<std::mutex> lock(mutex_);

                // Allocate from host memory (simulating device memory)
                void *ptr = std::malloc(bytes);
                if (ptr)
                {
                    allocations_[ptr] = {bytes, device_id};
                    LOG_DEBUG("[MockBackend] Allocated " << bytes << " bytes on device "
                                                         << device_id << " at " << ptr);
                }

                return ptr;
            }

            void free(void *ptr, int device_id) override
            {
                if (!ptr)
                    return;

                std::lock_guard<std::mutex> lock(mutex_);

                auto it = allocations_.find(ptr);
                if (it != allocations_.end())
                {
                    LOG_DEBUG("[MockBackend] Freed " << it->second.bytes << " bytes on device "
                                                     << device_id << " at " << ptr);
                    allocations_.erase(it);
                    std::free(ptr);
                }
                else
                {
                    LOG_WARN("[MockBackend] Attempt to free untracked pointer: " << ptr);
                }
            }

            bool memset(void *ptr, int value, size_t bytes, int device_id, void *stream = nullptr) override
            {
                (void)device_id;
                if (ptr && bytes > 0)
                {
                    std::memset(ptr, value, bytes);
                }
                return true;
            }

            // =================================================================
            // IBackend Mapped Memory Operations
            // =================================================================

            void *allocateMapped(size_t bytes, int device_id, void **device_ptr) override
            {
                std::lock_guard<std::mutex> lock(mutex_);

                // For mock, host and device pointers are the same (simulating zero-copy)
                void *ptr = std::malloc(bytes);
                if (ptr)
                {
                    mapped_allocations_[ptr] = {bytes, device_id};
                    if (device_ptr)
                    {
                        *device_ptr = ptr; // Same pointer for mapped memory
                    }
                    LOG_DEBUG("[MockBackend] Allocated mapped " << bytes << " bytes on device "
                                                                << device_id << " at " << ptr);
                }
                else if (device_ptr)
                {
                    *device_ptr = nullptr;
                }

                return ptr;
            }

            void freeMapped(void *host_ptr, int device_id) override
            {
                if (!host_ptr)
                    return;

                std::lock_guard<std::mutex> lock(mutex_);

                auto it = mapped_allocations_.find(host_ptr);
                if (it != mapped_allocations_.end())
                {
                    LOG_DEBUG("[MockBackend] Freed mapped " << it->second.bytes << " bytes on device "
                                                            << device_id << " at " << host_ptr);
                    mapped_allocations_.erase(it);
                    std::free(host_ptr);
                }
                else
                {
                    LOG_WARN("[MockBackend] Attempt to free untracked mapped pointer: " << host_ptr);
                }
            }

            // =================================================================
            // IBackend Device Query Operations
            // =================================================================

            int deviceCount() const override
            {
                return num_devices_;
            }

            std::string backendName() const override
            {
                return "MockBackend";
            }

            std::string deviceName(int device_id) const override
            {
                return "MockDevice_" + std::to_string(device_id);
            }

            size_t deviceMemoryTotal(int device_id) const override
            {
                (void)device_id;
                return 16ULL * 1024 * 1024 * 1024; // 16 GB
            }

            size_t deviceMemoryFree(int device_id) const override
            {
                (void)device_id;
                return 12ULL * 1024 * 1024 * 1024; // 12 GB free
            }

            // =================================================================
            // IBackend Capability Queries
            // =================================================================

            bool supportsBF16(int device_id) const override
            {
                (void)device_id;
                return true;
            }

            bool supportsFP16(int device_id) const override
            {
                (void)device_id;
                return true;
            }

            bool supportsINT8(int device_id) const override
            {
                (void)device_id;
                return true;
            }

            // =================================================================
            // IBackend Compute Operations
            // =================================================================

            bool gemmIQ4NL(
                const void *A_device,
                const void *B_device,
                void *C_device,
                int m,
                int n,
                int k,
                int device_id) override
            {
                (void)A_device;
                (void)B_device;
                (void)C_device;
                (void)m;
                (void)n;
                (void)k;
                (void)device_id;
                // Mock: no actual GEMM computation
                return true;
            }

            DeviceType backendDeviceType() const override { return mock_device_type_; }

            /// Set the device type reported by backendDeviceType() (default: CPU)
            void setMockDeviceType(DeviceType type) { mock_device_type_ = type; }

            // =================================================================
            // IBackend Async Operations (trivial mock implementations)
            // =================================================================

            std::future<bool> deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id) override
            {
                std::promise<bool> promise;
                promise.set_value(deviceToHost(dst, src, bytes, device_id));
                return promise.get_future();
            }

            std::future<bool> hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id) override
            {
                std::promise<bool> promise;
                promise.set_value(hostToDevice(dst, src, bytes, device_id));
                return promise.get_future();
            }

            std::future<bool> synchronizeAsync(int device_id) override
            {
                std::promise<bool> promise;
                promise.set_value(synchronize(device_id));
                return promise.get_future();
            }

            std::future<void *> allocateAsync(size_t bytes, int device_id) override
            {
                std::promise<void *> promise;
                promise.set_value(allocate(bytes, device_id));
                return promise.get_future();
            }

            std::future<void> freeAsync(void *ptr, int device_id) override
            {
                free(ptr, device_id);
                std::promise<void> promise;
                promise.set_value();
                return promise.get_future();
            }

            std::future<bool> memsetAsync(void *ptr, int value, size_t bytes, int device_id) override
            {
                std::promise<bool> promise;
                promise.set_value(memset(ptr, value, bytes, device_id));
                return promise.get_future();
            }

            // =================================================================
            // Transfer Tracking API (Test-specific)
            // =================================================================

            /**
             * @brief Get current transfer statistics
             * @return Copy of current stats (thread-safe)
             */
            TransferStats getTransferStats() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return stats_;
            }

            /**
             * @brief Reset all transfer statistics
             */
            void resetTransferStats()
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stats_.clear();
            }

            /**
             * @brief Get count of H2D transfers since last reset
             */
            size_t getH2DCount() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return stats_.h2d_count;
            }

            /**
             * @brief Get count of D2H transfers since last reset
             */
            size_t getD2HCount() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return stats_.d2h_count;
            }

            /**
             * @brief Get count of synchronize() calls
             */
            size_t getSyncCount() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return sync_count_;
            }

            /**
             * @brief Get count of streamSynchronize() calls
             */
            size_t getStreamSyncCount() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return stream_sync_count_;
            }

            // =================================================================
            // Allocation Tracking API (Test-specific)
            // =================================================================

            /**
             * @brief Get number of active device allocations
             */
            size_t getAllocationCount() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return allocations_.size();
            }

            /**
             * @brief Get total bytes currently allocated on device
             */
            size_t getTotalAllocatedBytes() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                size_t total = 0;
                for (const auto &[ptr, info] : allocations_)
                {
                    total += info.bytes;
                }
                return total;
            }

            /**
             * @brief Check if a pointer is a tracked device allocation
             */
            bool isAllocated(void *ptr) const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return allocations_.find(ptr) != allocations_.end();
            }

            /**
             * @brief Check if a pointer is a tracked mapped allocation
             */
            bool isMappedAllocation(void *ptr) const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return mapped_allocations_.find(ptr) != mapped_allocations_.end();
            }

            // =================================================================
            // Test Configuration
            // =================================================================

            /**
             * @brief Set number of simulated devices
             */
            void setDeviceCount(int count)
            {
                num_devices_ = count;
            }

            // =================================================================
            // Event Tracking API (Test-specific)
            // =================================================================

            /**
             * @brief Get all event operation records
             */
            std::vector<EventRecord> getEventRecords() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return event_records_;
            }

            /**
             * @brief Get count of event create operations
             */
            size_t getEventCreateCount() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return std::count_if(event_records_.begin(), event_records_.end(),
                                     [](const EventRecord &r)
                                     { return r.type == EventRecord::CREATE; });
            }

            /**
             * @brief Get count of event record operations
             */
            size_t getEventRecordCount() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return std::count_if(event_records_.begin(), event_records_.end(),
                                     [](const EventRecord &r)
                                     { return r.type == EventRecord::RECORD; });
            }

            /**
             * @brief Get count of event wait operations
             */
            size_t getEventWaitCount() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return std::count_if(event_records_.begin(), event_records_.end(),
                                     [](const EventRecord &r)
                                     { return r.type == EventRecord::WAIT; });
            }

            /**
             * @brief Get event record operations that used a specific stream
             */
            std::vector<EventRecord> getEventRecordsForStream(void *stream) const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                std::vector<EventRecord> result;
                for (const auto &r : event_records_)
                {
                    if (r.type == EventRecord::RECORD && r.stream == stream)
                        result.push_back(r);
                }
                return result;
            }

            /**
             * @brief Reset all event tracking records
             */
            void resetEventRecords()
            {
                std::lock_guard<std::mutex> lock(mutex_);
                event_records_.clear();
            }

            /**
             * @brief Reset all tracking (transfers + events)
             */
            void resetAll()
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stats_.clear();
                event_records_.clear();
                sync_count_ = 0;
                stream_sync_count_ = 0;
            }

        private:
            struct AllocationInfo
            {
                size_t bytes;
                int device_id;
            };

            bool isValidDevicePtr(const void *ptr) const
            {
                // Check both regular and mapped allocations
                return allocations_.find(const_cast<void *>(ptr)) != allocations_.end() ||
                       mapped_allocations_.find(const_cast<void *>(ptr)) != mapped_allocations_.end();
            }

            mutable std::mutex mutex_;
            TransferStats stats_;
            size_t sync_count_ = 0;
            size_t stream_sync_count_ = 0;
            std::vector<EventRecord> event_records_;
            std::map<void *, AllocationInfo> allocations_;
            std::map<void *, AllocationInfo> mapped_allocations_;
            int num_devices_ = 1;
            int current_device_ = 0;
            uintptr_t next_event_id_ = 0x1000;
            DeviceType mock_device_type_ = DeviceType::CPU;
        };

    } // namespace test
} // namespace llaminar2
