/**
 * @file MockProgressPublisher.h
 * @brief Mock IProgressPublisher for unit testing weight load progress without MPI
 *
 * This mock enables:
 * - Testing WeightLoadProgress rendering logic without MPI runtime
 * - Verifying publish calls are made with correct parameters
 * - Simulating aggregator behavior for multi-rank progress display
 */

#pragma once

#include "loaders/IProgressPublisher.h"
#include <mutex>
#include <string>
#include <vector>

namespace llaminar2::test
{

    /**
     * @brief Records all IProgressPublisher calls for verification.
     *
     * Thread-safe: all methods can be called from multiple threads.
     */
    class MockProgressPublisher : public IProgressPublisher
    {
    public:
        struct DeviceRegistration
        {
            std::string label;
            size_t total_bytes;
            int returned_idx;
        };

        struct ProgressUpdate
        {
            int local_device_idx;
            size_t bytes_loaded;
        };

        struct FinishEvent
        {
            int local_device_idx;
        };

        int publishDevice(const std::string &label, size_t total_bytes) override
        {
            std::lock_guard<std::mutex> lock(mu_);
            int idx = static_cast<int>(registrations_.size());
            registrations_.push_back({label, total_bytes, idx});
            return idx;
        }

        void publishProgress(int local_device_idx, size_t bytes_loaded) override
        {
            std::lock_guard<std::mutex> lock(mu_);
            progress_updates_.push_back({local_device_idx, bytes_loaded});
        }

        void publishFinished(int local_device_idx) override
        {
            std::lock_guard<std::mutex> lock(mu_);
            finish_events_.push_back({local_device_idx});
        }

        // --- Accessors for verification ---

        std::vector<DeviceRegistration> registrations() const
        {
            std::lock_guard<std::mutex> lock(mu_);
            return registrations_;
        }

        std::vector<ProgressUpdate> progressUpdates() const
        {
            std::lock_guard<std::mutex> lock(mu_);
            return progress_updates_;
        }

        std::vector<FinishEvent> finishEvents() const
        {
            std::lock_guard<std::mutex> lock(mu_);
            return finish_events_;
        }

        size_t registrationCount() const
        {
            std::lock_guard<std::mutex> lock(mu_);
            return registrations_.size();
        }

        size_t progressUpdateCount() const
        {
            std::lock_guard<std::mutex> lock(mu_);
            return progress_updates_.size();
        }

        size_t finishEventCount() const
        {
            std::lock_guard<std::mutex> lock(mu_);
            return finish_events_.size();
        }

        void reset()
        {
            std::lock_guard<std::mutex> lock(mu_);
            registrations_.clear();
            progress_updates_.clear();
            finish_events_.clear();
        }

    private:
        mutable std::mutex mu_;
        std::vector<DeviceRegistration> registrations_;
        std::vector<ProgressUpdate> progress_updates_;
        std::vector<FinishEvent> finish_events_;
    };

    /**
     * @brief Mock publisher that can simulate failure (returns -1 from publishDevice).
     */
    class FailingProgressPublisher : public IProgressPublisher
    {
    public:
        int publishDevice(const std::string & /*label*/, size_t /*total_bytes*/) override
        {
            return -1; // Simulate MAX_DEVICES exceeded
        }

        void publishProgress(int /*local_device_idx*/, size_t /*bytes_loaded*/) override {}
        void publishFinished(int /*local_device_idx*/) override {}
    };

} // namespace llaminar2::test
