#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <mpi.h>

#include "IProgressPublisher.h"

namespace llaminar2
{

    class WeightLoadProgress;

    /**
     * @brief MPI RMA-based aggregator that lets rank 0 display progress from all ranks.
     *
     * Architecture:
     *   ┌──────────────────────────────────────────────────────────────────┐
     *   │  MPI Window (one contiguous allocation per rank)                 │
     *   │                                                                  │
     *   │  Rank 0 slot: [dev0: bytes_loaded, total, finished, label]      │
     *   │               [dev1: bytes_loaded, total, finished, label]      │
     *   │               ...                                                │
     *   │  Rank 1 slot: [dev0: bytes_loaded, total, finished, label]      │
     *   │               [dev1: bytes_loaded, total, finished, label]      │
     *   │               ...                                                │
     *   └──────────────────────────────────────────────────────────────────┘
     *
     * Protocol:
     *   1. create() — collective (all ranks). Creates MPI window.
     *   2. publishDevice() — local write to window memory (no MPI call).
     *   3. publishProgress() — local write (no MPI call, zero overhead).
     *   4. Rank 0 poll thread — MPI_Get from remote windows every 50ms.
     *   5. finalize() — collective barrier, rank 0 renders final summary.
     *
     * Requirements:
     *   - MPI_THREAD_MULTIPLE (rank 0's poll thread calls MPI_Get)
     *   - MPI_Win_create (collective at construction)
     *
     * Rank != 0: Only writes to local window memory. No MPI calls during loading.
     * Rank == 0: Background thread polls remote windows.
     */
    class WeightLoadProgressAggregator : public IProgressPublisher
    {
    public:
        static constexpr int MAX_DEVICES_PER_RANK = 4;
        static constexpr int LABEL_LEN = 48;
        static constexpr int POLL_INTERVAL_MS = 50;

        /// Per-device progress slot within a rank's window.
        /// Layout must be trivially copyable (used with MPI_Get).
        struct alignas(8) DeviceSlot
        {
            uint64_t bytes_loaded; // Updated during loading
            uint64_t total_bytes;  // Set at registration
            uint32_t active;       // 1 = device registered and loading
            uint32_t finished;     // 1 = loading complete
            char label[LABEL_LEN]; // Null-terminated device label
        };
        static_assert(sizeof(DeviceSlot) == 72, "DeviceSlot must be 72 bytes for MPI transfer");

        /// Per-rank window region (contiguous array of device slots).
        struct RankWindow
        {
            DeviceSlot devices[MAX_DEVICES_PER_RANK];
        };
        static_assert(sizeof(RankWindow) == MAX_DEVICES_PER_RANK * sizeof(DeviceSlot));

        /// Create the aggregator. COLLECTIVE — all ranks must call together.
        /// @param comm   Communicator (typically MPI_COMM_WORLD or intra-node)
        /// @param rank   This process's rank
        /// @param world_size  Total ranks in comm
        /// @return shared_ptr to aggregator (or nullptr if MPI window creation fails)
        static std::shared_ptr<WeightLoadProgressAggregator>
        create(MPI_Comm comm, int rank, int world_size);

        ~WeightLoadProgressAggregator();

        // Non-copyable, non-movable (owns MPI window)
        WeightLoadProgressAggregator(const WeightLoadProgressAggregator &) = delete;
        WeightLoadProgressAggregator &operator=(const WeightLoadProgressAggregator &) = delete;

        /// Register a device on this rank. Returns local device index (0..MAX_DEVICES_PER_RANK-1).
        /// Simply writes to local window memory — no MPI call, no lock.
        /// @param label       Rank-qualified device label (e.g. "1:ROCm:0")
        /// @param total_bytes Total bytes this device will load
        /// @return local device index, or -1 if MAX_DEVICES_PER_RANK exceeded
        int publishDevice(const std::string &label, size_t total_bytes) override;

        /// Update progress for a local device. Just a memory write.
        void publishProgress(int local_device_idx, size_t bytes_loaded) override;

        /// Mark local device as finished. Just a memory write.
        void publishFinished(int local_device_idx) override;

        /// Start background polling thread (rank 0 only).
        /// The thread polls remote windows and updates the renderer.
        void startPolling(std::shared_ptr<WeightLoadProgress> renderer);

        /// Stop polling thread and perform final poll. Call before finalize().
        void stopPolling();

        /// Collective barrier — ensures all ranks are done loading.
        /// Rank 0 does one final poll after the barrier.
        void barrier();

        /// Release the MPI window. COLLECTIVE — all ranks must call together.
        /// Must be called while all ranks are still synchronized (e.g., after barrier()).
        /// After this call, the destructor will not attempt MPI_Win_free.
        void freeWindow();

        int rank() const { return rank_; }
        int worldSize() const { return world_size_; }

    private:
        WeightLoadProgressAggregator(MPI_Comm comm, int rank, int world_size,
                                     MPI_Win win, RankWindow *local_window);

        /// Rank 0: read all remote windows and update renderer.
        void pollOnce();

        /// Background poll loop (runs on rank 0's poll thread).
        void pollLoop();

        MPI_Comm comm_;
        int rank_;
        int world_size_;
        MPI_Win win_;
        RankWindow *local_window_; // Points to this rank's local MPI window memory

        // Rank 0 state
        std::shared_ptr<WeightLoadProgress> renderer_;
        std::vector<RankWindow> remote_buffers_;       // Scratch for MPI_Get results
        std::vector<std::vector<int>> remote_dev_idx_; // Maps (rank, local_dev) → renderer device idx
        std::atomic<bool> polling_{false};
        std::thread poll_thread_;
        int local_device_count_ = 0;
    };

} // namespace llaminar2
