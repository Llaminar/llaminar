#include "WeightLoadProgressAggregator.h"
#include "WeightLoadProgress.h"
#include "../utils/Logger.h"

#include <algorithm>
#include <cstring>

namespace llaminar2
{

    // =============================================================================
    // Factory (collective)
    // =============================================================================

    std::shared_ptr<WeightLoadProgressAggregator>
    WeightLoadProgressAggregator::create(MPI_Comm comm, int rank, int world_size)
    {
        if (comm == MPI_COMM_NULL || world_size <= 1)
            return nullptr; // No aggregation needed for single-rank

        // Allocate window memory — each rank exposes one RankWindow
        RankWindow *local_mem = nullptr;
        MPI_Win win = MPI_WIN_NULL;

        int rc = MPI_Win_allocate(
            static_cast<MPI_Aint>(sizeof(RankWindow)),
            /*disp_unit=*/1,
            MPI_INFO_NULL,
            comm,
            &local_mem,
            &win);

        if (rc != MPI_SUCCESS || win == MPI_WIN_NULL)
        {
            LOG_WARN("[ProgressAggregator] MPI_Win_allocate failed (rc=" << rc << "); "
                                                                                  "falling back to rank-0-only progress display");
            return nullptr;
        }

        // Zero-initialize local window
        std::memset(local_mem, 0, sizeof(RankWindow));

        // Open passive-target epoch (remains open for the lifetime of the aggregator)
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

        return std::shared_ptr<WeightLoadProgressAggregator>(
            new WeightLoadProgressAggregator(comm, rank, world_size, win, local_mem));
    }

    // =============================================================================
    // Constructor / Destructor
    // =============================================================================

    WeightLoadProgressAggregator::WeightLoadProgressAggregator(
        MPI_Comm comm, int rank, int world_size,
        MPI_Win win, RankWindow *local_window)
        : comm_(comm), rank_(rank), world_size_(world_size),
          win_(win), local_window_(local_window)
    {
        if (rank == 0)
        {
            remote_buffers_.resize(world_size);
            remote_dev_idx_.resize(world_size, std::vector<int>(MAX_DEVICES_PER_RANK, -1));
            std::memset(remote_buffers_.data(), 0, world_size * sizeof(RankWindow));
        }
    }

    WeightLoadProgressAggregator::~WeightLoadProgressAggregator()
    {
        stopPolling();

        // If freeWindow() was not called explicitly, the window is leaked.
        // This is intentional — MPI_Win_free is collective and cannot be safely
        // called from a destructor (ranks may destroy at different times).
        if (win_ != MPI_WIN_NULL)
        {
            LOG_WARN("[ProgressAggregator] MPI window not freed before destructor — "
                     "call freeWindow() collectively before dropping the last reference");
        }
    }

    void WeightLoadProgressAggregator::freeWindow()
    {
        if (win_ != MPI_WIN_NULL)
        {
            MPI_Win_unlock_all(win_);
            MPI_Win_free(&win_);
            win_ = MPI_WIN_NULL;
            local_window_ = nullptr;
        }
    }

    // =============================================================================
    // Local writes (all ranks — no MPI calls)
    // =============================================================================

    int WeightLoadProgressAggregator::publishDevice(const std::string &label, size_t total_bytes)
    {
        if (local_device_count_ >= MAX_DEVICES_PER_RANK)
        {
            LOG_WARN("[ProgressAggregator] Rank " << rank_ << " exceeded MAX_DEVICES_PER_RANK ("
                                                  << MAX_DEVICES_PER_RANK << "); device '" << label << "' will not be tracked remotely");
            return -1;
        }

        int idx = local_device_count_++;
        DeviceSlot &slot = local_window_->devices[idx];
        slot.total_bytes = total_bytes;
        slot.bytes_loaded = 0;
        slot.finished = 0;

        // Copy label (truncate if needed)
        std::strncpy(slot.label, label.c_str(), LABEL_LEN - 1);
        slot.label[LABEL_LEN - 1] = '\0';

        // Activate slot LAST (acts as a publish fence for rank 0's poll)
        std::atomic_thread_fence(std::memory_order_release);
        slot.active = 1;

        // Flush local window so remote MPI_Get sees the update
        MPI_Win_flush_local(rank_, win_);

        return idx;
    }

    void WeightLoadProgressAggregator::publishProgress(int local_device_idx, size_t bytes_loaded)
    {
        if (local_device_idx < 0 || local_device_idx >= local_device_count_)
            return;
        local_window_->devices[local_device_idx].bytes_loaded = bytes_loaded;
        // No flush needed — passive target mode allows eventual visibility
    }

    void WeightLoadProgressAggregator::publishFinished(int local_device_idx)
    {
        if (local_device_idx < 0 || local_device_idx >= local_device_count_)
            return;
        auto &slot = local_window_->devices[local_device_idx];
        slot.bytes_loaded = slot.total_bytes;
        slot.finished = 1;
        MPI_Win_flush_local(rank_, win_);
    }

    // =============================================================================
    // Polling (rank 0 only)
    // =============================================================================

    void WeightLoadProgressAggregator::startPolling(std::shared_ptr<WeightLoadProgress> renderer)
    {
        if (rank_ != 0 || !renderer)
            return;

        renderer_ = std::move(renderer);
        polling_.store(true, std::memory_order_release);
        poll_thread_ = std::thread(&WeightLoadProgressAggregator::pollLoop, this);
    }

    void WeightLoadProgressAggregator::stopPolling()
    {
        if (!polling_.load(std::memory_order_acquire))
            return;

        polling_.store(false, std::memory_order_release);
        if (poll_thread_.joinable())
            poll_thread_.join();

        // One final poll to capture any remaining updates
        if (rank_ == 0 && renderer_)
            pollOnce();
    }

    void WeightLoadProgressAggregator::barrier()
    {
        MPI_Barrier(comm_);

        // After barrier, all ranks are done — do final poll
        if (rank_ == 0 && renderer_)
            pollOnce();
    }

    void WeightLoadProgressAggregator::pollLoop()
    {
        while (polling_.load(std::memory_order_acquire))
        {
            pollOnce();
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        }
    }

    void WeightLoadProgressAggregator::pollOnce()
    {
        if (!renderer_)
            return;

        // Read all remote rank windows via MPI_Get
        for (int r = 0; r < world_size_; ++r)
        {
            if (r == rank_)
                continue; // Skip self — rank 0's own progress is updated directly

            MPI_Get(&remote_buffers_[r],
                    static_cast<int>(sizeof(RankWindow)),
                    MPI_BYTE,
                    r, // target rank
                    0, // displacement
                    static_cast<int>(sizeof(RankWindow)),
                    MPI_BYTE,
                    win_);
        }

        // Complete all outstanding Gets
        MPI_Win_flush_all(win_);

        // Process remote data → update renderer
        for (int r = 0; r < world_size_; ++r)
        {
            if (r == rank_)
                continue;

            const RankWindow &rw = remote_buffers_[r];
            for (int d = 0; d < MAX_DEVICES_PER_RANK; ++d)
            {
                const DeviceSlot &slot = rw.devices[d];
                if (!slot.active)
                    continue;

                // Lazy registration: first time we see this remote device
                if (remote_dev_idx_[r][d] < 0)
                {
                    std::string label(slot.label);
                    if (label.empty())
                        label = std::to_string(r) + ":dev" + std::to_string(d);
                    remote_dev_idx_[r][d] = renderer_->registerDevice(label, slot.total_bytes);
                }

                int renderer_idx = remote_dev_idx_[r][d];

                if (slot.finished)
                {
                    renderer_->finish(renderer_idx);
                }
                else
                {
                    renderer_->update(renderer_idx, slot.bytes_loaded);
                }
            }
        }
    }

} // namespace llaminar2
