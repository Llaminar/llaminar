#pragma once

#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "loaders/gpu_pipeline/PinnedRingBuffer.h"
#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

class IBackend;

class LoadOrchestrator
{
  public:
    struct DeviceContext
    {
        int device_id = -1;
        std::unique_ptr<WeightVRAMPool> pool;
        std::unique_ptr<PinnedRingBuffer> pinned_ring;
        std::vector<WeightJob> pending_jobs;
    };

    /// @param backend  GPU backend (CUDA or ROCm). Must outlive the orchestrator.
    explicit LoadOrchestrator(IBackend* backend = nullptr);
    ~LoadOrchestrator();

    /// Add a device to manage.
    void addDevice(int device_id);

    /// Plan a weight for a specific device.
    void planWeight(int device_id, const std::string& name,
                    int N, int K, int payload_bytes_per_block,
                    bool is_asymmetric, bool has_emins,
                    size_t raw_gguf_bytes);

    /// Allocate all device pools. Throws on failure.
    /// @param pinned_slot_size  Max raw GGUF weight size (bytes per pinned slot).
    /// @param num_h2d_streams   Number of ring buffer slots.
    void allocate(size_t pinned_slot_size, int num_h2d_streams = 3);

    /// Get pool for a device. Returns nullptr if not found.
    WeightVRAMPool* getPool(int device_id);
    const WeightVRAMPool* getPool(int device_id) const;

    /// Number of managed devices.
    size_t numDevices() const;

    /// Add a weight job to be loaded on a specific device.
    void addWeightJob(int device_id, const WeightJob& job);

    /// Execute all pending weight jobs with pipelined H2D + GPU repack.
    /// Each device is processed via a DeviceLoadPipeline. Throws on failure.
    void load();

    /// Release staging regions after loading completes.
    void finalize() { }

    /// Release all resources.
    void release();

  private:
    DeviceContext* findDevice(int device_id);
    const DeviceContext* findDevice(int device_id) const;

    /// Create backend-appropriate RepackKernels function pointer struct.
    RepackKernels createRepackKernels() const;

    IBackend* backend_ = nullptr;
    std::vector<DeviceContext> devices_;
};

} // namespace llaminar2
