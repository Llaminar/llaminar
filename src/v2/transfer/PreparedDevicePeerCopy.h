/**
 * @file PreparedDevicePeerCopy.h
 * @brief Frozen direct-peer byte-copy capability over existing admitted
 * tensors.
 *
 * TransferEngine alone imports the producer event and enables the directed
 * physical edge before capture. Strong tensor and original worker owners keep
 * pointers and generations alive; no payload, stream or event is allocated
 * here. Retire all graphs and these capabilities before releasing payload PMA
 * claims.
 */
#pragma once
#include "backends/DeviceId.h"
#include "backends/GPUDeviceContextPool.h"
#include <cstddef>
#include <memory>
#include <utility>
namespace llaminar2 {
class TensorBase;
class TransferEngine;
/** @brief One exact destination-stream peer edge; contains no communicator or
 * alternate transport. */
class PreparedDevicePeerCopy final {
public:
  PreparedDevicePeerCopy(const PreparedDevicePeerCopy &) = delete;
  PreparedDevicePeerCopy &operator=(const PreparedDevicePeerCopy &) = delete;
  PreparedDevicePeerCopy(PreparedDevicePeerCopy &&) noexcept = default;
  PreparedDevicePeerCopy &
  operator=(PreparedDevicePeerCopy &&) noexcept = default;
  ~PreparedDevicePeerCopy() = default;
  /** @brief Enqueue only the live range on the frozen destination
   * worker/stream.
   * @param source_offset Source byte offset. @param destination_offset
   * Destination byte offset.
   * @param live_bytes Logical bytes; zero validates identity/bounds then does
   * no work; capacity is never the transfer extent.
   * @throws std::exception For stale/rebound storage, foreign worker/stream or
   * native failure.
   */
  void enqueue(std::size_t source_offset, std::size_t destination_offset,
               std::size_t live_bytes) const;
  /** @return Original physical source. */
  DeviceId sourceDevice() const noexcept { return source_device_; }
  /** @return Original physical destination. */
  DeviceId destinationDevice() const noexcept { return destination_device_; }
  /** @return Frozen source allocation extent. */
  std::size_t sourceCapacity() const noexcept { return source_capacity_; }
  /** @return Frozen destination allocation extent. */
  std::size_t destinationCapacity() const noexcept {
    return destination_capacity_;
  }

private:
  friend class TransferEngine;
  /** @brief Only TransferEngine can issue a prepared, producer-ordered
   * capability. */
  PreparedDevicePeerCopy(std::shared_ptr<TensorBase> source,
                         std::shared_ptr<TensorBase> destination,
                         DeviceId source_device, DeviceId destination_device,
                         void *stream, GPUContextProvenance source_worker,
                         GPUContextProvenance destination_worker,
                         const void *source_pointer, void *destination_pointer,
                         std::size_t source_capacity,
                         std::size_t destination_capacity)
      : source_(std::move(source)), destination_(std::move(destination)),
        source_device_(source_device), destination_device_(destination_device),
        stream_(stream), source_worker_(std::move(source_worker)),
        destination_worker_(std::move(destination_worker)),
        source_pointer_(source_pointer),
        destination_pointer_(destination_pointer),
        source_capacity_(source_capacity),
        destination_capacity_(destination_capacity) {}
  std::shared_ptr<TensorBase> source_, destination_;
  DeviceId source_device_, destination_device_;
  void *stream_;
  GPUContextProvenance source_worker_, destination_worker_;
  const void *source_pointer_;
  void *destination_pointer_;
  std::size_t source_capacity_, destination_capacity_;
};
} // namespace llaminar2
