/**
 * @file ImmutableNativeDeviceStorage.h
 * @brief PMA-owned source-native device bytes with a setup-only upload capability.
 *
 * Only TransferEngine may allocate, append, seal or join this backing. Mutable
 * upload storage cannot expose a model pointer. A successful immutable read
 * token retains the sealed allocation and exact consumer-stream event join;
 * captures validate that token without transfers, synchronization or mutation.
 */
#pragma once
#include "backends/DeviceId.h"
#include <cstddef>
#include <cstdint>
#include <memory>
namespace llaminar2
{
class TransferEngine;
struct ImmutableNativeDeviceStorageState;
struct ImmutableNativeUploadStagingState;
/** @brief Sole setup writer; destruction retires only already-drained operations. */
class ImmutableNativeDeviceUpload final
{
public:
    ~ImmutableNativeDeviceUpload();
    ImmutableNativeDeviceUpload(ImmutableNativeDeviceUpload &&) noexcept;
    ImmutableNativeDeviceUpload &operator=(ImmutableNativeDeviceUpload &&) noexcept;
    ImmutableNativeDeviceUpload(const ImmutableNativeDeviceUpload &) = delete;
    ImmutableNativeDeviceUpload &operator=(const ImmutableNativeDeviceUpload &) = delete;
    /** @return Actual original bytes successfully uploaded, not allocation capacity. */
    std::size_t copiedBytes() const noexcept;
private:
    friend class TransferEngine;
    /** @brief TransferEngine alone acquires the matching native allocation/PMA claim. */
    explicit ImmutableNativeDeviceUpload(std::unique_ptr<ImmutableNativeDeviceStorageState> state);
    std::unique_ptr<ImmutableNativeDeviceStorageState> state_;
};
/** @brief Bounded PMA-priced reused host slot; no table-sized host storage exists. */
class ImmutableNativeUploadStaging final
{
public:
    ~ImmutableNativeUploadStaging();
    ImmutableNativeUploadStaging(const ImmutableNativeUploadStaging &) = delete;
    ImmutableNativeUploadStaging &operator=(const ImmutableNativeUploadStaging &) = delete;
    /** @return Original loader slot capacity, independent of the table byte count. */
    std::size_t sizeBytes() const noexcept;
    /** @return Writable bytes only after the previous setup append drained this slot. */
    std::byte *data() const noexcept;
private:
    friend class TransferEngine;
    /** @brief TransferEngine alone binds the drained-loader receipt and staging lease. */
    explicit ImmutableNativeUploadStaging(std::unique_ptr<ImmutableNativeUploadStagingState> state);
    std::unique_ptr<ImmutableNativeUploadStagingState> state_;
};
/** @brief Read-only device allocation, retained by every captured consumer. */
class ImmutableNativeDeviceRegion final
{
public:
    ~ImmutableNativeDeviceRegion();
    ImmutableNativeDeviceRegion(const ImmutableNativeDeviceRegion &) = delete;
    ImmutableNativeDeviceRegion &operator=(const ImmutableNativeDeviceRegion &) = delete;
    /** @return Exact owning GPU; not a selectable transfer destination. */
    DeviceId device() const noexcept;
    /** @return Original native source allocation extent. */
    std::size_t sizeBytes() const noexcept;
    /** @return Immutable native pointer, exposed only after complete successful upload. */
    const std::byte *data() const noexcept;
    /** @return Unique immutable allocation identity, never a live execution counter. */
    std::uint64_t generation() const noexcept;
    /** @return Backend runtime generation authenticated during allocation and every join. */
    std::uint64_t runtimeGeneration() const noexcept;
private:
    friend class TransferEngine;
    /** @brief Seal consumes the sole writer only after complete source coverage. */
    explicit ImmutableNativeDeviceRegion(std::unique_ptr<ImmutableNativeDeviceStorageState> state);
    std::unique_ptr<ImmutableNativeDeviceStorageState> state_;
};
/** @brief Immutable exact-stream event-acquired read authority; retains the allocation. */
class ImmutableNativeDeviceRead final
{
public:
    /** @return Same immutable backing borrowed by the captured consumer. */
    const std::shared_ptr<const ImmutableNativeDeviceRegion> &owner() const noexcept { return owner_; }
    /** @return Exact stream which acquired the completed setup producer event. */
    void *consumerStream() const noexcept { return stream_; }
private:
    friend class TransferEngine;
    /** @brief Only a successful TransferEngine event join mints a read authority. */
    ImmutableNativeDeviceRead(std::shared_ptr<const ImmutableNativeDeviceRegion> owner, void *stream)
        : owner_(std::move(owner)), stream_(stream) {}
    std::shared_ptr<const ImmutableNativeDeviceRegion> owner_;
    void *stream_;
};
}
