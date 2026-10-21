/**
 * @file SnapshotNativeSource.h
 * @brief Checked original GPU snapshot source without guessed ownership or null-base arithmetic.
 *
 * This setup/capture value borrows an existing native allocation; it never
 * allocates, transfers, publishes authority or repairs a stale view. The stage's
 * declared arena contract materializes storage before snapshot preparation.
 */
#pragma once
#include "tensors/ITensor.h"
#include <optional>
namespace llaminar2 {
/** @brief Immutable checked byte subview of one exact participant-owned native tensor. */
class SnapshotNativeSource final {
public:
    /** @brief Resolve an existing in-bounds source only on its actual allocation owner.
     * @param tensor Original retained tensor or correctly published native view.
     * @param device Exact participant collecting/recording this snapshot.
     * @param offset Byte origin within the original tensor allocation.
     * @param bytes Required declared source capacity, independent of mutable live rows.
     * @return No binding for absent/stale ownership, null base, bad device or invalid extent.
     */
    static std::optional<SnapshotNativeSource> bind(const ITensor& tensor,DeviceId device,
        std::size_t offset,std::size_t bytes) {
        const auto owner=tensor.current_device();
        const auto* base=static_cast<const std::byte*>(tensor.gpu_data_ptr());
        const auto capacity=tensor.size_bytes();
        if(!device.is_gpu() || !owner || *owner!=device || !base || !bytes ||
            offset>capacity || bytes>capacity-offset)return std::nullopt;
        return SnapshotNativeSource(base+offset,device,bytes);
    }
    /** @brief Exact checked source pointer; null arithmetic never creates it. */
    const void* data()const noexcept{return data_;}
    /** @brief Original native allocation owner, never an inferred target fallback. */
    DeviceId device()const noexcept{return device_;}
    /** @brief Declared source byte extent; native semantic counts select actual copied rows. */
    std::size_t bytes()const noexcept{return bytes_;}
private:
    /** @brief Construct only after the complete owner/base/extent contract succeeds. */
    SnapshotNativeSource(const void* data,DeviceId device,std::size_t bytes):data_(data),device_(device),bytes_(bytes){}
    const void*data_;DeviceId device_;std::size_t bytes_;
};
} // namespace llaminar2
