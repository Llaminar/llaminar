/**
 * @file DeviceGenerationControlBinding.h
 * @brief Immutable borrow of the canonical request controller's native storage.
 *
 * The generation parent owns initialization, execution ordering and terminal
 * observation. Compute stages may borrow its exact health/error or counter
 * words with their actual arena access declarations, without copying mutable values to the
 * host or creating a storage-service boundary. This setup capability verifies
 * the arena role, request, backend and runtime generation on every reuse.
 */
#pragma once
#include "backends/IBackend.h"
#include "kernels/common/DeviceGenerationControlLayout.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
/** @brief Checked storage identity; construction never reads controller contents. */
class DeviceGenerationControlBinding final
{
public:
    /**
     * @brief Borrow one already materialized canonical controller row.
     * @param arena Exact participant execution arena, retained by the enclosing graph.
     * @param device GPU that owns this controller allocation.
     * @param request Original request slot, independent of model-layer or bucket ordinal.
     * @return Sealed metadata capability; no storage or live state is created.
     * @throws std::invalid_argument For external, missing, foreign or malformed storage.
     */
    static DeviceGenerationControlBinding admit(BufferArena &arena, DeviceId device, int request)
    {
        auto *owner = arena.getSharedTensor(BufferId::MTP_GENERATION_CONTROL).get();
        if (!device.is_gpu() || !owner || arena.getTensor(BufferId::MTP_GENERATION_CONTROL) != owner ||
            owner->is_view() || owner->transferStorageOwner() != owner || owner->native_type() != TensorType::INT32 ||
            request < 0 || owner->shape().size() != 2 ||
            owner->shape()[1] != sampling_math::kDeviceGenerationControlCount ||
            static_cast<std::size_t>(request) >= owner->shape()[0] || owner->current_device() != device)
            throw std::invalid_argument("Generation control borrow requires its canonical GPU arena owner and request");
        auto *base = static_cast<std::int32_t *>(arena.getDevicePtr(BufferId::MTP_GENERATION_CONTROL, device));
        if (!base || reinterpret_cast<std::uintptr_t>(base) % alignof(std::uint64_t))
            throw std::invalid_argument("Generation control borrow requires prepared aligned native storage");
        auto *backend = owner->resolveBackend(device);
        const auto generation = backend ? backend->deviceRuntimeGeneration(device.ordinal) : 0;
        if (!generation)
            throw std::invalid_argument("Generation control borrow requires an authenticated backend runtime generation");
        return DeviceGenerationControlBinding(arena, *owner, device, base, request, backend, generation);
    }

    /** @brief Reject changed storage/device/runtime identity without reading any live words. */
    void requireFor(DeviceId device) const
    {
        if (device != device_ || arena_->getTensor(BufferId::MTP_GENERATION_CONTROL) != owner_ ||
            arena_->getSharedTensor(BufferId::MTP_GENERATION_CONTROL).get() != owner_ ||
            owner_->is_view() || owner_->transferStorageOwner() != owner_ ||
            owner_->native_type() != TensorType::INT32 || owner_->current_device() != device_ ||
            arena_->getDevicePtr(BufferId::MTP_GENERATION_CONTROL, device_) != base_ ||
            owner_->shape().size() != 2 || owner_->shape()[1] != sampling_math::kDeviceGenerationControlCount ||
            owner_->shape()[0] != requests_ || owner_->resolveBackend(device_) != backend_ ||
            backend_->deviceRuntimeGeneration(device_.ordinal) != generation_)
            throw std::invalid_argument("Generation control owner/device/runtime changed after binding");
    }

    /** @brief Authenticate an exact canonical request slot before adding a consumer node. */
    void requireRequest(DeviceId device, int request) const
    {
        requireFor(device);
        if (request != request_)
            throw std::invalid_argument("Generation control borrow belongs to another request");
    }

    /** @return Canonical arena root for explicit read/write lifetime and publication contracts. */
    TensorBase *owner() const noexcept {return owner_;}

    /** @return Original native row, used only by typed device operations and terminal counters. */
    std::int32_t *row() const noexcept
    {
        return base_ + static_cast<std::size_t>(request_) * sampling_math::kDeviceGenerationControlCount;
    }
    /** @return Canonical device health word, never a host-owned shadow. */
    int *healthy() const noexcept { return row() + sampling_math::kDeviceGenerationControlOk; }
    /** @return Canonical first terminal error word. */
    int *error() const noexcept { return row() + sampling_math::kDeviceGenerationControlErrorCode; }

    /** @return Complete immutable storage identity, excluding all mutable word values. */
    std::uint64_t captureIdentity() const noexcept
    {
        std::uint64_t value = 1469598103934665603ULL;
        const auto mix = [&](std::uint64_t word) { value ^= word; value *= 1099511628211ULL; };
        mix(1); mix(reinterpret_cast<std::uintptr_t>(arena_)); mix(reinterpret_cast<std::uintptr_t>(owner_));
        mix(static_cast<std::uint64_t>(BufferId::MTP_GENERATION_CONTROL));
        mix(static_cast<std::uint64_t>(device_.type)); mix(device_.ordinal);
        mix(reinterpret_cast<std::uintptr_t>(base_)); mix(request_); mix(requests_);
        mix(reinterpret_cast<std::uintptr_t>(backend_)); mix(generation_);
        mix(sampling_math::kDeviceGenerationControlCount);
        return value ? value : 1;
    }
private:
    /** @brief Retain validated owner metadata while the graph retains its arena lifetime. */
    DeviceGenerationControlBinding(BufferArena &arena, TensorBase &owner, DeviceId device,
        std::int32_t *base, int request, IBackend *backend, std::uint64_t generation)
        : arena_(&arena), owner_(&owner), device_(device), base_(base), request_(request),
          requests_(owner.shape()[0]), backend_(backend), generation_(generation) {}
    BufferArena *arena_;
    TensorBase *owner_;
    DeviceId device_;
    std::int32_t *base_;
    int request_;
    std::size_t requests_;
    IBackend *backend_;
    std::uint64_t generation_;
};
} // namespace llaminar2
