/**
 * @file FP32ArenaReshape.h
 * @brief Sealed, full-storage FP32 shape handles retaining one admitted arena owner.
 *
 * Admission does not prepare GPU storage. The ordinary stage preparation hook
 * seals the root address and runtime generation before capture. There is no
 * child allocation, event, physical reservation, mutable count or copied data.
 */
#pragma once
#include "TensorClasses.h"
#include "memory/BufferId.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
namespace llaminar2
{
class BufferArena;
class FP32StorageReshape;
/** @brief Shared lifetime of immutable logical shape and its one physical arena root. */
class FP32ArenaReshape final
{
public:
    /** @brief Immutable declaration identity available before physical GPU preparation. */
    struct DeclarationIdentity
    {
        const TensorBase *root; ///< Retained arena allocation authority, never an external pointer.
        BufferId buffer; ///< Original physical arena slot.
        DeviceId device; ///< Exact participant.
        size_t bytes; ///< Complete root storage extent.
        std::array<size_t, 2> shape; ///< Immutable logical matrix geometry.
        bool operator==(const DeclarationIdentity &) const = default;
    };
    /** @brief Complete storage/shape identity for one prepared captured operand. */
    struct CaptureIdentity
    {
        const TensorBase *root;
        BufferId buffer;
        DeviceId device;
        const void *address;
        size_t bytes;
        size_t byte_offset;
        std::array<size_t, 2> shape;
        std::uint64_t runtime_generation;
        /** @brief Compare immutable identity without reading tensor contents or allocating. */
        bool operator==(const CaptureIdentity &other) const
        {
            return root == other.root && buffer == other.buffer && device == other.device &&
                address == other.address && bytes == other.bytes && byte_offset == other.byte_offset &&
                runtime_generation == other.runtime_generation && shape.size() == other.shape.size() &&
                std::equal(shape.begin(), shape.end(), other.shape.begin());
        }
    };
    /**
     * @brief Admit shape-only metadata from the existing arena-owned shared tensor.
     * @param arena Existing admission/allocation owner, never an external raw pointer.
     * @param root Physical BufferId retained in stage read/write contracts.
     * @param shape Positive rank-two matrix shape covering the complete root exactly.
     * @param participant Explicit CPU/CUDA/ROCm participant owning the buffer.
     * @return Retained FP32 shape handle; no GPU pointer is required yet.
     * @throws std::exception For invalid type, external/legacy views, device or shape.
     */
    static FP32ArenaReshape fromArena(BufferArena &arena, BufferId root,
        const std::vector<size_t> &shape, DeviceId participant);
    /** @return Concrete FP32 operand preserving native kernel dynamic casts. */
    FP32Tensor &tensor() const;
    /** @return Canonical physical arena identity, never another allocated BufferId. */
    BufferId rootBuffer() const noexcept { return buffer_; }
    /** @return Immutable whole-storage offset; this contract cannot express subranges. */
    size_t byteOffset() const noexcept { return 0; }
    /** @brief Seal already-prepared addresses after ordinary arena preparation, outside capture. */
    void sealStorage() const;
    /** @brief Reject changed root/device/address/capacity/runtime generation after preparation. */
    void validateStorage() const;
    /** @return True only after the explicit prepare/capture storage seal. */
    bool isSealed() const noexcept;
    /** @return Immutable root/shape declaration, validating the physical seal when one exists. */
    DeclarationIdentity declarationIdentity() const;
    /** @return Complete sealed identity; throws before sealing or after stale storage. */
    CaptureIdentity captureIdentity() const;
private:
    /** @brief Retain a validated concrete view; only arena admission constructs handles. */
    FP32ArenaReshape(std::shared_ptr<FP32StorageReshape> view, BufferId buffer)
        : view_(std::move(view)), buffer_(buffer) {}
    std::shared_ptr<FP32StorageReshape> view_;
    BufferId buffer_;
};
} // namespace llaminar2
