/**
 * @file NativeWeightSourceTransaction.h
 * @brief One serialized PMA-owned native input and simultaneous consumer workspace.
 *
 * The original rank PMA supplies its shared CPU transaction scope. The lock spans source
 * reads, final consumer writes and release of the actual arena lease. Final
 * tensor/engine storage has its own existing typed PMA owner. No byte balance or
 * source cache is maintained here; GPU event rings remain independent owners.
 */
#pragma once
#include "loaders/IModelLoader.h"
#include "loaders/NativeWeightLoadStaging.h"
#include "tensors/NativeWeightSourceRange.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <span>
#include <string>
namespace llaminar2 {
/** @brief Scoped native input/scratch arena used by every bounded CPU weight consumer. */
class NativeWeightSourceTransaction final {
    friend class WeightManager;
private:
    /**
     * @brief Hold the original serialized load owner through arena retirement.
     * @param loader Same sealed source loader used by the original weight plan.
     * @param name Original physical tensor name, not a generated window identity.
     * @param request Consumer-authored native unit and simultaneous scratch quote.
     * @param placement Same admitted original CPU allocation policy as planning.
     * @param memory Sole original rank physical-memory authority.
     * @param original_allocator Exact declared CPU allocation identity, independent of NUMA placement.
     * @param budget Existing configured startup source/preparation cap.
     * @param source_scope Same PMA transaction acquired before final consumer construction.
     * @throws std::invalid_argument For an indivisible unit over budget.
     * @note Only the original manager may construct this scope after validating
     * ordinary factory placement or the retained source member's exact placement.
     */
    NativeWeightSourceTransaction(IModelLoader &loader,
        std::string name, const NativeWeightLoadStagingRequest &request,
        CPUStoragePlacement placement, std::shared_ptr<PhysicalMemoryAuthority> memory,
        DeviceId original_allocator, std::optional<size_t> budget, PhysicalMemoryCPUWeightSourceLease source_scope)
        : lock_(std::move(source_scope)), loader_(loader), name_(std::move(name)),
          source_(loader_.nativeWeightSourceGeometry(name_)),
          geometry_(resolveNativeWeightLoadStaging(request, placement, budget))
    {
        const size_t bytes = geometry_.scratch_bytes ?
            geometry_.scratch_offset + geometry_.scratch_bytes : geometry_.native_bytes;
        const CPUStorageRoot<uint8_t> root(bytes, PhysicalMemoryOwner::WeightLoadStaging,
            original_allocator, placement, CPUStorageInitialization::Uninitialized);
        if (root.physicalBytes() != geometry_.physical_bytes)
            throw std::logic_error("Native source arena differs from its original physical quote");
        storage_ = root.allocate(std::move(memory));
    }
public:
    NativeWeightSourceTransaction(const NativeWeightSourceTransaction &) = delete;
    NativeWeightSourceTransaction &operator=(const NativeWeightSourceTransaction &) = delete;

    /**
     * @brief Read and consume each exact window before reusing its physical bytes.
     * @param range Original matrix rectangle within the same complete source.
     * @param consume Synchronous final writer accepting (window,native,scratch).
     * @note The consumer commits to final admitted storage while the load owner
     * remains locked. Neither native nor scratch spans may escape this call.
     */
    template<class Consumer>
    void visit(const NativeWeightSourceWindow &range, Consumer &&consume)
    {
        forEachNativeWeightSourceWindow(source_, range, geometry_.native_bytes,
            [&](const NativeWeightSourceWindow &window) {
                auto native = std::span<std::byte>(reinterpret_cast<std::byte *>(storage_.data()), window.nativeBytes());
                loader_.readNativeWeightSourceWindow(name_, window, native);
                auto scratch = std::span<std::byte>(reinterpret_cast<std::byte *>(storage_.data()) +
                    geometry_.scratch_offset, geometry_.scratch_bytes);
                consume(window, std::span<const std::byte>(native), scratch);
            });
    }
    /** @return Same original native declaration used to construct the final writer. */
    const NativeWeightSourceGeometry &source() const noexcept { return source_; }
private:
    // Members retire in reverse declaration order: the arena's real PMA lease
    // disappears before releasing the serialization owner for the next source.
    PhysicalMemoryCPUWeightSourceLease lock_;
    IModelLoader &loader_;
    const std::string name_;
    const NativeWeightSourceGeometry source_;
    const NativeWeightLoadStagingGeometry geometry_;
    AlignedVector<uint8_t> storage_;
};
}
