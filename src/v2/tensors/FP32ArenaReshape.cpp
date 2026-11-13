/**
 * @file FP32ArenaReshape.cpp
 * @brief Delegate every live storage operation to one retained arena FP32 root.
 *
 * The derived object owns shape metadata only. TransferEngine canonicalizes
 * its existing owner hook before allocation, publication and exact event joins.
 * Native kernels keep FP32Tensor dispatch; no mutable raw-pointer rebinding exists.
 */
#include "FP32ArenaReshape.h"
#include "memory/BufferArena.h"
#include "backends/IBackend.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include <limits>
#include <optional>
#include <stdexcept>
namespace llaminar2
{
namespace
{
/** @brief Check all products before constructing ordinary FP32 shape metadata. */
size_t checkedBytes(const std::vector<size_t> &shape)
{
    if (shape.size() != 2) throw std::invalid_argument("Arena reshape requires a positive rank-two matrix");
    size_t elements = 1;
    for (size_t dimension : shape)
    {
        if (!dimension) throw std::invalid_argument("Arena reshape rejects zero dimensions");
        if (elements > std::numeric_limits<size_t>::max() / dimension)
            throw std::overflow_error("Arena reshape element count overflows");
        elements *= dimension;
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::overflow_error("Arena reshape byte count overflows");
    return elements * sizeof(float);
}
/** @brief Reject operations that would give shape metadata another storage authority. */
[[noreturn]] void rejectStorageMutation()
{
    throw std::logic_error("Arena reshape cannot resize, rebind or release storage; use its canonical arena root");
}
}
/** @brief Concrete FP32 type carrying immutable shape and a retained canonical storage owner. */
class FP32StorageReshape final : public FP32Tensor
{
public:
    /** @brief Construct metadata only after the factory authenticates root and complete extent. */
    FP32StorageReshape(std::shared_ptr<FP32Tensor> root, const std::vector<size_t> &shape, BufferId buffer)
        : FP32Tensor(shape, root->home_device(), MappedStorageConstructionTag{}),
          root_(std::move(root)), root_shape_(root_->shape()), bytes_(root_->size_bytes()), buffer_(buffer) {}
    /** @return The sole allocation/event/coherence owner; a stale sealed binding fails closed. */
    TensorBase *transferStorageOwner() override { validateIfSealed(); return root_.get(); }
    /** @return The same canonical root from const consumers. */
    const TensorBase *transferStorageOwner() const override { validateIfSealed(); return root_.get(); }
    /** @return Root-authoritative current device, independent of logical shape. */
    std::optional<DeviceId> current_device() const override { validateIfSealed(); return root_->current_device(); }
    /** @return The original execution participant. */
    DeviceId home_device() const override { return root_->home_device(); }
    /** @return Root value authority; no child shadow state is read. */
    std::optional<DeviceId> getAuthoritativeDevice() const override { return root_->getAuthoritativeDevice(); }
    /** @return Root coherence shared by every shape handle. */
    TensorCoherenceState coherenceState() const override { return root_->coherenceState(); }
    /** @return Root storage policy, immutable through a shape handle. */
    MemoryResidency memoryResidency() const override { return root_->memoryResidency(); }
    /** @return Root prepared-state query without marking activation weights. */
    bool hasPreparedDeviceState() const override { return root_->hasPreparedDeviceState(); }
    /** @return Whether root storage is valid on the exact requested device. */
    bool is_on_device(DeviceId device) const override { return root_->is_on_device(device); }
    /** @return Root host validity rather than dormant inherited fields. */
    bool isHostValid() const override { return root_->isHostValid(); }
    /** @return Root address after sealed identity validation, without preparing or copying. */
    void *gpu_data_ptr() override { validateIfSealed(); return root_->gpu_data_ptr(); }
    /** @return Root address for native read-only consumers. */
    const void *gpu_data_ptr() const override { validateIfSealed(); return root_->gpu_data_ptr(); }
    /** @return Root terminal host observation, using its exact producer event. */
    const float *data() const override { validateIfSealed(); return root_->data(); }
    /** @return Root mutable host storage after existing event/coherence acquisition. */
    float *mutable_data() override { validateIfSealed(); return root_->mutable_data(); }
    /**
     * @brief Prepare the retained root through the public transfer authority.
     * @param target Exact participant admitted for this immutable reshape.
     * @param stream Exact non-null GPU consumer stream; unused for CPU storage.
     * @return True after the canonical root owns valid input on its participant.
     * @throws std::exception For stale identity, wrong participant or invalid preparation.
     * @details Canonicalizing to the root before placement prevents recursion and
     * keeps GPU stream/capture validation identical to ordinary transfer callers.
     */
    bool ensureOnDevice(DeviceId target, void *stream = nullptr) override
    {
        validateIfSealed();
        requireParticipant(target);
        if (target.is_gpu())
            TransferEngine::prepareDeviceInput(root_.get(), target, stream);
        else
            TransferEngine::prepareHostInput(root_.get());
        return true;
    }
    /** @brief Delegate setup output placement without creating a child allocation. */
    bool allocateOnDevice(DeviceId target, void *stream = nullptr) override
    {
        validateIfSealed(); requireParticipant(target); return root_->allocateOnDevice(target, stream);
    }
    /** @brief Acquire host data through the root event; no view-local transfer record exists. */
    bool ensureOnHost(void *stream = nullptr) override { validateIfSealed(); return root_->ensureOnHost(stream); }
    /** @brief Host writes invalidate the root value while retaining its original allocation. */
    void invalidateGpuData() override { validateIfSealed(); root_->invalidateGpuData(); }
    /** @return This metadata object is a view, while the root remains its transfer owner. */
    bool is_view() const override { return true; }
    /** @brief Releasing root storage through a shape-only handle is forbidden. */
    void release_raw_data() override { rejectStorageMutation(); }
    /** @brief Releasing host/source storage through a shape-only handle is forbidden. */
    void release_host_weight_data() override { rejectStorageMutation(); }
    /** @brief A view cannot alter the root's placement policy. */
    void setGpuOnly() override { rejectStorageMutation(); }
    /** @brief A view cannot silently select another storage mode. */
    void setHostResident() override { rejectStorageMutation(); }
    /** @brief Mapped storage is outside the admitted ordinary FP32 reshape policy. */
    void releaseMmapHostRegistration() override { rejectStorageMutation(); }
    /** @brief Legacy host-subview construction must not escape the sealed arena contract. */
    std::shared_ptr<TensorBase> create_view(const std::vector<size_t> &, size_t) override { rejectStorageMutation(); }
    /** @brief Transfer contents through TransferEngine, never a dormant child host vector. */
    bool copyFrom(const TensorBase *) override { rejectStorageMutation(); }
    /** @brief This legacy direct host writer is outside prepared arena projection execution. */
    bool from_int32_with_scales(const int32_t *, int, int, const float *, const float *, const float *) override
    { rejectStorageMutation(); }
    /** @return Root host-release state. */
    bool is_raw_data_released() const override { return root_->is_raw_data_released(); }
    /** @brief Seal existing physical storage once; ordinary preparation owns materialization. */
    void seal()
    {
        if (isGraphCaptureActive()) throw std::logic_error("Arena reshape storage seal must precede native capture");
        if (seal_) { validate(); return; }
        validateRoot();
        Seal next{root_->home_device().is_gpu() ? root_->gpu_data_ptr() : root_->raw_data(), 0};
        if (!next.address) throw std::logic_error("Arena reshape seal requires normally prepared native storage");
        if (root_->home_device().is_gpu())
        {
            requireParticipant(root_->current_device().value_or(DeviceId::invalid()));
            auto *backend = root_->resolveBackend(home_device());
            if (!backend || !(next.generation = backend->deviceRuntimeGeneration(home_device().gpu_ordinal())))
                throw std::logic_error("Arena reshape requires an authenticated native runtime generation");
        }
        seal_ = next;
    }
    /** @brief Reject a rebound owner or physical address before native use. */
    void validate() const
    {
        if (!seal_) throw std::logic_error("Arena reshape storage is not sealed for capture");
        validateRoot();
        const void *address = home_device().is_gpu() ? root_->gpu_data_ptr() : root_->raw_data();
        if (address != seal_->address) throw std::logic_error("Arena reshape physical storage changed after sealing");
        if (home_device().is_gpu())
        {
            requireParticipant(root_->current_device().value_or(DeviceId::invalid()));
            auto *backend = root_->resolveBackend(home_device());
            if (!backend || backend->deviceRuntimeGeneration(home_device().gpu_ordinal()) != seal_->generation)
                throw std::logic_error("Arena reshape runtime generation changed after sealing");
        }
    }
    /** @return Whether physical preparation has been explicitly sealed. */
    bool sealed() const noexcept { return seal_.has_value(); }
    /** @return Declaration metadata without forcing physical materialization before capture. */
    FP32ArenaReshape::DeclarationIdentity declaration() const
    {
        validateRoot();
        validateIfSealed();
        return {root_.get(), buffer_, home_device(), bytes_, {shape()[0], shape()[1]}};
    }
    /** @return Immutable complete capture key borrowed from the retained view. */
    FP32ArenaReshape::CaptureIdentity identity() const
    {
        validate();
        return {root_.get(), buffer_, home_device(), seal_->address, bytes_, 0, {shape()[0], shape()[1]}, seal_->generation};
    }
protected:
    /** @return Already-owned host bytes, without observing or repairing live device data. */
    void *raw_host_data_ptr() override { validateIfSealed(); return const_cast<void *>(root_->raw_data()); }
    /** @return Root host backing used by the canonical transfer authority. */
    const void *raw_host_data_ptr() const override { validateIfSealed(); return root_->raw_data(); }
private:
    struct Seal { const void *address; std::uint64_t generation; };
    /** @brief Reject stale state only after the normal preparation seal exists. */
    void validateIfSealed() const { if (seal_) validate(); }
    /** @brief Validate root lifetime, complete shape and storage policy without requiring a GPU pointer. */
    void validateRoot() const
    {
        if (!root_ || root_->transferStorageOwner() != root_.get() || root_->is_view() ||
            root_->shape() != root_shape_ || root_->size_bytes() != bytes_ || root_->is_raw_data_released() ||
            root_->isMapped() || root_->memoryResidency() != MemoryResidency::STANDARD)
            throw std::logic_error("Arena reshape lost its immutable ordinary FP32 root");
    }
    /** @brief Reject backend/ordinal changes rather than materialize another participant allocation. */
    void requireParticipant(DeviceId target) const
    {
        if (target != home_device()) throw std::logic_error("Arena reshape participant does not match its root");
    }
    std::shared_ptr<FP32Tensor> root_;
    const std::vector<size_t> root_shape_;
    const size_t bytes_;
    const BufferId buffer_;
    std::optional<Seal> seal_;
};
FP32ArenaReshape FP32ArenaReshape::fromArena(BufferArena &arena, BufferId id,
    const std::vector<size_t> &shape, DeviceId participant)
{
    const size_t bytes = checkedBytes(shape);
    auto root = std::dynamic_pointer_cast<FP32Tensor>(arena.getSharedTensor(id));
    if (!root || arena.getTensor(id) != root.get() || root->transferStorageOwner() != root.get() ||
        root->is_view() || root->is_raw_data_released() || root->isMapped() || root->memoryResidency() != MemoryResidency::STANDARD ||
        root->home_device() != participant || root->size_bytes() != bytes)
        throw std::invalid_argument("Arena reshape requires one owned ordinary FP32 root, exact device and whole extent");
    return FP32ArenaReshape(std::make_shared<FP32StorageReshape>(std::move(root), shape, id), id);
}
FP32Tensor &FP32ArenaReshape::tensor() const
{
    if (!view_) throw std::logic_error("Moved arena reshape has no storage owner");
    return *view_;
}
void FP32ArenaReshape::sealStorage() const
{
    (void)tensor(); view_->seal();
}
void FP32ArenaReshape::validateStorage() const
{
    (void)tensor(); view_->validate();
}
bool FP32ArenaReshape::isSealed() const noexcept { return view_ && view_->sealed(); }
FP32ArenaReshape::DeclarationIdentity FP32ArenaReshape::declarationIdentity() const
{
    (void)tensor(); return view_->declaration();
}
FP32ArenaReshape::CaptureIdentity FP32ArenaReshape::captureIdentity() const
{
    (void)tensor(); return view_->identity();
}
} // namespace llaminar2
