/**
 * @file W4A4Tensor.cpp
 * @brief Publish and retire native W4A4/H tensor leases without changing arithmetic or ownership.
 *
 * Complete-row views and prepared engines share the original physical owner.
 * A released tensor retains pointer-free geometry only. Every floating,
 * requantizing or writable interface fails before touching caller storage.
 */
#include "W4A4Tensor.h"
#include "../kernels/KernelFactory.h"

namespace llaminar2
{
    namespace
    {
        /** @return Exact public tensor identity for a source-admitted online transform. */
        TensorType tensorType(w4a4::Transform transform)
        {
            switch (transform)
            {
            case w4a4::Transform::Identity: return TensorType::W4A4;
            case w4a4::Transform::PaleySylvester: return TensorType::W4A4H;
            }
            throw std::invalid_argument("W4A4 tensor has an unsupported online transform");
        }
    }

    W4A4Tensor::W4A4Tensor(w4a4::WeightStorage storage)
        : W4A4Tensor(std::move(storage), ViewKind::Matrix) {}

    W4A4Tensor::W4A4Tensor(w4a4::WeightStorage storage, ViewKind kind)
        : contract_(storage.view().contract()),
          shape_{static_cast<size_t>(contract_.rows()), static_cast<size_t>(contract_.columns())},
          type_(tensorType(contract_.transform())), allocator_(storage.allocator()),
          view_kind_(kind), storage_(std::move(storage)) {}

    W4A4Tensor::~W4A4Tensor()
    {
        retireHostTransferLifetimeBeforeStorageDestruction();
    }

    w4a4::WeightStorage W4A4Tensor::retainWeightStorage() const
    {
        if (!storage_) throw std::logic_error("W4A4 tensor source lease has retired");
        return *storage_;
    }

    const w4a4::SourceBlock *W4A4Tensor::typed_data() const
    {
        if (!storage_) throw std::logic_error("W4A4 tensor source lease has retired");
        return storage_->view().blocks().data();
    }

    void W4A4Tensor::release_raw_data()
    {
        // Retire the exact producer/registration lifetime before dropping this
        // lease. Other complete-row views and prepared engines remain owners.
        retireHostTransferLifetimeBeforeStorageDestruction();
        storage_.reset();
    }

    void W4A4Tensor::release_host_weight_data() { release_raw_data(); }

    std::unique_ptr<ITensorGemm> W4A4Tensor::createGemm()
    {
        using llaminar::v2::kernels::KernelFactory;
        return KernelFactory::createGemm(this, KernelFactory::getDeviceType(allocator_));
    }

    std::shared_ptr<TensorBase> W4A4Tensor::create_view(const std::vector<size_t> &shape, size_t offset)
    {
        if (shape.size() != 2 || shape[1] != shape_[1] || offset % shape_[1] != 0)
            throw std::invalid_argument("W4A4 tensor views require complete, original-K output rows");
        const auto storage = retainWeightStorage();
        return std::shared_ptr<TensorBase>(new W4A4Tensor(
            storage.rowSlice(offset / shape_[1], shape[0]), ViewKind::CompleteRows));
    }

    void W4A4Tensor::setLayout(TensorLayout layout)
    {
        if (layout != TensorLayout::ROW_MAJOR_2D)
            throw std::invalid_argument("W4A4 tensor layout must retain its complete native rows");
    }

    [[noreturn]] void W4A4Tensor::rejectConversion()
    {
        throw std::logic_error("W4A4/H requires native online-A4 execution, not floating weights or alternate quantization");
    }
    [[noreturn]] void W4A4Tensor::rejectMutation()
    {
        throw std::logic_error("W4A4/H native source bytes and policy are immutable");
    }
    [[noreturn]] void W4A4Tensor::rejectTransfer()
    {
        throw std::logic_error("W4A4/H requires native weight preparation, not generic tensor transfer");
    }

    const float *W4A4Tensor::data() const { rejectConversion(); }
    float *W4A4Tensor::mutable_data() { rejectMutation(); }
    void *W4A4Tensor::raw_mutable_data() { rejectMutation(); }
    void *W4A4Tensor::active_mutable_data_ptr() { rejectMutation(); }
    float *W4A4Tensor::mutable_fp32_data() { rejectMutation(); }
    bool W4A4Tensor::copyFrom(const TensorBase *) { rejectMutation(); }
    TensorBase *W4A4Tensor::transferStorageOwner() { rejectTransfer(); }
    const TensorBase *W4A4Tensor::transferStorageOwner() const { rejectTransfer(); }
    bool W4A4Tensor::ensureOnDevice(DeviceId, void *) { rejectTransfer(); }
    bool W4A4Tensor::allocateOnDevice(DeviceId, void *) { rejectTransfer(); }
    bool W4A4Tensor::ensureOnHost(void *) { (void)retainWeightStorage(); return true; }
    void W4A4Tensor::to_fp32(float *) const { rejectConversion(); }
    void W4A4Tensor::to_bf16(uint16_t *) const { rejectConversion(); }
    void W4A4Tensor::to_fp16(uint16_t *) const { rejectConversion(); }
    void W4A4Tensor::to_int8_blocked(int8_t *, float *, size_t) const { rejectConversion(); }
    bool W4A4Tensor::to_int8_perchannel(int8_t *, float *, float *) const { rejectConversion(); }
    void W4A4Tensor::to_fp32_row(size_t, float *) const { rejectConversion(); }
    void W4A4Tensor::to_fp32_span(size_t, size_t, float *) const { rejectConversion(); }
    void *W4A4Tensor::raw_host_data_ptr() { rejectMutation(); }
    const void *W4A4Tensor::raw_host_data_ptr() const
    {
        return storage_ ? storage_->view().blocks().data() : nullptr;
    }
} // namespace llaminar2
