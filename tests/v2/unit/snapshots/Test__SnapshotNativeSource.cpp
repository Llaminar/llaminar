/**
 * @file Test__SnapshotNativeSource.cpp
 * @brief Device-free GPU source identity, null-offset and checked-subview regressions.
 *
 * Fake view addresses are inspected as metadata only; no GPU backend initializes,
 * no pointer is dereferenced and no native storage is allocated by these tests.
 */
#include "snapshots/SnapshotNativeSource.h"
#include "tensors/GpuTensorView.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
using namespace llaminar2;
namespace {
/** @brief Metadata-only view deliberately lacking a published allocation owner. */
class UnownedView final:public GpuTensorView {
public:using GpuTensorView::GpuTensorView;
    std::optional<DeviceId> current_device()const override{return std::nullopt;}
};
}
/** @brief Original owner and base are mandatory even when a nonzero subview offset looks non-null. */
TEST(SnapshotNativeSource, NullBaseNonzeroOffsetAndMissingOwnerCannotMasqueradeAsGPUStorage){
    alignas(64)std::array<std::byte,64>storage{};
    GpuTensorView stale(storage.data(),4,4,TensorType::FP32,DeviceId::cuda(0));
    stale.update_view(nullptr,4);
    EXPECT_FALSE(SnapshotNativeSource::bind(stale,DeviceId::cuda(0),8,16));
    EXPECT_FALSE(SnapshotNativeSource::bind(stale,DeviceId::cuda(0),0,16));
    UnownedView unpublished(storage.data(),4,4,TensorType::FP32,DeviceId::cuda(0));
    EXPECT_FALSE(SnapshotNativeSource::bind(unpublished,DeviceId::cuda(0),8,16));
    FP32Tensor host({4,4});EXPECT_FALSE(SnapshotNativeSource::bind(host,DeviceId::cuda(0),8,16));
    EXPECT_EQ(host.gpu_data_ptr(),nullptr);
}
/** @brief Valid native view offsets retain exact allocation backend/ordinal and cannot escape declared storage. */
TEST(SnapshotNativeSource, CheckedSubviewPreservesExactOwnerAndRejectsOverflow){
    alignas(64)std::array<std::byte,64>storage{};
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(1)}){
        GpuTensorView view(storage.data(),4,4,TensorType::FP32,device);
        const auto bound=SnapshotNativeSource::bind(view,device,8,56);ASSERT_TRUE(bound);
        EXPECT_EQ(bound->data(),storage.data()+8);EXPECT_EQ(bound->bytes(),56u);EXPECT_EQ(bound->device(),device);
        EXPECT_FALSE(SnapshotNativeSource::bind(view,DeviceId::cpu(),8,16));
        EXPECT_FALSE(SnapshotNativeSource::bind(view,device.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(1),8,16));
        EXPECT_FALSE(SnapshotNativeSource::bind(view,device,64,4));EXPECT_FALSE(SnapshotNativeSource::bind(view,device,60,8));
        EXPECT_FALSE(SnapshotNativeSource::bind(view,device,std::numeric_limits<std::size_t>::max(),4));
        EXPECT_FALSE(SnapshotNativeSource::bind(view,device,0,std::numeric_limits<std::size_t>::max()));
        EXPECT_FALSE(SnapshotNativeSource::bind(view,device,0,0));
    }
}
