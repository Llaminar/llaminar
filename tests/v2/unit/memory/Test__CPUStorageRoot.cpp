/**
 * @file Test__CPUStorageRoot.cpp
 * @brief Device-free replay of the common CPU physical-root contract.
 *
 * All six arena-native dtypes adopt actual admitted backing. Aliases and native
 * views keep that backing alive, independent copies receive exact fresh claims,
 * and failed admission leaves every existing allocation unchanged.
 */
#include "memory/CPUStorageRoot.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include <gtest/gtest.h>
#include <array>
using namespace llaminar2;
namespace
{
const DeviceId cpu1{DeviceType::CPU, 1};
constexpr auto owner = PhysicalMemoryOwner::ExecutionWorkspace;
/** @brief Admit one exact bounded CPU1 workspace owner without memory sampling. */
std::shared_ptr<PhysicalMemoryAuthority> authority(size_t bytes)
{
    const auto capacity = std::max(size_t{1}, bytes);
    PhysicalMemoryPlanBuilder builder;
    builder.add(PhysicalMemoryResource{.world_rank = 0, .device = cpu1,
        .total_bytes = capacity, .admission_available_bytes = capacity}, owner, bytes);
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
}
/** @return Observed exact live physical bytes on the declared allocator. */
size_t live(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
{
    return memory->claimedBytes(cpu1, owner, PhysicalMemoryMaterializationKind::NewAllocation);
}
/** @return Actual physical byte extent for one of the six arena-native dtypes. */
size_t dtypeBytes(int dtype)
{
    switch (dtype)
    {
    case 0: return CPUStorageRoot<float>(66, owner, cpu1).physicalBytes();
    case 1: case 2: return CPUStorageRoot<uint16_t>(66, owner, cpu1).physicalBytes();
    case 3: return CPUStorageRoot<int32_t>(66, owner, cpu1).physicalBytes();
    case 4: return CPUStorageRoot<uint8_t>(nativeMatrixExtent<Q8_1Block>(2,33), owner, cpu1).physicalBytes();
    default: return CPUStorageRoot<uint8_t>(nativeMatrixExtent<Q16_1Block>(2,33), owner, cpu1).physicalBytes();
    }
}
/** @brief Move final root backing into the exact production native tensor owner. */
std::shared_ptr<TensorBase> tensorFor(int dtype, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
{
    const std::vector<size_t> shape{2,33};
    switch (dtype)
    {
    case 0: return std::make_shared<FP32Tensor>(shape, CPUStorageRoot<float>(66,owner,cpu1).allocate(memory), cpu1);
    case 1: return std::make_shared<FP16Tensor>(shape, CPUStorageRoot<uint16_t>(66,owner,cpu1).allocate(memory), cpu1);
    case 2: return std::make_shared<BF16Tensor>(shape, CPUStorageRoot<uint16_t>(66,owner,cpu1).allocate(memory), cpu1);
    case 3: return std::make_shared<INT32Tensor>(shape, CPUStorageRoot<int32_t>(66,owner,cpu1).allocate(memory), cpu1);
    case 4: return std::make_shared<Q8_1Tensor>(shape, CPUStorageRoot<uint8_t>(nativeMatrixExtent<Q8_1Block>(2,33),owner,cpu1).allocate(memory), cpu1);
    default: return std::make_shared<Q16_1Tensor>(shape, CPUStorageRoot<uint8_t>(nativeMatrixExtent<Q16_1Block>(2,33),owner,cpu1).allocate(memory), cpu1);
    }
}
}
TEST(CPUStorageRoot, AllSixDtypesAdoptZeroedBackingAndRetainAliasLifetime)
{
    for (int dtype = 0; dtype < 6; ++dtype)
    {
        SCOPED_TRACE(dtype);
        const size_t bytes = dtypeBytes(dtype);
        auto memory = authority(bytes);
        for (int replay = 0; replay < 3; ++replay)
        {
            auto tensor = tensorFor(dtype, memory);
            ASSERT_EQ(tensor->home_device(), cpu1);
            ASSERT_EQ(live(memory), bytes);
            const auto *raw = static_cast<const uint8_t *>(tensor->raw_data());
            ASSERT_NE(raw, nullptr);
            for (size_t i=0; i<tensor->size_bytes(); ++i) ASSERT_EQ(raw[i], 0u);
            auto alias = std::make_shared<TensorSlice>(tensor, SliceMetadata{});
            EXPECT_EQ(alias->raw_data(), raw);
            tensor.reset();
            EXPECT_EQ(live(memory), bytes);
            alias.reset();
            EXPECT_EQ(live(memory), 0u);
        }
    }
}
TEST(CPUStorageRoot, NativeViewsRetainOriginalBackingAndCpuIdentity)
{
    for (int dtype : {0,1,2,4,5})
    {
        SCOPED_TRACE(dtype);
        auto memory = authority(dtypeBytes(dtype));
        auto tensor = tensorFor(dtype, memory);
        auto view = tensor->create_view({1,33},33);
        ASSERT_NE(view, nullptr);
        EXPECT_EQ(view->home_device(), cpu1);
        tensor.reset();
        EXPECT_EQ(live(memory), dtypeBytes(dtype));
        view.reset();
        EXPECT_EQ(live(memory), 0u);
    }
}
TEST(CPUStorageRoot, CopyMoveResetAndZeroRootPreserveIssuer)
{
    const CPUStorageRoot<float> root(17, owner, cpu1);
    auto memory = authority(2*root.physicalBytes());
    auto original = root.allocate(memory);
    EXPECT_EQ(original.allocationBytes(),root.physicalBytes());
    original[0] = 7;
    {
        AlignedVector<float> copy(original);
        EXPECT_EQ(copy[0],7);
        EXPECT_EQ(live(memory),2*root.physicalBytes());
        AlignedVector<float> moved(std::move(copy));
        EXPECT_EQ(live(memory),2*root.physicalBytes());
        moved.clear();moved.shrink_to_fit();
        EXPECT_EQ(live(memory),root.physicalBytes());
        EXPECT_THROW(moved.resize_uninitialized(1),std::logic_error);
    }
    original={};
    EXPECT_EQ(live(memory),0u);
    auto zero = CPUStorageRoot<float>(0,owner,cpu1).allocate(memory);
    EXPECT_EQ(zero.allocationBytes(),0u);
    EXPECT_THROW(zero.resize_uninitialized(1),std::logic_error);
}
TEST(CPUStorageRoot, RejectsInvalidDeclarationsForeignAuthorityAndExhaustion)
{
    EXPECT_THROW((CPUStorageRoot<float>(1,PhysicalMemoryOwner::Count,cpu1)),std::invalid_argument);
    EXPECT_THROW((CPUStorageRoot<float>(1,owner,DeviceId::cuda(0))),std::invalid_argument);
    const CPUStorageRoot<float> root(1,owner,cpu1);
    auto memory=authority(root.physicalBytes());
    EXPECT_THROW(root.allocate(nullptr),std::invalid_argument);
    EXPECT_THROW((CPUStorageRoot<float>(1,owner,DeviceId::cpu()).allocate(memory)),std::invalid_argument);
    auto storage=root.allocate(memory);
    EXPECT_THROW(root.allocate(memory),std::exception);
    EXPECT_EQ(live(memory),root.physicalBytes());
    storage={};
    EXPECT_EQ(live(memory),0u);
    EXPECT_EQ(root,(CPUStorageRoot<float>(1,owner,cpu1)));
    EXPECT_NE(root,(CPUStorageRoot<float>(1,owner,cpu1,CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized)));
}
TEST(CPUStorageRoot, Q8NativeViewCopyClaimsItsOwnPhysicalBytes)
{
    const auto bytes=nativeMatrixExtent<Q8_1Block>(2,33);
    const CPUStorageRoot<uint8_t> root(bytes,owner,cpu1);
    const size_t viewBytes=AlignedVector<uint8_t>::requiredAllocationBytes(nativeMatrixExtent<Q8_1Block>(1,33));
    auto memory=authority(root.physicalBytes()+viewBytes);
    auto parent=std::make_shared<Q8_1Tensor>(std::vector<size_t>{2,33},root.allocate(memory),cpu1);
    auto view=std::dynamic_pointer_cast<Q8_1Tensor>(parent->create_view({1,33},33));
    ASSERT_NE(view,nullptr);
    {
        Q8_1Tensor copy(*view);
        EXPECT_EQ(copy.home_device(),cpu1);
        EXPECT_EQ(live(memory),root.physicalBytes()+viewBytes);
    }
    EXPECT_EQ(live(memory),root.physicalBytes());
    parent.reset();view.reset();
    EXPECT_EQ(live(memory),0u);
}
TEST(CPUStorageRoot, Q16NativeBlockGeometriesRetainViewAndCopyOwnership)
{
    for(auto block : {Q16BlockSize::BLOCK_32,Q16BlockSize::BLOCK_64,Q16BlockSize::BLOCK_128})
    {
        const size_t blockBytes=block==Q16BlockSize::BLOCK_32?sizeof(Q16_1Block):block==Q16BlockSize::BLOCK_64?sizeof(Q16_1Block_64):sizeof(Q16_1Block_128);
        const auto bytes=nativeMatrixExtent(2,129,size_t(block),blockBytes);
        const auto rowBytes=nativeMatrixExtent(1,129,size_t(block),blockBytes);
        const CPUStorageRoot<uint8_t> root(bytes,owner,cpu1);
        auto memory=authority(root.physicalBytes()+AlignedVector<uint8_t>::requiredAllocationBytes(rowBytes));
        auto parent=std::make_shared<Q16_1Tensor>(std::vector<size_t>{2,129},root.allocate(memory),cpu1,block);
        auto view=std::dynamic_pointer_cast<Q16_1Tensor>(parent->create_view({1,129},129));
        ASSERT_NE(view,nullptr);
        EXPECT_EQ(static_cast<const uint8_t *>(view->raw_data())-static_cast<const uint8_t *>(parent->raw_data()),rowBytes);
        {
            Q16_1Tensor copy(*view);
            EXPECT_EQ(copy.home_device(),cpu1);
            EXPECT_EQ(copy.size_bytes(),rowBytes);
            EXPECT_EQ(live(memory),root.physicalBytes()+AlignedVector<uint8_t>::requiredAllocationBytes(rowBytes));
        }
        parent.reset();view.reset();
        EXPECT_EQ(live(memory),0u);
    }
}

TEST(CPUStorageRoot, GeometryUsesOneAllocatorForHeapAndDedicatedPages)
{
    for(size_t count : {0u,1u,63u,64u,65u,4095u,4096u,4097u})
    {
        const CPUStorageRoot<uint8_t> heap(count,owner,cpu1);
        const CPUStorageRoot<uint8_t> pages(count,owner,cpu1,CPUStoragePlacement::onNode(0));
        EXPECT_EQ(heap.physicalBytes(),AlignedVector<uint8_t>::requiredAllocationBytes(count));
        EXPECT_EQ(pages.physicalBytes(),AlignedVector<uint8_t>::requiredAllocationBytes(count,
            AlignedVector<uint8_t>::ALIGNMENT,AlignedVector<uint8_t>::StorageKind::AnonymousPageMapping));
    }
    EXPECT_THROW(CPUStoragePlacement::onNode(-1),std::invalid_argument);
}
