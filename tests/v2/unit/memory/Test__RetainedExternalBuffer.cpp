/**
 * @file Test__RetainedExternalBuffer.cpp
 * @brief Device-free actual arena external-root lifetime, rebind and failed-registration regressions.
 *
 * Aliasing Tensor handles retain their original physical owner without an arena
 * allocation/claim. These tests require a freshly compiled arena/core ABI.
 */
#include "memory/BufferArena.h"
#include "tensors/TensorClasses.h"
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <stdexcept>
namespace llaminar2::test {
namespace {
/** @brief Model/source-style owner whose canonical Tensor alias shares the original control block. */
struct RootOwner {
    std::unique_ptr<INT8Tensor> tensor=std::make_unique<INT8Tensor>(std::vector<std::size_t>{128});
    int* retired;
    explicit RootOwner(int& count):retired(&count){}
    ~RootOwner(){++*retired;}
};
/** @brief Inject metadata failure before any slot commit; no device/backend activity occurs. */
class ThrowingShape final:public INT8Tensor {
public:
    ThrowingShape():INT8Tensor({128}){}
    const std::vector<std::size_t>& shape()const override{throw std::runtime_error("authored metadata failure");}
};
/** @brief Canonical metadata only; this tensor creates no GPU allocation, stream, backend or device query. */
class FakeDeviceRoot final:public INT8Tensor {
public:
    /** @brief Publish authored canonical state for a host-only interface replay. */
    FakeDeviceRoot(TensorCoherenceState state,std::optional<DeviceId> device):INT8Tensor({128}),state(state),device(device){}
    /** @return Authored canonical tensor metadata, never an arena-maintained mirror. */
    TensorCoherenceState coherenceState()const override{return state;}
    /** @return Exact fake allocation identity without consulting a GPU runtime. */
    std::optional<DeviceId> current_device()const override{return device;}
    TensorCoherenceState state;std::optional<DeviceId> device;
};

}
TEST(RetainedExternalBuffer, OriginalAliasingOwnerSurvivesCallerAndStageReleaseUntilArenaDestruction)
{
    int retired=0;std::weak_ptr<RootOwner> weak;
    {
        BufferArena arena;auto owner=std::make_shared<RootOwner>(retired);weak=owner;
        std::shared_ptr<TensorBase> alias(owner,owner->tensor.get());const auto* original=alias.get();
        ASSERT_TRUE(arena.registerExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,alias));
        owner.reset();alias.reset();
        EXPECT_FALSE(weak.expired());EXPECT_EQ(retired,0);
        auto acquired=arena.getSharedTensor(BufferId::QFN_POOLED_OWNER_FRAME);
        ASSERT_NE(acquired,nullptr);EXPECT_EQ(acquired.get(),original);acquired.reset();
        EXPECT_EQ(arena.getTensor(BufferId::QFN_POOLED_OWNER_FRAME),original);
    }
    EXPECT_TRUE(weak.expired());EXPECT_EQ(retired,1);
}
TEST(RetainedExternalBuffer, RebindAndDuplicateRegistrationPreserveTheOriginalRetainedOwner)
{
    BufferArena arena;auto first=std::make_shared<INT8Tensor>(std::vector<std::size_t>{128});
    auto second=std::make_shared<INT8Tensor>(std::vector<std::size_t>{128});
    ASSERT_TRUE(arena.registerExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,std::static_pointer_cast<TensorBase>(first)));
    EXPECT_FALSE(arena.bindExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,first.get()));
    EXPECT_FALSE(arena.bindExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,second.get()));
    EXPECT_FALSE(arena.registerExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,std::static_pointer_cast<TensorBase>(second)));
    std::weak_ptr<TensorBase> rejected=second;second.reset();EXPECT_TRUE(rejected.expired());
    auto actual=arena.getSharedTensor(BufferId::QFN_POOLED_OWNER_FRAME);
    EXPECT_EQ(actual.get(),first.get());EXPECT_FALSE(actual.owner_before(first));EXPECT_FALSE(first.owner_before(actual));
}
TEST(RetainedExternalBuffer, FailedRegistrationPublishesNoSlotAndRetainsNoRejectedOwner)
{
    BufferArena arena;
    EXPECT_FALSE(arena.registerExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,std::shared_ptr<TensorBase>{}));
    EXPECT_FALSE(arena.isRegistered(BufferId::QFN_POOLED_OWNER_FRAME));
    auto invalid=std::make_shared<INT8Tensor>(std::vector<std::size_t>{128});std::weak_ptr<TensorBase> weak=invalid;
    EXPECT_FALSE(arena.registerExternalBuffer(BufferId::_COUNT,std::static_pointer_cast<TensorBase>(invalid)));
    invalid.reset();EXPECT_TRUE(weak.expired());
    auto throwing=std::make_shared<ThrowingShape>();std::weak_ptr<TensorBase> failed=throwing;
    EXPECT_THROW(arena.registerExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,std::static_pointer_cast<TensorBase>(throwing)),std::runtime_error);
    EXPECT_FALSE(arena.isRegistered(BufferId::QFN_POOLED_OWNER_FRAME));EXPECT_EQ(arena.getTensor(BufferId::QFN_POOLED_OWNER_FRAME),nullptr);
    throwing.reset();EXPECT_TRUE(failed.expired());
}

/** @test Registration, introspection and host preparation follow the sole tensor state for both GPU vendors. */
TEST(RetainedExternalBuffer, CoherenceProjectsCanonicalTensorWithoutALiveArenaAuthority)
{
    for(const auto device:{DeviceId::cuda(3),DeviceId::rocm(4)})
        for(const auto initial:{TensorCoherenceState::HOST_ONLY,TensorCoherenceState::HOST_AUTHORITATIVE,
            TensorCoherenceState::DEVICE_AUTHORITATIVE,TensorCoherenceState::SYNCED,TensorCoherenceState::MAPPED}){
            BufferArena arena;auto root=std::make_shared<FakeDeviceRoot>(initial,device);
            ASSERT_TRUE(arena.registerExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,std::static_pointer_cast<TensorBase>(root)));
            auto projected=arena.getCoherenceState(BufferId::QFN_POOLED_OWNER_FRAME);
            EXPECT_EQ(projected.authority,initial==TensorCoherenceState::DEVICE_AUTHORITATIVE?CoherenceState::DEVICE:CoherenceState::HOST);
            if(initial==TensorCoherenceState::DEVICE_AUTHORITATIVE){EXPECT_EQ(projected.authoritative_device,device);EXPECT_FALSE(projected.needsTransferTo(device));}
            EXPECT_EQ(projected.needsTransferTo(device,initial),::llaminar2::needsHostToDeviceUpload(initial));
            root->state=TensorCoherenceState::DEVICE_AUTHORITATIVE;
            EXPECT_EQ(arena.getCoherenceState(BufferId::QFN_POOLED_OWNER_FRAME).authoritative_device,device);
            root->state=TensorCoherenceState::HOST_AUTHORITATIVE;
            EXPECT_EQ(arena.getCoherenceState(BufferId::QFN_POOLED_OWNER_FRAME).authority,CoherenceState::HOST);
            // A direct canonical host publication cannot leave stale arena DEVICE
            // history that trips the legacy tracker invariant or performs D2H.
            EXPECT_TRUE(arena.prepareForRead(BufferId::QFN_POOLED_OWNER_FRAME,DeviceId::cpu(),nullptr));
            root->state=TensorCoherenceState::SYNCED;
            EXPECT_FALSE(arena.getCoherenceState(BufferId::QFN_POOLED_OWNER_FRAME).needsTransferTo(device,root->state));
            root->state=TensorCoherenceState::MAPPED;
            EXPECT_FALSE(arena.getCoherenceState(BufferId::QFN_POOLED_OWNER_FRAME).needsTransferTo(DeviceId::cpu(),root->state));
        }
}
/** @test Invalid canonical metadata rejects before retaining a root or publishing any arena slot. */
TEST(RetainedExternalBuffer, CoherenceFailurePublishesNoSlotAndRetainsNoRoot)
{
    for(const auto state:{TensorCoherenceState::DEVICE_AUTHORITATIVE,TensorCoherenceState::INVALID}){
        BufferArena arena;auto root=std::make_shared<FakeDeviceRoot>(state,std::nullopt);std::weak_ptr<TensorBase> weak=root;
        EXPECT_THROW(arena.registerExternalBuffer(BufferId::QFN_POOLED_OWNER_FRAME,std::static_pointer_cast<TensorBase>(root)),std::invalid_argument);
        EXPECT_FALSE(arena.isRegistered(BufferId::QFN_POOLED_OWNER_FRAME));root.reset();EXPECT_TRUE(weak.expired());
    }
}
}
