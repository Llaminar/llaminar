/**
 * @file PreparedCPUWeightTensorLifecycle.inc
 * @brief Actual streamed Store commit retains final packing after native source retirement.
 *
 * These device-free tests construct no metadata readiness substitute: only the
 * production Store writer can create the prepared-only tensor. CPU ordinals
 * identify allocation resources; they do not claim physical socket placement.
 */
#include "tensors/PreparedCPUWeightTensor.h"
#include "tensors/NativeWeightSourceRange.h"
#include "memory/CPUStorageRoot.h"
#include "../../utils/CPUPreparedStoreTestAdmission.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include <type_traits>

static_assert(!std::is_constructible_v<PreparedCPUWeightTensor,
    std::shared_ptr<ITensorGemm>,std::vector<size_t>,TensorType,DeviceId>);
static_assert(!std::is_base_of_v<IINT8Unpackable,PreparedCPUWeightTensor>);

TEST(PreparedCPUWeightTensorLifecycle, StoreCommitRetainsOnlyActualPreparedRoot)
{
    for(const auto device:{DeviceId::cpu(),DeviceId{DeviceType::CPU,1}}) {
        auto source=makeQ8_0Tensor(3,64);
        const CPUPreparedWeightSource spec{
            .original={{64,3},TensorType::Q8_0,32,34},
            .range={.rows=3,.columns=64,.native_row_bytes=68}};
        const auto request=quoteCPUWeightPreparation(TensorType::Q8_0,3,64,
            WeightPrecision::NATIVE,CPUWeightConsumer::PreparedGemm);
        ASSERT_TRUE(request);
        const auto geometry=resolveNativeWeightLoadStaging(*request,CPUStoragePlacement::local(),std::nullopt);
        const auto packed_bytes=cpuPreparedStoreTestBytes(*source);
        PhysicalMemoryPlanBuilder budget;
        const auto total=packed_bytes+geometry.physical_bytes;
        const PhysicalMemoryResource resource{0,device,total,total};
        budget.add(resource,PhysicalMemoryOwner::PrimaryModelWeights,packed_bytes);
        budget.add(resource,PhysicalMemoryOwner::WeightLoadStaging,geometry.physical_bytes);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
        PreparedWeightStore store(ModelContextId{99});
        store.installPhysicalMemoryAuthority(memory);
        auto input=makeStoreBinding(991,"blk.0.ffn_gate.weight",device);
        auto transaction=store.beginCPUPreparation(input,spec,PhysicalMemoryOwner::PrimaryModelWeights,
            CPUStoragePlacement::local());
        {
            auto arena=CPUStorageRoot<uint8_t>(geometry.scratch_offset+geometry.scratch_bytes,
                PhysicalMemoryOwner::WeightLoadStaging,device,CPUStoragePlacement::local(),
                CPUStorageInitialization::Uninitialized).allocate(memory);
            forEachNativeWeightSourceWindow(spec.original,spec.range,geometry.native_bytes,[&](const auto& window) {
                auto* bytes=reinterpret_cast<std::byte*>(arena.data());
                const auto* original=static_cast<const std::byte*>(source->raw_data());
                std::memcpy(bytes,original+window.first_row*68+window.first_column/32*34,window.nativeBytes());
                transaction->append(window,{bytes,window.nativeBytes()},
                    {bytes+geometry.scratch_offset,geometry.scratch_bytes});
            });
        }
        auto committed=store.finishCPUPreparation(std::move(transaction));
        auto metadata=std::dynamic_pointer_cast<PreparedCPUWeightTensor>(committed.tensor_owner);
        ASSERT_TRUE(metadata);
        EXPECT_EQ(metadata->home_device(),device);
        EXPECT_EQ(metadata->shape(),(std::vector<size_t>{3,64}));
        EXPECT_EQ(metadata->native_type(),TensorType::Q8_0);
        EXPECT_EQ(metadata->size_bytes(),0u);
        EXPECT_EQ(metadata->raw_data(),nullptr);
        EXPECT_FALSE(committed.residency.raw_host_data_available);
        EXPECT_EQ(metadata->raw_mutable_data(),nullptr);
        EXPECT_EQ(IINT8Unpackable::fromTensor(metadata.get()),nullptr);
        auto engine=store.gemmKernelLifetime(*committed.prepared);
        ASSERT_TRUE(engine);
        // gemmKernelLifetime aliases its Store handle, while metadata retains
        // the actual kernel control block. Borrow the exact live kernel under
        // metadata and prove its physical root instead of comparing wrappers.
        auto* metadata_engine=engine.get();
        const auto live_packed_bytes=[&]{return memory->claimedBytes(device,PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation);};
        EXPECT_EQ(live_packed_bytes(),packed_bytes);
        EXPECT_TRUE(metadata->hasPreparedDeviceState());
        CPUProjectionTestWorkspace workspace(1,64,cpuProjectionTestRequirements(1,{engine.get()}));
        FP32Tensor hidden({1,64}),before({1,3}),after({1,3});
        std::fill_n(hidden.mutable_data(),64,0.25f);
        engine->multiply_tensor(&hidden,&before,1,3,64,true,1.f,0.f,nullptr,nullptr,-1,workspace.get());
        source.reset();
        std::array<float,192> denied;denied.fill(123.f);
        EXPECT_THROW(metadata->to_fp32(denied.data()),std::logic_error);
        EXPECT_TRUE(std::all_of(denied.begin(),denied.end(),[](float v){return v==123.f;}));
        EXPECT_THROW(metadata->createGemm(),std::logic_error);
        EXPECT_THROW(metadata->create_view({1,64}),std::logic_error);
        EXPECT_THROW(metadata->ensureOnHost(),std::logic_error);
        metadata->release_raw_data();metadata->release_host_weight_data();
        store.clear();committed={};engine.reset();
        EXPECT_EQ(live_packed_bytes(),packed_bytes);
        EXPECT_TRUE(metadata->hasPreparedDeviceState());
        metadata_engine->multiply_tensor(&hidden,&after,1,3,64,true,1.f,0.f,nullptr,nullptr,-1,workspace.get());
        EXPECT_EQ(std::memcmp(before.data(),after.data(),3*sizeof(float)),0);
        metadata.reset();
        EXPECT_EQ(live_packed_bytes(),0u);
    }
}

TEST(PreparedCPUWeightTensorLifecycle, IncompleteWriterCannotPublishMetadata)
{
    auto source=makeQ8_0Tensor(3,64);
    PreparedWeightStore store(ModelContextId{99});
    admitCPUPreparedStoreForTest(store,{source.get()});
    const CPUPreparedWeightSource spec{.original={{64,3},TensorType::Q8_0,32,34},
        .range={.rows=3,.columns=64,.native_row_bytes=68}};
    auto transaction=store.beginCPUPreparation(makeStoreBinding(992,"blk.0.ffn_gate.weight",DeviceId::cpu()),
        spec,PhysicalMemoryOwner::PrimaryModelWeights,CPUStoragePlacement::local());
    EXPECT_THROW(store.finishCPUPreparation(std::move(transaction)),std::logic_error);
    EXPECT_FALSE(store.preparedRefForBinding(992,DeviceId::cpu()).has_value());
}

TEST(PreparedCPUWeightTensorLifecycle, StoreCommitPublishesActualFinalRawAvailability)
{
    for(const auto device:{DeviceId::cpu(),DeviceId{DeviceType::CPU,1}}) {
        for(const auto consumer:{CPUWeightConsumer::ResidentTensor,CPUWeightConsumer::PreparedGemm}) {
            constexpr size_t rows=3,columns=64,elements=rows*columns;
            const auto physical=CPUStoragePlacement::local().physicalBytes<float>(elements);
            PhysicalMemoryPlanBuilder budget;const PhysicalMemoryResource resource{0,device,physical,physical};
            budget.add(resource,PhysicalMemoryOwner::PrimaryModelWeights,physical);
            auto memory=std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
            PreparedWeightStore store(ModelContextId{99});store.installPhysicalMemoryAuthority(memory);
            auto original=makeStoreBinding(993,"blk.0.attn_q.weight",device);
            original.residency.raw_host_data_available=false;
            CPUPreparedWeightSource source{
                .original={{columns,rows},TensorType::FP32,1,sizeof(float)},
                .range={.rows=rows,.columns=columns,.native_row_bytes=columns*sizeof(float)},
                .consumer=consumer};
            auto transaction=store.beginCPUPreparation(original,source,PhysicalMemoryOwner::PrimaryModelWeights,CPUStoragePlacement::local());
            EXPECT_FALSE(transaction->stagingRequest());
            std::array<float,elements> input{};for(size_t i=0;i<elements;++i)input[i]=float(i)*0.03125f;
            transaction->readDirect(source.range,[&](std::span<std::byte> target){
                ASSERT_EQ(target.size(),sizeof(input));std::memcpy(target.data(),input.data(),target.size());
            });
            auto committed=store.finishCPUPreparation(std::move(transaction));
            EXPECT_TRUE(committed.residency.raw_host_data_available);
            ASSERT_NE(committed.tensor_owner,nullptr);
            EXPECT_EQ(committed.tensor_owner->hasPreparedDeviceState(),consumer==CPUWeightConsumer::PreparedGemm);EXPECT_EQ(committed.tensor_owner->home_device(),device);
            EXPECT_FALSE(committed.tensor_owner->is_raw_data_released());ASSERT_NE(committed.tensor_owner->raw_data(),nullptr);
            EXPECT_EQ(0,std::memcmp(committed.tensor_owner->raw_data(),input.data(),sizeof(input)));
            const auto stored=store.binding(*committed.prepared);ASSERT_TRUE(stored);
            EXPECT_TRUE(stored->residency.raw_host_data_available);
            EXPECT_EQ(stored->tensor->hasPreparedDeviceState(),consumer==CPUWeightConsumer::PreparedGemm);
            if(consumer==CPUWeightConsumer::PreparedGemm) {
                auto engine=store.gemmKernelLifetime(*committed.prepared);ASSERT_TRUE(engine);
                EXPECT_FALSE(engine->canReleaseSourceWeightTensor());EXPECT_EQ(engine->packedWeightBytes(),0u);
            } else EXPECT_EQ(committed.prepared->kind,PreparedWeightKind::CPUResidentTensor);
        }
    }
}
