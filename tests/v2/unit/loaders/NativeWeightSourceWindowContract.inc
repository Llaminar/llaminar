/**
 * @file NativeWeightSourceWindowContract.inc
 * @brief Real common reader preserves native byte rectangles larger than its admitted arena.
 *
 * Included in the existing all-format original GGUF fixture. This tests source
 * I/O, allocation extent and byte ownership; it grants no prepared/native worker
 * readiness and does not substitute for model execution.
 */
TEST_P(ModelLoaderExpertSelectionTest, CommonNativeReaderStreamsBeyondCapWithExactOriginalBytes)
{
    rebuildProjectionSource(true,kTensorName,400);
    const auto* source_info=loader_->ggufDirectory()->findTensor(kTensorName);
    ASSERT_NE(source_info,nullptr);
    const auto source_mapping=loader_->getMmapRegion(source_info);ASSERT_TRUE(source_mapping);
    EXPECT_EQ(bufferedCPUJointMappedRSSKiB(*source_mapping),0u);
    const auto original=loader_->nativeWeightSourceGeometry(kTensorName);
    ASSERT_EQ(original.original_type,expectedTensorType(GetParam()));
    const auto placement=loader_->nativeWeightSourcePlacement();
    const size_t cap=placement.physicalBytes<uint8_t>(4u*original.native_group_bytes);
    ASSERT_GT(bytes_per_expert_,cap);
    const auto geometry=resolveNativeWeightLoadStaging(
        {original.native_group_bytes,0,0,columns_/original.native_group_elements},placement,cap);
    PhysicalMemoryPlanBuilder budget;
    budget.add({0,DeviceId::cpu(),cap,cap},PhysicalMemoryOwner::WeightLoadStaging,geometry.physical_bytes);
    const auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
    size_t windows=0,useful=0;
    {
        auto arena=CPUStorageRoot<uint8_t>(geometry.native_bytes,PhysicalMemoryOwner::WeightLoadStaging,
            DeviceId::cpu(),placement,CPUStorageInitialization::Uninitialized).allocate(memory);
        ASSERT_EQ(arena.allocationBytes(),geometry.physical_bytes);
        size_t ordinal=0;
        for(const size_t expert:{size_t{4},size_t{0},size_t{2}}) {
            const NativeWeightSourceWindow range{.original_expert=expert,.selected_expert_ordinal=ordinal++,
                .first_row=0,.rows=kRowsPerExpert,.first_column=0,.columns=columns_,
                .native_row_bytes=bytes_per_expert_/kRowsPerExpert};
            size_t source_cursor=0;
            forEachNativeWeightSourceWindow(original,range,geometry.native_bytes,[&](const auto& window) {
                const auto live=window.nativeBytes();
                ASSERT_LE(live,arena.size());
                std::fill(arena.begin(),arena.end(),uint8_t{0xcd});
                loader_->readNativeWeightSourceWindow(kTensorName,window,
                    {reinterpret_cast<std::byte*>(arena.data()),live});
                const size_t offset=window.first_row*range.native_row_bytes+
                    window.first_column/original.native_group_elements*original.native_group_bytes;
                EXPECT_EQ(offset,source_cursor);
                EXPECT_EQ(window.original_expert,expert);
                EXPECT_EQ(window.selected_expert_ordinal,range.selected_expert_ordinal);
                EXPECT_EQ(std::memcmp(arena.data(),source_bytes_.data()+expert*bytes_per_expert_+offset,live),0);
                for(size_t i=live;i<arena.size();++i)ASSERT_EQ(arena[i],0xcd);
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation),geometry.physical_bytes);
                source_cursor+=live;useful+=live;++windows;
            });
            EXPECT_EQ(source_cursor,bytes_per_expert_);
        }
        EXPECT_EQ(useful,3u*bytes_per_expert_);
        EXPECT_GT(windows,3u*kRowsPerExpert);
        auto invalid=NativeWeightSourceWindow{.original_expert=0,.first_row=0,.rows=1,
            .first_column=columns_,.columns=original.native_group_elements,.native_row_bytes=original.native_group_bytes};
        std::fill(arena.begin(),arena.end(),uint8_t{0xcd});
        EXPECT_THROW(loader_->readNativeWeightSourceWindow(kTensorName,invalid,
            {reinterpret_cast<std::byte*>(arena.data()),original.native_group_bytes}),std::invalid_argument);
        for(const auto byte:arena)EXPECT_EQ(byte,0xcd);
        invalid.first_column=0;invalid.rows=0;
        EXPECT_THROW(loader_->readNativeWeightSourceWindow(kTensorName,invalid,
            {reinterpret_cast<std::byte*>(arena.data()),original.native_group_bytes}),std::invalid_argument);
        for(const auto byte:arena)EXPECT_EQ(byte,0xcd);
        invalid.rows=1;invalid.original_expert=kExpertCount;
        EXPECT_THROW(loader_->readNativeWeightSourceWindow(kTensorName,invalid,
            {reinterpret_cast<std::byte*>(arena.data()),original.native_group_bytes}),std::invalid_argument);
        for(const auto byte:arena)EXPECT_EQ(byte,0xcd);
    }
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(bufferedCPUJointMappedRSSKiB(*source_mapping),0u);
}

/** @test Rank2 parser normalization cannot replace original non-square K/N source coordinates. */
TEST_P(ModelLoaderExpertSelectionTest, CommonNativeReaderRestoresNonSquareOrdinaryAxes)
{
    constexpr const char* name="blk.7.ffn_gate.weight";
    rebuildProjectionSource(true,name,400,2);
    const auto original=loader_->nativeWeightSourceGeometry(name);
    EXPECT_EQ(original.original_shape,(std::vector<size_t>{columns_,kRowsPerExpert}));
    const auto normalized=loader_->getTensorShape(name);ASSERT_TRUE(normalized);
    EXPECT_EQ(*normalized,(std::vector<size_t>{kRowsPerExpert,columns_}));
    const auto* info=loader_->ggufDirectory()->findTensor(name);ASSERT_NE(info,nullptr);
    const auto mapping=loader_->getMmapRegion(info);ASSERT_TRUE(mapping);
    EXPECT_EQ(bufferedCPUJointMappedRSSKiB(*mapping),0u);
    const auto placement=loader_->nativeWeightSourcePlacement();
    const size_t row_bytes=bytes_per_expert_/kRowsPerExpert;
    for(const size_t useful_capacity:{4u*original.native_group_bytes,bytes_per_expert_}) {
        const size_t cap=placement.physicalBytes<uint8_t>(useful_capacity);
        const auto geometry=resolveNativeWeightLoadStaging({original.native_group_bytes,0,0,
            bytes_per_expert_/original.native_group_bytes},placement,cap);
        PhysicalMemoryPlanBuilder budget;
        budget.add({0,DeviceId::cpu(),cap,cap},PhysicalMemoryOwner::WeightLoadStaging,geometry.physical_bytes);
        const auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
        size_t cursor=0,windows=0;
        {
            auto arena=CPUStorageRoot<uint8_t>(geometry.native_bytes,PhysicalMemoryOwner::WeightLoadStaging,
                DeviceId::cpu(),placement,CPUStorageInitialization::Uninitialized).allocate(memory);
            const NativeWeightSourceWindow range{.first_row=0,.rows=kRowsPerExpert,
                .first_column=0,.columns=columns_,.native_row_bytes=row_bytes};
            forEachNativeWeightSourceWindow(original,range,geometry.native_bytes,[&](const auto& window) {
                const size_t live=window.nativeBytes();ASSERT_LE(live,arena.size());
                std::fill(arena.begin(),arena.end(),uint8_t{0xcd});
                loader_->readNativeWeightSourceWindow(name,window,{reinterpret_cast<std::byte*>(arena.data()),live});
                const size_t offset=window.first_row*row_bytes+
                    window.first_column/original.native_group_elements*original.native_group_bytes;
                EXPECT_EQ(offset,cursor);
                EXPECT_EQ(std::memcmp(arena.data(),source_bytes_.data()+offset,live),0);
                for(size_t i=live;i<arena.size();++i)ASSERT_EQ(arena[i],0xcd);
                cursor+=live;++windows;
            });
            EXPECT_EQ(cursor,bytes_per_expert_);
            if(useful_capacity<row_bytes)EXPECT_GT(windows,kRowsPerExpert);
            else EXPECT_EQ(windows,1u);
        }
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
    EXPECT_EQ(bufferedCPUJointMappedRSSKiB(*mapping),0u);
}
