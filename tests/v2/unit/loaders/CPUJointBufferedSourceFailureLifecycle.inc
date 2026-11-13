/**
 * @file CPUJointBufferedSourceFailureLifecycle.inc
 * @brief Real split-GGUF CPU source windows, mapped-page isolation and failed-read retirement.
 *
 * This follower uses the actual loader and original source-derived CPU cohort.
 * Linux smaps observes only the unique original source VMAs; file-buffer page
 * cache residency is deliberately distinct from mmap PTE residency.
 */
namespace {
/** @brief Own exactly this test's fresh two-file source fixture. */
struct BufferedCPUJointFiles {
    static constexpr size_t H=8,F=20,E=512;
    std::filesystem::path directory;
    std::array<std::filesystem::path,2> files;
    /** @brief Create two genuine original GGUF shards with distinct role/expert bytes. */
    BufferedCPUJointFiles() {
        std::string pattern=(std::filesystem::temp_directory_path()/"llaminar_cpu_buffered_XXXXXX").string();
        auto* created=::mkdtemp(pattern.data());
        if(!created)throw std::runtime_error("Cannot create private buffered source fixture");
        directory=created;files={directory/"weights-00001-of-00002.gguf",directory/"weights-00002-of-00002.gguf"};
        try {write(0,{0});write(1,{1,2});}
        catch(...) {std::error_code error;std::filesystem::remove_all(directory,error);throw;}
    }
    /** @brief Remove only the original private fixture after loader/mappings retire. */
    ~BufferedCPUJointFiles(){std::error_code error;std::filesystem::remove_all(directory,error);}
    /** @return Exact finite source value, distinct for every role/expert/native element. */
    static float value(size_t role,size_t expert,size_t element) {
        return float(role*1000000+expert*1000+element);
    }
    /** @brief Serialize native FP32 source bytes and standard split metadata. */
    void write(size_t shard,std::initializer_list<size_t> roles) {
        std::ofstream stream(files[shard],std::ios::binary|std::ios::trunc);
        if(!stream)throw std::runtime_error("Cannot create original GGUF split");
        stream.write("GGUF",4);writeScalar(stream,uint32_t{3});writeScalar(stream,uint64_t(roles.size()));writeScalar(stream,uint64_t{2});
        writeString(stream,"split.count");writeScalar(stream,uint32_t{4});writeScalar(stream,uint32_t{2});
        writeString(stream,"split.no");writeScalar(stream,uint32_t{4});writeScalar(stream,uint32_t(shard));
        size_t offset=0;
        for(const auto role:roles) {
            writeString(stream,name(role));writeScalar(stream,uint32_t{3});
            writeScalar(stream,uint64_t(role==2?F:H));writeScalar(stream,uint64_t(role==2?H:F));
            writeScalar(stream,uint64_t{E});writeScalar(stream,uint32_t{0});writeScalar(stream,uint64_t(offset));
            offset+=E*F*H*sizeof(float);
        }
        const auto end=size_t(stream.tellp()),aligned=(end+31)/32*32;
        std::vector<char> padding(aligned-end,0);stream.write(padding.data(),std::streamsize(padding.size()));
        for(const auto role:roles)for(size_t expert=0;expert<E;++expert)for(size_t element=0;element<F*H;++element)
            writeScalar(stream,value(role,expert,element));
        if(!stream)throw std::runtime_error("Cannot finish original GGUF split");
    }
    /** @return Original semantic tensor name, never a compact-window alias. */
    static std::string name(size_t role) {
        return std::string("blk.7.ffn_")+(role==0?"gate":role==1?"up":"down")+"_exps.weight";
    }
};
/** @return One canonical loader-derived original sparse selection (no fabricated source authority). */
std::shared_ptr<const CPUExpertShardSourceSelection> bufferedCPUJointSelection(ModelLoader& loader) {
    auto plan=localCPUJointSourceTestPlan();
    for(size_t expert=0;expert<512;++expert)plan.placements.front().routed_expert_tier[expert]=expert%8==0?0:1;
    const auto runtime=resolveMoEExpertOverlayRuntimePlan(std::make_shared<const MoERoutedExpertPlacementPlan>(plan),
        {.current_world_rank=0});
    if(!runtime)throw std::runtime_error("Original buffered source plan is absent");
    const auto prepared=MoEExpertOverlayPreparationPlan::build(*runtime,loader).filteredForDevice(DeviceId::cpu());
    if(prepared.requests().empty() || !prepared.requests().front().cpu_joint_source)
        throw std::runtime_error("Original buffered source selection is absent");
    return prepared.requests().front().cpu_joint_source;
}
/** @return Original PMA staging authority quoted by the production allocator geometry. */
std::shared_ptr<PhysicalMemoryAuthority> bufferedCPUJointMemory(ModelLoader& loader,const CPUExpertShardSourceSelection& selection) {
    const auto native=loader.nativeWeightSourceGeometry(BufferedCPUJointFiles::name(0));
    const auto projection=selection.ownership().projection(WeightRole::MoEExpertGate);
    const NativeWeightSourceWindow range{.original_expert=size_t(selection.expertIds().front()),
        .rows=size_t(projection.rows),.columns=size_t(projection.columns),
        .native_row_bytes=size_t(projection.columns)/native.native_group_elements*native.native_group_bytes};
    const auto geometry=resolveNativeWeightLoadStaging(nativeReadRequestForTest(native,range),
        CPUStoragePlacement::onNode(selection.participant().address.numa_node),size_t{4096});
    PhysicalMemoryPlanBuilder builder;
    builder.add({0,selection.participant().device,geometry.physical_bytes,geometry.physical_bytes},
        PhysicalMemoryOwner::WeightLoadStaging,geometry.physical_bytes);
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
}
/** @return Resident mapped KiB for the exact original unique-file VMA, excluding file-cache residency. */
size_t bufferedCPUJointMappedRSSKiB(const MmapRegion& original) {
    std::ifstream smaps("/proc/self/smaps");
    if(!smaps)throw std::runtime_error("CPU buffered source regression requires readable /proc/self/smaps");
    const auto address=reinterpret_cast<uintptr_t>(original.data());
    std::string line;bool selected=false;
    while(std::getline(smaps,line)) {
        if(line.find('-')!=std::string::npos) {
            std::istringstream header(line);uintptr_t begin=0,end=0;char separator=0;
            if(header>>std::hex>>begin>>separator>>end;header && separator=='-')selected=begin<=address && address<end;
        }
        if(selected && line.rfind("Rss:",0)==0) {
            std::istringstream field(line.substr(4));size_t kib=0;std::string units;
            if(!(field>>kib>>units) || units!="kB")throw std::runtime_error("Malformed source VMA Rss observation");
            return kib;
        }
    }
    throw std::runtime_error("Original source VMA is absent from smaps");
}
}
/** @test CPU-owned windows read main/split streams without faulting either original mmap. */
TEST(CPUJointBufferedSource, SplitOriginalBytesAndMappedPagesRemainUnfaulted) {
    BufferedCPUJointFiles files;const auto mpi=std::make_shared<MPIContext>(0,1,MPI_COMM_NULL);
    TensorFactory factory(*mpi);ModelLoader loader(&factory);
    loader.setPayloadAccessPattern(ModelPayloadAccessPattern::SparseSelection);ASSERT_TRUE(loader.loadModel(files.files[0].string()));
    const auto selected=bufferedCPUJointSelection(loader);ASSERT_EQ(selected->expertIds().size(),448u);
    EXPECT_EQ(selected->ownership().geometry().experts,512);
    const auto memory=bufferedCPUJointMemory(loader,*selected);
    std::array<std::shared_ptr<MmapRegion>,2> regions;
    regions[0]=loader.getMmapRegion(loader.ggufDirectory()->findTensor(BufferedCPUJointFiles::name(0)));
    regions[1]=loader.getMmapRegion(loader.ggufDirectory()->findTensor(BufferedCPUJointFiles::name(1)));
    for(const auto& region:regions){ASSERT_TRUE(region);EXPECT_EQ(bufferedCPUJointMappedRSSKiB(*region),0u);}
    for(size_t role=0;role<3;++role) {
        const auto name=BufferedCPUJointFiles::name(role);const auto* info=loader.ggufDirectory()->findTensor(name);ASSERT_NE(info,nullptr);
        EXPECT_EQ(info->split_idx,role?1:0);
        const auto weight_role=role==0?WeightRole::MoEExpertGate:role==1?WeightRole::MoEExpertUp:WeightRole::MoEExpertDown;
        const auto native=loader.nativeWeightSourceGeometry(name);
        const auto original_request=nativeJointRequirementForTest(selected,weight_role,name);
        const auto original_ranges=projectNativeWeightSource(original_request,native);
        ASSERT_EQ(original_ranges.size(),selected->expertIds().size());
        for(const size_t begin:{size_t{0},size_t{17},size_t{445}}) {
            // Prefixes retain original sparse E identity. Generic windows own
            // no Tensor parent; a metadata copy cannot prolong raw arena life.
            for(size_t ordinal=begin;ordinal<std::min(begin+17,original_ranges.size());++ordinal) {
                const auto range=original_ranges[ordinal].range;
                EXPECT_EQ(range.original_expert,size_t(selected->expertIds()[ordinal]));
                const auto visits=visitNativeSourceForTest(loader,name,range,nativeReadRequestForTest(native,range),
                    CPUStoragePlacement::onNode(selected->participant().address.numa_node),memory,
                    selected->participant().device,size_t{4096},[&](const auto& window,auto bytes,auto) {
                        const auto* data=reinterpret_cast<const float*>(bytes.data());
                        const size_t full_k=role==2?BufferedCPUJointFiles::F:BufferedCPUJointFiles::H;
                        for(size_t n=0;n<window.rows;++n)for(size_t k=0;k<window.columns;++k)
                            EXPECT_EQ(data[n*window.columns+k],BufferedCPUJointFiles::value(role,
                                window.original_expert,(window.first_row+n)*full_k+window.first_column+k));
                        EXPECT_LE(memory->claimedBytes(selected->participant().device,PhysicalMemoryOwner::WeightLoadStaging,
                            PhysicalMemoryMaterializationKind::NewAllocation),4096u);
                    });
                EXPECT_GT(visits,0u);
                EXPECT_EQ(memory->claimedBytes(selected->participant().device,PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation),0u);
            }
        }
    }
    for(const auto& region:regions)EXPECT_EQ(bufferedCPUJointMappedRSSKiB(*region),0u);
}
/** @test Actual short main/split reads fail and release the pre-existing original raw staging claim. */
TEST(CPUJointBufferedSource, TruncatedMainAndSplitReadReleaseOriginalStaging) {
    for(const size_t shard:{size_t{0},size_t{1}}) {
        SCOPED_TRACE(shard);BufferedCPUJointFiles files;
        const auto mpi=std::make_shared<MPIContext>(0,1,MPI_COMM_NULL);TensorFactory factory(*mpi);ModelLoader loader(&factory);
        // Error cleanup is independent of mmap SIGBUS behavior; this explicitly
        // exercises the same existing buffered stream with no mapped payload.
        loader.setUseMmap(false);ASSERT_TRUE(loader.loadModel(files.files[0].string()));
        const auto selected=bufferedCPUJointSelection(loader);const auto memory=bufferedCPUJointMemory(loader,*selected);
        std::filesystem::resize_file(files.files[shard],0);
        try {
            const auto name=BufferedCPUJointFiles::name(shard);
            const auto native=loader.nativeWeightSourceGeometry(name);
            const auto request=nativeJointRequirementForTest(selected,
                shard?WeightRole::MoEExpertUp:WeightRole::MoEExpertGate,name);
            const auto range=projectNativeWeightSource(request,native).back().range;
            (void)visitNativeSourceForTest(loader,name,range,nativeReadRequestForTest(native,range),
                CPUStoragePlacement::onNode(selected->participant().address.numa_node),memory,
                selected->participant().device,size_t{4096},[](const auto&,auto,auto){});
            FAIL()<<"Truncated original source unexpectedly loaded";
        } catch(const std::runtime_error& failure) {
            EXPECT_NE(std::string(failure.what()).find("buffered"),std::string::npos)<<failure.what();
        }
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
