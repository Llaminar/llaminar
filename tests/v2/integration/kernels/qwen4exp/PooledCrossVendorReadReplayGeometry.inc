/** @file PooledCrossVendorReadReplayGeometry.inc
 * @brief Same authoritative cross-vendor plan consumed by native materialization and device-free admission replay.
 */
/** @brief Complete authored cross-vendor case, declared before the original selected worker population. */
struct CrossVendorReadReplaySpec final {
    std::shared_ptr<ProviderFile> file;
    std::uint32_t slots,maximum=64,unique=16,ramSlots=2;
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    bool authoredRemote=false,handshake=false;
#endif
    /** @brief Reject incomplete geometry before source preparation or any capture owner exists. */
    void requireComplete()const{
        requireProvider(file && maximum>=16 && !(maximum%16) && unique && unique<=maximum && unique<=file->rows && slots<=file->rows,
            "Cross-vendor program requires its complete original source/row geometry");
    }
};
/** @brief Authoritative actual declaration; no synthetic physical capacity is introduced. */
struct CrossVendorReadReplayGeometry final {
    ImmutableRowTableLayoutGeometry layout;
    q::PooledImmutablePhysicalOwnerGeometry physical;
    q::PooledImmutableCoordinatorGeometry policy;
    CpuAdaptiveImmutableRowCacheGeometry cpuGeometry;
    q::ImmutableLookupLayout channel;
    std::array<CPUTensorStorageDeclaration,5> declarations;
    std::size_t activation;
    PhysicalMemoryPlan payload,retained;
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    std::optional<PooledRemoteGraphProgram> remoteProgram;
#endif
    /** @return Original roots/channels and separately retained graph/event obligations before final admission. */
    static CrossVendorReadReplayGeometry create(const CrossVendorReadReplaySpec& spec,DeviceId sourceDevice,DeviceId ownerDevice,const PreparedNativeLookupFixture& session,const RetainedHostServiceProgram& ioProgram,
        const NativeOrderingEventBudgetProjection& ordering=NativeOrderingEventBudgetProjection::installed()){
        spec.requireComplete();
        const PhysicalMemoryAllocatorIdentity sourceId{0,sourceDevice},ownerId{0,ownerDevice},hostId{0,DeviceId::cpu()};
        const auto& file_=*spec.file;const auto Maximum=spec.maximum,Unique=spec.unique,slots=spec.slots,ramSlots=spec.ramSlots;
        const auto source=file_.descriptor();const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
            ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
            {{{ownerId,ImmutableRowStorage::VRAM},0,slots,{}},{{hostId,ImmutableRowStorage::RAM},0,ramSlots,{}}},{sourceId},{},
            {{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},slots?std::optional(sourceId):std::nullopt,Unique);
        const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{ownerId},sourceId);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},Maximum);
        const auto cpuGeometry=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,sourceId);
        const auto src=session.resource(sourceDevice),own=session.resource(ownerDevice),host=session.resource(DeviceId::cpu());
        PhysicalMemoryPlanBuilder bom;const auto add=[&](const PhysicalMemoryPlan& plan){for(const auto& resource:plan.resources())bom.add(resource);};
        add(cpuGeometry.bom(host));add(physical.cacheBOM(0,own));add(physical.mandatoryBOM(0,own));add(policy.policyBOM(src));add(policy.mandatoryBOM(src));
        add(q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,0,own));add(q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,src));
        add(q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,source,src));

        add(q::PooledImmutableDirectedTransport::completeBOM(physical,policy,host,std::array{src,own}));
        const auto channel=q::ImmutableLookupLayout::createSubordinate(Unique,source.rowBytes());bom.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
        const std::array declarations{CPUTensorStorageDeclaration(1,Maximum*2,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(Maximum/16,16*file_.columns,BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
        const auto activation=Maximum*sizeof(std::uint64_t)+sizeof(int)+sampling_math::kDeviceGenerationControlCount*sizeof(int)+Maximum*file_.columns*sizeof(float)+sizeof(std::int32_t);
        bom.add(src,PhysicalMemoryOwner::ActivationArena,activation);for(const auto& root:declarations)bom.add(host,PhysicalMemoryOwner::ActivationArena,root.physicalBytes());
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        const auto sourceExecutables=3*GPUGraphMemoryContract::reservationBytesPerExecutable(sourceDevice,GPUGraphExecutableClass::General);
        const auto ownerExecutable=GPUGraphMemoryContract::reservationBytesPerExecutable(ownerDevice,GPUGraphExecutableClass::General);
#endif
        PhysicalMemoryPlanBuilder retained;
        #ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        retained.add(src,PhysicalMemoryOwner::NativeGraphExecutable,sourceExecutables);retained.add(own,PhysicalMemoryOwner::NativeGraphExecutable,ownerExecutable);
#else
        const auto sourceGeneral=RetainedParticipantGraphExecutable::generalBOM(host,src,3);
        for(const auto& row:sourceGeneral.resources())retained.add(row);
        const auto ownerGeneral=RetainedParticipantGraphExecutable::generalBOM(host,own,1);
        for(const auto& row:ownerGeneral.resources())retained.add(row);
#endif
        for(const auto program:{q::PooledImmutableCoordinatorNativePlan::orderingProgram(physical,policy),
            q::PooledImmutableSourceGeometry::orderingProgram(physical,policy,source),NativeOrderingEventProgram::lookupSubmissionRetirement()}){
            const auto group=ordering.bom(program,host,src);for(const auto& row:group.resources())retained.add(row);
        }
        for(const auto program:{q::PooledImmutableOwnerNativePlan::orderingProgram(physical,policy,0),NativeOrderingEventProgram::lookupSubmissionRetirement()}){
            const auto group=ordering.bom(program,host,own);for(const auto& row:group.resources())retained.add(row);
        }
        const auto cpu=ioProgram.bom(host);
        for(const auto& row:cpu.resources()){
            if(row.bytes(PhysicalMemoryOwner::RetainedHostServiceStorage))bom.add(host,PhysicalMemoryOwner::RetainedHostServiceStorage,row.bytes(PhysicalMemoryOwner::RetainedHostServiceStorage));
            if(row.bytes(PhysicalMemoryOwner::RetainedHostServiceRuntime))retained.add(host,PhysicalMemoryOwner::RetainedHostServiceRuntime,row.bytes(PhysicalMemoryOwner::RetainedHostServiceRuntime));
        }
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        std::optional<PooledRemoteGraphProgram> remote;
        if(spec.authoredRemote){remote=q::PooledImmutableDirectedOwnerProgram::declare(physical,policy,0);
            const auto known=PooledRemoteGraphCalibration::knownBOM(*remote,host,own);add(known);}
#endif
        return {layout,physical,policy,cpuGeometry,channel,declarations,activation,bom.build(),retained.build()
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
            ,std::move(remote)
#endif
        };
    }
};
/** @return Serialized payload envelope plus every independent original native driver obligation. */
PhysicalMemoryPlan crossVendorReadProgramBOM(std::span<const CrossVendorReadReplayGeometry> declarations){
    if(declarations.empty())throw std::invalid_argument("Cross-vendor program requires its complete immutable declarations");
    std::vector<PhysicalMemoryPlan> alternatives;PhysicalMemoryPlanBuilder retained;
    for(const auto& geometry:declarations){alternatives.push_back(geometry.payload);for(const auto& row:geometry.retained.resources())retained.add(row);}
    PhysicalMemoryPlanBuilder complete;complete.addMutuallyExclusive(alternatives);const auto native=retained.build();for(const auto& row:native.resources())complete.add(row);return complete.build();
}
