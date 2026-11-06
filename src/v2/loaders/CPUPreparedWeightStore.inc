/**
 * @file CPUPreparedWeightStore.inc
 * @brief Original-model commit boundary for streamed CPU final weight ownership.
 *
 * Included in PreparedWeightStore's namespace. A source transaction retains the
 * original Store/PMA and cannot publish a partial engine, substitute a model or
 * manufacture readable source storage from packed bytes.
 */
struct PreparedWeightStore::CPUPreparation::Impl {
    PreparedWeightStore* store=nullptr;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    WeightBinding original;
    PhysicalMemoryOwner owner;
    std::unique_ptr<CPUPreparedWeightWriter> writer;
};
PreparedWeightStore::CPUPreparation::CPUPreparation(std::unique_ptr<Impl> value):impl_(std::move(value)){}
PreparedWeightStore::CPUPreparation::~CPUPreparation()=default;
std::optional<NativeWeightLoadStagingRequest> PreparedWeightStore::CPUPreparation::stagingRequest() const{return impl_->writer->stagingRequest();}
const CPUPreparedWeightSource& PreparedWeightStore::CPUPreparation::source() const & noexcept{return impl_->writer->source();}
void PreparedWeightStore::CPUPreparation::append(const NativeWeightSourceWindow& w,std::span<const std::byte> bytes,std::span<std::byte> scratch){impl_->writer->append(w,bytes,scratch);}
void PreparedWeightStore::CPUPreparation::readDirect(const NativeWeightSourceWindow& w,const std::function<void(std::span<std::byte>)>& read){impl_->writer->readDirect(w,read);}
std::unique_ptr<PreparedWeightStore::CPUPreparation> PreparedWeightStore::beginCPUPreparation(
    WeightBinding original,CPUPreparedWeightSource source,PhysicalMemoryOwner owner,CPUStoragePlacement node) {
    const auto device=original.residency.resident_device.value_or(original.residency.home_device);
    const auto memory=physicalMemoryAuthority();
    if(!memory || !device.is_cpu() || !memory->contains(device) || !isPersistentWeightOwner(owner) ||
       !original.binding_id || !original.identity.instance_id || original.identity.model_id!=model_id_ ||
       original.tensor || original.tensor_owner || original.prepared || original.projection_source_owner ||
       original.floating_operand_precision!=source.operand_precision)
        throw std::invalid_argument("CPU source preparation requires exact unmaterialized original model/PMA binding");
    if(original.native_source && (original.native_source->original!=source.original || original.native_source->range!=source.range || original.native_source->matrix_count!=source.matrix_count || original.native_source->assembly!=source.assembly))
        throw std::invalid_argument("CPU source preparation changed its original native projection");
    if(original.cpu_joint_source) {
        const auto& selection=*original.cpu_joint_source;
        if(!original.projection_ownership || selection.ownership()!=*original.projection_ownership ||
           !selection.contains(original.identity.expert) || selection.layer()!=original.identity.layer ||
           node!=CPUStoragePlacement::onNode(selection.participant().address.numa_node) ||
           selection.participant().domain_name!=original.identity.overlay_domain ||
           selection.participant().world_rank!=original.identity.overlay_participant_world_rank ||
           selection.participant().domain_participant_index!=original.identity.overlay_participant_index ||
           source.range.original_expert!=size_t(original.identity.expert))
            throw std::invalid_argument("CPU source preparation changed its original fragment/NUMA identity");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(entries_.contains(keyFor(original.binding_id,device)) || embedding_entries_.contains(keyFor(original.binding_id,device)))
            throw std::logic_error("CPU source binding already has completed final storage");
    }
    auto placement=node.node()<0?CPUWeightStoragePlacement::local():CPUWeightStoragePlacement::onNode(node.node());
    placement=CPUWeightStoragePlacement::accounted(memory,device,owner,placement);
    auto state=std::make_unique<CPUPreparation::Impl>();state->store=this;state->memory=memory;
    state->original=std::move(original);state->owner=owner;
    state->writer=CPUPreparedWeightWriter::begin(std::move(source),placement);
    return std::unique_ptr<CPUPreparation>(new CPUPreparation(std::move(state)));
}
WeightBinding PreparedWeightStore::finishCPUPreparation(std::unique_ptr<CPUPreparation> preparation) {
    if(!preparation || !preparation->impl_ || preparation->impl_->store!=this ||
       preparation->impl_->memory.get()!=physicalMemoryAuthority().get() ||
       preparation->impl_->original.identity.model_id!=model_id_)
        throw std::invalid_argument("CPU final commit requires the original live Store/PMA transaction");
    auto& state=*preparation->impl_;const auto source=state.writer->source();
    auto result=state.writer->finish();auto binding=state.original;
    const auto device=binding.residency.resident_device.value_or(binding.residency.home_device);
    if(!result.final_tensor) {
        if(!result.kernel)throw std::logic_error("CPU final commit has neither actual engine nor final tensor");
        result.final_tensor=PreparedCPUWeightTensor::fromCommitted(result.kernel,{source.assembly?source.assembly->final_rows:source.range.rows,source.assembly?source.assembly->final_columns:source.range.columns},source.destination_precision==WeightPrecision::CONVERT_TO_INT8?TensorType::Q8_0:source.original.original_type,device);
    }
    binding.tensor_owner=result.final_tensor;binding.tensor=result.final_tensor.get();binding.immutable=true;
    // The incomplete source binding deliberately has no raw bytes. Publication
    // reports the committed execution owner: packed metadata stays raw-absent,
    // while genuine resident/floating storage remains live CPU input.
    binding.residency.raw_host_data_available=!binding.tensor->is_raw_data_released() && binding.tensor->raw_data()!=nullptr;
    // This is execution storage, not a temporary source copy. Floating GEMM
    // and native embedding handles borrow these exact final bytes after graph
    // publication; no transient-source policy may authorize their reclamation.
    // Packed metadata has no raw bytes and retains its separate engine root.
    if(binding.residency.raw_host_data_available)
        binding.residency.host_policy=WeightHostPolicy::RequiredForCPUExecution;
    binding.native_source=NativeWeightSourceProjection{.original=source.original,.range=source.range,.matrix_count=source.matrix_count,.assembly=source.assembly};
    binding.projection_source_owner.reset();
    const auto kind=source.consumer==CPUWeightConsumer::PreparedGemm?PreparedWeightKind::CpuPackedGemm:PreparedWeightKind::CPUResidentTensor;
    auto ref=makeRef(binding.binding_id,kind,device);binding.prepared=ref;
    std::shared_ptr<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle> handle;
    if(source.consumer==CPUWeightConsumer::PreparedGemm) {
        if(!result.kernel)throw std::logic_error("CPU GEMM commit is missing its completed actual engine");
        using Factory=llaminar::v2::kernels::KernelFactory;
        handle=std::make_shared<Factory::PreparedGemmHandle>();handle->tensor=binding.tensor;handle->device_id=device;
        handle->kind=isW4A4Format(binding.tensor->native_type()) ? Factory::GemmPreparationKind::CPU_NATIVE_ONLINE_A4 :
            result.kernel->canReleaseSourceWeightTensor()?Factory::GemmPreparationKind::CPU_PACKED:Factory::GemmPreparationKind::FLOATING_POINT;
        handle->variant=static_cast<int>(binding.tensor->native_type());handle->prepared_weights=std::make_shared<Factory::PreparedGemmWeights>();
        handle->prepared_weights->kind=handle->kind;handle->prepared_weights->owned_kernel=std::move(result.kernel);
        handle->prepared_weights->kernel=handle->prepared_weights->owned_kernel.get();
    }
    validateBindingForStore(binding,model_id_,kind);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto key=keyFor(ref);
    if(entries_.contains(key) || embedding_entries_.contains(key))throw std::logic_error("CPU final commit would replace an existing completed owner");
    entries_.emplace(key,Entry{binding,ref,std::move(handle),state.owner});
    // Publication follows the actual Store-owned engine/root, including a
    // floating engine that aliases its final raw tensor. Resident-only roots
    // are not prepared kernels and must retain their unprepared state.
    if(kind==PreparedWeightKind::CpuPackedGemm)binding.tensor->publishPreparedDeviceState();
    WeightLifecycleTrace::record(WeightLifecycleEventType::RegisterPrepared,binding.identity.canonical_name,
        binding.identity.role,binding.identity.layer,device,toString(kind));
    return binding;
}
