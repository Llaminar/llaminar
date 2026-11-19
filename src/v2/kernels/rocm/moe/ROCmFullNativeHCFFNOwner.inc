/**
 * @file ROCmFullNativeHCFFNOwner.inc
 * @brief Complete HC-prefix capture identity and cross-region storage checks for the existing FFN owner.
 *
 * Included inside ROCmMoEKernel.cpp's llaminar2 namespace. The original HC stage
 * retains its tensor/source/count leases and validates them before backend
 * submission. These helpers seal the borrowed values and validate only their
 * composition with the already resolved FFN tables/workspace; they allocate no
 * device storage and introduce no second descriptor or execution-state owner.
 */
namespace {
/**
 * @brief Include every semantic HC prefix field in the existing prepared signature.
 * @tparam Mix Existing ordered signature accumulator.
 * @param prefix Exact admitted prefix snapshot, including its original row authorities.
 * @param mix Existing FFN owner's fieldwise accumulator; no struct padding is hashed.
 */
template<class Mix>
void mixFullNativeHCFFNPrefix(const rocm::FullNativeHCFFNPrefixArgs& prefix,const Mix& mix)
{
    const auto pointer=[&](const void* value){mix(reinterpret_cast<std::uintptr_t>(value));};
    const auto rows=[&](DeviceRowRange value) {
        const auto identity=value.captureIdentity();
        pointer(identity.count);mix(identity.count_capacity);mix(identity.first_physical_row);
        mix(identity.physical_rows);mix(identity.repeats_per_row);mix(identity.schema_version);
    };
    const auto matrix=[&](const DeviceNativeQuantisedMatrixDesc& value) {
        pointer(value.payload);pointer(value.scales);pointer(value.mins);pointer(value.emins);
        mix(value.n);mix(value.k);mix(value.blocks_per_row);mix(value.codebook_id);
        mix(value.allocation_payload_bytes_per_block);mix(value.allocation_has_mins);mix(value.allocation_has_emins);
        mix(value.source_codebook_id);mix(value.source_is_superblock);mix(value.source_identity_present);
        mix(value.reserved);mix(value.reserved_tail);
    };
    const auto& hc=prefix.hc;
    mix(prefix.output_demand.bits());
    mix(static_cast<std::uint32_t>(prefix.selected_descriptor_demand));
    pointer(prefix.barrier_state);
    pointer(prefix.write.residual);pointer(prefix.write.block);pointer(prefix.write.gateLogits);pointer(prefix.write.output);
    mix(hc.norm.extent.rows);pointer(hc.norm.extent.liveRows);mix(hc.norm.hidden);
    mix(std::bit_cast<std::uint32_t>(hc.norm.epsilon));
    pointer(hc.norm.input);pointer(hc.norm.gain);pointer(hc.norm.output);
    matrix(hc.down_weights);
    pointer(hc.input_codes);pointer(hc.input_scales);pointer(hc.down_partials);pointer(hc.raw_down);
    rows(hc.publication.rows);mix(hc.publication.columns);pointer(hc.publication.activated);
    pointer(hc.publication.primary);pointer(hc.publication.residual);
    pointer(hc.publication.primaryScales);pointer(hc.publication.residualScales);
    matrix(hc.up.weights);rows(hc.up.rows);mix(static_cast<unsigned>(hc.up.quantizer));
    pointer(hc.up.input);pointer(hc.up.output);pointer(hc.up.primary);pointer(hc.up.residual);
    pointer(hc.up.primaryScales);pointer(hc.up.residualScales);pointer(hc.mixed);
    pointer(hc.injection.weights);pointer(hc.injection.output);rows(hc.injection.rows);mix(hc.injection.load_pairs);
    mix(hc.partitions);mix(static_cast<unsigned>(hc.grid_policy));mix(hc.resident_grid);
}

/**
 * @brief Reject a prefix snapshot that no longer names the original suffix boundary.
 * @param prefix Complete admitted prefix, retaining independently declared row policies.
 * @param suffix Existing canonical suffix binding, including its original hidden output/input edge.
 * @details Source/type/capacity and physical alias rules within HC belong to its
 * original stage and the full low-level admission. This checks the additional
 * relations created by composition without inferring a live count on the host.
 */
void validateFullNativeHCFFNPrefixBinding(const rocm::FullNativeHCFFNPrefixArgs& prefix,
    const rocm::NativeLayoutRouterFFNArgs& suffix)
{
    const auto& hc=prefix.hc;
    if(!prefix.barrier_state||suffix.barrier_state!=prefix.barrier_state+FullNativeHCFFNBarrierLayout::suffix_word_offset||
       hc.norm.hidden!=suffix.ffn.hidden_columns||hc.mixed!=suffix.ffn.hidden||
       prefix.write.output!=hc.norm.input||!prefix.write.residual||!prefix.write.block||!prefix.write.gateLogits||
       hc.up.input!=hc.publication.activated||hc.up.primary!=hc.publication.primary||
       hc.up.residual!=hc.publication.residual||hc.up.primaryScales!=hc.publication.primaryScales||
       hc.up.residualScales!=hc.publication.residualScales||hc.publication.columns!=hc.up.weights.k||
       hc.down_weights.n!=hc.up.weights.k||hc.norm.extent.rows!=1||
       hc.publication.rows.physicalRows()!=1||hc.up.rows.physicalRows()!=1||
       hc.resident_grid!=suffix.ffn.resident_grid)
        throw std::logic_error("Full native HC/FFN changed its original prefix/suffix binding relation");
}

/**
 * @brief Protect the existing FFN source/control/public bytes from new HC writes, and conversely.
 * @param prefix Admitted original HC operands; its stage retains their physical leases.
 * @param suffix Already resolved original router/FFN binding and selected publisher tables.
 * @param scratch Existing retained FFN named intervals, including the complete barrier bank.
 * @param workspace Original shared workspace authority for exact HC temporary-bank bindings.
 * @details These temporary consumed-byte intervals perform alias validation only.
 * They are not allocations, a live capacity ledger, a descriptor mirror or an
 * indirect weight walk. Original exact raw/activated and residual/post HC aliases
 * remain the stage's authority. The sole new cross-region tensor edge is mixed
 * hidden to suffix.hidden; all other retained public/control spans stay distinct.
 * Exact named HC temporaries may overlap suffix temporaries because the canonical
 * mixed-ready grid edge retires every HC scratch reader before suffix writes.
 * Such Q8/partial banks are not retained public outputs: intermediate numerical
 * observations require a separate prefix replay or a disjoint test arena, never
 * new duplicate production banks merely for diagnostics.
 */
void validateFullNativeHCFFNStorage(const rocm::FullNativeHCFFNPrefixArgs& prefix,
    const rocm::NativeLayoutRouterFFNArgs& suffix,std::span<const ResidentFFNScratchBinding> scratch,
    const DeviceWorkspaceManager& workspace)
{
    struct Span {const void* pointer;std::size_t bytes;};
    const auto span=[](const void* pointer,std::size_t count,std::size_t element=1)->Span {
        if(count>std::numeric_limits<std::size_t>::max()/element)
            throw std::invalid_argument("Full native HC/FFN consumed extent overflows");
        const auto bytes=count*element,begin=reinterpret_cast<std::uintptr_t>(pointer);
        if(bytes&&(!pointer||bytes>std::numeric_limits<std::uintptr_t>::max()-begin))
            throw std::invalid_argument("Full native HC/FFN consumed interval is absent or overflowing");
        return {pointer,bytes};
    };
    const auto overlaps=[](Span a,Span b) {
        const auto x=reinterpret_cast<std::uintptr_t>(a.pointer),y=reinterpret_cast<std::uintptr_t>(b.pointer);
        return a.bytes&&b.bytes&&x<y+b.bytes&&y<x+a.bytes;
    };
    const auto& hc=prefix.hc;const auto& ffn=suffix.ffn;
    if(hc.norm.hidden<=0||hc.norm.hidden>std::numeric_limits<int>::max()/4||hc.down_weights.n<=0||hc.partitions<=0)
        throw std::invalid_argument("Full native HC/FFN has no admitted prefix extent");
    const auto H=std::size_t(hc.norm.hidden),K=4*H,R=std::size_t(hc.down_weights.n);
    const auto optional=[&](FullNativeHCFFNOutputRole role,const void* pointer,std::size_t count) {
        return span(pointer,prefix.output_demand.contains(role)?count:0,sizeof(float));
    };
    const std::array public_prefix{
        span(prefix.write.output,K,sizeof(float)),span(hc.norm.output,K,sizeof(float)),
        span(hc.raw_down,R,sizeof(float)),optional(FullNativeHCFFNOutputRole::HCActivated,hc.publication.activated,R),
        optional(FullNativeHCFFNOutputRole::HCUp,hc.up.output,K),span(hc.mixed,H,sizeof(float))};
    std::vector<Span> writes(public_prefix.begin(),public_prefix.end());
    if(hc.injection.weights)writes.push_back(span(hc.injection.output,4,sizeof(float)));
    // Only these original named banks have the proved HC-before-suffix lifetime.
    // Checking their actual workspace identity prevents an arbitrary public tensor
    // pointer from acquiring this temporary-storage exception.
    std::vector<Span> retired_writes;
    const auto retired=[&](Span value,const std::string& name) {
        if(workspace.getBuffer(name)!=value.pointer||workspace.getBufferSize(name)<value.bytes)
            throw std::invalid_argument("Full native HC temporary changed its original named workspace binding");
        retired_writes.push_back(value);
    };
    retired(span(hc.input_codes,K),GemmWorkspaceBuffers::QUANT_A);
    retired(span(hc.input_scales,K/32,sizeof(float)),GemmWorkspaceBuffers::SCALES_A_BLOCKWISE);
    retired(span(hc.down_partials,std::size_t(hc.partitions)*R,sizeof(float)),GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL);
    retired(span(hc.up.primary,R),kGPUProjectionPrimaryQ8);
    retired(span(hc.up.residual,R),kGPUProjectionResidualQ8);
    retired(span(hc.up.primaryScales,R/32,sizeof(float)),kGPUProjectionPrimaryQ8Scales);
    retired(span(hc.up.residualScales,R/32,sizeof(float)),kGPUProjectionResidualQ8Scales);
    std::vector<Span> protected_suffix{
        span(suffix.router_weights,std::size_t(ffn.experts)*H,sizeof(std::uint16_t)),
        span(suffix.router_indices,ffn.top_k,sizeof(float)),span(suffix.router_weights_output,ffn.top_k,sizeof(float)),
        span(ffn.runtime,1,sizeof(DeviceMoELayerRuntime))};
    const auto tables=[&](const DeviceNativeQuantisedMatrixDesc* gate,const DeviceNativeQuantisedMatrixDesc* up,
        const DeviceNativeQuantisedMatrixDesc* down,std::size_t experts) {
        for(const auto* table:{gate,up,down})protected_suffix.push_back(span(table,experts,sizeof(DeviceNativeQuantisedMatrixDesc)));
    };
    tables(ffn.gates,ffn.ups,ffn.downs,ffn.experts);
    tables(ffn.shared_gate,ffn.shared_up,ffn.shared_down,1);
    tables(suffix.router_publication.gate,suffix.router_publication.up,suffix.router_publication.down,
        suffix.router_publication.expert_count);
    const std::array suffix_outputs{
        optional(FullNativeHCFFNOutputRole::RouteContributions,ffn.route_contributions,std::size_t(ffn.top_k)*H),
        optional(FullNativeHCFFNOutputRole::Routed,ffn.routed_output,H),
        optional(FullNativeHCFFNOutputRole::Shared,ffn.shared_output,H),span(ffn.combined_output,H,sizeof(float)),
        optional(FullNativeHCFFNOutputRole::SharedGate,ffn.shared_gate_output,ffn.shared_columns),
        optional(FullNativeHCFFNOutputRole::SharedUp,ffn.shared_up_output,ffn.shared_columns)};
    std::vector<Span> suffix_writes(suffix_outputs.begin(),suffix_outputs.end());
    for(const auto value:{span(suffix.router_logits,ffn.experts,sizeof(float)),
        span(suffix.router_indices,ffn.top_k,sizeof(float)),span(suffix.router_weights_output,ffn.top_k,sizeof(float)),
        span(ffn.runtime,1,sizeof(DeviceMoELayerRuntime))})suffix_writes.push_back(value);
    for(const auto* table:{suffix.router_publication.gate,suffix.router_publication.up,suffix.router_publication.down})
        suffix_writes.push_back(span(table,suffix.router_publication.expert_count,sizeof(DeviceNativeQuantisedMatrixDesc)));
    for(const auto output:suffix_outputs)protected_suffix.push_back(output);
    // Original tables/control/public tensors must survive both phases. Only
    // suffix-produced arithmetic scratch (including raw router logits) may reuse
    // the exact retired named HC temporary bytes after mixed-ready.
    for(const auto write:retired_writes)
        for(const auto protected_value:protected_suffix)
            if(overlaps(write,protected_value))throw std::invalid_argument("Full native HC temporary aliases retained FFN source/control/output storage");
    const auto suffix_temporary=[](const char* name) {
        const std::string_view value{name};
        return value==MoEWorkspaceBuffers::DECODE_HIDDEN_INT8||value==MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES||
            value==MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS||value==MoEWorkspaceBuffers::GATEUP_UP_PARTIALS||
            value==MoEWorkspaceBuffers::DOWN_PARTIALS||value==MoEWorkspaceBuffers::DECODE_SWIGLU_INT8||
            value==MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES||value==MoEWorkspaceBuffers::ROUTE_LOGITS;
    };
    for(const auto& binding:scratch) {
        const auto value=span(binding.pointer,binding.bytes);
        if(!suffix_temporary(binding.name))
            for(const auto write:retired_writes)
                if(overlaps(write,value))throw std::invalid_argument("Full native HC temporary aliases retained FFN metadata");
        protected_suffix.push_back(value);
    }
    protected_suffix.push_back(span(suffix.router_logits,ffn.experts,sizeof(float)));
    for(const auto write:writes)
        for(const auto protected_value:protected_suffix)
            if(overlaps(write,protected_value))throw std::invalid_argument("Full native HC write aliases FFN source/control/output storage");
    // Mixed hidden is the one intended original producer/consumer edge. Other
    // prefix writes must not corrupt the retained shared input-gate weight.
    for(const auto write:writes)
        if(overlaps(write,span(ffn.shared_input_gate,H,sizeof(float))))
            throw std::invalid_argument("Full native HC write aliases the shared gate source");
    for(const auto write:retired_writes)
        if(overlaps(write,span(ffn.shared_input_gate,H,sizeof(float))))
            throw std::invalid_argument("Full native HC temporary aliases the shared gate source");
    // This original input has a distinct proved lifetime: all prefix write
    // readers retire at normalized-ready, before the suffix publishes combined.
    // Keep its role separate so an identical address in another retained input
    // or public-plane record cannot acquire this exception by pointer equality.
    struct PreviousWriteBlockInput {Span interval;};
    const PreviousWriteBlockInput previous_block{span(prefix.write.block,H,sizeof(float))};
    std::vector<Span> prefix_reads{
        span(prefix.write.residual,K,sizeof(float)),
        span(prefix.write.gateLogits,4,sizeof(float)),span(hc.norm.gain,K,sizeof(float))};
    for(const auto value:public_prefix)prefix_reads.push_back(value);
    const auto source=[&](const DeviceNativeQuantisedMatrixDesc& matrix) {
        const auto blocks=std::size_t(matrix.n)*matrix.blocks_per_row;
        prefix_reads.push_back(span(matrix.payload,blocks,32));prefix_reads.push_back(span(matrix.scales,blocks,sizeof(std::uint16_t)));
    };
    source(hc.down_weights);source(hc.up.weights);
    if(hc.injection.weights) {
        prefix_reads.push_back(span(hc.injection.weights,4*K,sizeof(float)));
        prefix_reads.push_back(span(hc.injection.output,4,sizeof(float)));
    }
    for(const auto* count:{hc.norm.extent.liveRows,hc.publication.rows.countOwner(),hc.up.rows.countOwner(),hc.injection.rows.countOwner()})
        if(count)prefix_reads.push_back(span(count,1,sizeof(std::int32_t)));
    for(const auto output:suffix_writes) {
        if(!overlaps(previous_block.interval,output))continue;
        const bool exact_combined_reuse=previous_block.interval.pointer==ffn.combined_output&&
            previous_block.interval.bytes==H*sizeof(float)&&output.pointer==ffn.combined_output&&
            output.bytes==previous_block.interval.bytes;
        if(!exact_combined_reuse)
            throw std::invalid_argument("Full native FFN output aliases the previous block outside its exact combined reuse");
    }
    for(const auto& binding:scratch)
        if(overlaps(previous_block.interval,span(binding.pointer,binding.bytes)))
            throw std::invalid_argument("Full native FFN scratch aliases the original previous block");
    for(const auto protected_value:prefix_reads) {
        for(const auto output:suffix_writes)
            if(overlaps(protected_value,output))throw std::invalid_argument("Full native FFN output aliases retained HC data");
        for(const auto& binding:scratch)
            if(overlaps(protected_value,span(binding.pointer,binding.bytes)))
                throw std::invalid_argument("Full native FFN scratch aliases retained HC data");
    }
}
} // namespace
