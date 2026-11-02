/**
 * @file PooledImmutableNativeOperands.cpp
 * @brief Closed exact input schemas and strong original-root/runtime seals.
 *
 * No method reads mutable device values. Required byte capacities and alignment
 * come from the original geometry/action; actual transport bytes remain native.
 * Immutable input subranges may share one canonical root only under one arena
 * role and only when their read ranges are disjoint.
 */
#include "PooledImmutableNativeOperands.h"
#include "PooledImmutableNativeRootStorage.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;
namespace p=adaptive_lookup::pooled;
/** @return Closed operand index. */
constexpr std::size_t index(PooledNativeOperand role){return static_cast<std::size_t>(role);}
/** @brief Declare one exact positive capacity; zero means the operand must be absent. */
void need(PooledNativeOperandSchema& schema,PooledNativeOperand role,std::size_t bytes,std::size_t alignment)
{schema[index(role)]={bytes,alignment};}
/** @return Checked capacity multiplication, independent of live counts. */
std::size_t times(std::size_t a,std::size_t b)
{
    if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::overflow_error("Pooled native operand extent overflows");
    return a*b;
}
/** @brief Reject missing/extra/foreign/aliased declarations before physical sealing or native work. */
void requireSpanDeclarations(const PooledNativeOperandSpans& spans,const PooledNativeOperandSchema& schema)
{
    for(std::size_t i=0;i<spans.size();++i){
        if(bool(spans[i])!=(schema[i].bytes!=0))throw std::invalid_argument("Pooled native phase has missing or unused operands");
        if(!spans[i])continue;
        const auto& s=*spans[i];const auto& r=schema[i];
        if(!s.root || s.root->is_view() || s.root->transferStorageOwner()!=s.root.get() ||
            s.buffer>=BufferId::_COUNT || s.bytes!=r.bytes || !r.alignment || s.offset%r.alignment ||
            s.offset>s.root->size_bytes() || s.bytes>s.root->size_bytes()-s.offset ||
            (s.root->native_type()!=TensorType::INT8 && s.root->native_type()!=TensorType::INT32))
            throw std::invalid_argument("Pooled native operand requires its canonical exact type/role/extent");
        if((i==index(PooledNativeOperand::Ids) || i==index(PooledNativeOperand::OriginalCount)) &&
            s.root->native_type()!=TensorType::INT32)
            throw std::invalid_argument("Pooled original IDs/count require their canonical INT32 packed/native root");
        for(std::size_t prior=0;prior<i;++prior)if(spans[prior]){
            const auto& t=*spans[prior];
            if((s.root==t.root)!=(s.buffer==t.buffer))
                throw std::invalid_argument("Pooled native arena role does not name its same original root");
            if(s.root==t.root && s.offset<t.offset+t.bytes && t.offset<s.offset+s.bytes)
                throw std::invalid_argument("Pooled immutable operand roles overlap");
        }
    }
}
/** @return Strong unique root declarations, built only before capture. */
std::vector<TensorBase*> roots(const PooledNativeOperandSpans& spans)
{
    std::vector<TensorBase*> result;result.reserve(spans.size());
    for(const auto& s:spans)if(s && std::find(result.begin(),result.end(),s->root.get())==result.end())result.push_back(s->root.get());
    return result;
}
/** @return Already-prepared exact input pointer; no transfer/allocation occurs. */
const void* address(const PooledNativeOperandSpans& spans,PooledNativeOperand role)
{
    const auto& span=spans[index(role)];
    return span?static_cast<const std::byte*>(span->root->gpu_data_ptr())+span->offset:nullptr;
}
/** @brief Validate exact prepared input pointers and original native alignment. */
void requirePrepared(const PooledNativeOperandSpans& spans,const PooledNativeOperandSchema& schema,DeviceId device)
{
    for(std::size_t i=0;i<spans.size();++i)if(spans[i]){
        const auto& span=*spans[i];
        if(span.root->current_device()!=device || !span.root->gpu_data_ptr() ||
            reinterpret_cast<std::uintptr_t>(address(spans,static_cast<PooledNativeOperand>(i)))%schema[i].alignment)
            throw std::logic_error("Pooled native input changed device/address/alignment after preparation");
    }
}
/** @brief Borrow each exact original allocation once, outside capture; aliases share the same allocator token and never another PMA balance. */
void retainInputs(const PooledNativeOperandSpans& spans,DeviceId device,
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>,PooledNativeOperandCount>& captured)
{
    for(std::size_t i=0;i<spans.size();++i)if(spans[i] && !captured[i]){
        for(std::size_t prior=0;prior<i;++prior)if(spans[prior] && spans[prior]->root==spans[i]->root){captured[i]=captured[prior];break;}
        if(!captured[i])captured[i]=TransferEngine::instance().retainCapturedDeviceStorage(*spans[i]->root,device,spans[i]->root->size_bytes());
    }
}
/** @brief Validate retained original allocation ownership without registry scans, allocation or mutable value reads. */
void requireInputs(const PooledNativeOperandSpans& spans,DeviceId device,
    const std::array<std::shared_ptr<const CapturedDeviceStorageLease>,PooledNativeOperandCount>& captured)
{
    for(std::size_t i=0;i<spans.size();++i)if(spans[i]){
        if(!captured[i])throw std::logic_error("Pooled native operand has no retained original allocation borrower");
        TransferEngine::instance().requireCapturedDeviceStorage(*captured[i],*spans[i]->root,device,spans[i]->root->size_bytes());
    }
}
/** @return Stable declaration identity excluding mutable values and not requiring cold input addresses. */
std::uint64_t declaration(const PooledNativeOperandSpans& spans,std::uint64_t geometry,std::uint64_t action)
{
    std::uint64_t value=1469598103934665603ULL;const auto mix=[&](std::uint64_t x){value=(value^x)*1099511628211ULL;};
    mix(n::Version);mix(geometry);mix(action);
    for(std::size_t i=0;i<spans.size();++i){
        mix(i);mix(bool(spans[i]));if(!spans[i])continue;const auto& span=*spans[i];
        mix(reinterpret_cast<std::uintptr_t>(span.root.get()));mix(static_cast<std::uint64_t>(span.buffer));
        mix(span.offset);mix(span.bytes);mix(static_cast<std::uint64_t>(span.root->native_type()));
    }
    return value?value:1;
}
/** @return Complete immutable combination, with no hash of structure padding. */
std::uint64_t combine(std::uint64_t first,std::uint64_t second){return ((first^second)*1099511628211ULL)|1;}
}
PooledNativeOperandSchema pooledNativeSchema(n::OwnerAction action,const n::OwnerParameters& plan,n::SourceCondition condition,n::RecordLayout records)
{
    // Remote packing requires the original admitted ingress/payload family.
    // Source17 owns the separate direct-local path and never calls this schema.
    if(action==n::OwnerAction::ReadRows && !plan.native.ingress)
        throw std::invalid_argument("Public pinned-hit packing requires its original remote owner ingress; local reads use Source17");
    PooledNativeOperandSchema schema{};const auto c=plan.native.reply_capacity,participants=plan.owner.participants();
    need(schema,PooledNativeOperand::Invocation,sizeof(n::Invocation),8);
    switch(action){
    case n::OwnerAction::AcquireReaderRetirement:need(schema,PooledNativeOperand::Frontier,sizeof(n::ReadersRetiredGate),8);break;
    case n::OwnerAction::ReadRows:
        need(schema,PooledNativeOperand::Records,times(c,sizeof(p::Grant)),8);
        need(schema,PooledNativeOperand::RecordBytes,c?8:0,8);break;
    case n::OwnerAction::Pin:
        need(schema,PooledNativeOperand::Records,times(c,sizeof(p::OwnerCommand)),8);
        need(schema,PooledNativeOperand::RecordBytes,c?8:0,8);break;
    case n::OwnerAction::AcquirePins:need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(p::PinHeader)),8);break;
    case n::OwnerAction::AcceptFree:
        need(schema,PooledNativeOperand::Records,times(c,sizeof(p::AllocationDecision)),8);
        need(schema,PooledNativeOperand::Frontier,sizeof(n::FreePlanHeader),8);break;
    case n::OwnerAction::AcquireFree:need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(p::FreeAllocationReceipt)),8);break;
    case n::OwnerAction::AcceptVictim:
        need(schema,PooledNativeOperand::Records,times(c,sizeof(p::AllocationDecision)),8);
        need(schema,PooledNativeOperand::Frontier,sizeof(n::VictimPlanHeader),8);break;
    case n::OwnerAction::ApplySourceDelivery:
        need(schema,PooledNativeOperand::Records,times(c,sizeof(n::Fill)),8);
        need(schema,PooledNativeOperand::RecordBytes,c?8:0,8);
        need(schema,PooledNativeOperand::Payload,times(c,plan.row_bytes),1);
        need(schema,PooledNativeOperand::PayloadBytes,c?8:0,8);
        need(schema,PooledNativeOperand::Frontier,sizeof(n::SourceFrontier),8);break;
    case n::OwnerAction::AcquirePublication:need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(p::PublishHeader)),8);break;
    case n::OwnerAction::SealSource:
    case n::OwnerAction::CancelSource:case n::OwnerAction::DrainSourceFailure:need(schema,PooledNativeOperand::Frontier,sizeof(n::SourceFrontier),8);break;
    case n::OwnerAction::Begin:case n::OwnerAction::SealPins:case n::OwnerAction::Availability:
    case n::OwnerAction::ExecuteFree:case n::OwnerAction::FreeReceipt:case n::OwnerAction::ExecuteVictim:
    case n::OwnerAction::VictimReceipt:case n::OwnerAction::ValidateFills:case n::OwnerAction::MakeReady:
    case n::OwnerAction::ReadyReceipt:case n::OwnerAction::Release:case n::OwnerAction::CancelFree:
    case n::OwnerAction::CancelVictim:case n::OwnerAction::Retire:
    case n::OwnerAction::PublishWire:case n::OwnerAction::ReadyWire:case n::OwnerAction::PublicationAcquiredWire:case n::OwnerAction::RetiredWire:break;
    default:throw std::invalid_argument("Unknown pooled native owner phase schema");
    }
    if(records==n::RecordLayout::DirectedCommandBank){
        if(action!=n::OwnerAction::Pin || !c)
            throw std::invalid_argument("Directed commands require an original positive local Pin consumer");
        need(schema,PooledNativeOperand::Records,times(plan.owner.maximumUnique(),sizeof(p::OwnerCommand)),8);
        need(schema,PooledNativeOperand::RecordBytes,0,8);
        need(schema,PooledNativeOperand::AllocationSlice,sizeof(n::DirectedSlice),8);
    }else if(records==n::RecordLayout::TransportedAllocationPrefix){
        if(action!=n::OwnerAction::AcceptFree && action!=n::OwnerAction::AcceptVictim)
            throw std::invalid_argument("Transported allocation count is valid only for its nominal allocation consumer");
        need(schema,PooledNativeOperand::RecordBytes,c?8:0,8);
    }else if(records!=n::RecordLayout::Compact) {
        if(records!=n::RecordLayout::DirectedAllocationBank || !c ||
           (action!=n::OwnerAction::AcceptFree && action!=n::OwnerAction::AcceptVictim))
            throw std::invalid_argument("Directed allocation records require a positive nominal free/victim consumer");
        need(schema,PooledNativeOperand::Records,times(plan.owner.maximumUnique(),sizeof(p::AllocationDecision)),8);
        need(schema,PooledNativeOperand::AllocationSlice,sizeof(n::DirectedSlice),8);
    }
    if(condition!=n::SourceCondition::Always) {
        const bool success=condition==n::SourceCondition::AfterSuccess &&
            (action==n::OwnerAction::ReadRows || action==n::OwnerAction::ApplySourceDelivery || action==n::OwnerAction::ValidateFills ||
             action==n::OwnerAction::AcquirePublication || action==n::OwnerAction::MakeReady || action==n::OwnerAction::ReadyReceipt);
        const bool failure=condition==n::SourceCondition::AfterFailure && action==n::OwnerAction::CancelSource;
        if(!success && !failure)throw std::invalid_argument("Pooled source gate is incompatible with its nominal owner operation");
        if(action!=n::OwnerAction::ApplySourceDelivery && action!=n::OwnerAction::CancelSource)
            need(schema,PooledNativeOperand::Outcome,sizeof(n::SourceFrontier),8);
    }
    if(action==n::OwnerAction::AcquireReaderRetirement && condition!=n::SourceCondition::Always)
        throw std::invalid_argument("Reader-retirement gate must authenticate its own fresh nominal source producer");
    if(action==n::OwnerAction::ReadRows && condition!=n::SourceCondition::AfterSuccess)
        throw std::invalid_argument("Pinned-hit read requires its original acquired source outcome");
    return schema;
}
PooledNativeOperandSchema pooledNativeSchema(n::CoordinatorAction action,const n::CoordinatorParameters& plan,std::optional<pooled_native::OriginalRows> original_rows,n::SourceCondition condition)
{
    PooledNativeOperandSchema schema{};const auto participants=plan.policy.participants(),c=plan.native.reply_capacity;
    if((action==n::CoordinatorAction::Begin)!=bool(original_rows))
        throw std::invalid_argument("Pooled native query descriptor belongs exactly to Begin");
    switch(action){
    case n::CoordinatorAction::Begin:{
        const auto descriptor=*original_rows;const auto rows=descriptor.rows;
        if(descriptor.id_width!=n::OriginalIdWidth::UInt32 && descriptor.id_width!=n::OriginalIdWidth::UInt64)
            throw std::invalid_argument("Pooled original producer width is unsupported");
        const auto bytes=descriptor.id_width==n::OriginalIdWidth::UInt32?4u:8u;
        const auto id=rows.captureIdentity();
        if(id.first_physical_row || id.physical_rows<=0 || rows.physicalRows()!=rows.capacity() ||
            static_cast<std::uint64_t>(rows.physicalRows())>plan.policy.maximumOriginal())
            throw std::invalid_argument("Pooled native Begin requires the complete original counted prefix");
        need(schema,PooledNativeOperand::Ids,times(rows.physicalRows(),bytes),bytes);
        need(schema,PooledNativeOperand::Admission,8,8);
        need(schema,PooledNativeOperand::OriginalCount,id.count?4:0,4);break;
    }
    case n::CoordinatorAction::AcquirePins:case n::CoordinatorAction::AcquireFree:case n::CoordinatorAction::AcquireVictims:
        need(schema,PooledNativeOperand::Records,times(c,sizeof(p::Grant)),8);
        need(schema,PooledNativeOperand::RecordBytes,c?8:0,8);break;
    case n::CoordinatorAction::SealPins:need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(p::PinHeader)),8);break;
    case n::CoordinatorAction::PlanFree:need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(n::Availability)),8);break;
    case n::CoordinatorAction::CompleteFree:need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(p::FreeAllocationReceipt)),8);break;
    case n::CoordinatorAction::CompleteVictims:need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(p::VictimAllocationReceipt)),8);break;
    case n::CoordinatorAction::SealSource:need(schema,PooledNativeOperand::Source,sizeof(n::SourceFrontier),8);break;
    case n::CoordinatorAction::Publish:
        need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(p::ReadyHeader)),8);
        need(schema,PooledNativeOperand::Source,sizeof(n::SourceFrontier),8);break;
    case n::CoordinatorAction::Cancel:
        need(schema,PooledNativeOperand::Records,times(c,sizeof(p::ReleaseReceipt)),8);
        need(schema,PooledNativeOperand::RecordBytes,c?8:0,8);
        need(schema,PooledNativeOperand::Source,sizeof(n::SourceFrontier),8);break;
    case n::CoordinatorAction::Finish:need(schema,PooledNativeOperand::Frontier,times(participants,sizeof(p::RetiredHeader)),8);break;
    case n::CoordinatorAction::ImportLocalPinHeader:case n::CoordinatorAction::ImportLocalAvailability:
    case n::CoordinatorAction::ImportLocalFreeReceipt:case n::CoordinatorAction::ImportLocalVictimReceipt:
        need(schema,PooledNativeOperand::Frontier,n::localFrontierBytes(action),8);
        need(schema,PooledNativeOperand::RecordBytes,sizeof(std::uint64_t),8);
        need(schema,PooledNativeOperand::OwnerControl,sizeof(p::OwnerFrame),8);
        need(schema,PooledNativeOperand::AllocationControl,sizeof(p::AllocationFrame),8);
        need(schema,PooledNativeOperand::HeaderKind,sizeof(n::OwnerHeaderKind),alignof(n::OwnerHeaderKind));break;
    case n::CoordinatorAction::PinCommands:case n::CoordinatorAction::PlanVictims:break;
    default:throw std::invalid_argument("Unknown pooled native coordinator phase schema");
    }
    if(condition!=n::SourceCondition::Always &&
       !((condition==n::SourceCondition::AfterSuccess && action==n::CoordinatorAction::Publish) ||
         (condition==n::SourceCondition::AfterFailure && action==n::CoordinatorAction::Cancel)))
        throw std::invalid_argument("Pooled source gate is incompatible with its nominal coordinator operation");
    return schema;
}
PooledImmutableOwnerOperands::PooledImmutableOwnerOperands(n::OwnerAction action,const PooledImmutableOwnerNativePlan& plan,PooledNativeOperandSpans spans,n::SourceCondition condition,n::RecordLayout records,std::optional<PooledImmutableCoordinatorNativePlan> localCoordinator)
    :action_(action),condition_(condition),records_(records),plan_(plan.parameters()),schema_(pooledNativeSchema(action,plan_,condition,records)),spans_(std::move(spans))
{
    if((records_==n::RecordLayout::DirectedCommandBank)!=bool(localCoordinator))
        throw std::invalid_argument("Local directed commands require their exact original coordinator plan only");
    if(localCoordinator){
        localCoordinator->requireOwnerPlan(plan);
        if(localCoordinator->allocator()!=plan.allocator())throw std::invalid_argument("Local commands require the same original allocator");
        const auto& c=localCoordinator->parameters();
        const auto& invocation=spans_[index(PooledNativeOperand::Invocation)];
        const auto& commands=spans_[index(PooledNativeOperand::Records)];
        const auto& slice=spans_[index(PooledNativeOperand::AllocationSlice)];
        const auto coordinate=plan_.owner.coordinate();
        if(!invocation || !commands || !slice || invocation->root!=commands->root || slice->root!=commands->root ||
            invocation->buffer!=BufferId::QFN_POOLED_COORDINATOR_NATIVE || commands->buffer!=invocation->buffer || slice->buffer!=invocation->buffer ||
            invocation->offset!=c.native.invocation || commands->offset!=c.native.pin_commands ||
            slice->offset!=c.native.pin_slices+size_t{coordinate}*sizeof(n::DirectedSlice) ||
            commands->bytes!=times(c.native.reply_capacity,sizeof(p::OwnerCommand)))
            throw std::invalid_argument("Local commands replaced the original coordinator command region or owner slice");
        need(schema_,PooledNativeOperand::Records,commands->bytes,8);local_coordinator_identity_=c.identity;
    }
    declarations();const auto original=roots(spans_);prepared_.emplace(plan_.device,original);
}
void PooledImmutableOwnerOperands::declarations()const{requireSpanDeclarations(spans_,schema_);}
void PooledImmutableOwnerOperands::seal(){declarations();prepared_->seal();requirePrepared(spans_,schema_,plan_.device);retainInputs(spans_,plan_.device,captured_);}
void PooledImmutableOwnerOperands::validateIfSealed()const{declarations();prepared_->validateIfSealed();}
void PooledImmutableOwnerOperands::requireDeclarations(n::OwnerAction action,const PooledImmutableOwnerNativePlan& plan)const
{
    if(action!=action_ || plan.parameters().identity!=plan_.identity)throw std::invalid_argument("Pooled owner operands changed operation or geometry");
    validateIfSealed();
}
void PooledImmutableOwnerOperands::require(n::OwnerAction action,const PooledImmutableOwnerNativePlan& plan)const
{
    requireDeclarations(action,plan);prepared_->validate();requirePrepared(spans_,schema_,plan_.device);requireInputs(spans_,plan_.device,captured_);
}
std::uint64_t PooledImmutableOwnerOperands::declarationIdentity()const{validateIfSealed();return combine(combine(combine(declaration(spans_,plan_.identity,static_cast<std::uint64_t>(action_)),static_cast<unsigned>(condition_)),static_cast<unsigned>(records_)),local_coordinator_identity_);}
std::uint64_t PooledImmutableOwnerOperands::captureIdentity()const{requireInputs(spans_,plan_.device,captured_);return combine(declarationIdentity(),prepared_->preparedIdentity());}
n::OwnerInputs PooledImmutableOwnerOperands::native()const
{
    return {static_cast<const n::Invocation*>(address(spans_,PooledNativeOperand::Invocation)),
        address(spans_,PooledNativeOperand::Records),static_cast<const std::uint64_t*>(address(spans_,PooledNativeOperand::RecordBytes)),
        address(spans_,PooledNativeOperand::Frontier),static_cast<const std::byte*>(address(spans_,PooledNativeOperand::Payload)),
        static_cast<const std::uint64_t*>(address(spans_,PooledNativeOperand::PayloadBytes)),condition_,
        condition_==n::SourceCondition::Always?nullptr:static_cast<const n::SourceFrontier*>(address(spans_,
            action_==n::OwnerAction::ApplySourceDelivery || action_==n::OwnerAction::CancelSource?
                PooledNativeOperand::Frontier:PooledNativeOperand::Outcome)),records_,
        static_cast<const n::DirectedSlice*>(address(spans_,PooledNativeOperand::AllocationSlice)),
        records_==n::RecordLayout::DirectedAllocationBank?spans_[index(PooledNativeOperand::Records)]->offset:0,
        (records_==n::RecordLayout::DirectedAllocationBank || records_==n::RecordLayout::DirectedCommandBank)?spans_[index(PooledNativeOperand::Records)]->bytes:0};
}
PooledImmutableCoordinatorOperands::PooledImmutableCoordinatorOperands(n::CoordinatorAction action,const PooledImmutableCoordinatorNativePlan& plan,
    std::optional<pooled_native::OriginalRows> original_rows,PooledNativeOperandSpans spans,n::SourceCondition condition,std::shared_ptr<PooledImmutableOwnerPreparedStorage> localOwner)
    :action_(action),condition_(condition),plan_(plan.parameters()),rows_(original_rows),local_owner_(std::move(localOwner)),schema_(pooledNativeSchema(action,plan_,rows_,condition)),spans_(std::move(spans))
{
    const bool localGrants=action_==n::CoordinatorAction::AcquirePins || action_==n::CoordinatorAction::AcquireFree || action_==n::CoordinatorAction::AcquireVictims;
    if((n::localFrontierImport(action_) && !local_owner_) || (local_owner_ && !n::localFrontierImport(action_) && !localGrants))
        throw std::invalid_argument("Original local owner belongs only to local metadata imports or direct Grant acquisition");
    if(local_owner_){
        if(n::localFrontierImport(action_) && (plan_.policy.participants()<2 || !plan_.native.ingress_frontier))
            throw std::invalid_argument("Local import requires the original distributed frontier, never P1 offset zero");
        plan.requireOwnerPlan(local_owner_->plan());
        if(plan.allocator()!=local_owner_->plan().allocator())throw std::invalid_argument("Local metadata import requires the original equal allocator");
        const auto roots=local_owner_->roots();const auto& o=local_owner_->plan().parameters();
        const auto& publication=spans_[index(n::localFrontierImport(action_)?PooledNativeOperand::Frontier:PooledNativeOperand::Records)];
        const auto& count=spans_[index(PooledNativeOperand::RecordBytes)];
        const auto offset=n::localFrontierImport(action_)?o.native.header:o.native.replies;
        const auto countOffset=o.native.progress+(n::localFrontierImport(action_)?offsetof(n::OwnerProgress,header_bytes):offsetof(n::OwnerProgress,reply_bytes));
        if(!publication || !count || publication->root!=roots.native || count->root!=roots.native ||
            publication->buffer!=BufferId::QFN_POOLED_OWNER_NATIVE || count->buffer!=publication->buffer ||
            publication->offset!=offset || count->offset!=countOffset || (localGrants && !o.native.reply_capacity))
            throw std::invalid_argument("Local import/acquisition replaced its actual original owner publication");
        if(n::localFrontierImport(action_)){
            const auto& control=spans_[index(PooledNativeOperand::OwnerControl)];const auto& allocation=spans_[index(PooledNativeOperand::AllocationControl)];
            const auto& kind=spans_[index(PooledNativeOperand::HeaderKind)];
            if(!control || !allocation || control->root!=roots.frame || allocation->root!=roots.frame ||
                control->buffer!=BufferId::QFN_POOLED_OWNER_FRAME || allocation->buffer!=control->buffer ||
                control->offset!=o.frame.control || allocation->offset!=o.frame.allocation_control || !kind || kind->root!=roots.native ||
                kind->buffer!=BufferId::QFN_POOLED_OWNER_NATIVE || kind->offset!=o.native.progress+offsetof(n::OwnerProgress,header_kind))
                throw std::invalid_argument("Local metadata import replaced its actual original producer phase/wave controls");
        }
        if(localGrants)need(schema_,PooledNativeOperand::Records,times(o.native.reply_capacity,sizeof(p::Grant)),8);
        local_coordinate_=o.owner.coordinate();local_reply_capacity_=o.native.reply_capacity;local_owner_identity_=local_owner_->captureIdentity();
    }
    declarations();const auto original=roots(spans_);
    // Metadata-only phases may have no input tensors. Their storage owner
    // remains independently sealed; no dummy operand is manufactured.
    if(!original.empty())prepared_.emplace(plan_.device,original);
}
void PooledImmutableCoordinatorOperands::declarations()const{
    requireSpanDeclarations(spans_,schema_);
    if(local_owner_ && local_owner_->captureIdentity()!=local_owner_identity_)
        throw std::logic_error("Local metadata import owner capture identity changed");
}
void PooledImmutableCoordinatorOperands::seal()
{
    declarations();if(prepared_)prepared_->seal();requirePrepared(spans_,schema_,plan_.device);retainInputs(spans_,plan_.device,captured_);
    if(action_==n::CoordinatorAction::Begin && rows_->rows.countOwner()!=address(spans_,PooledNativeOperand::OriginalCount))
        throw std::logic_error("Pooled original row range differs from its exact original count owner");
}
void PooledImmutableCoordinatorOperands::validateIfSealed()const{declarations();if(prepared_)prepared_->validateIfSealed();}
void PooledImmutableCoordinatorOperands::requireDeclarations(n::CoordinatorAction action,const PooledImmutableCoordinatorNativePlan& plan)const
{
    if(action!=action_ || plan.parameters().identity!=plan_.identity)throw std::invalid_argument("Pooled coordinator operands changed operation or geometry");
    validateIfSealed();
}
void PooledImmutableCoordinatorOperands::require(n::CoordinatorAction action,const PooledImmutableCoordinatorNativePlan& plan)const
{
    requireDeclarations(action,plan);if(prepared_)prepared_->validate();requirePrepared(spans_,schema_,plan_.device);requireInputs(spans_,plan_.device,captured_);
    if(action_==n::CoordinatorAction::Begin && rows_->rows.countOwner()!=address(spans_,PooledNativeOperand::OriginalCount))
        throw std::logic_error("Pooled original count owner changed after sealing");
}
std::uint64_t PooledImmutableCoordinatorOperands::declarationIdentity()const
{
    validateIfSealed();auto value=combine(declaration(spans_,plan_.identity,static_cast<std::uint64_t>(action_)),static_cast<unsigned>(condition_));
    if(local_owner_)value=combine(combine(combine(value,local_coordinate_),local_reply_capacity_),local_owner_identity_);
    if(rows_){
    value=combine(value,static_cast<unsigned>(rows_->id_width));
    const auto id=rows_->rows.captureIdentity();
    for(const auto word:{std::uint64_t{reinterpret_cast<std::uintptr_t>(id.count)},std::uint64_t{static_cast<unsigned>(id.count_capacity)},
        std::uint64_t{static_cast<unsigned>(id.first_physical_row)},std::uint64_t{static_cast<unsigned>(id.physical_rows)},
        std::uint64_t{static_cast<unsigned>(id.repeats_per_row)},std::uint64_t{id.schema_version}})value=combine(value,word);
    }
    return value;
}
std::uint64_t PooledImmutableCoordinatorOperands::captureIdentity()const
{requireInputs(spans_,plan_.device,captured_);return combine(declarationIdentity(),prepared_?prepared_->preparedIdentity():pooled_native::Version);}
n::CoordinatorInputs PooledImmutableCoordinatorOperands::native()const
{
    return {address(spans_,PooledNativeOperand::Records),
        static_cast<const std::uint64_t*>(address(spans_,PooledNativeOperand::RecordBytes)),address(spans_,PooledNativeOperand::Frontier),
        static_cast<const n::SourceFrontier*>(address(spans_,PooledNativeOperand::Source)),condition_,
        condition_==n::SourceCondition::Always?nullptr:static_cast<const n::SourceFrontier*>(address(spans_,PooledNativeOperand::Source)),local_coordinate_,local_reply_capacity_,
        static_cast<const p::OwnerFrame*>(address(spans_,PooledNativeOperand::OwnerControl)),
        static_cast<const p::AllocationFrame*>(address(spans_,PooledNativeOperand::AllocationControl)),
        static_cast<const n::OwnerHeaderKind*>(address(spans_,PooledNativeOperand::HeaderKind))};
}
n::CoordinatorQuery PooledImmutableCoordinatorOperands::query()const
{
    if(action_!=n::CoordinatorAction::Begin || !rows_)throw std::logic_error("Pooled native phase has no original Begin query");
    return {*rows_,address(spans_,PooledNativeOperand::Ids),
        static_cast<const std::uint64_t*>(address(spans_,PooledNativeOperand::Admission))};
}
}
