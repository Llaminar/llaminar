/**
 * @file Test__ImmutableAdaptiveLookupDevice.cpp
 * @brief Actual captured CUDA/HIP adaptive component over protocol2 source/PMA ownership.
 *
 * Exact native output and source/ACK/frame observations are terminal test oracles.
 * The bundle, source service, subordinate provider, original arena controller,
 * count and actual protocol are production implementations. This is component
 * qualification, not Rank/DGO/selector/serving or asynchronous fault-injection
 * certification. No test-only service/ACK publisher replaces the production one.
 */
#include "fixtures/qwen4exp/AdaptiveLookupNativeComponent.h"
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "kernels/qwen4exp/QwenAdaptiveTerminalCounters.h"
#include "utils/EmbeddingVerifierFormats.h"
#include <gtest/gtest.h>
#include <numeric>
#include <sstream>
#include <cstdio>

namespace llaminar2::test::adaptive_native {
namespace {
/** @return Exact immutable original source tensor from the canonical native-format inventory.
 * @param type Original native format. @param columns Exact complete-block logical row width. */
std::unique_ptr<TensorBase> nativeSource(TensorType type,int columns){
    const std::vector<std::size_t> shape{97,static_cast<std::size_t>(columns)};
    if(type==TensorType::FP32)return TestTensorFactory::createFP32Random(shape,-1.f,1.f,1701);
    if(type==TensorType::FP16)return TestTensorFactory::createFP16Random(shape,-1.f,1.f,1701);
    if(type==TensorType::BF16)return TestTensorFactory::createBF16Random(shape,-1.f,1.f,1701);
    for(const auto &format:quantizedVerifierFormats())if(format.tensor_type==type)return format.create(shape,1701);
    throw std::logic_error("Original adaptive format absent from canonical native fixture inventory");
}
/** @return Original coordinates with explicit duplicate reuse; Component authors padding.
 * @param rows Live original token count. @param workingSet Number of distinct source rows.
 * @param offset First selected source row; callers keep the complete range inside the source. */
std::vector<std::int64_t> workload(int rows,std::size_t workingSet,std::uint64_t offset=0){
    std::vector<std::int64_t> result(static_cast<std::size_t>(rows)*16);
    for(std::size_t i=0;i<result.size();++i)result[i]=static_cast<std::int64_t>(offset+i%workingSet);return result;
}
/** @brief Check every live output bit and untouched inactive suffix against original native bytes.
 * @param c Actual completed native component. @param ids Original live source coordinates. */
void outputAndRecords(Component &c,std::span<const std::int64_t> ids){
    const auto got=c.read(c.output(),c.maximumIds*c.source.columns);
    auto expected=std::vector<float>(got.size(),-12345.f);const auto bytes=c.geometry->rowBytes();
    std::vector<std::byte> selected(ids.size()*bytes);
    for(std::size_t i=0;i<ids.size();++i)std::memcpy(selected.data()+i*bytes,c.source.native.data()+static_cast<std::size_t>(ids[i])*bytes,bytes);
    decodeCPUImmutableLookupRows(c.geometry->format(),c.source.columns,bytes,selected.data(),ids.size(),expected.data());
    EXPECT_EQ(std::memcmp(got.data(),expected.data(),got.size()*sizeof(float)),0);
    const auto f=c.frameState();EXPECT_EQ(f.phase,policy::Phase::Idle);EXPECT_EQ(f.status,policy::Status::Success);
    EXPECT_EQ(f.original_ids,ids.size());EXPECT_EQ(f.observed_ids,ids.size());EXPECT_EQ(f.held_pins,0u);
    const auto p=c.protocolState();EXPECT_EQ(p.issuedSequence,p.consumedSequence);EXPECT_EQ(p.generation,p.admissionGeneration);EXPECT_EQ(p.admissionGeneration,c.generation);
    EXPECT_EQ(c.read(c.control->error(),1).front(),0);
    EXPECT_EQ(c.kernels->captureIdentity(*c.storage),c.identity);EXPECT_EQ(c.control->captureIdentity(),c.controlIdentity);EXPECT_EQ(c.query->token_rows.captureIdentity(),c.rowIdentity);
    const auto *base=static_cast<const std::byte*>(c.frame->gpu_data_ptr());
    const auto unique=c.read(reinterpret_cast<const policy::UniqueRecord*>(base+c.geometry->frame().unique),f.unique_rows);
    const auto acks=c.read(reinterpret_cast<const ImmutableRowInvalidationAcknowledgement*>(base+c.geometry->frame().acknowledgements),f.source_rows);
    const auto response=c.read(base+c.geometry->frame().native_response,c.maximumIds*bytes);
    EXPECT_TRUE(std::all_of(response.begin()+f.source_rows*bytes,response.end(),[](std::byte b){return b==std::byte{0xa5};}));
    for(const auto &u:unique){
        EXPECT_EQ(u.flags&4u,0u)<<"Every actual native reader pin retired";
        if(!(u.flags&policy::InitialHit)){
            ASSERT_LT(u.response_coordinate,acks.size());
            EXPECT_EQ(acks[u.response_coordinate].destination_grant,(u.flags&policy::ReservedFill)?u.expected_slot_generation:0u);
            EXPECT_EQ(std::memcmp(response.data()+u.response_coordinate*bytes,c.source.native.data()+u.row*bytes,bytes),0);
        }
        if(u.slot!=policy::None){
            ASSERT_LT(u.slot,c.geometry->slots());
            const auto *cache=static_cast<const std::byte*>(c.cache->gpu_data_ptr());
            const auto slot=c.read(reinterpret_cast<const policy::Slot32*>(cache+c.geometry->cache().slots)+u.slot,1).front();
            EXPECT_EQ(slot.generation,u.expected_slot_generation);EXPECT_EQ(slot.row,u.row);
            EXPECT_EQ(policy::Operations<policy::Slot32>::state(slot),policy::SlotState::Ready);
            EXPECT_FALSE(policy::Operations<policy::Slot32>::pinned(slot));
            const auto native=c.read(cache+c.geometry->cache().payload+u.slot*c.geometry->slotStride(),bytes);
            EXPECT_EQ(std::memcmp(native.data(),c.source.native.data()+u.row*bytes,bytes),0);
        }
    }
}
/** @brief Preserve the original terminal state before close changes RequestComplete or any new admission.
 * @details These are bounded diagnostic copies on the actual fixture stream, never inference authorities.
 * The source worker may still be Serving: mapped records are an observation, not a newly acquired response.
 */
struct FormatFrameObservation
{
    policy::FrameControl frame;
    ImmutableLookupDeviceState protocol;
    std::vector<std::int32_t> controller;
    std::string mapped;
};
/** @brief Serialize only live wire words, excluding each aligned header's storage padding.
 * @tparam T Actual protocol record type. @param out Failure report. @param name Wire record label.
 * @param value Original observed record. @param liveBytes Canonical wire extent, not allocation capacity.
 */
template<class T>
void appendWireWords(std::ostream &out,const char *name,const T &value,std::size_t liveBytes)
{
    require(liveBytes<=sizeof(T)&&liveBytes%sizeof(std::uint64_t)==0,"Invalid diagnostic wire extent");
    std::array<std::uint64_t,sizeof(T)/sizeof(std::uint64_t)> words{};
    std::memcpy(words.data(),&value,liveBytes);
    out<<name<<" wire_words=[";
    for(std::size_t i=0;i<liveBytes/sizeof(std::uint64_t);++i){if(i)out<<',';out<<words[i];}
    out<<"]\n";
}
/** @return Bounded original mapped request/completion/retirement and live intent/ACK observations.
 * @param c Completed native component retaining the actual source aliases.
 * @param requestRows Original observed protocol count; corrupt counts never become copy extents.
 */
std::string observeMappedFormatFrame(Component &c,std::uint64_t requestRows)
{
    const auto &view=c.sourceBinding->deviceView();
    std::ostringstream out;
    const auto request=c.read(view.base.request,1).front();
    const auto completion=c.read(view.base.completion,1).front();
    const auto acknowledgement=c.read(view.base.acknowledgement,1).front();
    appendWireWords(out,"request",request,kLookupRequestLiveBytes);
    appendWireWords(out,"completion",completion,kLookupCompletionLiveBytes);
    out<<"mapped_ack consumed="<<acknowledgement.consumedSequence
       <<" closed_generation="<<acknowledgement.closedGeneration<<'\n';
    if(requestRows>c.maximumIds)
    {
        out<<"live_records_not_read: protocol.idCount="<<requestRows
           <<" exceeds admitted maximumIds="<<c.maximumIds<<'\n';
        return out.str();
    }
    const auto count=static_cast<std::size_t>(requestRows);
    const auto ids=c.read(view.base.ids,count);
    const auto intents=c.read(view.intents,count);
    const auto acks=c.read(view.acknowledgements,count);
    for(std::size_t i=0;i<count;++i)
        out<<"mapped_row["<<i<<"] id="<<ids[i]<<" intent="<<static_cast<std::uint64_t>(intents[i].intent)
           <<" grant="<<intents[i].destination_grant<<" ack="<<acks[i].destination_grant<<'\n';
    return out.str();
}
/** @return First terminal frame/protocol/controller observations on their original roots.
 * @param c Actual completed native component, before any close or Preserve admission.
 */
FormatFrameObservation observeFormatFrame(Component &c)
{
    return {c.frameState(),c.protocolState(),
            c.read(c.control->row(),sampling_math::kDeviceGenerationControlCount),{}};
}
/** @return Whether the preserved frame can enter the existing successful-format assertions.
 * @param observed Original terminal observations; no state is repaired or inferred from output bytes.
 */
bool successfulFormatFrame(const FormatFrameObservation &observed)
{
    return observed.frame.phase==policy::Phase::Idle&&observed.frame.status==policy::Status::Success&&
        observed.frame.held_pins==0&&observed.protocol.status==0&&
        observed.protocol.issuedSequence==observed.protocol.consumedSequence&&
        observed.controller[sampling_math::kDeviceGenerationControlErrorCode]==0&&
        observed.controller[sampling_math::kDeviceGenerationControlOk]!=0;
}
/** @brief Retire the actual source interval before diagnosing failure; never replay a poisoned owner.
 * @param c Actual completed component with its current service ticket still retained.
 * @param ids Original live IDs used by the unchanged successful output/ACK assertions.
 * @return Success only after the unchanged successful-frame checks and actual close/await.
 * @note A failed native frame is observed before close and reported after retirement. If close throws,
 * service diagnostics are withheld: Component's existing destructor owns final fence/cancel/await cleanup.
 */
::testing::AssertionResult verifyAndRetireFormatFrame(Component &c,std::span<const std::int64_t> ids)
{
    auto observed=observeFormatFrame(c);
    const bool frameOk=successfulFormatFrame(observed);
    if(frameOk)
        outputAndRecords(c,ids);
    // Existing EXPECTs and helper-local ASSERTs are authoritative too. Retire
    // before returning fatal failure even if the coarse protocol words passed.
    const bool observationOk=frameOk&&!::testing::Test::HasFailure();
    if(!observationOk)
        observed.mapped=observeMappedFormatFrame(c,observed.protocol.idCount);
    std::ostringstream out;
    out<<"First adaptive format frame failed; no Preserve admission/replay follows.\n"
       <<"frame phase="<<static_cast<std::uint32_t>(observed.frame.phase)
       <<" status="<<static_cast<std::uint32_t>(observed.frame.status)
       <<" original_ids="<<observed.frame.original_ids<<" unique_rows="<<observed.frame.unique_rows
       <<" source_rows="<<observed.frame.source_rows<<" observed_ids="<<observed.frame.observed_ids
       <<" held_pins="<<observed.frame.held_pins<<" policy_started="<<observed.frame.policy_started
       <<" promotions="<<observed.frame.promotions<<" evictions="<<observed.frame.evictions<<'\n';
    appendWireWords(out,"original_protocol",observed.protocol,sizeof(observed.protocol));
    out<<"original_controller=[";
    for(std::size_t i=0;i<observed.controller.size();++i){if(i)out<<',';out<<observed.controller[i];}
    out<<"]\n";
    if(!observationOk)
    {
        out<<"mapped_before_close (source worker may still be Serving):\n"<<observed.mapped;
        // An absent original service thread can block the first await itself.
        // Emit and flush the preserved observation before attempting retirement;
        // this failure is never deferred until a possibly unreachable return.
        ADD_FAILURE()<<out.str();
        std::fflush(stdout);
        std::fflush(stderr);
    }
    bool retired=false,closeOk=false;
    std::string closeFailure;
    try
    {
        closeOk=c.close();
        retired=true; // close() returned only after the real service await, including a false result.
    }
    catch(const std::exception &error)
    {
        closeFailure=error.what();
    }
    if(observationOk&&closeOk)
        return ::testing::AssertionSuccess();
    out<<"close_returned="<<retired<<" close_ok="<<closeOk<<" close_exception="<<closeFailure<<'\n';
    if(retired)
    {
        // Unlike the pre-close snapshot, the worker can no longer mutate these records or its diagnostics.
        out<<"mapped_after_source_retirement:\n"<<observeMappedFormatFrame(c,observed.protocol.idCount);
        const auto stats=c.service->stats();
        out<<"source requests="<<stats.requests<<" ids="<<stats.ids<<" native_bytes="<<stats.nativeBytes
           <<" failures="<<stats.failures<<" terminal_bytes="<<stats.terminalBytes
           <<" last_failure="<<c.service->lastFailure()<<'\n';
    }
    else
        out<<"Source diagnostics withheld until destructor-owned worker retirement.\n";
    return ::testing::AssertionFailure()<<out.str();
}
/** @brief Capture once, then sweep live prefixes, warm no-I/O, and exact reset preservation.
 * @param device Exact backend endpoint from the explicit test lane. */
void lifecycle(DeviceId device){
    auto *backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal){GTEST_SKIP()<<"Requested backend unavailable";}
    if(!hasCPUBackend())initCPUBackend(-1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{
        for(const std::uint32_t slots:{0u,1u,64u})for(const int capacity:{1,15,512,1024,2048}){
            SCOPED_TRACE(::testing::Message()<<device.toString()<<" C="<<slots<<" M="<<capacity);
            const auto native=iq4Source(97,160);Source source(native,97,160,GGUFTensorType::IQ4_NL);
            Component c(source,device,worker,capacity,slots,AdaptiveFixtureGraphFamily::LookupAndTerminal);
            ASSERT_TRUE(c.graphLease);
            EXPECT_EQ(c.graphLease->bytes(), 2u * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
            if (capacity == 15 && slots == 1)
            {
                EXPECT_THROW(c.createBundle(),std::logic_error);
                c.declareStorage();
                EXPECT_THROW(c.createBundle(),std::logic_error);
            }
            c.prepareBundle();c.initializeMetadata();c.admit();
            const auto initial=workload(capacity,1);c.workload(capacity,initial);c.capture();
            EXPECT_EQ(c.sourceBinding->declareStorage(c.geometry->nativePlan(),c.cache,c.frame,c.protocol),c.storage);
            auto replacement=std::make_shared<INT8Tensor>(std::vector<std::size_t>{c.geometry->frame().bytes});
            EXPECT_THROW(c.sourceBinding->declareStorage(c.geometry->nativePlan(),c.cache,replacement,c.protocol),std::invalid_argument);
            EXPECT_THROW(c.kernels->gather(*c.storage,*c.query,*c.control,nullptr),std::invalid_argument);
            EXPECT_THROW(c.kernels->gather(*c.storage,*c.query,*c.control,reinterpret_cast<void*>(1)),std::invalid_argument);
            if(capacity==15&&slots==1){
                Component other(source,device,worker,capacity,slots,AdaptiveFixtureGraphFamily::None);other.prepareBundle();
                EXPECT_FALSE(other.graphLease);
                EXPECT_THROW(other.kernels->admit(*other.storage,1,ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce,other.stream),std::invalid_argument);
                EXPECT_THROW(other.kernels->gather(*other.storage,*other.query,*other.control,other.stream),std::logic_error);
                auto denied=worker.createGraphCapture(c.stream);ASSERT_TRUE(denied);
                {
                    ScopedBackendGraphCapture scope(worker,*denied,"adaptive initialization rejection");
                    ASSERT_TRUE(scope.begin());
                    EXPECT_THROW(other.kernels->initialize(*other.storage,other.stream),std::logic_error);
                    scope.finish();
                }
                other.initializeMetadata();other.admit();
                EXPECT_THROW(c.kernels->gather(*other.storage,*c.query,*c.control,c.stream),std::invalid_argument);
            }
            if (capacity >= 15)
            {
                auto invalid=*c.query;
                invalid.token_rows=DeviceRowRange::deviceCounted(capacity/2,c.live()).repeatEachRow(2);
                EXPECT_THROW(c.kernels->gather(*c.storage,invalid,*c.control,c.stream),std::invalid_argument);
            }
            EXPECT_THROW(c.kernels->initialize(*c.storage,c.stream),std::logic_error);
            EXPECT_THROW(c.kernels->admit(*c.storage,c.generation,ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce,c.stream),std::invalid_argument);
            const auto before=c.protocolState();c.launch();c.join();outputAndRecords(c,initial);
            EXPECT_EQ(c.protocolState().issuedSequence,before.issuedSequence+1);EXPECT_TRUE(c.close());
            for(const int live:{capacity,std::min(9,capacity),0,1,capacity}){
                const auto stats=c.service->stats();const auto seq=c.protocolState().issuedSequence;
                c.admit();const auto ids=workload(live,1);c.workload(live,ids);c.launch();c.join();outputAndRecords(c,ids);
                const auto f=c.frameState();const auto p=c.protocolState();
                const std::uint64_t misses=(!live||slots)?0u:1u;
                EXPECT_EQ(f.source_rows,misses);EXPECT_EQ(p.issuedSequence,seq+misses);EXPECT_TRUE(c.close());
                const auto after=c.service->stats();EXPECT_EQ(after.requests-stats.requests,misses);
                EXPECT_EQ(after.ids-stats.ids,misses);EXPECT_EQ(after.nativeBytes-stats.nativeBytes,misses*90);
                EXPECT_EQ(after.idBytes-stats.idBytes,misses*8);
                EXPECT_EQ(after.acknowledgementBytes-stats.acknowledgementBytes,misses*8);
                EXPECT_EQ(after.metadataBytes-stats.metadataBytes,misses*(kLookupRequestLiveBytes+kLookupCompletionLiveBytes+sizeof(ImmutableRowSubordinateIntent)+sizeof(ImmutableRowInvalidationAcknowledgement)));
                if(!misses){EXPECT_EQ(after.ramRows,stats.ramRows);EXPECT_EQ(after.sourceRows,stats.sourceRows);EXPECT_EQ(after.duplicateRows,stats.duplicateRows);EXPECT_EQ(after.cacheAdmissions,stats.cacheAdmissions);EXPECT_EQ(after.cacheEvictions,stats.cacheEvictions);EXPECT_EQ(after.adaptiveBatches,stats.adaptiveBatches);}
                // One GPU close word and the CPU worker's generation/invocation
                // retirement pair are required even for an empty interval.
                EXPECT_EQ(after.terminalBytes-stats.terminalBytes,3u*sizeof(std::uint64_t));
            }
            if(capacity>1&&slots==64){
                const auto pinned=workload(capacity,64);c.admit();c.workload(capacity,pinned);c.launch();c.join();outputAndRecords(c,pinned);EXPECT_TRUE(c.close());
                const auto pinnedBefore=c.protocolState().issuedSequence;
                auto missBeforeHits=workload(capacity,64);missBeforeHits[0]=96;
                c.admit();c.workload(capacity,missBeforeHits);c.launch();c.join();outputAndRecords(c,missBeforeHits);
                EXPECT_EQ(c.frameState().unique_rows,65u);EXPECT_EQ(c.frameState().source_rows,1u);
                EXPECT_EQ(c.protocolState().issuedSequence,pinnedBefore+1);EXPECT_TRUE(c.close());
                const auto unique=c.read(reinterpret_cast<const policy::UniqueRecord*>(static_cast<const std::byte*>(c.frame->gpu_data_ptr())+c.geometry->frame().unique),65);
                ASSERT_EQ(unique.front().row,96u);EXPECT_EQ(unique.front().slot,policy::None);
                EXPECT_EQ(unique.front().flags&policy::ReservedFill,0u)<<"All original hits were pinned before the earlier miss could choose a victim";
                const auto *ownerAddress=reinterpret_cast<const policy::OwnerControl*>(static_cast<const std::byte*>(c.cache->gpu_data_ptr())+c.geometry->cache().owner);
                const auto beforeEviction=c.read(ownerAddress,1).front();
                for(int repeat=0;repeat<3;++repeat){const auto ids=workload(capacity,1,96);c.admit();c.workload(capacity,ids);c.launch();c.join();outputAndRecords(c,ids);EXPECT_TRUE(c.close());}
                const auto afterEviction=c.read(ownerAddress,1).front();
                EXPECT_EQ(afterEviction.filled,64u);EXPECT_GT(afterEviction.next_slot_generation,beforeEviction.next_slot_generation);
                EXPECT_EQ(c.frameState().source_rows,0u)<<"Repeated hot row must become a real retained Ready hit after an actual full-cache replacement";
            }
        }
    });
}
/** @brief Cross every original native codec through captured source/cache/decode, without fallback.
 * @param device Exact backend endpoint from the explicit test lane. */
void formats(DeviceId device){
    auto *backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal){GTEST_SKIP()<<"Requested backend unavailable";}
    if(!hasCPUBackend())initCPUBackend(-1);auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{
        const std::array types{GGUFTensorType::F32,GGUFTensorType::F16,GGUFTensorType::BF16,GGUFTensorType::Q4_0,GGUFTensorType::Q4_1,GGUFTensorType::Q5_0,GGUFTensorType::Q5_1,GGUFTensorType::Q8_0,GGUFTensorType::Q2_K,GGUFTensorType::Q3_K,GGUFTensorType::Q4_K,GGUFTensorType::Q5_K,GGUFTensorType::Q6_K,GGUFTensorType::Q8_K,GGUFTensorType::IQ1_S,GGUFTensorType::IQ1_M,GGUFTensorType::IQ2_XXS,GGUFTensorType::IQ2_XS,GGUFTensorType::IQ2_S,GGUFTensorType::IQ3_XXS,GGUFTensorType::IQ3_S,GGUFTensorType::IQ4_NL,GGUFTensorType::IQ4_XS};
        for(const auto type:types)for(const std::uint32_t slots:{0u,1u}){
            SCOPED_TRACE(::testing::Message()<<device.toString()<<" type="<<int(type)<<" C="<<slots);
            auto original=nativeSource(ggufToTensorType(type),256);Source source({static_cast<const std::byte*>(original->raw_data()),original->size_bytes()},97,256,type);
            Component c(source,device,worker,15,slots,AdaptiveFixtureGraphFamily::LookupAndTerminal);c.prepareBundle();c.initializeMetadata();c.admit();const auto ids=workload(15,1);c.workload(15,ids);c.capture();
            c.launch();c.join();
            ASSERT_TRUE(verifyAndRetireFormatFrame(c,ids));
            c.admit();c.workload(1,workload(1,1));c.launch();c.join();
            ASSERT_TRUE(verifyAndRetireFormatFrame(c,workload(1,1)));
        }
    });
}
/** @brief Storage failures cannot publish output; clean controller cancellation preserves warm rows.
 * @param device Exact backend endpoint from the explicit test lane. */
void failures(DeviceId device){
    auto *backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal){GTEST_SKIP()<<"Requested backend unavailable";}
    if(!hasCPUBackend())initCPUBackend(-1);auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{
        for(const int failure:{0,1,2}){
            const auto bytes=iq4Source(97,160);Source source(bytes,97,160,GGUFTensorType::IQ4_NL);Component c(source,device,worker,15,1,AdaptiveFixtureGraphFamily::LookupAndTerminal);
            c.prepareBundle();c.initializeMetadata();c.admit();auto ids=workload(9,1);if(failure==0)ids[0]=-1;c.workload(9,ids);c.capture();
            if(failure==2){
                c.launch();c.join();outputAndRecords(c,ids);EXPECT_TRUE(c.close());
                c.admit();c.workload(9,ids);
                const int error=static_cast<int>(sampling_math::DeviceGenerationError::InvalidController);c.upload(c.control->error(),std::span<const int>(&error,1));
            }
            if(failure==1)source.removePayload();
            c.launch();c.join();
            EXPECT_NE(c.read(c.control->error(),1).front(),0);EXPECT_EQ(c.read(c.control->healthy(),1).front(),0);
            const auto actual=c.read(c.output(),c.maximumIds*160);EXPECT_TRUE(std::all_of(actual.begin(),actual.end(),[](float x){return x==-12345.f;}));
            EXPECT_EQ(c.frameState().held_pins,0u);EXPECT_EQ(c.frameState().phase,policy::Phase::Idle);
            const auto observedWords = c.read(c.control->row(), sampling_math::kDeviceGenerationControlCount);
            QwenAdaptiveCounterTotals observed;
            ASSERT_TRUE(readQwenAdaptiveCounters(observedWords.data(), observedWords.size(), observed));
            EXPECT_EQ(observed.calls, 1u);
            EXPECT_EQ(observed.completed_live_ids, 0u);
            EXPECT_EQ(observed.failed_frames, 1u);
            EXPECT_EQ(observed.source_rows, 0u);
            EXPECT_EQ(observed.source_bytes, 0u);
            EXPECT_EQ(observed.promotions, 0u);
            EXPECT_EQ(observed.evictions, 0u);
            EXPECT_TRUE(validQwenAdaptiveCounters(observed, c.geometry->rowBytes(), c.maximumIds));
            const auto state=c.protocolState();EXPECT_EQ(state.issuedSequence,state.consumedSequence);
            const bool serviceOk=c.close();
            if (failure == 1)
            {
                EXPECT_FALSE(serviceOk);
                EXPECT_THROW(c.service->requireHealthy(),std::runtime_error);
            }
            else
            {
                EXPECT_TRUE(serviceOk);
            }
            const auto owner=c.read(reinterpret_cast<const policy::OwnerControl*>(static_cast<const std::byte*>(c.cache->gpu_data_ptr())+c.geometry->cache().owner),1).front();
            if(failure==2){
                EXPECT_EQ(owner.flags,policy::OwnerInitialized);EXPECT_NE(owner.directory_root,policy::None);
                EXPECT_EQ(c.read(c.control->error(),1).front(),static_cast<int>(sampling_math::DeviceGenerationError::InvalidController));
                const auto before=c.service->stats();const auto sequence=c.protocolState().issuedSequence;
                c.admit();c.workload(9,ids);c.launch();c.join();outputAndRecords(c,ids);
                EXPECT_EQ(c.frameState().source_rows,0u);EXPECT_EQ(c.protocolState().issuedSequence,sequence);EXPECT_TRUE(c.close());
                const auto after=c.service->stats();EXPECT_EQ(after.requests,before.requests);EXPECT_EQ(after.ids,before.ids);EXPECT_EQ(after.nativeBytes,before.nativeBytes);
            }else {
                EXPECT_NE(owner.flags&policy::OwnerFailed,0u);EXPECT_EQ(owner.directory_root,policy::None);
                if (failure == 1)
                {
                    const auto generation=c.generation;
                    EXPECT_EQ(c.kernels->captureIdentity(*c.storage),c.identity);
                    EXPECT_THROW(c.admit(),std::runtime_error);
                    EXPECT_EQ(c.generation,generation);
                    EXPECT_FALSE(c.armed);
                }
                else {
                    const auto requests=c.service->stats().requests;const auto valid=workload(9,1);
                    c.admit();c.workload(9,valid);c.launch();c.join();
                    EXPECT_EQ(c.read(c.control->error(),1).front(),static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
                    const auto rejected=c.read(c.output(),c.maximumIds*160);EXPECT_TRUE(std::all_of(rejected.begin(),rejected.end(),[](float x){return x==-12345.f;}));
                    EXPECT_TRUE(c.close());EXPECT_EQ(c.service->stats().requests,requests);
                }
            }
        }
    });
}
} // namespace
#ifdef HAVE_CUDA
TEST(ImmutableAdaptiveLookupDevice,CUDALiveResetAndU0){lifecycle(DeviceId::cuda(0));}
TEST(ImmutableAdaptiveLookupDevice,CUDAOriginalFormats){formats(DeviceId::cuda(0));}
TEST(ImmutableAdaptiveLookupDevice,CUDAFailureIsolation){failures(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(ImmutableAdaptiveLookupDevice,ROCmLiveResetAndU0){lifecycle(DeviceId::rocm(0));}
TEST(ImmutableAdaptiveLookupDevice,ROCmOriginalFormats){formats(DeviceId::rocm(0));}
TEST(ImmutableAdaptiveLookupDevice,ROCmFailureIsolation){failures(DeviceId::rocm(0));}
#endif
} // namespace llaminar2::test::adaptive_native
