/**
 * @file Test__PooledImmutableProviderSourceDevice.cpp
 * @brief Retained real-provider pooled graph for the full loadable native-format inventory and sparse high-row IQ4_NL.
 *
 * Original canonical-format rows and sparse IQ4_NL high rows use the real subordinate service. The
 * production graph builder records every owner/coordinator/source operation;
 * no diagnostic delivery producer substitutes bytes or ACKs. Input uploads and
 * terminal readbacks below are test boundaries, never inference fallbacks.
 * Rejection traces add only a recorded invalid-input producer before original
 * captured phases; their scalar oracle and recorded resets grant no recovery claim.
 * Cold-declaration replay leaves original ID/output storage unprepared through
 * production graph creation and identity/contract queries; the actual executor
 * then materializes storage before capture-only seals retain its allocations.
 * Original local gates remain; new cross-vendor reader component gates retain
 * the same real-provider/root authorities through independent complete captures.
 * Serving, same-vendor native peer factories and opaque context admission remain
 * separately unqualified.
 */
#include "fixtures/qwen4exp/PooledImmutableNativeReplayBytes.h"
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "planning/ClusterInventoryGatherer.h"
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
#include "fixtures/execution/RetainedHostServiceCalibrationAccess.h"
#endif
#include "execution/qwen4exp/PooledImmutableLocalLookupGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantReadGraph.h"
#include "execution/qwen4exp/PooledImmutableCompleteCohort.h"
#include "execution/qwen4exp/PooledImmutableParticipantRetirementGraph.h"
#include "PooledACKReadyCancellation.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "tensors/NativeTensorExtent.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/NativeOrderingEventGroup.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "transfer/TransferEngine.h"
#include "utils/EmbeddingVerifierFormats.h"
#include "utils/PooledNgramReplayTrace.h"
#include "fixtures/qwen4exp/PooledImmutableNativeSourceFixture.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "kernels/qwen4exp/PooledImmutableNativeOps.h"
#define LLAMINAR_POOLED_IDENTITY_REJECTION_TESTS 1
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <map>
#include <type_traits>
#include <unistd.h>
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=n::p;
#include "PooledImmutableProviderReplay.inc"
/** @brief Actual native source-only/positive-owner successful cycle and request reset, serially on one explicit participant. */
void providerReplay(DeviceId device){
    if(!hasCPUBackend())initCPUBackend(-1);
    std::vector<ProviderReplaySpec> specs;
    for(auto slots:{0u,4u})for(auto maximum:{32u,513u})specs.push_back({std::make_shared<ProviderFile>(),maximum,slots,slots?2u:0u,maximum<=512?n::OriginalIdWidth::UInt32:n::OriginalIdWidth::UInt64});
    ProviderReplayPrograms programs(specs);programs.prepare(device);
    for(std::size_t i=0;i<specs.size();++i)programs.run(i,[&](ProviderReplay& replay){const auto maximum=replay.maximum;
        replay.admit(1);replay.replay(std::array<std::uint64_t,2>{101,102});replay.fullSourceCapacity();replay.replay(std::array<std::uint64_t,2>{101,102});
        const auto hot=maximum<=512?900:ProviderFile::High;std::vector<std::uint64_t> large(maximum,hot);large[0]=700;large[1]=701;
        replay.replay(large);replay.replay(std::array<std::uint64_t,1>{hot});replay.replay({});replay.close();
        replay.admit(2);replay.replay(std::array<std::uint64_t,1>{hot});replay.replay(large);replay.close();replay.rejectUniqueOverflow();
    });programs.retire();
}
/** @brief Use the same original retained graph for real provider failure and GPU-owned exact I/O abort. */
void providerFailureReplay(DeviceId device){
    if(!hasCPUBackend())initCPUBackend(-1);
    std::vector<ProviderReplaySpec> specs;std::vector<bool> aborts;
    for(auto slots:{0u,4u})for(const bool abort:{false,true}){specs.push_back({std::make_shared<ProviderFile>(),513,slots,slots?2u:0u,n::OriginalIdWidth::UInt64});aborts.push_back(abort);}
    ProviderReplayPrograms programs(specs);programs.prepare(device);
    for(std::size_t i=0;i<specs.size();++i)programs.run(i,[&](ProviderReplay& replay){const bool abortBeforePublication=aborts[i];
        replay.admit(1);replay.replay(std::array<std::uint64_t,2>{101,102});replay.close();
            replay.admit(2);std::vector<std::uint64_t> original(513,ProviderFile::High);original[0]=700;original[1]=701;
            replay.replayFailure(original,abortBeforePublication);
    });programs.retire();
}
#include "PooledCrossVendorReadReplay.inc"
#include "PooledExactIdentityBoundaryReplay.inc"
#include "PooledCrossVendorMultiOwnerReplay.inc"

/** @brief Cross the canonical original-native inventory through the new actual pooled source consumer on one backend. */
void providerAllFormatsReplay(DeviceId device,bool productionPLE=false){
    if(!hasCPUBackend())initCPUBackend(-1);std::vector<ProviderReplaySpec> specs;std::vector<std::string> labels;
    std::size_t supported=0,unsupported=0;
    for(const auto& entry:embeddingVerifierFormats()){
        const std::size_t columns=entry.prepared_native_quantized?768:257;
        auto original=entry.create({97,columns},1701);const auto format=providerFormat(original->native_type());
        if(!format){requireProvider(original->native_type()==TensorType::Q8_1,"Canonical embedding format has no supported original GGUF provider descriptor");
            EXPECT_THROW((void)ggufToTensorType(static_cast<GGUFTensorType>(9)),std::invalid_argument);
            std::cout<<"POOLED_PROVIDER_FORMAT_UNSUPPORTED format="<<entry.label<<" reason=no_original_GGUF_provider_descriptor\n";++unsupported;continue;}
        ++supported;
        for(auto slots:{0u,3u}){specs.push_back({std::make_shared<ProviderFile>(*original,*format,columns),productionPLE?64u:33u,slots,slots?2u:0u,productionPLE?n::OriginalIdWidth::UInt64:n::OriginalIdWidth::UInt32,productionPLE});labels.emplace_back(entry.label);}
    }
    ProviderReplayPrograms programs(specs);programs.prepare(device);
    for(std::size_t i=0;i<specs.size();++i)programs.run(i,[&](ProviderReplay& replay){auto& file=replay.file;const auto columns=file.columns,slots=std::size_t(replay.slots);const auto& label=labels[i];
                    std::cout<<"POOLED_PROVIDER_FORMAT format="<<label<<" columns="<<columns<<" row_bytes="<<file.row_bytes<<" slots="<<slots
                        <<" source_row_key_bits=32 original_id_bits="<<(productionPLE?64:32)<<" production_ple="<<productionPLE<<'\n';
                    if(productionPLE){
                        std::vector<std::uint64_t> large(64,81),small(16,81),first(16,1);
                        large[0]=79;large[1]=80;first[1]=2;
                        replay.admit(1);replay.replay(first);replay.fullSourceCapacity(16);
                        replay.replay(large);replay.replay(small);replay.replay({});replay.close();
                        replay.admit(2);replay.replay(first);replay.replay(large);replay.replay({});replay.close();
                    }else{
                        replay.admit(1);replay.replay(std::array<std::uint64_t,2>{1,2});replay.fullSourceCapacity(16);
                        std::vector<std::uint64_t> large(33,81);large[0]=79;large[1]=80;
                        replay.replay(large);replay.replay(std::array<std::uint64_t,1>{81});replay.replay({});replay.close();
                        replay.admit(2);replay.replay(std::array<std::uint64_t,3>{1,2,1});replay.replay(large);
                        std::array<std::uint64_t,16> fresh;for(std::size_t i=0;i<fresh.size();++i)fresh[i]=48+i;replay.replay(fresh);replay.replay({});replay.close();
                    }
    });programs.retire();
    EXPECT_EQ(unsupported,1u);EXPECT_EQ(supported+unsupported,embeddingVerifierFormats().size());
    std::cout<<"POOLED_PROVIDER_FORMAT_INVENTORY supported="<<supported<<" unsupported="<<unsupported<<" source_only_and_positive_owner=1 capture_reset_empty_tail=1\n";
}
/** @brief Execute the model builder's exact INT64/token-count lookup helper through retained native capture. */
void productionPLELookupReplay(DeviceId device){
    if(!hasCPUBackend())initCPUBackend(-1);std::vector<ProviderReplaySpec> specs;
    for(auto slots:{0u,4u})specs.push_back({std::make_shared<ProviderFile>(),64,slots,slots?2u:0u,n::OriginalIdWidth::UInt64,true});
    ProviderReplayPrograms programs(specs);programs.prepare(device);
    for(std::size_t i=0;i<specs.size();++i)programs.run(i,[&](ProviderReplay& replay){auto& file=replay.file;
        const auto signature=replay.boundary->captureIdentity();
        const auto ids=replay.arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=replay.arena->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
        const auto output=replay.arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        const auto tokens=DeviceRowRange::deviceCounted(4,static_cast<const std::int32_t*>(count->gpu_data_ptr()));
        ComputeGraph narrowed,missing;
        EXPECT_THROW(appendPooledImmutablePLELookupGraph(narrowed,*replay.arena,replay.boundary,replay.code,tokens,
            (q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,64*sizeof(std::uint32_t)}),
            (q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,4}),*replay.control,
            (q::PooledNativeTensorSpan{output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()}),"narrowed","narrowed_gather"),std::invalid_argument);
        EXPECT_THROW(appendPooledImmutablePLELookupGraph(missing,*replay.arena,replay.boundary,replay.code,tokens,
            (q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,64*sizeof(std::uint64_t)}),std::nullopt,*replay.control,
            (q::PooledNativeTensorSpan{output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()}),"missing","missing_gather"),std::invalid_argument);
        EXPECT_TRUE(narrowed.getExecutionOrder().empty());EXPECT_TRUE(missing.getExecutionOrder().empty());
        const auto* reader=replay.compute.getNode("ple_gather");ASSERT_TRUE(reader);
        EXPECT_EQ(reader->stage->type(),ComputeStageType::QWEN4EXP_NGRAM_LOOKUP);
        const auto dump=reader->stage->getDumpInfo();ASSERT_EQ(dump.outputs.size(),1u);
        EXPECT_EQ(dump.outputs.front().rows,4u);EXPECT_EQ(dump.outputs.front().cols,16*file.columns);
        std::vector<std::uint64_t> large(64,ProviderFile::High),small(16,900);
        for(std::size_t i=0;i<large.size();++i){if(i%4==1)large[i]=7;else if(i%4==2)large[i]=700;}
        EXPECT_NE(std::memcmp(file.original(ProviderFile::High).data(),file.original(7).data(),file.row_bytes),0);
        for(std::size_t i=0;i<small.size();++i)if(i%2)small[i]=ProviderFile::High;
        replay.admit(1);replay.replay(large);replay.replay(small);replay.replay({});replay.close();
        replay.admit(2);replay.replay(large);replay.replay(small);replay.close();
        EXPECT_EQ(replay.boundary->captureIdentity(),signature);
        const auto observation=replay.boundary->pooledStorage()->retainedOwnerObservation();
        EXPECT_EQ(observation.authority().get(),replay.memory.get());
        EXPECT_EQ(observation.lookupStagingBytes(),replay.memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation));
    });programs.retire();
}
/** @brief Replay full/partial/empty and request reset through one actual ACK/fill/cancellation frontier twenty times. */
void pooledACKReadyCancellation(DeviceId device){
    if(!hasCPUBackend())initCPUBackend(-1);std::vector<ProviderReplaySpec> specs;
    for(auto slots:{0u,2u})specs.push_back({std::make_shared<ProviderFile>(),64,slots,0,n::OriginalIdWidth::UInt64,false,OwnerAdmission::Original,ACKInterleaving::Cancel900BeforeReady});
    specs.push_back({std::make_shared<ProviderFile>(),64,2,0,n::OriginalIdWidth::UInt64,false,OwnerAdmission::Original,ACKInterleaving::OriginalUnchecked900});
    ProviderReplayPrograms programs(specs);programs.prepare(device);
    for(std::size_t i=0;i<2;++i)programs.run(i,[&](ProviderReplay& replay){
        auto* retained=replay.graph;const auto identity=replay.identity;std::uint64_t epoch=0;
        for(unsigned iteration=0;iteration<20;++iteration){
            const std::vector<std::uint64_t> warm(64,7);
            replay.admit(++epoch);replay.replay(warm);replay.replay({});replay.close();
            const auto count=iteration%3==0?64u:iteration%3==1?17u:1u;
            std::vector<std::uint64_t> cancelled(count,7);cancelled.back()=900;
            replay.admit(++epoch);replay.cancelledACK(cancelled);replay.close();
            replay.admit(++epoch);replay.replay(warm);replay.replay({});replay.close();
            EXPECT_EQ(replay.graph,retained);EXPECT_EQ(replay.boundary->captureIdentity(),identity);
        }
    });programs.run(2,[&](ProviderReplay& original){
        const std::vector<std::uint64_t> warm(64,7);original.admit(1);original.replay(warm);original.close();
        std::vector<std::uint64_t> cancelled(17,7);cancelled.back()=900;
        original.admit(2);original.cancelledACK(cancelled);original.close();
        std::cout<<"POOLED_ACK_ORIGINAL_UNCHECKED_CONTROL expected_ready900=1 expected_promotions=1 expected_consumed_witness=1\n";
    });programs.retire();
}
/**
 * @brief Replay original cold declarations, exact arena materialization and strict post-capture root seals.
 * @param device Explicit CUDA or ROCm participant; no backend substitution is admitted.
 * @note C0 and positive cache ownership share the real RAM/SSD provider. Full,
 * partial, empty and repeated request resets reuse the same retained graph.
 */
void productionColdArenaSealReplay(DeviceId device){
    if(!hasCPUBackend())initCPUBackend(-1);std::vector<ProviderReplaySpec> specs;
    for(auto slots:{0u,4u})for(auto maximum:{16u,64u,256u,1024u})specs.push_back({std::make_shared<ProviderFile>(),maximum,slots,slots?2u:0u,n::OriginalIdWidth::UInt64,true,OwnerAdmission::Original,ACKInterleaving::None,ActivationPreparation::ColdGraph});
    ProviderReplayPrograms programs(specs);programs.prepare(device);
    for(std::size_t i=0;i<specs.size();++i)programs.run(i,[&](ProviderReplay& replay){const auto maximum=replay.maximum;
        std::vector<std::uint64_t> large(maximum,ProviderFile::High),small(16,700);
        for(std::uint64_t generation=1;generation<=3;++generation){
            replay.admit(generation);replay.replay(large);replay.replay(small);replay.replay({});replay.close();
        }
        const auto ids=replay.arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS);
        const auto output=replay.arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        // Retained shared arena roles reject substitution before a stage could
        // acquire another owner's pointer, including rebinding the same root.
        EXPECT_FALSE(replay.arena->bindExternalBuffer(BufferId::QFN_NGRAM_ROW_IDS,ids.get()));
        EXPECT_FALSE(replay.arena->bindExternalBuffer(BufferId::QFN_PLE_EMBEDDING,output.get()));
        auto* reader=replay.compute.getNode("ple_gather")->stage.get();
        EXPECT_NO_THROW(reader->graphCaptureVariantSignature());
        replay.join();
        // Real allocator leases reject retirement while the retained executable
        // owns these original activation pointers. No fake pointer or runtime
        // generation is installed to manufacture a physically invalid state.
        EXPECT_THROW(replay.backend->free(output->gpu_data_ptr(),device.ordinal),std::logic_error);
        EXPECT_THROW(replay.backend->free(ids->gpu_data_ptr(),device.ordinal),std::logic_error);
        EXPECT_NO_THROW(reader->graphCaptureVariantSignature());
        auto* begin=replay.compute.getNode(replay.compute.getExecutionOrder().front())->stage.get();
        EXPECT_NO_THROW(begin->graphCaptureVariantSignature());
    });programs.retire();
}
/** @brief Reject actual foreign equal-sized owner leases before source roots publish. */
void productionSourceAuthorityRejection(DeviceId device){
    if(!hasCPUBackend())initCPUBackend(-1);std::vector<ProviderReplaySpec> specs;
    for(auto slots:{0u,4u})specs.push_back({std::make_shared<ProviderFile>(),64,slots,slots?2u:0u,n::OriginalIdWidth::UInt64,true,OwnerAdmission::RejectForeignAuthority});
    ProviderReplayPrograms programs(specs);programs.prepare(device);
    for(std::size_t i=0;i<specs.size();++i)programs.run(i,[](ProviderReplay&){});programs.retire();
}
}
/** @brief The actual source file belongs to the original process guardian, including abrupt-exit cleanup. */
TEST(PooledImmutableProviderSourceDevice,ProviderBackingUsesOwnedTemporaryRoot){
    const auto* declared=std::getenv("TMPDIR");ASSERT_NE(declared,nullptr);ASSERT_NE(*declared,'\0');
    const auto owner=std::filesystem::canonical(declared);ASSERT_NE(owner,std::filesystem::path("/tmp"));
    const std::array<std::uint64_t,2> rows{1,65537};std::string path;
    {ProviderFile file(1048576,rows);path=file.path;
        EXPECT_EQ(std::filesystem::canonical(path).parent_path(),owner);
        EXPECT_TRUE(std::filesystem::exists(path));EXPECT_EQ(file.descriptor().path,path);}
    EXPECT_FALSE(std::filesystem::exists(path));
}
#ifdef HAVE_CUDA
TEST(PooledImmutableProviderSourceDevice,CUDACapturedRealProviderAllFormatsReplay){providerAllFormatsReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDACapturedRealProviderFailureReplay){providerFailureReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDACapturedRealProviderResetReplay){providerReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionPLEInt64RepeatedTokenReplay){productionPLELookupReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionPLEAllFormatsReplay){providerAllFormatsReplay(DeviceId::cuda(0),true);}
TEST(PooledImmutableProviderSourceDevice,CUDAACKReadyCancellationReplay){pooledACKReadyCancellation(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionColdArenaSealReplay){productionColdArenaSealReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionSourceAuthorityRejection){productionSourceAuthorityRejection(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(PooledImmutableProviderSourceDevice,ROCmCapturedRealProviderAllFormatsReplay){providerAllFormatsReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmCapturedRealProviderFailureReplay){providerFailureReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmCapturedRealProviderResetReplay){providerReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionPLEInt64RepeatedTokenReplay){productionPLELookupReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionPLEAllFormatsReplay){providerAllFormatsReplay(DeviceId::rocm(0),true);}
TEST(PooledImmutableProviderSourceDevice,ROCmACKReadyCancellationReplay){pooledACKReadyCancellation(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionColdArenaSealReplay){productionColdArenaSealReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionSourceAuthorityRejection){productionSourceAuthorityRejection(DeviceId::rocm(0));}
#endif
#ifdef HAVE_CUDA
/** @test Explicit observed NoPeer mode reuses production capture, live-byte and reset oracles. */
TEST(PooledImmutableProviderSourceDevice,NoPeerCapturedCUDAConsumerCUDAStorageC0ReadAllFormats){crossVendorReadFormats(DeviceId::cuda(0),DeviceId::cuda(1),0);}
TEST(PooledImmutableProviderSourceDevice,NoPeerCapturedCUDAConsumerCUDAStoragePositiveReadAllFormats){crossVendorReadFormats(DeviceId::cuda(0),DeviceId::cuda(1),3);}
#endif
#ifdef HAVE_ROCM
/** @test Symmetric ROCm NoPeer mode rejects incompatible hardware rather than substituting transport. */
TEST(PooledImmutableProviderSourceDevice,NoPeerCapturedROCmConsumerROCmStorageC0ReadAllFormats){crossVendorReadFormats(DeviceId::rocm(0),DeviceId::rocm(1),0);}
TEST(PooledImmutableProviderSourceDevice,NoPeerCapturedROCmConsumerROCmStoragePositiveReadAllFormats){crossVendorReadFormats(DeviceId::rocm(0),DeviceId::rocm(1),3);}
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
/** @test Actual cold reset words, response admission, repeated captured reset and sticky failure across opposite-vendor source owners. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorCUDAConsumerROCmStorageColdPrefillFirstError){crossVendorColdPrefillLifecycle(DeviceId::cuda(0),DeviceId::rocm(0));}
/** @test Symmetric actual ROCm source/CUDA owner first-error lifecycle. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorROCmConsumerCUDAStorageColdPrefillFirstError){crossVendorColdPrefillLifecycle(DeviceId::rocm(0),DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CrossVendorCapturedCUDAConsumerROCmStorageC0ReadAllFormats){crossVendorReadFormats(DeviceId::cuda(0),DeviceId::rocm(0),0);}
TEST(PooledImmutableProviderSourceDevice,CrossVendorCapturedCUDAConsumerROCmStoragePositiveReadAllFormats){crossVendorReadFormats(DeviceId::cuda(0),DeviceId::rocm(0),3);}
TEST(PooledImmutableProviderSourceDevice,CrossVendorCapturedROCmConsumerCUDAStorageC0ReadAllFormats){crossVendorReadFormats(DeviceId::rocm(0),DeviceId::cuda(0),0);}
TEST(PooledImmutableProviderSourceDevice,CrossVendorCapturedROCmConsumerCUDAStoragePositiveReadAllFormats){crossVendorReadFormats(DeviceId::rocm(0),DeviceId::cuda(0),3);}
/** @test Actual CUDA Pin live identity threshold 63, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin63){exactIdentityNativeBoundary(DeviceId::cuda(0),63,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA Pin live identity threshold 64, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin64){exactIdentityNativeBoundary(DeviceId::cuda(0),64,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA Pin live identity threshold 65, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin65){exactIdentityNativeBoundary(DeviceId::cuda(0),65,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA RetainedPinsAndAllocation live identity threshold 63, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation63){exactIdentityNativeBoundary(DeviceId::cuda(0),63,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual CUDA RetainedPinsAndAllocation live identity threshold 64, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation64){exactIdentityNativeBoundary(DeviceId::cuda(0),64,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual CUDA RetainedPinsAndAllocation live identity threshold 65, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation65){exactIdentityNativeBoundary(DeviceId::cuda(0),65,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm Pin live identity threshold 63, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin63){exactIdentityNativeBoundary(DeviceId::rocm(0),63,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm Pin live identity threshold 64, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin64){exactIdentityNativeBoundary(DeviceId::rocm(0),64,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm Pin live identity threshold 65, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin65){exactIdentityNativeBoundary(DeviceId::rocm(0),65,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm RetainedPinsAndAllocation live identity threshold 63, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation63){exactIdentityNativeBoundary(DeviceId::rocm(0),63,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm RetainedPinsAndAllocation live identity threshold 64, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation64){exactIdentityNativeBoundary(DeviceId::rocm(0),64,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm RetainedPinsAndAllocation live identity threshold 65, retained capture/reset and exact owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation65){exactIdentityNativeBoundary(DeviceId::rocm(0),65,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}

/** @test Actual CUDA cooperative identity live127, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin127){exactIdentityNativeBoundary(DeviceId::cuda(0),127,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA cooperative identity live128, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin128){exactIdentityNativeBoundary(DeviceId::cuda(0),128,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA cooperative identity live129, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin129){exactIdentityNativeBoundary(DeviceId::cuda(0),129,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA cooperative identity live255, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin255){exactIdentityNativeBoundary(DeviceId::cuda(0),255,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA cooperative identity live256, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin256){exactIdentityNativeBoundary(DeviceId::cuda(0),256,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA cooperative identity live257, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin257){exactIdentityNativeBoundary(DeviceId::cuda(0),257,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA cooperative identity live513, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDAPin513){exactIdentityNativeBoundary(DeviceId::cuda(0),513,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual CUDA cooperative identity live127, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation127){exactIdentityNativeBoundary(DeviceId::cuda(0),127,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual CUDA cooperative identity live128, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation128){exactIdentityNativeBoundary(DeviceId::cuda(0),128,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual CUDA cooperative identity live129, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation129){exactIdentityNativeBoundary(DeviceId::cuda(0),129,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual CUDA cooperative identity live255, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation255){exactIdentityNativeBoundary(DeviceId::cuda(0),255,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual CUDA cooperative identity live256, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation256){exactIdentityNativeBoundary(DeviceId::cuda(0),256,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual CUDA cooperative identity live257, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation257){exactIdentityNativeBoundary(DeviceId::cuda(0),257,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual CUDA cooperative identity live513, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityCUDARetainedPinsAndAllocation513){exactIdentityNativeBoundary(DeviceId::cuda(0),513,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm cooperative identity live127, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin127){exactIdentityNativeBoundary(DeviceId::rocm(0),127,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm cooperative identity live128, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin128){exactIdentityNativeBoundary(DeviceId::rocm(0),128,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm cooperative identity live129, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin129){exactIdentityNativeBoundary(DeviceId::rocm(0),129,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm cooperative identity live255, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin255){exactIdentityNativeBoundary(DeviceId::rocm(0),255,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm cooperative identity live256, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin256){exactIdentityNativeBoundary(DeviceId::rocm(0),256,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm cooperative identity live257, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin257){exactIdentityNativeBoundary(DeviceId::rocm(0),257,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm cooperative identity live513, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmPin513){exactIdentityNativeBoundary(DeviceId::rocm(0),513,ExactIdentityBoundaryTrace::Pin);}
/** @test Actual ROCm cooperative identity live127, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation127){exactIdentityNativeBoundary(DeviceId::rocm(0),127,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm cooperative identity live128, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation128){exactIdentityNativeBoundary(DeviceId::rocm(0),128,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm cooperative identity live129, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation129){exactIdentityNativeBoundary(DeviceId::rocm(0),129,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm cooperative identity live255, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation255){exactIdentityNativeBoundary(DeviceId::rocm(0),255,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm cooperative identity live256, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation256){exactIdentityNativeBoundary(DeviceId::rocm(0),256,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm cooperative identity live257, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation257){exactIdentityNativeBoundary(DeviceId::rocm(0),257,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}
/** @test Actual ROCm cooperative identity live513, original capture/reset and complete owner evidence. */
TEST(PooledImmutableProviderSourceDevice,ExactIdentityROCmRetainedPinsAndAllocation513){exactIdentityNativeBoundary(DeviceId::rocm(0),513,ExactIdentityBoundaryTrace::RetainedPinsAndAllocation);}

/** @test Actual captured CUDA Pin malformed65 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAPin65Row32){exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,false);}
/** @test Actual captured CUDA Pin malformed65 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAPin65Row64){exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,true);}
/** @test Actual captured CUDA Pin malformed257 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAPin257Row32){exactIdentityNativeRejection(DeviceId::cuda(0),257,n::OwnerAction::Pin,false);}
/** @test Actual captured CUDA Pin malformed257 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAPin257Row64){exactIdentityNativeRejection(DeviceId::cuda(0),257,n::OwnerAction::Pin,true);}
/** @test Actual captured CUDA Free malformed65 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAFree65Row32){exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::AcceptFree,false);}
/** @test Actual captured CUDA Free malformed65 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAFree65Row64){exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::AcceptFree,true);}
/** @test Actual captured CUDA Free malformed257 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAFree257Row32){exactIdentityNativeRejection(DeviceId::cuda(0),257,n::OwnerAction::AcceptFree,false);}
/** @test Actual captured CUDA Free malformed257 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAFree257Row64){exactIdentityNativeRejection(DeviceId::cuda(0),257,n::OwnerAction::AcceptFree,true);}
/** @test Actual captured CUDA Victim malformed65 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAVictim65Row32){exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::AcceptVictim,false);}
/** @test Actual captured CUDA Victim malformed65 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAVictim65Row64){exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::AcceptVictim,true);}
/** @test Actual captured CUDA Victim malformed257 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAVictim257Row32){exactIdentityNativeRejection(DeviceId::cuda(0),257,n::OwnerAction::AcceptVictim,false);}
/** @test Actual captured CUDA Victim malformed257 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionCUDAVictim257Row64){exactIdentityNativeRejection(DeviceId::cuda(0),257,n::OwnerAction::AcceptVictim,true);}
/** @test Actual captured ROCm Pin malformed65 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmPin65Row32){exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,false);}
/** @test Actual captured ROCm Pin malformed65 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmPin65Row64){exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,true);}
/** @test Actual captured ROCm Pin malformed257 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmPin257Row32){exactIdentityNativeRejection(DeviceId::rocm(0),257,n::OwnerAction::Pin,false);}
/** @test Actual captured ROCm Pin malformed257 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmPin257Row64){exactIdentityNativeRejection(DeviceId::rocm(0),257,n::OwnerAction::Pin,true);}
/** @test Actual captured ROCm Free malformed65 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmFree65Row32){exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::AcceptFree,false);}
/** @test Actual captured ROCm Free malformed65 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmFree65Row64){exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::AcceptFree,true);}
/** @test Actual captured ROCm Free malformed257 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmFree257Row32){exactIdentityNativeRejection(DeviceId::rocm(0),257,n::OwnerAction::AcceptFree,false);}
/** @test Actual captured ROCm Free malformed257 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmFree257Row64){exactIdentityNativeRejection(DeviceId::rocm(0),257,n::OwnerAction::AcceptFree,true);}
/** @test Actual captured ROCm Victim malformed65 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmVictim65Row32){exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::AcceptVictim,false);}
/** @test Actual captured ROCm Victim malformed65 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmVictim65Row64){exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::AcceptVictim,true);}
/** @test Actual captured ROCm Victim malformed257 identities, physical Row32, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmVictim257Row32){exactIdentityNativeRejection(DeviceId::rocm(0),257,n::OwnerAction::AcceptVictim,false);}
/** @test Actual captured ROCm Victim malformed257 identities, physical Row64, original scalar oracle. */
TEST(PooledImmutableProviderSourceDevice,CooperativeRejectionROCmVictim257Row64){exactIdentityNativeRejection(DeviceId::rocm(0),257,n::OwnerAction::AcceptVictim,true);}

/** @test Complete actual cross-vendor 2-owner Positive trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorCUDAConsumer2ROCmOwnersPositiveReplay){crossVendorMultiOwnerTrajectory(DeviceId::cuda(0),{DeviceId::rocm(1),DeviceId::rocm(0)},{3,3});}
/** @test Complete actual cross-vendor 2-owner MixedC0 trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorCUDAConsumer2ROCmOwnersMixedC0Replay){crossVendorMultiOwnerTrajectory(DeviceId::cuda(0),{DeviceId::rocm(1),DeviceId::rocm(0)},{0,3});}
/** @test Complete actual cross-vendor 2-owner AllC0 trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorCUDAConsumer2ROCmOwnersAllC0Replay){crossVendorMultiOwnerTrajectory(DeviceId::cuda(0),{DeviceId::rocm(1),DeviceId::rocm(0)},{0,0});}
/** @test Complete actual cross-vendor 4-owner Positive trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorCUDAConsumer4ROCmOwnersPositiveReplay){crossVendorMultiOwnerTrajectory(DeviceId::cuda(0),{DeviceId::rocm(3),DeviceId::rocm(2),DeviceId::rocm(1),DeviceId::rocm(0)},{3,3,3,3});}
/** @test Complete actual cross-vendor 4-owner MixedC0 trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorCUDAConsumer4ROCmOwnersMixedC0Replay){crossVendorMultiOwnerTrajectory(DeviceId::cuda(0),{DeviceId::rocm(3),DeviceId::rocm(2),DeviceId::rocm(1),DeviceId::rocm(0)},{1,0,0,3});}
/** @test Complete actual cross-vendor 4-owner AllC0 trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorCUDAConsumer4ROCmOwnersAllC0Replay){crossVendorMultiOwnerTrajectory(DeviceId::cuda(0),{DeviceId::rocm(3),DeviceId::rocm(2),DeviceId::rocm(1),DeviceId::rocm(0)},{0,0,0,0});}
/** @test Complete actual cross-vendor 2-owner Positive trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorROCmConsumer2CUDAOwnersPositiveReplay){crossVendorMultiOwnerTrajectory(DeviceId::rocm(0),{DeviceId::cuda(1),DeviceId::cuda(0)},{3,3});}
/** @test Complete actual cross-vendor 2-owner MixedC0 trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorROCmConsumer2CUDAOwnersMixedC0Replay){crossVendorMultiOwnerTrajectory(DeviceId::rocm(0),{DeviceId::cuda(1),DeviceId::cuda(0)},{0,3});}
/** @test Complete actual cross-vendor 2-owner AllC0 trajectory; hardware absence fails explicitly. */
TEST(PooledImmutableProviderSourceDevice,CrossVendorROCmConsumer2CUDAOwnersAllC0Replay){crossVendorMultiOwnerTrajectory(DeviceId::rocm(0),{DeviceId::cuda(1),DeviceId::cuda(0)},{0,0});}

#endif

/** @test Immutable record rejection and ordered partial pin commit use the actual captured owner. */
#ifdef HAVE_CUDA
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinSlotBoundsEarlyRow32)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinSlotBounds,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinSlotBoundsEarlyRow64)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinSlotBounds,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinSlotBoundsLateRow32)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinSlotBounds,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinSlotBoundsLateRow64)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinSlotBounds,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinOrdinalEarlyRow32)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinOrdinal,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinOrdinalEarlyRow64)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinOrdinal,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinOrdinalLateRow32)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinOrdinal,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinOrdinalLateRow64)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinOrdinal,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinCommitFlagsEarlyRow32)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinCommitFlags,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinCommitFlagsEarlyRow64)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinCommitFlags,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinCommitFlagsLateRow32)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinCommitFlags,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsCUDAPinCommitFlagsLateRow64)
{
    exactIdentityNativeRejection(DeviceId::cuda(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinCommitFlags,false);
}
#endif
#ifdef HAVE_ROCM
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinSlotBoundsEarlyRow32)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinSlotBounds,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinSlotBoundsEarlyRow64)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinSlotBounds,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinSlotBoundsLateRow32)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinSlotBounds,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinSlotBoundsLateRow64)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinSlotBounds,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinOrdinalEarlyRow32)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinOrdinal,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinOrdinalEarlyRow64)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinOrdinal,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinOrdinalLateRow32)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinOrdinal,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinOrdinalLateRow64)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinOrdinal,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinCommitFlagsEarlyRow32)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinCommitFlags,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinCommitFlagsEarlyRow64)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinCommitFlags,true);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinCommitFlagsLateRow32)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,false,identity_fixture::IdentityFault::PinCommitFlags,false);
}
/** @test Actual captured immutable-field rejection or original partial-prefix commit. */
TEST(PooledImmutableProviderSourceDevice,CooperativeFieldsROCmPinCommitFlagsLateRow64)
{
    exactIdentityNativeRejection(DeviceId::rocm(0),65,n::OwnerAction::Pin,true,identity_fixture::IdentityFault::PinCommitFlags,false);
}
#endif

}
