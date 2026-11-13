/**
 * @file PooledRemoteGraphCalibration.h
 * @brief Target-only raw remote-definition checkpoints on its original authored program.
 *
 * Checkpoints create no native handles and never query an SDK, including after
 * reset. Observer metadata/heap/TLS costs remain in measured results; this is
 * neither a certificate nor a timing result or whole-process byte proof.
 */
#pragma once
#ifndef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
#error "Remote calibration requires the explicit raw native diagnostic target"
#endif
#include "fixtures/backends/PooledRemoteGraphHostReplayAccess.h"
#include "integration/backends/NativeContextSourceProbeSupport.h"
#include "integration/backends/PooledNativeProgramObservation.h"
#include "execution/qwen4exp/PooledImmutableDirectedOwnerProgram.h"
#include "execution/qwen4exp/PooledImmutableDirectedLookupGraph.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <mutex>
#include <unistd.h>
namespace llaminar2::test {
/** @brief Freeze original metadata without keeping a native worker alive through reset. */
class PooledRemoteGraphCalibration final {
public:
    /** @brief Bind immutable original source facts before the first graph creator. */
    PooledRemoteGraphCalibration(const PreparedGPUContextUsePermit& original,const PooledRemoteGraphProgram& program,std::size_t instance,bool handshake):handshake_(handshake){
        const auto key=original.provenance().originalBirthKey();const auto& g=program.geometry();
        identity_={{"instance",instance},{"definition_role","authored_remote_owner"},{"original_pma",reinterpret_cast<std::uintptr_t>(original.memory().get())},
            {"original_worker",reinterpret_cast<std::uintptr_t>(original.worker().get())},{"original_generation",original.generation()},
            {"original_stream",reinterpret_cast<std::uintptr_t>(original.worker()->defaultStream())},{"birth_key",nativeSourceBirthKeyJson(*key)},
            {"resource_source",NativeAuxiliaryResourceCatalog::resourceSourceIdentity()},{"portable_program",program.qualificationClass().identity()},{"capture_identity",program.captureIdentity()},
            {"geometry",{{"coordinator",g.coordinator.device.toString()},{"owner",g.owner.device.toString()},{"coordinate",g.coordinate},{"participants",g.participants},{"capacity",g.capacity},
                {"maximum_original_ids",g.maximumOriginalIds},{"maximum_unique_ids",g.maximumUniqueIds},{"source_rows",g.sourceRows},{"row_bytes",g.rowBytes},{"slot_stride",g.slotStride},
                {"native_layout_identity",g.nativeLayoutIdentity},{"row_key_bits",g.rowKeyBits},{"nonempty_root_mask",g.nonemptyRootMask}}},
            {"expected_native_structure",{{"nodes",program.limits().maximumNodes},{"edges",program.limits().maximumEdges},{"kind","kernel"},{"linear_chain",true}}}};
    }
    /** @return Known original CPU0 roots; unknown opaque host backing grants no allowance. */
    static PhysicalMemoryPlan knownBOM(const PooledRemoteGraphProgram& program,PhysicalMemoryResource cpu,PhysicalMemoryResource gpu){
        return PooledRemoteGraphHostReplayAccess::bom(program,cpu,gpu,std::nullopt,nativeGraphCaptureObjectBytes(gpu.device));
    }
    /** @return Same descriptor-aware original creator as the installed/recorded body. */
    static auto create(const PreparedGPUContextUsePermit& original,const PooledRemoteGraphProgram& program){return PooledRemoteGraphHostReplayAccess::raw(original,program);}
    /** @brief Publish a cold/final host-only sample and optional external-observer handshake.
     * @param stage Exact documented phase. @param extra CPU-owned input/oracle observations.
     */
    void checkpoint(const char* stage,nlohmann::json extra=nlohmann::json::object()){
        std::lock_guard lock(mutex_);const auto h=pooled_program_observation::sample();
        std::cout<<nlohmann::json{{"schema","llaminar.pooled_remote_graph_calibration.v1"},{"pid",::getpid()},{"sequence",sequence_++},{"stage",stage},{"qualified",false},{"certified_limit",nullptr},
            {"instrumented",true},{"performance_qualified",false},{"whole_process_byte_completeness",false},{"identity",identity_},
            {"host",{{"rss",h.rss},{"peak",h.peak},{"maxrss_kib",h.peak/1024},{"allocator_live",h.allocator_live},{"allocator_arena",h.allocator_arena},{"allocator_mapped",h.allocator_mapped},{"monotonic_ns",h.monotonic_ns}}},{"extra",std::move(extra)}}.dump()<<'\n'<<std::flush;
        if(handshake_){std::string ack;if(!std::getline(std::cin,ack) || ack!="continue")throw std::runtime_error("Remote graph observation handshake failed");}
    }
private:nlohmann::json identity_;bool handshake_;std::size_t sequence_=0;std::mutex mutex_;
};
}
