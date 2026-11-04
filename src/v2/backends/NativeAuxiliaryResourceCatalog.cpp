/** @file NativeAuxiliaryResourceCatalog.cpp
 * @brief Select supported-backend resource-family allowances and retain original live ownership metadata.
 */
#include "NativeAuxiliaryResourceCatalog.h"
#include "NativeAuxiliaryQualifiedLimits.h"
#include "GPUDeviceContextPool.h"
#include "PooledRemoteGraphHostAllowance.h"
#include "utils/Sha256.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
namespace {
/** @return Whether a diagnostic evidence/provider digest is complete; it is not an allowance class. */
bool digest(std::string_view s){return s.size()==64 && s.find_first_not_of('0')!=s.npos && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
/** @brief Length-delimited immutable metadata encoding. */
void append(std::string& out,std::string_view s){out+=std::to_string(s.size())+":";out+=s;}
/** @return SHA256 of metadata only; no native-library file is reopened. */
std::string hash(const std::string& value){const auto out=sha256BytesHex(value);if(!out)throw std::runtime_error("Native auxiliary source identity hashing failed");return *out;}
}
std::string NativeAuxiliaryRemoteFamilyScope::identity()const{
    return "remote-family:"+std::to_string(version)+":"+std::to_string(native.maximumNodes)+":"+
        std::to_string(native.maximumEdges)+":"+std::to_string(arguments.count)+":"+std::to_string(arguments.bytes)+":"+
        std::to_string(arguments.alignment);
}
std::string NativeAuxiliaryResourceProgram::identity()const{
    if(isReadAheadSide())return "side-stream1-ordering2-v1";
    if(isPooledRemoteOwner())return remoteProgramClass().identity();
    return "ordering-v1:"+std::to_string(static_cast<unsigned>(orderingProgram().kind()))+":"+std::to_string(orderingProgram().nonemptyRootMask());
}
const std::string& NativeAuxiliaryResourceCatalog::resourceSourceIdentity(){static const std::string value="native-auxiliary-operations";return value;}
std::string NativeAuxiliaryResourceCatalog::portableSourceIdentity(const NativeContextBirthKey& k){
    k.requireComplete();std::string out="native-auxiliary-source-class-v1:";
    for(const auto& s:{std::to_string(static_cast<unsigned>(k.device.type)),std::to_string(k.source_program_version),std::to_string(static_cast<unsigned>(k.population)),k.architecture,
        std::to_string(k.compute_units),std::to_string(k.native_wave_width),k.runtime_version,k.driver_version,k.backend_sha256,k.runtime_sha256,k.driver_sha256,k.runtime_policy_sha256,k.hsa_sha256,k.pthread_defaults_identity,
        std::to_string(k.worker_stack_bytes),std::to_string(k.worker_guard_bytes),std::to_string(k.context_object_bytes),std::to_string(k.context_object_alignment),
        std::to_string(k.primary_streams),std::to_string(k.primary_blas_handles),std::to_string(k.secondary_blas_handles),std::to_string(k.dependency_events)})append(out,s);
    return hash(out);
}
std::string NativeAuxiliaryResourceCatalog::liveSourceIdentity(const NativeContextBirthKey& k){
    std::string out=portableSourceIdentity(k);for(const auto& s:{k.device.toString(),k.physical_uuid,k.visibility_identity,std::to_string(k.visible_device_count)})append(out,s);return hash(out);
}
NativeAuxiliaryResourceAllowance NativeAuxiliaryResourceCatalog::selectAllowance(DeviceType backend,
    NativeAuxiliaryResourceProgram program,std::span<const detail::NativeAuxiliaryQualifiedClass> rows){
    if(backend!=DeviceType::CUDA && backend!=DeviceType::ROCm)
        throw std::invalid_argument("Native auxiliary allowance requires a supported GPU backend");
    for(const auto& row:rows){if(row.backend!=backend)continue;
        if(row.remoteFamily && (!row.remoteFamily->valid() || !row.program.isPooledRemoteOwner() ||
            !row.remoteFamily->accepts(row.program.remoteProgramClass().limits(),PooledRemoteGraphProgramClass::argumentLayout())))
            throw std::logic_error("Installed remote graph family has invalid reviewed source scope");
        const bool remote=row.remoteFamily && program.isPooledRemoteOwner() &&
            row.remoteFamily->accepts(program.remoteProgramClass().limits(),PooledRemoteGraphProgramClass::argumentLayout());
        if(row.remoteFamily ? !remote : row.program!=program)continue;
        if(program.isPooledRemoteOwner() && !row.remoteFamily)
            throw std::logic_error("Remote graph host qualification requires reviewed family ceilings");
        if(!row.hostBytes || (program.isReadAheadSide() && !row.deviceBytes) || !digest(row.evidence) || (row.lifetime!=NativePhysicalLifetime::ContextGeneration && row.lifetime!=NativePhysicalLifetime::ProcessRuntime))throw std::logic_error("Installed auxiliary program has incomplete reviewed evidence/lifetime");
        if(program.isPooledRemoteOwner() && (row.deviceBytes || row.lifetime!=NativePhysicalLifetime::ProcessRuntime))
            throw std::logic_error("Remote graph host class must keep original device bytes separate and retain host backing through ProcessRuntime");
        const auto hostBytes=program.isPooledRemoteOwner()?PooledRemoteGraphHostAllowance::scaledBytes(program.remoteProgramClass().limits().maximumNodes,row.hostBytes):row.hostBytes;
        return {hostBytes,row.deviceBytes,std::string(row.evidence),row.lifetime,program,row.remoteFamily};}
    throw std::runtime_error("Native auxiliary program is unqualified: no reviewed source/runtime/program class for "+std::to_string(static_cast<unsigned>(backend))+" "+program.identity());
}
NativeAuxiliaryResourceAllowance NativeAuxiliaryResourceCatalog::allowance(DeviceType backend,NativeAuxiliaryResourceProgram program){
    return selectAllowance(backend,program,detail::kNativeAuxiliaryQualifiedClasses);
}
NativeAuxiliaryResourceSelection NativeAuxiliaryResourceCatalog::select(const NativeContextBirthKey& k,std::uint64_t generation,
    NativeAuxiliaryResourceProgram program,std::string_view resource,std::span<const detail::NativeAuxiliaryQualifiedClass> rows){
    k.requireComplete();if(!generation || resource.empty())throw std::invalid_argument("Native auxiliary class requires original generation/source");
    const auto bound=selectAllowance(k.device.type,program,rows);
    return {bound.hostBytes,bound.deviceBytes,bound.evidence,liveSourceIdentity(k),std::string(resource),bound.lifetime,generation,program,bound.remoteFamily};
}
NativeAuxiliaryResourceSelection NativeAuxiliaryResourceCatalog::require(DeviceId device,NativeAuxiliaryResourceProgram program){
    if(!device.is_gpu())throw std::invalid_argument("Native auxiliary program requires an actual GPU");
    if(detail::kNativeAuxiliaryQualifiedClasses.empty())throw std::runtime_error("Native auxiliary program is unqualified: reviewed data catalog is empty for "+device.toString()+" "+program.identity());
    const auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(device);
    const auto key=original.originalBirthKey(); // Strong alias of actual original contract; no cache/query/hash.
    return select(*key,original.generation(),program,resourceSourceIdentity(),detail::kNativeAuxiliaryQualifiedClasses);
}
}
