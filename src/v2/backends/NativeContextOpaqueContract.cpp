/**
 * @file NativeContextOpaqueContract.cpp
 * @brief Supported-backend context-birth family allowances with original physical lifetime.
 *
 * Opaque units cover the supported native worker family while current explicit
 * storage stays exact. Calibration emits separate observations for review; it
 * cannot append a certificate, adjust a late free sample, or authorize native
 * creation automatically. Supported backend/resource families use roomy fixed allowances. Runtime ABI
 * admission remains with the configured loader; exact object storage remains
 * independently charged through its original authority.
 */
#include "NativeContextOpaqueContract.h"
#include "NativeContextQualifiedLimits.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
namespace {
/** @return Whether original runtime/provider evidence is complete; no historical digest equality is required. */
bool digest(const std::string& value){return value.size()==64 &&
    std::all_of(value.begin(),value.end(),[](unsigned char c){return (c>='0'&&c<='9') || (c>='a'&&c<='f');}) &&
    value.find_first_not_of('0')!=std::string::npos;}
}
void CUDANativeProgramStack::requirePrepared(std::size_t required,std::size_t actual) {
    if(required>capacityBytes || actual<capacityBytes)
        throw std::runtime_error("CUDA original native-program stack capacity mismatch: required="+
            std::to_string(required)+" prepared="+std::to_string(capacityBytes)+
            " actual="+std::to_string(actual));
}
void NativeContextBirthKey::requireComplete() const {
    if(!device.is_gpu() || source_program_version!=1 || physical_uuid.empty() || architecture.empty() ||
       runtime_version.empty() || driver_version.empty() || visibility_identity.empty() || pthread_defaults_identity.empty() ||
       !digest(backend_sha256) || !digest(runtime_sha256) || !digest(driver_sha256) ||
       !digest(runtime_policy_sha256) || (device.is_rocm() && !digest(hsa_sha256)) ||
       !compute_units || !native_wave_width || !visible_device_count || !worker_stack_bytes || !context_object_bytes || !context_object_alignment ||
       (context_object_alignment&(context_object_alignment-1)) || primary_streams!=1 || primary_blas_handles!=1 || secondary_blas_handles!=1 || dependency_events!=32 ||
       (population!=NativeContextBirthPopulation::InitialBackendContext && population!=NativeContextBirthPopulation::AdditionalContext))
        throw std::invalid_argument("Original GPU context birth needs its complete physical/source/runtime/library/pthread/creator closure");
}
NativeContextFamilyBudget NativeContextOpaqueCatalog::planningBudget(DeviceType backend,NativeContextBirthPopulation population){
    for(const auto& row:detail::kNativeContextQualifiedClasses){
        if(row.backend!=backend || row.population!=population)continue;
        if((row.physical_lifetime!=NativePhysicalLifetime::ContextGeneration && row.physical_lifetime!=NativePhysicalLifetime::ProcessRuntime) || !row.host_upper_bytes || !row.device_upper_bytes || !digest(std::string(row.evidence_sha256)))
            throw std::logic_error("Installed native context class has incomplete qualified limits/evidence");
        return {row.host_upper_bytes,row.device_upper_bytes};
    }
    throw std::runtime_error("Native planning context family is unqualified");
}
NativeContextOpaqueContract NativeContextOpaqueCatalog::require(const NativeContextBirthKey& key){
    key.requireComplete();
    const auto budget=planningBudget(key.device.type,key.population);
    for(const auto& row:detail::kNativeContextQualifiedClasses){
        if(row.backend!=key.device.type || row.population!=key.population)continue;
        if((row.physical_lifetime!=NativePhysicalLifetime::ContextGeneration && row.physical_lifetime!=NativePhysicalLifetime::ProcessRuntime) || !row.host_upper_bytes || !row.device_upper_bytes || !digest(std::string(row.evidence_sha256)))
            throw std::logic_error("Installed native context class has incomplete qualified limits/evidence");
        // The live original physical identity is bound here, not matched to a
        // host UUID list. Bounds remain the tested class's immutable data.
        return {key,budget.hostUpperBytes(),budget.deviceUpperBytes(),std::string(row.evidence_sha256),row.physical_lifetime};
    }
    throw std::runtime_error("Original GPU context birth family is unqualified: no installed opaque upper-limit certificate for "+key.device.toString());
}
}
