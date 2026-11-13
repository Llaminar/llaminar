/**
 * @file NativeContextSourceProbeSupport.h
 * @brief Explicit integration-only facts from exact native SDK bindings and retained original events.
 *
 * This observer grants no constructor, registration or certificate capability.
 * The event bridge accepts only an original prepared permit, authenticates its
 * strong owner/generation, and runs on that worker before exposing existing32handles.
 */
#pragma once
#include "backends/NativeContextPreparation.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "utils/Sha256.h"
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
namespace llaminar2::test {
using SourceProbeJson=nlohmann::ordered_json;
/** @brief Read-only original symbol evidence, checked against CMake's exact SDK provider. */
struct NativeSourceSDKBinding final {std::string symbol,path,sha256,expected_path;};
/** @brief Observe actual resolved provider; never search or replace a production symbol. */
inline NativeSourceSDKBinding nativeSourceSDKBinding(const char* symbol,const void* address,const char* expected){
    Dl_info info{};if(!address || !dladdr(address,&info) || !info.dli_fname)throw std::runtime_error("Original SDK symbol has no loaded provider");
    const auto path=std::filesystem::canonical(info.dli_fname),configured=std::filesystem::canonical(expected);
    if(path==std::filesystem::canonical("/proc/self/exe") || !std::filesystem::equivalent(path,configured))
        throw std::runtime_error("Native source probe creator is interposed or differs from exact configured SDK provider");
    const auto hash=sha256FileHex(path);if(!hash)throw std::runtime_error("Original SDK provider could not be hashed");
    return {symbol,path.string(),*hash,configured.string()};
}
/** @brief Obtain actual CUDA BLAS/Lt provider facts without invoking a creator. */
std::vector<NativeSourceSDKBinding> nativeSourceCUDABindings();
/** @brief Obtain actual HIP BLAS/Lt provider facts without invoking a creator. */
std::vector<NativeSourceSDKBinding> nativeSourceROCmBindings();
/** @brief Named read-only observer of original context32ordering handles; no creation or substitution. */
struct NativeContextSourceLifecycleDiagnosticAccess {
    /** @brief Verify exact retained source worker, shared owner and generation on that worker. */
    static void requireOriginalWorker(const PreparedGPUContextUsePermit& permit){
        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(permit.device(),permit.memory());
        original.requireWorker(permit.worker());permit.provenance().requireWorker(original.worker());
        if(original.generation()!=permit.generation() || !permit.worker()->ownsCurrentThread())
            throw std::logic_error("Original event observation requires its retained prepared worker/generation on owner thread");
    }
    /** @brief Read actual CUDA owner handles under a proven original permit. */
    static std::vector<void*> cudaEvents(const PreparedGPUContextUsePermit&);
    /** @brief Read actual HIP owner handles under a proven original permit. */
    static std::vector<void*> rocmEvents(const PreparedGPUContextUsePermit&);
};
/** @brief Serialize every original production key field, never a certificate. */
inline SourceProbeJson nativeSourceBirthKeyJson(const NativeContextBirthKey& key){
    key.requireComplete();return {{"schema","llaminar.native_context_birth_key.v1"},{"device",key.device.toString()},
        {"population",key.population==NativeContextBirthPopulation::AdditionalContext?"additional_context":"initial_backend_context"},
        {"source_program_version",key.source_program_version},{"visible_device_count",key.visible_device_count},{"compute_units",key.compute_units},{"native_wave_width",key.native_wave_width},
        {"physical_uuid",key.physical_uuid},{"architecture",key.architecture},{"runtime_version",key.runtime_version},{"driver_version",key.driver_version},
        {"source_sha256",key.source_sha256},{"backend_sha256",key.backend_sha256},{"runtime_sha256",key.runtime_sha256},{"driver_sha256",key.driver_sha256},
        {"runtime_policy_sha256",key.runtime_policy_sha256},{"hsa_sha256",key.hsa_sha256},{"visibility_identity",key.visibility_identity},{"pthread_defaults_identity",key.pthread_defaults_identity},
        {"worker_stack_bytes",key.worker_stack_bytes},{"worker_guard_bytes",key.worker_guard_bytes},{"context_object_bytes",key.context_object_bytes},{"context_object_alignment",key.context_object_alignment},
        {"primary_streams",key.primary_streams},{"primary_blas_handles",key.primary_blas_handles},{"secondary_blas_handles",key.secondary_blas_handles},{"dependency_events",key.dependency_events}};
}
}
