/**
 * @file NativeContextOpaqueContract.h
 * @brief Qualified upper commitments for one exact original GPU context-birth family.
 *
 * These bounds cover genuinely opaque driver/library pools only. Source-owned
 * worker stack/object storage is a separate exact composite allocation declaration. No caller can
 * supply byte limits or turn a diagnostic peak into an installed certificate.
 */
#pragma once
#include "DeviceId.h"
#include "NativePhysicalLifetime.h"
#include <cstddef>
#include <cstdint>
#include <string>
namespace llaminar2 {
namespace test { struct NativeContextPreparationReplayAccess; }
/**
 * @brief Fixed CUDA native-program stack policy within the existing worker opaque commitment.
 *
 * The 2 KiB candidate covers the selected native library program before source
 * sealing. It is not a physical-byte estimate or a per-graph allowance. Worker
 * birth provisions it; graph inspection may only verify, never grow, this cap.
 * Native qualification must exercise the actual configured library kernels.
 */
struct CUDANativeProgramStack final {
    static constexpr std::size_t capacityBytes = 2048u;
    /** @brief Reject an unmet source capacity or an oversized captured program.
     * @param required Actual captured-kernel local bytes per thread.
     * @param actual Actual driver stack capacity after original source birth.
     */
    static void requirePrepared(std::size_t required, std::size_t actual);
};
/** @brief The actual constructor population, not a selectable transport or inference mode. */
enum class NativeContextBirthPopulation { InitialBackendContext, AdditionalContext };
/** @brief Complete immutable physical/source/geometry closure of the measured constructor. */
struct NativeContextBirthKey final {
    DeviceId device=DeviceId::invalid();
    NativeContextBirthPopulation population=NativeContextBirthPopulation::InitialBackendContext;
    std::uint64_t source_program_version=1;
    std::size_t visible_device_count=0,compute_units=0,native_wave_width=0;
    std::string physical_uuid,architecture,runtime_version,driver_version;
    std::string source_sha256; ///< Optional historical evidence only; never an admission or live-identity input.
    std::string backend_sha256,runtime_sha256,driver_sha256;
    std::string runtime_policy_sha256,hsa_sha256;
    std::string visibility_identity,pthread_defaults_identity;
    std::size_t worker_stack_bytes=0,worker_guard_bytes=0,context_object_bytes=0,context_object_alignment=0;
    std::uint32_t primary_streams=0,primary_blas_handles=0,secondary_blas_handles=0,dependency_events=0;
    bool operator==(const NativeContextBirthKey&) const=default;
    /** @brief Reject an incomplete recorded family before any native creator can run. */
    void requireComplete() const;
};
/** @brief Source-installed certificate; its numeric bounds cannot be provided by a production caller. */
class NativeContextOpaqueContract final {
public:
    NativePhysicalLifetime lifetime() const noexcept { return lifetime_; }
    const NativeContextBirthKey& key() const noexcept { return key_; }
    std::size_t hostUpperBytes() const noexcept { return host_; }
    std::size_t deviceUpperBytes() const noexcept { return device_; }
    const std::string& evidenceSha256() const noexcept { return evidence_; }
private:
    friend class NativeContextOpaqueCatalog;
    friend struct test::NativeContextPreparationReplayAccess;
    NativeContextOpaqueContract(NativeContextBirthKey key,std::size_t host,std::size_t device,std::string evidence,NativePhysicalLifetime lifetime=NativePhysicalLifetime::ContextGeneration)
        :key_(std::move(key)),host_(host),device_(device),evidence_(std::move(evidence)),lifetime_(lifetime){}
    const NativeContextBirthKey key_;
    const std::size_t host_,device_;
    const std::string evidence_;
    const NativePhysicalLifetime lifetime_;
};
/** @brief Installed opaque family limits usable for planning, never a birth authorization. */
class NativeContextFamilyBudget final {
public:
    std::size_t hostUpperBytes()const noexcept{return host_;}
    std::size_t deviceUpperBytes()const noexcept{return device_;}
private:
    friend class NativeContextOpaqueCatalog;
    NativeContextFamilyBudget(std::size_t host,std::size_t device):host_(host),device_(device){}
    std::size_t host_,device_;
};
/** @brief Only reviewed source/driver/runtime/architecture-bound certificates authorize birth. */
class NativeContextOpaqueCatalog final {
public:
    /** @brief Return the exact installed certificate or fail before context/runtime discovery. */
    static NativeContextOpaqueContract require(const NativeContextBirthKey&);
    /** @brief Price the installed class without querying providers or granting native birth. */
    static NativeContextFamilyBudget planningBudget(DeviceType,NativeContextBirthPopulation);
};
}
