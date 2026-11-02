/**
 * @file NativeContextRuntimePolicy.h
 * @brief Immutable original GPU runtime environment policy at metadata startup.
 *
 * Runtime initialization consumes host configuration once. Preserve that exact
 * policy before the first discovery SDK call and reject later changes, rather
 * than certifying a new getenv view against already initialized native pools.
 */
#pragma once
#include "DeviceId.h"
#include "utils/Sha256.h"
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <stdexcept>
#include <unistd.h>
namespace llaminar2 {
/** @brief Original startup-policy identity; contains no native state or allocation ledger. */
class NativeContextRuntimePolicy final {
public:
    /** @brief Require unchanged original live policy and return the portable resource-class digest. */
    [[nodiscard]] const std::string& identity() const { requireUnchanged(); return resource_digest_; }
    /** @brief Return complete original host configuration identity for this live source only. */
    [[nodiscard]] const std::string& liveIdentity() const { requireUnchanged(); return live_digest_; }
    /** @brief Obtain one backend's original policy before its discovery SDK calls. */
    [[nodiscard]] static const NativeContextRuntimePolicy& original(DeviceType backend) {
        if (backend == DeviceType::CUDA) { static const NativeContextRuntimePolicy value(backend); return value; }
        if (backend == DeviceType::ROCm) { static const NativeContextRuntimePolicy value(backend); return value; }
        throw std::invalid_argument("Native context startup policy requires a GPU backend");
    }
    /**
     * @brief Hash resource-relevant policy from a host environment record.
     * @param backend Actual vendor namespace being authenticated.
     * @param entries Complete recorded NAME=VALUE host configuration entries.
     * @return Portable policy digest; physical selectors remain solely in live identity.
     * @note This pure projection grants no certificate or native birth authority.
     */
    [[nodiscard]] static std::string resourceIdentity(DeviceType backend,std::span<const std::string> entries) {
        if(backend!=DeviceType::CUDA && backend!=DeviceType::ROCm)throw std::invalid_argument("Resource policy requires a GPU backend");
        return digest(serialize(backend,entries,IdentityKind::ResourceClass));
    }
private:
    /** @brief Physical selectors authenticate a live process but cannot restrict portable classes to host UUIDs. */
    enum class IdentityKind { FullLive,ResourceClass };
    /** @brief Copy original host configuration at the cold startup authority boundary. */
    static std::vector<std::string> environment(){
        std::vector<std::string> result;for(char** entry=::environ;entry && *entry;++entry)result.emplace_back(*entry);return result;
    }
    /** @brief Serialize complete vendor namespaces with explicit lengths and absent/empty distinction. */
    static std::string serialize(DeviceType backend,std::span<const std::string> entries,IdentityKind kind) {
        std::vector<std::string> selected;
        for(const auto& entry:entries){const std::string_view value(entry);const auto name=value.substr(0,value.find('='));
            const bool common=name=="LD_LIBRARY_PATH" || name=="LD_PRELOAD";
            const bool vendor=backend==DeviceType::CUDA
                ? (name.starts_with("CUDA_") || name.starts_with("CUBLAS_") || name.starts_with("CUBLASLT_") || name.starts_with("NVIDIA_"))
                : (name.starts_with("HIP_") || name.starts_with("HIPBLAS_") || name.starts_with("HIPBLASLT_") || name.starts_with("ROCBLAS_") || name.starts_with("ROC_") || name.starts_with("HSA_") || name.starts_with("ROCR_") || name.starts_with("AMD_") || name.starts_with("GPU_"));
            const bool physicalSelector=name=="CUDA_VISIBLE_DEVICES" || name=="CUDA_DEVICE_ORDER" || name=="NVIDIA_VISIBLE_DEVICES" ||
                name=="HIP_VISIBLE_DEVICES" || name=="ROCR_VISIBLE_DEVICES" || name=="GPU_DEVICE_ORDINAL";
            if((common || vendor) && !(kind==IdentityKind::ResourceClass && physicalSelector))selected.emplace_back(value);
        }
        std::sort(selected.begin(),selected.end());std::string result="native-runtime-policy-v2:";
        for(const auto& value:selected)result+=std::to_string(value.size())+":"+value;return result;
    }
    /** @brief Hash canonical policy bytes without admitting or mutating native resources. */
    static std::string digest(const std::string& policy){const auto value=sha256BytesHex(policy);if(!value)throw std::runtime_error("Cannot hash original GPU runtime environment policy");return *value;}
    /** @brief Reject any full live policy change even when its portable resource class is unchanged. */
    void requireUnchanged()const{
        if(serialize(backend_,environment(),IdentityKind::FullLive)!=policy_)throw std::logic_error("GPU runtime environment changed after original metadata startup");
    }
    /** @brief Capture once before SDK startup; portable and live identities remain distinct immutable values. */
    explicit NativeContextRuntimePolicy(DeviceType backend):NativeContextRuntimePolicy(backend,environment()){}
    /** @brief Build both immutable identities from one original environment observation. */
    NativeContextRuntimePolicy(DeviceType backend,const std::vector<std::string>& entries)
        :backend_(backend),policy_(serialize(backend,entries,IdentityKind::FullLive)),live_digest_(digest(policy_)),resource_digest_(resourceIdentity(backend,entries)){}
    const DeviceType backend_;const std::string policy_,live_digest_,resource_digest_;
};
}
