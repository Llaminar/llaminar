/** @file NativeContextBirthDiagnosticHooks.cpp
 * @brief Explicit raw diagnostic interception of original worker creator boundaries.
 * Vendor-internal creators remain untouched. Actual constructor rollback owns all
 * handles; this file stores immutable observations and never destroys resources.
 */
#include "NativeContextBirthDiagnosticHooks.h"
#include <atomic>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string_view>
namespace llaminar2::test::birth_diagnostic { namespace {
std::atomic<bool> active{false};
std::size_t failure=0;
std::vector<Observation> observations;
std::function<void(const Observation&,bool)> checkpoint;
std::mutex stateMutex;

}
void begin(std::size_t failCreator,std::function<void(const Observation&,bool)> callback){
    std::lock_guard lock(stateMutex);
    if(active || !callback || failCreator>35)throw std::invalid_argument("Invalid raw diagnostic creator interval");
    observations.clear();observations.reserve(35);failure=failCreator;checkpoint=std::move(callback);active=true;
}
std::vector<Observation> snapshot(){std::lock_guard lock(stateMutex);return observations;}
std::vector<Observation> finish(){std::lock_guard lock(stateMutex);active=false;checkpoint={};return observations;}
bool isProjectCaller(void* address){
    if(!active.load())return false;
    Dl_info location{};if(!::dladdr(address,&location) || !location.dli_fname)return false;
    const std::string_view path(location.dli_fname);const auto slash=path.find_last_of('/');const auto name=path.substr(slash==path.npos?0:slash+1);
    return name=="v2_perf_native_context_birth_calibration" || name.starts_with("libllaminar") || name.starts_with("libcuda_backend") || name.starts_with("librocm_backend");
}
bool before(Creator creator,void* address){
    if(!isProjectCaller(address))return true;
    std::lock_guard lock(stateMutex);Dl_info location{};::dladdr(address,&location);
    const auto index=observations.size()+1;
    observations.push_back({creator,index,location.dli_fname,index==failure,false,nullptr});
    checkpoint(observations.back(),true);return index!=failure;
}
void after(Creator creator,bool success,void* handle,void* address,int status){
    if(!isProjectCaller(address))return;
    std::lock_guard lock(stateMutex);
    if(observations.empty() || observations.back().creator!=creator)throw std::logic_error("Diagnostic creator observation order changed");
    auto& value=observations.back();value.success=success;value.handle=handle;value.native_status=status;checkpoint(value,false);
}
void released(Creator creator,void* handle,bool success,void* address,int status){
    if(!isProjectCaller(address))return;
    std::lock_guard lock(stateMutex);
    for(auto& value:observations)if(value.creator==creator && value.handle==handle && value.success){
        if(value.destroyed)throw std::logic_error("Original constructor family destroyed an owner twice");
        value.destroyed=true;value.destroy_success=success;value.destroy_status=status;checkpoint(value,false);return;
    }
    throw std::logic_error("Original destructor lost its observed creator owner");
}
} // namespace llaminar2::test::birth_diagnostic
