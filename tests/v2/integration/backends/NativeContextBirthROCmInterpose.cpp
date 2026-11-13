/** @file NativeContextBirthROCmInterpose.cpp
 * @brief Diagnostic-only HIP creator prefixes through original resolved SDK symbols.
 * Actual HIP/BLAS constructors and the original context rollback remain owners.
 */
#include "NativeContextBirthDiagnosticHooks.h"
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <dlfcn.h>
#include <stdexcept>
namespace {
template<class F>F original(const char* name){auto* value=::dlsym(RTLD_NEXT,name);if(!value)throw std::runtime_error("Missing original diagnostic HIP symbol");return reinterpret_cast<F>(value);}
}
// Honor the configured SDK's exact version/PTDS symbol spelling, never route
// an intercepted alias through a different legacy runtime entrypoint.
#define LLAMINAR_BIRTH_SYMBOL_IMPL(value) #value
#define LLAMINAR_BIRTH_SYMBOL(value) LLAMINAR_BIRTH_SYMBOL_IMPL(value)
using namespace llaminar2::test::birth_diagnostic;
extern "C" hipError_t hipStreamCreateWithFlags(hipStream_t* value,unsigned flags){
    auto* caller=__builtin_return_address(0);const bool allowed=before(Creator::PrimaryStream,caller);
    const auto result=allowed?original<decltype(&hipStreamCreateWithFlags)>(LLAMINAR_BIRTH_SYMBOL(hipStreamCreateWithFlags))(value,flags):hipErrorOutOfMemory;
    after(Creator::PrimaryStream,result==hipSuccess,result==hipSuccess?*value:nullptr,caller,static_cast<int>(result));return result;
}
extern "C" hipblasStatus_t hipblasCreate(hipblasHandle_t* value){
    auto* caller=__builtin_return_address(0);const bool allowed=before(Creator::PrimaryBLAS,caller);
    const auto result=allowed?original<decltype(&hipblasCreate)>(LLAMINAR_BIRTH_SYMBOL(hipblasCreate))(value):HIPBLAS_STATUS_ALLOC_FAILED;
    after(Creator::PrimaryBLAS,result==HIPBLAS_STATUS_SUCCESS,result==HIPBLAS_STATUS_SUCCESS?*value:nullptr,caller,static_cast<int>(result));return result;
}
extern "C" hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* value){
    auto* caller=__builtin_return_address(0);const bool allowed=before(Creator::SecondaryBLAS,caller);
    const auto result=allowed?original<decltype(&hipblasLtCreate)>(LLAMINAR_BIRTH_SYMBOL(hipblasLtCreate))(value):HIPBLAS_STATUS_ALLOC_FAILED;
    after(Creator::SecondaryBLAS,result==HIPBLAS_STATUS_SUCCESS,result==HIPBLAS_STATUS_SUCCESS?*value:nullptr,caller,static_cast<int>(result));return result;
}
extern "C" hipError_t hipEventCreateWithFlags(hipEvent_t* value,unsigned flags){
    auto* caller=__builtin_return_address(0);const bool allowed=before(Creator::OrderingEvent,caller);
    const auto result=allowed?original<decltype(&hipEventCreateWithFlags)>(LLAMINAR_BIRTH_SYMBOL(hipEventCreateWithFlags))(value,flags):hipErrorOutOfMemory;
    after(Creator::OrderingEvent,result==hipSuccess,result==hipSuccess?*value:nullptr,caller,static_cast<int>(result));return result;
}

extern "C" hipError_t hipStreamDestroy(hipStream_t value){
    auto* caller=__builtin_return_address(0);const auto result=original<decltype(&hipStreamDestroy)>(LLAMINAR_BIRTH_SYMBOL(hipStreamDestroy))(value);
    released(Creator::PrimaryStream,value,result==hipSuccess,caller,static_cast<int>(result));return result;
}

extern "C" hipblasStatus_t hipblasDestroy(hipblasHandle_t value){
    auto* caller=__builtin_return_address(0);const auto result=original<decltype(&hipblasDestroy)>(LLAMINAR_BIRTH_SYMBOL(hipblasDestroy))(value);
    released(Creator::PrimaryBLAS,value,result==HIPBLAS_STATUS_SUCCESS,caller,static_cast<int>(result));return result;
}

extern "C" hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t value){
    auto* caller=__builtin_return_address(0);const auto result=original<decltype(&hipblasLtDestroy)>(LLAMINAR_BIRTH_SYMBOL(hipblasLtDestroy))(value);
    released(Creator::SecondaryBLAS,value,result==HIPBLAS_STATUS_SUCCESS,caller,static_cast<int>(result));return result;
}

extern "C" hipError_t hipEventDestroy(hipEvent_t value){
    auto* caller=__builtin_return_address(0);const auto result=original<decltype(&hipEventDestroy)>(LLAMINAR_BIRTH_SYMBOL(hipEventDestroy))(value);
    released(Creator::OrderingEvent,value,result==hipSuccess,caller,static_cast<int>(result));return result;
}
