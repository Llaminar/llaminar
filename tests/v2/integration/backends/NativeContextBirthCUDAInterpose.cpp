/** @file NativeContextBirthCUDAInterpose.cpp
 * @brief Diagnostic-only CUDA creator prefixes through original resolved SDK symbols.
 * Every successful call invokes the real SDK. Injected errors exercise the
 * original unpublished-constructor rollback; no production object is patched.
 */
#include "NativeContextBirthDiagnosticHooks.h"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <dlfcn.h>
#include <stdexcept>
namespace {
template<class F>F original(const char* name){auto* value=::dlsym(RTLD_NEXT,name);if(!value)throw std::runtime_error("Missing original diagnostic CUDA symbol");return reinterpret_cast<F>(value);}
}
// Honor the configured SDK's exact version/PTDS symbol spelling, never route
// an intercepted alias through a different legacy runtime entrypoint.
#define LLAMINAR_BIRTH_SYMBOL_IMPL(value) #value
#define LLAMINAR_BIRTH_SYMBOL(value) LLAMINAR_BIRTH_SYMBOL_IMPL(value)
using namespace llaminar2::test::birth_diagnostic;
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* value,unsigned flags){
    auto* caller=__builtin_return_address(0);const bool allowed=before(Creator::PrimaryStream,caller);
    const auto result=allowed?original<decltype(&cudaStreamCreateWithFlags)>(LLAMINAR_BIRTH_SYMBOL(cudaStreamCreateWithFlags))(value,flags):cudaErrorMemoryAllocation;
    after(Creator::PrimaryStream,result==cudaSuccess,result==cudaSuccess?*value:nullptr,caller,static_cast<int>(result));return result;
}
extern "C" cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* value){
    auto* caller=__builtin_return_address(0);const bool allowed=before(Creator::PrimaryBLAS,caller);
    const auto result=allowed?original<decltype(&cublasCreate_v2)>(LLAMINAR_BIRTH_SYMBOL(cublasCreate_v2))(value):CUBLAS_STATUS_ALLOC_FAILED;
    after(Creator::PrimaryBLAS,result==CUBLAS_STATUS_SUCCESS,result==CUBLAS_STATUS_SUCCESS?*value:nullptr,caller,static_cast<int>(result));return result;
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* value){
    auto* caller=__builtin_return_address(0);const bool allowed=before(Creator::SecondaryBLAS,caller);
    const auto result=allowed?original<decltype(&cublasLtCreate)>(LLAMINAR_BIRTH_SYMBOL(cublasLtCreate))(value):CUBLAS_STATUS_ALLOC_FAILED;
    after(Creator::SecondaryBLAS,result==CUBLAS_STATUS_SUCCESS,result==CUBLAS_STATUS_SUCCESS?*value:nullptr,caller,static_cast<int>(result));return result;
}
extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* value,unsigned flags){
    auto* caller=__builtin_return_address(0);const bool allowed=before(Creator::OrderingEvent,caller);
    const auto result=allowed?original<decltype(&cudaEventCreateWithFlags)>(LLAMINAR_BIRTH_SYMBOL(cudaEventCreateWithFlags))(value,flags):cudaErrorMemoryAllocation;
    after(Creator::OrderingEvent,result==cudaSuccess,result==cudaSuccess?*value:nullptr,caller,static_cast<int>(result));return result;
}

extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t value){
    auto* caller=__builtin_return_address(0);const auto result=original<decltype(&cudaStreamDestroy)>(LLAMINAR_BIRTH_SYMBOL(cudaStreamDestroy))(value);
    released(Creator::PrimaryStream,value,result==cudaSuccess,caller,static_cast<int>(result));return result;
}

extern "C" cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t value){
    auto* caller=__builtin_return_address(0);const auto result=original<decltype(&cublasDestroy_v2)>(LLAMINAR_BIRTH_SYMBOL(cublasDestroy_v2))(value);
    released(Creator::PrimaryBLAS,value,result==CUBLAS_STATUS_SUCCESS,caller,static_cast<int>(result));return result;
}

extern "C" cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t value){
    auto* caller=__builtin_return_address(0);const auto result=original<decltype(&cublasLtDestroy)>(LLAMINAR_BIRTH_SYMBOL(cublasLtDestroy))(value);
    released(Creator::SecondaryBLAS,value,result==CUBLAS_STATUS_SUCCESS,caller,static_cast<int>(result));return result;
}

extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t value){
    auto* caller=__builtin_return_address(0);const auto result=original<decltype(&cudaEventDestroy)>(LLAMINAR_BIRTH_SYMBOL(cudaEventDestroy))(value);
    released(Creator::OrderingEvent,value,result==cudaSuccess,caller,static_cast<int>(result));return result;
}
