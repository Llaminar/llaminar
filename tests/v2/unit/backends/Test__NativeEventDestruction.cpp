/**
 * @file Test__NativeEventDestruction.cpp
 * @brief Exact four production cleanup bodies replayed against recorded fake SDK failures.
 *
 * The build extracts actual method bodies without rewriting them. Child processes
 * prove failure terminates before a caller can release its original obligation.
 * Frozen original bodies reproduce the unsafe return. Installed tests need no
 * compiler, vendor SDK, accelerator, model, or native context.
 */
#include <gtest/gtest.h>
#include <exception>
#include <unistd.h>

namespace {
/** @brief Independent recorded SDK statuses and native call evidence. */
struct SDK {int select_status=0,destroy_status=0,selections=0,destructions=0,selected_ordinal=-1;} sdk;
using cudaEvent_t=void*;
using hipEvent_t=void*;
using cudaError_t=int;
constexpr int cudaSuccess=0,hipSuccess=0;
int cudaSetDevice(int ordinal){++sdk.selections;if(!sdk.select_status)sdk.selected_ordinal=ordinal;return sdk.select_status;}
int hipSetDevice(int ordinal){++sdk.selections;if(!sdk.select_status)sdk.selected_ordinal=ordinal;return sdk.select_status;}
int cudaEventDestroy(void*){++sdk.destructions;return sdk.destroy_status;}
int hipEventDestroy(void*){++sdk.destructions;return sdk.destroy_status;}
const char* cudaGetErrorString(int){return "recorded SDK failure";}
bool setAMDDeviceForResource(int ordinal,const char*){++sdk.selections;if(!sdk.select_status)sdk.selected_ordinal=ordinal;return !sdk.select_status;}
struct HipDeviceSaveRestore {};
#define LOG_ERROR(...) do {} while(false)
#define CUDA_CHECK_VOID(call) do {if((call)!=cudaSuccess)return;} while(false)
#define HIP_CHECK_VOID(call) do {if((call)!=hipSuccess)return;} while(false)
#define CUDA_WARN_IF_FAIL(call) do {(void)(call);} while(false)
#define HIP_WARN_IF_FAIL(call) do {(void)(call);} while(false)
#define DECLARE_REPLAY_OWNERS \
struct NvidiaDeviceContext {int device_ordinal_=0;void destroyEvent(void*);}; \
struct AMDDeviceContext {int device_ordinal_=0;void destroyEvent(void*);}; \
struct CUDABackend {int device_count_=2;void destroyEvent(void*,int);}; \
struct ROCmBackend {int device_count_=2;void destroyEvent(void*,int);};
namespace current {
DECLARE_REPLAY_OWNERS
#include "NativeEventDestructionBodies.inc"
}
namespace original {
DECLARE_REPLAY_OWNERS
#include "fixtures/OriginalNativeEventDestructionBodies.inc"
}
#undef DECLARE_REPLAY_OWNERS
/** @brief Actual method selection changes only fixture owner kind, never its body. */
enum class Owner {CUDAWorker,ROCmWorker,CUDABackend,ROCmBackend};
template<bool Original> void call(Owner owner,void* event,int device){
    if constexpr(Original){
        switch(owner){
        case Owner::CUDAWorker:original::NvidiaDeviceContext{}.destroyEvent(event);break;
        case Owner::ROCmWorker:original::AMDDeviceContext{}.destroyEvent(event);break;
        case Owner::CUDABackend:original::CUDABackend{}.destroyEvent(event,device);break;
        case Owner::ROCmBackend:original::ROCmBackend{}.destroyEvent(event,device);break;
        }
    }else{
        switch(owner){
        case Owner::CUDAWorker:current::NvidiaDeviceContext{}.destroyEvent(event);break;
        case Owner::ROCmWorker:current::AMDDeviceContext{}.destroyEvent(event);break;
        case Owner::CUDABackend:current::CUDABackend{}.destroyEvent(event,device);break;
        case Owner::ROCmBackend:current::ROCmBackend{}.destroyEvent(event,device);break;
        }
    }
}
/** @brief A return models the exact point where an outer opaque owner could release. */
template<bool Original> void failedCleanup(Owner owner,int selection,int destruction,int device=0){
    std::set_terminate([]{_exit(86);});
    sdk={selection,destruction,0,0};
    call<Original>(owner,reinterpret_cast<void*>(0x1000),device);
    _exit(87); // Unsafe: cleanup returned and the caller can drop its commitment.
}
class NativeEventDestruction:public ::testing::TestWithParam<Owner>{};
/** @test Null denotes no created resource and must not consult either runtime. */
TEST_P(NativeEventDestruction, NullIsNoop){
    sdk={7,9,0,0};call<false>(GetParam(),nullptr,-1);
    EXPECT_EQ(sdk.selections,0);EXPECT_EQ(sdk.destructions,0);
}
/** @test Successful cleanup selects original device then destroys exactly once on every replay. */
TEST_P(NativeEventDestruction, SuccessAndResetReplay){
    for(int replay=0;replay<20;++replay){
        sdk={};call<false>(GetParam(),reinterpret_cast<void*>(0x1000),1);
        EXPECT_EQ(sdk.selections,1);EXPECT_EQ(sdk.destructions,1);
        EXPECT_EQ(sdk.selected_ordinal,(GetParam()==Owner::CUDAWorker || GetParam()==Owner::ROCmWorker)?0:1);
    }
}
/** @test Failed selection cannot return to a commitment-releasing caller. */
TEST_P(NativeEventDestruction, SelectionFailureIsFatal){
    GTEST_FLAG_SET(death_test_style,"threadsafe");const auto owner=GetParam();
    EXPECT_EXIT(failedCleanup<false>(owner,7,0),::testing::ExitedWithCode(86),"");
}
/** @test Failed destruction cannot return to a commitment-releasing caller. */
TEST_P(NativeEventDestruction, DestructionFailureIsFatal){
    GTEST_FLAG_SET(death_test_style,"threadsafe");const auto owner=GetParam();
    EXPECT_EXIT(failedCleanup<false>(owner,0,9),::testing::ExitedWithCode(86),"");
}
/** @test Actual pre-fix bodies return after both classes of failure. */
TEST_P(NativeEventDestruction, OriginalBodiesExposePrematureRelease){
    GTEST_FLAG_SET(death_test_style,"threadsafe");const auto owner=GetParam();
    EXPECT_EXIT(failedCleanup<true>(owner,7,0),::testing::ExitedWithCode(87),"");
    EXPECT_EXIT(failedCleanup<true>(owner,0,9),::testing::ExitedWithCode(87),"");
}
INSTANTIATE_TEST_SUITE_P(ActualOwners,NativeEventDestruction,::testing::Values(
    Owner::CUDAWorker,Owner::ROCmWorker,Owner::CUDABackend,Owner::ROCmBackend));
/** @test Invalid non-null backend owner cannot masquerade as an absent resource. */
TEST(NativeEventDestructionBackend, InvalidOriginalDeviceIsFatal){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto owner:{Owner::CUDABackend,Owner::ROCmBackend})for(const int device:{-1,2}){
        EXPECT_EXIT(failedCleanup<false>(owner,0,0,device),::testing::ExitedWithCode(86),"");
        EXPECT_EXIT(failedCleanup<true>(owner,0,0,device),::testing::ExitedWithCode(87),"");
    }
}
}
