/**
 * @file Test__ROCmEnumerationFailures.cpp
 * @brief Actual ROCm enumeration body replay with deterministic native API doubles.
 *
 * Original immutable metadata is supplied through its real typed inventory.
 * No backend, primary context or capacity is created. Native statuses, current
 * ordinal and reference-release calls are observations, not an admission ledger.
 */
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/rocm/ROCmRuntimeStartup.h"
#include "backends/GPUEnumeration.h"
#include <hip/hip_runtime_api.h>
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>
namespace {
/** @brief One exact deterministic native call outcome program, reset between independent replays. */
struct NativeReplay
{
    int saved=2,current=2,get_calls=0,startup=0,initialize=0;
    bool fail_get=false;
    int fail_device_get=-1,fail_state=-1,fail_memory=-1,fail_select_call=-1,fail_release=-1;
    std::pair<int,int> fail_peer{-1,-1};
    std::array<bool,3> had_context{false,false,true};
    std::vector<int> selects,device_gets,states,memories,releases;
    std::vector<std::pair<int,int>> peers;
    std::vector<std::string> trace;
    std::unique_ptr<const llaminar2::ROCmHardwareInventory> inventory;
    /** @brief Reset both original immutable inventory and volatile recorded native observations. */
    void reset(int count=3)
    {
        *this=NativeReplay{};std::vector<llaminar2::ROCmDeviceHardware> devices;
        for(int i=0;i<count;++i){
            llaminar2::ROCmDeviceHardware d;d.ordinal=i;d.name="Recorded AMD "+std::to_string(i);d.isa_target=i==1?"gfx1100":"gfx906:xnack-";
            d.pci_address="0000:0"+std::to_string(i+1)+":00.0";d.uuid[0]=i+1;d.compute_units=64;
            d.native_wave_width=64;d.maximum_threads_per_block=1024;d.wall_clock_rate_khz=100000;d.lds_per_block=65536;
            d.total_memory=1000;d.last_level_cache=4096;devices.push_back(std::move(d));
        }
        inventory=std::make_unique<const llaminar2::ROCmHardwareInventory>(std::move(devices));
    }
} native;
}
extern "C" {
/** @brief Return the original current ordinal only when native observation succeeds. */
hipError_t hipGetDevice(int* device){++native.get_calls;native.trace.push_back("get");if(native.fail_get)return hipErrorInvalidValue;*device=native.current;return hipSuccess;}
/** @brief Record each exact selection, including failed restoration without changing its ordinal. */
hipError_t hipSetDevice(int device){native.selects.push_back(device);native.trace.push_back("select"+std::to_string(device));if(static_cast<int>(native.selects.size())==native.fail_select_call)return hipErrorInvalidDevice;native.current=device;return hipSuccess;}
/** @brief Failed calls deliberately poison outputs; production may never publish them. */
hipError_t hipMemGetInfo(size_t* free_bytes,size_t* total_bytes){native.memories.push_back(native.current);native.trace.push_back("memory"+std::to_string(native.current));*free_bytes=native.current==1?0:41+native.current;*total_bytes=1000;if(native.current==native.fail_memory){*free_bytes=777;return hipErrorUnknown;}return hipSuccess;}
/** @brief Preserve original native handles; an unknown handle observation is never an absent context. */
hipError_t hipDeviceGet(hipDevice_t* device,int ordinal){native.device_gets.push_back(ordinal);native.trace.push_back("device"+std::to_string(ordinal));if(ordinal==native.fail_device_get)return hipErrorInvalidDevice;*device=ordinal;return hipSuccess;}
/** @brief Replay the exact pre-existing primary-context observation. */
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t device,unsigned int* flags,int* active){native.states.push_back(device);native.trace.push_back("state"+std::to_string(device));if(device==native.fail_state)return hipErrorUnknown;*flags=0;*active=native.had_context.at(device);return hipSuccess;}
/** @brief Record reference-release requests only; no context destruction or memory return is simulated. */
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t device){native.releases.push_back(device);native.trace.push_back("release"+std::to_string(device));return device==native.fail_release?hipErrorUnknown:hipSuccess;}
/** @brief Directed asymmetric successful answers and explicit native failures are distinct. */
hipError_t hipDeviceCanAccessPeer(int* access,int source,int destination){native.peers.emplace_back(source,destination);*access=source<destination;if(std::pair{source,destination}==native.fail_peer){*access=1;return hipErrorUnknown;}return hipSuccess;}
/** @brief Stable native diagnostic witness, never a substituted successful status. */
const char* hipGetErrorString(hipError_t status){switch(status){case hipSuccess:return "hipSuccess";case hipErrorInvalidValue:return "hipErrorInvalidValue";case hipErrorInvalidDevice:return "hipErrorInvalidDevice";default:return "hipErrorUnknown";}}
}
namespace llaminar2 {
/** @brief Test-owned once-boundary observation; the production enumeration calls its actual startup interface. */
void requireROCmRuntimeStartup(){++native.startup;}
/** @brief Record immutable inventory publication boundary without opening a runtime. */
void initializeROCmHardwareMetadata(){++native.initialize;}
/** @return The real validated immutable inventory type, with no lazy device queries. */
const ROCmHardwareInventory& rocmHardwareMetadata(){return *native.inventory;}
namespace pcie_enumeration {
/** @return An empty sysfs observation; PCI identity itself remains the original immutable record. */
PCIeLinkInfo read_pcie_link_info(int,int,int){return {};}
}
}
#ifndef ROCM_ENUMERATION_BODY
#define ROCM_ENUMERATION_BODY "backends/ROCmEnumeration.cpp"
#endif
#include ROCM_ENUMERATION_BODY
namespace {
namespace rocm=llaminar2::rocm_enumeration;
/** @brief Reset each case; no global runtime/factory is shared with another test suite. */
class ROCmEnumerationFailures:public ::testing::Test
{
protected:
    /** @brief Retain original immutable facts and empty native call history for every replay. */
    void SetUp()override{native.reset();}
};
/** @brief Native errors must include the exact failed operation/ordinal; no partial vector may escape. */
template<class Operation> void requiresError(Operation operation,const std::string& text)
{
    try{operation();FAIL()<<"native failure returned a successful enumeration";}
    catch(const std::runtime_error& error){EXPECT_NE(std::string(error.what()).find(text),std::string::npos)<<error.what();}
}
TEST_F(ROCmEnumerationFailures, SuccessPreservesExactCurrentDeviceCapacityAndExistingContexts)
{
    const auto devices=rocm::enumerate_rocm_devices();ASSERT_EQ(devices.size(),3u);
    EXPECT_EQ(devices[0].free_memory_bytes,41u);EXPECT_EQ(devices[1].free_memory_bytes,0u);EXPECT_EQ(devices[2].free_memory_bytes,43u);
    EXPECT_EQ(devices[0].total_memory_bytes,1000u);EXPECT_EQ(devices[0].pci_bus_address,"0000:01:00.0");
    EXPECT_EQ(native.current,native.saved);EXPECT_EQ(native.selects,(std::vector{0,1,2,2}));EXPECT_EQ(native.releases,(std::vector{0,1}));
    EXPECT_EQ(native.device_gets,(std::vector{0,1,2}));EXPECT_EQ(native.states,(std::vector{0,1,2}));
    EXPECT_EQ(native.startup,1);EXPECT_EQ(native.initialize,1);
    const auto first_select=std::find(native.trace.begin(),native.trace.end(),"select0");
    EXPECT_GT(first_select-native.trace.begin(),std::find(native.trace.begin(),native.trace.end(),"state2")-native.trace.begin());
}
TEST_F(ROCmEnumerationFailures, EmptyInventoryDoesNotSelectOrObserveContexts)
{
    native.reset(0);EXPECT_TRUE(rocm::enumerate_rocm_devices().empty());EXPECT_EQ(native.get_calls,0);
    EXPECT_TRUE(native.selects.empty());EXPECT_TRUE(native.device_gets.empty());EXPECT_TRUE(native.releases.empty());
}
TEST_F(ROCmEnumerationFailures, SavedCurrentFailureCannotInventOrdinalZero)
{
    native.fail_get=true;requiresError([]{rocm::enumerate_rocm_devices();},"preserve current device");
    EXPECT_EQ(native.current,native.saved);EXPECT_TRUE(native.selects.empty());EXPECT_TRUE(native.device_gets.empty());EXPECT_TRUE(native.releases.empty());
}
TEST_F(ROCmEnumerationFailures, FailedDeviceHandleObservationCannotBecomeInactiveContext)
{
    native.fail_device_get=1;requiresError([]{rocm::enumerate_rocm_devices();},"hipDeviceGet failed at ordinal 1");
    EXPECT_EQ(native.current,native.saved);EXPECT_TRUE(native.selects.empty());EXPECT_TRUE(native.memories.empty());EXPECT_TRUE(native.releases.empty());
}
TEST_F(ROCmEnumerationFailures, FailedPrimaryStateObservationCannotReleaseExistingContext)
{
    native.had_context[1]=true;native.fail_state=1;
    requiresError([]{rocm::enumerate_rocm_devices();},"hipDevicePrimaryCtxGetState failed at ordinal 1");
    EXPECT_EQ(native.current,native.saved);EXPECT_TRUE(native.selects.empty());EXPECT_TRUE(native.memories.empty());EXPECT_TRUE(native.releases.empty());
}
TEST_F(ROCmEnumerationFailures, FailedCapacitySelectionRestoresCallerAndNeverInventsFreeVRAM)
{
    native.fail_select_call=2;requiresError([]{rocm::enumerate_rocm_devices();},"could not select ordinal 1: hipErrorInvalidDevice");
    EXPECT_EQ(native.current,native.saved);EXPECT_EQ(native.selects,(std::vector{0,1,2}));
    EXPECT_EQ(native.memories,(std::vector{0}));EXPECT_EQ(native.releases,(std::vector{0}));
}
TEST_F(ROCmEnumerationFailures, FailedMemoryObservationRestoresAndCannotPublishPoisonedOutput)
{
    native.fail_memory=1;requiresError([]{rocm::enumerate_rocm_devices();},"failed at ordinal 1: hipErrorUnknown");
    EXPECT_EQ(native.current,native.saved);EXPECT_EQ(native.selects,(std::vector{0,1,2}));
    EXPECT_EQ(native.memories,(std::vector{0,1}));EXPECT_EQ(native.releases,(std::vector{0,1}));
}
TEST_F(ROCmEnumerationFailures, FailedExplicitRestorationPropagatesWithoutRetry)
{
    native.saved=native.current=0;native.fail_select_call=4;
    requiresError([]{rocm::enumerate_rocm_devices();},"restore exact current ordinal 0: hipErrorInvalidDevice");
    EXPECT_EQ(native.selects,(std::vector{0,1,2,0}));EXPECT_EQ(native.current,2);EXPECT_EQ(native.releases,(std::vector{0,1}));
}
TEST_F(ROCmEnumerationFailures, FailedExplicitPrimaryReleaseIsCheckedAndEveryEligibleReleaseAttempted)
{
    native.fail_release=0;requiresError([]{rocm::enumerate_rocm_devices();},"hipDevicePrimaryCtxRelease failed at ordinal 0: hipErrorUnknown");
    EXPECT_EQ(native.current,native.saved);EXPECT_EQ(native.releases,(std::vector{0,1}));EXPECT_EQ(native.selects.size(),4u);
}
TEST_F(ROCmEnumerationFailures, SimultaneousExplicitRestoreAndReleaseErrorsAreBothReported)
{
    native.saved=native.current=0;native.fail_select_call=4;native.fail_release=1;
    try{rocm::enumerate_rocm_devices();FAIL()<<"native cleanup errors returned a successful inventory";}
    catch(const std::runtime_error& error){
        EXPECT_NE(std::string(error.what()).find("restore exact current ordinal 0: hipErrorInvalidDevice"),std::string::npos);
        EXPECT_NE(std::string(error.what()).find("hipDevicePrimaryCtxRelease failed at ordinal 1: hipErrorUnknown"),std::string::npos);
    }
    EXPECT_EQ(native.selects.size(),4u);EXPECT_EQ(native.releases,(std::vector{0,1}));
}
TEST_F(ROCmEnumerationFailures, SuccessfulRestorationOfOriginallyInactiveSavedDeviceIsReleasedOnce)
{
    native.saved=native.current=0;native.fail_select_call=1;
    requiresError([]{rocm::enumerate_rocm_devices();},"could not select ordinal 0");
    EXPECT_EQ(native.current,0);EXPECT_EQ(native.selects,(std::vector{0,0}));EXPECT_EQ(native.releases,(std::vector{0}));
}
TEST_F(ROCmEnumerationFailures, FailedUnwindRestorationIsFatal)
{
    EXPECT_EXIT({std::set_terminate([]{std::_Exit(91);});native.fail_memory=1;native.fail_select_call=3;
        try{rocm::enumerate_rocm_devices();}catch(...){std::_Exit(92);}std::_Exit(93);},::testing::ExitedWithCode(91),"ROCm discovery cleanup failed");
}
TEST_F(ROCmEnumerationFailures, FailedUnwindReleaseIsFatal)
{
    EXPECT_EXIT({std::set_terminate([]{std::_Exit(91);});native.fail_memory=1;native.fail_release=0;
        try{rocm::enumerate_rocm_devices();}catch(...){std::_Exit(92);}std::_Exit(93);},::testing::ExitedWithCode(91),"ROCm discovery cleanup failed");
}
TEST_F(ROCmEnumerationFailures, DirectedPeerAnswersPreserveOriginalOrderAndLegitimateAbsentEdges)
{
    std::vector<llaminar2::ComputeDevice> devices(3);devices[0].device_id=7;devices[1].device_id=3;devices[2].device_id=9;
    const auto matrix=rocm::query_p2p_matrix(devices);EXPECT_EQ(matrix.device_ids,(std::vector{7,3,9}));
    EXPECT_TRUE(matrix.can_access[0][0]);EXPECT_FALSE(matrix.can_access[0][1]);EXPECT_TRUE(matrix.can_access[1][0]);EXPECT_EQ(native.peers.size(),6u);
}
TEST_F(ROCmEnumerationFailures, FailedPeerQueryCannotBecomeAnAbsentEdge)
{
    std::vector<llaminar2::ComputeDevice> devices(3);devices[0].device_id=7;devices[1].device_id=3;devices[2].device_id=9;
    native.fail_peer={3,7};requiresError([&]{rocm::query_p2p_matrix(devices);},"ordinals 3 -> 7: hipErrorUnknown");
    EXPECT_EQ(native.peers,(std::vector<std::pair<int,int>>{{7,3},{7,9},{3,7}}));EXPECT_TRUE(native.selects.empty());
}
}
