/** @file Test__NativeGraphDestruction.cpp
 * @brief Actual native reset bodies with SDK-return-code injection and no GPU runtime.
 */
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>
#include "../../../../src/v2/backends/NativeParallelGraphBranch.h"
#define LOG_ERROR(...) ((void)0)
#define CUDA_WARN_IF_FAIL(call) ((void)(call))
#define HIP_WARN_IF_FAIL(call) ((void)(call))
namespace llaminar2 {
using cudaError_t=int;using hipError_t=int;using hipGraph_t=void*;
constexpr int cudaSuccess=0,hipSuccess=0;
struct SDK {static inline int fail=0,activate=0,exec=0,graph=0;static void reset(int f){fail=f;activate=exec=graph=0;}};
int cudaSetDevice(int){++SDK::activate;return SDK::fail==1?1:0;}
int cudaGraphExecDestroy(void*){++SDK::exec;return SDK::fail==2?1:0;}
int cudaGraphDestroy(void*){++SDK::graph;return SDK::fail==3?1:0;}
int hipGraphExecDestroy(void*){++SDK::exec;return SDK::fail==2?1:0;}
int hipGraphDestroy(void*){++SDK::graph;return SDK::fail==3?1:0;}
int hipMemGetInfo(std::size_t*a,std::size_t*b){*a=100;*b=200;return 0;}
bool vramBomEnabled(){return false;}
void logVramBomLine(const char*,const std::string&){}
struct OrderedTimelineRecording{enum class State{Failed};State state=State::Failed;};
struct Common {
 // Match the production definition authority and imported-frontier types so
 // replaying a real reset revokes the same metadata before native teardown.
 std::shared_ptr<const void> definition_scope_=std::make_shared<const std::byte>();
 detail::NativeCapturedFragmentFrontiers<void*,void*> fragment_frontiers_;
 bool native_capture_active_=false;int device_ordinal_=0;
 // Recording-definition scope and fragment frontiers are retired first by the
 // production bodies, before any native handle is released.
 std::shared_ptr<const void> definition_scope_=std::make_shared<const int>(1);
 struct FragmentFrontiers{std::vector<int>nodes{1,2};void clear(){nodes.clear();}} fragment_frontiers_;
 void*exec_=reinterpret_cast<void*>(0x100);void*graph_=reinterpret_cast<void*>(0x200);
 std::size_t resident_memory_bytes_=1024,node_count_=5;int cleanup=0;
 bool activateOwner(const char*){++SDK::activate;return SDK::fail!=1;}
 int endNativeCapture(hipGraph_t*out){*out=nullptr;native_capture_active_=false;return 0;}
 bool endCapture(){native_capture_active_=false;return true;}
 void destroyOrderedTimelineTimingEvents(){++cleanup;}
};
struct CUDAGraphCapture:Common {
 enum class RecordingRole{Owner,Fragment};RecordingRole recording_role_=RecordingRole::Owner;
 enum class FragmentState{Unrecorded};FragmentState fragment_state_=FragmentState::Unrecorded;
 std::shared_ptr<OrderedTimelineRecording> timeline_recording_;
 std::vector<int>fragment_nodes_;void*fragment_entry_=nullptr,*fragment_exit_=nullptr;
 void reset();
};
struct HIPGraphCapture:Common {void reset();};
#include "NativeGraphDestructionBodies.inc"
template<class Capture> void failure(int phase){
 SDK::reset(phase);Capture capture;
 std::set_terminate([]{std::_Exit(78);});
 // The termination observer checks actual still-owned handle state and every
 // successful-prefix SDK call before accepting fatal cleanup behavior.
 static Capture* original=nullptr;original=&capture;
 std::set_terminate([]{
  const bool handles=SDK::fail==3?!original->exec_&&original->graph_:original->exec_&&original->graph_;
  const bool calls=SDK::activate==1&&SDK::exec==(SDK::fail==1?0:1)&&SDK::graph==(SDK::fail==3?1:0);
  std::_Exit(handles&&calls&&!original->definition_scope_&&original->cleanup==0?77:79);
 });
 capture.reset();std::_Exit(0);
}
TEST(NativeGraphDestruction, CUDAActualFailureReturnsPreserveOriginalHandles){
 for(int phase:{1,2,3})EXPECT_EXIT(failure<CUDAGraphCapture>(phase),::testing::ExitedWithCode(77),"");
}
TEST(NativeGraphDestruction, ROCmActualFailureReturnsPreserveOriginalHandles){
 for(int phase:{1,2,3})EXPECT_EXIT(failure<HIPGraphCapture>(phase),::testing::ExitedWithCode(77),"");
}
template<class Capture>void success(){
 for(bool executable:{false,true})for(bool graph:{false,true}){
  SDK::reset(0);Capture capture;if(!executable)capture.exec_=nullptr;if(!graph)capture.graph_=nullptr;
  capture.reset();EXPECT_EQ(capture.exec_,nullptr);EXPECT_EQ(capture.graph_,nullptr);
  EXPECT_EQ(capture.definition_scope_,nullptr);EXPECT_TRUE(capture.fragment_frontiers_.nodes.empty());
  EXPECT_EQ(SDK::exec,executable?1:0);EXPECT_EQ(SDK::graph,graph?1:0);EXPECT_EQ(capture.cleanup,1);
  capture.reset();EXPECT_EQ(SDK::exec,executable?1:0);EXPECT_EQ(SDK::graph,graph?1:0);
 }
}
/** @brief Retained original definitions must not preserve an invalidated parent frontier. */
template<class Capture>void metadataReset(){
 SDK::reset(0);Capture capture;
 const auto definition=capture.definition_scope_;
 const auto original_graph=capture.graph_;
 const auto original_nodes=capture.node_count_;
 void* exit=reinterpret_cast<void*>(0x300);
 capture.fragment_frontiers_.remember(&capture,definition,original_graph,original_nodes,
                                     std::span<void* const>(&exit,1));
 ASSERT_EQ(capture.fragment_frontiers_.require(&capture,definition,original_graph,original_nodes).size(),1u);
 capture.reset();
 EXPECT_EQ(capture.definition_scope_,nullptr);
 // Keep the original definition alive independently: expiration alone must
 // not conceal a reset that forgot to clear imported native frontier state.
 EXPECT_TRUE(capture.fragment_frontiers_.require(&capture,definition,original_graph,original_nodes).empty());
 capture.reset();
 EXPECT_TRUE(capture.fragment_frontiers_.require(&capture,definition,original_graph,original_nodes).empty());
 EXPECT_EQ(SDK::exec,1);EXPECT_EQ(SDK::graph,1);
}
TEST(NativeGraphDestruction, CUDAActualDefinitionAndFrontierRevocation){metadataReset<CUDAGraphCapture>();}
TEST(NativeGraphDestruction, ROCmActualDefinitionAndFrontierRevocation){metadataReset<HIPGraphCapture>();}
TEST(NativeGraphDestruction, CUDAActualPartialAndRepeatedCleanup){success<CUDAGraphCapture>();}
TEST(NativeGraphDestruction, ROCmActualPartialAndRepeatedCleanup){success<HIPGraphCapture>();}
}
