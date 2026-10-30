/**
 * @file probe.cpp
 * @brief Device-free replay of the actual NCCL plugin initialization body.
 *
 * The included native source owns selection and lifetime transitions. Mock
 * providers return authored hardware-discovery outcomes; no CUDA/IB provider,
 * copied selection algorithm or replacement native struct executes here.
 */
#include <cstdio>
#include <cstdlib>
#include <memory>
#ifndef NATIVE_NET_SOURCE
#error The exact native implementation source must be bound explicitly
#endif
#include NATIVE_NET_SOURCE

namespace {
struct Observation {
  int gdaki_init=0, gdaki_devices=0, gdaki_finalize=0;
  int proxy_init=0, proxy_devices=0;
  int net_init=0;
} seen;
ncclResult_t probe_init=ncclSuccess, probe_devices=ncclSuccess;
int devices=1;
int64_t selection=-1;
int checks=0;
/** @brief Report the exact replay condition instead of continuing after drift. */
void require(bool value,const char* label,const char* condition) {
  ++checks;
  if(!value){std::fprintf(stderr,"%s: %s\n",label,condition);std::exit(2);}
}
/** @brief Ordinary network provider is present independently of GDAKI capability. */
ncclResult_t netInit(void** ctx,uint64_t,ncclNetCommConfig_t*,ncclDebugLogger_t,ncclProfilerCallback_t) {
  ++seen.net_init;*ctx=reinterpret_cast<void*>(0x101);return ncclSuccess;
}
/** @brief Return an ordinary network independently of GDAKI capability. */
ncclResult_t netDevices(int* count){*count=1;return ncclSuccess;}
/** @brief Retire the authored network context without native resources. */
ncclResult_t netFinalize(void*){return ncclSuccess;}
/** @brief Authored GDAKI probe results; successful contexts have an exact finalizer. */
ncclResult_t gdakiInit(void** ctx,uint64_t,ncclDebugLogger_t) {
  ++seen.gdaki_init;if(probe_init==ncclSuccess)*ctx=reinterpret_cast<void*>(0x202);return probe_init;
}
/** @brief Publish the authored compatible-device discovery result. */
ncclResult_t gdakiDevices(int* count){++seen.gdaki_devices;*count=devices;return probe_devices;}
/** @brief Verify that every successful temporary probe retires its own context. */
ncclResult_t gdakiFinalize(void* ctx) {
  require(ctx==reinterpret_cast<void*>(0x202),"finalize","original probe context");++seen.gdaki_finalize;return ncclSuccess;
}
/** @brief Existing native auto candidate, with independent successful inventory. */
ncclResult_t proxyInit(void** ctx,uint64_t,ncclDebugLogger_t){++seen.proxy_init;*ctx=reinterpret_cast<void*>(0x303);return ncclSuccess;}
/** @brief The existing native proxy candidate has one usable device. */
ncclResult_t proxyDevices(int* count){++seen.proxy_devices;*count=1;return ncclSuccess;}
/** @brief Retire the authored proxy context without native resources. */
ncclResult_t proxyFinalize(void*){return ncclSuccess;}
}

int ncclDebugLevel=0;
/** @brief Keep the native logger device-free and deterministic. */
void ncclDebugLog(ncclDebugLogLevel,unsigned long,const char*,int,const char*,...) {}
/** @brief Native callback symbol; no profiler or accelerator is initialized. */
ncclResult_t ncclProfilerCallback(void**,int,void*,int64_t,void*){return ncclSuccess;}
/** @brief Supply each explicitly authored native selection policy. */
int64_t ncclParamGinType(){return selection;}
ncclGin_t ncclGinIbGdaki{};
ncclGin_t ncclGinIbProxy{};

namespace {
/**
 * @brief Run the real native body twice, checking initial and warm-plugin state.
 * @param label Exact authored control identity.
 * @param mode Native -1 auto, explicit GDAKI or explicit proxy policy.
 * @param initial Actual plugin descriptor, unresolved auto sentinel or null.
 * @param expected Actual selected descriptor; explicit choices never change.
 * @param state Native final state expected after provider observations.
 * @param expected_ginit Expected probe plus selected-provider initialization calls.
 * @param expected_gdevices Expected GDAKI inventory queries.
 * @param expected_finalizers Expected successful temporary-context retirement.
 * @param expected_pinit Expected selected proxy initialization calls.
 */
void replay(const char* label,int64_t mode,ncclGin_t* initial,ncclGin_t* expected,
    ncclNetPluginState_t state,int expected_ginit,int expected_gdevices,int expected_finalizers,int expected_pinit) {
  std::fprintf(stderr, "replay %s\n", label);
  seen={};selection=mode;
  ncclNet_t net{};net.name="authored-native-network";net.init=netInit;net.devices=netDevices;net.finalize=netFinalize;
  netPluginLib_t plugin{};plugin.ncclNet=&net;plugin.ncclNetPluginState=ncclNetPluginStateInitReady;
  plugin.ncclGin=initial;plugin.ncclGinPluginState=initial?ncclNetPluginStateInitReady:ncclNetPluginStateLoadFailed;
  auto comm=std::make_unique<ncclComm>();comm->commHash=0xabc;
  require(ncclNetPluginInit(comm.get(),&plugin)==ncclSuccess,label,"native init result");
  require(plugin.ncclGin==expected,label,"exact descriptor selection");
  require(plugin.ncclGinPluginState==state,label,"native plugin state");
  require(seen.gdaki_init==expected_ginit,label,"GDAKI init count");
  require(seen.gdaki_devices==expected_gdevices,label,"GDAKI device-query count");
  require(seen.gdaki_finalize==expected_finalizers,label,"temporary context finalizes once");
  require(seen.proxy_init==expected_pinit,label,"proxy init count");
  require(plugin.ncclGin!=reinterpret_cast<ncclGin_t*>(-1),label,"no unresolved callable sentinel");
  if(state==ncclNetPluginStateEnabled){
    auto next=std::make_unique<ncclComm>();next->commHash=0xdef;
    require(ncclNetPluginInit(next.get(),&plugin)==ncclSuccess,label,"warm native init result");
    require(plugin.ncclGin==expected && seen.gdaki_init==expected_ginit && seen.gdaki_finalize==expected_finalizers && seen.proxy_init==expected_pinit,label,"warm selection does not repeat discovery");
  }
}
}
/** @brief Replay provider discovery and explicit selection without GPU or IB calls. */
int main(){
  ncclGinIbGdaki.name="authored-GDAKI";ncclGinIbGdaki.init=gdakiInit;ncclGinIbGdaki.devices=gdakiDevices;ncclGinIbGdaki.finalize=gdakiFinalize;
  ncclGinIbProxy.name="authored-proxy";ncclGinIbProxy.init=proxyInit;ncclGinIbProxy.devices=proxyDevices;ncclGinIbProxy.finalize=proxyFinalize;
  ncclGin_t external=ncclGinIbProxy;external.name="authored-external";
  for(int repeat=0;repeat<20;++repeat){
    probe_init=probe_devices=ncclSuccess;devices=2;
    replay("auto-available",-1,reinterpret_cast<ncclGin_t*>(-1),&ncclGinIbGdaki,ncclNetPluginStateEnabled,2,2,1,0);
    devices=0;replay("auto-zero-devices",-1,reinterpret_cast<ncclGin_t*>(-1),&ncclGinIbProxy,ncclNetPluginStateEnabled,1,1,1,1);
    devices=-1;replay("auto-negative-inventory",-1,reinterpret_cast<ncclGin_t*>(-1),&ncclGinIbProxy,ncclNetPluginStateEnabled,1,1,1,1);
    probe_devices=ncclSystemError;replay("auto-query-error",-1,reinterpret_cast<ncclGin_t*>(-1),&ncclGinIbProxy,ncclNetPluginStateEnabled,1,1,1,1);
    probe_init=ncclSystemError;replay("auto-probe-error",-1,reinterpret_cast<ncclGin_t*>(-1),&ncclGinIbProxy,ncclNetPluginStateEnabled,1,0,0,1);
    probe_init=probe_devices=ncclSuccess;devices=2;
    replay("explicit-GDAKI",NCCL_NET_DEVICE_GIN_GDAKI,&ncclGinIbGdaki,&ncclGinIbGdaki,ncclNetPluginStateEnabled,1,1,0,0);
    devices=0;replay("explicit-GDAKI-unavailable",NCCL_NET_DEVICE_GIN_GDAKI,&ncclGinIbGdaki,&ncclGinIbGdaki,ncclNetPluginStateDisabled,1,1,0,0);
    probe_init=ncclSystemError;replay("explicit-GDAKI-error",NCCL_NET_DEVICE_GIN_GDAKI,&ncclGinIbGdaki,&ncclGinIbGdaki,ncclNetPluginStateDisabled,1,0,0,0);
    replay("explicit-proxy",NCCL_NET_DEVICE_GIN_PROXY,&ncclGinIbProxy,&ncclGinIbProxy,ncclNetPluginStateEnabled,0,0,0,1);
    replay("external-auto",-1,&external,&external,ncclNetPluginStateEnabled,0,0,0,1);
    replay("no-GIN",-1,nullptr,nullptr,ncclNetPluginStateLoadFailed,0,0,0,0);
  }
  std::printf("PASS: 220 actual-source selection replays, %d assertions, zero GPU/IB calls\n",checks);
}
