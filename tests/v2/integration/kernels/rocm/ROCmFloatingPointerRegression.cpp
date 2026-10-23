/**
 * @file ROCmFloatingPointerRegression.cpp
 * @brief Public prepared floating engines versus frozen immediate-original staged kernels.
 *
 * The unchanged C++ engine API allows this exact fixture to calibrate against
 * pre102 production before installing descriptor102. Storage/streams/events
 * reuse the existing reduction fixture owners; no test dispatch enters serving.
 */
#include "ROCmFloatingRegressionSupport.h"
#include "ROCmOriginalFloatingPointerOracle.h"
#include "kernels/rocm/gemm/ROCmFP32ProjectionLoadPolicy.h"
#include <cxxabi.h>
#include "kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#if __has_include("kernels/common/FixedFloatingProjectionGroup.h")
#include "kernels/common/FloatingPointVerifierLaunch.h"
#define QFN_POINTER102_VALUE_ABI 1
#else
#define QFN_POINTER102_VALUE_ABI 0
#endif
namespace qfn_pointer102_regression {
using namespace llaminar2;
using namespace qfn_reduction_regression;
using Kernel=rocm::ROCmFloatingPointGemmKernel;
/** @brief Distinct finite full-mantissa values with opposing signs/exponents. */
float sample(std::size_t i,unsigned seed) {
 unsigned x=unsigned(i)+seed;x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
 return std::bit_cast<float>((x&0x807fffffu)|((123u+(x>>24)%9u)<<23));
}
/**
 * @brief Authenticate exact custom/original node names and compact live geometry.
 * @param graph Retained graph under inspection.
 * @param expectedZ Exact consumer-grid z multiset; includes every independent binding.
 * @param expectedPublications Original staged nodes, zero for value-only custom graphs.
 * @param n Original columns. @param m Original physical rows. @param k Immutable dot width.
 * @param original Require every kernel to belong to the separately prefixed frozen oracle.
 * @param type Original FP32/native16 consumer family.
 * @param copyTarget Optional exact mapped output destination.
 * @param copySource Optional exact original HBM redirect source.
 */
void inspect(Captured &graph,std::vector<unsigned> expectedZ,int expectedPublications,int n,int m,int k,
 bool original,int type,void *copyTarget=nullptr,void *copySource=nullptr) {
 std::size_t count=0;checked(hipGraphGetNodes(graph.graph,nullptr,&count));
 std::vector<hipGraphNode_t> nodes(count);checked(hipGraphGetNodes(graph.graph,nodes.data(),&count));
 int publications=0,copies=0;std::vector<unsigned> actualZ;
 ASSERT_EQ(count,expectedZ.size()+std::size_t(expectedPublications+(copyTarget?1:0)));
 for(auto node:nodes) {
  hipGraphNodeType kind;checked(hipGraphNodeGetType(node,&kind));
  if(kind==hipGraphNodeTypeMemcpy) {
   ++copies;ASSERT_NE(copyTarget,nullptr);hipMemcpy3DParms params{};checked(hipGraphMemcpyNodeGetParams(node,&params));
   EXPECT_EQ(params.kind,hipMemcpyDeviceToDevice);EXPECT_EQ(params.extent.width,std::size_t(m)*n*sizeof(float));
   EXPECT_EQ(params.extent.height,1u);EXPECT_EQ(params.extent.depth,1u);
   EXPECT_EQ(params.dstPtr.ptr,copyTarget);EXPECT_EQ(params.srcPtr.ptr,copySource);continue;
  }
  ASSERT_EQ(kind,hipGraphNodeTypeKernel);
  hipKernelNodeParams params{};checked(hipGraphKernelNodeGetParams(node,&params));
  const char *raw=hipKernelNameRefByPtr(params.func,graph.stream);
  ASSERT_NE(raw,nullptr);const std::string name(raw);ASSERT_FALSE(name.empty());
  EXPECT_EQ(name.find("pointer102Original_")!=std::string::npos,original)<<name;
  EXPECT_EQ(params.blockDim.y,1u);EXPECT_EQ(params.blockDim.z,1u);
  if(name.find("fp32_stage_batched_projection_pointers_kernel")!=std::string::npos) {
   ++publications;EXPECT_EQ(params.blockDim.x,8u);EXPECT_EQ(params.gridDim.x,1u);
   EXPECT_EQ(params.gridDim.y,1u);EXPECT_EQ(params.gridDim.z,1u);
  } else {
   ASSERT_NE(name.find(type==0?"fp32_small_n_batched_projection_kernel":"fp32x16_batched_projection_kernel"),std::string::npos)<<name;
   EXPECT_EQ(params.blockDim.x,256u);EXPECT_EQ(params.gridDim.x,unsigned(n));EXPECT_EQ(params.gridDim.y,unsigned(m));
   if(!original && type==0) {
    // Graph topology is proof: validate the actual mode of this native chunk,
    // including partial final chunks, without inspecting mutable live counts.
    std::string readable=name;
    if(name.starts_with("_Z")) {
     int status=0;
     std::unique_ptr<char,decltype(&std::free)> decoded(
      abi::__cxa_demangle(name.c_str(),nullptr,nullptr,&status),&std::free);
     ASSERT_EQ(status,0)<<name;ASSERT_NE(decoded.get(),nullptr);readable=decoded.get();
    }
    const int mode=static_cast<int>(rocm::selectFP32ProjectionLoadPairs(m,n,int(params.gridDim.z),k));
    const std::string expected="fp32_small_n_batched_projection_kernel<"+std::to_string(mode)+">";
    ASSERT_NE(readable.find(expected),std::string::npos)<<readable<<" expected "<<expected;
   }
   actualZ.push_back(params.gridDim.z);
  }
 }
 std::sort(expectedZ.begin(),expectedZ.end());std::sort(actualZ.begin(),actualZ.end());
 EXPECT_EQ(actualZ,expectedZ);EXPECT_EQ(std::accumulate(actualZ.begin(),actualZ.end(),0u),std::accumulate(expectedZ.begin(),expectedZ.end(),0u));
 ASSERT_EQ(publications,expectedPublications);ASSERT_EQ(copies,copyTarget?1:0);
}
/** @brief Authored source witnesses; never an alternate production arithmetic policy. */
enum class Witness : unsigned {
 Finite, SourceNaNs, ActivationNaNs, SignedZeros, SubnormalInputs, NativeSpecialWords, DualNaNs, OrderedDotOne
};
/** @brief Diagnostic graph selected after independent byte proof; never serving policy. */
enum class ProfileSide { None, Candidate, Original };
/** @brief Parse the complete diagnostic selection; absent selectors preserve paired events. */
struct ProfileSelection { int type,m,n,k; ProfileSide side; };
/** @brief Reject partial, empty or unknown shape/side selectors before any GPU work. */
std::optional<ProfileSelection> profileSelection() {
 const char *shape=std::getenv("QFN_POINTER102_PROFILE_SHAPE");
 const char *side=std::getenv("QFN_POINTER102_PROFILE_SIDE");
 if(!shape&&!side)return std::nullopt;
 if(!shape||!side)throw std::invalid_argument("Pointer profile requires both shape and side");
 const std::string_view name(shape),which(side);
 const auto selected=which=="candidate"?ProfileSide::Candidate:which=="original"?ProfileSide::Original:ProfileSide::None;
 if(selected==ProfileSide::None)throw std::invalid_argument("Pointer profile side must be candidate or original");
 for(const auto &entry:std::array{std::pair{"hc-m1",ProfileSelection{0,1,4,10240,selected}},
  std::pair{"hc-m512",ProfileSelection{0,512,4,10240,selected}},
  std::pair{"key-m1",ProfileSelection{2,1,128,2560,selected}},
  std::pair{"key-m512",ProfileSelection{2,512,128,2560,selected}},
  std::pair{"query-m1",ProfileSelection{2,1,512,2560,selected}},
  std::pair{"query-m512",ProfileSelection{2,512,512,2560,selected}}})
  if(name==entry.first)return entry.second;
 throw std::invalid_argument("Pointer profile shape is not a source-exact case");
}
/** @brief Named fixture recording choices; defaults preserve the ordinary shared-input prefix case. */
struct Options {
 bool independent=false; ///< Each public single/descriptor entry owns a distinct A bank.
 bool holes=false; ///< Original request-major lengths replace the prefix authority.
 bool mapped=false; ///< Exercise ordinary full-live mapped redirect publication.
 Witness witness=Witness::Finite; ///< Independently authored source words.
 bool timing=false; ///< Explicit disabled native-event benchmark, never preflight.
 ProfileSide profile=ProfileSide::None; ///< One-copy isolated diagnostic after byte proof.
 bool direct=false; ///< Post102-only raw descriptor audit; original calibration uses public singles.
};
/**
 * @brief One public-engine capture compared with independent frozen staged math.
 * @param type Native source FP32/FP16/BF16. @param m Physical rows.
 * @param n Columns. @param k Reduction width. @param count Independent B/C owners.
 * @param options Named input, row-publication, source-witness and observation choices.
 */
void run(int type,int m,int n,int k,int count,Options options={}) {
 const auto [independent,holes,mapped,pattern,timing,profile,direct]=options;
 SCOPED_TRACE(::testing::Message()<<"type="<<type<<" M/N/K="<<m<<'/'<<n<<'/'<<k
  <<" count="<<count<<" independent="<<independent<<" holes="<<holes<<" mapped="<<mapped<<" pattern="<<static_cast<unsigned>(pattern));
 if(holes && (m%3 || independent || mapped))throw std::invalid_argument("Pointer fixture holes require three shared-input requests");
 if(mapped && count!=1)throw std::invalid_argument("Mapped fixture uses the public single redirect API");
 const auto device=DeviceId::rocm(0);const std::size_t inSize=std::size_t(m)*k,outSize=std::size_t(m+1)*n;
 // All borrowed DMA sources precede native stream owners, including exception paths.
 std::vector<std::vector<float>> hostA(count),hostB(count),poisoned(count),dirty(count),actual(count),gold(count);
 std::vector<std::vector<std::uint16_t>> native(count);
 std::vector<std::unique_ptr<FP32Tensor>> inputs,outputs;
 std::vector<std::shared_ptr<Allocation<float>>> wide;
 std::vector<std::shared_ptr<Allocation<std::uint16_t>>> narrow;
 std::vector<std::unique_ptr<Allocation<float>>> oracleOutputs;
 std::vector<std::unique_ptr<Kernel>> engines;
 Allocation<const float*> oldA(8),oldB(8);Allocation<float*> oldC(8);Allocation<std::int32_t> live(3);
 std::array<std::int32_t,3> lengths{m,m,m};
 std::vector<const float*> a(count);std::vector<const void*> b(count);std::vector<float*> c(count),oc(count);
 for(int p=0;p<count;++p) {
  hostA[p].resize(inSize);hostB[p].resize(std::size_t(n)*k);native[p].resize(std::size_t(n)*k);
  for(std::size_t i=0;i<inSize;++i)hostA[p][i]=sample(i,17u+unsigned(p)*29);
  for(std::size_t i=0;i<hostB[p].size();++i) {
   hostB[p][i]=sample(i,91u+unsigned(p)*37);
   native[p][i]=type==1?std::uint16_t(0x3800u+((i*17+p*31)&1023u)):
    std::uint16_t(std::bit_cast<unsigned>(hostB[p][i])>>16);
  }
  if(pattern==Witness::OrderedDotOne) {
   if(type!=0 || k<768)throw std::invalid_argument("Ordered-dot witness requires FP32 and three lane iterations");
   std::fill(hostA[p].begin(),hostA[p].end(),0x1p12f);
   std::fill(hostB[p].begin(),hostB[p].end(),0.f);
   // The last live 256-coordinate bank may have only one lane. Keep every
   // authored word in bounds, and retain one lane's original iteration order.
   const int lanes=std::min(64,(k-1)%256+1),origin=((k-1)/256-2)*256;
   for(int column=0;column<n;++column) {
    const int lane=(p*7+column*13)%lanes;
    const bool negative=(p+column)&1;
    const std::size_t base=std::size_t(column)*k+origin+lane;
    hostB[p][base]=negative?-0x1p12f:0x1p12f;
    hostB[p][base+256]=negative?0x1p12f:-0x1p12f;
    hostB[p][base+512]=0x1p-12f;
   }
  }
  if((pattern==Witness::SourceNaNs||pattern==Witness::DualNaNs) && k>=129) {
   if(type==0){hostB[p][0]=std::bit_cast<float>(0x7fc23456u+unsigned(p));hostB[p][128]=std::bit_cast<float>(0xffc65432u+unsigned(p));}
   else {native[p][0]=std::uint16_t((type==1?0x7e11:0x7fc1)+p);native[p][128]=std::uint16_t((type==1?0xfe22:0xffe2)+p);}
  }
  if(pattern==Witness::ActivationNaNs||pattern==Witness::DualNaNs){hostA[p][0]=std::bit_cast<float>(0x7fc12345u+unsigned(p));if(k>128)hostA[p][128]=std::bit_cast<float>(0xffc54321u+unsigned(p));}
  if(pattern==Witness::SignedZeros)for(std::size_t i=0;i<inSize;++i)hostA[p][i]=(i&1)?-0.f:0.f;
  if(pattern==Witness::SubnormalInputs)for(std::size_t i=0;i<inSize;++i)hostA[p][i]=(i&1)?-0x1p-140f:0x1p-140f;
  if(pattern==Witness::NativeSpecialWords) {
   constexpr unsigned words[]{0,0x80000000u,1,0x007fffffu,0x00800000u,0x7f7fffffu,0x7f800000u,0xff800000u};
   constexpr std::uint16_t halves[]{0,0x8000,1,0x03ff,0x0400,0x7bff,0x7c00,0xfc00};
   constexpr std::uint16_t bfloat[]{0,0x8000,1,0x007f,0x0080,0x7f7f,0x7f80,0xff80};
   for(int column=0;column<n;++column) {
    const std::size_t index=std::size_t(column)*k;
    if(type==0)hostB[p][index]=std::bit_cast<float>(words[(column+p)%8]);
    else native[p][index]=type==1?halves[(column+p)%8]:bfloat[(column+p)%8];
   }
  }
  inputs.push_back(std::make_unique<FP32Tensor>(std::vector<std::size_t>{std::size_t(m),std::size_t(k)}));
  std::copy(hostA[p].begin(),hostA[p].end(),inputs.back()->mutable_data());
  outputs.push_back(mapped?FP32Tensor::createMapped({std::size_t(m+1),std::size_t(n)},device):
   std::make_unique<FP32Tensor>(std::vector<std::size_t>{std::size_t(m+1),std::size_t(n)}));
  if(!outputs.back())throw std::runtime_error("Mapped fixture allocation failed");
  wide.push_back(std::make_shared<Allocation<float>>(std::size_t(n)*k));
  narrow.push_back(std::make_shared<Allocation<std::uint16_t>>(std::size_t(n)*k));
  oracleOutputs.push_back(std::make_unique<Allocation<float>>(outSize));
  b[p]=type?static_cast<const void*>(narrow.back()->p):wide.back()->p;
  engines.push_back(std::make_unique<Kernel>(b[p],n,k,0,type==0?Kernel::Precision::FP32:
   type==1?Kernel::Precision::FP16:Kernel::Precision::BF16,type?std::static_pointer_cast<void>(narrow.back()):std::static_pointer_cast<void>(wide.back())));
  dirty[p].resize(outSize);actual[p].resize(outSize);gold[p].resize(outSize);oc[p]=oracleOutputs.back()->p;
 }
 auto requirements=engines.front()->getWorkspaceRequirements(m,n,k);
 if(mapped)requirements.buffers.push_back({GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT,std::size_t(m)*n*sizeof(float),256,true});
 const auto bytes=requirements.total_bytes_with_alignment();
 PhysicalMemoryPlanBuilder plan;plan.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},PhysicalMemoryOwner::ExecutionWorkspace,bytes);
 auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
 DeviceWorkspaceManager workspace(device,bytes,memory);if(!workspace.allocate(requirements))throw std::runtime_error("Pointer fixture workspace allocation failed");
 Captured candidate,oracle;
 for(int p=0;p<count;++p) {
  if(!inputs[p]->ensureOnDevice(device,candidate.stream)||(!mapped&&!outputs[p]->allocateOnDevice(device,candidate.stream)))
   throw std::runtime_error("Pointer fixture matrix admission failed");
  a[p]=static_cast<const float*>(inputs[p]->gpu_data_ptr());c[p]=static_cast<float*>(outputs[p]->gpu_data_ptr());
  upload(wide[p]->p,hostB[p],candidate.stream);upload(narrow[p]->p,native[p],candidate.stream);
  engines[p]->bindWorkspace(&workspace);engines[p]->setGPUStream(candidate.stream);
 }
 checked(hipMemcpyAsync(live.p,lengths.data(),sizeof(lengths),hipMemcpyHostToDevice,candidate.stream));checked(hipStreamSynchronize(candidate.stream));
 const auto prefix=DeviceRowRange::deviceCounted(m,live.p);
 const auto full=DeviceRowRange::fullyActive(m);
 const auto requests=DeviceRequestRowRanges::deviceCounted(3,holes?m/3:m,live.p);
 std::vector<ITensorGemm::TensorProjectionDesc> projections;
 for(int p=0;p<count;++p)projections.emplace_back(engines[p].get(),outputs[p].get(),n,nullptr,"pointer102");
 auto launchCandidate=[&](hipStream_t stream) {
  (void)stream;
#if QFN_POINTER102_VALUE_ABI
  if(direct) {
   if(count>8)throw std::invalid_argument("Direct descriptor fixture exceeds native cardinality");
   bool ok;
   if(type==0) {
    auto group=FixedFP32ProjectionGroup::single(a[0],static_cast<const float*>(b[0]),c[0]);
    for(int p=1;p<count;++p)group.append(a[p],static_cast<const float*>(b[p]),c[p]);
    ok=rocmFp32_small_n_batched_projection(group,m,n,k,0,stream,&prefix);
   } else {
    auto group=FixedFP32x16ProjectionGroup::single(a[0],static_cast<const std::uint16_t*>(b[0]),c[0]);
    for(int p=1;p<count;++p)group.append(a[p],static_cast<const std::uint16_t*>(b[p]),c[p]);
    ok=rocmFp32x16_batched_projection(group,m,n,k,type-1,0,stream,&prefix);
   }
   if(!ok)throw std::runtime_error("Direct descriptor consumer rejected complete bindings");
   return;
  }
#endif
  if(independent||mapped) {
   ITensorGemm::VerifierKernelModeScope scope(mapped?full:prefix);
   for(int p=0;p<count;++p)if(!engines[p]->multiply_tensor(inputs[p].get(),outputs[p].get(),m,n,k,true,1.f,0.f,nullptr,nullptr,-1,&workspace))
    throw std::runtime_error("Public independent custom projection failed");
  } else {
   auto requestScope=holes?engines[0]->beginFloatingRequestRowsScope(requests):nullptr;
   std::unique_ptr<ITensorGemm::VerifierKernelModeScope> prefixScope;
   if(!holes)prefixScope=std::make_unique<ITensorGemm::VerifierKernelModeScope>(prefix);
   if(!engines[0]->multiply_fused_verifier_rows_decode_equivalent(inputs[0].get(),projections,m,k,nullptr,&workspace))
    throw std::runtime_error("Public fused custom projection failed");
  }
 };
 auto launchOracle=[&](hipStream_t stream) {
  for(int offset=0;offset<count;) {
   const int batch=(independent&&!direct)||mapped?1:std::min(8,count-offset);
   std::array<const float*,8> oa{};std::array<const void*,8> ob{};std::array<float*,8> out{};
   for(int p=0;p<batch;++p){oa[p]=a[independent?offset+p:0];ob[p]=b[offset+p];out[p]=oc[offset+p];}
   if(!qfnPointer102OriginalStaged(oa.data(),ob.data(),out.data(),oldA.p,oldB.p,oldC.p,m,n,k,batch,type,stream,holes?nullptr:(mapped?&full:&prefix),holes?&requests:nullptr))
    throw std::runtime_error("Frozen immediate-original staged projection failed");
   offset+=batch;
  }
 };
 const int copies=profile!=ProfileSide::None?1:timing?(m==1?256:1):1;
 candidate.capture(launchCandidate,copies);oracle.capture(launchOracle,copies);
 if(!timing) {
  const auto chunks=[&](bool singles) {
   std::vector<unsigned> z;
   for(int left=count;left>0;) {const unsigned live=singles?1u:unsigned(std::min(8,left));z.push_back(live);left-=int(live);}
   return z;
  };
  const auto originalZ=chunks((independent&&!direct)||mapped);
  const auto candidateZ=(direct&&QFN_POINTER102_VALUE_ABI)?std::vector<unsigned>{unsigned(count)}:chunks(independent||mapped);
  inspect(candidate,candidateZ,QFN_POINTER102_VALUE_ABI?0:int(candidateZ.size()),n,m,k,false,type,mapped?c[0]:nullptr,
   mapped?workspace.getBuffer(GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT):nullptr);
  inspect(oracle,originalZ,int(originalZ.size()),n,m,k,true,type);
 }
 for(int replay=0;replay<(timing?1:20);++replay) {
  lengths=holes?(replay%5==0?std::array<std::int32_t,3>{m/3,m/3,m/3}:
   replay%5==1?std::array<std::int32_t,3>{0,1,m/3-1}:replay%5==2?std::array<std::int32_t,3>{m/3,0,1}:
   replay%5==3?std::array<std::int32_t,3>{1,m/3,0}:std::array<std::int32_t,3>{0,0,0}):
   std::array<std::int32_t,3>{replay%4==0?m:replay%4==1?std::min(m,1):replay%4==2?0:std::max(0,m-1),0,0};
  if(timing||mapped)lengths={m,m,m};
  checked(hipMemcpyAsync(live.p,lengths.data(),sizeof(lengths),hipMemcpyHostToDevice,candidate.stream));
  for(int p=0;p<count;++p) {
   poisoned[p]=hostA[p];
   for(int row=0;row<m;++row) {
    const bool active=holes?row%(m/3)<lengths[row/(m/3)]:row<lengths[0];
    if(!active)std::fill_n(poisoned[p].data()+std::size_t(row)*k,k,std::numeric_limits<float>::quiet_NaN());
   }
   upload(const_cast<float*>(a[p]),poisoned[p],candidate.stream);
   std::fill(dirty[p].begin(),dirty[p].end(),-987.25f-replay-p);
   upload(c[p],dirty[p],candidate.stream);upload(oc[p],dirty[p],candidate.stream);
  }
  checked(hipStreamSynchronize(candidate.stream));
  checked(hipGraphLaunch(candidate.exec,candidate.stream));checked(hipStreamSynchronize(candidate.stream));
  checked(hipGraphLaunch(oracle.exec,oracle.stream));checked(hipStreamSynchronize(oracle.stream));
  for(int p=0;p<count;++p) {
   checked(hipMemcpyAsync(actual[p].data(),c[p],outSize*4,hipMemcpyDeviceToHost,candidate.stream));
   checked(hipMemcpyAsync(gold[p].data(),oc[p],outSize*4,hipMemcpyDeviceToHost,oracle.stream));
  }
  checked(hipStreamSynchronize(candidate.stream));checked(hipStreamSynchronize(oracle.stream));
  for(int p=0;p<count;++p) {
   ASSERT_EQ(std::memcmp(actual[p].data(),gold[p].data(),outSize*4),0)<<"replay="<<replay<<" projection="<<p;
   for(int row=0;row<=m;++row) {
    const bool active=row<m&&(holes?row%(m/3)<lengths[row/(m/3)]:row<lengths[0]);
    if(!active) {
     ASSERT_EQ(std::memcmp(actual[p].data()+std::size_t(row)*n,dirty[p].data()+std::size_t(row)*n,std::size_t(n)*4),0);
    } else if(pattern==Witness::OrderedDotOne) {
     for(int column=0;column<n;++column)ASSERT_EQ(gold[p][std::size_t(row)*n+column],1.f);
    }
   }
  }
 }
 if(profile!=ProfileSide::None) {
  // Both sides have executed once above for byte proof. Exactly five selected
  // warm replays make occurrence seven the sole diagnostic sample. Each graph
  // has one copy, so the same ordinal applies to M1/M512 and original staging.
  auto &selected=profile==ProfileSide::Candidate?candidate:oracle;
  Events events;for(int warm=0;warm<5;++warm)(void)events.sample(selected,1);
  const float elapsed=events.sample(selected,1);
  std::cout<<"POINTER102_PROFILE,"<<QFN_POINTER102_VALUE_ABI<<','<<type<<','<<m<<','<<n<<','<<k
   <<','<<(profile==ProfileSide::Candidate?"candidate":"original")<<",7,"<<elapsed<<'\n';
  return;
 }
 if(timing) {
  Events events;for(int warm=0;warm<5;++warm){(void)events.sample(candidate,copies);(void)events.sample(oracle,copies);}
  for(int pair=0;pair<12;++pair) {
   float now,old;if(pair&1){old=events.sample(oracle,copies);now=events.sample(candidate,copies);}
   else{now=events.sample(candidate,copies);old=events.sample(oracle,copies);}
   std::cout<<"POINTER102_PIPELINE_PAIR,"<<QFN_POINTER102_VALUE_ABI<<','<<type<<','<<m<<','<<n<<','<<k<<','<<count<<','<<pair<<','<<now<<','<<old<<'\n';
  }
 }
}
/** @brief Enter the existing initialized worker context before constructing prepared engines. */
template<class F> void onWorker(F work) {
 auto &context=GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));context.submitAndWait(work);
}
} // namespace qfn_pointer102_regression
/** @test Calibrate unchanged public prepared APIs, then certify the same captures after descriptor102. */
TEST(ROCmFloatingPointer,ImmediateOriginalStagedPipelineCapturedByteExact) {
 using namespace qfn_pointer102_regression;
 onWorker([] {
  for(int type:{0,1,2})for(int count=1;count<=8;++count) {
   run(type,3,5,257,count);
   run(type,3,5,257,count,{.independent=true});
   run(type,9,5,257,count,{.holes=true});
   run(type,3,5,257,count,{.independent=true,.direct=true});
  }
  for(int type:{0,1,2}) {
   for(int count=1;count<=8;++count)for(auto pattern:{Witness::SourceNaNs,Witness::ActivationNaNs,Witness::DualNaNs}) {
    run(type,1,5,257,count,{.witness=pattern});
    run(type,1,5,257,count,{.independent=true,.witness=pattern});
    run(type,1,5,257,count,{.independent=true,.witness=pattern,.direct=true});
   }
   run(type,9,5,257,19,{.holes=true});
   run(type,3,5,257,1,{.mapped=true});
   for(auto pattern:{Witness::SourceNaNs,Witness::ActivationNaNs,Witness::SignedZeros,Witness::SubnormalInputs,Witness::NativeSpecialWords})
    run(type,3,17,257,2,{.witness=pattern});
   for(auto shape:{std::pair{4,1},std::pair{33,31},std::pair{4,256},std::pair{5,255}})run(type,1,shape.first,shape.second,2);
  }
  for(int m:{1,512})for(auto shape:{std::array{0,4,10240},std::array{2,128,2560},std::array{2,512,2560}})
   run(shape[0],m,shape[1],shape[2],1);
 });
}
/** @test FP32 grouped tile/tail chains and special operands remain original through dirty captured replay. */
TEST(ROCmFloatingPointer,FP32LoadAheadGroupedOriginalChainCapturedByteExact) {
 using namespace qfn_pointer102_regression;
 onWorker([] {
  for(int k:{1023,1024,1025,2047,2048,2049,10240})for(int m:{1,3})
   run(0,m,4,k,2,{.witness=Witness::OrderedDotOne});
  // K clamping and each CTA-selected scheduling region use immutable geometry.
  for(int k:{1023,1024,2047,2048})for(int m:{1,17,65})
   for(auto witness:{Witness::Finite,Witness::OrderedDotOne})
    run(0,m,4,k,1,{.witness=witness});
  // Both exact CTA boundaries and nearby M/N factorizations execute actual
  // templated kernels. All witnesses retain the fixture's 20 dirty histories.
  for(auto shape:{std::array{1,64,1},std::array{1,65,1},std::array{1,256,1},std::array{1,257,1},
                 std::array{3,21,1},std::array{3,22,1},std::array{3,85,1},std::array{3,86,1},
                 std::array{1,8,8},std::array{1,13,5},std::array{1,32,8},std::array{1,37,7},
                 std::array{3,8,17},std::array{3,11,9}})
   for(auto witness:{Witness::Finite,Witness::OrderedDotOne,Witness::SourceNaNs,
                    Witness::ActivationNaNs,Witness::DualNaNs})
    run(0,shape[0],shape[1],4097,shape[2],{.witness=witness});
  // A request-hole capture whose first eight-projection chunk selects scalar
  // and final one-projection chunk selects eight loads retains those modes
  // through full, partial, empty and large-to-small live-count publications.
  run(0,3,11,4097,9,{.holes=true,.witness=Witness::OrderedDotOne});
  // Existing companion gates retain the full cardinality and native16 sweep.
  // These cases add sensitivity at descriptor capacity and request/chunk edges.
  run(0,3,4,1025,8,{.independent=true,.witness=Witness::OrderedDotOne,.direct=true});
  run(0,9,4,2049,19,{.holes=true,.witness=Witness::OrderedDotOne});
  for(int k:{1025,2049})
   for(auto witness:{Witness::SourceNaNs,Witness::ActivationNaNs,Witness::DualNaNs,
                    Witness::SignedZeros,Witness::SubnormalInputs,Witness::NativeSpecialWords})
    run(0,3,17,k,2,{.witness=witness});
 });
}
/** @test Native events compare entire staged-original versus actual public candidate captured pipelines. */
TEST(ROCmFloatingPointer,DISABLED_ExactShapePairedPipelineEvents) {
 using namespace qfn_pointer102_regression;
 const auto selection=profileSelection();
 onWorker([&] {
  if(selection) {
   run(selection->type,selection->m,selection->n,selection->k,1,{.timing=true,.profile=selection->side});
   return;
  }
  // HC injection K is4H=10240. N4/K2560 is a separately requested diagnostic.
  for(int m:{1,512})for(auto shape:{std::array{0,4,10240},std::array{0,4,2560},std::array{2,128,2560},std::array{2,512,2560}})
   run(shape[0],m,shape[1],shape[2],1,{.timing=true});
 });
}
namespace qfn_pointer102_regression {
/** @brief Original real table consumers remain ordered between custom consumers in one graph. */
void mixedTableConsumers() {
 constexpr int m=33,k=256;const auto device=DeviceId::rocm(0);
 std::vector<float> host(std::size_t(m)*k);for(std::size_t i=0;i<host.size();++i)host[i]=sample(i,71);
 std::vector<float> weight(std::size_t(128)*k);for(std::size_t i=0;i<weight.size();++i)weight[i]=sample(i,91);
 std::vector<std::uint16_t> half(std::size_t(4)*k,0x3800);
 std::vector<std::vector<float>> dirty(5),got(5),gold(5);
 FP32Tensor input({m,k});std::copy(host.begin(),host.end(),input.mutable_data());
 auto w=std::make_shared<Allocation<float>>(weight.size());auto h=std::make_shared<Allocation<std::uint16_t>>(half.size());
 std::vector<std::unique_ptr<Kernel>> engines;
 engines.push_back(std::make_unique<Kernel>(w->p,4,k,0,Kernel::Precision::FP32,w));
 for(int p=0;p<2;++p)engines.push_back(std::make_unique<Kernel>(w->p,128,k,0,Kernel::Precision::FP32,w));
 for(int p=0;p<2;++p)engines.push_back(std::make_unique<Kernel>(h->p,4,k,0,Kernel::Precision::FP16,h,FloatingActivationOperandPrecision::FP16RNE));
 std::vector<std::unique_ptr<FP32Tensor>> out,ref;
 WorkspaceRequirements requirements;
 for(int p=0;p<5;++p) {
  const int n=p==1||p==2?128:4;requirements.merge(engines[p]->getWorkspaceRequirements(m,n,k));
  out.push_back(std::make_unique<FP32Tensor>(std::vector<std::size_t>{m+1,std::size_t(n)}));
  ref.push_back(std::make_unique<FP32Tensor>(std::vector<std::size_t>{m+1,std::size_t(n)}));
  dirty[p].resize(std::size_t(m+1)*n,-911.75f-p);got[p].resize(dirty[p].size());gold[p].resize(dirty[p].size());
 }
 const auto bytes=requirements.total_bytes_with_alignment();PhysicalMemoryPlanBuilder plan;
 plan.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},PhysicalMemoryOwner::ExecutionWorkspace,bytes);
 auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
 DeviceWorkspaceManager workspace(device,bytes,memory);if(!workspace.allocate(requirements))throw std::runtime_error("Mixed table workspace admission failed");
 Captured candidate,reference;
 if(!input.ensureOnDevice(device,candidate.stream))throw std::runtime_error("Mixed table input admission failed");
 upload(w->p,weight,candidate.stream);upload(h->p,half,candidate.stream);
 for(int p=0;p<5;++p) {
  if(!out[p]->allocateOnDevice(device,candidate.stream)||!ref[p]->allocateOnDevice(device,candidate.stream))throw std::runtime_error("Mixed table output admission failed");
  engines[p]->bindWorkspace(&workspace);
 }
 checked(hipStreamSynchronize(candidate.stream));
 auto launch=[&](std::vector<std::unique_ptr<FP32Tensor>>& outputs,hipStream_t stream) {
  for(auto &engine:engines)engine->setGPUStream(stream);
  if(!engines[0]->multiply_tensor(&input,outputs[0].get(),m,4,k,true,1.f,0.f,nullptr,nullptr,-1,&workspace))throw std::runtime_error("Mixed custom projection failed");
  std::vector<ITensorGemm::TensorProjectionDesc> vendor,physical;
  for(int p=1;p<3;++p)vendor.emplace_back(engines[p].get(),outputs[p].get(),128,nullptr,"actual_blas_table_neighbor");
  for(int p=3;p<5;++p)physical.emplace_back(engines[p].get(),outputs[p].get(),4,nullptr,"actual_half_table_neighbor");
  // Ordinary M33/N128 FP32 grouped route is genuinely table-backed hipBLAS.
  if(!engines[1]->multiply_fused_tensor(&input,vendor,m,k,nullptr,&workspace))throw std::runtime_error("Actual grouped BLAS neighbor failed");
  // M33 exceeds the short-half group domain31 and retains half table consumers.
  if(!engines[3]->multiply_fused_verifier_rows_decode_equivalent(&input,physical,m,k,nullptr,&workspace))throw std::runtime_error("Actual physical-half table neighbor failed");
 };
 candidate.capture([&](hipStream_t stream){launch(out,stream);});
 reference.capture([&](hipStream_t stream){launch(ref,stream);});
 std::size_t count=0;checked(hipGraphGetNodes(candidate.graph,nullptr,&count));std::vector<hipGraphNode_t> nodes(count);checked(hipGraphGetNodes(candidate.graph,nodes.data(),&count));
 int publications=0;
 for(auto node:nodes) {
  hipGraphNodeType type;checked(hipGraphNodeGetType(node,&type));ASSERT_EQ(type,hipGraphNodeTypeKernel);
  hipKernelNodeParams params{};checked(hipGraphKernelNodeGetParams(node,&params));
  if(params.blockDim.x==8) {
   const char *raw=hipKernelNameRefByPtr(params.func,candidate.stream);
   ASSERT_NE(raw,nullptr);const std::string name(raw);
   ASSERT_NE(name.find("fp32_stage_batched_projection_pointers_kernel"),std::string::npos)<<name;
   EXPECT_EQ(name.find("pointer102Original_"),std::string::npos);
   EXPECT_EQ(params.blockDim.y,1u);EXPECT_EQ(params.blockDim.z,1u);
   EXPECT_EQ(params.gridDim.x,1u);EXPECT_EQ(params.gridDim.y,1u);EXPECT_EQ(params.gridDim.z,1u);
   ++publications;
  }
 }
 ASSERT_EQ(publications,QFN_POINTER102_VALUE_ABI?2:3);
 for(int replay=0;replay<20;++replay) {
  for(int p=0;p<5;++p){upload(static_cast<float*>(out[p]->gpu_data_ptr()),dirty[p],candidate.stream);upload(static_cast<float*>(ref[p]->gpu_data_ptr()),dirty[p],candidate.stream);}
  checked(hipStreamSynchronize(candidate.stream));checked(hipGraphLaunch(reference.exec,reference.stream));checked(hipStreamSynchronize(reference.stream));
  checked(hipGraphLaunch(candidate.exec,candidate.stream));checked(hipStreamSynchronize(candidate.stream));
  for(int p=0;p<5;++p) {
   checked(hipMemcpyAsync(got[p].data(),out[p]->gpu_data_ptr(),got[p].size()*4,hipMemcpyDeviceToHost,candidate.stream));
   checked(hipMemcpyAsync(gold[p].data(),ref[p]->gpu_data_ptr(),gold[p].size()*4,hipMemcpyDeviceToHost,candidate.stream));
  }
  checked(hipStreamSynchronize(candidate.stream));
  for(int p=0;p<5;++p){ASSERT_EQ(std::memcmp(got[p].data(),gold[p].data(),got[p].size()*4),0);const int n=p==1||p==2?128:4;
   ASSERT_EQ(std::memcmp(got[p].data()+std::size_t(m)*n,dirty[p].data()+std::size_t(m)*n,std::size_t(n)*4),0);}
 }
}
}
/** @test Real BLAS/physical-half device pointer tables survive neighboring custom work. */
TEST(ROCmFloatingPointer,MixedActualTableConsumersRemainCaptured) {
 qfn_pointer102_regression::onWorker([] {qfn_pointer102_regression::mixedTableConsumers();});
}
