/**
 * @file Test__FloatingRequestRows.cpp
 * @brief Device-free proof of canonical request selection, scope isolation and exact skinny extents.
 */
#include "kernels/common/FloatingProjectionRows.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include <gtest/gtest.h>
#include <array>
#include <barrier>
#include <cstring>
#include <exception>
#include <limits>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
using namespace llaminar2;
namespace {
/** @brief Test engine that opts into recording metadata without launching native work. */
struct ScopeEngine final:ITensorGemm {
    bool supports_device(int) const override{return false;}
    bool multiply_tensor(const TensorBase*,TensorBase*,int,int,int,bool,float,float,const TensorBase*,
        const IMPIContext*,int,DeviceWorkspaceManager*,int) override{return false;}
    std::unique_ptr<VerifierKernelModeScope> beginFloatingRequestRowsScope(DeviceRequestRowRanges rows) override
    {return makeFloatingRequestRowsScope(rows);}
};
/** @brief Preadmitted host pages with one activation row per page; inactive reads fault. */
class Rows {
public:
    explicit Rows(int count):count_(count),page_(::sysconf(_SC_PAGESIZE)),bytes_(count*page_) {
        data_=static_cast<float *>(::mmap(nullptr,bytes_,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
        if(data_==MAP_FAILED)throw std::runtime_error("mmap failed");
    }
    ~Rows(){::munmap(data_,bytes_);}
    float *data(){return data_;}
    int width()const{return page_/sizeof(float);}
    void protect(int row,bool hidden){if(::mprotect(reinterpret_cast<char *>(data_)+row*page_,page_,hidden?PROT_NONE:PROT_READ|PROT_WRITE))throw std::runtime_error("mprotect failed");}
private:int count_,page_;std::size_t bytes_;float *data_;
};
}
TEST(FloatingRequestRows, CanonicalAdmissionRejectsOverlapAndPreservesHoles) {
    std::array<int,3> live{0,5,2};const auto requests=DeviceRequestRowRanges::deviceCounted(3,5,live.data());
    const auto selected=FloatingProjectionRows::bind(15,nullptr,&requests);
    const auto prefix=DeviceRowRange::fullyActive(15);
    EXPECT_THROW(FloatingProjectionRows::bind(15,&prefix,&requests),std::invalid_argument);
    EXPECT_THROW(FloatingProjectionRows::bind(14,nullptr,&requests),std::invalid_argument);
    for(int row=0;row<15;++row)EXPECT_EQ(selected.hostRowIsActive(row),row%5<live[row/5]);
    EXPECT_EQ(selected.hostTiles(4),6);
    const std::array<int,6> first{0,4,5,9,10,14},count{0,0,4,1,2,0};
    for(int tile=0;tile<6;++tile){EXPECT_EQ(selected.hostTileFirst(tile,4),first[tile]);EXPECT_EQ(selected.hostTileRows(tile,4),count[tile]);}
    live[2]=6;EXPECT_THROW(selected.validateHostPublications(),std::invalid_argument);
}
TEST(FloatingRequestRows, NestedExceptionsRestoreScopeAndRejectPrefixOverlap) {
    ScopeEngine engine;std::array<int,3> a{1,0,2};std::array<int,2>b{0,1};
    const auto outer=DeviceRequestRowRanges::deviceCounted(3,5,a.data());
    const auto inner=DeviceRequestRowRanges::deviceCounted(2,3,b.data());
    EXPECT_EQ(ITensorGemm::VerifierKernelModeScope::requestsFor(15),nullptr);
    {
        auto scope=engine.beginFloatingRequestRowsScope(outer);
        EXPECT_EQ(ITensorGemm::VerifierKernelModeScope::requestsFor(15)->lengthOwner(),a.data());
        EXPECT_THROW(engine.beginVerifierDecodeEquivalentScope(DeviceRowRange::fullyActive(15)),std::invalid_argument);
        EXPECT_THROW((void)ITensorGemm::VerifierKernelModeScope::rowsFor(15),std::logic_error);
        try {auto nested=engine.beginFloatingRequestRowsScope(inner);throw std::runtime_error("test");}catch(const std::runtime_error&){}
        EXPECT_EQ(ITensorGemm::VerifierKernelModeScope::requestsFor(15)->lengthOwner(),a.data());
        EXPECT_THROW((void)ITensorGemm::VerifierKernelModeScope::requestsFor(14),std::logic_error);
    }
    EXPECT_EQ(ITensorGemm::VerifierKernelModeScope::requestsFor(15),nullptr);
    auto prefix=engine.beginVerifierDecodeEquivalentScope(DeviceRowRange::fullyActive(15));
    EXPECT_THROW(engine.beginFloatingRequestRowsScope(outer),std::invalid_argument);
    EXPECT_EQ(ITensorGemm::VerifierKernelModeScope::rowsFor(15)->physicalRows(),15);
}
TEST(FloatingRequestRows, RecordingThreadsRetainIndependentImmutableOwners) {
    std::barrier barrier(2);std::array<bool,2> valid{};ScopeEngine engine;
    std::thread a([&]{int length[2]{0,3};auto rows=DeviceRequestRowRanges::deviceCounted(2,3,length);auto scope=engine.beginFloatingRequestRowsScope(rows);barrier.arrive_and_wait();valid[0]=ITensorGemm::VerifierKernelModeScope::requestsFor(6)->lengthOwner()==length;});
    std::thread b([&]{int length[3]{1,0,1};auto rows=DeviceRequestRowRanges::deviceCounted(3,2,length);auto scope=engine.beginFloatingRequestRowsScope(rows);barrier.arrive_and_wait();valid[1]=ITensorGemm::VerifierKernelModeScope::requestsFor(6)->lengthOwner()==length;});
    a.join();b.join();EXPECT_TRUE(valid[0]);EXPECT_TRUE(valid[1]);
}
TEST(FloatingRequestRows, NativeFloatingFamiliesSkipProtectedInactiveInputPagesAcrossDirtyReplays) {
    constexpr int requests=3,stride=5,m=requests*stride,n=7;Rows input(m);const int k=input.width();
    std::vector<float> weights(n*k),output(m*n,-91),serial(m*n),before;
    std::vector<std::uint16_t> bf16(n*k),fp16(n*k);
    for(int row=0;row<m;++row)for(int d=0;d<k;++d)input.data()[row*k+d]=float((row*719+d*37)%1999-999)/997.0f;
    for(int i=0;i<n*k;++i){weights[i]=float((i*713)%1999-999)/991.0f;bf16[i]=simd::fp32_to_bf16(weights[i]);fp16[i]=fp32_to_fp16(weights[i]);}
    for(int format:{0,1,2}) {
        for(int row=0;row<m;++row) {
            if(format==1)ASSERT_TRUE(gemm::run_fp32xbf16_skinny_matmul(input.data()+row*k,bf16.data(),serial.data()+row*n,1,n,k,true));
            else if(format==2)ASSERT_TRUE(gemm::run_fp32xfp16_skinny_matmul(input.data()+row*k,fp16.data(),serial.data()+row*n,1,n,k,true));
            else ASSERT_TRUE(gemm::run_fp32_skinny_matmul(input.data()+row*k,weights.data(),serial.data()+row*n,1,n,k,true));
        }
        std::array<int,requests> live{};const auto rows=DeviceRequestRowRanges::deviceCounted(requests,stride,live.data());
        const auto selected=FloatingProjectionRows::bind(m,nullptr,&rows);
        for(const auto counts:{std::array<int,requests>{5,5,5},{1,0,2},{0,0,0},{0,4,1},{5,1,0},{0,0,0}}) {
            live=counts;before=output;
            for(int row=0;row<m;++row)input.protect(row,row%stride>=live[row/stride]);
            if(format==1)ASSERT_TRUE(gemm::run_fp32xbf16_skinny_matmul(input.data(),bf16.data(),output.data(),m,n,k,true,1,0,&selected));
            else if(format==2)ASSERT_TRUE(gemm::run_fp32xfp16_skinny_matmul(input.data(),fp16.data(),output.data(),m,n,k,true,1,0,&selected));
            else ASSERT_TRUE(gemm::run_fp32_skinny_matmul(input.data(),weights.data(),output.data(),m,n,k,true,1,0,nullptr,&selected));
            for(int row=0;row<m;++row){input.protect(row,false);const float*expected=row%stride<live[row/stride]?serial.data():before.data();EXPECT_EQ(std::memcmp(output.data()+row*n,expected+row*n,n*sizeof(float)),0);}
        }
        live={1,2,6};before=output;
        EXPECT_THROW(gemm::run_fp32_skinny_matmul(input.data(),weights.data(),output.data(),m,n,k,true,1,0,nullptr,&selected),std::invalid_argument);
        EXPECT_EQ(std::memcmp(output.data(),before.data(),output.size()*sizeof(float)),0);
    }
}

/** @test All-empty replay touches neither weight data nor any physical input/output page. */
TEST(FloatingRequestRows, AllEmptyTouchesNoProtectedOperandOrOutputPage) {
    Rows input(15),weights(7),output(15);const int k=input.width(),n=7;
    int counts[3]{0,0,0};const auto requests=DeviceRequestRowRanges::deviceCounted(3,5,counts);
    const auto rows=FloatingProjectionRows::bind(15,nullptr,&requests);
    for(int row=0;row<15;++row){input.protect(row,true);output.protect(row,true);}
    for(int row=0;row<7;++row)weights.protect(row,true);
    ASSERT_TRUE(gemm::run_fp32_skinny_matmul(input.data(),weights.data(),output.data(),15,n,k,true,1,0,nullptr,&rows));
    ASSERT_TRUE(gemm::run_fp32xbf16_skinny_matmul(input.data(),reinterpret_cast<const std::uint16_t*>(weights.data()),output.data(),15,n,k,true,1,0,&rows));
    ASSERT_TRUE(gemm::run_fp32xfp16_skinny_matmul(input.data(),reinterpret_cast<const std::uint16_t*>(weights.data()),output.data(),15,n,k,true,1,0,&rows));
}

TEST(FloatingRequestRows, RecordingIdentityTracksGeometryAndOwnerButNeverMutableLengths) {
    int a[5]{1,0,2,0,0},b[5]{1,0,2,0,0};
    const auto rows=DeviceRequestRowRanges::deviceCounted(3,5,a);
    const auto original=FloatingProjectionRows::requestRecordingIdentity(rows);
    a[0]=5;a[1]=5;a[2]=0;
    EXPECT_EQ(original,FloatingProjectionRows::requestRecordingIdentity(rows));
    EXPECT_NE(original,FloatingProjectionRows::requestRecordingIdentity(DeviceRequestRowRanges::deviceCounted(3,5,b)));
    EXPECT_NE(original,FloatingProjectionRows::requestRecordingIdentity(DeviceRequestRowRanges::deviceCounted(5,3,a)));
    EXPECT_NE(original,FloatingProjectionRows::requestRecordingIdentity(DeviceRequestRowRanges::fullyActiveRequests(3,5)));
    static_assert(!std::is_convertible_v<FloatingProjectionRows,DeviceRowRange>);
}
