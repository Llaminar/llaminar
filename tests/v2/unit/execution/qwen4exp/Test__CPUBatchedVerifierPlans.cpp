/**
 * @file Test__CPUBatchedVerifierPlans.cpp
 * @brief Prove compact GDN and short-conv owners join atomic CPU publication.
 *
 * Real grouped CPU kernels materialize their sole request banks. Stages borrow
 * those native addresses and stage-specific capture workspaces for planning.
 * GDN publishes the full checkpoint/history encoding while convolution copies
 * its native snapshots. Independent serial GDN calls are only the test oracle;
 * publication itself performs no recurrence or model execution.
 */
#include "backends/BackendManager.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "kernels/cpu/gdn/CPUShortConvolution.h"
#include "fixtures/CPUGDNStateTestUtils.h"
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>
using namespace llaminar2;

/** @test Mixed compact/copy plans validate every request before either stage writes. */
TEST(CPUBatchedVerifierPlans, CompactGDNAndConvolutionBanksUseValidatedPlans)
{
    llaminar::v2::ThreadCountGuard threads(1);
    if(!hasCPUBackend())initCPUBackend(1);
    constexpr int requests=3,stride=3,rows=requests*stride,elements=4;
    const CPUGDNStateLayout layout(1,2,2);
    const std::vector<int> live{3,0,2};
    CPUGatedDeltaNet gdn;CPUShortConvolution convolution;
    auto gdnPublic=encodeCPUGDNTestState({1,2,3,4},1,2,2);
    std::array<float,elements> convPublic{5,6,7,8};
    const auto gdnInitial=gdnPublic;
    GDNRecurrenceStage::Params gp;
    gp.device_id=DeviceId::cpu();gp.seq_len=rows;gp.request_count=requests;gp.request_seq_len=stride;
    gp.request_seq_lens_host=&live;gp.n_heads=1;gp.n_k_heads=1;gp.d_k=2;gp.d_v=2;
    gp.layer_idx=0;gp.verifier_state_capture_rows=rows;gp.recurrence_state=gdnPublic.data();gp.kernel=&gdn;
    ShortConv1dStage::Params cp;
    cp.device_id=DeviceId::cpu();cp.seq_len=rows;cp.request_count=requests;cp.request_seq_len=stride;
    cp.request_seq_lens_host=&live;cp.channels=4;cp.kernel_size=2;cp.layer_idx=1;
    cp.verifier_state_capture_rows=rows;cp.conv_state=convPublic.data();cp.kernel=&convolution;
    // Workspace ownership outlives each stage's retained binding on every exit.
    std::unique_ptr<DeviceWorkspaceManager> gw,cw;
    GDNRecurrenceStage gdnStage(gp);ShortConv1dStage convStage(cp);
    auto gr=gdnStage.getWorkspaceRequirements(rows),cr=convStage.getWorkspaceRequirements(rows);
    gw=std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(),gr.total_bytes_with_alignment()+1024);
    cw=std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(),cr.total_bytes_with_alignment()+1024);
    ASSERT_TRUE(gw->allocate(gr));ASSERT_TRUE(cw->allocate(cr));
    gdnStage.bindWorkspace(gw.get());convStage.bindWorkspace(cw.get());
    std::vector<float> q(rows*2,0.0f),v(rows*2,0.0f),alpha(rows,0.0f),beta(rows,0.0f),gout(rows*2);
    for(size_t i=0;i<q.size();++i) {
        q[i]=static_cast<float>(static_cast<int>(i%7)-3)/16.0f;
        v[i]=static_cast<float>(static_cast<int>(i%5)-2)/32.0f;
    }
    float aLog=0.0f,bias=0.0f;
    ASSERT_TRUE(gdn.chunkForwardBatchedRequestsWithHostSeqLens(q.data(),q.data(),v.data(),alpha.data(),beta.data(),
        &aLog,&bias,gout.data(),gdnPublic.data(),rows,requests,stride,1,2,2,64,true,live.data()));
    std::vector<float> input(rows*4,0.0f),weight(4*2,1.0f),cout(rows*4);
    ASSERT_TRUE(convolution.forwardBatchedRequestsWithHostSeqLens(input.data(),weight.data(),nullptr,
        cout.data(),convPublic.data(),rows,requests,stride,4,2,live.data(),false));
    auto gbank=gdn.cpuRequestStateBankView(),cbank=convolution.cpuRequestStateBankView();
    ASSERT_TRUE(gbank.contains(requests,layout.floats()));ASSERT_TRUE(cbank.contains(requests,elements));
    float *gslots=nullptr,*cslots=nullptr;
    for(const auto &b:gr.buffers)if(b.name.find("speculative_state_slots")!=std::string::npos)
        gslots=static_cast<float *>(gw->getBuffer(b.name));
    for(const auto &b:cr.buffers)if(b.name.find("speculative_state_slots")!=std::string::npos)
        cslots=static_cast<float *>(cw->getBuffer(b.name));
    ASSERT_NE(gslots,nullptr);ASSERT_NE(cslots,nullptr);
    const size_t gslotCount=requests*CPUSharedGDNSnapshots::headFloats(layout,stride);
    const std::vector<float> gslotsOriginal(gslots,gslots+gslotCount),
        cslotsOriginal(cslots,cslots+rows*elements);
    const auto gpOriginal=gdnPublic;
    const auto cpOriginal=convPublic;
    const std::vector<float> gbOriginal(gbank.storage.begin(),gbank.storage.end()),
        cbOriginal(cbank.storage.begin(),cbank.storage.end());
    // Serial decode independently supplies complete accepted checkpoint/history
    // bytes; an empty request preserves its original native bank.
    std::array<std::vector<float>,requests> expected;
    const std::array<int,requests> accepted{1,0,2};
    for(int request=0;request<requests;++request) {
        expected[request]=request==0?gdnInitial:std::vector<float>(layout.floats(),0.0f);
        CPUGatedDeltaNet serial;
        std::array<float,2> output;
        for(int row=0;row<accepted[request];++row) {
            const int flat=request*stride+row;
            ASSERT_TRUE(serial.recurrent_step(q.data()+flat*2,q.data()+flat*2,v.data()+flat*2,
                alpha.data()+flat,beta.data()+flat,&aLog,&bias,output.data(),expected[request].data(),1,2,2,true));
        }
    }
    MTPSpecStepPlanBatch batch;batch.ok=true;batch.request_count=requests;
    for(int r=0;r<requests;++r)batch.steps.push_back({.request_index=r,.draft_count=2,.target_rows=stride,.accepted_count=accepted[r]});
    const int invalid[requests]{0,-1,8};
    auto result=publishAcceptedMTPSpecStateFromVerifierRows(batch,invalid,{&gdnStage,&convStage},DeviceId::cpu(),nullptr,true);
    EXPECT_FALSE(result.ok);EXPECT_EQ(gdnPublic,gpOriginal);EXPECT_EQ(convPublic,cpOriginal);
    EXPECT_TRUE(std::equal(gbank.storage.begin(),gbank.storage.end(),gbOriginal.begin()));
    EXPECT_TRUE(std::equal(cbank.storage.begin(),cbank.storage.end(),cbOriginal.begin()));
    EXPECT_EQ(std::memcmp(gslots,gslotsOriginal.data(),gslotCount*sizeof(float)),0);
    EXPECT_EQ(std::memcmp(cslots,cslotsOriginal.data(),rows*elements*sizeof(float)),0);
    const int valid[requests]{0,-1,7};
    result=publishAcceptedMTPSpecStateFromVerifierRows(batch,valid,{&gdnStage,&convStage},DeviceId::cpu(),nullptr,true);
    ASSERT_TRUE(result.ok)<<result.error;
    EXPECT_EQ(std::memcmp(gdnPublic.data(),expected[0].data(),layout.floats()*sizeof(float)),0);
    for(int request=0;request<requests;++request)
        EXPECT_EQ(std::memcmp(gbank.storage.data()+request*layout.floats(),expected[request].data(),
            layout.floats()*sizeof(float)),0)<<"request "<<request;
    for(int d=0;d<elements;++d) {
        EXPECT_EQ(convPublic[d],cslots[d]);
        EXPECT_EQ(cbank.storage[d],cslots[d]);
        EXPECT_EQ(cbank.storage[elements+d],cbOriginal[elements+d]);
        EXPECT_EQ(cbank.storage[2*elements+d],cslots[7*elements+d]);
    }
    EXPECT_EQ(std::memcmp(gslots,gslotsOriginal.data(),gslotCount*sizeof(float)),0);
    EXPECT_EQ(std::memcmp(cslots,cslotsOriginal.data(),rows*elements*sizeof(float)),0);
}
