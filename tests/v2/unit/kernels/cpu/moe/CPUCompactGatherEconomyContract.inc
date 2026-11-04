/**
 * @file CPUCompactGatherEconomyContract.inc
 * @brief Actual compact pack/sparse combine/fixed fold under a reusable padded capacity.
 *
 * Pure source contracts grant no transport, ticket or native CPU readiness.
 * The separate original two-socket stage case validates actual gather traffic.
 */
#include "execution/moe/MoEOverlayCanonicalHostReturn.h"

/** @test Live9/0/1 rows use original sparse slots, compact prefixes and fixed source order. */
TEST(CPUShardRouteCollective, Padded64CompactGatherAndReducerEconomyContract)
{
    constexpr int capacity=64,K=4,H=7,P=3;
    constexpr size_t slots=capacity*K,recordWidth=H+1;
    constexpr float sentinel=-991.0f;
    std::array<std::vector<float>,P> packed;
    for(auto& bank:packed)bank.assign(slots*recordWidth+1,sentinel);
    std::vector<float> receive(P*slots*recordWidth,sentinel),combined(slots*H,sentinel),output(capacity*H,sentinel);
    std::vector<int32_t> maps(P*slots,-31);
    std::vector<float> required(slots,-41),denseWeights(slots,0);
    const std::array<int,2> selected{64,511};
    const auto value=[](int participant,int slot,int column) {
        if(slot%K==0)return participant==0?1e20f:participant==1?-1e20f:float(7+column);
        return float((participant+1)*3+column);
    };
    for(const int live:{9,0,1,9})for(const bool reverse:{false,true}) {
        SCOPED_TRACE(live);
        SCOPED_TRACE(reverse);
        std::vector<int32_t> original,experts;
        std::vector<float> weights;
        std::fill(denseWeights.begin(),denseWeights.end(),0);
        for(int row=0;row<live;++row)for(int route:{0,2}) {
            const int slot=row*K+route;original.push_back(slot);experts.push_back(route==0?64:511);
            weights.push_back(route==0?.5f:.75f);denseWeights[slot]=weights.back();
        }
        const size_t count=original.size();
        std::array<RootedFloatRecordRange,P> ranges;
        std::fill(receive.begin(),receive.end(),sentinel);
        std::fill(combined.begin(),combined.end(),sentinel);std::fill(output.begin(),output.end(),sentinel);
        size_t receiveCount=0;
        for(int arrival=0;arrival<P;++arrival) {
            const int participant=reverse?P-1-arrival:arrival;
            std::vector<float> rows(slots*H,sentinel);std::vector<int32_t> rowIDs(slots,-1);
            for(size_t entry=0;entry<count;++entry) {
                const size_t coordinate=reverse?count-1-entry:entry;rowIDs[entry]=original[coordinate];
                for(int h=0;h<H;++h)rows[entry*H+h]=value(participant,rowIDs[entry],h);
            }
            std::fill(packed[participant].begin(),packed[participant].end(),sentinel);
            MoEOverlayReturnRows packet;packet.residency_epoch=17;packet.d_model=H;
            packet.layout=MoEOverlayReturnLayout::CanonicalExpertRoutes;packet.live_row_count=count;
            packet.row_capacity=slots;packet.row_ids_host=rowIDs.data();packet.output_rows_fp32=rows.data();
            ASSERT_TRUE(gatherMoEOverlayCanonicalHostReturn(packet,packed[participant],MoEOverlayCanonicalGatherBoundary::Begin));
            ASSERT_EQ(canonical_moe_route_record::readRecordCount(packed[participant].data(),packed[participant].size()),count);
            EXPECT_EQ(compactMoEOverlayReturnBytes(packet),count*(sizeof(int32_t)+H*sizeof(float)));
            EXPECT_TRUE(std::all_of(packed[participant].begin()+count*recordWidth,packed[participant].end()-1,
                [](float v){return v==sentinel;}));
            // This is the transport's exact completed prefix projection, not a native send claim.
            std::copy_n(packed[participant].data(),count*recordWidth,receive.data()+receiveCount*recordWidth);
            ranges[participant]={participant,receiveCount,count};receiveCount+=count;
        }
        EXPECT_EQ(receiveCount*recordWidth*sizeof(float),P*count*(H+1)*sizeof(float));
        EXPECT_TRUE(std::all_of(receive.begin()+receiveCount*recordWidth,receive.end(),[](float v){return v==sentinel;}));
        CanonicalShardRouteRecordArgs args{std::span(receive).first(receiveCount*recordWidth),ranges,{},{},selected,
            maps,required,combined,512,live,K,H,CanonicalShardSparseRouteView{original,experts,weights}};
        ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
        if(live)ASSERT_TRUE(reduceCanonicalRouteRowsOrderedFMA({combined.data(),denseWeights.data(),nullptr,output.data(),live,K,H}));
        else EXPECT_FALSE(reduceCanonicalRouteRowsOrderedFMA({combined.data(),denseWeights.data(),nullptr,output.data(),live,K,H})); // Dense fold has no zero-row launch contract.
        for(int row=0;row<live;++row)for(int h=0;h<H;++h) {
            float expected=0;
            for(int route=0;route<K;++route) {
                const size_t slot=row*K+route;
                if(route==0 || route==2) {
                    float sum=value(0,int(slot),h);for(int member=1;member<P;++member)sum+=value(member,int(slot),h);
                    EXPECT_EQ(std::bit_cast<uint32_t>(combined[slot*H+h]),std::bit_cast<uint32_t>(sum));
                    expected=std::fma(denseWeights[slot],sum,expected);
                } else EXPECT_EQ(combined[slot*H+h],sentinel);
            }
            EXPECT_EQ(std::bit_cast<uint32_t>(output[row*H+h]),std::bit_cast<uint32_t>(expected));
        }
        EXPECT_TRUE(std::all_of(combined.begin()+size_t(live)*K*H,combined.end(),[](float v){return v==sentinel;}));
        EXPECT_TRUE(std::all_of(output.begin()+size_t(live)*H,output.end(),[](float v){return v==sentinel;}));
    }
}
