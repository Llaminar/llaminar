/**
 * @file CPUCanonicalPreweightReplay.inc
 * @brief Actual CPU sparse-route preweight arithmetic and complete-prefix replay.
 *
 * Uses the original collective workspace metadata without a mapped ticket,
 * native CPU readiness, device pointers or a fabricated publication outcome.
 */
#include "execution/moe/MoEOverlaySparseCollective.h"

/** @test Either original or compact coordinates apply each immutable weight exactly once. */
TEST(Test__MoELocalExpertStage_PreparedWeights, CPUCanonicalPreweightCoordinatesAndDirtyReplay)
{
    constexpr size_t rows=65,width=8,slots=rows*2;
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(rows,slots,width,2,DeviceId::cpu());
    auto input=workspace.localExpertInput(0,0);
    for(size_t i=0;i<rows;++i) {
        input.original_route_slots_host[i]=static_cast<int32_t>(2*i+1);
        input.compact_route_slots_host[i]=static_cast<int32_t>(i);
        input.route_weights_host[i]=float((i%3)+1)*0.25f;
    }
    using Index=MoEOverlayCanonicalRouteSourceIndex;
    for(auto source_index:{Index::CompactRouteSlot,Index::OriginalRouteSlot})
    for(auto destination_index:{Index::CompactRouteSlot,Index::OriginalRouteSlot}) {
        std::vector<float> source(slots*width),destination(slots*width,-917.0f),expected=destination;
        for(size_t i=0;i<source.size();++i)source[i]=float(i+1);
        const auto untouched_source=source;
        for(size_t count:{size_t(65),size_t(17),size_t(1),size_t(0),size_t(65)}) {
            input.live_row_count=count;input.live_entry_count=count;
            for(size_t entry=0;entry<count;++entry) {
                const size_t from=source_index==Index::OriginalRouteSlot?2*entry+1:entry;
                const size_t to=destination_index==Index::OriginalRouteSlot?2*entry+1:entry;
                for(size_t col=0;col<width;++col)
                    expected[to*width+col]=source[from*width+col]*input.route_weights_host[entry];
            }
            for(int replay=0;replay<2;++replay) {
                const auto source_layout=source_index==Index::OriginalRouteSlot
                    ?MoEOverlayCanonicalRouteSourceLayout::originalSlots()
                    :MoEOverlayCanonicalRouteSourceLayout::compactSlots();
                ASSERT_TRUE(preweightMoEOverlayCanonicalRouteRows(input,source,source_layout,destination,destination_index));
                EXPECT_EQ(destination,expected) << "count="<<count<<" replay="<<replay;
                EXPECT_EQ(source,untouched_source);
            }
        }
    }
}

/** @test An invalid final metadata item must never partially overwrite earlier valid rows. */
TEST(Test__MoELocalExpertStage_PreparedWeights, CPUCanonicalPreweightInvalidCompletePrefixLeavesDestinationUntouched)
{
    constexpr size_t rows=17,width=8,slots=rows*2;
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(rows,slots,width,2,DeviceId::cpu());
    auto input=workspace.localExpertInput(0,0);
    input.live_row_count=rows;input.live_entry_count=rows;
    std::vector<float> source(slots*width,8.0f),destination(slots*width,-733.0f);
    const auto sentinel=destination;
    using Index=MoEOverlayCanonicalRouteSourceIndex;
    auto reset=[&] {
        for(size_t i=0;i<rows;++i) {
            input.original_route_slots_host[i]=static_cast<int32_t>(2*i+1);
            input.compact_route_slots_host[i]=static_cast<int32_t>(i);
            input.route_weights_host[i]=0.25f;
        }
    };
    for(int invalid=0;invalid<7;++invalid) {
        reset();
        if(invalid==0)input.original_route_slots_host[rows-1]=input.original_route_slots_host[rows-2];
        if(invalid==1)input.compact_route_slots_host[rows-1]=input.compact_route_slots_host[rows-2];
        if(invalid==2)input.route_weights_host[rows-1]=std::numeric_limits<float>::quiet_NaN();
        if(invalid==3)input.route_weights_host[rows-1]=std::numeric_limits<float>::infinity();
        if(invalid==4)input.original_route_slots_host[rows-1]=slots;
        if(invalid==5)input.compact_route_slots_host[rows-1]=slots;
        if(invalid==6)input.original_route_slots_host[rows-1]=-1;
        EXPECT_FALSE(preweightMoEOverlayCanonicalRouteRows(input,source,MoEOverlayCanonicalRouteSourceLayout::originalSlots(),destination,Index::CompactRouteSlot));
        // Compact out-of-range is tested as its actual source coordinate too.
        if(invalid==5)EXPECT_FALSE(preweightMoEOverlayCanonicalRouteRows(input,source,MoEOverlayCanonicalRouteSourceLayout::compactSlots(),destination,Index::OriginalRouteSlot));
        EXPECT_EQ(destination,sentinel)<<invalid;
    }
    reset();
    EXPECT_FALSE(preweightMoEOverlayCanonicalRouteRows(input,source,MoEOverlayCanonicalRouteSourceLayout::originalSlots(),
        std::span<float>(destination).first(width),Index::OriginalRouteSlot));
    EXPECT_EQ(destination,sentinel);
    const auto original_source=source;
    EXPECT_FALSE(preweightMoEOverlayCanonicalRouteRows(input,source,MoEOverlayCanonicalRouteSourceLayout::originalSlots(),source,Index::CompactRouteSlot));
    EXPECT_EQ(source,original_source);
}

/** @test Packet coordinates survive changing local compute strides and dirty-buffer replay. */
TEST(Test__MoELocalExpertStage_PreparedWeights, CPUCanonicalPreweightExecutionStrideReplay)
{
    constexpr size_t rows=17,width=8,top_k=8,slots=rows*top_k;
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(rows,slots,width,top_k,DeviceId::cpu());
    auto input=workspace.localExpertInput(0,0);
    using Index=MoEOverlayCanonicalRouteSourceIndex;
    using Layout=MoEOverlayCanonicalRouteSourceLayout;
    for(const size_t execution_width:{8u,1u,2u,7u,3u,4u,5u,6u,1u,8u}) {
        SCOPED_TRACE(execution_width);
        std::vector<float> source(rows*execution_width*width);
        for(size_t i=0;i<source.size();++i)source[i]=float(int(i%53)-26)*0.03125f;
        const auto original_source=source;
        for(const auto destination_index:{Index::CompactRouteSlot,Index::OriginalRouteSlot}) {
            std::vector<float> destination(slots*width,-917.0f),expected=destination;
            for(const size_t live_rows:{0u,1u,3u,17u,1u,17u,0u}) {
                input.live_row_count=live_rows;input.live_entry_count=0;
                for(size_t row=0;row<live_rows;++row) {
                    const size_t routes=row==0?execution_width:1+row%execution_width;
                    for(size_t route=0;route<routes;++route) {
                        const size_t entry=input.live_entry_count++;
                        const size_t compact=row*top_k+route;
                        const size_t original=row*top_k+top_k-routes+route;
                        input.compact_route_slots_host[entry]=static_cast<int32_t>(compact);
                        input.original_route_slots_host[entry]=static_cast<int32_t>(original);
                        input.route_weights_host[entry]=0.1137f*float(route+1);
                        const size_t from=row*execution_width+route;
                        const size_t to=destination_index==Index::OriginalRouteSlot?original:compact;
                        for(size_t column=0;column<width;++column)
                            expected[to*width+column]=source[from*width+column]*input.route_weights_host[entry];
                    }
                }
                for(int replay=0;replay<2;++replay) {
                    ASSERT_TRUE(preweightMoEOverlayCanonicalRouteRows(input,source,
                        Layout::compactExecutionRows(execution_width),destination,destination_index));
                    EXPECT_EQ(std::memcmp(destination.data(),expected.data(),destination.size()*sizeof(float)),0)
                        <<"live_rows="<<live_rows<<" replay="<<replay;
                    EXPECT_EQ(source,original_source);
                }
            }
        }
    }
}

/** @test An invalid narrowed route or truncated source rejects before any destination writes. */
TEST(Test__MoELocalExpertStage_PreparedWeights, CPUCanonicalPreweightExecutionStrideRejectsInvalidPrefix)
{
    constexpr size_t rows=3,width=8,top_k=8,execution_width=2;
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(rows,rows*top_k,width,top_k,DeviceId::cpu());
    auto input=workspace.localExpertInput(0,0);
    input.live_row_count=rows;input.live_entry_count=rows;
    for(size_t row=0;row<rows;++row) {
        input.original_route_slots_host[row]=static_cast<int32_t>(row*top_k+1);
        input.compact_route_slots_host[row]=static_cast<int32_t>(row*top_k);
        input.route_weights_host[row]=0.25f;
    }
    using Index=MoEOverlayCanonicalRouteSourceIndex;
    using Layout=MoEOverlayCanonicalRouteSourceLayout;
    std::vector<float> source(rows*execution_width*width,8.0f),destination(rows*top_k*width,-733.0f);
    const auto sentinel=destination;
    EXPECT_THROW((void)Layout::compactExecutionRows(0),std::invalid_argument);
    const auto layout=Layout::compactExecutionRows(execution_width);
    input.compact_route_slots_host[rows-1]+=execution_width;
    EXPECT_FALSE(preweightMoEOverlayCanonicalRouteRows(input,source,layout,destination,Index::CompactRouteSlot));
    EXPECT_EQ(destination,sentinel);
    input.compact_route_slots_host[rows-1]-=execution_width;
    EXPECT_FALSE(preweightMoEOverlayCanonicalRouteRows(input,std::span<const float>(source).first(4*width),
        layout,destination,Index::CompactRouteSlot));
    EXPECT_EQ(destination,sentinel);
    for(const size_t count:{rows,size_t(0)}) {
        input.live_row_count=count;input.live_entry_count=count;
        EXPECT_FALSE(preweightMoEOverlayCanonicalRouteRows(input,source,
            Layout::compactExecutionRows(top_k+1),destination,Index::CompactRouteSlot));
        EXPECT_EQ(destination,sentinel);
    }
}
