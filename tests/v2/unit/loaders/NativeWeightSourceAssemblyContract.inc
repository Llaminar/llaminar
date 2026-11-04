/**
 * @file NativeWeightSourceAssemblyContract.inc
 * @brief Original GDN head selections retain source coordinates through dense final assembly.
 *
 * These device-free geometry checks grant no model, native context or readiness
 * authority. The actual source projector and shared stripe walker validate exact
 * source facts and destination coverage before a consumer can publish output.
 */
#include "config/GDNHeadAssignment.h"
#include "tensors/NativeWeightSourceRange.h"
namespace llaminar2 {
namespace {
TEST(NativeWeightSourceAssemblyContract, OriginalGDNHeadStripesKeepSourceAndFinalGeometrySeparate) {
    constexpr int keys=8,values=32,width=32;
    constexpr size_t k=256,n=(2*keys+values)*width;
    const NativeWeightSourceGeometry source{{k,n},TensorType::Q4_0,32,18};
    for(const int degree:{2,4,8}) {
        SCOPED_TRACE(degree);
        const auto assignment=GDNHeadAssignment::forEqualRank(keys,values,degree-1,degree);
        const auto key=assignment.keyElementSpan(width);
        std::vector<GDNHeadSpan> spans{key,{.start=keys*width+key.start,.count=key.count}};
        for(const auto value:assignment.valueElementSpans(width))
            spans.push_back({.start=2*keys*width+value.start,.count=value.count});
        NativeWeightSourceAssembly assembly{.axis=NativeWeightSourceAssembly::Axis::Rows,
            .final_rows=assignment.localFusedRows(width),.final_columns=k};
        for(const auto span:spans)
            assembly.segments.push_back({.original_expert=0,.selected_expert_ordinal=0,
                .first_row=size_t(span.start),.rows=size_t(span.count),.first_column=0,.columns=k,.native_row_bytes=k/32*18});
        WeightRequirement requirement;
        requirement.canonical_name="blk.0.attn_qkv.weight";
        requirement.role=WeightRole::GDNProjection;
        requirement.derivation=WeightDerivationKind::FusedSubblockConcat;
        requirement.native_source_assembly=assembly;
        const auto projected=projectNativeWeightSource(requirement,source);
        ASSERT_EQ(projected.size(),1u);
        EXPECT_EQ(projected.front().original,source);
        EXPECT_EQ(projected.front().range.rows,n);
        EXPECT_EQ(projected.front().range.columns,k);
        EXPECT_EQ(projected.front().finalRows(),assignment.localFusedRows(width));
        EXPECT_EQ(projected.front().finalColumns(),k);
        size_t rows=0,index=0;
        forEachNativeWeightSourceStripe(source,projected.front().range,projected.front().assembly,
            [&](const auto& stripe,size_t destination_row,size_t destination_column) {
                ASSERT_LT(index,spans.size());
                EXPECT_EQ(stripe.first_row,size_t(spans[index].start));
                EXPECT_EQ(stripe.rows,size_t(spans[index].count));
                EXPECT_EQ(destination_row,rows);EXPECT_EQ(destination_column,0u);
                rows+=stripe.rows;++index;
            });
        EXPECT_EQ(index,spans.size());EXPECT_EQ(rows,assignment.localFusedRows(width));
        requirement.derivation=WeightDerivationKind::Source;
        EXPECT_THROW(projectNativeWeightSource(requirement,source),std::invalid_argument);
    }
}
TEST(NativeWeightSourceAssemblyContract, RejectsIncompleteOrForeignStripesBeforeAnyPublication) {
    const NativeWeightSourceGeometry source{{64,8},TensorType::Q4_0,32,18};
    const NativeWeightSourceWindow range{0,0,0,8,0,64,36};
    const NativeWeightSourceAssembly valid{NativeWeightSourceAssembly::Axis::Rows,3,64,
        {{0,0,6,2,0,64,36},{0,0,1,1,0,64,36}}};
    const auto reject=[&](auto original,auto assembly) {
        size_t writes=0;
        EXPECT_THROW(forEachNativeWeightSourceStripe(original,range,assembly,
            [&](const auto&,size_t,size_t){++writes;}),std::invalid_argument);
        EXPECT_EQ(writes,0u);
    };
    auto bad=valid;bad.final_rows=4;reject(source,bad);
    bad=valid;bad.segments.back().first_row=8;reject(source,bad);
    bad=valid;bad.segments.back().columns=32;bad.segments.back().native_row_bytes=18;reject(source,bad);
    bad=valid;bad.segments.back().original_expert=1;reject(source,bad);
    bad=valid;bad.segments.clear();reject(source,bad);
    bad=valid;bad.axis=static_cast<NativeWeightSourceAssembly::Axis>(99);reject(source,bad);
    auto outer=source;outer.original_shape.push_back(1);reject(outer,valid);
    bad={NativeWeightSourceAssembly::Axis::Columns,8,32,{{0,0,0,8,1,32,18}}};reject(source,bad);
}
TEST(NativeWeightSourceAssemblyContract, ColumnAndVectorStripesHaveExactCumulativeDestinations) {
    const auto assignment=GDNHeadAssignment::forEqualRank(8,32,1,4);
    for(const bool vector:{false,true}) {
        SCOPED_TRACE(vector);
        const size_t unit=vector?1:32;
        const size_t source_columns=32*unit;
        const NativeWeightSourceGeometry source{vector?std::vector<size_t>{source_columns}:
            std::vector<size_t>{source_columns,3},TensorType::FP32,1,4};
        const NativeWeightSourceWindow range{0,0,0,vector?size_t{1}:size_t{3},0,source_columns,source_columns*4};
        NativeWeightSourceAssembly assembly{NativeWeightSourceAssembly::Axis::Columns,range.rows,
            size_t(assignment.localValueHeads())*unit,{}};
        for(const auto span:assignment.valueElementSpans(int(unit)))
            assembly.segments.push_back({0,0,0,range.rows,size_t(span.start),size_t(span.count),size_t(span.count)*4});
        size_t columns=0;
        forEachNativeWeightSourceStripe(source,range,assembly,[&](const auto& stripe,size_t row,size_t col){
            EXPECT_EQ(row,0u);EXPECT_EQ(col,columns);columns+=stripe.columns;
        });
        EXPECT_EQ(columns,assembly.final_columns);
    }
}
} // namespace
} // namespace llaminar2
