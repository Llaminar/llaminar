/**
 * @file RecordedGDNSourceGeometry.h
 * @brief Complete original model geometry for the canonical recorded GDN source loader.
 *
 * The production TP1 fused-QKV slicer authenticates both ordinary attention and
 * GDN geometry. Retain the actual parsed profile's two declarations together;
 * zeroing the ordinary axes would make TP1 lose its GDN source identity.
 */
#pragma once
#include "RecordedObservation.h"
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "loaders/WeightManagerConfig.h"
#include "config/GDNHeadAssignment.h"

namespace llaminar2::test::recorded_gdn
{
/**
 * @brief Project both complete source declarations from the canonical parsed QFN profile.
 * @param attention Actual original sparse-attention declaration, never dummy dimensions.
 * @param gdn Actual original GDN declaration from the same retained profile.
 * @return Complete production ModelDimensions used identically by TP1/2/4 loading.
 * @throws std::invalid_argument For incomplete or nonrepresentable source axes.
 */
inline ModelDimensions originalSourceDimensions(const Qwen4ExpSparseAttentionGeometry &attention,
    const Qwen4ExpDeltaNetGeometry &gdn)
{
    for(const auto value:{attention.query_heads,attention.kv_heads,attention.head_dimension,
        gdn.key_heads,gdn.value_heads,gdn.head_dimension})
        recorded_observation::require(value>0 && value<=std::uint32_t(std::numeric_limits<int>::max()),
            "Recorded GDN requires the complete original attention and recurrent source geometry");
    return {.n_heads=int(attention.query_heads),.n_kv_heads=int(attention.kv_heads),.head_dim=int(attention.head_dimension),
        .gdn_n_k_heads=int(gdn.key_heads),.gdn_n_v_heads=int(gdn.value_heads),.gdn_d_state=int(gdn.head_dimension)};
}
/**
 * @brief Retain equal original query-head intervals for the production GDN loader.
 * @param dimensions Complete original source declaration, including ordinary attention axes.
 * @param degree Exact supported equal diagnostic cohort width.
 * @return Coordinate-only CPU source-loading assignments on the original query-head axis.
 * @throws std::invalid_argument If any original query/key head boundary is cut.
 */
inline std::vector<DeviceShardingAssignment> originalSourceAssignments(const ModelDimensions &dimensions,int degree)
{
    recorded_observation::require(dimensions.isValid() && dimensions.n_kv_heads>0 && dimensions.hasGDN() &&
        (degree==1 || degree==2 || degree==4) && dimensions.n_heads%degree==0,
        "Recorded GDN source assignment requires complete original integral attention/GDN geometry");
    std::vector<DeviceShardingAssignment> assignments;
    for(int rank=0;rank<degree;++rank)
    {
        const int first=rank*(dimensions.n_heads/degree),count=dimensions.n_heads/degree;
        (void)GDNHeadAssignment::fromPartition(dimensions.gdn_n_k_heads,dimensions.gdn_n_v_heads,first,count,dimensions.n_heads);
        assignments.push_back({.device=DeviceId::cpu(),.head_start=first,.head_count=count,.local_rank=rank});
    }
    return assignments;
}
} // namespace llaminar2::test::recorded_gdn
