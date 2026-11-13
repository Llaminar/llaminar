/**
 * @file TensorParallelAssignmentDeclaration.h
 * @brief Immutable original-source parameters for the canonical TP assignment factory.
 *
 * This value retains source/consumer geometry, not a second assignment or live
 * byte ledger. TensorParallelConfig alone distributes Q/KV/FFN/vocabulary.
 * Actual endpoints and complete assignments are authenticated before physical
 * weight, workspace or cache owners may consume the declaration.
 */
#pragma once
#include "TensorParallelConfig.h"
#include <span>
#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
/** @brief Checked original source/consumer identity shared by planning and actual construction. */
class TensorParallelAssignmentDeclaration final
{
public:
    /**
     * @brief Bind legacy or explicitly declared query-consumer coverage to original source geometry.
     * @param source Original complete Q/KV bounds, never local counts or a replica union.
     * @param coverage Existing typed schema/retained-family coverage policy.
     * @param queries Complete original retained consumer grain, absent for unchanged legacy families.
     * @return Immutable source declaration; no interval is computed here.
     * @throws std::invalid_argument For missing, foreign or contradictory declarations.
     */
    [[nodiscard]] static TensorParallelAssignmentDeclaration fromOriginal(
        KVHeadCoverageDeclaration source,KVHeadCoveragePolicy coverage,
        std::optional<QueryHeadPartitionDeclaration> queries=std::nullopt)
    {
        if(coverage==KVHeadCoveragePolicy::QueryInterval)validateKVHeadCoverageDeclaration(source);
        if(coverage!=KVHeadCoveragePolicy::Legacy && coverage!=KVHeadCoveragePolicy::QueryInterval)
            throw std::invalid_argument("unknown original TP coverage declaration");
        if((coverage==KVHeadCoveragePolicy::QueryInterval)!=queries.has_value() ||
            (queries && queries->originalQueryHeads()!=source.query_heads))
            throw std::invalid_argument("TP original consumer declaration is missing or foreign");
        return TensorParallelAssignmentDeclaration(source,coverage,std::move(queries));
    }
    /** @return The exact schema/retained-family coverage, with no implicit substitution. */
    [[nodiscard]] KVHeadCoveragePolicy coverage() const noexcept {return coverage_;}
    /** @return Complete immutable original Q/KV geometry. */
    [[nodiscard]] KVHeadCoverageDeclaration source() const noexcept {return source_;}
    /** @return Source-owned original GDN consumer declaration when coverage is query-minimal. */
    [[nodiscard]] const std::optional<QueryHeadPartitionDeclaration> &queries() const noexcept {return queries_;}
    /**
     * @brief Delegate source geometry and physical workshares to the sole canonical distributor.
     * @param devices Actual ordered endpoints in this rank-local domain, not placeholders.
     * @param weights Original configured workshares; empty means equal, exactly as existing contexts.
     * @param dff Original FFN semantic width from the same model authority.
     * @param vocabulary Original full vocabulary width.
     * @return Complete authenticated assignment owned by TensorParallelConfig.
     */
    [[nodiscard]] TensorParallelConfig partition(const std::vector<DeviceId> &devices,
        const std::vector<float> &weights,int dff,int vocabulary) const
    {
        requireEndpoints(devices);
        auto actual_weights=weights;
        if(actual_weights.empty())actual_weights.assign(devices.size(),1.f);
        auto result=TensorParallelConfig::proportionalSplit(devices,actual_weights,
            source_.query_heads,source_.kv_heads,dff,vocabulary,coverage_,queries_);
        validate(result,devices);return result;
    }
    /**
     * @brief Authenticate retained assignment identity before using its physical slices.
     * @param assignment Existing canonical config retained by weight/graph/planning owner.
     * @param devices Same actual ordered participant endpoints used at construction.
     * @throws std::invalid_argument For absent policy, stale source/grain, wrong rank/device or invalid intervals.
     */
    void validate(const TensorParallelConfig &assignment,std::span<const DeviceId> devices) const
    {
        requireEndpoints(devices);
        if(!assignment.validate() || assignment.worldSize()!=int(devices.size()) ||
            assignment.totalHeads()!=source_.query_heads || assignment.totalKVHeads()!=source_.kv_heads ||
            assignment.kvHeadCoveragePolicy()!=coverage_ ||
            assignment.queryHeadPartitionDeclaration().has_value()!=queries_.has_value())
            throw std::invalid_argument("retained TP assignment differs from its original declaration");
        if(queries_)
        {
            const auto &actual=*assignment.queryHeadPartitionDeclaration();
            if(actual.originalQueryHeads()!=queries_->originalQueryHeads() ||
                actual.granularity()!=queries_->granularity() || actual.consumers()!=queries_->consumers())
                throw std::invalid_argument("retained TP assignment lost an original consumer");
        }
        for(std::size_t rank=0;rank<devices.size();++rank)
            if(assignment.forRank(int(rank)).local_rank!=int(rank) || assignment.forRank(int(rank)).device!=devices[rank])
                throw std::invalid_argument("retained TP assignment names another physical participant");
    }
private:
    /** @brief Only the checked source factory may publish original consumer identity. */
    TensorParallelAssignmentDeclaration(KVHeadCoverageDeclaration source,KVHeadCoveragePolicy coverage,
        std::optional<QueryHeadPartitionDeclaration> queries):source_(source),coverage_(coverage),queries_(std::move(queries)){}
    /**
     * @brief Reject fabricated/duplicate/mixed endpoints for the explicitly implemented query family.
     * @param devices Exact ordered device identities; no discovery or native query is performed.
     */
    void requireEndpoints(std::span<const DeviceId> devices) const
    {
        if(devices.empty())throw std::invalid_argument("TP declaration lacks actual endpoints");
        if(coverage_!=KVHeadCoveragePolicy::QueryInterval)return;
        for(std::size_t i=0;i<devices.size();++i)
        {
            if(!devices[i].is_gpu() || devices[i].ordinal<0 || devices[i].type!=devices.front().type)
                throw std::invalid_argument("query TP declaration requires homogeneous actual GPU endpoints");
            for(std::size_t j=0;j<i;++j)if(devices[i]==devices[j])
                throw std::invalid_argument("query TP declaration has duplicated physical endpoints");
        }
    }
    KVHeadCoverageDeclaration source_;
    KVHeadCoveragePolicy coverage_;
    std::optional<QueryHeadPartitionDeclaration> queries_;
};
} // namespace llaminar2
