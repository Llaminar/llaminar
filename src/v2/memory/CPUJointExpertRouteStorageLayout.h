/**
 * @file CPUJointExpertRouteStorageLayout.h
 * @brief Original-cohort CPU route roots shared by mutually exclusive layer/row families.
 *
 * This immutable declaration prices and materializes the same native roots.
 * It grants no dispatch or source authority and owns no allocation ledger.
 * PhysicalMemoryAuthority and CPUStorageRoot retain all physical byte claims.
 */
#pragma once
#include "CPUStorageRoot.h"
#include "collective/RootedFloatRecordRange.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include <memory>
#include <optional>
#include <limits>
#include <span>
#include <set>
namespace llaminar2 {
/** @brief Source-authenticated serial cohort/member policy, independent of active route counts. */
class CPUJointExpertRouteStorageLayout final {
public:
    /** @brief Validate all retained original source classes before expert quota selection.
     * @param layers Complete retained layer manifests in original global-layer order.
     * @param source Original full E/H/F geometry, never the selected tier quota or compact parent.
     * @param members Exact canonical declared participants, including global IDs/ranks/NUMA.
     * @return Representative physical-geometry cohort; it grants no per-layer source or dispatch authority.
     * @throws std::invalid_argument For missing/duplicate layers, unsupported source blocks, or differing serial H/F/member partitions.
     * @note Runtime retains each layer's actual source lease. A serial arena may share roots only after every original class agrees with these dimensions.
     */
    static std::shared_ptr<const MoEExpertShardCohort> checkedSerialCohort(
        std::span<const MoEOverlayLayerWeightManifest> layers,MoEExpertProjectionOwnership::Geometry source,
        const std::vector<MoEExpertShardCohort::Participant>& members) {
        if(layers.empty())throw std::invalid_argument("CPU joint serial declaration requires every original retained source layer");
        std::shared_ptr<const MoEExpertShardCohort> first;
        int previous=-1;
        for(const auto& layer:layers) {
            if(layer.layer_idx<=previous)throw std::invalid_argument("CPU joint serial source layers must be original, ordered and unique");
            previous=layer.layer_idx;
            auto cohort=std::make_shared<const MoEExpertShardCohort>(MoEExpertShardCohort::checked(layer,source.experts,members,
                MoEExpertShardCohort::balancedIntervals(layer,static_cast<int>(members.size()))));
            if(cohort->ownership(0).geometry()!=source)
                throw std::invalid_argument("CPU joint serial source differs from original full E/H/F geometry");
            if(first)for(size_t member=0;member<members.size();++member)
                if(cohort->ownership(member)!=first->ownership(member))
                    throw std::invalid_argument("CPU joint serial source classes require different local F partitions; one fixed arena cannot claim them equal");
            if(!first)first=std::move(cohort);
        }
        return first;
    }
    /**
     * @brief Declare one member's maximum packed native fragment records and root-only reduction scratch.
     * @param cohort Checked original native source geometry and exact rank/NUMA membership.
     * @param member Original domain ordinal.
     * @param rows Maximum live logical token rows admitted by the serial arena.
     * @param top_k Original source router width.
     * @param root_member Original declared reduction root, never arrival order.
     * @throws std::invalid_argument For foreign membership or incompatible source geometry.
     * @throws std::overflow_error For unrepresentable allocation roots.
     */
    CPUJointExpertRouteStorageLayout(std::shared_ptr<const MoEExpertShardCohort> cohort,
        size_t member,size_t rows,int top_k,size_t root_member=0)
        :cohort_(std::move(cohort)),member_(member),rows_(rows),root_(root_member),top_k_(top_k) {
        if(!cohort_ || member_>=cohort_->participants().size() || root_>=cohort_->participants().size() || !rows_ || top_k_<=0)
            throw std::invalid_argument("CPU joint route storage requires original bounded source/member geometry");
        const auto& source=cohort_->ownership(member_).geometry();
        const auto columns=static_cast<size_t>(source.model_columns);
        const auto slots=multiply(rows_,static_cast<size_t>(top_k_));
        const auto population=cohort_->participants().size();
        if(slots>static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            throw std::overflow_error("CPU joint route slots exceed native packed coordinate");
        const auto placement=CPUStoragePlacement::onNode(cohort_->participants()[member_].address.numa_node);
        const auto records=multiply(slots,isRoot()?population:1);
        packed_.emplace(add(multiply(records,add(columns,1)),1),PhysicalMemoryOwner::ActivationArena,
            DeviceId::cpu(),placement);
        if(isRoot()) {
            ranges_.emplace(population,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
            maps_.emplace(multiply(slots,population),PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
            required_.emplace(slots,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
        }
    }
    /** @return Same immutable checked source cohort used to declare the roots. */
    const std::shared_ptr<const MoEExpertShardCohort>& cohort() const noexcept {return cohort_;}
    /** @return Original member coordinate, never a CPU device ordinal. */
    size_t member() const noexcept {return member_;}
    /** @return Whether this original member owns the single combined route publication. */
    bool isRoot() const noexcept {return member_==root_;}
    /** @return Maximum logical rows; communication uses the actual record count instead. */
    size_t maximumRows() const noexcept {return rows_;}
    /** @return Original declared reduction coordinate. */
    size_t rootMember() const noexcept {return root_;}
    /** @return Equivalent original source/member roots at a smaller serial maximum. */
    CPUJointExpertRouteStorageLayout withMaximumRows(size_t rows) const {
        if(rows>rows_)throw std::invalid_argument("CPU joint serial projection exceeds original admitted maximum");
        return {cohort_,member_,rows,top_k_,root_};
    }
    /** @return Source router width. */
    int topK() const noexcept {return top_k_;}
    /** @return Original native fragment width from the checked source partition. */
    int intermediate() const {return cohort_->ownership(member_).projection(WeightRole::MoEExpertUp).rows;}
    /** @return Exact packed producer/receive allocation, including its native count word. */
    const CPUStorageRoot<float>& packed() const noexcept {return *packed_;}
    /** @return Root-only original source-range records, absent on contributors. */
    const std::optional<CPUStorageRoot<RootedFloatRecordRange>>& ranges() const noexcept {return ranges_;}
    /** @return Root-only original-route maps, absent on contributors. */
    const std::optional<CPUStorageRoot<int32_t>>& maps() const noexcept {return maps_;}
    /** @return Root-only original route-presence scratch, absent on contributors. */
    const std::optional<CPUStorageRoot<float>>& required() const noexcept {return required_;}
    /** @return Exact disjoint physical allocation bytes under the existing ActivationArena owner. */
    size_t physicalBytes() const {
        auto bytes=packed_->physicalBytes();
        if(ranges_)bytes=add(bytes,ranges_->physicalBytes());
        if(maps_)bytes=add(bytes,maps_->physicalBytes());
        if(required_)bytes=add(bytes,required_->physicalBytes());
        return bytes;
    }
    /** @return Value equality of original cohort, member and complete allocation geometry. */
    bool operator==(const CPUJointExpertRouteStorageLayout& other) const {
        return *cohort_==*other.cohort_ && member_==other.member_ && root_==other.root_ && rows_==other.rows_ && top_k_==other.top_k_;
    }
private:
    /** @brief Checked native allocation-size product. */
    static size_t multiply(size_t a,size_t b) {
        if(a && b>std::numeric_limits<size_t>::max()/a)throw std::overflow_error("CPU joint route storage product overflow");
        return a*b;
    }
    /** @brief Checked native allocation-size sum. */
    static size_t add(size_t a,size_t b) {
        if(b>std::numeric_limits<size_t>::max()-a)throw std::overflow_error("CPU joint route storage sum overflow");
        return a+b;
    }
    std::shared_ptr<const MoEExpertShardCohort> cohort_;
    size_t member_,rows_,root_;
    int top_k_;
    std::optional<CPUStorageRoot<float>> packed_;
    std::optional<CPUStorageRoot<RootedFloatRecordRange>> ranges_;
    std::optional<CPUStorageRoot<int32_t>> maps_;
    std::optional<CPUStorageRoot<float>> required_;
};
}
