/**
 * @file CPUSerialExpertStorageLayout.h
 * @brief Declarative CPU compact-family roots and frozen prefill admission envelope.
 *
 * One declaration prices and materializes each retained allocation. The layout
 * owns no bytes and keeps no allocation ledger: PhysicalMemoryAuthority alone
 * grants the backing-owned leases carried by CPUStorageRoot allocations.
 * Per-layer CPU packet roots retain their own exact backing allocation claims.
 */
#pragma once
#include "memory/CPUStorageRoot.h"
#include "memory/CPUCompactPacketStorageLayout.h"
#include "memory/CPUGroupedMoEStorageLayout.h"
#include "memory/CPUMappedPublicationRoot.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace llaminar2
{
/** @brief Immutable allocation declarations for one exclusively executing CPU participant. */
class CPUSerialExpertStorageLayout final
{
public:
    /** @brief Common packet declaration; family ownership remains serial-arena specific. */
    using Family=CPUCompactPacketStorageLayout;

    /**
     * @brief Declare an exact normalized family manifest without allocating storage.
     * @param allocator Exact admitted CPU resource.
     * @param maximum_rows Largest admitted token-row capacity.
     * @param hidden_width Model compute width, excluding residual trunk widening.
     * @param route_width Router top-k axis of each compact packet.
     * @param buckets All separately coherent token-row families, including the maximum.
     * @param canonical_routes Whether this endpoint retains raw canonical route rows.
     * @param expert_intermediate Grouped expert width; zero only when grouped scratch is absent.
     * @param num_experts Logical expert count; absent exactly when grouped width is zero.
     * @param publication_consumer Exact GPU alias consumer, when a second mapped canonical bank exists.
     * @throws std::invalid_argument For foreign allocator or incomplete geometry.
     * @throws std::overflow_error For unrepresentable root extents.
     */
    CPUSerialExpertStorageLayout(DeviceId allocator, size_t maximum_rows,
        int hidden_width, int route_width, std::vector<size_t> buckets, bool canonical_routes,
        int expert_intermediate=0,int num_experts=0,std::optional<DeviceId> publication_consumer=std::nullopt)
        : allocator_(allocator), maximum_rows_(maximum_rows), hidden_width_(hidden_width),
          route_width_(route_width), canonical_(canonical_routes
              ? std::optional<CPUStorageRoot<float>>(CPUStorageRoot<float>(
                    multiply(multiply(maximum_rows,positive(route_width)),positive(hidden_width)),
                    PhysicalMemoryOwner::ActivationArena,allocator)) : std::nullopt),
          mapped_(publication_consumer?std::optional<CPUMappedPublicationRoot>(CPUMappedPublicationRoot(
              multiply(multiply(multiply(maximum_rows,positive(route_width)),positive(hidden_width)),sizeof(float)),
              allocator,*publication_consumer)):std::nullopt),
          grouped_(expert_intermediate>0 && num_experts>0?std::optional<CPUGroupedMoEStorageLayout>(
              CPUGroupedMoEStorageLayout::declare(maximum_rows,hidden_width,expert_intermediate,num_experts,route_width,allocator)):
              std::nullopt)
    {
        if (!allocator.is_cpu() || !maximum_rows || hidden_width <= 0 || route_width <= 0 || buckets.empty() ||
            (publication_consumer && !canonical_routes) || expert_intermediate<0 || num_experts<0 ||
            ((expert_intermediate==0)!=(num_experts==0)))
            throw std::invalid_argument("CPU serial storage requires exact CPU geometry and a declared family manifest");
        if (!std::is_sorted(buckets.begin(),buckets.end()) ||
            std::adjacent_find(buckets.begin(),buckets.end()) != buckets.end() || buckets.back()!=maximum_rows)
            throw std::invalid_argument("CPU serial storage family manifest must be unique, ordered and include its maximum");
        for (const auto rows : buckets)
        {
            if (!rows || rows > maximum_rows)
                throw std::invalid_argument("CPU serial storage has an undeclared or zero family capacity");
            families_.push_back(CPUCompactPacketStorageLayout::declare(rows,hidden_width,route_width,allocator));
        }
    }
    /** @return Exact declared CPU allocator. */
    [[nodiscard]] DeviceId allocator() const noexcept { return allocator_; }
    /** @return Maximum token rows admitted for the independently executing owner. */
    [[nodiscard]] size_t maximumRows() const noexcept { return maximum_rows_; }
    /** @return Model compute width of every hidden/output root. */
    [[nodiscard]] int hiddenWidth() const noexcept { return hidden_width_; }
    /** @return Top-k metadata width of every routing root. */
    [[nodiscard]] int routeWidth() const noexcept { return route_width_; }
    /** @return Complete immutable separately coherent root declarations. */
    [[nodiscard]] const std::vector<Family>& families() const noexcept { return families_; }
    /** @return Single shared maximum canonical allocation, when declared. */
    [[nodiscard]] const std::optional<CPUStorageRoot<float>>& canonical() const noexcept { return canonical_; }
    /** @brief Compare every root policy and family capacity.
     * @param other Independently declared layout.
     * @return Whether physical ownership, initialization, placement and geometry match.
     */
    bool operator==(const CPUSerialExpertStorageLayout& other) const = default;
    /** @return Separate mapped publication bank; compute and publication never alias. */
    [[nodiscard]] const std::optional<CPUMappedPublicationRoot>& mappedCanonical() const noexcept { return mapped_; }
    /** @return Setup-owned grouped workspace roots shared only by serial roles. */
    [[nodiscard]] const std::optional<CPUGroupedMoEStorageLayout>& grouped() const noexcept { return grouped_; }
    /** @return Exact ExecutionWorkspace contribution of the declared grouped roots. */
    [[nodiscard]] size_t workspaceBytes() const { return grouped_?grouped_->physicalBytes():0; }
    /** @return Exact ActivationArena BOM contribution for one independent owner. */
    [[nodiscard]] size_t physicalBytes() const
    {
        size_t result = canonical_ ? canonical_->physicalBytes() : 0;
        if(mapped_) result=add(result,mapped_->physicalBytes());
        for (const auto &family : families_) result = add(result,family.physicalBytes());
        return result;
    }
    /** @brief Aggregate byte terms without wrapping.
     * @param a Existing byte total.
     * @param b Additional declared bytes.
     * @return Their exact sum.
     * @throws std::overflow_error When the sum cannot be represented.
     */
    [[nodiscard]] static size_t add(size_t a,size_t b)
    {
        if (b > std::numeric_limits<size_t>::max()-a) throw std::overflow_error("CPU serial storage sum overflow");
        return a+b;
    }
    /** @brief Multiply element geometry without wrapping.
     * @param a First geometry dimension.
     * @param b Second geometry dimension.
     * @return Their exact product.
     * @throws std::overflow_error When the product cannot be represented.
     */
    [[nodiscard]] static size_t multiply(size_t a,size_t b)
    {
        if (a && b>std::numeric_limits<size_t>::max()/a) throw std::overflow_error("CPU serial storage product overflow");
        return a*b;
    }
    /** @brief Declare the compact decode ladder without allocating roots.
     * @param maximum Exact positive final capacity, including non-power-of-two tails.
     * @return Strictly increasing capacities ending at the requested maximum.
     * @throws std::invalid_argument For zero capacity.
     */
    [[nodiscard]] static std::vector<size_t> decodeBuckets(size_t maximum)
    {
        if (!maximum) throw std::invalid_argument("CPU serial decode capacity must be positive");
        std::vector<size_t> result;
        size_t rows=1;
        while(rows<maximum)
        {
            result.push_back(rows);
            if(rows>std::numeric_limits<size_t>::max()/2) break;
            rows*=2;
        }
        result.push_back(maximum);
        return result;
    }
private:
    /** @brief Reject invalid signed dimensions before converting element geometry. */
    [[nodiscard]] static size_t positive(int value)
    {
        if(value<=0) throw std::invalid_argument("CPU serial storage dimensions must be positive");
        return static_cast<size_t>(value);
    }
    DeviceId allocator_;
    size_t maximum_rows_;
    int hidden_width_,route_width_;
    std::vector<Family> families_;
    std::optional<CPUStorageRoot<float>> canonical_;
    std::optional<CPUMappedPublicationRoot> mapped_;
    std::optional<CPUGroupedMoEStorageLayout> grouped_;
};

/** @brief Existing materialization policies, selected before storage admission. */
enum class CPUSerialExpertFamilyPolicy : uint8_t
{
    LocalPowerOfTwo, ///< One mapped/local worker retains the complete token-row ladder.
    PublishedSchedule, ///< Cross-rank follower retains exactly the declared prefill/decode manifest.
};
/**
 * @brief Freeze compact-family admission before collective capacity reduction.
 *
 * Runtime may select a smaller common prefill capacity. Its complete family
 * manifest must project the immutable authored bucket rules; it cannot add
 * independently coherent shapes after admission.
 */
class CPUSerialExpertStorageEnvelope final
{
public:
    /**
     * @brief Freeze one endpoint's policy and maximum physical root declaration.
     * @param allocator Exact rank-local admitted CPU allocator.
     * @param prefill_rows Largest admitted prefill capacity before collective reduction.
     * @param decode_rows Immutable flattened decode/verifier capacity.
     * @param hidden_width Model compute width of hidden/output roots.
     * @param route_width Exact top-k metadata width.
     * @param bucket_inventory Authored captured-prefill capacity inventory.
     * @param bucket_floor Minimum raw-prompt prefill capacity policy.
     * @param bucket_limit Maximum independently retained prefill identities.
     * @param canonical_routes Whether raw canonical route storage is retained.
     * @param family_policy Existing local-worker or remote-follower family policy.
     * @param expert_intermediate Grouped expert width, zero only when grouped scratch is absent.
     * @param num_experts Logical expert count, absent exactly with grouped width.
     * @throws std::invalid_argument For incomplete or unsupported policy/geometry.
     * @throws std::overflow_error For unrepresentable root extents.
     */
    CPUSerialExpertStorageEnvelope(DeviceId allocator,int prefill_rows,int decode_rows,
        int hidden_width,int route_width,std::vector<int> bucket_inventory,int bucket_floor,
        size_t bucket_limit,bool canonical_routes,CPUSerialExpertFamilyPolicy family_policy,
        int expert_intermediate=0,int num_experts=0)
        : allocator_(allocator),prefill_rows_(prefill_rows),decode_rows_(decode_rows),
          hidden_width_(hidden_width),route_width_(route_width),inventory_(std::move(bucket_inventory)),
          floor_(bucket_floor),limit_(bucket_limit),canonical_(canonical_routes),policy_(family_policy),
          expert_intermediate_(expert_intermediate),num_experts_(num_experts)
    {
        if(prefill_rows<=0 || decode_rows<=0 || !limit_ ||
            (policy_!=CPUSerialExpertFamilyPolicy::LocalPowerOfTwo && policy_!=CPUSerialExpertFamilyPolicy::PublishedSchedule))
            throw std::invalid_argument("CPU serial admission requires bounded prefill/decode policy");
        admitted_=project(prefill_rows_,decode_rows_,retainedRawPrefillGraphBucketLadder(
            inventory_,prefill_rows_,floor_,limit_));
    }
    /** @return Maximal declared layout used by the admitted BOM. */
    [[nodiscard]] const std::shared_ptr<const CPUSerialExpertStorageLayout>& admitted() const noexcept { return admitted_; }
    /** @brief Compare immutable declarations independently of cached pointer identity.
     * @param other Independently retained envelope.
     * @return Whether every authored policy and geometry field matches.
     */
    bool operator==(const CPUSerialExpertStorageEnvelope& other) const
    {
        return allocator_==other.allocator_ && prefill_rows_==other.prefill_rows_ && decode_rows_==other.decode_rows_ &&
            hidden_width_==other.hidden_width_ && route_width_==other.route_width_ && inventory_==other.inventory_ &&
            floor_==other.floor_ && limit_==other.limit_ && canonical_==other.canonical_ && policy_==other.policy_ &&
            expert_intermediate_==other.expert_intermediate_ && num_experts_==other.num_experts_;
    }
    /** @return Exact CPU allocator of this immutable declaration. */
    [[nodiscard]] DeviceId allocator() const noexcept { return allocator_; }
    /**
     * @brief Authenticate and project the final collective schedule into allocation roots.
     * @param prefill_rows Final common capacity, no larger than admitted.
     * @param decode_rows Retained flattened decode/verifier capacity.
     * @param prefill_buckets Actual schedule-owned prefill manifest.
     * @param publication_consumer Exact GPU consumer of the distinct mapped publication bank.
     * @return One immutable layout consumed directly by materialization.
     * @throws std::invalid_argument For undeclared policy, buckets or capacities.
     */
    [[nodiscard]] std::shared_ptr<const CPUSerialExpertStorageLayout> project(
        int prefill_rows,int decode_rows,const std::vector<int>& prefill_buckets,
        std::optional<DeviceId> publication_consumer=std::nullopt) const
    {
        if((publication_consumer && (policy_!=CPUSerialExpertFamilyPolicy::LocalPowerOfTwo || !publication_consumer->is_gpu())) ||
            prefill_rows<=0 || prefill_rows>prefill_rows_ || decode_rows!=decode_rows_ ||
            prefill_buckets!=retainedRawPrefillGraphBucketLadder(inventory_,prefill_rows,floor_,limit_))
            throw std::invalid_argument("CPU serial materialization is outside its frozen admission envelope");
        auto rows=CPUSerialExpertStorageLayout::decodeBuckets(static_cast<size_t>(
            policy_==CPUSerialExpertFamilyPolicy::LocalPowerOfTwo ? std::max(prefill_rows,decode_rows) : decode_rows));
        for(const int bucket:prefill_buckets)
        {
            if(bucket<=0 || bucket>prefill_rows) throw std::invalid_argument("CPU serial prefill bucket is undeclared");
            if(policy_==CPUSerialExpertFamilyPolicy::PublishedSchedule)
                rows.push_back(static_cast<size_t>(bucket));
        }
        const auto maximum=static_cast<size_t>(std::max(prefill_rows,decode_rows));
        rows.push_back(maximum);
        std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
        auto layout=std::make_shared<const CPUSerialExpertStorageLayout>(allocator_,maximum,
            hidden_width_,route_width_,std::move(rows),canonical_,expert_intermediate_,num_experts_,publication_consumer);
        const auto mapped_bytes=publication_consumer && admitted_
            ? CPUMappedPublicationRoot(CPUSerialExpertStorageLayout::multiply(
                admitted_->canonical()->capacity(),sizeof(float)),allocator_,*publication_consumer).physicalBytes() : 0;
        if(admitted_ && layout->physicalBytes()>CPUSerialExpertStorageLayout::add(admitted_->physicalBytes(),mapped_bytes))
            throw std::invalid_argument("CPU serial projected root declarations exceed their admitted physical layout");
        return layout;
    }
    /** @brief Project a local worker's power-of-two family from frozen policy.
     * @param prefill_rows Final common prefill capacity.
     * @param decode_rows Immutable flattened decode/verifier capacity.
     * @param consumer Exact GPU publication consumer, absent for a CPU controller.
     * @return Immutable root declarations for this independently retained worker.
     * @throws std::invalid_argument For foreign capacity, policy or consumer identity.
     */
    [[nodiscard]] std::shared_ptr<const CPUSerialExpertStorageLayout> projectLocal(
        int prefill_rows,int decode_rows,std::optional<DeviceId> consumer) const
    {
        if(policy_!=CPUSerialExpertFamilyPolicy::LocalPowerOfTwo)
            throw std::invalid_argument("CPU serial local worker contradicts its frozen family policy");
        return project(prefill_rows,decode_rows,retainedRawPrefillGraphBucketLadder(
            inventory_,prefill_rows,floor_,limit_),consumer);
    }
    /** @brief Project the maximum retained layout for an independent controller.
     * @param controller Declared CPU controller or exact consuming GPU.
     * @return Maximum root declarations used by physical BOM construction.
     * @throws std::invalid_argument For an incompatible family/publication policy.
     */
    [[nodiscard]] std::shared_ptr<const CPUSerialExpertStorageLayout> forController(DeviceId controller) const
    {
        return project(prefill_rows_,decode_rows_,retainedRawPrefillGraphBucketLadder(
            inventory_,prefill_rows_,floor_,limit_),controller.is_gpu()?std::optional{controller}:std::nullopt);
    }
private:
    DeviceId allocator_;
    int prefill_rows_,decode_rows_,hidden_width_,route_width_;
    std::vector<int> inventory_;
    int floor_;
    size_t limit_;
    bool canonical_;
    CPUSerialExpertFamilyPolicy policy_;
    int expert_intermediate_,num_experts_;
    std::shared_ptr<const CPUSerialExpertStorageLayout> admitted_;
};

/** @brief Frozen allocation ownership identities accompanying the admitted physical BOM. */
struct CPUSerialExpertStorageAdmission
{
    PhysicalMemoryAllocatorIdentity allocator; ///< Exact rank-local physical CPU.
    std::vector<int> participant_ids; ///< Each endpoint owns its own root family.
    std::vector<DeviceId> controller_devices; ///< Independently retained worker instances.
    std::shared_ptr<const CPUSerialExpertStorageEnvelope> envelope; ///< Immutable storage declaration.
    /** @brief Compare all worker identities and root declarations structurally.
     * @param other Independently retained declaration.
     * @return Whether endpoint, controller, allocator and root identities match.
     */
    bool operator==(const CPUSerialExpertStorageAdmission& other) const
    {
        return allocator==other.allocator && participant_ids==other.participant_ids &&
            controller_devices==other.controller_devices && bool(envelope)==bool(other.envelope) &&
            (!envelope || *envelope==*other.envelope);
    }
    /** @return Exact concurrent ActivationArena contribution of every declared worker/endpoint. */
    [[nodiscard]] size_t physicalBytes() const
    {
        if(!envelope || !allocator.valid() || allocator.device!=envelope->allocator() || participant_ids.empty() || controller_devices.empty() ||
            !std::is_sorted(participant_ids.begin(),participant_ids.end()) ||
            std::adjacent_find(participant_ids.begin(),participant_ids.end())!=participant_ids.end() || participant_ids.front()<0 ||
            !std::is_sorted(controller_devices.begin(),controller_devices.end()) ||
            std::adjacent_find(controller_devices.begin(),controller_devices.end())!=controller_devices.end() ||
            std::any_of(controller_devices.begin(),controller_devices.end(),[](DeviceId d){return !d.is_valid();}))
            throw std::invalid_argument("CPU serial admission lacks immutable ownership identities");
        size_t bytes=0;
        for(const auto controller:controller_devices)
            bytes=CPUSerialExpertStorageLayout::add(bytes,envelope->forController(controller)->physicalBytes());
        return CPUSerialExpertStorageLayout::multiply(bytes,participant_ids.size());
    }
};

/**
 * @brief Require one exact admitted CPU endpoint and independently executing controller.
 * @param admissions Frozen declarations retained by the admitted stage/model.
 * @param allocator Exact rank and CPU allocator of the target endpoint.
 * @param participant Logical endpoint identity in the routed topology.
 * @param controller Independently executing graph controller owning this family.
 * @return The sole declaration authenticating all requested ownership identities.
 * @throws std::invalid_argument For missing, duplicated or foreign declarations.
 */
inline const CPUSerialExpertStorageAdmission& requireCPUSerialExpertStorageAdmission(
    std::span<const std::shared_ptr<const CPUSerialExpertStorageAdmission>> admissions,
    PhysicalMemoryAllocatorIdentity allocator,int participant,DeviceId controller)
{
    const CPUSerialExpertStorageAdmission* found=nullptr;
    for(const auto &pointer:admissions)
    {
        if(!pointer) throw std::invalid_argument("CPU serial admission contains a null declaration");
        const auto &admission=*pointer;
        if(admission.allocator==allocator &&
           std::find(admission.participant_ids.begin(),admission.participant_ids.end(),participant)!=admission.participant_ids.end() &&
           std::find(admission.controller_devices.begin(),admission.controller_devices.end(),controller)!=admission.controller_devices.end())
        {
            if(found || !admission.envelope) throw std::invalid_argument("CPU serial endpoint has duplicate/incomplete storage declarations");
            found=&admission;
        }
    }
    if(!found) throw std::invalid_argument("CPU serial endpoint/controller lacks its frozen admitted storage declaration");
    return *found;
}
}
