/** @file RetainedHostServiceDeclarations.h
 * @brief Rank-local immutable worker declarations and admitted move-only instance capabilities.
 *
 * Runtime configuration carries declarations only. The admitted binding and
 * live worker remain on their original rank/PMA; neither is a wire payload.
 */
#pragma once
#include "RetainedHostServiceProgram.h"
#include <span>
#include <vector>
#include <mutex>
namespace llaminar2 {
struct RetainedHostServiceInstanceState;
class RetainedHostServiceBinding;
class ResolvedImmutableRowTablePlacement;
namespace test {struct RetainedHostServiceReplayAccess;}
/** @brief Consumer labels select declarations; physical storage always belongs to CPU0. */
struct RetainedHostServiceInstance final {
    int worldRank;
    DeviceId consumer;
    RetainedHostServiceRole role;
    std::uint64_t ordinal=0; ///< Original cold program instance; temporal MTP variants share one ordinal.
    bool operator==(const RetainedHostServiceInstance&)const=default;
};
/** @brief One already claimed instance; moves once from admitted binding to actual consumer. */
class RetainedHostServiceUseToken final {
public:
    RetainedHostServiceUseToken(const RetainedHostServiceUseToken&)=delete;
    RetainedHostServiceUseToken& operator=(const RetainedHostServiceUseToken&)=delete;
    RetainedHostServiceUseToken(RetainedHostServiceUseToken&&)noexcept;
    RetainedHostServiceUseToken& operator=(RetainedHostServiceUseToken&&)noexcept;
    ~RetainedHostServiceUseToken();
    /** @return Original immutable role; moved-from tokens reject. */
    const RetainedHostServiceProgram& program()const;
    /** @return Original admitted authority; moved-from tokens reject. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const;
private:
    friend class RetainedHostServiceBinding;
    friend class RetainedHostServiceDeclarations;
    friend class PreparedRetainedHostService;
    friend struct test::RetainedHostServiceReplayAccess;
    explicit RetainedHostServiceUseToken(std::unique_ptr<RetainedHostServiceInstanceState>);
    /** @brief Claim before mapping; only the frozen binding/recorded fixture can mint this capability. */
    static RetainedHostServiceUseToken claim(std::shared_ptr<PhysicalMemoryAuthority>,RetainedHostServiceProgram,RetainedHostServiceInstance);
    std::unique_ptr<RetainedHostServiceInstanceState> state_;
};
/** @brief Original rank receipt identity; numerical limits remain installed source data. */
struct RetainedHostServiceClassProjection final {
    RetainedHostServiceRole role;
    std::string resourceIdentity,liveIdentity;
    RetainedHostServiceResourceDeclaration declaration; ///< Original rank geometry and shared-policy opaque demand.
};
/** @brief Sealed metadata budget projection; never a remote native-use capability. */
class RetainedHostServiceBudgetProjection final {
public:
    /** @return Both existing role identities from one original loaded-provider evidence session. */
    static RetainedHostServiceBudgetProjection original();
    /** @return Serializable bounded role identities only, without reservations or native owners. */
    std::vector<RetainedHostServiceClassProjection> classes()const;
    /** @brief Validate bounded original-rank receipt identity; only selected roles require an installed class. */
    static RetainedHostServiceBudgetProjection installed(std::span<const RetainedHostServiceClassProjection>);
    /** @return One exact role from the original rank's receipt; no local evidence is substituted. */
    RetainedHostServiceProgram program(RetainedHostServiceRole)const;
private:
    friend struct test::RetainedHostServiceReplayAccess;
    explicit RetainedHostServiceBudgetProjection(std::vector<RetainedHostServiceProgram> programs):programs_(std::move(programs)){}
    explicit RetainedHostServiceBudgetProjection(std::vector<RetainedHostServiceClassProjection> receipts):receipts_(std::move(receipts)){}
    std::vector<RetainedHostServiceProgram> programs_;
    std::vector<RetainedHostServiceClassProjection> receipts_;
};
/** @brief Immutable candidate-local service demand, without live owners or reservations. */
class RetainedHostServiceDeclarations final {
public:
    /** @brief Construct the only mutable declaration scope; freeze consumes it. */
    class Builder final {
    public:
        explicit Builder(int originalRank);
        /** @brief Add one topology-authored unique role/participant, never a native creator. */
        Builder& add(RetainedHostServiceInstance,RetainedHostServiceProgram);
        /** @brief Add only the exact original-rank sealed metadata projection, before quotas. */
        Builder& addProjected(RetainedHostServiceInstance,const RetainedHostServiceBudgetProjection&);
        /** @brief Reuse one immutable original evidence snapshot for every authored role in this session. */
        Builder& addOriginal(RetainedHostServiceInstance);
        /** @brief Consume declarations before optional quotas or materialization. */
        std::shared_ptr<const RetainedHostServiceDeclarations> freeze()&&;
    private:
        int rank_;
        std::shared_ptr<const RetainedHostServiceProgram::Evidence> original_evidence_;
        std::vector<std::pair<RetainedHostServiceInstance,RetainedHostServiceProgram>> entries_;
    };
    /** @return Immutable authored instance labels, without owners or reservations. */
    std::vector<RetainedHostServiceInstance> instances()const;
    /** @return Exact selected instance program; missing/foreign roles fail before any reservation. */
    const RetainedHostServiceProgram& program(RetainedHostServiceInstance)const;
    /** @return Whether this original-rank role exists, without creating metadata/owners. */
    bool contains(RetainedHostServiceInstance)const;
    /** @return Actual final roles: fixed executor demand plus only selected hosted lookup arms. */
    std::vector<RetainedHostServiceInstance> selectedForLookup(const ResolvedImmutableRowTablePlacement*)const;
    /** @return Original rank, not a GPU generation or remote pointer identity. */
    int worldRank()const noexcept{return rank_;}
    /** @return Selected role's typed mandatory contribution from the original CPU0 observation. */
    PhysicalMemoryPlan bom(RetainedHostServiceInstance,PhysicalMemoryResource)const;
    /** @brief Bind selected metadata to the original admitted shared owner; take claims each original instance. */
    std::unique_ptr<RetainedHostServiceBinding> bind(std::shared_ptr<PhysicalMemoryAuthority>,std::span<const RetainedHostServiceInstance>)const;
private:
    friend class Builder;
    RetainedHostServiceDeclarations(int,std::vector<std::pair<RetainedHostServiceInstance,RetainedHostServiceProgram>>);
    int rank_;
    std::vector<std::pair<RetainedHostServiceInstance,RetainedHostServiceProgram>> entries_;
};
/** @brief Local original-PMA instance authority; never stored in RuntimeConfig or serialized. */
class RetainedHostServiceBinding final {
public:
    /** @brief Claim/transfer one original instance, rejecting duplicates across separate bindings and models. */
    RetainedHostServiceUseToken take(RetainedHostServiceInstance);
private:
    friend class RetainedHostServiceDeclarations;
    struct Entry{RetainedHostServiceInstance key;RetainedHostServiceProgram program;bool consumed=false;};
    RetainedHostServiceBinding(std::shared_ptr<PhysicalMemoryAuthority>,std::vector<Entry>);
    std::mutex mutex_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::vector<Entry> entries_;
};
}
