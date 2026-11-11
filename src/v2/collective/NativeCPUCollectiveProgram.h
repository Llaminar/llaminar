/**
 * @file NativeCPUCollectiveProgram.h
 * @brief Original CPU expert-domain MPI/runtime commitment and native retirement token.
 *
 * The domain envelope is separate from the exact SharedSpin arena. The process
 * envelope covers incremental first-domain growth after MPI_Init and has one
 * owner in MPIProcessSession through MPI_Finalize. Later admissions borrow that
 * original commitment without reserving or claiming resident bytes again.
 */
#pragma once
#include "backends/GlobalDeviceAddress.h"
#include "config/CollectiveBackendType.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
#include <span>
#include <vector>

namespace llaminar2 {
class GlobalTPContext;
namespace detail { struct MPICollectiveUse; }
namespace test { struct NativeCPUCollectiveProgramReplayAccess; }
/** @brief Closed, ordered operation family for actual CPU expert partial routing. */
class NativeCPUCollectiveProgram final {
public:
    /** @brief Original rank and exact rank-local CPU0/NUMA endpoint, never a logical CPU ordinal. */
    struct Participant {
        int world_rank;
        GlobalDeviceAddress address;
        bool operator==(const Participant&) const = default;
    };
    /** @brief Reservation handed to one original domain creator before any native MPI entry. */
    class DomainUse final {
    public:
        DomainUse(DomainUse&&) noexcept;
        DomainUse& operator=(DomainUse&&) = delete;
        ~DomainUse() noexcept;
        DomainUse(const DomainUse&) = delete;
        DomainUse& operator=(const DomainUse&) = delete;
        /** @return Exact original PMA for the arena creator and context lifetime. */
        [[nodiscard]] const std::shared_ptr<PhysicalMemoryAuthority>& originalMemory() const &;
        const std::shared_ptr<PhysicalMemoryAuthority>& originalMemory() const && = delete;
        /** @return Immutable program authenticated before the first native communicator operation. */
        [[nodiscard]] const NativeCPUCollectiveProgram& program() const &;
        const NativeCPUCollectiveProgram& program() const && = delete;
        /** @return Actual process rank, present in program membership. */
        [[nodiscard]] int originalWorldRank() const;
    private:
        friend class NativeCPUCollectiveProgram;
        friend class GlobalTPContext;
        friend struct test::NativeCPUCollectiveProgramReplayAccess;
        struct State;
        explicit DomainUse(std::unique_ptr<State>);
        /** @brief Arm immediately before MPI_Comm_split; failure still requires original native cleanup. */
        void beginNative();
        /** @brief Release only after original communicator/backend/arena destruction succeeds. */
        void retireAfterNativeDestruction();
        std::unique_ptr<State> state_;
    };
    /** @brief Original process-only pool scope for a nonmember entering the SAME parent split.
     * No domain reservation or CPU membership is fabricated. The original session
     * retains the pool after this native split participation has actually returned. */
    class ProcessUse final {
    public:
        ProcessUse(ProcessUse&&) noexcept;
        ProcessUse& operator=(ProcessUse&&) = delete;
        ~ProcessUse() noexcept;
        ProcessUse(const ProcessUse&) = delete;
        ProcessUse& operator=(const ProcessUse&) = delete;
        /** @return Original excluded process PMA; borrowing requires a retained lvalue token. */
        [[nodiscard]] const std::shared_ptr<PhysicalMemoryAuthority>& originalMemory() const &;
        const std::shared_ptr<PhysicalMemoryAuthority>& originalMemory() const && = delete;
        /** @return Same immutable original CPU domain, without pretending this process is a member. */
        [[nodiscard]] const NativeCPUCollectiveProgram& program() const &;
        const NativeCPUCollectiveProgram& program() const && = delete;
        /** @return Original parent process world rank, absent from CPU domain membership. */
        [[nodiscard]] int originalWorldRank() const;
    private:
        friend class NativeCPUCollectiveProgram;
        friend class GlobalTPContext;
        friend struct test::NativeCPUCollectiveProgramReplayAccess;
        struct State;
        explicit ProcessUse(std::unique_ptr<State>);
        /** @brief Arm before the original excluded parent's MPI_Comm_split entry. */
        void beginNative();
        /** @brief Retire after that original split returns and any partial native handle is cleaned up. */
        void retireAfterNativeSplit();
        std::unique_ptr<State> state_;
    };
    /** @brief Authenticate the original NODE_LOCAL ordered CPU participant family, with one split and optional rooted duplicate.
     * @throws std::invalid_argument for unsupported backend, rank population or unknown/fake CPU endpoint.
     */
    [[nodiscard]] static NativeCPUCollectiveProgram expertShardRoutePartials(
        int domain_id, CollectiveBackendType backend, std::vector<Participant> ordered_members);
    /** @return Exact original program identity, including ordered rank/NUMA membership and backend. */
    bool operator==(const NativeCPUCollectiveProgram&) const = default;
    /** @return Original authored domain identity. */
    [[nodiscard]] int domainId() const noexcept { return domain_id_; }
    /** @return Closed selected CPU backend; AUTO and GPU backends are never inferred. */
    [[nodiscard]] CollectiveBackendType backend() const noexcept { return backend_; }
    /** @return Exact communicator order. */
    [[nodiscard]] std::span<const Participant> participants() const & noexcept { return participants_; }
    std::span<const Participant> participants() const && = delete;
    /** @return Native domain handle population: split plus rooted duplicate for multi-rank domains. */
    [[nodiscard]] std::size_t communicatorCount() const noexcept { return participants_.size() > 1 ? 2 : 1; }
    /** @return Conservative separately declared domain runtime envelope per participating rank. */
    [[nodiscard]] static constexpr std::size_t domainBytes() noexcept { return 16u*1024u*1024u; }
    /** @return Conservative original process first-domain envelope, excluding MPI_Init. */
    [[nodiscard]] static constexpr std::size_t processPoolBytes() noexcept { return 16u*1024u*1024u; }
    /** @brief Contribute one domain envelope per concurrently admitted domain; arena is separate. */
    void contributeDomainBOM(PhysicalMemoryPlanBuilder&, const PhysicalMemoryResource&, int world_rank) const;
    /** @brief Contribute potential first-domain growth ONCE per process allocator/admission, not per domain. */
    static void contributeProcessPoolBOM(PhysicalMemoryPlanBuilder&, const PhysicalMemoryResource&);
    /** @brief Commit the domain and original session pool before the creator enters MPI. */
    [[nodiscard]] DomainUse admit(std::shared_ptr<PhysicalMemoryAuthority>, int world_rank) const;
    /** @brief Commit only the original session pool before a nonmember enters the SAME parent split. */
    [[nodiscard]] ProcessUse admitExcludedParent(std::shared_ptr<PhysicalMemoryAuthority>, int world_rank) const;
private:
    NativeCPUCollectiveProgram(int id, CollectiveBackendType backend, std::vector<Participant> members)
        :domain_id_(id),backend_(backend),participants_(std::move(members)){}
    int domain_id_;
    CollectiveBackendType backend_;
    std::vector<Participant> participants_;
};
}
