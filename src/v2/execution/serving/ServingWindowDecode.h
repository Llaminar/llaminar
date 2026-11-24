/**
 * @file ServingWindowDecode.h
 * @brief TYPED per-row serving-window decode ABI (design §2/§5.2/§6/§7/§8).
 *
 * The current cohort ABI (prefillBatch/decodeStepBatch) advances logical
 * request batches. Continuous batching advances one admitted row by EXACTLY
 * its grant-issued intrinsic window under the request's own admitted
 * sampling law and resolved seed. This entry is an EXECUTION COMMAND bound to:
 *   - the OWNER-ISSUED one-window grant through the NON-COPYABLE
 *     WindowGrantAuthority (live state queried; a copied Outstanding record
 *     does not survive the real resolution; window policy and destination
 *     reservation come from the authority, §5.2/§10),
 *   - the FULL BINDING/ISSUANCE identity (authority identity, generation,
 *     grant id, monotonic issuance nonce; moved/reissued/foreign commands are
 *     stale, §6),
 *   - the ADMITTED per-row sampling table: the command RETAINS the exact law
 *     node (shared_ptr of the per-issue node) — a later wrapper replacement
 *     or reissue can NEVER retarget an already-minted command (#137),
 *     - the resolved nonzero seed (§7.2), the complete-transaction epoch (§8),
 *   - the row's OWN committed position: ServingRowProgress (engine-owned) and
 *     the mint-only ServingWindowCommandFactory — the command itself has
 *     CONST fields (no caller-mutable position/law).
 *
 * INTRINSIC-GRID endpoint semantics (overflow-safe, no start+W):
 *   servingWindowBound = min(quantum - (start % quantum), limit - start);
 *   servingWindowIsTerminal = (quantum - (start % quantum)) >= (limit - start).
 *
 * COMPLETION IS AN AUTHENTIC FAILURE-ATOMIC COMMIT (design §7.1 outcomes):
 *   completeWindow(authority, command, produced, outcome) re-verifies the full
 *   binding, computes the grid bound and:
 *     - Continue/Terminal require produced == bound (full window);
 *       a window reaching the response limit MUST be Terminal (or an
 *       authenticated exception), never Continue;
 *     - EndOfSequence / Forced / StopMatched are OWNER-AUTHENTICATED
 *       exceptions allowing 0 < produced <= bound (no caller assertion);
 *     - Cancelled is a separate authenticated path (cancelWindow), which
 *       cancels the unused grant exactly once.
 *   Any refusal is FAILURE-ATOMIC (the grant is NOT consumed); repeated/
 *   foreign/stale completion cannot be minted; there is NO public success
 *   constructor.
 *
 * INGRESS IS MANDATORY: the public entry is non-virtual; it validates first
 * (precise diagnostics), then dispatches to the private virtual
 * implementation — an override can never bypass the checks. The base
 * implementation remains fail-closed (explicit unsupported); CPU/CUDA/ROCm
 * row-invariant implementations qualify separately with their own
 * native/parity/economy evidence. Device-owned selection/progress is
 * retained: no host readback is introduced by this ABI.
 */

#pragma once

#include "execution/serving/WindowGrant.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    class ServingLawAdmission; // forward (weak_ptr member of the binding)
    class IOrchestrationRunner; // forward (input friend: ingress reads the node)

    /** @brief PER-ISSUE admitted law table node: each issued binding owns one;
     *         retirement is exactly once (last-holder). The node is NOT publicly
     *         constructible and its fields are PRIVATE (read-only accessors);
     *         the issuing admission authenticates ownership through its issued
     *         registry — a fabricated node cannot be attached. */
    class ServingLawBindingNode
    {
    public:
        ServingLawBindingNode() = delete; // NOT publicly constructible
        [[nodiscard]] bool retired() const noexcept
        {
            return retired_.load(std::memory_order_acquire);
        }
        [[nodiscard]] std::size_t pins() const noexcept
        {
            return pins_.load(std::memory_order_acquire);
        }
        [[nodiscard]] std::uint64_t admissionIdentity() const noexcept
        {
            return admission_identity_;
        }
        [[nodiscard]] std::uint64_t issueIdentity() const noexcept
        {
            return issue_identity_;
        }
        [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
        [[nodiscard]] std::uint64_t tableToken() const noexcept { return table_token_; }
        [[nodiscard]] std::uint64_t schemaVersion() const noexcept { return schema_version_; }
        [[nodiscard]] std::uint64_t tableCapacity() const noexcept { return table_capacity_; }
        /** @brief OWNER AUTHENTICATION: the issuing admission still owns this
         *         exact node (registry membership; fabricated nodes fail). */
        [[nodiscard]] bool authenticated() const noexcept;

    private:
        friend class ServingLawAdmission;
        friend class ServingLawBinding;
        friend class ServingWindowCommandFactory;
        friend class ServingWindowDecodeInput;
        struct CreateKey
        {
        private:
            friend class ServingLawAdmission;
            CreateKey() = default;
        };
        explicit ServingLawBindingNode(CreateKey) {}
        std::atomic<bool> retired_{false};
        std::atomic<std::size_t> pins_{1}; ///< Wrapper (1) + retained command/factory pins.
        std::weak_ptr<ServingLawAdmission> admission_; ///< Retirement callback owner.
        std::uint64_t admission_identity_ = 0; ///< Issuing admission identity (owner auth).
        std::uint64_t issue_identity_ = 0;     ///< Never-reused per-issue identity.
        std::uint64_t epoch_ = 0;
        std::uint64_t table_token_ = 0;
        std::uint64_t schema_version_ = 0;
        std::uint64_t table_capacity_ = 0;
    };

    /** @brief ADMITTED per-row sampling-law table binding (§6/§6.1): move-only
     *         RAII; destruction/move-over-live retires the node exactly once;
     *         a released or moved-from handle is dead and never revives. */
    class ServingLawBinding
    {
    public:
        ServingLawBinding() = default;
        ServingLawBinding(const ServingLawBinding &) = delete;
        ServingLawBinding &operator=(const ServingLawBinding &) = delete;
        ServingLawBinding(ServingLawBinding &&other) noexcept;
        ServingLawBinding &operator=(ServingLawBinding &&other) noexcept;
        ~ServingLawBinding();
        /** @brief Whether this WRAPPER was explicitly released (its pin was
         *         consumed; its storage was reset). */
        [[nodiscard]] bool released() const noexcept { return released_; }

        [[nodiscard]] bool valid() const noexcept
        {
            return admission_owner_ != nullptr && node_ != nullptr && !node_->retired() &&
                   node_->pins() > 0;
        }
        [[nodiscard]] std::uint64_t epoch() const noexcept
        {
            return node_ ? node_->epoch() : 0;
        }
        [[nodiscard]] std::uint64_t tableToken() const noexcept
        {
            return node_ ? node_->tableToken() : 0;
        }
        [[nodiscard]] std::uint64_t schemaVersion() const noexcept
        {
            return node_ ? node_->schemaVersion() : 0;
        }
        [[nodiscard]] std::uint64_t tableCapacity() const noexcept
        {
            return node_ ? node_->tableCapacity() : 0;
        }
        [[nodiscard]] std::uint64_t issueIdentity() const noexcept
        {
            return node_ ? node_->issueIdentity() : 0;
        }
        /** @brief The retained authority owner (acyclic; the lease keeps the
         *         owner alive while the binding lives). */
        [[nodiscard]] std::shared_ptr<ServingLawAdmission> admissionOwner() const noexcept
        {
            return admission_owner_;
        }
        /** @brief The admitted node's issuing admission identity (0 = no node). */
        [[nodiscard]] std::uint64_t admissionIdentityOfNode() const noexcept
        {
            return node_ ? node_->admissionIdentity() : 0;
        }
    private:
        friend class ServingLawAdmission;
        friend class ServingWindowCommandFactory;
        friend class ServingWindowDecodeInput;
        friend class GroupedServingEngine; // engine adopts the row's law node+owner
        friend class ServingLawNodePin;    // RAII pin owner (retain/release)
        /** @brief Retain the node (adds a holder pin; no retirement). */
        [[nodiscard]] static std::shared_ptr<ServingLawBindingNode> retain(
            const std::shared_ptr<ServingLawBindingNode> &node) noexcept
        {
            if (node != nullptr)
                node->pins_.fetch_add(1, std::memory_order_acq_rel);
            return node;
        }
        /** @brief Drop one holder pin; the LAST drop retires the node exactly
         *         once (idempotent; a released/moved-from handle never revives). */
        static void releasePin(const std::shared_ptr<ServingLawBindingNode> &node) noexcept;
        [[nodiscard]] std::shared_ptr<ServingLawBindingNode> node() const noexcept
        {
            return node_;
        }
        ServingLawBinding(std::weak_ptr<ServingLawAdmission> admission,
                          std::shared_ptr<ServingLawAdmission> owner,
                          std::shared_ptr<ServingLawBindingNode> node)
            : admission_(std::move(admission)), admission_owner_(std::move(owner)),
              node_(std::move(node))
        {
        }
        void retireSelf() noexcept;
        friend class ServingLawAdmission; // release() consumes + resets
        std::weak_ptr<ServingLawAdmission> admission_;
        std::shared_ptr<ServingLawAdmission> admission_owner_; ///< ACYCLIC retained authority.
        std::shared_ptr<ServingLawBindingNode> node_;
        bool released_ = false;
    };

    /** @brief DEVICE-FREE FREE OBSERVER: a shared, test/witness-accessible record
     *         of the node-free event (mirrors the accepted S3.1 recording-deleter
     *         seam): frees counts the actual deallocations and
     *         owner_alive_at_free records whether the admission was still alive
     *         at the exact free moment (ordering witness, no fabricated
     *         counters). */
    struct LawFreeObserver
    {
        std::atomic<std::size_t> frees{0};
        std::atomic<bool> owner_alive_at_free{true};
    };

    /** @brief Sole admission resolver for per-row sampling tables: each issue
     *         creates a DISTINCT node (never retargets another handle), enforces
     *         the admitted table capacity as the LIVE physical-node bound
     *         (released nodes are freed with the last holder), assigns a
     *         NEVER-REUSED monotonic issue identity (exhaustion-gated), fails
     *         cleanly BEFORE mutation when not shared, and tracks the exact
     *         live count; it retains no retired node. */
    class ServingLawAdmission : public std::enable_shared_from_this<ServingLawAdmission>
    {
    public:
        explicit ServingLawAdmission(
            std::uint64_t table_capacity = 0,
            std::uint64_t max_issues = std::numeric_limits<std::uint64_t>::max())
            : base_capacity_(table_capacity), max_issues_(max_issues)
        {
            auto cur = next_admission_identity_.load(std::memory_order_acquire);
            while (cur < std::numeric_limits<std::uint64_t>::max() - 1)
            {
                if (next_admission_identity_.compare_exchange_weak(
                        cur, cur + 1, std::memory_order_acq_rel))
                {
                    admission_identity_ = cur + 1;
                    break;
                }
            }
            if (admission_identity_ == 0)
            {
                next_admission_identity_exhausted_.store(true, std::memory_order_release);
                throw std::logic_error("Admission identity space exhausted (sticky)");
            }
        }
        [[nodiscard]] std::uint64_t admissionIdentity() const noexcept
        {
            return admission_identity_;
        }

        /** @brief Attach a shared free observer (device-free witness; the free
         *         event is recorded with owner-aliveness at the exact moment). */
        void observeFree(std::shared_ptr<LawFreeObserver> observer)
        {
            std::lock_guard lock(mutex_);
            observer_ = std::move(observer);
        }

        [[nodiscard]] ServingLawBinding issue(std::uint64_t epoch,
                                              std::uint64_t table_token,
                                              std::uint64_t schema_version,
                                              std::uint64_t table_capacity)
        {
            if (epoch == 0 || table_token == 0 || schema_version == 0 || table_capacity == 0)
                throw std::logic_error(
                    "Law table admission requires positive epoch/token/schema/capacity");
            if (table_capacity > base_capacity_)
                throw std::logic_error("Law table admission exceeds its admitted capacity");
            if (weak_from_this().expired())
                throw std::logic_error(
                    "Law admission must be shared before issuing (no stack/unshared issue)");
            std::lock_guard lock(mutex_);
            if (closed_)
                throw std::logic_error("Law admission is closed (no new bindings)");
            if (physical_nodes_.load(std::memory_order_acquire) >= base_capacity_)
                throw std::logic_error(
                    "Law admission is full (physical-node authority bound)");
            if (issued_count_ >= max_issues_)
                throw std::logic_error("Law issue identity exhausted (bounded)");
            auto self = shared_from_this();
            pruneRegistry(); // bounded scan: expired ownership entries removed first
            auto raw = new ServingLawBindingNode(ServingLawBindingNode::CreateKey{});
            ++physical_nodes_; // physical node counted BEFORE it exists (no underflow)
            auto observer = observer_;
            std::shared_ptr<ServingLawBindingNode> node(
                raw, [self_weak = weak_from_this(), observer](ServingLawBindingNode *p) {
                    // TRUE FREE BOUNDARY: memory is freed first, then the physical
                    // count reflects the free exactly. WEAK capture: the deleter
                    // never retains the admission beyond its holders.
                    delete p;
                    bool owner_alive = false;
                    if (auto a = self_weak.lock())
                    {
                        owner_alive = true;
                        a->physical_nodes_.fetch_sub(1, std::memory_order_acq_rel);
                    }
                    if (observer)
                    {
                        observer->owner_alive_at_free.store(owner_alive,
                                                            std::memory_order_release);
                        observer->frees.fetch_add(1, std::memory_order_release);
                    }
                });
            node->admission_ = self;
            node->admission_identity_ = admission_identity_;
            node->epoch_ = epoch;
            node->table_token_ = table_token;
            node->schema_version_ = schema_version;
            node->table_capacity_ = table_capacity;
            // FAILURE-ATOMIC PUBLICATION: the registry insert happens BEFORE any
            // count is charged (a throwing push_back leaves physical correctly
            // returned by the deleter and live_count/issued_count unchanged).
            issued_registry_.push_back(node);
            ++issued_count_; // monotonic, never reused (identity exhaustion gate)
            node->issue_identity_ = issued_count_;
            ++live_count_;
            return ServingLawBinding(shared_from_this(), self, std::move(node));
        }

        /** @brief Release a LIVE binding that belongs to THIS admission exactly
         *         once (a foreign binding is refused; both owners stay intact). */
        void release(ServingLawBinding &binding)
        {
            if (binding.released_)
                throw std::logic_error("Law binding already released");
            if (binding.admission_.lock().get() != this)
                throw std::logic_error("Law binding does not belong to this admission");
            // Consume THIS wrapper's pin and RESET its storage (a released
            // wrapper no longer holds the node; repeated release is refused; a
            // retained command/factory pin keeps the node valid).
            ServingLawBinding::releasePin(binding.node_);
            binding.node_.reset();
            binding.admission_.reset();
            binding.admission_owner_.reset(); // a released wrapper retains no owner
            binding.released_ = true;
        }

        void close() noexcept
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }

        /** @brief OWNER AUTHENTICATION: whether this admission ISSUED the node
         *         (registry membership; a fabricated node is never admitted). */
        [[nodiscard]] bool owns(const ServingLawBindingNode *node) const noexcept
        {
            if (node == nullptr)
                return false;
            std::lock_guard lock(mutex_);
            for (const auto &w : issued_registry_)
            {
                auto n = w.lock();
                if (n.get() == node)
                    return true;
            }
            return false;
        }
        [[nodiscard]] std::size_t liveCount() const noexcept { return live_count_.load(); }
        [[nodiscard]] std::uint64_t issuedCount() const noexcept
        {
            return issued_count_.load(std::memory_order_acquire);
        }
        [[nodiscard]] std::size_t physicalNodeCount() const noexcept
        {
            return physical_nodes_.load(std::memory_order_acquire);
        }

    private:
        friend class ServingLawBinding;
        void pruneRegistry() noexcept
        {
            // Bounded scan: expired ownership entries are removed (the scan cost
            // tracks the LIVE set, never total lifetime issues).
            for (auto it = issued_registry_.begin(); it != issued_registry_.end();)
            {
                if (it->expired())
                    it = issued_registry_.erase(it);
                else
                    ++it;
            }
        }
        void retireNode(const std::shared_ptr<ServingLawBindingNode> &node) noexcept
        {
            if (node == nullptr || node->admission_.lock().get() != this)
                return; // foreign/no-op
            if (!node->retired_.exchange(true, std::memory_order_acq_rel))
            {
                std::lock_guard lock(mutex_);
                if (live_count_ > 0)
                    --live_count_;
            }
        }
        mutable std::mutex mutex_;
        bool closed_ = false;
        std::size_t base_capacity_ = 0;
        std::atomic<std::size_t> live_count_{0};
        std::atomic<std::uint64_t> issued_count_{0}; ///< Monotonic never-reused identity.
        std::uint64_t max_issues_ = 0;   ///< Exhaustion bound for issue identities.
        std::uint64_t admission_identity_ = 0; ///< Stable per-admission identity.
        std::atomic<std::size_t> physical_nodes_{0}; ///< TRUE physical node count.
        mutable std::vector<std::weak_ptr<ServingLawBindingNode>> issued_registry_;
        std::shared_ptr<LawFreeObserver> observer_; ///< Optional free witness.
        static inline std::atomic<std::uint64_t> next_admission_identity_{0};
        static inline std::atomic<bool> next_admission_identity_exhausted_{false};
        friend struct AdmissionIdentityExhaustionProbe;
    };

    inline bool ServingLawBindingNode::authenticated() const noexcept
    {
        auto a = admission_.lock();
        return a != nullptr && a->owns(this);
    }

    inline ServingLawBinding::ServingLawBinding(ServingLawBinding &&other) noexcept
        : admission_(std::move(other.admission_)), admission_owner_(std::move(other.admission_owner_)),
          node_(std::move(other.node_)), released_(other.released_)
    {
        other.node_.reset();
        other.admission_.reset();
        other.released_ = true;
    }
    inline ServingLawBinding &ServingLawBinding::operator=(
        ServingLawBinding &&other) noexcept
    {
        if (this != &other)
        {
            retireSelf();
            admission_ = std::move(other.admission_);
            admission_owner_ = std::move(other.admission_owner_);
            node_ = std::move(other.node_);
            released_ = other.released_;
            other.node_.reset();
            other.admission_.reset();
            other.released_ = true;
        }
        return *this;
    }
    inline ServingLawBinding::~ServingLawBinding()
    {
        if (!released_)
            retireSelf();
    }
    inline void ServingLawBinding::retireSelf() noexcept
    {
        if (node_ != nullptr)
            ServingLawBinding::releasePin(node_);
        node_.reset();
        admission_.reset();
        admission_owner_.reset();
        released_ = true;
    }
    inline void ServingLawBinding::releasePin(
        const std::shared_ptr<ServingLawBindingNode> &node) noexcept
    {
        if (node == nullptr || node->retired_.load(std::memory_order_acquire))
            return;
        if (node->pins_.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            if (auto admission = node->admission_.lock())
                admission->retireNode(node); // once via exchange
        }
    }

    /** @brief RAII OWNER of a retained law node (the engine's per-row law):
     *         retains the exact node with a holder pin at construction, releases
     *         it exactly once at destruction (any unwind path included), and
     *         drops the owner AFTER the node (the weak deleter always accounts).
     *         No manual release discipline — the row can never leak the pin. */
    class ServingLawNodePin
    {
    public:
        ServingLawNodePin() = default;
        explicit ServingLawNodePin(const ServingLawBinding &law)
            : owner_(law.admissionOwner()),
              node_(ServingLawBinding::retain(law.node()))
        {
        }
        ServingLawNodePin(const ServingLawNodePin &) = delete;
        ServingLawNodePin &operator=(const ServingLawNodePin &) = delete;
        ServingLawNodePin(ServingLawNodePin &&other) noexcept
            : owner_(std::move(other.owner_)), node_(std::move(other.node_))
        {
        }
        ServingLawNodePin &operator=(ServingLawNodePin &&other) noexcept
        {
            if (this != &other)
            {
                // ORDER: free/retire the old node FIRST, then drop the old owner
                // (the weak deleter can always account; the old node never
                // outlives the old owner).
                ServingLawBinding::releasePin(node_);
                node_.reset();
                owner_.reset();
                owner_ = std::move(other.owner_);
                node_ = std::move(other.node_);
            }
            return *this;
        }
        ~ServingLawNodePin() { ServingLawBinding::releasePin(node_); }
        /** @brief NON-OWNING raw view (NO copyable owning export — a caller can
         *         never smuggle an unpinned strong alias out of the pin). */
        [[nodiscard]] const ServingLawBindingNode *node() const noexcept
        {
            return node_.get();
        }
        /** @brief The retained admission owner (plain shared ownership; the
         *         admission is not pin-counted). */
        [[nodiscard]] const std::shared_ptr<ServingLawAdmission> &owner() const noexcept
        {
            return owner_;
        }

    private:
        friend class GroupedServingEngine;
        friend class ServingWindowDecodeInput;
        /** @brief ENGINE-ONLY retained handle (the pin owner mints the ABI input
         *         through this; the input RETAINS before using). */
        [[nodiscard]] const std::shared_ptr<ServingLawBindingNode> &retainedNode() const noexcept
        {
            return node_;
        }
        // DECLARED BEFORE node_: destruction drops the node first, then the owner.
        std::shared_ptr<ServingLawAdmission> owner_;
        std::shared_ptr<ServingLawBindingNode> node_;
    };

    /** @brief INTRINSIC-GRID endpoint bound (overflow-safe; no start+W). */
    [[nodiscard]] inline std::uint64_t servingWindowBound(std::uint64_t start,
                                                          std::uint64_t quantum,
                                                          std::uint64_t limit) noexcept
    {
        if (quantum == 0 || start >= limit)
            return 0;
        const auto grid_remaining = quantum - (start % quantum);
        return std::min(grid_remaining, limit - start);
    }

    /** @brief INTRINSIC-GRID terminal test (overflow-safe). */
    [[nodiscard]] inline bool servingWindowIsTerminal(std::uint64_t start,
                                                      std::uint64_t quantum,
                                                      std::uint64_t limit) noexcept
    {
        if (quantum == 0 || start >= limit)
            return false;
        const auto grid_remaining = quantum - (start % quantum);
        return grid_remaining >= (limit - start);
    }

    /** @brief ROW-OWNED PROGRESS (engine-owned): the engine constructs the
     *         row's progress; a command is minted from THIS. */
    class ServingRowProgress
    {
    public:
        ServingRowProgress(const WindowGrantAuthority &authority,
                           std::uint64_t committed_position, bool terminal, bool carry)
            : authority_identity_(authority.authorityIdentity()),
              generation_(authority.current().generation),
              grant_id_(authority.current().grant_id),
              nonce_(authority.issuanceNonce()),
              committed_position_(committed_position), terminal_(terminal), carry_(carry)
        {
            if (carry && committed_position != 0)
                throw std::logic_error("Carry progress is only the first window (position 0)");
        }
        ServingRowProgress(const ServingRowProgress &) = default;
        [[nodiscard]] std::uint64_t committedPosition() const noexcept
        {
            return committed_position_;
        }
        [[nodiscard]] bool terminal() const noexcept { return terminal_; }
        [[nodiscard]] bool carry() const noexcept { return carry_; }

        /** @return The issuing row authority identity (mint binding). */
        [[nodiscard]] std::uint64_t authorityIdentity() const noexcept
        {
            return authority_identity_;
        }
        [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
        [[nodiscard]] std::uint64_t grantId() const noexcept { return grant_id_; }
        [[nodiscard]] std::uint64_t nonce() const noexcept { return nonce_; }

    private:
        std::uint64_t authority_identity_;
        std::uint64_t generation_;
        std::uint64_t grant_id_;
        std::uint64_t nonce_;
        std::uint64_t committed_position_;
        bool terminal_;
        bool carry_;
    };

    /** @brief TYPED OUTPUT SINK (design §10 destination publication, device-free
     *         interface): the production decode publishes the window output bytes
     *         INTO the destination backing through this sink; the row owner
     *         (engine) then publishes the window from its backing. The engine's
     *         DestinationBacking provides the implementation. */
    class ServingWindowOutputSink
    {
    public:
        virtual ~ServingWindowOutputSink() = default;
        /** @brief STAGE the window output bytes (no backing write yet — the bytes
         *         are committed ONLY after a valid authenticated completion). */
        virtual void writeOutput(const std::byte *src, std::uint64_t bytes) = 0;
        /** @brief Total committed output bytes in the destination so far. */
        [[nodiscard]] virtual std::uint64_t outputBytes() const = 0;
    };

    /** @brief IMMUTABLE EXECUTION COMMAND (const fields; mint-only via
     *         ServingWindowCommandFactory; retains the EXACT law node). */
    class ServingWindowDecodeInput
    {
    public:
        ServingWindowDecodeInput(WindowGrantAuthority &authority,
                                 std::shared_ptr<ServingLawBindingNode> law_node,
                                 std::shared_ptr<ServingLawAdmission> law_owner,
                                 std::uint32_t seed, std::uint64_t transaction_epoch,
                                 std::uint64_t expected_slot,
                                 std::uint64_t expected_instance,
                                 std::uint64_t expected_authority,
                                 std::uint64_t expected_generation,
                                 std::uint64_t expected_grant_id,
                                 std::uint64_t expected_nonce,
                                 const std::uint64_t window_epoch,
                                 std::uint64_t window_start, bool final_partial,
                                 bool first_prefill_carry,
                                 ServingWindowOutputSink *output = nullptr)
            : authority(authority),
              output(output),
              law_node(ServingLawBinding::retain(std::move(law_node))), // PINNED exact node (#137)
              law_owner(std::move(law_owner)), // ACYCLIC retained authority
              seed(seed), transaction_epoch(transaction_epoch),
              expected_slot(expected_slot), expected_instance(expected_instance),
              expected_authority(expected_authority),
              expected_generation(expected_generation), expected_grant_id(expected_grant_id),
              expected_nonce(expected_nonce), window_epoch(window_epoch),
              window_start(window_start), final_partial(final_partial),
              first_prefill_carry(first_prefill_carry)
        {
        }
        /** @brief Convenience: retain the exact node of an admitted binding. */
        ServingWindowDecodeInput(WindowGrantAuthority &authority,
                                 const ServingLawBinding &law_binding,
                                 std::uint32_t seed, std::uint64_t transaction_epoch,
                                 std::uint64_t expected_authority,
                                 std::uint64_t expected_generation,
                                 std::uint64_t expected_grant_id,
                                 std::uint64_t expected_nonce,
                                 const std::uint64_t window_epoch,
                                 std::uint64_t window_start, bool final_partial,
                                 bool first_prefill_carry,
                                 ServingWindowOutputSink *output = nullptr)
            : ServingWindowDecodeInput(authority, law_binding.node(),
                                       law_binding.admissionOwner(), seed,
                                       transaction_epoch,
                                       authority.current().slot,
                                       authority.current().serving_instance,
                                       expected_authority,
                                       expected_generation, expected_grant_id,
                                       expected_nonce, window_epoch, window_start,
                                       final_partial, first_prefill_carry, output)
        {
        }
        ServingWindowDecodeInput(const ServingWindowDecodeInput &) = delete;
        ServingWindowDecodeInput &operator=(const ServingWindowDecodeInput &) = delete;
        ServingWindowDecodeInput &operator=(ServingWindowDecodeInput &&) = delete;
        /** @brief Move: the pin + owner transfer (the moved-from command is
         *         empty; its destructor releases nothing). */
        ServingWindowDecodeInput(ServingWindowDecodeInput &&other) noexcept
            : authority(other.authority), law_node(std::move(other.law_node)),
              law_owner(std::move(other.law_owner)), seed(other.seed),
              transaction_epoch(other.transaction_epoch),
              expected_slot(other.expected_slot), expected_instance(other.expected_instance),
              expected_authority(other.expected_authority),
              expected_generation(other.expected_generation),
              expected_grant_id(other.expected_grant_id),
              expected_nonce(other.expected_nonce), window_epoch(other.window_epoch),
              window_start(other.window_start),
              final_partial(other.final_partial), first_prefill_carry(other.first_prefill_carry),
              output(other.output)
        {
        }
        ~ServingWindowDecodeInput() { ServingLawBinding::releasePin(law_node); }

        [[nodiscard]] bool valid() const noexcept
        {
            const auto &g = authority.current();
            if (law_owner == nullptr || !authority.mayExecute() || g.generation == 0 ||
                law_node == nullptr ||
                law_node->retired() || law_node->admissionIdentity() == 0 ||
                !law_node->authenticated() || law_node->issueIdentity() == 0 ||
                law_node->epoch() == 0 || law_node->tableToken() == 0 ||
                law_node->schemaVersion() == 0 || law_node->tableCapacity() == 0 ||
                seed == 0 ||
                transaction_epoch == 0 || window_epoch == 0 || expected_authority == 0 ||
                authority.authorityIdentity() != expected_authority ||
                g.generation != expected_generation || g.grant_id != expected_grant_id ||
                authority.issuanceNonce() != expected_nonce ||
                window_start >= g.window.response_limit)
                return false;
            const bool terminal_geometry = servingWindowIsTerminal(
                window_start, g.window.quantum, g.window.response_limit);
            if (final_partial && !terminal_geometry)
                return false;
            if (first_prefill_carry && window_start != 0)
                return false;
            return true;
        }

    private:
    public:
        WindowGrantAuthority &authority; ///< Non-const: completion resolves it.
    private:
        friend class IOrchestrationRunner; // ingress reads the retained node
        // DECLARED BEFORE law_node so the owner is destroyed AFTER the node
        // (reverse destruction: node freed first, then the owner is dropped).
        std::shared_ptr<ServingLawAdmission> law_owner; ///< Acyclic retained authority.
        std::shared_ptr<ServingLawBindingNode> law_node; ///< Retained exact node (PRIVATE).
    public:
        /** @brief Typed destination output sink (mint-supplied; the row owner). */
        ServingWindowOutputSink *output = nullptr;
        /** @brief Owning-free const view (no strong ref export; no mutation:
         *         the node fields are private). */
        [[nodiscard]] const ServingLawBindingNode *lawNode() const noexcept
        {
            return law_node.get();
        }
        const std::uint32_t seed;
        const std::uint64_t transaction_epoch;
        const std::uint64_t expected_slot;
        const std::uint64_t expected_instance;
        const std::uint64_t expected_authority;
        const std::uint64_t expected_generation;
        const std::uint64_t expected_grant_id;
        const std::uint64_t expected_nonce;
        const std::uint64_t window_epoch; ///< Sealed per-step window epoch (§8).
        const std::uint64_t window_start;
        const bool final_partial;
        const bool first_prefill_carry;
    };

    /** @brief ROW-OWNER COMMAND FACTORY: the engine/execution side (the row
     *         owner) mints the execution command from the row's OWN progress —
     *         the position is engine-owned; the HTTP caller never writes it.
     *         The factory retains the exact law node (a temporary or replaced
     *         wrapper cannot dangle or retarget the minted command). */
    class ServingWindowCommandFactory
    {
    public:
        ServingWindowCommandFactory(WindowGrantAuthority &authority,
                                    const ServingLawBinding &law_binding,
                                    std::uint32_t seed, std::uint64_t transaction_epoch,
                                    std::uint64_t window_epoch = 0)
            : authority_(authority),
              law_node_(ServingLawBinding::retain(law_binding.node())),
              law_owner_(law_binding.admissionOwner()), seed_(seed),
              transaction_epoch_(transaction_epoch), window_epoch_(window_epoch)
        {
            if (law_node_ == nullptr)
                throw std::logic_error("Command factory requires an admitted law node");
        }
        ServingWindowCommandFactory(const ServingWindowCommandFactory &) = delete;
        ServingWindowCommandFactory &operator=(const ServingWindowCommandFactory &) = delete;
        ServingWindowCommandFactory(ServingWindowCommandFactory &&) = delete;
        ServingWindowCommandFactory &operator=(ServingWindowCommandFactory &&) = delete;
        ~ServingWindowCommandFactory() { ServingLawBinding::releasePin(law_node_); }

        /** @brief Mint the execution command for the row's OWN progress. */
        [[nodiscard]] ServingWindowDecodeInput forRow(const ServingRowProgress &row) const
        {
            if (row.authorityIdentity() != authority_.authorityIdentity() ||
                row.generation() != authority_.current().generation ||
                row.grantId() != authority_.current().grant_id ||
                row.nonce() != authority_.issuanceNonce())
                throw std::logic_error(
                    "Row progress belongs to a different/stale issuance (foreign or "
                    "resolved->reissued row)");
            return ServingWindowDecodeInput(authority_, law_node_, law_owner_, seed_,
                                            transaction_epoch_,
                                            authority_.current().slot,
                                            authority_.current().serving_instance,
                                            authority_.authorityIdentity(),
                                            authority_.current().generation,
                                            authority_.current().grant_id,
                                            authority_.issuanceNonce(), window_epoch_,
                                            row.committedPosition(), row.terminal(), row.carry());
        }

    private:
        WindowGrantAuthority &authority_;
        std::shared_ptr<ServingLawAdmission> law_owner_; ///< Acyclic retained authority.
        std::shared_ptr<ServingLawBindingNode> law_node_; ///< Retained exact node.
        std::uint32_t seed_;
        std::uint64_t transaction_epoch_;
        std::uint64_t window_epoch_;
    };

    /** @brief Window outcome of one serving-window decode (design §7.1). */
    enum class WindowTermination : std::uint8_t
    {
        Continue,      ///< Normal full window; the row continues.
        Terminal,      ///< The window reached the response limit (row completes).
        EndOfSequence, ///< AUTHENTICATED EOS (owner confirmed).
        Forced,        ///< AUTHENTICATED forced-token commit.
        StopMatched,   ///< AUTHENTICATED stop-token match.
        Cancelled,     ///< AUTHENTICATED cancellation (separate path).
    };

    /** @brief TYPED, invariant-carrying result of one serving-window decode. */
    class ServingWindowDecodeResult
    {
    public:
        [[nodiscard]] static ServingWindowDecodeResult failure(std::string error)
        {
            if (error.empty())
                throw std::logic_error("Serving-window failure requires a diagnostic");
            return ServingWindowDecodeResult(std::move(error), 0, WindowGrant{}, false,
                                             WindowTermination::Continue, false);
        }


        [[nodiscard]] const std::string &error() const noexcept { return error_; }
        [[nodiscard]] std::uint64_t produced() const noexcept { return produced_; }
        [[nodiscard]] std::uint64_t servingInstance() const noexcept
        {
            return consumed_.serving_instance;
        }
        [[nodiscard]] std::uint32_t slot() const noexcept { return consumed_.slot; }
        [[nodiscard]] std::uint64_t generation() const noexcept { return consumed_.generation; }
        [[nodiscard]] std::uint64_t grantId() const noexcept { return consumed_.grant_id; }
        [[nodiscard]] std::uint64_t authorityIdentity() const noexcept
        {
            return authority_identity_;
        }
        [[nodiscard]] std::uint64_t issuanceNonce() const noexcept { return issuance_nonce_; }
        [[nodiscard]] std::uint64_t transactionEpoch() const noexcept
        {
            return transaction_epoch_;
        }
        [[nodiscard]] std::uint64_t lawIssueIdentity() const noexcept
        {
            return law_issue_identity_;
        }
        [[nodiscard]] std::uint64_t lawAdmissionIdentity() const noexcept
        {
            return law_admission_identity_;
        }
        [[nodiscard]] std::uint64_t windowStart() const noexcept { return window_start_; }
        [[nodiscard]] std::uint64_t windowQuantum() const noexcept { return window_quantum_; }
        [[nodiscard]] std::uint64_t responseLimit() const noexcept { return response_limit_; }
        [[nodiscard]] std::uint64_t windowEpoch() const noexcept { return window_epoch_; }
        [[nodiscard]] std::uint32_t seed() const noexcept { return seed_; }
        [[nodiscard]] bool finalPartial() const noexcept { return final_partial_; }
        [[nodiscard]] bool firstPrefillCarry() const noexcept { return first_prefill_carry_; }
        [[nodiscard]] bool grantConsumed() const noexcept { return grant_consumed_; }
        [[nodiscard]] WindowTermination termination() const noexcept { return termination_; }
        [[nodiscard]] bool ok() const noexcept
        {
            return error_.empty() && grant_consumed_ && !cancelled_;
        }
        [[nodiscard]] bool cancelled() const noexcept { return cancelled_; }

    private:
        friend ServingWindowDecodeResult completeWindow(WindowGrantAuthority &,
                                                        const ServingWindowDecodeInput &,
                                                        std::uint64_t,
                                                        WindowTermination);
        friend ServingWindowDecodeResult cancelWindow(WindowGrantAuthority &,
                                                      const ServingWindowDecodeInput &);
        /** @brief AUTHENTICATED CANCELLATION RECEIPT: minted ONLY by cancelWindow
         *         after the actual transition (never while Outstanding). */
        [[nodiscard]] static ServingWindowDecodeResult cancelled(
            const WindowGrant &cancelled, std::uint64_t authority_identity,
            std::uint64_t issuance_nonce, std::uint64_t transaction_epoch,
            std::uint64_t law_issue_identity, std::uint64_t law_admission_identity,
            std::uint64_t window_epoch, std::uint64_t window_start,
            std::uint64_t window_quantum, std::uint64_t response_limit,
            std::uint32_t seed, bool final_partial, bool first_prefill_carry)
        {
            if (cancelled.grant_id == 0)
                throw std::logic_error("Cancellation receipt requires a grant identity");
            return ServingWindowDecodeResult(std::string{}, 0, cancelled, false,
                                             WindowTermination::Cancelled, true,
                                             authority_identity, issuance_nonce,
                                             transaction_epoch, law_issue_identity,
                                             law_admission_identity, window_epoch,
                                             window_start, window_quantum, response_limit,
                                             seed, final_partial, first_prefill_carry);
        }
        ServingWindowDecodeResult(std::string error, std::uint64_t produced,
                                  WindowGrant consumed, bool grant_consumed,
                                  WindowTermination termination, bool cancelled,
                                  std::uint64_t authority_identity = 0,
                                  std::uint64_t issuance_nonce = 0,
                                  std::uint64_t transaction_epoch = 0,
                                  std::uint64_t law_issue_identity = 0,
                                  std::uint64_t law_admission_identity = 0,
                                  std::uint64_t window_epoch = 0,
                                  std::uint64_t window_start = 0,
                                  std::uint64_t window_quantum = 0,
                                  std::uint64_t response_limit = 0,
                                  std::uint32_t seed = 0, bool final_partial = false,
                                  bool first_prefill_carry = false)
            : error_(std::move(error)), produced_(produced), consumed_(consumed),
              grant_consumed_(grant_consumed), termination_(termination),
              cancelled_(cancelled), authority_identity_(authority_identity),
              issuance_nonce_(issuance_nonce), transaction_epoch_(transaction_epoch),
              law_issue_identity_(law_issue_identity),
              law_admission_identity_(law_admission_identity), window_epoch_(window_epoch),
              window_start_(window_start), window_quantum_(window_quantum),
              response_limit_(response_limit), seed_(seed),
              final_partial_(final_partial), first_prefill_carry_(first_prefill_carry)
        {
        }

        std::string error_;
        std::uint64_t produced_ = 0;
        WindowGrant consumed_;
        bool grant_consumed_ = false;
        WindowTermination termination_ = WindowTermination::Continue;
        bool cancelled_ = false;
        std::uint64_t authority_identity_ = 0;
        std::uint64_t issuance_nonce_ = 0;
        std::uint64_t transaction_epoch_ = 0;
        std::uint64_t law_issue_identity_ = 0;
        std::uint64_t law_admission_identity_ = 0;
        std::uint64_t window_epoch_ = 0;
        std::uint64_t window_start_ = 0;
        std::uint64_t window_quantum_ = 0;
        std::uint64_t response_limit_ = 0;
        std::uint32_t seed_ = 0;
        bool final_partial_ = false;
        bool first_prefill_carry_ = false;
    };

    /** @brief AUTHENTIC FAILURE-ATOMIC COMMIT (design §7.1 outcome rules). */
    [[nodiscard]] inline ServingWindowDecodeResult completeWindow(
        WindowGrantAuthority &authority, const ServingWindowDecodeInput &command,
        std::uint64_t produced, WindowTermination termination)
    {
        if (&authority != &command.authority)
            throw std::logic_error("Completion authority differs from the command authority");
        if (authority.authorityIdentity() != command.expected_authority ||
            !command.valid())
            throw std::logic_error("Stale or invalid command (binding mismatch)");
        const auto &g = authority.current();
        const auto bound = servingWindowBound(command.window_start, g.window.quantum,
                                              g.window.response_limit);
        const auto remaining = g.window.response_limit - command.window_start;
        const bool at_limit_boundary = (bound == remaining);
        if (termination == WindowTermination::Continue && at_limit_boundary)
            throw std::logic_error(
                "Window reaches the response limit: the outcome must be Terminal "
                "(or an authenticated exception), not Continue");
        switch (termination)
        {
        case WindowTermination::Continue:
            if (!at_limit_boundary && produced != bound)
                throw std::logic_error(
                    "Serving-window success requires produced == the intrinsic-grid "
                    "bound (full window; no under-production without an authenticated "
                    "exception)");
            break;
        case WindowTermination::Terminal:
            if (!at_limit_boundary)
                throw std::logic_error(
                    "Terminal outcome requires the response-limit boundary (an "
                    "early completion is an authenticated EOS/forced/stop observation)");
            if (produced != bound)
                throw std::logic_error(
                    "Terminal outcome requires produced == the intrinsic-grid bound");
            break;
        case WindowTermination::EndOfSequence:
        case WindowTermination::Forced:
        case WindowTermination::StopMatched:
            // AUTHENTICATED OBSERVATION: 0 < produced <= bound (a full window
            // ending in EOS/forced/stop at the boundary is legitimate — the count
            // alone is NOT the authentication; the production decode path is).
            if (produced == 0 || produced > bound)
                throw std::logic_error(
                    "Authenticated exception requires 0 < produced <= intrinsic-grid bound");
            break;
        case WindowTermination::Cancelled:
            throw std::logic_error(
                "Cancellation is a separate authenticated path (cancelWindow)");
        default:
            throw std::logic_error("Unknown window outcome (rejected)");
        }
        authority.resolve(); // exactly once; repeated completion throws
        return ServingWindowDecodeResult(
            std::string{}, produced, g, true, termination, false,
            command.expected_authority, command.expected_nonce,
            command.transaction_epoch, command.lawNode()->issueIdentity(),
            command.lawNode()->admissionIdentity(), command.window_epoch,
            command.window_start,
            g.window.quantum, g.window.response_limit, command.seed,
            command.final_partial, command.first_prefill_carry);
    }

    /** @brief AUTHENTICATED CANCELLATION (exactly once; unused grant cancelled). */
    [[nodiscard]] inline ServingWindowDecodeResult cancelWindow(
        WindowGrantAuthority &authority, const ServingWindowDecodeInput &command)
    {
        if (&authority != &command.authority || !command.valid())
            throw std::logic_error("Cancellation command does not match the authority");
        authority.cancel(); // exactly once; repeated cancellation throws
        return ServingWindowDecodeResult::cancelled(
            authority.current(), command.expected_authority, command.expected_nonce,
            command.transaction_epoch, command.lawNode()->issueIdentity(),
            command.lawNode()->admissionIdentity(), command.window_epoch,
            command.window_start,
            authority.current().window.quantum, authority.current().window.response_limit,
            command.seed, command.final_partial, command.first_prefill_carry);
    }
    /** @brief TEST-ONLY exhaustion probe is DEFINED IN TEST CODE (friend only). */
    struct AdmissionIdentityExhaustionProbe; // defined in tests


} // namespace llaminar2
