/**
 * @file PlacementTopology.h
 * @brief OpenMP-free placement authority: shared raw topology mechanics plus
 *        the strict receipt admission API.
 *
 * AUTHORITY STRUCTURE (single copy of every mechanic):
 * - Raw mechanics (file reads, cpu-list parsing, capacity probing, grouping,
 *   ordering) exist ONCE, in this module.
 * - The strict receipt admission API (fail-closed, typed rejects) is the only
 *   admission authority in this module. It NEVER falls back: unknown topology
 *   and missing/mixed capacity evidence are typed rejections, never silent
 *   substitutions.
 * - The legacy bootstrap binder (MPIBootstrapPhase) keeps its EXISTING
 *   boundary translations (permissive grammar, its documented no-capacity
 *   evidence -> cpu-order libgomp-parity contract) as thin code at its own
 *   call boundary, delegating every mechanic to this module. Those
 *   translations are temporary: final production integration retires the
 *   manual binder machinery; this module never grows a legacy fallback mode.
 *
 * PARSING: one parser implementation, two checked grammars:
 * - CpuListGrammar::StrictReceipt: trusted-receipt grammar (full consumption,
 *   bounds, no empty tokens, no duplicates, ascending ranges).
 * - CpuListGrammar::PermissiveBootstrap: the legacy bootstrap grammar
 *   (empty tokens skipped, stoi partial consumption, negatives passed
 *   through, reversed ranges swapped, output sorted+deduped). Exists ONLY for
 *   the legacy boundary; the strict receipt path never selects it.
 *   Range expansion is bounded (64-bit width, CPU_SETSIZE cap) in BOTH
 *   grammars — the legacy unbounded expansion was an overflow hazard, and
 *   kernel-provided lists never exceed CPU_SETSIZE.
 *
 * TYPED REJECTS: strict-path rejections carry a RejectReason via
 * PlacementTopologyReject (IS-A std::invalid_argument, so existing
 * invalid_argument handling keeps working).
 */
#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::placement
{
    /** @brief Where a capacity number came from. The units differ; values are
     *         only ever compared within a single-source cohort (strict path
     *         enforces this; legacy binder compares raw values at its own
     *         documented boundary). */
    enum class CapacitySource
    {
        Unknown,              ///< Default/fallback marker only; strict derivation never produces it.
        KernelCapacityUnits,  ///< sysfs .../cpu_capacity — kernel normalized units (NOT kHz).
        CpufreqKhz            ///< sysfs .../cpufreq/cpuinfo_max_freq — frequency in kHz.
    };

    /** @brief One place = the admitted CPUs of one physical core.
     *
     *  representative INVARIANT: == sibling_cpus.front() (the lowest admitted
     *  CPU — the CPU the cluster path has always ranked by). Both
     *  deriveLocalReceipt and parseOmpPlaces maintain it, so a
     *  parse(serialize(derive(...))) roundtrip compares equal on
     *  representative and sibling_cpus. Capacity evidence is topology-derived:
     *  only derived places carry it (parsed places leave it Unknown). */
    struct CorePlace
    {
        int representative = -1;
        std::vector<int> sibling_cpus; ///< Admitted CPUs of this physical core, sorted ascending.
        CapacitySource capacity_source = CapacitySource::Unknown;
        long capacity_value = 0; ///< Evidence value of the representative, in capacity_source's units.
    };

    /** @brief Typed reason carried by every strict-path rejection. */
    enum class RejectReason
    {
        MalformedInput,        ///< Unparsable content (junk, empty/trailing tokens, bad grammar).
        DuplicateAdmittedId,   ///< The same CPU admitted twice (set or place list).
        IdOutOfRange,          ///< CPU id outside [0, CPU_SETSIZE).
        MissingTopology,       ///< sysfs sibling file absent (typed unknown topology).
        MissingCapacityEvidence, ///< Neither cpu_capacity nor cpuinfo_max_freq readable.
        IncomparableCohort,    ///< Mixed capacity sources across the cohort (no invented normalization).
        ChecksumMismatch       ///< Transport content checksum does not cover the received bytes (corruption/tampering).
    };

    /** @brief Strict-path rejection. IS-A std::invalid_argument for backward
     *         compatible handling at existing boundaries. */
    struct PlacementTopologyReject : std::invalid_argument
    {
        RejectReason reason;
        PlacementTopologyReject(RejectReason reason_value, const std::string &what)
            : std::invalid_argument(what), reason(reason_value) {}
    };

    struct PlacementTransportView; // defined below; friends the receipt class

    /**
     * @brief Immutable local placement receipt — CONSTRUCTION BARRIER.
     *
     * A receipt cannot be default-constructed or assembled from arbitrary
     * parts: the only constructors are the two VALIDATING factories
     * (deriveLocalReceipt and parseReceiptTransport, declared friends), both
     * of which fully validate before producing the value. No mutators exist;
     * copies of a valid receipt are valid. This makes an invalid receipt
     * value unrepresentable in the type system, so the policy-free mechanics
     * (including the groupAdmittedCoresByCapacity template) cannot construct
     * or mutate one.
     */
    class PlacementReceipt
    {
    public:
        PlacementReceipt(const PlacementReceipt &) = default;
        PlacementReceipt &operator=(const PlacementReceipt &) = default;
        ~PlacementReceipt() = default;

        const std::vector<CorePlace> &ordered_places() const { return ordered_places_; }
        const std::string &omp_places_string() const { return omp_places_string_; }
        const std::string &content_checksum() const { return content_checksum_; }

    private:
        struct ConstructionPasskey
        {
        }; // usable only by the friend factories
        friend PlacementReceipt deriveLocalReceipt(
            const std::vector<int> &, const std::string &);
        friend std::optional<PlacementReceipt> parseReceiptTransport(
            const std::string &, PlacementTopologyReject &);
        friend std::optional<PlacementTransportView> parseReceiptTransportView(
            const std::string &, PlacementTopologyReject &);
        explicit PlacementReceipt(ConstructionPasskey,
            std::vector<CorePlace> ordered_places, std::string omp_places_string,
            std::string content_checksum)
            : ordered_places_(std::move(ordered_places)),
              omp_places_string_(std::move(omp_places_string)),
              content_checksum_(std::move(content_checksum)) {}

        std::vector<CorePlace> ordered_places_; ///< Capacity-ordered (fastest first).
        std::string omp_places_string_;         ///< Explicit "{a},{b,c},..." serialization.
        std::string content_checksum_;          ///< SHA-256 CONTENT CHECKSUM of omp_places_string (integrity only; NOT authentication, NOT ownership).
    };

    /** @brief Per-session identity fields carried by the receipt transport.
     *  rank comes from the INDEPENDENT MPI/session environment observed by
     *  the helper (PMI-class variables), never invented by the helper. */
    struct ReceiptSessionIdentity
    {
        std::string session; ///< Launcher-generated unique token per mpirun invocation (NOT a secret; no authorization).
        std::string host;    ///< gethostname() at helper time.
        std::string rank;    ///< Independent MPI rank identity (e.g. PMI/OMPI variable value).
        long budget = 0;     ///< Resolved worker budget (positive).
        bool budget_origin_explicit = false; ///< true = operator-authored value; false = auto-resolved from THIS helper's local admitted places.
    };

    /**
     * @brief Serialize a validated receipt + session identity into the ONE
     *     transport schema (v1): fixed field order, `;`-separated key=value
     *     pairs, restricted charset for identity fields, SHA-256 content
     *     checksum over the exact bytes of all preceding fields appended as
     *     the final field.
     * @throws PlacementTopologyReject for empty/charset-invalid identity
     *     fields or a non-positive budget.
     */
    std::string serializeReceiptTransport(const PlacementReceipt &receipt,
        const ReceiptSessionIdentity &identity);

    /**
     * @brief STRICT transport ingestion (the only other receipt factory).
     *
     * Validates: exact field order and schema version, identity charset and
     * non-emptiness, positive budget, strict parse of admitted + places,
     * place_count consistency, exact partition of admitted by the places,
     * representative invariants, single capacity source with one value per
     * place, and the SHA-256 content checksum over the exact preceding
     * bytes (any byte difference = corruption, never tolerated).
     *
     * @return The validated receipt, or std::nullopt with @p reject filled
     *     (typed reason) — the transport layer reports instead of throwing
     *     so callers can log the typed diagnostic before their own fatal.
     */
    std::optional<PlacementReceipt> parseReceiptTransport(
        const std::string &transport, PlacementTopologyReject &reject);

    /** @brief Full transport view: the validated receipt AND its validated
     *         session identity (the rank-side verification needs the
     *         identity fields for independent correlation). */
    struct PlacementTransportView
    {
        PlacementReceipt receipt;
        ReceiptSessionIdentity identity;
    };

    /** @brief Same strict validation as parseReceiptTransport, returning the
     *         identity fields as well. */
    std::optional<PlacementTransportView> parseReceiptTransportView(
        const std::string &transport, PlacementTopologyReject &reject);

    /** @brief Grammar selector for the ONE cpu-list parser implementation. */
    enum class CpuListGrammar
    {
        PermissiveBootstrap, ///< Legacy bootstrap grammar (legacy boundary ONLY).
        StrictReceipt        ///< Trusted-receipt grammar (full consumption, bounds, duplicates rejected).
    };

    /**
     * @brief Shared raw mechanic (M1): read one cpu's sibling list.
     * @return The file's first line, or std::nullopt when the file is ABSENT
     *     (typed unknown topology). An existing-but-empty file yields "".
     */
    std::optional<std::string> readSiblingList(int cpu, const std::string &sysfs_root);

    /**
     * @brief THE cpu-list parser (one implementation, grammar-checked).
     * @throws PlacementTopologyReject in StrictReceipt mode for any violation;
     *     in PermissiveBootstrap mode only for an expansion width beyond
     *     CPU_SETSIZE (legacy unbounded expansion was an overflow hazard).
     */
    std::vector<int> parseCpuList(const std::string &cpulist, CpuListGrammar grammar);

    /** @brief Strict-grammar entry (kept for receipt-path readability). */
    inline std::vector<int> parseCpuListStrict(const std::string &cpulist)
    {
        return parseCpuList(cpulist, CpuListGrammar::StrictReceipt);
    }

    /**
     * @brief Shared raw mechanic (M3): capacity evidence for one cpu.
     * @return (value, source) — cpu_capacity checked FIRST (original priority),
     *     then cpuinfo_max_freq — or std::nullopt when neither is readable.
     */
    std::optional<std::pair<long, CapacitySource>> cpuCapacity(int cpu,
        const std::string &sysfs_root);

    /** @brief Shared ordering comparator (M5, one copy): value desc, id asc. */
    constexpr bool capacityOrderBefore(long value_a, int id_a, long value_b, int id_b)
    {
        return value_a != value_b ? value_a > value_b : id_a < id_b;
    }

    /**
     * @brief Physical-core representative for @p cpu (STRICT: typed unknown).
     * @return The minimum sibling id, or std::nullopt when the sibling file
     *     is MISSING. Present-but-malformed content throws
     *     PlacementTopologyReject(MalformedInput).
     */
    std::optional<int> physicalRepresentativeForCpu(int cpu,
        const std::string &sysfs_root);

    /**
     * @brief Group @p allowed_cpus by physical core and capacity-order them
     *        (STRICT receipt path — fail-closed, typed rejects).
     */
    std::vector<CorePlace> capacityOrderedCorePlaces(
        const std::vector<int> &allowed_cpus, const std::string &sysfs_root);

    /**
     * @brief Serialize places to explicit OMP_PLACES "{a},{b,c},..." syntax.
     * @throws PlacementTopologyReject for an empty list, an empty group, an
     *     id outside [0, CPU_SETSIZE), or a cpu in more than one place.
     */
    std::string serializeOmpPlaces(const std::vector<CorePlace> &places);

    /**
     * @brief STRICT parser for explicit OMP_PLACES "{a},{b,c},..." receipts.
     * @throws PlacementTopologyReject for any violation (full consumption).
     */
    std::vector<CorePlace> parseOmpPlaces(const std::string &serialized);

    /**
     * @brief STANDARDS-COMPLETE authored OMP_PLACES validator (SEPARATE from
     *        the managed receipt parser).
     *
     * COMPATIBILITY SCOPE (honest): this validator mirrors the GNU libgomp
     * PARSER as an implementation reference (env.c parse_one_place/
     * parse_places_var, pinned GCC 13.3.0 — sources bound in
     * artifacts/cb-c1-staging/libgomp-src/). It is NOT yet certified against
     * the normative OpenMP standard grammar text; that comparison is a
     * separate pending gate before any "standards-complete" claim. Envelope:
     * cpu ids are validated against [0, CPU_SETSIZE) (the libgomp cpuset
     * envelope on this platform); machines publishing ids beyond that
     * envelope are outside this validator's domain and must not be claimed
     * supported. Machine availability of in-envelope ids is the RUNTIME's
     * finalize decision (libgomp trims unavailable places with a warning —
     * see finalize_place_list) and is never certified here. Original
     * grammar facts mirrored: case-insensitive abstract names
     * threads/cores/sockets/ll_caches/numa_domains with an optional
     * "( count )" parameter and NOTHING after it; explicit place lists of
     * bare cpu ids or brace groups; per-element and per-place '!' exclusion;
     * element ranges "start:LENGTH[:stride]" (LENGTH is a COUNT, nonzero;
     * stride is SIGNED — negative strides are legal with wraparound
     * rejection); place replication "place:LENGTH[:stride]" (LENGTH in
     * [1,65535]; each copy shifts the previous copy's cpus by stride);
     * whitespace tolerated around every token; total place count <= 65536.
     *
     * Machine-independent semantics only: cpus outside [0, CPU_SETSIZE) are
     * typed IdOutOfRange here; whether a cpu is ACTUALLY available is the
     * runtime's finalize decision (libgomp trims unavailable places with a
     * warning — never certified by this validator). Exclusion semantics
     * mirror libgomp: a negated place removes the FIRST EARLIER place with
     * an equal cpu set; removing a non-existing place, or removing all
     * places, is invalid.
     *
     * This function NEVER feeds the managed receipt parser and NEVER
     * certifies libgomp warn-and-default behavior: a value failing this
     * validation is invalid BEFORE the runtime sees it.
     */
    struct AuthoredPlacesInterpretation
    {
        enum class Kind { Abstract, Explicit };
        Kind kind = Kind::Explicit;
        /// Explicit only: the deterministic ordered place sequence
        /// (replication expanded, exclusions applied).
        std::vector<std::vector<int>> deterministic_places;
        /// Abstract only: canonical lowercase name; count -1 = unspecified.
        std::string abstract_name;
        long abstract_count = -1;
        bool place_exclusion_used = false;
    };

    AuthoredPlacesInterpretation validateAuthoredOmpPlaces(
        const std::string &authored);

    /**
     * @brief Derive the local placement receipt (STRICT, fail-closed).
     * @throws PlacementTopologyReject for an empty/negative/out-of-range/
     *     duplicate admitted set, missing topology, missing capacity evidence,
     *     or a mixed-source cohort.
     */
    PlacementReceipt deriveLocalReceipt(const std::vector<int> &allowed_cpus,
        const std::string &sysfs_root);

    /** @brief One collected physical-core group: lowest admitted cpu, its
     *         capacity evidence, and the group's sorted admitted cpus. */
    struct GroupedCoreEvidence
    {
        int first = 0;
        std::pair<long, CapacitySource> evidence{0, CapacitySource::Unknown};
        std::vector<int> admitted;
    };

    /**
     * @brief Shared grouping+ordering mechanics (M4+M5, ONE copy; policy-free).
     *
     * Groups @p allowed_cpus by physical core id (resolve_core_id), sorts each
     * group's admitted cpus ascending, takes the lowest admitted cpu as the
     * ranking key, gathers capacity evidence per key (evidence_for), hands the
     * full evidence set to validate_evidence (may throw — e.g. the strict
     * cohort rule), orders groups by capacityOrderBefore, and emits each group
     * in final order.
     *
     * Admission-policy behavior lives ENTIRELY in the call-site callables:
     * the strict receipt path passes fail-closed resolvers/evidence + the
     * cohort validator; the legacy binder boundary passes its documented
     * translations. This template carries no policy of its own.
     */
    template <typename ResolveCoreId, typename EvidenceFor,
        typename ValidateEvidence, typename Emit>
    inline void groupAdmittedCoresByCapacity(const std::vector<int> &allowed_cpus,
        ResolveCoreId &&resolve_core_id, EvidenceFor &&evidence_for,
        ValidateEvidence &&validate_evidence, Emit &&emit)
    {
        std::map<int, std::vector<int>> cores;
        for (const int cpu : allowed_cpus)
            cores[resolve_core_id(cpu)].push_back(cpu);
        std::vector<GroupedCoreEvidence> records;
        records.reserve(cores.size());
        for (auto &[core_id, cpus] : cores)
        {
            (void)core_id;
            std::sort(cpus.begin(), cpus.end());
            GroupedCoreEvidence record;
            record.first = cpus.front();
            record.admitted = std::move(cpus);
            record.evidence = evidence_for(record.first);
            records.push_back(std::move(record));
        }
        validate_evidence(records);
        std::stable_sort(records.begin(), records.end(),
            [](const GroupedCoreEvidence &a, const GroupedCoreEvidence &b) {
                return capacityOrderBefore(a.evidence.first, a.first,
                    b.evidence.first, b.first);
            });
        for (const auto &record : records)
            emit(record);
    }

} // namespace llaminar2::placement
