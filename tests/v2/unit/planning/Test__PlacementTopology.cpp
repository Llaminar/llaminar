/** @brief Device-free unit tests for the OpenMP-free STRICT placement
 *         topology module.
 *
 * The module contains ONLY strict parsing (no permissive parser exists in
 * it, so the receipt admission chain structurally cannot reach one). Legacy
 * permissive bootstrap semantics live at the legacy boundary
 * (MPIBootstrapPhase's own static methods) and are covered by the legacy
 * bootstrap tests, not here.
 *
 * Verified here: strict cpu-list parser (full consumption, bounds, bounded
 * range expansion, duplicates); typed unknown topology (nullopt, no silent
 * substitution); fail-closed receipt derivation (missing topology, missing
 * capacity evidence, mixed-source cohorts all rejected); original
 * cpu_capacity-before-cpufreq priority; serialization bounds/overlap
 * validation; strict OMP_PLACES parse; full receipt roundtrip
 * parse(serialize(derive(valid))).
 */
#include <gtest/gtest.h>
#include <filesystem>
#include <sched.h>

#include <openssl/sha.h>
#include <cstdio>

#include "planning/PlacementTopology.h"

#include <fstream>
#include <set>
#include <map>

namespace
{
    using namespace llaminar2::placement;

    std::string makeSysfsRoot(const std::string &name,
        const std::map<int, std::string> &siblings,
        const std::map<int, long> &freqs)
    {
        auto root = "/tmp/placement-topology-" + name;
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        for (const auto &[cpu, sib] : siblings)
        {
            auto dir = root + "/cpu" + std::to_string(cpu) + "/topology";
            std::filesystem::create_directories(dir);
            std::ofstream(dir + "/thread_siblings_list") << sib;
            auto freq_dir = root + "/cpu" + std::to_string(cpu) + "/cpufreq";
            std::filesystem::create_directories(freq_dir);
            auto freq = freqs.find(cpu);
            if (freq != freqs.end())
                std::ofstream(freq_dir + "/cpuinfo_max_freq") << std::to_string(freq->second);
        }
        return root;
    }

    void writeCpuCapacity(const std::string &root, int cpu, long value)
    {
        std::filesystem::create_directories(root + "/cpu" + std::to_string(cpu));
        std::ofstream(root + "/cpu" + std::to_string(cpu) + "/cpu_capacity")
            << std::to_string(value);
    }
}

TEST(PlacementTopology, StrictCpuListParserAcceptsCanonicalForms)
{
    EXPECT_EQ(parseCpuListStrict("5"), (std::vector<int>{5}));
    EXPECT_EQ(parseCpuListStrict("0-3"), (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(parseCpuListStrict("0,2-4,7"), (std::vector<int>{0, 2, 3, 4, 7}));
    EXPECT_EQ(parseCpuListStrict("1023"), (std::vector<int>{1023})); // CPU_SETSIZE-1
}

TEST(PlacementTopology, StrictCpuListParserRejectsJunkAndBoundsViolations)
{
    // Junk / partial consumption
    EXPECT_THROW(parseCpuListStrict("1junk"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("0x"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("1x2"), std::invalid_argument);
    // Negative
    EXPECT_THROW(parseCpuListStrict("-1"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("0,-1"), std::invalid_argument);
    // Empty tokens
    EXPECT_THROW(parseCpuListStrict(""), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("0,"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict(",0"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("0,,2"), std::invalid_argument);
    // Malformed ranges
    EXPECT_THROW(parseCpuListStrict("3-0"), std::invalid_argument); // reversed
    EXPECT_THROW(parseCpuListStrict("0--3"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("0-3-5"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("0-"), std::invalid_argument);
    // Bounds + overflow-safe expansion
    EXPECT_THROW(parseCpuListStrict("1024"), std::invalid_argument); // CPU_SETSIZE
    EXPECT_THROW(parseCpuListStrict("999999"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("99999999999999999999"), std::invalid_argument); // ERANGE
    EXPECT_THROW(parseCpuListStrict("0-1024"), std::invalid_argument); // end >= CPU_SETSIZE
    EXPECT_EQ(parseCpuListStrict("0-1023").size(), 1024u); // full range OK
    // Duplicates
    EXPECT_THROW(parseCpuListStrict("0,0"), std::invalid_argument);
    EXPECT_THROW(parseCpuListStrict("0-2,1"), std::invalid_argument);
}

TEST(PlacementTopology, PhysicalRepresentativeTypedUnknownWhenTopologyMissing)
{
    // Typed unknown-topology result: nullopt, NOT a silent cpu fallback.
    auto root = makeSysfsRoot("missing", {}, {});
    EXPECT_FALSE(physicalRepresentativeForCpu(5, root).has_value());
}

TEST(PlacementTopology, PhysicalRepresentativeReturnsMinSiblingForSMT)
{
    auto root = makeSysfsRoot("smt", {{0, "0,72\n"}, {72, "0,72\n"}}, {});
    EXPECT_EQ(physicalRepresentativeForCpu(72, root), std::optional<int>(0));
    EXPECT_EQ(physicalRepresentativeForCpu(0, root), std::optional<int>(0));
}

TEST(PlacementTopology, PhysicalRepresentativeReturnsSelfForSingleton)
{
    auto root = makeSysfsRoot("singleton", {{5, "5\n"}}, {});
    EXPECT_EQ(physicalRepresentativeForCpu(5, root), std::optional<int>(5));
}

TEST(PlacementTopology, PhysicalRepresentativeRejectsMalformedSiblingList)
{
    // Present but unparsable: checked topology, not unknown.
    auto junk = makeSysfsRoot("junksib", {{0, "1junk\n"}}, {});
    EXPECT_THROW(physicalRepresentativeForCpu(0, junk), std::invalid_argument);
    // Present but does not contain the cpu itself: malformed topology.
    auto foreign = makeSysfsRoot("foreign", {{0, "1,2\n"}}, {});
    EXPECT_THROW(physicalRepresentativeForCpu(0, foreign), std::invalid_argument);
    // Present but empty: malformed.
    auto empty = makeSysfsRoot("emptysib", {{0, "\n"}}, {});
    EXPECT_THROW(physicalRepresentativeForCpu(0, empty), std::invalid_argument);
}

TEST(PlacementTopology, CpuCapacityPriorityRestoredCpuCapacityFirst)
{
    // ORIGINAL selection contract: cpu_capacity checked BEFORE cpufreq.
    // A cpu with BOTH fields present must yield cpu_capacity evidence.
    auto root = makeSysfsRoot("both", {{0, "0"}}, {{0, 3000}});
    writeCpuCapacity(root, 0, 1024);
    auto cap = cpuCapacity(0, root);
    ASSERT_TRUE(cap.has_value());
    EXPECT_EQ(cap->first, 1024);
    EXPECT_EQ(cap->second, CapacitySource::KernelCapacityUnits);
}

TEST(PlacementTopology, CpuCapacityTypedSourcesDistinguished)
{
    // cpuinfo_max_freq only -> kHz source
    auto freq_root = makeSysfsRoot("freq", {{0, "0"}}, {{0, 3000}});
    auto cap = cpuCapacity(0, freq_root);
    ASSERT_TRUE(cap.has_value());
    EXPECT_EQ(cap->first, 3000);
    EXPECT_EQ(cap->second, CapacitySource::CpufreqKhz);

    // cpu_capacity only -> kernel normalized units (NOT kHz)
    auto cap_root = makeSysfsRoot("cap", {{0, "0"}}, {});
    writeCpuCapacity(cap_root, 0, 1024);
    auto cap2 = cpuCapacity(0, cap_root);
    ASSERT_TRUE(cap2.has_value());
    EXPECT_EQ(cap2->second, CapacitySource::KernelCapacityUnits);

    // Neither file -> typed unknown
    auto none_root = makeSysfsRoot("none", {{0, "0"}}, {});
    EXPECT_FALSE(cpuCapacity(0, none_root).has_value());
}

TEST(PlacementTopology, CapacityOrderedCorePlacesSortsFastestFirstNonIdentity)
{
    // Single-source cohort (cpufreq only), non-identity order.
    auto root = makeSysfsRoot("nonidentity", {{0, "0"}, {1, "1"}}, {{0, 1000}, {1, 3000}});
    auto places = capacityOrderedCorePlaces({0, 1}, root);
    ASSERT_EQ(places.size(), 2u);
    EXPECT_EQ(places[0].representative, 1);
    EXPECT_EQ(places[1].representative, 0);
    EXPECT_EQ(places[0].capacity_source, CapacitySource::CpufreqKhz);
    EXPECT_EQ(places[0].capacity_value, 3000);
    EXPECT_EQ(places[1].capacity_value, 1000);
}

TEST(PlacementTopology, CapacityOrderedCorePlacesRejectsEmptyAndDuplicateAdmittedSets)
{
    auto root = makeSysfsRoot("badset", {{0, "0"}, {1, "1"}}, {{0, 2000}, {1, 3000}});
    EXPECT_THROW(capacityOrderedCorePlaces({}, root), std::invalid_argument);
    EXPECT_THROW(capacityOrderedCorePlaces({0, 0, 1}, root), std::invalid_argument);
    EXPECT_THROW(capacityOrderedCorePlaces({-1}, root), std::invalid_argument);
    EXPECT_THROW(capacityOrderedCorePlaces({1024}, root), std::invalid_argument);
}

TEST(PlacementTopology, ReceiptDerivationThrowsOnMissingTopologyNoSingletonSubstitution)
{
    // cpu 0 has topology; cpu 3 does not. The old code silently substituted
    // the cpu itself (anonymous singleton). The strict receipt path must
    // FAIL CLOSED instead.
    auto root = makeSysfsRoot("missingtopo", {{0, "0"}}, {{0, 2000}});
    EXPECT_THROW(deriveLocalReceipt({0, 3}, root), std::invalid_argument);
    // The per-cpu query is typed: nullopt for the missing cpu.
    EXPECT_FALSE(physicalRepresentativeForCpu(3, root).has_value());
}

TEST(PlacementTopology, ReceiptDerivationThrowsOnMissingCapacityEvidence)
{
    // Both cpus have topology; cpu 1 has no capacity files -> incomplete cohort.
    auto root = makeSysfsRoot("nocap", {{0, "0"}, {1, "1"}}, {{0, 2000}});
    EXPECT_THROW(deriveLocalReceipt({0, 1}, root), std::invalid_argument);
}

TEST(PlacementTopology, ReceiptDerivationRejectsMixedSourceCohort)
{
    // cpu 0 carries kernel cpu_capacity units; cpu 1 carries only cpufreq
    // kHz. The values are different units; a receipt cohort must carry ONE
    // comparable dimension — no invented normalization.
    auto root = makeSysfsRoot("mixed", {{0, "0"}, {1, "1"}}, {{1, 3000}});
    writeCpuCapacity(root, 0, 1024);
    EXPECT_THROW(deriveLocalReceipt({0, 1}, root), std::invalid_argument);
}

TEST(PlacementTopology, BothFieldsPresentCohortUsesCpuCapacityUniformly)
{
    // Every cpu has BOTH fields: original priority selects cpu_capacity for
    // all, the cohort source is uniform KernelCapacityUnits, ordering uses
    // cpu_capacity values.
    auto root = makeSysfsRoot("bothcohort", {{0, "0"}, {1, "1"}}, {{0, 3000}, {1, 4000}});
    writeCpuCapacity(root, 0, 1024);
    writeCpuCapacity(root, 1, 2048);
    auto receipt = deriveLocalReceipt({0, 1}, root);
    ASSERT_EQ(receipt.ordered_places().size(), 2u);
    EXPECT_EQ(receipt.ordered_places()[0].representative, 1); // 2048 > 1024
    EXPECT_EQ(receipt.ordered_places()[0].capacity_source, CapacitySource::KernelCapacityUnits);
    EXPECT_EQ(receipt.ordered_places()[0].capacity_value, 2048);
    EXPECT_EQ(receipt.ordered_places()[1].capacity_value, 1024);
}

TEST(PlacementTopology, SerializeOmpPlacesProducesExplicitBraceSyntax)
{
    CorePlace a; a.representative = 34; a.sibling_cpus = {34};
    CorePlace b; b.representative = 47; b.sibling_cpus = {47};
    CorePlace smt; smt.representative = 0; smt.sibling_cpus = {0, 72};
    EXPECT_EQ(serializeOmpPlaces({a, b, smt}), "{34},{47},{0,72}");
}

TEST(PlacementTopology, SerializeOmpPlacesRejectsEmptyListAndEmptyGroup)
{
    EXPECT_THROW(serializeOmpPlaces({}), std::invalid_argument);
    CorePlace empty_place;
    EXPECT_THROW(serializeOmpPlaces({empty_place}), std::invalid_argument);
}

TEST(PlacementTopology, SerializeOmpPlacesRejectsOverlappingGroups)
{
    CorePlace a; a.sibling_cpus = {0, 72};
    CorePlace b; b.sibling_cpus = {72, 1}; // 72 overlaps with a
    EXPECT_THROW(serializeOmpPlaces({a, b}), std::invalid_argument);
}

TEST(PlacementTopology, SerializeOmpPlacesRejectsNegativeCpu)
{
    CorePlace bad; bad.sibling_cpus = {-1};
    EXPECT_THROW(serializeOmpPlaces({bad}), std::invalid_argument);
}

TEST(PlacementTopology, SerializeOmpPlacesRejectsCpuBeyondCpuSetSize)
{
    // Same bound the parser enforces; the serializer must agree.
    CorePlace bad; bad.sibling_cpus = {CPU_SETSIZE};
    EXPECT_THROW(serializeOmpPlaces({bad}), std::invalid_argument);
    CorePlace way_bad; way_bad.sibling_cpus = {999999};
    EXPECT_THROW(serializeOmpPlaces({way_bad}), std::invalid_argument);
}

TEST(PlacementTopology, ParseOmpPlacesRoundtripWithSMTGroups)
{
    auto places = parseOmpPlaces("{0,72},{1,73}");
    ASSERT_EQ(places.size(), 2u);
    EXPECT_EQ(places[0].sibling_cpus, (std::vector<int>{0, 72}));
    EXPECT_EQ(places[0].representative, 0);
    EXPECT_EQ(places[1].sibling_cpus, (std::vector<int>{1, 73}));
    EXPECT_EQ(serializeOmpPlaces(places), "{0,72},{1,73}");
}

TEST(PlacementTopology, ParseOmpPlacesPartialSMTOnly72Admitted)
{
    auto places = parseOmpPlaces("{72}");
    ASSERT_EQ(places.size(), 1u);
    EXPECT_EQ(places[0].representative, 72);
    EXPECT_EQ(places[0].sibling_cpus, (std::vector<int>{72}));
}

TEST(PlacementTopology, ParseOmpPlacesPartialSMTOnly0Admitted)
{
    auto places = parseOmpPlaces("{0}");
    ASSERT_EQ(places.size(), 1u);
    EXPECT_EQ(places[0].representative, 0);
    EXPECT_EQ(places[0].sibling_cpus, (std::vector<int>{0}));
}

TEST(PlacementTopology, ParseOmpPlacesStrictFullConsumption)
{
    EXPECT_THROW(parseOmpPlaces("{0},{1}junk"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{0} "), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{0"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{0,{1}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("0},{1}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces(""), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{},{}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{},{0}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{,1}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{1,,2}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{1,}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{1junk}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{1x2}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{-1}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{999999}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{99999999999999999999}"), std::invalid_argument); // ERANGE
    EXPECT_THROW(parseOmpPlaces("{0,0}"), std::invalid_argument);
    EXPECT_THROW(parseOmpPlaces("{0},{0}"), std::invalid_argument);
}

TEST(PlacementTopology, DeriveLocalReceiptRejectsEmptyAndInvalidAdmittedSets)
{
    auto root = makeSysfsRoot("receipt", {{0, "0"}, {1, "1"}}, {{0, 2000}, {1, 3000}});
    EXPECT_THROW(deriveLocalReceipt({}, root), std::invalid_argument);
    EXPECT_THROW(deriveLocalReceipt({-1}, root), std::invalid_argument);
    EXPECT_THROW(deriveLocalReceipt({999999}, root), std::invalid_argument);
    EXPECT_THROW(deriveLocalReceipt({0, 0, 1}, root), std::invalid_argument);
}

TEST(PlacementTopology, DeriveLocalReceiptProducesContentChecksum)
{
    auto root = makeSysfsRoot("receiptck", {{0, "0"}, {1, "1"}}, {{0, 2000}, {1, 3000}});
    auto receipt = deriveLocalReceipt({0, 1}, root);
    EXPECT_EQ(receipt.ordered_places().size(), 2u);
    EXPECT_EQ(receipt.ordered_places()[0].representative, 1); // fastest first
    EXPECT_FALSE(receipt.omp_places_string().empty());
    EXPECT_EQ(receipt.content_checksum().size(), 64u); // SHA-256 content checksum
    // Deterministic: same admitted set + same topology -> same checksum.
    auto receipt2 = deriveLocalReceipt({0, 1}, root);
    EXPECT_EQ(receipt.content_checksum(), receipt2.content_checksum());
}

TEST(PlacementTopology, ReceiptRoundtripThroughRealPath)
{
    // FULL receipt path: derive -> serialize == receipt string ->
    // parse(serialize(valid)) -> every field matches (including the
    // representative invariant).
    auto root = makeSysfsRoot("roundtrip",
        {{0, "0,72"}, {72, "0,72"}, {1, "1"}, {2, "2"}},
        {{0, 1024}, {72, 1024}, {1, 2048}, {2, 2048}});
    auto receipt = deriveLocalReceipt({0, 72, 1, 2}, root);
    // Cohort: all CpufreqKhz. Order: 1 and 2 tie at 2048 (cpu asc), then 0.
    ASSERT_EQ(receipt.ordered_places().size(), 3u);
    EXPECT_EQ(receipt.omp_places_string(), "{1},{2},{0,72}");
    EXPECT_EQ(serializeOmpPlaces(receipt.ordered_places()), receipt.omp_places_string());
    auto parsed = parseOmpPlaces(receipt.omp_places_string());
    ASSERT_EQ(parsed.size(), receipt.ordered_places().size());
    for (std::size_t i = 0; i < parsed.size(); ++i)
    {
        EXPECT_EQ(parsed[i].sibling_cpus, receipt.ordered_places()[i].sibling_cpus);
        EXPECT_EQ(parsed[i].representative, receipt.ordered_places()[i].representative);
    }
}

// ---- shared mechanics + typed rejects (extraction round) ----

TEST(PlacementTopology, ParseCpuListPermissiveGrammarPreservesLegacyBootstrapSemantics)
{
    namespace p = llaminar2::placement;
    using G = p::CpuListGrammar;
    // Legacy: empty tokens skipped, stoi partial consumption, negatives pass,
    // reversed ranges swap, output sorted+deduped, empty input -> empty list.
    EXPECT_EQ(p::parseCpuList("", G::PermissiveBootstrap), (std::vector<int>{}));
    EXPECT_EQ(p::parseCpuList("1junk", G::PermissiveBootstrap), (std::vector<int>{1}));
    // Legacy reality: a single "-1" token takes the RANGE branch (find('-')
    // hits position 0) and stoi("") throws invalid_argument — negatives only
    // pass through via ranges like "0--1".
    EXPECT_THROW(p::parseCpuList("-1", G::PermissiveBootstrap), std::invalid_argument);
    EXPECT_EQ(p::parseCpuList("0--1", G::PermissiveBootstrap), (std::vector<int>{-1, 0}));
    EXPECT_EQ(p::parseCpuList("3-0", G::PermissiveBootstrap), (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(p::parseCpuList("0,0,2", G::PermissiveBootstrap), (std::vector<int>{0, 2}));
    EXPECT_EQ(p::parseCpuList("7,2-4,0", G::PermissiveBootstrap), (std::vector<int>{0, 2, 3, 4, 7}));
    // Shared bound in BOTH grammars: legacy unbounded expansion was an
    // overflow hazard; the bound is the disclosed defect fix.
    EXPECT_THROW(p::parseCpuList("0-2000000000", G::PermissiveBootstrap),
        p::PlacementTopologyReject);
    // stoi's own out_of_range still propagates in permissive mode (legacy).
    EXPECT_THROW(p::parseCpuList("99999999999999999999", G::PermissiveBootstrap),
        std::out_of_range);
}

TEST(PlacementTopology, StrictRejectionsCarryTypedReasons)
{
    namespace p = llaminar2::placement;
    auto check = [](auto &&action, p::RejectReason expected) {
        try
        {
            action();
            return ::testing::AssertionFailure() << "expected rejection";
        }
        catch (const p::PlacementTopologyReject &reject)
        {
            if (reject.reason != expected)
                return ::testing::AssertionFailure()
                    << "reason " << static_cast<int>(reject.reason)
                    << " != expected " << static_cast<int>(expected);
            return ::testing::AssertionSuccess();
        }
    };
    EXPECT_TRUE(check([] { p::parseCpuListStrict("1junk"); },
        p::RejectReason::MalformedInput));
    EXPECT_TRUE(check([] { p::parseCpuListStrict("0,0"); },
        p::RejectReason::DuplicateAdmittedId));
    EXPECT_TRUE(check([] { p::parseCpuListStrict("999999"); },
        p::RejectReason::IdOutOfRange));
    EXPECT_TRUE(check([] { p::parseOmpPlaces("{0},{0}"); },
        p::RejectReason::DuplicateAdmittedId));
    EXPECT_TRUE(check([] { p::parseOmpPlaces("{1junk}"); },
        p::RejectReason::MalformedInput));
    // Typed unknown topology stays an OPTIONAL (not a throw); the receipt
    // path turns it into a typed MissingTopology rejection.
    auto root = makeSysfsRoot("reasons", {{0, "0"}, {1, "1"}}, {{0, 2000}});
    EXPECT_TRUE(check([&] { p::deriveLocalReceipt({0, 3}, root); },
        p::RejectReason::MissingTopology));
    EXPECT_TRUE(check([&] { p::deriveLocalReceipt({0, 1}, root); },
        p::RejectReason::MissingCapacityEvidence));
    auto mixed = makeSysfsRoot("reasons-mixed", {{0, "0"}, {1, "1"}}, {{1, 3000}});
    writeCpuCapacity(mixed, 0, 1024);
    EXPECT_TRUE(check([&] { p::deriveLocalReceipt({0, 1}, mixed); },
        p::RejectReason::IncomparableCohort));
}

TEST(PlacementTopology, ReadSiblingListTypedUnknownVsMalformed)
{
    namespace p = llaminar2::placement;
    // Absent file -> typed unknown (nullopt).
    auto root = makeSysfsRoot("readsib", {{0, "0,72\n"}}, {});
    EXPECT_FALSE(p::readSiblingList(3, root).has_value());
    // Present file -> first line content.
    auto value = p::readSiblingList(0, root);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, "0,72");
}

TEST(PlacementTopology, SharedCapacityOrderComparatorValueDescThenIdAsc)
{
    namespace p = llaminar2::placement;
    EXPECT_TRUE(p::capacityOrderBefore(3000, 9, 1000, 0)); // value desc
    EXPECT_TRUE(p::capacityOrderBefore(1000, 0, 1000, 5)); // tie -> id asc
    EXPECT_FALSE(p::capacityOrderBefore(1000, 5, 1000, 0));
    EXPECT_FALSE(p::capacityOrderBefore(1000, 0, 3000, 9));
}

TEST(PlacementTopology, PermissiveBootstrapGrammarUnreachableThroughPublicReceiptApi)
{
    // Proof contract: the PermissiveBootstrap grammar is reachable ONLY at
    // the legacy binder boundary. Every PUBLIC receipt entry point must
    // reject, with typed MalformedInput, inputs that ONLY the permissive
    // grammar would accept (if any strict path leaked into the permissive
    // grammar, these would silently parse and this test fails).
    namespace p = llaminar2::placement;
    // Inputs the permissive grammar accepts (junk -> stoi prefix; trailing
    // separators skipped; empty tokens skipped):
    const char *permissive_only[] = {"1junk", "1,", ",1", "1,,2", "1junk,2x"};
    for (const char *input : permissive_only)
    {
        // parseCpuListStrict: strict grammar rejects what permissive accepts.
        try
        {
            p::parseCpuListStrict(input);
            FAIL() << "parseCpuListStrict accepted permissive-only input: " << input;
        }
        catch (const p::PlacementTopologyReject &reject)
        {
            EXPECT_EQ(reject.reason, p::RejectReason::MalformedInput);
        }
        // parseOmpPlaces: same adversarial discipline on the place grammar.
        const std::string place_input = std::string("{") + input + "}";
        try
        {
            p::parseOmpPlaces(place_input);
            FAIL() << "parseOmpPlaces accepted permissive-only input: " << place_input;
        }
        catch (const p::PlacementTopologyReject &reject)
        {
            EXPECT_EQ(reject.reason, p::RejectReason::MalformedInput);
        }
    }
    // Sibling-list content (the sysfs observation path) is parsed with the
    // strict grammar inside the strict receipt chain: junk sibling content
    // must be a typed malformed rejection, never a permissive partial parse.
    auto root = makeSysfsRoot("unreachable", {{0, "1junk"}}, {{0, 2000}});
    try
    {
        p::capacityOrderedCorePlaces({0}, root);
        FAIL() << "strict receipt chain accepted junk sibling content";
    }
    catch (const p::PlacementTopologyReject &reject)
    {
        EXPECT_EQ(reject.reason, p::RejectReason::MalformedInput);
    }
}

// ---- receipt construction barrier + transport schema (approved module scope) ----

namespace
{
    llaminar2::placement::ReceiptSessionIdentity testIdentity()
    {
        llaminar2::placement::ReceiptSessionIdentity identity;
        identity.session = "session-token-1";
        identity.host = "node-a";
        identity.rank = "0";
        identity.budget = 72;
        identity.budget_origin_explicit = true;
        return identity;
    }
}

TEST(PlacementTopology, TransportRoundtripPreservesReceiptAndIdentity)
{
    namespace p = llaminar2::placement;
    auto root = makeSysfsRoot("transport", {{0, "0,72"}, {72, "0,72"}, {1, "1"}, {2, "2"}},
        {{0, 1024}, {72, 1024}, {1, 2048}, {2, 2048}});
    const auto receipt = p::deriveLocalReceipt({0, 72, 1, 2}, root);
    // The transport invariant budget<=place_count: this derivation has
    // 3 places, so the identity budget must be within [1, 3].
    auto identity = testIdentity();
    identity.budget = 3;
    const auto transport = p::serializeReceiptTransport(receipt, identity);
    // Fixed field order, prefix visible for audit.
    EXPECT_EQ(transport.substr(0, 3), "v=1");
    EXPECT_NE(transport.find(";session=session-token-1;"), std::string::npos);
    EXPECT_NE(transport.find(";host=node-a;"), std::string::npos);
    EXPECT_NE(transport.find(";rank=0;"), std::string::npos);
    EXPECT_NE(transport.find(";budget=3;"), std::string::npos);
    EXPECT_NE(transport.find(";budget_origin=Explicit;"), std::string::npos);
    p::PlacementTopologyReject reject(p::RejectReason::MalformedInput, "unset");
    const auto parsed = p::parseReceiptTransport(transport, reject);
    ASSERT_TRUE(parsed.has_value()) << reject.what();
    // Roundtrip: every place (including typed capacity evidence) survives.
    ASSERT_EQ(parsed->ordered_places().size(), receipt.ordered_places().size());
    for (std::size_t i = 0; i < receipt.ordered_places().size(); ++i)
    {
        EXPECT_EQ(parsed->ordered_places()[i].sibling_cpus,
            receipt.ordered_places()[i].sibling_cpus);
        EXPECT_EQ(parsed->ordered_places()[i].representative,
            receipt.ordered_places()[i].representative);
        EXPECT_EQ(parsed->ordered_places()[i].capacity_source,
            receipt.ordered_places()[i].capacity_source);
        EXPECT_EQ(parsed->ordered_places()[i].capacity_value,
            receipt.ordered_places()[i].capacity_value);
    }
    EXPECT_EQ(parsed->omp_places_string(), receipt.omp_places_string());
}

TEST(PlacementTopology, TransportRejectsAnyByteTampering)
{
    namespace p = llaminar2::placement;
    auto root = makeSysfsRoot("tamper", {{0, "0"}, {1, "1"}}, {{0, 2000}, {1, 3000}});
    const auto receipt = p::deriveLocalReceipt({0, 1}, root);
    auto transport = p::serializeReceiptTransport(receipt, testIdentity());
    // Flip the budget digit: checksum must fail (corruption, never tolerated).
    auto tampered = transport;
    const auto budget_pos = tampered.find(";budget=");
    ASSERT_NE(budget_pos, std::string::npos);
    tampered[budget_pos + 8] = (tampered[budget_pos + 8] == '7') ? '8' : '7';
    p::PlacementTopologyReject reject(p::RejectReason::MalformedInput, "unset");
    EXPECT_FALSE(p::parseReceiptTransport(tampered, reject).has_value());
    EXPECT_EQ(reject.reason, p::RejectReason::ChecksumMismatch);
    // Flip a place cpu: same result.
    auto tampered2 = transport;
    const auto places_pos = tampered2.find(";places=");
    ASSERT_NE(places_pos, std::string::npos);
    tampered2[places_pos + 8] = (tampered2[places_pos + 8] == '1') ? '3' : '1';
    p::PlacementTopologyReject reject2(p::RejectReason::MalformedInput, "unset");
    EXPECT_FALSE(p::parseReceiptTransport(tampered2, reject2).has_value());
    EXPECT_EQ(reject2.reason, p::RejectReason::ChecksumMismatch);
    // Truncate one byte: checksum mismatch.
    p::PlacementTopologyReject reject3(p::RejectReason::MalformedInput, "unset");
    EXPECT_FALSE(p::parseReceiptTransport(transport.substr(0, transport.size() - 1), reject3).has_value());
    EXPECT_EQ(reject3.reason, p::RejectReason::ChecksumMismatch);
}

TEST(PlacementTopology, TransportRejectsSchemaViolationsWithTypedReasons)
{
    namespace p = llaminar2::placement;
    auto root = makeSysfsRoot("schema", {{0, "0"}, {1, "1"}}, {{0, 2000}, {1, 3000}});
    const auto receipt = p::deriveLocalReceipt({0, 1}, root);
    const auto transport = p::serializeReceiptTransport(receipt, testIdentity());
    p::PlacementTopologyReject reject(p::RejectReason::MalformedInput, "unset");
    auto check = [&](const std::string &bad, p::RejectReason expected) {
        p::PlacementTopologyReject local(p::RejectReason::MalformedInput, "unset");
        const auto parsed = p::parseReceiptTransport(bad, local);
        EXPECT_FALSE(parsed.has_value()) << bad;
        EXPECT_EQ(local.reason, expected) << bad;
    };
    check("", p::RejectReason::MalformedInput);                                  // empty
    check("v=2;session=s;host=h;rank=0;budget=1;budget_origin=AutoLocal;"
          "admitted=0,1;place_count=2;places={0},{1};"
          "capacity_source=CpufreqKhz;capacity_values=1,2;checksum=x",
        p::RejectReason::ChecksumMismatch);                                       // wrong version + bad checksum
    // Field order violation (session/host swapped) -> order rejection before checksum matters
    {
        const auto session_pos = transport.find("session=");
        const auto host_pos = transport.find("host=");
        // Rebuild with swapped identity values (same field order but host/session swapped)
        std::string swapped = transport;
        const auto session_value = transport.substr(session_pos + 8, transport.find(';', session_pos) - session_pos - 8);
        const auto host_value = transport.substr(host_pos + 5, transport.find(';', host_pos) - host_pos - 5);
        swapped.replace(session_pos + 8, session_value.size(), host_value);
        swapped.replace(host_pos + 5, host_value.size(), session_value);
        p::PlacementTopologyReject local(p::RejectReason::MalformedInput, "unset");
        EXPECT_FALSE(p::parseReceiptTransport(swapped, local).has_value());
        EXPECT_EQ(local.reason, p::RejectReason::ChecksumMismatch);
    }
    // Charset violation in identity field (spaces not allowed) -> MalformedInput even before checksum matters? No: checksum covers it, so tampering yields ChecksumMismatch; a HAND-BUILT transport with valid checksum and bad charset must be rejected. Build one:
    {
        std::string crafted = "v=1;session=bad session;host=h;rank=0;budget=1;"
            "budget_origin=AutoLocal;admitted=0;place_count=1;places={0};"
            "capacity_source=CpufreqKhz;capacity_values=1";
        // append a valid checksum for the crafted payload to isolate the charset check
        unsigned char sha[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char *>(crafted.data()), crafted.size(), sha);
        char hex[2 * SHA256_DIGEST_LENGTH + 1];
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", sha[i]);
        crafted += std::string(";checksum=") + hex;
        p::PlacementTopologyReject local(p::RejectReason::MalformedInput, "unset");
        EXPECT_FALSE(p::parseReceiptTransport(crafted, local).has_value());
        EXPECT_EQ(local.reason, p::RejectReason::MalformedInput);
    }
    // Non-partition: places do not cover admitted (hand-built, valid checksum)
    {
        std::string crafted = "v=1;session=s;host=h;rank=0;budget=1;"
            "budget_origin=AutoLocal;admitted=0,1,2;place_count=2;places={0},{1};"
            "capacity_source=CpufreqKhz;capacity_values=1,2";
        unsigned char sha[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char *>(crafted.data()), crafted.size(), sha);
        char hex[2 * SHA256_DIGEST_LENGTH + 1];
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", sha[i]);
        crafted += std::string(";checksum=") + hex;
        p::PlacementTopologyReject local(p::RejectReason::MalformedInput, "unset");
        EXPECT_FALSE(p::parseReceiptTransport(crafted, local).has_value());
        EXPECT_EQ(local.reason, p::RejectReason::MalformedInput);
    }
    // Duplicate across places (hand-built, valid checksum)
    {
        std::string crafted = "v=1;session=s;host=h;rank=0;budget=1;"
            "budget_origin=AutoLocal;admitted=0;place_count=2;places={0},{0};"
            "capacity_source=CpufreqKhz;capacity_values=1,2";
        unsigned char sha[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char *>(crafted.data()), crafted.size(), sha);
        char hex[2 * SHA256_DIGEST_LENGTH + 1];
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", sha[i]);
        crafted += std::string(";checksum=") + hex;
        p::PlacementTopologyReject local(p::RejectReason::MalformedInput, "unset");
        EXPECT_FALSE(p::parseReceiptTransport(crafted, local).has_value());
        EXPECT_EQ(local.reason, p::RejectReason::DuplicateAdmittedId);
    }
    // Non-positive budget (hand-built, valid checksum)
    {
        std::string crafted = "v=1;session=s;host=h;rank=0;budget=0;"
            "budget_origin=AutoLocal;admitted=0;place_count=1;places={0};"
            "capacity_source=CpufreqKhz;capacity_values=1";
        unsigned char sha[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char *>(crafted.data()), crafted.size(), sha);
        char hex[2 * SHA256_DIGEST_LENGTH + 1];
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", sha[i]);
        crafted += std::string(";checksum=") + hex;
        p::PlacementTopologyReject local(p::RejectReason::MalformedInput, "unset");
        EXPECT_FALSE(p::parseReceiptTransport(crafted, local).has_value());
        EXPECT_EQ(local.reason, p::RejectReason::MalformedInput);
    }
    // Serializer-side guards: empty/charset-invalid identity, non-positive budget
    EXPECT_THROW(p::serializeReceiptTransport(receipt,
        [] { auto id = testIdentity(); id.session = ""; return id; }()),
        std::invalid_argument);
    EXPECT_THROW(p::serializeReceiptTransport(receipt,
        [] { auto id = testIdentity(); id.host = "bad host"; return id; }()),
        std::invalid_argument);
    EXPECT_THROW(p::serializeReceiptTransport(receipt,
        [] { auto id = testIdentity(); id.budget = 0; return id; }()),
        std::invalid_argument);
}

TEST(PlacementTopology, ReceiptConstructionBarrier)
{
    namespace p = llaminar2::placement;
    // The passkey type is private and the default constructor is deleted:
    // these lines would not compile (documented barrier; compile-time proof
    // lives in code review, runtime proof = no public path exists):
    //   p::PlacementReceipt r;                                     // deleted
    //   p::PlacementReceipt r{{}};                                 // passkey is private
    // The ONLY public producers are the two validating factories.
    auto root = makeSysfsRoot("barrier", {{0, "0"}}, {{0, 2000}});
    const auto receipt = p::deriveLocalReceipt({0}, root);
    const auto copy = receipt; // copies of a VALID receipt are valid
    EXPECT_EQ(copy.ordered_places().size(), 1u);
    // parseReceiptTransport of a corrupted transport never yields a receipt.
    p::PlacementTopologyReject reject(p::RejectReason::MalformedInput, "unset");
    EXPECT_FALSE(p::parseReceiptTransport("garbage", reject).has_value());
}

TEST(PlacementTopology, TransportRejectsBudgetExceedingPlaceCount)
{
    namespace p = llaminar2::placement;
    auto root = makeSysfsRoot("budgetcap", {{0, "0"}, {1, "1"}}, {{0, 2000}, {1, 3000}});
    const auto receipt = p::deriveLocalReceipt({0, 1}, root);
    auto identity = testIdentity();
    identity.budget = 3; // > place_count (2) — managed invariant violation
    p::PlacementTopologyReject reject(p::RejectReason::MalformedInput, "unset");
    EXPECT_FALSE(p::parseReceiptTransport(
        p::serializeReceiptTransport(receipt, identity), reject).has_value());
    EXPECT_EQ(reject.reason, p::RejectReason::MalformedInput);
    // budget == place_count is the managed boundary and is accepted.
    identity.budget = 2;
    p::PlacementTopologyReject accept(p::RejectReason::MalformedInput, "unset");
    EXPECT_TRUE(p::parseReceiptTransport(
        p::serializeReceiptTransport(receipt, identity), accept).has_value());
}

// ---- standards-complete authored OMP_PLACES validator (grammar per libgomp env.c) ----

TEST(PlacementTopology, AuthoredGrammarExplicitListsWithLibgompRangeSemantics)
{
    namespace p = llaminar2::placement;
    // Ranges are start:LENGTH[:stride] (LENGTH is a COUNT, not an upper
    // bound) — the v5-draft lower:upper reading was wrong.
    // A range fills ONE place with multiple cpus (libgomp
    // gomp_affinity_add_cpus adds into the same place set).
    auto v = p::validateAuthoredOmpPlaces("{0:4}");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0, 1, 2, 3}}));
    v = p::validateAuthoredOmpPlaces("{0:4:2}");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0, 2, 4, 6}}));
    // Negative stride descends (libgomp gomp_affinity_add_cpus allows it).
    v = p::validateAuthoredOmpPlaces("{7:4:-2}");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{1, 3, 5, 7}}));
    // Bare numbers are legal single-cpu places.
    v = p::validateAuthoredOmpPlaces("0,1,2");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0}, {1}, {2}}));
    // Mixed braces + bare + whitespace tolerance.
    v = p::validateAuthoredOmpPlaces(" { 0 , 1 } , 2 ");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0, 1}, {2}}));
}

TEST(PlacementTopology, AuthoredGrammarDuplicatesReplicationAndExclusion)
{
    namespace p = llaminar2::placement;
    // Duplicate places are LEGAL in the authored grammar (oversubscription
    // binding by design) — the managed parser's uniqueness does NOT apply.
    auto v = p::validateAuthoredOmpPlaces("{0},{0}");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0}, {0}}));
    // Place replication "{place}:LENGTH[:stride]": LENGTH copies, each copy
    // SHIFTED by stride (gomp_affinity_copy_place: "{0}:3" -> {0},{1},{2}).
    v = p::validateAuthoredOmpPlaces("{0}:3");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0}, {1}, {2}}));
    v = p::validateAuthoredOmpPlaces("{0}:3:2");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0}, {2}, {4}}));
    v = p::validateAuthoredOmpPlaces("{0,1}:2:1");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0, 1}, {1, 2}}));
    // Element exclusion requires the cpu to be IN the place (libgomp
    // remove_cpu errors on a cpu not in the set): '{0,1,!1}' -> {0};
    // '{0,!1}' is INVALID (1 is not in the built place).
    v = p::validateAuthoredOmpPlaces("{0,1,!1}");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0}}));
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0,!1}"), p::PlacementTopologyReject);
    // Place exclusion: '!{...}' removes the FIRST EARLIER place with an
    // equal cpu set.
    v = p::validateAuthoredOmpPlaces("{0},{1},!{1}");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0}}));
    // Bare-number exclusion.
    v = p::validateAuthoredOmpPlaces("0,1,!1");
    EXPECT_EQ(v.deterministic_places, (std::vector<std::vector<int>>{{0}}));
    // Excluding a non-existing place is invalid (libgomp errors).
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0},!{2}"), p::PlacementTopologyReject);
    // Excluding everything is invalid ("All places have been removed").
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0},!{0}"), p::PlacementTopologyReject);
    // Element exclusion of a cpu the group does not contain is invalid.
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0,!5}"), p::PlacementTopologyReject);
}

TEST(PlacementTopology, AuthoredGrammarAbstractNamesCaseInsensitiveWithParameter)
{
    namespace p = llaminar2::placement;
    auto v = p::validateAuthoredOmpPlaces("cores");
    EXPECT_EQ(v.kind, p::AuthoredPlacesInterpretation::Kind::Abstract);
    EXPECT_EQ(v.abstract_name, "cores");
    EXPECT_EQ(v.abstract_count, -1);
    v = p::validateAuthoredOmpPlaces("THREADS(2)");
    EXPECT_EQ(v.abstract_name, "threads");
    EXPECT_EQ(v.abstract_count, 2);
    v = p::validateAuthoredOmpPlaces("NUMA_DOMAINS");
    EXPECT_EQ(v.abstract_name, "numa_domains");
    v = p::validateAuthoredOmpPlaces("sockets(4)");
    EXPECT_EQ(v.abstract_name, "sockets");
    EXPECT_EQ(v.abstract_count, 4);
    v = p::validateAuthoredOmpPlaces("ll_caches");
    EXPECT_EQ(v.abstract_name, "ll_caches");
    // Trailing content after an abstract form is invalid (libgomp requires END).
    EXPECT_THROW(p::validateAuthoredOmpPlaces("cores(2),{0}"), p::PlacementTopologyReject);
    EXPECT_THROW(p::validateAuthoredOmpPlaces("coresx"), p::PlacementTopologyReject);
    EXPECT_THROW(p::validateAuthoredOmpPlaces("threads(2"), p::PlacementTopologyReject);
}

TEST(PlacementTopology, AuthoredGrammarRejectsMalformedAndOutOfRange)
{
    namespace p = llaminar2::placement;
    EXPECT_THROW(p::validateAuthoredOmpPlaces(""), p::PlacementTopologyReject);
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{"), p::PlacementTopologyReject);
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0"), p::PlacementTopologyReject);
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{,}"), p::PlacementTopologyReject);
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0,}"), p::PlacementTopologyReject);
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0:0}"), p::PlacementTopologyReject);      // zero length
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0:99999999999999999999}"), p::PlacementTopologyReject); // ERANGE
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0:-1}"), p::PlacementTopologyReject);     // length unsigned
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0:2:-1}"), p::PlacementTopologyReject);   // element negated range
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0},{1},baddata"), p::PlacementTopologyReject); // junk after place
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{1024}"), p::PlacementTopologyReject);     // >= CPU_SETSIZE
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0:65537}"), p::PlacementTopologyReject);  // range wraps past representable
    // Place replication bounds: LENGTH < 65536 AND every shifted copy must
    // stay representable (libgomp copy_place errors on out-of-range shifts,
    // so "{0}:65535" is invalid on this machine; "{0}:1024" is the stride-1
    // maximum from cpu 0).
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0}:65536"), p::PlacementTopologyReject);
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{0}:65535"), p::PlacementTopologyReject);
    EXPECT_NO_THROW(p::validateAuthoredOmpPlaces("{0}:1024"));
    // Replication shift out of range is rejected (copy_place wraparound).
    EXPECT_THROW(p::validateAuthoredOmpPlaces("{1023}:2:1"), p::PlacementTopologyReject);
}
