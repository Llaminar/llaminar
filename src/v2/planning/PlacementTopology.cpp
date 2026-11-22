/**
 * @file PlacementTopology.cpp
 * @brief OpenMP-free placement authority. Raw mechanics exist ONCE here; the
 *        strict receipt admission is fail-closed with typed rejects. The
 *        legacy bootstrap binder delegates every mechanic to this module and
 *        keeps only its own documented boundary translations.
 */
#include "planning/PlacementTopology.h"

#include <cerrno>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

#include <sched.h>

#include <openssl/sha.h>

namespace llaminar2::placement
{
    namespace
    {
        constexpr long kCpuSetSize = CPU_SETSIZE;

        // Strict bounded long: full strtol consumption, ERANGE checked,
        // bounds enforced. The single numeric-parse primitive.
        long parseStrictLong(const std::string &token, const char *context,
            long lo, long hi)
        {
            if (token.empty())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    std::string(context) + ": empty token");
            errno = 0;
            char *end = nullptr;
            const long value = strtol(token.c_str(), &end, 10);
            if (end == token.c_str() || *end != '\0')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    std::string(context) + ": malformed id '" + token + "'");
            if (errno == ERANGE || value < lo || value > hi)
                throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                    std::string(context) + ": id out of range '" + token + "'");
            return value;
        }

        // Strict single token: "N" or "N-M", N <= M, full consumption, bounds.
        void strictParseCpuToken(const std::string &token, std::vector<int> &out,
            std::set<int> &seen)
        {
            if (token.empty())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "cpu list: empty token");
            const std::size_t dash = token.find('-', 1);
            long start = 0;
            long end = 0;
            if (dash == std::string::npos)
            {
                start = end = parseStrictLong(token, "cpu list", 0, kCpuSetSize - 1);
            }
            else
            {
                start = parseStrictLong(token.substr(0, dash), "cpu list", 0, kCpuSetSize - 1);
                end = parseStrictLong(token.substr(dash + 1), "cpu list", 0, kCpuSetSize - 1);
                if (end < start)
                    throw PlacementTopologyReject(RejectReason::MalformedInput,
                        "cpu list: reversed range '" + token + "'");
            }
            // Shared bounded expansion: 64-bit width, CPU_SETSIZE cap in BOTH
            // grammars (the legacy unbounded expansion was an overflow hazard;
            // kernel-provided lists never exceed CPU_SETSIZE).
            const long width = end - start + 1;
            if (width > kCpuSetSize)
                throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                    "cpu list: range '" + token + "' exceeds CPU_SETSIZE");
            for (long cpu = start; cpu <= end; ++cpu)
            {
                if (!seen.insert(static_cast<int>(cpu)).second)
                    throw PlacementTopologyReject(RejectReason::DuplicateAdmittedId,
                        "cpu list: duplicate cpu " + std::to_string(cpu));
                out.push_back(static_cast<int>(cpu));
            }
        }

        // Permissive single token: legacy bootstrap semantics (stoi partial
        // consumption, negatives pass through, reversed ranges swap). Only an
        // over-wide expansion is rejected (shared bound; disclosed defect fix).
        void permissiveParseCpuToken(const std::string &token, std::vector<int> &out)
        {
            auto dash = token.find('-');
            if (dash == std::string::npos)
            {
                out.push_back(std::stoi(token));
                return;
            }
            int start = std::stoi(token.substr(0, dash));
            int end = std::stoi(token.substr(dash + 1));
            if (end < start) std::swap(start, end);
            const long width = static_cast<long>(end) - static_cast<long>(start) + 1;
            if (width > kCpuSetSize)
                throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                    "cpu list: range '" + token + "' exceeds CPU_SETSIZE");
            for (long cpu = start; cpu <= end; ++cpu)
                out.push_back(static_cast<int>(cpu));
        }
    } // namespace

    std::optional<std::string> readSiblingList(int cpu, const std::string &sysfs_root)
    {
        if (cpu < 0 || cpu >= kCpuSetSize)
            throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                "topology: cpu id out of range: " + std::to_string(cpu));
        std::ifstream siblings_file(sysfs_root + "/cpu" + std::to_string(cpu) +
            "/topology/thread_siblings_list");
        if (!siblings_file.is_open())
            return std::nullopt; // typed unknown topology — caller decides
        std::string siblings;
        std::getline(siblings_file, siblings);
        return siblings;
    }

    std::vector<int> parseCpuList(const std::string &cpulist, CpuListGrammar grammar)
    {
        if (grammar == CpuListGrammar::StrictReceipt)
        {
            if (cpulist.empty())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "cpu list: empty input");
            if (cpulist.back() == ',')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "cpu list: trailing separator");
            std::vector<int> cpus;
            std::set<int> seen;
            std::stringstream ss(cpulist);
            std::string token;
            while (std::getline(ss, token, ','))
                strictParseCpuToken(token, cpus, seen);
            return cpus;
        }
        // PermissiveBootstrap: legacy semantics preserved at the legacy
        // boundary — empty tokens skipped, stoi partial consumption,
        // negatives passed through, reversed ranges swapped, output
        // sorted+deduped, empty input yields an empty list.
        std::vector<int> cpus;
        std::stringstream ss(cpulist);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            if (token.empty()) continue;
            permissiveParseCpuToken(token, cpus);
        }
        std::sort(cpus.begin(), cpus.end());
        cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
        return cpus;
    }

    std::optional<std::pair<long, CapacitySource>> cpuCapacity(int cpu,
        const std::string &sysfs_root)
    {
        if (cpu < 0 || cpu >= kCpuSetSize)
            throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                "capacity: cpu id out of range: " + std::to_string(cpu));
        const std::string root = sysfs_root + "/cpu" + std::to_string(cpu);
        // ORIGINAL priority preserved: cpu_capacity is checked BEFORE cpufreq.
        {
            std::ifstream file(root + "/cpu_capacity");
            long value = 0;
            if (file >> value && value > 0)
                return std::make_pair(value, CapacitySource::KernelCapacityUnits);
        }
        {
            std::ifstream file(root + "/cpufreq/cpuinfo_max_freq");
            long value = 0;
            if (file >> value && value > 0)
                return std::make_pair(value, CapacitySource::CpufreqKhz);
        }
        return std::nullopt;
    }

    std::optional<int> physicalRepresentativeForCpu(int cpu,
        const std::string &sysfs_root)
    {
        auto siblings = readSiblingList(cpu, sysfs_root);
        if (!siblings.has_value())
            return std::nullopt; // typed unknown topology — caller decides
        if (siblings->empty())
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "topology: empty sibling list for cpu " + std::to_string(cpu));
        const auto sibling_cpus = parseCpuListStrict(*siblings);
        if (std::find(sibling_cpus.begin(), sibling_cpus.end(), cpu) ==
            sibling_cpus.end())
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "topology: sibling list of cpu " + std::to_string(cpu) +
                " does not contain the cpu itself");
        return *std::min_element(sibling_cpus.begin(), sibling_cpus.end());
    }

    std::vector<CorePlace> capacityOrderedCorePlaces(
        const std::vector<int> &allowed_cpus, const std::string &sysfs_root)
    {
        if (allowed_cpus.empty())
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "placement: empty admitted CPU set");
        std::set<int> admitted;
        for (const int cpu : allowed_cpus)
        {
            if (cpu < 0 || cpu >= kCpuSetSize)
                throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                    "placement: cpu id out of range: " + std::to_string(cpu));
            if (!admitted.insert(cpu).second)
                throw PlacementTopologyReject(RejectReason::DuplicateAdmittedId,
                    "placement: duplicate admitted cpu id: " + std::to_string(cpu));
        }
        // Strict receipt path: fail-closed resolvers + cohort validation;
        // the grouping/ordering mechanics are the shared skeleton.
        std::vector<CorePlace> places;
        groupAdmittedCoresByCapacity(allowed_cpus,
            [&](const int cpu) -> int {
                auto representative = physicalRepresentativeForCpu(cpu, sysfs_root);
                if (!representative.has_value())
                    throw PlacementTopologyReject(RejectReason::MissingTopology,
                        "placement: missing sysfs topology for cpu " +
                        std::to_string(cpu) +
                        " — unknown topology is not admissible for a trusted receipt");
                return *representative;
            },
            [&](const int first) -> std::pair<long, CapacitySource> {
                auto evidence = cpuCapacity(first, sysfs_root);
                if (!evidence.has_value())
                    throw PlacementTopologyReject(RejectReason::MissingCapacityEvidence,
                        "placement: no capacity evidence for cpu " +
                        std::to_string(first) + " — incomplete cohort");
                return *evidence;
            },
            [&](const std::vector<GroupedCoreEvidence> &records) {
                for (std::size_t i = 1; i < records.size(); ++i)
                    if (records[i].evidence.second != records[0].evidence.second)
                        throw PlacementTopologyReject(RejectReason::IncomparableCohort,
                            "placement: mixed capacity evidence sources across the cohort"
                            " (kernel cpu_capacity units vs cpufreq kHz) — incomparable,"
                            " no invented normalization");
            },
            [&](const GroupedCoreEvidence &record) {
                CorePlace place;
                place.representative = record.first;
                place.sibling_cpus = record.admitted;
                place.capacity_source = record.evidence.second;
                place.capacity_value = record.evidence.first;
                places.push_back(std::move(place));
            });
        return places;
    }

    std::string serializeOmpPlaces(const std::vector<CorePlace> &places)
    {
        if (places.empty())
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "placement: cannot serialize an empty place list");
        std::set<int> all_cpus;
        for (const auto &place : places)
        {
            if (place.sibling_cpus.empty())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "placement: cannot serialize an empty place group");
            for (const int cpu : place.sibling_cpus)
            {
                if (cpu < 0 || cpu >= kCpuSetSize)
                    throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                        "placement: cpu id out of range: " + std::to_string(cpu));
                if (!all_cpus.insert(cpu).second)
                    throw PlacementTopologyReject(RejectReason::DuplicateAdmittedId,
                        "placement: cpu " + std::to_string(cpu) +
                        " appears in more than one place (duplicate/overlap)");
            }
        }
        std::string result;
        for (std::size_t i = 0; i < places.size(); ++i)
        {
            if (i) result += ",";
            result += "{";
            for (std::size_t k = 0; k < places[i].sibling_cpus.size(); ++k)
            {
                if (k) result += ",";
                result += std::to_string(places[i].sibling_cpus[k]);
            }
            result += "}";
        }
        return result;
    }

    std::vector<CorePlace> parseOmpPlaces(const std::string &serialized)
    {
        if (serialized.empty())
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "OMP_PLACES: empty string");
        std::vector<CorePlace> places;
        std::set<int> all_cpus;
        std::size_t pos = 0;
        while (pos < serialized.size())
        {
            if (serialized[pos] != '{')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "OMP_PLACES: expected '{' at offset " + std::to_string(pos));
            const auto close = serialized.find('}', pos);
            if (close == std::string::npos)
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "OMP_PLACES: missing '}' after offset " + std::to_string(pos));
            const std::string inner = serialized.substr(pos + 1, close - pos - 1);
            if (inner.empty())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "OMP_PLACES: empty place group");
            if (inner.back() == ',')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "OMP_PLACES: trailing separator in place group");
            if (inner.front() == ',')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "OMP_PLACES: leading separator in place group");
            CorePlace place;
            std::stringstream ss(inner);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                if (token.empty())
                    throw PlacementTopologyReject(RejectReason::MalformedInput,
                        "OMP_PLACES: empty cpu token in place group");
                const long val = parseStrictLong(token, "OMP_PLACES", 0, kCpuSetSize - 1);
                if (!all_cpus.insert(static_cast<int>(val)).second)
                    throw PlacementTopologyReject(RejectReason::DuplicateAdmittedId,
                        "OMP_PLACES: duplicate cpu " + std::to_string(val));
                place.sibling_cpus.push_back(static_cast<int>(val));
            }
            if (place.sibling_cpus.empty())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "OMP_PLACES: empty place group");
            std::sort(place.sibling_cpus.begin(), place.sibling_cpus.end());
            place.representative = place.sibling_cpus.front();
            places.push_back(std::move(place));
            pos = close + 1;
            if (pos < serialized.size() && serialized[pos] != ',')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "OMP_PLACES: expected ',' after '}' at offset " + std::to_string(pos));
            if (pos < serialized.size()) ++pos;
        }
        if (places.empty())
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "OMP_PLACES: no places parsed");
        return places;
    }

    PlacementReceipt deriveLocalReceipt(const std::vector<int> &allowed_cpus,
        const std::string &sysfs_root)
    {
        if (allowed_cpus.empty())
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "placement: empty admitted CPU set");
        std::set<int> admitted;
        for (const int cpu : allowed_cpus)
        {
            if (cpu < 0 || cpu >= kCpuSetSize)
                throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                    "placement: cpu id out of range: " + std::to_string(cpu));
            if (!admitted.insert(cpu).second)
                throw PlacementTopologyReject(RejectReason::DuplicateAdmittedId,
                    "placement: duplicate admitted cpu id: " + std::to_string(cpu));
        }
        auto ordered = capacityOrderedCorePlaces(allowed_cpus, sysfs_root);
        const auto places_string = serializeOmpPlaces(ordered);
        unsigned char sha[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char *>(places_string.data()),
            places_string.size(), sha);
        char hex[2 * SHA256_DIGEST_LENGTH + 1];
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
            snprintf(hex + 2 * i, 3, "%02x", sha[i]);
        return PlacementReceipt(PlacementReceipt::ConstructionPasskey{},
            std::move(ordered), places_string, hex);
    }

    namespace
    {
        // Transport charset for identity fields: [A-Za-z0-9._-]
        bool transportIdentityCharsetOk(const std::string &value)
        {
            if (value.empty()) return false;
            for (const char c : value)
            {
                const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
                if (!ok) return false;
            }
            return true;
        }

        std::string sha256Hex(const std::string &bytes)
        {
            unsigned char sha[SHA256_DIGEST_LENGTH];
            SHA256(reinterpret_cast<const unsigned char *>(bytes.data()),
                bytes.size(), sha);
            char hex[2 * SHA256_DIGEST_LENGTH + 1];
            for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
                snprintf(hex + 2 * i, 3, "%02x", sha[i]);
            return hex;
        }
    } // namespace

    std::string serializeReceiptTransport(const PlacementReceipt &receipt,
        const ReceiptSessionIdentity &identity)
    {
        if (!transportIdentityCharsetOk(identity.session))
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "transport: session token empty or charset-invalid");
        if (!transportIdentityCharsetOk(identity.host))
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "transport: host empty or charset-invalid");
        if (!transportIdentityCharsetOk(identity.rank))
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "transport: rank empty or charset-invalid");
        if (identity.budget <= 0)
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "transport: budget must be positive");
        if (receipt.ordered_places().empty())
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "transport: receipt carries no places");
        const CapacitySource source = receipt.ordered_places().front().capacity_source;
        if (source == CapacitySource::Unknown)
            throw PlacementTopologyReject(RejectReason::MissingCapacityEvidence,
                "transport: receipt carries Unknown capacity source");
        // The places ARE the partition; admitted is their ascending union.
        std::set<int> union_cpus;
        for (const auto &place : receipt.ordered_places())
        {
            if (place.sibling_cpus.empty() ||
                place.sibling_cpus.front() != place.representative)
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "transport: representative invariant violated");
            for (const int cpu : place.sibling_cpus)
                if (!union_cpus.insert(cpu).second)
                    throw PlacementTopologyReject(RejectReason::DuplicateAdmittedId,
                        "transport: place list is not a partition (duplicate cpu " +
                        std::to_string(cpu) + ")");
            if (place.capacity_source != source)
                throw PlacementTopologyReject(RejectReason::IncomparableCohort,
                    "transport: mixed capacity sources across the receipt");
        }
        std::string admitted_list;
        for (const int cpu : union_cpus)
        {
            if (!admitted_list.empty()) admitted_list += ",";
            admitted_list += std::to_string(cpu);
        }
        std::string values;
        for (const auto &place : receipt.ordered_places())
        {
            if (!values.empty()) values += ",";
            values += std::to_string(place.capacity_value);
        }
        std::string transport;
        transport += "v=1";
        transport += ";session=" + identity.session;
        transport += ";host=" + identity.host;
        transport += ";rank=" + identity.rank;
        transport += ";budget=" + std::to_string(identity.budget);
        transport += identity.budget_origin_explicit
            ? ";budget_origin=Explicit" : ";budget_origin=AutoLocal";
        transport += ";admitted=" + admitted_list;
        transport += ";place_count=" + std::to_string(receipt.ordered_places().size());
        transport += ";places=" + receipt.omp_places_string();
        transport += ";capacity_source=";
        transport += source == CapacitySource::KernelCapacityUnits
            ? "KernelCapacityUnits" : "CpufreqKhz";
        transport += ";capacity_values=" + values;
        transport += ";checksum=" + sha256Hex(transport);
        return transport;
    }

    std::optional<PlacementTransportView> parseReceiptTransportView(
        const std::string &transport, PlacementTopologyReject &reject)
    {
        const auto fail = [&reject](RejectReason reason, const std::string &message) {
            reject = PlacementTopologyReject(reason, message);
            return std::nullopt;
        };
        const auto checksum_marker = std::string(";checksum=");
        const auto checksum_pos = transport.rfind(checksum_marker);
        if (checksum_pos == std::string::npos)
            return fail(RejectReason::MalformedInput, "transport: missing checksum field");
        const std::string payload = transport.substr(0, checksum_pos);
        const std::string received_checksum = transport.substr(
            checksum_pos + checksum_marker.size());
        if (received_checksum != sha256Hex(payload))
            return fail(RejectReason::ChecksumMismatch,
                "transport: content checksum does not cover the received bytes");
        // Fixed field order: v,session,host,rank,budget,budget_origin,
        // admitted,place_count,places,capacity_source,capacity_values
        std::vector<std::pair<std::string, std::string>> fields;
        std::stringstream ss(payload);
        std::string pair;
        while (std::getline(ss, pair, ';'))
        {
            const auto eq = pair.find('=');
            if (eq == std::string::npos)
                return fail(RejectReason::MalformedInput,
                    "transport: field without '=': '" + pair + "'");
            fields.emplace_back(pair.substr(0, eq), pair.substr(eq + 1));
        }
        const char *expected_order[] = {"v", "session", "host", "rank", "budget",
            "budget_origin", "admitted", "place_count", "places",
            "capacity_source", "capacity_values"};
        if (fields.size() != sizeof(expected_order) / sizeof(expected_order[0]))
            return fail(RejectReason::MalformedInput,
                "transport: wrong field count");
        for (std::size_t i = 0; i < fields.size(); ++i)
            if (fields[i].first != expected_order[i])
                return fail(RejectReason::MalformedInput,
                    "transport: field order violation at " + std::to_string(i) +
                    " ('" + fields[i].first + "' where '" + expected_order[i] +
                    "' is required)");
        if (fields[0].second != "1")
            return fail(RejectReason::MalformedInput,
                "transport: unsupported schema version '" + fields[0].second + "'");
        for (const int index : {1, 2, 3})
            if (!transportIdentityCharsetOk(fields[index].second))
                return fail(RejectReason::MalformedInput,
                    "transport: identity field '" + fields[index].first +
                    "' empty or charset-invalid");
        // budget: positive decimal integer, full consumption
        {
            const auto &budget = fields[4].second;
            if (budget.empty() ||
                budget.find_first_not_of("0123456789") != std::string::npos)
                return fail(RejectReason::MalformedInput,
                    "transport: budget not a positive integer");
            if (std::stoll(budget) <= 0)
                return fail(RejectReason::MalformedInput,
                    "transport: budget must be positive");
        }
        if (fields[5].second != "Explicit" && fields[5].second != "AutoLocal")
            return fail(RejectReason::MalformedInput,
                "transport: budget_origin must be Explicit or AutoLocal");
        const auto admitted = [&]() -> std::optional<std::vector<int>> {
            try { return parseCpuListStrict(fields[6].second); }
            catch (const PlacementTopologyReject &r) { reject = r; return std::nullopt; }
        }();
        if (!admitted) return std::nullopt;
        std::size_t place_count = 0;
        {
            const auto &value = fields[7].second;
            if (value.empty() ||
                value.find_first_not_of("0123456789") != std::string::npos)
                return fail(RejectReason::MalformedInput,
                    "transport: place_count not a non-negative integer");
            place_count = static_cast<std::size_t>(std::stoull(value));
        }
        auto places = [&]() -> std::optional<std::vector<CorePlace>> {
            try { return parseOmpPlaces(fields[8].second); }
            catch (const PlacementTopologyReject &r) { reject = r; return std::nullopt; }
        }();
        if (!places) return std::nullopt;
        if (places->size() != place_count)
            return fail(RejectReason::MalformedInput,
                "transport: place_count does not match the parsed place list");
        // Managed invariant: the worker budget cannot exceed the place count
        // (AutoLocal derivation guarantees it; an Explicit value violating it
        // is rejected at the transport layer, not deferred to runtime).
        if (std::stoll(fields[4].second) > static_cast<long long>(place_count))
            return fail(RejectReason::MalformedInput,
                "transport: budget " + fields[4].second +
                " exceeds place_count " + fields[7].second +
                " — managed receipts cannot request more workers than places");
        // Partition proof: the places must exactly partition `admitted`.
        {
            std::set<int> union_cpus;
            for (const auto &place : *places)
            {
                if (place.sibling_cpus.front() != place.representative)
                    return fail(RejectReason::MalformedInput,
                        "transport: representative invariant violated");
                for (const int cpu : place.sibling_cpus)
                    if (!union_cpus.insert(cpu).second)
                        return fail(RejectReason::DuplicateAdmittedId,
                            "transport: cpu " + std::to_string(cpu) +
                            " appears in more than one place");
            }
            if (union_cpus !=
                std::set<int>(admitted->begin(), admitted->end()))
                return fail(RejectReason::MalformedInput,
                    "transport: places do not exactly partition the admitted set");
        }
        CapacitySource source = CapacitySource::Unknown;
        if (fields[9].second == "KernelCapacityUnits")
            source = CapacitySource::KernelCapacityUnits;
        else if (fields[9].second == "CpufreqKhz")
            source = CapacitySource::CpufreqKhz;
        else
            return fail(RejectReason::MalformedInput,
                "transport: capacity_source '" + fields[9].second + "' unknown");
        // capacity_values: exactly place_count positive integers
        {
            std::vector<std::string> values;
            std::stringstream vs(fields[10].second);
            std::string value;
            while (std::getline(vs, value, ','))
            {
                if (value.empty() ||
                    value.find_first_not_of("0123456789") != std::string::npos)
                    return fail(RejectReason::MalformedInput,
                        "transport: capacity value not a positive integer");
                values.push_back(value);
            }
            if (values.size() != place_count)
                return fail(RejectReason::MalformedInput,
                    "transport: capacity_values count does not match place_count");
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                const long parsed = std::stoll(values[i]);
                if (parsed <= 0)
                    return fail(RejectReason::MalformedInput,
                        "transport: capacity value must be positive");
                (*places)[i].capacity_value = parsed;
                (*places)[i].capacity_source = source;
            }
        }
        ReceiptSessionIdentity identity;
        identity.session = fields[1].second;
        identity.host = fields[2].second;
        identity.rank = fields[3].second;
        identity.budget = std::stoll(fields[4].second);
        identity.budget_origin_explicit = fields[5].second == "Explicit";
        PlacementTransportView view{
            PlacementReceipt(PlacementReceipt::ConstructionPasskey{},
                std::move(*places), fields[8].second, received_checksum),
            std::move(identity)};
        return view;
    }

    std::optional<PlacementReceipt> parseReceiptTransport(
        const std::string &transport, PlacementTopologyReject &reject)
    {
        auto view = parseReceiptTransportView(transport, reject);
        if (!view.has_value()) return std::nullopt;
        return std::move(view->receipt);
    }

// ---- standards-complete authored OMP_PLACES validator (libgomp grammar) ----

namespace
{
    struct AuthoredCursor
    {
        const std::string &s;
        std::size_t pos = 0;
        bool atEnd() const { return pos >= s.size(); }
        char peek() const { return s[pos]; }
        void skipWs()
        {
            while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
        }
    };

    // strtoul with ERANGE rejection; the caller checks the delimiter next.
    unsigned long authoredUlong(AuthoredCursor &c, const char *what)
    {
        errno = 0;
        char *end = nullptr;
        const unsigned long value = std::strtoul(c.s.c_str() + c.pos, &end, 10);
        if (end == c.s.c_str() + c.pos)
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                std::string("authored OMP_PLACES: expected ") + what);
        if (errno == ERANGE)
            throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                std::string("authored OMP_PLACES: ") + what + " out of range");
        c.pos = static_cast<std::size_t>(end - c.s.c_str());
        return value;
    }

    long authoredLong(AuthoredCursor &c, const char *what)
    {
        errno = 0;
        char *end = nullptr;
        const long value = std::strtol(c.s.c_str() + c.pos, &end, 10);
        if (end == c.s.c_str() + c.pos)
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                std::string("authored OMP_PLACES: expected ") + what);
        if (errno == ERANGE)
            throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                std::string("authored OMP_PLACES: ") + what + " out of range");
        c.pos = static_cast<std::size_t>(end - c.s.c_str());
        return value;
    }

    // libgomp gomp_affinity_add_cpus semantics: len entries stepping by
    // stride, with libgomp's exact wraparound checks and the static
    // [0, CPU_SETSIZE) bound (machine availability is the runtime's call).
    void authoredAddCpus(std::vector<int> &place, unsigned long num,
        unsigned long len, long stride)
    {
        for (;;)
        {
            if (num >= kCpuSetSize)
                throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                    "authored OMP_PLACES: cpu " + std::to_string(num) +
                    " outside the representable range");
            place.push_back(static_cast<int>(num));
            if (--len == 0) return;
            if ((stride < 0 && num + stride > num) ||
                (stride > 0 && num + stride < num))
                throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                    "authored OMP_PLACES: cpu " + std::to_string(num) + "+" +
                    std::to_string(stride) + " wraps around");
            num = static_cast<unsigned long>(static_cast<long>(num) + stride);
        }
    }

    // libgomp gomp_affinity_copy_place semantics: every cpu shifted by
    // stride, same wraparound checks.
    std::vector<int> authoredCopyPlace(const std::vector<int> &source, long stride)
    {
        std::vector<int> shifted;
        shifted.reserve(source.size());
        for (const int cpu : source)
        {
            const long moved = static_cast<long>(cpu) + stride;
            if ((stride < 0 && moved > cpu) || (stride > 0 && (moved < cpu || moved >= kCpuSetSize)))
                throw PlacementTopologyReject(RejectReason::IdOutOfRange,
                    "authored OMP_PLACES: replication shift " +
                    std::to_string(stride) + " moves cpu " +
                    std::to_string(cpu) + " out of range");
            shifted.push_back(static_cast<int>(moved));
        }
        std::sort(shifted.begin(), shifted.end());
        return shifted;
    }

    bool authoredSamePlace(const std::vector<int> &a, const std::vector<int> &b)
    {
        return a == b;
    }
} // namespace

AuthoredPlacesInterpretation validateAuthoredOmpPlaces(
    const std::string &authored)
{
    AuthoredCursor c{authored};
    c.skipWs();
    if (c.atEnd())
        throw PlacementTopologyReject(RejectReason::MalformedInput,
            "authored OMP_PLACES: empty");
    // Abstract names, case-insensitive (libgomp strncasecmp), optional
    // "( count )", NOTHING may follow.
    static const struct { const char *name; unsigned level; } kAbstract[] = {
        {"threads", 1}, {"cores", 2}, {"sockets", 3},
        {"ll_caches", 4}, {"numa_domains", 5}};
    for (const auto &kind : kAbstract)
    {
        const std::size_t len = std::strlen(kind.name);
        if (c.s.size() - c.pos >= len)
        {
            bool match = true;
            for (std::size_t i = 0; i < len; ++i)
                if (std::tolower(static_cast<unsigned char>(c.s[c.pos + i])) !=
                    kind.name[i])
                {
                    match = false;
                    break;
                }
            if (!match) continue;
            c.pos += len;
            c.skipWs();
            AuthoredPlacesInterpretation interpretation;
            interpretation.kind = AuthoredPlacesInterpretation::Kind::Abstract;
            interpretation.abstract_name = kind.name;
            if (c.atEnd())
            {
                interpretation.abstract_count = -1; // all
                return interpretation;
            }
            if (c.peek() != '(')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: trailing content after abstract name");
            ++c.pos; c.skipWs();
            interpretation.abstract_count = static_cast<long>(authoredUlong(c, "abstract count"));
            c.skipWs();
            if (c.atEnd() || c.peek() != ')')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: abstract count requires ')'");
            ++c.pos; c.skipWs();
            if (!c.atEnd())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: trailing content after abstract places");
            return interpretation;
        }
    }
    // Explicit place list (mirrors libgomp's two-phase parse exactly).
    AuthoredPlacesInterpretation interpretation;
    interpretation.kind = AuthoredPlacesInterpretation::Kind::Explicit;
    std::vector<std::vector<int>> places;
    unsigned long total = 0;
    bool any_negate = false;
    std::size_t guard = 0;
    while (true)
    {
        if (++guard > 65536 + 8)
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "authored OMP_PLACES: place count exceeds 65536");
        c.skipWs();
        bool place_negate = false;
        if (!c.atEnd() && c.peek() == '!')
        {
            place_negate = true;
            ++c.pos;
            c.skipWs();
        }
        std::vector<int> place;
        bool has_braces = true;
        if (!c.atEnd() && c.peek() == '{')
        {
            ++c.pos; c.skipWs();
            std::vector<int> added;
            std::vector<int> removed;
            if (!c.atEnd() && c.peek() == '}')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: empty place group");
            while (true)
            {
                c.skipWs();
                bool element_negate = false;
                if (!c.atEnd() && c.peek() == '!')
                {
                    element_negate = true;
                    ++c.pos; c.skipWs();
                }
                const unsigned long num = authoredUlong(c, "cpu id");
                c.skipWs();
                unsigned long len = 1;
                long stride = 1;
                if (!c.atEnd() && c.peek() == ':')
                {
                    if (element_negate)
                        throw PlacementTopologyReject(RejectReason::MalformedInput,
                            "authored OMP_PLACES: negated element cannot take a range");
                    ++c.pos; c.skipWs();
                    len = authoredUlong(c, "range length");
                    if (len == 0)
                        throw PlacementTopologyReject(RejectReason::MalformedInput,
                            "authored OMP_PLACES: range length must be nonzero");
                    c.skipWs();
                    if (!c.atEnd() && c.peek() == ':')
                    {
                        ++c.pos; c.skipWs();
                        stride = authoredLong(c, "range stride");
                        c.skipWs();
                    }
                }
                if (element_negate) removed.push_back(static_cast<int>(num));
                else authoredAddCpus(added, num, len, stride);
                if (c.atEnd() || c.peek() != ',')
                    break;
                ++c.pos;
            }
            c.skipWs();
            if (c.atEnd() || c.peek() != '}')
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: missing '}'");
            ++c.pos;
            // libgomp two-pass composition: adds first, then removals.
            place = added;
            for (const int cpu : removed)
            {
                auto it = std::find(place.begin(), place.end(), cpu);
                if (it == place.end())
                    throw PlacementTopologyReject(RejectReason::MalformedInput,
                        "authored OMP_PLACES: excluding cpu " + std::to_string(cpu) +
                        " which the place does not contain");
                place.erase(it);
            }
            if (place.empty())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: place group emptied by exclusion");
            std::sort(place.begin(), place.end());
        }
        else
        {
            has_braces = false;
            const unsigned long num = authoredUlong(c, "cpu id");
            authoredAddCpus(place, num, 1, 1);
        }
        (void)has_braces;
        c.skipWs();
        // Place replication: ": length [: stride]", length in [1,65535].
        unsigned long replication = 1;
        long replication_stride = 1;
        if (!c.atEnd() && c.peek() == ':')
        {
            if (place_negate)
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: negated place cannot be replicated");
            ++c.pos; c.skipWs();
            replication = authoredUlong(c, "replication length");
            if (replication == 0 || replication >= 65536)
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: replication length must be in [1,65535]");
            c.skipWs();
            if (!c.atEnd() && c.peek() == ':')
            {
                ++c.pos; c.skipWs();
                replication_stride = authoredLong(c, "replication stride");
                c.skipWs();
            }
        }
        // libgomp count accounting: non-negated adds replication length;
        // only the FIRST negated place counts 1.
        if (place_negate)
        {
            if (!any_negate) total += 1;
            any_negate = true;
        }
        else
        {
            total += replication;
        }
        if (total > 65536)
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "authored OMP_PLACES: place count exceeds 65536");
        if (place_negate)
        {
            interpretation.place_exclusion_used = true;
            std::size_t match = places.size();
            for (std::size_t i = 0; i < places.size(); ++i)
                if (authoredSamePlace(places[i], place))
                {
                    match = i;
                    break;
                }
            if (match == places.size())
                throw PlacementTopologyReject(RejectReason::MalformedInput,
                    "authored OMP_PLACES: excluding a place that does not exist earlier");
            places.erase(places.begin() + static_cast<std::ptrdiff_t>(match));
        }
        else
        {
            places.push_back(place);
            for (unsigned long copy = 1; copy < replication; ++copy)
                places.push_back(authoredCopyPlace(places.back(), replication_stride));
        }
        c.skipWs();
        if (c.atEnd()) break;
        if (c.peek() != ',')
            throw PlacementTopologyReject(RejectReason::MalformedInput,
                "authored OMP_PLACES: expected ',' between places");
        ++c.pos;
    }
    if (places.empty())
        throw PlacementTopologyReject(RejectReason::MalformedInput,
            "authored OMP_PLACES: all places have been removed");
    interpretation.deterministic_places = std::move(places);
    return interpretation;
}

} // namespace llaminar2::placement
