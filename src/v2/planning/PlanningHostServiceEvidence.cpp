/**
 * @file PlanningHostServiceEvidence.cpp
 * @brief Exchange only bounded original CPU service identities before candidate search.
 *
 * The existing planning transport owns collective failure handling. Each rank
 * observes its own loaded providers once, after cost preparation has initialized
 * the selected native providers. Receipt parsing cannot create threads or change
 * capacity, and candidate rank compaction never substitutes root's runtime facts.
 */
#include "planning/PlanningHostServiceEvidence.h"
#include "planning/AutomaticPlanningStartup.h"
#include "planning/PlanningPublication.h"
#include "planning/ExecutionRankMembership.h"
#include "planning/MemoryPlanner.h"
#include "interfaces/IMPIContext.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>
#include <limits>

namespace llaminar2 {
namespace {
constexpr std::array<std::uint8_t, 4> kHeader{'H', 'S', 'P', 2};
constexpr std::size_t kIdentityBytes = 64;
constexpr std::size_t kRoleBytes = 1 + 2 * kIdentityBytes + 7 * sizeof(std::uint64_t);
constexpr std::uint8_t kLookup = 1, kExecutor = 2;

/** @return Stable protocol role, independent of the C++ enum representation. */
std::uint8_t wireRole(RetainedHostServiceRole role) {
    switch (role) {
    case RetainedHostServiceRole::ImmutableLookupIO: return kLookup;
    case RetainedHostServiceRole::ExecutorParentTicket: return kExecutor;
    }
    throw std::invalid_argument("Unknown planning CPU service role");
}

/** @return Closed authored role; unknown values cannot become a native program. */
RetainedHostServiceRole nativeRole(std::uint8_t role) {
    switch (role) {
    case kLookup: return RetainedHostServiceRole::ImmutableLookupIO;
    case kExecutor: return RetainedHostServiceRole::ExecutorParentTicket;
    }
    throw std::invalid_argument("Unknown planning CPU service wire role");
}

/** @brief Bound receipt strings before allocation or installed-class lookup. */
void requireIdentity(std::string_view value) {
    if (value.size() != kIdentityBytes || !std::all_of(value.begin(), value.end(),
            [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
        throw std::invalid_argument("Planning CPU service identity is not a complete declaration digest");
}

/**
 * @return Potential roles for this physical rank, before candidate selection.
 * CPU-only ranks never create either GPU continuation service. The shared
 * topology declaration later selects actual roles; these are only evidence
 * collection requirements, not allocation counts or capacity charges.
 */
std::uint8_t potentialRoles(const AutomaticPlanningPreparation& context, int rank) {
    const auto& inventory = context.inventory();
    if (rank < 0 || rank >= inventory.world_size ||
        inventory.ranks.at(rank).rank != rank)
        throw std::invalid_argument("Planning CPU service evidence has a foreign discovery rank");
    const auto& local = inventory.ranks.at(rank);
    const bool gpu = std::any_of(local.gpus.begin(), local.gpus.end(), [](const auto& device) {
        return device.type == DeviceType::CUDA || device.type == DeviceType::ROCm;
    });
    if (!gpu) return 0;
    const auto& model = context.metadata().memoryProfile();
    return (model.flash_next ? kLookup : 0) | (model.expert_count > 0 ? kExecutor : 0);
}
}

std::vector<std::uint8_t> encodePlanningHostServiceClasses(
    std::span<const RetainedHostServiceClassProjection> classes) {
    if (classes.size() > 2)
        throw std::invalid_argument("Planning CPU service receipt exceeds its authored role set");
    std::vector<std::uint8_t> bytes(kHeader.begin(), kHeader.end());
    bytes.reserve(kHeader.size() + 1 + classes.size() * kRoleBytes);
    bytes.push_back(static_cast<std::uint8_t>(classes.size()));
    std::uint8_t seen = 0;
    for (const auto& value : classes) {
        const auto role = wireRole(value.role);
        if (seen & role) throw std::invalid_argument("Planning CPU service role is duplicated");
        seen |= role;
        requireIdentity(value.resourceIdentity);
        requireIdentity(value.liveIdentity);
        value.declaration.requireValid();
        bytes.push_back(role);
        bytes.insert(bytes.end(), value.resourceIdentity.begin(), value.resourceIdentity.end());
        bytes.insert(bytes.end(), value.liveIdentity.begin(), value.liveIdentity.end());
        const auto& d=value.declaration;
        for(const auto field:{d.stackBytes,d.guardBytes,d.pageBytes,d.objectBytes,d.objectAlignment,d.tlsBytes,d.opaqueBytes})
            for(unsigned shift=0;shift<64;shift+=8)bytes.push_back(static_cast<std::uint8_t>(std::uint64_t(field)>>shift));
    }
    return bytes;
}

std::vector<RetainedHostServiceClassProjection> decodePlanningHostServiceClasses(
    std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kHeader.size() + 1 ||
        !std::equal(kHeader.begin(), kHeader.end(), bytes.begin()) || bytes[kHeader.size()] > 2 ||
        bytes.size() != kHeader.size() + 1 + bytes[kHeader.size()] * kRoleBytes)
        throw std::invalid_argument("Planning CPU service receipt has an incompatible or incomplete frame");
    const std::size_t count = bytes[kHeader.size()];
    std::vector<RetainedHostServiceClassProjection> classes;
    classes.reserve(count);
    std::size_t offset = kHeader.size() + 1;
    std::uint8_t seen = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto role = bytes[offset++];
        const auto native = nativeRole(role);
        if (seen & role) throw std::invalid_argument("Planning CPU service receipt duplicates a role");
        seen |= role;
        std::string resource(bytes.begin() + offset, bytes.begin() + offset + kIdentityBytes);
        offset += kIdentityBytes;
        std::string live(bytes.begin() + offset, bytes.begin() + offset + kIdentityBytes);
        offset += kIdentityBytes;
        requireIdentity(resource);
        requireIdentity(live);
        RetainedHostServiceResourceDeclaration declaration;
        for(auto* field:{&declaration.stackBytes,&declaration.guardBytes,&declaration.pageBytes,&declaration.objectBytes,&declaration.objectAlignment,&declaration.tlsBytes,&declaration.opaqueBytes}){
            std::uint64_t value=0;for(unsigned shift=0;shift<64;shift+=8)value|=std::uint64_t(bytes[offset++])<<shift;
            if(value>std::numeric_limits<std::size_t>::max())throw std::invalid_argument("CPU rank declaration does not fit local size type");*field=static_cast<std::size_t>(value);
        }
        declaration.requireValid();
        classes.push_back({native, std::move(resource), std::move(live),declaration});
    }
    return classes;
}

PlanningHostServiceBudgets collectPlanningHostServiceBudgets(
    const AutomaticPlanningPreparation& context) {
    PlanningHostServiceBudgets budgets;
    exchangePlanningSamples(context.mpi(), [&](int count) {
        if (count != context.inventory().world_size)
            throw std::invalid_argument("CPU service discovery changed its original communicator");
        std::vector<std::vector<std::uint8_t>> requests;
        requests.reserve(count);
        for (int rank = 0; rank < count; ++rank)
            requests.push_back({'H', 'S', 'R', 1, potentialRoles(context, rank)});
        return requests;
    }, [&](std::span<const std::uint8_t> request) {
        const int rank = context.mpi() ? context.mpi()->rank() : 0;
        const std::array<std::uint8_t, 5> expected{'H', 'S', 'R', 1, potentialRoles(context, rank)};
        if (request.size() != expected.size() || !std::equal(expected.begin(), expected.end(), request.begin()))
            throw std::invalid_argument("CPU service discovery request differs from its original rank/source");
        std::vector<RetainedHostServiceClassProjection> classes;
        if (expected.back()) {
            // One original evidence session prices both roles. No native creator
            // runs here, and CPU-only ranks never hash GPU provider closures.
            auto original = RetainedHostServiceBudgetProjection::original();
            for (auto& value : original.classes())
                if (wireRole(value.role) & expected.back()) classes.push_back(std::move(value));
        }
        return encodePlanningHostServiceClasses(classes);
    }, [&](std::span<const RankPlanningSample> samples) {
        if (samples.size() != static_cast<std::size_t>(context.inventory().world_size))
            throw std::invalid_argument("CPU service discovery omitted an original rank");
        for (std::size_t index = 0; index < samples.size(); ++index) {
            const auto& sample = samples[index];
            if (sample.discovery_rank != static_cast<int>(index))
                throw std::invalid_argument("CPU service discovery reordered or duplicated original ranks");
            const auto classes = decodePlanningHostServiceClasses(sample.bytes);
            std::uint8_t roles = 0;
            for (const auto& value : classes) roles |= wireRole(value.role);
            if (roles != potentialRoles(context, sample.discovery_rank))
                throw std::invalid_argument("CPU service discovery returned an incomplete role set");
            budgets.emplace(sample.discovery_rank, RetainedHostServiceBudgetProjection::installed(classes));
        }
    });
    return budgets;
}

RetainedHostServiceBudgetProjection projectPlanningHostServiceBudget(
    const PlanningHostServiceBudgets& budgets,
    const ExecutionRankMembership& membership, int rank) {
    const int discovery = membership.discoveryRanks().at(static_cast<std::size_t>(rank));
    const auto found = budgets.find(discovery);
    return found == budgets.end() ? RetainedHostServiceBudgetProjection::installed({}) : found->second;
}

std::shared_ptr<const RetainedHostServiceDeclarations> retainedHostServicesFromInputs(
    std::span<const DevicePlanConfig> inputs, int rank) {
    std::shared_ptr<const RetainedHostServiceDeclarations> result;
    for (const auto& input : inputs) {
        if (input.world_rank != rank || !input.retained_host_services ||
            input.retained_host_services->worldRank() != rank)
            throw std::logic_error("Admitted CPU service declarations lost their original rank/input session");
        if (result && result != input.retained_host_services)
            throw std::logic_error("Admitted CPU service inputs contain multiple declaration sessions");
        result = input.retained_host_services;
    }
    return result;
}
}
