/**
 * @file OriginalRankLocalNativeReplay.h
 * @brief Native replay ownership selected from the existing original MPI inventory.
 *
 * Original rank discovery owns physical device placement. This fixture scope
 * only selects an existing complete rank-local cohort and uses production
 * planning consensus to keep idle ranks in setup and retirement.
 */
#pragma once
#include "utils/MPIContext.h"
#include "planning/PlanningPublication.h"
#include "planning/RankHardwareOwnership.h"
#include "../../utils/ObservedNativeReplayOwner.h"
#include <functional>
#include <algorithm>
#include <array>
#include <span>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string_view>
namespace llaminar2::test {
    /**
     * @brief Execute only on an original rank owning every declared native family.
     * @param families Exact positive family counts in replay coordinate order.
     * @param replay Fixture body; coordinates index this actual rank-local cohort.
     * @param owner_accepts Optional observed capability predicate applied before owner selection.
     * Idle original ranks participate in declaration and retirement consensus without GPU owners.
     */
    template<class Replay>
    void originalRankLocalReplay(std::span<const NativeReplayFamily> families, Replay replay,
        const std::function<bool(const RankInventory&)> &owner_accepts = {})
    {
        const auto check = [](bool valid, const char *message) { if (!valid) throw std::runtime_error(message); };
        check(!families.empty(), "Native replay requires its original native families");
        for (size_t index = 0; index < families.size(); ++index)
        {
            const auto &entry = families[index];
            check((entry.family == DeviceType::CUDA || entry.family == DeviceType::ROCm) && entry.degree > 0,
                "Native replay requires a positive count for each GPU family");
            check(std::none_of(families.begin(), families.begin() + index,
                [&](const auto &prior) { return prior.family == entry.family; }),
                "Native replay cannot repeat a native family declaration");
        }
        const auto mpi = MPIContextFactory::global();
        check(mpi && mpi->clusterInventory(), "Native replay requires original MPI inventory");
        // Topology construction can enter original MPI collectives. Every rank
        // completes it in a separate setup phase before any owner-only body.
        exchangePlanningSamples(mpi, [](int ranks) {
            return std::vector<std::vector<uint8_t>>(ranks, std::vector<uint8_t>{1});
        }, [&](auto request) {
            check(request.size() == 1 && request[0] == 1, "Original native topology declaration disagrees");
            check(mpi->topology() != nullptr, "Original native replay topology is unavailable");
            return std::vector<uint8_t>{1};
        }, [&](auto receipts) {
            for (const auto &receipt : receipts)
                check(receipt.bytes.size() == 1 && receipt.bytes[0] == 1,
                    "Original native topology setup omitted a rank");
        });
        const auto chosen = selectObservedNativeReplayOwner(*mpi->clusterInventory(), families, owner_accepts);
        check(chosen.has_value(), "Required native family/edge cohort is absent from every original NUMA-local rank");
        const auto owner = chosen->discovery_owner;
        const auto &devices = chosen->devices;
        exchangePlanningSamples(mpi, [](int ranks) {
            return std::vector<std::vector<uint8_t>>(ranks, std::vector<uint8_t>{1});
        }, [&](auto request) {
            check(request.size() == 1 && request[0] == 1, "Original native replay declaration disagrees");
            if (mpi->rank() == owner) replay(devices);
            return std::vector<uint8_t>{1};
        }, [&](auto receipts) {
            for (const auto &receipt : receipts)
                check(receipt.bytes.size() == 1 && receipt.bytes[0] == 1,
                    "Original native replay omitted a retirement participant");
        });
    }

    /**
     * @brief Preserve homogeneous replay coordinates using the same original family selector.
     * @param family Exact native backend family.
     * @param degree Positive participant count.
     * @param replay Fixture body on the actual owning rank.
     * @param native_isa Optional exact native ISA before target feature suffixes.
     */
    template<class Replay>
    void originalRankLocalReplay(DeviceType family, size_t degree, Replay replay, std::string_view native_isa = {})
    {
        const std::array families{NativeReplayFamily{family, degree, native_isa}};
        originalRankLocalReplay(std::span<const NativeReplayFamily>(families), std::move(replay));
    }

}
