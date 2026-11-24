/**
 * @file CUDAMoEArrivalPoolProof.h
 * @brief Matched native-pool certificate for final-owner CUDA arrival imports.
 *
 * Compare the same physical work kernel, arguments, owner shape and upload
 * lifecycle, with and without the extra control/work child and update binding.
 * No fixture graph runs: the comparison covers setup, not payload semantics.
 */
#pragma once

#include "backends/cuda/CUDALiveLaunchMemoryContract.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoERebalanceCadenceRecipe.h"
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#include "planning/CapturedGraphMemoryEstimator.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <gtest/gtest.h>
#include <array>
#include <iostream>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    enum class CUDAMoEArrivalPoolOwner
    {
        StandaloneCopy, StandaloneCadence, OrdinaryDecode, NativeMTPFixed,
        NativeMTPDynamic, OrdinaryFollower, SpeculativeFollower, ForwardTimeline,
    };

    /** @brief Record, instantiate and upload matched retained families on the original worker. */
    inline void proveCUDAMoEArrivalPool(IWorkerGPUContext &worker, DeviceId device,
        GPUGraphMemoryObservationScope scope, CUDAMoEArrivalPoolOwner kind,
        CUDAArrivalOwnerRole role, bool mixed_families, std::size_t sites)
    {
        worker.submitAndWait([&] {
            constexpr std::size_t owner_count = 8;
            constexpr auto memory_owner = PhysicalMemoryOwner::NativeGraphExecutable;
            auto inventory = CapturedGraphExecutableInventory::monolithic(2 * owner_count + 1);
            const auto base_bytes = estimateCapturedGraphExecutableBytes(device, inventory);
            inventory.cuda_arrival_owners.add({role, owner_count, sites});
            const auto family_bytes = estimateCapturedGraphExecutableBytes(device, inventory);
            const PhysicalMemoryResource resource{.world_rank = 0, .device = device,
                .total_bytes = family_bytes, .admission_available_bytes = family_bytes};
            PhysicalMemoryPlanBuilder plan;
            plan.add(resource, memory_owner, family_bytes);
            auto authority = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
            // Both measured families and one warm owner can coexist. Hold the
            // single opaque family reservation before recording any of them.
            const auto reservation = authority->reserveNewAllocations(device, memory_owner, family_bytes);
            ASSERT_EQ(authority->plannedBytes(device, memory_owner), family_bytes);
            ASSERT_EQ(authority->reservedBytes(device, memory_owner), family_bytes);
            const auto stream = static_cast<cudaStream_t>(worker.defaultStream());
            struct State
            {
                DeviceMoERebalancePlanEntry plans[31];
                DeviceMoERebalanceCommandBufferHeader header;
                DeviceMoEExpertDirectoryEntry slots[31];
                DeviceMoERebalanceApplyStatus status;
                DeviceMoERebalanceArrivalScratch scratch[63];
                DeviceMoETransferSlotRange ranges[5];
                uint32_t due;
                int controls[4];
                uint8_t payload[31 * (sizeof(DeviceMoEExpertDirectoryEntry) + 256u)];
            };
            State *state = nullptr;
            ASSERT_EQ(cudaMalloc(&state, sizeof(State)), cudaSuccess);
            std::unique_ptr<State, decltype(&cudaFree)> storage(state, cudaFree);
            ASSERT_EQ(cudaMemsetAsync(state, 0, sizeof(State), stream), cudaSuccess);
            DeviceMoERebalanceConfig config;
            config.num_layers = 5;
            config.num_experts = config.top_k = config.window_size_tokens = 1;
            config.participant_count = 2;
            config.transfer_slot_layer_ranges = state->ranges;
            config.histogram_layout = moe_rebalance_abi::HistogramLayout::TransferFamilyOccupancy;
            config.active_transfer_slot_capacity = 15;
            config.transfer_slot_directory_capacity = mixed_families ? 31 : 19;
            // Five routed layers, three active slots per layer, and four
            // staging slots per format family. Two layers share family zero.
            const std::array<DeviceMoETransferSlotRange, 5> ranges = mixed_families
                ? std::array<DeviceMoETransferSlotRange, 5>{{{0, 10, 6}, {0, 10, 6},
                    {10, 17, 3}, {17, 24, 3}, {24, 31, 3}}}
                : std::array<DeviceMoETransferSlotRange, 5>{{{0, 19, 15}, {0, 19, 15},
                    {0, 19, 15}, {0, 19, 15}, {0, 19, 15}}};
            ASSERT_EQ(cudaMemcpyAsync(state->ranges, ranges.data(), sizeof(state->ranges),
                cudaMemcpyHostToDevice, stream), cudaSuccess);
            CUDAMoEKernel kernel(device.ordinal);
            const auto record = [&](IGPUGraphCapture &capture, const auto &submit) {
                ScopedBackendGraphCapture transaction(worker, capture, "CUDA arrival graph pool certificate");
                ASSERT_TRUE(transaction.begin());
                submit();
                transaction.finish();
            };
            const auto record_live_arrival = [&] {
                ASSERT_TRUE(kernel.unpackDeviceRebalanceCollectivePayloads({.stream = stream}, state->plans, nullptr,
                    config.transfer_slot_directory_capacity, &state->header, state->payload,
                    config.transfer_slot_directory_capacity, sizeof(DeviceMoEExpertDirectoryEntry) + 256u,
                    state->slots, config.transfer_slot_directory_capacity, config, &state->status, state->scratch));
            };
            auto live_arrival = worker.createGraphCapture(stream);
            ASSERT_NO_FATAL_FAILURE(record(*live_arrival, record_live_arrival));
            // The baseline keeps the exact kernel/argument upload footprint,
            // but records it as an ordinary kernel node with no update owner.
            std::size_t one = 1u;
            cudaGraphNode_t child{}, control{}, work{};
            cudaGraph_t pair{};
            cudaKernelNodeParams parameters{};
            ASSERT_EQ(cudaGraphGetRootNodes(dynamic_cast<CUDAGraphCapture &>(*live_arrival).graph(), &child, &one), cudaSuccess);
            ASSERT_EQ(cudaGraphChildGraphNodeGetGraph(child, &pair), cudaSuccess);
            ASSERT_EQ(cudaGraphGetRootNodes(pair, &control, &one), cudaSuccess);
            ASSERT_EQ(cudaGraphNodeGetDependentNodes(control, &work, nullptr, &one), cudaSuccess);
            ASSERT_EQ(cudaGraphKernelNodeGetParams(work, &parameters), cudaSuccess);
            const auto record_fixed_arrival = [&] {
                ASSERT_EQ(cudaLaunchKernel(parameters.func, parameters.gridDim, parameters.blockDim,
                    parameters.kernelParams, parameters.sharedMemBytes, stream), cudaSuccess);
            };
            auto fixed_arrival = worker.createGraphCapture(stream);
            ASSERT_NO_FATAL_FAILURE(record(*fixed_arrival, record_fixed_arrival));
            auto command = worker.createGraphCapture(stream);
            ASSERT_NO_FATAL_FAILURE(record(*command, [&] {
                ASSERT_EQ(cudaMemsetAsync(&state->due, 1, sizeof(unsigned), stream), cudaSuccess);
            }));
            auto forward = worker.createGraphCapture(stream);
            ASSERT_NO_FATAL_FAILURE(record(*forward, [&] {
                // Retain the existing production-checkpoint node scale outside
                // the arrival path; those nodes are identical in both owners.
                for (unsigned i = 0; i < 1525; ++i)
                {
                    ASSERT_EQ(cudaMemsetAsync(&state->payload[0], i, 128, stream), cudaSuccess);
                    ASSERT_EQ(cudaMemcpyAsync(&state->payload[128], &state->payload[0], 128,
                        cudaMemcpyDeviceToDevice, stream), cudaSuccess);
                }
            }));

            const auto incremental_quote = quoteCUDALiveLaunchReservationBytes(sites);
            const auto build_owner = [&](bool exact, CUDAGraphCapture &owner) {
                const auto *arrival = exact ? live_arrival.get() : fixed_arrival.get();
                const DeviceMoERebalanceCadenceRecipe recipe(*command, *arrival, *command, &state->due);
                const bool follower = kind == CUDAMoEArrivalPoolOwner::OrdinaryFollower ||
                    kind == CUDAMoEArrivalPoolOwner::SpeculativeFollower;
                std::vector<DeviceControlledLoopFragment> iteration{{"forward", forward.get()}};
                if (kind == CUDAMoEArrivalPoolOwner::NativeMTPFixed || kind == CUDAMoEArrivalPoolOwner::NativeMTPDynamic)
                    iteration.push_back({"first draft", command.get()});
                if (kind == CUDAMoEArrivalPoolOwner::NativeMTPDynamic)
                    iteration.push_back({.name = "second draft", .capture = command.get(),
                        .execution = DeviceControlledLoopFragmentExecution::IfDeviceSelectorAtLeast, .minimum_selector = 2});
                if (kind == CUDAMoEArrivalPoolOwner::SpeculativeFollower)
                    iteration.push_back({"verifier commit", command.get()});
                iteration.insert(iteration.end(), recipe.fragments().begin(), recipe.fragments().end());
                if (follower)
                    iteration.push_back({"next pipeline command", command.get()});
                const DeviceControlledLoopEntryFragment entry[]{{"pipeline command", command.get()}};
                const DeviceControlledLoopProgram program{
                    .entry = follower ? std::span(entry) : std::span<const DeviceControlledLoopEntryFragment>{},
                    .iteration = iteration};
                const DeviceControlledLoopPredicate predicate{state->controls, 4, 1, 0, 1};
                switch (kind)
                {
                case CUDAMoEArrivalPoolOwner::StandaloneCopy:
                    if (exact)
                        ASSERT_NO_FATAL_FAILURE(record(owner, record_live_arrival));
                    else
                        ASSERT_NO_FATAL_FAILURE(record(owner, record_fixed_arrival));
                    break;
                case CUDAMoEArrivalPoolOwner::StandaloneCadence:
                    ASSERT_TRUE(owner.buildDeviceControlledTransaction(recipe.fragments()));
                    break;
                case CUDAMoEArrivalPoolOwner::OrdinaryDecode:
                case CUDAMoEArrivalPoolOwner::NativeMTPFixed:
                case CUDAMoEArrivalPoolOwner::OrdinaryFollower:
                case CUDAMoEArrivalPoolOwner::SpeculativeFollower:
                    ASSERT_TRUE(owner.buildDeviceControlledWhileLoop(program, predicate));
                    break;
                case CUDAMoEArrivalPoolOwner::NativeMTPDynamic:
                    ASSERT_TRUE(owner.buildDeviceControlledSelectorWhileLoop(program, predicate,
                        {state->controls, 4, 1, 0, 1, 2, 3, 1, 2, 42}));
                    break;
                case CUDAMoEArrivalPoolOwner::ForwardTimeline:
                {
                    std::vector<GPUOrderedTimelineStep> steps{{.name = "forward", .capture = forward.get()}};
                    for (std::size_t site = 0; site < sites; ++site)
                        steps.push_back({.name = "compact arrival", .capture = arrival});
                    ASSERT_TRUE(owner.buildOrderedTimelineTransaction(steps));
                    break;
                }
                }
                ASSERT_EQ(authority->reservedBytes(device, memory_owner), family_bytes);
                EXPECT_EQ(owner.liveLaunchReservationBytes(), exact ? incremental_quote : 0u);
                ASSERT_TRUE(owner.instantiate());
                EXPECT_EQ(owner.loweredLiveLaunchSiteCount(), exact ? sites : 0u);
                if (!exact)
                    ASSERT_EQ(cudaGraphUpload(owner.executable(), stream), cudaSuccess);
            };

            // Warm both paths before taking a before/after free-pool sample.
            // All retained siblings stay alive until the complete sample; no
            // individual owner's delta is treated as its allocated bytes.
            for (bool exact : {false, true})
            {
                CUDAGraphCapture warm(stream, device.ordinal);
                ASSERT_NO_FATAL_FAILURE(build_owner(exact, warm));
                ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            }
            using Owners = std::vector<std::unique_ptr<CUDAGraphCapture>>;
            const auto measure = [&](bool exact, Owners &owners, std::size_t &growth) {
                std::size_t before = 0u, after = 0u, total = 0u;
                ASSERT_EQ(cudaMemGetInfo(&before, &total), cudaSuccess);
                for (std::size_t i = 0; i < owner_count; ++i)
                {
                    owners.push_back(std::make_unique<CUDAGraphCapture>(stream, device.ordinal));
                    ASSERT_NO_FATAL_FAILURE(build_owner(exact, *owners.back()));
                    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                }
                ASSERT_EQ(cudaMemGetInfo(&after, &total), cudaSuccess);
                growth = before > after ? before - after : 0u;
            };
            // Keep both families live so driver reuse of retired baseline
            // executables cannot conceal the incremental update storage.
            Owners fixed_owners, live_owners;
            std::size_t fixed = 0u, live = 0u;
            ASSERT_NO_FATAL_FAILURE(measure(false, fixed_owners, fixed));
            ASSERT_NO_FATAL_FAILURE(measure(true, live_owners, live));
            const auto growth = live > fixed ? live - fixed : 0u;
            std::size_t materialized_quote = 0;
            for (const auto &owner : live_owners)
                materialized_quote += quoteCUDALiveLaunchReservationBytes(owner->loweredLiveLaunchSiteCount());
            EXPECT_EQ(family_bytes - base_bytes, materialized_quote);
            EXPECT_EQ(authority->reservedBytes(device, memory_owner), family_bytes);
            EXPECT_TRUE(GPUGraphMemoryContract::acceptsFamilyObservation(device,
                {.bytes = growth, .scope = scope}, materialized_quote));
            GPUGraphMemoryGrowth whole_family;
            whole_family.add(fixed, scope);
            whole_family.add(live, scope);
            EXPECT_TRUE(GPUGraphMemoryContract::acceptsFamilyObservation(device, whole_family, family_bytes));
            std::cout << "[CUDA arrival graph pool certificate] owner=" << static_cast<int>(kind)
                      << " role=" << static_cast<int>(role)
                      << " scope=" << (scope == GPUGraphMemoryObservationScope::DevicePool ? "device-pool" : "system-wide")
                      << " mixed_families=" << mixed_families << " sites=" << sites
                      << " owner_count=" << owner_count << " fixed_bytes=" << fixed
                      << " live_bytes=" << live << " family_delta_bytes=" << growth
                      << " admitted_delta_bytes=" << materialized_quote
                      << " reserved_family_bytes=" << family_bytes << '\n';
        });
    }
} // namespace llaminar2::test
