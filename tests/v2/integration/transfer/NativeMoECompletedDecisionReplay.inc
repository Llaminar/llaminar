/**
 * @file NativeMoECompletedDecisionReplay.inc
 * @brief Original-owned capture of actual policy finalization and request reset.
 *
 * Recorded planner/apply inputs exercise production reserve/finalize/reset
 * bodies. This publication-protocol replay complements actual model/physical
 * movement tests. Original inventory, prepared workers and PMA own every bank,
 * executable and terminal copy; no alternate communicator or rank is created.
 */
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoEOverlayDeviceEpochProtocol.h"
#include "execution/moe/NativeMoEMovementArchive.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
namespace {
/** @brief Stable replay inputs and actual device-authored terminal state. */
struct alignas(64) NativeDecisionReplayBank {
    DeviceMoEOverlayEpochControl epochs;
    DeviceMoEOverlayEpochStatus publication;
    std::uint64_t candidate = 2;
    DeviceMoELayerRuntime runtime[2];
    DeviceMoERebalanceGraphControllerState controller;
    DeviceMoERebalanceStatus policy;
    DeviceMoERebalanceApplyStatus applied;
    DeviceMoERebalanceCommandBufferHeader headers[2];
    DeviceMoERebalanceWaveState wave_states[2];
    std::uint32_t plan_counts[2]{};
    DeviceMoERebalancePlanEntry plans[4];
    DeviceMoERebalanceMovementWave waves[2];
    DeviceMoERebalanceMovementEdge edges[4];
};
static_assert(sizeof(NativeDecisionReplayBank) % sizeof(std::int32_t) == 0);
/** @brief Replay only on the actual original rank owning the entire native cohort. */
void nativeCompletedDecisionReplay(DeviceType family) {
    using Graph = test::RetainedParticipantGraphExecutableFixtureAccess;
    originalRankLocalReplay(family, 2, [&](const auto &devices) {
        const int rank = MPIContextFactory::global()->rank();
        auto original = test::PreparedNativeLookupFixture::prepare(devices, rank);
        const CPUTensorStorageDeclaration declaration(1, sizeof(NativeDecisionReplayBank) / 4,
            BufferTensorType::INT32, PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        const auto snapshot_bytes = devices.size() * sizeof(NativeDecisionReplayBank);
        PhysicalMemoryPlanBuilder plan;
        plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace,
            devices.size() * declaration.physicalBytes());
        plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
        for (const auto device : devices) {
            plan.add(original->resource(device), PhysicalMemoryOwner::ExecutionWorkspace, sizeof(NativeDecisionReplayBank));
            const auto graphs = RetainedParticipantGraphExecutable::generalBOM(original->resource(DeviceId::cpu()), original->resource(device), 3);
            for (const auto &line : graphs.resources()) plan.add(line);
        }
        original->admit(plan.build()); auto memory = original->memory();
        {
            const std::size_t root = 1; // Actual non-first participant.
            std::vector<std::shared_ptr<IWorkerGPUContext>> workers;
            std::vector<void *> streams;
            std::vector<std::shared_ptr<TensorBase>> banks;
            std::vector<std::unique_ptr<IMoEKernel>> kernels;
            std::vector<std::array<std::shared_ptr<RetainedParticipantGraphExecutable>, 3>> graphs(devices.size());
            std::vector<std::array<std::uint64_t, 3>> submissions(devices.size());
            std::vector<std::array<char, 3>> stage_ids(devices.size());
            std::vector<NativeDecisionReplayBank> states(devices.size());
            NativeMoEMovementArchiveConfig geometry{.workspace_generation = 1, .layers = 2, .experts = 2, .wave_capacity = 2, .edge_capacity = 4};
            for (const auto device : devices) geometry.participants.push_back({device, rank, 0});
            NativeMoEMovementArchive archive(std::move(geometry));
            auto snapshot_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
            auto snapshot = TransferEngine::instance().allocateMappedHostRegion(snapshot_bytes, devices);
            for (std::size_t i = 0; i < devices.size(); ++i) {
                workers.push_back(original->permit(devices[i]).worker()); streams.push_back(workers.back()->defaultStream());
                banks.push_back(declaration.materialize(memory));
                kernels.push_back(llaminar::v2::kernels::KernelFactory::createMoEKernel(devices[i]));
                require(bool(kernels.back()), "Native decision backend kernel missing");
                states[i].controller.participant_count = devices.size(); states[i].controller.participant_id = i;
                MoEOverlayDeviceEpochProtocol::initialize(states[i].epochs, 1);
                workers[i]->submitAndWait([&, i] {
                    std::memcpy(banks[i]->raw_mutable_data(), &states[i], sizeof(states[i]));
                    TransferEngine::publishHostWrite(banks[i].get());
                    TransferEngine::prepareDeviceInput(banks[i].get(), devices[i], streams[i]);
                });
                for (auto &graph : graphs[i]) graph = Graph::original(original->permit(devices[i]), streams[i]);
            }
            /** Only whole-replay terminal observation; no inference scheduling/state readback. */
            const auto await = [&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                for (std::size_t i = 0; i < devices.size(); ++i) for (;;) {
                    GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
                    workers[i]->submitAndWait([&] { state = workers[i]->queryStreamExecutionState(streams[i], "native decision terminal"); });
                    if (state == GPUStreamExecutionState::Complete) break;
                    require(std::chrono::steady_clock::now() < deadline, "Native decision terminal timeout");
                    std::this_thread::yield();
                }
            };
            NativeReplayFailureDrain drain;
            drain.abort_and_join = [&] {
                await();
                for (std::size_t i = 0; i < graphs.size(); ++i) for (std::size_t kind = 0; kind < 3; ++kind)
                    if (submissions[i][kind]) require(Graph::observeTerminal(graphs[i][kind]), "Native decision failure lost terminal proof");
                graphs.clear();
            };
            for (std::size_t i = 0; i < devices.size(); ++i) for (std::size_t kind = 0; kind < 3; ++kind) {
                GraphCaptureDependencyLedger ledger(devices[i], streams[i], {{.stage_identity = &stage_ids[i][kind],
                    .stage_name = "native completed Dynamic decision", .external_inputs = {banks[i].get()}, .outputs = {banks[i].get()}}}, "native decision protocol");
                workers[i]->submitAndWait([&, i, kind] {
                    auto &definition = Graph::capture(graphs[i][kind]);
                    auto *live = static_cast<NativeDecisionReplayBank *>(banks[i]->gpu_data_ptr());
                    const MoEKernelLaunchContext launch{.stream = streams[i]};
                    // Each retained definition starts from an external bank.
                    // Join its exact publication before capture; capture cannot
                    // import an upload or another definition's producer event.
                    TransferEngine::requireDeviceInput(banks[i].get(), devices[i], streams[i]);
                    ScopedBackendGraphCapture recording(*workers[i], definition, "native completed decision capture", &ledger);
                    require(recording.begin(), "Native decision capture begin failed");
                    ScopedGraphCaptureStage stage(&stage_ids[i][kind], streams[i]);
                    TransferEngine::requireDeviceInput(banks[i].get(), devices[i], streams[i]);
                    if (kind == 2) {
                        DeviceMoERebalanceConfig config;
                        config.num_layers = 2; config.num_experts = 2; config.top_k = 1; config.window_size_tokens = 4;
                        config.participant_id = i; config.participant_count = devices.size(); config.root_participant = root;
                        require(kernels[i]->resetDeviceRebalanceGraphTransactionForRequest(launch, &live->controller,
                            live->headers, live->wave_states, live->plan_counts, 2, config), "Native decision request reset failed");
                    } else {
                        if (!kind) require(kernels[i]->reserveMoEOverlayEpochCandidate(launch,
                            &live->epochs, &live->candidate, &live->publication), "Native decision reserve failed");
                        const DeviceMoERebalanceMovementJournalView journal{&live->controller.movement_journal,
                            live->waves, live->edges, 2, 4, i == root ? &live->policy : nullptr};
                        require(kernels[i]->finalizeMoEOverlayRebalancePublication(launch, live->runtime, 2, 2,
                            &live->epochs, &live->candidate, &live->publication, &live->applied, &live->controller,
                            live->headers, 2, live->plans, 2, 4096, journal), "Native decision finalize failed");
                    }
                    TransferEngine::publishDeviceWrite(banks[i].get(), devices[i], streams[i]); stage.complete(); recording.finish();
                    require(definition.prepareRuntimeContextStorage(memory) && definition.instantiate(), "Native decision instantiate failed");
                });
            }
            /** Replay inputs change only outside capture; addresses and executable identity stay fixed. */
            const auto upload = [&] {
                for (std::size_t i = 0; i < devices.size(); ++i) workers[i]->submitAndWait([&, i] {
                    std::memcpy(banks[i]->raw_mutable_data(), &states[i], sizeof(states[i]));
                    TransferEngine::publishHostWrite(banks[i].get()); TransferEngine::prepareDeviceInput(banks[i].get(), devices[i], streams[i]);
                });
            };
            /** Original terminal event protects every readback; retirement remains off-worker. */
            const auto run = [&](std::size_t kind) {
                for (std::size_t i = 0; i < devices.size(); ++i) workers[i]->submitAndWait([&, i] {
                    require(Graph::submit(graphs[i][kind], ++submissions[i][kind]), "Native decision graph submit failed");
                    TransferEngine::publishDeviceWrite(banks[i].get(), devices[i], streams[i]);
                });
                await();
                for (std::size_t i = 0; i < devices.size(); ++i) require(Graph::observeTerminal(graphs[i][kind]), "Native decision graph not terminal");
                for (std::size_t i = 0; i < devices.size(); ++i) workers[i]->submitAndWait([&, i] {
                    TransferEngine::requireDeviceInput(banks[i].get(), devices[i], streams[i]);
                    auto *destination = static_cast<std::byte *>(snapshot->mutableHostData()) + i * sizeof(NativeDecisionReplayBank);
                    require(getBackendFor(devices[i])->deviceToHostOnStream(destination, banks[i]->gpu_data_ptr(),
                        sizeof(NativeDecisionReplayBank), devices[i].ordinal, streams[i]), "Native decision terminal copy failed");
                });
                await();
                for (std::size_t i = 0; i < devices.size(); ++i)
                    std::memcpy(&states[i], static_cast<const std::byte *>(snapshot->mutableHostData()) + i * sizeof(NativeDecisionReplayBank), sizeof(states[i]));
            };
            /** Repeated passive observations cannot recount the same actual device completion. */
            const auto observe = [&](std::uint64_t request) {
                const auto &live = states[root]; const auto &journal = live.controller.movement_journal;
                for (int repeat = 0; repeat < 2; ++repeat) archive.observe(request, 1, journal,
                    std::span(live.waves).first(journal.committed_waves), std::span(live.edges).first(journal.committed_edges), live.controller.completed_decision);
                for (std::size_t i = 0; i < devices.size(); ++i) if (i != root) EXPECT_EQ(states[i].controller.completed_decision.completed_windows, 0);
            };
            std::uint64_t expected = 0;
            for (std::uint64_t request = 1; request <= 2; ++request) {
                std::vector<std::optional<DeviceMoEOverlayEpochTicket>> held_readers(devices.size());
                if (request == 2) {
                    run(2);
                    for (const auto &live : states) {
                        EXPECT_EQ(live.controller.completed_decision, DeviceMoERebalanceCompletedDecision{});
                        EXPECT_EQ(live.controller.last_error_code, 0);
                    }
                    observe(request); EXPECT_EQ(archive.completedDecisionWindows(), expected);
                }
                for (std::uint32_t replay = 0; replay < 20; ++replay) {
                    const bool not_ready = replay % 5 == 0, invalid = replay == 19, moving = replay == 17 || replay == 18;
                    for (std::size_t participant = 0; participant < states.size(); ++participant) {
                        auto &live = states[participant];
                        live.policy = {}; live.policy.status_code = not_ready ? 1 : 0;
                        live.policy.windows_observed = not_ready ? 0 : 1; live.policy.skipped_not_ready = not_ready;
                        live.policy.window_required_slots = 4; live.policy.window_ready_slots = not_ready ? 0 : 4;
                        live.policy.pre_wave_load_total = not_ready ? 0 : 37 + replay;
                        live.controller.decode_rounds_committed += 3;
                        live.applied = {};
                        if (invalid) live.applied.status_code = static_cast<std::uint32_t>(DeviceMoERebalanceStatusCode::InvalidRuntime);
                        if (moving) {
                            // Recorded successful apply/arrival evidence from one paired ownership wave.
                            // Copy kernels are covered by native movement tests; this replay proves its
                            // receipt cannot complete before the real publication finalizer commits.
                            const auto epoch = static_cast<std::uint32_t>(live.candidate);
                            const auto bank = static_cast<std::uint32_t>(live.epochs.published_selector & 1u);
                            // A retained ready wave can complete after a later planner finds it busy.
                            // Deliberately conflicting scratch must not become its provenance.
                            live.policy.status_code = 1; live.policy.skipped_not_ready = live.policy.skipped_busy_wave = 1;
                            live.policy.windows_observed = 0; live.policy.window_ready_slots = 0;
                            live.policy.pre_wave_load_total = 999; live.policy.dynamic_ownership_swap_accepts = 0;
                            const auto wave_index = live.controller.active_wave;
                            auto &wave = live.controller.waves[wave_index]; wave = {};
                            wave.epoch = epoch; wave.state = static_cast<std::uint32_t>(DeviceMoERebalanceWaveLifecycle::PreparedForPublication);
                            wave.planned_layer_count = wave.applied_layer_count = 2;
                            wave.command_count = wave.copied_arrivals = 2; wave.copied_payload_bytes = 6144;
                            auto &header = live.headers[wave_index]; header = {};
                            header.participant_id = participant; header.participant_count = devices.size();
                            header.epoch = epoch; header.command_count = header.command_capacity = 2;
                            header.movement_decision.kind = DeviceMoERebalanceMovementDecisionKind::NativeOwnershipSpread;
                            header.movement_decision.load = {40, 100, 60, 200, 200, 0, 20, 200, 200, 1, 40, 300, 1};
                            live.applied.changed_layers = 1; live.applied.transaction_wave_index = wave_index;
                            live.applied.transaction_epoch = epoch; live.applied.transaction_command_count = 2;
                            for (auto &runtime : live.runtime) {
                                runtime = DeviceMoELayerRuntime{}; runtime.active_bank = bank; runtime.active_epoch = epoch - 1;
                                runtime.expert_count = runtime.participant_count = 2;
                                auto &source = runtime.banks[bank]; source.epoch = epoch - 1; source.expert_count = 2;
                                source.experts[0].owner_participant = 0; source.experts[1].owner_participant = 1;
                            }
                            auto &candidate = live.runtime[0].banks[1u - bank];
                            candidate = live.runtime[0].banks[bank]; candidate.epoch = epoch;
                            candidate.experts[0].owner_participant = 1; candidate.experts[1].owner_participant = 0;
                            for (std::uint32_t expert = 0; expert < 2; ++expert) {
                                auto &plan = live.plans[2 * wave_index + expert]; plan = {};
                                plan.op = static_cast<std::uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer);
                                plan.expert = expert; plan.source_participant = expert; plan.destination_participant = 1 - expert;
                                plan.activation_count = expert ? 20 : 100;
                            }
                        }
                    }
                    if (replay == 17) {
                        // Record a real acquisition on the old published bank
                        // before native publication makes that bank Retiring.
                        // Consecutive waves keep intervening no-op reservations
                        // from legitimately reclaiming the unpinned bank.
                        for (std::size_t participant = 0; participant < states.size(); ++participant) {
                            MoEOverlayDeviceEpochProtocol epochs(states[participant].epochs);
                            held_readers[participant] = epochs.tryAcquirePublished();
                            ASSERT_TRUE(held_readers[participant]);
                        }
                    }
                    if (replay == 18) {
                        // The first successful moving publication left a genuine Retiring bank.
                        // Replay its delayed original reader, then the later skipped planner output.
                        // No new executable, transport or command identity is manufactured.
                        const auto prepared = states;
                        for (auto &live : states) {
                            const auto peer = 1u - static_cast<std::uint32_t>(live.epochs.published_selector & 1u);
                            EXPECT_EQ(live.epochs.bank_states[peer], static_cast<std::uint32_t>(DeviceMoEOverlayEpochBankState::Retiring));
                            EXPECT_EQ(live.epochs.bank_readers[peer], 1u);
                            live.applied = {};
                            live.controller.waves[live.controller.active_wave].state = static_cast<std::uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply);
                        }
                        const auto previous = states[root].controller.completed_decision;
                        upload(); run(0);
                        for (const auto &live : states) {
                            EXPECT_EQ(live.publication.code, static_cast<std::uint32_t>(DeviceMoEOverlayEpochStatusCode::Busy));
                            EXPECT_EQ(live.headers[live.controller.active_wave].command_count, 2);
                            EXPECT_EQ(live.controller.waves[live.controller.active_wave].state, static_cast<std::uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply));
                            EXPECT_EQ(live.controller.last_error_code, 0);
                        }
                        EXPECT_EQ(states[root].controller.completed_decision, previous);
                        observe(request); EXPECT_EQ(archive.completedDecisionWindows(), expected);
                        states = prepared; // Recorded reader retirement/apply inputs, same original buffers.
                        for (std::size_t participant = 0; participant < states.size(); ++participant) {
                            MoEOverlayDeviceEpochProtocol epochs(states[participant].epochs);
                            ASSERT_TRUE(held_readers[participant]);
                            ASSERT_TRUE(epochs.release(*held_readers[participant]));
                            held_readers[participant].reset();
                        }
                    }
                    upload(); run(0); if (!not_ready && !invalid) ++expected; observe(request);
                    EXPECT_EQ(archive.completedDecisionWindows(), expected);
                    if (invalid) for (const auto &live : states) EXPECT_NE(live.controller.last_error_code, 0);
                    if (!not_ready && !invalid) {
                        ASSERT_TRUE(archive.lastDecision()); EXPECT_EQ(archive.lastDecision()->noMovement(), !moving);
                        EXPECT_EQ(archive.lastDecision()->command_count, moving ? 2u : 0u);
                        EXPECT_EQ(archive.lastDecision()->snapshot_observations, moving ? 200u : 37u + replay);
                        EXPECT_EQ(archive.lastDecision()->rejected_cycles_observed, !moving);
                        EXPECT_EQ(archive.lastDecision()->source_request_epoch, request);
                    }
                    const auto before = states[root].controller.completed_decision;
                    run(1); EXPECT_EQ(states[root].controller.completed_decision, before);
                    observe(request); EXPECT_EQ(archive.completedDecisionWindows(), expected);
                }
            }
            EXPECT_EQ(expected, 30);
            EXPECT_EQ(archive.totals().transactions, 4);
            EXPECT_EQ(archive.totals().physical_bytes, 4 * 6144u);
            graphs.clear(); drain.completed = true;
        }
        original->requirePayloadRetired(); original->retire(); original->requireExactRetired();
    });
}
}
#ifdef HAVE_CUDA
/** @test Native CUDA no-move completion, empty work, failure, repeat and reset. */
TEST(NativeMoECompletedDecision, CUDA2) { nativeCompletedDecisionReplay(DeviceType::CUDA); }
#endif
#ifdef HAVE_ROCM
/** @test Symmetric original-NUMA HIP completion protocol and terminal projection. */
TEST(NativeMoECompletedDecision, ROCm2) { nativeCompletedDecisionReplay(DeviceType::ROCm); }
#endif
