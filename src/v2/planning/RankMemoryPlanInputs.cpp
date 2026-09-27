/**
 * @file RankMemoryPlanInputs.cpp
 * @brief Canonical ordinary rank-plan to physical BOM input construction.
 *
 * Keep allocation geometry identical for plan/auto and runtime admission.
 * This module neither owns free-byte balances nor performs driver discovery:
 * the inventory is an observation and PhysicalMemoryAuthority remains the
 * sole ledger. Local TP uses the graph's exact proportional slice, not an
 * independent integer split reconstructed from the shard count.
 */
#include "planning/RankMemoryPlanInputs.h"
#include "planning/ActivationBufferSizing.h"
#include "backends/DeviceAddressAdapter.h"
#include "config/BackendSelector.h"

namespace llaminar2
{
    std::vector<DevicePlanConfig> buildRankMemoryPlanInputs(
        const RankMemoryPlanInputRequest &request)
    {
        if (request.inventory.rank != request.plan.rank || request.plan.rank < 0)
            throw std::invalid_argument("Memory BOM inventory does not name the execution rank");
        if (request.snapshot_capacity && !request.snapshot_capacity->valid())
            throw std::invalid_argument("Memory BOM snapshot policy has no capacity");
        const auto &profile = request.model;
        /** @brief Physical observation used to fill a typed DevicePlanConfig. */
        struct DevicePlanningInventory
        {
            size_t total_bytes = 0;
            size_t free_bytes = 0;
            int compute_units = 0;
        };
        const auto inventoryForDevice = [&](DeviceId device)
        {
            const auto &rank = request.inventory;
            const auto observed = [&](size_t total, size_t available, int units)
            {
                // Validate the raw observation before a smaller explicit cap
                // could conceal corrupt geometry. PMA owns resource validity.
                if (!PhysicalMemoryResource{.world_rank = rank.rank, .device = device,
                        .total_bytes = total, .admission_available_bytes = available}.valid())
                    throw std::invalid_argument("Memory BOM has invalid observed capacity for " + device.toString());
                return DevicePlanningInventory{total, available, units};
            };
            if (device.is_cpu())
                return observed(
                    rank.cpu.memory_bytes > 0 ? rank.cpu.memory_bytes : rank.cpu_memory_bytes,
                    rank.cpu.free_memory_bytes, rank.cpu_worker_threads);
            for (const auto &gpu : rank.gpus)
                if (gpu.type == device.type && gpu.local_device_id == device.ordinal)
                    return observed(gpu.memory_bytes, gpu.free_memory_bytes, gpu.compute_units);
            throw std::invalid_argument("Memory BOM device absent from canonical rank inventory: " + device.toString());
        };

        std::vector<DevicePlanConfig> device_configs;
        auto makeConfigForDevice = [&](DeviceId device, int shard_index, int total_shards)
        {
            const DevicePlanningInventory inventory =
                inventoryForDevice(device);
            const int current_rank = request.plan.rank;

            DevicePlanConfig cfg;
            cfg.world_rank = current_rank;
            cfg.device = device;
            cfg.device_total_bytes = inventory.total_bytes;
            cfg.device_free_bytes = PhysicalMemoryAuthority::admissionCapacity(inventory.free_bytes,
                device.is_cpu() ? request.max_cpu_memory_bytes : request.max_gpu_memory_bytes);
            cfg.device_compute_units = device.is_cpu()
                ? request.inventory.cpuWorkerThreads() : inventory.compute_units;
            if (device.is_cpu()) cfg.cpu_execution = request.inventory.cpu_execution;
            if (device.is_gpu() && request.snapshot_capacity.has_value())
            {
                cfg.graph_snapshot_memory =
                    *request.snapshot_capacity;
            }
            cfg.shard_index = shard_index;
            cfg.total_shards = total_shards;
            cfg.first_layer = request.plan.first_layer;
            cfg.last_layer = request.plan.last_layer;
            cfg.owns_embedding = request.plan.has_embedding;
            cfg.batch_size = request.plan.runtime.batch_size;
            cfg.generation_request_capacity =
                std::max(1, request.plan.runtime.mtp.max_request_batch);
            cfg.max_seq_len = request.plan.runtime.max_seq_len;
            cfg.activation_seq_len = resolveActivationBufferSeqLen(cfg.max_seq_len, device);
            /* Memory planning describes resident capacity, not whether this
             * request executes MTP. An MTP-off request may deliberately retain
             * the same sidecar/checkpoint envelope as a later enabled request. */
            cfg.mtp_enabled =
                retainsMTPGraphCapacity(request.plan.runtime.mtp);
            cfg.mtp_shifted_kv_head_layout =
                resolveMTPShiftedKVHeadLayout(
                    request.plan.runtime.mtp,
                    /*dense_tensor_parallel=*/total_shards > 1,
                    total_shards);
            cfg.mtp_target_query_rows =
                cfg.mtp_enabled
                    ? resolveMTPRetainedTargetQueryRows(
                          request.plan.runtime.mtp)
                    : 1;
            cfg.mtp_terminal_logits_layout =
                resolveMTPTerminalLogitsLayout(
                    total_shards > 1,
                    request.plan.runtime.mtp.terminal_head_policy);
            /*
             * Ordinary TP graphs use GraphConfig's canonical tensor-parallel
             * dense policy.  Resolve every concurrently retained auxiliary
             * weight view from that same typed policy and the request's MTP
             * policy before PhysicalMemoryAuthority publishes its immutable
             * admission. A resolved mirrored terminal head remains live for the
             * serial decode oracle even when this request disables MTP; CPU's
             * resolved vocabulary-sharded policy does not retain that mirror.
             */
            cfg.additional_weight_sets =
                resolveAdditionalPersistentWeightSets(
                    DenseParallelPolicy::TensorParallel,
                    total_shards,
                    request.plan.runtime.mtp);

            if (device.is_gpu() && request.weight_load_geometry)
            {
                cfg.weight_load_staging = {
                    .device_bytes =
                        request.weight_load_geometry->staging_bytes,
                    .host_bytes =
                        request.weight_load_geometry->host_staging_bytes,
                };
            }



            cfg.kv_precision = activationPrecisionToString(
                resolveKVCacheStoragePrecision(
                    request.plan.runtime.kv_cache_precision,
                    device.is_cpu()));
            cfg.prefix_cache = request.plan.runtime.prefix_cache;
            const bool owns_host_prefix_tier =
                cfg.prefix_cache.enabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Disabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Device;
            if (device.is_gpu() &&
                (owns_host_prefix_tier ||
                 cfg.weight_load_staging.host_bytes != 0u))
            {
                const DevicePlanningInventory host =
                    inventoryForDevice(DeviceId::cpu());
                cfg.associated_host_memory = PhysicalMemoryResource{
                    .world_rank = current_rank,
                    .device = DeviceId::cpu(),
                    .total_bytes = host.total_bytes,
                    .admission_available_bytes = PhysicalMemoryAuthority::admissionCapacity(
                        host.free_bytes, request.max_cpu_memory_bytes),
                };
            }

            if (total_shards > 1 && profile.n_kv_heads > 0)
            {
                cfg.local_kv_heads = profile.n_kv_heads / total_shards;
                if (cfg.local_kv_heads < 1)
                    cfg.local_kv_heads = 1;
            }
            return cfg;
        };

        /**
         * Bind the same exact rank-local TP assignment used by graph and
         * weight construction. Memory admission must not independently divide
         * model dimensions because GQA KV replication and remainder shards
         * make that reconstruction lossy.
         */
        const auto bindRankLocalTPAssignment = [&profile](
            DevicePlanConfig &cfg,
            const std::vector<GlobalDeviceAddress> &participants,
            const std::vector<float> &configured_weights)
        {
            if (participants.size() <= 1u)
                return;

            std::vector<DeviceId> devices;
            devices.reserve(participants.size());
            for (const auto &participant : participants)
                devices.push_back(participant.toLocalDeviceId());

            std::vector<float> weights = configured_weights;
            if (weights.empty())
                weights.assign(devices.size(), 1.0f);
            const auto assignment =
                TensorParallelConfig::proportionalSplit(
                    devices,
                    weights,
                    profile.n_heads,
                    profile.n_kv_heads,
                    profile.d_ff,
                    profile.vocab_size);
            cfg.bindTensorParallelAssignment(
                assignment.forRank(cfg.shard_index));
        };

        if (request.plan.usesLocalPP())
        {
            // LOCAL PP: each PP stage has its own layer range. Create per-device
            // configs with the correct layer boundaries for each stage.
            const auto &pp_devices = request.plan.local_pp_devices;
            const auto &boundaries = request.plan.local_pp_layer_boundaries;
            const auto &stage_tp = request.plan.local_pp_stage_tp_info;
            // Only the homogeneous, one-device-per-stage composition installs
            // the native pipeline communicator. Nested/heterogeneous transport
            // has a different owner and must not be priced as this resource.
            const bool native_pipeline = pp_devices.size() > 1 &&
                pp_devices.front().toLocalDeviceId().is_gpu() &&
                std::all_of(pp_devices.begin(), pp_devices.end(), [&](const auto &address) {
                    return address.device_type == pp_devices.front().device_type;
                }) && std::all_of(stage_tp.begin(), stage_tp.end(), [](const auto &stage) {
                    return stage.devices.size() <= 1;
                });
            if (boundaries.size() != pp_devices.size() + 1 ||
                !std::is_sorted(boundaries.begin(), boundaries.end()) ||
                std::adjacent_find(boundaries.begin(), boundaries.end()) != boundaries.end())
                throw std::invalid_argument("Pipeline memory BOM requires one nonempty interval per stage");

            // Cross-backend channels attach to domain leaders, not every TP
            // member. Source order also assigns host mapping ownership, so the
            // same physical slots cannot be charged once by each endpoint.
            const auto bindCapturedBoundaries = [&](DevicePlanConfig &cfg, size_t stage) {
                if (!cfg.device.is_gpu() || cfg.shard_index != 0) return;
                const auto cross_backend = [&](size_t peer) {
                    const auto device = pp_devices[peer].toLocalDeviceId();
                    return device.is_gpu() && device.type != cfg.device.type;
                };
                if (stage && cross_backend(stage - 1))
                    cfg.captured_pipeline_boundaries.push_back(PipelineBoundarySide::LaterDomain);
                if (stage + 1 < pp_devices.size() && cross_backend(stage + 1))
                {
                    cfg.captured_pipeline_boundaries.push_back(PipelineBoundarySide::EarlierDomain);
                    if (!cfg.associated_host_memory)
                    {
                        const auto host = inventoryForDevice(DeviceId::cpu());
                        cfg.associated_host_memory = PhysicalMemoryResource{
                            .world_rank = request.plan.rank, .device = DeviceId::cpu(),
                            .total_bytes = host.total_bytes,
                            .admission_available_bytes = PhysicalMemoryAuthority::admissionCapacity(
                                host.free_bytes, request.max_cpu_memory_bytes)};
                    }
                }
            };

            for (size_t stage = 0; stage < pp_devices.size(); ++stage)
            {
                int stage_first = boundaries[stage];
                int stage_last = boundaries[stage + 1] - 1;

                // Check if this PP stage has TP composition (multiple devices per stage)
                if (stage < stage_tp.size() && stage_tp[stage].devices.size() > 1)
                {
                    // PP+TP: each device in this stage gets the stage's layer range + TP shard
                    const auto &tp_info = stage_tp[stage];
                    int tp_degree = static_cast<int>(tp_info.devices.size());
                    std::vector<DeviceId> tp_devices;
                    tp_devices.reserve(tp_info.devices.size());
                    for (const auto &participant : tp_info.devices)
                        tp_devices.push_back(participant.toLocalDeviceId());
                    const CollectiveBackendType resolved_backend =
                        BackendSelector::resolve(
                            tp_info.tp_backend, tp_devices);
                    for (int tp_idx = 0; tp_idx < tp_degree; ++tp_idx)
                    {
                        auto cfg = makeConfigForDevice(
                            tp_info.devices[tp_idx].toLocalDeviceId(),
                            tp_idx, tp_degree);
                        bindRankLocalTPAssignment(
                            cfg,
                            tp_info.devices,
                            tp_info.tp_weights);
                        cfg.local_tp_backend = resolved_backend;
                        cfg.first_layer = stage_first;
                        cfg.last_layer = stage_last;
                        cfg.owns_embedding =
                            request.plan.has_embedding && stage_first == 0;
                        bindCapturedBoundaries(cfg, stage);
                        device_configs.push_back(cfg);
                    }
                }
                else
                {
                    // PP only: single device per stage with that stage's full layer range
                    auto cfg = makeConfigForDevice(
                        pp_devices[stage].toLocalDeviceId(), 0, 1);
                    cfg.first_layer = stage_first;
                    cfg.last_layer = stage_last;
                    cfg.owns_embedding =
                        request.plan.has_embedding && stage_first == 0;
                    if (native_pipeline)
                        cfg.local_pipeline_backend = cfg.device.is_cuda()
                            ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                    bindCapturedBoundaries(cfg, stage);
                    device_configs.push_back(cfg);
                }
            }
        }
        else if (request.plan.usesLocalTP())
        {
            const int total_shards = static_cast<int>(request.plan.local_tp_devices.size());
            std::vector<DeviceId> tp_devices;
            tp_devices.reserve(request.plan.local_tp_devices.size());
            for (const auto &participant : request.plan.local_tp_devices)
                tp_devices.push_back(participant.toLocalDeviceId());
            const CollectiveBackendType resolved_backend =
                BackendSelector::resolve(
                    request.plan.local_tp_backend, tp_devices);
            device_configs.reserve(request.plan.local_tp_devices.size());
            for (int index = 0; index < total_shards; ++index)
            {
                auto cfg = makeConfigForDevice(
                    request.plan.local_tp_devices[static_cast<size_t>(index)].toLocalDeviceId(),
                    index,
                    total_shards);
                bindRankLocalTPAssignment(
                    cfg,
                    request.plan.local_tp_devices,
                    request.plan.local_tp_weights);
                cfg.local_tp_backend = resolved_backend;
                device_configs.push_back(std::move(cfg));
            }
        }
        else
        {
            DeviceId device = DeviceAddressAdapter::toDeviceId(request.plan.primary_device);
            device_configs.push_back(makeConfigForDevice(
                device,
                request.plan.weight_shard.shard_index,
                request.plan.weight_shard.total_shards));
        }

        if (request.captured_prefill_buckets)
        {
            const std::vector<int> configured_prefill_buckets =
                normalizePrefillGraphBuckets(*request.captured_prefill_buckets);
            if (configured_prefill_buckets.empty())
            {
                throw std::invalid_argument(
                    "Native serving graph memory admission has no configured prefill bucket inventory");
            }

            /*
             * Ordinary GPU setup retains every admitted prefill bucket, one
             * history-bearing serial decode graph, and—when the model keeps
             * an MTP sidecar family—one restored-prefix decode bridge. Native
             * drivers allocate opaque storage for those executables outside
             * BufferArena, so publish the exact setup inventory into the same
             * DevicePlanConfig that prices tensors and workspaces. Overlay
             * capacity has its own cross-rank executable inventory and must
             * not be charged a second time here.
             */
            const CapturedServingGraphMemoryInventory graph_inventory =
                resolveCapturedServingGraphMemoryInventory(
                    configured_prefill_buckets,
                    request.plan.runtime.mtp);
            for (auto &cfg : device_configs)
            {
                if (!cfg.device.is_gpu())
                    continue;
                cfg.captured_serving_graphs = graph_inventory;
                if (cfg.graph_snapshot_memory.effective_kv)
                {
                    // A per-executable bound also covers dedicated decode and
                    // sidecar arenas. Shared prefill/verifier alternatives can
                    // consume less, but cannot exceed this declared inventory.
                    cfg.graph_snapshot_memory.effective_kv->retained_arena_count =
                        graph_inventory.prefill_bucket_rows.size() +
                        graph_inventory.fixed_executable_count +
                        graph_inventory.mtp_graph_owners.sidecarGraphSlots();
                }
            }
        }
        return device_configs;
    }
}
