/**
 * @file WeightManager.cpp
 * @brief WeightManager cache, decode shards, proportional slicing, device-aware access, and preparation.
 */
#include "WeightManagerInternal.h"

namespace llaminar2
{

    std::shared_ptr<TensorBase> WeightManager::createModelPreparedFloatingOverride(
        const std::string &name,
        WeightRole role,
        const TensorBase *source,
        DeviceId target_device,
        const WeightSliceSpec &logical_slice,
        WeightDerivationKind logical_derivation)
    {
        if (!source)
            return nullptr;

        const auto representation =
            PreparedWeightRepresentationContract::resolve(
                role, source->native_type(), name, loader_.architecture(),
                debugEnv().gemm.gdn_projection_arithmetic, debugEnv().gemm.moe_router_arithmetic);
        if (representation == ModelPreparedWeightRepresentation::SourceNative) return nullptr;
        const auto runtime_type = representation == ModelPreparedWeightRepresentation::FP16 ?
            TensorType::FP16 : TensorType::FP32;

        const ModelPreparedFloatingOverrideKey key{
            .canonical_name = name,
            .role = role,
            .target_device = target_device,
            .slice = logical_slice,
            .representation = representation,
        };

        /*
         * The same mutex serializes raw-source release.  Holding it across this
         * setup-only source conversion makes the cache-miss transition atomic: a
         * release sweep cannot reclaim source bytes between validation and
         * conversion, and every concurrent runner observes one authority.
         */
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (const auto cached = model_prepared_floating_overrides_.find(key);
            cached != model_prepared_floating_overrides_.end())
        {
            if (!cached->second ||
                cached->second->native_type() != runtime_type ||
                cached->second->shape() != source->shape())
            {
                throw std::runtime_error(
                    "[WeightManager] Invalid cached floating override for " + name +
                    " role=" + toString(role) +
                    " device=" + target_device.to_string());
            }
            if (cached->second->is_raw_data_released() ||
                cached->second->raw_data() == nullptr)
            {
                throw std::runtime_error(
                    "[WeightManager] Model-owned floating override lost its host "
                    "source lifetime for " +
                    name + " role=" + toString(role) +
                    " device=" + target_device.to_string());
            }
            if (PerfStatsCollector::isDomainEnabled("weight_loading"))
            {
                PerfStatsCollector::addCounter(
                    "weight_loading",
                    "model_prepared_floating_override_cache_hits",
                    1.0,
                    "materialize",
                    target_device.to_string(),
                    {{"weight_role", toString(role)}});
            }
            return cached->second;
        }

        if (source->is_raw_data_released() || source->raw_data() == nullptr)
        {
            throw std::runtime_error(
                "[WeightManager] Cannot create model-owned floating override for " +
                name + " role=" + toString(role) +
                " device=" + target_device.to_string() +
                ": raw source bytes were released before the override was materialized");
        }

        std::shared_ptr<TensorBase> prepared;
        if (representation == ModelPreparedWeightRepresentation::FP16)
            prepared = materializePreparedHalfWeights(*source);
        else
        {
            auto fp32 = std::make_shared<FP32Tensor>(source->shape(), DeviceId::cpu());
            source->to_fp32(fp32->mutable_data());
            prepared = std::move(fp32);
        }
        const char *purpose = role == WeightRole::SharedExpertInputGate
                                  ? "shared-expert-input-gate"
                                  : role == WeightRole::MoERouter ? "moe-router" : "ssm-projection";
        prepared->setDebugName(name + (runtime_type == TensorType::FP16 ? "@fp16-" : "@fp32-") + purpose);

        /*
         * PreparedWeightStore adoption compares logical slices as well as
         * names and shapes. Register the override before publishing it in the
         * cache so every observer sees the complete identity atomically.
         */
        registerDerivedMetadata(
            name,
            prepared,
            logical_derivation,
            logical_slice,
            target_device);
        model_prepared_floating_overrides_.emplace(key, prepared);

        if (PerfStatsCollector::isDomainEnabled("weight_loading"))
        {
            PerfStatsCollector::addCounter(
                "weight_loading",
                "model_prepared_floating_override_creations",
                1.0,
                "materialize",
                target_device.to_string(),
                {{"weight_role", toString(role)}});
        }
        LOG_DEBUG("[WeightManager] Dequantized " << name
                                                  << " from " << source->dtype_name()
                                                  << " to " << prepared->dtype_name() << " for "
                                                  << target_device.to_string()
                                                  << " " << purpose << " path");
        return prepared;
    }

    const char *WeightManager::weightPrepStateName(WeightPrepState state)
    {
        switch (state)
        {
        case WeightPrepState::UNKNOWN:
            return "UNKNOWN";
        case WeightPrepState::LOADED_HOST:
            return "LOADED_HOST";
        case WeightPrepState::PACKED_HOST:
            return "PACKED_HOST";
        case WeightPrepState::UPLOADED_DEVICE:
            return "UPLOADED_DEVICE";
        case WeightPrepState::READY:
            return "READY";
        case WeightPrepState::FAILED:
            return "FAILED";
        default:
            return "INVALID";
        }
    }

    void WeightManager::registerExpectedDeviceForWeight(const std::string &name, DeviceId device)
    {
        std::lock_guard<std::mutex> lock(prep_ticket_mutex_);
        expected_devices_by_weight_[name].insert(device.to_string());
    }

    void WeightManager::markPrepState(const std::string &name,
                                      DeviceId device,
                                      WeightPrepState state,
                                      bool is_gemm,
                                      const std::string &detail)
    {
        std::lock_guard<std::mutex> lock(prep_ticket_mutex_);

        const std::string device_key = device.to_string();
        expected_devices_by_weight_[name].insert(device_key);

        auto &ticket = prep_tickets_[name][device_key];
        ticket.state = state;
        ticket.is_gemm = is_gemm;
        ticket.detail = detail;
    }

    void WeightManager::evaluateReclaimEligibility(const std::string &name, bool is_gemm)
    {
        if (!is_gemm)
        {
            return;
        }

        bool should_attempt_reclaim = false;

        {
            std::lock_guard<std::mutex> lock(prep_ticket_mutex_);

            auto exp_it = expected_devices_by_weight_.find(name);
            auto ticket_it = prep_tickets_.find(name);
            if (exp_it == expected_devices_by_weight_.end() || ticket_it == prep_tickets_.end())
            {
                return;
            }

            const auto &expected = exp_it->second;
            const auto &tickets = ticket_it->second;

            if (expected.empty())
            {
                return;
            }

            const bool all_ready = std::all_of(expected.begin(), expected.end(), [&](const std::string &device_key)
                                               {
                auto it = tickets.find(device_key);
                return it != tickets.end() && it->second.state == WeightPrepState::READY; });

            if (!all_ready)
            {
                return;
            }

            reclaim_ready_weights_.insert(name);
            if (reclaim_applied_weights_.find(name) == reclaim_applied_weights_.end())
            {
                should_attempt_reclaim = true;
            }
        }

        if (!should_attempt_reclaim)
        {
            return;
        }

        if (tryReleaseReclaimHostRawData(name))
        {
            std::lock_guard<std::mutex> lock(prep_ticket_mutex_);
            reclaim_applied_weights_.insert(name);
            LOG_DEBUG("[WeightManager][Phase2] Reclaimed GEMM host raw data: " << name);
        }
    }

    bool WeightManager::tryReleaseReclaimHostRawData(const std::string &name)
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        std::unordered_set<TensorBase *> released;
        bool found_any = false;

        // Helper: safely release a tensor's host data only if it has a device
        // copy (GPU upload or kernel-managed packed data). Floating-point CPU
        // GEMM kernels (oneDNN) read weight_tensor_->data() live at every
        // inference call, so host data must be retained for CPU-only weights.
        auto safe_release = [&](TensorBase *ptr, const std::string &key)
        {
            if (!ptr || !released.insert(ptr).second)
                return;

            if (ptr->is_raw_data_released())
                return; // Already released

            // Same safety check as releaseAllHostWeightData: only release if
            // the tensor has a valid GPU copy OR kernel-managed device data.
            if (!ptr->deviceValid())
            {
                bool has_kernel_device_data = ptr->hasPreparedDeviceState();

                if (!has_kernel_device_data)
                {
                    LOG_DEBUG("[WeightManager][Phase2] Skipped reclaim for host-only weight: " << key
                                                                                               << " (no device copy)");
                    return;
                }
            }

            ptr->release_host_weight_data();
            found_any = true;
        };

        auto base_it = cache_.find(name);
        if (base_it != cache_.end() && base_it->second)
        {
            safe_release(base_it->second.get(), name);
        }

        for (const auto &[key, tensor] : per_device_cache_)
        {
            if (!tensor)
            {
                continue;
            }
            if (key.name == name)
                safe_release(tensor.get(), key.describe());
        }

        if (!found_any)
        {
            LOG_DEBUG("[WeightManager][Phase2] Reclaim requested but no reclaimable tensors for: " << name);
        }

        return found_any;
    }

    WeightManager::WeightManager(IModelLoader &loader,
                                 std::shared_ptr<IMPIContext> mpi_ctx,
                                 std::shared_ptr<WeightPlacementMap> placement_map,
                                 WeightDistributionStrategy strategy,
                                 WeightPrecision weight_precision)
        : loader_(loader), mpi_ctx_(mpi_ctx), placement_map_(placement_map),
          strategy_(strategy), weight_precision_(weight_precision),
          weight_metadata_(std::make_shared<WeightMetadataRegistry>()),
          mmap_reclaim_lifecycle_([this]() { return performMmapReclaim(); })
    {
        // Source identity is immutable before any native, prepared, or prefix owner observes it.
        loader_.sealTensorSources();
        int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;

        if (rank == 0)
        {
            std::string strategy_name;
            switch (strategy_)
            {
            case WeightDistributionStrategy::REPLICATED:
                strategy_name = "REPLICATED (full copy per rank)";
                break;
            case WeightDistributionStrategy::SHARDED:
                strategy_name = "SHARDED (partitioned across ranks)";
                break;
            case WeightDistributionStrategy::INTERLEAVED:
                strategy_name = "INTERLEAVED (NUMA-aware global)";
                break;
            case WeightDistributionStrategy::LAYER_PARTITIONED:
                strategy_name = "LAYER_PARTITIONED (PP: only assigned layers)";
                break;
            }
            LOG_DEBUG("[WeightManager] Initialized with strategy: " << strategy_name);

            // Log weight precision mode
            const char *precision_name = "UNKNOWN";
            switch (weight_precision_)
            {
            case WeightPrecision::NATIVE:
                precision_name = "NATIVE (weights in original GGUF format, dequantize on-the-fly)";
                break;
            case WeightPrecision::CONVERT_TO_FP32:
                precision_name = "CONVERT_TO_FP32 (all weights dequantized to FP32 at load)";
                break;
            case WeightPrecision::CONVERT_TO_BF16:
                precision_name = "CONVERT_TO_BF16 (all weights dequantized to BF16 at load)";
                break;
            case WeightPrecision::CONVERT_TO_FP16:
                precision_name = "CONVERT_TO_FP16 (all weights dequantized to FP16 at load)";
                break;
            case WeightPrecision::CONVERT_TO_INT8:
                precision_name = "CONVERT_TO_INT8 (all weights dequantized to INT8 at load)";
                break;
            }
            LOG_DEBUG("[WeightManager] Weight precision: " << precision_name);

            if (strategy_ == WeightDistributionStrategy::SHARDED && mpi_ctx_)
            {
                LOG_DEBUG("[WeightManager] Sharding enabled with " << mpi_ctx_->world_size() << " ranks");
                LOG_DEBUG("[WeightManager] Column-parallel: Gate/Up (split output dim, local output)");
                LOG_DEBUG("[WeightManager] Input-parallel: Down (split input dim to match Gate/Up, allreduce after)");
                LOG_DEBUG("[WeightManager] Input-parallel: Wo (split input dim to match column-parallel QKV, allreduce after)");
                LOG_DEBUG("[WeightManager] Replicated: QKV (column-parallel via head sharding), norms, biases, embeddings, LM head");
            }
        }
    }

    WeightManager::~WeightManager()
    {
        try
        {
            // The worker may borrow cache tensors and loader mappings. Join it
            // while every WeightManager member and the owning ModelLoader are
            // still alive, before ordinary member destruction begins.
            (void)mmap_reclaim_lifecycle_.awaitBeforeHostAllocation();
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[WeightManager] asynchronous mmap reclaim failed during teardown: "
                      << error.what());
        }
        catch (...)
        {
            LOG_ERROR("[WeightManager] asynchronous mmap reclaim failed during teardown with an unknown error");
        }
    }

    WeightSliceSpec WeightManager::fullSliceSpec(const TensorBase &tensor) const
    {
        WeightSliceSpec spec;
        const auto &shape = tensor.shape();
        spec.source_rows = shape.empty() ? 0 : shape[0];
        spec.source_cols = shape.size() > 1 ? shape[1] : 1;
        spec.row_start = 0;
        spec.row_count = spec.source_rows;
        spec.col_start = 0;
        spec.col_count = spec.source_cols;
        if (shape.size() > 2)
        {
            spec.expert_start = 0;
            spec.expert_count = shape[2];
        }
        return spec;
    }

    std::vector<int> WeightManager::expectedRoutedExpertIds(
        const std::string &name,
        int participant_index,
        int participant_count,
        int layer_idx) const
    {
        if (participant_count <= 0 || participant_index < 0 ||
            participant_index >= participant_count)
        {
            throw std::runtime_error(
                "[WeightManager] Invalid routed-expert owner coordinates for " + name +
                ": participant=" + std::to_string(participant_index) + "/" +
                std::to_string(participant_count));
        }

        const auto dimensions = loader_.getTensorShape(name);
        if (!dimensions || dimensions->size() != 3u || (*dimensions)[2] == 0u)
        {
            throw std::runtime_error(
                "[WeightManager] Cannot establish routed-expert cache identity for " +
                name + ": expected a non-empty 3D source tensor");
        }

        const int resolved_layer_idx =
            layer_idx >= 0 ? layer_idx : inferWeightLayer(name);
        if (resolved_layer_idx < 0)
        {
            throw std::runtime_error(
                "[WeightManager] Cannot establish routed-expert cache identity for " +
                name + ": layer index is unknown");
        }

        return routed_expert_ownership::expertIdsForParticipant(
            static_cast<int>((*dimensions)[2]),
            participant_count,
            participant_index,
            resolved_layer_idx,
            routed_expert_assignment_.owner_order);
    }

    bool WeightManager::isCachedFullRoutedExpertSource(
        const std::string &name,
        const std::shared_ptr<TensorBase> &tensor) const
    {
        if (!tensor)
            return false;

        const auto dimensions = loader_.getTensorShape(name);
        const auto metadata = weight_metadata_->metadata(tensor.get());
        if (!dimensions || dimensions->size() != 3u || !metadata)
            return false;

        const auto derivation = metadata->identity.derivation;
        const bool source_derivation =
            derivation == WeightDerivationKind::Source ||
            derivation == WeightDerivationKind::DeviceClone;
        const auto &slice = metadata->slice;
        const bool unsliced_identity =
            metadata->identity.canonical_name == name &&
            source_derivation &&
            !slice.inner_is_presliced &&
            slice.expert_ids.empty() &&
            (slice.expert_count == 0u ||
             slice.expert_count == (*dimensions)[2]);
        return unsliced_identity && tensor->shape() == *dimensions;
    }

    void WeightManager::validateCachedRoutedExpertSlice(
        const std::string &name,
        const std::shared_ptr<TensorBase> &tensor,
        const std::vector<int> &expected_expert_ids,
        const std::string &cache_key) const
    {
        if (!tensor)
        {
            throw std::runtime_error(
                "[WeightManager] Routed-expert cache entry is null: " + cache_key);
        }

        const auto metadata = weight_metadata_->metadata(tensor.get());
        if (!metadata)
        {
            throw std::runtime_error(
                "[WeightManager] Routed-expert cache entry has no weight metadata: " +
                cache_key);
        }

        const auto &slice = metadata->slice;
        const auto &shape = tensor->shape();
        const bool identity_matches =
            metadata->identity.canonical_name == name &&
            metadata->identity.derivation == WeightDerivationKind::ExpertSlice;
        const bool slice_matches =
            slice.inner_is_presliced &&
            slice.expert_ids == expected_expert_ids &&
            slice.expert_count == expected_expert_ids.size() &&
            (!expected_expert_ids.empty() &&
             slice.expert_start == static_cast<size_t>(expected_expert_ids.front()));
        const bool shape_matches =
            shape.size() == 3u && shape[2] == expected_expert_ids.size();

        if (!identity_matches || !slice_matches || !shape_matches)
        {
            throw std::runtime_error(
                "[WeightManager] Fatal routed-expert cache identity mismatch for " +
                cache_key + ": expected_ids=" + formatExpertIds(expected_expert_ids) +
                " cached_ids=" + formatExpertIds(slice.expert_ids) +
                " derivation=" +
                std::to_string(static_cast<int>(metadata->identity.derivation)) +
                " packed_experts=" +
                std::to_string(shape.size() == 3u ? shape[2] : 0u));
        }
    }

    void WeightManager::registerSourceMetadata(
        const std::string &name,
        const std::shared_ptr<TensorBase> &tensor,
        DeviceId device)
    {
        if (!tensor || !weight_metadata_)
            return;

        weight_metadata_->registerSource(tensor.get(), name, device);

        WeightLifecycleTrace::record(
            WeightLifecycleEventType::SourceLoad,
            name,
            inferWeightRole(name),
            inferWeightLayer(name),
            device,
            "source tensor loaded");
    }

    void WeightManager::registerDerivedMetadata(
        const std::string &name,
        const std::shared_ptr<TensorBase> &tensor,
        WeightDerivationKind derivation,
        WeightSliceSpec slice,
        DeviceId device)
    {
        if (!tensor || !weight_metadata_)
            return;

        WeightIdentity identity = makeSourceWeightIdentity(name);
        identity.derivation = derivation;
        WeightResidency residency{device,device};
        if(tensor->home_device().is_cpu()) {
            const auto *view=dynamic_cast<const TensorSlice *>(tensor.get());
            if(view) if(const auto original=weight_metadata_->metadata(view->inner());
                original && original->identity.model_id.value) {
                identity=original->identity;
                residency=original->residency;
            }
        }
        weight_metadata_->registerWeight(tensor.get(), std::move(identity), slice, residency);

        WeightLifecycleTrace::record(
            derivation == WeightDerivationKind::DeviceClone ? WeightLifecycleEventType::Clone : WeightLifecycleEventType::Slice,
            name,
            inferWeightRole(name),
            inferWeightLayer(name),
            device,
            toString(derivation));
    }

    void WeightManager::registerCloneMetadata(
        const std::string &name,
        const std::shared_ptr<TensorBase> &source,
        const std::shared_ptr<TensorBase> &clone,
        DeviceId device)
    {
        if (!source || !clone || !weight_metadata_)
            return;

        if (!weight_metadata_->has(source.get()))
            registerSourceMetadata(name, source, DeviceId::cpu());

        if (!weight_metadata_->registerDerived(
                clone.get(), source.get(), WeightDerivationKind::DeviceClone, fullSliceSpec(*source), device))
        {
            registerDerivedMetadata(name, clone, WeightDerivationKind::DeviceClone, fullSliceSpec(*source), device);
        }

        WeightLifecycleTrace::record(
            WeightLifecycleEventType::Clone,
            name,
            inferWeightRole(name),
            inferWeightLayer(name),
            device,
            "device clone");
    }

    bool WeightManager::isGemmWeight(const std::string &name) const
    {
        const auto role = inferWeightRole(name);
        if (role == WeightRole::ImmutableRowTable || isImmutableFP32WeightRole(role)) return false;
        if (isFlashNextProjectionRole(role)) return true;
        if (!has_sharding_config_)
        {
            throw std::runtime_error(
                "[WeightManager] isGemmWeight called without sharding config. "
                "Call setWeightShardingConfig() with model schema first.");
        }
        return !sharding_config_.isNonGemmWeight(name);
    }

    bool WeightManager::hasOriginalCPUFinalOwner(const std::shared_ptr<TensorBase> &tensor,DeviceId device,
        const std::shared_ptr<PreparedWeightStore> &store) const
    {
        if(!tensor || !device.is_cpu() || !weight_metadata_)return false;
        const TensorBase *root=tensor.get();
        while(const auto *slice=dynamic_cast<const TensorSlice *>(root))root=slice->inner();
        const auto metadata=weight_metadata_->metadata(root);
        if(!metadata || !metadata->identity.model_id.value || !store ||
            store->physicalMemoryAuthority()!=physicalMemoryAuthority() || root->home_device()!=device)return false;
        for(const auto kind:{PreparedWeightKind::CpuPackedGemm,PreparedWeightKind::CPUResidentTensor}) {
            const auto binding=store->binding({metadata->identity.model_id,metadata->identity.instance_id,kind,device});
            if(binding && binding->tensor==root && binding->tensor_owner.get()==root &&
                binding->identity.sameSourceDefinition(metadata->identity) && binding->native_source && binding->immutable &&
                binding->prepared && binding->prepared->kind==kind && binding->prepared->device==device &&
                binding->prepared->binding_id==metadata->identity.instance_id &&
                binding->prepared->model_id==metadata->identity.model_id) {
                const bool raw_live=!root->is_raw_data_released() && root->raw_data()!=nullptr;
                if(kind==PreparedWeightKind::CPUResidentTensor)return raw_live;
                const auto engine=store->gemmKernelLifetime(*binding->prepared);
                return engine && engine->weights_converted() &&
                    (engine->canReleaseSourceWeightTensor() || raw_live);
            }
        }
        return false;
    }

    std::shared_ptr<TensorBase> WeightManager::originalGPUPreparationSource(
        const WeightRequirement &requirement,const std::string &name)
    {
        if(!requirement.target_device.is_gpu() || !requirement.lookup_device || !requirement.lookup_device->is_cpu() ||
            requirement.joint_source || requirement.role==WeightRole::ImmutableRowTable)
            throw std::invalid_argument("GPU preparation requires its declared original host-source lookup purpose");
        std::shared_ptr<TensorBase> source;
        if(requirement.bypass_tensor_parallel)source=getReplicatedWeight(name,requirement.target_device);
        else if(requirement.tp_domain>=0) {
            const auto original_tp=tensorParallelConfig();
            if(!original_tp || requirement.tp_rank_or_device_index<0)
                throw std::invalid_argument("GPU host-source lookup lost its original TP assignment");
            source=getShardedWeightForAssignment(name,requirement.target_device,
                original_tp->forRank(requirement.tp_rank_or_device_index),requirement.layer);
        } else source=getWeightForDevice(name,requirement.target_device,requirement.layer);
        if(!source)return source;
        const TensorBase *root=source.get();
        while(const auto *slice=dynamic_cast<const TensorSlice *>(root))root=slice->inner();
        bool cpu_final_purpose=false;
        const auto source_metadata=weight_metadata_?weight_metadata_->metadata(root):std::nullopt;
        const auto source_store=preparedWeightStoreIfInitialized();
        if(source_metadata && source_store) for(const auto kind:{PreparedWeightKind::CpuPackedGemm,PreparedWeightKind::CPUResidentTensor}) {
            const auto final=source_store->binding({source_metadata->identity.model_id,source_metadata->identity.instance_id,
                kind,*requirement.lookup_device});
            cpu_final_purpose=cpu_final_purpose || (final && final->tensor==root && final->native_source.has_value());
        }
        if(root->home_device()!=*requirement.lookup_device || cpu_final_purpose)
            throw std::invalid_argument("GPU host-source lookup is not the original CPU source owner");
        // Source residence is a real host property. The frozen requirement below
        // independently carries execution placement; no CPU final consumer was born.
        // Retired original metadata remains valid for an exact model-owned
        // prepared override hit. The consumer, not this address classifier,
        // requires live bytes on a real preparation miss; no source is reloaded.
        if(!root->is_raw_data_released() && root->raw_data())source->setHostResident();
        return source;
    }

    std::shared_ptr<TensorBase> WeightManager::loadCPUWeightRange(
        const std::string &name, DeviceId device, WeightDerivationKind derivation,
        WeightSliceSpec slice, std::optional<NativeWeightSourceAssembly> assembly, int original_tp_rank)
    {
        if (!device.is_cpu())
            throw std::invalid_argument("CPU source range requires its original CPU allocator");
        if (!loader_.hasTensor(name)) return nullptr;
        const auto model = preparedWeightStore()->modelId();
        if (!model.value || !physicalMemoryAuthority())
            throw std::logic_error("CPU source range requires original bound model/preparation authority");
        const auto role = inferWeightRole(name);
        if (isRoutedExpertRole(role))
            throw std::logic_error("Prepared CPU experts are consumed through their complete original registry bank, not a raw parent");
        WeightRequirement requirement;
        requirement.canonical_name = name;
        requirement.role = role;
        requirement.target_device = device;
        requirement.layer = inferWeightLayer(name);
        requirement.derivation = derivation;
        requirement.slice = std::move(slice);
        requirement.native_source_assembly = std::move(assembly);
        InferenceStrategy strategy;
        strategy.model_id = model;
        strategy.devices = {device};
        if (original_tp_rank>=0) {
            const auto original_tp=tensorParallelConfig();
            const int degree=original_tp?original_tp->worldSize():mpi_ctx_?mpi_ctx_->world_size():0;
            if (original_tp) (void)original_tp->forRank(original_tp_rank);
            else if (!mpi_ctx_ || mpi_ctx_->rank()!=original_tp_rank)
                throw std::invalid_argument("CPU source replica has no original TP participant authority");
            if(degree<=0 || original_tp_rank>=degree)
                throw std::invalid_argument("CPU source replica exceeds its original TP participant extent");
            requirement.tp_domain=0;
            requirement.tp_rank_or_device_index=original_tp_rank;
            strategy.tp_degree=degree;
        }
        WeightPlan plan(strategy);
        plan.add(std::move(requirement));
        const auto completed = materialize(plan);
        if (completed.bindings().size()!=1 || !completed.bindings().front().tensor_owner)
            throw std::logic_error("CPU source range did not commit exactly one final consumer");
        const auto &binding=completed.bindings().front();
        if (weight_metadata_)
            weight_metadata_->registerWeight(binding.tensor,binding.identity,binding.slice,binding.residency);
        return binding.tensor_owner;
    }

    std::shared_ptr<TensorBase> WeightManager::getReplicatedWeight(const std::string &name, DeviceId device, int original_tp_rank)
    {
        if (device.is_cpu())
            return loadCPUWeightRange(name,device,WeightDerivationKind::Source,{},{},original_tp_rank);
        // Phase 1: Simple replication - each rank loads independently
        // No MPI coordination needed

        auto tensor = loader_.loadTensor(name, device,
            isImmutableFP32WeightRole(inferWeightRole(name)) ? WeightPrecision::CONVERT_TO_FP32 : weight_precision_);
        if (!tensor)
        {
            // LOG_DEBUG, not LOG_ERROR: returning nullptr is a valid result for optional weights.
            // The caller (InferenceRunnerFactory) decides severity based on schema context.
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            LOG_DEBUG("[WeightManager] Rank " << rank << " weight not found: " << name);
            return nullptr;
        }

        registerSourceMetadata(name, tensor, device);
        return tensor;
    }

    ShardingMode WeightManager::toShardingMode(WeightShardingMode mode)
    {
        switch (mode)
        {
        case WeightShardingMode::ColumnParallel:
            return ShardingMode::COLUMN_PARALLEL;
        case WeightShardingMode::RowParallel:
            return ShardingMode::ROW_PARALLEL;
        case WeightShardingMode::InputParallel:
            return ShardingMode::INPUT_PARALLEL;
        case WeightShardingMode::ExpertIdApportioned:
            return ShardingMode::EXPERT_ID_APPORTIONED;
        case WeightShardingMode::Replicate:
        default:
            return ShardingMode::REPLICATE;
        }
    }

    ShardingMode WeightManager::determineShardingMode(const std::string &name) const
    {
        if (!has_sharding_config_)
        {
            throw std::runtime_error(
                "[WeightManager] determineShardingMode called without sharding config. "
                "Call setWeightShardingConfig() with model schema first.");
        }
        WeightShardingMode mode = sharding_config_.getMode(name);
        return toShardingMode(mode);
    }

    ShardingMode WeightManager::getShardingMode(const std::string &name) const
    {
        std::lock_guard<std::mutex> lock(sharding_mode_cache_mutex_);

        // Check cache first
        auto it = sharding_mode_cache_.find(name);
        if (it != sharding_mode_cache_.end())
        {
            return it->second;
        }

        // Determine and cache (cache is mutable)
        ShardingMode mode = determineShardingMode(name);
        sharding_mode_cache_[name] = mode;
        return mode;
    }

    bool WeightManager::isWeightSharded(const std::string &name) const
    {
        if (strategy_ != WeightDistributionStrategy::SHARDED)
        {
            return false;
        }
        ShardingMode mode = getShardingMode(name);
        return mode != ShardingMode::REPLICATE;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceColumns(
        const std::shared_ptr<TensorBase> &full_tensor,
        int rank, int world_size)
    {
        // For weight matrix [out_dim, in_dim], extract [out_local, in_dim]
        // where out_local = out_dim / world_size for this rank

        const auto &shape = full_tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] Column slicing requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];
        size_t out_local = out_dim / world_size;
        size_t out_start = out_local * rank;

        // Handle remainder: last rank gets extra rows if not evenly divisible
        if (rank == world_size - 1)
        {
            out_local = out_dim - out_start;
        }

        LOG_DEBUG("[WeightManager] Column slicing: [" << out_dim << ", " << in_dim
                                                      << "] -> rank " << rank << " gets rows [" << out_start << ", "
                                                      << (out_start + out_local) << ")");

        // Currently only FP32 slicing is supported
        // For quantized tensors, we dequantize first then slice
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] Column slicing currently requires FP32 tensor. "
                                     "Use CONVERT_TO_FP32 weight precision for sharded strategy.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {out_local, in_dim};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy data rows
        const float *src = fp32_tensor->data() + out_start * in_dim;
        float *dst = sliced->mutable_data();
        std::memcpy(dst, src, out_local * in_dim * sizeof(float));

        return sliced;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceRows(
        const std::shared_ptr<TensorBase> &full_tensor,
        int rank, int world_size)
    {
        // For weight matrix [out_dim, in_dim], extract [out_dim, in_local]
        // where in_local = in_dim / world_size for this rank

        const auto &shape = full_tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] Row slicing requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];
        size_t in_local = in_dim / world_size;
        size_t in_start = in_local * rank;

        // Handle remainder: last rank gets extra columns if not evenly divisible
        if (rank == world_size - 1)
        {
            in_local = in_dim - in_start;
        }

        LOG_DEBUG("[WeightManager] Row slicing: [" << out_dim << ", " << in_dim
                                                   << "] -> rank " << rank << " gets cols [" << in_start << ", "
                                                   << (in_start + in_local) << ")");

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] Row slicing currently requires FP32 tensor. "
                                     "Use CONVERT_TO_FP32 weight precision for sharded strategy.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {out_dim, in_local};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy data - need to extract columns, which is non-contiguous
        const float *src = fp32_tensor->data();
        float *dst = sliced->mutable_data();

#pragma omp parallel for
        for (size_t row = 0; row < out_dim; ++row)
        {
            const float *src_row = src + row * in_dim + in_start;
            float *dst_row = dst + row * in_local;
            std::memcpy(dst_row, src_row, in_local * sizeof(float));
        }

        return sliced;
    }

    bool WeightManager::hostDataRequired(const TensorBase *tensor, const std::string &key) const
    {
        // Phase 9 final: All expert GEMM preparation happens upfront at graph-build
        // time. No lazy host-data fallback exists. Expert weights are always
        // releasable after device preparation completes. Dynamic rebalancing uses
        // serialized transfer blobs, never raw host tensor data.
        //
        // The only case where host data must be retained is explicit CPU execution
        // (RequiredForCPUExecution policy in metadata).

        if (weight_metadata_ && weight_metadata_->has(tensor))
        {
            auto res = weight_metadata_->residency(tensor);
            if (res)
            {
                switch (res->host_policy)
                {
                case WeightHostPolicy::RequiredForCPUExecution:
                    return true;
                case WeightHostPolicy::Released:
                case WeightHostPolicy::ReleasableAfterPreparation:
                case WeightHostPolicy::RequiredUntilPreparedOrTransferred:
                case WeightHostPolicy::RequiredUntilGraphMaterialized:
                    return false;
                }
            }
        }

        // No metadata or no explicit policy: host data is not required.
        // All preparation (GEMM packing, embedding prep, GPU upload) completes
        // before releaseAllHostWeightData() is called.
        return false;
    }



    std::shared_ptr<TensorBase> WeightManager::getShardedWeight(const std::string &name, DeviceId device)
    {
        const auto tp_config = tensorParallelConfig();
        if(device.is_cpu() && isRoutedExpertRole(inferWeightRole(name)))
            throw std::logic_error("CPU expert source access requires its complete original prepared registry bank, not a raw 3D parent");
        // For SHARDED strategy:
        // 1. Determine sharding mode based on weight name
        // 2. Load full tensor in native format (quantized)
        // 3. Wrap in TensorSlice with sharding metadata
        // 4. Slicing happens during kernel creation (preserves quantization)

        if (!mpi_ctx_ || mpi_ctx_->world_size() == 1)
        {
            // Single rank: no sharding needed
            return getReplicatedWeight(name, device);
        }

        ShardingMode mode = getShardingMode(name);
        int rank = mpi_ctx_->rank();
        int world_size = mpi_ctx_->world_size();

        /*
         * Global MPI TP and LocalTP must materialize exactly the same packed
         * modulo-linked GDN layout on every backend. Resolve the source
         * geometry before the generic rank/world slicer so no device can
         * silently replicate Q/K or select a different V-head order.
         */
        if (has_gdn_dimensions_ && world_size > 1)
        {
            const auto dimensions = loader_.getTensorShape(name);
            if (dimensions && !dimensions->empty())
            {
                const GDNHeadAssignment head_assignment =
                    gdnHeadAssignmentForEqualRank(rank, world_size);
                const size_t expected_fused_rows =
                    (2 * static_cast<size_t>(gdn_n_k_heads_) +
                     static_cast<size_t>(gdn_n_v_heads_)) *
                    static_cast<size_t>(gdn_d_state_);

                if (mode == ShardingMode::COLUMN_PARALLEL &&
                    dimensions->size() == 2 &&
                    has_sharding_config_ &&
                    sharding_config_.getDimensionType(name) ==
                        WeightDimensionType::FusedQKVHeads &&
                    (*dimensions)[0] == expected_fused_rows)
                {
                    return loadGDNFusedQKVColumnParallel(
                        name, device, head_assignment, rank, world_size,
                        *dimensions);
                }

                if (mode == ShardingMode::COLUMN_PARALLEL &&
                    (dimensions->size() == 1 || dimensions->size() == 2) &&
                    gdnValueElementsPerHead(name, (*dimensions)[0]) > 0)
                {
                    return loadGDNValueRows(
                        name, device, head_assignment, rank, world_size,
                        *dimensions);
                }

                if (mode == ShardingMode::INPUT_PARALLEL &&
                    dimensions->size() == 2 &&
                    gdnValueElementsPerHead(name, (*dimensions)[1]) > 0)
                {
                    return loadGDNValueColumns(
                        name, device, head_assignment, rank, world_size,
                        *dimensions);
                }
            }
        }

        if (mode == ShardingMode::REPLICATE)
        {
            // This weight should not be sharded (norms, biases, etc.)
            return getReplicatedWeight(name, device, rank);
        }

        if (mode == ShardingMode::ROW_PARALLEL)
        {
            // Row-parallel: Load ONLY the slice from GGUF (memory efficient!)
            // Get tensor dimensions
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for row-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " invalid tensor for row-parallel (expected 2D): " + name);
            }
            const auto &dims = *dims_opt;

            size_t total_rows = dims[0];
            size_t cols = dims[1];

            // Calculate row range for this rank
            // Use proportional slicing if TensorParallelConfig is set, otherwise equal split
            size_t row_start, row_count;
            if (tp_config)
            {
                // Row-parallel actually splits ROWS (for Wo, which has shape [d_model, n_heads*head_dim])
                // But the input dimension to match is from the column-parallel output
                // For Wo: we split rows based on head assignment since input comes from Q output
                auto [start, count] = calculateProportionalRowSlice(name, total_rows, *tp_config);
                row_start = start;
                row_count = count;
            }
            else
            {
                size_t rows_per_rank = total_rows / world_size;
                row_start = rows_per_rank * rank;
                row_count = (rank == world_size - 1) ? (total_rows - row_start) : rows_per_rank;
            }
            size_t row_end = row_start + row_count;

            // Load ONLY the slice from GGUF file (memory efficient!)
            auto slice_tensor = (device.is_cpu() ? loadCPUWeightRange(name,device,WeightDerivationKind::RowSlice,
                WeightSliceSpec{.row_start=row_start,.row_count=(row_end)-(row_start),.inner_is_presliced=true},{},rank) : loader_.loadTensorRowSlice(
                name, row_start, row_end, device, WeightPrecision::NATIVE));

            if (!slice_tensor)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " failed to load row slice for: " + name);
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " row-parallel " << name
                                              << " [" << total_rows << ", " << cols
                                              << "] -> loaded ONLY rows [" << row_start << ", " << row_end
                                              << ") = " << row_count << " rows"
                                              << (tp_config ? " (proportional)" : " (equal)"));

            // Wrap in TensorSlice with metadata (inner_is_presliced=true)
            // The inner tensor already contains only the slice data
            auto meta = SliceMetadata::forRowParallel(
                total_rows, cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
            WeightSliceSpec row_slice;
            row_slice.source_rows = total_rows;
            row_slice.source_cols = cols;
            row_slice.row_start = row_start;
            row_slice.row_count = row_count;
            row_slice.col_start = 0;
            row_slice.col_count = cols;
            row_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, slice, WeightDerivationKind::RowSlice, row_slice, device);
            return slice;
        }
        else if (mode == ShardingMode::COLUMN_PARALLEL)
        {
            // Column-parallel: split the OUTPUT dimension of the weight matrix
            //
            // Terminology clarification:
            // - "Column-parallel" in tensor parallelism means each rank produces independent
            //   columns of the OUTPUT tensor (Y = X @ W^T)
            // - For weight matrix W of shape [N, K] (N=output features, K=input features):
            //   * Column-parallel splits N (the rows of W), NOT K (the columns)
            //   * Each rank loads rows [row_start, row_end) of W
            //   * GEMM output is [M, N/world_size], then concatenated via AllGather
            //
            // For FFN Gate/Up weights [d_ff=4864, d_model=896]:
            //   * Each rank loads [d_ff/2=2432, d_model=896]
            //   * GEMM input: [seq, d_model=896]
            //   * GEMM output: [seq, d_ff/2=2432]
            //
            // For 1D bias tensors [N] (e.g., Q/K/V biases):
            //   * Each rank loads elements [start, end) where N is split across ranks

            int rank = mpi_ctx_->rank();
            int world_size = mpi_ctx_->world_size();

            // Get tensor dimensions
            // For tied embeddings: if output.weight is missing, substitute token_embd.weight
            std::string load_name = name;
            auto dims_opt = loader_.getTensorShape(name);
            if ((!dims_opt || dims_opt->empty()) && name == "output.weight")
            {
                auto embd_dims = loader_.getTensorShape("token_embd.weight");
                if (embd_dims && embd_dims->size() == 2)
                {
                    LOG_DEBUG("[WeightManager] Rank " << rank
                                                      << " output.weight not in GGUF — using tied embedding "
                                                      << "token_embd.weight as column-parallel LM head");
                    dims_opt = embd_dims;
                    load_name = "token_embd.weight";
                }
            }
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for column-parallel: " << name);
                return nullptr;
            }
            const auto &dims = *dims_opt;

            // Handle 1D tensors (biases)
            // For biases like Q/K/V biases [n_heads * head_dim], we load full then slice in memory.
            // This is acceptable since biases are tiny (e.g., 896 floats = 3.5KB).
            if (dims.size() == 1)
            {
                size_t total_elements = dims[0];

                // Calculate slice range - use proportional if config set
                size_t start, slice_elements;
                if (tp_config)
                {
                    // Use column slice calculation (1D bias treated as single-row 2D)
                    auto [s, count] = calculateProportionalColumnSlice(name, total_elements, *tp_config);
                    start = s;
                    slice_elements = count;
                }
                else
                {
                    size_t elements_per_rank = total_elements / world_size;
                    start = elements_per_rank * rank;
                    slice_elements = (rank == world_size - 1) ? (total_elements - start) : elements_per_rank;
                }
                size_t end = start + slice_elements;

                if (device.is_cpu())
                    return loadCPUWeightRange(name,device,WeightDerivationKind::RowSlice,
                        WeightSliceSpec{.row_start=start,.row_count=slice_elements,.inner_is_presliced=true},{},rank);
                // Load full 1D tensor first (biases are small, ~3KB)
                auto full_tensor = loader_.loadTensor(name, device,
                    isImmutableFP32WeightRole(inferWeightRole(name)) ? WeightPrecision::CONVERT_TO_FP32 : WeightPrecision::NATIVE);
                if (!full_tensor)
                {
                    throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                             " failed to load 1D tensor for: " + name);
                }

                // Cast to FP32 (biases are stored as FP32)
                auto *fp32_full = dynamic_cast<FP32Tensor *>(full_tensor.get());
                if (!fp32_full)
                {
                    throw std::runtime_error("[WeightManager] 1D tensor is not FP32 for: " + name);
                }

                // Create sliced 1D tensor
                auto slice_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{slice_elements});
                std::memcpy(slice_tensor->mutable_data(),
                            fp32_full->data() + start,
                            slice_elements * sizeof(float));

                LOG_TRACE("[WeightManager] Rank " << rank << " column-parallel 1D " << name
                                                  << " [" << total_elements
                                                  << "] -> sliced elements [" << start << ", " << end
                                                  << ") = " << slice_elements << " elements"
                                                  << (tp_config ? " (proportional)" : " (equal)"));

                // For 1D tensors, use row-parallel metadata with cols=1
                auto meta = SliceMetadata::forRowParallel(
                    total_elements, 1, rank, world_size,
                    true /* inner_is_presliced */);

                auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
                return slice;
            }

            // Handle 2D tensors (weight matrices)
            if (dims.size() != 2)
            {
                throw std::runtime_error("[WeightManager] Column-parallel requires 1D or 2D tensor, got " +
                                         std::to_string(dims.size()) + "D for: " + name);
            }

            size_t total_rows = dims[0]; // N = output features
            size_t cols = dims[1];       // K = input features

            // FusedQKVHeads requires special sub-block slicing: the weight is
            // [Q_all | K_all | V_all] vertically. A simple contiguous row slice
            // crosses sub-block boundaries. Instead, split each sub-block independently.
            //
            // GQA support: Q, K, V sub-blocks may have DIFFERENT sizes when
            // n_kv_heads < n_heads. We compute actual sub-block sizes from
            // model dimensions when available.
            //
            // GDN (non-QKV) weights tagged FusedQKVHeads may have total_rows
            // that doesn't match the expected Q+K+V size. These fall back to
            // simple equal row splitting.
            if (has_sharding_config_ &&
                sharding_config_.getDimensionType(name) == WeightDimensionType::FusedQKVHeads)
            {
                // Compute expected Q, K, V row counts from model dimensions
                bool use_sub_block_slicing = false;
                size_t q_rows = 0, kv_rows = 0, v_rows = 0;

                if (has_model_dimensions_ && model_n_heads_ > 0 && model_head_dim_ > 0)
                {
                    q_rows = static_cast<size_t>(model_n_heads_) * model_head_dim_;
                    kv_rows = static_cast<size_t>(model_n_kv_heads_) * model_head_dim_;
                    const size_t expected_qkv = q_rows + 2 * kv_rows;

                    if (total_rows == expected_qkv)
                    {
                        // True GQA layout: [Q(n_heads*hd) | K(n_kv_heads*hd) | V(n_kv_heads*hd)]
                        v_rows = kv_rows;
                        use_sub_block_slicing = true;
                        LOG_TRACE("[WeightManager] FusedQKV " << name
                                                              << " Q=" << q_rows << " K=" << kv_rows << " V=" << v_rows
                                                              << " (GQA: n_heads=" << model_n_heads_
                                                              << " n_kv_heads=" << model_n_kv_heads_ << ")");
                    }
                    else if (total_rows % 3 == 0)
                    {
                        // Some models (e.g. Qwen3.5) store K/V at full n_heads size in the
                        // fused weight, applying GQA only at attention time. Fall back to
                        // 3 equal sub-blocks.
                        q_rows = total_rows / 3;
                        kv_rows = total_rows / 3;
                        v_rows = total_rows / 3;
                        use_sub_block_slicing = true;
                        LOG_TRACE("[WeightManager] FusedQKV " << name
                                                              << " total_rows=" << total_rows
                                                              << " != expected GQA=" << expected_qkv
                                                              << " but divisible by 3 — using 3 equal sub-blocks"
                                                              << " (" << q_rows << " each)");
                    }
                    else
                    {
                        // Try GDN layout: [Q(n_k*d) | K(n_k*d) | V(n_v*d)]
                        // GDN has asymmetric Q/K/V when n_k_heads != n_v_heads
                        if (has_gdn_dimensions_ && gdn_n_k_heads_ > 0 && gdn_d_state_ > 0)
                        {
                            const size_t gdn_key_dim = static_cast<size_t>(gdn_n_k_heads_) * gdn_d_state_;
                            const size_t gdn_value_dim = static_cast<size_t>(gdn_n_v_heads_) * gdn_d_state_;
                            const size_t expected_gdn_qkv = 2 * gdn_key_dim + gdn_value_dim;

                            if (total_rows == expected_gdn_qkv)
                            {
                                q_rows = gdn_key_dim;   // Q = n_k_heads * d_state
                                kv_rows = gdn_key_dim;  // K = n_k_heads * d_state (same as Q)
                                v_rows = gdn_value_dim; // V = n_v_heads * d_state (may differ from K)
                                use_sub_block_slicing = true;

                                LOG_TRACE("[WeightManager] FusedQKV " << name
                                                                      << " matched GDN layout: Q=" << q_rows
                                                                      << " K=" << kv_rows << " V=" << gdn_value_dim
                                                                      << " (n_k=" << gdn_n_k_heads_
                                                                      << " n_v=" << gdn_n_v_heads_
                                                                      << " d=" << gdn_d_state_ << ")");
                            }
                            else
                            {
                                LOG_DEBUG("[WeightManager] FusedQKV " << name
                                                                      << " total_rows=" << total_rows
                                                                      << " != expected Q+K+V=" << expected_qkv
                                                                      << " and not GDN=" << expected_gdn_qkv
                                                                      << " and not divisible by 3"
                                                                      << " — using simple equal row split");
                            }
                        }
                        else
                        {
                            LOG_DEBUG("[WeightManager] FusedQKV " << name
                                                                  << " total_rows=" << total_rows
                                                                  << " != expected Q+K+V=" << expected_qkv
                                                                  << " and not divisible by 3"
                                                                  << " — using simple equal row split (non-standard fused weight)");
                        }
                    }
                }
                else
                {
                    // No model dimensions: try 3 equal sub-blocks (non-GQA fallback)
                    if (total_rows % 3 == 0)
                    {
                        q_rows = total_rows / 3;
                        kv_rows = total_rows / 3;
                        v_rows = total_rows / 3;
                        use_sub_block_slicing = true;
                    }
                    else
                    {
                        LOG_DEBUG("[WeightManager] FusedQKV " << name
                                                              << " total_rows=" << total_rows
                                                              << " not divisible by 3 and no model dimensions"
                                                              << " — using simple equal row split");
                    }
                }

                if (use_sub_block_slicing)
                {
                    // Sub-block sizes: [Q_rows, K_rows, V_rows]
                    // For GDN layers, V may differ from K (n_v_heads != n_k_heads)
                    const size_t sub_block_sizes[3] = {q_rows, kv_rows, v_rows > 0 ? v_rows : kv_rows};

                    // GDN modular repeat (repeat_type=1): v_head j uses k_head j%n_k.
                    // With contiguous V sharding, v_heads on one rank may need k_heads
                    // from another rank. Fix: replicate Q and K sub-blocks on every rank,
                    // only shard V. This ensures all k_heads are locally available.
                    const bool gdn_replicate_qk = has_gdn_dimensions_ && gdn_n_k_heads_ > 0 && gdn_n_v_heads_ > gdn_n_k_heads_ && (total_rows == 2 * static_cast<size_t>(gdn_n_k_heads_) * gdn_d_state_ + static_cast<size_t>(gdn_n_v_heads_) * gdn_d_state_);

                    // Compute per-rank slice within each sub-block
                    // For GDN with replicated Q/K: Q and K are loaded in full, only V is sharded.
                    auto compute_slice = [rank, world_size, gdn_replicate_qk](size_t block_rows, int sub_block_idx) -> std::pair<size_t, size_t>
                    {
                        // Sub-blocks 0 (Q) and 1 (K): replicate for GDN, shard otherwise
                        if (gdn_replicate_qk && sub_block_idx < 2)
                            return {0, block_rows}; // Full sub-block (replicated)

                        size_t rows_per_rank = block_rows / world_size;
                        size_t start = rows_per_rank * rank;
                        size_t count = (rank == world_size - 1)
                                           ? (block_rows - start)
                                           : rows_per_rank;
                        return {start, count};
                    };

                    // Validate sub-block divisibility BEFORE loading slices.
                    // If any sharded sub-block has fewer rows than the world_size,
                    // the model is too small for this TP degree — fail early and clearly.
                    {
                        static constexpr const char *sub_names[3] = {"Q", "K", "V"};
                        for (size_t s = 0; s < 3; s++)
                        {
                            // Replicated sub-blocks (GDN Q/K) are always valid
                            if (gdn_replicate_qk && s < 2)
                                continue;

                            if (sub_block_sizes[s] % static_cast<size_t>(world_size) != 0)
                            {
                                std::ostringstream err;
                                err << "[WeightManager] Cannot shard FusedQKV weight '" << name
                                    << "': sub-block " << sub_names[s]
                                    << " has " << sub_block_sizes[s]
                                    << " rows, which is not evenly divisible by TP degree "
                                    << world_size << ". ";
                                if (gdn_replicate_qk)
                                    err << "GDN modular repeat replicates Q/K, but V ("
                                        << sub_block_sizes[2] << " rows = "
                                        << gdn_n_v_heads_ << " heads * " << gdn_d_state_
                                        << " d_state) must be divisible by " << world_size << ". ";
                                err << "Reduce the TP degree or use a larger model.";
                                LOG_ERROR(err.str());
                                throw std::invalid_argument(err.str());
                            }

                            if (sub_block_sizes[s] / static_cast<size_t>(world_size) == 0)
                            {
                                std::ostringstream err;
                                err << "[WeightManager] Cannot shard FusedQKV weight '" << name
                                    << "': sub-block " << sub_names[s]
                                    << " has " << sub_block_sizes[s]
                                    << " rows but TP degree is " << world_size
                                    << " — each rank would get 0 rows. "
                                    << "Reduce the TP degree or use a larger model.";
                                LOG_ERROR(err.str());
                                throw std::invalid_argument(err.str());
                            }
                        }
                    }

                    if (device.is_cpu()) {
                        std::vector<GDNHeadSpan> spans;
                        size_t source_offset=0;
                        for (size_t block=0;block<3;++block) {
                            const auto [first,count]=compute_slice(sub_block_sizes[block],int(block));
                            if (source_offset+first>size_t(std::numeric_limits<int>::max()) || count>size_t(std::numeric_limits<int>::max()))
                                throw std::invalid_argument("CPU fused-QKV source stripe exceeds original head coordinate range");
                            spans.push_back({.start=int(source_offset+first),.count=int(count)});
                            source_offset+=sub_block_sizes[block];
                        }
                        auto packed=loadNativeRowSpanConcat(load_name,device,spans,cols,rank);
                        auto result=std::make_shared<TensorSlice>(std::move(packed),
                            SliceMetadata::forColumnParallel(total_rows,cols,rank,world_size,true));
                        WeightSliceSpec original_slice;
                        original_slice.source_rows=total_rows;original_slice.source_cols=cols;
                        original_slice.row_count=result->shape()[0];original_slice.col_count=cols;
                        original_slice.inner_is_presliced=true;
                        registerDerivedMetadata(name,result,WeightDerivationKind::FusedSubblockConcat,original_slice,device);
                        return result;
                    }

                    // Load each sub-block's slice from GGUF
                    std::shared_ptr<TensorBase> slices[3];
                    size_t total_out_rows = 0;
                    size_t abs_offset = 0; // Running offset into the weight

                    for (size_t s = 0; s < 3; s++)
                    {
                        auto [local_start, local_count] = compute_slice(sub_block_sizes[s], static_cast<int>(s));
                        const size_t abs_row_start = abs_offset + local_start;
                        const size_t abs_row_end = abs_row_start + local_count;

                        slices[s] = loader_.loadTensorRowSlice(
                            load_name, abs_row_start, abs_row_end, device, WeightPrecision::NATIVE);

                        if (!slices[s])
                        {
                            throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                                     " failed to load fused-QKV sub-block " + std::to_string(s) +
                                                     " for: " + name);
                        }

                        total_out_rows += local_count;
                        abs_offset += sub_block_sizes[s];
                    }

                    // Concatenate raw bytes from the 3 sub-block slices
                    size_t total_bytes = 0;
                    for (size_t s = 0; s < 3; s++)
                        total_bytes += slices[s]->size_bytes();

                    std::vector<uint8_t> combined_raw(total_bytes);
                    size_t byte_offset = 0;
                    for (size_t s = 0; s < 3; s++)
                    {
                        const void *src = slices[s]->raw_data();
                        if (!src)
                        {
                            throw std::runtime_error("[WeightManager] Null raw_data for fused-QKV sub-block " +
                                                     std::to_string(s) + ": " + name);
                        }
                        std::memcpy(combined_raw.data() + byte_offset, src, slices[s]->size_bytes());
                        byte_offset += slices[s]->size_bytes();
                    }

                    std::vector<size_t> out_shape = {total_out_rows, cols};
                    TensorType native_type = slices[0]->native_type();

                    auto result_tensor = createTensorFromRawData(
                        native_type, out_shape, std::move(combined_raw));

                    if (!result_tensor)
                    {
                        throw std::runtime_error("[WeightManager] Failed to create fused-QKV tensor for: " + name);
                    }

                    auto meta = SliceMetadata::forColumnParallel(
                        total_rows, cols, rank, world_size,
                        true /* inner_is_presliced */);

                    auto result = std::make_shared<TensorSlice>(
                        std::shared_ptr<TensorBase>(std::move(result_tensor)), meta);

                    WeightSliceSpec fused_slice;
                    fused_slice.source_rows = total_rows;
                    fused_slice.source_cols = cols;
                    fused_slice.row_start = 0;
                    fused_slice.row_count = total_out_rows;
                    fused_slice.col_start = 0;
                    fused_slice.col_count = cols;
                    fused_slice.inner_is_presliced = true;
                    registerDerivedMetadata(name, result, WeightDerivationKind::FusedSubblockConcat, fused_slice, device);

                    LOG_DEBUG("[WeightManager] Rank " << rank
                                                      << " fused-QKV column-parallel " << name
                                                      << " [" << total_rows << ", " << cols << "]"
                                                      << " -> [" << total_out_rows << ", " << cols << "]"
                                                      << " (Q=" << sub_block_sizes[0]
                                                      << " K=" << sub_block_sizes[1]
                                                      << " V=" << sub_block_sizes[2] << ")");

                    return result;
                }
                // Fall through to simple equal row split below
            }

            // Calculate this rank's ROW range (splitting output dimension)
            // Use proportional slicing if TensorParallelConfig is set, otherwise equal split
            size_t row_start, row_count;
            if (tp_config)
            {
                auto [start, count] = calculateProportionalColumnSlice(name, total_rows, *tp_config);
                row_start = start;
                row_count = count;
            }
            else
            {
                size_t rows_per_rank = total_rows / world_size;
                row_start = rows_per_rank * rank;
                row_count = (rank == world_size - 1) ? (total_rows - row_start) : rows_per_rank;
            }
            size_t row_end = row_start + row_count;

            // Load only this rank's row slice (preserves quantized format)
            auto slice_tensor = (device.is_cpu() ? loadCPUWeightRange(load_name,device,WeightDerivationKind::RowSlice,
                WeightSliceSpec{.row_start=row_start,.row_count=(row_end)-(row_start),.inner_is_presliced=true},{},rank) : loader_.loadTensorRowSlice(
                load_name, row_start, row_end, device, WeightPrecision::NATIVE));

            if (!slice_tensor)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " failed to load row slice for: " + name);
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " column-parallel " << name
                                              << " [" << total_rows << ", " << cols
                                              << "] -> loaded ONLY rows [" << row_start << ", " << row_end
                                              << ") = " << row_count << " rows"
                                              << (tp_config ? " (proportional)" : " (equal)"));

            // Wrap in TensorSlice with metadata (inner_is_presliced=true)
            // Note: forColumnParallel stores original dimensions for allgather reconstruction
            auto meta = SliceMetadata::forColumnParallel(
                total_rows, cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
            WeightSliceSpec column_slice;
            column_slice.source_rows = total_rows;
            column_slice.source_cols = cols;
            column_slice.row_start = row_start;
            column_slice.row_count = row_count;
            column_slice.col_start = 0;
            column_slice.col_count = cols;
            column_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, slice, WeightDerivationKind::RowSlice, column_slice, device);
            return slice;
        }
        else if (mode == ShardingMode::INPUT_PARALLEL)
        {
            // Input-parallel: split the INPUT dimension of the weight matrix
            //
            // This is used for FFN Down when Gate/Up are column-parallel:
            // - Gate/Up output: [seq, d_ff_local] where d_ff_local = d_ff / world_size
            // - Down weight: [d_model, d_ff] → each rank needs [d_model, d_ff_local]
            // - Down output: [seq, d_model], needs allreduce-sum across ranks
            //
            // For weight matrix W of shape [N, K] (N=output, K=input):
            //   * Input-parallel splits K (the columns of W)
            //   * Each rank loads cols [col_start, col_end) of W
            //   * GEMM input is [M, K_local], output is [M, N]
            //   * Outputs are allreduce-summed

            int rank = mpi_ctx_->rank();
            int world_size = mpi_ctx_->world_size();

            // Get tensor dimensions
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for input-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                throw std::runtime_error("[WeightManager] Invalid tensor for input-parallel (expected 2D): " + name);
            }
            const auto &dims = *dims_opt;

            size_t rows = dims[0];       // N = output features (d_model)
            size_t total_cols = dims[1]; // K = input features (d_ff)

            // Calculate this rank's COLUMN range (splitting input dimension)
            // Use proportional slicing if TensorParallelConfig is set, otherwise equal split
            size_t col_start, col_count;
            if (tp_config)
            {
                auto [start, count] = calculateProportionalRowSlice(name, total_cols, *tp_config);
                col_start = start;
                col_count = count;
            }
            else
            {
                size_t cols_per_rank = total_cols / world_size;
                col_start = cols_per_rank * rank;
                col_count = (rank == world_size - 1) ? (total_cols - col_start) : cols_per_rank;
            }
            size_t col_end = col_start + col_count;

            // Load only this rank's column slice (preserves quantized format)
            auto slice_tensor = (device.is_cpu() ? loadCPUWeightRange(name,device,WeightDerivationKind::ColumnSlice,
                WeightSliceSpec{.col_start=col_start,.col_count=(col_end)-(col_start),.inner_is_presliced=true},{},rank) : loader_.loadTensorColumnSlice(
                name, col_start, col_end, device, WeightPrecision::NATIVE));

            if (!slice_tensor)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " failed to load column slice for: " + name);
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " input-parallel " << name
                                              << " [" << rows << ", " << total_cols
                                              << "] -> loaded ONLY cols [" << col_start << ", " << col_end
                                              << ") = " << col_count << " cols"
                                              << (tp_config ? " (proportional)" : " (equal)"));

            // Wrap in TensorSlice with row-parallel metadata
            // (input-parallel is mathematically like row-parallel but with column slicing)
            auto meta = SliceMetadata::forRowParallel(
                rows, total_cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
            WeightSliceSpec input_slice;
            input_slice.source_rows = rows;
            input_slice.source_cols = total_cols;
            input_slice.row_start = 0;
            input_slice.row_count = rows;
            input_slice.col_start = col_start;
            input_slice.col_count = col_count;
            input_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, slice, WeightDerivationKind::ColumnSlice, input_slice, device);
            return slice;
        }
        else if (mode == ShardingMode::EXPERT_ID_APPORTIONED)
        {
            // Expert-ID apportionment splits the expert axis of a 3D MoE tensor.
            //
            // Expert weights are stored as 3D tensors [ne0, ne1, num_experts] where:
            //   ne0 = columns (fastest varying)
            //   ne1 = rows per expert
            //   ne2 = num_experts (slowest varying)
            //
            // Each rank loads only its assigned expert subset:
            //   expert_start = rank * experts_per_rank
            //   expert_count = experts_per_rank (or remainder for last rank)
            //
            // The resulting tensor has shape [ne0, ne1, local_count] and contains
            // only the local expert data. extractExpertViews() must account for
            // the reduced expert count when creating 2D views.

            const int rank = routed_expert_assignment_.participant_index;
            const int world_size = routed_expert_assignment_.participant_count;
            if (!routed_expert_assignment_.valid() ||
                world_size != mpi_ctx_->world_size() ||
                rank != mpi_ctx_->rank())
            {
                throw std::runtime_error(
                    "[WeightManager] Routed-expert physical assignment does not match the active MPI participant coordinates for " +
                    name);
            }

            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for expert-id-apportioned loading: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 3)
            {
                throw std::runtime_error("[WeightManager] Expert-ID apportionment requires a 3D tensor, got " +
                                         std::to_string(dims_opt->size()) + "D for: " + name);
            }
            const auto &dims = *dims_opt;
            size_t ne0 = dims[0]; // cols
            size_t ne1 = dims[1]; // rows per expert
            size_t ne2 = dims[2]; // num_experts

            const int layer_idx = inferWeightLayer(name);
            if (layer_idx < 0)
                throw std::runtime_error(
                    "[WeightManager] Cannot derive a layer index for routed-expert weight " + name);
            const std::vector<int> selected_experts =
                routed_expert_ownership::expertIdsForParticipant(
                    static_cast<int>(ne2),
                    world_size,
                    rank,
                    layer_idx,
                    routed_expert_assignment_.owner_order);
            recordRoutedExpertWeightSelection(
                selected_experts,
                routed_expert_assignment_.owner_order,
                rank,
                layer_idx,
                device);
            std::vector<size_t> selected_expert_indices;
            selected_expert_indices.reserve(selected_experts.size());
            for (const int expert_id : selected_experts)
                selected_expert_indices.push_back(static_cast<size_t>(expert_id));

            auto slice_tensor = loader_.loadTensorExpertSelection(
                name,
                selected_expert_indices,
                device,
                WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " failed to load expert slice for: " + name);
            }

            LOG_DEBUG("[WeightManager] Rank " << rank << " expert-id-apportioned " << name
                                              << " [" << ne0 << ", " << ne1 << ", " << ne2
                                              << "] -> loaded " << selected_experts.size()
                                              << "/" << ne2 << " explicit experts"
                                              << " owner_order="
                                              << routedExpertOwnerOrderToString(
                                                     routed_expert_assignment_.owner_order));

            // Return the sliced 3D tensor directly (no TensorSlice wrapping needed —
            // expert-ID apportionment combines routed outputs, not weights)
            WeightSliceSpec expert_slice;
            expert_slice.source_rows = ne0;
            expert_slice.source_cols = ne1;
            expert_slice.row_start = 0;
            expert_slice.row_count = ne0;
            expert_slice.col_start = 0;
            expert_slice.col_count = ne1;
            expert_slice.expert_start = selected_experts.empty()
                                              ? 0u
                                              : static_cast<size_t>(selected_experts.front());
            expert_slice.expert_count = selected_experts.size();
            expert_slice.expert_ids = selected_experts;
            expert_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, slice_tensor, WeightDerivationKind::ExpertSlice, expert_slice, device);
            return slice_tensor;
        }

        throw std::runtime_error("[WeightManager] Unknown sharding mode for: " + name);
    }

    std::shared_ptr<TensorBase> WeightManager::getInterleavedWeight(const std::string &name, DeviceId device)
    {
        // Phase 3: Interleaved strategy (not yet implemented)
        // TODO: Implement NUMA-aware allocation with page interleaving
        //
        // For shared-memory multi-socket systems:
        // 1. Allocate with NUMA interleave policy (mbind)
        // 2. Memory distributed across NUMA nodes
        // 3. All ranks can access, but with varying latency
        // 4. Good for systems with fast interconnect (NVLink, etc.)

        LOG_ERROR("[WeightManager] INTERLEAVED strategy not yet implemented, falling back to REPLICATED");
        return getReplicatedWeight(name, device);
    }

    void WeightManager::forEachWeight(std::function<void(const std::string &, TensorBase *)> visitor) const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (const auto &[name, tensor] : cache_)
        {
            if (tensor)
                visitor(name, tensor.get());
        }
        for(const auto &[key,tensor]:per_device_cache_)
            if(key.device.is_cpu() && tensor)visitor(key.name,tensor.get());
    }

    void WeightManager::forEachPreparedWeight(std::function<void(const std::string &, TensorBase *)> visitor) const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        // Primary cache (original tensors, used in single-device mode)
        for (const auto &[name, tensor] : cache_)
        {
            if (tensor)
                visitor(name, tensor.get());
        }
        // Per-device cache (TP-sliced tensors, used in LOCAL TP mode)
        // Canonical names are typed fields, independent of device/rank spelling.
        for (const auto &[cache_key, tensor] : per_device_cache_)
        {
            if (!tensor)
                continue;
            visitor(cache_key.name, tensor.get());
        }
    }

    std::shared_ptr<PreparedWeightStore> WeightManager::preparedWeightStore()
    {
        std::shared_ptr<PreparedWeightStore> store;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            if (!prepared_weight_store_)
                prepared_weight_store_ = std::make_shared<PreparedWeightStore>();
            store = prepared_weight_store_;
        }
        if (const auto authority = physicalMemoryAuthority())
            store->installPhysicalMemoryAuthority(authority);
        return store;
    }

    std::shared_ptr<PreparedWeightStore> WeightManager::preparedWeightStoreIfInitialized() const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return prepared_weight_store_;
    }

    size_t WeightManager::preparedRecordCountForDevice(DeviceId device) const
    {
        const auto store = preparedWeightStoreIfInitialized();
        const size_t store_records =
            store ? store->sizeForDevice(device) : 0u;
        return store_records +
               expert_gemm_registry_
                   .countOwnedEnginesForDeviceAcrossScopes(device);
    }

    ReleasedWeightLoadStaging WeightManager::releasedWeightLoadStaging(DeviceId device) const
    {
        std::lock_guard<std::mutex> lock(physical_memory_authority_mutex_);
        const auto found = released_weight_load_staging_.find(device);
        if (found == released_weight_load_staging_.end())
            throw std::logic_error("Resident native table has no drained ordinary weight-loader slot");
        return found->second;
    }

    size_t WeightManager::retainedPreparedDeviceBytes(
        DeviceId device) const
    {
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "Retained prepared-device byte accounting requires an exact GPU");
        }

        size_t bytes = prepared_device_allocations_.liveBytes(device);

        const auto store = preparedWeightStoreIfInitialized();
        const size_t embedding_bytes = store
            ? store->preparedEmbeddingAllocationBytesForDevice(device)
            : 0u;
        if (embedding_bytes > std::numeric_limits<size_t>::max() - bytes)
        {
            throw std::overflow_error(
                "Prepared device model-allocation BOM overflows size_t");
        }
        return bytes + embedding_bytes;
    }

    void WeightManager::setPreparedWeightStore(std::shared_ptr<PreparedWeightStore> store)
    {
        if (store)
        {
            if (const auto authority = physicalMemoryAuthority())
                store->installPhysicalMemoryAuthority(authority);
        }
        std::lock_guard<std::mutex> lock(cache_mutex_);
        prepared_weight_store_ = std::move(store);
    }

    void WeightManager::installPhysicalMemoryAuthority(
        std::shared_ptr<PhysicalMemoryAuthority> authority)
    {
        if (!authority)
        {
            throw std::invalid_argument(
                "WeightManager requires a non-null physical-memory authority");
        }
        {
            std::lock_guard<std::mutex> lock(
                physical_memory_authority_mutex_);
            if (physical_memory_authority_ &&
                physical_memory_authority_.get() != authority.get())
            {
                throw std::logic_error(
                    "WeightManager cannot replace a live physical-memory authority");
            }
            physical_memory_authority_ = std::move(authority);
        }

        /*
         * The prepared store is an identity-preserving carrier used by graph
         * builders. Publish after releasing the authority mutex so concurrent
         * store creation cannot invert WeightManager's independent lock order.
         */
        const auto published = physicalMemoryAuthority();
        const auto store = preparedWeightStoreIfInitialized();
        if (store)
            store->installPhysicalMemoryAuthority(published);
    }

    std::shared_ptr<PhysicalMemoryAuthority>
    WeightManager::physicalMemoryAuthority() const
    {
        std::lock_guard<std::mutex> lock(
            physical_memory_authority_mutex_);
        return physical_memory_authority_;
    }

    WeightBinding WeightManager::prepareCPUWeightSource(WeightBinding original,
        const std::string &name, CPUPreparedWeightSource source,
        PhysicalMemoryOwner memory_owner, CPUStoragePlacement placement,
        std::optional<size_t> budget)
    {
        const auto memory = physicalMemoryAuthority();
        if (!memory || !original.identity.model_id.value || original.tensor || original.tensor_owner || original.prepared ||
            !original.residency.home_device.is_valid() || !original.residency.home_device.is_cpu() ||
            source.original != loader_.nativeWeightSourceGeometry(name))
            throw std::invalid_argument("CPU weight population requires original unmaterialized plan/source/PMA identity");
        size_t outer = 1;
        for (size_t axis = 2; axis < source.original.original_shape.size(); ++axis) {
            const auto dimension = source.original.original_shape[axis];
            if (!dimension || outer > std::numeric_limits<size_t>::max()/dimension)
                throw std::invalid_argument("CPU source population has invalid original outer geometry");
            outer *= dimension;
        }
        if (!source.matrix_count || source.range.original_expert >= outer ||
            source.matrix_count > outer-source.range.original_expert)
            throw std::invalid_argument("CPU source population exceeds its original outer extent");
        if (original.joint_source) {
            const auto &member = original.joint_source->participant();
            if (member.world_rank != memory->worldRank() || original.residency.home_device != member.device ||
                placement != CPUStoragePlacement::onNode(member.address.numa_node))
                throw std::invalid_argument("CPU source population changed original retained member placement");
        } else if (placement != loader_.nativeWeightSourcePlacement()) {
            throw std::invalid_argument("CPU source population changed original backend placement");
        }
        const auto store = preparedWeightStore();
        if (!store->bindModelIdIfUnset(original.identity.model_id))
            throw std::invalid_argument("CPU source population belongs to another original model store");
        auto source_scope = memory->acquireCPUWeightSourceTransaction(original.residency.home_device);
        original.native_source = NativeWeightSourceProjection{source.original,source.range,source.matrix_count,source.assembly};
        const auto kind = source.consumer == CPUWeightConsumer::PreparedGemm ?
            PreparedWeightKind::CpuPackedGemm : PreparedWeightKind::CPUResidentTensor;
        const PreparedWeightRef expected_ref{original.identity.model_id,original.binding_id,kind,original.residency.home_device};
        if (const auto retained = store->binding(expected_ref)) {
            TensorType expected_type = source.original.original_type;
            switch (source.destination_precision) {
            case WeightPrecision::NATIVE: break;
            case WeightPrecision::CONVERT_TO_FP32: expected_type=TensorType::FP32; break;
            case WeightPrecision::CONVERT_TO_FP16: expected_type=TensorType::FP16; break;
            case WeightPrecision::CONVERT_TO_BF16: expected_type=TensorType::BF16; break;
            case WeightPrecision::CONVERT_TO_INT8: expected_type=TensorType::Q8_0; break;
            }
            if (!retained->identity.sameSourceDefinition(original.identity) || retained->slice != original.slice ||
                retained->native_source != original.native_source || retained->projection_ownership != original.projection_ownership ||
                bool(retained->joint_source) != bool(original.joint_source) ||
                (original.joint_source && (*retained->joint_source != *original.joint_source ||
                    retained->joint_source->cohort() != original.joint_source->cohort())) ||
                !retained->tensor || retained->tensor_owner.get() != retained->tensor ||
                retained->tensor->native_type() != expected_type || retained->tensor->home_device() != original.residency.home_device ||
                retained->floating_operand_precision != source.operand_precision || !retained->prepared ||
                retained->prepared->kind != kind || retained->prepared->device != original.residency.home_device ||
                store->physicalMemoryAuthority() != memory)
                throw std::invalid_argument("CPU source reuse changed the original completed Store definition");
            if (kind == PreparedWeightKind::CpuPackedGemm) {
                const auto engine=store->gemmKernelLifetime(expected_ref);
                if (!engine || !engine->weights_converted())
                    throw std::logic_error("CPU source reuse has no actual complete original engine");
            }
            return *retained;
        }
        if (!original.joint_source && isRoutedExpertRole(original.identity.role)) {
            const auto role = original.identity.role==WeightRole::MoEExpertGate ? ExpertGemmRegistry::WeightRole::GATE :
                original.identity.role==WeightRole::MoEExpertUp ? ExpertGemmRegistry::WeightRole::UP : ExpertGemmRegistry::WeightRole::DOWN;
            auto *existing = original.identity.overlay_domain.empty() ?
                expert_gemm_registry_.getEngine(original.residency.home_device,original.identity.layer,original.identity.expert,role) :
                expert_gemm_registry_.getEngineForParticipant(original.identity.overlay_domain,original.residency.home_device,
                    original.identity.overlay_participant_world_rank,original.identity.overlay_participant_index,
                    original.identity.layer,original.identity.expert,role);
            if (existing)
                throw std::logic_error("CPU prepared registry has no matching original committed source record");
        }
        // The same rank owner serializes distinct managers, completed-record
        // reuse and fresh final construction; no scratch can escape its cap.
        auto preparation = store->beginCPUPreparation(original, source, memory_owner, placement);
        if (const auto request = preparation->stagingRequest()) {
            NativeWeightSourceTransaction transaction(loader_, name, *request,
                placement, memory, original.residency.home_device, budget, std::move(source_scope));
            for (size_t matrix = 0; matrix < source.matrix_count; ++matrix) {
                auto range = source.range;
                range.original_expert += matrix;
                range.selected_expert_ordinal += matrix;
                forEachNativeWeightSourceStripe(source.original,range,source.assembly,
                    [&](const auto &stripe,size_t,size_t) {
                        transaction.visit(stripe,[&](const auto &window,auto native,auto scratch) {
                            preparation->append(window,native,scratch);
                        });
                    });
            }
            // Commit the actual final roots while the source owner still holds
            // the arena. It retires before returning the published binding.
            return store->finishCPUPreparation(std::move(preparation));
        }
        // Unchanged resident bytes are final storage, not transient input. A
        // large contiguous final destination may be filled directly without a
        // second raw matrix or a source mmap becoming process-resident.
        for (size_t matrix = 0; matrix < source.matrix_count; ++matrix) {
            auto range = source.range;
            range.original_expert += matrix;
            range.selected_expert_ordinal += matrix;
            forEachNativeWeightSourceStripe(source.original,range,source.assembly,
                [&](const auto &stripe,size_t,size_t) {
                    auto capacity=budget.value_or(std::numeric_limits<size_t>::max());
                    // A column stripe is separated by final row gaps: direct final
                    // reads therefore borrow one row (or one bounded group strip).
                    if(source.assembly && source.assembly->axis==NativeWeightSourceAssembly::Axis::Columns)
                        capacity=std::min(capacity,stripe.native_row_bytes);
                    forEachNativeWeightSourceWindow(source.original,stripe,capacity,[&](const auto &window) {
                        preparation->readDirect(window,[&](std::span<std::byte> destination) {
                            loader_.readNativeWeightSourceWindow(name,window,destination);
                        });
                    });
                });
        }
        return store->finishCPUPreparation(std::move(preparation));
    }

    void WeightManager::materializeCPURequirement(const WeightRequirement &requirement,
        const InferenceStrategy &strategy, PhysicalMemoryOwner memory_owner,
        std::optional<size_t> budget, ModelWeightSetBuilder &builder,
        const ExpertGemmRegistry::StaticProjectionLayer *existing,
        const WeightRequirement *original_tied_source)
    {
        const auto role = requirement.role == WeightRole::Other ? inferWeightRole(requirement.canonical_name) : requirement.role;
        memory_owner = resolveCPUWeightRequirementOwner(requirement, memory_owner);
        const auto selected = requirement.joint_source;
        const auto parent_id = stableMaterializedBindingId(requirement,strategy);
        if (selected) {
            const auto memory = physicalMemoryAuthority();
            if (!memory || memory->worldRank() != selected->participant().world_rank ||
                !requirement.projection_ownership || *requirement.projection_ownership != selected->ownership() ||
                memory_owner != PhysicalMemoryOwner::RoutedExpertWeights || weight_precision_ != WeightPrecision::NATIVE)
                throw std::invalid_argument("Retained CPU projection changed its original source/placement/native purpose");
            if (existing) {
                if (!existing->joint_source || *existing->joint_source != *selected ||
                    existing->joint_source->cohort() != selected->cohort())
                    throw std::invalid_argument("Retained CPU projection replaced its original selection owner");
                size_t count = 0;
                for (const auto &source : existing->sources) if (source.identity.role == requirement.role) {
                    if (source.identity.model_id != strategy.model_id || source.identity.source_instance_id != parent_id ||
                        source.binding_id != staticProjectedExpertBindingId(parent_id,requirement.canonical_name,source.identity.expert))
                        throw std::invalid_argument("Retained CPU projection changed original parent/expert identity");
                    ExpertGemmRegistry::validateStaticProjectionSource(source);
                    builder.addBinding(source); ++count;
                }
                if (count != selected->expertIds().size())
                    throw std::logic_error("Retained CPU projection reuse omitted original matrices");
                return;
            }
        }
        const auto name = requirement.source_name.empty() ? requirement.canonical_name : requirement.source_name;
        if (!loader_.hasTensor(name)) {
            if (!requirement.required) return;
            throw std::invalid_argument("Required CPU native source is absent: " + name);
        }
        const auto geometry = loader_.nativeWeightSourceGeometry(name);
        const auto projections = projectNativeWeightSource(requirement, geometry);
        const bool expert = geometry.original_shape.size() == 3 && isRoutedExpertRole(role);
        const auto representation = PreparedWeightRepresentationContract::resolve(role,geometry.original_type,
            requirement.canonical_name,loader_.architecture(),debugEnv().gemm.gdn_projection_arithmetic,
            debugEnv().gemm.moe_router_arithmetic);
        const auto operand = PreparedWeightRepresentationContract::operandPrecision(role,geometry.original_type,requirement.canonical_name,
            loader_.architecture(),debugEnv().gemm.gdn_projection_arithmetic,debugEnv().gemm.moe_router_arithmetic);
        const auto precision = PreparedWeightRepresentationContract::resolveCPUWeightPrecision(
            role,geometry.original_type,weight_precision_,representation,operand);
        const auto classification = PreparedWeightIdentityContract::classifyOriginalMatrix(role,requirement.target_device);
        bool gemm = expert || (geometry.original_shape.size() == 2 && classification == PreparedGemmClassification::Gemm);
        if (!expert && classification == PreparedGemmClassification::RequiresSchema) {
            if (requirement.expected_prepared_kind == PreparedWeightKind::CPUResidentTensor) gemm = false;
            else if (requirement.expected_prepared_kind == PreparedWeightKind::CpuPackedGemm) gemm = true;
            else if (has_sharding_config_) gemm = geometry.original_shape.size() == 2 && isGemmWeight(requirement.canonical_name);
            else throw std::invalid_argument("CPU weight source has no declared schema/consumer: " + requirement.canonical_name);
        }
        const auto placement = selected ? CPUStoragePlacement::onNode(selected->participant().address.numa_node) :
            loader_.nativeWeightSourcePlacement();
        if (selected && projections.size() != selected->expertIds().size())
            throw std::invalid_argument("Retained CPU projection changed original selected extent");
        std::optional<WeightBinding> tied_owner;
        if (original_tied_source) {
            // A name is not a storage capability. The enclosing plan supplies
            // its actual source declaration, including original PP/TP scope.
            if (requirement.derivation != WeightDerivationKind::TiedAlias || requirement.source_name.empty() ||
                requirement.source_name == requirement.canonical_name || selected || expert ||
                original_tied_source == &requirement || original_tied_source->canonical_name != name ||
                original_tied_source->target_device != requirement.target_device ||
                original_tied_source->derivation == WeightDerivationKind::TiedAlias ||
                original_tied_source->joint_source || original_tied_source->projection_ownership ||
                (!original_tied_source->source_name.empty() && original_tied_source->source_name != name))
                throw std::invalid_argument("CPU tied alias changed its original plan source/selection");
            const auto source_role=original_tied_source->role==WeightRole::Other
                ?inferWeightRole(name):original_tied_source->role;
            const auto source_execution=PreparedWeightRepresentationContract::resolve(source_role,geometry.original_type,
                name,loader_.architecture(),debugEnv().gemm.gdn_projection_arithmetic,debugEnv().gemm.moe_router_arithmetic);
            const auto source_operand=PreparedWeightRepresentationContract::operandPrecision(source_role,geometry.original_type,name,
                loader_.architecture(),debugEnv().gemm.gdn_projection_arithmetic,debugEnv().gemm.moe_router_arithmetic);
            const auto source_precision=PreparedWeightRepresentationContract::resolveCPUWeightPrecision(
                source_role,geometry.original_type,weight_precision_,source_execution,source_operand);
            const auto source_projections=projectNativeWeightSource(*original_tied_source,geometry);
            // Different authored slices/scalars are distinct final representations,
            // not aliases of a vaguely compatible full matrix. Their ordinary
            // window consumer remains responsible for its own exact final root.
            if (gemm && projections.size()==1 && projections.front().matrix_count==1 &&
                source_projections==projections && source_precision==precision && source_operand==operand) {
                ModelWeightSetBuilder source_publication(strategy);
                materializeCPURequirement(*original_tied_source,strategy,memory_owner,budget,source_publication);
                auto published=source_publication.freezeBindings();
                if (published.size()!=1)
                    throw std::logic_error("CPU tied source did not publish its single original final matrix");
                const auto& source=published.front();
                const auto memory=physicalMemoryAuthority();
                const auto store=preparedWeightStore();
                // Materialization replaces the source's temporary load policy
                // when its bytes are the final CPU execution representation.
                // Authenticate that committed lifetime, not the earlier intent.
                const auto source_policy=source.residency.raw_host_data_available
                    ?WeightHostPolicy::RequiredForCPUExecution:original_tied_source->host_policy;
                if (!memory || store->physicalMemoryAuthority()!=memory || store->modelId()!=strategy.model_id ||
                    source.identity.model_id!=strategy.model_id || source.identity.canonical_name!=name ||
                    source.identity.derivation!=original_tied_source->derivation || !source.immutable ||
                    !source.tensor || source.tensor_owner.get()!=source.tensor || !source.prepared ||
                    source.residency.home_device!=requirement.target_device ||
                    source.residency.resident_device!=requirement.target_device ||
                    source.tensor->home_device()!=requirement.target_device ||
                    source.residency.host_policy!=source_policy ||
                    source.floating_operand_precision!=operand || !source.native_source ||
                    *source.native_source!=projections.front())
                    throw std::invalid_argument("CPU tied source has no exact original final Store owner");
                tied_owner=source;
            }
        }
        for (const auto &projection : projections) {
            if (selected) {
                const auto expected = selected->ownership().projection(role);
                if (projection.range.selected_expert_ordinal >= selected->expertIds().size() ||
                    projection.range.original_expert != size_t(selected->expertIds()[projection.range.selected_expert_ordinal]) ||
                    projection.range.first_row != size_t(expected.first_row) || projection.range.rows != size_t(expected.rows) ||
                    projection.range.first_column != size_t(expected.first_column) || projection.range.columns != size_t(expected.columns))
                    throw std::invalid_argument("Retained CPU projection changed its exact original source rectangle");
            }
            WeightBinding original;
            original.binding_id = expert ? staticProjectedExpertBindingId(parent_id,requirement.canonical_name,
                int(projection.range.original_expert)) : parent_id;
            original.identity = makeSourceWeightIdentity(requirement.canonical_name,strategy.model_id,original.binding_id);
            original.identity.instance_id = original.binding_id;
            if (expert) {
                original.identity.canonical_name = requirement.canonical_name + "@expert" + std::to_string(projection.range.original_expert);
                original.identity.logical_id = stableWeightLogicalId(original.identity.canonical_name);
            }
            original.identity.source_instance_id = expert ? std::optional<uint64_t>(parent_id) : std::nullopt;
            original.identity.derivation = expert ? WeightDerivationKind::ExpertSlice : requirement.derivation;
            original.identity.role = role;
            original.identity.layer = requirement.layer >= 0 ? requirement.layer : inferWeightLayer(requirement.canonical_name);
            original.identity.expert = expert ? int(projection.range.original_expert) : requirement.expert;
            original.identity.pp_stage = requirement.pp_stage;
            original.identity.tp_domain = requirement.tp_domain;
            original.identity.tp_rank_or_device_index = requirement.tp_rank_or_device_index;
            original.identity.residency_category = requirement.residency_category;
            original.identity.overlay_domain = requirement.overlay_domain;
            original.identity.overlay_participant_index = requirement.overlay_participant_index;
            original.identity.overlay_participant_world_rank = requirement.overlay_participant_world_rank;
            original.slice = requirement.slice;
            if (geometry.original_shape.size() == 1) {
                original.slice.source_rows = geometry.original_shape[0]; original.slice.source_cols = 1;
                original.slice.row_start = projection.range.first_column; original.slice.row_count = projection.finalColumns();
                original.slice.col_start = 0; original.slice.col_count = 1;
            } else {
                original.slice.source_rows = geometry.original_shape[1]; original.slice.source_cols = geometry.original_shape[0];
                original.slice.row_start = projection.range.first_row; original.slice.row_count = projection.finalRows();
                original.slice.col_start = projection.range.first_column; original.slice.col_count = projection.finalColumns();
            }
            if (expert) {
                original.slice.expert_start = projection.range.original_expert;
                original.slice.expert_count = 1;
                original.slice.expert_ids = {int(projection.range.original_expert)};
            }
            original.slice.inner_is_presliced = true;
            original.projection_ownership = requirement.projection_ownership;
            original.joint_source = selected;
            original.residency.home_device = requirement.target_device;
            original.residency.resident_device = requirement.target_device;
            original.residency.host_policy = requirement.host_policy;
            original.residency.raw_host_data_available = false;
            original.floating_operand_precision = operand;
            original.immutable = true;
            CPUPreparedWeightSource consumer{.original=geometry,.range=projection.range,
                .matrix_count=projection.matrix_count,.destination_precision=precision,
                .consumer=gemm?CPUWeightConsumer::PreparedGemm:CPUWeightConsumer::ResidentTensor,
                .operand_precision=operand,.numerical_policy=expert?CPUProjectionNumericalPolicy::GPUAlignedExpert:
                    CPUProjectionNumericalPolicy::BackendNative,
                .half_conversion=precision==WeightPrecision::CONVERT_TO_FP16 && geometry.original_type!=TensorType::FP16 &&
                    (representation==ModelPreparedWeightRepresentation::FP16 || operand==FloatingActivationOperandPrecision::FP16RNE) ?
                    CPUPreparedHalfConversion::ModelPreparedHalf : CPUPreparedHalfConversion::GenericTensor,
                .assembly=projection.assembly};
            auto committed = [&]() -> WeightBinding {
                if (!tied_owner)
                    return prepareCPUWeightSource(std::move(original),name,std::move(consumer),memory_owner,placement,budget);
                const auto store=preparedWeightStore();
                // Serialize this manager's alias check/publication without a
                // new cache or holding a source token across the packer's own
                // bounded source transaction. Store methods do not reenter the
                // manager cache; source construction completed before this lock.
                std::lock_guard alias_publication(cache_mutex_);
                original.identity.source_instance_id=tied_owner->binding_id;
                original.native_source=projection;
                original.tensor_owner=tied_owner->tensor_owner;
                original.tensor=original.tensor_owner.get();
                original.residency.raw_host_data_available=!original.tensor->is_raw_data_released() &&
                    original.tensor->raw_data()!=nullptr;
                // Alias publication shares the source's actual final CPU root.
                // Its caller-authored load policy cannot shorten that owner's
                // execution lifetime or discard a live floating/embedding input.
                if(original.residency.raw_host_data_available) {
                    if(tied_owner->residency.host_policy!=WeightHostPolicy::RequiredForCPUExecution)
                        throw std::logic_error("CPU tied alias source lost its final execution-storage policy");
                    original.residency.host_policy=tied_owner->residency.host_policy;
                }
                const PreparedWeightRef ref{strategy.model_id,original.binding_id,PreparedWeightKind::CpuPackedGemm,
                    requirement.target_device};
                if (const auto retained=store->binding(ref)) {
                    const auto engine=store->gemmKernelLifetime(ref);
                    if (!retained->identity.sameSourceDefinition(original.identity) || retained->slice!=original.slice ||
                        retained->native_source!=original.native_source || retained->tensor_owner!=original.tensor_owner ||
                        retained->tensor!=original.tensor || retained->residency.home_device!=requirement.target_device ||
                        retained->residency.resident_device!=requirement.target_device ||
                        retained->floating_operand_precision!=operand || !retained->immutable ||
                        retained->residency.host_policy!=original.residency.host_policy ||
                        retained->residency.raw_host_data_available!=original.residency.raw_host_data_available ||
                        retained->projection_ownership!=original.projection_ownership || retained->joint_source ||
                        !engine || !engine->weights_converted() ||
                        engine->get_n()!=int(projection.finalRows()) || engine->get_k()!=int(projection.finalColumns()))
                        throw std::invalid_argument("CPU tied alias replaced its original committed final owner");
                    return *retained;
                }
                const auto publish_engine=[&](std::shared_ptr<ITensorGemm> engine) {
                    if (!engine || !engine->weights_converted() || engine->get_n()!=int(projection.finalRows()) ||
                        engine->get_k()!=int(projection.finalColumns()) ||
                        engine->floatingActivationOperandPrecision()!=operand)
                        throw std::logic_error("CPU tied source lost its exact actual prepared engine");
                    using Factory=llaminar::v2::kernels::KernelFactory;
                    auto handle=std::make_shared<Factory::PreparedGemmHandle>();
                    handle->tensor=original.tensor;handle->device_id=requirement.target_device;
                    handle->kind=engine->canReleaseSourceWeightTensor()?Factory::GemmPreparationKind::CPU_PACKED:
                        Factory::GemmPreparationKind::FLOATING_POINT;
                    handle->variant=static_cast<int>(original.tensor->native_type());
                    handle->prepared_weights=std::make_shared<Factory::PreparedGemmWeights>();
                    handle->prepared_weights->kind=handle->kind;handle->prepared_weights->owned_kernel=std::move(engine);
                    handle->prepared_weights->kernel=handle->prepared_weights->owned_kernel.get();
                    return store->registerPreparedGemmHandle(original,PreparedWeightKind::CpuPackedGemm,
                        requirement.target_device,std::move(handle),memory_owner);
                };
                if (tied_owner->prepared->kind==PreparedWeightKind::CpuPackedGemm) {
                    original.prepared=publish_engine(store->gemmKernelLifetime(*tied_owner->prepared));
                } else if (original.tensor->native_type()==TensorType::FP32 ||
                    original.tensor->native_type()==TensorType::FP16 || original.tensor->native_type()==TensorType::BF16) {
                    // A floating LM head retains the source's actual final root;
                    // a raw-pointer factory would not own that lifetime itself.
                    original.prepared=publish_engine(gemm::FloatingPointGemmKernel::fromFinalCPUStorage(
                        original.tensor_owner,CPUProjectionNumericalPolicy::BackendNative,operand));
                } else {
                    // Native embedding residency and VNNI execution need distinct
                    // final layouts. Keep the SAME real resident tensor while
                    // the canonical bounded packer owns the necessary engine.
                    if (original.tensor->is_raw_data_released() || !original.tensor->raw_data())
                        throw std::logic_error("CPU tied native resident has no live original final bytes");
                    original.prepared=store->prepareGemm(original,memory_owner);
                }
                return original;
            }();
            if (requirement.expected_prepared_kind == PreparedWeightKind::PreparedEmbedding) {
                if (role != WeightRole::Embedding || projection.matrix_count != 1 || projection.range.columns > size_t(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("CPU prepared embedding declaration has foreign source geometry/role");
                committed.prepared = preparedWeightStore()->prepareEmbedding(committed,int(projection.finalColumns()),
                    projection.range.first_row,geometry.original_shape.size()>1?geometry.original_shape[1]:1,memory_owner);
            }
            if (expert && selected) {
                expert_gemm_registry_.registerCommittedCPUProjection(committed,*preparedWeightStore());
            } else if (expert) {
                if (!committed.prepared) throw std::logic_error("CPU expert source committed no actual prepared engine");
                auto owner = preparedWeightStore()->gemmKernelLifetime(*committed.prepared);
                if (!owner) throw std::logic_error("CPU expert source lost original prepared ownership");
                auto *engine = owner.get();
                const auto engine_role = role == WeightRole::MoEExpertGate ? ExpertGemmRegistry::WeightRole::GATE :
                    role == WeightRole::MoEExpertUp ? ExpertGemmRegistry::WeightRole::UP : ExpertGemmRegistry::WeightRole::DOWN;
                auto *retained = requirement.overlay_domain.empty() ?
                    expert_gemm_registry_.getEngine(requirement.target_device,committed.identity.layer,committed.identity.expert,engine_role) :
                    expert_gemm_registry_.getEngineForParticipant(requirement.overlay_domain,requirement.target_device,
                        requirement.overlay_participant_world_rank,requirement.overlay_participant_index,
                        committed.identity.layer,committed.identity.expert,engine_role);
                if (retained && retained != engine)
                    throw std::logic_error("CPU source reuse replaced an original physical prepared slot");
                if (requirement.overlay_domain.empty()) {
                    expert_gemm_registry_.registerEngine(requirement.target_device,committed.identity.layer,
                        committed.identity.expert,engine_role,engine,std::move(owner));
                } else {
                    expert_gemm_registry_.registerEngineForParticipant(requirement.overlay_domain,requirement.target_device,
                        requirement.overlay_participant_world_rank,requirement.overlay_participant_index,
                        committed.identity.layer,committed.identity.expert,engine_role,engine,owner);
                    expert_gemm_registry_.registerEngineForDomain(requirement.overlay_domain,requirement.target_device,
                        committed.identity.layer,committed.identity.expert,engine_role,engine,std::move(owner));
                }
            }
            builder.addBinding(std::move(committed));
        }
    }

    size_t WeightManager::originalWeightSourceBytes(const std::string &name) const
    {
        if (!loader_.hasTensor(name)) return 0;
        const auto source = loader_.nativeWeightSourceGeometry(name);
        if (source.original_shape.empty())
            throw std::invalid_argument("Original weight source has no dimensions: " + name);
        const auto outer = std::span<const size_t>(source.original_shape).subspan(1);
        const size_t rows = outer.empty() ? 1 : nativeTensorExtent(outer, 1, 1);
        return nativeMatrixExtent(rows, source.original_shape.front(),
            source.native_group_elements, source.native_group_bytes);
    }

    std::optional<WeightRequirement> WeightManager::projectCPUDenseTPRequirement(
        const WeightRequirement &requirement,
        const std::shared_ptr<const TensorParallelConfig> &tp_config) const
    {
        // Explicit vocabulary and expert selections, assembled source windows,
        // and replicated views already carry their complete source authority.
        if (!tp_config || tp_config->worldSize() <= 1 || requirement.tp_domain < 0 ||
            requirement.bypass_tensor_parallel || requirement.derivation != WeightDerivationKind::Source ||
            requirement.slice != WeightSliceSpec{} || requirement.native_source_assembly ||
            requirement.projection_ownership || isRoutedExpertRole(requirement.role))
            return std::nullopt;

        const auto &assignment = tp_config->forRank(requirement.tp_rank_or_device_index);
        const auto &name = requirement.source_name.empty() ? requirement.canonical_name : requirement.source_name;
        const auto mode = getShardingMode(name);
        if (mode == ShardingMode::REPLICATE || !loader_.hasTensor(name))
            return std::nullopt;
        const auto source = loader_.nativeWeightSourceGeometry(name);
        const auto &shape = source.original_shape;
        const bool input_parallel = mode == ShardingMode::INPUT_PARALLEL;
        if ((mode != ShardingMode::COLUMN_PARALLEL && mode != ShardingMode::ROW_PARALLEL && !input_parallel) ||
            (shape.size() != 2 && !(mode == ShardingMode::COLUMN_PARALLEL && shape.size() == 1)))
            throw std::invalid_argument("CPU dense TP source has incompatible schema or original axes: " + name);
        // GGUF describes matrices as K/N, while source rows/columns and schema
        // sharding describe the logical N/K matrix. A vector uses logical rows.
        const bool vector = shape.size() == 1;
        const size_t total = input_parallel || vector ? shape[0] : shape[1];
        const bool fused = mode == ShardingMode::COLUMN_PARALLEL && !vector &&
            sharding_config_.getDimensionType(name) == WeightDimensionType::FusedQKVHeads;
        const int gdn_elements = (mode == ShardingMode::COLUMN_PARALLEL || input_parallel)
            ? gdnValueElementsPerHead(name, total) : 0;
        auto projected = requirement;
        if (fused || gdn_elements > 0) {
            const auto spans = fused
                ? fusedQKVSourceSpans(name, total, assignment, tp_config)
                : gdnHeadAssignmentFor(assignment, *tp_config).valueElementSpans(gdn_elements);
            using Axis = NativeWeightSourceAssembly::Axis;
            projected.native_source_assembly = orderedCPUNativeAssembly(
                source, input_parallel || vector ? Axis::Columns : Axis::Rows, spans);
            projected.derivation = WeightDerivationKind::FusedSubblockConcat;
        } else {
            size_t start = 0, count = 0;
            computeSliceBoundaries(name, total, assignment, *tp_config, start, count);
            if (!count || start > total || count > total - start)
                throw std::invalid_argument("CPU dense TP source exceeds its original slice axis: " + name);
            if (input_parallel) {
                projected.slice = {.col_start = start, .col_count = count, .inner_is_presliced = true};
                projected.derivation = WeightDerivationKind::ColumnSlice;
            } else {
                projected.slice = {.row_start = start, .row_count = count, .inner_is_presliced = true};
                projected.derivation = WeightDerivationKind::RowSlice;
            }
        }
        return projected;
    }

    FrozenModelWeightSet WeightManager::materialize(const WeightPlan &plan)
    {
        const auto tp_config = tensorParallelConfig();
        ModelWeightSetBuilder builder(plan.strategy());
        const auto source_budget = configuredNativeWeightLoadStagingBudgetBytes();
        using JointSnapshot=std::pair<std::shared_ptr<const ExpertShardSourceSelection>,
            std::optional<ExpertGemmRegistry::StaticProjectionLayer>>;
        std::vector<JointSnapshot> joint_snapshots;
        // Capture complete prior publications before loading any new role. A
        // partially failed layer is fatal, never an invitation to retry/merge.
        for(const auto &requirement:plan.requirements())if(requirement.target_device.is_cpu() && requirement.joint_source) {
            const auto selected=requirement.joint_source;
            if(std::none_of(joint_snapshots.begin(),joint_snapshots.end(),
                [&](const auto &entry){return *entry.first==*selected;}))
                joint_snapshots.emplace_back(selected,expert_gemm_registry_.acquireJointLayer(
                    plan.strategy().model_id,selected->participant(),selected->layer()));
        }

        size_t expected_cpu_experts = 0, retained_cpu_experts = 0;
        for (const auto &requirement : plan.requirements()) {
            const auto role = requirement.role==WeightRole::Other ? inferWeightRole(requirement.canonical_name) : requirement.role;
            if (!requirement.target_device.is_cpu() || requirement.joint_source || !isRoutedExpertRole(role)) continue;
            const auto &name = requirement.source_name.empty()?requirement.canonical_name:requirement.source_name;
            if (!loader_.hasTensor(name)) {
                if (!requirement.required) continue;
                throw std::invalid_argument("Required CPU native source is absent: " + name);
            }
            const auto geometry = loader_.nativeWeightSourceGeometry(name);
            const auto projections = projectNativeWeightSource(requirement,geometry);
            const auto registry_role = role==WeightRole::MoEExpertGate?ExpertGemmRegistry::WeightRole::GATE:
                role==WeightRole::MoEExpertUp?ExpertGemmRegistry::WeightRole::UP:ExpertGemmRegistry::WeightRole::DOWN;
            const int layer = requirement.layer>=0?requirement.layer:inferWeightLayer(requirement.canonical_name);
            for (const auto &source : projections) {
                const int expert = int(source.range.original_expert);
                ITensorGemm *engine = nullptr;
                if (requirement.overlay_domain.empty()) engine=expert_gemm_registry_.getEngine(requirement.target_device,layer,expert,registry_role);
                else {
                    engine=expert_gemm_registry_.getEngineForParticipant(requirement.overlay_domain,requirement.target_device,
                        requirement.overlay_participant_world_rank,requirement.overlay_participant_index,layer,expert,registry_role);
                    if (engine != expert_gemm_registry_.getEngineForDomain(requirement.overlay_domain,requirement.target_device,layer,expert,registry_role))
                        throw std::logic_error("CPU source participant/domain prepared aliases disagree");
                }
                ++expected_cpu_experts;
                retained_cpu_experts += engine != nullptr;
            }
        }
        if (retained_cpu_experts && retained_cpu_experts != expected_cpu_experts)
            throw std::logic_error("CPU original prepared source bank is partial; source reload/merge is forbidden");

        for (const auto &requirement : plan.requirements())
        {
            if (requirement.target_device.is_cpu() && requirement.role != WeightRole::ImmutableRowTable) {
                const ExpertGemmRegistry::StaticProjectionLayer *existing = nullptr;
                if (requirement.joint_source) {
                    const auto found = std::find_if(joint_snapshots.begin(),joint_snapshots.end(),
                        [&](const auto &entry) { return *entry.first == *requirement.joint_source &&
                            entry.first->cohort() == requirement.joint_source->cohort(); });
                    if (found == joint_snapshots.end()) throw std::logic_error("Original CPU source snapshot is absent");
                    if (found->second) existing = &*found->second;
                }
                const WeightRequirement* tied_source=nullptr;
                if (requirement.derivation==WeightDerivationKind::TiedAlias) {
                    if (requirement.source_name.empty() || requirement.source_name==requirement.canonical_name)
                        throw std::invalid_argument("CPU tied alias requires a distinct original named source");
                    for (const auto& candidate:plan.requirements())
                        if (candidate.canonical_name==requirement.source_name &&
                            candidate.target_device==requirement.target_device) {
                            if (tied_source)
                                throw std::invalid_argument("CPU tied alias has ambiguous original plan source");
                            tied_source=&candidate;
                        }
                }
                const auto projected = projectCPUDenseTPRequirement(requirement, tp_config);
                const auto projected_tied = tied_source ? projectCPUDenseTPRequirement(*tied_source, tp_config) : std::nullopt;
                materializeCPURequirement(projected ? *projected : requirement, plan.strategy(),
                    plan.physicalMemoryOwner(), source_budget, builder, existing,
                    projected_tied ? &*projected_tied : tied_source);
                continue;
            }
            const std::string &load_name = requirement.source_name.empty()
                                               ? requirement.canonical_name
                                               : requirement.source_name;
            const DeviceId lookup_device = requirement.lookup_device.value_or(requirement.target_device);
            std::shared_ptr<TensorBase> tensor;
            if (requirement.projection_ownership)
            {
                if (weight_precision_ != WeightPrecision::NATIVE)
                    throw std::invalid_argument("Static expert projection requires original native weights, without conversion");
                tensor = BoundedGPUExpertSource::declare(loader_, load_name,
                    projectNativeWeightSource(requirement, loader_.nativeWeightSourceGeometry(load_name)),
                    requirement.target_device);
                const std::vector<size_t> expected{requirement.slice.row_count,
                    requirement.slice.col_count, requirement.slice.expert_count};
                if (!tensor || tensor->shape() != expected)
                    throw std::runtime_error("Static expert projection did not materialize its exact compact original source: " + load_name);
                auto compact = requirement.slice;
                compact.inner_is_presliced = true;
                registerDerivedMetadata(requirement.canonical_name, tensor,
                    WeightDerivationKind::ExpertSlice, compact, lookup_device);
                // Do not enter getWeightForDevice on any failed explicit source
                // request: that generic path could materialize the full parent.
            }
            else if (requirement.derivation == WeightDerivationKind::ExpertSlice &&
                !requirement.slice.expert_ids.empty())
            {
                /*
                 * ExpertOverlay participants own an explicit, often
                 * non-contiguous set of logical experts.  Loading the complete
                 * 3-D parent here would fault every unrelated expert page and
                 * defeat the participant-only weight plan.  Materialize the
                 * declared expert axis directly from the GGUF instead.  The
                 * resulting tensor keeps packed slots in the same order as
                 * slice.expert_ids, which the GPU expert pipeline and CPU
                 * participant preparation both consume as authoritative.
                 */
                std::vector<size_t> selected_experts;
                selected_experts.reserve(requirement.slice.expert_ids.size());
                for (const int expert_id : requirement.slice.expert_ids)
                {
                    if (expert_id < 0)
                    {
                        throw std::runtime_error(
                            "[WeightManager] Expert-slice requirement contains a negative expert id: " +
                            requirement.canonical_name);
                    }
                    selected_experts.push_back(static_cast<size_t>(expert_id));
                }

                tensor = loader_.loadTensorExpertSelection(
                    load_name,
                    selected_experts,
                    lookup_device,
                    weight_precision_);
                if (!tensor)
                {
                    throw std::runtime_error(
                        "[WeightManager] Explicit expert selection could not materialize its declared source: " +
                        load_name);
                }
                // Expert-axis selection preserves the original K/N geometry.
                // Reject a full parent or malformed selection before publishing
                // logical subset metadata; neither may enter generic loading.
                const auto source_shape = loader_.getTensorShape(load_name);
                if (!source_shape || source_shape->size() != 3u ||
                    tensor->shape() != std::vector<size_t>{
                        (*source_shape)[0], (*source_shape)[1], selected_experts.size()})
                {
                    throw std::runtime_error(
                        "[WeightManager] Explicit expert selection changed original matrix geometry or selected expert extent: " +
                        load_name + " (selected experts=" + std::to_string(selected_experts.size()) + ")");
                }
                WeightSliceSpec selected_slice = requirement.slice;
                selected_slice.expert_start = selected_experts.front();
                selected_slice.expert_count = selected_experts.size();
                selected_slice.inner_is_presliced = true;
                registerDerivedMetadata(
                    requirement.canonical_name,
                    tensor,
                    WeightDerivationKind::ExpertSlice,
                    selected_slice,
                    lookup_device);
            }
            else if(requirement.target_device.is_gpu() && requirement.lookup_device && requirement.lookup_device->is_cpu())
            {
                tensor=originalGPUPreparationSource(requirement,load_name);
            }
            else if (requirement.bypass_tensor_parallel)
            {
                tensor = getReplicatedWeight(load_name, lookup_device);
                if (!tensor && load_name == "output.weight" && requirement.canonical_name == "output.weight")
                    tensor = getReplicatedWeight("token_embd.weight", lookup_device);
            }
            else if (tp_config && requirement.tp_domain >= 0 && requirement.tp_rank_or_device_index >= 0)
            {
                try
                {
                    const auto &assignment = tp_config->forRank(requirement.tp_rank_or_device_index);
                    tensor = getShardedWeightForAssignment(
                        load_name,
                        lookup_device,
                        assignment,
                        requirement.layer);
                }
                catch (const std::out_of_range &)
                {
                    LOG_WARN("[WeightManager] TP rank " << requirement.tp_rank_or_device_index
                                                        << " not in TensorParallelConfig while materializing "
                                                        << requirement.canonical_name
                                                        << "; falling back to device lookup");
                }
            }
            if (!tensor)
            {
                tensor = getWeightForDevice(
                    load_name,
                    lookup_device,
                    requirement.layer);
            }

            if (!tensor)
            {
                if (requirement.required)
                {
                    throw std::runtime_error("[WeightManager] Required weight missing during materialization: " +
                                             requirement.canonical_name +
                                             (load_name == requirement.canonical_name ? "" : " (source: " + load_name + ")"));
                }
                continue;
            }

            WeightBinding binding;
            binding.binding_id = stableMaterializedBindingId(requirement, plan.strategy());
            binding.identity = makeSourceWeightIdentity(
                requirement.canonical_name,
                plan.strategy().model_id,
                binding.binding_id);
            binding.identity.derivation = requirement.derivation;
            binding.identity.role = requirement.role == WeightRole::Other
                                        ? inferWeightRole(requirement.canonical_name)
                                        : requirement.role;
            binding.identity.layer = requirement.layer >= 0
                                         ? requirement.layer
                                         : inferWeightLayer(requirement.canonical_name);
            binding.identity.expert = requirement.expert >= 0
                                          ? requirement.expert
                                          : inferWeightExpert(requirement.canonical_name);
            binding.identity.pp_stage = requirement.pp_stage;
            binding.identity.tp_domain = requirement.tp_domain;
            binding.identity.tp_rank_or_device_index = requirement.tp_rank_or_device_index;
            binding.identity.residency_category = requirement.residency_category;
            binding.identity.overlay_domain = requirement.overlay_domain;
            binding.identity.overlay_participant_index = requirement.overlay_participant_index;
            binding.identity.overlay_participant_world_rank = requirement.overlay_participant_world_rank;

            binding.slice = requirement.slice;
            binding.projection_ownership = requirement.projection_ownership;
            binding.joint_source = requirement.joint_source;
            binding.residency.home_device = requirement.target_device;
            binding.residency.resident_device = requirement.target_device;
            binding.residency.host_policy = requirement.host_policy;
            binding.tensor_owner = tensor;
            binding.tensor = tensor.get();

            if (weight_metadata_)
            {
                if (auto metadata = weight_metadata_->metadata(tensor.get()))
                {
                    /*
                     * Tensor metadata owns physical facts such as slice and
                     * residency.  The WeightPlan owns graph binding identity:
                     * canonical name, role, layer, expert, and TP/PP ownership.
                     * Keeping that boundary strict prevents stale pointer-keyed
                     * metadata from reclassifying a freshly materialized global
                     * binding after a previous runner has been destroyed.
                     */
                    binding.identity = metadata->identity;
                    binding.identity.canonical_name = requirement.canonical_name;
                    binding.identity.model_id = plan.strategy().model_id;
                    binding.identity.logical_id = stableWeightLogicalId(requirement.canonical_name);
                    binding.identity.role = requirement.role == WeightRole::Other
                                                ? inferWeightRole(requirement.canonical_name)
                                                : requirement.role;
                    binding.identity.derivation = requirement.derivation;
                    binding.identity.layer = requirement.layer;
                    binding.identity.expert = requirement.expert;
                    binding.identity.pp_stage = requirement.pp_stage;
                    binding.identity.tp_domain = requirement.tp_domain;
                    binding.identity.tp_rank_or_device_index = requirement.tp_rank_or_device_index;
                    binding.identity.residency_category = requirement.residency_category;
                    binding.identity.overlay_domain = requirement.overlay_domain;
                    binding.identity.overlay_participant_index = requirement.overlay_participant_index;
                    binding.identity.overlay_participant_world_rank = requirement.overlay_participant_world_rank;
                    binding.slice = metadata->slice;
                    binding.residency = metadata->residency;
                    binding.residency.home_device = requirement.target_device;
                    binding.residency.resident_device = requirement.target_device;
                    binding.residency.host_policy = requirement.host_policy;
                }
            }

            binding.slice = mergeRequirementSlice(binding.slice, requirement.slice);

            /*
             * Register the binding's consumption policy against the physical
             * source before any prepared representation can make that source
             * look reclaimable. ExpertOverlay may bind disjoint accelerator and
             * CPU participants through the same model authority; the metadata
             * registry performs the monotonic policy join so registration order
             * cannot retire bytes still read by a CPU floating-point engine.
             */
            if (weight_metadata_ &&
                !weight_metadata_->mergeHostPolicy(
                    tensor.get(), requirement.host_policy))
            {
                throw std::runtime_error(
                    "[WeightManager] Materialized binding has no source metadata for host-policy accounting: " +
                    requirement.canonical_name);
            }

            if (!hasExplicitSlice(binding.slice) && tensor)
                binding.slice = fullSliceSpec(*tensor);

            // Original source format is captured before the store-owned override.
            const TensorType gdn_source_type = tensor->native_type();
            const size_t gdn_source_bytes = tensor->size_bytes();
            if (auto prepared_override = createModelPreparedFloatingOverride(
                    requirement.canonical_name,
                    binding.identity.role,
                    tensor.get(),
                    requirement.target_device,
                    binding.slice,
                    binding.identity.derivation))
            {
                tensor = prepared_override;
                binding.tensor_owner = prepared_override;
                binding.tensor = prepared_override.get();
            }

            const auto gdn_arithmetic = PreparedWeightRepresentationContract::effectiveGDNArithmetic(
                binding.identity.role, gdn_source_type, requirement.canonical_name, loader_.architecture(),
                debugEnv().gemm.gdn_projection_arithmetic);
            if (gdn_arithmetic != GDNProjectionArithmetic::Native)
            {
                const bool half = gdn_arithmetic == GDNProjectionArithmetic::FP16;
                if (tensor->native_type() != (half ? TensorType::FP16 : TensorType::FP32))
                    throw std::runtime_error("GDN floating setup failed to materialize " + requirement.canonical_name);
                PerfStatsCollector::addCounter("weight_loading", "gdn_projection_arithmetic_prepared_weight", 1.0,
                    "materialize", requirement.target_device.to_string(),
                    {{"canonical_name", requirement.canonical_name},
                     {"weight_role", toString(binding.identity.role)},
                     {"policy", std::string(gdnProjectionArithmeticName(debugEnv().gemm.gdn_projection_arithmetic))},
                     {"source_format", std::string(tensorTypeName(gdn_source_type))},
                     {"prepared_format", half ? "FP16" : "FP32"},
                     {"source_path", half ? "source_decoded_half_rne" :
                         (gdn_source_type == TensorType::FP32 ? "original_fp32" : "source_dequantized_fp32")},
                     {"source_bytes", std::to_string(gdn_source_bytes)},
                     {"prepared_bytes", std::to_string(tensor->size_bytes())}});
            }
            if (binding.identity.role == WeightRole::MoERouter &&
                resolveMoERouterArithmetic(debugEnv().gemm.moe_router_arithmetic, loader_.architecture()) == MoERouterArithmetic::FP16)
            {
                if (tensor->native_type() != TensorType::FP16)
                    throw std::runtime_error("Router half setup failed to materialize " + requirement.canonical_name);
                PerfStatsCollector::addCounter("weight_loading", "moe_router_arithmetic_prepared_weight", 1.0,
                    "materialize", requirement.target_device.to_string(),
                    {{"canonical_name", requirement.canonical_name},
                     {"weight_role", toString(binding.identity.role)}, {"policy", "fp16"},
                     {"source_format", std::string(tensorTypeName(gdn_source_type))},
                     {"prepared_format", "FP16"}, {"activation_operand", "half_rne"},
                     {"accumulation", "fp32"},
                     {"source_bytes", std::to_string(gdn_source_bytes)},
                     {"prepared_bytes", std::to_string(tensor->size_bytes())}});
            }
            binding.floating_operand_precision = PreparedWeightRepresentationContract::operandPrecision(
                binding.identity.role, gdn_source_type, requirement.canonical_name, loader_.architecture(),
                debugEnv().gemm.gdn_projection_arithmetic, debugEnv().gemm.moe_router_arithmetic);
            binding.identity.instance_id = binding.binding_id;

            if (requirement.expected_prepared_kind != PreparedWeightKind::None)
            {
                binding.prepared = PreparedWeightRef{
                    plan.strategy().model_id,
                    0,
                    requirement.expected_prepared_kind,
                    requirement.target_device};
            }

            auto &added = builder.addBinding(std::move(binding));
            if (added.prepared)
                added.prepared->binding_id = added.binding_id;

            // Source/slice/residency and prepared-reference identity are final.
            // Seal this declaration before adoption publishes a store-owned copy;
            // freezing only the returned vector would leave that copy mutable.
            added.immutable = true;
            if (added.prepared.has_value())
            {
                if (auto store = preparedWeightStoreIfInitialized())
                {
                    if (store->adoptPreparedForBinding(added, requirement.target_device))
                    {
                        WeightLifecycleTrace::record(
                            WeightLifecycleEventType::RegisterPrepared,
                            requirement.canonical_name,
                            added.identity.role,
                            added.identity.layer,
                            requirement.target_device,
                            "adopted prepared handle for materialized binding");
                    }
                }
            }

            WeightLifecycleTrace::record(
                WeightLifecycleEventType::GraphBind,
                requirement.canonical_name,
                added.identity.role,
                added.identity.layer,
                requirement.target_device,
                "materialized binding " + std::to_string(added.binding_id));
        }

        FrozenModelWeightSet frozen(
            plan.strategy(),
            builder.freezeBindings(),
            plan.physicalMemoryOwner());
        frozen.validateForGraph();
        return frozen;
    }

    // =========================================================================
    // Decode Weight Shard Support (CPU Decode Participation - Option A)
    // =========================================================================

    std::shared_ptr<TensorBase> WeightManager::getDecodeWeight(
        const std::string &name,
        DeviceId decode_device,
        float fraction,
        int layer_idx)
    {
        // Validate fraction
        if (!std::isfinite(fraction) || fraction <= 0.0f || fraction > 1.0f)
        {
            throw std::runtime_error("[WeightManager] Invalid decode fraction: " +
                                     std::to_string(fraction) + " (must be 0 < fraction <= 1)");
        }

        const auto mode=has_sharding_config_?getShardingMode(name):ShardingMode::REPLICATE;
        const DecodeWeightCacheKey key{name,decode_device,std::bit_cast<uint32_t>(fraction),mode};
        if (decode_device.is_cpu()) {
            std::shared_ptr<TensorBase> cached;
            std::shared_ptr<PreparedWeightStore> original_store;
            {
                std::lock_guard snapshot(cache_mutex_);
                original_store=prepared_weight_store_;
                const auto entry=decode_cache_.find(key);
                if(entry!=decode_cache_.end())cached=entry->second;
            }
            if(cached) {
                if(!hasOriginalCPUFinalOwner(cached,decode_device,original_store))
                    throw std::logic_error("CPU decode index lost its original final Store owner");
                std::lock_guard verify(cache_mutex_);
                if(prepared_weight_store_!=original_store)
                    throw std::logic_error("CPU decode index changed its original Store while authenticating");
                return cached;
            }

            const auto geometry=loader_.nativeWeightSourceGeometry(name);
            WeightSliceSpec slice;
            if (mode!=ShardingMode::REPLICATE && mode!=ShardingMode::EXPERT_ID_APPORTIONED) {
                if (geometry.original_shape.size()!=2)
                    throw std::invalid_argument("CPU decode range requires original matrix geometry");
                slice.source_rows=geometry.original_shape[1];
                slice.source_cols=geometry.original_shape[0];
                slice.row_count=slice.source_rows; slice.col_count=slice.source_cols;
                if (mode==ShardingMode::COLUMN_PARALLEL) {
                    slice.row_count=size_t(std::ceil(double(slice.source_rows)*fraction));
                    slice.row_start=slice.source_rows-slice.row_count;
                } else if (mode==ShardingMode::ROW_PARALLEL || mode==ShardingMode::INPUT_PARALLEL) {
                    slice.col_count=size_t(std::ceil(double(slice.source_cols)*fraction));
                    slice.col_start=slice.source_cols-slice.col_count;
                } else throw std::invalid_argument("CPU decode source has unsupported original sharding policy");
                slice.inner_is_presliced=true;
            }
            auto completed=loadCPUWeightRange(name,decode_device,mode==ShardingMode::REPLICATE?
                WeightDerivationKind::Source:WeightDerivationKind::DecodeShard,std::move(slice));
            if(!completed)return completed;
            const auto completed_store=preparedWeightStoreIfInitialized();
            if(!hasOriginalCPUFinalOwner(completed,decode_device,completed_store))
                throw std::logic_error("CPU decode publication has no original final Store owner");
            std::lock_guard publish(cache_mutex_);
            if(prepared_weight_store_!=completed_store)
                throw std::logic_error("CPU decode publication changed its original Store");
            const auto [entry,inserted]=decode_cache_.emplace(key,completed);
            if(!inserted && entry->second!=completed)
                throw std::logic_error("CPU decode publication replaced an original final owner");
            return entry->second;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Generate cache key (includes fraction to allow different shard sizes)
        const auto &cache_key=key;

        // Check decode cache first
        auto it = decode_cache_.find(cache_key);
        if (it != decode_cache_.end())
        {
            return it->second;
        }

        // Get the full weight tensor from cache or load fresh
        // NOTE: We cannot call getWeightForDevice() here because we already hold cache_mutex_
        // Instead, check cache directly and load via getReplicatedWeight if needed
        std::shared_ptr<TensorBase> full_tensor;
        auto cache_it = cache_.find(name);
        if (cache_it != cache_.end())
        {
            full_tensor = cache_it->second;
        }
        else
        {
            // Load fresh - use replicated loading (decode shards are always on CPU)
            full_tensor = getReplicatedWeight(name, DeviceId::cpu());
            if (full_tensor)
            {
                cache_[name] = full_tensor;
            }
        }

        if (!full_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load full weight for decode shard: " + name);
        }

        std::shared_ptr<TensorBase> decode_shard;

        switch (mode)
        {
        case ShardingMode::REPLICATE:
        {
            // Replicated weights (norms, biases) - return full copy
            // These are small, so no need to slice
            if (fraction >= 1.0f)
            {
                // Full copy requested - just return the tensor
                decode_shard = full_tensor;
            }
            else
            {
                // Still return full tensor for replicated weights
                // Slicing doesn't make sense for norms/biases
                LOG_DEBUG("[WeightManager] Decode shard for REPLICATE weight " << name
                                                                               << " - returning full copy (fraction ignored)");
                decode_shard = full_tensor;
            }
            break;
        }

        case ShardingMode::COLUMN_PARALLEL:
        {
            // Column-parallel weights (Q, K, V, Gate, Up) - slice tail rows
            // These weights have shape [out_dim, in_dim]
            // We slice the output dimension (rows) for decode
            decode_shard = sliceTailRows(full_tensor, fraction);
            if (!decode_shard)
            {
                throw std::runtime_error("[WeightManager] Failed to slice tail rows for decode: " + name);
            }
            LOG_DEBUG("[WeightManager] Decode shard for COLUMN_PARALLEL " << name
                                                                          << ": tail " << (fraction * 100) << "% rows -> ["
                                                                          << decode_shard->shape()[0] << ", "
                                                                          << decode_shard->shape()[1] << "]");
            break;
        }

        case ShardingMode::ROW_PARALLEL:
        case ShardingMode::INPUT_PARALLEL:
        {
            // Row/Input-parallel weights (Wo, Down) - slice tail columns
            // These weights have shape [out_dim, in_dim]
            // We slice the input dimension (columns) for decode
            decode_shard = sliceTailColumns(full_tensor, fraction);
            if (!decode_shard)
            {
                throw std::runtime_error("[WeightManager] Failed to slice tail columns for decode: " + name);
            }
            LOG_DEBUG("[WeightManager] Decode shard for " << (mode == ShardingMode::ROW_PARALLEL ? "ROW_PARALLEL" : "INPUT_PARALLEL")
                                                          << " " << name
                                                          << ": tail " << (fraction * 100) << "% cols -> ["
                                                          << decode_shard->shape()[0] << ", "
                                                          << decode_shard->shape()[1] << "]");
            break;
        }

        case ShardingMode::EXPERT_ID_APPORTIONED:
        {
            // Expert-ID-apportioned weights are 3D MoE tensors; decode sharding
            // not applicable. Return the full tensor as-is.
            decode_shard = full_tensor;
            break;
        }

        default:
            throw std::runtime_error("[WeightManager] Unknown sharding mode for decode weight: " + name);
        }

        // Cache the decode shard
        if (decode_shard)
        {
            if (decode_shard.get() != full_tensor.get())
            {
                WeightSliceSpec decode_slice = fullSliceSpec(*full_tensor);
                decode_slice.inner_is_presliced = true;
                if (mode == ShardingMode::COLUMN_PARALLEL)
                {
                    decode_slice.row_count = decode_shard->shape()[0];
                    decode_slice.row_start = decode_slice.source_rows - decode_slice.row_count;
                    decode_slice.col_start = 0;
                    decode_slice.col_count = decode_slice.source_cols;
                }
                else
                {
                    decode_slice.row_start = 0;
                    decode_slice.row_count = decode_slice.source_rows;
                    decode_slice.col_count = decode_shard->shape().size() > 1 ? decode_shard->shape()[1] : 1;
                    decode_slice.col_start = decode_slice.source_cols - decode_slice.col_count;
                }
                registerDerivedMetadata(name, decode_shard, WeightDerivationKind::DecodeShard, decode_slice, decode_device);
            }
            decode_cache_[cache_key] = decode_shard;
        }

        return decode_shard;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceTailRows(
        const std::shared_ptr<TensorBase> &full_tensor,
        float fraction)
    {
        // For weight matrix [out_dim, in_dim], extract tail rows
        // Result: [out_local, in_dim] where out_local = out_dim * fraction

        const auto &shape = full_tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] Tail row slicing requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];

        // Calculate slice dimensions
        size_t out_local = static_cast<size_t>(std::ceil(out_dim * fraction));
        if (out_local == 0)
        {
            out_local = 1; // At least one row
        }
        if (out_local > out_dim)
        {
            out_local = out_dim;
        }

        size_t out_start = out_dim - out_local; // Tail portion

        LOG_TRACE("[WeightManager] Tail row slicing: [" << out_dim << ", " << in_dim
                                                        << "] -> rows [" << out_start << ", " << out_dim
                                                        << ") = [" << out_local << ", " << in_dim << "]");

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] Tail row slicing currently requires FP32 tensor. "
                                     "Use CONVERT_TO_FP32 weight precision for decode shards.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {out_local, in_dim};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy tail rows
        const float *src = fp32_tensor->data() + out_start * in_dim;
        float *dst = sliced->mutable_data();
        std::memcpy(dst, src, out_local * in_dim * sizeof(float));

        return sliced;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceTailColumns(
        const std::shared_ptr<TensorBase> &full_tensor,
        float fraction)
    {
        // For weight matrix [out_dim, in_dim], extract tail columns
        // Result: [out_dim, in_local] where in_local = in_dim * fraction

        const auto &shape = full_tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] Tail column slicing requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];

        // Calculate slice dimensions
        size_t in_local = static_cast<size_t>(std::ceil(in_dim * fraction));
        if (in_local == 0)
        {
            in_local = 1; // At least one column
        }
        if (in_local > in_dim)
        {
            in_local = in_dim;
        }

        size_t in_start = in_dim - in_local; // Tail portion

        LOG_TRACE("[WeightManager] Tail column slicing: [" << out_dim << ", " << in_dim
                                                           << "] -> cols [" << in_start << ", " << in_dim
                                                           << ") = [" << out_dim << ", " << in_local << "]");

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] Tail column slicing currently requires FP32 tensor. "
                                     "Use CONVERT_TO_FP32 weight precision for decode shards.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {out_dim, in_local};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy tail columns (non-contiguous in memory)
        const float *src = fp32_tensor->data();
        float *dst = sliced->mutable_data();

#pragma omp parallel for
        for (size_t row = 0; row < out_dim; ++row)
        {
            const float *src_row = src + row * in_dim + in_start;
            float *dst_row = dst + row * in_local;
            std::memcpy(dst_row, src_row, in_local * sizeof(float));
        }

        return sliced;
    }

    // =========================================================================
    // Proportional Slicing (TensorParallelConfig support)
    // =========================================================================

    WeightManager::WeightCategory WeightManager::categorizeWeight(const std::string &name) const
    {
        // Attention Q/K/V projections (column-parallel by heads)
        if (name.find("attn_q.") != std::string::npos ||
            name.find("attn_k.") != std::string::npos ||
            name.find("attn_v.") != std::string::npos ||
            name.find("attn_qkv.") != std::string::npos)
        {
            return WeightCategory::ATTENTION_QKV;
        }

        // Attention output projection (row-parallel)
        if (name.find("attn_output.") != std::string::npos)
        {
            return WeightCategory::ATTENTION_WO;
        }

        // FFN gate and up projections (column-parallel by d_ff)
        if (name.find("ffn_gate.") != std::string::npos ||
            name.find("ffn_up.") != std::string::npos ||
            name.find("ffn_gate_up.") != std::string::npos)
        {
            return WeightCategory::FFN_GATE_UP;
        }

        // FFN down projection (input-parallel)
        if (name.find("ffn_down.") != std::string::npos)
        {
            return WeightCategory::FFN_DOWN;
        }

        // LM head (column-parallel by vocab)
        if (name.find("output.weight") != std::string::npos)
        {
            return WeightCategory::LM_HEAD;
        }

        // Everything else is replicated
        return WeightCategory::REPLICATE;
    }

    std::pair<size_t, size_t> WeightManager::calculateProportionalColumnSlice(
        const std::string &name, size_t total_rows, const TensorParallelConfig &assignment_config) const
    {
        const auto *tp_config = &assignment_config;
        if (!mpi_ctx_)
        {
            // Fallback to equal split
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            int world_size = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
            size_t rows_per_rank = total_rows / world_size;
            size_t start = rows_per_rank * rank;
            size_t count = (rank == world_size - 1) ? (total_rows - start) : rows_per_rank;
            return {start, count};
        }

        int rank = mpi_ctx_->rank();
        const auto &assignment = tp_config->forRank(rank);
        WeightCategory category = categorizeWeight(name);

        switch (category)
        {
        case WeightCategory::ATTENTION_QKV:
        {
            // Column-parallel by heads
            // For Q: full head_dim per head, for K/V: use kv_head mapping
            // Q weight shape: [n_heads * head_dim, d_model]
            // K/V weight shape: [n_kv_heads * head_dim, d_model]
            // Each rank gets [local_head_count * head_dim, d_model]
            const bool is_kv_weight = (name.find("attn_k.") != std::string::npos ||
                                       name.find("attn_v.") != std::string::npos);

            // CRITICAL: K/V weights use totalKVHeads(), not totalHeads()!
            const int total_heads_for_weight = is_kv_weight
                                                   ? tp_config->totalKVHeads()
                                                   : tp_config->totalHeads();
            if (total_heads_for_weight <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_heads/total_kv_heads in TensorParallelConfig");
            }
            const size_t head_dim = total_rows / static_cast<size_t>(total_heads_for_weight);

            if (is_kv_weight)
            {
                // K/V use KV heads
                size_t start = assignment.kv_head_start * head_dim;
                size_t count = assignment.kv_head_count * head_dim;
                LOG_TRACE("[WeightManager] Proportional KV slice for " << name
                                                                       << ": total_kv_heads=" << total_heads_for_weight
                                                                       << " head_dim=" << head_dim
                                                                       << " kv_heads [" << assignment.kv_head_start << ", "
                                                                       << (assignment.kv_head_start + assignment.kv_head_count) << ")"
                                                                       << " -> rows [" << start << ", " << (start + count) << ")");
                return {start, count};
            }
            else
            {
                // Q uses full Q heads
                size_t start = assignment.head_start * head_dim;
                size_t count = assignment.head_count * head_dim;
                LOG_TRACE("[WeightManager] Proportional Q slice for " << name
                                                                      << ": total_heads=" << total_heads_for_weight
                                                                      << " head_dim=" << head_dim
                                                                      << " heads [" << assignment.head_start << ", "
                                                                      << (assignment.head_start + assignment.head_count) << ")"
                                                                      << " -> rows [" << start << ", " << (start + count) << ")");
                return {start, count};
            }
        }

        case WeightCategory::FFN_GATE_UP:
        {
            // Column-parallel by d_ff
            // Weight shape: [d_ff, d_model]
            // Each rank gets [d_ff_count, d_model]
            size_t start = assignment.d_ff_start;
            size_t count = assignment.d_ff_count;
            LOG_TRACE("[WeightManager] Proportional FFN gate/up slice for " << name
                                                                            << ": d_ff [" << start << ", " << (start + count) << ")");
            return {start, count};
        }

        case WeightCategory::LM_HEAD:
        {
            // Column-parallel by vocab
            // Weight shape: [vocab_size, d_model]
            // Each rank gets [vocab_count, d_model]
            size_t start = assignment.vocab_start;
            size_t count = assignment.vocab_count;
            LOG_TRACE("[WeightManager] Proportional LM head slice for " << name
                                                                        << ": vocab [" << start << ", " << (start + count) << ")");
            return {start, count};
        }

        default:
        {
            // Check schema-based dimension type for weights not covered by WeightCategory
            // (e.g. GDN ssm_alpha/beta/dt.bias/ssm_a which are ProportionalHeads)
            if (has_sharding_config_)
            {
                WeightDimensionType dim_type = sharding_config_.getDimensionType(name);
                if (dim_type == WeightDimensionType::ProportionalHeads)
                {
                    const int total_heads = tp_config->totalHeads();
                    if (total_heads > 0)
                    {
                        size_t start = total_rows * static_cast<size_t>(assignment.head_start) / static_cast<size_t>(total_heads);
                        size_t end = total_rows * static_cast<size_t>(assignment.head_start + assignment.head_count) / static_cast<size_t>(total_heads);
                        LOG_TRACE("[WeightManager] Proportional heads slice for " << name
                                                                                  << ": total_heads=" << total_heads
                                                                                  << " weight_rows=" << total_rows
                                                                                  << " -> [" << start << ", " << end << ")");
                        return {start, end - start};
                    }
                }
                else if (dim_type == WeightDimensionType::Heads)
                {
                    const int total_heads = tp_config->totalHeads();
                    if (total_heads > 0)
                    {
                        const size_t head_dim = total_rows / static_cast<size_t>(total_heads);
                        size_t start = static_cast<size_t>(assignment.head_start) * head_dim;
                        size_t count = static_cast<size_t>(assignment.head_count) * head_dim;
                        LOG_TRACE("[WeightManager] Schema heads slice for " << name
                                                                            << ": total_heads=" << total_heads
                                                                            << " head_dim=" << head_dim
                                                                            << " -> [" << start << ", " << (start + count) << ")");
                        return {start, count};
                    }
                }
                else if (dim_type == WeightDimensionType::KVHeads)
                {
                    const int total_kv_heads = tp_config->totalKVHeads();
                    if (total_kv_heads > 0)
                    {
                        const size_t head_dim = total_rows / static_cast<size_t>(total_kv_heads);
                        size_t start = static_cast<size_t>(assignment.kv_head_start) * head_dim;
                        size_t count = static_cast<size_t>(assignment.kv_head_count) * head_dim;
                        LOG_TRACE("[WeightManager] Schema KV heads slice for " << name
                                                                               << " -> [" << start << ", " << (start + count) << ")");
                        return {start, count};
                    }
                }
            }
            LOG_WARN("[WeightManager] calculateProportionalColumnSlice called for non-column weight: " << name);
            return {0, total_rows};
        }
        } // end switch
        return {0, total_rows}; // unreachable but silences -Wreturn-type
    }

    std::pair<size_t, size_t> WeightManager::calculateProportionalRowSlice(
        const std::string &name, size_t total_cols, const TensorParallelConfig &assignment_config) const
    {
        const auto *tp_config = &assignment_config;
        if (!mpi_ctx_)
        {
            // Fallback to equal split
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            int world_size = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
            size_t cols_per_rank = total_cols / world_size;
            size_t start = cols_per_rank * rank;
            size_t count = (rank == world_size - 1) ? (total_cols - start) : cols_per_rank;
            return {start, count};
        }

        int rank = mpi_ctx_->rank();
        const auto &assignment = tp_config->forRank(rank);
        WeightCategory category = categorizeWeight(name);

        switch (category)
        {
        case WeightCategory::ATTENTION_WO:
        {
            // Row-parallel: input dimension matches column-parallel Q output
            // Weight shape: [d_model, n_heads * head_dim]
            // Each rank needs columns [head_start * head_dim, (head_start + head_count) * head_dim)
            int total_heads = tp_config->totalHeads();
            if (total_heads <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_heads in TensorParallelConfig");
            }
            size_t head_dim = total_cols / total_heads;
            size_t start = assignment.head_start * head_dim;
            size_t count = assignment.head_count * head_dim;
            LOG_TRACE("[WeightManager] Proportional Wo slice for " << name
                                                                   << ": cols [" << start << ", " << (start + count) << ")");
            return {start, count};
        }

        case WeightCategory::FFN_DOWN:
        {
            // Input-parallel: input dimension matches column-parallel gate/up output
            // Weight shape: [d_model, d_ff]
            // Each rank needs columns [d_ff_start, d_ff_start + d_ff_count)
            size_t start = assignment.d_ff_start;
            size_t count = assignment.d_ff_count;
            LOG_TRACE("[WeightManager] Proportional FFN down slice for " << name
                                                                         << ": cols [" << start << ", " << (start + count) << ")");
            return {start, count};
        }

        default:
        {
            // Check schema-based dimension type for weights not covered by WeightCategory
            // (e.g. GDN ssm_out which is InputParallel with ProportionalHeads or Heads)
            if (has_sharding_config_)
            {
                WeightDimensionType dim_type = sharding_config_.getDimensionType(name);
                if (dim_type == WeightDimensionType::ProportionalHeads)
                {
                    const int total_heads = tp_config->totalHeads();
                    if (total_heads > 0)
                    {
                        size_t start = total_cols * static_cast<size_t>(assignment.head_start) / static_cast<size_t>(total_heads);
                        size_t end = total_cols * static_cast<size_t>(assignment.head_start + assignment.head_count) / static_cast<size_t>(total_heads);
                        LOG_TRACE("[WeightManager] Proportional heads row-slice for " << name
                                                                                      << ": total_heads=" << total_heads
                                                                                      << " weight_cols=" << total_cols
                                                                                      << " -> [" << start << ", " << end << ")");
                        return {start, end - start};
                    }
                }
                else if (dim_type == WeightDimensionType::Heads)
                {
                    const int total_heads = tp_config->totalHeads();
                    if (total_heads > 0)
                    {
                        const size_t head_dim = total_cols / static_cast<size_t>(total_heads);
                        size_t start = static_cast<size_t>(assignment.head_start) * head_dim;
                        size_t count = static_cast<size_t>(assignment.head_count) * head_dim;
                        LOG_WARN("[WeightManager] Schema heads row-slice for " << name
                                                                               << " total_cols=" << total_cols
                                                                               << " total_heads=" << total_heads
                                                                               << " head_dim=" << head_dim
                                                                               << " assignment.head_start=" << assignment.head_start
                                                                               << " head_count=" << assignment.head_count
                                                                               << " -> [" << start << ", " << (start + count) << ")");
                        return {start, count};
                    }
                }
                else if (dim_type == WeightDimensionType::KVHeads)
                {
                    const int total_kv_heads = tp_config->totalKVHeads();
                    if (total_kv_heads > 0)
                    {
                        const size_t head_dim = total_cols / static_cast<size_t>(total_kv_heads);
                        size_t start = static_cast<size_t>(assignment.kv_head_start) * head_dim;
                        size_t count = static_cast<size_t>(assignment.kv_head_count) * head_dim;
                        LOG_TRACE("[WeightManager] Schema KV heads row-slice for " << name
                                                                                   << " -> [" << start << ", " << (start + count) << ")");
                        return {start, count};
                    }
                }
            }
            LOG_WARN("[WeightManager] calculateProportionalRowSlice called for non-row weight: " << name);
            return {0, total_cols};
        }
        } // end switch
        return {0, total_cols}; // unreachable but silences -Wreturn-type
    }

    // =========================================================================
    // Thread-safe cache accessors
    // =========================================================================

    size_t WeightManager::cacheSize() const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cache_.size()+std::count_if(per_device_cache_.begin(),per_device_cache_.end(),
            [](const auto &entry){return entry.first.device.is_cpu();});
    }

    void WeightManager::clearCache()
    {
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache_.clear();
            per_device_cache_.clear();
        }

        {
            std::lock_guard<std::mutex> lock(prep_ticket_mutex_);
            prep_tickets_.clear();
            expected_devices_by_weight_.clear();
            reclaim_ready_weights_.clear();
            reclaim_applied_weights_.clear();
        }
    }

    size_t WeightManager::decodeCacheSize() const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return decode_cache_.size();
    }

    void WeightManager::clearDecodeCache()
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        decode_cache_.clear();
    }

    // =========================================================================
    // Device-aware weight access (for multi-device LOCAL TP)
    // =========================================================================

    std::shared_ptr<TensorBase> WeightManager::cloneTensorForDevice(
        const std::string &name,
        const std::shared_ptr<TensorBase> &original,
        DeviceId target_device)
    {
        if (!original)
        {
            return nullptr;
        }

        TensorType tensor_type = original->native_type();
        size_t byte_count = original->size_bytes();
        const void *raw_ptr = original->raw_data();

        if (!raw_ptr || byte_count == 0)
        {
            throw std::runtime_error("[WeightManager] Cannot clone tensor with no data: " + name);
        }

        // Copy raw bytes
        std::vector<uint8_t> raw_copy(byte_count);
        std::memcpy(raw_copy.data(), raw_ptr, byte_count);

        auto unique_clone = createTensorFromRawData(
            tensor_type, original->shape(), std::move(raw_copy));
        if (!unique_clone)
        {
            throw std::runtime_error("[WeightManager] Failed to create clone for tensor type " +
                                     std::to_string(static_cast<int>(tensor_type)));
        }
        std::shared_ptr<TensorBase> clone = std::shared_ptr<TensorBase>(std::move(unique_clone));

        clone->setDebugName(name + "@" + target_device.to_string());

        LOG_TRACE("[WeightManager] Cloned tensor for " << target_device.to_string()
                                                       << ": " << name << " ["
                                                       << original->shape()[0] << "x"
                                                       << (original->shape().size() > 1 ? original->shape()[1] : 1) << "]"
                                                       << " type=" << static_cast<int>(tensor_type));

        // Enhanced logging for non-GEMM weights (biases, norms) to debug multi-GPU pointer issues
        if (name.find("bias") != std::string::npos || name.find("norm") != std::string::npos)
        {
            LOG_TRACE("[WeightManager] NON-GEMM WEIGHT CLONE: " << name
                                                                << " original_ptr=" << static_cast<const void *>(original->raw_data())
                                                                << " clone_ptr=" << static_cast<const void *>(clone->raw_data())
                                                                << " target_device=" << target_device.to_string()
                                                                << " size_bytes=" << clone->size_bytes());
        }

        registerCloneMetadata(name, original, clone, target_device);
        return clone;
    }

    bool WeightManager::isGpuPreparationSource(
        const std::string &name, const TensorBase &tensor) const
    {
        /*
         * Expert tensors are deliberately excluded from isGemmWeight(): the
         * MoEExpertWeightService, rather than an ordinary ITensorGemm, owns
         * their device slabs. They still obey the same immutable-source rule.
         */
        const bool is_moe_expert =
            name.find("_exps.weight") != std::string::npos;
        if (inferWeightRole(name) == WeightRole::Embedding)
            return IINT8Unpackable::fromTensor(&tensor) != nullptr;
        return is_moe_expert || isGemmWeight(name);
    }

    std::shared_ptr<TensorBase> WeightManager::getWeightForDevice(
        const std::string &name,
        DeviceId device,
        int layer_idx)
    {
        const auto tp_config = tensorParallelConfig();
        if (inferWeightRole(name) == WeightRole::ImmutableRowTable)
            throw std::invalid_argument("[WeightManager] " + name +
                " requires its typed ImmutableRowTable store owner; ordinary weight materialization is forbidden");
        std::unique_lock<std::mutex> lock(cache_mutex_);

        // For LAYER_PARTITIONED strategy, filter out weights not in our layer range
        if (strategy_ == WeightDistributionStrategy::LAYER_PARTITIONED && has_layer_range_)
        {
            if (!isWeightInLayerRange(name))
            {
                LOG_DEBUG("[WeightManager] Layer filter: skipping " << name << " (not in range ["
                                                                    << layer_first_ << ", " << layer_last_ << "))");
                return nullptr;
            }
        }

        // =======================================================================
        // LOCAL TP support: Check TensorParallelConfig FIRST for device-specific slicing
        // =======================================================================
        // When tp_config is set, each device gets a DIFFERENT slice of the weights
        // based on its DeviceShardingAssignment. This applies regardless of the
        // distribution strategy (SHARDED, REPLICATED, or LAYER_PARTITIONED).
        //
        // For hybrid PP+TP mode:
        // - The PP stage context uses LAYER_PARTITIONED strategy for layer filtering
        // - But the nested MDO still needs TP column-parallel slicing within those layers
        // - So we check tp_config alone, not strategy_
        if (tp_config)
        {
            try
            {
                const auto assignment = tp_config->forDevice(device);
                lock.unlock();
                return getShardedWeightForAssignment(name, device, assignment, layer_idx);
            }
            catch (const std::out_of_range &)
            {
                // Device not in config - fall through to standard path
                LOG_TRACE("[WeightManager] Device " << device.to_string()
                                                    << " not in TensorParallelConfig, using standard path for: " << name);
            }
        }

        if (device.is_cpu()) {
            const auto key=DeviceWeightCacheKey::forDevice(device,name);
            const auto prior=per_device_cache_.find(key);
            const auto cached=prior==per_device_cache_.end()?std::shared_ptr<TensorBase>{}:prior->second;
            const auto original_store=prepared_weight_store_;
            lock.unlock();
            if(cached) {
                if(!hasOriginalCPUFinalOwner(cached,device,original_store))
                    throw std::logic_error("CPU final index lost its original completed Store owner");
                std::lock_guard verify(cache_mutex_);
                if(prepared_weight_store_!=original_store)
                    throw std::logic_error("CPU final index changed its original Store while authenticating");
                return cached;
            }
            std::shared_ptr<TensorBase> completed;
            switch(strategy_) {
            case WeightDistributionStrategy::REPLICATED:
            case WeightDistributionStrategy::LAYER_PARTITIONED:
                completed=getReplicatedWeight(name,device);break;
            case WeightDistributionStrategy::SHARDED:
                completed=getShardedWeight(name,device);break;
            case WeightDistributionStrategy::INTERLEAVED:
                throw std::logic_error("CPU interleaved storage has no declared canonical placement implementation");
            default:throw std::invalid_argument("Unknown original CPU weight distribution policy");
            }
            if(completed) {
                const auto completed_store=preparedWeightStoreIfInitialized();
                if(!hasOriginalCPUFinalOwner(completed,device,completed_store))
                    throw std::logic_error("CPU source getter has no original completed Store owner");
                std::lock_guard publish(cache_mutex_);
                if(prepared_weight_store_!=completed_store)
                    throw std::logic_error("CPU source publication changed its original completed Store");
                const auto [entry,inserted]=per_device_cache_.emplace(key,completed);
                if(!inserted && entry->second!=completed)
                    throw std::logic_error("CPU source publication replaced an original final handle");
                return entry->second;
            }
            return completed;
        }

        std::optional<std::vector<int>> expected_expert_ids;
        if (strategy_ == WeightDistributionStrategy::SHARDED &&
            getShardingMode(name) == ShardingMode::EXPERT_ID_APPORTIONED)
        {
            expected_expert_ids = expectedRoutedExpertIds(
                name,
                routed_expert_assignment_.participant_index,
                routed_expert_assignment_.participant_count,
                layer_idx);
        }

        // Track first device - original tensors stay on this device
        if (!first_device_.has_value())
        {
            first_device_ = device;
            LOG_DEBUG("[WeightManager] First device for weights: " << device.to_string());
        }

        // Helper lambda to load weight if not in cache (reuses getWeight logic without re-locking)
        auto ensureWeightLoaded = [this, &name, &device, layer_idx, &lock,
                                   &expected_expert_ids]() -> std::shared_ptr<TensorBase>
        {
            auto it = cache_.find(name);
            if (it != cache_.end())
            {
                if (expected_expert_ids)
                {
                    if (isCachedFullRoutedExpertSource(name, it->second))
                    {
                        /*
                         * This exact full source was admitted before TP policy
                         * became authoritative. Retire only the colliding cache
                         * binding; getShardedWeight() below materializes the
                         * requested typed ExpertSlice directly from the loader.
                         */
                        cache_.erase(it);
                    }
                    else
                    {
                        validateCachedRoutedExpertSlice(
                            name, it->second, *expected_expert_ids, name);
                        return it->second;
                    }
                }
                else
                {
                    return it->second;
                }
            }

            // Weight not in cache - load it now
            LOG_DEBUG("[WeightManager] getWeightForDevice: loading " << name << " (not in cache)");

            // Determine device from placement map if not explicitly provided
            DeviceId target_device = device;
            if (!target_device.is_valid() && placement_map_)
            {
                target_device = placement_map_->getDeviceForWeight(name, layer_idx);
            }
            if (!target_device.is_valid())
            {
                target_device = DeviceId::cpu();
            }

            // Load based on strategy (same logic as getWeight)
            std::shared_ptr<TensorBase> tensor;
            lock.unlock();
            switch (strategy_)
            {
            case WeightDistributionStrategy::REPLICATED:
            case WeightDistributionStrategy::LAYER_PARTITIONED:
                // LAYER_PARTITIONED uses same loading as REPLICATED, but getWeightForDevice()
                // filters which weights are allowed based on layer range
                tensor = getReplicatedWeight(name, target_device);
                break;
            case WeightDistributionStrategy::SHARDED:
                tensor = getShardedWeight(name, target_device);
                break;
            case WeightDistributionStrategy::INTERLEAVED:
                tensor = getInterleavedWeight(name, target_device);
                break;
            default:
                throw std::runtime_error("[WeightManager] Unknown strategy: " +
                                         std::to_string(static_cast<int>(strategy_)));
            }

            lock.lock();

            // Double-check cache after loading in case another thread won the race.
            auto cached_it = cache_.find(name);
            if (cached_it != cache_.end())
            {
                if (expected_expert_ids)
                {
                    if (isCachedFullRoutedExpertSource(name, cached_it->second))
                    {
                        cache_.erase(cached_it);
                    }
                    else
                    {
                        validateCachedRoutedExpertSlice(
                            name, cached_it->second, *expected_expert_ids, name);
                        return cached_it->second;
                    }
                }
                else
                {
                    return cached_it->second;
                }
            }

            if (tensor)
            {
                cache_[name] = tensor;
                LOG_DEBUG("[WeightManager] Cached tensor via getWeightForDevice: " << name);
            }
            return tensor;
        };

        /*
         * GPU GEMM, quantized embedding, and routed-expert preparation reads
         * immutable host bytes and publishes a separate device-owned packed
         * representation.  One source tensor is therefore authoritative for
         * every participant; cloning it per device only duplicates the model
         * in RAM and makes replicated-dense startup scale with device count.
         *
         * Keep this decision under cache_mutex_: concurrent runner creation
         * must observe the source as HOST_RESIDENT before any caller can
         * manufacture a device clone.  Raw runtime operands deliberately skip
         * this branch and continue into the device-qualified cache below.
         */
        if (device.is_gpu() && has_sharding_config_)
        {
            auto source = ensureWeightLoaded();
            if (source && isGpuPreparationSource(name, *source))
            {
                source->setHostResident();
                return source;
            }
        }

        // For the first device, return the original tensor
        if (first_device_.value() == device)
        {
            auto result = ensureWeightLoaded();
            // Log non-GEMM weights (biases, norms) for multi-GPU debugging
            if (result && (name.find("bias") != std::string::npos || name.find("norm") != std::string::npos))
            {
                LOG_DEBUG("[WeightManager] NON-GEMM WEIGHT (first device): " << name
                                                                             << " ptr=" << static_cast<const void *>(result->raw_data())
                                                                             << " device=" << device.to_string()
                                                                             << " size_bytes=" << result->size_bytes());
            }
            return result;
        }

        // Check per-device cache for subsequent devices
        const auto cache_key = DeviceWeightCacheKey::forDevice(device, name);
        auto it = per_device_cache_.find(cache_key);
        if (it != per_device_cache_.end())
        {
            // Log non-GEMM weights (biases, norms) for multi-GPU debugging
            if (name.find("bias") != std::string::npos || name.find("norm") != std::string::npos)
            {
                LOG_DEBUG("[WeightManager] NON-GEMM WEIGHT (cached clone): " << name
                                                                             << " ptr=" << static_cast<const void *>(it->second->raw_data())
                                                                             << " device=" << device.to_string()
                                                                             << " size_bytes=" << it->second->size_bytes());
            }
            return it->second;
        }

        // Need to create a clone for this device
        // First, ensure the original is loaded
        auto original = ensureWeightLoaded();
        if (!original)
        {
            // LOG_DEBUG, not LOG_ERROR: returning nullptr is valid for optional weights.
            // The caller decides severity based on schema context.
            LOG_DEBUG("[WeightManager] getWeightForDevice: weight not found: " << name);
            return nullptr;
        }

        auto clone = cloneTensorForDevice(name, original, device);
        if (!clone)
        {
            throw std::runtime_error("[WeightManager] getWeightForDevice: failed to clone " + name);
        }

        per_device_cache_[cache_key] = clone;
        return clone;
    }

    bool WeightManager::preloadForDevices(const std::vector<DeviceId> &devices)
    {
        if (devices.empty())
        {
            LOG_WARN("[WeightManager] preloadForDevices called with empty device list");
            return true;
        }

        LOG_DEBUG("[WeightManager] Pre-loading weights for " << devices.size() << " devices");

        std::vector<std::string> weight_names;
        bool seeded_from_loader = false;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // Set first device if not already set
            if (!first_device_.has_value())
            {
                first_device_ = devices[0];
                LOG_DEBUG("[WeightManager] First device set to: " << devices[0].to_string());
            }

            // Prefer already-cached names (fast path)
            weight_names.reserve(cache_.size());
            for (const auto &[name, _] : cache_)
            {
                weight_names.push_back(name);
            }
        }

        // If cache is empty, seed from model tensor list and let getWeightForDevice()
        // materialize/cache per-device tensors on demand.
        if (weight_names.empty())
        {
            weight_names = loader_.materializationNames();
            seeded_from_loader = true;

            if (weight_names.empty())
            {
                LOG_WARN("[WeightManager] No tensor names available for preloading");
                return true;
            }

            LOG_DEBUG("[WeightManager] Cache empty; seeding preload from model tensor list ("
                      << weight_names.size() << " tensors)");
        }

        // Add virtual weights that exist in the sharding config but not in
        // the GGUF file. Example: "output.weight" when tied to
        // "token_embd.weight" — the GGUF has no "output.weight" tensor, but
        // getShardedWeightForAssignment() knows how to synthesize one.
        // Without this, packGemmWeightsViaPipeline() never sees the LM head,
        // and the later buildWeights() call creates a different tensor object
        // that misses the prepared-GEMM registry.
        if (has_sharding_config_)
        {
            std::unordered_set<std::string> name_set(weight_names.begin(), weight_names.end());
            for (const auto &[name, mode] : sharding_config_.exact_matches)
            {
                if (name_set.find(name) == name_set.end())
                {
                    weight_names.push_back(name);
                    LOG_DEBUG("[WeightManager] Adding virtual weight to preload: " << name);
                }
            }
        }

        size_t loaded_tensors = 0;
        size_t total_uploads = 0;

        for (const auto &device : devices)
        {
            LOG_DEBUG("[WeightManager] Preloading weights for device " << device.to_string());

            for (const auto &name : weight_names)
            {
                // A lookup store retains native rows under its own admitted
                // lifetime. Ordinary preload must never upload this table.
                if (inferWeightRole(name) == WeightRole::ImmutableRowTable) continue;
                // Respect PP layer filtering semantics during seeded preload.
                if (strategy_ == WeightDistributionStrategy::LAYER_PARTITIONED && has_layer_range_)
                {
                    if (!isWeightInLayerRange(name))
                    {
                        continue;
                    }
                }

                auto tensor = getWeightForDevice(name, device);
                const bool is_gemm_weight = isGemmWeight(name);
                registerExpectedDeviceForWeight(name, device);
                if (!tensor)
                {
                    markPrepState(name, device, WeightPrepState::FAILED, is_gemm_weight, "getWeightForDevice failed during preload");
                    throw std::runtime_error(
                        "[WeightManager] Required weight '" + name +
                        "' could not be materialized for " + device.to_string());
                }

                markPrepState(name, device, WeightPrepState::LOADED_HOST, is_gemm_weight, "weight loaded for device");

                // On GPU, mark tensors HOST_RESIDENT when the TransferEngine
                // upload would produce a dead copy:
                //
                // - Quantized token_embd.weight: preparation owns unchanged native blocks.
                //   Native FP32/FP16/BF16 tables instead remain raw device reads.
                //
                // - GEMM weights (attn_q/k/v/output, ffn_gate/up/down, etc.):
                //   ROCmQuantisedGemmKernel uploads its own VNNI-repacked copy
                //   and never reads the raw Q8_0 gpu_data_ptr_. Without this,
                //   both the raw Q8_0 AND the VNNI copy sit on GPU — ~1.9 GB
                //   wasted per device.
                //
                // - MoE 3D expert tensors (_exps.weight): MoEExpertWeightService
                //   reads host views and batch-packs into VNNI slabs with its own
                //   GPU upload.  Uploading the raw 3D tensor here wastes ~8 GB/device
                //   and fragments VRAM, making subsequent hipMalloc calls very slow.
                if (device.is_gpu() && isGpuPreparationSource(name, *tensor))
                {
                    tensor->setHostResident();
                }

                ++loaded_tensors;

                if (device.type != DeviceType::CPU)
                {
                    const auto upload =
                        TransferEngine::instance().upload(tensor.get(), device);
                    if (upload.success)
                    {
                        markPrepState(name, device, WeightPrepState::UPLOADED_DEVICE, is_gemm_weight, "preload TransferEngine upload completed");
                        if (!is_gemm_weight)
                        {
                            markPrepState(name, device, WeightPrepState::READY, false, "preload non-GEMM ready");
                        }
                        ++total_uploads;
                    }
                    else
                    {
                        markPrepState(name, device, WeightPrepState::FAILED, is_gemm_weight, "preload TransferEngine upload failed");
                        throw std::runtime_error(
                            "[WeightManager] Required weight '" + name +
                            "' failed upload to " + device.to_string() +
                            ": " + upload.error);
                    }
                }
            }
        }

        LOG_DEBUG("[WeightManager] Preload complete: loaded=" << loaded_tensors
                                                              << ", uploads=" << total_uploads
                                                              << (seeded_from_loader ? " (seeded from loader names)" : ""));
        return true;
    }

    // =========================================================================
    // Weight Preparation (repack + upload, NO host release)
    // =========================================================================

    bool WeightManager::prepareWeightsForDevice(DeviceId device)
    {
        return prepareWeightsForDeviceImpl(
            device,
            nullptr,
            nullptr,
            true,
            PhysicalMemoryOwner::PrimaryModelWeights);
    }

    bool WeightManager::prepareWeightsForDevice(
        const FrozenModelWeightSet &frozen_weights,
        DeviceId device,
        bool include_expert_jobs)
    {
        validateStaticProjectionPreparation(frozen_weights, device, include_expert_jobs);
        const PhysicalMemoryOwner persistent_owner =
            frozen_weights.physicalMemoryOwner();
        LOG_DEBUG("[WeightManager] prepareWeightsForDevice(" << device.to_string()
                                                             << ", frozen_bindings=" << frozen_weights.bindings().size()
                                                             << ", memory_owner=" << toString(persistent_owner) << ")");
        if (!lifecycle_gates_.materialization_complete)
            markMaterializationComplete();

        if (device.is_cpu())
        {
            auto store = preparedWeightStore();
            int registered = 0;
            std::vector<std::pair<ExpertGemmRegistry::ParticipantLayerScope,std::vector<WeightBinding>>> joint_layers;
            for(const auto& binding:frozen_weights.bindings()) {
                if(!binding.joint_source || !bindingTargetsDevice(binding,device))continue;
                const auto scope=ExpertGemmRegistry::staticProjectionScope(binding);
                auto found=std::find_if(joint_layers.begin(),joint_layers.end(),[&](const auto& layer){return layer.first==scope;});
                if(found==joint_layers.end()){joint_layers.push_back({scope,{}});found=joint_layers.end()-1;}
                found->second.push_back(binding);
            }
            for(const auto& [scope,sources]:joint_layers) {
                const auto& projection=*sources.front().projection_ownership;
                if(expert_gemm_registry_.acquireStaticProjectionLayer(scope,projection,sources))continue;
                // The dense pass may validate a committed native bank, but only
                // explicit expert preparation may publish a missing bank.
                if (!include_expert_jobs)
                    throw std::logic_error("CPU dense preparation requires the complete original joint fragment bank");
                // The registry validates the complete selected declaration before
                // packing; each packed allocation retains the original PMA/NUMA.
                for(const auto& binding:sources) {
                    auto requested=binding;
                    if(!store->adoptPreparedForBinding(requested,device))store->prepareGemm(binding,persistent_owner);
                    const auto ref=store->preparedRefForBinding(binding.binding_id,device,PreparedWeightKind::CpuPackedGemm);
                    if(!ref)throw std::logic_error("CPU joint prepared store omitted its original matrix");
                    auto engine=store->gemmKernelLifetime(*ref);
                    if(!engine)throw std::logic_error("CPU joint prepared store has no owned engine");
                    if (binding.native_source) expert_gemm_registry_.registerCommittedCPUProjection(binding,*store);
                    else {
                        auto* raw_engine=engine.get();
                        expert_gemm_registry_.registerStaticProjection(binding,raw_engine,std::move(engine));
                    }
                    ++registered;
                }
                if(!expert_gemm_registry_.acquireStaticProjectionLayer(scope,projection,sources))
                    throw std::logic_error("CPU joint preparation failed to publish its complete selected layer");
            }

            for (const auto &binding : frozen_weights.bindings())
            {
                if(binding.joint_source)continue;
                if (!shouldPrepareFrozenGemmBinding(*this, binding, device, include_expert_jobs))
                    continue;
                const PreparedWeightKind expected_kind =
                    binding.prepared.has_value()
                        ? binding.prepared->kind
                        : PreparedWeightKind::CpuPackedGemm;
                if (!store->preparedRefForBinding(
                         binding.binding_id,
                         device,
                         expected_kind)
                         .has_value())
                {
                    store->prepareGemm(binding, persistent_owner);
                    ++registered;
                }
            }

            if (!lifecycle_gates_.device_preparation_complete)
                markDevicePreparationComplete();
            LOG_DEBUG("[WeightManager] Binding-driven CPU preparation registered "
                      << registered << " GEMM bindings for " << device.to_string());

            // Every transient native/conversion window already retired at the
            // common consumer boundary. Remaining readable CPU bytes are actual
            // final tensor owners (possibly shared by tied embedding/head uses).
            // A representation-wide raw-data sweep would destroy those owners.
            return true;
        }

        const bool ok = prepareWeightsForDeviceImpl(
            device,
            nullptr,
            &frozen_weights,
            include_expert_jobs,
            persistent_owner);
        if (ok && !lifecycle_gates_.device_preparation_complete)
            markDevicePreparationComplete();
        return ok;
    }

    bool WeightManager::prepareWeightsForDevice(
        DeviceId device,
        int first_layer, int last_layer,
        bool has_embedding, bool has_lm_head)
    {
        auto layer_filter = [this, first_layer, last_layer, has_embedding, has_lm_head](
                                const std::string &name) -> bool
        {
            return isWeightInLayerRange(name, first_layer, last_layer, has_embedding, has_lm_head);
        };

        LOG_DEBUG("[WeightManager] prepareWeightsForDevice(" << device.to_string()
                                                             << " layers=[" << first_layer << ", " << last_layer << ")"
                                                             << " embed=" << has_embedding << " lm_head=" << has_lm_head << ")");

        return prepareWeightsForDeviceImpl(
            device,
            layer_filter,
            nullptr,
            true,
            PhysicalMemoryOwner::PrimaryModelWeights);
    }

    bool WeightManager::prepareWeightsForDeviceImpl(
        DeviceId device,
        std::function<bool(const std::string &)> layer_filter,
        const FrozenModelWeightSet *frozen_weights,
        bool include_expert_jobs,
        PhysicalMemoryOwner persistent_owner)
    {
        const bool is_gpu = device.is_gpu();
        const char *device_name = device.is_rocm() ? "ROCm" : (device.is_cuda() ? "CUDA" : "CPU");

        // Step 1: Pack GEMM weights through the participant-local pipeline.
        // GPU devices retain pipelined H2D and repack stream concurrency; sibling
        // participants also prepare independently. Non-GEMM upload must wait for
        // this pipeline to release its admitted staging ring before reusing it.
        // CPU devices use the CPU-side VNNI packing path.
        std::future<bool> gemm_future;
        if (is_gpu)
        {
            LOG_DEBUG("[WeightManager] Using GPU weight loading pipeline for " << device_name);
            // GPU loading is independent control work. Only its future child
            // inherits the admitted partition; future joins, non-GEMM upload
            // and every CPU packing path retain original caller placement.
            const ScopedOpenMPPartitionAffinity control_affinity;
            gemm_future = std::async(
                std::launch::async,
                [this, device, &layer_filter, frozen_weights,
                 include_expert_jobs, persistent_owner]()
                {
                    return packGemmWeightsViaPipeline(
                        device,
                        layer_filter,
                        frozen_weights,
                        include_expert_jobs,
                        nullptr,
                        std::nullopt,
                        persistent_owner);
                });
        }

        bool gemm_ok = true;
        if (is_gpu)
        {
            gemm_ok = gemm_future.get();
            if (!gemm_ok)
            {
                LOG_ERROR("[WeightManager] GEMM weight packing failed for " << device_name);
                return false;
            }
        }

        // Step 2: Reuse the finalized pipeline's staging allowance for non-GEMM
        // upload. GPU packing retains raw source views: tied head/embedding
        // bytes stay live until the later, lifecycle-gated host release sweep.
        bool non_gemm_ok = true;
        if (is_gpu)
        {
            non_gemm_ok = frozen_weights
                              ? uploadFrozenNonGemmWeights(
                                    device,
                                    *frozen_weights,
                                    layer_filter,
                                    include_expert_jobs,
                                    persistent_owner)
                              : uploadNonGemmWeights(device, layer_filter);
            if (!non_gemm_ok)
            {
                LOG_ERROR("[WeightManager] Non-GEMM weight upload failed for " << device_name);
            }
        }

        // Step 3: CPU GEMM packing has no GPU upload staging to share.
        if (!is_gpu)
        {
            // Wire progress tracker for CPU weight packing (count-based progress).
            // We use weight count as the progress unit since CPU packing is compute-bound.
            PreloadProgressCallback cpu_progress_cb;
            int cpu_progress_idx = -1;
            if (weight_load_progress_)
            {
                // Estimate total bytes from cache (GEMM weights only)
                size_t estimated_bytes = 0;
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    for (const auto &[name, tensor] : cache_)
                    {
                        if (isGemmWeight(name) && tensor)
                        {
                            if (layer_filter && !layer_filter(name))
                                continue;
                            estimated_bytes += tensor->size_bytes();
                        }
                    }
                }
                if (estimated_bytes > 0)
                {
                    cpu_progress_idx = weight_load_progress_->registerDevice(
                        weight_load_progress_->makeDeviceLabel(device.to_string()), estimated_bytes);
                    cpu_progress_cb = [this, cpu_progress_idx, estimated_bytes](size_t current, size_t total, const std::string &)
                    {
                        // Map count progress to byte progress
                        const size_t approx_bytes = (current * estimated_bytes) / total;
                        weight_load_progress_->update(cpu_progress_idx, approx_bytes);
                    };
                }
            }

            gemm_ok = packGemmWeights(device, cpu_progress_cb, /*release_raw_data=*/true, layer_filter);

            if (weight_load_progress_ && cpu_progress_idx >= 0)
            {
                weight_load_progress_->finish(cpu_progress_idx);
            }

            // Return freed memory to the OS after releasing raw quantized weight data.
            // VNNI packing creates its own packed buffer; the original Q4_K/Q5_K/Q6_K/etc
            // heap data (from TensorSlice sharding) is now released. Without
            // malloc_trim, glibc retains these freed pages in its arena indefinitely.
#ifdef __linux__
            if (gemm_ok)
            {
                ::malloc_trim(0);
            }
#endif
        }

        if (!gemm_ok)
        {
            LOG_WARN("[WeightManager] GEMM weight packing failed for " << device_name
                                                                       << "; required prepared kernels may be unavailable");
        }

        return gemm_ok && non_gemm_ok;
    }

    // =========================================================================
    // Weight Lifecycle (convenience entry points)
    // =========================================================================

    bool WeightManager::finalizeForDevice(DeviceId device)
    {
        const bool is_gpu = device.is_gpu();
        const char *device_name = device.is_rocm() ? "ROCm" : (device.is_cuda() ? "CUDA" : "CPU");

        // Phase 9: Mark materialization complete before preparation
        if (!lifecycle_gates_.materialization_complete)
        {
            markMaterializationComplete();
        }

        // Phase 1: Prepare (pack + upload, no release)
        bool ok = prepareWeightsForDevice(device);

        // Phase 9: Mark device preparation complete after packing/upload
        if (ok && !lifecycle_gates_.device_preparation_complete)
        {
            markDevicePreparationComplete();
        }

        // Phase 2: Release host copies (GPU only — CPU quantized weights are
        // released inline during packGemmWeights via release_raw_data=true).
        if (is_gpu && ok)
        {
            size_t released = releaseAllHostWeightData();
            LOG_DEBUG("[WeightManager] finalizeForDevice(" << device_name
                                                           << "): released " << released << " host tensors");
        }

        return ok;
    }

    bool WeightManager::finalizeForDevices(const std::vector<DeviceId> &devices,
                                           bool release_host_data,
                                           bool include_expert_jobs)
    {
        if (devices.empty())
        {
            LOG_WARN("[WeightManager] finalizeForDevices called with empty device list");
            return true;
        }

        // Step 1: Clone and upload all weights to all devices
        LOG_DEBUG("[WeightManager] finalizeForDevices: pre-loading weights for "
                  << devices.size() << " devices");
        if (!preloadForDevices(devices))
        {
            LOG_ERROR("[WeightManager] finalizeForDevices: preloadForDevices failed");
            return false;
        }

        // Step 2: Pack GEMM weights per device.
        // GPU devices use the GPU-native pipeline (LoadOrchestrator) with
        // pipelined H2D + GPU repack kernels. CPU devices use CPU-side packing.
        // Each GPU gets its own LoadOrchestrator, VRAM pool, and pinned ring
        // buffer — fully independent, safe to parallelize.
        bool all_ok = true;

        const auto gemm_wall_start = std::chrono::high_resolution_clock::now();

        const auto &gpu_load_cfg = debugEnv().rocm;
        const std::optional<size_t> per_gpu_staging_budget =
            gpuPerDeviceLoadStagingBudgetBytes(gpu_load_cfg.repack_budget_mb);

        {
            std::vector<std::future<bool>> gemm_futures;
            std::vector<std::chrono::high_resolution_clock::time_point> per_device_start(devices.size());
            gemm_futures.reserve(devices.size());

            for (size_t i = 0; i < devices.size(); ++i)
            {
                const auto &dev = devices[i];
                per_device_start[i] = std::chrono::high_resolution_clock::now();

                if (dev.is_gpu())
                {
                    LOG_DEBUG("[WeightManager] finalizeForDevices: launching GPU pipeline for "
                              << dev.toString());
                    gemm_futures.push_back(std::async(std::launch::async,
                                                      [this, dev, include_expert_jobs, per_gpu_staging_budget]()
                                                      {
                                                          return packGemmWeightsViaPipeline(
                                                              dev, nullptr, nullptr, include_expert_jobs,
                                                              nullptr, per_gpu_staging_budget);
                                                      }));
                }
                else
                {
                    gemm_futures.push_back(std::async(std::launch::async,
                                                      [this, dev]()
                                                      { return packGemmWeights(dev, nullptr, /*release_raw_data=*/false); }));
                }
            }

            for (size_t i = 0; i < devices.size(); ++i)
            {
                const bool ok = gemm_futures[i].get();
                const auto pack_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::high_resolution_clock::now() - per_device_start[i])
                                         .count();
                if (!ok)
                {
                    LOG_WARN("[WeightManager] GEMM packing failed for device " << devices[i].toString());
                    all_ok = false;
                }
                const char *mode = devices[i].is_gpu() ? "GPU pipeline" : "CPU repack";
                LOG_DEBUG("[WeightManager] packGemmWeights for " << devices[i].toString()
                                                                 << " completed in " << pack_ms << " ms (" << mode << ")");
            }
        }

        const auto gemm_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::high_resolution_clock::now() - gemm_wall_start)
                                      .count();
        LOG_DEBUG("[WeightManager] All " << devices.size() << " devices GEMM packed in "
                                         << gemm_wall_ms << " ms wall-clock"
                                         << (devices.size() > 1 ? " (parallel)" : ""));

        // Step 3: Release all host weight data (unless caller opts out — e.g.
        // nested TP-in-PP, where a later PP stage on a different device still
        // needs host copies to clone from).
        if (release_host_data)
        {
            size_t released = releaseAllHostWeightData();
            LOG_DEBUG("[WeightManager] finalizeForDevices: released " << released
                                                                      << " host tensors across " << devices.size() << " devices");
        }
        else
        {
            LOG_DEBUG("[WeightManager] finalizeForDevices: retaining host weight data "
                      "(nested TP-in-PP; outer caller will release)");
        }

        // Step 4: Return freed memory to the OS.
        // glibc malloc keeps freed blocks in its arena and only returns memory
        // above the brk/mmap threshold.  After releasing hundreds of weight
        // tensors (several GB), the arena is heavily fragmented.  malloc_trim
        // forces glibc to release free pages back to the kernel via madvise.
#ifdef __linux__
        if (release_host_data)
        {
            auto report_rss = [](const char *label)
            {
                std::ifstream status("/proc/self/status");
                std::string line;
                while (std::getline(status, line))
                {
                    if (line.compare(0, 6, "VmRSS:") == 0 ||
                        line.compare(0, 8, "RssAnon:") == 0)
                    {
                        LOG_DEBUG("[WeightManager] " << label << " " << line);
                    }
                }
            };
            report_rss("Pre-trim");
            ::malloc_trim(0);
            report_rss("Post-trim");
        }
#endif

        return all_ok;
    }

} // namespace llaminar2
