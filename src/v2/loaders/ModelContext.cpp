/**
 * @file ModelContext.cpp
 * @brief Model-owned immutable sources and exactly-once participant preparation.
 *
 * A native table, fixed shard or GPU/subordinate adaptive domain is prepared
 * only through its admitted placement and original PMA. Setup joins occur
 * before inference; no model mutex spans file I/O or native metadata submission.
 * Graph families retain the original source/boundary and alias-owned roots,
 * preserving actual leases after the context releases its own references.
 * @author David Sanftenberg
 */

#include "ModelContext.h"
#include "ImmutableLookupSetupObservation.h"
#include "models/GraphTypes.h"
#include "execution/qwen4exp/PooledImmutableRetainedParticipants.h"
#include "backends/IWorkerGPUContext.h"
#include "transfer/TransferEngine.h"
#include "AdaptiveImmutableLookupGeometry.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/PooledImmutableNativeKernels.h"
#include <array>
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "../utils/Logger.h"
#include <exception>
#include <algorithm>
#include <iostream>
#include <limits>

namespace llaminar2
{

    /**
     * @brief Internal wrapper to expose ModelLoader (by reference) as IModelLoader shared_ptr
     *
     * This wrapper allows ModelContext to return IModelLoader interface while the actual
     * ModelLoader is owned by value within ModelContext. The wrapper holds a reference
     * and relies on the ModelContext outliving any use of the interface pointer.
     */
    class ModelLoaderInterfaceWrapper : public IModelLoader
    {
    public:
        explicit ModelLoaderInterfaceWrapper(ModelLoader &loader) : loader_(loader) {}

        /** @copydoc IModelLoader::nativeWeightSourcePlacement */
        CPUStoragePlacement nativeWeightSourcePlacement() const override
        {
            return loader_.nativeWeightSourcePlacement();
        }

        /** @copydoc IModelLoader::nativeWeightSourceGeometry */
        NativeWeightSourceGeometry nativeWeightSourceGeometry(const std::string &name) override
        {
            return loader_.nativeWeightSourceGeometry(name);
        }

        /** @copydoc IModelLoader::readNativeWeightSourceWindow */
        void readNativeWeightSourceWindow(
            const std::string &name, const NativeWeightSourceWindow &window,
            std::span<std::byte> destination) override
        {
            loader_.readNativeWeightSourceWindow(name, window, destination);
        }

        bool isLoaded() const override { return loader_.isLoaded(); }
        /** @brief Expose the immutable directory owned by the enclosing model context. */
        const GGUFModel *ggufDirectory() const noexcept override { return &loader_.getModel(); }
        /** @brief Keep predictor metadata and policy separate from the unchanged target directory. */
        const GGUFModel *ggufPredictorDirectory() const noexcept override { return loader_.ggufPredictorDirectory(); }
        /** @brief Borrow the loader-owned immutable source manifest. */
        const Qwen4ExpSidecarManifest *predictorManifest() const noexcept override { return loader_.predictorManifest(); }

        std::shared_ptr<TensorBase> loadTensor(
            const std::string &name,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensor(name, device, weight_precision);
        }

        std::shared_ptr<TensorBase> loadTensorRowSlice(
            const std::string &name,
            size_t row_start, size_t row_end,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensorRowSlice(name, row_start, row_end, device, weight_precision);
        }

        std::shared_ptr<TensorBase> loadTensorColumnSlice(
            const std::string &name,
            size_t col_start, size_t col_end,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensorColumnSlice(name, col_start, col_end, device, weight_precision);
        }

        std::shared_ptr<TensorBase> loadTensorExpertSlice(
            const std::string &name,
            size_t expert_start, size_t expert_end,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensorExpertSlice(name, expert_start, expert_end, device, weight_precision);
        }

        std::shared_ptr<TensorBase> loadTensorExpertSelection(
            const std::string &name,
            const std::vector<size_t> &expert_ids,
            DeviceId device,
            WeightPrecision weight_precision) override
        {
            return loader_.loadTensorExpertSelection(
                name, expert_ids, device, weight_precision);
        }

        bool hasTensor(const std::string &name) const override { return loader_.hasTensor(name); }
        std::vector<std::string> tensorNames() const override { return loader_.tensorNames(); }
        /** @brief Project the same consumed source inventory into WeightManager preload. */
        std::vector<std::string> materializationNames() const override { return loader_.materializationNames(); }
        /** @brief Forward immutable source sealing to the context-owned loader. */
        void sealTensorSources() override { loader_.sealTensorSources(); }
        std::string architecture() const override { return loader_.architecture(); }
        size_t tensorCount() const override { return loader_.tensorCount(); }
        size_t totalBytes() const override { return loader_.totalBytes(); }

        int getInt(const std::string &key, int default_val) const override
        {
            return loader_.getInt(key, default_val);
        }
        uint64_t getUInt64(const std::string &key, uint64_t default_val) const override
        {
            return loader_.getUInt64(key, default_val);
        }
        float getFloat(const std::string &key, float default_val) const override
        {
            return loader_.getFloat(key, default_val);
        }
        std::string getString(const std::string &key, const std::string &default_val) const override
        {
            return loader_.getString(key, default_val);
        }

        uint64_t blockCount() const override { return loader_.blockCount(); }
        uint64_t embeddingLength() const override { return loader_.embeddingLength(); }
        uint64_t headCount() const override { return loader_.headCount(); }
        uint64_t headCountKV() const override { return loader_.headCountKV(); }
        uint64_t vocabSize() const override { return loader_.vocabSize(); }
        uint64_t contextLength() const override { return loader_.contextLength(); }
        uint64_t feedForwardLength() const override { return loader_.feedForwardLength(); }
        uint64_t keyLength() const override { return loader_.keyLength(); }
        float ropeTheta() const override { return loader_.ropeTheta(); }
        float rmsNormEps() const override { return loader_.rmsNormEps(); }

        std::optional<std::vector<size_t>> getTensorShape(const std::string &name) const override
        {
            return loader_.getTensorShape(name);
        }

    private:
        ModelLoader &loader_;
    };

    std::shared_ptr<HostImmutableRowProvider> ModelContext::hostImmutableRowProvider() const
    {
        std::lock_guard lock(immutable_row_table_mutex_);
        if (immutable_row_table_preparation_failure_)
            std::rethrow_exception(immutable_row_table_preparation_failure_);
        return host_row_provider_;
    }

    std::shared_ptr<const ResolvedImmutableRowTablePlacement> ModelContext::resolvedImmutableRowTablePlacement() const
    {
        std::lock_guard lock(immutable_row_table_mutex_);
        return immutable_row_table_placement_;
    }

    std::shared_ptr<const ResidentImmutableRowTable> ModelContext::residentImmutableRowTable(DeviceId device) const
    {
        std::lock_guard lock(immutable_row_table_mutex_);
        if (immutable_row_table_preparation_failure_)
            std::rethrow_exception(immutable_row_table_preparation_failure_);
        const auto found = resident_row_tables_.find(device);
        return found == resident_row_tables_.end() ? nullptr : found->second;
    }

    std::shared_ptr<const ImmutableLookupSetupObservation> ModelContext::immutableLookupSetupObservation()const
    {
        std::lock_guard lock(immutable_row_table_mutex_);
        return immutable_lookup_setup_observation_;
    }

    std::size_t ModelContext::residentImmutableRowTableBytes(DeviceId device) const
    {
        const auto table = residentImmutableRowTable(device);
        return table ? table->nativeBytes() : 0;
    }

    ImmutableRowTableSource ModelContext::immutableRowTableSource() const
    {
        const auto &directory = loader_.getModel();
        const auto *table = directory.findTensor("per_layer_token_embd.weight");
        if (!table || table->dimensions.size() != 2)
            throw std::invalid_argument("native table requires its original canonical [rows,columns] tensor");
        const auto split = static_cast<std::size_t>(table->split_idx);
        const auto paths = artifactPaths();
        if (split >= paths.size())
            throw std::invalid_argument("native table source shard is absent from immutable artifact identity");
        const auto data_offset = directory.split_count > 1
            ? directory.split_data_offsets.at(split) : directory.data_offset;
        if (table->offset > std::numeric_limits<std::uint64_t>::max() - data_offset)
            throw std::overflow_error("native table source file offset overflow");
        ImmutableRowTableSource source{.path=paths.at(split),.tensor_name=table->name,
            .file_offset=data_offset+table->offset,.rows=table->dimensions[0],
            .row_elements=table->dimensions[1],.format=table->type};
        if (source.tensorBytes() != table->size_bytes)
            throw std::invalid_argument("native table extent differs from its original tensor directory");
        return source;
    }

    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> ModelContext::adaptiveImmutableLookupBoundary(DeviceId device) const
    {
        std::lock_guard lock(immutable_row_table_mutex_);
        if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
        const auto found=adaptive_lookup_consumers_.find(device);
        if(found==adaptive_lookup_consumers_.end())
        {
            if(!adaptive_lookup_consumers_.empty())
                throw std::invalid_argument("Adaptive lookup boundary queried by a foreign GPU participant");
            return {};
        }
        return found->second.boundary;
    }

    std::optional<ImmutableLookupNativeOwnerObservation>
    ModelContext::retainedAdaptiveImmutableLookupOwner(PhysicalMemoryAllocatorIdentity allocator)const
    {
        std::lock_guard lock(immutable_row_table_mutex_);
        if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
        if(!immutable_row_table_placement_ || !immutable_row_table_placement_->isGPUAdaptive())return std::nullopt;
        if(!immutable_row_table_authority_ || allocator.world_rank!=immutable_row_table_authority_->worldRank() ||
           !allocator.device.is_gpu())
            throw std::invalid_argument("Adaptive retained roots require the original rank and GPU allocator");
        const auto& placement=*immutable_row_table_placement_;
        const auto& layout=placement.gpuAdaptiveLayout();
        const bool consumer=std::find(placement.consumers().begin(),placement.consumers().end(),allocator)!=placement.consumers().end();
        const bool storage_owner=std::any_of(layout.owners().begin(),layout.owners().end(),[&](const auto& owner){
            return owner.key.tier==ImmutableRowStorage::VRAM && owner.key.allocator==allocator;
        });
        // A prepared expert follower may have no ngram role at all.
        if(!consumer && !storage_owner)return std::nullopt;
        const bool source_owner=layout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal
            ? consumer : AdaptiveImmutableLookupGeometry::sourceLayout(layout,placement.gpuAdaptiveRequest()).consumers().front()==allocator;
        std::optional<ImmutableLookupNativeOwnerObservation> result;
        std::vector<const qwen4exp::PooledImmutableSourcePreparedStorage*> seen;
        for(const auto& [device,entry]:adaptive_lookup_consumers_)
        {
            // Query-only execution entries may already have retired. Only real
            // source boundaries retain the model's original physical roots.
            if(!entry.boundary)continue;
            const auto source=entry.boundary->pooledStorage();
            if(!source)throw std::logic_error("Adaptive retained source lost its original pooled roots");
            if(std::find(seen.begin(),seen.end(),source.get())!=seen.end())continue;
            seen.push_back(source.get());
            const auto observed=source->geometry().projectRetainedOwnerObservation(allocator,
                [&]{return source->retainedOwnerObservation();},
                [&](std::size_t index){return source->originalOwners()[index]->retainedOwnerObservation();});
            if(!observed)continue;
            if((source->geometry().allocator()==allocator)!=source_owner || (!source_owner && !storage_owner))
                throw std::logic_error("Adaptive retained observation changed its declared source/storage role");
            const auto& authority=observed->authority();
            if(authority.get()!=immutable_row_table_authority_.get() ||
               authority.owner_before(immutable_row_table_authority_) || immutable_row_table_authority_.owner_before(authority) ||
               observed->allocator()!=allocator)
                throw std::logic_error("Adaptive retained root observation changed its original allocator/PMA");
            if(result)throw std::logic_error("Adaptive retained roots contain overlapping source/owner observations");
            result=observed;
        }
        if(!result && (source_owner || storage_owner))
            throw std::logic_error("Adaptive retained allocator lost its original live source/storage roots");
        return result;
    }

    void ModelContext::prepareAdaptiveImmutableLookupParticipants(const GraphConfig& graph)
    {
        std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
        {
            std::unique_lock lock(immutable_row_table_mutex_);
            const auto found=adaptive_lookup_consumers_.find(graph.default_device);
            if(found==adaptive_lookup_consumers_.end())
                throw std::logic_error("Remote participant preparation has no original ready source entry");
            if(!found->second.boundary)
            {
                auto retained=found->second.participants;
                if(!retained)throw std::logic_error("Query-only reader has no original retained source participants");
                lock.unlock();
                retained->requireForPolicy(graph);
                return;
            }
            auto& consumer=found->second;
            while(consumer.participant_preparation &&
                  consumer.participant_preparation->state==ImmutableRowPreparationState::Preparing)
            {
                if(consumer.participant_preparation->preparing_thread==std::this_thread::get_id())
                    throw std::logic_error("Recursive original participant preparation");
                immutable_row_table_ready_.wait(lock);
            }
            if(consumer.participant_failure)std::rethrow_exception(consumer.participant_failure);
            boundary=consumer.boundary;
            if(!boundary->isPooled() || boundary->device()!=graph.default_device)
                throw std::logic_error("Remote participant preparation changed its original source boundary");
            const auto storage=boundary->pooledStorage();
            const bool remote=std::any_of(storage->originalOwners().begin(),storage->originalOwners().end(),[&](const auto& owner){
                return owner->plan().allocator()!=storage->coordinator()->plan().allocator();
            });
            if(!remote)return;
            if(consumer.participants){
                auto retained=consumer.participants;lock.unlock();retained->requireForPolicy(graph);return;
            }
            consumer.participant_preparation=ImmutableRowPreparation{
                ImmutableRowPreparationState::Preparing,std::this_thread::get_id()};
        }
        try{
            // Existing remote topology guards remain authoritative. A local
            // consumer set never substitutes replicas for a shared-owner request.
            const auto& placement=*graph.ngram_table_placement;
            const auto geometry=placement.gpuAdaptiveLayout().adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal
                ? AdaptiveImmutableLookupGeometry::forRequester(placement.gpuAdaptiveLayout(),*immutable_row_table_source_,
                    placement.gpuAdaptiveRequest(),{immutable_row_table_authority_->worldRank(),graph.default_device})
                : AdaptiveImmutableLookupGeometry::fromLayout(placement.gpuAdaptiveLayout(),
                    *immutable_row_table_source_,placement.gpuAdaptiveRequest());
            auto prepared=qwen4exp::PooledImmutableRetainedParticipants::create(boundary,graph,
                immutable_source_mpi_ ? immutable_source_mpi_->clusterInventory() : nullptr,geometry.readerGeometry());
            {std::lock_guard lock(immutable_row_table_mutex_);
             auto& consumer=adaptive_lookup_consumers_.at(graph.default_device);
             consumer.participants=std::move(prepared);
             consumer.participant_preparation->state=ImmutableRowPreparationState::Ready;}
            immutable_row_table_ready_.notify_all();
        }catch(...){
            {std::lock_guard lock(immutable_row_table_mutex_);
             auto& consumer=adaptive_lookup_consumers_.at(graph.default_device);
             consumer.participant_failure=std::current_exception();
             consumer.participant_preparation->state=ImmutableRowPreparationState::Failed;}
            immutable_row_table_ready_.notify_all();throw;
        }
    }
    std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> ModelContext::adaptiveImmutableLookupParticipants(DeviceId device)const
    {
        std::lock_guard lock(immutable_row_table_mutex_);
        const auto found=adaptive_lookup_consumers_.find(device);
        if(found==adaptive_lookup_consumers_.end())
        {
            if(!adaptive_lookup_consumers_.empty())
                throw std::invalid_argument("Remote participants queried by a foreign source consumer");
            return {};
        }
        if(found->second.participant_failure)std::rethrow_exception(found->second.participant_failure);
        return found->second.participants;
    }

    void ModelContext::bindNativeContextPreparation(std::shared_ptr<NativeContextPreparation> source)
    {
        if(!source)throw std::invalid_argument("Model source preparation cannot be null");
        std::lock_guard lock(immutable_row_table_mutex_);
        if(native_context_preparation_ && native_context_preparation_!=source)
            throw std::logic_error("Model cannot replace its original native source preparation");
        if(immutable_row_table_placement_ || !immutable_row_table_preparations_.empty())
            throw std::logic_error("Model source preparation must precede table materialization");
        native_context_preparation_=std::move(source);
    }
    std::shared_ptr<NativeContextPreparation> ModelContext::nativeContextPreparation() const
    {
        std::lock_guard lock(immutable_row_table_mutex_);return native_context_preparation_;
    }

    void ModelContext::bindRetainedHostServices(std::shared_ptr<const RetainedHostServiceDeclarations> declarations,
        std::shared_ptr<PhysicalMemoryAuthority> memory,std::span<const RetainedHostServiceInstance> selected)
    {
        if(!declarations || !memory || declarations->worldRank()!=memory->worldRank())throw std::logic_error("Model CPU service binding lost original rank/PMA declarations");
        std::lock_guard lock(host_service_mutex_);
        if(host_service_binding_){
            if(host_service_declarations_!=declarations || host_service_memory_.get()!=memory.get() ||
               host_service_memory_.owner_before(memory)||memory.owner_before(host_service_memory_) ||
               host_service_selected_!=std::vector<RetainedHostServiceInstance>(selected.begin(),selected.end()))
                throw std::logic_error("Model CPU service binding cannot reprice/rebind original instances");
            return;
        }
        auto binding=declarations->bind(memory,selected);
        host_service_selected_.assign(selected.begin(),selected.end());
        host_service_declarations_=std::move(declarations);host_service_memory_=std::move(memory);host_service_binding_=std::move(binding);
    }
    RetainedHostServiceUseToken ModelContext::takeRetainedHostService(PhysicalMemoryAllocatorIdentity consumer,
        RetainedHostServiceRole role,const std::shared_ptr<PhysicalMemoryAuthority>& memory)
    {
        const auto binding=[&]{
            std::lock_guard lock(host_service_mutex_);
            if(!host_service_binding_||host_service_memory_.get()!=memory.get()||host_service_memory_.owner_before(memory)||memory.owner_before(host_service_memory_))
                throw std::logic_error("Model CPU service consumer lost original admitted binding/PMA control block");
            return host_service_binding_;
        }();
        return binding->take({consumer.world_rank,consumer.device,role});
    }

    void ModelContext::prepareGPUAdaptiveImmutableLookup(const ResolvedImmutableRowTablePlacement &placement,
        std::shared_ptr<PhysicalMemoryAuthority> authority,const ImmutableRowTableSource &source,DeviceId consumer,
        const std::optional<AdaptiveImmutableLookupDevicePreparation> &setup)
    {
        const auto &layout=placement.gpuAdaptiveLayout();
        const PhysicalMemoryAllocatorIdentity requester{authority->worldRank(),consumer};
        const bool independent=layout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal;
        const bool shared=!independent && layout.consumers().size()>1;
        if(std::find(layout.consumers().begin(),layout.consumers().end(),requester)==layout.consumers().end() ||
            placement.gpuAdaptiveRowElements()!=source.row_elements || !consumer.is_gpu() ||
            (setup && (setup->device!=consumer || !setup->stream || reinterpret_cast<std::uintptr_t>(setup->stream)<=2)))
            throw std::invalid_argument("Adaptive setup requires the exact sealed native source and original GPU worker stream");
        const auto host=layout.sourceServices().front().allocator;
        const auto nativeSource=nativeContextPreparation();
        if(shared)
        {
            const auto sourceLayout=AdaptiveImmutableLookupGeometry::sourceLayout(layout,placement.gpuAdaptiveRequest());
            const auto coordinator=sourceLayout.consumers().front();
            if(requester!=coordinator)
            {
                // The actual factory prepares the genuine coordinator policy
                // first. A foreign graph receives no substitute source owner.
                if(!nativeSource)
                    throw std::logic_error("Shared PLE reader requires its original prepared worker source");
                const auto permit=nativeSource->acquire(consumer);
                if(permit.worker()->ownsCurrentThread() || !setup ||
                    setup->stream!=permit.worker()->defaultStream())
                    throw std::logic_error("Shared PLE reader requires cold preparation on its exact original stream");
                std::lock_guard lock(immutable_row_table_mutex_);
                if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
                const auto original=adaptive_lookup_consumers_.find(coordinator.device);
                if(!nativeSource || nativeSource->memory().get()!=authority.get() ||
                    nativeSource->memory().owner_before(authority) || authority.owner_before(nativeSource->memory()) ||
                    !immutable_row_table_placement_ || *immutable_row_table_placement_!=placement ||
                    immutable_row_table_authority_.get()!=authority.get() ||
                    immutable_row_table_authority_.owner_before(authority) || authority.owner_before(immutable_row_table_authority_) ||
                    !immutable_row_table_source_ || *immutable_row_table_source_!=source ||
                    original==adaptive_lookup_consumers_.end() || !original->second.boundary ||
                    !original->second.participants)
                    throw std::logic_error("Shared PLE reader requires its genuine coordinator source prepared first");
                const auto retained=original->second.participants;
                retained->requireCurrent();
                const auto transport=retained->transport();
                if(!transport->readerGeometry() || transport->readerGeometry()->coordinator()!=coordinator ||
                    std::none_of(transport->readers().begin(),transport->readers().end(),[&](const auto& reader){
                        return reader->originalReader().device()==consumer;
                    }))
                    throw std::logic_error("Shared PLE reader changed its original source transport membership");
                const auto [entry,inserted]=adaptive_lookup_consumers_.try_emplace(consumer);
                if(!inserted && (entry->second.boundary || entry->second.participants!=retained))
                    throw std::logic_error("Shared PLE reader cannot replace its original source authority");
                entry->second.participants=retained;
                immutable_row_table_preparations_.insert_or_assign(consumer,
                    ImmutableRowPreparation{ImmutableRowPreparationState::Ready,std::this_thread::get_id()});
                return;
            }
        }
        std::vector<PhysicalMemoryAllocatorIdentity> orderedOwners;
        const auto& requestedOwners=placement.gpuAdaptiveRequest().placement;
        if(requestedOwners.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal)
        {
            requestedOwners.validate();
            const bool needsPeerObservation = std::any_of(requestedOwners.owners.begin(), requestedOwners.owners.end(),
                [&](const auto& owner) {
                    return owner.participant != *requestedOwners.coordinator &&
                        owner.participant.backend == requestedOwners.coordinator->backend;
                });
            if (needsPeerObservation) {
                const auto inventory = immutable_source_mpi_ ? immutable_source_mpi_->clusterInventory() : nullptr;
                if (!inventory || immutable_source_mpi_->rank() != authority->worldRank())
                    throw std::logic_error("NoPeer pooled source requires its original rank inventory owner");
                const auto original = ImmutableRowTableDirectedTransportPolicy::observe(requestedOwners,
                    inventory->getRank(authority->worldRank()));
                if (original != layout.peerAccess())
                    throw std::logic_error("Pooled source changed its original declared peer coverage");
            }
            // Layout metadata canonicalizes allocator keys. The executable
            // physical coordinate order remains the authored owner request,
            // including C0 participants, and fromLayout authenticates both.
            for(const auto& owner:requestedOwners.owners)
                orderedOwners.push_back({owner.participant.world_rank,owner.participant.device()});
        }
        else if(independent)
            orderedOwners.push_back(requester);
        else
            for(const auto& owner:layout.owners())if(owner.key.tier==ImmutableRowStorage::VRAM)
                orderedOwners.push_back(owner.key.allocator);
        if(nativeSource && (nativeSource->memory().get()!=authority.get() ||
            nativeSource->memory().owner_before(authority) || authority.owner_before(nativeSource->memory())))
            throw std::logic_error("Model loader received a foreign source-preparation PMA");
        if(!nativeSource && (layout.consumers().size()!=1 || orderedOwners.size()!=1 || orderedOwners.front()!=requester))
            throw std::invalid_argument("Distributed adaptive source requires its original prepared participant authority");
        std::vector<PreparedGPUContextUsePermit> participantPermits;
        std::optional<PreparedGPUContextUsePermit> coordinatorPermit;
        if(nativeSource){
            coordinatorPermit.emplace(nativeSource->acquire(consumer));
            for(const auto& owner:orderedOwners){
                if(owner.world_rank!=authority->worldRank())
                    throw std::invalid_argument("Cross-rank source requires its admitted native rank factory");
                // The original inventory observation above selects an explicit
                // mapped or peer route. Reuse the same policy validator here;
                // a same-vendor endpoint is never an automatic peer fallback.
                ImmutableRowTableDirectedTransportPolicy::require(requester,owner,layout.peerAccess());
                participantPermits.push_back(nativeSource->acquire(owner.device));
            }
            if(coordinatorPermit->worker()->ownsCurrentThread() ||
               std::any_of(participantPermits.begin(),participantPermits.end(),[](const auto& permit){return permit.worker()->ownsCurrentThread();}))
                throw std::logic_error("Multi-participant source preparation must run cold outside every original worker");
            if(!setup || setup->stream!=coordinatorPermit->worker()->defaultStream())
                throw std::invalid_argument("Prepared source setup requires the exact original coordinator primary stream");
        }
        const auto geometry=independent
            ? AdaptiveImmutableLookupGeometry::forRequester(layout,source,placement.gpuAdaptiveRequest(),requester)
            : AdaptiveImmutableLookupGeometry::fromLayout(layout,source,placement.gpuAdaptiveRequest(),orderedOwners);
        std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> ready;
        bool prepare_host=false;
        {
            std::unique_lock lock(immutable_row_table_mutex_);
            if(immutable_row_table_placement_ && (*immutable_row_table_placement_!=placement ||
                immutable_row_table_authority_.get()!=authority.get() ||
                immutable_row_table_authority_.owner_before(authority) || authority.owner_before(immutable_row_table_authority_) ||
                !immutable_row_table_source_ ||
                *immutable_row_table_source_!=source))
                throw std::logic_error("Adaptive model cannot replace its source, sole layout or canonical PMA");
            if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
            for(;;)
            {
                const auto found=immutable_row_table_preparations_.find(consumer);
                if(found==immutable_row_table_preparations_.end())break;
                if(found->second.state==ImmutableRowPreparationState::Ready){ready=adaptive_lookup_consumers_.at(consumer).boundary;break;}
                if(found->second.state==ImmutableRowPreparationState::Failed)
                    throw std::logic_error("Adaptive source preparation previously failed");
                if(found->second.preparing_thread==std::this_thread::get_id())
                    throw std::logic_error("Adaptive source setup cannot join its own preparing thread");
                immutable_row_table_ready_.wait(lock,[&] {
                    return immutable_row_table_preparation_failure_ ||
                        immutable_row_table_preparations_.at(consumer).state!=ImmutableRowPreparationState::Preparing;
                });
                if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
            }
            if(!ready)
            {
                if(!setup)throw std::invalid_argument("Fresh adaptive source needs its exact ordinary metadata setup stream");
                if(!resident_row_tables_.empty() || (host_row_provider_ && !adaptive_lookup_service_))
                    throw std::logic_error("Adaptive source cannot alias an already materialized independent storage owner");
                if(!immutable_row_table_placement_)
                {
                    immutable_row_table_placement_=std::make_shared<const ResolvedImmutableRowTablePlacement>(placement);
                    immutable_row_table_authority_=authority;immutable_row_table_source_=source;
                }
                immutable_row_table_preparations_.emplace(consumer,
                    ImmutableRowPreparation{ImmutableRowPreparationState::Preparing,std::this_thread::get_id()});
                adaptive_lookup_consumers_.try_emplace(consumer);
                if(!immutable_row_table_preparations_.contains(host.device))
                {
                    immutable_row_table_preparations_.emplace(host.device,
                        ImmutableRowPreparation{ImmutableRowPreparationState::Preparing,std::this_thread::get_id()});
                    prepare_host=true;
                }
                for(const auto& owner:orderedOwners)if(owner.device!=consumer)
                    immutable_row_table_preparations_.emplace(owner.device,
                        ImmutableRowPreparation{ImmutableRowPreparationState::Preparing,std::this_thread::get_id()});
            }
        }
        if(ready)
        {
            // Source authentication is an owner readiness operation, never an
            // empty gather or a CPU readback of warmed GPU cache state.
            ready->pooledStorage()->requireFor(geometry.physical(),geometry.coordinator(),source,geometry.computeReaderCount(),authority);
            const auto observed=ready->retainedSourceOwnerObservation();
            if(observed.authority().get()!=authority.get() || observed.allocator()!=host ||
                observed.hostPlacement()!=placement.hostPlacement())
                throw std::logic_error("Ready adaptive source lost its actual source/PMA/locality authority");
            return;
        }
        try
        {
            // No model mutex is held through file authentication, physical
            // materialization or native metadata initialization. Failure is
            // sticky; an in-flight partial source is never replaced or retried.
            if(prepare_host)
            {
                const auto host_geometry=independent
                    ? AdaptiveImmutableLookupGeometry::sharedHostGeometry(layout,placement.gpuAdaptiveRequest())
                    : geometry.host();
                auto provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(
                    CpuAdaptiveImmutableRowCache::create(source,host_geometry,authority));
                std::vector<DeviceId> consumers;
                if(shared)consumers.push_back(consumer);
                else for(const auto& original:layout.consumers())consumers.push_back(original.device);
                auto service=qwen4exp::ImmutableLookupService::create(provider,geometry.channelLayout(),
                    consumers,authority,host.device,
                    takeRetainedHostService({authority->worldRank(),placement.lookupIOConsumer(consumer)},
                        RetainedHostServiceRole::ImmutableLookupIO,authority));
                const auto actual=provider->identity();
                {
                    std::lock_guard lock(immutable_row_table_mutex_);
                    if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
                    if(immutable_row_table_file_identity_ && *immutable_row_table_file_identity_!=actual)
                        throw std::logic_error("Adaptive source generation conflicts with the retained original model identity");
                    immutable_row_table_file_identity_=actual;
                    host_row_provider_=std::move(provider);
                    adaptive_lookup_service_=std::move(service);
                    immutable_row_table_preparations_.at(host.device).state=ImmutableRowPreparationState::Ready;
                }
                immutable_row_table_ready_.notify_all();
            }
            std::shared_ptr<HostImmutableRowProvider> provider;
            std::shared_ptr<qwen4exp::ImmutableLookupService> service;
            {
                std::unique_lock lock(immutable_row_table_mutex_);
                const auto host_ready=[&]{return immutable_row_table_preparation_failure_ ||
                    immutable_row_table_preparations_.at(host.device).state!=ImmutableRowPreparationState::Preparing;};
                if(!host_ready() && immutable_row_table_preparations_.at(host.device).preparing_thread==std::this_thread::get_id())
                    throw std::logic_error("Adaptive shared source cannot join its own preparing thread");
                immutable_row_table_ready_.wait(lock,host_ready);
                if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
                provider=host_row_provider_;service=adaptive_lookup_service_;
                if(!provider || !service)throw std::logic_error("Adaptive source has no original shared provider/service");
            }
            const auto binding=service->bindSubordinate(consumer);
            const auto table=binding.deviceView().base.table,generation=binding.storageGeneration();
            const auto policyPlan=qwen4exp::PooledImmutableCoordinatorNativePlan::fromGeometry(
                geometry.physical(),geometry.coordinator(),table,generation);
            std::vector<qwen4exp::PooledImmutableOwnerNativePlan> ownerPlans;
            std::vector<std::shared_ptr<qwen4exp::PooledImmutableOwnerPreparedStorage>> owners;
            for(std::size_t coordinate=0;coordinate<orderedOwners.size();++coordinate){
                ownerPlans.push_back(qwen4exp::PooledImmutableOwnerNativePlan::fromGeometry(
                    geometry.physical(),geometry.coordinator(),coordinate,table,generation));
                const auto initialize=[&]{
                    void* stream=nativeSource?participantPermits[coordinate].worker()->defaultStream():setup->stream;
                    auto owner=qwen4exp::PooledImmutableOwnerPreparedStorage::create(ownerPlans.back(),authority,stream);
                    std::unique_ptr<qwen4exp::IPooledImmutableNativeKernels> code;
#ifdef HAVE_CUDA
                    if(orderedOwners[coordinate].device.is_cuda())code=qwen4exp::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
                    if(orderedOwners[coordinate].device.is_rocm())code=qwen4exp::createROCmPooledImmutableNativeKernels();
#endif
                    if(!code)throw std::invalid_argument("Adaptive owner requires its compiled original native backend");
                    code->initialize(*owner,stream);owners.push_back(std::move(owner));
                };
                if(nativeSource)participantPermits[coordinate].worker()->submitAndWait(initialize);else initialize();
            }
            std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
            const auto initializeCoordinator=[&]{
                auto coordinator=qwen4exp::PooledImmutableCoordinatorPreparedStorage::create(policyPlan,authority,setup->stream);
                std::unique_ptr<qwen4exp::IPooledImmutableNativeKernels> code;
#ifdef HAVE_CUDA
                if(consumer.is_cuda())code=qwen4exp::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
                if(consumer.is_rocm())code=qwen4exp::createROCmPooledImmutableNativeKernels();
#endif
                if(!code)throw std::invalid_argument("Adaptive coordinator requires its compiled original native backend");
                code->initialize(*coordinator,setup->stream);
                const auto packets=qwen4exp::PooledImmutableSourceGeometry::fromPlans(ownerPlans,policyPlan,source,geometry.computeReaderCount());
                boundary=qwen4exp::ImmutableLookupExecutionBoundary::createPooled(
                    service,packets,std::move(owners),std::move(coordinator),authority,setup->stream);
                if(nativeSource)boundary->pooledStorage()->bindPreparedSourceWorkers(nativeSource,*coordinatorPermit,participantPermits);
            };
            if(nativeSource)coordinatorPermit->worker()->submitAndWait(initializeCoordinator);else initializeCoordinator();
            const auto actual=provider->identity();
            {
                std::lock_guard lock(immutable_row_table_mutex_);
                if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
                if(immutable_row_table_file_identity_ && *immutable_row_table_file_identity_!=actual)
                    throw std::logic_error("Adaptive source generation conflicts with the already retained model identity");
                if(host_row_provider_!=provider || adaptive_lookup_service_!=service)
                    throw std::logic_error("Adaptive consumer replaced its shared original source/worker");
                adaptive_lookup_consumers_.at(consumer).boundary=std::move(boundary);
                immutable_row_table_preparations_.at(consumer).state=ImmutableRowPreparationState::Ready;
                immutable_row_table_preparations_.at(host.device).state=ImmutableRowPreparationState::Ready;
                for(const auto& owner:orderedOwners)immutable_row_table_preparations_.at(owner.device).state=ImmutableRowPreparationState::Ready;
            }
            immutable_row_table_ready_.notify_all();
        }
        catch(...)
        {
            {
                std::lock_guard lock(immutable_row_table_mutex_);
                if(!immutable_row_table_preparation_failure_)immutable_row_table_preparation_failure_=std::current_exception();
                immutable_row_table_preparations_.at(consumer).state=ImmutableRowPreparationState::Failed;
                if(prepare_host && immutable_row_table_preparations_.at(host.device).state==ImmutableRowPreparationState::Preparing)
                    immutable_row_table_preparations_.at(host.device).state=ImmutableRowPreparationState::Failed;
                for(const auto& owner:orderedOwners)immutable_row_table_preparations_.at(owner.device).state=ImmutableRowPreparationState::Failed;
            }
            immutable_row_table_ready_.notify_all();throw;
        }
    }

    void ModelContext::prepareImmutableRowTableForConsumer(const ResolvedImmutableRowTablePlacement &placement,
        std::shared_ptr<PhysicalMemoryAuthority> authority, DeviceId consumer,
        const std::optional<ImmutableRowTableDevicePreparation> &upload, const TransferEngine &transfers,
        const std::optional<AdaptiveImmutableLookupDevicePreparation> &adaptive_setup)
    {
        if (isGraphCaptureActive())
            throw std::logic_error("native table materialization is forbidden during graph capture");
        const auto source = immutableRowTableSource();
        if (!authority || placement.worldRank() != authority->worldRank() ||
            placement.extent() != ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()))
            throw std::invalid_argument("native table requires its exact sealed source/PMA rank");
        const auto execution = placement.lookupForConsumer({authority->worldRank(),consumer});
        if(execution==ImmutableRowLookupExecution::AdaptiveGPUCache)
        {
            if(upload)throw std::invalid_argument("Adaptive lookup cannot borrow a whole-table upload authority");
            prepareGPUAdaptiveImmutableLookup(placement,std::move(authority),source,consumer,adaptive_setup);
            return;
        }
        if(adaptive_setup)throw std::invalid_argument("Other immutable table modes cannot receive adaptive setup");
        const bool resident = immutableRowLookupUsesNativeDeviceStorage(execution);
        if ((!resident && upload) || (upload && (upload->device != consumer || !upload->stream)))
            throw std::invalid_argument("native table participant upload does not match its exact storage/consumer");
        const DeviceId owner = resident ? consumer : placement.allocators().front().device;
        std::shared_ptr<HostImmutableRowProvider> ready_host;
        std::shared_ptr<const ResidentImmutableRowTable> ready_resident;
        {
            std::unique_lock lock(immutable_row_table_mutex_);
            if (immutable_row_table_placement_ && (*immutable_row_table_placement_ != placement ||
                immutable_row_table_authority_.get() != authority.get() ||
                !immutable_row_table_source_ || *immutable_row_table_source_ != source))
                throw std::logic_error("native table cannot replace its declared source/placement/PMA authority");
            if (immutable_row_table_preparation_failure_)
                std::rethrow_exception(immutable_row_table_preparation_failure_);
            for (;;)
            {
                const auto found = immutable_row_table_preparations_.find(owner);
                if (found == immutable_row_table_preparations_.end()) break;
                if (found->second.state == ImmutableRowPreparationState::Ready)
                {
                    if (resident) ready_resident = resident_row_tables_.at(owner);
                    else ready_host = host_row_provider_;
                    break;
                }
                if (found->second.state == ImmutableRowPreparationState::Failed)
                    throw std::logic_error("native table participant setup previously failed");
                if (found->second.preparing_thread == std::this_thread::get_id())
                    throw std::logic_error("native table participant setup cannot recursively join its own writer");
                // This is a setup-owner join, not a GPU execution-state wait.
                immutable_row_table_ready_.wait(lock,[&] {
                    return immutable_row_table_preparation_failure_ ||
                        immutable_row_table_preparations_.at(owner).state != ImmutableRowPreparationState::Preparing;
                });
                if (immutable_row_table_preparation_failure_)
                    std::rethrow_exception(immutable_row_table_preparation_failure_);
            }
            if (!ready_host && !ready_resident)
            {
                if (resident && !upload)
                    throw std::invalid_argument("fresh resident table participant requires its drained-loader upload authority");
                if (resident) upload->staging.requireFor(authority,consumer,source.rowBytes());
                if (!immutable_row_table_placement_)
                {
                    immutable_row_table_placement_=std::make_shared<const ResolvedImmutableRowTablePlacement>(placement);
                    immutable_row_table_authority_=authority;
                    immutable_row_table_source_=source;
                }
                immutable_row_table_preparations_.emplace(owner,
                    ImmutableRowPreparation{ImmutableRowPreparationState::Preparing,std::this_thread::get_id()});
            }
        }
        if (ready_host || ready_resident)
        {
            // Reuse authenticates source outside the global model mutex and
            // needs neither the old staging receipt nor an upload stream.
            if (ready_host) ready_host->authenticateSource();
            else
            {
                const auto expected = placement.isFixedHeadShards()
                    ? placement.fixedHeadSourceRangeFor({authority->worldRank(),consumer})
                    : ImmutableRowTableSourceRange::whole(placement.extent());
                if (ready_resident->device()!=consumer || ready_resident->source()!=source ||
                    ready_resident->sourceRange()!=expected || ready_resident->nativeBytes()!=expected.nativeBytes())
                    throw std::logic_error("Ready native table differs from its exact original source/owner/range");
                ready_resident->authenticateSource();
            }
            std::lock_guard lock(immutable_row_table_mutex_);
            if (immutable_row_table_preparation_failure_)
                std::rethrow_exception(immutable_row_table_preparation_failure_);
            return;
        }
        try
        {
            // Ordinary participant preparation has already drained its exact
            // staging lease. No model mutex is held during file I/O or H2D.
            std::shared_ptr<HostImmutableRowProvider> host;
            if(!resident)
            {
                if(placement.isHostAdaptive())
                {
                    const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(placement.hostAdaptiveLayout(),
                        {authority->worldRank(),owner},placement.hostAdaptiveMaximumLiveRows(),placement.hostAdaptivePolicy());
                    host=HostImmutableRowProvider::fromAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,geometry,authority));
                }
                else host=HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(
                    source,placement.storagePolicy(),owner,authority,placement.hostPlacement()));
            }
            std::shared_ptr<const ResidentImmutableRowTable> gpu;
            if (resident)
                gpu = execution == ImmutableRowLookupExecution::FixedHeadShardGPU
                    ? ResidentImmutableRowTable::prepareRange(source,placement,authority,
                        consumer,upload->staging,upload->stream,transfers)
                    : ResidentImmutableRowTable::prepare(source,placement,authority,
                        consumer,upload->staging,upload->stream,transfers);
            if (host) host->authenticateSource();
            const auto identity = host ? host->identity() : gpu->identity();
            {
                std::lock_guard lock(immutable_row_table_mutex_);
                if (immutable_row_table_preparation_failure_)
                    std::rethrow_exception(immutable_row_table_preparation_failure_);
                if (immutable_row_table_file_identity_ && *immutable_row_table_file_identity_ != identity)
                    throw std::runtime_error("native table consumers do not share one immutable source generation");
                immutable_row_table_file_identity_=identity;
                if (resident) resident_row_tables_.emplace(owner,std::move(gpu));
                else host_row_provider_=std::move(host);
                immutable_row_table_preparations_.at(owner).state=ImmutableRowPreparationState::Ready;
            }
            immutable_row_table_ready_.notify_all();
        }
        catch (...)
        {
            {
                std::lock_guard lock(immutable_row_table_mutex_);
                if (!immutable_row_table_preparation_failure_)
                    immutable_row_table_preparation_failure_=std::current_exception();
                immutable_row_table_preparations_.at(owner).state=ImmutableRowPreparationState::Failed;
            }
            immutable_row_table_ready_.notify_all();
            throw;
        }
    }

    void ModelContext::requireImmutableRowTablePlacementComplete(const ResolvedImmutableRowTablePlacement &placement,
        const std::shared_ptr<PhysicalMemoryAuthority> &authority) const
    {
        std::unique_lock lock(immutable_row_table_mutex_);
        if (immutable_row_table_preparation_failure_)
            std::rethrow_exception(immutable_row_table_preparation_failure_);
        if (!authority || !immutable_row_table_placement_ || *immutable_row_table_placement_ != placement ||
            immutable_row_table_authority_.get() != authority.get() || !immutable_row_table_file_identity_)
            throw std::logic_error("native table placement is not completely materialized by its exact PMA");
        for (const auto &allocator : placement.allocators())
        {
            const auto found=immutable_row_table_preparations_.find(allocator.device);
            if (found == immutable_row_table_preparations_.end() ||
                found->second.state != ImmutableRowPreparationState::Ready)
                throw std::logic_error("native table placement still has an unprepared participant");
        }
        if(placement.isGPUAdaptive())
        {
            const auto source=immutable_row_table_source_;
            const auto provider=host_row_provider_;
            const auto service=adaptive_lookup_service_;
            if(!source || !provider || !service || !resident_row_tables_.empty() ||
                adaptive_lookup_consumers_.size()!=placement.consumers().size())
                throw std::logic_error("Ready adaptive model lost its exact consumer/source population");
            const auto& fullLayout=placement.gpuAdaptiveLayout();
            const bool shared=fullLayout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator &&
                placement.consumers().size()>1;
            const auto sourceLayout=shared
                ? AdaptiveImmutableLookupGeometry::sourceLayout(fullLayout,placement.gpuAdaptiveRequest()) : fullLayout;
            const auto expectedRequesters=sourceLayout.consumers();
            const auto requesters=provider->subordinateRequesters();
            if(requesters.size()!=expectedRequesters.size() ||
                !std::equal(requesters.begin(),requesters.end(),expectedRequesters.begin()))
                throw std::logic_error("Ready adaptive source has a different declared requester set");
            std::vector<std::pair<DeviceId,std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary>>> boundaries;
            for(const auto& consumer:placement.consumers())
            {
                const auto found=adaptive_lookup_consumers_.find(consumer.device);
                if(found==adaptive_lookup_consumers_.end())
                    throw std::logic_error("Ready adaptive model omitted an original consumer entry");
                if(shared && consumer!=expectedRequesters.front())
                {
                    const auto original=adaptive_lookup_consumers_.find(expectedRequesters.front().device);
                    if(found->second.boundary || original==adaptive_lookup_consumers_.end() ||
                        !found->second.participants || found->second.participants!=original->second.participants)
                        throw std::logic_error("Ready query-only reader replaced its original shared source");
                    continue;
                }
                if(!found->second.boundary)
                    throw std::logic_error("Ready adaptive source omitted its original boundary");
                boundaries.emplace_back(consumer.device,found->second.boundary);
            }
            lock.unlock();
            const auto& layout=placement.gpuAdaptiveLayout();
            for(const auto& [device,boundary]:boundaries)
            {
                const auto geometry=layout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal
                    ? AdaptiveImmutableLookupGeometry::forRequester(layout,*source,placement.gpuAdaptiveRequest(),
                        {authority->worldRank(),device})
                    : AdaptiveImmutableLookupGeometry::fromLayout(layout,*source,placement.gpuAdaptiveRequest());
                boundary->pooledStorage()->requireFor(geometry.physical(),geometry.coordinator(),*source,geometry.computeReaderCount(),authority);
                const auto host=boundary->retainedSourceOwnerObservation();
                // This is the SAME shared physical/source observation on each
                // channel, not a per-consumer credit or another reservation.
                if(host.authority().get()!=authority.get() ||
                    host.authority().owner_before(authority) || authority.owner_before(host.authority()) ||
                    host.allocator()!=layout.sourceServices().front().allocator || host.hostPlacement()!=placement.hostPlacement())
                    throw std::logic_error("Ready mixed source/channel claims differ from the original placement/PMA");
            }
            std::vector<std::shared_ptr<const qwen4exp::PooledImmutableSourcePreparedStorage>> originalSources;
            for(const auto& [device,boundary]:boundaries)originalSources.push_back(boundary->pooledStorage());
            const auto& request=placement.gpuAdaptiveRequest();
            const auto hostGeometry=layout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal
                ? AdaptiveImmutableLookupGeometry::sharedHostGeometry(layout,request)
                : AdaptiveImmutableLookupGeometry::fromLayout(layout,*source,request).host();
            auto admitted=std::make_shared<const ImmutableLookupSetupObservation>(ImmutableLookupSetupObservation::captureAdaptive(
                layout,hostGeometry,service,originalSources));
            lock.lock();
            if(immutable_row_table_preparation_failure_)std::rethrow_exception(immutable_row_table_preparation_failure_);
            if(host_row_provider_!=provider || adaptive_lookup_service_!=service)
                throw std::logic_error("Adaptive shared source changed during readiness validation");
            for(const auto& [device,boundary]:boundaries)
                if(adaptive_lookup_consumers_.at(device).boundary!=boundary)
                    throw std::logic_error("Adaptive consumer boundary changed during readiness validation");
            if(immutable_lookup_setup_observation_ && !immutable_lookup_setup_observation_->sameCapacity(*admitted))
                throw std::logic_error("Ready lookup setup replaced its original admitted capacity evidence");
            if(!immutable_lookup_setup_observation_) {
                immutable_lookup_setup_observation_=std::move(admitted);
                immutable_lookup_setup_observation_->publish();
            }
            return;
        }
        if (placement.isFixedHeadShards() || (!placement.isHostAdaptive() && !placement.isGPUAdaptive() && placement.storage()==ImmutableRowStorage::VRAM))
        {
            if (host_row_provider_)
                throw std::logic_error("Native GPU ngrams cannot retain a hidden complete host provider");
            for (const auto &allocator : placement.allocators())
            {
                const auto table=resident_row_tables_.find(allocator.device);
                const auto expected=placement.isFixedHeadShards() ? placement.fixedHeadSourceRangeFor(allocator)
                    : ImmutableRowTableSourceRange::whole(placement.extent());
                if (table==resident_row_tables_.end() || !table->second || !immutable_row_table_source_ ||
                    table->second->device()!=allocator.device || table->second->source()!=*immutable_row_table_source_ ||
                    table->second->sourceRange()!=expected || table->second->nativeBytes()!=expected.nativeBytes() ||
                    table->second->identity()!=*immutable_row_table_file_identity_)
                    throw std::logic_error("Ready native ngrams do not match every exact source/range/PMA owner");
            }
        }
        else
        {
            if(!host_row_provider_)throw std::logic_error("Ready host ngrams have no materialized provider");
            const auto observed=host_row_provider_->retainedOwnerObservation();
            if(observed.authority().get()!=authority.get() || !placement.containsAllocator(observed.allocator()) ||
                observed.hostPlacement()!=placement.hostPlacement() ||
                (placement.isHostAdaptive() ? host_row_provider_->mode()!=HostImmutableRowMode::AdaptiveRAMSSD :
                    host_row_provider_->wholeStorage()!=placement.storage()))
                throw std::logic_error("Ready host ngrams do not match their exact sealed mode/source/PMA owner");
        }
    }

    void ModelContext::prepareImmutableRowTable(const ResolvedImmutableRowTablePlacement &placement,
        std::shared_ptr<PhysicalMemoryAuthority> authority,
        std::span<const ImmutableRowTableDevicePreparation> uploads,const TransferEngine &transfers)
    {
        // Existing Ready storage needs no new staging/stream. This wrapper is
        // a setup convenience; production participant factories use the same
        // primitive immediately after each exact ordinary loader drains.
        bool complete=false;
        {
            std::lock_guard lock(immutable_row_table_mutex_);
            if (immutable_row_table_placement_ && (*immutable_row_table_placement_ != placement ||
                immutable_row_table_authority_.get() != authority.get()))
                throw std::logic_error("native table batch cannot replace its declared placement/PMA authority");
            if (immutable_row_table_preparation_failure_)
                std::rethrow_exception(immutable_row_table_preparation_failure_);
            complete=immutable_row_table_placement_ && *immutable_row_table_placement_ == placement &&
                immutable_row_table_authority_.get() == authority.get() &&
                std::all_of(placement.allocators().begin(),placement.allocators().end(),[&](const auto &a) {
                    const auto found=immutable_row_table_preparations_.find(a.device);
                    return found != immutable_row_table_preparations_.end() &&
                        found->second.state == ImmutableRowPreparationState::Ready;
                });
        }
        std::map<DeviceId,const ImmutableRowTableDevicePreparation *> unique;
        const bool resident=placement.isFixedHeadShards() ||
            (!placement.isHostAdaptive() && !placement.isGPUAdaptive() && placement.storage()==ImmutableRowStorage::VRAM);
        if (!resident && !uploads.empty())
            throw std::invalid_argument("host native table cannot receive GPU upload authorities");
        if (resident && (!uploads.empty() || !complete))
        {
            if (!authority || uploads.size() != placement.allocators().size())
                throw std::invalid_argument("native table batch setup must cover every planned GPU exactly once");
            for (const auto &upload : uploads)
            {
                if (!upload.stream || !placement.containsAllocator({authority->worldRank(),upload.device}) ||
                    !unique.emplace(upload.device,&upload).second)
                    throw std::invalid_argument("native table batch setup has duplicate/foreign GPU upload ownership");
                upload.staging.requireFor(authority,upload.device,placement.extent().rowBytes());
            }
        }
        for (const auto &consumer : placement.consumers())
        {
            const auto found=unique.find(consumer.device);
            const auto upload=found == unique.end() ? std::optional<ImmutableRowTableDevicePreparation>{} :
                std::optional<ImmutableRowTableDevicePreparation>{*found->second};
            prepareImmutableRowTableForConsumer(placement,authority,consumer.device,upload,transfers);
        }
        requireImmutableRowTablePlacementComplete(placement,authority);
    }

    ModelContext::ModelContext(const std::string &model_path,
                               std::shared_ptr<IMPIContext> mpi_ctx,
                               std::shared_ptr<WeightPlacementMap> placement_map,
                               TensorFactory *factory,
                               WeightDistributionStrategy strategy)
        : model_path_(model_path), immutable_source_mpi_(mpi_ctx), loader_(factory)
    {
        // WeightManager will be created after model is loaded (see create())
        // placement_map is passed through to WeightManager
    }

    std::shared_ptr<ModelContext> ModelContext::create(
        const std::string &model_path,
        std::shared_ptr<IMPIContext> mpi_ctx,
        std::shared_ptr<WeightPlacementMap> placement_map,
        TensorFactory *factory,
        WeightDistributionStrategy strategy,
        WeightPrecision weight_precision)
    {
        // Create TensorFactory from MPI context if not provided
        // This ensures proper NUMA-aware allocation and avoids the ModelLoader
        // creating an internal single-rank MPI context
        std::unique_ptr<TensorFactory> owned_factory;
        if (!factory && mpi_ctx)
        {
            owned_factory = std::make_unique<TensorFactory>(*mpi_ctx);
            factory = owned_factory.get();
        }

        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, placement_map, factory, strategy));

        // Store owned factory so it lives as long as the context
        if (owned_factory)
        {
            ctx->owned_test_factory_ = std::move(owned_factory);
        }

        // Load model metadata
        try
        {
            ctx->loader_.loadModel(model_path);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[ModelContext] Failed to load model: " << model_path << " (" << e.what() << ")");
            return nullptr;
        }

        // Create WeightManager with loaded model and placement map
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, mpi_ctx, placement_map, strategy, weight_precision);

        // Set up interface wrappers
        ctx->loader_interface_ = std::make_shared<ModelLoaderInterfaceWrapper>(ctx->loader_);

        return ctx;
    }

    std::shared_ptr<ModelContext> ModelContext::create(
        const std::string &model_path,
        const ModelContextConfig &config)
    {
        // Validate config
        auto errors = config.validate();
        if (!errors.empty())
        {
            for (const auto &e : errors)
            {
                LOG_ERROR("[ModelContext] Config error: " << e);
            }
            return nullptr;
        }

        // Determine strategy - auto-select if layer partitioned
        WeightDistributionStrategy strategy = config.strategy;
        if (config.isLayerPartitioned() && strategy == WeightDistributionStrategy::REPLICATED)
        {
            strategy = WeightDistributionStrategy::LAYER_PARTITIONED;
        }

        // Create TensorFactory from MPI context if not provided
        std::unique_ptr<TensorFactory> owned_factory;
        TensorFactory *factory = config.factory;
        if (!factory && config.mpi_ctx)
        {
            owned_factory = std::make_unique<TensorFactory>(*config.mpi_ctx);
            factory = owned_factory.get();
        }

        // Create context
        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, config.mpi_ctx, config.placement_map, factory, strategy));

        // Store owned factory so it lives as long as the context
        if (owned_factory)
        {
            ctx->owned_test_factory_ = std::move(owned_factory);
        }

        // Configure mmap before loading (must precede loadModel)
        ctx->loader_.setUseMmap(config.use_mmap);
        ctx->loader_.setSkipMmapCacheEviction(config.skip_mmap_cache_eviction);
        ctx->loader_.setPayloadAccessPattern(config.payload_access_pattern);

        // Load model metadata
        try
        {
            ctx->loader_.loadModel(model_path);
            if (config.shared_predictor) ctx->loader_.bindSharedPredictor(*config.shared_predictor, config.predictor_materialization);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[ModelContext] Failed to load model: " << model_path << " (" << e.what() << ")");
            return nullptr;
        }

        // Create WeightManager
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, config.mpi_ctx, config.placement_map, strategy, config.weight_precision);

        // Configure layer range if partitioned
        if (strategy == WeightDistributionStrategy::LAYER_PARTITIONED)
        {
            // ModelContextConfig uses inclusive last_layer (-1 = all layers)
            // WeightManager uses exclusive last_layer [first, last)
            if (config.last_layer >= 0)
            {
                ctx->weight_manager_->setLayerRange(config.first_layer, config.last_layer + 1);
            }
            else if (config.first_layer > 0)
            {
                // Has offset but no upper limit — shouldn't happen, but handle gracefully
                ctx->weight_manager_->setLayerRange(config.first_layer, std::numeric_limits<int>::max());
            }
            // If first_layer==0 and last_layer==-1, skip setLayerRange (all layers pass through)
            ctx->weight_manager_->setHasEmbedding(config.has_embedding);
            ctx->weight_manager_->setHasLmHead(config.has_lm_head);
        }

        // Configure sharding if needed (TODO: implement setShardInfo in WeightManager)
        // if (config.isSharded()) {
        //     ctx->weight_manager_->setShardInfo(
        //         config.shard_index, config.total_shards, config.work_fraction);
        // }

        // Set up interface wrappers
        ctx->loader_interface_ = std::make_shared<ModelLoaderInterfaceWrapper>(ctx->loader_);

        LOG_DEBUG("[ModelContext] Created with " << config.toString());
        return ctx;
    }

    std::shared_ptr<ModelContext> ModelContext::createForPPStage(
        const std::string &model_path,
        int first_layer,
        int last_layer,
        bool has_embedding,
        bool has_lm_head,
        std::shared_ptr<IMPIContext> mpi_ctx,
        WeightPrecision weight_precision)
    {
        // Create context with LAYER_PARTITIONED strategy
        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, nullptr, nullptr, WeightDistributionStrategy::LAYER_PARTITIONED));

        // Load model metadata
        try
        {
            ctx->loader_.loadModel(model_path);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[ModelContext] Failed to load model: " << model_path << " (" << e.what() << ")");
            return nullptr;
        }

        // Create WeightManager with LAYER_PARTITIONED strategy
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, mpi_ctx, nullptr, WeightDistributionStrategy::LAYER_PARTITIONED, weight_precision);

        // Configure layer range for filtering
        ctx->weight_manager_->setLayerRange(first_layer, last_layer, has_embedding, has_lm_head);

        // Set up interface wrappers
        ctx->loader_interface_ = std::make_shared<ModelLoaderInterfaceWrapper>(ctx->loader_);

        // CRITICAL: Override blockCount() to return the layer count for THIS stage
        // This ensures nested MDO (TP within PP) builds graphs with correct layer count
        ctx->pp_block_count_override_ = last_layer - first_layer;

        LOG_DEBUG("[ModelContext] Created PP stage context for layers [" << first_layer << ", " << last_layer
                                                                         << "), embedding=" << (has_embedding ? "yes" : "no")
                                                                         << ", lm_head=" << (has_lm_head ? "yes" : "no"));

        return ctx;
    }

    std::shared_ptr<ModelContext> ModelContext::createForTesting(
        const std::string &model_path,
        std::shared_ptr<IMPIContext> mpi_ctx,
        uint32_t block_count,
        bool with_weight_manager)
    {
        // Create TensorFactory from MPI context (if provided) to prevent ModelLoader
        // from creating internal MPI_COMM_NULL context that conflicts with test's MPI_COMM_WORLD
        std::unique_ptr<TensorFactory> owned_factory;
        TensorFactory *factory = nullptr;
        if (mpi_ctx)
        {
            owned_factory = std::make_unique<TensorFactory>(*mpi_ctx);
            factory = owned_factory.get();
        }

        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, nullptr, factory, WeightDistributionStrategy::REPLICATED));

        // Store owned factory so it lives as long as the context
        if (owned_factory)
        {
            ctx->owned_test_factory_ = std::move(owned_factory);
        }

        // Initialize minimal valid model structure to prevent accessing uninitialized memory
        ctx->loader_.initializeTestModel(block_count);

        // Set up interface wrappers (no WeightManager for test contexts by default)
        ctx->loader_interface_ = std::make_shared<ModelLoaderInterfaceWrapper>(ctx->loader_);
        if (with_weight_manager)
        {
            ctx->weight_manager_ = std::make_shared<WeightManager>(
                ctx->loader_, mpi_ctx, nullptr, WeightDistributionStrategy::REPLICATED);
        }

        return ctx;
    }

} // namespace llaminar2
