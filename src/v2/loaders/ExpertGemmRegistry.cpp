/**
 * @file ExpertGemmRegistry.cpp
 * @brief Implements scoped prepared MoE GEMM registration and lifetime lookup.
 *
 * Physical projection layout is part of prepared identity, not mutable expert
 * placement. A caller requesting a whole expert cannot borrow a down slice;
 * partition-aware callers must authenticate the original source and interval.
 * Static F shards retain the exact original frozen view and never publish an
 * unscoped alias; complete acquisition authenticates all original E triplets.
 */

#include "ExpertGemmRegistry.h"
#include "tensors/TensorKernels.h"
#include "tensors/TensorClasses.h"

#include <algorithm>
#include <functional>
#include <set>
#include <mutex>
#include <shared_mutex>
#include <tuple>
#include <stdexcept>
#include <limits>

namespace llaminar2
{

    namespace
    {
        /** @return Registry role for a semantic original projection, rejecting all other tensors. */
        ExpertGemmRegistry::WeightRole staticRegistryRole(llaminar2::WeightRole role)
        {
            if (role == llaminar2::WeightRole::MoEExpertGate) return ExpertGemmRegistry::WeightRole::GATE;
            if (role == llaminar2::WeightRole::MoEExpertUp) return ExpertGemmRegistry::WeightRole::UP;
            if (role == llaminar2::WeightRole::MoEExpertDown) return ExpertGemmRegistry::WeightRole::DOWN;
            throw std::invalid_argument("Static expert source has a foreign semantic role");
        }

        /** @brief Check the exact native matrix/source rectangle before reading or publishing an engine. */
        void validateStaticMatrix(const WeightBinding &source, bool require_source_bytes = false)
        {
            (void)ExpertGemmRegistry::staticProjectionScope(source);
            (void)staticRegistryRole(source.identity.role);
            const auto &ownership = *source.projection_ownership;
            const auto projection = ownership.projection(source.identity.role);
            const auto &slice = source.slice;
            const int expert = source.identity.expert;
            if (!source.projection_source_owner ||
                source.projection_source_owner->shape() != std::vector<size_t>{size_t(projection.columns), size_t(projection.rows), size_t(ownership.geometry().experts)} ||
                source.projection_source_owner->native_type() != source.tensor->native_type() ||
                source.binding_id == 0 || source.identity.instance_id != source.binding_id ||
                source.identity.model_id.value == 0 || !source.identity.source_instance_id || !source.prepared ||
                source.floating_operand_precision != FloatingActivationOperandPrecision::Native ||
                inferWeightRole(source.identity.canonical_name) != source.identity.role ||
                inferWeightLayer(source.identity.canonical_name) != source.identity.layer ||
                source.prepared->kind != (source.prepared->device.is_cuda()
                    ? PreparedWeightKind::CudaInt8PackedGemm : PreparedWeightKind::RocmInt8PackedGemm) ||
                source.prepared->model_id.value != source.identity.model_id.value ||
                source.prepared->binding_id != source.binding_id ||
                expert < 0 || expert >= ownership.geometry().experts ||
                source.tensor->shape() != std::vector<size_t>{size_t(projection.rows), size_t(projection.columns)} ||
                source.identity.derivation != WeightDerivationKind::ExpertSlice ||
                slice.source_rows != size_t(projection.source_rows) ||
                slice.source_cols != size_t(projection.source_columns) ||
                slice.row_start != size_t(projection.first_row) || slice.row_count != size_t(projection.rows) ||
                slice.col_start != size_t(projection.first_column) || slice.col_count != size_t(projection.columns) ||
                slice.expert_start != size_t(expert) || slice.expert_count != 1 ||
                slice.expert_ids != std::vector<int>{expert} || !slice.inner_is_presliced)
                throw std::invalid_argument("Static expert matrix lost its exact original source/rectangle identity");
            const size_t bytes = source.tensor->size_bytes();
            if (source.projection_source_owner->size_bytes() % size_t(ownership.geometry().experts) ||
                bytes != source.projection_source_owner->size_bytes() / size_t(ownership.geometry().experts))
                throw std::invalid_argument("Static expert source has a different native byte extent");
            const auto parent_address = reinterpret_cast<uintptr_t>(source.projection_source_owner->raw_data());
            const auto matrix_address = reinterpret_cast<uintptr_t>(source.tensor->raw_data());
            const size_t offset = size_t(expert) * bytes; // Bounded by the actual parent byte extent above.
            if (require_source_bytes && (!parent_address || !matrix_address ||
                source.projection_source_owner->is_raw_data_released()))
                throw std::invalid_argument("Static expert registration requires original native source bytes");
            if (!source.projection_source_owner->is_raw_data_released() && (parent_address || matrix_address) &&
                (!parent_address || offset > std::numeric_limits<uintptr_t>::max() - parent_address ||
                 matrix_address != parent_address + offset))
                throw std::invalid_argument("Static expert view does not address its actual original source slot");
        }
    }

    ExpertGemmRegistry::ParticipantLayerScope ExpertGemmRegistry::staticProjectionScope(const WeightBinding &source)
    {
        const auto device = source.residency.resident_device.value_or(source.residency.home_device);
        if (!source.immutable || !source.tensor || source.tensor_owner.get() != source.tensor ||
            !source.projection_ownership || source.projection_ownership->hasMovableProjectionSet() ||
            !device.is_gpu() || source.identity.overlay_domain.empty() ||
            source.identity.overlay_participant_world_rank < 0 || source.identity.layer < 0 ||
            source.identity.overlay_participant_index != source.projection_ownership->participant() ||
            source.identity.tp_rank_or_device_index != source.projection_ownership->participant() ||
            (source.prepared && source.prepared->device != device))
            throw std::invalid_argument("Static expert GPU preparation requires its declared original participant scope and owned source");
        return {source.identity.overlay_domain, device, source.identity.overlay_participant_world_rank,
                source.identity.overlay_participant_index, source.identity.layer};
    }

    void ExpertGemmRegistry::registerStaticProjection(const WeightBinding &source, ITensorGemm *engine,
                                                      std::shared_ptr<ITensorGemm> owner)
    {
        validateStaticMatrix(source, true);
        const auto scope = staticProjectionScope(source);
        if (!engine || owner.get() != engine || !engine->weights_converted())
            throw std::invalid_argument("Static expert prepared engine has foreign lifetime or unfinished preparation");
        const int n = int(source.tensor->shape()[0]);
        const int k = int(source.tensor->shape()[1]);
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(source.tensor);
        const auto *native = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (native)
        {
            DeviceNativeVNNIMatrixDesc descriptor;
            NativeVnniSourceIdentity original;
            if (!engine->exportNativeVNNIMatrixDesc(descriptor) || !descriptor.valid() ||
                descriptor.n != n || descriptor.k != k || descriptor.blocks_per_row != uint32_t(k / 32) ||
                !descriptor.source_identity_present || descriptor.source_codebook_id != native->codebook_id ||
                bool(descriptor.source_is_superblock) != native->is_superblock ||
                !engine->exportNativeVNNISourceIdentity(original) || !original.present ||
                original.codebook_id != native->codebook_id || original.is_superblock != native->is_superblock)
                throw std::invalid_argument("Static expert prepared native descriptor lost original geometry/codebook");
        }
        else
        {
            ContiguousFloatingPointWeightDescriptor descriptor;
            if (!engine->exportContiguousFloatingPointWeights(descriptor) || !descriptor.valid() ||
                descriptor.n != n || descriptor.k != k || descriptor.type != source.tensor->native_type())
                throw std::invalid_argument("Static expert prepared floating descriptor lost original geometry/format");
        }
        const auto role = staticRegistryRole(source.identity.role);
        auto retained = std::make_shared<const WeightBinding>(source);
        Key key{scope.domain_name, scope.device, scope.layer, source.identity.expert, role,
                scope.participant_world_rank, scope.participant_index};
        std::unique_lock lock(mutex_);
        if (engines_.contains(key))
            throw std::logic_error("Static expert participant key was already published");
        engines_.emplace(std::move(key), Entry{engine, std::move(owner), source.projection_ownership, std::move(retained)});
    }

    std::optional<ExpertGemmRegistry::StaticProjectionLayer> ExpertGemmRegistry::acquireStaticProjectionLayer(
        const ParticipantLayerScope &scope, const MoEExpertProjectionOwnership &projection,
        std::span<const WeightBinding> sources) const
    {
        if (size_t(projection.geometry().experts) > std::numeric_limits<size_t>::max() / 3 ||
            projection.hasMovableProjectionSet() || scope.participant_index != projection.participant() ||
            scope.domain_name.empty() || !scope.device.is_gpu() || scope.layer < 0 || scope.participant_world_rank < 0 ||
            sources.size() != size_t(projection.geometry().experts) * 3)
            throw std::invalid_argument("Static expert acquisition requires the exact complete participant source declaration");
        std::vector<const WeightBinding *> ordered(sources.size(), nullptr);
        for (const auto &source : sources)
        {
            validateStaticMatrix(source);
            if (staticProjectionScope(source) != scope || source.projection_ownership != projection ||
                source.identity.model_id.value != sources.front().identity.model_id.value)
                throw std::invalid_argument("Static expert acquisition has a foreign domain/participant/partition");
            const auto index = size_t(source.identity.expert) * 3 + size_t(staticRegistryRole(source.identity.role));
            if (ordered[index]) throw std::invalid_argument("Static expert acquisition repeats an original expert role");
            ordered[index] = &source;
        }
        std::vector<const Entry *> registered(ordered.size(), nullptr);
        size_t present = 0;
        std::shared_lock lock(mutex_);
        for (size_t index = 0; index < ordered.size(); ++index)
        {
            if (!ordered[index]) throw std::invalid_argument("Static expert acquisition omits an original expert role");
            const auto &source = *ordered[index];
            Key key{scope.domain_name, scope.device, scope.layer, int(index / 3), WeightRole(index % 3),
                    scope.participant_world_rank, scope.participant_index};
            const auto found = engines_.find(key);
            if (found == engines_.end()) continue;
            const auto &entry = found->second;
            if (!entry.static_source || entry.projection_ownership != projection || !entry.engine ||
                entry.ownership.get() != entry.engine || entry.static_source->binding_id != source.binding_id ||
                entry.static_source->identity.model_id.value != source.identity.model_id.value ||
                entry.static_source->identity.source_instance_id != source.identity.source_instance_id ||
                entry.static_source->tensor != source.tensor ||
                entry.static_source->tensor_owner.owner_before(source.tensor_owner) ||
                source.tensor_owner.owner_before(entry.static_source->tensor_owner) ||
                entry.static_source->projection_source_owner != source.projection_source_owner ||
                entry.static_source->projection_source_owner.owner_before(source.projection_source_owner) ||
                source.projection_source_owner.owner_before(entry.static_source->projection_source_owner))
                throw std::logic_error("Static expert acquisition found a stale source or another prepared partition");
            registered[index] = &entry;
            ++present;
        }
        if (!present) return std::nullopt;
        if (present != ordered.size())
            throw std::logic_error("Static expert participant registry is partially published; model preparation failed");
        StaticProjectionLayer result{scope, projection, {}, {}};
        result.experts.resize(size_t(projection.geometry().experts));
        result.sources.reserve(sources.size());
        for (size_t index = 0; index < registered.size(); ++index)
        {
            const auto &entry = *registered[index];
            auto &expert = result.experts[index / 3];
            expert.scope = scope;
            expert.expert = int(index / 3);
            if (index % 3 == 0) expert.gate = entry.ownership;
            else if (index % 3 == 1) expert.up = entry.ownership;
            else expert.down = entry.ownership;
            result.sources.push_back(*entry.static_source);
        }
        return result;
    }

    bool ExpertGemmRegistry::Key::operator==(const Key &other) const
    {
        return domain_name == other.domain_name &&
               device == other.device &&
               layer == other.layer &&
               expert == other.expert &&
               role == other.role &&
               participant_world_rank == other.participant_world_rank &&
               participant_index == other.participant_index;
    }

    size_t ExpertGemmRegistry::KeyHash::operator()(const Key &k) const
    {
        size_t hash = std::hash<std::string>{}(k.domain_name);
        hash ^= std::hash<std::string>{}(k.device.to_string()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(k.layer) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(k.expert) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.role)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(k.participant_world_rank) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(k.participant_index) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }

    void ExpertGemmRegistry::registerEngine(DeviceId device, int layer, int expert, WeightRole role,
                                            ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership)
    {
        registerEngineForDomain({}, device, layer, expert, role, engine, std::move(ownership));
    }

    void ExpertGemmRegistry::registerEngineForDomain(const std::string &domain_name,
                                                     DeviceId device, int layer, int expert, WeightRole role,
                                                     ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership,
                                                     std::optional<MoEExpertProjectionOwnership> projection)
    {
        registerEngineForParticipant(domain_name, device, -1, -1, layer, expert, role,
            engine, std::move(ownership), std::move(projection));
    }

    void ExpertGemmRegistry::registerEngineForParticipant(const std::string &domain_name,
                                                          DeviceId device, int participant_world_rank, int participant_index,
                                                          int layer, int expert, WeightRole role,
                                                          ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership,
                                                          std::optional<MoEExpertProjectionOwnership> projection)
    {
        if (projection && !projection->hasMovableProjectionSet())
            throw std::invalid_argument("Static original projection registration requires its frozen source API");
        if (projection && (!device.is_gpu() || domain_name.empty() || !engine || layer < 0 ||
            static_cast<unsigned>(role) > static_cast<unsigned>(WeightRole::DOWN) ||
            ownership.get() != engine || expert < 0 || expert >= projection->geometry().experts ||
            (participant_index >= 0 && participant_index != projection->participant())))
            throw std::invalid_argument("Projected expert registration has inconsistent participant/source identity");
        std::unique_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        key.participant_world_rank = participant_world_rank;
        key.participant_index = participant_index;
        const auto previous = engines_.find(key);
        if (previous != engines_.end() && (previous->second.static_source || previous->second.projection_ownership != projection))
            throw std::logic_error("Prepared expert registration cannot change projection layout: " + domain_name);
        engines_[key] = Entry{engine, std::move(ownership), std::move(projection)};
    }

    bool ExpertGemmRegistry::aliasEngineForDomainFromDevice(const std::string &domain_name,
                                                            DeviceId device, int layer, int expert, WeightRole role)
    {
        return aliasEngineForParticipantFromDevice(domain_name, device, -1, -1, layer, expert, role);
    }

    bool ExpertGemmRegistry::aliasEngineForParticipantFromDevice(const std::string &domain_name,
                                                                 DeviceId device, int participant_world_rank, int participant_index,
                                                                 int layer, int expert, WeightRole role)
    {
        std::unique_lock lock(mutex_);
        Key source_key{{}, device, layer, expert, role};
        source_key.participant_world_rank = -1;
        source_key.participant_index = -1;
        auto source = engines_.find(source_key);
        if (source == engines_.end() || source->second.engine == nullptr)
            return false;

        Key alias_key{domain_name, device, layer, expert, role};
        alias_key.participant_world_rank = participant_world_rank;
        alias_key.participant_index = participant_index;
        const auto previous = engines_.find(alias_key);
        if (previous != engines_.end() && previous->second.projection_ownership != source->second.projection_ownership)
            throw std::logic_error("Prepared expert alias cannot replace another projection layout: " + domain_name);
        engines_[alias_key] = source->second;
        return true;
    }

    ITensorGemm *ExpertGemmRegistry::getEngine(DeviceId device, int layer, int expert, WeightRole role) const
    {
        return getEngineForDomain({}, device, layer, expert, role);
    }

    ITensorGemm *ExpertGemmRegistry::getEngineForDomain(const std::string &domain_name,
                                                        DeviceId device, int layer, int expert, WeightRole role,
                                                        const std::optional<MoEExpertProjectionOwnership> &projection) const
    {
        return getEngineForParticipant(domain_name, device, -1, -1, layer, expert, role, projection);
    }

    ITensorGemm *ExpertGemmRegistry::getEngineForParticipant(const std::string &domain_name,
                                                             DeviceId device, int participant_world_rank, int participant_index,
                                                             int layer, int expert, WeightRole role,
                                                             const std::optional<MoEExpertProjectionOwnership> &projection) const
    {
        std::shared_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        key.participant_world_rank = participant_world_rank;
        key.participant_index = participant_index;
        auto it = engines_.find(key);
        if (it == engines_.end())
            return nullptr;
        if (it->second.static_source)
            throw std::logic_error("Static expert matrices require complete source-authenticated layer acquisition");
        if (it->second.projection_ownership != projection)
            throw std::logic_error("Prepared expert lookup disagrees with projection identity: " + domain_name);
        return it->second.engine;
    }

    std::shared_ptr<ITensorGemm> ExpertGemmRegistry::getEngineLifetime(
        DeviceId device,
        int layer,
        int expert,
        WeightRole role) const
    {
        return getEngineLifetimeForParticipant(
            {}, device, -1, -1, layer, expert, role);
    }

    std::shared_ptr<ITensorGemm>
    ExpertGemmRegistry::getEngineLifetimeForDomain(
        const std::string &domain_name,
        DeviceId device,
        int layer,
        int expert,
        WeightRole role,
        const std::optional<MoEExpertProjectionOwnership> &projection) const
    {
        return getEngineLifetimeForParticipant(
            domain_name, device, -1, -1, layer, expert, role, projection);
    }

    std::shared_ptr<ITensorGemm>
    ExpertGemmRegistry::getEngineLifetimeForParticipant(
        const std::string &domain_name,
        DeviceId device,
        int participant_world_rank,
        int participant_index,
        int layer,
        int expert,
        WeightRole role,
        const std::optional<MoEExpertProjectionOwnership> &projection) const
    {
        std::shared_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        key.participant_world_rank = participant_world_rank;
        key.participant_index = participant_index;
        const auto found = engines_.find(key);
        if (found == engines_.end() || !found->second.engine ||
            !found->second.ownership ||
            found->second.ownership.get() != found->second.engine)
        {
            return nullptr;
        }
        if (found->second.static_source)
            throw std::logic_error("Static expert matrices require complete source-authenticated layer acquisition");
        if (found->second.projection_ownership != projection)
            throw std::logic_error("Prepared expert lifetime disagrees with projection identity: " + domain_name);
        return found->second.ownership;
    }

    bool ExpertGemmRegistry::hasCompleteRole(DeviceId device, int layer, int num_experts, WeightRole role) const
    {
        return hasCompleteRoleForDomain({}, device, layer, num_experts, role);
    }

    bool ExpertGemmRegistry::hasCompleteRoleForDomain(const std::string &domain_name,
                                                      DeviceId device, int layer, int num_experts, WeightRole role) const
    {
        if (num_experts < 0)
            return false;

        std::shared_lock lock(mutex_);
        for (int e = 0; e < num_experts; ++e)
        {
            auto it = engines_.find(Key{domain_name, device, layer, e, role});
            if (it == engines_.end() || it->second.engine == nullptr || it->second.projection_ownership)
                return false;
        }

        return true;
    }

    bool ExpertGemmRegistry::hasCompleteRoleForExperts(DeviceId device, int layer,
                                                       const std::vector<int> &expert_ids,
                                                       WeightRole role) const
    {
        return hasCompleteRoleForExpertsInDomain({}, device, layer, expert_ids, role);
    }

    bool ExpertGemmRegistry::hasCompleteRoleForExpertsInDomain(const std::string &domain_name,
                                                               DeviceId device, int layer,
                                                               const std::vector<int> &expert_ids,
                                                               WeightRole role) const
    {
        std::shared_lock lock(mutex_);
        for (int expert : expert_ids)
        {
            if (expert < 0)
                return false;
            auto it = engines_.find(Key{domain_name, device, layer, expert, role});
            if (it == engines_.end() || it->second.engine == nullptr || it->second.projection_ownership)
                return false;
        }

        return true;
    }

    bool ExpertGemmRegistry::hasCompleteLayer(DeviceId device, int layer, int num_experts) const
    {
        return hasCompleteLayerInDomain({}, device, layer, num_experts);
    }

    bool ExpertGemmRegistry::hasCompleteLayerInDomain(const std::string &domain_name,
                                                      DeviceId device, int layer, int num_experts) const
    {
        if (num_experts < 0)
            return false;

        std::shared_lock lock(mutex_);
        for (int e = 0; e < num_experts; ++e)
        {
            for (WeightRole role : {WeightRole::GATE, WeightRole::UP, WeightRole::DOWN})
            {
                auto it = engines_.find(Key{domain_name, device, layer, e, role});
                if (it == engines_.end() || it->second.engine == nullptr || it->second.projection_ownership)
                    return false;
            }
        }

        return true;
    }

    std::vector<int> ExpertGemmRegistry::completeExpertsForLayer(DeviceId device, int layer, int num_experts) const
    {
        return completeExpertsForLayerInDomain({}, device, layer, num_experts);
    }

    std::vector<int> ExpertGemmRegistry::completeExpertsForLayerInDomain(const std::string &domain_name,
                                                                         DeviceId device, int layer, int num_experts) const
    {
        std::vector<int> experts;
        if (num_experts < 0)
            return experts;

        std::shared_lock lock(mutex_);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            bool complete = true;
            for (WeightRole role : {WeightRole::GATE, WeightRole::UP, WeightRole::DOWN})
            {
                auto it = engines_.find(Key{domain_name, device, layer, expert, role});
                if (it == engines_.end() || it->second.engine == nullptr || it->second.projection_ownership)
                {
                    complete = false;
                    break;
                }
            }
            if (complete)
                experts.push_back(expert);
        }
        return experts;
    }

    size_t ExpertGemmRegistry::countCompleteExpertsForLayer(DeviceId device, int layer, int num_experts) const
    {
        return completeExpertsForLayer(device, layer, num_experts).size();
    }

    size_t ExpertGemmRegistry::countCompleteExpertsForLayerInDomain(const std::string &domain_name,
                                                                    DeviceId device, int layer, int num_experts) const
    {
        return completeExpertsForLayerInDomain(domain_name, device, layer, num_experts).size();
    }

    size_t ExpertGemmRegistry::countEnginesForDevice(DeviceId device) const
    {
        return countEnginesForDeviceInDomain({}, device);
    }

    size_t ExpertGemmRegistry::countEnginesForDeviceInDomain(const std::string &domain_name, DeviceId device) const
    {
        std::shared_lock lock(mutex_);
        size_t count = 0;
        for (const auto &[key, entry] : engines_)
        {
            if (key.domain_name == domain_name && key.device == device && entry.engine != nullptr)
                ++count;
        }
        return count;
    }

    size_t ExpertGemmRegistry::countOwnedEnginesForDeviceAcrossScopes(
        DeviceId device) const
    {
        std::shared_lock lock(mutex_);
        size_t count = 0;
        for (const auto &[key, entry] : engines_)
        {
            if (key.device == device && entry.engine != nullptr &&
                entry.ownership)
            {
                ++count;
            }
        }
        return count;
    }

    size_t ExpertGemmRegistry::countEnginesForLayer(DeviceId device, int layer) const
    {
        return countEnginesForLayerInDomain({}, device, layer);
    }

    size_t ExpertGemmRegistry::countEnginesForLayerInDomain(const std::string &domain_name, DeviceId device, int layer) const
    {
        std::shared_lock lock(mutex_);
        size_t count = 0;
        for (const auto &[key, entry] : engines_)
        {
            if (key.domain_name == domain_name && key.device == device && key.layer == layer && entry.engine != nullptr)
                ++count;
        }
        return count;
    }

    bool ExpertGemmRegistry::populateExpertEngines(DeviceId device, int layer, int num_experts,
                                                   std::vector<ITensorGemm *> &gate_out,
                                                   std::vector<ITensorGemm *> &up_out,
                                                   std::vector<ITensorGemm *> &down_out) const
    {
        return populateExpertEnginesForDomain({}, device, layer, num_experts,
                                              gate_out, up_out, down_out);
    }

    bool ExpertGemmRegistry::populateExpertEnginesForDomain(const std::string &domain_name,
                                                            DeviceId device, int layer, int num_experts,
                                                            std::vector<ITensorGemm *> &gate_out,
                                                            std::vector<ITensorGemm *> &up_out,
                                                            std::vector<ITensorGemm *> &down_out) const
    {
        return populateExpertEnginesForParticipant(
            domain_name, device, -1, -1, layer, num_experts,
            gate_out, up_out, down_out);
    }

    bool ExpertGemmRegistry::populateExpertEnginesForParticipant(const std::string &domain_name,
                                                                 DeviceId device, int participant_world_rank, int participant_index,
                                                                 int layer, int num_experts,
                                                                 std::vector<ITensorGemm *> &gate_out,
                                                                 std::vector<ITensorGemm *> &up_out,
                                                                 std::vector<ITensorGemm *> &down_out) const
    {
        if (num_experts < 0)
        {
            gate_out.clear();
            up_out.clear();
            down_out.clear();
            return false;
        }

        gate_out.assign(static_cast<size_t>(num_experts), nullptr);
        up_out.assign(static_cast<size_t>(num_experts), nullptr);
        down_out.assign(static_cast<size_t>(num_experts), nullptr);

        std::shared_lock lock(mutex_);
        bool complete = true;

        for (int e = 0; e < num_experts; ++e)
        {
            Key gate_key{domain_name, device, layer, e, WeightRole::GATE};
            gate_key.participant_world_rank = participant_world_rank;
            gate_key.participant_index = participant_index;
            auto it_gate = engines_.find(gate_key);
            if (it_gate != engines_.end() && it_gate->second.projection_ownership)
                throw std::logic_error("Whole-expert population cannot consume a projected gate/up bank");
            if (it_gate != engines_.end())
                gate_out[e] = it_gate->second.engine;

            Key up_key{domain_name, device, layer, e, WeightRole::UP};
            up_key.participant_world_rank = participant_world_rank;
            up_key.participant_index = participant_index;
            auto it_up = engines_.find(up_key);
            if (it_up != engines_.end() && it_up->second.projection_ownership)
                throw std::logic_error("Whole-expert population cannot consume a projected gate/up bank");
            if (it_up != engines_.end())
                up_out[e] = it_up->second.engine;

            Key down_key{domain_name, device, layer, e, WeightRole::DOWN};
            down_key.participant_world_rank = participant_world_rank;
            down_key.participant_index = participant_index;
            auto it_down = engines_.find(down_key);
            if (it_down != engines_.end() && it_down->second.projection_ownership)
                throw std::logic_error("Whole-expert population cannot consume a projected down bank");
            if (it_down != engines_.end())
                down_out[e] = it_down->second.engine;

            if (gate_out[e] == nullptr || up_out[e] == nullptr || down_out[e] == nullptr)
                complete = false;
        }

        return complete;
    }

    bool ExpertGemmRegistry::replaceParticipantResidency(
        std::span<const ParticipantLayerScope> scopes,
        std::span<const ParticipantExpertBinding> bindings,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        try
        {
            using ScopeIdentity =
                std::tuple<std::string, std::string, int, int, int>;
            const auto scope_identity = [](const ParticipantLayerScope &scope)
            {
                return ScopeIdentity{
                    scope.domain_name,
                    scope.device.to_string(),
                    scope.participant_world_rank,
                    scope.participant_index,
                    scope.layer,
                };
            };

            std::set<ScopeIdentity> declared_scopes;
            for (const auto &scope : scopes)
            {
                if (scope.domain_name.empty() || !scope.device.is_valid() ||
                    scope.participant_world_rank < -1 ||
                    scope.participant_index < 0 || scope.layer < 0 ||
                    !declared_scopes.emplace(scope_identity(scope)).second)
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay registry replacement contains an invalid or duplicate participant/layer scope";
                    }
                    return false;
                }
            }
            if (declared_scopes.empty())
            {
                if (error)
                    *error = "ExpertOverlay registry replacement has no scopes";
                return false;
            }

            std::unordered_map<Key, Entry, KeyHash> participant_entries;
            std::unordered_map<Key, Entry, KeyHash> domain_entries;
            participant_entries.reserve(bindings.size() * 3u);
            domain_entries.reserve(bindings.size() * 3u);
            for (const auto &binding : bindings)
            {
                if (binding.expert < 0 || !binding.complete() ||
                    declared_scopes.count(
                        scope_identity(binding.scope)) == 0u)
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay registry replacement has an incomplete or out-of-scope expert binding";
                    }
                    return false;
                }

                const auto add_role = [&](WeightRole role,
                                          const std::shared_ptr<ITensorGemm> &engine)
                {
                    Key participant_key{
                        binding.scope.domain_name,
                        binding.scope.device,
                        binding.scope.layer,
                        binding.expert,
                        role,
                    };
                    participant_key.participant_world_rank =
                        binding.scope.participant_world_rank;
                    participant_key.participant_index =
                        binding.scope.participant_index;
                    const Entry entry{engine.get(), engine};
                    if (!participant_entries.emplace(
                            participant_key, entry).second)
                    {
                        return false;
                    }

                    Key domain_key{
                        binding.scope.domain_name,
                        binding.scope.device,
                        binding.scope.layer,
                        binding.expert,
                        role,
                    };
                    const auto [found, inserted] =
                        domain_entries.emplace(domain_key, entry);
                    return inserted ||
                           (found->second.engine == entry.engine &&
                            found->second.ownership.get() ==
                                entry.ownership.get());
                };

                if (!add_role(WeightRole::GATE, binding.gate) ||
                    !add_role(WeightRole::UP, binding.up) ||
                    !add_role(WeightRole::DOWN, binding.down))
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay registry replacement repeats a participant expert or gives one domain alias conflicting engines";
                    }
                    return false;
                }
            }

            std::unique_lock lock(mutex_);
            auto replacement = engines_;
            for (auto entry = replacement.begin();
                 entry != replacement.end();)
            {
                const auto &key = entry->first;
                const bool participant_match = std::any_of(
                    scopes.begin(),
                    scopes.end(),
                    [&](const auto &scope)
                    {
                        return key.domain_name == scope.domain_name &&
                               key.device == scope.device &&
                               key.layer == scope.layer &&
                               key.participant_world_rank ==
                                   scope.participant_world_rank &&
                               key.participant_index ==
                                   scope.participant_index;
                    });
                const bool domain_match =
                    key.participant_world_rank == -1 &&
                    key.participant_index == -1 &&
                    std::any_of(
                        scopes.begin(),
                        scopes.end(),
                        [&](const auto &scope)
                        {
                            return key.domain_name == scope.domain_name &&
                                   key.device == scope.device &&
                                   key.layer == scope.layer;
                        });
                if ((participant_match || domain_match) && entry->second.projection_ownership)
                    throw std::logic_error("Whole-expert residency replacement cannot retire fixed projection slices");
                if (participant_match || domain_match)
                    entry = replacement.erase(entry);
                else
                    ++entry;
            }
            replacement.insert(
                participant_entries.begin(), participant_entries.end());
            replacement.insert(domain_entries.begin(), domain_entries.end());
            engines_.swap(replacement);
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
            {
                *error =
                    "ExpertOverlay registry replacement failed before publication: " +
                    std::string(exception.what());
            }
        }
        catch (...)
        {
            if (error)
            {
                *error =
                    "ExpertOverlay registry replacement failed before publication with a non-standard exception";
            }
        }
        return false;
    }

    void ExpertGemmRegistry::replaceEngine(DeviceId device, int layer, int expert, WeightRole role,
                                           ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership)
    {
        replaceEngineForDomain({}, device, layer, expert, role, engine, std::move(ownership));
    }

    void ExpertGemmRegistry::replaceEngineForDomain(const std::string &domain_name,
                                                    DeviceId device, int layer, int expert, WeightRole role,
                                                    ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership)
    {
        std::unique_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        const auto previous = engines_.find(key);
        if (previous != engines_.end() && previous->second.projection_ownership)
            throw std::logic_error("Whole-expert replacement cannot change a projected engine");
        engines_[key] = Entry{engine, std::move(ownership)};
    }

    bool ExpertGemmRegistry::removeEngine(DeviceId device, int layer, int expert, WeightRole role)
    {
        return removeEngineForDomain({}, device, layer, expert, role);
    }

    bool ExpertGemmRegistry::removeEngineForDomain(const std::string &domain_name,
                                                   DeviceId device, int layer, int expert, WeightRole role)
    {
        std::unique_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        const auto previous = engines_.find(key);
        if (previous != engines_.end() && previous->second.projection_ownership)
            throw std::logic_error("Whole-expert departure cannot retire a fixed projection layout");
        return engines_.erase(key) > 0;
    }

    size_t ExpertGemmRegistry::size() const
    {
        std::shared_lock lock(mutex_);
        return engines_.size();
    }

    bool ExpertGemmRegistry::hasEnginesForLayer(DeviceId device, int layer) const
    {
        return hasEnginesForLayerInDomain({}, device, layer);
    }

    bool ExpertGemmRegistry::hasEnginesForLayerInDomain(const std::string &domain_name, DeviceId device, int layer) const
    {
        std::shared_lock lock(mutex_);
        for (const auto &[key, entry] : engines_)
        {
            if (key.domain_name == domain_name && key.device == device && key.layer == layer)
                return true;
        }
        return false;
    }

    void ExpertGemmRegistry::clear()
    {
        std::unique_lock lock(mutex_);
        engines_.clear();
    }

} // namespace llaminar2
