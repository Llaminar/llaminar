/**
 * @file CapturedTransferChannel.cpp
 * @brief TransferEngine lowering of admitted rank-local captured message channels.
 *
 * Setup claims every route-specific physical allocation through PMA, resolves kernels,
 * then joins the exact cursor-initialization events once before returning ready.
 * Live submission only validates frozen ownership and records acquire, parallel
 * byte copy and release. No live device epoch is read or reset by this host code.
 */
#include "CapturedTransferChannel.h"
#include "CapturedTransferChannelBatch.h"
#include "PipelineActivationFanoutProtocol.h"
#include "CapturedTransferKernelPlan.h"
#include "planning/PlanningPublication.h"
#include "planning/RankHardwareOwnership.h"
#include "execution/moe/MoEOverlaySharedBackingStartup.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <cerrno>
#include <system_error>
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include "tensors/TensorClasses.h"

#include <atomic>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>

namespace llaminar2
{
    namespace
    {
        /** @return A never-reused allocation identity, not a request/generation clock. */
        std::uint64_t nextChannelIdentity()
        {
            static std::atomic<std::uint64_t> next{1};
            auto value = next.load(std::memory_order_relaxed);
            do
            {
                if (value == kCapturedTransferAbortEpoch)
                    throw std::overflow_error("Captured transfer allocation identity exhausted");
            } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
            return value;
        }

        /** @return One validated endpoint index; unknown enum values fail before native calls. */
        size_t endpointIndex(CapturedTransferEndpoint role)
        {
            switch (role)
            {
            case CapturedTransferEndpoint::Producer: return 0;
            case CapturedTransferEndpoint::Consumer: return 1;
            }
            throw std::invalid_argument("Captured transfer has an invalid endpoint role");
        }

        /** @brief Reject misaligned count storage or aliasing with payload capacity.
         * @param count Exact device address of a retained uint64 count.
         * @param payload Exact start of the bound device payload.
         * @param capacity Capture-frozen maximum payload extent. */
        void validateExtentAddress(void *count, void *payload, size_t capacity,
            size_t count_bytes = sizeof(std::uint64_t), size_t alignment = alignof(std::uint64_t))
        {
            const auto address = reinterpret_cast<std::uintptr_t>(count);
            const auto start = reinterpret_cast<std::uintptr_t>(payload);
            if (address % alignment != 0)
                throw std::invalid_argument("Device-counted transfer extent must be uint64 aligned");
            // Difference comparisons avoid overflowing one-past-end arithmetic.
            if (address >= start ? address - start < capacity : start - address < count_bytes)
                throw std::invalid_argument("Device-counted transfer extent overlaps its payload capacity");
        }
    }

    CapturedTransferRankPlan CapturedTransferRankPlan::fromInventory(const ClusterInventory &inventory,
        CapturedTransferRankChannelId id, CapturedTransferRankGPU producer,
        CapturedTransferRankGPU consumer, size_t capacity, CapturedTransferEndpoint host_endpoint)
    {
        (void)endpointIndex(host_endpoint);
        if (!id.generation || !id.boundary || producer.rank == consumer.rank || inventory.world_size <= 0 ||
            inventory.ranks.size() != size_t(inventory.world_size))
            throw std::invalid_argument("Rank channel requires original positive generation/key and distinct rank roles");
        const auto connection = inventory.connectionBetweenRanks(producer.rank, consumer.rank);
        if (connection.locality() != RankConnectionLocality::SameNode)
            throw std::invalid_argument("Rank captured channel requires authenticated same-node original ranks");
        std::array<std::string, 2> uuids;
        const std::array roles{producer, consumer};
        for (size_t index = 0; index < roles.size(); ++index)
        {
            const auto role = roles[index];
            const auto &original = inventory.getRank(role.rank);
            requireRankLocalGPUOwnership(original, role.device, "Rank captured channel endpoint");
            const auto found = std::find_if(original.gpus.begin(), original.gpus.end(), [&](const auto &gpu) {
                return gpu.type == role.device.type && gpu.local_device_id == role.device.ordinal;
            });
            if (found == original.gpus.end() || found->uuid.empty())
                throw std::invalid_argument("Rank channel endpoint lacks its original physical GPU identity");
            uuids[index] = found->uuid;
        }
        (void)CapturedTransferChannel::memoryFor(capacity);
        return {id, producer, consumer, connection.sourceNode(), capacity, std::move(uuids), host_endpoint};
    }

    CapturedTransferRankGPU CapturedTransferRankPlan::endpoint(CapturedTransferEndpoint role) const
    { return endpoints_[endpointIndex(role)]; }

    std::optional<CapturedTransferEndpoint> CapturedTransferRankPlan::localRole(int rank) const noexcept
    {
        if (rank == endpoints_[0].rank) return CapturedTransferEndpoint::Producer;
        if (rank == endpoints_[1].rank) return CapturedTransferEndpoint::Consumer;
        return std::nullopt;
    }

    std::vector<std::uint8_t> CapturedTransferRankPlan::serialize() const
    {
        const auto text = nlohmann::json{{"generation", id_.generation}, {"boundary", id_.boundary},
            {"node", node_}, {"capacity", capacity_}, {"host_endpoint", int(host_endpoint_)},
            {"producer", {endpoints_[0].rank, int(endpoints_[0].device.type), endpoints_[0].device.ordinal, uuids_[0]}},
            {"consumer", {endpoints_[1].rank, int(endpoints_[1].device.type), endpoints_[1].device.ordinal, uuids_[1]}}}.dump();
        return {text.begin(), text.end()};
    }

    void CapturedTransferRankPlan::contributeMemory(CapturedTransferEndpoint role, PhysicalMemoryResource gpu,
        std::optional<PhysicalMemoryResource> host, PhysicalMemoryPlanBuilder &builder) const
    {
        const auto original = endpoint(role);
        if (gpu.world_rank != original.rank || gpu.device != original.device ||
            (role == host_endpoint_
                ? !host || host->world_rank != original.rank || host->device != DeviceId::cpu()
                : host.has_value()))
            throw std::invalid_argument("Rank channel BOM does not match its original GPU/host owner");
        const auto geometry = CapturedTransferChannel::memoryFor(capacity_);
        constexpr auto owner = PhysicalMemoryOwner::ActivationTransportStaging;
        if (host) builder.add(*host, owner, geometry.mapped_host_bytes);
        builder.add(gpu, owner, geometry.cursor_bytes_per_device);
    }

    namespace
    {
        /** @brief Original authenticated node-local shared backing, retained through alias teardown.
         * The declared allocator owns the file name; a peer only closes its descriptor/mapping.
         * No object state or process-local pointer is sent to the other rank.
         */
        class RankCapturedPages final
        {
        public:
            /** @brief Authenticate and map one original exact file; declared allocator initializes control once.
             * @param plan Same positive immutable declaration used by PMA.
             * @param mpi Original context, only the two actual endpoint ranks enter this P2P step.
             * @param creator_identity Nonzero only on the original host allocator that minted this identity.
             */
            RankCapturedPages(const CapturedTransferRankPlan &plan, const IMPIContext &mpi,
                CapturedTransferChannelIdentity creator_identity)
                : bytes_(CapturedTransferChannel::memoryFor(plan.capacityBytes()).mapped_host_bytes)
            {
                const auto source = plan.endpoint(CapturedTransferEndpoint::Producer);
                const auto target = plan.endpoint(CapturedTransferEndpoint::Consumer);
                if (!mpi.topology()) throw std::invalid_argument("Rank channel has no original node topology");
                std::array<int, 2> participants{source.rank, target.rank};
                std::sort(participants.begin(), participants.end());
                const auto id = plan.originalId();
                const auto name = "pipeline-channel-" + std::to_string(id.generation) + "-" +
                    std::to_string(id.boundary) + "-" + std::to_string(source.rank) + "-" + std::to_string(target.rank);
                file_ = establishMoEOverlaySharedBacking(mpi,
                    {.kind = MoEOverlaySharedBackingKind::PipelineActivation,
                     .node_namespace = mpi.topology()->node_shared_memory_namespace(), .generation = id.generation,
                     .fingerprint_low = id.boundary,
                     .fingerprint_high = (std::uint64_t(source.rank) << 32) | std::uint32_t(target.rank)},
                    plan.hostRank(), participants, name, bytes_);
                file_.requireExtent(bytes_);
                if (mpi.rank() == plan.hostRank() && !creator_identity.valid())
                    throw std::logic_error("Rank source has no original allocation identity");
                allocation_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, file_.descriptor(), 0);
                if (allocation_ == MAP_FAILED)
                {
                    allocation_ = nullptr;
                    throw std::system_error(errno, std::generic_category(), "Cannot map original rank channel backing");
                }
                if (mpi.rank() == plan.hostRank())
                {
                    // Complete first touch remains the declared host allocator's work.
                    std::memset(allocation_, 0, bytes_);
                    auto *control = ::new (allocation_) CapturedTransferChannelControl{};
                    control->identity.capacity = creator_identity.capacity;
                    std::atomic_ref(control->identity.nonce).store(creator_identity.nonce, std::memory_order_release);
                }
            }
            /** @brief Unmap only after native graphs/aliases retire; SharedFile performs creator-only unlink. */
            ~RankCapturedPages() { if (allocation_ && ::munmap(allocation_, bytes_) != 0) std::terminate(); }
            /** @return Exact local process alias, never another process's pointer. */
            void *allocation() const noexcept { return allocation_; }
            /** @return Actual original file identity carried into the native capture seal. */
            std::array<std::uint64_t, 3> identity() const
            {
                const auto original = file_.identity();
                return {original.device, original.inode, original.uid};
            }
        private:
            MPISessionOwner::SharedFile file_;
            size_t bytes_;
            void *allocation_ = nullptr;
        };
    }

    struct CapturedTransferRankSession::Entry final
    {
        PhysicalMemoryAllocationLease host_lease; ///< Released only after original global unmap consensus.
        std::shared_ptr<RankCapturedPages> pages; ///< Retains each original file/map beyond every local GPU alias.
        std::weak_ptr<CapturedTransferChannel> channel; ///< A live original graph binding prevents retirement.
    };

    std::shared_ptr<CapturedTransferRankSession> CapturedTransferRankSession::create(std::shared_ptr<IMPIContext> original)
    {
        if (!original) throw std::invalid_argument("Rank channel session requires its original shared MPI owner");
        std::uint64_t generation = 0;
        std::shared_ptr<CapturedTransferRankSession> result;
        // Reuse the existing allocation-identity authority once for the whole
        // enclosing cold cohort. No request/epoch counter or communicator is added.
        exchangePlanningSamples(original, [](int count) {
            const auto text = nlohmann::json(nextChannelIdentity()).dump();
            return std::vector<std::vector<uint8_t>>(count, std::vector<uint8_t>(text.begin(), text.end()));
        }, [&](auto identity) {
            const auto value = nlohmann::json::parse(identity);
            if (!value.is_number_unsigned() || !(generation = value.template get<std::uint64_t>()) ||
                generation == kCapturedTransferAbortEpoch)
                throw std::invalid_argument("Original rank channel session has an invalid allocation scope");
            result = std::shared_ptr<CapturedTransferRankSession>(new CapturedTransferRankSession(original, generation));
            return std::vector<uint8_t>(identity.begin(), identity.end());
        }, [&](auto receipts) {
            for (const auto &receipt : receipts)
                if (nlohmann::json::parse(receipt.bytes) != generation)
                    throw std::invalid_argument("Original rank channel session scope disagrees");
        });
        // topology() is lazy and may construct original WORLD collectives.
        // Every rank initializes it here, never inside a two-role file branch
        // while uninvolved ranks enter another phase's completion collective.
        exchangePlanningSamples(original, [](int count) {
            return std::vector<std::vector<uint8_t>>(count, std::vector<uint8_t>{1});
        }, [&](auto identity) {
            // No asymmetric validation/role branch precedes lazy discovery:
            // its original internal WORLD collectives require every rank.
            const auto *topology = original->topology();
            if (identity.size() != 1 || identity[0] != 1 || !topology ||
                !topology->node_shared_memory_namespace())
                throw std::invalid_argument("Original rank channel session has no authenticated node topology");
            return std::vector<uint8_t>{1};
        }, [](auto) {});
        return result;
    }

    CapturedTransferRankSession::~CapturedTransferRankSession()
    {
        if (!retired_ && !entries_.empty())
        {
            // Never free a source charge while an original remote mapping may
            // still exist. Existing process/session fatal cleanup owns this edge.
            std::fputs("[FATAL] Original rank channel session abandoned before graph/alias retirement consensus\n", stderr);
            std::terminate();
        }
    }

    void CapturedTransferRankSession::requireOriginalContext(const std::shared_ptr<IMPIContext> &original) const
    {
        if (retired_ || !original || original.get() != original_.get() || original.owner_before(original_) ||
            original_.owner_before(original))
            throw std::invalid_argument("Rank channel session requires the same original MPI ownership control block");
    }

    void CapturedTransferRankSession::retireAfterOriginalGraphRelease()
    {
        if (retired_) return;
        const auto phase = [&](const auto &operation) {
            exchangePlanningSamples(original_, [&](int count) {
                const auto text = nlohmann::json(generation_).dump();
                return std::vector<std::vector<uint8_t>>(count, std::vector<uint8_t>(text.begin(), text.end()));
            }, [&](auto identity) {
                if (nlohmann::json::parse(identity) != generation_)
                    throw std::invalid_argument("Original rank channel retirement cohort disagrees");
                operation();
                return std::vector<uint8_t>{1};
            }, [](auto receipts) {
                for (const auto &receipt : receipts)
                    if (receipt.bytes.size() != 1 || receipt.bytes[0] != 1)
                        throw std::invalid_argument("Original rank channel retirement omitted an endpoint");
            });
        };
        phase([&] {
            for (const auto &entry : entries_)
                if (!entry->channel.expired())
                    throw std::logic_error("Original rank channel graph/alias borrow is still live at model retirement");
        });
        // Every original channel/registration is gone BEFORE any process
        // unmaps/closes its file. The completion consensus then proves no peer
        // VMA/FD can keep source-owned physical pages live behind a zero claim.
        phase([&] { for (const auto &entry : entries_) entry->pages.reset(); });
        phase([&] { for (const auto &entry : entries_) entry->host_lease = PhysicalMemoryAllocationLease{}; });
        entries_.clear();
        retired_ = true;
    }

    std::shared_ptr<CapturedTransferChannel> TransferEngine::createRankCapturedTransferChannel(
        const CapturedTransferRankPlan &plan, const std::shared_ptr<CapturedTransferRankSession> &session,
        const std::shared_ptr<PhysicalMemoryAuthority> &local_authority, void *local_setup_stream) const
    {
        if (!session) throw std::invalid_argument("Rank channel setup requires its original model retirement owner");
        const auto mpi = session->original_;
        std::shared_ptr<CapturedTransferRankSession::Entry> retained;
        std::shared_ptr<CapturedTransferChannel> channel;
        std::shared_ptr<RankCapturedPages> pages;
        std::optional<CapturedTransferEndpoint> role;
        const auto phase = [&](const auto &execute) {
            exchangePlanningSamples(mpi, [&](int count) {
                return std::vector<std::vector<uint8_t>>(count, plan.serialize());
            }, [&](auto identity) {
                if (std::vector<uint8_t>(identity.begin(), identity.end()) != plan.serialize())
                    throw std::invalid_argument("Original rank channel declarations disagree");
                execute();
                return std::vector<uint8_t>{1};
            }, [](auto receipts) {
                for (const auto &receipt : receipts)
                    if (receipt.bytes.size() != 1 || receipt.bytes[0] != 1)
                        throw std::invalid_argument("Rank channel setup has an incomplete original receipt");
            });
        };
        phase([&] {
            if (session->retired_ || plan.originalId().generation != session->generation_ || !mpi || isGraphCaptureActive())
                throw std::invalid_argument("Original rank channel construction requires its existing context outside capture");
            const auto inventory = mpi->clusterInventory();
            if (!inventory || plan != CapturedTransferRankPlan::fromInventory(*inventory, plan.originalId(),
                    plan.endpoint(CapturedTransferEndpoint::Producer), plan.endpoint(CapturedTransferEndpoint::Consumer), plan.capacityBytes(), plan.hostEndpoint()))
                throw std::invalid_argument("Rank channel does not match the original discovery publication");
            role = plan.localRole(mpi->rank());
            if (!role)
            {
                if (local_authority || local_setup_stream)
                    throw std::invalid_argument("Uninvolved rank cannot own a channel GPU or PMA claim");
                return;
            }
            if (!local_authority || local_authority->worldRank() != mpi->rank() ||
                reinterpret_cast<std::uintptr_t>(local_setup_stream) <= 2)
                throw std::invalid_argument("Rank endpoint requires its original stage PMA and exact local stream");
            const auto geometry = CapturedTransferChannel::memoryFor(plan.capacityBytes());
            retained = std::make_shared<CapturedTransferRankSession::Entry>();
            session->entries_.push_back(retained);
            channel = std::shared_ptr<CapturedTransferChannel>(new CapturedTransferChannel);
            retained->channel = channel;
            channel->rank_plan_ = plan;
            channel->rank_role_ = role;
            for (const auto endpoint_role : {CapturedTransferEndpoint::Producer, CapturedTransferEndpoint::Consumer})
                channel->endpoints_[endpointIndex(endpoint_role)].device = plan.endpoint(endpoint_role).device;
            auto &endpoint = channel->endpoints_[endpointIndex(*role)];
            endpoint.original.emplace(GPUDeviceContextPool::instance().retainExistingContextProvenance(endpoint.device));
            if (endpoint.original->worker()->ownsCurrentThread())
                throw std::invalid_argument("Rank channel construction requires the original setup caller");
            endpoint.backend = resolveBackend(endpoint.device);
            endpoint.setup_stream = local_setup_stream;
            const auto validate = [&] {
                (void)endpoint.original->worker()->queryStreamExecutionState(local_setup_stream, "rank channel original setup stream");
            };
            endpoint.original->worker()->submitAndWait(validate);
            endpoint.runtime_generation = endpoint.backend ? endpoint.backend->deviceRuntimeGeneration(endpoint.device.ordinal) : 0;
            if (!endpoint.runtime_generation) throw std::invalid_argument("Rank endpoint has no original native runtime");
            constexpr auto owner = PhysicalMemoryOwner::ActivationTransportStaging;
            if (*role == plan.hostEndpoint())
            {
                retained->host_lease = local_authority->claimNewAllocation(DeviceId::cpu(), owner, geometry.mapped_host_bytes);
                channel->identity_ = {nextChannelIdentity(), plan.capacityBytes()};
            }
            endpoint.lease = local_authority->claimNewAllocation(endpoint.device, owner, geometry.cursor_bytes_per_device);
            if (!endpoint.backend->prepareCapturedTransferChannelKernels(endpoint.device.ordinal,
                    collective_timeout_policy::kDefaultCollectiveTimeoutMs, &endpoint.timeout_ticks) || !endpoint.timeout_ticks)
                throw std::runtime_error("Original rank endpoint kernel preparation failed");
        });
        phase([&] {
            if (role)
            {
                pages = std::make_shared<RankCapturedPages>(plan, *mpi, channel->identity_);
                retained->pages = pages;
            }
        });
        phase([&] {
            if (!role) return;
            auto *control = static_cast<CapturedTransferChannelControl *>(pages->allocation());
            const auto nonce = std::atomic_ref(control->identity.nonce).load(std::memory_order_acquire);
            if (!nonce || nonce == kCapturedTransferAbortEpoch || control->identity.capacity != plan.capacityBytes())
                throw std::invalid_argument("Rank endpoint acquired an incomplete shared allocation identity");
            channel->identity_ = {nonce, control->identity.capacity};
            channel->backing_identity_ = pages->identity();
            auto &endpoint = channel->endpoints_[endpointIndex(*role)];
            const std::array devices{endpoint.device};
            const auto geometry = CapturedTransferChannel::memoryFor(plan.capacityBytes());
            channel->mapped_ = registerExternalMappedHostRegion(pages->allocation(), geometry.mapped_host_bytes, devices, pages);
            endpoint.cursor = allocateDeviceTransferBuffer(geometry.cursor_bytes_per_device, endpoint.device);
            endpoint.initialization_deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
            endpoint.initialization = CapturedTransferChannel::Endpoint::Initialization::Pending;
            if (!endpoint.backend->memset(endpoint.cursor->mutableDeviceData(), 0, geometry.cursor_bytes_per_device,
                    endpoint.device.ordinal, local_setup_stream))
                throw std::runtime_error("Original rank endpoint cursor initialization failed");
            CapturedTransferChannel::completeColdInitialization(endpoint);
        });
        return channel;
    }

    CapturedTransferChannelMemory CapturedTransferChannel::memoryFor(size_t capacity)
    {
        if (!capacity) throw std::invalid_argument("Captured transfer needs a positive payload capacity");
        if (capacity > std::numeric_limits<size_t>::max() - payload_offset)
            throw std::overflow_error("Captured transfer payload/control size overflow");
        return {TransferEngine::mappedHostRegionAllocationBytes(payload_offset + capacity),
            sizeof(CapturedTransferCursor)};
    }

    void CapturedTransferChannel::beginInvocation(CapturedTransferInvocation original)
    {
        if (!original.valid()) throw std::invalid_argument("Captured channel requires an original admission identity");
        if (rank_role_) requireOriginalEndpoint(*rank_role_);
        else
        {
            requireOriginalEndpoint(CapturedTransferEndpoint::Producer);
            requireOriginalEndpoint(CapturedTransferEndpoint::Consumer);
        }
        std::lock_guard lock(invocation_mutex_);
        if (terminal_cancelled_ || invocation_active_ ||
            original.generation < original_invocation_.generation ||
            (original.generation == original_invocation_.generation &&
                original.invocation <= original_invocation_.invocation))
            throw std::logic_error("Captured channel admission is stale, overlapping or terminal");
        original_invocation_ = original;
        invocation_active_ = true;
    }

    CapturedTransferInvocationRetirement CapturedTransferChannel::finishInvocation(CapturedTransferInvocation original)
    {
        std::lock_guard lock(invocation_mutex_);
        if (!invocation_active_ || original != original_invocation_)
            throw std::logic_error("Captured channel retirement lost its original admission");
        invocation_active_ = false;
        return terminal_cancelled_ ? CapturedTransferInvocationRetirement::Cancelled
            : CapturedTransferInvocationRetirement::Completed;
    }

    bool CapturedTransferChannel::cancelInvocation(CapturedTransferInvocation original)
    {
        std::lock_guard lock(invocation_mutex_);
        if (original != original_invocation_) return false;
        if (terminal_cancelled_) return true;
        if (!invocation_active_) return false;
        if (!terminal_cancelled_)
        {
            auto *control = static_cast<CapturedTransferChannelControl *>(mapped_->mutableHostData());
            std::atomic_ref<std::uint64_t>(control->cancelled_nonce).store(identity_.nonce, std::memory_order_release);
            terminal_cancelled_ = true;
        }
        return true;
    }

    void TransferEngine::beginCapturedTransferInvocation(const std::shared_ptr<CapturedTransferChannel> &channel,
        CapturedTransferInvocation original) const
    {
        if (!channel) throw std::invalid_argument("Captured channel admission requires its original owner");
        channel->beginInvocation(original);
    }

    CapturedTransferInvocationRetirement TransferEngine::finishCapturedTransferInvocation(const std::shared_ptr<CapturedTransferChannel> &channel,
        CapturedTransferInvocation original) const
    {
        if (!channel) throw std::invalid_argument("Captured channel retirement requires its original owner");
        return channel->finishInvocation(original);
    }

    CapturedTransferReceiveCompletion TransferEngine::capturedReceiveCompletion(
        const CapturedTransferBinding& binding, void* stream) const
    {
        if (binding.role() != CapturedTransferEndpoint::Consumer)
            throw std::invalid_argument("Captured receive completion requires its original consumer binding");
        const auto endpoint = resolveCapturedTransferEndpoint(binding, stream, nullptr);
        return CapturedTransferReceiveCompletion(endpoint.binding);
    }

    bool TransferEngine::cancelCapturedTransferInvocation(const std::shared_ptr<CapturedTransferChannel> &channel,
        CapturedTransferInvocation original) const
    {
        if (!channel) throw std::invalid_argument("Captured channel cancellation requires its original owner");
        return channel->cancelInvocation(original);
    }

    void CapturedTransferChannel::completeColdInitialization(Endpoint& endpoint)
    {
        if(endpoint.initialization!=Endpoint::Initialization::Pending || !endpoint.original || !endpoint.setup_stream)
            throw std::logic_error("Cold channel initialization requires its original pending stream");
        const auto& worker=endpoint.original->worker();
        const auto deadline=endpoint.initialization_deadline;
        for(;;){
            GPUStreamExecutionState state=GPUStreamExecutionState::Pending;
            const auto query=[&]{state=worker->queryStreamExecutionState(endpoint.setup_stream,"captured channel cold initialization");};
            if(worker->ownsCurrentThread())query();else worker->submitAndWait(query);
            if(state==GPUStreamExecutionState::Complete){endpoint.initialization=Endpoint::Initialization::Ready;return;}
            if(state!=GPUStreamExecutionState::Pending)throw std::runtime_error("Captured channel original setup stream failed");
            if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Captured channel cold setup deadline expired");
            std::this_thread::yield();
        }
    }
    void CapturedTransferChannel::requireOriginalEndpoint(CapturedTransferEndpoint role)const
    {
        const auto& endpoint=endpoints_[endpointIndex(role)];
        if (rank_role_ && role != *rank_role_)
            throw std::invalid_argument("Rank channel cannot bind or execute the original peer's endpoint");
        if(endpoint.initialization!=Endpoint::Initialization::Ready || !endpoint.original)
            throw std::logic_error("Captured channel endpoint was not originally initialized");
        const auto current=GPUDeviceContextPool::instance().retainExistingContextProvenance(endpoint.device);
        endpoint.original->requireWorker(current.worker());
        if(current.generation()!=endpoint.original->generation() || current.device()!=endpoint.device ||
           !endpoint.backend || endpoint.backend->deviceRuntimeGeneration(endpoint.device.ordinal)!=endpoint.runtime_generation)
            throw std::logic_error("Captured channel original worker/runtime generation changed");
    }
    CapturedTransferChannel::~CapturedTransferChannel()
    {
        // Ready channels rely on the enclosing captured parent's terminal join.
        // A partially queued cold memset must finish before any cursor is freed;
        // inability to prove that boundary is fatal before member destruction.
        for(auto& endpoint:endpoints_)if(endpoint.initialization==Endpoint::Initialization::Pending)
            try{completeColdInitialization(endpoint);}catch(...){std::terminate();}
        for(const auto role:{CapturedTransferEndpoint::Producer,CapturedTransferEndpoint::Consumer})
            if(endpoints_[endpointIndex(role)].initialization==Endpoint::Initialization::Ready)
                try{requireOriginalEndpoint(role);}catch(...){std::terminate();}
    }

    DeviceId CapturedTransferChannel::endpointDevice(CapturedTransferEndpoint role) const
    {
        const auto index = endpointIndex(role);
        if (!rank_role_ || role == *rank_role_) requireOriginalEndpoint(role);
        else if (!rank_plan_) throw std::logic_error("Remote endpoint lacks its original rank declaration");
        return endpoints_[index].device;
    }

    std::shared_ptr<CapturedTransferChannel> TransferEngine::createCapturedTransferChannel(
        PhysicalMemoryAuthority &authority, DeviceId host_device,
        DeviceId producer, void *producer_setup_stream,
        DeviceId consumer, void *consumer_setup_stream, size_t capacity) const
    {
        if (isGraphCaptureActive() || !host_device.is_cpu() || !producer.is_gpu() || !consumer.is_gpu() || producer == consumer ||
            reinterpret_cast<std::uintptr_t>(producer_setup_stream)<=2 || reinterpret_cast<std::uintptr_t>(consumer_setup_stream)<=2)
            throw std::invalid_argument("Captured transfer requires CPU backing, distinct local GPUs and explicit setup streams");
        const auto geometry = CapturedTransferChannel::memoryFor(capacity);
        auto channel = std::shared_ptr<CapturedTransferChannel>(new CapturedTransferChannel);
        channel->identity_ = {nextChannelIdentity(), capacity};
        constexpr auto owner = PhysicalMemoryOwner::ActivationTransportStaging;
        const std::array devices{producer, consumer};
        const std::array streams{producer_setup_stream, consumer_setup_stream};

        // Claim the complete BOM before materializing any allocation. Partial
        // setup unwinds storage first and each canonical lease second.
        channel->mapped_lease_ = authority.claimNewAllocation(host_device, owner, geometry.mapped_host_bytes);
        for (size_t i = 0; i < devices.size(); ++i)
        {
            auto &endpoint = channel->endpoints_[i];
            endpoint.device = devices[i];
            endpoint.backend = resolveBackend(devices[i]);
            endpoint.original.emplace(GPUDeviceContextPool::instance().retainExistingContextProvenance(devices[i]));
            endpoint.original->requireWorker(endpoint.original->worker());
            endpoint.setup_stream=streams[i];
            const auto validate=[&]{(void)endpoint.original->worker()->queryStreamExecutionState(streams[i],"captured channel original setup stream validation");};
            if(endpoint.original->worker()->ownsCurrentThread())validate();else endpoint.original->worker()->submitAndWait(validate);
            endpoint.runtime_generation=endpoint.backend?endpoint.backend->deviceRuntimeGeneration(devices[i].ordinal):0;
            if(!endpoint.runtime_generation)throw std::invalid_argument("Captured channel requires original native runtime generation");
            endpoint.lease = authority.claimNewAllocation(devices[i], owner, geometry.cursor_bytes_per_device);
            if (!endpoint.backend || !endpoint.backend->prepareCapturedTransferChannelKernels(devices[i].ordinal,
                    collective_timeout_policy::kDefaultCollectiveTimeoutMs, &endpoint.timeout_ticks) || !endpoint.timeout_ticks)
                throw std::runtime_error("Captured transfer kernel preparation failed on " + devices[i].toString());
        }

        channel->mapped_ = allocateMappedHostRegion(geometry.mapped_host_bytes, devices);
        // The first-touch allocator zeroes every page; placement construction
        // publishes only the immutable identity before either GPU can run.
        auto *control = ::new (channel->mapped_->mutableHostData()) CapturedTransferChannelControl{};
        control->identity = channel->identity_;
        const auto initializationDeadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        for (size_t i = 0; i < devices.size(); ++i)
        {
            auto &endpoint = channel->endpoints_[i];
            endpoint.initialization_deadline=initializationDeadline;
            endpoint.cursor = allocateDeviceTransferBuffer(geometry.cursor_bytes_per_device, devices[i]);
            endpoint.initialization=CapturedTransferChannel::Endpoint::Initialization::Pending;
            if(!endpoint.backend->memset(endpoint.cursor->mutableDeviceData(),0,geometry.cursor_bytes_per_device,devices[i].ordinal,streams[i]))
                throw std::runtime_error("Captured channel initialization memset failed on "+devices[i].toString());
        }
        for(auto& endpoint:channel->endpoints_)CapturedTransferChannel::completeColdInitialization(endpoint);
        return channel;
    }

    CapturedTransferBinding TransferEngine::bindCapturedTransfer(
        std::shared_ptr<CapturedTransferChannel> channel, CapturedTransferEndpoint role,
        CapturedTransferMessage message, std::shared_ptr<DeviceTransferBuffer> storage, size_t offset) const
    {
        const size_t index = endpointIndex(role);
        if (!channel || !message.fits(channel->identity_.capacity) || !storage || !storage->isBound())
            throw std::invalid_argument("Captured transfer binding needs admitted channel, exact message and bound storage");
        channel->requireOriginalEndpoint(role);
        const auto &endpoint = channel->endpoints_[index];
        if (storage->device_ != endpoint.device || storage->backend_ != endpoint.backend ||
            resolveBackend(endpoint.device) != endpoint.backend)
            throw std::invalid_argument("Captured transfer buffer/backend does not own the selected endpoint");
        if (!storage->contains(offset, message.bytes))
            throw std::out_of_range("Captured transfer message exceeds device storage");
        CapturedTransferBinding binding;
        binding.channel_ = std::move(channel);
        binding.device_ = endpoint.device;
        binding.device_bytes_ = storage->mutableDeviceData(offset);
        binding.offset_ = offset;
        binding.storage_ = std::move(storage);
        binding.native_ = {
            .control = static_cast<CapturedTransferChannelControl *>(binding.channel_->mapped_->deviceAlias(endpoint.device)),
            .cursor = static_cast<CapturedTransferCursor *>(endpoint.cursor->mutableDeviceData()),
            .expected = binding.channel_->identity_, .message = message, .role = role,
            .timeout_ticks = endpoint.timeout_ticks};
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindCapturedTransfer(
        std::shared_ptr<CapturedTransferChannel> channel, CapturedTransferEndpoint role,
        CapturedTransferMessage message, std::shared_ptr<ITensor> tensor, size_t offset) const
    {
        const size_t index = endpointIndex(role);
        auto *view = dynamic_cast<TensorBase *>(tensor.get());
        auto *base = view ? view->transferStorageOwner() : nullptr;
        if (!channel || !message.fits(channel->identity_.capacity) || !base || base->transferStorageOwner() != base)
            throw std::invalid_argument("Captured tensor transfer needs a canonical physical owner and exact message");
        channel->requireOriginalEndpoint(role);
        const auto &endpoint = channel->endpoints_[index];
        std::lock_guard lock(base->coherence_mutex_);
        if (!base->gpu_data_ptr_ || !base->gpu_device_ || *base->gpu_device_ != endpoint.device ||
            resolveBackend(endpoint.device) != endpoint.backend || base->resolveBackend(endpoint.device) != endpoint.backend)
            throw std::invalid_argument("Captured tensor transfer needs existing storage on the exact endpoint");
        if (offset > base->size_bytes() || message.bytes > base->size_bytes() - offset)
            throw std::out_of_range("Captured tensor transfer exceeds physical storage");
        CapturedTransferBinding binding;
        binding.channel_ = std::move(channel);
        binding.device_ = endpoint.device;
        binding.device_bytes_ = static_cast<unsigned char *>(base->gpu_data_ptr_) + offset;
        binding.offset_ = offset;
        // Alias the supplied lifetime instead of requiring a second ownership
        // convention from tensor views which delegate to their physical owner.
        binding.storage_ = std::shared_ptr<TensorBase>(std::move(tensor), base);
        binding.native_ = {
            .control = static_cast<CapturedTransferChannelControl *>(binding.channel_->mapped_->deviceAlias(endpoint.device)),
            .cursor = static_cast<CapturedTransferCursor *>(endpoint.cursor->mutableDeviceData()),
            .expected = binding.channel_->identity_, .message = message, .role = role,
            .timeout_ticks = endpoint.timeout_ticks};
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindCapturedTransfer(
        std::shared_ptr<CapturedTransferChannel> channel, CapturedTransferEndpoint role,
        CapturedTransferMessage message, std::shared_ptr<const WorkspaceBufferLease> storage, size_t offset) const
    {
        const size_t index = endpointIndex(role);
        if (!channel || !message.fits(channel->identity_.capacity) || !storage)
            throw std::invalid_argument("Captured workspace transfer requires an admitted channel and retained region");
        channel->requireOriginalEndpoint(role);
        const auto &endpoint = channel->endpoints_[index];
        if (storage->device() != endpoint.device || storage->backend() != endpoint.backend ||
            resolveBackend(endpoint.device) != endpoint.backend)
            throw std::invalid_argument("Captured workspace transfer has foreign endpoint/backend ownership");
        if (!storage->contains(offset, message.bytes))
            throw std::out_of_range("Captured workspace message exceeds its leased region");
        CapturedTransferBinding binding;
        binding.channel_ = std::move(channel);
        binding.device_ = endpoint.device;
        binding.device_bytes_ = storage->data(offset);
        binding.offset_ = offset;
        binding.storage_ = std::move(storage);
        binding.native_ = {
            .control = static_cast<CapturedTransferChannelControl *>(binding.channel_->mapped_->deviceAlias(endpoint.device)),
            .cursor = static_cast<CapturedTransferCursor *>(endpoint.cursor->mutableDeviceData()),
            .expected = binding.channel_->identity_, .message = message, .role = role,
            .timeout_ticks = endpoint.timeout_ticks};
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindDeviceCountedTransfer(
        CapturedTransferBinding binding, std::shared_ptr<const WorkspaceBufferLease> storage, size_t offset) const
    {
        if (isGraphCaptureActive() || !binding.channel_ || !binding.native_.valid() ||
            binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage || !storage)
            throw std::invalid_argument("Device-counted transfer requires setup-time fixed capacity and retained count storage");
        binding.channel_->requireOriginalEndpoint(binding.native_.role);
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        if (storage->device() != binding.device_ || storage->backend() != endpoint.backend ||
            resolveBackend(binding.device_) != endpoint.backend)
            throw std::invalid_argument("Device-counted transfer extent belongs to a foreign endpoint/backend");
        if (!storage->contains(offset, sizeof(std::uint64_t)))
            throw std::out_of_range("Device-counted transfer extent exceeds its retained workspace");
        auto *address = storage->data(offset);
        validateExtentAddress(address, binding.device_bytes_, binding.native_.message.bytes);
        binding.extent_storage_ = std::move(storage);
        binding.extent_offset_ = offset;
        binding.native_.device_extent = static_cast<std::uint64_t *>(address);
        binding.native_.extent_source = binding.native_.role == CapturedTransferEndpoint::Producer
            ? CapturedTransferExtentSource::ProducerDevice : CapturedTransferExtentSource::ProducerPublication;
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindDeviceCountedTransfer(
        CapturedTransferBinding binding, std::shared_ptr<DeviceTransferBuffer> storage, size_t offset) const
    {
        if (isGraphCaptureActive() || !binding.channel_ || !binding.native_.valid() ||
            binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage || !storage || !storage->isBound())
            throw std::invalid_argument("Device-counted transfer requires setup-time fixed capacity and bound count storage");
        binding.channel_->requireOriginalEndpoint(binding.native_.role);
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        if (storage->device_ != binding.device_ || storage->backend_ != endpoint.backend ||
            resolveBackend(binding.device_) != endpoint.backend)
            throw std::invalid_argument("Device-counted transfer extent belongs to a foreign endpoint/backend");
        if (!storage->contains(offset, sizeof(std::uint64_t)))
            throw std::out_of_range("Device-counted transfer extent exceeds its retained buffer");
        auto *address = storage->mutableDeviceData(offset);
        validateExtentAddress(address, binding.device_bytes_, binding.native_.message.bytes);
        binding.extent_storage_ = std::move(storage);
        binding.extent_offset_ = offset;
        binding.native_.device_extent = static_cast<std::uint64_t *>(address);
        binding.native_.extent_source = binding.native_.role == CapturedTransferEndpoint::Producer
            ? CapturedTransferExtentSource::ProducerDevice : CapturedTransferExtentSource::ProducerPublication;
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindDeviceCountedTransfer(
        CapturedTransferBinding binding, std::shared_ptr<ITensor> tensor, size_t offset) const
    {
        auto *view = dynamic_cast<TensorBase *>(tensor.get());
        auto *owner = view ? view->transferStorageOwner() : nullptr;
        if (isGraphCaptureActive() || !binding.channel_ || !binding.native_.valid() ||
            binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage ||
            !owner || owner->transferStorageOwner() != owner)
            throw std::invalid_argument("Device-counted tensor extent requires setup-time fixed capacity and a canonical owner");
        binding.channel_->requireOriginalEndpoint(binding.native_.role);
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        void *address = nullptr;
        {
            std::lock_guard lock(owner->coherence_mutex_);
            if (!owner->gpu_data_ptr_ || !owner->gpu_device_ || *owner->gpu_device_ != binding.device_ ||
                owner->resolveBackend(binding.device_) != endpoint.backend || resolveBackend(binding.device_) != endpoint.backend)
                throw std::invalid_argument("Device-counted tensor extent has foreign endpoint/backend ownership");
            if (offset > owner->size_bytes() || sizeof(std::uint64_t) > owner->size_bytes() - offset)
                throw std::out_of_range("Device-counted tensor extent exceeds its original physical owner");
            address = static_cast<unsigned char *>(owner->gpu_data_ptr_) + offset;
            validateExtentAddress(address, binding.device_bytes_, binding.native_.message.bytes);
        }
        // Retain the allocator's existing record outside the coherence lock.
        // This borrow forbids physical free/reuse; replay never reacquires it.
        auto captured = retainCapturedDeviceStorage(*owner, binding.device_, owner->size_bytes());
        if (static_cast<unsigned char *>(owner->gpu_data_ptr()) + offset != address)
            throw std::logic_error("Device-counted tensor extent changed during preparation");
        binding.extent_storage_ = std::shared_ptr<TensorBase>(std::move(tensor), owner);
        binding.extent_capture_ = std::move(captured);
        binding.extent_offset_ = offset;
        binding.native_.device_extent = static_cast<std::uint64_t *>(address);
        binding.native_.extent_source = binding.native_.role == CapturedTransferEndpoint::Producer
            ? CapturedTransferExtentSource::ProducerDevice : CapturedTransferExtentSource::ProducerPublication;
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindDeviceRowCountedTransfer(
        CapturedTransferBinding binding, std::shared_ptr<ITensor> tensor, size_t offset, std::uint64_t row_bytes) const
    {
        auto *view = dynamic_cast<TensorBase *>(tensor.get());
        auto *owner = view ? view->transferStorageOwner() : nullptr;
        if (isGraphCaptureActive() || !binding.channel_ || !binding.native_.valid() ||
            binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage ||
            !owner || owner->transferStorageOwner() != owner || owner->native_type() != TensorType::INT32 ||
            !row_bytes || binding.native_.message.bytes % row_bytes != 0)
            throw std::invalid_argument("Row-counted tensor extent requires setup-time fixed capacity and a canonical owner");
        binding.channel_->requireOriginalEndpoint(binding.native_.role);
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        void *address = nullptr;
        {
            std::lock_guard lock(owner->coherence_mutex_);
            if (!owner->gpu_data_ptr_ || !owner->gpu_device_ || *owner->gpu_device_ != binding.device_ ||
                owner->resolveBackend(binding.device_) != endpoint.backend || resolveBackend(binding.device_) != endpoint.backend)
                throw std::invalid_argument("Row-counted tensor extent has foreign endpoint/backend ownership");
            if (offset > owner->size_bytes() || sizeof(std::int32_t) > owner->size_bytes() - offset)
                throw std::out_of_range("Row-counted tensor extent exceeds its original physical owner");
            address = static_cast<unsigned char *>(owner->gpu_data_ptr_) + offset;
            validateExtentAddress(address, binding.device_bytes_, binding.native_.message.bytes,
                sizeof(std::int32_t), alignof(std::int32_t));
        }
        // Retain the allocator's existing record outside the coherence lock.
        // This borrow forbids physical free/reuse; replay never reacquires it.
        auto captured = retainCapturedDeviceStorage(*owner, binding.device_, owner->size_bytes());
        if (static_cast<unsigned char *>(owner->gpu_data_ptr()) + offset != address)
            throw std::logic_error("Row-counted tensor extent changed during preparation");
        binding.extent_storage_ = std::shared_ptr<TensorBase>(std::move(tensor), owner);
        binding.extent_capture_ = std::move(captured);
        binding.extent_offset_ = offset;
        binding.native_.device_rows = static_cast<const std::int32_t *>(address);
        binding.native_.row_bytes = row_bytes;
        binding.native_.extent_source = binding.native_.role == CapturedTransferEndpoint::Producer
            ? CapturedTransferExtentSource::ProducerDevice : CapturedTransferExtentSource::ProducerPublication;
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindDeviceSlicedTransfer(
        CapturedTransferBinding binding, std::shared_ptr<ITensor> tensor,
        size_t slice_offset, size_t source_span_bytes, std::uint64_t source_offset_origin) const
    {
        auto *view = dynamic_cast<TensorBase *>(tensor.get());
        auto *owner = view ? view->transferStorageOwner() : nullptr;
        if (isGraphCaptureActive() || !binding.channel_ || !binding.native_.valid() ||
            binding.native_.role != CapturedTransferEndpoint::Producer ||
            binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage || !owner ||
            source_span_bytes == 0 || source_span_bytes < binding.native_.message.bytes ||
            source_offset_origin > ~std::uint64_t{0} - source_span_bytes || slice_offset % alignof(CapturedTransferSourceSlice))
            throw std::invalid_argument("Sliced transfer requires a setup-time producer, original descriptor and positive bounded source span");
        if (slice_offset > owner->size_bytes() || sizeof(CapturedTransferSourceSlice) > owner->size_bytes() - slice_offset)
            throw std::out_of_range("Sliced transfer descriptor exceeds its original physical owner");
        const bool source_fits = std::visit([&](const auto &storage) {
            using Owner = std::decay_t<decltype(storage)>;
            if (!storage) return false;
            if constexpr (std::is_same_v<Owner, std::shared_ptr<TensorBase>>)
                return binding.offset_ <= storage->size_bytes() && source_span_bytes <= storage->size_bytes() - binding.offset_;
            else return storage->contains(binding.offset_, source_span_bytes);
        }, binding.storage_);
        if (!source_fits) throw std::out_of_range("Sliced transfer source region exceeds its retained physical owner");
        binding = bindDeviceCountedTransfer(std::move(binding), std::move(tensor),
            slice_offset + offsetof(CapturedTransferSourceSlice, live_bytes));
        auto *descriptor = reinterpret_cast<const CapturedTransferSourceSlice *>(
            reinterpret_cast<const std::byte *>(binding.native_.device_extent) - offsetof(CapturedTransferSourceSlice, live_bytes));
        const auto begin = reinterpret_cast<std::uintptr_t>(binding.device_bytes_);
        const auto slice = reinterpret_cast<std::uintptr_t>(descriptor);
        if (slice <= begin ? begin - slice < sizeof(CapturedTransferSourceSlice) : slice - begin < source_span_bytes)
            throw std::invalid_argument("Sliced transfer descriptor overlaps its complete source region");
        if (const auto *source = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_))
        {
            const auto &extent = std::get<std::shared_ptr<TensorBase>>(binding.extent_storage_);
            binding.source_capture_ = source->get() == extent.get() ? binding.extent_capture_ :
                retainCapturedDeviceStorage(**source, binding.device_, (*source)->size_bytes());
        }
        binding.native_.source_slice = descriptor;
        binding.native_.source_span_bytes = source_span_bytes;
        binding.native_.source_offset_origin = source_offset_origin;
        return binding;
    }

    std::uint64_t CapturedTransferBinding::captureIdentity() const
    {
        if(!channel_)throw std::logic_error("Captured channel identity has no original owner");
        channel_->requireOriginalEndpoint(native_.role);
        const auto& endpoint=channel_->endpoints_[endpointIndex(native_.role)];
        std::uint64_t identity = 1469598103934665603ULL;
        const auto mix = [&](std::uint64_t word) { identity = (identity ^ word) * 1099511628211ULL; };
        mix(5); // Payload identity is now exclusively the original mapped allocation.
        if (channel_->rank_plan_)
        {
            const auto &plan = *channel_->rank_plan_;
            mix(plan.originalId().generation); mix(plan.originalId().boundary); mix(plan.physicalNode());
            for (const auto word : channel_->backing_identity_) mix(word);
            for (const auto role : {CapturedTransferEndpoint::Producer, CapturedTransferEndpoint::Consumer})
            {
                const auto endpoint = plan.endpoint(role);
                mix(endpoint.rank); mix(static_cast<std::uint64_t>(endpoint.device.type)); mix(endpoint.device.ordinal);
            }
        }
        mix(static_cast<std::uint64_t>(device_.type)); mix(device_.ordinal);
        mix(endpoint.original->generation());mix(endpoint.runtime_generation);
        mix(reinterpret_cast<std::uintptr_t>(endpoint.original->worker().get()));mix(reinterpret_cast<std::uintptr_t>(endpoint.setup_stream));
        mix(reinterpret_cast<std::uintptr_t>(native_.control)); mix(reinterpret_cast<std::uintptr_t>(native_.cursor));
        mix(native_.expected.nonce); mix(native_.expected.capacity); mix(native_.message.key); mix(native_.message.bytes);
        mix(static_cast<std::uint64_t>(native_.role)); mix(native_.timeout_ticks);
        mix(static_cast<std::uint64_t>(native_.extent_source)); mix(reinterpret_cast<std::uintptr_t>(native_.device_extent));
        mix(reinterpret_cast<std::uintptr_t>(native_.device_rows)); mix(native_.row_bytes);
        mix(reinterpret_cast<std::uintptr_t>(native_.source_slice)); mix(native_.source_span_bytes); mix(native_.source_offset_origin);
        mix(reinterpret_cast<std::uintptr_t>(device_bytes_)); mix(offset_); mix(extent_offset_);
        mix(storage_.index()); std::visit([&](const auto &owner) { mix(reinterpret_cast<std::uintptr_t>(owner.get())); }, storage_);
        mix(extent_storage_.index()); std::visit([&](const auto &owner) {
            using Owner = std::decay_t<decltype(owner)>;
            if constexpr (!std::is_same_v<Owner, std::monostate>) mix(reinterpret_cast<std::uintptr_t>(owner.get()));
        }, extent_storage_);
        // Retained borrow objects bind original native allocation generations;
        // a reset/free cannot recycle those bytes while a captured borrower lives.
        mix(reinterpret_cast<std::uintptr_t>(extent_capture_.get())); mix(reinterpret_cast<std::uintptr_t>(source_capture_.get()));
        return identity ? identity : 1;
    }

    void TransferEngine::enqueuePipelineActivationFanoutValidation(
        const PipelineActivationFanoutValidation &binding, DeviceId device, void *stream) const
    {
        auto *backend = resolveBackend(device);
        if (!device.is_gpu() || !binding.valid() || reinterpret_cast<std::uintptr_t>(stream) <= 2 || !backend ||
            !backend->enqueuePipelineActivationFanoutValidation(binding, device.ordinal, stream))
            throw std::runtime_error("Original pipeline native fanout validation enqueue failed");
    }

    void TransferEngine::enqueueCapturedTransfer(const CapturedTransferBinding &binding, void *stream) const
    {
        enqueueCapturedTransferImpl(binding, stream, nullptr);
    }

    void TransferEngine::enqueueCapturedExchange(
        std::span<const CapturedTransferBinding> producers,
        std::span<const CapturedTransferBinding> consumers,
        CapturedTransferEndpointScheduling scheduling, void *stream) const
    {
        (void)capturedTransferEndpointSchedulingName(scheduling);
        validateCapturedTransferBatch(producers, stream, nullptr);
        validateCapturedTransferBatch(consumers, stream, nullptr);
        if (producers.front().native_.role != CapturedTransferEndpoint::Producer ||
            consumers.front().native_.role != CapturedTransferEndpoint::Consumer ||
            producers.front().device_ != consumers.front().device_)
            throw std::invalid_argument("Captured exchange requires ordered same-device send/receive phases");
        enqueueCapturedExchangeImpl(producers, consumers, scheduling, stream, nullptr);
    }

    void TransferEngine::enqueueCapturedExchangeImpl(
        std::span<const CapturedTransferBinding> producers,
        std::span<const CapturedTransferBinding> consumers,
        CapturedTransferEndpointScheduling scheduling, void *stream,
        const AcquiredDeviceTransferInput *fork) const
    {
        if (scheduling == CapturedTransferEndpointScheduling::Individual)
        {
            for (const auto &binding : producers) enqueueCapturedTransferImpl(binding, stream, fork);
            for (const auto &binding : consumers) enqueueCapturedTransferImpl(binding, stream, fork);
        }
        else
        {
            // Scheduling was checked before either phase was recorded. Preserve
            // the same send-before-receive ordering and native failure contract.
            enqueueCapturedTransferBatchImpl(producers, stream, fork);
            enqueueCapturedTransferBatchImpl(consumers, stream, fork);
        }
    }

    void TransferEngine::enqueueForkedCapturedExchange(
        std::span<const CapturedTransferBinding> producers,
        std::span<const CapturedTransferBinding> consumers,
        const AcquiredDeviceTransferInput &input, ITensor *output,
        CapturedTransferEndpointScheduling scheduling) const
    {
        (void)capturedTransferEndpointSchedulingName(scheduling);
        auto *destination = dynamic_cast<TensorBase *>(output);
        if (!input.valid() || producers.empty() || consumers.empty() || !destination ||
            destination->transferStorageOwner() != destination || input.sourceOwner() == destination)
            throw std::invalid_argument("Forked captured exchange requires an acquired input and disjoint canonical output");
        requireDeviceOutput(destination, input.device(), input.consumerStream());
        requireDeviceOutput(input.sourceOwner(), input.device(), input.consumerStream());
        const auto source_address = reinterpret_cast<std::uintptr_t>(input.sourceOwner()->gpu_data_ptr());
        const auto destination_address = reinterpret_cast<std::uintptr_t>(destination->gpu_data_ptr());
        if (source_address <= destination_address
                ? destination_address - source_address < input.sourceOwner()->size_bytes()
                : source_address - destination_address < destination->size_bytes())
            throw std::invalid_argument("Forked captured exchange packet banks physically overlap");
        // Authenticate the complete participant before recording any channel.
        // A caller cannot use a valid fork to authorize a different tensor,
        // foreign device, receive-as-send, or unowned workspace publication.
        const auto validate = [&](const auto &bindings, CapturedTransferEndpoint role, TensorBase *owner) {
            for (const auto &binding : bindings)
            {
                const auto *storage = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_);
                if (binding.device_ != input.device() || binding.native_.role != role ||
                    !storage || storage->get() != owner)
                    throw std::invalid_argument("Forked captured exchange endpoint disagrees with its acquired arena owners");
            }
        };
        validate(producers, CapturedTransferEndpoint::Producer, input.sourceOwner());
        validate(consumers, CapturedTransferEndpoint::Consumer, destination);
        // Resolve both phases before the first native operation. A malformed
        // receive cannot leave an otherwise valid producer batch in flight.
        validateCapturedTransferBatch(producers, input.consumerStream(), &input);
        validateCapturedTransferBatch(consumers, input.consumerStream(), &input);
        enqueueCapturedExchangeImpl(producers, consumers, scheduling, input.consumerStream(), &input);
    }

    CapturedTransferChannelBatchEntry TransferEngine::resolveCapturedTransferEndpoint(
        const CapturedTransferBinding &binding, void *stream,
        const AcquiredDeviceTransferInput *fork) const
    {
        if (!stream || !binding.channel_ || !binding.native_.valid())
            throw std::invalid_argument("Captured transfer submission needs a frozen binding and explicit stream");
        binding.channel_->requireOriginalEndpoint(binding.native_.role);
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        auto *backend = resolveBackend(binding.device_);
        if (!backend || backend != endpoint.backend)
            throw std::logic_error("Captured transfer backend identity changed after binding");
        if (binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage)
        {
            const size_t count_bytes = binding.native_.device_rows ? sizeof(std::int32_t) : sizeof(std::uint64_t);
            const void *count_address = binding.native_.device_rows
                ? static_cast<const void *>(binding.native_.device_rows) : binding.native_.device_extent;
            const bool retained = std::visit([&](const auto &storage) {
                using Owner = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<Owner, std::monostate>) return false;
                else if constexpr (std::is_same_v<Owner, std::shared_ptr<DeviceTransferBuffer>>)
                    return storage && storage->device_ == binding.device_ && storage->backend_ == backend &&
                        storage->contains(binding.extent_offset_, count_bytes) &&
                        storage->mutableDeviceData(binding.extent_offset_) == count_address;
                else if constexpr (std::is_same_v<Owner, std::shared_ptr<TensorBase>>)
                {
                    if (!storage || !binding.extent_capture_ || storage->transferStorageOwner() != storage.get() ||
                        storage->resolveBackend(binding.device_) != backend ||
                        storage->size_bytes() != binding.extent_capture_->bytes_)
                        return false;
                    requireCapturedDeviceStorage(*binding.extent_capture_, *storage, binding.device_, storage->size_bytes());
                    if (binding.extent_offset_ > storage->size_bytes() ||
                        count_bytes > storage->size_bytes() - binding.extent_offset_ ||
                        static_cast<unsigned char *>(storage->gpu_data_ptr()) + binding.extent_offset_ != count_address)
                        return false;
                    if (binding.native_.role == CapturedTransferEndpoint::Producer || binding.native_.device_rows)
                    {
                        if (fork)
                        {
                            if (!fork->includesSource(storage.get()))
                                throw std::invalid_argument("Forked counted transfer did not acquire its original tensor extent");
                        }
                        else requireDeviceInput(storage.get(), binding.device_, stream);
                    }
                    else
                    {
                        const auto *packet = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_);
                        if (fork && (!packet || packet->get() != storage.get()))
                            throw std::invalid_argument("Forked counted receive extent requires the existing destination join owner");
                        requireDeviceOutput(storage.get(), binding.device_, stream);
                    }
                    return true;
                }
                else
                    return storage && storage->device() == binding.device_ && storage->backend() == backend &&
                        storage->contains(binding.extent_offset_, count_bytes) &&
                        storage->data(binding.extent_offset_) == count_address;
            }, binding.extent_storage_);
            if (!retained) throw std::logic_error("Device-counted transfer lost its retained extent owner");
        }
        const auto storage_bytes = binding.native_.source_slice
            ? binding.native_.source_span_bytes : binding.native_.message.bytes;
        TensorBase *tensor = nullptr;
        if (const auto *storage = std::get_if<std::shared_ptr<DeviceTransferBuffer>>(&binding.storage_))
        {
            if (!*storage || (*storage)->device_ != binding.device_ || (*storage)->backend_ != backend ||
                !(*storage)->contains(binding.offset_, storage_bytes) ||
                (*storage)->mutableDeviceData(binding.offset_) != binding.device_bytes_)
                throw std::logic_error("Captured transfer scratch binding is stale");
        }
        else if (const auto *storage = std::get_if<std::shared_ptr<const WorkspaceBufferLease>>(&binding.storage_))
        {
            // The lease carries the original block, not the current workspace
            // name map. Growth/retirement cannot redirect this captured address.
            if (!*storage || (*storage)->device() != binding.device_ || (*storage)->backend() != backend ||
                !(*storage)->contains(binding.offset_, storage_bytes) ||
                (*storage)->data(binding.offset_) != binding.device_bytes_)
                throw std::logic_error("Captured workspace transfer lost its retained physical owner");
        }
        else
        {
            tensor = std::get<std::shared_ptr<TensorBase>>(binding.storage_).get();
            if (!tensor || tensor->transferStorageOwner() != tensor || tensor->resolveBackend(binding.device_) != backend)
                throw std::logic_error("Captured transfer tensor owner changed");
            if (binding.native_.source_slice)
            {
                if (!binding.source_capture_ || tensor->size_bytes() != binding.source_capture_->bytes_)
                    throw std::logic_error("Sliced transfer lost its original source allocation geometry");
                requireCapturedDeviceStorage(*binding.source_capture_, *tensor, binding.device_, tensor->size_bytes());
            }
            if (binding.native_.role == CapturedTransferEndpoint::Producer && !fork)
                requireDeviceInput(tensor, binding.device_, stream);
            else
                requireDeviceOutput(tensor, binding.device_, stream);
            if (binding.offset_ > tensor->size_bytes() || storage_bytes > tensor->size_bytes() - binding.offset_ ||
                static_cast<unsigned char *>(tensor->gpu_data_ptr()) + binding.offset_ != binding.device_bytes_)
                throw std::logic_error("Captured transfer tensor storage changed after binding");
        }
        // Setup certified initialization. Submit one complete typed operation;
        // the backend may fuse nodes without splitting ownership or making
        // TransferEngine dictate a needlessly fragmented native graph.
        auto *mapped_bytes = binding.channel_->mapped_->deviceAlias(binding.device_, CapturedTransferChannel::payload_offset);
        const bool outbound = binding.native_.role == CapturedTransferEndpoint::Producer;
        void *destination = outbound ? mapped_bytes : binding.device_bytes_;
        const void *source = outbound ? binding.device_bytes_ : mapped_bytes;
        return {.binding = binding.native_, .destination = destination, .source = source};
    }

    void TransferEngine::enqueueCapturedTransferImpl(const CapturedTransferBinding &binding,
        void *stream, const AcquiredDeviceTransferInput *fork) const
    {
        const auto entry = resolveCapturedTransferEndpoint(binding, stream, fork);
        auto *const backend = resolveBackend(binding.device_);
        if (!backend->enqueueCapturedTransferChannel(entry.binding, entry.destination, entry.source,
                binding.device_.ordinal, stream))
            throw std::runtime_error("Captured transfer submission failed");
        if (const auto *storage = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_))
            if (binding.native_.role == CapturedTransferEndpoint::Consumer && !fork)
                publishDeviceWrite(storage->get(), binding.device_, stream);
        if (binding.native_.role == CapturedTransferEndpoint::Consumer && !fork && !binding.native_.device_rows)
            if (const auto *extent = std::get_if<std::shared_ptr<TensorBase>>(&binding.extent_storage_))
            {
                const auto *packet = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_);
                if (!packet || packet->get() != extent->get())
                    publishDeviceWrite(extent->get(), binding.device_, stream);
            }
    }

    void TransferEngine::enqueueCapturedTransferBatch(
        std::span<const CapturedTransferBinding> bindings, void *stream) const
    {
        enqueueCapturedTransferBatchImpl(bindings, stream, nullptr);
    }

    void TransferEngine::validateCapturedTransferBatch(
        std::span<const CapturedTransferBinding> bindings, void *stream,
        const AcquiredDeviceTransferInput *fork) const
    {
        if (bindings.empty() || !stream)
            throw std::invalid_argument("Captured endpoint batch requires nonempty original bindings and exact stream");
        const auto device = bindings.front().device_;
        const auto role = bindings.front().native_.role;
        // Full validation precedes submission, including across packet boundaries.
        // A receive bank cannot use a zero current count to disguise aliasing.
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto &binding = bindings[i];
            if (binding.device_ != device || binding.native_.role != role)
                throw std::invalid_argument("Captured endpoint batch mixes physical devices or endpoint roles");
            const auto begin = reinterpret_cast<std::uintptr_t>(binding.device_bytes_);
            for (size_t j = 0; j < i; ++j)
            {
                const auto &prior = bindings[j];
                if (binding.native_.cursor == prior.native_.cursor || binding.native_.control == prior.native_.control)
                    throw std::invalid_argument("Captured endpoint batch duplicates an original channel");
                if (role == CapturedTransferEndpoint::Consumer)
                {
                    const auto previous = reinterpret_cast<std::uintptr_t>(prior.device_bytes_);
                    if (begin <= previous ? previous - begin < binding.native_.message.bytes
                        : begin - previous < prior.native_.message.bytes)
                        throw std::invalid_argument("Captured endpoint batch has overlapping receive capacities");
                    // Publication counts are mutable outputs, unlike shared
                    // producer count inputs. Authenticate their complete byte
                    // ranges against every other endpoint before any enqueue.
                    const auto count = reinterpret_cast<std::uintptr_t>(binding.native_.device_rows
                        ? static_cast<const void *>(binding.native_.device_rows) : binding.native_.device_extent);
                    const auto previous_count = reinterpret_cast<std::uintptr_t>(prior.native_.device_rows
                        ? static_cast<const void *>(prior.native_.device_rows) : prior.native_.device_extent);
                    const auto overlaps = [](std::uintptr_t first, size_t first_bytes,
                                             std::uintptr_t second, size_t second_bytes) {
                        return first <= second ? second - first < first_bytes : first - second < second_bytes;
                    };
                    const size_t count_bytes = binding.native_.device_rows ? sizeof(std::int32_t) : sizeof(std::uint64_t);
                    const size_t previous_count_bytes = prior.native_.device_rows ? sizeof(std::int32_t) : sizeof(std::uint64_t);
                    const bool count_output = binding.native_.device_extent != nullptr;
                    const bool previous_count_output = prior.native_.device_extent != nullptr;
                    // Shared expected-row INPUTS may alias each other. They may
                    // never overlap any receive payload or byte-count OUTPUT.
                    if ((count && previous_count && (count_output || previous_count_output) &&
                            overlaps(count, count_bytes, previous_count, previous_count_bytes)) ||
                        (count && overlaps(count, count_bytes, previous, prior.native_.message.bytes)) ||
                        (previous_count && overlaps(previous_count, previous_count_bytes, begin, binding.native_.message.bytes)))
                        throw std::invalid_argument("Captured endpoint batch count publication overlaps another count or payload capacity");
                }
            }
        }
        // Reject every cross-endpoint alias before importing a producer event.
        // Endpoint resolution may enqueue those legitimate dependency waits;
        // immutable geometry must therefore be valid for the entire batch first.
        for (const auto &binding : bindings)
            (void)resolveCapturedTransferEndpoint(binding, stream, fork);
    }

    void TransferEngine::enqueueCapturedTransferBatchImpl(
        std::span<const CapturedTransferBinding> bindings, void *stream,
        const AcquiredDeviceTransferInput *fork) const
    {
        validateCapturedTransferBatch(bindings, stream, fork);
        const auto device = bindings.front().device_;
        const auto role = bindings.front().native_.role;
        auto *const backend = resolveBackend(device);
        for (size_t begin = 0; begin < bindings.size(); begin += CapturedTransferChannelBatch::kCapacity)
        {
            CapturedTransferChannelBatch packet;
            packet.count = static_cast<unsigned>(std::min(bindings.size() - begin,
                size_t{CapturedTransferChannelBatch::kCapacity}));
            bool single = true;
            for (unsigned i = 0; i < packet.count; ++i)
            {
                packet.entries[i] = resolveCapturedTransferEndpoint(bindings[begin + i], stream, fork);
                single = single && capturedTransferBatchKernelPlan(packet.entries[i].binding.message.bytes) ==
                    CapturedTransferKernelPlan::SingleBlock;
            }
            if (!backend->enqueueCapturedTransferChannelBatch(packet, device.ordinal, stream))
                throw std::runtime_error("Captured endpoint batch native submission failed; partial submission cannot be retried");
            // The native graph owns argument copies. Original retained bindings
            // still own all aliases/cursors; no new lifetime or PMA charge exists.
            for (unsigned i = 0; i < packet.count; ++i)
            {
                const auto &binding = bindings[begin + i];
                if (const auto *storage = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_))
                    if (role == CapturedTransferEndpoint::Consumer && !fork)
                        publishDeviceWrite(storage->get(), device, stream);
                if (role == CapturedTransferEndpoint::Consumer && !fork && !binding.native_.device_rows)
                    if (const auto *extent = std::get_if<std::shared_ptr<TensorBase>>(&binding.extent_storage_))
                    {
                        const auto *payload = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_);
                        if (!payload || payload->get() != extent->get())
                            publishDeviceWrite(extent->get(), device, stream);
                    }
            }
            if (PerfStatsCollector::isDomainEnabled("transfer"))
                PerfStatsCollector::addCounter("transfer", "captured_endpoint_batch_submissions", 1.0,
                    {}, device.toString(), {{"endpoints", std::to_string(packet.count)},
                        {"role", role == CapturedTransferEndpoint::Producer ? "producer" : "consumer"},
                        {"lowering", single ? "fused" : "acquire_copy_publish"},
                        {"accounting", "host_submission_or_graph_template"}});
        }
    }
}
