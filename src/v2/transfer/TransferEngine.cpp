/**
 * @file TransferEngine.cpp
 * @brief Canonical event-ordered tensor movement and coherence publication.
 *
 * TransferEngine owns the relationship between byte movement, tensor
 * authority, and producer/consumer ordering. GPU copies are submitted on one
 * exact non-null stream and publish an exact event. Device consumers import
 * that event without blocking the host; host consumers wait only for the event
 * that makes their destination bytes observable. Backend copy functions remain
 * low-level submission mechanisms and never decide tensor coherence.
 */

#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include <sys/mman.h>
#include <unistd.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "collective/BackendRouter.h"
#include "collective/ICollectiveBackend.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "utils/DebugEnv.h"
#include "utils/KernelProfiler.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/StackTrace.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Resolve the coherence implementation hidden behind ITensor.
         *
         * Production tensors are TensorBase implementations. Keeping this
         * checked conversion inside TransferEngine preserves ITensor as the
         * orchestration-facing abstraction and turns an unsupported tensor
         * implementation into an immediate contract failure.
         */
        TensorBase *requireTensorBase(ITensor *tensor, const char *operation)
        {
            if (!tensor)
            {
                throw std::invalid_argument(
                    std::string(operation) + " requires a tensor");
            }

            auto *base = dynamic_cast<TensorBase *>(tensor);
            if (!base)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " requires a coherence-aware TensorBase implementation");
            }
            return base;
        }

        /**
         * @brief Resolve the one tensor that owns physical transfer state.
         *
         * Tensor wrappers deliberately delegate pointer and coherence queries
         * to an inner tensor.  TransferEngine must therefore lock and mutate
         * that same inner object, never the wrapper's dormant TensorBase
         * fields.  Requiring a canonical owner at this boundary gives every
         * transfer API the same structural rule.
         */
        TensorBase *requireTransferStorageOwner(
            ITensor *tensor,
            const char *operation)
        {
            TensorBase *base = requireTensorBase(tensor, operation);
            TensorBase *owner = base->transferStorageOwner();
            if (!owner)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " resolved a null transfer-storage owner");
            }
            if (owner->transferStorageOwner() != owner)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " resolved a non-canonical transfer-storage owner");
            }
            return owner;
        }

        /**
         * @brief Resolve the persistent stream owned by a GPU transfer context.
         *
         * Backend APIs deliberately reject null streams. Tensor-aware transfer
         * operations choose their stream here, before touching a backend, so
         * setup, staging, and publication all name one stable owner.
         */
        void *requireTransferStream(
            DeviceId device,
            const char *operation)
        {
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    std::string(operation) +
                    " requires a GPU transfer endpoint");
            }

            void *const stream =
                GPUDeviceContextPool::instance()
                    .getContext(device)
                    .defaultStream();
            if (!stream)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " could not resolve a non-null transfer stream for " +
                    device.toString());
            }
            return stream;
        }

        /**
         * @brief Describe the tensor ownership contract at a failed transfer boundary.
         *
         * A stage can prepare both arena activations and model weights through
         * the same TransferEngine API.  Include the state that controls
         * `uploadFull()` so a hard residency failure identifies the offending
         * tensor and distinguishes an invalid arena publication from a
         * host-resident or prepared-weight classification error.
         */
        std::string tensorTransferState(const TensorBase &tensor)
        {
            std::ostringstream description;
            description
                << " tensor='"
                << (tensor.debugName().empty()
                        ? std::string("(unnamed)")
                        : tensor.debugName())
                << "' coherence=" << to_string(tensor.coherenceState())
                << " residency=" << to_string(tensor.memoryResidency())
                << " prepared_device_state="
                << (tensor.hasPreparedDeviceState() ? "true" : "false")
                << " current_device="
                << (tensor.current_device().has_value()
                        ? tensor.current_device()->toString()
                        : std::string("none"))
                << " gpu_ptr=" << tensor.gpu_data_ptr();
            return description.str();
        }
    } // namespace

    const char *to_string(DeviceMemoryReclamationIntent intent) noexcept
    {
        switch (intent)
        {
        case DeviceMemoryReclamationIntent::RetiredExecutionTopology:
            return "retired_execution_topology";
        case DeviceMemoryReclamationIntent::ExclusiveModelRetirement:
            return "exclusive_model_retirement";
        }
        return "unknown";
    }

    size_t ModelDeviceMemoryRetention::totalBytes() const
    {
        if (prepared_weight_bytes >
            std::numeric_limits<size_t>::max() -
                reusable_workspace_bytes)
        {
            throw std::overflow_error(
                "Model device-memory retention BOM overflows size_t");
        }
        return prepared_weight_bytes + reusable_workspace_bytes;
    }

    bool ModelDeviceMemoryRetention::valid() const noexcept
    {
        if (!device.is_gpu())
            return false;
        if (prepared_weight_bytes >
            std::numeric_limits<size_t>::max() -
                reusable_workspace_bytes)
        {
            return false;
        }
        return prepared_weight_bytes + reusable_workspace_bytes > 0u;
    }

    DeviceMemoryReclamationRequest
    DeviceMemoryReclamationRequest::retiredExecutionTopology(DeviceId device)
    {
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "Retired execution-topology reclamation requires an exact GPU device");
        }
        return DeviceMemoryReclamationRequest(device);
    }

    ExclusiveModelRetirementTicket
    TransferEngine::beginExclusiveModelRetirement(
        const ModelDeviceMemoryRetention &retention) const
    {
        if (!retention.valid())
        {
            throw std::invalid_argument(
                "Exclusive model retirement requires one exact GPU and a positive, representable allocation BOM");
        }

        IBackend *const backend = resolveBackend(retention.device);
        if (!backend)
        {
            throw std::runtime_error(
                "Exclusive model retirement has no backend for " +
                retention.device.toString());
        }
        const DeviceAllocationAccounting accounting =
            backend->deviceAllocationAccounting(
                retention.device.gpu_ordinal());
        if (!accounting.supported)
        {
            throw std::runtime_error(
                "Exclusive model retirement cannot observe canonical allocations for " +
                retention.device.toString() + ": " +
                (accounting.diagnostic.empty()
                     ? std::string("backend returned unsupported accounting")
                     : accounting.diagnostic));
        }
        const size_t expected = retention.totalBytes();
        if (accounting.active_bytes < expected)
        {
            std::ostringstream error;
            error
                << "Exclusive model-retirement BOM exceeds canonical live "
                   "allocation ownership for "
                << retention.device.toString()
                << ": expected_retired_bytes=" << expected
                << " canonical_active_allocations="
                << accounting.active_allocations
                << " canonical_active_bytes=" << accounting.active_bytes;
            throw std::runtime_error(error.str());
        }
        const size_t baseline = backend->deviceMemoryFree(
            retention.device.gpu_ordinal());

        LOG_INFO(
            "[DeviceMemoryReclamation] begin device="
            << retention.device.toString()
            << " intent="
            << to_string(DeviceMemoryReclamationIntent::ExclusiveModelRetirement)
            << " driver_free_before_owner_release=" << baseline
            << " prepared_weight_bytes=" << retention.prepared_weight_bytes
            << " reusable_workspace_bytes="
            << retention.reusable_workspace_bytes
            << " expected_retired_bytes=" << expected
            << " canonical_active_allocations="
            << accounting.active_allocations
            << " canonical_active_bytes=" << accounting.active_bytes);
        return ExclusiveModelRetirementTicket(
            retention,
            baseline,
            accounting.active_allocations,
            accounting.active_bytes);
    }

    DeviceMemoryReclamationReceipt
    TransferEngine::completeExclusiveModelRetirement(
        ExclusiveModelRetirementTicket &&ticket) const
    {
        std::vector<ExclusiveModelRetirementTicket> tickets;
        tickets.reserve(1u);
        tickets.push_back(std::move(ticket));
        auto receipts = completeExclusiveModelRetirements(
            std::move(tickets));
        if (receipts.size() != 1u)
        {
            throw std::logic_error(
                "Single-device model retirement produced an invalid receipt cardinality");
        }
        return std::move(receipts.front());
    }

    std::vector<DeviceMemoryReclamationReceipt>
    TransferEngine::completeExclusiveModelRetirements(
        std::vector<ExclusiveModelRetirementTicket> &&tickets) const
    {
        if (tickets.empty())
        {
            throw std::invalid_argument(
                "Exclusive model-retirement batch requires at least one ticket");
        }

        struct PendingRetirement final
        {
            ExclusiveModelRetirementTicket *ticket = nullptr;
            DeviceId device = DeviceId::invalid();
            IBackend *backend = nullptr;
        };

        std::set<DeviceId> devices;
        std::vector<PendingRetirement> pending;
        pending.reserve(tickets.size());
        for (auto &ticket : tickets)
        {
            if (!ticket.valid_)
            {
                throw std::logic_error(
                    "Exclusive model-retirement batch contains a moved or already consumed ticket");
            }
            const DeviceId device = ticket.retention_.device;
            if (!devices.insert(device).second)
            {
                throw std::invalid_argument(
                    "Exclusive model-retirement batch contains duplicate device " +
                    device.toString());
            }
            IBackend *const backend = resolveBackend(device);
            if (!backend)
            {
                throw std::runtime_error(
                    "Exclusive model retirement has no backend for " +
                    device.toString());
            }
            pending.push_back(PendingRetirement{
                .ticket = &ticket,
                .device = device,
                .backend = backend,
            });
        }

        /* Completion is an irrevocable topology transition. Consume every
         * obligation together before destroying a collective or worker owner;
         * a partially reset model cannot safely retry the remaining tickets. */
        for (auto &retirement : pending)
            retirement.ticket->valid_ = false;

        /* Collective libraries own native streams, events, communicators, and
         * internal allocations outside IBackend's tensor ledger. Retire the
         * complete participant set before touching any primary context so a
         * peer communicator cannot retain another device's generation. */
        for (const auto &retirement : pending)
        {
            const CollectiveRuntimeRetirementReceipt collective_retirement =
                GlobalBackendRouter::retireForExclusiveDeviceRuntimeReset(
                    retirement.device);
            if (!collective_retirement.complete())
            {
                std::ostringstream error;
                error
                    << "Exclusive runtime-generation retirement found live "
                       "collective ownership for "
                    << retirement.device.toString()
                    << " state="
                    << static_cast<int>(collective_retirement.state)
                    << " active_owners="
                    << collective_retirement.active_owners
                    << ": "
                    << (collective_retirement.diagnostic.empty()
                            ? "collective authority returned an incomplete receipt"
                            : collective_retirement.diagnostic);
                throw std::runtime_error(error.str());
            }
        }

        /* Hold every acquisition-exclusion scope simultaneously. Destroying
         * all worker streams, BLAS handles, events, and peer mappings before
         * the first cudaDeviceReset/hipDeviceReset is the critical batch edge:
         * sequential context destruction lets an unretired peer keep hundreds
         * of MiB of driver state alive across the reset. */
        std::vector<ExclusiveGPUDeviceGenerationRetirement>
            context_retirements;
        std::vector<GPUDeviceContextGenerationRetirementReceipt>
            context_receipts;
        context_retirements.reserve(pending.size());
        context_receipts.reserve(pending.size());
        for (const auto &retirement : pending)
        {
            context_retirements.push_back(
                GPUDeviceContextPool::instance()
                    .beginExclusiveGenerationRetirement(
                        retirement.device));
            context_receipts.push_back(
                context_retirements.back().receipt());
        }

        std::vector<DeviceRuntimeGenerationRetirementResult>
            runtime_receipts;
        runtime_receipts.reserve(pending.size());
        for (const auto &retirement : pending)
        {
            const DeviceRuntimeGenerationRetirementRequest runtime_request(
                retirement.device.gpu_ordinal());
            runtime_receipts.push_back(
                retirement.backend
                    ->retireExclusiveDeviceRuntimeGeneration(
                        runtime_request));
            const auto &runtime_receipt = runtime_receipts.back();
            if (!runtime_receipt.supported || !runtime_receipt.success ||
                !runtime_receipt.reset_invoked ||
                runtime_receipt.retired_generation == 0u ||
                runtime_receipt.successor_generation !=
                    runtime_receipt.retired_generation + 1u ||
                runtime_receipt.post_reset_state !=
                    DeviceRuntimePostResetState::Quiescent ||
                runtime_receipt.tracked_device_allocations != 0u ||
                runtime_receipt.tracked_device_allocation_bytes != 0u ||
                runtime_receipt.tracked_host_registrations != 0u)
            {
                std::ostringstream error;
                error
                    << "Exclusive runtime-generation retirement failed for "
                    << retirement.device.toString()
                    << " backend=" << retirement.backend->backendName()
                    << " reset_invoked="
                    << (runtime_receipt.reset_invoked ? "true" : "false")
                    << " retired_generation="
                    << runtime_receipt.retired_generation
                    << " successor_generation="
                    << runtime_receipt.successor_generation
                    << " post_reset_state="
                    << to_string(runtime_receipt.post_reset_state)
                    << " tracked_device_allocations="
                    << runtime_receipt.tracked_device_allocations
                    << " tracked_device_allocation_bytes="
                    << runtime_receipt.tracked_device_allocation_bytes
                    << " tracked_host_registrations="
                    << runtime_receipt.tracked_host_registrations
                    << ": "
                    << (runtime_receipt.diagnostic.empty()
                            ? "backend returned an incomplete runtime-reset receipt"
                            : runtime_receipt.diagnostic);
                throw std::runtime_error(error.str());
            }
        }

        std::vector<DeviceMemoryReclamationReceipt> receipts;
        receipts.reserve(pending.size());
        for (size_t index = 0u; index < pending.size(); ++index)
        {
            const auto &retirement = pending[index];
            const auto &ticket = *retirement.ticket;
            const auto &context_receipt = context_receipts[index];
            const auto &runtime_receipt = runtime_receipts[index];
            const DeviceId device = retirement.device;
            const size_t expected_retired_bytes =
                ticket.retention_.totalBytes();

            /* Native generation reset already destroys graph and default-pool
             * caches. Build the exclusive receipt directly instead of creating
             * a fresh generation only to trim it again. */
            DeviceMemoryReclamationReceipt receipt{
                .device = device,
                .intent =
                    DeviceMemoryReclamationIntent::ExclusiveModelRetirement,
                .expected_retired_bytes = expected_retired_bytes,
                .driver_free_bytes_before_owner_release =
                    ticket.driver_free_bytes_before_owner_release_,
                .canonical_allocations_before_owner_release =
                    ticket.canonical_allocations_before_owner_release_,
                .canonical_allocation_bytes_before_owner_release =
                    ticket.canonical_allocation_bytes_before_owner_release_,
                .canonical_allocations_before_runtime_reset =
                    runtime_receipt.tracked_device_allocations,
                .canonical_allocation_bytes_before_runtime_reset =
                    runtime_receipt.tracked_device_allocation_bytes,
                .driver_free_bytes_before =
                    runtime_receipt.driver_free_bytes_before,
            };
            if (receipt.releasedCanonicalBytes() <
                receipt.expected_retired_bytes)
            {
                std::ostringstream error;
                error
                    << "Exclusive model-retirement canonical allocation proof "
                       "is incomplete for "
                    << receipt.device.toString()
                    << ": released_canonical_bytes="
                    << receipt.releasedCanonicalBytes()
                    << " expected_retired_bytes="
                    << receipt.expected_retired_bytes
                    << " canonical_allocations_before_owner_release="
                    << receipt.canonical_allocations_before_owner_release
                    << " canonical_allocation_bytes_before_owner_release="
                    << receipt.canonical_allocation_bytes_before_owner_release
                    << " canonical_allocations_before_runtime_reset="
                    << receipt.canonical_allocations_before_runtime_reset
                    << " canonical_allocation_bytes_before_runtime_reset="
                    << receipt.canonical_allocation_bytes_before_runtime_reset;
                throw std::runtime_error(error.str());
            }
            receipt.retired_context_generation =
                context_receipt.retired_generation;
            receipt.runtime_reset_invoked = runtime_receipt.reset_invoked;
            receipt.retired_runtime_generation =
                runtime_receipt.retired_generation;
            receipt.successor_runtime_generation =
                runtime_receipt.successor_generation;
            receipt.runtime_post_reset_state =
                runtime_receipt.post_reset_state;
            receipt.runtime_driver_free_bytes_before =
                runtime_receipt.driver_free_bytes_before;
            PerfStatsCollector::addCounter(
                "device_memory",
                "retired_context_generations",
                context_receipt.retiredLiveContext() ? 1.0 : 0.0,
                "model_lifecycle",
                device.toString(),
                {{"generation",
                  std::to_string(context_receipt.retired_generation)}});
            PerfStatsCollector::addCounter(
                "device_memory",
                "retired_runtime_generations",
                1.0,
                "model_lifecycle",
                device.toString(),
                {{"retired_generation",
                  std::to_string(runtime_receipt.retired_generation)},
                 {"successor_generation",
                  std::to_string(runtime_receipt.successor_generation)},
                 {"post_reset_state",
                  to_string(runtime_receipt.post_reset_state)},
                 {"driver_free_before_bytes",
                  std::to_string(runtime_receipt.driver_free_bytes_before)},
                 {"released_canonical_bytes",
                  std::to_string(receipt.releasedCanonicalBytes())},
                 {"expected_retired_bytes",
                  std::to_string(receipt.expected_retired_bytes)}});
            LOG_INFO(
                "[DeviceRuntimeGeneration] device="
                << device.toString()
                << " backend=" << retirement.backend->backendName()
                << " context_generation="
                << context_receipt.retired_generation
                << " runtime_generation="
                << runtime_receipt.retired_generation << "->"
                << runtime_receipt.successor_generation
                << " post_reset_state="
                << to_string(runtime_receipt.post_reset_state)
                << " canonical_allocation_bytes="
                << receipt.canonical_allocation_bytes_before_owner_release
                << "->"
                << receipt.canonical_allocation_bytes_before_runtime_reset
                << " driver_free_before_reset="
                << runtime_receipt.driver_free_bytes_before);
            receipts.push_back(std::move(receipt));
        }

        PerfStatsCollector::addCounter(
            "device_memory",
            "retired_runtime_generation_batches",
            1.0,
            "model_lifecycle",
            "process",
            {{"participants", std::to_string(receipts.size())}});
        LOG_INFO(
            "[DeviceRuntimeGenerationBatch] participants="
            << receipts.size() << " state=complete");
        return receipts;
    }

    DeviceMemoryReclamationReceipt TransferEngine::reclaimDeviceMemory(
        const DeviceMemoryReclamationRequest &request) const
    {
        IBackend *const backend = resolveBackend(request.device());
        if (!backend)
        {
            throw std::runtime_error(
                "Device-memory reclamation has no backend for " +
                request.device().toString());
        }

        const DeviceMemoryCacheReclamationResult raw =
            backend->trimUnusedDeviceMemoryCaches(
                request.device().gpu_ordinal());
        if (!raw.supported || !raw.success)
        {
            std::ostringstream error;
            error << "Device-memory reclamation failed for "
                  << request.device().toString()
                  << " intent=" << to_string(request.intent())
                  << " backend=" << backend->backendName()
                  << ": "
                  << (raw.diagnostic.empty()
                          ? "backend returned an incomplete reclamation result"
                          : raw.diagnostic);
            throw std::runtime_error(error.str());
        }
        if (!raw.before.graph_accounting_available ||
            !raw.after.graph_accounting_available ||
            raw.before.async_pool_accounting_available !=
                raw.after.async_pool_accounting_available ||
            !raw.graph_trim_invoked)
        {
            throw std::runtime_error(
                "Device-memory reclamation backend returned an internally inconsistent receipt for " +
                request.device().toString());
        }

        DeviceMemoryReclamationReceipt receipt{
            .device = request.device(),
            .intent = request.intent(),
            .driver_free_bytes_before = raw.before.driver_free_bytes,
            .driver_free_bytes_after = raw.after.driver_free_bytes,
            .graph_used_bytes_before = raw.before.graph_used_bytes,
            .graph_used_bytes_after = raw.after.graph_used_bytes,
            .graph_reserved_bytes_before = raw.before.graph_reserved_bytes,
            .graph_reserved_bytes_after = raw.after.graph_reserved_bytes,
            .async_pool_used_bytes_before = raw.before.async_pool_used_bytes,
            .async_pool_used_bytes_after = raw.after.async_pool_used_bytes,
            .async_pool_reserved_bytes_before =
                raw.before.async_pool_reserved_bytes,
            .async_pool_reserved_bytes_after =
                raw.after.async_pool_reserved_bytes,
            .graph_accounting_available =
                raw.before.graph_accounting_available,
            .async_pool_accounting_available =
                raw.before.async_pool_accounting_available,
            .graph_trim_invoked = raw.graph_trim_invoked,
            .async_pool_trim_invoked = raw.async_pool_trim_invoked,
        };

        LOG_INFO(
            "[DeviceMemoryReclamation] device="
            << receipt.device.toString()
            << " intent=" << to_string(receipt.intent)
            << " driver_free_before=" << receipt.driver_free_bytes_before
            << " driver_free_before_owner_release="
            << receipt.driver_free_bytes_before_owner_release
            << " driver_free_after=" << receipt.driver_free_bytes_after
            << " reclaimed=" << receipt.reclaimedDriverBytes()
            << " driver_bytes_visible_since_owner_release="
            << receipt.driverBytesVisibleSinceOwnerRelease()
            << " expected_retired_bytes="
            << receipt.expected_retired_bytes
            << " graph_used=" << receipt.graph_used_bytes_before
            << "->" << receipt.graph_used_bytes_after
            << " graph_reserved=" << receipt.graph_reserved_bytes_before
            << "->" << receipt.graph_reserved_bytes_after
            << " async_pool_used=" << receipt.async_pool_used_bytes_before
            << "->" << receipt.async_pool_used_bytes_after
            << " async_pool_reserved="
            << receipt.async_pool_reserved_bytes_before
            << "->" << receipt.async_pool_reserved_bytes_after
            << " async_pool_accounted="
            << (receipt.async_pool_accounting_available ? "true" : "false"));
        PerfStatsCollector::addCounter(
            "device_memory",
            "reclamation_receipts",
            1.0,
            "model_lifecycle",
            receipt.device.toString(),
            {{"intent", to_string(receipt.intent)},
             {"driver_free_before_bytes",
              std::to_string(receipt.driver_free_bytes_before)},
             {"driver_free_before_owner_release_bytes",
              std::to_string(
                  receipt.driver_free_bytes_before_owner_release)},
             {"driver_free_after_bytes",
              std::to_string(receipt.driver_free_bytes_after)},
             {"reclaimed_driver_bytes",
              std::to_string(receipt.reclaimedDriverBytes())},
             {"driver_bytes_visible_since_owner_release",
              std::to_string(
                  receipt.driverBytesVisibleSinceOwnerRelease())},
             {"expected_retired_bytes",
              std::to_string(receipt.expected_retired_bytes)},
             {"graph_used_before_bytes",
              std::to_string(receipt.graph_used_bytes_before)},
             {"graph_used_after_bytes",
              std::to_string(receipt.graph_used_bytes_after)},
             {"graph_reserved_before_bytes",
              std::to_string(receipt.graph_reserved_bytes_before)},
             {"graph_reserved_after_bytes",
              std::to_string(receipt.graph_reserved_bytes_after)},
             {"async_pool_used_before_bytes",
              std::to_string(receipt.async_pool_used_bytes_before)},
             {"async_pool_used_after_bytes",
              std::to_string(receipt.async_pool_used_bytes_after)},
             {"async_pool_reserved_before_bytes",
              std::to_string(receipt.async_pool_reserved_bytes_before)},
             {"async_pool_reserved_after_bytes",
              std::to_string(receipt.async_pool_reserved_bytes_after)}});
        return receipt;
    }

    PinnedHostTransferBuffer::PinnedHostTransferBuffer(
        size_t bytes,
        DeviceId registration_device)
        : bytes_(bytes),
          registration_device_(registration_device)
    {
        if (bytes_ == 0 || !registration_device_.is_gpu())
        {
            throw std::invalid_argument(
                "PinnedHostTransferBuffer requires positive bytes and a GPU registration device");
        }
    }

    void PinnedHostTransferBuffer::bind(IBackend *backend)
    {
        if (!backend)
        {
            throw std::invalid_argument(
                "PinnedHostTransferBuffer::bind requires the exact backend");
        }
        if (allocation_)
        {
            if (backend_ == backend)
                return;
            throw std::logic_error(
                "PinnedHostTransferBuffer cannot change backend after binding");
        }

        void *const allocation = backend->allocatePinned(
            bytes_, registration_device_.gpu_ordinal());
        if (!allocation)
        {
            throw std::runtime_error(
                "PinnedHostTransferBuffer allocation failed for " +
                registration_device_.toString() +
                " bytes=" + std::to_string(bytes_));
        }

        /*
         * A captured copy can execute during the transaction that creates the
         * graph.  Deterministic initialization makes any missing protocol
         * publication observable as zeros instead of exposing stale host
         * pages while the packet-level correctness checks diagnose the fault.
         */
        std::memset(allocation, 0, bytes_);
        backend_ = backend;
        allocation_ = allocation;
        ownership_ = Ownership::BackendAllocation;
    }

    void PinnedHostTransferBuffer::bindExternal(
        IBackend *backend,
        void *allocation,
        std::shared_ptr<void> lifetime)
    {
        if (!backend || !allocation || !lifetime)
        {
            throw std::invalid_argument(
                "PinnedHostTransferBuffer::bindExternal requires a backend, address, and lifetime");
        }
        if (allocation_ || ownership_ != Ownership::Unbound)
        {
            throw std::logic_error(
                "PinnedHostTransferBuffer external registration cannot be rebound");
        }
        if (!backend->pinHostMemory(
                allocation, bytes_, registration_device_.gpu_ordinal()))
        {
            throw std::runtime_error(
                "PinnedHostTransferBuffer external registration failed for " +
                registration_device_.toString() +
                " bytes=" + std::to_string(bytes_));
        }
        backend_ = backend;
        allocation_ = allocation;
        ownership_ = Ownership::ExternalRegistration;
        external_lifetime_ = std::move(lifetime);
    }

    PinnedHostTransferBuffer::~PinnedHostTransferBuffer()
    {
        if (allocation_ && backend_ && registration_device_.is_gpu())
        {
            if (ownership_ == Ownership::BackendAllocation)
            {
                backend_->freePinned(
                    allocation_, registration_device_.gpu_ordinal());
            }
            else if (ownership_ == Ownership::ExternalRegistration)
            {
                if (!backend_->unpinHostMemory(
                        allocation_, registration_device_.gpu_ordinal()))
                {
                    LOG_ERROR(
                        "PinnedHostTransferBuffer could not unregister external pages for "
                        << registration_device_.toString());
                    std::terminate();
                }
            }
        }
        allocation_ = nullptr;
        bytes_ = 0;
        backend_ = nullptr;
        registration_device_ = DeviceId::invalid();
        ownership_ = Ownership::Unbound;
        external_lifetime_.reset();
    }

    void *PinnedHostTransferBuffer::mutableData(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0))
        {
            throw std::out_of_range(
                "PinnedHostTransferBuffer mutable access requires a bound in-range allocation");
        }
        return static_cast<void *>(
            static_cast<unsigned char *>(allocation_) + offset);
    }

    const void *PinnedHostTransferBuffer::data(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0))
        {
            throw std::out_of_range(
                "PinnedHostTransferBuffer access requires a bound in-range allocation");
        }
        return static_cast<const void *>(
            static_cast<const unsigned char *>(allocation_) + offset);
    }

    DeviceTransferBuffer::DeviceTransferBuffer(
        size_t bytes,
        DeviceId device)
        : bytes_(bytes), device_(device)
    {
        if (bytes_ == 0u || !device_.is_gpu())
        {
            throw std::invalid_argument(
                "DeviceTransferBuffer requires positive bytes and an exact GPU");
        }
    }

    void DeviceTransferBuffer::bind(IBackend *backend)
    {
        if (!backend)
        {
            throw std::invalid_argument(
                "DeviceTransferBuffer::bind requires the exact backend");
        }
        if (allocation_)
        {
            if (backend_ == backend)
                return;
            throw std::logic_error(
                "DeviceTransferBuffer cannot change backend after binding");
        }
        void *const allocation = backend->allocate(
            bytes_, device_.gpu_ordinal());
        if (!allocation)
        {
            throw std::runtime_error(
                "DeviceTransferBuffer allocation failed for " +
                device_.toString() + " bytes=" + std::to_string(bytes_));
        }
        backend_ = backend;
        allocation_ = allocation;
    }

    DeviceTransferBuffer::~DeviceTransferBuffer()
    {
        if (allocation_ && backend_ && device_.is_gpu())
            backend_->free(allocation_, device_.gpu_ordinal());
        allocation_ = nullptr;
        bytes_ = 0u;
        device_ = DeviceId::invalid();
        backend_ = nullptr;
    }

    void *DeviceTransferBuffer::mutableDeviceData(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0u))
        {
            throw std::out_of_range(
                "DeviceTransferBuffer mutable access requires a bound in-range allocation");
        }
        return static_cast<void *>(
            static_cast<unsigned char *>(allocation_) + offset);
    }

    PersistentTransferStagingSlice::PersistentTransferStagingSlice(
        std::shared_ptr<MappedHostTransferRegion> mapped,
        std::shared_ptr<DeviceTransferBuffer> device_storage,
        size_t offset,
        size_t bytes,
        DeviceId device) noexcept
        : mapped_(std::move(mapped)),
          device_storage_(std::move(device_storage)),
          offset_(offset),
          bytes_(bytes),
          device_(device)
    {
    }

    bool PersistentTransferStagingSlice::valid() const noexcept
    {
        return mapped_ && device_storage_ && mapped_->isBound() &&
               device_storage_->isBound() && device_.is_gpu() && bytes_ > 0u &&
               mapped_->hasDevice(device_) &&
               device_storage_->device() == device_ &&
               mapped_->contains(offset_, bytes_) &&
               device_storage_->contains(offset_, bytes_);
    }

    void *PersistentTransferStagingSlice::mutablePinnedData() const
    {
        if (!valid())
        {
            throw std::logic_error(
                "Persistent transfer staging slice has no valid pinned region");
        }
        return mapped_->mutableHostData(offset_);
    }

    void *PersistentTransferStagingSlice::mutableDeviceData() const
    {
        if (!valid())
        {
            throw std::logic_error(
                "Persistent transfer staging slice has no valid device region");
        }
        return device_storage_->mutableDeviceData(offset_);
    }

    PersistentTransferExecutionLane::PersistentTransferExecutionLane(
        std::shared_ptr<void> pool_lifetime,
        void *stream,
        DeviceId device,
        size_t lane_index) noexcept
        : pool_lifetime_(std::move(pool_lifetime)),
          stream_(stream),
          device_(device),
          lane_index_(lane_index)
    {
    }

    bool PersistentTransferExecutionLane::valid() const noexcept
    {
        return pool_lifetime_ && stream_ && device_.is_gpu();
    }

    void *PersistentTransferExecutionLane::stream() const
    {
        if (!valid())
        {
            throw std::logic_error(
                "Persistent transfer execution lane is incomplete");
        }
        return stream_;
    }

    const void *DeviceTransferBuffer::deviceData(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0u))
        {
            throw std::out_of_range(
                "DeviceTransferBuffer access requires a bound in-range allocation");
        }
        return static_cast<const void *>(
            static_cast<const unsigned char *>(allocation_) + offset);
    }

    MappedHostTransferRegion::MappedHostTransferRegion(
        void *allocation,
        size_t bytes,
        std::span<const DeviceId> devices,
        std::shared_ptr<void> lifetime,
        BackingKind backing_kind)
        : allocation_(allocation),
          bytes_(bytes),
          devices_(devices.begin(), devices.end()),
          lifetime_(std::move(lifetime)),
          backing_kind_(backing_kind)
    {
        if (!allocation_ || bytes_ == 0u || devices_.empty() ||
            !lifetime_)
        {
            throw std::invalid_argument(
                "MappedHostTransferRegion requires stable pages, positive bytes, endpoints, and a retained lifetime");
        }
        std::sort(devices_.begin(), devices_.end());
        for (std::size_t index = 0; index < devices_.size(); ++index)
        {
            if (!devices_[index].is_valid())
            {
                throw std::invalid_argument(
                    "MappedHostTransferRegion contains an invalid endpoint");
            }
            if (index != 0u && devices_[index] == devices_[index - 1u])
            {
                throw std::invalid_argument(
                    "MappedHostTransferRegion contains a duplicate endpoint " +
                    devices_[index].toString());
            }
        }
        aliases_.reserve(devices_.size());
        registrations_.reserve(2u);
    }

    MappedHostTransferRegion::~MappedHostTransferRegion()
    {
        /*
         * The enclosing graph family proves stream quiescence before releasing
         * this owner. External pages must be unregistered while their lifetime
         * is still held; backend-owned pages are freed by the lifetime deleter;
         * slices merely release their parent. Keeping those cases explicit
         * makes it impossible to unregister a native mapped allocation.
         */
        if (backing_kind_ == BackingKind::ExternalRegistration)
        {
            for (auto registration = registrations_.rbegin();
                 registration != registrations_.rend(); ++registration)
            {
                if (registration->backend &&
                    !registration->backend->unregisterExternalMappedHostMemory(
                        allocation_, registration->registration_ordinal))
                {
                    LOG_ERROR(
                        "MappedHostTransferRegion could not unregister external pages for backend family "
                        << static_cast<int>(registration->type)
                        << " registration_device="
                        << registration->registration_ordinal);
                }
            }
        }
        else if (!registrations_.empty())
        {
            LOG_ERROR(
                "MappedHostTransferRegion non-external backing unexpectedly owns backend registrations");
        }
        bound_ = false;
        aliases_.clear();
        registrations_.clear();
        allocation_ = nullptr;
        bytes_ = 0u;
        devices_.clear();
        lifetime_.reset();
    }

    bool MappedHostTransferRegion::isBound() const noexcept
    {
        return bound_ && allocation_ && lifetime_ &&
               aliases_.size() == devices_.size();
    }

    void *MappedHostTransferRegion::mutableHostData(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0u))
        {
            throw std::out_of_range(
                "MappedHostTransferRegion host access requires a bound in-range region");
        }
        return static_cast<void *>(
            static_cast<unsigned char *>(allocation_) + offset);
    }

    void *MappedHostTransferRegion::deviceAlias(
        DeviceId device,
        size_t offset) const
    {
        if (!isBound() || !contains(offset, 0u))
        {
            throw std::out_of_range(
                "MappedHostTransferRegion alias access requires a bound in-range region");
        }
        const auto found = std::find_if(
            aliases_.begin(),
            aliases_.end(),
            [&](const DeviceAlias &alias)
            {
                return alias.device == device;
            });
        if (found == aliases_.end() || !found->address)
        {
            throw std::invalid_argument(
                "MappedHostTransferRegion has no alias for " +
                device.toString());
        }
        return static_cast<void *>(
            static_cast<unsigned char *>(found->address) + offset);
    }

    bool MappedHostTransferRegion::hasDevice(DeviceId device) const noexcept
    {
        return isBound() && std::any_of(
            aliases_.begin(),
            aliases_.end(),
            [&](const DeviceAlias &alias)
            {
                return alias.device == device && alias.address != nullptr;
            });
    }

    IBackend *MappedHostTransferRegion::backendFor(
        DeviceId device) const noexcept
    {
        const auto found = std::find_if(
            aliases_.begin(),
            aliases_.end(),
            [&](const DeviceAlias &alias)
            {
                return alias.device == device;
            });
        return found == aliases_.end() ? nullptr : found->backend;
    }

    MappedHostTransferArena::MappedHostTransferArena(
        IBackend *(*backend_resolver)(DeviceId),
        std::span<const DeviceId> devices)
        : backend_resolver_(backend_resolver),
          devices_(devices.begin(), devices.end())
    {
        if (devices_.empty() ||
            std::any_of(
                devices_.begin(),
                devices_.end(),
                [](DeviceId device) { return !device.is_gpu(); }))
        {
            throw std::invalid_argument(
                "MappedHostTransferArena requires a non-empty local GPU endpoint set");
        }
        std::sort(devices_.begin(), devices_.end());
        if (std::adjacent_find(devices_.begin(), devices_.end()) !=
            devices_.end())
        {
            throw std::invalid_argument(
                "MappedHostTransferArena contains a duplicate GPU endpoint");
        }
    }

    std::shared_ptr<MappedHostTransferRegion>
    MappedHostTransferArena::allocate(size_t bytes, size_t alignment)
    {
        if (bytes == 0u || alignment == 0u)
        {
            throw std::invalid_argument(
                "MappedHostTransferArena allocation requires positive bytes and alignment");
        }
        if (isGraphCaptureActive())
        {
            throw std::logic_error(
                "MappedHostTransferArena cannot grow or suballocate during GPU graph capture");
        }

        const auto aligned_offset =
            [alignment](const MappedHostTransferRegion &region,
                        size_t cursor) -> size_t
        {
            const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(
                region.mutableHostData());
            if (cursor >
                std::numeric_limits<std::uintptr_t>::max() - base)
            {
                throw std::overflow_error(
                    "MappedHostTransferArena address arithmetic overflow");
            }
            const std::uintptr_t address = base + cursor;
            const size_t remainder = static_cast<size_t>(address % alignment);
            const size_t padding = remainder == 0u ? 0u : alignment - remainder;
            if (cursor > std::numeric_limits<size_t>::max() - padding)
            {
                throw std::overflow_error(
                    "MappedHostTransferArena alignment arithmetic overflow");
            }
            return cursor + padding;
        };

        std::lock_guard<std::mutex> lock(mutex_);
        TransferEngine transfer(backend_resolver_);

        const auto allocate_from =
            [&](BackingRegion &backing)
                -> std::shared_ptr<MappedHostTransferRegion>
        {
            if (!backing.region || !backing.region->isBound())
            {
                throw std::logic_error(
                    "MappedHostTransferArena contains an unbound backing region");
            }
            const size_t offset =
                aligned_offset(*backing.region, backing.used_bytes);
            if (!backing.region->contains(offset, bytes))
                return nullptr;

            auto slice = transfer.sliceMappedHostRegion(
                backing.region, offset, bytes);
            const auto host_address = reinterpret_cast<std::uintptr_t>(
                slice->mutableHostData());
            if (host_address % alignment != 0u)
            {
                throw std::logic_error(
                    "MappedHostTransferArena produced a misaligned host slice");
            }
            for (const DeviceId device : devices_)
            {
                const auto device_address =
                    reinterpret_cast<std::uintptr_t>(
                        slice->deviceAlias(device));
                if (device_address % alignment != 0u)
                {
                    throw std::logic_error(
                        "MappedHostTransferArena backend alias does not preserve requested alignment for " +
                        device.toString());
                }
            }

            const size_t padding = offset - backing.used_bytes;
            backing.used_bytes = offset + bytes;
            if (allocated_bytes_ >
                    std::numeric_limits<size_t>::max() - bytes ||
                alignment_padding_bytes_ >
                    std::numeric_limits<size_t>::max() - padding)
            {
                throw std::overflow_error(
                    "MappedHostTransferArena allocation accounting overflow");
            }
            allocated_bytes_ += bytes;
            alignment_padding_bytes_ += padding;
            ++slice_count_;
            return slice;
        };

        /* Search newest-first so the current growth slab absorbs adjacent
         * ticket payloads while older captured addresses remain untouched. */
        for (auto backing = backing_regions_.rbegin();
             backing != backing_regions_.rend(); ++backing)
        {
            if (auto slice = allocate_from(*backing))
                return slice;
        }

        if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1u))
        {
            throw std::overflow_error(
                "MappedHostTransferArena backing capacity overflow");
        }
        const size_t minimum_backing_bytes = bytes + alignment - 1u;
        const size_t requested_backing_bytes =
            std::max(minimum_backing_bytes, committed_bytes_);
        auto region = transfer.allocateMappedHostRegion(
            requested_backing_bytes, devices_);
        if (!region || !region->isBound())
        {
            throw std::runtime_error(
                "MappedHostTransferArena could not bind a backing region");
        }
        if (committed_bytes_ >
            std::numeric_limits<size_t>::max() - region->sizeBytes())
        {
            throw std::overflow_error(
                "MappedHostTransferArena committed capacity overflow");
        }
        committed_bytes_ += region->sizeBytes();
        backing_regions_.push_back({
            .region = std::move(region),
            .used_bytes = 0u,
        });

        PerfStatsCollector::addCounter(
            "moe_overlay_activation_epoch",
            "mapped_arena_backing_regions",
            1.0,
            "setup",
            "heterogeneous",
            {{"growth", "geometric"},
             {"registration_cardinality", "logarithmic"}});
        PerfStatsCollector::addCounter(
            "moe_overlay_activation_epoch",
            "mapped_arena_committed_bytes",
            static_cast<double>(
                backing_regions_.back().region->sizeBytes()),
            "setup",
            "heterogeneous",
            {{"growth", "geometric"},
             {"registration_cardinality", "logarithmic"}});

        auto slice = allocate_from(backing_regions_.back());
        if (!slice)
        {
            throw std::logic_error(
                "MappedHostTransferArena newly committed backing cannot satisfy its triggering allocation");
        }
        return slice;
    }

    MappedHostTransferArena::Snapshot
    MappedHostTransferArena::snapshot() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            .committed_bytes = committed_bytes_,
            .allocated_bytes = allocated_bytes_,
            .alignment_padding_bytes = alignment_padding_bytes_,
            .backing_region_count = backing_regions_.size(),
            .slice_count = slice_count_,
        };
    }

    std::shared_ptr<PinnedHostTransferBuffer>
    TransferEngine::declarePinnedHostBuffer(
        size_t bytes,
        DeviceId registration_device) const
    {
        return std::shared_ptr<PinnedHostTransferBuffer>(
            new PinnedHostTransferBuffer(bytes, registration_device));
    }

    void TransferEngine::bindPinnedHostBuffer(
        PinnedHostTransferBuffer &buffer) const
    {
        IBackend *const backend = resolveBackend(buffer.registration_device_);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::bindPinnedHostBuffer has no backend for " +
                buffer.registration_device_.toString());
        }
        buffer.bind(backend);
    }

    std::shared_ptr<PinnedHostTransferBuffer>
    TransferEngine::allocatePinnedHostBuffer(
        size_t bytes,
        DeviceId registration_device) const
    {
        if (bytes == 0 || !registration_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::allocatePinnedHostBuffer requires positive bytes and a GPU registration device");
        }
        IBackend *const backend = resolveBackend(registration_device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::allocatePinnedHostBuffer has no backend for " +
                registration_device.toString());
        }
        auto buffer = declarePinnedHostBuffer(bytes, registration_device);
        buffer->bind(backend);
        return buffer;
    }

    std::shared_ptr<DeviceTransferBuffer>
    TransferEngine::allocateDeviceTransferBuffer(
        size_t bytes,
        DeviceId device) const
    {
        if (bytes == 0u || !device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::allocateDeviceTransferBuffer requires positive bytes and an exact GPU");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::allocateDeviceTransferBuffer has no backend for " +
                device.toString());
        }
        auto buffer = std::shared_ptr<DeviceTransferBuffer>(
            new DeviceTransferBuffer(bytes, device));
        buffer->bind(backend);
        return buffer;
    }

    std::vector<PersistentTransferStagingSlice>
    TransferEngine::allocatePersistentTransferStagingSlices(
        size_t bytes_per_slice,
        size_t slice_count,
        DeviceId device) const
    {
        if (bytes_per_slice == 0u || slice_count == 0u || !device.is_gpu())
        {
            throw std::invalid_argument(
                "Persistent transfer staging requires positive geometry and an exact GPU");
        }
        if (slice_count >
            std::numeric_limits<size_t>::max() / bytes_per_slice)
        {
            throw std::overflow_error(
                "Persistent transfer staging slab byte size overflowed");
        }
        const size_t total_bytes = bytes_per_slice * slice_count;
        const std::array devices{device};
        auto mapped = allocateMappedHostRegion(total_bytes, devices);
        auto device_storage = allocateDeviceTransferBuffer(total_bytes, device);

        std::vector<PersistentTransferStagingSlice> slices;
        slices.reserve(slice_count);
        for (size_t index = 0u; index < slice_count; ++index)
        {
            slices.push_back(PersistentTransferStagingSlice(
                mapped,
                device_storage,
                index * bytes_per_slice,
                bytes_per_slice,
                device));
        }
        return slices;
    }

    std::shared_ptr<MappedHostTransferRegion> TransferEngine::mappedStagingView(
        const PersistentTransferStagingSlice &staging) const
    {
        if (!staging.valid())
            throw std::invalid_argument("Mapped staging view requires a complete exclusive slice");
        return sliceMappedHostRegion(staging.mapped_, staging.offset_, staging.bytes_);
    }

    std::vector<PersistentTransferExecutionLane>
    TransferEngine::allocatePersistentTransferExecutionLanes(
        size_t lane_count,
        DeviceId device,
        const std::string &pool_name) const
    {
        if (isGraphCaptureActive())
            throw std::logic_error("Persistent transfer lanes must be prepared before graph capture");
        if (lane_count == 0u || !device.is_gpu() || pool_name.empty())
        {
            throw std::invalid_argument(
                "Persistent transfer execution lanes require positive geometry, an exact GPU, and a stable pool name");
        }

        auto streams = std::make_shared<std::vector<void *>>(
            lane_count, nullptr);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        size_t created_count = 0u;
        context.submitAndWait(
            [&]
            {
                // Resolve native function handles while building the pool.
                // Every lease returned below proves this setup edge completed;
                // no first-use module load can synchronize a live inference graph.
                if (isGraphCaptureActive())
                    throw std::logic_error("Transfer function preparation cannot enter a captured worker");
                auto *backend = resolveBackend(device);
                if (!backend || !backend->prepareMappedHostCopyKernels(device.gpu_ordinal()))
                    throw std::runtime_error(
                        "Persistent transfer pool could not prepare mapped-copy kernels on " +
                        device.toString());
                for (size_t index = 0u; index < lane_count; ++index)
                {
                    bool created = false;
                    (*streams)[index] = context.getOrCreateAuxiliaryStream(
                        "persistent_transfer_execution:" + pool_name + ":" +
                            device.toString() + ":lane:" +
                            std::to_string(index),
                        GPUAuxiliaryStreamSchedulingClass::
                            BackgroundMaintenance,
                        &created);
                    if (!(*streams)[index])
                    {
                        throw std::runtime_error(
                            "Persistent transfer execution pool could not materialize lane " +
                            std::to_string(index) + " on " +
                            device.toString());
                    }
                    created_count += created ? 1u : 0u;
                }
            });

        std::vector<PersistentTransferExecutionLane> lanes;
        lanes.reserve(lane_count);
        const std::shared_ptr<void> pool_lifetime = streams;
        for (size_t index = 0u; index < lane_count; ++index)
        {
            lanes.push_back(PersistentTransferExecutionLane(
                pool_lifetime,
                (*streams)[index],
                device,
                index));
        }
        PerfStatsCollector::addCounter(
            "transfer",
            "persistent_execution_stream_pools_materialized",
            1.0,
            "model_setup",
            device.toString(),
            {{"pool", pool_name},
             {"lane_count", std::to_string(lane_count)},
             {"new_stream_count", std::to_string(created_count)},
             {"stream_class", "background_maintenance"}});
        return lanes;
    }

    std::shared_ptr<PinnedHostTransferBuffer>
    TransferEngine::registerExternalPinnedHostBuffer(
        void *allocation,
        size_t bytes,
        DeviceId registration_device,
        std::shared_ptr<void> lifetime) const
    {
        if (!allocation || bytes == 0 || !registration_device.is_gpu() ||
            !lifetime)
        {
            throw std::invalid_argument(
                "TransferEngine::registerExternalPinnedHostBuffer requires stable pages, positive bytes, a GPU, and a lifetime");
        }
        IBackend *const backend = resolveBackend(registration_device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::registerExternalPinnedHostBuffer has no backend for " +
                registration_device.toString());
        }
        auto buffer = declarePinnedHostBuffer(bytes, registration_device);
        buffer->bindExternal(
            backend, allocation, std::move(lifetime));
        return buffer;
    }

    std::shared_ptr<MappedHostTransferRegion>
    TransferEngine::registerExternalMappedHostRegion(
        void *allocation,
        size_t bytes,
        std::span<const DeviceId> devices,
        std::shared_ptr<void> lifetime) const
    {
        auto region = std::shared_ptr<MappedHostTransferRegion>(
            new MappedHostTransferRegion(
                allocation,
                bytes,
                devices,
                std::move(lifetime),
                MappedHostTransferRegion::BackingKind::ExternalRegistration));

        for (const DeviceId device : region->devices_)
        {
            if (device.is_cpu())
            {
                region->aliases_.push_back({
                    .device = device,
                    .address = allocation,
                    .backend = nullptr,
                });
                continue;
            }
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    "TransferEngine mapped region endpoint is neither CPU nor GPU");
            }

            IBackend *const backend = resolveBackend(device);
            if (!backend)
            {
                throw std::runtime_error(
                    "TransferEngine mapped region has no backend for " +
                    device.toString());
            }
            auto registration = std::find_if(
                region->registrations_.begin(),
                region->registrations_.end(),
                [&](const MappedHostTransferRegion::BackendRegistration &item)
                {
                    return item.type == device.type;
                });
            if (registration == region->registrations_.end())
            {
                const size_t family_endpoint_count =
                    static_cast<size_t>(std::count_if(
                        region->devices_.begin(),
                        region->devices_.end(),
                        [&](DeviceId endpoint)
                        { return endpoint.type == device.type; }));
                const MappedHostRegistrationScope scope =
                    family_endpoint_count == 1u
                        ? MappedHostRegistrationScope::DeviceLocal
                        : MappedHostRegistrationScope::BackendPortable;
                if (!backend->registerExternalMappedHostMemory(
                        allocation,
                        bytes,
                        device.gpu_ordinal(),
                        scope))
                {
                    throw std::runtime_error(
                        "TransferEngine could not register mapped external pages for " +
                        device.toString());
                }
                region->registrations_.push_back({
                    .type = device.type,
                    .backend = backend,
                    .registration_ordinal = device.gpu_ordinal(),
                    .scope = scope,
                });
            }
            else if (registration->backend != backend)
            {
                throw std::logic_error(
                    "TransferEngine resolved multiple backend authorities for one mapped device family");
            }

            void *device_alias = nullptr;
            if (!backend->externalMappedHostDevicePointer(
                    allocation,
                    device.gpu_ordinal(),
                    &device_alias) ||
                !device_alias)
            {
                throw std::runtime_error(
                    "TransferEngine could not resolve mapped external alias for " +
                    device.toString());
            }
            region->aliases_.push_back({
                .device = device,
                .address = device_alias,
                .backend = backend,
            });
        }
        region->bound_ = true;
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            const PerfStatsCollector::Tags tags{
                {"scope", "node_local"},
                {"mapping", "typed_external_host_pages"},
                {"blocking", "false"},
            };
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "mapped_regions_registered",
                1.0,
                "setup",
                "heterogeneous",
                tags);
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "mapped_endpoint_aliases",
                static_cast<double>(region->aliases_.size()),
                "setup",
                "heterogeneous",
                tags);
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "mapped_backend_families",
                static_cast<double>(region->registrations_.size()),
                "setup",
                "heterogeneous",
                tags);
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "mapped_region_bytes",
                static_cast<double>(bytes),
                "setup",
                "heterogeneous",
                tags);
            for (const auto &registration : region->registrations_)
            {
                const PerfStatsCollector::Tags registration_tags{
                    {"scope", "node_local"},
                    {"registration_scope", to_string(registration.scope)},
                    {"backend", registration.type == DeviceType::CUDA
                                    ? "cuda"
                                    : "rocm"},
                    {"blocking", "false"},
                };
                PerfStatsCollector::addCounter(
                    "moe_overlay_activation_epoch",
                    "mapped_backend_registrations",
                    1.0,
                    "setup",
                    "heterogeneous",
                    registration_tags);
            }
        }
        return region;
    }

    std::shared_ptr<MappedHostTransferRegion>
    TransferEngine::sliceMappedHostRegion(
        std::shared_ptr<MappedHostTransferRegion> parent,
        size_t offset,
        size_t bytes) const
    {
        if (!parent || !parent->isBound() || bytes == 0u ||
            !parent->contains(offset, bytes))
        {
            throw std::invalid_argument(
                "TransferEngine mapped-host slice requires a bound parent and positive in-range geometry");
        }

        void *const host_start = parent->mutableHostData(offset);
        std::shared_ptr<void> parent_lifetime = parent;
        auto slice = std::shared_ptr<MappedHostTransferRegion>(
            new MappedHostTransferRegion(
                host_start,
                bytes,
                parent->devices_,
                std::move(parent_lifetime),
                MappedHostTransferRegion::BackingKind::ParentSlice));
        slice->aliases_.reserve(parent->aliases_.size());
        for (const auto &alias : parent->aliases_)
        {
            slice->aliases_.push_back({
                .device = alias.device,
                .address = static_cast<void *>(
                    static_cast<unsigned char *>(alias.address) + offset),
                .backend = alias.backend,
            });
        }
        /* A slice owns no native registration. Its external lifetime retains
         * the parent, whose destructor unregisters only after the final child
         * and arena owner disappear. */
        slice->bound_ = true;

        PerfStatsCollector::addCounter(
            "moe_overlay_activation_epoch",
            "mapped_arena_slices",
            1.0,
            "setup",
            "heterogeneous",
            {{"mapping", "parent_subregion"},
             {"native_registration", "false"}});
        PerfStatsCollector::addCounter(
            "moe_overlay_activation_epoch",
            "mapped_arena_slice_bytes",
            static_cast<double>(bytes),
            "setup",
            "heterogeneous",
            {{"mapping", "parent_subregion"},
             {"native_registration", "false"}});
        return slice;
    }

    std::shared_ptr<MappedHostTransferRegion>
    TransferEngine::allocateMappedHostRegion(
        size_t bytes,
        std::span<const DeviceId> devices) const
    {
        if (bytes == 0u || devices.empty() ||
            std::any_of(
                devices.begin(),
                devices.end(),
                [](DeviceId device) { return !device.is_gpu(); }))
        {
            throw std::invalid_argument(
                "TransferEngine::allocateMappedHostRegion requires positive bytes and local GPU endpoints");
        }
        const long page_size_value = ::sysconf(_SC_PAGESIZE);
        if (page_size_value <= 0)
        {
            throw std::runtime_error(
                "TransferEngine could not resolve the host page size");
        }
        const size_t page_size = static_cast<size_t>(page_size_value);
        if (bytes > std::numeric_limits<size_t>::max() - (page_size - 1u))
        {
            throw std::overflow_error(
                "TransferEngine mapped-host capacity alignment overflow");
        }
        const size_t mapping_bytes =
            ((bytes + page_size - 1u) / page_size) * page_size;

        if (devices.size() == 1u)
        {
            const DeviceId device = devices.front();
            IBackend *const backend = resolveBackend(device);
            if (!backend)
            {
                throw std::runtime_error(
                    "TransferEngine mapped-host allocation has no backend for " +
                    device.toString());
            }

            /*
             * A native mapped allocation is not merely a convenience wrapper.
             * On Vega20, registering an already-faulted anonymous range emits
             * an ATS invalidation interrupt for roughly every host page. Four
             * participants materializing retained graph tickets concurrently
             * can therefore overrun the fixed 4 KiB IH2 retry ring. KFD-owned
             * hipHostMallocMapped pages establish the same zero-copy alias
             * without that interrupt storm; cudaHostAllocMapped gives CUDA the
             * symmetric ownership contract. The backend call is setup-only and
             * serialized by its runtime-lifecycle authority.
             */
            void *device_alias = nullptr;
            void *const allocation = backend->allocateMapped(
                mapping_bytes, device.gpu_ordinal(), &device_alias);
            if (!allocation || !device_alias)
            {
                if (allocation)
                    backend->freeMapped(allocation, device.gpu_ordinal());
                throw std::runtime_error(
                    "TransferEngine backend-owned mapped allocation failed for " +
                    device.toString());
            }
            auto lifetime = std::shared_ptr<void>(
                allocation,
                [backend, ordinal = device.gpu_ordinal()](void *address)
                {
                    backend->freeMapped(address, ordinal);
                });

            /* The calling setup thread owns NUMA placement. Touch the complete
             * allocation once before capture so neither CPU service nor a GPU
             * mapped write encounters a first-use host fault. */
            std::memset(allocation, 0, mapping_bytes);
            auto region = std::shared_ptr<MappedHostTransferRegion>(
                new MappedHostTransferRegion(
                    allocation,
                    mapping_bytes,
                    devices,
                    std::move(lifetime),
                    MappedHostTransferRegion::BackingKind::BackendAllocation));
            region->aliases_.push_back({
                .device = device,
                .address = device_alias,
                .backend = backend,
            });
            region->bound_ = true;

            if (PerfStatsCollector::isDomainEnabled(
                    "moe_overlay_activation_epoch"))
            {
                const PerfStatsCollector::Tags tags{
                    {"scope", "node_local"},
                    {"mapping", "backend_owned_mapped_host_pages"},
                    {"backend", device.type == DeviceType::CUDA
                                    ? "cuda"
                                    : "rocm"},
                    {"blocking", "false"},
                };
                PerfStatsCollector::addCounter(
                    "moe_overlay_activation_epoch",
                    "mapped_regions_allocated",
                    1.0,
                    "setup",
                    "heterogeneous",
                    tags);
                PerfStatsCollector::addCounter(
                    "moe_overlay_activation_epoch",
                    "mapped_endpoint_aliases",
                    1.0,
                    "setup",
                    "heterogeneous",
                    tags);
                PerfStatsCollector::addCounter(
                    "moe_overlay_activation_epoch",
                    "mapped_backend_families",
                    1.0,
                    "setup",
                    "heterogeneous",
                    tags);
                PerfStatsCollector::addCounter(
                    "moe_overlay_activation_epoch",
                    "mapped_backend_allocations",
                    1.0,
                    "setup",
                    "heterogeneous",
                    tags);
                PerfStatsCollector::addCounter(
                    "moe_overlay_activation_epoch",
                    "mapped_region_bytes",
                    static_cast<double>(mapping_bytes),
                    "setup",
                    "heterogeneous",
                    tags);
            }
            return region;
        }

        void *const mapping = ::mmap(
            nullptr,
            mapping_bytes,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS,
            -1,
            0);
        if (mapping == MAP_FAILED)
        {
            throw std::runtime_error(
                "TransferEngine anonymous mapped-host allocation failed");
        }
        auto lifetime = std::shared_ptr<void>(
            mapping,
            [mapping_bytes](void *address)
            {
                if (address && address != MAP_FAILED)
                    (void)::munmap(address, mapping_bytes);
            });
#if defined(MADV_HUGEPAGE)
        (void)::madvise(mapping, mapping_bytes, MADV_HUGEPAGE);
#endif
        /* Current-thread first touch is the only reliable placement authority
         * on the production NUMA host; callers establish setup affinity first. */
        std::memset(mapping, 0, mapping_bytes);
        return registerExternalMappedHostRegion(
            mapping,
            mapping_bytes,
            devices,
            std::move(lifetime));
    }

    std::vector<std::shared_ptr<MappedHostTransferRegion>>
    TransferEngine::allocateMappedHostTransferSlices(
        size_t bytes_per_slice,
        size_t slice_count,
        DeviceId device) const
    {
        if (bytes_per_slice == 0u || slice_count == 0u || !device.is_gpu())
        {
            throw std::invalid_argument(
                "Mapped host transfer slices require positive geometry and an exact GPU");
        }
        const long page_size_value = ::sysconf(_SC_PAGESIZE);
        if (page_size_value <= 0)
        {
            throw std::runtime_error(
                "Mapped host transfer slices could not resolve the host page size");
        }
        const size_t page_size = static_cast<size_t>(page_size_value);
        if (bytes_per_slice >
            std::numeric_limits<size_t>::max() - (page_size - 1u))
        {
            throw std::overflow_error(
                "Mapped host transfer slice alignment overflowed");
        }
        const size_t stride =
            ((bytes_per_slice + page_size - 1u) / page_size) * page_size;
        if (slice_count > std::numeric_limits<size_t>::max() / stride)
        {
            throw std::overflow_error(
                "Mapped host transfer slab byte size overflowed");
        }
        const size_t total_bytes = stride * slice_count;
        const std::array<DeviceId, 1> endpoints{device};
        auto slab = allocateMappedHostRegion(total_bytes, endpoints);
        if (!slab || !slab->isBound() || !slab->hasDevice(device))
        {
            throw std::runtime_error(
                "Mapped host transfer slab did not bind its exact GPU endpoint");
        }

        std::vector<std::shared_ptr<MappedHostTransferRegion>> slices;
        slices.reserve(slice_count);
        for (size_t index = 0u; index < slice_count; ++index)
        {
            auto slice = sliceMappedHostRegion(
                slab, index * stride, bytes_per_slice);
            if (!slice || !slice->isBound() ||
                !slice->hasDevice(device) ||
                slice->sizeBytes() != bytes_per_slice)
            {
                throw std::logic_error(
                    "Mapped host transfer slab produced an incomplete slice");
            }
            slices.push_back(std::move(slice));
        }

        PerfStatsCollector::addCounter(
            "transfer",
            "mapped_host_transfer_pools_materialized",
            1.0,
            "model_setup",
            device.toString(),
            {{"slice_count", std::to_string(slice_count)},
             {"slice_bytes", std::to_string(bytes_per_slice)},
             {"slab_bytes", std::to_string(total_bytes)},
             {"native_allocation_count", "1"}});
        return slices;
    }

    std::shared_ptr<MappedHostTransferArena>
    TransferEngine::createMappedHostArena(
        std::span<const DeviceId> devices) const
    {
        return std::shared_ptr<MappedHostTransferArena>(
            new MappedHostTransferArena(resolve_, devices));
    }

    void *TransferEngine::resolveMappedTimelineKernelSignal64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device,
        const char *operation) const
    {
        if (!operation || operation[0] == '\0' || !region.isBound() ||
            !device.is_gpu() || value == 0u ||
            !region.contains(signal_offset, sizeof(std::uint64_t)) ||
            !region.hasDevice(device))
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline kernel binding requires a bound region, aligned signal, positive value, and declared GPU endpoint");
        }
        void *const signal = region.deviceAlias(device, signal_offset);
        if ((reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline kernel signal is not 64-bit aligned");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            !backend->supportsStreamTimelineSignal64(device.gpu_ordinal()))
        {
            throw std::runtime_error(
                std::string("TransferEngine could not bind mapped timeline kernel ") +
                operation + " for " + device.toString());
        }
        return signal;
    }

    MappedTimelineKernelWait64Binding
    TransferEngine::bindMappedTimelineKernelWait64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device) const
    {
        auto *const signal = static_cast<const std::uint64_t *>(
            resolveMappedTimelineKernelSignal64(
                region, signal_offset, value, device, "wait"));
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "device_timeline_kernel_wait_bindings",
                1.0,
                "graph_setup",
                device.toString(),
                {{"scope", "node_local"},
                 {"ordering", "fused_packet_system_acquire"},
                 {"host_blocking", "false"}});
        }
        return MappedTimelineKernelWait64Binding(signal, value, device);
    }

    MappedTimelineKernelPublish64Binding
    TransferEngine::bindMappedTimelineKernelPublish64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device) const
    {
        auto *const signal = static_cast<std::uint64_t *>(
            resolveMappedTimelineKernelSignal64(
                region, signal_offset, value, device, "publication"));
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "device_timeline_kernel_publication_bindings",
                1.0,
                "graph_setup",
                device.toString(),
                {{"scope", "node_local"},
                 {"ordering", "fused_packet_system_release"},
                 {"host_blocking", "false"}});
        }
        return MappedTimelineKernelPublish64Binding(signal, value, device);
    }

    void TransferEngine::enqueueMappedTimelineWait64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device,
        void *stream) const
    {
        if (!region.isBound() || !device.is_gpu() || !stream || value == 0u ||
            !region.contains(signal_offset, sizeof(std::uint64_t)))
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline wait requires a bound region, aligned signal, positive value, GPU, and exact stream");
        }
        void *const signal = region.deviceAlias(device, signal_offset);
        if ((reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline wait signal is not 64-bit aligned");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            !backend->supportsStreamTimelineSignal64(device.gpu_ordinal()) ||
            !backend->streamWaitTimelineSignal64(
                stream,
                signal,
                value,
                device.gpu_ordinal()))
        {
            throw std::runtime_error(
                "TransferEngine could not enqueue mapped 64-bit timeline wait for " +
                device.toString());
        }
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "device_timeline_waits_enqueued",
                1.0,
                "device_epoch",
                device.toString(),
                {
                    {"scope", "node_local"},
                    {"ordering", "exact_stream_64bit_geq"},
                    {"host_blocking", "false"},
                });
        }
    }

    void TransferEngine::enqueueMappedTimelinePublish64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device,
        void *stream) const
    {
        if (!region.isBound() || !device.is_gpu() || !stream || value == 0u ||
            !region.contains(signal_offset, sizeof(std::uint64_t)))
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline publication requires a bound region, aligned signal, positive value, GPU, and exact stream");
        }
        void *const signal = region.deviceAlias(device, signal_offset);
        if ((reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline publication signal is not 64-bit aligned");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            !backend->supportsStreamTimelineSignal64(device.gpu_ordinal()) ||
            !backend->streamPublishTimelineSignal64(
                stream,
                signal,
                value,
                device.gpu_ordinal()))
        {
            throw std::runtime_error(
                "TransferEngine could not enqueue mapped 64-bit timeline publication for " +
                device.toString());
        }
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "device_timeline_publications_enqueued",
                1.0,
                "device_epoch",
                device.toString(),
                {
                    {"scope", "node_local"},
                    {"ordering", "exact_stream_64bit_fenced"},
                    {"host_blocking", "false"},
                });
        }
    }

    void TransferEngine::enqueueMappedHostToDevice(
        const MappedHostTransferRegion &region,
        size_t source_offset,
        ITensor *destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!destination || !region.isBound() || !device.is_gpu() ||
            !stream || bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueMappedHostToDevice requires a bound region, destination, positive bytes, GPU, and exact stream");
        }
        if (!region.contains(source_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueMappedHostToDevice source region exceeds registered shared pages");
        }

        TensorBase *const destination_owner = requireTransferStorageOwner(
            destination,
            "TransferEngine::enqueueMappedHostToDevice");
        const size_t destination_bytes = destination_owner->size_bytes();
        if (destination_offset > destination_bytes ||
            bytes > destination_bytes - destination_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueueMappedHostToDevice destination region exceeds tensor storage");
        }
        requireDeviceOutput(destination_owner, device, stream);

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueMappedHostToDevice backend identity does not match the mapped registration");
        }
        auto *const destination_ptr =
            static_cast<unsigned char *>(destination_owner->gpu_data_ptr()) +
            destination_offset;
        if (!backend->hostToDeviceOnStream(
                destination_ptr,
                region.mutableHostData(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueMappedHostToDevice failed to enqueue the exact H2D copy");
        }
        publishDeviceWrite(destination_owner, device, stream);
    }

    void TransferEngine::enqueueMappedHostToDevice(
        const MappedHostTransferRegion &region,
        size_t source_offset,
        DeviceTransferBuffer &destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!region.isBound() || !destination.isBound() ||
            !device.is_gpu() || !stream || bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueMappedHostToDevice transport scratch requires bound regions, positive bytes, a GPU, and an exact stream");
        }
        if (!region.contains(source_offset, bytes) ||
            !destination.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueMappedHostToDevice transport scratch exceeds a fixed region");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            backend != destination.backend_ || destination.device_ != device)
        {
            throw std::runtime_error(
                "TransferEngine::enqueueMappedHostToDevice transport scratch backend identity mismatch");
        }
        if (!backend->hostToDeviceOnStream(
                destination.mutableDeviceData(destination_offset),
                region.mutableHostData(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueMappedHostToDevice transport scratch copy enqueue failed");
        }
    }

    void TransferEngine::enqueueDeviceToMappedHost(
        ITensor *source,
        size_t source_offset,
        const MappedHostTransferRegion &region,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source || !region.isBound() || !device.is_gpu() || !stream ||
            bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueDeviceToMappedHost requires a source, bound region, positive bytes, GPU, and exact stream");
        }
        if (!region.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost destination region exceeds registered shared pages");
        }

        TensorBase *const source_owner = requireTransferStorageOwner(
            source,
            "TransferEngine::enqueueDeviceToMappedHost");
        const size_t source_bytes = source_owner->size_bytes();
        if (source_offset > source_bytes || bytes > source_bytes - source_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost source region exceeds tensor storage");
        }
        requireDeviceInput(source_owner, device, stream);

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost backend identity does not match the mapped registration");
        }
        const auto *const source_ptr =
            static_cast<const unsigned char *>(source_owner->gpu_data_ptr()) +
            source_offset;
        if (!backend->deviceToHostOnStream(
                region.mutableHostData(destination_offset),
                source_ptr,
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost failed to enqueue the exact D2H copy");
        }
    }

    DeviceTransferInputFork TransferEngine::recordDeviceInputFork(
        ITensor *source,
        DeviceId device,
        void *producer_stream,
        void *event) const
    {
        if (!source || !device.is_gpu() || !producer_stream || !event)
        {
            throw std::invalid_argument(
                "TransferEngine::recordDeviceInputFork requires a tensor, GPU, exact producer stream, and retained event");
        }

        TensorBase *const source_owner = requireTransferStorageOwner(
            source,
            "TransferEngine::recordDeviceInputFork");
        /* Validate against the canonical graph stream before branching. The
         * dependency ledger must never be asked to reinterpret an auxiliary
         * stream as the tensor's producer. */
        requireDeviceInput(source_owner, device, producer_stream);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        if (!context.recordEventChecked(event, producer_stream))
        {
            throw std::runtime_error(
                "TransferEngine::recordDeviceInputFork could not record the producer frontier");
        }
        return DeviceTransferInputFork(
            source_owner, device, producer_stream, event);
    }

    AcquiredDeviceTransferInput TransferEngine::acquireDeviceInputFork(
        const DeviceTransferInputFork &publication,
        void *consumer_stream) const
    {
        if (!publication.valid() || !consumer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::acquireDeviceInputFork requires a valid publication and exact consumer stream");
        }
        auto &context =
            GPUDeviceContextPool::instance().getContext(publication.device_);
        if (!context.waitEventChecked(publication.event_, consumer_stream))
        {
            throw std::runtime_error(
                "TransferEngine::acquireDeviceInputFork could not enqueue the producer-event wait");
        }
        return AcquiredDeviceTransferInput(
            publication.source_owner_,
            publication.device_,
            consumer_stream,
            publication.event_);
    }

    void TransferEngine::enqueueDeviceToMappedHost(
        const AcquiredDeviceTransferInput &source,
        size_t source_offset,
        const MappedHostTransferRegion &region,
        size_t destination_offset,
        size_t bytes) const
    {
        if (!source.valid() || !region.isBound() || bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueDeviceToMappedHost requires an acquired fork, bound region, and positive byte count");
        }
        if (!region.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost fork destination exceeds registered shared pages");
        }
        const size_t source_bytes = source.source_owner_->size_bytes();
        if (source_offset > source_bytes || bytes > source_bytes - source_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost fork source exceeds tensor storage");
        }
        const auto *const source_ptr =
            static_cast<const unsigned char *>(
                source.source_owner_->gpu_data_ptr()) +
            source_offset;
        IBackend *const backend = resolveBackend(source.device_);
        if (!source_ptr || !backend ||
            backend != region.backendFor(source.device_))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost fork backend or stable tensor storage is incomplete");
        }
        if (!backend->deviceToHostOnStream(
                region.mutableHostData(destination_offset),
                source_ptr,
                bytes,
                source.device_.gpu_ordinal(),
                source.consumer_stream_))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost fork D2H enqueue failed");
        }
    }

    void TransferEngine::enqueueDeviceToMappedHost(
        const DeviceTransferBuffer &source,
        size_t source_offset,
        const MappedHostTransferRegion &region,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source.isBound() || !region.isBound() || !device.is_gpu() ||
            !stream || bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueDeviceToMappedHost transport scratch requires bound regions, positive bytes, a GPU, and an exact stream");
        }
        if (!source.contains(source_offset, bytes) ||
            !region.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost transport scratch exceeds a fixed region");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            backend != source.backend_ || source.device_ != device)
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost transport scratch backend identity mismatch");
        }
        if (!backend->deviceToHostOnStream(
                region.mutableHostData(destination_offset),
                source.deviceData(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost transport scratch copy enqueue failed");
        }
    }

    void TransferEngine::enqueuePersistentDeviceRegionToMappedHost(
        const void *source,
        size_t source_capacity,
        size_t source_offset,
        const MappedHostTransferRegion &destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source || !destination.isBound() || !device.is_gpu() ||
            !stream || bytes == 0u ||
            source_offset > source_capacity ||
            bytes > source_capacity - source_offset ||
            !destination.contains(destination_offset, bytes))
        {
            throw std::invalid_argument(
                "TransferEngine::enqueuePersistentDeviceRegionToMappedHost requires bounded storage, mapped pages, a GPU, and an exact stream");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != destination.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine persistent D2H backend identity mismatch");
        }
        const auto *const source_bytes =
            static_cast<const unsigned char *>(source) + source_offset;
        if (!backend->deviceToHostOnStream(
                destination.mutableHostData(destination_offset),
                source_bytes,
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine persistent D2H DMA enqueue failed");
        }
    }

    void TransferEngine::enqueueMappedHostToPersistentDeviceRegion(
        const MappedHostTransferRegion &source,
        size_t source_offset,
        void *destination,
        size_t destination_capacity,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source.isBound() || !destination || !device.is_gpu() ||
            !stream || bytes == 0u ||
            !source.contains(source_offset, bytes) ||
            destination_offset > destination_capacity ||
            bytes > destination_capacity - destination_offset)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueMappedHostToPersistentDeviceRegion requires mapped pages, bounded storage, a GPU, and an exact stream");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != source.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine persistent H2D backend identity mismatch");
        }
        auto *const destination_bytes =
            static_cast<unsigned char *>(destination) + destination_offset;
        if (!backend->hostToDeviceOnStream(
                destination_bytes,
                source.mutableHostData(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine persistent H2D DMA enqueue failed");
        }
    }

    void TransferEngine::enqueuePersistentDeviceRegionToMappedHostByKernel(
        const void *source,
        size_t source_capacity,
        size_t source_offset,
        const MappedHostTransferRegion &destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source || source_capacity == 0u || !destination.isBound() ||
            bytes == 0u || !device.is_gpu() || !stream)
        {
            throw std::invalid_argument(
                "TransferEngine progress-kernel transfer requires persistent source bytes, a bound mapped destination, GPU, and exact stream");
        }
        if (source_offset > source_capacity ||
            bytes > source_capacity - source_offset ||
            !destination.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine progress-kernel transfer exceeds a persistent region");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != destination.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine progress-kernel backend identity does not match mapped registration");
        }
        const auto *const source_bytes =
            static_cast<const std::uint8_t *>(source) + source_offset;
        if (!backend->copyDeviceVisibleRegionByKernelOnStream(
                destination.deviceAlias(device, destination_offset),
                source_bytes,
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine progress-kernel launch was rejected");
        }
    }

    void TransferEngine::enqueueBackgroundMappedCopy(
        const PersistentTransferExecutionLane &lane,
        MappedTransferDirection direction,
        void *device_region,
        size_t device_capacity,
        size_t device_offset,
        const MappedHostTransferRegion &mapped_region,
        size_t mapped_offset,
        size_t bytes) const
    {
        if (!lane.valid() || !device_region || !mapped_region.isBound() ||
            bytes == 0u ||
            (direction != MappedTransferDirection::DeviceToHost &&
             direction != MappedTransferDirection::HostToDevice))
            throw std::invalid_argument(
                "Background mapped copy requires a prepared lane, bounded owners and an exact direction");
        if (device_offset > device_capacity ||
            bytes > device_capacity - device_offset ||
            !mapped_region.contains(mapped_offset, bytes))
            throw std::out_of_range("Background mapped copy exceeds its immutable region");
        const DeviceId device = lane.device();
        IBackend *backend = resolveBackend(device);
        if (!backend || backend != mapped_region.backendFor(device))
            throw std::runtime_error("Background mapped copy backend identity mismatch");
        auto *device_bytes = static_cast<std::uint8_t *>(device_region) + device_offset;
        void *mapped_bytes = mapped_region.deviceAlias(device, mapped_offset);
        if (!backend->enqueueBackgroundMappedCopyOnStream(
                device_bytes, mapped_region.mutableHostData(mapped_offset), mapped_bytes,
                bytes, direction, device.gpu_ordinal(), lane.stream()))
            throw std::runtime_error("Background mapped copy submission failed");
    }

    bool TransferEngine::enqueueBackgroundStagingCopy(
        const PersistentTransferExecutionLane &lane,
        MappedTransferDirection direction,
        void *device_region,
        size_t device_capacity,
        size_t device_offset,
        const PersistentTransferStagingSlice &staging,
        size_t staging_offset,
        size_t bytes,
        std::string *error) const noexcept
    {
        if (error)
            error->clear();
        try
        {
            if (!staging.valid() || !lane.valid() ||
                staging.device() != lane.device())
                throw std::invalid_argument(
                    "Background staging copy requires matching prepared lane and slice owners");
            if (staging_offset > staging.bytes_ ||
                bytes > staging.bytes_ - staging_offset)
                throw std::out_of_range(
                    "Background staging copy exceeds its exclusive slice");
            // Bounds belong to the exclusive slice before translating into the
            // shared slab. A neighboring slot is never additional capacity.
            enqueueBackgroundMappedCopy(lane, direction, device_region,
                device_capacity, device_offset, *staging.mapped_,
                staging.offset_ + staging_offset, bytes);
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
        }
        catch (...)
        {
            if (error)
                *error = "Background staging copy failed with a non-standard exception";
        }
        return false;
    }

    std::shared_ptr<DeviceTransferBuffer> TransferEngine::allocateMappedTransferServiceCursors(
        size_t capacity, DeviceId device, void *stream) const
    {
        if (!capacity || !device.is_gpu() || !stream ||
            capacity > std::numeric_limits<size_t>::max() / sizeof(MappedTransferServiceCursor))
            throw std::invalid_argument("Mapped transfer service requires bounded capacity, one GPU and an exact setup stream");
        if (isGraphCaptureActive())
            throw std::logic_error("Mapped transfer service allocation is a cold setup operation");
        auto cursors = allocateDeviceTransferBuffer(capacity * sizeof(MappedTransferServiceCursor), device);
        auto *backend = resolveBackend(device);
        if (!backend || !backend->initializeMappedTransferService(
                static_cast<MappedTransferServiceCursor *>(cursors->mutableDeviceData()),
                capacity, device.gpu_ordinal(), stream))
            throw std::runtime_error("Mapped transfer service initialization failed");
        return cursors;
    }

    void TransferEngine::enqueueMappedTransferInterval(
        DeviceTransferBuffer &interval, MappedTransferInterval value,
        void *stream) const
    {
        if (!interval.isBound() || !interval.contains(0u, sizeof(std::uint32_t)) || !stream ||
            (value != MappedTransferInterval::Open && value != MappedTransferInterval::Closed))
            throw std::invalid_argument("Mapped transfer interval requires a bound private word, typed state and exact stream");
        auto *backend = resolveBackend(interval.device());
        if (!backend || !backend->enqueueMappedTransferInterval(
                static_cast<std::uint32_t *>(interval.mutableDeviceData()), value,
                interval.device().gpu_ordinal(), stream))
            throw std::runtime_error("Mapped transfer interval publication failed");
    }

    void TransferEngine::enqueueMappedTransferService(
        const MappedHostTransferRegion &inbox, DeviceTransferBuffer &cursors,
        size_t maximum_bytes, const DeviceTransferBuffer *interval,
        MappedTransferServiceRun run, void *stream) const
    {
        const auto device = cursors.device();
        const auto capacity = cursors.sizeBytes() / sizeof(MappedTransferServiceCursor);
        constexpr auto record_bytes = sizeof(MappedTransferProgressCommand) +
                                      sizeof(MappedTransferProgressCompletion);
        if (!cursors.isBound() || !inbox.isBound() || !inbox.hasDevice(device) ||
            !stream || !maximum_bytes || !capacity ||
            cursors.sizeBytes() % sizeof(MappedTransferServiceCursor) != 0u ||
            capacity > inbox.sizeBytes() / record_bytes ||
            (run != MappedTransferServiceRun::PublishedPass && run != MappedTransferServiceRun::CapturedInterval) ||
            ((run == MappedTransferServiceRun::CapturedInterval) != (interval != nullptr)) ||
            (interval && (!interval->isBound() || interval->device() != device ||
                          !interval->contains(0u, sizeof(std::uint32_t)))))
            throw std::invalid_argument("Mapped transfer service has invalid physical inbox, cursor, interval or stream ownership");
        auto *backend = resolveBackend(device);
        if (!backend || backend != inbox.backendFor(device) ||
            !backend->enqueueMappedTransferService(
                static_cast<const MappedTransferProgressCommand *>(inbox.deviceAlias(device)),
                static_cast<MappedTransferProgressCompletion *>(inbox.deviceAlias(
                    device, capacity * sizeof(MappedTransferProgressCommand))),
                static_cast<MappedTransferServiceCursor *>(cursors.mutableDeviceData()),
                capacity, maximum_bytes,
                interval ? static_cast<const std::uint32_t *>(interval->deviceData()) : nullptr,
                run, device.gpu_ordinal(), stream))
            throw std::runtime_error("Mapped transfer service launch rejected by the exact backend");
    }

    void TransferEngine::enqueueMappedTransferProgressClaims(
        const MappedHostTransferRegion &mapped_region,
        size_t command_offset,
        DeviceTransferBuffer &claims,
        size_t slot_capacity,
        DeviceId device,
        void *stream) const
    {
        if (!mapped_region.isBound() || !claims.isBound() ||
            slot_capacity == 0u ||
            !device.is_gpu() || !stream || claims.device() != device)
        {
            throw std::invalid_argument(
                "TransferEngine mapped progress claims require bound arrays, positive geometry, one GPU, and an exact stream");
        }
        if (slot_capacity >
            std::numeric_limits<size_t>::max() /
                sizeof(MappedTransferProgressCommand) ||
            slot_capacity >
                std::numeric_limits<size_t>::max() /
                    sizeof(MappedTransferProgressClaim))
        {
            throw std::overflow_error(
                "TransferEngine mapped progress claim geometry overflowed");
        }
        const size_t command_bytes =
            slot_capacity * sizeof(MappedTransferProgressCommand);
        const size_t claim_bytes =
            slot_capacity * sizeof(MappedTransferProgressClaim);
        if (!mapped_region.contains(command_offset, command_bytes) ||
            !claims.contains(0u, claim_bytes))
        {
            throw std::out_of_range(
                "TransferEngine mapped progress claims exceed fixed array storage");
        }

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != mapped_region.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine mapped progress claim backend identity does not match its registered pages");
        }
        const auto *const commands =
            static_cast<const MappedTransferProgressCommand *>(
                mapped_region.deviceAlias(device, command_offset));
        auto *const claim_array =
            static_cast<MappedTransferProgressClaim *>(
                claims.mutableDeviceData());
        if (!backend->enqueueMappedTransferProgressClaims(
                commands,
                claim_array,
                slot_capacity,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine mapped progress claim launch was rejected");
        }
    }

    void TransferEngine::enqueueMappedTransferProgressCopies(
        const DeviceTransferBuffer &claims,
        const MappedHostTransferRegion &mapped_region,
        size_t completion_offset,
        size_t slot_capacity,
        size_t maximum_bytes,
        DeviceId device,
        void *stream) const
    {
        if (!mapped_region.isBound() || !claims.isBound() ||
            slot_capacity == 0u || maximum_bytes == 0u ||
            !device.is_gpu() || !stream || claims.device() != device)
        {
            throw std::invalid_argument(
                "TransferEngine mapped progress copies require bound arrays, positive geometry, one GPU, and an exact stream");
        }
        if (slot_capacity >
                std::numeric_limits<size_t>::max() /
                    sizeof(MappedTransferProgressCompletion) ||
            slot_capacity >
                std::numeric_limits<size_t>::max() /
                    sizeof(MappedTransferProgressClaim))
        {
            throw std::overflow_error(
                "TransferEngine mapped progress copy geometry overflowed");
        }
        const size_t completion_bytes =
            slot_capacity * sizeof(MappedTransferProgressCompletion);
        const size_t claim_bytes =
            slot_capacity * sizeof(MappedTransferProgressClaim);
        if (!mapped_region.contains(completion_offset, completion_bytes) ||
            !claims.contains(0u, claim_bytes))
        {
            throw std::out_of_range(
                "TransferEngine mapped progress copies exceed fixed array storage");
        }

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != mapped_region.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine mapped progress copy backend identity does not match its registered pages");
        }
        const auto *const claim_array =
            static_cast<const MappedTransferProgressClaim *>(
                claims.deviceData());
        auto *const completions =
            static_cast<MappedTransferProgressCompletion *>(
                mapped_region.deviceAlias(device, completion_offset));
        if (!backend->enqueueMappedTransferProgressCopies(
                claim_array,
                completions,
                slot_capacity,
                maximum_bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine mapped progress copy launch was rejected");
        }
    }

    void TransferEngine::buildMappedTimelineTransaction(
        IGPUGraphCapture &destination,
        std::span<const MappedTimelineTransactionStep> ordered_steps,
        DeviceId device) const
    {
        if (!device.is_gpu() || ordered_steps.empty() ||
            !destination.executionStream() || destination.hasExecutable() ||
            destination.nodeCount() != 0u)
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline transaction requires an empty exact-device graph and non-empty steps");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend ||
            !backend->supportsStreamTimelineSignal64(device.gpu_ordinal()))
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline transaction requires native 64-bit timeline support for " +
                device.toString());
        }

        std::vector<GPUOrderedTimelineStep> lowered;
        lowered.reserve(ordered_steps.size());
        const auto requireName = [](const char *name)
        {
            return name && name[0] != '\0';
        };
        for (const auto &step : ordered_steps)
        {
            if (const auto *fragment =
                    std::get_if<MappedTimelineCapturedFragment>(&step))
            {
                if (!requireName(fragment->name) || !fragment->capture ||
                    fragment->capture == &destination ||
                    !fragment->capture->executionStream() ||
                    fragment->capture->nodeCount() == 0u)
                {
                    throw std::invalid_argument(
                        "TransferEngine mapped timeline fragment is incomplete or aliases its destination");
                }
                lowered.push_back({
                    .name = fragment->name,
                    .kind =
                        GPUOrderedTimelineStepKind::CapturedFragment,
                    .capture = fragment->capture,
                    .signal = nullptr,
                    .value = 0u,
                });
                continue;
            }

            const MappedHostTransferRegion *region = nullptr;
            const char *name = nullptr;
            size_t signal_offset = 0u;
            std::uint64_t value = 0u;
            GPUOrderedTimelineStepKind kind =
                GPUOrderedTimelineStepKind::WaitValue64;
            if (const auto *wait =
                    std::get_if<MappedTimelineWait64>(&step))
            {
                region = wait->region;
                name = wait->name;
                signal_offset = wait->signal_offset;
                value = wait->value;
            }
            else
            {
                const auto &publish =
                    std::get<MappedTimelinePublish64>(step);
                region = publish.region;
                name = publish.name;
                signal_offset = publish.signal_offset;
                value = publish.value;
                kind = GPUOrderedTimelineStepKind::PublishValue64;
            }
            if (!requireName(name) || !region || !region->isBound() ||
                !region->hasDevice(device) || value == 0u ||
                !region->contains(signal_offset, sizeof(std::uint64_t)) ||
                region->backendFor(device) != backend)
            {
                throw std::invalid_argument(
                    "TransferEngine mapped timeline step has incomplete region/device/value ownership");
            }
            void *const signal = region->deviceAlias(device, signal_offset);
            if ((reinterpret_cast<std::uintptr_t>(signal) &
                 (alignof(std::uint64_t) - 1u)) != 0u)
            {
                throw std::invalid_argument(
                    "TransferEngine mapped timeline graph signal is not 64-bit aligned");
            }
            lowered.push_back({
                .name = name,
                .kind = kind,
                .capture = nullptr,
                .signal = signal,
                .value = value,
            });
        }
        const GPUOrderedTimelineInstrumentation instrumentation =
            PerfStatsCollector::gpuStageEventTimingEnabled()
                ? GPUOrderedTimelineInstrumentation::PerStepEvents
                : GPUOrderedTimelineInstrumentation::Disabled;
        if (!destination.buildOrderedTimelineTransaction(
                lowered, instrumentation))
        {
            throw std::runtime_error(
                "TransferEngine could not lower mapped timeline transaction for " +
                device.toString());
        }
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "native_graph_timeline_transactions_built",
                1.0,
                "graph_setup",
                device.toString(),
                {
                    {"host_blocking", "false"},
                    {"ordering", "device_owned_graph_timeline_nodes"},
                    {"steps", std::to_string(lowered.size())},
                });
        }
    }

    void TransferEngine::enqueuePinnedHostToDevice(
        const PinnedHostTransferBuffer &pinned_source,
        size_t source_offset,
        ITensor *destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!destination || !device.is_gpu() || !stream || bytes == 0)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueuePinnedHostToDevice requires a destination, positive bytes, a GPU, and an exact stream");
        }
        if (pinned_source.registration_device_ != device ||
            !pinned_source.contains(source_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueuePinnedHostToDevice source region or registration device is invalid");
        }

        TensorBase *const destination_owner = requireTransferStorageOwner(
            destination,
            "TransferEngine::enqueuePinnedHostToDevice");
        const size_t destination_bytes = destination_owner->size_bytes();
        if (destination_offset > destination_bytes ||
            bytes > destination_bytes - destination_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueuePinnedHostToDevice destination region exceeds tensor storage");
        }
        requireDeviceOutput(destination_owner, device, stream);

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != pinned_source.backend_)
        {
            throw std::runtime_error(
                "TransferEngine::enqueuePinnedHostToDevice backend identity does not match the pinned registration");
        }
        auto *const destination_ptr =
            static_cast<unsigned char *>(destination_owner->gpu_data_ptr()) +
            destination_offset;
        if (!backend->hostToDeviceOnStream(
                destination_ptr,
                pinned_source.data(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueuePinnedHostToDevice failed to enqueue the exact H2D copy");
        }
        publishDeviceWrite(destination_owner, device, stream);
    }

    void TransferEngine::enqueueDeviceToPinnedHost(
        ITensor *source,
        size_t source_offset,
        const PinnedHostTransferBuffer &pinned_destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source || !device.is_gpu() || !stream || bytes == 0)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueDeviceToPinnedHost requires a source, positive bytes, a GPU, and an exact stream");
        }
        if (pinned_destination.registration_device_ != device ||
            !pinned_destination.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToPinnedHost destination region or registration device is invalid");
        }

        TensorBase *const source_owner = requireTransferStorageOwner(
            source,
            "TransferEngine::enqueueDeviceToPinnedHost");
        const size_t source_bytes = source_owner->size_bytes();
        if (source_offset > source_bytes || bytes > source_bytes - source_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToPinnedHost source region exceeds tensor storage");
        }
        requireDeviceInput(source_owner, device, stream);

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != pinned_destination.backend_)
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToPinnedHost backend identity does not match the pinned registration");
        }
        const auto *const source_ptr =
            static_cast<const unsigned char *>(source_owner->gpu_data_ptr()) +
            source_offset;
        if (!backend->deviceToHostOnStream(
                pinned_destination.mutableData(destination_offset),
                source_ptr,
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToPinnedHost failed to enqueue the exact D2H copy");
        }
    }

    // ============================================================================
    // Singleton
    // ============================================================================

    TransferEngine &TransferEngine::instance()
    {
        static TransferEngine engine;
        return engine;
    }

    void TransferEngine::prepareDeviceInput(
        ITensor *tensor,
        DeviceId target_device,
        void *stream)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceInput requires a GPU target");
        }
        if (!stream)
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceInput requires the exact "
                "non-null consumer stream");
        }
        if (isGraphCaptureActive())
        {
            throw std::logic_error(
                "TransferEngine::prepareDeviceInput is forbidden during GPU "
                "graph capture; capture consumers must use requireDeviceInput");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::prepareDeviceInput");
        if (!base->ensureOnDevice(target_device, stream))
        {
            throw std::runtime_error(
                "TransferEngine::prepareDeviceInput failed to place tensor on " +
                target_device.toString());
        }
        if (!base->gpu_data_ptr() ||
            !base->is_on_device(target_device))
        {
            throw std::runtime_error(
                "TransferEngine::prepareDeviceInput completed without storage on " +
                target_device.toString() +
                tensorTransferState(*base));
        }
    }

    void TransferEngine::requireDeviceInput(
        ITensor *tensor,
        DeviceId target_device,
        void *consumer_stream)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceInput requires a GPU target");
        }
        if (!consumer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceInput requires the exact "
                "non-null consumer stream");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::requireDeviceInput");
        std::lock_guard<std::mutex> lock(base->coherence_mutex_);

        if (!base->gpu_data_ptr_ ||
            !base->gpu_device_.has_value() ||
            *base->gpu_device_ != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput requires pre-existing "
                "storage on the exact device " +
                target_device.toString() + tensorTransferState(*base));
        }

        /*
         * An earlier stage in this exact capture transaction has recorded, but
         * not yet executed, the producer kernel.  Stable storage plus the frozen
         * topological ledger is the complete proof here; publishing global device
         * authority or importing an old event would both describe the wrong
         * generation.  Unknown and graph-external inputs continue through the
         * strict coherence/event checks below.
         */
        if (auto *ledger = currentGraphCaptureDependencyLedger())
        {
            const auto disposition = ledger->classifyInput(
                base, target_device, consumer_stream);
            if (disposition ==
                    GraphCaptureDependencyLedger::InputDisposition::InternalRecorded ||
                disposition ==
                    GraphCaptureDependencyLedger::InputDisposition::RetainedParentRecorded ||
                disposition ==
                    GraphCaptureDependencyLedger::InputDisposition::SetupAddressOnlyExternal)
            {
                /*
                 * A retained-parent import is deliberately no more globally
                 * authoritative than an ordinary earlier recorded producer.
                 * Its child graph records a stable pointer read, and the
                 * topology-owned parent later inserts the exact child edge.
                 * A setup-only external is similarly address-authoritative but
                 * not payload-authoritative. It is admitted only when the
                 * ledger names it as a declared arena frontier; transaction
                 * zero performs the live-byte preflight before graph launch.
                 */
                return;
            }
        }

        if (!::llaminar2::isDeviceValid(base->coherence_state_))
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput requires globally valid "
                "external input bytes on " +
                target_device.toString() + tensorTransferState(*base));
        }

        if (!base->device_completion_event_ ||
            !base->completionEventProtectsDeviceValue_())
            return;

        /*
         * A capture body may consume only dependencies established before
         * beginCapture(). Importing an event recorded by uncaptured work here is
         * rejected by CUDA/HIP and, more importantly, hides an incomplete graph
         * boundary. The exact event/stream identity is established by the
         * executor's pre-capture input pass.
         */
        if (isGraphCaptureActive())
        {
            if (base->last_joined_completion_event_ ==
                    base->device_completion_event_ &&
                base->last_joined_consumer_stream_ == consumer_stream)
            {
                return;
            }

            throw std::runtime_error(
                "TransferEngine::requireDeviceInput found an external producer "
                "event that was not joined to the exact consumer stream before "
                "GPU graph capture began for " +
                target_device.toString() + tensorTransferState(*base));
        }

        if (!base->event_device_.has_value() ||
            *base->event_device_ != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput found completion-event "
                "ownership that does not match " +
                target_device.toString() + tensorTransferState(*base));
        }

        IBackend *backend = base->resolveBackend(target_device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput could not resolve backend "
                "for " +
                target_device.toString() + tensorTransferState(*base));
        }

        const int backend_device_id = target_device.gpu_ordinal();
        if (!backend->streamWaitEvent(
                consumer_stream,
                base->device_completion_event_,
                backend_device_id))
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput failed to join producer "
                "event on consumer stream for " +
                target_device.toString() + tensorTransferState(*base));
        }

        /*
         * Record only after the backend accepts the wait. This is a proof about
         * one event generation on one stream, not a general coherence flag.
         * The next device publication clears it before re-recording the event.
         */
        base->last_joined_completion_event_ = base->device_completion_event_;
        base->last_joined_consumer_stream_ = consumer_stream;
    }

    void TransferEngine::prepareHostInput(ITensor *tensor)
    {
        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::prepareHostInput");
        const TransferResult result = instance().download(base);
        if (!result.success)
        {
            throw std::runtime_error(
                "TransferEngine::prepareHostInput failed to materialize tensor: " +
                result.error);
        }
        if (!base->hostValid())
        {
            throw std::runtime_error(
                "TransferEngine::prepareHostInput completed without valid host storage");
        }
    }

    void TransferEngine::allocateDeviceStorage(
        ITensor *tensor,
        DeviceId target_device)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::allocateDeviceStorage requires a GPU target");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::allocateDeviceStorage");
        if (!base->allocateOnDevice(target_device))
        {
            throw std::runtime_error(
                "TransferEngine::allocateDeviceStorage failed to allocate tensor on " +
                target_device.toString());
        }
        if (!base->gpu_data_ptr() ||
            !base->current_device().has_value() ||
            *base->current_device() != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::allocateDeviceStorage completed without storage on " +
                target_device.toString());
        }
    }

    void TransferEngine::prepareDeviceOutput(
        ITensor *tensor,
        DeviceId target_device,
        void *stream)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceOutput requires the exact "
                "non-null producer stream");
        }
        if (isGraphCaptureActive())
        {
            throw std::logic_error(
                "TransferEngine::prepareDeviceOutput is forbidden during GPU "
                "graph capture; capture writers must use requireDeviceOutput");
        }
        allocateDeviceStorage(tensor, target_device);
    }

    void TransferEngine::requireDeviceOutput(
        ITensor *tensor,
        DeviceId target_device,
        void *producer_stream)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceOutput requires a GPU target");
        }
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceOutput requires the exact "
                "non-null producer stream");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::requireDeviceOutput");
        std::lock_guard<std::mutex> lock(base->coherence_mutex_);
        if (!base->gpu_data_ptr_ ||
            !base->gpu_device_.has_value() ||
            *base->gpu_device_ != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceOutput requires pre-existing "
                "storage on the exact device " +
                target_device.toString() + tensorTransferState(*base));
        }
    }

    void TransferEngine::publishDeviceWrite(
        TensorBase *tensor,
        DeviceId device,
        void *producer_stream)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishDeviceWrite requires a GPU device");
        }
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::publishDeviceWrite requires the exact "
                "non-null producer stream");
        }

        if (auto *ledger = currentGraphCaptureDependencyLedger())
        {
            ledger->recordStagePublication(
                tensor, device, producer_stream);
            std::lock_guard<std::mutex> lock(tensor->coherence_mutex_);
            if (!tensor->gpu_data_ptr_ ||
                !tensor->gpu_device_.has_value() ||
                *tensor->gpu_device_ != device)
            {
                throw std::runtime_error(
                    "TransferEngine::publishDeviceWrite recorded a graph output "
                    "without stable storage on " +
                    device.toString() + tensorTransferState(*tensor));
            }

            /*
             * Recording is not execution.  The graph boundary publishes the
             * actual completed generation after launch, with one exact stream
             * event.  Do not mutate coherence or add per-stage event nodes here.
             */
            return;
        }
        tensor->publishDeviceWriteStateWithEvent(device, producer_stream);
    }

    void TransferEngine::publishDeviceWrite(
        ITensor *tensor,
        DeviceId device,
        void *producer_stream)
    {
        publishDeviceWrite(
            requireTensorBase(tensor, "TransferEngine::publishDeviceWrite"),
            device,
            producer_stream);
    }

    void TransferEngine::publishCompletedDeviceWrite(
        TensorBase *tensor,
        DeviceId device)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishCompletedDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishCompletedDeviceWrite requires a GPU device");
        }
        tensor->publishCompletedDeviceWriteState(device);
    }

    void TransferEngine::publishCompletedDeviceWrite(
        ITensor *tensor,
        DeviceId device)
    {
        publishCompletedDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishCompletedDeviceWrite"),
            device);
    }

    void TransferEngine::publishCurrentDeviceWrite(
        TensorBase *tensor,
        void *producer_stream)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishCurrentDeviceWrite");
        const auto device = tensor->current_device();
        if (!device.has_value() || !device->is_gpu())
        {
            throw std::runtime_error(
                "TransferEngine::publishCurrentDeviceWrite requires an "
                "unambiguous current GPU device");
        }
        publishDeviceWrite(tensor, *device, producer_stream);
    }

    void TransferEngine::publishCurrentDeviceWrite(
        ITensor *tensor,
        void *producer_stream)
    {
        publishCurrentDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishCurrentDeviceWrite"),
            producer_stream);
    }

    void TransferEngine::publishGraphOwnedDeviceWrite(
        TensorBase *tensor,
        DeviceId device)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishGraphOwnedDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishGraphOwnedDeviceWrite requires a GPU device");
        }
        tensor->publishGraphOwnedDeviceWriteState(device);
    }

    void TransferEngine::publishGraphOwnedDeviceWrite(
        ITensor *tensor,
        DeviceId device)
    {
        publishGraphOwnedDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishGraphOwnedDeviceWrite"),
            device);
    }

    void TransferEngine::publishGraphOwnedCurrentDeviceWrite(
        TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishGraphOwnedCurrentDeviceWrite");
        const auto device = tensor->current_device();
        if (!device.has_value() || !device->is_gpu())
        {
            throw std::runtime_error(
                "TransferEngine::publishGraphOwnedCurrentDeviceWrite requires "
                "an unambiguous current GPU device");
        }
        publishGraphOwnedDeviceWrite(tensor, *device);
    }

    void TransferEngine::publishGraphOwnedCurrentDeviceWrite(ITensor *tensor)
    {
        publishGraphOwnedCurrentDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishGraphOwnedCurrentDeviceWrite"));
    }

    void TransferEngine::publishHostWrite(TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishHostWrite");
        tensor->publishHostWriteState();
    }

    void TransferEngine::publishHostWrite(ITensor *tensor)
    {
        publishHostWrite(
            requireTensorBase(tensor, "TransferEngine::publishHostWrite"));
    }

    void TransferEngine::publishSynchronized(TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishSynchronized");
        tensor->publishSynchronizedState();
    }

    void TransferEngine::publishSynchronized(ITensor *tensor)
    {
        publishSynchronized(
            requireTensorBase(tensor, "TransferEngine::publishSynchronized"));
    }

    // ============================================================================
    // planTransfer — pure logic, no side effects
    // ============================================================================

    TransferMethod TransferEngine::planTransfer(DeviceId src, DeviceId dst, MemoryResidency residency)
    {
        // Host-resident tensors never move to device
        if (residency == MemoryResidency::HOST_RESIDENT)
            return TransferMethod::NOOP;

        // Mapped memory is always in-place
        if (residency == MemoryResidency::MAPPED)
            return TransferMethod::MAPPED_NOOP;

        // Same device — nothing to do
        if (src == dst)
            return TransferMethod::NOOP;

        // CPU ↔ GPU
        if (src.is_cpu() && dst.is_gpu())
            return TransferMethod::HOST_TO_DEVICE;

        if (src.is_gpu() && dst.is_cpu())
            return TransferMethod::DEVICE_TO_HOST;

        // GPU ↔ GPU
        if (src.is_gpu() && dst.is_gpu())
        {
            // Same vendor (CUDA↔CUDA or ROCm↔ROCm) — direct P2P
            if (src.type == dst.type)
                return TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND;

            // Cross-vendor (CUDA↔ROCm) — host-staged transfer
            return TransferMethod::HOST_STAGED;
        }

        // CPU → CPU is a no-op (same host)
        if (src.is_cpu() && dst.is_cpu())
            return TransferMethod::NOOP;

        LOG_WARN("[TransferEngine] Unhandled device combination: "
                 << src.toString() << " → " << dst.toString());
        return TransferMethod::NOOP;
    }

    std::string TransferEngine::describeTransferPlan(DeviceId src, DeviceId dst, MemoryResidency residency)
    {
        auto method = planTransfer(src, dst, residency);
        std::ostringstream ss;
        ss << src.toString() << " → " << dst.toString()
           << " [" << to_string(residency) << "] → " << to_string(method);
        return ss.str();
    }

    // ============================================================================
    // execute — dispatch on TransferMethod
    // ============================================================================

    TransferResult TransferEngine::execute(const TransferRequest &request)
    {
        auto start = std::chrono::steady_clock::now();

        TransferResult result;
        switch (request.method)
        {
        case TransferMethod::NOOP:
        case TransferMethod::MAPPED_NOOP:
            result = TransferResult::ok(request.method);
            break;

        case TransferMethod::HOST_TO_DEVICE:
            result = executeHostToDevice(request);
            break;

        case TransferMethod::DEVICE_TO_HOST:
            result = executeDeviceToHost(request);
            break;

        case TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND:
            result = executeDeviceToDeviceSameBackend(request);
            break;

        case TransferMethod::HOST_STAGED:
            result = executeHostStaged(request);
            break;
        }

        auto end = std::chrono::steady_clock::now();
        result.elapsed_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        result.method_used = request.method;

        traceTransfer(request, result);
        return result;
    }

    // ============================================================================
    // High-level TensorBase API
    // ============================================================================

    TransferResult TransferEngine::upload(TensorBase *tensor, DeviceId target_device)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * uploadFull() is the single implementation of allocation, primary
         * device promotion, H2D transfer, event joining, and coherence
         * publication.  Keeping a second implementation here previously let a
         * successful copy land in secondary storage without updating
         * gpu_data_ptr_ or gpu_device_.  Public callers now enter the same
         * lifecycle as TensorBase::ensureOnDevice().
         */
        std::lock_guard<std::mutex> lock(tensor->coherence_mutex_);
        return uploadFull(tensor, target_device, nullptr);
    }

    TransferResult TransferEngine::download(TensorBase *tensor)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * downloadFull() owns completion-event validation, D2H ordering, and
         * host publication.  Delegating to it prevents wrapper and concrete
         * tensors from acquiring subtly different host-read semantics.
         */
        std::lock_guard<std::mutex> lock(tensor->coherence_mutex_);
        return downloadFull(tensor, nullptr);
    }

    TransferResult TransferEngine::transferActivation(
        TensorBase *tensor,
        DeviceId target_device,
        size_t bytes_override)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * Select the actual authoritative source. The authoritative copy may
         * live in a secondary device buffer after a previous pipeline handoff,
         * so gpu_device_ alone is not a sufficient source-of-truth.
         */
        DeviceId src = DeviceId::cpu();
        const auto authoritative_device = tensor->getAuthoritativeDevice();
        if (authoritative_device.has_value() &&
            authoritative_device->is_gpu() &&
            tensor->deviceValid())
        {
            src = *authoritative_device;
        }
        else if (!tensor->hostValid())
        {
            return TransferResult::fail(TransferMethod::NOOP,
                                        "tensor has no valid data on any device");
        }

        const MemoryResidency residency = tensor->memoryResidency();
        TransferMethod method = planTransfer(src, target_device, residency);

        if (method == TransferMethod::NOOP || method == TransferMethod::MAPPED_NOOP)
        {
            /*
             * A valid copy can be stored in the secondary map. Promote it so
             * gpu_data_ptr() and current_device() describe the same storage
             * without pretending that a transfer occurred.
             */
            if (target_device.is_gpu() &&
                (!tensor->gpu_device_.has_value() ||
                 *tensor->gpu_device_ != target_device))
            {
                void *target_ptr =
                    tensor->getOrAllocateDeviceBuffer(target_device);
                if (!target_ptr)
                {
                    return TransferResult::fail(
                        method,
                        "authoritative device has no activation buffer on " +
                            target_device.toString());
                }
                if (tensor->gpu_device_.has_value() && tensor->gpu_data_ptr_)
                {
                    tensor->secondary_device_buffers_
                        [TensorBase::packDeviceId(*tensor->gpu_device_)] =
                        tensor->gpu_data_ptr_;
                }
                tensor->gpu_device_ = target_device;
                tensor->gpu_data_ptr_ = target_ptr;
                tensor->secondary_device_buffers_.erase(
                    TensorBase::packDeviceId(target_device));
            }
            return TransferResult::ok(method);
        }

        TransferRequest req;
        req.source = makeMemoryDescriptor(tensor);
        if (bytes_override > req.source.size_bytes)
        {
            return TransferResult::fail(
                method,
                "activation byte override exceeds tensor allocation");
        }
        if (bytes_override != 0)
            req.source.size_bytes = bytes_override;
        req.source.device = src;
        if (src.is_gpu())
        {
            req.source.device_ptr =
                tensor->getOrAllocateDeviceBuffer(src);
            if (!req.source.device_ptr)
            {
                return TransferResult::fail(
                    method,
                    "authoritative source has no activation buffer on " +
                        src.toString());
            }
        }
        req.target_device = target_device;
        req.method = method;

        // For activation transfer, ensure destination buffer exists
        if (target_device.is_gpu())
        {
            void *dst_ptr = tensor->getOrAllocateDeviceBuffer(target_device);
            if (!dst_ptr)
                return TransferResult::fail(method,
                                            "failed to allocate device buffer for activation transfer to " +
                                                target_device.toString());
            req.target_ptr = dst_ptr;
        }

        auto result = execute(req);

        if (result.success)
        {
            if (target_device.is_gpu())
            {
                void *target_ptr = req.target_ptr;
                if (!tensor->gpu_device_.has_value() ||
                    *tensor->gpu_device_ != target_device)
                {
                    if (tensor->gpu_device_.has_value() &&
                        tensor->gpu_data_ptr_)
                    {
                        tensor->secondary_device_buffers_
                            [TensorBase::packDeviceId(*tensor->gpu_device_)] =
                            tensor->gpu_data_ptr_;
                    }
                    tensor->gpu_device_ = target_device;
                    tensor->gpu_data_ptr_ = target_ptr;
                    tensor->secondary_device_buffers_.erase(
                        TensorBase::packDeviceId(target_device));
                }
                publishDeviceWrite(
                    tensor,
                    target_device,
                    requireTransferStream(
                        target_device,
                        "TransferEngine::transferActivation publication"));
            }
            else
            {
                tensor->applyCoherenceOp_(CoherenceOp::DOWNLOAD);
            }
        }

        return result;
    }

    TransferResult TransferEngine::copyActivation(TensorBase *src, TensorBase *dst,
                                                  DeviceId dst_device, size_t bytes)
    {
        if (!src || !dst)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor in copyActivation");
        src = src->transferStorageOwner();
        dst = dst->transferStorageOwner();
        if (!src || !dst ||
            src->transferStorageOwner() != src ||
            dst->transferStorageOwner() != dst)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "activation has no canonical transfer-storage owner");
        }
        if (bytes == 0)
            return TransferResult::ok(TransferMethod::NOOP);

        // ----------------------------------------------------------------------
        // Host or mapped destination: a plain host-side copy is correct and
        // cheapest. Mapped memory shares host/device storage, so writing the
        // host side is immediately visible to the device — no transfer needed.
        // ----------------------------------------------------------------------
        if (dst_device.is_cpu() || dst->is_mapped_ || src->is_mapped_)
        {
            const void *host_src = src->data();   // ensures src is host-valid
            void *host_dst = dst->mutable_data(); // host (or mapped) storage
            if (!host_src || !host_dst)
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                            "null host pointer in copyActivation host path");
            std::memcpy(host_dst, host_src, bytes);
            if (dst_device.is_gpu())
            {
                /*
                 * Mapped host memory is immediately visible to the GPU, but the
                 * coherence publication still needs a concrete completion
                 * token. Record it on the destination backend's owned default
                 * stream so later GPU consumers never inherit eventless device
                 * authority from this host-visible transfer boundary.
                 */
                publishCompletedDeviceWrite(dst, dst_device);
            }
            return TransferResult::ok(dst->is_mapped_ ? TransferMethod::MAPPED_NOOP
                                                      : TransferMethod::DEVICE_TO_HOST);
        }

        // ----------------------------------------------------------------------
        // GPU destination: ensure a device buffer exists. We deliberately do NOT
        // upload host data here (the buffer is about to be overwritten by the
        // copy below).
        // ----------------------------------------------------------------------
        void *dst_ptr = dst->getOrAllocateDeviceBuffer(dst_device);
        if (!dst_ptr)
            return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                        "failed to allocate dst device buffer on " + dst_device.toString());

        // Locate the authoritative source data. NOTE: we use getAuthoritativeDevice()
        // rather than gpu_device_: after a cross-vendor transferActivation(), the
        // tensor's primary gpu_device_/gpu_data_ptr_ still point at the producing
        // GPU (e.g. CUDA), while the authoritative copy lives in a secondary buffer
        // on the consumer GPU (e.g. ROCm).
        const auto auth = src->getAuthoritativeDevice();
        const bool src_on_gpu = auth.has_value() && auth->is_gpu();
        const DeviceId src_device = src_on_gpu ? *auth : DeviceId::cpu();

        TransferResult result;

        // Decide whether a direct device-to-device path exists for this pair.
        // Same physical GPU is always direct (intra-VRAM memcpy). Different GPUs
        // of the SAME vendor go through the collective backend (NCCL/RCCL/peer
        // DMA). Only cross-vendor pairs (CUDA↔ROCm) have no direct path and must
        // bounce through the host.
        const bool same_physical_gpu = src_on_gpu && src_device == dst_device;
        const bool same_vendor_diff_gpu =
            src_on_gpu && !same_physical_gpu && src_device.type == dst_device.type;

        if (same_physical_gpu)
        {
            // Same physical GPU: pure device-to-device copy — no host bounce.
            void *src_ptr = src->getOrAllocateDeviceBuffer(src_device);
            IBackend *backend = resolveBackend(dst_device);
            if (!backend)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "no backend for " + dst_device.toString());
            if (!src_ptr)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "source has no device buffer on " + dst_device.toString());
            void *const destination_stream =
                requireTransferStream(
                    dst_device,
                    "TransferEngine::copyActivation same-device copy");
            if (!backend->deviceToDevice(
                    dst_ptr,
                    src_ptr,
                    bytes,
                    dst_device.gpu_ordinal(),
                    destination_stream))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "deviceToDevice failed on " + dst_device.toString());
            result = TransferResult::ok(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
        }
        else if (same_vendor_diff_gpu)
        {
            // Same-vendor, different GPU (e.g. cuda:0 → cuda:1): use the
            // collective backend's peer copy (NCCL/RCCL or peer DMA). No host
            // bounce — the data moves directly across the PCIe/NVLink fabric.
            void *src_ptr = src->getOrAllocateDeviceBuffer(src_device);
            auto *router = GlobalBackendRouter::get();
            ICollectiveBackend *backend =
                router ? router->getBackendForCopy(src_device, dst_device) : nullptr;
            if (!src_ptr)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "source has no device buffer on " + src_device.toString());
            if (!backend || !backend->supportsCopy(src_device, dst_device))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "no collective backend supports peer copy " +
                                                src_device.toString() + " -> " + dst_device.toString());
            if (!backend->copy(dst_ptr, dst_device, src_ptr, src_device, bytes))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "peer copy failed " + src_device.toString() +
                                                " -> " + dst_device.toString());
            result = TransferResult::ok(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
        }
        else
        {
            // Cross-vendor GPU pair (CUDA↔ROCm) or host-resident source: there
            // is no direct device path across vendors, so stage through the
            // source tensor's own host buffer and upload into the dst device
            // buffer. This is what makes heterogeneous CUDA↔ROCm PP work.
            const void *host_src = nullptr;
            if (src_on_gpu)
            {
                void *src_host = src->raw_host_data_ptr();
                void *src_dev = src->getOrAllocateDeviceBuffer(src_device);
                IBackend *src_backend = resolveBackend(src_device);
                if (!src_host || !src_dev || !src_backend)
                    return TransferResult::fail(TransferMethod::HOST_STAGED,
                                                "missing src host/device buffer or backend for staged copy");
                void *const source_stream =
                    requireTransferStream(
                        src_device,
                        "TransferEngine::copyActivation staged D2H");
                if (!src_backend->deviceToHost(
                        src_host,
                        src_dev,
                        bytes,
                        src_device.gpu_ordinal(),
                        source_stream))
                    return TransferResult::fail(TransferMethod::HOST_STAGED,
                                                "D2H step failed in copyActivation");
                host_src = src_host;
            }
            else
            {
                host_src = src->data(); // host-resident source
            }

            IBackend *dst_backend = resolveBackend(dst_device);
            if (!dst_backend || !host_src)
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "missing dst backend or host src in copyActivation");
            void *const destination_stream =
                requireTransferStream(
                    dst_device,
                    "TransferEngine::copyActivation staged H2D");
            if (!dst_backend->hostToDevice(
                    dst_ptr,
                    host_src,
                    bytes,
                    dst_device.gpu_ordinal(),
                    destination_stream))
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "H2D step failed in copyActivation");
            result = TransferResult::ok(src_on_gpu ? TransferMethod::HOST_STAGED
                                                   : TransferMethod::HOST_TO_DEVICE);
        }

        // ----------------------------------------------------------------------
        // Promote dst_ptr to the primary device buffer if it currently lives in
        // the secondary map (so later gpu_data_ptr() returns the buffer we just
        // wrote), then mark the destination authoritative on dst_device.
        // ----------------------------------------------------------------------
        if (!dst->gpu_device_.has_value() || *dst->gpu_device_ != dst_device)
        {
            // Preserve any existing primary buffer as a secondary so it is not leaked.
            if (dst->gpu_device_.has_value() && dst->gpu_data_ptr_)
                dst->secondary_device_buffers_[TensorBase::packDeviceId(*dst->gpu_device_)] =
                    dst->gpu_data_ptr_;
            dst->gpu_device_ = dst_device;
            dst->gpu_data_ptr_ = dst_ptr;
            dst->secondary_device_buffers_.erase(TensorBase::packDeviceId(dst_device));
        }
        /*
         * copyActivation() currently completes its selected transport before
         * returning. Publish a fresh backend event after that boundary rather
         * than erasing completion metadata with a plain coherence transition.
         * The event gives every later device consumer one uniform dependency
         * contract regardless of whether the bytes arrived via intra-device,
         * peer, or deliberately heterogeneous host-staged transport.
         */
        publishDeviceWrite(
            dst,
            dst_device,
            requireTransferStream(
                dst_device,
                "TransferEngine::copyActivation publication"));

        return result;
    }

    // ============================================================================
    // Private: execute*() methods
    // ============================================================================

    TransferResult TransferEngine::executeHostToDevice(const TransferRequest &req)
    {
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "source host_ptr is null");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null");

        IBackend *backend = resolveBackend(req.target_device);
        if (!backend)
            return TransferResult::fail(req.method,
                                        "no backend for device " + req.target_device.toString());

        bool ok = backend->hostToDevice(
            req.target_ptr,
            req.source.host_ptr,
            req.source.size_bytes,
            req.target_device.ordinal,
            requireTransferStream(
                req.target_device,
                "TransferEngine::executeHostToDevice"));
        if (!ok)
            return TransferResult::fail(req.method, "hostToDevice failed");

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeDeviceToHost(const TransferRequest &req)
    {
        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null");
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "destination host_ptr is null (no host buffer)");

        IBackend *backend = resolveBackend(req.source.device);
        if (!backend)
            return TransferResult::fail(req.method,
                                        "no backend for source device " + req.source.device.toString());

        bool ok = backend->deviceToHost(
            req.source.host_ptr,
            req.source.device_ptr,
            req.source.size_bytes,
            req.source.device.ordinal,
            requireTransferStream(
                req.source.device,
                "TransferEngine::executeDeviceToHost"));
        if (!ok)
            return TransferResult::fail(req.method, "deviceToHost failed");

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeDeviceToDeviceSameBackend(const TransferRequest &req)
    {
        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null");

        /*
         * Same-vendor movement is a collective-backend responsibility. NCCL
         * and RCCL select their best available P2P transport; silently bouncing
         * through host memory here would hide a broken device path and violate
         * the device-owned pipeline contract.
         */
        auto *router = GlobalBackendRouter::get();
        ICollectiveBackend *backend =
            router
                ? router->getBackendForCopy(
                      req.source.device,
                      req.target_device)
                : nullptr;
        if (!backend ||
            !backend->supportsCopy(
                req.source.device,
                req.target_device))
        {
            return TransferResult::fail(
                req.method,
                "no direct collective backend supports " +
                    req.source.device.toString() + " -> " +
                    req.target_device.toString());
        }
        if (!backend->copy(
                req.target_ptr,
                req.target_device,
                req.source.device_ptr,
                req.source.device,
                req.source.size_bytes))
        {
            return TransferResult::fail(
                req.method,
                "collective backend copy failed " +
                    req.source.device.toString() + " -> " +
                    req.target_device.toString());
        }

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeHostStaged(const TransferRequest &req)
    {
        // HOST_STAGED: D2H from source GPU → memcpy → H2D to target GPU
        // Used for cross-vendor transfers.

        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null for host staged");
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "host buffer needed for host staged bounce");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null for host staged");

        IBackend *src_backend = resolveBackend(req.source.device);
        if (!src_backend)
            return TransferResult::fail(req.method,
                                        "no backend for source " + req.source.device.toString());

        // Step 1: D2H
        bool d2h = src_backend->deviceToHost(
            req.source.host_ptr, req.source.device_ptr,
            req.source.size_bytes, req.source.device.ordinal,
            requireTransferStream(
                req.source.device,
                "TransferEngine::executeHostStaged source"));
        if (!d2h)
            return TransferResult::fail(req.method, "D2H step of host staged failed");

        // Step 2: H2D to target
        IBackend *dst_backend = resolveBackend(req.target_device);
        if (!dst_backend)
            return TransferResult::fail(req.method,
                                        "no backend for target " + req.target_device.toString());

        bool h2d = dst_backend->hostToDevice(
            req.target_ptr, req.source.host_ptr,
            req.source.size_bytes, req.target_device.ordinal,
            requireTransferStream(
                req.target_device,
                "TransferEngine::executeHostStaged destination"));
        if (!h2d)
            return TransferResult::fail(req.method, "H2D step of host staged failed");

        return TransferResult::ok(req.method);
    }

    // ============================================================================
    // Backend resolution
    // ============================================================================

    bool TransferEngine::waitForEventWithProxy(IBackend *backend, void *event, int device_id,
                                               const DeviceId &gpu_device)
    {
        return backend->waitForEvent(event, device_id);
    }

    void TransferEngine::waitForPendingHostSourceUseLocked(TensorBase *tensor)
    {
        if (!tensor ||
            !tensor->completionEventProtectsHostSource_())
        {
            return;
        }
        if (!tensor->device_completion_event_ ||
            !tensor->event_device_.has_value() ||
            !tensor->event_device_->is_gpu())
        {
            throw std::runtime_error(
                "Queued H2D host-source use has no exact completion event");
        }

        IBackend *const backend =
            tensor->resolveBackend(*tensor->event_device_);
        if (!backend ||
            backend->backendDeviceType() != tensor->event_device_->type)
        {
            throw std::runtime_error(
                "Queued H2D host-source event has no matching backend on " +
                tensor->event_device_->toString());
        }
        if (!waitForEventWithProxy(
                backend,
                tensor->device_completion_event_,
                tensor->event_device_->gpu_ordinal(),
                *tensor->event_device_))
        {
            throw std::runtime_error(
                "Queued H2D host-source completion event failed on " +
                tensor->event_device_->toString());
        }

        /*
         * The host wait proves both DMA completion and device visibility. The
         * event no longer carries an outstanding lifetime, so retire it rather
         * than letting a later generation accidentally reuse its identity.
         */
        if (tensor->completion_event_protection_ ==
            TensorBase::CompletionEventProtection::DeviceValueAndHostSource)
        {
            tensor->completion_event_protection_ =
                TensorBase::CompletionEventProtection::DeviceValue;
            return;
        }
        tensor->completion_event_protection_ =
            TensorBase::CompletionEventProtection::None;
        tensor->retireCompletionEvent_();
    }

    // ============================================================================
    // uploadFull — full ensureOnDevice lifecycle (called with coherence_mutex_ held)
    // ============================================================================

    TransferResult TransferEngine::uploadFull(TensorBase *tensor, DeviceId target_device, void *stream)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        // ===== HOST-RESIDENT FAST PATH =====
        // Tensors marked HOST_RESIDENT are consumed on host only (e.g., embedding
        // tables that get repacked into a device workspace by the kernel).
        // Skip device allocation and upload entirely.
        if (tensor->memoryResidency() == MemoryResidency::HOST_RESIDENT)
            return TransferResult::ok(TransferMethod::NOOP);

        const bool trace = debugEnv().rocm.trace_coherence;
        auto overall_start = std::chrono::high_resolution_clock::now();

        // ===== ZERO-COPY MAPPED MEMORY FAST PATH =====
        if (tensor->is_mapped_ && tensor->mapped_device_ptr_ != nullptr)
        {
            if (!tensor->gpu_device_.has_value() || *tensor->gpu_device_ != target_device)
            {
                tensor->gpu_device_ = target_device;
            }
            if (tensor->gpu_data_ptr_ != tensor->mapped_device_ptr_)
            {
                tensor->gpu_data_ptr_ = tensor->mapped_device_ptr_;
            }
            tensor->setCoherenceState_(TensorCoherenceState::MAPPED);

            if (trace)
            {
                LOG_TRACE("[TransferEngine::uploadFull] ZERO-COPY: Tensor is mapped, no memcpy needed");
            }
            return TransferResult::ok(TransferMethod::MAPPED_NOOP);
        }

        // ===== ALREADY ON TARGET DEVICE (with event wait) =====
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value() &&
            *tensor->gpu_device_ == target_device && ::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            if (tensor->device_completion_event_ &&
                tensor->completionEventProtectsDeviceValue_())
            {
                IBackend *backend = tensor->resolveBackend(target_device);
                if (backend)
                {
                    const int backend_device_id = target_device.gpu_ordinal();
                    if (stream)
                    {
                        // Non-blocking: make the consuming stream wait for the event.
                        // Same-stream waits are no-ops in hardware (ordering is implicit).
                        // Cross-stream waits correctly serialize without blocking the CPU.
                        if (!backend->streamWaitEvent(stream, tensor->device_completion_event_, backend_device_id))
                        {
                            return TransferResult::fail(
                                TransferMethod::NOOP,
                                "Stream event wait failed for tensor '" +
                                    (tensor->debug_name_.empty()
                                         ? std::string("(unnamed)")
                                         : tensor->debug_name_) +
                                    "' on " + target_device.toString() +
                                    "; refusing a host-blocking fallback");
                        }
                    }
                }
            }
            return TransferResult::ok(TransferMethod::NOOP);
        }

        // Get backend for target device
        IBackend *target_backend = tensor->resolveBackend(target_device);
        if (!target_backend)
        {
            return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                        "No backend available for device " + target_device.toString());
        }

        int backend_device_id = target_device.gpu_ordinal();

        // ===== DEVICE MIGRATION (secondary buffers) =====
        LOG_TRACE("[TransferEngine::uploadFull] tensor=" << static_cast<void *>(tensor)
                                                         << " target_device=" << target_device.toString()
                                                         << " gpu_data_ptr_=" << tensor->gpu_data_ptr_
                                                         << " gpu_device_=" << (tensor->gpu_device_.has_value() ? tensor->gpu_device_->toString() : "none")
                                                         << " device_completion_event_=" << tensor->device_completion_event_);
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value() && *tensor->gpu_device_ != target_device)
        {
            /*
             * A device migration may retire the old event and eventually reuse
             * the host allocation as a source for the new device. Quiesce only
             * an outstanding H2D source use before changing event ownership.
             */
            waitForPendingHostSourceUseLocked(tensor);
            DeviceId old_device = *tensor->gpu_device_;
            LOG_TRACE("[TransferEngine::uploadFull] Device migration: " << tensor->gpu_device_->toString()
                                                                        << " -> " << target_device.toString());

            const int target_key = TensorBase::packDeviceId(target_device);
            auto sec_it = tensor->secondary_device_buffers_.find(target_key);
            if (sec_it != tensor->secondary_device_buffers_.end() && sec_it->second != nullptr)
            {
                // Preserve current primary in secondary map
                int old_key = TensorBase::packDeviceId(*tensor->gpu_device_);
                if (tensor->secondary_device_buffers_.find(old_key) == tensor->secondary_device_buffers_.end())
                {
                    tensor->secondary_device_buffers_[old_key] = tensor->gpu_data_ptr_;
                }

                // Promote target secondary buffer to primary
                void *promoted_ptr = sec_it->second;
                tensor->secondary_device_buffers_.erase(sec_it);

                tensor->gpu_data_ptr_ = promoted_ptr;
                tensor->gpu_device_ = target_device;
                tensor->setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);

                if (tensor->device_completion_event_)
                {
                    tensor->retireCompletionEvent_();
                }

                LOG_TRACE("[TransferEngine::uploadFull] Promoted secondary buffer to primary for "
                          << target_device.toString() << " ptr=" << promoted_ptr);
            }
            else
            {
                // No existing target secondary buffer. Park current primary.
                int old_key = TensorBase::packDeviceId(*tensor->gpu_device_);
                if (tensor->secondary_device_buffers_.find(old_key) == tensor->secondary_device_buffers_.end())
                {
                    tensor->secondary_device_buffers_[old_key] = tensor->gpu_data_ptr_;
                }

                if (tensor->device_completion_event_)
                {
                    LOG_TRACE("[TransferEngine::uploadFull] Retiring old completion event on device "
                              << tensor->gpu_device_->toString() << " before migrating to " << target_device.toString());
                    tensor->retireCompletionEvent_();
                }

                tensor->gpu_data_ptr_ = nullptr;
                tensor->gpu_device_.reset();
                tensor->applyCoherenceOp_(CoherenceOp::RELEASE_DEVICE);

                LOG_TRACE("[TransferEngine::uploadFull] Parked previous primary buffer for "
                          << old_device.toString() << " and allocating fresh buffer for "
                          << target_device.toString());
            }
        }

        // ===== ALLOCATE ON TARGET DEVICE =====
        size_t bytes = tensor->byte_size();
        if (!tensor->gpu_data_ptr_)
        {
            auto alloc_start = std::chrono::high_resolution_clock::now();
            tensor->gpu_data_ptr_ = target_backend->allocate(bytes, backend_device_id);
            auto alloc_end = std::chrono::high_resolution_clock::now();
            auto alloc_us = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();

            if (trace)
            {
                LOG_TRACE("[TransferEngine::uploadFull] backend->allocate(" << bytes << " bytes) took " << alloc_us << " us");
            }

            LOG_TRACE("[GPU_ALLOC] tensor=" << static_cast<void *>(tensor)
                                            << " name=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                                            << " gpu_ptr=" << tensor->gpu_data_ptr_ << " bytes=" << bytes
                                            << " device=" << target_device.toString()
                                            << " ordinal=" << backend_device_id);

            if (!tensor->gpu_data_ptr_)
            {
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "Failed to allocate " + std::to_string(bytes) + " bytes on " + target_device.toString());
            }
            tensor->gpu_device_ = target_device;
            tensor->setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);

            // Pin host memory for fast DMA transfers
            auto pin_start = std::chrono::high_resolution_clock::now();
            tensor->ensureHostPinned();
            auto pin_end = std::chrono::high_resolution_clock::now();
            auto pin_us = std::chrono::duration_cast<std::chrono::microseconds>(pin_end - pin_start).count();

            if (trace && pin_us > 100)
            {
                LOG_TRACE("[TransferEngine::uploadFull] ensureHostPinned() took " << pin_us << " us");
            }
        }

        // ===== H2D UPLOAD =====
        if (!::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            if (!::llaminar2::isHostValid(tensor->coherence_state_))
            {
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "COHERENCE ERROR: Both host and device are invalid");
            }

            const void *src = tensor->raw_host_data_ptr();
            if (!src)
            {
                target_backend->free(tensor->gpu_data_ptr_, backend_device_id);
                tensor->gpu_data_ptr_ = nullptr;
                tensor->gpu_device_.reset();
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE, "Host data pointer is null");
            }

            // Transfer tracing
            const auto &trace_cfg = debugEnv().transfer_tracing;
            if (trace_cfg.enabled && !trace_cfg.only_d2h && bytes >= trace_cfg.min_bytes)
            {
                std::ostringstream msg;
                msg << "[TRANSFER TRACE] H2D transfer: " << bytes << " bytes"
                    << ", tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                    << ", shape=[" << tensor->rows() << "x" << tensor->cols() << "]"
                    << ", device=" << target_device.toString();

                if (trace_cfg.include_stacktrace)
                {
                    msg << "\n"
                        << captureStackTrace(2, 16);
                }

                if (trace_cfg.throw_on_transfer)
                {
                    throw TransferViolationException(msg.str(), captureStackTrace(2, 32));
                }
                else
                {
                    LOG_WARN(msg.str());
                }
            }

            void *const upload_stream =
                stream
                    ? stream
                    : requireTransferStream(
                          target_device,
                          "TransferEngine::uploadFull");
            /*
             * Order an overwrite after any previous producer of this device
             * allocation. This is a device-side edge; the submitting CPU never
             * waits for the prior generation.
             */
            if (tensor->device_completion_event_ &&
                tensor->completionEventProtectsDeviceValue_())
            {
                if (!tensor->event_device_.has_value() ||
                    *tensor->event_device_ != target_device)
                {
                    return TransferResult::fail(
                        TransferMethod::HOST_TO_DEVICE,
                        "H2D upload found completion-event ownership on the wrong device");
                }
                if (!target_backend->streamWaitEvent(
                        upload_stream,
                        tensor->device_completion_event_,
                        backend_device_id))
                {
                    return TransferResult::fail(
                        TransferMethod::HOST_TO_DEVICE,
                        "H2D upload could not join the prior device producer event");
                }
            }

            bool created_event = false;
            if (!tensor->device_completion_event_)
            {
                tensor->device_completion_event_ =
                    target_backend->createEvent(backend_device_id);
                if (!tensor->device_completion_event_)
                {
                    return TransferResult::fail(
                        TransferMethod::HOST_TO_DEVICE,
                        "H2D upload could not create its completion event");
                }
                tensor->event_device_ = target_device;
                created_event = true;
            }

            auto h2d_start = std::chrono::high_resolution_clock::now();
            bool h2d_ok = target_backend->hostToDeviceOnStream(
                tensor->gpu_data_ptr_,
                src,
                bytes,
                backend_device_id,
                upload_stream);
            auto h2d_end = std::chrono::high_resolution_clock::now();
            auto h2d_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(h2d_end - h2d_start).count();
            auto h2d_us = h2d_ns / 1000;

            if (trace_cfg.enabled)
            {
                trace_cfg.recordH2D(bytes);
            }

            if (trace)
            {
                LOG_TRACE("[TransferEngine::uploadFull] queued hostToDevice(" << bytes
                                                                               << " bytes) in "
                                                                               << h2d_us << " us");
            }

            if (!h2d_ok)
            {
                if (created_event)
                {
                    target_backend->destroyEvent(
                        tensor->device_completion_event_,
                        backend_device_id);
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                }
                return TransferResult::fail(
                    TransferMethod::HOST_TO_DEVICE,
                    "asynchronous hostToDevice enqueue failed");
            }

            if (!target_backend->recordEvent(
                    tensor->device_completion_event_,
                    backend_device_id,
                    upload_stream))
            {
                /*
                 * The runtime accepted DMA but failed to publish the only safe
                 * source-lifetime boundary. Continuing could free or overwrite
                 * pinned host bytes still in use, so this is unrecoverable.
                 */
                LOG_ERROR("[TransferEngine::uploadFull] Accepted H2D copy could not publish its exact completion event on "
                          << target_device.toString());
                std::terminate();
            }

            tensor->last_joined_completion_event_ = nullptr;
            tensor->last_joined_consumer_stream_ = nullptr;
            tensor->completion_event_protection_ =
                TensorBase::CompletionEventProtection::
                    DeviceValueAndHostSource;
            tensor->applyCoherenceOp_(CoherenceOp::UPLOAD);
            tensor->authoritative_device_.reset();
            TransferProfiler::recordH2D(bytes);

            // GPU_ONLY policy: free host data now that device has it
            if (tensor->memoryResidency() == MemoryResidency::GPU_ONLY &&
                !tensor->is_raw_data_released())
            {
                waitForPendingHostSourceUseLocked(tensor);
                tensor->release_host_weight_data();
                tensor->setCoherenceState_(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
                tensor->authoritative_device_ = target_device;
            }

            LOG_TRACE("[TransferEngine::uploadFull] Uploaded " << bytes
                                                               << " bytes to device " << target_device.toString()
                                                               << " (backend device ID: " << backend_device_id << ")");
        }

        auto overall_end = std::chrono::high_resolution_clock::now();
        auto overall_us = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start).count();
        if (trace && overall_us > 1000)
        {
            LOG_TRACE("[TransferEngine::uploadFull] TOTAL took " << overall_us << " us for " << bytes << " bytes");
        }

        return TransferResult::ok(TransferMethod::HOST_TO_DEVICE);
    }

    // ============================================================================
    // downloadFull — full ensureOnHost lifecycle (called with coherence_mutex_ held)
    // ============================================================================

    TransferResult TransferEngine::downloadFull(TensorBase *tensor, void *stream)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        // ===== ZERO-COPY MAPPED MEMORY PATH =====
        if (tensor->is_mapped_)
        {
            if (tensor->mapped_needs_sync_ && tensor->gpu_device_.has_value())
            {
                IBackend *backend = tensor->resolveBackend(*tensor->gpu_device_);
                if (backend)
                {
                    int backend_device_id = tensor->gpu_device_->gpu_ordinal();

                    auto t0 = std::chrono::high_resolution_clock::now();

                    if (tensor->device_completion_event_)
                    {
                        LOG_TRACE("[TransferEngine::downloadFull] ZERO-COPY: Waiting on completion event");
                        if (!waitForEventWithProxy(backend, tensor->device_completion_event_, backend_device_id, *tensor->gpu_device_))
                        {
                            return TransferResult::fail(
                                TransferMethod::MAPPED_NOOP,
                                "Mapped tensor completion event wait failed");
                        }
                    }
                    else
                    {
                        return TransferResult::fail(
                            TransferMethod::MAPPED_NOOP,
                            "Mapped GPU tensor has no completion event");
                    }

                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (elapsed_ms > 1.0)
                    {
                        LOG_WARN("[TransferEngine::downloadFull] MAPPED SYNC took " << elapsed_ms << " ms"
                                                                                    << " (event=" << (tensor->device_completion_event_ ? "yes" : "NO") << ")");
                    }
                }
                tensor->mapped_needs_sync_ = false;
            }

            tensor->setCoherenceState_(TensorCoherenceState::MAPPED);
            tensor->authoritative_device_ = std::nullopt;
            return TransferResult::ok(TransferMethod::MAPPED_NOOP);
        }

        // ===== HOST ALREADY VALID =====
        if (::llaminar2::isHostValid(tensor->coherence_state_))
        {
            LOG_TRACE("[TransferEngine::downloadFull] Host already valid, skipping sync");
            return TransferResult::ok(TransferMethod::NOOP);
        }

        // Device must be valid if host is invalid
        if (!::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                        "COHERENCE ERROR: Both host and device are invalid");
        }

        // ===== STANDARD GPU D2H =====
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value())
        {
            IBackend *backend = tensor->resolveBackend(*tensor->gpu_device_);
            if (!backend)
            {
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                            "No backend available for device " + tensor->gpu_device_->toString());
            }

            int backend_device_id = tensor->gpu_device_->gpu_ordinal();

            size_t bytes = tensor->byte_size();
            void *dst = tensor->raw_host_data_ptr();
            if (!dst)
            {
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST, "Host data pointer is null");
            }

            if (!stream &&
                (!tensor->device_completion_event_ ||
                 !tensor->completionEventProtectsDeviceValue_()))
            {
                return TransferResult::fail(
                    TransferMethod::DEVICE_TO_HOST,
                    "GPU tensor has no completion event or explicit producer stream");
            }

            // Transfer tracing for D2H debugging
            const auto &trace_cfg = debugEnv().transfer_tracing;
            if (trace_cfg.enabled && bytes >= trace_cfg.min_bytes)
            {
                std::ostringstream msg;
                msg << "[TRANSFER TRACE] D2H transfer: " << bytes << " bytes"
                    << ", tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                    << ", shape=[" << tensor->rows() << "x" << tensor->cols() << "]"
                    << ", device=" << tensor->gpu_device_->toString();

                if (trace_cfg.include_stacktrace)
                {
                    msg << "\n"
                        << captureStackTrace(2, 16);
                }

                if (trace_cfg.throw_on_transfer)
                {
                    throw TransferViolationException(msg.str(), captureStackTrace(2, 32));
                }
                else
                {
                    LOG_WARN(msg.str());
                }
            }

            LOG_TRACE("[TransferEngine::downloadFull] ATTEMPTING D2H: "
                      << "tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                      << " gpu_data_ptr=" << static_cast<const void *>(tensor->gpu_data_ptr_)
                      << " dst=" << static_cast<const void *>(dst)
                      << " bytes=" << bytes
                      << " device=" << tensor->gpu_device_->toString()
                      << " backend_device_id=" << backend_device_id);

            void *const download_stream =
                stream
                    ? stream
                    : requireTransferStream(
                          *tensor->gpu_device_,
                          "TransferEngine::downloadFull");

            /*
             * A caller-provided stream is the exact producer stream and already
             * carries the source dependency. Otherwise import the published
             * producer event onto the dedicated transfer stream. Neither path
             * blocks the host before the D2H copy is submitted.
             */
            if (!stream)
            {
                if (!tensor->completionEventProtectsDeviceValue_() ||
                    !tensor->event_device_.has_value() ||
                    *tensor->event_device_ != *tensor->gpu_device_ ||
                    !backend->streamWaitEvent(
                        download_stream,
                        tensor->device_completion_event_,
                        backend_device_id))
                {
                    return TransferResult::fail(
                        TransferMethod::DEVICE_TO_HOST,
                        "D2H transfer could not join the exact device producer event");
                }
            }

            bool created_event = false;
            if (!tensor->device_completion_event_)
            {
                tensor->device_completion_event_ =
                    backend->createEvent(backend_device_id);
                if (!tensor->device_completion_event_)
                {
                    return TransferResult::fail(
                        TransferMethod::DEVICE_TO_HOST,
                        "D2H transfer could not create its host-publication event");
                }
                tensor->event_device_ = *tensor->gpu_device_;
                created_event = true;
            }

            auto d2h_start = std::chrono::high_resolution_clock::now();
            bool d2h_ok = backend->deviceToHostOnStream(
                dst,
                tensor->gpu_data_ptr_,
                bytes,
                backend_device_id,
                download_stream);
            if (!d2h_ok)
            {
                if (created_event)
                {
                    backend->destroyEvent(
                        tensor->device_completion_event_,
                        backend_device_id);
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                }
                return TransferResult::fail(
                    TransferMethod::DEVICE_TO_HOST,
                    "asynchronous deviceToHost enqueue failed");
            }
            if (!backend->recordEvent(
                    tensor->device_completion_event_,
                    backend_device_id,
                    download_stream))
            {
                LOG_ERROR("[TransferEngine::downloadFull] Accepted D2H copy could not publish its exact host-completion event on "
                          << tensor->gpu_device_->toString());
                std::terminate();
            }
            tensor->last_joined_completion_event_ = nullptr;
            tensor->last_joined_consumer_stream_ = nullptr;
            if (!waitForEventWithProxy(
                    backend,
                    tensor->device_completion_event_,
                    backend_device_id,
                    *tensor->gpu_device_))
            {
                return TransferResult::fail(
                    TransferMethod::DEVICE_TO_HOST,
                    "D2H host-publication event wait failed; completion event "
                    "is invalid or could not be observed");
            }
            tensor->completion_event_protection_ =
                TensorBase::CompletionEventProtection::DeviceValue;
            auto d2h_end = std::chrono::high_resolution_clock::now();
            auto d2h_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(d2h_end - d2h_start).count();

            // Optional transfer trace diagnostic. This only samples the first
            // few floats, so keep it out of normal logs and validation output.
            if (trace_cfg.enabled)
            {
                const float *fp = static_cast<const float *>(dst);
                size_t check_count = std::min(bytes / sizeof(float), static_cast<size_t>(8));
                bool all_zero = true;
                for (size_t i = 0; i < check_count; ++i)
                {
                    if (fp[i] != 0.0f)
                    {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero && check_count > 0 && bytes >= 1024)
                {
                    LOG_TRACE("[TransferEngine::downloadFull] D2H leading sample is zero: tensor="
                              << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                              << " bytes=" << bytes
                              << " device=" << tensor->gpu_device_->toString()
                              << " gpu_ptr=" << static_cast<const void *>(tensor->gpu_data_ptr_)
                              << " dst=" << static_cast<const void *>(dst)
                              << " d2h_ns=" << d2h_ns
                              << " had_event=" << (tensor->device_completion_event_ ? "yes" : "no"));
                }
            }

            TransferProfiler::recordD2H(bytes, static_cast<uint64_t>(d2h_ns));

            if (trace_cfg.enabled)
            {
                trace_cfg.recordD2H(bytes);
            }

            tensor->applyCoherenceOp_(CoherenceOp::DOWNLOAD);
            tensor->authoritative_device_ = std::nullopt;
            tensor->retireCompletionEvent_();

            LOG_TRACE("[TransferEngine::downloadFull] Downloaded " << bytes
                                                                   << " bytes from device " << tensor->gpu_device_->toString()
                                                                   << " (backend device ID: " << backend_device_id << ")");
        }

        return TransferResult::ok(TransferMethod::DEVICE_TO_HOST);
    }

    IBackend *TransferEngine::resolveBackend(DeviceId device) const
    {
        if (resolve_)
            return resolve_(device);
        return getBackendFor(device);
    }

    // ============================================================================
    // Transfer tracing
    // ============================================================================

    void TransferEngine::traceTransfer(const TransferRequest &req, const TransferResult &result) const
    {
        const auto &tracing = debugEnv().transfer_tracing;
        if (!tracing.enabled)
            return;

        auto method_str = to_string(req.method);
        auto src_str = req.source.device.toString();
        auto dst_str = req.target_device.toString();

        if (result.success)
        {
            LOG_TRACE("[TransferEngine] " << method_str
                                          << " " << src_str << " → " << dst_str
                                          << " (" << req.source.size_bytes << " bytes"
                                          << ", " << result.elapsed_ns / 1000 << " μs)");
        }
        else
        {
            LOG_WARN("[TransferEngine] FAILED " << method_str
                                                << " " << src_str << " → " << dst_str
                                                << ": " << result.error);
        }
    }

    // ============================================================================
    // MemoryDescriptor factory
    // ============================================================================

    MemoryDescriptor makeMemoryDescriptor(const TensorBase *tensor)
    {
        MemoryDescriptor desc;

        if (!tensor)
            return desc;
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
            return desc;

        // Host pointer
        desc.host_ptr = const_cast<void *>(tensor->raw_data());
        desc.size_bytes = tensor->byte_size();

        // GPU pointer and device
        if (tensor->gpu_device_.has_value())
        {
            desc.device = tensor->gpu_device_.value();
            desc.device_ptr = tensor->gpu_data_ptr_;
        }
        else
        {
            desc.device = DeviceId::cpu();
        }

        // Memory residency from canonical source
        desc.residency = tensor->memoryResidency();
        if (desc.residency == MemoryResidency::MAPPED)
        {
            desc.mapped_host_ptr = tensor->mapped_host_ptr_;
            desc.mapped_device_ptr = tensor->mapped_device_ptr_;
        }

        return desc;
    }

} // namespace llaminar2
