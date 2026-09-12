/**
 * @file ModelContextRetirement.cpp
 * @brief Implements exact final-owner retirement for prepared model contexts.
 *
 * The implementation keeps the ownership transition deliberately linear:
 * validate one reusable final owner, capture every device ticket, release all
 * model/workspace owners, then complete every runtime-generation retirement.
 * No driver free-memory heuristic or safety reserve participates in the proof.
 * Ordinary disposal must not perform physical restoration for an absent reuse
 * owner; the terminal overlay policy below makes that distinction explicit.
 */

#include "ModelContextRetirement.h"

#include "execution/moe/MoEOverlayDeviceControllerGraphService.h"
#include "transfer/TransferEngine.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    MoEOverlayDeviceControllerDrainIntent modelContextOverlayDrainIntent(
        const std::shared_ptr<ModelContextReuseAuthority> &authority,
        MoERebalanceRuntimeMode movement)
    {
        switch (movement)
        {
        case MoERebalanceRuntimeMode::Off:
        case MoERebalanceRuntimeMode::Observe:
        case MoERebalanceRuntimeMode::Dynamic:
            break;
        default:
            throw std::logic_error("Model disposal has an invalid movement policy");
        }

        // No consumer can reuse these prepared weights. Finish admitted work,
        // then release it; restoring a soon-to-be-freed owner table wastes IO.
        if (!authority)
            return MoEOverlayDeviceControllerDrainIntent::ReleaseResources;

        const auto state = authority->state();
        if (state != ModelContextReuseAuthority::State::RunnerExclusive &&
            state != ModelContextReuseAuthority::State::Sealing)
        {
            throw std::logic_error(
                "Overlay disposal does not own its retained model context");
        }
        return movement == MoERebalanceRuntimeMode::Dynamic
            ? MoEOverlayDeviceControllerDrainIntent::RestorePreparedContext
            : MoEOverlayDeviceControllerDrainIntent::ReleaseResources;
    }

    PendingExclusiveModelRetirement
    PendingExclusiveModelRetirement::begin(
        const std::vector<ModelDeviceMemoryRetention> &retention)
    {
        if (retention.empty())
        {
            return PendingExclusiveModelRetirement{
                Kind::HostOnly,
                {}};
        }

        std::set<DeviceId> devices;
        std::vector<ExclusiveModelRetirementTicket> tickets;
        tickets.reserve(retention.size());
        for (const ModelDeviceMemoryRetention &row : retention)
        {
            if (!row.valid() || !devices.insert(row.device).second)
            {
                throw std::logic_error(
                    "Model-context retirement BOM contains an invalid or duplicate GPU row");
            }
            tickets.push_back(
                TransferEngine::instance().beginExclusiveModelRetirement(
                    row));
        }
        return PendingExclusiveModelRetirement{
            Kind::DeviceBatch,
            std::move(tickets)};
    }

    ModelContextRetirementReceipt
    PendingExclusiveModelRetirement::complete()
    {
        if (completed_)
        {
            throw std::logic_error(
                "Pending model-context retirement cannot complete twice");
        }
        completed_ = true;

        ModelContextRetirementReceipt result;
        if (kind_ == Kind::HostOnly)
        {
            if (!tickets_.empty())
            {
                throw std::logic_error(
                    "Host-only model-context retirement unexpectedly owns GPU tickets");
            }
            return result;
        }
        if (tickets_.empty())
        {
            throw std::logic_error(
                "GPU model-context retirement has no device tickets");
        }
        result.device_receipts =
            TransferEngine::instance().completeExclusiveModelRetirements(
                std::move(tickets_));
        return result;
    }

    ModelContextRetirementReceipt
    retireExclusiveModelContextReuseContract(
        ModelContextReuseContract &contract)
    {
        if (!contract.context || !contract.reuse_authority ||
            !contract.reusable_execution_workspaces ||
            !contract.physical_memory_authority)
        {
            throw std::invalid_argument(
                "Model-context retirement requires complete model, workspace, physical-memory, and lifecycle authorities");
        }
        if (contract.reuse_authority->state() !=
            ModelContextReuseAuthority::State::Reusable)
        {
            throw std::logic_error(
                "Model-context retirement requires the Reusable lifecycle boundary");
        }
        if (contract.context.use_count() != 1u)
        {
            throw std::logic_error(
                "Model-context retirement requires the exclusive final ModelContext owner; use_count=" +
                std::to_string(contract.context.use_count()));
        }
        if (contract.reusable_execution_workspaces.use_count() != 1u)
        {
            throw std::logic_error(
                "Model-context retirement requires the exclusive final reusable-workspace owner; use_count=" +
                std::to_string(
                    contract.reusable_execution_workspaces.use_count()));
        }
        if (contract.reuse_authority.use_count() != 1u)
        {
            throw std::logic_error(
                "Model-context retirement requires the exclusive final lifecycle authority; use_count=" +
                std::to_string(contract.reuse_authority.use_count()));
        }

        std::string retention_error;
        const auto sealed_retention =
            contract.reuse_authority->sealedDeviceMemoryRetention(
                &retention_error);
        if (!sealed_retention)
        {
            throw std::logic_error(
                "Model-context retirement has no sealed allocation BOM: " +
                (retention_error.empty()
                     ? std::string("unknown lifecycle failure")
                     : retention_error));
        }

        PendingExclusiveModelRetirement pending =
            PendingExclusiveModelRetirement::begin(*sealed_retention);

        /* Every ticket now binds the live allocator ledger.  Drop the complete
         * prepared-model authority as one ownership edge before any runtime is
         * reset; completing a ticket while another contract field can still
         * reach device storage would make the backend proof meaningless. */
        contract.context.reset();
        contract.reusable_execution_workspaces.reset();
        contract.physical_memory_authority.reset();
        contract.reuse_authority.reset();
        contract.prepared_routed_weight_plan.reset();
        contract.expert_overlay_memory_admission.reset();
        contract.routed_weight_authority_identity.clear();

#if defined(__GLIBC__)
        /* CPU prepared weights are ordinary allocator owners.  Returning free
         * arenas here is outside inference and makes the same final-owner edge
         * useful to a host-tier JIT replacement without imposing a hot-path
         * synchronization. */
        ::malloc_trim(0);
#endif

        return pending.complete();
    }

} // namespace llaminar2
