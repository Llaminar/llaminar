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
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "transfer/TransferEngine.h"
#include "backends/NativeContextPreparation.h"
#include "loaders/ModelContext.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    ModelContextPhysicalSealBoundary modelContextPhysicalSealBoundary(
        const std::shared_ptr<ModelContextReuseAuthority> &authority,
        const std::vector<MoEOverlayMigrationStorageKind> &storage)
    {
        bool mutable_storage = false;
        for (const auto kind : storage)
        {
            switch (kind)
            {
            case MoEOverlayMigrationStorageKind::NoPeer:
            case MoEOverlayMigrationStorageKind::Disabled:
            case MoEOverlayMigrationStorageKind::DeviceTransferDirectory:
                break;
            case MoEOverlayMigrationStorageKind::PhysicalResidencyFabric:
                mutable_storage = true;
                break;
            default:
                throw std::logic_error("Prepared context sealing has an invalid storage owner");
            }
        }
        if (!authority)
            return ModelContextPhysicalSealBoundary::Unretained;
        if (authority->state() != ModelContextReuseAuthority::State::RunnerExclusive &&
            authority->state() != ModelContextReuseAuthority::State::Sealing)
            throw std::logic_error("Prepared context sealing does not own its reuse authority");
        return mutable_storage ? ModelContextPhysicalSealBoundary::LiveMutableResidency
                               : ModelContextPhysicalSealBoundary::RetiredImmutableSources;
    }

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

    ModelContextRetirementReceipt PendingExclusiveModelRetirement::complete(
        const std::shared_ptr<NativeContextPreparation>& original_source)
    {
        if(completed_)throw std::logic_error("Pending model-context retirement cannot complete twice");
        // begin() only captured allocation facts. Worker destruction starts in
        // complete(), so source-held calibration loans must close before it.
        // A rejected close leaves this same pending boundary unconsumed.
        if(original_source)original_source->retirePreparedCalibrations();
        return complete();
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

        // The weight/workspace BOM need not contain storage-only GPU workers.
        // Retain the original source through final owner destruction so its
        // remaining births receive the same reset proof as ordinary shutdown.
        auto final_native_source = contract.context->nativeContextPreparation();
        if (final_native_source &&
            (final_native_source->memory().get() != contract.physical_memory_authority.get() ||
             final_native_source->memory().owner_before(contract.physical_memory_authority) ||
             contract.physical_memory_authority.owner_before(final_native_source->memory())))
            throw std::logic_error("Final model source requires its original physical-memory authority");
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

        // This receipt continues to describe the sealed model-allocation BOM;
        // residual birth reset is an additional original-source lifetime proof.
        auto receipt = pending.complete(final_native_source);
        if (final_native_source)
            final_native_source->retireReleasedContexts();
        return receipt;
    }

    InvalidModelContextDiscardReceipt discardExclusiveInvalidModelContextReuseContract(ModelContextReuseContract& contract)
    {
        if(!contract.context || !contract.reusable_execution_workspaces || !contract.reuse_authority || !contract.physical_memory_authority)
            throw std::invalid_argument("Invalid model discard requires complete original final owners");
        if(contract.reuse_authority->state()!=ModelContextReuseAuthority::State::Invalid)
            throw std::logic_error("Invalid model discard requires the terminal Invalid lifecycle boundary");
        const auto diagnostic=contract.reuse_authority->diagnostic();
        if(contract.context.use_count()!=1u || contract.reusable_execution_workspaces.use_count()!=1u ||
           contract.reuse_authority.use_count()!=1u)
            throw std::logic_error("Invalid model discard still has borrowed model/workspace/lifecycle owners: "+diagnostic);
        auto source=contract.context->nativeContextPreparation();
        if(source){
            const auto memory=source->memory();
            if(memory.get()!=contract.physical_memory_authority.get() || memory.owner_before(contract.physical_memory_authority) ||
               contract.physical_memory_authority.owner_before(memory))
                throw std::logic_error("Invalid model discard requires its original source/PMA control block: "+diagnostic);
        }else{
            for(const auto& row:contract.physical_memory_authority->rankAttestation())
                if(row.identity.device.is_gpu())
                    throw std::logic_error("Invalid GPU model discard requires its original native source: "+diagnostic);
        }
        // Invalid means reuse failed, not that physical ownership disappeared.
        // Keep the original source through final owner destruction and then use
        // its existing exact central-ledger-zero native retirement transaction.
        contract.context.reset();
        contract.reusable_execution_workspaces.reset();
        contract.physical_memory_authority.reset();
        contract.reuse_authority.reset();
        contract.prepared_routed_weight_plan.reset();
        contract.expert_overlay_memory_admission.reset();
        contract.routed_weight_authority_identity.clear();
        return {source ? source->retireReleasedContexts() : 0u};
    }

} // namespace llaminar2
