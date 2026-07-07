#include "DeviceMoETransferSlotDirectory.h"

#include "../../backends/IBackend.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../../utils/Logger.h"
#include "../../utils/VramBillOfMaterials.h"

#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        DeviceNativeVNNIMatrixDesc descriptorFromSlot(
            const WeightVRAMPool::WeightSlot &slot,
            const DeviceMoETransferSlotDirectory::ProjectionSpec &spec)
        {
            DeviceNativeVNNIMatrixDesc desc;
            desc.payload = slot.d_native_vnni_payload;
            desc.scales = slot.d_native_vnni_scales;
            desc.mins = slot.d_native_vnni_mins;
            desc.emins = slot.d_native_vnni_emins;
            desc.n = spec.N;
            desc.k = spec.K;
            desc.blocks_per_row = static_cast<uint32_t>(spec.K / 32);
            desc.codebook_id = spec.codebook_id;
            return desc;
        }

        void setProjectionDescriptor(
            DeviceMoEExpertDescriptor &expert_desc,
            const std::string &label,
            const DeviceNativeVNNIMatrixDesc &matrix_desc)
        {
            if (label == "gate")
                expert_desc.gate = matrix_desc;
            else if (label == "up")
                expert_desc.up = matrix_desc;
            else if (label == "down")
                expert_desc.down = matrix_desc;
            else
                throw std::invalid_argument(
                    "DeviceMoETransferSlotDirectory projection label must be gate, up, or down");
        }

        void validateSpecs(
            const std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec> &specs)
        {
            if (specs.size() != 3)
                throw std::invalid_argument("DeviceMoETransferSlotDirectory requires gate/up/down specs");

            bool has_gate = false;
            bool has_up = false;
            bool has_down = false;
            for (const auto &spec : specs)
            {
                if (spec.label == "gate")
                    has_gate = true;
                else if (spec.label == "up")
                    has_up = true;
                else if (spec.label == "down")
                    has_down = true;
                else
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory projection label must be gate, up, or down");

                if (spec.N <= 0 || spec.K <= 0 || (spec.K % 32) != 0 ||
                    spec.payload_bytes_per_block <= 0)
                {
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory projection spec has invalid NativeVNNI shape");
                }
            }

            if (!has_gate || !has_up || !has_down)
                throw std::invalid_argument("DeviceMoETransferSlotDirectory requires gate/up/down specs");
        }

        size_t projectionPayloadBytes(
            const DeviceMoETransferSlotDirectory::ProjectionSpec &spec)
        {
            const size_t block_count =
                static_cast<size_t>(spec.N) * static_cast<size_t>(spec.K / 32);
            const size_t scales_bytes = block_count * sizeof(uint16_t);
            return block_count * static_cast<size_t>(spec.payload_bytes_per_block) +
                   scales_bytes +
                   (spec.is_asymmetric ? scales_bytes : 0u) +
                   (spec.has_emins ? block_count * sizeof(uint32_t) : 0u);
        }

        size_t expertPayloadBytes(
            const std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec> &specs)
        {
            size_t bytes = 0;
            for (const auto &spec : specs)
                bytes += projectionPayloadBytes(spec);
            return bytes;
        }
    } // namespace

    std::shared_ptr<DeviceMoETransferSlotDirectory>
    DeviceMoETransferSlotDirectory::create(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        uint32_t participant_id,
        uint32_t slot_count,
        std::vector<ProjectionSpec> specs,
        size_t vram_safety_margin_bytes)
    {
        if (!backend)
            throw std::invalid_argument("DeviceMoETransferSlotDirectory requires a backend");
        if (!device.is_gpu())
            throw std::invalid_argument("DeviceMoETransferSlotDirectory requires a GPU device");
        if (device_ordinal < 0)
            throw std::invalid_argument("DeviceMoETransferSlotDirectory requires a GPU device ordinal");
        if (slot_count == 0)
            throw std::invalid_argument("DeviceMoETransferSlotDirectory requires at least one slot");
        validateSpecs(specs);

        auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
        orchestrator->setVramPreflightSafetyMarginBytes(vram_safety_margin_bytes);
        orchestrator->addDevice(device_ordinal);

        for (uint32_t slot = 0; slot < slot_count; ++slot)
        {
            for (const auto &spec : specs)
            {
                orchestrator->planWeight(
                    device_ordinal,
                    slotName(slot, spec.label),
                    spec.N,
                    spec.K,
                    spec.payload_bytes_per_block,
                    spec.is_asymmetric,
                    spec.has_emins,
                    /*raw_gguf_bytes=*/0);
            }
        }
        orchestrator->allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0);

        const auto *weight_pool = orchestrator->getPool(device_ordinal);
        if (!weight_pool || !weight_pool->isAllocated())
            throw std::runtime_error("DeviceMoETransferSlotDirectory failed to allocate slot weights");
        const size_t planned_bytes = weight_pool->totalPlannedBytes();
        const size_t slot_payload_bytes = expertPayloadBytes(specs);

        std::vector<DeviceMoEExpertDirectoryEntry> host_entries(slot_count);
        for (uint32_t slot = 0; slot < slot_count; ++slot)
        {
            DeviceMoEExpertDirectoryEntry entry;
            entry.layer = kDeviceMoEInvalidSlot;
            entry.expert = kDeviceMoEInvalidSlot;
            entry.participant = participant_id;
            entry.slot_index = slot;
            entry.descriptor.logical_expert_id = -1;
            entry.descriptor.owner_participant = static_cast<int32_t>(participant_id);
            entry.descriptor.local_slot = static_cast<int32_t>(slot);
            entry.descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                                      DeviceMoEExpertFlags::TransferSlot);
            entry.flags =
                static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
                static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot);
            entry.payload_bytes_per_block =
                static_cast<uint8_t>(specs.front().payload_bytes_per_block);
            entry.is_asymmetric = specs.front().is_asymmetric ? 1u : 0u;
            entry.has_emins = specs.front().has_emins ? 1u : 0u;

            for (const auto &spec : specs)
            {
                auto maybe_slot = weight_pool->getSlot(slotName(slot, spec.label));
                if (!maybe_slot.has_value())
                    throw std::runtime_error("DeviceMoETransferSlotDirectory missing planned transfer slot");
                setProjectionDescriptor(
                    entry.descriptor,
                    spec.label,
                    descriptorFromSlot(*maybe_slot, spec));
            }
            if (!deviceMoETransferSlotReadyForCopy(
                    entry,
                    participant_id,
                    /*layer=*/0,
                    /*expert=*/0))
            {
                throw std::runtime_error(
                    "DeviceMoETransferSlotDirectory produced an unusable generic transfer slot");
            }
            host_entries[slot] = entry;
        }

        DeviceMoEExpertDirectoryEntry *device_entries =
            static_cast<DeviceMoEExpertDirectoryEntry *>(
                backend->allocate(host_entries.size() * sizeof(host_entries[0]), device_ordinal));
        if (!device_entries)
            throw std::runtime_error("DeviceMoETransferSlotDirectory failed to allocate device directory");

        void *upload_stream = backend->createStream(device_ordinal);
        if (!upload_stream)
        {
            backend->free(device_entries, device_ordinal);
            throw std::runtime_error("DeviceMoETransferSlotDirectory failed to create upload stream");
        }

        const size_t directory_bytes = host_entries.size() * sizeof(host_entries[0]);
        const bool uploaded =
            backend->hostToDeviceOnStream(
                device_entries,
                host_entries.data(),
                directory_bytes,
                device_ordinal,
                upload_stream) &&
            backend->synchronizeStream(upload_stream, device_ordinal);
        backend->destroyStream(upload_stream, device_ordinal);
        if (!uploaded)
        {
            backend->free(device_entries, device_ordinal);
            throw std::runtime_error("DeviceMoETransferSlotDirectory failed to upload device directory");
        }

        auto directory = std::shared_ptr<DeviceMoETransferSlotDirectory>(
            new DeviceMoETransferSlotDirectory(
                backend,
                device,
                device_ordinal,
                participant_id,
                slot_count,
                std::move(specs),
                std::move(orchestrator),
                device_entries,
                std::move(host_entries),
                planned_bytes,
                slot_payload_bytes));

        logVramBomLine(
            "moe_device_transfer_slot_directory",
            "backend=" + std::string(backend->backendName()) +
                " device=" + device.to_string() +
                " device_id=" + std::to_string(device_ordinal) +
                " participant=" + std::to_string(participant_id) +
                " slots=" + std::to_string(slot_count) +
                " projections_per_slot=" + std::to_string(directory->specs_.size()) +
                " collective_slot_payload_bytes=" + std::to_string(slot_payload_bytes) +
                " slot_payload_bytes=" + std::to_string(planned_bytes) +
                " slot_payload_mib=" + vramBomMiB(planned_bytes) +
                " directory_bytes=" + std::to_string(directory_bytes) +
                " directory_mib=" + vramBomMiB(directory_bytes));
        return directory;
    }

    DeviceMoETransferSlotDirectory::DeviceMoETransferSlotDirectory(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        uint32_t participant_id,
        uint32_t slot_count,
        std::vector<ProjectionSpec> specs,
        std::shared_ptr<LoadOrchestrator> orchestrator,
        DeviceMoEExpertDirectoryEntry *device_entries,
        std::vector<DeviceMoEExpertDirectoryEntry> host_entries,
        size_t planned_bytes,
        size_t slot_payload_bytes)
        : backend_(backend),
          device_(device),
          device_ordinal_(device_ordinal),
          participant_id_(participant_id),
          slot_count_(slot_count),
          specs_(std::move(specs)),
          orchestrator_(std::move(orchestrator)),
          device_entries_(device_entries),
          host_entries_(std::move(host_entries)),
          planned_bytes_(planned_bytes),
          slot_payload_bytes_(slot_payload_bytes)
    {
    }

    DeviceMoETransferSlotDirectory::~DeviceMoETransferSlotDirectory()
    {
        if (device_entries_ && backend_)
        {
            backend_->free(device_entries_, device_ordinal_);
            device_entries_ = nullptr;
        }
    }

    bool DeviceMoETransferSlotDirectory::descriptorForSlot(
        uint32_t slot_index,
        uint32_t logical_expert,
        DeviceMoEExpertDescriptor &out) const
    {
        if (slot_index >= host_entries_.size())
            return false;

        const auto &entry = host_entries_[static_cast<size_t>(slot_index)];
        if (!deviceMoETransferSlotReadyForCopy(
                entry,
                participant_id_,
                /*layer=*/0,
                /*expert=*/0))
        {
            return false;
        }

        out = entry.descriptor;
        out.logical_expert_id = static_cast<int32_t>(logical_expert);
        out.local_slot = static_cast<int32_t>(slot_index);
        out.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                      DeviceMoEExpertFlags::Resident |
                                      DeviceMoEExpertFlags::TransferSlot);
        return true;
    }

    std::string DeviceMoETransferSlotDirectory::slotName(
        uint32_t slot_index,
        const std::string &label)
    {
        return "moe_device_rebalance_transfer_slot_" +
               std::to_string(slot_index) + "_" + label;
    }

} // namespace llaminar2
