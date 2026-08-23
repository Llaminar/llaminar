/**
 * @file CPUMoEKernel.cpp
 * @brief CPU implementation of MoE kernel operations
 *
 * Extracted from MoEExpertComputeStage.cpp to enable device-agnostic stage wiring.
 * Uses ISA-dispatched vector primitives for all compute-bound operations.
 */

#include "CPUMoEKernel.h"
#include "../../../execution/moe/MoEOverlayDeviceControllerABI.h"
#include "../../../execution/moe/MoEOverlayDeviceEpochProtocol.h"
#include "../../cpu/primitives/SoftmaxPrimitives_New.h"
#include "../../cpu/primitives/SwiGLUPrimitives.h"
#include "../../cpu/primitives/VectorPrimitives.h"
#include "../gemm/CPUNativeVNNIGemv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace llaminar2
{
    namespace
    {
        /** @brief Publish one complete CPU epoch-operation diagnostic. */
        void writeOverlayEpochStatus(
            DeviceMoEOverlayEpochStatus *status,
            DeviceMoEOverlayEpochOperation operation,
            DeviceMoEOverlayEpochStatusCode code,
            std::uint64_t epoch = 0u,
            std::uint64_t selector = 0u,
            std::uint32_t bank = kDeviceMoEOverlayInvalidBank,
            DeviceMoEOverlayEpochBankState state =
                DeviceMoEOverlayEpochBankState::Empty) noexcept
        {
            if (!status)
                return;
            *status = {
                .epoch = epoch,
                .selector = selector,
                .operation = static_cast<std::uint32_t>(operation),
                .code = static_cast<std::uint32_t>(code),
                .bank = bank,
                .observed_state = static_cast<std::uint32_t>(state),
            };
        }

        /** @return Current CPU selector reconstructed from atomic protocol reads. */
        std::uint64_t currentOverlaySelector(
            const MoEOverlayDeviceEpochProtocol &protocol) noexcept
        {
            const std::uint64_t generation =
                protocol.publicationGeneration();
            const std::uint32_t bank = protocol.publishedBank();
            if (generation == 0u || bank >= kDeviceMoEOverlayEpochBankCount)
                return 0u;
            return deviceMoEOverlayEpochSelector(generation, bank);
        }

        /** @return Bank retaining @p epoch in the authoritative CPU control block. */
        std::uint32_t overlayBankForEpoch(
            const MoEOverlayDeviceEpochProtocol &protocol,
            std::uint64_t epoch) noexcept
        {
            for (std::uint32_t bank = 0u;
                 bank < kDeviceMoEOverlayEpochBankCount;
                 ++bank)
            {
                if (protocol.bankEpoch(bank) == epoch)
                    return bank;
            }
            return kDeviceMoEOverlayInvalidBank;
        }

        /**
         * @brief CPU oracle for the node-local continuation admission barrier.
         *
         * The caller owns one extra `acquisitions_in_flight` guard for the whole
         * operation. This mirrors CUDA/HIP exactly: every participant protects
         * its local retiring bank before publishing arrival, then the immutable
         * publisher freezes the live topology admission once for all siblings.
         */
        bool resolveSynchronizedAdmission(
            const std::uint64_t *external_admission_epoch,
            DeviceMoEOverlayEpochAdmissionBarrierBinding barrier,
            std::uint64_t *required_epoch) noexcept
        {
            if (!external_admission_epoch || !barrier.record ||
                !required_epoch)
            {
                return false;
            }
            auto &record = *barrier.record;
            constexpr std::uint32_t kParticipantCount =
                kMoEOverlayDeviceControllerInferenceEpochMaxParticipants;
            constexpr std::uint32_t kValidMask =
                (1u << kParticipantCount) - 1u;
            const std::uint32_t participant = barrier.participant_id;
            const std::uint32_t mask = record.participant_mask;
            const std::uint32_t publisher =
                record.publisher_participant_id;
            if (record.magic !=
                    kMoEOverlayDeviceControllerInferenceEpochMagic ||
                record.version !=
                    kMoEOverlayDeviceControllerInferenceEpochVersion ||
                record.topology_fingerprint == 0u || mask == 0u ||
                (mask & ~kValidMask) != 0u ||
                participant >= kParticipantCount ||
                publisher >= kParticipantCount ||
                (mask & (1u << participant)) == 0u ||
                (mask & (1u << publisher)) == 0u)
            {
                return false;
            }

            auto own_arrival = std::atomic_ref<std::uint64_t>(
                record.arrival_sequence[participant]);
            const std::uint64_t previous = own_arrival.load(
                std::memory_order_acquire);
            if (previous == std::numeric_limits<std::uint64_t>::max())
                return false;
            const std::uint64_t sequence = previous + 1u;
            own_arrival.store(sequence, std::memory_order_release);

            auto publication = std::atomic_ref<std::uint64_t>(
                record.publication_sequence);
            if (participant == publisher)
            {
                for (std::uint32_t member = 0u;
                     member < kParticipantCount;
                     ++member)
                {
                    if ((mask & (1u << member)) == 0u)
                        continue;
                    auto arrival = std::atomic_ref<std::uint64_t>(
                        record.arrival_sequence[member]);
                    while (arrival.load(std::memory_order_acquire) < sequence)
                        std::this_thread::yield();
                }

                const std::uint64_t published = publication.load(
                    std::memory_order_acquire);
                if (published ==
                        std::numeric_limits<std::uint64_t>::max() ||
                    published + 1u != sequence)
                {
                    if (published < sequence)
                    {
                        std::atomic_ref<std::uint64_t>(record.epoch).store(
                            0u, std::memory_order_release);
                        publication.store(
                            sequence, std::memory_order_release);
                    }
                    return false;
                }
                const std::uint64_t frozen =
                    std::atomic_ref<std::uint64_t>(
                        *const_cast<std::uint64_t *>(
                            external_admission_epoch))
                        .load(std::memory_order_acquire);
                std::atomic_ref<std::uint64_t>(record.epoch).store(
                    frozen, std::memory_order_release);
                publication.store(sequence, std::memory_order_release);
            }
            else
            {
                std::uint64_t published = publication.load(
                    std::memory_order_acquire);
                while (published < sequence)
                {
                    std::this_thread::yield();
                    published = publication.load(std::memory_order_acquire);
                }
                if (published != sequence)
                    return false;
            }

            *required_epoch = std::atomic_ref<std::uint64_t>(record.epoch).load(
                std::memory_order_acquire);
            return *required_epoch != 0u;
        }
    } // namespace

    bool CPUMoEKernel::acquireMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status,
        const std::uint64_t *external_admission_epoch,
        DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier,
        MoEOverlayPeerPlacementEpochBinding peer_placement_epoch)
    {
        (void)launch;
        if (!status)
            return false;
        if (peer_placement_epoch.valid())
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Acquire,
                DeviceMoEOverlayEpochStatusCode::InvalidControl);
            return true;
        }
        if (!control)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Acquire,
                DeviceMoEOverlayEpochStatusCode::InvalidControl);
            return true;
        }
        if (!ticket || ticket->epoch != 0u || ticket->selector != 0u)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Acquire,
                DeviceMoEOverlayEpochStatusCode::InvalidTicket);
            return true;
        }

        MoEOverlayDeviceEpochProtocol protocol(*control);
        std::uint64_t admitted_epoch = 0u;
        bool admission_valid = true;
        const bool synchronized = admission_barrier.record != nullptr;
        if (synchronized)
        {
            /* Protect this participant's old bank before advertising arrival.
             * tryAcquireAdmitted owns its usual narrower guard as well; the
             * outer guard is dropped only after that reader is installed. */
            std::atomic_ref<std::uint64_t>(
                control->acquisitions_in_flight)
                .fetch_add(1u, std::memory_order_seq_cst);
            admission_valid = resolveSynchronizedAdmission(
                external_admission_epoch,
                admission_barrier,
                &admitted_epoch);
        }
        else if (admission_barrier.participant_id != 0xffffffffu)
        {
            admission_valid = false;
        }
        else if (external_admission_epoch)
        {
            admitted_epoch = std::atomic_ref<std::uint64_t>(
                                 *const_cast<std::uint64_t *>(
                                     external_admission_epoch))
                                 .load(std::memory_order_acquire);
        }

        const auto acquired = admission_valid
                                  ? protocol.tryAcquireAdmitted(admitted_epoch)
                                  : std::nullopt;
        if (synchronized)
        {
            std::atomic_ref<std::uint64_t>(
                control->acquisitions_in_flight)
                .fetch_sub(1u, std::memory_order_seq_cst);
        }
        if (!acquired.has_value())
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Acquire,
                DeviceMoEOverlayEpochStatusCode::InvalidControl,
                0u,
                currentOverlaySelector(protocol));
            return true;
        }

        *ticket = *acquired;
        writeOverlayEpochStatus(
            status,
            DeviceMoEOverlayEpochOperation::Acquire,
            DeviceMoEOverlayEpochStatusCode::Success,
            ticket->epoch,
            ticket->selector,
            ticket->bank(),
            protocol.bankState(ticket->bank()));
        return true;
    }

    bool CPUMoEKernel::releaseMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status)
    {
        (void)launch;
        if (!status)
            return false;
        if (!control)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Release,
                DeviceMoEOverlayEpochStatusCode::InvalidControl);
            return true;
        }
        if (!ticket || !ticket->valid())
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Release,
                DeviceMoEOverlayEpochStatusCode::InvalidTicket);
            return true;
        }

        MoEOverlayDeviceEpochProtocol protocol(*control);
        const DeviceMoEOverlayEpochTicket released = *ticket;
        const bool released_reader = protocol.release(released);
        if (released_reader)
            *ticket = {};
        writeOverlayEpochStatus(
            status,
            DeviceMoEOverlayEpochOperation::Release,
            released_reader
                ? DeviceMoEOverlayEpochStatusCode::Success
                : DeviceMoEOverlayEpochStatusCode::ReaderUnderflow,
            released.epoch,
            released.selector,
            released.bank(),
            protocol.bankState(released.bank()));
        return true;
    }

    bool CPUMoEKernel::reserveMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        (void)launch;
        if (!status)
            return false;
        if (!control)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidControl);
            return true;
        }
        if (!candidate_epoch || *candidate_epoch == 0u)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidArgument);
            return true;
        }

        MoEOverlayDeviceEpochProtocol protocol(*control);
        try
        {
            const std::uint32_t reusable_bank =
                1u - protocol.publishedBank();
            if (protocol.bankState(reusable_bank) ==
                DeviceMoEOverlayEpochBankState::Retiring)
            {
                /* Match the GPU captured reserve contract: a later maintenance
                 * poll reclaims the predecessor itself after all readers and
                 * delayed admissions drain. */
                (void)protocol.tryRetire(
                    protocol.bankEpoch(reusable_bank));
            }
            const auto bank = protocol.reserveCandidate(*candidate_epoch);
            const auto code = bank.has_value()
                                  ? DeviceMoEOverlayEpochStatusCode::Success
                                  : DeviceMoEOverlayEpochStatusCode::Busy;
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                code,
                *candidate_epoch,
                currentOverlaySelector(protocol),
                bank.value_or(kDeviceMoEOverlayInvalidBank),
                bank.has_value()
                    ? protocol.bankState(*bank)
                    : DeviceMoEOverlayEpochBankState::Empty);
        }
        catch (const std::invalid_argument &)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidArgument,
                *candidate_epoch,
                currentOverlaySelector(protocol));
        }
        catch (const std::logic_error &)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidControl,
                *candidate_epoch,
                currentOverlaySelector(protocol));
        }
        return true;
    }

    bool CPUMoEKernel::markMoEOverlayEpochCandidateReady(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        (void)launch;
        if (!status)
            return false;
        if (!control || !candidate_epoch || *candidate_epoch == 0u)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::MarkCandidateReady,
                control
                    ? DeviceMoEOverlayEpochStatusCode::InvalidArgument
                    : DeviceMoEOverlayEpochStatusCode::InvalidControl);
            return true;
        }

        MoEOverlayDeviceEpochProtocol protocol(*control);
        const std::uint32_t bank =
            overlayBankForEpoch(protocol, *candidate_epoch);
        try
        {
            protocol.markCandidateReady(*candidate_epoch);
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::MarkCandidateReady,
                DeviceMoEOverlayEpochStatusCode::Success,
                *candidate_epoch,
                currentOverlaySelector(protocol),
                bank,
                protocol.bankState(bank));
        }
        catch (const std::logic_error &)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::MarkCandidateReady,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                *candidate_epoch,
                currentOverlaySelector(protocol),
                bank,
                protocol.bankState(bank));
        }
        return true;
    }

    bool CPUMoEKernel::publishMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        (void)launch;
        if (!status)
            return false;
        if (!control || !candidate_epoch || *candidate_epoch == 0u)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                control
                    ? DeviceMoEOverlayEpochStatusCode::InvalidArgument
                    : DeviceMoEOverlayEpochStatusCode::InvalidControl);
            return true;
        }

        MoEOverlayDeviceEpochProtocol protocol(*control);
        const std::uint32_t bank =
            overlayBankForEpoch(protocol, *candidate_epoch);
        try
        {
            const auto publication =
                protocol.publishReadyCandidate(*candidate_epoch);
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::Success,
                publication.published_epoch,
                deviceMoEOverlayEpochSelector(
                    publication.generation,
                    publication.published_bank),
                publication.published_bank,
                protocol.bankState(publication.published_bank));
        }
        catch (const std::logic_error &)
        {
            const bool generation_exhausted =
                protocol.publicationGeneration() ==
                (std::numeric_limits<std::uint64_t>::max() >> 1u);
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                generation_exhausted
                    ? DeviceMoEOverlayEpochStatusCode::GenerationOverflow
                    : DeviceMoEOverlayEpochStatusCode::NotReady,
                *candidate_epoch,
                currentOverlaySelector(protocol),
                bank,
                protocol.bankState(bank));
        }
        return true;
    }

    bool CPUMoEKernel::abortMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        (void)launch;
        if (!status)
            return false;
        if (!control || !candidate_epoch || *candidate_epoch == 0u)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::AbortCandidate,
                control
                    ? DeviceMoEOverlayEpochStatusCode::InvalidArgument
                    : DeviceMoEOverlayEpochStatusCode::InvalidControl);
            return true;
        }

        MoEOverlayDeviceEpochProtocol protocol(*control);
        const std::uint32_t bank =
            overlayBankForEpoch(protocol, *candidate_epoch);
        const bool aborted = protocol.abortCandidate(*candidate_epoch);
        writeOverlayEpochStatus(
            status,
            DeviceMoEOverlayEpochOperation::AbortCandidate,
            aborted
                ? DeviceMoEOverlayEpochStatusCode::Success
                : DeviceMoEOverlayEpochStatusCode::NotReady,
            *candidate_epoch,
            currentOverlaySelector(protocol),
            bank,
            protocol.bankState(bank));
        return true;
    }

    bool CPUMoEKernel::retireMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *retiring_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        (void)launch;
        if (!status)
            return false;
        if (!control || !retiring_epoch || *retiring_epoch == 0u)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                control
                    ? DeviceMoEOverlayEpochStatusCode::InvalidArgument
                    : DeviceMoEOverlayEpochStatusCode::InvalidControl);
            return true;
        }

        MoEOverlayDeviceEpochProtocol protocol(*control);
        const std::uint32_t bank =
            overlayBankForEpoch(protocol, *retiring_epoch);
        try
        {
            const bool retired = protocol.tryRetire(*retiring_epoch);
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                retired
                    ? DeviceMoEOverlayEpochStatusCode::Success
                    : DeviceMoEOverlayEpochStatusCode::Busy,
                *retiring_epoch,
                currentOverlaySelector(protocol),
                bank,
                protocol.bankState(bank));
        }
        catch (const std::logic_error &)
        {
            writeOverlayEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                *retiring_epoch,
                currentOverlaySelector(protocol),
                bank,
                protocol.bankState(bank));
        }
        return true;
    }

    void CPUMoEKernel::invalidateRouterQ8HiddenPublication() noexcept
    {
        router_q8_hidden_source_ = nullptr;
        router_q8_hidden_rows_ = 0;
        router_q8_hidden_d_model_ = 0;
        router_q8_hidden_valid_ = false;
    }

    bool CPUMoEKernel::publishRouterQ8Hidden(
        const float *source,
        int rows,
        int d_model)
    {
        invalidateRouterQ8HiddenPublication();
        if (!source || rows < 1 || d_model <= 0)
            return false;

        const int blocks_per_row = (d_model + Q8_1Block::BLOCK_SIZE - 1) /
                                   Q8_1Block::BLOCK_SIZE;
        router_q8_hidden_.resize(
            static_cast<size_t>(rows) * static_cast<size_t>(blocks_per_row));

        if (!quantizeRouterQ8Rows(
                source,
                rows,
                d_model,
                /*source_row_indices=*/nullptr,
                rows,
                router_q8_hidden_.data(),
                router_q8_hidden_.size()))
        {
            return false;
        }

        router_q8_hidden_source_ = source;
        router_q8_hidden_rows_ = rows;
        router_q8_hidden_d_model_ = d_model;
        router_q8_hidden_valid_ = true;
        PerfStatsCollector::addCounter(
            "kernel",
            "cpu_moe_router_q8_hidden_publication_calls",
            1.0,
            "moe",
            "cpu",
            {{"rows", std::to_string(rows)},
             {"d_model", std::to_string(d_model)}});
        return true;
    }

    bool CPUMoEKernel::quantizeRouterQ8Rows(
        const float *source,
        int source_rows,
        int d_model,
        const int *source_row_indices,
        int output_rows,
        Q8_1Block *destination,
        size_t destination_blocks)
    {
        if (!source || source_rows < 1 || d_model <= 0 ||
            output_rows < 1 || !destination)
        {
            return false;
        }

        const int blocks_per_row =
            (d_model + Q8_1Block::BLOCK_SIZE - 1) /
            Q8_1Block::BLOCK_SIZE;
        if (static_cast<size_t>(output_rows) >
            std::numeric_limits<size_t>::max() /
                static_cast<size_t>(blocks_per_row))
        {
            return false;
        }
        const size_t required_blocks =
            static_cast<size_t>(output_rows) *
            static_cast<size_t>(blocks_per_row);
        if (required_blocks > destination_blocks)
            return false;
        if (source_row_indices)
        {
            for (int output_row = 0; output_row < output_rows; ++output_row)
            {
                const int source_row = source_row_indices[output_row];
                if (source_row < 0 || source_row >= source_rows)
                    return false;
            }
        }

        /*
         * Every row is arithmetically independent, so distributing complete
         * rows cannot change a Q8_1 byte. Keep MTP-sized publications serial:
         * there is too little work to repay an OpenMP fork. A transported
         * prefill packet can contain hundreds of rows, however, and the old
         * serial loop left the other physical cores idle before every CPU
         * expert layer. Require at least 16384 FP32 values (64 KiB) per useful
         * worker before opening a team; this derives the crossover from both
         * runtime geometry and the configured physical-core budget instead of
         * baking in one model-specific row threshold.
         */
        const bool rows_are_block_aligned = (d_model % Q8_1Block::BLOCK_SIZE) == 0;
        auto quantize_row = [&](int output_row)
        {
            const int source_row = source_row_indices
                                       ? source_row_indices[output_row]
                                       : output_row;
            const float *row_source =
                source + static_cast<size_t>(source_row) *
                             static_cast<size_t>(d_model);
            Q8_1Block *row_q8 =
                destination +
                static_cast<size_t>(output_row) *
                    static_cast<size_t>(blocks_per_row);
            int block = 0;
#if defined(__AVX512F__)
            if (rows_are_block_aligned)
            {
                for (; block + 1 < blocks_per_row; block += 2)
                {
                    simd::quantize_two_blocks_avx512(
                        row_source + static_cast<size_t>(block) * Q8_1Block::BLOCK_SIZE,
                        row_q8[block],
                        row_q8[block + 1]);
                }
            }
#endif
            for (; block < blocks_per_row; ++block)
            {
                const int block_start = block * Q8_1Block::BLOCK_SIZE;
                simd::quantize_single_block(
                    row_source + block_start,
                    row_q8[block],
                    std::min(
                        static_cast<int>(Q8_1Block::BLOCK_SIZE),
                        d_model - block_start));
            }
        };

        constexpr size_t kMinimumValuesPerWorker = 16384u;
        const int useful_workers =
            std::min(output_rows, omp_get_max_threads());
        const size_t total_values =
            static_cast<size_t>(output_rows) * static_cast<size_t>(d_model);
        const bool use_parallel_rows =
            useful_workers > 1 &&
            total_values >=
                static_cast<size_t>(useful_workers) *
                    kMinimumValuesPerWorker;
        if (use_parallel_rows)
        {
            auto quantize_rows = [&]()
            {
#pragma omp for schedule(static)
                for (int row = 0; row < output_rows; ++row)
                    quantize_row(row);
            };
            OMP_WORKSHARE_REGION(quantize_rows);
        }
        else
        {
            for (int row = 0; row < output_rows; ++row)
                quantize_row(row);
        }
        return true;
    }

    const Q8_1Block *CPUMoEKernel::publishedRouterQ8Hidden(
        const float *source,
        int rows,
        int d_model) const noexcept
    {
        if (!router_q8_hidden_valid_ ||
            !source ||
            router_q8_hidden_source_ != source ||
            rows <= 0 ||
            router_q8_hidden_rows_ < rows ||
            d_model <= 0 ||
            router_q8_hidden_d_model_ != d_model ||
            router_q8_hidden_.empty())
        {
            return nullptr;
        }
        return router_q8_hidden_.data();
    }

    bool CPUMoEKernel::publishTransportedRouterQ8Hidden(
        const float *source,
        int rows,
        int d_model)
    {
        return publishRouterQ8Hidden(source, rows, d_model);
    }

    bool CPUMoEKernel::publishTransportedRouterQ8HiddenExpertMajor(
        const float *source,
        int source_rows,
        int d_model,
        std::span<const int> expert_major_source_rows,
        std::span<Q8_1Block> destination)
    {
        invalidateRouterQ8HiddenPublication();
        if (expert_major_source_rows.empty() ||
            expert_major_source_rows.size() >
                static_cast<size_t>(std::numeric_limits<int>::max()))
            return false;
        const bool published = quantizeRouterQ8Rows(
            source,
            source_rows,
            d_model,
            expert_major_source_rows.data(),
            static_cast<int>(expert_major_source_rows.size()),
            destination.data(),
            destination.size());
        if (published)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_moe_transport_router_q8_expert_major_publication_calls",
                1.0,
                "moe",
                "cpu",
                {{"source_rows", std::to_string(source_rows)},
                 {"route_rows",
                  std::to_string(expert_major_source_rows.size())},
                 {"d_model", std::to_string(d_model)}});
        }
        return published;
    }

    bool CPUMoEKernel::reserveRouterQ8HiddenCapacity(
        int rows,
        int d_model)
    {
        invalidateRouterQ8HiddenPublication();
        if (rows <= 0 || d_model <= 0)
            return false;
        const size_t blocks_per_row =
            (static_cast<size_t>(d_model) + Q8_1Block::BLOCK_SIZE - 1u) /
            Q8_1Block::BLOCK_SIZE;
        if (static_cast<size_t>(rows) >
            std::numeric_limits<size_t>::max() / blocks_per_row)
        {
            return false;
        }
        router_q8_hidden_.resize(
            static_cast<size_t>(rows) * blocks_per_row);
        return true;
    }

    bool CPUMoEKernel::route(
        const float *hidden,
        const float *gate_weights,
        int seq_len, int d_model,
        int num_experts, int top_k,
        bool normalize_weights,
        MoERoutingResult &result)
    {
        invalidateRouterQ8HiddenPublication();
        if (!hidden || !gate_weights || seq_len <= 0 || d_model <= 0 ||
            num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[CPUMoEKernel::route] invalid routing pointer or dimensions");
            return false;
        }

        /*
         * Publish every routed row before routing so the immediately following
         * native expert stage never repeats this activation transform once per
         * selected expert. The publication vector grows to the runtime request
         * shape; speculative depth is not a four-row ABI.
         */
        if (!publishRouterQ8Hidden(hidden, seq_len, d_model))
        {
            LOG_ERROR("[CPUMoEKernel::route] failed to publish verifier Q8_1 hidden rows");
            return false;
        }

        result.expert_indices.resize(static_cast<size_t>(seq_len) * top_k);
        result.expert_weights.resize(static_cast<size_t>(seq_len) * top_k);
        result.router_logits.resize(static_cast<size_t>(seq_len) * num_experts);

        /*
         * One schedule serves both serial decode and grouped verification.
         * Parallelizing only the outer row loop leaves almost the whole socket
         * idle at the production MTP depths: M=3 engages three workers while
         * each worker streams a complete 2 MiB Qwen3.6-35B gate matrix. Flatten
         * the independent (row, expert) dot products across the whole team so
         * every positive M retains the M=1 router's expert parallelism.
         *
         * The first workshare writes each independent scalar logit with the
         * exact ISA-dispatched dot product used by M=1. Its implicit barrier
         * publishes the complete matrix before the second workshare assigns
         * whole rows to workers. Softmax, partial_sort, top-k summation, and
         * normalization then retain their serial per-row operation order. The
         * schedule changes ownership only; it cannot change any row's bytes.
         */
        auto route_rows = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < seq_len; ++row)
            {
                for (int expert = 0; expert < num_experts; ++expert)
                {
                    result.router_logits[
                        static_cast<size_t>(row) * num_experts + expert] =
                        primitives::vec_dot(
                            gate_weights +
                                static_cast<size_t>(expert) * d_model,
                            hidden + static_cast<size_t>(row) * d_model,
                            d_model);
                }
            }

#pragma omp for schedule(static)
            for (int row = 0; row < seq_len; ++row)
            {
                float *probabilities =
                    result.router_logits.data() +
                    static_cast<size_t>(row) * num_experts;
                primitives::softmax_row_fp32(probabilities, num_experts);

                /*
                 * The vector is private to one OpenMP worker and retains its
                 * capacity across calls. Consequently, warmed decode and MTP
                 * routing perform no per-transaction heap allocation.
                 */
                thread_local std::vector<int> indices;
                indices.resize(static_cast<size_t>(num_experts));
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(
                    indices.begin(),
                    indices.begin() + top_k,
                    indices.end(),
                    [probabilities](int left, int right)
                    {
                        return probabilities[left] > probabilities[right];
                    });

                float topk_sum = 0.0f;
                for (int k = 0; k < top_k; ++k)
                    topk_sum += probabilities[indices[k]];

                for (int k = 0; k < top_k; ++k)
                {
                    const size_t route_index =
                        static_cast<size_t>(row) * top_k + k;
                    result.expert_indices[route_index] = indices[k];
                    result.expert_weights[route_index] =
                        normalize_weights
                            ? probabilities[indices[k]] / topk_sum
                            : probabilities[indices[k]];
                }
            }
        };
        OMP_WORKSHARE_REGION(route_rows);

        if (seq_len > 1)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_moe_grouped_router_calls",
                1.0,
                "verifier",
                "cpu",
                {{"rows", std::to_string(seq_len)},
                 {"num_experts", std::to_string(num_experts)},
                 {"schedule", "row_expert_then_row_finalize"},
                 {"arithmetic_order", "serial_decode_per_row"}});
        }

        return true;
    }

    bool CPUMoEKernel::routeWithTensors(
        ITensor *hidden,
        ITensor *gate_weights,
        int seq_len,
        int d_model,
        int num_experts,
        int top_k,
        bool normalize_weights,
        ITensor *output_indices,
        ITensor *output_weights,
        MoERoutingResult &host_result)
    {
        if (!hidden || !gate_weights || !output_indices || !output_weights)
        {
            LOG_ERROR("[CPUMoEKernel::routeWithTensors] missing routing tensor");
            return false;
        }

        if (!route(hidden->data(),
                   gate_weights->data(),
                   seq_len,
                   d_model,
                   num_experts,
                   top_k,
                   normalize_weights,
                   host_result))
        {
            return false;
        }

        const size_t route_count =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (output_indices->numel() < route_count ||
            output_weights->numel() < route_count)
        {
            LOG_ERROR("[CPUMoEKernel::routeWithTensors] output capacity is too small");
            return false;
        }

        float *indices = output_indices->mutable_data();
        float *weights = output_weights->mutable_data();
        for (size_t route_index = 0; route_index < route_count; ++route_index)
        {
            indices[route_index] =
                static_cast<float>(host_result.expert_indices[route_index]);
            weights[route_index] = host_result.expert_weights[route_index];
        }
        return true;
    }

    void CPUMoEKernel::gatherTokenBatch(
        const float *hidden,
        float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model)
    {
        for (int i = 0; i < num_tokens; ++i)
        {
            const float *src = hidden + token_indices[i] * d_model;
            std::copy(src, src + d_model, batch_buffer + i * d_model);
        }
    }

    void CPUMoEKernel::scatterAddWeighted(
        float *output,
        const float *expert_output,
        const int *token_indices,
        const float *weights,
        int num_tokens, int d_model)
    {
        for (int i = 0; i < num_tokens; ++i)
        {
            primitives::vec_axpy(
                output + token_indices[i] * d_model,
                expert_output + i * d_model,
                weights[i], d_model);
        }
    }

    void CPUMoEKernel::sharedExpertGate(
        const float *input,
        const float *gate_inp,
        float *shared_output,
        int seq_len, int d_model)
    {
        // Fast serial path for decode (seq_len=1): OMP fork/join overhead
        // dominates for a single dot product + sigmoid + scale.
        if (seq_len <= 2)
        {
            for (int t = 0; t < seq_len; ++t)
            {
                const float *x = input + t * d_model;
                float dot = primitives::vec_dot(gate_inp, x, d_model);
                float gate = 1.0f / (1.0f + std::exp(-dot));
                float *out = shared_output + t * d_model;
                primitives::vec_scale(out, gate, d_model);
            }
            return;
        }

        auto do_work = [=]()
        {
#pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                const float *x = input + t * d_model;
                float dot = primitives::vec_dot(gate_inp, x, d_model);
                float gate = 1.0f / (1.0f + std::exp(-dot));

                float *out = shared_output + t * d_model;
                primitives::vec_scale(out, gate, d_model);
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }

    void CPUMoEKernel::swiGLU(float *gate, const float *up, int count)
    {
        primitives::compute_swiglu(gate, up, gate, count);
    }

} // namespace llaminar2
