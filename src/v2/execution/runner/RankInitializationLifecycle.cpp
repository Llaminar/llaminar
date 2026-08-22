/**
 * @file RankInitializationLifecycle.cpp
 * @brief Diagnostics for the typed rank initialization lifecycle.
 *
 * @author David Sanftenberg
 * @date August 2026
 */

#include "RankInitializationLifecycle.h"

#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <type_traits>
#include <vector>

namespace llaminar2
{
    namespace
    {
        constexpr std::size_t kMaximumInitializationPhaseNameBytes = 160;

        /** @brief Fixed-size, allocation-free wire record for one rank vote. */
        struct RankInitializationWireRecord
        {
            std::uint32_t ordinal{0};
            std::uint16_t name_size{0};
            std::uint8_t local_outcome{0};
            std::uint8_t reserved{0};
            std::array<char, kMaximumInitializationPhaseNameBytes> name{};
        };

        static_assert(
            std::is_trivially_copyable_v<RankInitializationWireRecord>);

        /** @brief Convert an MPI error code into a useful transport diagnostic. */
        std::string mpiInitializationError(
            std::string_view operation,
            int error_code)
        {
            std::array<char, MPI_MAX_ERROR_STRING> buffer{};
            int length = 0;
            if (MPI_Error_string(error_code, buffer.data(), &length) !=
                MPI_SUCCESS)
            {
                return std::string(operation) + " failed with MPI error " +
                       std::to_string(error_code);
            }
            return std::string(operation) + " failed: " +
                   std::string(buffer.data(), static_cast<std::size_t>(length));
        }
    } // namespace

    std::string RankInitializationPhaseResult::diagnostic(
        std::string_view workflow) const
    {
        std::string message(workflow);
        switch (status)
        {
        case RankInitializationPhaseStatus::Succeeded:
            message += " succeeded at step '";
            break;
        case RankInitializationPhaseStatus::LocalStepReturnedFailure:
            message += " step returned failure: '";
            break;
        case RankInitializationPhaseStatus::LocalStepThrewException:
            message += " step threw: '";
            break;
        case RankInitializationPhaseStatus::PeerStepFailed:
            message += " failed on another rank at step '";
            break;
        case RankInitializationPhaseStatus::PhaseIdentityMismatch:
            message += " phase identity mismatch at step '";
            break;
        case RankInitializationPhaseStatus::ConsensusTransportFailed:
            message += " consensus transport failed at step '";
            break;
        case RankInitializationPhaseStatus::InvalidConsensusResult:
            message += " received an invalid consensus result at step '";
            break;
        }
        message.append(identity.name);
        message += "'";
        if (!detail.empty())
        {
            message += ": ";
            message += detail;
        }
        return message;
    }

    RankInitializationConsensusResult MPIRankInitializationConsensus::reach(
        MPI_Comm communicator,
        RankInitializationPhaseIdentity identity,
        RankInitializationLocalOutcome local_outcome)
    {
        if (communicator == MPI_COMM_NULL)
        {
            return {
                .outcome =
                    RankInitializationConsensusOutcome::TransportFailure,
                .detail = "initialization communicator is MPI_COMM_NULL",
            };
        }
        if (identity.name.empty() ||
            identity.name.size() > kMaximumInitializationPhaseNameBytes)
        {
            return {
                .outcome =
                    RankInitializationConsensusOutcome::TransportFailure,
                .detail =
                    "initialization phase name is empty or exceeds the wire contract",
            };
        }

        int world_size = 0;
        const int size_result = MPI_Comm_size(communicator, &world_size);
        if (size_result != MPI_SUCCESS)
        {
            return {
                .outcome =
                    RankInitializationConsensusOutcome::TransportFailure,
                .detail = mpiInitializationError(
                    "MPI_Comm_size", size_result),
            };
        }
        if (world_size <= 0)
        {
            return {
                .outcome =
                    RankInitializationConsensusOutcome::TransportFailure,
                .detail = "initialization communicator has no ranks",
            };
        }

        RankInitializationWireRecord local;
        local.ordinal = identity.ordinal;
        local.name_size = static_cast<std::uint16_t>(identity.name.size());
        local.local_outcome = static_cast<std::uint8_t>(local_outcome);
        std::memcpy(local.name.data(), identity.name.data(), identity.name.size());

        std::vector<RankInitializationWireRecord> records(
            static_cast<std::size_t>(world_size));
        const int gather_result = MPI_Allgather(
            &local,
            static_cast<int>(sizeof(local)),
            MPI_BYTE,
            records.data(),
            static_cast<int>(sizeof(local)),
            MPI_BYTE,
            communicator);
        if (gather_result != MPI_SUCCESS)
        {
            return {
                .outcome =
                    RankInitializationConsensusOutcome::TransportFailure,
                .detail = mpiInitializationError(
                    "MPI_Allgather", gather_result),
            };
        }

        bool every_rank_succeeded = true;
        for (std::size_t rank = 0; rank < records.size(); ++rank)
        {
            const auto &record = records[rank];
            const bool same_name =
                record.name_size == identity.name.size() &&
                std::memcmp(
                    record.name.data(),
                    identity.name.data(),
                    identity.name.size()) == 0;
            if (record.ordinal != identity.ordinal || !same_name)
            {
                std::ostringstream detail;
                detail << "rank " << rank << " submitted phase ordinal "
                       << record.ordinal << " name '"
                       << std::string(
                              record.name.data(),
                              std::min<std::size_t>(
                                  record.name_size,
                                  record.name.size()))
                       << "' while local rank expected ordinal "
                       << identity.ordinal << " name '" << identity.name
                       << "'";
                return {
                    .outcome = RankInitializationConsensusOutcome::
                        PhaseIdentityMismatch,
                    .detail = detail.str(),
                };
            }

            if (record.local_outcome >
                static_cast<std::uint8_t>(
                    RankInitializationLocalOutcome::ThrewException))
            {
                return {
                    .outcome = RankInitializationConsensusOutcome::
                        PhaseIdentityMismatch,
                    .detail =
                        "peer submitted an invalid initialization outcome",
                };
            }
            every_rank_succeeded &=
                record.local_outcome == static_cast<std::uint8_t>(
                    RankInitializationLocalOutcome::Succeeded);
        }

        return {
            .outcome = every_rank_succeeded
                           ? RankInitializationConsensusOutcome::
                                 AllRanksSucceeded
                           : RankInitializationConsensusOutcome::
                                 AtLeastOneRankFailed,
            .detail = {},
        };
    }

} // namespace llaminar2
