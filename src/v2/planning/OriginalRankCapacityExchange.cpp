/**
 * @file OriginalRankCapacityExchange.cpp
 * @brief Existing CAP4 and final immutable-table metadata exchange on original rank transports.
 *
 * The byte codec and parent collectives are shared by ordinary overlay and
 * global pipeline admission. These are cold certified BOM contributions,
 * never model payloads, mutable inference state or a second capacity ledger.
 */
#include "OriginalRankCapacityExchange.h"
#include "execution/runner/RankInitializationLifecycle.h"
#include "execution/moe/MoEOverlayLocalCapacityPlanner.h"
#include <algorithm>
#include <climits>
#include <functional>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>
namespace llaminar2 {
namespace {
    /** @brief Append one unsigned scalar in a host-independent wire order. */
    template <typename UInt>
    void appendCapacityWireScalar(
        std::vector<std::uint8_t> &bytes,
        UInt value)
    {
        static_assert(std::is_unsigned_v<UInt>);
        for (std::size_t offset = 0; offset < sizeof(UInt); ++offset)
        {
            bytes.push_back(static_cast<std::uint8_t>(
                value >> (offset * 8u)));
        }
    }

    /** @brief Read one bounds-checked unsigned little-endian scalar. */
    template <typename UInt>
    UInt readCapacityWireScalar(
        std::span<const std::uint8_t> bytes,
        std::size_t &cursor,
        std::size_t end)
    {
        static_assert(std::is_unsigned_v<UInt>);
        if (cursor > end || sizeof(UInt) > end - cursor)
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity wire record is truncated");
        }
        UInt value = 0;
        for (std::size_t offset = 0; offset < sizeof(UInt); ++offset)
        {
            value |= static_cast<UInt>(bytes[cursor++]) <<
                     (offset * 8u);
        }
        return value;
    }

}
    /** @brief Serialize rank-local capacity records without ABI padding. */
    std::vector<std::uint8_t> serializeOverlayCapacityBudgets(
        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &budgets)
    {
        constexpr std::uint32_t kMagic = 0x43415034u; // "CAP4"
        constexpr std::uint32_t kVersion = 5; // v5 adds physical backing + pool observation
        constexpr std::size_t kHeaderBytes = 52;
        constexpr std::size_t kOwnerBytes = 16;
        constexpr std::size_t kRecordBytes =
            kHeaderBytes +
            PhysicalMemoryBOM::ownerCount() * kOwnerBytes;
        std::vector<std::uint8_t> bytes;
        if (budgets.size() >
            std::numeric_limits<std::size_t>::max() / kRecordBytes)
        {
            throw std::overflow_error(
                "ExpertOverlay capacity wire record count overflows size_t");
        }
        bytes.reserve(budgets.size() * kRecordBytes);
        for (const auto &budget : budgets)
        {
            appendCapacityWireScalar(bytes, kMagic);
            appendCapacityWireScalar(bytes, kVersion);
            appendCapacityWireScalar(
                bytes, static_cast<std::uint32_t>(budget.worldRank()));
            appendCapacityWireScalar(
                bytes, static_cast<std::uint32_t>(budget.device().type));
            appendCapacityWireScalar(
                bytes, static_cast<std::uint32_t>(budget.device().ordinal));
            appendCapacityWireScalar(
                bytes,
                static_cast<std::uint32_t>(
                    PhysicalMemoryBOM::ownerCount()));
            const auto &bom = budget.bom();
            appendCapacityWireScalar(
                bytes,
                static_cast<std::uint64_t>(
                    bom.resource().total_bytes));
            appendCapacityWireScalar(
                bytes,
                static_cast<std::uint64_t>(
                    bom.resource().admission_available_bytes));
            appendCapacityWireScalar(
                bytes, static_cast<std::uint32_t>(bom.resource().backing));
            appendCapacityWireScalar(
                bytes,
                static_cast<std::uint64_t>(
                    bom.resource().pool_available_bytes));
            for (std::size_t owner_index = 0;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto &charge = bom.charge(
                    PhysicalMemoryBOM::ownerAt(owner_index));
                appendCapacityWireScalar(
                    bytes,
                    static_cast<std::uint64_t>(
                        charge.planned_bytes));
                appendCapacityWireScalar(
                    bytes,
                    static_cast<std::uint64_t>(
                        charge.already_resident_bytes));
            }
        }
        return bytes;
    }

    /** @brief Parse and authenticate one rank segment of capacity records. */
    void parseOverlayCapacityBudgetSegment(
        std::span<const std::uint8_t> bytes,
        std::size_t begin,
        std::size_t end,
        int contributing_rank,
        std::vector<MoEOverlayBoundPhysicalMemoryBudget> &output)
    {
        constexpr std::uint32_t kMagic = 0x43415034u;
        constexpr std::uint32_t kVersion = 5; // v5 adds physical backing + pool observation
        constexpr std::size_t kHeaderBytes = 52;
        constexpr std::size_t kOwnerBytes = 16;
        constexpr std::size_t kRecordBytes =
            kHeaderBytes +
            PhysicalMemoryBOM::ownerCount() * kOwnerBytes;
        if (begin > end || end > bytes.size() ||
            (end - begin) % kRecordBytes != 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity rank segment has invalid framing");
        }
        std::size_t cursor = begin;
        while (cursor < end)
        {
            const auto magic = readCapacityWireScalar<std::uint32_t>(
                bytes, cursor, end);
            const auto version = readCapacityWireScalar<std::uint32_t>(
                bytes, cursor, end);
            const int world_rank = static_cast<int>(
                readCapacityWireScalar<std::uint32_t>(
                    bytes, cursor, end));
            const auto device_type = static_cast<DeviceType>(
                readCapacityWireScalar<std::uint32_t>(
                    bytes, cursor, end));
            const int ordinal = static_cast<int>(
                readCapacityWireScalar<std::uint32_t>(
                    bytes, cursor, end));
            const auto owner_count =
                readCapacityWireScalar<std::uint32_t>(
                    bytes, cursor, end);
            const auto total_bytes =
                readCapacityWireScalar<std::uint64_t>(
                    bytes, cursor, end);
            const auto available_bytes =
                readCapacityWireScalar<std::uint64_t>(
                    bytes, cursor, end);
            const auto backing_value =
                readCapacityWireScalar<std::uint32_t>(
                    bytes, cursor, end);
            const auto pool_available_bytes =
                readCapacityWireScalar<std::uint64_t>(
                    bytes, cursor, end);
            if (backing_value > std::numeric_limits<std::uint8_t>::max() ||
                pool_available_bytes > std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument(
                    "ExpertOverlay capacity wire backing failed range validation");
            const auto backing = physicalMemoryBackingFromWire(
                static_cast<std::uint8_t>(backing_value));
            const DeviceId device(device_type, ordinal);
            if (magic != kMagic || version != kVersion ||
                world_rank != contributing_rank ||
                owner_count != PhysicalMemoryBOM::ownerCount() ||
                !device.is_valid() ||
                (!device.is_cpu() && !device.is_gpu()) ||
                total_bytes > std::numeric_limits<std::size_t>::max() ||
                available_bytes >
                    std::numeric_limits<std::size_t>::max())
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity wire record failed identity or range validation");
            }
            PhysicalMemoryBOMBuilder builder({
                .world_rank = world_rank,
                .device = device,
                .total_bytes = static_cast<std::size_t>(total_bytes),
                .admission_available_bytes =
                    static_cast<std::size_t>(available_bytes),
                .backing = backing,
                .pool_available_bytes =
                    static_cast<std::size_t>(pool_available_bytes),
            });
            for (std::size_t owner_index = 0;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto planned =
                    readCapacityWireScalar<std::uint64_t>(
                        bytes, cursor, end);
                const auto resident =
                    readCapacityWireScalar<std::uint64_t>(
                        bytes, cursor, end);
                if (planned >
                        std::numeric_limits<std::size_t>::max() ||
                    resident >
                        std::numeric_limits<std::size_t>::max())
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity wire owner charge exceeds size_t");
                }
                builder.add(
                    PhysicalMemoryBOM::ownerAt(owner_index),
                    static_cast<std::size_t>(planned),
                    static_cast<std::size_t>(resident));
            }
            output.emplace_back(
                MoEOverlayLocalCapacityPlanner::physicalResourceId(
                    world_rank, device),
                PhysicalMemoryAdmissionCertificate(builder.build()));
        }
    }

    /** @brief All-gather variable rank-local physical BOMs in rank order. */
    std::vector<MoEOverlayBoundPhysicalMemoryBudget>
    gatherOverlayCapacityBudgets(
        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &local,
        const std::shared_ptr<IMPIContext> &mpi_context)
    {
        if (!mpi_context || mpi_context->world_size() <= 1)
            return local;

        const auto send = serializeOverlayCapacityBudgets(local);
        if (send.size() > static_cast<std::size_t>(INT_MAX))
        {
            throw std::overflow_error(
                "ExpertOverlay rank-local capacity wire payload exceeds MPI int count");
        }
        const int world_size = mpi_context->world_size();
        const int send_count = static_cast<int>(send.size());
        std::vector<int> counts(static_cast<std::size_t>(world_size), 0);
        mpi_context->allgather_bytes(
            &send_count, counts.data(), sizeof(send_count));

        std::vector<int> displacements(
            static_cast<std::size_t>(world_size), 0);
        std::size_t total = 0;
        for (int rank = 0; rank < world_size; ++rank)
        {
            if (counts[static_cast<std::size_t>(rank)] < 0 ||
                total > static_cast<std::size_t>(INT_MAX))
            {
                throw std::overflow_error(
                    "ExpertOverlay global capacity wire payload exceeds MPI int displacement");
            }
            displacements[static_cast<std::size_t>(rank)] =
                static_cast<int>(total);
            total += static_cast<std::size_t>(
                counts[static_cast<std::size_t>(rank)]);
        }
        if (total > static_cast<std::size_t>(INT_MAX))
        {
            throw std::overflow_error(
                "ExpertOverlay global capacity wire payload exceeds MPI int count");
        }

        std::vector<std::uint8_t> received(total);
        mpi_context->allgatherv_bytes(
            send.empty() ? nullptr : send.data(),
            send_count,
            received.empty() ? nullptr : received.data(),
            counts.data(),
            displacements.data());

        std::vector<MoEOverlayBoundPhysicalMemoryBudget> result;
        for (int rank = 0; rank < world_size; ++rank)
        {
            const std::size_t begin = static_cast<std::size_t>(
                displacements[static_cast<std::size_t>(rank)]);
            const std::size_t end = begin + static_cast<std::size_t>(
                counts[static_cast<std::size_t>(rank)]);
            parseOverlayCapacityBudgetSegment(
                received, begin, end, rank, result);
        }
        return result;
    }

    /**
     * @brief Gather immutable final PLE requests after the higher-priority row/quota search.
     * @param frame Exact sender bytes validated before readiness consensus.
     * @param world Existing overlay startup communicator; no inference state is exchanged.
     * @param profile Canonical source metadata authenticating received requests.
     * @return Rank-ordered source/consumer/mode BOM requests for one final certificate.
     * @throws std::exception For malformed framing, foreign identity or an oversized MPI extent.
     */
    std::vector<ImmutableRowTableAdmissionRequest> gatherOverlayImmutableTableRequests(
        const ImmutableRowTableAdmissionSendFrame &frame,
        const std::shared_ptr<IMPIContext> &world,const ModelMemoryProfile &profile)
    {
        if(!world||world->world_size()<=1)
            return parseImmutableRowTableAdmissionRequests(frame.bytes(),frame.senderRank(),&profile);
        // Sender validation and serialization already succeeded on every
        // peer's input-readiness vote. No fallible local wire preparation
        // remains between that agreement and the first collective.
        const auto send=frame.bytes();
        const int n=world->world_size(),send_count=frame.byteCount();
        std::vector<int> counts(n),offsets(n);
        world->allgather_bytes(&send_count,counts.data(),sizeof(send_count));
        std::size_t total=0;
        for(int rank=0;rank<n;++rank)
        {
            if(counts[rank]<0||total>static_cast<std::size_t>(INT_MAX)-static_cast<std::size_t>(counts[rank]))
                throw std::overflow_error("Ngram admission aggregate metadata exceeds MPI extent");
            offsets[rank]=static_cast<int>(total);total+=static_cast<std::size_t>(counts[rank]);
        }
        std::vector<std::uint8_t> received(total);
        world->allgatherv_bytes(send.data(),send_count,received.data(),counts.data(),offsets.data());
        std::vector<ImmutableRowTableAdmissionRequest> requests;
        for(int rank=0;rank<n;++rank)
        {
            const auto part=std::span<const std::uint8_t>(received).subspan(
                static_cast<std::size_t>(offsets[rank]),static_cast<std::size_t>(counts[rank]));
            auto decoded=parseImmutableRowTableAdmissionRequests(part,rank,&profile);
            requests.insert(requests.end(),std::make_move_iterator(decoded.begin()),std::make_move_iterator(decoded.end()));
        }
        return requests;
    }

    /**
     * @brief Prove every overlay rank will execute the same admission loop.
     *
     * Capacity admission performs collectives once per candidate. A rank
     * with a different configured bucket ladder would otherwise leave its
     * peers inside an unmatched collective. First gathering the count gives
     * every rank the same safe exit point; only equal counts permit the
     * second, fixed-width gather of the exact candidate values.
     */
    void requireIdenticalOverlayGraphRowCandidates(
        const std::vector<int> &local_candidates,
        const std::shared_ptr<IMPIContext> &mpi_context)
    {
        if (local_candidates.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity admission requires at least one graph-row candidate");
        }
        const bool uncaptured_contract =
            local_candidates.size() == 1u &&
            local_candidates.front() == 0;
        if (!uncaptured_contract &&
            (std::any_of(
                 local_candidates.begin(),
                 local_candidates.end(),
                 [](int rows) { return rows <= 0; }) ||
             !std::is_sorted(
                 local_candidates.begin(),
                 local_candidates.end(),
                 std::greater<int>{})))
        {
            throw std::invalid_argument(
                "ExpertOverlay graph-row candidates must be positive and descending");
        }
        if (!mpi_context || mpi_context->world_size() <= 1)
            return;
        if (local_candidates.size() >
            static_cast<std::size_t>(INT_MAX) / sizeof(int))
        {
            throw std::overflow_error(
                "ExpertOverlay graph-row candidate payload exceeds MPI int count");
        }

        const int world_size = mpi_context->world_size();
        const int local_count = static_cast<int>(local_candidates.size());
        std::vector<int> counts(
            static_cast<std::size_t>(world_size), 0);
        mpi_context->allgather_bytes(
            &local_count, counts.data(), sizeof(local_count));
        if (!std::all_of(
                counts.begin(), counts.end(),
                [&](int count) { return count == local_count; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay ranks disagree on the captured-prefill candidate count");
        }

        const auto candidate_count =
            static_cast<std::size_t>(local_count);
        if (candidate_count >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(world_size))
        {
            throw std::overflow_error(
                "ExpertOverlay gathered graph-row candidates overflow size_t");
        }
        std::vector<int> gathered(
            candidate_count * static_cast<std::size_t>(world_size));
        mpi_context->allgather_bytes(
            local_candidates.data(),
            gathered.data(),
            candidate_count * sizeof(int));
        for (int rank = 0; rank < world_size; ++rank)
        {
            const auto begin = gathered.begin() +
                               static_cast<std::ptrdiff_t>(
                                   candidate_count *
                                   static_cast<std::size_t>(rank));
            if (!std::equal(
                    local_candidates.begin(),
                    local_candidates.end(),
                    begin))
            {
                throw std::invalid_argument(
                    "ExpertOverlay ranks disagree on the captured-prefill candidate ladder");
            }
        }
    }

std::optional<std::string> runOriginalPipelineCapacityTrial(
    const IMPIContext& original, int rows, std::string_view phase, std::string_view diagnostic,
    const std::function<void()>& work)
{
    const auto vote = [&](std::string_view name, auto&& step) {
        auto result = RankInitializationLifecycle::execute(
            {static_cast<std::uint32_t>(rows), name}, std::forward<decltype(step)>(step),
            [&](auto identity, auto outcome) {
                return MPIRankInitializationConsensus::reach(original.communicator(), identity, outcome);
            });
        switch (result.status) {
        case RankInitializationPhaseStatus::Succeeded:
        case RankInitializationPhaseStatus::LocalStepReturnedFailure:
        case RankInitializationPhaseStatus::LocalStepThrewException:
        case RankInitializationPhaseStatus::PeerStepFailed:
            return result;
        default:
            throw std::runtime_error(result.diagnostic(diagnostic));
        }
    };
    std::string exhausted;
    const auto ready = vote(phase, [&] {
        try {
            work();
            return true;
        } catch (const PhysicalMemoryCapacityExhausted& error) {
            exhausted = error.what();
            return false;
        }
    });
    if (ready.succeeded())
        return std::nullopt;

    // A peer must distinguish malformed input from actual capacity pressure
    // before either rank can enter the next candidate or payload collective.
    const auto retry = vote("pipeline-capacity.rejected-row-kind", [&] {
        return ready.status != RankInitializationPhaseStatus::LocalStepThrewException;
    });
    if (!retry.succeeded())
        throw std::invalid_argument(ready.diagnostic(diagnostic));
    auto detail = ready.diagnostic(diagnostic);
    if (!exhausted.empty())
        detail += ": " + exhausted;
    return detail;
}

namespace {
/** @brief Authenticate original stage membership without creating a rank/context or probing hardware. */
std::vector<int> originalLocalStageIDs(const GlobalPPTopology& topology,int sender) {
    if(sender<0 || sender>=topology.world_size || !topology.validate().empty())
        throw std::invalid_argument("Pipeline capacity frame requires its original complete topology/rank");
    std::vector<int> result;
    for(const auto& stage:topology.stages)if(stage.rankParticipates(sender))result.push_back(stage.stage_id);
    return result;
}
constexpr std::uint32_t kPipelineCapacityMagic=0x31504350u,kPipelineCapacityVersion=1;
}
OriginalPipelineCapacitySendFrame OriginalPipelineCapacitySendFrame::prepare(
    const GlobalPPTopology& topology,int sender,std::span<const OriginalPipelineFixedContribution> local) {
    const auto expected=originalLocalStageIDs(topology,sender);
    if(local.size()!=expected.size() || local.size()>static_cast<size_t>(INT_MAX))
        throw std::invalid_argument("Pipeline capacity sender omitted/duplicated an original local stage");
    std::vector<std::uint8_t> bytes;
    appendCapacityWireScalar(bytes,kPipelineCapacityMagic);appendCapacityWireScalar(bytes,kPipelineCapacityVersion);
    appendCapacityWireScalar(bytes,static_cast<std::uint32_t>(sender));
    appendCapacityWireScalar(bytes,static_cast<std::uint32_t>(local.size()));
    std::set<int> seen;
    for(const auto& stage:local) {
        if(!std::binary_search(expected.begin(),expected.end(),stage.stage_id) || !seen.insert(stage.stage_id).second)
            throw std::invalid_argument("Pipeline capacity sender changed its original stage membership");
        for(const auto& budget:stage.budgets)
            if(budget.worldRank()!=sender || budget.resourceId()!=
                MoEOverlayLocalCapacityPlanner::physicalResourceId(sender,budget.device()))
                throw std::invalid_argument("Pipeline capacity contribution has a foreign original allocator");
        const auto body=serializeOverlayCapacityBudgets(stage.budgets);
        if(body.empty() || body.size()>static_cast<size_t>(INT_MAX)-12 || bytes.size()>static_cast<size_t>(INT_MAX)-12-body.size())
            throw std::overflow_error("Pipeline capacity frame exceeds its exact original MPI extent");
        appendCapacityWireScalar(bytes,static_cast<std::uint32_t>(stage.stage_id));
        appendCapacityWireScalar(bytes,static_cast<std::uint64_t>(body.size()));
        bytes.insert(bytes.end(),body.begin(),body.end());
    }
    // Decode through the same original receiver before readiness can succeed.
    (void)parseOriginalPipelineCapacityFrame(bytes,sender,topology);
    return {std::move(bytes),sender};
}
std::vector<OriginalPipelineStageBudgets> parseOriginalPipelineCapacityFrame(
    std::span<const std::uint8_t> bytes,int sender,const GlobalPPTopology& topology) {
    const auto expected=originalLocalStageIDs(topology,sender);
    if(bytes.size()>static_cast<size_t>(INT_MAX))
        throw std::invalid_argument("Pipeline capacity frame exceeds its exact MPI extent");
    size_t cursor=0;
    if(readCapacityWireScalar<std::uint32_t>(bytes,cursor,bytes.size())!=kPipelineCapacityMagic ||
        readCapacityWireScalar<std::uint32_t>(bytes,cursor,bytes.size())!=kPipelineCapacityVersion ||
        readCapacityWireScalar<std::uint32_t>(bytes,cursor,bytes.size())!=static_cast<std::uint32_t>(sender) ||
        readCapacityWireScalar<std::uint32_t>(bytes,cursor,bytes.size())!=expected.size())
        throw std::invalid_argument("Pipeline capacity frame changed original sender/stage framing");
    std::vector<OriginalPipelineStageBudgets> result;
    for(const auto& stage:topology.stages)result.push_back({stage.stage_id,{}});
    std::set<int> seen;
    for(size_t count=0;count<expected.size();++count) {
        const auto id=readCapacityWireScalar<std::uint32_t>(bytes,cursor,bytes.size());
        const auto extent=readCapacityWireScalar<std::uint64_t>(bytes,cursor,bytes.size());
        if(id>static_cast<std::uint32_t>(INT_MAX) || !std::binary_search(expected.begin(),expected.end(),static_cast<int>(id)) ||
            !seen.insert(static_cast<int>(id)).second || extent==0 || extent>bytes.size()-cursor)
            throw std::invalid_argument("Pipeline CAP4 segment changed original stage identity or extent");
        const auto stage=std::find_if(result.begin(),result.end(),[&](const auto& entry){return entry.stage_id==static_cast<int>(id);});
        const auto end=cursor+static_cast<size_t>(extent);
        parseOverlayCapacityBudgetSegment(bytes,cursor,end,sender,stage->budgets);cursor=end;
        std::set<DeviceId> allocators;
        for(const auto& budget:stage->budgets)
            if(!allocators.insert(budget.device()).second)
                throw std::invalid_argument("Pipeline CAP4 stage duplicated an original allocator contribution");
    }
    if(cursor!=bytes.size())throw std::invalid_argument("Pipeline capacity frame contains trailing/omitted stage metadata");
    return result;
}
std::vector<OriginalPipelineStageBudgets> gatherOriginalPipelineCapacityBudgets(
    const OriginalPipelineCapacitySendFrame& frame,const std::shared_ptr<IMPIContext>& original,
    const GlobalPPTopology& topology) {
    if(!original || original->communicator()==MPI_COMM_NULL || original->world_size()!=topology.world_size ||
        frame.senderRank()!=original->rank())
        throw std::invalid_argument("Pipeline capacity exchange lost its same original parent authority");
    const int count=static_cast<int>(frame.bytes().size()),population=original->world_size();
    std::vector<int> counts(population),offsets(population);
    original->allgather_bytes(&count,counts.data(),sizeof(count));
    size_t total=0;
    for(int rank=0;rank<population;++rank) {
        if(counts[rank]<0 || static_cast<size_t>(counts[rank])>static_cast<size_t>(INT_MAX)-total)
            throw std::overflow_error("Pipeline original-parent CAP4 aggregate exceeds MPI extent");
        offsets[rank]=static_cast<int>(total);total+=static_cast<size_t>(counts[rank]);
    }
    std::vector<std::uint8_t> received(total);
    original->allgatherv_bytes(frame.bytes().data(),count,received.data(),counts.data(),offsets.data());
    std::vector<OriginalPipelineStageBudgets> result;
    for(const auto& stage:topology.stages)result.push_back({stage.stage_id,{}});
    for(int rank=0;rank<population;++rank) {
        auto contribution=parseOriginalPipelineCapacityFrame(std::span<const std::uint8_t>(received).subspan(
            static_cast<size_t>(offsets[rank]),static_cast<size_t>(counts[rank])),rank,topology);
        for(size_t stage=0;stage<result.size();++stage)
            result[stage].budgets.insert(result[stage].budgets.end(),
                std::make_move_iterator(contribution[stage].budgets.begin()),std::make_move_iterator(contribution[stage].budgets.end()));
    }
    return result;
}
}
