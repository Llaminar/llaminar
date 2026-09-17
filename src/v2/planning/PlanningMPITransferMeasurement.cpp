/**
 * @file PlanningMPITransferMeasurement.cpp
 * @brief Prepared host MPI sampling with topology-authenticated completed costs.
 *
 * A private communicator owns the measurement packets. PMA claims precede
 * payload allocation; requests retire before buffers or their claims. Root's
 * immutable sample plan is validated on every discovery rank before traffic.
 * Setup failures reach the ordinary publication consensus. Once requests are
 * posted, a failed transport is fatal because MPI still owns their addresses.
 * All timings use one initiator-local steady clock; no clock synchronization,
 * link symmetry, hostname inference, or bandwidth-based locality is assumed.
 */
#include "planning/PlanningMPITransferMeasurement.h"
#include "planning/PlanningExecutionMeasurement.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/PlanningPublication.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "utils/MPIContext.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace llaminar2
{
    namespace
    {
        constexpr std::array<uint8_t, 4> kMagic{'L', 'T', 'M', 1};
        constexpr int kRequestTag = 0; ///< Private communicator, never inference tags.
        constexpr int kReplyTag = 1;

        /** @brief Encode fixed-width integers with explicit little-endian wire order. */
        template<class UInt>
        void append(std::vector<uint8_t> &bytes, UInt value)
        {
            static_assert(std::is_unsigned_v<UInt>);
            for (size_t byte = 0; byte != sizeof(UInt); ++byte)
                bytes.push_back(static_cast<uint8_t>(value >> (8 * byte)));
        }

        /** @brief Consume one checked fixed-width integer, rejecting truncated evidence. */
        template<class UInt>
        UInt take(std::span<const uint8_t> &bytes)
        {
            static_assert(std::is_unsigned_v<UInt>);
            if (bytes.size() < sizeof(UInt)) throw std::invalid_argument("Truncated MPI planning sample");
            UInt value = 0;
            for (size_t byte = 0; byte != sizeof(UInt); ++byte)
                value |= static_cast<UInt>(bytes[byte]) << (8 * byte);
            bytes = bytes.subspan(sizeof(UInt));
            return value;
        }

        /** @brief Require the exact version, including for a valid empty sample batch. */
        void consumeHeader(std::span<const uint8_t> &bytes)
        {
            if (bytes.size() < kMagic.size() || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
                throw std::invalid_argument("Invalid MPI planning sample version");
            bytes = bytes.subspan(kMagic.size());
        }

        /** @return Root's fixed-width metadata; no execution payload crosses publication. */
        std::vector<uint8_t> encodeRequests(std::span<const PlanningMPITransferRequest> requests)
        {
            std::vector<uint8_t> bytes(kMagic.begin(), kMagic.end());
            for (const auto &request : requests)
            {
                append(bytes, static_cast<uint32_t>(request.initiator));
                append(bytes, static_cast<uint32_t>(request.responder));
                append(bytes, static_cast<uint32_t>(request.request_bytes));
                append(bytes, static_cast<uint32_t>(request.reply_bytes));
            }
            return bytes;
        }

        /** @return Checked metadata, preserving endpoint order without hostname interpretation. */
        std::vector<PlanningMPITransferRequest> decodeRequests(std::span<const uint8_t> bytes)
        {
            consumeHeader(bytes);
            if (bytes.size() % 16 != 0) throw std::invalid_argument("Partial MPI planning geometry record");
            std::vector<PlanningMPITransferRequest> requests;
            requests.reserve(bytes.size() / 16);
            while (!bytes.empty())
            {
                const auto initiator = take<uint32_t>(bytes), responder = take<uint32_t>(bytes);
                if (initiator > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                    responder > static_cast<uint32_t>(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("MPI planning endpoint is not representable");
                requests.push_back({static_cast<int>(initiator), static_cast<int>(responder),
                    take<uint32_t>(bytes), take<uint32_t>(bytes)});
            }
            return requests;
        }

        /**
         * @brief Own one rank's exact reusable payload and its preceding PMA claim.
         *
         * Declaration/destruction order keeps the claim alive until after the
         * allocation is freed. No caller-created buffer or unadmitted test-only
         * allocation path is accepted by the production measurement.
         */
        struct SampleWorkspace final
        {
            PhysicalMemoryAllocationLease claim;
            std::unique_ptr<uint8_t[]> bytes;
            /** @brief Claim first, then allocate/first-touch on the admitted rank. */
            SampleWorkspace(PhysicalMemoryAuthority &memory, DeviceId device, size_t size)
                : claim(memory.claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, size)),
                  bytes(std::make_unique<uint8_t[]>(size)) {}
        };

        /** @brief Report before aborting so a dead peer cannot hide the active operation. */
        [[noreturn]] void failTransport(MPI_Comm comm, int rank, const char *operation) noexcept
        {
            std::fprintf(stderr, "[Planning][FATAL][MPI] rank=%d operation=%s; pending request ownership cannot be released\n",
                rank, operation);
            std::fflush(stderr);
            MPI_Abort(comm, EXIT_FAILURE);
            std::abort();
        }

        /** @brief Isolate setup packet tags and retain the communicator until all requests retire. */
        class SampleCommunicator final
        {
        public:
            /** @brief Every discovery rank enters after successful allocation consensus. */
            SampleCommunicator(MPI_Comm parent, int rank) : rank_(rank)
            {
                if (MPI_Comm_dup(parent, &comm_) != MPI_SUCCESS || comm_ == MPI_COMM_NULL)
                    failTransport(parent, rank, "duplicate sample communicator");
                if (MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN) != MPI_SUCCESS)
                    failTransport(comm_, rank, "install sample error handler");
            }
            /** @brief All requests have completed; never free an active packet lane. */
            ~SampleCommunicator()
            {
                if (MPI_Comm_free(&comm_) != MPI_SUCCESS)
                    failTransport(MPI_COMM_WORLD, rank_, "retire sample communicator");
            }
            SampleCommunicator(const SampleCommunicator &) = delete;
            SampleCommunicator &operator=(const SampleCommunicator &) = delete;
            /** @return Borrowed private handle for the ordinary MPIContext interface. */
            MPI_Comm get() const noexcept { return comm_; }
        private:
            MPI_Comm comm_ = MPI_COMM_NULL;
            int rank_;
        };

        /**
         * @brief Retain one asynchronous packet until successful native completion.
         *
         * MPI_Test also drives MPI's normal progress engine. Sleeping between
         * polls changes measured link service on non-offloaded MPI transports.
         * A terminal error aborts; throwing/unwinding a live receive is unsafe.
         */
        class Packet final
        {
        public:
            /** @brief Adopt exactly one submitted request from the canonical MPI interface. */
            Packet(const IMPIContext &context, MPI_Request request) : context_(context), request_(request) {}
            /** @brief Guard against a new exception path abandoning outstanding MPI work. */
            ~Packet()
            {
                if (request_ != MPI_REQUEST_NULL)
                    failTransport(context_.communicator(), context_.rank(), "abandoned sample packet");
            }
            Packet(const Packet &) = delete;
            Packet &operator=(const Packet &) = delete;
            /** @brief Complete within the standard rendezvous deadline; validate receive size. */
            void complete(std::optional<size_t> received_bytes = {})
            {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
                    collective_timeout_policy::kDefaultCollectiveTimeoutMs);
                MPI_Status status{};
                try
                {
                    while (!context_.test(&request_, &status))
                        if (std::chrono::steady_clock::now() >= deadline)
                            failTransport(context_.communicator(), context_.rank(), "sample packet timeout");
                    if (received_bytes && context_.getCount(status, MPI_BYTE) != static_cast<int>(*received_bytes))
                        failTransport(context_.communicator(), context_.rank(), "sample packet size mismatch");
                }
                catch (...)
                {
                    failTransport(context_.communicator(), context_.rank(), "sample packet completion");
                }
            }
        private:
            const IMPIContext &context_;
            MPI_Request request_;
        };

        /** @brief Execute the ordered host request/reply protocol without allocation or barriers. */
        void exchange(const IMPIContext &context, const PlanningMPITransferRequest &request,
            uint8_t *outbound, uint8_t *reply)
        {
            try
            {
                if (context.rank() == request.initiator)
                {
                    Packet receive(context, context.irecv(reply, request.reply_bytes, MPI_BYTE, request.responder, kReplyTag));
                    Packet send(context, context.isend(outbound, request.request_bytes, MPI_BYTE, request.responder, kRequestTag));
                    send.complete();
                    receive.complete(request.reply_bytes);
                }
                else
                {
                    Packet receive(context, context.irecv(outbound, request.request_bytes, MPI_BYTE, request.initiator, kRequestTag));
                    receive.complete(request.request_bytes);
                    // Completion, not submission, authorizes the response.
                    Packet send(context, context.isend(reply, request.reply_bytes, MPI_BYTE, request.initiator, kReplyTag));
                    send.complete();
                }
            }
            catch (...)
            {
                failTransport(context.communicator(), context.rank(), "sample packet submission");
            }
        }
    }

    PlanningMPITransferObservation::PlanningMPITransferObservation(
        PlanningMPITransferRequest request, RankConnectionTopology topology, double seconds)
        : request_(request), topology_(topology), seconds_(seconds)
    {
        if (!std::isfinite(seconds) || seconds <= 0 ||
            topology.sourceRank() != request.initiator || topology.destinationRank() != request.responder)
            throw std::invalid_argument("MPI planning observation needs completed time and matching physical membership");
    }

    size_t PlanningMPITransferMeasurement::workspaceBytes(
        std::span<const PlanningMPITransferRequest> requests, int rank, int world_size)
    {
        if (world_size <= 0 || rank < 0 || rank >= world_size)
            throw std::invalid_argument("MPI planning workspace requires a present discovery rank");
        size_t largest = 0;
        for (const auto &request : requests)
        {
            if (request.initiator < 0 || request.responder < 0 || request.initiator >= world_size ||
                request.responder >= world_size || request.initiator == request.responder ||
                request.request_bytes == 0 || request.reply_bytes == 0 ||
                request.request_bytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                request.reply_bytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                request.request_bytes > std::numeric_limits<size_t>::max() - request.reply_bytes)
                throw std::invalid_argument("MPI planning exchange requires distinct present ranks and positive bounded payloads");
            if (rank == request.initiator || rank == request.responder)
                largest = std::max(largest, request.request_bytes + request.reply_bytes);
        }
        return largest;
    }

    std::vector<PlanningMPITransferObservation> PlanningMPITransferMeasurement::measure(
        const std::shared_ptr<IMPIContext> &mpi, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
        DeviceId host_device, const std::function<std::vector<PlanningMPITransferRequest>()> &describe)
    {
        if (!mpi) throw std::invalid_argument("MPI transfer samples require a live discovery context");
        // Discovery is an existing context-owned publication, not inferred from
        // the timings below. Do not lazily enter it inside a local-phase callback.
        const auto inventory = mpi->clusterInventory();
        std::vector<PlanningMPITransferRequest> requests;
        std::vector<double> seconds;
        std::unique_ptr<SampleWorkspace> workspace;
        exchangePlanningArtifact(mpi, PlanningArtifact::TransferSamples, [&] {
            if (!describe) throw std::invalid_argument("Missing MPI sample description");
            const auto plan = describe();
            (void)workspaceBytes(plan, 0, mpi->world_size());
            return encodeRequests(plan);
        }, [&](auto bytes) {
            requests = decodeRequests(bytes);
            if (!inventory || inventory->world_size != mpi->world_size())
                throw std::invalid_argument("MPI samples require the exact discovery inventory");
            for (const auto &request : requests)
                (void)inventory->connectionBetweenRanks(request.initiator, request.responder);
            const auto size = workspaceBytes(requests, mpi->rank(), mpi->world_size());
            seconds.resize(requests.size());
            if (size == 0) return;
            if (!memory || !host_device.is_cpu() || memory->worldRank() != mpi->rank())
                throw std::invalid_argument("MPI sample workspace lacks its rank-local CPU memory authority");
            workspace = std::make_unique<SampleWorkspace>(*memory, host_device, size);
        });

        // No live packet can precede the all-rank successful preparation vote.
        // All ranks enter this communicator, including nonparticipants in a pair.
        const SampleCommunicator communication(mpi->communicator(), mpi->rank());
        const MPIContext packets(mpi->rank(), mpi->world_size(), communication.get());
        for (size_t index = 0; index != requests.size(); ++index)
        {
            const auto &request = requests[index];
            if (mpi->rank() == request.initiator || mpi->rank() == request.responder)
            {
                auto *outbound = workspace->bytes.get();
                auto *reply = outbound + request.request_bytes;
                std::fill_n(outbound, request.request_bytes, mpi->rank() == request.initiator ? uint8_t{0x35} : uint8_t{0});
                std::fill_n(reply, request.reply_bytes, mpi->rank() == request.responder ? uint8_t{0xa7} : uint8_t{0});
                exchange(packets, request, outbound, reply); // Untimed first use, including page/protocol contact.
                const auto start = std::chrono::steady_clock::now();
                for (int sample = 0; sample != PlanningExecutionMeasurement::kTimedInvocations; ++sample)
                    exchange(packets, request, outbound, reply);
                if (mpi->rank() == request.initiator)
                    seconds[index] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() /
                                     PlanningExecutionMeasurement::kTimedInvocations;
                // Outside timing, retain an error sentinel for the final collective
                // evidence validation. Never throw while another pair is still active.
                if (!std::all_of(outbound, outbound + request.request_bytes, [](auto value) { return value == 0x35; }) ||
                    !std::all_of(reply, reply + request.reply_bytes, [](auto value) { return value == 0xa7; }))
                    seconds[index] = -1;
            }
            // Uninvolved ranks cannot race ahead and start a different pair:
            // that would contaminate uncontended link evidence and make the
            // reusable workspace's concurrency assumption false. This bounded
            // setup-only rendezvous is outside the measured interval.
            MPI_Request rendezvous = MPI_REQUEST_NULL;
            if (MPI_Ibarrier(communication.get(), &rendezvous) != MPI_SUCCESS)
                failTransport(communication.get(), mpi->rank(), "sample pair rendezvous");
            Packet barrier(packets, rendezvous);
            barrier.complete();
        }

        std::vector<PlanningMPITransferObservation> observations;
        exchangePlanningSamples(mpi,
            [](int count) { return std::vector<std::vector<uint8_t>>(static_cast<size_t>(count), {1}); },
            [&](auto) {
                std::vector<uint8_t> bytes(kMagic.begin(), kMagic.end());
                for (const double time : seconds) append(bytes, std::bit_cast<uint64_t>(time));
                return bytes;
            }, [&](auto results) {
                std::vector<double> complete(requests.size());
                for (const auto &result : results)
                {
                    auto bytes = result.bytes;
                    consumeHeader(bytes);
                    if (bytes.size() / sizeof(double) != requests.size() || bytes.size() % sizeof(double) != 0)
                        throw std::invalid_argument("MPI planning evidence omitted or invented requests");
                    for (size_t index = 0; index != requests.size(); ++index)
                    {
                        const auto time = std::bit_cast<double>(take<uint64_t>(bytes));
                        if (result.discovery_rank == requests[index].initiator)
                        {
                            if (!std::isfinite(time) || time <= 0)
                                throw std::invalid_argument("MPI planning initiator did not complete its exchange");
                            complete[index] = time;
                        }
                        else if (time != 0)
                            throw std::invalid_argument("MPI planning non-initiator reported a time or corrupt payload");
                    }
                }
                observations.reserve(requests.size());
                for (size_t index = 0; index != requests.size(); ++index)
                    observations.push_back(PlanningMPITransferObservation(requests[index],
                        inventory->connectionBetweenRanks(requests[index].initiator, requests[index].responder), complete[index]));
            });
        return observations;
    }
}
