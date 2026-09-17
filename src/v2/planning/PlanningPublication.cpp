/**
 * @file PlanningPublication.cpp
 * @brief Shared failure-atomic metadata and selected-plan publication.
 *
 * Each fallible local step terminates in the existing initialization consensus.
 * MPI owns the rank namespace; immutable configuration bytes use the existing
 * strict codec. Root selection, receive allocation and decoding cannot strand
 * a peer in the next broadcast, and publication never selects a substitute plan.
 */
#include "planning/PlanningPublication.h"
#include "planning/PlanningExpertSample.h"
#include "planning/PlanningMatrixSample.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/TensorClasses.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "config/OrchestrationConfigDocument.h"
#include "execution/runner/RankInitializationLifecycle.h"
#include "interfaces/IMPIContext.h"
#include <array>
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace llaminar2
{
    namespace
    {
        using ArtifactPhases = std::array<std::string_view, 3>;

        /** @return Known artifact phases, or absence to reject inside collective consensus. */
        std::optional<ArtifactPhases> phasesFor(PlanningArtifact artifact)
        {
            switch (artifact)
            {
            case PlanningArtifact::AutomaticRequest:
                return ArtifactPhases{"planning-request.describe", "planning-request.receive-storage", "planning-request.validate"};
            case PlanningArtifact::ModelMetadata:
                return ArtifactPhases{"planning-model.read", "planning-model.receive-storage", "planning-model.validate"};
            case PlanningArtifact::SelectedOrchestration:
                return ArtifactPhases{"planning-selection.select", "planning-selection.receive-storage", "planning-selection.validate"};
            case PlanningArtifact::TransferSamples:
                return ArtifactPhases{"planning-transfers.describe", "planning-transfers.receive-storage", "planning-transfers.prepare"};
            case PlanningArtifact::ExpertSample:
                return ArtifactPhases{"planning-expert.describe", "planning-expert.receive-description", "planning-expert.validate-description"};
            case PlanningArtifact::MatrixSample:
                return ArtifactPhases{"planning-matrix.describe", "planning-matrix.receive-description", "planning-matrix.validate-description"};
            }
            // Invalid enum state still reaches the first common consensus.
            // Throwing while choosing phase names could leave a valid peer waiting.
            return std::nullopt;
        }

        /** @brief Fail on a transport error, never decode partial planning bytes. */
        void requireMPI(int code, const char *operation)
        {
            if (code != MPI_SUCCESS)
                throw std::runtime_error(std::string("Planning publication ") + operation + " failed");
        }

        /**
         * @brief One checked communicator and phase-consensus adapter for planning.
         *
         * Broadcast and scatter/gather share identity and failure semantics.
         * Actual MPI membership, never wrapper hints, sizes payload directories.
         */
        class PlanningExchange final
        {
        public:
            /** @brief Borrow the exact context; process-local work never enters MPI. */
            explicit PlanningExchange(const std::shared_ptr<IMPIContext> &context) : context_(context)
            {
                if (!context_) return;
                comm_ = context_->communicator();
                if (comm_ == MPI_COMM_NULL)
                    throw std::invalid_argument("Planning publication requires a live admission communicator");
                requireMPI(MPI_Comm_rank(comm_, &rank_), "rank query");
                requireMPI(MPI_Comm_size(comm_, &count_), "size query");
            }
            /** @return Authenticated communicator rank, or zero for local work. */
            int rank() const noexcept { return rank_; }
            /** @return Actual discovery membership, including ranks not ultimately selected. */
            int count() const noexcept { return count_; }
            /** @return Borrowed MPI handle, used only when the caller supplied a context. */
            MPI_Comm comm() const noexcept { return comm_; }
            /** @brief Check wrapper claims inside the first collective failure phase. */
            void authenticate() const
            {
                if (context_ && (context_->rank() != rank_ || context_->world_size() != count_ ||
                                 context_->is_root() != (rank_ == 0)))
                    throw std::invalid_argument("Planning context identity disagrees with its communicator");
            }
            /**
             * @brief Finish a fallible local operation in the existing rank consensus.
             * @param index Ordered phase ID within this complete transaction.
             * @param name Static operation identity; a mismatched protocol is fatal.
             * @param work Local work only; callbacks may not nest MPI collectives.
             */
            template<class Work>
            void phase(unsigned index, std::string_view name, Work &&work) const
            {
                const auto result = RankInitializationLifecycle::execute({index, name}, [&] {
                    work();
                    return true;
                }, [&](auto identity, auto outcome) {
                    if (context_) return MPIRankInitializationConsensus::reach(comm_, identity, outcome);
                    return RankInitializationConsensusResult{
                        outcome == RankInitializationLocalOutcome::Succeeded ?
                            RankInitializationConsensusOutcome::AllRanksSucceeded :
                            RankInitializationConsensusOutcome::AtLeastOneRankFailed, {}};
                });
                if (!result.succeeded()) throw std::runtime_error(result.diagnostic("Planning publication"));
            }
        private:
            const std::shared_ptr<IMPIContext> &context_;
            MPI_Comm comm_ = MPI_COMM_NULL;
            int rank_ = 0;
            int count_ = 1;
        };

        /**
         * @brief Complete a fixed batch of already validated final native source buffers.
         * @param exchange Exact authenticated discovery communicator and rank.
         * @param addresses Addresses resolved in allocation consensus, before posting anything.
         * @param counts Validated positive MPI byte counts matching the final tensor allocations.
         *
         * Expert triplets and ordinary matrices share one transport lifecycle.
         * No fallible tensor access or allocation occurs with posted requests;
         * a transport failure aborts without unwinding live receive storage.
         */
        template<size_t Count>
        void publishNativePayload(const PlanningExchange &exchange,
            const std::array<void *, Count> &addresses, const std::array<int, Count> &counts)
        {
            if (exchange.comm() == MPI_COMM_NULL) return; // Explicit process-local publication.
            std::array<MPI_Request, Count> requests;
            requests.fill(MPI_REQUEST_NULL);
            const auto fail = [&](const char *operation) {
                std::fprintf(stderr, "Planning native payload rank=%d: %s\n", exchange.rank(), operation);
                MPI_Abort(exchange.comm(), MPI_ERR_OTHER);
                std::abort();
            };
            for (size_t i = 0; i < Count; ++i)
                if (MPI_Ibcast(addresses[i], counts[i], MPI_BYTE, 0, exchange.comm(), &requests[i]) != MPI_SUCCESS)
                    fail("failed to post native payload broadcast");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
                collective_timeout_policy::kDefaultCollectiveTimeoutMs);
            int completed = 0;
            while (!completed)
            {
                if (MPI_Testall(static_cast<int>(Count), requests.data(), &completed, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
                    fail("native payload broadcast failed");
                if (!completed && std::chrono::steady_clock::now() >= deadline)
                    fail("native payload broadcast exceeded collective deadline");
            }
        }

        /** @return Exact MPI byte count; an empty envelope cannot attest completed work. */
        int payloadSize(size_t size)
        {
            if (size == 0 || size > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Planning envelope is empty or exceeds MPI payload capacity");
            return static_cast<int>(size);
        }

        /**
         * @brief Compute checked vector-collective displacements without wrapping.
         * @param counts One positive envelope size per discovery rank.
         * @param offsets Preallocated output directory with the same membership.
         * @return Aggregate byte count, representable by MPI's count/displacement ABI.
         */
        int payloadOffsets(std::span<const int> counts, std::span<int> offsets)
        {
            if (counts.empty() || counts.size() != offsets.size())
                throw std::invalid_argument("Planning payload directories have inconsistent membership");
            int total = 0;
            for (size_t index = 0; index < counts.size(); ++index)
            {
                if (counts[index] <= 0 || counts[index] > std::numeric_limits<int>::max() - total)
                    throw std::invalid_argument("Planning aggregate exceeds MPI payload capacity");
                offsets[index] = total;
                total += counts[index];
            }
            return total;
        }

        /** @brief Require an executable selection, not another unresolved search request. */
        void validateSelection(const OrchestrationConfig &config, int discovery_size)
        {
            if (!std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(config)) ||
                !config.execution_rank_selection || config.model_path.empty())
                throw std::invalid_argument("Selected orchestration requires a model, apply intent and explicit discovery selection");
            if (config.mpi_procs != discovery_size)
                throw std::invalid_argument("Selected orchestration lost its discovery process count");
            for (const int rank : config.execution_rank_selection->discoveryRanks())
                if (rank >= discovery_size)
                    throw std::invalid_argument("Selected orchestration names an absent discovery rank");
            const auto errors = config.validate();
            if (!errors.empty())
                throw std::invalid_argument("Selected orchestration is invalid: " + errors.front());
        }
    }

    void exchangePlanningArtifact(const std::shared_ptr<IMPIContext> &mpi,
        PlanningArtifact artifact, const std::function<std::vector<uint8_t>()> &produce,
        const std::function<void(std::span<const uint8_t>)> &accept)
    {
        const PlanningExchange exchange(mpi);
        const int rank = exchange.rank();
        const auto names = phasesFor(artifact);
        const auto phase = [&](unsigned index, auto &&work) {
            exchange.phase(index + 1, names ? (*names)[index] : "planning-invalid", work);
        };

        std::vector<uint8_t> payload;
        int size = 0;
        phase(0, [&] {
            exchange.authenticate();
            // Registration and validation are one decision; adding an artifact
            // cannot leave a second enum allowlist out of sync.
            if (!names)
                throw std::invalid_argument("Unknown planning artifact");
            if (!accept || (rank == 0 && !produce))
                throw std::invalid_argument("Planning publication requires a root producer and every rank's decoder");
            if (rank != 0) return;
            payload = produce();
            size = payloadSize(payload.size());
        });
        if (mpi) requireMPI(MPI_Bcast(&size, 1, MPI_INT, 0, exchange.comm()), "size broadcast");
        phase(1, [&] {
            if (size <= 0) throw std::invalid_argument("Planning artifact size is not positive");
            if (rank != 0) payload.resize(static_cast<size_t>(size));
        });
        if (mpi) requireMPI(MPI_Bcast(payload.data(), size, MPI_BYTE, 0, exchange.comm()), "payload broadcast");
        // Even root decodes its own serialization. A producer's C++ value is
        // not proof that the exact wire value can be applied on another rank.
        phase(2, [&] { accept(payload); });
    }

    void acceptPlanningCostPreparation(const std::shared_ptr<IMPIContext> &mpi,
        const std::function<void()> &validate)
    {
        const PlanningExchange exchange(mpi);
        exchange.phase(1, "planning-cost.ready", [&] {
            exchange.authenticate();
            if (!validate) throw std::invalid_argument("Planning cost preparation requires local validation");
            validate();
        });
    }

    void exchangePlanningSamples(const std::shared_ptr<IMPIContext> &mpi,
        const std::function<std::vector<std::vector<uint8_t>>(int)> &distribute,
        const std::function<std::vector<uint8_t>(std::span<const uint8_t>)> &execute,
        const std::function<void(std::span<const RankPlanningSample>)> &accept)
    {
        const PlanningExchange exchange(mpi);
        const bool root = exchange.rank() == 0;
        std::vector<int> counts, offsets;
        std::vector<uint8_t> outgoing, request, observation, incoming;
        int request_size = 0, observation_size = 0;
        exchange.phase(1, "planning-samples.describe", [&] {
            exchange.authenticate();
            if (!execute || (root && (!distribute || !accept)))
                throw std::invalid_argument("Planning samples require a root plan/validator and every rank's sampler");
            if (!root) return;
            auto requests = distribute(exchange.count());
            if (requests.size() != static_cast<size_t>(exchange.count()))
                throw std::invalid_argument("Planning sample requests omit or invent discovery ranks");
            counts.resize(requests.size());
            offsets.resize(requests.size());
            for (size_t index = 0; index < requests.size(); ++index)
                counts[index] = payloadSize(requests[index].size());
            outgoing.resize(payloadOffsets(counts, offsets));
            for (size_t index = 0; index < requests.size(); ++index)
                std::copy(requests[index].begin(), requests[index].end(), outgoing.begin() + offsets[index]);
            request_size = counts.front();
        });
        if (mpi) requireMPI(MPI_Scatter(counts.data(), 1, MPI_INT,
            &request_size, 1, MPI_INT, 0, exchange.comm()), "sample request-size scatter");
        exchange.phase(2, "planning-samples.receive-request", [&] {
            if (request_size <= 0) throw std::invalid_argument("Planning sample request is empty");
            request.resize(static_cast<size_t>(request_size));
        });
        if (mpi) requireMPI(MPI_Scatterv(outgoing.data(), counts.data(), offsets.data(), MPI_BYTE,
            request.data(), request_size, MPI_BYTE, 0, exchange.comm()), "sample request scatter");
        else std::copy(outgoing.begin(), outgoing.end(), request.begin());
        exchange.phase(3, "planning-samples.execute", [&] {
            // Results are sealed by the local owner after its work completes.
            // Failed preparation/capture/events cannot become a zero rate.
            observation = execute(request);
            observation_size = payloadSize(observation.size());
        });
        if (mpi) requireMPI(MPI_Gather(&observation_size, 1, MPI_INT,
            counts.data(), 1, MPI_INT, 0, exchange.comm()), "sample result-size gather");
        else counts.front() = observation_size;
        exchange.phase(4, "planning-samples.receive-results", [&] {
            if (root) incoming.resize(payloadOffsets(counts, offsets));
        });
        if (mpi) requireMPI(MPI_Gatherv(observation.data(), observation_size, MPI_BYTE,
            incoming.data(), counts.data(), offsets.data(), MPI_BYTE, 0, exchange.comm()), "sample result gather");
        else std::copy(observation.begin(), observation.end(), incoming.begin());
        exchange.phase(5, "planning-samples.validate", [&] {
            if (!root) return;
            std::vector<RankPlanningSample> samples;
            samples.reserve(static_cast<size_t>(exchange.count()));
            for (int rank = 0; rank < exchange.count(); ++rank)
                samples.push_back({rank, {incoming.data() + offsets[rank], static_cast<size_t>(counts[rank])}});
            accept(samples);
        });
    }

    PlanningExpertSamplePlan PlanningExpertSamplePublication::describe(const std::shared_ptr<IMPIContext> &mpi,
        const std::function<PlanningExpertSamplePlan()> &root_describe)
    {
        std::optional<PlanningExpertSamplePlan> plan;
        exchangePlanningArtifact(mpi, PlanningArtifact::ExpertSample,
            [&] { return root_describe().serialize(); },
            [&](auto bytes) { plan.emplace(PlanningExpertSamplePlan::deserialize(bytes)); });
        return std::move(*plan);
    }

    PlanningLoadedExpertSample PlanningExpertSamplePublication::publish(const std::shared_ptr<IMPIContext> &mpi,
        const PlanningExpertSamplePlan &plan, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
        DeviceId host_device, const std::function<PlanningLoadedExpertSample()> &root_load)
    {
        const PlanningExchange exchange(mpi);
        // Authentication precedes allocation and traffic. Even a same-size but
        // different source plan on one rank must fail collectively, not broadcast
        // bytes into a compatible-looking tensor and mislabel the measurement.
        exchangePlanningArtifact(mpi, PlanningArtifact::ExpertSample,
            [&] { return plan.serialize(); }, [&](auto bytes) {
                if (plan.serialize() != std::vector<uint8_t>(bytes.begin(), bytes.end()))
                    throw std::invalid_argument("Planning expert source plans disagree across ranks");
                if (!memory || !host_device.is_cpu())
                    throw std::invalid_argument("Planning expert publication requires admitted CPU storage");
                for (const auto &geometry : plan.description().matrices) payloadSize(geometry.source_bytes);
            });
        std::optional<PlanningLoadedExpertSample> loaded;
        std::array<void *, 3> receive_addresses{};
        std::array<int, 3> receive_counts{};
        exchange.phase(1, "planning-expert.allocate-payload", [&] {
            if (exchange.rank() == 0)
            {
                if (!root_load) throw std::invalid_argument("Planning expert publication omits root loader");
                loaded.emplace(root_load());
                if (loaded->plan().serialize() != plan.serialize())
                    throw std::invalid_argument("Planning expert root payload disagrees with published plan");
            }
            else
            {
                const auto allocate = [&](size_t i) {
                    return PlanningLoadedModelSample::allocateReceive(plan.tensorType(i),
                        plan.description().matrices[i], memory, host_device);
                };
                loaded = PlanningLoadedExpertSample(plan, {allocate(0), allocate(1), allocate(2)});
            }
            for (size_t i = 0; i < receive_addresses.size(); ++i)
            {
                const auto &geometry = plan.description().matrices[i];
                if (loaded->tensor(i).native_type() != plan.tensorType(i) ||
                    loaded->tensor(i).size_bytes() != geometry.source_bytes ||
                    loaded->tensor(i).shape() != std::vector<size_t>{geometry.n, geometry.k})
                    throw std::runtime_error("Planning expert publication changed native tensor identity");
                // Resolve every potentially throwing tensor accessor before
                // posting any request. No accessor may unwind live MPI buffers.
                receive_addresses[i] = loaded->matrices_[i].receiveData();
                receive_counts[i] = payloadSize(geometry.source_bytes);
                if (!receive_addresses[i]) throw std::runtime_error("Planning expert payload address is null");
            }
        });
        publishNativePayload(exchange, receive_addresses, receive_counts);
        // Local MPI completion alone does not certify that every peer has
        // completed. Publication becomes externally visible only after consensus.
        exchange.phase(2, "planning-expert.seal-payload", [] {});
        return std::move(*loaded);
    }

    PlanningMatrixSamplePlan PlanningMatrixSamplePublication::describe(const std::shared_ptr<IMPIContext> &mpi,
        const std::function<PlanningMatrixSamplePlan()> &root_describe)
    {
        std::optional<PlanningMatrixSamplePlan> plan;
        exchangePlanningArtifact(mpi, PlanningArtifact::MatrixSample,
            [&] { return root_describe().serialize(); },
            [&](auto bytes) { plan.emplace(PlanningMatrixSamplePlan::deserialize(bytes)); });
        return std::move(*plan);
    }

    PlanningLoadedMatrixSample PlanningMatrixSamplePublication::publish(const std::shared_ptr<IMPIContext> &mpi,
        const PlanningMatrixSamplePlan &plan, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
        DeviceId host_device, const std::function<PlanningLoadedMatrixSample()> &root_load)
    {
        const PlanningExchange exchange(mpi);
        exchangePlanningArtifact(mpi, PlanningArtifact::MatrixSample,
            [&] { return plan.serialize(); }, [&](auto bytes) {
                if (plan.serialize() != std::vector<uint8_t>(bytes.begin(), bytes.end()))
                    throw std::invalid_argument("Planning matrix source plans disagree across ranks");
                if (!memory || !host_device.is_cpu())
                    throw std::invalid_argument("Planning matrix publication requires admitted CPU storage");
                payloadSize(plan.geometry().source_bytes);
            });
        std::optional<PlanningLoadedMatrixSample> loaded;
        std::array<void *, 1> addresses{};
        std::array<int, 1> counts{};
        exchange.phase(1, "planning-matrix.allocate-payload", [&] {
            if (exchange.rank() == 0)
            {
                if (!root_load) throw std::invalid_argument("Planning matrix publication omits root loader");
                loaded.emplace(root_load());
                if (loaded->plan().serialize() != plan.serialize())
                    throw std::invalid_argument("Planning matrix root payload disagrees with published plan");
            }
            else loaded = PlanningLoadedMatrixSample(plan, PlanningLoadedModelSample::allocateReceive(
                plan.tensorType(), plan.geometry(), memory, host_device));
            const auto &geometry = plan.geometry();
            if (loaded->tensor().native_type() != plan.tensorType() ||
                loaded->tensor().size_bytes() != geometry.source_bytes ||
                loaded->tensor().shape() != std::vector<size_t>{geometry.n, geometry.k})
                throw std::runtime_error("Planning matrix publication changed native tensor identity");
            addresses[0] = loaded->sample_.receiveData();
            counts[0] = payloadSize(geometry.source_bytes);
            if (!addresses[0]) throw std::runtime_error("Planning matrix payload address is null");
        });
        publishNativePayload(exchange, addresses, counts);
        exchange.phase(2, "planning-matrix.seal-payload", [] {});
        return std::move(*loaded);
    }

    OrchestrationConfig exchangeSelectedOrchestration(const std::shared_ptr<IMPIContext> &mpi,
        const std::function<OrchestrationConfig()> &select)
    {
        std::optional<OrchestrationConfig> selected;
        exchangePlanningArtifact(mpi, PlanningArtifact::SelectedOrchestration, [&] {
            const auto document = serializeOrchestrationConfig(select());
            return std::vector<uint8_t>(document.begin(), document.end());
        }, [&](std::span<const uint8_t> bytes) {
            auto config = deserializeOrchestrationConfig(std::string_view(
                reinterpret_cast<const char *>(bytes.data()), bytes.size()));
            validateSelection(config, mpi ? mpi->world_size() : 1);
            selected = std::move(config);
        });
        return std::move(*selected);
    }
}
