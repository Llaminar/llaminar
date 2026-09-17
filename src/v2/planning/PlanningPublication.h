/**
 * @file PlanningPublication.h
 * @brief One root-owned publication transaction for immutable planning artifacts.
 *
 * Metadata and selected configurations share the same read/allocate/validate
 * consensus. This setup-only boundary neither discovers devices nor owns live
 * execution state. A follower never invokes the root producer. No participant
 * may return a decoded value before every participant accepts the publication.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace llaminar2
{
    class IMPIContext;
    struct OrchestrationConfig;

    /** @brief Protocol identity prevents two different artifacts matching by byte count. */
    enum class PlanningArtifact
    {
        AutomaticRequest,
        ModelMetadata,
        SelectedOrchestration,
        TransferSamples,
        ExpertSample,
        MatrixSample,
    };

    /**
     * @brief Publish and validate one immutable artifact on the exact communicator.
     * @param mpi Admission context; null is an explicitly process-local transaction.
     * @param artifact Typed protocol identity, authenticated by phase consensus.
     * @param produce Root-only serializer; failures precede any payload broadcast.
     * @param accept Rank-local decoder; its failures reach consensus before return.
     * @throws std::runtime_error for a rank-local or protocol failure on any peer.
     *
     * Infrastructure callers retain their decoded value locally until this
     * function succeeds. The callbacks must not perform collectives or publish
     * live state. Communicator size/rank come from MPI, not an unchecked wrapper;
     * a wrapper claiming size one cannot silently take a local-only branch.
     */
    void exchangePlanningArtifact(const std::shared_ptr<IMPIContext> &mpi,
        PlanningArtifact artifact, const std::function<std::vector<uint8_t>()> &produce,
        const std::function<void(std::span<const uint8_t>)> &accept);

    /**
     * @brief Authenticate completion of rank-local automatic-cost preparation.
     * @param mpi Exact discovery context; null denotes explicit process-local work.
     * @param validate Local finalization/validation, with no nested MPI operations.
     * @throws std::runtime_error on any rank's failure before selection can begin.
     *
     * Distributed sample collectors own their internal collective failure phases.
     * After those collectors return, this common boundary admits the completed
     * evaluator and publishes any local construction exception. It is not a
     * wrapper that can make an arbitrary throwing MPI callback failure-atomic.
     */
    void acceptPlanningCostPreparation(const std::shared_ptr<IMPIContext> &mpi,
        const std::function<void()> &validate);

    /**
     * @brief Borrowed immutable observation from one authenticated discovery rank.
     *
     * The transport supplies the rank; it is not taken from untrusted sample
     * metadata or an execution-rank map. Bytes live only during root acceptance.
     */
    struct RankPlanningSample
    {
        int discovery_rank;
        std::span<const uint8_t> bytes;
    };

    /**
     * @brief Scatter bounded sample requests and gather completed observations.
     * @param mpi Exact discovery communicator, before selected ranks are retired.
     * @param distribute Root-only producer of exactly one nonempty envelope per rank.
     *        The argument is authenticated communicator size, not a caller hint.
     * @param execute Rank-local decoder/sampler/serializer, invoked once with this
     *        rank's envelope. Its result must be a nonempty completed observation.
     * @param accept Root-only validator of every rank's immutable result, in rank order.
     * @throws std::runtime_error on any local preparation, execution, validation or MPI failure.
     *
     * Callbacks must not enter MPI collectives or publish live execution state.
     * Native device/CPU work belongs to its ordinary local owner and PMA claims;
     * this boundary owns only control envelopes, not buffers for model execution.
     * An unused rank still exchanges a typed empty-work envelope. Every fallible
     * phase reaches the existing initialization consensus before another
     * collective. Root cannot price a partial set or treat a failed rank as slow.
     * A null context denotes the same one-rank transaction without initializing MPI.
     */
    void exchangePlanningSamples(const std::shared_ptr<IMPIContext> &mpi,
        const std::function<std::vector<std::vector<uint8_t>>(int)> &distribute,
        const std::function<std::vector<uint8_t>(std::span<const uint8_t>)> &execute,
        const std::function<void(std::span<const RankPlanningSample>)> &accept);

    /**
     * @brief Distribute the root's complete explicit selection using the sole config codec.
     * @param mpi Discovery context, before selected execution ranks are split.
     * @param select Root-only selector returning the complete admitted configuration.
     * @return Independent, byte-equivalent configuration on every discovery rank.
     * @throws std::runtime_error for failed selection or invalid publication.
     *
     * The result requires apply intent, an ordered selection within discovery,
     * and the original process count. It is not a new physical-memory grant or
     * hardware certificate. Ordinary runtime admission still validates the
     * actual selected hardware and reserves the production BOM before loading.
     */
    OrchestrationConfig exchangeSelectedOrchestration(const std::shared_ptr<IMPIContext> &mpi,
        const std::function<OrchestrationConfig()> &select);
}
