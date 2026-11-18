/**
 * @file ResidentMoEFFN.h
 * @brief Typed original-source local FFN region preparation and retained submission.
 *
 * The request names existing public tensors and private table identities; only
 * the owning backend resolves arena addresses. A prepared transaction performs
 * routed/shared local math and publication, never routing or communication.
 * Its original runtime bank remains the sole live source/placement authority.
 */
#pragma once
#include "kernels/common/DeviceRowRange.h"
#include "ResidentMoEFFNProgramEvidence.h"
#include "execution/moe/MoEIntermediateShardDescriptorDeclaration.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/ExpertHistogramSource.h"
#include <memory>
#include <optional>

namespace llaminar2
{
class TensorBase;
class ITensor;

/**
 * @brief Original model placement coordinates for the retained decode receipt.
 * @details Source geometry and physical kernel metadata remain with their existing owners.
 */
struct ResidentMoEFFNModelCoordinates
{
    int source_layer;
    unsigned shared_source_first_column;
    MoERuntimeHistogramProducerRole histogram_producer; ///< Original model router authorization.
};

/** @brief Complete source-authenticated ordinary M1 region after the original router. */
struct ResidentMoEFFNRequest
{
    MoEIntermediateShardDescriptorDeclaration original;
    DeviceMoELayerRuntime* runtime=nullptr;
    TensorBase* hidden=nullptr;
    DeviceRowRange rows=DeviceRowRange::fullyActive(1);
    int gate_up_table=-1,down_table=-1,top_k=0;
    DeviceMoEExpertDescriptor shared;
    TensorBase* shared_input_gate=nullptr;
    ITensor* route_contributions=nullptr;
    ITensor* routed_output=nullptr;
    ITensor* shared_output=nullptr;
    ITensor* combined_output=nullptr;
    ITensor* shared_gate_scratch=nullptr;
    ITensor* shared_up_scratch=nullptr;
    std::optional<ResidentMoEFFNModelCoordinates> model_main_inference;
};

/** @brief Backend-owned sealed local transaction; no public private-buffer accessor. */
class PreparedResidentMoEFFN
{
public:
    /** @brief Retire only after the owning executor retires native graph borrowers. */
    virtual ~PreparedResidentMoEFFN()=default;
    /** @return Exact ordered native launch/resource observations, independent of submission mechanism. */
    virtual const ResidentMoEFFNProgramEvidence& programEvidence() const noexcept=0;
    /** @return Complete semantic pointer/scalar capture identity, without hashing source data. */
    virtual std::uint64_t captureSignature() const noexcept=0;
    /** @brief Reject changed arena/table/runtime bindings before retained replay, without native queries. */
    virtual void validateBinding() const=0;
    /**
     * @brief Enqueue the sealed ordered local transaction on its original prepared stream.
     * @param stream Exact existing graph-stage stream, never null/default.
     * @throws std::logic_error For rebinding or stale workspace generation.
     * @throws std::runtime_error For native submission failure, without another path.
     */
    virtual void enqueue(void* stream) const=0;
};

/** @brief Narrow optional backend feature interface, acquired only by an explicit region declaration. */
class IResidentMoEFFNPreparation
{
public:
    virtual ~IResidentMoEFFNPreparation()=default;
    /**
     * @brief Resolve original table IDs and admitted scratch into one prepared program.
     * @param request Original family, already-authenticated shared source and public arena bindings.
     * @param stream Same non-default stream used by original table preparation and later capture.
     * @return Backend-owned program retaining no host mirror of live routes or model state.
     * @throws std::invalid_argument For an unsupported or incomplete declared source contract.
     * @throws std::logic_error During capture or on changed preparation ownership.
     */
    virtual std::unique_ptr<PreparedResidentMoEFFN> prepareResidentMoEFFN(
        const ResidentMoEFFNRequest& request,void* stream)=0;
};
}
