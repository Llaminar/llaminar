/**
 * @file PooledImmutableLocalHotGraph.h
 * @brief Separate model-ready and auxiliary retirement endpoints for combined local/shared lookup.
 *
 * LocalRead is the one ordinary arena output writer. Neither private native
 * response arrival nor auxiliary cache maintenance substitutes for that writer.
 * These cold node observations grant no source/transport/readiness authority.
 */
#pragma once
#include "execution/local_execution/graph/IRetainedGraphParticipants.h"
#include <optional>
#include "execution/mtp/DeviceGenerationControlBinding.h"
namespace llaminar2 {
class IRetainedGraphStorageBoundary;
namespace qwen4exp {class PooledImmutableDirectedTransport;class PooledImmutableLocalHotPreparedStorage;class IPooledImmutableLocalHotKernels;class PooledImmutableLocalHotPoolQuery;}
/** @brief Require actual typed original local/transport stages and complete same-source auxiliary ownership. */
void requirePooledLocalHotAuxiliary(const ComputeGraph&,const RetainedGraphParticipantAuxiliaryRange&,
    const std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport>&,
    const std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>&);
/** @brief Authenticate an actual closed compact Begin and its exact source union-seal dependency. */
void requirePooledCompactQueryProducer(const ComputeGraph&,const qwen4exp::PooledImmutableLocalHotPoolQuery&,const std::string& beginName);
/** @brief Complete actual local roster for native deliveries; neither geometry nor these observations issue readiness. */
struct PooledImmutableCompactReadRequest final {
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport> transport;
    std::vector<std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>> consumers;
    std::shared_ptr<const qwen4exp::IPooledImmutableLocalHotKernels> source_code;
};
/** @brief Actual reader's compact native delivery acquisition and original fixed ACK frontier. */
struct PooledImmutableCompactConsumerCompletion final {
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport> transport;
    std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage> local;
    std::string requested,header_received,payload_received,received,accepted,ack_sent;
};
/** @brief Authenticate actual original request/result/receipt/LocalAcceptResult/ACK stages. */
void requirePooledCompactConsumerCompletion(const ComputeGraph&,const PooledImmutableCompactConsumerCompletion&);
/** @return Unique authenticated original compact reader frontier in this actual participant graph. */
PooledImmutableCompactConsumerCompletion findPooledCompactConsumerCompletion(const ComputeGraph&,
    const std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport>&,
    const std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>&);
/** @brief Actual source-native pack, complete payload/ACK and last reader observations. */
struct PooledImmutableCompactReadCompletion final {
    PooledImmutableCompactReadRequest original;
    std::string native_rows_ready,terminal;
    std::vector<std::string> packed,header_sent,payload_sent,ack_received,ack_validated;
};
/** @brief Append actual compact native source deliveries behind the original sealed read image. */
std::shared_ptr<const PooledImmutableCompactReadCompletion> appendPooledCompactNativeDelivery(
    ComputeGraph&,BufferArena&,PooledImmutableCompactReadRequest,DeviceGenerationControlBinding,
    const std::string& nativeRowsReady,const std::string& identity);
/** @brief Require every actual pack, native payload receipt and ACK before original pool retirement. */
void requirePooledCompactNativeDelivery(const ComputeGraph&,const PooledImmutableCompactReadCompletion&);
/** @brief Actual combined-mode current-demand handoff; never a PureReader/source alias. */
struct PooledImmutableLocalHotDemandFrontier final {
    std::string published,consume,terminal;
    DeviceId participant=DeviceId::invalid();
};
/** @brief Authenticate real Probe/mapped-ready wait/LocalRead stage roots and static causality. */
void validatePooledLocalHotDemandFrontier(const ComputeGraph&,const PooledImmutableLocalHotDemandFrontier&);
/** @brief Add only model-trunk→main readywait dependencies; never an auxiliary retirement edge. */
void bindPooledLocalHotDemandConsumption(ComputeGraph&,const PooledImmutableLocalHotDemandFrontier&,
    const std::vector<std::string>& trunk);
/** @brief Exact-once cold combined-mode handoff; no source role or runtime permission is minted. */
class PooledImmutableLocalHotDemandHandoff final {
public:
    /** @brief Bind actual local-hot stages/query/hash on the original graph before model construction continues. */
    PooledImmutableLocalHotDemandHandoff(ComputeGraph&,PooledImmutableLocalHotDemandFrontier,
        std::uint64_t query,std::uint64_t hash);
    PooledImmutableLocalHotDemandHandoff(const PooledImmutableLocalHotDemandHandoff&)=delete;
    PooledImmutableLocalHotDemandHandoff& operator=(const PooledImmutableLocalHotDemandHandoff&)=delete;
    /** @return Actual Probe producer, never storage-owner response/retirement. */
    const std::string& producer()const noexcept{return frontier_.published;}
    /** @brief Consume actual same-device trunk once; ancestor/suffix cycles reject before graph mutation. */
    void consume(ComputeGraph&,std::uint64_t query,std::uint64_t hash,const std::vector<std::string>& trunk);
    /** @brief Require the actual cold model-consumer handoff; reset never changes topology. */
    void requireConsumed()const;
private:
    ComputeGraph* graph_;
    std::shared_ptr<IRetainedGraphStorageBoundary> boundary_;
    std::optional<RetainedGraphParticipantDefinitionDomain> domain_;
    PooledImmutableLocalHotDemandFrontier frontier_;
    std::uint64_t query_,hash_;
    bool consumed_=false; ///< Graph construction only, not native progress/readiness.
};
/** @brief Model-ready edges are distinct from the original native auxiliary terminal. */
struct PooledImmutableLocalHotGraphEndpoints final {
    std::string first,last; ///< Probe and actual LocalRead, respectively; never auxiliary retirement.
    std::string probe,ready_wait,read,consumed,retired;
    PooledImmutableLocalHotDemandFrontier demand;
    RetainedGraphParticipantAuxiliaryRange auxiliary;
};
} // namespace llaminar2
