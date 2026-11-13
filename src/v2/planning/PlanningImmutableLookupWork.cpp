/**
 * @file PlanningImmutableLookupWork.cpp
 * @brief Cold live-row work and explicit measured-service proxies for immutable lookup.
 *
 * Each physical reader/source comes from the admitted placement. Fixed head
 * ownership is authenticated against original head boundaries, including unequal
 * pipeline groups; no traffic is inferred from table bytes or cache quotas.
 * Serial composition is conservative about overlap, not a measured latency
 * guarantee. Source dequantization/random-access latency is an explicit proxy
 * limitation. Unsupported storage service fails before a candidate can win.
 */
#include "PlanningImmutableLookupWork.h"
#include "PlanningWeightServiceModel.h"
#include "PlanningCommunicationCost.h"
#include "AutomaticPlanningStartup.h"
#include "OrchestrationCandidateAdmission.h"
#include "MemoryPlanner.h"
#include "WeightShardGeometry.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "execution/qwen4exp/ImmutableLookupProtocol.h"
#include "kernels/qwen4exp/ImmutableFixedHeadLookupKernels.h"
#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <stdexcept>

namespace llaminar2 {
namespace {
/** @return Checked arithmetic on live payloads; capacity overflow is not a speedup. */
std::size_t multiply(std::size_t a,std::size_t b) {
    if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::overflow_error("Immutable lookup live traffic overflow");
    return a*b;
}
/** @return Checked additive payload and protocol metadata. */
std::size_t add(std::size_t a,std::size_t b) {
    if(a>std::numeric_limits<std::size_t>::max()-b)throw std::overflow_error("Immutable lookup protocol traffic overflow");
    return a+b;
}
/** @return Exact measurement endpoint, preserving original discovery rank. */
PlanningImmutableEndpoint endpoint(PhysicalMemoryAllocatorIdentity owner,std::span<const int> ranks) {
    if(owner.world_rank<0 || static_cast<std::size_t>(owner.world_rank)>=ranks.size() || !owner.valid())
        throw std::invalid_argument("Immutable lookup owner is outside admitted execution membership");
    return {owner.world_rank,ranks[owner.world_rank],owner.device};
}
/** @return Number of complete original heads in one fixed source interval. */
std::size_t headsInRange(const Qwen4ExpMemoryGeometry &g,const ImmutableRowTableSourceRange &range) {
    std::size_t heads=0;
    for(std::size_t i=0;i<g.ple_head_offsets.size();++i) {
        const auto first=g.ple_head_offsets[i],count=g.ple_head_vocab_sizes[i];
        const auto end=first+count,range_end=range.firstRow()+range.rows();
        if(first<range_end && end>range.firstRow()) {
            if(first<range.firstRow() || end>range_end)throw std::invalid_argument("Fixed immutable range cuts an original head");
            ++heads;
        }
    }
    if(!heads)throw std::invalid_argument("Fixed immutable owner has no original query heads");
    return heads;
}
/** @brief Charge one live request/response between actual source and requester. */
void exchange(PlanningImmutableLookupWork &out,PlanningImmutableEndpoint owner,
              PlanningImmutableEndpoint reader,std::size_t queries,std::size_t native_bytes) {
    if(owner.execution_rank!=reader.execution_rank)throw std::invalid_argument("Immutable lookup cross-rank protocol has no declared planning service");
    if(owner.device==reader.device)return;
    if(owner.device.is_cpu()) {
        if(!reader.device.is_gpu())throw std::invalid_argument("Immutable CPU-to-CPU lookup lacks explicit service ownership");
        out.channels.push_back({PlanningImmutableChannelKind::HostRequestReply,owner,reader,{},
            add(multiply(queries,sizeof(std::uint64_t)),sizeof(qwen4exp::ImmutableLookupRequest)),
            add(native_bytes,add(sizeof(qwen4exp::ImmutableLookupCompletion),sizeof(qwen4exp::ImmutableLookupAcknowledgement)))});
    } else {
        if(!reader.device.is_gpu() || owner.device.type!=reader.device.type)
            throw std::invalid_argument("Immutable native lookup requires same-vendor GPU endpoints");
        out.channels.push_back({PlanningImmutableChannelKind::NativeRequestReply,owner,reader,{owner.device,reader.device},
            multiply(queries,sizeof(std::uint64_t)),native_bytes});
    }
}
}

PlanningImmutableLookupVolume PlanningImmutableLookupVolume::forTokenHeads(std::size_t rows,std::size_t batch,
    std::size_t heads,std::size_t native_row_bytes,std::size_t row_elements) {
    if(!batch || !heads || !native_row_bytes || !row_elements)throw std::invalid_argument("Immutable lookup live volume has empty source geometry");
    const auto queries=multiply(multiply(rows,batch),heads);
    return {queries,multiply(queries,native_row_bytes),multiply(multiply(queries,row_elements),sizeof(float))};
}

PlanningImmutableLookupWork compilePlanningImmutableLookupWork(const ModelMemoryProfile &profile,
    std::span<const DevicePlanConfig> devices,std::span<const int> ranks,std::size_t rows) {
    PlanningImmutableLookupWork out;
    const bool table=std::any_of(profile.tensors.begin(),profile.tensors.end(),[](const auto &t){return inferWeightRole(t.name)==WeightRole::ImmutableRowTable;});
    if(!table)return out;
    if(!profile.flash_next)throw std::invalid_argument("Immutable lookup source has no typed head geometry");
    const auto &g=*profile.flash_next;
    if(!g.ple_heads || !g.ple_row_elements || !g.ple_table_row_bytes ||
       g.ple_head_offsets.size()!=g.ple_heads || g.ple_head_vocab_sizes.size()!=g.ple_heads)
        throw std::invalid_argument("Immutable lookup has incomplete original row/head metadata");
    std::set<const ResolvedImmutableRowTablePlacement*> fixed_done;
    bool consumed=false;
    for(const auto &device:devices) {
        const int last=device.last_layer<0?profile.mainLayerCount()-1:device.last_layer;
        const bool owns=device.execution_role==DeviceExecutionMemoryRole::ContinuationGraph &&
            device.first_layer<=static_cast<int>(g.ple_decoder_layer) && last>=static_cast<int>(g.ple_decoder_layer);
        if(!owns)continue;
        consumed=true;
        const auto &placement=device.ngram_table_placement;
        if(!rows && (!placement || !placement->isFixedHeadShards()))continue;
        if(!placement)throw std::invalid_argument("Immutable lookup cost requires final admitted source placement");
        if(placement->extent().rowBytes()!=g.ple_table_row_bytes || placement->extent().rows()!=g.ple_table_rows ||
           placement->extent().nativeBytes()!=g.ple_table_native_bytes ||
           static_cast<std::uint32_t>(placement->extent().format())!=g.ple_table_gguf_type)
            throw std::invalid_argument("Immutable lookup placement changed original native row geometry");
        if(device.ngram_execution_policy)placement->requireRequestedPolicy(*device.ngram_execution_policy);
        const PhysicalMemoryAllocatorIdentity identity{device.world_rank,device.device};
        const auto mode=placement->lookupForConsumer(identity);
        const auto reader=endpoint(identity,ranks);
        if(device.batch_size<=0)throw std::invalid_argument("Immutable lookup requires a positive admitted request batch");
        const auto volume=PlanningImmutableLookupVolume::forTokenHeads(rows,device.batch_size,g.ple_heads,g.ple_table_row_bytes,g.ple_row_elements);
        const auto queries=volume.queries();
        out.live_queries=add(out.live_queries,queries);
        // Every actual PLE reader decodes the full gathered source-head vector;
        // fixed storage ownership only partitions the original native reads.
        if(rows)out.memory.push_back({reader,0,volume.decodedBytes()});
        if(mode==ImmutableRowLookupExecution::FixedHeadShardGPU) {
            if(!fixed_done.insert(placement.get()).second)continue;
            std::vector<DeviceId> readers;
            const auto consumers=placement->consumers();
            for(const auto consumer:consumers) {
                if(consumer.world_rank!=identity.world_rank || !consumer.device.is_gpu() ||
                    consumer.device.type!=identity.device.type)
                    throw std::invalid_argument("Immutable gather consumers must retain one original native rank/backend");
                readers.push_back(consumer.device);
            }
            std::vector<std::size_t> fragments(readers.size(),0);
            const auto *pipeline=placement->fixedHeadPipeline();
            if(pipeline && pipeline->consumers().size()!=consumers.size())
                throw std::invalid_argument("Immutable gather pipeline lost original consumer extent");
            for(const auto owner:placement->allocators()) {
                const auto heads=headsInRange(g,placement->fixedHeadSourceRangeFor(owner));
                const auto owned=PlanningImmutableLookupVolume::forTokenHeads(rows,device.batch_size,heads,g.ple_table_row_bytes,g.ple_row_elements);
                const auto count=owned.queries(),bytes=owned.nativeBytes();
                const auto source=endpoint(owner,ranks);if(rows)out.memory.push_back({source,bytes,0});
                if(pipeline) {
                    const auto &role=pipeline->owner(owner);
                    if(role.consumer_index<0 || static_cast<size_t>(role.consumer_index)>=consumers.size() ||
                        consumers[role.consumer_index]!=pipeline->consumers()[role.consumer_index])
                        throw std::invalid_argument("Immutable gather pipeline changed original consumer order");
                    const auto destination=endpoint(pipeline->consumers()[role.consumer_index],ranks);
                    if(rows)exchange(out,source,destination,count,bytes);
                    fragments[role.consumer_index]=add(fragments[role.consumer_index],bytes);
                } else {
                    const auto consumer=std::find(consumers.begin(),consumers.end(),owner);
                    if(consumer==consumers.end())throw std::invalid_argument("Immutable local head owner has no original gather consumer");
                    const auto index=static_cast<size_t>(consumer-consumers.begin());
                    fragments[index]=add(fragments[index],bytes);
                }
            }
            size_t gathered_bytes=0;for(const auto bytes:fragments)gathered_bytes=add(gathered_bytes,bytes);
            if(gathered_bytes!=volume.nativeBytes())throw std::invalid_argument("Immutable gather lost original live native head coverage");
            if(readers.size()>1) {
                // The consumer's assembled fragment includes its exact local
                // and remote storage heads. Native broadcast/gather algorithm
                // choice does not turn this unique ownership into peer fanout.
                out.channels.push_back({PlanningImmutableChannelKind::NativeGather,reader,reader,std::move(readers),0,
                    gathered_bytes,std::move(fragments),
                    pipeline && pipeline->requiresVariableGather()
                        ?PlanningImmutableGatherProtocol::CountedDevicePrefix:PlanningImmutableGatherProtocol::EqualDeviceRows,
                    sizeof(qwen4exp::FixedHeadValidationHeader)});
            }
            continue;
        }
        PhysicalMemoryAllocatorIdentity owner=identity;
        auto storage=ImmutableRowStorage::VRAM;
        if(mode==ImmutableRowLookupExecution::AdaptiveGPUCache) {
            const auto &layout=placement->gpuAdaptiveLayout();
            owner=layout.sourceServices().front().allocator;
            storage=placement->gpuAdaptiveRequest().sourceStorage();
        } else if(mode==ImmutableRowLookupExecution::HostProvider) {
            if(placement->isHostAdaptive()) {
                owner=placement->hostAdaptiveLayout().sourceServices().front().allocator;
                storage=ImmutableRowStorage::MemoryMapped; // Cold misses receive no unobserved host-cache credit.
            } else {
                storage=placement->storage();
                const auto allocators=placement->allocators();
                const auto found=std::find_if(allocators.begin(),allocators.end(),[](const auto &a){return a.device.is_cpu();});
                if(found==allocators.end())throw std::invalid_argument("Hosted immutable lookup lost its source owner");
                owner=*found;
            }
        } else if(mode!=ImmutableRowLookupExecution::ResidentGPU)
            throw std::invalid_argument("Unknown immutable lookup execution contract");
        const auto source=endpoint(owner,ranks);const auto bytes=volume.nativeBytes();
        if(storage==ImmutableRowStorage::MemoryMapped)out.requires_storage_read_service=true;
        out.memory.push_back({source,bytes,0});
        if(mode==ImmutableRowLookupExecution::AdaptiveGPUCache) {
            const auto coordinator=placement->gpuAdaptiveLayout().gpuCoordinatorFor(identity).value_or(identity);
            const auto query=endpoint(coordinator,ranks);
            exchange(out,source,query,queries,bytes);
            exchange(out,query,reader,queries,bytes);
        } else exchange(out,source,reader,queries,bytes);
    }
    if(!consumed && rows)throw std::invalid_argument("Immutable source has no admitted main-forward PLE consumer");
    return out;
}

PlanningImmutableLookupCost pricePlanningImmutableLookupWork(const PlanningModelMetadata &model,
    const AdmittedOrchestrationCandidate &candidate,std::size_t rows,const PlanningWeightServiceModel &weights,
    const PlanningCommunicationCost &communication) {
    const auto work=compilePlanningImmutableLookupWork(model.memoryProfile(),candidate.devicePlans(),candidate.membership().discoveryRanks(),rows);
    if(work.requires_storage_read_service)throw std::invalid_argument("Immutable lookup cold SSD read cost lacks measured storage/random-read service; DRAM bandwidth cannot price SSD");
    PlanningImmutableLookupCost result;
    if(work.memory.empty() && work.channels.empty())return result;
    result.qualifications={"immutable lookup: cold native-row misses; no cache-hit, deduplication or future hot-cache benefit",
        "immutable lookup: measured streaming-memory traffic proxy; random-access/dequantization/control latency unmeasured",
        "immutable lookup: serial owner/reader/link composition without unproved overlap credit",
        "immutable host source: original rank-bound CPU streaming team proxy, not sampled single-worker/random-row IO latency"};
    for(const auto &traffic:work.memory)result.compute_seconds+=weights.memorySeconds(traffic.endpoint.discovery_rank,
        traffic.endpoint.device,static_cast<double>(add(traffic.read_bytes,traffic.write_bytes)));
    for(const auto &channel:work.channels) {
        if(channel.kind==PlanningImmutableChannelKind::HostRequestReply) {
            const auto &gpu=channel.consumer;const auto &host=channel.producer;
            result.communication_seconds+=communication.hostTransfer(gpu.discovery_rank,gpu.device,host.discovery_rank,
                PlanningHostTransferMechanism::MappedKernel,MappedTransferDirection::DeviceToHost,channel.request_bytes).seconds;
            result.communication_seconds+=communication.hostTransfer(gpu.discovery_rank,gpu.device,host.discovery_rank,
                PlanningHostTransferMechanism::MappedKernel,MappedTransferDirection::HostToDevice,channel.response_bytes).seconds;
        } else if(channel.kind==PlanningImmutableChannelKind::NativeRequestReply) {
            // Query IDs travel from the actual consumer to the original owner;
            // native row bytes return in the opposite direction, without FP32
            // expansion or alignment-padding charges in the selection score.
            const auto request=communication.physicalPeerTransfer(channel.consumer.discovery_rank,
                channel.consumer.device,channel.producer.device,channel.request_bytes);
            const auto response=communication.physicalPeerTransfer(channel.producer.discovery_rank,
                channel.producer.device,channel.consumer.device,channel.response_bytes);
            result.communication_seconds+=request.seconds+response.seconds;
            result.qualifications.push_back(request.evidence);result.qualifications.push_back(response.evidence);
        } else {
            size_t total=0;for(const auto bytes:channel.owner_live_bytes)total=add(total,bytes);
            if(channel.request_bytes || total!=channel.response_bytes ||
                channel.owner_live_bytes.size()!=channel.native_group.size())
                throw std::invalid_argument("Immutable gather cost lost original per-consumer native bytes");
            if(channel.header_bytes_per_owner!=sizeof(qwen4exp::FixedHeadValidationHeader))
                throw std::invalid_argument("Immutable gather lost its original mandatory validation header");
            const auto header=communication.calibratedNativeAllgather(channel.producer.discovery_rank,
                channel.native_group,channel.header_bytes_per_owner);
            result.communication_seconds+=header.seconds;result.qualifications.push_back(header.evidence);
            if(channel.gather_protocol!=PlanningImmutableGatherProtocol::EqualDeviceRows &&
                channel.gather_protocol!=PlanningImmutableGatherProtocol::CountedDevicePrefix)
                throw std::invalid_argument("Immutable gather has an unknown actual payload protocol");
            // Equal-row zero payload has no data service. Counted-prefix zero
            // still runs the original frame/extent/completion sequence; its
            // measured zero-byte control floor must not disappear.
            if(total || channel.gather_protocol==PlanningImmutableGatherProtocol::CountedDevicePrefix) {
                const auto payload=[&] {
                    if(channel.gather_protocol==PlanningImmutableGatherProtocol::CountedDevicePrefix)
                        return communication.calibratedNativeCountedAllgather(channel.producer.discovery_rank,
                            channel.native_group,channel.owner_live_bytes);
                    if(channel.gather_protocol!=PlanningImmutableGatherProtocol::EqualDeviceRows ||
                        !std::all_of(channel.owner_live_bytes.begin(),channel.owner_live_bytes.end(),
                            [&](size_t bytes){return bytes==channel.owner_live_bytes.front();}))
                        throw std::invalid_argument("Immutable equal-row gather changed original owner payload widths");
                    return communication.calibratedNativeAllgather(channel.producer.discovery_rank,
                        channel.native_group,channel.owner_live_bytes.front());
                }();
                result.communication_seconds+=payload.seconds;result.qualifications.push_back(payload.evidence);
            }
            result.qualifications.push_back("immutable gather: separate mandatory validation header and exact native owner fragments; calibrated fixed timing estimates DeviceRows payload service, not measured full DeviceRows control latency");
        }
    }
    return result;
}
}
