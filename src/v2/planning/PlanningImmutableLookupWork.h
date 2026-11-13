/**
 * @file PlanningImmutableLookupWork.h
 * @brief Live immutable-row lookup traffic from admitted source/consumer ownership.
 *
 * This cold metadata projection is neither a source allocator nor a cache-hit
 * predictor. Native row width and head ownership determine traffic; table and
 * cache capacities never become per-invocation bytes. SSD service remains an
 * explicit unmeasured component, never an invented DRAM bandwidth.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace llaminar2 {
class PlanningModelMetadata;
class AdmittedOrchestrationCandidate;
class PlanningWeightServiceModel;
class PlanningCommunicationCost;
struct ModelMemoryProfile;
struct DevicePlanConfig;
/** @brief Checked live volume; source table/cache capacity is not an input. */
class PlanningImmutableLookupVolume final {
public:
    /** @brief Multiply actual token/batch/head coordinates and native/decoded row widths.
     * @param rows Live token rows; zero denotes no invocation work.
     * @param batch Actual request batch, positive even for empty work.
     * @param heads Actual requested source heads, not stored table rows.
     * @param native_row_bytes Original complete native row width.
     * @param row_elements Original decoded elements per source row.
     * @return Checked immutable live query/native/FP32-output volumes.
     * @throws std::exception For empty geometry or overflow. */
    static PlanningImmutableLookupVolume forTokenHeads(std::size_t rows,std::size_t batch,
        std::size_t heads,std::size_t native_row_bytes,std::size_t row_elements);
    /** @return Actual original row IDs requested. */
    std::size_t queries() const noexcept { return queries_; }
    /** @return Exact native bytes read/returned, excluding unused cache/table rows. */
    std::size_t nativeBytes() const noexcept { return native_; }
    /** @return Actual decoded FP32 vector bytes. */
    std::size_t decodedBytes() const noexcept { return decoded_; }
private:
    /** @brief Only checked live coordinates construct a volume. */
    PlanningImmutableLookupVolume(std::size_t queries,std::size_t native,std::size_t decoded)
        :queries_(queries),native_(native),decoded_(decoded) {}
    std::size_t queries_,native_,decoded_;
};
/** @brief Exact physical execution namespace and its original measurement owner. */
struct PlanningImmutableEndpoint {
    int execution_rank;
    int discovery_rank;
    DeviceId device;
};
/** @brief One owner-local live native read or decoded output write. */
struct PlanningImmutableMemoryTraffic {
    PlanningImmutableEndpoint endpoint;
    std::size_t read_bytes = 0, write_bytes = 0;
};
/** @brief Distinct measured service families; no transport choice is made here. */
enum class PlanningImmutableChannelKind { HostRequestReply, NativeRequestReply, NativeGather };
/** @brief Actual fixed-head graph payload transport; headers use equal native row gather. */
enum class PlanningImmutableGatherProtocol { EqualDeviceRows, CountedDevicePrefix };
/** @brief Exact live payload plus explicit protocol metadata, never admitted capacity. */
struct PlanningImmutableChannel {
    PlanningImmutableChannelKind kind;
    PlanningImmutableEndpoint producer, consumer;
    std::vector<DeviceId> native_group;
    std::size_t request_bytes = 0, response_bytes = 0;
    /** NativeGather only: exact assembled native fragment at each corresponding consumer.
     * Storage-stage owners contribute to their declared pipeline consumer;
     * the vector is never an equal-size division of total response bytes. */
    std::vector<std::size_t> owner_live_bytes;
    PlanningImmutableGatherProtocol gather_protocol=PlanningImmutableGatherProtocol::EqualDeviceRows;
    std::size_t header_bytes_per_owner=0; ///< Separate mandatory validation header, not native payload bytes.
};
/** @brief One complete invocation, with cold misses and no unobserved cache benefit. */
struct PlanningImmutableLookupWork {
    std::vector<PlanningImmutableMemoryTraffic> memory;
    std::vector<PlanningImmutableChannel> channels;
    std::size_t live_queries = 0;
    bool requires_storage_read_service = false;
};
/**
 * @brief Compile exact live query volume through already-admitted row placements.
 * @param profile Original native tensor/head geometry; no payload is opened.
 * @param devices Compiled temporal participants, including storage-only PP owners.
 * @param discovery_ranks Original ordered membership for compact execution ranks.
 * @param rows Actual invocation rows; zero omits payload work but retains fixed-head validation headers.
 * @return Native reads, decoded writes and required channels, with SSD absence explicit.
 * @throws std::exception For missing/foreign placement, invalid head ranges or overflow.
 */
PlanningImmutableLookupWork compilePlanningImmutableLookupWork(const ModelMemoryProfile &profile,
    std::span<const DevicePlanConfig> devices, std::span<const int> discovery_ranks, std::size_t rows);
/** @brief Qualified dependent lookup cost, separate from ordinary projection/parameter work. */
struct PlanningImmutableLookupCost {
    double compute_seconds = 0, communication_seconds = 0;
    std::vector<std::string> qualifications;
};
/**
 * @brief Price mandatory lookup work using existing measured memory/link services.
 * @param model Metadata that admitted the candidate.
 * @param candidate Complete physical placement, never an unresolved request.
 * @param rows Live invocation rows, not graph capacity or total table rows.
 * @param weights Observed CPU/GPU streaming-memory service.
 * @param communication Observed host-device/native communication service.
 * @return Serialized traffic-service proxy without speculative overlap/hit credit.
 * @throws std::exception For missing measured service, including cold SSD reads.
 */
PlanningImmutableLookupCost pricePlanningImmutableLookupWork(const PlanningModelMetadata &model,
    const AdmittedOrchestrationCandidate &candidate, std::size_t rows,
    const PlanningWeightServiceModel &weights, const PlanningCommunicationCost &communication);
}
