/** @file PlanningNativeContextEvidence.cpp
 * @brief Project installed native family budgets from exact original-rank
 * physical declarations. The existing failure-atomic planning exchange reports
 * each original rank once.
 */
#include "planning/PlanningNativeContextEvidence.h"
#include "backends/NativeContextPreparation.h"
#include "backends/PreparedGPUWorkerThread.h"
#include "interfaces/IMPIContext.h"
#include "planning/AutomaticPlanningStartup.h"
#include "planning/ExecutionRankMembership.h"
#include "planning/PlanningPublication.h"
#include "planning/RankHardwareOwnership.h"
#include <algorithm>
#include <nlohmann/json.hpp>
#include <set>
namespace llaminar2 {
namespace {
using Json = nlohmann::json;
/** @return Only GPU observers allowed by the actual automatic request. */
std::vector<DeviceId> devices(const AutomaticPlanningPreparation &c, int rank) {
  std::vector<DeviceId> result;
  const auto request = std::get<AutomaticOrchestrationRequest>(
      resolveOrchestrationIntent(c.request()));
  const auto &local = c.inventory().ranks.at(rank);
  for (const auto &gpu : local.gpus)
    if (request.allows(gpu.type) && rankOwnsGPUDevice(local, gpu))
      result.push_back({gpu.type, gpu.local_device_id});
  return result;
}
/** @return Exact source-owned geometry, without provider or runtime identity
 * hashing. */
Json encode(const PlanningNativeContextDeclaration &k) {
  return Json{{"type", static_cast<int>(k.device.type)},
              {"ordinal", k.device.ordinal},
              {"uuid", k.physical_uuid},
              {"stack", k.stack_bytes},
              {"guard", k.guard_bytes},
              {"page", k.page_bytes},
              {"object", k.object_bytes},
              {"alignment", k.object_alignment}};
}
/** @return Original rank's physical declaration; installed catalog owns all
 * opaque limits. */
PlanningNativeContextDeclaration decode(const Json &j) {
  if (!j.is_object() || j.size() != 8)
    throw std::invalid_argument(
        "Native planning declaration has unexpected fields");
  PlanningNativeContextDeclaration k{
      {static_cast<DeviceType>(j.at("type").get<int>()),
       j.at("ordinal").get<int>()},
      j.at("uuid").get<std::string>(),
      j.at("stack").get<size_t>(),
      j.at("guard").get<size_t>(),
      j.at("page").get<size_t>(),
      j.at("object").get<size_t>(),
      j.at("alignment").get<size_t>()};
  if (k.physical_uuid.empty())
    throw std::invalid_argument(
        "Native planning declaration has no original UUID");
  (void)GPUWorkerStackProgram::planningAllocationBytes(
      k.stack_bytes, k.guard_bytes, k.page_bytes, k.object_bytes,
      k.object_alignment);
  (void)NativeContextOpaqueCatalog::planningBudget(
      k.device.type, NativeContextBirthPopulation::AdditionalContext);
  return k;
}
} // namespace
const std::vector<PlanningNativeContextDeclaration> &
projectPlanningNativeContextBudget(const PlanningNativeContextBudgets &budgets,
                                   const ExecutionRankMembership &membership,
                                   int execution) {
  const auto discovery = membership.discoveryRanks().at(execution);
  const auto found = budgets.find(discovery);
  if (found == budgets.end())
    throw std::invalid_argument(
        "Native birth budget omitted original discovery rank");
  return found->second;
}
PhysicalMemoryPlan
planningNativeContextBOM(std::span<const PlanningNativeContextDeclaration> keys,
                         const PhysicalMemoryResource &host,
                         std::span<const PhysicalMemoryResource> selected) {
  PhysicalMemoryPlanBuilder b;
  std::set<std::pair<DeviceType, int>> seen;
  for (const auto &gpu : selected) {
    if (!gpu.device.is_gpu() || !host.device.is_cpu() ||
        gpu.world_rank != host.world_rank ||
        !seen.emplace(gpu.device.type, gpu.device.ordinal).second)
      throw std::invalid_argument("Native planning budget requires unique "
                                  "same-rank original CPU/GPU allocators");
    const auto key = std::find_if(keys.begin(), keys.end(), [&](const auto &k) {
      return k.device == gpu.device;
    });
    if (key == keys.end() || key->physical_uuid.empty())
      throw std::invalid_argument(
          "Selected GPU has no original prepared native birth evidence");
    if (std::count_if(keys.begin(), keys.end(), [&](const auto &k) {
          return k.device == gpu.device;
        }) != 1)
      throw std::invalid_argument("Duplicate native birth evidence");
    const auto contract = NativeContextOpaqueCatalog::planningBudget(
        key->device.type, NativeContextBirthPopulation::AdditionalContext);
    b.add(gpu, PhysicalMemoryOwner::NativeExecutionContext,
          contract.deviceUpperBytes());
    b.add(host, PhysicalMemoryOwner::NativeExecutionContext,
          contract.hostUpperBytes());
    b.add(host, PhysicalMemoryOwner::LocalCollective,
          GPUWorkerStackProgram::planningAllocationBytes(
              key->stack_bytes, key->guard_bytes, key->page_bytes,
              key->object_bytes, key->object_alignment));
  }
  return b.build();
}
PlanningNativeContextBudgets
collectPlanningNativeContextBudgets(const AutomaticPlanningPreparation &c) {
  PlanningNativeContextBudgets result;
  exchangePlanningSamples(
      c.mpi(),
      [&](int count) {
        if (count != c.inventory().world_size)
          throw std::invalid_argument(
              "Native birth discovery communicator changed");
        return std::vector<std::vector<std::uint8_t>>(count, {'N', 'B', 1});
      },
      [&](std::span<const std::uint8_t> request) {
        if (request.size() != 3 || request[0] != 'N' || request[1] != 'B' ||
            request[2] != 1)
          throw std::invalid_argument("Native birth request mismatch");
        const int rank = c.mpi() ? c.mpi()->rank() : 0;
        Json j = Json::array();
        for (const auto device : devices(c, rank)) {
          const auto program = GPUWorkerStackProgram::originalDefaults();
          const auto object = PreparedGPUContextFactory::objectGeometry(device);
          const auto &rank_inventory = c.inventory().ranks.at(rank);
          const auto physical =
              std::find_if(rank_inventory.gpus.begin(),
                           rank_inventory.gpus.end(), [&](const auto &gpu) {
                             return gpu.type == device.type &&
                                    gpu.local_device_id == device.ordinal;
                           });
          if (physical == rank_inventory.gpus.end() || physical->uuid.empty())
            throw std::invalid_argument(
                "Native planning endpoint lacks original inventory UUID");
          j.push_back(encode({device, physical->uuid, program.stackBytes(),
                              program.guardBytes(), program.pageBytes(),
                              object.bytes(), object.alignment()}));
        }
        const auto text = j.dump();
        return std::vector<std::uint8_t>(text.begin(), text.end());
      },
      [&](std::span<const RankPlanningSample> samples) {
        if (samples.size() != size_t(c.inventory().world_size))
          throw std::invalid_argument("Native birth discovery omitted ranks");
        for (size_t rank = 0; rank < samples.size(); ++rank) {
          if (samples[rank].discovery_rank != int(rank))
            throw std::invalid_argument(
                "Native birth discovery rank/extent mismatch");
          const auto j = Json::parse(samples[rank].bytes);
          const auto expected = devices(c, rank);
          if (!j.is_array() || j.size() != expected.size())
            throw std::invalid_argument(
                "Native birth discovery omitted GPU evidence");
          std::vector<PlanningNativeContextDeclaration> keys;
          for (size_t i = 0; i < j.size(); ++i) {
            auto key = decode(j[i]);
            if (key.device != expected[i])
              throw std::invalid_argument(
                  "Native birth discovery substituted endpoint");
            const auto &original = c.inventory().ranks.at(rank).gpus;
            const auto physical = std::find_if(
                original.begin(), original.end(), [&](const auto &gpu) {
                  return gpu.type == key.device.type &&
                         gpu.local_device_id == key.device.ordinal;
                });
            if (physical == original.end() ||
                physical->uuid != key.physical_uuid)
              throw std::invalid_argument(
                  "Native planning declaration changed original UUID");
            keys.push_back(std::move(key));
          }
          result.emplace(rank, std::move(keys));
        }
      });
  return result;
}
} // namespace llaminar2
