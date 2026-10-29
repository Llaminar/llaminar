/**
 * @file Qwen4ExpSourceAdmissionCI.h
 * @brief Source-only typed discovery and native hardware observation for CI.
 *
 * Numerical parameters keep their original printer and approved schema bytes.
 * This wrapper changes neither case names nor runtime configuration. A fresh
 * native inventory is emitted only by an explicit single-rank metadata launch;
 * it owns no model and uses the same startup hardware authority as admission.
 */
#pragma once
#include "Qwen4ExpSourceAdmissionCases.h"
#include "../ModelParityRuntimeExport.h"
#include "backends/HardwareInventory.h"
#include <algorithm>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <mpi.h>
#include <fcntl.h>
#include <set>
#include <unistd.h>

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Source-only parameter; the canonical case remains its sole policy. */
struct Qwen4ExpSourceAdmissionCase : ModelParityCase
{
    /** @param cell Existing expanded cell, copied without policy mutation. */
    Qwen4ExpSourceAdmissionCase(const ModelParityCase &cell) : ModelParityCase(cell) {}
    /** @param probe Separately authored capacity-only request and its public cap. */
    Qwen4ExpSourceAdmissionCase(const Qwen4ExpSourceCapacityAdmissionCase &probe)
        : ModelParityCase(probe.cell), gpu_memory_cap_mb(probe.gpu_memory_cap_mb) {}
    /** Provisioning prerequisite only; never an allocation or free-capacity estimate. */
    std::optional<std::size_t> minimum_cuda_total_memory_bytes;
    std::optional<std::size_t> gpu_memory_cap_mb; ///< Authored public cap; absent on ordinary source cases.
};

/** @brief Project existing cells to a distinct printer without another expander. */
inline std::vector<Qwen4ExpSourceAdmissionCase> qwen4ExpSourceAdmissionParameters(
    const std::vector<ModelParityCase> &cells)
{
    return {cells.begin(), cells.end()};
}

/**
 * @brief Author the conservative large-card class for original fixed-head positives.
 * @param cells Existing full fixed-head source family, expanded without renaming.
 * @return Same identities/configurations; only CUDA TP2 eight-expert active/full15
 * VRAM requests require 32GiB physical total on every participating CUDA GPU.
 *
 * This reviewed test provisioning class is deliberately independent of PMA
 * prices and currently free bytes. Off, bounded families and negative controls
 * never enter this source-family projection and remain mandatory.
 */
inline std::vector<Qwen4ExpSourceAdmissionCase> qwen4ExpFixedHeadSourceAdmissionParameters(
    const std::vector<ModelParityCase> &cells)
{
    auto result = qwen4ExpSourceAdmissionParameters(cells);
    for (auto &cell : result)
    {
        const auto &plan = cell.topology.expert_overlay_plan;
        if (cell.mtp == ModelParityMTP::Off || cell.retained_mtp_draft_capacity != 15 ||
            !cell.ngram_table_policy || !cell.ngram_table_policy->isFixedHeadSharded() ||
            !plan || plan->routed_tiers.empty() || plan->routed_tiers.front().max_experts_per_layer != 8)
            continue;
        const auto domain = std::find_if(plan->domains.begin(), plan->domains.end(),
            [&](const auto &entry) { return entry.name == plan->continuation_domain; });
        if (domain == plan->domains.end()) throw std::logic_error("Fixed-head source provisioning lost its continuation");
        if (domain->participants.size() == 2u &&
            std::all_of(domain->participants.begin(), domain->participants.end(),
                [](const auto &address) { return address.device_type == DeviceType::CUDA; }))
            cell.minimum_cuda_total_memory_bytes = std::size_t{32} * 1024 * 1024 * 1024;
    }
    return result;
}

/** @brief Preserve a typed capacity probe without changing its underlying model policy. */
inline std::vector<Qwen4ExpSourceAdmissionCase> qwen4ExpSourceAdmissionParameters(
    const std::vector<Qwen4ExpSourceCapacityAdmissionCase> &cells)
{
    return {cells.begin(), cells.end()};
}

/** @brief Emit exact declared identities; names and CLI fragments are not topology. */
inline void PrintTo(const Qwen4ExpSourceAdmissionCase &cell, std::ostream *out)
{
    *out << "{\"qfn_source_admission_schema\":2,\"id\":"
         << modelParityJsonString(cell.testName());
    if (cell.minimum_cuda_total_memory_bytes)
        *out << ",\"minimum_cuda_total_memory_bytes\":" << *cell.minimum_cuda_total_memory_bytes;
    if (cell.gpu_memory_cap_mb)
        *out << ",\"gpu_memory_cap_mb\":" << *cell.gpu_memory_cap_mb;
    *out << ",\"mtp_enabled\":" << (cell.mtp != ModelParityMTP::Off ? "true" : "false")
         << ",\"retained_mtp_draft_capacity\":" << cell.retained_mtp_draft_capacity;
    *out << ",\"mpi_ranks\":" << cell.topology.mpi_ranks << ",\"participants\":[";
    bool first = true;
    for (const auto &participant : cell.topology.participants)
    {
        const auto &address = participant.address;
        *out << (first ? "" : ",") << "{\"rank\":" << (participant.world_rank ? std::to_string(*participant.world_rank) : "null")
             << ",\"backend\":" << modelParityJsonString(address.isCPU() ? "cpu" : address.isCUDA() ? "cuda" : address.isROCm() ? "rocm" : "unsupported")
             << ",\"ordinal\":" << address.device_ordinal << ",\"numa_node\":" << address.numa_node << '}';
        first = false;
    }
    *out << "]}";
}

/** @brief Adapter used exclusively by gold-free source fixtures. */
class Qwen4ExpSourceAdmissionParameter
    : public ::testing::WithParamInterface<Qwen4ExpSourceAdmissionCase>
{
public:
    /** @return Unmodified canonical policy for the production admission path. */
    const ModelParityCase &modelParityCase() const { return GetParam(); }
};

/**
 * @brief Write a fresh, single-process native hardware observation.
 * @param path Absolute caller-owned output, required not to exist.
 * @throws std::exception On malformed output or incomplete hardware evidence.
 *
 * The caller performs normal context/collective retirement and MPI finalization
 * before Python admits this document. Native failure never means zero devices.
 */
inline void exportQwen4ExpSourceHardware(const std::filesystem::path &path)
{
    int ranks = 0;
    if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS || ranks != 1 ||
        !path.is_absolute() || !std::filesystem::is_directory(path.parent_path()) ||
        std::filesystem::exists(path) || std::filesystem::is_symlink(path))
        throw std::invalid_argument("QFN source inventory requires one rank and an absolute caller-owned output");
    auto &manager = DeviceManager::instance();
    manager.initialize(-1, false);
    const auto *hardware = manager.hardware();
    if (!hardware) throw std::logic_error("QFN source inventory has no native hardware authority");
    std::ostringstream out;
    out << "{\"qfn_source_hardware_schema\":2,\"scope\":\"local-host-all-ranks\",\"devices\":[";
    bool first = true;
    const auto append = [&](const char *backend, int ordinal, int numa, std::size_t total = 0u) {
        out << (first ? "" : ",") << "{\"backend\":" << modelParityJsonString(backend)
            << ",\"ordinal\":" << ordinal << ",\"numa_node\":" << numa;
        if (total) out << ",\"total_memory_bytes\":" << total;
        out << '}';
        first = false;
    };
    std::set<int> localities;
    for (const auto &socket : hardware->cpu_sockets)
        if (socket.numa_node >= 0) localities.insert(socket.numa_node);
    for (const auto locality : localities)
    {
        (void)hardware->cpuDevice(locality);
        append("cpu", 0, locality);
    }
    for (const auto &device : hardware->cuda_devices) append("cuda", device.device_id, device.numa_node, device.total_memory_bytes);
    for (const auto &device : hardware->rocm_devices) append("rocm", device.device_id, device.numa_node, device.total_memory_bytes);
    out << "]}\n";
    const auto data = out.str();
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) throw std::runtime_error("QFN source inventory output must be fresh and writable");
    const auto written = ::write(fd, data.data(), data.size());
    const auto closed = ::close(fd);
    if (written != static_cast<ssize_t>(data.size()) || closed != 0)
        throw std::runtime_error("QFN source inventory output publication failed");
}
} // namespace llaminar2::test::parity::qwen4exp
