/**
 * @file GDNLiveStateWorkspaceContract.h
 * @brief Original named LocalTP GDN handoff workspace shared by graph and BOM.
 *
 * These six per-global-layer banks coexist with the producer graph and are
 * distinct from kernel-owned recurrent live state. PhysicalMemoryAuthority
 * alone owns their eventual backing. No native context or payload is required
 * to enumerate their exact source geometry, stable names and original lifetime.
 */
#pragma once
#include "GDNLinkedLiveStateGeometry.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include <stdexcept>
#include <string>

namespace llaminar2::gdn_live_state_workspace
{
inline constexpr const char *LocalConv = "gdn_live_state_local_conv";
inline constexpr const char *GatheredConv = "gdn_live_state_gathered_conv";
inline constexpr const char *FullConv = "gdn_live_state_full_conv";
inline constexpr const char *LocalRecurrence = "gdn_live_state_local_recurrence";
inline constexpr const char *GatheredRecurrence = "gdn_live_state_gathered_recurrence";
inline constexpr const char *FullRecurrence = "gdn_live_state_full_recurrence";

/** @brief Original local collective degree and model-owned linked-state layout. */
struct Geometry
{
    int degree = 0;
    GDNLinkedLiveStateGeometry state;
};

/**
 * @brief Enumerate exactly the actual handoff's local/raw-gathered/reassembled banks.
 * @param layer Original global layer coordinate, never a stage-local renumbering.
 * @param geometry Actual equal native LocalTP membership and source state layout.
 * @return Same six 256-aligned, required Any-regime transient names as the stage.
 * @throws std::invalid_argument For invalid layer or incomplete linked geometry.
 */
inline WorkspaceRequirements requirements(int layer, const Geometry &geometry)
{
    if (layer < 0 || geometry.degree <= 1)
        throw std::invalid_argument("GDN handoff workspace requires a global layer and LocalTP degree");
    const auto conv = geometry.state.resolve(GDNLinkedLiveStateKind::ConvHistory, geometry.degree);
    const auto recurrence = geometry.state.resolve(GDNLinkedLiveStateKind::Recurrence, geometry.degree);
    if (!conv || !recurrence)
        throw std::invalid_argument("GDN handoff workspace has no complete equal-count source layout");
    WorkspaceRequirements result;
    const auto append = [&](const char *local, const char *gathered, const char *full,
                            const GDNLinkedLiveStateShape &shape)
    {
        const auto name = [&](const char *stem) { return std::string(stem) + "_layer" + std::to_string(layer); };
        result.buffers.push_back({name(local), static_cast<size_t>(shape.local_state_floats) * sizeof(float), 256, true});
        result.buffers.push_back({name(gathered), static_cast<size_t>(shape.full_state_floats) * sizeof(float), 256, true});
        result.buffers.push_back({name(full), static_cast<size_t>(shape.full_state_floats) * sizeof(float), 256, true});
    };
    append(LocalConv, GatheredConv, FullConv, *conv);
    append(LocalRecurrence, GatheredRecurrence, FullRecurrence, *recurrence);
    return result;
}
}
