/**
 * @file CPUW4A4Workspace.h
 * @brief Canonical stage/PMA workspace binding for native CPU W4A4/H arithmetic.
 *
 * Declaration and execution use the same typed arithmetic counts. The stage's
 * DeviceWorkspaceManager retains storage and the physical admission; shared
 * prepared kernels borrow only the current invocation's live prefixes. Names
 * may merge by maximum only across nonconcurrent operations, following the
 * existing arena lifetime contract. No private allocator or ledger is present.
 */
#pragma once

#include "CPUW4A4Arithmetic.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include <type_traits>

namespace llaminar2::w4a4
{
    inline const std::string kCPUActivationBlocks = "cpu_w4a4_blocks"; ///< Packed A4 operands.
    inline const std::string kCPUGroupRanges = "cpu_w4a4_group_ranges"; ///< Group extrema before token reduction.
    inline const std::string kCPUTokenRanges = "cpu_w4a4_token_ranges"; ///< Exact unrounded whole-token maxima.
    inline const std::string kCPUButterfly = "cpu_w4a4_butterfly"; ///< Ordered Sylvester producer bank.
    inline const std::string kCPUTransformed = "cpu_w4a4_transformed"; ///< Normalized Paley consumer bank.
    inline const std::string kCPUActiveRows = "cpu_w4a4_active_rows"; ///< Validity of prepared original-coordinate operands.

    inline const std::string kCPUProjectionBundle = "cpu_w4a4_projection_bundle"; ///< Stage-owned invocation metadata.

    /** @brief Borrowed per-leaf references, local for one leaf and arena-owned for a bundle. */
    struct ProjectionInvocation
    {
        const SourceView *weights;          ///< Borrowed immutable prepared owner.
        CPUActivationBuffers activation;   ///< Exact live prefixes of the shared banks.
        std::span<float> output;            ///< Already-published CPU output authority.
        std::span<const float> bias;        ///< Optional already-published CPU bias.
    };
    static_assert(std::is_trivially_copyable_v<ProjectionInvocation>);

    /**
     * @brief Declare variable-length fused-leaf metadata in the existing stage arena.
     * @param projections Exact declared leaf count, not model layers or row capacity.
     * @return One aligned bank for multiple leaves; one leaf needs only local references.
     *
     * A singleton verifier invokes the same grouped arithmetic as a bundle, but
     * its fixed descriptor fits in the invocation frame. It owns no tensor data
     * and must not acquire a persistent metadata bank just to name one output.
     * @throws std::invalid_argument or std::overflow_error For empty/overflowing bundles.
     */
    [[nodiscard]] inline WorkspaceRequirements cpuFusedProjectionWorkspaceRequirements(std::size_t projections)
    {
        if (!projections) throw std::invalid_argument("W4A4 fusion requires a nonempty projection inventory");
        WorkspaceRequirements result;
        if (projections == 1) return result;
        result.buffers.emplace_back(kCPUProjectionBundle,
            CPUInvocationWorkspace::multiply(projections, sizeof(ProjectionInvocation)), 64);
        return result;
    }

    /**
     * @brief Contribute exact persistent scratch to the existing workspace/BOM authority.
     * @param transform Original format's source-declared online transform.
     * @param columns Complete original K; rotation block length does not change storage.
     * @param rows Maximum simultaneous input rows, not KV capacity or a thread count.
     * @return Required aligned banks; identity transforms own no rotation scratch.
     * @throws AssertionError or std::overflow_error For invalid or unrepresentable geometry.
     */
    [[nodiscard]] inline WorkspaceRequirements cpuWorkspaceRequirements(
        Transform transform, int columns, int rows)
    {
        const auto counts = cpuWorkspaceCounts(transform, columns, rows);
        WorkspaceRequirements requirements;
        const auto append = [&requirements](const std::string &name, std::size_t count, std::size_t width)
        {
            const auto bytes = CPUInvocationWorkspace::multiply(count, width);
            if (bytes) requirements.buffers.emplace_back(name, bytes, 64);
        };
        append(kCPUActivationBlocks, counts.blocks, sizeof(ActivationBlock));
        append(kCPUGroupRanges, counts.ranges, sizeof(ActivationRange));
        append(kCPUTokenRanges, counts.row_ranges, sizeof(float));
        append(kCPUButterfly, counts.butterfly_floats, sizeof(float));
        append(kCPUTransformed, counts.transformed_floats, sizeof(float));
        append(kCPUActiveRows, counts.active_rows, sizeof(std::uint8_t));
        return requirements;
    }

    /**
     * @brief Derive runtime requirements from exactly the same metadata geometry.
     * @param contract Validated source policy and original K.
     * @param rows Simultaneous physical invocation rows.
     * @return Same named banks as pre-allocation pricing.
     */
    [[nodiscard]] inline WorkspaceRequirements cpuWorkspaceRequirements(const MatrixContract &contract, int rows)
    {
        return cpuWorkspaceRequirements(contract.transform(), contract.columns(), rows);
    }

    /**
     * @brief Borrow only live prefixes of the stage's already admitted CPU banks.
     * @param contract Exact policy used for the declaration and ensuing arithmetic.
     * @param rows Live input rows, possibly smaller than the declared envelope.
     * @param workspace Stage-owned manager; no pointer is retained by this function.
     * @return Typed borrowed spans whose lifetime ends with this invocation.
     * @throws std::runtime_error For missing, foreign, undersized or misaligned storage.
     *
     * Zero work needs no manager or bank. Positive work never allocates,
     * resizes, repairs a binding, or substitutes a thread-local workspace.
     */
    [[nodiscard]] inline CPUActivationBuffers borrowCPUWorkspace(
        const MatrixContract &contract, int rows, DeviceWorkspaceManager *workspace)
    {
        const auto counts = cpuWorkspaceCounts(contract, rows);
        return {
            CPUInvocationWorkspace::require<ActivationBlock>(workspace, kCPUActivationBlocks, counts.blocks),
            CPUInvocationWorkspace::require<ActivationRange>(workspace, kCPUGroupRanges, counts.ranges),
            CPUInvocationWorkspace::require<float>(workspace, kCPUTokenRanges, counts.row_ranges),
            CPUInvocationWorkspace::require<float>(workspace, kCPUButterfly, counts.butterfly_floats),
            CPUInvocationWorkspace::require<float>(workspace, kCPUTransformed, counts.transformed_floats),
            CPUInvocationWorkspace::require<std::uint8_t>(workspace, kCPUActiveRows, counts.active_rows)};
    }
} // namespace llaminar2::w4a4
