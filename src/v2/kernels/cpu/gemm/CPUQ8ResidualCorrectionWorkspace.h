/**
 * @file CPUQ8ResidualCorrectionWorkspace.h
 * @brief Shared named CPU two-term activation banks for planning and execution.
 *
 * Both the metadata BOM and the native prepared engine call this factory. It
 * describes two simultaneous Q8 operands and neither allocates storage nor
 * introduces a second admission balance.
 */
#pragma once
#include "kernels/common/Q8ResidualCorrection.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
namespace llaminar2 {
/**
 * @brief Describe the exact primary and residual native CPU activation banks.
 * @param m Admitted physical matrix rows, independent of the live prefix.
 * @param n Native output width, checked even though no output scratch is needed.
 * @param k Native reduction width, in complete32-value blocks.
 * @return Existing primary bank and one distinct residual bank with64-byte alignment.
 * @throws std::exception For unsupported dimensions or unrepresentable storage.
 */
inline WorkspaceRequirements cpuQ8ResidualCorrectionWorkspaceRequirements(int m,int n,int k)
{
    validateQ8ResidualCorrectionGeometry(m,n,k);
    auto result=cpuProjectionQ8WorkspaceRequirements(m,k);
    result.buffers.push_back({kCPUProjectionResidualQ8,
        cpuProjectionQ8Blocks(m,k)*sizeof(Q8_1Block),64});
    return result;
}
} // namespace llaminar2
