/**
 * @file CPUProjectionWorkspaceContract.h
 * @brief Canonical CPU projection scratch requirements before and after preparation.
 *
 * Metadata and prepared engines contribute the same named buffers. This pure
 * contract has no device discovery, allocations, capacity decisions or ledger;
 * callers supply the participant's published execution geometry and contribute
 * the result to PhysicalMemoryAuthority before materialization.
 */
#pragma once
#include "backends/CPUExecutionGeometry.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "CPUNativeVNNIPreparedFootprint.h"
#include "tensors/TensorType.h"
#include <string_view>

namespace llaminar2
{
    /** @brief Complete execution geometry for one independently named projection. */
    struct CPUProjectionWorkspaceGeometry
    {
        int rows = 0, n = 0, k = 0; ///< Physical simultaneous matrix dimensions.
        int serial_n = 0; ///< Zero means physical N; mirrored heads retain their serial policy width.
        int workers = 0; ///< Admitted workshare size, not local planner hardware.
        CPUExecutionGeometry execution;
        CPUProjectionNumericalPolicy numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
    };

    /** @brief Single named-buffer declaration for CPU projection engines and metadata BOMs. */
    class CPUProjectionWorkspaceContract final
    {
    public:
        /**
         * @brief Declare a source-native, unrotated projection without loading weights.
         * @param format Canonical source format, including F32/F16/BF16.
         * @param geometry Participant and physical matrix geometry.
         * @return Exact named Q8/partial envelope; floating projections require neither.
         * @throws std::invalid_argument For unknown formats or incomplete geometry.
         */
        static WorkspaceRequirements sourceNative(std::string_view format,
            const CPUProjectionWorkspaceGeometry &geometry);
        /**
         * @brief Adapt a loader's typed source identity to the same metadata contract.
         * @param format Original tensor format, not the packed execution codebook.
         * @param geometry Participant and physical matrix geometry.
         * @return The same named buffers as the canonical GGUF-name overload.
         * @throws std::invalid_argument For unsupported formats or invalid geometry.
         */
        static WorkspaceRequirements sourceNative(TensorType format,
            const CPUProjectionWorkspaceGeometry &geometry);
        /**
         * @brief Declare the same bank from an actually prepared encoding.
         * @param footprint Live prepared encoding, including optional rotated encoding.
         * @param codebook Arithmetic source identity retained by the prepared engine.
         * @param geometry Exact dimensions and participant policy observation.
         * @return Checked named Q8 and ordered-partial envelope.
         * @throws std::invalid_argument For incomplete matrix, cache, ISA or worker geometry.
         */
        static WorkspaceRequirements prepared(const cpu::native_vnni::NativeVNNIPreparedFootprint &footprint,
            std::uint8_t codebook, const CPUProjectionWorkspaceGeometry &geometry);
    };
}
