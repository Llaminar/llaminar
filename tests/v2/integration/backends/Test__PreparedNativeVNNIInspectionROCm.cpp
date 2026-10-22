/**
 * @file Test__PreparedNativeVNNIInspectionROCm.cpp
 * @brief ROCm-only native all-node proof for passive prepared metadata.
 *
 * This translation unit includes exactly one vendor runtime. The common
 * fixture remains backend-neutral; an inspection failure cannot become a
 * cached zero or a kernel-only empty-graph claim.
 */
#include "backends/rocm/HIPGraphCapture.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "utils/PreparedNativeVNNIInspectionFixture.h"
#include <stdexcept>

namespace llaminar2::test
{
namespace
{
/**
 * @brief Inspect every actual native node, failing on a foreign graph or failed native query.
 * @param graph Original backend-owned retained graph after recording ends.
 * @return Actual native node cardinality, including all nonkernel node types.
 * @throws std::runtime_error For an incompatible graph or native inspection error.
 */
std::size_t checkedNativeNodes(IGPUGraphCapture &graph)
{
    auto *native=dynamic_cast<HIPGraphCapture *>(&graph);
    if(!native) throw std::runtime_error("Passive ROCm inspection received a foreign graph");
    std::size_t nodes=0;
    if(hipGraphGetNodes(native->graph(),nullptr,&nodes)!=hipSuccess)
        throw std::runtime_error("Passive ROCm all-node graph inspection failed");
    return nodes;
}
} // namespace
/** @brief Prepared and unprepared ROCm adapters retain source/state and create zero native nodes. */
TEST(PreparedNativeVNNIInspection, ROCm)
{
    prepared_native_inspection::provePassiveInspection<llaminar2::rocm::ROCmQuantisedGemmKernel>(
        DeviceId::rocm(0),checkedNativeNodes);
}
} // namespace llaminar2::test
