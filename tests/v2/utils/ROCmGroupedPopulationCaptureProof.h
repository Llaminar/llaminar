/**
 * @file ROCmGroupedPopulationCaptureProof.h
 * @brief Authenticate and omit the exclusive GU and fused down population producers.
 *
 * Capture inspection checks immutable decision/directory bindings and complete
 * role inventories. Mutations copy control arguments into the instantiated
 * executable, including every repeated operation. The lease restores the
 * original controls before its borrowed graph retires. Device observations
 * remain outside production and join the exact replay stream first.
 */
#pragma once
#include "backends/rocm/HIPGraphCapture.h"
#include "kernels/common/MoEGroupedDirectory.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedPopulationProjection.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedPopulationWork.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedPopulationPublication.h"
#include <hip/hip_runtime.h>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace llaminar2::test
{
/** @brief Physical arithmetic ownership authenticated from the native specialization. */
enum class PopulationCapturedArithmetic { Mixed, Ordered, Native };
/** @param symbol Actual mangled function name. @param gateup Role from the captured packet.
 * @return Exact compiled arithmetic; throws for an unsupported decoder, role or template ABI. */
inline PopulationCapturedArithmetic capturedPopulationArithmetic(std::string_view symbol,bool gateup)
{
    if (symbol.find("populationProjection") == std::string_view::npos ||
        symbol.find(gateup ? "Format13ELb1E" : "Format24ELb0E") == std::string_view::npos)
        throw std::runtime_error("Population decoder or role changed");
    if (!gateup && symbol.find("PopulationArithmeticE0EE") != std::string_view::npos)
        return PopulationCapturedArithmetic::Mixed;
    if (gateup && symbol.find("PopulationArithmeticE1EE") != std::string_view::npos)
        return PopulationCapturedArithmetic::Ordered;
    if (gateup && symbol.find("PopulationArithmeticE2EE") != std::string_view::npos)
        return PopulationCapturedArithmetic::Native;
    throw std::runtime_error("Population physical arithmetic changed");
}

/** @brief Borrow one complete captured operation and its device decision. */
class ROCmGroupedPopulationCaptureProof final
{
    /** @brief Stable host copies of the captured arguments used for executable mutation. */
    struct Consumer
    {
        hipGraphNode_t node;
        hipKernelNodeParams params;
        rocm::MoEGroupedWmmaArguments projection;
        rocm::MoEGroupedPopulationControl control;
        PopulationCapturedArithmetic arithmetic;
    };
    HIPGraphCapture &graph_;
    hipStream_t stream_;
    std::vector<Consumer> consumers_;
    std::optional<rocm::ROCmMoEGroupedRouteAdmission> admission_;
    bool omitted_ = false;
    /** @param status Native operation result. @throws std::runtime_error On failure. */
    static void require(hipError_t status)
    { if (status != hipSuccess) throw std::runtime_error("Population capture inspection failed"); }
    /** @param params Actual captured ABI. @param index Argument ordinal.
     * @return Immutable argument copy. @throws std::runtime_error On an unsupported ABI. */
    template<class T> static T argument(const hipKernelNodeParams &params,size_t index)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!params.kernelParams || !params.kernelParams[index] || params.extra)
            throw std::runtime_error("Population capture argument ABI is missing");
        std::array<std::byte,sizeof(T)> bytes;
        std::memcpy(bytes.data(),params.kernelParams[index],bytes.size());
        return std::bit_cast<T>(bytes);
    }
    /** @param consumer Stable original launch and current control copy.
     * @return Native executable-parameter update result; HIP copies argument values during this call. */
    hipError_t apply(Consumer &consumer) noexcept
    {
        auto params = consumer.params;
        void *arguments[]{&consumer.projection,&consumer.control};
        params.kernelParams = arguments;
        params.extra = nullptr;
        return hipGraphExecKernelNodeSetParams(graph_.executable(),consumer.node,&params);
    }
public:
    /** @param graph Instantiated production parent. @param stream Its exact execution stream.
     * @param domain Independently expected admission (Qwen or Unmeasured).
     * @param selection Arena-owned decision. @param counts Actual expert counts.
     * @param directory Actual role directory bank. @param experts Exact expert capacity.
     * @param repeats Complete repeated operations.
     * @throws std::runtime_error For missing, extra or incorrectly bound producers/consumers. */
    ROCmGroupedPopulationCaptureProof(HIPGraphCapture &graph,hipStream_t stream,
        rocm::MoEGroupedPopulationDomain domain,uint32_t *selection,const int *counts,
        const uint32_t *directory,int experts,int repeats)
        : graph_(graph),stream_(stream)
    {
        using namespace rocm;
        if (!graph.graph() || !graph.executable() || !stream ||
            stream == hipStreamLegacy || stream == hipStreamPerThread || !selection || !counts || !directory ||
            (domain != MoEGroupedPopulationDomain::Qwen && domain != MoEGroupedPopulationDomain::Unmeasured))
            throw std::runtime_error("Population proof requires complete capture bindings");
        size_t size = 0;
        require(hipGraphGetNodes(graph.graph(),nullptr,&size));
        std::vector<hipGraphNode_t> nodes(size);
        require(hipGraphGetNodes(graph.graph(),nodes.data(),&size));
        std::array<int,2> roles{}, directories{}, ordered{};
        int classifiers = 0;
        std::array<int,2> gateup_arithmetic{};
        const bool admitted = domain == MoEGroupedPopulationDomain::Qwen;
        for (auto node : nodes)
        {
            hipGraphNodeType type;
            require(hipGraphNodeGetType(node,&type));
            if (type != hipGraphNodeTypeKernel) continue;
            hipKernelNodeParams params{};
            require(hipGraphKernelNodeGetParams(node,&params));
            const char *name = hipKernelNameRefByPtr(params.func,stream);
            if (!name) throw std::runtime_error("Population proof has an unnamed kernel");
            const std::string_view symbol(name);
            if (symbol.find("moe_grouped_build_population_directory_kernel") != std::string_view::npos)
            {
                const bool gu = symbol.find("ProjectionRoleE0") != std::string_view::npos;
                const bool down = symbol.find("ProjectionRoleE1") != std::string_view::npos;
                if (!admitted || gu == down || argument<const int *>(params,0) != counts ||
                    argument<uint32_t *>(params,1) != directory || argument<int>(params,2) != experts ||
                    (gu ? argument<MoEGroupedPopulationPublisher>(params,5).address() :
                        argument<MoEGroupedPopulationBinding>(params,5).address()) != selection ||
                    params.gridDim.x != 1 || params.gridDim.y != 1 || params.gridDim.z != 1 ||
                    params.blockDim.x != moe_grouped_directory::directoryThreads(experts) ||
                    params.blockDim.y != 1 || params.blockDim.z != 1)
                    throw std::runtime_error("Population directory bindings changed");
                if (gu)
                {
                    admission_ = argument<ROCmMoEGroupedRouteAdmission>(params,4);
                    ++classifiers;
                }
                ++directories[gu ? 0 : 1];
            }
            else if (symbol.find("moe_grouped_population_select_kernel") != std::string_view::npos)
                throw std::runtime_error("Automatic retained a retired standalone classifier");
            else if (symbol.find("grouped_wmma_detail") != std::string_view::npos)
            {
                if (!admitted || symbol.find("populationProjection") == std::string_view::npos)
                    throw std::runtime_error("Automatic retained a separate native family");
                const auto projection = argument<MoEGroupedWmmaArguments>(params,0);
                const auto control = argument<MoEGroupedPopulationControl>(params,1);
                const bool gu = projection.second_descriptors != nullptr;
                if (control.binding.address() != selection || control.omitted != MoEGroupedPopulationFamily::Invalid ||
                    projection.branch_suppression != MoEGroupedBranchSuppression::None ||
                    projection.group_counts != counts || projection.directory != directory || projection.experts != experts ||
                    projection.rows != 512 || projection.top_k != 8 || projection.n != (gu ? 512 : 2048) ||
                    projection.k != (gu ? 2048 : 512) || params.blockDim.x != 128 || params.blockDim.y != 1 ||
                    params.blockDim.z != 1 || params.gridDim.x != kMoEGroupedPopulationWorkers ||
                    params.gridDim.y != 1 ||
                    params.gridDim.z != (gu ? 2u : 1u) ||
                    projection.dot_admission.rowPlan().floor() != MoEGroupedWmmaRowFloor::Sixteen)
                    throw std::runtime_error("Population fused consumer identity changed");
                const auto arithmetic = capturedPopulationArithmetic(symbol,gu);
                unsigned enabled = 0;
                require(hipGraphNodeGetEnabled(graph_.executable(),node,&enabled));
                if (enabled != 1) throw std::runtime_error("Population consumer starts disabled");
                if (gu) ++gateup_arithmetic[arithmetic == PopulationCapturedArithmetic::Native ? 1 : 0];
                consumers_.push_back({node,params,projection,control,arithmetic});
                ++roles[gu ? 0 : 1];
            }
            else if (symbol.find("moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel") != std::string_view::npos)
                ++ordered[0];
            else if (symbol.find("moe_grouped_down_batch_invariant_tiled_prefill_kernel") != std::string_view::npos)
                ++ordered[1];
        }
        const int expected = admitted ? repeats : 0;
        if (classifiers != expected || roles != std::array<int,2>{2*expected,expected} ||
            directories != std::array<int,2>{expected,expected} || gateup_arithmetic != std::array<int,2>{expected,expected} ||
            (admitted ? ordered != std::array<int,2>{0,0} : (ordered[0] < repeats || ordered[1] < repeats)))
            throw std::runtime_error("Population retained producer/consumer inventory is incomplete");
    }
    /** @brief Restore borrowed executable controls before graph destruction. */
    ~ROCmGroupedPopulationCaptureProof() { restore(); }
    ROCmGroupedPopulationCaptureProof(const ROCmGroupedPopulationCaptureProof &) = delete;
    ROCmGroupedPopulationCaptureProof &operator=(const ROCmGroupedPopulationCaptureProof &) = delete;
    /** @param role Actual projection role whose production packet is required.
     * @return Copied immutable bindings for direct submission boundary regressions.
     * @throws std::runtime_error For an unsupported or absent captured role. */
    [[nodiscard]] std::pair<rocm::MoEGroupedWmmaArguments,rocm::MoEGroupedPopulationControl>
    capturedRoleBindings(rocm::MoEPrefillProjectionRole role) const
    {
        if (role != rocm::MoEPrefillProjectionRole::GateUp && role != rocm::MoEPrefillProjectionRole::Down)
            throw std::runtime_error("Invalid population projection role");
        for (const auto &consumer : consumers_)
            if ((consumer.projection.second_descriptors != nullptr) == (role == rocm::MoEPrefillProjectionRole::GateUp))
                return {consumer.projection,consumer.control};
        throw std::runtime_error("Missing captured population role");
    }
    /** @param slots Fixture-known actual route count. @return Whether tiled families own this replay. */
    [[nodiscard]] bool tiled(int slots) const
    { return admission_ && admission_->strategyForSlots(slots) == rocm::ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled; }
    /** @param family Logical arithmetic choice. @return Whether the shared consumer can execute that choice. */
    [[nodiscard]] bool contains(rocm::MoEGroupedPopulationFamily family) const
    { return family < rocm::MoEGroupedPopulationFamily::Invalid && !consumers_.empty(); }
    /** @param family Suppressed logical choice in every repeated shared GU/down consumer.
     * @throws std::runtime_error For an absent choice, active omission, or native mutation failure. */
    void omit(rocm::MoEGroupedPopulationFamily family)
    {
        if (omitted_ || !contains(family)) throw std::runtime_error("Invalid population family omission");
        omitted_ = true;
        for (auto &consumer : consumers_) { consumer.control.omitted = family; require(apply(consumer)); }
    }
    /** @param branch Physical GU arithmetic producer disabled; unaffected experts retain their intermediates.
     * @throws std::runtime_error For unsupported suppression or an already active mutation. */
    void omitArithmetic(rocm::MoEGroupedBranchSuppression branch)
    {
        if (omitted_ || consumers_.empty() || (branch != rocm::MoEGroupedBranchSuppression::Ordered &&
            branch != rocm::MoEGroupedBranchSuppression::Wmma))
            throw std::runtime_error("Invalid fused arithmetic omission");
        omitted_ = true;
        const auto arithmetic = branch == rocm::MoEGroupedBranchSuppression::Ordered
            ? PopulationCapturedArithmetic::Ordered : PopulationCapturedArithmetic::Native;
        for (auto &consumer : consumers_)
            if (consumer.arithmetic == arithmetic)
                require(hipGraphNodeSetEnabled(graph_.executable(),consumer.node,0));
    }
    /** @brief Drain the exact stream and restore all controls, including partial mutation failures. */
    void restore() noexcept
    {
        if (!omitted_) return;
        if (hipStreamSynchronize(stream_) != hipSuccess) std::terminate();
        for (auto &consumer : consumers_)
        {
            consumer.control.omitted = rocm::MoEGroupedPopulationFamily::Invalid;
            consumer.projection.branch_suppression = rocm::MoEGroupedBranchSuppression::None;
            if (apply(consumer) != hipSuccess ||
                hipGraphNodeSetEnabled(graph_.executable(),consumer.node,1) != hipSuccess) std::terminate();
        }
        omitted_ = false;
    }
};
}
