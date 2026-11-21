/**
 * @file ROCmGroupedBranchMutation.h
 * @brief Test-only omission of one arithmetic branch in actual retained MoE nodes.
 *
 * Normal production admission rejects every suppression value except None.
 * This lease changes only that immutable argument in an already instantiated
 * adaptive projection role. It retains both argument packets, joins the exact
 * capture stream and restores every successful patch before local storage dies.
 * Imported timing-parent nodes are inspected directly, never their source graph.
 */
#pragma once

#include "backends/rocm/HIPGraphCapture.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedWmma.h"
#include <array>
#include <bit>
#include <cstddef>
#include <type_traits>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test
{
/** @brief Authenticate immutable row plans in the actual retained wide projection nodes.
 * @param capture Instantiated source or imported parent graph. @param stream Exact capture stream.
 * @param floor Expected whole-expert cutoff. @param layout Expected original directory encoding.
 * @return Gate/up and down node counts, in that order.
 * @throws std::runtime_error For malformed captures or symbol/packet/geometry disagreement. */
inline std::array<size_t,2> inspectROCmGroupedWideCapture(HIPGraphCapture &capture, hipStream_t stream,
    rocm::MoEGroupedWmmaRowFloor floor, MoEGroupingPackedLayout layout=MoEGroupingPackedLayout::WideFloor8)
{
    const auto require = [](hipError_t status) {
        if (status != hipSuccess) throw std::runtime_error("Wide capture inspection failed");
    };
    if (!capture.graph() || !capture.executable() || !stream ||
        stream == hipStreamLegacy || stream == hipStreamPerThread)
        throw std::runtime_error("Wide inspection requires an instantiated graph and exact stream");
    size_t count = 0;
    require(hipGraphGetNodes(capture.graph(),nullptr,&count));
    std::vector<hipGraphNode_t> nodes(count);
    require(hipGraphGetNodes(capture.graph(),nodes.data(),&count));
    std::array<size_t,2> roles{};
    for (const auto node : nodes)
    {
        hipGraphNodeType type;
        require(hipGraphNodeGetType(node,&type));
        if (type != hipGraphNodeTypeKernel) continue;
        hipKernelNodeParams params{};
        require(hipGraphKernelNodeGetParams(node,&params));
        const char *name = hipKernelNameRefByPtr(params.func,stream);
        if (!name) throw std::runtime_error("Wide capture node has no symbol");
        const std::string_view symbol(name);
        if (symbol.find("grouped_wmma_detail") == std::string_view::npos) continue;
        if (symbol.find("wideProjection") == std::string_view::npos ||
            (params.gridDim.z != 1 && params.gridDim.z != 2) ||
            !params.kernelParams || !params.kernelParams[0] || params.extra)
            throw std::runtime_error("Wide capture has an incompatible native producer");
        using Arguments = rocm::MoEGroupedWmmaArguments;
        std::array<std::byte,sizeof(Arguments)> bytes;
        std::memcpy(bytes.data(),params.kernelParams[0],bytes.size());
        const auto args = std::bit_cast<Arguments>(bytes);
        const bool gate_up = params.gridDim.z == 2;
        const auto role = gate_up ? rocm::MoEPrefillProjectionRole::GateUp : rocm::MoEPrefillProjectionRole::Down;
        const unsigned threads = gate_up ? 256u : 128u;
        const unsigned native_columns = threads / 4;
        const bool columns=layout==MoEGroupingPackedLayout::ColumnFloor8;
        const auto plan=columns ? rocm::MoEGroupedRowPlan::column(role) : rocm::MoEGroupedRowPlan::wide(role,floor);
        const unsigned launch_columns=columns ? threads : native_columns;
        const std::string expected_suffix=std::string(gate_up ? "ELb1" : "ELb0")+
            (args.dot_admission.mayUseOrdered() ? "ELb1" : "ELb0")+(columns ? "ELb1EE" : "ELb0EE");
        if (symbol.find(expected_suffix)==std::string_view::npos ||
            args.dot_admission.rowPlan() != plan || args.directory_layout!=layout ||
            !args.dot_admission.binds(role,args.n,args.k) ||
            (args.second_descriptors != nullptr) != gate_up || (args.second_output != nullptr) != gate_up ||
            args.branch_suppression != rocm::MoEGroupedBranchSuppression::None ||
            params.blockDim.x != threads || params.blockDim.y != 1 || params.blockDim.z != 1 ||
            params.gridDim.x != (args.n+launch_columns-1)/launch_columns ||
            params.gridDim.y != static_cast<unsigned>(args.directory_capacity))
            throw std::runtime_error("Wide capture row plan or launch geometry changed");
        ++roles[gate_up ? 0 : 1];
    }
    return roles;
}

/** @brief Own a temporary branch omission on every repeated node of one adaptive role. */
class ROCmGroupedBranchMutation final
{
    /** @brief Stable original/fault packets and the unchanged native launch geometry. */
    struct Patch
    {
        hipGraphNode_t node = nullptr;
        hipKernelNodeParams params{};
        rocm::MoEGroupedWmmaArguments original, fault;
        std::array<void *,1> original_parameters{}, fault_parameters{};
        bool applied = false;
    };
    hipGraphExec_t executable_;
    hipStream_t stream_;
    std::vector<Patch> patches_;

    /** @param result Native API outcome. @param operation Cold test operation name.
     * @throws std::runtime_error If native inspection or argument mutation failed. */
    static void require(hipError_t result, const char *operation)
    {
        if (result != hipSuccess)
            throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
    }

public:
    /** @brief Inspect actual executable nodes before changing any argument.
     * @param capture Instantiated graph, including an imported timing parent.
     * @param stream Exact non-default stream retained by this test graph.
     * @param role Projection role whose arithmetic branch is omitted; defaults to down.
     * @throws std::runtime_error For missing adaptive nodes or incompatible argument packets. */
    ROCmGroupedBranchMutation(HIPGraphCapture &capture, hipStream_t stream,
        rocm::MoEPrefillProjectionRole role = rocm::MoEPrefillProjectionRole::Down)
        : executable_(capture.executable()), stream_(stream)
    {
        if ((role != rocm::MoEPrefillProjectionRole::GateUp && role != rocm::MoEPrefillProjectionRole::Down) ||
            !capture.graph() || !executable_ || !stream ||
            stream == hipStreamLegacy || stream == hipStreamPerThread)
            throw std::runtime_error("Grouped branch mutation requires a retained graph and exact stream");
        size_t count = 0;
        require(hipGraphGetNodes(capture.graph(),nullptr,&count),"count grouped nodes");
        std::vector<hipGraphNode_t> nodes(count);
        require(hipGraphGetNodes(capture.graph(),nodes.data(),&count),"inspect grouped nodes");
        patches_.reserve(count);
        const bool gate_up = role == rocm::MoEPrefillProjectionRole::GateUp;
        for (auto node : nodes)
        {
            hipGraphNodeType type;
            require(hipGraphNodeGetType(node,&type),"inspect grouped node type");
            if (type != hipGraphNodeTypeKernel) continue;
            hipKernelNodeParams params{};
            require(hipGraphKernelNodeGetParams(node,&params),"inspect grouped kernel arguments");
            const char *name = hipKernelNameRefByPtr(params.func,stream_);
            if (!name) throw std::runtime_error("Grouped node has no native symbol");
            const std::string_view symbol(name);
            if (symbol.find("grouped_wmma_detail") == std::string_view::npos || params.gridDim.z != (gate_up ? 2u : 1u))
                continue;
            const bool wide = symbol.find("wideProjection") != std::string_view::npos;
            // Ordinary leaves encode role/adaptive; wide leaves additionally
            // encode column ownership. Authenticate that last bit against the packet.
            const std::string role_suffix=gate_up ? "ELb1ELb1" : "ELb0ELb1";
            if ((!wide && symbol.find(role_suffix+"EE")==std::string_view::npos) ||
                (wide && symbol.find(role_suffix+"ELb")==std::string_view::npos) ||
                !params.kernelParams || !params.kernelParams[0] || params.extra)
                throw std::runtime_error("Branch mutation requires an actual adaptive projection node");
            using Arguments = rocm::MoEGroupedWmmaArguments;
            static_assert(std::is_trivially_copyable_v<Arguments>);
            std::array<std::byte,sizeof(Arguments)> bytes;
            std::memcpy(bytes.data(),params.kernelParams[0],bytes.size());
            const auto original = std::bit_cast<Arguments>(bytes);
            if (wide && symbol.find(role_suffix+(original.dot_admission.rowPlan().isColumn() ? "ELb1EE" : "ELb0EE"))==std::string_view::npos)
                throw std::runtime_error("Grouped column template and row plan disagree");
            if ((!wide && symbol.find("10projection") == std::string_view::npos) ||
                original.dot_admission.rowPlan().isWide() != wide ||
                original.dot_admission.rowPlan().role() != role ||
                !original.dot_admission.binds(role,original.n,original.k))
                throw std::runtime_error("Grouped symbol and captured row plan disagree");
            Patch patch{.node=node,.params=params,.original=original,.fault=original};
            if (patch.original.branch_suppression != rocm::MoEGroupedBranchSuppression::None ||
                !patch.original.dot_admission.mayUseOrdered() ||
                (patch.original.second_descriptors != nullptr) != gate_up ||
                (patch.original.second_output != nullptr) != gate_up ||
                (!gate_up && patch.original.input != rocm::MoEGroupedWmmaInput::GroupedRows))
                throw std::runtime_error("Grouped argument packet is incompatible with branch omission");
            patch.fault = patch.original;
            patches_.push_back(patch);
        }
        if (patches_.empty()) throw std::runtime_error("No adaptive writers of the requested role were captured");
        // The vector is complete before any pointer borrows its argument storage.
        for (auto &patch : patches_)
        {
            patch.original_parameters[0] = &patch.original;
            patch.fault_parameters[0] = &patch.fault;
        }
    }

    /** @brief Join observations and restore every patched node on all test exits. */
    ~ROCmGroupedBranchMutation() { restore(); }
    ROCmGroupedBranchMutation(const ROCmGroupedBranchMutation &) = delete;
    ROCmGroupedBranchMutation &operator=(const ROCmGroupedBranchMutation &) = delete;

    /** @return Number of patched candidates, including repeated imported operations. */
    [[nodiscard]] size_t nodeCount() const noexcept { return patches_.size(); }

    /** @brief Omit one selected producer while retaining all other launch arguments.
     * @param branch Ordered or Wmma; None is reserved for restoration.
     * @throws std::runtime_error For invalid state or failed native mutation.
     * The enclosing lease already exists, so a partial failure is restored. */
    void suppress(rocm::MoEGroupedBranchSuppression branch)
    {
        if (branch != rocm::MoEGroupedBranchSuppression::Ordered &&
            branch != rocm::MoEGroupedBranchSuppression::Wmma)
            throw std::runtime_error("A branch omission must select one arithmetic producer");
        for (auto &patch : patches_)
        {
            if (patch.applied) throw std::runtime_error("Restore the previous branch omission first");
            patch.fault.branch_suppression = branch;
            auto params = patch.params;
            params.kernelParams = patch.fault_parameters.data();
            require(hipGraphExecKernelNodeSetParams(executable_,patch.node,&params),"omit grouped arithmetic branch");
            patch.applied = true;
        }
    }

    /** @brief Drain the exact stream and restore original packets; failure is fatal. */
    void restore() noexcept
    {
        bool applied = false;
        for (const auto &patch : patches_) applied |= patch.applied;
        if (!applied) return;
        if (hipStreamSynchronize(stream_) != hipSuccess) std::terminate();
        for (auto &patch : patches_)
        {
            if (!patch.applied) continue;
            auto params = patch.params;
            params.kernelParams = patch.original_parameters.data();
            if (hipGraphExecKernelNodeSetParams(executable_,patch.node,&params) != hipSuccess)
                std::terminate();
            patch.applied = false;
        }
    }
};
}
