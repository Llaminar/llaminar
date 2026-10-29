/**
 * @file CPUGroupedMoEStorageLayout.h
 * @brief Allocation-free declaration of every grouped CPU expert scratch root.
 *
 * The serial model envelope, physical BOM and runtime workspace consume this
 * same typed value. It has no execution/configuration dependencies or live
 * ledger; CPUStorageRoot owns exact allocator geometry and materialization.
 */
#pragma once
#include "memory/CPUStorageRoot.h"
#include "CPURouterQ8StorageLayout.h"
#include "tensors/BlockStructures.h"
#include <limits>
#include <algorithm>
#include <initializer_list>
namespace llaminar2
{
/** @brief Complete physical roots and authenticated source geometry for grouped MoE. */
struct CPUGroupedMoEStorageLayout
{
    CPUStorageRoot<float> batch, gate, up, output, route_outputs;
    CPUStorageRoot<Q8_1Block> router_q8, swiglu_q8, kernel_router_q8;
    size_t route_capacity = 0; ///< Logical maximum routes, independent of rounding.
    size_t row_capacity = 0; ///< Exact maximum compact rows.
    int d_model = 0; ///< Original hidden width.
    int expert_intermediate = 0; ///< Original expert width.
    int num_experts = 0; ///< Exact logical expert geometry.
    int routing_top_k = 0; ///< Exact maximum route width.
    bool operator==(const CPUGroupedMoEStorageLayout &) const = default;
    /**
     * @brief Reject partial or manually inconsistent root geometry before publication.
     * @throws std::invalid_argument For invalid source geometry or noncanonical roots.
     * @throws std::overflow_error For an unrepresentable canonical extent.
     */
    void validate() const
    {
        const auto canonical=declare(row_capacity,d_model,expert_intermediate,num_experts,routing_top_k,batch.allocator(),batch.placement(),batch.initialization());
        if(*this!=canonical)
            throw std::invalid_argument("CPU grouped-MoE declaration is not its canonical complete eight-root layout");
    }
    /** @brief Merge only row capacity within one exact source/allocation domain.
     * @param other Complete declaration from another serial graph participant.
     * @return Canonical maximum-row declaration; all other source policies agree.
     * @throws std::invalid_argument For incomplete or incompatible source policies.
     * @throws std::overflow_error For an unrepresentable canonical extent.
     */
    [[nodiscard]] CPUGroupedMoEStorageLayout merge(const CPUGroupedMoEStorageLayout &other) const
    {
        validate();other.validate();
        if(d_model!=other.d_model || expert_intermediate!=other.expert_intermediate ||
            num_experts!=other.num_experts || routing_top_k!=other.routing_top_k ||
            batch.allocator()!=other.batch.allocator() || batch.placement()!=other.batch.placement() ||
            batch.initialization()!=other.batch.initialization())
            throw std::invalid_argument("CPU grouped-MoE serial participants have incompatible source geometry or allocator");
        return declare(std::max(row_capacity,other.row_capacity),d_model,expert_intermediate,
            num_experts,routing_top_k,batch.allocator(),batch.placement(),batch.initialization());
    }
    /** @brief Authenticate a late request against an already frozen physical envelope.
     * @param requested Exact participant declaration; only its row count may be smaller.
     * @return Whether it fits without allocation, growth or rebinding.
     * @throws std::invalid_argument For incomplete or incompatible source policies.
     * @throws std::overflow_error For an unrepresentable canonical extent.
     */
    [[nodiscard]] bool contains(const CPUGroupedMoEStorageLayout &requested) const
    {
        return merge(requested)==*this;
    }

    /** @return The canonical publication root embedded in this grouped envelope. */
    [[nodiscard]] CPURouterQ8StorageLayout routerLayout() const
    {
        validate();
        return CPURouterQ8StorageLayout::declare(row_capacity, d_model,
            batch.allocator(), batch.placement(), batch.initialization());
    }

    /** @return Checked sum of all eight declared physical backing extents. */
    size_t physicalBytes() const
    {
        size_t total = 0;
        for (const size_t bytes : {batch.physicalBytes(), gate.physicalBytes(), up.physicalBytes(),
             output.physicalBytes(), route_outputs.physicalBytes(), router_q8.physicalBytes(),
             swiglu_q8.physicalBytes(), kernel_router_q8.physicalBytes()})
        {
            if (bytes > std::numeric_limits<size_t>::max() - total)
                throw std::overflow_error("CPU grouped-MoE root byte sum overflows");
            total += bytes;
        }
        return total;
    }


    /** @brief Declare complete roots from exact CPU geometry without allocating.
     * @param rows Maximum compact input rows.
     * @param hidden_width Exact hidden/output width.
     * @param intermediate_width Exact expert SwiGLU width.
     * @param experts Logical expert-table width.
     * @param top_k Maximum routes per row.
     * @param device Exact admitted CPU allocator.
     * @param placement Exact shared physical placement policy.
     * @param initialization Exact shared producer initialization policy.
     * @return Structurally comparable immutable-root declaration.
     * @throws std::invalid_argument For missing or invalid geometry.
     * @throws std::overflow_error For an unrepresentable root extent.
     */
    static CPUGroupedMoEStorageLayout declare(size_t rows, int hidden_width,
        int intermediate_width, int experts, int top_k, DeviceId device,
        CPUStoragePlacement placement=CPUStoragePlacement::local(),
        CPUStorageInitialization initialization=CPUStorageInitialization::Zero)
    {
        if (!device.is_cpu() || rows == 0 || hidden_width <= 0 ||
            intermediate_width <= 0 || experts <= 0 ||
            top_k <= 0 || top_k > experts)
            throw std::invalid_argument("CPU grouped-MoE workspace requires exact CPU allocator and positive geometry");
        const auto multiply = [](size_t a, size_t b)
        {
            if (a && b > std::numeric_limits<size_t>::max() / a)
                throw std::overflow_error("CPU grouped-MoE root extent overflows");
            return a * b;
        };
        const size_t routes = multiply(rows, static_cast<size_t>(top_k));
        if (routes > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("CPU grouped-MoE workspace route capacity exceeds int");
        const size_t hidden = static_cast<size_t>(hidden_width);
        const size_t intermediate = static_cast<size_t>(intermediate_width);
        const size_t hidden_blocks = (hidden + Q8_1Block::BLOCK_SIZE - 1u) / Q8_1Block::BLOCK_SIZE;
        const size_t intermediate_blocks = (intermediate + Q8_1Block::BLOCK_SIZE - 1u) / Q8_1Block::BLOCK_SIZE;
        const auto fp32 = [&](size_t count) { return CPUStorageRoot<float>(
            count, PhysicalMemoryOwner::ExecutionWorkspace, device,placement,initialization); };
        const auto q8 = [&](size_t count) { return CPUStorageRoot<Q8_1Block>(
            count, PhysicalMemoryOwner::ExecutionWorkspace, device,placement,initialization); };
        return {fp32(multiply(rows, hidden)), fp32(multiply(routes, intermediate)),
            fp32(multiply(routes, intermediate)), fp32(multiply(routes, hidden)),
            fp32(multiply(routes, std::max(hidden, intermediate))), q8(multiply(routes, hidden_blocks)),
            q8(multiply(routes, intermediate_blocks)),
            CPURouterQ8StorageLayout::declare(rows, hidden_width, device, placement, initialization).root, routes,
            rows, hidden_width, intermediate_width, experts, top_k};
    }

};
}
