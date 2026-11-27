/**
 * @file CUDARetainedHCFFN.h
 * @brief Original CUDA post-attention HC, router and local FFN launch operands.
 *
 * The acquired runtime bank owns expert storage. The resident program publishes
 * its original routed and shared contribution buffers, and the existing graph
 * owns their ordered merge and transport to other participants.
 */
#pragma once

#include "CUDARetainedHCPrefix.h"
#include "execution/config/RoutedExpertPolicy.h"
#include "kernels/common/FullNativeHCFFNOutputDemand.h"
#include "kernels/common/NativeCooperativeKernel.h"
#include <functional>
#include <optional>

namespace llaminar2
{
    struct DeviceMoELayerRuntime;
}

namespace llaminar2::cuda::moe
{
    /** @brief Router operands from the original participant-local declaration and epoch. */
    struct RetainedM1Router
    {
        DeviceMoELayerRuntime* runtime; ///< Device-owned acquired runtime view, including selected route slots.
        const uint16_t* weights; ///< Original FP16 expert rows in [expert][hidden] order.
        float* logits; ///< Original router working values and probabilities.
        float* route_indices; ///< Original complete route indices exported to the ticket producer.
        float* route_weights; ///< Original complete route weights exported to the ticket producer.
        const int32_t* position; ///< Original absolute position for replicated-route assignment.
        int experts; ///< Admitted original source expert count.
        int top_k; ///< Exact original selected route count.
        RoutedExpertRowExecutionPolicy local_policy; ///< Graph-declared ownership for this row.
        bool normalize; ///< Original selected-probability normalization.
    };

    /** @brief Prepared native descriptors of one original expert and its local feed-forward width. */
    struct RetainedM1Expert
    {
        DeviceNativeQuantisedMatrixDesc gate;
        DeviceNativeQuantisedMatrixDesc up;
        DeviceNativeQuantisedMatrixDesc down;
    };

    /** @brief Prebound canonical Down operands published by local Gate/Up and SwiGLU. */
    struct RetainedM1FFNWorkspace
    {
        int8_t* down_codes; ///< Complete original Q8 SwiGLU packets.
        float* down_scales; ///< Original rounded scale of each published packet.
    };

    /** @brief Local publications are raw or preweighted according to their existing graph consumers. */
    enum class RetainedM1RoutePublication : uint8_t { Raw, Preweighted };

    /** @brief The participant publishes the shared partial before its graph collective or gates a complete row. */
    enum class RetainedM1SharedPublication : uint8_t { PartialBeforeCollective, GatedComplete };

    /** @brief Arguments of one admitted cooperative CUDA program, read by value during capture. */
    struct RetainedM1HCFFN
    {
        RetainedHCPrefix hc;
        RetainedM1Router router;
        const float* shared_gate_input; ///< Original H-wide input-gating weights.
        RetainedM1Expert shared; ///< Original immutable shared FFN triplet for this participant.
        RetainedM1Expert* selected; ///< Prepared descriptors indexed by compact local route.
        int* route_order; ///< Live and canonical counts followed by original slots, grouped by physical decoder.
        int* route_plan_ready; ///< Device-scope publication from the selector CTA to routed math CTAs.
        int routed_width; ///< Actual selected-expert fragment width admitted for this participant.
        RetainedM1FFNWorkspace routed_workspace;
        RetainedM1FFNWorkspace shared_workspace;
        int8_t* hidden_codes; ///< Original prepared K32 activation banks for local Gate/Up.
        float* hidden_scales;
        float* shared_gate_factor; ///< Original rounded input gate before ordered output publication.
        float* routed_contributions; ///< Original [route][hidden] values for the explicit graph join.
        float* shared_contribution; ///< Original participant-local shared output.
        float* suffix_partials; ///< L2-resident split-K Gate/Up partition partials for device-sized suffix.
        uint32_t* suffix_done; ///< Per-packet completion counters reset by RoutePrefix.
        void* suffix_gate_up_node; ///< Opaque cudaGraphDeviceNode_t for device-sized SuffixGateUp.
        void* suffix_down_node; ///< Opaque cudaGraphDeviceNode_t for device-sized SuffixDown.
        uint64_t* profile_ns = nullptr; ///< Optional diagnostic globaltimer slots.
        RetainedM1RoutePublication route_publication;
        RetainedM1SharedPublication shared_publication;
        uint8_t gate_codebook = 0; ///< Admitted canonical Gate/Up physical codebook.
        uint8_t down_codebook = 0; ///< Admitted canonical Down physical codebook.
        uint8_t gate_promoted_codebook = 0xffu; ///< Optional dynamic promotion codebook for Gate/Up (0xff = none).
        uint8_t down_promoted_codebook = 0xffu; ///< Optional dynamic promotion codebook for Down (0xff = none).
        uint8_t gate_source_superblock = 0; ///< Expected source superblock provenance when promoted.
        uint8_t down_source_superblock = 0; ///< Expected source superblock provenance when promoted.
        uint8_t suffix_resident_loop = 0; ///< Non-zero when suffix runs as cloneable resident grid-stride (no device-updatable handles).
    };

    static_assert(std::is_trivially_copyable_v<RetainedM1HCFFN>);

    /**
     * @brief Resolve the exact original QFN CUDA entry before native resource admission.
     * @param gate Prepared routed Gate and Up execution codebook.
     * @param down Prepared routed Down execution codebook.
     * @param demand Original graph reads of retained HC intermediates.
     * @param observe Original graph's runtime histogram producer authorization.
     * @param role Original placement policy resolved by the graph before residency admission.
     * @return Exact cooperative entry, or null when the declared source has no compiled program.
     */
    const void* retainedQFNM1Function(uint8_t gate, uint8_t down, FullNativeHCFFNOutputDemand demand,
        bool observe, RoutedExpertRowExecutionPolicy role);

    /** @brief Lookup sharded 1D Gate/Up suffix entry in Shard 0 (standard/legacy/K-quant + ExpandedInt8Min). */
    const void* retainedQFNM1SuffixGateUpShard0(uint8_t codebook) noexcept;
    /** @brief Lookup sharded 1D Gate/Up suffix entry in Shard 1 (IQ codebooks + IQ4XSSourceScale). */
    const void* retainedQFNM1SuffixGateUpShard1(uint8_t codebook) noexcept;
    /** @brief Lookup sharded 1D Down suffix entry in Shard 0 (standard/legacy/K-quant + ExpandedInt8Min). */
    const void* retainedQFNM1SuffixDownShard0(uint8_t codebook) noexcept;
    /** @brief Lookup sharded 1D Down suffix entry in Shard 1 (IQ codebooks + IQ4XSSourceScale). */
    const void* retainedQFNM1SuffixDownShard1(uint8_t codebook) noexcept;

    /** @return Whether the sharded all-codebook NativeCompletion suffix covers both physical codebooks. */
    [[nodiscard]] inline bool retainedQFNM1NativeCompletionSupported(uint8_t gate, uint8_t down) noexcept
    {
        const bool has_gate = retainedQFNM1SuffixGateUpShard0(gate) != nullptr ||
                              retainedQFNM1SuffixGateUpShard1(gate) != nullptr;
        const bool has_down = retainedQFNM1SuffixDownShard0(down) != nullptr ||
                              retainedQFNM1SuffixDownShard1(down) != nullptr;
        return has_gate && has_down;
    }

    /** @brief Ordering selected from the original graph's early packet consumers. */
    enum class RetainedM1PacketOrdering { CompleteProgram, NativeCompletion };
    /** @brief One always-live half-warp of shared Down columns for each legacy cooperative suffix CTA. */
    inline constexpr unsigned retained_shared_down_frontier_columns = 16;
    /** @brief Non-cooperative device-sized Gate/Up suffix launch width (10 warps = 2 partitions x 5 K-blocks). */
    inline constexpr unsigned retained_suffix_gate_up_threads = 320;
    /** @brief Split-K CTAs per 32-column Gate/Up packet (16 partitions / 2 partitions per CTA = 8 CTAs). */
    inline constexpr unsigned retained_suffix_gate_up_k_splits = 8;
    /** @brief Non-cooperative device-sized Down suffix launch width (10 half-warps x 16 cols = 160 threads). */
    inline constexpr unsigned retained_suffix_down_threads = 160;


    /** @brief Loaded resident entry, sealed geometry and all permanent participant bindings. */
    class RetainedM1Program final
    {
    public:
        /**
         * @brief Seal the exact CUDA entry and resource envelope outside capture.
         * @param device Original owning CUDA participant.
         * @param arguments Original prepared arena operands and publications.
         * @param gate Physical routed Gate/Up source codebook.
         * @param down Physical routed Down source codebook.
         * @param demand Graph's actual external HC observation roles.
         * @param observe Whether the original router publishes selected/local demand.
         * @param workspace_generation Owner generation of every borrowed workspace plane.
         * @param ordering Whether the original graph declares independent consumers of the completed route prefix.
         * @return A resident program on the measured SM86 geometry.
         */
        static RetainedM1Program prepare(DeviceId device, const RetainedM1HCFFN& arguments,
            uint8_t gate, uint8_t down, FullNativeHCFFNOutputDemand demand, bool observe,
            uint64_t workspace_generation, RetainedM1PacketOrdering ordering = RetainedM1PacketOrdering::CompleteProgram);

        /**
         * @brief Submit the resident program on its exact stream or record its counted positive branch.
         * @param stream Owning stream with the original environment and producer edges.
         * @param workspace_generation Current generation of the original prepared arena.
         * @param record_ready Construction-only binder called after recording the prefix node.
         * @throws std::exception For changed generation, invalid stream, or incomplete native capture.
         */
        void launch(void* stream, uint64_t workspace_generation,
            const std::function<void()>& record_ready = {}) const;

        /** @return Measured resource admission of the exact loaded entry. */
        const NativeCooperativeKernelAdmission& admission() const noexcept { return admission_; }

    private:
        /** @brief Bind a successful native admission to its checked operands and grid. */
        RetainedM1Program(NativeCooperativeKernelAdmission admission, RetainedM1HCFFN arguments, unsigned blocks,
            std::optional<NativeCooperativeKernelAdmission> route_prefix, unsigned prefix_blocks,
            const void* route_plan = nullptr, const void* suffix_gate_up = nullptr, const void* suffix_down = nullptr)
            : admission_(admission), arguments_(arguments), blocks_(blocks), route_prefix_(route_prefix),
              prefix_blocks_(prefix_blocks), route_plan_(route_plan),
              suffix_gate_up_(suffix_gate_up), suffix_down_(suffix_down) {}

        NativeCooperativeKernelAdmission admission_;
        RetainedM1HCFFN arguments_;
        unsigned blocks_; ///< Measured resident work width on the owning GPU.
        std::optional<NativeCooperativeKernelAdmission> route_prefix_; ///< Sealed native HC+Router cooperative phase.
        unsigned prefix_blocks_; ///< Original always-live HC+Router CTA extent.
        const void* route_plan_ = nullptr; ///< Spill-free 1-CTA TopK + descriptor + device-grid planner.
        const void* suffix_gate_up_ = nullptr; ///< Spill-free device-sized SuffixGateUp entry.
        const void* suffix_down_ = nullptr; ///< Spill-free device-sized SuffixDown entry.
    };
}
