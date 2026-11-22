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
        int routed_width; ///< Actual selected-expert fragment width admitted for this participant.
        RetainedM1FFNWorkspace routed_workspace;
        RetainedM1FFNWorkspace shared_workspace;
        int8_t* hidden_codes; ///< Original prepared K32 activation banks for local Gate/Up.
        float* hidden_scales;
        float* shared_gate_factor; ///< Original rounded input gate before ordered output publication.
        float* routed_contributions; ///< Original [route][hidden] values for the explicit graph join.
        float* shared_contribution; ///< Original participant-local shared output.
        RetainedM1RoutePublication route_publication;
        RetainedM1SharedPublication shared_publication;
        std::uint64_t* early_packet_ready = nullptr; ///< Arena word released after the HC row and route packet are complete.
    };

    static_assert(std::is_trivially_copyable_v<RetainedM1HCFFN>);

    /**
     * @brief Resolve the exact original QFN CUDA entry before native resource admission.
     * @param gate Prepared routed Gate and Up execution codebook.
     * @param down Prepared routed Down execution codebook.
     * @param demand Original graph reads of retained HC intermediates.
     * @param observe Original graph's runtime histogram producer authorization.
     * @return Exact cooperative entry, or null when the declared source has no compiled program.
     */
    const void* retainedQFNM1Function(uint8_t gate, uint8_t down, FullNativeHCFFNOutputDemand demand,
        bool observe);

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
         * @return A resident program on the measured SM86 geometry.
         */
        static RetainedM1Program prepare(DeviceId device, const RetainedM1HCFFN& arguments,
            uint8_t gate, uint8_t down, FullNativeHCFFNOutputDemand demand, bool observe,
            uint64_t workspace_generation);

        /**
         * @brief Submit the resident program on its exact stream or record its counted positive branch.
         * @param stream Owning stream with the original environment and producer edges.
         * @param workspace_generation Current generation of the original prepared arena.
         * @throws std::exception For changed generation, invalid stream, or incomplete native capture.
         */
        void launch(void* stream, uint64_t workspace_generation) const;

        /** @return Measured resource admission of the exact loaded entry. */
        const NativeCooperativeKernelAdmission& admission() const noexcept { return admission_; }

    private:
        /** @brief Bind a successful native admission to its checked operands and grid. */
        RetainedM1Program(NativeCooperativeKernelAdmission admission, RetainedM1HCFFN arguments, unsigned blocks)
            : admission_(admission), arguments_(arguments), blocks_(blocks) {}

        NativeCooperativeKernelAdmission admission_;
        RetainedM1HCFFN arguments_;
        unsigned blocks_; ///< Measured resident work width on the owning GPU.
    };
}
