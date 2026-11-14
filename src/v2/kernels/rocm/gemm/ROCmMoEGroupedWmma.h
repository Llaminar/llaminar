/**
 * @file ROCmMoEGroupedWmma.h
 * @brief Native matrix leaves for the retained grouped MoE projection transaction.
 *
 * Dot-engine recording is independent of the device-owned route strategy. WMMA
 * replaces only ExpertTiled gate/up and down leaves; quantization, routing,
 * SwiGLU and ordered publication retain their existing owners and allocations.
 */
#pragma once
#include "ROCmMoEPrefillArithmetic.h"
#include "ROCmMoEGroupedPrefillRoutePolicy.h"

namespace llaminar2::rocm
{
/** @brief Explicit recording choices; Automatic retains installed measured policy. */
enum class MoEGroupedPrefillEngine { Automatic, OrderedDot4, WmmaM16N32, WmmaM16N64 };

/** @brief Nestable worker-local selection, embedded in the captured native nodes. */
class ScopedMoEGroupedPrefillEngine final
{
public:
    /** @brief Set the recording engine. @param engine Explicit candidate or installed policy. */
    explicit ScopedMoEGroupedPrefillEngine(MoEGroupedPrefillEngine engine);
    /** @brief Restore the enclosing recording policy without changing retained graphs. */
    ~ScopedMoEGroupedPrefillEngine();
    ScopedMoEGroupedPrefillEngine(const ScopedMoEGroupedPrefillEngine &) = delete;
    ScopedMoEGroupedPrefillEngine &operator=(const ScopedMoEGroupedPrefillEngine &) = delete;
private:
    MoEGroupedPrefillEngine previous_;
};

/** @brief Resolve recording against immutable native hardware metadata.
 * @param device Visible HIP ordinal owning the complete transaction.
 * @return Concrete Dot4 or native WMMA engine; never Automatic.
 * @throws std::invalid_argument When an explicit native candidate is unsupported. */
MoEGroupedPrefillEngine selectedMoEGroupedPrefillEngine(int device);

/** @brief Which authoritative row index addresses the quantized activation bank. */
enum class MoEGroupedWmmaInput { OriginalTokens, OriginalRouteSlots, GroupedRows };

/** @brief Frozen projection bindings; live extents come only from the group plan. */
struct MoEGroupedWmmaArguments
{
    const int8_t *activations = nullptr;
    const float *activation_scales = nullptr;
    const int *group_token_indices = nullptr;
    const int *group_counts = nullptr;
    const int *group_offsets = nullptr;
    const uint32_t *directory = nullptr;
    const DeviceNativeQuantisedMatrixDesc *descriptors = nullptr;
    const DeviceNativeQuantisedMatrixDesc *second_descriptors = nullptr;
    float *output = nullptr;
    float *second_output = nullptr;
    int directory_capacity = 0;
    int rows = 0;
    int top_k = 0;
    int n = 0;
    int k = 0;
    int experts = 0;
    int device = -1;
    MoEGroupedWmmaInput input = MoEGroupedWmmaInput::GroupedRows;
    MoEPrefillSerialM1KPartitionPolicy partitions;
    ROCmMoEGroupedRouteAdmission admission; ///< Required existing group-plan authority.
};

/** @brief Record all physical-codebook leaves for one gate/up or down projection.
 * @param engine Concrete native tile selected for this device at recording time.
 * @param codebooks Actual physical descriptor mask, preserving separate IQ4 ids.
 * @param args Frozen bindings and original full-width arithmetic partition policy.
 * @param stream Explicit non-default producer stream owned by args.device.
 * @return Whether every required native launch was accepted. */
bool submitMoEGroupedWmma(MoEGroupedPrefillEngine engine, uint32_t codebooks,
                         const MoEGroupedWmmaArguments &args, void *stream);

/** @brief Publish this TU's immutable IQ grids during backend initialization.
 * @param device Owning native device. @param iq3s IQ3_S uint32 table.
 * @param iq3xxs IQ3_XXS uint32 table. @param iq2s IQ2_S uint64 table.
 * @param iq2xs IQ2_XS uint64 table. @param iq2xxs IQ2_XXS uint64 table.
 * @param iq1s Shared IQ1 uint64 table. @return Whether all tables were published. */
bool initializeMoEGroupedWmmaGrids(int device, const void *iq3s, const void *iq3xxs,
    const void *iq2s, const void *iq2xs, const void *iq2xxs, const void *iq1s);
}
