/**
 * @file ROCmGroupedFFNFrontier.h
 * @brief Typed MI50 M4 publication for an original prepared local FFN.
 */
#pragma once
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "kernels/common/DeviceRowRange.h"
#include "execution/moe/MoEGroupedPlanDemand.h"
#include "backends/DeviceId.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cstddef>
#include <cstdint>

namespace llaminar2 {
class TensorBase;
class ITensor;
struct DeviceMoELayerRuntime;
namespace rocm::grouped_ffn {
inline constexpr int capacity=4, hidden=2560, routedWidth=128, sharedWidth=160, topK=10, experts=512;
inline constexpr std::size_t sharedCodesBytes=capacity*sharedWidth;
inline constexpr std::size_t sharedScalesBytes=capacity*(sharedWidth/32)*sizeof(float);
inline constexpr std::size_t sharedGateBytes=capacity*sizeof(float);
/** @brief Cold-selected routed width for the exact Q4_K/Q5_1 stage. */
enum class RoutedGeometry : int { F128=128, F160=160 };
inline constexpr int routedColumns(RoutedGeometry geometry) { return static_cast<int>(geometry); }

/** @brief Exact immutable reachable routed decoder pair. */
enum class RoutedDecoder { Q4K_Q51, Q4K_Q80, Q5K_Q80 };

/** @brief Identify the measured physical formats that benefit from K32 sums. */
constexpr bool supportsExactK32Sums(RoutedDecoder decoder) noexcept
{
    return decoder==RoutedDecoder::Q4K_Q51 || decoder==RoutedDecoder::Q5K_Q80;
}
constexpr unsigned routedGateCodebook(RoutedDecoder decoder)
{
    return decoder==RoutedDecoder::Q5K_Q80?7u:5u;
}
constexpr unsigned routedDownCodebook(RoutedDecoder decoder)
{
    return decoder==RoutedDecoder::Q4K_Q51?7u:19u;
}

inline constexpr std::size_t bankIdentityBytes=3*sizeof(std::uint64_t);

/** @brief Disjoint typed tail in the already admitted, serial frontier input scratch.
 * The shared SwiGLU codes occupy [0,640). Router and gate publishers write only their own
 * live K32 sums through the same graph stream, before either consumer can read them.
 */
struct ExactK32Sums {
    std::int16_t hiddenRows[capacity*(hidden/32)];
    std::int16_t routedRows[capacity*topK*(160/32)];
};
static_assert(sharedCodesBytes % alignof(ExactK32Sums)==0);
inline constexpr auto k32SumWorkspaceBytes=sharedCodesBytes+sizeof(ExactK32Sums);

/** @brief Consume a scoped opt-in during graph construction, before capture identity seals. */
inline bool exactK32SumsOptIn()
{
    static const bool enabled=[] {
        const char* mode=std::getenv("LLAMINAR_QFN_ROCM_GROUPED_FFN_AFFINE_SUMS");
        if(!mode || std::strcmp(mode,"0")==0)return false;
        if(std::strcmp(mode,"1")==0)return true;
        throw std::invalid_argument("LLAMINAR_QFN_ROCM_GROUPED_FFN_AFFINE_SUMS accepts 0 or 1");
    }();
    return enabled;
}

/** @brief Form the disjoint typed sum view after the caller validates its lease extent. */
inline ExactK32Sums* exactK32Sums(void* scratch)
{
    return reinterpret_cast<ExactK32Sums*>(static_cast<std::byte*>(scratch)+sharedCodesBytes);
}


/** @brief Router publication handed directly to the qualified two-node transaction. */
struct DirectBankRouter {
    ITensor* indices=nullptr;
    ITensor* weights=nullptr;
    MoEGroupedPlanDemand demand=MoEGroupedPlanDemand::None;
};

/** @brief Original graph-owned source and arena operands, separate from mutable route state. */
struct LocalPublication {
    DeviceRowRange rows;
    std::array<DeviceNativeQuantisedMatrixDesc,3> sharedWeights; ///< Fixed shared gate, up, down.
    TensorBase* sharedInputGate;
    TensorBase* sharedGateScratch;
    TensorBase* sharedUpScratch;
    TensorBase* routedOutput;
    TensorBase* sharedOutput;
    TensorBase* combined;
    DirectBankRouter directRouter{};
    RoutedGeometry routedGeometry=RoutedGeometry::F128;
    RoutedDecoder routedDecoder=RoutedDecoder::Q4K_Q51;
};

/** @brief Canonical original mutable plan views from the complete grouped-plan owner. */
struct RoutePlan {
    const std::int32_t* live;
    const std::int32_t* originalToGrouped;
    const std::int32_t* originalExpertIds;
    const float* groupWeights;
    const std::int8_t* hiddenCodes;
    const float* hiddenScales;
    const float* hiddenFP32;
    const DeviceNativeQuantisedMatrixDesc* routedGate;
    const DeviceNativeQuantisedMatrixDesc* routedUp;
    const DeviceNativeQuantisedMatrixDesc* routedDown;
};

/** @brief Exact runtime, routing inputs, and device workspace for the acquired bank identity. */
struct DirectBankInput {
    DeviceMoELayerRuntime* runtime=nullptr;
    const float* indices=nullptr;
    const float* weights=nullptr;
    MoEGroupedPlanDemand demand=MoEGroupedPlanDemand::None;
    std::uint64_t* bankIdentity=nullptr;
    ExactK32Sums* affineSums=nullptr; ///< Same stream and graph workspace as the two-role scratch.
};

/** @brief Every native publication is bound to the same original stream and workspace lifetime. */
struct Output {
    std::int8_t* routedCodes;
    float* routedScales;
    std::int8_t* sharedCodes; ///< The original prefill input scratch is dead when router Q8 is selected.
    float* sharedScales; ///< The corresponding original input scale scratch.
    float* gateValues; ///< The original shared gate scratch.
    const float* sharedInputGate;
    float* sharedGateScratch;
    float* sharedUpScratch;
    float* routedOutput;
    float* sharedOutput;
    float* combined;
    float* routeContributions;
};

/**
 * @brief Enqueue the measured paired gate/up publisher and ordered token-local down.
 * @param plan Original device-owned count, compaction and descriptor views.
 * @param sharedWeights Original admitted immutable shared descriptor triple.
 * @param output All original accumulators plus explicitly admitted workspace.
 * @param device Exact admitted ROCm ordinal.
 * @param stream Exact non-default stream following the original plan.
 * @return True when both native submissions succeed.
 */
bool launch(const RoutePlan& plan,const std::array<DeviceNativeQuantisedMatrixDesc,3>& sharedWeights,
            const Output& output,int device,void* stream,const DirectBankInput& direct={},
            RoutedGeometry geometry=RoutedGeometry::F128,RoutedDecoder decoder=RoutedDecoder::Q4K_Q51);

} // namespace rocm::grouped_ffn
} // namespace llaminar2
