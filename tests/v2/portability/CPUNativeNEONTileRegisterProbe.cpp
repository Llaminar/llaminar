/**
 * @file CPUNativeNEONTileRegisterProbe.cpp
 * @brief Build-time register-allocation witness for every native NEON projection tile program.
 *
 * Explicit instantiations pull in every row tile (1..4), every compiled
 * integer family (ASIMD, SDOT and, in an I8MM envelope, SMMLA row pairs),
 * both arithmetic policies and every non-compact encoding. The build compiles
 * this TU with the real target's command at -O3 and CPUNativeTileRegisterEvidence.py
 * rejects any FP/SIMD stack traffic inside the K-loop programs and their call
 * closure: the tiles must keep their whole working set in registers. The
 * inventory string names exactly the instantiations below (same guards), so
 * the checker can require every expected program body.
 */
#include "kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"

namespace llaminar2::cpu::native_quantised
{
template void CPUNativeNEONProjectionTile<1>::execute<multi_scale::NEONByteDot::ASIMD>() const;
template void CPUNativeNEONProjectionTile<2>::execute<multi_scale::NEONByteDot::ASIMD>() const;
template void CPUNativeNEONProjectionTile<3>::execute<multi_scale::NEONByteDot::ASIMD>() const;
template void CPUNativeNEONProjectionTile<4>::execute<multi_scale::NEONByteDot::ASIMD>() const;
#if defined(__ARM_FEATURE_DOTPROD)
template void CPUNativeNEONProjectionTile<1>::execute<multi_scale::NEONByteDot::DotProd>() const;
template void CPUNativeNEONProjectionTile<2>::execute<multi_scale::NEONByteDot::DotProd>() const;
template void CPUNativeNEONProjectionTile<3>::execute<multi_scale::NEONByteDot::DotProd>() const;
template void CPUNativeNEONProjectionTile<4>::execute<multi_scale::NEONByteDot::DotProd>() const;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
template void CPUNativeNEONProjectionTile<2>::execute<multi_scale::NEONByteDot::I8MM>() const;
template void CPUNativeNEONProjectionTile<3>::execute<multi_scale::NEONByteDot::I8MM>() const;
template void CPUNativeNEONProjectionTile<4>::execute<multi_scale::NEONByteDot::I8MM>() const;
#endif
} // namespace llaminar2::cpu::native_quantised

/// Instantiation inventory read by CPUNativeTileRegisterEvidence.py; mirrors the guards above.
extern "C" const char llaminar_native_neon_tile_probe_inventory[] = "llaminar-native-neon-tile-probe:"
    "1:ASIMD,2:ASIMD,3:ASIMD,4:ASIMD"
#if defined(__ARM_FEATURE_DOTPROD)
    ",1:DotProd,2:DotProd,3:DotProd,4:DotProd"
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    ",2:I8MM,3:I8MM,4:I8MM"
#endif
    ;
