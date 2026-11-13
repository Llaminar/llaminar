/**
 * @file PooledImmutableSourceRootBinding.h
 * @brief Register only original participant-local roles for a pooled source graph.
 *
 * A source always owns coordinator and source-protocol roots. Its declared
 * physical cache owners may all be remote, including when exactly one exists.
 * The checked geometry alone determines whether a local cache role is present;
 * original registration callbacks retain their existing arena/owner checks.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableSourceGeometry.h"
namespace llaminar2::qwen4exp::detail {
/**
 * @brief Bind the original source graph's exact existing root roles.
 * @param geometry Original complete physical/source declaration.
 * @param local Register the original local owner at its canonical coordinate, only when present.
 * @param coordinator Register the original coordinator roots.
 * @param source Register the original source roots.
 * @throws std::exception Propagates original role registration failures without selecting another path.
 * @note This selects declared roles only; callbacks retain their original validation and failure semantics.
 */
template<class Local,class Coordinator,class Source>
void bindOriginalPooledSourceGraphRoots(const PooledImmutableSourceGeometry& geometry,
    Local&& local,Coordinator&& coordinator,Source&& source){
    const auto coordinate=geometry.parameters().local_owner;
    if(coordinate!=adaptive_lookup::None)local(coordinate);
    coordinator();
    source();
}
}
