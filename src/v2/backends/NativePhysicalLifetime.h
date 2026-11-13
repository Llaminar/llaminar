/**
 * @file NativePhysicalLifetime.h
 * @brief Source-certified physical lifetime shared by native context and auxiliary opaque roots.
 *
 * A successful handle destructor or context reset does not imply process-pool
 * reclamation. Only closed source contracts select this lifetime; no caller
 * can convert a process obligation into an early-release reservation.
 */
#pragma once
namespace llaminar2 {
/** @brief Actual physical reclamation boundary, independent from handle/execution lifetime. */
enum class NativePhysicalLifetime {ContextGeneration,ProcessRuntime};
}
