/**
 * @file ROCmPersistentHCRead.h
 * @brief Borrowed original HC-read phases for one admitted native cooperative kernel.
 *
 * Preparation consumes original prepared engines and an existing admitted
 * workspace/tensor binding. The graph owner retains their physical leases;
 * this ABI allocates no buffer, descriptor table, live ledger or argument pool.
 * Public norm/down/activation/logits/mixed and optional injection values remain
 * independently observable. Cross-device collectives and HC write stay outside.
 */
#pragma once
#include "kernels/common/FullNativeHCFFNOutputDemand.h"
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include "kernels/common/HCDownOutputPublicationArgs.h"
#include "kernels/common/NativeCooperativeKernel.h"
#include <cstdint>
#include <type_traits>
namespace llaminar2 {class ITensorGemm;}
namespace llaminar2::rocm {
/** @brief Existing normalized-input boundary; no residual write is requested. */
struct PersistentHCNoWrite {};
/**
 * @brief Original previous-write operands, borrowed for one fused write/norm phase.
 * @details The cold owner must prove that output is exactly norm.input and that
 * write and norm have the same original device row authority. No row count or
 * allocation is inferred here. Every pointer remains an original public root.
 */
struct PersistentHCWriteInput {
 const float* residual=nullptr; ///< Original retained four-branch residual.
 const float* block=nullptr; ///< Original completed H-wide block result.
 const float* gateLogits=nullptr; ///< Original four injection logits.
 float* output=nullptr; ///< Original rounded write, exactly the following norm input.
};
static_assert(std::is_trivially_copyable_v<PersistentHCWriteInput>);
/**
 * @brief Physical issue geometry without changing original arithmetic tasks.
 * @tparam Threads Exact admitted CTA size; standalone256 or composed1024.
 * @details Every contributor authors a complete original K32 contribution. The
 * same32-cell window is folded in increasing K order, independently of threads.
 * Host admission and the device body consume this single geometry authority.
 */
template<unsigned Threads> struct PersistentHCReadGeometry {
 static_assert(Threads==256||Threads==1024,"HC prefix requires its declared256/1024-thread program");
 static constexpr unsigned threads=Threads;
 static constexpr unsigned down_columns=Threads==1024?32:64;
 static constexpr unsigned down_contributors=Threads/down_columns;
 static constexpr unsigned down_window=32;
 static constexpr unsigned quant_cohorts=Threads/32;
 static constexpr unsigned up_channels=8;
};
/** @brief Existing original row-major FP32 injection; absent pointers omit the independent phase. */
struct PersistentHCInjectionArgs {
 const float* weights=nullptr; ///< Original contiguousFP32 prepared4-by-4H source.
 float* output=nullptr; ///< Original exported four-logit allocation, not a private stash.
 DeviceRowRange rows=DeviceRowRange::fullyActive(1); ///< Original projection authority, independent of norm/up rows.
 int load_pairs=8; ///< Cold original FP32 load policy; numerical chain remains stride256.
};
/** @brief Explicit cold grid economy experiment; neither policy changes numerical tasks. */
enum class PersistentHCGridPolicy : unsigned char { OnePerProcessor, MaximumResident };
/** @brief Trivially copied participant-local argument binding, with no mutable host execution state. */
struct PersistentHCReadArgs {
 qwen4exp::BranchRMSNormArgs norm; ///< Original256-lane FMA/tree/gain and retained FP32 output.
 DeviceNativeQuantisedMatrixDesc down_weights; ///< Exact original prepared signed-Q8 source.
 std::int8_t* input_codes=nullptr; ///< Existing QUANT_A bank, reused only after down retires.
 float* input_scales=nullptr; ///< Existing SCALES_A_BLOCKWISE bank.
 float* down_partials=nullptr; ///< Existing original[P][R] scratch, publicly diagnosable.
 float* raw_down=nullptr; ///< Original raw R output; activated may exactly alias it.
 GPUHCDownOutputView publication; ///< Original /4-SiLU/two-term Q8 destinations and publisher rows.
 GPUQ8ResidualCorrectionArgs up; ///< Original up source, query rows, Q8 planes and public logits.
 float* mixed=nullptr; ///< Original H-wide branch read destination.
 PersistentHCInjectionArgs injection; ///< Optional original independent normalized-input projection.
 int partitions=0; ///< Cold source-resolved fixed partition tree; not an occupancy choice.
 PersistentHCGridPolicy grid_policy=PersistentHCGridPolicy::OnePerProcessor; ///< Named immutable construction policy.
 unsigned resident_grid=0; ///< Cold workload-sized grid bounded by exact native occupancy.
};
static_assert(std::is_trivially_copyable_v<PersistentHCReadArgs>);
/**
 * @brief Export exact source-owned descriptors through existing prepared-engine APIs.
 * @param args Existing physical tensor/workspace bindings; receives source descriptors only.
 * @param down Original immutable prepared HC down engine.
 * @param up Original immutable prepared two-term HC up engine.
 * @param injection Optional original prepared FP32 four-row engine, never inferred from raw source type.
 * @throws std::invalid_argument For absent/unprepared/incompatible source; no conversion or alternate path.
 */
void resolvePersistentHCReadSources(PersistentHCReadArgs& args,const ITensorGemm& down,
 const ITensorGemm& up,const ITensorGemm* injection=nullptr);
/**
 * @brief Resolve and validate original HC arithmetic without admitting a native function.
 * @param args Original source and physical bindings; receives canonical split-K partitions.
 * @param output_demand Full-region external publications; standalone entries require every original output.
 * @details Standalone and composed entries share this cold source authority, then
 * admit their own actual functions. This routine neither allocates nor launches.
 * @throws std::invalid_argument For incompatible original bindings or explicit arithmetic overrides.
 */
void resolvePersistentHCReadPolicy(PersistentHCReadArgs& args,
 FullNativeHCFFNOutputDemand output_demand=FullNativeHCFFNOutputDemand::all());
/**
 * @brief Validate original arithmetic/bindings and admit one exact native cooperative program cold.
 * @param device Exact worker participant.
 * @param args Borrowed complete HC binding; receives original partitions and admitted live-task grid.
 * @param generation Existing arena/workspace program generation, nonzero and retained by graph owner.
 * @return Exact function/resource/occupancy admission; zero-private/spill policy remains mandatory.
 */
NativeCooperativeKernelAdmission preparePersistentHCRead(DeviceId device,PersistentHCReadArgs& args,std::uint64_t generation);
/** @brief Enqueue exactly the admitted kernel on its exact non-default stream; no queries, allocations or waits.
 * @param admission Original cold exact-program capability.
 * @param args Original frozen physical binding, including its admitted grid.
 * @param stream Exact prepared participant stream.
 * @param generation Original retained program/workspace generation.
 */
void launchPersistentHCRead(const NativeCooperativeKernelAdmission& admission,const PersistentHCReadArgs& args,
 void* stream,std::uint64_t generation);
}
