/**
 * @file Q8ResidualCorrection.h
 * @brief Explicit two-term Q8 projection arithmetic over unchanged prepared weights.
 *
 * The first term is the ordinary Q8 representation. The second quantizes the
 * FP32 residual against that exact stored representation. Independent native
 * integer dots retain ascending K-block folds and are added once at output.
 * Scratch is instance-owned and contributed to the existing workspace/PMA BOM;
 * no residual FP32 matrix, output matrix or new weight copy is required.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 */
#pragma once
#include "QuantizedActivationOperandPrecision.h"
#include "DeviceRowRange.h"
#include "DeviceNativeQuantisedMatrixDesc.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include <limits>
#include <optional>
#include <memory>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
namespace llaminar2 {
class TensorBase;
class HCDownOutputPublication;
struct NativeEmbeddingIQTables;
/** @brief Explicit GPU-only short-K projection/read epilogue, retaining gate logits.
 * The immutable normalized branches remain borrowed from their original arena;
 * mixed output is separate. Neither operand changes the Q8 publication policy.
 */
struct HCResidualShortKReadOutput {
 const TensorBase& normalized; ///< Existing original four normalized branches.
 TensorBase& mixed; ///< Existing mixed output, with independent captured lifetime.
};
/** @brief Original HC bottleneck values and the retained result of /4 SiLU.
 * @details References prohibit missing operands. Engines validate native FP32
 * geometry, exact alias or disjoint storage, and ownership before submission.
 * The activated tensor is authored even when it aliases raw exactly.
 */
struct HCResidualDownInput {
 const TensorBase& raw; ///< Existing down projection, with no new representation.
 TensorBase& activated; ///< Original FP32 activated arena output.
 std::optional<HCResidualShortKReadOutput> short_k_read; ///< Explicit exact-grid M1 operation; absent preserves original projection.
 std::shared_ptr<const HCDownOutputPublication> prepublished; ///< Required original reducer-produced banks; no second publisher.
};

/** @brief Stable participant-owned second activation bank for CPU native blocks. */
inline const std::string kCPUProjectionResidualQ8="cpu_projection_residual_q8";
/** @brief Stable GPU secondary signed-byte bank; primary uses the normal input bank. */
inline const std::string kGPUProjectionResidualQ8="quant_a_residual_q8";
/** @brief GPU FP32 sidecar for each exact backend-native secondary block scale. */
inline const std::string kGPUProjectionResidualQ8Scales="scales_a_residual_q8";
/** @brief Validate source-independent admitted matrix extent, without reading counts.
 * @param m Physical retained row capacity.
 * @param n Native output width; the final SIMD tile may be partial.
 * @param k Native reduction width in complete32-value blocks.
 * @throws std::invalid_argument For unsupported/incomplete physical geometry.
 */
inline void validateQ8ResidualCorrectionGeometry(int m,int n,int k) {
 if(m<=0||n<=0||k<=0||k%32||k>std::numeric_limits<int>::max()-31||
    static_cast<std::size_t>(m)>std::numeric_limits<std::size_t>::max()/sizeof(float)/static_cast<unsigned>(n)||
    static_cast<std::size_t>(m)>std::numeric_limits<std::size_t>::max()/sizeof(float)/static_cast<unsigned>(k))
  throw std::invalid_argument("Two-term Q8 requires positive output width and complete32-reduction native blocks and representable storage");
}
/**
 * @brief Authenticate an original quantized format without changing its prepared bytes.
 * @param format Original GGUF format, resolved by the canonical NativeQuantised catalog.
 * @param m Positive physical activation-row capacity.
 * @param n Actual participant-local output width.
 * @param k Complete prepared K32 reduction width.
 * @return Immutable original source metadata, including asymmetric and superblock identity.
 * @throws std::invalid_argument For an uncataloged source or invalid matrix geometry.
 * @note Prepared source slices can end at native K32 boundaries even when their
 * original source uses K256 blocks. The source loader owns that original block
 * validation; this operation must not reject a legal already-prepared slice.
 */
inline const NativeQuantisedFormatInfo &requireQ8ResidualSourceFormat(
 std::string_view format,int m,int n,int k) {
 validateQ8ResidualCorrectionGeometry(m,n,k);
 const auto *source=native_quantised_formats::forQuantType(format);
 if(!source)throw std::invalid_argument("Two-term Q8 requires a cataloged original native quantized weight format");
 return *source;
}
/**
 * @brief Authenticate the original provenance retained by a prepared engine.
 * @param identity Original codebook, source-block family and explicit presence.
 * @param m Positive physical activation-row capacity.
 * @param n Actual participant-local output width.
 * @param k Complete prepared K32 reduction width.
 * @return Canonical source metadata; physical decoder validation remains backend-owned.
 * @throws std::invalid_argument For missing/foreign provenance or invalid geometry.
 */
inline const NativeQuantisedFormatInfo &requireQ8ResidualSourceFormat(
 NativeQuantisedSourceIdentity identity,int m,int n,int k) {
 validateQ8ResidualCorrectionGeometry(m,n,k);
 const auto *source=identity.present ? native_quantised_formats::forSourceIdentity(
  identity.codebook_id,identity.is_superblock) : nullptr;
 if(!source)throw std::invalid_argument("Two-term Q8 requires complete original native weight provenance");
 return *source;
}
/** @brief Reject tensors smaller than the immutable admitted physical operation.
 * @param m Physical row capacity, independent of the current live count.
 * @param n Native output width.
 * @param k Native reduction width.
 * @param inputElements Actual input tensor element capacity.
 * @param outputElements Actual output tensor element capacity.
 * @throws std::invalid_argument Before any publication if an operand is undersized.
 */
inline void validateQ8ResidualTensorCapacities(int m,int n,int k,
 std::size_t inputElements,std::size_t outputElements) {
 validateQ8ResidualCorrectionGeometry(m,n,k);
 if(inputElements<static_cast<std::size_t>(m)*k||outputElements<static_cast<std::size_t>(m)*n)
  throw std::invalid_argument("Two-term Q8 tensor capacity is smaller than its admitted physical geometry");
}
/** @brief A checked native byte interval; zero bytes explicitly represent an absent optional read. */
struct Q8ResidualStorageSpan {
 const void* data; ///< Actual prepared address; never a second allocation authority.
 std::size_t bytes; ///< Exact consumed physical interval for this invocation.
 std::size_t alignment=1; ///< Required native load/store alignment.
};
/** @brief Reject overflowing, misaligned or overlapping writable intervals before publication.
 * @param writes Exact output and activation banks, each with one exclusive writer.
 * @param reads Immutable input, source weight planes and optional live-count word.
 * Read/read overlap is harmless. Every write must be disjoint from every other
 * consumed byte, including an interior alias with a different starting pointer.
 * Address comparisons inspect metadata only; no device pointer is dereferenced.
 */
inline void validateQ8ResidualStorage(std::span<const Q8ResidualStorageSpan> writes,
 std::span<const Q8ResidualStorageSpan> reads) {
 const auto checked=[](const Q8ResidualStorageSpan&s) {
  const auto begin=reinterpret_cast<std::uintptr_t>(s.data);
  if(s.bytes&&(!s.data||!s.alignment||begin%s.alignment||
     s.bytes>std::numeric_limits<std::uintptr_t>::max()-begin))
   throw std::invalid_argument("Two-term Q8 storage has an invalid, overflowing or misaligned byte interval");
 };
 const auto overlaps=[](const Q8ResidualStorageSpan&a,const Q8ResidualStorageSpan&b) {
  if(!a.bytes||!b.bytes)return false;
  const auto ab=reinterpret_cast<std::uintptr_t>(a.data),bb=reinterpret_cast<std::uintptr_t>(b.data);
  return ab<bb+b.bytes&&bb<ab+a.bytes;
 };
 for(const auto&s:writes)checked(s);
 for(const auto&s:reads)checked(s);
 for(std::size_t i=0;i<writes.size();++i) {
  for(std::size_t j=i+1;j<writes.size();++j)if(overlaps(writes[i],writes[j]))
   throw std::invalid_argument("Two-term Q8 writable byte intervals overlap");
  for(const auto&r:reads)if(overlaps(writes[i],r))
   throw std::invalid_argument("Two-term Q8 output or activation bank overlaps a consumed immutable input");
 }
}
/** @brief Admit the same complete prefix authority as original HC primitives.
 * @param rows Original physical/live prefix, never a request-hole or expanded tile.
 * @param physicalRows Exact stage/native FP32 row extent.
 * @throws std::invalid_argument For a partial, shifted or expanded authority.
 */
inline void validateHCResidualDownRows(DeviceRowRange rows,int physicalRows) {
 const auto identity=rows.captureIdentity();
 if(identity.physical_rows!=physicalRows||identity.count_capacity!=physicalRows||
    identity.first_physical_row!=0||identity.repeats_per_row!=1)
  throw std::invalid_argument("HC publication requires its complete original prefix authority");
}
/** @brief Existing canonical primary signed-byte bank, retained as one stable name. */
inline const std::string kGPUProjectionPrimaryQ8=GemmWorkspaceBuffers::QUANT_A;
/** @brief Existing canonical primary scale bank, avoiding per-invocation string allocation. */
inline const std::string kGPUProjectionPrimaryQ8Scales=GemmWorkspaceBuffers::SCALES_A_BLOCKWISE;
/** @brief Additional named GPU activation banks beyond the ordinary Q8 projection.
 * @param m Retained physical capacity, never a communication/live extent.
 * @param n Exact native output width.
 * @param k Exact native input width.
 * @return Only the residual byte and scale banks, suitable for metadata-only BOMs.
 * The ordinary primary banks retain their existing names and admission authority.
 */
inline WorkspaceRequirements gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(int m,int n,int k) {
 validateQ8ResidualCorrectionGeometry(m,n,k);const auto values=static_cast<std::size_t>(m)*k;
 WorkspaceRequirements result;
 result.buffers.push_back({kGPUProjectionResidualQ8,values,256,true});
 result.buffers.push_back({kGPUProjectionResidualQ8Scales,values/32*sizeof(float),256,true});
 return result;
}
/** @brief Minimal named banks for both simultaneous GPU Q8 terms.
 * @param m Retained physical capacity, never a communication/live extent.
 * @param n Exact native output width.
 * @param k Exact native input width.
 * @return Two byte banks and two block-scale banks; no output/weight duplicate.
 */
inline WorkspaceRequirements gpuQ8ResidualCorrectionWorkspaceRequirements(int m,int n,int k) {
 const auto additional=gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(m,n,k);
 WorkspaceRequirements result;
 result.buffers.push_back({kGPUProjectionPrimaryQ8,additional.buffers[0].size_bytes,256,true});
 result.buffers.push_back(additional.buffers[0]);
 result.buffers.push_back({kGPUProjectionPrimaryQ8Scales,additional.buffers[1].size_bytes,256,true});
 result.buffers.push_back(additional.buffers[1]);
 return result;
}
/** @brief Exact prepared HC-up primary program, independent of physical M.
 * CUDA retains its blockwise FP32 scale and native reciprocal/multiply/RNE.
 * ROCm borrows canonical finite-half q8Scale and quantizeQ8 with the native
 * reciprocal used by its standalone/grouped/prefill blockwise producers.
 * The heterogeneous MoE deterministic reciprocal is a different operation;
 * substituting it would change this primary representation and is forbidden.
 */
enum class Q8ResidualQuantizerPolicy { CPUHalfAfterQuantization, CUDAFloatScaleNativeReciprocal, ROCmFiniteHalfNativeReciprocal };
/** @brief Capture arguments borrow one actual prepared source and exact live-row authority. */
struct GPUQ8ResidualCorrectionArgs {
 DeviceNativeQuantisedMatrixDesc weights; ///< Exact prepared native descriptor, never a reconstructed source.
 DeviceRowRange rows; ///< Device-owned live prefix; setup never reads its count.
 Q8ResidualQuantizerPolicy quantizer; ///< Frozen actual backend primary quantizer, included in capture arguments.
 const float* input; ///< Prepared FP32 activation rows before either Q8 publication.
 float* output; ///< Distinct prepared FP32 output, authored only for live rows.
 signed char *primary,*residual; ///< Persistent independently owned signed-byte Q8 banks.
 float *primaryScales,*residualScales; ///< Backend-native stored scales per physical row/Kblock.
};
/** @brief Exact original owner that authored the current HC bank prefix. */
enum class HCDownQ8PublicationSource { Standalone, OriginalReducer };
/** @brief Native HC producer declaration, retaining the existing projection operands. */
struct GPUHCResidualDownPublication {
 GPUQ8ResidualCorrectionArgs projection; ///< Same weights, rows, banks and destination.
 float* activated; ///< Retained FP32 /4 SiLU output; exact input alias is permitted.
 HCDownQ8PublicationSource source=HCDownQ8PublicationSource::Standalone; ///< Typed sealed required publication alternative.
};
/**
 * @brief Complete local HC read with retained projection and mixed destinations.
 *
 * The up projection has four branch-major output vectors. Each mixed channel
 * consumes those four exact FP32 values in branch order. Their common owning
 * CTA may therefore finish the read without a second launch or global barrier.
 * Publication remains a separate producer so raw/activated exact aliasing is
 * safe across independent projection tiles.
 */
struct GPUHCResidualReadProjection {
 GPUHCResidualDownPublication down; ///< Original /4 SiLU and two-term producer.
 const float* normalized; ///< Original read-only [M,4H] normalized branches.
 float* mixed; ///< Distinct [M,H] result; inactive rows remain untouched.
};
/**
 * @brief Publish two-term HC activations and fuse CUDA up projection/read.
 * @param args Exact original weights, count, banks and retained FP32 operands.
 * @param stream Exact non-default stream owning both retained native nodes.
 * @param tables Original prepared codebook table views, bound before capture.
 * @throws std::invalid_argument Before submission for malformed geometry or aliases.
 * @throws std::runtime_error For native submission failure.
 */
void cudaHCResidualReadQ8Projection(const GPUHCResidualReadProjection& args,void* stream,NativeEmbeddingIQTables tables);
/** @copydoc cudaHCResidualReadQ8Projection */
void rocmHCResidualReadQ8Projection(const GPUHCResidualReadProjection& args,void* stream,NativeEmbeddingIQTables tables);
/** @brief Original HC publication followed by the exact producer-grid M1 up/read.
 * @param args Original retained operands; gate logits are authored unchanged.
 * @param stream Exact non-default producer/consumer stream.
 * @throws std::invalid_argument Before publication unless physical M1, codebook19,
 * short-K geometry, four branches and all original disjoint storage rules hold.
 */
void cudaHCResidualShortKReadQ8Projection(const GPUHCResidualReadProjection& args,void* stream);
/** @copydoc cudaHCResidualShortKReadQ8Projection */
void rocmHCResidualShortKReadQ8Projection(const GPUHCResidualReadProjection& args,void* stream);
/** @brief Fuse original HC activation and Q8 publication, then the unchanged CUDA projection.
 * @param args Complete original projection and retained activation destination.
 * @param stream Exact non-default producer/consumer stream.
 * @param tables Original cold generation-bound native IQ tables.
 */
void cudaHCResidualDownQ8Projection(const GPUHCResidualDownPublication& args,void* stream,NativeEmbeddingIQTables tables);
/** @brief Fuse original HC activation and Q8 publication, then the unchanged HIP projection.
 * @param args Complete original projection and retained activation destination.
 * @param stream Exact non-default producer/consumer stream.
 * @param tables Original cold generation-bound native IQ tables.
 */
void rocmHCResidualDownQ8Projection(const GPUHCResidualDownPublication& args,void* stream,NativeEmbeddingIQTables tables);
/** @brief Launch CUDA publication and paired integer dots on an exact non-default stream.
 * @param args Prepared source, admitted scratch and borrowed live geometry.
 * @param stream Exact ordered native CUDA stream; null/default is rejected.
 */
void cudaQ8ResidualCorrectedProjection(const GPUQ8ResidualCorrectionArgs&args,void*stream);
/** @brief Launch HIP publication and paired integer dots on an exact non-default stream.
 * @param args Prepared source, admitted scratch and borrowed live geometry.
 * @param stream Exact ordered native HIP stream; null/default is rejected.
 */
void rocmQ8ResidualCorrectedProjection(const GPUQ8ResidualCorrectionArgs&args,void*stream);
/**
 * @brief Explicit CUDA original-column diagnostic oracle, independent of production dispatch.
 * @param args PhysicalM1 native signed-Q8 source, row authority and original admitted banks.
 * @param stream Exact non-default native stream owning publication and reference projection.
 * @throws std::invalid_argument Before publication for any unsupported source or geometry.
 * @throws std::runtime_error When native submission fails.
 * @return No value; submits the original publisher and ordered column projection.
 */
void cudaQ8ResidualOriginalColumnReference(const GPUQ8ResidualCorrectionArgs &args, void *stream);
/**
 * @brief Explicit HIP original-column diagnostic oracle, independent of production dispatch.
 * @param args PhysicalM1 native signed-Q8 source, row authority and original admitted banks.
 * @param stream Exact non-default native stream owning publication and reference projection.
 * @throws std::invalid_argument Before publication for any unsupported source or geometry.
 * @throws std::runtime_error When native submission fails.
 * @return No value; submits the original publisher and ordered column projection.
 */
void rocmQ8ResidualOriginalColumnReference(const GPUQ8ResidualCorrectionArgs &args, void *stream);
}
