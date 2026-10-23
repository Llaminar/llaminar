/**
 * @file ROCmQwenSparseAttention.hip
 * @brief Allocation-free HIP launch implementation of sparse Qwen4Exp execution.
 *
 * This boundary rejects default streams and surfaces native launch failures.
 * Dynamic shape/state diagnostics remain device-owned inside captured kernels.
 */
#include "../../qwen4exp/QwenSparseReplay.h"
#include "ROCmQwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseGPU.cuh"
#include "ROCmQwenSparseFP16Attention.cuh"
#include <stdexcept>
#include <string>
namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Reject null and special default streams before any graph operation. */
void requireStream(hipStream_t stream)
{
    if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread)
        throw std::invalid_argument("ROCm Qwen sparse attention requires an exact non-default stream");
}
/** @brief Surface an immediate native launch failure without synchronization. */
void checkLaunch()
{
    const auto status = hipGetLastError();
    if (status != hipSuccess)
        throw std::runtime_error(std::string("ROCm Qwen sparse launch: ") + hipGetErrorString(status));
}
}
void rocmQwenSparseIndexer(const QwenSparseIndexer &p, hipStream_t stream)
{
    requireStream(stream);
    validateQwenSparseIndexer(p);
    rocm_sparse_detail::preparePools<<<(p.row_capacity + 6) / 4, 128, 0, stream>>>(p);
    checkLaunch();
    rocm_sparse_detail::prepareQueries<<<p.row_capacity * 4, 128, 0, stream>>>(p);
    checkLaunch();
    rocm_sparse_detail::scorePools<<<dim3((p.block_capacity + 7) / 8, p.row_capacity), 256, 0, stream>>>(p);
    checkLaunch();
    rocmQwenSparseSelectPools(p, stream);
}
void rocmQwenSparseSelectPools(const QwenSparseIndexer &p, hipStream_t stream)
{
    requireStream(stream);
    validateQwenSparseSelection(p);
    switch (p.pool_selection)
    {
    case QwenSparsePoolSelection::Installed:
        if (p.row_capacity <= attention::kMaxGroupedVerifierAttentionRows)
            rocm_sparse_detail::selectPoolsWide<<<p.row_capacity, 256, 0, stream>>>(p);
        else rocm_sparse_detail::selectPools<<<(p.row_capacity + 7) / 8, 256, 0, stream>>>(p);
        break;
    case QwenSparsePoolSelection::Bounded256_512:
        rocm_sparse_detail::selectPoolsBounded<256, 512><<<p.row_capacity, 256, 0, stream>>>(p); break;
    case QwenSparsePoolSelection::Bounded512_512:
        rocm_sparse_detail::selectPoolsBounded<512, 512><<<p.row_capacity, 256, 0, stream>>>(p); break;
    case QwenSparsePoolSelection::Bounded1024_1024:
        rocm_sparse_detail::selectPoolsBounded<1024, 1024><<<p.row_capacity, 256, 0, stream>>>(p); break;
    default: throw std::invalid_argument("QSA selection policy is unknown");
    }
    checkLaunch();
}

void rocmQwenSparseAttention(const QwenSparseAttention &p, hipStream_t stream)
{
    requireStream(stream);
    validateQwenSparseAttentionArithmetic(p.arithmetic,DeviceType::ROCm);
    if(p.arithmetic==QwenSparseAttentionArithmetic::TileFP16WithFP32Totals ||
       p.arithmetic==QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33) {
        rocmQwenSparseFP16Attention(p,stream);
        return;
    }
    validateQwenSparseAttention(p);
    if (!p.gpu_partials) throw std::invalid_argument("ROCm Qwen sparse attention requires admitted fixed-partition scratch");
    if (detail::isRotated(p.kv.keys))
    {
        rocm_sparse_detail::rotateQueries<<<dim3(p.row_capacity, p.query_heads), 256, 0, stream>>>(p);
        checkLaunch();
    }
    rocm_sparse_detail::dispatchAttention(p.kv, [&]<TensorType Key, TensorType Value>() {
        if constexpr(Key==TensorType::TQ4 || Key==TensorType::TQ8)
            rocm_sparse_detail::selectedAttentionPartitions<Key,Value,false><<<dim3(p.row_capacity,p.query_heads,QwenSparsePartitions),256,0,stream>>>(p);
        else if(p.kv.key_policy.transformsOnRead())
            rocm_sparse_detail::selectedAttentionPartitions<Key,Value,true><<<dim3(p.row_capacity,p.query_heads,QwenSparsePartitions),256,0,stream>>>(p);
        else rocm_sparse_detail::selectedAttentionPartitions<Key,Value,false><<<dim3(p.row_capacity,p.query_heads,QwenSparsePartitions),256,0,stream>>>(p);
        checkLaunch();
        rocm_sparse_detail::reduceAttentionPartitions<Value><<<dim3(p.row_capacity, p.query_heads), 256, 0, stream>>>(p);
    });
    checkLaunch();
}
void rocmQwenSparseFP16Attention(const QwenSparseAttention& p, hipStream_t stream)
{
    requireStream(stream);
    if(p.arithmetic!=QwenSparseAttentionArithmetic::TileFP16WithFP32Totals &&
       p.arithmetic!=QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33)
        throw std::invalid_argument("ROCm FP16 tiled entry requires a sealed FP16 arithmetic program");
    validateQwenSparseAttention(p);
    if (!p.gpu_partials || p.kv.keys.codec != TensorType::FP16 || p.kv.values.codec != TensorType::FP16)
        throw std::invalid_argument("ROCm FP16 tiled attention requires native FP16 KV and admitted program-specific scratch");
    for (const auto* bank : {&p.kv.keys, &p.kv.values})
        if (reinterpret_cast<std::uintptr_t>(bank->data) % 16 || bank->token_stride_bytes % 16 || bank->head_stride_bytes % 16)
            throw std::invalid_argument("ROCm FP16 tiled attention requires16-byte aligned native bank rows");
    if (p.kv.key_policy.transformsOnRead() && p.kv.key_rotary_dim != 64)
        throw std::invalid_argument("ROCm FP16 tiled attention requires the explicit64-coordinate pre-RoPE policy");
    // Graph admission seals the arithmetic and exact partial extent. Both
    // programs use the same native entry; no test-only execution path remains.
    const auto launch = [&]<int Parts>() {
        constexpr int GroupHeads = Parts == 33 ? 6 : 4;
        constexpr int Threads = GroupHeads * 64;
        const dim3 grid(p.row_capacity,
            (p.query_head_start % GroupHeads + p.query_heads + GroupHeads - 1) / GroupHeads, Parts);
        if (p.kv.key_policy.transformsOnRead())
            rocm_sparse_detail::selectedAttentionFP16Tiles<true,Parts><<<grid,Threads,0,stream>>>(p);
        else rocm_sparse_detail::selectedAttentionFP16Tiles<false,Parts><<<grid,Threads,0,stream>>>(p);
        checkLaunch();
        rocm_sparse_detail::reduceAttentionPartitions<TensorType::FP16,Parts>
            <<<dim3(p.row_capacity,p.query_heads),256,0,stream>>>(p);
        checkLaunch();
    };
    if(p.arithmetic==QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33)
        launch.template operator()<33>();
    else
        launch.template operator()<8>();
}


void rocmQwenSparseSnapshotPrior(const QwenSparsePriorSnapshot& p,void* rawStream)
{
    const auto stream=static_cast<hipStream_t>(rawStream);
    requireStream(stream);validateQwenSparsePriorSnapshot(p);
    rocm_sparse_detail::snapshotPrior<<<1,128,0,stream>>>(p);checkLaunch();
}
void rocmQwenSparseCommitPartial(const QwenSparsePartialCommit &p, hipStream_t stream)
{
    requireStream(stream);
    validateQwenSparsePartialCommit(p);
    rocm_sparse_detail::commitPartial<<<1, 128, 0, stream>>>(p);
    checkLaunch();
}
namespace
{
/** @brief Typed HIP implementation of the compute-stage sparse primitive boundary. */
class ROCmQwenSparseAttentionKernels final : public IQwenSparseAttentionKernels
{
public:
    /** @copydoc IQwenSparseAttentionKernels::indexer */
    void indexer(const QwenSparseIndexer &p, void *stream) const override { rocmQwenSparseIndexer(p, static_cast<hipStream_t>(stream)); }
    /** @copydoc IQwenSparseAttentionKernels::attention */
    void attention(const QwenSparseAttention &p, void *stream) const override { rocmQwenSparseAttention(p, static_cast<hipStream_t>(stream)); }
    /** @copydoc IQwenSparseAttentionKernels::prepareReplay */
    void prepareReplay(const QwenSparsePrepareReplay &p,void *stream) const override { rocmQwenSparsePrepareReplay(p,stream); }
    /** @copydoc IQwenSparseAttentionKernels::snapshotPrior */
    void snapshotPrior(const QwenSparsePriorSnapshot& p,void* stream) const override { rocmQwenSparseSnapshotPrior(p,stream); }
    /** @copydoc IQwenSparseAttentionKernels::commitPartial */
    void commitPartial(const QwenSparsePartialCommit &p, void *stream) const override { rocmQwenSparseCommitPartial(p, static_cast<hipStream_t>(stream)); }
    /** @copydoc IQwenSparseAttentionKernels::retainCandidates */
    void retainCandidates(const QwenSparseCandidateRetention &p,void *stream) const override
    {
        rocmQwenStateTransfer(qwenSparseCandidateTransfer(p),stream);
    }
};
}
std::unique_ptr<IQwenSparseAttentionKernels> createROCmQwenSparseAttentionKernels()
{
    return std::make_unique<ROCmQwenSparseAttentionKernels>();
}
}
