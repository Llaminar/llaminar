/**
 * @file CUDAQwenSparseAttention.cu
 * @brief Allocation-free CUDA launch implementation of sparse Qwen4Exp execution.
 *
 * This boundary rejects default streams and surfaces native launch failures.
 * Dynamic shape/state diagnostics remain device-owned inside captured kernels.
 */
#include "../../qwen4exp/QwenSparseReplay.h"
#include "CUDAQwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseGPU.cuh"
#include <stdexcept>
#include <string>
namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Reject null and special default streams before any graph operation. */
void requireStream(cudaStream_t stream)
{
    if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        throw std::invalid_argument("CUDA Qwen sparse attention requires an exact non-default stream");
}
/** @brief Surface an immediate native launch failure without synchronization. */
void checkLaunch()
{
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA Qwen sparse launch: ") + cudaGetErrorString(status));
}
}
void cudaQwenSparseIndexer(const QwenSparseIndexer &p, cudaStream_t stream)
{
    requireStream(stream);
    validateQwenSparseIndexer(p);
    cuda_sparse_detail::preparePools<<<(p.row_capacity + 6) / 4, 128, 0, stream>>>(p);
    checkLaunch();
    cuda_sparse_detail::prepareQueries<<<p.row_capacity * 4, 128, 0, stream>>>(p);
    checkLaunch();
    cuda_sparse_detail::scorePools<<<dim3((p.block_capacity + 7) / 8, p.row_capacity), 256, 0, stream>>>(p);
    checkLaunch();
    cudaQwenSparseSelectPools(p, stream);
}
void cudaQwenSparseSelectPools(const QwenSparseIndexer &p, cudaStream_t stream)
{
    requireStream(stream);
    validateQwenSparseSelection(p);
    switch (p.pool_selection)
    {
    case QwenSparsePoolSelection::Installed:
        if (p.row_capacity <= attention::kMaxGroupedVerifierAttentionRows)
            cuda_sparse_detail::selectPoolsWide<<<p.row_capacity, 256, 0, stream>>>(p);
        else cuda_sparse_detail::selectPools<<<(p.row_capacity + 7) / 8, 256, 0, stream>>>(p);
        break;
    case QwenSparsePoolSelection::Bounded256_512:
        cuda_sparse_detail::selectPoolsBounded<256, 512><<<p.row_capacity, 256, 0, stream>>>(p); break;
    case QwenSparsePoolSelection::Bounded512_512:
        cuda_sparse_detail::selectPoolsBounded<512, 512><<<p.row_capacity, 256, 0, stream>>>(p); break;
    case QwenSparsePoolSelection::Bounded1024_1024:
        cuda_sparse_detail::selectPoolsBounded<1024, 1024><<<p.row_capacity, 256, 0, stream>>>(p); break;
    default: throw std::invalid_argument("QSA selection policy is unknown");
    }
    checkLaunch();
}

void cudaQwenSparseAttention(const QwenSparseAttention &p, cudaStream_t stream)
{
    requireStream(stream);
    validateQwenSparseAttention(p);
    if (!p.gpu_partials) throw std::invalid_argument("CUDA Qwen sparse attention requires admitted fixed-partition scratch");
    if (detail::isRotated(p.kv.keys))
    {
        cuda_sparse_detail::rotateQueries<<<dim3(p.row_capacity, p.query_heads), 256, 0, stream>>>(p);
        checkLaunch();
    }
    cuda_sparse_detail::dispatchAttention(p.kv, [&]<TensorType Key, TensorType Value>() {
        if constexpr(Key==TensorType::TQ4 || Key==TensorType::TQ8)
            cuda_sparse_detail::selectedAttentionPartitions<Key,Value,false><<<dim3(p.row_capacity,p.query_heads,QwenSparsePartitions),256,0,stream>>>(p);
        else if(p.kv.key_policy.transformsOnRead())
            cuda_sparse_detail::selectedAttentionPartitions<Key,Value,true><<<dim3(p.row_capacity,p.query_heads,QwenSparsePartitions),256,0,stream>>>(p);
        else cuda_sparse_detail::selectedAttentionPartitions<Key,Value,false><<<dim3(p.row_capacity,p.query_heads,QwenSparsePartitions),256,0,stream>>>(p);
        checkLaunch();
        cuda_sparse_detail::reduceAttentionPartitions<Value><<<dim3(p.row_capacity, p.query_heads), 256, 0, stream>>>(p);
    });
    checkLaunch();
}
void cudaQwenSparseSnapshotPrior(const QwenSparsePriorSnapshot& p,void* rawStream)
{
    const auto stream=static_cast<cudaStream_t>(rawStream);
    requireStream(stream);validateQwenSparsePriorSnapshot(p);
    cuda_sparse_detail::snapshotPrior<<<1,128,0,stream>>>(p);checkLaunch();
}
void cudaQwenSparseCommitPartial(const QwenSparsePartialCommit &p, cudaStream_t stream)
{
    requireStream(stream);
    validateQwenSparsePartialCommit(p);
    cuda_sparse_detail::commitPartial<<<1, 128, 0, stream>>>(p);
    checkLaunch();
}
namespace
{
/** @brief Typed CUDA implementation of the compute-stage sparse primitive boundary. */
class CUDAQwenSparseAttentionKernels final : public IQwenSparseAttentionKernels
{
public:
    /** @copydoc IQwenSparseAttentionKernels::indexer */
    void indexer(const QwenSparseIndexer &p, void *stream) const override { cudaQwenSparseIndexer(p, static_cast<cudaStream_t>(stream)); }
    /** @copydoc IQwenSparseAttentionKernels::attention */
    void attention(const QwenSparseAttention &p, void *stream) const override { cudaQwenSparseAttention(p, static_cast<cudaStream_t>(stream)); }
    /** @copydoc IQwenSparseAttentionKernels::prepareReplay */
    void prepareReplay(const QwenSparsePrepareReplay &p,void *stream) const override { cudaQwenSparsePrepareReplay(p,stream); }
    /** @copydoc IQwenSparseAttentionKernels::snapshotPrior */
    void snapshotPrior(const QwenSparsePriorSnapshot& p,void* stream) const override { cudaQwenSparseSnapshotPrior(p,stream); }
    /** @copydoc IQwenSparseAttentionKernels::commitPartial */
    void commitPartial(const QwenSparsePartialCommit &p, void *stream) const override { cudaQwenSparseCommitPartial(p, static_cast<cudaStream_t>(stream)); }
    /** @copydoc IQwenSparseAttentionKernels::retainCandidates */
    void retainCandidates(const QwenSparseCandidateRetention &p,void *stream) const override
    {
        cudaQwenStateTransfer(qwenSparseCandidateTransfer(p),stream);
    }
};
}
std::unique_ptr<IQwenSparseAttentionKernels> createCUDAQwenSparseAttentionKernels()
{
    return std::make_unique<CUDAQwenSparseAttentionKernels>();
}
}
