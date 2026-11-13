/**
 * @file QwenSparseQueryRowWork.h
 * @brief QSA query demand over the indexer's original replay publication.
 *
 * This helper supplies model-specific demand to the generic row-work descriptor.
 * Graph projection ownership, striding and immutable table publication are owned
 * exclusively by ProjectionRowWorkBinding; kernels only borrow original words.
 */
#pragma once
#include "kernels/common/ProjectionRowWork.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"

namespace llaminar2::qwen4exp {
/**
 * @brief Bind native query preparation to its original replay without reading live words.
 * @param indexer Exact captured native indexer invocation.
 * @return Unexpanded original token-row work; native head launch repeats rows explicitly.
 * @throws std::invalid_argument For unknown demand or incomplete replay geometry.
 */
[[nodiscard]] inline ProjectionRowWork qwenSparseIndexerQueryWork(const QwenSparseIndexer &indexer) {
    if(!indexer.replay||indexer.row_capacity<=0||indexer.token_capacity<indexer.row_capacity)
        throw std::invalid_argument("QSA query preparation requires exact original replay geometry");
    switch(indexer.query_demand) {
    case QwenSparseQueryDemand::AllLive:
        return ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(indexer.row_capacity,&indexer.replay->live_rows));
    case QwenSparseQueryDemand::ProvenNoRanking:
        throw std::invalid_argument("ProvenNoRanking has no query projection/preparation work descriptor");
    case QwenSparseQueryDemand::RankingOnly:
        return ProjectionRowWork::causalStrided(1,indexer.row_capacity,&indexer.replay->prefix_tokens,&indexer.replay->live_rows,
            1,indexer.token_capacity,QwenSparseFirstRankingLogicalRow);
    }
    throw std::invalid_argument("QSA query preparation has an unknown demand policy");
}
} // namespace llaminar2::qwen4exp
