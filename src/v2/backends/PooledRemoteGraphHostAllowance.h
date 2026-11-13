/**
 * @file PooledRemoteGraphHostAllowance.h
 * @brief Conservative opaque host pricing for source-declared remote graph size.
 *
 * The measured closed program's entire cold RSS increase was below625MB.
 * A1GiB unit deliberately includes substantial headroom and shared loading;
 * it is not an exact allocator attribution. Declared graph count is charged
 * additively through the original PMA. Native structure/ABI/ownership seals
 * stay exact and independent from this coarse physical allowance.
 */
#pragma once
#include "PooledRemoteGraphProgram.h"
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace llaminar2 {
/** @brief Pure typed-family estimator; creates no resource, ledger or runtime eligibility decision. */
class PooledRemoteGraphHostAllowance final {
public:
    static constexpr std::uint64_t kNodesPerUnit=1024;
    static constexpr std::size_t kHostBytesPerUnit=std::size_t{1}<<30;
    /** @brief Price started node buckets with checked arithmetic; recorded rows may provide their own sealed toy unit.
     * @param nodes Positive actual source-declared node count. @param bytesPerUnit Installed/recorded immutable row unit.
     * @return Host commitment contributed to the original PMA before creation.
     * @throws std::invalid_argument For empty geometry/unit. @throws std::overflow_error If the commitment is unrepresentable.
     */
    static std::size_t scaledBytes(std::uint64_t nodes,std::size_t bytesPerUnit){
        if(!nodes || !bytesPerUnit)throw std::invalid_argument("Remote graph allowance needs declared nodes and a positive family unit");
        const auto units=nodes/kNodesPerUnit+(nodes%kNodesPerUnit!=0);
        if(units>std::numeric_limits<std::size_t>::max()/bytesPerUnit)
            throw std::overflow_error("Remote graph host allowance exceeds representable PMA bytes");
        return static_cast<std::size_t>(units)*bytesPerUnit;
    }
    /** @return Installed1GiB units for any positive representable declared node geometry. */
    static std::size_t bytesForDeclaredNodes(std::uint64_t nodes){return scaledBytes(nodes,kHostBytesPerUnit);}
    /** @return Actual source-authored graph contribution; observed node counts impose no support ceiling. */
    static std::size_t bytesFor(const PooledRemoteGraphProgram& program){return bytesForDeclaredNodes(program.limits().maximumNodes);}
};
}
