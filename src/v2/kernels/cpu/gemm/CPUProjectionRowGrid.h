/**
 * @file CPUProjectionRowGrid.h
 * @brief ISA-neutral task coordinates over original CPU request publications.
 *
 * A synchronous invocation borrows existing host-owned row/frontier words.
 * Physical tiles retain their original request boundaries and input/output
 * strides; slicing, head expansion and inactive prefixes never compact rows.
 * Native ARM and x86 schedulers use this one geometry authority.
 */
#pragma once
#include "kernels/common/ProjectionRowWork.h"
#include "utils/OpenMPUtils.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief Request-aligned CPU work grid over one borrowed original publication.
     *
     * Construct before entering the workshare. Workers borrow this immutable value;
     * they never fetch recorder TLS or compact row IDs. Fixed physical tiles
     * share packed-weight decoding while every row retains serial K arithmetic.
     */
    class CPUProjectionRowGrid final
    {
    public:
        /** @brief Validate all original request publications before any destination write.
         * @param physical Original matrix/partial row capacity.
         * @param work Optional invocation-owned immutable row admission.
         * @throws std::invalid_argument For bad geometry or publication values.
         */
        CPUProjectionRowGrid(int physical,const ProjectionRowWork *work)
            :physical_(physical),work_(work)
        {
            if(physical<=0||(work&&work->physicalRows()!=physical))
                throw std::invalid_argument("CPU projection work requires exact physical matrix rows");
            if(work_)for(int request=0;request<work_->requests();++request)(void)work_->resolveHost(request);
        }
        /** @return Whether any original request owns useful rows in this physical slice. */
        [[nodiscard]] bool hasAny() const
        {
            if(!work_)return true;
            for(int request=0;request<work_->requests();++request)
                if(work_->resolveHost(request).size()>0)return true;
            return false;
        }
        /** @brief Admit work without losing an empty retained-team completion edge.
         * @return Whether useful work must enter its ordinary collective workshare.
         * @details All members of an existing active team call this boundary;
         * original publications stay immutable until completion, even when empty.
         */
        [[nodiscard]] bool beginWorkshare() const
        { return llaminar::v2::beginWorkshare(hasAny()); }

        /** @brief Test an original physical row before any activation or partial read.
         * @param row Original matrix-relative row, never a compacted useful index.
         * @return Whether that exact row is authored.
         */
        [[nodiscard]] bool active(int row) const
        {
            if(row<0||row>=physical_)return false;
            if(!work_)return true;
            return work_->resolveHost(work_->requestForRow(row)).contains(row);
        }
        /** @brief Resolve immutable request-aligned task geometry.
         * @param width Positive physical microkernel reuse width.
         * @return Fixed tile count, independent of live word values.
         * @throws std::invalid_argument For a nonpositive width.
         */
        [[nodiscard]] int tiles(int width) const
        {
            if(width<=0)throw std::invalid_argument("CPU projection row tile width must be positive");
            return work_ ? work_->requests()*((work_->rowsPerRequest()-1)/width+1) : (physical_-1)/width+1;
        }
        /** @brief Resolve both endpoints of one original request-aligned work tile.
         * @param tile Immutable task-grid row tile index.
         * @param width Positive physical microkernel reuse width.
         * @return Original matrix-relative useful span, including midtile suffixes.
         * @throws std::out_of_range For an invalid immutable tile coordinate.
         */
        [[nodiscard]] ProjectionRowSpan tileSpan(int tile,int width) const
        {
            if(tile<0||tile>=tiles(width))throw std::out_of_range("CPU projection work tile exceeds geometry");
            if(!work_) {
                const int first=tile*width;
                return {first,first+std::min(width,physical_-first)};
            }
            const int per=(work_->rowsPerRequest()-1)/width+1,request=tile/per;
            const auto authored=work_->resolveHost(request);
            const auto identity=work_->captureIdentity();
            const auto base=static_cast<std::int64_t>(request)*work_->rowsPerRequest();
            const auto originalFirst=base+static_cast<std::int64_t>(tile%per)*width;
            const auto originalEnd=std::min(originalFirst+width,base+work_->rowsPerRequest());
            const int first=static_cast<int>(std::clamp<std::int64_t>(originalFirst-identity.first_physical_row,0,physical_));
            const int end=static_cast<int>(std::clamp<std::int64_t>(originalEnd-identity.first_physical_row,0,physical_));
            const int selectedEnd=std::min(end,authored.end);
            return {std::min(selectedEnd,std::max(first,authored.first)),selectedEnd};
        }
    private:
        int physical_; ///< Original tensor/partial capacity; never useful row count.
        const ProjectionRowWork *work_; ///< Invocation-owned immutable metadata, live through the workshare.
    };
} // namespace llaminar2::cpu::native_quantised
