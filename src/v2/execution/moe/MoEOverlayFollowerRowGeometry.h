/**
 * @file MoEOverlayFollowerRowGeometry.h
 * @brief Exact retained follower execution rows over original admitted storage families.
 *
 * A dynamic MTP ticket can name every live decode width, while storage may
 * retain economical power-of-two capacities. This metadata-only declaration
 * binds exact execution identities to existing backing without adding roots,
 * padding work/traffic, mutating a prefill ladder or recording graphs at runtime.
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace llaminar2 {
/** @brief Immutable exact row identities and their original supporting packet capacities. */
class MoEOverlayFollowerRowGeometry final
{
public:
    /** @brief One exact execution shape, sharing an already-admitted serial storage family. */
    struct Shape
    {
        int physical_rows; ///< Exact graph/ticket work rows; never rounded for execution.
        std::size_t backing_rows; ///< Original retained tensor capacity, possibly larger.
        bool operator==(const Shape&) const = default;
    };
    /**
     * @brief Declare every live decode row plus only source-declared prefill identities.
     * @param maximum_rows Original graph-family bound (a sidecar may retain only one row).
     * @param decode_rows Original admitted live decode/verifier bound.
     * @param prefill_rows Original ordered unique source-owned prefill ladder.
     * @param storage_rows Original ordered unique retained packet-family capacities.
     * @return Exact immutable shape manifest with proved supporting backing for each row.
     * @throws std::invalid_argument For incomplete bounds, ladders or backing coverage.
     */
    static MoEOverlayFollowerRowGeometry declare(int maximum_rows,int decode_rows,
        std::span<const int> prefill_rows,std::span<const std::size_t> storage_rows)
    {
        if(maximum_rows<=0 || decode_rows<=0 || storage_rows.empty() || storage_rows.front()==0 ||
           !std::is_sorted(storage_rows.begin(),storage_rows.end()) ||
           std::adjacent_find(storage_rows.begin(),storage_rows.end())!=storage_rows.end() ||
           !std::is_sorted(prefill_rows.begin(),prefill_rows.end()) ||
           std::adjacent_find(prefill_rows.begin(),prefill_rows.end())!=prefill_rows.end() ||
           (!prefill_rows.empty() && prefill_rows.front()<=0) ||
           storage_rows.back()<static_cast<std::size_t>(maximum_rows))
            throw std::invalid_argument("Follower exact rows require original bounded execution/storage manifests");
        std::vector<int> rows;
        const auto decode_limit=static_cast<std::size_t>(std::min(maximum_rows,decode_rows));
        for(std::size_t row=1;row<=decode_limit;++row)rows.push_back(static_cast<int>(row));
        for(const int row:prefill_rows)if(row<=maximum_rows)rows.push_back(row);
        rows.push_back(maximum_rows);
        std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
        MoEOverlayFollowerRowGeometry result;
        result.shapes_.reserve(rows.size());
        for(const int row:rows)
        {
            const auto backing=std::lower_bound(storage_rows.begin(),storage_rows.end(),static_cast<std::size_t>(row));
            if(backing==storage_rows.end())throw std::invalid_argument("Follower exact row exceeds original admitted storage");
            result.shapes_.push_back({row,*backing});
        }
        return result;
    }
    /** @return Original declared exact execution identities; this owns no packet storage. */
    const std::vector<Shape>& shapes() const noexcept {return shapes_;}
    /**
     * @param rows Requested exact execution width.
     * @return Exact identity only, or null; no runtime bucket substitution is permitted.
     */
    const Shape* forRows(int rows) const noexcept
    {
        const auto found=std::lower_bound(shapes_.begin(),shapes_.end(),rows,
            [](const Shape& shape,int requested){return shape.physical_rows<requested;});
        return found==shapes_.end() || found->physical_rows!=rows?nullptr:&*found;
    }
    /**
     * @brief Describe an absent exact execution shape without changing policy or storage.
     * @param rows Missing requested width.
     * @param retained Actual retained exact execution identities.
     * @return Diagnostic containing the rejected width and retained manifest.
     */
    static std::string missingShapeDiagnostic(int rows,std::span<const int> retained)
    {
        std::ostringstream message;
        message<<"Mapped ExpertOverlay follower has no setup-owned exact executable for physical_rows="<<rows
               <<"; retained_rows=[";
        for(std::size_t index=0;index<retained.size();++index)message<<(index?",":"")<<retained[index];
        message<<']';return message.str();
    }
private:
    /** @brief Only declare() may construct a manifest after validating its inputs. */
    MoEOverlayFollowerRowGeometry() = default;
    std::vector<Shape> shapes_; ///< Cold metadata only; backing ownership remains in the original arena/PMA.
};
}
