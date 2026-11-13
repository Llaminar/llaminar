/**
 * @file ProjectionRowPublicationTable.h
 * @brief Retained proof of one immutable original-request address table.
 *
 * The workspace remains the sole physical allocation and accounting owner.
 * This lease attests its completed setup publication and exact address pairs;
 * it contains no live count, frontier, activation or host execution shadow.
 */
#pragma once
#include "WorkspaceBufferLease.h"
#include "kernels/common/ProjectionRowWork.h"
#include <memory>
#include <span>
#include <vector>

namespace llaminar2 {
class ProjectionRowWorkBinding;
/** @brief Immutable metadata publication, constructed only after an ordered setup copy. */
class ProjectionRowPublicationTable final {
public:
    /** @return Exact native table embedded by captured consumers. */
    const ProjectionRequestPublication *data() const noexcept {
        return static_cast<const ProjectionRequestPublication *>(storage_->data());
    }
    /** @return Workspace generation that originally admitted this table. */
    std::uint64_t generation() const noexcept {return generation_;}
    /** @return Original physical allocation/extent owner, never a new PMA claim. */
    const WorkspaceBufferLease &storage() const noexcept {return *storage_;}
    /** @return Original address pairs authenticated before publication. */
    std::span<const ProjectionRequestPublication> entries() const noexcept {return entries_;}
private:
    friend class ProjectionRowWorkBinding;
    /** @brief Seal the original allocation after its entire immutable payload is published. */
    ProjectionRowPublicationTable(std::shared_ptr<const WorkspaceBufferLease> storage,
        std::uint64_t generation,std::vector<ProjectionRequestPublication> entries)
        :storage_(std::move(storage)),generation_(generation),entries_(std::move(entries)) {}
    const std::shared_ptr<const WorkspaceBufferLease> storage_;
    const std::uint64_t generation_;
    const std::vector<ProjectionRequestPublication> entries_;
};
}
