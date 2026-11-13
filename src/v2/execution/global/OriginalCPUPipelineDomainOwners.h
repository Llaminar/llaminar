/**
 * @file OriginalCPUPipelineDomainOwners.h
 * @brief Explicit terminal lifetime of original admitted CPU pipeline domains.
 *
 * Ordinary stage entries borrow these actual contexts. The global owner first
 * destroys every local stage graph/worker/completion borrower, then retires
 * original native mappings and communicators in fixed reverse stage order.
 * PMA/process-session commitments outlive that terminal native acknowledgement.
 */
#pragma once
#include "collective/GlobalTPContext.h"
#include <exception>
namespace llaminar2 {
class DomainCommunicatorRegistry;
/** @brief Sole original CPU domain-owner set, never a graph or participant substitute. */
class OriginalCPUPipelineDomainOwners final {
public:
    OriginalCPUPipelineDomainOwners()=default;
    OriginalCPUPipelineDomainOwners(const OriginalCPUPipelineDomainOwners&)=delete;
    OriginalCPUPipelineDomainOwners& operator=(const OriginalCPUPipelineDomainOwners&)=delete;
    /** @brief Unretired native duties cannot silently release opaque PMA commitments. */
    ~OriginalCPUPipelineDomainOwners() noexcept {if(!domains_.empty())std::terminate();}
    /**
     * @brief Retire only after every original graph/worker borrower has actually disappeared.
     * @throws std::logic_error if any live borrower remains; original context enforces exact identity.
     * @note Collective only among each original domain's members, in reverse stage order.
     */
    void retireAfterGraphRelease() {
        while(!domains_.empty()) {
            if(!domains_.back()->retireOriginalCPUExpertDomain()) std::terminate();
            domains_.pop_back();
        }
    }
private:
    friend class DomainCommunicatorRegistry;
    /** @brief Retain only the actual just-created admitted original context. */
    void retain(std::shared_ptr<GlobalTPContext> context) {
        if(!context || !context->authenticatesOriginalCPUExpertBorrow(context))
            throw std::invalid_argument("CPU pipeline owner requires its actual admitted native domain");
        domains_.push_back(std::move(context));
    }
    std::vector<std::shared_ptr<GlobalTPContext>> domains_;
};
}
