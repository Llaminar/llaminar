/**
 * @file HostSparseQueryResponseScope.h
 * @brief One immutable sparse-query admission across a host-owned response call.
 */
#pragma once
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include <optional>
#include <stdexcept>
namespace llaminar2
{
/**
 * @brief Borrow an outer response admission or own exactly one idle begin/end pair.
 *
 * No response progress is recorded here. The optional contains only immutable
 * initial admission; budgets and token progress remain with their existing owner.
 */
class HostSparseQueryResponseScope
{
public:
    /**
     * @brief Establish one response scope, with nested calls borrowing its identity.
     * @param runner Existing composed execution owner.
     * @param active Owner's single immutable host admission slot.
     * @param request New boundary admission, absent for a nested borrow or resident path.
     * @throws std::invalid_argument If admission is invalid or nesting supplies another request.
     * @throws std::runtime_error If a participant rejects admission.
     */
    HostSparseQueryResponseScope(IInferenceRunner &runner,
        std::optional<DeviceGenerationAdmissionRequest> &active,
        std::optional<DeviceGenerationAdmissionRequest> request)
        : runner_(runner), active_(active)
    {
        if (active_)
        {
            if (request) throw std::invalid_argument("Nested sparse-query response cannot replace admission");
            return;
        }
        if (!request) return;
        if (!request->valid()) throw std::invalid_argument("Invalid host sparse-query response admission");
        try
        {
            if (!runner_.admitSparseQueryExecutionProgram(*request))
                throw std::runtime_error("Participant rejected host sparse-query response admission");
            active_.emplace(std::move(*request));
        }
        catch (...)
        {
            // A previous child may have accepted before another child rejected.
            runner_.endSparseQueryExecutionProgram();
            throw;
        }
        owns_ = true;
    }
    /** @brief Expire policy on every exit without changing counters, data or graph owners. */
    ~HostSparseQueryResponseScope()
    {
        if (owns_)
        {
            runner_.endSparseQueryExecutionProgram();
            active_.reset();
        }
    }
    HostSparseQueryResponseScope(const HostSparseQueryResponseScope &) = delete;
    HostSparseQueryResponseScope &operator=(const HostSparseQueryResponseScope &) = delete;
private:
    IInferenceRunner &runner_;
    std::optional<DeviceGenerationAdmissionRequest> &active_;
    bool owns_ = false;
};
}
