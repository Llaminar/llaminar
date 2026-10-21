/**
 * @file CPURequestStateBankView.h
 * @brief Borrow existing CPU recurrent request banks for read-only publication planning.
 *
 * These addresses refer to the kernel's sole accepted-state allocation. The
 * view neither allocates nor copies state, counts capacity, or changes ownership.
 * GPU implementations have no host view and never implement this CPU-only API.
 */
#pragma once
#include <span>
#include <cstddef>
namespace llaminar2 {
/** @brief Exact native bank geometry, stable until its current publication retires. */
struct CPURequestStateBankView {
    std::span<float> storage;
    int requests=0,stateElements=0;
    /** @brief Validate a requested prefix against the owner's actual native span. */
    bool contains(int count,int elements)const noexcept {
        return count>0 && elements>0 && count<=requests && elements==stateElements &&
            std::size_t(count)<=storage.size()/std::size_t(elements);
    }
};
/** @brief CPU-only read-only access to an already materialized request state owner. */
class ICPURequestStateBankView {
public:
    /** @brief Retire only the interface; the concrete kernel owns its native bank. */
    virtual ~ICPURequestStateBankView()=default;
    /** @brief Borrow exact existing backing without mutation, allocation, or workspace rebinding. */
    virtual CPURequestStateBankView cpuRequestStateBankView()noexcept=0;
};
}
