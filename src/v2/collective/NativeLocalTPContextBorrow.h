/**
 * @file NativeLocalTPContextBorrow.h
 * @brief Revocable original LocalTP borrowing without reverse communicator ownership.
 *
 * Every public factory receives its borrow from the original context itself.
 * Native teardown revokes all issued borrows and waits for in-flight host
 * submissions to leave their scoped use. A retained graph binding therefore
 * cannot dereference a destroyed context, including when never installed in it.
 * This is host lifetime protection, never GPU execution or progress authority.
 */
#pragma once
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

namespace llaminar2 {
class ILocalTPContext;
class NativeLocalTPContextLifetime;
/** @brief One original context pointer protected by its scoped-use mutex. */
struct NativeLocalTPContextBorrowState final {
    std::shared_mutex mutex;
    ILocalTPContext* context;
    /** @brief Bind the actual original interface, without assuming shared ownership. */
    explicit NativeLocalTPContextBorrowState(ILocalTPContext& original):context(&original){}
};
/** @brief Move-only host submission scope; revocation waits for its departure. */
class NativeLocalTPContextUse final {
public:
    NativeLocalTPContextUse(NativeLocalTPContextUse&&)=default;
    /** @return The checked original context while this scope prevents teardown. */
    ILocalTPContext& context()const noexcept{return *state_->context;}
private:
    friend class NativeLocalTPContextBorrow;
    /** @brief Acquire the owner-issued lifetime guard before examining its pointer. */
    explicit NativeLocalTPContextUse(std::shared_ptr<NativeLocalTPContextBorrowState> state)
        :state_(std::move(state)),lock_(state_->mutex) {
        if(!state_->context)throw std::logic_error("Original native LocalTP context borrow has retired");
    }
    std::shared_ptr<NativeLocalTPContextBorrowState> state_;
    std::shared_lock<std::shared_mutex> lock_;
};
/** @brief Copyable revocable identity; retaining it never retains its communicator. */
class NativeLocalTPContextBorrow final {
public:
    NativeLocalTPContextBorrow()=default;
    /** @return Whether both borrows name the same actual original lifetime authority. */
    bool sameOriginal(const NativeLocalTPContextBorrow& other)const noexcept{return state_ && state_==other.state_;}
    /** @return Scoped original context use, rejecting missing or revoked authority. */
    NativeLocalTPContextUse acquire()const {
        if(!state_)throw std::logic_error("Missing original native LocalTP context borrow");
        return NativeLocalTPContextUse(state_);
    }
private:
    friend class NativeLocalTPContextLifetime;
    /** @brief Only the original context lifetime may issue this borrow. */
    explicit NativeLocalTPContextBorrow(std::shared_ptr<NativeLocalTPContextBorrowState> state):state_(std::move(state)){}
    std::shared_ptr<NativeLocalTPContextBorrowState> state_;
};
/** @brief Unique original owner; destruction revokes even non-production interface implementations. */
class NativeLocalTPContextLifetime final {
    friend class ILocalTPContext;
    /** @brief Establish the one original revocation authority. */
    explicit NativeLocalTPContextLifetime(ILocalTPContext& original)
        :state_(std::make_shared<NativeLocalTPContextBorrowState>(original)){}
public:
    NativeLocalTPContextLifetime(const NativeLocalTPContextLifetime&)=delete;
    NativeLocalTPContextLifetime& operator=(const NativeLocalTPContextLifetime&)=delete;
    /** @brief Revoke any surviving borrows before releasing the owner's final state reference. */
    ~NativeLocalTPContextLifetime(){revoke();}
    /** @return A checked borrow of this same original context; no new context is constructed. */
    NativeLocalTPContextBorrow borrow()const{return NativeLocalTPContextBorrow(state_);}
    /** @brief Close host admission before native resources are torn down; repeat calls are harmless. */
    void revoke()noexcept {
        std::unique_lock lock(state_->mutex);
        state_->context=nullptr;
    }
private:
    std::shared_ptr<NativeLocalTPContextBorrowState> state_;
};
} // namespace llaminar2
