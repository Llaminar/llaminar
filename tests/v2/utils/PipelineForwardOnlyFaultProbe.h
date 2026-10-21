/**
 * @file PipelineForwardOnlyFaultProbe.h
 * @brief Isolated-executable observations of native PP enqueue and retirement.
 *
 * No callback performs device work or reads controller/model data. An armed
 * caller first observes every real parent launch, then overrides exactly one
 * successful terminal copy's RETURN status. Its real enqueue is unchanged.
 * Successful native event fences must cover that copy before public return and
 * every parent before destruction. This models an API error return after an
 * accepted enqueue, not a hardware failure or a native runtime sticky error.
 */
#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <thread>
#include <vector>

namespace llaminar2::test::pp_fault
{
/** @brief Resolve the real vendor DSO entry; missing interposition closure is fatal. */
template<class Function> Function original(const char *name)
{
    auto *symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) std::abort();
    return reinterpret_cast<Function>(symbol);
}

/** @brief Bounded observations only; no replica of production execution state. */
class Probe final
{
    struct Parent {
        void *executable = nullptr, *stream = nullptr;
        int device = -1;
        uint64_t launched = 0, last_copy = 0, completed = 0;
        bool destroyed = false;
    };
    struct Event { void *handle = nullptr, *stream = nullptr; int device = -1; uint64_t recorded = 0; };
    inline static std::atomic<Probe *> active_{nullptr};
    std::mutex mutex_;
    std::thread::id caller_ = std::this_thread::get_id();
    std::vector<int> expected_;
    std::array<Parent, 4> parents_{};
    std::array<Event, 128> events_{};
    size_t parent_count_ = 0, copy_count_ = 0, selected_ = 0, injected_bytes_ = 0;
    uint64_t clock_ = 0;
    bool injected_ = false, returned_ = false, copies_retired_before_return_ = false, valid_ = true;

    /** @brief Find one previously observed exact stream/device pair. */
    Parent *parent(void *stream, int device)
    {
        for (size_t i = 0; i < parent_count_; ++i)
            if (parents_[i].stream == stream && parents_[i].device == device) return &parents_[i];
        return nullptr;
    }
public:
    /** @brief Prepare after serving admission/prefill/pending teacher; launch observations arm injection. */
    Probe(std::vector<int> devices, size_t terminal_copy) : expected_(std::move(devices)), selected_(terminal_copy)
    {
        if (expected_.empty() || expected_.size() > parents_.size() || selected_ == 0) std::abort();
        Probe *empty = nullptr;
        if (!active_.compare_exchange_strong(empty, this)) std::abort();
    }
    /** @brief Disarm after runner teardown; all callbacks remain owned until then. */
    ~Probe() { active_.store(nullptr); }
    Probe(const Probe &) = delete;
    Probe &operator=(const Probe &) = delete;
    /** @return Sole active isolated test probe, or null for an ordinary forwarded call. */
    static Probe *active() { return active_.load(); }

    /** @brief Count real accepted parent launches; no terminal injection before all peers. */
    void launch(void *executable, void *stream, int device, bool success)
    {
        std::lock_guard lock(mutex_);
        if (std::this_thread::get_id() != caller_ || returned_) return;
        if (!success || !executable || !stream || parent_count_ >= expected_.size()) { valid_ = false; return; }
        for (size_t i = 0; i < parent_count_; ++i)
            if (parents_[i].executable == executable || parents_[i].device == device) { valid_ = false; return; }
        if (device != expected_[parent_count_]) valid_ = false;
        parents_[parent_count_++] = {executable, stream, device, ++clock_, 0, 0, false};
    }
    /** @return Whether this successful real D2H enqueue receives the one synthetic error return. */
    bool copy(void *destination, const void *source, size_t bytes, void *stream, int device, bool success)
    {
        std::lock_guard lock(mutex_);
        if (std::this_thread::get_id() != caller_ || returned_) return false;
        auto *owner = parent(stream, device);
        if (!owner) return false;
        if (!success || !destination || !source || !bytes || parent_count_ != expected_.size()) { valid_ = false; return false; }
        owner->last_copy = ++clock_;
        if (++copy_count_ != selected_ || injected_) return false;
        injected_ = true;
        injected_bytes_ = bytes;
        return true;
    }
    /** @brief Bind the actual native event record to its exact stream and device. */
    void record(void *handle, void *stream, int device, bool success)
    {
        std::lock_guard lock(mutex_);
        if (!parent(stream, device)) return;
        if (!success || !handle) { valid_ = false; return; }
        for (auto &event : events_)
            if (!event.handle || event.handle == handle) { event = {handle, stream, device, ++clock_}; return; }
        valid_ = false;
    }
    /** @brief Only a real successful event synchronization can prove prior stream work complete. */
    void wait(void *handle, int device, bool success)
    {
        std::lock_guard lock(mutex_);
        for (const auto &event : events_)
            if (event.handle == handle && event.device == device)
            {
                auto *owner = parent(event.stream, device);
                if (!owner || !success) { valid_ = false; return; }
                if (event.recorded > owner->launched && event.recorded > owner->completed)
                    owner->completed = event.recorded;
                return;
            }
    }
    /** @brief Authenticate retirement before the actual parent executable destruction call. */
    void destroy(void *executable, int device, bool before, bool success = true)
    {
        std::lock_guard lock(mutex_);
        for (size_t i = 0; i < parent_count_; ++i)
            if (parents_[i].executable == executable)
            {
                auto &owner = parents_[i];
                if (device != owner.device || !success) valid_ = false;
                if (before && (owner.completed <= owner.launched || owner.completed < owner.last_copy)) valid_ = false;
                if (!before) owner.destroyed = success;
                return;
            }
    }
    /** @brief Check accepted host writes have completed before the public force call returns. */
    void publicReturned()
    {
        std::lock_guard lock(mutex_);
        returned_ = true;
        copies_retired_before_return_ = true;
        for (size_t i = 0; i < parent_count_; ++i)
            if (parents_[i].last_copy && parents_[i].completed < parents_[i].last_copy)
                copies_retired_before_return_ = false;
    }
    /** @return Exact synthetic-return and destination-retirement observations. */
    bool returnedSafely(size_t expected_bytes)
    {
        std::lock_guard lock(mutex_);
        return valid_ && injected_ && returned_ && copies_retired_before_return_ &&
            parent_count_ == expected_.size() && copy_count_ == selected_ && injected_bytes_ == expected_bytes;
    }
    /** @return Whether every original accepted parent was fenced before actual teardown. */
    bool retiredSafely()
    {
        std::lock_guard lock(mutex_);
        if (!valid_ || parent_count_ != expected_.size()) return false;
        for (size_t i = 0; i < parent_count_; ++i) if (!parents_[i].destroyed) return false;
        return true;
    }
};
} // namespace llaminar2::test::pp_fault
