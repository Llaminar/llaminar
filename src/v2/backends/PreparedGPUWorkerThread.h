/**
 * @file PreparedGPUWorkerThread.h
 * @brief Exact source-owned stack and compiled object for the original GPU worker.
 *
 * Stack/guard geometry is selected from the original pthread defaults before
 * native runtime birth. The one Preparing PMA preclaims stack pages before
 * mmap; the prepared context appends its exact compiled object extent in the
 * same root. Callbacks remain retained until actual destruction/join and unmap. Driver/libc
 * opaque storage is a separate named commitment, never part of this extent.
 */
#pragma once
#include <cstddef>
#include <memory>
#include <new>
#include <thread>
#include <string>
#include <pthread.h>
namespace llaminar2 {
class NativeStoragePreparationPmaBinding;
struct NativeStorageCreatorDeclaration;
namespace test { struct PreparedGPUWorkerThreadReplayAccess; struct NativeContextPreparationReplayAccess; struct NativeContextBirthDiagnosticAccess; }
class PreparedGPUContextBirth;
class PreparedGPUContextFactory;
class GPUDeviceContextPool;
/** @brief Exact virtual guard and physical prefaulted stack extents from pthread source policy. */
class GPUWorkerStackProgram final {
public:
    /** @brief Read original process thread defaults without creating a native device or thread. */
    static GPUWorkerStackProgram originalDefaults();
    /** @return Canonical digest of original native defaults, excluding later stack-address binding. */
    std::string identity()const;
    /** @brief Shared checked original stack/object geometry, without mapping or spawning.
     * @param stack Original stack bytes. @param guard Virtual guard bytes. @param page Native page bytes.
     * @param object Compiled object size. @param alignment Compiled alignment. @return Physical stack/object extent. */
    static std::size_t planningAllocationBytes(std::size_t stack,std::size_t guard,std::size_t page,std::size_t object,std::size_t alignment);

    std::size_t stackBytes() const noexcept { return stack_; }
    std::size_t guardBytes() const noexcept { return guard_; }
    std::size_t pageBytes() const noexcept { return page_; }
    /** @return Exact physical stack plus compiled context-object pages, excluding virtual guard. */
    std::size_t allocationBytes() const noexcept { return stack_+object_pages_; }
    /** @return Actual typed context object extent, distinct from page rounding. */
    std::size_t contextObjectBytes() const noexcept {return object_;}
    std::size_t contextObjectAlignment() const noexcept {return object_alignment_;}
private:
    friend class PreparedGPUWorkerThread;
    friend class PreparedGPUContextBirth;
    friend struct test::NativeContextPreparationReplayAccess;
    friend struct test::NativeContextBirthDiagnosticAccess;
    /** @brief Append only the certified compiled typed object extent before physical mapping. */
    GPUWorkerStackProgram withContextObject(std::size_t bytes,std::size_t alignment) const;
    struct Attributes;
    GPUWorkerStackProgram(std::size_t stack,std::size_t guard,std::size_t page,std::shared_ptr<Attributes> original)
        :stack_(stack),guard_(guard),page_(page),attributes_(std::move(original)){}
    std::size_t stack_,guard_,page_;
    std::size_t object_=0,object_alignment_=1,object_pages_=0;
    std::shared_ptr<Attributes> attributes_; ///< Original complete defaults, including affinity/scheduling; creation serializes its stack binding.
};
/** @brief Actual mmap stack root retaining its exact original native PMA claim. */
class PreparedGPUWorkerStack final {
public:
    /** @brief Preclaim, map and prefault the exact source-selected stack; guards are never writable. */
    static std::shared_ptr<PreparedGPUWorkerStack> create(std::shared_ptr<NativeStoragePreparationPmaBinding>,GPUWorkerStackProgram);
    /** @brief Unmap only after a successful join; cleanup failure is fatal before claim release. */
    ~PreparedGPUWorkerStack() noexcept;
    PreparedGPUWorkerStack(const PreparedGPUWorkerStack&)=delete;
    PreparedGPUWorkerStack& operator=(const PreparedGPUWorkerStack&)=delete;
    /** @return Actual stack creator declaration consumed by the same final source-admission snapshot. */
    NativeStorageCreatorDeclaration creatorDeclaration() const;
    const GPUWorkerStackProgram& program() const noexcept;
private:
    friend class PreparedGPUWorkerThread;
    friend class PreparedGPUContextFactory;
    friend class GPUDeviceContextPool;
    friend struct test::NativeContextPreparationReplayAccess;
    /** @brief Validate compiled type geometry and return its preclaimed source-owned placement. */
    void* contextObjectStorage(std::size_t bytes,std::size_t alignment) const;
    /** @brief Reject a prepared worker whose base pointer escapes the exact source object extent. */
    void requireContextObjectPointer(const void*)const;
    struct State;
    explicit PreparedGPUWorkerStack(std::unique_ptr<State>);
    std::unique_ptr<State> state_;
};
/** @brief Narrow original context-worker thread owner; other thread families are unaffected. */
class PreparedGPUWorkerThread final {
public:
    PreparedGPUWorkerThread()=default;
    /** @brief Preserve ordinary existing context behavior as its explicit original std::thread owner. */
    PreparedGPUWorkerThread& operator=(std::thread&& thread);
    /** @brief Require an explicit successful join just as std::thread does. */
    ~PreparedGPUWorkerThread() noexcept;
    /** @brief Start the original worker entry on one admitted exact stack; failure creates no running alias. */
    void start(std::shared_ptr<PreparedGPUWorkerStack>,void(*entry)(void*),void* argument);
    bool joinable() const noexcept;
    /** @brief Join before dropping stack/root ownership; a failed native join terminates before cleanup. */
    void join();
private:
    friend struct test::PreparedGPUWorkerThreadReplayAccess;
    using Create=int(*)(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
    using Join=int(*)(pthread_t,void**);
    /** @brief Enter only the original source worker body; no wrapper inference or policy executes here. */
    static void* invoke(void*);
    std::thread ordinary_;
    pthread_t prepared_{};
    std::shared_ptr<PreparedGPUWorkerStack> stack_;
    void(*entry_)(void*)=nullptr;
    void* argument_=nullptr;
    bool running_=false;
    Create create_=pthread_create;
    Join join_=pthread_join;
};
}
