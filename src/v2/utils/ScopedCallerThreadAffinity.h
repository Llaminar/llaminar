/**
 * @file ScopedCallerThreadAffinity.h
 * @brief Preserve exact caller-owned CPU binding across cold native discovery.
 *
 * Vendor discovery may retarget its calling thread. It cannot change the CPU
 * ownership later published by the application. This scope neither expands a
 * binding nor updates OpenMP places or any existing worker. Native operations
 * target Linux pid=0 (the calling thread); the internal operation type permits fast
 * authored replay without GPU startup or assuming multiple available CPUs.
 */
#pragma once

#include <sched.h>
#include <cerrno>
#include <cstdio>
#include <exception>
#include <system_error>

namespace llaminar2
{
    namespace detail
    {
        /** @brief Native access to this caller only; errors retain their errno. */
        struct NativeCallerThreadAffinity
        {
            /** @return Zero after reading this thread's exact scheduler mask, or errno. */
            static int read(cpu_set_t &mask) noexcept
            {
                CPU_ZERO(&mask);
                return sched_getaffinity(0,sizeof(mask),&mask)==0?0:errno;
            }
            /** @return Zero after restoring this caller's mask, or errno. */
            static int write(const cpu_set_t &mask) noexcept
            {
                return sched_setaffinity(0,sizeof(mask),&mask)==0?0:errno;
            }
        };

        /**
         * @brief Own one exact caller binding through success or exception.
         * @tparam Access Native calling-thread operations, or authored replay operations.
         */
        template<class Access>
        class ScopedCallerThreadAffinityImpl final
        {
        public:
            /** @brief Capture before borrowing the thread; failure aborts before native work. */
            ScopedCallerThreadAffinityImpl()
            {
                if(const int error=Access::read(original_))
                    throw std::system_error(error,std::generic_category(),"Cannot retain native discovery caller affinity");
                if(CPU_COUNT(&original_)==0)
                    throw std::system_error(EPROTO,std::generic_category(),"Native discovery caller affinity is empty");
            }
            /** @brief Restore before returning/unwinding; a lost binding is fatal. */
            ~ScopedCallerThreadAffinityImpl() noexcept
            {
                cpu_set_t actual{};
                if(const int error=Access::read(actual))fail(error);
                if(CPU_EQUAL(&actual,&original_))return;
                if(const int error=Access::write(original_))fail(error);
                if(const int error=Access::read(actual))fail(error);
                if(!CPU_EQUAL(&actual,&original_))fail(EPROTO);
            }
            ScopedCallerThreadAffinityImpl(const ScopedCallerThreadAffinityImpl&)=delete;
            ScopedCallerThreadAffinityImpl& operator=(const ScopedCallerThreadAffinityImpl&)=delete;
        private:
            /** @brief Never continue or finish unwinding with uncertain caller ownership.
             * @param error Original operation error, or EPROTO for a contradictory result.
             */
            [[noreturn]] static void fail(int error) noexcept
            {
                std::fprintf(stderr,"Fatal: cannot restore exact native discovery caller affinity (error=%d)\n",error);
                std::terminate();
            }
            cpu_set_t original_{}; ///< Exact original mask, not a NUMA/rank estimate.
        };
    }

    /** @brief Cold native discovery preserves only the calling OS thread. */
    using ScopedCallerThreadAffinity =
        detail::ScopedCallerThreadAffinityImpl<detail::NativeCallerThreadAffinity>;
}
