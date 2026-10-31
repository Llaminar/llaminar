/**
 * @file Test__MPISessionEnvironment.cpp
 * @brief C++17 replay of the live typed MPI bootstrap environment boundary.
 *
 * The independent helper and native module must observe ordered set/unset
 * transitions rather than a cached engine configuration. This standalone CPU
 * test uses the actual shared declaration and reader, with no MPI, core, SDK,
 * model or accelerator dependency. Every modified environment value is restored.
 */
#include "utils/MPISessionEnvironment.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>

namespace
{
    using namespace llaminar2::mpi_session;

    /** @brief Stop immediately on incomplete live-read or restoration observations. */
    void require(bool condition)
    {
        if (!condition)
            std::abort();
    }

    /** @brief Restore this process's exact original presence/value after one key replay. */
    class OriginalValue final
    {
    public:
        /** @brief Copy only this test's original value before the first mutation. */
        explicit OriginalValue(EnvironmentKey key) : name_(environmentName(key))
        {
            if (const char *value = processEnvironmentValue(key))
                value_ = value;
        }

        /** @brief Restore original unset, empty, or nonempty state before native owner retirement. */
        ~OriginalValue()
        {
            require(value_ ? ::setenv(name_, value_->c_str(), 1) == 0 : ::unsetenv(name_) == 0);
        }

    private:
        const char *name_;
        std::optional<std::string> value_;
    };
}

/** @brief Replay every declared live key across mutation, empty, and unset transitions. */
int main()
{
    using namespace llaminar2::mpi_session;
    static_assert(!std::is_invocable_v<decltype(processEnvironmentValue), const char *>);
    static_assert(!std::is_invocable_v<decltype(processEnvironmentValue), int>);
    unsigned observations = 0;
    for (int replay = 0; replay < 20; ++replay)
    {
        for (unsigned raw = 0; raw < static_cast<unsigned>(EnvironmentKey::Count); ++raw)
        {
            const auto key = static_cast<EnvironmentKey>(raw);
            OriginalValue original(key);
            const char *name = environmentName(key);
            require(::unsetenv(name) == 0 && processEnvironmentValue(key) == nullptr);
            require(::setenv(name, "", 1) == 0);
            const char *empty = processEnvironmentValue(key);
            require(empty != nullptr && *empty == '\0');
            require(::setenv(name, "first", 1) == 0 && std::strcmp(processEnvironmentValue(key), "first") == 0);
            require(::setenv(name, "changed", 1) == 0 && std::strcmp(processEnvironmentValue(key), "changed") == 0);
            require(::unsetenv(name) == 0 && processEnvironmentValue(key) == nullptr);
            observations += 5;
        }
        for (const auto malformed : {EnvironmentKey::Count, static_cast<EnvironmentKey>(255)})
        {
            bool rejected = false;
            try
            {
                (void)processEnvironmentValue(malformed);
            }
            catch (const std::invalid_argument &)
            {
                rejected = true;
            }
            require(rejected);
            ++observations;
        }
    }
    std::printf("{\"replays\":20,\"observations\":%u,\"complete\":true}\n", observations);
}
