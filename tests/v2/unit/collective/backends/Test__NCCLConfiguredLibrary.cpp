/**
 * @file Test__NCCLConfiguredLibrary.cpp
 * @brief Replay actual NCCL/RCCL loader path ownership without loading a DSO.
 * This TU gives the source body a distinct namespace and intercepts only loader
 * and logging boundaries. It never loads an SDK/DSO, resolves symbols, creates
 * a communicator or calls a CUDA/runtime API. The original-source mode requires
 * all twenty exact legacy path and partial-admission observations and exits successfully
 * only after their assertions pass. Explicit successful-symbol ABI
 * coverage remains in NativeRowsABI.
 */
#include "collective/NativeCollectiveRows.h"
#include "collective/backends/NCCLNetworkPolicy.h"
#include "utils/Logger.h"
#include <climits>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace
{
#if defined(LLAMINAR_RCCL_PATH_REPLAY)
    constexpr const char *selectedLibrary = "/unit sdk selected/lib/librccl.so.1";
    constexpr const char *vendorName = "RCCL";
#else
    constexpr const char *selectedLibrary = "/unit sdk selected/lib/libllaminar_nccl.so.2";
    constexpr const char *vendorName = "NCCL";
#endif
    struct Observation
    {
        std::vector<std::string> paths;
        std::vector<int> flags;
        unsigned symbolCalls = 0, closeCalls = 0;
        bool admitSymbols = false;
        std::string missingSymbol;
        unsigned liveCallbackCloses = 0, safeCallbackCloses = 0;
    } observation;
    /**
     * @brief Observe the actual production open boundary without loading a DSO.
     * @param path Exact path selected by the production body.
     * @param flags Production symbol resolution/isolation flags.
     * @return An inert non-DSO handle for symbol-admission traces; otherwise null.
     */
    void *recordOpen(const char *path, int flags)
    {
        observation.paths.emplace_back(path ? path : "<null>");
        observation.flags.push_back(flags);
        static std::byte inertHandle{};
        return observation.admitSymbols ? &inertHandle : nullptr;
    }
    /** @brief Trap any attempted native ABI invocation; path tests resolve only. */
    void unexpectedNativeCall()
    {
        std::abort();
    }
    /** @brief Return inert uncalled symbols only for explicit successful-load traces. */
    void *recordSymbol(void *, const char *name)
    {
        ++observation.symbolCalls;
        return observation.admitSymbols && observation.missingSymbol != name
                   ? reinterpret_cast<void *>(&unexpectedNativeCall)
                   : nullptr;
    }
    /** @brief Observe withdrawal at the actual close boundary after source definitions. */
    int recordClose(void *);
    /** @brief Supply deterministic loader diagnostics without querying a real runtime. */
    char *recordError()
    {
        static char value[] = "inert missing dependency";
        return value;
    }
} // namespace

#if defined(LLAMINAR_RCCL_PATH_REPLAY)
#undef LLAMINAR_RCCL_LIBRARY_PATH
#if defined(LLAMINAR_NCCL_TEST_EMPTY_CONFIGURATION)
#define LLAMINAR_RCCL_LIBRARY_PATH ""
#elif !defined(LLAMINAR_NCCL_TEST_UNCONFIGURED)
#define LLAMINAR_RCCL_LIBRARY_PATH "/unit sdk selected/lib/librccl.so.1"
#endif
#ifndef LLAMINAR_NCCL_LOADER_CPP
#define LLAMINAR_NCCL_LOADER_CPP "collective/backends/RCCLDynamicLoader.cpp"
#endif
#else
#undef LLAMINAR_NCCL_LIBRARY_PATH
#if defined(LLAMINAR_NCCL_TEST_EMPTY_CONFIGURATION)
#define LLAMINAR_NCCL_LIBRARY_PATH ""
#elif !defined(LLAMINAR_NCCL_TEST_UNCONFIGURED)
#define LLAMINAR_NCCL_LIBRARY_PATH "/unit sdk selected/lib/libllaminar_nccl.so.2"
#endif
#ifndef LLAMINAR_NCCL_LOADER_CPP
#define LLAMINAR_NCCL_LOADER_CPP "collective/backends/NCCLDynamicLoader.cpp"
#endif
#endif
#undef LOG_DEBUG
#undef LOG_INFO
#undef LOG_ERROR
#undef LOG_WARN
#define LOG_DEBUG(...)                                                                             \
    do                                                                                             \
    {                                                                                              \
    } while (false)
#define LOG_INFO(...)                                                                              \
    do                                                                                             \
    {                                                                                              \
    } while (false)
#define LOG_ERROR(...)                                                                             \
    do                                                                                             \
    {                                                                                              \
    } while (false)
#define LOG_WARN(...) LOG_ERROR(__VA_ARGS__)
#if defined(LLAMINAR_RCCL_PATH_REPLAY)
#define rccl_dynamic rccl_configured_path_replay
#else
#define nccl_dynamic nccl_configured_path_replay
#endif
#define dlopen recordOpen
#define dlsym recordSymbol
#define dlclose recordClose
#define dlerror recordError
#include LLAMINAR_NCCL_LOADER_CPP
#undef dlerror
#undef dlclose
#undef dlsym
#undef dlopen
#undef nccl_dynamic
#undef rccl_dynamic

namespace
{
#if defined(LLAMINAR_RCCL_PATH_REPLAY)
    namespace loader = llaminar2::rccl_configured_path_replay;
#else
    namespace loader = llaminar2::nccl_configured_path_replay;
#endif
    /** @brief Assert close ordering without invoking any unresolved/stale native address. */
    int recordClose(void *)
    {
        ++observation.closeCalls;
        if (loader::fp_ncclGetUniqueId)
            ++observation.liveCallbackCloses;
        else
        {
            if (loader::ncclGetUniqueId(nullptr) == loader::ncclInternalError)
                ++observation.safeCallbackCloses;
        }
        return 0;
    }

    /** @brief Clear passive observations only after the actual loader retires its state. */
    class NCCLConfiguredLibrary : public ::testing::Test
    {
    protected:
        /** @brief Start each trace with the actual loader unloaded and no observations. */
        void SetUp() override
        {
            loader::unload();
            observation = {};
        }
        /** @brief Retire any test-local loader state before its observations disappear. */
        void TearDown() override
        {
            loader::unload();
        }
    };

    TEST_F(NCCLConfiguredLibrary, DefaultUsesExactConfiguredPath)
    {
        for (int replay = 0; replay < 20; ++replay)
        {
            observation = {};
            EXPECT_FALSE(loader::load(nullptr));
#if defined(LLAMINAR_NCCL_TEST_ORIGINAL) && !defined(LLAMINAR_RCCL_PATH_REPLAY)
            ASSERT_EQ(observation.paths.size(), 1u);
            EXPECT_EQ(observation.paths.front(), "libllaminar_nccl.so.2");
            EXPECT_EQ(observation.flags.front(), RTLD_NOW | RTLD_LOCAL);
            EXPECT_NE(std::string(loader::getLastError()).find("libllaminar_nccl.so.2"),
                      std::string::npos);
#elif defined(LLAMINAR_NCCL_TEST_UNCONFIGURED) || defined(LLAMINAR_NCCL_TEST_EMPTY_CONFIGURATION)
            EXPECT_TRUE(observation.paths.empty());
            EXPECT_NE(std::string(loader::getLastError()).find(vendorName), std::string::npos);
#else
            ASSERT_EQ(observation.paths.size(), 1u);
            EXPECT_EQ(observation.paths.front(), selectedLibrary);
            EXPECT_EQ(observation.flags.front(), RTLD_NOW | RTLD_LOCAL);
            EXPECT_NE(std::string(loader::getLastError()).find(selectedLibrary), std::string::npos);
#endif
            EXPECT_FALSE(loader::isLoaded());
            EXPECT_EQ(observation.symbolCalls, 0u);
            EXPECT_EQ(observation.closeCalls, 0u);
        }
    }

    TEST_F(NCCLConfiguredLibrary, ExplicitPathRemainsExactWithoutSearch)
    {
        constexpr const char *explicitPath = "/explicit other sdk/libllaminar_nccl.so.2";
        for (int replay = 0; replay < 20; ++replay)
        {
            observation = {};
            EXPECT_FALSE(loader::load(explicitPath));
            ASSERT_EQ(observation.paths.size(), 1u);
            EXPECT_EQ(observation.paths.front(), explicitPath);
            EXPECT_EQ(observation.flags.front(), RTLD_NOW | RTLD_LOCAL);
            EXPECT_NE(std::string(loader::getLastError()).find(explicitPath), std::string::npos);
            EXPECT_EQ(observation.symbolCalls, 0u);
        }
    }

    TEST_F(NCCLConfiguredLibrary, EmptyExplicitPathRejectsBeforeOpen)
    {
        for (int replay = 0; replay < 20; ++replay)
        {
            observation = {};
            EXPECT_FALSE(loader::load(""));
#if defined(LLAMINAR_NCCL_TEST_ORIGINAL) && !defined(LLAMINAR_RCCL_PATH_REPLAY)
            ASSERT_EQ(observation.paths.size(), 1u);
            EXPECT_EQ(observation.paths.front(), "");
            EXPECT_EQ(observation.flags.front(), RTLD_NOW | RTLD_LOCAL);
#else
            EXPECT_TRUE(observation.paths.empty());
#endif
            EXPECT_NE(std::string(loader::getLastError()).find(vendorName), std::string::npos);
            EXPECT_FALSE(loader::isLoaded());
            EXPECT_EQ(observation.symbolCalls, 0u);
        }
    }
    /** @test Retained selection identity cannot be replaced while its callbacks live. */
    TEST_F(NCCLConfiguredLibrary, LiveHandleRejectsDifferentSelectionUntilUnload)
    {
        constexpr const char *otherLibrary = "/unit other sdk/libllaminar_nccl.so.2";
        for (int replay = 0; replay < 20; ++replay)
        {
            loader::unload();
            observation = {};
            observation.admitSymbols = true;
            std::string requestedPath(selectedLibrary);
            ASSERT_TRUE(loader::load(requestedPath.c_str()));
            requestedPath.assign(otherLibrary); // The loader must own its admitted identity.
            ASSERT_TRUE(loader::isLoaded());
            ASSERT_EQ(observation.paths.size(), 1u);
            ASSERT_GT(observation.symbolCalls, 0u);
            const auto symbols = observation.symbolCalls;
            EXPECT_TRUE(loader::load(selectedLibrary));
#if defined(LLAMINAR_NCCL_TEST_ORIGINAL)
            EXPECT_TRUE(loader::load(nullptr));
#elif defined(LLAMINAR_NCCL_TEST_UNCONFIGURED) || defined(LLAMINAR_NCCL_TEST_EMPTY_CONFIGURATION)
            EXPECT_FALSE(loader::load(nullptr));
#else
            EXPECT_TRUE(loader::load(nullptr));
#endif
#if defined(LLAMINAR_NCCL_TEST_ORIGINAL)
            EXPECT_TRUE(loader::load(otherLibrary));
            EXPECT_TRUE(loader::load(""));
#else
            EXPECT_FALSE(loader::load(otherLibrary));
            const std::string mismatch = loader::getLastError();
            EXPECT_NE(mismatch.find(selectedLibrary), std::string::npos);
            EXPECT_NE(mismatch.find(otherLibrary), std::string::npos);
            EXPECT_FALSE(loader::load(""));
#endif
            EXPECT_TRUE(loader::isLoaded());
            EXPECT_EQ(observation.paths.size(), 1u);
            EXPECT_EQ(observation.symbolCalls, symbols);
            EXPECT_EQ(observation.closeCalls, 0u);
            loader::unload();
            EXPECT_FALSE(loader::isLoaded());
            EXPECT_EQ(observation.closeCalls, 1u);
            ASSERT_TRUE(loader::load(otherLibrary));
            ASSERT_EQ(observation.paths.size(), 2u);
            EXPECT_EQ(observation.paths.back(), otherLibrary);
#if defined(LLAMINAR_NCCL_TEST_ORIGINAL)
            EXPECT_TRUE(loader::load(nullptr));
#else
            // Configured/default selection cannot silently replace an explicit one.
            EXPECT_FALSE(loader::load(nullptr));
#endif
            loader::unload();
            EXPECT_EQ(observation.closeCalls, 2u);
        }
    }

    /** @test Failed middle/last resolution withdraws callbacks before close and allows reload. */
    TEST_F(NCCLConfiguredLibrary, PartialAdmissionWithdrawsCallbacksBeforeCloseAndReload)
    {
        for (const char *missing : {"ncclAllGather", "ncclGroupEnd"})
            for (int replay = 0; replay < 20; ++replay)
            {
                loader::unload();
                observation = {};
                observation.admitSymbols = true;
                observation.missingSymbol = missing;
                EXPECT_FALSE(loader::load(selectedLibrary));
                EXPECT_FALSE(loader::isLoaded());
                EXPECT_EQ(observation.paths.size(), 1u);
                EXPECT_EQ(observation.closeCalls, 1u);
                EXPECT_NE(std::string(loader::getLastError()).find(missing), std::string::npos);
#if defined(LLAMINAR_NCCL_TEST_ORIGINAL) && !defined(LLAMINAR_RCCL_PATH_REPLAY)
                EXPECT_EQ(observation.liveCallbackCloses, 1u);
                EXPECT_EQ(observation.safeCallbackCloses, 0u);
                EXPECT_NE(loader::fp_ncclGetUniqueId, nullptr);
#else
                EXPECT_EQ(observation.liveCallbackCloses, 0u);
                EXPECT_EQ(observation.safeCallbackCloses, 1u);
                EXPECT_EQ(loader::ncclGetUniqueId(nullptr), loader::ncclInternalError);
                EXPECT_EQ(loader::ncclCommInitRank(nullptr, 1, {}, 0), loader::ncclInternalError);
                EXPECT_EQ(loader::ncclAllReduce(nullptr, nullptr, 0, loader::ncclFloat32,
                                                loader::ncclSum, nullptr, nullptr),
                          loader::ncclInternalError);
                EXPECT_EQ(loader::ncclGroupStart(), loader::ncclInternalError);
#endif
                // Even the original faulty load remains safely retireable via the
                // actual corrected unload boundary; no stale callback is invoked.
                loader::unload();
                EXPECT_EQ(loader::ncclGetUniqueId(nullptr), loader::ncclInternalError);
                observation = {};
                observation.admitSymbols = true;
                ASSERT_TRUE(loader::load(selectedLibrary));
                EXPECT_TRUE(loader::isLoaded());
                loader::unload();
                EXPECT_FALSE(loader::isLoaded());
                EXPECT_EQ(observation.liveCallbackCloses, 0u);
                EXPECT_EQ(observation.safeCallbackCloses, 1u);
            }
    }

#if defined(LLAMINAR_RCCL_PATH_REPLAY)
    /** @test Optional RCCL entrypoints remain optional across retained-path replay. */
    TEST_F(NCCLConfiguredLibrary, OptionalSymbolsDoNotChangeSelectionContract)
    {
        constexpr const char *otherLibrary = "/unit other sdk/librccl.so.1";
        for (const char *optional : {"ncclCommAbort", "ncclCommFinalize"})
            for (int replay = 0; replay < 20; ++replay)
            {
                loader::unload();
                observation = {};
                observation.admitSymbols = true;
                observation.missingSymbol = optional;
                ASSERT_TRUE(loader::load(selectedLibrary));
                EXPECT_TRUE(loader::isLoaded());
                EXPECT_TRUE(loader::load(selectedLibrary));
#if defined(LLAMINAR_NCCL_TEST_ORIGINAL)
                EXPECT_TRUE(loader::load(otherLibrary));
#else
                EXPECT_FALSE(loader::load(otherLibrary));
#endif
                EXPECT_EQ(observation.paths.size(), 1u);
                if (std::string(optional) == "ncclCommAbort")
                    EXPECT_EQ(loader::fp_ncclCommAbort, nullptr);
                else
                    EXPECT_EQ(loader::fp_ncclCommFinalize, nullptr);
                loader::unload();
                EXPECT_EQ(observation.safeCallbackCloses, 1u);
                EXPECT_EQ(loader::ncclCommInitAll(nullptr, 0, nullptr), loader::ncclInternalError);
            }
    }
#endif

} // namespace
