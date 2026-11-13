/**
 * @file Test__ROCmHSAConfiguredLibrary.cpp
 * @brief Replay all actual HSA loader callers without a compiler SDK or runtime.
 *
 * The build retains original production method bodies. Only their loader/logging
 * boundaries and enclosing SDK-heavy class declaration are inert. Failed opens
 * cannot reach symbols, close a nonhandle or read hardware metadata. Original
 * mode requires the exact former bare-name observation as a successful witness.
 */
#include <gtest/gtest.h>
#include <dlfcn.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

namespace {
[[maybe_unused]] constexpr auto selectedHSA = "/unit selected SDK/lib/libhsa-runtime64.so.1.0";
std::vector<std::string> paths;
std::vector<int> flags;
unsigned symbols = 0, closes = 0;
/** @brief Observe a failed original load without calling any dynamic loader. */
void *recordOpen(const char *path, int mode) { paths.emplace_back(path ? path : "<null>"); flags.push_back(mode); return nullptr; }
/** @brief Any symbol lookup after a rejected open is an ownership error. */
void *recordSymbol(void *, const char *) { ++symbols; return nullptr; }
/** @brief A failed open never owns a handle to release. */
int recordClose(void *) { ++closes; return 0; }
/** @return An inert diagnostic without querying loader state. */
char *recordError() { static char message[] = "authored missing selected HSA"; return message; }
}
namespace llaminar2 {
/** @brief Metadata is unavailable in this path-only proof; any read fails. */
struct UnavailableMetadata { std::string name, isa_target; };
const std::vector<UnavailableMetadata> &rocmHardwareMetadata() { throw std::logic_error("HSA path replay cannot read native metadata"); }
/** @brief Minimal declaration around the unchanged five method bodies; no backend exists. */
class ROCmBackend {
public:
    bool hsaMemoryLock(void *, std::size_t, void **);
    void hsaMemoryUnlock(void *);
    bool hsaInteropMapBuffer(int, std::size_t *, void **);
    void hsaInteropUnmapBuffer(void *);
    bool getHsaAgent(int, std::uint64_t *);
private:
    int device_count_ = 1; ///< Authored argument-admission extent, not hardware evidence.
};
#define LLAMINAR_HSA_RUNTIME_LIBRARY_PATH selectedHSA
#define dlopen recordOpen
#define dlsym recordSymbol
#define dlclose recordClose
#define dlerror recordError
#define LOG_INFO(...) do { std::ostringstream message; message << __VA_ARGS__; } while (false)
#define LOG_WARN(...) do { std::ostringstream message; message << __VA_ARGS__; } while (false)
#define LOG_ERROR(...) do { std::ostringstream message; message << __VA_ARGS__; } while (false)
#include LLAMINAR_ROCM_HSA_REPLAY_CPP
#undef LOG_ERROR
#undef LOG_WARN
#undef LOG_INFO
#undef dlerror
#undef dlclose
#undef dlsym
#undef dlopen
#undef LLAMINAR_HSA_RUNTIME_LIBRARY_PATH
}

/** @test All original callers select exactly one SDK path and never search another library. */
TEST(ROCmHSAConfiguredLibrary, AllFiveCallersPreserveSelectedPathAndFailureLifetime)
{
    paths.clear();flags.clear();symbols=closes=0;
    llaminar2::ROCmBackend backend;
    std::byte host[8]{};void *pointer=nullptr;std::size_t extent=0;std::uint64_t agent=0;
    EXPECT_FALSE(backend.hsaMemoryLock(host,sizeof(host),&pointer));
    backend.hsaMemoryUnlock(host);
    EXPECT_FALSE(backend.hsaInteropMapBuffer(3,&extent,&pointer));
    backend.hsaInteropUnmapBuffer(host);
    EXPECT_FALSE(backend.getHsaAgent(0,&agent));
    ASSERT_EQ(paths.size(),5u);
    ASSERT_EQ(flags.size(),5u);
    for(std::size_t i=0;i<paths.size();++i) {
#ifdef LLAMINAR_ROCM_HSA_ORIGINAL
        EXPECT_EQ(paths[i],"libhsa-runtime64.so");
#else
        EXPECT_EQ(paths[i],selectedHSA);
#endif
        EXPECT_EQ(flags[i],RTLD_NOW|RTLD_GLOBAL);
    }
    EXPECT_EQ(symbols,0u);
    EXPECT_EQ(closes,0u);
}
