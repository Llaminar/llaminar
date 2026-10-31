/**
 * @file probe.cpp
 * @brief Actual SDK SHM lifecycle replay with real files and partial-create faults.
 *
 * Descriptor zero and caller-owned symlinks also exercise native branches.
 * Only fallocate/mmap outcomes are injected; the selected complete shmutils.cc
 * implements creation, attachment and retirement. GPU functions abort if used.
 * Every diagnostic leak is removed by its exact recorded inode, never by glob.
 */
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <initializer_list>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char* scratchDirectory = nullptr;

enum ncclResult_t
{
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5
};
enum ncclDebugLogLevel
{
    NCCL_LOG_NONE = 0
};
int ncclDebugLevel = 0;
std::uint64_t ncclDebugMask = 0;
/** @brief Suppress native diagnostics locally; all syscall outcomes are checked explicitly. */
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...)
{
}
using ncclShmHandle_t = void*;
ncclResult_t ncclShmOpen(char*, std::size_t, std::size_t, void**, void**, int, ncclShmHandle_t*);
ncclResult_t ncclShmClose(ncclShmHandle_t);
/** @name Forbidden GPU calls
 * Any GPU/runtime call invalidates this host-file certificate immediately.
 * @{
 */
extern "C" int hipHostRegister(void*, std::size_t, unsigned int)
{
    std::abort();
}
extern "C" int hipHostGetDevicePointer(void**, void*, unsigned int)
{
    std::abort();
}
extern "C" int hipHostUnregister(void*)
{
    std::abort();
}
extern "C" const char* hipGetErrorString(int)
{
    std::abort();
}
extern "C" int cudaHostRegister(void*, std::size_t, unsigned int)
{
    std::abort();
}
extern "C" int cudaHostGetDevicePointer(void**, void*, unsigned int)
{
    std::abort();
}
extern "C" int cudaHostUnregister(void*)
{
    std::abort();
}
extern "C" const char* cudaGetErrorString(int)
{
    std::abort();
}
extern "C" int cudaThreadExchangeStreamCaptureMode(void*)
{
    std::abort();
}
/** @} */
extern "C" int __real_fallocate(int, int, off_t, off_t);
extern "C" void* __real_mmap(void*, std::size_t, int, int, int, off_t);

enum class Fault
{
    None,
    Extent,
    Mapping
};
static Fault fault = Fault::None;
static struct stat latest_created
{
};
static int failed_names = 0;
static int replaced_existing_files = 0;
static bool inspectPathAllocation = false;
static void* failedPathAllocation = nullptr;
static unsigned failedOpenPathAllocations = 0;
extern "C" void* __real_malloc(std::size_t);
/** @brief Observe only the native path allocation after a creator open failed. */
extern "C" void* __wrap_malloc(std::size_t bytes)
{
    void* result = __real_malloc(bytes);
    if (inspectPathAllocation)
    {
        if (failedPathAllocation)
            std::abort();
        failedPathAllocation = result;
        ++failedOpenPathAllocations;
    }
    return result;
}
/** @brief Terminate the replay on the first absent ownership observation. */
static void require(bool condition, const char* detail)
{
    if (!condition)
    {
        std::fprintf(stderr, "%s (errno=%d)\n", detail, errno);
        std::exit(2);
    }
}
/** @brief Observe the exact created inode and inject only the requested extent failure. */
extern "C" int __wrap_fallocate(int fd, int mode, off_t offset, off_t length)
{
    require(fstat(fd, &latest_created) == 0, "cannot inspect actual created fd");
    if (fault == Fault::Extent)
    {
        errno = ENOSPC;
        return -1;
    }
    return __real_fallocate(fd, mode, offset, length);
}
/** @brief Inject only mapping failure; successful mappings retain ordinary native behavior. */
extern "C" void* __wrap_mmap(void* address, std::size_t bytes, int protection, int flags, int fd,
                             off_t offset)
{
    if (fault == Fault::Mapping)
    {
        errno = ENOMEM;
        return MAP_FAILED;
    }
    return __real_mmap(address, bytes, protection, flags, fd, offset);
}
/** @brief Require an unambiguous ENOENT result for an expected retired name. */
static bool absent(const char* name)
{
    struct stat actual
    {
    };
    if (lstat(name, &actual) == 0)
        return false;
    require(errno == ENOENT, "unexpected stat failure");
    return true;
}
/** @brief Authenticate the exact test-owned inode before any diagnostic unlink. */
static void requireIdentity(const char* name, const struct stat& original)
{
    struct stat actual
    {
    };
    require(lstat(name, &actual) == 0 && actual.st_dev == original.st_dev &&
                actual.st_ino == original.st_ino && actual.st_uid == getuid(),
            "recorded file identity changed");
}
/** @brief Record and exactly retire original-only leaked names after a real failed create. */
static void inspectFailedCreate(Fault selected)
{
    char name[128]{};
    void* pointer = nullptr;
    ncclShmHandle_t handle = nullptr;
    fault = selected;
    const auto code = ncclShmOpen(name, sizeof(name), 4096, &pointer, nullptr, 1, &handle);
    fault = Fault::None;
    require(code == ncclSystemError && !pointer && !handle, "failed creator published live outputs");
    if (!absent(name))
    {
        ++failed_names;
        requireIdentity(name, latest_created);
        require(unlink(name) == 0, "exact diagnostic cleanup failed");
    }
}
/** @brief Explicit creator intent means a new name; no preexisting file is writable capacity. */
static void explicitCreators()
{
    const char* base = scratchDirectory;
    require(base && *base, "fixture requires original process-owned scratch directory");
    char path[4096]{};
    require(std::snprintf(path, sizeof(path), "%s/sdk-explicit-XXXXXX", base) <
                static_cast<int>(sizeof(path)),
            "fixture path overflow");
    int fd = mkstemp(path);
    require(fd >= 0, "fixture name acquisition failed");
    require(close(fd) == 0 && unlink(path) == 0, "fixture fresh name release failed");
    void* memory = nullptr;
    ncclShmHandle_t owner = nullptr;
    require(ncclShmOpen(path, sizeof(path), 4096, &memory, nullptr, 1, &owner) == ncclSuccess,
            "explicit fresh creator failed");
    require(ncclShmClose(owner) == ncclSuccess && absent(path), "explicit fresh creator retirement failed");

    // This file is owned by the caller, not by an SDK producer with the same name.
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    require(fd >= 0, "fixture existing creator failed");
    const unsigned char marker = 0xa5;
    require(ftruncate(fd, 4096) == 0 && pwrite(fd, &marker, 1, 17) == 1, "fixture marker publication failed");
    struct stat original
    {
    };
    require(fstat(fd, &original) == 0, "fixture identity unavailable");
    owner = nullptr;
    memory = nullptr;
    const auto code = ncclShmOpen(path, sizeof(path), 4096, &memory, nullptr, 1, &owner);
    if (code == ncclSuccess)
    {
        ++replaced_existing_files;
        require(owner && memory, "original unexpectedly omitted live outputs");
        require(ncclShmClose(owner) == ncclSuccess, "original existing-file cleanup failed");
    }
    else
    {
        require(code == ncclSystemError && !owner && !memory, "existing creator did not reject precisely");
        requireIdentity(path, original);
        struct stat retained
        {
        };
        unsigned char observed = 0;
        require(fstat(fd, &retained) == 0 && retained.st_size == 4096 && pread(fd, &observed, 1, 17) == 1 &&
                    observed == marker,
                "rejected creator modified original file");
        require(unlink(path) == 0, "fixture original name cleanup failed");
    }
    require(close(fd) == 0, "fixture original descriptor retirement failed");

    // Never-opened creator failure owns no file name or allocated path metadata.
    require(std::snprintf(path, sizeof(path), "%s/missing-parent/sdk-file", base) <
                static_cast<int>(sizeof(path)),
            "missing parent path overflow");
    owner = nullptr;
    memory = nullptr;
    inspectPathAllocation = true;
    const auto failedOpen = ncclShmOpen(path, sizeof(path), 4096, &memory, nullptr, 1, &owner);
    inspectPathAllocation = false;
    require(failedOpen == ncclSystemError && !owner && !memory && absent(path),
            "missing-parent creator failure changed ownership");
    // The original loses this exact captured pointer; diagnostic cleanup owns
    // only that known allocation, without claiming production retirement.
    std::free(failedPathAllocation);
    failedPathAllocation = nullptr;
}
/** @brief Replay creator abandonment, faulted neighbors, pre-decrement attach failure and explicit names. */
static void replay()
{
    // Creator-only abandonment: published positive refcount still owns a name.
    char solo[128]{};
    void* host = nullptr;
    ncclShmHandle_t owner = nullptr;
    require(ncclShmOpen(solo, sizeof(solo), 4096, &host, nullptr, 1, &owner) == ncclSuccess,
            "solo create failed");
    require(!absent(solo), "solo name absent before close");
    require(ncclShmClose(owner) == ncclSuccess && absent(solo), "solo close did not unlink");

    // Another live segment must survive both early creator faults unchanged.
    char live[128]{};
    host = nullptr;
    owner = nullptr;
    require(ncclShmOpen(live, sizeof(live), 4096, &host, nullptr, 1, &owner) == ncclSuccess,
            "live create failed");
    struct stat original
    {
    };
    require(lstat(live, &original) == 0, "live identity missing");
    std::memset(host, 0x5a, 4096);
    for (const auto selected : {Fault::Extent, Fault::Mapping})
    {
        inspectFailedCreate(selected);
        requireIdentity(live, original);
        for (std::size_t i = 0; i < 4096; ++i)
            require(static_cast<unsigned char*>(host)[i] == 0x5a, "live neighbor data changed");
    }

    // Failed attachment owns neither the creator name nor its reference.
    fault = Fault::Mapping;
    void* peer = nullptr;
    ncclShmHandle_t attachment = nullptr;
    require(ncclShmOpen(live, sizeof(live), 4096, &peer, nullptr, -1, &attachment) == ncclSystemError &&
                !peer && !attachment,
            "failed attach published a mapping");
    fault = Fault::None;
    requireIdentity(live, original);
    require(ncclShmOpen(live, sizeof(live), 4096, &peer, nullptr, -1, &attachment) == ncclSuccess,
            "subsequent real attach failed");
    require(absent(live), "last successful attachment did not unlink");
    require(static_cast<unsigned char*>(peer)[17] == 0x5a, "attached bytes changed");
    require(ncclShmClose(attachment) == ncclSuccess && ncclShmClose(owner) == ncclSuccess,
            "mapped retirement failed");
    char multiple[128]{};
    void* primary = nullptr;
    ncclShmHandle_t primaryOwner = nullptr;
    require(ncclShmOpen(multiple, sizeof(multiple), 4096, &primary, nullptr, 2, &primaryOwner) == ncclSuccess,
            "two-recipient creator failed");
    static_cast<unsigned char*>(primary)[17] = 0x6b;
    void* first = nullptr;
    ncclShmHandle_t firstOwner = nullptr;
    require(ncclShmOpen(multiple, sizeof(multiple), 4096, &first, nullptr, -1, &firstOwner) == ncclSuccess &&
                !absent(multiple),
            "nonfinal attachment removed creator name");
    void* last = nullptr;
    ncclShmHandle_t lastOwner = nullptr;
    require(ncclShmOpen(multiple, sizeof(multiple), 4096, &last, nullptr, -1, &lastOwner) == ncclSuccess &&
                absent(multiple),
            "final attachment retained creator name");
    require(ncclShmClose(primaryOwner) == ncclSuccess && static_cast<unsigned char*>(first)[17] == 0x6b &&
                static_cast<unsigned char*>(last)[17] == 0x6b,
            "source-first close retired live attachment backing");
    require(ncclShmClose(firstOwner) == ncclSuccess && ncclShmClose(lastOwner) == ncclSuccess,
            "two-recipient mappings failed retirement");

    explicitCreators();
}
/** @brief Preserve valid fd0 ownership and reject caller-owned symlink targets. */
static void descriptorAndSymlinkReplays(bool candidate)
{
    char directory[] = "/tmp/qfn-shm-fd0-symlink-XXXXXX";
    require(mkdtemp(directory) != nullptr, "private directory creation failed");
    for (unsigned iteration = 0; iteration < 20; ++iteration)
    {
        for (const auto selected : {Fault::None, Fault::Extent, Fault::Mapping})
        {
            const std::string path = std::string(directory) + "/fd0-" + std::to_string(iteration) + "-" +
                                     std::to_string(static_cast<int>(selected));
            char name[512];
            std::strcpy(name, path.c_str());
            int saved = dup(STDIN_FILENO);
            if (saved < 0)
                require(errno == EBADF, "unexpected stdin duplicate failure");
            require(close(STDIN_FILENO) == 0 || errno == EBADF, "close original stdin failed");
            fault = selected;
            void* memory = nullptr;
            ncclShmHandle_t handle = nullptr;
            const auto result = ncclShmOpen(name, sizeof(name), 4096, &memory, nullptr, 1, &handle);
            fault = Fault::None;
            if (selected == Fault::None)
            {
                require(result == ncclSuccess && memory && handle, "fd0 creator failed");
                struct stat actual
                {
                };
                require(fstat(STDIN_FILENO, &actual) == 0 && actual.st_ino == latest_created.st_ino &&
                            actual.st_dev == latest_created.st_dev,
                        "SDK did not own valid fd0");
                require(ncclShmClose(handle) == ncclSuccess && absent(name),
                        "successful fd0 retirement failed");
            }
            else
            {
                require(result == ncclSystemError && !memory && !handle, "failed fd0 outputs invalid");
                require(absent(name) == candidate, "failed fd0 exact name disposition wrong");
                if (!candidate)
                {
                    ++failed_names;
                    requireIdentity(name, latest_created);
                    require(unlink(name) == 0, "exact fd0 diagnostic cleanup failed");
                }
            }
            require(fcntl(STDIN_FILENO, F_GETFD) == -1 && errno == EBADF, "SDK failed to close owned fd0");
            if (saved >= 0)
                require(dup2(saved, STDIN_FILENO) == STDIN_FILENO && close(saved) == 0,
                        "restore original stdin failed");
        }
        const std::string target = std::string(directory) + "/target-" + std::to_string(iteration);
        const std::string link = std::string(directory) + "/link-" + std::to_string(iteration);
        int fd = open(target.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        const unsigned char marker = 0xa5;
        require(fd >= 0 && ftruncate(fd, 4096) == 0 && pwrite(fd, &marker, 1, 17) == 1,
                "target creation failed");
        struct stat original
        {
        };
        require(fstat(fd, &original) == 0 && symlink(target.c_str(), link.c_str()) == 0,
                "symlink publication failed");
        struct stat original_link
        {
        };
        require(lstat(link.c_str(), &original_link) == 0, "symlink identity failed");
        char name[512];
        std::strcpy(name, link.c_str());
        void* memory = nullptr;
        ncclShmHandle_t handle = nullptr;
        const auto result = ncclShmOpen(name, sizeof(name), 4096, &memory, nullptr, 1, &handle);
        if (candidate)
        {
            require(result == ncclSystemError && !memory && !handle, "preexisting symlink was accepted");
            requireIdentity(name, original_link);
            require(unlink(name) == 0, "exact symlink cleanup failed");
        }
        else
        {
            require(result == ncclSuccess && memory && handle, "original symlink witness missing");
            require(ncclShmClose(handle) == ncclSuccess && absent(name),
                    "original symlink disposition wrong");
        }
        struct stat retained
        {
        };
        unsigned char observed = 0;
        require(fstat(fd, &retained) == 0 && retained.st_dev == original.st_dev &&
                    retained.st_ino == original.st_ino && pread(fd, &observed, 1, 17) == 1,
                "target identity lost");
        require(observed == (candidate ? marker : 0), "target clobber expectation wrong");
        if (!candidate)
            ++replaced_existing_files;
        require(close(fd) == 0 && unlink(target.c_str()) == 0, "exact target retirement failed");
    }
    require(rmdir(directory) == 0, "private files remain");
}
/** @brief Verify all 20 complete native/original observations before returning ordinary success. */
int main(int argc, char** argv)
{
    require(argc == 3 && (std::strcmp(argv[1], "original") == 0 || std::strcmp(argv[1], "candidate") == 0),
            "select exact original or candidate expectation");
    scratchDirectory = argv[2];
    for (int iteration = 0; iteration < 20; ++iteration)
        replay();
    descriptorAndSymlinkReplays(std::strcmp(argv[1], "candidate") == 0);
    const int expected = std::strcmp(argv[1], "original") == 0 ? 80 : 0;
    const unsigned expected_allocations = std::strcmp(argv[1], "original") == 0 ? 20 : 0;
    require(failedOpenPathAllocations == expected_allocations,
            "failed-open path allocation expectation failed");
    const int expected_replaced = std::strcmp(argv[1], "original") == 0 ? 40 : 0;
    std::printf("{\"schema\":1,\"replays\":20,\"fd0_cases\":60,\"symlink_cases\":20,\"multi_attach_cases\":"
                "20,\"case_executions\":260,"
                "\"failed_open_path_allocations\":%u,\"failed_creator_names_retained\":%d,"
                "\"expected_retained_names\":%d,\"live_neighbor_and_failed_attach_preserved\":true,"
                "\"preexisting_files_clobbered\":%d,\"all_exact_diagnostic_files_removed\":true}\n",
                failedOpenPathAllocations, failed_names, expected, replaced_existing_files);
    require(failed_names == expected, "partial creator cleanup expectation failed");
    require(replaced_existing_files == expected_replaced, "explicit creator ownership expectation failed");
}
