/**
 * @file MPISessionOwner.cpp
 * @brief Linux pidfd/control-channel ownership for MPI runtime artifacts.
 *
 * A clean, detached guardian owns launcher descendants using the existing OS
 * parent/subreaper relationship. It releases only its fresh private directories
 * after waitpid proves all children retired, including singleton MPI daemons.
 * Actual frontend entrypoints enter this ancestor before runtime initialization.
 * No PID-name matching, shared /tmp sweep, model state or accelerator API is used.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "utils/MPISessionOwner.h"
#include "utils/MPISessionEnvironment.h"
#include <dlfcn.h>
#include <array>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <optional>
#include <poll.h>
#include <spawn.h>
#include <span>
#include <stdexcept>
#include <system_error>
#include <string_view>
#include <sys/prctl.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>
extern char **environ;

namespace llaminar2
{
namespace
{
    using namespace mpi_session;
    constexpr int kControl = 3;
    constexpr int kParentHandle = 4;
    constexpr int kOwnerBorrow = 5;
    constexpr int kTemporaryBorrow = 6;
    constexpr int kSharedBorrow = 7;
    constexpr int kProtocolBorrow = 8;
    constexpr auto kRetirementGrace = std::chrono::seconds(30);
    constexpr auto kKilledGrace = std::chrono::seconds(5);
    /** @brief Bounded local control, never mutable model or MPI execution state. */
    enum class Kind : unsigned { Ready = 1, Signal, Done };
    /** @brief One same-host packet with bounded original directory identities. */
    struct Packet {
        Kind kind{};
        int value{};
        std::array<char, 4096> temporary{};
        std::array<char, 4096> shared{};
    };
    /** @brief Immutable startup loan; kernel peer credentials bind its original guardian. */
    struct RootIdentity {
        std::uint64_t device=0, inode=0, uid=0;
    };
    struct BorrowWitness {
        std::uint64_t magic=0x4c4d504953455331ull;
        std::uint32_t version=2;
        std::uint32_t bytes=sizeof(BorrowWitness);
        RootIdentity temporary,shared;
    };
    /** @brief Scope-bound original OS descriptor. */
    struct FD {
        int value = -1;
        explicit FD(int v = -1) noexcept : value(v) {}
        FD(FD &&o) noexcept : value(std::exchange(o.value, -1)) {}
        FD(const FD &) = delete;
        FD &operator=(const FD &) = delete;
        ~FD() { if (value >= 0) ::close(value); }
    };
    /** @brief Fail precisely; a failed ownership witness never authorizes deletion. */
    [[noreturn]] void fail(const std::string &what)
    { throw std::runtime_error("MPI session ownership: " + what + ": " + std::strerror(errno)); }
    /** @brief Send one immutable local ownership packet without SIGPIPE termination. */
    void sendPacket(int fd, const Packet &packet)
    {
        ssize_t result;
        do { result = ::send(fd, &packet, sizeof(packet), MSG_NOSIGNAL); } while (result < 0 && errno == EINTR);
        if (result != sizeof(packet)) fail("cannot publish guardian control");
    }
    /** @return False when the actual guardian already closed its terminal channel. */
    bool forwardSignal(int fd,int signal)
    {
        const Packet packet{.kind=Kind::Signal,.value=signal};
        ssize_t result;
        do { result=::send(fd,&packet,sizeof(packet),MSG_NOSIGNAL); } while(result<0 && errno==EINTR);
        if(result<0 && (errno==EPIPE || errno==ECONNRESET))return false;
        if(result!=sizeof(packet))fail("cannot forward lifecycle signal");
        return true;
    }
    /** @return False only for exact channel closure; reject partial protocol messages. */
    bool receivePacket(int fd, Packet &packet)
    {
        ssize_t result;
        do { result = ::recv(fd, &packet, sizeof(packet), MSG_TRUNC); } while (result < 0 && errno == EINTR);
        if (result == 0) return false;
        if (result != sizeof(packet)) fail("partial guardian control");
        return true;
    }
    /** @return Exact-process observation immune to PID reuse; unsupported kernels reject. */
    FD processHandle(pid_t pid)
    {
        int fd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
        if (fd < 0) fail("cannot acquire exact parent pidfd");
        return FD(fd);
    }
    /** @brief Mask only lifecycle signals and consume them through an exact descriptor. */
    struct Signals {
        sigset_t previous{}, selected{};
        struct sigaction previous_child{};
        FD descriptor;
        /** @brief Preserve waitability even when the invoking process ignored SIGCHLD. */
        Signals() {
            sigemptyset(&selected);
            for (int s : {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGCHLD}) sigaddset(&selected, s);
            struct sigaction waitable{}; waitable.sa_handler=SIG_DFL; sigemptyset(&waitable.sa_mask);
            if(::sigaction(SIGCHLD,&waitable,&previous_child)!=0)fail("cannot retain waitable child ownership");
            if (::sigprocmask(SIG_BLOCK, &selected, &previous) != 0) {
                ::sigaction(SIGCHLD,&previous_child,nullptr); fail("cannot mask lifecycle signals");
            }
            descriptor.value = ::signalfd(-1, &selected, SFD_CLOEXEC | SFD_NONBLOCK);
            if (descriptor.value < 0) {
                const int error=errno;
                ::sigaction(SIGCHLD,&previous_child,nullptr); ::sigprocmask(SIG_SETMASK,&previous,nullptr);
                errno=error; fail("cannot observe lifecycle signals");
            }
        }
        /** @brief Restore the original caller disposition only after joining the actual guardian. */
        ~Signals() {
            ::sigaction(SIGCHLD,&previous_child,nullptr);
            ::sigprocmask(SIG_SETMASK,&previous,nullptr);
        }
        /** @return Next pending lifecycle signal, or zero when drained. */
        int next() const {
            signalfd_siginfo info{};
            const auto n = ::read(descriptor.value, &info, sizeof(info));
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) return 0;
            if (n != sizeof(info)) fail("cannot consume lifecycle signal");
            return static_cast<int>(info.ssi_signo);
        }
    };
    /** @brief Remove an owned tree relative to its actual opened directory; never follow links. */
    void removeContents(int directory)
    {
        // dup() shares the original open-file-description cursor with every
        // inherited loan. A reader may have exhausted it without changing any
        // file. Open the same retained inode with an independent cursor.
        const int duplicate = ::openat(directory, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (duplicate < 0) fail("cannot enumerate owned directory");
        DIR *entries = ::fdopendir(duplicate);
        if (!entries) {
            const int error=errno;
            ::close(duplicate);
            errno=error;
            fail("cannot inspect owned directory");
        }
        try {
            for (;;) {
                errno = 0;
                auto *entry = ::readdir(entries);
                if (!entry) {
                    if (errno != 0) fail("cannot enumerate owned directory entries");
                    break;
                }
                const std::string name(entry->d_name);
                if (name == "." || name == "..") continue;
                struct stat info{};
                if (::fstatat(directory, name.c_str(), &info, AT_SYMLINK_NOFOLLOW) != 0) fail("cannot inspect owned artifact");
                const bool child_directory = S_ISDIR(info.st_mode);
                if (child_directory) {
                    FD child(::openat(directory, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
                    if (child.value < 0) fail("cannot open owned child directory");
                    removeContents(child.value);
                }
                if (::unlinkat(directory, name.c_str(), child_directory ? AT_REMOVEDIR : 0) != 0)
                    fail("cannot retire owned artifact");
            }
        } catch (...) { ::closedir(entries); throw; }
        if (::closedir(entries) != 0) fail("cannot close owned directory enumeration");
    }
    /** @brief One fresh private directory, authenticated again before cleanup. */
    struct Directory {
        std::filesystem::path path;
        FD parent, directory;
        struct stat identity{};
        enum class Lifetime { Unpublished, Published, Retired, Failed };
        Lifetime lifetime = Lifetime::Unpublished;
        explicit Directory(const std::filesystem::path &base)
        {
            path = std::filesystem::canonical(base) / ("llaminar-mpi-" + std::to_string(::geteuid()) + "-XXXXXX");
            std::string pattern = path.string();
            if (!::mkdtemp(pattern.data())) fail("cannot create private session directory");
            path = pattern;
            parent.value = ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            directory.value = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
            if (parent.value < 0 || directory.value < 0 || ::fstat(directory.value, &identity) != 0 ||
                identity.st_uid != ::geteuid() || (identity.st_mode & 0777) != 0700)
            {
                const int original_error = errno;
                // This root has never been published or populated by a child.
                ::rmdir(path.c_str());
                errno = original_error;
                fail("cannot authenticate private session directory");
            }
        }
        /** @brief Unpublished construction failure has no runtime consumers. */
        ~Directory() {
            if (lifetime == Lifetime::Unpublished && directory.value >= 0 && parent.value >= 0) {
                try { retire(); } catch (const std::exception &error) { std::fprintf(stderr,"%s\n",error.what()); }
            }
        }
        /** @brief Publish the original directory once, before any child can consume it. */
        void publish()
        {
            if (lifetime != Lifetime::Unpublished)
                throw std::logic_error("MPI session directory publication has no unpublished owner");
            lifetime = Lifetime::Published;
        }
        /** @brief Retire the original root once; failed physical cleanup stays retained. */
        void retire()
        {
            if (lifetime == Lifetime::Retired) return;
            if (lifetime == Lifetime::Failed)
                throw std::runtime_error("MPI session directory cleanup previously failed; backing retained: " + path.string());
            // Any exceptional exit retains the original obligation. Never
            // replay a partially completed filesystem operation from the top.
            lifetime = Lifetime::Failed;
            struct stat current{};
            if (::fstatat(parent.value, path.filename().c_str(), &current, AT_SYMLINK_NOFOLLOW) != 0 ||
                current.st_ino != identity.st_ino || current.st_dev != identity.st_dev ||
                current.st_uid != identity.st_uid || !S_ISDIR(current.st_mode))
                throw std::runtime_error("MPI session root identity changed; backing retained: " + path.string());
            removeContents(directory.value);
            if (::unlinkat(parent.value, path.filename().c_str(), AT_REMOVEDIR) != 0) fail("cannot remove retired session root");
            lifetime = Lifetime::Retired;
        }
    };
    /** @brief Joint temporary and real shared-memory roots with one actual guardian. */
    struct Roots {
        Directory temporary, shared;
        explicit Roots(const std::filesystem::path &base) : temporary(base), shared("/dev/shm") {
            for(const auto *directory : {&temporary,&shared}) {
                std::ofstream marker(directory->path / ".owner");
                marker << "guardian_pid=" << ::getpid() << "\nuid=" << ::geteuid()
                       << "\nsession_owner_protocol=1\ntemporary=" << temporary.path.string()
                       << "\nshared=" << shared.path.string() << '\n';
                marker.close();
                if (!marker) throw std::runtime_error("Cannot persist MPI session ownership witness");
            }

        }
        /** @brief Cleanup only after all actual consumers have retired. */
        void retire() { shared.retire(); temporary.retire(); }
        /** @return Immutable original directory identities for the owning process. */
        Packet ready() const {
            Packet p{.kind = Kind::Ready};
            const auto tmp = temporary.path.string(), shm = shared.path.string();
            if (tmp.size() >= p.temporary.size() || shm.size() >= p.shared.size())
                throw std::runtime_error("MPI session paths exceed bounded guardian control");
            std::memcpy(p.temporary.data(), tmp.c_str(), tmp.size()+1);
            std::memcpy(p.shared.data(), shm.c_str(), shm.size()+1);
            return p;
        }
    };
    /** @brief One canonical native executable classification shared by every owner boundary. */
    enum class NativeProgram { Application, Launcher, Daemon };
    /** @return Native role for supported public aliases and their canonical installed names. */
    NativeProgram nativeProgram(const std::filesystem::path &program)
    {
        const auto name=program.filename().string();
        for(const char *alias:{"mpirun","mpiexec","mpirun.openmpi","mpiexec.openmpi","orterun","prterun"})
            if(name==alias)return NativeProgram::Launcher;
        for(const char *alias:{"orted","prted"})if(name==alias)return NativeProgram::Daemon;
        return NativeProgram::Application;
    }
    /** @return Values projected from the same original physical roots for every location binding. */
    std::map<std::string,std::string> bindings(const Packet &p)
    {
        std::map<std::string,std::string> result;
        for(const auto &binding:kLocationBindings)
            result.emplace(binding.name,binding.root==RuntimeRoot::Shared?p.shared.data():p.temporary.data());
        return result;
    }
    /** @brief Reject native env-list routing instead of guessing its configurable grammar. */
    void requireSupportedExportEnvironment()
    {
        for(const auto &binding:kUnsupportedExportBindings)
            if(processEnvironmentValue(binding.key))throw std::invalid_argument("Owned MPI launch does not support inherited mca_base_env_list exports; use -x for non-location variables");
    }
    /** @brief Preserve native default selection; an explicit selector must include this required hook. */
    void requireNativeSelection(std::string_view selector)
    {
        if(selector.empty())return;
        const bool exclude=selector.front()=='^';
        if(exclude)selector.remove_prefix(1);
        bool found=false;
        for(std::size_t start=0;start<=selector.size();) {
            const auto end=selector.find(',',start);
            found|=selector.substr(start,end==std::string_view::npos?end:end-start)=="llaminar";
            if(end==std::string_view::npos)break;
            start=end+1;
        }
        if(found==exclude)throw std::invalid_argument("Owned MPI native environment component is not selected");
    }
    /** @brief Authenticate the helper-sibling native module before any runtime initialization. */
    std::filesystem::path nativeModuleDirectory()
    {
        const auto directory=std::filesystem::canonical("/proc/self/exe").parent_path()/kComponentDirectory;
        const auto module=directory/kComponentFilename;
        if(!std::filesystem::is_regular_file(module))throw std::runtime_error("Owned MPI native environment component is missing");
        void *loaded=::dlopen(module.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!loaded)throw std::runtime_error("Owned MPI native environment component ABI is incompatible");
        const auto query=reinterpret_cast<NativeContractQuery>(::dlsym(loaded,kContractSymbol));
        const NativeModuleContract expected;
        const auto *contract=query?query():nullptr;
        const bool valid=contract && contract->magic==expected.magic && contract->protocol==expected.protocol &&
            contract->openmpi_major==expected.openmpi_major && contract->openmpi_minor==expected.openmpi_minor &&
            contract->openmpi_release==expected.openmpi_release && contract->schizo_module_bytes==expected.schizo_module_bytes &&
            ::dlsym(loaded,"mca_schizo_llaminar_component");
        ::dlclose(loaded);
        if(!valid)throw std::runtime_error("Owned MPI native environment component ABI is incompatible");
        return directory;
    }
    /** @brief Configure original roots and mandatory native hook without replacing default personalities. */
    void publishEnvironment(const Packet &p)
    {
        requireSupportedExportEnvironment();
        if(const char *selector=processEnvironmentValue(EnvironmentKey::Selector))requireNativeSelection(selector);
        if(const char *disabled=processEnvironmentValue(EnvironmentKey::DynamicComponents);disabled && std::string_view(disabled)!="0")
            throw std::invalid_argument("Owned MPI native environment component is not selected: dynamic components disabled");
        const auto module=nativeModuleDirectory().string();
        const char *component_path=processEnvironmentValue(EnvironmentKey::ComponentPath);
        std::string path=component_path?component_path:"SYSTEM_DEFAULT:USER_DEFAULT";
        if(const char *old=processEnvironmentValue(EnvironmentKey::ModuleDirectory);old && path.starts_with(std::string(old)+":"))path.erase(0,std::strlen(old)+1);
        path=module+":"+path;
        for (const auto &[name,value]:bindings(p))
            if(::setenv(name.c_str(),value.c_str(),1)!=0)fail("cannot bind MPI runtime files");
        for(const char *name:kUnsetLocations)
            if(::unsetenv(name)!=0)fail("cannot clear incompatible MPI location alias");
        for(const auto &[name,value]:std::map<std::string,std::string>{{kRequiredEnv,kProtocol},{kModuleDirectoryEnv,module},
            {kComponentPathEnv,path},{kDynamicComponentsEnv,"0"}})
            if(::setenv(name.c_str(),value.c_str(),1)!=0)fail("cannot bind native MPI environment component");
        if(::unsetenv(kReadyEnv)!=0)fail("cannot retire previous native rank publication");
    }
    /** @return Whether this native control could bypass or forge the required fork hook. */
    bool nativeControl(std::string_view name)
    {
        return name=="mca_base_component_path" || name=="mca_base_component_disable_dlopen" ||
            name=="llaminar_session_protocol" || name=="llaminar_session_ready" || name=="llaminar_session_module_directory";
    }
    /** @return Whether one native location parameter belongs to the session owner. */
    bool locationParameter(std::string_view name)
    {
        const auto matches=[name](std::string_view environment) {
            for(const std::string_view prefix:{"OMPI_MCA_","PRTE_MCA_"})
                if(environment.starts_with(prefix) && environment.substr(prefix.size())==name)return true;
            return false;
        };
        for(const auto &binding:kLocationBindings)if(matches(binding.name))return true;
        for(const char *alias:kUnsetLocations)if(matches(alias))return true;
        return false;
    }
    /** @return Whether one exported binding could replace host-local runtime roots. */
    bool locationExport(std::string_view binding)
    {
        const auto name=binding.substr(0,binding.find('='));
        if(name=="TMPDIR" || name=="TMP" || name=="TEMP")return true;
        for(auto prefix:{"OMPI_MCA_","PRTE_MCA_"}) if(name.starts_with(prefix))
            return locationParameter(name.substr(std::strlen(prefix))) || nativeControl(name.substr(std::strlen(prefix))) || name.substr(std::strlen(prefix))=="schizo";
        return false;
    }
    /** @return Whether this is the native MCA option, whose key/value arity MPI defines. */
    bool mcaOption(std::string_view word)
    { return word=="--mca" || word=="-mca" || word=="--gmca" || word=="-gmca"; }
    /**
     * @brief Validate only a native MPI option prefix with an explicit application boundary.
     * @param command Exact mpirun/mpiexec argv; ordinary frontend argv is uninterpreted.
     *
     * Requiring MPI's own -- delimiter avoids a second parser for every MPI
     * option's arity, and keeps application arguments out of ownership policy.
     */
    void requireOwnedLocations(std::span<const std::string> command)
    {
        if(nativeProgram(command.front())!=NativeProgram::Launcher)return;
        for(std::size_t i=1;i<command.size();++i) {
            const auto &word=command[i];
            if(word=="--") {
                if(i+1==command.size())break;
                return; // Application argv is never inspected as native MPI options.
            }
            if(mcaOption(word)) {
                if(i+2>=command.size())throw std::invalid_argument("Incomplete native MPI MCA argument before application delimiter");
                const auto &key=command[i+1];
                if(key=="mca_base_env_list" || key=="mca_base_env_list_internal")
                    throw std::invalid_argument("Owned MPI launch does not support mca_base_env_list exports; use -x for non-location variables");
                if(key=="schizo")requireNativeSelection(command[i+2]);
                if(nativeControl(key))throw std::invalid_argument("Owned MPI native environment component controls belong to the session owner");
                if(locationParameter(key))
                    throw std::invalid_argument("MPI runtime location MCA overrides belong to the session owner");
                i+=2;continue; // An opaque MCA value may itself be -- or another option.
            }
            if(word=="--tmpdir" || word.starts_with("--tmpdir="))
                throw std::invalid_argument("MPI --tmpdir conflicts with the session owner's private root");
            if(word=="-x") {
                if(++i>=command.size())throw std::invalid_argument("Incomplete native MPI export before application delimiter");
                if(locationExport(command[i]))
                    throw std::invalid_argument("MPI -x cannot export launcher-local runtime paths over a host-local session owner");
                continue;
            }
            if(word.starts_with("-x") && word.size()>2 && locationExport(std::string_view(word).substr(2)))
                throw std::invalid_argument("MPI -x cannot override owned runtime file locations");
        }
        throw std::invalid_argument("Owned native MPI launch requires -- before the application executable");
    }
    /** @return Native daemon argv normalized only for owned file locations and foreground lifetime. */
    std::vector<std::string> localDaemon(std::span<const std::string> command)
    {
        if(command.empty() || std::filesystem::path(command.front()).filename()!="orted")
            throw std::invalid_argument("MPI daemon ownership requires the selected OpenMPI orted program");
        std::vector<std::string> result{command.front()};
        for(std::size_t i=1;i<command.size();++i) {
            const auto &word=command[i];
            if(word=="--daemonize")continue;
            if(mcaOption(word)) {
                if(i+2>=command.size())throw std::invalid_argument("Incomplete native daemon MCA argument");
                if(command[i+1]=="schizo")requireNativeSelection(command[i+2]);
                if(!locationParameter(command[i+1]) && !nativeControl(command[i+1])) {
                    result.push_back(word);result.push_back(command[i+1]);result.push_back(command[i+2]);
                }
                i+=2;continue;
            }
            if(word=="--tmpdir") {
                if(++i>=command.size())throw std::invalid_argument("Incomplete native daemon tmpdir override");
                continue;
            }
            if(word.starts_with("--tmpdir="))continue;
            result.push_back(word);
        }
        return result;
    }
    /** @return POSIX-compatible original child status. */
    int statusCode(int status)
    { return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status); }
    /** @brief Original kernel-backed loans issued together by this guardian. */
    struct ProcessBorrow {
        int guardian = -1;
        int temporary = -1;
        int shared = -1;
        int witness = -1;
    };
    /** @brief Spawn without running C++ or backend code in a forked multi-threaded child. */
    pid_t spawn(const std::vector<std::string> &arguments, int control, bool detached, const sigset_t &mask, int parent_handle = -1, const ProcessBorrow &borrow = {}, const std::filesystem::path &executable = {})
    {
        std::vector<char *> argv;
        for (const auto &arg : arguments) argv.push_back(const_cast<char *>(arg.c_str()));
        argv.push_back(nullptr);
        posix_spawnattr_t attributes;
        posix_spawn_file_actions_t actions;
        const auto require = [](int error) {
            if (error != 0) { errno=error; fail("cannot prepare exact child spawn"); }
        };
        require(::posix_spawnattr_init(&attributes));
        const int initialized=::posix_spawn_file_actions_init(&actions);
        if(initialized!=0) { ::posix_spawnattr_destroy(&attributes); require(initialized); }
        pid_t pid=-1;
        int result=0;
        try {
            short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF;
            if (detached) flags |= POSIX_SPAWN_SETSID;
            else flags |= POSIX_SPAWN_SETPGROUP;
            require(::posix_spawnattr_setflags(&attributes, flags));
            require(::posix_spawnattr_setsigmask(&attributes, &mask));
            sigset_t defaults; sigemptyset(&defaults); sigaddset(&defaults,SIGCHLD);
            require(::posix_spawnattr_setsigdefault(&attributes,&defaults));
            if (!detached) require(::posix_spawnattr_setpgroup(&attributes, 0));
            if (control >= 0) require(::posix_spawn_file_actions_adddup2(&actions, control, kControl));
            else require(::posix_spawn_file_actions_addclose(&actions, kControl));
            if (parent_handle >= 0) require(::posix_spawn_file_actions_adddup2(&actions, parent_handle, kParentHandle));
            else require(::posix_spawn_file_actions_addclose(&actions, kParentHandle));
            for(const auto &[original,target] : {std::pair{borrow.guardian,kOwnerBorrow},
                    std::pair{borrow.temporary,kTemporaryBorrow},std::pair{borrow.shared,kSharedBorrow},
                    std::pair{borrow.witness,kProtocolBorrow}}) {
                if(original>=0)require(::posix_spawn_file_actions_adddup2(&actions,original,target));
                else require(::posix_spawn_file_actions_addclose(&actions,target));
            }
            if(executable.empty())result=::posix_spawnp(&pid,argv[0],&actions,&attributes,argv.data(),environ);
            else result=::posix_spawn(&pid,executable.c_str(),&actions,&attributes,argv.data(),environ);
        } catch (...) {
            ::posix_spawn_file_actions_destroy(&actions); ::posix_spawnattr_destroy(&attributes);
            throw;
        }
        ::posix_spawn_file_actions_destroy(&actions); ::posix_spawnattr_destroy(&attributes);
        require(result);
        return pid;
    }
    /** @brief Signal only actual child identities, never a reused numeric process group. */
    void signalChildren(int signal, std::map<pid_t,int> &sent)
    {
        std::ifstream children("/proc/self/task/" + std::to_string(::getpid()) + "/children");
        pid_t child;
        while (children >> child) {
            if (sent.contains(child) && sent.at(child) == signal) continue;
            const int handle = static_cast<int>(::syscall(SYS_pidfd_open, child, 0));
            if (handle < 0) { if (errno == ESRCH) continue; fail("cannot acquire exact child identity"); }
            FD exact(handle);
            siginfo_t state{};
            if (::waitid(P_PIDFD, handle, &state, WEXITED|WNOHANG|WNOWAIT) != 0) {
                if (errno == ECHILD) continue;
                fail("cannot authenticate guardian child relationship");
            }
            if (::syscall(SYS_pidfd_send_signal, handle, signal, nullptr, 0) != 0 && errno != ESRCH)
                fail("cannot signal exact guardian child");
            sent[child] = signal;
        }
        if (!children.eof()) throw std::runtime_error("Cannot inspect exact guardian-owned children");
    }
    /** @brief Wait/reap the actual owned subtree; never delete while a descendant remains. */
    int runCommand(const std::vector<std::string> &command, int control, int parent_pidfd, const Roots &roots, const std::filesystem::path &executable)
    {
        if (::prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) fail("cannot retain daemon child ownership");
        const auto paths=roots.ready();
        publishEnvironment(paths);
        Signals signals;
        auto owner = processHandle(::getpid());
        FD borrowed(::fcntl(owner.value,F_DUPFD_CLOEXEC,64));
        if (borrowed.value < 0) fail("cannot retain original session-owner borrow");
        FD temporary(::fcntl(roots.temporary.directory.value,F_DUPFD_CLOEXEC,64));
        FD shared(::fcntl(roots.shared.directory.value,F_DUPFD_CLOEXEC,64));
        if(temporary.value<0 || shared.value<0)fail("cannot retain original runtime-directory loans");
        int protocol[2];
        if(::socketpair(AF_UNIX,SOCK_SEQPACKET|SOCK_CLOEXEC,0,protocol)!=0)fail("cannot create immutable owner loan");
        FD lender(protocol[0]), reader(protocol[1]);
        BorrowWitness witness;
        witness.temporary={roots.temporary.identity.st_dev,roots.temporary.identity.st_ino,roots.temporary.identity.st_uid};
        witness.shared={roots.shared.identity.st_dev,roots.shared.identity.st_ino,roots.shared.identity.st_uid};
        if(::send(lender.value,&witness,sizeof(witness),MSG_NOSIGNAL)!=sizeof(witness))fail("cannot publish immutable owner loan");
        FD inherited_witness(::fcntl(reader.value,F_DUPFD_CLOEXEC,64));
        if(inherited_witness.value<0)fail("cannot retain immutable owner loan");
        const bool mpi=nativeProgram(executable.empty()?std::filesystem::path(command.front()):executable)!=NativeProgram::Application;
        const pid_t launched=spawn(command,-1,false,signals.previous,-1,
            mpi?ProcessBorrow{}:ProcessBorrow{borrowed.value,temporary.value,shared.value,inherited_witness.value},executable);
        ::close(reader.value);reader.value=-1;
        ::close(inherited_witness.value);inherited_witness.value=-1;
        int command_status = 1, requested_signal = 0;
        bool command_done = false, parent_alive = true;
        std::map<pid_t,int> sent;
        std::optional<std::chrono::steady_clock::time_point> stopping, killed;
        for (;;) {
            int status = 0;
            pid_t child;
            while ((child = ::waitpid(-1, &status, WNOHANG)) > 0) {
                sent.erase(child);
                if (!command_done && child == launched) { command_done = true; command_status = statusCode(status); }
            }
            if (child < 0 && errno == ECHILD) return command_status;
            if (child < 0 && errno != EINTR) fail("cannot certify child process retirement");
            if (command_done && !stopping) stopping = std::chrono::steady_clock::now();
            pollfd events[] = {{parent_alive ? control : -1, POLLIN|POLLHUP, 0}, {parent_alive ? parent_pidfd : -1, POLLIN, 0}, {signals.descriptor.value, POLLIN, 0}};
            if (::poll(events, 3, 50) < 0 && errno != EINTR) fail("cannot await owned launcher");
            if (events[0].revents && parent_alive) {
                Packet packet;
                if (!receivePacket(control, packet)) parent_alive = false;
                else if (packet.kind == Kind::Signal &&
                         (packet.value == SIGTERM || packet.value == SIGINT || packet.value == SIGHUP || packet.value == SIGQUIT)) {
                    if (!requested_signal) requested_signal = packet.value;
                    signalChildren(packet.value, sent);
                    if (!stopping) stopping = std::chrono::steady_clock::now();
                } else throw std::runtime_error("Unexpected launcher guardian control");
            }
            if (events[1].revents) parent_alive = false;
            for (int signal; (signal = signals.next()) != 0;) {
                if (signal != SIGCHLD) {
                    if (!requested_signal) requested_signal = signal;
                    signalChildren(signal, sent);
                    if (!stopping) stopping = std::chrono::steady_clock::now();
                }
            }
            if (!parent_alive && !stopping) { stopping = std::chrono::steady_clock::now(); signalChildren(SIGTERM, sent); }
            const auto now = std::chrono::steady_clock::now();
            if (stopping && !killed) {
                // Adopted daemon children can appear after the original child exits.
                signalChildren(requested_signal ? requested_signal : SIGTERM, sent);
                if (now-*stopping >= kRetirementGrace) { signalChildren(SIGKILL, sent); killed = now; }
            }
            if (killed) {
                // Killing a launcher reparents its surviving descendants to
                // this subreaper. Each newly adopted exact child needs the
                // same terminal signal before retirement can be certified.
                signalChildren(SIGKILL, sent);
                if (now-*killed >= kKilledGrace)
                    throw std::runtime_error("MPI children did not retire after SIGKILL; owned backing retained");
            }
        }
    }
    /** @brief Execute one clean independent guardian, shared by launcher and singleton modes. */
    int guardian(pid_t parent, int control, const std::filesystem::path &base, const std::vector<std::string> &command, const std::filesystem::path &executable)
    {
        // This descriptor was acquired by the original parent before spawn.
        // Reopening a numeric parent PID here could attach to a reused PID.
        FD exact_parent(kParentHandle);
        ucred peer{}; socklen_t peer_bytes = sizeof(peer);
        if (::getsockopt(control, SOL_SOCKET, SO_PEERCRED, &peer, &peer_bytes) != 0 ||
            peer.pid != parent || peer.uid != ::geteuid())
            throw std::runtime_error("MPI guardian lost its original control peer");
        Roots roots(base);
        const auto paths = roots.ready();
        bool published = false;
        try {
            sendPacket(control, paths);
            roots.temporary.publish(); roots.shared.publish();
            published = true;
            const int result = runCommand(command, control, exact_parent.value, roots, executable);
            roots.retire();
            // A killed owner has closed the channel; cleanup still succeeds independently.
            Packet done{.kind=Kind::Done, .value=result};
            const auto sent = ::send(control, &done, sizeof(done), MSG_NOSIGNAL);
            if (sent < 0 && errno != EPIPE && errno != ECONNRESET) fail("cannot publish retired session");
            return result;
        } catch (const std::exception &error) {
            // Preserve the first error. Cleanup of a later root must not hide
            // it by attempting to rediscover a root already physically retired.
            std::fprintf(stderr,"%s\n",error.what());
            int status = 0;
            if (!published || (!command.empty() && ::waitpid(-1, &status, WNOHANG) < 0 && errno == ECHILD)) {
                try { roots.retire(); }
                catch (const std::exception &cleanup_error) {
                    std::fprintf(stderr,"MPI session cleanup remains incomplete: %s\n",cleanup_error.what());
                }
                return 125;
            }
            std::fprintf(stderr, "MPI backing retained at %s and %s\n",
                roots.temporary.path.c_str(), roots.shared.path.c_str());
            return 125;
        }
    }
    /** @brief One exact proxy/guardian control pair, before any MPI creation. */
    struct Guardian {
        FD control;
        pid_t pid = -1;
        Packet paths;
    };
    /** @return An independently retained guardian, with authenticated original directory paths. */
    Guardian beginGuardian(const std::filesystem::path &helper, const std::filesystem::path &base,
                           std::span<const std::string> command, const std::filesystem::path &executable)
    {
        int sockets[2];
        if (::socketpair(AF_UNIX, SOCK_SEQPACKET|SOCK_CLOEXEC, 0, sockets) != 0) fail("cannot create guardian channel");
        FD owner(sockets[0]), child(sockets[1]);
        std::vector<std::string> args{helper.string(), "--guardian-parent", std::to_string(::getpid()),
            "--control-fd", std::to_string(kControl)};
        if(!executable.empty()) { args.push_back("--owned-executable");args.push_back(executable.string()); }
        args.insert(args.end(),{"--session-base",base.string(),"--"});
        args.insert(args.end(), command.begin(), command.end());
        sigset_t empty; sigemptyset(&empty);
        auto exact_parent = processHandle(::getpid());
        FD inherited_control(::fcntl(child.value,F_DUPFD_CLOEXEC,64));
        FD inherited_parent(::fcntl(exact_parent.value,F_DUPFD_CLOEXEC,64));
        if (inherited_control.value < 0 || inherited_parent.value < 0) fail("cannot retain guardian handoff descriptors");
        const pid_t pid = spawn(args, inherited_control.value, true, empty, inherited_parent.value);
        ::close(child.value); child.value = -1;
        ::close(inherited_control.value); inherited_control.value = -1;
        Packet packet;
        if (!receivePacket(owner.value, packet) || packet.kind != Kind::Ready ||
            packet.temporary.back() != '\0' || packet.shared.back() != '\0') {
            int status; while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error("MPI guardian failed before owned backing publication");
        }
        return {std::move(owner), pid, packet};
    }
    /** @return Original guardian child status after exact retirement. */
    int join(pid_t pid)
    {
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0) if (errno != EINTR) fail("cannot join original MPI guardian");
        return statusCode(status);
    }
    /** @return Launcher-local requested base; remote daemon commands explicitly choose /tmp. */
    std::filesystem::path sessionBase()
    { const char *base = processEnvironmentValue(EnvironmentKey::TemporaryDirectory); return base && *base ? base : "/tmp"; }
}

    std::filesystem::path MPISessionOwner::launcherExecutable()
    {
        std::array<char,4096> path{};
        const auto length = ::readlink("/proc/self/exe",path.data(),path.size()-1);
        if (length <= 0 || length == path.size()-1) fail("cannot resolve actual application executable");
        return std::filesystem::path(std::string(path.data(),length)).parent_path()/"llaminar_mpi_session";
    }
    namespace
    {
        /** @return Whether the MPI launcher, rather than an app rank, owns job backing. */
        bool externalMPI()
        {
            for (const auto &binding : kExternalBindings)
                if (processEnvironmentValue(binding.key)) return true;
            return false;
        }
        /** @brief A present original loan with incomplete evidence cannot become a fresh owner. */
        class BrokenBorrow final : public std::logic_error {
        public:
            explicit BrokenBorrow(const std::string &reason) : std::logic_error("Broken MPI session borrow: "+reason) {}
        };
        /** @brief Match the original opened directory, not a marker PID or guessed path. */
        void requireDirectory(int original,const char *path,const RootIdentity &expected)
        {
            if(!path || !*path)throw BrokenBorrow("missing original runtime location");
            struct stat loan{},current{};
            FD actual(::open(path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC));
            if(::fstat(original,&loan)!=0 || actual.value<0 || ::fstat(actual.value,&current)!=0 ||
                !S_ISDIR(loan.st_mode) || loan.st_uid!=::geteuid() || (loan.st_mode&0777)!=0700 ||
                loan.st_dev!=expected.device || loan.st_ino!=expected.inode || loan.st_uid!=expected.uid ||
                current.st_dev!=loan.st_dev || current.st_ino!=loan.st_ino)
                throw BrokenBorrow("runtime location lost its original directory identity");
        }
        /** @return Whether an actual inherited directory descriptor identifies this source-owned loan. */
        bool directoryLoan(int descriptor)
        {
            struct stat directory{};
            if(::fstat(descriptor,&directory)!=0 || !S_ISDIR(directory.st_mode) ||
               directory.st_uid!=::geteuid() || (directory.st_mode&0777)!=0700)return false;
            FD marker(::openat(descriptor,".owner",O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK));
            struct stat regular{};
            if(marker.value<0 || ::fstat(marker.value,&regular)!=0 || !S_ISREG(regular.st_mode))return false;
            std::array<char,4096> record{};
            const auto bytes=::read(marker.value,record.data(),record.size()-1);
            return bytes>0 && std::string_view(record.data(),bytes).find("\nsession_owner_protocol=1\n")!=std::string_view::npos;
        }
        /** @return Source-owned immutable witness; incidental inherited descriptors are uninterpreted. */
        struct OriginalBorrow { pid_t guardian; BorrowWitness witness; };
        std::optional<OriginalBorrow> protocolOwner()
        {
            int type=0; socklen_t type_bytes=sizeof(type);
            if(::getsockopt(kProtocolBorrow,SOL_SOCKET,SO_TYPE,&type,&type_bytes)!=0 || type!=SOCK_SEQPACKET)return std::nullopt;
            BorrowWitness observed{};
            const auto bytes=::recv(kProtocolBorrow,&observed,sizeof(observed),MSG_PEEK|MSG_DONTWAIT|MSG_TRUNC);
            const BorrowWitness expected;
            if(bytes<static_cast<ssize_t>(sizeof(observed.magic)) || observed.magic!=expected.magic)return std::nullopt;
            if(bytes!=sizeof(observed) || observed.version!=expected.version || observed.bytes!=expected.bytes)
                throw BrokenBorrow("incompatible immutable owner protocol");
            ucred creator{};socklen_t credential_bytes=sizeof(creator);
            if(::getsockopt(kProtocolBorrow,SOL_SOCKET,SO_PEERCRED,&creator,&credential_bytes)!=0 ||
               creator.uid!=::geteuid() || creator.pid<=0)throw BrokenBorrow("foreign immutable loan creator");
            pollfd live{kProtocolBorrow,POLLRDHUP,0};
            if(::poll(&live,1,0)!=0)throw BrokenBorrow("original immutable loan peer retired");
            return OriginalBorrow{creator.pid,observed};
        }
        /** @brief Exact opened native shared root authenticated against its daemon publication. */
        struct NativeSharedRoot final { FD directory;std::string path;RootIdentity original; };
        /** @return One original opened shared directory, never a post-authentication pathname adoption. */
        std::optional<NativeSharedRoot> authenticateNativeRankRoots()
        {
            const char *required=processEnvironmentValue(EnvironmentKey::RequiredProtocol);
            if(!required)return std::nullopt; // Explicit external unowned MPI keeps its existing semantics.
            const char *ready=processEnvironmentValue(EnvironmentKey::ReadyPublication);
            const char *temporary=processEnvironmentValue(EnvironmentKey::TemporaryDirectory);
            const char *shared=processEnvironmentValue(EnvironmentKey::VaderBackingDirectory);
            if(std::string_view(required)!=kProtocol || !ready || !temporary || !shared)
                throw std::logic_error("Owned MPI rank has no native environment publication");
            const std::string expected_publication=ready,temporary_path=temporary,shared_path=shared;
            FD temporary_fd(::open(temporary_path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC));
            FD shared_fd(::open(shared_path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC));
            struct stat t{},h{};
            if(temporary_fd.value<0 || shared_fd.value<0 || ::fstat(temporary_fd.value,&t)!=0 || ::fstat(shared_fd.value,&h)!=0)
                throw std::logic_error("Owned MPI rank lost its original opened root publication");
            const std::string actual=std::string(kProtocol)+":"+std::to_string(t.st_dev)+":"+std::to_string(t.st_ino)+":"+
                std::to_string(h.st_dev)+":"+std::to_string(h.st_ino);
            if(actual!=expected_publication)
                throw std::logic_error("Owned MPI rank opened roots do not match native publication");
            const RootIdentity temporary_identity{t.st_dev,t.st_ino,t.st_uid},shared_identity{h.st_dev,h.st_ino,h.st_uid};
            for(const auto &binding:kLocationBindings) {
                const bool is_shared=binding.root==RuntimeRoot::Shared;
                requireDirectory(is_shared?shared_fd.value:temporary_fd.value,processEnvironmentValue(binding.key),
                    is_shared?shared_identity:temporary_identity);
            }
            for(const auto &binding:kUnsetBindings)if(processEnvironmentValue(binding.key))
                throw std::logic_error("Owned MPI rank has conflicting native location aliases");
            return NativeSharedRoot{std::move(shared_fd),shared_path,shared_identity};
        }
        /** @brief Reuse the same exact-opened-root authentication before native MPI initialization. */
        void requireNativeRankPublication() { (void)authenticateNativeRankRoots(); }
        /** @return Present valid original loan; all-absent fresh main entries can create a distinct scope. */
        std::optional<OriginalBorrow> managedProcess()
        {
            const auto original=protocolOwner();
            if(!original) {
                if(directoryLoan(kTemporaryBorrow) || directoryLoan(kSharedBorrow)) {
                    // Native MPI deliberately retires the startup tuple before
                    // exec. Its authenticated fork publication makes these FD
                    // numbers ordinary application slots again. A recognized
                    // immutable managed witness above always takes precedence.
                    if(externalMPI() && processEnvironmentValue(EnvironmentKey::ReadyPublication) &&
                       authenticateNativeRankRoots())return std::nullopt;
                    throw BrokenBorrow("original directory loan lost its immutable protocol witness");
                }
                return std::nullopt;
            }
            std::array<char,64> target{};
            const std::string descriptor="/proc/self/fd/"+std::to_string(kOwnerBorrow);
            const auto length=::readlink(descriptor.c_str(),target.data(),target.size()-1);
            if(length<=0 || std::string_view(target.data(),length)!="anon_inode:[pidfd]")
                throw BrokenBorrow("immutable loan lost its original guardian pidfd");
            std::ifstream identity("/proc/self/fdinfo/" + std::to_string(kOwnerBorrow));
            std::string line; pid_t owner = -1;
            while(std::getline(identity,line)) if(line.starts_with("Pid:")) owner=std::stoi(line.substr(4));
            pollfd alive{kOwnerBorrow,POLLIN,0};
            if(owner!=original->guardian || ::poll(&alive,1,0)!=0)throw BrokenBorrow("original guardian has retired or changed identity");
            const char *temporary=processEnvironmentValue(EnvironmentKey::TemporaryDirectory), *shared=processEnvironmentValue(EnvironmentKey::VaderBackingDirectory);
            requireDirectory(kTemporaryBorrow,temporary,original->witness.temporary);
            requireDirectory(kSharedBorrow,shared,original->witness.shared);
            Packet paths{};
            if(std::strlen(temporary)>=paths.temporary.size() || std::strlen(shared)>=paths.shared.size())
                throw BrokenBorrow("runtime paths exceed original protocol extent");
            std::strcpy(paths.temporary.data(),temporary);std::strcpy(paths.shared.data(),shared);
            for(const auto &binding:kLocationBindings) {
                const bool shared_location=binding.root==RuntimeRoot::Shared;
                requireDirectory(shared_location?kSharedBorrow:kTemporaryBorrow,processEnvironmentValue(binding.key),
                    shared_location?original->witness.shared:original->witness.temporary);
            }
            for(const auto &binding:kUnsetBindings)
                if(processEnvironmentValue(binding.key))throw BrokenBorrow("incompatible MPI location alias was republished");
            FD marker(::openat(kTemporaryBorrow,".owner",O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK));
            struct stat regular{};
            if(marker.value<0 || ::fstat(marker.value,&regular)!=0 || !S_ISREG(regular.st_mode))
                throw BrokenBorrow("original owner marker is missing");
            std::array<char,4096> record{};
            const auto bytes=::read(marker.value,record.data(),record.size()-1);
            if(bytes<=0 || !std::string_view(record.data(),bytes).starts_with("guardian_pid="))
                throw BrokenBorrow("original owner marker is malformed");
            const pid_t expected=std::stoi(std::string(record.data()+13));
            if(owner!=expected)throw BrokenBorrow("foreign guardian identity");
            return original;
        }
    }
    /** @brief Retained original directory; no child creates another process/file owner. */
    struct MPISessionOwner::SharedDirectory::State final {
        int descriptor=-1;
        SharedIdentity original;
        std::string path;
        ~State() { if(descriptor>=0)::close(descriptor); }
        /** @brief Reject path replacement without following a new directory or symlink. */
        void requireOriginal() const
        {
            struct stat loan{},current{};
            FD named(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC));
            if(::fstat(descriptor,&loan)!=0 || named.value<0 || ::fstat(named.value,&current)!=0 ||
               !S_ISDIR(loan.st_mode) || loan.st_uid!=::geteuid() || (loan.st_mode&0777)!=0700 ||
               SharedIdentity{loan.st_dev,loan.st_ino,loan.st_uid}!=original ||
               SharedIdentity{current.st_dev,current.st_ino,current.st_uid}!=original)
                throw std::runtime_error("Owned shared directory lost its original inode");
        }
        /** @brief Restrict names to one non-reserved component before every relative syscall. */
        void requireName(const std::string &name) const
        {
            requireOriginal();
            const long maximum=::fpathconf(descriptor,_PC_NAME_MAX);
            if(maximum<1)throw std::runtime_error("Cannot inspect owned shared-directory name extent");
            if(name.empty() || name=="." || name==".." || name==".owner" ||
               name.find('/')!=std::string::npos || name.find('\0')!=std::string::npos ||
               name.size()>static_cast<std::size_t>(maximum))
                throw std::invalid_argument("Owned shared backing requires one exact non-reserved basename");
        }
    };
    /** @brief Original file descriptor/name; mapped payload ownership remains with its caller. */
    struct MPISessionOwner::SharedFile::State final {
        std::shared_ptr<const SharedDirectory::State> directory;
        std::string basename,path;
        SharedIdentity original;
        int descriptor=-1;
        bool creator=false,unlinked=false;
        ~State() { if(descriptor>=0)::close(descriptor); }
    };
    MPISessionOwner::SharedIdentity MPISessionOwner::SharedDirectory::identity() const
    { if(!state_)throw std::logic_error("Moved owned shared directory has no identity");return state_->original; }
    const std::string &MPISessionOwner::SharedDirectory::path() const
    { if(!state_)throw std::logic_error("Moved owned shared directory has no path");return state_->path; }
    MPISessionOwner::SharedFile MPISessionOwner::SharedDirectory::createExclusive(const std::string &basename) const
    {
        if(!state_)throw std::logic_error("Moved owned shared directory cannot open backing");
        state_->requireName(basename);
        auto file=std::make_unique<SharedFile::State>();
        file->directory=state_;file->basename=basename;file->path=state_->path+"/"+basename;file->creator=true;
        file->descriptor=::openat(state_->descriptor,basename.c_str(),O_CREAT|O_EXCL|O_RDWR|O_NOFOLLOW|O_CLOEXEC,0600);
        if(file->descriptor<0)throw std::system_error(errno,std::generic_category(),"cannot create exact owned shared backing");
        struct stat actual{};
        if(::fstat(file->descriptor,&actual)!=0 || !S_ISREG(actual.st_mode) || actual.st_uid!=::geteuid() ||
           (actual.st_mode&0777)!=0600 || actual.st_dev!=state_->original.device)
            throw std::runtime_error("Owned shared backing creation has invalid physical identity");
        file->original={actual.st_dev,actual.st_ino,actual.st_uid};
        return SharedFile(std::move(file));
    }
    MPISessionOwner::SharedFile MPISessionOwner::SharedDirectory::openExisting(const std::string &basename, SharedIdentity creator,
        std::uint64_t expected_bytes) const
    {
        if(!state_)throw std::logic_error("Moved owned shared directory cannot open backing");
        state_->requireName(basename);
        auto file=std::make_unique<SharedFile::State>();
        file->directory=state_;file->basename=basename;file->path=state_->path+"/"+basename;
        file->descriptor=::openat(state_->descriptor,basename.c_str(),O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if(file->descriptor<0)throw std::system_error(errno,std::generic_category(),"cannot open exact owned shared backing");
        struct stat actual{};
        if(::fstat(file->descriptor,&actual)!=0 || !S_ISREG(actual.st_mode) || actual.st_uid!=::geteuid() ||
           (actual.st_mode&0777)!=0600 || actual.st_dev!=state_->original.device)
            throw std::runtime_error("Owned shared backing attachment has invalid physical identity");
        file->original={actual.st_dev,actual.st_ino,actual.st_uid};
        if(file->original!=creator || actual.st_size<0 || static_cast<std::uint64_t>(actual.st_size)!=expected_bytes)
            throw std::runtime_error("Owned shared backing does not match its creator-published identity/extent");
        return SharedFile(std::move(file));
    }
    MPISessionOwner::SharedFile::SharedFile() noexcept=default;
    MPISessionOwner::SharedFile::SharedFile(std::unique_ptr<State> state) : state_(std::move(state)) {}
    MPISessionOwner::SharedFile::~SharedFile() { reset(); }
    MPISessionOwner::SharedFile::SharedFile(SharedFile &&) noexcept=default;
    MPISessionOwner::SharedFile &MPISessionOwner::SharedFile::operator=(SharedFile &&other) noexcept
    { if(this!=&other) { reset();state_=std::move(other.state_); }return *this; }
    void MPISessionOwner::SharedFile::reset() noexcept
    {
        if(state_ && state_->creator && !state_->unlinked)try { unlink(); }
        catch(const std::exception &error) { std::fprintf(stderr,"Owned shared backing retained for ancestor cleanup: %s\n",error.what()); }
        state_.reset();
    }
    int MPISessionOwner::SharedFile::descriptor() const noexcept
    { return state_?state_->descriptor:-1; }
    MPISessionOwner::SharedIdentity MPISessionOwner::SharedFile::identity() const
    { if(!state_)throw std::logic_error("Moved owned shared backing has no identity");return state_->original; }
    const std::string &MPISessionOwner::SharedFile::path() const
    { if(!state_)throw std::logic_error("Moved owned shared backing has no path");return state_->path; }
    void MPISessionOwner::SharedFile::requireExtent(std::uint64_t expected_bytes) const
    {
        if(!state_)throw std::logic_error("Moved owned shared backing has no extent");
        state_->directory->requireOriginal();
        struct stat actual{};
        if(::fstat(state_->descriptor,&actual)!=0 ||
           SharedIdentity{actual.st_dev,actual.st_ino,actual.st_uid}!=state_->original ||
           actual.st_size<0 || static_cast<std::uint64_t>(actual.st_size)!=expected_bytes)
            throw std::runtime_error("Owned shared backing lost its original identity/extent before mapping");
    }
    void MPISessionOwner::SharedFile::unlink()
    {
        if(!state_)throw std::logic_error("Moved owned shared backing cannot retire a name");
        state_->directory->requireName(state_->basename);
        if(state_->unlinked)return;
        struct stat original{},current{};
        if(::fstat(state_->descriptor,&original)!=0 ||
           SharedIdentity{original.st_dev,original.st_ino,original.st_uid}!=state_->original)
            throw std::runtime_error("Owned shared backing lost its original descriptor identity");
        // An authenticated peer may already have retired this exact name. The
        // retained original FD, not a later successor at the basename, proves
        // that lifetime has ended. A renamed original still has links and must
        // continue to reject a replacement below.
        if(original.st_nlink==0) {state_->unlinked=true;return;}
        if(::fstatat(state_->directory->descriptor,state_->basename.c_str(),&current,AT_SYMLINK_NOFOLLOW)!=0) {
            if(errno==ENOENT) {state_->unlinked=true;return;}
            fail("cannot inspect original owned shared backing name");
        }
        if(!S_ISREG(current.st_mode) || SharedIdentity{current.st_dev,current.st_ino,current.st_uid}!=state_->original)
            throw std::runtime_error("Owned shared backing name was replaced by another inode");
        if(::unlinkat(state_->directory->descriptor,state_->basename.c_str(),0)!=0 && errno!=ENOENT)
            fail("cannot retire original owned shared backing name");
        state_->unlinked=true;
    }
    std::optional<MPISessionOwner::SharedDirectory> MPISessionOwner::sharedDirectory()
    {
        auto state=std::make_shared<SharedDirectory::State>();
        if(const auto original=managedProcess()) {
            const char *path=processEnvironmentValue(EnvironmentKey::VaderBackingDirectory);
            if(!path || !*path)throw BrokenBorrow("original shared-directory path disappeared");
            state->path=path;
            state->descriptor=::fcntl(kSharedBorrow,F_DUPFD_CLOEXEC,9);
            if(state->descriptor<0)fail("cannot duplicate authenticated original shared-directory loan");
            state->original={original->witness.shared.device,original->witness.shared.inode,original->witness.shared.uid};
        } else {
            if(!externalMPI())return std::nullopt;
            auto native=authenticateNativeRankRoots();
            if(!native)return std::nullopt;
            state->path=std::move(native->path);
            state->descriptor=std::exchange(native->directory.value,-1);
            state->original={native->original.device,native->original.inode,native->original.uid};
        }
        state->requireOriginal();
        return SharedDirectory(std::move(state));
    }
    void MPISessionOwner::requireProcessOwner()
    {
        // A broken present loan rejects even when external MPI markers exist.
        if(managedProcess())return;
        if(!externalMPI())throw std::logic_error("MPI initialization requires an explicit process-entry session owner; wrap external mpirun with llaminar_mpi_session");
        requireNativeRankPublication();
    }
    void MPISessionOwner::retireStartupLoansForMPILaunch()
    {
        if(managedProcess()) {
            for(int descriptor:{kOwnerBorrow,kTemporaryBorrow,kSharedBorrow,kProtocolBorrow})
                if(::close(descriptor)!=0)fail("cannot retire original startup loan before MPI exec");
        } else if(!externalMPI())throw std::logic_error("Native MPI exec requires its original process owner");
    }
    namespace {
        int launchOwned(const std::vector<std::string> &,const std::filesystem::path &,
                        const std::filesystem::path &,const std::filesystem::path &);
    }
    std::optional<int> MPISessionOwner::enterProcess(int argc,char **argv)
    {
        try {
            if(managedProcess())return std::nullopt;
            if(externalMPI()) { requireNativeRankPublication();return std::nullopt; }
            if(argc<1 || !argv || !argv[0])throw std::invalid_argument("Missing original frontend argv");
            const auto executable=std::filesystem::canonical("/proc/self/exe");
            std::vector<std::string> command;
            for(int i=0;i<argc;++i) {
                if(!argv[i])throw std::invalid_argument("Incomplete original frontend argv");
                command.emplace_back(argv[i]);
            }
            return launchOwned(command,sessionBase(),launcherExecutable(),executable);
        } catch(const std::exception &error) {
            std::fprintf(stderr,"%s\n",error.what());return 125;
        }
    }
    int MPISessionOwner::launch(const std::vector<std::string> &command,
                                const std::filesystem::path &base,const std::filesystem::path &helper)
    { return launchOwned(command,base,helper,{}); }
    namespace {
    /** @brief One owner loop for exact public commands and authenticated frontend executables. */
    int launchOwned(const std::vector<std::string> &command,const std::filesystem::path &base,
                    const std::filesystem::path &helper,const std::filesystem::path &executable)
    {
        if (command.empty() || command.front().empty()) throw std::invalid_argument("MPI session requires exact command argv");
        requireSupportedExportEnvironment();
        if(executable.empty() || nativeProgram(executable)!=NativeProgram::Application)requireOwnedLocations(command);
        Signals signals;
        auto guard = beginGuardian(helper,base,command,executable);
        bool done = false;
        while (!done) {
            pollfd events[]={{guard.control.value,POLLIN|POLLHUP,0},{signals.descriptor.value,POLLIN,0}};
            if (::poll(events,2,-1)<0) { if(errno==EINTR)continue; fail("cannot observe MPI launcher retirement"); }
            if(events[0].revents) {
                Packet packet;
                if(!receivePacket(guard.control.value,packet)) done=true;
                else if(packet.kind==Kind::Done) done=true;
                else throw std::runtime_error("Unexpected MPI launcher completion packet");
            }
            // Completion may race the signal descriptor. A closed peer still
            // owns an exact wait receipt; it cannot turn child success into125.
            for(int signal;(signal=signals.next())!=0;) if(signal!=SIGCHLD && !done)
                if(!forwardSignal(guard.control.value,signal))done=true;
        }
        return join(guard.pid);
    }
    } // namespace
    int MPISessionOwner::launcherMain(int argc,char **argv)
    {
        try {
            std::filesystem::path base = sessionBase();
            pid_t parent=-1; int control=-1, index=1;
            bool daemon = false;
            std::filesystem::path executable;
            if(index<argc && std::string_view(argv[index])=="--guardian-parent") {
                if(index+3>=argc)throw std::invalid_argument("Incomplete guardian identity");
                parent=std::stoi(argv[++index]); ++index;
                if(std::string_view(argv[index++])!="--control-fd")throw std::invalid_argument("Missing guardian control");
                control=std::stoi(argv[index++]);
                if(parent!=::getppid()||control!=kControl)throw std::invalid_argument("Foreign guardian parent/control");
            }
            if(parent>0 && index<argc && std::string_view(argv[index])=="--owned-executable") {
                if(++index>=argc)throw std::invalid_argument("Missing authenticated frontend executable");
                executable=argv[index++];
            }
            if(index<argc && std::string_view(argv[index])=="--mpi-daemon") { daemon=true; ++index; }
            if(index<argc && std::string_view(argv[index])=="--session-base") {
                if(++index>=argc)throw std::invalid_argument("Missing MPI session base");
                base=argv[index++];
            }
            if(index>=argc||std::string_view(argv[index++])!="--")
                throw std::invalid_argument("Usage: llaminar_mpi_session --session-base PATH -- command argv");
            std::vector<std::string> command;
            for(;index<argc;++index)command.emplace_back(argv[index]);
            if(daemon)command=localDaemon(command);
            if(parent>0) {
                if(command.empty())throw std::invalid_argument("Guardian requires its original command");
                return guardian(parent,control,base,command,executable);
            }
            return launch(command,base,std::filesystem::canonical("/proc/self/exe"));
        } catch(const std::exception &error) {
            std::fprintf(stderr,"%s\n",error.what()); return 125;
        }
    }
}
