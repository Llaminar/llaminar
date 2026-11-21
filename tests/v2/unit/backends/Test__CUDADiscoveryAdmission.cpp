/**
 * @file Test__CUDADiscoveryAdmission.cpp
 * @brief Contract and actual loader controls for the typed CUDA discovery boundary.
 *
 * The pure admission contract covers every observed status and binding pair,
 * including the original fatal behavior for present-but-unusable installations.
 * The binding classifier is additionally driven through the one shared native
 * probe (probeNativeLibraryBinding, the same implementation the production
 * driver-presence authority calls) with genuine loader outcomes: a genuinely
 * missing requested SONAME, build-owned fixtures whose dependency and import
 * contracts this tree owns (one NEEDED entry provided only by a link-time
 * stub that never ships, one import nothing defines), a present-but-invalid
 * binary and a loadable build-owned library. Every real-fault control is
 * architecture-neutral and asserts the loader's own diagnostic names the
 * owned fault. Invalid-binary bytes live in exclusive per-test temporary
 * storage with scope-bound cleanup. No CUDA SDK, driver or device is
 * required, and no arbitrary system library is inspected.
 */
#include "backends/cuda/CUDADiscoveryAdmission.h"
#include <gtest/gtest.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef LLAMINAR_NATIVE_LOADER_MISSING_DEPENDENCY_FIXTURE
#error "the native loader proof requires its build-owned missing-dependency fixture"
#endif
#ifndef LLAMINAR_NATIVE_LOADER_UNDEFINED_SYMBOL_FIXTURE
#error "the native loader proof requires its build-owned undefined-symbol fixture"
#endif
#ifndef LLAMINAR_NATIVE_LOADER_LOADABLE_FIXTURE
#error "the native loader proof requires its build-owned loadable fixture"
#endif

namespace llaminar2::test
{
    namespace
    {
        using Status = CUDADeviceCountStatus;
        using Binding = CUDADriverBindingStatus;
        using Disposition = CUDADiscoveryDisposition;

        /**
         * @brief Atomically reserved exclusive probe directory with scope-bound cleanup.
         *
         * mkdtemp both creates and reserves one unique 0700 directory in a
         * single atomic step, so concurrent test processes can never share,
         * collide on or pre-create another run's storage; a stale process id
         * cannot make a name predictable. A failed reservation refuses
         * construction fatally with the operating system's own error, so an
         * owner can never exist without reserved storage and later probe
         * paths can never silently resolve relative to the working
         * directory. Destruction removes the directory tree even when
         * assertions fail during unwinding.
         */
        class ExclusiveProbeDirectory final
        {
        public:
            /** @brief Reserve one exclusive directory or refuse construction fatally. */
            explicit ExclusiveProbeDirectory(const std::string &stem)
            {
                const std::string pattern =
                    (std::filesystem::temp_directory_path() / (stem + "-XXXXXX")).string();
                std::vector<char> buffer(pattern.begin(), pattern.end());
                buffer.push_back('\0');
                char *const created = ::mkdtemp(buffer.data());
                if (created == nullptr)
                    throw std::system_error{errno, std::generic_category(),
                        "cannot reserve an exclusive probe directory"};
                path_ = created;
            }
            ExclusiveProbeDirectory(const ExclusiveProbeDirectory &) = delete;
            ExclusiveProbeDirectory &operator=(const ExclusiveProbeDirectory &) = delete;
            /** @brief Remove the reserved tree, never throwing. */
            ~ExclusiveProbeDirectory()
            {
                if (!path_.empty())
                {
                    std::error_code ignored;
                    std::filesystem::remove_all(path_, ignored);
                }
            }
            /** @return One file path inside the exclusive directory. */
            [[nodiscard]] std::filesystem::path file(const std::string &name) const
            {
                return path_ / name;
            }

        private:
            std::filesystem::path path_;
        };

        /** @brief The build-owned fixture whose dependency is genuinely absent. */
        constexpr std::string_view missingDependencyFixture{
            LLAMINAR_NATIVE_LOADER_MISSING_DEPENDENCY_FIXTURE};
        /** @brief The build-owned fixture with one unresolvable import. */
        constexpr std::string_view undefinedSymbolFixture{
            LLAMINAR_NATIVE_LOADER_UNDEFINED_SYMBOL_FIXTURE};
        /** @brief The build-owned dependency-light loadable fixture. */
        constexpr std::string_view loadableFixture{
            LLAMINAR_NATIVE_LOADER_LOADABLE_FIXTURE};
        /** @brief The needed SONAME has no runtime file under that name. */
        constexpr std::string_view absentDependencyName{
            "libllaminar-absent-dependency.so.9"};
        /** @brief The import nothing in this tree or a system provides. */
        constexpr std::string_view undefinedImportName{
            "llaminar_test_undefined_import_v1"};
    }

    TEST(CUDADiscoveryAdmissionContract, SuccessAndNoDeviceAdmitWithoutProbingBreakage)
    {
        EXPECT_EQ(classifyCUDADiscovery(Status::Success, Binding::NotInstalled), Disposition::EnumerateVisibility);
        EXPECT_EQ(classifyCUDADiscovery(Status::Success, Binding::PresentButUnbindable), Disposition::EnumerateVisibility);
        EXPECT_EQ(classifyCUDADiscovery(Status::Success, Binding::Loadable), Disposition::EnumerateVisibility);
        EXPECT_EQ(classifyCUDADiscovery(Status::NoDevice, Binding::NotInstalled), Disposition::EmptyVisibility);
        EXPECT_EQ(classifyCUDADiscovery(Status::NoDevice, Binding::PresentButUnbindable), Disposition::EmptyVisibility);
        EXPECT_EQ(classifyCUDADiscovery(Status::NoDevice, Binding::Loadable), Disposition::EmptyVisibility);
    }

    TEST(CUDADiscoveryAdmissionContract, OnlyPositivelyDriverlessInsufficiencyIsEmpty)
    {
        EXPECT_EQ(classifyCUDADiscovery(Status::InsufficientDriver, Binding::NotInstalled),
            Disposition::EmptyVisibility);
        // Original fatal behavior is retained whenever any driver installation
        // is present: an incompatible driver and an unbindable one are defects.
        EXPECT_EQ(classifyCUDADiscovery(Status::InsufficientDriver, Binding::Loadable),
            Disposition::FatalQueryFailure);
        EXPECT_EQ(classifyCUDADiscovery(Status::InsufficientDriver, Binding::PresentButUnbindable),
            Disposition::FatalQueryFailure);
    }

    TEST(CUDADiscoveryAdmissionContract, MismatchAndQueryFailuresStayFatalForEveryBinding)
    {
        for (const auto binding : {Binding::NotInstalled, Binding::PresentButUnbindable, Binding::Loadable})
        {
            EXPECT_EQ(classifyCUDADiscovery(Status::SystemDriverMismatch, binding), Disposition::FatalQueryFailure);
            EXPECT_EQ(classifyCUDADiscovery(Status::QueryFailure, binding), Disposition::FatalQueryFailure);
        }
    }

    TEST(CUDADriverBindingContract, LoadableLibraryIsPresent)
    {
        EXPECT_EQ(classifyCUDADriverBinding(true, "libcuda.so.1", {}), Binding::Loadable);
        EXPECT_EQ(classifyCUDADriverBinding(true, "libcuda.so.1", "ignored diagnostic"), Binding::Loadable);
    }

    TEST(CUDADriverBindingContract, OnlyTheCompleteExactNonInstallationDiagnosticAbsents)
    {
        constexpr std::string_view soname = "libcuda.so.1";
        EXPECT_EQ(classifyCUDADriverBinding(false, soname,
            "libcuda.so.1: cannot open shared object file: No such file or directory"), Binding::NotInstalled);
        // Authored policy controls for outcomes the loader's own words must
        // never turn into absence: another named object, an unresolved symbol,
        // an invalid format, an access failure, no diagnostic at all, an empty
        // request, truncated, suffixed or otherwise malformed text.
        EXPECT_EQ(classifyCUDADriverBinding(false, soname,
            "libother.so.6: cannot open shared object file: No such file or directory"), Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, "libcuda.so.1: undefined symbol: cuInit"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, "libcuda.so.1: file not recognized"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, "libcuda.so.1: cannot open shared object file: Permission denied"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, {}), Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, {}, "libcuda.so.1: cannot open shared object file: No such file or directory"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, "libcuda.so.1: cannot open"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname,
            "libcuda.so.1: cannot open shared object file: No such file or directory: but present"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname,
            "libcuda.so.1: cannot open shared object file: Datei nicht gefunden"), Binding::PresentButUnbindable);
    }

    TEST(CUDADriverBindingContract, ProbeDirectoryConstructionRefusesFatally)
    {
        // Construction-failure control: the stem names a directory whose
        // parent was reserved exclusively but whose own parent path does not
        // exist, so the atomic reservation cannot succeed. Construction must
        // refuse fatally with a system error, never hand out a usable-looking
        // owner, and never leave a probe file resolved relative to the
        // working directory.
        ExclusiveProbeDirectory owner{"llaminar-probe-construction-root"};
        const std::string unreachable =
            (owner.file("missing-parent") / "probe").string();
        ASSERT_FALSE(std::filesystem::exists(unreachable));
        EXPECT_THROW(
            { [[maybe_unused]] ExclusiveProbeDirectory refused{unreachable}; },
            std::system_error);
        std::error_code ignored;
        EXPECT_FALSE(std::filesystem::exists(
            std::filesystem::current_path() / "probe", ignored))
            << "a refused owner must not create a relative probe file";
    }

    TEST(CUDADriverBindingContract, SharedNativeProbeClassifiesGenuineLoaderOutcomes)
    {
        // A genuinely missing requested object: the loader's own diagnostic
        // names exactly the requested SONAME as absent.
        const std::string absentRequest = "llaminar-genuinely-absent-probe-dso.so.1";
        const auto missing = probeNativeLibraryBinding(absentRequest);
        ASSERT_FALSE(missing.loadable);
        EXPECT_EQ(classifyCUDADriverBinding(missing.loadable, absentRequest, missing.diagnostic),
            Binding::NotInstalled);

        // A real present DSO whose one dependency cannot be opened: the
        // build-owned fixture linked against a stub that existed only during
        // the link. The loader's diagnostic names the missing dependency,
        // never the requested object's own absence. The contract is owned by
        // this tree, so the control is identical on every architecture.
        {
            const auto outcome = probeNativeLibraryBinding(missingDependencyFixture);
            ASSERT_FALSE(outcome.loadable);
            EXPECT_NE(outcome.diagnostic.find(absentDependencyName), std::string::npos)
                << "the loader must name the missing dependency: " << outcome.diagnostic;
            EXPECT_EQ(classifyCUDADriverBinding(outcome.loadable, missingDependencyFixture,
                outcome.diagnostic), Binding::PresentButUnbindable);
        }

        // A real present DSO with an unresolvable symbol: the build-owned
        // fixture importing a name nothing provides. The loader's diagnostic
        // reports the undefined-symbol fault and names the import against
        // the requested object itself.
        {
            const auto outcome = probeNativeLibraryBinding(undefinedSymbolFixture);
            ASSERT_FALSE(outcome.loadable);
            EXPECT_NE(outcome.diagnostic.find("undefined symbol"), std::string::npos)
                << "the loader must report the fault kind: " << outcome.diagnostic;
            EXPECT_NE(outcome.diagnostic.find(undefinedImportName), std::string::npos)
                << "the loader must name the unresolvable import: " << outcome.diagnostic;
            EXPECT_EQ(classifyCUDADriverBinding(outcome.loadable, undefinedSymbolFixture,
                outcome.diagnostic), Binding::PresentButUnbindable);
        }

        // A present-but-invalid binary: the file exists, so the loader's
        // diagnostic cannot be the requested object's own absence.
        {
            ExclusiveProbeDirectory invalid{"llaminar-invalid-elf-probe"};
            const std::filesystem::path probe = invalid.file("not-an-elf.so");
            {
                std::ofstream output{probe, std::ios::binary | std::ios::trunc};
                output << "this is not an ELF shared object";
            }
            const auto outcome = probeNativeLibraryBinding(probe.string());
            ASSERT_FALSE(outcome.loadable);
            EXPECT_EQ(outcome.diagnostic.find("No such file"), std::string::npos)
                << "an existing file must not be reported absent: " << outcome.diagnostic;
            EXPECT_EQ(classifyCUDADriverBinding(outcome.loadable, probe.string(), outcome.diagnostic),
                Binding::PresentButUnbindable);
        }

        // A loadable build-owned library proves the positive presence path
        // through the same shared probe.
        const auto loadable = probeNativeLibraryBinding(loadableFixture);
        ASSERT_TRUE(loadable.loadable);
        EXPECT_EQ(classifyCUDADriverBinding(loadable.loadable, loadableFixture,
            loadable.diagnostic), Binding::Loadable);
    }
}
