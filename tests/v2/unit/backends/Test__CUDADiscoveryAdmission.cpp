/**
 * @file Test__CUDADiscoveryAdmission.cpp
 * @brief Contract and actual loader controls for the typed CUDA discovery boundary.
 *
 * The pure admission contract covers every observed status and binding pair,
 * including the original fatal behavior for present-but-unusable installations.
 * The binding classifier is additionally driven through the one shared native
 * probe (probeNativeLibraryBinding, the same implementation the production
 * driver-presence authority calls) with genuine loader outcomes: a genuinely
 * missing requested SONAME, a real DSO whose dependency is missing, a real DSO
 * with an unresolvable symbol, a present-but-invalid binary and a loadable
 * system library. Fixture files live in exclusive per-test temporary storage
 * with scope-bound cleanup. No CUDA SDK, driver or device is required.
 */
#include "backends/cuda/CUDADiscoveryAdmission.h"
#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using Status = CUDADeviceCountStatus;
        using Binding = CUDADriverBindingStatus;
        using Disposition = CUDADiscoveryDisposition;

        /** @brief Exclusive temporary probe file removed with its scope, even on failure. */
        class UniqueProbeFile final
        {
        public:
            /** @brief Reserve one unreadable-or-writable unique path beside the others. */
            explicit UniqueProbeFile(const std::string &stem)
                : path_(std::filesystem::temp_directory_path() /
                        (stem + "-" + std::to_string(::getpid()) + "-" +
                         std::to_string(++counter()) + ".so"))
            {
            }
            UniqueProbeFile(const UniqueProbeFile &) = delete;
            UniqueProbeFile &operator=(const UniqueProbeFile &) = delete;
            /** @brief Remove the probe file and any leftover, never throwing. */
            ~UniqueProbeFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }
            /** @return The exclusive probe path. */
            [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }

        private:
            /** @return Process-local monotonic counter for concurrent-safe names. */
            static unsigned long long &counter()
            {
                static unsigned long long value = 0;
                return value;
            }
            std::filesystem::path path_;
        };

        /** @brief Copy a real system library and rename one NUL-delimited byte string. */
        std::filesystem::path editedSystemLibraryCopy(const std::filesystem::path &destination,
            const std::string &from, const std::string &to)
        {
            EXPECT_EQ(from.size(), to.size());
            std::ifstream source("/lib/x86_64-linux-gnu/libm.so.6", std::ios::binary);
            EXPECT_TRUE(source.is_open());
            std::vector<char> bytes{std::istreambuf_iterator<char>{source},
                                    std::istreambuf_iterator<char>{}};
            const std::string delimited{std::string{'\0'} + from + '\0'};
            const auto position = std::string(bytes.data(), bytes.size()).find(delimited);
            EXPECT_NE(position, std::string::npos) << "system library lacks the expected string";
            if (position == std::string::npos)
                return {};
            std::memcpy(bytes.data() + position + 1, to.data(), to.size());
            std::ofstream output{destination, std::ios::binary | std::ios::trunc};
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            return destination;
        }
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

    TEST(CUDADriverBindingContract, SharedNativeProbeClassifiesGenuineLoaderOutcomes)
    {
        // A genuinely missing requested object: the loader's own diagnostic
        // names exactly the requested SONAME as absent.
        const std::string absentRequest = "llaminar-genuinely-absent-probe-dso.so.1";
        const auto missing = probeNativeLibraryBinding(absentRequest);
        ASSERT_FALSE(missing.loadable);
        EXPECT_EQ(classifyCUDADriverBinding(missing.loadable, absentRequest, missing.diagnostic),
            Binding::NotInstalled);

        // A real present DSO whose dependency cannot be opened: a copy of the
        // real system math library with one NEEDED entry renamed to a
        // nonexistent same-length SONAME. The loader's diagnostic names the
        // missing dependency, never the requested object's own absence.
        {
            UniqueProbeFile missingDependency{"llaminar-missing-dependency-probe"};
            const std::filesystem::path fixture = editedSystemLibraryCopy(
                missingDependency.path(), "libc.so.6", "libZ.so.9");
            const auto outcome = probeNativeLibraryBinding(fixture.string());
            ASSERT_FALSE(outcome.loadable);
            EXPECT_EQ(classifyCUDADriverBinding(outcome.loadable, fixture.string(), outcome.diagnostic),
                Binding::PresentButUnbindable);
        }

        // A real present DSO with an unresolvable symbol: the same system
        // library copy with one undefined import renamed to a nonexistent
        // same-length name. The loader's diagnostic reports the undefined
        // symbol against the requested object itself.
        {
            UniqueProbeFile undefinedSymbol{"llaminar-undefined-symbol-probe"};
            const std::filesystem::path fixture = editedSystemLibraryCopy(
                undefinedSymbol.path(), "qsort", "qsorz");
            const auto outcome = probeNativeLibraryBinding(fixture.string());
            ASSERT_FALSE(outcome.loadable);
            EXPECT_EQ(classifyCUDADriverBinding(outcome.loadable, fixture.string(), outcome.diagnostic),
                Binding::PresentButUnbindable);
        }

        // A present-but-invalid binary: the file exists, so the loader's
        // diagnostic cannot be the requested object's own absence.
        {
            UniqueProbeFile invalid{"llaminar-invalid-elf-probe"};
            {
                std::ofstream output{invalid.path(), std::ios::binary | std::ios::trunc};
                output << "this is not an ELF shared object";
            }
            const auto outcome = probeNativeLibraryBinding(invalid.path().string());
            ASSERT_FALSE(outcome.loadable);
            EXPECT_EQ(classifyCUDADriverBinding(outcome.loadable, invalid.path().string(), outcome.diagnostic),
                Binding::PresentButUnbindable);
        }

        // A loadable system library proves the positive presence path through
        // the same shared probe.
        const auto loadable = probeNativeLibraryBinding("libm.so.6");
        ASSERT_TRUE(loadable.loadable);
        EXPECT_EQ(classifyCUDADriverBinding(loadable.loadable, "libm.so.6", loadable.diagnostic),
            Binding::Loadable);
    }
}
