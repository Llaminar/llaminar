/**
 * @file Test__CUDADiscoveryAdmission.cpp
 * @brief Contract and actual loader controls for the typed CUDA discovery boundary.
 *
 * The pure admission contract covers every observed status and binding pair,
 * including the original fatal behavior for present-but-unusable installations.
 * The binding classifier is additionally driven by real loader outcomes: a
 * genuinely missing optional library, a present-but-invalid binary and a
 * loadable system library all produce their genuine dlopen diagnostics, so
 * absence is distinguished from breakage by the actual producer, not by a
 * hand-supplied enum. No CUDA SDK, driver or device is required.
 */
#include "backends/cuda/CUDADiscoveryAdmission.h"
#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <string_view>

namespace llaminar2::test
{
    namespace
    {
        using Status = CUDADeviceCountStatus;
        using Binding = CUDADriverBindingStatus;
        using Disposition = CUDADiscoveryDisposition;

        /** @brief One real dlopen attempt returning its genuine loader diagnostic. */
        struct LoaderOutcome
        {
            bool loadable = false; ///< Whether the loader returned a handle.
            std::string diagnostic; ///< The loader's exact dlerror text, empty when none.
        };

        /** @brief Drive the actual loader for one requested object name or path. */
        LoaderOutcome probeLoader(const std::string &request)
        {
            (void)dlerror(); // Clear only the loader's thread-local diagnostic.
            if (void *handle = dlopen(request.c_str(), RTLD_NOW | RTLD_LOCAL))
            {
                dlclose(handle);
                return {true, {}};
            }
            const char *error = dlerror();
            return {false, error != nullptr ? std::string{error} : std::string{}};
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

    TEST(CUDADriverBindingContract, OnlyTheRequestedSonamesOwnAbsenceIsNonInstallation)
    {
        constexpr std::string_view soname = "libcuda.so.1";
        EXPECT_EQ(classifyCUDADriverBinding(false, soname,
            "libcuda.so.1: cannot open shared object file: No such file or directory"), Binding::NotInstalled);
        // A diagnostic naming any other object, an unresolved symbol, an invalid
        // format, an access failure or no diagnostic at all means the requested
        // library was found: a present, broken installation.
        EXPECT_EQ(classifyCUDADriverBinding(false, soname,
            "libother.so.6: cannot open shared object file: No such file or directory"), Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, "libcuda.so.1: undefined symbol: cuInit"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, "libcuda.so.1: file not recognized"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, "libcuda.so.1: cannot open shared object file: Permission denied"),
            Binding::PresentButUnbindable);
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, {}), Binding::PresentButUnbindable);
        // A truncated diagnostic cannot positively establish absence.
        EXPECT_EQ(classifyCUDADriverBinding(false, soname, "libcuda.so.1: cannot open"), Binding::PresentButUnbindable);
    }

    TEST(CUDADriverBindingContract, ActualLoaderOutcomesClassifyThroughTheProducer)
    {
        // A genuinely missing optional library: the real loader names exactly
        // the requested object as absent.
        const auto missing = probeLoader("llaminar-genuinely-absent-probe-dso.so.1");
        ASSERT_FALSE(missing.loadable);
        EXPECT_EQ(classifyCUDADriverBinding(missing.loadable, "llaminar-genuinely-absent-probe-dso.so.1",
            missing.diagnostic), Binding::NotInstalled);

        // A present-but-invalid binary: the file exists, so the loader's
        // diagnostic cannot be the requested object's own absence.
        const std::filesystem::path broken = std::filesystem::temp_directory_path() /
            "llaminar-invalid-elf-probe.so";
        {
            FILE *file = std::fopen(broken.c_str(), "wb");
            ASSERT_NE(file, nullptr);
            const char payload[] = "this is not an ELF shared object";
            ASSERT_EQ(std::fwrite(payload, 1, sizeof(payload), file), sizeof(payload));
            ASSERT_EQ(std::fclose(file), 0);
        }
        const auto invalid = probeLoader(broken.string());
        ASSERT_FALSE(invalid.loadable);
        EXPECT_EQ(classifyCUDADriverBinding(invalid.loadable, broken.string(), invalid.diagnostic),
            Binding::PresentButUnbindable);

        // A loadable system library proves the positive presence path.
        const auto loadable = probeLoader("libm.so.6");
        ASSERT_TRUE(loadable.loadable);
        EXPECT_EQ(classifyCUDADriverBinding(loadable.loadable, "libm.so.6", loadable.diagnostic),
            Binding::Loadable);
    }
}
