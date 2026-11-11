/**
 * @file MoEPipelinePreparationMain.cpp
 * @brief Launch homogeneous preparation on its observed native NUMA owner.
 *
 * CTest's ordinary socket mapper supplies mixed two-rank jobs. Homogeneous
 * jobs instead declare one complete native cohort before MPI initialization;
 * the production bootstrap maps one original MPI process to that cohort's
 * observed NUMA node using its cached hwloc rankfile authority. The original
 * inventory stays complete, and the child never rebinds or relaunches itself.
 */
#include "app/MPIBootstrapPhase.h"
#include "app/MPIProcessSession.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include <gtest/gtest.h>
#include <charconv>
#include <map>
#include <set>
#include <optional>
#include <string_view>
#include <stdexcept>

namespace
{
    /** @brief Typed launch declaration; no test-name parsing selects hardware. */
    struct NativeLaunch
    {
        llaminar2::ComputeBackendType family; ///< Exact native device family.
        int degree; ///< Complete distinct-device cohort required on one NUMA node.
    };

    /**
     * @brief Parse only the fixture's explicit launch contract, retaining original argv.
     * @param argc Original unmodified argument count.
     * @param argv Original argv retained through production opaque-wrapper bootstrap.
     * @return Absent for ordinary externally launched mixed/source-lifetime jobs.
     * @throws std::invalid_argument For partial, repeated or unsupported declarations.
     */
    std::optional<NativeLaunch> nativeLaunch(int argc, char **argv)
    {
        std::optional<llaminar2::ComputeBackendType> family;
        std::optional<int> degree;
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view argument(argv[i]);
            constexpr std::string_view family_prefix = "--pipeline-native-family=";
            constexpr std::string_view degree_prefix = "--pipeline-native-degree=";
            if (argument.starts_with(family_prefix))
            {
                if (family) throw std::invalid_argument("Repeated native pipeline family");
                const auto value = argument.substr(family_prefix.size());
                if (value == "cuda") family = llaminar2::ComputeBackendType::GPU_CUDA;
                else if (value == "rocm") family = llaminar2::ComputeBackendType::GPU_ROCM;
                else throw std::invalid_argument("Pipeline native family must be cuda or rocm");
            }
            else if (argument.starts_with(degree_prefix))
            {
                if (degree) throw std::invalid_argument("Repeated native pipeline degree");
                const auto value = argument.substr(degree_prefix.size());
                int parsed = 0;
                const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed <= 0)
                    throw std::invalid_argument("Pipeline native degree must be a positive integer");
                degree = parsed;
            }
        }
        if (family.has_value() != degree.has_value())
            throw std::invalid_argument("Native pipeline launch requires both family and complete degree");
        if (family) return NativeLaunch{*family, *degree};
        return std::nullopt;
    }

    /**
     * @brief Build launch intent from the canonical full hardware publication.
     * @param declaration Explicit native family and complete physical cohort degree.
     * @return Production one-process launch configuration naming actual observed endpoints.
     * @throws std::runtime_error If the required distinct, NUMA-qualified cohort is absent.
     */
    llaminar2::OrchestrationConfig nativeConfiguration(const NativeLaunch &declaration)
    {
        llaminar2::OrchestrationConfig config;
        config.mpi_procs = 1;
        // Classify accelerator intent before discovery without restricting the
        // driver's visible devices. The child only needs this classification:
        // execute() recognizes original MPI launch and never launches again.
        const bool cuda = declaration.family == llaminar2::ComputeBackendType::GPU_CUDA;
        config.tp_devices.push_back(cuda ? llaminar2::GlobalDeviceAddress::cuda(0) : llaminar2::GlobalDeviceAddress::rocm(0));
        llaminar2::MPIBootstrapPhase::installDeviceStartupIntent(config);
        if (llaminar2::MPIBootstrap::detectMPIEnvironment().is_mpi_process) return config;
        auto &manager = llaminar2::DeviceManager::instance();
        manager.initialize(-1, false);
        std::map<int, std::vector<llaminar2::GlobalDeviceAddress>> cohorts;
        std::set<std::string> identities;
        for (const auto &device : manager.devices())
            if (device.type == declaration.family && device.numa_node >= 0)
            {
                if (device.uuid.empty() || !identities.insert(device.uuid).second)
                    throw std::runtime_error("Native pipeline launch requires unique observed GPU UUIDs");
                cohorts[device.numa_node].push_back(cuda ? llaminar2::GlobalDeviceAddress::cuda(device.device_id) :
                    llaminar2::GlobalDeviceAddress::rocm(device.device_id));
            }
        for (const auto &[numa, devices] : cohorts)
            if (devices.size() >= static_cast<size_t>(declaration.degree))
            {
                config.tp_devices.assign(devices.begin(), devices.begin() + declaration.degree);
                return config;
            }
        throw std::runtime_error("Required complete native pipeline cohort is absent from every observed NUMA node");
    }
}

/**
 * @brief Retain canonical process ownership across optional bootstrap, MPI and fixture teardown.
 * @param argc Original launch/test argument count.
 * @param argv Original opaque launch/test argument vector.
 * @return Production bootstrap status or the complete original-rank GTest result.
 */
int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;
    if (const auto declaration = nativeLaunch(argc, argv))
    {
        const auto configuration = nativeConfiguration(*declaration);
        const auto bootstrap = llaminar2::MPIBootstrapPhase{}.execute(configuration, argc, argv,
            llaminar2::MPIBootstrapArgumentPolicy::PreserveOriginalArguments);
        if (bootstrap.action == llaminar2::BootstrapResult::Action::EXIT) return bootstrap.exit_code;
        // Only wrapper arguments are consumed, after child bootstrap has seen
        // their unchanged contract. GTest receives its exact original options.
        int retained = 1;
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view argument(argv[i]);
            if (!argument.starts_with("--pipeline-native-family=") && !argument.starts_with("--pipeline-native-degree="))
                argv[retained++] = argv[i];
        }
        argc = retained;
        argv[argc] = nullptr;
    }
    auto mpi_process = llaminar2::MPIProcessSession::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
