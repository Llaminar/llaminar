/**
 * @file MPISessionEnvironment.h
 * @brief One native environment declaration shared by file owner and MPI fork hook.
 *
 * This C++17 header contains immutable startup protocol/configuration names,
 * not a live memory ledger or rank/model shadow. The native hook projects these
 * same daemon-owned values after OpenMPI's ordinary application-env merge.
 * Live bootstrap reads use only the closed key vocabulary below; they cannot
 * consume DebugEnv's cached engine snapshot across setup/exec mutations.
 */
#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <sys/stat.h>

namespace llaminar2::mpi_session
{
    /** @brief Closed vocabulary for live process-entry/native-daemon environment reads. */
    enum class EnvironmentKey : std::uint8_t
    {
        TemporaryDirectory, TemporaryAlias, TempAlias,
        OrteTemporaryDirectory, PrteTemporaryDirectory,
        VaderBackingDirectory, SmBackingDirectory, MmapBackingDirectory,
        OrteLocalTemporaryDirectory, OrteRemoteTemporaryDirectory,
        PrteLocalTemporaryDirectory, PrteRemoteTemporaryDirectory,
        RequiredProtocol, ReadyPublication, ModuleDirectory, ComponentPath,
        DynamicComponents, Selector, ExportList, InternalExportList,
        OmWorldSize, PmiSize, PmixNamespace, MpiLocalRank, Count
    };
    /** @brief One source-declared variable; callers cannot supply arbitrary names to the reader. */
    struct EnvironmentBinding
    {
        EnvironmentKey key;
        const char *name;
    };
    /** @brief Physical root selected by one original process/file owner. */
    enum class RuntimeRoot { Temporary, Shared };
    /** @brief Exact native environment binding and its original physical root. */
    struct LocationBinding
    {
        const char *name;
        RuntimeRoot root;
        EnvironmentKey key;
    };
    inline constexpr LocationBinding kLocationBindings[] = {
        {"TMPDIR", RuntimeRoot::Temporary, EnvironmentKey::TemporaryDirectory},
        {"TMP", RuntimeRoot::Temporary, EnvironmentKey::TemporaryAlias},
        {"TEMP", RuntimeRoot::Temporary, EnvironmentKey::TempAlias},
        {"OMPI_MCA_orte_tmpdir_base", RuntimeRoot::Temporary, EnvironmentKey::OrteTemporaryDirectory},
        {"PRTE_MCA_prte_tmpdir_base", RuntimeRoot::Temporary, EnvironmentKey::PrteTemporaryDirectory},
        {"OMPI_MCA_btl_vader_backing_directory", RuntimeRoot::Shared, EnvironmentKey::VaderBackingDirectory},
        {"OMPI_MCA_btl_sm_backing_directory", RuntimeRoot::Shared, EnvironmentKey::SmBackingDirectory},
        {"OMPI_MCA_shmem_mmap_backing_file_base_dir", RuntimeRoot::Shared, EnvironmentKey::MmapBackingDirectory}};
    inline constexpr const char *kUnsetLocations[]={
        "OMPI_MCA_orte_local_tmpdir_base","OMPI_MCA_orte_remote_tmpdir_base",
        "PRTE_MCA_prte_local_tmpdir_base","PRTE_MCA_prte_remote_tmpdir_base"};
    inline constexpr char kProtocol[]="v1-openmpi-4.1.6";
    inline constexpr char kRequiredEnv[]="OMPI_MCA_llaminar_session_protocol";
    inline constexpr char kReadyEnv[]="OMPI_MCA_llaminar_session_ready";
    inline constexpr char kModuleDirectoryEnv[]="OMPI_MCA_llaminar_session_module_directory";
    inline constexpr char kComponentPathEnv[]="OMPI_MCA_mca_base_component_path";
    inline constexpr char kDynamicComponentsEnv[]="OMPI_MCA_mca_base_component_disable_dlopen";
    inline constexpr char kSelectorEnv[]="OMPI_MCA_schizo";
    inline constexpr char kComponentDirectory[]="llaminar_mpi_components";
    inline constexpr char kComponentFilename[]="mca_schizo_llaminar.so";
    inline constexpr char kContractSymbol[]="llaminar_mpi_native_contract_v1";
    /** @brief Per-host native controls that must follow the same daemon as its file roots. */
    inline constexpr const char *kProjectedControls[]={
        kRequiredEnv,kModuleDirectoryEnv,kComponentPathEnv,kDynamicComponentsEnv};
    /** @brief Typed views of the same conflicting aliases retired by native publication. */
    inline constexpr EnvironmentBinding kUnsetBindings[] = {
        {EnvironmentKey::OrteLocalTemporaryDirectory, kUnsetLocations[0]},
        {EnvironmentKey::OrteRemoteTemporaryDirectory, kUnsetLocations[1]},
        {EnvironmentKey::PrteLocalTemporaryDirectory, kUnsetLocations[2]},
        {EnvironmentKey::PrteRemoteTemporaryDirectory, kUnsetLocations[3]}};
    /** @brief Existing native controls, read fresh at each setup/exec transition. */
    inline constexpr EnvironmentBinding kControlBindings[] = {
        {EnvironmentKey::RequiredProtocol, kRequiredEnv},
        {EnvironmentKey::ReadyPublication, kReadyEnv},
        {EnvironmentKey::ModuleDirectory, kModuleDirectoryEnv},
        {EnvironmentKey::ComponentPath, kComponentPathEnv},
        {EnvironmentKey::DynamicComponents, kDynamicComponentsEnv},
        {EnvironmentKey::Selector, kSelectorEnv}};
    /** @brief Native export syntaxes whose owned-launch rejection requires a presence read. */
    inline constexpr EnvironmentBinding kUnsupportedExportBindings[] = {
        {EnvironmentKey::ExportList, "OMPI_MCA_mca_base_env_list"},
        {EnvironmentKey::InternalExportList, "OMPI_MCA_mca_base_env_list_internal"}};
    /** @brief External MPI launch markers; ownership remains with that explicit external parent. */
    inline constexpr EnvironmentBinding kExternalBindings[] = {
        {EnvironmentKey::OmWorldSize, "OMPI_COMM_WORLD_SIZE"},
        {EnvironmentKey::PmiSize, "PMI_SIZE"},
        {EnvironmentKey::PmixNamespace, "PMIX_NAMESPACE"},
        {EnvironmentKey::MpiLocalRank, "MPI_LOCALRANKID"}};

    /**
     * @brief Resolve a closed bootstrap key through its single source declaration.
     * @param key A declared MPI process-entry, native control, or location key.
     * @return Stable source-owned variable name, without reading mutable process state.
     * @throws std::invalid_argument for malformed/sentinel keys.
     */
    inline constexpr const char *environmentName(EnvironmentKey key)
    {
        for (const auto &binding : kLocationBindings)
            if (binding.key == key)
                return binding.name;
        for (const auto &binding : kUnsetBindings)
            if (binding.key == key)
                return binding.name;
        for (const auto &binding : kControlBindings)
            if (binding.key == key)
                return binding.name;
        for (const auto &binding : kUnsupportedExportBindings)
            if (binding.key == key)
                return binding.name;
        for (const auto &binding : kExternalBindings)
            if (binding.key == key)
                return binding.name;
        throw std::invalid_argument("Unknown MPI bootstrap environment key");
    }

    /**
     * @brief Read current process-entry/native-daemon state without a mutable cache.
     * @param key Closed source-declared MPI infrastructure variable.
     * @return Borrowed libc value, preserving unset versus empty. The pointer is valid
     * only until the next process environment mutation; callers must inspect/copy it
     * before setenv, unsetenv, or exec preparation changes that environment.
     * @throws std::invalid_argument for malformed/sentinel keys, before any libc read.
     *
     * This is the narrowly guarded bootstrap bridge for the independent MPI helper
     * and native module. DebugEnv remains the engine's runtime/debug configuration
     * authority. Its startup snapshot cannot describe these ordered environment
     * mutations, and its engine headers cannot enter the standalone C++17 module.
     */
    inline const char *processEnvironmentValue(EnvironmentKey key)
    {
        return std::getenv(environmentName(key));
    }

    /** @brief Fixed native module ABI; fixture descriptors cannot silently enter this protocol. */
    struct NativeModuleContract {
        std::uint64_t magic=0x4c4d50494e415431ull;
        std::uint32_t protocol=1;
        std::uint32_t openmpi_major=4,openmpi_minor=1,openmpi_release=6;
        std::uint32_t schizo_module_bytes=10*sizeof(void*);
    };
    /** @return Exact original daemon root identities, checked again before rank MPI_Init. */
    inline std::string rootReadiness(const char *temporary,const char *shared)
    {
        struct stat t{},s{};
        if(!temporary || !shared || ::stat(temporary,&t)!=0 || ::stat(shared,&s)!=0 ||
           !S_ISDIR(t.st_mode) || !S_ISDIR(s.st_mode))
            throw std::runtime_error("Owned MPI rank has no native environment publication: missing roots");
        return std::string(kProtocol)+":"+std::to_string(t.st_dev)+":"+std::to_string(t.st_ino)+":"+
            std::to_string(s.st_dev)+":"+std::to_string(s.st_ino);
    }
    using NativeContractQuery=const NativeModuleContract *(*)();
}
