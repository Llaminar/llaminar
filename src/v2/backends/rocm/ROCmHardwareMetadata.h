/**
 * @file ROCmHardwareMetadata.h
 * @brief Immutable physical device facts published once before ROCm execution.
 *
 * Visibility is frozen for the process lifetime. Runtime-context generations,
 * loaded kernel images and compiler resource observations have separate owners;
 * resetting a context never refreshes this inventory or creates another context.
 * The injected source keeps publication/race/identity regressions device-free.
 */
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace llaminar2
{
    /** @brief Historical public precision answers, not instruction/MMA qualification. */
    struct ROCmLegacyPrecisionQueries
    {
        bool fp16 = false, bf16 = false, int8 = false;
    };
    /** @brief One visible device's physical identity and limits, independent of compiled wave width. */
    struct ROCmDeviceHardware
    {
        int ordinal = -1;
        std::string name, isa_target, pci_address;
        std::array<std::uint8_t, 16> uuid{};
        int pci_domain = 0, pci_bus = 0, pci_device = 0;
        int compute_units = 0, native_wave_width = 0, maximum_threads_per_block = 0;
        bool can_use_stream_wait_value = false; ///< SDK-reported device/runtime support for 32/64-bit stream wait-value primitives.
        int wall_clock_rate_khz = 0; ///< Constant-frequency wall_clock64 clock, not the variable core clock.
        std::size_t lds_per_block = 0, total_memory = 0, last_level_cache = 0;
        // Compatibility only: these preserve existing API answers and must not
        // select a native kernel, wave geometry or a future matrix implementation.
        ROCmLegacyPrecisionQueries legacy_precision;
    };
    /**
     * @brief Observe the exact native stream-wait capability at the once-only discovery boundary.
     * @param ordinal Exact visible native participant.
     * @param query Native hipDeviceGetAttribute adapter or a recorded source returning native status.
     * @return Successful supported=1 or unsupported=0 observation, never a query failure.
     * @throws std::runtime_error On native query failure or incomplete/nonboolean successful output.
     */
    template<class Query>
    [[nodiscard]] bool observeROCmStreamWaitValueCapability(int ordinal, Query &&query)
    {
        int supported = -1;
        const int status = static_cast<int>(std::invoke(std::forward<Query>(query), &supported, ordinal));
        if (status != 0)
            throw std::runtime_error("hipDeviceGetAttribute(CanUseStreamWaitValue) failed at ordinal " +
                std::to_string(ordinal) + ": native status=" + std::to_string(status));
        if (supported != 0 && supported != 1)
            throw std::runtime_error("Incomplete ROCm stream-wait capability at ordinal " + std::to_string(ordinal));
        return supported == 1;
    }
    /** @brief Validated process-lifetime inventory with no fixed ordinal ceiling. */
    class ROCmHardwareInventory final
    {
    public:
        /**
         * @brief Admit a complete visible-device snapshot atomically.
         * @param devices Ordered physical records; an actually empty visibility set is valid.
         * @throws std::invalid_argument On partial, misordered or invalid records.
         */
        explicit ROCmHardwareInventory(std::vector<ROCmDeviceHardware> devices)
            : devices_(std::move(devices))
        {
            std::set<std::array<std::uint8_t, 16>> uuids;
            std::set<std::string> endpoints;
            for (std::size_t index = 0; index < devices_.size(); ++index)
            {
                const auto &device = devices_[index];
                if (device.ordinal != static_cast<int>(index) || device.name.empty() ||
                    device.isa_target.empty() || device.pci_address.empty() ||
                    device.compute_units <= 0 || device.native_wave_width <= 0 ||
                    device.maximum_threads_per_block <= 0 || device.wall_clock_rate_khz <= 0 || device.lds_per_block == 0 ||
                    device.total_memory == 0)
                    throw std::invalid_argument("Incomplete ROCm physical metadata at ordinal " + std::to_string(index));
                if (std::all_of(device.uuid.begin(), device.uuid.end(), [](auto byte) { return byte == 0; }))
                    throw std::invalid_argument("Absent ROCm physical UUID at ordinal " + std::to_string(index));
                if (!uuids.insert(device.uuid).second)
                    throw std::invalid_argument("Duplicate ROCm physical UUID at ordinal " + std::to_string(index));
                if (!endpoints.insert(device.pci_address).second)
                    throw std::invalid_argument("Duplicate ROCm PCI endpoint at ordinal " + std::to_string(index));
            }
        }
        /** @return Frozen visible-device count; no runtime query occurs. */
        [[nodiscard]] std::size_t size() const noexcept { return devices_.size(); }
        /**
         * @brief Acquire one immutable visible-device record.
         * @param ordinal Exact process-local visible ordinal.
         * @return Stable reference retained across runtime context resets.
         * @throws std::out_of_range On an invalid ordinal; never aliases device zero.
         */
        [[nodiscard]] const ROCmDeviceHardware &at(int ordinal) const
        {
            if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= devices_.size())
                throw std::out_of_range("Invalid ROCm physical device ordinal " + std::to_string(ordinal));
            return devices_[static_cast<std::size_t>(ordinal)];
        }
    private:
        std::vector<ROCmDeviceHardware> devices_;
    };
    /**
     * @brief Bind the admitted constant-frequency clock for service tick conversion.
     * @param hardware Exact owning device's immutable physical record.
     * @return Wall-clock kilohertz; never substitutes variable-frequency core clockRate.
     * @throws std::invalid_argument For an incomplete physical clock binding.
     */
    [[nodiscard]] inline std::uint64_t rocmServiceWallClockRateKHz(const ROCmDeviceHardware &hardware)
    {
        if (hardware.ordinal < 0 || hardware.wall_clock_rate_khz <= 0)
            throw std::invalid_argument("Incomplete ROCm service wall-clock binding");
        return static_cast<std::uint64_t>(hardware.wall_clock_rate_khz);
    }
    /** @brief Single publication authority; steady-state readers use one acquire load. */
    class ROCmHardwareAuthority final
    {
    public:
        /**
         * @brief Enumerate once at an explicit ROCm startup boundary.
         * @param source Complete hardware reader; tests supply device-free records.
         * @throws Any source/admission exception, retained permanently without retry.
         */
        void initialize(const std::function<std::vector<ROCmDeviceHardware>()> &source)
        {
            std::call_once(initialization_, [&] {
                try {
                    inventory_ = std::make_unique<const ROCmHardwareInventory>(source());
                    published_.store(inventory_.get(), std::memory_order_release);
                } catch (...) { failure_ = std::current_exception(); }
            });
            if (failure_) std::rethrow_exception(failure_);
        }
        /**
         * @brief Read the already-admitted snapshot without a mutex or HIP query.
         * @return The process-lifetime hardware inventory.
         * @throws std::logic_error Before successful explicit initialization.
         */
        [[nodiscard]] const ROCmHardwareInventory &snapshot() const
        {
            const auto *inventory = published_.load(std::memory_order_acquire);
            if (!inventory) throw std::logic_error("ROCm physical metadata was not admitted before execution");
            return *inventory;
        }
    private:
        std::once_flag initialization_;
        std::unique_ptr<const ROCmHardwareInventory> inventory_;
        std::atomic<const ROCmHardwareInventory *> published_{nullptr};
        std::exception_ptr failure_;
    };
    /**
     * @brief Bind host graph-recording hardware only after a successful current-device query.
     * @tparam Query Native status-returning query accepting an output ordinal pointer.
     * @tparam Lookup Immutable hardware lookup callable receiving the authenticated ordinal.
     * @param query Production hipGetDevice adapter or a recorded host-test query.
     * @param lookup Hardware authority lookup, invoked only after native query success.
     * @return The lookup's exact hardware reference; no default ordinal is invented.
     * @throws std::runtime_error On query failure, before any lookup or launch is possible.
     */
    template <typename Query, typename Lookup>
    [[nodiscard]] decltype(auto) requireROCmCurrentHardware(Query &&query, Lookup &&lookup)
    {
        int ordinal = -1;
        const int status = static_cast<int>(std::invoke(std::forward<Query>(query), &ordinal));
        if (status != 0)
            throw std::runtime_error("hipGetDevice failed before ROCm decode hardware lookup: status=" +
                                     std::to_string(status));
        return std::invoke(std::forward<Lookup>(lookup), ordinal);
    }
    /** @brief Explicit ROCm startup enumeration; CPU-only startup never calls this. */
    void initializeROCmHardwareMetadata();
    /** @return Already-admitted immutable metadata; no lazy driver/context initialization. */
    [[nodiscard]] const ROCmHardwareInventory &rocmHardwareMetadata();
}
