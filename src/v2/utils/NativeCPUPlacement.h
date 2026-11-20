/**
 * @file NativeCPUPlacement.h
 * @brief Original-entry admission of native OpenMP physical-core ownership.
 *
 * The process owner prepares placement before application/runtime work. OpenMP
 * owns the resulting place table through worker retirement and recreation;
 * immutable admission data never acts as a manually maintained worker roster.
 */
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <omp.h>

namespace llaminar2
{
    class MPISessionOwner;
    class MPIBootstrapPhase;

    /** @brief Immutable original-entry evidence of this caller's native physical-core places. */
    class NativeOpenMPCorePlaces final
    {
        friend class NativeCPUPlacement;
        friend class MPIBootstrapPhase;
    public:
        NativeOpenMPCorePlaces(const NativeOpenMPCorePlaces&)=delete;
        NativeOpenMPCorePlaces& operator=(const NativeOpenMPCorePlaces&)=delete;
        /** @return Borrowed immutable native geometry for this admission's lifetime. */
        const std::vector<std::vector<int>>& places() const & noexcept { return places_; }
        const std::vector<std::vector<int>>& places() const && = delete;
    private:
        /**
         * @brief Seal an original-entry native snapshot to its process/thread.
         * @param places Actual physical-core place order.
         * @param encoding Canonical native environment representation.
         * @param binding Actual native binding policy.
         * @param process Original process identity.
         * @param thread Original rank-entry thread identity.
         */
        NativeOpenMPCorePlaces(std::vector<std::vector<int>> places,std::string encoding,
                              omp_proc_bind_t binding,long process,long thread)
            : places_(std::move(places)),encoding_(std::move(encoding)),binding_(binding),
              process_(process),thread_(thread) {}
        std::vector<std::vector<int>> places_;
        std::string encoding_;
        omp_proc_bind_t binding_;
        long process_,thread_;
    };

    /** @brief One native startup authority, shared by the frontend and its CPU-only owner. */
    class NativeCPUPlacement final
    {
        friend class MPISessionOwner;
    public:
        /** @return Original-entry admission; foreign threads or absent core admission fail. */
        static const NativeOpenMPCorePlaces& admission();
        /** @return Whether original entry admitted a native physical-core place policy. */
        static bool hasCoreAdmission() noexcept;
        /**
         * @brief Validate a capacity receipt against independently supplied observed topology.
         * @param receipt Versioned process/place publication.
         * @param native Observed native place order, never inferred from the receipt.
         * @param process Expected original process identity.
         * @param root CPU sysfs root, including recorded device-free fixtures.
         * @throws std::runtime_error for malformed, stale or noncanonical publication.
         * @details This data-only check neither mints process admission nor executes a program.
         */
        static void validateCapacityReceipt(const std::string &receipt,
            const std::vector<std::vector<int>> &native,long process,
            const std::string &root="/sys/devices/system/cpu");
        /** @return Sorted unique CPUs in a Linux list/range string. */
        static std::vector<int> parseCpuList(const std::string &list);
        /** @return First sibling identifying a physical core, or the CPU itself when unavailable. */
        static int physicalRepresentativeForCpu(int cpu);
        /**
         * @brief Read relative kernel core capacity or maximum frequency.
         * @param cpu Logical CPU id.
         * @param root CPU sysfs root, including device-free fixtures.
         * @return Positive observed capacity, or zero for unknown/uniform topology.
         */
        static long cpuCapacity(int cpu,const std::string &root="/sys/devices/system/cpu");
        /**
         * @brief Order allowed representatives by descending capacity and ascending id.
         * @param cpus Admitted CPU ids.
         * @param root CPU sysfs root.
         * @return Ordered copies of the supplied ids, never extra CPUs.
         */
        static std::vector<int> capacityOrderedCpus(std::vector<int> cpus,
            const std::string &root="/sys/devices/system/cpu");
        /**
         * @brief Group allowed siblings into capacity-ordered physical-core places.
         * @param cpus Admitted CPU ids.
         * @param root CPU sysfs root.
         * @return Disjoint places containing only supplied CPUs.
         */
        static std::vector<std::vector<int>> capacityOrderedCorePlaces(const std::vector<int> &cpus,
            const std::string &root="/sys/devices/system/cpu");
    private:
        /** @brief Claim original main-thread entry once, before any worker or MPI runtime. */
        static void claimProcessEntry();
        /**
         * @brief Prepare native places only inside authenticated original process entry.
         * @param argc Original application argument count.
         * @param argv Original application arguments, preserved across initial exec.
         */
        static void prepareProcessEntry(int argc,char *argv[]);
    };
}
