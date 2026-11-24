/**
 * @file Test__RoundTripContracts.cpp
 * @brief Device-free executable contracts for explicit candidates, live geometry and data oracle.
 *
 * This small diagnostic test requires neither MPI nor a GPU. It uses checked
 * runtime assertions even in Release; source scanning cannot prove geometry.
 */
#include "RoundTripContract.h"
#include <iostream>
#include <cstdlib>
#include <new>
#include <limits>
#include <vector>

using namespace llaminar2::mpi_roundtrip;
namespace { std::size_t heapAllocations = 0; }
/** @brief Observe actual host allocations so successful transport checks cannot hide heap traffic. */
void* operator new(std::size_t bytes) {
    ++heapAllocations;
    if (void* result = std::malloc(bytes ? bytes : 1)) return result;
    throw std::bad_alloc();
}
/** @brief Retire observed allocations through their original C heap owner. */
void operator delete(void* value) noexcept { std::free(value); }
/** @brief Preserve the sized-deallocation ABI used by optimized C++ builds. */
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
/** @brief Fail the executable on a false invariant in every build type. */
void require(bool value) { if (!value) throw std::runtime_error("roundtrip contract failed"); }
/** @brief Require a typed invalid-input diagnostic without executing native work. */
template<class F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
/** @brief Exercise boundary geometry and transaction-varying exact FP32 values. */
int main() {
    volatile int successfulStatus = 0;
    const auto beforeChecks = heapAllocations;
    for (unsigned i = 0; i < 1000; ++i)
        checkNativeStatus(successfulStatus, "long native operation diagnostic exceeding small-string storage");
    require(heapAllocations == beforeChecks);
    bool nativeFailure = false;
    try { checkNativeStatus(22, "native operation"); }
    catch (const std::runtime_error& error) { nativeFailure = std::string_view(error.what()).find("22") != std::string_view::npos; }
    require(nativeFailure);
    validateNativeMr(VerbsRegistration::Ordinary, 0x10000, 4096, 0x10000, 4096);
    validateNativeMr(VerbsRegistration::DmaBuf, 0, 4096, 0x10000, 4096);
    validateNativeMr(VerbsRegistration::DmaBuf, 4096, 4096, 0x10000, 4096, 4096);
    for (const auto registration : {VerbsRegistration::Ordinary, VerbsRegistration::DmaBuf}) {
        bool wrongAddress = false, wrongExtent = false;
        try { validateNativeMr(registration, 123, 4096, 0x10000, 4096); }
        catch (const std::runtime_error&) { wrongAddress = true; }
        try { validateNativeMr(registration, registration == VerbsRegistration::Ordinary ? 0x10000 : 0, 8192, 0x10000, 4096); }
        catch (const std::runtime_error&) { wrongExtent = true; }
        require(wrongAddress && wrongExtent);
    }
    require(parseAllocation("mapped") == Allocation::Mapped);
    require(parseAllocation("write-combined") == Allocation::WriteCombined);
    require(parseAllocation("registered") == Allocation::Registered);
    require(parseAllocation("device") == Allocation::Device);
    rejects([] { parseAllocation("auto"); });
    require(parseRunPolicy("mapped").plane == DataPlane::MPI);
    require(parseRunPolicy("mapped-dmabuf").plane == DataPlane::VerbsDmaBuf);
    require(parseRunPolicy("mapped-dmabuf").allocation == Allocation::Mapped);
    require(parseRunPolicy("write-combined-verbs").plane == DataPlane::VerbsOrdinary);
    require(parseRunPolicy("write-combined-verbs").allocation == Allocation::WriteCombined);
    rejects([] { parseRunPolicy("device-dmabuf"); });
    for (std::size_t words : {1, 2, 17, 257, 65536}) for (std::size_t rails : {1, 2, 3, 5, 17}) {
        LiveRailPlan plan(Geometry(words * 4), rails);
        require(plan.activeRails() == std::min(words, rails));
        std::size_t consumed = 0;
        for (std::size_t i = 0; i < plan.activeRails(); ++i) {
            const auto slice = plan.slice(i);
            require(slice.offsetWords == consumed && slice.words > 0);
            consumed += slice.words;
        }
        require(consumed == words);
        rejects([&] { plan.slice(plan.activeRails()); });
    }
    rejects([] { LiveRailPlan(Geometry(4), 0); });
    TransportReceiptFrontier frontier;
    require(!frontier.ready(1));
    frontier.observe(1); frontier.observe(2); // Next receive may precede previous send CQE.
    require(frontier.ready(1) && frontier.ready(2) && !frontier.ready(3));
    for (unsigned stale : {0, 1, 2, 4}) {
        bool rejected = false;
        try { frontier.observe(stale); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected);
    }
    frontier.observe(3); require(frontier.ready(3));
    rejects([] { Geometry(4).alignedAllocationBytes(0); });
    rejects([] { Geometry(4).alignedAllocationBytes(3); });
    for (std::size_t page : {4096, 16384, 65536}) {
        const Geometry geometry(16384);
        const auto extent = geometry.alignedAllocationBytes(page);
        require(extent % page == 0 && extent >= geometry.allocationBytes());
        require(extent - geometry.allocationBytes() < page);
    }
    for (const auto* input : {"", "0", "-1", "+1", "1foo", "184467440737095516160"})
        rejects([&] { positive(input); });
    require(positive("16384") == 16384);
    rejects([] { Geometry(0); }); rejects([] { Geometry(3); });
    rejects([] { Geometry((static_cast<std::size_t>(INT_MAX) + 1) * sizeof(float)); });
    for (std::size_t bytes : {4, 1020, 1024, 1028, 16384, 65536, 1048576}) {
        Geometry g(bytes);
        require(g.words * 4 == bytes);
        require(g.allocationBytes() == bytes + 2 * Geometry::guardWords * 4);
        for (unsigned workers : {1, 3, 5, 20, 129}) {
            auto blocks = g.blocks(256, workers);
            require(blocks > 0 && blocks <= workers && (blocks - 1) * 256 < g.words);
            std::vector<unsigned> visits(g.words);
            for (unsigned worker = 0; worker < blocks * 256; ++worker)
                for (std::size_t i = worker; i < g.words; i += blocks * 256) ++visits[i];
            for (const auto count : visits) require(count == 1);
        }
    }
    rejects([] { Geometry(4).blocks(0, 1); });
    rejects([] { Geometry(4).blocks(256, 0); });
    for (unsigned long long epoch : {1, 2, 511, 1000000})
        for (std::size_t i = 0; i < 8193; ++i) {
            const float sum = reducedValue(i, epoch);
            require(sum == sourceValue(i, epoch) + peerValue(i, epoch));
            require(sum > 0 && sum <= 3071);
            require(sourceValue(i, epoch) != sourceValue(i, epoch + 1));
        }
    std::cout << "roundtrip contracts PASS\n";
}
