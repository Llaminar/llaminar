/**
 * @file CUDAHCFFNRooflineWork.cpp
 * @brief Enumerate compulsory packed-weight requests and arithmetic for a local QFN decode row.
 *
 * The original source roster supplies the FFN roles. The prepared storage
 * catalog owns their byte counts. Work is counted before the explicit
 * collective, and only local routed expert projections enter the estimate.
 */
#include "fixtures/qwen4exp/FullNativeHCFFNFamilyCases.h"
#include "tensors/NativeQuantisedFormatInfo.h"

#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Original model dimensions and the exact positive branch's live local route count. */
        struct Geometry
        {
            size_t hidden, rank, routed, shared, experts, selected, local;
        };

        /** @brief Independent reference work for one phase on the native timeline. */
        struct Phase
        {
            size_t bytes, int_macs, fp32_macs;
        };

        /** @brief Resolve one original source or reject an incomplete GGUF K block. */
        const NativeQuantisedFormatInfo* source(const char* name, size_t reduction)
        {
            const auto* format = native_quantised_formats::forQuantType(name);
            return format && reduction % (format->is_superblock ? 256 : 32) == 0 ? format : nullptr;
        }

        /** @brief Report one exact source tuple and its ordered local phases. */
        void printFamily(const FullNativeHCFFNFamilyCase& recipe, const Geometry& g)
        {
            using native_quantised_formats::Q8_0;
            const auto* gate = source(recipe.gate, g.hidden);
            const auto* down = source(recipe.down, g.routed);
            const auto* shared = source(recipe.shared, g.hidden);
            const auto* shared_down_name = recipe.shared_down ? recipe.shared_down : recipe.shared;
            const auto* shared_down = source(shared_down_name, g.shared);
            const bool admitted = gate && down && shared && shared_down;
            std::printf("%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s,%s,%s", recipe.name, g.hidden,
                g.rank, g.routed, g.shared, g.experts, g.selected, g.local, recipe.gate, recipe.down,
                recipe.shared, shared_down_name, admitted ? "complete" : "incompatible_source_k");
            if (!admitted)
            {
                std::puts(",,,,,,,,,,,,,,,");
                return;
            }

            const auto bytes = [](size_t n, size_t k, const auto& info) {
                return nativeQuantisedPackedRegionSizes(n, k, info).logicalBytes();
            };
            const size_t hc = 4 * g.hidden * g.rank;
            const size_t routed = g.local * g.hidden * g.routed;
            const size_t shared_work = g.hidden * g.shared;
            // Both HC Up terms multiply the same weight matrix by different Q8 rows.
            const std::array phases{
                Phase{bytes(g.rank, 4 * g.hidden, Q8_0), hc, 0},
                Phase{bytes(4 * g.hidden, g.rank, Q8_0), 2 * hc, 0},
                Phase{g.hidden * g.experts * sizeof(uint16_t), 0, g.hidden * g.experts},
                Phase{2 * (g.local * bytes(g.routed, g.hidden, *gate) + bytes(g.shared, g.hidden, *shared)),
                      2 * (routed + shared_work), 0},
                Phase{g.local * bytes(g.hidden, g.routed, *down) + bytes(g.hidden, g.shared, *shared_down),
                      routed + shared_work, 0},
            };
            for (const auto& phase : phases)
                std::printf(",%zu,%zu,%zu", phase.bytes, phase.int_macs, phase.fp32_macs);
            std::putchar('\n');
        }
    }

    /**
     * @brief Emit reproducible work coordinates for the 15 MI50 families and the additional QFN model tuples.
     * @param argc Original probe command argument count, including the work-mode selector.
     * @param argv Original positive shape arguments and possibly zero local routed expert count.
     * @return Process exit status, with invalid shape arguments rejected before any GPU operation.
     */
    int cudaHCFFNRooflineWork(int argc, char** argv)
    {
        if (argc != 9)
        {
            std::fprintf(stderr, "Usage: %s hc-ffn-work H R F S E top-k local-routes\n", argv[0]);
            return EXIT_FAILURE;
        }
        std::array<size_t, 7> values{};
        for (size_t i = 0; i < values.size(); ++i)
        {
            const std::string_view argument(argv[i + 2]);
            const auto result = std::from_chars(argument.data(), argument.data() + argument.size(), values[i]);
            if (result.ec != std::errc{} || result.ptr != argument.data() + argument.size() ||
                (!values[i] && i != values.size() - 1))
            {
                std::fprintf(stderr, "Invalid HC/FFN geometry\n");
                return EXIT_FAILURE;
            }
        }
        const auto [h, r, f, s, e, selected, local] = values;
        constexpr auto max = std::numeric_limits<size_t>::max();
        // Each output field is bounded by the sum of three projection roles
        // with at most four prepared bytes per logical matrix element.
        if ((h | r | f | s) % 32 || local > selected || selected > e ||
            h > max / 16 / r || h > max / 16 / e || h > max / 16 / s ||
            h > max / 16 / f / (local ? local : 1))
        {
            std::fprintf(stderr, "HC/FFN geometry is incomplete or exceeds the counter representation\n");
            return EXIT_FAILURE;
        }

        const Geometry geometry{h, r, f, s, e, selected, local};
        std::puts("family,H,R,F,S,E,top_k,local_routes,gate,down,shared_gate,shared_down,source_geometry,"
                  "hc_down_bytes,hc_down_int_macs,hc_down_fp32_macs,hc_up_bytes,hc_up_int_macs,hc_up_fp32_macs,"
                  "router_bytes,router_int_macs,router_fp32_macs,gate_up_bytes,gate_up_int_macs,gate_up_fp32_macs,"
                  "down_bytes,down_int_macs,down_fp32_macs");
        for (const auto& recipe : fullNativeHCFFNFamilyCases())
            printFamily(recipe, geometry);
        constexpr std::array model_cases{
            FullNativeHCFFNFamilyCase{0, "QFN_Q4_K_DownQ8_0", "Q4_K", "Q8_0", "Q8_0"},
            FullNativeHCFFNFamilyCase{0, "QFN_Q5_K_DownQ8_0", "Q5_K", "Q8_0", "Q8_0"},
            FullNativeHCFFNFamilyCase{0, "QFN_Drafter_Q8_0", "Q8_0", "Q8_0", "Q8_0"},
        };
        for (const auto& recipe : model_cases)
            printFamily(recipe, geometry);
        return EXIT_SUCCESS;
    }
}
