/** @brief Child process of the isolated helper launch chain.
 *
 * Executed BY the placement helper (verifying the real exec boundary).
 * Validates, with the strict module authority: the receipt transport parses;
 * the expected session/rank/budget fields are present in the transport;
 * OMP_PLACES equals the receipt's place string; OMP_NUM_THREADS equals the
 * expected budget. Exits 0 on full agreement, 1 with a stderr diagnostic
 * otherwise. Expected values arrive via LLAMINAR_HELPER_CHECK_* environment
 * variables set by the test parent (inherited through the helper exec).
 */
#include "planning/PlacementTopology.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main()
{
    const char *transport = std::getenv("LLAMINAR_PLACEMENT_RECEIPT_V1");
    if (!transport || !*transport)
    {
        std::fprintf(stderr, "[child-check] receipt transport missing\n");
        return 1;
    }
    llaminar2::placement::PlacementTopologyReject reject(
        llaminar2::placement::RejectReason::MalformedInput, "unset");
    const auto receipt = llaminar2::placement::parseReceiptTransport(transport, reject);
    if (!receipt.has_value())
    {
        std::fprintf(stderr, "[child-check] receipt rejected: %s\n", reject.what());
        return 1;
    }
    const std::string transport_string = transport;
    for (const char *field : {"budget", "session", "rank"})
    {
        const std::string key = "LLAMINAR_HELPER_CHECK_" + std::string(field);
        const char *value = std::getenv(key.c_str());
        // budget expectation is OPTIONAL: absent means the parent could not
        // know an AutoLocal budget a priori; the child then verifies the
        // AutoLocal origin marker instead of the exact number.
        if (value == nullptr && std::string(field) == "budget")
        {
            if (transport_string.find(";budget_origin=AutoLocal;") == std::string::npos)
            {
                std::fprintf(stderr, "[child-check] expected AutoLocal origin\n");
                return 1;
            }
            continue;
        }
        if (value == nullptr || *value == '\0')
        {
            std::fprintf(stderr, "[child-check] missing expectation for %s\n", field);
            return 1;
        }
        const std::string needle = std::string(";") + field + "=" + value + ";";
        if (transport_string.find(needle) == std::string::npos)
        {
            std::fprintf(stderr, "[child-check] transport %s mismatch\n", field);
            return 1;
        }
    }
    if (const char *places = std::getenv("OMP_PLACES"))
    {
        if (places != receipt->omp_places_string())
        {
            std::fprintf(stderr, "[child-check] OMP_PLACES != receipt places\n");
            return 1;
        }
    }
    else
    {
        std::fprintf(stderr, "[child-check] OMP_PLACES missing\n");
        return 1;
    }
    const char *threads = std::getenv("OMP_NUM_THREADS");
    if (!threads || *threads == '\0')
    {
        std::fprintf(stderr, "[child-check] OMP_NUM_THREADS missing\n");
        return 1;
    }
    if (const char *expected_budget = std::getenv("LLAMINAR_HELPER_CHECK_budget"))
        if (std::strcmp(threads, expected_budget) != 0)
        {
            std::fprintf(stderr, "[child-check] OMP_NUM_THREADS mismatch\n");
            return 1;
        }
    return 0;
}
