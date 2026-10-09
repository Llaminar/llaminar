/**
 * @file ControllerMovementTransportFixture.cpp
 * @brief Export actual C++ movement telemetry for the independent Python consumer.
 *
 * Four immutable metadata waves exercise exact large IDs, different GPU vendors,
 * real versus estimated bytes and a final transaction after the HTTP cutoff.
 * No backend is initialized and no payload, model or accelerator is accessed.
 */
#include "ControllerMovementFixture.h"
#include "app/modes/MoEMovementTransportJson.h"
#include <cstdlib>
#include <iostream>

/** @brief Emit production counter/terminal projections of device-free fixture metadata. */
int main()
{
    using namespace llaminar2;
    ::setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
    ::unsetenv("LLAMINAR_PERF_STATS_FILTER");
    PerfStatsCollector::reset();
    MoEOptimizationMovementLedger complete;
    nlohmann::json earlier;
    for (unsigned wave = 0; wave < 4; ++wave)
    {
        auto completed = test::controllerMovementFixture((std::uint64_t{1} << 54) + wave + 1, wave + 9, wave);
        if (wave == 0)
            completed.device_publications.front().physical_payload_bytes = (std::uint64_t{1} << 54) + 1;
        recordMoEControllerCompletedMovement(completed.device_publications.front(), completed.edges, "overlay");
        complete.edges.insert(complete.edges.end(), completed.edges.begin(), completed.edges.end());
        complete.economy.push_back(completed.economy.front());
        complete.device_publications.push_back(completed.device_publications.front());
        if (wave == 1)
            earlier = moeMovementLedgerJson(complete);
    }
    const auto observed = nlohmann::json::parse(PerfStatsCollector::jsonString()).at("records");
    auto records = nlohmann::json::array();
    auto terminal = nlohmann::json::array();
    for (int rank = 0; rank < 2; ++rank)
    {
        for (auto record : observed)
        {
            record["rank"] = rank;
            records.push_back(std::move(record));
        }
        if (rank != 0)
            complete.economy.clear(); // Only the leader owns placement economics.
        auto history = moeMovementTransportJson(complete);
        history["rank"] = rank;
        terminal.push_back(std::move(history));
    }
    std::cout << nlohmann::json{{"records", records}, {"terminal_movement", terminal}, {"http_movement", earlier}}.dump() << '\n';
}
