/**
 * @file MoEOverlayServiceTelemetryCoverageTest.h
 * @brief Shared CUDA/ROCm proof for stratified MoE service markers.
 *
 * Dynamic ExpertOverlay retains a complete layer-indexed telemetry ABI, but
 * graph replay should pay timing-kernel overhead for only a deterministic
 * sublinear sample of each exact weight equivalence class. This helper builds
 * nine equivalent floating-point layers on a real backend and proves the three
 * ordinal strata bind storage while every unsampled peer remains marker-free.
 */

#pragma once

#include "execution/moe/MoEOverlayEconomyCalibrationPlanner.h"
#include "execution/moe/MoERuntimeTable.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Exercise representative telemetry selection on one live GPU.
     * @param device Available CUDA or ROCm device selected by the caller.
     */
    inline void verifyStratifiedMoEOverlayServiceTelemetryCoverage(
        DeviceId device)
    {
        const auto projection = [](
                                    ExpertTierWeightProjection role,
                                    int n,
                                    int k)
        {
            return MoEOverlayProjectionWeightManifest{
                .projection = role,
                .N = n,
                .K = k,
                .format = ExpertWeightFormat::floating(TensorType::FP16),
            };
        };
        const auto layer = [&projection](int layer_idx)
        {
            return MoEOverlayLayerWeightManifest{
                .layer_idx = layer_idx,
                .projections = {{
                    projection(ExpertTierWeightProjection::Gate, 64, 32),
                    projection(ExpertTierWeightProjection::Up, 64, 32),
                    projection(ExpertTierWeightProjection::Down, 32, 64),
                }},
            };
        };

        std::vector<MoEOverlayLayerWeightManifest> manifest;
        for (int layer_idx = 0; layer_idx < 9; ++layer_idx)
            manifest.push_back(layer(layer_idx));
        auto catalog = std::make_shared<
            MoEOverlayEconomyCalibrationLayerCatalog>(manifest);
        ASSERT_EQ(catalog->representativeLayers(), (std::vector<int>{0}));
        ASSERT_EQ(
            catalog->serviceTelemetryLayers(),
            (std::vector<int>{1, 4, 7}));

        DeviceMoERuntimeTable table({
            .device_id = device,
            .num_layers = 9,
            .num_experts = 4,
            .top_k = 2,
            .mirror_to_device = true,
            .overlay_service_telemetry_coverage =
                MoEOverlayServiceTelemetryCoverage::CatalogStratifiedSample,
            .overlay_service_telemetry_catalog = catalog,
        });

        for (int layer_idx = 0; layer_idx < 9; ++layer_idx)
        {
            const bool sampled =
                layer_idx == 1 || layer_idx == 4 || layer_idx == 7;
            EXPECT_EQ(
                table.collectsDeviceOverlayServiceTelemetryForLayer(layer_idx),
                sampled);
            EXPECT_EQ(
                table.deviceOverlayServiceTelemetryBinding(layer_idx).valid(),
                sampled);
            EXPECT_EQ(
                table.deviceOverlayServiceTelemetryBinding(layer_idx).empty(),
                !sampled);
        }
        EXPECT_NE(table.deviceOverlayServiceTelemetry(), nullptr)
            << "The complete publication ABI remains allocated even though "
               "only stratified graph stages enqueue marker kernels";
        EXPECT_NE(table.deviceOverlayServiceTelemetrySample(8), nullptr);
    }
} // namespace llaminar2::test
