/**
 * @file GpuResidualAddTailHarness.h
 * @brief Actual retained GPU residual kernels on odd and partial flat extents.
 *
 * The public FP32 kernel API processes grouped rows and independently aligned
 * M1 row operands. Named canonical workspace owns all storage before recording;
 * no test kernel, allocation, host transfer or rebind occurs inside capture.
 * Both retained arms consume freshly published inputs and distinct poison.
 */
#pragma once

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "utils/ResidualAddTailCases.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace llaminar2::test::residual_tail
{
/**
 * @brief Reject an unsuccessful native lifecycle operation.
 * @param result Exact requested operation's success value.
 * @param operation Stable failure diagnostic.
 * @throws std::runtime_error on failure.
 */
inline void require(bool result, const char *operation)
{
    if (!result) throw std::runtime_error(operation);
}

/** @brief Last-declared guard retires DMA before host, graphs and workspace. */
struct ObservationJoin
{
    IBackend *backend;
    DeviceId device;
    void *stream;
    /** @brief Join submitted work even when an assertion/exception unwinds. */
    ~ObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
};

/**
 * @brief Prove exact tail values and bounds through changed retained replays.
 * @tparam Kernel Production FP32 residual kernel specialization.
 * @param device Explicit endpoint owning workspace, graphs and exact stream.
 * @throws std::runtime_error if a required native lifecycle operation fails.
 */
template <typename Kernel>
void runGPU(DeviceId device)
{
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        auto *backend = getBackendFor(device);
        void *stream = context.defaultStream();
        require(backend != nullptr && stream != nullptr, "Residual tail endpoint");
        Kernel kernel(&context);
        kernel.setGPUStream(stream);
        for (const size_t cols : widths)
        {
            SCOPED_TRACE("cols=" + std::to_string(cols));
            const size_t capacity = rows.back() * cols;
            WorkspaceRequirements requirements;
            requirements.buffers = {
                {"tail_lhs", (capacity + guards) * sizeof(float), 256, true},
                {"tail_rhs", (capacity + guards) * sizeof(float), 256, true},
                {"tail_grouped", (capacity + guards) * sizeof(float), 256, true},
                {"tail_serial", (capacity + guards) * sizeof(float), 256, true},
                {"tail_row_lhs", (cols + guards) * sizeof(float), 256, true},
                {"tail_row_rhs", (cols + guards) * sizeof(float), 256, true},
                {"tail_row_output", (cols + guards) * sizeof(float), 256, true}};
            DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
            require(workspace.allocate(requirements), "Residual tail workspace");
            auto buffer = [&](const char *name)
            {
                return static_cast<float *>(workspace.getBuffer(name));
            };
            float *lhs = buffer("tail_lhs"), *rhs = buffer("tail_rhs");
            float *grouped = buffer("tail_grouped"), *serial = buffer("tail_serial");
            float *row_lhs = buffer("tail_row_lhs"), *row_rhs = buffer("tail_row_rhs");
            float *row_output = buffer("tail_row_output");
            require(lhs && rhs && grouped && serial && row_lhs && row_rhs && row_output,
                "Residual tail named buffers");
            std::vector<float> input, residual, expected;
            std::vector<float> grouped_host(capacity + guards), serial_host(capacity + guards);
            std::vector<float> row_host(cols + guards);
            std::array<std::unique_ptr<IGPUGraphCapture>, rows.size()> grouped_graphs;
            std::array<std::unique_ptr<IGPUGraphCapture>, rows.size()> serial_graphs;
            std::array<std::unique_ptr<IGPUGraphCapture>, rows.size()> alias_graphs;
            // Every borrowed host vector and graph is declared before the guard.
            // Capture RAII unwinds first; submitted copies join before retirement.
            ObservationJoin join{backend, device, stream};
            for (size_t slot = 0; slot < rows.size(); ++slot)
            {
                const size_t count = rows[slot];
                grouped_graphs[slot] = context.createGraphCapture(stream);
                serial_graphs[slot] = context.createGraphCapture(stream);
                alias_graphs[slot] = context.createGraphCapture(stream);
                require(grouped_graphs[slot] && serial_graphs[slot] && alias_graphs[slot], "Residual tail graph ownership");
                {
                    ScopedBackendGraphCapture recording(context, *grouped_graphs[slot], "Grouped residual tail");
                    require(recording.begin(), "Residual grouped capture begin");
                    require(kernel.apply(lhs, rhs, grouped, count * cols, nullptr, device.ordinal),
                        "Residual grouped captured kernel");
                    recording.finish();
                }
                require(grouped_graphs[slot]->instantiate(), "Residual grouped instantiate");
                {
                    ScopedBackendGraphCapture recording(context, *serial_graphs[slot], "Serial residual tail");
                    require(recording.begin(), "Residual serial capture begin");
                    for (size_t row = 0; row < count; ++row)
                    {
                        // Odd-width row offsets are not float4 aligned. Stage
                        // exactly one row into independently aligned buffers.
                        require(backend->deviceCopyAsync(row_lhs, lhs + row * cols,
                            cols * sizeof(float), device.ordinal, stream), "Residual serial lhs");
                        require(backend->deviceCopyAsync(row_rhs, rhs + row * cols,
                            cols * sizeof(float), device.ordinal, stream), "Residual serial rhs");
                        require(backend->memset(row_output, 0xfe, (cols + guards) * sizeof(float), device.ordinal, stream),
                            "Residual serial captured row poison");
                        require(kernel.apply(row_lhs, row_rhs, row_output, cols, nullptr, device.ordinal),
                            "Residual serial captured kernel");
                        require(backend->deviceCopyAsync(serial + row * cols, row_output,
                            cols * sizeof(float), device.ordinal, stream), "Residual serial output");
                    }
                    recording.finish();
                }
                require(serial_graphs[slot]->instantiate(), "Residual serial instantiate");
                {
                    ScopedBackendGraphCapture recording(context, *alias_graphs[slot], "In-place residual tail");
                    require(recording.begin(), "Residual alias capture begin");
                    require(kernel.apply(lhs, rhs, rhs, count * cols, nullptr, device.ordinal),
                        "Residual alias captured kernel");
                    recording.finish();
                }
                require(alias_graphs[slot]->instantiate(), "Residual alias instantiate");
            }
            for (size_t replay = 0; replay < 4; ++replay)
            {
                operands(capacity, replay, input, residual, expected);
                for (size_t count : replay_order)
                {
                    SCOPED_TRACE("M=" + std::to_string(count) + " replay=" + std::to_string(replay));
                    const size_t slot = static_cast<size_t>(std::find(rows.begin(), rows.end(), count) - rows.begin());
                    const size_t live = count * cols;
                    require(backend->hostToDevice(lhs, input.data(), live * sizeof(float), device.ordinal, stream),
                        "Residual changed lhs publication");
                    require(backend->hostToDevice(rhs, residual.data(), live * sizeof(float), device.ordinal, stream),
                        "Residual changed rhs publication");
                    require(backend->memset(grouped, 0xff, (capacity + guards) * sizeof(float), device.ordinal, stream),
                        "Residual grouped NaN poison");
                    require(grouped_graphs[slot]->launch(), "Residual grouped replay");
                    require(backend->deviceToHost(grouped_host.data(), grouped,
                        grouped_host.size() * sizeof(float), device.ordinal, stream), "Residual grouped observation");
                    require(backend->synchronizeStream(stream, device.ordinal), "Residual grouped terminal join");
                    expect(grouped_host, expected, live, 0xffffffffu);
                    ASSERT_FALSE(::testing::Test::HasFatalFailure());

                    // Row-output poison is distinct from the grouped arm and
                    // remains unchanged for every unused guard. Poison before
                    // each M1 call, inside capture, defeats prior-row leftovers.
                    require(backend->memset(serial, 0xfe, (capacity + guards) * sizeof(float), device.ordinal, stream),
                        "Residual serial distinct poison");
                    require(serial_graphs[slot]->launch(), "Residual serial replay");
                    require(backend->deviceToHost(serial_host.data(), serial,
                        serial_host.size() * sizeof(float), device.ordinal, stream), "Residual serial observation");
                    require(backend->deviceToHost(row_host.data(), row_output,
                        row_host.size() * sizeof(float), device.ordinal, stream), "Residual row guard observation");
                    require(backend->synchronizeStream(stream, device.ordinal), "Residual serial terminal join");
                    expect(serial_host, expected, live, 0xfefefefeu);
                    ASSERT_FALSE(::testing::Test::HasFatalFailure());
                    for (size_t i = 0; i < live; ++i)
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(grouped_host[i]),
                            std::bit_cast<std::uint32_t>(serial_host[i])) << "element=" << i;
                    for (size_t i = cols; i < row_host.size(); ++i)
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(row_host[i]), 0xfefefefeu);

                    // Restore the finite residual before the in-place arm;
                    // output poison must never replace a legitimate operand.
                    require(backend->memset(rhs, 0xfd, (capacity + guards) * sizeof(float), device.ordinal, stream),
                        "Residual alias guard poison");
                    require(backend->hostToDevice(rhs, residual.data(), live * sizeof(float), device.ordinal, stream),
                        "Residual alias live source restore");
                    require(alias_graphs[slot]->launch(), "Residual alias replay");
                    require(backend->deviceToHost(grouped_host.data(), rhs,
                        grouped_host.size() * sizeof(float), device.ordinal, stream), "Residual alias observation");
                    require(backend->synchronizeStream(stream, device.ordinal), "Residual alias terminal join");
                    for (size_t i = 0; i < grouped_host.size(); ++i)
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(grouped_host[i]), i < live
                            ? std::bit_cast<std::uint32_t>(expected[i]) : 0xfdfdfdfdu) << "element=" << i;
                }
            }
        }
    });
}
} // namespace llaminar2::test::residual_tail
