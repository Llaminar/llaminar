/**
 * @file ResidualBlockRead.h
 * @brief Single-use read/write pairing for widened residual trunks.
 *
 * A residual trunk (hc copies of the hidden state) is read before each block
 * and written after it. Two mixer families share the QwenGraphBase hooks:
 * Qwen4Exp's low-rank gated read (Qwen4ExpHyperconnectionBuilder::BlockRead)
 * and DeepSeek-V4's Sinkhorn mixer (DeepSeekV4BlockRead). The capability is
 * move-only and consumed exactly once by the paired write.
 */
#pragma once

#include "../../backends/DeviceId.h"
#include "../../execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace llaminar2
{
/**
 * @brief DeepSeek-V4 Sinkhorn read capability: the read already published
 *        post/comb into admitted scratch; the write consumes them.
 */
class DeepSeekV4BlockRead final
{
public:
    DeepSeekV4BlockRead(std::string terminal, std::string prefix, DeviceId device, Qwen4ExpStageRows rows)
        : terminal_(std::move(terminal)), prefix_(std::move(prefix)), device_(device), rows_(std::move(rows)) {}
    DeepSeekV4BlockRead(DeepSeekV4BlockRead &&) noexcept = default;
    DeepSeekV4BlockRead(const DeepSeekV4BlockRead &) = delete;
    DeepSeekV4BlockRead &operator=(const DeepSeekV4BlockRead &) = delete;
    DeepSeekV4BlockRead &operator=(DeepSeekV4BlockRead &&) = delete;
    /** @brief Producer of the normalized block input. */
    const std::string &terminal() const noexcept { return terminal_; }
    /** @brief Stage-name prefix of the paired read (e.g. "layer3_hc_attn_"). */
    const std::string &prefix() const noexcept { return prefix_; }
    /** @brief Exact participant. */
    DeviceId device() const noexcept { return device_; }
    /** @brief Admitted rows and optional live-row count of the read. */
    const Qwen4ExpStageRows &rows() const noexcept { return rows_; }

private:
    std::string terminal_;
    std::string prefix_;
    DeviceId device_;
    Qwen4ExpStageRows rows_;
};

/** @brief One residual-trunk read capability of either mixer family. */
class ResidualBlockRead final
{
public:
    explicit ResidualBlockRead(Qwen4ExpHyperconnectionBuilder::BlockRead &&read) : value_(std::move(read)) {}
    explicit ResidualBlockRead(DeepSeekV4BlockRead &&read) : value_(std::move(read)) {}
    ResidualBlockRead(ResidualBlockRead &&) noexcept = default;
    ResidualBlockRead(const ResidualBlockRead &) = delete;
    ResidualBlockRead &operator=(const ResidualBlockRead &) = delete;
    ResidualBlockRead &operator=(ResidualBlockRead &&) = delete;
    /** @brief Producer of the compute-width block input. */
    const std::string &terminal() const noexcept
    {
        return std::visit([](const auto &read) -> const std::string & { return read.terminal(); }, value_);
    }
    /** @brief Mutable access for the paired write. */
    std::variant<Qwen4ExpHyperconnectionBuilder::BlockRead, DeepSeekV4BlockRead> &value() noexcept { return value_; }

private:
    std::variant<Qwen4ExpHyperconnectionBuilder::BlockRead, DeepSeekV4BlockRead> value_;
};
} // namespace llaminar2
