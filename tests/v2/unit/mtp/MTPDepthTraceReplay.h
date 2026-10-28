/**
 * @file MTPDepthTraceReplay.h
 * @brief Strict device-free execution of actual recorded controller inputs.
 *
 * This test/tool boundary neither samples a model nor invents feedback after a
 * candidate chooses a different depth. Source identities are compiled into the
 * producer and this interpreter; companion receipts retain capture provenance.
 */
#pragma once
#include "kernels/common/SamplingMath.h"
#include <nlohmann/json.hpp>
#include <string_view>
namespace llaminar2::test::mtp_trace_replay
{
/** @return This executable's actual compiled source/table identity. */
nlohmann::json compiledIdentity();
/** @brief Parse strict JSON with duplicate-key rejection.
 * @param text Exact request bytes. @return Parsed request; semantic checks occur in replay.
 * @throws std::invalid_argument On duplicate keys or malformed JSON.
 */
nlohmann::json parseDocument(std::string_view text);
/** @brief Fully replay one recorded baseline with its actual production implementation.
 * @param request Complete normalized immutable trace and independent terminal witnesses.
 * @return Exact per-transaction decisions and derived windows, never an economy claim.
 * A completed resident result also exposes controller_terminal_projection as
 * explicit owned ABI word/value pairs. Candidate intent does not require this
 * endpoint to equal the origin; recorded regressions may assert that separately.
 * Resident chronology also exposes actual word26 as promotion_streak: it owns
 * measured-path proposal confirmation, or legacy promotion hysteresis otherwise.
 * Host chronology exposes the same field only when its actual typed measured
 * owner is active. A deferred target may differ from current without a selector
 * change; word39 and word26 remain in the complete53-word terminal projection.
 * Unsupported prefixes expose no final projection and consume no later feedback.
 * @throws std::invalid_argument On identity, input, decision or terminal mismatch.
 */
nlohmann::json replay(const nlohmann::json &request);
/** @brief Probe another resident request policy only while its input depth remains observed.
 * @param request Baseline trace, fully validated first. @param candidate Explicit typed policy.
 * @return Complete same-path result or the first unsupported depth divergence; no future input consumed.
 * @throws std::invalid_argument On invalid baseline/candidate policy.
 */
nlohmann::json replayResidentCandidate(const nlohmann::json &request,
    const sampling_math::DeviceGenerationPolicy &candidate);
/** @brief Run only the explicit offline request form; no accelerator initialization.
 * @param argc Exact argument count. @param argv --replay-depth-trace and absolute request path.
 * @return Zero on complete replay, nonzero with a precise rejection diagnostic.
 */
int runCLI(int argc, char **argv);
}
