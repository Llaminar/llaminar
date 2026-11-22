/**
 * @file PromptImageEncoderFactory.cpp
 * @brief Validate the text/vision GGUF pair from metadata only, then load the host vision tower.
 */
#include "PromptImageEncoderFactory.h"

#include "../models/deepseek4/DeepSeekV4PromptImages.h"
#include "../planning/PlanningModelMetadata.h"
#include "../utils/Logger.h"

#include <stdexcept>

namespace llaminar2
{
std::shared_ptr<const multimodal::IPromptImageEncoder> createPromptImageEncoder(const std::string &mmproj_path,
                                                                                const std::string &model_path,
                                                                                const IMPIContext &mpi)
{
    if (mmproj_path.empty()) return nullptr;
    // Metadata only: no weight payload of the text model is touched.
    const auto metadata = readPlanningModelMetadata(model_path);
    const auto &profile = metadata.memoryProfile();
    if (profile.architecture != "deepseek4")
        throw std::invalid_argument("--mmproj is implemented for DeepSeek-V4 vision checkpoints only, not " +
                                    profile.architecture);
    if (!profile.deepseek_v4 || profile.deepseek_v4->image_bank_rows == 0)
        throw std::invalid_argument("--mmproj requires a DeepSeek-V4 vision checkpoint (exp_probs_b_vl on every layer)");
    std::shared_ptr<const deepseek4::DeepSeekV4VisionEncoder> tower =
        deepseek4::DeepSeekV4VisionEncoder::load(mmproj_path, mpi, profile.d_model);
    // Fail-closed exactness check: the metadata-derived byte counter that the
    // physical-memory envelope admitted must equal the loaded tower, and the
    // metadata-only workspace bound must match the loaded encoder's closed form.
    const std::size_t expected_weights =
        deepseek4::DeepSeekV4VisionEncoder::mmprojHostWeightBytes(mmproj_path, mpi);
    if (tower->weightBytes() != expected_weights)
        throw std::runtime_error("DeepSeek-V4 vision tower bytes disagree with the admitted mmproj metadata");
    const std::size_t expected_workspace = deepseek4::visionTowerWorkspaceBytes(mmproj_path, mpi, profile.d_model);
    if (expected_workspace != tower->encodeWorkspaceBytes())
        throw std::runtime_error("DeepSeek-V4 vision tower workspace disagrees with the admitted metadata");
    LOG_INFO("[Vision] Loaded " << mmproj_path << " (" << (tower->weightBytes() >> 20) << " MiB host FP32)"
             << " admitted workspace " << (expected_workspace >> 20) << " MiB");
    return std::make_shared<deepseek4::DeepSeekV4PromptImageEncoder>(std::move(tower));
}
} // namespace llaminar2
