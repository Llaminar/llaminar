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
    LOG_INFO("[Vision] Loaded " << mmproj_path << " (" << (tower->weightBytes() >> 20) << " MiB host FP32)");
    return std::make_shared<deepseek4::DeepSeekV4PromptImageEncoder>(std::move(tower));
}
} // namespace llaminar2
