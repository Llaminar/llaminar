/**
 * @file PromptImageEncoderFactory.h
 * @brief Startup construction of the request image pipeline selected by --mmproj.
 */
#pragma once

#include "../multimodal/PromptImageEncoder.h"

#include <memory>
#include <string>

namespace llaminar2
{
class IMPIContext;

/**
 * @brief Load the vision projector for the served text model.
 * @param mmproj_path ``--mmproj``; empty yields null (text-only serving).
 * @param model_path Text model GGUF; its metadata must admit image sentinels.
 * @throws std::invalid_argument When the pair is not a supported vision checkpoint.
 */
std::shared_ptr<const multimodal::IPromptImageEncoder> createPromptImageEncoder(const std::string &mmproj_path,
                                                                                const std::string &model_path,
                                                                                const IMPIContext &mpi);
} // namespace llaminar2
