/**
 * @file ThinkingMode.h
 * @brief Request reasoning mode used to select model-owned generation defaults.
 *
 * Thinking and direct-answer requests can require different sampling laws even
 * when they share one weight architecture. Keep that choice explicit at the
 * sampling-policy boundary; an architecture name alone cannot select the law.
 */
#pragma once

namespace llaminar2
{
    /** @brief Whether the request asks the chat template to open reasoning. */
    enum class ThinkingMode
    {
        Enabled,
        Disabled,
        ModelDefault, ///< Use the loaded revision's documented default mode.
    };
}
