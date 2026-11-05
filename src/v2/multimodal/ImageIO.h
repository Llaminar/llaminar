/**
 * @file ImageIO.h
 * @brief Host-side image decoding and Pillow-exact RGB8 geometry for vision preprocessing.
 *
 * Vision models are trained and served (vLLM, the official repositories) on
 * images decoded and resampled by Pillow. Patch values are a deterministic
 * function of the resampled RGB8 pixels, so reproducing Pillow's integer
 * resampling exactly keeps Llaminar's vision inputs byte-identical to the
 * production pipeline for losslessly coded images:
 *  - decoding uses stb_image (PNG/BMP/GIF/TGA exact; baseline/progressive
 *    JPEG decoded with stb's IDCT and chroma upsampling, which may differ by
 *    a few code values from Pillow's libjpeg-turbo);
 *  - convert("RGB") drops alpha and expands gray/palette, like Pillow;
 *  - resizeBicubic is a line-by-line port of Pillow 13 libImaging/Resample.c
 *    (separable 8-bit fixed-point convolution with 22 fractional bits, the
 *    horizontal/vertical pass order and source-row trimming included);
 *  - pad is ImageOps.pad (contain, Python round-half-even, centered paste).
 * Everything here is request admission work on the host; nothing touches a
 * device.
 */
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::multimodal
{
    /** @brief Packed 8-bit RGB image, row-major, 3 bytes per pixel. */
    struct RgbImage
    {
        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> rgb; ///< width * height * 3 bytes.

        /** @brief Byte offset of pixel (x, y). */
        [[nodiscard]] std::size_t offset(int x, int y) const noexcept
        {
            return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3;
        }
    };

    /**
     * @brief Decode an encoded image (PNG, JPEG, BMP, GIF, TGA) to RGB8 like Pillow's convert("RGB").
     * @param bytes Encoded file contents.
     * @param max_pixels Upper bound on width * height, rejected before decoding the payload.
     * @return Decoded image.
     * @throws std::invalid_argument For unsupported, malformed or oversized images.
     */
    RgbImage decodeImage(std::span<const std::uint8_t> bytes, std::uint64_t max_pixels);

    /**
     * @brief Decode standard base64 (RFC 4648, '=' padding, no whitespace tolerance beyond CR/LF).
     * @throws std::invalid_argument For malformed input.
     */
    std::vector<std::uint8_t> decodeBase64(std::string_view text);

    /**
     * @brief Payload bytes of a base64 ``data:`` URL.
     * @throws std::invalid_argument When @p url is not a base64 data URL.
     */
    std::vector<std::uint8_t> decodeDataUrl(std::string_view url);

    /**
     * @brief Pillow ``Image.resize(size, BICUBIC)`` of an RGB8 image (bit-exact port).
     * @details Returns a copy when the size is unchanged, exactly like Pillow.
     */
    RgbImage resizeBicubic(const RgbImage &image, int width, int height);

    /**
     * @brief Pillow ``ImageOps.pad(image, size, BICUBIC, color, centering=(0.5, 0.5))``.
     * @param fill RGB background colour.
     */
    RgbImage pad(const RgbImage &image, int width, int height, const std::uint8_t fill[3]);

    /** @brief Python ``round()`` (half to even) of a double, as used by Pillow's geometry. */
    long long pythonRound(double value);
} // namespace llaminar2::multimodal
