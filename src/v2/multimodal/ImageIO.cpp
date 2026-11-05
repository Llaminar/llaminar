/**
 * @file ImageIO.cpp
 * @brief stb_image decoding, base64 and the Pillow 13 bicubic resampling port.
 *
 * Resampling follows libImaging/Resample.c (Pillow 13.0.0) exactly:
 *  - precompute_coeffs: per output pixel, center = in0 + (xx + 0.5) * scale,
 *    support = 2 * max(scale, 1), bounds by truncating center -/+ support + 0.5,
 *    weights filter((x + xmin - center + 0.5) / filterscale) normalized to sum 1;
 *  - normalize_coeffs_8bpc: weights to INT32 with PRECISION_BITS = 22,
 *    rounding half away from zero;
 *  - each output byte = clip8(2^21 + sum(in * k)) with clip8 = clamp(v >> 22);
 *  - ImagingResampleInner: horizontal first unless the height shrinks by more
 *    than twice the width; the horizontal pass covers only the source rows
 *    the vertical pass reads (ybox_first .. ybox_last).
 * The bicubic kernel uses a = -0.5 and support 2.
 */
#include "ImageIO.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <climits>
#include <cmath>
#include <stdexcept>

#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace llaminar2::multimodal
{
    namespace
    {
        constexpr int kPrecisionBits = 32 - 8 - 2;

        double bicubicFilter(double x)
        {
            constexpr double a = -0.5;
            if (x < 0.0) x = -x;
            if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
            return 0.0;
        }

        std::uint8_t clip8(int in)
        {
            const int v = in >> kPrecisionBits; // arithmetic shift, as Pillow's table index
            return static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }

        /** @brief precompute_coeffs + normalize_coeffs_8bpc. */
        struct Coefficients
        {
            int ksize = 0;
            std::vector<int> bounds;  ///< [out * 2] (xmin, count)
            std::vector<std::int32_t> kk; ///< [out * ksize]
        };

        Coefficients precompute(int in_size, float in0, float in1, int out_size)
        {
            constexpr double kSupport = 2.0;
            double scale = static_cast<double>(in1 - in0) / out_size;
            double filterscale = scale < 1.0 ? 1.0 : scale;
            const double support = kSupport * filterscale;
            Coefficients c;
            c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
            if (out_size > INT_MAX / (c.ksize * static_cast<int>(sizeof(double))))
                throw std::invalid_argument("image resize coefficient table overflows");
            std::vector<double> prekk(static_cast<std::size_t>(out_size) * c.ksize, 0.0);
            c.bounds.assign(static_cast<std::size_t>(out_size) * 2, 0);
            const double inv_filterscale = 1.0 / filterscale;
            for (int xx = 0; xx < out_size; ++xx)
            {
                const double center = in0 + (xx + 0.5) * scale;
                double ww = 0.0;
                int xmin = static_cast<int>(center - support + 0.5);
                if (xmin < 0) xmin = 0;
                int xmax = static_cast<int>(center + support + 0.5);
                if (xmax > in_size) xmax = in_size;
                xmax -= xmin;
                double *k = &prekk[static_cast<std::size_t>(xx) * c.ksize];
                int x = 0;
                for (; x < xmax; ++x)
                {
                    const double w = bicubicFilter((x + xmin - center + 0.5) * inv_filterscale);
                    k[x] = w;
                    ww += w;
                }
                if (ww != 0.0)
                    for (x = 0; x < xmax; ++x) k[x] /= ww;
                c.bounds[xx * 2 + 0] = xmin;
                c.bounds[xx * 2 + 1] = xmax;
            }
            c.kk.resize(prekk.size());
            for (std::size_t i = 0; i < prekk.size(); ++i)
                c.kk[i] = prekk[i] < 0 ? static_cast<int>(-0.5 + prekk[i] * (1 << kPrecisionBits))
                                       : static_cast<int>(0.5 + prekk[i] * (1 << kPrecisionBits));
            return c;
        }

        RgbImage horizontal(const RgbImage &in, int out_w, int out_h, int offset, const Coefficients &c)
        {
            RgbImage out{out_w, out_h, std::vector<std::uint8_t>(static_cast<std::size_t>(out_w) * out_h * 3)};
            for (int yy = 0; yy < out_h; ++yy)
                for (int xx = 0; xx < out_w; ++xx)
                {
                    const int xmin = c.bounds[xx * 2 + 0], xmax = c.bounds[xx * 2 + 1];
                    const std::int32_t *k = &c.kk[static_cast<std::size_t>(xx) * c.ksize];
                    int ss0 = 1 << (kPrecisionBits - 1), ss1 = ss0, ss2 = ss0;
                    for (int x = 0; x < xmax; ++x)
                    {
                        const std::uint8_t *p = &in.rgb[in.offset(x + xmin, yy + offset)];
                        ss0 += p[0] * k[x];
                        ss1 += p[1] * k[x];
                        ss2 += p[2] * k[x];
                    }
                    std::uint8_t *o = &out.rgb[out.offset(xx, yy)];
                    o[0] = clip8(ss0), o[1] = clip8(ss1), o[2] = clip8(ss2);
                }
            return out;
        }

        RgbImage vertical(const RgbImage &in, int out_w, int out_h, const Coefficients &c)
        {
            RgbImage out{out_w, out_h, std::vector<std::uint8_t>(static_cast<std::size_t>(out_w) * out_h * 3)};
            for (int yy = 0; yy < out_h; ++yy)
            {
                const std::int32_t *k = &c.kk[static_cast<std::size_t>(yy) * c.ksize];
                const int ymin = c.bounds[yy * 2 + 0], ymax = c.bounds[yy * 2 + 1];
                for (int xx = 0; xx < out_w; ++xx)
                {
                    int ss0 = 1 << (kPrecisionBits - 1), ss1 = ss0, ss2 = ss0;
                    for (int y = 0; y < ymax; ++y)
                    {
                        const std::uint8_t *p = &in.rgb[in.offset(xx, y + ymin)];
                        ss0 += p[0] * k[y];
                        ss1 += p[1] * k[y];
                        ss2 += p[2] * k[y];
                    }
                    std::uint8_t *o = &out.rgb[out.offset(xx, yy)];
                    o[0] = clip8(ss0), o[1] = clip8(ss1), o[2] = clip8(ss2);
                }
            }
            return out;
        }

        int base64Value(char c)
        {
            if (c >= 'A' && c <= 'Z') return c - 'A';
            if (c >= 'a' && c <= 'z') return c - 'a' + 26;
            if (c >= '0' && c <= '9') return c - '0' + 52;
            if (c == '+') return 62;
            if (c == '/') return 63;
            return -1;
        }
    } // namespace

    long long pythonRound(double value)
    {
        // Python's round() is round-half-to-even; nearbyint honours the
        // default FE_TONEAREST mode, which is exactly that.
        const int saved = std::fegetround();
        std::fesetround(FE_TONEAREST);
        const double r = std::nearbyint(value);
        std::fesetround(saved);
        return static_cast<long long>(r);
    }

    RgbImage decodeImage(std::span<const std::uint8_t> bytes, std::uint64_t max_pixels)
    {
        if (bytes.empty() || bytes.size() > static_cast<std::size_t>(INT_MAX))
            throw std::invalid_argument("image payload is empty or too large");
        int w = 0, h = 0, channels = 0;
        const auto *data = reinterpret_cast<const stbi_uc *>(bytes.data());
        const int len = static_cast<int>(bytes.size());
        if (!stbi_info_from_memory(data, len, &w, &h, &channels))
            throw std::invalid_argument(std::string("unsupported or malformed image: ") + stbi_failure_reason());
        if (w <= 0 || h <= 0 || static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h) > max_pixels)
            throw std::invalid_argument("image dimensions exceed the admitted pixel budget");
        stbi_uc *pixels = stbi_load_from_memory(data, len, &w, &h, &channels, 3);
        if (!pixels)
            throw std::invalid_argument(std::string("image decoding failed: ") + stbi_failure_reason());
        RgbImage image{w, h, std::vector<std::uint8_t>(pixels, pixels + static_cast<std::size_t>(w) * h * 3)};
        stbi_image_free(pixels);
        return image;
    }

    std::vector<std::uint8_t> decodeBase64(std::string_view text)
    {
        std::vector<std::uint8_t> out;
        out.reserve(text.size() / 4 * 3);
        std::uint32_t acc = 0;
        int bits = 0, padding = 0;
        for (const char c : text)
        {
            if (c == '\r' || c == '\n') continue;
            if (c == '=')
            {
                ++padding;
                continue;
            }
            const int v = base64Value(c);
            if (v < 0 || padding)
                throw std::invalid_argument("malformed base64 image payload");
            acc = (acc << 6) | static_cast<std::uint32_t>(v);
            bits += 6;
            if (bits >= 8)
            {
                bits -= 8;
                out.push_back(static_cast<std::uint8_t>((acc >> bits) & 0xFF));
            }
        }
        if (padding > 2 || bits >= 6)
            throw std::invalid_argument("malformed base64 image payload");
        return out;
    }

    std::vector<std::uint8_t> decodeDataUrl(std::string_view url)
    {
        if (url.substr(0, 5) != "data:")
            throw std::invalid_argument("image URLs must be base64 data: URLs");
        const auto comma = url.find(',');
        if (comma == std::string_view::npos)
            throw std::invalid_argument("malformed data: URL");
        const auto header = url.substr(5, comma - 5);
        if (header.size() < 7 || header.substr(header.size() - 7) != ";base64")
            throw std::invalid_argument("image data: URLs must be base64 encoded");
        return decodeBase64(url.substr(comma + 1));
    }

    RgbImage resizeBicubic(const RgbImage &in, int xsize, int ysize)
    {
        if (xsize <= 0 || ysize <= 0 || in.width <= 0 || in.height <= 0)
            throw std::invalid_argument("image resize requires positive dimensions");
        if (xsize == in.width && ysize == in.height)
            return in; // Pillow returns a copy for an unchanged size
        const float box[4] = {0.0f, 0.0f, static_cast<float>(in.width), static_cast<float>(in.height)};
        const bool need_horizontal = xsize != in.width || box[0] != 0.0f || box[2] != xsize;
        const bool need_vertical = ysize != in.height || box[1] != 0.0f || box[3] != ysize;
        const bool horizontal_first = !((in.height - ysize) > 0 && (in.height - ysize) > (in.width - xsize) * 2);

        Coefficients vert, horiz;
        if ((need_horizontal && horizontal_first) || need_vertical)
            vert = precompute(in.height, box[1], box[3], ysize);
        int ybox_first = 0, ybox_last = 0;
        if (need_horizontal)
        {
            if (horizontal_first)
            {
                ybox_first = vert.bounds[0];
                ybox_last = vert.bounds[ysize * 2 - 2] + vert.bounds[ysize * 2 - 1];
                if (ybox_first != 0)
                    for (int i = 0; i < ysize; ++i) vert.bounds[i * 2] -= ybox_first;
            }
            horiz = precompute(in.width, box[0], box[2], xsize);
        }

        RgbImage cur = in;
        if (horizontal_first)
        {
            if (need_horizontal) cur = horizontal(cur, xsize, ybox_last - ybox_first, ybox_first, horiz);
            if (need_vertical) cur = vertical(cur, xsize, ysize, vert);
        }
        else
        {
            if (need_vertical) cur = vertical(cur, cur.width, ysize, vert);
            if (need_horizontal) cur = horizontal(cur, xsize, ysize, 0, horiz);
        }
        return cur;
    }

    RgbImage pad(const RgbImage &image, int width, int height, const std::uint8_t fill[3])
    {
        if (width <= 0 || height <= 0 || image.width <= 0 || image.height <= 0)
            throw std::invalid_argument("image pad requires positive dimensions");
        // ImageOps.contain
        const double im_ratio = static_cast<double>(image.width) / image.height;
        const double dest_ratio = static_cast<double>(width) / height;
        int cw = width, ch = height;
        if (im_ratio != dest_ratio)
        {
            if (im_ratio > dest_ratio)
            {
                const auto new_height = pythonRound(static_cast<double>(image.height) / image.width * width);
                if (new_height != height) ch = static_cast<int>(new_height);
            }
            else
            {
                const auto new_width = pythonRound(static_cast<double>(image.width) / image.height * height);
                if (new_width != width) cw = static_cast<int>(new_width);
            }
        }
        if (cw <= 0 || ch <= 0)
            throw std::invalid_argument("image pad produced an empty resized image");
        RgbImage resized = resizeBicubic(image, cw, ch);
        if (resized.width == width && resized.height == height)
            return resized;
        RgbImage out{width, height, std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height * 3)};
        for (std::size_t i = 0; i < out.rgb.size(); i += 3)
            out.rgb[i] = fill[0], out.rgb[i + 1] = fill[1], out.rgb[i + 2] = fill[2];
        int ox = 0, oy = 0;
        if (resized.width != width) ox = static_cast<int>(pythonRound((width - resized.width) * 0.5));
        else oy = static_cast<int>(pythonRound((height - resized.height) * 0.5));
        for (int y = 0; y < resized.height; ++y)
            std::copy_n(&resized.rgb[resized.offset(0, y)], static_cast<std::size_t>(resized.width) * 3,
                        &out.rgb[out.offset(ox, y + oy)]);
        return out;
    }
} // namespace llaminar2::multimodal
