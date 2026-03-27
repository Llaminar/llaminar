/**
 * @file TurboQuantCodebook.h
 * @brief Lloyd-Max optimal codebook centroids for TurboQuant quantization
 * @author David Sanftenberg
 *
 * Provides pre-computed Lloyd-Max optimal codebook centroids for the standard
 * normal distribution N(0,1). These are used by TurboQuant to quantize each
 * rotated coordinate of a unit-norm vector.
 *
 * The centroids are optimal in the MSE sense for N(0,1). For the unit-sphere
 * marginal N(0, 1/d), the input is scaled by sqrt(d) before quantization
 * and the codebook values are divided by sqrt(d) during dequantization.
 *
 * Reference: Max (1960), "Quantizing for minimum distortion", IRE Trans. Info. Theory.
 *
 * The codebook provides:
 *   - Sorted centroid tables (for dequantization lookup)
 *   - Sorted threshold tables (for fast nearest-centroid via interval lookup)
 *   - Nearest-centroid functions (scalar + SIMD)
 *   - Lloyd-Max verification function (re-derives centroids from first principles)
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <array>
#include <algorithm>

namespace llaminar2
{

    // ========================================================================
    // 2-bit codebook (4 centroids for N(0,1))
    // ========================================================================

    inline constexpr std::array<float, 4> TQ2_CENTROIDS = {
        -1.510418f, -0.452780f, 0.452780f, 1.510418f};

    inline constexpr std::array<float, 3> TQ2_THRESHOLDS = {
        -0.981599f, 0.000000f, 0.981599f};

    // ========================================================================
    // 4-bit codebook (16 centroids for N(0,1))
    // ========================================================================

    /**
     * @brief Lloyd-Max optimal 16-level centroids for N(0,1), sorted ascending.
     *
     * Index 0 = most negative, index 15 = most positive.
     * Symmetric about 0: centroid[k] = -centroid[15-k].
     */
    inline constexpr std::array<float, 16> TQ4_CENTROIDS = {
        -2.732897f, -2.069364f, -1.618400f, -1.256565f,
        -0.942629f, -0.656982f, -0.388189f, -0.128443f,
        0.128443f, 0.388189f, 0.656982f, 0.942629f,
        1.256565f, 1.618400f, 2.069364f, 2.732897f};

    /**
     * @brief Decision thresholds for 4-bit quantization.
     *
     * 15 thresholds partition the real line into 16 intervals.
     * If x falls in interval [threshold[k-1], threshold[k]), assign index k.
     * threshold[-1] = -inf, threshold[15] = +inf.
     *
     * Each threshold = midpoint of adjacent centroids.
     */
    inline constexpr std::array<float, 15> TQ4_THRESHOLDS = {
        -2.401131f, -1.843882f, -1.437483f, -1.099597f,
        -0.799805f, -0.522585f, -0.258316f, 0.000000f,
        0.258316f, 0.522585f, 0.799805f, 1.099597f,
        1.437483f, 1.843882f, 2.401131f};

    // ========================================================================
    // 3-bit codebook (8 centroids for N(0,1))
    // ========================================================================

    /**
     * @brief Lloyd-Max optimal 8-level centroids for N(0,1), sorted ascending.
     *
     * Index 0 = most negative, index 7 = most positive.
     * Symmetric about 0: centroid[k] = -centroid[7-k].
     */
    inline constexpr std::array<float, 8> TQ3_CENTROIDS = {
        -2.151946f, -1.343909f, -0.756005f, -0.245094f,
        0.245094f, 0.756005f, 1.343909f, 2.151946f};

    /**
     * @brief Decision thresholds for 3-bit quantization.
     *
     * 7 thresholds partition the real line into 8 intervals.
     */
    inline constexpr std::array<float, 7> TQ3_THRESHOLDS = {
        -1.747928f, -1.049957f, -0.500550f, 0.000000f,
        0.500550f, 1.049957f, 1.747928f};

    // ========================================================================
    // 8-bit codebook (256 centroids for N(0,1))
    // ========================================================================

    /**
     * @brief 256-level Lloyd-Max optimal centroids for N(0,1).
     *
     * MSE = 1.32×10⁻⁴, SQNR = 38.79 dB.
     * Used for TQ8 K-projection quantization where score accuracy is critical.
     *
     * Generated via iterative Lloyd-Max optimization on 1M N(0,1) samples.
     */
    inline constexpr std::array<float, 256> TQ8_CENTROIDS = {
        -3.78920960f, -3.30915570f, -3.01834941f, -2.80658674f, -2.63340139f, -2.48449373f, -2.35561466f, -2.24269128f,
        -2.14211559f, -2.05341673f, -1.97122753f, -1.89669085f, -1.82880616f, -1.76743543f, -1.71260118f, -1.66326785f,
        -1.61619723f, -1.57234073f, -1.53127813f, -1.49353135f, -1.45867503f, -1.42624652f, -1.39503074f, -1.36621320f,
        -1.33910429f, -1.31285179f, -1.28799033f, -1.26373041f, -1.24022365f, -1.21824205f, -1.19704747f, -1.17650998f,
        -1.15636790f, -1.13686514f, -1.11771226f, -1.09937215f, -1.08145535f, -1.06413019f, -1.04700780f, -1.02985537f,
        -1.01289260f, -0.99635839f, -0.97985989f, -0.96346331f, -0.94718844f, -0.93142581f, -0.91591436f, -0.90098518f,
        -0.88661534f, -0.87208009f, -0.85743922f, -0.84298599f, -0.82868934f, -0.81421226f, -0.79995292f, -0.78608519f,
        -0.77253389f, -0.75930405f, -0.74655443f, -0.73399001f, -0.72171885f, -0.70948768f, -0.69729090f, -0.68519193f,
        -0.67294139f, -0.66097361f, -0.64903301f, -0.63699722f, -0.62520957f, -0.61343849f, -0.60164148f, -0.58989501f,
        -0.57817614f, -0.56638294f, -0.55476522f, -0.54346019f, -0.53228927f, -0.52154797f, -0.51064700f, -0.49990472f,
        -0.48919857f, -0.47810543f, -0.46704516f, -0.45603955f, -0.44486380f, -0.43340886f, -0.42221144f, -0.41127491f,
        -0.40069824f, -0.39020312f, -0.37980038f, -0.36931747f, -0.35883522f, -0.34820479f, -0.33763838f, -0.32712156f,
        -0.31680506f, -0.30650702f, -0.29633904f, -0.28598318f, -0.27587256f, -0.26576686f, -0.25534680f, -0.24487814f,
        -0.23456042f, -0.22439688f, -0.21428297f, -0.20450732f, -0.19472016f, -0.18452135f, -0.17464319f, -0.16479470f,
        -0.15506494f, -0.14514914f, -0.13510469f, -0.12490919f, -0.11475672f, -0.10477102f, -0.09481984f, -0.08470155f,
        -0.07488193f, -0.06495188f, -0.05493221f, -0.04509697f, -0.03520501f, -0.02532661f, -0.01568576f, -0.00597589f,
        0.00372220f, 0.01344266f, 0.02319991f, 0.03286659f, 0.04291259f, 0.05293084f, 0.06276978f, 0.07249805f,
        0.08250652f, 0.09261067f, 0.10255274f, 0.11248623f, 0.12212010f, 0.13166577f, 0.14129025f, 0.15095016f,
        0.16084716f, 0.17080866f, 0.18087213f, 0.19098914f, 0.20085125f, 0.21076398f, 0.22071999f, 0.23092821f,
        0.24115537f, 0.25143579f, 0.26179457f, 0.27212587f, 0.28246218f, 0.29251403f, 0.30247530f, 0.31238413f,
        0.32260880f, 0.33276638f, 0.34315947f, 0.35366595f, 0.36426541f, 0.37491897f, 0.38535890f, 0.39584789f,
        0.40643987f, 0.41705394f, 0.42789838f, 0.43888989f, 0.44957283f, 0.46015123f, 0.47093272f, 0.48179030f,
        0.49290875f, 0.50388461f, 0.51482475f, 0.52595741f, 0.53740937f, 0.54880124f, 0.56035239f, 0.57217956f,
        0.58371091f, 0.59516978f, 0.60679603f, 0.61841190f, 0.63014859f, 0.64220595f, 0.65432996f, 0.66648364f,
        0.67858601f, 0.69097805f, 0.70347679f, 0.71617913f, 0.72882068f, 0.74160296f, 0.75484610f, 0.76858562f,
        0.78214169f, 0.79558426f, 0.80941713f, 0.82344604f, 0.83751440f, 0.85179305f, 0.86665273f, 0.88163447f,
        0.89666569f, 0.91165745f, 0.92661709f, 0.94195455f, 0.95747328f, 0.97319126f, 0.98969555f, 1.00626135f,
        1.02299738f, 1.03975272f, 1.05691993f, 1.07431412f, 1.09259737f, 1.11127019f, 1.13069904f, 1.15059435f,
        1.17105842f, 1.19241130f, 1.21456146f, 1.23746312f, 1.26111495f, 1.28548789f, 1.31057048f, 1.33675921f,
        1.36415398f, 1.39277720f, 1.42348731f, 1.45656335f, 1.49078155f, 1.52804673f, 1.56803429f, 1.61146212f,
        1.65781057f, 1.70859325f, 1.76440656f, 1.82453620f, 1.89094412f, 1.96517146f, 2.04702711f, 2.13570380f,
        2.23243070f, 2.34583116f, 2.47412658f, 2.62253070f, 2.79892921f, 3.01957417f, 3.30903292f, 3.74206471f};

    /**
     * @brief 255 thresholds for TQ8 interval-based nearest-centroid.
     *
     * Midpoints of adjacent centroids: threshold[i] = (centroid[i] + centroid[i+1]) / 2.
     */
    inline constexpr std::array<float, 255> TQ8_THRESHOLDS = {
        -3.54918265f, -3.16375256f, -2.91246808f, -2.71999407f, -2.55894756f, -2.42005420f, -2.29915297f, -2.19240344f,
        -2.09776616f, -2.01232213f, -1.93395919f, -1.86274850f, -1.79812080f, -1.74001831f, -1.68793452f, -1.63973254f,
        -1.59426898f, -1.55180943f, -1.51240474f, -1.47610319f, -1.44246078f, -1.41063863f, -1.38062197f, -1.35265875f,
        -1.32597804f, -1.30042106f, -1.27586037f, -1.25197703f, -1.22923285f, -1.20764476f, -1.18677872f, -1.16643894f,
        -1.14661652f, -1.12728870f, -1.10854220f, -1.09041375f, -1.07279277f, -1.05556899f, -1.03843158f, -1.02137399f,
        -1.00462550f, -0.98810914f, -0.97166160f, -0.95532587f, -0.93930712f, -0.92367008f, -0.90844977f, -0.89380026f,
        -0.87934771f, -0.86475965f, -0.85021260f, -0.83583766f, -0.82145080f, -0.80708259f, -0.79301906f, -0.77930954f,
        -0.76591897f, -0.75292924f, -0.74027222f, -0.72785443f, -0.71560326f, -0.70338929f, -0.69124141f, -0.67906666f,
        -0.66695750f, -0.65500331f, -0.64301512f, -0.63110340f, -0.61932403f, -0.60753998f, -0.59576824f, -0.58403558f,
        -0.57227954f, -0.56057408f, -0.54911271f, -0.53787473f, -0.52691862f, -0.51609749f, -0.50527586f, -0.49455164f,
        -0.48365200f, -0.47257529f, -0.46154235f, -0.45045167f, -0.43913633f, -0.42781015f, -0.41674317f, -0.40598658f,
        -0.39545068f, -0.38500175f, -0.37455893f, -0.36407635f, -0.35352001f, -0.34292158f, -0.33237997f, -0.32196331f,
        -0.31165604f, -0.30142303f, -0.29116111f, -0.28092787f, -0.27081971f, -0.26055683f, -0.25011247f, -0.23971928f,
        -0.22947865f, -0.21933993f, -0.20939515f, -0.19961374f, -0.18962076f, -0.17958227f, -0.16971894f, -0.15992982f,
        -0.15010704f, -0.14012691f, -0.13000694f, -0.11983296f, -0.10976387f, -0.09979543f, -0.08976069f, -0.07979174f,
        -0.06991690f, -0.05994205f, -0.05001459f, -0.04015099f, -0.03026581f, -0.02050619f, -0.01083082f, -0.00112684f,
        0.00858243f, 0.01832129f, 0.02803325f, 0.03788959f, 0.04792172f, 0.05785031f, 0.06763391f, 0.07750228f,
        0.08755860f, 0.09758171f, 0.10751949f, 0.11730316f, 0.12689293f, 0.13647801f, 0.14612021f, 0.15589866f,
        0.16582791f, 0.17584039f, 0.18593063f, 0.19592019f, 0.20580761f, 0.21574198f, 0.22582410f, 0.23604179f,
        0.24629558f, 0.25661518f, 0.26696022f, 0.27729402f, 0.28748810f, 0.29749466f, 0.30742972f, 0.31749646f,
        0.32768759f, 0.33796293f, 0.34841271f, 0.35896568f, 0.36959219f, 0.38013893f, 0.39060339f, 0.40114388f,
        0.41174690f, 0.42247616f, 0.43339413f, 0.44423136f, 0.45486203f, 0.46554197f, 0.47636151f, 0.48734953f,
        0.49839668f, 0.50935468f, 0.52039108f, 0.53168339f, 0.54310530f, 0.55457681f, 0.56626597f, 0.57794523f,
        0.58944035f, 0.60098290f, 0.61260396f, 0.62428024f, 0.63617727f, 0.64826795f, 0.66040680f, 0.67253482f,
        0.68478203f, 0.69722742f, 0.70982796f, 0.72249991f, 0.73521182f, 0.74822453f, 0.76171586f, 0.77536365f,
        0.78886297f, 0.80250069f, 0.81643158f, 0.83048022f, 0.84465373f, 0.85922289f, 0.87414360f, 0.88915008f,
        0.90416157f, 0.91913727f, 0.93428582f, 0.94971392f, 0.96533227f, 0.98144341f, 0.99797845f, 1.01462936f,
        1.03137505f, 1.04833633f, 1.06561702f, 1.08345574f, 1.10193378f, 1.12098461f, 1.14064670f, 1.16082639f,
        1.18173486f, 1.20348638f, 1.22601229f, 1.24928904f, 1.27330142f, 1.29802918f, 1.32366484f, 1.35045660f,
        1.37846559f, 1.40813226f, 1.44002533f, 1.47367245f, 1.50941414f, 1.54804051f, 1.58974820f, 1.63463634f,
        1.68320191f, 1.73649991f, 1.79447138f, 1.85774016f, 1.92805779f, 2.00609928f, 2.09136546f, 2.18406725f,
        2.28913093f, 2.40997887f, 2.54832864f, 2.71072996f, 2.90925169f, 3.16430354f, 3.52554882f};

    // ========================================================================
    // Nearest-centroid lookup (scalar)
    // ========================================================================

    /**
     * @brief Find nearest 4-bit centroid index for a scalar value.
     *
     * Uses threshold-based interval lookup (4 comparisons via binary search).
     * Input should be pre-scaled by sqrt(d) for unit-sphere quantization.
     *
     * @param x Scalar value (in N(0,1) scale after sqrt(d) scaling)
     * @return Index 0-15 of nearest centroid
     */
    inline uint8_t tq4_nearest_centroid(float x)
    {
        // Binary search through 15 thresholds to find interval
        // Unrolled for performance (exactly 4 comparisons for 16 levels)
        uint8_t idx = 0;
        idx += (x > TQ4_THRESHOLDS[7]) ? 8 : 0;
        idx += (x > TQ4_THRESHOLDS[idx + 3]) ? 4 : 0;
        idx += (x > TQ4_THRESHOLDS[idx + 1]) ? 2 : 0;
        // Final comparison: idx is now in {0..14}, check if we need +1
        if (idx < 15 && x > TQ4_THRESHOLDS[idx])
            idx++;
        return idx;
    }

    /**
     * @brief Find nearest 3-bit centroid index for a scalar value.
     *
     * Uses threshold-based interval lookup (3 comparisons via binary search).
     *
     * @param x Scalar value (in N(0,1) scale after sqrt(d) scaling)
     * @return Index 0-7 of nearest centroid
     */
    inline uint8_t tq3_nearest_centroid(float x)
    {
        // Binary search through 7 thresholds (3 comparisons for 8 levels)
        uint8_t idx = 0;
        idx += (x > TQ3_THRESHOLDS[3]) ? 4 : 0;
        idx += (x > TQ3_THRESHOLDS[idx + 1]) ? 2 : 0;
        if (idx < 7 && x > TQ3_THRESHOLDS[idx])
            idx++;
        return idx;
    }

    inline uint8_t tq2_nearest_centroid(float x)
    {
        uint8_t idx = 0;
        idx += (x > TQ2_THRESHOLDS[1]) ? 2 : 0;
        if (idx < 3 && x > TQ2_THRESHOLDS[idx])
            idx++;
        return idx;
    }

    /**
     * @brief Find nearest 8-bit centroid index for a scalar value.
     *
     * Uses threshold-based binary search (8 comparisons for 256 levels).
     * Input should be pre-scaled by sqrt(d) for unit-sphere quantization.
     *
     * @param x Scalar value (in N(0,1) scale after sqrt(d) scaling)
     * @return Index 0-255 of nearest centroid
     */
    inline uint8_t tq8_nearest_centroid(float x)
    {
        // Binary search through 255 thresholds (8 comparisons for 256 levels)
        uint8_t idx = 0;
        idx += (x > TQ8_THRESHOLDS[idx + 127]) ? 128 : 0;
        idx += (x > TQ8_THRESHOLDS[idx + 63]) ? 64 : 0;
        idx += (x > TQ8_THRESHOLDS[idx + 31]) ? 32 : 0;
        idx += (x > TQ8_THRESHOLDS[idx + 15]) ? 16 : 0;
        idx += (x > TQ8_THRESHOLDS[idx + 7]) ? 8 : 0;
        idx += (x > TQ8_THRESHOLDS[idx + 3]) ? 4 : 0;
        idx += (x > TQ8_THRESHOLDS[idx + 1]) ? 2 : 0;
        if (idx < 255 && x > TQ8_THRESHOLDS[idx])
            idx++;
        return idx;
    }

    // ========================================================================
    // Lloyd-Max verification utilities
    // ========================================================================

    /**
     * @brief Standard normal PDF φ(x) = exp(-x²/2) / √(2π)
     */
    inline double gaussian_pdf(double x)
    {
        constexpr double INV_SQRT_2PI = 0.3989422804014327;
        return INV_SQRT_2PI * std::exp(-0.5 * x * x);
    }

    /**
     * @brief Standard normal CDF Φ(x) using erfc
     */
    inline double gaussian_cdf(double x)
    {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }

    /**
     * @brief Conditional expectation E[X | a ≤ X ≤ b] for X ~ N(0,1)
     *
     * = (φ(a) - φ(b)) / (Φ(b) - Φ(a))
     */
    inline double gaussian_conditional_mean(double a, double b)
    {
        double prob = gaussian_cdf(b) - gaussian_cdf(a);
        if (prob < 1e-15)
            return 0.5 * (a + b);
        return (gaussian_pdf(a) - gaussian_pdf(b)) / prob;
    }

    /**
     * @brief Apply one Lloyd-Max iteration to a set of centroids.
     *
     * Given current centroids, computes thresholds as midpoints, then updates
     * each centroid to the conditional expectation E[X | interval] for N(0,1).
     *
     * @tparam K Number of quantization levels (must be even for symmetry)
     * @param centroids Current centroids (updated in-place)
     * @return Maximum absolute centroid movement from this iteration
     */
    template <int K>
    inline double lloyd_max_iteration(std::array<double, K> &centroids)
    {
        static_assert(K >= 2 && K % 2 == 0, "K must be even and >= 2");

        // Step 1: thresholds = midpoints of adjacent centroids
        std::array<double, K - 1> thresholds;
        for (int k = 0; k < K - 1; ++k)
            thresholds[k] = 0.5 * (centroids[k] + centroids[k + 1]);

        // Step 2: centroids = conditional expectations over each interval
        std::array<double, K> new_centroids;
        new_centroids[0] = gaussian_conditional_mean(-10.0, thresholds[0]);
        for (int k = 1; k < K - 1; ++k)
            new_centroids[k] = gaussian_conditional_mean(thresholds[k - 1], thresholds[k]);
        new_centroids[K - 1] = gaussian_conditional_mean(thresholds[K - 2], 10.0);

        double max_delta = 0.0;
        for (int k = 0; k < K; ++k)
            max_delta = std::max(max_delta, std::abs(new_centroids[k] - centroids[k]));

        centroids = new_centroids;
        return max_delta;
    }

    /**
     * @brief Compute Lloyd-Max optimal centroids for N(0,1) with K levels.
     *
     * Iterates the Lloyd-Max algorithm from a uniform initialization until
     * convergence (centroid movement < tolerance). No arbitrary iteration cap.
     *
     * @tparam K Number of quantization levels (must be even for symmetry)
     * @param tolerance Convergence tolerance on centroid movement (default 1e-12)
     * @return Array of K sorted centroid values
     */
    template <int K>
    inline std::array<double, K> compute_lloyd_max_centroids(double tolerance = 1e-12)
    {
        static_assert(K >= 2 && K % 2 == 0, "K must be even and >= 2");

        std::array<double, K> centroids;
        for (int k = 0; k < K; ++k)
            centroids[k] = -3.0 + (k + 0.5) * 6.0 / K;

        while (lloyd_max_iteration<K>(centroids) >= tolerance)
        {
        }

        return centroids;
    }

    /**
     * @brief Verify that pre-computed codebook centroids are a Lloyd-Max fixed point.
     *
     * Seeds one Lloyd-Max iteration with the stored centroids and measures how
     * much they move. A true fixed point moves by 0 (limited by float32 precision).
     * This avoids iterating from scratch and any dependence on iteration counts.
     *
     * @tparam K Number of centroids
     * @param stored_centroids Pre-computed centroid table to verify
     * @return Maximum centroid movement after one Lloyd-Max iteration
     */
    template <int K>
    inline double verify_codebook(const std::array<float, K> &stored_centroids)
    {
        // Promote to double and apply one Lloyd-Max step
        std::array<double, K> centroids;
        for (int k = 0; k < K; ++k)
            centroids[k] = static_cast<double>(stored_centroids[k]);

        return lloyd_max_iteration<K>(centroids);
    }

    /**
     * @brief Theoretical MSE of Lloyd-Max quantizer for N(0, σ²) with K levels.
     *
     * Computes the distortion D = E[(X - Q(X))²] for the optimal K-level quantizer.
     * Uses the converged centroids and thresholds.
     *
     * @param num_levels Number of quantization levels (8 or 16)
     * @param sigma Standard deviation of the Gaussian
     * @return Expected MSE per element
     */
    inline double lloyd_max_mse(int num_levels, double sigma = 1.0)
    {
        if (num_levels == 16)
        {
            auto centroids_d = compute_lloyd_max_centroids<16>();
            std::array<double, 15> thresholds_d;
            for (int k = 0; k < 15; ++k)
                thresholds_d[k] = 0.5 * (centroids_d[k] + centroids_d[k + 1]);

            double mse = 0.0;
            for (int k = 0; k < 16; ++k)
            {
                double a = (k == 0) ? -10.0 : thresholds_d[k - 1];
                double b = (k == 15) ? 10.0 : thresholds_d[k];
                double c = centroids_d[k];
                // E[(X - c)² | a ≤ X ≤ b] * P(a ≤ X ≤ b)
                // = E[X² | a ≤ X ≤ b] * p - 2c * E[X | a ≤ X ≤ b] * p + c² * p
                double p = gaussian_cdf(b) - gaussian_cdf(a);
                double ex = gaussian_conditional_mean(a, b) * p;
                // E[X² | a ≤ X ≤ b] * p = (x*φ(x) evaluated at boundaries) + p
                // Actually: E[X²] in interval = integral of x² * φ(x)
                // = [a·φ(a) - b·φ(b)] + p  for N(0,1)
                double ex2 = (a * gaussian_pdf(a) - b * gaussian_pdf(b)) + p;
                mse += ex2 - 2.0 * c * ex + c * c * p;
            }
            return mse * sigma * sigma;
        }
        if (num_levels == 8)
        {
            auto centroids_d = compute_lloyd_max_centroids<8>();
            std::array<double, 7> thresholds_d;
            for (int k = 0; k < 7; ++k)
                thresholds_d[k] = 0.5 * (centroids_d[k] + centroids_d[k + 1]);

            double mse = 0.0;
            for (int k = 0; k < 8; ++k)
            {
                double a = (k == 0) ? -10.0 : thresholds_d[k - 1];
                double b = (k == 7) ? 10.0 : thresholds_d[k];
                double c = centroids_d[k];
                double p = gaussian_cdf(b) - gaussian_cdf(a);
                double ex = gaussian_conditional_mean(a, b) * p;
                double ex2 = (a * gaussian_pdf(a) - b * gaussian_pdf(b)) + p;
                mse += ex2 - 2.0 * c * ex + c * c * p;
            }
            return mse * sigma * sigma;
        }
        return -1.0; // unsupported
    }

} // namespace llaminar2
