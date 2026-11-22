/**
 * @file FullNativeHCFFNFamilyCases.h
 * @brief Original source recipes shared by the HC/router/FFN hardware comparisons.
 *
 * These fixtures preserve independent source identities for the six FFN
 * projections. Each backend resolves those identities to its own prepared
 * program and launch geometry.
 */
#pragma once

#include <array>

namespace llaminar2::test
{
/** @brief Immutable test source recipe, never a production decoder override. */
struct FullNativeHCFFNFamilyCase {
    unsigned groups; ///< Smallest source-authenticated decoder union selected on MI50.
    const char* name; ///< Stable gtest/receipt identity.
    const char* gate; ///< Same original gate/up source codebook.
    const char* down; ///< Independent original routed-down source codebook.
    const char* shared; ///< Original shared gate/up source and default shared down.
    const char* shared_down = nullptr; ///< Independently encoded original shared down projection when it differs.
};
/** @return Genuine source recipes covering each compiled union without padding K. */
inline constexpr std::array<FullNativeHCFFNFamilyCase, 15> fullNativeHCFFNFamilyCases() {
    return {{
        {1,"G01_Q4_K_DownQ5_1_SharedQ8_0","Q4_K","Q5_1","Q8_0"},
        {2,"G02_Q4_0","Q4_0","Q4_0","Q4_0"},
        {3,"G03_Q4_0_SharedQ8_0","Q4_0","Q4_0","Q8_0"},
        {4,"G04_IQ4_NL","IQ4_NL","IQ4_NL","IQ4_NL"},
        {5,"G05_IQ4_NL_SharedQ8_0","IQ4_NL","IQ4_NL","Q8_0"},
        {6,"G06_IQ4_NL_DownQ4_0_SharedQ4_0","IQ4_NL","Q4_0","Q4_0"},
        {7,"G07_IQ4_NL_DownQ4_0_SharedQ8_0","IQ4_NL","Q4_0","Q8_0"},
        {8,"G08_IQ2_S","IQ2_S","IQ2_S","IQ2_S"},
        {9,"G09_IQ2_S_SharedQ8_0","IQ2_S","IQ2_S","Q8_0"},
        {10,"G10_IQ2_S_DownQ4_0_SharedQ4_0","IQ2_S","Q4_0","Q4_0"},
        {11,"G11_IQ2_S_DownQ6_K_SharedQ8_0","IQ2_S","Q6_K","Q8_0"},
        {12,"G12_IQ3_S_DownIQ2_S_SharedIQ4_NL","IQ3_S","IQ2_S","IQ4_NL"},
        {13,"G13_IQ3_XXS_DownQ5_1_SharedIQ2_S","IQ3_XXS","Q5_1","IQ2_S"},
        {14,"G14_IQ3_S_DownQ6_K_SharedIQ2_S","IQ3_S","Q6_K","IQ2_S"},
        {15,"G15_IQ3_XXS_DownQ6_K_SharedIQ2_S_DownQ8_0","IQ3_XXS","Q6_K","IQ2_S","Q8_0"}
    }};
}
} // namespace llaminar2::test
