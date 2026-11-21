/**
 * @file ROCmNativeQuantisedGDNVerifierPolicy.inc
 * @brief Measured complete GDN widths for exact hosted gfx1100 transactions.
 *
 * Bit M-2 selects native at M2..16; columns are SingleQ8 and TwoTermQ8.
 * Every key wins all 31 forced pairs with at least a 5% median gain for the
 * entire four-role stage. Automatic confirmation owns final qualification.
 * Evidence: GFX1100_MTP_GDN_VERIFIER.md and its frozen seed receipt.
 */
constexpr std::array<GDNVerifierMeasuredWidths,21> kGDNVerifierMeasuredWidths{{
    {{0,false,true}, {0x7800,0x0000}}, // Q4_0
    {{4,false,true}, {0x7800,0x0000}}, // IQ4_NL
    {{4,true,true}, {0x7800,0x0800}}, // IQ4_XS
    {{5,false,true}, {0x7800,0x0000}}, // Q4_1
    {{5,true,true}, {0x7800,0x0000}}, // Q4_K
    {{6,false,true}, {0x7f00,0x7f00}}, // Q5_0
    {{7,false,true}, {0x7f00,0x7800}}, // Q5_1
    {{7,true,true}, {0x7f80,0x7800}}, // Q5_K
    {{8,true,true}, {0x7f80,0x7b80}}, // Q6_K
    {{9,true,true}, {0x7800,0x1000}}, // Q3_K
    {{10,true,true}, {0x7800,0x0000}}, // Q2_K
    {{11,true,true}, {0x7e00,0x7800}}, // IQ3_S
    {{12,true,true}, {0x7d00,0x2000}}, // IQ3_XXS
    {{13,true,true}, {0x7f80,0x3000}}, // IQ2_S
    {{14,true,true}, {0x7f00,0x0000}}, // IQ2_XS
    {{15,true,true}, {0x7800,0x0000}}, // IQ2_XXS
    {{16,true,true}, {0x7000,0x0000}}, // IQ1_S
    {{17,true,true}, {0x6000,0x0000}}, // IQ1_M
    {{19,false,true}, {0x7f80,0x7800}}, // Q8_0
    {{20,false,true}, {0x7f00,0x7880}}, // Q8_1
    {{21,true,true}, {0x7f80,0x7800}}, // Q8_K
}};
