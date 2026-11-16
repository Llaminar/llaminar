/**
 * @file ROCmNativeQuantisedVerifierGateUpPolicy.inc
 * @brief Measured original two-source GU widths; down retains its existing scalar authority.
 *
 * Complete connected FFN qualification compares forced GU/Automatic down against
 * current Automatic at K5120 and both N17408. Each admitted source/M wins all31
 * pairs with at least5% median gain. This seed filter is not a promised minimum.
 * Unknown geometry, mixed sources and unmeasured widths remain ordered.
 * Seed SHA256 862098fd6eb536efecd9a331e6946d2effb5ad95dd21b84c8b08f2a2fd3fad54.
 */
constexpr std::array<VerifierGateUpMeasuredWidths,21> kVerifierGateUpMeasuredWidths{{
    {{0,false,true},0x7f80}, // Q4_0
    {{4,false,true},0x7f80}, // IQ4_NL
    {{4,true,true},0x7b80}, // IQ4_XS
    {{5,false,true},0x7f00}, // Q4_1
    {{5,true,true},0x7e00}, // Q4_K
    {{6,false,true},0x7f80}, // Q5_0
    {{7,false,true},0x7f80}, // Q5_1
    {{7,true,true},0x7f80}, // Q5_K
    {{8,true,true},0x7c00}, // Q6_K
    {{9,true,true},0x7800}, // Q3_K
    {{10,true,true},0x7800}, // Q2_K
    {{11,true,true},0x7e00}, // IQ3_S
    {{12,true,true},0x7800}, // IQ3_XXS
    {{13,true,true},0x7c00}, // IQ2_S
    {{14,true,true},0x7800}, // IQ2_XS
    {{15,true,true},0x7800}, // IQ2_XXS
    {{16,true,true},0x4000}, // IQ1_S
    {{17,true,true},0x7000}, // IQ1_M
    {{19,false,true},0x7ff8}, // Q8_0
    {{20,false,true},0x7ff8}, // Q8_1
    {{21,true,true},0x7ff8}, // Q8_K
}};
