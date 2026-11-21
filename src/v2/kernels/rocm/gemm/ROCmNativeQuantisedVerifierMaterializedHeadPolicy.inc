/**
 * @file ROCmNativeQuantisedVerifierMaterializedHeadPolicy.inc
 * @brief Source-exact widths for complete row-selection and full-vocabulary head work.
 *
 * Columns retain original K2048/K5120, with N248320. Bit M-2 names exactly
 * one physical width. Each admitted seed wins all31 pairs with at least5%
 * complete-operation median gain; actual Automatic is qualified separately.
 * Source aliases, smaller vocabularies and counted rows cannot borrow a key.
 * Seed SHA256 69d3e89177fc843e696a5f13ff80aefa2b38304f322a1cfadc18a31719fc30ed.
 */
constexpr std::array<VerifierMaterializedHeadMeasuredWidths,21> kVerifierMaterializedHeadMeasuredWidths{{
    {{0,false,true},{0x7ff8,0x7ff8}}, // Q4_0
    {{4,false,true},{0x7ff8,0x7ff8}}, // IQ4_NL
    {{4,true,true},{0x7ff8,0x7ff8}}, // IQ4_XS
    {{5,false,true},{0x7ff8,0x7ff8}}, // Q4_1
    {{5,true,true},{0x7ff8,0x7ff8}}, // Q4_K
    {{6,false,true},{0x7ff8,0x7ff8}}, // Q5_0
    {{7,false,true},{0x7ff8,0x7ff8}}, // Q5_1
    {{7,true,true},{0x7ff8,0x7ff8}}, // Q5_K
    {{8,true,true},{0x7ff8,0x7ff8}}, // Q6_K
    {{9,true,true},{0x7ff8,0x7ff8}}, // Q3_K
    {{10,true,true},{0x7ff8,0x7ff8}}, // Q2_K
    {{11,true,true},{0x7ff8,0x7ff8}}, // IQ3_S
    {{12,true,true},{0x7ff8,0x7ff8}}, // IQ3_XXS
    {{13,true,true},{0x7ff8,0x7ff8}}, // IQ2_S
    {{14,true,true},{0x7ff8,0x7ff8}}, // IQ2_XS
    {{15,true,true},{0x7ff8,0x7ff8}}, // IQ2_XXS
    {{16,true,true},{0x7ff0,0x7f80}}, // IQ1_S
    {{17,true,true},{0x7f80,0x7e80}}, // IQ1_M
    {{19,false,true},{0x7ff8,0x7ff8}}, // Q8_0
    {{20,false,true},{0x7ff8,0x7ff8}}, // Q8_1
    {{21,true,true},{0x7ff8,0x7ff8}}, // Q8_K
}};
