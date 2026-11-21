/**
 * @file ROCmNativeQuantisedVerifierWoPolicy.inc
 * @brief Measured ordinary projection widths from actual Wo stage transactions.
 *
 * Shapes retain full and TP-sized local extents; TP source-slice loading is a
 * separate contract. Only the original Projection operation may use these masks.
 * SwiGLU and enclosing fused bundles cannot borrow a standalone projection win.
 * Each seed key wins all31 pairs with at least5% complete-operation median gain;
 * the first actual-Automatic panel must retain that median margin as well.
 * These are qualification margins, not promised future performance floors.
 * Automatic selection receipt SHA256 610667564d6f7d617c8e36b4f2312f2f03ecb6ccab1a9fbe89b7c9617a1f8add.
 * Seed SHA256 1c9e1bbbb442aa318809f0a167883963683ce744a2161ac789d39c136754ad0c.
 */
constexpr std::array<std::pair<int,int>,6> kVerifierWoShapes{{
    {5120,6144}, // Dense
    {5120,3072}, // DenseTP2
    {5120,1536}, // DenseTP4
    {2048,4096}, // MoE
    {2048,2048}, // MoETP2
    {2048,1024}, // MoETP4
}};
constexpr std::array<VerifierWoMeasuredWidths,21> kVerifierWoMeasuredWidths{{
    {{0,false,true},{0x4000,0x3400,0x3800,0x7f80,0x7800,0x0000}}, // Q4_0,
    {{4,false,true},{0x7800,0x7b00,0x7e00,0x7f80,0x7f00,0x7800}}, // IQ4_NL,
    {{4,true,true},{0x7800,0x6f00,0x7f00,0x7f80,0x7f00,0x7000}}, // IQ4_XS,
    {{5,false,true},{0x6000,0x5800,0x7f00,0x7fc0,0x7c00,0x7800}}, // Q4_1,
    {{5,true,true},{0x0000,0x2600,0x7e00,0x7f80,0x7c00,0x7800}}, // Q4_K,
    {{6,false,true},{0x7f80,0x7fe8,0x7f80,0x7ff8,0x7fc0,0x7f00}}, // Q5_0,
    {{7,false,true},{0x7f80,0x7f80,0x7f80,0x7f80,0x7f80,0x7f80}}, // Q5_1,
    {{7,true,true},{0x7f80,0x7f80,0x7f80,0x7ff0,0x7f80,0x7d80}}, // Q5_K,
    {{8,true,true},{0x7f80,0x7f80,0x7fe0,0x7ff8,0x7f80,0x7f80}}, // Q6_K,
    {{9,true,true},{0x7f00,0x7b00,0x7800,0x7f80,0x7e00,0x7800}}, // Q3_K,
    {{10,true,true},{0x7c00,0x7c00,0x7800,0x7f00,0x7800,0x3000}}, // Q2_K,
    {{11,true,true},{0x7f80,0x7f80,0x7c00,0x7fc0,0x7f80,0x7f00}}, // IQ3_S,
    {{12,true,true},{0x7f80,0x7f80,0x7800,0x7fc0,0x7f80,0x7e00}}, // IQ3_XXS,
    {{13,true,true},{0x7f80,0x7f80,0x7800,0x7f80,0x7f80,0x7f00}}, // IQ2_S,
    {{14,true,true},{0x7f80,0x7f80,0x6800,0x7f80,0x7f80,0x7e00}}, // IQ2_XS,
    {{15,true,true},{0x7f80,0x7f80,0x0000,0x7f80,0x7f80,0x7000}}, // IQ2_XXS,
    {{16,true,true},{0x7d00,0x7b80,0x0000,0x7fc0,0x7f80,0x7e00}}, // IQ1_S,
    {{17,true,true},{0x7800,0x7c00,0x0000,0x7e00,0x7800,0x3000}}, // IQ1_M,
    {{19,false,true},{0x7800,0x0000,0x7e80,0x7f80,0x7800,0x0000}}, // Q8_0,
    {{20,false,true},{0x7800,0x0800,0x3e00,0x7f00,0x7800,0x0000}}, // Q8_1,
    {{21,true,true},{0x7a00,0x2000,0x7f00,0x7f00,0x7800,0x0000}}, // Q8_K
}};
