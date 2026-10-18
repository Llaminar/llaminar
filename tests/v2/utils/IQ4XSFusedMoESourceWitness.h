/**
 * @file IQ4XSFusedMoESourceWitness.h
 * @brief Original-source role witnesses for the real optimized MoE transaction.
 *
 * No expected weight is read from a prepared plane or a proposed factor helper.
 * One live 32-value dot fixes the reduction independently of partition geometry.
 * Control roles use exact-factor IQ4_XS or literal exact Q8_0 blocks. Q8 activations,
 * IQ4_NL remains an unchanged compatibility control. Canonical SwiGLU and the ordinary route fold remain enabled and unchanged.
 */
#pragma once
#include "tensors/TensorClasses.h"
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/DeviceSwiGLUNumericalContract.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <utility>

namespace llaminar2::test::iq4xs_fused_source_witness
{
constexpr int width = 256;
constexpr int blocks = width / 32;
/** @brief Only the target role has an inexact derived-half factor in each witness. */
enum class Role { Gate, Up, Down, IQ4NLDown, MixedGate, MixedUp };
/** @brief Original-format pairs retained by the source-scale representation change. */
inline constexpr std::array<Role,2> mixedRoles{Role::MixedGate,Role::MixedUp};
/** @brief Ordinary isolated roles, kept distinct from mixed-pair admission tests. */
inline constexpr std::array<Role,4> ordinaryRoles{Role::Gate,Role::Up,Role::Down,Role::IQ4NLDown};
/** @brief Whether the gate carries the inexact original IQ4_XS factor. */
inline bool gateTarget(Role role) { return role==Role::Gate||role==Role::MixedGate; }
/** @brief Whether the up projection carries the inexact original IQ4_XS factor. */
inline bool upTarget(Role role) { return role==Role::Up||role==Role::MixedUp; }
/** @brief Whether paired inputs have different original IQ4 source formats. */
inline bool mixedPair(Role role) { return role==Role::MixedGate||role==Role::MixedUp; }
/** @brief Literal source owners retained through all kernel and capture users. */
struct Sources
{
    std::unique_ptr<TensorBase> gate, up, down;
};
/**
 * @brief Construct one immutable source tensor from literal block bytes.
 * @tparam Block Original GGUF block, never a prepared record.
 * @tparam Tensor Matching original source tensor class.
 * @param values Complete logical row-major native blocks.
 * @return Source owner with exact input bytes.
 */
template<class Block, class Tensor>
std::unique_ptr<TensorBase> source(std::vector<Block> values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(Block));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return std::make_unique<Tensor>(std::vector<size_t>{width, width}, bytes);
}
/**
 * @brief Create the same ordinary source-factor loss in every output row.
 * @param d Original source half word.
 * @param coefficient Original unsigned six-bit coefficient, before subtracting32.
 * @return Original source words, IQ4_NL grid1 and original signed-scale code.
 */
inline std::unique_ptr<TensorBase> iq4(uint16_t d=0x3c01u, uint8_t coefficient=35u)
{
    std::vector<IQ4_XSBlock> values(width);
    for (auto &b : values)
    {
        b.d = d;
        std::fill_n(b.qs, 128, uint8_t{0x88}); // LUT index8 is exactly integer1.
        for(int sub=0;sub<8;++sub)
        {
            b.scales_l[sub/2] |= uint8_t((coefficient&15u) << (4*(sub%2)));
            b.scales_h |= uint16_t((coefficient>>4) << (2*sub));
        }
    }
    return source<IQ4_XSBlock, IQ4_XSTensor>(std::move(values));
}
/**
 * @brief Preserve the existing IQ4_NL source/layout as an exact compatibility control.
 * @param d Literal original half factor, without conversion or requantization.
 * @return Original half factor and integer-grid1 in the first logical block only.
 */
inline std::unique_ptr<TensorBase> iq4nl(uint16_t d=0x3c00u)
{
    std::vector<IQ4_NLBlock> values(width*blocks);
    for(int n=0;n<width;++n)
    {
        auto &b=values[n*blocks]; b.d=d;
        std::fill_n(b.qs,16,uint8_t{0x88});
    }
    return source<IQ4_NLBlock, IQ4_NLTensor>(std::move(values));
}
/**
 * @brief Create exact source controls without changing paired-codebook eligibility.
 * @param role IQ4_XS role whose other two projections are being isolated.
 * @param projection Zero gate, one up, two down.
 * @return Literal original source owner; no floating-to-quantized conversion.
 */
inline std::unique_ptr<TensorBase> control(Role role, int projection)
{
    // Equal LUT payload families may carry different source-scale programs.
    // Keep the control factor exact so only the selected XS role loses precision.
    if(projection < 2 && (gateTarget(role)||upTarget(role)))
    {
        const uint16_t d=gateTarget(role)?0x2020u:0x3c20u;
        return mixedPair(role)?iq4nl(d):iq4(d,33u);
    }
    std::vector<Q8_0Block> values(width * blocks);
    for (auto &b : values) b.d = 0x3c00u;
    for (int n = 0; n < width; ++n)
    {
        auto &b = values[n * blocks];
        if (projection == 2) std::fill_n(b.qs, 32, int8_t{1});
        else if (projection == 0) std::fill_n(b.qs, 32, int8_t{1});
        else if (n == 0 || n == 31) // only1 and127 are nonzero; dot128
        {
            b.d = 0x2000u; // 1/128
            std::fill_n(b.qs, 32, int8_t(n == 31 ? 127 : 1));
        }
    }
    return source<Q8_0Block, Q8_0Tensor>(std::move(values));
}
/**
 * @brief Allocate the role-isolated immutable original source matrices.
 * @param role Only this target projection loses source precision in derived-half packing.
 * @return Three original tensor owners.
 */
inline Sources sources(Role role)
{
    return {gateTarget(role) ? iq4() : control(role, 0),
            upTarget(role) ? iq4() : control(role, 1),
            role == Role::Down ? iq4() : role == Role::IQ4NLDown ? iq4nl() : control(role, 2)};
}
/**
 * @brief Define the exact unchanged hidden row before canonical quantization.
 * @return Integers1,127 and zero; active Q8 scale1 and dot128.
 */
inline std::array<float, width> hidden()
{
    std::array<float, width> x{}; x[0] = 1.0f; x[31] = 127.0f; return x;
}
/**
 * @brief Decode one integer directly from the original nibble bytes and literal LUT.
 * @param b Original IQ4_XS block.
 * @param p Logical position0..255.
 * @return Signed source integer without any scale.
 */
inline int integer(const IQ4_XSBlock &b, int p)
{
    const int sub=p/32, lane=p%32;
    const uint8_t q=b.qs[sub*16+lane%16];
    constexpr std::array<int,16> grid{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    return grid[(q>>(lane<16?0:4))&15];
}
/**
 * @brief Independently evaluate original half times signed six-bit integer.
 * @param d Original half base bits.
 * @param signed_coefficient Original signed six-bit coefficient.
 * @return Exact finite FP32 original source factor; no derived-half publication.
 */
inline float factor(uint16_t d, int signed_coefficient)
{
    // Binary16 times signed6-bit integer is exactly representable in FP32.
    return float(double(fp16_to_fp32(d))*double(signed_coefficient));
}
/**
 * @brief Evaluate one source IQ4_XS row with the consumed Q8 publication.
 * @param source Original superblock for this 256-value output row.
 * @param input Eight immutable Q8 blocks actually consumed by the operator.
 * @return Original-source contribution with existing single-dot rounding edges.
 * @details All witnesses have only one nonzero integer-dot block; other trees
 * contain +0, so this comparison does not prescribe a new reduction order.
 */
inline float iq4Dot(const IQ4_XSBlock &source, const Q8_1Block *input)
{
    using namespace device_fp32_contract;
    float result=0.0f;
    for(int g=0;g<blocks;++g)
    {
        int dot=0;
        for(int i=0;i<32;++i) dot+=integer(source,g*32+i)*int(input[g].qs[i]);
        const int code=((source.scales_l[g/2]>>(4*(g%2)))&15)|(((source.scales_h>>(2*g))&3)<<4);
        result=add(result,multiply(multiply(fp16_to_fp32(input[g].d),factor(source.d,code-32)),float(dot)));
    }
    return result;
}
/**
 * @brief Define source gate/up checkpoint values for one isolated role.
 * @param role IQ4_XS role.
 * @param gate Output original-source gate row.
 * @param up Output original-source up row.
 */
inline void expectedGateUp(Role role, std::array<float,width> &gate,
    std::array<float,width> &up)
{
    gate.fill(gateTarget(role) ? 384.375f : upTarget(role) ? 132.0f : 128.0f);
    up.fill(upTarget(role) ? 384.375f : gateTarget(role) ? 33.0f/32.0f : 0.0f);
    if(role==Role::Down || role==Role::IQ4NLDown) { up[0]=1.0f; up[31]=127.0f; }
}
/**
 * @brief Define middle bytes from source projection values and unchanged policy.
 * @param role IQ4_XS role.
 * @return Canonical Q8 source-equation middle, including raw integer sums.
 * @details This shares only the deliberately fixed activation program, never
 * weight factor resolution, prepared metadata or a production projection.
 */
inline std::array<Q8_1Block,blocks> expectedMiddle(Role role)
{
    std::array<float,width> gate{},up{},middle{};
    expectedGateUp(role,gate,up);
    for(int i=0;i<width;++i) middle[i]=device_swiglu_contract::swigluValue(gate[i],up[i]);
    std::array<Q8_1Block,blocks> result{};
    for(int g=0;g<blocks;++g)
    {
        float maximum=0.0f;
        for(int i=0;i<32;++i) maximum=std::fmax(maximum,std::fabs(middle[g*32+i]));
        const float d=device_q8_activation_contract::scale(maximum);
        // Returned d is exactly a half value; conversion cannot round it again.
        result[g].d=fp32_to_fp16(d);
        for(int i=0;i<32;++i)
        {
            result[g].qs[i]=int8_t(device_q8_activation_contract::quantize(middle[g*32+i],d));
            result[g].sum_qs+=result[g].qs[i];
        }
    }
    return result;
}
/**
 * @brief Close down using actual middle words while avoiding packed IQ4_XS data.
 * @param role IQ4_XS role.
 * @param source Original down tensor retained by the fixture.
 * @param actual_middle Actual publication consumed by this transaction's down.
 * @return Defining original-source output at route weight1.
 */
inline float expectedDown(Role role, const TensorBase &source, const Q8_1Block *actual_middle)
{
    using namespace device_fp32_contract;
    if(role==Role::Down)
        return iq4Dot(*static_cast<const IQ4_XSTensor&>(source).blocks(),actual_middle);
    // IQ4_NL compatibility and literal Q8 controls both have scale1/grid1.
    int dot=0; for(int i=0;i<32;++i) dot+=actual_middle[0].qs[i];
    return multiply(fp16_to_fp32(actual_middle[0].d),float(dot));
}
} // namespace llaminar2::test::iq4xs_fused_source_witness
