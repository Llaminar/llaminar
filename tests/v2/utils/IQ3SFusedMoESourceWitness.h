/**
 * @file IQ3SFusedMoESourceWitness.h
 * @brief Original-source role witnesses for the real optimized MoE transaction.
 *
 * No expected weight is read from a prepared plane or a proposed factor helper.
 * One live 32-value dot fixes the reduction independently of partition geometry.
 * Control roles use exact-factor IQ3_S or literal exact Q8_0 blocks. Q8 activations,
 * canonical SwiGLU and the ordinary route fold remain enabled and unchanged.
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

namespace llaminar2::test::iq3s_source_witness
{
constexpr int width = 256;
constexpr int blocks = width / 32;
/** @brief Only the target role has an inexact derived-half factor in each witness. */
enum class Role { Gate, Up, Down };
/** @brief Role-isolated target factors; all paired controls retain exact factors. */
inline constexpr std::array<Role,3> ordinaryRoles{Role::Gate,Role::Up,Role::Down};
/** @brief Whether gate alone owns the inexact legacy-derived factor. */
inline bool gateTarget(Role role) { return role==Role::Gate; }
/** @brief Whether up alone owns the inexact legacy-derived factor. */
inline bool upTarget(Role role) { return role==Role::Up; }
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
 * @brief Create literal original half/nibble blocks with grid0 (all magnitudes1).
 * @param d Original half base. @param nibble Original unsigned source coefficient.
 * @return Original source owner, never dequantized/requantized or prepared metadata.
 */
inline std::unique_ptr<TensorBase> iq3(uint16_t d=0x3c01u,uint8_t nibble=1u)
{
    std::vector<IQ3_SBlock> values(width);
    for(auto &b:values)
    {
        b.d=d;
        for(int sub=0;sub<8;++sub)b.scales[sub/2]|=uint8_t(nibble<<(4*(sub%2)));
    }
    return source<IQ3_SBlock,IQ3_STensor>(std::move(values));
}
/**
 * @brief Create exact source controls without changing paired-codebook eligibility.
 * @param role IQ3_S role whose other two projections are being isolated.
 * @param projection Zero gate, one up, two down.
 * @return Literal original source owner; no floating-to-quantized conversion.
 */
inline std::unique_ptr<TensorBase> control(Role role, int projection)
{
    // Keep both IQ3 payload families eligible for the existing fused pair.
    // The control odd coefficient is1, so only the target loses source precision.
    if(projection < 2 && (gateTarget(role)||upTarget(role)))
    {
        const uint16_t d=gateTarget(role)?0x2020u:0x3c20u;
        return iq3(d,0u);
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
    return {gateTarget(role) ? iq3() : control(role, 0),
            upTarget(role) ? iq3() : control(role, 1),
            role == Role::Down ? iq3() : control(role, 2)};
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
 * @brief Decode one integer directly from the original nine-bit grid indices and sign bytes.
 * @param b Original IQ3_S block.
 * @param p Logical position0..255.
 * @return Signed source integer without any scale.
 */
inline int integer(const IQ3_SBlock &b,int p)
{
    const int sub=p/32,lane=p%32,group=lane/4;
    const unsigned index=b.qs[sub*8+group]|(((b.qh[sub]>>group)&1u)<<8);
    const int magnitude=int((iq3s_grid[index]>>(8*(lane%4)))&255u);
    const unsigned signs=b.signs[sub*4+group/2]>>(4*(group%2));
    return signs&(1u<<(lane%4))?-magnitude:magnitude;
}
/**
 * @brief Independently evaluate original half times source odd coefficient.
 * @param d Original half base bits.
 * @param odd_coefficient Original1+2*nibble coefficient.
 * @return Exact finite FP32 original source factor; no derived-half publication.
 */
inline float factor(uint16_t d, int odd_coefficient)
{
    // Binary16 times odd integer at most31 is exactly representable in FP32.
    return float(double(fp16_to_fp32(d))*double(odd_coefficient));
}
/**
 * @brief Evaluate one source IQ3_S row with the consumed Q8 publication.
 * @param source Original superblock for this 256-value output row.
 * @param input Eight immutable Q8 blocks actually consumed by the operator.
 * @return Original-source contribution with existing single-dot rounding edges.
 * @details All witnesses have only one nonzero integer-dot block; other trees
 * contain +0, so this comparison does not prescribe a new reduction order.
 */
inline float iq3Dot(const IQ3_SBlock &source, const Q8_1Block *input)
{
    using namespace device_fp32_contract;
    float result=0.0f;
    for(int g=0;g<blocks;++g)
    {
        int dot=0;
        for(int i=0;i<32;++i) dot+=integer(source,g*32+i)*int(input[g].qs[i]);
        const int nibble=(source.scales[g/2]>>(4*(g%2)))&15;
        result=add(result,multiply(multiply(fp16_to_fp32(input[g].d),factor(source.d,1+2*nibble)),float(dot)));
    }
    return result;
}
/**
 * @brief Define source gate/up checkpoint values for one isolated role.
 * @param role IQ3_S role.
 * @param gate Output original-source gate row.
 * @param up Output original-source up row.
 */
inline void expectedGateUp(Role role, std::array<float,width> &gate,
    std::array<float,width> &up)
{
    gate.fill(gateTarget(role) ? 384.375f : upTarget(role) ? 132.0f : 128.0f);
    up.fill(upTarget(role) ? 384.375f : gateTarget(role) ? 33.0f/32.0f : 0.0f);
    if(role==Role::Down) { up[0]=1.0f; up[31]=127.0f; }
}
/**
 * @brief Define middle bytes from source projection values and unchanged policy.
 * @param role IQ3_S role.
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
 * @brief Close down using actual middle words while avoiding packed IQ3_S data.
 * @param role IQ3_S role.
 * @param source Original down tensor retained by the fixture.
 * @param actual_middle Actual publication consumed by this transaction's down.
 * @return Defining original-source output at route weight1.
 */
inline float expectedDown(Role role, const TensorBase &source, const Q8_1Block *actual_middle)
{
    using namespace device_fp32_contract;
    if(role==Role::Down)
        return iq3Dot(*static_cast<const IQ3_STensor&>(source).blocks(),actual_middle);
    // Literal Q8 down controls have exact scale1/grid1.
    int dot=0; for(int i=0;i<32;++i) dot+=actual_middle[0].qs[i];
    return multiply(fp16_to_fp32(actual_middle[0].d),float(dot));
}
} // namespace llaminar2::test::iq3s_source_witness
