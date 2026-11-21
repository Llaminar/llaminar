/**
 * @file Test__CPUQwenSparseExactBasis.cpp
 * @brief Actual QSA TQ basis/query-alias bytes against an independent ordered-FMA oracle.
 *
 * Nonsymmetric orthogonal matrices distinguish R from R-transpose. The oracle
 * independently unpacks original codes, rounds radius-first products, executes
 * declared eight/sixteen-chain scores and ordered value/transpose FMAs. It never
 * calls a production decoder, dot helper or attention entrypoint to obtain gold.
 * Reverse-within-eight and transposed-index references must differ, so a shared
 * wrong/zero value implementation cannot authenticate itself through M1 replay.
 */
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/cpu/attention/CPUQwenSparseFP32Program.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
namespace llaminar2::qwen4exp::test
{
namespace
{
/** @brief Retain one independent diagnostic FP32 multiply.
 * @param a Original operand; @param b Original operand; @return Rounded product.
 */
float multiply(float a,float b){volatile float value=a*b;return value;}
/** @brief Retain one independent diagnostic FP32 addition.
 * @param a Original operand; @param b Original operand; @return Rounded sum.
 */
float add(float a,float b){volatile float value=a+b;return value;}
/** @brief Retain one independent diagnostic FP32 subtraction.
 * @param a Original operand; @param b Original operand; @return Rounded difference.
 */
float subtract(float a,float b){volatile float value=a-b;return value;}
/** @brief Retain one independent diagnostic FP32 division.
 * @param a Original numerator; @param b Original denominator; @return Rounded quotient.
 */
float divide(float a,float b){volatile float value=a/b;return value;}
/** @brief Read one original scalar word without typed packed-source aliasing.
 * @param p Original bytes; @return Original FP32 word.
 */
float word(const unsigned char*p){float value;std::memcpy(&value,p,4);return value;}
/** @brief Initialize one original metadata word only during fixture setup.
 * @param p Owned original bytes; @param value Original stored word.
 */
void put(unsigned char*p,float value){std::memcpy(p,&value,4);}
/** @brief Independent native-ISA QSA chain/fold reference, never a closest-output oracle.
 * @param a Original first operand; @param b Original second operand.
 * @param width Declared eight/sixteen-chain program; @return Exact declared score.
 */
float dot(const float*a,const float*b,unsigned width)
{
    std::array<float,16>lanes{};for(unsigned c=0;c<256;++c)lanes[c%width]=std::fma(a[c],b[c],lanes[c%width]);
    for(unsigned step=width/2;step;step/=2)for(unsigned lane=0;lane<step;++lane)lanes[lane]=add(lanes[lane],lanes[lane+step]);
    return lanes[0];
}
/** @brief Preserve the original caller's OpenMP settings outside this exact leaf/caller proof. */
struct TeamPolicy final
{
    int threads=omp_get_max_threads(),dynamic=omp_get_dynamic();
    /** @brief Admit one fixed caller thread without a host affinity or budget change. */
    TeamPolicy(){omp_set_dynamic(0);omp_set_num_threads(1);}
    /** @brief Restore the original execution policy on every test exit. */
    ~TeamPolicy(){omp_set_num_threads(threads);omp_set_dynamic(dynamic);}
};
/** @brief Own original native codes/tables/nonsymmetric bases and complete guarded Q/output storage. */
struct Fixture final
{
    static constexpr int guard=8,prefix=2;
    TensorType codec;int rows,tokens;std::size_t bytes;QwenSparseReplay replay;
    std::vector<std::uint32_t>keys,values;std::vector<float>keyBook,valueBook,keyMatrix,valueMatrix,query;
    std::vector<std::int32_t>selected,counts;QwenSparseAttention p;
    /** @brief Construct independently encoded original TQ banks and nonsymmetric orthogonal bases.
     * @param encoding Original TQ4/TQ8 physical codec; @param rowCount Original grouped request rows.
     */
    Fixture(TensorType encoding,int rowCount):codec(encoding),rows(rowCount),tokens(prefix+rows),bytes(codec==TensorType::TQ4?136:264),
        replay{prefix,rows,0},keys(std::size_t(tokens)*bytes/4),values(keys.size()),keyBook(codec==TensorType::TQ4?16:256),
        valueBook(keyBook.size()),keyMatrix(256*256),valueMatrix(keyMatrix.size()),query(std::size_t(rows)*256+guard+9,919.25f),
        selected(std::size_t(rows)*QwenSparseSelected,-987),counts(rows)
    {
        for(unsigned i=0;i<keyBook.size();++i)
        {keyBook[i]=float(int((i*37)%127)-63)*0.0156251f;valueBook[i]=float(int((i*53+17)%127)-63)*0.017321f;}
        for(unsigned r=0;r<256;++r)for(unsigned c=0;c<256;++c)
        {
            keyMatrix[r*256+c]=(std::popcount(((r+1)&255)&c)%2?-1.0f:1.0f)*0.0625f;
            valueMatrix[r*256+c]=(std::popcount(((r*5+3)&255)&c)%2?-1.0f:1.0f)*0.0625f;
        }
        for(int token=0;token<tokens;++token)for(bool key:{true,false})
        {
            auto*data=reinterpret_cast<unsigned char*>((key?keys:values).data())+std::size_t(token)*bytes;
            put(data,1.25f+float(token)*0.03125f);put(data+4,0.7f+float(token)*0.019f);
            for(unsigned c=0;c<256;++c)
            {
                const unsigned code=((c*c*13+(c>>4)*7+unsigned(token)*5+c*unsigned(token)*3)^(c>>3)^unsigned(token*11))&(codec==TensorType::TQ4?15:255);
                if(codec==TensorType::TQ8)data[8+c]=std::uint8_t(code);
                else
                {
                    const unsigned bit=c*3,byte=bit/8,shift=bit%8;data[8+byte]|=std::uint8_t((code&7)<<shift);
                    if(shift>5)data[9+byte]|=std::uint8_t((code&7)>>(8-shift));
                    data[104+c/8]|=std::uint8_t((code>>3)<<(c%8));
                }
            }
        }
        for(int row=0;row<rows;++row)
        {
            counts[row]=prefix+row+1;for(int i=0;i<counts[row];++i)selected[std::size_t(row)*QwenSparseSelected+i]=i;
            for(int c=0;c<256;++c)query[guard+std::size_t(row)*256+c]=8.0f+float((c*17+row*13)%127)*0.0078125f;
        }
        const auto bank=[&](const auto&storage,const auto&book,const auto&matrix)
        {return QwenSparseNativeBank{.data=storage.data(),.token_stride_bytes=bytes,.head_stride_bytes=bytes,.codec=codec,
            .centroids=book.data(),.rotations={matrix.data(),nullptr}};};
        p={.replay=&replay,.query=query.data()+guard,.kv={.keys=bank(keys,keyBook,keyMatrix),.values=bank(values,valueBook,valueMatrix),
            .addressing=QwenSparseKVAddressing::FixedLogicalView,.logical_view={.logical_row_origin=tokens-2,.physical_row_capacity=tokens}},
            .selected=selected.data(),.selected_count=counts.data(),.row_capacity=rows,.token_capacity=tokens,.query_heads=1,.kv_heads=1};
    }
    /** @brief Independently unpack one original radius-first native coordinate.
     * @param key Original key/value radial role; @param physical Original physical token.
     * @param c Original coordinate; @return Exact separately rounded native-basis value.
     */
    float decode(bool key,int physical,unsigned c)const
    {
        const auto*data=reinterpret_cast<const unsigned char*>((key?keys:values).data())+std::size_t(physical)*bytes;
        unsigned code=0;if(codec==TensorType::TQ8)code=data[8+c];
        else{const unsigned bit=c*3,byte=bit/8,shift=bit%8;unsigned low=data[8+byte]>>shift;if(shift>5)low|=unsigned(data[9+byte])<<(8-shift);code=(low&7)|(((data[104+c/8]>>(c%8))&1)<<3);}
        const auto radius=word(data+(key?0:4));return multiply((key?keyBook:valueBook)[code],multiply(radius,0.0625f));
    }
};
/** @brief Three independently computed spatial programs; only ordered is the declared oracle. */
struct Gold{std::vector<float>ordered,reverseWithinEight,wrongTranspose;};
/** @brief Compute all original selected operations without invoking production math.
 * @param f Owned original data/bases; @param width Declared eight/sixteen-chain dot.
 * @return Correct spatial bytes and two explicit wrong arithmetic/indexing controls.
 */
Gold oracle(const Fixture&f,unsigned width)
{
    Gold out;for(auto*p:{&out.ordered,&out.reverseWithinEight,&out.wrongTranspose})p->assign(f.query.size(),919.25f);
    for(int row=0;row<f.rows;++row)
    {
        std::array<float,256>rotated{},decoded{},weighted{};std::vector<float>logits(f.counts[row]);
        const auto*query=f.query.data()+Fixture::guard+std::size_t(row)*256;
        for(int c=0;c<256;++c)rotated[c]=dot(f.keyMatrix.data()+c*256,query,width);
        float maximum=-INFINITY;
        for(int i=0;i<f.counts[row];++i)
        {
            const auto logical=f.selected[std::size_t(row)*QwenSparseSelected+i];const int physical=(f.tokens-2+logical)%f.tokens;
            for(unsigned c=0;c<256;++c)decoded[c]=f.decode(true,physical,c);
            logits[i]=multiply(dot(rotated.data(),decoded.data(),width),0.0625f);maximum=std::max(maximum,logits[i]);
        }
        float denominator=0;for(auto&value:logits){value=std::exp(subtract(value,maximum));denominator=add(denominator,value);}
        for(int i=0;i<f.counts[row];++i)
        {
            const auto logical=f.selected[std::size_t(row)*QwenSparseSelected+i];const int physical=(f.tokens-2+logical)%f.tokens;
            const auto probability=divide(logits[i],denominator);
            for(unsigned c=0;c<256;++c)weighted[c]=std::fma(f.decode(false,physical,c),probability,weighted[c]);
        }
        for(unsigned d=0;d<256;++d)
        {
            float ordered=0,reverse=0,transpose=0;
            for(unsigned c=0;c<256;++c)
            {ordered=std::fma(f.valueMatrix[c*256+d],weighted[c],ordered);transpose=std::fma(f.valueMatrix[d*256+c],weighted[c],transpose);}
            for(unsigned base=0;base<256;base+=8)for(int c=int(base)+7;c>=int(base);--c)reverse=std::fma(f.valueMatrix[unsigned(c)*256+d],weighted[unsigned(c)],reverse);
            const auto offset=Fixture::guard+std::size_t(row)*256+d;out.ordered[offset]=ordered;out.reverseWithinEight[offset]=reverse;out.wrongTranspose[offset]=transpose;
        }
    }
    return out;
}
/** @brief Compare every result and guard word, rejecting nonfinite/no-op/partial output.
 * @param actual Complete actual guarded storage; @param expected Independent ordered gold.
 */
void authenticate(const std::vector<float>&actual,const std::vector<float>&expected)
{
    ASSERT_EQ(actual.size(),expected.size());for(std::size_t i=0;i<actual.size();++i)
    {ASSERT_TRUE(std::isfinite(actual[i]));ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(expected[i]))<<"word="<<i;}
}
} // namespace
/** @test Actual nonsymmetric TQ4/TQ8 bases retain independent ordered bytes, including complete query/output alias. */
TEST(CPUQwenSparseExactBasis,OriginalNonsymmetricBasisAndQueryAliasMatchOrderedFMABytes)
{
    const auto isa=activeISALevel();::llaminar2::cpu::qsa_native::requireNativeProgram(isa);TeamPolicy team;
    for(auto codec:{TensorType::TQ4,TensorType::TQ8})for(int rows:{1,3,16})
    {
        SCOPED_TRACE(::testing::Message()<<"codec="<<int(codec)<<" rows="<<rows);
        Fixture f(codec,rows);ASSERT_NE(f.keyMatrix[1],f.keyMatrix[256]);ASSERT_NE(f.valueMatrix[1],f.valueMatrix[256]);
        const auto gold=oracle(f,isa==ISALevel::AVX512?16:8);
        ASSERT_NE(std::memcmp(gold.ordered.data(),gold.reverseWithinEight.data(),gold.ordered.size()*sizeof(float)),0);
        ASSERT_NE(std::memcmp(gold.ordered.data(),gold.wrongTranspose.data(),gold.ordered.size()*sizeof(float)),0);
        const auto keys=f.keys,values=f.values;const auto query=f.query;const auto keyMatrix=f.keyMatrix,valueMatrix=f.valueMatrix;
        const auto keyBook=f.keyBook,valueBook=f.valueBook;const auto selected=f.selected,counts=f.counts;const auto replay=f.replay;
        for(bool alias:{false,true})
        {
            SCOPED_TRACE(alias?"exact-query-alias":"disjoint-original-query");
            auto actual=alias?f.query:gold.ordered;
            for(std::size_t i=Fixture::guard;i<Fixture::guard+std::size_t(rows)*256;++i)
            {
                ASSERT_TRUE(std::isfinite(gold.ordered[i]));
                if(alias)ASSERT_NE(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(gold.ordered[i]));
                else actual[i]=std::bit_cast<float>(std::bit_cast<std::uint32_t>(gold.ordered[i])^0x80000000u);
            }
            auto p=f.p;p.output=actual.data()+Fixture::guard;if(alias)p.query=p.output;
            cpuQwenSparseAttention(p);authenticate(actual,gold.ordered);
        }
        EXPECT_EQ(f.keys,keys);EXPECT_EQ(f.values,values);EXPECT_EQ(f.query,query);EXPECT_EQ(f.keyMatrix,keyMatrix);EXPECT_EQ(f.valueMatrix,valueMatrix);
        EXPECT_EQ(f.keyBook,keyBook);EXPECT_EQ(f.valueBook,valueBook);EXPECT_EQ(f.selected,selected);EXPECT_EQ(f.counts,counts);
        EXPECT_EQ(f.replay.prefix_tokens,replay.prefix_tokens);EXPECT_EQ(f.replay.live_rows,replay.live_rows);EXPECT_EQ(f.replay.position_origin,replay.position_origin);
    }
}
} // namespace llaminar2::qwen4exp::test
