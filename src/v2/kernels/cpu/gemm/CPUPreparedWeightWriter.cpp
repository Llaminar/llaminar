/**
 * @file CPUPreparedWeightWriter.cpp
 * @brief Ordered bounded source writes into one admitted final CPU weight owner.
 *
 * The reader owns original source identity and staging. This synchronous writer
 * authenticates exact rectangle coverage and publishes only a completed final
 * root. Failure is absorbing; unpublished partial roots retire by normal RAII.
 */
#include "CPUPreparedWeightWriter.h"
#include "CPUNativeVNNIGemmKernel.h"
#include "FloatingPointGemmKernel.h"
#include "tensors/TensorFactory.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/NativeWeightSourceRange.h"
#include <cstring>
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include <limits>
#include <cmath>
#include "kernels/common/DeviceHalfMetadataContract.h"
namespace llaminar2 {
namespace {
/** @return Checked product for exact native/final extents. */
size_t product(size_t a,size_t b) {
 if(a && b>std::numeric_limits<size_t>::max()/a)throw std::overflow_error("CPU final weight extent overflows");return a*b;
}
/**
 * @brief Preserve generic FP32Tensor::to_fp16 vector/tail choices across source windows.
 * @param value Original FP32 element in the logical materialized source selection.
 * @param index Index within that complete logical selection, never the current window.
 * @param count Complete logical selection extent used by the original tensor conversion.
 * @return The same ISA vector-RNE or scalar-tail bits as its original 4096-element chunks.
 * @details Register-only conversion avoids a decoded source shadow or per-window
 * vector tails. Model-owned half preparation uses its separate canonical contract.
 */
uint16_t genericTensorHalfBits(float value,size_t index,size_t count) {
 constexpr size_t chunk_size=4096;
 const size_t chunk_start=index/chunk_size*chunk_size;
 const size_t chunk_count=std::min(chunk_size,count-chunk_start);
 const size_t within=index-chunk_start;
#if defined(__AVX512F__)
 if(within<chunk_count/16*16) {
  const auto halves=_mm512_cvtps_ph(_mm512_set1_ps(value),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  return static_cast<uint16_t>(_mm_cvtsi128_si32(_mm256_castsi256_si128(halves)));
 }
#elif defined(__AVX2__) && defined(__F16C__)
 if(within<chunk_count/8*8)
  return static_cast<uint16_t>(_cvtss_sh(value,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
#else
 (void)chunk_count;(void)within;
#endif
 return fp32_to_fp16(value);
}
}
/** @brief Private unpublished final storage and next exact source coordinate. */
struct CPUPreparedWeightWriter::Impl {
 CPUPreparedWeightSource spec;
 std::optional<NativeWeightLoadStagingRequest> quote;
 CPUWeightStoragePlacement placement;
 cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
 std::shared_ptr<TensorBase> final;
 std::byte* final_bytes=nullptr;
 size_t native_group=0,native_group_bytes=0,row=0,column=0,matrix=0,logical_elements=0;
 size_t stripe_index=0,destination_extent=0,final_rows=0,final_columns=0;
 TensorType final_type=TensorType::FP32;
 bool failed=false,finished=false,native_packed=false,q8_conversion=false;
 /** @brief Validate original source and create its single final allocation. */
 Impl(CPUPreparedWeightSource source,CPUWeightStoragePlacement owner):spec(std::move(source)),placement(std::move(owner)) {
  const auto& g=spec.original;const auto& r=spec.range;
  if(spec.half_conversion!=CPUPreparedHalfConversion::GenericTensor &&
     spec.half_conversion!=CPUPreparedHalfConversion::ModelPreparedHalf)
   throw std::invalid_argument("CPU source has an unknown half-conversion contract");
  const bool model_half=spec.destination_precision==WeightPrecision::CONVERT_TO_FP16 &&
      spec.operand_precision==FloatingActivationOperandPrecision::FP16RNE && g.original_type!=TensorType::FP16;
  if((spec.half_conversion==CPUPreparedHalfConversion::ModelPreparedHalf)!=model_half)
   throw std::invalid_argument("Half conversion must retain its original sealed FP16RNE source preparation contract");
  if(g.original_shape.empty() || !r.rows || !r.columns || !spec.matrix_count ||
     !g.native_group_elements || !g.native_group_bytes ||
     (spec.consumer==CPUWeightConsumer::PreparedGemm && spec.matrix_count!=1))
   throw std::invalid_argument("CPU final writer requires exact original positive source geometry");
  if(spec.consumer==CPUWeightConsumer::PreparedGemm && g.original_shape.size()<2)
   throw std::invalid_argument("CPU GEMM source must be an actual matrix");
  const auto* original_format=native_vnni_formats::forQuantType(tensorTypeName(g.original_type));
  const bool original_floating=g.original_type==TensorType::FP32 || g.original_type==TensorType::FP16 || g.original_type==TensorType::BF16;
  // Integer tables (DeepSeek-V4 token-id hash routing) are one 4-byte value
  // per element, copied verbatim into a resident INT32 root and never a GEMM.
  const bool original_integer=g.original_type==TensorType::INT32;
  if(original_integer && (spec.consumer!=CPUWeightConsumer::ResidentTensor ||
                          spec.destination_precision!=WeightPrecision::NATIVE))
   throw std::invalid_argument("CPU INT32 tables are native resident tensors only");
  if(!original_format && !original_floating && !original_integer)
   throw std::invalid_argument("CPU source has no canonical native geometry");
  const size_t expected_group=original_format?(original_format->is_superblock?256:32):1;
  const size_t expected_bytes=original_format?cpu::native_vnni::native_block_bytes_for_format(original_format->codebook_id,original_format->is_superblock):
    (g.original_type==TensorType::FP32 || original_integer?sizeof(float):sizeof(uint16_t));
  if(g.native_group_elements!=expected_group || g.native_group_bytes!=expected_bytes || g.original_shape[0]%expected_group)
   throw std::invalid_argument("CPU source geometry differs from its original canonical native encoding");
  const size_t k=g.original_shape[0],n=g.original_shape.size()>1?g.original_shape[1]:1;
  size_t outer=1;for(size_t i=2;i<g.original_shape.size();++i)outer=product(outer,g.original_shape[i]);
  if(r.first_row>n || r.rows>n-r.first_row || r.first_column>k || r.columns>k-r.first_column ||
     r.original_expert>=outer || spec.matrix_count>outer-r.original_expert ||
     r.first_column%g.native_group_elements || r.columns%g.native_group_elements ||
     r.native_row_bytes!=product(r.columns/g.native_group_elements,g.native_group_bytes))
   throw std::invalid_argument("CPU final writer source range differs from original native geometry");
  if(spec.consumer==CPUWeightConsumer::ResidentTensor && g.original_shape.size()>2 &&
     (r.original_expert || r.selected_expert_ordinal || r.first_row || r.first_column || r.rows!=n || r.columns!=k || spec.matrix_count!=outer))
   throw std::invalid_argument("CPU resident outer population must preserve complete original shape");
  if(spec.assembly && spec.matrix_count!=1)
   throw std::invalid_argument("CPU source assembly requires one declared final matrix/vector");
  size_t maximum_stripe_bytes=0;
  forEachNativeWeightSourceStripe(g,r,spec.assembly,[&](const auto& stripe,size_t,size_t destination_column){
   maximum_stripe_bytes=std::max(maximum_stripe_bytes,stripe.nativeBytes());
   if(original_floating && spec.destination_precision==WeightPrecision::CONVERT_TO_INT8 &&
      (stripe.columns%32 || destination_column%32))
    throw std::invalid_argument("Q8_0 source stripes must preserve complete final quantizer groups");
  });
  final_rows=spec.assembly?spec.assembly->final_rows:r.rows;
  final_columns=spec.assembly?spec.assembly->final_columns:r.columns;
  native_group=g.native_group_elements;native_group_bytes=g.native_group_bytes;
  quote=quoteCPUWeightPreparation(g.original_type,final_rows,final_columns,spec.destination_precision,spec.consumer,spec.operand_precision);
  if(quote && spec.assembly) {
   if(maximum_stripe_bytes%quote->native_bytes_per_unit)
    throw std::invalid_argument("CPU source stripe splits its consumer's indivisible native unit");
   const size_t maximum_units=maximum_stripe_bytes/quote->native_bytes_per_unit;
   if(!maximum_units || maximum_units>quote->maximum_units)
    throw std::invalid_argument("CPU source stripes disagree with their final consumer geometry");
   // Stripes are serialized through the same input/scratch arena. Its largest
   // useful stripe, rather than their sum, owns the simultaneous setup peak.
   quote->maximum_units=maximum_units;
  }
  const auto* fmt=native_vnni_formats::forQuantType(tensorTypeName(g.original_type));
  q8_conversion=!fmt && spec.destination_precision==WeightPrecision::CONVERT_TO_INT8;
  const auto* prepared_format=q8_conversion?native_vnni_formats::forQuantType("Q8_0"):fmt;
  native_packed=prepared_format && (spec.destination_precision==WeightPrecision::NATIVE || q8_conversion) && spec.consumer==CPUWeightConsumer::PreparedGemm;
  if(native_packed) {
   if(final_rows>size_t(std::numeric_limits<int>::max()) || final_columns>size_t(std::numeric_limits<int>::max()))throw std::overflow_error("CPU GEMM final dimensions exceed native ABI");
   cpu::native_vnni::initializeNativeWindowOutput(*prepared_format,int(final_rows),int(final_columns),placement,packed);
   return;
  }
  final_type=g.original_type;
  switch(spec.destination_precision) {
   case WeightPrecision::NATIVE:break;
   case WeightPrecision::CONVERT_TO_FP32:final_type=TensorType::FP32;break;
   case WeightPrecision::CONVERT_TO_FP16:final_type=TensorType::FP16;break;
   case WeightPrecision::CONVERT_TO_BF16:final_type=TensorType::BF16;break;
   case WeightPrecision::CONVERT_TO_INT8:final_type=TensorType::Q8_0;break;
  }
  std::vector<size_t> shape;
  if(g.original_shape.size()==1)shape={final_columns};
  else if(g.original_shape.size()==2 || spec.consumer==CPUWeightConsumer::PreparedGemm)shape={final_rows,final_columns};
  else shape=g.original_shape;
  // Q8_1 is a matrix-only runtime/activation tensor. Its constructor and
  // row/view operations do not represent an original higher-rank parent.
  // Reject before allocating final bytes instead of flattening that parent.
  if(final_type==TensorType::Q8_1 && shape.size()!=2)
   throw std::invalid_argument("Q8_1 final CPU resident storage supports only original matrices; non-matrix source shape is unsupported");
  const size_t elements=product(product(final_rows,final_columns),spec.matrix_count);
  logical_elements=elements;
  switch(final_type) {
   case TensorType::FP32: {auto data=placement.allocate<float>(elements);final_bytes=reinterpret_cast<std::byte*>(data.data());final=std::make_shared<FP32Tensor>(shape,std::move(data),placement.allocatorDevice());break;}
   case TensorType::FP16: {auto data=placement.allocate<uint16_t>(elements);final_bytes=reinterpret_cast<std::byte*>(data.data());final=std::make_shared<FP16Tensor>(shape,std::move(data),placement.allocatorDevice());break;}
   case TensorType::BF16: {auto data=placement.allocate<uint16_t>(elements);final_bytes=reinterpret_cast<std::byte*>(data.data());final=std::make_shared<BF16Tensor>(shape,std::move(data),placement.allocatorDevice());break;}
   case TensorType::INT32: {auto data=placement.allocate<int32_t>(elements);final_bytes=reinterpret_cast<std::byte*>(data.data());final=std::make_shared<INT32Tensor>(shape,std::move(data),placement.allocatorDevice());break;}
   default: {const size_t row_bytes=q8_conversion?product(final_columns/32,sizeof(Q8_0Block)):product(final_columns/native_group,native_group_bytes);
    auto data=placement.allocate<uint8_t>(product(product(final_rows,row_bytes),spec.matrix_count));final_bytes=reinterpret_cast<std::byte*>(data.data());final=TensorFactory::adoptQuantizedFinalStorage(final_type,shape,std::move(data),placement.allocatorDevice());break;}
  }
 }
 /** @return Current original stripe; rectangular sources retain their original range object. */
 const NativeWeightSourceWindow& stripe() const noexcept {
  return spec.assembly?spec.assembly->segments[stripe_index]:spec.range;
 }
 /** @return Actual useful final row origin for the next original window. */
 size_t destinationRow() const noexcept {
  return row+(spec.assembly && spec.assembly->axis==NativeWeightSourceAssembly::Axis::Rows?destination_extent:0);
 }
 /** @return Actual useful final K origin for the next original window. */
 size_t destinationColumn() const noexcept {
  return column+(spec.assembly && spec.assembly->axis==NativeWeightSourceAssembly::Axis::Columns?destination_extent:0);
 }
 /** @return Final logical element index for one window word, including its genuine row stride. */
 size_t destinationIndex(size_t index,const NativeWeightSourceWindow& w) const noexcept {
  return (matrix*final_rows+destinationRow()+index/w.columns)*final_columns+destinationColumn()+index%w.columns;
 }
 /** @brief Check next exact original rectangle before mutation. @param w Actual proposed original source rectangle. */
 void check(const NativeWeightSourceWindow& w) const {
  if(failed || finished || matrix>=spec.matrix_count)
   throw std::invalid_argument("CPU final writer has no remaining original source coverage");
  const auto& r=stripe();
  if(w.original_expert!=r.original_expert+matrix ||
     w.selected_expert_ordinal!=r.selected_expert_ordinal+matrix || w.first_row!=r.first_row+row ||
     w.first_column!=r.first_column+column || !w.rows || !w.columns || w.rows>r.rows-row ||
     w.columns>r.columns-column || w.columns%native_group ||
     w.native_row_bytes!=product(w.columns/native_group,native_group_bytes) ||
     (w.rows>1 && (column || w.columns!=r.columns)))
   throw std::invalid_argument("CPU final writer rejects foreign, duplicate, gapped or unaligned source window");
 }
 /** @brief Advance original coverage only after synchronous writes. @param w Exactly the successfully written rectangle. */
 void advance(const NativeWeightSourceWindow& w) {
  const auto& current=stripe();
  column+=w.columns;
  if(column==current.columns) {column=0;row+=w.rows;}
  if(row!=current.rows)return;
  row=0;
  if(spec.assembly) {
   destination_extent+=spec.assembly->axis==NativeWeightSourceAssembly::Axis::Rows?current.rows:current.columns;
   if(++stripe_index<spec.assembly->segments.size())return;
   stripe_index=0;destination_extent=0;
  }
  ++matrix;
 }

};
CPUPreparedWeightWriter::CPUPreparedWeightWriter(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}
CPUPreparedWeightWriter::~CPUPreparedWeightWriter()=default;
std::unique_ptr<CPUPreparedWeightWriter> CPUPreparedWeightWriter::begin(CPUPreparedWeightSource source,CPUWeightStoragePlacement placement) {
 return std::unique_ptr<CPUPreparedWeightWriter>(new CPUPreparedWeightWriter(std::make_unique<Impl>(std::move(source),std::move(placement))));
}
const CPUPreparedWeightSource& CPUPreparedWeightWriter::source() const & noexcept{return impl_->spec;}
std::optional<NativeWeightLoadStagingRequest> CPUPreparedWeightWriter::stagingRequest() const{return impl_->quote;}
void CPUPreparedWeightWriter::append(const NativeWeightSourceWindow& w,std::span<const std::byte> native,std::span<std::byte> scratch) {
 try {
  impl_->check(w);
  if(native.size()!=w.nativeBytes() || !impl_->quote)throw std::invalid_argument("CPU append requires exact original source bytes and transient consumer");
  if(impl_->q8_conversion) {
   const size_t count=product(w.rows,w.columns),units=count/32;
   if(w.columns%32 || impl_->destinationColumn()%32 || scratch.size()<product(units,impl_->quote->scratch_bytes_per_unit)+impl_->quote->fixed_scratch_bytes || reinterpret_cast<uintptr_t>(scratch.data())%alignof(float))
    throw std::invalid_argument("Q8_0 conversion requires admitted whole-group source and scratch");
   auto* values=reinterpret_cast<float*>(scratch.data());
   for(size_t i=0;i<count;++i) {
    if(impl_->spec.original.original_type==TensorType::FP32)std::memcpy(values+i,native.data()+i*sizeof(float),sizeof(float));
    else {uint16_t bits;std::memcpy(&bits,native.data()+i*sizeof(uint16_t),sizeof(bits));
     values[i]=impl_->spec.original.original_type==TensorType::FP16?fp16_to_fp32(bits):simd::bf16_to_fp32(bits);}
   }
   const size_t encoded_bytes=product(units,sizeof(Q8_0Block));
   auto* encoded=impl_->native_packed?scratch.data()+count*sizeof(float):nullptr;
   for(size_t row=0;row<w.rows;++row) {
    auto* output=impl_->native_packed?reinterpret_cast<Q8_0Block*>(encoded)+row*w.columns/32:
      reinterpret_cast<Q8_0Block*>(impl_->final_bytes)+impl_->destinationIndex(row*w.columns,w)/32;
    Q8_0Tensor::quantize_fp32_row(values+row*w.columns,output,w.columns);
   }
   if(impl_->native_packed) {
    const size_t pack_offset=(count*sizeof(float)+encoded_bytes+63)&~size_t(63);
    const auto* format=native_vnni_formats::forQuantType("Q8_0");
    TensorFactory::visitNativeWeightWindow(TensorType::Q8_0,w.rows,w.columns,{encoded,encoded_bytes},[&](const TensorBase& source){
     auto* codec=dynamic_cast<const IINT8Unpackable*>(&source);
     cpu::native_vnni::appendNativeWeightGroups(*codec,*format,int(w.columns),0,0,int(w.rows),int(w.columns/32),
       int(impl_->destinationRow()),int(impl_->destinationColumn()/32),scratch.subspan(pack_offset),impl_->packed);
    });
   }
   impl_->advance(w);return;
  }
  if(!impl_->native_packed) {
   const size_t offset=impl_->destinationIndex(0,w);
   const bool contiguous=w.rows==1 || w.columns==impl_->final_columns;
   const size_t count=product(w.rows,w.columns);
   const bool original_float=impl_->spec.original.original_type==TensorType::FP32 ||
      impl_->spec.original.original_type==TensorType::FP16 || impl_->spec.original.original_type==TensorType::BF16;
   float* decoded=nullptr;
   if(!original_float) {
    if(impl_->final_type==TensorType::FP32) {if(contiguous)decoded=reinterpret_cast<float*>(impl_->final_bytes)+offset;}
    else {
     if(scratch.size()<product(count,sizeof(float)) || reinterpret_cast<uintptr_t>(scratch.data())%alignof(float))
      throw std::invalid_argument("Native floating conversion lacks admitted complete-window scratch");
     decoded=reinterpret_cast<float*>(scratch.data());
    }
    TensorFactory::visitNativeWeightWindow(impl_->spec.original.original_type,w.rows,w.columns,native,
      [&](const TensorBase& source){
       if(impl_->spec.half_conversion!=CPUPreparedHalfConversion::ModelPreparedHalf) {
        if(impl_->final_type==TensorType::FP32 && !contiguous) {
         const auto* decoder=dynamic_cast<const ITensorGemmTileDataProvider*>(&source);
         if(!decoder || !decoder->block_size() || decoder->block_size()>256 ||
            decoder->decoder_rows()!=w.rows || decoder->decoder_cols()!=w.columns || w.columns%decoder->block_size())
          throw std::invalid_argument("Column source decode requires complete original native blocks");
         const size_t width=decoder->block_size();
         // Each complete source block writes its exact final row subrange;
         // no full-window FP32 shadow or row-sized stack workspace is needed.
#pragma omp parallel for schedule(static)
         for(size_t row=0;row<w.rows;++row)
          for(size_t first=0;first<w.columns;first+=width)
           decoder->decode_block_at(row,first/width,reinterpret_cast<float*>(impl_->final_bytes)+impl_->destinationIndex(row*w.columns,w)+first);
        } else source.to_fp32(decoded);
        return;
       }
       // The old model override consumes the original block decoder, whose
       // source factor/arithmetic order can differ from a generic row decode.
       const auto* decoder=dynamic_cast<const ITensorGemmTileDataProvider*>(&source);
       if(!decoder || !decoder->block_size() || decoder->block_size()>256 ||
          decoder->decoder_rows()!=w.rows || decoder->decoder_cols()!=w.columns || w.columns%decoder->block_size())
        throw std::invalid_argument("Prepared half window requires its original complete source decoder blocks");
       const size_t width=decoder->block_size();
#pragma omp parallel for schedule(static)
       for(size_t row=0;row<w.rows;++row)
        for(size_t first=0;first<w.columns;first+=width)
         decoder->decode_block_at(row,first/width,decoded+row*w.columns+first);
      });
   }
   if(!original_float && impl_->final_type==TensorType::FP32) {impl_->advance(w);return;}
   unsigned invalid=0;
#pragma omp parallel for schedule(static) reduction(|:invalid)
   for(size_t i=0;i<count;++i) {
    const size_t destination=contiguous?offset+i:impl_->destinationIndex(i,w);
    float value;
    if(decoded)value=decoded[i];
    else if(impl_->spec.original.original_type==TensorType::FP32)std::memcpy(&value,native.data()+i*sizeof(float),sizeof(float));
    else {uint16_t bits;std::memcpy(&bits,native.data()+i*sizeof(uint16_t),sizeof(bits));
     value=impl_->spec.original.original_type==TensorType::FP16?fp16_to_fp32(bits):simd::bf16_to_fp32(bits);}
    if(impl_->final_type==TensorType::FP32)reinterpret_cast<float*>(impl_->final_bytes)[destination]=value;
    else if(impl_->final_type==TensorType::FP16) {
     uint16_t bits;
     if(impl_->spec.half_conversion==CPUPreparedHalfConversion::ModelPreparedHalf) {
      bits=canonicalPreparedHalfBits(value);
      invalid|=!std::isfinite(value) || (bits&0x7c00u)==0x7c00u;
     } else bits=impl_->spec.original.original_type==TensorType::FP32?
        genericTensorHalfBits(value,destination,impl_->logical_elements):fp32_to_fp16(value);
     reinterpret_cast<uint16_t*>(impl_->final_bytes)[destination]=bits;
    } else reinterpret_cast<uint16_t*>(impl_->final_bytes)[destination]=simd::fp32_to_bf16(value);
   }
   if(invalid)throw std::invalid_argument("Prepared half weights decoded a nonfinite or half-overflowing source value");
   impl_->advance(w);return;
  }
  const auto* format=native_vnni_formats::forQuantType(tensorTypeName(impl_->spec.original.original_type));
  TensorFactory::visitNativeWeightWindow(impl_->spec.original.original_type,w.rows,w.columns,native,[&](const TensorBase& source){
   auto* codec=dynamic_cast<const IINT8Unpackable*>(&source);
   if(!codec)throw std::logic_error("Original native window has no canonical codec");
   cpu::native_vnni::appendNativeWeightGroups(*codec,*format,int(w.columns),0,0,int(w.rows),int(w.columns/32),
    int(impl_->destinationRow()),int(impl_->destinationColumn()/32),scratch,impl_->packed);
  });
  impl_->advance(w);
 }catch(...) {impl_->failed=true;throw;}
}
void CPUPreparedWeightWriter::readDirect(const NativeWeightSourceWindow& w,const std::function<void(std::span<std::byte>)>& read) {
 try {
  impl_->check(w);
  if(impl_->quote || !impl_->final_bytes || !read)throw std::invalid_argument("CPU direct write requires unchanged genuine final storage");
  if(w.rows>1 && w.columns!=impl_->final_columns)
   throw std::invalid_argument("Direct CPU column stripes require one original row per final contiguous write");
  const size_t offset=product(impl_->destinationIndex(0,w)/impl_->native_group,impl_->native_group_bytes);
  read({impl_->final_bytes+offset,w.nativeBytes()});impl_->advance(w);
 }catch(...) {impl_->failed=true;throw;}
}
CPUPreparedWeightResult CPUPreparedWeightWriter::finish() {
 if(impl_->failed || impl_->finished || impl_->matrix!=impl_->spec.matrix_count || impl_->row || impl_->column) {
  impl_->failed=true;
  throw std::logic_error("CPU final weights cannot publish incomplete, failed or already-finished coverage");
 }
 CPUPreparedWeightResult result;
 if(impl_->native_packed)result.kernel=std::make_shared<cpu::native_vnni::CPUNativeVNNIGemmKernel>(std::move(impl_->packed),impl_->spec.numerical_policy);
 else {
  result.final_tensor=impl_->final;
  if(impl_->spec.consumer==CPUWeightConsumer::PreparedGemm)
   result.kernel=gemm::FloatingPointGemmKernel::fromFinalCPUStorage(result.final_tensor,impl_->spec.numerical_policy,impl_->spec.operand_precision);
 }
 impl_->finished=true;return result;
}
}
