/**
 * @file CPUNativeWeightWindowContract.inc
 * @brief Actual all-format bounded CPU packing and final writer byte contracts.
 *
 * Included inside the existing canonical native packing fixture namespace.
 * The oracle remains the existing complete canonical pack, while the writer
 * receives exact native strips with nonzero source origin and partial N64.
 */
TEST(NativeWindowPackingContract, SameCanonicalBytesAcrossAllFormats) {
 for (const auto& entry:native_quantised_formats::kAllSourceFormats) {
  SCOPED_TRACE(std::string(entry.quant_type));
  auto source=createSourceTensor(entry.quant_type,{67,512});
  auto* native=dynamic_cast<const IINT8Unpackable*>(source.get()); ASSERT_NE(native,nullptr);
  const auto& fmt=*native->vnniFormatInfo();
  CPUNativeQuantisedPackedWeights expected,actual;
  ASSERT_TRUE(packWeightsCPUNativeQuantised(source.get(),expected,1,66));
  initializeNativeWindowOutput(fmt,65,512,CPUWeightStoragePlacement::local(),actual);
  const auto scratchLayout=NativePackGroupScratch::checked(fmt);
  AlignedVector<std::byte> scratch(size_t(65)*(fmt.is_superblock?2:16)*scratchLayout.bytes);
  for(int row=0;row<65;++row) for(int block=0;block<16;block+=(fmt.is_superblock?8:1))
   appendNativeWeightGroups(*native,fmt,512,row+1,block,1,fmt.is_superblock?8:1,row,block,
    std::span<std::byte>(scratch.data(),scratch.size()),actual);
  ASSERT_EQ(expected.native_interleaved.size(),actual.native_interleaved.size());
  EXPECT_EQ(0,std::memcmp(expected.native_interleaved.data(),actual.native_interleaved.data(),actual.native_interleaved.size()));
 }
}
TEST(NativeWindowWriterContract, SameFinalNativeBytesAcrossAllFormats) {
 for(const auto& entry:native_quantised_formats::kAllSourceFormats) {
  SCOPED_TRACE(std::string(entry.quant_type));
  auto source=createSourceTensor(entry.quant_type,{67,512});
  auto* codec=dynamic_cast<const IINT8Unpackable*>(source.get());const auto& fmt=*codec->vnniFormatInfo();
  const size_t group=fmt.is_superblock?256:32;
  const size_t groupBytes=native_block_bytes_for_format(fmt.codebook_id,fmt.is_superblock);
  const size_t rowBytes=512/group*groupBytes;
  CPUPreparedWeightSource spec{.original={.original_shape={512,67},.original_type=source->native_type(),.native_group_elements=group,.native_group_bytes=groupBytes},
    .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=1,.rows=65,.first_column=0,.columns=512,.native_row_bytes=rowBytes}};
  auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
  const auto quote=writer->stagingRequest();ASSERT_TRUE(quote);
  AlignedVector<std::byte> scratch(quote->scratch_bytes_per_unit);
  for(size_t row=1;row<66;++row)for(size_t column=0;column<512;column+=group) {
   NativeWeightSourceWindow window{.original_expert=0,.selected_expert_ordinal=0,.first_row=row,.rows=1,.first_column=column,.columns=group,.native_row_bytes=groupBytes};
   const auto* bytes=static_cast<const std::byte*>(source->raw_data())+row*rowBytes+column/group*groupBytes;
   writer->append(window,{bytes,groupBytes},{scratch.data(),scratch.size()});
  }
  auto result=writer->finish();ASSERT_NE(result.kernel,nullptr);ASSERT_EQ(result.final_tensor,nullptr);
  auto* actual=result.kernel->exportCPUNativeQuantisedPackedWeights();ASSERT_NE(actual,nullptr);
  CPUNativeQuantisedPackedWeights expected;ASSERT_TRUE(packWeightsCPUNativeQuantised(source.get(),expected,1,66));
  ASSERT_EQ(actual->native_interleaved.size(),expected.native_interleaved.size());
  EXPECT_EQ(0,std::memcmp(actual->native_interleaved.data(),expected.native_interleaved.data(),expected.native_interleaved.size()));
  // A second genuine window geometry exercises concurrent disjoint chunk/K
  // tasks instead of only the minimum single-group reader budget.
  {
   ScopedOMPThreadCount threads(2);
   auto wide=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
   const size_t units=65*(512/group);
   AlignedVector<std::byte> wide_scratch(units*quote->scratch_bytes_per_unit);
   NativeWeightSourceWindow window{.original_expert=0,.selected_expert_ordinal=0,.first_row=1,.rows=65,.first_column=0,.columns=512,.native_row_bytes=rowBytes};
   wide->append(window,{static_cast<const std::byte*>(source->raw_data())+rowBytes,65*rowBytes},
      {wide_scratch.data(),wide_scratch.size()});
   auto wide_result=wide->finish();ASSERT_NE(wide_result.kernel,nullptr);
   const auto* wide_packed=wide_result.kernel->exportCPUNativeQuantisedPackedWeights();ASSERT_NE(wide_packed,nullptr);
   ASSERT_EQ(wide_packed->native_interleaved.size(),expected.native_interleaved.size());
   EXPECT_EQ(0,std::memcmp(wide_packed->native_interleaved.data(),expected.native_interleaved.data(),expected.native_interleaved.size()));
  }
  EXPECT_THROW(writer->finish(),std::logic_error);
 }
}

TEST(NativeWindowWriterContract, IncompleteAndForeignCoverageCannotPublish)
{
    CPUPreparedWeightSource spec{
        .original={.original_shape={64,2},.original_type=TensorType::Q4_0,.native_group_elements=32,.native_group_bytes=sizeof(Q4_0Block)},
        .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=2,.first_column=0,.columns=64,.native_row_bytes=2*sizeof(Q4_0Block)}};
    auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
    EXPECT_THROW(writer->finish(),std::logic_error);
    auto source=createSourceTensor("Q4_0",{2,64});
    NativeWeightSourceWindow first{.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=1,.first_column=0,.columns=32,.native_row_bytes=sizeof(Q4_0Block)};
    AlignedVector<std::byte> scratch(writer->stagingRequest()->scratch_bytes_per_unit);
    EXPECT_THROW(writer->append(first,{static_cast<const std::byte*>(source->raw_data()),sizeof(Q4_0Block)},
        {scratch.data(),scratch.size()}),std::invalid_argument);
    auto foreign=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
    first.first_row=1;
    EXPECT_THROW(foreign->append(first,{static_cast<const std::byte*>(source->raw_data()),sizeof(Q4_0Block)},
        {scratch.data(),scratch.size()}),std::invalid_argument);
    EXPECT_THROW(foreign->finish(),std::logic_error);
    spec.destination_precision=WeightPrecision::CONVERT_TO_INT8;
    EXPECT_THROW(CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local()),std::invalid_argument);
}

TEST(NativeWindowWriterContract, ExplicitFloatingConversionKeepsOriginalNativeDecode)
{
    for(const auto& entry:native_quantised_formats::kAllSourceFormats) {
        SCOPED_TRACE(std::string(entry.quant_type));
        const auto* format=native_quantised_formats::forQuantType(entry.quant_type);
        const size_t k=format->is_superblock?256:32;
        const size_t bytes=native_block_bytes_for_format(format->codebook_id,format->is_superblock);
        auto source=createSourceTensor(entry.quant_type,{3,k});
        std::vector<float> expected(3*k);source->to_fp32(expected.data());
        for(const auto precision:{WeightPrecision::CONVERT_TO_FP32,WeightPrecision::CONVERT_TO_FP16,WeightPrecision::CONVERT_TO_BF16}) {
            CPUPreparedWeightSource spec{
                .original={.original_shape={k,3},.original_type=source->native_type(),.native_group_elements=k,.native_group_bytes=bytes},
                .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=3,.first_column=0,.columns=k,.native_row_bytes=bytes},
                .destination_precision=precision};
            auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
            const auto request=writer->stagingRequest();ASSERT_TRUE(request);
            AlignedVector<std::byte> scratch(request->scratch_bytes_per_unit);
            for(size_t row=0;row<3;++row) {
                NativeWeightSourceWindow window{.original_expert=0,.selected_expert_ordinal=0,.first_row=row,.rows=1,.first_column=0,.columns=k,.native_row_bytes=bytes};
                writer->append(window,{static_cast<const std::byte*>(source->raw_data())+row*bytes,bytes},{scratch.data(),scratch.size()});
            }
            auto result=writer->finish();ASSERT_NE(result.kernel,nullptr);ASSERT_NE(result.final_tensor,nullptr);
            EXPECT_FALSE(result.kernel->canReleaseSourceWeightTensor());EXPECT_EQ(result.kernel->packedWeightBytes(),0u);
            const auto* raw=result.final_tensor->raw_data();ASSERT_NE(raw,nullptr);
            for(size_t i=0;i<expected.size();++i) {
                if(precision==WeightPrecision::CONVERT_TO_FP32)
                    EXPECT_EQ(std::bit_cast<uint32_t>(static_cast<const float*>(raw)[i]),std::bit_cast<uint32_t>(expected[i]));
                else EXPECT_EQ(static_cast<const uint16_t*>(raw)[i],precision==WeightPrecision::CONVERT_TO_FP16?
                    fp32_to_fp16(expected[i]):simd::fp32_to_bf16(expected[i]));
            }
        }
    }
}

TEST(NativeWindowWriterContract, FloatingEightBitUsesCanonicalQ8BlocksAndFinalNativeKernel)
{
    constexpr size_t n=3,k=64;
    AlignedVector<float> values(n*k);
    for(size_t i=0;i<values.size();++i)values[i]=float(int(i%31)-15)*0.03125f;
    auto source=std::make_shared<FP32Tensor>(std::vector<size_t>{n,k},std::move(values));
    AlignedVector<uint8_t> expected_bytes(n*(k/32)*sizeof(Q8_0Block));
    for(size_t row=0;row<n;++row)Q8_0Tensor::quantize_fp32_row(source->data()+row*k,
        reinterpret_cast<Q8_0Block*>(expected_bytes.data())+row*k/32,k);
    auto expected_source=std::make_unique<Q8_0Tensor>(std::vector<size_t>{n,k},std::move(expected_bytes));
    CPUNativeQuantisedPackedWeights expected;ASSERT_TRUE(packWeightsCPUNativeQuantised(expected_source.get(),expected));
    for(const auto consumer:{CPUWeightConsumer::PreparedGemm,CPUWeightConsumer::ResidentTensor}) {
        CPUPreparedWeightSource spec{
            .original={.original_shape={k,n},.original_type=TensorType::FP32,.native_group_elements=1,.native_group_bytes=sizeof(float)},
            .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=n,.first_column=0,.columns=k,.native_row_bytes=k*sizeof(float)},
            .destination_precision=WeightPrecision::CONVERT_TO_INT8,.consumer=consumer};
        auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
        const auto request=writer->stagingRequest();ASSERT_TRUE(request);
        AlignedVector<std::byte> scratch(request->scratch_bytes_per_unit+request->fixed_scratch_bytes);
        for(size_t row=0;row<n;++row)for(size_t col=0;col<k;col+=32) {
            NativeWeightSourceWindow window{.original_expert=0,.selected_expert_ordinal=0,.first_row=row,.rows=1,.first_column=col,.columns=32,.native_row_bytes=32*sizeof(float)};
            writer->append(window,{static_cast<const std::byte*>(source->raw_data())+(row*k+col)*sizeof(float),window.nativeBytes()},
                {scratch.data(),scratch.size()});
        }
        auto result=writer->finish();
        if(consumer==CPUWeightConsumer::PreparedGemm) {
            ASSERT_NE(result.kernel,nullptr);const auto* actual=result.kernel->exportCPUNativeQuantisedPackedWeights();ASSERT_NE(actual,nullptr);
            ASSERT_EQ(actual->native_interleaved.size(),expected.native_interleaved.size());
            EXPECT_EQ(0,std::memcmp(actual->native_interleaved.data(),expected.native_interleaved.data(),expected.native_interleaved.size()));
        }else {
            ASSERT_NE(result.final_tensor,nullptr);EXPECT_EQ(result.final_tensor->native_type(),TensorType::Q8_0);
            EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),expected_source->raw_data(),n*k/32*sizeof(Q8_0Block)));
        }
    }
}

TEST(NativeWindowWriterContract, DirectFinalWritesNeedNoTransientArenaAndFailuresStayUnpublished)
{
    std::array<float,128> original{};
    for(size_t i=0;i<original.size();++i)original[i]=float(i)*0.125f;
    CPUPreparedWeightSource spec{
        .original={.original_shape={64,2},.original_type=TensorType::FP32,.native_group_elements=1,.native_group_bytes=sizeof(float)},
        .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=2,.first_column=0,.columns=64,.native_row_bytes=64*sizeof(float)},
        .consumer=CPUWeightConsumer::ResidentTensor};
    auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
    EXPECT_FALSE(writer->stagingRequest());
    NativeWeightSourceWindow window{.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=2,.first_column=0,.columns=64,.native_row_bytes=64*sizeof(float)};
    writer->readDirect(window,[&](std::span<std::byte> final){ASSERT_EQ(final.size(),sizeof(original));std::memcpy(final.data(),original.data(),final.size());});
    auto result=writer->finish();ASSERT_NE(result.final_tensor,nullptr);EXPECT_EQ(result.kernel,nullptr);
    EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),original.data(),sizeof(original)));
    auto failed=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
    EXPECT_THROW(failed->readDirect(window,[](std::span<std::byte>){throw std::runtime_error("actual reader failure");}),std::runtime_error);
    EXPECT_THROW(failed->finish(),std::logic_error);
}

TEST(NativeWindowWriterContract, ResidentParentsPreserveOriginalRankIncludingSingletonOuterAxes)
{
    for(const auto type:{TensorType::FP32,TensorType::FP16,TensorType::BF16}) {
        const size_t element_bytes=type==TensorType::FP32?sizeof(float):sizeof(uint16_t);
        for(const auto& shape:std::vector<std::vector<size_t>>{{8,3,1},{8,3,2},{8,3,1,1},{8,3,2,2}}) {
            SCOPED_TRACE(tensorTypeName(type));
            SCOPED_TRACE(shape.size());
            size_t outer=1;for(size_t axis=2;axis<shape.size();++axis)outer*=shape[axis];
            const size_t matrix_bytes=shape[0]*shape[1]*element_bytes;
            std::vector<std::byte> original(matrix_bytes*outer);
            for(size_t i=0;i<original.size();++i)original[i]=std::byte((i*13+7)%251);
            CPUPreparedWeightSource spec{
                .original={.original_shape=shape,.original_type=type,.native_group_elements=1,.native_group_bytes=element_bytes},
                .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=shape[1],.first_column=0,.columns=shape[0],.native_row_bytes=shape[0]*element_bytes},
                .matrix_count=outer,.consumer=CPUWeightConsumer::ResidentTensor};
            auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());ASSERT_FALSE(writer->stagingRequest());
            for(size_t matrix=0;matrix<outer;++matrix) {
                auto window=spec.range;window.original_expert=matrix;window.selected_expert_ordinal=matrix;
                writer->readDirect(window,[&](std::span<std::byte> target){
                    ASSERT_EQ(target.size(),matrix_bytes);std::memcpy(target.data(),original.data()+matrix*matrix_bytes,target.size());
                });
            }
            auto result=writer->finish();ASSERT_NE(result.final_tensor,nullptr);EXPECT_EQ(result.kernel,nullptr);
            EXPECT_EQ(result.final_tensor->shape(),shape);EXPECT_EQ(result.final_tensor->native_type(),type);
            EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),original.data(),original.size()));
            auto partial=spec;partial.range.first_row=1;partial.range.rows=shape[1]-1;
            EXPECT_THROW(CPUPreparedWeightWriter::begin(partial,CPUWeightStoragePlacement::local()),std::invalid_argument);
            if(outer>1) {
                auto single=spec;single.matrix_count=1;single.range.original_expert=1;single.range.selected_expert_ordinal=1;
                EXPECT_THROW(CPUPreparedWeightWriter::begin(single,CPUWeightStoragePlacement::local()),std::invalid_argument);
            }
        }
    }
}

TEST(NativeWindowWriterContract, NativeResidentParentsPreserveRankWithoutInventingQ8_1Parents)
{
    for(const auto& entry:native_quantised_formats::kAllSourceFormats) {
        SCOPED_TRACE(std::string(entry.quant_type));const auto* format=native_quantised_formats::forQuantType(entry.quant_type);
        const size_t k=format->is_superblock?256:32;
        const size_t bytes=native_block_bytes_for_format(format->codebook_id,format->is_superblock);
        for(const size_t outer:{size_t(1),size_t(2)}) {
            constexpr size_t n=2;
            auto source=createSourceTensor(entry.quant_type,{n*outer,k});
            const std::vector<size_t> original_shape{k,n,outer};
            CPUPreparedWeightSource spec{
                .original={.original_shape=original_shape,.original_type=source->native_type(),.native_group_elements=k,.native_group_bytes=bytes},
                .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=n,.first_column=0,.columns=k,.native_row_bytes=bytes},
                .matrix_count=outer,.consumer=CPUWeightConsumer::ResidentTensor};
            if(source->native_type()==TensorType::Q8_1) {
                // Q8_1's actual owning/runtime representation is matrix-only;
                // a general original parent must never be disguised as [N,K].
                EXPECT_THROW(CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local()),std::invalid_argument);
                continue;
            }
            auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());ASSERT_FALSE(writer->stagingRequest());
            for(size_t matrix=0;matrix<outer;++matrix) {
                auto window=spec.range;window.original_expert=matrix;window.selected_expert_ordinal=matrix;
                writer->readDirect(window,[&](std::span<std::byte> target){
                    ASSERT_EQ(target.size(),n*bytes);std::memcpy(target.data(),static_cast<const std::byte*>(source->raw_data())+matrix*n*bytes,target.size());
                });
            }
            auto result=writer->finish();ASSERT_NE(result.final_tensor,nullptr);
            EXPECT_EQ(result.final_tensor->shape(),original_shape);EXPECT_EQ(result.final_tensor->native_type(),source->native_type());
            EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),source->raw_data(),outer*n*bytes));
        }
    }
}

TEST(NativeWindowWriterContract, GenericFP32HalfKeepsWholeSelectionVectorAndScalarTailAcrossWindows)
{
    ScopedOMPThreadCount threads(2);
    for(const size_t k:{size_t(19),size_t(1367)}) {
        constexpr size_t n=3,first_row=2;
        const std::array<float,10> edges{std::ldexp(1.75f,-24),-std::ldexp(1.75f,-24),
            std::ldexp(1.5f,-24),std::ldexp(2.5f,-24),std::ldexp(3.5f,-24),
            1.0f+std::ldexp(1.0f,-11),1.0f+3*std::ldexp(1.0f,-11),-0.0f,65504.0f,0.125f};
        AlignedVector<float> data(7*k);for(size_t i=0;i<data.size();++i)data[i]=edges[i%edges.size()];
        auto source=std::make_shared<FP32Tensor>(std::vector<size_t>{7,k},std::move(data));
        auto selected=source->create_view({n,k},first_row*k);ASSERT_NE(selected,nullptr);
        std::vector<uint16_t> expected(n*k);selected->to_fp16(expected.data());
        for(const size_t width:{k==19?size_t(1):size_t(37),k}) {
            CPUPreparedWeightSource spec{
                .original={.original_shape={k,7},.original_type=TensorType::FP32,.native_group_elements=1,.native_group_bytes=sizeof(float)},
                .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=first_row,.rows=n,.first_column=0,.columns=k,.native_row_bytes=k*sizeof(float)},
                .destination_precision=WeightPrecision::CONVERT_TO_FP16};
            auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
            for(size_t row=first_row;row<first_row+n;++row)for(size_t column=0;column<k;) {
                const size_t columns=std::min(width,k-column);
                NativeWeightSourceWindow window{.original_expert=0,.selected_expert_ordinal=0,.first_row=row,.rows=1,.first_column=column,.columns=columns,.native_row_bytes=columns*sizeof(float)};
                writer->append(window,{static_cast<const std::byte*>(source->raw_data())+(row*k+column)*sizeof(float),window.nativeBytes()},{});
                column+=columns;
            }
            auto result=writer->finish();ASSERT_NE(result.final_tensor,nullptr);
            EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),expected.data(),expected.size()*sizeof(uint16_t)));
        }
    }
}

TEST(NativeWindowWriterContract, ModelHalfMatchesActualOriginalOverrideAcrossNativeAndFloatingSources)
{
    ScopedOMPThreadCount threads(2);
    std::vector<std::shared_ptr<TensorBase>> sources;
    for(const auto& entry:native_quantised_formats::kAllSourceFormats) {
        const auto* format=native_quantised_formats::forQuantType(entry.quant_type);
        sources.emplace_back(createSourceTensor(entry.quant_type,{2,format->is_superblock?256u:32u}));
    }
    AlignedVector<float> floats(38);AlignedVector<uint16_t> bf16(38),half(38);
    for(size_t i=0;i<floats.size();++i) {
        floats[i]=i%2?std::ldexp(1.75f,-24):1.0f+std::ldexp(1.0f,-11);
        bf16[i]=simd::fp32_to_bf16(floats[i]);half[i]=fp32_to_fp16(floats[i]);
    }
    sources.push_back(std::make_shared<FP32Tensor>(std::vector<size_t>{2,19},std::move(floats)));
    sources.push_back(std::make_shared<BF16Tensor>(std::vector<size_t>{2,19},std::move(bf16)));
    sources.push_back(std::make_shared<FP16Tensor>(std::vector<size_t>{2,19},std::move(half)));
    for(const auto& source:sources) {
        SCOPED_TRACE(tensorTypeName(source->native_type()));const size_t n=source->rows(),k=source->cols();
        const bool already_half=source->native_type()==TensorType::FP16;
        // The original representation owner leaves an already-FP16 source
        // native. Every actual override uses this exact source decoder oracle.
        auto expected=already_half?std::shared_ptr<FP16Tensor>{}:materializePreparedHalfWeights(*source);
        const auto* format=native_quantised_formats::forQuantType(tensorTypeName(source->native_type()));
        const size_t group=format?(format->is_superblock?256:32):1;
        const size_t group_bytes=format?native_block_bytes_for_format(format->codebook_id,format->is_superblock):
            source->native_type()==TensorType::FP32?sizeof(float):sizeof(uint16_t);
        for(const size_t width:{group,k}) {
            CPUPreparedWeightSource spec{
                .original={.original_shape={k,n},.original_type=source->native_type(),.native_group_elements=group,.native_group_bytes=group_bytes},
                .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=n,.first_column=0,.columns=k,.native_row_bytes=k/group*group_bytes},
                .destination_precision=already_half?WeightPrecision::NATIVE:WeightPrecision::CONVERT_TO_FP16,
                .operand_precision=FloatingActivationOperandPrecision::FP16RNE,
                .half_conversion=already_half?CPUPreparedHalfConversion::GenericTensor:CPUPreparedHalfConversion::ModelPreparedHalf};
            auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
            const auto request=writer->stagingRequest();
            AlignedVector<std::byte> scratch(request?width/group*request->scratch_bytes_per_unit+request->fixed_scratch_bytes:0);
            for(size_t row=0;row<n;++row)for(size_t column=0;column<k;column+=width) {
                NativeWeightSourceWindow window{.original_expert=0,.selected_expert_ordinal=0,.first_row=row,.rows=1,.first_column=column,.columns=width,.native_row_bytes=width/group*group_bytes};
                const auto* input=static_cast<const std::byte*>(source->raw_data())+row*spec.range.native_row_bytes+column/group*group_bytes;
                if(request)writer->append(window,{input,window.nativeBytes()},{scratch.data(),scratch.size()});
                else writer->readDirect(window,[&](std::span<std::byte> target){std::memcpy(target.data(),input,target.size());});
            }
            auto result=writer->finish();ASSERT_NE(result.final_tensor,nullptr);
            EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),already_half?source->raw_data():expected->raw_data(),n*k*sizeof(uint16_t)));
        }
    }
}

TEST(NativeWindowWriterContract, ModelHalfRejectsOriginalNonfiniteAndHalfOverflowBeforePublication)
{
    ScopedOMPThreadCount threads(2);
    for(const float invalid:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),70000.0f}) {
        AlignedVector<float> data(8);std::fill(data.begin(),data.end(),0.25f);data[3]=invalid;
        auto source=std::make_shared<FP32Tensor>(std::vector<size_t>{1,8},std::move(data));
        EXPECT_THROW(materializePreparedHalfWeights(*source),std::invalid_argument);
        CPUPreparedWeightSource spec{
            .original={.original_shape={8,1},.original_type=TensorType::FP32,.native_group_elements=1,.native_group_bytes=sizeof(float)},
            .range={.original_expert=0,.selected_expert_ordinal=0,.first_row=0,.rows=1,.first_column=0,.columns=8,.native_row_bytes=8*sizeof(float)},
            .destination_precision=WeightPrecision::CONVERT_TO_FP16,.operand_precision=FloatingActivationOperandPrecision::FP16RNE,
            .half_conversion=CPUPreparedHalfConversion::ModelPreparedHalf};
        auto wrong_policy=spec;wrong_policy.half_conversion=CPUPreparedHalfConversion::GenericTensor;
        EXPECT_THROW(CPUPreparedWeightWriter::begin(wrong_policy,CPUWeightStoragePlacement::local()),std::invalid_argument);
        auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
        EXPECT_THROW(writer->append(spec.range,{static_cast<const std::byte*>(source->raw_data()),spec.range.nativeBytes()},{}),std::invalid_argument);
        EXPECT_THROW(writer->finish(),std::logic_error);
    }
}
