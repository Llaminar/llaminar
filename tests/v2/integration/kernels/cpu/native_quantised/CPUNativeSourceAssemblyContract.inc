/**
 * @file CPUNativeSourceAssemblyContract.inc
 * @brief Tiny original-byte oracles for ordered CPU source stripes and final arithmetic.
 *
 * The existing native source fixture supplies genuine encoded bytes. The
 * production stripe iterator and final writer consume only useful windows.
 * Expected row/K selection is reconstructed directly from original bytes;
 * no production source parent is concatenated or manufactured by these tests.
 */
namespace {
/**
 * @brief Feed the actual writer using one production-sized source/scratch arena.
 * @param original Genuine complete native/floating fixture source.
 * @param spec Original source declaration and exact typed stripe assembly.
 * @param minimum_window Select the smallest admitted useful consumer unit.
 * @return Actual final CPU owner/engine after complete coverage.
 */
CPUPreparedWeightResult writeOriginalAssembly(const TensorBase& original,CPUPreparedWeightSource spec,bool minimum_window)
{
    auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
    const auto request=writer->stagingRequest();
    std::optional<NativeWeightLoadStagingGeometry> geometry;
    AlignedVector<std::byte> arena;
    if(request) {
        auto unit=*request;unit.maximum_units=1;
        const auto budget=resolveNativeWeightLoadStaging(minimum_window?unit:*request,CPUStoragePlacement::local(),std::nullopt).physical_bytes;
        geometry=resolveNativeWeightLoadStaging(*request,CPUStoragePlacement::local(),budget);
        if(geometry->physical_bytes>budget)throw std::logic_error("Assembly fixture exceeded its actual source cap");
        arena.resize_uninitialized(geometry->scratch_bytes?geometry->scratch_offset+geometry->scratch_bytes:geometry->native_bytes);
    }
    const size_t source_row_bytes=spec.original.original_shape[0]/spec.original.native_group_elements*spec.original.native_group_bytes;
    forEachNativeWeightSourceStripe(spec.original,spec.range,spec.assembly,[&](const auto& stripe,size_t,size_t){
        size_t capacity=geometry?geometry->native_bytes:minimum_window?spec.original.native_group_bytes:stripe.nativeBytes();
        if(!request && spec.assembly && spec.assembly->axis==NativeWeightSourceAssembly::Axis::Columns)
            capacity=std::min(capacity,stripe.native_row_bytes);
        forEachNativeWeightSourceWindow(spec.original,stripe,capacity,[&](const auto& window){
            const auto read=[&](std::span<std::byte> destination) {
                if(destination.size()!=window.nativeBytes())throw std::logic_error("Assembly fixture read extent changed");
                for(size_t row=0;row<window.rows;++row)
                    std::memcpy(destination.data()+row*window.native_row_bytes,
                        static_cast<const std::byte*>(original.raw_data())+(window.first_row+row)*source_row_bytes+
                        window.first_column/spec.original.native_group_elements*spec.original.native_group_bytes,
                        window.native_row_bytes);
            };
            if(request) {
                auto native=std::span<std::byte>(arena.data(),window.nativeBytes());read(native);
                writer->append(window,native,{arena.data()+geometry->scratch_offset,geometry->scratch_bytes});
            }else writer->readDirect(window,read);
        });
    });
    return writer->finish();
}
}

TEST(NativeSourceAssemblyContract, AllNativeFormatsPreserveIndependentOriginalRowAndColumnBytes)
{
    ScopedOMPThreadCount threads(2);
    for(const auto& entry:native_quantised_formats::kAllSourceFormats) {
        SCOPED_TRACE(std::string(entry.quant_type));const auto* format=native_quantised_formats::forQuantType(entry.quant_type);
        const size_t group=format->is_superblock?256:32;
        const size_t bytes=native_block_bytes_for_format(format->codebook_id,format->is_superblock);
        for(const auto axis:{NativeWeightSourceAssembly::Axis::Rows,NativeWeightSourceAssembly::Axis::Columns}) {
            const size_t source_rows=axis==NativeWeightSourceAssembly::Axis::Rows?6:4;
            const size_t source_columns=(axis==NativeWeightSourceAssembly::Axis::Rows?4:6)*group;
            auto original=createSourceTensor(entry.quant_type,{source_rows,source_columns});
            NativeWeightSourceWindow range{.rows=source_rows,.columns=source_columns,.native_row_bytes=source_columns/group*bytes};
            NativeWeightSourceAssembly assembly{.axis=axis,.final_rows=3,.final_columns=(axis==NativeWeightSourceAssembly::Axis::Rows?4:3)*group,.segments={}};
            if(axis==NativeWeightSourceAssembly::Axis::Rows) {
                range.first_row=1;range.rows=4;
                assembly.segments={{.first_row=1,.rows=2,.columns=source_columns,.native_row_bytes=range.native_row_bytes},
                    {.first_row=4,.rows=1,.columns=source_columns,.native_row_bytes=range.native_row_bytes}};
            }else {
                range.first_row=1;range.rows=3;range.first_column=group;range.columns=4*group;range.native_row_bytes=4*bytes;
                assembly.segments={{.first_row=1,.rows=3,.first_column=group,.columns=2*group,.native_row_bytes=2*bytes},
                    {.first_row=1,.rows=3,.first_column=4*group,.columns=group,.native_row_bytes=bytes}};
            }
            const size_t output_row_bytes=assembly.final_columns/group*bytes;
            AlignedVector<uint8_t> selected(assembly.final_rows*output_row_bytes);
            const auto* raw=static_cast<const uint8_t*>(original->raw_data());
            for(size_t row=0;row<3;++row) {
                if(axis==NativeWeightSourceAssembly::Axis::Rows) {
                    const size_t source_row=row<2?row+1:4;
                    std::memcpy(selected.data()+row*output_row_bytes,raw+source_row*(source_columns/group*bytes),output_row_bytes);
                }else {
                    const auto* source=raw+(row+1)*(source_columns/group*bytes);
                    std::memcpy(selected.data()+row*output_row_bytes,source+bytes,2*bytes);
                    std::memcpy(selected.data()+row*output_row_bytes+2*bytes,source+4*bytes,bytes);
                }
            }
            auto reference=TensorFactory::adoptQuantizedFinalStorage(original->native_type(),{3,assembly.final_columns},std::move(selected),DeviceId::cpu());
            CPUNativeQuantisedPackedWeights expected;ASSERT_TRUE(packWeightsCPUNativeQuantised(reference.get(),expected));
            for(const auto consumer:{CPUWeightConsumer::ResidentTensor,CPUWeightConsumer::PreparedGemm})for(const bool small:{true,false}) {
                CPUPreparedWeightSource spec{
                    .original={{source_columns,source_rows},original->native_type(),group,bytes},.range=range,
                    .consumer=consumer,.assembly=assembly};
                auto result=writeOriginalAssembly(*original,spec,small);
                if(consumer==CPUWeightConsumer::ResidentTensor) {
                    ASSERT_NE(result.final_tensor,nullptr);EXPECT_EQ(result.final_tensor->shape(),(std::vector<size_t>{3,assembly.final_columns}));
                    EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),reference->raw_data(),3*output_row_bytes));
                }else {
                    ASSERT_NE(result.kernel,nullptr);const auto* actual=result.kernel->exportCPUNativeQuantisedPackedWeights();ASSERT_NE(actual,nullptr);
                    ASSERT_EQ(actual->native_interleaved.size(),expected.native_interleaved.size());
                    EXPECT_EQ(0,std::memcmp(actual->native_interleaved.data(),expected.native_interleaved.data(),expected.native_interleaved.size()));
                }
            }
            std::vector<float> expected_fp32(3*assembly.final_columns);reference->to_fp32(expected_fp32.data());
            auto expected_half=materializePreparedHalfWeights(*reference);
            for(const auto precision:{WeightPrecision::CONVERT_TO_FP32,WeightPrecision::CONVERT_TO_FP16})for(const bool small:{true,false}) {
                CPUPreparedWeightSource spec{
                    .original={{source_columns,source_rows},original->native_type(),group,bytes},.range=range,
                    .destination_precision=precision,
                    .operand_precision=precision==WeightPrecision::CONVERT_TO_FP16?FloatingActivationOperandPrecision::FP16RNE:FloatingActivationOperandPrecision::FP32,
                    .half_conversion=precision==WeightPrecision::CONVERT_TO_FP16?CPUPreparedHalfConversion::ModelPreparedHalf:CPUPreparedHalfConversion::GenericTensor,
                    .assembly=assembly};
                auto result=writeOriginalAssembly(*original,spec,small);ASSERT_NE(result.final_tensor,nullptr);
                EXPECT_EQ(result.final_tensor->shape(),(std::vector<size_t>{3,assembly.final_columns}));
                if(precision==WeightPrecision::CONVERT_TO_FP32)
                    EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),expected_fp32.data(),expected_fp32.size()*sizeof(float)));
                else EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),expected_half->raw_data(),3*assembly.final_columns*sizeof(uint16_t)));
            }
        }
    }
}

TEST(NativeSourceAssemblyContract, FloatingHalfUsesFinalDenseIndicesAndLiteralModelOverride)
{
    ScopedOMPThreadCount threads(2);
    constexpr size_t source_rows=4,source_columns=19,final_rows=3,final_columns=14;
    AlignedVector<float> values(source_rows*source_columns);
    for(size_t i=0;i<values.size();++i)values[i]=i%2?std::ldexp(1.75f,-24):1.0f+std::ldexp(1.0f,-11);
    auto original=std::make_shared<FP32Tensor>(std::vector<size_t>{source_rows,source_columns},std::move(values));
    AlignedVector<float> selected(final_rows*final_columns);
    for(size_t row=0;row<final_rows;++row) {
        const auto* source=original->data()+(row+1)*source_columns;auto* output=selected.data()+row*final_columns;
        std::copy_n(source,5,output);std::copy_n(source+9,7,output+5);std::copy_n(source+17,2,output+12);
    }
    auto dense=std::make_shared<FP32Tensor>(std::vector<size_t>{final_rows,final_columns},std::move(selected));
    NativeWeightSourceAssembly assembly{.axis=NativeWeightSourceAssembly::Axis::Columns,.final_rows=final_rows,.final_columns=final_columns,
        .segments={{.first_row=1,.rows=3,.columns=5,.native_row_bytes=5*sizeof(float)},
                   {.first_row=1,.rows=3,.first_column=9,.columns=7,.native_row_bytes=7*sizeof(float)},
                   {.first_row=1,.rows=3,.first_column=17,.columns=2,.native_row_bytes=2*sizeof(float)}}};
    for(const auto policy:{CPUPreparedHalfConversion::GenericTensor,CPUPreparedHalfConversion::ModelPreparedHalf}) {
        std::vector<uint16_t> expected(final_rows*final_columns);
        if(policy==CPUPreparedHalfConversion::GenericTensor)dense->to_fp16(expected.data());
        else {auto model=materializePreparedHalfWeights(*dense);std::memcpy(expected.data(),model->raw_data(),expected.size()*sizeof(uint16_t));}
        for(const bool small:{true,false}) {
            CPUPreparedWeightSource spec{
                .original={{source_columns,source_rows},TensorType::FP32,1,sizeof(float)},
                .range={.first_row=1,.rows=3,.columns=source_columns,.native_row_bytes=source_columns*sizeof(float)},
                .destination_precision=WeightPrecision::CONVERT_TO_FP16,
                .operand_precision=policy==CPUPreparedHalfConversion::ModelPreparedHalf?FloatingActivationOperandPrecision::FP16RNE:FloatingActivationOperandPrecision::Native,
                .half_conversion=policy,.assembly=assembly};
            auto result=writeOriginalAssembly(*original,spec,small);ASSERT_NE(result.final_tensor,nullptr);
            EXPECT_EQ(result.final_tensor->shape(),(std::vector<size_t>{final_rows,final_columns}));
            EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),expected.data(),expected.size()*sizeof(uint16_t)));
        }
    }
}

TEST(NativeSourceAssemblyContract, ScalarHeadStripesRetainVectorRankAndDirectColumnsRejectMultirow)
{
    AlignedVector<float> data(16);for(size_t i=0;i<data.size();++i)data[i]=float(i)+0.25f;
    auto vector=std::make_shared<FP32Tensor>(std::vector<size_t>{16},std::move(data));
    NativeWeightSourceAssembly assembly{.axis=NativeWeightSourceAssembly::Axis::Columns,.final_rows=1,.final_columns=3,
        .segments={{.rows=1,.first_column=1,.columns=1,.native_row_bytes=sizeof(float)},
                   {.rows=1,.first_column=3,.columns=1,.native_row_bytes=sizeof(float)},
                   {.rows=1,.first_column=7,.columns=1,.native_row_bytes=sizeof(float)}}};
    CPUPreparedWeightSource spec{.original={{16},TensorType::FP32,1,sizeof(float)},
        .range={.rows=1,.first_column=1,.columns=7,.native_row_bytes=7*sizeof(float)},
        .consumer=CPUWeightConsumer::ResidentTensor,.assembly=assembly};
    for(const bool small:{true,false}) {
        auto result=writeOriginalAssembly(*vector,spec,small);ASSERT_NE(result.final_tensor,nullptr);
        EXPECT_EQ(result.final_tensor->shape(),(std::vector<size_t>{3}));
        const std::array<float,3> expected{1.25f,3.25f,7.25f};
        EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),expected.data(),sizeof(expected)));
    }
    spec.original.original_shape={16,2};spec.range.rows=2;
    spec.assembly->final_rows=2;for(auto& stripe:spec.assembly->segments)stripe.rows=2;
    auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
    size_t callbacks=0;
    EXPECT_THROW(writer->readDirect(spec.assembly->segments.front(),[&](std::span<std::byte>){++callbacks;}),std::invalid_argument);
    EXPECT_EQ(callbacks,0u);EXPECT_THROW(writer->finish(),std::logic_error);
}

TEST(NativeSourceAssemblyContract, InvalidOrForeignStripesRejectBeforeAnySourceRead)
{
    NativeWeightSourceAssembly assembly{.axis=NativeWeightSourceAssembly::Axis::Columns,.final_rows=2,.final_columns=64,
        .segments={{.rows=2,.columns=32,.native_row_bytes=sizeof(Q4_0Block)},
                   {.rows=2,.first_column=64,.columns=32,.native_row_bytes=sizeof(Q4_0Block)}}};
    CPUPreparedWeightSource spec{.original={{96,2},TensorType::Q4_0,32,sizeof(Q4_0Block)},
        .range={.rows=2,.columns=96,.native_row_bytes=3*sizeof(Q4_0Block)},.assembly=assembly};
    for(int variant=0;variant<4;++variant) {
        auto invalid=spec;
        if(variant==0)++invalid.assembly->final_columns;
        if(variant==1)invalid.assembly->segments[1].original_expert=1;
        if(variant==2)invalid.assembly->segments[1].first_column=32+1;
        if(variant==3)invalid.assembly->segments[1].rows=1;
        EXPECT_THROW(CPUPreparedWeightWriter::begin(invalid,CPUWeightStoragePlacement::local()),std::invalid_argument);
    }
    auto writer=CPUPreparedWeightWriter::begin(spec,CPUWeightStoragePlacement::local());
    std::array<std::byte,2*sizeof(Q4_0Block)> native{};
    AlignedVector<std::byte> scratch(2*writer->stagingRequest()->scratch_bytes_per_unit);
    EXPECT_THROW(writer->append(spec.assembly->segments.back(),native,{scratch.data(),scratch.size()}),std::invalid_argument);
    EXPECT_THROW(writer->finish(),std::logic_error);
}

TEST(NativeSourceAssemblyContract, FloatingQ8ColumnStripesScatterActualEncodedRows)
{
    ScopedOMPThreadCount threads(2);
    constexpr size_t source_rows=4,source_columns=192,final_rows=3,final_columns=96;
    AlignedVector<float> values(source_rows*source_columns);
    for(size_t i=0;i<values.size();++i)values[i]=float(int(i%37)-18)*0.03125f;
    auto original=std::make_shared<FP32Tensor>(std::vector<size_t>{source_rows,source_columns},std::move(values));
    std::vector<float> selected(final_rows*final_columns);
    for(size_t row=0;row<3;++row) {
        const auto* source=original->data()+(row+1)*source_columns;
        std::copy_n(source+32,64,selected.data()+row*final_columns);
        std::copy_n(source+128,32,selected.data()+row*final_columns+64);
    }
    AlignedVector<uint8_t> encoded(final_rows*final_columns/32*sizeof(Q8_0Block));
    for(size_t row=0;row<3;++row)
        Q8_0Tensor::quantize_fp32_row(selected.data()+row*final_columns,reinterpret_cast<Q8_0Block*>(encoded.data())+row*final_columns/32,final_columns);
    auto reference=TensorFactory::adoptQuantizedFinalStorage(TensorType::Q8_0,{final_rows,final_columns},std::move(encoded),DeviceId::cpu());
    CPUNativeQuantisedPackedWeights expected;ASSERT_TRUE(packWeightsCPUNativeQuantised(reference.get(),expected));
    NativeWeightSourceAssembly assembly{.axis=NativeWeightSourceAssembly::Axis::Columns,.final_rows=final_rows,.final_columns=final_columns,
        .segments={{.first_row=1,.rows=3,.first_column=32,.columns=64,.native_row_bytes=64*sizeof(float)},
                   {.first_row=1,.rows=3,.first_column=128,.columns=32,.native_row_bytes=32*sizeof(float)}}};
    CPUPreparedWeightSource spec{.original={{source_columns,source_rows},TensorType::FP32,1,sizeof(float)},
        .range={.first_row=1,.rows=3,.first_column=32,.columns=128,.native_row_bytes=128*sizeof(float)},
        .destination_precision=WeightPrecision::CONVERT_TO_INT8,.assembly=assembly};
    for(const auto consumer:{CPUWeightConsumer::ResidentTensor,CPUWeightConsumer::PreparedGemm})for(const bool small:{true,false}) {
        spec.consumer=consumer;auto result=writeOriginalAssembly(*original,spec,small);
        if(consumer==CPUWeightConsumer::ResidentTensor) {
            ASSERT_NE(result.final_tensor,nullptr);EXPECT_EQ(result.final_tensor->shape(),(std::vector<size_t>{3,96}));
            EXPECT_EQ(0,std::memcmp(result.final_tensor->raw_data(),reference->raw_data(),3*96/32*sizeof(Q8_0Block)));
        }else {
            ASSERT_NE(result.kernel,nullptr);const auto* actual=result.kernel->exportCPUNativeQuantisedPackedWeights();ASSERT_NE(actual,nullptr);
            ASSERT_EQ(actual->native_interleaved.size(),expected.native_interleaved.size());
            EXPECT_EQ(0,std::memcmp(actual->native_interleaved.data(),expected.native_interleaved.data(),expected.native_interleaved.size()));
        }
    }
    auto incomplete_group=spec;incomplete_group.assembly->segments[0].columns=63;
    incomplete_group.assembly->segments[0].native_row_bytes=63*sizeof(float);
    incomplete_group.assembly->segments[1].columns=33;
    incomplete_group.assembly->segments[1].first_column=127;
    incomplete_group.assembly->segments[1].native_row_bytes=33*sizeof(float);
    EXPECT_THROW(CPUPreparedWeightWriter::begin(incomplete_group,CPUWeightStoragePlacement::local()),std::invalid_argument);
}
