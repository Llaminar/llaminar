/**
 * @file CanonicalOrderedDown.inc
 * @brief Register-owned native down rows with Llaminar's existing arithmetic.
 *
 * This diagnostic leaf is included inside qfn_castagna_native_impl after the
 * native readers and CanonicalFFNQ8Block declaration. The enclosing translation
 * unit includes DeviceNativeQuantisedContributionContract and
 * MoEProjectionNumericalContract at global scope. It owns neither allocations
 * nor synchronization: one lane retains one output's complete ordered K fold,
 * then the original row-local route fold. Native prepared weights are borrowed
 * without conversion. Independent columns provide parallelism; no K partial or
 * completed route is written to global memory merely to reduce it later.
 *
 * The first matched source tuple has native Q5_1 routed down and Q8_0 shared
 * down. Other formats must use their actual canonical decoders before this
 * diagnostic can become a general production implementation.
 */
#pragma once

/**
 * @brief Task-local borrowed fields preserving descriptor address provenance.
 * @tparam NeedMins Whether the actual source codebook consumes a minimum plane.
 * @details Routed sources are pointers into the immutable global table; shared
 * sources are by-value kernel arguments. The original native reader has distinct
 * overloads for those authorities. Never take the address of a kernarg copy and
 * feed it to the scalar-memory global-table reader.
 */
template<bool NeedMins>
class CurrentDownFields {
    unsigned columns_;
    const uint8_t* payload_;
    const unsigned short *scales_, *mins_;
    uint8_t layout_ = 0; ///< Packed metadata layout (per-block or superblock-coded).
public:
    /**
     * @brief Borrow the original descriptor without changing its representation.
     * @tparam Descriptor Original global-table pointer or by-value matrix type.
     * @param descriptor Actual retained descriptor, uniform in this wave task.
     */
    template<class Descriptor>
    __device__ __forceinline__ explicit CurrentDownFields(const Descriptor& descriptor)
    {
        native_descriptor_fields<NeedMins>(descriptor, columns_, payload_, scales_, mins_);
        layout_ = native_descriptor_layout(descriptor);
    }

    /**
     * @brief Issue the exact three existing routed pointers together for the cold-bound output extent.
     * @tparam Columns Original immutable routed down output columns admitted by the graph owner.
     * @param descriptor Published same-epoch native down record for a local selected job.
     * @pre The retained leaf's cold binding and original bank producers enforce Columns.
     * @details The packet is entirely within the descriptor and also covers its optional
     * unused extended minimum pointer. Reuse the original scalar wait and epoch lifetime.
     */
    template<unsigned Columns>
    __device__ __forceinline__ explicit CurrentDownFields(const NativeMatrix* descriptor,
        std::integral_constant<unsigned,Columns>) : columns_(Columns)
    {
        static_assert(Columns>0);
        static_assert(offsetof(NativeMatrix,payload)==0 && offsetof(NativeMatrix,scales)==8 &&
            offsetof(NativeMatrix,mins)==16 && offsetof(NativeMatrix,emins)==24 && sizeof(NativeMatrix)>=32);
        const auto address=reinterpret_cast<uint64_t>(native_weight_uniform(descriptor));
        if constexpr(NeedMins) {
            using PointerWords = unsigned __attribute__((ext_vector_type(8)));
            PointerWords bits;
            asm volatile("s_load_dwordx8 %0, %1, 0\n\t"
                         "s_waitcnt lgkmcnt(0)" : "=&s"(bits) : "s"(address) : "memory");
            payload_=reinterpret_cast<const uint8_t*>(uint64_t(bits[0])|(uint64_t(bits[1])<<32));
            scales_=reinterpret_cast<const unsigned short*>(uint64_t(bits[2])|(uint64_t(bits[3])<<32));
            mins_=reinterpret_cast<const unsigned short*>(uint64_t(bits[4])|(uint64_t(bits[5])<<32));
            layout_=native_descriptor_layout(descriptor);
        } else {
            NativeScalarWords4 bits;
            asm volatile("s_load_dwordx4 %0, %1, 0\n\t"
                         "s_waitcnt lgkmcnt(0)" : "=&s"(bits) : "s"(address) : "memory");
            payload_=reinterpret_cast<const uint8_t*>(uint64_t(bits[0])|(uint64_t(bits[1])<<32));
            scales_=reinterpret_cast<const unsigned short*>(uint64_t(bits[2])|(uint64_t(bits[3])<<32));
            mins_=nullptr;
        }
    }

    /**
     * @brief Borrow shared Q8 down payload and scale from their original kernargs.
     * @tparam Columns Output extent proved by shared source preparation.
     * @param descriptor Original by-value shared down matrix descriptor.
     */
    template<unsigned Columns>
    __device__ __forceinline__ explicit CurrentDownFields(const NativeMatrix& descriptor,
        std::integral_constant<unsigned,Columns>) :
        columns_(Columns), payload_(static_cast<const uint8_t*>(descriptor.payload)),
        scales_(static_cast<const unsigned short*>(descriptor.scales)), mins_(nullptr),
        layout_(native_descriptor_layout(descriptor))
    {
        static_assert(!NeedMins && Columns>0);
    }

    /**
     * @brief Expose already borrowed fields to the unchanged native payload reader.
     * @tparam ReadMins Whether that reader requires the secondary plane.
     * @param columns Original matrix output extent.
     * @param payload Original globally resident payload.
     * @param scales Original globally resident scale plane.
     * @param mins Original secondary plane, or null when it is not consumed.
     */
    /** @return The borrowed matrix's packed metadata layout. */
    __device__ __forceinline__ uint8_t layout() const { return layout_; }

    template<bool ReadMins>
    __device__ __forceinline__ void borrow(unsigned& columns, const uint8_t*& payload,
        const unsigned short*& scales, const unsigned short*& mins) const
    {
        static_assert(!ReadMins || NeedMins);
        columns = columns_; payload = payload_; scales = scales_;
        if constexpr (ReadMins) mins = mins_; else mins = nullptr;
    }
};

/**
 * @brief Adapt task fields to the existing native reader without a descriptor load.
 * @tparam ReadMins Actual secondary-plane consumption.
 * @tparam NeedMins Plane set retained by the original task.
 * @param fields Task-local register borrow.
 * @param columns Original matrix output width.
 * @param payload Original payload address.
 * @param scales Original scale-plane address.
 * @param mins Original minimum-plane address when needed.
 */
template<bool ReadMins, bool NeedMins>
__device__ __forceinline__ void native_descriptor_fields(const CurrentDownFields<NeedMins>& fields,
    unsigned& columns, const uint8_t*& payload, const unsigned short*& scales,
    const unsigned short*& mins)
{
    fields.template borrow<ReadMins>(columns, payload, scales, mins);
}

/** @brief Metadata layout already borrowed by task-local down fields. */
template<bool NeedMins>
__device__ __forceinline__ uint8_t native_descriptor_layout(const CurrentDownFields<NeedMins>& fields)
{
    return fields.layout();
}

/**
 * @brief Form one canonical corrected Q5_1 K32 contribution.
 * @param words Exact native scale/min, high bits, and four low-bit words.
 * @param activation Original canonical Q8 codes, FP32 scale, and integer sum.
 * @return Rounded dot contribution plus separately rounded min correction.
 * @details Integer dot ordering is immaterial within its proven INT32 range;
 * every floating-point rounding edge is the existing shared contract.
 */
__device__ __forceinline__ float currentDownQ51Block(
    const int (&words)[6], const CanonicalFFNQ8Block& activation)
{
    namespace numeric = ::llaminar2::device_native_quantised_contract;
    const unsigned high = static_cast<unsigned>(words[1]);
    int dot = 0;
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        const unsigned packed = static_cast<unsigned>(words[word + 2]);
        const unsigned low_high = ((high >> (word * 4)) & 15u);
        const unsigned high_high = ((high >> (word * 4 + 16)) & 15u);
        const int low = static_cast<int>((packed & 0x0f0f0f0fu) |
            (((low_high * 0x00204081u) & 0x01010101u) << 4));
        const int upper = static_cast<int>(((packed >> 4) & 0x0f0f0f0fu) |
            (((high_high * 0x00204081u) & 0x01010101u) << 4));
        dot = ::llaminar2::rocm_integer_contract::signedDot4(
            low, activation.qs[word], dot);
        dot = ::llaminar2::rocm_integer_contract::signedDot4(
            upper, activation.qs[word + 4], dot);
    }
    const float2 coefficients = __half22float2(__builtin_bit_cast(half2, words[0]));
    return numeric::accumulate(
        numeric::singleScaleBlock(dot, coefficients.x, activation.scale),
        numeric::singleScaleCorrection(activation.sum, coefficients.y, activation.scale));
}

/**
 * @brief Read and evaluate one exact native Q8_0 block, without pairing its sum.
 * @param fields Retained task-local immutable descriptor fields.
 * @param column Original output coordinate.
 * @param block Original K32 coordinate.
 * @param activation Canonical activation block at that coordinate.
 * @return One independently rounded canonical block contribution.
 */
template<class Fields>
__device__ __forceinline__ float currentDownQ8Block(
    const Fields& fields, unsigned column, unsigned block,
    const CanonicalFFNQ8Block& activation)
{
    unsigned columns;
    const uint8_t* payload;
    const unsigned short *scales, *unused;
    native_descriptor_fields<false>(fields, columns, payload, scales, unused);
    const size_t offset = size_t(block) * columns + column;
    const int4 first = native_global_weight_read<int4>(payload + offset * 32);
    const int4 second = native_global_weight_read<int4>(payload + offset * 32 + 16);
    int dot = 0;
    const int words[8] = {first.x, first.y, first.z, first.w,
                          second.x, second.y, second.z, second.w};
#pragma unroll
    for (unsigned word = 0; word < 8; ++word)
        dot = ::llaminar2::rocm_integer_contract::signedDot4(
            words[word], activation.qs[word], dot);
    const unsigned short bits = native_global_weight_read<unsigned short>(scales + offset);
    const float scale = __half2float(__builtin_bit_cast(__half, bits));
    return ::llaminar2::device_native_quantised_contract::singleScaleBlock(
        dot, scale, activation.scale);
}

/**
 * @brief Evaluate a canonical source block using the matched tuple's decoder.
 * @tparam Codebook Native physical format: Q5_1 (7) or Q8_0 (19).
 * @param fields Immutable prepared matrix fields shared by the current wave task.
 * @param column Output coordinate owned by this lane.
 * @param block Exact K32 block coordinate.
 * @param activation Canonical Q8 block, including its original code sum.
 * @return Independently rounded block value before its ordered accumulation.
 */
template<unsigned Codebook, int CodedRepresentation, class Fields>
__device__ __forceinline__ float currentDownBlock(
    const Fields& fields, unsigned column, unsigned block,
    const CanonicalFFNQ8Block& activation)
{
    static_assert(Codebook == 7 || Codebook == 19,
        "Matched diagnostic requires its original native Q5_1/Q8_0 sources");
    if constexpr (Codebook == 7) {
        int words[6];
        native_q51_block<true, CodedRepresentation>(fields, int(column), int(block), words);
        return currentDownQ51Block(words, activation);
    } else {
        return currentDownQ8Block(fields, column, block, activation);
    }
}

/**
 * @brief Complete one original K-partitioned down row entirely in registers.
 * @tparam Codebook Actual prepared physical source format.
 * @param matrix Original retained native descriptor, uniform in this wave task.
 * @param column Output coordinate; different lanes may own different columns.
 * @param width Actual source reduction width, positive and K32 aligned.
 * @param activation Complete canonical Q8 row for this expert.
 * @return The same FP32 value as canonical K32 commits and increasing-P fold.
 * @details Logical partitions come from the existing numerical authority, never
 * grid size. For widths with one block per partition, the loop retains both
 * positive-zero additions required by the original program but removes nested
 * partition bookkeeping. For longer widths only one partition accumulator is
 * live; no capacity-sized array or cross-lane reduction is necessary.
 */
template<unsigned Codebook, int CodedRepresentation, class Fields>
__device__ __forceinline__ float currentDownRowBlocks(
    const Fields& fields, unsigned column, unsigned width,
    const CanonicalFFNQ8Block* activation)
{
    namespace numeric = ::llaminar2::device_native_quantised_contract;
    namespace fp = ::llaminar2::device_fp32_contract;
    const unsigned blocks = width / 32;
    const unsigned parts = blocks < unsigned(::llaminar2::MoEProjectionNumericalContract::ordered_k_partitions)
        ? blocks : unsigned(::llaminar2::MoEProjectionNumericalContract::ordered_k_partitions);
    float result = 0.0f;
    if (blocks == parts) {
#pragma unroll 1
        for (unsigned block = 0; block < blocks; ++block) {
            const float partial = fp::multiply(numeric::accumulate(0.0f,
                currentDownBlock<Codebook, CodedRepresentation>(fields, column, block, activation[block])), 1.0f);
            result = parts == 1 ? partial : numeric::accumulate(result, partial);
        }
        return result;
    }
    const unsigned per_part = (blocks + parts - 1) / parts;
#pragma unroll 1
    for (unsigned part = 0; part < parts; ++part) {
        float partial = 0.0f;
        const unsigned first = part * per_part;
        const unsigned last = min(first + per_part, blocks);
#pragma unroll 1
        for (unsigned block = first; block < last; ++block)
            partial = numeric::accumulate(partial,
                currentDownBlock<Codebook, CodedRepresentation>(fields, column, block, activation[block]));
        partial = fp::multiply(partial, 1.0f);
        result = numeric::accumulate(result, partial);
    }
    return result;
}

/**
 * @brief Complete one row after selecting the matrix's metadata representation once.
 * @details The representation is uniform for the whole expert matrix; the
 * per-block loop keeps the original coefficient reads, the superblock-coded
 * loop reconstructs identical words (Q5_K). Arithmetic and order are shared.
 */
template<unsigned Codebook, unsigned FixedColumns=0, class Descriptor>
__device__ __forceinline__ float currentDownRow(
    const Descriptor& matrix, unsigned column, unsigned width,
    const CanonicalFFNQ8Block* activation)
{
    const auto fields=[&] {
        if constexpr(FixedColumns)
            return CurrentDownFields<Codebook==7>(matrix,std::integral_constant<unsigned,FixedColumns>{});
        else return CurrentDownFields<Codebook==7>(matrix);
    }();
    if constexpr(Codebook == 7)
        if(native_descriptor_layout(fields)!=0)
            return currentDownRowBlocks<Codebook,1>(fields,column,width,activation);
    return currentDownRowBlocks<Codebook,0>(fields,column,width,activation);
}

/**
 * @brief Produce one routed/shared FFN output in original per-row route order.
 * @param routed Retained native down descriptors indexed by expert identity.
 * @param shared Original shared-expert down matrix.
 * @param ids Original row-local selected expert IDs, independent of dedup order.
 * @param weights Original rounded selected weights in the same row-local order.
 * @param top_k Exact number of live selected routes.
 * @param column Output coordinate owned by this lane.
 * @param routed_width Real expert intermediate width on this participant.
 * @param shared_width Real shared-expert width on this participant.
 * @param routed_q8 Original route-major activation blocks.
 * @param shared_q8 Original shared-expert activation blocks.
 * @param shared_gate Original prepared scalar gate, already sigmoid activated.
 * @param observation Optional untimed diagnostic outputs from the common harness.
 * @param token Original row index for the diagnostic public planes.
 * @param hidden Actual output width; observation addressing only.
 * @tparam ObserveCanonical Compile-time diagnostic publication, absent in events.
 * @return Canonical combined output, with no intermediate global publication.
 */
template<bool ObserveCanonical = false>
__device__ __forceinline__ float currentDownColumn(
    const NativeMatrix* routed, const NativeMatrix& shared,
    const int* ids, const float* weights, unsigned top_k, unsigned column,
    unsigned routed_width, unsigned shared_width,
    const CanonicalFFNQ8Block* routed_q8, const CanonicalFFNQ8Block* shared_q8,
    float shared_gate, const qfn_castagna_native::CanonicalPortObservation& observation,
    unsigned token, unsigned hidden)
{
    namespace numeric = ::llaminar2::device_native_quantised_contract;
    namespace fp = ::llaminar2::device_fp32_contract;
    float result = 0.0f;
#pragma unroll 1
    for (unsigned route = 0; route < top_k; ++route) {
        const unsigned expert = native_weight_uniform(unsigned(ids[route]));
        const float value = currentDownRow<7>(routed + expert, column, routed_width,
            routed_q8 + size_t(route) * (routed_width / 32));
        const float weighted = numeric::weightRoute(weights[route], value);
        if constexpr (ObserveCanonical)
            if (observation.route_contributions)
                observation.route_contributions[(size_t(token) * top_k + route) * hidden + column] = weighted;
        result = numeric::accumulate(result, weighted);
    }
    const float shared_value = currentDownRow<19>(shared, column, shared_width, shared_q8);
    const float shared_complete = numeric::accumulate(0.0f,
        numeric::weightRoute(1.0f, shared_value));
    if constexpr (ObserveCanonical) {
        if (observation.routed_output)
            observation.routed_output[size_t(token) * hidden + column] = result;
        if (observation.shared_output)
            observation.shared_output[size_t(token) * hidden + column] = shared_value;
    }
    return fp::add(result, fp::multiply(shared_gate, shared_complete));
}
