/**
 * @file CUDARetainedGroupedFFNReplay.inc
 * @brief Replay QFN grouped packet publications recorded from original serial CUDA.
 *
 * Included by the retained CUDA suite to share its exact packed source recipe
 * and storage ownership. Capacity includes six exact recorded rows. The
 * admitted graph replays the full, partial and empty prefixes without adding
 * or changing a host graph node between frames.
 */

namespace llaminar2::cuda
{
    bool cudaQuantGemm_quantizeActivationsBlockwiseScoped(const float*, int8_t*, float*, int, int, int, void*);
}

namespace
{
/** @brief One original QFN physical source range and its staged output observations. */
struct GroupedTrace
{
    int width;
    uint8_t gate;
    uint8_t down;
    RetainedGroupedFFNRole role;
    std::array<uint64_t, 5> exact;
};

constexpr GroupedTrace grouped_traces[]{
    {320, 5, 7, RetainedGroupedFFNRole::Routed,
        {0xb8f224453b9f8225, 0xeef8fbf310208d31, 0xf4fe85e4afc0964c, 0x34160b5164c036d3, 0x3b96550b351865f3}},
    {320, 5, 19, RetainedGroupedFFNRole::Routed,
        {0xb8f224453b9f8225, 0xbf557f8b04f86960, 0xda9b2654f43f47be, 0x412a127abac3729e, 0x45205589d21569fa}},
    {320, 5, 23, RetainedGroupedFFNRole::Routed,
        {0xb8f224453b9f8225, 0xeef8fbf310208d31, 0xf4fe85e4afc0964c, 0x34160b5164c036d3, 0x3b96550b351865f3}},
    {320, 7, 19, RetainedGroupedFFNRole::Routed,
        {0xb8f224453b9f8225, 0x2d82a07d3bc126c9, 0x0b7e84d8fb1167ff, 0xe799bbb473a3bf3e, 0x35e3f8a8a5b60553}},
    {320, 19, 19, RetainedGroupedFFNRole::Routed,
        {0xb8f224453b9f8225, 0xd430edfaba17e3f8, 0xf2ab45308ae22458, 0xbb661c0e4f7308de, 0x108015c316489368}},
    {320, 19, 19, RetainedGroupedFFNRole::Shared,
        {0xb8f224453b9f8225, 0xd430edfaba17e3f8, 0xf2ab45308ae22458, 0xbb661c0e4f7308de, 0x108015c316489368}},
    {320, 23, 19, RetainedGroupedFFNRole::Routed,
        {0xb8f224453b9f8225, 0x2d82a07d3bc126c9, 0x0b7e84d8fb1167ff, 0xe799bbb473a3bf3e, 0x35e3f8a8a5b60553}},
    {640, 5, 7, RetainedGroupedFFNRole::Routed,
        {0x078c56d975841125, 0x356eb43ee2b19339, 0xdd5b9628258efb27, 0x3aac42018bc8975c, 0xf8cf6e8c4dd88976}},
    {640, 5, 19, RetainedGroupedFFNRole::Routed,
        {0x078c56d975841125, 0x589afcc72a12992c, 0x8f58be7a49d809d2, 0xad76ce603a5c7d91, 0x290df7a0a314cc83}},
    {640, 5, 23, RetainedGroupedFFNRole::Routed,
        {0x078c56d975841125, 0x356eb43ee2b19339, 0xdd5b9628258efb27, 0x3aac42018bc8975c, 0xf8cf6e8c4dd88976}},
    {640, 7, 19, RetainedGroupedFFNRole::Routed,
        {0x078c56d975841125, 0x417fea5cdbd89144, 0x5c6543c49275cfe8, 0x38923366f4a2222d, 0x9242f0f650e16e3b}},
    {640, 19, 19, RetainedGroupedFFNRole::Routed,
        {0x078c56d975841125, 0xce0f45196e463cd5, 0x779a5772c6acb182, 0x98b9ac6cdd3dd7ae, 0x011546f12b9d2c99}},
    {640, 19, 19, RetainedGroupedFFNRole::Shared,
        {0x078c56d975841125, 0xce0f45196e463cd5, 0x779a5772c6acb182, 0x98b9ac6cdd3dd7ae, 0x011546f12b9d2c99}},
    {640, 23, 19, RetainedGroupedFFNRole::Routed,
        {0x078c56d975841125, 0x417fea5cdbd89144, 0x5c6543c49275cfe8, 0x38923366f4a2222d, 0x9242f0f650e16e3b}},
};

/** @brief Replay one recorded grouped operand range using its actual retained kernels. */
void replayGroupedPackets(const GroupedTrace& trace, int ordinal)
{
    constexpr int capacity = 6;
    constexpr int hidden_packets = hidden / 32;
    const int packets = trace.width / 32;
    const Stream stream;
    const auto native = stream.value;
    std::vector<std::unique_ptr<Weight>> weights;
    const auto weight = [&](int n, int k, NativeQuantisedFormatInfo format, unsigned seed, bool moved) {
        weights.push_back(std::make_unique<Weight>(native, n, k, format, seed, moved));
        return weights.back()->view;
    };
    const auto gate_source = trace.gate == formats::Q4_K.codebook_id ? formats::Q4_K :
        trace.gate == formats::Q8_0.codebook_id ? formats::Q8_0 : formats::Q5_K;
    const auto down_source = trace.down == formats::Q8_0.codebook_id ? formats::Q8_0 : formats::Q5_1;
    Array<int> live{1};
    Array<float> input{capacity * hidden}, input_scales{capacity * hidden_packets},
        down_scales{std::size_t(capacity * packets)}, output{capacity * hidden};
    Array<int8_t> input_codes{capacity * hidden}, down_codes{std::size_t(capacity * trace.width)};
    Array<RetainedGroupedFFNRow> work{capacity};
    work.set(native, [&](std::size_t i) {
        const auto seed = 51u + 3u * unsigned(i);
        const auto moved_gate = trace.gate == kNativeQuantisedExpandedInt8MinCodebook;
        const auto moved_down = trace.down == kNativeQuantisedExpandedInt8MinCodebook;
        return RetainedGroupedFFNRow{{weight(trace.width, hidden, gate_source, seed, moved_gate),
                weight(trace.width, hidden, gate_source, seed + 1, moved_gate),
                weight(hidden, trace.width, down_source, seed + 2, moved_down)},
            input_codes.data() + i * hidden, input_scales.data() + i * hidden_packets,
            down_codes.data() + i * trace.width, down_scales.data() + i * packets,
            output.data() + i * hidden};
    });

    constexpr uint64_t generation = 1;
    const auto program = RetainedGroupedFFNProgram::prepare(DeviceId::cuda(ordinal),
        {work.data(), live.data(), capacity, trace.width, trace.gate, trace.down, trace.role}, native, generation);
    CUDAGraphCapture parent(native, ordinal);
    auto fragment = parent.createOrderedTimelineFragment();
    ASSERT_TRUE(fragment);
    ASSERT_TRUE(fragment->beginCapture());
    {
        ScopedCUDAFinalTimelineRecording scope(*fragment);
        program.record(native, generation);
    }
    ASSERT_TRUE(fragment->endCapture());
    const std::array steps{GPUOrderedTimelineStep{.name = "grouped_ffn", .capture = fragment.get()}};
    ASSERT_TRUE(parent.buildOrderedTimelineTransaction(steps));
    ASSERT_TRUE(parent.instantiate());

    constexpr std::array counts{0, 1, 3, 5, 6};
    for (const int count : {6, 3, 0, 5, 1, 0, 6})
    {
        SCOPED_TRACE(count);
        live.set(native, [count](std::size_t) { return count; });
        input.set(native, [count](std::size_t i) { return signedValue(i, 100 + count); });
        ASSERT_TRUE(cuda::cudaQuantGemm_quantizeActivationsBlockwiseScoped(input.data(), input_codes.data(),
            input_scales.data(), capacity, hidden, ordinal, native));
        down_codes.set(native, [](std::size_t) { return int8_t{-79}; });
        down_scales.set(native, [](std::size_t) { return -79.f; });
        output.set(native, [](std::size_t) { return -79.f; });
        ASSERT_TRUE(parent.launch());
        auto digest = down_codes.digest(native);
        digest = down_scales.digest(native, digest);
        digest = output.digest(native, digest);
        EXPECT_EQ(digest, trace.exact[std::find(counts.begin(), counts.end(), count) - counts.begin()]);
    }
}
}

/** @test Original serial bytes and guard rows survive grouped native partial, empty and repeated launches. */
TEST(CUDARetainedHCFFN, RecordedGroupedPacketPublications)
{
    const int ordinal = selectVisibleSM86();
    for (const auto& trace : grouped_traces)
    {
        SCOPED_TRACE(trace.width);
        SCOPED_TRACE(unsigned(trace.gate));
        SCOPED_TRACE(unsigned(trace.down));
        SCOPED_TRACE(unsigned(trace.role));
        ASSERT_NO_THROW(replayGroupedPackets(trace, ordinal));
    }
}
