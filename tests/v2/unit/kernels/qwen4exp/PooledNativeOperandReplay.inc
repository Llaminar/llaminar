/**
 * @file PooledNativeOperandReplay.inc
 * @brief Recorded immutable operand binding replay through the actual cold production publisher.
 *
 * The canonical owning TU and original-write negative compile this same test.
 * Recorded pointer inputs grant no native readiness or received/terminal proof.
 */
namespace llaminar2::test {
/** @brief Actual cold operand publication cannot overwrite an earlier captured graph's immutable pointer/key binding. */
TEST(PooledImmutableDirectedTransport, NativeCompletionOperandCannotBeReboundAcrossVariants)
{
    std::uint64_t table = 0, admission = 0, sequence = 0, alternate = 0;
    std::int32_t error = 0, otherError = 0;
    ncclLlaminarByteCompletion receipt{}, otherReceipt{};
    ncclLlaminarCompletionOperand slot{};
    const ncclLlaminarCompletionOperand original{&table, &admission, &sequence, &error,
        &receipt, nullptr, 99, 7, 0};
    qwen4exp::detail::sealPooledNativeOperand(slot, original);
    qwen4exp::detail::sealPooledNativeOperand(slot, original);
    for (unsigned field = 0; field < 9; ++field) {
        auto changed = original;
        switch (field) {
            case 0: changed.table = &alternate; break;
            case 1: changed.admission = &alternate; break;
            case 2: changed.sequence = &alternate; break;
            case 3: changed.firstError = &otherError; break;
            case 4: changed.receipt = &otherReceipt; break;
            case 5: changed.outgoingPayloadBytes = reinterpret_cast<unsigned long long*>(&alternate); break;
            case 6: ++changed.message; break;
            case 7: ++changed.failureCode; break;
            case 8: ++changed.reserved; break;
        }
        EXPECT_THROW(qwen4exp::detail::sealPooledNativeOperand(slot, changed), std::exception);
        EXPECT_EQ(slot.table, original.table);
        EXPECT_EQ(slot.admission, original.admission);
        EXPECT_EQ(slot.sequence, original.sequence);
        EXPECT_EQ(slot.firstError, original.firstError);
        EXPECT_EQ(slot.receipt, original.receipt);
        EXPECT_EQ(slot.outgoingPayloadBytes, original.outgoingPayloadBytes);
        EXPECT_EQ(slot.message, original.message);
        EXPECT_EQ(slot.failureCode, original.failureCode);
        EXPECT_EQ(slot.reserved, original.reserved);
    }
    // Reset changes execution values, not these captured storage identities.
    // No native readiness or successful execution is implied by construction.
    for (unsigned reset = 0; reset < 20; ++reset) {
        ++table; ++admission; ++sequence; error = static_cast<std::int32_t>(reset);
        EXPECT_NO_THROW(qwen4exp::detail::sealPooledNativeOperand(slot, original));
        EXPECT_EQ(slot.table, &table);
        EXPECT_EQ(slot.firstError, &error);
        EXPECT_EQ(error, static_cast<std::int32_t>(reset));
    }
    namespace rr = qwen4exp::replicated_reader;
    const auto key = rr::nativeMessageKey(rr::Wire::Payload, {1, 2, 16, 64}, 1);
    for (const auto rows : {16u, 32u, 256u})
        EXPECT_EQ(key, rr::nativeMessageKey(rr::Wire::Payload, {1, 2, rows, 64}, 1));
    EXPECT_NE(key, rr::nativeMessageKey(rr::Wire::Header, {1, 2, 16, 64}, 1));
    EXPECT_NE(key, rr::nativeMessageKey(rr::Wire::Ack, {1, 2, 16, 64}, 1));
    EXPECT_NE(key, rr::nativeMessageKey(rr::Wire::Payload, {0, 2, 16, 64}, 1));
    EXPECT_NE(key, rr::nativeMessageKey(rr::Wire::Payload, {1, 2, 16, 32}, 1));
    EXPECT_NE(key, rr::nativeMessageKey(rr::Wire::Payload, {1, 2, 16, 64}, 2));
    ncclLlaminarCompletionOperand partial{};
    partial.message = 1;
    EXPECT_THROW(qwen4exp::detail::sealPooledNativeOperand(partial, original), std::logic_error);
    EXPECT_EQ(partial.table, nullptr);
    EXPECT_EQ(partial.message, 1u);
}
} // namespace llaminar2::test
