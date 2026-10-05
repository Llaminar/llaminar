/**
 * @file NativeCollectiveRowsContract.h
 * @brief Shared device-free admission for equal-prefix native collective banks.
 *
 * LocalTP and native coordinators reject malformed geometry before submission.
 * These checks never inspect device values, choose transport, or admit memory.
 * Exact in-place allreduce is legal; gather/scatter require disjoint banks so
 * inactive physical tails and every other participant's bank remain untouched.
 */
#pragma once
#include "ICollectiveBackend.h"
#include <cstdint>
#include <limits>

namespace llaminar2
{
    /**
     * @brief Validate native operation/type, byte ranges and explicit ordering.
     * @param operation Equal-prefix native operation.
     * @param send Input range; never dereferenced by this check.
     * @param receive Output range; never dereferenced by this check.
     * @param rows Frozen capacity/width and optional borrowed device count.
     * @param dtype Exact native scalar type.
     * @param reduction SUM/MIN/MAX, or ALLGATHER for gather.
     * @param degree Number of communicator participants, at least two.
     * @param participant Calling communicator coordinate.
     * @param stream Explicit non-null ordering stream.
     * @return True only for a representable, non-overlapping native contract.
     */
    inline bool nativeCollectiveRowsValid(NativeRowCollective operation,
        const void *send, void *receive, const NativeCollectiveRows &rows,
        CollectiveDataType dtype, CollectiveOp reduction, int degree,
        int participant, void *stream) noexcept
    {
        if (!send || !receive || !stream || degree < 2 || participant < 0 || participant >= degree)
            return false;
        std::size_t element_bytes = 0;
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
        case CollectiveDataType::INT32: element_bytes = 4; break;
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16: element_bytes = 2; break;
        case CollectiveDataType::INT8: element_bytes = 1; break;
        default: return false;
        }
        int send_banks = 1, receive_banks = 1;
        switch (operation)
        {
        case NativeRowCollective::AllGather:
            if (reduction != CollectiveOp::ALLGATHER) return false;
            receive_banks = degree;
            break;
        case NativeRowCollective::ReduceScatter: send_banks = degree; [[fallthrough]];
        case NativeRowCollective::AllReduce:
            if (reduction != CollectiveOp::ALLREDUCE_SUM && reduction != CollectiveOp::ALLREDUCE_MIN &&
                reduction != CollectiveOp::ALLREDUCE_MAX) return false;
            break;
        default: return false;
        }
        if (!rows.byteGeometryValid(element_bytes, degree)) return false;
        const auto count_address = reinterpret_cast<std::uintptr_t>(rows.rows().countOwner());
        if (count_address % alignof(std::int32_t)) return false;
        const auto a = reinterpret_cast<std::uintptr_t>(send);
        const auto b = reinterpret_cast<std::uintptr_t>(receive);
        const auto maximum = std::numeric_limits<std::uintptr_t>::max();
        const auto send_bytes = rows.bankElements() * element_bytes * send_banks;
        const auto receive_bytes = rows.bankElements() * element_bytes * receive_banks;
        if (a % element_bytes || b % element_bytes || a > maximum - send_bytes || b > maximum - receive_bytes)
            return false;
        // The native output may not overwrite its count authority. Checking
        // intervals, rather than tensor IDs, also catches aliases/views.
        if (count_address && (count_address > maximum - sizeof(std::int32_t) ||
            (count_address < b + receive_bytes && b < count_address + sizeof(std::int32_t)))) return false;
        const auto receipt = reinterpret_cast<std::uintptr_t>(rows.payloadReceipt());
        if (receipt && (receipt % alignof(unsigned long long) ||
            receipt > maximum - sizeof(unsigned long long) ||
            (receipt < a + send_bytes && a < receipt + sizeof(unsigned long long)) ||
            (receipt < b + receive_bytes && b < receipt + sizeof(unsigned long long)) ||
            (count_address < receipt + sizeof(unsigned long long) && receipt < count_address + sizeof(std::int32_t))))
            return false;
        return (a == b && operation == NativeRowCollective::AllReduce) ||
            a + send_bytes <= b || b + receive_bytes <= a;
    }
}
