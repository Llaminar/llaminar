/**
 * @file SnapshotSemanticAxis.h
 * @brief Stable acquired diagnostic axes, independent of stage interfaces.
 *
 * Existing values retain their serialized identity. Authored token rows carry
 * per-request original coordinates rather than pretending packed rows are a prefix.
 */
#pragma once
namespace llaminar2 {
/** @brief Mathematical axis retained after execution bindings leave scope. */
enum class SnapshotSemanticAxis : unsigned char {
    Legacy,TokenRows,StatePrefixRows,CountedSelectionRows,AuthoredTokenRows
};
}
