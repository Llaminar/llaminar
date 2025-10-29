/**
 * @file GemmMicroKernelInit.cpp
 * @brief Forces static constructors in MicroKernel instantiations to run
 * @author David Sanftenberg
 *
 * When MicroKernel instantiations are in a static library (.a), their
 * __attribute__((constructor)) functions don't run unless symbols from
 * those objects are referenced. This file forces all instantiation objects
 * to be linked by referencing external symbols from each file.
 */

#include "GemmMicroKernelRegistry.h"

// Forward declare all force-link functions (defined in generated files with extern "C")
extern "C" {
    void forceLink_GemmMicroKernelInstantiations_00();
    void forceLink_GemmMicroKernelInstantiations_01();
    void forceLink_GemmMicroKernelInstantiations_02();
    void forceLink_GemmMicroKernelInstantiations_03();
    void forceLink_GemmMicroKernelInstantiations_04();
    void forceLink_GemmMicroKernelInstantiations_05();
    void forceLink_GemmMicroKernelInstantiations_06();
    void forceLink_GemmMicroKernelInstantiations_07();
    void forceLink_GemmMicroKernelInstantiations_08();
    void forceLink_GemmMicroKernelInstantiations_09();
    void forceLink_GemmMicroKernelInstantiations_10();
    void forceLink_GemmMicroKernelInstantiations_11();
    void forceLink_GemmMicroKernelInstantiations_12();
    void forceLink_GemmMicroKernelInstantiations_13();
    void forceLink_GemmMicroKernelInstantiations_14();
    void forceLink_GemmMicroKernelInstantiations_15();
    void forceLink_GemmMicroKernelInstantiations_16();
    void forceLink_GemmMicroKernelInstantiations_17();
    void forceLink_GemmMicroKernelInstantiations_18();
    void forceLink_GemmMicroKernelInstantiations_19();
    void forceLink_GemmMicroKernelInstantiations_20();
    void forceLink_GemmMicroKernelInstantiations_21();
    void forceLink_GemmMicroKernelInstantiations_22();
    void forceLink_GemmMicroKernelInstantiations_23();
    void forceLink_GemmMicroKernelInstantiations_24();
    void forceLink_GemmMicroKernelInstantiations_25();
    void forceLink_GemmMicroKernelInstantiations_26();
    void forceLink_GemmMicroKernelInstantiations_27();
    void forceLink_GemmMicroKernelInstantiations_28();
    void forceLink_GemmMicroKernelInstantiations_29();
    void forceLink_GemmMicroKernelInstantiations_30();
    void forceLink_GemmMicroKernelInstantiations_31();
    void forceLink_GemmMicroKernelInstantiations_32();
    void forceLink_GemmMicroKernelInstantiations_33();
    void forceLink_GemmMicroKernelInstantiations_34();
    void forceLink_GemmMicroKernelInstantiations_35();
    void forceLink_GemmMicroKernelInstantiations_36();
    void forceLink_GemmMicroKernelInstantiations_37();
    void forceLink_GemmMicroKernelInstantiations_38();
    void forceLink_GemmMicroKernelInstantiations_39();
    void forceLink_GemmMicroKernelInstantiations_40();
    void forceLink_GemmMicroKernelInstantiations_41();
    void forceLink_GemmMicroKernelInstantiations_42();
    void forceLink_GemmMicroKernelInstantiations_43();
    void forceLink_GemmMicroKernelInstantiations_44();
    void forceLink_GemmMicroKernelInstantiations_45();
    void forceLink_GemmMicroKernelInstantiations_46();
    void forceLink_GemmMicroKernelInstantiations_47();
    void forceLink_GemmMicroKernelInstantiations_48();
    void forceLink_GemmMicroKernelInstantiations_49();
    void forceLink_GemmMicroKernelInstantiations_50();
    void forceLink_GemmMicroKernelInstantiations_51();
    void forceLink_GemmMicroKernelInstantiations_52();
    void forceLink_GemmMicroKernelInstantiations_53();
    void forceLink_GemmMicroKernelInstantiations_54();
    void forceLink_GemmMicroKernelInstantiations_55();
    void forceLink_GemmMicroKernelInstantiations_56();
    void forceLink_GemmMicroKernelInstantiations_57();
    void forceLink_GemmMicroKernelInstantiations_58();
    void forceLink_GemmMicroKernelInstantiations_59();
    void forceLink_GemmMicroKernelInstantiations_60();
    void forceLink_GemmMicroKernelInstantiations_61();
    void forceLink_GemmMicroKernelInstantiations_62();
    void forceLink_GemmMicroKernelInstantiations_63();
}

namespace llaminar2 {
namespace kernels {
namespace gemm {

/**
 * Call this function before using MicroKernelRegistry to ensure
 * all template instantiations have registered themselves.
 */
void ensureMicroKernelsRegistered() {
    // Call all force-link functions (empty - just force linking)
    forceLink_GemmMicroKernelInstantiations_00();
    forceLink_GemmMicroKernelInstantiations_01();
    forceLink_GemmMicroKernelInstantiations_02();
    forceLink_GemmMicroKernelInstantiations_03();
    forceLink_GemmMicroKernelInstantiations_04();
    forceLink_GemmMicroKernelInstantiations_05();
    forceLink_GemmMicroKernelInstantiations_06();
    forceLink_GemmMicroKernelInstantiations_07();
    forceLink_GemmMicroKernelInstantiations_08();
    forceLink_GemmMicroKernelInstantiations_09();
    forceLink_GemmMicroKernelInstantiations_10();
    forceLink_GemmMicroKernelInstantiations_11();
    forceLink_GemmMicroKernelInstantiations_12();
    forceLink_GemmMicroKernelInstantiations_13();
    forceLink_GemmMicroKernelInstantiations_14();
    forceLink_GemmMicroKernelInstantiations_15();
    forceLink_GemmMicroKernelInstantiations_16();
    forceLink_GemmMicroKernelInstantiations_17();
    forceLink_GemmMicroKernelInstantiations_18();
    forceLink_GemmMicroKernelInstantiations_19();
    forceLink_GemmMicroKernelInstantiations_20();
    forceLink_GemmMicroKernelInstantiations_21();
    forceLink_GemmMicroKernelInstantiations_22();
    forceLink_GemmMicroKernelInstantiations_23();
    forceLink_GemmMicroKernelInstantiations_24();
    forceLink_GemmMicroKernelInstantiations_25();
    forceLink_GemmMicroKernelInstantiations_26();
    forceLink_GemmMicroKernelInstantiations_27();
    forceLink_GemmMicroKernelInstantiations_28();
    forceLink_GemmMicroKernelInstantiations_29();
    forceLink_GemmMicroKernelInstantiations_30();
    forceLink_GemmMicroKernelInstantiations_31();
    forceLink_GemmMicroKernelInstantiations_32();
    forceLink_GemmMicroKernelInstantiations_33();
    forceLink_GemmMicroKernelInstantiations_34();
    forceLink_GemmMicroKernelInstantiations_35();
    forceLink_GemmMicroKernelInstantiations_36();
    forceLink_GemmMicroKernelInstantiations_37();
    forceLink_GemmMicroKernelInstantiations_38();
    forceLink_GemmMicroKernelInstantiations_39();
    forceLink_GemmMicroKernelInstantiations_40();
    forceLink_GemmMicroKernelInstantiations_41();
    forceLink_GemmMicroKernelInstantiations_42();
    forceLink_GemmMicroKernelInstantiations_43();
    forceLink_GemmMicroKernelInstantiations_44();
    forceLink_GemmMicroKernelInstantiations_45();
    forceLink_GemmMicroKernelInstantiations_46();
    forceLink_GemmMicroKernelInstantiations_47();
    forceLink_GemmMicroKernelInstantiations_48();
    forceLink_GemmMicroKernelInstantiations_49();
    forceLink_GemmMicroKernelInstantiations_50();
    forceLink_GemmMicroKernelInstantiations_51();
    forceLink_GemmMicroKernelInstantiations_52();
    forceLink_GemmMicroKernelInstantiations_53();
    forceLink_GemmMicroKernelInstantiations_54();
    forceLink_GemmMicroKernelInstantiations_55();
    forceLink_GemmMicroKernelInstantiations_56();
    forceLink_GemmMicroKernelInstantiations_57();
    forceLink_GemmMicroKernelInstantiations_58();
    forceLink_GemmMicroKernelInstantiations_59();
    forceLink_GemmMicroKernelInstantiations_60();
    forceLink_GemmMicroKernelInstantiations_61();
    forceLink_GemmMicroKernelInstantiations_62();
    forceLink_GemmMicroKernelInstantiations_63();
}

} // namespace gemm
} // namespace kernels
} // namespace llaminar2
