/**
 * @file IntegerGemmKernelInit.cpp
 * @brief Forces static constructors in Integer GEMM instantiations to run
 * @author David Sanftenberg
 *
 * When Integer GEMM instantiations are in a static library (.a), their
 * __attribute__((constructor)) functions don't run unless symbols from
 * those objects are referenced. This file forces all instantiation objects
 * to be linked by referencing external symbols from each file.
 */

#include "IntegerGemmKernelRegistry.h"

// Forward declare all force-link functions (defined in generated files with extern "C")
extern "C"
{
    void forceLink_IntegerGemmInstantiations_00();
    void forceLink_IntegerGemmInstantiations_01();
    void forceLink_IntegerGemmInstantiations_02();
    void forceLink_IntegerGemmInstantiations_03();
    void forceLink_IntegerGemmInstantiations_04();
    void forceLink_IntegerGemmInstantiations_05();
    void forceLink_IntegerGemmInstantiations_06();
    void forceLink_IntegerGemmInstantiations_07();
    void forceLink_IntegerGemmInstantiations_08();
    void forceLink_IntegerGemmInstantiations_09();
    void forceLink_IntegerGemmInstantiations_10();
    void forceLink_IntegerGemmInstantiations_11();
    void forceLink_IntegerGemmInstantiations_12();
    void forceLink_IntegerGemmInstantiations_13();
    void forceLink_IntegerGemmInstantiations_14();
    void forceLink_IntegerGemmInstantiations_15();
    void forceLink_IntegerGemmInstantiations_16();
    void forceLink_IntegerGemmInstantiations_17();
    void forceLink_IntegerGemmInstantiations_18();
    void forceLink_IntegerGemmInstantiations_19();
    void forceLink_IntegerGemmInstantiations_20();
    void forceLink_IntegerGemmInstantiations_21();
    void forceLink_IntegerGemmInstantiations_22();
    void forceLink_IntegerGemmInstantiations_23();
    void forceLink_IntegerGemmInstantiations_24();
    void forceLink_IntegerGemmInstantiations_25();
    void forceLink_IntegerGemmInstantiations_26();
    void forceLink_IntegerGemmInstantiations_27();
    void forceLink_IntegerGemmInstantiations_28();
    void forceLink_IntegerGemmInstantiations_29();
    void forceLink_IntegerGemmInstantiations_30();
    void forceLink_IntegerGemmInstantiations_31();
    void forceLink_IntegerGemmInstantiations_32();
    void forceLink_IntegerGemmInstantiations_33();
    void forceLink_IntegerGemmInstantiations_34();
    void forceLink_IntegerGemmInstantiations_35();
    void forceLink_IntegerGemmInstantiations_36();
    void forceLink_IntegerGemmInstantiations_37();
    void forceLink_IntegerGemmInstantiations_38();
    void forceLink_IntegerGemmInstantiations_39();
    void forceLink_IntegerGemmInstantiations_40();
    void forceLink_IntegerGemmInstantiations_41();
    void forceLink_IntegerGemmInstantiations_42();
    void forceLink_IntegerGemmInstantiations_43();
    void forceLink_IntegerGemmInstantiations_44();
    void forceLink_IntegerGemmInstantiations_45();
    void forceLink_IntegerGemmInstantiations_46();
    void forceLink_IntegerGemmInstantiations_47();
    void forceLink_IntegerGemmInstantiations_48();
    void forceLink_IntegerGemmInstantiations_49();
    void forceLink_IntegerGemmInstantiations_50();
    void forceLink_IntegerGemmInstantiations_51();
    void forceLink_IntegerGemmInstantiations_52();
    void forceLink_IntegerGemmInstantiations_53();
    void forceLink_IntegerGemmInstantiations_54();
    void forceLink_IntegerGemmInstantiations_55();
    void forceLink_IntegerGemmInstantiations_56();
    void forceLink_IntegerGemmInstantiations_57();
    void forceLink_IntegerGemmInstantiations_58();
    void forceLink_IntegerGemmInstantiations_59();
    void forceLink_IntegerGemmInstantiations_60();
    void forceLink_IntegerGemmInstantiations_61();
    void forceLink_IntegerGemmInstantiations_62();
    void forceLink_IntegerGemmInstantiations_63();
}

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * Call this function before using IntegerGemmKernelRegistry to ensure
             * all template instantiations have registered themselves.
             */
            void ensureIntegerGemmKernelsRegistered()
            {
                // Call all force-link functions (empty - just force linking)
                forceLink_IntegerGemmInstantiations_00();
                forceLink_IntegerGemmInstantiations_01();
                forceLink_IntegerGemmInstantiations_02();
                forceLink_IntegerGemmInstantiations_03();
                forceLink_IntegerGemmInstantiations_04();
                forceLink_IntegerGemmInstantiations_05();
                forceLink_IntegerGemmInstantiations_06();
                forceLink_IntegerGemmInstantiations_07();
                forceLink_IntegerGemmInstantiations_08();
                forceLink_IntegerGemmInstantiations_09();
                forceLink_IntegerGemmInstantiations_10();
                forceLink_IntegerGemmInstantiations_11();
                forceLink_IntegerGemmInstantiations_12();
                forceLink_IntegerGemmInstantiations_13();
                forceLink_IntegerGemmInstantiations_14();
                forceLink_IntegerGemmInstantiations_15();
                forceLink_IntegerGemmInstantiations_16();
                forceLink_IntegerGemmInstantiations_17();
                forceLink_IntegerGemmInstantiations_18();
                forceLink_IntegerGemmInstantiations_19();
                forceLink_IntegerGemmInstantiations_20();
                forceLink_IntegerGemmInstantiations_21();
                forceLink_IntegerGemmInstantiations_22();
                forceLink_IntegerGemmInstantiations_23();
                forceLink_IntegerGemmInstantiations_24();
                forceLink_IntegerGemmInstantiations_25();
                forceLink_IntegerGemmInstantiations_26();
                forceLink_IntegerGemmInstantiations_27();
                forceLink_IntegerGemmInstantiations_28();
                forceLink_IntegerGemmInstantiations_29();
                forceLink_IntegerGemmInstantiations_30();
                forceLink_IntegerGemmInstantiations_31();
                forceLink_IntegerGemmInstantiations_32();
                forceLink_IntegerGemmInstantiations_33();
                forceLink_IntegerGemmInstantiations_34();
                forceLink_IntegerGemmInstantiations_35();
                forceLink_IntegerGemmInstantiations_36();
                forceLink_IntegerGemmInstantiations_37();
                forceLink_IntegerGemmInstantiations_38();
                forceLink_IntegerGemmInstantiations_39();
                forceLink_IntegerGemmInstantiations_40();
                forceLink_IntegerGemmInstantiations_41();
                forceLink_IntegerGemmInstantiations_42();
                forceLink_IntegerGemmInstantiations_43();
                forceLink_IntegerGemmInstantiations_44();
                forceLink_IntegerGemmInstantiations_45();
                forceLink_IntegerGemmInstantiations_46();
                forceLink_IntegerGemmInstantiations_47();
                forceLink_IntegerGemmInstantiations_48();
                forceLink_IntegerGemmInstantiations_49();
                forceLink_IntegerGemmInstantiations_50();
                forceLink_IntegerGemmInstantiations_51();
                forceLink_IntegerGemmInstantiations_52();
                forceLink_IntegerGemmInstantiations_53();
                forceLink_IntegerGemmInstantiations_54();
                forceLink_IntegerGemmInstantiations_55();
                forceLink_IntegerGemmInstantiations_56();
                forceLink_IntegerGemmInstantiations_57();
                forceLink_IntegerGemmInstantiations_58();
                forceLink_IntegerGemmInstantiations_59();
                forceLink_IntegerGemmInstantiations_60();
                forceLink_IntegerGemmInstantiations_61();
                forceLink_IntegerGemmInstantiations_62();
                forceLink_IntegerGemmInstantiations_63();
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
