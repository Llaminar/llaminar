/**
 * @file GDNLinkedStateLayout.h
 * @brief GPU entrypoints for modulo-linked GDN state layout conversion.
 *
 * Raw device allgather publishes participant banks in rank-major order.  The
 * recurrence and short-convolution kernels consume global semantic-group
 * order.  These explicit-stream wrappers perform the required device-side
 * permutation without host staging or synchronization.
 */

#pragma once

#ifdef HAVE_CUDA
extern "C"
{
    /**
     * @brief Reassemble rank-major CUDA allgather output into group-major state.
     */
    bool cudaGDN_reassemble_modulo_linked_state(
        const float *gathered,
        float *full,
        int degree,
        int global_key_heads,
        int global_value_heads,
        int key_elements_per_head,
        int value_elements_per_head,
        int prefix_group_count,
        int device_idx,
        void *stream);

}
#endif

#ifdef HAVE_ROCM
extern "C"
{
    /**
     * @brief Reassemble rank-major ROCm allgather output into group-major state.
     */
    bool rocmGDN_reassemble_modulo_linked_state(
        const float *gathered,
        float *full,
        int degree,
        int global_key_heads,
        int global_value_heads,
        int key_elements_per_head,
        int value_elements_per_head,
        int prefix_group_count,
        int device_idx,
        void *stream);

}
#endif
