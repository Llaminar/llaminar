#pragma once

#ifndef HAVE_ONEDNN
#error "OneDNN support is required to use gemm_v4"
#endif

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    namespace gemm_v4
    {
        inline dnnl::engine &onednn_engine()
        {
            static dnnl::engine engine_instance(dnnl::engine::kind::cpu, 0);
            return engine_instance;
        }

        inline dnnl::stream &onednn_stream()
        {
            static dnnl::stream stream_instance(onednn_engine());
            return stream_instance;
        }

        inline bool run_onednn_int8_matmul(const int8_t *A,
                                           const int8_t *B,
                                           int32_t *C,
                                           int M,
                                           int N,
                                           int K)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::s8, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::s8, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::s32, tag::ab);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<int8_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<int8_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &)
            {
                return false;
            }

            return true;
        }

        inline bool run_onednn_fp32_matmul(const float *A,
                                           const float *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &)
            {
                return false;
            }

            return true;
        }
    } // namespace gemm_v4
} // namespace llaminar2
