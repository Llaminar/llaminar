# Changelog

## [0.2.0](https://github.com/Llaminar/llaminar/compare/v0.1.0...v0.2.0) (2026-04-17)


### Features

* activation rotation for Q8_1 quantization fidelity ([2c9a32b](https://github.com/Llaminar/llaminar/commit/2c9a32bc97694a5cb0ad14aab9d6f02d1aa41888))
* add --serve HTTP server mode with E2E integration tests ([a03cf07](https://github.com/Llaminar/llaminar/commit/a03cf074e956c90dcbedc86c929834cb77438c67))
* Add kernel profiling and benchmarking capabilities ([bf3f583](https://github.com/Llaminar/llaminar/commit/bf3f58344299f8f786b09fbd365ff3bbfb186cb9))
* **chat:** Add ChatTemplate for chat formatting support ([55c7265](https://github.com/Llaminar/llaminar/commit/55c7265806fb6f6e0180edab8ac886c12d1f54c0))
* **cuda:** per-format GEMV dispatch heuristic (16 formats, ML-based sweep) ([dd58e6f](https://github.com/Llaminar/llaminar/commit/dd58e6f12f55f8eb0b4a0f0388e41d0591b1c6ba))
* Enhance MPI context handling for TensorFactory and KV cache initialization ([c76304a](https://github.com/Llaminar/llaminar/commit/c76304a8d2fbb72ef86d74eff5508dbf564c30b4))
* fuse weight rotation into VNNI packing (eliminate Q8_0 copies) ([dfd1235](https://github.com/Llaminar/llaminar/commit/dfd12351e82e8b735863a4c4b9a4d0d2036ea117))
* **hook:** enable parallel execution for integration tests in pre-commit hook ([e022a26](https://github.com/Llaminar/llaminar/commit/e022a26d337c5fc9a9913783d8ddadebdf11966e))
* Implement automatic KernelFactory cache invalidation on TensorBase destruction and add corresponding unit tests ([a401535](https://github.com/Llaminar/llaminar/commit/a4015351ec0f9ffe02129f84010c7800a3096606))
* Implement graph caching optimization for decode mode to enhance performance ([00f9ae8](https://github.com/Llaminar/llaminar/commit/00f9ae8b81f7012da6168f6f09f7a663dc9ba789))
* Implement Qwen2 model executor with compute graph orchestration ([c54e80a](https://github.com/Llaminar/llaminar/commit/c54e80adab021b057118821b3340059cdc694e31))
* kernel dynamic state lifecycle for multi-turn inference ([8666332](https://github.com/Llaminar/llaminar/commit/8666332fbec3b96965f0c72c51e68ef484ec50fe))
* **kv-cache:** Q16_1 KV cache — VNNI int16 decode + tiled flash prefill ([189ca69](https://github.com/Llaminar/llaminar/commit/189ca6948166b109d5dcb15f90beedb1c2fc5a4f))
* Qwen3 architecture support ([708bcda](https://github.com/Llaminar/llaminar/commit/708bcda7a84b19c74641f5d6c16afc77b8631334))
* **qwen35:** add comprehensive snapshot hooks for GDN parity testing ([13cbbe9](https://github.com/Llaminar/llaminar/commit/13cbbe9e406919e7679cbe35c0d823fc5309bfb1))
* **qwen35:** Qwen3.5 single-device CPU parity — GDN + FA hybrid attention ([df4b66b](https://github.com/Llaminar/llaminar/commit/df4b66be999d52dccd7d2057133e108a16439481))
* R2 logit comparison diagnostic in CompletionMode ([a4fd4b3](https://github.com/Llaminar/llaminar/commit/a4fd4b37aad7453d03b7ea93cff2277fd00447a3))
* **rocm:** multi-stream concurrent fused GEMM prefill (default ON) ([d185e3c](https://github.com/Llaminar/llaminar/commit/d185e3c69b932645572be61183fe449e448b012e))
* **tokenizer:** Integrate chat templates + fix BPE encoding bugs ([88cd86f](https://github.com/Llaminar/llaminar/commit/88cd86fb4ddf61ffb3562f32e97f1b5caf6248f7))
* **tp:** Polymorphic ITPContext, NodeLocalTP, Qwen3.5 TP support ([362fd6b](https://github.com/Llaminar/llaminar/commit/362fd6bb31fddf8d0baebffea2266a4753156501))
* **v2:** TurboQuant KV cache, RoPE-on-read for CPU, CUDA kernel refactoring ([c02a742](https://github.com/Llaminar/llaminar/commit/c02a7424541b8bff21f21082ec803cd505f70477))


### Bug Fixes

* Always read row 0 in sampleGreedyOnDevice(). ([519e219](https://github.com/Llaminar/llaminar/commit/519e219a1e3fb40ee9067360f98331127090606e))
* BAR tensor coherence, same-device copy, kv_cache_scale, PP transfer reliability ([ab034f7](https://github.com/Llaminar/llaminar/commit/ab034f7b55820462b435717e156e51499c5ec956))
* capture-stream dangling pointer and multi-device orchestration fixes ([ee698fc](https://github.com/Llaminar/llaminar/commit/ee698fc3a18d50e84c7219e82b7f40a0e542b8a8))
* **cpu:** Q4_0 fused GEMV M=1 decode bug + parity CSV export + regression test ([30dd571](https://github.com/Llaminar/llaminar/commit/30dd57147882ee7992b631e6d17e875d92494507))
* decode K/V from stale activation buffer + cleanup diagnostic code ([2757d48](https://github.com/Llaminar/llaminar/commit/2757d4818c98a2b65d08d830549ae21d3e691839))
* embedding stale token_ids + attention stream fencing + cache invalidation ([eeca83d](https://github.com/Llaminar/llaminar/commit/eeca83dd717a43abf6dbfb87ee7b5f287d7461f5))
* **exp:** correct range reduction in SiLU/sigmoid AVX512 exp() polynomial ([b9e19c7](https://github.com/Llaminar/llaminar/commit/b9e19c70eaea03912062fe7b54ae414f6ab9dff0))
* gate fastDecode() profiling on LLAMINAR_PROFILING env var ([7d65436](https://github.com/Llaminar/llaminar/commit/7d654365a70fa7016a6956ee5aefa7259ca0bd9a))
* GPU argmax reads row 0 where LmHeadStage writes logits ([519e219](https://github.com/Llaminar/llaminar/commit/519e219a1e3fb40ee9067360f98331127090606e))
* GPU Q8_1 KV cache prefill accuracy + Q8_0 parity thresholds ([dc7af97](https://github.com/Llaminar/llaminar/commit/dc7af977d5cc904cccdda01986439656a53d5b30))
* GQA-aware FusedQKV sub-block sharding for Qwen3.5 TP ([fa2623c](https://github.com/Llaminar/llaminar/commit/fa2623cc375f940bb87747f95e6fb3500b6f9d7f))
* **graph-capture:** dynamic KV conversion for CUDA graph replay correctness ([a7f9f79](https://github.com/Llaminar/llaminar/commit/a7f9f7921967d6ba45992e33c331edd10e78f788))
* **mpi:** batched execution parity fixes with unit tests ([91b50ae](https://github.com/Llaminar/llaminar/commit/91b50aefddf0b61fe5159a339c387a52eff64280))
* Qwen3.5 GDN tensor parallel + test hardening ([352b13e](https://github.com/Llaminar/llaminar/commit/352b13e691173ef7d314aaa2ac4d45de1e05f0d9))
* **qwen35:** fix attn_output buffer size for GDN+FA hybrid TP ([9b8b0a2](https://github.com/Llaminar/llaminar/commit/9b8b0a2c67aad91c7e64fce7d540ac410c73274a))
* remove weight rotation preprocessor (Q16_1 parity regression) ([24b8f43](https://github.com/Llaminar/llaminar/commit/24b8f431097aedd09636b57ddcf32c3811482814))
* **rocm:** correct iq_apply_signs_4 scatter constant 0x08040201→0x00204081 ([8b583f8](https://github.com/Llaminar/llaminar/commit/8b583f892f95cb0e6181e3a032376294277e6049))
* route Q8_0 to INT8-VNNI path instead of native-VNNI ([42b94ce](https://github.com/Llaminar/llaminar/commit/42b94ce23cfb827e98719ee99d92798a67b3ecab))
* **test:** update KVCachePP test to use KV_CACHE_APPEND stage type ([38e5b07](https://github.com/Llaminar/llaminar/commit/38e5b077a4424bb81a4cdf55e2b1e45939e531ca))
* **v2:** CPU TP crash/deadlock + profiling overhaul + fast decode schedule ([c40ad1f](https://github.com/Llaminar/llaminar/commit/c40ad1fb0f2da8938d2d55af8d76f62122d556f0))
* wire sampling params through all execution modes + extract testable ChatCompletionHandler ([57b935b](https://github.com/Llaminar/llaminar/commit/57b935b798859663c27248580863df582476aa60))


### Performance Improvements

* **attention:** Optimize prefill V accumulation and hoist division ([14fb80d](https://github.com/Llaminar/llaminar/commit/14fb80dcc8d153de29160baadfebebcc87bc7c81))
* AVX-512 vectorization + OMP optimization for Qwen3.5 GDN kernels ([603adf9](https://github.com/Llaminar/llaminar/commit/603adf9eff8e324507693272e90e9af049d937e2))
* blockwise native-VNNI perf test — GEMV + GEMM sweep across 15 formats ([ce589d8](https://github.com/Llaminar/llaminar/commit/ce589d8805903c06f4f7e52196b72482bd510a20))
* **cuda,rocm:** redirect all mapped output GEMM paths to HBM staging ([5c6a903](https://github.com/Llaminar/llaminar/commit/5c6a90326b8667ae6087241e6970e28b5c5b8372))
* **cuda:** FA2 template optimization + warp-cooperative quantization ([d31eea3](https://github.com/Llaminar/llaminar/commit/d31eea3016fe132646f2f3ad622e5f3f073e1cc4))
* **cuda:** optimize decode kernels — fused RMS+RoPE, GEMV tuning, adaptive flash attention splits ([aeb83f0](https://github.com/Llaminar/llaminar/commit/aeb83f0d5082a0bef7187dce927b2924ce11d80b))
* FA2 FP16 KV direct path + GEMM BM=128 tile with occupancy tuning ([844729e](https://github.com/Llaminar/llaminar/commit/844729e2e1e89e0d280d36a6b2222df42ff24a73))
* **gemv:** F16C hardware FP16 + 4-block unroll (+7.6% decode) ([e3b4ad3](https://github.com/Llaminar/llaminar/commit/e3b4ad31c37ff254d67270bbe216db87f4edeb86))
* **GEMV:** multi-GPU + HipBLAS cosine verification for throughput test ([464fd9b](https://github.com/Llaminar/llaminar/commit/464fd9bf94274c3acd92d480e814ac992b8ba620))
* **GEMV:** relax cosine gate for IQ1_M (0.98 vs 0.99) ([4bb9458](https://github.com/Llaminar/llaminar/commit/4bb945862f2ecd3b757211df19251129960cf009))
* **native-vnni:** move IQ1_M from Pattern DA to Pattern D (+30% GEMM) ([ec1f82b](https://github.com/Llaminar/llaminar/commit/ec1f82b38d79b71b84627f17767b36c9a236da02))
* optimize Qwen3.5 CPU kernels + branch-aware pre-commit hook ([5b0a93d](https://github.com/Llaminar/llaminar/commit/5b0a93d88daeb65b1f868d2980522f887fe3efb2))
* **rocm:** INT8-input batched scatter GEMV — quantize-once + batched QKV/GateUp dispatch ([bc34158](https://github.com/Llaminar/llaminar/commit/bc34158970f59f2cf37d1408abff8b583cf6a635))
* **rocm:** redirect GEMV decode output from mapped→HBM (+10.3% decode) ([065a038](https://github.com/Llaminar/llaminar/commit/065a038c60022122371865f25b51f156e3240d57))
* **rocm:** self-reducing scatter kernel + hybrid dispatch (+30% cumulative) ([b42c128](https://github.com/Llaminar/llaminar/commit/b42c12866ba952b1f8d5ad9814b7c017e106d4bc))
* **vnni:** set n_block_chunks=1 for FFN GEMV + GEMM dispatch ([61bc493](https://github.com/Llaminar/llaminar/commit/61bc493c52d14b934d4c1a77bcc00299b1a21bef))


### Refactors

* decouple graph/schema factories via registries + coherence hardening ([7949c94](https://github.com/Llaminar/llaminar/commit/7949c9416a991399a5c765ada3c45ce1ed21cffe))
* migrate TensorBase coherence to explicit state machine + TransferEngine ([55cc8ad](https://github.com/Llaminar/llaminar/commit/55cc8ad374b8b178bad2e4eb870e88a68469ce66))
* move GEMM/GEMV CUDA kernels into kernels/cuda/gemm/ subfolder ([e462943](https://github.com/Llaminar/llaminar/commit/e4629436e35457c89d069262a18028f1bd0e795a))
* **Qwen2Graph:** declarative migration — remove imperative patterns ([fac6cc0](https://github.com/Llaminar/llaminar/commit/fac6cc0051ac1a8d59a5d4da2146db7d9e364d34))
* remove cooperative prefill kernel (replaced by rocBLAS two-pass) ([3ec7677](https://github.com/Llaminar/llaminar/commit/3ec7677a67ceeafa0f0328cef1cd1ab4bbb47103))
* StageRunPolicy unified execution + fix GPU token duplication ([74de820](https://github.com/Llaminar/llaminar/commit/74de820ec527b5a8d1c67d8200a5bb00cca3f630))
* **tests:** move E2E parity tests to Integration category ([7c70f21](https://github.com/Llaminar/llaminar/commit/7c70f21f33ac6b6920802841fc6873265da889ad))
* update attention kernel usage to CPUAttentionKernelTyped for better compatibility ([ea768b2](https://github.com/Llaminar/llaminar/commit/ea768b2910c00f4a0cb2cc77bd79a07ced0cbba3))


### Reverts

* restore original pre-commit hook (full suite on every commit) ([52653d2](https://github.com/Llaminar/llaminar/commit/52653d2fdb6f41da080b5c9e195c785ca32c98c2))


### Documentation

* Add chat mode implementation plan ([7292d11](https://github.com/Llaminar/llaminar/commit/7292d113b6113732ab2b06efa99c3990315a98b8))
* add CSV output documentation to parity testing section ([b91ddc5](https://github.com/Llaminar/llaminar/commit/b91ddc5376d7ac6b66f1a467e5c553eb5830590f))
* add final native-VNNI perf results and ISA audit to expansion doc ([88715fb](https://github.com/Llaminar/llaminar/commit/88715fbe38d68ee59ddbd639bd925c74d69a5bd6))
* GRAPH_DECOUPLING_PLAN.md, COHERENCE_HARDENING_PLAN.md, MDO_DECOUPLING_PLAN.md ([7949c94](https://github.com/Llaminar/llaminar/commit/7949c9416a991399a5c765ada3c45ce1ed21cffe))
* update README with Qwen3.5 GDN, execution modes, TurboQuant, and project structure ([e19cdab](https://github.com/Llaminar/llaminar/commit/e19cdabc29d4a46ed777b683a905643fe88d554f))
