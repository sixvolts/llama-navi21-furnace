// Run one FLASH_ATTN_EXT (Qwen3.5 shape) on the GPU backend and dump the output bytes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 1
#endif
int main(int argc, char ** argv) {
    const int D = 256, NQ = argc > 2 ? atoi(argv[2]) : 512, NKV = argc > 3 ? atoi(argv[3]) : 4096, HQ = 24, HKV = 4;
    const bool cpu = getenv("FACMP_CPU") != nullptr;
    ggml_backend_t be = cpu ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) { printf("no gpu backend\n"); return 1; }
    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, HQ, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, HKV, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, HKV, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, NKV, GGML_PAD(NQ, GGML_KQ_MASK_PAD), 1, 1);
    ggml_tensor * r = ggml_flash_attn_ext(ctx, q, k, v, m, getenv("FACMP_SCALE") ? atof(getenv("FACMP_SCALE")) : 1.0f/sqrtf((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::mt19937 rng(123); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> qf(D*NQ*HQ); for (auto & x : qf) x = nd(rng); ggml_backend_tensor_set(q, qf.data(), 0, ggml_nbytes(q));
    std::vector<ggml_fp16_t> kf((size_t) D*NKV*HKV), vf((size_t) D*NKV*HKV);
    for (auto & x : kf) x = ggml_fp32_to_fp16(nd(rng)); for (auto & x : vf) x = ggml_fp32_to_fp16(nd(rng));
    ggml_backend_tensor_set(k, kf.data(), 0, ggml_nbytes(k)); ggml_backend_tensor_set(v, vf.data(), 0, ggml_nbytes(v));
    std::vector<ggml_fp16_t> mf((size_t) NKV*GGML_PAD(NQ, GGML_KQ_MASK_PAD));
    for (int j = 0; j < GGML_PAD(NQ, GGML_KQ_MASK_PAD); ++j) for (int i = 0; i < NKV; ++i) // causal: query j sees keys <= NKV-NQ+j
        mf[(size_t) j*NKV + i] = ggml_fp32_to_fp16(i <= NKV - NQ + j ? 0.0f : -INFINITY);
    ggml_backend_tensor_set(m, mf.data(), 0, ggml_nbytes(m));
    ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, r);
    ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
    if (const char * reps = getenv("FACMP_REPS")) {   // timing: repeat the compute, report ms and TFLOPS
        const int n = atoi(reps);
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < n; ++i) ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        const double ms = (ggml_time_us() - t0) / 1000.0 / n;
        const double flops = 4.0 * NQ * NKV * D * HQ;
        printf("time %.3f ms/compute, %.1f TFLOPS (NQ=%d NKV=%d)\n", ms, flops / (ms * 1e-3) / 1e12, NQ, NKV);
    }
    std::vector<char> out(ggml_nbytes(r)); ggml_backend_tensor_get(r, out.data(), 0, out.size());
    FILE * f = fopen(argv[1], "wb"); fwrite(out.data(), 1, out.size(), f); fclose(f);
    printf("wrote %zu bytes (NQ=%d NKV=%d)\n", out.size(), NQ, NKV);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be); return 0;
}
