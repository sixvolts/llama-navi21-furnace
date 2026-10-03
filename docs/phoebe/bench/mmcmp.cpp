// One MUL_MAT (quantized weights x N f32 columns) on the GPU backend; dumps the f32 output bytes.
// usage: mmcmp <out.bin> <type: q4_K|q5_K|q6_K|q8_0|q4_0> <K> <M rows> <N cols> [seed]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cstdint>
int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    const char * tn = argv[2]; const int K = atoi(argv[3]), M = atoi(argv[4]), N = atoi(argv[5]); const int seed = argc > 6 ? atoi(argv[6]) : 7;
    ggml_type t = !strcmp(tn,"q4_K") ? GGML_TYPE_Q4_K : !strcmp(tn,"q5_K") ? GGML_TYPE_Q5_K : !strcmp(tn,"q6_K") ? GGML_TYPE_Q6_K : !strcmp(tn,"q8_0") ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) { printf("no gpu backend\n"); return 1; }
    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, t, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * r = ggml_mul_mat(ctx, w, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> wf((size_t) K*M); for (auto & v : wf) v = nd(rng);
    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(t, wf.data(), wq.data(), 0, M, K, nullptr);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    std::vector<float> xf((size_t) K*N); for (auto & v : xf) v = nd(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
    ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, r);
    ggml_backend_graph_compute(be, gf);
    if (const char * reps = getenv("MMCMP_REPS")) {
        const int n = atoi(reps); ggml_backend_synchronize(be);
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < n; ++i) ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        const double ms = (ggml_time_us() - t0) / 1000.0 / n;
        const double gb = ggml_nbytes(w) / 1e9;
        printf("time %.4f ms, %.0f GB/s weights, %.1f TOPS (%s K=%d M=%d N=%d)\n", ms, gb / (ms*1e-3), 2.0*K*M*N/(ms*1e-3)/1e12, tn, K, M, N);
    }
    std::vector<float> out((size_t) M*N); ggml_backend_tensor_get(r, out.data(), 0, ggml_nbytes(r));
    FILE * f = fopen(argv[1], "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
    printf("wrote %zu floats (%s K=%d M=%d N=%d)\n", out.size(), tn, K, M, N);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be); return 0;
}
