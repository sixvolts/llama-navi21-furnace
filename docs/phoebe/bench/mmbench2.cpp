// NMAT independent quantized matmuls (distinct weights, > infinity cache) in one graph; time per matmul.
// usage: mmbench2 <type> <K> <M> <N> [NMAT=16] [REPS=10]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
int main(int argc, char ** argv) {
    const char * tn = argv[1]; const int K = atoi(argv[2]), M = atoi(argv[3]), N = atoi(argv[4]);
    const int NMAT = argc > 5 ? atoi(argv[5]) : 16, REPS = argc > 6 ? atoi(argv[6]) : 10;
    ggml_type t = !strcmp(tn,"q4_K") ? GGML_TYPE_Q4_K : !strcmp(tn,"q5_K") ? GGML_TYPE_Q5_K : !strcmp(tn,"q6_K") ? GGML_TYPE_Q6_K : !strcmp(tn,"q8_0") ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_init_params ip = { (size_t) (4*NMAT+8)*ggml_tensor_overhead() + ggml_graph_overhead_custom(4*NMAT+8, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    std::vector<ggml_tensor *> ws, rs;
    for (int i = 0; i < NMAT; ++i) { ggml_tensor * w = ggml_new_tensor_2d(ctx, t, K, M); ws.push_back(w); rs.push_back(ggml_mul_mat(ctx, w, x)); }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> wf((size_t) K*M); for (auto & v : wf) v = nd(rng);
    std::vector<uint8_t> wq(ggml_nbytes(ws[0])); ggml_quantize_chunk(t, wf.data(), wq.data(), 0, M, K, nullptr);
    for (auto * w : ws) ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    std::vector<float> xf((size_t) K*N); for (auto & v : xf) v = nd(rng); ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4*NMAT+8, false); for (auto * r : rs) ggml_build_forward_expand(gf, r);
    ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < REPS; ++i) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    const double ms = (ggml_time_us() - t0) / 1000.0 / REPS / NMAT;
    const double gb = ggml_nbytes(ws[0]) / 1e9;
    printf("%s K=%d M=%d N=%d: %.4f ms/matmul, %.0f GB/s weights, %.2f TOPS (%d matrices = %.2f GB)\n", tn, K, M, N, ms, gb/(ms*1e-3), 2.0*K*M*N/(ms*1e-3)/1e12, NMAT, gb*NMAT);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be); return 0;
}
