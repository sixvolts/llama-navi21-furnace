// Experiment: Q4_K weights repacked so that a wave streams contiguous memory with one row per
// lane, activations (q8_1) broadcast through the scalar path. Compares against a double
// reference and reports time for the model's shapes.
//   repack_bench <K> <M> <N> [NMAT=16] [REPS=10]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

// block_q4_K: half2 dm (4 B), uint8 scales[12], uint8 qs[128] = 144 B = 9 pieces of 16 B
static constexpr int Q4K_BYTES  = 144;
static constexpr int Q4K_PIECES = 9;
static constexpr int ROWS_PER_WAVE = 32;

__device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }

__device__ __forceinline__ int byte_of(uint32_t w, int i) { return (w >> (8*i)) & 0xFF; }

// get_scale_min_k4 from the 16-byte header {dm, scales[12]} held as four dwords
__device__ __forceinline__ void scale_min(const uint4 & h, int j, int & sc, int & m) {
    if (j < 4) {
        sc = byte_of(h.y, j) & 63;
        m  = byte_of(h.z, j) & 63;
    } else {
        const int q4 = byte_of(h.w, j - 4);
        const int q0 = byte_of(h.y, j - 4);
        const int q1 = byte_of(h.z, j - 4);
        sc = (q4 & 0xF) | ((q0 >> 6) << 4);
        m  = (q4 >>  4) | ((q1 >> 6) << 4);
    }
}

__device__ __forceinline__ float2 half2_to_float2(uint32_t w) {
    const __half2 h = *reinterpret_cast<const __half2 *>(&w);
    return make_float2(__low2float(h), __high2float(h));
}

// wp: repacked weights [tile][sb][piece][32 lanes][16 B]
// yq: activations [col][sb][8 blocks][8 ints] (q8_1 quants, 32 per block); yds: [col][sb][8] half2 {d, sum}
template <int NCOLS>
__global__ __launch_bounds__(32) void q4k_rowpack(const uint4 * __restrict__ wp, const int * __restrict__ yq,
                                                  const uint32_t * __restrict__ yds, float * __restrict__ dst,
                                                  const int nsb, const int M) {
    const int r    = threadIdx.x;
    const int tile = blockIdx.x;
    const int nsplit = gridDim.y;
    const int b0 = (int) ((int64_t) nsb *  blockIdx.y      / nsplit);
    const int b1 = (int) ((int64_t) nsb * (blockIdx.y + 1) / nsplit);
    const uint4 * base = wp + (size_t) tile * nsb * Q4K_PIECES * ROWS_PER_WAVE;
    dst += (size_t) blockIdx.y * NCOLS * M;   // partial sums per split

    float acc[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) acc[j] = 0.0f;

    for (int b = b0; b < b1; ++b) {
        const uint4 * blk = base + (size_t) b * Q4K_PIECES * ROWS_PER_WAVE;
        const uint4 hdr = blk[r];
        const float2 dm = half2_to_float2(hdr.x);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint4 q0 = blk[(1 + 2*c) * ROWS_PER_WAVE + r];
            const uint4 q1 = blk[(2 + 2*c) * ROWS_PER_WAVE + r];
            int vlo[8], vhi[8];
            vlo[0] =  q0.x       & 0x0F0F0F0F; vlo[1] =  q0.y       & 0x0F0F0F0F; vlo[2] =  q0.z       & 0x0F0F0F0F; vlo[3] =  q0.w       & 0x0F0F0F0F;
            vlo[4] =  q1.x       & 0x0F0F0F0F; vlo[5] =  q1.y       & 0x0F0F0F0F; vlo[6] =  q1.z       & 0x0F0F0F0F; vlo[7] =  q1.w       & 0x0F0F0F0F;
            vhi[0] = (q0.x >> 4) & 0x0F0F0F0F; vhi[1] = (q0.y >> 4) & 0x0F0F0F0F; vhi[2] = (q0.z >> 4) & 0x0F0F0F0F; vhi[3] = (q0.w >> 4) & 0x0F0F0F0F;
            vhi[4] = (q1.x >> 4) & 0x0F0F0F0F; vhi[5] = (q1.y >> 4) & 0x0F0F0F0F; vhi[6] = (q1.z >> 4) & 0x0F0F0F0F; vhi[7] = (q1.w >> 4) & 0x0F0F0F0F;

            int sc, m;
            scale_min(hdr, 2*c,     sc, m);
            const float rs0 = dm.x * (float) sc, rm0 = dm.y * (float) m;
            scale_min(hdr, 2*c + 1, sc, m);
            const float rs1 = dm.x * (float) sc, rm1 = dm.y * (float) m;

#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                // uniform addresses: the compiler lowers these to scalar loads
                const int      * u0 = yq  + ((size_t) (j*nsb + b)*8 + 2*c) * 8;
                const int      * u1 = u0 + 8;
                const uint32_t   d0 = yds[(size_t) (j*nsb + b)*8 + 2*c];
                const uint32_t   d1 = yds[(size_t) (j*nsb + b)*8 + 2*c + 1];
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    s0 = dp4a(vlo[i], u0[i], s0);
                    s1 = dp4a(vhi[i], u1[i], s1);
                }
                const float2 ds0 = half2_to_float2(d0);
                const float2 ds1 = half2_to_float2(d1);
#ifdef EXACT_SUM
                int q0s = 0, q1s = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) { q0s = dp4a(0x01010101, u0[i], q0s); q1s = dp4a(0x01010101, u1[i], q1s); }
                const float sum0 = ds0.x * (float) q0s, sum1 = ds1.x * (float) q1s;
#else
                const float sum0 = ds0.y, sum1 = ds1.y;
#endif
                acc[j] = fmaf((float) s0, rs0 * ds0.x, acc[j]);
                acc[j] = fmaf(-rm0, sum0, acc[j]);
                acc[j] = fmaf((float) s1, rs1 * ds1.x, acc[j]);
                acc[j] = fmaf(-rm1, sum1, acc[j]);
            }
        }
    }

    const int row = tile*ROWS_PER_WAVE + r;
    if (row < M) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) dst[(size_t) j*M + row] = acc[j];
    }
}

__global__ void reduce_splits(const float * __restrict__ part, float * __restrict__ dst, const int n, const int nsplit) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int k = 0; k < nsplit; ++k) s += part[(size_t) k*n + i];
    dst[i] = s;
}

int main(int argc, char ** argv) {
    const int K = atoi(argv[1]), M = atoi(argv[2]), N = atoi(argv[3]);
    const int NMAT = argc > 4 ? atoi(argv[4]) : 16, REPS = argc > 5 ? atoi(argv[5]) : 10;
    const int NSPLIT = argc > 6 ? atoi(argv[6]) : 1;
    const int nsb = K / 256, ntiles = (M + 31) / 32;

    // random weights, quantized with ggml
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> wf((size_t) K*M); for (auto & v : wf) v = nd(rng);
    const size_t wbytes = (size_t) M * nsb * Q4K_BYTES;
    std::vector<uint8_t> wq(wbytes);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, wf.data(), wq.data(), 0, M, K, nullptr);

    // repack: [tile][sb][piece][lane][16 B]
    std::vector<uint8_t> wp((size_t) ntiles * nsb * Q4K_PIECES * 32 * 16, 0);
    for (int t = 0; t < ntiles; ++t)
        for (int b = 0; b < nsb; ++b)
            for (int p = 0; p < Q4K_PIECES; ++p)
                for (int r = 0; r < 32; ++r) {
                    const int row = t*32 + r; if (row >= M) continue;
                    memcpy(wp.data() + ((((size_t) t*nsb + b)*Q4K_PIECES + p)*32 + r)*16, wq.data() + ((size_t) row*nsb + b)*Q4K_BYTES + p*16, 16);
                }

    // activations: quantize to q8_1 per 32 (d = amax/127, sum = sum of x), layout [col][k]
    std::vector<float> xf((size_t) K*N); for (auto & v : xf) v = nd(rng);
    std::vector<int8_t> yq((size_t) N*K); std::vector<uint32_t> yds((size_t) N*K/32);
    std::vector<float> xdq((size_t) K*N);   // what the quantized activations represent
    for (int j = 0; j < N; ++j)
        for (int b = 0; b < K/32; ++b) {
            float amax = 0.f, sum = 0.f;
            for (int i = 0; i < 32; ++i) { const float v = xf[(size_t) j*K + b*32 + i]; amax = fmaxf(amax, fabsf(v)); sum += v; }
            const float d = amax / 127.f;
            for (int i = 0; i < 32; ++i) {
                const float v = xf[(size_t) j*K + b*32 + i];
                const int8_t q = amax == 0.f ? 0 : (int8_t) roundf(v / d);
                yq[(size_t) j*K + b*32 + i] = q; xdq[(size_t) j*K + b*32 + i] = d * q;
            }
            const ggml_fp16_t dh = ggml_fp32_to_fp16(d), sh = ggml_fp32_to_fp16(sum);
            uint16_t dh16, sh16; memcpy(&dh16, &dh, 2); memcpy(&sh16, &sh, 2);
            yds[(size_t) j*(K/32) + b] = (uint32_t) dh16 | ((uint32_t) sh16 << 16);
        }

    // reference: dequantized weights x original activations, double
    std::vector<float> wdq((size_t) K*M);
    ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(wq.data(), wdq.data(), (int64_t) K*M);
    std::vector<double> ref((size_t) N*M), ref2((size_t) N*M);
    for (int j = 0; j < N; ++j)
        for (int row = 0; row < M; ++row) {
            double s = 0, s2 = 0; const float * w = wdq.data() + (size_t) row*K; const float * x = xf.data() + (size_t) j*K; const float * x2 = xdq.data() + (size_t) j*K;
            for (int k = 0; k < K; ++k) { s += (double) w[k] * x[k]; s2 += (double) w[k] * x2[k]; }
            ref[(size_t) j*M + row] = s; ref2[(size_t) j*M + row] = s2;
        }

    // device buffers: NMAT copies of the repacked weights (distinct, > cache)
    std::vector<uint4 *> d_w(NMAT);
    for (int i = 0; i < NMAT; ++i) { CHECK(hipMalloc(&d_w[i], wp.size())); CHECK(hipMemcpy(d_w[i], wp.data(), wp.size(), hipMemcpyHostToDevice)); }
    int * d_yq; uint32_t * d_yds; float * d_out;
    CHECK(hipMalloc(&d_yq, yq.size())); CHECK(hipMemcpy(d_yq, yq.data(), yq.size(), hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_yds, yds.size()*4)); CHECK(hipMemcpy(d_yds, yds.data(), yds.size()*4, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_out, (size_t) N*M*4));
    float * d_part; CHECK(hipMalloc(&d_part, (size_t) NSPLIT*N*M*4));
    float * d_k = NSPLIT > 1 ? d_part : d_out;
    const dim3 grid(ntiles, NSPLIT);

    auto launch = [&](int i) {
        switch (N) {
            case 1: q4k_rowpack<1><<<grid, 32>>>(d_w[i], d_yq, d_yds, d_k, nsb, M); break;
            case 2: q4k_rowpack<2><<<grid, 32>>>(d_w[i], d_yq, d_yds, d_k, nsb, M); break;
            case 4: q4k_rowpack<4><<<grid, 32>>>(d_w[i], d_yq, d_yds, d_k, nsb, M); break;
            case 6: q4k_rowpack<6><<<grid, 32>>>(d_w[i], d_yq, d_yds, d_k, nsb, M); break;
            case 8: q4k_rowpack<8><<<grid, 32>>>(d_w[i], d_yq, d_yds, d_k, nsb, M); break;
            default: fprintf(stderr, "N must be 1,2,4,6,8\n"); exit(1);
        }
        if (NSPLIT > 1) reduce_splits<<<(N*M + 255)/256, 256>>>(d_part, d_out, N*M, NSPLIT);
    };
    launch(0); CHECK(hipDeviceSynchronize());
    std::vector<float> out((size_t) N*M); CHECK(hipMemcpy(out.data(), d_out, out.size()*4, hipMemcpyDeviceToHost));
    double maxerr = 0, maxerr2 = 0, rms = 0;
    for (size_t i = 0; i < out.size(); ++i) { maxerr = fmax(maxerr, fabs(out[i] - ref[i])); maxerr2 = fmax(maxerr2, fabs(out[i] - ref2[i])); rms += ref[i]*ref[i]; }
    rms = sqrt(rms / out.size());

    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    CHECK(hipEventRecord(e0));
    for (int rep = 0; rep < REPS; ++rep) for (int i = 0; i < NMAT; ++i) launch(i);
    CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
    float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); ms /= REPS * NMAT;
    printf("q4_K rowpack K=%d M=%d N=%d split=%d: %.4f ms/matmul, %.0f GB/s weights; err vs true %.2e, vs dequantized-activation ref %.2e (of rms %.3e)\n",
           K, M, N, NSPLIT, ms, wbytes / 1e9 / (ms*1e-3), maxerr/rms, maxerr2/rms, rms);
    return 0;
}
