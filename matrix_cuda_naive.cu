// Naive CUDA matrix multiplication: each thread computes one element of C
// by reading A and B directly from global memory.
//
// Usage:  matrix_cuda_naive <n> [reps]
// Timing uses CUDA events around the kernel only (copies excluded).
// Prints exactly ONE JSON line to stdout; diagnostics go to stderr.
#include "common.h"
#include "common_cuda.h"

#include <cstdlib>
#include <cuda_runtime.h>

constexpr int BLOCK = 16;

__global__ void matmulNaiveKernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <n> [reps]\n", argv[0]);
        return 1;
    }
    const int n = std::atoi(argv[1]);
    const int reps = (argc >= 3) ? std::atoi(argv[2]) : 5;
    if (n <= 0 || reps <= 0) {
        std::fprintf(stderr, "n and reps must be positive\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "Device: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);
    auto ha = mlab::random_floats(count, 1u);
    auto hb = mlab::random_floats(count, 2u);
    std::vector<float> hc(count, 0.0f);

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK, BLOCK);
    dim3 blocks((n + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

    // Warm-up launch (not recorded).
    matmulNaiveKernel<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        matmulNaiveKernel<<<blocks, threads>>>(da, db, dc, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(static_cast<double>(ms));
        std::fprintf(stderr, "rep %d: %.4f ms\n", r + 1, ms);
    }

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));
    auto check = mlab::check_matmul_sampled(ha, hb, hc, n);

    mlab::RunResult res;
    res.kernel = "cuda_naive_matmul";
    res.n = n;
    res.reps = reps;
    res.device = prop.name;
    res.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    res.time_scope = "kernel_only_cuda_events";
    res.times_ms = times;
    double med_s = mlab::median_of(times) * 1e-3;
    res.metric.type = "gflops";
    res.metric.value = 2.0 * n * (double)n * n / med_s / 1e9;
    res.verified = check.verified;
    res.max_abs_error = check.max_abs_error;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    mlab::print_result_json(res);
    return 0;
}
