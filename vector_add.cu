// CUDA vector addition: c[i] = a[i] + b[i].
//
// Usage:  vector_add <num_elements> [reps]
// Timing uses CUDA events around the kernel launch only; host<->device
// copies are excluded and reported separately in the stderr log.
// Prints exactly ONE JSON line to stdout; diagnostics go to stderr.
#include "common.h"
#include "common_cuda.h"

#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(const float* a, const float* b, float* c, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <num_elements> [reps]\n", argv[0]);
        return 1;
    }
    const long long nll = std::atoll(argv[1]);
    const int reps = (argc >= 3) ? std::atoi(argv[2]) : 10;
    if (nll <= 0 || reps <= 0) {
        std::fprintf(stderr, "num_elements and reps must be positive\n");
        return 1;
    }
    const size_t n = static_cast<size_t>(nll);
    const size_t bytes = n * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "Device: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    auto ha = mlab::random_floats(n, 1u);
    auto hb = mlab::random_floats(n, 2u);
    std::vector<float> hc(n, 0.0f);

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    cudaEvent_t h2d0, h2d1;
    CUDA_CHECK(cudaEventCreate(&h2d0));
    CUDA_CHECK(cudaEventCreate(&h2d1));
    CUDA_CHECK(cudaEventRecord(h2d0));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d1));
    CUDA_CHECK(cudaEventSynchronize(h2d1));
    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d0, h2d1));
    std::fprintf(stderr, "Host->device copy (not timed in results): %.3f ms\n", h2d_ms);

    const int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((n + threads - 1) / threads);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up launch (not recorded).
    vectorAddKernel<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> times;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        vectorAddKernel<<<blocks, threads>>>(da, db, dc, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(static_cast<double>(ms));
        std::fprintf(stderr, "rep %d: %.4f ms\n", r + 1, ms);
    }

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));
    auto check = mlab::check_vector_add(ha, hb, hc);

    mlab::RunResult res;
    res.kernel = "cuda_vector_add";
    res.n = static_cast<long long>(n);
    res.reps = reps;
    res.device = prop.name;
    res.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    res.time_scope = "kernel_only_cuda_events";
    res.times_ms = times;
    double med_s = mlab::median_of(times) * 1e-3;
    res.metric.type = "gbps";
    res.metric.value = 3.0 * static_cast<double>(bytes) / med_s / 1e9;   // read a, read b, write c
    res.verified = check.verified;
    res.max_abs_error = check.max_abs_error;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(h2d0));
    CUDA_CHECK(cudaEventDestroy(h2d1));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    mlab::print_result_json(res);
    return 0;
}
