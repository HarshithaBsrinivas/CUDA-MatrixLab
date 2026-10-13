// Sequential CPU matrix multiplication baseline: C = A * B
// (n x n, row-major float). Single-threaded, loop order i-k-j for
// cache-friendly row access.
//
// Usage:  matrix_cpu <n> [reps]
// Prints exactly ONE JSON line to stdout; diagnostics go to stderr.
#include "common.h"

#include <chrono>
#include <cstdlib>

static void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        float* crow = C + (size_t)i * n;
        for (int k = 0; k < n; ++k) {
            const float a = A[(size_t)i * n + k];
            const float* brow = B + (size_t)k * n;
            for (int j = 0; j < n; ++j) crow[j] += a * brow[j];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <n> [reps]\n", argv[0]);
        return 1;
    }
    const int n = std::atoi(argv[1]);
    const int reps = (argc >= 3) ? std::atoi(argv[2]) : 3;
    if (n <= 0 || reps <= 0) {
        std::fprintf(stderr, "n and reps must be positive integers\n");
        return 1;
    }

    const size_t count = (size_t)n * n;
    auto A = mlab::random_floats(count, 1u);
    auto B = mlab::random_floats(count, 2u);
    std::vector<float> C(count, 0.0f);

    // Warm-up run (not recorded).
    matmul_cpu(A.data(), B.data(), C.data(), n);

    std::vector<double> times;
    for (int r = 0; r < reps; ++r) {
        std::fill(C.begin(), C.end(), 0.0f);
        auto t0 = std::chrono::steady_clock::now();
        matmul_cpu(A.data(), B.data(), C.data(), n);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
        std::fprintf(stderr, "rep %d: %.3f ms\n", r + 1, ms);
    }

    auto check = mlab::check_matmul_sampled(A, B, C, n);

    mlab::RunResult res;
    res.kernel = "cpu_matmul";
    res.n = n;
    res.reps = reps;
    res.device = "host-cpu";
    res.compute_capability = "";
    res.time_scope = "wall_clock_compute_only";
    res.times_ms = times;
    double med = mlab::median_of(times);
    res.metric.type = "gflops";
    res.metric.value = 2.0 * n * (double)n * n / (med * 1e-3) / 1e9;
    res.verified = check.verified;
    res.max_abs_error = check.max_abs_error;

    mlab::print_result_json(res);
    return 0;
}
