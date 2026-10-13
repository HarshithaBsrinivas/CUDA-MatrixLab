// Shared helpers for the CUDA-MatrixLab benchmark programs.
// Pure C++ (no CUDA headers) so the CPU baseline can use it too.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace mlab {

struct Metric {
    std::string type;   // "gflops" or "gbps"
    double value = 0.0;
};

struct RunResult {
    std::string kernel;               // e.g. "cuda_shared_matmul"
    long long n = 0;                  // matrix dimension, or element count for vector add
    int reps = 0;                     // timed repetitions (warm-up excluded)
    std::string device;               // GPU name, or "host-cpu"
    std::string compute_capability;   // e.g. "7.5"; empty for CPU
    std::string time_scope;           // what the timer covers
    std::vector<double> times_ms;
    Metric metric;
    bool verified = false;
    double max_abs_error = 0.0;
};

// Reproducible pseudo-random floats in [0, 1).
inline std::vector<float> random_floats(size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(count);
    for (auto& x : v) x = dist(rng);
    return v;
}

inline double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? 0.5 * (v[m - 1] + v[m]) : v[m];
}

inline double min_of(const std::vector<double>& v) {
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

struct CheckResult {
    bool verified;
    double max_abs_error;
};

// Spot-check C = A * B (n x n, row-major) against a double-precision reference
// computed for randomly chosen entries. A sample fails if its error exceeds
// a relative tolerance of 1e-3. NaN results always fail.
inline CheckResult check_matmul_sampled(const std::vector<float>& A,
                                        const std::vector<float>& B,
                                        const std::vector<float>& C,
                                        int n, int samples = 64, unsigned seed = 12345u) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, n - 1);
    CheckResult r{true, 0.0};
    for (int s = 0; s < samples; ++s) {
        int i = pick(rng);
        int j = pick(rng);
        double ref = 0.0;
        for (int k = 0; k < n; ++k) {
            ref += static_cast<double>(A[(size_t)i * n + k]) *
                   static_cast<double>(B[(size_t)k * n + j]);
        }
        double got = static_cast<double>(C[(size_t)i * n + j]);
        double err = std::fabs(got - ref);
        if (std::isnan(err)) err = INFINITY;
        r.max_abs_error = std::max(r.max_abs_error, err);
        double tol = 1e-3 * std::max(1.0, std::fabs(ref));
        if (!(err <= tol)) r.verified = false;
    }
    return r;
}

// Check c[i] == a[i] + b[i] for every element (tolerance 1e-6 relative).
inline CheckResult check_vector_add(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    const std::vector<float>& c) {
    CheckResult r{true, 0.0};
    for (size_t i = 0; i < c.size(); ++i) {
        double ref = static_cast<double>(a[i]) + static_cast<double>(b[i]);
        double err = std::fabs(static_cast<double>(c[i]) - ref);
        if (std::isnan(err)) err = INFINITY;
        r.max_abs_error = std::max(r.max_abs_error, err);
        double tol = 1e-6 * std::max(1.0, std::fabs(ref));
        if (!(err <= tol)) r.verified = false;
    }
    return r;
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (static_cast<unsigned char>(c) < 0x20) out.push_back(' ');
        else out.push_back(c);
    }
    return out;
}

inline std::string json_number(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream os;
    os << std::setprecision(10) << x;
    return os.str();
}

// Prints exactly one JSON object on stdout. All diagnostics must go to stderr.
inline void print_result_json(const RunResult& r) {
    std::ostringstream os;
    os << "{";
    os << "\"kernel\":\"" << json_escape(r.kernel) << "\",";
    os << "\"n\":" << r.n << ",";
    os << "\"reps\":" << r.reps << ",";
    os << "\"device\":\"" << json_escape(r.device) << "\",";
    std::string cc = r.compute_capability.empty()
                         ? std::string("null")
                         : std::string("\"") + json_escape(r.compute_capability) + "\"";
    os << "\"compute_capability\":" << cc << ",";
    os << "\"time_scope\":\"" << json_escape(r.time_scope) << "\",";
    os << "\"times_ms\":[";
    for (size_t i = 0; i < r.times_ms.size(); ++i) {
        if (i) os << ",";
        os << json_number(r.times_ms[i]);
    }
    os << "],";
    os << "\"median_ms\":" << json_number(median_of(r.times_ms)) << ",";
    os << "\"min_ms\":" << json_number(min_of(r.times_ms)) << ",";
    os << "\"metric\":{\"type\":\"" << json_escape(r.metric.type)
       << "\",\"value\":" << json_number(r.metric.value) << "},";
    os << "\"verified\":" << (r.verified ? "true" : "false") << ",";
    os << "\"max_abs_error\":" << json_number(r.max_abs_error);
    os << "}";
    std::puts(os.str().c_str());
    std::fflush(stdout);
}

}  // namespace mlab
