#pragma once

#include <cmath>
#include <cstddef>

constexpr int kDatasetCount = 2;
constexpr int kMathTerms = 24;
constexpr float kValidationTolerance = 0.0001f;

struct Options {
    int blocks = 32;
    int threads = 128;
    int elements = 250000;
    int iterations = 10;
    bool help = false;
};

#if defined(__CUDACC__)
__host__ __device__
#endif
inline float calculateValue(float input) {
    float result = 0.0f;
    for (int term = 1; term <= kMathTerms; ++term) {
        const float shifted = input + 0.125f * term;
        result += ::sinf(shifted) * ::sqrtf(shifted * shifted + 1.0f) / term;
    }
    return result;
}

Options parseOptions(int argc, char** argv);
void printUsage(const char* executable);
void fillInput(float* input, std::size_t elements, int dataset);
void validateOutput(const float* input, const float* output, std::size_t elements);
void runAssignment(const Options& options);
