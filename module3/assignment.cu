#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// data size value
constexpr std::size_t DATA_SIZE = 1'000'000;
constexpr int CPU_REPEATS = 5;
constexpr int GPU_REPEATS = 20;
constexpr int GPU_WARMUP_RUNS = 2;

constexpr float INPUT_A_WEIGHT = 0.45F;
constexpr float INPUT_B_WEIGHT = 0.35F;
constexpr float INPUT_C_WEIGHT = 0.20F;
constexpr float BRANCH_THRESHOLD = 50.0F;
constexpr float HIGH_VALUE_MULTIPLIER = 1.12F;
constexpr float LOW_VALUE_MULTIPLIER = 0.98F;
constexpr float VALIDATION_TOLERANCE = 0.001F;

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(error));                \
        }                                                                       \
    } while (false)

struct Timings {
    double cpuBranchlessMs;
    float gpuBranchlessMs;
    double cpuBranchingMs;
    float gpuBranchingMs;
};

int parsePositiveInteger(const char* text, const char* argumentName) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (text == end || *end != '\0' || value <= 0 || value > 2'000'000'000L) {
        throw std::invalid_argument(std::string(argumentName) +
                                    " must be a positive integer");
    }
    return static_cast<int>(value);
}

// A simple weighted calculation performed for every element in the data set.
__host__ __device__ inline float calculateWeightedValue(float inputA,
                                                         float inputB,
                                                         float inputC) {
    return inputA * INPUT_A_WEIGHT +
           inputB * INPUT_B_WEIGHT +
           inputC * INPUT_C_WEIGHT;
}

NOINLINE void calculateBranchlessOnCpu(const float* inputA,
                                       const float* inputB,
                                       const float* inputC,
                                       float* output,
                                       std::size_t sampleCount) {
    for (std::size_t index = 0; index < sampleCount; ++index) {
        output[index] = calculateWeightedValue(inputA[index],
                                               inputB[index],
                                               inputC[index]);
    }
}

NOINLINE void calculateBranchingOnCpu(const float* inputA,
                                      const float* inputB,
                                      const float* inputC,
                                      float* output,
                                      std::size_t sampleCount) {
    for (std::size_t index = 0; index < sampleCount; ++index) {
        const float weightedValue = calculateWeightedValue(inputA[index],
                                                           inputB[index],
                                                           inputC[index]);

        if (inputA[index] >= BRANCH_THRESHOLD) {
            output[index] = weightedValue * HIGH_VALUE_MULTIPLIER;
        } else {
            output[index] = weightedValue * LOW_VALUE_MULTIPLIER;
        }
    }
}

__global__ void calculateBranchlessOnGpu(const float* inputA,
                                         const float* inputB,
                                         const float* inputC,
                                         float* output,
                                         std::size_t sampleCount) {
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;

    for (; index < sampleCount; index += stride) {
        output[index] = calculateWeightedValue(inputA[index],
                                               inputB[index],
                                               inputC[index]);
    }
}

__global__ void calculateBranchingOnGpu(const float* inputA,
                                        const float* inputB,
                                        const float* inputC,
                                        float* output,
                                        std::size_t sampleCount) {
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;

    for (; index < sampleCount; index += stride) {
        const float weightedValue = calculateWeightedValue(inputA[index],
                                                           inputB[index],
                                                           inputC[index]);

        // Randomized input values place neighboring threads on both paths.
        if (inputA[index] >= BRANCH_THRESHOLD) {
            output[index] = weightedValue * HIGH_VALUE_MULTIPLIER;
        } else {
            output[index] = weightedValue * LOW_VALUE_MULTIPLIER;
        }
    }
}

template <typename CpuFunction>
double measureCpu(CpuFunction function,
                  const std::vector<float>& inputA,
                  const std::vector<float>& inputB,
                  const std::vector<float>& inputC,
                  std::vector<float>& output) {
    double totalMilliseconds = 0.0;

    for (int repeat = 0; repeat < CPU_REPEATS; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        function(inputA.data(), inputB.data(), inputC.data(), output.data(),
                 DATA_SIZE);
        const auto stop = std::chrono::steady_clock::now();
        totalMilliseconds +=
            std::chrono::duration<double, std::milli>(stop - start).count();
    }

    return totalMilliseconds / CPU_REPEATS;
}

float measureBranchlessGpu(const float* deviceInputA,
                           const float* deviceInputB,
                           const float* deviceInputC,
                           float* deviceOutput,
                           int blocks,
                           int threadsPerBlock) {
    for (int repeat = 0; repeat < GPU_WARMUP_RUNS; ++repeat) {
        calculateBranchlessOnGpu<<<blocks, threadsPerBlock>>>(
            deviceInputA, deviceInputB, deviceInputC, deviceOutput, DATA_SIZE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int repeat = 0; repeat < GPU_REPEATS; ++repeat) {
        calculateBranchlessOnGpu<<<blocks, threadsPerBlock>>>(
            deviceInputA, deviceInputB, deviceInputC, deviceOutput, DATA_SIZE);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&totalMilliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return totalMilliseconds / GPU_REPEATS;
}

float measureBranchingGpu(const float* deviceInputA,
                          const float* deviceInputB,
                          const float* deviceInputC,
                          float* deviceOutput,
                          int blocks,
                          int threadsPerBlock) {
    for (int repeat = 0; repeat < GPU_WARMUP_RUNS; ++repeat) {
        calculateBranchingOnGpu<<<blocks, threadsPerBlock>>>(
            deviceInputA, deviceInputB, deviceInputC, deviceOutput, DATA_SIZE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int repeat = 0; repeat < GPU_REPEATS; ++repeat) {
        calculateBranchingOnGpu<<<blocks, threadsPerBlock>>>(
            deviceInputA, deviceInputB, deviceInputC, deviceOutput, DATA_SIZE);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&totalMilliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return totalMilliseconds / GPU_REPEATS;
}

bool outputsMatch(const std::vector<float>& cpuOutput,
                  const std::vector<float>& gpuOutput,
                  float& maximumDifference) {
    maximumDifference = 0.0F;
    for (std::size_t index = 0; index < DATA_SIZE; ++index) {
        const float difference = std::fabs(cpuOutput[index] - gpuOutput[index]);
        maximumDifference = std::max(maximumDifference, difference);
        if (difference > VALIDATION_TOLERANCE) {
            return false;
        }
    }
    return true;
}

void fillInputData(std::vector<float>& inputA,
                   std::vector<float>& inputB,
                   std::vector<float>& inputC) {
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> valueDistribution(0.0F, 100.0F);

    for (std::size_t index = 0; index < DATA_SIZE; ++index) {
        inputA[index] = valueDistribution(generator);
        inputB[index] = valueDistribution(generator);
        inputC[index] = valueDistribution(generator);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <total CUDA threads> <threads per block>\n"
                  << "Example: " << argv[0] << " 512 256\n";
        return 1;
    }

    try {
        const int requestedTotalThreads =
            parsePositiveInteger(argv[1], "Total CUDA threads");
        const int threadsPerBlock =
            parsePositiveInteger(argv[2], "Threads per block");

        cudaDeviceProp deviceProperties{};
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

        if (threadsPerBlock > deviceProperties.maxThreadsPerBlock) {
            throw std::invalid_argument(
                "Threads per block exceeds this GPU's supported maximum of " +
                std::to_string(deviceProperties.maxThreadsPerBlock));
        }

        const int blocks =
            (requestedTotalThreads + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > deviceProperties.maxGridSize[0]) {
            throw std::invalid_argument("Requested block count is too large");
        }

        const int launchedThreads = blocks * threadsPerBlock;
        const std::size_t bytes = DATA_SIZE * sizeof(float);

        std::vector<float> inputA(DATA_SIZE);
        std::vector<float> inputB(DATA_SIZE);
        std::vector<float> inputC(DATA_SIZE);
        std::vector<float> cpuBranchlessOutput(DATA_SIZE);
        std::vector<float> cpuBranchingOutput(DATA_SIZE);
        std::vector<float> gpuOutput(DATA_SIZE);
        fillInputData(inputA, inputB, inputC);

        float* deviceInputA = nullptr;
        float* deviceInputB = nullptr;
        float* deviceInputC = nullptr;
        float* deviceOutput = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInputA), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInputB), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInputC), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), bytes));

        CUDA_CHECK(cudaMemcpy(deviceInputA, inputA.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceInputB, inputB.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceInputC, inputC.data(), bytes,
                              cudaMemcpyHostToDevice));

        Timings timings{};
        timings.cpuBranchlessMs =
            measureCpu(calculateBranchlessOnCpu, inputA, inputB, inputC,
                       cpuBranchlessOutput);
        timings.gpuBranchlessMs =
            measureBranchlessGpu(deviceInputA, deviceInputB, deviceInputC,
                                 deviceOutput, blocks, threadsPerBlock);
        CUDA_CHECK(cudaMemcpy(gpuOutput.data(), deviceOutput, bytes,
                              cudaMemcpyDeviceToHost));

        float branchlessMaximumDifference = 0.0F;
        const bool branchlessMatches =
            outputsMatch(cpuBranchlessOutput, gpuOutput,
                         branchlessMaximumDifference);

        timings.cpuBranchingMs =
            measureCpu(calculateBranchingOnCpu, inputA, inputB, inputC,
                       cpuBranchingOutput);
        timings.gpuBranchingMs =
            measureBranchingGpu(deviceInputA, deviceInputB, deviceInputC,
                                deviceOutput, blocks, threadsPerBlock);
        CUDA_CHECK(cudaMemcpy(gpuOutput.data(), deviceOutput, bytes,
                              cudaMemcpyDeviceToHost));

        float branchingMaximumDifference = 0.0F;
        const bool branchingMatches =
            outputsMatch(cpuBranchingOutput, gpuOutput,
                         branchingMaximumDifference);

        CUDA_CHECK(cudaFree(deviceInputA));
        CUDA_CHECK(cudaFree(deviceInputB));
        CUDA_CHECK(cudaFree(deviceInputC));
        CUDA_CHECK(cudaFree(deviceOutput));

        std::cout << "GPU: " << deviceProperties.name << '\n'
                  << "Data size: " << DATA_SIZE << " elements\n"
                  << "Requested total threads: " << requestedTotalThreads << '\n'
                  << "Threads per block: " << threadsPerBlock << '\n'
                  << "Blocks: " << blocks << '\n'
                  << "Actually launched threads: " << launchedThreads << "\n\n";

        std::cout << std::fixed << std::setprecision(4)
                  << "Average CPU branchless: " << timings.cpuBranchlessMs
                  << " ms\n"
                  << "Average GPU branchless kernel: " << timings.gpuBranchlessMs
                  << " ms\n"
                  << "Average CPU branching: " << timings.cpuBranchingMs
                  << " ms\n"
                  << "Average GPU branching kernel: " << timings.gpuBranchingMs
                  << " ms\n\n"
                  << "Branchless CPU/GPU match: "
                  << (branchlessMatches ? "yes" : "NO")
                  << " (max difference " << branchlessMaximumDifference << ")\n"
                  << "Branching CPU/GPU match: "
                  << (branchingMatches ? "yes" : "NO")
                  << " (max difference " << branchingMaximumDifference << ")\n";

        return (branchlessMatches && branchingMatches) ? 0 : 2;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
}
