#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__       \
                      << ": " << cudaGetErrorString(error) << '\n';             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

constexpr int kCoefficientCount = 16;
constexpr int kMemoryModes = 3;
constexpr int kDefaultElements = 1'000'000;
constexpr int kCpuIterations = 5;
constexpr int kGpuIterations = 50;

enum MemoryMode {
    kGlobalCoefficients = 0,
    kConstantCoefficients = 1,
    kSharedCoefficients = 2
};

// Constant memory is cached and shared by every thread on the device.
__constant__ float deviceConstantCoefficients[kCoefficientCount];

__global__ void polynomialTransform(const float* input,
                                    float* output,
                                    const float* globalCoefficients,
                                    int elementCount,
                                    int memoryMode) {
    // Each block can cache the same coefficients in shared memory.
    extern __shared__ float sharedCoefficients[];

    const unsigned int localIndex = threadIdx.x;
    for (int coefficientIndex = static_cast<int>(localIndex);
         coefficientIndex < kCoefficientCount;
         coefficientIndex += static_cast<int>(blockDim.x)) {
        sharedCoefficients[coefficientIndex] = globalCoefficients[coefficientIndex];
    }
    __syncthreads();

    const unsigned long long firstIndex =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + localIndex;
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;

    for (unsigned long long index = firstIndex;
         index < static_cast<unsigned long long>(elementCount);
         index += stride) {
        // These thread-local scalar values are register candidates.
        const float x = input[index];
        float result = 0.0f;

        // All three modes execute the same polynomial. Only the coefficient
        // storage location changes, which makes the timing comparison fair.
        for (int coefficientIndex = 0;
             coefficientIndex < kCoefficientCount;
             ++coefficientIndex) {
            float coefficient = 0.0f;

            if (memoryMode == kGlobalCoefficients) {
                coefficient = globalCoefficients[coefficientIndex];
            } else if (memoryMode == kConstantCoefficients) {
                coefficient = deviceConstantCoefficients[coefficientIndex];
            } else {
                coefficient = sharedCoefficients[coefficientIndex];
            }

            result = result * x + coefficient;
        }

        output[index] = result;
    }
}

int parsePositiveInteger(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);

    if (end == text || *end != '\0' || value <= 0
        || value > std::numeric_limits<int>::max()) {
        std::cerr << "Error: " << name << " must be a positive integer.\n";
        std::exit(EXIT_FAILURE);
    }

    return static_cast<int>(value);
}

void runCpuTransform(const std::vector<float>& input,
                     std::vector<float>& output,
                     const std::array<float, kCoefficientCount>& coefficients) {
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float x = input[index];
        float result = 0.0f;

        for (const float coefficient : coefficients) {
            result = result * x + coefficient;
        }

        output[index] = result;
    }
}

float timeGpuKernel(const float* deviceInput,
                    float* deviceOutput,
                    const float* deviceGlobalCoefficients,
                    int elementCount,
                    int blocks,
                    int threadsPerBlock,
                    std::size_t sharedBytes,
                    MemoryMode memoryMode) {
    // Warm up this mode before collecting its timing.
    polynomialTransform<<<blocks, threadsPerBlock, sharedBytes>>>(
        deviceInput,
        deviceOutput,
        deviceGlobalCoefficients,
        elementCount,
        static_cast<int>(memoryMode));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int iteration = 0; iteration < kGpuIterations; ++iteration) {
        polynomialTransform<<<blocks, threadsPerBlock, sharedBytes>>>(
            deviceInput,
            deviceOutput,
            deviceGlobalCoefficients,
            elementCount,
            static_cast<int>(memoryMode));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMilliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMilliseconds / kGpuIterations;
}

float maximumDifference(const std::vector<float>& expected,
                        const std::vector<float>& actual) {
    float difference = 0.0f;

    for (std::size_t index = 0; index < expected.size(); ++index) {
        difference = std::max(difference, std::fabs(expected[index] - actual[index]));
    }

    return difference;
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <blocks> <threads_per_block> [num_elements]\n";
        return EXIT_FAILURE;
    }

    const int blocks = parsePositiveInteger(argv[1], "blocks");
    const int threadsPerBlock = parsePositiveInteger(argv[2], "threads_per_block");
    const int elementCount = argc == 4
                                 ? parsePositiveInteger(argv[3], "num_elements")
                                 : kDefaultElements;

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    if (threadsPerBlock > deviceProperties.maxThreadsPerBlock) {
        std::cerr << "Error: this GPU supports at most "
                  << deviceProperties.maxThreadsPerBlock << " threads per block.\n";
        return EXIT_FAILURE;
    }

    if (blocks > deviceProperties.maxGridSize[0]) {
        std::cerr << "Error: this GPU supports at most "
                  << deviceProperties.maxGridSize[0] << " blocks in the x dimension.\n";
        return EXIT_FAILURE;
    }

    const std::size_t dataBytes = static_cast<std::size_t>(elementCount) * sizeof(float);
    const std::size_t coefficientBytes = sizeof(float) * kCoefficientCount;
    const std::size_t sharedBytes = coefficientBytes;

    if (sharedBytes > deviceProperties.sharedMemPerBlock) {
        std::cerr << "Error: the kernel requires " << sharedBytes
                  << " bytes of shared memory, but this GPU supports "
                  << deviceProperties.sharedMemPerBlock << " bytes per block.\n";
        return EXIT_FAILURE;
    }

    std::array<float, kCoefficientCount> coefficients{};
    for (int index = 0; index < kCoefficientCount; ++index) {
        const float magnitude = 1.0f / static_cast<float>(index + 1);
        coefficients[index] = index % 2 == 0 ? magnitude : -magnitude;
    }

    // Host memory arrays.
    std::vector<float> hostInput(elementCount);
    std::vector<float> hostCpuOutput(elementCount);
    std::array<std::vector<float>, kMemoryModes> hostGpuOutputs = {
        std::vector<float>(elementCount),
        std::vector<float>(elementCount),
        std::vector<float>(elementCount)
    };

    for (int index = 0; index < elementCount; ++index) {
        hostInput[index] = static_cast<float>((index % 2048) - 1024) / 2048.0f;
    }

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < kCpuIterations; ++iteration) {
        runCpuTransform(hostInput, hostCpuOutput, coefficients);
    }
    const auto cpuStop = std::chrono::high_resolution_clock::now();
    const double averageCpuMs =
        std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count()
        / kCpuIterations;

    // Device global memory arrays.
    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;
    float* deviceGlobalCoefficients = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInput), dataBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), dataBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGlobalCoefficients),
                          coefficientBytes));

    CUDA_CHECK(cudaMemcpy(deviceInput,
                          hostInput.data(),
                          dataBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceGlobalCoefficients,
                          coefficients.data(),
                          coefficientBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(deviceConstantCoefficients,
                                  coefficients.data(),
                                  coefficientBytes));

    const std::array<MemoryMode, kMemoryModes> modes = {
        kGlobalCoefficients,
        kConstantCoefficients,
        kSharedCoefficients
    };
    const std::array<const char*, kMemoryModes> modeNames = {
        "Global coefficients",
        "Constant coefficients",
        "Shared coefficients"
    };

    std::array<float, kMemoryModes> gpuTimes{};
    std::array<float, kMemoryModes> maxDifferences{};
    bool allResultsMatch = true;

    for (int modeIndex = 0; modeIndex < kMemoryModes; ++modeIndex) {
        gpuTimes[modeIndex] = timeGpuKernel(deviceInput,
                                            deviceOutput,
                                            deviceGlobalCoefficients,
                                            elementCount,
                                            blocks,
                                            threadsPerBlock,
                                            sharedBytes,
                                            modes[modeIndex]);

        CUDA_CHECK(cudaMemcpy(hostGpuOutputs[modeIndex].data(),
                              deviceOutput,
                              dataBytes,
                              cudaMemcpyDeviceToHost));

        maxDifferences[modeIndex] =
            maximumDifference(hostCpuOutput, hostGpuOutputs[modeIndex]);
        allResultsMatch = allResultsMatch && maxDifferences[modeIndex] < 0.0001f;
    }

    const std::size_t hostBytes = dataBytes * (2 + kMemoryModes) + coefficientBytes;
    const std::size_t globalBytes = dataBytes * 2 + coefficientBytes;
    const unsigned long long launchedThreads =
        static_cast<unsigned long long>(blocks) * threadsPerBlock;

    std::cout << "GPU: " << deviceProperties.name << '\n'
              << "Elements: " << elementCount << '\n'
              << "Blocks: " << blocks << '\n'
              << "Threads per block: " << threadsPerBlock << '\n'
              << "Total launched threads: " << launchedThreads << "\n\n"
              << "CUDA memory used:\n"
              << "  Host arrays: " << hostBytes << " bytes\n"
              << "  Device global arrays: " << globalBytes << " bytes\n"
              << "  Constant coefficients: " << coefficientBytes << " bytes\n"
              << "  Shared coefficients per block: " << sharedBytes << " bytes\n"
              << "  Register candidates: x, result, coefficient, index, and stride\n\n"
              << std::fixed << std::setprecision(4)
              << "Average CPU time: " << averageCpuMs << " ms\n";

    for (int modeIndex = 0; modeIndex < kMemoryModes; ++modeIndex) {
        std::cout << "Average GPU time (" << modeNames[modeIndex] << "): "
                  << gpuTimes[modeIndex] << " ms\n";
    }

    std::cout << '\n' << std::setprecision(6);
    for (int modeIndex = 0; modeIndex < kMemoryModes; ++modeIndex) {
        std::cout << modeNames[modeIndex] << " max difference: "
                  << maxDifferences[modeIndex] << '\n';
    }

    std::cout << "All CPU/GPU results match: "
              << (allResultsMatch ? "yes" : "no") << '\n';

    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFree(deviceGlobalCoefficients));

    return allResultsMatch ? EXIT_SUCCESS : EXIT_FAILURE;
}
