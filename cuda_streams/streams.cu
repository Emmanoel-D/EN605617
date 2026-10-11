#include "assignment.h"
#include <cuda_runtime.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class Buffer {
public:
    Buffer(std::size_t bytes, bool host) : host_(host) {
        if (host_) checkCuda(cudaMallocHost(reinterpret_cast<void**>(&data_), bytes), "Allocate pinned host array");
        else checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), "Allocate device array");
    }
    ~Buffer() {
        if (host_) cudaFreeHost(data_);
        else cudaFree(data_);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    float* data() const { return data_; }
private:
    bool host_;
    float* data_ = nullptr;
};

class Stream {
public:
    Stream() { checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "Create stream"); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_{};
};

class Event {
public:
    explicit Event(unsigned int flags = cudaEventDefault) {
        checkCuda(cudaEventCreateWithFlags(&event_, flags), "Create event");
    }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_{};
};

struct Job {
    Buffer hostInput, hostOutput, deviceInput, deviceOutput;
    Stream stream;
    explicit Job(std::size_t bytes)
        : hostInput(bytes, true), hostOutput(bytes, true),
          deviceInput(bytes, false), deviceOutput(bytes, false) {}
};

struct Workspace {
    std::array<Job, kDatasetCount> jobs;
    Event start, stop;
    Event secondDone{cudaEventDisableTiming};
    explicit Workspace(std::size_t bytes) : jobs{Job(bytes), Job(bytes)} {}
    // Finish outstanding operations before releasing arrays on an error path.
    ~Workspace() { cudaDeviceSynchronize(); }
};

__global__ void transformArray(const float* input, float* output, std::size_t elements) {
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    // A grid-stride loop covers the entire array for every tested launch size.
    for (std::size_t index = first; index < elements; index += stride) {
        output[index] = calculateValue(input[index]);
    }
}

cudaDeviceProp selectDevice(const Options& options) {
    checkCuda(cudaSetDevice(0), "Select CUDA device");
    cudaDeviceProp device{};
    checkCuda(cudaGetDeviceProperties(&device, 0), "Read GPU properties");
    if (options.threads > device.maxThreadsPerBlock || options.blocks > device.maxGridSize[0]) {
        throw std::invalid_argument("The requested launch exceeds this GPU's limits.");
    }
    return device;
}

void enqueueJob(Job& job, const Options& options) {
    const std::size_t bytes = static_cast<std::size_t>(options.elements) * sizeof(float);
    checkCuda(cudaMemcpyAsync(job.deviceInput.data(), job.hostInput.data(), bytes,
                              cudaMemcpyHostToDevice, job.stream.get()), "Queue input copy");
    transformArray<<<options.blocks, options.threads, 0, job.stream.get()>>>(
        job.deviceInput.data(), job.deviceOutput.data(), options.elements);
    checkCuda(cudaGetLastError(), "Launch mathematical kernel");
    checkCuda(cudaMemcpyAsync(job.hostOutput.data(), job.deviceOutput.data(), bytes,
                              cudaMemcpyDeviceToHost, job.stream.get()), "Queue output copy");
}

float timeBatch(Workspace& workspace, const Options& options) {
    const cudaStream_t first = workspace.jobs[0].stream.get();
    const cudaStream_t second = workspace.jobs[1].stream.get();
    checkCuda(cudaEventRecord(workspace.start.get(), first), "Record batch start");
    checkCuda(cudaStreamWaitEvent(second, workspace.start.get(), 0), "Wait for batch start");
    enqueueJob(workspace.jobs[0], options);
    enqueueJob(workspace.jobs[1], options);
    checkCuda(cudaEventRecord(workspace.secondDone.get(), second), "Record second stream completion");
    // The stop event follows BOTH streams, so the time covers the complete batch.
    checkCuda(cudaStreamWaitEvent(first, workspace.secondDone.get(), 0), "Join second stream");
    checkCuda(cudaEventRecord(workspace.stop.get(), first), "Record batch stop");
    checkCuda(cudaEventSynchronize(workspace.stop.get()), "Wait for complete batch");
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, workspace.start.get(), workspace.stop.get()), "Read elapsed time");
    return milliseconds;
}

double measureBatches(Workspace& workspace, const Options& options) {
    timeBatch(workspace, options); // Warm up before the measured repetitions.
    double total = 0.0;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        total += timeBatch(workspace, options);
    }
    return total / options.iterations;
}

void printResults(const Options& options, const cudaDeviceProp& device, double milliseconds) {
    std::cout << "GPU: " << device.name << '\n'
              << "Arrays per batch: " << kDatasetCount << '\n'
              << "Elements per array: " << options.elements << '\n'
              << "Blocks per kernel: " << options.blocks << '\n'
              << "Threads per block: " << options.threads << '\n'
              << "Total threads per kernel: " << static_cast<long long>(options.blocks) * options.threads << '\n'
              << "Timing repetitions: " << options.iterations << '\n'
              << std::fixed << std::setprecision(4)
              << "Average two-stream batch time: " << milliseconds << " ms\n"
              << "Timing includes input copies, two kernel executions, and output copies.\n"
              << "Validation: both output arrays match the CPU calculation.\n";
}

} // namespace

void runAssignment(const Options& options) {
    const cudaDeviceProp device = selectDevice(options);
    const std::size_t bytes = static_cast<std::size_t>(options.elements) * sizeof(float);
    Workspace workspace(bytes);
    for (int dataset = 0; dataset < kDatasetCount; ++dataset) {
        fillInput(workspace.jobs[dataset].hostInput.data(), options.elements, dataset);
    }
    const double milliseconds = measureBatches(workspace, options);
    for (const Job& job : workspace.jobs) {
        validateOutput(job.hostInput.data(), job.hostOutput.data(), options.elements);
    }
    printResults(options, device, milliseconds);
}
