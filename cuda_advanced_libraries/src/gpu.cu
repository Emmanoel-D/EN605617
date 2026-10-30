#include "gpu.hpp"
#include "reference.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace audio {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void fft_check(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuFFT error " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class Stream {
public:
    Stream() { cuda_check(cudaStreamCreate(&value), "create stream"); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t value = nullptr;
};

class Plan {
public:
    Plan(const Layout& l, bool inverse, cudaStream_t stream) {
        int size[] = {l.fft};
        int real_embed[] = {l.fft};
        int complex_embed[] = {l.bins};
        const cufftType type = inverse ? CUFFT_C2R : CUFFT_R2C;
        fft_check(cufftPlanMany(&value, 1, size,
                  inverse ? complex_embed : real_embed, 1,
                  inverse ? l.bins : l.fft,
                  inverse ? real_embed : complex_embed, 1,
                  inverse ? l.fft : l.bins, type, l.frames), "plan FFT");
        try {
            fft_check(cufftSetStream(value, stream), "set FFT stream");
        } catch (...) {
            cufftDestroy(value);
            throw;
        }
    }
    ~Plan() { cufftDestroy(value); }
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    cufftHandle value = 0;
};

class Timer {
public:
    Timer() {
        cuda_check(cudaEventCreate(&begin), "create start event");
        try {
            cuda_check(cudaEventCreate(&end), "create stop event");
        } catch (...) {
            cudaEventDestroy(begin);
            throw;
        }
    }
    ~Timer() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    void start(cudaStream_t stream) {
        cuda_check(cudaEventRecord(begin, stream), "record start");
    }
    float stop(cudaStream_t stream) {
        cuda_check(cudaEventRecord(end, stream), "record stop");
        cuda_check(cudaEventSynchronize(end), "wait for pipeline");
        float result = 0.0f;
        cuda_check(cudaEventElapsedTime(&result, begin, end), "elapsed time");
        return result;
    }
private:
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
};

template <class T>
T* raw(thrust::device_vector<T>& values) {
    return thrust::raw_pointer_cast(values.data());
}

__host__ __device__ float window(int i, int size) {
    return sinf(static_cast<float>(kPi) * (i + 0.5f) / size);
}

// A virtual index packs overlapping frames directly in device memory.
struct PackFrame {
    const float* samples;
    Layout l;
    __host__ __device__ float operator()(int index) const {
        const int i = index % l.fft;
        const int source = (index / l.fft) * l.hop - l.hop + i;
        const float value = source < 0 || source >= l.samples ?
                            0.0f : samples[source];
        return value * window(i, l.fft);
    }
};

// Bin-major virtual keys group every calibration frame for the same bin.
struct NoiseKey {
    int frames;
    __host__ __device__ int operator()(int index) const {
        return index / frames;
    }
};

struct NoisePower {
    const cufftComplex* spectrum;
    Layout l;
    __host__ __device__ float operator()(int index) const {
        const int bin = index / l.noise_frames;
        const int frame = 1 + index % l.noise_frames;
        const auto value = spectrum[frame * l.bins + bin];
        return (value.x * value.x + value.y * value.y) / l.noise_frames;
    }
};

// Real gains preserve phase and the Hermitian symmetry needed by C2R.
struct SuppressBin {
    const cufftComplex* spectrum;
    const float* noise;
    Layout l;
    Config config;
    __host__ __device__ cufftComplex operator()(int index) const {
        const auto value = spectrum[index];
        const float power = fmaxf(value.x * value.x + value.y * value.y,
                                  1.0e-20f);
        const float ratio = 1 - config.strength * noise[index % l.bins] /
                                power;
        const float gain = sqrtf(fminf(1.0f,
            fmaxf(config.floor * config.floor, ratio)));
        const bool real_bin = index % l.bins == 0 ||
                              index % l.bins == l.bins - 1;
        return {value.x * gain, real_bin ? 0.0f : value.y * gain};
    }
};

// Each output sample gathers two frames: no atomics or overlapping writes.
struct OverlapAdd {
    const float* frames;
    Layout l;
    __host__ __device__ float operator()(int sample) const {
        const int position = sample + l.hop;
        const int latest = position / l.hop;
        float sum = 0.0f;
        float weight = 0.0f;
        for (int f = latest - 1; f <= latest; ++f) {
            const int i = position - f * l.hop;
            const float w = window(i, l.fft);
            sum += frames[f * l.fft + i] * w;
            weight += w * w;
        }
        // cuFFT does not normalize inverse transforms.
        return sum / (l.fft * weight);
    }
};

struct AbsoluteError {
    const float* first;
    const float* second;
    __host__ __device__ float operator()(int i) const {
        const float difference = fabsf(first[i] - second[i]);
        return isfinite(difference) ? difference : INFINITY;
    }
};

struct State {
    Stream stream;
    Layout l;
    Plan forward;
    Plan inverse;
    thrust::device_vector<float> input;
    thrust::device_vector<float> frames;
    thrust::device_vector<cufftComplex> spectrum;
    thrust::device_vector<cufftComplex> filtered;
    thrust::device_vector<float> noise;
    thrust::device_vector<float> output;

    explicit State(Layout layout)
        : l(layout), forward(l, false, stream.value),
          inverse(l, true, stream.value), input(l.samples),
          frames(l.frames * l.fft), spectrum(l.frames * l.bins),
          filtered(l.frames * l.bins), noise(l.bins), output(l.samples) {}
};

void forward(State& s) {
    const auto count = thrust::make_counting_iterator(0);
    thrust::transform(thrust::cuda::par.on(s.stream.value), count,
        count + s.l.frames * s.l.fft, s.frames.begin(),
        PackFrame{raw(s.input), s.l});
    fft_check(cufftExecR2C(s.forward.value, raw(s.frames), raw(s.spectrum)),
              "execute forward FFT");
}

void reconstruct(State& s) {
    fft_check(cufftExecC2R(s.inverse.value, raw(s.filtered), raw(s.frames)),
              "execute inverse FFT");
    const auto count = thrust::make_counting_iterator(0);
    thrust::transform(thrust::cuda::par.on(s.stream.value), count,
        count + s.l.samples, s.output.begin(),
        OverlapAdd{raw(s.frames), s.l});
}

void estimate_noise(State& s) {
    const auto count = thrust::make_counting_iterator(0);
    const auto keys = thrust::make_transform_iterator(
        count, NoiseKey{s.l.noise_frames});
    const auto power = thrust::make_transform_iterator(
        count, NoisePower{raw(s.spectrum), s.l});
    const int size = s.l.bins * s.l.noise_frames;
    // Fused virtual transpose + power calculation + segmented reduction.
    thrust::reduce_by_key(thrust::cuda::par.on(s.stream.value), keys,
        keys + size, power, thrust::make_discard_iterator(), s.noise.begin());
}

void suppress(State& s, const Config& config) {
    const auto count = thrust::make_counting_iterator(0);
    thrust::transform(thrust::cuda::par.on(s.stream.value), count,
        count + s.l.frames * s.l.bins, s.filtered.begin(),
        SuppressBin{raw(s.spectrum), raw(s.noise), s.l, config});
}

double check_roundtrip(State& s) {
    thrust::copy(thrust::cuda::par.on(s.stream.value), s.spectrum.begin(),
                 s.spectrum.end(), s.filtered.begin());
    reconstruct(s);
    const auto count = thrust::make_counting_iterator(0);
    return thrust::transform_reduce(thrust::cuda::par.on(s.stream.value),
        count, count + s.l.samples, AbsoluteError{raw(s.input), raw(s.output)},
        0.0f, thrust::maximum<float>());
}

double check_dft(State& s, const Audio& input) {
    std::vector<cufftComplex> frame(s.l.bins);
    cuda_check(cudaMemcpyAsync(frame.data(), raw(s.spectrum) + s.l.bins,
        frame.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost,
        s.stream.value), "read validation spectrum");
    cuda_check(cudaStreamSynchronize(s.stream.value), "wait for spectrum");
    double error = 0.0;
    for (const int k : {0, 1, 2, 17, 31, s.l.bins - 1}) {
        const auto expected = direct_bin(input, s.l.fft, k);
        const std::complex<double> actual(frame[k].x, frame[k].y);
        error = std::max(error, std::abs(actual - expected) /
                                std::max(1.0, std::abs(expected)));
    }
    return error;
}

}  // namespace

void print_environment() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "detect CUDA devices");
    if (count == 0) {
        throw std::runtime_error("A CUDA-capable NVIDIA GPU is required.");
    }
    cuda_check(cudaSetDevice(0), "select GPU");
    cudaDeviceProp device{};
    cuda_check(cudaGetDeviceProperties(&device, 0), "get GPU properties");
    int runtime = 0;
    int driver = 0;
    int library = 0;
    cuda_check(cudaRuntimeGetVersion(&runtime), "runtime version");
    cuda_check(cudaDriverGetVersion(&driver), "driver version");
    fft_check(cufftGetVersion(&library), "cuFFT version");
    std::cout << "GPU: " << device.name << " (SM " << device.major << '.'
              << device.minor << ")\nCUDA runtime: " << runtime
              << "; driver: " << driver << "; cuFFT: " << library << '\n';
    std::cout << "Libraries: cuFFT (Module 8), Thrust (Module 9)\n";
}

Processed gpu_process(const Audio& input, const Config& config) {
    State s(make_layout(input, config));
    cuda_check(cudaMemcpyAsync(raw(s.input), input.samples.data(),
        input.samples.size() * sizeof(float), cudaMemcpyHostToDevice,
        s.stream.value), "upload audio");
    forward(s);
    Processed result;
    result.dft_error = check_dft(s, input);
    result.roundtrip_error = check_roundtrip(s);
    Timer timer;
    timer.start(s.stream.value);
    forward(s);
    estimate_noise(s);
    suppress(s, config);
    reconstruct(s);
    result.milliseconds = timer.stop(s.stream.value);
    result.audio.rate = input.rate;
    result.audio.samples.resize(input.samples.size());
    cuda_check(cudaMemcpy(result.audio.samples.data(), raw(s.output),
        input.samples.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download audio");
    return result;
}

}  // namespace audio
