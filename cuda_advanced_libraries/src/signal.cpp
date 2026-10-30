#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace audio {
namespace {

float uniform(std::uint32_t& state) {
    state = 1664525u * state + 1013904223u;
    return (static_cast<float>(state >> 8) + 1.0f) / 16777217.0f;
}

float normal(std::uint32_t& state) {
    const double radius = std::sqrt(-2.0 * std::log(uniform(state)));
    return static_cast<float>(radius * std::cos(2 * kPi * uniform(state)));
}

float desired_signal(double time) {
    if (time < kNoiseSeconds) {
        return 0.0f;
    }
    const double t = time - kNoiseSeconds;
    const double ramp = std::min(1.0, t / 0.05);
    const double envelope = ramp * (0.8 + 0.2 * std::sin(2 * kPi * t));
    const double tone = 0.18 * std::sin(2 * kPi * 440 * t);
    const double harmonic = 0.10 * std::sin(2 * kPi * 880 * t);
    const double sweep = 0.07 * std::sin(2 * kPi * (1200 * t + 80 * t * t));
    return static_cast<float>(envelope * (tone + harmonic + sweep));
}

}  // namespace

Layout make_layout(const Audio& input, const Config& config) {
    const int n = config.fft;
    if (n < 256 || n > 8192 || (n & (n - 1)) != 0) {
        throw std::runtime_error("FFT must be a power of two, 256..8192.");
    }
    if (input.rate < 8000 || input.rate > 192000) {
        throw std::runtime_error("Sample rate must be 8000..192000 Hz.");
    }
    if (input.samples.size() < static_cast<std::size_t>(n) ||
        input.samples.size() > 32000000u) {
        throw std::runtime_error("Input must contain FFT..32000000 samples.");
    }
    const double calibration = config.noise_seconds * input.rate;
    if (!std::isfinite(calibration) || calibration < n ||
        calibration >= input.samples.size()) {
        throw std::runtime_error("Noise prefix must be FFT..length-1 samples.");
    }
    if (!std::isfinite(config.strength) || config.strength < 0.0f ||
        !std::isfinite(config.floor) || config.floor < 0 || config.floor > 1) {
        throw std::runtime_error("Invalid suppression strength or gain floor.");
    }
    const int size = static_cast<int>(input.samples.size());
    const int hop = n / 2;
    const int frames = (size + hop - 1) / hop + 1;
    return {n, hop, n / 2 + 1, frames,
            (static_cast<int>(calibration) - n) / hop + 1, size};
}

Demo make_demo(const std::string& name, float noise, float hum, int fft) {
    Demo demo;
    demo.name = name;
    demo.config.fft = fft;
    const int count = static_cast<int>(kDemoSeconds * kDemoRate);
    demo.clean.samples.resize(count);
    demo.noisy.samples.resize(count);
    std::uint32_t state = 20261029u;
    for (int i = 0; i < count; ++i) {
        const double t = static_cast<double>(i) / kDemoRate;
        const float clean = desired_signal(t);
        const float interference = hum * std::sin(2 * kPi * 60 * t);
        demo.clean.samples[i] = clean;
        demo.noisy.samples[i] = clean + noise * normal(state) + interference;
    }
    return demo;
}

double snr(const Audio& clean, const Audio& test, int start) {
    if (clean.samples.size() != test.samples.size() || start < 0 ||
        static_cast<std::size_t>(start) >= clean.samples.size()) {
        throw std::runtime_error("Mismatched audio or invalid SNR interval.");
    }
    double signal = 0.0;
    double error = 0.0;
    for (std::size_t i = start; i < clean.samples.size(); ++i) {
        const double reference = clean.samples[i];
        const double delta = reference - test.samples[i];
        signal += reference * reference;
        error += delta * delta;
    }
    return 10.0 * std::log10(std::max(signal, 1.0e-30) /
                            std::max(error, 1.0e-30));
}

double max_error(const Audio& first, const Audio& second) {
    if (first.samples.size() != second.samples.size()) {
        throw std::runtime_error("Cannot compare different audio lengths.");
    }
    double result = 0.0;
    for (std::size_t i = 0; i < first.samples.size(); ++i) {
        const double delta = static_cast<double>(first.samples[i]) -
                             static_cast<double>(second.samples[i]);
        if (!std::isfinite(delta)) {
            throw std::runtime_error("Non-finite audio detected.");
        }
        result = std::max(result, std::abs(delta));
    }
    return result;
}

}  // namespace audio
