#include "reference.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace audio {
namespace {

using Complex = std::complex<double>;
using Frame = std::vector<Complex>;

double window(int index, int size) {
    return std::sin(kPi * (index + 0.5) / size);
}

void bit_reverse(Frame& values) {
    const int size = static_cast<int>(values.size());
    for (int i = 1, j = 0; i < size; ++i) {
        int bit = size >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }
}

void fft(Frame& values, bool inverse) {
    bit_reverse(values);
    const int size = static_cast<int>(values.size());
    for (int width = 2; width <= size; width *= 2) {
        const double angle = (inverse ? 2 : -2) * kPi / width;
        const Complex step = std::polar(1.0, angle);
        for (int begin = 0; begin < size; begin += width) {
            Complex rotation(1.0, 0.0);
            for (int j = 0; j < width / 2; ++j) {
                const Complex odd = values[begin + j + width / 2] * rotation;
                const Complex even = values[begin + j];
                values[begin + j] = even + odd;
                values[begin + j + width / 2] = even - odd;
                rotation *= step;
            }
        }
    }
    if (inverse) {
        for (auto& value : values) {
            value /= size;
        }
    }
}

std::vector<Frame> analyze(const Audio& input, const Layout& l) {
    std::vector<Frame> frames(l.frames, Frame(l.fft));
    for (int f = 0; f < l.frames; ++f) {
        for (int i = 0; i < l.fft; ++i) {
            const int source = f * l.hop - l.hop + i;
            const double value = source < 0 || source >= l.samples ?
                                 0.0 : input.samples[source];
            frames[f][i] = value * window(i, l.fft);
        }
        fft(frames[f], false);
    }
    return frames;
}

std::vector<double> noise_profile(const std::vector<Frame>& frames,
                                  const Layout& l) {
    std::vector<double> noise(l.bins, 0.0);
    for (int f = 1; f <= l.noise_frames; ++f) {
        for (int k = 0; k < l.bins; ++k) {
            noise[k] += std::norm(frames[f][k]) / l.noise_frames;
        }
    }
    return noise;
}

void suppress(std::vector<Frame>& frames, const std::vector<double>& noise,
              const Layout& l, const Config& config) {
    for (auto& frame : frames) {
        for (int k = 0; k < l.bins; ++k) {
            const double power = std::max(std::norm(frame[k]), 1.0e-20);
            const double ratio = 1 - config.strength * noise[k] / power;
            const double floor = config.floor * config.floor;
            const double gain = std::sqrt(std::min(1.0,
                                                  std::max(floor, ratio)));
            frame[k] *= gain;
            if (k > 0 && k < l.fft / 2) {
                frame[l.fft - k] = std::conj(frame[k]);
            }
        }
        fft(frame, true);
    }
}

Audio synthesize(const std::vector<Frame>& frames, const Layout& l,
                 int sample_rate) {
    Audio result;
    result.rate = sample_rate;
    result.samples.resize(l.samples);
    for (int s = 0; s < l.samples; ++s) {
        const int position = s + l.hop;
        const int latest = position / l.hop;
        double sum = 0.0;
        double weight = 0.0;
        for (int f = latest - 1; f <= latest; ++f) {
            const int i = position - f * l.hop;
            const double w = window(i, l.fft);
            sum += frames[f][i].real() * w;
            weight += w * w;
        }
        result.samples[s] = static_cast<float>(sum / weight);
    }
    return result;
}

}  // namespace

Audio cpu_reference(const Audio& input, const Config& config) {
    const auto layout = make_layout(input, config);
    auto frames = analyze(input, layout);
    const auto noise = noise_profile(frames, layout);
    suppress(frames, noise, layout, config);
    return synthesize(frames, layout, input.rate);
}

std::complex<double> direct_bin(const Audio& input, int size, int bin) {
    Complex sum(0.0, 0.0);
    for (int i = 0; i < size; ++i) {
        const double angle = -2.0 * kPi * bin * i / size;
        sum += static_cast<double>(input.samples[i]) * window(i, size) *
               std::polar(1.0, angle);
    }
    return sum;
}

}  // namespace audio
