#include "model.hpp"

#include <cmath>

namespace motorcl {
namespace {

std::uint32_t hash(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float jitter(std::size_t index, std::uint32_t seed, std::uint32_t channel) {
    const auto bits = hash(static_cast<std::uint32_t>(index) ^ seed ^
                           (channel * 0x9e3779b9u));
    return static_cast<float>(bits & 65535u) / 32767.5f - 1.0f;
}

std::uint32_t injected_fault(std::size_t index) {
    const auto phase = index % 8192;
    if (phase >= 1024 && phase < 2048) return 1;
    if (phase >= 2560 && phase < 2816) return 2;
    if (phase >= 4096 && phase < 4608) return 4;
    if (phase >= 6144 && phase < 6656) return 8;
    return 0;
}

Vec4 make_sample(std::size_t i, std::uint32_t seed, std::uint32_t fault) {
    const double angle = 6.283185307179586 * 40.0 *
                         static_cast<double>(i) / kSampleRate;
    const float amplitude = fault == 4 ? 5.0f : 0.8f;
    Vec4 result;
    result.s[0] = (fault == 1 ? 95.0f : 60.0f) + jitter(i, seed, 0);
    result.s[1] = (fault == 2 ? 32.0f : 12.0f) + 0.3f * jitter(i, seed, 1);
    result.s[2] = amplitude * static_cast<float>(std::sin(angle)) +
                  0.05f * jitter(i, seed, 2);
    result.s[3] = (fault == 8 ? 500.0f : kNominalRpm) +
                  8.0f * jitter(i, seed, 3);
    return result;
}

}  // namespace

Telemetry generate_telemetry(const Options& o) {
    Telemetry result;
    result.values.resize(o.samples);
    result.injected.resize(o.samples);
    for (std::size_t i = 0; i < o.samples; ++i) {
        result.injected[i] = injected_fault(i);
        result.values[i] = make_sample(i, o.seed, result.injected[i]);
    }
    return result;
}

Vec4 limits(const Options& o) {
    return {{o.temperature_limit, o.current_limit,
             o.vibration_limit, o.speed_limit}};
}

Statistics host_statistics(const Output& output) {
    Statistics result;
    for (std::size_t i = 0; i < output.flags.size(); ++i) {
        for (std::size_t c = 0; c < 4; ++c) {
            result.counts[c] += (output.flags[i] >> c) & 1u;
            result.peak = std::fmax(result.peak, output.scores[i].s[c]);
        }
    }
    return result;
}

}  // namespace motorcl
