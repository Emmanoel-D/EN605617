#include "model.hpp"

#include <algorithm>
#include <cmath>

namespace motorcl {
namespace {

Vec4 window_features(const Telemetry& input, std::size_t i,
                      const Options& o) {
    double sums[4] = {};
    double vibration_energy = 0;
    const auto radius = static_cast<std::int64_t>(o.window / 2);
    for (auto offset = -radius; offset <= radius; ++offset) {
        const auto index = std::clamp(static_cast<std::int64_t>(i) + offset,
            std::int64_t{0}, static_cast<std::int64_t>(o.samples) - 1);
        const auto& value = input.values[static_cast<std::size_t>(index)];
        for (std::size_t c = 0; c < 4; ++c) sums[c] += value.s[c];
        vibration_energy += static_cast<double>(value.s[2]) * value.s[2];
    }
    Vec4 result;
    for (std::size_t c = 0; c < 4; ++c) {
        result.s[c] = static_cast<float>(sums[c] /
                                         static_cast<double>(o.window));
    }
    result.s[2] = static_cast<float>(std::sqrt(vibration_energy /
                                               o.window));
    return result;
}

Vec4 severity(const Vec4& features, const Vec4& thresholds) {
    Vec4 result;
    for (std::size_t c = 0; c < 3; ++c) {
        result.s[c] = features.s[c] / thresholds.s[c];
    }
    result.s[3] = std::abs(features.s[3] - kNominalRpm) / thresholds.s[3];
    return result;
}

}  // namespace

Output cpu_reference(const Telemetry& input, const Options& o) {
    Output result;
    result.features.resize(o.samples);
    result.scores.resize(o.samples);
    result.flags.resize(o.samples);
    const Vec4 thresholds = limits(o);
    for (std::size_t i = 0; i < o.samples; ++i) {
        result.features[i] = window_features(input, i, o);
        result.scores[i] = severity(result.features[i], thresholds);
        for (std::size_t c = 0; c < 4; ++c) {
            if (result.scores[i].s[c] > 1.0f) {
                result.flags[i] |= (1u << c);
            }
        }
    }
    return result;
}

}  // namespace motorcl
