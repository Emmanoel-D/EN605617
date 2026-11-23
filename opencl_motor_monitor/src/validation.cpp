#include "validation.hpp"

#include <algorithm>
#include <cmath>

namespace motorcl {
namespace {

double feature_tolerance(float expected) {
    return 2e-4 + 1e-6 * std::abs(static_cast<double>(expected));
}

void check_channel(Validation& v, const Output& actual,
                    const Output& expected, const Vec4& thresholds,
                    std::size_t i, std::size_t c) {
    const auto feature = actual.features[i].s[c];
    const auto score = actual.scores[i].s[c];
    const auto reference = expected.scores[i].s[c];
    const double error = std::abs(static_cast<double>(feature) -
                                  expected.features[i].s[c]);
    const double limit = feature_tolerance(expected.features[i].s[c]);
    const double score_limit = limit / thresholds.s[c] +
                               4e-6 * std::max(1.0f, std::abs(reference));
    const double score_error = std::abs(static_cast<double>(score) - reference);
    v.feature_error[c] = std::max(v.feature_error[c], error);
    v.score_error = std::max(v.score_error, score_error);
    if (!std::isfinite(feature) || !std::isfinite(score) || error > limit ||
        score_error > score_limit) v.passed = false;
    const bool detected = (actual.flags[i] & (1u << c)) != 0;
    if (detected != (score > 1.0f)) v.passed = false;
    if (detected != ((expected.flags[i] & (1u << c)) != 0)) {
        // A threshold crossing within propagated FP error is inconclusive.
        if (std::abs(reference - 1.0f) <= score_limit) ++v.threshold_ties;
        else v.passed = false;
    }
}

}  // namespace

Validation validate(const Output& actual, const Output& reference,
                     const Statistics& reduced, const Options& options) {
    Validation result;
    const auto thresholds = limits(options);
    for (std::size_t i = 0; i < actual.features.size(); ++i) {
        if (actual.flags[i] & ~15u) result.passed = false;
        for (std::size_t c = 0; c < 4; ++c) {
            check_channel(result, actual, reference, thresholds, i, c);
        }
    }
    const auto host = host_statistics(actual);
    if (host.counts != reduced.counts || !std::isfinite(reduced.peak) ||
        host.peak != reduced.peak) result.passed = false;
    return result;
}

}  // namespace motorcl
