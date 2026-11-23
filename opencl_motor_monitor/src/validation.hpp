#pragma once

#include "model.hpp"

namespace motorcl {

struct Validation {
    std::array<double, 4> feature_error{};
    double score_error = 0;
    std::size_t threshold_ties = 0;
    bool passed = true;
};

Validation validate(const Output& actual, const Output& reference,
                     const Statistics& reduced, const Options& options);

}  // namespace motorcl
