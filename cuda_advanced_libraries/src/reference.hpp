#pragma once

#include "types.hpp"

#include <complex>

namespace audio {

Audio cpu_reference(const Audio& input, const Config& config);
std::complex<double> direct_bin(const Audio& input, int fft, int bin);

}  // namespace audio
