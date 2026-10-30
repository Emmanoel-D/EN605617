#pragma once

#include "types.hpp"

namespace audio {

void print_environment();
Processed gpu_process(const Audio& input, const Config& config);

}  // namespace audio
