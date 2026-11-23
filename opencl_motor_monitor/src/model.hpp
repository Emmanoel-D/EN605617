#pragma once

#include "options.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace motorcl {

// Exactly matches the 16-byte OpenCL float4 ABI, without host vector math.
struct alignas(16) Vec4 {
    float s[4] = {};
};
static_assert(sizeof(Vec4) == 16, "float4 storage must occupy 16 bytes");

constexpr float kNominalRpm = 1800.0f;
constexpr double kSampleRate = 1000.0;

struct Telemetry {
    std::vector<Vec4> values;
    std::vector<std::uint32_t> injected;
};

struct Output {
    std::vector<Vec4> features;
    std::vector<Vec4> scores;
    std::vector<std::uint32_t> flags;
};

struct Statistics {
    std::array<std::uint64_t, 4> counts{};
    float peak = 0;
};

Telemetry generate_telemetry(const Options& options);
Vec4 limits(const Options& options);
Output cpu_reference(const Telemetry& input, const Options& options);
Statistics host_statistics(const Output& output);

}  // namespace motorcl
