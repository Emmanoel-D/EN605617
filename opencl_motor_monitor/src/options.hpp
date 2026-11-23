#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace motorcl {

struct Options {
    std::size_t samples = 262144;
    std::size_t local_size = 128;
    std::size_t window = 33;
    std::size_t iterations = 5;
    std::size_t export_samples = 8192;
    std::uint32_t seed = 617;
    float temperature_limit = 80.0f;
    float current_limit = 22.0f;
    float vibration_limit = 2.5f;
    float speed_limit = 300.0f;
    int device_index = -1;
    bool list = false;
    bool help = false;
    bool out_of_order = false;
    std::string device = "auto";
    std::string mode = "both";
    std::string output = "results";
    std::string kernels = "kernels/motor_monitor.cl";
};

Options parse_options(int argc, char** argv);
void print_usage();

}  // namespace motorcl
