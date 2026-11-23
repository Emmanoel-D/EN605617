#include "options.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace motorcl {
namespace {

std::size_t integer(const std::string& text, std::size_t low,
                    std::size_t high) {
    if (text.empty() || text.find_first_not_of("0123456789") !=
                            std::string::npos) {
        throw std::runtime_error("Expected an unsigned integer: " + text);
    }
    const auto value = std::stoull(text);
    if (value < low || value > high) {
        throw std::runtime_error("Integer outside allowed range: " + text);
    }
    return static_cast<std::size_t>(value);
}

float limit_value(const std::string& text) {
    std::size_t used = 0;
    const float value = std::stof(text, &used);
    if (used != text.size() || !std::isfinite(value) || value < 0.01f ||
        value > 100000.0f) {
        throw std::runtime_error("Limits must be finite, 0.01..100000.");
    }
    return value;
}

void numeric_option(Options& o, const std::string& key,
                    const std::string& value) {
    if (key == "--samples") o.samples = integer(value, 1, 4194304);
    else if (key == "--window") o.window = integer(value, 3, 129);
    else if (key == "--local-size") o.local_size = integer(value, 32, 256);
    else if (key == "--iterations") o.iterations = integer(value, 1, 100);
    else if (key == "--export-samples") {
        o.export_samples = integer(value, 0, 4194304);
    } else if (key == "--seed") {
        o.seed = static_cast<std::uint32_t>(integer(
            value, 0, std::numeric_limits<std::uint32_t>::max()));
    } else if (key == "--device-index") {
        o.device_index = static_cast<int>(integer(value, 0, 10000));
    } else if (key == "--temperature-limit") {
        o.temperature_limit = limit_value(value);
    } else if (key == "--current-limit") {
        o.current_limit = limit_value(value);
    } else if (key == "--vibration-limit") {
        o.vibration_limit = limit_value(value);
    } else if (key == "--speed-limit") {
        o.speed_limit = limit_value(value);
    } else throw std::runtime_error("Unknown argument: " + key);
}

void valued_option(Options& o, const std::string& key,
                   const std::string& value) {
    if (key == "--device") o.device = value;
    else if (key == "--mode") o.mode = value;
    else if (key == "--output") o.output = value;
    else if (key == "--kernels") o.kernels = value;
    else if (key == "--queue") {
        if (value != "in-order" && value != "out-of-order") {
            throw std::runtime_error("Queue: in-order or out-of-order.");
        }
        o.out_of_order = value == "out-of-order";
    } else numeric_option(o, key, value);
}

void check_options(const Options& o) {
    if (o.window % 2 != 1) {
        throw std::runtime_error("Window must be odd, 3..129.");
    }
    if ((o.local_size & (o.local_size - 1)) != 0) {
        throw std::runtime_error("Local size must be 32, 64, 128, or 256.");
    }
    if (o.device != "auto" && o.device != "gpu" && o.device != "cpu") {
        throw std::runtime_error("Device must be auto, gpu, or cpu.");
    }
    if (o.mode != "both" && o.mode != "naive" && o.mode != "tiled") {
        throw std::runtime_error("Mode must be both, naive, or tiled.");
    }
    if (o.output.empty() || o.kernels.empty()) {
        throw std::runtime_error("Output and kernel paths cannot be empty.");
    }
}

}  // namespace

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") options.help = true;
        else if (key == "--list-devices") options.list = true;
        else {
            if (i + 1 == argc) {
                throw std::runtime_error("Missing value for " + key);
            }
            valued_option(options, key, argv[++i]);
        }
    }
    check_options(options);
    return options;
}

void print_usage() {
    std::cout
        << "OpenCL electric motor fault monitor\n"
        << "  --samples N                1..4194304 (262144)\n"
        << "  --window N                 odd, 3..129 (33)\n"
        << "  --local-size N             32, 64, 128, 256 (128)\n"
        << "  --iterations N             timed passes, 1..100 (5)\n"
        << "  --seed N                   unsigned 32-bit seed (617)\n"
        << "  --temperature-limit F      Celsius (80)\n"
        << "  --current-limit F          amperes (22)\n"
        << "  --vibration-limit F        RMS mm/s (2.5)\n"
        << "  --speed-limit F            RPM deviation from 1800 (300)\n"
        << "  --mode both|naive|tiled     implementation (both)\n"
        << "  --device auto|gpu|cpu       GPU preferred (auto)\n"
        << "  --device-index N           index from --list-devices\n"
        << "  --queue in-order|out-of-order  (in-order)\n"
        << "  --output DIR               reports and CSV (results)\n"
        << "  --export-samples N         trace/SVG prefix, 0 disables\n"
        << "  --kernels FILE             OpenCL C kernel source\n"
        << "  --list-devices             print devices and exit\n"
        << "  --help                     print this text\n";
}

}  // namespace motorcl
