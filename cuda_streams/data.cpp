#include "assignment.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int positiveInteger(const char* text, const char* name) {
    const std::string value(text);
    if (value.empty() || !std::all_of(value.begin(), value.end(),
                                    [](unsigned char digit) { return std::isdigit(digit) != 0; })) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer.");
    }
    std::size_t consumed = 0;
    long long number = 0;
    try {
        number = std::stoll(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " is outside the supported integer range.");
    }
    if (consumed != value.size() || number < 1 || number > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " must be between 1 and 2147483647.");
    }
    return static_cast<int>(number);
}

} // namespace

Options parseOptions(int argc, char** argv) {
    Options options;
    if (argc == 2 && std::string(argv[1]) == "--help") {
        options.help = true;
        return options;
    }
    if (argc != 1 && (argc < 3 || argc > 5)) {
        throw std::invalid_argument("Usage: assignment blocks threads [elements] [iterations]");
    }
    if (argc >= 3) {
        options.blocks = positiveInteger(argv[1], "Blocks");
        options.threads = positiveInteger(argv[2], "Threads per block");
    }
    if (argc >= 4) options.elements = positiveInteger(argv[3], "Elements");
    if (argc == 5) options.iterations = positiveInteger(argv[4], "Iterations");
    if (options.threads > 1024) {
        throw std::invalid_argument("Threads per block cannot exceed 1024.");
    }
    return options;
}

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable << " blocks threads [elements] [iterations]\n"
              << "Defaults with no arguments: 32 blocks, 128 threads per block,\n"
              << "250000 elements per array, and 10 timing repetitions.\n"
              << "Each run processes two arrays using two CUDA streams.\n";
}

void fillInput(float* input, std::size_t elements, int dataset) {
    for (std::size_t index = 0; index < elements; ++index) {
        input[index] = static_cast<float>(index % 1024) / 1024.0f + dataset * 0.25f;
    }
}

void validateOutput(const float* input, const float* output, std::size_t elements) {
    for (std::size_t index = 0; index < elements; ++index) {
        const float expected = calculateValue(input[index]);
        const float allowedError = kValidationTolerance * std::max(1.0f, std::abs(expected));
        if (!std::isfinite(output[index]) || std::abs(expected - output[index]) > allowedError) {
            throw std::runtime_error("Output validation failed at element " + std::to_string(index));
        }
    }
}
