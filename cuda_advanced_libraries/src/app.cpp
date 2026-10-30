#include "gpu.hpp"
#include "reference.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace audio {
namespace {

struct Options {
    std::string input;
    std::string output = "results";
    Config config;
    bool help = false;
};

double number(const std::string& text) {
    std::size_t used = 0;
    const double result = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(result)) {
        throw std::runtime_error("Invalid numeric argument: " + text);
    }
    return result;
}

void set_option(Options& options, const std::string& key,
                const std::string& value) {
    if (key == "--input") {
        options.input = value;
    } else if (key == "--output") {
        options.output = value;
    } else if (key == "--noise-seconds") {
        options.config.noise_seconds = number(value);
    } else if (key == "--fft") {
        const double size = number(value);
        if (size < 256 || size > 8192 || std::floor(size) != size) {
            throw std::runtime_error("FFT must be an integer, 256..8192.");
        }
        options.config.fft = static_cast<int>(size);
    } else {
        throw std::runtime_error("Unknown option: " + key);
    }
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            options.help = true;
        } else {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + key);
            }
            set_option(options, key, argv[++i]);
        }
    }
    if (options.input.empty() && (options.config.fft != 1024 ||
        options.config.noise_seconds != kNoiseSeconds)) {
        throw std::runtime_error("--fft/--noise-seconds need --input.");
    }
    return options;
}

void help() {
    std::cout << "Usage: audio_denoiser [--output DIR]\n"
              << "   or: audio_denoiser --input FILE [--output DIR]\n"
              << "       [--fft 1024] [--noise-seconds 0.5]\n\n"
              << "No input: run three deterministic validation demos.\n"
              << "Input: mono PCM16 WAV with a stationary-noise-only prefix.\n"
              << "FFT: power of two, 256..8192. Noise prefix: >= FFT samples.\n"
              << "The noise prefix must be shorter than the full recording.\n";
}

bool numerical_checks(const Processed& output, double reference_error) {
    return std::isfinite(output.roundtrip_error) &&
           std::isfinite(output.dft_error) && std::isfinite(reference_error) &&
           output.roundtrip_error <= kRoundtripTolerance &&
           output.dft_error <= kDftTolerance &&
           reference_error <= kReferenceTolerance;
}

void print_checks(const Processed& output, double reference_error) {
    std::cout << std::scientific << "  bypass max error: "
              << output.roundtrip_error << " (limit "
              << kRoundtripTolerance << ")\n  direct DFT error: "
              << output.dft_error << " (limit " << kDftTolerance
              << ")\n  CPU reference max error: " << reference_error
              << " (limit " << kReferenceTolerance << ")\n";
    std::cout << std::fixed << "  GPU pipeline: " << output.milliseconds
              << " ms (excludes planning, transfers, validation, file I/O)\n";
}

void save_audio(const std::filesystem::path& directory, const Demo& demo,
                const Processed& output) {
    std::filesystem::create_directories(directory);
    write_wav((directory / "clean.wav").string(), demo.clean);
    write_wav((directory / "noisy.wav").string(), demo.noisy);
    write_wav((directory / "denoised.wav").string(), output.audio);
}

bool run_demo(const Demo& demo, const std::filesystem::path& root,
               std::ostream& csv) {
    std::cout << "\nCase: " << demo.name << "; FFT=" << demo.config.fft << '\n';
    const auto output = gpu_process(demo.noisy, demo.config);
    const auto reference = cpu_reference(demo.noisy, demo.config);
    const double error = max_error(output.audio, reference);
    const int active = static_cast<int>(kNoiseSeconds * demo.clean.rate);
    const double before = snr(demo.clean, demo.noisy, active);
    const double after = snr(demo.clean, output.audio, active);
    const double improvement = after - before;
    const bool passed = numerical_checks(output, error) &&
                        std::isfinite(improvement) &&
                        improvement >= kRequiredImprovement;
    save_audio(root / demo.name, demo, output);
    std::cout << std::fixed << std::setprecision(6)
              << "  SNR before: " << before << " dB; after: " << after
              << " dB; improvement: " << improvement << " dB\n";
    print_checks(output, error);
    csv << demo.name << ',' << demo.config.fft << ',' << before << ','
        << after << ',' << improvement << ',' << output.milliseconds << ','
        << output.roundtrip_error << ',' << output.dft_error << ','
        << error << ',' << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << "  " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

int run_demos(const std::string& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream csv(std::filesystem::path(directory) / "summary.csv");
    if (!csv) {
        throw std::runtime_error("Cannot write summary.csv.");
    }
    csv << std::setprecision(12);
    csv << "case,fft,input_snr_db,output_snr_db,improvement_db,gpu_ms,"
        << "bypass_max_error,dft_relative_error,cpu_max_error,status\n";
    bool passed = true;
    for (const auto& demo : {
            make_demo("white_noise", 0.04f, 0.0f, 1024),
            make_demo("mains_hum", 0.08f, 0.06f, 1024),
            make_demo("heavy_noise", 0.12f, 0.06f, 2048)}) {
        passed = run_demo(demo, directory, csv) && passed;
    }
    csv.flush();
    if (!csv) {
        throw std::runtime_error("Could not finish summary.csv.");
    }
    std::cout << "\n" << (passed ? "ALL CHECKS PASSED" : "CHECKS FAILED")
              << "; results: " << directory << '\n';
    return passed ? 0 : 1;
}

int run_file(const Options& options) {
    const auto input = read_wav(options.input);
    const auto output = gpu_process(input, options.config);
    const auto reference = cpu_reference(input, options.config);
    const double error = max_error(output.audio, reference);
    const std::filesystem::path root(options.output);
    std::filesystem::create_directories(root);
    write_wav((root / "denoised.wav").string(), output.audio);
    print_checks(output, error);
    const bool passed = numerical_checks(output, error);
    std::cout << (passed ? "NUMERICAL CHECKS PASSED" : "CHECKS FAILED")
              << "; output: " << (root / "denoised.wav").string() << '\n';
    return passed ? 0 : 1;
}

}  // namespace

int application(int argc, char** argv) {
    const auto options = parse(argc, argv);
    if (options.help) {
        help();
        return 0;
    }
    print_environment();
    return options.input.empty() ? run_demos(options.output) :
                                  run_file(options);
}

}  // namespace audio

int main(int argc, char** argv) {
    try {
        return audio::application(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
