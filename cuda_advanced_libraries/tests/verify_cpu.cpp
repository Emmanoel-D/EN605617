#include "reference.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_demo(const audio::Demo& demo) {
    const auto result = audio::cpu_reference(demo.noisy, demo.config);
    const int active = static_cast<int>(audio::kNoiseSeconds * demo.clean.rate);
    const double before = audio::snr(demo.clean, demo.noisy, active);
    const double after = audio::snr(demo.clean, result, active);
    std::cout << demo.name << ": " << before << " -> " << after
              << " dB; improvement=" << after - before << " dB\n";
    require(after - before >= audio::kRequiredImprovement,
            "Noise suppression did not improve SNR by at least 3 dB.");
    auto bypass = demo.config;
    bypass.strength = 0;
    const auto reconstructed = audio::cpu_reference(demo.noisy, bypass);
    const double error = audio::max_error(demo.noisy, reconstructed);
    std::cout << "bypass max error=" << error << '\n';
    require(error < 1.0e-6, "CPU FFT/overlap reconstruction failed.");
}

void check_boundaries() {
    auto demo = audio::make_demo("boundaries", 0, 0, 1024);
    demo.noisy.samples.assign(32013, 0.0f);
    demo.noisy.samples.front() = 0.7f;
    demo.noisy.samples.back() = -0.5f;
    demo.config.strength = 0;
    auto output = audio::cpu_reference(demo.noisy, demo.config);
    require(audio::max_error(demo.noisy, output) < 1.0e-6,
            "Non-hop-aligned boundary impulses were lost.");
    for (auto& sample : demo.noisy.samples) {
        sample = 0;
    }
    output = audio::cpu_reference(demo.noisy, demo.config);
    require(audio::max_error(demo.noisy, output) == 0, "Silence failed.");
    for (std::size_t i = 0; i < demo.noisy.samples.size(); ++i) {
        demo.noisy.samples[i] = 0.1f + (i % 2 ? -0.2f : 0.2f);
    }
    output = audio::cpu_reference(demo.noisy, demo.config);
    require(audio::max_error(demo.noisy, output) < 1.0e-6,
            "DC/Nyquist reconstruction failed.");
    std::cout << "Boundary impulses, silence, DC/Nyquist: PASS\n";
}

void check_wav(const std::filesystem::path& path) {
    auto demo = audio::make_demo("wav", 0.04f, 0.06f, 1024);
    audio::write_wav(path.string(), demo.noisy);
    const auto loaded = audio::read_wav(path.string());
    require(loaded.rate == demo.noisy.rate, "WAV rate changed.");
    require(audio::max_error(loaded, demo.noisy) <= 1.0 / 32768,
            "PCM16 roundtrip exceeded one quantization step.");
    std::ofstream broken(path, std::ios::binary | std::ios::trunc);
    broken << "RIFF";
    broken.close();
    bool rejected = false;
    try {
        audio::read_wav(path.string());
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Truncated WAV was not rejected.");
    std::filesystem::remove(path);
    std::cout << "WAV roundtrip and truncated-input rejection: PASS\n";
}

void check_invalid() {
    auto demo = audio::make_demo("invalid", 0, 0, 1000);
    bool rejected = false;
    try {
        audio::make_layout(demo.noisy, demo.config);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Invalid FFT length was not rejected.");
    demo.config.fft = 1024;
    demo.config.noise_seconds = 0.001;
    rejected = false;
    try {
        audio::make_layout(demo.noisy, demo.config);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Too-short noise prefix was not rejected.");
    std::cout << "Invalid configuration rejection: PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        for (const auto& demo : {
                audio::make_demo("white_noise", 0.04f, 0, 1024),
                audio::make_demo("mains_hum", 0.08f, 0.06f, 1024),
                audio::make_demo("heavy_noise", 0.12f, 0.06f, 2048)}) {
            check_demo(demo);
        }
        check_boundaries();
        check_wav(argc > 1 ? argv[1] : "cpu_test.wav");
        check_invalid();
        std::cout << "CPU REFERENCE CHECKS PASSED (not GPU evidence)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
