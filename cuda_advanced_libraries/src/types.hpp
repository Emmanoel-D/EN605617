#pragma once

#include <string>
#include <vector>

namespace audio {

constexpr double kPi = 3.14159265358979323846;
constexpr int kDemoRate = 16000;
constexpr double kDemoSeconds = 4.0;
constexpr double kNoiseSeconds = 0.5;
constexpr float kStrength = 2.0f;
constexpr float kGainFloor = 0.06f;
constexpr double kRoundtripTolerance = 2.0e-5;
constexpr double kReferenceTolerance = 2.0e-4;
constexpr double kDftTolerance = 2.0e-4;
constexpr double kRequiredImprovement = 3.0;

struct Audio {
    int rate = kDemoRate;
    std::vector<float> samples;
};

struct Config {
    int fft = 1024;
    double noise_seconds = kNoiseSeconds;
    float strength = kStrength;
    float floor = kGainFloor;
};

struct Layout {
    int fft;
    int hop;
    int bins;
    int frames;
    int noise_frames;
    int samples;
};

struct Processed {
    Audio audio;
    float milliseconds = 0.0f;
    double roundtrip_error = 0.0;
    double dft_error = 0.0;
};

struct Demo {
    std::string name;
    Audio clean;
    Audio noisy;
    Config config;
};

Layout make_layout(const Audio& input, const Config& config);
Audio read_wav(const std::string& path);
void write_wav(const std::string& path, const Audio& input);
Demo make_demo(const std::string& name, float noise, float hum, int fft);
double snr(const Audio& clean, const Audio& test, int start);
double max_error(const Audio& first, const Audio& second);

}  // namespace audio
