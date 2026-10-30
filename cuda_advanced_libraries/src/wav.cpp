#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace audio {
namespace {

std::uint32_t read_le(std::istream& stream, int bytes) {
    std::uint32_t result = 0;
    for (int i = 0; i < bytes; ++i) {
        const int value = stream.get();
        if (value < 0) {
            throw std::runtime_error("Truncated WAV file.");
        }
        result |= static_cast<std::uint32_t>(value) << (8 * i);
    }
    return result;
}

void write_le(std::ostream& stream, std::uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        stream.put(static_cast<char>((value >> (8 * i)) & 255));
    }
}

std::string read_tag(std::istream& stream) {
    std::string tag(4, '\0');
    stream.read(&tag[0], 4);
    if (!stream) {
        throw std::runtime_error("Truncated WAV chunk tag.");
    }
    return tag;
}

int read_format(std::istream& stream, std::uint32_t size) {
    if (size < 16) {
        throw std::runtime_error("Invalid WAV fmt chunk.");
    }
    const auto encoding = read_le(stream, 2);
    const auto channels = read_le(stream, 2);
    const auto rate = read_le(stream, 4);
    const auto byte_rate = read_le(stream, 4);
    const auto alignment = read_le(stream, 2);
    const auto bits = read_le(stream, 2);
    if (encoding != 1 || channels != 1 || bits != 16 || alignment != 2 ||
        rate < 8000 || rate > 192000 || byte_rate != rate * 2) {
        throw std::runtime_error("Use mono PCM16 WAV, 8000..192000 Hz.");
    }
    stream.seekg(size - 16, std::ios::cur);
    return static_cast<int>(rate);
}

std::vector<float> read_samples(std::istream& stream, std::uint32_t bytes) {
    if (bytes == 0 || bytes % 2 != 0 || bytes / 2 > 32000000u) {
        throw std::runtime_error("Invalid or oversized WAV data chunk.");
    }
    std::vector<float> samples(bytes / 2);
    for (auto& sample : samples) {
        const auto word = read_le(stream, 2);
        const int signed_value = word < 32768u ? static_cast<int>(word) :
                                 static_cast<int>(word) - 65536;
        sample = static_cast<float>(signed_value) / 32768.0f;
    }
    return samples;
}

void write_header(std::ostream& stream, const Audio& input) {
    const auto bytes = static_cast<std::uint32_t>(input.samples.size() * 2);
    stream.write("RIFF", 4);
    write_le(stream, bytes + 36, 4);
    stream.write("WAVEfmt ", 8);
    write_le(stream, 16, 4);
    write_le(stream, 1, 2);
    write_le(stream, 1, 2);
    write_le(stream, input.rate, 4);
    write_le(stream, input.rate * 2, 4);
    write_le(stream, 2, 2);
    write_le(stream, 16, 2);
    stream.write("data", 4);
    write_le(stream, bytes, 4);
}

}  // namespace

Audio read_wav(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream || read_tag(stream) != "RIFF") {
        throw std::runtime_error("Cannot open a RIFF WAV: " + path);
    }
    const auto riff_bytes = read_le(stream, 4);
    if (riff_bytes < 4 || read_tag(stream) != "WAVE") {
        throw std::runtime_error("Invalid WAVE header.");
    }
    Audio result;
    bool have_format = false;
    const std::uint64_t riff_end = std::uint64_t(riff_bytes) + 8;
    while (stream && std::uint64_t(stream.tellg()) + 8 <= riff_end) {
        const auto tag = read_tag(stream);
        const auto size = read_le(stream, 4);
        const std::uint64_t end = std::uint64_t(stream.tellg()) + size;
        if (end + (size & 1u) > riff_end) {
            throw std::runtime_error("WAV chunk extends outside RIFF.");
        }
        if (tag == "fmt ") {
            result.rate = read_format(stream, size);
            have_format = true;
        } else if (tag == "data" && have_format) {
            result.samples = read_samples(stream, size);
            return result;
        } else {
            stream.seekg(size, std::ios::cur);
        }
        stream.seekg(size & 1u, std::ios::cur);
    }
    throw std::runtime_error("WAV needs fmt before a nonempty data chunk.");
}

void write_wav(const std::string& path, const Audio& input) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot write WAV: " + path);
    }
    write_header(stream, input);
    for (const auto sample : input.samples) {
        if (!std::isfinite(sample)) {
            throw std::runtime_error("Cannot write non-finite samples.");
        }
        const double clipped = std::clamp(double(sample), -1.0,
                                           32767.0 / 32768);
        const int value = static_cast<int>(std::lround(clipped * 32768));
        write_le(stream, static_cast<std::uint32_t>(value) & 65535u, 2);
    }
    if (!stream) {
        throw std::runtime_error("WAV write failed: " + path);
    }
}

}  // namespace audio
