#include "reports.hpp"

#include <algorithm>
#include <cmath>

namespace motorcl {
namespace {

constexpr double kLeft = 80;
constexpr double kPlotWidth = 920;
constexpr double kPanelHeight = 160;
constexpr double kPlotHeight = 110;
constexpr const char* kTitles[] = {"Temperature (C)", "Current (A)",
    "Vibration magnitude and RMS (mm/s)", "Speed (RPM)"};

double ordinate(float value, std::size_t c, double minimum, double maximum) {
    const double normalized = std::clamp((value - minimum) /
                                         (maximum - minimum), 0.0, 1.0);
    return 100 + static_cast<double>(c) * kPanelHeight +
           (1.0 - normalized) * kPlotHeight;
}

void curve(std::ostream& file, const std::vector<Vec4>& data,
            std::size_t count, std::size_t c, double minimum, double maximum,
            const char* color, bool magnitude) {
    file << "<polyline fill=\"none\" stroke=\"" << color
         << "\" stroke-width=\"1\" points=\"";
    const auto step = std::max<std::size_t>(1, (count + 1799) / 1800);
    for (std::size_t i = 0; i < count; i += step) {
        const auto value = magnitude ? std::abs(data[i].s[c]) : data[i].s[c];
        file << kLeft + kPlotWidth * static_cast<double>(i) /
                    static_cast<double>(std::max<std::size_t>(1, count - 1))
             << ',' << ordinate(value, c, minimum, maximum) << ' ';
    }
    file << "\"/>\n";
}

void threshold_line(std::ostream& file, float value, std::size_t c,
                     double minimum, double maximum) {
    if (value < minimum || value > maximum) return;
    const double y = ordinate(value, c, minimum, maximum);
    file << "<path d=\"M80 " << y << " H1000\" stroke=\"#bb3b39\" "
         << "stroke-dasharray=\"5 4\"/>\n";
}

void panel(std::ostream& file, const Telemetry& input, const Output& output,
            const Options& o, std::size_t c, std::size_t count) {
    const double minimum = 0;
    double maximum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        maximum = std::max(maximum,
            static_cast<double>(std::abs(input.values[i].s[c])));
    }
    const auto thresholds = limits(o);
    const float upper = c == 3 ? kNominalRpm + o.speed_limit : thresholds.s[c];
    maximum = std::max(maximum, static_cast<double>(upper)) * 1.08 + 0.01;
    const double top = 100 + static_cast<double>(c) * kPanelHeight;
    file << "<text x=\"80\" y=\"" << top - 12 << "\">"
         << kTitles[c] << "</text>\n"
         << "<path d=\"M80 " << top << " V" << top + kPlotHeight
         << " H1000\" fill=\"none\" stroke=\"#77828c\"/>\n"
         << "<text x=\"8\" y=\"" << top + 5 << "\">"
         << std::round(maximum * 10) / 10 << "</text>\n"
         << "<text x=\"48\" y=\"" << top + kPlotHeight << "\">0</text>\n";
    curve(file, input.values, count, c, minimum, maximum, "#c1c7cc", c == 2);
    curve(file, output.features, count, c, minimum, maximum, "#176da6", false);
    threshold_line(file, upper, c, minimum, maximum);
    if (c == 3) threshold_line(file, kNominalRpm - o.speed_limit,
                               c, minimum, maximum);
    file << "<text x=\"80\" y=\"" << top + 128 << "\">0 s</text>\n"
         << "<text x=\"930\" y=\"" << top + 128 << "\">"
         << static_cast<double>(count - 1) / kSampleRate << " s</text>\n";
}

}  // namespace

void write_chart(const Telemetry& input, const Output& output,
                  const Options& o, const std::filesystem::path& path) {
    const auto count = std::min(o.samples, o.export_samples);
    auto file = output_file(path);
    file << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
         << "width=\"1040\" height=\"760\" viewBox=\"0 0 1040 760\">\n"
         << "<rect width=\"1040\" height=\"760\" fill=\"white\"/>\n"
         << "<g font-family=\"sans-serif\" font-size=\"13\" fill=\"#24323d\">\n"
         << "<text x=\"80\" y=\"30\" font-size=\"22\">"
         << "Electric motor telemetry</text>\n"
         << "<text x=\"80\" y=\"56\">Synthetic 1000 Hz samples | "
         << "gray: raw | blue: window feature | red: fault limit</text>\n";
    for (std::size_t c = 0; c < 4; ++c) panel(file, input, output, o, c, count);
    file << "<text x=\"80\" y=\"750\">Displayed prefix: " << count
         << " samples. Detection and validation use the full dataset."
         << "</text></g></svg>\n";
    file.close();
}

}  // namespace motorcl
