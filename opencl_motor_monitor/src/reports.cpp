#include "reports.hpp"

#include <algorithm>
#include <iomanip>

namespace motorcl {
namespace {

constexpr const char* kNames[] = {"overheating", "overcurrent",
                                 "vibration", "speed_deviation"};

void trace(const Telemetry& input, const Output& output,
            const Options& o, const std::filesystem::path& directory) {
    if (!o.export_samples) return;
    auto file = output_file(directory / "trace.csv");
    file << "sample,time_s,temperature_C,current_A,vibration_mm_s,rpm,"
         << "mean_temperature_C,mean_current_A,rms_vibration_mm_s,"
         << "mean_rpm,temperature_score,current_score,vibration_score,"
         << "speed_score,detected_mask,injected_mask\n";
    const auto count = std::min(o.samples, o.export_samples);
    for (std::size_t i = 0; i < count; ++i) {
        file << i << ',' << static_cast<double>(i) / kSampleRate;
        for (const auto value : input.values[i].s) file << ',' << value;
        for (const auto value : output.features[i].s) file << ',' << value;
        for (const auto value : output.scores[i].s) file << ',' << value;
        file << ',' << output.flags[i] << ',' << input.injected[i] << '\n';
    }
    file.close();
}

void events(const Output& output, const std::filesystem::path& directory) {
    auto file = output_file(directory / "events.csv");
    file << "fault,start_sample,end_sample_exclusive,start_s,end_s,"
         << "duration_s,peak_severity\n";
    for (std::size_t c = 0; c < 4; ++c) {
        std::size_t i = 0;
        while (i < output.flags.size()) {
            if (!(output.flags[i] & (1u << c))) { ++i; continue; }
            const auto start = i;
            float peak = 0;
            while (i < output.flags.size() && (output.flags[i] & (1u << c))) {
                peak = std::max(peak, output.scores[i].s[c]);
                ++i;
            }
            file << kNames[c] << ',' << start << ',' << i << ','
                 << static_cast<double>(start) / kSampleRate << ','
                 << static_cast<double>(i) / kSampleRate << ','
                 << static_cast<double>(i - start) / kSampleRate << ','
                 << peak << '\n';
        }
    }
    file.close();
}

void metrics(const Telemetry& input, const Output& output,
              const std::filesystem::path& directory) {
    auto file = output_file(directory / "fault_metrics.csv");
    file << "fault,true_positive,false_positive,false_negative,"
         << "precision,recall\n";
    for (std::size_t c = 0; c < 4; ++c) {
        std::uint64_t tp = 0, fp = 0, fn = 0;
        for (std::size_t i = 0; i < input.values.size(); ++i) {
            const bool detected = (output.flags[i] & (1u << c)) != 0;
            const bool injected = (input.injected[i] & (1u << c)) != 0;
            tp += detected && injected;
            fp += detected && !injected;
            fn += !detected && injected;
        }
        file << kNames[c] << ',' << tp << ',' << fp << ',' << fn << ',';
        if (tp + fp) file << static_cast<double>(tp) / (tp + fp);
        else file << "NA";
        file << ',';
        if (tp + fn) file << static_cast<double>(tp) / (tp + fn);
        else file << "NA";
        file << '\n';
    }
    file.close();
}

}  // namespace

std::ofstream output_file(const std::filesystem::path& path) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write " + path.string());
    // Report a disk-full or write failure instead of claiming success.
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file << std::setprecision(9);
    return file;
}

std::ofstream summary_file(const Options& o) {
    std::filesystem::create_directories(o.output);
    auto file = output_file(std::filesystem::path(o.output) / "summary.csv");
    file << "mode,samples,window,local_size,iterations,queue,"
         << "features_ms,classify_ms,reduce_ms,pipeline_span_ms,"
         << "temperature_flags,current_flags,vibration_flags,speed_flags,"
         << "peak_severity,temp_error,current_error,vibration_error,"
         << "rpm_error,score_error,threshold_ties,validation\n";
    return file;
}

void write_summary(std::ostream& out, const Options& o,
                    const std::string& mode, const Timing& t,
                    const Statistics& stats, const Validation& v) {
    out << mode << ',' << o.samples << ',' << o.window << ','
        << o.local_size << ',' << o.iterations << ','
        << (o.out_of_order ? "out-of-order" : "in-order") << ','
        << t.features << ',' << t.classify << ',' << t.reduce << ',' << t.span;
    for (const auto value : stats.counts) out << ',' << value;
    out << ',' << stats.peak;
    for (const auto value : v.feature_error) out << ',' << value;
    out << ',' << v.score_error << ',' << v.threshold_ties << ','
        << (v.passed ? "PASS" : "FAIL") << '\n';
    out.flush();
}

void write_reports(const Telemetry& input, const Output& output,
                    const Options& o, const std::string& mode) {
    const auto directory = std::filesystem::path(o.output) / mode;
    std::filesystem::create_directories(directory);
    events(output, directory);
    metrics(input, output, directory);
    trace(input, output, o, directory);
    if (o.export_samples) {
        write_chart(input, output, o, directory / "telemetry.svg");
    } else {
        std::filesystem::remove(directory / "trace.csv");
        std::filesystem::remove(directory / "telemetry.svg");
    }
}

}  // namespace motorcl
