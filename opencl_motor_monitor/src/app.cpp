#include "reports.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace motorcl {
namespace {

Timing benchmark(Pipeline& pipeline, const Options& o,
                   const std::string& mode) {
    pipeline.execute(mode);  // Untimed warm-up for this implementation.
    Timing total;
    for (std::size_t i = 0; i < o.iterations; ++i) {
        const auto t = pipeline.execute(mode);
        total.features += t.features;
        total.classify += t.classify;
        total.reduce += t.reduce;
        total.span += t.span;
    }
    const auto divisor = static_cast<double>(o.iterations);
    return {total.features / divisor, total.classify / divisor,
            total.reduce / divisor, total.span / divisor};
}

void print_result(const std::string& mode, const Timing& t,
                    const Statistics& stats, const Validation& v) {
    std::cout << "\nMode: " << mode << '\n'
              << "  Mean device ms: features=" << t.features
              << ", classify=" << t.classify << ", reduce=" << t.reduce
              << ", span=" << t.span << '\n'
              << "  Flagged samples [temperature,current,vibration,speed]: ";
    for (const auto value : stats.counts) std::cout << value << ' ';
    std::cout << "\n  Peak normalized severity: " << stats.peak
              << "\n  Max feature errors [C,A,mm/s,RPM]: ";
    for (const auto value : v.feature_error) std::cout << value << ' ';
    std::cout << "\n  Max score error: " << v.score_error
              << "; threshold ties: " << v.threshold_ties
              << "\n  CPU reference / flags / reduction: "
              << (v.passed ? "PASS" : "FAIL") << '\n';
}

Timing run_mode(Pipeline& pipeline, const Telemetry& input,
                 const Output& reference, const Options& o,
                 const std::string& mode, std::ostream& summary) {
    const auto timing = benchmark(pipeline, o, mode);
    const auto output = pipeline.read_output();
    const auto statistics = pipeline.read_statistics();
    const auto validation = validate(output, reference, statistics, o);
    print_result(mode, timing, statistics, validation);
    write_summary(summary, o, mode, timing, statistics, validation);
    if (!validation.passed) {
        throw std::runtime_error("Numerical validation failed in " + mode);
    }
    write_reports(input, output, o, mode);
    return timing;
}

void describe(const Runtime& runtime, const Options& o) {
    const auto id = runtime.device.id;
    std::cout << "Platform: " << runtime.device.platform_name
              << "\nDevice: " << runtime.device.name
              << "\nDevice type: "
              << ((runtime.device.type & CL_DEVICE_TYPE_GPU)
                  ? "GPU" : "CPU/other")
              << "\nOpenCL: " << device_string(id, CL_DEVICE_VERSION)
              << "\nDriver: " << device_string(id, CL_DRIVER_VERSION)
              << "\nOpenCL C: " << device_string(id, CL_DEVICE_OPENCL_C_VERSION)
              << "\nSamples: " << o.samples << "; window: " << o.window
              << "; local size: " << o.local_size << "; timed passes: "
              << o.iterations << "; seed: " << o.seed
              << "\nQueue: " << (o.out_of_order ? "out-of-order" : "in-order")
              << "\nCentered window uses clamped endpoint samples.\n";
}

void run(const Options& o, const Device& device) {
    Runtime runtime(device, o);
    describe(runtime, o);
    Pipeline pipeline(runtime, o);
    const auto input = generate_telemetry(o);
    const auto begin = std::chrono::steady_clock::now();
    const auto reference = cpu_reference(input, o);
    const auto end = std::chrono::steady_clock::now();
    const auto cpu_ms = std::chrono::duration<double, std::milli>(end - begin);
    std::cout << "CPU reference: " << cpu_ms.count() << " ms\n";
    pipeline.upload(input);
    std::cout << "Input transfer: " << pipeline.upload_ms() << " ms\n";
    auto summary = summary_file(o);
    Timing naive, tiled;
    if (o.mode != "tiled") naive = run_mode(pipeline, input, reference,
                                            o, "naive", summary);
    if (o.mode != "naive") tiled = run_mode(pipeline, input, reference,
                                            o, "tiled", summary);
    if (o.mode == "both" && tiled.features > 0) {
        std::cout << "\nNaive/tiled feature-kernel time ratio: "
                  << naive.features / tiled.features << '\n';
    }
    std::cout << "Reports: " << o.output << "\nALL CHECKS PASSED\n";
}

}  // namespace
}  // namespace motorcl

int main(int argc, char** argv) {
    try {
        std::cout << std::fixed << std::setprecision(6);
        const auto options = motorcl::parse_options(argc, argv);
        if (options.help) { motorcl::print_usage(); return 0; }
        const auto devices = motorcl::enumerate_devices();
        if (options.list) { motorcl::print_devices(devices); return 0; }
        motorcl::run(options, motorcl::choose_device(devices, options));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
