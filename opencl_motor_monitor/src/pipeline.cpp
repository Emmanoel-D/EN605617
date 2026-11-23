#include "pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace motorcl {
namespace {

template <typename T>
void argument(const Kernel& kernel, cl_uint index, const T& value) {
    cl_check(clSetKernelArg(kernel.get(), index, sizeof(T), &value),
             "clSetKernelArg");
}

void local_argument(const Kernel& kernel, cl_uint index, std::size_t size) {
    cl_check(clSetKernelArg(kernel.get(), index, size, nullptr),
             "local clSetKernelArg");
}

void read_buffer(const Runtime& r, cl_mem buffer, void* destination,
                  std::size_t bytes) {
    cl_check(clEnqueueReadBuffer(r.queue.get(), buffer, CL_TRUE, 0,
        bytes, destination, 0, nullptr, nullptr), "clEnqueueReadBuffer");
}

void check_kernel(const Runtime& r, const Kernel& kernel,
                   std::size_t work_items) {
    std::size_t maximum = 0;
    cl_check(clGetKernelWorkGroupInfo(kernel.get(), r.device.id,
        CL_KERNEL_WORK_GROUP_SIZE, sizeof(maximum), &maximum, nullptr),
        "kernel work-group limit");
    cl_ulong local = 0;
    cl_check(clGetKernelWorkGroupInfo(kernel.get(), r.device.id,
        CL_KERNEL_LOCAL_MEM_SIZE, sizeof(local), &local, nullptr),
        "kernel local-memory usage");
    if (work_items > maximum || local > device_value<cl_ulong>(
            r.device.id, CL_DEVICE_LOCAL_MEM_SIZE)) {
        throw std::runtime_error(
            "Kernel resource limit exceeded; try --local-size 32.");
    }
}

}  // namespace

Pipeline::Pipeline(const Runtime& r, const Options& o)
    : runtime_(r), options_(o),
      groups_((o.samples + o.local_size - 1) / o.local_size),
      input_(create_buffer(r, o.samples * sizeof(Vec4), CL_MEM_READ_ONLY)),
      workspace_(r, o.samples),
      limits_(create_buffer(r, sizeof(Vec4), CL_MEM_READ_ONLY)),
      counts_(create_buffer(r, groups_ * sizeof(cl_uint4))),
      peaks_(create_buffer(r, groups_ * sizeof(float))),
      naive_(create_kernel(r, "features_naive")),
      tiled_(create_kernel(r, "features_tiled")),
      classify_(create_kernel(r, "classify_faults")),
      reduce_(create_kernel(r, "reduce_faults")) {
    configure();
    check_launches();
}

void Pipeline::configure() {
    const cl_int count = static_cast<cl_int>(options_.samples);
    const cl_int radius = static_cast<cl_int>(options_.window / 2);
    for (const auto* kernel : {&naive_, &tiled_}) {
        argument(*kernel, 0, input_.get());
        argument(*kernel, 1, workspace_.views[0].get());
        argument(*kernel, 2, count);
        argument(*kernel, 3, radius);
    }
    local_argument(tiled_, 4, (options_.local_size +
        options_.window - 1) * sizeof(Vec4));
    argument(classify_, 0, workspace_.views[0].get());
    argument(classify_, 1, workspace_.views[1].get());
    argument(classify_, 2, workspace_.views[2].get());
    argument(classify_, 3, limits_.get());
    argument(classify_, 4, count);
    argument(reduce_, 0, workspace_.views[2].get());
    argument(reduce_, 1, workspace_.views[1].get());
    argument(reduce_, 2, counts_.get());
    argument(reduce_, 3, peaks_.get());
    argument(reduce_, 4, count);
    local_argument(reduce_, 5, options_.local_size * sizeof(cl_uint4));
    local_argument(reduce_, 6, options_.local_size * sizeof(float));
}

void Pipeline::check_launches() const {
    const auto& r = runtime_;
    const auto dims = device_value<cl_uint>(r.device.id,
                                           CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    std::vector<std::size_t> sizes(dims);
    cl_check(clGetDeviceInfo(r.device.id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizes.size() * sizeof(std::size_t), sizes.data(), nullptr),
        "work-item limits");
    if (options_.local_size > sizes[0] || options_.local_size >
        device_value<std::size_t>(r.device.id, CL_DEVICE_MAX_WORK_GROUP_SIZE)) {
        throw std::runtime_error("Device cannot support this local size.");
    }
    if (options_.mode != "tiled") check_kernel(r, naive_, options_.local_size);
    if (options_.mode != "naive") check_kernel(r, tiled_, options_.local_size);
    check_kernel(r, classify_, options_.local_size);
    check_kernel(r, reduce_, options_.local_size);
}

void Pipeline::upload(const Telemetry& input) {
    const auto thresholds = limits(options_);
    // Blocking constant upload completes before the first kernel is queued.
    cl_check(clEnqueueWriteBuffer(runtime_.queue.get(), limits_.get(),
        CL_TRUE, 0, sizeof(thresholds), &thresholds, 0, nullptr, nullptr),
        "upload limits");
    cl_event event = nullptr;
    // The caller keeps input alive until all dependent commands finish.
    cl_check(clEnqueueWriteBuffer(runtime_.queue.get(), input_.get(),
        CL_FALSE, 0, input.values.size() * sizeof(Vec4), input.values.data(),
        0, nullptr, &event), "upload telemetry");
    upload_event_ = Event(event);
    wait_for(upload_event_);
}

Event Pipeline::launch(const Kernel& kernel, const Event& dependency) const {
    const auto local = options_.local_size;
    const auto global = groups_ * local;
    const auto before = dependency.get();
    cl_event after = nullptr;
    cl_check(clEnqueueNDRangeKernel(runtime_.queue.get(), kernel.get(), 1,
        nullptr, &global, &local, 1, &before, &after),
        "clEnqueueNDRangeKernel");
    return Event(after);
}

Timing Pipeline::execute(const std::string& mode) {
    const auto& features = mode == "naive" ? naive_ : tiled_;
    const auto first = launch(features, upload_event_);
    const auto second = launch(classify_, first);
    const auto third = launch(reduce_, second);
    // Waiting also prevents the next pass from overwriting workspace data.
    wait_for(third);
    return {event_ms(first), event_ms(second), event_ms(third),
            event_span_ms(first, third)};
}

Output Pipeline::read_output() const {
    Output result;
    result.features.resize(options_.samples);
    result.scores.resize(options_.samples);
    result.flags.resize(options_.samples);
    read_buffer(runtime_, workspace_.views[0].get(), result.features.data(),
                result.features.size() * sizeof(Vec4));
    read_buffer(runtime_, workspace_.views[1].get(), result.scores.data(),
                result.scores.size() * sizeof(Vec4));
    read_buffer(runtime_, workspace_.views[2].get(), result.flags.data(),
                result.flags.size() * sizeof(cl_uint));
    return result;
}

Statistics Pipeline::read_statistics() const {
    std::vector<cl_uint4> counts(groups_);
    std::vector<float> peaks(groups_);
    read_buffer(runtime_, counts_.get(), counts.data(),
                counts.size() * sizeof(cl_uint4));
    read_buffer(runtime_, peaks_.get(), peaks.data(),
                peaks.size() * sizeof(float));
    Statistics result;
    for (std::size_t i = 0; i < groups_; ++i) {
        if (!std::isfinite(peaks[i])) {
            throw std::runtime_error("Non-finite reduction output.");
        }
        for (std::size_t c = 0; c < 4; ++c) result.counts[c] += counts[i].s[c];
        result.peak = std::max(result.peak, peaks[i]);
    }
    return result;
}

}  // namespace motorcl
