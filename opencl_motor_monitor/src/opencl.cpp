#include "opencl.hpp"

#include <fstream>
#include <iostream>
#include <iterator>

namespace motorcl {
namespace {

Context make_context(const Device& device) {
    const cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(device.platform), 0};
    cl_int status = 0;
    Context result(clCreateContext(properties, 1, &device.id, nullptr,
                                    nullptr, &status));
    cl_check(status, "clCreateContext");
    return result;
}

Queue make_queue(cl_context context, const Device& device, bool unordered) {
    cl_command_queue_properties flags = CL_QUEUE_PROFILING_ENABLE;
    if (unordered) {
        const auto supported = device_value<cl_command_queue_properties>(
            device.id, CL_DEVICE_QUEUE_PROPERTIES);
        if (!(supported & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
            throw std::runtime_error("Device lacks out-of-order queues.");
        }
        flags |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }
    cl_int status = 0;
    Queue result(clCreateCommandQueue(context, device.id, flags, &status));
    cl_check(status, "clCreateCommandQueue");
    return result;
}

Program build_program(cl_context context, cl_device_id device,
                      const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open kernels: " + path);
    const std::string source((std::istreambuf_iterator<char>(file)), {});
    const char* text = source.c_str();
    const std::size_t length = source.size();
    cl_int status = 0;
    Program result(clCreateProgramWithSource(context, 1, &text, &length,
                                              &status));
    cl_check(status, "clCreateProgramWithSource");
    // Avoid relaxed math: validation checks every sample and sensor channel.
    status = clBuildProgram(result.get(), 1, &device, "-cl-std=CL1.2",
                            nullptr, nullptr);
    if (status != CL_SUCCESS) {
        std::size_t size = 0;
        clGetProgramBuildInfo(result.get(), device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &size);
        std::string log(size, '\0');
        if (size) clGetProgramBuildInfo(result.get(), device,
            CL_PROGRAM_BUILD_LOG, size, &log[0], nullptr);
        throw std::runtime_error("OpenCL build failed:\n" + log);
    }
    return result;
}

cl_ulong timestamp(const Event& event, cl_profiling_info field) {
    cl_ulong value = 0;
    cl_check(clGetEventProfilingInfo(event.get(), field, sizeof(value),
                                     &value, nullptr), "event timestamp");
    return value;
}

}  // namespace

void cl_check(cl_int code, const std::string& operation) {
    if (code != CL_SUCCESS) {
        throw std::runtime_error(operation + " failed (OpenCL " +
                                 std::to_string(code) + ").");
    }
}

Runtime::Runtime(const Device& selected, const Options& options)
    : device(selected), context(make_context(device)),
      queue(make_queue(context.get(), device, options.out_of_order)),
      program(build_program(context.get(), device.id, options.kernels)) {}

Kernel create_kernel(const Runtime& runtime, const char* name) {
    cl_int status = 0;
    Kernel result(clCreateKernel(runtime.program.get(), name, &status));
    cl_check(status, std::string("clCreateKernel ") + name);
    return result;
}

Buffer create_buffer(const Runtime& runtime, std::size_t bytes,
                     cl_mem_flags flags) {
    const auto limit = device_value<cl_ulong>(runtime.device.id,
                                             CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    if (bytes > limit) throw std::runtime_error("Device allocation too large.");
    cl_int status = 0;
    Buffer result(clCreateBuffer(runtime.context.get(), flags, bytes,
                                  nullptr, &status));
    cl_check(status, "clCreateBuffer");
    return result;
}

void wait_for(const Event& event) {
    const auto raw = event.get();
    cl_check(clWaitForEvents(1, &raw), "clWaitForEvents");
}

double event_ms(const Event& event) {
    const auto first = timestamp(event, CL_PROFILING_COMMAND_START);
    const auto last = timestamp(event, CL_PROFILING_COMMAND_END);
    return static_cast<double>(last - first) * 1e-6;
}

double event_span_ms(const Event& first, const Event& last) {
    const auto begin = timestamp(first, CL_PROFILING_COMMAND_START);
    const auto end = timestamp(last, CL_PROFILING_COMMAND_END);
    return static_cast<double>(end - begin) * 1e-6;
}

}  // namespace motorcl
