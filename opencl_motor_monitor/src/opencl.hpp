#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "options.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace motorcl {

void cl_check(cl_int code, const std::string& operation);

// Move-only ownership releases every OpenCL handle, including on exceptions.
template <typename T, cl_int (CL_API_CALL *Release)(T)>
class Handle {
public:
    Handle() = default;
    explicit Handle(T value) : value_(value) {}
    ~Handle() { if (value_) Release(value_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            if (value_) Release(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    T get() const { return value_; }
private:
    T value_ = nullptr;
};

using Context = Handle<cl_context, clReleaseContext>;
using Queue = Handle<cl_command_queue, clReleaseCommandQueue>;
using Program = Handle<cl_program, clReleaseProgram>;
using Kernel = Handle<cl_kernel, clReleaseKernel>;
using Buffer = Handle<cl_mem, clReleaseMemObject>;
using Event = Handle<cl_event, clReleaseEvent>;

struct Device {
    cl_platform_id platform = nullptr;
    cl_device_id id = nullptr;
    cl_device_type type = 0;
    std::string name;
    std::string platform_name;
};

std::vector<Device> enumerate_devices();
void print_devices(const std::vector<Device>& devices);
Device choose_device(const std::vector<Device>& devices, const Options& o);
std::string device_string(cl_device_id id, cl_device_info field);

template <typename T>
T device_value(cl_device_id id, cl_device_info field) {
    T value{};
    cl_check(clGetDeviceInfo(id, field, sizeof(T), &value, nullptr),
             "clGetDeviceInfo");
    return value;
}

struct Runtime {
    Device device;
    Context context;
    Queue queue;
    Program program;
    Runtime(const Device& selected, const Options& options);
};

Kernel create_kernel(const Runtime& runtime, const char* name);
Buffer create_buffer(const Runtime& runtime, std::size_t bytes,
                     cl_mem_flags flags = CL_MEM_READ_WRITE);
double event_ms(const Event& event);
double event_span_ms(const Event& first, const Event& last);
void wait_for(const Event& event);

}  // namespace motorcl
