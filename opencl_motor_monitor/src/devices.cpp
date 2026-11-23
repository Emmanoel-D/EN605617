#include "opencl.hpp"

#include <iostream>

namespace motorcl {
namespace {

std::string platform_name(cl_platform_id id) {
    std::size_t size = 0;
    cl_check(clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size),
             "platform name size");
    std::string name(size, '\0');
    cl_check(clGetPlatformInfo(id, CL_PLATFORM_NAME, size, &name[0],
                               nullptr), "platform name");
    if (!name.empty() && name.back() == '\0') name.pop_back();
    return name;
}

bool matches(const Device& device, const std::string& kind) {
    return kind == "auto" ||
        (kind == "gpu" && (device.type & CL_DEVICE_TYPE_GPU)) ||
        (kind == "cpu" && (device.type & CL_DEVICE_TYPE_CPU));
}

}  // namespace

std::string device_string(cl_device_id id, cl_device_info field) {
    std::size_t size = 0;
    cl_check(clGetDeviceInfo(id, field, 0, nullptr, &size), "device text size");
    std::string result(size, '\0');
    cl_check(clGetDeviceInfo(id, field, size, &result[0], nullptr),
             "device text");
    if (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
}

std::vector<Device> enumerate_devices() {
    cl_uint count = 0;
    const cl_int status = clGetPlatformIDs(0, nullptr, &count);
    if (status == -1001 || (status == CL_SUCCESS && count == 0)) {
        throw std::runtime_error(
            "No OpenCL platform. Install a GPU driver or PoCL CPU runtime.");
    }
    cl_check(status, "clGetPlatformIDs count");
    std::vector<cl_platform_id> platforms(count);
    cl_check(clGetPlatformIDs(count, platforms.data(), nullptr),
             "clGetPlatformIDs");
    std::vector<Device> result;
    for (const auto platform : platforms) {
        cl_uint size = 0;
        const auto code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0,
                                        nullptr, &size);
        if (code == CL_DEVICE_NOT_FOUND) continue;
        cl_check(code, "clGetDeviceIDs count");
        std::vector<cl_device_id> devices(size);
        cl_check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, size,
                                devices.data(), nullptr), "clGetDeviceIDs");
        for (const auto id : devices) {
            if (!device_value<cl_bool>(id, CL_DEVICE_AVAILABLE)) continue;
            result.push_back({platform, id,
                device_value<cl_device_type>(id, CL_DEVICE_TYPE),
                device_string(id, CL_DEVICE_NAME), platform_name(platform)});
        }
    }
    return result;
}

void print_devices(const std::vector<Device>& devices) {
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i];
        std::cout << '[' << i << "] " << d.platform_name << " / " << d.name
                  << ((d.type & CL_DEVICE_TYPE_GPU) ? " [GPU]" : " [CPU/other]")
                  << '\n';
    }
    if (devices.empty()) std::cout << "No available OpenCL devices.\n";
}

Device choose_device(const std::vector<Device>& devices, const Options& o) {
    if (o.device_index >= 0) {
        const auto index = static_cast<std::size_t>(o.device_index);
        if (index >= devices.size() || !matches(devices[index], o.device)) {
            throw std::runtime_error("Device index/type selection is invalid.");
        }
        return devices[index];
    }
    if (o.device == "auto") {
        for (const auto& d : devices) {
            if (d.type & CL_DEVICE_TYPE_GPU) return d;
        }
    }
    for (const auto& d : devices) {
        if (matches(d, o.device)) return d;
    }
    throw std::runtime_error("No device matches --device " + o.device);
}

}  // namespace motorcl
