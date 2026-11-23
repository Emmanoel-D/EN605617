#include "pipeline.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

namespace motorcl {
namespace {

std::size_t alignment_bytes(const Runtime& r) {
    const auto bits = device_value<cl_uint>(r.device.id,
                                           CL_DEVICE_MEM_BASE_ADDR_ALIGN);
    // The device reports bits; sub-buffer origins are specified in bytes.
    const auto bytes = std::max<std::size_t>(1, (bits + 7u) / 8u);
    return std::lcm(bytes, sizeof(Vec4));
}

std::size_t round_up(std::size_t size, std::size_t unit) {
    return ((size + unit - 1) / unit) * unit;
}

Buffer sub_buffer(cl_mem parent, std::size_t origin, std::size_t bytes) {
    const cl_buffer_region region{origin, bytes};
    cl_int status = 0;
    Buffer result(clCreateSubBuffer(parent, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &status));
    cl_check(status, "clCreateSubBuffer");
    return result;
}

}  // namespace

Workspace::Workspace(const Runtime& r, std::size_t samples)
    : alignment(alignment_bytes(r)),
      stride(round_up(samples * sizeof(Vec4), alignment)),
      parent(create_buffer(r, 2 * stride + samples * sizeof(cl_uint))) {
    views[0] = sub_buffer(parent.get(), 0, samples * sizeof(Vec4));
    views[1] = sub_buffer(parent.get(), stride, samples * sizeof(Vec4));
    views[2] = sub_buffer(parent.get(), 2 * stride,
                          samples * sizeof(cl_uint));
    std::cout << "Sub-buffer alignment: " << alignment << " bytes\n"
              << "Origins: features=0, scores=" << stride
              << ", flags=" << 2 * stride << " bytes\n";
}

}  // namespace motorcl
