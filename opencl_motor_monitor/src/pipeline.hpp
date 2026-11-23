#pragma once

#include "model.hpp"
#include "opencl.hpp"

#include <array>

namespace motorcl {

struct Timing {
    double features = 0;
    double classify = 0;
    double reduce = 0;
    double span = 0;
};

struct Workspace {
    std::size_t alignment;
    std::size_t stride;
    Buffer parent;
    std::array<Buffer, 3> views;
    Workspace(const Runtime& runtime, std::size_t samples);
};

class Pipeline {
public:
    Pipeline(const Runtime& runtime, const Options& options);
    void upload(const Telemetry& input);
    Timing execute(const std::string& mode);
    Output read_output() const;
    Statistics read_statistics() const;
    double upload_ms() const { return event_ms(upload_event_); }
private:
    const Runtime& runtime_;
    const Options& options_;
    std::size_t groups_;
    Buffer input_;
    Workspace workspace_;
    Buffer limits_;
    Buffer counts_;
    Buffer peaks_;
    Kernel naive_;
    Kernel tiled_;
    Kernel classify_;
    Kernel reduce_;
    Event upload_event_;
    void configure();
    void check_launches() const;
    Event launch(const Kernel& kernel, const Event& dependency) const;
};

}  // namespace motorcl
