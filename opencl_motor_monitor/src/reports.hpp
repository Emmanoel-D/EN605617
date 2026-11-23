#pragma once

#include "pipeline.hpp"
#include "validation.hpp"

#include <filesystem>
#include <fstream>

namespace motorcl {

std::ofstream output_file(const std::filesystem::path& path);
std::ofstream summary_file(const Options& options);
void write_summary(std::ostream& out, const Options& options,
                    const std::string& mode, const Timing& timing,
                    const Statistics& statistics, const Validation& validation);
void write_reports(const Telemetry& input, const Output& output,
                    const Options& options, const std::string& mode);
void write_chart(const Telemetry& input, const Output& output,
                  const Options& options, const std::filesystem::path& path);

}  // namespace motorcl
