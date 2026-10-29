#pragma once

#include <filesystem>
#include <iosfwd>
#include <vector>

#include "pim_command.h"

namespace Ramulator::TierRuntime{
    class AttAccTraceWriter{
        public:
            static void write(std::ostream& output, const std::vector<PIMCommand>& commands);
            static void write_file(
                const std::filesystem::path& path,
                const std::vector<PIMCommand>& commands
            );
    };
}   // namespace Ramulator::TierRuntime