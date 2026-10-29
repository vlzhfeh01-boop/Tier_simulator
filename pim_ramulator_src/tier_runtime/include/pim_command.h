#pragma once

#include <cstdint>

namespace Ramulator::TierRuntime{

    // Internal trace operations
    enum class PIMOpcode : std::uint8_t {
        WriteGemvBuffer,
        MacAllBank,
        MoveSoftmaxBuffer,
        Barrier,
        MemoryWrite,
        StoreFence,
        Softmax,
        MoveGemvBuffer
    };

    struct PIMCommand {
        PIMOpcode opcode;
        std::uint64_t address;
    };
} // namespace Ramulator::TierRuntime