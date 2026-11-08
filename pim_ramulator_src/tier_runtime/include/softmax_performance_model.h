#pragma once

#include <cstdint>

namespace Ramulator::TierRuntime {
    struct SoftmaxPerformanceConfig{
        long double flops_per_second = 670.4 * 1000 * 1000 * 1000;
        long double buffer_bytes_per_second = 670.4 * 1000 * 1000 * 1000;
        std::uint64_t cycle_time_ps = 0; // Set from the initialized DRAM's tCK_ps
    };

    // Analytic cost for ONE query head's complete score vector. No scheduling,
    // transfer, unit contention, or completion timestamp is inferred here.
    [[nodiscard]] std::uint64_t estimate_softmax_cycles(
        std::uint64_t valid_token_count, const SoftmaxPerformanceConfig& config);
    
} // namespace Ramulator::TierRuntime