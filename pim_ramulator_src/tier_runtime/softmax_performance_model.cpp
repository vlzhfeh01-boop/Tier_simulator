#include "softmax_performance_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Ramulator::TierRuntime {
    std::uint64_t estimate_softmax_cycles(
        std::uint64_t valid_token_count, const SoftmaxPerformanceConfig& config) {
            if (!std::isfinite(config.flops_per_second)
            || config.flops_per_second <= 0.0L
            || !std::isfinite(config.buffer_bytes_per_second)
            || config.buffer_bytes_per_second <= 0.0L
            || config.cycle_time_ps == 0) {
            throw std::invalid_argument("softmax rates and cycle_time_ps must be positive");
        }
        if (valid_token_count == 0) {
            throw std::invalid_argument("softmax requires at least one valid token");
        }

        // AttAcc's aggregate cost formula, not a new transfer or queue model.
        const auto tokens = static_cast<long double>(valid_token_count);
        const auto cycle_ps = static_cast<long double>(config.cycle_time_ps);
        const auto operations_per_cycle = config.flops_per_second * cycle_ps / 1.0e12L;
        const auto bytes_per_cycle = config.buffer_bytes_per_second * cycle_ps / 1.0e12L;
        const auto compute_cycles = 5.0L * tokens / operations_per_cycle;
        const auto buffer_cycles = 4.0L * tokens / bytes_per_cycle;
        const auto cycles = std::ceil(std::max(compute_cycles, buffer_cycles));

        // An exclusive bound avoids an out-of-range floating-to-integer cast.
        if (!std::isfinite(cycles) || cycles >= std::ldexp(1.0L, 64)) {
            throw std::overflow_error("softmax execution cycles overflow");
        }
        return std::max(std::uint64_t{1}, static_cast<std::uint64_t>(cycles));
    }
    

}