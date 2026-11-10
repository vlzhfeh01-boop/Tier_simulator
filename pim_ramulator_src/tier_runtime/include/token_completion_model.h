#pragma once

#include <cstdint>
#include<initializer_list>
#include <limits>
#include <stdexcept>

namespace Ramulator::TierRuntime {
    struct TokenCompletionConfig {
        std::uint64_t sv_drain_cycles = 0;
        std::uint64_t context_merge_cycles = 0;
        std::uint64_t non_pim_cycles = 0;
    };

    class TokenCompletionModel {
    public:
        explicit TokenCompletionModel(TokenCompletionConfig config) : m_config(config){}

        [[nodiscard]] std::uint64_t earliest_cycle(std::uint64_t last_sv_issue) const
        {
            auto cycle = last_sv_issue;
            for (const auto delay : {m_config.sv_drain_cycles,
                                    m_config.context_merge_cycles,
                                    m_config.non_pim_cycles}) {
                if (delay > std::numeric_limits<std::uint64_t>::max() - cycle) {
                    throw std::overflow_error("token completion cycle overflow");
                }
                cycle += delay;
            }
            return cycle;
        }
    private:
        TokenCompletionConfig m_config;
    };
    
} // namespace Ramulator::TierRuntime