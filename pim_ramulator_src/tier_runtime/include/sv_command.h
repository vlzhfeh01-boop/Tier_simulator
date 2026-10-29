#pragma once

#include <cstdint>
#include <vector>

#include "pim_command.h"

namespace Ramulator::TierRuntime{
    // One query head's SV. V is selected by layer/KV head; its probabilities
    // belong to the query head currently scheduled by the caller.
    struct SVCommandInput{
        std::uint64_t layer_index = 0;
        std::uint64_t kv_head_index = 0;
        std::uint64_t valid_token_count = 0;
        std::vector<int> allocated_channel_ids;
        std::uint64_t initial_channel_count = 0;
    };
    
    // Trace commands and the channels that contributed to the partial result.
    struct SVCommandBatch {
        std::vector<PIMCommand> commands;
        std::vector<int> active_channel_ids;
    };

} // namespace Ramulator::TierRuntime