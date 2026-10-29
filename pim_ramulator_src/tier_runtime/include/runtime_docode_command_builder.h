#pragma once

#include <cstdint>
#include <vector>

#include "tier_channel_runtime.h"
#include "pim_command_generator.h"

namespace Ramulator::TierRuntime {
    // Trace commands do not contain a request ID.
    // Keep it alongside the commands so, a future request-aware executor can associate completion with the token.
    struct RuntimeDeocdeCommandBatch
    {
        std::int64_t request_id = -1;
        std::uint64_t token_index = 0;
        std::vector<int> active_channel_ids;
        bool requires_global_normalization = false;
        bool requires_cross_channel_merge = false;
        std::vector<PIMCommand> commands;
    };

    // Assemble ONE layer/head's attention phase for a plan returned by
    // TierChannelScheduler::try_begin_token(). This does not complete the token.
    [[nodiscard]] RuntimeDeocdeCommandBatch build_runtime_decode_commands(
        const RequestContext& request,
        const DecodeTokenPlan& plan,
        std::uint64_t layer_index,
        std::uint64_t kv_head_index,
        const PIMCommandGenerator& generator,
        const AttAccAddressEncoder& encoder
    );
    
} // namespace Ramulator::TierRuntime