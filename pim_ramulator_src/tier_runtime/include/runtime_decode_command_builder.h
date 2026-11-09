#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "tier_channel_runtime.h"
#include "pim_command_generator.h"

namespace Ramulator::TierRuntime {

    // A query head selects a KV slice; it does not own a separate KV copy.
    [[nodiscard]] std::uint64_t kv_head_for_query(
        std::uint64_t query_head_index,
        std::uint64_t query_head_count,
        std::uint64_t kv_head_count);
    
    struct RuntimeDecodeLayerPlan {
        std::int64_t request_id = -1;
        std::uint64_t token_index = 0;
        std::uint64_t layer_index = 0;
        std::uint64_t valid_token_count = 0;
        std::uint64_t initial_channel_count = 0;
        std::uint64_t query_head_count = 0;
        std::uint64_t kv_head_count = 0;
        int store_channel_id = -1;
        std::vector<int> allocated_channel_ids;
        std::vector<PIMCommand> kv_append_and_fence;
    };

    // Build the writes exactly once for each KV head. 
    [[nodiscard]] RuntimeDecodeLayerPlan build_runtime_decode_layer_plan(
        const RequestContext& request,
        const DecodeTokenPlan& plan,
        std::uint64_t layer_index,
        std::uint64_t query_head_count,
        std::uint64_t kv_head_count,
        const PIMCommandGenerator& generator,
        const AttAccAddressEncoder& encoder);

    enum class DecodePhase {
        KVAppendFence,
        QK,
        Softmax,
        SV,
        ContextMerge,
        Done
    };

    struct DecodeTokenRange {
        int channel_id = -1;
        std::uint64_t first_token = 0;
        std::uint64_t token_count = 0;
    };

    struct DecodePhaseWork {
        DecodePhase phase = DecodePhase::Done;
        std::uint64_t query_head_index = 0;
        std::uint64_t kv_head_index = 0;
        std::uint64_t valid_token_count = 0;
        int softmax_routing_channel_id = -1;
        std::vector<int> active_channel_ids;
        std::vector<PIMCommand> commands;
        std::vector<DecodeTokenRange> token_ranges;
    };

    // Returns ONE phase at a tome and requires a completion event before the next.
    class StagedDecodeLayer {
        public :
            StagedDecodeLayer(RuntimeDecodeLayerPlan plan,
                                const PIMCommandGenerator& generator,
                                const AttAccAddressEncoder& encoder);
            [[nodiscard]] std::optional<DecodePhaseWork> issue_next();
            void mark_complete(DecodePhase phase, std::uint64_t query_head_index);
            [[nodiscard]] bool done() const noexcept;
        
        private:
            void advance_head();

            RuntimeDecodeLayerPlan m_plan;
            const PIMCommandGenerator& m_generator;
            const AttAccAddressEncoder& m_encoder;
            DecodePhase m_phase = DecodePhase::KVAppendFence;
            std::optional<DecodePhase> m_inflight;
            std::uint64_t m_query_head_index = 0;
            QKCommandBatch m_current_qk;
            std::vector<DecodeTokenRange> m_token_ranges;
    };
    
} // namespace Ramulator::TierRuntime