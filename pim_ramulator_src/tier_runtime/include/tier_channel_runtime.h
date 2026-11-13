#pragma once

#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#include "kv_capacity_model.h"
#include "ordered_block_mapper.h"
#include "request_context.h"

namespace Ramulator::TierRuntime {
    struct DecodeTokenPlan {
        std::uint64_t token_index = 0;
        std::uint64_t valid_token_count = 0;
        int store_channel_id = -1;
        std::vector<int> allocated_channel_ids;
        std::uint64_t initial_channel_count = 0;
        bool grew = false;
    }; // 이 struct를 바탕으로 Trace Generator를 만든다.

    class TierChannelRuntime {
        public:
            TierChannelRuntime(const KVCapacityModel& capacity,
                               const OrderedBlockMapper& mapper,
                              std::uint64_t physical_channel_count);
            
            // Minimum initial channels for prompt + predicted output (at least one.)
            [[nodiscard]] std::uint64_t predicted_initial_channel_count(
                const RequestSpec& spec) const;

            // Snapshot of currently unowned physical channels in ascending ID order.
            [[nodiscard]] std::vector<int> free_channel_ids() const;
            
            // The admission policy chooses the ordered initial channel IDs
            void admit(RequestContext& request, std::vector<int> initial_ids, Clk_t cycle);
            [[nodiscard]] bool needs_growth(const RequestContext& request) const;

            // If a+1 channel is needed but none is offered, return nullopt without mutation.
            // On success, reserve the one offered channel and mark the token in progress.
            [[nodiscard]] std::optional<DecodeTokenPlan> begin_token(
                RequestContext& request, std::optional<int>growth_channel = std::nullopt);
            
            // The caller invokes this only after all commands of this token complete.
            void complete_token(RequestContext& request, Clk_t completion_cycle);
            void release_completed(RequestContext& request);
            // Abort a running request at a token boundary and release its channels.
            void evict(RequestContext& request);

        private:
            [[nodiscard]] std::uint64_t current_length(const RequestContext& request) const;
            void check_running(const RequestContext& request) const;

            const KVCapacityModel& m_capacity;
            const OrderedBlockMapper& m_mapper;
            std::vector<std::int64_t> m_owner_by_channel; // -1 means physically free.
            std::unordered_set<std::int64_t> m_active_request_ids;
    };

} // namespace Ramulator::TierRuntime