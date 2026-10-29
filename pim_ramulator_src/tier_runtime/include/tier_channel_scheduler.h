#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "tier_channel_runtime.h"

namespace Ramulator::TierRuntime{

    // Only decides which free IDs to use; the runtime owns the actual reservation.
    class ChannelSelectionPolicy {
        public:
            virtual ~ChannelSelectionPolicy() = default;

            [[nodiscard]] virtual std::vector<int> select_initial(
                const RequestSpec& spec, std::uint64_t needed,
                const std::vector<int>& free_ids) const = 0;
                
            [[nodiscard]] virtual std::optional<int> select_growth(
                const RequestContext& request, const std::vector<int>& free_ids) const = 0;
    };

    class FirstFreeChannelPolicy final : public ChannelSelectionPolicy {
        public:
            [[nodiscard]] std::vector<int> select_initial(
                const RequestSpec& spec, std::uint64_t needed,
                const std::vector<int>& free_ids) const override;
            [[nodiscard]] std::optional<int> select_growth(
                const RequestContext& request,
                const std::vector<int>& free_ids) const override;
    };

    class TierChannelScheduler {
        public:
            TierChannelScheduler(TierChannelRuntime& runtime,
                                 const ChannelSelectionPolicy& policy);

            // False : insufficient channels or policy deferred; request stays Waiting.
            [[nodiscard]] bool try_admit(RequestContext& request, Clk_t cycle);
            
            // nullopt : growth needs a free channel, so retry after request releses one.
            [[nodiscard]] std::optional<DecodeTokenPlan> try_begin_token(RequestContext& request);

        private:
            TierChannelRuntime& m_runtime;
            const ChannelSelectionPolicy& m_policy;
    };

} // namespace Ramulator::TierRuntime