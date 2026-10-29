#include "tier_channel_scheduler.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace Ramulator::TierRuntime {
    std::vector<int> FirstFreeChannelPolicy::select_initial(
        const RequestSpec& , std::uint64_t needed,
        const std::vector<int>& free_ids
    ) const {
        // 초기 할당 불가
        if(needed > free_ids.size()) {
            return {};
        }
        std::vector<int> selected;
        selected.reserve(static_cast<std::size_t>(needed));
        for(std::uint64_t i = 0; i< needed; ++i) {
            selected.push_back(free_ids[static_cast<std::size_t>(i)]);
        }
        return selected;
    }

    std::optional<int> FirstFreeChannelPolicy::select_growth(
        const RequestContext& , const std::vector<int>& free_ids
    ) const {
        if(free_ids.empty()) {
            return std::nullopt;
        }
        return free_ids.front();
    }

    TierChannelScheduler::TierChannelScheduler(TierChannelRuntime& runtime,
                                 const ChannelSelectionPolicy& policy):m_runtime(runtime),m_policy(policy){}

    bool TierChannelScheduler::try_admit(RequestContext& request, Clk_t cycle){
        if(request.state != RequestState::Waiting) {
            throw std::logic_error("request is not waiting for admission");
        }

        const auto needed = m_runtime.predicted_initial_channel_count(request.spec);
        const auto free_ids = m_runtime.free_channel_ids();
        if(needed > free_ids.size()) {
            return false;
        }
        auto chosen = m_policy.select_initial(request.spec, needed, free_ids);
        if(chosen.empty()) {
            return false; // A policy may postpone this request despite capacity.
        }
        m_runtime.admit(request, std::move(chosen),cycle);
        return true;
    }

    std::optional<DecodeTokenPlan> TierChannelScheduler::try_begin_token(RequestContext& request) {
        if(!m_runtime.needs_growth(request)) {
            return m_runtime.begin_token(request);
        }
        if(request.generated_tokens >= request.spec.actual_output_tokens) {
            throw std::logic_error("all requested output tokens are complete");
        }
        const auto free_ids = m_runtime.free_channel_ids();
        const auto chosen = m_policy.select_growth(request,free_ids);
        if(!chosen) {
            return std::nullopt;
        }
        if(std::find(free_ids.begin(), free_ids.end(), *chosen) == free_ids.end()) {
            throw std::invalid_argument("policy selected a channel that is not free");
        }
        return m_runtime.begin_token(request,*chosen);
    }

} // namespace Ramulator::TierRuntime