#include "tier_channel_runtime.h"

#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace Ramulator::TierRuntime {
    namespace {
        std::size_t checked_channel_count(std::uint64_t value) {
            if(value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("physical channel count is invalid");
            }
            return static_cast<std::size_t>(value);
        }
    } // namespace
    TierChannelRuntime::TierChannelRuntime(
        const KVCapacityModel& capacity, const OrderedBlockMapper& mapper,
        std::uint64_t physical_channel_count)
        : m_capacity(capacity),m_mapper(mapper),
          m_owner_by_channel(checked_channel_count(physical_channel_count),-1) {
        if (capacity.blocks_per_channel() != mapper.blocks_per_channel()) {
            throw std::invalid_argument("invalid or inconsistent channel geometry");
        }
    }

    std::uint64_t TierChannelRuntime::current_length(const RequestContext& request) const {
        return static_cast<std::uint64_t>(request.spec.prompt_tokens) + request.generated_tokens;
    }

    void TierChannelRuntime::check_running(const RequestContext& request) const {
        if(request.state != RequestState::Running || request.initial_channel_count == 0 ||
           request.initial_channel_count > request.allocated_channels.size()) {
            throw std::logic_error("request is not an admitted running request");
        }
    }

    std::uint64_t TierChannelRuntime::predicted_initial_channel_count(const RequestSpec& spec) const {
        const auto predicted_total = m_capacity.total_tokens(spec.prompt_tokens, spec.predicted_output_tokens);
        return std::max<std::uint64_t>(
            1, m_capacity.required_channels_for_tokens(predicted_total));
    }

    std::vector<int> TierChannelRuntime::free_channel_ids() const {
        std::vector<int> free;
        for(std::size_t i = 0; i< m_owner_by_channel.size(); ++i) {
            if(m_owner_by_channel[i] == -1) {
                free.push_back(static_cast<int>(i));
            }
        }
        return free;
    }

    void TierChannelRuntime::admit(RequestContext& request, std::vector<int> initial_ids, Clk_t cycle){
        if(request.state != RequestState::Waiting || request.spec.request_id < 0 ||
           request.generated_tokens != 0 || !request.allocated_channels.empty() || 
           request.inflight_commands != 0 || cycle < request.spec.arrival_cycle || 
           initial_ids.empty()) {
            throw std::invalid_argument("invalid request admission");
        }
        if(m_active_request_ids.count(request.spec.request_id) != 0) {
            throw std::invalid_argument("request ID is already admitted");
        }
        if (predicted_initial_channel_count(request.spec) > initial_ids.size()) {
        throw std::out_of_range("initial allocation cannot hold predicted total KV");
        }
        std::unordered_set<int> unique;
        for(int channel : initial_ids) {
            if(channel < 0 || static_cast<std::size_t>(channel) >= m_owner_by_channel.size() ||
               !unique.insert(channel).second){
                throw std::invalid_argument("initial channel IDs must be unique and physical");
            }
            if(m_owner_by_channel[static_cast<std::size_t>(channel)] != -1 ) {
                throw std::logic_error("initial channel belongs to another request");
            }
        }
        m_active_request_ids.insert(request.spec.request_id);
        for(int channel : initial_ids) {
            m_owner_by_channel[static_cast<std::size_t>(channel)] = request.spec.request_id;
        }
        request.initial_channel_count = initial_ids.size();
        request.allocated_channels = std::move(initial_ids);
        request.admitted_cycle = cycle;
        request.state = RequestState::Running;
    }

    bool TierChannelRuntime::needs_growth(const RequestContext& request) const {
        check_running(request);
        if(request.token_in_progress) {
            throw std::logic_error("a token is already in progress");
        }
        return m_capacity.next_token_requires_growth(current_length(request),request.allocated_channels.size());
    }

    std::optional<DecodeTokenPlan> TierChannelRuntime::begin_token(
        RequestContext& request, std::optional<int> growth_channel
    ) {
        const bool grow = needs_growth(request);
        if (request.generated_tokens >= request.spec.actual_output_tokens) {
            throw std::logic_error("all requested output tokens are complete");
        }
        const auto token_index = current_length(request);
        if(grow && ! growth_channel) {
            return std::nullopt; // Scheduler may retry once a free channel exists.
        }
        if(!grow && growth_channel) {
            throw std::invalid_argument("growth is not needed for this token");
        }

        auto proposed_ids = request.allocated_channels;
        if(grow) {
            const int channel = *growth_channel;
            if(channel <0 || static_cast<std::size_t>(channel) >= m_owner_by_channel.size()){
                throw std::out_of_range("growth channel is not physical");
            }
            if (m_owner_by_channel[static_cast<std::size_t>(channel)] != -1) {
                return std::nullopt; // Free-channel policy must choose a different ID.
            }
            proposed_ids.push_back(channel);
        }
        const auto placement = m_mapper.locate_block( // 이 함수 다시 한번 확인해보기.
            token_index / m_capacity.tokens_per_block(), proposed_ids,
            request.initial_channel_count);
        if(grow && placement.channel_id != *growth_channel) {
            throw std::logic_error("growth mapping did not select the appended channel");
        }
        DecodeTokenPlan plan{token_index, token_index + 1, placement.channel_id,
                         proposed_ids, request.initial_channel_count, grow};
        if(grow) {
            m_owner_by_channel[static_cast<std::size_t>(*growth_channel)] = request.spec.request_id;
            request.allocated_channels.push_back(*growth_channel);
        }
        request.token_in_progress = true;
        return plan;
    }

    void TierChannelRuntime::complete_token(RequestContext& request, Clk_t completion_cycle) {
        check_running(request);
        if (!request.token_in_progress || request.inflight_commands != 0 ||
            completion_cycle < request.admitted_cycle ||
            (request.last_token_cycle >= 0 && completion_cycle < request.last_token_cycle)) {
            throw std::logic_error("token has not completed at this cycle");
        }
        if (request.generated_tokens == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("generated token count overflow");
        }
        if(request.first_token_cycle < 0) {
            request.first_token_cycle = completion_cycle;
        }
        request.last_token_cycle = completion_cycle;
        ++request.generated_tokens;
        request.token_in_progress = false;
        if(request.generated_tokens == request.spec.actual_output_tokens){
            request.state = RequestState::Completed;
            request.completion_cycle = completion_cycle;
        }
    }

    void TierChannelRuntime::release_completed(RequestContext& request) {
    if (request.state != RequestState::Completed || request.token_in_progress ||
        request.inflight_commands != 0 || request.initial_channel_count == 0) {
        throw std::logic_error("cannot release unfinished request");
    }
    for (int channel : request.allocated_channels) {
        if (m_owner_by_channel.at(static_cast<std::size_t>(channel)) != request.spec.request_id) {
            throw std::logic_error("request no longer owns one of its channels");
        }
    }
    for (int channel : request.allocated_channels) {
        m_owner_by_channel[static_cast<std::size_t>(channel)] = -1;
    }
    request.allocated_channels.clear();
    request.initial_channel_count = 0;
    m_active_request_ids.erase(request.spec.request_id);
}
    
    void TierChannelRuntime::evict(RequestContext& request) {
        check_running(request);
        if (!request.is_evictable() ||
           m_active_request_ids.count(request.spec.request_id) != 1) {
            throw std::logic_error("cannot evict a request with a token in progress");
        }
        for (int channel : request.allocated_channels) {
            if (m_owner_by_channel.at(static_cast<std::size_t>(channel)) !=
                request.spec.request_id) {
                throw std::logic_error("eviction lost a channel owner");
            }
        }
        for (int channel : request.allocated_channels) {
            m_owner_by_channel[static_cast<std::size_t>(channel)] = -1;
        }
        request.allocated_channels.clear();
        request.initial_channel_count = 0;
        m_active_request_ids.erase(request.spec.request_id);
        request.state = RequestState::Evicted;
    }
    

} // namespace Ramulator::TierRuntime