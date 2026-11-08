#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "pim_command.h"

namespace Ramulator::TierRuntime {
    // Tracks final DRAM-command issue for ONE QK batch (ONE query head).
    // Issue is a simulator scheduling event, not completion of a physical transfer.
    // Count every QK command, including trailing barriers, because the controller
    // scheduler is not guaranteed to issue requests in their trace order.
    class QKIssueGate{
        public:
            using ChannelOfAddress = std::function<int(std::uint64_t)>;

            QKIssueGate(const std::vector<PIMCommand>& commands,
                        std::vector<int> active_channel_ids,
                        ChannelOfAddress channel_of_address);
            
            // command_index is its ordinal in the generated batch, captured by the
            // request callback, cycle is Request::depart set at final-command issue.
            void mark_issued(std::size_t command_index, std::uint64_t cycle);
            

            [[nodiscard]] bool ready() const noexcept;
            [[nodiscard]] std::uint64_t ready_cycle() const;
            [[nodiscard]] std::vector<int> waiting_channels() const;
            
        private:
            std::vector<int> m_channels;
            std::vector<std::size_t> m_command_channels;
            std::vector<std::size_t> m_remaining_by_channel;
            std::vector<bool> m_issued;
            std::size_t m_remaining = 0;
            std::uint64_t m_last_issue_cycle = 0;
    };
}