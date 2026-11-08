#include "qk_issue_gate.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Ramulator::TierRuntime {
    QKIssueGate::QKIssueGate(const std::vector<PIMCommand>& commands,
                         std::vector<int> active_channel_ids,
                         ChannelOfAddress channel_of_address)
        :m_channels(std::move(active_channel_ids)),
         m_remaining_by_channel(m_channels.size(),0),
         m_issued(commands.size(),false),
         m_remaining(commands.size()) {
        if (commands.empty() || m_channels.empty() || !channel_of_address) {
            throw std::invalid_argument("QK issue gate needs commands and channels");
        }
        for(std::size_t i = 0; i<m_channels.size(); ++i) {
            if(m_channels[i] < 0 ||
               std::find(m_channels.begin(), m_channels.begin() +
               static_cast<std::ptrdiff_t>(i), m_channels[i]) !=
               m_channels.begin() + static_cast<std::ptrdiff_t>(i)) {
                throw std::invalid_argument("duplicate or negative active channel");
            }
        }

        std::vector<bool> has_score_move(m_channels.size(),false);
        m_command_channels.reserve(commands.size());
        for(const auto& command:commands) {
            const int channel = channel_of_address(command.address);
            const auto found = std::find(m_channels.begin(),m_channels.end(),channel);
            if(found == m_channels.end()) {
                throw std::invalid_argument("QK command targets an inactive channel");
            }
            const auto index = static_cast<std::size_t>(found - m_channels.begin());
            m_command_channels.push_back(index);
            ++m_remaining_by_channel[index];
            if(command.opcode == PIMOpcode::MoveSoftmaxBuffer) {
                has_score_move[index] = true;
            }
        }

        if(std::find(has_score_move.begin(),has_score_move.end(),false) != has_score_move.end() ) {
            throw std::invalid_argument("each active channel needs a QK score move");
        }
    }

    void QKIssueGate::mark_issued(std::size_t index, std::uint64_t cycle) {
        if (index >= m_issued.size() || m_issued[index]) {
            throw std::logic_error("unknown or duplicate QK issue callback");
        }
        m_issued[index] = true;
        --m_remaining;
        --m_remaining_by_channel[m_command_channels[index]];
        m_last_issue_cycle = std::max(m_last_issue_cycle,cycle);
    }

    bool QKIssueGate::ready() const noexcept {
        return m_remaining == 0;
    }

    std::uint64_t QKIssueGate::ready_cycle() const {
        if(!ready()) {
            throw std::logic_error("QK issue gate is still waiting");
        }
        return m_last_issue_cycle;
    }

    std::vector<int> QKIssueGate::waiting_channels() const {
        std::vector<int> result;
        for(std::size_t i = 0; i < m_channels.size(); ++i) {
            if(m_remaining_by_channel[i] != 0) {
                result.push_back(m_channels[i]);
            }
        }
        return result;
    }

} // namespace Ramulator::TierRuntime