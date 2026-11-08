#include "qk_phase_submitter.h"

#include <stdexcept>
#include <utility>

namespace Ramulator::TierRuntime {
    QKPhaseSubmitter::QKPhaseSubmitter(
        std::vector<PIMCommand> commands, std::vector<int> active_channels,
        QKIssueGate::ChannelOfAddress channel_of_address)
        : m_commands(std::move(commands)),
          m_gate(m_commands, std::move(active_channels),
                 std::move(channel_of_address)),
          m_events(std::make_shared<std::vector<Event>>()) {}
    

    void QKPhaseSubmitter::tick(const Send& send) {
        if(!send) {
            throw std::invalid_argument("QK send function is missing");
        }

        // Exchange so callbacks caused by send() cannot advance this phase
        // reentrantly in the same frontend tick.

        std::vector<Event> issued;
        issued.swap(*m_events);
        for(const auto& [index, cycle] : issued) {
            if(index >= m_accepted) {
                throw std::logic_error("QK callback for unaccepted request");
            }
            m_gate.mark_issued(index,cycle);
        }
        while(m_accepted < m_commands.size()) {
            const std::size_t index = m_accepted;
            std::weak_ptr<std::vector<Event>> inbox = m_events;
            IssueCallback callback = [inbox, index](std::uint64_t cycle) {
                if(const auto events = inbox.lock()) {
                    events->emplace_back(index,cycle);
                }   
            };
            if (!send(m_commands[index], std::move(callback))) {
                break;
            }
            ++m_accepted;
        }
    }

    bool QKPhaseSubmitter::ready() const noexcept {
        return m_accepted == m_commands.size() && m_gate.ready();
    }

    std::uint64_t QKPhaseSubmitter::ready_cycle() const {
        if (!ready()) {
            throw std::logic_error("QK requests are not all issued");
        }
        return m_gate.ready_cycle();
    }

    std::size_t QKPhaseSubmitter::accepted_count() const noexcept {
        return m_accepted;
    }
    
} // namespace Ramulator::TierRuntime