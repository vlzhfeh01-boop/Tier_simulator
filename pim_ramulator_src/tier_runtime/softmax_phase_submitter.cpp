#include "softmax_phase_submitter.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace Ramulator::TierRuntime {
    SoftmaxPhaseSubmitter::SoftmaxPhaseSubmitter(
        PIMCommand command, std::uint64_t qk_last_issue_cycle,
        std::uint64_t modeled_latency_cycles)
        : m_command(command), m_qk_last_issue_cycle(qk_last_issue_cycle),
          m_latency_cycles(modeled_latency_cycles),
          m_events(std::make_shared<std::vector<std::uint64_t>>()) {
        if (command.opcode != PIMOpcode::Softmax) {
            throw std::invalid_argument("softmax phase requires one SFM command");
        }
    }

    void SoftmaxPhaseSubmitter::tick(std::uint64_t current_cycle, const Send& send) {
        if(!send) {
            throw std::invalid_argument("softmax send function is missing");
        }
        std::vector<std::uint64_t> issued;
        issued.swap(*m_events);
        for (const auto cycle : issued) {
            if(!m_accepted || m_sv_eligible_cycle || cycle > current_cycle || 
                cycle < m_qk_last_issue_cycle) {
                throw std::logic_error("invalid or repeated SFM issue callback");
            }
            if(cycle > std::numeric_limits<std::uint64_t>::max() - m_latency_cycles) {
                throw std::overflow_error("softmax completion cycle overflow");
            }
            m_sv_eligible_cycle = cycle + m_latency_cycles;
        }
        if(!m_accepted && current_cycle >= m_qk_last_issue_cycle) {
            std::weak_ptr<std::vector<std::uint64_t>> inbox = m_events;
            IssueCallback callback = [inbox](std::uint64_t cycle) {
                if(const auto events = inbox.lock()) {
                    events->push_back(cycle);
                }
            };
            m_accepted = send(m_command, std::move(callback));
        }
    }

    bool SoftmaxPhaseSubmitter::ready(std::uint64_t current_cycle) const noexcept {
        return m_sv_eligible_cycle && current_cycle >= *m_sv_eligible_cycle;
    }
    
    std::uint64_t SoftmaxPhaseSubmitter::sv_eligible_cycle() const {
        if(!m_sv_eligible_cycle) {
            throw std::logic_error("SFM has not issued yet");
        }
        return *m_sv_eligible_cycle;
    }
    bool SoftmaxPhaseSubmitter::accepted() const noexcept {
        return m_accepted;
    }

} // namespace Ramulator::TierRuntime