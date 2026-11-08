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
        
        
    }
    
}