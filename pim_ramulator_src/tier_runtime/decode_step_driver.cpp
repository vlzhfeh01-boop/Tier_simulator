#include "decode_step_driver.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Ramulator::TierRuntime {
    DecodeStepDriver::DecodeStepDriver(RuntimeDecodeLayerPlan plan,
                                   const PIMCommandGenerator& generator,
                                   const AttAccAddressEncoder& encoder,
                                   std::uint64_t softmax_delay_cycles,
                                   PhaseComplete on_phase_complete)
    : m_layer(std::move(plan), generator, encoder),
      m_encoder(encoder),
      m_softmax_delay_cycles(softmax_delay_cycles),
      m_on_phase_complete(std::move(on_phase_complete)),
      m_events(std::make_shared<std::vector<Event>>()) {}

    void DecodeStepDriver::begin_phase() {
        m_work = m_layer.issue_next();
        if(!m_work) {
            throw std::logic_error("decode layer has no pending phase");
        }
        m_accepted = 0;
        m_issued.clear();
        m_events->clear();
        if(m_work->phase == DecodePhase::QK) {
            m_qk = std::make_unique<QKPhaseSubmitter>(
                m_work->commands, m_work->active_channel_ids,
                [this](std::uint64_t address) {
                    return m_encoder.decode(address).channel_id;
                }
            );
        } else if(m_work->phase == DecodePhase::Softmax) {
            if(!m_last_qk_issue || m_work->commands.size() != 1) {
                throw std::logic_error("softmax must follow one completed QK phase");
            }
            m_softmax = std::make_unique<SoftmaxPhaseSubmitter>(
                m_work->commands.front(),*m_last_qk_issue,
                m_softmax_delay_cycles);
        } else if(m_work->phase == DecodePhase::SV) {
            if(m_work->commands.empty()) {
                throw std::logic_error("SV phase has no commands");
            }
            m_issued.assign(m_work->commands.size(),false);
        } else if(m_work->phase == DecodePhase::ContextMerge && !m_work->commands.empty()){
            throw std::logic_error("unexpected context merge commnads");
        }
    }

    void DecodeStepDriver::finish_phase(std::uint64_t cycle) {
        const auto phase = m_work->phase;
        const auto head = m_work->query_head_index;
        m_layer.mark_complete(phase,head);
        if(m_on_phase_complete) {
            m_on_phase_complete(phase,head,cycle);
        }
        m_work.reset();
        m_qk.reset();
        m_softmax.reset();
    }

    void DecodeStepDriver::submit_kv(std::uint64_t cycle, const Send& send) {
        const auto& commands = m_work->commands;
        if(commads.empty() || commands.back().opcode != PIMOpcode::StoreFence) {
            throw std::logic_error("KV append must end in ST_FENCE");
        }
        while(m_accepted < commands.size()) {
            const auto& cmd = commands[m_accepted];
            if(cmd.opcode != PIMOpcode::MemoryWrite &&
               cmd.opcode != PIMOpcode::StoreFence) {
                throw std::logic_error("unsupported KV append command");
            }
            if(!send(cmd, {})) {
                return ;
            }
            ++m_accepted;
        }
        // The controller accepts ST_FENCE only after preceding writes are queued.
        // Its channel guard rejects dependent QK work until they are safe.
        finish_phase(cycle);
    }

    void DecodeStepDriver::submit_sv(const Send& send) {
        std::vector<Event> events;
        events.swap(*m_events);
        
    }
} // namespace Ramulator::TierRuntime 