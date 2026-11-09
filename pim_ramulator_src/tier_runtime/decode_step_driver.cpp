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
            throw std::logic_error("unexpected context merge commands");
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
        if(commands.empty() || commands.back().opcode != PIMOpcode::StoreFence) {
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
        for (const auto& [index,issued_cycle] : events) {
            if(index >= m_accepted || m_issued[index]) {
                throw std::logic_error("invalid or duplicate SV issue callback");
            }
            m_issued[index] = true;
            m_last_sv_issue = std::max(m_last_sv_issue.value_or(0),issued_cycle);
        }
        while (m_accepted < m_work->commands.size()) {
            const auto index = m_accepted;
            std::weak_ptr<std::vector<Event>> inbox = m_events;
            IssueCallback callback = [inbox,index](std::uint64_t cycle) {
                if(const auto events = inbox.lock()) {
                    events->emplace_back(index,cycle);
                }
            };
            if(!send(m_work->commands[index],std::move(callback))) {
                break;
            }
            ++m_accepted;
        }
        if(m_accepted == m_issued.size()&& 
            std::all_of(m_issued.begin(),m_issued.end(),
                        [](bool issued){return issued;})) {
            finish_phase(*m_last_sv_issue);
        }
    }

    void DecodeStepDriver::tick(std::uint64_t cycle, const Send& send) {
        if(!send) {
            throw std::invalid_argument("decode send function is missing");
        }
        if(done()) {
            return ;
        }
        (void)pending_phase();
        switch(m_work->phase) {
            case DecodePhase::KVAppendFence:
                submit_kv(cycle,send);
                break;
            case DecodePhase::QK:
                m_qk->tick(send);
                if(m_qk->ready()) {
                    m_last_qk_issue = m_qk->ready_cycle();
                    finish_phase(*m_last_qk_issue);
                }
                break;
            case DecodePhase::Softmax:
                m_softmax->tick(cycle,send);
                if(m_softmax->ready(cycle)) {
                    finish_phase(m_softmax->sv_eligible_cycle());
                }
                break;
            case DecodePhase::SV:
                submit_sv(send);
                break;
            case DecodePhase::ContextMerge:
                // No cross-channel reduction latency is modeled in this step.
                finish_phase(m_last_sv_issue.value());
                break;
            case DecodePhase::Done:
                throw std::logic_error("done phase cannot be submitted");
        }
    }

    const DecodePhaseWork& DecodeStepDriver::pending_phase() {
            if (done()) {
                throw std::logic_error("completed decode layer has no pending phase");
            }
            if (!m_work) {
                begin_phase();
            }
            return *m_work;
    }

    bool DecodeStepDriver::done() const noexcept {
        return m_layer.done();
    }    

    std::uint64_t DecodeStepDriver::last_sv_issue_cycle() const {
        if(!done() || !m_last_sv_issue) {
            throw std::logic_error("SV commands hav not all issued");
        }
        return *m_last_sv_issue;
    }
    
} // namespace Ramulator::TierRuntime 