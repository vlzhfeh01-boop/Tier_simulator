#include "runtime_decode_command_builder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace Ramulator::TierRuntime {
    std::uint64_t kv_head_for_query(std::uint64_t query_head_index,
                                    std::uint64_t query_head_count,
                                    std::uint64_t kv_head_count) {
        if(query_head_count == 0 || kv_head_count == 0 || 
            query_head_count % kv_head_count != 0 ||
            query_head_index >= query_head_count) {
            throw std::invalid_argument("invalid GQA head configuration");
        }
        return query_head_index / (query_head_count / kv_head_count);
    }

    RuntimeDecodeLayerPlan build_runtime_decode_layer_plan(
        const RequestContext& request, const DecodeTokenPlan& plan,
        std::uint64_t layer_index, std::uint64_t query_head_count,
        std::uint64_t kv_head_count, const PIMCommandGenerator& generator,
        const AttAccAddressEncoder& encoder) {
        (void)kv_head_for_query(0,query_head_count,kv_head_count);
        const auto expected_token_index = 
            static_cast<std::uint64_t>(request.spec.prompt_tokens) + 
            static_cast<std::uint64_t>(request.generated_tokens);
        if (request.state != RequestState::Running || !request.token_in_progress ||
        request.spec.request_id < 0 || plan.token_index != expected_token_index ||
        plan.token_index == std::numeric_limits<std::uint64_t>::max() ||
        plan.valid_token_count != plan.token_index + 1 ||
        plan.allocated_channel_ids != request.allocated_channels ||
        plan.initial_channel_count != request.initial_channel_count ||
        plan.initial_channel_count == 0 || plan.allocated_channel_ids.empty() ||
        std::find(plan.allocated_channel_ids.begin(),
                  plan.allocated_channel_ids.end(), plan.store_channel_id) ==
            plan.allocated_channel_ids.end()) {
            throw std::invalid_argument("decode plan no longer matches the running request");
        }

        RuntimeDecodeLayerPlan result;
        result.request_id = request.spec.request_id;
        result.token_index = plan.token_index;
        result.layer_index = layer_index;
        result.valid_token_count = plan.valid_token_count;
        result.initial_channel_count = plan.initial_channel_count;
        result.query_head_count = query_head_count;
        result.kv_head_count = kv_head_count;
        result.store_channel_id = plan.store_channel_id;
        result.allocated_channel_ids = plan.allocated_channel_ids;

        // K/V cache has kv_head_count distinct slices
        for(std::uint64_t kv = 0; kv < kv_head_count; ++kv) {
            const auto stores = generator.generate_kv_append(KVAppendCommandInput{
                layer_index,kv,plan.token_index,plan.allocated_channel_ids,
                plan.initial_channel_count
            });
            if(stores.channel_id != plan.store_channel_id) {
                throw std::logic_error("KV append channel differs from runtime placement");
            }
            result.kv_append_and_fence.insert(
                result.kv_append_and_fence.end(),stores.commands.begin(),
                stores.commands.end());
        }
        AttAccAddressCoordinates fence_coordinates{};
        fence_coordinates.channel_id = plan.store_channel_id;
        result.kv_append_and_fence.push_back(PIMCommand{
            PIMOpcode::StoreFence, encoder.encode(fence_coordinates)
        });
        return result;
    }

    StagedDecodeLayer::StagedDecodeLayer(RuntimeDecodeLayerPlan plan,
                                        const PIMCommandGenerator& generator,
                                        const AttAccAddressEncoder& encoder) 
        : m_plan(std::move(plan)), m_generator(generator), m_encoder(encoder) {
        (void)kv_head_for_query(0, m_plan.query_head_count, m_plan.kv_head_count);
        if (m_plan.valid_token_count == 0 || m_plan.kv_append_and_fence.empty() ||
            m_plan.allocated_channel_ids.empty() || m_plan.initial_channel_count == 0) {
            throw std::invalid_argument("incomplete staged decode layer plan");
        }
    }

    std::optional<DecodePhaseWork> StagedDecodeLayer::issue_next() {
        if(m_inflight) {
            throw std::logic_error("previous decode phase has not completed");
        }
        if(m_phase == DecodePhase::Done) {
            return std::nullopt;
        }

        DecodePhaseWork work;
        work.phase = m_phase;
        work.query_head_index = m_query_head_index;
        work.valid_token_count = m_plan.valid_token_count;
        work.kv_head_index = kv_head_for_query(
            m_query_head_index, m_plan.query_head_count, m_plan.kv_head_count);
        
        if(m_phase == DecodePhase::KVAppendFence) {
            work.commands = std::move(m_plan.kv_append_and_fence);
            work.active_channel_ids = {m_plan.store_channel_id};
        } else if (m_phase == DecodePhase::QK) {
            m_current_qk = m_generator.generate_qk(QKCommandInput{
                m_plan.layer_index, work.kv_head_index, m_plan.valid_token_count,
                m_plan.allocated_channel_ids, m_plan.initial_channel_count
            });
            if (m_current_qk.active_channel_ids.empty()) {
                throw std::logic_error("a nonempty decode needs QK participants");
            }
            std::uint64_t token = 0;
            m_token_ranges.clear();
            for(std::size_t index = 0; index < m_current_qk.blocks.size(); ++index) {
                const auto& block = m_current_qk.blocks[index];
                if (block.request_block_index != static_cast<std::uint64_t>(index) ||
                    block.valid_token_count == 0 ||
                    block.valid_token_count > m_plan.valid_token_count - token) {
                    throw std::logic_error("QK block token coverage is invalid");
                }
                m_token_ranges.push_back(DecodeTokenRange{
                    block.first_mac_location.channel_id, token, block.valid_token_count
                });
                token += block.valid_token_count;
            }
            if (token != m_plan.valid_token_count) {
                throw std::logic_error("QK did not cover the complete score vector");
            }
            work.commands = std::move(m_current_qk.commands);
        } else if(m_phase == DecodePhase::Softmax) {
            work.commands = m_generator.generate_softmax(m_current_qk);
            if(work.commands.size() != 1 ||
               work.commands.front().opcode != PIMOpcode::Softmax) {
                throw std::logic_error("one query head requires one SFM command");
               }
        }  else if (m_phase == DecodePhase::SV) {
            auto sv = m_generator.generate_sv(SVCommandInput{
                m_plan.layer_index, work.kv_head_index, m_plan.valid_token_count,
                m_plan.allocated_channel_ids, m_plan.initial_channel_count
            });
            if (sv.active_channel_ids != m_current_qk.active_channel_ids) {
                throw std::logic_error("QK and SV use different active channels");
            }
            work.commands = std::move(sv.commands);
        } 

        if (m_phase != DecodePhase::KVAppendFence) {
            work.active_channel_ids = m_current_qk.active_channel_ids;
            work.softmax_routing_channel_id = work.active_channel_ids.front();
            if (m_phase == DecodePhase::QK ||
                m_phase == DecodePhase::Softmax) {
                work.token_ranges = m_token_ranges;
            }
        }
        m_inflight = m_phase;
        return work;
    }

    void StagedDecodeLayer::advance_head() {
        m_current_qk = {};
        m_token_ranges.clear();
        ++m_query_head_index;
        m_phase = (m_query_head_index == m_plan.query_head_count)
            ? DecodePhase::Done : DecodePhase::QK;
    }

    void StagedDecodeLayer::mark_complete(DecodePhase phase,
                                      std::uint64_t query_head_index) {
        if (!m_inflight || phase != *m_inflight ||
            query_head_index != m_query_head_index) {
            throw std::logic_error("out-of-order decode phase completion");
        }
        switch (phase) {
        case DecodePhase::KVAppendFence: m_phase = DecodePhase::QK; break;
        case DecodePhase::QK: m_phase = DecodePhase::Softmax; break;
        case DecodePhase::Softmax: m_phase = DecodePhase::SV; break;
        case DecodePhase::SV:
            if (m_current_qk.active_channel_ids.size() > 1) {
                m_phase = DecodePhase::ContextMerge;
            } else {
                advance_head();
            }
            break;
        case DecodePhase::ContextMerge: advance_head(); break;
        case DecodePhase::Done: throw std::logic_error("cannot complete finished decode");
        }
        m_inflight.reset();
    }

    bool StagedDecodeLayer::done() const noexcept {
        return m_phase == DecodePhase::Done && !m_inflight;
    }
} // namespace Ramulator::TierRuntime
