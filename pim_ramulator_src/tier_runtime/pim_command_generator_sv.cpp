#include "pim_command_generator.h"

#include <algorithm>
#include <cstddef>

namespace Ramulator::TierRuntime {
namespace {
    // AttAcc : 16 x 32Byte input slots per GEMV input buffer.
    // A slot contains 16 FP16 weights;
    constexpr std::size_t sv_input_slot_count = 16;
    std::uint64_t sv_ceil_div(std::uint64_t value, std::uint64_t divisor) {
        return value / divisor + (value % divisor != 0);
    }
} //namespace 
    // 가중치 pt를 넣을 GEMV 입력 버퍼 지정
    std::uint64_t PIMCommandGenerator::encode_probability_buffer(
        int channel_id, std::uint64_t rank, std::uint64_t bank_group,
        std::uint64_t slot, std::uint64_t value_slice_base_row
    ) const {
        // context_cpvec(): the slot is a raw low address, not slot * 32
        return m_encoder.encode(AttAccAddressCoordinates{
            channel_id,0,rank,bank_group,0,value_slice_base_row,0,slot});
    }
    // SV 부분 결과를 수집할 위치 지정.
    std::uint64_t PIMCommandGenerator::encode_context_buffer(
        int channel_id, std::uint64_t rank, std::uint64_t bank,
        std::uint64_t value_slice_base_row
    ) const {
        return m_encoder.encode(AttAccAddressCoordinates{
            channel_id, 0, rank, 0, bank, value_slice_base_row,0,0
        });
    }

    SVCommandBatch PIMCommandGenerator::generate_sv(const SVCommandInput& input) const {
        validate_allocation(input.allocated_channel_ids, input.initial_channel_count);
        const auto value_base_row = m_layout.slice_base_row(
            KVKind::Value, input.layer_index, input.kv_head_index
        );
        const auto tokens_per_block = m_layout.tokens_per_block();
        const auto block_count = sv_ceil_div(input.valid_token_count, tokens_per_block);
        if(block_count != 0) {
            (void)m_address_mapper.locate_mac(
                KVKind::Value, input.layer_index, input.kv_head_index,
                block_count - 1,0,
                input.allocated_channel_ids, input.initial_channel_count
            ); 
        }

        SVCommandBatch batch;
        const auto& geometry = m_layout.geometry();
        const auto steps = m_layout.columns_per_block_in_slice();
        if(block_count == 0) {
            return batch;
        }

        // Same allocation order as QK; only channels with valid KV participate.
        // OrderedBlockMapper gives each participating channel a contiguous local
        // block prefix, including sequentially filled growth channels.
        std::vector<std::vector<PIMMACLocation>> channel_blocks(input.allocated_channel_ids.size());
        for(std::uint64_t block = 0; block < block_count; ++block) {
            const auto location = m_address_mapper.locate_mac(
                KVKind::Value, input.layer_index, input.kv_head_index,
                block, 0, input.allocated_channel_ids, input.initial_channel_count
            );
            const auto channel_it = std::find(
                input.allocated_channel_ids.begin(), input.allocated_channel_ids.end(),
                location.channel_id
            );
            const auto channel_index = static_cast<std::size_t> (
                channel_it - input.allocated_channel_ids.begin()
            );
            channel_blocks[channel_index].push_back(location); // 여기서 이미 전체 block 에 대한 좌표를 전부 가져오게 된다.
        }
        std::size_t maximum_local_blocks = 0;
        for(std::size_t i = 0; i< channel_blocks.size(); ++i) {
            if(!channel_blocks[i].empty()) {
                batch.active_channel_ids.push_back(input.allocated_channel_ids[i]);
                maximum_local_blocks = std::max(maximum_local_blocks,channel_blocks[i].size());
            }
        }

        const auto append = [&batch](PIMOpcode opcode, std::uint64_t address) {
            batch.commands.push_back(PIMCommand{opcode, address});
        };

        const auto barriers = [&](std::size_t begin) {
            for(std::size_t i = 0; i < channel_blocks.size(); ++i) {
                if(begin < channel_blocks[i].size()) {
                    append(PIMOpcode::Barrier, encode_channel(input.allocated_channel_ids[i]));
                }
            }
        };

        const auto weights_per_slot = static_cast<std::size_t>(m_layout.elements_per_column());
        const auto resident_blocks = sv_input_slot_count * weights_per_slot; // 256

        const auto load_weights = [&](std::size_t begin, std::size_t count, bool streamed_only) {
            const auto pages = count / weights_per_slot + (count % weights_per_slot != 0);
            // Original context_cpvec() : rank -> BG -> slot -> channel.
            for(std::uint64_t rank =0; rank < geometry.ranks_per_pseudochannel; ++rank) {
                for(std::uint64_t bg = 0; bg < geometry.bank_groups_per_rank; ++bg) {
                    for(std::size_t slot = 0; slot < pages; ++slot) {
                        const auto first = begin + slot * weights_per_slot;
                        for(std::size_t i=0; i<channel_blocks.size(); ++i) {
                            if(streamed_only && channel_blocks[i].size() <= resident_blocks){
                                continue;
                            }
                            if(first >= channel_blocks[i].size()) {
                                continue;
                            }
                            const auto channel = input.allocated_channel_ids[i];
                            append(PIMOpcode::MoveGemvBuffer,
                                encode_probability_buffer(channel,rank,bg,static_cast<std::uint64_t>(slot), value_base_row));
                        }
                    }
                }
            }
        };

        const auto mac_blocks = [&](std::size_t begin, std::size_t count, std::uint64_t step) {
            for(std::size_t offset=0; offset < count; ++offset) {
                const auto local = begin + offset;
                for(std::size_t i = 0; i < channel_blocks.size(); ++i) {
                    if(local >= channel_blocks[i].size()) {
                        continue;
                    }
                    auto location = channel_blocks[i][local];
                    location.col += step; // The mapper guarantees no block row crossing.
                    append(PIMOpcode::MacAllBank,
                    m_encoder.encode_mac(location));
                }
            }
        };

        const auto collect_output = [&]() {
            for (std::uint64_t bank = 0; bank < geometry.banks_per_bank_group; ++bank) {
                for (std::uint64_t rank = 0; rank < geometry.ranks_per_pseudochannel; ++rank) {
                    for (std::size_t i = 0; i < channel_blocks.size(); ++i) {
                        if(channel_blocks[i].empty()) {
                            continue;
                        }
                        // Softmax buffer를 SV결과를 모으는 장소로도 사용한다.
                        const auto channel = input.allocated_channel_ids[i];
                        append(PIMOpcode::MoveSoftmaxBuffer,
                            encode_context_buffer(channel, rank, bank, value_base_row));
                    }
                }
            }
        };

        // Separates the caller's preceding SFM commands from these input moves.
        // Existing PIM_BARRIER is a scheduler boundary, not a new latency model.
        barriers(0);
        load_weights(0,std::min(resident_blocks, maximum_local_blocks), false);
        barriers(0);
        // Keep output dimensions fixed while accumulating contributions from
        // all token blocks. Refilling inputs does not reset lane accumulators.
        for (std::uint64_t step = 0; step < steps; ++step) {
            for (std::size_t begin = 0; begin < maximum_local_blocks;) {
                const auto count = std::min(resident_blocks, maximum_local_blocks - begin);
                if(begin != 0 || (step != 0 && maximum_local_blocks > resident_blocks)) {
                    // Window zero is already resident on the first step. On later
                    // steps, reload it only on channels whose inputs did not fit.
                    load_weights(begin, count, begin == 0);
                    barriers(begin);
                }
                mac_blocks(begin, count, step);
                barriers(begin); // Consume this window before reusing the slots.
                begin += count;
            }
            collect_output();
            barriers(0); // Collect this dimension step before reusing its accumulators.
        }
        return batch;
    }
} // namespace Ramulator::TierRuntime