#include "pim_command_generator.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace Ramulator::TierRuntime{
    namespace {
        constexpr std::size_t score_blocks_per_flush = 16;

        std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
            return value / divisor + (value % divisor != 0);
        }
    }// namespace

    PIMCommandGenerator::PIMCommandGenerator(
        const RowAlignedKVLayout& layout,
        const OrderedBlockMapper& block_mapper,
        std::uint64_t total_channel_count
    ) : m_layout(layout), m_address_mapper(layout, block_mapper),
        m_encoder(layout, total_channel_count),m_total_channel_count(total_channel_count) {
            const auto& geometry = m_layout.geometry();

            if (geometry.pseudochannels_per_channel != 2 ||
                geometry.ranks_per_pseudochannel != 2 ||
                geometry.bank_groups_per_rank != 4 ||
                geometry.banks_per_bank_group != 4 ||
                geometry.columns_per_row != 32 ||
                geometry.bytes_per_column != 32 ||
                m_layout.shape().bytes_per_element() != 2) {
                throw std::invalid_argument(
                    "QK generator requires the verified bank-level FP16 HBM3 configuration"
                );
            }
    }

    std::uint64_t PIMCommandGenerator::encode_query_buffer(
        int channel_id, std::uint64_t bank, std::uint64_t slot,
        std::uint64_t key_slice_base_row
    ) const {
        // Source convention: channel_base + bank_base + key_slice_base + slot.
        // Do not multiply slot by the transaction size or by element size.
        return m_encoder.encode(AttAccAddressCoordinates{
            channel_id,0,0,0,bank,key_slice_base_row,0,slot
        });
    }

    std::uint64_t PIMCommandGenerator::encode_score_buffer(
        int channel_id, std::uint64_t rank, std::uint64_t bank_group,
        std::uint64_t key_slice_base_row
    ) const {
        // pCH is broadcast; QK outputs are selected by rank/BG, bank is zero.
        return m_encoder.encode(AttAccAddressCoordinates{
            channel_id,0,rank,bank_group,0,key_slice_base_row,0,0
        });
    }

    std::uint64_t PIMCommandGenerator::encode_channel(int channel_id) const {
        AttAccAddressCoordinates coordinates;
        coordinates.channel_id = channel_id;
        return m_encoder.encode(coordinates);
    }
    
    void PIMCommandGenerator::validate_allocation(
        const std::vector<int>& channel_ids,
        std::uint64_t initial_channel_count
    ) const {
        // Check even unused trailing channels: the allocation list is a runtime
        // ownership contract, not a list of arbitrary addresses.
        if(initial_channel_count == 0 || initial_channel_count> channel_ids.size()) {
            throw std::invalid_argument("initial_channel_count is inconsistent with allocation.");
        }
        std::unordered_set<int> seen;
        for(const int channel_id : channel_ids) {
            if(channel_id < 0 || 
                static_cast<std::uint64_t>(channel_id) >= m_total_channel_count) {
                    throw std::out_of_range("allocated channel is outside physical geometry");
            }
            if(!seen.insert(channel_id).second) { 
                throw std::invalid_argument("allocated channel IDs must be unique.");
            }
        }

    }

    KVAppendCommandBatch PIMCommandGenerator::generate_kv_append(
        const KVAppendCommandInput& input
    ) const {
        validate_allocation(input.allocated_channel_ids, input.initial_channel_count);

        // A length of N means position[0,N) exist; append at position N.
        const auto token_index = input.existing_token_count;
        const auto first_key = m_address_mapper.locate_element(
            KVKind::Key, input.layer_index, input.kv_head_index, token_index, 0,
            input.allocated_channel_ids, input.initial_channel_count
        ); // Validates layer/head and capacity before building the batch.

        KVAppendCommandBatch batch;
        batch.token_index = token_index;
        batch.request_block_index = token_index / m_layout.tokens_per_block();
        batch.channel_id = first_key.channel_id;

        const auto transaction_bytes = m_encoder.transaction_bytes();
        std::unordered_set<std::uint64_t> written_transactions; // 중복없는 주소 보관.
        const auto append_vector = [&](KVKind kind) {
            std::uint64_t write_count = 0;
            // dimension 원소 총 256개를 순회해도 요청은 transaction 별로 16개만 생성한다.
            for(std::uint64_t dimension = 0;
                dimension < m_layout.shape().head_dimension(); ++dimension) {
                const auto physical = m_address_mapper.locate_element(
                    kind, input.layer_index, input.kv_head_index,
                    token_index, dimension,
                    input.allocated_channel_ids, input.initial_channel_count
                );
                const auto element_address = m_encoder.encode_element(physical);
                // 같은 32B영역에 속하는 원소들을 모두 동일한 시작 주소로 바꾸는 계산 진행.
                const auto transaction_address = 
                    element_address - element_address % transaction_bytes;

                // Many FP16 elements shaer one 32-byte DRAM write request.
                // Iterate dimensions deterministically; 
                if(written_transactions.insert(transaction_address).second){
                    batch.commands.push_back(PIMCommand{PIMOpcode::MemoryWrite, transaction_address});
                    ++write_count;
                }
            }
            return write_count;
        };
        batch.key_write_count = append_vector(KVKind::Key);
        batch.value_write_count = append_vector(KVKind::Value);
        return batch;
    }

    QKCommandBatch PIMCommandGenerator::generate_qk(const QKCommandInput& input) const {
        validate_allocation(input.allocated_channel_ids, input.initial_channel_count);

        // Validates layer/head even for an empty operation.
        const auto key_base_row = m_layout.slice_base_row(KVKind::Key, input.layer_index, input.kv_head_index);

        const auto block_count = ceil_div(input.valid_token_count, m_layout.tokens_per_block());
        if(block_count != 0) {
            // Validate capacity before constructing a potentially large vector.
            (void)m_address_mapper.locate_mac(
                KVKind::Key, input.layer_index, input.kv_head_index,
                block_count - 1, 0,
                input.allocated_channel_ids, input.initial_channel_count
            );
        }

        QKCommandBatch batch; // 결과 객체.
        const auto dimension_steps = m_layout.columns_per_block_in_slice();
        const auto physical_dimensions = m_layout.geometry().banks_per_bank_group * dimension_steps * m_layout.elements_per_column();
        batch.padded_dimensions_per_token = physical_dimensions - m_layout.shape().head_dimension();
        if(block_count == 0) {
            return batch;
        }

        // Preserve allocation order. A channel participates only if it has valid KV.
        std::vector<std::vector<PIMMACLocation>> channel_blocks(input.allocated_channel_ids.size());
        std::uint64_t remainig_tokens = input.valid_token_count;
        for(std::uint64_t block_index = 0; block_index < block_count; ++block_index){
            const auto location = m_address_mapper.locate_mac(
                KVKind::Key, input.layer_index, input.kv_head_index,block_index,0,
                input.allocated_channel_ids, input.initial_channel_count);

            const auto channel_it = std::find(input.allocated_channel_ids.begin(), input.allocated_channel_ids.end(),location.channel_id);
            // 할당 채널 목록의 몇 번째 채널인지 찾는다.
            const auto channel_index = static_cast<std::size_t> (channel_it - input.allocated_channel_ids.begin());
            
            channel_blocks[channel_index].push_back(location);
            const auto valid_tokens = std::min(remainig_tokens, m_layout.tokens_per_block());
            batch.blocks.push_back(QKBlockWork{block_index,valid_tokens,location});
            remainig_tokens -= valid_tokens;
        }
        std::size_t maximum_local_blocks = 0; // 작업이 가장 많은 채널까지 계속 실행하기 위한 기준.
        for(std::size_t i = 0; i < channel_blocks.size(); ++i) {
            if(!channel_blocks[i].empty()) {
                batch.active_channel_ids.push_back(input.allocated_channel_ids[i]);
                maximum_local_blocks = std::max(maximum_local_blocks,channel_blocks[i].size());
            }
        }

        const auto append = [&batch](PIMOpcode opcode, std::uint64_t address) {
            batch.commands.push_back(PIMCommand{opcode, address});
        };

        // Query broadcast once per active channel, independent of its block count.
        for(std::uint64_t bank = 0; bank < m_layout.geometry().banks_per_bank_group; ++bank) {
            for(std::uint64_t slot = 0; slot < dimension_steps; ++slot) {
                for(const int channel_id : batch.active_channel_ids){
                    append(PIMOpcode::WriteGemvBuffer, encode_query_buffer(channel_id,bank,slot,key_base_row));
                }
            }
        }

        for(const int channel_id : batch.active_channel_ids) {
            append(PIMOpcode::Barrier, encode_channel(channel_id));
        }

        // A Simple phase-separated schedule. Channels can run in parallel;
        // groups do not overlap query writes/result moves with MACs on the same channel.
        // group : MAC을 수행한 뒤 결과를 한 번 이동하는 block처리 묶음이다.
        std::size_t group_begin = 0; // 이번 그룹이 시작하는 채널별 block 목록의 위치.
        while(group_begin < maximum_local_blocks) {
            // group_size : 이번 그룹에서 처리할 최대 block 수
            const auto group_size = std::min(
                score_blocks_per_flush, maximum_local_blocks-group_begin
            ); // 마지막 그룹에 남은 block이 16개보다 적을 수도 있다.

            // MAC 생성
            for(std::size_t local_offset = 0; local_offset < group_size; ++local_offset) {
                for(std::uint64_t step = 0; step < dimension_steps; ++step) {
                    for(const auto& locations : channel_blocks) {
                        const auto local_index = group_begin + local_offset;
                        if(local_index < locations.size()) {
                            auto location = locations[local_index];
                            location.col += step;
                            append(PIMOpcode::MacAllBank, m_encoder.encode_mac(location));
                        }
                    }
                }
            }
            for(std::size_t i = 0; i < channel_blocks.size(); ++i) {
                if(group_begin < channel_blocks[i].size()) {
                    append(PIMOpcode::Barrier, encode_channel(input.allocated_channel_ids[i]));
                }
            }

            // Match QK score_mvsb() : BG -> rank -> channel; pCH is broadcast.
            for(std::uint64_t bg = 0; bg < m_layout.geometry().bank_groups_per_rank; ++bg) {
                for(std::uint64_t rank = 0; rank < m_layout.geometry().ranks_per_pseudochannel; ++rank) {
                    for(std::size_t i = 0; i < channel_blocks.size(); ++i) {
                        if(group_begin < channel_blocks[i].size()) {
                            append(PIMOpcode::MoveSoftmaxBuffer, encode_score_buffer(
                                input.allocated_channel_ids[i], rank, bg, key_base_row
                            ));
                        }    
                    }
                }
            }

            for (std::size_t i = 0; i < channel_blocks.size(); ++i) {
                if (group_begin < channel_blocks[i].size()) {
                    append(PIMOpcode::Barrier, encode_channel(input.allocated_channel_ids[i]));
                }
            }
            group_begin += group_size; // 그룹을 마무리 하고 다음 그룹으로 이동한다.
        }
        return batch;
    }

    std::vector<PIMCommand> PIMCommandGenerator::generate_softmax(
        const QKCommandBatch& qk_batch
    ) const {
        std::vector<PIMCommand> commands;
        commands.reserve(qk_batch.active_channel_ids.size());
        for (const int channel_id : qk_batch.active_channel_ids) {
            commands.push_back(PIMCommand{
                PIMOpcode::Softmax, encode_channel(channel_id)
            });
        }
        return commands;
    }
}