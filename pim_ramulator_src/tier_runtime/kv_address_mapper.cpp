#include "kv_address_mapper.h"

#include <stdexcept>

namespace Ramulator::TierRuntime {

    KVAddressMapper::KVAddressMapper(const RowAlignedKVLayout& layout, const OrderedBlockMapper& block_mapper)
        : m_layout(layout),m_block_mapper(block_mapper) {
            if(m_layout.max_blocks_per_channel() != m_block_mapper.blocks_per_channel()) {
                throw std::invalid_argument(
                    "layout and block mapper have different blocks_per_channel"
                );
            }
        }

    PhysicalKVAddress KVAddressMapper::locate_element(
            KVKind kind,
            std::uint64_t layer_index,
            std::uint64_t kv_head_index,
            std::uint64_t token_index,
            std::uint64_t dimension_index,
            const std::vector<int>& allocated_channel_ids,
            std::uint64_t initial_channel_count
    ) const {
        // 1. Split a request-global token index into block and intra-block offset.
        const auto tokens_per_block = m_layout.tokens_per_block();
        const auto request_block_index = token_index / tokens_per_block;
        const auto token_offset_in_block = token_index % tokens_per_block;

        // 2. Find the physical channel and that channel's local block index.
        const auto block = m_block_mapper.locate_block(
            request_block_index,
            allocated_channel_ids,
            initial_channel_count
        );

        // 3. Split the token axis and dimension axis across physical partitions.
        const auto token_partition = m_layout.locate_token_partition(token_offset_in_block);

        const auto dimension_partition = m_layout.locate_dimension_partition(dimension_index);

        // 4. The K/V slice and local block choose the row and column.
        const auto row_column = m_layout.locate_mac(
            kind,
            layer_index,
            kv_head_index,
            block.channel_local_block_index,
            dimension_partition.dimension_step
        );

        // lane < elements_per_column, so this product fits within bytes_per_column.
        const auto byte_offset = dimension_partition.lane * m_layout.shape().bytes_per_element();
        
        return PhysicalKVAddress{
            block.channel_id,
            token_partition.pseudochannel,
            token_partition.rank,
            token_partition.bank_group,
            dimension_partition.bank,
            row_column.row,
            row_column.column,
            dimension_partition.lane,
            byte_offset
        };
    }

    PIMMACLocation KVAddressMapper::locate_mac(
        KVKind kind,
        std::uint64_t layer_index,
        std::uint64_t kv_head_index,
        std::uint64_t request_block_index,
        std::uint64_t dimension_step,
        const std::vector<int>& allocated_channel_ids,
        std::uint64_t initial_channel_count
    ) const {
        const auto block = m_block_mapper.locate_block(
        request_block_index,
        allocated_channel_ids,
        initial_channel_count
    );

        const auto row_column = m_layout.locate_mac(
            kind,
            layer_index,
            kv_head_index,
            block.channel_local_block_index,
            dimension_step
        );

        return PIMMACLocation{
            block.channel_id,
            row_column.row,
            row_column.column
        };
    }

} // namespace