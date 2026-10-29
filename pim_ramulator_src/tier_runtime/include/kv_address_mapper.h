/* 
    기존 블록 배치와 블록 내부 layout을 연결하는 역할을 한다.

    OrderedBlockMapper
    RowAlignedKVLayout
    두가지로부터의 결과를 하나의 물리좌표로 조합한다.
*/
#pragma once

#include <cstdint>
#include <vector>

#include "ordered_block_mapper.h"
#include "row_aligned_kv_layout.h"

namespace Ramulator::TierRuntime {
    // Coordinates of one K/V element. This is not an encoded AttAcc trace address.
    struct PhysicalKVAddress {
        int channel_id = -1;
        std::uint64_t pseudochannel = 0;
        std::uint64_t rank = 0;
        std::uint64_t bank_group = 0;
        std::uint64_t bank = 0;
        std::uint64_t row = 0;
        std::uint64_t column = 0;
        std::uint64_t lane = 0; // column안의 원소번호 의미.
        std::uint64_t byte_offset_in_column = 0;
    };

    // MAC_AB broadcasts to the banks of a channel at this row/column.
    // Do not create one MAC_AB per PhysicalKVAddress returned by locate_element().
    struct PIMMACLocation {
        int channel_id = -1;
        std::uint64_t row = 0;
        std::uint64_t col = 0;
    };

    class KVAddressMapper {
        public:
            // Copy immutable configurations so the mapper owns their lifetimes.
            KVAddressMapper(
                const RowAlignedKVLayout& layout,
                const OrderedBlockMapper& block_mapper
            );

            [[nodiscard]]
            PhysicalKVAddress locate_element(
                KVKind kind,
                std::uint64_t layer_index,
                std::uint64_t kv_head_index,
                std::uint64_t token_index,
                std::uint64_t dimension_index,
                const std::vector<int>& allocated_channel_ids,
                std::uint64_t initial_channel_count
            ) const;

            [[nodiscard]]
            PIMMACLocation locate_mac(
                KVKind kind,
                std::uint64_t layer_index,
                std::uint64_t kv_head_index,
                std::uint64_t request_block_index,
                std::uint64_t dimension_step,
                const std::vector<int>& allocated_channel_ids,
                std::uint64_t initial_channel_count
            ) const;

            private:
                // Both methods inherit OrderedBlockMapper's channel ownership/order
                // preconditions. Runtime must distinguish valid KV from reserved capacity
                // and handle partly valid blocks/padded dimensions during command generation.

                RowAlignedKVLayout m_layout;
                OrderedBlockMapper m_block_mapper;
    };
} // namespace Ramulator::TierRuntime