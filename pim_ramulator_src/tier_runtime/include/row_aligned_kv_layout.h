#pragma once

#include <cstdint>

#include "kv_block_shape.h"

namespace Ramulator::TierRuntime {

    enum class KVKind : std::uint8_t {
        Key = 0,
        Value = 1
    };

    struct PIMGeometryConfig {
        std::uint64_t pseudochannels_per_channel = 0;
        std::uint64_t ranks_per_pseudochannel = 0;
        std::uint64_t bank_groups_per_rank = 0;
        std::uint64_t banks_per_bank_group = 0;

        std::uint64_t rows_per_bank = 0;
        std::uint64_t columns_per_row = 0;
        std::uint64_t bytes_per_column = 0;
    };

    struct TokenPartition {
        std::uint64_t pseudochannel = 0;
        std::uint64_t rank = 0;
        std::uint64_t bank_group = 0;
    };
    
    struct DimensionPartition {
        std::uint64_t bank = 0;
        std::uint64_t dimension_step = 0;
        std::uint64_t lane = 0;
    };

    struct RowColumnLocation {
        std::uint64_t row = 0;
        std::uint64_t column = 0;
    };

    class RowAlignedKVLayout {
        public :
            RowAlignedKVLayout(
                const KVBlockShape& shape,
                const PIMGeometryConfig& geometry
            );

            const KVBlockShape& shape() const noexcept;
            const PIMGeometryConfig& geometry() const noexcept;
            // Getter 함수
            std::uint64_t token_partition_count() const noexcept;
            std::uint64_t dimension_partition_count() const noexcept;
            std::uint64_t physical_bank_count() const noexcept;

            std::uint64_t tokens_per_block() const noexcept;
            std::uint64_t elements_per_column() const noexcept;
            std::uint64_t dimensions_per_bank() const noexcept;

            std::uint64_t columns_per_block_in_slice() const noexcept;
            std::uint64_t tiles_per_row() const noexcept;
            std::uint64_t tokens_per_row() const noexcept;

            std::uint64_t slice_count() const noexcept;
            std::uint64_t rows_per_slice_capacity() const noexcept;
            std::uint64_t unused_rows_per_bank() const noexcept;

            std::uint64_t max_blocks_per_channel() const noexcept;
            std::uint64_t max_tokens_per_channel() const noexcept;

            std::uint64_t row_bytes_per_bank() const noexcept;

            TokenPartition locate_token_partition(std::uint64_t token_in_tile) const;

            DimensionPartition locate_dimension_partition(std::uint64_t dimension_index) const;

            std::uint64_t slice_index(KVKind kind, std::uint64_t layer_index, std::uint64_t kv_head_index) const;

            std::uint64_t slice_base_row(KVKind kind, std::uint64_t layer_index, std::uint64_t kv_head_index) const;

            RowColumnLocation locate_mac(KVKind kind, std::uint64_t layer_index, std::uint64_t kv_head_index, 
                std::uint64_t channel_tile_index,
                std::uint64_t dimension_step) const;
            
            std::uint64_t required_rows_per_slice(std::uint64_t local_tile_count) const;

            std::uint64_t occupied_bytes(std::uint64_t local_tile_count) const;

        private :
            static std::uint64_t ceil_div(
                std::uint64_t value,
                std::uint64_t divisor
            );

            static std::uint64_t checked_multiply(
                std::uint64_t lhs,
                std::uint64_t rhs
            );

            KVBlockShape m_shape;
            PIMGeometryConfig m_geometry;

            std::uint64_t m_token_partition_count = 0;
            std::uint64_t m_dimension_partition_count = 0;
            std::uint64_t m_physical_bank_count = 0;

            std::uint64_t m_tokens_per_tile = 0;
            std::uint64_t m_elements_per_column = 0;
            std::uint64_t m_dimensions_per_bank = 0;

            std::uint64_t m_columns_per_tile = 0;
            std::uint64_t blocks_per_row_in_slice = 0;
            std::uint64_t m_tokens_per_row = 0;

            std::uint64_t m_slice_count = 0;
            std::uint64_t m_rows_per_slice_capacity = 0;
            std::uint64_t m_unused_rows_per_bank = 0;

            std::uint64_t m_max_tiles_per_channel = 0;
            std::uint64_t m_max_tokens_per_channel = 0;

            std::uint64_t m_row_bytes_per_bank = 0;
    };
}