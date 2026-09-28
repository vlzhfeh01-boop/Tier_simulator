#include "row_aligned_kv_layout.h"

#include <limits>
#include <stdexcept>

namespace Ramulator::TierRuntime {
    RowAlignedKVLayout::RowAlignedKVLayout(
        const KVBlockShape& shape,
        const PIMGeometryConfig& geometry
    ) : m_shape(shape),m_geometry(geometry) {
        if (m_geometry.pseudochannels_per_channel == 0 ||
              m_geometry.ranks_per_pseudochannel == 0 ||
              m_geometry.bank_groups_per_rank == 0 ||
              m_geometry.banks_per_bank_group == 0 ||
              m_geometry.rows_per_bank == 0 ||
              m_geometry.columns_per_row == 0 ||
              m_geometry.bytes_per_column == 0) {
                throw std::invalid_argument(
                    "All PIM geometry values must be greater than zero.");
                }
        if(m_geometry.bytes_per_column % m_shape.bytes_per_element() !=0) {
            throw std::invalid_argument(
                "bytes_per_column must be divisible by bytes_per_element."
            );
        }

        m_token_partition_count = checked_multiply(
            m_geometry.pseudochannels_per_channel,
            m_geometry.ranks_per_pseudochannel
        );

        m_token_partition_count = checked_multiply(
            m_token_partition_count,
            m_geometry.bank_groups_per_rank
        );
        // m_token_partition_count : 채널내의 token 나누는 기준

        m_dimension_partition_count = m_geometry.banks_per_bank_group;

        m_physical_bank_count = checked_multiply(
            m_token_partition_count,m_dimension_partition_count
        );

        m_tokens_per_tile = m_token_partition_count; 

        if (m_shape.tokens_per_block() != m_tokens_per_tile) {
            throw std::invalid_argument(
                "KVBlockShape::tokens_per_block must match "
                "the PIM token partition count.");
        }

        m_elements_per_column = m_geometry.bytes_per_column / m_shape.bytes_per_element();

        m_dimensions_per_bank = ceil_div(m_shape.head_dimension(),m_dimension_partition_count);

        m_columns_per_tile = ceil_div(m_dimensions_per_bank , m_elements_per_column);

        if (m_columns_per_tile > m_geometry.columns_per_row) {
            throw std::invalid_argument(
                "A token tile does not fit in one row.");
        }

        blocks_per_row_in_slice = m_geometry.columns_per_row / m_columns_per_tile; // 하나의 row당 차지하는 tile수
        if (blocks_per_row_in_slice == 0) {
            throw std::logic_error(
                "tiles_per_row must be greater than zero.");
        }

        m_tokens_per_row = checked_multiply(blocks_per_row_in_slice,m_tokens_per_tile);

        m_slice_count = checked_multiply(2,m_shape.layer_head_count()); // K,V slice

        if (m_slice_count > m_geometry.rows_per_bank) {
            throw std::invalid_argument(
                "There are more KV slices than rows per bank.");
        }

        m_rows_per_slice_capacity = m_geometry.rows_per_bank / m_slice_count;

        m_unused_rows_per_bank = m_geometry.rows_per_bank % m_slice_count;

        m_max_tiles_per_channel = checked_multiply(m_rows_per_slice_capacity,blocks_per_row_in_slice);

        m_max_tokens_per_channel = checked_multiply(m_max_tiles_per_channel, m_tokens_per_tile);

        m_row_bytes_per_bank = checked_multiply(m_geometry.columns_per_row,m_geometry.bytes_per_column);

    }

    const KVBlockShape& RowAlignedKVLayout::shape() const noexcept {
        return m_shape;
    }
    const PIMGeometryConfig& RowAlignedKVLayout::geometry() const noexcept {
        return m_geometry;
    }
    std::uint64_t RowAlignedKVLayout::token_partition_count() const noexcept {
        return m_token_partition_count;
    }


    std::uint64_t RowAlignedKVLayout::dimension_partition_count() const noexcept {
        return m_dimension_partition_count;
    }


    std::uint64_t RowAlignedKVLayout::physical_bank_count() const noexcept {
        return m_physical_bank_count;
    }


    std::uint64_t   RowAlignedKVLayout::tokens_per_block() const noexcept {
        return m_tokens_per_tile;
    }


    std::uint64_t   RowAlignedKVLayout::elements_per_column() const noexcept {
        return m_elements_per_column;
    }


    std::uint64_t RowAlignedKVLayout::dimensions_per_bank() const noexcept {
        return m_dimensions_per_bank;
    }


    std::uint64_t RowAlignedKVLayout::columns_per_block_in_slice() const noexcept {
        return m_columns_per_tile;
    }


    std::uint64_t RowAlignedKVLayout::tiles_per_row() const noexcept {
        return blocks_per_row_in_slice;
    }


    std::uint64_t RowAlignedKVLayout::tokens_per_row() const noexcept {
        return m_tokens_per_row;
    }


    std::uint64_t RowAlignedKVLayout::slice_count() const noexcept {
        return m_slice_count;
    }


    std::uint64_t RowAlignedKVLayout::rows_per_slice_capacity() const noexcept {
        return m_rows_per_slice_capacity;
    }


    std::uint64_t RowAlignedKVLayout::unused_rows_per_bank() const noexcept {
        return m_unused_rows_per_bank;
    }


    std::uint64_t RowAlignedKVLayout::max_blocks_per_channel() const noexcept {
        return m_max_tiles_per_channel;
    }


    std::uint64_t RowAlignedKVLayout::max_tokens_per_channel() const noexcept {
        return m_max_tokens_per_channel;
    }


    std::uint64_t RowAlignedKVLayout::row_bytes_per_bank() const noexcept {
        return m_row_bytes_per_bank;
    }

    TokenPartition RowAlignedKVLayout::locate_token_partition(
        std::uint64_t token_offset_in_tile) const {
            if(token_offset_in_tile >= m_tokens_per_tile) {
                throw std::out_of_range(
                    "token_in_tile is outside the token tile."
                );
            }

            std::uint64_t remaining = token_offset_in_tile;

            const std::uint64_t bank_group = remaining % m_geometry.bank_groups_per_rank;

            remaining /= m_geometry.bank_groups_per_rank;

            const std::uint64_t rank = remaining % m_geometry.ranks_per_pseudochannel;

            remaining /= m_geometry.ranks_per_pseudochannel;

            const std::uint64_t pseudochannel = remaining;

            return {
                pseudochannel,
                rank,
                bank_group
            };
        }

        DimensionPartition RowAlignedKVLayout::locate_dimension_partition(
            std::uint64_t dimension_index) const {
                if(dimension_index >= m_shape.head_dimension()) {
                    throw std::out_of_range(
                        "dimension_index is outside the head dimension"
                    );
                }   

                const std::uint64_t bank = dimension_index / m_dimensions_per_bank;

                const std::uint64_t dimension_in_bank = dimension_index % m_dimensions_per_bank;

                const std::uint64_t dimension_step = dimension_in_bank / m_elements_per_column;

                const std::uint64_t lane = dimension_in_bank % m_elements_per_column;

                return {
                    bank,
                    dimension_step,
                    lane
                };
        }

        std::uint64_t RowAlignedKVLayout::slice_index(
            KVKind kind, 
            std::uint64_t layer_index, 
            std::uint64_t kv_head_index) const {
                if(layer_index >= m_shape.layers_on_hbm()) {
                    throw std::out_of_range(
                        "layer_index is outside layers_on_hbm."
                    );
                }
            
                if(kv_head_index >= m_shape.kv_heads_per_layer_on_hbm()) {
                    throw std::out_of_range(
                        "kv_head_index is outside kv_heads_per_layer_on_hbm."
                    );
                }
                // Slice하나가 한개의 row를 차지한다.
                // layer 하나당 여러개의 kv head 존재.
                const std::uint64_t local_slice_index = 
                    layer_index * m_shape.kv_heads_per_layer_on_hbm() + kv_head_index;

                switch(kind) {
                    case KVKind::Key :
                        return local_slice_index;
                    case KVKind::Value :
                        return m_shape.layer_head_count() + local_slice_index;
                }

                throw std::invalid_argument(
                    "Unkown KVKind Value."
                );
        }
        // 특정 채널 내부에서 특정 K/V , layer, head의 local slice 영역이 시작하는 row
        std::uint64_t RowAlignedKVLayout::slice_base_row(
            KVKind kind,
            std::uint64_t layer_index,
            std::uint64_t kv_head_index) const {
                // index : 시작하는 row를 의미한다.
                const std::uint64_t index = slice_index(kind,layer_index,kv_head_index);
                // 
                return checked_multiply(index, m_rows_per_slice_capacity);
            }
        
        RowColumnLocation RowAlignedKVLayout::locate_mac(
            KVKind kind,
            std::uint64_t layer_index,
            std::uint64_t kv_head_index,
            std::uint64_t channel_block_index, // local tile index를 의미
            std::uint64_t dimension_step) const {
                if(channel_block_index >= m_max_tiles_per_channel) {
                    throw std::out_of_range(
                        "channel_tile_index exceeds channel capacity."
                    );
                }
                if(dimension_step >= m_columns_per_tile) {
                    throw std::out_of_range(
                        "dimension_step exceeds columns_per_tile."
                    );
                }

                const std::uint64_t row_offset_in_slice = channel_block_index / blocks_per_row_in_slice;

                const std::uint64_t tile_slot_in_row = channel_block_index % blocks_per_row_in_slice;

                const std::uint64_t first_column = tile_slot_in_row * m_columns_per_tile;

                const std::uint64_t row = slice_base_row(kind,layer_index,kv_head_index) + row_offset_in_slice;

                const std::uint64_t column = first_column + dimension_step;

                return {row,column};
            }
            // slice 하나당 여러개의 연속된 row를 사용할 수 있다.
        std::uint64_t RowAlignedKVLayout::required_rows_per_slice(
            std::uint64_t local_tile_count) const  {
                if(local_tile_count > m_max_tiles_per_channel) {
                    throw std::out_of_range(
                        "local_tile_count exceeds channel capacity."
                    );
                }

                if(local_tile_count == 0)
                    return 0;

                return ceil_div(local_tile_count, blocks_per_row_in_slice);
            }
        
        std::uint64_t RowAlignedKVLayout::occupied_bytes(std::uint64_t local_tile_count) const {
            const std::uint64_t used_rows_per_slice = 
                required_rows_per_slice(local_tile_count);

            const std::uint64_t used_row_stripes =
                checked_multiply(used_rows_per_slice,m_slice_count);

            const std::uint64_t bytes_per_row_stripe = 
                checked_multiply(m_row_bytes_per_bank,m_physical_bank_count);

            return checked_multiply(used_row_stripes,bytes_per_row_stripe);

        }

        std::uint64_t RowAlignedKVLayout::ceil_div(
            std::uint64_t value,
            std::uint64_t divisor) {
            if (divisor == 0) {
                throw std::invalid_argument(
                "ceil_div divisor must be greater than zero.");
            }

            return value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
        }


        std::uint64_t RowAlignedKVLayout::checked_multiply(
            std::uint64_t lhs,
            std::uint64_t rhs) {
            if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
                throw std::overflow_error(
                "Unsigned integer multiplication overflow.");
            }

            return lhs * rhs;
        }
}   