#include "ordered_block_mapper.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace Ramulator{
namespace TierRuntime{
    OrderedBlockMapper::OrderedBlockMapper(const KVCapacityModel &capacity_model)
        :m_blocks_per_channel(capacity_model.blocks_per_channel()) {
            if(m_blocks_per_channel == 0) {
                throw std::invalid_argument(
                    "blocks_per_channel must be greater than zero"
                );
            }
    }
    
    std::uint64_t OrderedBlockMapper::blocks_per_channel() const noexcept {
        return m_blocks_per_channel;
    }

    std::uint64_t OrderedBlockMapper::checked_add(std::uint64_t lhs, std::uint64_t rhs) {
        const auto max_value = std::numeric_limits<std::uint64_t>::max();

        if(lhs > max_value - rhs){
            throw std::overflow_error("block mapping addition overflow");
        }
        return lhs + rhs;
    }

    std::uint64_t OrderedBlockMapper::checked_multiply(std::uint64_t lhs,std::uint64_t rhs) {
        const auto max_value = std::numeric_limits<std::uint64_t>::max();
        if (rhs != 0 && lhs > max_value / rhs) {
            throw std::overflow_error("block mapping multiplication overflow");
        }
        return lhs * rhs;
    }

    BlockPlacement OrderedBlockMapper::locate_block(std::uint64_t request_block_index,
        const std::vector<int>& allocated_channel_ids,
        std::uint64_t initial_channel_count) const {
            if(allocated_channel_ids.empty()) {
                throw std::invalid_argument("allocated_channel_ids must not be empty");
            }
            if(initial_channel_count == 0) {
                throw std::invalid_argument("initial_channel_count must be greater than zero");
            }
            if(initial_channel_count > allocated_channel_ids.size()) {
                throw std::invalid_argument("initial_channel_count exceeds the number of allocated channels");
            }

            // The initial channel group has a fixed size for the request lifetime.
            const std::uint64_t initial_block_capacity = checked_multiply(initial_channel_count,m_blocks_per_channel);

            std::uint64_t channel_index = 0;
            std::uint64_t channel_local_block_index = 0;

            if(request_block_index < initial_block_capacity) {
                // Stripe initial blocks across the original channels in allocation order.
                channel_index = request_block_index % initial_channel_count;
                channel_local_block_index = request_block_index / initial_channel_count;
            }
            else {
                // Fill each appended growth channel before using the next one.
                const auto growth_block_index = request_block_index - initial_block_capacity;

                channel_index = checked_add(initial_channel_count, growth_block_index/m_blocks_per_channel);
                channel_local_block_index = growth_block_index % m_blocks_per_channel;
            }

            if(channel_index >= allocated_channel_ids.size()) {
                throw std::out_of_range(
                    "request block requires a channel that has not been allocated."
                );
            }

            const int channel_id = allocated_channel_ids[static_cast<std::size_t>(channel_index)];
            if(channel_id < 0) {
                throw std::invalid_argument("selected channel_id must be non-negative.");
            }
            return BlockPlacement{
                request_block_index,
                channel_id,
                channel_local_block_index};
        }


} // TierRuntime
} // Ramulator