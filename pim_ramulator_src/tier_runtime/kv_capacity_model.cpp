#include "kv_capacity_model.h"

#include <limits>
#include <stdexcept>

namespace Ramulator{
namespace TierRuntime{
    namespace {
        std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) noexcept {
            return value / divisor + static_cast<std::uint64_t>(value %divisor !=0);
        }
    }
     
    KVCapacityModel::KVCapacityModel(const RowAlignedKVLayout& layout)
        : m_tokens_per_block(layout.tokens_per_block()),
          m_blocks_per_channel(layout.max_blocks_per_channel()) {

            if(m_tokens_per_block == 0) {
                throw std::invalid_argument(
                    "tokens_per_block must be greater than zero."
                );
            }

            if(m_blocks_per_channel == 0) {
                throw std::invalid_argument(
                    "blocks_per_channel must be greater than zero."
                );
            }
        }

    std::uint64_t KVCapacityModel::tokens_per_block() const noexcept {
        return m_tokens_per_block;
    }

    std::uint64_t KVCapacityModel::blocks_per_channel() const noexcept {
        return m_blocks_per_channel;
    }

    std::uint64_t KVCapacityModel::total_tokens(
        std::uint64_t prompt_tokens, 
        std::uint64_t future_output_tokens) const {
            return checked_add(prompt_tokens,future_output_tokens);
    }

    std::uint64_t KVCapacityModel::blocks_for_tokens(std::uint64_t token_count) const noexcept {
        if(token_count == 0)    
            return 0;
        
        return ceil_div(token_count,m_tokens_per_block);
    }

    std::uint64_t KVCapacityModel::required_channels_for_blocks (
        std::uint64_t block_count ) const noexcept {
            if(block_count == 0)
                return 0;

            return ceil_div(block_count, m_blocks_per_channel);
        }
    
    std::uint64_t KVCapacityModel::required_channels_for_tokens(
        std::uint64_t token_count) const noexcept {
            const std::uint64_t required_blocks = 
                blocks_for_tokens(token_count);
            
            return required_channels_for_blocks(required_blocks);
        }
    
    KVCapacityEstimate KVCapacityModel::estimate(
        std::uint64_t prompt_tokens,
        std::uint64_t future_output_tokens) const {
            const std::uint64_t estimated_total_tokens =
                total_tokens(prompt_tokens,future_output_tokens);

            const std::uint64_t estimated_total_blocks = 
                blocks_for_tokens(estimated_total_tokens);

            const std::uint64_t estimated_required_channels = 
                required_channels_for_blocks(estimated_total_blocks);

            KVCapacityEstimate result;
            result.total_tokens = estimated_total_tokens;
            result.total_blocks = estimated_total_blocks;
            result.required_channels = estimated_required_channels;

            return result;
        }

    std::uint64_t KVCapacityModel::block_capacity(std::uint64_t channel_count) const {
            return checked_multiply(channel_count,m_blocks_per_channel);
    }

    bool KVCapacityModel::fits(
        std::uint64_t token_count,
        std::uint64_t channel_count) const noexcept {
            const std::uint64_t required_channels = required_channels_for_tokens(token_count);

            return required_channels <= channel_count;
        }
    
    bool KVCapacityModel::next_token_starts_new_block(std::uint64_t current_token_count) const noexcept {
        return current_token_count % m_tokens_per_block == 0;
    }
    
    bool KVCapacityModel::next_token_requires_growth(
        std::uint64_t current_token_count,
        std::uint64_t allocated_channel_count) const {
            if(allocated_channel_count == 0) {
                throw std::logic_error(
                    "an active request must own at least one channel."
                );
            }
            if(!fits(current_token_count, allocated_channel_count)) {
                throw std::logic_error(
                    "Current KV cache already exceeds the allocated channel capacity."
                );
            }
            if(!next_token_starts_new_block(current_token_count))
                return false;
            
            const std::uint64_t current_block_count = blocks_for_tokens(current_token_count);

            const std::uint64_t current_block_capacity = block_capacity(allocated_channel_count);
            // 현재 마지막 block까지 모두 사용하고 있다. 를 의미.
            return current_block_count == current_block_capacity;
        }
    

    std::uint64_t KVCapacityModel::checked_add(
        std::uint64_t lhs,
        std::uint64_t rhs) {
            const std::uint64_t max_value = std::numeric_limits<std::uint64_t>::max();
            
            if(rhs > max_value - lhs) {
                throw std::overflow_error(
                    "uint64 addition overflow"
                );
            }
            return lhs + rhs;
        }
    std::uint64_t KVCapacityModel::checked_multiply(
        std::uint64_t lhs,
        std::uint64_t rhs){
            if (lhs == 0 || rhs == 0) {
                return 0;
            }

            const std::uint64_t max_value =
                std::numeric_limits<std::uint64_t>::max();

            if (lhs > max_value / rhs) {
                throw std::overflow_error(
                    "uint64 multiplication overflow");
            }

            return lhs * rhs;
    }
}
}