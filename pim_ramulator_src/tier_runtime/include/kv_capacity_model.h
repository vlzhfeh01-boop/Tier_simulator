#pragma once

#include <cstdint>

#include "row_aligned_kv_layout.h"

/* 
token, block, channel 개수 계산
*/
namespace Ramulator::TierRuntime{
    // 초기 admission 단계에서 한 번의 계산으로 필요한 결과를 반환하는 구조체
    struct KVCapacityEstimate {
        std::uint64_t total_tokens = 0;
        std::uint64_t total_blocks = 0;
        std::uint64_t required_channels = 0;
    };

    class KVCapacityModel {
        public : 
        // layout으로 부터 값을 가져온다.
            explicit KVCapacityModel(const Ramulator::TierRuntime::RowAlignedKVLayout &layout);

            [[nodiscard]]
            std::uint64_t tokens_per_block() const noexcept;
            
            [[nodiscard]]
            std::uint64_t blocks_per_channel() const noexcept;
            
            [[nodiscard]]
            std::uint64_t total_tokens(
                std::uint64_t prompt_tokens,
                std::uint64_t future_output_tokens) const;
            
            [[nodiscard]] // 반환값을 사용하지 않으면 컴파일러가 경고할 수 있게 한다.
            std::uint64_t blocks_for_tokens(
                std::uint64_t token_count) const noexcept;

            [[nodiscard]]
            std::uint64_t required_channels_for_blocks(
                std::uint64_t block_count) const noexcept;

            [[nodiscard]]
            std::uint64_t required_channels_for_tokens(
                std::uint64_t token_count) const noexcept;

            [[nodiscard]]
            KVCapacityEstimate estimate(
                std::uint64_t prompt_tokens,
                std::uint64_t future_output_tokens) const;

            [[nodiscard]]
            std::uint64_t block_capacity(
                std::uint64_t channel_count) const;

            [[nodiscard]]
            bool fits(
                std::uint64_t token_count,
                std::uint64_t channel_count) const noexcept;

            [[nodiscard]]
            bool next_token_starts_new_block(
                std::uint64_t current_token_count) const noexcept;

            [[nodiscard]]
            bool next_token_requires_growth(
                std::uint64_t current_token_count,
                std::uint64_t allocated_channel_count) const;

        private:
            static std::uint64_t checked_add(
                std::uint64_t lhs,
                std::uint64_t rhs);

            static std::uint64_t checked_multiply(
                std::uint64_t lhs,
                std::uint64_t rhs);

            std::uint64_t m_tokens_per_block;
            std::uint64_t m_blocks_per_channel;
    };
}