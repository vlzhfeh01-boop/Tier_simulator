// 이 블록은 할당된 채널 중 어디에 놓는가?
/* 
    초기 채널에는 round-robin으로 배치
    초기 채널이 모두 차면 확장 채널을 하나씩 순차적으로 채움
    확장 후에도 기존 블록의 위치를 유지
*/
#pragma once

#include <cstdint>
#include <vector>

#include "kv_capacity_model.h"

namespace Ramulator {
namespace TierRuntime{
    // mapping 결과만 들어간다.
    struct BlockPlacement{
        std::uint64_t request_block_index = 0; // 요청 전체에서 몇 번째 block인지 나타낸다.
        int channel_id = -1; 
        std::uint64_t channel_local_block_index = 0; // 선택된 채널 내부에서 몇 번쨰 block인지 나타낸다.
    };
    class OrderedBlockMapper {
        public:
            explicit OrderedBlockMapper(const KVCapacityModel &capacity_model);

            [[nodiscard]]
            std::uint64_t blocks_per_channel() const noexcept;

            [[nodiscard]]
            BlockPlacement locate_block(
                std::uint64_t request_block_index,
                const std::vector<int>& allocated_channel_ids, // 요청이 현재 가지고 있는 채널을 할당 순서대로 저장한다.
                std::uint64_t initial_channel_count // 최초로 할당받았던 채널 수
            ) const;

        private:
            static std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs);

            static std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs);

            std::uint64_t m_blocks_per_channel;
    };

}   // TierRuntime
}   // Ramulator