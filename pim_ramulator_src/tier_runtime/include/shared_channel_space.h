#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "attacc_address_encoder.h"
#include "pim_command.h"

namespace Ramulator::TierRuntime {
    
    // One physical row across all channels and banks is a 1 MiB chunk with the
    // supported AttAcc geometry. Each K/V slice needs its own chunk; requests
    // grow by one logical row per slice without moving previously allocated rows.
    class SharedKVRowSpace{
        public:
            SharedKVRowSpace(const RowAlignedKVLayout& layout,
                             const AttAccAddressEncoder& encoder,
                             std::uint64_t channel_count,
                             std::uint64_t row_slot_limit = 0);
            // Admit using the existing prompt length; an empty prompt reserves one
            // row for the first generated token. Failed allocation has no side effects.
            [[nodiscard]] bool reserve(std::int64_t request_id,
                                       std::uint64_t prompt_tokens);
            // Grow on demand before submitting the append for this token. Returns
            // false when the row pool has insufficient space; the caller may retry.
            [[nodiscard]] bool ensure_capacity(std::int64_t request_id,
                                               std::uint64_t valid_token_count);
            void release(std::int64_t request_id);
            [[nodiscard]] std::uint64_t row_offset(std::int64_t request_id) const;
            [[nodiscard]] std::uint64_t physical_row(std::int64_t request_id,
                                                     std::uint64_t logical_row) const;
            [[nodiscard]] PIMCommand translate(std::int64_t request_id,
                                               PIMCommand command) const;
            [[nodiscard]] std::uint64_t rows_allocated(std::int64_t request_id) const;
            [[nodiscard]] std::uint64_t tokens_per_row_stripe() const noexcept;
            [[nodiscard]] std::uint64_t chunks_per_row_stripe() const noexcept;            
            
        private:
            const AttAccAddressEncoder& m_encoder;
            std::uint64_t m_rows_per_slice = 0; // 하나의 slice를 위해 할당된 rows의미.
            std::uint64_t m_slice_count = 0;
            std::uint64_t m_tokens_per_stripe = 0;
            std::vector<std::optional<std::int64_t>> m_slot_owners; // 각 물리 row를 어느 요청이 점유 중인지 기록한 표
            // Logical row index -> physical row within every slice. This small VA2PA
            // table lets a request grow into non-contiguous physical row stripes.
            std::unordered_map<std::int64_t, std::vector<std::uint64_t>> m_rows_by_request;
    };

    // A request keeps its PIM input/output buffer state from KV append through
    // the end of its token. Acquire is atomic across all participating channels.
    class ChannelPhaseGuard {
        public:
            explicit ChannelPhaseGuard(std::size_t channel_count);

            [[nodiscard]] bool try_acquire(std::int64_t request_id,
                                            const std::vector<int>& channels);
            void release(std::int64_t request_id, const std::vector<int>& channels);
        
        private:
                std::vector<std::optional<std::int64_t>> m_owners;
    };

} // namespace Ramulator::TierRuntime