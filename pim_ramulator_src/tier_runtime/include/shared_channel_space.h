#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "attacc_address_encoder.h"
#include "pim_command.h"

namespace Ramulator::TierRuntime{

    // Reserve disjoint row windows in every K/V slice for requests that reuse physical channels.
    // The window size is a capacity constraint.
    class SharedKVRowSpace{
        public:
            SharedKVRowSpace(const RowAlignedKVLayout& layout,
                             const AttAccAddressEncoder& encoder,
                             std::uint64_t rows_per_request);

            [[nodiscard]] bool reserve(std::int64_t request_id);
            void release(std::int64_t request_id);
            [[nodiscard]] std::uint64_t row_offset(std::int64_t request_id) const;
            [[nodiscard]] PIMCommand translate(std::int64_t request_id,
                                               PIMCommand command) const;
            [[nodiscard]] std::uint64_t rows_per_request() const noexcept;
        
        private:
            const AttAccAddressEncoder& m_encoder;
            std::uint64_t m_rows_per_slice = 0;
            std::uint64_t m_rows_per_request = 0;
            std::uint64_t m_slice_count = 0;
            std::vector<std::optional<std::int64_t>> m_slot_owners;
            std::unordered_map<std::int64_t, std::uint64_t> m_slot_by_request;
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