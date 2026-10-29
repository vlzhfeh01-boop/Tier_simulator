/* 
    물리 좌표를 AttAcc가 해석할 수 있는 하나의 정수 주소로 조립하는 역할.
*/

#pragma once

#include <array>
#include <cstdint>

#include "kv_address_mapper.h"

namespace Ramulator::TierRuntime{
    // The seven coordinates decoded by AttAcc plus the low transaction-offset bits.
    // lane is element-type dependent, so decoding returns a byte offset instead.
    struct AttAccAddressCoordinates {
        int channel_id = -1;
        std::uint64_t pseudochannel = 0;
        std::uint64_t rank = 0;
        std::uint64_t bank_group = 0;
        std::uint64_t bank = 0;
        std::uint64_t row = 0;
        std::uint64_t column = 0;
        std::uint64_t byte_offset_in_column = 0;
    };

    // Supports AddrMapper impl: HBM3-PIM (not HBM3-PIM-Custom).
    // All geometry values must match the DRAM/YAML configuration.
    class AttAccAddressEncoder {
        public:
            AttAccAddressEncoder(
                const RowAlignedKVLayout& layout,
                std::uint64_t total_channel_count
            );

            [[nodiscard]]
            std::uint64_t encode(const AttAccAddressCoordinates& cordinates) const;

            [[nodiscard]]
            std::uint64_t encode_element(const PhysicalKVAddress& address) const;

            [[nodiscard]]
            std::uint64_t encode_mac(const PIMMACLocation& location) const;

            [[nodiscard]]
            AttAccAddressCoordinates decode(std::uint64_t address) const;

            [[nodiscard]]
            std::uint64_t transaction_bytes() const noexcept;

            [[nodiscard]]
            unsigned byte_offset_bits() const noexcept;

            [[nodiscard]]
            unsigned address_bits() const noexcept;

        private:
            static unsigned bits_for_power_of_two(std::uint64_t count);

            // Order: channel, pseudochannel, rank, bank group, bank, row, column.
            std::array<std::uint64_t, 7> m_counts{};
            std::array<unsigned, 7> m_bits{};
            std::array<unsigned, 7> m_shifts{};
            std::uint64_t m_transaction_bytes = 0;
            std::uint64_t m_bytes_per_element = 0;
            unsigned m_byte_offset_bits = 0;
            unsigned m_address_bits = 0;
    };
} // namespace Ramulator::TierRuntime