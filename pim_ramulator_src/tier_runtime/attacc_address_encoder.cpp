#include "attacc_address_encoder.h"

#include <stdexcept>
#include <string>

namespace Ramulator::TierRuntime {
    // 지수 n이 얼마인지 구하는 함수
    unsigned AttAccAddressEncoder::bits_for_power_of_two(std::uint64_t count) {
        // 2의 거듭제곱 판별 테크닉.
        /* 
            count가 2의 거듭제곱수이면,
            1000 & 0111 = 0이어야 한다.
        */
        if (count == 0 || (count & (count - 1)) != 0) {
            throw std::invalid_argument(
                "AttAcc linear address mapping requires power-of-two counts"
            );
        }
        unsigned bits = 0;
        while (count > 1) {
            count >>= 1;
            ++bits;
        }
        return bits;
    }


    AttAccAddressEncoder::AttAccAddressEncoder(
        const RowAlignedKVLayout& layout,
        std::uint64_t total_channel_count
    ) : m_counts{
        total_channel_count, // 시스템 전체 채널 수.
        layout.geometry().pseudochannels_per_channel,
        layout.geometry().ranks_per_pseudochannel,
        layout.geometry().bank_groups_per_rank,
        layout.geometry().banks_per_bank_group,
        layout.geometry().rows_per_bank,
        layout.geometry().columns_per_row
    }, m_transaction_bytes(layout.geometry().bytes_per_column),
    m_bytes_per_element(layout.shape().bytes_per_element()) {
        // In HBM3-PIM, tx_bytes = internal_prefetch_size * channel_width / 8.
        // The caller must use that same value for geometry.bytes_per_column.
        // 한번의 transaction에서 32B가 움직인다. 32B내부의 offset 표현에 5bits 필요.
        m_byte_offset_bits = bits_for_power_of_two(m_transaction_bytes);
        if(m_byte_offset_bits > 30) {
            throw std::invalid_argument("transaction size exceeds AttAcc's int geometry");
        }
        for(std::size_t i = 0; i<m_counts.size(); ++i) {
            m_bits[i] = bits_for_power_of_two(m_counts[i]);
            if(m_bits[i] > 30) {
                throw std::invalid_argument("address field exceeds AttAcc decoder width");
            }
        }

        unsigned next_shift = m_byte_offset_bits;
        // 주소의 낮은 비트부터 배치. column부터 시작.
        for(std::size_t remaining = m_counts.size(); remaining>0 ; --remaining) {
            const auto i = remaining - 1;
            m_shifts[i] = next_shift;
            // Addr_t is int64_t and the trace parser uses stoll. Keep bit 63 clear.
            if(m_bits[i] > 63 - next_shift) {
                throw std::overflow_error("AttAcc address needs more than 63 bits");
            }
            next_shift += m_bits[i];
        }
        m_address_bits = next_shift;
    }

    std::uint64_t AttAccAddressEncoder::encode(
        const AttAccAddressCoordinates& coordinates
    ) const {
        if (coordinates.channel_id < 0) {
            throw std::out_of_range("channel_id must be nonnegative");
        }
        const std::array<std::uint64_t, 7> values{
            static_cast<std::uint64_t>(coordinates.channel_id),
            coordinates.pseudochannel,
            coordinates.rank,
            coordinates.bank_group,
            coordinates.bank,
            coordinates.row,
            coordinates.column
        };
        static constexpr std::array<const char*, 7> names{
            "channel", "pseudochannel", "rank", "bank_group", "bank", "row", "column"
        };
        if (coordinates.byte_offset_in_column >= m_transaction_bytes) {
            throw std::out_of_range("byte_offset_in_column exceeds transaction size");
        }
        auto address = coordinates.byte_offset_in_column;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (values[i] >= m_counts[i]) {
                throw std::out_of_range(std::string(names[i]) + " exceeds AttAcc geometry");
            }
            address |= values[i] << m_shifts[i];
        }
        return address;
    }

    std::uint64_t AttAccAddressEncoder::encode_element(const PhysicalKVAddress &address) const {
        const auto elements_per_column = m_transaction_bytes / m_bytes_per_element;

        if (address.lane >= elements_per_column) {
        throw std::out_of_range("lane exceeds elements per column");
        }
        if (address.byte_offset_in_column != address.lane * m_bytes_per_element) {
            throw std::invalid_argument("lane and byte offset are inconsistent");
        }
        return encode(AttAccAddressCoordinates{
            address.channel_id,
            address.pseudochannel,
            address.rank,
            address.bank_group,
            address.bank,
            address.row,
            address.column,
            address.byte_offset_in_column
        });

    }

    std::uint64_t AttAccAddressEncoder::encode_mac(const PIMMACLocation& location) const {
        // MAC_AB는 하나의 채널 내 여러 bank에 걸친 연산이다.
        // 특정 원소의 bank, lane을 선택하는 주소 필요 X.
        return encode(AttAccAddressCoordinates{
        location.channel_id, 0, 0, 0, 0, location.row, location.col, 0
        }); 

    }

    AttAccAddressCoordinates AttAccAddressEncoder::decode(std::uint64_t address) const {
        // 설정된 주소 폭으로 표현할 수 있는 최댓값
        const auto maximum_address = (std::uint64_t{1} << m_address_bits) - 1;

        if (address > maximum_address) {
            throw std::out_of_range("address has bits outside the configured geometry");
        }
        std::array<std::uint64_t, 7> values{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto mask = m_counts[i] - 1;
            values[i] = (address >> m_shifts[i]) & mask;
        }
        return AttAccAddressCoordinates{
            static_cast<int>(values[0]),
            values[1], values[2], values[3], values[4], values[5], values[6],
            address & (m_transaction_bytes - 1)
        };
    }
    std::uint64_t AttAccAddressEncoder::transaction_bytes() const noexcept {
        return m_transaction_bytes;
    }

    unsigned AttAccAddressEncoder::byte_offset_bits() const noexcept {
        return m_byte_offset_bits;
    }

    unsigned AttAccAddressEncoder::address_bits() const noexcept {
        return m_address_bits;
    }

} // namespace Ramulator::TierRuntime