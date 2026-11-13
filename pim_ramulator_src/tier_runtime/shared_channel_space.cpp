#include "shared_channel_space.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace Ramulator::TierRuntime{
    SharedKVRowSpace::SharedKVRowSpace(const RowAlignedKVLayout& layout,
                                       const AttAccAddressEncoder& encoder,
                                       std::uint64_t channel_count,
                                       std::uint64_t row_slot_limit)
        : m_encoder(encoder),
          m_rows_per_slice(layout.rows_per_slice_capacity()),
          m_slice_count(layout.slice_count()) {
        const auto& geometry = layout.geometry();
        // The currently modeled AttAcc stack has 16 channels, 64 banks/channel,
        // and 32 columns * 32 bytes/row/bank: 16 * 64 * 1024 = 1 MiB.
        if (channel_count == 0 || m_rows_per_slice == 0 ||
            m_rows_per_slice > geometry.rows_per_bank / m_slice_count ||
            layout.row_bytes_per_bank() != 1024 ||
            layout.physical_bank_count() != 64 || channel_count != 16) {
            throw std::invalid_argument("shared KV requires a 1 MiB AttAcc row stripe");
        }
        if (layout.tokens_per_row() >
            std::numeric_limits<std::uint64_t>::max() / channel_count) {
            throw std::overflow_error("shared KV stripe token count overflow");
        }
        if(row_slot_limit > m_rows_per_slice) {
            throw std::invalid_argument("shared row slot limit exceeds physical capacity");
        }
        m_tokens_per_stripe = layout.tokens_per_row() * channel_count;
        // Zero retains the physical capacity. A smaller limit tests pressure
        // without changing the size or address layout of each 1 MiB chunk.
        m_slot_owners.resize(static_cast<std::size_t>(
            row_slot_limit == 0 ? m_rows_per_slice : row_slot_limit));
    }
    bool SharedKVRowSpace::reserve(std::int64_t request_id,
                                   std::uint64_t prompt_tokens) {
        if (request_id < 0 || m_rows_by_request.count(request_id)) {
            throw std::invalid_argument("invalid or repeated shared KV request ID");
        }
        m_rows_by_request.emplace(request_id, std::vector<std::uint64_t>{});
        try {
            // There is always a first append, even for an empty prompt.
            if (ensure_capacity(request_id, std::max<std::uint64_t>(prompt_tokens, 1))) {
                return true;
            }
        } catch (...) {
            m_rows_by_request.erase(request_id);
            throw;
        }
        m_rows_by_request.erase(request_id);
        return false;
    }

    bool SharedKVRowSpace::ensure_capacity(std::int64_t request_id,
                                           std::uint64_t valid_token_count) {
        auto found = m_rows_by_request.find(request_id);
        if (found == m_rows_by_request.end()) {
            throw std::logic_error("request has no shared KV mapping");
        }
        // 해당 토큰 수에 필요한 논리 row수
        // 올림 계산
        const auto needed = valid_token_count / m_tokens_per_stripe +
                            static_cast<std::uint64_t>(valid_token_count % m_tokens_per_stripe != 0);
        // 해당 요청이 이미 확보한 row 수
        auto& rows = found->second;
        if (needed <= rows.size()) return true;
        if (needed > m_slot_owners.size()) return false;
        std::vector<std::uint64_t> free_rows;
        free_rows.reserve(static_cast<std::size_t>(needed - rows.size()));
        // 점유되지 않은 물리 row를 free_rows에 추가
        for (std::size_t index = 0; index < m_slot_owners.size() &&
                                    free_rows.size() < needed - rows.size(); ++index) {
            if (!m_slot_owners[index]) free_rows.push_back(index);
        }
        // 필요한 row 수만큼 확보되지 못했으면 false 반환
        if (free_rows.size() != needed - rows.size()) return false;
        // 필요한 row 수만큼 확보되었으면 rows에 추가하고 m_slot_owners에 점유 기록
        rows.reserve(static_cast<std::size_t>(needed));
        for (const auto row : free_rows) {
            rows.push_back(row);
            m_slot_owners[static_cast<std::size_t>(row)] = request_id;
        }
        return true;
    }
    
     void SharedKVRowSpace::release(std::int64_t request_id) {
        const auto found = m_rows_by_request.find(request_id);
        if (found == m_rows_by_request.end()) {
            throw std::logic_error("request has no shared KV mapping");
        }
        for (const auto row : found->second) {
            m_slot_owners[static_cast<std::size_t>(row)].reset();
        }
        m_rows_by_request.erase(found);
    }

    PIMCommand SharedKVRowSpace::translate(std::int64_t request_id,
                                           PIMCommand command) const {
        const auto found = m_rows_by_request.find(request_id);
        if (found == m_rows_by_request.end()) {
            throw std::logic_error("request has no shared KV mapping");
        }
        // SFM and barriers are control operations; their channel-routing address
        // is not a K/V row. ST_FENCE is handled by the controller in software.
        if (command.opcode == PIMOpcode::Softmax ||
            command.opcode == PIMOpcode::Barrier ||
            command.opcode == PIMOpcode::StoreFence) {
            return command;
        }
        auto coords = m_encoder.decode(command.address);
        const auto slice = coords.row / m_rows_per_slice; // slice 총 갯수
        const auto local_row = coords.row % m_rows_per_slice;
        if (slice >= m_slice_count || local_row >= found->second.size()) {
            throw std::out_of_range("request has no chunk for its logical KV row");
        }
        coords.row = slice * m_rows_per_slice + found->second[static_cast<std::size_t>(local_row)]; 
        return {command.opcode, m_encoder.encode(coords)};
    }

    std::uint64_t SharedKVRowSpace::row_offset(std::int64_t request_id) const {
        return physical_row(request_id,0);
    }
    
    std::uint64_t SharedKVRowSpace::physical_row(
        std::int64_t request_id, std::uint64_t logical_row) const {
        const auto found = m_rows_by_request.find(request_id);
        if (found == m_rows_by_request.end()) {
            throw std::logic_error("request has no shared KV mapping");
        }
        if(logical_row >= found->second.size()) {
            throw std::out_of_range("request has no physical row for logical row");
        }
        return found->second[static_cast<std::size_t>(logical_row)];
    }

    std::uint64_t SharedKVRowSpace::rows_allocated(std::int64_t request_id) const {
        const auto found = m_rows_by_request.find(request_id);
        if (found == m_rows_by_request.end()) {
            throw std::logic_error("request has no shared KV mapping");
        }
        return found->second.size();
    }

    std::uint64_t SharedKVRowSpace::tokens_per_row_stripe() const noexcept {
        return m_tokens_per_stripe;
    }

    std::uint64_t SharedKVRowSpace::chunks_per_row_stripe() const noexcept {
        // 요청의 논리 row를 하나 늘릴 때에 모든 slice에 물리 row stripe를  하나씩 할당한다.
        return m_slice_count; // 하나의 모델에 대한 slice 수
    }

    ChannelPhaseGuard::ChannelPhaseGuard(std::size_t channel_count)
        : m_owners(channel_count) {
        if (channel_count == 0) {
            throw std::invalid_argument("channel guard needs physical channels");
        }
    }
// channels : 내가 할당 받으려는 channel list
    bool ChannelPhaseGuard::try_acquire(std::int64_t request_id,
                                         const std::vector<int>& channels) {
        if(request_id < 0 || channels.empty()) {
            throw std::invalid_argument("invalid channel phase lease");
        }
        std::unordered_set<int> unique;
        for(int channel : channels) {
            if (channel < 0 || static_cast<std::size_t>(channel) >= m_owners.size() ||
                !unique.insert(channel).second) {
                throw std::invalid_argument("invalid or duplicate channel in lease");
            }
            if(m_owners[static_cast<std::size_t>(channel)] == request_id) {
                throw std::logic_error("request already holds a channel phase lease");
            }
            // 해당 channel이 이미 다른 요청에 의해 점유되어 있으면 false 반환
            if (m_owners[static_cast<std::size_t>(channel)]) {
                return false;
            }
        }
        // 모든 channel을 성공적으로 할당
        for (int channel : channels) {
            m_owners[static_cast<std::size_t>(channel)] = request_id;
        }
        return true;
    }

    void ChannelPhaseGuard::release(std::int64_t request_id,
                                     const std::vector<int>& channels) {
        for(int channel : channels) {
            if(channel < 0 || static_cast<std::size_t>(channel) >= m_owners.size() ||
                m_owners[static_cast<std::size_t>(channel)] != request_id) {
                throw std::logic_error("request does not hold its channel phase lease");
            }
        }
        for (int channel : channels) {
            m_owners[static_cast<std::size_t>(channel)].reset();
        }
    }
} // namespace Ramulator::TierRuntime