#include "shared_channel_space.h"

#include <stdexcept>
#include <unordered_set>

namespace Ramulator::TierRuntime{
    SharedKVRowSpace::SharedKVRowSpace(const RowAlignedKVLayout& layout,
                                       const AttAccAddressEncoder& encoder,
                                       std::uint64_t rows_per_request)
        : m_encoder(encoder),
         m_rows_per_slice(layout.rows_per_slice_capacity()),
         m_rows_per_request(rows_per_request),
         m_slice_count(layout.slice_count()) {
        if(rows_per_request == 0 || rows_per_request > m_rows_per_slice || 
           m_rows_per_slice > layout.geometry().rows_per_bank / m_slice_count) {
            throw std::invalid_argument("invalid shared KV row window");
        }
        m_slot_owners.resize(static_cast<std::size_t>(m_rows_per_slice / rows_per_request));
    }

    bool SharedKVRowSpace::reserve(std::int64_t request_id) {
        if(request_id < 0 || m_slot_by_request.count(request_id)) {
            throw std::invalid_argument("invalid or repeated shared KV request ID");
        }
        for(std::size_t slot = 0; slot < m_slot_owners.size(); ++slot) {
            if(!m_slot_owners[slot]) {
                m_slot_owners[slot] = request_id;
                m_slot_by_request.emplace(request_id,slot);
                return true;
            }
        }
        return false;
    }

    void SharedKVRowSpace::release(std::int64_t request_id) {
        const auto slot = m_slot_by_request.find(request_id);
        if(slot == m_slot_by_request.end()) {
            throw std::logic_error("request has no shared KV row window");
        }
        m_slot_owners[static_cast<std::size_t>(slot->second)].reset();
        m_slot_by_request.erase(slot);
    }

    PIMCommand SharedKVRowSpace::translate(std::int64_t request_id,
                                           PIMCommand command) const {
        const auto slot = m_slot_by_request.find(request_id);
        if (slot == m_slot_by_request.end()) {
            throw std::logic_error("request has no shared KV row window");
        }
        // SFM and barriers are control operations; their channel-routing address
        // is not a K/V row. ST_FENCE is handled by the controller in software.
        if(command.opcode == PIMOpcode::Softmax || 
           command.opcode == PIMOpcode::Barrier ||
           command.opcode == PIMOpcode::StoreFence) {
            return command;
        }
        auto coords = m_encoder.decode(command.address);
        const auto slice = coords.row / m_rows_per_slice;
        const auto local_row = coords.row % m_rows_per_slice;
        if(slice >= m_slice_count || local_row >= m_rows_per_request) {
            throw std::out_of_range("request exceeded its shared KV row window");
        }
        // slot * rows_per_request <= rows_per_slice, so no slice can overlap
        coords.row += slot->second * m_rows_per_request;
        return {command.opcode, m_encoder.encode(coords)};
    }

    std::uint64_t SharedKVRowSpace::row_offset(std::int64_t request_id) const {
        const auto slot = m_slot_by_request.find(request_id);
        if (slot == m_slot_by_request.end()) {
            throw std::logic_error("request has no shared KV row window");
        }
        return slot->second * m_rows_per_request; // channel * rows_per_request
    }

    std::uint64_t SharedKVRowSpace::rows_per_request()const noexcept {
        return m_rows_per_request;
    }

    ChannelPhaseGuard::ChannelPhaseGuard(std::size_t channel_count)
        : m_owners(channel_count){
        if (channel_count == 0) {
            throw std::invalid_argument("channel guard needs physical channels");
        }
    }

    bool ChannelPhaseGuard::try_acquire(std::int64_t request_id,
                                        const std::vector<int>& channels) {
        if(request_id <0 || channels.empty()) {
            throw std::invalid_argument("invalid channel phase lease");
        }
        std::unordered_set<int> unique;
        for (int channel : channels) {
            if(channel < 0 || static_cast<std::size_t>(channel) >= m_owners.size() || 
               !unique.insert(channel).second) {
                throw std::invalid_argument("invalid or duplicate channel in lease");
            }
            if(m_owners[static_cast<std::size_t>(channel)] == request_id) {
                throw std::logic_error("request already holds a channel phase lease");
            }
            if(m_owners[static_cast<std::size_t>(channel)]) {
                return false;
            }
        }
        for(int channel : channels) {
            // 지금 해당 채널에서 A의 QK또는 SV 실행구간을 처리중이다 라는 뜻을 갖는다.
            m_owners[static_cast<std::size_t>(channel)] = request_id;
        }
        return true;
    }

    void ChannelPhaseGuard::release(std::int64_t request_id,
                                    const std::vector<int>& channels) {
        for(int channel:channels) {
            if (channel < 0 || static_cast<std::size_t>(channel) >= m_owners.size() ||
                m_owners[static_cast<std::size_t>(channel)] != request_id) {
                throw std::logic_error("request does not hold its channel phase lease");
            }
        }
        for (int channel : channels) {
            m_owners[static_cast<std::size_t>(channel)].reset();
        }
    }
}