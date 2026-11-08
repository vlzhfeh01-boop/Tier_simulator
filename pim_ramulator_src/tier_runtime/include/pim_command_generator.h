/* 
    한 layer, head의 query 하나에 대해, 현재 저장된 모든 유효 K token의 
    QK연산 명령을 생성한다.
*/
#pragma once

#include <cstdint>
#include <vector>

#include "attacc_address_encoder.h"
#include "pim_command.h"
#include "sv_command.h"

namespace Ramulator::TierRuntime {
    struct QKCommandInput
    {
        std::uint64_t layer_index = 0;
        std::uint64_t kv_head_index = 0; // 그 layer에서 어느 head의 K를 읽을지 결정
        std::uint64_t valid_token_count = 0; // 실제 연산할 token 범위 결정
        std::vector<int> allocated_channel_ids;
        std::uint64_t initial_channel_count = 0;
    };
    // 마지막 block의 유효범위를 알기 위해서 다음 구조 추가.
    struct QKBlockWork {
        std::uint64_t request_block_index = 0; // 요청 block 식별
        std::uint64_t valid_token_count = 0;   // 부분 block의 유효 범위
        PIMMACLocation first_mac_location;  // dimension step 0의 channel, row, column을 기록
    };
    struct QKCommandBatch {
        std::vector<PIMCommand> commands; // 무엇을 어떤 순서로 실행했는가?
        std::vector<QKBlockWork> blocks; // 어떤 block을 처리했고, 유효범위는 얼마인가?
        std::vector<int> active_channel_ids; 
        std::uint64_t padded_dimensions_per_token = 0; // 물리 연산 범위에 추가된 padding은 얼마인가?
    };

    // Append one token's K and V for one layer/KV head.
    struct KVAppendCommandInput {
        // 어느 슬라이스에 저장할지 결정
        std::uint64_t layer_index = 0;
        std::uint64_t kv_head_index = 0;
        // 기존길이가 곧 index가 된다.
        std::uint64_t existing_token_count = 0;
        std::vector<int> allocated_channel_ids; 
        std::uint64_t initial_channel_count = 0;
    };
    struct KVAppendCommandBatch {
        std::vector<PIMCommand> commands;
        std::uint64_t token_index = 0;
        std::uint64_t request_block_index = 0;
        int channel_id = -1;
        std::uint64_t key_write_count = 0;
        std::uint64_t value_write_count = 0;
    };
    
    class PIMCommandGenerator {
    public:
        // The layout, mapper and encoder are bound to the same configuration. 
        // Initial support : bank-level FP16 HBM3 with 2 pCH, 2 ranks, 4 BG,
        // 4 banks/BG, 32 columns, and a 32-byte transaction.
        PIMCommandGenerator(
            const RowAlignedKVLayout &layout, // 주소 계산과 인코딩이 동일한 geometry를 기준으로 동작한다.
            const OrderedBlockMapper& block_mapper,
            std::uint64_t total_channel_count
        );

        [[nodiscard]]
        QKCommandBatch generate_qk(const QKCommandInput& input) const;

        // qk_batch must be an unmodified result from this generator's generate_qk().
        // One head-wide SFM, routed through the first active channel's address.
        // The caller enforces the all-channel QK boundary; no latency model here.
        [[nodiscard]]
        std::vector<PIMCommand> generate_softmax(const QKCommandBatch& qk_batch) const;

        // Softmax probabilites for this head/length must be readt first.
        // Generate SV traffic and channel partial outputs, not numerical values.
        // or the final cross-channel reduction/its completion time.
        [[nodiscard]]
        SVCommandBatch generate_sv(const SVCommandInput& input) const;


        // Generates addresses only. Does not allocate channels, modify valid
        // length, model payload values, or wait for stores to complete.
        [[nodiscard]]
        KVAppendCommandBatch generate_kv_append(const KVAppendCommandInput& input) const;
        
    private:

        void validate_allocation(
            const std::vector<int>& channel_ids,
            std::uint64_t initial_channel_count
        )const;

        std::uint64_t encode_query_buffer(
            int channel_id, std::uint64_t bank, std::uint64_t slot,
            std::uint64_t key_slice_base_row
        ) const;

        std::uint64_t encode_score_buffer(
            int channel_id, std::uint64_t rank, std::uint64_t bank_group,
            std::uint64_t key_slice_base_row
        ) const;

        std::uint64_t encode_channel(int channel_id) const;

        std::uint64_t encode_probability_buffer (
            int channel_id, std::uint64_t rank, std::uint64_t bank_group,
            std::uint64_t slot, std::uint64_t value_slice_base_row
        ) const;

        std::uint64_t encode_context_buffer(
            int channel_id, std::uint64_t rank, std::uint64_t bank,
            std::uint64_t value_slice_base_row
        ) const;

        RowAlignedKVLayout m_layout; // block 크기, slice, dimension step등의 배치 규칙
        KVAddressMapper m_address_mapper; //요청 block의 물리 위치 계산
        AttAccAddressEncoder m_encoder; // 물리 위치를 정수 주소로 변환
        std::uint64_t m_total_channel_count; // 입력 채널 ID의 범위 검사
    };
}