#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

#include "decode_step_driver.h"

using namespace Ramulator::TierRuntime;

struct DelayedIssue {
    std::uint64_t when;
    PIMOpcode opcode;
    DecodeStepDriver::IssueCallback callback;
};

int main() {
    const KVBlockShape shape({16,1,1,128,2});
    const RowAlignedKVLayout layout(shape,{2,2,4,4,16384,32,32});
    const KVCapacityModel capacity(layout);
    const OrderedBlockMapper mapper(capacity);
    const PIMCommandGenerator generator(layout,mapper,16);
    const AttAccAddressEncoder encoder(layout,16);

    RuntimeDecodeLayerPlan plan;
    plan.request_id = 1;
    plan.token_index = 33;
    plan.valid_token_count = 34;
    plan.initial_channel_count = 2;
    plan.query_head_count = 1;
    plan.kv_head_count = 1;
    plan.store_channel_id = 2;
    plan.allocated_channel_ids = {2,5};
    plan.kv_append_and_fence = generator.generate_kv_append(
        KVAppendCommandInput{0,0,33,{2,5},2}).commands;
    AttAccAddressCoordinates fence{};
    fence.channel_id = 2;
    plan.kv_append_and_fence.push_back(
        {PIMOpcode::StoreFence, encoder.encode(fence)}
    );

    std::optional<std::uint64_t> qk_end;
    std::optional<std::uint64_t> sfm_ready;
    std::optional<std::uint64_t> sv_end;
    std::uint64_t sfm_issue;
    std::uint64_t sfm_send = 0;
    std::uint64_t first_sv_send = 0;
    int sfm_count = 0;
    int rejected_count = 0;
    std::vector<DelayedIssue> pending;
    DecodePhase last_completed = DecodePhase::Done;
    DecodeStepDriver driver(std::move(plan), generator, encoder,4,
        [&](DecodePhase phase, std::uint64_t head, std::uint64_t cycle) {
            assert(head==0);
            last_completed = phase;
            if(phase == DecodePhase::QK) qk_end = cycle;
            if(phase == DecodePhase::Softmax) sfm_ready = cycle;
            if(phase == DecodePhase::SV) sv_end = cycle;
        });
    
    for(std::uint64_t cycle = 1; cycle < 2000 && !driver.done(); ++cycle) {
        for(auto it = pending.begin(); it != pending.end();) {
            if(it->when <= cycle) {
                if(it->opcode == PIMOpcode::Softmax) sfm_issue = it->when;
                it->callback(it->when); // 해당 명령이 이 타이밍에 발행됐다는 통지이다.
                it = pending.erase(it);
            } else {
                ++it;
            }
        }
        driver.tick(cycle, [&](const PIMCommand& cmd, 
                                DecodeStepDriver::IssueCallback cb) {
            if(last_completed == DecodePhase::KVAppendFence && 
                rejected_count == 0) {
                ++rejected_count;
                return false;
            }
            if(cmd.opcode == PIMOpcode::Softmax) {
                ++sfm_count;
                sfm_send = cycle;
            } else if (last_completed == DecodePhase::Softmax && 
                        first_sv_send == 0) {
                first_sv_send = cycle;
            }
            if(cb) {
                const auto channel = encoder.decode(cmd.address).channel_id;
                pending.push_back({cycle + (channel == 5? 3U : 2U),
                                    cmd.opcode, std::move(cb)});
            }
            return true;
        });
    }

    assert(driver.done() && qk_end && sfm_ready && sv_end);
    assert(rejected_count == 1 && sfm_count == 1 && pending.empty());
    assert(sfm_send > *qk_end);
    assert(first_sv_send >= sfm_issue + 4);
    assert(*sfm_ready == sfm_issue + 4);
    assert(driver.last_sv_issue_cycle() == *sv_end);
    std::cout << "PASS: real two-channel commands, retry, QK/SFM/SV callbacks\n";
}