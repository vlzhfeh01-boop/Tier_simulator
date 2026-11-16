#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "decode_step_driver.h"

using namespace Ramulator::TierRuntime;

struct PendingIssue {
    std::uint64_t cycle;
    DecodeStepDriver::IssueCallback callback;
};

struct PhaseEvent {
    DecodePhase phase;
    std::uint64_t head;
    std::uint64_t cycle;
};

int main() {
    // Qwen3-8B's 4 query heads per KV head, scaled down to two layers.
    const KVBlockShape shape({16,2,2,128,2});
    const RowAlignedKVLayout layout(shape, {2,2,4,4,16384,32,32});
    // Limit block capacity only to exercise two active channels in this test.
    const KVCapacityModel capacity(layout, 2);
    const OrderedBlockMapper mapper(capacity);
    const PIMCommandGenerator generator(layout, mapper, 16);
    const AttAccAddressEncoder encoder(layout, 16);
    TierChannelRuntime runtime(capacity, mapper, 16);

    RequestContext request;
    request.spec.request_id = 7;
    request.spec.prompt_tokens = 33;
    request.spec.predicted_output_tokens = 1;
    request.spec.actual_output_tokens = 1;
    runtime.admit(request, {2, 5}, 0);
    const auto token = runtime.begin_token(request);
    assert(token && token->valid_token_count == 34);
    assert((token->allocated_channel_ids == std::vector<int>{2, 5}));
    assert(kv_head_for_query(0, 8, 2) == 0);
    assert(kv_head_for_query(3, 8, 2) == 0);
    assert(kv_head_for_query(4, 8, 2) == 1);
    assert(kv_head_for_query(7, 8, 2) == 1);

    std::uint64_t clock = 1;
    std::uint64_t sfm_count = 0;
    std::uint64_t final_sv_issue = 0;
    for (std::uint64_t layer = 0; layer < 2; ++layer) {
        auto plan = build_runtime_decode_layer_plan(
            request, *token, layer, 8, 2, generator, encoder);
        std::uint64_t writes[2][2] = {};
        std::uint64_t fences = 0;
        for (const auto& cmd : plan.kv_append_and_fence) {
            if (cmd.opcode == PIMOpcode::StoreFence) {
                ++fences;
                assert(encoder.decode(cmd.address).channel_id == token->store_channel_id);
                continue;
            }
            assert(cmd.opcode == PIMOpcode::MemoryWrite);
            const auto row = encoder.decode(cmd.address).row;
            bool found = false;
            for (std::uint64_t kv = 0; kv < 2; ++kv) {
                for (std::uint64_t kind = 0; kind < 2; ++kind) {
                    const auto base = layout.slice_base_row(
                        kind == 0 ? KVKind::Key : KVKind::Value, layer, kv);
                    if (row >= base && row - base < layout.rows_per_slice_capacity()) {
                        ++writes[kv][kind];
                        found = true;
                    }
                }
            }
            assert(found);
        }
        assert(fences == 1);
        for(const auto& per_kv : writes) {
            assert(per_kv[0] > 0 && per_kv[1] > 0);
        }
        
        std::vector<PendingIssue> pending;
        std::vector<PhaseEvent> events;
        std::optional<std::uint64_t> mac_row[8];
        std::uint64_t layer_sfm_count = 0;
        DecodeStepDriver driver(std::move(plan), generator, encoder, 0,
            [&](DecodePhase phase, std::uint64_t head, std::uint64_t cycle) {
                events.push_back({phase, head, cycle});
            });
        const auto inspect = [&](const DecodePhaseWork& work) {
            if (work.phase == DecodePhase::KVAppendFence) {
                assert(work.query_head_index == 0);
                return;
            }
            const auto head = work.query_head_index;
            assert(head < 8);
            assert(work.kv_head_index == head / 4);
            assert((work.active_channel_ids == std::vector<int>{2, 5}));
            if (work.phase == DecodePhase::Softmax) {
                assert(work.commands.size() == 1);
                assert(work.commands[0].opcode == PIMOpcode::Softmax);
            }
            if (work.phase == DecodePhase::QK || work.phase == DecodePhase::SV) {
                const auto mac = std::find_if(work.commands.begin(), work.commands.end(),
                    [](const PIMCommand& cmd) {
                        return cmd.opcode == PIMOpcode::MacAllBank;
                    });
                assert(mac != work.commands.end());
                const auto base = layout.slice_base_row(
                    work.phase == DecodePhase::QK ? KVKind::Key : KVKind::Value,
                    layer, head / 4);
                const auto row = encoder.decode(mac->address).row;
                assert(row >= base && row - base < layout.rows_per_slice_capacity());
                if (work.phase == DecodePhase::QK) mac_row[head] = row;
            }
            if (work.phase == DecodePhase::ContextMerge) {
                assert(work.commands.empty());
            }
        };
        const auto deadline = clock + 20000;
        while (!driver.done() && clock < deadline) {
            for (auto it = pending.begin(); it != pending.end();) {
                if (it->cycle <= clock) {
                    it->callback(it->cycle);
                    it = pending.erase(it);
                } else {
                    ++it;
                }
            }
            const auto& work = driver.pending_phase();
            inspect(work);
            driver.tick(clock, [&](const PIMCommand& cmd,
                                   DecodeStepDriver::IssueCallback callback) {
                if (cmd.opcode == PIMOpcode::Softmax) {
                    assert(work.phase == DecodePhase::Softmax);
                    ++sfm_count;
                    ++layer_sfm_count;
                }
                if (callback) pending.push_back({clock + 2, std::move(callback)});
                return true;
            });
            ++clock;
        }
        assert(driver.done() && pending.empty());
        assert(layer_sfm_count == 8);
        assert(events.size() == 1 + 8 * 4);
        assert(events[0].phase == DecodePhase::KVAppendFence);
        for (std::uint64_t head = 0; head < 8; ++head) {
            const auto index = 1 + 4 * head;
            assert(events[index].phase == DecodePhase::QK);
            assert(events[index + 1].phase == DecodePhase::Softmax);
            assert(events[index + 2].phase == DecodePhase::SV);
            assert(events[index + 3].phase == DecodePhase::ContextMerge);
            for (std::size_t offset = 0; offset < 4; ++offset) {
                assert(events[index + offset].head == head);
            }
            assert(events[index + 2].cycle <= events[index + 3].cycle);
            if (head > 0) {
                assert(events[index].cycle > events[index - 1].cycle);
            }
            assert(mac_row[head]);
        }
        assert(mac_row[0] == mac_row[1]);
        assert(mac_row[3] != mac_row[4]);
        final_sv_issue = driver.last_sv_issue_cycle();
        assert(request.token_in_progress && request.generated_tokens == 0);
    }
    assert(sfm_count == 16);
    runtime.complete_token(request, static_cast<Ramulator::Clk_t>(final_sv_issue + 1));
    assert(request.state == RequestState::Completed);
    assert(request.generated_tokens == 1);
    runtime.release_completed(request);
    std::cout << "PASS: two layers, 8:2 GQA, 16 head SFM commands, ordered SV and KV rows\n";
}