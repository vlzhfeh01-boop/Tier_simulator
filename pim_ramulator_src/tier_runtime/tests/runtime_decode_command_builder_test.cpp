#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "runtime_docode_command_builder.h"
#include "tier_channel_scheduler.h"

using namespace Ramulator::TierRuntime;

int main() {
    const KVBlockShape shape({16, 2, 2, 128, 2});
    const RowAlignedKVLayout layout(shape, {2, 2, 4, 4, 8, 32, 32});
    const KVCapacityModel capacity(layout);
    const OrderedBlockMapper mapper(capacity);
    TierChannelRuntime runtime(capacity,mapper,16);
    FirstFreeChannelPolicy policy;
    TierChannelScheduler scheduler(runtime, policy);
    const PIMCommandGenerator generator(layout, mapper, 16);
    const AttAccAddressEncoder encoder(layout, 16);

    // Leave channel 2 as the first free channel, without hard-coding it in A.
    RequestContext blocker0;
    blocker0.spec.request_id = 10;
    blocker0.spec.actual_output_tokens = 1;
    RequestContext blocker1;
    blocker1.spec.request_id = 11;
    blocker1.spec.actual_output_tokens = 1;
    assert(scheduler.try_admit(blocker0, 0));
    assert(scheduler.try_admit(blocker1, 0));

    RequestContext a;
    a.spec.request_id = 12;
    a.spec.prompt_tokens = 33;
    a.spec.predicted_output_tokens = 1;
    a.spec.actual_output_tokens = 1;
    assert(scheduler.try_admit(a, 0));
    const auto plan = scheduler.try_begin_token(a);
    assert(plan && plan->token_index == 33);
    assert(plan->store_channel_id == 2);

    const auto batch = build_runtime_decode_commands(
        a, *plan, 0, 0, generator, encoder);
    assert(batch.request_id == 12 && batch.token_index == 33);
    assert((batch.active_channel_ids == std::vector<int>{2}));
    assert(!batch.requires_global_normalization && !batch.requires_cross_channel_merge);
    assert(!batch.commands.empty());
    const auto fence = std::find_if(batch.commands.begin(), batch.commands.end(),
        [](const PIMCommand& command) {
            return command.opcode == PIMOpcode::StoreFence;
        });
    assert(fence != batch.commands.end());
    assert(encoder.decode(fence->address).channel_id == 2);
    assert(a.generated_tokens == 0 && a.token_in_progress);

    // We only built a command list. complete_token() requires actual completion.
    std::cout << "PASS: selected channel 2 feeds append/fence/QK/SFM/SV; "
              << "completion remains pending\n";
}