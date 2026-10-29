#include <cassert>
#include <iostream>

#include "pim_command_generator.h"
#include "tier_channel_runtime.h"

int main() {
    using namespace Ramulator::TierRuntime;
    // Small row count makes the growth boundary visible: 64 blocks/channel.
    const KVBlockShape shape({16, 1, 1, 128, 2});
    const RowAlignedKVLayout layout(shape, {2, 2, 4, 4, 8, 32, 32});
    const KVCapacityModel capacity(layout);
    const OrderedBlockMapper mapper(capacity);
    const PIMCommandGenerator generator(layout, mapper, 16);
    TierChannelRuntime runtime(capacity, mapper, 16);

    RequestContext request;
    request.spec.request_id = 42;
    request.spec.prompt_tokens = 2047;
    request.spec.predicted_output_tokens = 1;
    request.spec.actual_output_tokens = 2;
    assert(runtime.predicted_initial_channel_count(request.spec) == 2);
    runtime.admit(request,{2,5},0);

    const auto original_block = mapper.locate_block(0,request.allocated_channels, request.initial_channel_count);

    auto first = runtime.begin_token(request);
    assert(first && !first->grew && first->token_index == 2047);
    const auto first_stores = generator.generate_kv_append({
        0,0,first->token_index,first->allocated_channel_ids, first->initial_channel_count
    });
    assert(first_stores.channel_id == first->store_channel_id);
    std::cout << "token " << first->token_index << " -> channel "
              << first->store_channel_id << " (existing block)\n";
    runtime.complete_token(request,10); // In a real run, use a completion callback.

    assert(runtime.needs_growth(request));
    assert(!runtime.begin_token(request)); // No free ID supplied; state unchanged.
    assert(request.generated_tokens == 1 && request.allocated_channels.size() == 2);
    auto second = runtime.begin_token(request, 9); // The allocation policy chose 9.
    assert(second && second->grew && second->token_index == 2048);
    assert(second->store_channel_id == 9 && second->initial_channel_count == 2);
    const auto second_stores = generator.generate_kv_append({
        0, 0, second->token_index, second->allocated_channel_ids, second->initial_channel_count
    });
    assert(second_stores.channel_id == second->store_channel_id);
    assert(mapper.locate_block(0, request.allocated_channels,
                               request.initial_channel_count).channel_id == original_block.channel_id);
    std::cout << "token " << second->token_index << " -> channel "
              << second->store_channel_id << " (+1 growth)\n";
    runtime.complete_token(request, 20);
    assert(request.state == RequestState::Completed);
    runtime.release_completed(request);
    std::cout << "PASS: initial {2,5} remains fixed; +1 channel 9; KV writes follow placement\n";
}