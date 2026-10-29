#include <cassert>
#include <iostream>

#include "tier_channel_scheduler.h"

int main() {
    using namespace Ramulator::TierRuntime;
    const KVBlockShape shape({16,1,1,128,2});
    const RowAlignedKVLayout layout(shape,{2,2,4,4,8,32,32});
    const KVCapacityModel capacity(layout);
    const OrderedBlockMapper mapper(capacity);
    TierChannelRuntime runtime(capacity,mapper,3);
    FirstFreeChannelPolicy policy;
    TierChannelScheduler scheduler(runtime,policy);

    RequestContext a;
    a.spec.request_id = 1;
    a.spec.prompt_tokens = 2047;
    a.spec.predicted_output_tokens = 1;
    a.spec.actual_output_tokens = 2;
    const bool admitted_a = scheduler.try_admit(a,0);
    assert(admitted_a); // Automatic initial allocation: {0,1}.

    RequestContext b;
    b.spec.request_id = 2;
    b.spec.actual_output_tokens = 1;
    const bool admitted_b = scheduler.try_admit(b,0);
    assert(admitted_b); // Automatic allocation : {2}.

    auto first = scheduler.try_begin_token(a);
    assert(first && first->token_index == 2047 && first->store_channel_id ==1);
}