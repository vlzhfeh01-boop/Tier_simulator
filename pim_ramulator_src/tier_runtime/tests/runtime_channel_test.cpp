#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>

#include "tier_channel_runtime.h"

using namespace Ramulator::TierRuntime;

template<class Exception> void rejects(const std::function<void()>& run) {
    try { run(); } catch (const Exception&) { return; }
    throw std::runtime_error("expected an exception");
}

int main() {
    const KVBlockShape shape({16, 1, 1, 128, 2});
    const RowAlignedKVLayout layout(shape, {2, 2, 4, 4, 8, 32, 32});
    const KVCapacityModel capacity(layout);
    const OrderedBlockMapper mapper(capacity);
    TierChannelRuntime runtime(capacity, mapper, 16);
    RequestContext a;
    a.spec.request_id = 1;
    a.spec.prompt_tokens = 2047;
    a.spec.predicted_output_tokens = 1;
    a.spec.actual_output_tokens = 2;
    assert(runtime.predicted_initial_channel_count(a.spec) == 2);
    runtime.admit(a, {2, 5}, 0);
    assert(a.initial_channel_count == 2 && !runtime.needs_growth(a));

    RequestContext b;
    b.spec.request_id = 2;
    b.spec.actual_output_tokens = 1;
    rejects<std::logic_error>([&] { runtime.admit(b, {5}, 0); }); // 잘못된 admission을 거부하는지 검사한다.
    rejects<std::invalid_argument>([&] { runtime.admit(b, {3, 3}, 0); });
    RequestContext duplicate_id;
    duplicate_id.spec.request_id = 1;
    rejects<std::invalid_argument>([&] { runtime.admit(duplicate_id, {4}, 0); });
    runtime.admit(b, {9}, 0); // 채널 9 점유상태.

    auto first = runtime.begin_token(a);
    assert(first && first->store_channel_id == 5 && !first->grew);
    rejects<std::logic_error>([&] { (void)runtime.begin_token(a); });
    rejects<std::logic_error>([&] { runtime.complete_token(a, -1); });
    runtime.complete_token(a, 10); // 첫 토큰 완료
    assert(runtime.needs_growth(a)); // 2048이 되어서 다음 토큰을 위한 needs_growth가 참이 된다.
    assert(!runtime.begin_token(a));
    assert(!runtime.begin_token(a, 9)); // Owned by request b.
    assert(a.allocated_channels.size() == 2 && !a.token_in_progress);
    rejects<std::out_of_range>([&] { (void)runtime.begin_token(a, 17); });

    auto second = runtime.begin_token(a, 7); // 채널 7로 성장
    assert(second && second->grew && second->store_channel_id == 7);
    assert(second->allocated_channel_ids == std::vector<int>({2, 5, 7}));
    assert(a.initial_channel_count == 2 && a.token_in_progress);
    assert(mapper.locate_block(127, a.allocated_channels, 2).channel_id == 5);
    assert(mapper.locate_block(128, a.allocated_channels, 2).channel_id == 7);
    a.inflight_commands = 1;
    rejects<std::logic_error>([&] { runtime.complete_token(a, 20); }); // 토큰 완료 거부
    a.inflight_commands = 0;
    runtime.complete_token(a, 20); // 생성 토큰 수가 2가 된다.
    assert(a.state == RequestState::Completed && a.generated_tokens == 2); // Completed
    runtime.release_completed(a);
    rejects<std::logic_error>([&] { runtime.release_completed(a); });
    RequestContext c;
    c.spec.request_id = 3;
    c.spec.actual_output_tokens = 1;
    runtime.admit(c, {7}, 20); // Released channel is free again.
    assert(c.initial_channel_count == 1);

    TierChannelRuntime predicted_runtime(capacity, mapper, 16);
    RequestContext predicted;
    predicted.spec.request_id = 4;
    predicted.spec.prompt_tokens = 2047;
    predicted.spec.predicted_output_tokens = 2; // Total 2049 needs three channels.
    predicted.spec.actual_output_tokens = 2;
    assert(predicted_runtime.predicted_initial_channel_count(predicted.spec) == 3);
    rejects<std::out_of_range>([&] { predicted_runtime.admit(predicted, {2, 5}, 0); });
    assert(predicted.state == RequestState::Waiting && predicted.allocated_channels.empty());
    predicted_runtime.admit(predicted, {2, 5, 9}, 0);
    assert(predicted.initial_channel_count == 3 && !predicted_runtime.needs_growth(predicted));
    std::cout << "PASS: admission isolation, growth boundary, fixed initial group, "
                 "predicted capacity, no mutation while blocked, completion and reuse\n";
}
