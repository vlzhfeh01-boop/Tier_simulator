#include "runtime_docode_command_builder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace Ramulator::TierRuntime{
    RuntimeDeocdeCommandBatch build_runtime_decode_commands(
        const RequestContext& request,
        const DecodeTokenPlan& plan,
        std::uint64_t layer_index,
        std::uint64_t kv_head_index,
        const PIMCommandGenerator& generator,
        const AttAccAddressEncoder& encoder
    ) {
        // A plan is a snapshot. It must still describe the request that owns it.
        const auto expected_token_index = 
                static_cast<std::uint64_t>(request.spec.prompt_tokens) + request.generated_tokens;
        if( request.state != RequestState::Running || !request.token_in_progress || 
            request.spec.request_id < 0 || plan.token_index != expected_token_index ||
            plan.token_index == std::numeric_limits<std::uint64_t>::max() ||
            plan.valid_token_count != plan.token_index + 1 ||
            plan.allocated_channel_ids != request.allocated_channels ||
            plan.initial_channel_count != request.initial_channel_count ||
            std::find(plan.allocated_channel_ids.begin(),
                      plan.allocated_channel_ids.end(), plan.store_channel_id) ==
                 plan.allocated_channel_ids.end()) {
            throw std::invalid_argument("decode plan no longer matches the running request");
        }

        const auto stores = generator.generate_kv_append(KVAppendCommandInput{
            layer_index,kv_head_index,plan.token_index,
            plan.allocated_channel_ids, plan.initial_channel_count
        });
        if(stores.channel_id != plan.store_channel_id) {
            throw std::logic_error("KV append channel differs from runtime placement");
        }

        const auto qk = generator.generate_qk(QKCommandInput{
            layer_index, kv_head_index, plan.token_index,
            plan.allocated_channel_ids, plan.initial_channel_count
        });
        const auto sfm = generator.generate_softmax(qk);
        const auto sv = generator.generate_sv(SVCommandInput{
            layer_index,kv_head_index,plan.valid_token_count,
            plan.allocated_channel_ids,plan.initial_channel_count
        });
        if(qk.active_channel_ids != sv.active_channel_ids) {
            throw std::logic_error("QK and SV use different active channels");
        }

        RuntimeDeocdeCommandBatch batch;
        batch.request_id = request.spec.request_id;
        batch.token_index = plan.token_index;
        batch.active_channel_ids = qk.active_channel_ids;
        batch.requires_global_normalization = qk.active_channel_ids.size() > 1;
        batch.requires_cross_channel_merge = sv.active_channel_ids.size() > 1;
        batch.commands.reserve(stores.commands.size() + 1 + qk.commands.size() + 
                                sfm.size() + sv.commands.size());
        const auto append = [&batch](const std::vector<PIMCommand>& part) {
            batch.commands.insert(batch.commands.end(), part.begin(), part.end());
        };
        append(stores.commands);
        AttAccAddressCoordinates fence_coordinates;
        fence_coordinates.channel_id = stores.channel_id;
        batch.commands.push_back({PIMOpcode::StoreFence, encoder.encode(fence_coordinates)});
        append(qk.commands);
        append(sfm);
        append(sv.commands);
        return batch;
    }
} // namespace Ramulator::TierRuntime