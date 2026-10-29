#include <exception>
#include <iostream>
#include <vector>

#include "attacc_trace_writer.h"
#include "pim_command_generator.h"

int main(int argc, char** argv) {
    using namespace Ramulator::TierRuntime;

    try{
        const KVBlockShape shape({16,2,2,128,2});
        const RowAlignedKVLayout layout(shape, {2, 2, 4, 4, 16384, 32, 32});
        const KVCapacityModel capacity(layout);
        const OrderedBlockMapper mapper(capacity);
        const PIMCommandGenerator generator(layout, mapper, 16);
        const AttAccAddressEncoder encoder(layout, 16);

        // KV tokens [0,33) already exist. Append token 33 and attend to [0,34) 
        const KVAppendCommandInput append_input{0,0,33,{2,5},2};
        const auto stores = generator.generate_kv_append(append_input);
        const QKCommandInput attention_input{
            append_input.layer_index, append_input.kv_head_index,
            append_input.existing_token_count+1,
            append_input.allocated_channel_ids, append_input.initial_channel_count
        };
        const auto qk = generator.generate_qk(attention_input);
        const auto softmax = generator.generate_softmax(qk);
        const auto sv = generator.generate_sv(SVCommandInput{
            attention_input.layer_index, attention_input.kv_head_index,
            attention_input.valid_token_count,
            attention_input.allocated_channel_ids, attention_input.initial_channel_count
        });

        std::vector<PIMCommand> commands = stores.commands;
        // Only channel 2 receives stores. The fence protects its dependent PIM work.
        AttAccAddressCoordinates fence_address;
        fence_address.channel_id = stores.channel_id;
        commands.push_back({PIMOpcode::StoreFence, encoder.encode(fence_address)});
        commands.insert(commands.end(), qk.commands.begin(), qk.commands.end());
        commands.insert(commands.end(), softmax.begin(), softmax.end());
        commands.insert(commands.end(), sv.commands.begin(), sv.commands.end());

        const auto path = argc > 1 ? argv[1] : "decode_attention_34_tokens.trace";
        AttAccTraceWriter::write_file(path, commands);
        std::cout << stores.commands.size() << " ST + 1 ST_FENCE + "
                  << qk.commands.size() << " QK + " << softmax.size()
                  << " SFM + " << sv.commands.size() << " SV = "
                  << commands.size() << " requests -> " << path << '\n';

        } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}