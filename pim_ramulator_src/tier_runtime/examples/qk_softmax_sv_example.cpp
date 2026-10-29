#include <exception>
#include <iostream>

#include "attacc_trace_writer.h"
#include "pim_command_generator.h"

int main(int argc, char** argv) {
    using namespace Ramulator::TierRuntime;

    try {
        const KVBlockShape shape({16, 2, 2, 128, 2});
        const RowAlignedKVLayout layout(shape, {2, 2, 4, 4, 16384, 32, 32});
        const KVCapacityModel capacity(layout);
        const OrderedBlockMapper mapper(capacity);
        const PIMCommandGenerator generator(layout, mapper, 16);


        // Assume KV tokens [0, 34) have already been stored. 
        const QKCommandInput input{0,0,34,{2,5},2};
        const auto qk = generator.generate_qk(input);
        const auto softmax = generator.generate_softmax(qk);
        const auto sv = generator.generate_sv(SVCommandInput{
            input.layer_index, input.kv_head_index, input.valid_token_count,
            input.allocated_channel_ids, input.initial_channel_count
        });

        auto commands = qk.commands;
        commands.insert(commands.end(), softmax.begin(),softmax.end());
        commands.insert(commands.end(), sv.commands.begin(), sv.commands.end());
        const auto path = argc > 1 ? argv[1] : "attention_34_tokens.trace";
        AttAccTraceWriter::write_file(path, commands);
        std::cout << qk.commands.size() << " QK + " << softmax.size()
                  << " SFM + " << sv.commands.size() << " SV = "
                  << commands.size() << " trace entries -> " << path << '\n';
        std::cout << "SV: " << sv.active_channel_ids.size()
                  << " active channels contribute partial outputs\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}