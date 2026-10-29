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
        const AttAccAddressEncoder encoder(layout, 16);

        const KVAppendCommandInput append_input{0, 0, 33, {2, 5}, 2};
        const auto store = generator.generate_kv_append(append_input);
        const QKCommandInput qk_input{0, 0, 34, {2, 5}, 2};
        const auto qk = generator.generate_qk(qk_input);

        auto commands = store.commands;
        AttAccAddressCoordinates fence_location;
        fence_location.channel_id = store.channel_id;
        commands.push_back(
            {PIMOpcode::StoreFence, encoder.encode(fence_location)}
        );
        commands.insert(commands.end(), qk.commands.begin(), qk.commands.end());

        const auto path = argc > 1 ? argv[1] : "kv_append_qk.trace";
        AttAccTraceWriter::write_file(path, commands);
        std::cout << store.commands.size() << " stores + 1 ST_FENCE + "
                  << qk.commands.size() << " QK requests = "
                  << commands.size() << " trace entries -> " << path << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}