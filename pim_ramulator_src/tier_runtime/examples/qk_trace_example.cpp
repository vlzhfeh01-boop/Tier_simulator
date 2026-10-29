#include <exception>
#include <iostream>

#include "attacc_trace_writer.h"
#include "pim_command_generator.h"

int main(int argc, char** argv) {
    using namespace Ramulator::TierRuntime;
    try {
        const KVBlockShape shape({16, 2, 2, 128, 2});
        const PIMGeometryConfig geometry{2, 2, 4, 4, 16384, 32, 32};
        const RowAlignedKVLayout layout(shape, geometry);
        const KVCapacityModel capacity(layout);
        const OrderedBlockMapper block_mapper(capacity);
        const PIMCommandGenerator generator(layout, block_mapper, 16);

        QKCommandInput input;
        input.layer_index = 0;
        input.kv_head_index = 0;
        input.valid_token_count = 33;
        input.allocated_channel_ids = {2, 5};
        input.initial_channel_count = 2;

        const auto batch = generator.generate_qk(input);
        const auto path = argc > 1 ? argv[1] : "qk_33_tokens.trace";
        AttAccTraceWriter::write_file(path, batch.commands);
        std::cout << batch.blocks.size() << " blocks, "
                  << batch.commands.size() << " commands -> " << path << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
