#include <exception>
#include <iostream>

#include "attacc_trace_writer.h"
#include "pim_command_generator.h"

int main(int argc , char** argv) {
    using namespace Ramulator::TierRuntime;

    try {
        const KVBlockShape shape({16, 2, 2, 128, 2});
        const RowAlignedKVLayout layout(shape, {2, 2, 4, 4, 16384, 32, 32});
        const KVCapacityModel capacity(layout);
        const OrderedBlockMapper mapper(capacity);
        const PIMCommandGenerator generator(layout, mapper, 16);

        const auto qk = generator.generate_qk({0,0,33,{2,5},2});
        const auto softmax = generator.generate_softmax(qk);

        auto commands = qk.commands;
        commands.insert(commands.end(), softmax.begin(),softmax.end());

        const auto path = argc > 1 ? argv[1] : "qk_softmax_33_tokens.trace";
        AttAccTraceWriter::write_file(path,commands);
        std::cout << qk.commands.size() << " QK requests + "
                  << softmax.size() << " PIM_SFM requests = "
                  << commands.size() << " trace entries -> " << path << '\n';
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}