#include "attacc_trace_writer.h"

#include <charconv> // 문자열과 숫자간의 상호변환을 고성능으로 처리하는 헤더.
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

namespace Ramulator::TierRuntime{
    namespace{
        const char* opcode_name(PIMOpcode opcode) {
            switch(opcode){
                case PIMOpcode::WriteGemvBuffer: return "PIM_WR_GB";
                case PIMOpcode::MacAllBank: return "PIM_MAC_AB";
                case PIMOpcode::MoveSoftmaxBuffer: return "PIM_MV_SB";
                case PIMOpcode::Barrier: return "PIM_BARRIER";
                case PIMOpcode::MemoryWrite: return "ST";
                case PIMOpcode::StoreFence: return "ST_FENCE";
                case PIMOpcode::Softmax: return "PIM_SFM";
                case PIMOpcode::MoveGemvBuffer: return "PIM_MV_GB";
            }
            throw std::invalid_argument("unsupported PIM opcode");
        }

        void validate_commands(const std::vector<PIMCommand>& commands) {
            for(const auto& command : commands) {
                (void)opcode_name(command.opcode);
                if(command.address > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                    throw std::out_of_range("trace address exceeds AttAcc's signed address range");
                }
            }
        }
    }    // namespace

    void AttAccTraceWriter::write(
        std::ostream& output, const std::vector<PIMCommand>& commands
    ) {
        validate_commands(commands);
        for(const auto& command : commands) {
            char digits[16];
            const auto converted = std::to_chars(digits, digits+sizeof(digits), command.address, 16);

            const auto name = opcode_name(command.opcode);
            // 저수준 바이트 쓰기 메서드 사용.
            // write는 메모리 블록을 그대로 스트림 버퍼로 복사한다.
            output.write(name, static_cast<std::streamsize>(std::char_traits<char>::length(name)));
            output.write(" 0x",3);
            output.write(digits,static_cast<std::streamsize>(converted.ptr - digits));
            output.put('\n');
        }
        if(!output) {
            throw std::runtime_error("failed to write AttAcc trace");
        }
    }

    void AttAccTraceWriter::write_file(
        const std::filesystem::path& path,
        const std::vector<PIMCommand>& commands
    ) {
        validate_commands(commands);
        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if(!output.is_open()) {
            throw std::runtime_error("cannot open trace file: " + path.string());
        }
        write(output,commands);
        output.close();
        if(!output) {
            throw std::runtime_error("failed to close trace file: " + path.string());
        }
    }
}    // namespace Ramulator::TierRuntime