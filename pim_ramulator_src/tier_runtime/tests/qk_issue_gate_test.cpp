#include "qk_issue_gate.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace Ramulator::TierRuntime;

int main() {
    const auto ch = [](std::uint64_t address) {
        return static_cast<int>((address >> 30) & 0xfU);
    };
    const std::vector<PIMCommand> qk{
        {PIMOpcode::WriteGemvBuffer, 2ULL << 30},
        {PIMOpcode::MacAllBank, 5ULL << 30},
        {PIMOpcode::MoveSoftmaxBuffer, 2ULL << 30},
        {PIMOpcode::MoveSoftmaxBuffer, 5ULL << 30},
        {PIMOpcode::Barrier, 2ULL << 30},
        {PIMOpcode::Barrier, 5ULL << 30},
    };
    QKIssueGate gate(qk, {2, 5}, ch);
    assert(!gate.ready());
    gate.mark_issued(3, 41); // Channel 5 score move issues first.
    gate.mark_issued(0, 40);
    gate.mark_issued(1, 43);
    gate.mark_issued(2, 44); // Both score moves issued, other work remains.
    assert(!gate.ready());
    gate.mark_issued(5, 49);
    assert(!gate.ready());
    assert(gate.waiting_channels() == std::vector<int>{2});
    gate.mark_issued(4, 48);
    assert(gate.ready());
    assert(gate.ready_cycle() == 49);
    assert(gate.waiting_channels().empty());
    bool rejected_duplicate = false;
    try { gate.mark_issued(4, 50); }
    catch (const std::logic_error&) { rejected_duplicate = true; }
    assert(rejected_duplicate);

    bool rejected_wrong_channel = false;
    try { (void)QKIssueGate({{PIMOpcode::MoveSoftmaxBuffer, 7ULL << 30}},
                            {2}, ch); }
    catch (const std::invalid_argument&) { rejected_wrong_channel = true; }
    assert(rejected_wrong_channel);
    std::cout << "PASS: both channels and every QK command issue before SFM\n";
}
