#include "qk_phase_submitter.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

using namespace Ramulator::TierRuntime;

int main() {
    const auto channel = [](std::uint64_t a) {
        return static_cast<int>((a >> 30) & 0xfU);
    };
    std::vector<PIMCommand> commands{
        {PIMOpcode::MacAllBank, 2ULL << 30},
        {PIMOpcode::MoveSoftmaxBuffer, 2ULL << 30},
        {PIMOpcode::MacAllBank, 5ULL << 30},
        {PIMOpcode::MoveSoftmaxBuffer, 5ULL << 30},
        {PIMOpcode::Barrier, 2ULL << 30},
        {PIMOpcode::Barrier, 5ULL << 30}
    };
    QKPhaseSubmitter submitter(commands, {2, 5}, channel);
    std::vector<QKPhaseSubmitter::IssueCallback> callbacks; // function interface

    bool reject_once = true;
    auto send = [&](const PIMCommand& command, QKPhaseSubmitter::IssueCallback cb) {
        if (reject_once && callbacks.size() == 2) {
            reject_once = false;
            return false;
        }
        assert(command.address == commands[callbacks.size()].address);
        callbacks.push_back(std::move(cb));
        return true;
    };

    submitter.tick(send);
    assert(submitter.accepted_count() == 2 && !submitter.ready());
    submitter.tick(send);
    assert(submitter.accepted_count() == commands.size() && !submitter.ready());
    // Channel 5's score move issues before channel 2's final barrier.
    callbacks[3](40);
    callbacks[0](41);
    callbacks[1](42);
    callbacks[2](43);
    callbacks[5](47);
    submitter.tick(send);
    assert(!submitter.ready());
    callbacks[4](46);
    assert(!submitter.ready()); // Controller callback only enqueues an event.
    submitter.tick(send);
    assert(submitter.ready() && submitter.ready_cycle() == 47);
    std::cout << "PASS: backpressure, deferred callback, all-channel issue gate\n";
}