#include "qk_phase_submitter.h"
#include "softmax_phase_submitter.h"
#include "softmax_performance_model.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace Ramulator::TierRuntime;

int main() {
    const auto channel = [](std::uint64_t address) {
        return static_cast<int>((address>>30) & 0xfU);
    };
    const std::vector<PIMCommand> qk_commands{
        {PIMOpcode::MacAllBank, 2ULL << 30},
        {PIMOpcode::MoveSoftmaxBuffer, 2ULL << 30},
        {PIMOpcode::MacAllBank, 5ULL << 30},
        {PIMOpcode::MoveSoftmaxBuffer, 5ULL << 30},
        {PIMOpcode::Barrier, 2ULL << 30},
        {PIMOpcode::Barrier, 5ULL << 30}
    };
    QKPhaseSubmitter qk(qk_commands, {2,5}, channel);
    std::vector<QKPhaseSubmitter::IssueCallback> qk_callbacks;
    qk.tick([&](const PIMCommand&, QKPhaseSubmitter::IssueCallback cb){
        qk_callbacks.push_back(std::move(cb));
        return true;
    });
    for(std::size_t i = 0; i<qk_callbacks.size(); ++i) {
        qk_callbacks[i](40+static_cast<std::uint64_t>(i));
    }
    assert(!qk.ready());
    qk.tick([&](const PIMCommand&, QKPhaseSubmitter::IssueCallback) {
        assert(false && "QK command was submitted twice");
        return false;
    });
    assert(qk.ready() && qk.ready_cycle() == 45);

    SoftmaxPhaseSubmitter trace_sfm({PIMOpcode::Softmax,2ULL<<30},qk.ready_cycle());
    SoftmaxPhaseSubmitter::IssueCallback trace_callback;
    trace_sfm.tick(48, [&](const PIMCommand&, SoftmaxPhaseSubmitter::IssueCallback cb) {
        trace_callback = std::move(cb);
        return true;
    });
    trace_callback(50);
    assert(!trace_sfm.ready(50)); // Callback remains deferred.
    trace_sfm.tick(51, [&](const PIMCommand&, SoftmaxPhaseSubmitter::IssueCallback) {
        assert(false && "SFM command was submitted twice");
        return false;
    });
    assert(trace_sfm.ready(51) && trace_sfm.sv_eligible_cycle() == 50);

    const SoftmaxPerformanceConfig attacc_config{
        670400000000.0L, 670400000000.0L, 769
    };
    assert(estimate_softmax_cycles(34, attacc_config) == 1);
    assert(estimate_softmax_cycles(128, attacc_config) == 2);
    assert(estimate_softmax_cycles(4096, attacc_config) == 40);

    // Optional analytic model: artificial rates produce exactly 10 cycles.
    const SoftmaxPerformanceConfig performance{100.0e9L, 100.0e9L, 1000};
    const auto modeled_delay = estimate_softmax_cycles(200, performance);
    assert(modeled_delay == 10);
    SoftmaxPhaseSubmitter sfm({PIMOpcode::Softmax, 2ULL << 30},
                               qk.ready_cycle(), modeled_delay);
    SoftmaxPhaseSubmitter::IssueCallback sfm_callback;
    int sends = 0;
    auto send_sfm = [&](const PIMCommand& command,
                        SoftmaxPhaseSubmitter::IssueCallback cb) {
        ++sends;
        assert(command.opcode == PIMOpcode::Softmax);
        if (sends == 1) return false; // Controller queue is full.
        sfm_callback = std::move(cb);
        return true;
    };
    sfm.tick(48, send_sfm);
    assert(!sfm.accepted() && !sfm.ready(48));
    sfm.tick(49, send_sfm);
    assert(sfm.accepted() && !sfm.ready(49));
    sfm_callback(53); // Callback reports DRAM command issue, not completion.
    assert(!sfm.ready(100)); // Callback processing is deferred to next tick.
    sfm.tick(54, send_sfm);
    assert(sends == 2 && sfm.sv_eligible_cycle() == 63);
    assert(!sfm.ready(62));
    assert(sfm.ready(63)); // Only now may the frontend request SV.

    bool invalid = false;
    try {
        SoftmaxPhaseSubmitter wrong({PIMOpcode::MacAllBank, 2ULL << 30},
                                    45, modeled_delay);
        (void)wrong;
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    assert(invalid);

    SoftmaxPhaseSubmitter overflow({PIMOpcode::Softmax, 2ULL << 30},
                                   45, modeled_delay);
    SoftmaxPhaseSubmitter::IssueCallback overflow_callback;
    overflow.tick(50, [&](const PIMCommand&, SoftmaxPhaseSubmitter::IssueCallback cb) {
        overflow_callback = std::move(cb);
        return true;
    });
    overflow_callback(std::numeric_limits<std::uint64_t>::max() - 1);
    bool exceeded = false;
    try {
        overflow.tick(std::numeric_limits<std::uint64_t>::max(),
                      [&](const PIMCommand&, SoftmaxPhaseSubmitter::IssueCallback) {
                          assert(false);
                          return false;
                      });
    } catch (const std::overflow_error&) {
        exceeded = true;
    }
    assert(exceeded);

    bool invalid_input = false;
    try {
        (void)estimate_softmax_cycles(0, attacc_config);
    } catch (const std::invalid_argument&) {
        invalid_input = true;
    }
    assert(invalid_input);

    // Never submit an SFM before the QK gate's last issue cycle.
    SoftmaxPhaseSubmitter early({PIMOpcode::Softmax, 2ULL << 30}, 45);
    early.tick(44, [&](const PIMCommand&, SoftmaxPhaseSubmitter::IssueCallback) {
        assert(false && "SFM submitted before QK issue boundary");
        return false;
    });
    assert(!early.accepted());

    std::cout << "PASS: all-channel QK, SFM backpressure, deferred issue, timed SV\n";
}