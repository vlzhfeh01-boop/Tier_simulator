#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "qk_issue_gate.h"

namespace Ramulator::TierRuntime {

    // Frontend-side bridge between QK Request issue callbacks and QKIssueGate.
    // A callback only enqueues an event. Drain it on a later frontend tick.
    class QKPhaseSubmitter {
        public:
            using IssueCallback = std::function<void(std::uint64_t)>;
            using Send = std::function<bool(const PIMCommand&, IssueCallback)>;

            QKPhaseSubmitter(std::vector<PIMCommand> commands,
                             std::vector<int> active_channels,
                             QKIssueGate::ChannelOfAddress channel_of_address);

            // Invoke once per frontend tick. A rejected command is retried next tick
            // with its original ordinal; callbacks are observed on a subsequent tick.
            void tick(const Send& send);
            [[nodiscard]] bool ready() const noexcept;
            [[nodiscard]] std::uint64_t ready_cycle() const;
            [[nodiscard]] std::size_t accepted_count() const noexcept;

        private:
            using Event = std::pair<std::size_t, std::uint64_t>;
            std::vector<PIMCommand> m_commands;
            QKIssueGate m_gate;
            std::size_t m_accepted = 0;
            std::shared_ptr<std::vector<Event>> m_events;
    };
}   // namespace Ramulator::TierRuntime