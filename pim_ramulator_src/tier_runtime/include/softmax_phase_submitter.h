#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "pim_command.h"

namespace Ramulator::TierRuntime {
    // One SFM command per query head. With zero modeled latency, SV becomes
    // eligible when the SFM issue callback is drained on a later frontend tick.
    // An optional analytic delay can be supplied for a separate sensitivity run.
    class SoftmaxPhaseSubmitter {
    public:
        using IssueCallback = std::function<void(std::uint64_t)>;
        using Send = std::function<bool(const PIMCommand&, IssueCallback)>;

        SoftmaxPhaseSubmitter(PIMCommand command,
                                std::uint64_t qk_last_issue_cycle,
                                std::uint64_t modeled_latency_cycles = 0);
        
        // Call on each frontend tick. Callbacks are drained on a later tick.
        // send() false means the request was not accepted and can be retried
        void tick(std::uint64_t current_cycle, const Send& send);
        [[nodiscard]] bool ready(std::uint64_t current_cycle) const noexcept;
        [[nodiscard]] std::uint64_t sv_eligible_cycle() const;
        [[nodiscard]] bool accepted() const noexcept;
    private:
        PIMCommand m_command;
        std::uint64_t m_qk_last_issue_cycle;
        std::uint64_t m_latency_cycles;
        bool m_accepted = false;
        std::optional<std::uint64_t> m_sv_eligible_cycle;
        std::shared_ptr<std::vector<std::uint64_t>> m_events;
    };
} // namespace Ramulator::TierRuntime