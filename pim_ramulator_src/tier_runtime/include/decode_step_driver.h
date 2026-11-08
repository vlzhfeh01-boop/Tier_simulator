#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "qk_phase_submitter.h"
#include "runtime_decode_command_builder.h"
#include "softmax_phase_submitter.h"

namespace Ramulator::TierRuntime {
    // Drives exactly one already-admitted token and one decoder layer.
    // Cycles are command ISSUE cycles. A separate serving model must decide
    // when a full token (including later layers and non-PIM work) completes.
    class DecodeStepDriver {
    public:
        using IssueCallback = std::function<void(std::uint64_t)>;
        using Send = std::function<bool(const PIMCommand&, IssueCallback)>;
        using PhaseComplete = std::function<void(DecodePhase, std::uint64_t /* query head*/, std::uint64_t /* cycle */)>;

        DecodeStepDriver(RuntimeDecodeLayerPlan plan,
                            const PIMCommandGenerator& generator,
                            const AttAccAddressEncoder& encoder, 
                            std::uint64_t sodtmax_delay_cycles = 0,
                            PhaseComplete on_phase_complete = {});
        
        void tick(std::uint64_t cycle, const Send& send);
        [[nodiscard]] bool done() const noexcept;
        [[nodiscard]] std::uint64_t last_sv_issue_cycle() const;
    
    private:
        using Event = std::pair<std::size_t, std::uint64_t>;

        void begin_phase();
        void finish_phase(std::uint64_t cycle);
        void submit_kv(std::uint64_t cycle, const Send& send);
        void submit_sv(const Send& send);

        StagedDecodeLayer m_layer;
        const AttAccAddressEncoder& m_encoder;
        const std::uint64_t m_softmax_delay_cycles;
        PhaseComplete m_on_phase_complete;
        std::optional<DecodePhaseWork> m_work;
        std::unique_ptr<QKPhaseSubmitter> m_qk;
        std::unique_ptr<SoftmaxPhaseSubmitter> m_softmax;
        std::size_t m_accepted = 0;
        std::vector<bool> m_issued;
        std::shared_ptr<std::vector<Event>> m_events;
        std::optional<std::uint64_t> m_last_qk_issue;
        std::optional<std::uint64_t> m_last_sv_issue;
    };
} // namespace Ramulator::TierRuntime