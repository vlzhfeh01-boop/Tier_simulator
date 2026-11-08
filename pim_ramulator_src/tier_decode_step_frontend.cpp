#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "frontend/frontend.h"
#include "decode_step_driver.h"
#include "tier_channel_runtime.h"
#include "tier_channel_scheduler.h"
#include "token_completion_model.h"

namespace Ramulator {
    namespace {
        int request_type(TierRuntime::PIMOpcode opcode) {
            using TierRuntime::PIMOpcode;
            switch(opcode) {
                case PIMOpcode::MemoryWrite:        return Request::Type::Write;
                case PIMOpcode::StoreFence:         return Request::Type::StoreFence;
                case PIMOpcode::WriteGemvBuffer:    return Request::Type::PIM_WR_GB;
                case PIMOpcode::MacAllBank:         return Request::Type::PIM_MAC_AB;
                case PIMOpcode::MoveSoftmaxBuffer:  return Request::Type::PIM_MV_SB;
                case PIMOpcode::Barrier:            return Request::Type::PIM_BARRIER;
                case PIMOpcode::Softmax:            return Request::Type::PIM_SFM;
                case PIMOpcode::MoveGemvBuffer:     return Request::Type::PIM_MV_GB;
            }
            throw std::invalid_argument("unsupported decode opcode");
        }

        const char* phase_name(TierRuntime::DecodePhase phase) {
            using TierRuntime::DecodePhase;
            switch (phase) {
            case DecodePhase::KVAppendFence: return "KV_APPEND_FENCE";
            case DecodePhase::QK: return "QK_LAST_ISSUE";
            case DecodePhase::Softmax: return "SFM_SV_ELIGIBLE";
            case DecodePhase::SV: return "SV_LAST_ISSUE";
            case DecodePhase::ContextMerge: return "CONTEXT_MERGE_ZERO_COST";
            case DecodePhase::Done: return "DONE";
            }
            throw std::invalid_argument("invalid decode phase");
        }
    } // namespace

    // All admitted Tier requests use the same command generator and phase driver.
    // Channel ownership is exclusive; shared-channel KV placement is not modeled.
    class TierDecideStepFrontend final : public IFrontEnd, public Implementation {
        RAMULATOR_REGISTER_IMPLEMENTATION(
        IFrontEnd, TierDecodeStepFrontend, "TierDecodeStep",
        "Issue staged Tier decode requests against the real HBM controllers.");

        struct WorkItem {
            TierRuntime::RequestContext request;
            std::unique_ptr<TierRuntime::DecodeStepDriver> step;
            std::optional<std::uint64_t> token_ready_cycle;
            std::uint64_t last_sv_issue_cycle =0;
        }
    }
} // namespace Ramulator