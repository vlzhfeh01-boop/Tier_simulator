#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
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
#include "shared_channel_space.h"

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
    class TierDecodeStepFrontend final : public IFrontEnd, public Implementation {
        RAMULATOR_REGISTER_IMPLEMENTATION(
            IFrontEnd, TierDecodeStepFrontend, "TierDecodeStep",
            "Issue staged Tier decode requests against the real HBM controllers.");

        struct WorkItem {
            TierRuntime::RequestContext request;
            std::unique_ptr<TierRuntime::DecodeStepDriver> step;
            std::optional<std::uint64_t> token_ready_cycle;
            std::uint64_t last_sv_issue_cycle =0;
            std::unique_ptr<TierRuntime::TierChannelRuntime> shared_runtime;
            bool holds_channel_lease = false;
            std::vector<int> qk_channels;
            std::vector<int> leased_channels;
            std::optional<TierRuntime::DecodePhase> leased_phase;
            std::optional<std::uint64_t> sv_release_cycle;
        };

        int m_channel_a = 2;
        int m_channel_b = 5;
        std::uint64_t m_prompt_tokens = 33;
        std::uint64_t m_softmax_delay = 0;
        std::uint64_t m_sv_drain_cycles = 0;
        std::uint64_t m_max_cycles = 1000000;
        bool m_share_channels = false;
        std::uint64_t m_shared_region_rows = 8;
        std::string m_workload_path;
        std::unique_ptr<TierRuntime::RowAlignedKVLayout> m_layout;
        std::unique_ptr<TierRuntime::KVCapacityModel> m_capacity;
        std::unique_ptr<TierRuntime::OrderedBlockMapper> m_mapper;
        std::unique_ptr<TierRuntime::PIMCommandGenerator> m_generator;
        std::unique_ptr<TierRuntime::AttAccAddressEncoder> m_encoder;
        std::unique_ptr<TierRuntime::TierChannelRuntime> m_runtime;
        TierRuntime::FirstFreeChannelPolicy m_policy;
        std::unique_ptr<TierRuntime::TierChannelScheduler> m_scheduler;
        TierRuntime::TokenComletionModel m_completion{{}};
        std::unique_ptr<TierRuntime::SharedKVRowSpace> m_shared_space;
        TierRuntime::ChannelPhaseGuard m_channel_guard{16}; // 채널 16개
        std::vector<std::unique_ptr<WorkItem>> m_requests;
        std::optional<std::int64_t> m_sfm_owner;
        std::optional<std::uint64_t> m_last_sfm_submit_cycle;
        std::size_t m_next_request_index = 0;

        static std::uint64_t parse_integer(const std::string& field) {
            std::istringstream input(field);
            // 선행 공백 제거
            input >> std::ws;
            // 첫 글자 보기
            if(input.peek() == '-') {
                throw std::invalid_argument("negative request workload field");
            }
            std::uint64_t value = 0;
            // 정수 파싱, value 변수에 저장한다.
            if(!(input>>value)) {
                throw std::invalid_argument("invalid request workload number");
            }
            input >> std::ws;
            if(!input.eof()) {
                throw std::invalid_argument("unexpected characters in workload number");
            }
            return value;
        }
        void load_workload() {
            std::ifstream stream(m_workload_path);
            if(!stream) {
                throw std::invalid_argument("cannot open request workload: " + m_workload_path);
            }
            std::unordered_set<std::int64_t> ids;
            std::string line;
            while(std::getline(stream,line)) {
                if(line.find_first_not_of(" \t\r") == std::string::npos || 
                   line[line.find_first_not_of(" \t\r")] == '#') {
                    continue;
                }
                std::istringstream input(line);
                std::vector<std::string> fields;
            }
        }
    };
} // namespace Ramulator