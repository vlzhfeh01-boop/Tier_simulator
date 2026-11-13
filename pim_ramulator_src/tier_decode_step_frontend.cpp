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
            switch (opcode) {
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
    // Exclusive Tier requests and shared-channel requests use the same phase driver.
    class TierDecodeStepFrontend final : public IFrontEnd, public Implementation {
        RAMULATOR_REGISTER_IMPLEMENTATION(
            IFrontEnd, TierDecodeStepFrontend, "TierDecodeStep",
            "Issue staged Tier decode requests against the real HBM controllers.");

        struct WorkItem {
            TierRuntime::RequestContext request;
            std::unique_ptr<TierRuntime::DecodeStepDriver> step;
            std::optional<std::uint64_t> token_ready_cycle;
            std::uint64_t last_sv_issue_cycle = 0;
            std::unique_ptr<TierRuntime::TierChannelRuntime> shared_runtime;
            bool holds_channel_lease = false;
            std::vector<int> qk_channels;
            std::vector<int> leased_channels;
            std::optional<TierRuntime::DecodePhase> leased_phase;
            std::optional<std::uint64_t> sv_release_cycle;
            std::optional<std::uint64_t> kv_growth_wait_start;
            std::optional<std::uint64_t> channel_growth_wait_start;
        };

        int m_channel_a = 2;
        int m_channel_b = 5;
        std::uint64_t m_prompt_tokens = 33;
        std::uint64_t m_softmax_delay = 0;
        std::uint64_t m_sv_drain_cycles = 0;
        std::uint64_t m_max_cycles = 1000000;
        bool m_share_channels = false;
        std::uint64_t m_shared_row_slot_limit = 0;
        std::string m_workload_path;
        std::unique_ptr<TierRuntime::RowAlignedKVLayout> m_layout;
        std::unique_ptr<TierRuntime::KVCapacityModel> m_capacity;
        std::unique_ptr<TierRuntime::OrderedBlockMapper> m_mapper;
        std::unique_ptr<TierRuntime::PIMCommandGenerator> m_generator;
        std::unique_ptr<TierRuntime::AttAccAddressEncoder> m_encoder;
        std::unique_ptr<TierRuntime::TierChannelRuntime> m_runtime;
        TierRuntime::FirstFreeChannelPolicy m_policy;
        std::unique_ptr<TierRuntime::TierChannelScheduler> m_scheduler;
        TierRuntime::TokenCompletionModel m_completion{{}};
        std::unique_ptr<TierRuntime::SharedKVRowSpace> m_shared_space;
        TierRuntime::ChannelPhaseGuard m_channel_guard{16};
        std::vector<std::unique_ptr<WorkItem>> m_requests;
        std::optional<std::int64_t> m_sfm_owner;
        std::optional<std::uint64_t> m_last_sfm_submit_cycle;
        std::size_t m_next_request_index = 0;
        bool m_summary_reported = false;

        static std::uint64_t parse_integer(const std::string& field) {
            std::istringstream input(field);
            // 선행 공백 제거
            input >> std::ws;
            // 첫 글자 보기
            if (input.peek() == '-') {
                throw std::invalid_argument("negative request workload field");
            }
            std::uint64_t value = 0;
            // 정수 파싱, value 변수에 저장한다.
            if (!(input >> value)) {
                throw std::invalid_argument("invalid request workload number");
            }
            input >> std::ws;
            if (!input.eof()) {
                throw std::invalid_argument("unexpected characters in workload number");
            }
            return value;
        }

        void load_workload() {
            std::ifstream stream(m_workload_path);
            if (!stream) {
                throw std::invalid_argument("cannot open request workload: " + m_workload_path);
            }
            std::unordered_set<std::int64_t> ids;
            std::string line;
            while (std::getline(stream, line)) {
                // 공백, 탭, 캐리지 리턴이 아닌 첫 번쨰 유효한 글자의 위치를 찾는다.
                if (line.find_first_not_of(" \t\r") == std::string::npos || 
                   line[line.find_first_not_of(" \t\r")] == '#') {
                    continue;
                }
                std::istringstream input(line);
                std::vector<std::string> fields;
                std::string field;
                while (std::getline(input, field, ',')) {
                    fields.push_back(field);
                }
                if (fields.size() != 5) {
                    throw std::invalid_argument("workload needs id,arrival,prompt,predicted,actual");
                }
                const auto id = parse_integer(fields[0]);
                const auto arrival = parse_integer(fields[1]);
                const auto prompt = parse_integer(fields[2]);
                const auto predicted = parse_integer(fields[3]);
                const auto actual = parse_integer(fields[4]);
                if (id > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) || 
                    arrival > static_cast<std::uint64_t>(std::numeric_limits<Clk_t>::max()) || 
                    prompt > std::numeric_limits<std::uint32_t>::max() ||
                    predicted > std::numeric_limits<std::uint32_t>::max() ||
                    actual == 0 || actual > std::numeric_limits<std::uint32_t>::max() ||
                    !ids.insert(static_cast<std::int64_t>(id)).second) {
                        throw std::invalid_argument("invalid or repeated request workload entry");
                }
                auto item = std::make_unique<WorkItem>();
                auto& spec = item->request.spec;
                spec.request_id = static_cast<std::int64_t>(id);
                spec.arrival_cycle = static_cast<Clk_t>(arrival);
                spec.prompt_tokens = static_cast<std::uint32_t>(prompt);
                spec.predicted_output_tokens = static_cast<std::uint32_t>(predicted);
                spec.actual_output_tokens = static_cast<std::uint32_t>(actual);
                spec.max_output_tokens = spec.actual_output_tokens;
                if (m_runtime->predicted_initial_channel_count(spec) > 16 || 
                    m_capacity->required_channels_for_tokens(
                        m_capacity->total_tokens(prompt,actual)) > 16) {
                    throw std::invalid_argument("request exceeds 16 physical Tier channels");
                }
                if (m_share_channels && m_shared_row_slot_limit != 0) {
                    const auto total = m_capacity->total_tokens(prompt, actual);
                    const auto per_slot = m_shared_space->tokens_per_row_stripe();
                    const auto required = total / per_slot +
                                          static_cast<std::uint64_t>(total % per_slot != 0);
                    if (required > m_shared_row_slot_limit) { // shared row slot limit을 초과하는 경우 시뮬레이션 시작 전에 확인
                        throw std::invalid_argument(
                            "request can never fit in shared row slot limit");
                    }
                }
                m_requests.push_back(std::move(item));
            }
            if (!stream.eof() || m_requests.empty()) {
                throw std::invalid_argument("empty or unreadable request workload");
            }
            std::stable_sort(m_requests.begin(), m_requests.end(),
            [](const auto& a, const auto& b) {
                return a->request.spec.arrival_cycle < b->request.spec.arrival_cycle;
            });
        }

        void admit_waiting() {
            for (auto& item : m_requests) {
                auto& request = item->request;
                if (request.state != TierRuntime::RequestState::Waiting) {
                    continue;
                }
                if (request.spec.arrival_cycle > m_clk) break;
                // shared tier의 경우
                if (m_share_channels) {
                    if(!m_shared_space->reserve(request.spec.request_id, request.spec.prompt_tokens)) break;
                    item->shared_runtime = std::make_unique<TierRuntime::TierChannelRuntime> (
                        *m_capacity, *m_mapper, 16);
                    std::vector<int> all_channels(16);
                    // 0부터 1씩 증가하면서 벡터를 채운다.
                    std::iota(all_channels.begin(), all_channels.end(),0);
                    item->shared_runtime->admit(request,std::move(all_channels),m_clk);
                    std::cout << "TIER_KV_REGION request_id=" << request.spec.request_id
                              << " row_offset="
                            << m_shared_space->row_offset(request.spec.request_id)
                            << " rows_per_slice="
                            << m_shared_space->rows_allocated(request.spec.request_id)
                            << " chunks="
                            << m_shared_space->rows_allocated(request.spec.request_id) *
                                m_shared_space->chunks_per_row_stripe()
                            << '\n';
                } else if (!m_scheduler->try_admit(request,m_clk)) {
                    break; // FIFO: a later arrival cannot bypass this request.
                }
                std::cout << "TIER_ADMIT request_id=" << request.spec.request_id
                      << " cycle=" << m_clk
                      << " channels=" << request.allocated_channels.size() << '\n';
            }
        }

        // Reserve a running request's next KV row before admitting new requests.
        // A failed reservation leaves its current mapping intact for a later retry.
        bool ensure_shared_growth(WorkItem& item) {
            const auto& request = item.request;
            const auto next_token_count =
                static_cast<std::uint64_t>(request.spec.prompt_tokens) +
                request.generated_tokens + 1;
            const auto before = m_shared_space->rows_allocated(request.spec.request_id);
            if (!m_shared_space->ensure_capacity(request.spec.request_id, next_token_count)) {
                if (!item.kv_growth_wait_start) { //start waiting
                    item.kv_growth_wait_start = static_cast<std::uint64_t>(m_clk);
                    std::cout << "TIER_KV_WAIT request_id=" << request.spec.request_id
                              << " token_index=" << next_token_count - 1
                              << " cycle=" << m_clk << '\n';
                }
                return false;
            }
            if (item.kv_growth_wait_start) {
                std::cout << "TIER_KV_RESUME request_id=" << request.spec.request_id
                      << " token_index=" << next_token_count - 1
                      << " cycle=" << m_clk
                      << " wait_cycles=" << static_cast<std::uint64_t>(m_clk) -
                         *item.kv_growth_wait_start << '\n';
                item.kv_growth_wait_start.reset();
            }
            if (m_shared_space->rows_allocated(request.spec.request_id) != before) {
                std::cout << "TIER_KV_GROW request_id=" << request.spec.request_id
                          << " logical_row=" << before
                          << " row_offset=" << m_shared_space->physical_row(
                                 request.spec.request_id, before)
                          << " rows_per_slice=" <<
                             m_shared_space->rows_allocated(request.spec.request_id)
                          << " cycle=" << m_clk << '\n';
            }
            return true;
        }

        std::optional<TierRuntime::DecodeTokenPlan> reserve_tier_growth(WorkItem& item) {
            auto plan = m_scheduler->try_begin_token(item.request);
            if (!plan && !item.channel_growth_wait_start) { // start waiting
                item.channel_growth_wait_start = static_cast<std::uint64_t>(m_clk);
                std::cout << "TIER_CHANNEL_WAIT request_id="
                            << item.request.spec.request_id
                            << " cycle=" << m_clk << '\n';
            }
            if (plan && item.channel_growth_wait_start) { // resume waiting
                std::cout << "TIER_CHANNEL_RESUME request_id="
                          << item.request.spec.request_id
                          << " cycle=" << m_clk
                          << " wait_cycles=" << static_cast<std::uint64_t>(m_clk) -
                             *item.channel_growth_wait_start << '\n';
                item.channel_growth_wait_start.reset();
            }
            return plan;
        }

        // A full pool alone is not a deadlock: a request with a command or a
        // pending token completion can still finish and realease its resources.
        bool evict_if_deadlocked(const std::vector<bool>& growth_blocked) {
            std::optional<std::size_t> victim;
            std::uint64_t smallest_allocation = 0;
            for (std::size_t i = 0; i < m_requests.size(); ++i) {
                const auto& item = *m_requests[i];
                if (item.request.state != TierRuntime::RequestState::Running) continue;
                if (!growth_blocked[i]) return false;
                if (item.step || item.token_ready_cycle || item.sv_release_cycle ||
                    item.holds_channel_lease || 
                    m_sfm_owner == item.request.spec.request_id ||
                    !item.request.is_evictable()) {
                    return false;
                }
                const auto allocated = m_share_channels
                ? m_shared_space->rows_allocated(item.request.spec.request_id)
                : item.request.allocated_channels.size();
                if (!victim || allocated < smallest_allocation ||
                    (allocated == smallest_allocation && 
                      (item.request.spec.arrival_cycle >
                            m_requests[*victim]->request.spec.arrival_cycle ||
                        (item.request.spec.arrival_cycle ==
                            m_requests[*victim]->request.spec.arrival_cycle &&
                            item.request.spec.request_id >
                            m_requests[*victim]->request.spec.request_id)))) {
                    victim = i;
                    smallest_allocation = allocated;
                }
            }
            if (!victim) return false;
            auto& item = *m_requests[*victim];
            auto& request = item.request;
            auto& runtime = m_share_channels ? *item.shared_runtime : *m_runtime;
            runtime.evict(request);
            if (m_share_channels) m_shared_space->release(request.spec.request_id);
            item.kv_growth_wait_start.reset();
            item.channel_growth_wait_start.reset();
            std::cout << "TIER_EVICT request_id=" << request.spec.request_id
                  << " cycle=" << m_clk
                  << " released_units=" << smallest_allocation
                  << " generated_tokens=" << request.generated_tokens
                  << " reason=DEADLOCK\n";
            return true;
        }

        bool send_command(WorkItem& item, const TierRuntime::PIMCommand& cmd,
                          TierRuntime::DecodeStepDriver::IssueCallback on_issue) {
            const auto mapped = m_share_channels ? m_shared_space->translate(item.request.spec.request_id, cmd) : cmd;
            if (mapped.address >
                static_cast<std::uint64_t>(std::numeric_limits<Addr_t>::max())) {
                throw std::overflow_error("PIM address exceeds signed Addr_t");
            }
            const int type = request_type(cmd.opcode);
            if ((type == Request::Type::Write || type == Request::Type::StoreFence) &&
                on_issue) {
                throw std::logic_error("write/fence cannot report a PIM issue");
            }
            // A single buffer-die SFM unit serves one head at a time across channels.
            if (type == Request::Type::PIM_SFM &&
            (m_sfm_owner || m_last_sfm_submit_cycle ==
                                static_cast<std::uint64_t>(m_clk))) {
                return false; // 이번 제출을 거절한다. 
            }
            Request req(static_cast<Addr_t>(mapped.address),
                type, // type_id
                0,  // source_id
                [on_issue = std::move(on_issue)](Request& issued) {
                    if(!on_issue) return;
                    if(issued.depart < 0) {
                        throw std::logic_error("PIM issue callback lacks cycle");
                    }
                    on_issue(static_cast<std::uint64_t>(issued.depart));
                });
            if(!m_memory_system->send(req)) {
                return false;
            }
            if(type == Request::Type::PIM_SFM) {
                m_sfm_owner = item.request.spec.request_id;
                m_last_sfm_submit_cycle = static_cast<std::uint64_t>(m_clk);
            }
            return true;
        }

        void release_shared_lease(WorkItem& item, std::uint64_t last_issue_cycle) {
            if (!m_share_channels || !item.holds_channel_lease ||
                !item.leased_phase) {
                throw std::logic_error("request has no shared phase lease to release");
            }
            const auto phase = *item.leased_phase;
            m_channel_guard.release(item.request.spec.request_id, item.leased_channels);
            item.leased_channels.clear();
            item.leased_phase.reset();
            item.holds_channel_lease = false;
            std::cout << "TIER_LEASE request_id=" << item.request.spec.request_id
                    << " phase=" << phase_name(phase)
                    << " event=RELEASE cycle=" << m_clk
                    << " last_issue_cycle=" << last_issue_cycle << '\n';
        }

        void finish_token(WorkItem& item) {
            auto& request = item.request;
            auto& runtime = m_share_channels ? *item.shared_runtime : *m_runtime;
            if (item.holds_channel_lease || item.sv_release_cycle) {
                throw std::logic_error("token completed with a PIM phase lease");
            }
            runtime.complete_token(request, m_clk);
            std::cout << "TIER_TOKEN request_id=" << request.spec.request_id
                    << " token=" << request.generated_tokens
                    << " last_sv_issue_cycle=" << item.last_sv_issue_cycle
                    << " completion_cycle=" << m_clk << '\n';
            item.token_ready_cycle.reset();
            if (m_share_channels) {
                for (std::size_t index = 0; index < m_requests.size(); ++index) {
                    if (m_requests[index].get() == &item) {
                        m_next_request_index = (index + 1) % m_requests.size();
                        break;
                    }
                }
            }
            if (request.state == TierRuntime::RequestState::Completed) {
                std::cout << "TIER_REQUEST request_id=" << request.spec.request_id
                        << " arrival_cycle=" << request.spec.arrival_cycle
                        << " admitted_cycle=" << request.admitted_cycle
                        << " first_token_cycle=" << request.first_token_cycle
                        << " completion_cycle=" << request.completion_cycle << '\n';
                runtime.release_completed(request);
                if (m_share_channels) m_shared_space->release(request.spec.request_id);
            }
        }

    public:
        void init() override {
            m_clock_ratio = param<uint>("clock_ratio").required();
            m_channel_a = param<int>("channel_a").default_val(2);
            m_channel_b = param<int>("channel_b").default_val(5);
            m_prompt_tokens = param<std::uint64_t>("prompt_tokens").default_val(33);
            m_softmax_delay = param<std::uint64_t>("softmax_delay_cycles").default_val(0);
            m_sv_drain_cycles = param<std::uint64_t>("sv_drain_cycles").default_val(0);
            m_share_channels = param<bool>("shared_channels").default_val(false);
            m_shared_row_slot_limit = 
                param<std::uint64_t>("shared_row_slot_limit").default_val(0);
            m_workload_path = param<std::string>("workload_path").default_val("");
            m_max_cycles = param<std::uint64_t>("max_cycles").default_val(1000000);
            m_completion = TierRuntime::TokenCompletionModel({
                m_sv_drain_cycles,
                param<std::uint64_t>("context_merge_cycles").default_val(0),
                param<std::uint64_t>("non_pim_cycles").default_val(0)});
            if (m_clock_ratio != 1 || m_channel_a < 0 || m_channel_b < 0 ||
                m_channel_a == m_channel_b || m_channel_a >= 16 ||
                m_channel_b >= 16 || m_prompt_tokens > 47 || m_max_cycles == 0 ||
                (m_share_channels && m_workload_path.empty())) {
                throw std::invalid_argument(
                    "TierDecodeStep requires clock_ratio=1, two distinct channels "
                    "in [0,16), and prompt_tokens <= 47");
            }

            // The sample uses three 16-token blocks across two initial channels.
            const TierRuntime::KVBlockShape shape({16, 1, 1, 128, 2});
            m_layout = std::make_unique<TierRuntime::RowAlignedKVLayout>(
                shape, TierRuntime::PIMGeometryConfig{2, 2, 4, 4, 16384, 32, 32});
            m_capacity = std::make_unique<TierRuntime::KVCapacityModel>(*m_layout);
            m_mapper = std::make_unique<TierRuntime::OrderedBlockMapper>(*m_capacity);
            m_generator = std::make_unique<TierRuntime::PIMCommandGenerator>(
                *m_layout, *m_mapper, 16);
            m_encoder = std::make_unique<TierRuntime::AttAccAddressEncoder>(
                *m_layout, 16);
            m_runtime = std::make_unique<TierRuntime::TierChannelRuntime>(
                *m_capacity, *m_mapper, 16);
            if (m_share_channels) {
                m_shared_space = std::make_unique<TierRuntime::SharedKVRowSpace>(
                    *m_layout, *m_encoder, 16, m_shared_row_slot_limit);
            }
            m_scheduler = std::make_unique<TierRuntime::TierChannelScheduler>(
                *m_runtime, m_policy);

            if (!m_workload_path.empty()) {
                load_workload();
            } else {
                auto item = std::make_unique<WorkItem>();
                auto& request = item->request;
                request.spec.request_id = 1;
                request.spec.prompt_tokens = static_cast<std::uint32_t>(m_prompt_tokens);
                request.spec.predicted_output_tokens = 1;
                request.spec.actual_output_tokens = 1;
                m_runtime->admit(request, {m_channel_a, m_channel_b}, 0);
                m_requests.push_back(std::move(item));
            }
        }
        
        void tick() override {
            if (is_finished()){
                return;
            }
            ++m_clk;
            // admission 보다 먼저 growth를 처리한다.
            const auto start = m_share_channels ? m_next_request_index : 0;
            std::vector<bool> growth_checked(m_requests.size(), false); // 다음 토큰의 공간을 이번 cycle에 확인했는가.
            std::vector<bool> growth_blocked(m_requests.size(), false); // 필요한 공간을 확보하지 못했는가.
            std::vector<std::optional<TierRuntime::DecodeTokenPlan>> growth_plans(m_requests.size());
            // Reserve resources for eligible running requests before admission.
            // A token that is still finishing cannot request its next growth yet.
            for (std::size_t offset = 0; offset < m_requests.size(); ++offset) {
                const auto index = (start + offset) % m_requests.size();
                auto& item = *m_requests[index];
                if (item.request.state != TierRuntime::RequestState::Running ||
                    item.step || item.token_ready_cycle) continue;
                if (m_share_channels) {
                    growth_checked[index] = true;
                    growth_blocked[index] = !ensure_shared_growth(item);
                } else if (m_runtime->needs_growth(item.request)) {
                    growth_checked[index] = true;
                    growth_plans[index] = reserve_tier_growth(item);
                    growth_blocked[index] = !growth_plans[index];
                }
            }
            // 필요하면 eviction 후 재시도
            if (evict_if_deadlocked(growth_blocked)) {
            // The victim has released one or more units. Give the blocked
            // running requests another chance before admitting newcomers.
            for (std::size_t offset = 0; offset < m_requests.size(); ++offset) {
                const auto index = (start + offset) % m_requests.size();
                if (!growth_blocked[index] ||
                    m_requests[index]->request.state !=
                        TierRuntime::RequestState::Running) continue;
                if (m_share_channels) {
                    growth_blocked[index] = !ensure_shared_growth(*m_requests[index]);
                } else {
                    growth_plans[index] = reserve_tier_growth(*m_requests[index]);
                    growth_blocked[index] = !growth_plans[index];
                }
            }
            }
            admit_waiting();
            // 요청마다 각 요청이 지금 할 수 있는 다음 작업을 시도한다.
            for(std::size_t offset = 0; offset < m_requests.size(); ++offset) {
                const auto index = (start+offset) % m_requests.size();
                auto& item = m_requests[index];
                auto& request = item->request;
                if(request.state != TierRuntime::RequestState::Running) continue;
                if(item->sv_release_cycle && 
                   static_cast<std::uint64_t>(m_clk) >= *item->sv_release_cycle) {
                    if(item->leased_phase != TierRuntime::DecodePhase::SV){
                        throw std::logic_error("SV drain lost its channel lease");
                    }
                    release_shared_lease(*item, item->last_sv_issue_cycle);
                    item->sv_release_cycle.reset();
                }
                // 해당 토큰의 명령 발행 + 후속 모델 지연까지 반영
                if(item->token_ready_cycle) {
                    if(static_cast<std::uint64_t>(m_clk) >= *item->token_ready_cycle){
                        finish_token(*item);
                    }
                    continue;
                }
                if(!item->step) {
                    if (growth_blocked[index] ||
                        (m_share_channels && !growth_checked[index] &&
                        !ensure_shared_growth(*item))) continue;
                    auto plan = growth_plans[index]
                        ? std::move(growth_plans[index])
                        : m_share_channels
                            ? item->shared_runtime->begin_token(request)
                            : m_scheduler->try_begin_token(request);
                    if(!plan) {
                        if(m_share_channels) {
                            throw std::logic_error("shared token exceeds channel mapping capacity");
                        }
                        continue; // Wait for an exclusive growth channel.
                    }
                    const auto token_index = plan->token_index;
                    if(m_share_channels) {
                        // The existing ST_FENCE is channel-wide. Keep KV append
                        // and QK in one lease so another request cannot insert a
                        // store fence into a partially submitted QK stream.
                        item->qk_channels = m_generator->generate_qk(
                            TierRuntime::QKCommandInput{
                                0,0,plan->valid_token_count,
                                plan->allocated_channel_ids,
                                plan->initial_channel_count}).active_channel_ids;
                    }
                    auto layer = TierRuntime::build_runtime_decode_layer_plan(
                        request,*plan,0,1,1,*m_generator,*m_encoder);
                    const auto id = request.spec.request_id;
                    auto* const work = item.get();
                    item->step = std::make_unique<TierRuntime::DecodeStepDriver>(
                        std::move(layer), *m_generator, *m_encoder, m_softmax_delay,
                        // on phase complete
                        [this,work,id,token_index](TierRuntime::DecodePhase phase,
                                            std::uint64_t head, std::uint64_t cycle) {
                            if(m_share_channels && 
                                (phase == TierRuntime::DecodePhase::QK || 
                                phase == TierRuntime::DecodePhase::SV)) {
                                if(!work->holds_channel_lease || 
                                    work->leased_phase != phase) {
                                    throw std::logic_error("shared PIM phase lost its channel lease");
                                }
                            
                                if (phase == TierRuntime::DecodePhase::QK) {
                                    // 공유 채널 lease 반환
                                    release_shared_lease(*work,cycle);
                                } else {
                                    if(cycle > std::numeric_limits<std::uint64_t>::max() - m_sv_drain_cycles) {
                                        throw std::overflow_error("SV channel drain cycle overflow");
                                    }
                                    // 마지막 SV 발행 사이클 기록, 
                                    work->last_sv_issue_cycle = cycle;
                                    work->sv_release_cycle = cycle + m_sv_drain_cycles;
                                    if (static_cast<std::uint64_t>(m_clk) >= 
                                        *work->sv_release_cycle) {
                                            // sv_drain_cycles이후 채널 lease 반환 예약.
                                            // 잠시 잡아 둔 채널 실행 권한만 반환한다.
                                            release_shared_lease(*work,cycle);
                                            work->sv_release_cycle.reset();
                                    }
                                }
                            }
                            if(phase == TierRuntime::DecodePhase::Softmax) {
                                // m_sfm_owner 해제.
                                if(m_sfm_owner != id) {
                                    throw std::logic_error("shared SFM owner mismatch");
                                }
                                m_sfm_owner.reset();
                            }
                            std::cout << "TIER_PHASE head=" << head
                                  << " event=" << phase_name(phase)
                                  << " cycle=" << cycle
                                  << " request_id=" << id
                                  << " token_index=" << token_index << '\n';
                        });
                }
                if(m_share_channels) {
                    const auto& phase = item->step->pending_phase();
                    const bool qk_with_append = 
                        phase.phase == TierRuntime::DecodePhase::KVAppendFence || 
                        phase.phase == TierRuntime::DecodePhase::QK;
                    const bool uses_channel_buffers = qk_with_append ||
                        phase.phase == TierRuntime::DecodePhase::SV;
                    const auto lease_phase = qk_with_append
                        ? TierRuntime::DecodePhase::QK : phase.phase;
                    const auto& channels = qk_with_append
                        ? item->qk_channels : phase.active_channel_ids;
                    if (qk_with_append && phase.phase == TierRuntime::DecodePhase::QK &&
                        phase.active_channel_ids != channels) {
                        throw std::logic_error("QK participants changed after KV append");
                    }
                    if(uses_channel_buffers && !item->holds_channel_lease) {
                        if(!m_channel_guard.try_acquire(
                            request.spec.request_id,channels)) {
                                continue;
                        }
                        item->leased_channels = channels;
                        item->leased_phase = lease_phase;
                        item->holds_channel_lease = true;
                        std::cout << "TIER_LEASE request_id=" << request.spec.request_id
                                << " phase=" << phase_name(lease_phase)
                                << " event=ACQUIRE cycle=" << m_clk
                                << " channels=" << item->leased_channels.size() << '\n';
                    } else if(uses_channel_buffers && item->leased_phase != lease_phase) {
                        // 이전 SV의 drain이 아직 끝나지 않은 경우. 
                        // 이번 사이클은 기다린다.
                        if(item->leased_phase == TierRuntime::DecodePhase::SV && 
                            item->sv_release_cycle) {
                                continue; // Wait for the previous head's channel drain.
                        }
                        // 그 외에는 상태 관리 오류이므로 예외를 던진다.
                        throw std::logic_error("shared request holds the wrong PIM phase lease");
                        // 세번쨰 분기 : 채널을 쓰지 않는 phase인데 lease가 남는 경우
                        // 직전 SV의 drain 시간이 아직 남아있는 경우는 남아있는 lease를 허용한다.
                    } else if(!uses_channel_buffers && item->holds_channel_lease && 
                            !(item->leased_phase == TierRuntime::DecodePhase::SV && 
                                item->sv_release_cycle)) {
                        throw std::logic_error("shared PIM channels held outside QK/SV");
                    }
                }
                item->step->tick(static_cast<std::uint64_t>(m_clk),
                    [this,&item](const TierRuntime::PIMCommand& cmd,
                                TierRuntime::DecodeStepDriver::IssueCallback cb) {
                        return send_command(*item,cmd,std::move(cb));
                });
                if(item->step->done()) {
                    item->last_sv_issue_cycle = item->step->last_sv_issue_cycle();
                    item->token_ready_cycle = m_completion.earliest_cycle(item->last_sv_issue_cycle);
                    std::cout << "TIER_STEP last_sv_issue_cycle="
                            << item->last_sv_issue_cycle
                            << " request_id=" << request.spec.request_id << '\n';
                    item->step.reset();
                    if (static_cast<std::uint64_t>(m_clk) >= *item->token_ready_cycle) {
                        finish_token(*item);
                    }
                }
            }
            if (static_cast<std::uint64_t>(m_clk) >= m_max_cycles && !is_finished()) {
                throw std::runtime_error("Tier decode workload exceeded max_cycles");
            }
            if (is_finished() && !m_summary_reported) {
                const auto evicted = std::count_if(m_requests.begin(), m_requests.end(),
                    [](const auto& item) {
                        return item->request.state == TierRuntime::RequestState::Evicted;
                    });
                std::cout << "TIER_SUMMARY total_requests=" << m_requests.size()
                        << " completed=" << m_requests.size() - evicted
                        << " evicted=" << evicted
                        << " final_cycle=" << m_clk << '\n';
                m_summary_reported = true;
            }
            
        }

        bool is_finished() override {
            return std::all_of(m_requests.begin(),m_requests.end(),
                [](const auto& item) {
                    return item->request.is_terminal();
                });
        }
    };
} // namespace Ramulator
