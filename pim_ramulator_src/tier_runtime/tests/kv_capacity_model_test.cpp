#include "kv_capacity_model.h"
#include "row_aligned_kv_layout.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace Ramulator {
namespace TierRuntime {

namespace {
RowAlignedKVLayout make_test_layout() {
  const KVBlockShapeConfig shape_config{
      .tokens_per_block = 16,
      .layers_on_hbm = 2,
      .kv_heads_per_layer_on_hbm = 3,
      .head_dimension = 128,
      .bytes_per_element = 2,
  };

  const KVBlockShape shape(shape_config);

  const PIMGeometryConfig geometry{
      .pseudochannels_per_channel = 2,
      .ranks_per_pseudochannel = 2,
      .bank_groups_per_rank = 4,
      .banks_per_bank_group = 4,

      .rows_per_bank = 16384,
      .columns_per_row = 32,
      .bytes_per_column = 32,
  };

  return RowAlignedKVLayout(
      shape,
      geometry);
}
void test_block_count(
    const KVCapacityModel& capacity_model)
{
    const std::uint64_t tokens_per_block =
        capacity_model.tokens_per_block();

    assert(
        capacity_model.blocks_for_tokens(0)
        == 0);

    assert(
        capacity_model.blocks_for_tokens(1)
        == 1);

    assert(
        capacity_model.blocks_for_tokens(
            tokens_per_block)
        == 1);

    assert(
        capacity_model.blocks_for_tokens(
            tokens_per_block + 1)
        == 2);

    assert(
        capacity_model.blocks_for_tokens(
            tokens_per_block * 2)
        == 2);
}

void test_required_channel_count(
    const KVCapacityModel& capacity_model)
{
    const std::uint64_t blocks_per_channel =
        capacity_model.blocks_per_channel();

    assert(
        capacity_model.required_channels_for_blocks(0)
        == 0);

    assert(
        capacity_model.required_channels_for_blocks(1)
        == 1);

    assert(
        capacity_model.required_channels_for_blocks(
            blocks_per_channel)
        == 1);

    assert(
        capacity_model.required_channels_for_blocks(
            blocks_per_channel + 1)
        == 2);

    assert(
        capacity_model.required_channels_for_blocks(
            blocks_per_channel * 2)
        == 2);
}

void test_token_capacity_boundary(
    const KVCapacityModel& capacity_model)
{
    const std::uint64_t tokens_per_block =
        capacity_model.tokens_per_block();

    const std::uint64_t blocks_per_channel =
        capacity_model.blocks_per_channel();

    const std::uint64_t one_channel_token_capacity =
        tokens_per_block * blocks_per_channel;

    assert(
        capacity_model.required_channels_for_tokens(
            one_channel_token_capacity)
        == 1);

    assert(
        capacity_model.required_channels_for_tokens(
            one_channel_token_capacity + 1)
        == 2);

    assert(
        capacity_model.fits(
            one_channel_token_capacity,
            1));

    assert(
        !capacity_model.fits(
            one_channel_token_capacity + 1,
            1));

    assert(
        capacity_model.fits(
            one_channel_token_capacity + 1,
            2));
}

void test_new_block_boundary(
    const KVCapacityModel& capacity_model)
{
    const std::uint64_t tokens_per_block =
        capacity_model.tokens_per_block();

    assert(
        capacity_model.next_token_starts_new_block(0));

    assert(
        !capacity_model.next_token_starts_new_block(1));

    assert(
        !capacity_model.next_token_starts_new_block(
            tokens_per_block - 1));

    assert(
        capacity_model.next_token_starts_new_block(
            tokens_per_block));

    assert(
        capacity_model.next_token_starts_new_block(
            tokens_per_block * 2));
}

void test_growth_boundary(
    const KVCapacityModel& capacity_model)
{
    constexpr std::uint64_t allocated_channels = 2;

    const std::uint64_t tokens_per_block =
        capacity_model.tokens_per_block();

    const std::uint64_t total_block_capacity =
        capacity_model.block_capacity(
            allocated_channels);

    const std::uint64_t total_token_capacity =
        total_block_capacity
        * tokens_per_block;

    assert(
        !capacity_model.next_token_requires_growth(
            total_token_capacity - 1,
            allocated_channels));

    assert(
        capacity_model.next_token_requires_growth(
            total_token_capacity,
            allocated_channels));

    assert(
        !capacity_model.next_token_requires_growth(
            total_token_capacity,
            allocated_channels + 1));
}

void test_invalid_current_state(
    const KVCapacityModel& capacity_model)
{
    constexpr std::uint64_t allocated_channels = 1;

    const std::uint64_t tokens_per_block =
        capacity_model.tokens_per_block();

    const std::uint64_t total_block_capacity =
        capacity_model.block_capacity(
            allocated_channels);

    const std::uint64_t invalid_token_count =
        total_block_capacity
        * tokens_per_block
        + 1;

    bool exception_was_thrown = false;

    try {
        capacity_model.next_token_requires_growth(
            invalid_token_count,
            allocated_channels);
    }
    catch (const std::logic_error&) {
        exception_was_thrown = true;
    }

    assert(exception_was_thrown);
}

void test_zero_allocated_channel(
    const KVCapacityModel& capacity_model)
{
    bool exception_was_thrown = false;

    try {
        capacity_model.next_token_requires_growth(
            0,
            0);
    }
    catch (const std::logic_error&) {
        exception_was_thrown = true;
    }

    assert(exception_was_thrown);
}

void test_addition_overflow(
    const KVCapacityModel& capacity_model)
{
    bool exception_was_thrown = false;

    try {
        capacity_model.total_tokens(
            std::numeric_limits<std::uint64_t>::max(),
            1);
    }
    catch (const std::overflow_error&) {
        exception_was_thrown = true;
    }

    assert(exception_was_thrown);
}

void run_capacity_model_tests(
    const RowAlignedKVLayout& layout)
{
    const KVCapacityModel capacity_model(layout);

    test_block_count(capacity_model);
    test_required_channel_count(capacity_model);
    test_token_capacity_boundary(capacity_model);
    test_new_block_boundary(capacity_model);
    test_growth_boundary(capacity_model);
    test_invalid_current_state(capacity_model);
    test_zero_allocated_channel(capacity_model);
    test_addition_overflow(capacity_model);
}

}  // namespace

}  // namespace TierRuntime
}  // namespace Ramulator

int main()
{
    const auto layout =
        Ramulator::TierRuntime::make_test_layout();

    Ramulator::TierRuntime::run_capacity_model_tests(
        layout);

    std::cout
        << "KVCapacityModel tests passed.\n";

    return 0;
}