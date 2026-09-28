#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "kv_block_shape.h"
#include "row_aligned_kv_layout.h"

using namespace Ramulator::TierRuntime;

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


void test_derived_geometry(
    const RowAlignedKVLayout& layout) {
  assert(layout.token_partition_count() == 16);
  assert(layout.dimension_partition_count() == 4);
  assert(layout.physical_bank_count() == 64);

  assert(layout.tokens_per_block() == 16);
  assert(layout.elements_per_column() == 16);
  assert(layout.dimensions_per_bank() == 32);

  assert(layout.columns_per_block_in_slice() == 2);
  assert(layout.tiles_per_row() == 16);
  assert(layout.tokens_per_row() == 256);
}


void test_slice_capacity(
    const RowAlignedKVLayout& layout) {
  assert(layout.slice_count() == 12);

  assert(layout.rows_per_slice_capacity() == 1365);
  assert(layout.unused_rows_per_bank() == 4);

  assert(layout.max_blocks_per_channel() == 21840);
  assert(layout.max_tokens_per_channel() == 349440);
}


void test_token_partition(
    const RowAlignedKVLayout& layout) {
  {
    const TokenPartition location =
        layout.locate_token_partition(0);

    assert(location.pseudochannel == 0);
    assert(location.rank == 0);
    assert(location.bank_group == 0);
  }

  {
    const TokenPartition location =
        layout.locate_token_partition(5);

    assert(location.pseudochannel == 0);
    assert(location.rank == 1);
    assert(location.bank_group == 1);
  }

  {
    const TokenPartition location =
        layout.locate_token_partition(15);

    assert(location.pseudochannel == 1);
    assert(location.rank == 1);
    assert(location.bank_group == 3);
  }
}


void test_dimension_partition(
    const RowAlignedKVLayout& layout) {
  {
    const DimensionPartition location =
        layout.locate_dimension_partition(0);

    assert(location.bank == 0);
    assert(location.dimension_step == 0);
    assert(location.lane == 0);
  }

  {
    const DimensionPartition location =
        layout.locate_dimension_partition(31);

    assert(location.bank == 0);
    assert(location.dimension_step == 1);
    assert(location.lane == 15);
  }

  {
    const DimensionPartition location =
        layout.locate_dimension_partition(32);

    assert(location.bank == 1);
    assert(location.dimension_step == 0);
    assert(location.lane == 0);
  }

  {
    const DimensionPartition location =
        layout.locate_dimension_partition(70);

    assert(location.bank == 2);
    assert(location.dimension_step == 0);
    assert(location.lane == 6);
  }

  {
    const DimensionPartition location =
        layout.locate_dimension_partition(127);

    assert(location.bank == 3);
    assert(location.dimension_step == 1);
    assert(location.lane == 15);
  }
}


void test_slice_base_rows(
    const RowAlignedKVLayout& layout) {
  assert(
      layout.slice_index(
          KVKind::Key,
          0,
          0) == 0);

  assert(
      layout.slice_index(
          KVKind::Key,
          0,
          1) == 1);

  assert(
      layout.slice_index(
          KVKind::Key,
          1,
          0) == 3);

  assert(
      layout.slice_index(
          KVKind::Value,
          0,
          0) == 6);

  assert(
      layout.slice_index(
          KVKind::Value,
          1,
          2) == 11);

  assert(
      layout.slice_base_row(
          KVKind::Key,
          0,
          0) == 0);

  assert(
      layout.slice_base_row(
          KVKind::Key,
          0,
          1) == 1365);

  assert(
      layout.slice_base_row(
          KVKind::Key,
          1,
          0) == 4095);

  assert(
      layout.slice_base_row(
          KVKind::Value,
          0,
          0) == 8190);

  assert(
      layout.slice_base_row(
          KVKind::Value,
          1,
          2) == 15015);
}


void test_mac_locations(
    const RowAlignedKVLayout& layout) {
  {
    const RowColumnLocation location =
        layout.locate_mac(
            KVKind::Key,
            0,
            0,
            0,
            0);

    assert(location.row == 0);
    assert(location.column == 0);
  }

  {
    const RowColumnLocation location =
        layout.locate_mac(
            KVKind::Key,
            0,
            0,
            0,
            1);

    assert(location.row == 0);
    assert(location.column == 1);
  }

  {
    const RowColumnLocation location =
        layout.locate_mac(
            KVKind::Key,
            0,
            0,
            15,
            1);

    assert(location.row == 0);
    assert(location.column == 31);
  }

  {
    const RowColumnLocation location =
        layout.locate_mac(
            KVKind::Key,
            0,
            0,
            16,
            0);

    assert(location.row == 1);
    assert(location.column == 0);
  }

  {
    const RowColumnLocation location =
        layout.locate_mac(
            KVKind::Key,
            0,
            1,
            17,
            1);

    assert(location.row == 1366);
    assert(location.column == 3);
  }
}


void test_required_rows(
    const RowAlignedKVLayout& layout) {
  assert(layout.required_rows_per_slice(0) == 0);

  assert(layout.required_rows_per_slice(1) == 1);
  assert(layout.required_rows_per_slice(16) == 1);

  assert(layout.required_rows_per_slice(17) == 2);
  assert(layout.required_rows_per_slice(32) == 2);

  assert(layout.required_rows_per_slice(33) == 3);
}


void test_occupied_bytes(
    const RowAlignedKVLayout& layout) {
  assert(layout.occupied_bytes(0) == 0);

  assert(layout.occupied_bytes(1) == 786432);
  assert(layout.occupied_bytes(16) == 786432);

  assert(layout.occupied_bytes(17) == 1572864);
}


void test_invalid_indices(
    const RowAlignedKVLayout& layout) {
  bool token_error_thrown = false;

  try {
    static_cast<void>(
        layout.locate_token_partition(16));
  } catch (const std::out_of_range&) {
    token_error_thrown = true;
  }

  assert(token_error_thrown);


  bool dimension_error_thrown = false;

  try {
    static_cast<void>(
        layout.locate_dimension_partition(128));
  } catch (const std::out_of_range&) {
    dimension_error_thrown = true;
  }

  assert(dimension_error_thrown);


  bool block_error_thrown = false;

  try {
    static_cast<void>(
        layout.locate_mac(
            KVKind::Key,
            0,
            0,
            layout.max_blocks_per_channel(),
            0));
  } catch (const std::out_of_range&) {
    block_error_thrown = true;
  }

  assert(block_error_thrown);
}

}  // namespace


int main() {
  const RowAlignedKVLayout layout =
      make_test_layout();

  test_derived_geometry(layout);
  test_slice_capacity(layout);
  test_token_partition(layout);
  test_dimension_partition(layout);
  test_slice_base_rows(layout);
  test_mac_locations(layout);
  test_required_rows(layout);
  test_occupied_bytes(layout);
  test_invalid_indices(layout);

  std::cout
      << "All RowAlignedKVLayout tests passed.\n";

  return 0;
}