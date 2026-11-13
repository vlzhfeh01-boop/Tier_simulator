#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include "shared_channel_space.h"

using namespace Ramulator::TierRuntime;

int main() {
    const KVBlockShape shape({16, 1, 1, 128, 2});
    const RowAlignedKVLayout layout(shape, {2, 2, 4, 4, 16384, 32, 32});
    const AttAccAddressEncoder encoder(layout, 16);
    SharedKVRowSpace rows(layout, encoder, 16);
    assert(layout.row_bytes_per_bank() * layout.physical_bank_count() * 16 ==
           1024 * 1024);
    assert(rows.tokens_per_row_stripe() == 4096);
    // K and V are separate slices, so each row stripe has 2 chunks.
    assert(rows.chunks_per_row_stripe() == 2); // K and V are separate slices.

    assert(rows.reserve(11, 33));
    assert(rows.reserve(22, 33));
    assert(rows.rows_allocated(11) == 1);
    assert(rows.row_offset(11) == 0 && rows.row_offset(22) == 1);
    for (const KVKind kind : {KVKind::Key, KVKind::Value}) {
        const auto base = layout.slice_base_row(kind, 0, 0);
        const PIMCommand write{PIMOpcode::MemoryWrite,
            encoder.encode({2, 0, 0, 0, 1, base, 3, 0})};
        const auto a = encoder.decode(rows.translate(11, write).address);
        const auto b = encoder.decode(rows.translate(22, write).address);
        assert(a.row == base && b.row == base + 1);
        assert(a.channel_id == b.channel_id && b.channel_id == 2);
        assert(a.column == b.column && b.column == 3);
    }
    const PIMCommand sfm{PIMOpcode::Softmax,
                         encoder.encode({2, 0, 0, 0, 0, 0, 0, 0})};
    assert(rows.translate(22, sfm).address == sfm.address);
    const PIMCommand next_row{PIMOpcode::MacAllBank,
        encoder.encode({2, 0, 0, 0, 0, 1, 3, 0})};
    try {
        (void)rows.translate(11, next_row);
        assert(false && "growth was skipped");
    } catch (const std::out_of_range&) {}

    assert(rows.ensure_capacity(11, 4096));
    assert(rows.rows_allocated(11) == 1);
    assert(rows.ensure_capacity(11, 4097));
    assert(rows.physical_row(11, 1) == 2);
    // Releasing request 22 creates a hole. Subsequent growth must preserve
    // request 11's existing addresses while reusing the hole.
    rows.release(22);
    assert(rows.ensure_capacity(11, 8193));
    assert(rows.physical_row(11, 0) == 0);
    assert(rows.physical_row(11, 1) == 2);
    assert(rows.physical_row(11, 2) == 1);
    for (const auto& [logical_row, expected_row] :
         {std::pair<int, int>{1, 2}, {2, 1}}) {
        const auto base = layout.slice_base_row(KVKind::Value, 0, 0);
        const PIMCommand mac{PIMOpcode::MacAllBank,
                             encoder.encode({5, 0, 0, 0, 0, base + logical_row, 0, 0})};
        assert(encoder.decode(rows.translate(11, mac).address).row == base + expected_row);
    }

    const KVBlockShape full_shape({16, 32, 8, 128, 2});
    const RowAlignedKVLayout full_layout(full_shape, {2, 2, 4, 4, 16384, 32, 32});
    const AttAccAddressEncoder full_encoder(full_layout, 16);
    SharedKVRowSpace full_rows(full_layout, full_encoder, 16);
    assert(full_layout.slice_count() == 512); // 32 layers * 8 KV heads * K/V.
    assert(full_layout.rows_per_slice_capacity() == 32);
    assert(full_rows.chunks_per_row_stripe() == 512); 
    assert(full_rows.reserve(1,33));
    assert(full_rows.rows_allocated(1) == 1); // 512 1 MiB chunks per request.
    assert(full_rows.ensure_capacity(1, 4097)); // The first row stripe is already allocated, so growth adds a second stripe.
    assert(full_rows.rows_allocated(1) == 2);

    ChannelPhaseGuard guard(16);
    assert(guard.try_acquire(11,{2,5}));
    assert(!guard.try_acquire(22,{5,7}));
    assert(guard.try_acquire(22,{7}));
    guard.release(22,{7});
    guard.release(11,{2,5});

    rows.release(11);
    assert(rows.reserve(33,4097));
    assert(rows.rows_allocated(33) == 2);
    try {
        (void)rows.translate(11, sfm);
        assert(false && "released request retained its VA2PA mapping");
    } catch (const std::logic_error&) {}

    const RowAlignedKVLayout small_layout(shape, {2, 2, 4, 4, 4, 32, 32});
    const AttAccAddressEncoder small_encoder(small_layout, 16);
    SharedKVRowSpace small(small_layout, small_encoder, 16);
    assert(small.reserve(1, 4096));
    assert(small.reserve(2, 4096));
    assert(!small.reserve(3, 4096));
    assert(!small.ensure_capacity(1, 4097));
    assert(small.rows_allocated(1) == 1); // Failed growth is atomic.
    small.release(2);
    assert(small.ensure_capacity(1, 4097));

    std::cout << "PASS: 1 MiB per slice chunk, growth, reuse, 512-slice model\n";
}