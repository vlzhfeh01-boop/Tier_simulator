#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "shared_channel_space.h"

using namespace Ramulator::TierRuntime;

int main() {
    const KVBlockShape shape({16,1,1,128,2});
    const RowAlignedKVLayout layout(shape,{2,2,4,4,16384,32,32});
    const AttAccAddressEncoder encoder(layout, 16);
    SharedKVRowSpace rows(layout,encoder,8);
    assert(rows.reserve(11) && rows.reserve(22));
    assert(rows.row_offset(11) == 0 && rows.row_offset(22) == 8);

    for(const KVKind kind : {KVKind::Key, KVKind::Value}) {
        const auto base = layout.slice_base_row(kind,0,0);
        const PIMCommand write{PIMOpcode::MemoryWrite,
                               encoder.encode({2,0,0,0,1,base+2,3,0})};
        const auto a = rows.translate(11,write);
        const auto b = rows.translate(22,write);
        assert(a.address != b.address);
        assert(encoder.decode(a.address).row == base + 2);
        assert(encoder.decode(b.address).row == base + 10);
        assert(encoder.decode(a.address).channel_id == 2);
        assert(encoder.decode(b.address).column == 3);

        const PIMCommand mac{PIMOpcode::MacAllBank,
                             encoder.encode({2, 0, 0, 0, 0, base + 2, 3, 0})};
        assert(encoder.decode(rows.translate(22, mac).address).row == base + 10);                             
    }
    const PIMCommand sfm{PIMOpcode::Softmax,
                         encoder.encode({2, 0, 0, 0, 0, 0, 0, 0})};
    assert(rows.translate(22, sfm).address == sfm.address);
    const PIMCommand illegal{PIMOpcode::MacAllBank,
                             encoder.encode({2, 0, 0, 0, 0, 8, 0, 0})};
    try {
        (void) rows.translate(22,illegal);
        assert(false && "a command exceeded its 8-row reservation");
    } catch (const std::out_of_range&) {}

    ChannelPhaseGuard guard(16);
    assert(guard.try_acquire(11, {2, 5}));
    assert(!guard.try_acquire(22, {5, 7}));
    assert(guard.try_acquire(22, {7})); // Failed acquisition did not claim 7.
    guard.release(22, {7});
    guard.release(11, {2, 5});
    assert(guard.try_acquire(22, {2, 5}));
    guard.release(22, {2, 5});

    rows.release(11);
    assert(rows.reserve(33) && rows.row_offset(33) == 0);
    try {
        (void)rows.translate(11, sfm);
        assert(false && "released request kept its address window");
    } catch (const std::logic_error&) {}
    const RowAlignedKVLayout small_layout(shape, {2, 2, 4, 4, 16, 32, 32});
    const AttAccAddressEncoder small_encoder(small_layout, 16);
    SharedKVRowSpace one_slot(small_layout, small_encoder, 8);
    assert(one_slot.reserve(1));
    assert(!one_slot.reserve(2));
    one_slot.release(1);
    assert(one_slot.reserve(2));
    std::cout << "PASS: shared K/V rows isolated, bounds checked, channel lease atomic\n";
}