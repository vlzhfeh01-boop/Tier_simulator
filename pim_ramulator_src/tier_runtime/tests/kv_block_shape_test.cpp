#include <cassert>
#include <iostream>
#include <stdexcept>

#include "tier_runtime/kv_block_shape.h"

using Ramulator::TierRuntime::KVBlockShape;
using Ramulator::TierRuntime::KVBlockShapeConfig;

int main() {
    KVBlockShapeConfig config {
        .tokens_per_block=16,
        .layers_on_hbm=2,
        .kv_heads_per_layer_on_hbm=3,
        .head_dimension=128,
        .bytes_per_element=2
    };

    KVBlockShape shape(config);

    assert(shape.tokens_per_block() == 16);
    assert(shape.layers_on_hbm() == 2);

    assert(shape.kv_heads_per_layer_on_hbm() ==3 );
    assert(shape.head_dimension() == 128);
    assert(shape.bytes_per_element() == 2);

    assert(shape.layer_head_count() == 6);
    assert(shape.elements_per_k_or_v_slice() == 2048);
    assert(shape.payload_bytes_per_k_or_v_slice() == 4096);
    assert(shape.k_payload_bytes_per_block() == 24576); // 하나의 블록당 전체 slice

    assert(shape.kv_payload_bytes_per_block() == 49152);
    assert(shape.kv_payload_bytes_per_token() == 3072);

    bool invalid_config_detected = false;

    try {
        KVBlockShapeConfig invalid_config {
            .tokens_per_block = 0,
            .layers_on_hbm = 2,
            .kv_heads_per_layer_on_hbm = 3,
            .head_dimension = 128,
            .bytes_per_element = 2
        };
        
        KVBlockShape invalid_shape(invalid_config);

        (void)invalid_shape;
    }   catch(const std::invalid_argument& ) {
        invalid_config_detected = true;
    }
    assert(invalid_config_detected);
    std::cout << "kv_block_shape_test passed\n";


    return 0;
}