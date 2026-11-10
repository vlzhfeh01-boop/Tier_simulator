#!/usr/bin/env bash
set -eu

# Apply the overlay only to the pinned, clean submodule checkout.
if [ "$(git -C ramulator2 rev-parse HEAD)" != "b7c70275f04126c647edb989270cc429776955d1" ] ||
   [ -n "$(git -C ramulator2 status --porcelain)" ]; then
    echo "ramulator2 must be a clean checkout of b7c7027" >&2
    exit 1
fi

# The original request header has CRLF, while its enum patch expects LF.
python3 - <<'PY'
from pathlib import Path
path = Path('ramulator2/src/base/request.h')
path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
PY

# copy files
cp pim_ramulator_src/attacc_bank.yaml ramulator2/
cp pim_ramulator_src/attacc_bg.yaml ramulator2/
cp pim_ramulator_src/attacc_buffer.yaml ramulator2/
cp pim_ramulator_src/HBM3_base.yaml ramulator2/
cp pim_ramulator_src/hbm3_linear_mappers.cpp ramulator2/src/addr_mapper/impl/
cp pim_ramulator_src/hbm3_pim_linear_mappers.cpp ramulator2/src/addr_mapper/impl/
cp pim_ramulator_src/HBM3-PIM.cpp ramulator2/src/dram/impl/
cp pim_ramulator_src/hbm3_controller.cpp ramulator2/src/dram_controller/impl/
cp pim_ramulator_src/hbm3_pim_controller.cpp ramulator2/src/dram_controller/impl/
cp pim_ramulator_src/hbm3_trace_recorder.cpp ramulator2/src/dram_controller/impl/plugin/
cp pim_ramulator_src/all_bank_refresh_hbm3.cpp ramulator2/src/dram_controller/impl/refresh/
cp pim_ramulator_src/no_refresh.cpp ramulator2/src/dram_controller/impl/refresh/
cp pim_ramulator_src/pim_scheduler.cpp ramulator2/src/dram_controller/impl/scheduler/
cp pim_ramulator_src/pim_loadstore_trace.cpp ramulator2/src/frontend/impl/memory_trace/
cp pim_ramulator_src/tier_decode_step_frontend.cpp ramulator2/src/frontend/impl/memory_trace/
cp -r pim_ramulator_src/tier_runtime ramulator2/src/
cp pim_ramulator_src/tier_decode_step.yaml ramulator2/
cp pim_ramulator_src/tier_decode_workload.yaml ramulator2/
cp pim_ramulator_src/tier_decode_workload.csv ramulator2/
cp pim_ramulator_src/tier_shared_workload.yaml ramulator2/
cp pim_ramulator_src/tier_shared_workload.csv ramulator2/
cp pim_ramulator_src/PIM_DRAM_system.cpp ramulator2/src/memory_system/impl/
cp -r pim_ramulator_src/trace_gen ramulator2/
cp -r pim_ramulator_src/patches ramulator2/

# Apply patches
cd ramulator2

for f in ./patches/*.patch
do
    if [ "$(basename "$f")" = src_frontend_tier_decode.patch ]; then
        python3 - <<'PY'
from pathlib import Path
path = Path('src/frontend/CMakeLists.txt')
path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n').rstrip(b'\n') + b'\n')
PY
    fi
    patch --batch --forward -p1 < "$f"
done
