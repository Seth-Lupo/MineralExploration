#!/bin/sh
# Runs every policy / batch size on the 100 shared cases, 4 jobs at a time
# (each worker peaks near 3 GB at 10,000 tree queries).
# Results land in data/batch/; plot with scripts/plot_batch.py.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
julia --project=. scripts/run_batch_test.jl pomcpow 1 1 0  # generate shared cases
for cfg in "pomcpow 8" "pomcpow 4" "pomcpow 2" "pomcpow 1" "grid16 1" "grid9 1"; do
    for first in 1 11 21 31 41 51 61 71 81 91; do
        echo "$cfg $first $((first + 9))"
    done
done | xargs -P "${JOBS:-4}" -L 1 sh -c 'julia --project=. scripts/run_batch_test.jl $0 $1 $2 $3 > /dev/null 2>&1'
