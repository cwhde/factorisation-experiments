#!/bin/bash#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <integer>"
  exit 1
fi

n="$1"
if ! [[ $n =~ ^[0-9]+$ ]]; then
  echo "Error: argument must be a positive integer"
  exit 1
fi

warmup_runs=1000
measure_runs=10000

echo "Warming up ($warmup_runs runs)…"
for ((i=0; i < warmup_runs; i++)); do
  factor "$n" >/dev/null
done

echo "Measuring ($measure_runs runs)…"
perf stat -r "$measure_runs" factor "$n"