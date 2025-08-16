#!/bin/bash

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_to_factor>" >&2
    exit 1
fi

# The first command-line argument is the number to factor
number_to_factor="$1"

# Use hyperfine to benchmark gfactor with the provided number
hyperfine --warmup 1000 --runs 10000 "gfactor $number_to_factor"
