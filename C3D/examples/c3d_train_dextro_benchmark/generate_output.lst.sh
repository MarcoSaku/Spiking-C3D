#!/usr/bin/env bash

# generate output.lst file
cat dextro_benchmark_val_flow_smaller.txt| sed -e 's/\/media\/6TB\/Videos\/dextro\-benchmark/output/' | awk '{printf("%s_%05d\n", substr($1,0,length($1)-4), $2)}' > output.lst

# mkdir
#cat output.lst  | awk '{print "./" $1}' | sed -e 's/\/[0-9]\{5,5\}$//' | sort -n | xargs -I {} mkdir -p {}
mkdir output
