#!/usr/bin/env bash

# generate output.lst file
cat ../c3d_finetuning/test_01.lst | sed -e 's/\/media\/6TB\/Videos\/UCF-101/output/' | awk '{printf("%s%05d %d\n", $1, $2, $3)}' > output.lst

# mkdir
cat output.lst  | awk '{print "./" $1}' | sed -e 's/\/[0-9]\{5,5\}$//' | sort -n | xargs -I {} mkdir -p {}
