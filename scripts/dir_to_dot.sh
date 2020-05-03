#!/bin/bash

INP_DIR=$1
OUT_DIR=$2
for x in `ls $INP_DIR/*/*.dax`; do 
    echo $x;
    filename=$(basename $x)
    filename="${filename//\.dax/.dot}"
    result_filepath="$OUT_DIR/$filename"
    python3.6 ./source/tools/dax_to_dot.py "$x" "$result_filepath"
done;


  python3.6 -m pysimgrid.tools.dag_gen \
        -n 5 10 --fat 0.5 0.8 --density 0.1 --jump 2 \
        --mindata 1e6 --maxdata 1e9 --ccr 10 100 \
        --repeat 5 --seed 42 \
        "data/workflows/random/"