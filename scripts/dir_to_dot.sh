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