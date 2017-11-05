#!/bin/bash

input_data_dir="../../data/data"
output_data_dir="../../scratch/crnnlm/data"

mkdir -p $output_data_dir

python3 ./make_data.py \
  --input_file "$input_data_dir/二句.txt"\
  --vocab_file "$output_data_dir/2ju-vocab.txt" \
  --data_file "$output_data_dir/2ju-data.h5" \
  ;

python3 ./make_data.py \
  --input_file "$input_data_dir/四句.txt"\
  --vocab_file "$output_data_dir/4ju-vocab.txt" \
  --data_file "$output_data_dir/4ju-data.h5" \
  ;
