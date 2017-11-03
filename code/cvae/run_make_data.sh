#!/bin/bash

dropbox="../../dropbox/"

scratch_dir="$dropbox/scratch/cvae/data/"

mkdir -p $scratch_dir

python3 ./make_data.py \
  --input_file "$dropbox/data/二句.txt"\
  --vocab_file "$scratch_dir/2ju-vocab.txt" \
  --data_file "$scratch_dir/2ju-data.h5" \
  ;

python3 ./make_data.py \
  --input_file "$dropbox/data/四句.txt"\
  --vocab_file "$scratch_dir/4ju-vocab.txt" \
  --data_file "$scratch_dir/4ju-data.h5" \
  ;
