#!/bin/bash

dropbox="../../dropbox-local/"

data_dir="$dropbox/scratch/cvae/data"
data_signature=4ju
vocab_file="$data_dir/$data_signature-vocab.txt"
data_file="$data_dir/$data_signature-data.h5"
model=rnnlm
signature="model-$model-data_signature-$data_signature-dev"
save_dir="$dropbox/scratch/cvae/model/$signature/"

mkdir -p $save_dir

python3 ./train.py \
  --vocab_file "$vocab_file" \
  --data_file "$data_file" \
  --model $model \
  --save_dir "$save_dir"\
  ;
