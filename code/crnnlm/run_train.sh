#!/bin/bash

data_signature=4ju5yan
model=rnnlm
hidden_size=300
n_layers=4
dropout=0.1

gpu_id_list="0"

data_dir="../../scratch/crnnlm/data"
vocab_file="$data_dir/$data_signature-vocab.txt"
data_file="$data_dir/$data_signature-data.h5"
signature="model-$model-data_signature-$data_signature-hidden_size-$hidden_size-n_layers-$n_layers-dropout-$dropout-dev"
save_dir="../../scratch/crnnlm/model/$signature/"

mkdir -p $save_dir

echo "save to $save_dir"

python3 ./train.py \
  --vocab_file "$vocab_file" \
  --data_file "$data_file" \
  --model $model \
  --save_dir "$save_dir" \
  --hidden_size "$hidden_size" \
  --n_layers "$n_layers" \
  --dropout "$dropout" \
  --gpu_id_list "$gpu_id_list" \
  "$@" \
  ;
