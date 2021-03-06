#!/bin/bash

data_signature=4ju5yan
model=rnnlm
hidden_size=300
n_layers=4
dropout=0.1

host="0.0.0.0"
port="8002"
sample_t_low=2.7
sample_t_high=3.2

data_dir="../../scratch/crnnlm/data"
vocab_file="$data_dir/$data_signature-vocab.txt"
data_file="$data_dir/$data_signature-data.h5"
signature="model-$model-data_signature-$data_signature-hidden_size-$hidden_size-n_layers-$n_layers-dropout-$dropout-dev"
save_dir="../../scratch/crnnlm/model/$signature/"
buffer_size=20

echo "use $save_dir"



python3 ./webapi.py \
  --vocab_file "$vocab_file" \
  --model $model \
  --save_dir "$save_dir" \
  --hidden_size "$hidden_size" \
  --n_layers "$n_layers" \
  --dropout "$dropout" \
  --host "$host" \
  --port "$port" \
  --buffer_size "$buffer_size" \
  --sample_t_low "$sample_t_low" \
  --sample_t_high "$sample_t_high" \
  "$@" \
  ;
