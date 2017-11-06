#!/bin/bash

mkdir -p scratch_pack
rm -rf scratch_pack/*
mkdir -p scratch_pack/crnnlm
cp -r scratch/crnnlm/data scratch_pack/crnnlm/
model_sig="model-rnnlm-data_signature-4ju-hidden_size-300-n_layers-4-dropout-0.1-dev"
mkdir -p scratch_pack/crnnlm/model/"$model_sig"
cp scratch/crnnlm/model/"$model_sig"/*latest scratch_pack/crnnlm/model/"$model_sig"
tar -cvf ./scratch_pack.tar scratch_pack
rm -rf scratch_pack
