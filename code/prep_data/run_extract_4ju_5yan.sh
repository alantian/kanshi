#!/bin/bash

python3 ./extract_ju.py \
  --input_file "../../data/data/全唐詩.txt" \
  --output_file "../../data/data/四句五言.txt" \
  --nb_ju 4 \
  --force_5yan=true \
  ;
