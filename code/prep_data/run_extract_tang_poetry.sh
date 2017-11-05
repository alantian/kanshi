#!/bin/bash

python3 ./extract_tang_poetry.py \
  --input_files_pattern "../../data/corpus/全唐詩/卷*.txt" \
  --output_file "../../data/data/全唐詩.txt" \
  ;
