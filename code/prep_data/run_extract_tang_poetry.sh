#!/bin/bash

python3 ./extract_tang_poetry.py \
  --input_files_pattern "../../dropbox/original-corpus/全唐詩/卷*.txt" \
  --output_file "../../dropbox/data/全唐詩.txt" \
  ;
