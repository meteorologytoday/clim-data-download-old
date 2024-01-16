#!/bin/bash

data_dir=data/ERAinterim/24hr
year_beg=1993
year_end=2017

output_dir=climatology_quantile_${year_beg}-${year_end}_ERAInterim

mkdir -p $output_dir

python3 make_AR_climatology.py \
    --input-dir $data_dir \
    --output-dir $output_dir \
    --filename-prefix "ERAInterim-" \
    --year-beg $year_beg \
    --year-end $year_end \
    --months 1 2 3 4 5 9 10 11 12 \
    --nproc 1 




