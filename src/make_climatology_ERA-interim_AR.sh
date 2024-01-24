#!/bin/bash

data_dir=data/ERAinterim/AR/24hr
year_beg=1993
year_end=2017


for method in mean q85 ; do
    
    ndays=15
    output_dir=climatology_${year_beg}-${year_end}_${ndays}days_ERAInterim_${method}
    mkdir -p $output_dir

    python3 make_AR_climatology_mavg.py \
        --input-dir $data_dir \
        --output-dir $output_dir \
        --input-filename-prefix "ERAInterim-" \
        --output-filename-prefix "ERAInterim-clim-daily_" \
        --year-beg $year_beg \
        --year-end $year_end \
        --mavg-days $ndays  \
        --method "$method" \
        --nproc 5 

done

if [ ] ; then
for method in mean q85 ; do

    output_dir=climatology_${year_beg}-${year_end}_ERAInterim_${method}
    mkdir -p $output_dir

    python3 make_AR_climatology_monthly.py \
        --input-dir $data_dir \
        --output-dir $output_dir \
        --input-filename-prefix "ERAInterim-" \
        --year-beg $year_beg \
        --year-end $year_end \
        --method "$method" \
        --months 1 2 3 4  9 10 11 12 \
        --nproc 1 

    python3 make_AR_climatology_daily_from_monthly.py \
        --input-dir $output_dir \
        --output-dir $output_dir \
        --output-filename-prefix "ERAInterim-clim-daily_" \
        --nproc 1 

done
fi
