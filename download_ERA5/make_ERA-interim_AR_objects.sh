#!/bin/bash

data_dir=data/ERAinterim/AR/24hr
year_beg=1992
year_end=2018



for method in ANOMLEN ANOMLEN2 ANOMLEN3 ANOMLEN4; do

    output_dir=data/ERAinterim/ARObjects/${method}

    mkdir -p $output_dir

    clim_dir_suffix="NONE"
    if [ "$method" = "ANOMLEN" ]; then
        clim_dir_suffix=mean
    elif [ "$method" = "ANOMLEN2" ]; then
        clim_dir_suffix=q85
    elif [ "$method" = "ANOMLEN3" ]; then
        clim_dir_suffix=q85
    elif [ "$method" = "ANOMLEN4" ]; then
        clim_dir_suffix=q85
    else
        echo "Error: unknown method: `$method`"
        exit 1
    fi

    input_clim_dir="climatology_1993-2017_15days_ERAInterim_${clim_dir_suffix}"
    

    python3 make_ERA-interim_AR_objects.py \
        --input-dir  $data_dir \
        --output-dir $output_dir \
        --input-file-prefix "ERAInterim-" \
        --input-clim-dir         $input_clim_dir \
        --input-clim-file-prefix "ERAInterim-clim-daily_" \
        --method $method \
        --nproc 5 & 

done

wait
