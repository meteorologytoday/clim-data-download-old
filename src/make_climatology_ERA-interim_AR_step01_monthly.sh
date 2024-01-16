#!/bin/bash

data_dir=data/ERAinterim/24hr
year_beg=1993
year_end=2017

output_dir_ym=climatology_monthly_ERAInterim
output_dir=climatology_${year_beg}-${year_end}_ERAInterim

mkdir -p $output_dir_ym
mkdir -p $output_dir

    
for m in 1 2 3 4 5 9 10 11 12; do
    for y in $( seq $year_beg $year_end ); do

        y_str=$( printf "%04d" $y )
        m_str=$( printf "%02d" $m )

        output_filename=$output_dir_ym/ERAInterim_${y_str}-${m_str}.nc

        if [ -f "$output_filename" ]; then
            echo "File $output_filename already exists. Skip"
        else
            cmd="ncra -v IVT,IVT_x,IVT_y,IWV $data_dir/ERAInterim-${y_str}-${m_str}-??_00.nc $output_filename"
            echo ">> $cmd"
            eval $cmd
        fi


    done
    
    output_filename2=$output_dir/ERAInterim_climatology_monthly_${m_str}.nc

    if [ -f "$output_filename2" ]; then
        echo "File $output_filename2 already exists. Skip"
    else
        cmd="ncra -O $output_dir_ym/ERAInterim_{${year_beg}..${year_end}}-${m_str}.nc $output_filename2"
        echo ">> $cmd"
        eval $cmd
    fi
done




