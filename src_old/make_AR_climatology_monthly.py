
from multiprocessing import Pool
import xarray as xr
import numpy as np
import argparse
import pandas as pd
import os.path
import os
from pathlib import Path


parser = argparse.ArgumentParser(
                    prog = 'make_climatology.py',
                    description = 'Use ncra to do daily climatology',
)
parser.add_argument('--input-dir', required=True, help="Input directory.", type=str)
parser.add_argument('--output-dir', required=True, help="Output directory.", type=str)
parser.add_argument('--input-filename-prefix', required=True, help="The prefix of filename.", type=str)
parser.add_argument('--output-filename-prefix', help="The prefix of filename.", type=str, default="climatology_monthly-")
parser.add_argument('--method', required=True, help="The prefix of filename.", type=str, choices=["q85", "mean"])
parser.add_argument('--year-beg', required=True, help="The begin year.", type=int)
parser.add_argument('--year-end', required=True, help="The end year.", type=int)
parser.add_argument('--months', help="The end year.", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10,11,12])
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)

years = np.arange(args.year_beg, args.year_end+1, dtype=int)



def work(month, detect_phase=False):

    result = dict(status="UNKNOWN", month=month, output_filename="", detect_phase=detect_phase)

    try:

        output_filename = os.path.join(
            args.output_dir,
            "{output_filename_prefix:s}{month:02d}.nc".format(
                output_filename_prefix = args.output_filename_prefix,
                month = month,
            )
        )
        result["output_filename"] = output_filename

        if detect_phase:
            
            result["need_work"] = not os.path.exists(output_filename)
            result["status"] = "OK"
            
            return result
 

        else:

            # Create output directory if not exists
            dir_name = os.path.dirname(output_filename) 
            Path(dir_name).mkdir(parents=True, exist_ok=True)
                
            load_files = []

            for y in years:

                dt_beg = pd.Timestamp(year=y, month=month, day=1)
                dt_end = dt_beg + pd.tseries.offsets.DateOffset(months=1)

                for dt in pd.date_range(dt_beg, dt_end, freq="D", inclusive="left"):

                    filename = os.path.join(args.input_dir, "{input_filename_prefix:s}{datetime_str:s}".format(
                        input_filename_prefix = args.input_filename_prefix,
                        datetime_str    = dt.strftime("%Y-%m-%d_00.nc"),
                    ))

                    load_files.append(filename)

            print("[month=%d] Loading %d files..." % (month, len(load_files),))
            ds = xr.open_mfdataset(load_files)
            ds = ds.chunk({"time": -1, "latitude": "auto", "longitude": "auto"})
           
            print(ds)

                
            merged_data = []

            if args.method == "q85":

                quantile = .85
                for varname in ["IVT", ]:
                    _data = ds[varname].quantile(quantile, keep_attrs=True, skipna=False, dim="time")
                    _data = _data.expand_dims(
                        dim={"month": [month,]},
                        axis=0,
                    )
                    merged_data.append(_data)

            elif args.method == "mean":
 
                for varname in ["IVT", ]:
                    _data = ds[varname].mean(keep_attrs=True, skipna=False, dim="time")
                    _data = _data.expand_dims(
                        dim={"month": [month,]},
                        axis=0,
                    )
                    merged_data.append(_data)

                 
            print("[month=%d] Merging data..." % (month,))

            ds_new = xr.merge(merged_data).rename_dims(dict(latitude="lat", longitude="lon"))
            ds_new.attrs["method"] = args.method

            print("[month=%d] Outputting file: %s" % (month, output_filename,))
            ds_new.to_netcdf(
                output_filename,
                unlimited_dims="month",
            )

        ds.close()
        ds_new.close()
        
        result["status"] = "OK"

    except Exception as e:

        print("[month=%d] Error. Now print stacktrace..." % (month,))
        import traceback
        traceback.print_exc()
        
        result["status"] = "ERROR"

    return result

months = np.array(args.months)
failed_months = []

input_args = []
for month in months:

    print("Detect month: ", month)
    result = work(month, detect_phase=True)
    
    if result['status'] != 'OK':
        print("[detect] Failed to detect month %d " % (month,))
    
    else:
        if result['need_work'] is True:
            print("[detect] Need to work on file: ", result["output_filename"])
            input_args.append((month,))
        else:
            print("[detect] Files all exist for month =  %d." % (month,))


print("Ready to do work...")
        
with Pool(processes=args.nproc) as pool:

    results = pool.starmap(work, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('!!! Failed to generate output of month %d.' % (result['month'],))
            failed_months.append(result['month'])


print("Tasks finished.")

if len(failed_months) == 0:
    print("Congratualations! All months are successfully produced.")    
else:
    print("Failed months: ", failed_months)

print("Done.")
