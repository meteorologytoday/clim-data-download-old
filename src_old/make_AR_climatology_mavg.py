
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
parser.add_argument('--mavg-days', help="Moving average days, can only be an odd number.", type=int, default="15")
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)

years = np.arange(args.year_beg, args.year_end+1, dtype=int)


if args.mavg_days % 2 != 1:
    raise Exception("`--mavg-days` should be an odd number.")



def work(dt, detect_phase=False):

    result = dict(status="UNKNOWN", dt=dt, output_filename="", detect_phase=detect_phase)

    try:

            
        isfeb29 = dt.month == 2 and dt.day == 29

        mmddhh_str = dt.strftime("%m-%d_%H")
        output_filename = os.path.join(
            args.output_dir,
            "{output_filename_prefix:s}{mmddhh:s}.nc".format(
                output_filename_prefix = args.output_filename_prefix,
                mmddhh = mmddhh_str,
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

            mavg_days_arm_window = (args.mavg_days - 1) / 2

            for y in years:
                
                _dt_leap_test = pd.Timestamp(year=y, month=1, day=1)
                if isfeb29 and ( not _dt_leap_test.is_leap_year ):
                    # Use Feb28 if the selected year is not a leap year
                    dt_mid = pd.Timestamp(year=y, month=2, day=28)
                else:
                    dt_mid = pd.Timestamp(year=y, month=dt.month, day=dt.day)


                dt_beg = dt_mid - pd.Timedelta(days=mavg_days_arm_window)
                dt_end = dt_mid + pd.Timedelta(days=mavg_days_arm_window)
               
                print("[%s] Loading date range: %s ~ %s" % (mmddhh_str, dt_beg.strftime("%Y-%m-%d"), dt_end.strftime("%Y-%m-%d")))
 
                for _dt in pd.date_range(dt_beg, dt_end, freq="D", inclusive="both"):
                    
                    filename = os.path.join(args.input_dir, "{input_filename_prefix:s}{datetime_str:s}".format(
                        input_filename_prefix = args.input_filename_prefix,
                        datetime_str    = _dt.strftime("%Y-%m-%d_00.nc"),
                    ))
                    
                    load_files.append(filename)
            
            print("[%s] Loading %d files..." % (mmddhh_str, len(load_files),))
            ds = xr.open_mfdataset(load_files)
            ds = ds.chunk({"time": -1, "latitude": "auto", "longitude": "auto"})
           
            print(ds)

                
            merged_data = []

            if args.method == "q85":

                quantile = .85
                for varname in ["IVT", ]:
                    _data = ds[varname].quantile(quantile, keep_attrs=True, skipna=False, dim="time")
                    _data = _data.expand_dims(
                        dim={"time": [dt,]},
                        axis=0,
                    )
                    merged_data.append(_data)

            elif args.method == "mean":
 
                for varname in ["IVT", ]:
                    _data = ds[varname].mean(keep_attrs=True, skipna=False, dim="time")
                    _data = _data.expand_dims(
                        dim={"time": [dt,]},
                        axis=0,
                    )
                    merged_data.append(_data)

                 
            print("[%s] Merging data..." % (mmddhh_str,))

            ds_new = xr.merge(merged_data).rename_dims(dict(latitude="lat", longitude="lon"))
            ds_new.attrs["method"] = args.method

            print("[%s] Outputting file: %s" % (mmddhh_str, output_filename,))
            ds_new.to_netcdf(
                output_filename,
                unlimited_dims="time",
            )

        ds.close()
        ds_new.close()
        
        result["status"] = "OK"

    except Exception as e:

        print("[%s] Error. Now print stacktrace..." % (mmddhh_str,))
        import traceback
        traceback.print_exc()
        
        result["status"] = "ERROR"

    return result

def ifSkip(dt):

    if dt.month in [5,6,7,8]:
        return True

    return False

failed_dates = []
dts = pd.date_range("2000-01-01", "2001-01-01", freq="D", inclusive="left")

input_args = []

for dt in dts:

    dt_str = dt.strftime("%m-%d")


    if ifSkip(dt):
        print("Skip : %s" % (dt_str,))
        continue

    print("Detect date: ", dt_str)
    result = work(dt, detect_phase=True)
    
    if result['status'] != 'OK':
        print("[detect] Failed to detect month %d " % (month,))
    
    else:
        if result['need_work'] is True:
            print("[detect] Need to work on %s." % (dt_str,))
            input_args.append((dt,))
        else:
            print("[detect] Files all exist for month =  %s." % (dt_str,))




print("Ready to do work...")
with Pool(processes=args.nproc) as pool:

    results = pool.starmap(work, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('!!! Failed to generate output of month %s.' % (result['dt'].strftime("%m-%d"),))
            failed_dates.append(result['dt'])


print("Tasks finished.")

if len(failed_dates) == 0:
    print("Congratualations! All dates are successfully produced.")    
else:
    print("Failed dates: ")
    for i, failed_date in enumerate(failed_dates):
        print("[%d] %s" % (i, failed_date.strftime("%m-%d"),) )

print("Done.")       
