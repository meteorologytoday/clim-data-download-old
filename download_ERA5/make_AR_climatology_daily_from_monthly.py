
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
parser.add_argument('--input-filename-prefix', help="The prefix of filename.", type=str, default="climatology_monthly-")
parser.add_argument('--output-filename-prefix', help="The prefix of filename.", type=str, default="climatology_daily-")

parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)

ref_dt = pd.Timestamp("2000-01-01") # leap year

def getTimedeltaOfAMonth(dt):
    dt1 = pd.Timestamp(year=dt.year, month=dt.month, day=1)
    dt2 = dt1 + pd.DateOffset(months=1)
    return dt2 - dt1

def getMidDay(dt):
    return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day, hour=12)
 

def getMidMonth(dt):
    
    dt1 = pd.Timestamp(year=dt.year, month=dt.month, day=1)
    dt2 = dt1 + pd.DateOffset(months=1)

    mid_dt = dt1 + (dt2 - dt1) * 0.5
        
    return mid_dt

# dt1 - dt2
def timeDiff(dt1, dt2):
    
    _dt1 = pd.Timestamp(year=ref_dt.year, month=dt1.month, day=dt1.day, hour=dt1.hour)
    _dt2 = pd.Timestamp(year=ref_dt.year, month=dt2.month, day=dt2.day, hour=dt2.hour)
    
    if _dt1 < _dt2:
        _dt1 = _dt1 + pd.DateOffset(years=1)
    
    return _dt1 - _dt2




def nextNmonths(m, shift):
    return ((m-1)+shift) % 12 + 1


def work(dt, detect_phase=False):

    result = dict(status="UNKNOWN", output_filename="", detect_phase=detect_phase, dt=dt)
    dt_str = dt.strftime("%m-%d_%H")
    try:

        output_filename = os.path.join(
            args.output_dir,
            "{filename_prefix:s}{date_str}.nc".format(
                filename_prefix = args.output_filename_prefix,
                date_str = dt_str,
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


            # First, determine which months are used to interpolate            
            days_in_month = getTimedeltaOfAMonth(pd.Timestamp(year=ref_dt.year, month=dt.month, day=1)) / pd.Timedelta(days=1)

            day_of_month = (dt.day - 1) + 0.5
             
            if day_of_month < days_in_month / 2:
                ref_month_left  = nextNmonths(dt.month, -1) 
                ref_month_right = dt.month

            else:
                ref_month_left = dt.month
                ref_month_right = nextNmonths(dt.month, 1) 


            ref_dt_left  = getMidMonth(pd.Timestamp(year=ref_dt.year, month=ref_month_left, day=1))
            ref_dt_right = getMidMonth(pd.Timestamp(year=ref_dt.year, month=ref_month_right, day=1))
            
            load_files = [
                os.path.join(
                    args.input_dir,
                    "{filename_prefix:s}{month:02d}.nc".format(
                        filename_prefix = args.input_filename_prefix,
                        month = m,
                    )
                )

                for m in [ref_month_left, ref_month_right]
            ]

            print("[%s] Load files: " % (dt_str, ), load_files )
            ds = xr.open_mfdataset(load_files).rename_dims(dict(month="time")).rename(dict(month="time"))

                
            # Data is sorted when loading
            if ref_month_left == 12:
                ds = ds.isel(time = slice(None, None, -1))
            
            print("ds.coords['time'] = ", ds.coords["time"].to_numpy()) 

            ds = ds.assign_coords(time = [0.0, 1.0])

            print("[%s] left_dt: %s, mid_dt: %s, right_dt: %s. time=%.5f" % (
                dt_str,
                ref_dt_left.strftime("%m-%d_%H"),
                getMidDay(dt).strftime("%m-%d_%H"),
                ref_dt_right.strftime("%m-%d_%H"),
                timeDiff(getMidDay(dt), ref_dt_left) / timeDiff(ref_dt_right, ref_dt_left),
            ))

            ds = ds.interp(
                time = [timeDiff(getMidDay(dt), ref_dt_left) / timeDiff(ref_dt_right, ref_dt_left),]
            ).assign_coords(
                time = [dt,],
            )

            print("[%s] Outputting file: %s" % (dt_str, output_filename,))
            ds.to_netcdf(
                output_filename,
                unlimited_dims="time",
                encoding={'time':{'units':'hours since 1970-01-01 00:00:00'}}
            )


        ds.close()
        
        result["status"] = "OK"

    except Exception as e:

        print("[%s] Error. Now print stacktrace..." % (dt_str,))
        import traceback
        traceback.print_exc()
        
        result["status"] = "ERROR"

    return result


def ifSkip(dt):

    if dt.month in [4,5,6,7,8,9]:
        return True

    return False

failed_dates = []

input_args = []

dts = pd.date_range("2000-01-01", "2001-01-01", freq="D", inclusive="left")

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
