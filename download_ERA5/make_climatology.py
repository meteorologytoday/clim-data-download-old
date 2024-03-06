
from multiprocessing import Pool
import xarray as xr
import numpy as np
import argparse
import pandas as pd
import os.path
import os
from pathlib import Path


def pleaseRun(cmd):
    print(">> %s" % cmd)
    os.system(cmd)


parser = argparse.ArgumentParser(
                    prog = 'make_climatology.py',
                    description = 'Use ncra to do daily climatology',
)
parser.add_argument('--input-dir', required=True, help="Input directory.", type=str)
parser.add_argument('--output-dir', required=True, help="Output directory.", type=str)
parser.add_argument('--filename-prefix', required=True, help="The prefix of filename.", type=str)
parser.add_argument('--beg-year', required=True, help="The begin year.", type=int)
parser.add_argument('--end-year', required=True, help="The end year.", type=int)
parser.add_argument('--nproc', type=int, default=2)
args = parser.parse_args()
print(args)



def work(month):

    result = dict(status="UNKNOWN", month=month)

    try:

         output_filename = os.path.join(
            args.output_dir,
            "{output_filename_prefix:s}-{month:02d}.nc".format(
                output_filename_prefix = "climatology_monthly",
                month = month,
            )
        )

        if os.path.isfile(output_filename):
            print("[Month %s] File %s already exists, skip." % (target_date.strftime("%m"), output_filename,))

        else:

            # Create output directory if not exists
            dir_name = os.path.dirname(output_filename) 
            Path(dir_name).mkdir(parents=True, exist_ok=True)
                
            load_files = []

            for y in years:

                dt_beg = pd.Timestamp(year=y, month=month, day=1)
                dt_end = dt_beg + DateOffset(months=1)

                for dt in pd.date_range(dt_beg, dt_end, freq="D", inclusive="left"):

                    filename = path.os.join(args.input_dir, "{filename_prefix:s}-{datetime_str:s}".format(
                        filename_prefix = args.filename_prefix,
                        datetime_str    = dt.strftime("%Y-%m_00.nc"),
                    ))

                    load_files.append(filename)

            print("[month=%d] Loading %d files..." % (month, len(load_files),))
            ds = xr.load_mfdataset(load_files)
            
            quantiles = np.array([.85,])
            
            merged_data = []
            for varname in ["IVT", ]:
                merged_data.append(ds[varname].quantile(quantiles), keep_attrs=True, skipna=False, )
                
            print("[month=%d] Merging data..." % (month,))
            ds_new = xr.merge(merged_data)

            print("[month=%d] Outputting file: %s" % (output_filename,))
            ds_new.to_netcdf(
                output_filename,
            )

        result["status"] = "OK"

    except Exception as e:

        print("[month=%d] Error. Now print stacktrace..." % (month,))
        import traceback
        traceback.print_exc()

        return None


with Pool(processes=args.nproc) as pool:

    dts = pd.date_range('2020-01-01', '2020-12-31', freq="D", inclusive="both")
    
    it = pool.imap(doJob, dts)


    for result in it:

        print("Task for file %s compelete." % result)


    try:

        feb28 = "%s/%s%s.nc" % (
            args.output_dir,
            args.filename_prefix,
            "02-28",
        )
     
        feb29 = "%s/%s%s.nc" % (
            args.output_dir,
            args.filename_prefix,
            "02-29",
        )
     
        mar01 = "%s/%s%s.nc" % (
            args.output_dir,
            args.filename_prefix,
            "03-01",
        )
     

        pleaseRun("ncra -O %s %s %s" % (feb28, mar01, feb29))

    except Exception as e:

        print("Error. Cannot make Feb 29 data. Now print stacktrace...")
        import traceback
        traceback.print_exc()



print("Done.")
