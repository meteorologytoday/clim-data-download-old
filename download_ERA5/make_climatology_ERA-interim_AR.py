
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


def doJob(target_ym):

    try:


        


        if target_date.month == 2 and target_date.day == 29:
            print("Feb 29 encountered skip.")
            return

        # Decide the files to average
        
        filenames = [
            "%s/%s%04d%s.nc" % (
                args.input_dir,
                args.filename_prefix,
                yr,
                target_date.strftime("-%m-%d"),
            )

            for yr in range(args.beg_year, args.end_year+1)
        ]

     
        output_filename_tmp = "%s/%s%s.tmp.nc" % (
            args.output_dir,
            args.filename_prefix,
            target_date.strftime("%m-%d"),
        )
       
        output_filename = "%s/%s%s.nc" % (
            args.output_dir,
            args.filename_prefix,
            target_date.strftime("%m-%d"),
        )
     

        if os.path.isfile(output_filename):

            print("[%s] File %s already exists, skip." % (target_date.strftime("%m-%d"), output_filename,))

        else:
            
            print("[%s] Making file %s..." % (target_date.strftime("%m-%d"), output_filename,))
       
            dir_name = os.path.dirname(output_filename) 
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
            # average
            pleaseRun("ncra -O %s %s" % (
                " ".join(filenames),
                output_filename_tmp,
            ))

            ds = xr.open_dataset(output_filename_tmp)
            ds = ds.assign_coords(coords=dict(time=[target_date,]))
            ds.to_netcdf(
                output_filename,
                unlimited_dims=["time",],
                encoding={'time': {'dtype': 'i4'}},
            )

            pleaseRun("rm %s" % output_filename_tmp)


        return output_filename
        # done


    except Exception as e:

        print("[%s] Error. Now print stacktrace..." % (target_date.strftime("%m-%d")))
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
