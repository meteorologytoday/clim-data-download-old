with open("shared_header.py", "rb") as source_file:
    code = compile(source_file.read(), "shared_header.py", "exec")
exec(code)

import ARdetection_Tien
import numpy as np
import xarray as xr
import pandas as pd
import argparse
from earth_constants import r_E

parser = argparse.ArgumentParser(
                    prog = 'make_ERA5_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

parser.add_argument('--input-dir', required=True, type=str)
parser.add_argument('--output-dir', required=True, type=str)
parser.add_argument('--input-file-prefix', type=str, required=True)
parser.add_argument('--input-clim-dir', required=True, type=str)
parser.add_argument('--input-clim-file-prefix', type=str, required=True)
parser.add_argument('--output-file-prefix', type=str, default="ARobjs_")
parser.add_argument('--method', required=True, type=str, choices=["ANOMLEN", "ANOMLEN2", "ANOMLEN3", "ANOMLEN4"])
parser.add_argument('--leftmost-lon', type=float, help='The leftmost longitude on the map. It matters when doing object detection finding connectedness.', default=0)
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)


input_dir = args.input_dir
output_dir = args.output_dir
input_file_prefix = args.input_file_prefix
input_clim_file_prefix = args.input_clim_file_prefix
output_file_prefix = args.output_file_prefix

beg_time_str = beg_time.strftime("%Y-%m-%d")
end_time_str = end_time.strftime("%Y-%m-%d")

def doJob(dt, detect_phase=False):

    jobname = dt.strftime("%Y-%m-%d")
    result = dict(status="UNKNOWN", dt=dt, detect_phase=detect_phase)

    try: 
        
        # Load file
        AR_clim_file = "%s/%s%s.nc" % (args.input_clim_dir, input_clim_file_prefix, dt.strftime("%m-%d_%H"))
        AR_full_file = "%s/%s%s.nc" % (input_dir,        input_file_prefix,      dt.strftime("%Y-%m-%d_%H"))
        output_file  = "%s/%s%s.nc"  % (output_dir,      output_file_prefix,     dt.strftime("%Y-%m-%d_%H"))

        # Test if file exists
        if detect_phase == True:

            result["need_work"] = not os.path.isfile(output_file)
            result["status"] = "OK"
            return result

        
        print("[%s] Making %s with input files: %s, %s." % (jobname, output_file, AR_full_file, AR_clim_file))

        # Make output dir
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        # Load anom
        ds_full = xr.open_dataset(AR_full_file)
        ds_clim = xr.open_dataset(AR_clim_file)
        
        old_lon = ds_full.coords["lon"].to_numpy()

        # find the lon = args.leftmost_lon
        lon_first_zero = np.argmax(ds_full.coords["lon"].to_numpy() >= args.leftmost_lon)

        ds_full = ds_full.roll(lon=-lon_first_zero, roll_coords=True)
        ds_clim = ds_clim.roll(lon=-lon_first_zero, roll_coords=True)
        
        lat = ds_full.coords["lat"].to_numpy() 
        lon = old_lon  % 360
      
        # For some reason we need to reassign it otherwise the contourf will be broken... why??? 
        ds_full = ds_full.assign_coords(lon=lon) 
        ds_clim = ds_clim.assign_coords(lon=lon) 
        
        IVT_x = ds_full.IVT_x[0, :, :].to_numpy()
        IVT_y = ds_full.IVT_y[0, :, :].to_numpy()
        IVT = np.sqrt(IVT_x**2 + IVT_y**2)
        IVT_clim = ds_clim.IVT[0, :, :].to_numpy()

        # This is in degrees
        llat, llon = np.meshgrid(lat, lon, indexing='ij')

        # dlat, dlon are in radians
        dlat = np.zeros_like(lat)

        _tmp = np.deg2rad(lat[:-1] - lat[1:]) # ERA5 data is werid. Latitude start at north and move southward
        dlat[1:-1] = (_tmp[:-1] + _tmp[1:]) / 2.0
        dlat[0] = dlat[1]
        dlat[-1] = dlat[-2]

        # ERA-interim has equally spaced lon
        dlon = np.deg2rad(lon[1] - lon[0])

        area = r_E**2 * np.cos(np.deg2rad(llat)) * dlon * dlat[:, None]


        print("[%s] Compute AR_objets using method: %s" % (jobname, args.method))

        if args.method == "ANOMLEN": 

            labeled_array, AR_objs = ARdetection_Tien.detectARObjects(
                IVT_x, IVT_y, llat, llon, area,
                IVT_threshold=IVT_clim+250,
                weight=IVT,
                filter_func = ARdetection_Tien.ARFilter_ANOMLEN,
            )

        elif args.method == "ANOMLEN2":

            # Remember, the IVT_clim here should refer to 85th percentile
            # user should provide the correct information with
            # `--AR-clim-dir` parameter
            IVT_clim[IVT_clim < 100.0] = 100.0
 
            labeled_array, AR_objs = ARdetection_Tien.detectARObjects(
                IVT_x, IVT_y, llat, llon, area,
                IVT_threshold=IVT_clim, # Remember, the IVT_clim here should refer to 85th percentile
                weight=IVT,
                filter_func = ARdetection_Tien.ARFilter_ANOMLEN2,
            )


        elif args.method == "ANOMLEN3": 

            # Remember, the IVT_clim here should refer to 85th percentile
            # user should provide the correct information with
            # `--AR-clim-dir` parameter
            IVT_clim[IVT_clim < 100.0] = 100.0
 
            labeled_array, AR_objs = ARdetection_Tien.detectARObjects(
                IVT_x, IVT_y, llat, llon, area,
                IVT_threshold=IVT_clim, # Remember, the IVT_clim here should refer to 85th percentile
                weight=IVT,
                filter_func = ARdetection_Tien.ARFilter_ANOMLEN3,
            )

        elif args.method == "ANOMLEN4": 

            # Remember, the IVT_clim here should refer to 85th percentile
            # user should provide the correct information with
            # `--AR-clim-dir` parameter
            IVT_clim[IVT_clim < 100.0] = 100.0
 
            labeled_array, AR_objs = ARdetection_Tien.detectARObjects(
                IVT_x, IVT_y, llat, llon, area,
                IVT_threshold=IVT_clim, # Remember, the IVT_clim here should refer to 85th percentile
                weight=IVT,
                filter_func = ARdetection_Tien.ARFilter_ANOMLEN4,
            )


        else:
            raise Exception("Unknown method: %s" % (args.method,))
          
        # Convert AR object array into Dataset format
        Nobj = len(AR_objs)
        data_output = dict(
            feature_n    = np.zeros((Nobj,), dtype=int),
            area         = np.zeros((Nobj,),),
            length       = np.zeros((Nobj,),),
            centroid_lat = np.zeros((Nobj,),),
            centroid_lon = np.zeros((Nobj,),),
        )

        for i, AR_obj in enumerate(AR_objs):
            data_output['feature_n'][i] = AR_obj['feature_n']
            data_output['area'][i]      = AR_obj['area']
            data_output['length'][i]    = AR_obj['length']
            data_output['centroid_lat'][i] = AR_obj['centroid'][0]
            data_output['centroid_lon'][i] = AR_obj['centroid'][1]
         
        # Make Dataset
        ds_out = xr.Dataset(

            data_vars=dict(
                map          = (["time", "lat", "lon"], np.reshape(labeled_array, (1, *labeled_array.shape))),
                feature_n    = (["ARobj", ], data_output["feature_n"]),
                length       = (["ARobj", ], data_output["length"]),
                area         = (["ARobj", ], data_output["area"]),
                centroid_lat = (["ARobj", ], data_output["centroid_lat"]),
                centroid_lon = (["ARobj", ], data_output["centroid_lon"]),
            ),

            coords=dict(
                lon=(["lon"], old_lon, dict(units="degrees_east")),
                lat=(["lat"], lat, dict(units="degrees_north")),
                time=(["time"], [dt,]),
            ),

            attrs=dict(description="AR objects file."),
        )

        ds_out = ds_out.roll(lon=lon_first_zero, roll_coords=False)
      
        ds_out.to_netcdf(
            output_file,
            unlimited_dims=["time",],
            encoding={'time': {'dtype': 'i4'}},
        )

        result['status'] = 'OK'
    except Exception as e:

        print("[%s] Error. Now print stacktrace..." % (jobname,))
        import traceback
        traceback.print_exc()


    return result


dts = pd.date_range(beg_time_str, end_time_str, freq="D", inclusive="both")
failed_dates = []
input_args = []

for dt in dts:
    y = dt.year
    m = dt.month
    d = dt.day
    time_str = dt.strftime("%Y-%m-%d")

    if ifSkip(dt):
        print("Skip the date: %s" % (time_str,))
        continue

    result = doJob(dt, detect_phase=True)
    
    if result['status'] != 'OK':
        print("[detect] Failed to detect date %s " % (str(dt),))
    
    if result['need_work'] is False:
        print("[detect] Files all exist for date = %s." % (time_str,))
    else:
        input_args.append((dt, ))
    
with Pool(processes=args.nproc) as pool:

    results = pool.starmap(doJob, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('!!! Failed to generate output of date %s.' % (result['dt'].strftime("%Y-%m-%d_%H"), ))

            failed_dates.append(result['dt'])


print("Tasks finished.")

print("Failed dates: ")
for i, failed_date in enumerate(failed_dates):
    print("%d : %s" % (i+1, failed_date.strftime("%Y-%m-%d"),))


print("Done.")
