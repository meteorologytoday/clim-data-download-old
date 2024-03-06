with open("shared_header.py", "rb") as source_file:
    code = compile(source_file.read(), "shared_header.py", "exec")
exec(code)


import traceback
import cdsapi
import numpy as np
import pandas as pd

c = cdsapi.Client()


dataset_name = "ERA5"

def ifSkip(dt):

    return False

nproc = 5
# ERA5 data is output in hourly fashion.
# Each dhrs specify the averaged time after downloading data
# Example: dhrs = [ 3, 24, ] 
dhrs = [ 24,] 

for dhr in dhrs:
    if 24 % dhr != 0:
        raise Exception("Not cool. 24 / dhr (dhr = %d) is not an integer." % (dhr, ) )

varnames = [
#    'geopotential', 
    'mean_sea_level_pressure',
#    'sea_surface_temperature',
#    '10m_u_component_of_wind', 
#    '10m_v_component_of_wind', 
]

var_type = dict(
    
    pressure = [
        'geopotential', 
    ],
    
    surface  = [
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind', 
        'mean_sea_level_pressure',
        'sea_surface_temperature',
        'surface_pressure',
    ],

)


# This is the old version
area = [
    65, -180, 0, 180,
]

full_pressure_levels = [
    '1', '2', '3',
    '5', '7', '10',
    '20', '30', '50',
    '70', '100', '125',
    '150', '175', '200',
    '225', '250', '300',
    '350', '400', '450',
    '500', '550', '600',
    '650', '700', '750',
    '775', '800', '825',
    '850', '875', '900',
    '925', '950', '975',
    '1000',
]

pressure_levels = dict(
    geopotential = ['500', '850', '925', '1000', ],
)


total_days = (end_time - beg_time).days
    
download_tmp_dir = os.path.join(archive_root, dataset_name, "tmp")

print("Going to download %d days of data." % (total_days,))

def doJob(t, varname, detect_phase=False):
    # phase \in ['detect', 'work']
    result = dict(time=t, varname=varname, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        y = t.year
        m = t.month
        d = t.day
        time_str = t.strftime("%Y-%m-%d")
        
        file_prefix = "ERA5"
 
        tmp_filename_downloading = os.path.join(download_tmp_dir, "%s-%s-%s.nc.downloading.tmp" % (file_prefix, varname, time_str,))
        tmp_filename_downloaded  = os.path.join(download_tmp_dir, "%s-%s-%s.nc.downloaded.tmp" % (file_prefix, varname, time_str,))

       
        for dhr in dhrs:

            subcycles = int(24 / dhr)

            download_dir = os.path.join(archive_root, dataset_name, "%02dhr" % (dhr,), varname)
            if not os.path.isdir(download_dir):
                print("Create dir: %s" % (download_dir,))
                Path(download_dir).mkdir(parents=True, exist_ok=True)

            for i in range(subcycles):
               
                beg_hr = i * dhr 
                end_hr = (i+1) * dhr 

                full_time_str = "%s_%02d" % (time_str, beg_hr) 
                output_filename = os.path.join(download_dir, "%s-%s-%s.nc" % (file_prefix, varname, full_time_str, ))

                # First round is just to decide which files
                # to be processed to enhance parallel job 
                # distribution. I use variable `phase` to label
                # this stage.
                if detect_phase is True:
                    result['need_work'] = not os.path.isfile(output_filename)
                    result['status'] = 'OK' 
                    return result
                        

  
                if os.path.isfile(output_filename):
                    print("[%s] Data already exists. Skip." % (full_time_str, ))
                    continue
                else:
                    print("[%s] Now producing file: %s" % (full_time_str, output_filename,))


                # download hourly data is not yet found
                if not os.path.isfile(tmp_filename_downloaded): 

                    if varname in var_type['pressure']:
                        era5_dataset_name = 'reanalysis-era5-pressure-levels'
                        params = {
                                    'product_type': 'reanalysis',
                                    'format': 'netcdf',
                                    'area': area,
                                    'time': [
                                        '00:00', '01:00', '02:00',
                                        '03:00', '04:00', '05:00',
                                        '06:00', '07:00', '08:00',
                                        '09:00', '10:00', '11:00',
                                        '12:00', '13:00', '14:00',
                                        '15:00', '16:00', '17:00',
                                        '18:00', '19:00', '20:00',
                                        '21:00', '22:00', '23:00',
                                        ],
                                    'day': [
                                            "%02d" % d,
                                        ],
                                    'month': [
                                            "%02d" % m,
                                        ],
                                    'year': [
                                            "%04d" % y,
                                        ],
                                    'pressure_level': pressure_levels[varname] if varname in pressure_levels else full_pressure_levels,
                                    'variable': [varname,],
                        }

                    elif varname in var_type['surface']:
                            
                        era5_dataset_name = 'reanalysis-era5-single-levels'
                        params = {
                                    'product_type': 'reanalysis',
                                    'format': 'netcdf',
                                    'area': area,
                                    'time': [
                                        '00:00', '01:00', '02:00',
                                        '03:00', '04:00', '05:00',
                                        '06:00', '07:00', '08:00',
                                        '09:00', '10:00', '11:00',
                                        '12:00', '13:00', '14:00',
                                        '15:00', '16:00', '17:00',
                                        '18:00', '19:00', '20:00',
                                        '21:00', '22:00', '23:00',
                                        ],
                                    'day': [
                                            "%02d" % d,
                                        ],
                                    'month': [
                                            "%02d" % m,
                                        ],
                                    'year': [
                                            "%04d" % y,
                                        ],
                                    'variable': [varname,],
                        }

                    print("Downloading file: %s" % ( tmp_filename_downloading, ))
                    c.retrieve(era5_dataset_name, params, tmp_filename_downloading)
                    pleaseRun("ncks -O --mk_rec_dmn time %s %s" % (tmp_filename_downloading, tmp_filename_downloaded,))

                pleaseRun("ncra -O -d time,%d,%d %s %s" % (dhr*i, dhr*(i+1)-1, tmp_filename_downloaded, output_filename,))
                if os.path.isfile(output_filename):
                    print("[%s] File `%s` is generated." % (time_str, output_filename,))


        for remove_file in [tmp_filename_downloaded, tmp_filename_downloading]:
            if os.path.isfile(remove_file):
                print("[%s] Remove file: `%s` " % (time_str, remove_file))
                os.remove(remove_file)

        result['status'] = 'OK'

    except Exception as e:

        result['status'] = 'ERROR'
        traceback.print_stack()
        print(e)

    print("[%s] Done. " % (time_str,))

    return result


failed_dates = []
dts = pd.date_range(beg_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"), inclusive="both")

input_args = []

for dt in dts:

    y = dt.year
    m = dt.month
    d = dt.day
    time_str = dt.strftime("%Y-%m-%d")

    if ifSkip(dt):
        print("Skip the date: %s" % (time_str,))
        continue

    for varname in varnames:
    
        result = doJob(dt, varname, detect_phase=True)
        
        if result['status'] != 'OK':
            print("[detect] Failed to detect variable `%s` of date %s " % (varname, str(dt)))
        
        if result['need_work'] is False:
            print("[detect] Files all exist for (date, varname) =  (%s, %s)." % (time_str, varname))
        else:
            input_args.append((dt, varname,))
        
print("Create dir: %s" % (download_tmp_dir,))
Path(download_tmp_dir).mkdir(parents=True, exist_ok=True)

with Pool(processes=nproc) as pool:

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
