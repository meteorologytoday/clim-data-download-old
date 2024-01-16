with open("shared_header.py", "rb") as source_file:
    code = compile(source_file.read(), "shared_header.py", "exec")
exec(code)

with open("shared_header_3hr.py", "rb") as source_file:
    code = compile(source_file.read(), "shared_header_3hr.py", "exec")
exec(code)

import xarray as xr
import traceback
import numpy as np
import pandas as pd
import requests
import shutil
from urllib.parse import urlparse

# I think "sc" stands for "scalar"
url_fmts = dict(
    uv = "https://data.rda.ucar.edu/ds627.0/ei.oper.an.pl/%s/ei.oper.an.pl.regn128uv.%s",
    sc = "https://data.rda.ucar.edu/ds627.0/ei.oper.an.pl/%s/ei.oper.an.pl.regn128sc.%s",
)

labels = ["uv", "sc"]

reference_time = pd.Timestamp("1970-01-01T00:00:00")

print("!!!!! Please make sure xarray can use engine='cfgrib' (provided by ECMWF) !!!!!")

def computeARvariables(ds):
   
    lev_da = ds.coords["isobaricInhPa"]
    ds = ds.where((lev_da <= 1000) & (lev_da >=200), drop=True)
 
    lat = ds.coords["latitude"].to_numpy()
    lon = ds.coords["longitude"].to_numpy()
    lev = ds.coords["isobaricInhPa"].to_numpy()

    q = ds["q"].to_numpy()
    U = ds["u"].to_numpy()
    V = ds["v"].to_numpy()
    
    

    lev_weight = np.zeros((len(lev),))
    dlev = lev[:-1] - lev[1:]
    lev_weight[1:-1] = (dlev[1:] + dlev[:-1]) / 2
    lev_weight[0]    = dlev[0]  / 2
    lev_weight[-1]   = dlev[-1] / 2
    lev_weight *= 100.0 / g0  # convert into density
    
    print(lev_weight)
    lev_weight = lev_weight[None, :, None, None]
  
    print(lev)


    dims = ["time", "latitude", "longitude"]
    AR_vars = {
        'IWV'   : (dims, np.sum(q * lev_weight, axis=1)),
        'IVT'   : (dims, np.sqrt(np.sum(q * U * lev_weight, axis=1)**2 + np.sum(q * V * lev_weight, axis=1)**2)),
        'IVT_x' : (dims,np.sum(q * U * lev_weight, axis=1)),
        'IVT_y' : (dims,np.sum(q * V * lev_weight, axis=1)),
        'IWVKE' : (dims, np.sum(q * (U**2 + V**2) * lev_weight, axis=1),),
    }
    
    ds_AR = xr.Dataset(
        data_vars=AR_vars,
        coords=dict(
            lon=lon,
            lat=lat,
            time=ds.coords["time"],
            reference_time=reference_time,
        ),
    )

    return ds_AR 
    

def genERAInterimURL(dt, label):
    
    url = "https://data.rda.ucar.edu/ds627.0/ei.oper.an.pl/{yyyymm}/ei.oper.an.pl.regn128{label}.{yyyymmddhh}".format(
        yyyymm     = dt.strftime("%Y%m"),
        label      = label,
        yyyymmddhh = dt.strftime("%Y%m%d%H"),
    )
    
    return url


def downloadFile(url, output, verbose=False):

    verbose and print("Downloading url: %s to %s" % (url, output,))
    
    with requests.get(url, stream=True) as download_resp:
        
        if download_resp.status_code != 200:
            print("Something happened.")
            print(download_resp)
            raise Exception("Cannot download file.")


        with open(output, 'wb') as f:
            shutil.copyfileobj(download_resp.raw, f)
    

dataset_name = "ERAinterim"

nproc = 5

ERA_interim_dhr = 6
print("ERA_interim_dhr = ", ERA_interim_dhr)
# ERA-interim output every 6 hours
# Each dhrs specify the averaged time after downloading data
dhrs = [ 24, ] 
print("Going to output ERA interim data in the following time averages (hrs): ", dhrs)

for dhr in dhrs:
    if 24 % dhr != 0:
        raise Exception("Not cool. 24 / dhr (dhr = %d) is not an integer." % (dhr, ) )

    if dhr % ERA_interim_dhr != 0:
        raise Exception("Not cool. dhr / %d (dhr = %d) is not an integer." % (ERA_interim_dhr, dhr, ) )


download_tmp_dir = os.path.join(archive_root, dataset_name, "tmp")

#print("Going to download %d days of data." % (total_days,))

def doJob(t, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(time=t, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        y = t.year
        m = t.month
        d = t.day
        time_str = t.strftime("%Y-%m-%d")
        
        file_prefix = "ERAInterim"
 
        have_all_files = True 
        for dhr in dhrs:

            subcycle = int(24 / dhr)
            download_dir = os.path.join(archive_root, dataset_name, "%02dhr" % (dhr,), )


            for i in range(subcycle):
                beg_hr = i * dhr 
                end_hr = (i+1) * dhr 

                full_time = pd.Timestamp(year=y, month=m, day=d, hour=beg_hr)
                full_time_str = full_time.strftime("%Y-%m-%d_%H")

                output_filename = os.path.join(
                    download_dir,
                    "%s-%s.nc" % (file_prefix, full_time_str, )
                )

                # First round is just to decide which files
                # to be processed to enhance parallel job 
                # distribution. I use variable `phase` to label
                # this stage.
                have_all_files = have_all_files and os.path.isfile(output_filename)
                if not have_all_files:
                    break

        if detect_phase is True:
            result['need_work'] = not have_all_files
            result['status'] = 'OK' 
            return result

        if have_all_files:
            print("[%s] Data already exists. Skip." % (time_str, ))

        elif not have_all_files:

            # First, download all the files
            tmp_filenames = { label : [] for label in labels}
            datasets = {}
            
            merge_data = []
            for label in labels:
                
                for i in range(int(24/ERA_interim_dhr)):
                    beg_hr = i * ERA_interim_dhr 
                    end_hr = (i+1) * ERA_interim_dhr

                    full_time = pd.Timestamp(year=y, month=m, day=d, hour=beg_hr)
                    full_time_str = full_time.strftime("%Y-%m-%d_%H")

                    url = genERAInterimURL(dt=full_time, label=label)
                    tmp_filename = os.path.join(
                        download_tmp_dir,
                        "%s-%s-%s.nc.tmp" % (file_prefix, label, full_time_str,)
                    )
                    tmp_filenames[label].append(tmp_filename)
                    downloadFile(url, tmp_filename, verbose=True)
                
                merge_data.append(xr.open_mfdataset(
                    tmp_filenames[label],
                    combine="nested",
                    concat_dim="time",
                    engine="cfgrib",
                ))

            ds = xr.merge(merge_data)
            for dhr in dhrs:
                
                #print("Doing dhr = ", dhr)
                subcycle = int(24/dhr)
                idx_jmp_per_dhr = int(dhr / ERA_interim_dhr)
                #print("idx_jmp_per_dhr = ", idx_jmp_per_dhr)

                download_dir = os.path.join(archive_root, dataset_name, "%02dhr" % (dhr,), )
                
                if not os.path.isdir(download_dir):
                    print("Create dir: %s" % (download_dir,))
                    Path(download_dir).mkdir(parents=True, exist_ok=True)


                for i in range(subcycle):
                    
                    beg_hr = i * dhr 
                    end_hr = (i+1) * dhr 

                    full_time = pd.Timestamp(year=y, month=m, day=d, hour=beg_hr)
                    full_time_str = full_time.strftime("%Y-%m-%d_%H")
                    
                    time_slice = slice(i*idx_jmp_per_dhr, (i+1)*idx_jmp_per_dhr)

                    ds_sliced = ds.isel(time=slice(i*idx_jmp_per_dhr, (i+1)*idx_jmp_per_dhr)).mean(dim="time", keep_attrs=True).expand_dims(
                        dim = {
                            'time' : [full_time,],
                        },
                        axis=0,
                    )

                    ds_AR = computeARvariables(ds_sliced)
        
                    output_filename = os.path.join(
                        download_dir,
                        "%s-%s.nc" % (file_prefix, full_time_str, )
                    )
 
                    ds_AR.to_netcdf(
                        output_filename, unlimited_dims="time",
                        encoding={'time':{'units':'hours since 1970-01-01 00:00:00'}}
                    )
                    if os.path.isfile(output_filename):
                        print("[%s] File `%s` is generated." % (time_str, output_filename,))


            for _, _tmp_filenames in tmp_filenames.items():
                for remove_file in _tmp_filenames:
                    if os.path.isfile(remove_file):
                        print("[%s] Remove file: `%s` " % (time_str, remove_file))
                        os.remove(remove_file)


            
        result['status'] = 'OK'

    except Exception as e:

        result['status'] = 'ERROR'
        traceback.print_stack()
        traceback.print_exc()
        print(e)

    print("[%s] Done. " % (time_str,))

    return result


def ifSkip(dt):

    skip = False

    if dt.month in [5,6,7,8]:
        skip = True

    return skip

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

    result = doJob(dt, detect_phase=True)
    
    if result['status'] != 'OK':
        print("[detect] Failed to detect date %s " % (str(dt),))
    
    if result['need_work'] is False:
        print("[detect] Files all exist for date = %s." % (time_str,))
    else:
        input_args.append((dt, ))
    
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
