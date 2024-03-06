with open("shared_header.py", "rb") as source_file:
    code = compile(source_file.read(), "shared_header.py", "exec")
exec(code)

import cdsapi
import numpy as np

c = cdsapi.Client()

download_dir = "data/ERA5/AR"
processed_dir = "data/ERA5/AR_processed"
file_prefix = "ERA5_AR"

total_days = (end_time - beg_time).days

print("Going to download %d days of data." % (total_days,))


class JOB:

    def __init__(self, t):
        self.t = t


    def work(self):
        
        y = self.t.year
        m = self.t.month
        d = self.t.day

        time_now_str = "%04d-%02d-%02d" % (y, m, d)
        filename = "%s/%s_%s.nc" % (download_dir, file_prefix, time_now_str)
        tmp_filename = "%s/%s_%s.nc.tmp" % (download_dir, file_prefix, time_now_str)
        processed_filename = "%s/%s_%s.nc" % (processed_dir, file_prefix, time_now_str)

        already_exists = os.path.isfile(filename)

        if already_exists:

            print("[%s] Data already exists. Skip." % (time_now_str, ))


        else:

            print("[%s] Now download file: %s" % (time_now_str, filename,))

            c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'area': [
                            65, -180, 0, 180,
                            ],
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
                        'pressure_level': [
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
                            ],
                        'variable': [
                                'specific_humidity', 'u_component_of_wind', 'v_component_of_wind',
                                ],
                        },

            tmp_filename)

            pleaseRun("ncks -O --mk_rec_dmn time %s %s" % (tmp_filename, tmp_filename,))
            pleaseRun("ncra -O %s %s" % (tmp_filename, filename,))
            os.remove(tmp_filename)

        if os.path.isfile(processed_filename):

            print("[%s] Processed data already exists. Skip." % (time_now_str, ))

        else:

            print("[%s] Now postprocess to generate: %s" % (time_now_str, processed_filename,))

            ds = netCDF4.Dataset(filename, "r")

            lat = ds.variables["latitude"][:]
            lon = ds.variables["longitude"][:]
            lev = ds.variables["level"][:]

            lev_crop_idx = (lev >= 200) & (lev <= 1000)
            lev = lev[lev_crop_idx]

            q = ds.variables["q"][:, lev_crop_idx, :, :]
            U = ds.variables["u"][:, lev_crop_idx, :, :]
            V = ds.variables["v"][:, lev_crop_idx, :, :]

            ds.close()

            lev_weight = np.zeros((len(lev),))
            dlev = lev[1:] - lev[:-1]
            lev_weight[1:-1] = (dlev[1:] + dlev[:-1]) / 2
            lev_weight[0]    = dlev[0]  / 2
            lev_weight[-1]   = dlev[-1] / 2
            lev_weight *= 100.0 / g0  # convert into density
            
            lev_weight = lev_weight[None, :, None, None]
          
            AR_vars = {
                'IWV' : np.sum(q * lev_weight, axis=1),
                'IVT' : np.sqrt(np.sum(q * U * lev_weight, axis=1)**2 + np.sum(q * V * lev_weight, axis=1)**2),
                'IWVKE' : np.sum(q * (U**2 + V**2) * lev_weight, axis=1),
            }
            
            ds_out = netCDF4.Dataset(processed_filename, mode='w', format='NETCDF4_CLASSIC')
            lat_dim = ds_out.createDimension('lat', len(lat))
            lon_dim = ds_out.createDimension('lon', len(lon)) 
            time_dim = ds_out.createDimension('time', None)

            var_lat = ds_out.createVariable('lat', np.float32, ('lat',))
            var_lon = ds_out.createVariable('lon', np.float32, ('lon',))
            
            var_lat[:] = lat
            var_lon[:] = lon

            for k, d in AR_vars.items():
                
                _var = ds_out.createVariable(k, np.float32, ('time', 'lat', 'lon'))
                _var[0:d.shape[0], :, :] = d

            ds_out.close()

                        
            
def wrap_retrieve(job):

    job.work()

#for y in range(year_rng[0], year_rng[1]):
#    for m in range(1, 13):
#        jobs.append((y, m))

jobs = []
for d in range(total_days):
    new_d =  beg_time + datetime.timedelta(days=d)

    """
    if 5 <= new_d.month and new_d.month <= 8 :
        continue
 
    # We need extra days to compute dSST/dt
    if new_d.month == 4 and new_d.day != 1:
        continue
 
    if new_d.month == 9 and new_d.day != 30:
        continue
    """

    jobs.append(JOB(new_d))


print("Total jobs: %d" % (len(jobs),))

print("Create dir: %s" % (download_dir,))
Path(download_dir).mkdir(parents=True, exist_ok=True)

print("Create dir: %s" % (processed_dir,))
Path(processed_dir).mkdir(parents=True, exist_ok=True)




with Pool(processes=6) as pool:

    result = pool.map(wrap_retrieve, jobs)

print("Done.")
