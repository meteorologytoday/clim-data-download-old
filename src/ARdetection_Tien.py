import numpy as np
import xarray as xr
from scipy.ndimage import label, generate_binary_structure
from scipy import spatial
from earth_constants import r_E as r_earth
import sphereTools

"""
    `pts` must have the shape of (npts, dim), where dim=2 in AR detection.
    Also, the pts are in (lat, lon) format, and units are all in radians.
    
    This algorithm is copied from 
    https://stackoverflow.com/questions/50468643/finding-two-most-far-away-points-in-plot-with-many-points-in-python
"""
def getTheFarthestPtsOnSphere(pts):

    # Looking for the most distant points
    # two points which are fruthest apart will occur as vertices of the convex hulil

    try:
        candidates = pts[spatial.ConvexHull(pts).vertices, :]
    except Exception as e:
        print("Something happen with QhHull: ", str(e))

        candidates = pts

    # get distances between each pair of candidate points
    # dist_mat = spatial.distance_matrix(candidates, candidates)

    dist_mat = np.zeros((len(candidates), len(candidates)))

    for i in range(len(candidates)):
        for j in range(len(candidates)):

            if i >= j:
                dist_mat[i, j] = 0.0
                continue

            dist_mat[i, j] = sphereTools.getDistOnSphere(lat1=candidates[i, 0], lon1=candidates[i, 1], lat2=candidates[j, 0], lon2=candidates[j, 1], r=r_earth)

    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
            
    farthest_pair = ( candidates[i, :], candidates[j, :] )

    return farthest_pair, dist_mat[i, j]



def basicARFilter(AR_obj):

    result = True
    
    if AR_obj['aligned_fraction'] < 0.5:
        return False

    if AR_obj['length'] < 2000e3:
        return False

    if AR_obj['aspect_ratio'] < 2.0:
        return False

   
    return result


def detectARObjects(IVT_x, IVT_y, coord_lat, coord_lon, area, IVT_threshold=500.0, weight=None, filter_func=basicARFilter):

    # Pseudo code: 
    # 1. Generate object maps with intensity check
    # 2. Direction Check
    # 3. Geometry Check

    # =============================================

    # 1. Generate object maps with intensity check
    
    IVT = np.sqrt( IVT_x ** 2 + IVT_y ** 2 )
    IVT_binary = np.zeros_like(IVT, dtype=int)
    IVT_binary[IVT >= IVT_threshold] = 1
   
    # Using the default connectedness: four sides
    labeled_array, num_features = label(IVT_binary)

    AR_objs = []

    # Iterate through possible candidate
    for feature_n in range(1, num_features+1): # numbering starts at 1 

        print("feature_n = ", feature_n)
        # 2. Direction Check. This can remove features such as tropical cyclone
        idx = labeled_array == feature_n
        Npts = np.sum(idx)
        covered_area = area[idx]
        sum_covered_area = np.sum(covered_area)

        sub_IVT   = IVT[idx]
        sub_IVT_x = IVT_x[idx]
        sub_IVT_y = IVT_y[idx]

        mean_sub_IVT_x = np.sum(sub_IVT_x * covered_area) / sum_covered_area
        mean_sub_IVT_y = np.sum(sub_IVT_y * covered_area) / sum_covered_area

        mean_sub_IVT = np.sqrt(mean_sub_IVT_x**2 + mean_sub_IVT_y**2)       

        
        sub_IVTdir_x = sub_IVT_x / sub_IVT
        sub_IVTdir_y = sub_IVT_y / sub_IVT

        mean_sub_IVTdir_x = mean_sub_IVT_x / mean_sub_IVT
        mean_sub_IVTdir_y = mean_sub_IVT_y / mean_sub_IVT

        #print("(mean_sub_IVT_x, mean_sub_IVT_y) = (%.2f, %.2f)" % (mean_sub_IVT_x, mean_sub_IVT_y))

        # at this point, mean_sub_IVT_xy and sub_IVT_xy are all unit vectors
        inner_products = sub_IVTdir_x * mean_sub_IVTdir_x + sub_IVTdir_y * mean_sub_IVTdir_y

        aligned_pts = np.sum( inner_products >= np.cos(np.radians(45)) )
        aligned_fraction = aligned_pts / Npts
            
        # If no more than half grid points have
        # the same direction within 45 degress,
        # discard the object

       
        # 3. Geometry check
 
        # First find the length (greatest distance)
        pts = np.zeros((Npts, 2))
        pts[:, 0] = coord_lat[idx]
        pts[:, 1] = coord_lon[idx]
        pts = np.radians(pts)
        farthest_pair, farthest_dist = getTheFarthestPtsOnSphere(pts)
      
 
        # Compute centroid 
        if weight is None:
            _wgt = covered_area
        else:
            _wgt = covered_area * weight[idx]

        _sum_wgt = np.sum(_wgt)
        
            
             
        centroid = (
            np.sum(coord_lat[idx] * _wgt) / _sum_wgt,
            np.sum(coord_lon[idx] * _wgt) / _sum_wgt,
        )

        eqv_wid = sum_covered_area / farthest_dist
        aspect_ratio = farthest_dist / eqv_wid  # = farthest_dist**2 / sum_covered_area

        AR_obj = dict(
            feature_n     = feature_n,
            area          = sum_covered_area,
            aligned_fraction = aligned_fraction,
            centroid      = centroid,
            length        = farthest_dist,
            aspect_ratio  = aspect_ratio,
            farthest_pair = farthest_pair,
        )
 
        if (filter_func is not None) and (filter_func(AR_obj) is False):
            labeled_array[labeled_array == feature_n] = 0.0
            continue 

        AR_objs.append(AR_obj)
   
    print("How many AR_objs? ", len(AR_objs)) 
    for i, AR_obj in enumerate(AR_objs):
        print("[%d] area = %f km^2, length = %f km" % (i, AR_obj["area"]/1e6, AR_obj['length']/1e3,)) 
    return labeled_array, AR_objs



if __name__  == "__main__" :
    
    import xarray as xr

    test_file = "./data/ERAinterim/24hr/ERAInterim-2017-01-10_00.nc"
    test_clim_file = "./climatology_quantile_1993-2017_ERAInterim/ERAInterim-clim-daily_01-10_00.nc"
    
    ds = xr.open_dataset(test_file)
    ds_clim = xr.open_dataset(test_clim_file).sel(quantile=.85)

    print(ds)
    print(ds_clim)

    # find the lon=0
    lon_first_zero = np.argmax(ds.coords["lon"].to_numpy() >= 0)
    print("First longitude zero idx: ", lon_first_zero)
    ds = ds.roll(lon=-lon_first_zero, roll_coords=True)
    ds_clim = ds_clim.roll(lon=-lon_first_zero, roll_coords=True)
    
    lat = ds.coords["lat"].to_numpy() 
    lon = ds.coords["lon"].to_numpy()  % 360
  
    # For some reason we need to reassign it otherwise the contourf will be broken... ??? 
    ds = ds.assign_coords(lon=lon) 
    ds_clim = ds_clim.assign_coords(lon=lon) 
    
    IVT_x = ds["IVT_x"][0, :, :].to_numpy()
    IVT_y = ds["IVT_y"][0, :, :].to_numpy()
    IVT   = np.sqrt(IVT_x**2 + IVT_y**2)
    IVT_clim = ds_clim["IVT"].isel(time=0).to_numpy()

    print("Dimension check: dim(IVT_clim) = ", IVT_clim.shape)


    llat, llon = np.meshgrid(lat, lon, indexing='ij')

    dlat = np.deg2rad((lat[0] - lat[1]))
    dlon = np.deg2rad((lon[1] - lon[0]))

    R_earth = 6.4e6
 
    area = R_earth**2 * np.cos(np.deg2rad(llat)) * dlon * dlat

    print("Compute AR_objets")
    

    algo_results = dict(
         ANOMLEN2 = dict(
            result=detectARObjects(IVT_x, IVT_y, llat, llon, area, IVT_threshold=IVT_clim, weight=IVT, filter_func = basicARFilter),
            IVT=IVT,
        ),

        TOTIVT500 = dict(
            result=detectARObjects(IVT_x, IVT_y, llat, llon, area, IVT_threshold=500.0, weight=IVT, filter_func = basicARFilter),
            IVT=IVT,
        ),
    )


    print("Loading matplotlib") 
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as transforms
    from matplotlib.dates import DateFormatter
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    print("done")

    cent_lon = 180.0

    plot_lon_l = -180.0
    plot_lon_r = 180.0
    plot_lat_b = -90.0
    plot_lat_t = 90.0

    proj = ccrs.PlateCarree(central_longitude=cent_lon)
    proj_norm = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        len(list(algo_results.keys())), 1,
        figsize=(12, 8),
        subplot_kw=dict(projection=proj),
        gridspec_kw=dict(hspace=0.15, wspace=0.2),
        constrained_layout=False,
        squeeze=False,
    )


    for i, keyname in enumerate(["ANOMLEN2", "TOTIVT500"]):

        print("Plotting :", keyname)
        
        _labeled_array = algo_results[keyname]["result"][0]
        _AR_objs = algo_results[keyname]["result"][1]

        _IVT = algo_results[keyname]["IVT"]

        _labeled_array = _labeled_array.astype(float)
        _labeled_array[_labeled_array!=0] = 1.0

        _ax = ax[i, 0]

        _ax.set_title(keyname)
        
        levs = [np.linspace(0, 1000, 11), np.linspace(0, 1000, 11), np.linspace(-800, 800, 17)][i]
        cmap = cm.get_cmap([ "ocean_r", "ocean_r", "bwr_r" ][i])

        mappable = _ax.contourf(lon, lat, _IVT, levels=levs, cmap=cmap,  transform=proj_norm)
        plt.colorbar(mappable, ax=_ax, orientation="vertical")
        _ax.contour(lon, lat, _labeled_array, levels=[0.5,], colors='yellow',  transform=proj_norm, zorder=98, linewidth=1)


        for i, AR_obj in enumerate(_AR_objs):
            pts = AR_obj["farthest_pair"]
            cent = AR_obj["centroid"]
            _ax.plot(np.degrees([pts[0][1], pts[1][1]]), np.degrees([pts[0][0], pts[1][0]]), 'r-', transform=ccrs.Geodetic(), zorder=99)

            _ax.text(cent[1], cent[0], "%d" % (i+1), va="center", ha="center", color="cyan", transform=proj_norm, zorder=100)

        _ax.set_global()
        #_ax.coastlines()
        _ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

        gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')

        gl.xlabels_top   = False
        gl.ylabels_right = False

        #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

    plt.show()


