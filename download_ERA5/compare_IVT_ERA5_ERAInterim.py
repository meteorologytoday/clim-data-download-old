import numpy as np
import xarray as xr
import pandas as pd


date_selected = pd.Timestamp("1995-01-10")
ymd_str = date_selected.strftime("%Y-%m-%d")
md_str = date_selected.strftime("%m-%d")

input_files = dict(
    ERAInterim = "/data/SO2/t2hsu/data/ERAInterim/AR/24hr/ERAInterim-%s_00.nc" % (ymd_str,),
    ERA5       = "/data/SO2/t2hsu/data/ERA5/AR_processed/ERA5_AR_%s.nc" % (ymd_str,),
)

data = dict()

for k, input_file in input_files.items():

    print("[dataset:s] Loading file: {filename:s}".format(
        dataset = k,
        filename = input_file,
    ))

    ds = xr.open_dataset(input_file)
    lon_first_zero = np.argmax(ds.coords["lon"].to_numpy() >= 0)
    
    print("First longitude zero idx: ", lon_first_zero)
    ds = ds.roll(lon=-lon_first_zero, roll_coords=True)
    lat = ds.coords["lat"].to_numpy() 
    lon = ds.coords["lon"].to_numpy()  % 360

    #IVT_x = ds["IVT_x"][0, :, :].to_numpy()
    #IVT_y = ds["IVT_y"][0, :, :].to_numpy()
    #IVT   = np.sqrt(IVT_x**2 + IVT_y**2)

    #IVT_x = ds["IVT_x"][0, :, :].to_numpy()
    #IVT_y = ds["IVT_y"][0, :, :].to_numpy()
    IVT   = ds["IVT"][0, :, :].to_numpy()#np.sqrt(IVT_x**2 + IVT_y**2)


    data[k] = dict(
        lat = ds.coords["lat"].to_numpy(),
        lon = ds.coords["lon"].to_numpy()  % 360,
        IVT  = IVT,
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

import tool_fig_config
print("done")

cent_lon = 180.0

plot_lon_l = 0.0
plot_lon_r = 360.0
plot_lat_b =  0.0
plot_lat_t = 70.0

proj = ccrs.PlateCarree(central_longitude=cent_lon)
proj_norm = ccrs.PlateCarree()

ncol = 1
nrow = len(list(data.keys()))

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 9,
    h = [3,] * nrow,
    wspace = 1.0,
    hspace = 1.0,
    w_left = 1.0,
    w_right = 1.0,
    h_bottom = 1.0,
    h_top = 0.5,
    ncol = 1,
    nrow = 1,
)


subplot_kw = dict(
    projection = proj,
    aspect     = 'auto',
)

fig, ax = plt.subplots(
    nrow, ncol,
    figsize=figsize,
    subplot_kw=dict(projection=proj),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)


for i, (k, _data) in enumerate(data.items()):

    print("Plotting dataset :", k)
    
    _ax = ax[i, 0]

    _ax.set_title("Dataset: %s" % (k,))
    
    levs = np.linspace(0, 1000, 11)
    cmap = cm.get_cmap("ocean_r")
    
    mappable = _ax.contourf(_data["lon"], _data["lat"], _data["IVT"], levels=levs, cmap=cmap,  transform=proj_norm)

    
    
    _ax.set_global()
    _ax.coastlines()
    _ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

    gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')

    gl.xlabels_top   = False
    gl.ylabels_right = False

    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))#[120, 150, 180, -150, -120])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60, 70])

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.02, spacing=0.01)
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical")
    
    cb.ax.set_ylabel("IVT [kg/m/s]")

fig.suptitle("Selected date: %s" % (str(date_selected),))

fig.savefig("figure_IVT_comparison_%s.png" % (ymd_str,), dpi=200)
plt.show()
