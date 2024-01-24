from multiprocessing import Pool
import multiprocessing
import datetime
from pathlib import Path
import os.path
import os
import netCDF4

def pleaseRun(cmd):
    print(">> %s" % cmd)
    os.system(cmd)

def ifSkip(dt):

    skip = False

    if dt.month in [5,6,7,8]:
        skip = True

    if dt.month == 4 and dt.day > 15:
        skip = True

    if dt.month == 9 and dt.day < 15:
        skip = True

    return skip

beg_time = datetime.datetime(1992,    9, 1)
end_time = datetime.datetime(2017,    5, 1)

archive_root = "data"


g0 = 9.81
