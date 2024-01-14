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

beg_time = datetime.datetime(1993,     1, 1)
end_time = datetime.datetime(2005,    12, 1)

archive_root = "data"


g0 = 9.81
