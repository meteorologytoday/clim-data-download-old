#!/bin/bash

python3 download_ERA-interim_sfc.py
python3 download_ERA-interim_AR.py

bash ./make_ERA-interim_AR_objects.sh
bash ./make_climatology_ERA-interim_AR.sh
