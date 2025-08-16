# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:38:36 2025

@author: andre
"""

path = r'C:\Users\andre\OneDrive\GWU\International Trade'

import pandas as pd
import zipfile
import requests
import io
import geopandas as gpd
import matplotlib.pyplot as plt
import os

download = False
if download:
    url = "https://www.usitc.gov/data/gravity/itpd_e/r03/ITPDE_R03.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall(os.path.join(path,f"data\\itpd_data"))

# Find the main CSV file
csv_file = [f for f in zip_file.namelist() if f.endswith(".csv")][0]
itpd_data = pd.read_csv(os.path.join(path,f"data\\itpd_data\\{csv_file}"))

# Step 2: Sum trade by (exporter_iso3, year)
export_totals = itpd_data.groupby(["exporter_iso3", "year"])["trade"].sum().reset_index()
export_totals['trade'] = export_totals['trade'] / 1000

# Step 3: Select the most recent year
most_recent_year = export_totals["year"].max() 
exports = (export_totals[export_totals["year"] == most_recent_year]
           .sort_values(by='trade', ascending=False)
           .reset_index(drop=True)
           )

top15 = exports.loc[0:14]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top15["exporter_iso3"], top15["trade"], color="mediumseagreen", edgecolor="black", alpha=0.7)
ax.invert_yaxis()
ax.set_xlabel("Total trade, $ billion", fontsize=18)
ax.set_xlim([0, 5000])
plt.tight_layout()
plt.show()
