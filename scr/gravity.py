# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:23:47 2025

@author: cbezerradegoes
"""

from linearmodels.iv.absorbing import AbsorbingLS
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf

data = r'C:\Users\cbezerradegoes\OneDrive\GWU\trade-undergrad\data\Gravity_V202102.dta'

# === Load data ===
df = pd.read_stata(data)

# === Keep only pairs between existing countries ===
df = df[(df["country_exists_o"] == 1) & (df["country_exists_d"] == 1)].copy()

# dependent variable
df["lntrade"] = np.log(df["tradeflow_comtrade_d"])

# independent variables
df["lndist"] = np.log(df["distw"])
df["lang"]   = df["comlang_off"]
df["leg"]    = df["transition_legalchange"]
df["relig"]  = df["comrelig"]
df["colony"] = df["col_dep_ever"]
df["border"] = df["contig"]

# run simple regression

gravity_simple = smf.ols("lntrade ~ lndist", data=df).fit()
print(gravity_simple.summary())

# plot figure


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.scatterplot(x="lndist", y="lntrade", data=df, alpha=0.25, s=10)
sns.regplot(x="lndist", y="lntrade", data=df, scatter=False, color="red")
plt.xlabel("log(distance)")
plt.ylabel("ln(trade)")
plt.title("Trade flows vs. distance")
plt.tight_layout()
plt.savefig(r'C:\Users\cbezerradegoes\OneDrive\GWU\trade-undergrad\figs\gravity.png')
plt.show()

# absorb fixed effects from all variables
# === Construct the variables ===

# drop NAs
fe = ["iso3_o", "iso3_d"]
depvar = ["lntrade"]
indvar = ["lndist", "lang", "leg", "relig", "colony", "border"]
df = df.dropna(subset=fe + depvar + indvar).copy()
df['constant'] = 1

# convert FEs to categorical
for fe in ["iso3_o", "iso3_d"]:
    df[fe] = df[fe].astype("category")

# regressand and regressors 
X = df['constant'].astype(float).values

# run absorbing regressions (fixed effects only) for dependent variables
for var in depvar + indvar:
    y = df[var].astype(float).values
    model = AbsorbingLS(y, X, absorb=df[["iso3_o", "iso3_d"]])
    res = model.fit()
    df[var + "_r"] = res.resids
    print(f'Absorbed fixed effects for variable {var}')

# run gravity regression 
gravity = smf.ols("lntrade_r ~ lndist_r + lang_r + leg_r + relig_r + colony_r + border_r", data=df).fit()
print(gravity.summary())