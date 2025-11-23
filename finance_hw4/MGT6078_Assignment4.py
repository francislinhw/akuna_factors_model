
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from scipy.stats import f

###############################################
#   MGT6078 Assignment 4 – Integrated Script  #
###############################################

# ========= 1. Read Fama–French 5 Factors =========
ff5_raw = pd.read_csv("F-F_Research_Data_5_Factors_2x3.csv", skiprows=4)

mask = ff5_raw['Date'].astype(str).str.match(r'^\d{6}$')
ff5 = ff5_raw[mask].copy()

# Convert numeric columns
for col in ff5.columns[1:]:
    ff5[col] = pd.to_numeric(ff5[col], errors='coerce')

ff5['Date'] = pd.to_datetime(ff5['Date'].astype(str), format="%Y%m")
ff5 = ff5.set_index('Date') / 100.0
ff5 = ff5.rename(columns={" Mkt-RF": "MKT"})  # Trim leading space in column name

# ========= 2. Read Momentum (UMD) =========
mom_raw = pd.read_csv("F-F_Momentum_Factor.csv", skiprows=13)

mom = mom_raw.copy()
mom['Date'] = pd.to_datetime(mom['Unnamed: 0'].astype(str), format="%Y%m", errors='coerce')
mom = mom.dropna(subset=['Date'])

mom['Mom'] = pd.to_numeric(mom['Mom'], errors='coerce')
mom = mom.set_index('Date')
mom['UMD'] = mom['Mom'] / 100.0
mom = mom[['UMD']]

# ========= 3. Read 10 Industry Portfolios =========
lines_ind = open("10_Industry_Portfolios.csv").read().splitlines()
data_ind = [ln.strip() for ln in lines_ind if re.match(r'^\s*\d{6}', ln)]

dates_ind, vals_ind = [], []
for ln in data_ind:
    parts = [p for p in ln.split(',') if p != ""]
    d = parts[0].strip()
    nums = [float(p) for p in parts[1:]]
    if len(nums) != 10:
        continue
    dates_ind.append(d)
    vals_ind.append(nums)

ind10 = pd.DataFrame(
    vals_ind,
    columns=["NoDur","Durbl","Manuf","Enrgy","HiTec",
             "Telcm","Shops","Hlth","Utils","Other"]
)
ind10['Date'] = pd.to_datetime(dates_ind, format="%Y%m")
ind10 = ind10.set_index('Date') / 100.0

# ========= 4. Read 25 Size–BM Portfolios =========
lines25 = open("Developed_25_Portfolios_ME_BE-ME.csv").read().splitlines()
data25 = [ln.strip() for ln in lines25 if re.match(r'^\s*\d{6}', ln)]

dates25, vals25 = [], []
for ln in data25:
    parts = [p for p in ln.split(',') if p != ""]
    d = parts[0].strip()
    nums = [float(p) for p in parts[1:]]
    if len(nums) != 25:
        continue
    dates25.append(d)
    vals25.append(nums)

p25 = pd.DataFrame(vals25, columns=[f"P{i+1}" for i in range(25)])
p25['Date'] = pd.to_datetime(dates25, format="%Y%m")
p25 = p25.set_index('Date') / 100.0

print("Loaded datasets:")
print("FF5:", ff5.shape)
print("UMD:", mom.shape)
print("25 Portfolios:", p25.shape)
print("10 Industries:", ind10.shape)

# ========= Build Factor Matrices =========
factors5 = ff5[['MKT','SMB','HML','RMW','CMA']]
factors6 = factors5.join(mom, how='inner')   # 6-factor (traditional HML + UMD)

# ========= Compute Excess Returns =========
common25 = p25.index.intersection(ff5.index)
common10 = ind10.index.intersection(ff5.index)

ret25 = p25.loc[common25]
rf25 = ff5.loc[common25, 'RF']
ret25_excess = ret25.sub(rf25, axis=0)

ret10 = ind10.loc[common10]
rf10 = ff5.loc[common10, 'RF']
ret10_excess = ret10.sub(rf10, axis=0)

# ========= GRS Test Function =========
def grs_test(returns_df, factors_df):
    common_idx = returns_df.index.intersection(factors_df.index)

    R = returns_df.loc[common_idx].values   # T × N
    F = factors_df.loc[common_idx].values   # T × L

    T, N = R.shape
    L = F.shape[1]

    alphas = []
    residuals = []

    for i in range(N):
        y = R[:, i]
        X = sm.add_constant(F)
        model = sm.OLS(y, X).fit()
        alphas.append(model.params[0])
        residuals.append(model.resid)

    alphas = np.array(alphas).reshape(-1, 1)
    residuals = np.array(residuals).T

    Sigma = residuals.T @ residuals / (T - L - 1)
    Sigma_inv = np.linalg.inv(Sigma)

    mu = np.mean(F, axis=0).reshape(-1, 1)
    Omega = np.cov(F, rowvar=False)
    Omega_inv = np.linalg.inv(Omega)

    num = (T / N) * ((T - N - L) / (T - L - 1)) * (alphas.T @ Sigma_inv @ alphas)
    den = 1 + (mu.T @ Omega_inv @ mu)

    grs_stat = float(num / den)
    p_val = 1 - f.cdf(grs_stat, N, T - N - L)

    return grs_stat, p_val, alphas

# ========= Run GRS for 25 Portfolios =========
idx25_5 = ret25_excess.index.intersection(factors5.index)
idx25_6 = ret25_excess.index.intersection(factors6.index)

grs_25_ff5, p_25_ff5, alpha_25_ff5 = grs_test(
    ret25_excess.loc[idx25_5],
    factors5.loc[idx25_5]
)

grs_25_6, p_25_6, alpha_25_6 = grs_test(
    ret25_excess.loc[idx25_6],
    factors6.loc[idx25_6]
)

print("========== 25 Portfolios ==========")
print("FF5 GRS:", grs_25_ff5, "p-value:", p_25_ff5)
print("Six-Factor GRS:", grs_25_6, "p-value:", p_25_6)

# ========= Run GRS for 10 Industries =========
idx10_5 = ret10_excess.index.intersection(factors5.index)
idx10_6 = ret10_excess.index.intersection(factors6.index)

grs_10_ff5, p_10_ff5, alpha_10_ff5 = grs_test(
    ret10_excess.loc[idx10_5],
    factors5.loc[idx10_5]
)

grs_10_6, p_10_6, alpha_10_6 = grs_test(
    ret10_excess.loc[idx10_6],
    factors6.loc[idx10_6]
)

print("========== 10 Industries ==========")
print("FF5 GRS:", grs_10_ff5, "p-value:", p_10_ff5)
print("Six-Factor GRS:", grs_10_6, "p-value:", p_10_6)

# END OF SCRIPT
