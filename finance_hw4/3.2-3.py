import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from scipy.stats import f


###############################################
# Utility
###############################################
def drop_dupes_sort(df):
    df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


###############################################
# 1. Load Fama–French 5 factors
###############################################
ff5_raw = pd.read_csv("finance_hw4/F-F_Research_Data_5_Factors_2x3.csv", skiprows=4)

mask = ff5_raw["Date"].astype(str).str.match(r"^\d{6}$")
ff5 = ff5_raw[mask].copy()

for col in ff5.columns[1:]:
    ff5[col] = pd.to_numeric(ff5[col], errors="coerce")

ff5["Date"] = pd.to_datetime(ff5["Date"].astype(str), format="%Y%m")
ff5 = ff5.set_index("Date") / 100.0
ff5 = ff5.rename(columns={" Mkt-RF": "MKT"})
ff5 = drop_dupes_sort(ff5)

###############################################
# 2. Load Momentum (UMD)
###############################################
mom_raw = pd.read_csv("finance_hw4/F-F_Momentum_Factor.csv", skiprows=13)
mom_raw["Date"] = pd.to_datetime(
    mom_raw["Unnamed: 0"].astype(str), format="%Y%m", errors="coerce"
)
mom = mom_raw.dropna(subset=["Date"]).copy()
mom["Mom"] = pd.to_numeric(mom["Mom"], errors="coerce")
mom = mom.set_index("Date")
mom["UMD"] = mom["Mom"] / 100.0
mom = mom[["UMD"]]
mom = drop_dupes_sort(mom)

###############################################
# 3. Load HML Devil (USA) - ADDED SECTION
###############################################
dev = pd.read_excel(
    "finance_hw4/The Devil in HMLs Details Factors Monthly.xlsx", sheet_name="HML Devil"
)

# USA column = column index 24 (based on 3.1.py logic)
# Row 18 is where data usually starts in this AQR sheet
hml_dev = dev.iloc[18:, [0, 24]].copy()
hml_dev.columns = ["Date", "HML_DEV"]

# Convert date
hml_dev["Date"] = pd.to_datetime(hml_dev["Date"], errors="coerce")
hml_dev = hml_dev.dropna(subset=["Date"])

# Convert to decimal
hml_dev["HML_DEV"] = pd.to_numeric(hml_dev["HML_DEV"], errors="coerce") / 100.0

# Set index and align to monthly
hml_dev = hml_dev.set_index("Date")
hml_dev.index = hml_dev.index.to_period("M").to_timestamp()
hml_dev = hml_dev.loc[hml_dev.index >= "1963-07-01"]
hml_dev = drop_dupes_sort(hml_dev)

###############################################
# 4. Load 10 Industry Portfolios
###############################################
lines_ind = open("finance_hw4/10_Industry_Portfolios.csv").read().splitlines()
data_ind = [ln.strip() for ln in lines_ind if re.match(r"^\s*\d{6}", ln)]

dates_ind, vals_ind = [], []
for ln in data_ind:
    parts = [p for p in ln.split(",") if p != ""]
    d = parts[0].strip()
    nums = [float(p) for p in parts[1:]]
    if len(nums) != 10:
        continue
    dates_ind.append(d)
    vals_ind.append(nums)

ind10 = pd.DataFrame(
    vals_ind,
    columns=[
        "NoDur",
        "Durbl",
        "Manuf",
        "Enrgy",
        "HiTec",
        "Telcm",
        "Shops",
        "Hlth",
        "Utils",
        "Other",
    ],
)
ind10["Date"] = pd.to_datetime(dates_ind, format="%Y%m")
ind10 = ind10.set_index("Date") / 100.0
ind10 = drop_dupes_sort(ind10)

###############################################
# 5. Load 25 Size–Book Portfolios
###############################################
lines25 = open("finance_hw4/Developed_25_Portfolios_ME_BE-ME.csv").read().splitlines()
data25 = [ln.strip() for ln in lines25 if re.match(r"^\s*\d{6}", ln)]

dates25, vals25 = [], []
for ln in data25:
    parts = [p for p in ln.split(",") if p != ""]
    d = parts[0].strip()
    nums = [float(p) for p in parts[1:]]
    if len(nums) != 25:
        continue
    dates25.append(d)
    vals25.append(nums)

p25 = pd.DataFrame(vals25, columns=[f"P{i+1}" for i in range(25)])
p25["Date"] = pd.to_datetime(dates25, format="%Y%m")
p25 = p25.set_index("Date") / 100.0
p25 = drop_dupes_sort(p25)

###############################################
# Show loaded dataset shapes
###############################################
print("Loaded:")
print("FF5:", ff5.shape)
print("UMD:", mom.shape)
print("HML_DEV:", hml_dev.shape)
print("25 portfolios:", p25.shape)
print("10 industries:", ind10.shape)

###############################################
# 6. Factor matrices
###############################################
# Model 1: Fama-French 5 Factors
factors5 = ff5[["MKT", "SMB", "HML", "RMW", "CMA"]]

# Model 2: AQR Six-Factor Model
# Definition: MKT, SMB, RMW, CMA (from FF), HML_DEV (from AQR), UMD (from FF)
# 1. Start with FF components (excluding standard HML)
factors_aqr_base = ff5[["MKT", "SMB", "RMW", "CMA"]]

# 2. Join HML_DEV (Inner join to align dates)
factors_aqr_base = factors_aqr_base.join(hml_dev, how="inner")

# 3. Join Momentum (UMD)
factors6 = factors_aqr_base.join(mom, how="inner")

# Reorder columns for cleanliness
factors6 = factors6[["MKT", "SMB", "HML_DEV", "RMW", "CMA", "UMD"]]

###############################################
# 7. Excess returns
###############################################
# We need to intersect with factors5 to get the RF (Risk Free) rate
# Since factors6 is a subset of dates usually, we'll align based on the specific regression later
# But for calculating excess return, we need RF.
common25 = p25.index.intersection(ff5.index)
common10 = ind10.index.intersection(ff5.index)

ret25_excess = p25.loc[common25].sub(ff5.loc[common25, "RF"], axis=0)
ret10_excess = ind10.loc[common10].sub(ff5.loc[common10, "RF"], axis=0)


###############################################
# 8. Regression helper
###############################################
def regress_all(R_df, F_df):
    idx = R_df.index.intersection(F_df.index)
    R = R_df.loc[idx]
    F = F_df.loc[idx]
    X = sm.add_constant(F)

    alphas, tstats = [], []
    for col in R.columns:
        model = sm.OLS(R[col], X).fit()
        alphas.append(model.params[0])
        tstats.append(model.tvalues[0])

    return pd.DataFrame({"alpha": alphas, "tstat": tstats}, index=R.columns)


###############################################
# 9. GRS Test
###############################################
def grs_test(returns_df, factors_df):

    idx = returns_df.index.intersection(factors_df.index)
    R = returns_df.loc[idx].values
    F = factors_df.loc[idx].values

    T, N = R.shape
    L = F.shape[1]

    alphas = []
    residuals = []

    for i in range(N):
        model = sm.OLS(R[:, i], sm.add_constant(F)).fit()
        alphas.append(model.params[0])
        residuals.append(model.resid)

    alphas = np.array(alphas).reshape(-1, 1)
    residuals = np.array(residuals).T

    Sigma = (residuals.T @ residuals) / (T - L - 1)
    Sigma_inv = np.linalg.inv(Sigma)

    mu = np.mean(F, axis=0).reshape(-1, 1)
    Omega = np.cov(F, rowvar=False)
    Omega_inv = np.linalg.inv(Omega)

    num = (T / N) * ((T - N - L) / (T - L - 1)) * (alphas.T @ Sigma_inv @ alphas)
    den = 1 + (mu.T @ Omega_inv @ mu)

    grs_stat = float(num / den)
    p_val = 1 - f.cdf(grs_stat, N, T - N - L)

    return grs_stat, p_val, alphas


###############################################
# 10. Compute Alpha tables
###############################################
alpha25_ff5 = regress_all(ret25_excess, factors5)
alpha25_6 = regress_all(ret25_excess, factors6)

alpha10_ff5 = regress_all(ret10_excess, factors5)
alpha10_6 = regress_all(ret10_excess, factors6)

alpha25_ff5.to_csv("finance_hw4/alpha25_ff5.csv")
alpha25_6.to_csv("finance_hw4/alpha25_6.csv")
alpha10_ff5.to_csv("finance_hw4/alpha10_ff5.csv")
alpha10_6.to_csv("finance_hw4/alpha10_6.csv")

###############################################
# 11. Run GRS
###############################################
idx25_5 = ret25_excess.index.intersection(factors5.index)
idx25_6 = ret25_excess.index.intersection(factors6.index)

grs25_ff5, p25_ff5, _ = grs_test(ret25_excess.loc[idx25_5], factors5.loc[idx25_5])
grs25_6, p25_6, _ = grs_test(ret25_excess.loc[idx25_6], factors6.loc[idx25_6])

idx10_5 = ret10_excess.index.intersection(factors5.index)
idx10_6 = ret10_excess.index.intersection(factors6.index)

grs10_ff5, p10_ff5, _ = grs_test(ret10_excess.loc[idx10_5], factors5.loc[idx10_5])
grs10_6, p10_6, _ = grs_test(ret10_excess.loc[idx10_6], factors6.loc[idx10_6])

###############################################
# 12. Print Final Summary
###############################################
print("========== 25 Portfolios ==========")
print("FF5 GRS:", grs25_ff5, "p:", p25_ff5)
print("AQR Six-Factor GRS:", grs25_6, "p:", p25_6)

print("========== 10 Industries ==========")
print("FF5 GRS:", grs10_ff5, "p:", p10_ff5)
print("AQR Six-Factor GRS:", grs10_6, "p:", p10_6)
