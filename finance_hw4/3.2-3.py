import pandas as pd, numpy as np, re, statsmodels.api as sm
from scipy.stats import f

# Load FF5
ff5_raw = pd.read_csv("finance_hw4/F-F_Research_Data_5_Factors_2x3.csv", skiprows=4)
mask = ff5_raw["Date"].astype(str).str.match(r"^\d{6}$")
ff5 = ff5_raw[mask].copy()
for col in ff5.columns[1:]:
    ff5[col] = pd.to_numeric(ff5[col], errors="coerce")
ff5["Date"] = pd.to_datetime(ff5["Date"].astype(str), format="%Y%m")
ff5 = ff5.set_index("Date") / 100.0
ff5 = ff5.rename(columns={" Mkt-RF": "MKT"})
ff5 = ff5[~ff5.index.duplicated()]

# Load UMD
mom_raw = pd.read_csv("finance_hw4/F-F_Momentum_Factor.csv", skiprows=13)
mom = mom_raw.copy()
mom["Date"] = pd.to_datetime(
    mom["Unnamed: 0"].astype(str), format="%Y%m", errors="coerce"
)
mom = mom.dropna(subset=["Date"])
mom["Mom"] = pd.to_numeric(mom["Mom"], errors="coerce")
mom = mom.set_index("Date")
mom["UMD"] = mom["Mom"] / 100.0
mom = mom[["UMD"]]
mom = mom[~mom.index.duplicated()]

# Load 10 industries
lines_ind = open("finance_hw4/10_Industry_Portfolios.csv").read().splitlines()
data_ind = [ln.strip() for ln in lines_ind if re.match(r"^\s*\d{6}", ln)]
dates_ind = []
vals_ind = []
for ln in data_ind:
    parts = [p for p in ln.split(",") if p != ""]
    d = parts[0].strip()
    nums = [float(p) for p in parts[1:]]
    if len(nums) == 10:
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
ind10 = ind10[~ind10.index.duplicated()]

# Load 25 portfolios
lines25 = open("finance_hw4/Developed_25_Portfolios_ME_BE-ME.csv").read().splitlines()
data25 = [ln.strip() for ln in lines25 if re.match(r"^\s*\d{6}", ln)]
dates25 = []
vals25 = []
for ln in data25:
    parts = [p for p in ln.split(",") if p != ""]
    d = parts[0].strip()
    nums = [float(p) for p in parts[1:]]
    if len(nums) == 25:
        dates25.append(d)
        vals25.append(nums)
p25 = pd.DataFrame(vals25, columns=[f"P{i+1}" for i in range(25)])
p25["Date"] = pd.to_datetime(dates25, format="%Y%m")
p25 = p25.set_index("Date") / 100.0
p25 = p25[~p25.index.duplicated()]

# factors
factors5 = ff5[["MKT", "SMB", "HML", "RMW", "CMA"]]
factors6 = factors5.join(mom, how="inner")

# excess returns
comm25 = p25.index.intersection(ff5.index)
comm10 = ind10.index.intersection(ff5.index)
ret25_ex = p25.loc[comm25].sub(ff5.loc[comm25, "RF"], axis=0)
ret10_ex = ind10.loc[comm10].sub(ff5.loc[comm10, "RF"], axis=0)


def regress_all(R_df, F_df):
    common = R_df.index.intersection(F_df.index)
    R = R_df.loc[common]
    F = F_df.loc[common]
    X = sm.add_constant(F)
    alphas = []
    tstats = []
    for col in R.columns:
        model = sm.OLS(R[col], X).fit()
        alphas.append(model.params[0])
        tstats.append(model.tvalues[0])
    return pd.DataFrame({"alpha": alphas, "tstat": tstats}, index=R.columns)


alpha25_ff5 = regress_all(ret25_ex, factors5)
alpha25_6 = regress_all(ret25_ex, factors6)
alpha10_ff5 = regress_all(ret10_ex, factors5)
alpha10_6 = regress_all(ret10_ex, factors6)

# 3.2 25 portfolios
print(alpha25_ff5)
print(alpha25_6)

# 3.3 10 industries
print(alpha10_ff5)
print(alpha10_6)

# output to csv
alpha25_ff5.to_csv("finance_hw4/alpha25_ff5.csv")
alpha25_6.to_csv("finance_hw4/alpha25_6.csv")
alpha10_ff5.to_csv("finance_hw4/alpha10_ff5.csv")
alpha10_6.to_csv("finance_hw4/alpha10_6.csv")
