import pandas as pd
import numpy as np
import re
import statsmodels.api as sm

########################################################
#             MGT6078 – Replicate AQR Tables 1–4
#         (Factor-on-factor regressions with CSV output)
########################################################


############################
# 1. Load FF5 factors
############################

ff5_raw = pd.read_csv("finance_hw4/F-F_Research_Data_5_Factors_2x3.csv", skiprows=4)

mask = ff5_raw["Date"].astype(str).str.match(r"^\d{6}$")
ff5 = ff5_raw[mask].copy()

for col in ff5.columns[1:]:
    ff5[col] = pd.to_numeric(ff5[col], errors="coerce")

ff5["Date"] = pd.to_datetime(ff5["Date"].astype(str), format="%Y%m")
ff5 = ff5.set_index("Date") / 100.0
ff5 = ff5.rename(columns={" Mkt-RF": "MKT"})
ff5 = ff5[~ff5.index.duplicated(keep="first")]
ff5 = ff5.loc[ff5.index >= "1963-07-01"]


############################
# 2. Load Momentum (UMD)
############################

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
mom = mom.loc[mom.index >= "1963-07-01"]
mom = mom[~mom.index.duplicated(keep="first")]


############################
# 3. Load HML Devil (USA)
############################

dev = pd.read_excel(
    "finance_hw4/The Devil in HMLs Details Factors Monthly.xlsx", sheet_name="HML Devil"
)

# USA column = column index 24 in your sheet
hml = dev.iloc[18:, [0, 24]].copy()
hml.columns = ["Date", "HML_DEV"]

# Convert date
hml["Date"] = pd.to_datetime(hml["Date"], errors="coerce")
hml = hml.dropna(subset=["Date"])

# Convert to decimal
hml["HML_DEV"] = pd.to_numeric(hml["HML_DEV"], errors="coerce") / 100.0

# Set index
hml = hml.set_index("Date")

# ★★★ KEY STEP: Align to monthly frequency
hml.index = hml.index.to_period("M").to_timestamp()

hml = hml.loc[hml.index >= "1963-07-01"]
hml = hml[~hml.index.duplicated(keep="first")]


############################
# 4. Build Factor Sets
############################

# FF5
factors_ff5 = ff5[["MKT", "SMB", "HML", "RMW", "CMA"]]

# FF5 + UMD
factors_ff5_umd = factors_ff5.join(mom, how="inner")

# DEV (HML replaced with HML_DEV)
factors_dev = ff5[["MKT", "SMB", "RMW", "CMA"]].join(hml, how="inner")
factors_dev = factors_dev[["MKT", "SMB", "HML_DEV", "RMW", "CMA"]]

# DEV + UMD
factors_dev_umd = factors_dev.join(mom, how="inner")


############################
# 5. Regression Table Function
############################


def factor_reg_table(factors: pd.DataFrame):
    cols = factors.columns
    rows = []

    for dep in cols:
        y = factors[dep]
        X = factors.drop(columns=[dep])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        row = {
            "Factor": dep,
            "Intercept_Ann%": model.params["const"] * 12 * 100,
            "Intercept_t": model.tvalues["const"],
            "R2": model.rsquared,
        }

        for c in X.columns:
            if c != "const":
                row[c] = model.params[c]
                row[c + "_t"] = model.tvalues[c]

        rows.append(row)

    return pd.DataFrame(rows).set_index("Factor")


############################
# 6. Generate All Four Tables
############################

table1 = factor_reg_table(factors_ff5)
table2 = factor_reg_table(factors_ff5_umd)
table3 = factor_reg_table(factors_dev)
table4 = factor_reg_table(factors_dev_umd)


############################
# 7. Output to CSV
############################

table1.to_csv("table1_ff5.csv")
table2.to_csv("table2_ff5_umd.csv")
table3.to_csv("table3_dev.csv")
table4.to_csv("table4_dev_umd.csv")

print("\n===== DONE =====")
print("Generated CSV files:")
print("table1_ff5.csv")
print("table2_ff5_umd.csv")
print("table3_dev.csv")
print("table4_dev_umd.csv")
print("================\n")
