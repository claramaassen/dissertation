import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
np.random.seed(1)  # set ONCE, at the top, not inside the function
 
# ---- Parameters (same as step 2) ----
f = 0.015        # fee
sigma = 0.20     # sd of luck
phi0 = 0.065     # starting belief about skill
eta = 0.06       # starting uncertainty about skill
phi_bar = 0.03   # exit threshold
a = 1            # cost scale (only affects fund size)
T = 20           # number of years
N = 1000         # number of funds
 
omega = 1 / sigma**2   # precision of returns (25)
gamma = 1 / eta**2     # precision of starting belief (~277)
 
 
# ---- Step 2, turned into a function: simulates ONE fund ----
def simulate_one_fund(fund_id):
 
    # Happens once per fund
    alpha = np.random.normal(phi0, eta)   # true skill
    phi = phi0                            # starting belief
    q_old = phi**2 / (4 * a * f)          # starting fund size
 
    results = []   # one row (dictionary) per year
 
    # Happens every year
    for t in range(1, T + 1):
 
        # Block 1: return = skill + luck
        e = np.random.normal(0, sigma)
        R = alpha + e
 
        # Investor return, using last year's belief
        r = (2 * f / phi) * (R - phi)
 
        # Block 5: update belief
        phi = ((gamma + (t - 1) * omega) * phi + omega * R) / (gamma + t * omega)
 
        # Exit rule
        closed = phi < phi_bar
 
        # New fund size and flow (a closing fund has size 0, so flow = -100%)
        if closed:
            q = 0
        else:
            q = phi**2 / (4 * a * f)
        flow = q / q_old - 1
 
        # Record this year as one row
        results.append({
            "fund": fund_id,
            "t": t,
            "alpha": alpha,
            "R": R,
            "r": r,
            "phi": phi,
            "q": q,
            "flow": flow,
            "closed": closed,
        })
 
        if closed:
            break      # fund closes and stays closed
 
        q_old = q      # this year's size becomes "last year's" for next year
 
    return results
 
 
# ---- Run the function for every fund ----
all_rows = []
for i in range(N):
    all_rows.extend(simulate_one_fund(i))
 
# ---- Put everything into one table: one row per fund-year ----
df = pd.DataFrame(all_rows)
 
 
# =====================================================================
# STEP 4: CHECKS - does the simulation behave the way the paper says?
# =====================================================================
 
# ---- Check 1: investors earn zero on average (Block 4) ----
print("CHECK 1: Average investor return r (should be close to 0)")
print(f"  All fund-years: {df['r'].mean():.4f}")
print("  By year:")
print(df.groupby("t")["r"].mean().round(4).to_string())
print()
 
# ---- Check 2: survival - share of funds still open each year ----
# A fund is "open at the end of year t" if it has a row for year t and did not close that year
open_end = df[df["closed"] == False]
survival = open_end.groupby("t")["fund"].count() / N
print("CHECK 2: Share of funds still open at the end of each year")
print(survival.round(3).to_string())
print()
 
# ---- Check 3: survivors are the skilled managers ----
survivors = df[(df["t"] == T) & (df["closed"] == False)]
all_alpha = df.groupby("fund")["alpha"].first()     # one alpha per fund
print("CHECK 3: Average true skill (alpha)")
print(f"  All {N} funds:        {all_alpha.mean():.4f}")
print(f"  {len(survivors)} survivors to year {T}: {survivors['alpha'].mean():.4f}")
print()
 
# ---- Check 4: past returns don't predict future returns ----
# For each fund, put last year's r next to this year's r, then see if they move together
df["r_last_year"] = df.groupby("fund")["r"].shift(1)
corr = df["r"].corr(df["r_last_year"])
print("CHECK 4: Correlation between this year's r and last year's r (should be close to 0)")
print(f"  {corr:.4f}")
print()
 
# ---- Check 5: flows follow performance, more strongly for young funds ----
# Leave out closing years (flow = -100%), which would squash the picture
still_open = df[df["closed"] == False]
young = still_open[still_open["t"] == 1]
old = still_open[still_open["t"] == 10]
 
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(young["r"], young["flow"], s=8, alpha=0.5, label="Year 1 (young funds)")
ax.scatter(old["r"], old["flow"], s=8, alpha=0.5, label="Year 10 (older funds)")
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Investor return r")
ax.set_ylabel("Flow (% change in fund size)")
ax.set_title("Flow-performance relationship by fund age")
ax.legend()
fig.savefig("flow_performance.png", dpi=150, bbox_inches="tight")
print("CHECK 5: Plot saved as flow_performance.png")