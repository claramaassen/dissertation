import numpy as np
import pandas as pd
 
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
 
pd.set_option("display.max_columns", None)   # show every column
pd.set_option("display.width", 200)
print(df.head(25))
print()
print(f"Total fund-years simulated: {len(df)}")
survivors = df[(df["t"] == T) & (df["closed"] == False)]
print(f"Funds still open after {T} years: {len(survivors)} of {N}")