# =====================================================================
# Berk & Green (2004) baseline simulation
#
#   Cell 1  Imports and inputs        <- the ONLY place any number is set
#   Cell 2  Model functions
#   Cell 3  Formula check (old step 1)
#   Cell 4  Run the simulation (old steps 2 and 3)
#   Cell 5  Checks with explicit rejection thresholds (old step 4)
#   Cell 6  Plot
#   Cell 7  Replication of the paper's Figure 1 (flow of new funds by fund age)
# =====================================================================

#%% 1. Imports and inputs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.max_columns", None)   # show every column
pd.set_option("display.width", 200)

# ---- Model parameters (Table 1 of the paper) ----
# Every function below reads its parameters from this dictionary.
params = {
    "f": 0.015,       # fee
    "sigma": 0.20,    # sd of luck (noise in returns)
    "phi0": 0.065,    # starting belief about skill (prior mean)
    "eta": 0.06,      # starting uncertainty about skill (prior sd)
    "phi_bar": 0.03,  # exit threshold: fund closes if belief falls below this
    "a": 1,           # cost scale (only affects fund size in money terms)
    "T": 20,          # number of years each fund is followed
}

# ---- Simulation settings ----
N = 1000           # number of funds
SEED = 1           # random seed, so results can be reproduced exactly
SIG_LEVEL = 0.05   # rejection threshold for the statistical checks (5%)

# ---- Figure 1 replication settings (cell 7) ----
N_FIG = 5000                 # number of funds for the Figure 1 replication
FIG1_SEEDS = [1, 2, 3, 4, 5] # several seeds, to show the result doesn't depend on luck
FIG1_AGES = [2, 5, 10, 20]   # fund ages shown in the paper's Figure 1

# Note: omega (= 1/sigma^2) and gamma (= 1/eta^2) are deliberately NOT stored here.
# They are calculated from sigma and eta inside the functions, so they can never get out of step with them.


#%% 2. Model functions

def precisions(params):
    """Return (omega, gamma): precision of returns and of the prior."""
    omega = 1 / params["sigma"]**2   # 25 with the baseline values
    gamma = 1 / params["eta"]**2     # ~277 with the baseline values
    return omega, gamma


def investor_return(R, phi_old, params):
    """Return to investors after costs and fees, above the index.
    Uses LAST year's belief phi_old (the return is earned before beliefs update)."""
    return (2 * params["f"] / phi_old) * (R - phi_old)


def update_belief(phi_old, R, t, params):
    """Block 5 (eq. 28): new belief = weighted average of old belief and this year's return."""
    omega, gamma = precisions(params)
    return ((gamma + (t - 1) * omega) * phi_old + omega * R) / (gamma + t * omega)


def fund_size(phi, params):
    """Total fund size with quadratic costs (eq. 37)."""
    return phi**2 / (4 * params["a"] * params["f"])


def simulate_one_fund(params, fund_id):
    """Simulate ONE fund for up to T years. Returns a list of rows (one per year)."""

    # Happens once per fund
    alpha = np.random.normal(params["phi0"], params["eta"])   # true skill (Block 2)
    phi = params["phi0"]                                      # starting belief
    q_old = fund_size(phi, params)                            # starting fund size

    results = []

    # Happens every year
    for t in range(1, params["T"] + 1):

        # Block 1: return = skill + luck
        e = np.random.normal(0, params["sigma"])
        R = alpha + e

        # Investor return, using last year's belief (Block 6: return comes first)
        r = investor_return(R, phi, params)

        # Block 5: update belief
        phi = update_belief(phi, R, t, params)

        # Exit rule
        closed = phi < params["phi_bar"]

        # New fund size and flow (a closing fund has size 0, so flow = -100%)
        q = 0 if closed else fund_size(phi, params)
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


def run_simulation(params, N, seed):
    """Simulate N funds and return one table with one row per fund-year.
    The seed is set ONCE here, so every fund gets different draws but the
    whole run can be reproduced exactly."""
    np.random.seed(seed)
    all_rows = []
    for i in range(N):
        all_rows.extend(simulate_one_fund(params, i))
    return pd.DataFrame(all_rows)


#%% 3. Formula check (old step 1): one fund, one year, return set by hand
# This tests the SAME functions the simulation uses, with R = 20% chosen by hand, against values worked out with a calculator.

R_test = 0.20
r_test = investor_return(R_test, params["phi0"], params)
phi_test = update_belief(params["phi0"], R_test, 1, params)
q_test = fund_size(phi_test, params)

hand_values = {"r": (r_test, 0.0623), "phi": (phi_test, 0.0761), "q": (q_test, 0.0966)}

print("FORMULA CHECK (one year, R = 0.20)")
for name, (computed, expected) in hand_values.items():
    ok = abs(computed - expected) < 0.0001
    print(f"  {name:4s} computed {computed:.4f}, expected {expected:.4f}  ->  {'PASS' if ok else 'FAIL'}")
print()


#%% 4. Run the simulation (old steps 2 and 3)
df = run_simulation(params, N, SEED)

T = params["T"]
survivors = df[(df["t"] == T) & (df["closed"] == False)]

print(f"Total fund-years simulated: {len(df)}")
print(f"Funds still open after {T} years: {len(survivors)} of {N}")
print()

# One fund's whole story (what old step 2 showed): fund 0
print("Fund 0, year by year:")
print(df[df["fund"] == 0][["t", "alpha", "R", "r", "phi", "q", "flow", "closed"]].round(4).to_string(index=False))
print()


#%% 5. Checks with explicit rejection thresholds (old step 4)
# Each check states the model's prediction as a null hypothesis (H0).
# Reject H0 if the p-value is below SIG_LEVEL.
# "PASS" means the test does NOT reject the model's prediction
#
# Assumption: the tests treat fund-years as independent observations.
# Check 4 supports this for investor returns, but it is an assumption.
#
# Note: at a 5% level, a correct model will still "fail" about 1 run in 20 by chance.

summary = []   # collects one line per check for the summary table at the end


def verdict(p):
    return "FAIL (reject H0)" if p < SIG_LEVEL else "PASS (do not reject H0)"


# ---- Check 1: investors earn zero on average (Block 4, eq. 3) ----
# H0: mean investor return r = 0
t1 = stats.ttest_1samp(df["r"], 0)
print("CHECK 1: Mean investor return")
print("  H0: mean r = 0   |   test: one-sample t-test   |   reject if p < "
      f"{SIG_LEVEL}")
print(f"  mean r = {df['r'].mean():.4f}, t = {t1.statistic:.3f}, p = {t1.pvalue:.3f}  ->  {verdict(t1.pvalue)}")
print("  Mean r by year (descriptive):")
print(df.groupby("t")["r"].mean().round(4).to_string())
print()
summary.append(["1. Mean investor return = 0", f"{df['r'].mean():.4f}", f"{t1.pvalue:.3f}", verdict(t1.pvalue)])

# ---- Check 2: survival ----
# TODO: needs a comparand before it can have a threshold
open_end = df[df["closed"] == False]                    # open at the end of year t
survival = open_end.groupby("t")["fund"].count() / N
print("CHECK 2: Share of funds still open at the end of each year (descriptive, comparand TO DO)")
print(survival.round(3).to_string())
print()
summary.append(["2. Survival vs paper", f"{survival.loc[T]:.3f} at year {T}", "-", "TO DO: needs comparand"])

# ---- Check 3: survivors' true skill matches the public belief about them ----
# H0: among survivors, mean alpha = mean phi (beliefs are correct on average)
# Paired t-test, because each surviving fund has both an alpha and a phi.
t3 = stats.ttest_rel(survivors["alpha"], survivors["phi"])
all_alpha = df.groupby("fund")["alpha"].first()          # one alpha per fund
print("CHECK 3: Survivors' true skill vs public belief")
print("  H0: mean alpha = mean phi among survivors   |   test: paired t-test   |   reject if p < "
      f"{SIG_LEVEL}")
print(f"  survivors ({len(survivors)}): mean alpha = {survivors['alpha'].mean():.4f}, "
      f"mean phi = {survivors['phi'].mean():.4f}")
print(f"  t = {t3.statistic:.3f}, p = {t3.pvalue:.3f}  ->  {verdict(t3.pvalue)}")
print(f"  (descriptive: mean alpha of all {N} funds = {all_alpha.mean():.4f}, so survivors are more skilled)")
print()
summary.append(["3. Survivors: mean alpha = mean phi",
                f"{survivors['alpha'].mean():.4f} vs {survivors['phi'].mean():.4f}",
                f"{t3.pvalue:.3f}", verdict(t3.pvalue)])

# ---- Check 4: past returns don't predict future returns ----
# H0: correlation between this year's r and last year's r (same fund) = 0
df["r_last_year"] = df.groupby("fund")["r"].shift(1)
pairs = df.dropna(subset=["r_last_year"])                # rows that have a previous year
t4 = stats.pearsonr(pairs["r"], pairs["r_last_year"])
print("CHECK 4: Predictability of investor returns")
print("  H0: corr(r this year, r last year) = 0   |   test: Pearson correlation test   |   reject if p < "
      f"{SIG_LEVEL}")
print(f"  corr = {t4.statistic:.4f}, p = {t4.pvalue:.3f}  ->  {verdict(t4.pvalue)}")
print()
summary.append(["4. No return predictability", f"{t4.statistic:.4f}", f"{t4.pvalue:.3f}", verdict(t4.pvalue)])

# ---- Summary table ----
summary_df = pd.DataFrame(summary, columns=["Check", "Value", "p-value", "Verdict"])
print(f"SUMMARY (seed = {SEED}, N = {N}, significance level = {SIG_LEVEL})")
print(summary_df.to_string(index=False))
print()


#%% 6. Plot: flows follow performance, more strongly for young funds
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
print("Plot saved as flow_performance.png")


#%% 7. Replication of the paper's Figure 1 (p. 24): flow of new funds by fund age
#
# What Figure 1 shows:
#   x-axis: investor return r in the year after the fund reaches age a
#   y-axis: flow of NEW money (eq. 46, p. 22) = (q_t - q_{t-1}(1 + r)) / q_{t-1},
#           set to -1 if the fund closes
#   Each curve: the AVERAGE new flow across all funds alive at age a (eq. 47, p. 23)
#
# Age convention: the caption says the "2 year" curve uses the return from year 2
# to year 3, so paper age a corresponds to t = a + 1 in this code.

def eq46_new_flow(r, t, params):
    """Flow of new money for a fund that does NOT close (eq. 46, p. 22)."""
    omega, gamma = precisions(params)
    k = omega / (gamma + t * omega)                 # weight on new information
    f = params["f"]
    return (k / f - 1) * r + (k / (2 * f))**2 * r**2


def figure1_curve(df, age, params, r_grid):
    """Average new flow at each return in r_grid, for funds alive at the end of year `age` (eq. 47)."""
    t = age + 1                                                       # the year the return happens
    phis = df[(df["t"] == age) & (df["closed"] == False)]["phi"].to_numpy()

    # Every combination of (return on the grid) x (surviving fund):
    # rows = returns, columns = funds
    r = r_grid[:, None]
    phi_old = phis[None, :]

    # Turn the investor return r back into the manager's return R:
    # r = (2f/phi)(R - phi)  =>  R = phi + r / (2f/phi)
    R = phi_old + r / (2 * params["f"] / phi_old)

    # Would the fund close? Same functions as the simulation (equivalent to eq. 44)
    phi_new = update_belief(phi_old, R, t, params)
    closes = phi_new < params["phi_bar"]

    flows = np.where(closes, -1.0, eq46_new_flow(r, t, params))
    return flows.mean(axis=1)                                         # average over funds


r_grid = np.linspace(-0.9, 0.9, 361)

# ---- Run the simulation once per seed and compute all four curves ----
curves = {}       # curves[(seed, age)] = array of average new flows
sims = {}         # keep each simulated table for the data-points check
for seed in FIG1_SEEDS:
    df_fig = run_simulation(params, N_FIG, seed)
    sims[seed] = df_fig
    for age in FIG1_AGES:
        curves[(seed, age)] = figure1_curve(df_fig, age, params, r_grid)

# ---- Plot: same layout as the paper's Figure 1 ----
colours = {2: "tab:red", 5: "tab:green", 10: "tab:blue", 20: "tab:purple"}
fig, ax = plt.subplots(figsize=(8, 5.5))

for age in FIG1_AGES:
    for i, seed in enumerate(FIG1_SEEDS):
        ax.plot(r_grid, curves[(seed, age)], color=colours[age], linewidth=1,
                alpha=0.6, label=f"{age} years" if i == 0 else None)

# Check: actual simulated outcomes (seed 1) as dots, averaged within return bins.
# Where there is enough data, the dots should sit on the curves.
df1 = sims[FIG1_SEEDS[0]].copy()
df1["new_flow"] = np.where(df1["closed"], -1.0, df1["flow"] - df1["r"])
bins = np.arange(-0.4, 0.42, 0.04)
for age in FIG1_AGES:
    year = df1[df1["t"] == age + 1].copy()
    year["bin"] = pd.cut(year["r"], bins)
    binned = year.groupby("bin", observed=True).agg(r=("r", "mean"), nf=("new_flow", "mean"), n=("r", "size"))
    binned = binned[binned["n"] >= 30]                   # only bins with enough funds
    ax.scatter(binned["r"], binned["nf"], color=colours[age], s=18, edgecolor="black", linewidth=0.4, zorder=3)

ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlim(-0.9, 0.9)
ax.set_ylim(-1.1, 2.3)
ax.set_xlabel("Investor return r")
ax.set_ylabel("Flow of new funds")
ax.set_title(f"Replication of Figure 1: flow of new funds by fund age\n"
             f"(N = {N_FIG}, seeds {FIG1_SEEDS}; dots = simulated outcomes, seed {FIG1_SEEDS[0]})")
ax.legend(title="Fund age")
fig.savefig("figure1_replication.png", dpi=150, bbox_inches="tight")
print("Plot saved as figure1_replication.png")
print()

# ---- Table of values, to compare with the paper's Figure 1 ----
# Mean across seeds, and the largest difference between seeds (shows how much luck matters).
check_r = [-0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4]
rows = []
for age in FIG1_AGES:
    for rv in check_r:
        idx = np.argmin(np.abs(r_grid - rv))
        vals = [curves[(s, age)][idx] for s in FIG1_SEEDS]
        rows.append([age, rv, np.mean(vals), max(vals) - min(vals)])
fig1_table = pd.DataFrame(rows, columns=["age", "r", "mean new flow", "range across seeds"])
print("FIGURE 1 VALUES (compare with values read off the paper's Figure 1)")
print(fig1_table.round(3).to_string(index=False))