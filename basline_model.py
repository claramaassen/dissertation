# =====================================================================
# Berk & Green (2004) baseline simulation
#
#   Cell 1  Imports and inputs        <- the ONLY place any number is set
#   Cell 2  Model functions
#   Cell 3  Formula check (old step 1)
#   Cell 4  Run the simulation (old steps 2 and 3)
#   Cell 5  Checks with explicit rejection thresholds (old step 4)
#   Cell 6  Plots (flow-performance, survival vs Proposition 4)
#   Cell 7  Replication of the paper's Figure 1 (flow of new funds by fund age)
#   Cell 8  Power check: rejection rates of every check across many seeds
#   Cell 9  Figure 1 dots vs curves: eq. 47 vs conditioning on the realised return
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

# ---- Power check settings (cell 8) ----
RUN_POWER_CHECK = True       # set to False to skip cell 8 (it takes about a minute)
POWER_N = 1000               # funds per run (same as the main run)
POWER_SEEDS = 1000           # number of runs; each run repeats checks 1-5

# ---- Figure 1 dots-vs-curves settings (cell 9) ----
N_BIG = 50000                # large N, so small-sample noise can be ruled out
DOTS_AGES = [2, 5, 10]       # ages with enough funds at very negative returns
MIN_BIN_COUNT = 200          # only show return bins with at least this many funds

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


def add_standardised_surprise(df, params):
    """Add columns phi_prev (belief at the start of the year) and z (standardised surprise).

    Given everything known at the start of year t, the model says
        R_t - phi_{t-1} ~ Normal(0, 1/omega + 1/(gamma + (t-1)*omega))
    (luck plus remaining uncertainty about skill). Dividing by that sd gives z,
    which should be Normal(0, 1) for EVERY fund and year. The investor return r is
    this surprise times the active share 2f/phi, so r is more volatile for funds with
    low beliefs (eq. 50). z removes that difference in volatility."""
    omega, gamma = precisions(params)
    df = df.copy()
    df["phi_prev"] = df.groupby("fund")["phi"].shift(1).fillna(params["phi0"])
    sd = np.sqrt(1 / omega + 1 / (gamma + (df["t"] - 1) * omega))
    df["z"] = (df["R"] - df["phi_prev"]) / sd
    return df


def survival_prop4(params, n_grid=2000):
    """Survival probabilities P_1, ..., P_T from the paper's Proposition 4 (p. 18), computed
    numerically. This is the comparand for check 2: it comes from the model's FORMULA,
    not from the simulation, so the two can be compared.

    How it works: a fund survives to year t if its belief phi has stayed above phi_bar in
    every year so far. Averaging over managers' unknown skill (as Proposition 4 does),
    each year's change in belief is Normal with mean 0 and variance
        1/(gamma + (t-1)*omega) - 1/(gamma + t*omega)
    (the drop in the market's uncertainty about skill that year). So we track the
    density of phi on a grid, one year at a time: spread it out by that Normal change,
    then cut off the part below phi_bar (those funds close). The area that is left
    is P_t. This gives the same answer as eqs. 29-30 in Propositions 3-4."""
    omega, gamma = precisions(params)
    phi0, eta, phi_bar, T = params["phi0"], params["eta"], params["phi_bar"], params["T"]

    grid = np.linspace(phi_bar, phi0 + 8 * eta, n_grid)   # beliefs at or above phi_bar
    step = grid[1] - grid[0]

    P = []
    density = None
    for t in range(1, T + 1):
        sd = np.sqrt(1 / (gamma + (t - 1) * omega) - 1 / (gamma + t * omega))
        if t == 1:
            # Year 1: everyone starts at phi0
            density = stats.norm.pdf(grid, phi0, sd)
        else:
            # Spread last year's surviving density by this year's change in belief
            kernel = stats.norm.pdf(grid[:, None], grid[None, :], sd)   # rows: new phi, cols: old phi
            density = kernel @ density * step
        # Only the grid at or above phi_bar is kept, so funds below it drop out automatically
        P.append(density.sum() * step)
    return pd.Series(P, index=range(1, T + 1), name="P_t (Prop. 4)")


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
# Cell 8 checks whether each test rejects at the right rate (5%) across many seeds.
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

# ---- Check 2: survival vs the paper's formula (Proposition 4, p. 18; Figure 2, p. 27) ----
# Comparand: P_t, the survival probability the model predicts (Proposition 4).
# For each year t: H0: the true survival probability equals P_t.
# Test: binomial test (survivors out of N, against probability P_t).
# Because we test 20 years at once, a correct model would "fail" at least one of them
# fairly often by chance. So we use a Bonferroni correction: the check fails only if
# some year has p < SIG_LEVEL / T. (This is conservative, i.e. it fails less easily.)
open_end = df[df["closed"] == False]                    # open at the end of year t
n_open = open_end.groupby("t")["fund"].count().reindex(range(1, T + 1), fill_value=0)
survival = n_open / N
P_theory = survival_prop4(params)

pvals2 = [stats.binomtest(int(n_open.loc[t]), N, P_theory.loc[t]).pvalue for t in range(1, T + 1)]
check2 = pd.DataFrame({"simulated": survival, "Prop. 4": P_theory,
                       "difference": survival - P_theory, "p-value": pvals2})
bonf = SIG_LEVEL / T
p2_min = min(pvals2)
verdict2 = "FAIL (reject H0)" if p2_min < bonf else "PASS (do not reject H0)"

print("CHECK 2: Survival vs Proposition 4")
print(f"  H0 (each year t): survival probability = P_t from Prop. 4   |   test: binomial test   |   "
      f"reject if any p < {SIG_LEVEL}/{T} = {bonf:.4f} (Bonferroni)")
print(check2.round(4).to_string())
print(f"  smallest p-value = {p2_min:.3f}  ->  {verdict2}")
print()
summary.append(["2. Survival = Prop. 4 (all years)",
                f"{survival.loc[T]:.3f} vs {P_theory.loc[T]:.3f} at year {T}",
                f"{p2_min:.3f} (min)", verdict2])

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
# H0: corr(this year's surprise, last year's surprise) = 0, same fund.
# Why the standardised surprise z and not r itself: in the model the volatility of r
# differs across funds and persists over time (eq. 50: it depends on the belief phi).
# The Pearson test assumes every pair is equally noisy, so on r it rejects too often
# even when the true correlation is exactly 0 (see cell 8). z has the same volatility
# for every fund-year, so the test is valid on z. r is still reported, as a description.
df = add_standardised_surprise(df, params)
df["z_last_year"] = df.groupby("fund")["z"].shift(1)
df["r_last_year"] = df.groupby("fund")["r"].shift(1)
pairs = df.dropna(subset=["z_last_year"])                # rows that have a previous year
t4 = stats.pearsonr(pairs["z"], pairs["z_last_year"])
corr_r = pairs["r"].corr(pairs["r_last_year"])
print("CHECK 4: Predictability of returns")
print("  H0: corr(z this year, z last year) = 0   |   test: Pearson correlation test on z   |   reject if p < "
      f"{SIG_LEVEL}")
print(f"  corr(z) = {t4.statistic:.4f}, p = {t4.pvalue:.3f}  ->  {verdict(t4.pvalue)}")
print(f"  (descriptive: corr(r this year, r last year) = {corr_r:.4f}; sd of z = {df['z'].std():.3f}, model says 1)")
print()
summary.append(["4. No return predictability (z)", f"{t4.statistic:.4f}", f"{t4.pvalue:.3f}", verdict(t4.pvalue)])

# ---- Check 5: share of managers skilled enough to beat their fee (Figure 4, p. 29) ----
# The paper: "about 80% of managers" have alpha above the 1.5% fee. With skill drawn from
# Normal(phi0, eta^2), the exact share is 1 - Phi((f - phi0) / eta).
# H0: the share of simulated managers with alpha > f equals this value.
# Test: binomial test (managers with alpha > f, out of N).
share_theory = 1 - stats.norm.cdf((params["f"] - params["phi0"]) / params["eta"])
n_beat = int((all_alpha > params["f"]).sum())
share_sim = n_beat / N
t5 = stats.binomtest(n_beat, N, share_theory)
print("CHECK 5: Share of managers whose skill beats the fee (paper's Figure 4: about 80%)")
print("  H0: share with alpha > f = 1 - Phi((f - phi0)/eta)   |   test: binomial test   |   reject if p < "
      f"{SIG_LEVEL}")
print(f"  simulated share = {share_sim:.3f}, theory = {share_theory:.3f}, p = {t5.pvalue:.3f}  ->  {verdict(t5.pvalue)}")
print()
summary.append(["5. Share beating fee (Fig. 4)", f"{share_sim:.3f} vs {share_theory:.3f}",
                f"{t5.pvalue:.3f}", verdict(t5.pvalue)])

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

# ---- Survival: simulation vs Proposition 4 (compare with the dark bars of the paper's Figure 2) ----
fig, ax = plt.subplots(figsize=(8, 5))
years = np.arange(1, T + 1)
ax.bar(years - 0.2, survival.values, width=0.4, label=f"Simulated (N = {N}, seed {SEED})")
ax.bar(years + 0.2, P_theory.values, width=0.4, label="Proposition 4 (formula)")
ax.set_xticks(years)
ax.set_ylim(0, 1)
ax.set_xlabel("Years in business")
ax.set_ylabel("Share of funds surviving")
ax.set_title("Fund survival: simulation vs Proposition 4 (cf. paper's Figure 2)")
ax.legend()
fig.savefig("survival.png", dpi=150, bbox_inches="tight")
print("Plot saved as survival.png")


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
    r = r_grid[:, None]
    phi_old = phis[None, :]
    R = phi_old + r / (2 * params["f"] / phi_old)
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


#%% 8. Power check: how often does each check reject, across many seeds?
# One seed can fail a check by chance (at 5%, about 1 run in 20). To rule out code
# errors, repeat every check for POWER_SEEDS independent runs. If the code and the
# tests are correct, each check should reject in about 5% of runs.
# The last column tests that: H0: rejection rate = 5% (binomial test).

def run_checks(df, params, P_theory):
    """Run checks 1-5 on one simulated table. Returns one p-value per check."""
    T = params["T"]
    n = len(df["fund"].unique())
    survivors = df[(df["t"] == T) & (df["closed"] == False)]
    alpha = df.groupby("fund")["alpha"].first()

    p1 = stats.ttest_1samp(df["r"], 0).pvalue

    n_open = df[df["closed"] == False].groupby("t")["fund"].count().reindex(range(1, T + 1), fill_value=0)
    p2 = min(stats.binomtest(int(n_open.loc[t]), n, P_theory.loc[t]).pvalue for t in range(1, T + 1)) * T  # Bonferroni
    p3 = stats.ttest_rel(survivors["alpha"], survivors["phi"]).pvalue

    d = add_standardised_surprise(df, params)
    d["z_lag"] = d.groupby("fund")["z"].shift(1)
    d["r_lag"] = d.groupby("fund")["r"].shift(1)
    q = d.dropna(subset=["z_lag"])
    p4_z = stats.pearsonr(q["z"], q["z_lag"]).pvalue
    p4_r = stats.pearsonr(q["r"], q["r_lag"]).pvalue          # the old version of check 4, for comparison

    share = 1 - stats.norm.cdf((params["f"] - params["phi0"]) / params["eta"])
    p5 = stats.binomtest(int((alpha > params["f"]).sum()), n, share).pvalue
    return [p1, p2, p3, p4_z, p4_r, p5]


if RUN_POWER_CHECK:
    check_names = ["1. Mean investor return = 0", "2. Survival = Prop. 4 (Bonferroni)",
                   "3. Survivors: mean alpha = mean phi", "4. No predictability, on z (used)",
                   "4. No predictability, on r (old)", "5. Share beating fee (Fig. 4)"]
    P_theory_power = survival_prop4(params)
    pvals = np.array([run_checks(run_simulation(params, POWER_N, s), params, P_theory_power)
                      for s in range(1, POWER_SEEDS + 1)])

    rows = []
    for j, name in enumerate(check_names):
        k = int((pvals[:, j] < SIG_LEVEL).sum())
        rows.append([name, k, k / POWER_SEEDS, stats.binomtest(k, POWER_SEEDS, SIG_LEVEL).pvalue])
    power_table = pd.DataFrame(rows, columns=["Check", "rejections", "rejection rate", "p (rate = 5%)"])
    print(f"POWER CHECK: rejection rates across {POWER_SEEDS} seeds (N = {POWER_N}, level = {SIG_LEVEL})")
    print("  A correct test on correct code rejects in about 5% of runs.")
    print(power_table.round(3).to_string(index=False))
    print()


#%% 9. Figure 1: why do the dots fall below the curves for bad returns?
# Two different questions give two different curves:
#   (a) eq. 47 (the paper, and the curves in cell 7): average over ALL funds alive at
#       age a, as if every one of them received return r. Each fund gets equal weight.
#   (b) the dots: average over the funds that ACTUALLY earned return r.
# In the model, r is more volatile for funds with low beliefs (eq. 50: active share
# 2f/phi is larger). So among the funds that actually earned a very bad r, low-belief
# funds are over-represented, and more of them close. To get (b) from the formula,
# weight each fund by how likely it was to earn r given its belief (Bayes' rule):
#   weight = Normal density of r with mean 0 and sd (2f/phi) * sqrt(1/omega + 1/(gamma + (t-1)*omega))
# Using a large N rules out small-sample noise as the explanation.

def figure1_curve_weighted(df, age, params, r_values):
    """Like figure1_curve, but each fund is weighted by the likelihood of earning r (version b)."""
    omega, gamma = precisions(params)
    t = age + 1
    phis = df[(df["t"] == age) & (df["closed"] == False)]["phi"].to_numpy()
    r = np.asarray(r_values)[:, None]
    phi_old = phis[None, :]
    R = phi_old + r / (2 * params["f"] / phi_old)
    closes = update_belief(phi_old, R, t, params) < params["phi_bar"]
    flows = np.where(closes, -1.0, eq46_new_flow(r, t, params))
    sd_r = (2 * params["f"] / phi_old) * np.sqrt(1 / omega + 1 / (gamma + (t - 1) * omega))
    w = stats.norm.pdf(r, 0, sd_r)
    return (flows * w).sum(axis=1) / w.sum(axis=1)


df_big = run_simulation(params, N_BIG, SEED)
df_big["new_flow"] = np.where(df_big["closed"], -1.0, df_big["flow"] - df_big["r"])

fig, ax = plt.subplots(figsize=(8, 5.5))
r_neg = np.linspace(-0.45, 0.1, 221)
dots_rows = []
for age in DOTS_AGES:
    c = colours[age]
    ax.plot(r_neg, figure1_curve(df_big, age, params, r_neg), color=c, linestyle="--", linewidth=1,
            label=f"{age} yrs: eq. 47 (every fund gets r)")
    ax.plot(r_neg, figure1_curve_weighted(df_big, age, params, r_neg), color=c, linewidth=1.5,
            label=f"{age} yrs: weighted by P(r | phi)")

    year = df_big[df_big["t"] == age + 1].copy()
    year["bin"] = pd.cut(year["r"], np.arange(-0.42, 0.12, 0.04))
    b = year.groupby("bin", observed=True).agg(r=("r", "mean"), dots=("new_flow", "mean"), n=("r", "size"))
    b = b[b["n"] >= MIN_BIN_COUNT]
    ax.scatter(b["r"], b["dots"], color=c, s=20, edgecolor="black", linewidth=0.4, zorder=3)

    b["eq. 47"] = figure1_curve(df_big, age, params, b["r"].to_numpy())
    b["weighted"] = figure1_curve_weighted(df_big, age, params, b["r"].to_numpy())
    b.insert(0, "age", age)
    dots_rows.append(b[b["r"] < -0.05])

ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlim(-0.45, 0.1)
ax.set_ylim(-1.1, 0.4)
ax.set_xlabel("Investor return r")
ax.set_ylabel("Flow of new funds")
ax.set_title(f"Figure 1, negative returns: eq. 47 vs conditioning on the realised return\n"
             f"(N = {N_BIG}, seed {SEED}; dots = simulated outcomes)")
ax.legend(fontsize=8)
fig.savefig("figure1_dots_vs_curves.png", dpi=150, bbox_inches="tight")
print("Plot saved as figure1_dots_vs_curves.png")

dots_table = pd.concat(dots_rows)[["age", "n", "r", "dots", "eq. 47", "weighted"]]
print(f"FIGURE 1 DOTS vs CURVES (N = {N_BIG}, negative returns)")
print(dots_table.round(3).to_string(index=False))