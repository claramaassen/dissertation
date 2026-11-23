# =====================================================================
# Berk & Green (2004) baseline simulation
#
#   Cell 1  Imports and inputs        <- the ONLY place any number is set
#   Cell 2  Model functions
#   Cell 3  Formula check (old step 1)
#   Cell 4  Run the simulation (old steps 2 and 3)
#   Cell 5  Checks with explicit rejection thresholds (old step 4)
#   Cell 6  Plots (flow-performance, survival vs Proposition 4)
#   Cell 7  Figure 1: eq. 47 from theory vs simulation (beliefs and curves)
#   Cell 8  Power check: rejection rates of every check across many seeds
#   Cell 9  Figure 1: funds that actually earned r (dots) vs eq. 47 (theory)
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
R_MIN, R_MAX, R_STEP = -0.9, 0.9, 0.005   # returns on the x-axis of Figure 1

# ---- Power check settings (cell 8) ----
RUN_POWER_CHECK = True       # set to False to skip cell 8 (it takes about a minute)
POWER_N = 1000               # funds per run (same as the main run)
POWER_SEEDS = 1000           # number of runs; each run repeats checks 1-5

# ---- Figure 1 dots-vs-curves settings (cell 9) ----
N_BIG = 50000                # large N, so small-sample noise can be ruled out
DOTS_AGES = [2, 5, 10]       # ages with enough funds at very negative returns
MIN_BIN_COUNT = 200          # only show return bins with at least this many funds
DOTS_R_MIN, DOTS_R_MAX = -0.42, 0.10   # range of returns shown in cell 9
BIN_WIDTH = 0.04             # width of the return bins for the dots

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


def belief_densities(params, n_grid=2000):
    omega, gamma = precisions(params)
    phi0, eta, phi_bar, T = params["phi0"], params["eta"], params["phi_bar"], params["T"]

    grid = np.linspace(phi_bar, phi0 + 8 * eta, n_grid)   # beliefs at or above phi_bar
    step = grid[1] - grid[0]

    F = {}
    for t in range(1, T + 1):
        sd_change = np.sqrt(1 / (gamma + (t - 1) * omega) - 1 / (gamma + t * omega))
        if t == 1:
            F[t] = stats.norm.pdf(grid, phi0, sd_change)
        else:
            # new density at each grid point = sum over last year's beliefs of
            # (last year's density) x (chance of moving from there to here) x step
            move = stats.norm.pdf(grid[:, None], grid[None, :], sd_change)  # rows: new phi, cols: old phi
            F[t] = move @ F[t - 1] * step
        # Only grid points >= phi_bar exist, so closed funds are dropped automatically.
    return grid, step, F


def survival_prop4(params):
    """Survival probabilities P_1, ..., P_T from Proposition 4 (p. 18):
    P_t = area under F_t = share of funds still alive at the end of year t.
    This is the comparand for check 2."""
    grid, step, F = belief_densities(params)
    P = [F[t].sum() * step for t in range(1, params["T"] + 1)]
    return pd.Series(P, index=range(1, params["T"] + 1), name="P_t (Prop. 4)")


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


#%% 7. Figure 1 (p. 24): eq. 47 from THEORY, compared with the SIMULATION
#
# Figure 1 plots the flow of NEW money into funds of a given age (2, 5, 10, 20 years)
# against the investor return r they earn in the following year.
#   y-axis: new flow = (q_t - q_{t-1}(1 + r)) / q_{t-1}   (eq. 46), and -1 if the fund closes
#   Age convention: the caption's "2 year old fund" uses the return from year 2 to 3,
#   so paper age a  <->  t = a + 1 in this code.
#
# This cell builds the curve TWO independent ways and compares them:
#   (A) THEORY: eq. 47 exactly as in the paper. Uses F_{t-1}(phi) from Propositions 3-4
#       (belief_densities, cell 2) and does the integrals numerically. No simulated data.
#   (B) SIMULATION: the same averaging, but over the simulated funds that are alive at
#       age a, instead of over the theoretical density F_{t-1}.
# If the simulation is correct, (B) should match (A), and the simulated distribution
# of beliefs should match F_{t-1}. Discrepancies that survive a large N would point to
# an error in the code or in the theory.


def eq46_new_flow(r, t, params):
    """Flow of new money for a fund that does NOT close after return r in year t (eq. 46, p. 22)."""
    omega, gamma = precisions(params)
    k = omega / (gamma + t * omega)                # weight on new information in year t
    f = params["f"]
    return (k / f - 1) * r + (k / (2 * f))**2 * r**2


def rho(r, t, params):
    omega, gamma = precisions(params)
    k = omega / (gamma + t * omega)
    f = params["f"]
    if r <= -2 * f / k:
        return np.inf
    return params["phi_bar"] / (1 + (r / (2 * f)) * k)


def eq47_theory(r, age, params, grid, step, F):
    t = age + 1                                   # the year in which return r happens
    F_prev = F[age]                               # F_{t-1}: beliefs of funds alive at the end of year `age`

    cutoff = max(params["phi_bar"], rho(r, t, params))

    integral_alive = F_prev.sum() * step                          # integral from phi_bar to infinity
    integral_stay = F_prev[grid >= cutoff].sum() * step           # integral from the cutoff to infinity

    share_stay = integral_stay / integral_alive
    share_close = 1 - share_stay

    return share_stay * eq46_new_flow(r, t, params) + share_close * (-1)


def eq47_simulated(df, age, r, params):
    """(B) SIMULATION: the same calculation as eq47_theory, but the shares are COUNTED
    from the simulated funds alive at the end of year `age`, instead of integrated from F."""
    t = age + 1
    alive = df[(df["t"] == age) & (df["closed"] == False)]
    phis = alive["phi"].to_numpy()                # one belief per surviving fund

    # For every surviving fund: if it earned investor return r next year, would it close?
    R = phis + r / (2 * params["f"] / phis)       # investor return r -> manager return R
    new_phi = update_belief(phis, R, t, params)   # updated belief after that return
    would_close = new_phi < params["phi_bar"]     # True / False for each fund

    share_close = would_close.sum() / len(phis)   # count of closing funds / all funds
    share_stay = 1 - share_close

    return share_stay * eq46_new_flow(r, t, params) + share_close * (-1)


# ---- Theory: belief densities F_t (computed once) ----
grid, step, F = belief_densities(params)

# ---- Simulation: one run per seed ----
sims = {}
for seed in FIG1_SEEDS:
    sims[seed] = run_simulation(params, N_FIG, seed)


# ---- Comparison 1: distribution of beliefs, theory vs simulation ----
# H0: the beliefs of simulated survivors at age a follow the theoretical distribution F_a
# (normalised). Test: Kolmogorov-Smirnov test, reject if p < SIG_LEVEL.
print("FIGURE 1, COMPARISON 1: beliefs of surviving funds, theory (F_t) vs simulation")
print(f"  H0: simulated beliefs ~ theoretical F_t   |   test: Kolmogorov-Smirnov   |   reject if p < {SIG_LEVEL}")
fig, axes = plt.subplots(1, len(FIG1_AGES), figsize=(4 * len(FIG1_AGES), 3.5), sharey=True)
ks_rows = []
for ax, age in zip(axes, FIG1_AGES):
    theory_cdf = np.cumsum(F[age]) / F[age].sum()             # cumulative share of survivors below each belief
    sim_phis = sims[SEED][(sims[SEED]["t"] == age) & (sims[SEED]["closed"] == False)]["phi"].to_numpy()

    def cdf_at(x, theory_cdf=theory_cdf):
        return np.interp(x, grid, theory_cdf)
    ks = stats.kstest(sim_phis, cdf_at)
    ks_rows.append([age, len(sim_phis), ks.statistic, ks.pvalue, verdict(ks.pvalue)])

    sorted_phis = np.sort(sim_phis)
    ax.plot(grid, theory_cdf, color="black", linewidth=1.5, label="Theory (Prop. 3-4)")
    ax.step(sorted_phis, np.arange(1, len(sorted_phis) + 1) / len(sorted_phis), color="tab:orange",
            linewidth=1, label=f"Simulation (seed {SEED})")
    ax.set_xlim(params["phi_bar"], 0.2)
    ax.set_title(f"Age {age}")
    ax.set_xlabel("Belief phi")
axes[0].set_ylabel("Cumulative share of survivors")
axes[0].legend(fontsize=8)
fig.suptitle(f"Beliefs of surviving funds: theory vs simulation (N = {N_FIG})", y=1.04)
fig.savefig("beliefs_theory_vs_sim.png", dpi=150, bbox_inches="tight")
ks_table = pd.DataFrame(ks_rows, columns=["age", "survivors", "KS statistic", "p-value", "Verdict"])
print(ks_table.round(4).to_string(index=False))
print("  Plot saved as beliefs_theory_vs_sim.png")
print()


# ---- Comparison 2: the Figure 1 curves, theory vs simulation ----
r_grid = np.arange(R_MIN, R_MAX + R_STEP / 2, R_STEP)

theory_curves = {}                                # theory_curves[age] = list of flows, one per r
for age in FIG1_AGES:
    theory_curves[age] = [eq47_theory(r, age, params, grid, step, F) for r in r_grid]

sim_curves = {}                                   # sim_curves[(seed, age)] = list of flows, one per r
for seed in FIG1_SEEDS:
    for age in FIG1_AGES:
        sim_curves[(seed, age)] = [eq47_simulated(sims[seed], age, r, params) for r in r_grid]

colours = {2: "tab:red", 5: "tab:green", 10: "tab:blue", 20: "tab:purple"}
fig, ax = plt.subplots(figsize=(8, 5.5))
for age in FIG1_AGES:
    ax.plot(r_grid, theory_curves[age], color=colours[age], linewidth=2.5, alpha=0.5,
            label=f"{age} years: theory (eq. 47)")
    for i, seed in enumerate(FIG1_SEEDS):
        ax.plot(r_grid, sim_curves[(seed, age)], color=colours[age], linewidth=0.8, linestyle="--",
                label=f"{age} years: simulation ({len(FIG1_SEEDS)} seeds)" if i == 0 else None)
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlim(R_MIN, R_MAX)
ax.set_ylim(-1.1, 2.3)
ax.set_xlabel("Investor return r")
ax.set_ylabel("Flow of new funds")
ax.set_title(f"Replication of Figure 1: eq. 47 from theory vs simulation (N = {N_FIG})")
ax.legend(fontsize=7, ncol=2)
fig.savefig("figure1_replication.png", dpi=150, bbox_inches="tight")
print("Plot saved as figure1_replication.png")
print()

check_r = [-0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4]
rows = []
for age in FIG1_AGES:
    for rv in check_r:
        theory_value = eq47_theory(rv, age, params, grid, step, F)
        sim_values = [eq47_simulated(sims[s], age, rv, params) for s in FIG1_SEEDS]
        rows.append([age, rv, theory_value, np.mean(sim_values),
                     max(abs(v - theory_value) for v in sim_values)])
fig1_table = pd.DataFrame(rows, columns=["age", "r", "theory (eq. 47)", "simulation (mean of seeds)",
                                         "largest |sim - theory|"])
print("FIGURE 1, COMPARISON 2: eq. 47 theory vs simulation (also compare with values read off the paper)")
print(fig1_table.round(4).to_string(index=False))
print()


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


#%% 9. Figure 1: funds that ACTUALLY earned r (dots) vs eq. 47 (theory)
#
# Eq. 47 (cell 7) answers: "if every fund alive at age a received return r, what would
# the average new flow be?" Every surviving fund counts in proportion to F_{t-1}(phi),
# whatever its chance of actually earning r.
#
# The dots answer: "among the simulated funds that ACTUALLY earned (about) r, what was
# the average new flow?" This is what empirical studies measure when they group funds
# by realised returns.
#
# These differ because funds with low beliefs have a larger active share (2f/phi), so
# their returns are more volatile (eq. 50). They are therefore over-represented among
# funds that actually earn very bad returns, and they are the ones closest to exit.
#
# The theoretical version of the dots weights F_{t-1}(phi) by the likelihood of earning
# r given phi (Bayes' rule):
#     weight(phi) = F_{t-1}(phi) * P(r | phi),   P(r | phi) = Normal density, mean 0, sd = sd_r(phi)
#     sd_r(phi)   = (2f / phi) * sqrt(1/omega + 1/(gamma + (t-1)*omega))
# (active share x sd of the surprise R - phi: luck plus remaining uncertainty about skill).
# A large N rules out small-sample noise as the explanation.


def eq47_theory_weighted(r, age, params, grid, step, F):
    omega, gamma = precisions(params)
    t = age + 1
    F_prev = F[age]

    sd_r = (2 * params["f"] / grid) * np.sqrt(1 / omega + 1 / (gamma + (t - 1) * omega))
    likelihood = stats.norm.pdf(r, 0, sd_r)       # how likely each belief level is to produce r
    weight = F_prev * likelihood                  # Bayes: prior density x likelihood

    cutoff = max(params["phi_bar"], rho(r, t, params))
    share_stay = weight[grid >= cutoff].sum() / weight.sum()
    share_close = 1 - share_stay

    return share_stay * eq46_new_flow(r, t, params) + share_close * (-1)


# ---- Simulation with a large N ----
df_big = run_simulation(params, N_BIG, SEED)

# New flow for every fund-year (eq. 46 definition): -1 if the fund closed, otherwise
# total flow minus the growth from the return itself
df_big["new_flow"] = np.where(df_big["closed"], -1.0, df_big["flow"] - df_big["r"])

# ---- Dots: group funds by the return they actually earned ----
bin_edges = np.arange(DOTS_R_MIN, DOTS_R_MAX + BIN_WIDTH / 2, BIN_WIDTH)
dot_rows = []
for age in DOTS_AGES:
    year = df_big[df_big["t"] == age + 1]          # all funds alive at age a, in their next year
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = year[(year["r"] >= lo) & (year["r"] < hi)]
        if len(in_bin) < MIN_BIN_COUNT:
            continue                               # too few funds for a reliable average
        r_mean = in_bin["r"].mean()
        dot_rows.append([age, len(in_bin), r_mean, in_bin["new_flow"].mean(),
                         eq47_theory(r_mean, age, params, grid, step, F),
                         eq47_theory_weighted(r_mean, age, params, grid, step, F)])
dots_table = pd.DataFrame(dot_rows, columns=["age", "n", "r", "dots (simulated)",
                                             "eq. 47 (theory)", "weighted (theory)"])

# ---- Plot ----
r_neg = np.arange(DOTS_R_MIN, DOTS_R_MAX + 0.0025, 0.0025)
fig, ax = plt.subplots(figsize=(8, 5.5))
for age in DOTS_AGES:
    c = colours[age]
    ax.plot(r_neg, [eq47_theory(r, age, params, grid, step, F) for r in r_neg],
            color=c, linestyle="--", linewidth=1, label=f"{age} yrs: eq. 47 (every fund gets r)")
    ax.plot(r_neg, [eq47_theory_weighted(r, age, params, grid, step, F) for r in r_neg],
            color=c, linewidth=1.5, label=f"{age} yrs: weighted by P(r | phi)")
    d = dots_table[dots_table["age"] == age]
    ax.scatter(d["r"], d["dots (simulated)"], color=c, s=20, edgecolor="black", linewidth=0.4, zorder=3)
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlim(DOTS_R_MIN, DOTS_R_MAX)
ax.set_ylim(-1.1, 0.4)
ax.set_xlabel("Investor return r")
ax.set_ylabel("Flow of new funds")
ax.set_title(f"Funds that actually earned r (dots, N = {N_BIG}) vs theory\n"
             f"dashed: eq. 47; solid: eq. 47 weighted by the likelihood of earning r")
ax.legend(fontsize=8)
fig.savefig("figure1_dots_vs_theory.png", dpi=150, bbox_inches="tight")
print("Plot saved as figure1_dots_vs_theory.png")

print(f"FIGURE 1 DOTS vs THEORY (N = {N_BIG}, seed {SEED})")
print(dots_table[dots_table["r"] < -0.05].round(3).to_string(index=False))