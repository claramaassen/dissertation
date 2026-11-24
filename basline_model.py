# =====================================================================
# Berk & Green (2004): simulation, validation and comparative statics
#   Cell 1  Inputs (the only place numbers are set)
#   Cell 2  Model and theory functions
#   Cell 3  Formula check
#   Cell 4  Main simulation
#   Cell 5  Validation checks 1-5
#   Cell 6  Figure 1: eq. 47 from theory vs simulation (+ check 6)
#   Cell 7  Power check: rejection rates across many seeds
#   Cell 8  Funds that actually earned r vs eq. 47
#   Cell 9  Comparative statics: fee and return noise
#   Cell 10 Publication figures
# =====================================================================

#%% 1. Inputs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Model parameters (Table 1 of the paper)
params = {
    "f": 0.015,       # fee
    "sigma": 0.20,    # sd of luck
    "phi0": 0.065,    # prior mean of skill
    "eta": 0.06,      # prior sd of skill
    "phi_bar": 0.03,  # exit threshold (= 2f)
    "a": 1,           # cost scale (normalisation)
    "T": 20,          # years per fund
}

# Simulation settings
N, SEED = 1000, 1                          # main run
SIG_LEVEL = 0.05                           # rejection threshold for all tests
N_FIG, FIG1_SEEDS = 5000, [1, 2, 3, 4, 5]  # Figure 1 replication
FIG1_AGES = [2, 5, 10, 20]
R_MIN, R_MAX, R_STEP = -0.9, 0.9, 0.005    # returns on the Figure 1 x-axis
RUN_POWER_CHECK, POWER_SEEDS = True, 1000  # cell 7 (about a minute)
N_BIG, DOTS_AGES = 50000, [2, 5, 10]       # cell 8
DOTS_R_MIN, DOTS_R_MAX, BIN_WIDTH, MIN_BIN_COUNT = -0.42, 0.10, 0.04, 200
FEE_GRID = [0.005, 0.01, 0.015, 0.02, 0.025]   # cell 9
SIGMA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
CS_N, CS_SEEDS = 5000, [1, 2, 3]
FIG_DIR, FIG_DPI, FIG_WIDTH = "figures", 300, 6.3   # cell 10


#%% 2. Model and theory functions

def precisions(params):
    """omega = 1/sigma^2 (returns), gamma = 1/eta^2 (prior)."""
    return 1 / params["sigma"]**2, 1 / params["eta"]**2


def investor_return(R, phi_old, params):
    """Eq. (10): active share 2f/phi times the surprise R - phi (last year's belief)."""
    return (2 * params["f"] / phi_old) * (R - phi_old)


def update_belief(phi_old, R, t, params):
    """Eq. (3): weighted average of the old belief and the new return."""
    omega, gamma = precisions(params)
    return ((gamma + (t - 1) * omega) * phi_old + omega * R) / (gamma + t * omega)


def fund_size(phi, params):
    """Eq. (7): q = phi^2 / (4af)."""
    return phi**2 / (4 * params["a"] * params["f"])


def simulate_one_fund(params, fund_id):
    """One fund's life: one row per year, until it closes or T years pass."""
    alpha = np.random.normal(params["phi0"], params["eta"])   # true skill (fixed)
    phi = params["phi0"]                                      # starting belief
    q_old = fund_size(phi, params)                            # starting size
    rows = []
    for t in range(1, params["T"] + 1):
        R = alpha + np.random.normal(0, params["sigma"])      # 1. manager return = skill + luck
        r = investor_return(R, phi, params)                   # 2. investor return (old belief)
        phi = update_belief(phi, R, t, params)                # 3. update belief
        closed = phi < params["phi_bar"]                      # 4. exit rule
        q = 0 if closed else fund_size(phi, params)           # 5. new size and flow
        rows.append({"fund": fund_id, "t": t, "alpha": alpha, "R": R, "r": r,
                     "phi": phi, "q": q, "flow": q / q_old - 1, "closed": closed})
        if closed:
            break                                             # 6. closed funds stay closed
        q_old = q
    return rows


def run_simulation(params, N, seed):
    """N funds, seed set once; one row per fund-year."""
    np.random.seed(seed)
    rows = []
    for i in range(N):
        rows.extend(simulate_one_fund(params, i))
    return pd.DataFrame(rows)


def add_standardised_surprise(df, params):
    """z = (R_t - phi_{t-1}) / its sd under the model (eq. 11); N(0,1) for every fund-year."""
    omega, gamma = precisions(params)
    df = df.copy()
    df["phi_prev"] = df.groupby("fund")["phi"].shift(1).fillna(params["phi0"])
    df["z"] = (df["R"] - df["phi_prev"]) / np.sqrt(1 / omega + 1 / (gamma + (df["t"] - 1) * omega))
    return df


def belief_densities(params, n_grid=2000):
    """F_t(phi), Prop. 3-4: density of beliefs among funds alive at the end of year t.
    Each year beliefs move by Normal(0, v_t), v_t = 1/(gamma+(t-1)omega) - 1/(gamma+t omega);
    beliefs below phi_bar are dropped (the grid starts at phi_bar)."""
    omega, gamma = precisions(params)
    grid = np.linspace(params["phi_bar"], params["phi0"] + 8 * params["eta"], n_grid)
    step = grid[1] - grid[0]
    F = {}
    for t in range(1, params["T"] + 1):
        sd = np.sqrt(1 / (gamma + (t - 1) * omega) - 1 / (gamma + t * omega))
        if t == 1:
            F[t] = stats.norm.pdf(grid, params["phi0"], sd)
        else:
            move = stats.norm.pdf(grid[:, None], grid[None, :], sd)   # chance of old phi -> new phi
            F[t] = move @ F[t - 1] * step
    return grid, step, F


def survival_prop4(params):
    """Prop. 4: P_t = area under F_t."""
    grid, step, F = belief_densities(params)
    return pd.Series([F[t].sum() * step for t in range(1, params["T"] + 1)],
                     index=range(1, params["T"] + 1))


def eq46_new_flow(r, t, params):
    """Paper's eq. 46: new-money flow for a fund that survives return r in year t."""
    omega, gamma = precisions(params)
    k = omega / (gamma + t * omega)
    return (k / params["f"] - 1) * r + (k / (2 * params["f"]))**2 * r**2


def rho(r, t, params):
    """Paper's p. 22: a fund closes after return r if its belief was below rho(r) (eq. 12)."""
    omega, gamma = precisions(params)
    k = omega / (gamma + t * omega)
    if r <= -2 * params["f"] / k:
        return np.inf                                          # every fund closes
    return params["phi_bar"] / (1 + (r / (2 * params["f"])) * k)


def eq47_theory(r, age, params, grid, step, F, weighted=False):
    """Paper's eq. 47 from F_{t-1}: share staying x eq. 46 + share closing x (-1).
    weighted=True also weights F_{t-1} by P(r | phi): funds that actually earned r."""
    omega, gamma = precisions(params)
    t = age + 1                                                # return earned the year after age a
    weight = F[age].copy()
    if weighted:
        sd_r = (2 * params["f"] / grid) * np.sqrt(1 / omega + 1 / (gamma + (t - 1) * omega))
        weight = weight * stats.norm.pdf(r, 0, sd_r)
    cutoff = max(params["phi_bar"], rho(r, t, params))
    share_stay = weight[grid >= cutoff].sum() / weight.sum()
    return share_stay * eq46_new_flow(r, t, params) + (1 - share_stay) * (-1)


def eq47_simulated(df, age, r, params):
    """Same as eq47_theory, but shares are counted from simulated funds alive at age a."""
    t = age + 1
    phis = df[(df["t"] == age) & (df["closed"] == False)]["phi"].to_numpy()
    R = phis + r / (2 * params["f"] / phis)                    # investor return r -> manager return R
    would_close = update_belief(phis, R, t, params) < params["phi_bar"]
    share_stay = 1 - would_close.sum() / len(phis)
    return share_stay * eq46_new_flow(r, t, params) + (1 - share_stay) * (-1)


def run_checks(df, params, P_theory):
    """Checks 1-5 on one simulated table. Returns {check: (estimate, p-value, threshold)}."""
    T, n = params["T"], df["fund"].nunique()
    survivors = df[(df["t"] == T) & (df["closed"] == False)]
    alpha = df.groupby("fund")["alpha"].first()
    out = {}

    t1 = stats.ttest_1samp(df["r"], 0)                         # 1. mean r = 0
    out["1. Mean investor return = 0"] = (f"mean r = {df['r'].mean():.4f} (t = {t1.statistic:.3f})", t1.pvalue, SIG_LEVEL)

    n_open = df[df["closed"] == False].groupby("t")["fund"].count().reindex(range(1, T + 1), fill_value=0)
    p2 = min(stats.binomtest(int(n_open[t]), n, P_theory[t]).pvalue for t in range(1, T + 1))   # 2. survival
    out["2. Survival = Prop. 4"] = (f"year {T}: {n_open[T] / n:.3f} vs {P_theory[T]:.3f}", p2, SIG_LEVEL / T)

    t3 = stats.ttest_rel(survivors["alpha"], survivors["phi"])  # 3. survivors: alpha = phi
    out["3. Survivors: mean alpha = mean phi"] = (
        f"{survivors['alpha'].mean():.4f} vs {survivors['phi'].mean():.4f} (t = {t3.statistic:.3f})", t3.pvalue, SIG_LEVEL)

    d = add_standardised_surprise(df, params)                  # 4. no predictability (z, and old test on r)
    d["z_lag"] = d.groupby("fund")["z"].shift(1)
    d["r_lag"] = d.groupby("fund")["r"].shift(1)
    d = d.dropna(subset=["z_lag"])
    t4 = stats.pearsonr(d["z"], d["z_lag"])
    out["4. No predictability (z)"] = (f"corr = {t4.statistic:.4f}", t4.pvalue, SIG_LEVEL)
    t4r = stats.pearsonr(d["r"], d["r_lag"])
    out["4. No predictability (r, old)"] = (f"corr = {t4r.statistic:.4f}", t4r.pvalue, SIG_LEVEL)

    share = 1 - stats.norm.cdf((params["f"] - params["phi0"]) / params["eta"])   # 5. share beating fee
    t5 = stats.binomtest(int((alpha > params["f"]).sum()), n, share)
    out["5. Share beating the fee"] = (f"{(alpha > params['f']).mean():.3f} vs {share:.3f}", t5.pvalue, SIG_LEVEL)
    return out


def decision(p, threshold):
    return "Rejected" if p < threshold else "Not rejected"


#%% 3. Formula check: one year with R = 0.20, against hand calculations
r_test = investor_return(0.20, params["phi0"], params)
phi_test = update_belief(params["phi0"], 0.20, 1, params)
q_test = fund_size(phi_test, params)
print("FORMULA CHECK")
for name, got, expected in [("r", r_test, 0.0623), ("phi", phi_test, 0.0761), ("q", q_test, 0.0966)]:
    print(f"  {name:4s} {got:.4f} vs {expected:.4f} -> {'PASS' if abs(got - expected) < 1e-4 else 'FAIL'}")
print()


#%% 4. Main simulation
df = run_simulation(params, N, SEED)
T = params["T"]
P_theory = survival_prop4(params)
print(f"Fund-years: {len(df)}; funds open after {T} years: {((df['t'] == T) & ~df['closed']).sum()} of {N}\n")
print("Fund 0 (Table 2):")
print(df[df["fund"] == 0][["t", "alpha", "R", "r", "phi", "q", "flow", "closed"]].round(4).to_string(index=False))
print()


#%% 5. Validation checks 1-5 (Tables 4 and 6)
checks = run_checks(df, params, P_theory)
summary = pd.DataFrame([(k, est, round(p, 3), decision(p, th)) for k, (est, p, th) in checks.items()
                        if "old" not in k], columns=["Check", "Estimate", "p-value", "Decision"])
print(f"VALIDATION CHECKS (seed {SEED}, N = {N}; check 2 uses threshold {SIG_LEVEL}/{T})")
print(summary.to_string(index=False))
print()

n_open = df[df["closed"] == False].groupby("t")["fund"].count().reindex(range(1, T + 1), fill_value=0)
survival = n_open / N
survival_table = pd.DataFrame({"simulated": survival, "Prop. 4": P_theory, "difference": survival - P_theory,
                               "p-value": [stats.binomtest(int(n_open[t]), N, P_theory[t]).pvalue for t in range(1, T + 1)]})
print("SURVIVAL BY YEAR (Table 6)")
print(survival_table.round(4).to_string())
print()


#%% 6. Figure 1: eq. 47 from theory vs simulation, and check 6 (Tables 7 and 8)
grid, step, F = belief_densities(params)
sims = {seed: run_simulation(params, N_FIG, seed) for seed in FIG1_SEEDS}

# Check 6: beliefs of simulated survivors vs theoretical F_a (Kolmogorov-Smirnov)
ks_rows = []
for age in FIG1_AGES:
    theory_cdf = np.cumsum(F[age]) / F[age].sum()
    sim_phis = sims[SEED][(sims[SEED]["t"] == age) & (sims[SEED]["closed"] == False)]["phi"].to_numpy()
    ks = stats.kstest(sim_phis, lambda x, c=theory_cdf: np.interp(x, grid, c))
    ks_rows.append([age, len(sim_phis), ks.statistic, ks.pvalue, decision(ks.pvalue, SIG_LEVEL)])
print("CHECK 6: beliefs of survivors, simulation vs theory (Table 7)")
print(pd.DataFrame(ks_rows, columns=["age", "survivors", "KS", "p-value", "Decision"]).round(4).to_string(index=False))
print()

# Curves: theory (from F) vs simulation (counted from simulated survivors)
r_grid = np.arange(R_MIN, R_MAX + R_STEP / 2, R_STEP)
theory_curves = {age: [eq47_theory(r, age, params, grid, step, F) for r in r_grid] for age in FIG1_AGES}
rows = []
for age in FIG1_AGES:
    for r in [-0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4]:
        th = eq47_theory(r, age, params, grid, step, F)
        sim = [eq47_simulated(sims[s], age, r, params) for s in FIG1_SEEDS]
        rows.append([age, r, th, np.mean(sim), max(abs(v - th) for v in sim)])
print("FIGURE 1: eq. 47 theory vs simulation (Table 8)")
print(pd.DataFrame(rows, columns=["age", "r", "theory", "simulation", "max |sim - theory|"]).round(4).to_string(index=False))
print()


#%% 7. Power check: rejection rates across many seeds (Table 5)
if RUN_POWER_CHECK:
    rejections = {}
    for s in range(1, POWER_SEEDS + 1):
        for k, (est, p, th) in run_checks(run_simulation(params, N, s), params, P_theory).items():
            rejections[k] = rejections.get(k, 0) + (p < th)
    power = pd.DataFrame([(k, n_rej, n_rej / POWER_SEEDS, stats.binomtest(int(n_rej), POWER_SEEDS, SIG_LEVEL).pvalue)
                          for k, n_rej in rejections.items()],
                         columns=["Check", "rejections", "rate", "p (rate = 5%)"])
    print(f"POWER CHECK ({POWER_SEEDS} seeds, N = {N}) (Table 5)")
    print(power.round(3).to_string(index=False))
    print()


#%% 8. Funds that actually earned r (dots) vs eq. 47 (Table 9)
df_big = run_simulation(params, N_BIG, SEED)
df_big["new_flow"] = np.where(df_big["closed"], -1.0, df_big["flow"] - df_big["r"])   # eq. 46 definition

edges = np.arange(DOTS_R_MIN, DOTS_R_MAX + BIN_WIDTH / 2, BIN_WIDTH)
dot_rows = []
for age in DOTS_AGES:
    year = df_big[df_big["t"] == age + 1]                   # funds alive at age a, in their next year
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = year[(year["r"] >= lo) & (year["r"] < hi)]
        if len(b) >= MIN_BIN_COUNT:
            r_mean = b["r"].mean()
            dot_rows.append([age, len(b), r_mean, b["new_flow"].mean(),
                             eq47_theory(r_mean, age, params, grid, step, F),
                             eq47_theory(r_mean, age, params, grid, step, F, weighted=True)])
dots_table = pd.DataFrame(dot_rows, columns=["age", "n", "r", "simulated", "eq. 47", "weighted"])
print(f"REALISED RETURNS vs eq. 47 (N = {N_BIG}) (Table 9)")
print(dots_table[dots_table["r"] < -0.05].round(3).to_string(index=False))
print()


#%% 9. Comparative statics: change one parameter at a time
# The fee keeps the paper's assumption F = f^2/a, so the exit threshold moves with it (phi_bar = 2f).
def outcomes(p):
    """Key outputs for one parameter set: theory + mean over CS_SEEDS simulations."""
    omega, gamma = precisions(p)
    P = survival_prop4(p)
    sim = [run_simulation(p, CS_N, s) for s in CS_SEEDS]
    surv = lambda d, t: ((d["t"] == t) & ~d["closed"]).sum() / CS_N
    size20 = lambda d: d[(d["t"] == p["T"]) & ~d["closed"]]["q"].mean() * p["a"]   # mean size of survivors
    return {
        "survival 5y (sim)": np.mean([surv(d, 5) for d in sim]),
        "survival 5y (Prop. 4)": P[5],
        "survival 20y (sim)": np.mean([surv(d, p["T"]) for d in sim]),
        "survival 20y (Prop. 4)": P[p["T"]],
        "share alpha > f": 1 - stats.norm.cdf((p["f"] - p["phi0"]) / p["eta"]),
        "flow sensitivity age 2": omega / (gamma + 3 * omega) / p["f"] - 1,     # slope of eq. 46 at r = 0
        "flow sensitivity age 20": omega / (gamma + 21 * omega) / p["f"] - 1,
        "sd of r": np.mean([d["r"].std() for d in sim]),
        "mean r": np.mean([d["r"].mean() for d in sim]),                         # should stay ~0
        "survivor size": np.mean([size20(d) for d in sim]),
    }

cs = {}
for f in FEE_GRID:
    cs[("fee", f)] = outcomes({**params, "f": f, "phi_bar": 2 * f})
for s in SIGMA_GRID:
    cs[("sigma", s)] = outcomes({**params, "sigma": s})
cs_table = pd.DataFrame(cs).T
for name in ["fee", "sigma"]:                                  # fund size relative to the baseline row
    base = cs_table.loc[(name, params["f"] if name == "fee" else params["sigma"]), "survivor size"]
    cs_table.loc[name, "survivor size"] = (cs_table.loc[name, "survivor size"] / base).values
cs_table = cs_table.rename(columns={"survivor size": "survivor size (rel. to baseline)"})
print(f"COMPARATIVE STATICS (N = {CS_N}, seeds {CS_SEEDS}) (Table 10)")
print(cs_table.round(3).to_string())
print()


#%% 10. Publication figures (no titles: captions go in the document)
os.makedirs(FIG_DIR, exist_ok=True)
INK, GRID_C, SIM_C, BLUE = "#222222", "#d9d9d9", "#eb6834", "#2a78d6"
AGE_STYLE = {2: ("#3987e5", "-"), 5: ("#256abf", "--"), 10: ("#184f95", "-."), 20: ("#0d366b", ":")}
STYLE = {"font.family": "serif",
         "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
         "mathtext.fontset": "stix", "font.size": 10, "legend.fontsize": 9,
         "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
         "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.6,
         "legend.frameon": False, "savefig.bbox": "tight"}

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  saved {FIG_DIR}/{name}.png/.pdf")

print("FIGURES")
with plt.rc_context(STYLE):
    # Survival: simulation vs Proposition 4
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.2))
    years = np.arange(1, T + 1)
    ax.bar(years, survival.values, width=0.7, color=BLUE, alpha=0.85, label="Simulation")
    ax.plot(years, P_theory.values, color=INK, marker="o", markersize=4, linewidth=1.2, label="Proposition 4")
    ax.set(xticks=years, xlim=(0.4, T + 0.6), ylim=(0, 1), xlabel="Years in business", ylabel="Share of funds surviving")
    ax.legend(loc="upper right")
    save(fig, "fig_survival")

    # Beliefs of survivors: theoretical vs simulated CDF
    fig, axes = plt.subplots(1, len(FIG1_AGES), figsize=(FIG_WIDTH, 2.4), sharey=True)
    for ax, age in zip(axes, FIG1_AGES):
        phis = np.sort(sims[SEED][(sims[SEED]["t"] == age) & ~sims[SEED]["closed"]]["phi"].to_numpy())
        ax.plot(grid * 100, np.cumsum(F[age]) / F[age].sum(), color=INK, label="Theory")
        ax.step(phis * 100, np.arange(1, len(phis) + 1) / len(phis), color=SIM_C, linewidth=1, linestyle="--", label="Simulation")
        ax.set(xlim=(params["phi_bar"] * 100, 20), xticks=[5, 10, 15, 20], xlabel=r"Belief $\phi$ (%)")
        ax.text(0.95, 0.06, f"Age {age}", transform=ax.transAxes, ha="right")
    axes[0].set(ylabel="Cumulative share", ylim=(0, 1.02))
    fig.tight_layout(w_pad=0.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    save(fig, "fig_beliefs")

    # Figure 1 replication: theory lines, simulation markers
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.2))
    marker_r = np.round(np.arange(-0.8, 0.81, 0.1), 2)
    for age in FIG1_AGES:
        colour, style = AGE_STYLE[age]
        ax.plot(r_grid, theory_curves[age], color=colour, linestyle=style, label=f"Age {age} (theory)")
        sim_mean = [np.mean([eq47_simulated(sims[s], age, r, params) for s in FIG1_SEEDS]) for r in marker_r]
        ax.plot(marker_r, sim_mean, linestyle="none", marker="o", markersize=4.5,
                markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.1)
    ax.plot([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor=INK,
            label=f"Simulation (mean of {len(FIG1_SEEDS)} seeds)")
    ax.axhline(0, color=GRID_C, linewidth=0.8, zorder=0)
    ax.axvline(0, color=GRID_C, linewidth=0.8, zorder=0)
    ax.set(xlim=(R_MIN, R_MAX), ylim=(-1.1, 2.3), xlabel=r"Investor return $r$", ylabel="Flow of new funds")
    ax.legend(loc="upper left")
    save(fig, "fig_flow_performance")

    # Realised returns: simulation vs eq. 47 and weighted eq. 47
    r_neg = np.arange(DOTS_R_MIN, DOTS_R_MAX + 0.0025, 0.0025)
    fig, axes = plt.subplots(1, len(DOTS_AGES), figsize=(FIG_WIDTH, 2.6), sharey=True)
    for ax, age in zip(axes, DOTS_AGES):
        ax.plot(r_neg, [eq47_theory(r, age, params, grid, step, F) for r in r_neg],
                color=INK, linestyle="--", linewidth=1.2, label="Eq. 47")
        ax.plot(r_neg, [eq47_theory(r, age, params, grid, step, F, weighted=True) for r in r_neg],
                color=BLUE, label=r"Eq. 47 weighted by $P(r\mid\phi)$")
        d = dots_table[dots_table["age"] == age]
        ax.plot(d["r"], d["simulated"], linestyle="none", marker="o", markersize=5,
                markerfacecolor=SIM_C, markeredgecolor="white", markeredgewidth=0.8, label="Simulation")
        ax.axhline(0, color=GRID_C, linewidth=0.8, zorder=0)
        ax.set(xlim=(DOTS_R_MIN, DOTS_R_MAX), ylim=(-1.05, 0.25), xlabel=r"Investor return $r$")
        ax.text(0.05, 0.93, f"Age {age}", transform=ax.transAxes, va="top")
    axes[0].set_ylabel("Flow of new funds")
    fig.tight_layout(w_pad=0.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    save(fig, "fig_realised_returns")

    # Comparative statics: one row per parameter, three outcomes
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, 4.6))
    for row, (name, values, xlabel) in enumerate([("fee", FEE_GRID, r"Fee $f$ (%)"),
                                                   ("sigma", SIGMA_GRID, r"Return noise $\sigma$ (%)")]):
        x = np.array(values) * 100
        t_ = cs_table.loc[name]
        base_x = (params["f"] if name == "fee" else params["sigma"]) * 100
        panels = [
            ("Share surviving", [("survival 5y (Prop. 4)", "5 years", "-", "o"), ("survival 20y (Prop. 4)", "20 years", "--", "s")]),
            ("Flow sensitivity", [("flow sensitivity age 2", "Age 2", "-", "o"), ("flow sensitivity age 20", "Age 20", "--", "s")]),
            (r"s.d. of investor return $r$", [("sd of r", None, "-", "o")]),
        ]
        for col, (ylabel, series) in enumerate(panels):
            ax = axes[row, col]
            for key, label, ls, mk in series:
                ax.plot(x, t_[key].values, color=BLUE if ls == "-" else INK, linestyle=ls, marker=mk,
                        markersize=4, label=label)
            if col == 0:   # simulation markers on top of the Prop. 4 lines
                for key in ["survival 5y (sim)", "survival 20y (sim)"]:
                    ax.plot(x, t_[key].values, linestyle="none", marker="x", color=SIM_C, markersize=5,
                            label="Simulation" if key.startswith("survival 5y") else None)
            ax.axvline(base_x, color=GRID_C, linewidth=0.8, zorder=0)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            if row == 0 and label is not None:
                ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_comparative_statics")