import numpy as np
 
# ---- Parameters (from the symbol table) ----
f = 0.015        # fee
sigma = 0.20     # sd of luck
phi0 = 0.065     # starting belief about skill
eta = 0.06       # starting uncertainty about skill
phi_bar = 0.03   # exit threshold
a = 1            # cost scale (only affects fund size)
 
omega = 1 / sigma**2   # precision of returns (25)
gamma = 1 / eta**2     # precision of starting belief (~277)
 
# ---- One fund, one year, with a return chosen by hand ----
t = 1
phi_old = phi0
R = 0.20                                           # set by hand for testing
 
r = (2 * f / phi_old) * (R - phi_old)              # investor return
phi_new = ((gamma + (t - 1) * omega) * phi_old + omega * R) / (gamma + t * omega)  # Block 5
q_new = phi_new**2 / (4 * a * f)                   # new fund size
 
print(f"Investor return r: {r:.4f}")       # expect about 0.0623
print(f"New belief phi:    {phi_new:.4f}")  # expect about 0.0761
print(f"New fund size q:   {q_new:.4f}")    # expect about 0.0966