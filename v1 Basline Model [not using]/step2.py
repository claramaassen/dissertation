import numpy as np
 
np.random.seed(1)  # same "random" numbers every run while testing
 
# ---- Parameters (from the symbol table) ----
f = 0.015        # fee
sigma = 0.20     # sd of luck
phi0 = 0.065     # starting belief about skill
eta = 0.06       # starting uncertainty about skill
phi_bar = 0.03   # exit threshold
a = 1            # cost scale (only affects fund size)
T = 20           # number of years
 
omega = 1 / sigma**2   # precision of returns (25)
gamma = 1 / eta**2     # precision of starting belief (~277)
 
# ---- Happens once: draw the manager's true skill, set starting belief ----
alpha = np.random.normal(phi0, eta)   # Block 2: true skill, never changes
phi = phi0                            # market's starting guess
q = phi**2 / (4 * a * f)              # starting fund size
 
# Empty lists to record what happens each year
R_list = []
r_list = []
phi_list = []
q_list = []
 
# ---- Happens every year ----
for t in range(1, T + 1):
 
    # Block 1: this year's return = skill + luck
    e = np.random.normal(0, sigma)
    R = alpha + e
 
    # Investor return, using LAST year's belief (Block 6: return comes first)
    r = (2 * f / phi) * (R - phi)
 
    # Block 5: update belief (right-hand side uses the old phi)
    phi = ((gamma + (t - 1) * omega) * phi + omega * R) / (gamma + t * omega)
 
    # Record this year
    R_list.append(R)
    r_list.append(r)
    phi_list.append(phi)
 
    # Exit rule: belief too low -> fund closes and stays closed
    if phi < phi_bar:
        q_list.append(0)
        print(f"Fund closed in year {t}")
        break
 
    # Money flows in or out: new fund size
    q = phi**2 / (4 * a * f)
    q_list.append(q)
 
# ---- Results ----
print(f"True skill (alpha): {alpha:.4f}")
print(f"Years survived: {len(r_list)}")
print()
print(" t      R        r       phi       q")
for i in range(len(r_list)):
    print(f"{i+1:2d}  {R_list[i]:7.3f}  {r_list[i]:7.3f}  {phi_list[i]:7.4f}  {q_list[i]:7.4f}")