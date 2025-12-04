# physics.py — Núcleo físico unificado SPE-215031-PA

import math
import numpy as np

# ================= Conversión Field ↔ SI =================
PSI_TO_PA = 6894.757293168
FT_TO_M   = 0.3048
CP_TO_PAS = 1e-3
ND_TO_M2  = 9.869233e-22
DAY_TO_S  = 86400.0
STB_TO_M3 = 0.158987294928

def si_mu(mu_cp):     return mu_cp * CP_TO_PAS
def si_k(k_nD):       return k_nD * ND_TO_M2
def si_ct(ct_invpsi): return ct_invpsi / PSI_TO_PA
def si_h(h_ft):       return h_ft * FT_TO_M
def si_L(L_ft):       return L_ft * FT_TO_M
def field_p(p_pa):    return p_pa / PSI_TO_PA

# ================= Estabilidad numérica =================
def tanh_stable(x):
    x = np.asarray(x, float); out = np.empty_like(x)
    big = x > 20.0; sml = x < -20.0; mid = ~(big | sml)
    out[big] = 1.0; out[sml] = -1.0; out[mid] = np.tanh(x[mid])
    return out

def coth_stable(x):
    x = np.asarray(x, float); ax = np.abs(x); out = np.empty_like(x)
    tiny = ax < 1e-8
    out[tiny] = 1.0/x[tiny] + x[tiny]/3.0
    out[~tiny] = 1.0 / tanh_stable(x[~tiny])
    return out

def exp_clamped(z, lim=700.0):
    return np.exp(np.clip(z, -lim, lim))

# ================= Inversión de Laplace =================
def stehfest_weights(N:int)->np.ndarray:
    """Pesos Stehfest (N par)."""
    assert N % 2 == 0 and N > 0
    V = np.zeros(N+1)
    for k in range(1, N+1):
        s = 0.0
        jmin = (k+1)//2
        jmax = min(k, N//2)
        for j in range(jmin, jmax+1):
            num = j**(N//2)*math.factorial(2*j)
            den = (math.factorial(N//2 - j) * math.factorial(j) *
                   math.factorial(j-1) * math.factorial(k-j) *
                   math.factorial(2*j-k))
            s += num/den
        V[k] = s*((-1)**(k+N//2))
    return V[1:]

def invert_stehfest_vec(Fvec, t:float, N:int)->np.ndarray:
    if t <= 0: return np.nan
    V = stehfest_weights(N)
    s_nodes = (np.arange(1, N+1)*math.log(2.0))/max(t, 1e-30)
    vals = [np.asarray(Fvec(s), float) for s in s_nodes]
    vals = np.stack(vals, axis=0)
    return (math.log(2.0)/t) * (V[:,None]*vals).sum(axis=0)

def invert_gaver_euler_vec(Fvec, t:float, M:int=18, P:int=8)->np.ndarray:
    ln2 = math.log(2.0); S = []
    for n in range(1, M+P+1):
        acc = None
        for k in range(1, n+1):
            s = k*ln2/max(t, 1e-30)
            term = ((-1)**k)*math.comb(n,k)*np.asarray(Fvec(s),float)
            acc = term if acc is None else acc + term
        S.append(acc)
    S = np.stack(S, axis=0)
    E = sum(math.comb(P,p)*S[(M-1)+p] for p in range(0,P+1))/(2.0**P)
    return (-ln2/max(t,1e-30)) * E

# ================= Núcleo físico (Laplace) =================
def _lambda(mu, ct, k, s):
    return math.sqrt(max(s,1e-40)*max(mu*ct,1e-40)/max(k,1e-40))

# Doble porosidad (opcional, Eq. tipo Warren-Root)
def f_dualpor(s: float, omega: float, Lambda: float) -> float:
    s = max(s,1e-40)
    om = max(min(omega,0.999999),1e-12)
    Lam = max(Lambda,1e-12)
    a = math.sqrt(Lam*(1.0-om)/(3.0*s))
    b = math.sqrt(3.0*(1.0-om)*s/Lam)
    return om + a*math.tanh(b)

def R_slab_no_flow(mu, ct, k, h, L, s, use_dp=False, omega=0.5, Lambda=1.0):
    lam = _lambda(mu, ct, k, s)
    base = (mu/(k*max(h,1e-12))) * (coth_stable(lam*max(L,1e-12))/max(lam,1e-30))
    return f_dualpor(s, omega, Lambda)*base if use_dp else base

def R_semi_inf(mu, ct, k, h, s, use_dp=False, omega=0.5, Lambda=1.0):
    lam = _lambda(mu, ct, k, s)
    base = (mu/(k*max(h,1e-12))) * (1.0/max(lam,1e-30))
    return f_dualpor(s, omega, Lambda)*base if use_dp else base

def R_self(mu, ct, k_I, k_O, h, Lx_I, Lx_O_end, s,
           use_dp_I=False, omega_I=0.5, Lambda_I=1.0,
           use_dp_O=False, omega_O=0.5, Lambda_O=1.0):
    R_I = R_slab_no_flow(mu, ct, k_I, h, Lx_I, s, use_dp_I, omega_I, Lambda_I)
    R_O = (R_slab_no_flow(mu, ct, k_O, h, Lx_O_end, s, use_dp_O, omega_O, Lambda_O)
           if Lx_O_end>0 else
           R_semi_inf(mu, ct, k_O, h, s, use_dp_O, omega_O, Lambda_O))
    return R_I + R_O

def R_cross(mu, ct, k_O, h, Dij, s):
    lam = _lambda(mu, ct, k_O, s)
    return (mu/(k_O*max(h,1e-12))) * float(exp_clamped(-lam*max(Dij,0.0))) / max(lam,1e-30)

def p_res_from_adim(pD, mu, k, h, q):
    """Convierte presión adimensional en presión física (psi)."""
    return (141.2 * mu * q) / (k * h) * pD

def q_from_adim(qD, q_ref):
    """Convierte caudal adimensional a caudal físico."""
    return qD * q_ref